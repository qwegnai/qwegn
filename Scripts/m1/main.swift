// m1 — the M1 gate harness (spec-21 "Gate" section; spec 40 §4 T6).
// Two modes, one schema: `m1 <pack>` is the operator soak (the real
// engine, real steps); `m1 --harness` is the CPU-only reference soak (no
// model, no GPU). Both write the measured windows to a telemetry export
// (JSONL); the export is what Scripts/gen_soak_sample.py turns into the
// m1_soak_*.jsonl artifact that `qwg bench m1 --check` verifies.

import Foundation
import QWCli
import QWEngine
import QWQuant
import QWToken

let argv = CommandLine.arguments

/// True when the flag is on the command line.
func optFlag(_ name: String) -> Bool { argv.contains(name) }

/// The value that follows a flag, when there is one.
func optVal(_ name: String) -> String? {
    guard let i = argv.firstIndex(of: name), i + 1 < argv.count else { return nil }
    return argv[i + 1]
}

/// Stops the harness with a message. The gate fails loud.
func die(_ message: String, _ code: Int32 = 2) -> Never {
    FileHandle.standardError.write(Data("[m1] \(message)\n".utf8))
    exit(code)
}

/// The nearest-rank percentile of a sorted sample.
func percentile(_ sorted: [Double], _ q: Double) -> Double {
    guard !sorted.isEmpty else { return 0 }
    let idx = min(sorted.count - 1, max(0, Int((q * Double(sorted.count)).rounded(.up)) - 1))
    return sorted[idx]
}

/// The GiB the LLM co-tenants hold in the unified memory pool.
func coRunnerMemGiB() -> Double {
    let watched: Set<String> = ["MLXCore", "mlx-serve", "python3.13"]
    var pids = [Int32](repeating: 0, count: 8192)
    let bytes = pids.withUnsafeMutableBytes { proc_listallpids($0.baseAddress, Int32($0.count)) }
    var totalBytes: UInt64 = 0
    for i in 0..<max(0, Int(bytes) / MemoryLayout<Int32>.size) {
        let pid = pids[i]
        if pid <= 0 { continue }
        var pathBuf = [CChar](repeating: 0, count: 4096)
        let n = proc_pidpath(pid, &pathBuf, UInt32(pathBuf.count))
        if n <= 0 { continue }
        let path = pathBuf.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
        guard let base = path.split(separator: "/").last,
              watched.contains(String(base)) else { continue }
        let ps = Process()
        ps.executableURL = URL(fileURLWithPath: "/bin/ps")
        ps.arguments = ["-o", "rss=", "-p", String(pid)]
        let pipe = Pipe(); ps.standardOutput = pipe
        try? ps.run(); ps.waitUntilExit()
        let rss = (String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        totalBytes += UInt64((Double(rss) ?? 0) * 1024)
    }
    return Double(totalBytes) / 1_073_741_824
}

/// The real corpus text the prefill runs on: random ids would skip the
/// embed rows and fake the measurement.
let corpusLine = "The capital of France is Paris. The largest ocean on Earth is the Pacific Ocean. "
    + "Water boils at 100 degrees Celsius at sea level. The planet known as the red planet is Mars. "
    + "The chemical symbol for gold is Au. "

let usage = """
usage:
  m1 <pack> [--assets DIR] [--steps N] [--window W] [--max-tokens N] [--export FILE] [--force]
  m1 --harness [--steps N] [--window W] [--rows N] [--export FILE]

  <pack>          a model directory, a .qwp pack, or an alias.
  --harness       the CPU-only reference soak: no model, no GPU, no arena.
  --steps N       the steps the soak performs (default 10000).
  --window W      the steps in one window (default 1000).
  --rows N        the matrix side of one reference step (default 2048).
  --export FILE   the telemetry export the artifact is built from.
  --force         run even when a co-tenant holds the memory pool (not keepable).
"""

guard argv.count >= 2 else {
    print(usage)
    die("a pack or --harness is required", 2)
}

let steps = Int(optVal("--steps") ?? "") ?? 10_000
let window = Int(optVal("--window") ?? "") ?? 1_000
// The default export lives under .build/ (regenerable, git-ignored): the
// export is an input to the artifact, not a file the repo carries.
let repoRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
let buildDir = repoRoot.appendingPathComponent(".build", isDirectory: true)
try? FileManager.default.createDirectory(at: buildDir, withIntermediateDirectories: true)
let exportPath = optVal("--export")
    ?? QWCli.SoakRecorder.exportURL(in: buildDir).path
let exportURL = URL(fileURLWithPath: exportPath)
try? FileManager.default.removeItem(at: exportURL)
var recorder = QWCli.SoakRecorder(url: exportURL)

// ── mode 1: the CPU-only reference soak (no model, no GPU) ─────────────────
if optFlag("--harness") {
    if optFlag("--ratchet-update") {
        die("REFUSING baseline write: the CPU-only harness is not a certified source "
            + "for the engine baseline; run the operator soak against a pack", 2)
    }
    let rows = Int(optVal("--rows") ?? "") ?? 2_048
    let started = Date()
    let samples = QWCli.LocalSoak.run(.init(steps: steps, window: window, rows: rows))
    try recorder.record(samples)
    let wall = Date().timeIntervalSince(started)
    print("[m1] local reference soak: \(steps) steps in \(String(format: "%.1f", wall)) s "
          + "(\(samples.count) windows) — no model, no GPU, no arena")
    print("[m1] export: \(exportURL.path)")
    let verdict = QWCli.verifySoak(samples: samples)
    for line in verdict.lines { print(line) }
    let pass = verdict.allPass
    print(pass ? "[m1] VERDICT: PASS" : "[m1] VERDICT: FAIL")
    exit(pass ? 0 : 1)
}

// ── mode 2: the operator soak (the real engine) ───────────────────────────
let packPath = argv[1]
let modelLocation: QWModelLocation
do {
    modelLocation = try QWModelLocation(url: URL(fileURLWithPath: packPath),
        assetsOverride: optVal("--assets").map { URL(fileURLWithPath: $0) })
} catch {
    die("ABORT: \(error)", 2)
}
let engineMaxTokens = Int(optVal("--max-tokens") ?? "") ?? 8_192
let force = optFlag("--force")
if !force {
    let coMemGiB = coRunnerMemGiB()
    print(String(format: "[m1] co-runner memory census: %.1f GiB resident (MLXCore, mlx-serve, python3.13)",
                 coMemGiB))
    if coMemGiB >= 4.0 {
        die("ABORT: a co-tenant holds \(String(format: "%.1f", coMemGiB)) GiB of the unified memory pool; "
            + "park the LLM server or pass --force for a non-keepable smoke", 4)
    }
}

let prefillMode: GDNPrefillMode = optVal("--mode") == "naive" ? .naive : .chunked
let tLoad = Date()
let engine: QWEngineDriver
do {
    engine = try QWEngineDriver(packPath: packPath, maxTokens: engineMaxTokens,
                                ropeRows: engineMaxTokens, gdnPrefillMode: prefillMode,
                                weightArenaSizing: .planned, mtpDrafts: 0)
} catch {
    die("ABORT: the engine did not load: \(error)", 2)
}
print(String(format: "[m1] engine ready in %.1f s (arena %.2f GiB, weights %.2f/%.2f GiB)",
             Date().timeIntervalSince(tLoad), Double(engine.arenaBytes) / 1_073_741_824,
             Double(engine.weightUsedBytes) / 1_073_741_824,
             Double(engine.weightReservedBytes) / 1_073_741_824))

let tokenizer: QWTokenizer
do {
    let json = try Data(contentsOf: modelLocation.assets.appendingPathComponent("tokenizer.json"))
    tokenizer = try QWTokenizer(json: json)
} catch {
    die("ABORT: the tokenizer did not load: \(error)", 2)
}

/// A real-text prompt of the requested length.
func corpusPrompt(_ targetTokens: Int) throws -> [Int] {
    var ids: [Int] = []
    let paragraph = try tokenizer.encode(corpusLine)
    while ids.count < targetTokens { ids += paragraph }
    return Array(ids.prefix(targetTokens))
}

// Warm-up: one full session cycle, discarded. The first touch of the
// weights pages them in; that is not a sample.
do {
    let w = try engine.makeSession(promptTokens: window + 64)
    let seed = try corpusPrompt(64)
    _ = try engine.prefill(w, tokens: seed)
    var token = seed[seed.count - 1]
    for _ in 0..<40 { token = try engine.decodeStep(w, token: token) }
    engine.finishSession(w)
    print("[m1] warmup: 1 session cycle (64-token prefill + 40 steps) discarded")
} catch {
    die("ABORT: the warm-up failed: \(error)", 2)
}

var samples: [QWCli.SoakSample] = []
var done = 0
let soakStart = Date()
while done < steps {
    let n = min(window, steps - done)
    var times: [Double] = []
    times.reserveCapacity(n)
    do {
        let session = try engine.makeSession(promptTokens: window + 64)
        let seed = try corpusPrompt(64)
        _ = try engine.prefill(session, tokens: seed)
        var token = seed[seed.count - 1]
        for _ in 0..<n {
            let t0 = QWCli.ProcessTelemetry.clockGetNanos()
            token = try engine.decodeStep(session, token: token)
            times.append(Double(QWCli.ProcessTelemetry.clockGetNanos() - t0) / 1_000_000)
        }
        engine.finishSession(session)
    } catch {
        die("ABORT: the soak failed at \(done) steps: \(error)", 2)
    }
    done += n
    let ordered = times.sorted()
    let wallSeconds = ordered.reduce(0, +) / 1000
    let sample = QWCli.SoakSample(
        steps: done,
        windowTps: wallSeconds > 0 ? Double(n) / wallSeconds : 0,
        windowP50Ms: percentile(ordered, 0.50),
        windowP95Ms: percentile(ordered, 0.95),
        steadyAllocs: engine.steadyAllocs,
        rssKb: QWCli.ProcessTelemetry.rssKb(),
        fds: QWCli.ProcessTelemetry.fdCount(),
        cbErrors: engine.commandBufferErrors,
        kvFreePages: engine.freeKVPages,
        source: QWCli.SoakSample.engineSource,
        sampledAt: Date(),
        label: "engine decode soak — \(modelLocation.weights.lastPathComponent)",
        kind: QWCli.SoakSample.windowKind)
    samples.append(sample)
    do {
        try recorder.record(sample)
    } catch {
        die("ABORT: the export could not be written: \(error)", 2)
    }
    print(String(format: "[m1] soak %5d/%d  %.1f t/s  p50 %.1f ms  p95 %.1f ms  allocs %d  RSS %lld MiB  fd %d",
                 done, steps, sample.windowTps, sample.windowP50Ms, sample.windowP95Ms,
                 sample.steadyAllocs, sample.rssKb / 1024, sample.fds))
}
let wall = Date().timeIntervalSince(soakStart)
print("[m1] export: \(exportURL.path)")
print(String(format: "[m1] soak wall %.0f s · %d steps", wall, done))
let verdict = QWCli.verifySoak(samples: samples)
for line in verdict.lines { print(line) }
let pass = verdict.allPass
if optFlag("--ratchet-update") {
    // A baseline is written from a certified run only, and only from an
    // engine soak: the CPU-only harness does not measure the decode.
    let baselineURL = URL(fileURLWithPath: optVal("--baselines")
        ?? (repoRoot.path + "/Scripts/bench/baselines.json"))
    guard pass, !force else {
        die("REFUSING baseline write: the run did not certify (verdict=\(pass ? "PASS" : "FAIL"), "
            + "force=\(force)); a failing run cannot re-baseline", 1)
    }
    var metrics = (try? QWCli.Baselines.load(baselineURL))?.metrics ?? [:]
    for (key, value) in verdict.metrics {
        metrics[key] = QWCli.Metric(value, QWCli.ratchetDirections[key] ?? .lower, "")
    }
    let baselines = QWCli.Baselines(date: ISO8601DateFormatter().string(from: Date()),
                                    machine: "M5 Max", metrics: metrics)
    do {
        try baselines.save(baselineURL)
    } catch {
        die("the baseline could not be written: \(error)", 2)
    }
    print("[m1] baseline written: \(baselineURL.path) (\(metrics.count) metrics)")
}
print(pass ? "[m1] VERDICT: PASS" : "[m1] VERDICT: FAIL")
exit(pass ? 0 : 1)
