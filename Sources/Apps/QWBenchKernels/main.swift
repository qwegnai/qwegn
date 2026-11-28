// main.swift — the kernel shootout.
//
// The families' decode GEMVs against the incumbent's, and MLX's tile GEMM
// as the reference point. Every candidate is checked against the host at the
// shape it is timed at, and the report is the winner map the engine reads.
// Follows the `Scripts/m1` shape: no ArgumentParser, explicit flags, a
// VERDICT line, exit(code).

import Foundation
import QWCore
import QWHAL

let args = CommandLine.arguments
func optVal(_ flag: String) -> String? {
    guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
    return args[i + 1]
}
func optFlag(_ flag: String) -> Bool { args.contains(flag) }
func die(_ message: String, _ code: Int32 = 2) -> Never {
    FileHandle.standardError.write(Data("qwg-bench-kernels: \(message)\n".utf8))
    exit(code)
}

if optFlag("--help") || optFlag("-h") {
    print("""
    usage: qwg-bench-kernels [--out DIR] [--samples N] [--cols N] [--only-shvq] [--only-ink2-fp] [--only-core-scaling] [--only-apple9-prefill]
      --out DIR            where to write bench_kernels_report.json (default: .)
      --samples N          timed runs per candidate, median wins (default: 5)
      --repeats N          dispatches per timed run, to amortise the buffer (default: 1)
      --cols N             the activation width every candidate reads (default: 4096)
      --only-shvq          run only the cold-streaming SHVQ vs AQ live kernel suites
      --only-ink2-fp       run only Inkling-2 BF16/NVFP4/MXFP8/Sink-RelBias/Head suites
      --only-core-scaling  run active-TG core-scaling and dispatch/barrier/rotation tax suites
      --only-apple9-prefill run prefill GEMM, MoE prefill, and Ink2 FP prefill suites for Apple9
    """)
    exit(0)
}

let outDir = optVal("--out") ?? "."
let samples = max(1, Int(optVal("--samples") ?? "5") ?? 5)
let cols = max(BenchRunner.verifyCols,
               Int(optVal("--cols") ?? String(BenchSuites.defaultCols)) ?? BenchSuites.defaultCols)
let repeats = max(1, Int(optVal("--repeats") ?? "1") ?? 1)
let onlySHVQ = optFlag("--only-shvq")
let onlyInk2FP = optFlag("--only-ink2-fp")
let onlyCoreScaling = optFlag("--only-core-scaling")
let onlyApple9Prefill = optFlag("--only-apple9-prefill")

/// The candidates the bench races, in the order it reports them.
var candidates: [BenchCandidate] = []
for (bits, group) in BenchSuites.pairs {
    candidates.append(contentsOf: BenchSuites.gemvCandidates(bits: bits, group: group))
}

do {
    // The bench races the 27B's kernels, so the HAL carries the 27B's
    // function constants — named here, never defaulted.
    let hal = try QWHal(geometry: .qwen38_27b)

    if onlyCoreScaling {
        let rounds = max(6, samples)
        let reps = max(8, repeats)
        print("suite core_scaling (alternating min-of-2, \(rounds) rounds x \(reps) reps/CB)")
        let report = try CoreScalingSuites.run(hal: hal, samples: rounds, repeats: reps)
        for r in report.dispatchTax {
            print(String(format: "  tax %-26@ total=%.4f ms  us/dispatch=%.3f  delta=%+.3f us",
                         r.mode as NSString, r.totalMs, r.usPerDispatch, r.deltaUsVsSerial))
        }
        let dir = URL(fileURLWithPath: outDir, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let md = CoreScalingSuites.renderMarkdown(report, rounds: rounds, reps: reps)
        let mdURL = dir.appendingPathComponent("bench_core_scaling_report.md")
        try md.write(to: mdURL, atomically: true, encoding: .utf8)
        print("VERDICT: wrote \(mdURL.path) (\(report.scaling.count) scaling points, \(report.dispatchTax.count) tax modes)")
        exit(0)
    }

    if onlyInk2FP {
        let rounds = max(6, samples)
        let reps = max(8, repeats)
        print("suite ink2_fp (alternating min-of-2, \(rounds) rounds x \(reps) reps/CB)")
        let inkRows = try Ink2FPSuites.run(hal: hal, samples: rounds, repeats: reps)
        for r in inkRows {
            let candDesc = r.candMs.map { ms in
                String(format: "vs %@ %.4f ms (%.2fx)", r.candKernel, ms, r.baseMs / max(ms, 1e-9))
            } ?? "(\(r.candKernel): \(r.note))"
            print("  \(r.suite) [\(r.shape)]: \(r.baseKernel) "
                  + String(format: "%.4f", r.baseMs) + " ms \(candDesc)")
        }
        let dir = URL(fileURLWithPath: outDir, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let md = Ink2FPSuites.renderMarkdown(inkRows, rounds: rounds, reps: reps)
        let mdURL = dir.appendingPathComponent("bench_ink2_fp_report.md")
        try md.write(to: mdURL, atomically: true, encoding: .utf8)
        print("VERDICT: wrote \(mdURL.path) (\(inkRows.count) Inkling-2 FP benchmarks)")
        exit(0)
    }

    if onlyApple9Prefill {
        print("suite apple9_prefill (\(samples) samples x \(repeats) reps)")
        let prefillRows = try PrefillSuites.run(hal: hal, cols: cols, samples: samples, repeats: repeats)
        for race in prefillRows {
            let ratio = race.family.milliseconds / race.incumbent.milliseconds
            print("  prefill_gemm \(race.bits)b_g\(race.group): family "
                  + String(format: "%.4f", race.family.milliseconds) + " ms (\(race.family.name)), incumbent "
                  + String(format: "%.4f", race.incumbent.milliseconds) + " ms (\(race.incumbent.name)), ratio "
                  + String(format: "%.4f", ratio))
        }
        let moeRes = try MoESuites.run(hal: hal, suite: .prefill, cols: cols, samples: samples, repeats: repeats)
        for c in moeRes.candidates {
            print("  moe_prefill \(c.name): \(String(format: "%.4f", c.median)) ms")
        }
        let shvqRows = try SHVQSuites.run(hal: hal, samples: samples, repeats: repeats)
        for r in shvqRows where r.suite.hasPrefix("prefill") {
            let ratio = r.shvqMs / max(r.aqMs, 1e-9)
            print("  \(r.suite) [\(r.shape)] \(r.bits)b_g\(r.group): shvq "
                  + String(format: "%.4f", r.shvqMs) + " ms vs aq "
                  + String(format: "%.4f", r.aqMs) + " ms (" + String(format: "%.3fx", ratio) + ")")
        }
        print("VERDICT: apple9_prefill completed (\(prefillRows.count) prefill GEMM races, \(moeRes.candidates.count) MoE candidates, \(shvqRows.filter { $0.suite.hasPrefix("prefill") }.count) SHVQ prefill rows)")
        exit(0)
    }

    if onlySHVQ {
        print("suite shvq_vs_aq (cold 256 MiB flush, \(samples) alternating samples)")
        let shvqRows = try SHVQSuites.run(hal: hal, samples: samples, repeats: repeats)
        for r in shvqRows {
            let ratio = r.shvqMs / max(r.aqMs, 1e-9)
            print("  \(r.suite) [\(r.shape)] \(r.bits)b_g\(r.group): shvq "
                  + String(format: "%.4f", r.shvqMs) + " ms vs aq "
                  + String(format: "%.4f", r.aqMs) + " ms ("
                  + String(format: "%.3fx", ratio) + ")")
        }
        let dir = URL(fileURLWithPath: outDir, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let emptyMap = WinnerMap.make(from: [])
        let md = BenchReport.markdown(map: emptyMap, points: [], suites: [], moeSuites: [],
                                      prefill: [], shvq: shvqRows, cols: cols, samples: samples)
        let mdURL = dir.appendingPathComponent("bench_shvq_report.md")
        try md.write(to: mdURL, atomically: true, encoding: .utf8)
        print("VERDICT: wrote \(mdURL.path) (\(shvqRows.count) SHVQ cold-streaming pairs)")
        exit(0)
    }

    // A quick check first, so a broken candidate fails in a second rather
    // than after the whole sweep.
    print("verifying \(candidates.count) candidates at \(BenchRunner.verifyRows) rows")
    for candidate in candidates {
        let work = try BenchRunner.makeWork(hal: hal, candidate: candidate,
                                             rows: BenchRunner.verifyRows,
                                             cols: BenchRunner.verifyCols)
        _ = try BenchRunner.run(hal: hal, candidate: candidate, work: work,
                                 rows: BenchRunner.verifyRows, cols: BenchRunner.verifyCols)
        try BenchRunner.verify(candidate: candidate, work: work, rows: BenchRunner.verifyRows)
        print("  ok \(candidate.name)")
    }

    var points: [BenchPoint] = []
    var banks: [String: BenchWork] = [:]
    for bucket in BenchSuites.nBuckets {
        // The candidates that read the same format are measured ALTERNATELY,
        // so a clock ramp between them lands on both rather than on one.
        // MLX's candidate joins the family's 4-bit G64 in that alternation.
        var groups: [(bits: Int, group: Int)] = []
        for candidate in candidates {
            let key = (candidate.bits, candidate.group)
            if !groups.contains(where: { $0 == key }) { groups.append(key) }
        }
        for key in groups {
            let racers = candidates.filter {
                $0.bits == key.bits && $0.group == key.group
            }
            var times: [String: [Double]] = [:]
            for candidate in racers {
                // The candidates that read the same bank share it, so two
                // kernels race on the very same bytes.
                let bankKey = "\(bucket)|\(candidate.bankKey)"
                let work: BenchWork
                if let cached = banks[bankKey] {
                    work = cached
                } else {
                    work = try BenchRunner.makeWork(hal: hal, candidate: candidate,
                                                     rows: bucket, cols: cols)
                    banks[bankKey] = work
                }
                // The check runs at the shape that is about to be timed.
                _ = try BenchRunner.run(hal: hal, candidate: candidate, work: work,
                                         rows: bucket, cols: cols)
                try BenchRunner.verify(candidate: candidate, work: work, rows: bucket)
            }
            for _ in 0..<samples {
                for candidate in racers {
                    let bankKey = "\(bucket)|\(candidate.bankKey)"
                    guard let work = banks[bankKey] else { continue }
                    let t = try BenchRunner.run(hal: hal, candidate: candidate, work: work,
                                                rows: bucket, cols: cols, repeats: repeats)
                    times[candidate.name, default: []].append(t)
                }
            }
            for candidate in racers {
                let sorted = (times[candidate.name] ?? []).sorted()
                guard !sorted.isEmpty else { continue }
                points.append(BenchPoint(nBucket: bucket, op: candidate.op, bits: candidate.bits,
                                         group: candidate.group, candidate: candidate.name,
                                         gpuMilliseconds: sorted[sorted.count / 2]))
            }
        }
    }

    // The per-operator suites: each races two ways of doing one operator, on
    // the operator's own shape, and files its result under its own operation
    // — but only where the measurement could separate the two candidates.
    var noiseByOp: [String: Double] = [:]
    var suites: [OperatorSuites.OperatorResult] = []
    for suite in OperatorSuites.all {
        print("suite \(suite.rawValue)")
        let result = try OperatorSuites.run(hal: hal, suite: suite, cols: cols,
                                             samples: samples, repeats: repeats)
        for p in result.points {
            print("  \(p.candidate): \(String(format: "%.4f", p.gpuMilliseconds)) ms")
        }
        print("  separation \(String(format: "%.2f%%", result.separation * 100)), "
              + "noise floor \(String(format: "%.2f%%", result.noise * 100))")
        if !result.separable {
            print("  not separable — no entry is published, so the engine keeps "
                  + "the path it already runs")
        } else {
            print("  published to the map")
        }
        noiseByOp[suite.rawValue] = result.noise
        suites.append(result)
        points.append(contentsOf: result.points)
    }

    // The routed-MoE suites: the gate/up projection in the decode shape and
    // in the prefill shape, raced the same way.
    var moeSuites: [OperatorSuites.OperatorResult] = []
    for suite in MoESuite.allCases {
        print("suite \(suite.rawValue)")
        let result = try MoESuites.run(hal: hal, suite: suite, cols: cols,
                                       samples: samples, repeats: repeats)
        for c in result.candidates {
            let mark = c.publishable ? " " : " (bench-only)"
            print("  \(c.name)\(mark): \(String(format: "%.4f", c.median)) ms")
        }
        print("  separation \(String(format: "%.2f%%", result.separation * 100)), "
              + "noise floor \(String(format: "%.2f%%", result.noise * 100))")
        if !result.separable {
            print("  not separable — no entry is published, so the engine keeps "
                  + "the path it already runs")
        } else {
            print("  published to the map")
        }
        noiseByOp[suite.rawValue] = result.noise
        moeSuites.append(result)
        points.append(contentsOf: result.points)
    }

    // The prefill race: the incumbent's tiled GEMM against the family's, over
    // the same logical weights, for every pair the family supports.
    print("suite prefill_gemm")
    let prefill = try PrefillSuites.run(hal: hal, cols: cols, samples: samples, repeats: repeats)
    for race in prefill {
        let ratio = race.family.milliseconds / race.incumbent.milliseconds
        print("  \(race.bits)-bit G\(race.group): family "
              + String(format: "%.4f", race.family.milliseconds) + " ms, incumbent "
              + String(format: "%.4f", race.incumbent.milliseconds) + " ms, ratio "
              + String(format: "%.4f", ratio))
    }

    print("suite shvq_vs_aq")
    let shvqRows = try SHVQSuites.run(hal: hal, samples: samples, repeats: repeats)
    for r in shvqRows {
        let ratio = r.shvqMs / max(r.aqMs, 1e-9)
        print("  \(r.suite) [\(r.shape)] \(r.bits)b_g\(r.group): shvq "
              + String(format: "%.4f", r.shvqMs) + " ms vs aq "
              + String(format: "%.4f", r.aqMs) + " ms (" + String(format: "%.3fx", ratio) + ")")
    }

    // The bench and the engine share the vocabulary of operations, so the
    // bench checks itself against it before it writes anything.
    for point in points where !QWKernelDispatchTable.knownOps.contains(point.op) {
        die("the bench emitted the unknown operation \(point.op)", 1)
    }

    let map = WinnerMap.make(from: points)
    let data = try map.encoded()    // The report must survive its own reader: a map this build cannot read
    // back is not a report.
    _ = try QWKernelDispatchTable.decode(data)
    let dir = URL(fileURLWithPath: outDir, isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let url = dir.appendingPathComponent("bench_kernels_report.json")
    try data.write(to: url)
    let markdown = BenchReport.markdown(map: map, points: points, suites: suites,
                                         moeSuites: moeSuites, prefill: prefill,
                                         shvq: shvqRows, cols: cols, samples: samples)
    let mdURL = dir.appendingPathComponent("bench_kernels_report.md")
    try markdown.write(to: mdURL, atomically: true, encoding: .utf8)

    print("| rows | op | bits | group | winner | gpu ms | runner-up | margin | noise |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for e in map.entries {
        let ms = points.first { $0.nBucket == e.nBucket && $0.op == e.op && $0.bits == e.bits
            && $0.group == e.group && $0.candidate == e.winner }?.gpuMilliseconds ?? 0
        let margin = e.margin.map { String(format: "%.1f%%", $0 * 100) } ?? "-"
        let noise = noiseByOp[e.op].map { String(format: "%.1f%%", $0 * 100) } ?? "-"
        print("| \(e.nBucket) | \(e.op) | \(e.bits) | \(e.group) | \(e.winner) | "
              + String(format: "%.4f", ms) + " | \(e.runnerUp ?? "-") | " + margin + " | "
              + noise + " |")
    }
    print("VERDICT: wrote \(url.path) and \(mdURL.path) (\(map.entries.count) entries)")
    exit(0)
} catch {
    die("failed: \(error)", 1)
}
