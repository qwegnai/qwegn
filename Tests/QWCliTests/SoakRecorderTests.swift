// SoakRecorderTests.swift — The soak artifact has one producer, and the
// gate can still fail. The rules: the harness measures windows the gate
// accepts and every number in them is measured; a harness artifact skips
// the engine's decode laws; a seeded regression fails the gate; the
// recorder writes what the reader reads.
//
// The allocation law is checked on the producer's artifact, in its process.

import Foundation
import Testing
@testable import QWCli

/// A throwaway directory for the artifacts a test writes.
private func scratchDir() throws -> URL {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("soakrec-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

/// Something wrong with the harness run itself.
private struct HarnessError: Error, CustomStringConvertible {
    let what: String
    init(_ what: String) { self.what = what }
    var description: String { what }
}

/// The anchor for the bundle this file lives in: the harness binary is
/// built next to it.
private final class HarnessAnchor: NSObject {}

/// The windows the harness measured, in a process that does only the
/// workload — the soak's allocation number is this process's live heap
/// blocks, and only a process that does nothing else can attribute it.
private func harnessWindows(steps: Int = 1_000, window: Int = 250) throws -> [QWCli.SoakSample] {
    let built = Bundle(for: HarnessAnchor.self).bundleURL.deletingLastPathComponent()
    let binary = built.appendingPathComponent("m1")
    guard FileManager.default.isExecutableFile(atPath: binary.path) else {
        throw HarnessError("the m1 harness was not built next to the test bundle (\(binary.path))")
    }
    let dir = try scratchDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let export = dir.appendingPathComponent("m1_export.jsonl")
    let process = Process()
    process.executableURL = binary
    process.arguments = ["--harness", "--steps", "\(steps)", "--window", "\(window)",
                         "--rows", "128", "--export", export.path]
    process.standardOutput = FileHandle.nullDevice
    process.standardError = FileHandle.nullDevice
    try process.run()
    process.waitUntilExit()
    // The harness verifies the artifact against the gate's coverage law
    // (10 000 steps), which this shorter run does not meet, so its exit
    // status is not this test's verdict. The artifact is: the export must
    // be there, and the gate below must accept it.
    guard FileManager.default.fileExists(atPath: export.path) else {
        throw HarnessError("the m1 harness wrote no export to \(export.path)")
    }
    return try QWCli.SoakSample.load(export)
}

@Test("the harness measures windows the gate accepts")
func localSoakPassesTheGate() throws {
    let samples = try harnessWindows()
    #expect(samples.count == 4)
    #expect(samples.last?.steps == 1_000)
    let verdict = QWCli.verifySoak(samples: samples, minSteps: 1_000)
    #expect(verdict.allPass, Comment(rawValue: verdict.lines.joined(separator: "\n")))
    // every window says what measured it
    #expect(samples.allSatisfy { $0.source == QWCli.SoakSample.harnessSource })
    #expect(samples.allSatisfy { $0.kind == QWCli.SoakSample.windowKind })
    // the steady state retained nothing
    #expect(samples.allSatisfy { $0.steadyAllocs == 0 })
    // no command buffer ran: that is absent, not zero
    #expect(samples.allSatisfy { $0.cbErrors == nil })
    #expect(samples.allSatisfy { $0.kvFreePages == nil })
    // the steps ascend and the window measured real time
    #expect(zip(samples, samples.dropFirst()).allSatisfy { $0.steps < $1.steps })
    #expect(samples.allSatisfy { $0.windowTps > 0 })
}

@Test("a harness artifact skips the engine's decode laws")
func harnessSkipsTheEngineLaws() throws {
    let samples = try harnessWindows()
    let verdict = QWCli.verifySoak(samples: samples, minSteps: 1_000)
    #expect(verdict.allPass, Comment(rawValue: verdict.lines.joined(separator: "\n")))
    #expect(verdict.lines.contains { $0.contains("SKIP") && $0.contains("pooled decode floor") })
    #expect(verdict.lines.contains { $0.contains("SKIP") && $0.contains("jitter") })
    // the throughput is the harness's step rate, not a decode rate
    #expect(verdict.metrics["m1_step_rate_per_second"] != nil)
    #expect(verdict.metrics["m1_decode_tps"] == nil)
    // the KV pages were not measured, so they are not published
    #expect(verdict.metrics["m1_kv_free_pages"] == nil)
    #expect(verdict.metrics["m1_itl_p95_ms"] != nil)
    #expect(verdict.metrics["m1_steady_allocs"] == 0)
}

@Test("a seeded regression fails the gate")
func seededRegressionFails() throws {
    var samples = try harnessWindows()
    #expect(QWCli.verifySoak(samples: samples, minSteps: 1_000).allPass)
    // the seeded regression: the steady state retained heap blocks
    samples[1].steadyAllocs = 3
    let verdict = QWCli.verifySoak(samples: samples, minSteps: 1_000)
    #expect(!verdict.allPass)
    #expect(verdict.lines.contains { $0.contains("FAIL") && $0.contains("allocs") })
    // a second class: a window that does not advance the step count
    var scrambled = samples
    scrambled[1].steps = scrambled[0].steps
    let tampered = QWCli.verifySoak(samples: scrambled, minSteps: 1_000)
    #expect(!tampered.allPass)
    #expect(tampered.lines.contains { $0.contains("FAIL") && $0.contains("ascending") })
}

@Test("the recorder writes what the reader reads")
func recorderRoundTrip() throws {
    let dir = try scratchDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let url = QWCli.SoakRecorder.exportURL(in: dir)
    #expect(url.lastPathComponent.hasPrefix("m1_export_"))
    var recorder = QWCli.SoakRecorder(url: url)
    let samples = QWCli.LocalSoak.run(.init(steps: 500, window: 250, rows: 64))
    try recorder.record(samples)
    #expect(recorder.written == samples.count)
    let loaded = try QWCli.SoakSample.load(url)
    #expect(loaded.count == samples.count)
    #expect(loaded.map(\.steps) == samples.map(\.steps))
    #expect(loaded.map(\.steadyAllocs) == samples.map(\.steadyAllocs))
    #expect(loaded.map(\.source) == samples.map(\.source))
    #expect(loaded.map(\.windowTps) == samples.map(\.windowTps))
}

@Test("the reader skips the export's non-window lines")
func readerSkipsNonWindowLines() throws {
    let dir = try scratchDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let url = dir.appendingPathComponent("m1_export_1.jsonl")
    let metric = #"{"kind":"metric","name":"latency.ttft_ms","value":12.5}"#
    let request = #"{"kind":"request","id":"req-1","promptTokens":10}"#
    let window = #"{"kind":"window","steps":1000,"windowTps":30.0,"windowP50Ms":33.0,"windowP95Ms":34.0,"steadyAllocs":0,"rssKb":1000,"fds":4}"#
    try Data((metric + "\n" + request + "\n" + window + "\n").utf8).write(to: url)
    let loaded = try QWCli.SoakSample.load(url)
    #expect(loaded.count == 1)
    #expect(loaded.first?.steps == 1_000)
    #expect(loaded.first?.cbErrors == nil)
}

@Test("the ratchet compares a zero baseline without inventing a percentage")
func ratchetZeroBaseline() throws {
    let baseline = QWCli.Baselines(date: "2026-10-25T00:00:00Z", machine: "m5-max", metrics: [
        "m1_steady_allocs": QWCli.Metric(0, .lower, "blocks"),
        "m1_step_rate_per_second": QWCli.Metric(500, .higher, "steps/s"),
    ])
    // the same value is not a regression
    let same = QWCli.ratchetCheck(current: ["m1_steady_allocs": 0, "m1_step_rate_per_second": 500],
                                  baseline: baseline)
    #expect(same.allSatisfy { $0.ok })
    // a zero baseline that grows is a regression the ratio cannot express
    let grown = QWCli.ratchetCheck(current: ["m1_steady_allocs": 4, "m1_step_rate_per_second": 500],
                                   baseline: baseline)
    #expect(grown.first { $0.metric == "m1_steady_allocs" }?.ok == false)
    // a metric this run did not measure is skipped, never called a pass
    let missing = QWCli.ratchetCheck(current: [:], baseline: baseline)
    #expect(missing.allSatisfy { $0.ok && $0.current.isNaN })
}

@Test("an artifact with no window is not a passing artifact")
func emptyArtifactFails() throws {
    let dir = try scratchDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let url = dir.appendingPathComponent("m1_soak_1.jsonl")
    try Data().write(to: url)
    #expect(!QWCli.verifySoak(samples: try QWCli.SoakSample.load(url)).allPass)
}
