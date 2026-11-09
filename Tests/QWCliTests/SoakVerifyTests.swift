import Testing
import Foundation
@testable import QWCli

// verifySoak re-verifies the m1 artifact (the CI merge-gate math — pure
// file logic, so the CI gate itself has a gate). The conservative-jitter
// honesty is bound here: pooling window p95s can only UNDERSTATE the
// per-step tail, which the doc admits and the test pins.

private func win(_ steps: Int, _ tps: Double, p50: Double? = nil, p95: Double? = nil,
                 allocs: Int = 0, rss: Int64 = 40_000_000, fds: Int = 60, cb: Int = 0) -> QWCli.SoakSample {
    let p = p50 ?? (1000.0 / tps)
    return QWCli.SoakSample(steps: steps, windowTps: tps, windowP50Ms: p,
                            windowP95Ms: p95 ?? (p * 1.05), steadyAllocs: allocs,
                            rssKb: rss, fds: fds, cbErrors: cb)
}

@Test("verifySoak: healthy 10k log passes every line; pooled t/s is exact")
func soakVerifyPass() {
    var samples: [QWCli.SoakSample] = []
    for i in 1...10 { samples.append(win(i * 1000, 32.0)) }
    let v = QWCli.verifySoak(samples: samples)
    #expect(v.allPass)
    #expect(abs(v.metrics["m1_decode_tps"]! - 32.0) < 0.01)
    #expect(v.lines.filter { $0.contains("FAIL") }.isEmpty)
}

@Test("verifySoak: every failure class trips its own line")
func soakVerifyFailures() {
    // short coverage
    let short = QWCli.verifySoak(samples: [win(1000, 32)])
    #expect(!short.allPass && short.lines.contains { $0.contains("coverage") })
    // nonzero allocs anywhere poisons (counter is cumulative)
    let allocs = QWCli.verifySoak(samples: [win(5000, 32), win(10000, 32, allocs: 1)])
    #expect(!allocs.allPass && allocs.lines.contains { $0.contains("allocs") })
    // cb errors at end
    let cb = QWCli.verifySoak(samples: [win(5000, 32), win(10000, 32, cb: 3)])
    #expect(!cb.allPass)
    // slow pool: half the windows at 20 t/s → pooled ≈ 26.7 < 30
    let slow = QWCli.verifySoak(samples: [win(5000, 40), win(10000, 20)])
    #expect(!slow.allPass && slow.lines.contains { $0.contains("pooled") })
    // jitter: one window whose p95 is 2× weighted p50 → bound ≈ 1.6 > 1.10
    let jittery = QWCli.verifySoak(samples: [win(5000, 32), win(10000, 32, p95: 100)])
    #expect(!jittery.allPass && jittery.lines.contains { $0.contains("jitter") })
    // RSS creep 100 MiB across windows; fd creep 5
    let leak = QWCli.verifySoak(samples: [win(5000, 32, rss: 40_000_000),
                                          win(10000, 32, rss: 40_102_400, fds: 65)])
    #expect(!leak.allPass && leak.lines.contains { $0.contains("RSS") } && leak.lines.contains { $0.contains("fd") })
    // non-ascending log = tampering class
    let scrambled = QWCli.verifySoak(samples: [win(10000, 32), win(5000, 32)])
    #expect(!scrambled.allPass && scrambled.lines.contains { $0.contains("ascending") })
    // empty = fail loud (the silent-stub class this whole file retires)
    #expect(!QWCli.verifySoak(samples: []).allPass)
}

@Test("verifySoak jitter bound is CONSERVATIVE: a healthy per-window p95 passes even at +9% per window")
func jitterBoundDirection() {
    // every window p95 = 1.09× its p50 → bound = 1.09 ≤ 1.10 → PASS
    let ok = QWCli.verifySoak(samples: [win(5000, 32, p50: 31.25, p95: 31.25 * 1.09),
                                        win(10000, 32, p50: 31.25, p95: 31.25 * 1.09)])
    #expect(ok.allPass)
    // a uniform +20% window tail → the bound catches it (the bound is
    // max(window p95) / weighted p50 — pooling never hides a tail that shows
    // at a window edge; it can only understate a tail inside one window)
    let bad = QWCli.verifySoak(samples: [win(5000, 32, p50: 31.25, p95: 31.25 * 1.20),
                                         win(10000, 32, p50: 31.25, p95: 31.25 * 1.20)])
    #expect(!bad.allPass)
}

@Test("newestSoakLog: prefix+extension filter, newest wins; missing dir = nil")
func newestSoakSelection() throws {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("soakdir-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let old = dir.appendingPathComponent("m1_soak_100.jsonl")
    let new = dir.appendingPathComponent("m1_soak_200.jsonl")
    try Data("{}".utf8).write(to: old)
    try Data(" {}".utf8).write(to: new)     // same size class, later write
    try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSinceNow: -3600)],
                                          ofItemAtPath: old.path)
    // a decoy with the wrong prefix must never win
    let decoy = dir.appendingPathComponent("other_soak.jsonl")
    try Data("{}".utf8).write(to: decoy)
    let found = QWCli.newestSoakLog(in: dir)
    #expect(found?.lastPathComponent == "m1_soak_200.jsonl")
    #expect(QWCli.newestSoakLog(in: dir.appendingPathComponent("nope")) == nil)
}
