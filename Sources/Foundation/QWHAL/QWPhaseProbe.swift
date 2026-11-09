// QWPhaseProbe.swift — Environment-gated wall accumulators for Flash-Next
// cadence attribution (FN-AD-14). Enabled via QW_FN_PHASES=1; zero cost when
// disabled (single boolean check). Tracks per-label CPU/GPU spans plus
// command-buffer submission and wait counts across decode steps.
// Thread-safe via NSLock.

import Foundation

public final class QWPhaseProbe: @unchecked Sendable {
    public static let shared = QWPhaseProbe()
    private let enabled = ProcessInfo.processInfo.environment["QW_PHASES"] == "1"
        || ProcessInfo.processInfo.environment["QW_FN_PHASES"] == "1"
    private var totals: [String: (ms: Double, n: Int)] = [:]
    private let lock = NSLock()
    private var slowShown = 0

    public final class Span {
        let label: String
        var t0: UInt64
        init(_ label: String) { self.label = label; t0 = mach_absolute_time() }
        deinit { QWPhaseProbe.shared.add(label, t0: t0) }
    }

    public func span(_ label: String) -> Span? { enabled ? Span(label) : nil }
    public var isEnabled: Bool { enabled }

    /// Accumulate a span that started at a saved mach_absolute_time tick.
    public func addFrom(_ label: String, t0: UInt64) {
        guard enabled else { return }
        add(label, t0: t0)
    }

    private(set) var submissions = 0
    private(set) var waits = 0

    public func noteSubmission() {
        guard enabled else { return }
        lock.lock(); submissions += 1; lock.unlock()
    }

    public func noteWait() {
        guard enabled else { return }
        lock.lock(); waits += 1; lock.unlock()
    }

    /// Convert mach_absolute_time ticks to nanoseconds using hardware timebase.
    fileprivate static let tickNs: Double = {
        var info = mach_timebase_info_data_t(numer: 1, denom: 1)
        mach_timebase_info(&info)
        return Double(info.numer) / Double(info.denom)
    }()

    fileprivate func add(_ label: String, t0: UInt64) {
        let ns = Double(mach_absolute_time() &- t0) * Self.tickNs * 1e-9
        lock.lock()
        var e = totals[label] ?? (0, 0)
        e.ms += ns * 1000
        e.n += 1
        totals[label] = e
        if ns * 1000 > 150, slowShown < 60 {
            slowShown += 1
            print(String(format: "SLOW SPAN %@: %.0f ms (#%d)", label, ns * 1000, e.n))
        }
        lock.unlock()
    }

    /// Book a GPU execution interval from a completed command buffer.
    public func addGpu(_ label: String, ms: Double) {
        guard enabled else { return }
        lock.lock()
        var e = totals[label] ?? (0, 0)
        e.ms += ms
        e.n += 1
        totals[label] = e
        lock.unlock()
    }

    public func report() -> String {
        lock.lock(); defer { lock.unlock() }
        var s = "phase-probe: submissions=\(submissions) waits=\(waits)"
        let siteLabels = totals.keys.filter { $0.hasPrefix("cpu:flushWait:") || $0.hasPrefix("gpu:") }
        var drop: Set<String> = []
        if siteLabels.count > 64,
           ProcessInfo.processInfo.environment["QW_PHASES_ALL"] != "1",
           ProcessInfo.processInfo.environment["QW_FN_PHASES_ALL"] != "1" {
            drop = Set(siteLabels.sorted { totals[$0]!.ms > totals[$1]!.ms }.dropFirst(64))
        }
        for (k, v) in totals.sorted(by: { $0.value.ms > $1.value.ms }) {
            if drop.contains(k) { continue }
            s += String(format: " | %@ %.0f ms / %d (%.2f avg)", k, v.ms, v.n, v.ms / Double(v.n))
        }
        return s
    }
}
