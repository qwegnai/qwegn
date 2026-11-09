// PackWiring.swift — the single pin gate: headroom probe + RLIMIT-aware
// mlock decision. Formula preserves the shipped FN-AD-39 gate exactly
// (free+inactive+speculative); the older "exclude inactive" pin law is
// registered for adjudication in LOAD_PATH_DRY_PLAN, not silently applied.
// Call sites own their log lines.
import Foundation
import Darwin
import QWQuant

public enum PackWiring {
    public enum Skip: Sendable { case disabled, envSkip, headroom }

    public struct Outcome: Sendable {
        public let wiredBytes: Int
        public let packBytes: Int
        public let headroomBytes: Int
        public let skip: Skip?
        public var wired: Bool { skip == nil }
    }

    public static func headroomBytes() -> Int {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.stride / MemoryLayout<integer_t>.stride)
        let kr = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        let page = UInt64(sysconf(_SC_PAGESIZE))
        let free = UInt64(stats.free_count) * page
        let inactive = UInt64(stats.inactive_count) * page
        let speculative = UInt64(stats.speculative_count) * page
        return Int(free + inactive + speculative)
    }

    /// Gate only: who MAY pin, per policy + headroom. Performs no wiring —
    /// callers choose sync vs background execution (launch-to-available).
    public static func decide(_ pack: QWQuant.Pack, policy: LoadPolicy) -> Outcome {
        let packBytes = pack.packedByteCount
        if policy.wireSkipped {
            return Outcome(wiredBytes: 0, packBytes: packBytes, headroomBytes: 0, skip: .envSkip)
        }
        let headroom = headroomBytes()
        guard policy.wireForced || policy.wireByDefault else {
            return Outcome(wiredBytes: 0, packBytes: packBytes, headroomBytes: headroom, skip: .disabled)
        }
        guard headroom >= packBytes + policy.wireHeadroomMarginBytes || policy.wireForced else {
            return Outcome(wiredBytes: 0, packBytes: packBytes, headroomBytes: headroom, skip: .headroom)
        }
        return Outcome(wiredBytes: packBytes, packBytes: packBytes, headroomBytes: headroom, skip: nil)
    }

    /// Gate + synchronous pin.
    public static func wire(_ pack: QWQuant.Pack, policy: LoadPolicy) -> Outcome {
        let outcome = decide(pack, policy: policy)
        guard outcome.wired else { return outcome }
        let wired = pack.wire()
        return Outcome(wiredBytes: wired, packBytes: outcome.packBytes,
                       headroomBytes: outcome.headroomBytes, skip: nil)
    }
}
