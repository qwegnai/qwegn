// DriverErrors.swift — driver error cases and scratch-eviction policy
// (split from QWEngineDriver.swift).
import Foundation
import Darwin
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWMLP
import QWHead
import QWMTP
import QWQuant
import QWState

/// The driver's error is the engine's error: one vocabulary for every
/// family, so a consumer never names a concrete engine's error type.
public typealias QWEngineDriverError = QWEngineError

/// Eviction policy for the 3.07 GB prefill activation scratchpad. Once
/// prefill completes the engine decodes with `sc` (~15 MB); on a
/// RAM-constrained machine (<= 36 GB) the scratchpad is evicted.
public enum QWPrefillScratchEvictionPolicy: String, Sendable, CaseIterable {
    case auto
    case always
    case never

    /// The knob's spelling, resolved. An unknown mode fails loud.
    public static func parse(_ string: String?) throws -> QWPrefillScratchEvictionPolicy {
        guard let s = string?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(), !s.isEmpty else {
            return .auto
        }
        switch s {
        case "auto": return .auto
        case "always", "1", "true", "yes": return .always
        case "never", "0", "false", "no": return .never
        default:
            throw QWEngineError.state("invalid prefill scratch eviction mode '\(s)' (expected auto, always, or never)")
        }
    }

    public func shouldEvict(
        totalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory,
        wiredEngineBytes: Int = 0
    ) -> Bool {
        switch self {
        case .always:
            return true
        case .never:
            return false
        case .auto:
            let ramGB = Double(totalMemoryBytes) / 1_000_000_000.0
            if ramGB <= 36.5 { return true }
            return wiredEngineBytes > 0 && UInt64(wiredEngineBytes) * 4 >= totalMemoryBytes * 3
        }
    }
}

/// One admitted sequence: cache handle + GDN state slot + cursors.
///
