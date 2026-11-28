// QWModelCapabilities.swift — Geometry-derived capability resolution.
import Foundation
import QWCore

/// Resolves runtime serving capabilities from model geometry, weight format, and assets.
public enum QWModelCapabilities {
    /// Derives every knob capability supported by `descriptor` without family-based stripping.
    public static func resolve(
        descriptor: QWModelDescriptor,
        weightFormat: QWWeightFormat? = nil,
        hasVisionWeights: Bool = false
    ) -> Set<QWKnobCapability> {
        let _ = weightFormat
        guard descriptor.role != .speculativeDrafter else { return [] }
        if case .recurrentLoop(_, let ink) = descriptor.executionTopology, ink != nil {
            var caps: Set<QWKnobCapability> = [.prefixCache, .kvQuantQ8, .kvQuantQ4, .lsa, .mtpAdaptive]
            if ink?.dspark != nil { caps.insert(.speculative) }
            if descriptor.vocab <= 393_216 && descriptor.hidden > 0 { caps.insert(.dflash) }
            if hasVisionWeights { caps.insert(.vision) }
            return caps
        }
        let usesMultiStreamHC = (descriptor.hcCount ?? 0) > 1
        var caps: Set<QWKnobCapability> = [
            .prefixCache,
            .kvQuantQ8,
            .kvQuantQ4,
            .lsa,
            .speculative,
        ]
        if descriptor.vocab <= 393_216 && descriptor.hidden > 0 {
            caps.insert(.dflash)
        }
        if !usesMultiStreamHC {
            caps.insert(.ropeYaRN)
            caps.insert(.mtpAdaptive)
        }
        if !usesMultiStreamHC && !descriptor.linearAttentionLayers.isEmpty {
            caps.insert(.gdnPrefill)
        }
        if hasVisionWeights {
            caps.insert(.vision)
        }
        if (descriptor.numExperts ?? 0) > 0 && descriptor.hcCount == nil {
            caps.insert(.expertStreaming)
        }
        return caps
    }

    /// Supported KV cache storage mode strings for `descriptor`.
    public static func supportedKVStorageModes(for descriptor: QWModelDescriptor) -> Set<String> {
        (descriptor.hcCount ?? 0) > 1 ? ["f32", "f16", "q8", "q4"] : ["f16", "q8", "q4"]
    }
}
