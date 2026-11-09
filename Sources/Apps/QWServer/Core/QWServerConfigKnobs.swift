// QWServerConfigKnobs.swift — The knobs the engine is running with.
// The dashboard shows what the engine actually runs with, not what a file
// once said. A knob the engine does not carry is absent, never a zero.

import Foundation
import QWCore
import QWKV

extension QWEngineLifecycleActor.Config {
    /// The knob values this engine runs with, by knob id.
    public var knobValues: [String: QWKnobValue] {
        var out: [String: QWKnobValue] = [:]
        out["contextTokens"] = .int(maxContext)
        out["maxTokens"] = .int(maxTokens)
        out["temperature"] = .double(temperature)
        out["topP"] = .double(topP)
        out["topK"] = .int(topK)
        out["repetitionPenalty"] = .double(repetitionPenalty)
        out["seed"] = .int(seed)
        out["enableThinking"] = .bool(enableThinking)
        out["loadVision"] = .bool(loadVision)
        out["reasoningEffort"] = .string(reasoningEffort)
        out["systemPrompt"] = .string(systemPrompt)
        out["speculativeEnabled"] = .bool(speculativeDecodingEnabled)
        out["mtpDrafts"] = .int(mtpDrafts)
        out["prefillChunkRows"] = .int(prefillChunkSize)
        out["maxConcurrentSessions"] = .int(maxConcurrentSessions)
        out["warmCacheLimitGiB"] = .double(Double(cacheConfig.warmByteLimit) / 1_073_741_824)
        out["ssdCacheLimitGiB"] = .double(Double(cacheConfig.ssdByteLimit) / 1_073_741_824)
        out["cacheEntryLimit"] = .int(cacheConfig.entryLimit)
        out["cacheDir"] = .string(cacheConfig.cacheDirectory.path)
        out["prefixCacheEnabled"] = .bool(cacheConfig.warmByteLimit > 0)
        out["evictPrefillScratch"] = .string(evictPrefillScratch)
        out["gdnPrefill"] = .string(gdnPrefill)
        out["lsaMode"] = .string(lsaConfig.mode.rawValue)
        out["lsaBudget"] = .int(lsaConfig.budget)
        out["lsaClusterBlock"] = .int(lsaConfig.clusterBlock)
        out["lsaSidecarPath"] = .string(lsaConfig.sidecarPath)
        out["kvMode"] = .string(Self.modeName(kvMode))
        out["ropeScalingFactor"] = .double(ropeScalingFactor)
        out["mtpAdaptiveDepth"] = .bool(mtpAdaptiveDepth)
        out["mtpEntropyThreshold"] = .double(mtpAdaptiveEntropyThreshold)
        out["dflashSelectorTopK"] = .int(dflashSelectorTopK)
        // A knob with no value is absent: an engine that does not choose a
        // drafter says nothing about it.
        if let mtpBackend { out["mtpBackend"] = .string(mtpBackend) }
        if let dflashDraftPackPath { out["dflashDraftPackPath"] = .string(dflashDraftPackPath) }
        out.merge(engineTuning.knobValues) { _, tuned in tuned }
        return out
    }

    /// The KV mode as the registry names it.
    private static func modeName(_ mode: QWKVMode) -> String {
        switch mode {
        case .f16: return "f16"
        case .q8: return "q8"
        case .q4: return "q4"
        }
    }
}
