// QWSettingsCoherence.swift — What a settings change costs, and what the
// app should do about the server next. The rule lives here, in one place,
// so the menu bar and the server cannot disagree about it.

import Foundation
import QWCore
import QWEngine
import QWKV

/// When a changed setting takes effect.
public enum QWSettingScope: String, Sendable, Codable, CaseIterable {
    /// The value is read when it is needed; the change applies at once.
    case live
    /// The server read the value once, when its process started.
    case serverRestart
}

/// What the app should do about the server, given the settings as they
/// are now.
public enum QWServerStartDecision: String, Sendable, Equatable {
    /// Start the server.
    case start
    /// The server is already up; there is nothing to start.
    case alreadyRunning
    /// Leave the server down.
    case stayStopped
}

/// The rule that keeps the settings and the running server in agreement.
public enum QWSettingsCoherence {
    /// The settings the server resolves once, when its process starts (see
    /// QWServerConfigurationResolver). A change to one of them is not in
    /// force for the running server until the process restarts.
    public static let serverRestartKeys: Set<String> = [
        "host", "port",
        "modelPackPath", "modelName",
        "maxContext", "kvMode", "ropeScalingType", "ropeScalingFactor",
        "temperature", "topP", "topK", "maxTokens", "repetitionPenalty", "seed",
        "enableThinking", "reasoningEffort", "systemPrompt", "lazyLoad",
        "speculativeDecodingEnabled", "mtpDrafts", "mtpBackend",
        "mtpRelaxedMargin", "mtpRelaxedMinProb", "mtpDraftStrategy", "mtpQuadraticSeeding",
        "mtpAdaptiveDepth", "mtpEntropyThreshold",
        "dflashDraftPackPath", "dflashSelectorTopK",
        "gdnPrefill", "evictPrefillScratch", "prefillChunkSize", "maxConcurrentSessions",
        "lsaMode", "lsaBudget", "lsaClusterBlock", "lsaSidecarPath", "engineTuning",
        "warmRAMLimitGB", "ssdCacheEnabled", "ssdCacheLimitGB", "cacheDir",
        "logLevel",
        "keepServerRunningAfterQuit",
    ]

    /// The settings that are read on demand and apply at once: the alias
    /// table, the app's own log sink, and the app's own start-up choices.
    public static let liveKeys: Set<String> = [
        "modelAlias", "modelAliases", "loadVision",
        "logDir", "logFileName",
        "telemetryPollIntervalMs", "runAtStartup", "autoStartServer",
    ]

    /// When the value of `key` takes effect.
    public static func scope(of key: String) -> QWSettingScope {
        serverRestartKeys.contains(key) ? .serverRestart : .live
    }

    /// Whether a change to `key` waits for the server process to restart.
    public static func needsServerRestart(_ key: String) -> Bool {
        scope(of: key) == .serverRestart
    }

    /// The setting keys whose value differs between two revisions.
    public static func changedKeys(from old: QWServerSettings, to new: QWServerSettings) -> [String] {
        let before = canonicalValues(old)
        let after = canonicalValues(new)
        return Set(before.keys).union(after.keys)
            .filter { before[$0] != after[$0] }
            .sorted()
    }

    /// The changed keys the running server has not taken yet.
    public static func pendingServerRestartKeys(from old: QWServerSettings,
                                                to new: QWServerSettings) -> [String] {
        changedKeys(from: old, to: new).filter { needsServerRestart($0) }
    }

    /// The decision about the server, made from the settings as they are
    /// now — never from the copy the app was launched with.
    public static func startDecision(autoStartServer: Bool,
                                     serverRunning: Bool) -> QWServerStartDecision {
        if serverRunning { return .alreadyRunning }
        return autoStartServer ? .start : .stayStopped
    }

    /// What a setting does, in words the user can read.
    public static func effectNote(for key: String) -> String? {
        effectNotes[key]
    }

    /// A readable name for a setting, for the messages the user reads.
    public static func displayName(for key: String) -> String {
        displayNames[key] ?? key
    }

    private static let displayNames: [String: String] = [
        "host": "Listen host",
        "port": "Listen port",
        "modelPackPath": "Model pack path",
        "modelName": "Model",
        "maxContext": "Context limit",
        "kvMode": "KV cache mode",
        "ropeScalingType": "RoPE scaling",
        "ropeScalingFactor": "RoPE factor",
        "temperature": "Temperature",
        "topP": "Top-P",
        "topK": "Top-K",
        "maxTokens": "Max tokens",
        "repetitionPenalty": "Repetition penalty",
        "seed": "Random seed",
        "enableThinking": "Thinking mode",
        "reasoningEffort": "Reasoning effort",
        "systemPrompt": "Default system prompt",
        "lazyLoad": "Lazy load",
        "loadVision": "Load vision tower",
        "speculativeDecodingEnabled": "Speculative decoding",
        "mtpDrafts": "MTP draft tokens",
        "mtpBackend": "MTP backend",
        "mtpRelaxedMargin": "Variational acceptance margin",
        "mtpRelaxedMinProb": "Variational acceptance probability floor",
        "mtpDraftStrategy": "Draft proposal strategy",
        "mtpQuadraticSeeding": "Quadratic prefix seeding",
        "mtpAdaptiveDepth": "Adaptive MTP depth",
        "mtpEntropyThreshold": "MTP entropy threshold",
        "dflashDraftPackPath": "Drafter pack path",
        "dflashSelectorTopK": "Drafter selector top-K",
        "gdnPrefill": "GDN prefill mode",
        "evictPrefillScratch": "Scratch eviction",
        "prefillChunkSize": "Prefill chunk size",
        "maxConcurrentSessions": "Concurrent sessions",
        "lsaMode": "LSA mode",
        "lsaBudget": "LSA key budget",
        "lsaClusterBlock": "LSA cluster block size",
        "lsaSidecarPath": "LSA sidecar path",
        "engineTuning": "Engine performance & memory tuning",
        "warmRAMLimitGB": "Warm RAM limit",
        "ssdCacheEnabled": "Persist cache to SSD",
        "ssdCacheLimitGB": "Cold SSD limit",
        "cacheDir": "Cache directory",
        "logLevel": "Log level",
        "keepServerRunningAfterQuit": "Keep server running after quit",
    ]

    private static let effectNotes: [String: String] = [
        "keepServerRunningAfterQuit": "The server keeps serving after the app quits. The parent-pid watchdog is armed when the server starts, so this takes effect after a server restart.",
        "autoStartServer": "The server is started when the app starts.",
        "runAtStartup": "The app is started when you log in.",
        "telemetryPollIntervalMs": "How often the app asks the server for its numbers.",
        "modelAlias": "The name the active model answers to.",
        "modelAliases": "The names the installed models answer to.",
        "loadVision": "Loads or unloads the vision tower in place without restarting the engine or resetting the KV cache.",
        "logDir": "Where the app writes its log file.",
        "logFileName": "The app's log file name.",
    ]

    /// One canonical text per setting key, so two revisions can be
    /// compared field by field without a hand-written field list.
    private static func canonicalValues(_ settings: QWServerSettings) -> [String: String] {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(settings),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return [:]
        }
        var out: [String: String] = [:]
        for (key, value) in object {
            out[key] = canonicalText(value)
        }
        return out
    }

    private static func canonicalText(_ value: Any) -> String {
        let options: JSONSerialization.WritingOptions = [.sortedKeys, .fragmentsAllowed]
        if let data = try? JSONSerialization.data(withJSONObject: value, options: options),
           let text = String(data: data, encoding: .utf8) {
            return text
        }
        return String(describing: value)
    }

    /// Power-of-2 context stops from 64 tokens (2^6) up to 1M tokens (2^20 = 1,048,576).
    public static let contextSliderSteps: [Int] = (6...20).map { 1 << $0 }

    /// Maps an arbitrary token count to the nearest index in `contextSliderSteps`.
    public static func nearestContextStepIndex(for tokens: Int) -> Int {
        let clamped = max(contextSliderSteps.first ?? 64, min(contextSliderSteps.last ?? 1_048_576, tokens))
        var bestIdx = 0
        var bestDist = Double.greatestFiniteMagnitude
        let logVal = Foundation.log2(Double(clamped))
        for (idx, step) in contextSliderSteps.enumerated() {
            let dist = abs(Foundation.log2(Double(step)) - logVal)
            if dist < bestDist {
                bestDist = dist
                bestIdx = idx
            }
        }
        return bestIdx
    }

    /// Returns the token count for a slider step index in `0..<contextSliderSteps.count`.
    public static func contextTokens(forStepIndex index: Int) -> Int {
        let clamped = max(0, min(contextSliderSteps.count - 1, index))
        return contextSliderSteps[clamped]
    }

    /// Formats a context token count without static native/YaRN labels.
    public static func formatContextTokens(_ tokens: Int) -> String {
        let f = NumberFormatter()
        f.locale = Locale(identifier: "en_US")
        f.numberStyle = .decimal
        f.usesGroupingSeparator = true
        f.groupingSeparator = ","
        let formatted = f.string(from: NSNumber(value: tokens)) ?? "\(tokens)"
        if tokens >= 1_048_576 && tokens % 1_048_576 == 0 {
            return "\(tokens / 1_048_576)M (\(formatted) tokens)"
        }
        if tokens >= 1_024 && tokens % 1_024 == 0 {
            return "\(tokens / 1_024)k (\(formatted) tokens)"
        }
        return "\(formatted) tokens"
    }
}

extension QWServerSettings {
    private static let descriptorCacheLock = NSLock()
    nonisolated(unsafe) private static var cachedDescriptors: [String: QWModelDescriptor] = [:]

    /// Predicts total GPU memory required (weights + KV + scratch) for the current settings and model.
    public func estimateContextMemory(
        for descriptor: QWModelDescriptor? = nil,
        physicalMemoryBytes: Int? = QWAdmission.physicalMemoryBytes()
    ) -> QWContextMemoryEstimate? {
        let rawPath = modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let cacheKey = "\(modelName)|\(rawPath)"
        let resolvedDescriptor: QWModelDescriptor? = {
            if let descriptor { return descriptor }
            Self.descriptorCacheLock.lock()
            let hit = Self.cachedDescriptors[cacheKey]
            Self.descriptorCacheLock.unlock()
            if let hit { return hit }
            guard let loaded = try? validatePrimaryModel() else { return nil }
            Self.descriptorCacheLock.lock()
            Self.cachedDescriptors[cacheKey] = loaded
            Self.descriptorCacheLock.unlock()
            return loaded
        }()
        guard let desc = resolvedDescriptor else { return nil }
        let ref = resolveModelReference(rawPath.isEmpty ? modelName : rawPath)
        let resolvedKV: QWKVMode = switch kvMode.lowercased() {
        case "q8": .q8
        case "q4": .q4
        default: .f16
        }
        let drafts = speculativeDecodingEnabled ? mtpDrafts : 0
        let tuning = QWEngineTuning(values: engineTuning)
        return QWEngine.estimateContextMemory(
            descriptor: desc,
            packPath: ref.packPath.isEmpty ? nil : ref.packPath,
            contextTokens: maxContext,
            kvMode: resolvedKV,
            mtpDrafts: drafts,
            mtpBackend: mtpBackend,
            prefillChunkSize: prefillChunkSize,
            tuning: tuning,
            physicalMemoryBytes: physicalMemoryBytes
        )
    }
}
