// QWServerSettingsKnobs.swift — The legacy settings, read as knob values.
// The settings file is global; the knob profile is per model. This is the
// mapping that lets the user's existing settings carry over into a model's
// profile instead of being left behind when the settings file retires.

import Foundation
import QWCore
import QWEngine

extension QWServerSettings {
    /// The settings that name a knob, by knob id.
    public var knobValues: [String: QWKnobValue] {
        // A setting that names no knob is not a knob value and stays where
        // it is. A knob the settings do not mention is left to its default.
        var out: [String: QWKnobValue] = [:]
        out["contextTokens"] = .int(maxContext)
        out["maxTokens"] = .int(maxTokens)
        out["temperature"] = .double(temperature)
        out["topP"] = .double(topP)
        out["topK"] = .int(topK)
        out["repetitionPenalty"] = .double(repetitionPenalty)
        out["seed"] = .int(seed)
        out["enableThinking"] = .bool(enableThinking)
        out["reasoningEffort"] = .string(reasoningEffort)
        out["systemPrompt"] = .string(systemPrompt)
        out["speculativeEnabled"] = .bool(speculativeDecodingEnabled)
        out["mtpDrafts"] = .int(mtpDrafts)
        out["mtpBackend"] = .string(mtpBackend)
        out["mtpRelaxedMargin"] = .double(mtpRelaxedMargin)
        out["mtpRelaxedMinProb"] = .double(mtpRelaxedMinProb)
        out["mtpDraftStrategy"] = .string(mtpDraftStrategy)
        out["mtpQuadraticSeeding"] = .bool(mtpQuadraticSeeding)
        out["prefillChunkRows"] = .int(prefillChunkSize)
        out["maxConcurrentSessions"] = .int(maxConcurrentSessions)
        out["warmCacheLimitGiB"] = .double(warmRAMLimitGB)
        out["ssdCacheLimitGiB"] = .double(ssdCacheLimitGB)
        out["cacheDir"] = .string(cacheDir)
        out["gdnPrefill"] = .string(gdnPrefill)
        out["evictPrefillScratch"] = .string(evictPrefillScratch)
        out["lsaMode"] = .string(lsaMode)
        out["lsaBudget"] = .int(lsaBudget)
        out["lsaClusterBlock"] = .int(lsaClusterBlock)
        out["lsaSidecarPath"] = .string(lsaSidecarPath)
        out["lazyLoad"] = .bool(lazyLoad)
        out["loadVision"] = .bool(loadVision)
        out["logLevel"] = .string(logLevel)
        out["kvMode"] = .string(kvMode)
        // A scaling factor with no scaling is a setting the user turned
        // off. Carrying it would migrate a scaling nobody asked for.
        if ropeScalingType != "none" {
            out["ropeScalingFactor"] = .double(ropeScalingFactor)
        }
        out["mtpAdaptiveDepth"] = .bool(mtpAdaptiveDepth)
        out["mtpEntropyThreshold"] = .double(mtpEntropyThreshold)
        out["dflashDraftPackPath"] = .string(dflashDraftPackPath)
        out["dflashSelectorTopK"] = .int(dflashSelectorTopK)
        for id in QWEngineTuning.knobIds {
            out[id] = engineTuning[id] ?? QWKnobRegistry.descriptor(id)?.defaultValue
        }
        return out
    }

    /// Setting CodingKey names that are scoped per model rather than global to the server/host.
    public static let modelScopedSettingKeys: Set<String> = [
        "maxContext", "kvMode", "ropeScalingType", "ropeScalingFactor",
        "temperature", "topP", "topK", "maxTokens", "repetitionPenalty", "seed",
        "enableThinking", "reasoningEffort", "systemPrompt", "loadVision",
        "speculativeDecodingEnabled", "mtpDrafts", "mtpBackend",
        "mtpRelaxedMargin", "mtpRelaxedMinProb", "mtpDraftStrategy", "mtpQuadraticSeeding",
        "mtpAdaptiveDepth", "mtpEntropyThreshold",
        "dflashDraftPackPath", "dflashSelectorTopK",
        "gdnPrefill", "evictPrefillScratch", "prefillChunkSize", "maxConcurrentSessions",
        "lsaMode", "lsaBudget", "lsaClusterBlock", "lsaSidecarPath", "engineTuning"
    ]

    /// Maps knob IDs from `QWKnobProfile` to `QWServerSettings` CodingKey names.
    public static func settingKeys(forKnobIds knobIds: Set<String>) -> Set<String> {
        var keys = Set<String>()
        for id in knobIds {
            switch id {
            case "contextTokens": keys.insert("maxContext")
            case "maxTokens": keys.insert("maxTokens")
            case "temperature": keys.insert("temperature")
            case "topP": keys.insert("topP")
            case "topK": keys.insert("topK")
            case "repetitionPenalty": keys.insert("repetitionPenalty")
            case "seed": keys.insert("seed")
            case "enableThinking": keys.insert("enableThinking")
            case "reasoningEffort": keys.insert("reasoningEffort")
            case "systemPrompt": keys.insert("systemPrompt")
            case "speculativeEnabled": keys.insert("speculativeDecodingEnabled")
            case "mtpDrafts": keys.insert("mtpDrafts")
            case "mtpBackend": keys.insert("mtpBackend")
            case "mtpRelaxedMargin": keys.insert("mtpRelaxedMargin")
            case "mtpRelaxedMinProb": keys.insert("mtpRelaxedMinProb")
            case "mtpDraftStrategy": keys.insert("mtpDraftStrategy")
            case "mtpQuadraticSeeding": keys.insert("mtpQuadraticSeeding")
            case "prefillChunkRows": keys.insert("prefillChunkSize")
            case "maxConcurrentSessions": keys.insert("maxConcurrentSessions")
            case "warmCacheLimitGiB": keys.insert("warmRAMLimitGB")
            case "ssdCacheLimitGiB":
                keys.insert("ssdCacheLimitGB")
                keys.insert("ssdCacheEnabled")
            case "cacheDir": keys.insert("cacheDir")
            case "gdnPrefill": keys.insert("gdnPrefill")
            case "evictPrefillScratch": keys.insert("evictPrefillScratch")
            case "lsaMode": keys.insert("lsaMode")
            case "lsaBudget": keys.insert("lsaBudget")
            case "lsaClusterBlock": keys.insert("lsaClusterBlock")
            case "lsaSidecarPath": keys.insert("lsaSidecarPath")
            case "lazyLoad": keys.insert("lazyLoad")
            case "loadVision": keys.insert("loadVision")
            case "logLevel": keys.insert("logLevel")
            case "kvMode": keys.insert("kvMode")
            case "ropeScalingFactor":
                keys.insert("ropeScalingFactor")
                keys.insert("ropeScalingType")
            case "mtpAdaptiveDepth": keys.insert("mtpAdaptiveDepth")
            case "mtpEntropyThreshold": keys.insert("mtpEntropyThreshold")
            case "dflashDraftPackPath": keys.insert("dflashDraftPackPath")
            case "dflashSelectorTopK": keys.insert("dflashSelectorTopK")
            case _ where QWEngineTuning.knobIds.contains(id): keys.insert("engineTuning")
            default: break
            }
        }
        return keys
    }

    /// Validated knob values suitable for persisting in `QWKnobProfile`.
    public var validatedKnobValues: [String: QWKnobValue] {
        var out: [String: QWKnobValue] = [:]
        for (id, rawValue) in knobValues {
            var value = rawValue
            if id == "mtpDraftStrategy", value == .string("mtp") {
                value = .string("neural")
            }
            guard let descriptor = QWKnobRegistry.descriptor(id),
                  (try? descriptor.validate(value)) != nil else {
                continue
            }
            out[id] = value
        }
        return out
    }

    /// Applies knob values from a per-model `QWKnobProfile` onto these settings.
    public mutating func apply(knobProfile: QWKnobProfile) {
        apply(knobValues: knobProfile.values)
    }

    /// Applies a dictionary of `QWKnobValue` entries onto these settings.
    public mutating func apply(knobValues values: [String: QWKnobValue]) {
        if let v = values["contextTokens"]?.intValue { maxContext = v }
        if let v = values["maxTokens"]?.intValue { maxTokens = v }
        if let v = values["temperature"]?.doubleValue { temperature = v }
        if let v = values["topP"]?.doubleValue { topP = v }
        if let v = values["topK"]?.intValue { topK = v }
        if let v = values["repetitionPenalty"]?.doubleValue { repetitionPenalty = v }
        if let v = values["seed"]?.intValue { seed = v }
        if let v = values["enableThinking"]?.boolValue { enableThinking = v }
        if let v = values["reasoningEffort"]?.stringValue { reasoningEffort = v }
        if let v = values["systemPrompt"]?.stringValue { systemPrompt = v }
        if let v = values["speculativeEnabled"]?.boolValue { speculativeDecodingEnabled = v }
        if let v = values["mtpDrafts"]?.intValue { mtpDrafts = v }
        if let v = values["mtpBackend"]?.stringValue { mtpBackend = v }
        if let v = values["mtpRelaxedMargin"]?.doubleValue { mtpRelaxedMargin = v }
        if let v = values["mtpRelaxedMinProb"]?.doubleValue { mtpRelaxedMinProb = v }
        if let v = values["mtpDraftStrategy"]?.stringValue { mtpDraftStrategy = v }
        if let v = values["mtpQuadraticSeeding"]?.boolValue { mtpQuadraticSeeding = v }
        if let v = values["prefillChunkRows"]?.intValue { prefillChunkSize = v }
        if let v = values["maxConcurrentSessions"]?.intValue { maxConcurrentSessions = max(1, min(16, v)) }
        if let v = values["warmCacheLimitGiB"]?.doubleValue { warmRAMLimitGB = v }
        if let v = values["ssdCacheLimitGiB"]?.doubleValue {
            ssdCacheLimitGB = v
            ssdCacheEnabled = v > 0
        }
        if let v = values["cacheDir"]?.stringValue { cacheDir = v }
        if let v = values["gdnPrefill"]?.stringValue { gdnPrefill = v }
        if let v = values["evictPrefillScratch"]?.stringValue { evictPrefillScratch = v }
        if let v = values["lsaMode"]?.stringValue { lsaMode = v }
        if let v = values["lsaBudget"]?.intValue { lsaBudget = v }
        if let v = values["lsaClusterBlock"]?.intValue { lsaClusterBlock = v }
        if let v = values["lsaSidecarPath"]?.stringValue { lsaSidecarPath = v }
        if let v = values["lazyLoad"]?.boolValue { lazyLoad = v }
        if let v = values["loadVision"]?.boolValue { loadVision = v }
        if let v = values["logLevel"]?.stringValue { logLevel = v }
        if let v = values["kvMode"]?.stringValue { kvMode = v }
        if let v = values["ropeScalingFactor"]?.doubleValue {
            ropeScalingFactor = v
            ropeScalingType = v > 1.0 ? "yarn" : "none"
        }
        if let v = values["mtpAdaptiveDepth"]?.boolValue { mtpAdaptiveDepth = v }
        if let v = values["mtpEntropyThreshold"]?.doubleValue { mtpEntropyThreshold = v }
        if let v = values["dflashDraftPackPath"]?.stringValue { dflashDraftPackPath = v }
        if let v = values["dflashSelectorTopK"]?.intValue { dflashSelectorTopK = v }
        for id in QWEngineTuning.knobIds { if let v = values[id] { setEngineTuning(id, v) } }
    }

    /// Sets one engine tuning knob; a value equal to its default is dropped so profiles stay a diff.
    public mutating func setEngineTuning(_ id: String, _ value: QWKnobValue?) {
        guard let value, let descriptor = QWKnobRegistry.descriptor(id),
              (try? descriptor.validate(value)) != nil, value != descriptor.defaultValue else {
            engineTuning[id] = nil
            return
        }
        engineTuning[id] = value
    }

    /// The effective engine tuning these settings name (environment not applied).
    public var resolvedEngineTuning: QWEngineTuning {
        QWEngineTuning(values: engineTuning, environment: [:])
    }

    /// Resets model-scoped settings to the model's config files (`config.json` and
    /// `generation_config.json`) with built-in engine defaults for everything outside
    /// the model config, while preserving host/network/system/alias settings.
    public mutating func applyModelConfigDefaults(packPath: String) {
        let defaults = QWServerSettings()
        let expanded = packPath.trimmingCharacters(in: .whitespacesAndNewlines).expandingTildePath
        let assets = !expanded.isEmpty ? (try? QWModelRuntimeAssets.load(packPath: expanded)) : nil
        let descriptor = assets?.descriptor
        let genDefaults = assets?.generationDefaults

        speculativeParameters = QWSpeculativeParameters.topPerformanceDefaults(for: descriptor)
        maxContext = descriptor?.maxPositionEmbeddings ?? defaults.maxContext
        kvMode = defaults.kvMode
        ropeScalingType = defaults.ropeScalingType
        ropeScalingFactor = defaults.ropeScalingFactor
        temperature = genDefaults?.effectiveTemperature ?? defaults.temperature
        topP = genDefaults?.topP ?? defaults.topP
        topK = genDefaults?.topK ?? defaults.topK
        maxTokens = genDefaults?.maxNewTokens ?? defaults.maxTokens
        repetitionPenalty = genDefaults?.repetitionPenalty ?? defaults.repetitionPenalty
        seed = defaults.seed
        enableThinking = defaults.enableThinking
        reasoningEffort = defaults.reasoningEffort
        systemPrompt = defaults.systemPrompt
        loadVision = defaults.loadVision
        gdnPrefill = defaults.gdnPrefill
        evictPrefillScratch = defaults.evictPrefillScratch
        lsaMode = defaults.lsaMode
        lsaBudget = defaults.lsaBudget
        lsaClusterBlock = defaults.lsaClusterBlock
        lsaSidecarPath = defaults.lsaSidecarPath
        engineTuning = [:]
        let isMoE = (descriptor?.numExperts ?? 0) > 0 || descriptor?.family == .qwen38FlashNext
        let hasLargeUnifiedMemory = Double(ProcessInfo.processInfo.physicalMemory) / 1_000_000_000.0 > 36.5
        prefillChunkSize = (isMoE && hasLargeUnifiedMemory) ? 4096 : defaults.prefillChunkSize
        maxConcurrentSessions = defaults.maxConcurrentSessions
    }
}

extension QWServerConfigurationResolver {
    /// Variational acceptance floor: CLI, then `QW(G)_MTP_RELAXED_MIN_PROB`, then the config file; clamped to [0, 1].
    static func resolveRelaxedMinProb(cli: Double?, environment: [String: String], file: Double,
                                      sources: inout [String: String]) -> Double {
        if let cli { sources["mtpRelaxedMinProb"] = "cli"; return min(1.0, max(0.0, cli)) }
        if let raw = environment["QW_MTP_RELAXED_MIN_PROB"] ?? environment["QWG_MTP_RELAXED_MIN_PROB"],
           let value = Double(raw) {
            sources["mtpRelaxedMinProb"] = "env"; return min(1.0, max(0.0, value))
        }
        sources["mtpRelaxedMinProb"] = "config-file"; return min(1.0, max(0.0, file))
    }
}
