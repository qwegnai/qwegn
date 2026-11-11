import Foundation
import QWCore
import QWEngine
import QWQuant

/// Resolved runtime configuration adhering to the standard configuration hierarchy:
/// CLI Options > Environment Variables > Config File (~/.qwg/settings.json) > Built-in Defaults
public struct QWResolvedServerConfig: Sendable, Equatable {
    public var host: String, port: Int
    public var modelPackPath: String, modelName: String, modelAlias: String
    public var maxContext: Int, kvMode: String
    public var ropeScalingType: String, ropeScalingFactor: Double
    public var temperature: Double, topP: Double, topK: Int
    public var maxTokens: Int, modelMaxLength: Int?
    public var repetitionPenalty: Double, seed: Int
    public var enableThinking: Bool, reasoningEffort: String, systemPrompt: String
    public var speculativeDecodingEnabled: Bool, mtpDrafts: Int, mtpBackend: String?
    public var mtpRelaxedMargin: Double, mtpRelaxedMinProb: Double, mtpDraftStrategy: String, mtpQuadraticSeeding: Bool
    public var mtpAdaptiveDepth: Bool, mtpEntropyThreshold: Double
    public var dflashDraftPackPath: String, dflashSelectorTopK: Int
    public var gdnPrefill: String, evictPrefillScratch: String, prefillChunkSize: Int
    public var maxConcurrentSessions: Int
    public var lsaMode: String, lsaBudget: Int, lsaClusterBlock: Int, lsaSidecarPath: String
    public var warmRAMLimitGB: Double, ssdCacheEnabled: Bool, ssdCacheLimitGB: Double, cacheDir: String
    public var logDir: String, logFileName: String, logLevel: String
    public var lazy: Bool, sourceSummary: [String: String]
    public var loadVision: Bool = false
    /// Engine tuning from the settings/profile; `QW_*` env vars overlay it in the engine.
    public var engineTuning: QWEngineTuning = .default

    public init(
        host: String = "127.0.0.1", port: Int = 8080,
        modelPackPath: String = "", modelName: String = "Qwen3.8-27B", modelAlias: String = "",
        maxContext: Int = 4096, kvMode: String = "q4",
        ropeScalingType: String = "none", ropeScalingFactor: Double = 1.0,
        temperature: Double = 0.7, topP: Double = 0.9, topK: Int = 40,
        maxTokens: Int = 512, modelMaxLength: Int? = nil,
        repetitionPenalty: Double = 1.05, seed: Int = -1,
        enableThinking: Bool = true, reasoningEffort: String = "xhigh", systemPrompt: String = "",
        speculativeDecodingEnabled: Bool = true, mtpDrafts: Int = 3, mtpBackend: String? = "variational",
        mtpRelaxedMargin: Double = 1.5, mtpRelaxedMinProb: Double = QWSpeculativeParameters.defaultRelaxedMinProb,
        mtpDraftStrategy: String = "hybrid", mtpQuadraticSeeding: Bool = true,
        mtpAdaptiveDepth: Bool = false, mtpEntropyThreshold: Double = 0.7,
        dflashDraftPackPath: String = "", dflashSelectorTopK: Int = 16,
        gdnPrefill: String = "chunked", evictPrefillScratch: String = "auto",
        prefillChunkSize: Int = QWPrefillLimits.chunkRows,
        maxConcurrentSessions: Int = 8,
        lsaMode: String = "auto", lsaBudget: Int = QWLSAConfig.defaultBudget,
        lsaClusterBlock: Int = QWLSAConfig.defaultClusterBlock, lsaSidecarPath: String = "",
        warmRAMLimitGB: Double = 4.0, ssdCacheEnabled: Bool = false, ssdCacheLimitGB: Double = 10.0, cacheDir: String = "~/.qwg/cache",
        logDir: String = "~/.qwg/logs", logFileName: String = "qwg.log", logLevel: String = "info",
        lazy: Bool = true, loadVision: Bool = false, sourceSummary: [String: String] = [:]
    ) {
        self.host = host; self.port = port
        self.modelPackPath = modelPackPath; self.modelName = modelName; self.modelAlias = modelAlias
        self.maxContext = maxContext; self.kvMode = kvMode
        self.ropeScalingType = ropeScalingType; self.ropeScalingFactor = ropeScalingFactor
        self.temperature = temperature; self.topP = topP; self.topK = topK
        self.maxTokens = maxTokens; self.modelMaxLength = modelMaxLength
        self.repetitionPenalty = repetitionPenalty; self.seed = seed
        self.enableThinking = enableThinking; self.reasoningEffort = reasoningEffort; self.systemPrompt = systemPrompt
        self.speculativeDecodingEnabled = speculativeDecodingEnabled; self.mtpDrafts = mtpDrafts; self.mtpBackend = mtpBackend
        self.mtpRelaxedMargin = mtpRelaxedMargin; self.mtpRelaxedMinProb = mtpRelaxedMinProb; self.mtpDraftStrategy = mtpDraftStrategy; self.mtpQuadraticSeeding = mtpQuadraticSeeding
        self.mtpAdaptiveDepth = mtpAdaptiveDepth; self.mtpEntropyThreshold = mtpEntropyThreshold
        self.dflashDraftPackPath = dflashDraftPackPath; self.dflashSelectorTopK = dflashSelectorTopK
        self.gdnPrefill = gdnPrefill; self.evictPrefillScratch = evictPrefillScratch
        self.prefillChunkSize = prefillChunkSize
        self.maxConcurrentSessions = max(1, min(16, maxConcurrentSessions))
        self.lsaMode = lsaMode; self.lsaBudget = lsaBudget
        self.lsaClusterBlock = lsaClusterBlock; self.lsaSidecarPath = lsaSidecarPath
        self.warmRAMLimitGB = warmRAMLimitGB; self.ssdCacheEnabled = ssdCacheEnabled; self.ssdCacheLimitGB = ssdCacheLimitGB; self.cacheDir = cacheDir
        self.logDir = logDir; self.logFileName = logFileName; self.logLevel = logLevel
        self.lazy = lazy; self.loadVision = loadVision; self.sourceSummary = sourceSummary
    }

    /// Cohesive sampling parameters resolved for the server.
    public var samplingParameters: QWSamplingParameters {
        QWSamplingParameters(temperature: temperature, topP: topP, topK: topK,
                             maxTokens: maxTokens, repetitionPenalty: repetitionPenalty, seed: seed)
    }

    /// Cohesive speculative decoding parameters resolved for the server.
    public var speculativeParameters: QWSpeculativeParameters {
        QWSpeculativeParameters(enabled: speculativeDecodingEnabled, mtpDrafts: mtpDrafts, mtpBackend: mtpBackend,
                                mtpRelaxedMargin: mtpRelaxedMargin, mtpRelaxedMinProb: mtpRelaxedMinProb,
                                mtpDraftStrategy: mtpDraftStrategy, mtpQuadraticSeeding: mtpQuadraticSeeding,
                                adaptiveDepth: mtpAdaptiveDepth, entropyThreshold: mtpEntropyThreshold)
    }

    /// Cohesive Latent Sparse Attention configuration resolved for the server.
    public var lsaConfig: QWLSAConfig {
        QWLSAConfig(mode: QWLSAMode.resolve(lsaMode) ?? .off, budget: lsaBudget,
                    clusterBlock: lsaClusterBlock, sidecarPath: lsaSidecarPath)
    }
}

/// Resolves server settings with strict precedence:
/// Request > CLI Options > Environment Variables > Server Config > Model Config > Built-in Defaults
public enum QWServerConfigurationResolver {
    public enum Error: Swift.Error, CustomStringConvertible {
        case invalidSettingsFile(String)
        public var description: String {
            switch self {
            case .invalidSettingsFile(let path):
                return "QWServerConfigurationResolver: invalid settings JSON at '\(path)'"
            }
        }
    }

    private static func settingsSnapshot(at url: URL?) throws -> (settings: QWServerSettings, keys: Set<String>) {
        guard let url else { return (QWServerSettings(), []) }
        do {
            return try QWSettingsManager.shared.loadSettingsSnapshot(from: url)
        } catch let error as Error {
            throw error
        } catch {
            throw Error.invalidSettingsFile("\(url.path): \(error)")
        }
    }

    public static func resolve(
        cliConfigPath: String? = nil, cliPack: String? = nil, cliHost: String? = nil, cliPort: Int? = nil,
        cliModelName: String? = nil, cliModelAlias: String? = nil, cliMaxContext: Int? = nil, cliKVMode: String? = nil,
        cliTemperature: Double? = nil, cliTopP: Double? = nil, cliTopK: Int? = nil, cliMaxTokens: Int? = nil,
        cliRepetitionPenalty: Double? = nil, cliSeed: Int? = nil, cliEnableThinking: Bool? = nil,
        cliReasoningEffort: String? = nil, cliSystemPrompt: String? = nil, cliMTPDrafts: Int? = nil,
        cliMTPBackend: String? = nil, cliMTPRelaxedMargin: Double? = nil, cliMTPRelaxedMinProb: Double? = nil,
        cliMTPDraftStrategy: String? = nil, cliMTPQuadraticSeeding: Bool? = nil, cliMTPAdaptiveDepth: Bool? = nil,
        cliMTPEntropyThreshold: Double? = nil, cliDFlashDraftPack: String? = nil, cliDFlashBlockSize: Int? = nil,
        cliDFlashTopK: Int? = nil, cliGDNPrefill: String? = nil, cliEvictScratch: String? = nil,
        cliPrefillChunkSize: Int? = nil, cliMaxConcurrentSessions: Int? = nil, cliLSAMode: String? = nil,
        cliLSABudget: Int? = nil, cliLSAClusterBlock: Int? = nil, cliLSASidecarPath: String? = nil,
        cliWarmRAMLimitGB: Double? = nil, cliSSDCacheLimitGB: Double? = nil, cliCacheDir: String? = nil,
        cliLogDir: String? = nil, cliLogFileName: String? = nil, cliLogLevel: String? = nil, cliLazy: Bool? = nil,
        cliLoadVision: Bool? = nil, cliStreamNgrams: Bool? = nil,
        cliMemoryPolicy: String? = nil, cliOSReserveGiB: Double? = nil,
        cliResidentExpertsPerLayer: Int? = nil, cliMoESpecBudget: Int? = nil,
        cliIgnoreProfile: Bool = false,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> QWResolvedServerConfig {
        var sources: [String: String] = [:]
        let ignoreProfile = cliIgnoreProfile || environment["QWG_NO_PROFILE"] == "1"
        if ignoreProfile {
            sources["noProfile"] = cliIgnoreProfile ? "cli (--no-profile)" : "env (QWG_NO_PROFILE=1)"
        }

        // 1. Determine configuration file URL
        let resolvedConfigFileURL: URL? = {
            if let cliPath = cliConfigPath, !cliPath.isEmpty {
                sources["config"] = "cli (--config \(cliPath))"
                return cliPath.canonicalFileURL
            } else if let envPath = environment["QWG_CONFIG"], !envPath.isEmpty {
                sources["config"] = "env (QWG_CONFIG=\(envPath))"
                return envPath.canonicalFileURL
            } else if ignoreProfile {
                sources["config"] = "none (--no-profile)"
                return nil
            } else {
                let defaultURL = QWSettingsManager.defaultSettingsFileURL
                if FileManager.default.fileExists(atPath: defaultURL.path) {
                    sources["config"] = "default-file (~/.qwg/settings.json)"
                    return defaultURL
                }
                sources["config"] = "none (using defaults)"
                return nil
            }
        }()

        // 2. Load file settings if present (preserve alias lookup when --no-profile bypasses default settings.json)
        let snapshot = try settingsSnapshot(at: resolvedConfigFileURL)
        var rawFileSettings = snapshot.settings
        if ignoreProfile, resolvedConfigFileURL == nil,
           FileManager.default.fileExists(atPath: QWSettingsManager.defaultSettingsFileURL.path) {
            rawFileSettings.modelAliases = QWSettingsManager.shared.loadSettings().modelAliases
        }
        let rawConfiguredSettingKeys = snapshot.keys

        // 3. Resolve pack path: CLI > Env > File > Discovered in ~/.qwg/models
        var cliResolvedRef: (packPath: String, modelName: String, alias: String)?
        let resolvedPack: String = {
            if let p = cliPack, !p.isEmpty {
                sources["pack"] = "cli"
                let ref = rawFileSettings.resolveModelReference(p)
                cliResolvedRef = ref
                return ref.packPath
            }
            if let p = environment["QWG_PACK"] ?? environment["QWG_MODEL_PACK"], !p.isEmpty {
                sources["pack"] = "env"
                let ref = rawFileSettings.resolveModelReference(p)
                cliResolvedRef = ref
                return ref.packPath
            }
            if !rawFileSettings.modelPackPath.isEmpty {
                sources["pack"] = "config-file"
                return rawFileSettings.modelPackPath.expandingTildePath
            }
            let discovered = QWServerSettings.scanModelsDirectory()
            if let first = discovered.first(where: { !$0.packPath.isEmpty }) {
                sources["pack"] = "discovered (~/.qwg/models/\(first.name))"
                return first.packPath
            }
            sources["pack"] = "none"
            return ""
        }()

        let modelDescriptor: QWModelDescriptor?
        let modelDefaults: QWModelGenerationDefaults?
        if !resolvedPack.isEmpty, FileManager.default.fileExists(atPath: resolvedPack) {
            let assets = try QWModelRuntimeAssets.load(packPath: resolvedPack)
            modelDescriptor = assets.descriptor
            modelDefaults = assets.generationDefaults
        } else {
            modelDescriptor = nil
            modelDefaults = nil
        }

        // 4. Resolve Model Name: CLI > Env > CLI-Ref > File > Pack Derived
        let resolvedModelName: String = {
            if let m = cliModelName, !m.isEmpty {
                sources["modelName"] = "cli"
                return m
            }
            if let m = environment["QWG_MODEL_NAME"], !m.isEmpty {
                sources["modelName"] = "env"
                return m
            }
            if let ref = cliResolvedRef, !ref.modelName.isEmpty {
                sources["modelName"] = "cli"
                return ref.modelName
            }
            if !rawFileSettings.modelName.isEmpty {
                sources["modelName"] = "config-file"
                return rawFileSettings.modelName
            }
            if !resolvedPack.isEmpty {
                sources["modelName"] = "pack-derived"
                return QWModelRoot.canonicalModelName(for: URL(fileURLWithPath: resolvedPack))
            }
            sources["modelName"] = "default"
            return "Qwen3.8-27B"
        }()

        // 4b. Apply per-model profile or model-config defaults when switching models
        // or when no per-model profile exists yet for the selected model.
        let (fileSettings, configuredSettingKeys): (QWServerSettings, Set<String>) = {
            let modelProfile = ignoreProfile ? nil : resolvedConfigFileURL.flatMap {
                QWSettingsManager.shared.loadProfile(forModel: resolvedModelName, from: $0)
            }
            let fileSpecifiesAnotherModel = (rawConfiguredSettingKeys.contains("modelName")
                || rawConfiguredSettingKeys.contains("modelPackPath"))
                && resolvedModelName != rawFileSettings.modelName
                && modelDescriptor != nil
            let isDefaultConfigWithoutProfile = ((resolvedConfigFileURL?.standardizedFileURL
                == QWSettingsManager.defaultSettingsFileURL.standardizedFileURL)
                || (ignoreProfile && resolvedConfigFileURL == nil))
                && modelProfile == nil
                && modelDescriptor != nil

            if let profile = modelProfile {
                var effective = rawFileSettings
                if fileSpecifiesAnotherModel {
                    effective.applyModelConfigDefaults(packPath: resolvedPack)
                    effective.apply(knobProfile: profile)
                } else {
                    effective.apply(knobProfile: profile)
                    // Direct edits to settings.json for the active model take precedence when not switching models
                    for key in rawConfiguredSettingKeys {
                        switch key {
                        case "maxContext": effective.maxContext = rawFileSettings.maxContext
                        case "kvMode": effective.kvMode = rawFileSettings.kvMode
                        case "ropeScalingType": effective.ropeScalingType = rawFileSettings.ropeScalingType
                        case "ropeScalingFactor": effective.ropeScalingFactor = rawFileSettings.ropeScalingFactor
                        case "temperature": effective.temperature = rawFileSettings.temperature
                        case "topP": effective.topP = rawFileSettings.topP
                        case "topK": effective.topK = rawFileSettings.topK
                        case "maxTokens": effective.maxTokens = rawFileSettings.maxTokens
                        case "repetitionPenalty": effective.repetitionPenalty = rawFileSettings.repetitionPenalty
                        case "seed": effective.seed = rawFileSettings.seed
                        case "enableThinking": effective.enableThinking = rawFileSettings.enableThinking
                        case "loadVision": effective.loadVision = rawFileSettings.loadVision
                        case "reasoningEffort": effective.reasoningEffort = rawFileSettings.reasoningEffort
                        case "systemPrompt": effective.systemPrompt = rawFileSettings.systemPrompt
                        case "speculativeDecodingEnabled": effective.speculativeDecodingEnabled = rawFileSettings.speculativeDecodingEnabled
                        case "mtpDrafts": effective.mtpDrafts = rawFileSettings.mtpDrafts
                        case "mtpBackend": effective.mtpBackend = rawFileSettings.mtpBackend
                        case "mtpRelaxedMargin": effective.mtpRelaxedMargin = rawFileSettings.mtpRelaxedMargin
                        case "mtpRelaxedMinProb": effective.mtpRelaxedMinProb = rawFileSettings.mtpRelaxedMinProb
                        case "mtpDraftStrategy": effective.mtpDraftStrategy = rawFileSettings.mtpDraftStrategy
                        case "mtpQuadraticSeeding": effective.mtpQuadraticSeeding = rawFileSettings.mtpQuadraticSeeding
                        case "mtpAdaptiveDepth": effective.mtpAdaptiveDepth = rawFileSettings.mtpAdaptiveDepth
                        case "mtpEntropyThreshold": effective.mtpEntropyThreshold = rawFileSettings.mtpEntropyThreshold
                        case "dflashDraftPackPath": effective.dflashDraftPackPath = rawFileSettings.dflashDraftPackPath
                        case "dflashSelectorTopK": effective.dflashSelectorTopK = rawFileSettings.dflashSelectorTopK
                        case "gdnPrefill": effective.gdnPrefill = rawFileSettings.gdnPrefill
                        case "evictPrefillScratch": effective.evictPrefillScratch = rawFileSettings.evictPrefillScratch
                        case "prefillChunkSize": effective.prefillChunkSize = rawFileSettings.prefillChunkSize
                        case "maxConcurrentSessions": effective.maxConcurrentSessions = rawFileSettings.maxConcurrentSessions
                        case "lsaMode": effective.lsaMode = rawFileSettings.lsaMode
                        case "lsaBudget": effective.lsaBudget = rawFileSettings.lsaBudget
                        case "lsaClusterBlock": effective.lsaClusterBlock = rawFileSettings.lsaClusterBlock
                        case "lsaSidecarPath": effective.lsaSidecarPath = rawFileSettings.lsaSidecarPath
                        default: break
                        }
                    }
                }
                let profileKeys = QWServerSettings.settingKeys(forKnobIds: Set(profile.values.keys))
                let effectiveKeys = fileSpecifiesAnotherModel
                    ? rawConfiguredSettingKeys
                        .subtracting(QWServerSettings.modelScopedSettingKeys)
                        .union(profileKeys)
                    : rawConfiguredSettingKeys.union(profileKeys)
                return (effective, effectiveKeys)
            } else if fileSpecifiesAnotherModel || isDefaultConfigWithoutProfile {
                var effective = rawFileSettings
                effective.applyModelConfigDefaults(packPath: resolvedPack)
                let effectiveKeys = rawConfiguredSettingKeys
                    .subtracting(QWServerSettings.modelScopedSettingKeys)
                return (effective, effectiveKeys)
            } else {
                return (rawFileSettings, rawConfiguredSettingKeys)
            }
        }()

        // 5. Resolve Model Alias
        let resolvedModelAlias: String = {
            if let a = cliModelAlias, !a.isEmpty {
                sources["modelAlias"] = "cli"
                return a
            }
            if let a = environment["QWG_MODEL_ALIAS"], !a.isEmpty {
                sources["modelAlias"] = "env"
                return a
            }
            if let ref = cliResolvedRef, !ref.alias.isEmpty {
                sources["modelAlias"] = "config-file"
                return ref.alias
            }
            let mapped = fileSettings.alias(for: resolvedModelName)
            if !mapped.isEmpty {
                sources["modelAlias"] = "config-file"
                return mapped
            }
            if !fileSettings.modelAlias.isEmpty {
                sources["modelAlias"] = "config-file"
                return fileSettings.modelAlias
            }
            sources["modelAlias"] = "none"
            return ""
        }()

        // 6. Host & Port
        let resolvedHost: String = {
            if let h = cliHost, !h.isEmpty {
                sources["host"] = "cli"
                return h
            }
            if let h = environment["QWG_HOST"], !h.isEmpty {
                sources["host"] = "env"
                return h
            }
            sources["host"] = "config-file"
            return fileSettings.host
        }()

        let resolvedPort: Int = {
            if let p = cliPort, p > 0 {
                sources["port"] = "cli"
                return p
            }
            if let envStr = environment["QWG_PORT"], let p = Int(envStr), p > 0 {
                sources["port"] = "env"
                return p
            }
            sources["port"] = "config-file"
            return fileSettings.port
        }()

        // 7. Context & KV Mode
        let resolvedMaxContext: Int = {
            if let c = cliMaxContext, c > 0 {
                sources["maxContext"] = "cli"
                return c
            }
            if let envStr = environment["QWG_MAX_CONTEXT"], let c = Int(envStr), c > 0 {
                sources["maxContext"] = "env"
                return c
            }
            if configuredSettingKeys.contains("maxContext") {
                sources["maxContext"] = "config-file"
                return fileSettings.maxContext
            }
            if let nativeWindow = modelDescriptor?.maxPositionEmbeddings, nativeWindow > 0 {
                sources["maxContext"] = "model-config"
                return nativeWindow
            }
            sources["maxContext"] = "config-file"
            return fileSettings.maxContext
        }()

        let resolvedKVMode: String = {
            if let k = cliKVMode, !k.isEmpty {
                sources["kvMode"] = "cli"
                return k
            }
            if let k = environment["QWG_KV_MODE"], !k.isEmpty {
                sources["kvMode"] = "env"
                return k
            }
            sources["kvMode"] = "config-file"
            return fileSettings.kvMode
        }()

        // 8. Sampling & Generation Parameters
        let resolvedTemperature: Double = {
            if let t = cliTemperature { sources["temperature"] = "cli"; return t }
            if let envStr = environment["QWG_TEMPERATURE"], let t = Double(envStr) { sources["temperature"] = "env"; return t }
            if configuredSettingKeys.contains("temperature") { sources["temperature"] = "config-file"; return fileSettings.temperature }
            if let temperature = modelDefaults?.effectiveTemperature { sources["temperature"] = "model-config"; return temperature }
            sources["temperature"] = "config-file"; return fileSettings.temperature
        }()

        let resolvedTopP: Double = {
            if let p = cliTopP { sources["topP"] = "cli"; return p }
            if let envStr = environment["QWG_TOP_P"], let p = Double(envStr) { sources["topP"] = "env"; return p }
            if configuredSettingKeys.contains("topP") { sources["topP"] = "config-file"; return fileSettings.topP }
            if let topP = modelDefaults?.topP { sources["topP"] = "model-config"; return topP }
            sources["topP"] = "config-file"; return fileSettings.topP
        }()

        let resolvedTopK: Int = {
            if let k = cliTopK { sources["topK"] = "cli"; return k }
            if let envStr = environment["QWG_TOP_K"], let k = Int(envStr) { sources["topK"] = "env"; return k }
            if configuredSettingKeys.contains("topK") { sources["topK"] = "config-file"; return fileSettings.topK }
            if let topK = modelDefaults?.topK { sources["topK"] = "model-config"; return topK }
            sources["topK"] = "config-file"; return fileSettings.topK
        }()

        let resolvedMaxTokens: Int = {
            if let m = cliMaxTokens { sources["maxTokens"] = "cli"; return m }
            if let envStr = environment["QWG_MAX_TOKENS"], let m = Int(envStr) { sources["maxTokens"] = "env"; return m }
            if configuredSettingKeys.contains("maxTokens") { sources["maxTokens"] = "config-file"; return fileSettings.maxTokens }
            if let maxNewTokens = modelDefaults?.maxNewTokens { sources["maxTokens"] = "model-config"; return maxNewTokens }
            sources["maxTokens"] = "config-file"; return fileSettings.maxTokens
        }()

        let resolvedRepetitionPenalty: Double = {
            if let r = cliRepetitionPenalty { sources["repetitionPenalty"] = "cli"; return r }
            if let envStr = environment["QWG_REPETITION_PENALTY"], let r = Double(envStr) { sources["repetitionPenalty"] = "env"; return r }
            if configuredSettingKeys.contains("repetitionPenalty") { sources["repetitionPenalty"] = "config-file"; return fileSettings.repetitionPenalty }
            if let repetitionPenalty = modelDefaults?.repetitionPenalty { sources["repetitionPenalty"] = "model-config"; return repetitionPenalty }
            sources["repetitionPenalty"] = "config-file"; return fileSettings.repetitionPenalty
        }()

        let resolvedModelMaxLength: Int? = {
            let hasMaxTokensOverride = cliMaxTokens != nil || Int(environment["QWG_MAX_TOKENS"] ?? "") != nil
                || configuredSettingKeys.contains("maxTokens") || modelDefaults?.maxNewTokens != nil
            guard !hasMaxTokensOverride, let maxLength = modelDefaults?.maxLength else { return nil }
            sources["maxTokens"] = "model-config (max_length)"
            return maxLength
        }()

        let resolvedSeed: Int = {
            if let s = cliSeed { sources["seed"] = "cli"; return s }
            if let envStr = environment["QWG_SEED"], let s = Int(envStr) { sources["seed"] = "env"; return s }
            sources["seed"] = "config-file"; return fileSettings.seed
        }()

        // 9. Reasoning & System Prompt
        let resolvedEnableThinking: Bool = {
            if let th = cliEnableThinking { sources["enableThinking"] = "cli"; return th }
            if let envStr = environment["QWG_ENABLE_THINKING"] {
                sources["enableThinking"] = "env"; return envStr == "1" || envStr.lowercased() == "true"
            }
            sources["enableThinking"] = "config-file"; return fileSettings.enableThinking
        }()

        let resolvedReasoningEffort: String = {
            if let r = cliReasoningEffort, !r.isEmpty { sources["reasoningEffort"] = "cli"; return r }
            if let envStr = environment["QWG_REASONING_EFFORT"], !envStr.isEmpty { sources["reasoningEffort"] = "env"; return envStr }
            sources["reasoningEffort"] = "config-file"; return fileSettings.reasoningEffort
        }()

        let resolvedSystemPrompt: String = {
            if let s = cliSystemPrompt { sources["systemPrompt"] = "cli"; return s }
            if let envStr = environment["QWG_SYSTEM_PROMPT"] { sources["systemPrompt"] = "env"; return envStr }
            sources["systemPrompt"] = "config-file"; return fileSettings.systemPrompt
        }()

        // 10. MTP Speculation & DFlash 2
        let resolvedMTPDrafts: Int = {
            if let bs = cliDFlashBlockSize, (2...16).contains(bs) { sources["mtpDrafts"] = "cli (--dflash-window \(bs))"; return bs - 1 }
            if let d = cliMTPDrafts { sources["mtpDrafts"] = "cli"; return d }
            if let envBs = environment["QW_DFLASH_BLOCK_SIZE"] ?? environment["QWG_DFLASH_BLOCK_SIZE"],
               let bs = Int(envBs), (2...16).contains(bs) { sources["mtpDrafts"] = "env (QWG_DFLASH_BLOCK_SIZE=\(bs))"; return bs - 1 }
            if let envStr = environment["QWG_MTP_DRAFTS"], let d = Int(envStr) { sources["mtpDrafts"] = "env"; return d }
            sources["mtpDrafts"] = "config-file"; return fileSettings.mtpDrafts
        }()

        let resolvedDFlashDraftPackPath: String = {
            if let p = cliDFlashDraftPack, !p.isEmpty { sources["dflashDraftPackPath"] = "cli"; return p.canonicalPath }
            if let p = environment["QW_DFLASH_DRAFT_PACK"] ?? environment["QWG_DFLASH_DRAFT_PACK"], !p.isEmpty {
                sources["dflashDraftPackPath"] = "env"; return p.canonicalPath
            }
            if !fileSettings.dflashDraftPackPath.isEmpty {
                sources["dflashDraftPackPath"] = "config-file"; return fileSettings.dflashDraftPackPath.canonicalPath
            }
            if let first = QWServerSettings.scanDrafterModelsDirectory().first(where: { !$0.packPath.isEmpty }) {
                sources["dflashDraftPackPath"] = "discovered (~/.qwg/models/\(first.name))"; return first.packPath
            }
            sources["dflashDraftPackPath"] = "default"
            return QWDFlashConfig.defaultPackPath
        }()

        let resolvedMTPBackend: String? = {
            if let b = cliMTPBackend, !b.isEmpty { sources["mtpBackend"] = "cli"; return b }
            if let bs = cliDFlashBlockSize, (2...16).contains(bs) {
                sources["mtpBackend"] = "cli (--dflash-window)"
                return QWDFlashConfig.inferBackendAndMaxDrafts(packPath: resolvedDFlashDraftPackPath).backend
            }
            if let b = environment["QWG_MTP_BACKEND"], !b.isEmpty { sources["mtpBackend"] = "env"; return b }
            if !fileSettings.mtpBackend.isEmpty { sources["mtpBackend"] = "config-file"; return fileSettings.mtpBackend }
            return nil
        }()

        let resolvedDFlashSelectorTopK: Int = {
            if let tk = cliDFlashTopK, (1...16).contains(tk) { sources["dflashSelectorTopK"] = "cli"; return tk }
            if let envStr = environment["QW_DFLASH_TOP_K"] ?? environment["QWG_DFLASH_TOP_K"],
               let tk = Int(envStr), (1...16).contains(tk) { sources["dflashSelectorTopK"] = "env"; return tk }
            sources["dflashSelectorTopK"] = "config-file"; return max(1, min(16, fileSettings.dflashSelectorTopK))
        }()

        // 11. GDN Prefill & Eviction
        let resolvedGDNPrefill: String = {
            if let g = cliGDNPrefill, !g.isEmpty { sources["gdnPrefill"] = "cli"; return g }
            if let g = environment["QWG_GDN_PREFILL"], !g.isEmpty { sources["gdnPrefill"] = "env"; return g }
            sources["gdnPrefill"] = "config-file"; return fileSettings.gdnPrefill
        }()

        let resolvedEvictScratch: String = {
            if let e = cliEvictScratch, !e.isEmpty { sources["evictPrefillScratch"] = "cli"; return e }
            if let e = environment["QWG_EVICT_PREFILL_SCRATCH"], !e.isEmpty { sources["evictPrefillScratch"] = "env"; return e }
            sources["evictPrefillScratch"] = "config-file"; return fileSettings.evictPrefillScratch
        }()

        let resolvedPrefillChunkSize: Int = {
            if let c = cliPrefillChunkSize, c >= QWPrefillLimits.minRows {
                sources["prefillChunkSize"] = "cli"; return QWPrefillLimits.clampedRows(c)
            }
            if let envStr = environment["QWG_PREFILL_CHUNK_SIZE"], let c = Int(envStr), c >= QWPrefillLimits.minRows {
                sources["prefillChunkSize"] = "env"; return QWPrefillLimits.clampedRows(c)
            }
            sources["prefillChunkSize"] = "config-file"; return QWPrefillLimits.clampedRows(fileSettings.prefillChunkSize)
        }()

        let resolvedMaxConcurrentSessions: Int = {
            if let s = cliMaxConcurrentSessions, (1...16).contains(s) {
                sources["maxConcurrentSessions"] = "cli"; return s
            }
            if let envStr = environment["QWG_MAX_CONCURRENT_SESSIONS"] ?? environment["QW_MAX_CONCURRENT_SESSIONS"],
               let s = Int(envStr), (1...16).contains(s) {
                sources["maxConcurrentSessions"] = "env"; return s
            }
            sources["maxConcurrentSessions"] = "config-file"; return max(1, min(16, fileSettings.maxConcurrentSessions))
        }()

        let resolvedLSAMode: String = {
            if let m = cliLSAMode, !m.isEmpty { sources["lsaMode"] = "cli"; return m.lowercased() }
            if let m = environment["QWG_LSA_MODE"] ?? environment["QW_LSA_MODE"], !m.isEmpty {
                sources["lsaMode"] = "env"; return m.lowercased()
            }
            sources["lsaMode"] = "config-file"; return fileSettings.lsaMode.lowercased()
        }()

        let resolvedLSABudget: Int = {
            if let b = cliLSABudget { sources["lsaBudget"] = "cli"; return b }
            if let envStr = environment["QWG_LSA_BUDGET"] ?? environment["QW_LSA_BUDGET"], let b = Int(envStr) {
                sources["lsaBudget"] = "env"; return b
            }
            sources["lsaBudget"] = "config-file"; return fileSettings.lsaBudget
        }()

        let resolvedLSAClusterBlock: Int = {
            if let cb = cliLSAClusterBlock { sources["lsaClusterBlock"] = "cli"; return cb }
            if let envStr = environment["QWG_LSA_CLUSTER_BLOCK"] ?? environment["QW_LSA_CLUSTER_BLOCK"], let cb = Int(envStr) {
                sources["lsaClusterBlock"] = "env"; return cb
            }
            sources["lsaClusterBlock"] = "config-file"; return fileSettings.lsaClusterBlock
        }()

        let resolvedLSASidecarPath: String = {
            if let sp = cliLSASidecarPath, !sp.isEmpty { sources["lsaSidecarPath"] = "cli"; return sp }
            if let sp = environment["QWG_LSA_SIDECAR"] ?? environment["QW_LSA_SIDECAR"], !sp.isEmpty {
                sources["lsaSidecarPath"] = "env"; return sp
            }
            sources["lsaSidecarPath"] = "config-file"; return fileSettings.lsaSidecarPath
        }()

        // 12. Cache
        let resolvedWarmRAM: Double = {
            if let w = cliWarmRAMLimitGB { sources["warmRAM"] = "cli"; return w }
            if let envStr = environment["QWG_WARM_CACHE_LIMIT"], let w = Double(envStr) { sources["warmRAM"] = "env"; return w }
            sources["warmRAM"] = "config-file"; return fileSettings.warmRAMLimitGB
        }()

        let resolvedSSDCacheEnabled: Bool = {
            if let s = cliSSDCacheLimitGB { sources["ssdCacheEnabled"] = "cli"; return s > 0 }
            if let envStr = environment["QWG_SSD_CACHE_ENABLED"] ?? environment["QW_SSD_CACHE_ENABLED"] {
                sources["ssdCacheEnabled"] = "env"; return envStr == "1" || envStr.lowercased() == "true"
            }
            if let envStr = environment["QWG_SSD_CACHE_LIMIT"], let s = Double(envStr) { sources["ssdCacheEnabled"] = "env"; return s > 0 }
            sources["ssdCacheEnabled"] = "config-file"; return fileSettings.ssdCacheEnabled
        }()

        let resolvedSSDCache: Double = {
            if let s = cliSSDCacheLimitGB { sources["ssdCache"] = "cli"; return s }
            if let envStr = environment["QWG_SSD_CACHE_LIMIT"], let s = Double(envStr) { sources["ssdCache"] = "env"; return s }
            sources["ssdCache"] = "config-file"; return resolvedSSDCacheEnabled ? fileSettings.ssdCacheLimitGB : 0.0
        }()

        let resolvedCacheDir: String = {
            if let c = cliCacheDir, !c.isEmpty { sources["cacheDir"] = "cli"; return c }
            if let c = environment["QWG_CACHE_DIR"], !c.isEmpty { sources["cacheDir"] = "env"; return c }
            sources["cacheDir"] = "config-file"; return fileSettings.cacheDir
        }()

        // 13. Logging & Lazy
        let resolvedLogLevel: String = {
            if let l = cliLogLevel, !l.isEmpty { sources["logLevel"] = "cli"; return l }
            if let l = environment["QWG_LOG_LEVEL"], !l.isEmpty { sources["logLevel"] = "env"; return l }
            sources["logLevel"] = "config-file"; return fileSettings.logLevel
        }()

        let resolvedLogDir: String = {
            if let d = cliLogDir, !d.isEmpty { sources["logDir"] = "cli"; return d }
            if let d = environment["QWG_LOG_DIR"], !d.isEmpty { sources["logDir"] = "env"; return d }
            sources["logDir"] = "config-file"; return fileSettings.logDir
        }()

        let resolvedLogFile: String = {
            if let f = cliLogFileName, !f.isEmpty { sources["logFileName"] = "cli"; return f }
            if let f = environment["QWG_LOG_FILE"], !f.isEmpty { sources["logFileName"] = "env"; return f }
            sources["logFileName"] = "config-file"; return fileSettings.logFileName
        }()

        let resolvedLazy = cliLazy ?? fileSettings.lazyLoad

        let (resolvedRopeType, resolvedRopeFactor): (String, Double) = {
            if let envStr = environment["QW_ROPE_YARN_FACTOR"], let factor = Double(envStr), factor > 0 {
                sources["ropeScaling"] = "env (QW_ROPE_YARN_FACTOR=\(factor))"
                return (factor > 1.0 ? "yarn" : "none", factor)
            }
            let nativeMaxPos = (!resolvedPack.isEmpty
                ? (try? QWModelGeometry.load(packPath: resolvedPack).maxPositionEmbeddings)
                : nil) ?? QWModelGeometry.qwen38_27b.maxPositionEmbeddings
            if resolvedMaxContext > nativeMaxPos {
                let autoFactor = Double(resolvedMaxContext) / Double(nativeMaxPos)
                sources["ropeScaling"] = "auto-derived (context \(resolvedMaxContext) > native \(nativeMaxPos))"
                return ("yarn", autoFactor)
            }
            sources["ropeScaling"] = "config-file"
            return (fileSettings.ropeScalingType, fileSettings.ropeScalingFactor)
        }()

        let resolvedSpeculativeEnabled: Bool = {
            if let envStr = environment["QWG_SPECULATIVE_DECODING"] ?? environment["QW_SPECULATIVE"] {
                sources["speculativeDecodingEnabled"] = "env"
                return envStr == "1" || envStr.lowercased() == "true"
            }
            sources["speculativeDecodingEnabled"] = "config-file"; return fileSettings.speculativeDecodingEnabled
        }()

        let resolvedMTPAdaptiveDepth: Bool = {
            if let ad = cliMTPAdaptiveDepth { sources["mtpAdaptiveDepth"] = "cli"; return ad }
            if let envStr = environment["QW_MTP_ADAPTIVE_DEPTH"] ?? environment["QWG_MTP_ADAPTIVE_DEPTH"] {
                sources["mtpAdaptiveDepth"] = "env"; return envStr == "1" || envStr.lowercased() == "true"
            }
            sources["mtpAdaptiveDepth"] = "config-file"; return fileSettings.mtpAdaptiveDepth
        }()

        let resolvedMTPEntropyThreshold: Double = {
            if let et = cliMTPEntropyThreshold { sources["mtpEntropyThreshold"] = "cli"; return et }
            if let envStr = environment["QW_MTP_ADAPTIVE_ENTROPY_THRESHOLD"] ?? environment["QWG_MTP_ADAPTIVE_ENTROPY_THRESHOLD"],
               let val = Double(envStr) { sources["mtpEntropyThreshold"] = "env"; return val }
            sources["mtpEntropyThreshold"] = "config-file"; return fileSettings.mtpEntropyThreshold
        }()

        let resolvedMTPRelaxedMargin: Double = {
            if let rm = cliMTPRelaxedMargin { sources["mtpRelaxedMargin"] = "cli"; return max(0.0, rm) }
            if let envStr = environment["QW_MTP_RELAXED_MARGIN"] ?? environment["QWG_MTP_RELAXED_MARGIN"],
               let val = Double(envStr) { sources["mtpRelaxedMargin"] = "env"; return max(0.0, val) }
            sources["mtpRelaxedMargin"] = "config-file"; return max(0.0, fileSettings.mtpRelaxedMargin)
        }()
        let resolvedMTPRelaxedMinProb = resolveRelaxedMinProb(cli: cliMTPRelaxedMinProb, environment: environment,
                                                              file: fileSettings.mtpRelaxedMinProb, sources: &sources)

        let resolvedMTPDraftStrategy: String = {
            if let ds = cliMTPDraftStrategy, !ds.isEmpty { sources["mtpDraftStrategy"] = "cli"; return ds }
            if let envStr = environment["QW_MTP_DRAFT_STRATEGY"] ?? environment["QWG_MTP_DRAFT_STRATEGY"], !envStr.isEmpty {
                sources["mtpDraftStrategy"] = "env"; return envStr
            }
            sources["mtpDraftStrategy"] = "config-file"; return fileSettings.mtpDraftStrategy
        }()

        let resolvedMTPQuadraticSeeding: Bool = {
            if let qs = cliMTPQuadraticSeeding { sources["mtpQuadraticSeeding"] = "cli"; return qs }
            if let envStr = environment["QW_MTP_QUADRATIC_SEEDING"] ?? environment["QWG_MTP_QUADRATIC_SEEDING"] {
                sources["mtpQuadraticSeeding"] = "env"; return envStr == "1" || envStr.lowercased() == "true"
            }
            sources["mtpQuadraticSeeding"] = "config-file"; return fileSettings.mtpQuadraticSeeding
        }()

        let resolvedLoadVision: Bool = {
            if let cli = cliLoadVision {
                sources["loadVision"] = "cli (--load-vision)"; return cli
            }
            if let envStr = environment["QWG_LOAD_VISION"] ?? environment["QW_LOAD_VISION"] ?? environment["QW_VISION_EAGER"] {
                sources["loadVision"] = "env"; return envStr == "1" || envStr.lowercased() == "true"
            }
            sources["loadVision"] = "config-file"; return fileSettings.loadVision
        }()

        var resolvedTuning = fileSettings.resolvedEngineTuning
        if let cliPol = cliMemoryPolicy, let parsed = QWMemoryPolicy(rawValue: cliPol) {
            resolvedTuning.memoryPolicy = parsed
            sources["memoryPolicy"] = "cli (--memory-policy)"
        }
        if let cliOS = cliOSReserveGiB {
            resolvedTuning.osReserveGiB = cliOS
            sources["osReserveGiB"] = "cli (--os-reserve)"
        }
        if let cliRes = cliResidentExpertsPerLayer {
            resolvedTuning.residentExpertsPerLayer = cliRes > 0 ? cliRes : nil
            sources["residentExpertsPerLayer"] = "cli (--resident-experts-per-layer)"
        }
        if let cliBudget = cliMoESpecBudget {
            resolvedTuning.moeSpecBudget = cliBudget
            sources["moeSpecBudget"] = "cli (--moe-spec-budget)"
        }
        if let cliStream = cliStreamNgrams {
            resolvedTuning.streamNgrams = cliStream
            sources["streamNgrams"] = "cli (--stream-ngrams)"
        }

        for id in QWEngineTuning.knobIds where sources[id] == nil {
            if let env = QWEngineTuning.environmentNames[id], environment[env] != nil {
                sources[id] = "env"
            } else if fileSettings.engineTuning[id] != nil {
                sources[id] = "config-file"
            }
        }
        var resolvedConfig = QWResolvedServerConfig(
            host: resolvedHost, port: resolvedPort, modelPackPath: resolvedPack,
            modelName: resolvedModelName, modelAlias: resolvedModelAlias,
            maxContext: resolvedMaxContext, kvMode: resolvedKVMode,
            ropeScalingType: resolvedRopeType, ropeScalingFactor: resolvedRopeFactor,
            temperature: resolvedTemperature, topP: resolvedTopP, topK: resolvedTopK,
            maxTokens: resolvedMaxTokens, modelMaxLength: resolvedModelMaxLength,
            repetitionPenalty: resolvedRepetitionPenalty, seed: resolvedSeed,
            enableThinking: resolvedEnableThinking, reasoningEffort: resolvedReasoningEffort,
            systemPrompt: resolvedSystemPrompt, speculativeDecodingEnabled: resolvedSpeculativeEnabled,
            mtpDrafts: resolvedMTPDrafts, mtpBackend: resolvedMTPBackend,
            mtpRelaxedMargin: resolvedMTPRelaxedMargin, mtpRelaxedMinProb: resolvedMTPRelaxedMinProb, mtpDraftStrategy: resolvedMTPDraftStrategy,
            mtpQuadraticSeeding: resolvedMTPQuadraticSeeding, mtpAdaptiveDepth: resolvedMTPAdaptiveDepth,
            mtpEntropyThreshold: resolvedMTPEntropyThreshold, dflashDraftPackPath: resolvedDFlashDraftPackPath,
            dflashSelectorTopK: resolvedDFlashSelectorTopK, gdnPrefill: resolvedGDNPrefill,
            evictPrefillScratch: resolvedEvictScratch, prefillChunkSize: resolvedPrefillChunkSize,
            maxConcurrentSessions: resolvedMaxConcurrentSessions,
            lsaMode: resolvedLSAMode, lsaBudget: resolvedLSABudget,
            lsaClusterBlock: resolvedLSAClusterBlock, lsaSidecarPath: resolvedLSASidecarPath,
            warmRAMLimitGB: resolvedWarmRAM, ssdCacheEnabled: resolvedSSDCacheEnabled,
            ssdCacheLimitGB: resolvedSSDCache, cacheDir: resolvedCacheDir,
            logDir: resolvedLogDir, logFileName: resolvedLogFile,
            logLevel: resolvedLogLevel, lazy: resolvedLazy, loadVision: resolvedLoadVision, sourceSummary: sources
        )
        resolvedConfig.engineTuning = resolvedTuning
        return resolvedConfig
    }
}
