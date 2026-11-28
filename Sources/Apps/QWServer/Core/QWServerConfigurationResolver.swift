import Foundation
import QWCore
import QWEngine
import QWQuant

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
        cliApiKey: String? = nil, cliAllowLocalImagePaths: Bool? = nil, cliAllowRemoteImageURLs: Bool? = nil,
        cliPinModel: Bool? = nil,
        cliModelName: String? = nil, cliModelAlias: String? = nil, cliMaxContext: Int? = nil, cliKVMode: String? = nil,
        cliTemperature: Double? = nil, cliTopP: Double? = nil, cliTopK: Int? = nil, cliMaxTokens: Int? = nil,
        cliRepetitionPenalty: Double? = nil, cliSeed: Int? = nil, cliEnableThinking: Bool? = nil,
        cliReasoningEffort: String? = nil, cliSystemPrompt: String? = nil, cliMTPDrafts: Int? = nil,
        cliMTPBackend: String? = nil, cliMTPRelaxedMargin: Double? = nil, cliMTPRelaxedMinProb: Double? = nil,
        cliMTPDraftStrategy: String? = nil, cliMTPQuadraticSeeding: Bool? = nil, cliMTPAdaptiveDepth: Bool? = nil,
        cliMTPEntropyThreshold: Double? = nil, cliDFlashDraftPack: String? = nil, cliDFlashBlockSize: Int? = nil,
        cliDFlashTopK: Int? = nil, cliDSparkConfidenceSchedule: Bool? = nil, cliGDNPrefill: String? = nil, cliEvictScratch: String? = nil,
        cliPrefillChunkSize: Int? = nil, cliMaxConcurrentSessions: Int? = nil, cliLSAMode: String? = nil,
        cliLSABudget: Int? = nil, cliLSAClusterBlock: Int? = nil, cliLSASidecarPath: String? = nil,
        cliLSACrossLoopReuse: Bool? = nil,
        cliWarmRAMLimitGB: Double? = nil, cliSSDCacheLimitGB: Double? = nil, cliCacheDir: String? = nil,
        cliLogDir: String? = nil, cliLogFileName: String? = nil, cliLogLevel: String? = nil, cliLazy: Bool? = nil,
        cliLoadVision: Bool? = nil, cliStreamNgrams: Bool? = nil,
        cliMemoryPolicy: String? = nil, cliOSReserveGiB: Double? = nil,
        cliResidentExpertsPerLayer: Int? = nil, cliMoESpecBudget: Int? = nil,
        cliDecisionModelPackPath: String? = nil, cliDecisionComputeUnit: String? = nil,
        cliDecisionLoadVision: Bool? = nil, cliDecisionDefaultScoreReadout: String? = nil,
        cliDecisionDefaultTruncation: String? = nil, cliDecisionAlwaysEmitBox: Bool? = nil,
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

        let resolvedCatalogEntry: QWModelCatalogEntry? = (!resolvedPack.isEmpty && FileManager.default.fileExists(atPath: resolvedPack))
            ? QWModelCatalog.classify(name: ConfigNameHelper.name(for: resolvedPack), path: resolvedPack)
            : nil
        let isStandaloneDecisionTarget = resolvedCatalogEntry?.isDecisionCapable == true
        let modelDescriptor: QWModelDescriptor?
        let modelDefaults: QWModelGenerationDefaults?
        if !resolvedPack.isEmpty, FileManager.default.fileExists(atPath: resolvedPack) {
            if resolvedCatalogEntry?.isCoreAIBundle == true {
                modelDescriptor = nil
                modelDefaults = nil
            } else {
                let assets = try QWModelRuntimeAssets.load(packPath: resolvedPack)
                modelDescriptor = assets.descriptor
                modelDefaults = assets.generationDefaults
            }
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
                || (resolvedConfigFileURL == nil))
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
                        case "dsparkConfidenceSchedule": effective.dsparkConfidenceSchedule = rawFileSettings.dsparkConfidenceSchedule
                        case "gdnPrefill": effective.gdnPrefill = rawFileSettings.gdnPrefill
                        case "evictPrefillScratch": effective.evictPrefillScratch = rawFileSettings.evictPrefillScratch
                        case "prefillChunkSize": effective.prefillChunkSize = rawFileSettings.prefillChunkSize
                        case "maxConcurrentSessions": effective.maxConcurrentSessions = rawFileSettings.maxConcurrentSessions
                        case "lsaMode": effective.lsaMode = rawFileSettings.lsaMode
                        case "lsaBudget": effective.lsaBudget = rawFileSettings.lsaBudget
                        case "lsaClusterBlock": effective.lsaClusterBlock = rawFileSettings.lsaClusterBlock
                        case "lsaSidecarPath": effective.lsaSidecarPath = rawFileSettings.lsaSidecarPath
                        case "lsaCrossLoopReuse": effective.lsaCrossLoopReuse = rawFileSettings.lsaCrossLoopReuse
                        case "decisionModelPackPath": effective.decisionModelPackPath = rawFileSettings.decisionModelPackPath
                        case "decisionComputeUnit": effective.decisionComputeUnit = rawFileSettings.decisionComputeUnit
                        case "decisionLoadVision": effective.decisionLoadVision = rawFileSettings.decisionLoadVision
                        case "decisionDefaultScoreReadout": effective.decisionDefaultScoreReadout = rawFileSettings.decisionDefaultScoreReadout
                        case "decisionDefaultTruncation": effective.decisionDefaultTruncation = rawFileSettings.decisionDefaultTruncation
                        case "decisionAlwaysEmitBox": effective.decisionAlwaysEmitBox = rawFileSettings.decisionAlwaysEmitBox
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

        let resolvedApiKey: String = {
            if let k = cliApiKey, !k.isEmpty {
                sources["apiKey"] = "cli"
                return k
            }
            if let k = environment["QWG_API_KEY"], !k.isEmpty {
                sources["apiKey"] = "env"
                return k
            }
            sources["apiKey"] = fileSettings.apiKey.isEmpty ? "none" : "config-file"
            return fileSettings.apiKey
        }()

        let resolvedCorsAllowedOrigins: [String] = {
            if let raw = environment["QWG_CORS_ALLOWED_ORIGINS"], !raw.isEmpty {
                sources["corsAllowedOrigins"] = "env"
                return raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
            }
            sources["corsAllowedOrigins"] = fileSettings.corsAllowedOrigins.isEmpty ? "default" : "config-file"
            return fileSettings.corsAllowedOrigins
        }()

        let resolvedAllowLocalImagePaths: Bool = {
            if let v = cliAllowLocalImagePaths { sources["allowLocalImagePaths"] = "cli"; return v }
            if let envStr = environment["QWG_ALLOW_LOCAL_IMAGE_PATHS"] {
                sources["allowLocalImagePaths"] = "env"
                return envStr == "1" || envStr.lowercased() == "true"
            }
            sources["allowLocalImagePaths"] = "config-file"
            return fileSettings.allowLocalImagePaths
        }()

        let resolvedAllowRemoteImageURLs: Bool = {
            if let v = cliAllowRemoteImageURLs { sources["allowRemoteImageURLs"] = "cli"; return v }
            if let envStr = environment["QWG_ALLOW_REMOTE_IMAGE_URLS"] {
                sources["allowRemoteImageURLs"] = "env"
                return envStr == "1" || envStr.lowercased() == "true"
            }
            sources["allowRemoteImageURLs"] = "config-file"
            return fileSettings.allowRemoteImageURLs
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

        let extra = Self.resolveSpeculativeAndTuning(
            fileSettings: fileSettings, environment: environment, sources: &sources,
            resolvedPack: resolvedPack, resolvedMaxContext: resolvedMaxContext,
            cliMTPDrafts: cliMTPDrafts,
            cliMTPBackend: cliMTPBackend, cliDFlashDraftPack: cliDFlashDraftPack,
            cliDFlashBlockSize: cliDFlashBlockSize, cliDFlashTopK: cliDFlashTopK,
            cliDSparkConfidenceSchedule: cliDSparkConfidenceSchedule,
            cliGDNPrefill: cliGDNPrefill, cliEvictScratch: cliEvictScratch,
            cliPrefillChunkSize: cliPrefillChunkSize, cliMaxConcurrentSessions: cliMaxConcurrentSessions,
            cliLSAMode: cliLSAMode, cliLSABudget: cliLSABudget,
            cliLSAClusterBlock: cliLSAClusterBlock, cliLSASidecarPath: cliLSASidecarPath,
            cliLSACrossLoopReuse: cliLSACrossLoopReuse,
            cliWarmRAMLimitGB: cliWarmRAMLimitGB, cliSSDCacheLimitGB: cliSSDCacheLimitGB,
            cliCacheDir: cliCacheDir, cliLogDir: cliLogDir, cliLogFileName: cliLogFileName,
            cliLogLevel: cliLogLevel, cliLazy: cliLazy,
            cliMTPAdaptiveDepth: cliMTPAdaptiveDepth, cliMTPEntropyThreshold: cliMTPEntropyThreshold,
            cliMTPRelaxedMargin: cliMTPRelaxedMargin, cliMTPRelaxedMinProb: cliMTPRelaxedMinProb,
            cliMTPDraftStrategy: cliMTPDraftStrategy, cliMTPQuadraticSeeding: cliMTPQuadraticSeeding,
            cliLoadVision: cliLoadVision, cliMemoryPolicy: cliMemoryPolicy,
            cliOSReserveGiB: cliOSReserveGiB, cliResidentExpertsPerLayer: cliResidentExpertsPerLayer,
            cliMoESpecBudget: cliMoESpecBudget, cliStreamNgrams: cliStreamNgrams,
            cliDecisionModelPackPath: cliDecisionModelPackPath,
            cliDecisionComputeUnit: cliDecisionComputeUnit,
            cliDecisionLoadVision: cliDecisionLoadVision,
            cliDecisionDefaultScoreReadout: cliDecisionDefaultScoreReadout,
            cliDecisionDefaultTruncation: cliDecisionDefaultTruncation,
            cliDecisionAlwaysEmitBox: cliDecisionAlwaysEmitBox)
        let finalModelPackPath: String
        let finalDecisionModelPackPath: String
        if isStandaloneDecisionTarget {
            finalModelPackPath = ""
            if extra.decisionModelPackPath.isEmpty {
                finalDecisionModelPackPath = resolvedPack
                sources["decisionModelPackPath"] = sources["pack"] ?? "auto-decision"
            } else {
                finalDecisionModelPackPath = extra.decisionModelPackPath
            }
        } else {
            finalModelPackPath = resolvedPack
            finalDecisionModelPackPath = extra.decisionModelPackPath
        }
        let resolvedPinModel: Bool = {
            if let p = cliPinModel {
                sources["pinModel"] = "cli (--pin-model)"
                return p
            }
            if environment["QWG_PIN_MODEL"] == "1" {
                sources["pinModel"] = "env (QWG_PIN_MODEL=1)"
                return true
            }
            return false
        }()
        var resolvedConfig = QWResolvedServerConfig(
            host: resolvedHost, port: resolvedPort,
            apiKey: resolvedApiKey, corsAllowedOrigins: resolvedCorsAllowedOrigins,
            allowLocalImagePaths: resolvedAllowLocalImagePaths, allowRemoteImageURLs: resolvedAllowRemoteImageURLs,
            pinModel: resolvedPinModel,
            modelPackPath: finalModelPackPath,
            modelName: resolvedModelName, modelAlias: resolvedModelAlias,
            maxContext: resolvedMaxContext, kvMode: resolvedKVMode,
            ropeScalingType: extra.ropeType, ropeScalingFactor: extra.ropeFactor,
            temperature: resolvedTemperature, topP: resolvedTopP, topK: resolvedTopK,
            maxTokens: resolvedMaxTokens, modelMaxLength: resolvedModelMaxLength,
            repetitionPenalty: resolvedRepetitionPenalty, seed: resolvedSeed,
            enableThinking: resolvedEnableThinking, reasoningEffort: resolvedReasoningEffort,
            systemPrompt: resolvedSystemPrompt, speculativeDecodingEnabled: extra.speculativeEnabled,
            mtpDrafts: resolvedMTPDrafts, mtpBackend: extra.mtpBackend,
            mtpRelaxedMargin: extra.mtpRelaxedMargin, mtpRelaxedMinProb: extra.mtpRelaxedMinProb, mtpDraftStrategy: extra.mtpDraftStrategy,
            mtpQuadraticSeeding: extra.mtpQuadraticSeeding, mtpAdaptiveDepth: extra.mtpAdaptiveDepth,
            mtpEntropyThreshold: extra.mtpEntropyThreshold, dflashDraftPackPath: extra.dflashDraftPackPath,
            dflashSelectorTopK: extra.dflashSelectorTopK,
            dsparkConfidenceSchedule: extra.dsparkConfidenceSchedule,
            gdnPrefill: extra.gdnPrefill,
            evictPrefillScratch: extra.evictPrefillScratch, prefillChunkSize: extra.prefillChunkSize,
            maxConcurrentSessions: extra.maxConcurrentSessions,
            lsaMode: extra.lsaMode, lsaBudget: extra.lsaBudget,
            lsaClusterBlock: extra.lsaClusterBlock, lsaSidecarPath: extra.lsaSidecarPath,
            lsaCrossLoopReuse: extra.lsaCrossLoopReuse, lsaLoopShareStride: extra.lsaLoopShareStride,
            warmRAMLimitGB: extra.warmRAM, ssdCacheEnabled: extra.ssdCacheEnabled,
            ssdCacheLimitGB: extra.ssdCache, cacheDir: extra.cacheDir,
            logDir: extra.logDir, logFileName: extra.logFile,
            logLevel: extra.logLevel, lazy: extra.lazy, loadVision: extra.loadVision,
            decisionModelPackPath: finalDecisionModelPackPath,
            decisionComputeUnit: extra.decisionComputeUnit,
            decisionLoadVision: extra.decisionLoadVision,
            decisionDefaultScoreReadout: extra.decisionDefaultScoreReadout,
            decisionDefaultTruncation: extra.decisionDefaultTruncation,
            decisionAlwaysEmitBox: extra.decisionAlwaysEmitBox,
            sourceSummary: sources
        )
        resolvedConfig.engineTuning = extra.tuning
        return resolvedConfig
    }
}

private enum ConfigNameHelper {
    static func name(for path: String) -> String {
        QWModelRoot.canonicalModelName(for: URL(fileURLWithPath: path))
    }
}
