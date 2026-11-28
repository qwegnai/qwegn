// QWCliEngineConfigResolver.swift — Deep module resolving CLI flags, settings,
// and model assets into validated QWEngineBuildOptions for `qwg run` and `qwg repl`.

import ArgumentParser
import Foundation
import QWCli
import QWCore
import QWEngine
import QWHead
import QWKV
import QWQuant
import QWServer
import QWToken

struct QWResolvedCliMemoryPack {
    let rootURL: URL
    let manifest: QWMemoryManifest
    let artifact: QWMemoryManifest.CacheArtifact
    let cacheFileURL: URL

    static func resolve(
        explicitRepoPath: String?,
        disableMemoryPack: Bool,
        modelName: String,
        geometryDigest: String,
        preferredKVMode: String?
    ) -> QWResolvedCliMemoryPack? {
        guard !disableMemoryPack else { return nil }
        let startURL: URL
        if let explicitRepoPath, !explicitRepoPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            startURL = URL(fileURLWithPath: explicitRepoPath.expandingTildePath).standardizedFileURL
        } else {
            startURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).standardizedFileURL
        }
        guard let rootURL = QWMemoryPack.discoverManifestRoot(walkingUpFrom: startURL),
              let manifest = try? QWMemoryManifest.load(from: rootURL),
              let artifact = manifest.matchingCache(
                  modelName: modelName,
                  geometryDigest: geometryDigest,
                  kvMode: preferredKVMode
              ) else {
            return nil
        }
        let cacheURL = manifest.cacheFileURL(for: artifact, rootURL: rootURL)
        guard FileManager.default.fileExists(atPath: cacheURL.path) else {
            return nil
        }
        return QWResolvedCliMemoryPack(
            rootURL: rootURL,
            manifest: manifest,
            artifact: artifact,
            cacheFileURL: cacheURL
        )
    }

    func encodeDeltaTokens(
        chatTemplate: QWChatTemplate,
        tokenizer: QWTokenizer
    ) -> (tokens: [Int], delta: QWMemoryPack.DeltaOverlay) {
        let delta = QWMemoryPack.computeDelta(for: manifest)
        guard delta.hasChanges, !delta.deltaBlock.isEmpty else {
            return ([], delta)
        }
        var opts = QWChatTemplateOptions()
        opts.addGenerationPrompt = false
        opts.enableThinking = .bool(false)
        let sentinel = "__QWG_MEMORY_USER_SENTINEL__"
        let sysMsg: QWJson = .object([
            ("role", .string("system")),
            ("content", .string(delta.deltaBlock))
        ])
        let userMsg: QWJson = .object([
            ("role", .string("user")),
            ("content", .string(sentinel))
        ])
        guard let fullWire = try? chatTemplate.render(messages: [sysMsg, userMsg], options: opts) else {
            return ([], delta)
        }
        let wire = QWMemoryPack.sliceBeforeSentinelUserTurn(fullWire: fullWire, sentinel: sentinel)
        guard let ids = try? tokenizer.encode(wire) else {
            return ([], delta)
        }
        return (ids, delta)
    }
}

struct QWResolvedCliEngineConfig {
    let resolvedPack: String
    let location: QWModelLocation
    let runtimeAssets: QWModelRuntimeAssets
    let effectiveSettings: QWServerSettings
    let profileApplied: Bool
    let defaultContext: Int
    let effectiveMtpDrafts: Int
    let effectiveMtpBackend: String?
    let resolvedBackend: QWMTPBackend?
    let resolvedKVMode: QWKVMode
    let tokenizer: QWTokenizer
    let specialTokens: QWModelSpecialTokens
    let lsaConfig: QWLSAConfig
    let resolvedCacheDirURL: URL
    let memoryPack: QWResolvedCliMemoryPack?

    static func resolvePackAndMTP(
        verb: String,
        modelOption: String?,
        pack: String?,
        knobs: QWGKnobOptions,
        dflashWindow: Int?,
        settings: QWServerSettings
    ) throws -> (resolvedPack: String, effectiveMtpDrafts: Int, effectiveMtpBackend: String?, resolvedBackend: QWMTPBackend?) {
        let rawModel = modelOption ?? pack ?? settings.modelPackPath
        let resolvedPack = settings.resolveModelReference(rawModel).packPath
        guard !resolvedPack.isEmpty else {
            FileHandle.standardError.write("\(verb): error: missing model (specify <pack> or --model <name|alias|path>)\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        var effectiveMtpDrafts = knobs.mtpDrafts ?? 1
        var effectiveMtpBackend = knobs.mtpBackend
        if let bs = dflashWindow {
            guard (2...16).contains(bs) else {
                FileHandle.standardError.write("\(verb): error: --dflash-window must be 2...16 (got \(bs))\n".data(using: .utf8)!)
                throw ExitCode(2)
            }
            effectiveMtpDrafts = bs - 1
            if effectiveMtpBackend == nil {
                effectiveMtpBackend = QWDFlashConfig.inferBackendAndMaxDrafts(packPath: knobs.dflashDraftPack).backend
            }
        }
        if let tk = knobs.dflashTopK {
            guard (1...16).contains(tk) else {
                FileHandle.standardError.write("\(verb): error: --dflash-top-k must be 1...16 (got \(tk))\n".data(using: .utf8)!)
                throw ExitCode(2)
            }
        }
        let resolvedBackend: QWMTPBackend?
        do {
            resolvedBackend = try effectiveMtpBackend.map { try QWMTPBackend.resolve(name: $0) }
        } catch {
            FileHandle.standardError.write("\(verb): error: \(error)\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        let maxDrafts = 15
        if resolvedBackend == .dflash2, knobs.mtpDrafts == nil, dflashWindow == nil {
            let inferred = QWDFlashConfig.inferBackendAndMaxDrafts(
                packPath: knobs.dflashDraftPack
                    ?? (effectiveMtpBackend?.lowercased() == "dspine" ? QWDFlashConfig.defaultDSpinePackPath : nil)
            )
            effectiveMtpDrafts = inferred.maxDrafts
        }
        guard effectiveMtpDrafts >= 0 && effectiveMtpDrafts <= maxDrafts else {
            FileHandle.standardError.write("\(verb): error: --mtp-drafts must be 0...\(maxDrafts) (got \(effectiveMtpDrafts))\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        do {
            _ = try QWMTPBackend.resolveSelection(resolvedBackend, drafts: effectiveMtpDrafts)
        } catch {
            FileHandle.standardError.write("\(verb): error: \(error)\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        return (resolvedPack, effectiveMtpDrafts, effectiveMtpBackend, resolvedBackend)
    }

    static func resolveAssetsAndKnobs(
        verb: String,
        resolvedPack: String,
        knobs: QWGKnobOptions,
        dflashWindow: Int?,
        initialMtpDrafts: Int,
        effectiveMtpBackend: String?,
        resolvedBackend: QWMTPBackend?,
        image: String?,
        video: String?,
        settings: QWServerSettings,
        validateDefaultContextImmediately: Bool,
        repoPath: String? = nil,
        disableMemoryPack: Bool = false,
        ignoreProfile: Bool = false
    ) throws -> QWResolvedCliEngineConfig {
        let location = try QWModelLocation(url: URL(fileURLWithPath: resolvedPack), assetsOverride: nil)
        let runtimeAssets = try QWModelRuntimeAssets.load(packPath: resolvedPack, assetsOverride: location.assets)
        let modelName = QWModelRoot.canonicalModelName(for: URL(fileURLWithPath: resolvedPack))
        var baseSettings = ignoreProfile ? QWServerSettings() : settings
        if ignoreProfile { baseSettings.modelAliases = settings.modelAliases }
        let effectivePair = QWSettingsManager.shared.effectiveSettings(
            forModel: modelName.isEmpty ? baseSettings.modelName : modelName,
            packPath: resolvedPack,
            base: baseSettings,
            ignoreProfile: ignoreProfile
        )
        let memoryPack = QWResolvedCliMemoryPack.resolve(
            explicitRepoPath: repoPath,
            disableMemoryPack: disableMemoryPack,
            modelName: modelName.isEmpty ? settings.modelName : modelName,
            geometryDigest: runtimeAssets.descriptor.geometryDigestV3Hex,
            preferredKVMode: knobs.kvMode
        )
        let minMemoryContext = memoryPack.map { max(4096, (($0.artifact.tokenCount + 16384 + 255) / 256) * 256) } ?? 0
        let defaultContext = max(runtimeAssets.descriptor.maxPositionEmbeddings, minMemoryContext)
        let wDir = location.weights.hasDirectoryPath
            ? location.weights
            : location.weights.deletingLastPathComponent()
        let hasDSparkFile = [
            wDir.appendingPathComponent("dspark.qwp"),
            location.assets.appendingPathComponent("dspark.qwp"),
            wDir.appendingPathComponent("mtp.safetensors"),
            location.assets.appendingPathComponent("mtp.safetensors")
        ].contains { FileManager.default.fileExists(atPath: $0.path) }
        let defaultStrategy = knobs.mtpDraftStrategy ?? effectivePair.settings.mtpDraftStrategy
        let isWeightFreeLoopNgram = !runtimeAssets.descriptor.capabilities.contains(.mtp)
            && effectivePair.settings.speculativeDecodingEnabled
            && effectivePair.settings.mtpDrafts > 0
            && defaultStrategy.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "ngram"
        var effectiveMtpDrafts = initialMtpDrafts
        if resolvedBackend == .dflash2, knobs.mtpDrafts == nil, dflashWindow == nil {
            let autoPack = QWDFlashFootprint.resolveDrafterPackPath(
                targetDescriptor: runtimeAssets.descriptor,
                explicitPath: knobs.dflashDraftPack
                    ?? (effectiveMtpBackend?.lowercased() == "dspine" ? QWDFlashConfig.defaultDSpinePackPath : nil),
                targetPackPath: resolvedPack
            )
            effectiveMtpDrafts = QWDFlashConfig.inferBackendAndMaxDrafts(packPath: autoPack).maxDrafts
        } else if knobs.mtpDrafts == nil && dflashWindow == nil && resolvedBackend != .dflash2 {
            if let dsCfg = runtimeAssets.descriptor.inkling2Config?.dspark, hasDSparkFile {
                if effectivePair.settings.speculativeDecodingEnabled {
                    let profDrafts = effectivePair.profile?.values["mtpDrafts"]?.intValue ?? 3
                    effectiveMtpDrafts = min(max(1, dsCfg.blockSize - 1), profDrafts)
                } else {
                    effectiveMtpDrafts = 0
                }
            } else if let dsCfg = runtimeAssets.descriptor.dsparkConfig,
                      QWModelDescriptor.DSparkConfig.resolvePackURL(near: resolvedPack) != nil {
                effectiveMtpDrafts = (resolvedBackend == .dspark) ? min(15, dsCfg.maxDrafts) : 0
            } else if !runtimeAssets.descriptor.capabilities.contains(.mtp) {
                effectiveMtpDrafts = isWeightFreeLoopNgram ? effectivePair.settings.mtpDrafts : 0
            }
        }
        let dsparkPackPresent = runtimeAssets.descriptor.dsparkConfig != nil
            && QWModelDescriptor.DSparkConfig.resolvePackURL(near: resolvedPack) != nil
        let autoSelectDSpark = resolvedBackend == nil && effectiveMtpDrafts > 0 && dsparkPackPresent
        let resolvedBackendForConfig = resolvedBackend
            ?? (autoSelectDSpark ? .dspark
                : (knobs.mtpDrafts == nil && isWeightFreeLoopNgram
                    ? (try? QWMTPBackend.resolve(name: effectivePair.settings.mtpBackend)) : nil))
        let resolvedBackendNameForConfig = effectiveMtpBackend
            ?? (autoSelectDSpark ? "dspark"
                : (knobs.mtpDrafts == nil && isWeightFreeLoopNgram ? effectivePair.settings.mtpBackend : nil))
        let profileKV = effectivePair.profile?.values["kvMode"]?.stringValue
            ?? effectivePair.profile?.values["kv_mode"]?.stringValue
        let effectiveKVString = memoryPack?.artifact.kvMode ?? knobs.kvMode ?? profileKV ?? "q4"
        let resolvedKVMode: QWKVMode
        do {
            _ = try QWMTPBackend.resolveSelection(resolvedBackend, drafts: effectiveMtpDrafts)
            resolvedKVMode = try QWCli.resolveKVMode(effectiveKVString)
            if validateDefaultContextImmediately {
                try QWCli.validateContextBudget(tokens: knobs.contextTokens ?? defaultContext)
            } else if let explicitContext = knobs.contextTokens {
                try QWCli.validateContextBudget(tokens: explicitContext)
            }
            let mmOpts = QWCli.MultimodalInputOptions(imagePath: image, videoPath: video)
            try mmOpts.validate()
        } catch {
            FileHandle.standardError.write("\(verb): error: \(error)\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        let rawLSAMode = knobs.lsaMode ?? memoryPack?.artifact.lsaMode ?? effectivePair.settings.lsaMode
        guard let resolvedLSAMode = QWLSAMode.resolve(rawLSAMode) else {
            FileHandle.standardError.write("\(verb): error: invalid --lsa-mode '\(rawLSAMode)' (must be off, auto, sidecar, svd, or qsa-cluster)\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        let envLoopShare = QWLSAConfig.parseLoopShareEnv(ProcessInfo.processInfo.environment[QWLSAConfig.loopShareEnvKey])
        let resolvedCrossLoopReuse = knobs.lsaCrossLoopReuse
            ?? envLoopShare?.enabled
            ?? effectivePair.settings.lsaCrossLoopReuse
        let resolvedLoopShareStride = resolvedCrossLoopReuse ? (envLoopShare?.stride ?? 0) : 0
        let explicitBudget = (knobs.lsaBudget != nil) || (effectivePair.settings.lsaBudget != QWLSAConfig.defaultBudget)
        let lsaConfig = QWLSAConfig(
            mode: resolvedLSAMode,
            budget: knobs.lsaBudget ?? effectivePair.settings.lsaBudget,
            clusterBlock: knobs.lsaClusterBlock ?? memoryPack?.manifest.clusterBlockTokens ?? effectivePair.settings.lsaClusterBlock,
            sidecarPath: knobs.lsaSidecarPath ?? effectivePair.settings.lsaSidecarPath,
            crossLoopReuse: resolvedCrossLoopReuse,
            loopShareStride: resolvedLoopShareStride,
            explicitBudget: explicitBudget
        )
        let tok = try runtimeAssets.makeTokenizer()
        let specialTokens = tok.resolveSpecialTokens(fallback: runtimeAssets.descriptor.specialTokens)
        let resolvedCacheDirURL = knobs.cacheDir.map(\.canonicalFileURL) ?? QWCacheConfig.defaultCacheDirectory
        return QWResolvedCliEngineConfig(
            resolvedPack: resolvedPack,
            location: location,
            runtimeAssets: runtimeAssets,
            effectiveSettings: effectivePair.settings,
            profileApplied: effectivePair.profile != nil,
            defaultContext: defaultContext,
            effectiveMtpDrafts: effectiveMtpDrafts,
            effectiveMtpBackend: resolvedBackendNameForConfig,
            resolvedBackend: resolvedBackendForConfig,
            resolvedKVMode: resolvedKVMode,
            tokenizer: tok,
            specialTokens: specialTokens,
            lsaConfig: lsaConfig,
            resolvedCacheDirURL: resolvedCacheDirURL,
            memoryPack: memoryPack
        )
    }

    func defaultPrefillChunkSize(explicitChunkSize: Int?) -> Int {
        if let explicitChunkSize {
            return explicitChunkSize
        }
        if !profileApplied && effectiveSettings.prefillChunkSize == QWPrefillLimits.chunkRows {
            return QWServerSettings.recommendedPrefillChunkSize(for: runtimeAssets.descriptor)
        }
        return effectiveSettings.prefillChunkSize
    }

    func makeBuildOptions(
        knobs: QWGKnobOptions,
        maxTokens: Int,
        prefillChunkSize: Int,
        prefixCacheDirectory: URL?,
        cacheConfig: QWCacheConfig?,
        prefillScratchEviction: QWPrefillScratchEvictionPolicy = .auto,
        gdnPrefill: String = "chunked",
        dsparkConfidenceSchedule: Bool? = nil
    ) -> QWEngineBuildOptions {
        var tuning = effectiveSettings.resolvedEngineTuning
        if let rawPolicy = knobs.memoryPolicy, let parsed = QWMemoryPolicy(rawValue: rawPolicy) {
            tuning.memoryPolicy = parsed
        }
        if let osReserve = knobs.osReserve {
            tuning.osReserveGiB = osReserve
        }
        let resolvedResidentExperts = knobs.residentExpertsPerLayer ?? tuning.residentExpertsPerLayer
        if let resE = knobs.residentExpertsPerLayer {
            tuning.residentExpertsPerLayer = resE > 0 ? resE : nil
        }
        if let moeSpec = knobs.moeSpecBudget {
            tuning.moeSpecBudget = moeSpec
        }
        if let streamNgrams = knobs.streamNgrams {
            tuning.streamNgrams = streamNgrams
        }
        return QWEngineBuildOptions(
            packPath: resolvedPack,
            maxTokens: maxTokens,
            kvMode: String(describing: resolvedKVMode),
            osReserveGiB: knobs.osReserve ?? tuning.osReserveGiB,
            prefillChunkSize: prefillChunkSize,
            prefixCacheDirectory: prefixCacheDirectory,
            cacheConfig: cacheConfig,
            mtpDrafts: effectiveMtpDrafts,
            mtpBackend: resolvedBackend,
            prefillScratchEviction: prefillScratchEviction,
            mtpAdaptiveDepth: knobs.mtpAdaptiveDepth,
            mtpAdaptiveEntropyThreshold: knobs.mtpEntropyThreshold,
            mtpRelaxedMargin: knobs.mtpRelaxedMargin ?? effectiveSettings.mtpRelaxedMargin,
            mtpRelaxedMinProb: knobs.mtpRelaxedMinProb ?? QWSpeculativeParameters.defaultRelaxedMinProb,
            mtpDraftStrategy: knobs.mtpDraftStrategy ?? effectiveSettings.mtpDraftStrategy,
            mtpQuadraticSeeding: knobs.mtpQuadraticSeeding ?? true,
            dflashDraftPackPath: knobs.dflashDraftPack
                ?? (effectiveMtpBackend?.lowercased() == "dspine"
                    ? QWDFlashConfig.defaultDSpinePackPath
                    : (resolvedBackend == .dflash2
                        ? QWDFlashFootprint.resolveDrafterPackPath(
                            targetDescriptor: runtimeAssets.descriptor,
                            explicitPath: nil,
                            targetPackPath: resolvedPack)
                        : nil)),
            dflashSelectorTopK: knobs.dflashTopK,
            dsparkConfidenceSchedule: dsparkConfidenceSchedule,
            gdnPrefill: gdnPrefill,
            lsaConfig: lsaConfig,
            residentExpertsPerLayer: resolvedResidentExperts,
            tuning: tuning
        )
    }

    func makeSamplingParameters(knobs: QWGKnobOptions, maxTokens: Int) -> QWSamplingParameters {
        let effectiveTemperature = knobs.temperature ?? 0.0
        return QWSamplingParameters(
            temperature: effectiveTemperature,
            topP: knobs.topP ?? 1.0,
            topK: knobs.topK ?? (effectiveTemperature > 0 ? 20 : 1),
            maxTokens: maxTokens,
            repetitionPenalty: knobs.repetitionPenalty ?? 1.0,
            seed: knobs.seed ?? -1
        )
    }

    func makeSpeculativeParameters(
        knobs: QWGKnobOptions,
        dsparkConfidenceSchedule: Bool? = nil
    ) -> QWSpeculativeParameters {
        QWSpeculativeParameters(
            enabled: effectiveMtpDrafts > 0,
            mtpDrafts: effectiveMtpDrafts,
            mtpBackend: effectiveMtpBackend,
            mtpRelaxedMargin: knobs.mtpRelaxedMargin ?? effectiveSettings.mtpRelaxedMargin,
            mtpRelaxedMinProb: knobs.mtpRelaxedMinProb ?? QWSpeculativeParameters.defaultRelaxedMinProb,
            mtpDraftStrategy: knobs.mtpDraftStrategy ?? effectiveSettings.mtpDraftStrategy,
            mtpQuadraticSeeding: knobs.mtpQuadraticSeeding ?? true,
            adaptiveDepth: knobs.mtpAdaptiveDepth ?? false,
            entropyThreshold: knobs.mtpEntropyThreshold ?? 0.5,
            dsparkConfidenceSchedule: dsparkConfidenceSchedule
        )
    }
}
