// QWEngineLifecycleActor+MemoryPack.swift — Codebase & knowledge memory pack
// mounting, auto-discovery from agent system prompts, and kvMode enforcement.

import Foundation
import QWCli
import QWCore
import QWEngine
import QWKV
import QWToken

extension QWEngineLifecycleActor.Config {
    /// Returns a copy of this configuration with selected runtime fields updated while preserving all other fields and `engineTuning`.
    public func updating(
        maxContext: Int? = nil,
        kvMode: QWKVMode? = nil,
        lsaConfig: QWLSAConfig? = nil,
        sourceSummary: [String: String]? = nil
    ) -> QWEngineLifecycleActor.Config {
        var copy = QWEngineLifecycleActor.Config(
            packPath: packPath,
            modelName: modelName,
            maxContext: maxContext ?? self.maxContext,
            kvMode: kvMode ?? self.kvMode,
            speculativeDecodingEnabled: speculativeDecodingEnabled,
            mtpDrafts: mtpDrafts,
            mtpBackend: mtpBackend,
            mtpRelaxedMargin: mtpRelaxedMargin,
            mtpRelaxedMinProb: mtpRelaxedMinProb,
            mtpDraftStrategy: mtpDraftStrategy,
            mtpQuadraticSeeding: mtpQuadraticSeeding,
            mtpAdaptiveDepth: mtpAdaptiveDepth,
            mtpAdaptiveEntropyThreshold: mtpAdaptiveEntropyThreshold,
            dflashDraftPackPath: dflashDraftPackPath,
            dflashSelectorTopK: dflashSelectorTopK,
            ropeScalingType: ropeScalingType,
            ropeScalingFactor: ropeScalingFactor,
            gdnPrefill: gdnPrefill,
            evictPrefillScratch: evictPrefillScratch,
            prefillChunkSize: prefillChunkSize,
            maxConcurrentSessions: maxConcurrentSessions,
            lsaConfig: lsaConfig ?? self.lsaConfig,
            cacheConfig: cacheConfig,
            host: host,
            port: port,
            temperature: temperature,
            topP: topP,
            topK: topK,
            maxTokens: maxTokens,
            modelMaxLength: modelMaxLength,
            repetitionPenalty: repetitionPenalty,
            seed: seed,
            enableThinking: enableThinking,
            reasoningEffort: reasoningEffort,
            systemPrompt: systemPrompt,
            loadVision: loadVision,
            sourceSummary: sourceSummary ?? self.sourceSummary
        )
        copy.engineTuning = engineTuning
        return copy
    }

    /// Translates the server lifecycle configuration and resolved speculation policy into unified `QWEngineBuildOptions`.
    public func makeEngineBuildOptions(speculation: QWResolvedSpeculation) throws -> QWEngineBuildOptions {
        let resolvedEviction = try QWPrefillScratchEvictionPolicy.parse(evictPrefillScratch)
        let yarnFactor: Double? = ropeScalingType.lowercased() == "yarn" ? ropeScalingFactor : nil
        let fnCacheDir = (cacheConfig.ssdByteLimit > 0 || cacheConfig.warmByteLimit > 0)
            ? cacheConfig.cacheDirectory
            : nil
        return QWEngineBuildOptions(
            packPath: packPath,
            maxTokens: maxContext,
            kvMode: String(describing: kvMode),
            prefillChunkSize: prefillChunkSize,
            prefixCacheDirectory: fnCacheDir,
            cacheConfig: cacheConfig,
            mtpDrafts: speculation.engineDrafts,
            mtpBackend: speculation.engineBackend,
            prefillScratchEviction: resolvedEviction,
            mtpAdaptiveDepth: mtpAdaptiveDepth,
            mtpAdaptiveEntropyThreshold: mtpAdaptiveEntropyThreshold,
            mtpRelaxedMargin: mtpRelaxedMargin,
            mtpRelaxedMinProb: mtpRelaxedMinProb,
            mtpDraftStrategy: mtpDraftStrategy,
            mtpQuadraticSeeding: mtpQuadraticSeeding,
            ropeYarnFactor: yarnFactor,
            dflashDraftPackPath: dflashDraftPackPath,
            dflashSelectorTopK: dflashSelectorTopK,
            gdnPrefill: gdnPrefill,
            lsaConfig: lsaConfig,
            maxConcurrentSessions: maxConcurrentSessions,
            tuning: engineTuning
        )
    }
}

extension QWEngineLifecycleActor {

    /// Strips `<org>/` prefix and trailing quantization/packaging tags (`-hq8g32-hq4g64`, `-mlx-serve-mixed-4-8bit`, etc.).
    public static func canonicalModelFamilyStem(_ raw: String) -> String {
        let leaf = (raw.split(separator: "/").last.map(String.init) ?? raw)
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard !leaf.isEmpty else { return "" }
        let parts = leaf.split(separator: "-").map(String.init)
        var end = parts.count
        while end > 1 {
            let p = parts[end - 1]
            let isQuantTag = p.hasPrefix("hq") || p.hasPrefix("aq")
                || p == "mlx" || p == "serve" || p == "mixed"
                || p == "4bit" || p == "8bit" || p == "fp16" || p == "bf16"
                || (p.allSatisfy(\.isNumber) && end >= 2 && (parts[end - 2] == "mixed" || parts[end - 2] == "serve"))
            if isQuantTag {
                end -= 1
            } else {
                break
            }
        }
        return parts[0..<end].joined(separator: "-")
    }

    /// Switches to a different installed model in `~/.qwg/models` if `requestedModel` (by alias or model name)
    /// resolves to a different valid model pack than `config.packPath`.
    public func selectModelIfNeeded(_ requestedModel: String?, allowSameFamilyQuantSwitch: Bool = true) async throws {
        guard let rawReq = requestedModel?.trimmingCharacters(in: .whitespacesAndNewlines), !rawReq.isEmpty else {
            return
        }
        let req = String(rawReq.split(separator: "@", maxSplits: 1).first ?? Substring(rawReq))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !req.isEmpty else { return }
        while case .unloading = state {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        let settings = QWSettingsManager.shared.loadSettings()
        let activeAlias = settings.alias(for: config.modelName)
        if req.caseInsensitiveCompare(config.modelName) == .orderedSame
            || (!activeAlias.isEmpty && req.caseInsensitiveCompare(activeAlias) == .orderedSame) {
            return
        }
        if !allowSameFamilyQuantSwitch, !config.packPath.isEmpty {
            let reqStem = Self.canonicalModelFamilyStem(req)
            let curStem = Self.canonicalModelFamilyStem(config.modelName)
            let aliasStem = Self.canonicalModelFamilyStem(activeAlias)
            if !reqStem.isEmpty && (reqStem == curStem || (!aliasStem.isEmpty && reqStem == aliasStem)) {
                return
            }
        }
        let ref = settings.resolveModelReference(req)
        let refCanonical = URL(fileURLWithPath: ref.packPath).resolvingSymlinksInPath().path
        let curCanonical = URL(fileURLWithPath: config.packPath.expandingTildePath).resolvingSymlinksInPath().path
        guard !ref.packPath.isEmpty,
              refCanonical != curCanonical,
              FileManager.default.fileExists(atPath: ref.packPath) else {
            return
        }
        if !allowSameFamilyQuantSwitch, !config.packPath.isEmpty {
            let refStem = Self.canonicalModelFamilyStem(ref.modelName)
            let curStem = Self.canonicalModelFamilyStem(config.modelName)
            if !refStem.isEmpty && refStem == curStem {
                return
            }
        }
        let targetAssets = try QWModelRuntimeAssets.load(packPath: ref.packPath)
        let targetDesc = targetAssets.descriptor
        guard let module = QWModuleRegistry.moduleType(for: targetDesc.family) else {
            throw QWEngineError.unsupportedModel(targetDesc.modelType)
        }
        let settingsURL = configPath.map { $0.canonicalFileURL } ?? QWSettingsManager.defaultSettingsFileURL
        let (effectiveSettings, _) = QWSettingsManager.shared.effectiveSettings(
            forModel: ref.modelName,
            packPath: ref.packPath,
            base: settings,
            from: settingsURL
        )
        let speculation = try module.resolveSpeculation(
            effectiveSettings.speculativeParameters, descriptor: targetDesc,
            environment: ProcessInfo.processInfo.environment, modelSwitch: true)
        let resolvedKVMode: QWKVMode = switch effectiveSettings.kvMode.lowercased() {
        case "q8": .q8
        case "q4": .q4
        default: .f16
        }
        let resolvedCacheConfig: QWCacheConfig = (config.cacheConfig.warmByteLimit > 0 || config.cacheConfig.ssdByteLimit > 0)
            ? QWCacheConfig(
                warmByteLimit: Int(effectiveSettings.warmRAMLimitGB * 1_073_741_824),
                ssdByteLimit: effectiveSettings.ssdCacheEnabled ? Int(effectiveSettings.ssdCacheLimitGB * 1_073_741_824) : 0,
                entryLimit: config.cacheConfig.entryLimit,
                cacheDirectory: config.cacheConfig.cacheDirectory,
                wipeOnStart: false
            )
            : config.cacheConfig
        var newConfig = Config(
            packPath: ref.packPath, modelName: ref.modelName, maxContext: effectiveSettings.maxContext,
            kvMode: resolvedKVMode,
            speculativeDecodingEnabled: effectiveSettings.speculativeDecodingEnabled,
            mtpDrafts: speculation.parameters.mtpDrafts, mtpBackend: speculation.parameters.mtpBackend,
            mtpRelaxedMargin: effectiveSettings.mtpRelaxedMargin,
            mtpRelaxedMinProb: effectiveSettings.mtpRelaxedMinProb,
            mtpDraftStrategy: effectiveSettings.mtpDraftStrategy,
            mtpQuadraticSeeding: effectiveSettings.mtpQuadraticSeeding,
            mtpAdaptiveDepth: effectiveSettings.mtpAdaptiveDepth,
            mtpAdaptiveEntropyThreshold: effectiveSettings.mtpEntropyThreshold,
            dflashDraftPackPath: effectiveSettings.dflashDraftPackPath,
            dflashSelectorTopK: effectiveSettings.dflashSelectorTopK,
            ropeScalingType: effectiveSettings.ropeScalingType,
            ropeScalingFactor: effectiveSettings.ropeScalingFactor,
            gdnPrefill: effectiveSettings.gdnPrefill,
            evictPrefillScratch: effectiveSettings.evictPrefillScratch,
            prefillChunkSize: effectiveSettings.prefillChunkSize,
            maxConcurrentSessions: effectiveSettings.maxConcurrentSessions,
            lsaConfig: effectiveSettings.lsaConfig,
            cacheConfig: resolvedCacheConfig,
            host: config.host, port: config.port,
            temperature: effectiveSettings.temperature,
            topP: effectiveSettings.topP,
            topK: effectiveSettings.topK,
            maxTokens: effectiveSettings.maxTokens,
            modelMaxLength: targetAssets.generationDefaults.maxNewTokens == nil
                ? targetAssets.generationDefaults.maxLength
                : nil,
            repetitionPenalty: effectiveSettings.repetitionPenalty,
            seed: effectiveSettings.seed,
            enableThinking: effectiveSettings.enableThinking,
            reasoningEffort: effectiveSettings.reasoningEffort,
            systemPrompt: effectiveSettings.systemPrompt,
            loadVision: effectiveSettings.loadVision,
            sourceSummary: config.sourceSummary
        )
        newConfig.engineTuning = effectiveSettings.resolvedEngineTuning
        logger.log(.info, "Switching active model to '\(ref.modelName)' (pack: '\(ref.packPath)')...")
        await waitForInFlightRequests()
        _ = try await unloadExplicit(waitForInFlight: false)
        self.activeMemoryPack = nil
        self.preMountKVMode = nil
        self.preMountMaxContext = nil
        self.statusBox.updateMountedMemoryPack(nil)
        self.config = newConfig
        self.statusBox.updateConfig(newConfig)
        transition(to: .unloaded)
    }

    /// Mounts a `<root>/.qwg/manifest.json` codebase/knowledge memory pack as a shared base prefix,
    /// enforcing its `kvMode` and required context capacity.
    @discardableResult
    public func mountMemoryPack(at rawPath: String) async throws -> QWEngineInfoResponse.MountedMemoryPackInfo {
        let trimmed = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw QWMemoryPackError.directoryNotFound(rawPath)
        }
        let startURL = URL(fileURLWithPath: trimmed.expandingTildePath).standardizedFileURL
        let rootURL = QWMemoryPack.discoverManifestRoot(walkingUpFrom: startURL) ?? startURL
        let manifest = try QWMemoryManifest.load(from: rootURL)

        var geometryDigest: String? = nil
        if case .loaded(let loaded) = state {
            geometryDigest = loaded.descriptor.geometryDigestV2Hex
        } else if !config.packPath.isEmpty,
                  let assets = try? QWModelRuntimeAssets.load(packPath: config.packPath) {
            geometryDigest = assets.descriptor.geometryDigestV2Hex
        }

        guard let artifact = manifest.matchingCache(
            modelName: config.modelName,
            geometryDigest: geometryDigest,
            kvMode: nil
        ) ?? manifest.caches.first else {
            throw QWMemoryPackError.noMatchingCacheArtifact(modelName: config.modelName, kvMode: nil)
        }
        let cacheURL = manifest.cacheFileURL(for: artifact, rootURL: rootURL)
        guard FileManager.default.fileExists(atPath: cacheURL.path) else {
            throw QWMemoryPackError.cacheFileMissing(cacheURL.path)
        }

        let enforcedKVMode = try QWCli.resolveKVMode(artifact.kvMode)
        let minMemoryContext = min(
            QWKVCache.maxTokensPerSeq,
            max(4096, ((artifact.tokenCount + 16384 + 255) / 256) * 256)
        )
        let requiredContext = max(config.maxContext, minMemoryContext)

        if config.kvMode != enforcedKVMode || config.maxContext < requiredContext {
            if preMountKVMode == nil { preMountKVMode = config.kvMode }
            if preMountMaxContext == nil { preMountMaxContext = config.maxContext }
            let updatedLSA = QWLSAConfig(
                mode: QWLSAMode.resolve(artifact.lsaMode) ?? config.lsaConfig.mode,
                budget: config.lsaConfig.budget,
                clusterBlock: manifest.clusterBlockTokens,
                sidecarPath: config.lsaConfig.sidecarPath
            )
            var updatedSources = config.sourceSummary
            updatedSources["kvMode"] = "memory-pack"
            let newConfig = config.updating(
                maxContext: requiredContext,
                kvMode: enforcedKVMode,
                lsaConfig: updatedLSA,
                sourceSummary: updatedSources
            )
            logger.log(.info, "[MemoryPack] Enforcing kvMode=\(artifact.kvMode) and maxContext=\(requiredContext) for '\(manifest.packName)'")
            _ = try await unloadExplicit(waitForInFlight: false)
            self.config = newConfig
            self.statusBox.updateConfig(newConfig)
        }

        let loaded = try await ensureLoaded()
        loaded.modelBackend.unmountSharedBasePrefixes()
        let mounted = try loaded.modelBackend.mountStandalonePrefixCache(from: cacheURL)
        let effectiveTokens: [Int]
        let effectiveBytes: Int
        if !mounted.tokens.isEmpty {
            effectiveTokens = mounted.tokens
            effectiveBytes = mounted.byteCount
        } else {
            effectiveTokens = Array(repeating: 1, count: max(1, artifact.tokenCount))
            effectiveBytes = artifact.cacheFileBytes
        }

        let cacheState = loaded.modelBackend.prefixCache
        QWEngineTelemetryRegistry.shared.updateCacheUsage(
            warmBytes: cacheState.warmBytes,
            warmLimit: config.cacheConfig.warmByteLimit,
            ssdBytes: cacheState.ssdBytes,
            ssdLimit: config.cacheConfig.ssdByteLimit
        )

        let delta = QWMemoryPack.computeDelta(for: manifest)
        let modCount = delta.modifiedFiles.count + delta.deletedFiles.count
        let info = QWEngineInfoResponse.MountedMemoryPackInfo(
            packName: manifest.packName,
            rootPath: rootURL.path,
            enforcedKVMode: artifact.kvMode,
            lsaMode: artifact.lsaMode,
            tokenCount: effectiveTokens.count,
            fileCount: manifest.files.count,
            modifiedFileCount: modCount,
            cacheBytes: effectiveBytes,
            enforcesKVMode: true
        )
        self.activeMemoryPack = ActiveMemoryPackState(
            rootURL: rootURL,
            manifest: manifest,
            artifact: artifact,
            cacheFileURL: cacheURL,
            tokens: effectiveTokens,
            byteCount: effectiveBytes
        )
        self.statusBox.updateMountedMemoryPack(info)
        logger.log(.info, String(
            format: "[MemoryPack] Mounted '%@' (%d tokens, %.2f MiB, kv=%@, lsa=%@, delta=%d files)",
            manifest.packName, effectiveTokens.count, Double(effectiveBytes) / 1_048_576.0,
            artifact.kvMode, artifact.lsaMode, modCount
        ))
        return info
    }

    /// Unmounts the active codebase/knowledge memory pack and unlocks `kvMode`.
    public func unmountMemoryPack() async {
        if case .loaded(let loaded) = state {
            loaded.modelBackend.unmountSharedBasePrefixes()
            let cacheState = loaded.modelBackend.prefixCache
            QWEngineTelemetryRegistry.shared.updateCacheUsage(
                warmBytes: cacheState.warmBytes,
                warmLimit: config.cacheConfig.warmByteLimit,
                ssdBytes: cacheState.ssdBytes,
                ssdLimit: config.cacheConfig.ssdByteLimit
            )
        }
        let oldName = activeMemoryPack?.manifest.packName
        activeMemoryPack = nil
        statusBox.updateMountedMemoryPack(nil)
        if let savedKV = preMountKVMode {
            let savedCtx = preMountMaxContext ?? config.maxContext
            preMountKVMode = nil
            preMountMaxContext = nil
            var updatedSources = config.sourceSummary
            updatedSources.removeValue(forKey: "kvMode")
            let restored = config.updating(
                maxContext: savedCtx,
                kvMode: savedKV,
                sourceSummary: updatedSources
            )
            self.config = restored
            self.statusBox.updateConfig(restored)
        }
        if let oldName {
            logger.log(.info, "[MemoryPack] Unmounted '\(oldName)' and unlocked kvMode")
        }
    }

    /// Auto-discovers a repository memory pack from `request.model` (`model@repo`) or
    /// `system`/`developer` messages and returns the `0 ..< N_mem` (+ live delta) token prefix.
    func autoMountAndPrepareMemoryPrefix(for request: ChatCompletionRequest) async -> [Int] {
        let state = await autoMountAndPrepareMemoryState(for: request)
        return state.mountedTokens + state.deltaTokens
    }

    /// Auto-discovers a repository memory pack and returns its mounted Cortex tokens, live delta tokens, and manifest.
    func autoMountAndPrepareMemoryState(
        for request: ChatCompletionRequest
    ) async -> (mountedTokens: [Int], deltaTokens: [Int], manifest: QWMemoryManifest?) {
        var discoveredRoot: URL? = nil
        if let rawModel = request.model,
           let atIdx = rawModel.firstIndex(of: "@") {
            let repoSpec = String(rawModel[rawModel.index(after: atIdx)...]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !repoSpec.isEmpty {
                let candidateURL = URL(fileURLWithPath: repoSpec.expandingTildePath).standardizedFileURL
                discoveredRoot = QWMemoryPack.discoverManifestRoot(walkingUpFrom: candidateURL)
            }
        }
        if discoveredRoot == nil {
            for msg in request.messages where msg.role == "system" || msg.role == "developer" {
                let text = msg.extractTextContent()
                if let root = QWMemoryPack.discoverManifestRoot(inSystemPrompt: text) {
                    discoveredRoot = root
                    break
                }
            }
        }
        if let root = discoveredRoot,
           activeMemoryPack?.rootURL.standardizedFileURL != root.standardizedFileURL {
            _ = try? await mountMemoryPack(at: root.path)
        }
        guard let active = activeMemoryPack else {
            return ([], [], nil)
        }
        let delta = QWMemoryPack.computeDelta(for: active.manifest)
        let modCount = delta.modifiedFiles.count + delta.deletedFiles.count
        if let curInfo = statusBox.getMountedMemoryPackInfo(), curInfo.modifiedFileCount != modCount {
            statusBox.updateMountedMemoryPack(QWEngineInfoResponse.MountedMemoryPackInfo(
                packName: curInfo.packName,
                rootPath: curInfo.rootPath,
                enforcedKVMode: curInfo.enforcedKVMode,
                lsaMode: curInfo.lsaMode,
                tokenCount: curInfo.tokenCount,
                fileCount: curInfo.fileCount,
                modifiedFileCount: modCount,
                cacheBytes: curInfo.cacheBytes,
                enforcesKVMode: curInfo.enforcesKVMode
            ))
        }
        // Live working-tree delta is injected at the active conversation tail in QWChatTurnPreparer
        // so file edits never invalidate the 0 ..< N_mem + N_history KV prefix cache.
        return (active.tokens, [], active.manifest)
    }
}
