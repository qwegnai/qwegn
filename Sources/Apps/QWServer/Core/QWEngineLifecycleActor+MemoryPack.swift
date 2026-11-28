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
        speculativeDecodingEnabled: Bool? = nil,
        mtpDrafts: Int? = nil,
        mtpBackend: String? = nil,
        dflashDraftPackPath: String? = nil,
        lsaConfig: QWLSAConfig? = nil,
        sourceSummary: [String: String]? = nil
    ) -> QWEngineLifecycleActor.Config {
        var copy = QWEngineLifecycleActor.Config(
            packPath: packPath,
            modelName: modelName,
            maxContext: maxContext ?? self.maxContext,
            kvMode: kvMode ?? self.kvMode,
            speculativeDecodingEnabled: speculativeDecodingEnabled ?? self.speculativeDecodingEnabled,
            mtpDrafts: mtpDrafts ?? self.mtpDrafts,
            mtpBackend: mtpBackend ?? self.mtpBackend,
            mtpRelaxedMargin: mtpRelaxedMargin,
            mtpRelaxedMinProb: mtpRelaxedMinProb,
            mtpDraftStrategy: mtpDraftStrategy,
            mtpQuadraticSeeding: mtpQuadraticSeeding,
            mtpAdaptiveDepth: mtpAdaptiveDepth,
            mtpAdaptiveEntropyThreshold: mtpAdaptiveEntropyThreshold,
            dflashDraftPackPath: dflashDraftPackPath ?? self.dflashDraftPackPath,
            dflashSelectorTopK: dflashSelectorTopK,
            dsparkConfidenceSchedule: dsparkConfidenceSchedule,
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
        copy.pinModel = pinModel
        copy.allowLocalImagePaths = allowLocalImagePaths
        copy.allowRemoteImageURLs = allowRemoteImageURLs
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
            dsparkConfidenceSchedule: dsparkConfidenceSchedule,
            gdnPrefill: gdnPrefill,
            lsaConfig: lsaConfig,
            maxConcurrentSessions: maxConcurrentSessions,
            tuning: engineTuning
        )
    }
}

public enum QWModelSwapRefusedError: Error, CustomStringConvertible, Sendable, Equatable {
    case pinned(activeModel: String, requestedModel: String)
    case busyInFlight(activeModel: String, requestedModel: String)

    public var errorCode: String {
        switch self {
        case .pinned: return "model_pinned"
        case .busyInFlight: return "model_busy_in_flight"
        }
    }

    public var description: String {
        switch self {
        case .pinned(let active, let requested):
            return "Primary model slot is pinned to '\(active)'; implicit switch to '\(requested)' refused. Use '\(active)' or call POST /v1/models/load explicitly."
        case .busyInFlight(let active, let requested):
            return "Active inference is running on '\(active)'; concurrent implicit switch to '\(requested)' refused."
        }
    }
}

extension QWEngineLifecycleActor {

    /// Configures explicit allowed workspace roots for `/v1/memory/*` and `metadata.workspace_path` mounting.
    nonisolated public func configureAllowedWorkspaceRoots(_ roots: [String]) {
        QWWorkspacePolicyRegistry.shared.setAllowedRoots(roots, for: self)
    }

    nonisolated func effectiveAllowedWorkspaceRoots(override: [String] = []) -> [String] {
        QWWorkspacePolicyRegistry.shared.allowedRoots(for: self) + override
    }

    nonisolated func resolveValidatedMemoryRootURL(at rawPath: String, overrideRoots: [String] = []) throws -> URL {
        let configured = effectiveAllowedWorkspaceRoots(override: overrideRoots)
        let canonicalStart = try QWHttpSecurity.validateAllowedWorkspacePath(rawPath, configuredRoots: configured)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: canonicalStart.path, isDirectory: &isDir), isDir.boolValue else {
            throw QWMemoryPackError.directoryNotFound(canonicalStart.path)
        }
        if let discovered = QWMemoryPack.discoverManifestRoot(walkingUpFrom: canonicalStart) {
            let canonicalDiscovered = discovered.resolvingSymlinksInPath().standardizedFileURL
            let allowed = QWHttpSecurity.resolvedAllowedWorkspaceRootURLs(configuredRoots: configured)
            guard allowed.contains(where: { QWHttpSecurity.canonicalDirectoryContainment(canonicalDiscovered, inside: $0) }) else {
                throw QWMemoryWorkspaceAccessError(
                    "Discovered manifest root '\(canonicalDiscovered.path)' escapes allowed workspace roots"
                )
            }
            return canonicalDiscovered
        }
        return canonicalStart
    }

    /// Strips `<org>/` prefix and trailing quantization/packaging tags (`-QWP_4`, `-QWP-Mixed-Fast`, `-hq8g32-hq4g64`, `-mlx-serve-mixed-4-8bit`, etc.).
    public static func canonicalModelFamilyStem(_ raw: String) -> String {
        let leaf = (raw.split(separator: "/").last.map(String.init) ?? raw)
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard !leaf.isEmpty else { return "" }
        let parts = leaf.split(separator: "-").map(String.init)
        var end = parts.count
        while end > 1 {
            let p = parts[end - 1]
            let isQuantTag = p.hasPrefix("qwp") || p.hasPrefix("shvq")
                || p.hasPrefix("hq") || p.hasPrefix("aq")
                || p == "mlx" || p == "serve" || p == "mixed" || p == "fast"
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
    public func selectModelIfNeeded(
        _ requestedModel: String?,
        allowSameFamilyQuantSwitch: Bool = true,
        isExplicitLoadRoute: Bool = false
    ) async throws {
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
        let curCanonical = URL(fileURLWithPath: config.packPath.expandingTildePath)
            .resolvingSymlinksInPath().standardizedFileURL.path
        let allowOptIn = ProcessInfo.processInfo.environment["QWG_ALLOW_UNREGISTERED_MODEL_PATHS"] == "1"
        if !allowOptIn && QWServerSettings.isArbitraryFilesystemModelPath(req) {
            let reqCanonical = URL(fileURLWithPath: req.expandingTildePath)
                .resolvingSymlinksInPath().standardizedFileURL
            if reqCanonical.path == curCanonical {
                return
            }
            let inRegisteredRoot = QWHttpSecurity.canonicalDirectoryContainment(
                reqCanonical, inside: QWPaths.modelsDirectoryURL
            )
            if !allowSameFamilyQuantSwitch || !inRegisteredRoot {
                throw QWMetadataValidationError(
                    "Filesystem model path '\(req)' is not permitted; use a registered model under ~/.qwg/models"
                )
            }
        }
        if !allowSameFamilyQuantSwitch, !config.packPath.isEmpty {
            let reqStem = Self.canonicalModelFamilyStem(req)
            let curStem = Self.canonicalModelFamilyStem(config.modelName)
            let aliasStem = Self.canonicalModelFamilyStem(activeAlias)
            if !reqStem.isEmpty && (reqStem == curStem || (!aliasStem.isEmpty && reqStem == aliasStem)) {
                return
            }
        }
        let ref: (packPath: String, modelName: String, alias: String)
        if allowOptIn {
            ref = settings.resolveModelReference(req, allowArbitraryFilesystemPaths: true)
        } else if allowSameFamilyQuantSwitch && QWServerSettings.isArbitraryFilesystemModelPath(req) {
            let reqCanonical = URL(fileURLWithPath: req.expandingTildePath)
                .resolvingSymlinksInPath().standardizedFileURL
            let discovered = QWServerSettings.scanModelsDirectory()
            if let match = discovered.first(where: {
                !$0.packPath.isEmpty &&
                URL(fileURLWithPath: $0.packPath).resolvingSymlinksInPath().standardizedFileURL.path == reqCanonical.path
            }) {
                ref = (match.packPath, match.name, settings.alias(for: match.name))
            } else {
                let resolved = QWModelRoot.resolveSnapshotDirectory(reqCanonical)
                ref = (resolved.path, QWModelRoot.canonicalModelName(for: resolved), "")
            }
        } else {
            ref = settings.resolveModelReference(req, allowArbitraryFilesystemPaths: false)
        }
        let refCanonical = URL(fileURLWithPath: ref.packPath).resolvingSymlinksInPath().standardizedFileURL.path
        if !allowOptIn && !ref.packPath.isEmpty && refCanonical != curCanonical {
            guard QWHttpSecurity.canonicalDirectoryContainment(
                URL(fileURLWithPath: refCanonical), inside: QWPaths.modelsDirectoryURL
            ) else {
                return
            }
        }
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
        let catalogEntry = QWModelCatalog.classify(name: ref.modelName, path: ref.packPath)
        if catalogEntry.isCoreAIBundle {
            throw QWMetadataValidationError(
                "Model '\(ref.modelName)' is a System-1 decision/embedding model, not a chat completion model."
            )
        }
        let targetRoot = QWModelRoot(url: URL(fileURLWithPath: ref.packPath))
        let quickDesc: QWModelDescriptor? = targetRoot.configURL
            .flatMap { try? Data(contentsOf: $0) }
            .flatMap { try? QWModelDescriptor.decode(configJSON: $0) }
        if let desc = quickDesc {
            if Self.isNonCausalDecisionDescriptor(desc)
                || (!isExplicitLoadRoute && (desc.decisionConfig != nil || catalogEntry.isDecisionCapable)) {
                throw QWMetadataValidationError(
                    "Model '\(ref.modelName)' is a System-1 decision/embedding model, not a chat completion model."
                )
            }
        } else if !isExplicitLoadRoute && catalogEntry.isDecisionCapable {
            throw QWMetadataValidationError(
                "Model '\(ref.modelName)' is a System-1 decision/embedding model, not a chat completion model."
            )
        }
        if !isExplicitLoadRoute, !config.packPath.isEmpty {
            if config.pinModel {
                throw QWModelSwapRefusedError.pinned(activeModel: config.modelName, requestedModel: ref.modelName)
            }
            if inFlightRequests > 0 || modelTransitionActive {
                throw QWModelSwapRefusedError.busyInFlight(activeModel: config.modelName, requestedModel: ref.modelName)
            }
        }
        let targetAssets = try QWModelRuntimeAssets.load(packPath: ref.packPath)
        let targetDesc = targetAssets.descriptor
        if Self.isNonCausalDecisionDescriptor(targetDesc)
            || (!isExplicitLoadRoute && (targetDesc.decisionConfig != nil || catalogEntry.isDecisionCapable)) {
            throw QWMetadataValidationError(
                "Model '\(ref.modelName)' is a System-1 decision/embedding model, not a chat completion model."
            )
        }
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
        newConfig.pinModel = config.pinModel
        newConfig.allowLocalImagePaths = config.allowLocalImagePaths
        newConfig.allowRemoteImageURLs = config.allowRemoteImageURLs
        newConfig.engineTuning = effectiveSettings.resolvedEngineTuning
        logger.log(.info, "Switching active model to '\(ref.modelName)' (pack: '\(ref.packPath)')...")
        try await withExclusiveTransition {
            let curAfterWait = URL(fileURLWithPath: config.packPath.expandingTildePath)
                .resolvingSymlinksInPath().standardizedFileURL.path
            guard refCanonical != curAfterWait else { return }
            await waitForInFlightRequests()
            _ = try await unloadModel(waitForInFlight: false, forReconfigure: true)
            self.activeMemoryPack = nil
            self.preMountKVMode = nil
            self.preMountMaxContext = nil
            self.vaultSetBox.unmountPrimaryOnly()
            self.statusBox.updateMountedMemoryPack(nil)
            self.config = newConfig
            self.configRevision &+= 1
            self.statusBox.updateConfig(newConfig)
            transition(to: .unloaded)
        }
    }

    /// Mounts a `<root>/.qwg/manifest.json` codebase/knowledge memory pack as a shared base prefix,
    /// enforcing its `kvMode` and required context capacity.
    @discardableResult
    public func mountMemoryPack(
        at rawPath: String,
        waitForInFlight: Bool = true,
        skipIfAlreadyMounted: Bool = false
    ) async throws -> QWEngineInfoResponse.MountedMemoryPackInfo {
        try await withExclusiveTransition {
            try await self.mountMemoryPackHoldingTransition(
                at: rawPath,
                waitForInFlight: waitForInFlight,
                skipIfAlreadyMounted: skipIfAlreadyMounted
            )
        }
    }

    func mountMemoryPackHoldingTransition(
        at rawPath: String,
        waitForInFlight: Bool,
        skipIfAlreadyMounted: Bool
    ) async throws -> QWEngineInfoResponse.MountedMemoryPackInfo {
        let rootURL = try resolveValidatedMemoryRootURL(at: rawPath)
        if skipIfAlreadyMounted,
           let cur = activeMemoryPack,
           cur.rootURL.standardizedFileURL == rootURL.standardizedFileURL,
           let curInfo = statusBox.getMountedMemoryPackInfo() {
            return curInfo
        }
        let manifest = try QWMemoryManifest.load(from: rootURL)

        var geometryDigest: String? = nil
        if case .loaded(let loaded) = state {
            geometryDigest = loaded.descriptor.geometryDigestV3Hex
        } else if !config.packPath.isEmpty,
                  let assets = try? QWModelRuntimeAssets.load(packPath: config.packPath) {
            geometryDigest = assets.descriptor.geometryDigestV3Hex
        }

        guard let artifact = manifest.matchingCache(
            modelName: config.modelName,
            geometryDigest: geometryDigest,
            kvMode: nil,
            allowFallbackToAnyModel: false
        ) else {
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

        if waitForInFlight {
            await waitForInFlightRequests()
        }
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
            _ = try await unloadModel(waitForInFlight: false, forReconfigure: true)
            self.config = newConfig
            self.configRevision &+= 1
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

        let deltaCache = self.memoryDeltaCache
        let rootPath = rootURL.path
        let modCount = await Task.detached(priority: .utility) {
            deltaCache.invalidate()
            return deltaCache.cachedOrCompute(for: manifest, rootPath: rootPath)
        }.value
        let info = QWEngineInfoResponse.MountedMemoryPackInfo(
            packName: manifest.packName,
            rootPath: rootPath,
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
        self.vaultSetBox.upsert(QWMountedVaultSet.PackMount(
            packName: manifest.packName,
            rootURL: rootURL,
            manifest: manifest,
            isPrimaryCortex: true
        ))
        self.statusBox.updateMountedMemoryPack(info)
        logger.log(.info, String(
            format: "[MemoryPack] Mounted '%@' (%d tokens, %.2f MiB, kv=%@, lsa=%@, delta=%d files)",
            manifest.packName, effectiveTokens.count, Double(effectiveBytes) / 1_048_576.0,
            artifact.kvMode, artifact.lsaMode, modCount
        ))
        return info
    }

    /// Unmounts the active codebase/knowledge memory pack (or a named auxiliary pack) and unlocks `kvMode`.
    public func unmountMemoryPack(packIdentifier: String? = nil) async {
        if let rawId = packIdentifier?.trimmingCharacters(in: .whitespacesAndNewlines),
           !rawId.isEmpty, rawId.lowercased() != "all" {
            let isPrimaryMatch: Bool = if let active = activeMemoryPack {
                active.manifest.packName.caseInsensitiveCompare(rawId) == .orderedSame
                    || active.rootURL.standardizedFileURL.path == URL(fileURLWithPath: rawId.expandingTildePath).standardizedFileURL.path
            } else {
                false
            }
            if !isPrimaryMatch {
                _ = vaultSetBox.unmount(matching: rawId)
                return
            }
        }
        await withExclusiveTransition {
            await waitForInFlightRequests()
            memoryDeltaCache.invalidate()
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
            if packIdentifier == nil || packIdentifier?.lowercased() == "all" {
                vaultSetBox.unmountAll()
            } else {
                vaultSetBox.unmountPrimaryOnly()
            }
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
                self.configRevision &+= 1
                self.statusBox.updateConfig(restored)
            }
            if let oldName {
                logger.log(.info, "[MemoryPack] Unmounted '\(oldName)' and unlocked kvMode")
            }
        }
    }

    /// Auto-discovers a repository memory pack and returns its mounted Cortex tokens, live delta tokens, and manifest.
    nonisolated func autoMountAndPrepareMemoryState(
        for request: ChatCompletionRequest
    ) async -> (mountedTokens: [Int], deltaTokens: [Int], manifest: QWMemoryManifest?) {
        guard request.effectiveUseMemoryPack else {
            return ([], [], nil)
        }
        var discoveredRoot: URL? = nil
        if let explicitWs = request.effectiveWorkspacePath,
           let root = try? resolveValidatedMemoryRootURL(at: explicitWs),
           FileManager.default.fileExists(atPath: QWMemoryManifest.manifestURL(in: root).path) {
            discoveredRoot = root
        }
        if discoveredRoot == nil,
           let rawModel = request.model,
           let atIdx = rawModel.firstIndex(of: "@") {
            let repoSpec = String(rawModel[rawModel.index(after: atIdx)...]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !repoSpec.isEmpty,
               let root = try? resolveValidatedMemoryRootURL(at: repoSpec),
               FileManager.default.fileExists(atPath: QWMemoryManifest.manifestURL(in: root).path) {
                discoveredRoot = root
            }
        }
        if discoveredRoot == nil && statusBox.getMountedMemoryPackInfo() == nil {
            return ([], [], nil)
        }
        let active = await resolveActiveMemoryPack(mountingIfNeeded: discoveredRoot)
        guard let active else {
            return ([], [], nil)
        }
        let deltaCache = self.memoryDeltaCache
        let modCount = await Task.detached(priority: .utility) {
            deltaCache.cachedOrCompute(for: active.manifest, rootPath: active.rootURL.path)
        }.value
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

    private func resolveActiveMemoryPack(mountingIfNeeded discoveredRoot: URL?) async -> ActiveMemoryPackState? {
        if let root = discoveredRoot,
           activeMemoryPack?.rootURL.standardizedFileURL != root.standardizedFileURL {
            _ = try? await mountMemoryPack(at: root.path, waitForInFlight: true, skipIfAlreadyMounted: true)
        }
        return activeMemoryPack
    }

    static func isNonCausalDecisionDescriptor(_ desc: QWModelDescriptor) -> Bool {
        if desc.role == .decisionEncoder { return true }
        guard let dc = desc.decisionConfig else { return false }
        if case .bidirectionalEncoder = dc.topology { return true }
        switch dc.head {
        case .vocabVerbalizer:
            return false
        default:
            return true
        }
    }
}
