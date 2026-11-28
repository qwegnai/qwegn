// QWEngineLifecycleActor+MemoryIndex.swift — Live workspace inspection and streaming `.qwcache` indexing.
import Foundation
import QWCli
import QWCore
import QWEngine
import QWKV
import QWToken

extension QWEngineLifecycleActor {
    /// Renders a memory system envelope through `chatTemplate`, slicing before the trailing sentinel turn.
    public static func renderMemoryEnvelopeWire(
        systemContent: String,
        chatTemplate: QWChatTemplate
    ) throws -> String {
        var opts = QWChatTemplateOptions()
        opts.addGenerationPrompt = false
        opts.enableThinking = .bool(false)
        let sentinel = "__QWG_MEMORY_USER_SENTINEL__"
        let sysMsg: QWJson = .object([
            ("role", .string("system")),
            ("content", .string(systemContent))
        ])
        let userMsg: QWJson = .object([
            ("role", .string("user")),
            ("content", .string(sentinel))
        ])
        let fullWire = try chatTemplate.render(messages: [sysMsg, userMsg], options: opts)
        return QWMemoryPack.sliceBeforeSentinelUserTurn(fullWire: fullWire, sentinel: sentinel)
    }

    /// Inspects a workspace folder for `.qwg/manifest.json`, matching `.qwcache` artifacts, git churn, and live deltas.
    public func inspectMemoryPack(at rawPath: String) throws -> QWMemoryInspectResponse {
        let rootURL = try resolveInspectionRootURL(at: rawPath)
        let manifest = try? QWMemoryManifest.load(from: rootURL)
        let activeDigest = resolveActiveGeometryDigest()
        let activeKV = String(describing: config.kvMode)
        let activeLSA = resolveDefaultLSAModeName()
        let includeTests = manifest?.includeTests ?? false
        let discovered = (try? QWMemoryPack.discoverFiles(in: rootURL, includeTests: includeTests)) ?? []
        let gitInfo = QWMemoryPack.evaluateGitStaleness(manifestSHA: manifest?.gitHeadSHA, in: rootURL)

        let discoveredModels = QWServerSettings.scanModelsDirectory().filter { $0.supported && !$0.packPath.isEmpty }
        let settings = QWSettingsManager.shared.loadSettings()

        var availability: [QWMemoryInspectResponse.ArtifactAvailability] = []
        if let manifest {
            for art in manifest.caches {
                let exists = FileManager.default.fileExists(atPath: manifest.cacheFileURL(for: art, rootURL: rootURL).path)
                let matchesActive = artifactMatchesActiveModel(art, activeDigest: activeDigest)
                let installedName = resolveInstalledModelName(
                    for: art.modelName,
                    matchesActive: matchesActive,
                    discoveredModels: discoveredModels,
                    settings: settings
                )
                availability.append(QWMemoryInspectResponse.ArtifactAvailability(
                    artifact: art,
                    cacheFileExists: exists,
                    modelInstalled: installedName != nil,
                    installedModelName: installedName,
                    matchesActiveModel: matchesActive
                ))
            }
        }

        let validMatch: QWMemoryManifest.CacheArtifact?
        if let manifest,
           let matched = manifest.matchingCache(
               modelName: config.modelName,
               geometryDigest: activeDigest,
               kvMode: nil,
               allowFallbackToAnyModel: false
           ),
           FileManager.default.fileExists(atPath: manifest.cacheFileURL(for: matched, rootURL: rootURL).path) {
            validMatch = matched
        } else {
            validMatch = nil
        }

        let delta = manifest.map { QWMemoryPack.computeDelta(for: $0) }
        let modifiedFiles = delta?.modifiedFiles ?? []
        let deletedFiles = delta?.deletedFiles ?? []
        let hasAnyExistingFile = availability.contains { $0.cacheFileExists }
        let isStale = gitInfo.commitsBehind > 0 || gitInfo.diverged || (delta?.hasChanges ?? false)

        let status: String
        if validMatch != nil {
            status = isStale ? "stale" : "ready"
        } else if hasAnyExistingFile {
            status = "wrong_model"
        } else {
            status = "missing"
        }

        let recommendedScope: String
        if let manifest, manifest.files.allSatisfy(\.inCortex),
           (validMatch?.tokenCount ?? 0) > QWMemoryVault.defaultCortexTokenBudget {
            recommendedScope = "full_prefix"
        } else {
            recommendedScope = "cortex_vault"
        }

        let rec = QWMemoryInspectResponse.RecommendedIndexOptions(
            kvMode: validMatch?.kvMode ?? activeKV,
            lsaMode: validMatch?.lsaMode ?? activeLSA,
            scope: recommendedScope,
            includeTests: includeTests
        )

        return QWMemoryInspectResponse(
            status: status,
            rootPath: rootURL.path,
            packName: manifest?.packName ?? rootURL.lastPathComponent,
            manifestExists: manifest != nil,
            activeModelName: config.modelName,
            activeGeometryDigest: activeDigest,
            activeKVMode: activeKV,
            activeLSAMode: activeLSA,
            matchingArtifact: validMatch,
            availableArtifacts: availability,
            gitHeadSHA: gitInfo.headSHA,
            manifestGitHeadSHA: manifest?.gitHeadSHA,
            commitsBehind: gitInfo.commitsBehind,
            gitDiverged: gitInfo.diverged,
            modifiedFiles: modifiedFiles,
            deletedFiles: deletedFiles,
            candidateFileCount: discovered.count,
            recommendedOptions: rec
        )
    }

    /// Indexes a workspace folder into `<root>/.qwg/manifest.json` + `.qwcache` inside the live daemon, streaming progress.
    public func indexMemoryPack(
        request: QWMemoryIndexRequest,
        onProgress: (@Sendable (QWMemoryIndexProgressEvent) -> Void)? = nil
    ) async throws -> QWMemoryIndexResult {
        let indexStart = Date()
        let rootURL = try resolveInspectionRootURL(at: request.path)
        try await selectModelIfNeeded(request.model)
        return try await withExclusiveTransition {
            await waitForInFlightRequests()
            defer { memoryDeltaCache.invalidate() }

        let existingManifest = try? QWMemoryManifest.load(from: rootURL)
        let reqPackName = request.packName?.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedPackName = (reqPackName?.isEmpty == false ? reqPackName : nil)
            ?? existingManifest?.packName
            ?? rootURL.lastPathComponent

        let reqKV = request.kvMode?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let kvModeName = (reqKV?.isEmpty == false ? reqKV : nil) ?? String(describing: config.kvMode)
        let requestedKVMode = try QWCli.resolveKVMode(kvModeName)

        let reqLSA = request.lsaMode?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let resolvedLSAStr = (reqLSA?.isEmpty == false ? reqLSA : nil) ?? resolveDefaultLSAModeName()
        guard let resolvedLSAMode = QWLSAMode.resolve(resolvedLSAStr) else {
            throw QWEngineError.badConfig("Invalid LSA mode '\(resolvedLSAStr)'")
        }

        let includeTests = request.includeTests ?? existingManifest?.includeTests ?? false
        let clusterBlock = max(64, existingManifest?.clusterBlockTokens ?? (config.lsaConfig.clusterBlock > 0 ? config.lsaConfig.clusterBlock : 512))

        onProgress?(QWMemoryIndexProgressEvent(
            phase: "discovering",
            progressFraction: 0.02,
            message: "Scanning repository '\(resolvedPackName)'..."
        ))
        let files = try QWMemoryPack.discoverFiles(in: rootURL, includeTests: includeTests)
        let gitSHA = QWMemoryPack.currentGitHeadSHA(in: rootURL)
        onProgress?(QWMemoryIndexProgressEvent(
            phase: "discovering",
            totalFiles: files.count,
            progressFraction: 0.05,
            message: "Discovered \(files.count) files in '\(resolvedPackName)'."
        ))

        let tokContext = try await resolveTokenizerContext()
        let hardCeiling = QWKVCache.maxTokensPerSeq
        let defaultBudgetCeiling = max(4096, hardCeiling - 8192)
        let scope = request.scope?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? "cortex_vault"
        let effectiveMaxTokens: Int
        if let explicitMax = request.maxCortexTokens, explicitMax > 0 {
            effectiveMaxTokens = min(explicitMax, defaultBudgetCeiling)
        } else if scope == "full_prefix" {
            effectiveMaxTokens = defaultBudgetCeiling
        } else {
            effectiveMaxTokens = min(QWMemoryVault.defaultCortexTokenBudget, defaultBudgetCeiling)
        }

        let progressStep = files.count <= 500 ? 1 : max(1, files.count / 250)
        let prepared = try await prepareMemoryTokensOffActor(
            packName: resolvedPackName,
            rootPath: rootURL.path,
            files: files,
            clusterBlock: clusterBlock,
            kvModeName: kvModeName,
            maxTokens: effectiveMaxTokens,
            tokenizer: tokContext.tokenizer,
            chatTemplate: tokContext.chatTemplate
        ) { update in
            guard update.processedFiles == 1 || update.processedFiles == update.totalFiles || (update.processedFiles % progressStep == 0) else {
                return
            }
            let frac = 0.05 + 0.50 * (Double(update.processedFiles) / Double(max(1, update.totalFiles)))
            onProgress?(QWMemoryIndexProgressEvent(
                phase: "tokenizing",
                processedFiles: update.processedFiles,
                totalFiles: update.totalFiles,
                currentFile: update.currentFile,
                cortexTokens: update.tokenizedTokens,
                progressFraction: frac,
                message: "Indexing \(update.processedFiles)/\(update.totalFiles): \(update.currentFile)"
            ))
        }

        let built = prepared.built
        let memoryTokens = prepared.tokens
        let mountAfterIndex = request.mountAfterIndex ?? true
        let requiredCapacity = min(hardCeiling, max(4096, ((memoryTokens.count + 4096 + 255) / 256) * 256))
        let sidecarURL = QWLSAConfig.resolveSidecarURL(
            packPath: config.packPath,
            explicitSidecarPath: config.lsaConfig.sidecarPath.isEmpty ? nil : config.lsaConfig.sidecarPath
        )
        let updatedLSA = QWLSAConfig(
            mode: resolvedLSAMode,
            budget: config.lsaConfig.budget,
            clusterBlock: clusterBlock,
            sidecarPath: sidecarURL?.path ?? config.lsaConfig.sidecarPath
        )

        let reqBackendRaw = request.mtpBackend?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let reqDraftPackRaw = request.dflashDraftPackPath?.trimmingCharacters(in: .whitespacesAndNewlines)
        let reqDraftPack = (reqDraftPackRaw?.isEmpty == false) ? reqDraftPackRaw : nil
        let requestedBackend: String? = (reqBackendRaw?.isEmpty == false)
            ? reqBackendRaw
            : reqDraftPack.map { QWServerSettings.inferDrafterBackendAndMaxDrafts(packPath: $0).backend }
        if let requestedBackend {
            _ = try QWMTPBackend.resolve(name: requestedBackend)
        }
        let targetBackend = requestedBackend ?? config.mtpBackend
        let targetDraftPack = reqDraftPack ?? config.dflashDraftPackPath
        let targetIsDFlash = QWServerSettings.normalizeExternalDrafterBackend(targetBackend) != nil
        let targetSpecEnabled = (requestedBackend != nil && targetIsDFlash) ? true : config.speculativeDecodingEnabled
        let targetDrafts = (requestedBackend != nil && targetIsDFlash) ? max(1, config.mtpDrafts) : config.mtpDrafts

        let savedPreConfig = config
        let needsConfigReload = config.kvMode != requestedKVMode
            || config.maxContext < requiredCapacity
            || config.lsaConfig.mode != resolvedLSAMode
            || targetBackend != config.mtpBackend
            || targetDraftPack != config.dflashDraftPackPath
            || targetSpecEnabled != config.speculativeDecodingEnabled
            || targetDrafts != config.mtpDrafts

        if needsConfigReload {
            if mountAfterIndex {
                if preMountKVMode == nil { preMountKVMode = config.kvMode }
                if preMountMaxContext == nil { preMountMaxContext = config.maxContext }
            }
            onProgress?(QWMemoryIndexProgressEvent(
                phase: "loading_model",
                processedFiles: files.count,
                totalFiles: files.count,
                cortexFiles: built.cortexFileCount,
                vaultFiles: built.vaultFileCount,
                cortexTokens: memoryTokens.count,
                prefillTotalTokens: memoryTokens.count,
                progressFraction: 0.56,
                message: "Reconfiguring \(config.modelName) (kv=\(kvModeName), lsa=\(resolvedLSAMode.rawValue), ctx=\(requiredCapacity))..."
            ))
            var updatedSources = config.sourceSummary
            if mountAfterIndex { updatedSources["kvMode"] = "memory-pack" }
            let indexConfig = config.updating(
                maxContext: max(config.maxContext, requiredCapacity),
                kvMode: requestedKVMode,
                speculativeDecodingEnabled: targetSpecEnabled,
                mtpDrafts: targetDrafts,
                mtpBackend: targetBackend,
                dflashDraftPackPath: targetDraftPack,
                lsaConfig: updatedLSA,
                sourceSummary: updatedSources
            )
            _ = try await unloadModel(waitForInFlight: false, forReconfigure: true)
            self.config = indexConfig
            self.configRevision &+= 1
            self.statusBox.updateConfig(indexConfig)
        } else if case .unloaded = state {
            onProgress?(QWMemoryIndexProgressEvent(
                phase: "loading_model",
                processedFiles: files.count,
                totalFiles: files.count,
                cortexFiles: built.cortexFileCount,
                vaultFiles: built.vaultFileCount,
                cortexTokens: memoryTokens.count,
                prefillTotalTokens: memoryTokens.count,
                progressFraction: 0.56,
                message: "Loading \(config.modelName) weights..."
            ))
        }

        let loaded = try await ensureLoaded()
        let cacheFileName = "\(QWMemoryPack.safeSlug(resolvedPackName))_\(QWMemoryPack.safeSlug(config.modelName))_\(kvModeName).qwcache"
        let qwgDir = QWMemoryManifest.qwgDirectoryURL(in: rootURL)
        try FileManager.default.createDirectory(at: qwgDir, withIntermediateDirectories: true)
        QWMemoryPack.ensureGitExclude(in: rootURL)
        let cacheFileURL = qwgDir.appendingPathComponent(cacheFileName)

        let telemetry = QWEngineTelemetryRegistry.shared
        telemetry.beginPrefill(promptTokens: memoryTokens.count)
        defer { telemetry.resetToIdle() }

        onProgress?(QWMemoryIndexProgressEvent(
            phase: "prefilling",
            processedFiles: files.count,
            totalFiles: files.count,
            cortexFiles: built.cortexFileCount,
            vaultFiles: built.vaultFileCount,
            cortexTokens: memoryTokens.count,
            prefillProcessedTokens: 0,
            prefillTotalTokens: memoryTokens.count,
            progressFraction: 0.58,
            message: "Prefilling \(memoryTokens.count) Cortex tokens on Metal..."
        ))

        let totalPrefillTokens = memoryTokens.count
        let builtFilesTotal = files.count
        let cortexCount = built.cortexFileCount
        let vaultCount = built.vaultFileCount
        let backend = loaded.modelBackend

        let writtenBytes: Int = try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let bytes = try backend.buildStandalonePrefixCache(
                        tokens: memoryTokens,
                        outputURL: cacheFileURL,
                        pinAsSharedBase: false
                    ) { done, total in
                        let effTotal = max(1, max(total, totalPrefillTokens))
                        telemetry.updatePrefillProgress(completedTokens: done, totalTokens: effTotal)
                        let frac = 0.58 + 0.36 * min(1.0, Double(done) / Double(effTotal))
                        onProgress?(QWMemoryIndexProgressEvent(
                            phase: "prefilling",
                            processedFiles: builtFilesTotal,
                            totalFiles: builtFilesTotal,
                            cortexFiles: cortexCount,
                            vaultFiles: vaultCount,
                            cortexTokens: totalPrefillTokens,
                            prefillProcessedTokens: done,
                            prefillTotalTokens: effTotal,
                            progressFraction: frac,
                            message: "Prefilling KV/LSA state (\(done)/\(effTotal) tokens)..."
                        ))
                    }
                    cont.resume(returning: bytes)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }

        onProgress?(QWMemoryIndexProgressEvent(
            phase: "saving",
            processedFiles: builtFilesTotal,
            totalFiles: builtFilesTotal,
            cortexFiles: cortexCount,
            vaultFiles: vaultCount,
            cortexTokens: totalPrefillTokens,
            prefillProcessedTokens: totalPrefillTokens,
            prefillTotalTokens: totalPrefillTokens,
            progressFraction: 0.95,
            message: "Writing .qwg/manifest.json & search index..."
        ))

        var manifest = existingManifest ?? QWMemoryManifest(packName: resolvedPackName, rootPath: rootURL.path)
        manifest.packName = resolvedPackName
        manifest.rootPath = rootURL.path
        manifest.gitHeadSHA = gitSHA
        manifest.clusterBlockTokens = clusterBlock
        manifest.includeTests = includeTests
        manifest.files = built.fileEntries
        let vaultEntries = built.fileEntries.filter { !$0.inCortex }
        manifest.semanticIndex = vaultEntries.isEmpty
            ? nil
            : QWMemorySemanticIndex.build(entries: vaultEntries, rootPath: rootURL.path)
        try await Self.buildAndSaveVaultV2Artifacts(
            rootURL: rootURL,
            discoveredFiles: files,
            manifest: &manifest,
            embedBatchFn: makeOptionalEmbedBatchClosure(),
            enableDenseVectors: true
        )

        let geometryDigest = tokContext.geometryDigest ?? loaded.descriptor.geometryDigestV3Hex
        let artifact = QWMemoryManifest.CacheArtifact(
            modelName: config.modelName,
            geometryDigest: geometryDigest,
            kvMode: kvModeName,
            lsaMode: resolvedLSAMode.rawValue,
            lsaSidecarDigest: sidecarURL?.lastPathComponent,
            tokenCount: memoryTokens.count,
            cacheFileName: cacheFileName,
            cacheFileBytes: writtenBytes
        )
        manifest.caches.removeAll {
            ($0.geometryDigest == artifact.geometryDigest || $0.modelName.caseInsensitiveCompare(artifact.modelName) == .orderedSame)
                && $0.kvMode.lowercased() == artifact.kvMode.lowercased()
        }
        manifest.caches.insert(artifact, at: 0)
        try manifest.save(to: rootURL)

        var mountedInfo: QWEngineInfoResponse.MountedMemoryPackInfo? = nil
        if mountAfterIndex {
            onProgress?(QWMemoryIndexProgressEvent(
                phase: "mounting",
                processedFiles: builtFilesTotal,
                totalFiles: builtFilesTotal,
                cortexFiles: cortexCount,
                vaultFiles: vaultCount,
                cortexTokens: totalPrefillTokens,
                prefillProcessedTokens: totalPrefillTokens,
                prefillTotalTokens: totalPrefillTokens,
                progressFraction: 0.97,
                message: "Mounting .qwcache in warm RAM..."
            ))
            mountedInfo = try await mountMemoryPackHoldingTransition(
                at: rootURL.path,
                waitForInFlight: false,
                skipIfAlreadyMounted: false
            )
        } else if needsConfigReload {
            _ = try? await unloadModel(waitForInFlight: false, forReconfigure: true)
            self.config = savedPreConfig
            self.configRevision &+= 1
            self.statusBox.updateConfig(savedPreConfig)
        }

        let elapsed = Date().timeIntervalSince(indexStart)
        let result = QWMemoryIndexResult(
            packName: resolvedPackName,
            rootPath: rootURL.path,
            modelName: config.modelName,
            kvMode: kvModeName,
            lsaMode: resolvedLSAMode.rawValue,
            fileCount: built.fileEntries.count,
            cortexFileCount: cortexCount,
            vaultFileCount: vaultCount,
            tokenCount: memoryTokens.count,
            cacheFilePath: cacheFileURL.path,
            cacheBytes: writtenBytes,
            elapsedSeconds: elapsed,
            mounted: mountedInfo
        )
        let mib = Double(writtenBytes) / 1_048_576.0
        onProgress?(QWMemoryIndexProgressEvent(
            phase: "completed",
            processedFiles: builtFilesTotal,
            totalFiles: builtFilesTotal,
            cortexFiles: cortexCount,
            vaultFiles: vaultCount,
            cortexTokens: totalPrefillTokens,
            prefillProcessedTokens: totalPrefillTokens,
            prefillTotalTokens: totalPrefillTokens,
            progressFraction: 1.0,
            message: String(format: "Indexed %d files (%d tokens, %.2f MiB) in %.1fs",
                            builtFilesTotal, totalPrefillTokens, mib, elapsed),
                result: result
            ))
            return result
        }
    }

    private func resolveInspectionRootURL(at rawPath: String) throws -> URL {
        let trimmed = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw QWMemoryPackError.directoryNotFound(rawPath)
        }
        let startURL = URL(fileURLWithPath: trimmed.expandingTildePath).standardizedFileURL
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: startURL.path, isDirectory: &isDir), isDir.boolValue else {
            throw QWMemoryPackError.directoryNotFound(startURL.path)
        }
        return QWMemoryPack.discoverManifestRoot(walkingUpFrom: startURL) ?? startURL
    }

    private func resolveActiveGeometryDigest() -> String? {
        if case .loaded(let loaded) = state {
            return loaded.descriptor.geometryDigestV3Hex
        }
        if !config.packPath.isEmpty, let assets = try? QWModelRuntimeAssets.load(packPath: config.packPath) {
            return assets.descriptor.geometryDigestV3Hex
        }
        return nil
    }

    private func resolveDefaultLSAModeName() -> String {
        let hasSidecar = QWLSAConfig.resolveSidecarURL(
            packPath: config.packPath,
            explicitSidecarPath: config.lsaConfig.sidecarPath.isEmpty ? nil : config.lsaConfig.sidecarPath
        ) != nil
        if hasSidecar { return "sidecar" }
        if config.lsaConfig.mode != .off { return config.lsaConfig.mode.rawValue }
        return "auto"
    }

    private func artifactMatchesActiveModel(
        _ artifact: QWMemoryManifest.CacheArtifact,
        activeDigest: String?
    ) -> Bool {
        if let activeDigest, !activeDigest.isEmpty, !artifact.geometryDigest.isEmpty,
           artifact.geometryDigest.caseInsensitiveCompare(activeDigest) == .orderedSame {
            return true
        }
        if !config.modelName.isEmpty {
            return artifact.modelName.caseInsensitiveCompare(config.modelName) == .orderedSame
                || QWMemoryPack.safeSlug(artifact.modelName) == QWMemoryPack.safeSlug(config.modelName)
        }
        return false
    }

    private func resolveInstalledModelName(
        for artifactModelName: String,
        matchesActive: Bool,
        discoveredModels: [QWServerSettings.DiscoveredModel],
        settings: QWServerSettings
    ) -> String? {
        if matchesActive { return config.modelName }
        let targetSlug = QWMemoryPack.safeSlug(artifactModelName)
        if let disc = discoveredModels.first(where: {
            $0.name.caseInsensitiveCompare(artifactModelName) == .orderedSame
                || (!targetSlug.isEmpty && QWMemoryPack.safeSlug($0.name) == targetSlug)
                || (!settings.alias(for: $0.name).isEmpty
                    && settings.alias(for: $0.name).caseInsensitiveCompare(artifactModelName) == .orderedSame)
        }) {
            return disc.name
        }
        let ref = settings.resolveModelReference(artifactModelName)
        if !ref.packPath.isEmpty,
           FileManager.default.fileExists(atPath: (ref.packPath as NSString).appendingPathComponent("config.json")) {
            return ref.modelName
        }
        return nil
    }

    private func resolveTokenizerContext() async throws -> (tokenizer: QWTokenizer, chatTemplate: QWChatTemplate, geometryDigest: String?) {
        if case .loaded(let loaded) = state {
            return (loaded.tokenizer, loaded.chatTemplate, loaded.descriptor.geometryDigestV3Hex)
        }
        if !config.packPath.isEmpty,
           let assets = try? QWModelRuntimeAssets.load(packPath: config.packPath),
           let tok = try? QWTokenizer(json: assets.tokenizerJSON) {
            return (tok, assets.chatTemplate, assets.descriptor.geometryDigestV3Hex)
        }
        let loaded = try await ensureLoaded()
        return (loaded.tokenizer, loaded.chatTemplate, loaded.descriptor.geometryDigestV3Hex)
    }

    private func prepareMemoryTokensOffActor(
        packName: String,
        rootPath: String,
        files: [QWMemoryPack.DiscoveredFile],
        clusterBlock: Int,
        kvModeName: String,
        maxTokens: Int,
        tokenizer: QWTokenizer,
        chatTemplate: QWChatTemplate,
        onFileProgress: @escaping @Sendable (QWMemoryPack.FileProgressUpdate) -> Void
    ) async throws -> (built: QWMemoryPack.BuiltEnvelope, tokens: [Int]) {
        try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let built = try QWMemoryPack.buildEnvelope(
                        packName: packName,
                        rootPath: rootPath,
                        files: files,
                        clusterBlockTokens: clusterBlock,
                        kvMode: kvModeName,
                        maxTokens: maxTokens,
                        onFileProgress: onFileProgress
                    ) { chunk in
                        try tokenizer.encode(chunk).count
                    }
                    let wire = try Self.renderMemoryEnvelopeWire(
                        systemContent: built.systemMessageContent,
                        chatTemplate: chatTemplate
                    )
                    let encoded = try tokenizer.encode(wire)
                    let tokens = encoded.isEmpty
                        ? Array(repeating: 1, count: max(1, (built.totalBytes + 3) / 4))
                        : encoded
                    cont.resume(returning: (built, tokens))
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }
}
