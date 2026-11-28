// QwgIndex.swift — `qwg index` CLI command for building offline LSA/QSA codebase memory packs.
// Scans a repository or documentation root in 3-tier stability order, aligns sections to
// 512-token LSA Ward superblocks, prefills the prefix, and writes `<root>/.qwg/*.qwcache`.

import ArgumentParser
import Foundation
import QWCli
import QWCore
import QWEngine
import QWKV
import QWQuant
import QWServer
import QWToken

struct Index: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "index",
        abstract: "Build an offline LSA/QSA codebase memory pack (<root>/.qwg/manifest.json + .qwcache)."
    )

    @Argument(help: "Repository or knowledge directory roots to index (default: current directory).")
    var paths: [String] = []

    var path: String { paths.first ?? "." }

    @Option(name: .customLong("docs"),
            help: "Additional documentation/PDF directory roots to index as off-context Vault packs (repeatable).")
    var docs: [String] = []

    @Flag(name: .customLong("no-dense-vectors"),
          help: "Skip writing .qwg/dense_vectors.qvec even when an embedding sidecar is available.")
    var noDenseVectors: Bool = false

    @Option(name: .customLong("embed-model"),
            help: "Optional embedding-capable decision pack to compute 256d .qwg/dense_vectors.qvec.")
    var embedModel: String? = nil

    @Option(name: [.customLong("model"), .customShort("m")],
            help: "Model directory, name in ~/.qwg/models, or alias from settings.")
    var modelOption: String? = nil

    @Option(name: .customLong("name"),
            help: "Override memory pack name (default: directory basename).")
    var packName: String? = nil

    @Option(name: .customLong("kv-mode"),
            help: "KV cache quantization mode (q4, q8, f16). Defaults to model profile kv_mode or q4.")
    var kvMode: String? = nil

    @Option(name: .customLong("lsa-mode"),
            help: "LSA mode (auto, sidecar, svd, qsa-cluster, off).")
    var lsaMode: String? = nil

    @Option(name: .customLong("cluster-block"),
            help: "LSA Ward superblock token alignment (default: 512).")
    var clusterBlock: Int = 512

    @Option(name: .customLong("max-file-kb"),
            help: "Maximum size in KiB for a single indexed text file (default: 256).")
    var maxFileKB: Int = 256

    @Option(name: .customLong("max-tokens"),
            help: "Resident Cortex .qwcache token budget (0 = default 32768 Cortex + Off-Context Vault).")
    var maxTokens: Int = 0

    @Flag(name: .customLong("full-prefix"),
          help: "Pack all discovered files into a single linear prefix up to the 1M KV ceiling.")
    var fullPrefix: Bool = false

    @Flag(name: .customLong("include-tests"),
          help: "Include Tier 2 test and script files in the memory pack.")
    var includeTests: Bool = false

    @Flag(name: .customLong("manifest-only"),
          help: "Write .qwg/manifest.json and print token statistics without running GPU prefill.")
    var manifestOnly: Bool = false

    @Option(name: [.customLong("mtp-backend")],
            help: "MTP execution backend for speculative prefix state: standard, variational, dflash2 or dspine.",
            transform: { $0.lowercased() })
    var mtpBackend: String? = nil

    @Option(name: .customLong("dflash-draft-pack"),
            help: "Path to the DFlash 2 / DSpine drafter model pack.")
    var dflashDraftPack: String? = nil

    func run() throws {
        let targetRepoPaths = paths.isEmpty ? ["."] : paths
        for docRoot in docs where !docRoot.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            try indexSingleRoot(rawPath: docRoot, forceManifestOnly: true)
        }
        for (idx, rawRepo) in targetRepoPaths.enumerated() {
            try indexSingleRoot(rawPath: rawRepo, forceManifestOnly: manifestOnly, overridePackName: idx == 0 ? packName : nil)
        }
    }

    private func indexSingleRoot(rawPath: String, forceManifestOnly: Bool, overridePackName: String? = nil) throws {
        let effectiveManifestOnly = forceManifestOnly
        let rootURL = URL(fileURLWithPath: rawPath.expandingTildePath, isDirectory: true).standardizedFileURL
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: rootURL.path, isDirectory: &isDir), isDir.boolValue else {
            FileHandle.standardError.write("qwg index: error: directory not found at '\(rootURL.path)'\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        let trimmedPackName = (overridePackName ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedPackName = trimmedPackName.isEmpty ? rootURL.lastPathComponent : trimmedPackName

        let settings = QWSettingsManager.shared.loadSettings()
        let rawModelRef = modelOption ?? settings.modelPackPath
        let modelRef = settings.resolveModelReference(rawModelRef)
        let resolvedModelPack = modelRef.packPath
        let resolvedModelName = modelRef.modelName.isEmpty
            ? URL(fileURLWithPath: resolvedModelPack).lastPathComponent
            : modelRef.modelName

        let effectivePair = QWSettingsManager.shared.effectiveSettings(
            forModel: resolvedModelName,
            packPath: resolvedModelPack,
            base: settings
        )
        let profileKV = effectivePair.profile?.values["kvMode"]?.stringValue
            ?? effectivePair.profile?.values["kv_mode"]?.stringValue
        let effectiveKVString = (kvMode ?? profileKV ?? "q4").lowercased()
        let validatedKVMode: QWKVMode
        do {
            validatedKVMode = try QWCli.resolveKVMode(effectiveKVString)
        } catch {
            FileHandle.standardError.write("qwg index: error: \(error)\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        let kvModeName = String(describing: validatedKVMode)

        let files = try QWMemoryPack.discoverFiles(
            in: rootURL,
            includeTests: includeTests,
            maxFileBytes: max(1, maxFileKB) * 1024
        )
        guard !files.isEmpty else {
            FileHandle.standardError.write("qwg index: error: no indexable source files found in '\(rootURL.path)'\n".data(using: .utf8)!)
            throw ExitCode(2)
        }

        QWMemoryPack.ensureGitExclude(in: rootURL)
        let gitSHA = QWMemoryPack.currentGitHeadSHA(in: rootURL)

        if effectiveManifestOnly && resolvedModelPack.isEmpty {
            let requestedBudget = maxTokens > 0
                ? maxTokens
                : (fullPrefix ? nil : QWMemoryVault.defaultCortexTokenBudget)
            let built = QWMemoryPack.buildEnvelope(
                packName: resolvedPackName,
                rootPath: rootURL.path,
                files: files,
                clusterBlockTokens: clusterBlock,
                kvMode: kvModeName,
                maxTokens: requestedBudget
            )
            var manifest = (try? QWMemoryManifest.load(from: rootURL)) ?? QWMemoryManifest(
                packName: resolvedPackName,
                rootPath: rootURL.path
            )
            manifest.packName = resolvedPackName
            manifest.rootPath = rootURL.path
            manifest.gitHeadSHA = gitSHA
            manifest.clusterBlockTokens = clusterBlock
            manifest.includeTests = includeTests
            manifest.files = built.fileEntries
            let vaultOnly = built.fileEntries.filter { !$0.inCortex }
            manifest.semanticIndex = vaultOnly.isEmpty
                ? nil
                : QWMemorySemanticIndex.build(entries: vaultOnly, rootPath: rootURL.path)
            Self.saveVaultV2ArtifactsSync(
                rootURL: rootURL, files: files, manifest: &manifest,
                embedModel: embedModel, settings: settings, noDenseVectors: noDenseVectors
            )
            try manifest.save(to: rootURL)
            let estTokens = built.fileEntries.filter(\.inCortex).last.map { $0.tokenOffset + $0.tokenCount } ?? 0
            print("indexed \(built.fileEntries.count) files (\(built.cortexFileCount) cortex + \(built.vaultFileCount) vault, ~\(estTokens) cortex tokens) -> \(QWMemoryManifest.manifestURL(in: rootURL).path)")
            return
        }

        guard !resolvedModelPack.isEmpty else {
            FileHandle.standardError.write("qwg index: error: missing model (pass --model <name|alias|path> or configure a default model)\n".data(using: .utf8)!)
            throw ExitCode(2)
        }

        let sidecarURL = QWLSAConfig.resolveSidecarURL(
            packPath: resolvedModelPack,
            explicitSidecarPath: effectivePair.settings.lsaSidecarPath.isEmpty ? nil : effectivePair.settings.lsaSidecarPath
        )
        let defaultLSAMode: String
        if sidecarURL != nil {
            defaultLSAMode = "sidecar"
        } else if effectivePair.settings.lsaMode.lowercased() != "off" {
            defaultLSAMode = effectivePair.settings.lsaMode
        } else {
            defaultLSAMode = "auto"
        }
        let resolvedLSAModeStr = lsaMode ?? defaultLSAMode
        guard let resolvedLSAMode = QWLSAMode.resolve(resolvedLSAModeStr) else {
            FileHandle.standardError.write("qwg index: error: invalid --lsa-mode '\(resolvedLSAModeStr)'\n".data(using: .utf8)!)
            throw ExitCode(2)
        }

        let trimmedBackend = mtpBackend?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let explicitBackend = (trimmedBackend?.isEmpty == false) ? trimmedBackend : nil
        if let explicitBackend {
            do {
                _ = try QWMTPBackend.resolve(name: explicitBackend)
            } catch {
                FileHandle.standardError.write("qwg index: error: \(error)\n".data(using: .utf8)!)
                throw ExitCode(2)
            }
        }
        let trimmedDraftPack = dflashDraftPack?.trimmingCharacters(in: .whitespacesAndNewlines)
        let explicitDraftPack = (trimmedDraftPack?.isEmpty == false) ? trimmedDraftPack : nil

        if !effectiveManifestOnly, let runningPID = QWEngineInstanceLock.runningEnginePID() {
            let delegatedReq = QWMemoryIndexRequest(
                path: rootURL.path,
                packName: resolvedPackName,
                model: modelOption,
                kvMode: kvModeName,
                lsaMode: resolvedLSAMode.rawValue,
                scope: fullPrefix ? "full_prefix" : "cortex_vault",
                maxCortexTokens: maxTokens > 0 ? maxTokens : nil,
                includeTests: includeTests,
                mountAfterIndex: false,
                stream: true,
                mtpBackend: explicitBackend,
                dflashDraftPackPath: explicitDraftPack
            )
            if try Self.delegateIndexToRunningServer(
                request: delegatedReq,
                settings: settings,
                runningPID: runningPID
            ) {
                return
            }
        }

        let location = try QWModelLocation(url: URL(fileURLWithPath: resolvedModelPack), assetsOverride: nil)
        let runtimeAssets = try QWModelRuntimeAssets.load(packPath: resolvedModelPack, assetsOverride: location.assets)
        let tok = try QWTokenizer(json: runtimeAssets.tokenizerJSON)

        let hardCeiling = QWKVCache.maxTokensPerSeq
        let defaultBudgetCeiling = max(4096, hardCeiling - 8192)
        let effectiveMaxTokens: Int
        if maxTokens > 0 {
            effectiveMaxTokens = min(maxTokens, defaultBudgetCeiling)
        } else if fullPrefix {
            effectiveMaxTokens = defaultBudgetCeiling
        } else {
            effectiveMaxTokens = min(QWMemoryVault.defaultCortexTokenBudget, defaultBudgetCeiling)
        }

        let built = try QWMemoryPack.buildEnvelope(
            packName: resolvedPackName,
            rootPath: rootURL.path,
            files: files,
            clusterBlockTokens: clusterBlock,
            kvMode: kvModeName,
            maxTokens: effectiveMaxTokens,
            tokenCounter: { try tok.encode($0).count }
        )
        let wire = try QWEngineLifecycleActor.renderMemoryEnvelopeWire(
            systemContent: built.systemMessageContent,
            chatTemplate: runtimeAssets.chatTemplate
        )
        let memoryTokens = try tok.encode(wire)

        var manifest = (try? QWMemoryManifest.load(from: rootURL)) ?? QWMemoryManifest(
            packName: resolvedPackName,
            rootPath: rootURL.path
        )
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
        Self.saveVaultV2ArtifactsSync(
            rootURL: rootURL, files: files, manifest: &manifest,
            embedModel: embedModel, settings: settings, noDenseVectors: noDenseVectors
        )

        if effectiveManifestOnly {
            try manifest.save(to: rootURL)
            if built.vaultFileCount > 0 {
                print("indexed \(built.fileEntries.count) files (\(built.cortexFileCount) cortex + \(built.vaultFileCount) vault, \(memoryTokens.count) cortex tokens, \(built.totalBytes) B) -> \(QWMemoryManifest.manifestURL(in: rootURL).path)")
            } else {
                print("indexed \(built.fileEntries.count) files (\(memoryTokens.count) tokens, \(built.totalBytes) B) -> \(QWMemoryManifest.manifestURL(in: rootURL).path)")
            }
            return
        }

        let envLoopShare = QWLSAConfig.parseLoopShareEnv(ProcessInfo.processInfo.environment[QWLSAConfig.loopShareEnvKey])
        let crossLoopReuse = envLoopShare?.enabled ?? effectivePair.settings.lsaCrossLoopReuse
        let lsaConfig = QWLSAConfig(
            mode: resolvedLSAMode,
            budget: effectivePair.settings.lsaBudget,
            clusterBlock: clusterBlock,
            sidecarPath: sidecarURL?.path ?? effectivePair.settings.lsaSidecarPath,
            crossLoopReuse: crossLoopReuse,
            loopShareStride: crossLoopReuse ? (envLoopShare?.stride ?? 0) : 0
        )

        let modelSlug = QWMemoryPack.safeSlug(resolvedModelName)
        let packSlug = QWMemoryPack.safeSlug(resolvedPackName)
        let cacheFileName = "\(packSlug)_\(modelSlug)_\(kvModeName).qwcache"
        let cacheFileURL = QWMemoryManifest.qwgDirectoryURL(in: rootURL).appendingPathComponent(cacheFileName)

        let indexSpec = try Self.resolveIndexSpeculation(
            explicitBackend: explicitBackend,
            explicitDraftPack: explicitDraftPack,
            settings: effectivePair.settings,
            descriptor: runtimeAssets.descriptor
        )
        let requiredCapacity = min(hardCeiling, max(4096, ((memoryTokens.count + 4096 + 255) / 256) * 256))
        let configuredChunk = QWPrefillLimits.clampedRows(effectivePair.settings.prefillChunkSize)
        let options = QWEngineBuildOptions(
            packPath: resolvedModelPack,
            maxTokens: requiredCapacity,
            kvMode: kvModeName,
            prefillChunkSize: configuredChunk,
            prefixCacheDirectory: nil,
            cacheConfig: nil,
            mtpDrafts: indexSpec.drafts,
            mtpBackend: indexSpec.backend,
            dflashDraftPackPath: indexSpec.dflashDraftPackPath,
            lsaConfig: lsaConfig
        )

        let fileSummary = built.vaultFileCount > 0
            ? "\(built.fileEntries.count) files (\(built.cortexFileCount) cortex + \(built.vaultFileCount) vault)"
            : "\(built.fileEntries.count) files"
        FileHandle.standardError.write(
            "qwg index: packing '\(resolvedPackName)' (\(fileSummary), \(memoryTokens.count) cortex tokens) with \(resolvedModelName) [kv=\(kvModeName), lsa=\(resolvedLSAMode.rawValue)]...\n".data(using: .utf8)!
        )
        try QWEngineInstanceLock.acquireEngineLock()
        defer { QWEngineInstanceLock.releaseEngineLock() }

        let t0 = Date()
        let engine = try QWEngine(options: options, descriptor: runtimeAssets.descriptor)
        let writtenBytes = try engine.buildStandalonePrefixCache(
            tokens: memoryTokens,
            outputURL: cacheFileURL,
            pinAsSharedBase: false
        )
        let elapsed = Date().timeIntervalSince(t0)

        let artifact = QWMemoryManifest.CacheArtifact(
            modelName: resolvedModelName,
            geometryDigest: runtimeAssets.descriptor.geometryDigestV3Hex,
            kvMode: kvModeName,
            lsaMode: resolvedLSAMode.rawValue,
            lsaSidecarDigest: sidecarURL?.lastPathComponent,
            tokenCount: memoryTokens.count,
            cacheFileName: cacheFileName,
            cacheFileBytes: writtenBytes
        )
        manifest.caches.removeAll {
            $0.geometryDigest == artifact.geometryDigest && $0.kvMode.lowercased() == artifact.kvMode.lowercased()
        }
        manifest.caches.insert(artifact, at: 0)
        try manifest.save(to: rootURL)

        let mib = Double(writtenBytes) / 1_048_576.0
        print(String(
            format: "wrote %@ (%.2f MiB, %d tokens, %d files, kv=%@, lsa=%@ in %.1f s)",
            cacheFileURL.path, mib, memoryTokens.count, built.fileEntries.count,
            kvModeName, resolvedLSAMode.rawValue, elapsed
        ))
    }

    private static func delegateIndexToRunningServer(
        request: QWMemoryIndexRequest,
        settings: QWServerSettings,
        runningPID: Int32
    ) throws -> Bool {
        let rawHost = settings.host.trimmingCharacters(in: .whitespacesAndNewlines)
        let connectHost = (rawHost.isEmpty || rawHost == "0.0.0.0" || rawHost == "::") ? "127.0.0.1" : rawHost
        let port = settings.port > 0 ? settings.port : 8080
        guard let url = URL(string: "http://\(connectHost):\(port)/v1/memory/index") else {
            return false
        }
        var urlRequest = URLRequest(url: url, timeoutInterval: 3600)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        let envKey = ProcessInfo.processInfo.environment["QWG_API_KEY"]
            ?? ProcessInfo.processInfo.environment["QW_API_KEY"]
        let bearerKey = (envKey?.isEmpty == false ? envKey : nil)
            ?? (settings.apiKey.isEmpty ? nil : settings.apiKey)
        if let bearerKey {
            urlRequest.setValue("Bearer \(bearerKey)", forHTTPHeaderField: "Authorization")
        }
        urlRequest.httpBody = try JSONEncoder().encode(request)

        final class OutcomeBox: @unchecked Sendable {
            var reachable = false
            var result: QWMemoryIndexResult? = nil
            var failureMessage: String? = nil
        }

        let box = OutcomeBox()
        let sem = DispatchSemaphore(value: 0)
        let reqCopy = urlRequest
        let isTTY = isatty(STDERR_FILENO) != 0

        FileHandle.standardError.write(
            "qwg index: delegating to running server (PID \(runningPID), \(connectHost):\(port))...\n".data(using: .utf8)!
        )

        Task {
            defer { sem.signal() }
            do {
                let config = URLSessionConfiguration.ephemeral
                config.timeoutIntervalForRequest = 3600
                config.timeoutIntervalForResource = 3600
                let session = URLSession(configuration: config)
                let (bytes, response) = try await session.bytes(for: reqCopy)
                guard let http = response as? HTTPURLResponse else {
                    box.failureMessage = "Invalid HTTP response from local server."
                    return
                }
                box.reachable = true
                if http.statusCode != 200 {
                    var bodyBytes: [UInt8] = []
                    for try await byte in bytes { bodyBytes.append(byte) }
                    let text = String(bytes: bodyBytes, encoding: .utf8) ?? "HTTP \(http.statusCode)"
                    box.failureMessage = "Server rejected index request (HTTP \(http.statusCode)): \(text)"
                    return
                }

                var lastPrintedBucket = -1
                for try await line in bytes.lines {
                    guard line.hasPrefix("data: ") else { continue }
                    let payloadStr = String(line.dropFirst(6)).trimmingCharacters(in: .whitespacesAndNewlines)
                    if payloadStr == "[DONE]" { break }
                    guard let data = payloadStr.data(using: .utf8),
                          let ev = try? JSONDecoder().decode(QWMemoryIndexProgressEvent.self, from: data) else {
                        continue
                    }
                    if ev.phase == "error" {
                        box.failureMessage = ev.error ?? ev.message
                        break
                    }
                    if let res = ev.result {
                        box.result = res
                    }
                    let pct = Int((ev.progressFraction * 100.0).rounded())
                    if isTTY {
                        let lineMsg = String(format: "\r\u{1B}[Kqwg index [%3d%%]: %@", pct, ev.message)
                        FileHandle.standardError.write(lineMsg.data(using: .utf8)!)
                    } else {
                        let bucket = pct / 10
                        if ev.phase == "loading_model" || ev.phase == "saving" || ev.phase == "completed" || bucket > lastPrintedBucket {
                            lastPrintedBucket = max(lastPrintedBucket, bucket)
                            let lineMsg = String(format: "qwg index [%3d%%]: %@\n", pct, ev.message)
                            FileHandle.standardError.write(lineMsg.data(using: .utf8)!)
                        }
                    }
                }
                if isTTY {
                    FileHandle.standardError.write("\n".data(using: .utf8)!)
                }
            } catch {
                if !box.reachable {
                    return
                }
                box.failureMessage = error.localizedDescription
            }
        }

        sem.wait()
        if !box.reachable {
            return false
        }
        if let err = box.failureMessage {
            FileHandle.standardError.write("qwg index: error: \(err)\n".data(using: .utf8)!)
            throw ExitCode(1)
        }
        guard let result = box.result else {
            FileHandle.standardError.write("qwg index: error: server stream ended without completion result.\n".data(using: .utf8)!)
            throw ExitCode(1)
        }
        let mib = Double(result.cacheBytes) / 1_048_576.0
        print(String(
            format: "wrote %@ (%.2f MiB, %d tokens, %d files, kv=%@, lsa=%@ in %.1f s)",
            result.cacheFilePath, mib, result.tokenCount, result.fileCount,
            result.kvMode, result.lsaMode, result.elapsedSeconds
        ))
        return true
    }

    struct IndexSpeculation {
        let drafts: Int
        let backend: QWMTPBackend?
        let dflashDraftPackPath: String?
    }

    /// Resolves speculative prefix-cache state options for `qwg index` from CLI flags, settings, and target capabilities.
    static func resolveIndexSpeculation(
        explicitBackend: String?,
        explicitDraftPack: String?,
        settings: QWServerSettings,
        descriptor: QWModelDescriptor
    ) throws -> IndexSpeculation {
        let nativeDrafts = descriptor.capabilities.contains(.mtp) ? 1 : 0
        let backendName: String? = if let explicitBackend {
            explicitBackend
        } else if let explicitDraftPack {
            QWDFlashConfig.inferBackendAndMaxDrafts(packPath: explicitDraftPack).backend
        } else {
            settings.mtpBackend.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? nil : settings.mtpBackend.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        }
        guard let resolvedBackend = try backendName.map({ try QWMTPBackend.resolve(name: $0) }) else {
            return IndexSpeculation(drafts: nativeDrafts, backend: nil, dflashDraftPackPath: nil)
        }
        guard resolvedBackend == .dflash2 else {
            return IndexSpeculation(drafts: nativeDrafts, backend: resolvedBackend, dflashDraftPackPath: nil)
        }
        let configuredPack = settings.dflashDraftPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let candidatePack: String? = explicitDraftPack
            ?? (configuredPack.isEmpty ? nil : configuredPack)
            ?? (backendName == "dspine" ? QWDFlashConfig.defaultDSpinePackPath : nil)
        let explicitRequest = explicitBackend != nil || explicitDraftPack != nil
        if explicitRequest {
            do {
                try QWDFlashConfig.validateDrafter(targetDescriptor: descriptor, packPath: candidatePack, drafts: 1)
            } catch {
                FileHandle.standardError.write("qwg index: error: \(error)\n".data(using: .utf8)!)
                throw ExitCode(2)
            }
        } else if (try? QWDFlashConfig.validateDrafter(targetDescriptor: descriptor, packPath: candidatePack, drafts: 1)) == nil {
            return IndexSpeculation(drafts: nativeDrafts, backend: nil, dflashDraftPackPath: nil)
        }
        let inferred = QWDFlashConfig.inferBackendAndMaxDrafts(packPath: candidatePack)
        return IndexSpeculation(drafts: max(1, inferred.maxDrafts), backend: .dflash2, dflashDraftPackPath: candidatePack)
    }

    private static func saveVaultV2ArtifactsSync(
        rootURL: URL,
        files: [QWMemoryPack.DiscoveredFile],
        manifest: inout QWMemoryManifest,
        embedModel: String?,
        settings: QWServerSettings,
        noDenseVectors: Bool
    ) {
        let graph = QWMemorySymbolGraph.build(files: files.map { ($0.relativePath, $0.content) })
        if !graph.declarations.isEmpty, (try? graph.save(to: rootURL)) != nil {
            manifest.symbolGraphFile = "symbol_graph.json"
        }
        guard !noDenseVectors else { return }
        let rawEmbedRef = (embedModel?.trimmingCharacters(in: .whitespacesAndNewlines)).flatMap { $0.isEmpty ? nil : $0 }
            ?? (settings.decisionModelPackPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : settings.decisionModelPackPath)
        guard let rawEmbedRef else { return }
        let resolvedRef = settings.resolveModelReference(rawEmbedRef, allowArbitraryFilesystemPaths: true)
        let packPath = resolvedRef.packPath.isEmpty ? rawEmbedRef.expandingTildePath : resolvedRef.packPath
        guard FileManager.default.fileExists(atPath: packPath) else { return }

        final class VecBox: @unchecked Sendable { var vectors: [[Float]]? = nil }
        let box = VecBox()
        let sem = DispatchSemaphore(value: 0)
        let snippets = files.map { $0.relativePath + "\n" + String($0.content.prefix(2048)) }
        Task {
            defer { sem.signal() }
            let actor = QWEngineLifecycleActor(
                config: QWEngineLifecycleActor.Config(packPath: "", modelName: ""),
                logger: QWServerLogger(level: .none)
            )
            guard (try? await actor.loadDecisionSidecar(packPath: packPath)) != nil else { return }
            var out: [[Float]] = []
            var cursor = 0
            while cursor < snippets.count {
                let end = min(snippets.count, cursor + 16)
                let batch = Array(snippets[cursor..<end])
                guard let res = try? await actor.embedTexts(
                    batch, normalize: true, dimensions: QWMemoryDenseVectorStore.defaultDimension
                ), res.embeddings.count == batch.count else {
                    return
                }
                out.append(contentsOf: res.embeddings)
                cursor = end
            }
            box.vectors = out
        }
        sem.wait()
        if let vecs = box.vectors, vecs.count == files.count {
            let store = QWMemoryDenseVectorStore(
                dimension: QWMemoryDenseVectorStore.defaultDimension,
                relativePaths: files.map(\.relativePath),
                anchors: files.map { $0.spanAnchors.first?.label },
                vectors: vecs
            )
            if (try? store.save(to: QWMemoryManifest.denseVectorsURL(in: rootURL))) != nil {
                manifest.denseVectorFile = "dense_vectors.qvec"
                manifest.denseVectorDim = QWMemoryDenseVectorStore.defaultDimension
            }
        }
    }
}
