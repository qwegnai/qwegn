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

    @Argument(help: "Repository or knowledge directory root to index (default: current directory).")
    var path: String = "."

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

    func run() throws {
        let rootURL = URL(fileURLWithPath: path.expandingTildePath, isDirectory: true).standardizedFileURL
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: rootURL.path, isDirectory: &isDir), isDir.boolValue else {
            FileHandle.standardError.write("qwg index: error: directory not found at '\(rootURL.path)'\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        let resolvedPackName = packName?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
            ? packName!.trimmingCharacters(in: .whitespacesAndNewlines)
            : rootURL.lastPathComponent

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

        if manifestOnly && resolvedModelPack.isEmpty {
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
            try manifest.save(to: rootURL)
            let estTokens = built.fileEntries.filter(\.inCortex).last.map { $0.tokenOffset + $0.tokenCount } ?? 0
            print("indexed \(built.fileEntries.count) files (\(built.cortexFileCount) cortex + \(built.vaultFileCount) vault, ~\(estTokens) cortex tokens) -> \(QWMemoryManifest.manifestURL(in: rootURL).path)")
            return
        }

        guard !resolvedModelPack.isEmpty else {
            FileHandle.standardError.write("qwg index: error: missing model (pass --model <name|alias|path> or configure a default model)\n".data(using: .utf8)!)
            throw ExitCode(2)
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
        let wire = try Self.renderMemoryEnvelopeWire(
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

        if manifestOnly {
            try manifest.save(to: rootURL)
            if built.vaultFileCount > 0 {
                print("indexed \(built.fileEntries.count) files (\(built.cortexFileCount) cortex + \(built.vaultFileCount) vault, \(memoryTokens.count) cortex tokens, \(built.totalBytes) B) -> \(QWMemoryManifest.manifestURL(in: rootURL).path)")
            } else {
                print("indexed \(built.fileEntries.count) files (\(memoryTokens.count) tokens, \(built.totalBytes) B) -> \(QWMemoryManifest.manifestURL(in: rootURL).path)")
            }
            return
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
        let lsaConfig = QWLSAConfig(
            mode: resolvedLSAMode,
            budget: effectivePair.settings.lsaBudget,
            clusterBlock: clusterBlock,
            sidecarPath: sidecarURL?.path ?? effectivePair.settings.lsaSidecarPath
        )

        let modelSlug = QWMemoryPack.safeSlug(resolvedModelName)
        let packSlug = QWMemoryPack.safeSlug(resolvedPackName)
        let cacheFileName = "\(packSlug)_\(modelSlug)_\(kvModeName).qwcache"
        let cacheFileURL = QWMemoryManifest.qwgDirectoryURL(in: rootURL).appendingPathComponent(cacheFileName)

        let requiredCapacity = min(hardCeiling, max(4096, ((memoryTokens.count + 4096 + 255) / 256) * 256))
        let configuredChunk = QWPrefillLimits.clampedRows(effectivePair.settings.prefillChunkSize)
        let options = QWEngineBuildOptions(
            packPath: resolvedModelPack,
            maxTokens: requiredCapacity,
            kvMode: kvModeName,
            prefillChunkSize: configuredChunk,
            prefixCacheDirectory: nil,
            cacheConfig: nil,
            mtpDrafts: runtimeAssets.descriptor.capabilities.contains(.mtp) ? 1 : 0,
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
            geometryDigest: runtimeAssets.descriptor.geometryDigestV2Hex,
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

    static func renderMemoryEnvelopeWire(
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
}
