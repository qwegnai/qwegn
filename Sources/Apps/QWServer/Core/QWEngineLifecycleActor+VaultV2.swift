// QWEngineLifecycleActor+VaultV2.swift — Multi-pack Cognitive Vault v2 state,
// auxiliary `--repo`/`--docs` mounting, symbol-graph/dense indexing, and recall.

import Foundation
import QWCore
import QWDecision
import QWEngine

public struct QWMountedVaultPackSummary: Codable, Sendable, Equatable {
    public let packName: String
    public let rootPath: String
    public let isPrimaryCortex: Bool
    public let fileCount: Int
    public let pdfFileCount: Int
    public let markdownFileCount: Int
    public let astDeclarationCount: Int
    public let hasDenseVectors: Bool
    public let enforcedKVMode: String?
    public let tokenCount: Int?

    enum CodingKeys: String, CodingKey {
        case packName = "pack_name"
        case rootPath = "root_path"
        case isPrimaryCortex = "is_primary_cortex"
        case fileCount = "file_count"
        case pdfFileCount = "pdf_file_count"
        case markdownFileCount = "markdown_file_count"
        case astDeclarationCount = "ast_declaration_count"
        case hasDenseVectors = "has_dense_vectors"
        case enforcedKVMode = "enforced_kv_mode"
        case tokenCount = "token_count"
    }
}

final class QWMountedVaultSetBox: @unchecked Sendable {
    private let lock = NSLock()
    private var vaultSet = QWMountedVaultSet()
    private var trackersByRootPath: [String: QWMemoryIncrementalTracker] = [:]

    func snapshot(refreshIncremental: Bool = false) -> QWMountedVaultSet {
        lock.withLock {
            if refreshIncremental {
                let nowNs = DispatchTime.now().uptimeNanoseconds
                for idx in vaultSet.packs.indices {
                    let current = vaultSet.packs[idx]
                    let key = current.rootURL.standardizedFileURL.path
                    var tracker = trackersByRootPath[key] ?? QWMemoryIncrementalTracker(
                        rootURL: current.rootURL, manifest: current.manifest
                    )
                    if let updated = tracker.refreshIfNeeded(mount: current, nowUptimeNs: nowNs) {
                        vaultSet.upsert(updated)
                    }
                    trackersByRootPath[key] = tracker
                }
            }
            return vaultSet
        }
    }

    func upsert(_ mount: QWMountedVaultSet.PackMount) {
        lock.withLock {
            let key = mount.rootURL.standardizedFileURL.path
            vaultSet.upsert(mount)
            trackersByRootPath[key] = QWMemoryIncrementalTracker(rootURL: mount.rootURL, manifest: mount.manifest)
        }
    }

    @discardableResult
    func unmount(matching identifier: String) -> Bool {
        lock.withLock {
            let removed = vaultSet.unmount(matching: identifier)
            let activePaths = Set(vaultSet.packs.map { $0.rootURL.standardizedFileURL.path })
            trackersByRootPath = trackersByRootPath.filter { activePaths.contains($0.key) }
            return removed
        }
    }

    func unmountPrimaryOnly() {
        lock.withLock {
            let aux = vaultSet.packs.filter { !$0.isPrimaryCortex }
            vaultSet = QWMountedVaultSet(packs: aux)
            let activePaths = Set(aux.map { $0.rootURL.standardizedFileURL.path })
            trackersByRootPath = trackersByRootPath.filter { activePaths.contains($0.key) }
        }
    }

    func unmountAll() {
        lock.withLock {
            vaultSet.unmountAll()
            trackersByRootPath.removeAll()
        }
    }

    func summaries(primaryInfo: QWEngineInfoResponse.MountedMemoryPackInfo?) -> [QWMountedVaultPackSummary] {
        let packs = lock.withLock { vaultSet.packs }
        return packs.map { p in
            let pdfs = p.manifest.files.filter { $0.sourceKind == .pdfPageSpan }.count
            let mds = p.manifest.files.filter { $0.sourceKind == .markdown }.count
            return QWMountedVaultPackSummary(
                packName: p.packName,
                rootPath: p.rootURL.path,
                isPrimaryCortex: p.isPrimaryCortex,
                fileCount: p.manifest.files.count,
                pdfFileCount: pdfs,
                markdownFileCount: mds,
                astDeclarationCount: p.symbolGraph?.declarations.count ?? 0,
                hasDenseVectors: p.denseVectors != nil,
                enforcedKVMode: p.isPrimaryCortex ? primaryInfo?.enforcedKVMode : nil,
                tokenCount: p.isPrimaryCortex ? primaryInfo?.tokenCount : nil
            )
        }
    }
}

extension QWEngineLifecycleActor {
    /// Builds and saves Vault v2 sidecar artifacts (`symbol_graph.json` + optional `256d` `dense_vectors.qvec`).
    nonisolated public static func buildAndSaveVaultV2Artifacts(
        rootURL: URL,
        discoveredFiles: [QWMemoryPack.DiscoveredFile],
        manifest: inout QWMemoryManifest,
        embedBatchFn: (@Sendable ([String]) async throws -> [[Float]])? = nil,
        enableDenseVectors: Bool = true
    ) async throws {
        let graphInputs = discoveredFiles.map { ($0.relativePath, $0.content) }
        let symbolGraph = QWMemorySymbolGraph.build(files: graphInputs)
        if !symbolGraph.declarations.isEmpty {
            try symbolGraph.save(to: rootURL)
            manifest.symbolGraphFile = "symbol_graph.json"
        } else {
            manifest.symbolGraphFile = nil
        }

        guard enableDenseVectors, let embedBatchFn, !discoveredFiles.isEmpty else {
            return
        }
        let chunkInputs = QWMountedVaultSet.buildDenseChunkInputs(discoveredFiles: discoveredFiles)
        guard !chunkInputs.isEmpty else { return }
        var allVectors: [[Float]] = []
        allVectors.reserveCapacity(chunkInputs.count)
        let batchSize = 16
        var cursor = 0
        while cursor < chunkInputs.count {
            let end = min(chunkInputs.count, cursor + batchSize)
            let texts = chunkInputs[cursor..<end].map(\.snippet)
            guard let vecs = try? await embedBatchFn(texts), vecs.count == texts.count else {
                return
            }
            allVectors.append(contentsOf: vecs)
            cursor = end
        }
        let store = QWMemoryDenseVectorStore(
            dimension: QWMemoryDenseVectorStore.defaultDimension,
            relativePaths: chunkInputs.map(\.path),
            anchors: chunkInputs.map(\.anchor),
            vectors: allVectors
        )
        let qvecURL = QWMemoryManifest.denseVectorsURL(in: rootURL)
        try store.save(to: qvecURL)
        manifest.denseVectorFile = "dense_vectors.qvec"
        manifest.denseVectorDim = QWMemoryDenseVectorStore.defaultDimension
    }

    nonisolated public func listMountedVaultPacks() -> [QWMountedVaultPackSummary] {
        vaultSetBox.summaries(primaryInfo: statusBox.getMountedMemoryPackInfo())
    }

    /// Mounts an auxiliary `--repo` or `--docs` Vault pack (building a manifest on demand when missing if `autoIndex == true`).
    @discardableResult
    public func mountAuxiliaryVaultPack(
        at rawPath: String,
        packName: String? = nil,
        autoIndexIfMissing: Bool = true,
        enableDenseVectors: Bool = true,
        overrideRoots: [String] = []
    ) async throws -> QWMountedVaultPackSummary {
        let rootURL = try resolveValidatedMemoryRootURL(at: rawPath, overrideRoots: overrideRoots)
        let manifestURL = QWMemoryManifest.manifestURL(in: rootURL)
        var manifest: QWMemoryManifest
        if FileManager.default.fileExists(atPath: manifestURL.path),
           let loaded = try? QWMemoryManifest.load(from: rootURL) {
            manifest = loaded
            if QWMemorySymbolGraph.load(from: rootURL) == nil,
               let files = try? QWMemoryPack.discoverFiles(in: rootURL, includeTests: manifest.includeTests) {
                try? await Self.buildAndSaveVaultV2Artifacts(
                    rootURL: rootURL,
                    discoveredFiles: files,
                    manifest: &manifest,
                    embedBatchFn: makeOptionalEmbedBatchClosure(),
                    enableDenseVectors: enableDenseVectors
                )
                try? manifest.save(to: rootURL)
            }
        } else {
            guard autoIndexIfMissing else {
                throw QWMemoryPackError.cacheFileMissing(manifestURL.path)
            }
            let files = try QWMemoryPack.discoverFiles(in: rootURL, includeTests: false, includePDFs: true)
            let resolvedName = packName?.trimmingCharacters(in: .whitespacesAndNewlines)
            let effName = (resolvedName?.isEmpty == false ? resolvedName! : rootURL.lastPathComponent)
            let built = QWMemoryPack.buildEnvelope(
                packName: effName,
                rootPath: rootURL.path,
                files: files,
                maxTokens: QWMemoryVault.defaultCortexTokenBudget
            )
            let allVaultEntries = built.fileEntries.map { entry in
                QWMemoryManifest.FileEntry(
                    relativePath: entry.relativePath,
                    tier: entry.tier,
                    sha256: entry.sha256,
                    mtimeEpochSec: entry.mtimeEpochSec,
                    byteCount: entry.byteCount,
                    tokenOffset: -1,
                    tokenCount: entry.tokenCount,
                    inCortex: false,
                    symbols: entry.symbols,
                    sourceKind: entry.sourceKind,
                    spanAnchors: entry.spanAnchors
                )
            }
            let semIdx = allVaultEntries.isEmpty ? nil : QWMemorySemanticIndex.build(entries: allVaultEntries, rootPath: rootURL.path)
            QWMemoryPack.ensureGitExclude(in: rootURL)
            manifest = QWMemoryManifest(
                packName: effName,
                rootPath: rootURL.path,
                gitHeadSHA: QWMemoryPack.currentGitHeadSHA(in: rootURL),
                files: allVaultEntries,
                caches: [],
                semanticIndex: semIdx
            )
            try await Self.buildAndSaveVaultV2Artifacts(
                rootURL: rootURL,
                discoveredFiles: files,
                manifest: &manifest,
                embedBatchFn: makeOptionalEmbedBatchClosure(),
                enableDenseVectors: enableDenseVectors
            )
            try manifest.save(to: rootURL)
        }

        let isPrimary = activeMemoryPack?.rootURL.standardizedFileURL.path == rootURL.standardizedFileURL.path
        let mount = QWMountedVaultSet.PackMount(
            packName: packName ?? manifest.packName,
            rootURL: rootURL,
            manifest: manifest,
            isPrimaryCortex: isPrimary
        )
        vaultSetBox.upsert(mount)
        let summaries = vaultSetBox.summaries(primaryInfo: statusBox.getMountedMemoryPackInfo())
        return summaries.first(where: { $0.rootPath == rootURL.standardizedFileURL.path }) ?? QWMountedVaultPackSummary(
            packName: mount.packName,
            rootPath: rootURL.path,
            isPrimaryCortex: isPrimary,
            fileCount: manifest.files.count,
            pdfFileCount: manifest.files.filter { $0.sourceKind == .pdfPageSpan }.count,
            markdownFileCount: manifest.files.filter { $0.sourceKind == .markdown }.count,
            astDeclarationCount: mount.symbolGraph?.declarations.count ?? 0,
            hasDenseVectors: mount.denseVectors != nil,
            enforcedKVMode: nil,
            tokenCount: nil
        )
    }

    /// Queries mounted Vault packs (and/or an explicit `workspacePath`) using Hybrid RRF, AST 1-hop slices, and screening.
    nonisolated public func queryVaultV2(
        query: String,
        workspacePath: String? = nil,
        packNames: [String]? = nil,
        maxFiles: Int = 4,
        maxTokens: Int = QWMemoryVault.defaultVaultRecallTokenBudget,
        expandASTHops: Int = 1,
        screen: Bool = true,
        includeCortexFiles: Bool = true,
        overrideRoots: [String] = []
    ) async throws -> QWVaultV2RecallOutcome {
        var set = vaultSetBox.snapshot(refreshIncremental: true)
        if let ws = workspacePath?.trimmingCharacters(in: .whitespacesAndNewlines), !ws.isEmpty {
            let rootURL = try resolveValidatedMemoryRootURL(at: ws, overrideRoots: overrideRoots)
            if !set.packs.contains(where: { $0.rootURL.standardizedFileURL.path == rootURL.standardizedFileURL.path }) {
                let manifest = try QWMemoryManifest.load(from: rootURL)
                var graph = QWMemorySymbolGraph.load(from: rootURL)
                if graph == nil, let files = try? QWMemoryPack.discoverFiles(in: rootURL, includeTests: manifest.includeTests) {
                    graph = QWMemorySymbolGraph.build(files: files.map { ($0.relativePath, $0.content) })
                }
                set.upsert(QWMountedVaultSet.PackMount(
                    rootURL: rootURL,
                    manifest: manifest,
                    isPrimaryCortex: false,
                    symbolGraph: graph
                ))
            }
        }
        let filterSet: Set<String>? = if let packNames, !packNames.isEmpty {
            Set(packNames)
        } else {
            nil
        }
        let opts = QWMountedVaultSet.RecallOptions(
            maxFiles: max(1, maxFiles),
            maxTokens: max(256, maxTokens),
            includeCortexFiles: includeCortexFiles,
            expandASTHops: max(0, expandASTHops),
            screenPassages: screen,
            packFilter: filterSet
        )
        let embedFn: QWMountedVaultSet.EmbedQueryFn?
        if isDecisionSidecarLoaded() {
            embedFn = { [weak self] (q: String) -> [Float]? in
                guard let self else { return nil }
                let res = try? await self.embedTexts([q], dimensions: QWMemoryDenseVectorStore.defaultDimension)
                return res?.embeddings.first
            }
        } else {
            embedFn = nil
        }
        let screenFn: QWMountedVaultSet.ScreenEvalFn = { [weak self] (
            q: String,
            passages: [QWRerankPassageInput]
        ) -> (evaluated: [QWRerankScreenItem], mode: String, latencyMs: Double) in
            guard let self, self.isDecisionSidecarLoaded() else {
                return await QWCognitiveRails(runner: nil).evaluatePassageScreen(
                    query: q, passages: passages, relevanceThreshold: 0.15
                )
            }
            let rerankFn: QWCognitiveRails.RerankFn?
            if await self.isSequencePairCrossEncoderSidecarLoaded() {
                rerankFn = { (queryStr: String, docs: [String]) -> [QWRerankPairScore] in
                    try await self.rerankDocuments(query: queryStr, documents: docs).scores
                }
            } else {
                rerankFn = nil
            }
            let rails = QWCognitiveRails(
                decideFn: { req in try await self.decide(req).response },
                decideBatchFn: { reqs in try await self.decideBatch(reqs).map(\.response) },
                rerankFn: rerankFn
            )
            return await rails.evaluatePassageScreen(
                query: q, passages: passages, relevanceThreshold: 0.15
            )
        }
        return await set.recall(query: query, options: opts, embedFn: embedFn, screenFn: screenFn)
    }

    func isSequencePairCrossEncoderSidecarLoaded() -> Bool {
        guard let cfg = decisionSidecar?.runner.backend.decisionConfig else { return false }
        if case .sequencePair = cfg.framing, case .pooledClassifier = cfg.head {
            return true
        }
        return false
    }

    func makeOptionalEmbedBatchClosure() -> (@Sendable ([String]) async throws -> [[Float]])? {
        guard decisionSidecar?.runner.backend is any QWEmbeddingExecutionBackend else {
            return nil
        }
        return { [weak self] (texts: [String]) -> [[Float]] in
            guard let self else { return [] }
            let out = try await self.embedTexts(
                texts,
                normalize: true,
                dimensions: QWMemoryDenseVectorStore.defaultDimension
            )
            return out.embeddings
        }
    }
}
