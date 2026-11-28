// QWMountedVaultSet.swift — Multi-pack Cognitive Vault v2 coordinator uniting
// cross-pack Hybrid RRF, 1-hop AST declaration slices, PDF spans, and default screening.

import Foundation
import QWCore
import QWDecision

public struct QWVaultV2RecalledItem: Codable, Sendable, Equatable {
    public let packName: String
    public let workspace: String
    public let relativePath: String
    public let sourceKind: String
    public let anchor: String?
    public let rrfScore: Double
    public let relevanceScore: Double
    public let injectionScore: Double

    enum CodingKeys: String, CodingKey {
        case workspace, anchor
        case packName = "pack_name"
        case relativePath = "relative_path"
        case sourceKind = "source_kind"
        case rrfScore = "rrf_score"
        case relevanceScore = "relevance_score"
        case injectionScore = "injection_score"
    }
}

public struct QWVaultV2QuarantinedItem: Codable, Sendable, Equatable {
    public let id: String
    public let reason: String
    public let injectionScore: Double

    enum CodingKeys: String, CodingKey {
        case id, reason
        case injectionScore = "injection_score"
    }
}

public struct QWVaultV2RecallOutcome: Codable, Sendable, Equatable {
    public let mountedPacks: [String]
    public let recalledFiles: [String]
    public let recalledItems: [QWVaultV2RecalledItem]
    public let astSlices: [QWMemorySymbolGraph.DeclarationSlice]
    public let quarantinedCount: Int
    public let quarantinedItems: [QWVaultV2QuarantinedItem]
    public let estimatedTokens: Int
    public let screenMode: String
    public let latencyMs: Double
    public let recallBlock: String

    enum CodingKeys: String, CodingKey {
        case mountedPacks = "mounted_packs"
        case recalledFiles = "recalled_files"
        case recalledItems = "recalled_items"
        case astSlices = "ast_slices"
        case quarantinedCount = "quarantined_count"
        case quarantinedItems = "quarantined_items"
        case estimatedTokens = "estimated_tokens"
        case screenMode = "screen_mode"
        case latencyMs = "latency_ms"
        case recallBlock = "recall_block"
    }
}

public struct QWMountedVaultSet: Sendable {
    public struct PackMount: Sendable {
        public let packName: String
        public let rootURL: URL
        public let manifest: QWMemoryManifest
        public let isPrimaryCortex: Bool
        public let symbolGraph: QWMemorySymbolGraph?
        public let denseVectors: QWMemoryDenseVectorStore?

        public init(
            packName: String? = nil,
            rootURL: URL,
            manifest: QWMemoryManifest,
            isPrimaryCortex: Bool = false,
            symbolGraph: QWMemorySymbolGraph? = nil,
            denseVectors: QWMemoryDenseVectorStore? = nil
        ) {
            let stdRoot = rootURL.standardizedFileURL
            self.packName = packName ?? manifest.packName
            self.rootURL = stdRoot
            self.manifest = manifest
            self.isPrimaryCortex = isPrimaryCortex
            self.symbolGraph = symbolGraph ?? manifest.symbolGraphFile.flatMap {
                QWMemorySymbolGraph.load(from: stdRoot, fileName: $0)
            } ?? QWMemorySymbolGraph.load(from: stdRoot)
            self.denseVectors = denseVectors ?? manifest.denseVectorFile.flatMap {
                QWMemoryDenseVectorStore.load(from: QWMemoryManifest.denseVectorsURL(in: stdRoot, fileName: $0))
            } ?? QWMemoryDenseVectorStore.load(from: QWMemoryManifest.denseVectorsURL(in: stdRoot))
        }
    }

    public struct RecallOptions: Sendable {
        public var maxFiles: Int
        public var maxTokens: Int
        public var includeCortexFiles: Bool
        public var expandASTHops: Int
        public var screenPassages: Bool
        public var packFilter: Set<String>?
        public var excludedPaths: Set<String>

        public init(
            maxFiles: Int = 4,
            maxTokens: Int = QWMemoryVault.defaultVaultRecallTokenBudget,
            includeCortexFiles: Bool = false,
            expandASTHops: Int = 1,
            screenPassages: Bool = true,
            packFilter: Set<String>? = nil,
            excludedPaths: Set<String> = []
        ) {
            self.maxFiles = maxFiles
            self.maxTokens = maxTokens
            self.includeCortexFiles = includeCortexFiles
            self.expandASTHops = expandASTHops
            self.screenPassages = screenPassages
            self.packFilter = packFilter
            self.excludedPaths = excludedPaths
        }
    }

    public typealias EmbedQueryFn = @Sendable (String) async -> [Float]?
    public typealias ScreenEvalFn = @Sendable (String, [QWRerankPassageInput]) async -> (
        evaluated: [QWRerankScreenItem], mode: String, latencyMs: Double
    )

    public private(set) var packs: [PackMount]

    public init(packs: [PackMount] = []) {
        self.packs = packs
    }

    public mutating func upsert(_ mount: PackMount) {
        let stdPath = mount.rootURL.standardizedFileURL.path
        if let idx = packs.firstIndex(where: { $0.rootURL.standardizedFileURL.path == stdPath }) {
            packs[idx] = mount
        } else if mount.isPrimaryCortex {
            packs.insert(mount, at: 0)
        } else {
            packs.append(mount)
        }
    }

    @discardableResult
    public mutating func unmount(matching identifier: String) -> Bool {
        let trimmed = identifier.trimmingCharacters(in: .whitespacesAndNewlines)
        let stdCandidate = URL(fileURLWithPath: trimmed.expandingTildePath).standardizedFileURL.path
        let before = packs.count
        packs.removeAll {
            $0.packName.caseInsensitiveCompare(trimmed) == .orderedSame
                || $0.rootURL.standardizedFileURL.path == stdCandidate
        }
        return packs.count != before
    }

    public mutating func unmountAll() {
        packs.removeAll()
    }

    /// Recalls passages across all active packs via Hybrid RRF, screens for prompt injection, and appends 1-hop AST slices.
    public func recall(
        query: String,
        options: RecallOptions = RecallOptions(),
        embedFn: EmbedQueryFn? = nil,
        screenFn: ScreenEvalFn? = nil
    ) async -> QWVaultV2RecallOutcome {
        let t0 = CFAbsoluteTimeGetCurrent()
        let trimmedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        let activePacks = packs.filter { p in
            guard let filter = options.packFilter, !filter.isEmpty else { return true }
            return filter.contains(p.packName) || filter.contains(p.rootURL.path)
        }
        guard !trimmedQuery.isEmpty, !activePacks.isEmpty, options.maxFiles > 0, options.maxTokens >= 256 else {
            return QWVaultV2RecallOutcome(
                mountedPacks: activePacks.map(\.packName), recalledFiles: [], recalledItems: [],
                astSlices: [], quarantinedCount: 0, quarantinedItems: [], estimatedTokens: 0,
                screenMode: options.screenPassages ? "local-only" : "disabled", latencyMs: 0, recallBlock: ""
            )
        }

        let hasAnyDense = activePacks.contains { $0.denseVectors != nil }
        let queryVec: [Float]? = hasAnyDense ? await embedFn?(trimmedQuery) : nil
        let emeansQueryVec = QWMemorySemanticIndex.embedText(trimmedQuery)

        var unitLookup: [String: SpanCandidateUnit] = [:]
        var lexCandidates: [(id: String, score: Int)] = []
        var emeansCandidates: [(id: String, sim: Float)] = []
        var denseCandidates: [(id: String, anchor: String?, sim: Float)] = []

        for p in activePacks {
            let includeCortex = options.includeCortexFiles || !p.isPrimaryCortex
            var wholeFileUnits: [String: SpanCandidateUnit] = [:]
            var multiSpanPaths: Set<String> = []

            for entry in p.manifest.files {
                if !includeCortex && entry.inCortex { continue }
                let fileCid = "\(p.packName)::\(entry.relativePath)"
                if options.excludedPaths.contains(entry.relativePath) || options.excludedPaths.contains(fileCid) {
                    continue
                }
                let units = Self.buildSpanUnits(pack: p, entry: entry)
                guard !units.isEmpty else { continue }
                if units.count == 1 && units[0].cid == fileCid {
                    let u = units[0]
                    unitLookup[u.cid] = u
                    wholeFileUnits[entry.relativePath] = u
                    let lexScore = QWMemoryVault.scoreLexicalEntry(query: trimmedQuery, entry: entry, rootURL: p.rootURL)
                    if lexScore >= (includeCortex ? 12 : 20) {
                        lexCandidates.append((u.cid, lexScore))
                    }
                } else {
                    multiSpanPaths.insert(entry.relativePath)
                    for u in units {
                        unitLookup[u.cid] = u
                        let spanLex = Self.scoreSpanUnitLexical(query: trimmedQuery, unit: u)
                        if spanLex >= 14 {
                            lexCandidates.append((u.cid, spanLex))
                        }
                        let spanSim = Self.scoreSpanUnitEmeans(queryVec: emeansQueryVec, unit: u)
                        if spanSim >= 0.14 {
                            emeansCandidates.append((u.cid, spanSim))
                        }
                    }
                }
            }
            if let sem = QWMemoryVault.resolveSemanticIndex(manifest: p.manifest, includeCortexFiles: includeCortex) {
                for hit in sem.query(trimmedQuery, excludedPaths: options.excludedPaths, topK: max(10, options.maxFiles * 3), minCosine: 0.14) {
                    if !multiSpanPaths.contains(hit.relativePath), let u = wholeFileUnits[hit.relativePath] {
                        emeansCandidates.append((u.cid, hit.cosine))
                    }
                }
            }
            if let qVec = queryVec, let dv = p.denseVectors {
                for hit in dv.query(vector: qVec, excludedPaths: options.excludedPaths, topK: max(12, options.maxFiles * 3), minCosine: 0.14) {
                    let spanCid = hit.anchor.map { "\(p.packName)::\(hit.relativePath)#\($0)" }
                    let fileCid = "\(p.packName)::\(hit.relativePath)"
                    if let scid = spanCid, unitLookup[scid] != nil {
                        denseCandidates.append((scid, hit.anchor, hit.cosine))
                    } else if unitLookup[fileCid] != nil {
                        denseCandidates.append((fileCid, hit.anchor, hit.cosine))
                    }
                }
            }
        }

        lexCandidates.sort { $0.score != $1.score ? $0.score > $1.score : $0.id < $1.id }
        emeansCandidates.sort { abs($0.sim - $1.sim) > 1e-6 ? $0.sim > $1.sim : $0.id < $1.id }
        denseCandidates.sort { abs($0.sim - $1.sim) > 1e-6 ? $0.sim > $1.sim : $0.id < $1.id }

        var fused = QWMemoryHybridRRF.fuse(
            lexicalRankedIds: lexCandidates.map(\.id),
            emeansRankedIds: emeansCandidates.map(\.id),
            denseRankedItems: denseCandidates.map { ($0.id, $0.anchor) },
            topK: max(options.maxFiles * 3, 12)
        )
        if fused.isEmpty && options.includeCortexFiles {
            let fallback = unitLookup.keys.sorted().prefix(options.maxFiles)
            fused = fallback.enumerated().map { idx, cid in
                QWMemoryHybridRRF.RankedHit(id: cid, rrfScore: 1.0 / (60.0 + Double(idx + 1)), lexicalRank: nil, emeansRank: nil, denseRank: nil, anchor: nil)
            }
        }

        let astReserveCap = options.expandASTHops > 0 ? min(2048, options.maxTokens / 4) : 0
        let primaryBudget = max(256, options.maxTokens - astReserveCap)
        let perSpanTokenCap = max(1024, primaryBudget / 2)

        struct LoadedCandidate {
            let cid: String
            let pack: PackMount
            let entry: QWMemoryManifest.FileEntry
            let windowText: String
            let startLine: Int
            let endLine: Int
            let truncated: Bool
            let anchor: String?
            let rrfScore: Double
        }
        var loadedCands: [LoadedCandidate] = []
        for hit in fused {
            guard let unit = unitLookup[hit.id] else { continue }
            let fitted = Self.fitWindowToBudget(query: trimmedQuery, unit: unit, maxWindowTokens: perSpanTokenCap)
            let resolvedAnchor = hit.anchor ?? unit.anchorLabel ?? Self.matchBestSpanAnchor(query: trimmedQuery, entry: unit.entry, content: fitted.text)
            loadedCands.append(LoadedCandidate(
                cid: hit.id,
                pack: unit.pack,
                entry: unit.entry,
                windowText: fitted.text,
                startLine: fitted.startLine,
                endLine: fitted.endLine,
                truncated: fitted.truncated,
                anchor: resolvedAnchor,
                rrfScore: hit.rrfScore
            ))
        }

        var screenMode = "disabled"
        var quarantined: [QWVaultV2QuarantinedItem] = []
        var relevanceByCid: [String: Double] = [:]
        var injectionByCid: [String: Double] = [:]
        var allowedCids: Set<String> = Set(loadedCands.map(\.cid))

        if options.screenPassages && !loadedCands.isEmpty {
            let passageInputs = loadedCands.map { cand in
                QWRerankPassageInput(id: cand.cid, text: cand.windowText)
            }
            let evalOut: (evaluated: [QWRerankScreenItem], mode: String, latencyMs: Double)
            if let screenFn {
                evalOut = await screenFn(trimmedQuery, passageInputs)
            } else {
                evalOut = await QWCognitiveRails(runner: nil).evaluatePassageScreen(
                    query: trimmedQuery, passages: passageInputs, relevanceThreshold: 0.15
                )
            }
            screenMode = evalOut.mode
            allowedCids.removeAll()
            for item in evalOut.evaluated {
                relevanceByCid[item.id] = item.relevance
                injectionByCid[item.id] = item.injection
                if item.kept {
                    allowedCids.insert(item.id)
                } else if item.droppedReason == "local_injection_pattern" || item.droppedReason == "model_injection_gate" {
                    quarantined.append(QWVaultV2QuarantinedItem(
                        id: item.id,
                        reason: item.droppedReason ?? "injection_quarantine",
                        injectionScore: item.injection
                    ))
                }
            }
        }

        // Model-gated RRF + Cross-Encoder/Sidecar blend (preserve pure Hybrid RRF in local-only / disabled modes).
        var orderedKept = loadedCands.filter { allowedCids.contains($0.cid) }
        if (screenMode == "rerank+local" || screenMode == "decision+local"), !orderedKept.isEmpty {
            let maxRRF = max(1e-9, orderedKept.map(\.rrfScore).max() ?? 1.0)
            let alpha = (screenMode == "rerank+local") ? 0.45 : 0.60
            orderedKept.sort { a, b in
                let bA = alpha * (a.rrfScore / maxRRF) + (1.0 - alpha) * (relevanceByCid[a.cid] ?? 0.5)
                let bB = alpha * (b.rrfScore / maxRRF) + (1.0 - alpha) * (relevanceByCid[b.cid] ?? 0.5)
                if abs(bA - bB) > 1e-9 { return bA > bB }
                if abs(a.rrfScore - b.rrfScore) > 1e-12 { return a.rrfScore > b.rrfScore }
                return a.cid < b.cid
            }
        }

        var usedTokens = 32
        var sections: [String] = []
        var recalledFiles: [String] = []
        var recalledItems: [QWVaultV2RecalledItem] = []
        var primaryForAST: [(packName: String, relativePath: String, content: String)] = []
        var recalledKeySet: Set<String> = []
        var spansPerDocCount: [String: Int] = [:]
        let multiPack = activePacks.count > 1

        for cand in orderedKept {
            if recalledItems.count >= options.maxFiles { break }
            let docKey = "\(cand.pack.packName)::\(cand.entry.relativePath)"
            if (spansPerDocCount[docKey] ?? 0) >= Self.maxSpansPerDocument { continue }

            let displayPath = multiPack ? "\(cand.pack.packName):\(cand.entry.relativePath)" : cand.entry.relativePath
            let anchorSuffix = cand.anchor.map { " \($0)" } ?? ""
            let shortSHA = String(cand.entry.sha256.prefix(8))
            let truncTag = cand.truncated ? " truncated" : ""
            let normalized = cand.windowText.hasSuffix("\n") ? cand.windowText : (cand.windowText + "\n")
            let header = "[VAULT SPAN: \(displayPath)#L\(cand.startLine)-L\(cand.endLine)\(anchorSuffix) sha256:\(shortSHA)\(truncTag)]"
            var section = header + "\n" + normalized
            var est = max(1, (section.utf8.count + 3) / 4)
            if !recalledItems.isEmpty && usedTokens + est > primaryBudget { continue }
            if usedTokens + est > primaryBudget {
                let maxUTF8 = max(512, (primaryBudget - usedTokens) * 3)
                let clipped = String(normalized.prefix(maxUTF8))
                let clippedLines = max(1, clipped.components(separatedBy: .newlines).count)
                let endL = cand.startLine + clippedLines - 1
                section = "[VAULT SPAN: \(displayPath)#L\(cand.startLine)-L\(endL)\(anchorSuffix) sha256:\(shortSHA) truncated]\n" + clipped + "\n"
                est = max(1, (section.utf8.count + 3) / 4)
            }
            spansPerDocCount[docKey, default: 0] += 1
            sections.append(section)
            if !recalledFiles.contains(displayPath) {
                recalledFiles.append(displayPath)
            }
            recalledKeySet.insert("\(cand.pack.packName):\(cand.entry.relativePath)")
            recalledKeySet.insert(cand.entry.relativePath)
            primaryForAST.append((cand.pack.packName, cand.entry.relativePath, cand.windowText))
            recalledItems.append(QWVaultV2RecalledItem(
                packName: cand.pack.packName,
                workspace: cand.pack.rootURL.path,
                relativePath: cand.entry.relativePath,
                sourceKind: cand.entry.sourceKind.rawValue,
                anchor: cand.anchor,
                rrfScore: cand.rrfScore,
                relevanceScore: relevanceByCid[cand.cid] ?? 1.0,
                injectionScore: injectionByCid[cand.cid] ?? 0.0
            ))
            usedTokens += est
        }

        var astSlices: [QWMemorySymbolGraph.DeclarationSlice] = []
        if options.expandASTHops > 0 && !primaryForAST.isEmpty {
            let graphs: [(packName: String, manifest: QWMemoryManifest, graph: QWMemorySymbolGraph)] = activePacks.compactMap { p in
                guard let g = p.symbolGraph else { return nil }
                return (p.packName, p.manifest, g)
            }
            let remainingCap = min(2048, min(options.maxTokens / 4, max(0, options.maxTokens - usedTokens)))
            let primaryCortexName = activePacks.first(where: \.isPrimaryCortex)?.packName
            let expanded = QWMemorySymbolGraph.expandOneHopSlices(
                query: trimmedQuery,
                primaryFiles: primaryForAST,
                mountedGraphs: graphs,
                alreadyRecalledKeys: recalledKeySet,
                excludePrimaryCortexPackName: options.includeCortexFiles ? nil : primaryCortexName,
                maxSliceTokens: remainingCap
            )
            if !expanded.slices.isEmpty && !expanded.formattedBlock.isEmpty {
                var cleanSlices: [QWMemorySymbolGraph.DeclarationSlice] = []
                var cleanLines: [String] = []
                for slice in expanded.slices where !QWCognitiveRails.hasPromptInjectionPattern(slice.signaturePreview) {
                    cleanSlices.append(slice)
                    let pfx = slice.packName.map { "\($0):" } ?? ""
                    cleanLines.append("[AST 1-HOP SLICE: \(pfx)\(slice.relativePath)#L\(slice.startLine)-L\(slice.endLine) (\(slice.kind) \(slice.symbol))]\n\(slice.signaturePreview)")
                }
                if !cleanSlices.isEmpty {
                    astSlices = cleanSlices
                    let joined = cleanLines.joined(separator: "\n\n") + "\n"
                    sections.append(joined)
                    usedTokens += max(1, (joined.utf8.count + 3) / 4)
                }
            }
        }

        let latencyMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
        guard !sections.isEmpty else {
            return QWVaultV2RecallOutcome(
                mountedPacks: activePacks.map(\.packName), recalledFiles: [], recalledItems: [],
                astSlices: [], quarantinedCount: quarantined.count, quarantinedItems: quarantined,
                estimatedTokens: 0, screenMode: screenMode, latencyMs: latencyMs, recallBlock: ""
            )
        }
        let packBanner = activePacks.map(\.packName).joined(separator: "+")
        var block = "\n<|memory_vault_recall|>\n"
        block += "[QWG LSA VAULT RECALL: \(recalledItems.count) span(s) + \(astSlices.count) AST slice(s) from '\(packBanner)' — AUTHORITATIVE WORKSPACE SNAPSHOT; DO NOT RE-READ THESE RANGES VIA read_file/grep]\n"
        block += sections.joined(separator: "\n")
        if !block.hasSuffix("\n") { block += "\n" }
        block += "</|memory_vault_recall|>"
        return QWVaultV2RecallOutcome(
            mountedPacks: activePacks.map(\.packName),
            recalledFiles: recalledFiles,
            recalledItems: recalledItems,
            astSlices: astSlices,
            quarantinedCount: quarantined.count,
            quarantinedItems: quarantined,
            estimatedTokens: usedTokens,
            screenMode: screenMode,
            latencyMs: latencyMs,
            recallBlock: block
        )
    }

    private static func matchBestSpanAnchor(query: String, entry: QWMemoryManifest.FileEntry, content: String) -> String? {
        guard !entry.spanAnchors.isEmpty else { return nil }
        let qWords = Set(query.lowercased().split { !$0.isLetter && !$0.isNumber }.filter { $0.count >= 3 }.map(String.init))
        guard !qWords.isEmpty else { return entry.spanAnchors.first?.label }
        let lines = content.components(separatedBy: .newlines)
        var bestLabel: String? = entry.spanAnchors.first?.label
        var bestHits = 0
        for span in entry.spanAnchors {
            let lo = max(0, min(lines.count - 1, span.startLine - 1))
            let hi = max(lo, min(lines.count - 1, span.endLine - 1))
            let sliceText = (span.label + " " + lines[lo...hi].joined(separator: " ")).lowercased()
            let hits = qWords.reduce(0) { acc, w in sliceText.contains(w) ? (acc + 1) : acc }
            if hits > bestHits {
                bestHits = hits
                bestLabel = span.label
            }
        }
        return bestLabel
    }
}
