// QWMountedVaultSet+Spans.swift — Span-keyed retrieval units, per-span lexical
// & 128d e-means scoring, and smart code/document windowing for Vault v2.

import Accelerate
import Foundation
import QWCore

extension QWMountedVaultSet {
    /// Minimum byte size before a Markdown document is split into per-heading RRF span units.
    static let markdownMultiSpanMinBytes: Int = 4096
    /// Maximum spans recalled from any single document (e.g. TOC + 2 deep sections) without starving code files.
    static let maxSpansPerDocument: Int = 3

    struct SpanCandidateUnit {
        let cid: String
        let pack: PackMount
        let entry: QWMemoryManifest.FileEntry
        let anchorLabel: String?
        let startLine: Int
        let endLine: Int
        let windowText: String
    }

    /// Expands a manifest `FileEntry` into `1` whole-file unit or `N` span-keyed units (`pack::path#anchor`).
    static func buildSpanUnits(
        pack: PackMount,
        entry: QWMemoryManifest.FileEntry
    ) -> [SpanCandidateUnit] {
        let shouldSplitSpans = !entry.spanAnchors.isEmpty && (
            entry.sourceKind == .pdfPageSpan
                || (entry.sourceKind == .markdown && entry.byteCount > markdownMultiSpanMinBytes)
        )
        guard let fullText = QWDocumentIngestor.readFormattedText(for: entry, rootURL: pack.rootURL),
              !fullText.isEmpty else {
            return []
        }
        let lines = fullText.components(separatedBy: .newlines)
        let totalLines = max(1, lines.count)
        guard shouldSplitSpans, entry.spanAnchors.count > 1 else {
            let anchor = entry.spanAnchors.first?.label
            return [SpanCandidateUnit(
                cid: "\(pack.packName)::\(entry.relativePath)",
                pack: pack,
                entry: entry,
                anchorLabel: anchor,
                startLine: 1,
                endLine: totalLines,
                windowText: fullText
            )]
        }

        var units: [SpanCandidateUnit] = []
        units.reserveCapacity(entry.spanAnchors.count)
        for span in entry.spanAnchors {
            let lo = max(0, min(lines.count - 1, span.startLine - 1))
            let hi = max(lo, min(lines.count - 1, span.endLine - 1))
            let slice = lines[lo...hi].joined(separator: "\n")
            guard !slice.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
            units.append(SpanCandidateUnit(
                cid: "\(pack.packName)::\(entry.relativePath)#\(span.label)",
                pack: pack,
                entry: entry,
                anchorLabel: span.label,
                startLine: lo + 1,
                endLine: hi + 1,
                windowText: slice
            ))
        }
        return units
    }

    /// Builds dense `.qvec` chunk inputs (`1` per span anchor on multi-span PDF/Markdown docs, `1` per code file).
    public static func buildDenseChunkInputs(
        discoveredFiles: [QWMemoryPack.DiscoveredFile]
    ) -> [(path: String, anchor: String?, snippet: String)] {
        var out: [(path: String, anchor: String?, snippet: String)] = []
        for f in discoveredFiles {
            let splitSpans = f.spanAnchors.count > 1 && (
                f.sourceKind == .pdfPageSpan
                    || (f.sourceKind == .markdown && f.byteCount > markdownMultiSpanMinBytes)
            )
            if splitSpans {
                let lines = f.content.components(separatedBy: .newlines)
                for span in f.spanAnchors {
                    let lo = max(0, min(lines.count - 1, span.startLine - 1))
                    let hi = max(lo, min(lines.count - 1, span.endLine - 1))
                    let body = lines[lo...hi].joined(separator: "\n")
                    guard !body.isEmpty else { continue }
                    let snippet = "\(f.relativePath) \(span.label)\n" + String(body.prefix(2048))
                    out.append((f.relativePath, span.label, snippet))
                }
            } else {
                let firstAnchor = f.spanAnchors.first?.label
                let snippet = f.relativePath + "\n" + String(f.content.prefix(2048))
                out.append((f.relativePath, firstAnchor, snippet))
            }
        }
        return out
    }

    /// Scores a single span unit lexically across path, symbols, anchor label, and window body identifiers/words.
    static func scoreSpanUnitLexical(query: String, unit: SpanCandidateUnit) -> Int {
        let lowerQuery = query.lowercased()
        let lowerRel = unit.entry.relativePath.lowercased()
        let fileName = (lowerRel as NSString).lastPathComponent
        let stem = (fileName as NSString).deletingPathExtension

        var score = 0
        if lowerQuery.contains(lowerRel) { score += 120 }
        else if lowerQuery.contains(fileName) { score += 95 }
        else if stem.count >= 3 && lowerQuery.contains(stem) { score += 65 }

        let queryWords = Set(
            lowerQuery
                .split { !$0.isLetter && !$0.isNumber && $0 != "_" && $0 != "." && $0 != "-" }
                .map(String.init)
                .filter { $0.count >= 3 }
        )
        let queryIdents = Set(
            query
                .split { !$0.isLetter && !$0.isNumber && $0 != "_" }
                .map(String.init)
                .filter { $0.count >= 4 }
        )
        guard !queryWords.isEmpty || !queryIdents.isEmpty else { return score }

        if let anchor = unit.anchorLabel?.lowercased() {
            if lowerQuery.contains(anchor) { score += 45 }
            for w in queryWords where anchor.contains(w) { score += 18 }
        }

        let lowerBody = unit.windowText.lowercased()
        let bodyWords = Set(
            lowerBody
                .split { !$0.isLetter && !$0.isNumber && $0 != "_" && $0 != "-" }
                .map(String.init)
        )
        for ident in queryIdents where unit.windowText.contains(ident) {
            score += 36
        }
        for w in queryWords {
            if bodyWords.contains(w) {
                score += 14
            } else if w.count >= 5 && lowerBody.contains(w) {
                score += 8
            }
        }
        return score
    }

    /// Computes 128d `e-means` cosine similarity directly on a span unit's text.
    static func scoreSpanUnitEmeans(queryVec: [Float], unit: SpanCandidateUnit) -> Float {
        let header = unit.entry.relativePath + " " + (unit.anchorLabel ?? "")
        let docVec = QWMemorySemanticIndex.embedText(header + "\n" + String(unit.windowText.prefix(4096)))
        guard docVec.count == queryVec.count, !docVec.isEmpty else { return 0 }
        var dot: Float = 0
        queryVec.withUnsafeBufferPointer { qBuf in
            docVec.withUnsafeBufferPointer { dBuf in
                guard let qPtr = qBuf.baseAddress, let dPtr = dBuf.baseAddress else { return }
                vDSP_dotpr(qPtr, 1, dPtr, 1, &dot, vDSP_Length(queryVec.count))
            }
        }
        return dot
    }

    /// Keeps full code files verbatim when fitting budget; slices `±80` lines around best matched declaration when oversized.
    static func fitWindowToBudget(
        query: String,
        unit: SpanCandidateUnit,
        maxWindowTokens: Int
    ) -> (text: String, startLine: Int, endLine: Int, truncated: Bool) {
        let rawEst = max(1, (unit.windowText.utf8.count + 3) / 4)
        guard rawEst > maxWindowTokens else {
            return (unit.windowText, unit.startLine, unit.endLine, false)
        }
        let lines = unit.windowText.components(separatedBy: .newlines)
        if unit.entry.sourceKind == .code,
           let graph = unit.pack.symbolGraph {
            let fileDecls = graph.declarations.filter { $0.relativePath == unit.entry.relativePath }
            let qLower = query.lowercased()
            if let bestDecl = fileDecls.max(by: { a, b in
                let sa = qLower.contains(a.symbol.lowercased()) ? 10 : 0
                let sb = qLower.contains(b.symbol.lowercased()) ? 10 : 0
                return sa < sb
            }), qLower.contains(bestDecl.symbol.lowercased()) {
                let lo = max(0, bestDecl.startLine - 20)
                let hi = min(lines.count - 1, bestDecl.endLine + 80)
                let sliced = lines[lo...hi].joined(separator: "\n")
                let maxUTF8 = max(512, maxWindowTokens * 3)
                let clipped = sliced.utf8.count > maxUTF8 ? String(sliced.prefix(maxUTF8)) : sliced
                let clippedLines = max(1, clipped.components(separatedBy: .newlines).count)
                return (clipped, lo + 1, lo + clippedLines, sliced.utf8.count > maxUTF8)
            }
        }
        let maxUTF8 = max(512, maxWindowTokens * 3)
        let clipped = String(unit.windowText.prefix(maxUTF8))
        let clippedCount = max(1, clipped.components(separatedBy: .newlines).count)
        return (clipped, unit.startLine, unit.startLine + clippedCount - 1, true)
    }
}
