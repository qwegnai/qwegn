// QWMemorySymbolGraph.swift — Multi-language declaration & 1-hop reference slice
// graph persisted at `.qwg/symbol_graph.json` for Cognitive Vault v2.

import Foundation
import QWCore

public struct QWMemorySymbolGraph: Codable, Sendable, Equatable {
    public struct DeclarationSlice: Codable, Sendable, Equatable {
        public let symbol: String
        public let kind: String
        public let relativePath: String
        public let startLine: Int
        public let endLine: Int
        public let signaturePreview: String
        public let packName: String?

        public init(
            symbol: String,
            kind: String,
            relativePath: String,
            startLine: Int,
            endLine: Int,
            signaturePreview: String,
            packName: String? = nil
        ) {
            self.symbol = symbol
            self.kind = kind
            self.relativePath = relativePath
            self.startLine = startLine
            self.endLine = endLine
            self.signaturePreview = signaturePreview
            self.packName = packName
        }

        public func withPackName(_ name: String) -> DeclarationSlice {
            DeclarationSlice(
                symbol: symbol,
                kind: kind,
                relativePath: relativePath,
                startLine: startLine,
                endLine: endLine,
                signaturePreview: signaturePreview,
                packName: name
            )
        }
    }

    public static let maxSliceLines: Int = 24

    public let declarations: [DeclarationSlice]
    /// Maps `relativePath -> [referenced symbol names]` declared in other files.
    public let fileReferences: [String: [String]]

    public init(declarations: [DeclarationSlice] = [], fileReferences: [String: [String]] = [:]) {
        self.declarations = declarations
        self.fileReferences = fileReferences
    }

    private static let typeKeywords: Set<String> = [
        "protocol", "interface", "trait", "struct", "class", "actor", "enum", "type", "record", "message", "service",
    ]
    private static let callableKeywords: Set<String> = [
        "func", "function", "fn", "def", "rpc", "kernel",
    ]
    private static let ignoredModifiers: Set<String> = [
        "public", "private", "internal", "fileprivate", "open", "protected", "final", "static",
        "async", "unsafe", "mut", "export", "default", "abstract", "override", "nonisolated",
    ]

    /// Builds a declaration + cross-file reference graph across code files.
    public static func build(files: [(relativePath: String, content: String)]) -> QWMemorySymbolGraph {
        var decls: [DeclarationSlice] = []
        var bySymbol: [String: [DeclarationSlice]] = [:]

        for item in files {
            let ext = (item.relativePath.lowercased() as NSString).pathExtension
            if ["md", "mdx", "rst", "txt", "pdf", "json", "yaml", "yml", "toml", "csv", "tsv"].contains(ext) {
                continue
            }
            let fileDecls = extractDeclarations(from: item.content, relativePath: item.relativePath)
            for d in fileDecls {
                decls.append(d)
                bySymbol[d.symbol, default: []].append(d)
            }
        }

        let knownSymbols = Set(bySymbol.keys)
        var refsByFile: [String: [String]] = [:]
        if !knownSymbols.isEmpty {
            for item in files {
                let tokens = extractIdentifiers(from: item.content)
                let matched = tokens.intersection(knownSymbols).filter { sym in
                    guard let defs = bySymbol[sym] else { return false }
                    return defs.contains { $0.relativePath != item.relativePath }
                }
                if !matched.isEmpty {
                    refsByFile[item.relativePath] = matched.sorted()
                }
            }
        }
        return QWMemorySymbolGraph(declarations: decls, fileReferences: refsByFile)
    }

    public func save(to rootURL: URL, fileName: String = "symbol_graph.json") throws {
        let dir = QWMemoryManifest.qwgDirectoryURL(in: rootURL)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        let data = try enc.encode(self)
        try data.write(to: QWMemoryManifest.symbolGraphURL(in: rootURL, fileName: fileName), options: .atomic)
    }

    public static func load(from rootURL: URL, fileName: String = "symbol_graph.json") -> QWMemorySymbolGraph? {
        let url = QWMemoryManifest.symbolGraphURL(in: rootURL, fileName: fileName)
        guard FileManager.default.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url) else {
            return nil
        }
        return try? JSONDecoder().decode(QWMemorySymbolGraph.self, from: data)
    }

    /// Expands 1-hop referenced type/callable declarations across mounted packs up to `maxSliceTokens` (`<= 25%` Vault budget).
    public static func expandOneHopSlices(
        query: String,
        primaryFiles: [(packName: String, relativePath: String, content: String)],
        mountedGraphs: [(packName: String, manifest: QWMemoryManifest, graph: QWMemorySymbolGraph)],
        alreadyRecalledKeys: Set<String>,
        excludePrimaryCortexPackName: String? = nil,
        maxSliceTokens: Int = 2048
    ) -> (slices: [DeclarationSlice], formattedBlock: String, estimatedTokens: Int) {
        guard !primaryFiles.isEmpty, !mountedGraphs.isEmpty, maxSliceTokens >= 64 else {
            return ([], "", 0)
        }
        var referencedCounts: [String: Int] = [:]
        for pf in primaryFiles {
            if let g = mountedGraphs.first(where: { $0.packName == pf.packName })?.graph,
               let precomputed = g.fileReferences[pf.relativePath], !precomputed.isEmpty {
                for sym in precomputed { referencedCounts[sym, default: 0] += 2 }
            }
            for ident in extractIdentifiers(from: pf.content) {
                referencedCounts[ident, default: 0] += 1
            }
        }
        guard !referencedCounts.isEmpty else { return ([], "", 0) }

        let queryLower = query.lowercased()
        struct Candidate {
            let slice: DeclarationSlice
            let score: Int
        }
        var candidates: [Candidate] = []
        var seenSliceKeys: Set<String> = []

        for mg in mountedGraphs {
            let cortexSet: Set<String> = (mg.packName == excludePrimaryCortexPackName)
                ? Set(mg.manifest.files.filter(\.inCortex).map(\.relativePath))
                : []
            for d in mg.graph.declarations {
                guard let refCount = referencedCounts[d.symbol], refCount > 0 else { continue }
                let fileKey = "\(mg.packName):\(d.relativePath)"
                if alreadyRecalledKeys.contains(fileKey) || alreadyRecalledKeys.contains(d.relativePath) {
                    continue
                }
                if cortexSet.contains(d.relativePath) { continue }
                let sliceKey = "\(fileKey)#\(d.symbol)#\(d.startLine)"
                if seenSliceKeys.contains(sliceKey) { continue }
                seenSliceKeys.insert(sliceKey)

                var score = min(refCount, 6) * 15
                if typeKeywords.contains(d.kind) { score += 35 }
                if queryLower.contains(d.symbol.lowercased()) { score += 40 }
                candidates.append(Candidate(slice: d.withPackName(mg.packName), score: score))
            }
        }
        guard !candidates.isEmpty else { return ([], "", 0) }
        candidates.sort { a, b in
            if a.score != b.score { return a.score > b.score }
            if a.slice.relativePath != b.slice.relativePath { return a.slice.relativePath < b.slice.relativePath }
            return a.slice.startLine < b.slice.startLine
        }

        var chosen: [DeclarationSlice] = []
        var blocks: [String] = []
        var usedTokens = 16
        for cand in candidates {
            let d = cand.slice
            let pfx = d.packName.map { "\($0):" } ?? ""
            let header = "[AST 1-HOP SLICE: \(pfx)\(d.relativePath)#L\(d.startLine)-L\(d.endLine) (\(d.kind) \(d.symbol))]"
            let snippet = header + "\n" + d.signaturePreview + (d.signaturePreview.hasSuffix("\n") ? "" : "\n")
            let est = max(1, (snippet.utf8.count + 3) / 4)
            if usedTokens + est > maxSliceTokens { continue }
            chosen.append(d)
            blocks.append(snippet)
            usedTokens += est
            if chosen.count >= 8 { break }
        }
        guard !chosen.isEmpty else { return ([], "", 0) }
        return (chosen, blocks.joined(separator: "\n"), usedTokens)
    }

    private static func extractDeclarations(from content: String, relativePath: String) -> [DeclarationSlice] {
        let lines = content.components(separatedBy: .newlines)
        guard !lines.isEmpty else { return [] }
        var out: [DeclarationSlice] = []
        var seen: Set<String> = []

        for idx in 0..<lines.count {
            let raw = lines[idx]
            let trimmed = raw.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("//") || trimmed.hasPrefix("/*") || trimmed.hasPrefix("*") || trimmed.hasPrefix("#") {
                continue
            }
            let indent = raw.prefix { $0 == " " || $0 == "\t" }.count
            if indent > 4 { continue }
            let words = trimmed.split { !$0.isLetter && !$0.isNumber && $0 != "_" }.map(String.init)
            guard words.count >= 2 else { continue }

            var matchedKind: String?
            var matchedSym: String?
            for wIdx in 0..<(words.count - 1) {
                let kw = words[wIdx].lowercased()
                let isType = typeKeywords.contains(kw)
                let isCall = callableKeywords.contains(kw) && indent == 0
                guard isType || isCall else { continue }
                var nextIdx = wIdx + 1
                while nextIdx < words.count && ignoredModifiers.contains(words[nextIdx].lowercased()) {
                    nextIdx += 1
                }
                if nextIdx < words.count {
                    let cand = words[nextIdx]
                    if cand.count >= 4 && !ignoredModifiers.contains(cand.lowercased()) {
                        matchedKind = kw
                        matchedSym = cand
                        break
                    }
                }
            }
            guard let kind = matchedKind, let sym = matchedSym, !seen.contains(sym) else { continue }
            seen.insert(sym)
            let endIdx = computeSliceEndIndex(lines: lines, startIdx: idx, isType: typeKeywords.contains(kind))
            let preview = lines[idx...endIdx].joined(separator: "\n")
            out.append(DeclarationSlice(
                symbol: sym,
                kind: kind,
                relativePath: relativePath,
                startLine: idx + 1,
                endLine: endIdx + 1,
                signaturePreview: preview
            ))
            if out.count >= 48 { break }
        }
        return out
    }

    private static func computeSliceEndIndex(lines: [String], startIdx: Int, isType: Bool) -> Int {
        let capIdx = min(lines.count - 1, startIdx + (isType ? maxSliceLines - 1 : 5))
        var braceBalance = 0
        var sawOpenBrace = false
        for i in startIdx...capIdx {
            let line = lines[i]
            for ch in line {
                if ch == "{" { braceBalance += 1; sawOpenBrace = true }
                else if ch == "}" { braceBalance -= 1 }
            }
            if !isType && (sawOpenBrace || line.trimmingCharacters(in: .whitespaces).hasSuffix(":")) && i >= startIdx + 1 {
                return i
            }
            if sawOpenBrace && braceBalance <= 0 {
                return i
            }
        }
        return capIdx
    }

    private static func extractIdentifiers(from text: String) -> Set<String> {
        var set: Set<String> = []
        for sub in text.split(whereSeparator: { !$0.isLetter && !$0.isNumber && $0 != "_" }) {
            guard sub.count >= 4, sub.count <= 64 else { continue }
            let s = String(sub)
            let isInteresting = (s.first?.isUppercase == true) || s.contains("_")
            if isInteresting { set.insert(s) }
        }
        return set
    }
}
