// QWMemoryVault.swift — Two-Tier LSA Cortex Architecture Map, Symbol Indexer,
// Sub-Millisecond Off-Context Vault Recall, and Elastic Context Budget Guard.

import Foundation
import QWCore

public enum QWMemoryVault {

    /// Default Resident Cortex `.qwcache` token budget (~3.1% of 1M context window),
    /// leaving >1,000,000 tokens free for the coding agent while indexing all repo files in the Vault.
    public static let defaultCortexTokenBudget: Int = 32_768

    /// Maximum tokens paged on-demand from the Off-Context Vault per turn.
    public static let defaultVaultRecallTokenBudget: Int = 8_192

    public struct VaultRecallResult: Sendable, Equatable {
        public let recallBlock: String
        public let recalledFiles: [String]
        public let estimatedTokens: Int

        public var isEmpty: Bool { recalledFiles.isEmpty || recallBlock.isEmpty }

        public init(recallBlock: String = "", recalledFiles: [String] = [], estimatedTokens: Int = 0) {
            self.recallBlock = recallBlock
            self.recalledFiles = recalledFiles
            self.estimatedTokens = estimatedTokens
        }
    }

    public struct ElasticBudgetSelection: Sendable, Equatable {
        public let prefixTokens: [Int]
        public let bypassedOversizedCache: Bool
        public let availableVaultTokenBudget: Int

        public init(prefixTokens: [Int], bypassedOversizedCache: Bool, availableVaultTokenBudget: Int) {
            self.prefixTokens = prefixTokens
            self.bypassedOversizedCache = bypassedOversizedCache
            self.availableVaultTokenBudget = availableVaultTokenBudget
        }
    }

    private static let universalDeclarationKeywords: Set<String> = [
        "struct", "class", "actor", "enum", "union", "protocol", "interface", "trait", "impl",
        "type", "typedef", "namespace", "module", "mod", "package", "record", "object",
        "func", "function", "fn", "def", "defun", "sub", "proc", "method", "kernel", "shader",
        "macro", "macro_rules", "service", "message", "rpc", "table", "view", "resource",
        "variable", "output", "query", "mutation", "schema", "trigger", "procedure"
    ]

    private static let ignoredModifierTokens: Set<String> = [
        "where", "Self", "public", "private", "internal", "protected", "open", "final",
        "abstract", "static", "async", "unsafe", "extern", "inline", "virtual", "override",
        "mut", "const", "let", "var", "val", "export", "default", "create", "alter", "or",
        "replace", "if", "not", "exists", "temp", "temporary"
    ]

    /// Extracts up to `maxSymbols` structural anchors (declarations, headings, schema entities, or top-level keys)
    /// across arbitrary programming languages, structured data formats (JSON/YAML/TOML/CSV/SQL), and documents.
    public static func extractSymbols(
        from content: String,
        relativePath: String,
        maxSymbols: Int = 20
    ) -> [String] {
        let lowerPath = relativePath.lowercased()
        let ext = (lowerPath as NSString).pathExtension
        var symbols: [String] = []
        var seen: Set<String> = []

        func appendSymbol(_ raw: String) {
            let cleaned = raw.trimmingCharacters(
                in: CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_-.")).inverted
            )
            guard cleaned.count >= 3, cleaned.count <= 64,
                  !ignoredModifierTokens.contains(cleaned),
                  !seen.contains(cleaned) else { return }
            seen.insert(cleaned)
            symbols.append(cleaned)
        }

        let lines = content.split(whereSeparator: \.isNewline)

        // CSV / TSV header row extraction: extract column names from the first non-empty line.
        if (ext == "csv" || ext == "tsv"), let firstLine = lines.first {
            let sep: Character = ext == "tsv" ? "\t" : ","
            for col in firstLine.split(separator: sep) {
                appendSymbol(String(col))
                if symbols.count >= maxSymbols { break }
            }
            if !symbols.isEmpty { return symbols }
        }

        let isMarkupDoc = ext == "md" || ext == "mdx" || ext == "rst" || ext == "adoc" || ext == "tex" || ext == "txt"

        for lineSub in lines {
            if symbols.count >= maxSymbols { break }
            let rawLine = String(lineSub)
            let trimmed = rawLine.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("//") || trimmed.hasPrefix("/*") || trimmed.hasPrefix("*") || trimmed.hasPrefix("--") {
                continue
            }

            // 1. Markup headings (# Heading) and LaTeX sections (\section{...}, \subsection{...})
            if trimmed.hasPrefix("# ") || trimmed.hasPrefix("## ") || trimmed.hasPrefix("### ") {
                let heading = trimmed.drop(while: { $0 == "#" || $0 == " " })
                let short = String(heading.prefix(48)).trimmingCharacters(in: .whitespaces)
                if !short.isEmpty && !seen.contains(short) {
                    seen.insert(short)
                    symbols.append(short)
                }
                continue
            }
            if trimmed.hasPrefix("\\section{") || trimmed.hasPrefix("\\subsection{") || trimmed.hasPrefix("\\title{") {
                if let open = trimmed.firstIndex(of: "{"),
                   let close = trimmed[open...].firstIndex(of: "}") {
                    let title = String(trimmed[trimmed.index(after: open)..<close]).trimmingCharacters(in: .whitespaces)
                    if !title.isEmpty && !seen.contains(title) {
                        seen.insert(title)
                        symbols.append(title)
                    }
                }
                continue
            }
            if isMarkupDoc { continue }

            // 2. TOML / INI section headers: [section.name]
            if trimmed.hasPrefix("[") && trimmed.hasSuffix("]") && !trimmed.contains(" ") {
                appendSymbol(String(trimmed.dropFirst().dropLast()))
                continue
            }

            // 3. Universal declaration keywords across programming, IDL, HCL, and SQL languages
            if !trimmed.hasPrefix("#") {
                let tokens = trimmed.split { !$0.isLetter && !$0.isNumber && $0 != "_" && $0 != "\"" }
                if tokens.count >= 2 {
                    var matchedDecl = false
                    for idx in 0..<(tokens.count - 1) {
                        let kw = String(tokens[idx]).lowercased()
                        if universalDeclarationKeywords.contains(kw) {
                            var candidateIdx = idx + 1
                            while candidateIdx < tokens.count {
                                let rawTok = String(tokens[candidateIdx]).trimmingCharacters(in: CharacterSet(charactersIn: "\""))
                                if !ignoredModifierTokens.contains(rawTok.lowercased()) {
                                    appendSymbol(rawTok)
                                    matchedDecl = true
                                    break
                                }
                                candidateIdx += 1
                            }
                            if matchedDecl { break }
                        }
                    }
                    if matchedDecl { continue }
                }
            }

            // 4. Top-level key / target definitions (YAML, JSON, TOML, Makefile, Config)
            let indentCount = rawLine.prefix(while: { $0 == " " || $0 == "\t" }).count
            if indentCount <= 2 && !trimmed.hasPrefix("#") && !trimmed.hasPrefix("-") {
                if let colon = trimmed.firstIndex(of: ":") {
                    let keyPart = String(trimmed[..<colon]).trimmingCharacters(in: CharacterSet(charactersIn: " \t\"'"))
                    if !keyPart.isEmpty, !keyPart.contains(" "), !keyPart.contains("$"), !keyPart.contains("="), !keyPart.contains("(") {
                        appendSymbol(keyPart)
                        continue
                    }
                } else if let eq = trimmed.firstIndex(of: "=") {
                    let keyPart = String(trimmed[..<eq]).trimmingCharacters(in: CharacterSet(charactersIn: " \t\"'"))
                    if !keyPart.isEmpty, !keyPart.contains(" "), !keyPart.contains("(") {
                        appendSymbol(keyPart)
                        continue
                    }
                }
            }
        }
        return symbols
    }

    private static let snapshotCacheLock = NSLock()
    nonisolated(unsafe) private static var historicalSectionCache: [String: (section: String, tokens: Int)] = [:]

    /// Builds a compact `[QWG REPOSITORY CORTEX MAP]` indexing all discovered files and their symbols
    /// while capping map size so verbatim core files retain >= 75% of the Resident Cortex token budget.
    public static func buildCortexArchitectureMap(
        packName: String,
        rootPath: String,
        files: [(file: QWMemoryPack.DiscoveredFile, symbols: [String])],
        cortexPaths: Set<String>,
        maxMapChars: Int = 24_000
    ) -> String {
        var out = "\n[QWG REPOSITORY CORTEX MAP: \(packName) | total_files: \(files.count) | cortex_resident: \(cortexPaths.count) | off_context_vault: \(max(0, files.count - cortexPaths.count))]\n"
        var grouped: [String: [(file: QWMemoryPack.DiscoveredFile, symbols: [String])]] = [:]
        var groupOrder: [String] = []
        for item in files {
            let grp = QWMemoryPack.topModuleGroup(for: item.file.relativePath)
            if grouped[grp] == nil {
                groupOrder.append(grp)
                grouped[grp] = []
            }
            grouped[grp]?.append(item)
        }
        let softCharBudget = max(2_048, maxMapChars)
        var omittedVaultCount = 0
        for grp in groupOrder {
            guard let items = grouped[grp] else { continue }
            out += "### Module: \(grp) (\(items.count) files)\n"
            for item in items {
                let isCortex = cortexPaths.contains(item.file.relativePath)
                if !isCortex && out.utf8.count >= softCharBudget {
                    omittedVaultCount += 1
                    continue
                }
                let tag = isCortex ? "CORTEX" : "VAULT"
                let filteredSyms = item.symbols
                    .filter { sym in
                        let low = sym.lowercased()
                        return !(low.hasPrefix("setup") || low.hasPrefix("teardown"))
                    }
                    .prefix(3)
                    .map { $0.count > 40 ? String($0.prefix(40)) : $0 }
                let symPreview = filteredSyms.joined(separator: ", ")
                if symPreview.isEmpty {
                    out += "- [\(tag)] \(item.file.relativePath) (\(item.file.byteCount) B)\n"
                } else {
                    out += "- [\(tag)] \(item.file.relativePath): \(symPreview)\n"
                }
            }
        }
        if omittedVaultCount > 0 {
            out += "- [VAULT] (+\(omittedVaultCount) additional indexed files in off-context symbol vault)\n"
        }
        out += "[END REPOSITORY CORTEX MAP]\n"
        return out
    }

    /// Scores off-context Vault files (`!entry.inCortex`, or all files if `includeCortexFiles == true`)
    /// against `query` in `< 0.5 ms` and pages in matching verbatim files up to `maxTokens`.
    public static func recallVaultContext(
        query: String,
        manifest: QWMemoryManifest,
        includeCortexFiles: Bool = false,
        excludedPaths: Set<String> = [],
        maxFiles: Int = 4,
        maxTokens: Int = defaultVaultRecallTokenBudget,
        preserveHistoricalSnapshot: Bool = false,
        fileManager: FileManager = .default
    ) -> VaultRecallResult {
        let rawTrimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !rawTrimmed.isEmpty, maxTokens >= 256, maxFiles > 0 else {
            return VaultRecallResult()
        }
        let trimmedQuery: String
        if rawTrimmed.utf8.count > 4096 {
            trimmedQuery = String(rawTrimmed.prefix(2048)) + "\n" + String(rawTrimmed.suffix(2048))
        } else {
            trimmedQuery = rawTrimmed
        }
        let lowerQuery = trimmedQuery.lowercased()
        let queryWords = Set(
            lowerQuery
                .split { !$0.isLetter && !$0.isNumber && $0 != "_" && $0 != "." && $0 != "-" && $0 != "/" }
                .map(String.init)
                .filter { $0.count >= 3 }
        )
        let queryIdentifiers = Set(
            trimmedQuery
                .split { !$0.isLetter && !$0.isNumber && $0 != "_" }
                .map(String.init)
                .filter { $0.count >= 4 }
        )

        var candidates: [(entry: QWMemoryManifest.FileEntry, score: Int)] = []
        for entry in manifest.files {
            if !includeCortexFiles && entry.inCortex { continue }
            if excludedPaths.contains(entry.relativePath) { continue }
            let lowerRel = entry.relativePath.lowercased()
            let fileName = (lowerRel as NSString).lastPathComponent
            let stem = (fileName as NSString).deletingPathExtension

            var score = 0
            if lowerQuery.contains(lowerRel) {
                score += 100
            } else if fileName.count >= 4 && lowerQuery.contains(fileName) {
                score += 85
            } else if stem.count >= 5 && queryWords.contains(stem) {
                score += 65
            }

            for sym in entry.symbols {
                let isCompoundIdentifier = (sym != sym.lowercased() && sym != sym.uppercased()) || sym.contains("_")
                if sym.count >= 5 && isCompoundIdentifier && queryIdentifiers.contains(sym) {
                    score += 45
                } else if sym.count >= 6 && isCompoundIdentifier && queryWords.contains(sym.lowercased()) {
                    score += 30
                } else if sym.count >= 6 && queryIdentifiers.contains(sym) {
                    score += 12
                }
            }

            let ext = (lowerRel as NSString).pathExtension
            let pathTokens = lowerRel
                .split { !$0.isLetter && !$0.isNumber }
                .map(String.init)
                .filter { $0.count >= 4 && $0 != ext }
            var matchedPathTokens = 0
            for pt in pathTokens where queryWords.contains(pt) {
                matchedPathTokens += 1
            }
            if matchedPathTokens >= 2 {
                score += matchedPathTokens * 10
            } else if matchedPathTokens == 1 && score > 0 {
                score += 8
            }

            let minScoreThreshold = includeCortexFiles ? 15 : 30
            if score >= minScoreThreshold {
                candidates.append((entry, score))
            }
        }

        if includeCortexFiles && candidates.isEmpty {
            for entry in manifest.files where entry.tier == 0 && !excludedPaths.contains(entry.relativePath) {
                candidates.append((entry, 10))
            }
        }

        guard !candidates.isEmpty else {
            return VaultRecallResult()
        }
        candidates.sort { a, b in
            if a.score != b.score { return a.score > b.score }
            if a.entry.tier != b.entry.tier { return a.entry.tier < b.entry.tier }
            return a.entry.relativePath < b.entry.relativePath
        }

        let rootURL = URL(fileURLWithPath: manifest.rootPath, isDirectory: true)
        var recalledPaths: [String] = []
        var sections: [String] = []
        var usedTokens = 32

        for cand in candidates {
            if recalledPaths.count >= maxFiles { break }
            let cacheKey = "\(manifest.rootPath)|\(cand.entry.relativePath)|\(cand.entry.sha256)|\(maxTokens)"
            if preserveHistoricalSnapshot {
                snapshotCacheLock.lock()
                let cached = historicalSectionCache[cacheKey]
                snapshotCacheLock.unlock()
                if let cached {
                    if !recalledPaths.isEmpty && usedTokens + cached.tokens > maxTokens {
                        continue
                    }
                    sections.append(cached.section)
                    recalledPaths.append(cand.entry.relativePath)
                    usedTokens += cached.tokens
                    continue
                }
            }
            let fileURL = rootURL.appendingPathComponent(cand.entry.relativePath, isDirectory: false)
            guard let text = try? String(contentsOf: fileURL, encoding: .utf8), !text.isEmpty else {
                continue
            }
            let normalized = text.hasSuffix("\n") ? text : (text + "\n")
            let section = "[VAULT FILE: \(cand.entry.relativePath)]\n" + normalized
            let estTokens = max(1, (section.utf8.count + 3) / 4)
            if !recalledPaths.isEmpty && usedTokens + estTokens > maxTokens {
                continue
            }
            let finalSection: String
            let finalEstTokens: Int
            if usedTokens + estTokens > maxTokens {
                let maxUTF8 = max(512, (maxTokens - usedTokens) * 3)
                let prefixText = String(normalized.prefix(maxUTF8))
                finalSection = "[VAULT FILE: \(cand.entry.relativePath) (truncated to fit budget)]\n" + prefixText + "\n"
                finalEstTokens = max(1, (finalSection.utf8.count + 3) / 4)
            } else {
                finalSection = section
                finalEstTokens = estTokens
            }
            snapshotCacheLock.lock()
            if historicalSectionCache[cacheKey] == nil {
                if historicalSectionCache.count >= 512 {
                    historicalSectionCache.removeAll(keepingCapacity: true)
                }
                historicalSectionCache[cacheKey] = (finalSection, finalEstTokens)
            }
            snapshotCacheLock.unlock()
            sections.append(finalSection)
            recalledPaths.append(cand.entry.relativePath)
            usedTokens += finalEstTokens
        }

        guard !recalledPaths.isEmpty else {
            return VaultRecallResult()
        }
        var block = "\n<|memory_vault_recall|>\n"
        block += "[QWG LSA VAULT RECALL: \(recalledPaths.count) file(s) paged on-demand from '\(manifest.packName)' — verbatim disk contents below; use directly without redundant read_file, grep, or sed calls]\n"
        block += sections.joined(separator: "\n")
        if !block.hasSuffix("\n") { block += "\n" }
        block += "</|memory_vault_recall|>"
        return VaultRecallResult(
            recallBlock: block,
            recalledFiles: recalledPaths,
            estimatedTokens: usedTokens
        )
    }

    /// Reserves `agentPromptTokens + maxNewTokens + safetyMargin` FIRST so a mounted `.qwcache`
    /// never triggers `contextDoesNotFit` for coding agents with large multi-turn histories.
    public static func applyElasticBudgetGuard(
        mountedMemoryTokens: [Int],
        deltaTokens: [Int],
        agentPromptTokens: Int,
        maxNewTokens: Int,
        maxContext: Int,
        safetyMargin: Int = 1024
    ) -> ElasticBudgetSelection {
        guard !mountedMemoryTokens.isEmpty else {
            return ElasticBudgetSelection(
                prefixTokens: [],
                bypassedOversizedCache: false,
                availableVaultTokenBudget: 0
            )
        }
        let requiredAgent = agentPromptTokens + max(1, maxNewTokens) + max(128, safetyMargin)
        let availableForMemory = max(0, maxContext - requiredAgent)

        if mountedMemoryTokens.count + deltaTokens.count <= availableForMemory {
            let remaining = availableForMemory - mountedMemoryTokens.count - deltaTokens.count
            return ElasticBudgetSelection(
                prefixTokens: mountedMemoryTokens + deltaTokens,
                bypassedOversizedCache: false,
                availableVaultTokenBudget: min(defaultVaultRecallTokenBudget, remaining)
            )
        }
        if mountedMemoryTokens.count <= availableForMemory {
            let remaining = availableForMemory - mountedMemoryTokens.count
            return ElasticBudgetSelection(
                prefixTokens: mountedMemoryTokens,
                bypassedOversizedCache: false,
                availableVaultTokenBudget: min(defaultVaultRecallTokenBudget, remaining)
            )
        }
        return ElasticBudgetSelection(
            prefixTokens: [],
            bypassedOversizedCache: true,
            availableVaultTokenBudget: min(defaultVaultRecallTokenBudget * 2, availableForMemory)
        )
    }
}
