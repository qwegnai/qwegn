// QWMemoryPack.swift — Unified Codebase & Knowledge Memory Pack (.qwg/manifest.json + .qwcache).
// Implements domain-agnostic structural depth & stability discovery, 512-token LSA superblock packing,
// live working-tree delta overlays (<|memory_delta|>), and workspace auto-discovery.

import CryptoKit
import Foundation
import QWCore

public enum QWMemoryPackError: LocalizedError, Sendable {
    case directoryNotFound(String)
    case noMatchingCacheArtifact(modelName: String, kvMode: String?)
    case cacheFileMissing(String)

    public var errorDescription: String? {
        switch self {
        case .directoryNotFound(let path):
            return "Directory not found: '\(path)'"
        case .noMatchingCacheArtifact(let modelName, let kvMode):
            return "No matching .qwcache artifact in .qwg/manifest.json for model '\(modelName)' (kvMode=\(kvMode ?? "any"))"
        case .cacheFileMissing(let path):
            return "Memory pack .qwcache file not found at '\(path)'"
        }
    }
}

public struct QWMemoryManifest: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 1

    public struct FileEntry: Codable, Sendable, Equatable {
        public let relativePath: String
        public let tier: Int
        public let sha256: String
        public let mtimeEpochSec: Int64
        public let byteCount: Int
        public let tokenOffset: Int
        public let tokenCount: Int
        public let inCortex: Bool
        public let symbols: [String]

        public init(
            relativePath: String,
            tier: Int,
            sha256: String,
            mtimeEpochSec: Int64,
            byteCount: Int,
            tokenOffset: Int,
            tokenCount: Int,
            inCortex: Bool = true,
            symbols: [String] = []
        ) {
            self.relativePath = relativePath
            self.tier = tier
            self.sha256 = sha256
            self.mtimeEpochSec = mtimeEpochSec
            self.byteCount = byteCount
            self.tokenOffset = tokenOffset
            self.tokenCount = tokenCount
            self.inCortex = inCortex
            self.symbols = symbols
        }

        private enum CodingKeys: String, CodingKey {
            case relativePath, tier, sha256, mtimeEpochSec, byteCount, tokenOffset, tokenCount, inCortex, symbols
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.relativePath = try c.decode(String.self, forKey: .relativePath)
            self.tier = try c.decode(Int.self, forKey: .tier)
            self.sha256 = try c.decode(String.self, forKey: .sha256)
            self.mtimeEpochSec = try c.decode(Int64.self, forKey: .mtimeEpochSec)
            self.byteCount = try c.decode(Int.self, forKey: .byteCount)
            self.tokenOffset = try c.decode(Int.self, forKey: .tokenOffset)
            self.tokenCount = try c.decode(Int.self, forKey: .tokenCount)
            self.inCortex = try c.decodeIfPresent(Bool.self, forKey: .inCortex) ?? true
            self.symbols = try c.decodeIfPresent([String].self, forKey: .symbols) ?? []
        }
    }

    public struct CacheArtifact: Codable, Sendable, Equatable {
        public let modelName: String
        public let geometryDigest: String
        public let kvMode: String
        public let lsaMode: String
        public let lsaSidecarDigest: String?
        public let tokenCount: Int
        public let cacheFileName: String
        public let cacheFileBytes: Int
        public let createdAtEpochSec: Int64

        public init(
            modelName: String,
            geometryDigest: String,
            kvMode: String,
            lsaMode: String,
            lsaSidecarDigest: String? = nil,
            tokenCount: Int,
            cacheFileName: String,
            cacheFileBytes: Int,
            createdAtEpochSec: Int64 = Int64(Date().timeIntervalSince1970)
        ) {
            self.modelName = modelName
            self.geometryDigest = geometryDigest
            self.kvMode = kvMode
            self.lsaMode = lsaMode
            self.lsaSidecarDigest = lsaSidecarDigest
            self.tokenCount = tokenCount
            self.cacheFileName = cacheFileName
            self.cacheFileBytes = cacheFileBytes
            self.createdAtEpochSec = createdAtEpochSec
        }
    }

    public var schemaVersion: Int
    public var packName: String
    public var rootPath: String
    public var gitHeadSHA: String?
    public var clusterBlockTokens: Int
    public var includeTests: Bool
    public var files: [FileEntry]
    public var caches: [CacheArtifact]
    public var semanticIndex: QWMemorySemanticIndex?

    public init(
        schemaVersion: Int = Self.currentSchemaVersion,
        packName: String,
        rootPath: String,
        gitHeadSHA: String? = nil,
        clusterBlockTokens: Int = 512,
        includeTests: Bool = false,
        files: [FileEntry] = [],
        caches: [CacheArtifact] = [],
        semanticIndex: QWMemorySemanticIndex? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.packName = packName
        self.rootPath = rootPath
        self.gitHeadSHA = gitHeadSHA
        self.clusterBlockTokens = clusterBlockTokens
        self.includeTests = includeTests
        self.files = files
        self.caches = caches
        self.semanticIndex = semanticIndex
    }

    public static func qwgDirectoryURL(in rootURL: URL) -> URL {
        rootURL.standardizedFileURL.appendingPathComponent(".qwg", isDirectory: true)
    }

    public static func manifestURL(in rootURL: URL) -> URL {
        qwgDirectoryURL(in: rootURL).appendingPathComponent("manifest.json", isDirectory: false)
    }

    public static func load(from rootURL: URL) throws -> QWMemoryManifest {
        let url = manifestURL(in: rootURL)
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(QWMemoryManifest.self, from: data)
    }

    public func save(to rootURL: URL) throws {
        let dir = Self.qwgDirectoryURL(in: rootURL)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: Self.manifestURL(in: rootURL), options: .atomic)
    }

    public func cacheFileURL(for artifact: CacheArtifact, rootURL: URL? = nil) -> URL {
        let base = rootURL ?? URL(fileURLWithPath: rootPath, isDirectory: true)
        return Self.qwgDirectoryURL(in: base).appendingPathComponent(artifact.cacheFileName, isDirectory: false)
    }

    /// Selects the matching `.qwcache` artifact for the given model identity or geometry digest.
    public func matchingCache(
        modelName: String? = nil,
        geometryDigest: String? = nil,
        kvMode: String? = nil
    ) -> CacheArtifact? {
        let filteredByKV: [CacheArtifact]
        if let kv = kvMode?.lowercased(), !kv.isEmpty {
            let exactKV = caches.filter { $0.kvMode.lowercased() == kv }
            filteredByKV = exactKV.isEmpty ? caches : exactKV
        } else {
            filteredByKV = caches
        }
        if let digest = geometryDigest, !digest.isEmpty,
           let match = filteredByKV.first(where: { $0.geometryDigest == digest }) {
            return match
        }
        if let name = modelName, !name.isEmpty,
           let match = filteredByKV.first(where: { $0.modelName.caseInsensitiveCompare(name) == .orderedSame }) {
            return match
        }
        return filteredByKV.first
    }
}

public enum QWMemoryPack {
    public struct DiscoveredFile: Sendable, Equatable {
        public let relativePath: String
        public let tier: Int
        public let sha256: String
        public let mtimeEpochSec: Int64
        public let byteCount: Int
        public let content: String
    }

    public struct BuiltEnvelope: Sendable {
        public let systemMessageContent: String
        public let fileEntries: [QWMemoryManifest.FileEntry]
        public let totalBytes: Int
        public var cortexFileCount: Int { fileEntries.filter(\.inCortex).count }
        public var vaultFileCount: Int { fileEntries.filter { !$0.inCortex }.count }
    }

    public struct DeltaOverlay: Sendable, Equatable {
        public let deltaBlock: String
        public let modifiedFiles: [String]
        public let deletedFiles: [String]

        public var hasChanges: Bool {
            !modifiedFiles.isEmpty || !deletedFiles.isEmpty
        }
    }

    private static let excludedDirectories: Set<String> = [
        ".git", ".qwg", ".build", ".swiftpm", "node_modules", ".venv", ".venv-oracle",
        "venv", "__pycache__", "out", "build", "dist", "target", "Pods", "DerivedData",
        ".worktrees", ".idea", ".vscode", ".next", ".turbo"
    ]

    private static let excludedFileNames: Set<String> = [
        "package.resolved", "cargo.lock", "pnpm-lock.yaml", "package-lock.json",
        "yarn.lock", "poetry.lock", "uv.lock", "gemfile.lock", "composer.lock",
        ".ds_store"
    ]

    private static let excludedExtensions: Set<String> = [
        "qwp", "qlsa", "qwcache", "safetensors", "bin", "metallib", "air", "o", "a",
        "dylib", "so", "dll", "exe", "png", "jpg", "jpeg", "gif", "webp", "ico", "svg",
        "mp4", "mov", "avi", "mp3", "wav", "pdf", "zip", "tar", "gz", "tgz", "bz2",
        "xz", "7z", "npy", "npz", "pt", "pth", "onnx", "gguf", "mlx", "parquet", "db", "sqlite"
    ]

    /// Maximum single-file size packed into a memory envelope (256 KiB).
    public static let maxSingleFileBytes = 256 * 1024

    /// Discovers, filters, and sorts workspace files by universal structural depth and temporal stability.
    public static func discoverFiles(
        in rootURL: URL,
        includeTests: Bool = false,
        maxFileBytes: Int = maxSingleFileBytes
    ) throws -> [DiscoveredFile] {
        let root = rootURL.standardizedFileURL
        let ignoreMatcher = QWMemoryIgnoreMatcher(rootURL: root)
        let candidateRelPaths = listCandidateRelativePaths(in: root, ignoreMatcher: ignoreMatcher)
        let churnEpochByPath = queryGitLastCommitEpochs(in: root)
        let fm = FileManager.default

        var rawItems: [(file: DiscoveredFile, commitEpoch: Int64)] = []
        for relPath in candidateRelPaths {
            if shouldIgnore(relativePath: relPath, ignoreMatcher: ignoreMatcher) { continue }
            if !includeTests && isConventionalTestFile(relativePath: relPath) { continue }
            let depth = structuralDepth(relativePath: relPath)
            let fileURL = root.appendingPathComponent(relPath, isDirectory: false)
            guard let attrs = try? fm.attributesOfItem(atPath: fileURL.path),
                  let fileType = attrs[.type] as? FileAttributeType,
                  fileType == .typeRegular,
                  let size = (attrs[.size] as? NSNumber)?.intValue,
                  size > 0, size <= maxFileBytes else {
                continue
            }
            guard let data = try? Data(contentsOf: fileURL),
                  !data.contains(0),
                  let text = String(data: data, encoding: .utf8) else {
                continue
            }
            let mtime = Int64((attrs[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            let discovered = DiscoveredFile(
                relativePath: relPath,
                tier: depth,
                sha256: digest,
                mtimeEpochSec: mtime,
                byteCount: size,
                content: text
            )
            let commitEpoch = churnEpochByPath[relPath] ?? mtime
            rawItems.append((discovered, commitEpoch))
        }

        rawItems.sort { a, b in
            let aTest = isConventionalTestFile(relativePath: a.file.relativePath)
            let bTest = isConventionalTestFile(relativePath: b.file.relativePath)
            if aTest != bTest { return !aTest }
            if a.file.tier != b.file.tier { return a.file.tier < b.file.tier }
            if a.file.tier > 0 && a.commitEpoch != b.commitEpoch {
                return a.commitEpoch < b.commitEpoch
            }
            return a.file.relativePath < b.file.relativePath
        }
        return rawItems.map(\.file)
    }

    /// Packs discovered files into the `<|memory|>...</|memory|>` envelope with 512-token block alignment.
    public static func buildEnvelope(
        packName: String,
        rootPath: String,
        files: [DiscoveredFile],
        clusterBlockTokens: Int = 512,
        kvMode: String = "q4",
        maxTokens: Int? = nil
    ) -> BuiltEnvelope {
        buildEnvelope(
            packName: packName,
            rootPath: rootPath,
            files: files,
            clusterBlockTokens: clusterBlockTokens,
            kvMode: kvMode,
            maxTokens: maxTokens,
            tokenCounter: { max(1, ($0.utf8.count + 3) / 4) }
        )
    }

    /// Packs discovered files into the `<|memory|>...</|memory|>` envelope with a custom token counter.
    public static func buildEnvelope(
        packName: String,
        rootPath: String,
        files: [DiscoveredFile],
        clusterBlockTokens: Int = 512,
        kvMode: String = "q4",
        maxTokens: Int? = nil,
        tokenCounter: (String) throws -> Int
    ) rethrows -> BuiltEnvelope {
        let headerPrefix = "Memory<|memory|>\n[QWG MEMORY PACK: \(packName) | root: \(rootPath)]\n"
        var body = ""
        var runningTokens = try tokenCounter(headerPrefix)
        var entries: [QWMemoryManifest.FileEntry] = []
        var totalBytes = 0
        var lastTopGroup = ""
        let blockSize = max(64, clusterBlockTokens)
        let tokenCap = (maxTokens ?? 0) > 0 ? maxTokens! : nil

        struct PreparedSection {
            let file: DiscoveredFile
            let symbols: [String]
            let section: String
            let sectionTokens: Int
        }
        var prepared: [PreparedSection] = []
        var totalRawTokens = runningTokens
        for file in files {
            let symbols = QWMemoryVault.extractSymbols(from: file.content, relativePath: file.relativePath)
            let normalizedBody = file.content.hasSuffix("\n") ? file.content : (file.content + "\n")
            let section = "\n[FILE: \(file.relativePath)]\n" + normalizedBody
            let sectionTokens = try tokenCounter(section)
            prepared.append(PreparedSection(file: file, symbols: symbols, section: section, sectionTokens: sectionTokens))
            totalRawTokens += sectionTokens
        }

        let needsTwoTierVault = tokenCap != nil && (totalRawTokens + 96 > tokenCap!)
        let mapCharBudget = min(24_000, max(2_048, (tokenCap ?? 32_768) * 3 / 4))
        var reservedMapTokens = 0
        if needsTwoTierVault {
            let draftMap = QWMemoryVault.buildCortexArchitectureMap(
                packName: packName,
                rootPath: rootPath,
                files: prepared.map { ($0.file, $0.symbols) },
                cortexPaths: [],
                maxMapChars: mapCharBudget
            )
            let draftTokens = try tokenCounter(draftMap)
            reservedMapTokens = draftTokens
            runningTokens += draftTokens
        }

        var cortexPaths: Set<String> = []
        for item in prepared {
            let file = item.file
            let sectionTokens = item.sectionTokens
            let exceedsPerFileCap = (tokenCap ?? 0) >= 4096 && sectionTokens > max(2048, (tokenCap ?? 0) / 5)
            if let tokenCap, !cortexPaths.isEmpty, (runningTokens + sectionTokens + 96 > tokenCap || exceedsPerFileCap) {
                entries.append(
                    QWMemoryManifest.FileEntry(
                        relativePath: file.relativePath,
                        tier: file.tier,
                        sha256: file.sha256,
                        mtimeEpochSec: file.mtimeEpochSec,
                        byteCount: file.byteCount,
                        tokenOffset: -1,
                        tokenCount: sectionTokens,
                        inCortex: false,
                        symbols: item.symbols
                    )
                )
                continue
            }
            let group = topModuleGroup(for: file.relativePath)
            if !lastTopGroup.isEmpty && group != lastTopGroup {
                let offsetInBlock = runningTokens % blockSize
                let remaining = blockSize - offsetInBlock
                if remaining > 0 && remaining <= 48 {
                    let padLine = "\n" + String(repeating: "# ", count: max(1, remaining / 2)) + "\n"
                    let padTokens = try tokenCounter(padLine)
                    if tokenCap == nil || runningTokens + padTokens + sectionTokens + 96 <= tokenCap! {
                        body += padLine
                        runningTokens += padTokens
                    }
                }
            }
            lastTopGroup = group

            let startToken = runningTokens
            body += item.section
            runningTokens += sectionTokens
            totalBytes += file.byteCount
            cortexPaths.insert(file.relativePath)
            entries.append(
                QWMemoryManifest.FileEntry(
                    relativePath: file.relativePath,
                    tier: file.tier,
                    sha256: file.sha256,
                    mtimeEpochSec: file.mtimeEpochSec,
                    byteCount: file.byteCount,
                    tokenOffset: startToken,
                    tokenCount: sectionTokens,
                    inCortex: true,
                    symbols: item.symbols
                )
            )
        }

        if reservedMapTokens > 0 {
            let finalMap = QWMemoryVault.buildCortexArchitectureMap(
                packName: packName,
                rootPath: rootPath,
                files: prepared.map { ($0.file, $0.symbols) },
                cortexPaths: cortexPaths,
                maxMapChars: mapCharBudget
            )
            body = finalMap + body
        }

        let vaultCount = entries.count - cortexPaths.count
        let header = "Memory<|memory|>\n[QWG MEMORY PACK: \(packName) | root: \(rootPath) | files: \(entries.count) | cortex: \(cortexPaths.count) | vault: \(vaultCount)]\n"
        let footer = "</|memory|>Memory\n"
            + agentAwarenessNotice(
                packName: packName,
                fileCount: entries.count,
                cortexCount: cortexPaths.count,
                tokenCount: runningTokens,
                kvMode: kvMode
            )
        let out = header + body + footer
        return BuiltEnvelope(systemMessageContent: out, fileEntries: entries, totalBytes: totalBytes)
    }

    /// Reconstructs the deterministic `<|memory|>` envelope from the files recorded in `manifest`.
    public static func renderEnvelopeFromManifest(
        _ manifest: QWMemoryManifest,
        kvMode: String? = nil,
        includeDelta: Bool = true
    ) -> (envelope: String, delta: DeltaOverlay) {
        let rootURL = URL(fileURLWithPath: manifest.rootPath, isDirectory: true)
        let fm = FileManager.default
        var discovered: [DiscoveredFile] = []
        for entry in manifest.files {
            let fileURL = rootURL.appendingPathComponent(entry.relativePath, isDirectory: false)
            let content = (try? String(contentsOf: fileURL, encoding: .utf8)) ?? ""
            discovered.append(
                DiscoveredFile(
                    relativePath: entry.relativePath,
                    tier: entry.tier,
                    sha256: entry.sha256,
                    mtimeEpochSec: entry.mtimeEpochSec,
                    byteCount: entry.byteCount,
                    content: content
                )
            )
        }
        let effectiveKV = kvMode ?? manifest.caches.first?.kvMode ?? "q4"
        let built = buildEnvelope(
            packName: manifest.packName,
            rootPath: manifest.rootPath,
            files: discovered,
            clusterBlockTokens: manifest.clusterBlockTokens,
            kvMode: effectiveKV
        )
        let delta = includeDelta
            ? computeDelta(for: manifest, fileManager: fm)
            : DeltaOverlay(deltaBlock: "", modifiedFiles: [], deletedFiles: [])
        return (built.systemMessageContent, delta)
    }

    /// Computes `<|memory_delta|>` for files modified or deleted since `manifest` was indexed.
    public static func computeDelta(
        for manifest: QWMemoryManifest,
        excludingPaths: Set<String> = [],
        fileManager: FileManager = .default
    ) -> DeltaOverlay {
        let rootURL = URL(fileURLWithPath: manifest.rootPath, isDirectory: true)
        var modified: [String] = []
        var deleted: [String] = []
        var sections: [String] = []

        for entry in manifest.files {
            if excludingPaths.contains(entry.relativePath) { continue }
            let fileURL = rootURL.appendingPathComponent(entry.relativePath, isDirectory: false)
            guard fileManager.fileExists(atPath: fileURL.path),
                  let attrs = try? fileManager.attributesOfItem(atPath: fileURL.path) else {
                deleted.append(entry.relativePath)
                sections.append("[DELETED FILE: \(entry.relativePath)]")
                continue
            }
            let diskMtime = Int64((attrs[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)
            let diskSize = (attrs[.size] as? NSNumber)?.intValue ?? 0
            if diskMtime == entry.mtimeEpochSec && diskSize == entry.byteCount {
                continue
            }
            guard let data = try? Data(contentsOf: fileURL) else { continue }
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            if digest == entry.sha256 { continue }
            modified.append(entry.relativePath)
            if let text = String(data: data, encoding: .utf8) {
                let normalized = text.hasSuffix("\n") ? text : (text + "\n")
                sections.append("[UPDATED FILE: \(entry.relativePath)]\n" + normalized)
            }
        }

        guard !modified.isEmpty || !deleted.isEmpty else {
            return DeltaOverlay(deltaBlock: "", modifiedFiles: [], deletedFiles: [])
        }
        var block = "\n<|memory_delta|>\n"
        block += "[LIVE WORKING-TREE DELTA: \(modified.count) updated, \(deleted.count) deleted since .qwcache snapshot]\n"
        block += sections.joined(separator: "\n")
        if !block.hasSuffix("\n") { block += "\n" }
        block += "</|memory_delta|>"
        return DeltaOverlay(deltaBlock: block, modifiedFiles: modified, deletedFiles: deleted)
    }

    /// Generates the 1-line directive informing coding agents that the codebase is resident in LSA memory.
    public static func agentAwarenessNotice(
        packName: String,
        fileCount: Int,
        cortexCount: Int? = nil,
        tokenCount: Int,
        kvMode: String
    ) -> String {
        let effectiveCortex = cortexCount ?? fileCount
        let vaultCount = max(0, fileCount - effectiveCortex)
        if vaultCount > 0 {
            return "[QWG Memory Notice: The repository '\(packName)' (\(fileCount) indexed files: \(effectiveCortex) in Resident Cortex + \(vaultCount) in Off-Context LSA Vault, ~\(tokenCount) cortex tokens) is mounted in \(kvMode.uppercased()) LSA memory. Answer architecture and cross-file questions directly from the Cortex Map, Resident Cortex files, and auto-recalled Vault files without redundant grep, sed, or read_file calls.]"
        }
        return "[QWG Memory Notice: The repository '\(packName)' (\(fileCount) files, ~\(tokenCount) tokens) "
            + "is pre-loaded in your \(kvMode.uppercased()) LSA memory above. Answer architecture and "
            + "cross-file questions directly from memory without redundant grep, sed, or read_file calls.]"
    }

    /// Walks up parent directories from `startURL` looking for `<dir>/.qwg/manifest.json`.
    public static func discoverManifestRoot(
        walkingUpFrom startURL: URL,
        fileManager: FileManager = .default
    ) -> URL? {
        var current = startURL.standardizedFileURL
        var isDir: ObjCBool = false
        if fileManager.fileExists(atPath: current.path, isDirectory: &isDir), !isDir.boolValue {
            current = current.deletingLastPathComponent()
        }
        for _ in 0..<24 {
            let manifestURL = QWMemoryManifest.manifestURL(in: current)
            if fileManager.fileExists(atPath: manifestURL.path) {
                return current
            }
            let parent = current.deletingLastPathComponent()
            if parent.path == current.path { break }
            current = parent
        }
        return nil
    }

    /// Scans a system/developer prompt string for local directory paths containing `.qwg/manifest.json`.
    public static func discoverManifestRoot(
        inSystemPrompt prompt: String,
        fileManager: FileManager = .default
    ) -> URL? {
        guard !prompt.isEmpty else { return nil }
        let pattern = #"((?:~|/Users/|/home/|/opt/|/var/|/tmp/|/Volumes/)[A-Za-z0-9._\-/]+)"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let nsRange = NSRange(prompt.startIndex..<prompt.endIndex, in: prompt)
        let matches = regex.matches(in: prompt, range: nsRange)
        for match in matches {
            guard let range = Range(match.range(at: 1), in: prompt) else { continue }
            var rawPath = String(prompt[range])
            while let last = rawPath.last, [".", ",", ";", ":", ")", "]", "'", "\""].contains(String(last)) {
                rawPath.removeLast()
            }
            let expanded = (rawPath as NSString).expandingTildeInPath
            guard fileManager.fileExists(atPath: expanded) else { continue }
            if let found = discoverManifestRoot(walkingUpFrom: URL(fileURLWithPath: expanded), fileManager: fileManager) {
                return found
            }
        }
        return nil
    }

    /// Ensures `.qwg/` is listed in the repository's `.git/info/exclude` so `.qwcache` files stay untracked.
    public static func ensureGitExclude(in rootURL: URL) {
        let root = rootURL.standardizedFileURL
        let fm = FileManager.default
        var excludeURL = root.appendingPathComponent(".git/info/exclude", isDirectory: false)
        if !fm.fileExists(atPath: excludeURL.deletingLastPathComponent().path) {
            if let gitCommon = runGit(in: root, args: ["rev-parse", "--git-path", "info/exclude"])?
                .trimmingCharacters(in: .whitespacesAndNewlines), !gitCommon.isEmpty {
                excludeURL = gitCommon.hasPrefix("/")
                    ? URL(fileURLWithPath: gitCommon)
                    : root.appendingPathComponent(gitCommon)
            }
        }
        let infoDir = excludeURL.deletingLastPathComponent()
        guard fm.fileExists(atPath: infoDir.deletingLastPathComponent().path) else { return }
        try? fm.createDirectory(at: infoDir, withIntermediateDirectories: true)
        let existing = (try? String(contentsOf: excludeURL, encoding: .utf8)) ?? ""
        let lines = existing.split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespaces) }
        guard !lines.contains(".qwg/") && !lines.contains(".qwg") else { return }
        let suffix = (existing.isEmpty || existing.hasSuffix("\n")) ? ".qwg/\n" : "\n.qwg/\n"
        if let data = (existing + suffix).data(using: .utf8) {
            try? data.write(to: excludeURL, options: .atomic)
        }
    }

    /// Queries the current `HEAD` commit SHA for a repository root, or nil if not a git checkout.
    public static func currentGitHeadSHA(in rootURL: URL) -> String? {
        runGit(in: rootURL, args: ["rev-parse", "HEAD"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Produces a filesystem-safe slug for a model or pack name.
    public static func safeSlug(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
        var out = ""
        for scalar in value.unicodeScalars {
            out.append(allowed.contains(scalar) ? Character(scalar) : "_")
        }
        return out.isEmpty ? "model" : out
    }

    /// Slices a rendered chat-template wire string immediately before the trailing sentinel user turn.
    /// Uses `.backwards` search so source files containing literal `"<|im_start|>user"` never truncate the envelope.
    public static func sliceBeforeSentinelUserTurn(fullWire: String, sentinel: String) -> String {
        if let sentinelRange = fullWire.range(of: sentinel, options: .backwards) {
            let prefixBeforeSentinel = fullWire[..<sentinelRange.lowerBound]
            if let userMarkerRange = prefixBeforeSentinel.range(of: "<|im_start|>user", options: .backwards) {
                return String(fullWire[..<userMarkerRange.lowerBound])
            }
            return String(prefixBeforeSentinel)
        }
        if let userMarkerRange = fullWire.range(of: "<|im_start|>user", options: .backwards) {
            return String(fullWire[..<userMarkerRange.lowerBound])
        }
        return fullWire
    }

    // MARK: - Private Helpers

    /// Universal directory nesting depth from the workspace root (0 for root files, 1 for first-level subpaths, etc.).
    private static func structuralDepth(relativePath: String) -> Int {
        relativePath.utf8.reduce(into: 0) { count, byte in
            if byte == UInt8(ascii: "/") { count += 1 }
        }
    }

    /// Language-agnostic test file naming check used only when `includeTests == false`.
    private static func isConventionalTestFile(relativePath: String) -> Bool {
        let lower = relativePath.lowercased()
        let parts = lower.split(separator: "/")
        for part in parts.dropLast() {
            if part == "test" || part == "tests" || part == "__tests__" || part.hasSuffix("tests") {
                return true
            }
        }
        guard let fileName = parts.last else { return false }
        let stem = (String(fileName) as NSString).deletingPathExtension
        return stem.hasPrefix("test_")
            || stem.hasSuffix("_test")
            || stem.hasSuffix("_tests")
            || stem.hasSuffix("test")
            || stem.hasSuffix("tests")
            || stem.hasSuffix(".test")
            || stem.hasSuffix(".spec")
    }

    public static func topModuleGroup(for relativePath: String) -> String {
        let parts = relativePath.split(separator: "/")
        if parts.count >= 3 { return "\(parts[0])/\(parts[1])/\(parts[2])" }
        if parts.count >= 2 { return "\(parts[0])/\(parts[1])" }
        return String(parts.first ?? "root")
    }

    private static func shouldIgnore(relativePath: String, ignoreMatcher: QWMemoryIgnoreMatcher) -> Bool {
        let parts = relativePath.split(separator: "/").map(String.init)
        for part in parts.dropLast() where excludedDirectories.contains(part) || part.hasPrefix(".") {
            return true
        }
        guard let fileName = parts.last else { return true }
        let lowerName = fileName.lowercased()
        if excludedFileNames.contains(lowerName) || lowerName.hasPrefix(".") { return true }
        let ext = (lowerName as NSString).pathExtension
        if !ext.isEmpty && excludedExtensions.contains(ext) { return true }
        if ignoreMatcher.isIgnored(relativePath: relativePath, isDirectory: false) {
            return true
        }
        return false
    }

    private static func listCandidateRelativePaths(
        in rootURL: URL,
        ignoreMatcher: QWMemoryIgnoreMatcher
    ) -> [String] {
        if let gitOut = runGit(in: rootURL, args: ["ls-files", "-z", "--cached", "--others", "--exclude-standard"]),
           !gitOut.isEmpty {
            return gitOut.split(separator: "\0").map(String.init).filter { !$0.isEmpty }
        }
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: rootURL,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            return []
        }
        var paths: [String] = []
        let prefix = rootURL.standardizedFileURL.path.hasSuffix("/")
            ? rootURL.standardizedFileURL.path
            : (rootURL.standardizedFileURL.path + "/")
        for case let url as URL in enumerator {
            let std = url.standardizedFileURL.path
            guard std.hasPrefix(prefix) else { continue }
            let rel = String(std.dropFirst(prefix.count))
            let top = rel.split(separator: "/").first.map(String.init) ?? ""
            if excludedDirectories.contains(top) {
                enumerator.skipDescendants()
                continue
            }
            let isDir = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
            if isDir {
                continue
            }
            if ignoreMatcher.isIgnored(relativePath: rel, isDirectory: false) {
                continue
            }
            paths.append(rel)
        }
        return paths
    }

    private static func queryGitLastCommitEpochs(in rootURL: URL) -> [String: Int64] {
        guard let logOut = runGit(in: rootURL, args: ["log", "-n", "200", "--format=__COMMIT__%ct", "--name-only"]) else {
            return [:]
        }
        var map: [String: Int64] = [:]
        var currentEpoch: Int64 = 0
        for lineSub in logOut.split(whereSeparator: \.isNewline) {
            let line = lineSub.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty { continue }
            if line.hasPrefix("__COMMIT__") {
                currentEpoch = Int64(line.dropFirst("__COMMIT__".count)) ?? 0
            } else if map[line] == nil && currentEpoch > 0 {
                map[line] = currentEpoch
            }
        }
        return map
    }

    private static func runGit(in rootURL: URL, args: [String]) -> String? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        p.arguments = ["-C", rootURL.path] + args
        let outPipe = Pipe()
        p.standardOutput = outPipe
        p.standardError = FileHandle.nullDevice
        do {
            try p.run()
        } catch {
            return nil
        }
        let data = outPipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        guard p.terminationStatus == 0 else { return nil }
        return String(data: data, encoding: .utf8)
    }
}
