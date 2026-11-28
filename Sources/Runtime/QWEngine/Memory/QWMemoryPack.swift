// QWMemoryPack.swift — Unified Codebase & Knowledge Memory Pack envelope builder.
import CryptoKit
import Foundation
import QWCore

public enum QWMemoryPack {
    public struct DiscoveredFile: Sendable, Equatable {
        public let relativePath: String
        public let tier: Int
        public let sha256: String
        public let mtimeEpochSec: Int64
        public let byteCount: Int
        public let content: String
        public let sourceKind: QWMemoryManifest.SourceKind
        public let spanAnchors: [QWMemoryManifest.SpanAnchor]

        public init(
            relativePath: String,
            tier: Int,
            sha256: String,
            mtimeEpochSec: Int64,
            byteCount: Int,
            content: String,
            sourceKind: QWMemoryManifest.SourceKind? = nil,
            spanAnchors: [QWMemoryManifest.SpanAnchor] = []
        ) {
            self.relativePath = relativePath
            self.tier = tier
            self.sha256 = sha256
            self.mtimeEpochSec = mtimeEpochSec
            self.byteCount = byteCount
            self.content = content
            self.sourceKind = sourceKind ?? QWMemoryManifest.SourceKind.infer(from: relativePath)
            self.spanAnchors = spanAnchors
        }
    }

    public struct BuiltEnvelope: Sendable {
        public let systemMessageContent: String
        public let fileEntries: [QWMemoryManifest.FileEntry]
        public let totalBytes: Int
        public var cortexFileCount: Int { fileEntries.filter(\.inCortex).count }
        public var vaultFileCount: Int { fileEntries.filter { !$0.inCortex }.count }
    }

    /// Per-file tokenization and packing progress emitted by `buildEnvelope`.
    public struct FileProgressUpdate: Sendable, Equatable {
        public let processedFiles: Int
        public let totalFiles: Int
        public let currentFile: String
        public let tokenizedTokens: Int

        public init(processedFiles: Int, totalFiles: Int, currentFile: String, tokenizedTokens: Int) {
            self.processedFiles = processedFiles
            self.totalFiles = totalFiles
            self.currentFile = currentFile
            self.tokenizedTokens = tokenizedTokens
        }
    }

    public struct DeltaOverlay: Sendable, Equatable {
        public let deltaBlock: String
        public let modifiedFiles: [String]
        public let deletedFiles: [String]

        public var hasChanges: Bool {
            !modifiedFiles.isEmpty || !deletedFiles.isEmpty
        }
    }

    static let excludedDirectories: Set<String> = [
        ".git", ".qwg", ".build", ".swiftpm", "node_modules", ".venv", ".venv-oracle",
        "venv", "__pycache__", "out", "build", "dist", "target", "Pods", "DerivedData",
        ".worktrees", ".idea", ".vscode", ".next", ".turbo"
    ]

    static let excludedFileNames: Set<String> = [
        "package.resolved", "cargo.lock", "pnpm-lock.yaml", "package-lock.json",
        "yarn.lock", "poetry.lock", "uv.lock", "gemfile.lock", "composer.lock",
        ".ds_store"
    ]

    static let excludedExtensions: Set<String> = [
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
        includePDFs: Bool = true,
        maxFileBytes: Int = maxSingleFileBytes
    ) throws -> [DiscoveredFile] {
        let root = rootURL.standardizedFileURL
        var rootIsDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: root.path, isDirectory: &rootIsDir), rootIsDir.boolValue else {
            return []
        }
        let ignoreMatcher = QWMemoryIgnoreMatcher(rootURL: root)
        let candidateRelPaths = listCandidateRelativePaths(in: root, ignoreMatcher: ignoreMatcher)
        let churnEpochByPath = queryGitLastCommitEpochs(in: root)
        let fm = FileManager.default

        var rawItems: [(file: DiscoveredFile, commitEpoch: Int64)] = []
        for relPath in candidateRelPaths {
            if shouldIgnore(relativePath: relPath, ignoreMatcher: ignoreMatcher, includePDFs: includePDFs) { continue }
            if !includeTests && isConventionalTestFile(relativePath: relPath) { continue }
            let depth = structuralDepth(relativePath: relPath)
            let fileURL = root.appendingPathComponent(relPath, isDirectory: false)
            guard let containedURL = QWDocumentIngestor.validateContainedURL(fileURL, inside: root, fileManager: fm) else {
                continue
            }
            if relPath.lowercased().hasSuffix(".pdf") {
                guard includePDFs,
                      let pdfDoc = QWDocumentIngestor.ingestPDF(
                          at: containedURL, relativePath: relPath, rootURL: root, fileManager: fm
                      ) else {
                    continue
                }
                let discovered = DiscoveredFile(
                    relativePath: relPath,
                    tier: max(depth, 2),
                    sha256: pdfDoc.sha256,
                    mtimeEpochSec: pdfDoc.mtimeEpochSec,
                    byteCount: pdfDoc.byteCount,
                    content: pdfDoc.content,
                    sourceKind: .pdfPageSpan,
                    spanAnchors: pdfDoc.spanAnchors
                )
                rawItems.append((discovered, churnEpochByPath[relPath] ?? pdfDoc.mtimeEpochSec))
                continue
            }
            guard let attrs = try? fm.attributesOfItem(atPath: containedURL.path),
                  let fileType = attrs[.type] as? FileAttributeType,
                  fileType == .typeRegular,
                  let size = (attrs[.size] as? NSNumber)?.intValue,
                  size > 0, size <= maxFileBytes else {
                continue
            }
            guard let data = try? Data(contentsOf: containedURL),
                  !data.contains(0),
                  let text = String(data: data, encoding: .utf8) else {
                continue
            }
            let mtime = Int64((attrs[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            let kind = QWMemoryManifest.SourceKind.infer(from: relPath)
            let spans = kind == .markdown ? QWDocumentIngestor.extractMarkdownSpans(content: text) : []
            let discovered = DiscoveredFile(
                relativePath: relPath,
                tier: depth,
                sha256: digest,
                mtimeEpochSec: mtime,
                byteCount: size,
                content: text,
                sourceKind: kind,
                spanAnchors: spans
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
        try buildEnvelope(
            packName: packName,
            rootPath: rootPath,
            files: files,
            clusterBlockTokens: clusterBlockTokens,
            kvMode: kvMode,
            maxTokens: maxTokens,
            onFileProgress: nil,
            tokenCounter: tokenCounter
        )
    }

    /// Packs discovered files into the `<|memory|>...</|memory|>` envelope and emits per-file progress.
    public static func buildEnvelope(
        packName: String,
        rootPath: String,
        files: [DiscoveredFile],
        clusterBlockTokens: Int = 512,
        kvMode: String = "q4",
        maxTokens: Int? = nil,
        onFileProgress: ((FileProgressUpdate) -> Void)?,
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
        for (index, file) in files.enumerated() {
            let symbols = QWMemoryVault.extractSymbols(from: file.content, relativePath: file.relativePath)
            let normalizedBody = file.content.hasSuffix("\n") ? file.content : (file.content + "\n")
            let section = "\n[FILE: \(file.relativePath)]\n" + normalizedBody
            let sectionTokens = try tokenCounter(section)
            prepared.append(PreparedSection(file: file, symbols: symbols, section: section, sectionTokens: sectionTokens))
            totalRawTokens += sectionTokens
            onFileProgress?(
                FileProgressUpdate(
                    processedFiles: index + 1,
                    totalFiles: files.count,
                    currentFile: file.relativePath,
                    tokenizedTokens: totalRawTokens
                )
            )
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
            let isPDF = file.sourceKind == .pdfPageSpan
            let exceedsPerFileCap = (tokenCap ?? 0) >= 4096 && sectionTokens > max(2048, (tokenCap ?? 0) / 5)
            if let tokenCap, (!cortexPaths.isEmpty || isPDF), (isPDF || runningTokens + sectionTokens + 96 > tokenCap || exceedsPerFileCap) {
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
                        symbols: item.symbols,
                        sourceKind: file.sourceKind,
                        spanAnchors: file.spanAnchors
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
                    symbols: item.symbols,
                    sourceKind: file.sourceKind,
                    spanAnchors: file.spanAnchors
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

}
