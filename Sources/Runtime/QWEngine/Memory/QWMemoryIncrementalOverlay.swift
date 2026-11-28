// QWMemoryIncrementalOverlay.swift — Sub-millisecond APFS tracked-directory +
// POSIX `lstat` incremental Vault v2 in-memory overlay (1.0s debounce).

import CryptoKit
import Darwin
import Foundation
import QWCore

public struct QWMemoryIncrementalTracker: Sendable {
    public struct FileStatSignature: Sendable, Equatable {
        public let sec: Int
        public let nsec: Int
        public let size: Int64
        public let ino: UInt64
    }

    public struct DirStatSignature: Sendable, Equatable {
        public let sec: Int
        public let nsec: Int
    }

    public static let debounceNanoseconds: UInt64 = 1_000_000_000

    public var lastCheckedUptimeNs: UInt64
    public var fileSignatures: [String: FileStatSignature]
    public var dirSignatures: [String: DirStatSignature]

    public init(rootURL: URL, manifest: QWMemoryManifest) {
        self.lastCheckedUptimeNs = DispatchTime.now().uptimeNanoseconds
        var fSigs: [String: FileStatSignature] = [:]
        var dPaths: Set<String> = [""]
        for entry in manifest.files {
            let fullPath = rootURL.appendingPathComponent(entry.relativePath, isDirectory: false).path
            if let sig = Self.lstatFile(atPath: fullPath) {
                fSigs[entry.relativePath] = sig
            }
            let parent = (entry.relativePath as NSString).deletingLastPathComponent
            if !parent.isEmpty && parent != "." {
                dPaths.insert(parent)
            }
        }
        // Also watch top-level subdirectories (e.g. Sources/*, docs/*) so newly created folders/files are caught.
        Self.collectInitialWatchDirs(in: rootURL, relativePrefix: "", depth: 0, maxDepth: 2, into: &dPaths)
        var dSigs: [String: DirStatSignature] = [:]
        for relDir in dPaths {
            let fullDir = relDir.isEmpty ? rootURL.path : rootURL.appendingPathComponent(relDir, isDirectory: true).path
            if let dsig = Self.statDirectory(atPath: fullDir) {
                dSigs[relDir] = dsig
            }
        }
        self.fileSignatures = fSigs
        self.dirSignatures = dSigs
    }

    /// Checks tracked directories & files in `< 0.25 ms` (debounced by `1.0s`), returning an updated in-memory `PackMount` if any file changed.
    public mutating func refreshIfNeeded(
        mount: QWMountedVaultSet.PackMount,
        nowUptimeNs: UInt64 = DispatchTime.now().uptimeNanoseconds
    ) -> QWMountedVaultSet.PackMount? {
        if nowUptimeNs >= lastCheckedUptimeNs && (nowUptimeNs - lastCheckedUptimeNs) < Self.debounceNanoseconds {
            return nil
        }
        lastCheckedUptimeNs = nowUptimeNs

        let rootURL = mount.rootURL
        var dirtyPaths: Set<String> = []
        var deletedPaths: Set<String> = []
        var candidateNewPaths: Set<String> = []

        // 1. O(D) directory `st_mtimespec` scan: only enumerate directories whose `(sec, nsec)` changed.
        var newDirSigs = dirSignatures
        for (relDir, oldSig) in dirSignatures {
            let fullDirURL = relDir.isEmpty ? rootURL : rootURL.appendingPathComponent(relDir, isDirectory: true)
            guard let curSig = Self.statDirectory(atPath: fullDirURL.path) else {
                newDirSigs.removeValue(forKey: relDir)
                continue
            }
            if curSig != oldSig {
                newDirSigs[relDir] = curSig
                Self.scanChangedDirectory(
                    dirURL: fullDirURL,
                    relDir: relDir,
                    rootURL: rootURL,
                    knownFiles: fileSignatures,
                    newDirSigs: &newDirSigs,
                    candidateNewPaths: &candidateNewPaths
                )
            }
        }
        dirSignatures = newDirSigs

        // 2. O(F) indexed-file `lstat` check comparing `(sec, nsec, size, ino)`.
        for (relPath, oldSig) in fileSignatures {
            let fullPath = rootURL.appendingPathComponent(relPath, isDirectory: false).path
            if let curSig = Self.lstatFile(atPath: fullPath) {
                if curSig != oldSig {
                    dirtyPaths.insert(relPath)
                    fileSignatures[relPath] = curSig
                }
            } else {
                deletedPaths.insert(relPath)
                fileSignatures.removeValue(forKey: relPath)
            }
        }
        dirtyPaths.formUnion(candidateNewPaths)
        guard !dirtyPaths.isEmpty || !deletedPaths.isEmpty else {
            return nil
        }

        // 3. Re-ingest only the K dirty/new files into an in-memory overlay (never touching `.qwg/manifest.json` on disk).
        var entriesByPath: [String: QWMemoryManifest.FileEntry] = Dictionary(
            uniqueKeysWithValues: mount.manifest.files.map { ($0.relativePath, $0) }
        )
        for del in deletedPaths {
            let resolvedPath = rootURL.appendingPathComponent(del, isDirectory: false).resolvingSymlinksInPath().path
            QWDocumentIngestor.invalidateCachedPDFText(forResolvedPath: resolvedPath)
            entriesByPath.removeValue(forKey: del)
        }

        var updatedContents: [String: String] = [:]
        for relPath in dirtyPaths.sorted() {
            let fileURL = rootURL.appendingPathComponent(relPath, isDirectory: false)
            guard let resolved = QWDocumentIngestor.validateContainedURL(fileURL, inside: rootURL) else {
                entriesByPath.removeValue(forKey: relPath)
                fileSignatures.removeValue(forKey: relPath)
                continue
            }
            QWDocumentIngestor.invalidateCachedPDFText(forResolvedPath: resolved.path)
            if let sig = Self.lstatFile(atPath: resolved.path) {
                fileSignatures[relPath] = sig
            }
            let lower = relPath.lowercased()
            if lower.hasSuffix(".pdf") {
                guard let pdfDoc = QWDocumentIngestor.ingestPDF(at: fileURL, relativePath: relPath, rootURL: rootURL) else {
                    continue
                }
                updatedContents[relPath] = pdfDoc.content
                let syms = QWMemoryVault.extractSymbols(from: pdfDoc.content, relativePath: relPath)
                entriesByPath[relPath] = QWMemoryManifest.FileEntry(
                    relativePath: relPath, tier: 1, sha256: pdfDoc.sha256,
                    mtimeEpochSec: pdfDoc.mtimeEpochSec, byteCount: pdfDoc.byteCount,
                    tokenOffset: -1, tokenCount: max(1, (pdfDoc.content.utf8.count + 3) / 4),
                    inCortex: false, symbols: syms, sourceKind: .pdfPageSpan, spanAnchors: pdfDoc.spanAnchors
                )
            } else {
                guard let data = try? Data(contentsOf: resolved),
                      data.count <= 512 * 1024,
                      !data.prefix(2048).contains(0),
                      let text = String(data: data, encoding: .utf8), !text.isEmpty else {
                    continue
                }
                updatedContents[relPath] = text
                let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                let mtime = Int64(fileSignatures[relPath]?.sec ?? Int(Date().timeIntervalSince1970))
                let isMd = lower.hasSuffix(".md") || lower.hasSuffix(".mdx")
                let kind: QWMemoryManifest.SourceKind = isMd ? .markdown : .code
                let spans = isMd ? QWDocumentIngestor.extractMarkdownSpans(content: text) : []
                let syms = QWMemoryVault.extractSymbols(from: text, relativePath: relPath)
                entriesByPath[relPath] = QWMemoryManifest.FileEntry(
                    relativePath: relPath, tier: 1, sha256: sha,
                    mtimeEpochSec: mtime, byteCount: data.count,
                    tokenOffset: -1, tokenCount: max(1, (text.utf8.count + 3) / 4),
                    inCortex: false, symbols: syms, sourceKind: kind, spanAnchors: spans
                )
            }
        }

        let sortedEntries = entriesByPath.values.sorted { $0.relativePath < $1.relativePath }
        let updatedSem = sortedEntries.isEmpty ? nil : QWMemorySemanticIndex.build(entries: sortedEntries, rootPath: rootURL.path)
        var updatedManifest = mount.manifest
        updatedManifest.files = sortedEntries
        updatedManifest.semanticIndex = updatedSem

        let patchedGraph = Self.patchSymbolGraph(
            existing: mount.symbolGraph,
            entries: sortedEntries,
            dirtyOrNewContents: updatedContents,
            deletedPaths: deletedPaths,
            rootURL: rootURL
        )
        return QWMountedVaultSet.PackMount(
            packName: mount.packName,
            rootURL: rootURL,
            manifest: updatedManifest,
            isPrimaryCortex: mount.isPrimaryCortex,
            symbolGraph: patchedGraph,
            denseVectors: mount.denseVectors
        )
    }

    // MARK: - Helpers

    private static let skippedDirNames: Set<String> = [
        ".git", ".qwg", ".build", "node_modules", ".venv", "venv", "DerivedData", "Pods", "target", "dist", "build", ".worktrees",
    ]
    private static let allowedExtensions: Set<String> = [
        "swift", "metal", "metalh", "h", "hpp", "c", "cc", "cpp", "m", "mm", "py", "rs", "go",
        "ts", "tsx", "js", "jsx", "kt", "java", "rb", "sh", "sql", "proto", "graphql",
        "md", "mdx", "rst", "txt", "pdf", "json", "yaml", "yml", "toml",
    ]

    private static func patchSymbolGraph(
        existing: QWMemorySymbolGraph?,
        entries: [QWMemoryManifest.FileEntry],
        dirtyOrNewContents: [String: String],
        deletedPaths: Set<String>,
        rootURL: URL
    ) -> QWMemorySymbolGraph {
        let changedSet = Set(dirtyOrNewContents.keys).union(deletedPaths)
        guard let existing, !changedSet.isEmpty else {
            var allFiles: [(String, String)] = []
            for e in entries where e.sourceKind == .code {
                if let c = dirtyOrNewContents[e.relativePath] ?? QWDocumentIngestor.readFormattedText(for: e, rootURL: rootURL) {
                    allFiles.append((e.relativePath, c))
                }
            }
            return QWMemorySymbolGraph.build(files: allFiles)
        }
        var keptDecls = existing.declarations.filter { !changedSet.contains($0.relativePath) }
        let patchGraph = QWMemorySymbolGraph.build(files: dirtyOrNewContents.map { ($0.key, $0.value) })
        keptDecls.append(contentsOf: patchGraph.declarations)

        let knownSyms = Set(keptDecls.map(\.symbol))
        var refs = existing.fileReferences.filter { !deletedPaths.contains($0.key) }
        for (relPath, content) in dirtyOrNewContents {
            let idents = Self.extractSimpleIdentifiers(content)
            let matched = idents.intersection(knownSyms).filter { sym in
                keptDecls.contains { $0.symbol == sym && $0.relativePath != relPath }
            }
            if matched.isEmpty { refs.removeValue(forKey: relPath) }
            else { refs[relPath] = matched.sorted() }
        }
        return QWMemorySymbolGraph(declarations: keptDecls, fileReferences: refs)
    }

    private static func extractSimpleIdentifiers(_ text: String) -> Set<String> {
        Set(text.split { !$0.isLetter && !$0.isNumber && $0 != "_" }.filter { $0.count >= 3 }.map(String.init))
    }

    private static func collectInitialWatchDirs(
        in rootURL: URL, relativePrefix: String, depth: Int, maxDepth: Int, into out: inout Set<String>
    ) {
        guard depth < maxDepth else { return }
        let dirURL = relativePrefix.isEmpty ? rootURL : rootURL.appendingPathComponent(relativePrefix, isDirectory: true)
        guard let children = try? FileManager.default.contentsOfDirectory(atPath: dirURL.path) else { return }
        for name in children where !name.hasPrefix(".") && !skippedDirNames.contains(name) {
            let rel = relativePrefix.isEmpty ? name : "\(relativePrefix)/\(name)"
            let childURL = rootURL.appendingPathComponent(rel, isDirectory: true)
            guard QWDocumentIngestor.validateContainedURL(childURL, inside: rootURL) != nil,
                  statDirectory(atPath: childURL.path) != nil else { continue }
            out.insert(rel)
            collectInitialWatchDirs(in: rootURL, relativePrefix: rel, depth: depth + 1, maxDepth: maxDepth, into: &out)
        }
    }

    private static func scanChangedDirectory(
        dirURL: URL, relDir: String, rootURL: URL,
        knownFiles: [String: FileStatSignature],
        newDirSigs: inout [String: DirStatSignature],
        candidateNewPaths: inout Set<String>
    ) {
        guard let children = try? FileManager.default.contentsOfDirectory(atPath: dirURL.path) else { return }
        for name in children where !name.hasPrefix(".") && !skippedDirNames.contains(name) {
            let rel = relDir.isEmpty ? name : "\(relDir)/\(name)"
            let childURL = rootURL.appendingPathComponent(rel, isDirectory: false)
            guard let resolved = QWDocumentIngestor.validateContainedURL(childURL, inside: rootURL) else { continue }
            if let subDirSig = statDirectory(atPath: resolved.path) {
                if newDirSigs[rel] == nil {
                    newDirSigs[rel] = subDirSig
                    scanChangedDirectory(
                        dirURL: resolved, relDir: rel, rootURL: rootURL,
                        knownFiles: knownFiles, newDirSigs: &newDirSigs, candidateNewPaths: &candidateNewPaths
                    )
                }
            } else {
                let ext = (name.lowercased() as NSString).pathExtension
                if allowedExtensions.contains(ext) && knownFiles[rel] == nil {
                    candidateNewPaths.insert(rel)
                }
            }
        }
    }

    private static func lstatFile(atPath path: String) -> FileStatSignature? {
        var st = Darwin.stat()
        guard Darwin.lstat(path, &st) == 0, (st.st_mode & S_IFMT) == S_IFREG else { return nil }
        return FileStatSignature(sec: st.st_mtimespec.tv_sec, nsec: st.st_mtimespec.tv_nsec, size: st.st_size, ino: st.st_ino)
    }

    private static func statDirectory(atPath path: String) -> DirStatSignature? {
        var st = Darwin.stat()
        guard Darwin.lstat(path, &st) == 0, (st.st_mode & S_IFMT) == S_IFDIR else { return nil }
        return DirStatSignature(sec: st.st_mtimespec.tv_sec, nsec: st.st_mtimespec.tv_nsec)
    }
}
