// QWDocumentIngestor.swift — Contained workspace path guard, PDFKit page-span
// extractor, and Markdown heading span indexer for Cognitive Vault v2.

import CryptoKit
import Foundation
import PDFKit
import QWCore

public enum QWDocumentIngestor {
    /// Maximum single PDF file size ingested into Vault v2 (32 MiB).
    public static let maxPDFFileBytes: Int = 32 * 1024 * 1024

    public struct IngestedDocument: Sendable, Equatable {
        public let content: String
        public let sha256: String
        public let byteCount: Int
        public let mtimeEpochSec: Int64
        public let sourceKind: QWMemoryManifest.SourceKind
        public let spanAnchors: [QWMemoryManifest.SpanAnchor]
    }

    private static let pdfTextCacheLock = NSLock()
    nonisolated(unsafe) private static var pdfTextCache: [String: String] = [:]

    /// Validates that `rootURL` exists as a directory (Law 26) and `candidateURL` resolves strictly inside it.
    public static func validateContainedURL(
        _ candidateURL: URL,
        inside rootURL: URL,
        fileManager: FileManager = .default
    ) -> URL? {
        var rootIsDir: ObjCBool = false
        guard fileManager.fileExists(atPath: rootURL.path, isDirectory: &rootIsDir), rootIsDir.boolValue else {
            return nil
        }
        guard fileManager.fileExists(atPath: candidateURL.path) else {
            return nil
        }
        let canonicalRoot = rootURL.resolvingSymlinksInPath().standardizedFileURL.path
        let canonicalCand = candidateURL.resolvingSymlinksInPath().standardizedFileURL.path
        let rootPrefix = canonicalRoot.hasSuffix("/") ? canonicalRoot : (canonicalRoot + "/")
        guard canonicalCand == canonicalRoot || canonicalCand.hasPrefix(rootPrefix) else {
            return nil
        }
        return URL(fileURLWithPath: canonicalCand, isDirectory: false)
    }

    /// Extracts 1-based `[PDF: <relativePath>#p<N>]` page spans from an unlocked PDF inside `rootURL`.
    public static func ingestPDF(
        at candidateURL: URL,
        relativePath: String,
        rootURL: URL,
        maxBytes: Int = maxPDFFileBytes,
        fileManager: FileManager = .default
    ) -> IngestedDocument? {
        guard let resolvedURL = validateContainedURL(candidateURL, inside: rootURL, fileManager: fileManager),
              let attrs = try? fileManager.attributesOfItem(atPath: resolvedURL.path),
              let fileType = attrs[.type] as? FileAttributeType,
              fileType == .typeRegular,
              let size = (attrs[.size] as? NSNumber)?.intValue,
              size > 0, size <= maxBytes,
              let rawData = try? Data(contentsOf: resolvedURL) else {
            return nil
        }
        guard let doc = PDFDocument(data: rawData), !doc.isLocked, doc.pageCount > 0 else {
            return nil
        }
        let digest = SHA256.hash(data: rawData).map { String(format: "%02x", $0) }.joined()
        let mtime = Int64((attrs[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)

        var linesOut: [String] = []
        var anchors: [QWMemoryManifest.SpanAnchor] = []
        var currentLine = 1

        for idx in 0..<doc.pageCount {
            let pageNum = idx + 1
            guard let rawPage = doc.page(at: idx)?.string else { continue }
            let cleanedLines = rawPage
                .components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespaces) }
            let nonEmpty = cleanedLines.contains { !$0.isEmpty }
            guard nonEmpty else { continue }

            let marker = "[PDF: \(relativePath)#p\(pageNum)]"
            let startLine = currentLine
            linesOut.append(marker)
            currentLine += 1
            var prevBlank = false
            for ln in cleanedLines {
                if ln.isEmpty {
                    if !prevBlank {
                        linesOut.append("")
                        currentLine += 1
                        prevBlank = true
                    }
                } else {
                    linesOut.append(ln)
                    currentLine += 1
                    prevBlank = false
                }
            }
            let endLine = max(startLine, currentLine - 1)
            anchors.append(QWMemoryManifest.SpanAnchor(
                label: "#p\(pageNum)",
                startLine: startLine,
                endLine: endLine,
                pageNumber: pageNum
            ))
        }
        guard !linesOut.isEmpty else { return nil }
        let formatted = linesOut.joined(separator: "\n") + "\n"
        cachePDFText(formatted, key: "\(resolvedURL.path)|\(digest)")
        return IngestedDocument(
            content: formatted,
            sha256: digest,
            byteCount: size,
            mtimeEpochSec: mtime,
            sourceKind: .pdfPageSpan,
            spanAnchors: anchors
        )
    }

    /// Extracts ATX heading line-range anchors (`#`, `##`, `###`) from Markdown/text documents.
    public static func extractMarkdownSpans(content: String) -> [QWMemoryManifest.SpanAnchor] {
        let lines = content.components(separatedBy: .newlines)
        guard !lines.isEmpty else { return [] }
        var headings: [(title: String, line: Int)] = []
        var inFence = false
        for (i, raw) in lines.enumerated() {
            let trimmed = raw.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
                inFence.toggle()
                continue
            }
            if inFence { continue }
            if trimmed.hasPrefix("# ") || trimmed.hasPrefix("## ") || trimmed.hasPrefix("### ") {
                let label = String(trimmed.prefix(80))
                headings.append((label, i + 1))
            }
        }
        guard !headings.isEmpty else { return [] }
        var anchors: [QWMemoryManifest.SpanAnchor] = []
        anchors.reserveCapacity(headings.count)
        for (idx, h) in headings.enumerated() {
            let endLine = (idx + 1 < headings.count) ? max(h.line, headings[idx + 1].line - 1) : lines.count
            anchors.append(QWMemoryManifest.SpanAnchor(label: h.title, startLine: h.line, endLine: endLine))
        }
        return anchors
    }

    /// Reads formatted UTF-8 text for any manifest `FileEntry`, enforcing symlink containment inside `rootURL`.
    public static func readFormattedText(
        for entry: QWMemoryManifest.FileEntry,
        rootURL: URL,
        fileManager: FileManager = .default
    ) -> String? {
        let candidate = rootURL.appendingPathComponent(entry.relativePath, isDirectory: false)
        guard let resolved = validateContainedURL(candidate, inside: rootURL, fileManager: fileManager) else {
            return nil
        }
        if entry.sourceKind == .pdfPageSpan || entry.relativePath.lowercased().hasSuffix(".pdf") {
            let key = "\(resolved.path)|\(entry.sha256)"
            if let cached = pdfTextCacheLock.withLock({ pdfTextCache[key] }) {
                return cached
            }
            return ingestPDF(
                at: candidate,
                relativePath: entry.relativePath,
                rootURL: rootURL,
                fileManager: fileManager
            )?.content
        }
        guard let text = try? String(contentsOf: resolved, encoding: .utf8), !text.isEmpty else {
            return nil
        }
        return text
    }

    public static func invalidateCachedPDFText(forResolvedPath resolvedPath: String) {
        let prefix = resolvedPath + "|"
        pdfTextCacheLock.withLock {
            pdfTextCache = pdfTextCache.filter { !$0.key.hasPrefix(prefix) }
        }
    }

    private static func cachePDFText(_ text: String, key: String) {
        pdfTextCacheLock.withLock {
            if pdfTextCache.count >= 128 {
                pdfTextCache.removeAll(keepingCapacity: true)
            }
            pdfTextCache[key] = text
        }
    }
}
