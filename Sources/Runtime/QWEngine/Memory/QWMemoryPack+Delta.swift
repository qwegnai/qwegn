// QWMemoryPack+Delta.swift — Delta overlay, git churn, and manifest root discovery.
import CryptoKit
import Foundation
import QWCore

extension QWMemoryPack {
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
            guard fileManager.fileExists(atPath: fileURL.path) else {
                deleted.append(entry.relativePath)
                sections.append("[DELETED FILE: \(entry.relativePath)]")
                continue
            }
            guard let containedURL = QWDocumentIngestor.validateContainedURL(fileURL, inside: rootURL, fileManager: fileManager),
                  let attrs = try? fileManager.attributesOfItem(atPath: containedURL.path) else {
                continue
            }
            let diskMtime = Int64((attrs[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)
            let diskSize = (attrs[.size] as? NSNumber)?.intValue ?? 0
            if diskMtime == entry.mtimeEpochSec && diskSize == entry.byteCount {
                continue
            }
            guard let data = try? Data(contentsOf: containedURL) else { continue }
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            if digest == entry.sha256 { continue }
            modified.append(entry.relativePath)
            if let text = QWDocumentIngestor.readFormattedText(for: entry, rootURL: rootURL, fileManager: fileManager) {
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

    /// Git commit comparison between a manifest baseline SHA and repository `HEAD`.
    public struct GitStalenessInfo: Sendable, Equatable {
        public let isGitRepo: Bool
        public let headSHA: String?
        public let manifestSHA: String?
        public let commitsBehind: Int
        public let diverged: Bool

        public init(isGitRepo: Bool, headSHA: String?, manifestSHA: String?, commitsBehind: Int, diverged: Bool) {
            self.isGitRepo = isGitRepo
            self.headSHA = headSHA
            self.manifestSHA = manifestSHA
            self.commitsBehind = commitsBehind
            self.diverged = diverged
        }
    }

    /// Queries the current `HEAD` commit SHA for a repository root, or nil if not a git checkout.
    public static func currentGitHeadSHA(in rootURL: URL) -> String? {
        runGit(in: rootURL, args: ["rev-parse", "HEAD"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Evaluates how many commits `HEAD` has moved since `manifestSHA`.
    public static func evaluateGitStaleness(manifestSHA: String?, in rootURL: URL) -> GitStalenessInfo {
        guard let head = currentGitHeadSHA(in: rootURL), !head.isEmpty else {
            return GitStalenessInfo(
                isGitRepo: false,
                headSHA: nil,
                manifestSHA: manifestSHA,
                commitsBehind: 0,
                diverged: false
            )
        }
        let baseline = manifestSHA?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let baseline, !baseline.isEmpty else {
            return GitStalenessInfo(
                isGitRepo: true,
                headSHA: head,
                manifestSHA: nil,
                commitsBehind: 0,
                diverged: false
            )
        }
        if head.caseInsensitiveCompare(baseline) == .orderedSame {
            return GitStalenessInfo(
                isGitRepo: true,
                headSHA: head,
                manifestSHA: baseline,
                commitsBehind: 0,
                diverged: false
            )
        }
        if let revOut = runGit(in: rootURL, args: ["rev-list", "--left-right", "--count", "\(baseline)...HEAD"])?
            .trimmingCharacters(in: .whitespacesAndNewlines) {
            let parts = revOut.split(whereSeparator: \.isWhitespace).compactMap { Int($0) }
            if parts.count == 2 {
                let aheadOfHead = max(0, parts[0])
                let behindHead = max(0, parts[1])
                return GitStalenessInfo(
                    isGitRepo: true,
                    headSHA: head,
                    manifestSHA: baseline,
                    commitsBehind: behindHead,
                    diverged: aheadOfHead > 0
                )
            }
        }
        return GitStalenessInfo(
            isGitRepo: true,
            headSHA: head,
            manifestSHA: baseline,
            commitsBehind: 0,
            diverged: true
        )
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
    static func structuralDepth(relativePath: String) -> Int {
        relativePath.utf8.reduce(into: 0) { count, byte in
            if byte == UInt8(ascii: "/") { count += 1 }
        }
    }

    /// Language-agnostic test file naming check used only when `includeTests == false`.
    static func isConventionalTestFile(relativePath: String) -> Bool {
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

    static func shouldIgnore(relativePath: String, ignoreMatcher: QWMemoryIgnoreMatcher, includePDFs: Bool = true) -> Bool {
        let parts = relativePath.split(separator: "/").map(String.init)
        for part in parts.dropLast() where excludedDirectories.contains(part) || part.hasPrefix(".") {
            return true
        }
        guard let fileName = parts.last else { return true }
        let lowerName = fileName.lowercased()
        if excludedFileNames.contains(lowerName) || lowerName.hasPrefix(".") { return true }
        let ext = (lowerName as NSString).pathExtension
        if !ext.isEmpty && excludedExtensions.contains(ext) && !(includePDFs && ext == "pdf") { return true }
        if ignoreMatcher.isIgnored(relativePath: relativePath, isDirectory: false) {
            return true
        }
        return false
    }

    static func listCandidateRelativePaths(
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

    static func queryGitLastCommitEpochs(in rootURL: URL) -> [String: Int64] {
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

    static func runGit(in rootURL: URL, args: [String]) -> String? {
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
