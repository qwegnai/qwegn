// QWMemoryIgnoreMatcher.swift — Gitignore-compliant rule matcher for QWMemoryPack.
// Evaluates .gitignore, .git/info/exclude, global git excludes, and .qwgignore
// in layered order with full support for directory rules, root anchoring, globs, and !negation.

import Darwin
import Foundation

struct QWMemoryIgnoreMatcher: Sendable {
    struct Rule: Sendable {
        let basePrefix: String
        let pattern: String
        let isNegated: Bool
        let isDirectoryOnly: Bool
        let isAnchored: Bool
    }

    let rules: [Rule]

    init(rootURL: URL, fileManager: FileManager = .default) {
        let root = rootURL.standardizedFileURL
        var collected: [Rule] = []

        if let globalExclude = Self.resolveGlobalGitExcludeURL(in: root, fileManager: fileManager),
           let content = try? String(contentsOf: globalExclude, encoding: .utf8) {
            collected.append(contentsOf: Self.parseRules(content: content, basePrefix: ""))
        }

        if let gitExclude = Self.resolveGitInfoExcludeURL(in: root, fileManager: fileManager),
           let content = try? String(contentsOf: gitExclude, encoding: .utf8) {
            collected.append(contentsOf: Self.parseRules(content: content, basePrefix: ""))
        }

        let rootGitignore = root.appendingPathComponent(".gitignore", isDirectory: false)
        if let content = try? String(contentsOf: rootGitignore, encoding: .utf8) {
            collected.append(contentsOf: Self.parseRules(content: content, basePrefix: ""))
        }

        collected.append(contentsOf: Self.discoverNestedIgnoreRules(
            in: root,
            fileName: ".gitignore",
            fileManager: fileManager
        ))

        let rootQwgignore = root.appendingPathComponent(".qwgignore", isDirectory: false)
        if let content = try? String(contentsOf: rootQwgignore, encoding: .utf8) {
            collected.append(contentsOf: Self.parseRules(content: content, basePrefix: ""))
        }

        collected.append(contentsOf: Self.discoverNestedIgnoreRules(
            in: root,
            fileName: ".qwgignore",
            fileManager: fileManager
        ))

        self.rules = collected
    }

    init(rules: [Rule]) {
        self.rules = rules
    }

    func isIgnored(relativePath: String, isDirectory: Bool = false) -> Bool {
        var ignored = false
        let normalized = relativePath.hasPrefix("/")
            ? String(relativePath.dropFirst())
            : relativePath
        guard !normalized.isEmpty else { return false }

        for rule in rules {
            if Self.matches(rule: rule, relativePath: normalized, isDirectory: isDirectory) {
                ignored = !rule.isNegated
            }
        }
        return ignored
    }

    static func parseRules(content: String, basePrefix: String) -> [Rule] {
        let normalizedBase: String
        if basePrefix.isEmpty {
            normalizedBase = ""
        } else {
            normalizedBase = basePrefix.hasSuffix("/") ? basePrefix : (basePrefix + "/")
        }

        var parsed: [Rule] = []
        for rawLine in content.split(whereSeparator: \.isNewline) {
            var line = String(rawLine)
            while line.hasSuffix("\r") { line.removeLast() }
            while line.hasSuffix(" ") && !line.hasSuffix("\\ ") {
                line.removeLast()
            }
            let trimmedLead = line.drop(while: { $0 == " " || $0 == "\t" })
            if trimmedLead.isEmpty || trimmedLead.hasPrefix("#") {
                continue
            }
            var pat = String(trimmedLead)
            var isNegated = false
            if pat.hasPrefix("!") {
                isNegated = true
                pat.removeFirst()
            } else if pat.hasPrefix("\\#") || pat.hasPrefix("\\!") {
                pat.removeFirst()
            }
            guard !pat.isEmpty else { continue }

            var isDirectoryOnly = false
            if pat.hasSuffix("/") {
                isDirectoryOnly = true
                while pat.hasSuffix("/") { pat.removeLast() }
            }
            guard !pat.isEmpty else { continue }

            var isAnchored = false
            if pat.hasPrefix("/") {
                isAnchored = true
                while pat.hasPrefix("/") { pat.removeFirst() }
            } else if pat.contains("/") {
                isAnchored = true
            }
            guard !pat.isEmpty else { continue }

            parsed.append(Rule(
                basePrefix: normalizedBase,
                pattern: pat,
                isNegated: isNegated,
                isDirectoryOnly: isDirectoryOnly,
                isAnchored: isAnchored
            ))
        }
        return parsed
    }

    private static func matches(rule: Rule, relativePath: String, isDirectory: Bool) -> Bool {
        let subPath: String
        if rule.basePrefix.isEmpty {
            subPath = relativePath
        } else {
            guard relativePath.hasPrefix(rule.basePrefix) else { return false }
            subPath = String(relativePath.dropFirst(rule.basePrefix.count))
        }
        guard !subPath.isEmpty else { return false }

        let components = subPath.split(separator: "/").map(String.init)
        if rule.isDirectoryOnly {
            let dirPrefixCount = isDirectory ? components.count : max(0, components.count - 1)
            guard dirPrefixCount > 0 else { return false }
            for depth in 1...dirPrefixCount {
                let candidateDir = components.prefix(depth).joined(separator: "/")
                if matchesPathOrComponent(pattern: rule.pattern, isAnchored: rule.isAnchored,
                                          candidatePath: candidateDir, lastComponent: components[depth - 1]) {
                    return true
                }
            }
            return false
        }

        if matchesPathOrComponent(pattern: rule.pattern, isAnchored: rule.isAnchored,
                                  candidatePath: subPath, lastComponent: components.last ?? subPath) {
            return true
        }
        if components.count > 1 {
            for depth in 1..<(components.count) {
                let parentDir = components.prefix(depth).joined(separator: "/")
                if matchesPathOrComponent(pattern: rule.pattern, isAnchored: rule.isAnchored,
                                          candidatePath: parentDir, lastComponent: components[depth - 1]) {
                    return true
                }
            }
        }
        return false
    }

    private static func matchesPathOrComponent(
        pattern: String,
        isAnchored: Bool,
        candidatePath: String,
        lastComponent: String
    ) -> Bool {
        if isAnchored {
            return globMatch(pattern: pattern, text: candidatePath)
        }
        if globMatch(pattern: pattern, text: lastComponent) {
            return true
        }
        return globMatch(pattern: pattern, text: candidatePath)
    }

    static func globMatch(pattern: String, text: String) -> Bool {
        if pattern == "**" || pattern == "**/*" { return true }
        if !pattern.contains("**") {
            return pattern.withCString { patPtr in
                text.withCString { txtPtr in
                    fnmatch(patPtr, txtPtr, FNM_PATHNAME) == 0
                }
            }
        }
        let patSegs = pattern.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        let txtSegs = text.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        return matchSegments(patSegs, 0, txtSegs, 0)
    }

    private static func matchSegments(_ p: [String], _ pi: Int, _ t: [String], _ ti: Int) -> Bool {
        if pi == p.count && ti == t.count { return true }
        if pi == p.count { return false }
        if p[pi] == "**" {
            var nextPi = pi + 1
            while nextPi < p.count && p[nextPi] == "**" { nextPi += 1 }
            if nextPi == p.count { return true }
            for k in ti...t.count {
                if matchSegments(p, nextPi, t, k) { return true }
            }
            return false
        }
        if ti == t.count { return false }
        let segMatch = p[pi].withCString { patPtr in
            t[ti].withCString { txtPtr in
                fnmatch(patPtr, txtPtr, 0) == 0
            }
        }
        guard segMatch else { return false }
        return matchSegments(p, pi + 1, t, ti + 1)
    }

    private static func resolveGitInfoExcludeURL(in rootURL: URL, fileManager: FileManager) -> URL? {
        let direct = rootURL.appendingPathComponent(".git/info/exclude", isDirectory: false)
        if fileManager.fileExists(atPath: direct.path) {
            return direct
        }
        if let gitPath = runGitCommand(in: rootURL, args: ["rev-parse", "--git-path", "info/exclude"])?
            .trimmingCharacters(in: .whitespacesAndNewlines), !gitPath.isEmpty {
            let resolved = gitPath.hasPrefix("/")
                ? URL(fileURLWithPath: gitPath)
                : rootURL.appendingPathComponent(gitPath)
            if fileManager.fileExists(atPath: resolved.path) {
                return resolved
            }
        }
        return nil
    }

    private static func resolveGlobalGitExcludeURL(in rootURL: URL, fileManager: FileManager) -> URL? {
        if let cfgPath = runGitCommand(in: rootURL, args: ["config", "--get", "core.excludesFile"])?
            .trimmingCharacters(in: .whitespacesAndNewlines), !cfgPath.isEmpty {
            let expanded = (cfgPath as NSString).expandingTildeInPath
            if fileManager.fileExists(atPath: expanded) {
                return URL(fileURLWithPath: expanded)
            }
        }
        let defaultXDG = ("~/.config/git/ignore" as NSString).expandingTildeInPath
        if fileManager.fileExists(atPath: defaultXDG) {
            return URL(fileURLWithPath: defaultXDG)
        }
        return nil
    }

    private static func discoverNestedIgnoreRules(
        in rootURL: URL,
        fileName: String,
        fileManager: FileManager
    ) -> [Rule] {
        guard let enumerator = fileManager.enumerator(
            at: rootURL,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey],
            options: [.skipsPackageDescendants]
        ) else {
            return []
        }
        let rootPrefix = rootURL.path.hasSuffix("/") ? rootURL.path : (rootURL.path + "/")
        let skipDirs: Set<String> = [".git", ".qwg", ".build", ".swiftpm", "node_modules", ".venv", ".venv-oracle", ".worktrees"]
        var nested: [(relDir: String, rules: [Rule])] = []
        for case let url as URL in enumerator {
            let last = url.lastPathComponent
            if skipDirs.contains(last) {
                enumerator.skipDescendants()
                continue
            }
            guard last == fileName else { continue }
            let std = url.standardizedFileURL.path
            guard std.hasPrefix(rootPrefix) else { continue }
            let rel = String(std.dropFirst(rootPrefix.count))
            guard rel != fileName else { continue }
            let relDir = (rel as NSString).deletingLastPathComponent
            if let content = try? String(contentsOf: url, encoding: .utf8) {
                nested.append((relDir, parseRules(content: content, basePrefix: relDir)))
            }
        }
        nested.sort { $0.relDir < $1.relDir }
        return nested.flatMap(\.rules)
    }

    private static func runGitCommand(in rootURL: URL, args: [String]) -> String? {
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
