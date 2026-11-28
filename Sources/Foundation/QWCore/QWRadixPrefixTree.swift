// QWRadixPrefixTree.swift — 256-token page-chunked Radix tree for O(pages) prefix lookup.

import Foundation

/// Environment gate for H1.3 Radix-tree KV lookup, periodic checkpoints, and page deduplication.
public enum QWRadixKVGate {
    public static var isEnabled: Bool {
        ProcessInfo.processInfo.environment["QW_RADIX_KV"] != "0"
    }
}

/// Environment gate and slice-size resolver for scheduler-level prefill-decode interleaving.
public enum QWInterleavedPrefillGate {
    public static let defaultSliceTokens = 512

    public static var sliceTokens: Int {
        guard let raw = ProcessInfo.processInfo.environment["QW_INTERLEAVED_PREFILL_SLICE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else {
            return defaultSliceTokens
        }
        guard let parsed = Int(raw) else { return defaultSliceTokens }
        if parsed <= 0 { return 0 }
        return max(32, (parsed / 32) * 32)
    }

    public static var isEnabled: Bool {
        sliceTokens > 0
    }
}

/// Thread-safe 256-token page-chunked Radix tree indexing cached prompt prefixes by `(mode, tokens)`.
public final class QWRadixPrefixTree<Key: Hashable & Sendable>: @unchecked Sendable {
    public struct PageEdge: Hashable, Sendable {
        public let tokens: [Int]
        private let precomputedHash: Int

        public init(tokens: ArraySlice<Int>) {
            let copy = Array(tokens)
            self.tokens = copy
            var h: UInt64 = 0xcbf29ce484222325
            let prime: UInt64 = 0x100000001b3
            for tok in copy {
                let v = UInt64(bitPattern: Int64(tok))
                h = (h ^ (v & 0xFF)) &* prime
                h = (h ^ ((v >> 8) & 0xFF)) &* prime
                h = (h ^ ((v >> 16) & 0xFF)) &* prime
                h = (h ^ ((v >> 24) & 0xFF)) &* prime
            }
            self.precomputedHash = Int(truncatingIfNeeded: h)
        }

        public func hash(into hasher: inout Hasher) {
            hasher.combine(precomputedHash)
        }

        public static func == (lhs: PageEdge, rhs: PageEdge) -> Bool {
            lhs.precomputedHash == rhs.precomputedHash && lhs.tokens == rhs.tokens
        }
    }

    public struct CheckpointMatch: Sendable {
        public let key: Key
        public let matchedTokens: Int
        public let isSharedBase: Bool
    }

    public struct SharedPageDonorMatch: Sendable {
        public let donorKey: Key
        public let pageAlignedTokens: Int
    }

    private struct StoredCheckpoint {
        let key: Key
        let tailTokens: [Int]
        let tokenCount: Int
        let allowsSuffix: Bool
        let isSharedBase: Bool
    }

    private final class Node {
        var children: [PageEdge: Node] = [:]
        var checkpoints: [StoredCheckpoint] = []
        var suffixCapableKeys: Set<Key> = []
    }

    public let pageTokens: Int
    private let lock = NSLock()
    private var rootsByMode: [String: Node] = [:]
    private var keyPaths: [Key: (mode: String, edges: [PageEdge])] = [:]
    private var recentPromptsByMode: [String: [[Int]]] = [:]
    private let maxRecentPromptsPerMode = 16

    public init(pageTokens: Int = 256) {
        self.pageTokens = max(1, pageTokens)
    }

    public func insert(
        key: Key,
        tokens: [Int],
        mode: String,
        allowsSuffix: Bool,
        isSharedBase: Bool
    ) {
        guard !tokens.isEmpty, !mode.isEmpty else { return }
        lock.withLock {
            removeLocked(key: key)
            let root = rootsByMode[mode] ?? {
                let created = Node()
                rootsByMode[mode] = created
                return created
            }()
            let fullPages = tokens.count / pageTokens
            var current = root
            var edges: [PageEdge] = []
            edges.reserveCapacity(fullPages)
            for p in 0..<fullPages {
                let start = p * pageTokens
                let edge = PageEdge(tokens: tokens[start..<(start + pageTokens)])
                let next = current.children[edge] ?? {
                    let created = Node()
                    current.children[edge] = created
                    return created
                }()
                if allowsSuffix {
                    next.suffixCapableKeys.insert(key)
                }
                edges.append(edge)
                current = next
            }
            let tailStart = fullPages * pageTokens
            let tail = tailStart < tokens.count ? Array(tokens[tailStart..<tokens.count]) : []
            current.checkpoints.append(StoredCheckpoint(
                key: key,
                tailTokens: tail,
                tokenCount: tokens.count,
                allowsSuffix: allowsSuffix,
                isSharedBase: isSharedBase
            ))
            current.checkpoints.sort { $0.tokenCount > $1.tokenCount }
            keyPaths[key] = (mode: mode, edges: edges)
            recordRecentPromptLocked(tokens: tokens, mode: mode)
        }
    }

    public func remove(key: Key) {
        lock.withLock {
            removeLocked(key: key)
        }
    }

    /// Finds the longest valid checkpoint prefix (`< tokens.count` or `<= tokens.count` when `allowExact`) in `O(pages)`.
    public func lookupLongestCheckpoint(
        tokens: [Int],
        mode: String,
        fallbackSharedBaseMode: String? = nil,
        allowExact: Bool = true
    ) -> CheckpointMatch? {
        guard !tokens.isEmpty else { return nil }
        return lock.withLock {
            if let hit = lookupInModeLocked(
                tokens: tokens, mode: mode, sharedBaseOnly: false, allowExact: allowExact
            ) {
                return hit
            }
            if let baseMode = fallbackSharedBaseMode, baseMode != mode {
                return lookupInModeLocked(
                    tokens: tokens, mode: baseMode, sharedBaseOnly: true, allowExact: allowExact
                )
            }
            return nil
        }
    }

    /// Returns a suffix-capable donor key sharing the deepest `256`-token page boundary `< tokens.count`.
    public func lookupDeepestSharedPage(
        tokens: [Int],
        mode: String,
        fallbackSharedBaseMode: String? = nil,
        minExclusiveTokens: Int = 0
    ) -> SharedPageDonorMatch? {
        let maxUsableTokens = tokens.count - 1
        let maxPages = maxUsableTokens / pageTokens
        guard maxPages > 0 else { return nil }
        return lock.withLock {
            if let donor = deepestDonorInModeLocked(
                tokens: tokens, mode: mode, maxPages: maxPages, minExclusiveTokens: minExclusiveTokens
            ) {
                return donor
            }
            if let baseMode = fallbackSharedBaseMode, baseMode != mode {
                return deepestDonorInModeLocked(
                    tokens: tokens, mode: baseMode, maxPages: maxPages, minExclusiveTokens: minExclusiveTokens
                )
            }
            return nil
        }
    }

    /// Computes the longest common prefix (`>= minTokens` and `< tokens.count`) with recently seen prompts in `mode`.
    public func longestBranchPrefixLength(
        tokens: [Int],
        mode: String,
        minTokens: Int = 128
    ) -> Int? {
        guard tokens.count > minTokens else { return nil }
        return lock.withLock {
            guard let recent = recentPromptsByMode[mode] else { return nil }
            var best = 0
            for prev in recent {
                let bound = min(tokens.count, prev.count)
                var common = 0
                while common < bound && tokens[common] == prev[common] {
                    common += 1
                }
                if common >= minTokens && common < tokens.count && common < prev.count {
                    best = max(best, common)
                }
            }
            return best >= minTokens ? best : nil
        }
    }

    private func recordRecentPromptLocked(tokens: [Int], mode: String) {
        var list = recentPromptsByMode[mode] ?? []
        if let dupIdx = list.firstIndex(of: tokens) {
            list.remove(at: dupIdx)
        }
        list.append(tokens)
        if list.count > maxRecentPromptsPerMode {
            list.removeFirst(list.count - maxRecentPromptsPerMode)
        }
        recentPromptsByMode[mode] = list
    }

    private func lookupInModeLocked(
        tokens: [Int],
        mode: String,
        sharedBaseOnly: Bool,
        allowExact: Bool
    ) -> CheckpointMatch? {
        guard let root = rootsByMode[mode] else { return nil }
        let queryPages = tokens.count / pageTokens
        var pathNodes: [Node] = [root]
        pathNodes.reserveCapacity(queryPages + 1)
        var current = root
        var matchedFullPages = 0
        for p in 0..<queryPages {
            let start = p * pageTokens
            let edge = PageEdge(tokens: tokens[start..<(start + pageTokens)])
            guard let next = current.children[edge] else { break }
            current = next
            matchedFullPages = p + 1
            pathNodes.append(next)
        }
        for depth in stride(from: matchedFullPages, through: 0, by: -1) {
            let node = pathNodes[depth]
            let baseCount = depth * pageTokens
            for cp in node.checkpoints {
                if sharedBaseOnly && !cp.isSharedBase { continue }
                if cp.tokenCount > tokens.count { continue }
                if cp.tokenCount == tokens.count {
                    if !allowExact { continue }
                } else if !cp.allowsSuffix {
                    continue
                }
                if cp.tailTokens.isEmpty {
                    return CheckpointMatch(key: cp.key, matchedTokens: cp.tokenCount, isSharedBase: cp.isSharedBase)
                }
                let remEnd = baseCount + cp.tailTokens.count
                guard remEnd <= tokens.count else { continue }
                var matchesTail = true
                for i in 0..<cp.tailTokens.count where tokens[baseCount + i] != cp.tailTokens[i] {
                    matchesTail = false
                    break
                }
                if matchesTail {
                    return CheckpointMatch(key: cp.key, matchedTokens: cp.tokenCount, isSharedBase: cp.isSharedBase)
                }
            }
        }
        return nil
    }

    private func deepestDonorInModeLocked(
        tokens: [Int],
        mode: String,
        maxPages: Int,
        minExclusiveTokens: Int
    ) -> SharedPageDonorMatch? {
        guard let root = rootsByMode[mode] else { return nil }
        var current = root
        var best: SharedPageDonorMatch?
        for p in 0..<maxPages {
            let start = p * pageTokens
            let edge = PageEdge(tokens: tokens[start..<(start + pageTokens)])
            guard let next = current.children[edge] else { break }
            current = next
            let alignedTokens = (p + 1) * pageTokens
            if alignedTokens > minExclusiveTokens, let donorKey = next.suffixCapableKeys.first {
                best = SharedPageDonorMatch(donorKey: donorKey, pageAlignedTokens: alignedTokens)
            }
        }
        return best
    }

    private func nodeAtPathLocked(mode: String, edges: [PageEdge]) -> Node? {
        guard var current = rootsByMode[mode] else { return nil }
        for edge in edges {
            guard let next = current.children[edge] else { return nil }
            current = next
        }
        return current
    }

    private func removeLocked(key: Key) {
        guard let info = keyPaths.removeValue(forKey: key),
              let root = rootsByMode[info.mode] else { return }
        var stack: [(parent: Node, edge: PageEdge, child: Node)] = []
        stack.reserveCapacity(info.edges.count)
        var current = root
        for edge in info.edges {
            guard let next = current.children[edge] else { return }
            next.suffixCapableKeys.remove(key)
            stack.append((current, edge, next))
            current = next
        }
        current.checkpoints.removeAll { $0.key == key }
        for (parent, edge, child) in stack.reversed() {
            if child.children.isEmpty && child.checkpoints.isEmpty && child.suffixCapableKeys.isEmpty {
                parent.children.removeValue(forKey: edge)
            } else {
                break
            }
        }
        if root.children.isEmpty && root.checkpoints.isEmpty {
            rootsByMode.removeValue(forKey: info.mode)
        }
    }
}
