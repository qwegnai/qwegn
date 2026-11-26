// QWSessionTree.swift — Branching/forking/checkpointing session tree backed by Radix KV pin refcounts (G4).

import Foundation
import QWCore

public struct QWSessionNode: Codable, Sendable, Equatable {
    public let id: String
    public let parentId: String?
    public let label: String?
    public let modelName: String
    public let kvMode: String
    public let cacheSalt: String?
    public let tokenIds: [Int]
    public let prefixHash: UInt64
    public let recurrentCheckpointId: String?
    public let createdAtEpochMs: Int64
    public let metadata: [String: String]
    public internal(set) var childIds: [String]

    enum CodingKeys: String, CodingKey {
        case id, label, metadata
        case parentId = "parent_id"
        case modelName = "model_name"
        case kvMode = "kv_mode"
        case cacheSalt = "cache_salt"
        case tokenIds = "token_ids"
        case prefixHash = "prefix_hash"
        case recurrentCheckpointId = "recurrent_checkpoint_id"
        case createdAtEpochMs = "created_at_epoch_ms"
        case childIds = "child_ids"
    }
}

public struct QWSessionRadixPin: Codable, Sendable, Equatable {
    public let keyHash: UInt64
    public let mode: String
    public let tokenCount: Int
    public internal(set) var refCount: Int

    enum CodingKeys: String, CodingKey {
        case mode
        case keyHash = "key_hash"
        case tokenCount = "token_count"
        case refCount = "ref_count"
    }
}

public struct QWSessionRestoreResult: Codable, Sendable, Equatable {
    public let node: QWSessionNode
    public let kvCacheHit: Bool
    public let needsReplay: Bool
    public let pinnedPrefixTokens: Int

    enum CodingKeys: String, CodingKey {
        case node
        case kvCacheHit = "kv_cache_hit"
        case needsReplay = "needs_replay"
        case pinnedPrefixTokens = "pinned_prefix_tokens"
    }
}

public struct QWSessionTreeStats: Codable, Sendable, Equatable {
    public let nodeCount: Int
    public let rootCount: Int
    public let uniquePinnedPrefixes: Int
    public let totalPinRefs: Int
    public let journalPath: String?

    enum CodingKeys: String, CodingKey {
        case nodeCount = "node_count"
        case rootCount = "root_count"
        case uniquePinnedPrefixes = "unique_pinned_prefixes"
        case totalPinRefs = "total_pin_refs"
        case journalPath = "journal_path"
    }
}

public enum QWSessionTreeError: Error, Sendable, CustomStringConvertible, Equatable {
    case nodeNotFound(String)
    case invalidForkCount(Int)
    case cannotPruneWithChildren(String)
    case journalFailed(String)

    public var description: String {
        switch self {
        case .nodeNotFound(let id):
            return "session node '\(id)' not found"
        case .invalidForkCount(let count):
            return "invalid fork count \(count); must be in 1...64"
        case .cannotPruneWithChildren(let id):
            return "cannot prune non-leaf session node '\(id)' without cascade=true"
        case .journalFailed(let msg):
            return "session journal failed: \(msg)"
        }
    }
}

/// Thread-safe tree of conversation checkpoints supporting O(1) COW forks and Radix pin tracking.
public final class QWSessionTree: @unchecked Sendable {
    struct PinKey: Hashable, Sendable {
        let prefixHash: UInt64
        let mode: String
    }

    public let journalURL: URL?
    public let maxNodes: Int
    let lock = NSLock()
    var nodes: [String: QWSessionNode] = [:]
    var insertionOrder: [String] = []
    var pins: [PinKey: QWSessionRadixPin] = [:]

    public init(journalURL: URL? = nil, maxNodes: Int = 4096) {
        self.journalURL = journalURL?.standardizedFileURL
        self.maxNodes = max(1, maxNodes)
        if let url = self.journalURL {
            replayJournalLocked(from: url)
        }
    }

    static func effectiveMode(kvMode: String, cacheSalt: String?) -> String {
        let base = kvMode.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "f16" : kvMode
        if let salt = cacheSalt?.trimmingCharacters(in: .whitespacesAndNewlines), !salt.isEmpty {
            return "\(base)|salt=\(salt)"
        }
        return base
    }

    static func computePrefixHash(tokenIds: [Int], kvMode: String, cacheSalt: String?) -> UInt64 {
        let mode = effectiveMode(kvMode: kvMode, cacheSalt: cacheSalt)
        return QWPrefixHash.fnv1a64(tokens: tokenIds, mode: mode)
    }

    @discardableResult
    public func checkpoint(
        parentId: String? = nil,
        label: String? = nil,
        modelName: String,
        kvMode: String = "f16",
        cacheSalt: String? = nil,
        tokenIds: [Int],
        recurrentCheckpointId: String? = nil,
        metadata: [String: String] = [:],
        id explicitId: String? = nil
    ) throws -> QWSessionNode {
        let resolvedId = (explicitId?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false)
            ? explicitId!.trimmingCharacters(in: .whitespacesAndNewlines)
            : "sess_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(20)
        let nowMs = Int64(Date().timeIntervalSince1970 * 1000)
        return try lock.withLock {
            if let parentId, nodes[parentId] == nil {
                throw QWSessionTreeError.nodeNotFound(parentId)
            }
            let hash = Self.computePrefixHash(tokenIds: tokenIds, kvMode: kvMode, cacheSalt: cacheSalt)
            let node = QWSessionNode(
                id: resolvedId,
                parentId: parentId,
                label: label,
                modelName: modelName,
                kvMode: kvMode,
                cacheSalt: cacheSalt,
                tokenIds: tokenIds,
                prefixHash: hash,
                recurrentCheckpointId: recurrentCheckpointId,
                createdAtEpochMs: nowMs,
                metadata: metadata,
                childIds: []
            )
            insertNodeLocked(node)
            try appendJournalRecordLocked(.checkpoint(node: node))
            return node
        }
    }

    public func fork(
        from parentId: String,
        count: Int,
        labels: [String]? = nil,
        metadata: [String: String] = [:]
    ) throws -> [QWSessionNode] {
        guard count >= 1 && count <= 64 else {
            throw QWSessionTreeError.invalidForkCount(count)
        }
        let nowMs = Int64(Date().timeIntervalSince1970 * 1000)
        return try lock.withLock {
            guard let parent = nodes[parentId] else {
                throw QWSessionTreeError.nodeNotFound(parentId)
            }
            var created: [QWSessionNode] = []
            created.reserveCapacity(count)
            for idx in 0..<count {
                let cid = "sess_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(20)
                let forkLabel = (labels != nil && idx < labels!.count) ? labels![idx] : (parent.label.map { "\($0)#\(idx + 1)" })
                var mergedMeta = parent.metadata
                for (k, v) in metadata { mergedMeta[k] = v }
                mergedMeta["fork_index"] = "\(idx)"
                let child = QWSessionNode(
                    id: cid,
                    parentId: parent.id,
                    label: forkLabel,
                    modelName: parent.modelName,
                    kvMode: parent.kvMode,
                    cacheSalt: parent.cacheSalt,
                    tokenIds: parent.tokenIds,
                    prefixHash: parent.prefixHash,
                    recurrentCheckpointId: parent.recurrentCheckpointId,
                    createdAtEpochMs: nowMs,
                    metadata: mergedMeta,
                    childIds: []
                )
                insertNodeLocked(child)
                try appendJournalRecordLocked(.checkpoint(node: child))
                created.append(child)
            }
            return created
        }
    }

    public func restore(
        nodeId: String,
        cacheProbe: (@Sendable ([Int], String) -> Bool)? = nil
    ) throws -> QWSessionRestoreResult {
        let node: QWSessionNode = try lock.withLock {
            guard let found = nodes[nodeId] else {
                throw QWSessionTreeError.nodeNotFound(nodeId)
            }
            return found
        }
        let hit: Bool
        if let cacheProbe, !node.tokenIds.isEmpty {
            let mode = Self.effectiveMode(kvMode: node.kvMode, cacheSalt: node.cacheSalt)
            hit = cacheProbe(node.tokenIds, mode)
        } else {
            hit = false
        }
        return QWSessionRestoreResult(
            node: node,
            kvCacheHit: hit,
            needsReplay: !hit,
            pinnedPrefixTokens: node.tokenIds.count
        )
    }

    @discardableResult
    public func prune(nodeId: String, cascade: Bool = false) throws -> [String] {
        try lock.withLock {
            guard let target = nodes[nodeId] else {
                throw QWSessionTreeError.nodeNotFound(nodeId)
            }
            if !cascade && !target.childIds.isEmpty {
                throw QWSessionTreeError.cannotPruneWithChildren(nodeId)
            }
            var removed: [String] = []
            collectSubtreeLocked(nodeId, into: &removed)
            if let pid = target.parentId, var parent = nodes[pid] {
                parent.childIds.removeAll { $0 == nodeId }
                nodes[pid] = parent
            }
            for rid in removed {
                if let rNode = nodes.removeValue(forKey: rid) {
                    decrementPinLocked(for: rNode)
                }
            }
            let removedSet = Set(removed)
            insertionOrder.removeAll { removedSet.contains($0) }
            try appendJournalRecordLocked(.prune(nodeIds: removed))
            return removed
        }
    }

    public func node(id: String) -> QWSessionNode? {
        lock.withLock { nodes[id] }
    }

    public func lineage(of nodeId: String) throws -> [QWSessionNode] {
        try lock.withLock {
            guard var cur = nodes[nodeId] else {
                throw QWSessionTreeError.nodeNotFound(nodeId)
            }
            var chain: [QWSessionNode] = [cur]
            var guardCount = 0
            while let pid = cur.parentId, let parent = nodes[pid], guardCount < 4096 {
                chain.append(parent)
                cur = parent
                guardCount += 1
            }
            return chain.reversed()
        }
    }

    public func listNodes(modelName: String? = nil, limit: Int = 100) -> [QWSessionNode] {
        lock.withLock {
            let all = insertionOrder.reversed().compactMap { nodes[$0] }
            let filtered: [QWSessionNode]
            if let modelName, !modelName.isEmpty {
                filtered = all.filter { $0.modelName.caseInsensitiveCompare(modelName) == .orderedSame }
            } else {
                filtered = all
            }
            return Array(filtered.prefix(max(1, limit)))
        }
    }

    public func pinRefCount(tokenIds: [Int], kvMode: String = "f16", cacheSalt: String? = nil) -> Int {
        lock.withLock {
            let key = pinKey(tokenIds: tokenIds, kvMode: kvMode, cacheSalt: cacheSalt)
            return pins[key]?.refCount ?? 0
        }
    }

    public func stats() -> QWSessionTreeStats {
        lock.withLock {
            let roots = nodes.values.filter { $0.parentId == nil }.count
            let totalRefs = pins.values.reduce(0) { $0 + $1.refCount }
            return QWSessionTreeStats(
                nodeCount: nodes.count,
                rootCount: roots,
                uniquePinnedPrefixes: pins.count,
                totalPinRefs: totalRefs,
                journalPath: journalURL?.path
            )
        }
    }

    func insertNodeLocked(_ node: QWSessionNode) {
        if nodes[node.id] == nil {
            insertionOrder.append(node.id)
        }
        nodes[node.id] = node
        if let pid = node.parentId, var parent = nodes[pid], !parent.childIds.contains(node.id) {
            parent.childIds.append(node.id)
            nodes[pid] = parent
        }
        incrementPinLocked(for: node)
        evictOldestLeafIfNeededLocked(preserving: node.id)
    }

    private func pinKey(tokenIds: [Int], kvMode: String, cacheSalt: String?) -> PinKey {
        let mode = Self.effectiveMode(kvMode: kvMode, cacheSalt: cacheSalt)
        let hash = Self.computePrefixHash(tokenIds: tokenIds, kvMode: kvMode, cacheSalt: cacheSalt)
        return PinKey(prefixHash: hash, mode: mode)
    }

    func incrementPinLocked(for node: QWSessionNode) {
        guard !node.tokenIds.isEmpty else { return }
        let key = PinKey(prefixHash: node.prefixHash, mode: Self.effectiveMode(kvMode: node.kvMode, cacheSalt: node.cacheSalt))
        if var existing = pins[key] {
            existing.refCount += 1
            pins[key] = existing
        } else {
            pins[key] = QWSessionRadixPin(
                keyHash: node.prefixHash, mode: key.mode, tokenCount: node.tokenIds.count, refCount: 1
            )
        }
    }

    func decrementPinLocked(for node: QWSessionNode) {
        guard !node.tokenIds.isEmpty else { return }
        let key = PinKey(prefixHash: node.prefixHash, mode: Self.effectiveMode(kvMode: node.kvMode, cacheSalt: node.cacheSalt))
        guard var existing = pins[key] else { return }
        existing.refCount -= 1
        if existing.refCount <= 0 {
            pins.removeValue(forKey: key)
        } else {
            pins[key] = existing
        }
    }

    private func collectSubtreeLocked(_ id: String, into out: inout [String]) {
        guard let n = nodes[id] else { return }
        for cid in n.childIds { collectSubtreeLocked(cid, into: &out) }
        out.append(id)
    }

    private func evictOldestLeafIfNeededLocked(preserving keptId: String) {
        while nodes.count > maxNodes {
            guard let victimId = insertionOrder.first(where: { $0 != keptId && (nodes[$0]?.childIds.isEmpty ?? true) }),
                  let victim = nodes.removeValue(forKey: victimId) else {
                break
            }
            if let pid = victim.parentId, var parent = nodes[pid] {
                parent.childIds.removeAll { $0 == victimId }
                nodes[pid] = parent
            }
            decrementPinLocked(for: victim)
            insertionOrder.removeAll { $0 == victimId }
        }
    }
}
