// FNPrefixWarmStore.swift — Thin adapter around QWSlotPinnedLRUStore for FlashNext prefix cache entries.
// Unifies the warm RAM tier with QWEngine's prefix cache store.

import Foundation
import QWCore

public typealias FNPrefixCacheTier = QWPrefixCacheTier
public typealias FNPrefixCacheTiers = QWPrefixCacheTiers

/// The prefix cache's warm tier: backed by the unified `QWSlotPinnedLRUStore`.
public final class FNPrefixWarmStore: @unchecked Sendable {
    public static let defaultMaxSlots = QWSlotPinnedLRUStore<String, Data>.defaultMaxSlots
    public static var maxSlots: Int { defaultMaxSlots }
    public static let rootPrefixTokenCount = QWSlotPinnedLRUStore<String, Data>.defaultRootPrefixTokenCount

    private let core: QWSlotPinnedLRUStore<String, Data>
    public var onEvict: ((Int, Int) -> Void)?

    public init(byteLimit: Int, maxSlots: Int = FNPrefixWarmStore.defaultMaxSlots) {
        let store = QWSlotPinnedLRUStore<String, Data>(
            byteLimit: byteLimit,
            maxSlots: maxSlots,
            rootPrefixTokenCount: Self.rootPrefixTokenCount
        )
        self.core = store
        store.onEvict = { [weak self] _, entry in
            guard entry.charge > 0 else { return }
            self?.onEvict?(entry.tokenCount, entry.charge)
        }
    }

    public var maxSlots: Int { core.maxSlots }
    public var byteLimit: Int { core.byteLimit }
    public var isEnabled: Bool { core.isEnabled }
    public var state: (entries: Int, bytes: Int) { core.state }

    public func setMaxSlots(_ count: Int) {
        core.setMaxSlots(count)
    }

    public func ensureByteLimit(atLeast minimumBytes: Int) {
        core.ensureByteLimit(atLeast: minimumBytes)
    }

    public func removeSharedBaseEntries() {
        core.removeSharedBaseEntries()
    }

    public var hasSharedBase: Bool { core.hasSharedBase }

    public func canHold(bytes: Int) -> Bool {
        core.isEnabled && (core.sharedBaseBytes + bytes <= core.byteLimit)
    }

    public func isProtectedLastEntry(_ name: String, excludingSlot: Int? = nil) -> Bool {
        core.isProtectedLastKey(name, excludingSlot: excludingSlot)
    }

    public func isSharedBaseEntry(_ name: String) -> Bool {
        core.isSharedBaseKey(name)
    }

    public static func rootHash(history: [Int64], kvMode: FNKVStorageMode, hasMTP: Bool) -> UInt64 {
        let n = min(rootPrefixTokenCount, history.count)
        return FNGPUEngine.fnPrefixHash(tokens: history.prefix(n), kvMode: kvMode, hasMTP: hasMTP)
    }

    public func sharedBranchPrefixLength(for history: [Int64]) -> Int? {
        core.sharedBranchPrefixLength(for: history)
    }

    public func resolveSlot(rootHash: UInt64, history: [Int64], kvMode: FNKVStorageMode, hasMTP: Bool) -> Int {
        core.resolveSlot(rootHash: rootHash, history: history) { name, count in
            let expected = FNGPUEngine.fnPrefixHash(tokens: history.prefix(count), kvMode: kvMode, hasMTP: hasMTP)
            return Self.hash(in: name) == String(format: "%016llx", expected)
        }
    }

    public func prepareSlotForStore(
        slot: Int,
        rootHash: UInt64,
        history: [Int64],
        incomingBytes: Int,
        kvMode: FNKVStorageMode,
        hasMTP: Bool
    ) {
        core.prepareSlotForStore(slot: slot, history: history, incomingBytes: incomingBytes) { name, count in
            let expected = FNGPUEngine.fnPrefixHash(tokens: history.prefix(count), kvMode: kvMode, hasMTP: hasMTP)
            return Self.hash(in: name) == String(format: "%016llx", expected)
        }
    }

    public func makeRoom(for incomingBytes: Int, activeSlot: Int? = nil) {
        core.makeRoom(for: incomingBytes, activeSlot: activeSlot)
    }

    public func evictPrefixes(of history: [Int64], kvMode: FNKVStorageMode, hasMTP: Bool) {
        core.evictMatchingPrefixes(maxTokenCount: history.count) { name, count in
            let expected = FNGPUEngine.fnPrefixHash(tokens: history.prefix(count), kvMode: kvMode, hasMTP: hasMTP)
            return Self.hash(in: name) == String(format: "%016llx", expected)
        }
    }

    public func store(name: String, blob: Data, slot: Int? = nil, rootHash: UInt64? = nil,
                      history: [Int64]? = nil, isSharedBase: Bool = false) {
        guard core.isEnabled else { return }
        core.forget(name)
        guard blob.count <= core.byteLimit else { return }
        let tokCount = Self.tokenCount(in: name) ?? (history?.count ?? 0)
        _ = core.store(
            key: name,
            value: blob,
            charge: blob.count,
            tokenCount: tokCount,
            allowsSuffix: true,
            slot: slot,
            rootHash: rootHash,
            history: history,
            isSharedBase: isSharedBase
        )
    }

    public func blob(name: String) -> Data? {
        core.value(for: name, touchOnHit: true)?.value
    }

    public func contains(name: String) -> Bool {
        core.contains(name)
    }

    public func longestMatch(history: [Int64], kvMode: FNKVStorageMode,
                             hasMTP: Bool) -> (name: String, tokens: Int)? {
        let queryRootHash: UInt64? = history.count >= Self.rootPrefixTokenCount
            ? Self.rootHash(history: history, kvMode: kvMode, hasMTP: hasMTP)
            : nil
        var candidates: [(name: String, tokens: Int)] = []
        candidates.reserveCapacity(core.order.count)
        for name in core.order {
            guard let count = Self.tokenCount(in: name), count > 0, count < history.count else { continue }
            if count >= Self.rootPrefixTokenCount,
               let qHash = queryRootHash,
               let eHash = core.entries[name]?.rootHash,
               qHash != eHash {
                continue
            }
            candidates.append((name, count))
        }
        candidates.sort { $0.tokens > $1.tokens }
        for cand in candidates {
            let expected = FNGPUEngine.fnPrefixHash(tokens: history.prefix(cand.tokens),
                                                     kvMode: kvMode, hasMTP: hasMTP)
            if Self.hash(in: cand.name) == String(format: "%016llx", expected) {
                core.touchSlot(for: cand.name)
                return cand
            }
        }
        return nil
    }

    static func tokenCount(in name: String) -> Int? {
        let parts = entryParts(name)
        guard parts.count == 4 else { return nil }
        return Int(parts[3])
    }

    static func hash(in name: String) -> String? {
        let parts = entryParts(name)
        guard parts.count == 4 else { return nil }
        return String(parts[2])
    }

    private static func entryParts(_ name: String) -> [Substring] {
        var stem = Substring(name)
        if let dot = stem.lastIndex(of: ".") { stem = stem[stem.startIndex..<dot] }
        return stem.split(separator: "_")
    }
}
