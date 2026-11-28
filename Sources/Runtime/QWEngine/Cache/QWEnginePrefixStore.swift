// QWEnginePrefixStore.swift — Driver prefix store backed by QWSlotPinnedLRUStore + QWRadixPrefixTree.
import Foundation
import QWCore
import QWKV

final class QWEnginePrefixStore<Value: Sendable> {
    struct Key: Hashable, Sendable {
        let tokens: [Int]
        let mode: String
    }
    struct Hit {
        let value: Value
        let tokenCount: Int
    }

    static var defaultMaxSlots: Int { QWSlotPinnedLRUStore<Key, Value>.defaultMaxSlots }
    let vocab: Int
    private let core: QWSlotPinnedLRUStore<Key, Value>
    private let radixTree = QWRadixPrefixTree<Key>(pageTokens: QWKVCache.pageTokens)
    private let pageLedger = QWPrefixPageLedger<Key>()
    private let extractPhysicalPages: (@Sendable (Value) -> Set<QWPrefixPhysicalPage>)?
    var onEvict: ((Key, Value, Bool) -> Void)?

    init(byteLimit: Int, entryLimit: Int = 16, vocab: Int,
         maxSlots: Int = QWEnginePrefixStore.defaultMaxSlots,
         extractPhysicalPages: (@Sendable (Value) -> Set<QWPrefixPhysicalPage>)? = nil) throws {
        guard byteLimit > 0, entryLimit > 0, vocab > 0 else {
            throw QWEngineDriverError.state("prefix store limits must be positive")
        }
        self.vocab = vocab
        self.extractPhysicalPages = extractPhysicalPages
        let store = QWSlotPinnedLRUStore<Key, Value>(
            byteLimit: byteLimit,
            entryLimit: entryLimit,
            maxSlots: maxSlots,
            rootPrefixTokenCount: 64
        )
        self.core = store
        store.onRemove = { [weak self] key in
            self?.radixTree.remove(key: key)
            self?.pageLedger.remove(key: key)
        }
        store.onEvict = { [weak self] key, entry in
            self?.onEvict?(key, entry.value, entry.allowsSuffix)
        }
    }

    var maxSlots: Int { core.maxSlots }
    var byteLimit: Int { core.byteLimit }
    var entryLimit: Int { core.entryLimit }
    var count: Int { core.count }
    var chargedBytes: Int {
        if QWRadixKVGate.isEnabled && extractPhysicalPages != nil {
            return core.chargedBytes + pageLedger.uniqueBytes
        }
        return core.chargedBytes
    }
    var evictions: Int { core.evictions }
    var hasSharedBase: Bool { core.hasSharedBase }

    /// Non-mutating lookup checking whether `tokens` + `mode` is currently held in the warm or radix prefix store.
    func hasCachedPrefix(tokens: [Int], mode: String) -> Bool {
        guard !tokens.isEmpty, !mode.isEmpty,
              tokens.allSatisfy({ (0..<vocab).contains($0) }) else {
            return false
        }
        let base = Self.baseMode(of: mode)
        let exact = Key(tokens: tokens, mode: mode)
        if core.value(for: exact, touchOnHit: false) != nil {
            return true
        }
        if base != mode {
            let saltedBase = Key(tokens: tokens, mode: base)
            if core.isSharedBaseKey(saltedBase), core.value(for: saltedBase, touchOnHit: false) != nil {
                return true
            }
        }
        if QWRadixKVGate.isEnabled,
           let radixHit = radixTree.lookupLongestCheckpoint(
               tokens: tokens,
               mode: mode,
               fallbackSharedBaseMode: base != mode ? base : nil,
               allowExact: true
           ),
           radixHit.matchedTokens == tokens.count,
           core.value(for: radixHit.key, touchOnHit: false) != nil {
            return true
        }
        return false
    }

    static func baseMode(of mode: String) -> String {
        guard let range = mode.range(of: "|salt=") else { return mode }
        return String(mode[..<range.lowerBound])
    }

    func sharedBasePrefix(matching tokens: [Int], mode: String) -> Hit? {
        let base = Self.baseMode(of: mode)
        var best: Key?
        for key in core.sharedBaseKeys where (key.mode == mode || key.mode == base) &&
            key.tokens.count <= tokens.count && key.tokens.count > (best?.tokens.count ?? 0) {
            if core.entries[key] != nil && tokens.starts(with: key.tokens) {
                best = key
            }
        }
        guard let best, let entry = core.value(for: best, touchOnHit: false) else { return nil }
        return Hit(value: entry.value, tokenCount: best.tokens.count)
    }

    func sharedBranchPrefixLength(for tokens: [Int], mode: String? = nil) -> Int? {
        let slotBranch = core.sharedBranchPrefixLength(for: tokens.map(Int64.init))
        guard QWRadixKVGate.isEnabled, let mode else { return slotBranch }
        let radixBranch = radixTree.longestBranchPrefixLength(
            tokens: tokens, mode: mode, minTokens: core.rootPrefixTokenCount
        )
        switch (slotBranch, radixBranch) {
        case (let a?, let b?): return max(a, b)
        case (let a?, nil): return a
        case (nil, let b?): return b
        case (nil, nil): return nil
        }
    }

    func ensureByteLimit(atLeast minimumBytes: Int) {
        core.ensureByteLimit(atLeast: minimumBytes)
    }

    func removeSharedBaseEntries() {
        core.removeSharedBaseEntries()
    }

    func lookup(tokens: [Int], mode: String) throws -> Hit? {
        try validate(tokens, mode: mode)
        let base = Self.baseMode(of: mode)
        if QWRadixKVGate.isEnabled,
           let radixHit = radixTree.lookupLongestCheckpoint(
               tokens: tokens,
               mode: mode,
               fallbackSharedBaseMode: base != mode ? base : nil,
               allowExact: true
           ),
           let entry = core.value(for: radixHit.key, touchOnHit: true) {
            return Hit(value: entry.value, tokenCount: radixHit.matchedTokens)
        }
        let exact = Key(tokens: tokens, mode: mode)
        let saltedBase = base != mode ? Key(tokens: tokens, mode: base) : nil
        let queryRootHash: UInt64? = tokens.count >= 64
            ? QWPrefixHash.fnv1a64(tokens: Array(tokens.prefix(64)), mode: mode)
            : nil
        let hit = QWTwoTierPrefixCache.lookupWarmPrefix(
            in: core,
            exactKey: exact,
            saltedBaseKey: saltedBase,
            tokenCount: tokens.count,
            queryRootHash: queryRootHash,
            allowExact: true,
            allowPrefix: true,
            sharedBaseOnly: false,
            touchOnHit: true,
            isCandidate: { key, _, _ in
                key.mode == mode || (base != mode && key.mode == base && core.isSharedBaseKey(key))
            },
            matchesPrefix: { key, _, _ in
                tokens.starts(with: key.tokens)
            }
        )
        return hit.map { Hit(value: $0.value, tokenCount: $0.matchedTokens) }
    }

    func lookupDeepestSharedPageDonor(
        tokens: [Int],
        mode: String,
        minExclusiveTokens: Int = 0
    ) throws -> (donor: Value, pageAlignedTokens: Int)? {
        try validate(tokens, mode: mode)
        guard QWRadixKVGate.isEnabled else { return nil }
        let base = Self.baseMode(of: mode)
        guard let match = radixTree.lookupDeepestSharedPage(
            tokens: tokens,
            mode: mode,
            fallbackSharedBaseMode: base != mode ? base : nil,
            minExclusiveTokens: minExclusiveTokens
        ),
              let entry = core.value(for: match.donorKey, touchOnHit: true) else {
            return nil
        }
        return (entry.value, match.pageAlignedTokens)
    }

    /// Oversized valid entries are not retained. The caller records that result.
    @discardableResult
    func insert(tokens: [Int], mode: String, value: Value, payloadBytes: Int,
                allowsSuffix: Bool = false, isSharedBase: Bool = false, pinToSlot: Bool = true) throws -> Bool {
        let keyBytes = try keyBytes(tokens: tokens, mode: mode)
        guard payloadBytes >= 0 else {
            throw QWEngineDriverError.state("invalid prefix payload charge")
        }
        let key = Key(tokens: tokens, mode: mode)
        let history64 = tokens.map(Int64.init)
        let rootHash = QWPrefixHash.fnv1a64(tokens: Array(tokens.prefix(64)), mode: mode)

        if QWRadixKVGate.isEnabled, let extract = extractPhysicalPages {
            let pages = extract(value)
            let ownPageBytes = QWPrefixPageLedger<Key>.totalBytes(of: pages)
            let fixedPayloadBytes = max(0, payloadBytes - ownPageBytes)
            guard keyBytes <= byteLimit, fixedPayloadBytes <= byteLimit - keyBytes else { return false }
            let coreCharge = fixedPayloadBytes + keyBytes
            while core.count > 0 {
                let projectedPages = pageLedger.projectedUniqueBytes(adding: pages, replacing: key)
                let priorFixed = core.value(for: key, touchOnHit: false)?.charge ?? 0
                let totalAfter = (core.chargedBytes - priorFixed) + coreCharge + projectedPages
                if totalAfter <= byteLimit { break }
                if !core.evictOldest() { break }
            }
            let projectedPages = pageLedger.projectedUniqueBytes(adding: pages, replacing: key)
            let priorFixed = core.value(for: key, touchOnHit: false)?.charge ?? 0
            guard (core.chargedBytes - priorFixed) + coreCharge + projectedPages <= byteLimit else {
                return false
            }
            let stored: Bool
            if !pinToSlot && !isSharedBase {
                stored = core.store(
                    key: key, value: value, charge: coreCharge, tokenCount: tokens.count,
                    allowsSuffix: allowsSuffix, slot: nil, rootHash: rootHash,
                    history: history64, isSharedBase: false, isBranchCheckpoint: true
                )
            } else {
                stored = QWTwoTierPrefixCache.storeWithSlotPinning(
                    in: core,
                    key: key,
                    value: value,
                    charge: coreCharge,
                    tokenCount: tokens.count,
                    allowsSuffix: allowsSuffix,
                    rootHash: rootHash,
                    history: history64,
                    isSharedBase: isSharedBase
                ) { cand, _ in
                    cand.mode == key.mode && key.tokens.starts(with: cand.tokens)
                }
            }
            if stored {
                pageLedger.insert(key: key, pages: pages)
                radixTree.insert(
                    key: key, tokens: tokens, mode: mode,
                    allowsSuffix: allowsSuffix, isSharedBase: isSharedBase
                )
                while core.count > 1 && (core.chargedBytes + pageLedger.uniqueBytes > byteLimit) {
                    if !core.evictOldest() { break }
                }
            }
            return stored
        }

        guard keyBytes <= byteLimit, payloadBytes <= byteLimit - keyBytes else { return false }
        let charge = payloadBytes + keyBytes
        if !pinToSlot && !isSharedBase {
            return core.store(
                key: key, value: value, charge: charge, tokenCount: tokens.count,
                allowsSuffix: allowsSuffix, slot: nil, rootHash: rootHash,
                history: history64, isSharedBase: false, isBranchCheckpoint: true
            )
        }
        let stored = QWTwoTierPrefixCache.storeWithSlotPinning(
            in: core,
            key: key,
            value: value,
            charge: charge,
            tokenCount: tokens.count,
            allowsSuffix: allowsSuffix,
            rootHash: rootHash,
            history: history64,
            isSharedBase: isSharedBase
        ) { cand, _ in
            cand.mode == key.mode && key.tokens.starts(with: cand.tokens)
        }
        return stored
    }

    func isProtectedLastKey(_ key: Key, excludingSlot: Int? = nil) -> Bool {
        core.isProtectedLastKey(key, excludingSlot: excludingSlot)
    }

    @discardableResult
    func evictOldest(excludingSlot: Int? = nil) -> Bool {
        core.evictOldest(excludingSlot: excludingSlot)
    }

    func keyBytes(tokens: [Int], mode: String) throws -> Int {
        try validate(tokens, mode: mode)
        guard tokens.count <= Int.max / MemoryLayout<Int>.stride else {
            throw QWEngineDriverError.state("prefix key byte count overflows")
        }
        let tokenBytes = tokens.count * MemoryLayout<Int>.stride
        let modeBytes = mode.utf8.count
        guard modeBytes <= Int.max - tokenBytes else {
            throw QWEngineDriverError.state("prefix policy byte count overflows")
        }
        return tokenBytes + modeBytes
    }

    private func validate(_ tokens: [Int], mode: String) throws {
        guard !tokens.isEmpty, !mode.isEmpty,
              tokens.allSatisfy({ (0..<vocab).contains($0) }) else {
            throw QWEngineDriverError.state("invalid prefix token or policy key")
        }
    }
}

struct QWEngineCachedPrefix: Sendable {
    let prefix: QWEngineSharedPrefix
    let nextToken: Int
    let logits: Data
    var memoryBytes: Int { prefix.memoryBytes + logits.count }
}
