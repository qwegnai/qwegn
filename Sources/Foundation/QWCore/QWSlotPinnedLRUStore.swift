// QWSlotPinnedLRUStore.swift — Unified Three-Tier Prefix Cache primitives (Tier-0 watermarks, Tier-1 slot-pinned LRU, Tier-2 cold helpers).
// Shared across QWEngine and QWFlashNext so all model families use one prefix cache architecture.

import Foundation

/// The tier a prefix cache hit came from (`warm_ram` or `cold_ssd`).
public enum QWPrefixCacheTier: String, Sendable {
    case warmRAM = "warm_ram"
    case coldSSD = "cold_ssd"
}

/// Byte budgets for the warm RAM and cold SSD prefix cache tiers.
public struct QWPrefixCacheTiers: Sendable {
    public let warmByteLimit: Int
    public let ssdByteLimit: Int

    public init(warmByteLimit: Int, ssdByteLimit: Int) {
        self.warmByteLimit = warmByteLimit > 0 ? warmByteLimit : 0
        self.ssdByteLimit = warmByteLimit > 0 ? max(0, ssdByteLimit) : 0
    }

    public var isEnabled: Bool { warmByteLimit > 0 }
}

/// Deterministic FNV-1a-64 prefix hashing shared across warm RAM and cold SSD cache tiers.
public enum QWPrefixHash {
    public static func fnv1a64(tokens: ArraySlice<Int64>, modeTag: String, hasMTP: Bool) -> UInt64 {
        var h: UInt64 = 0xcbf29ce484222325
        let prime: UInt64 = 0x100000001b3
        for b in modeTag.utf8 {
            h ^= UInt64(b)
            h = h &* prime
        }
        h ^= UInt64(hasMTP ? 1 : 0)
        h = h &* prime
        h ^= UInt64(bitPattern: Int64(tokens.count))
        h = h &* prime
        tokens.withUnsafeBufferPointer { buf in
            for tok in buf {
                let v = UInt64(bitPattern: tok)
                h = (h ^ (v & 0xFF)) &* prime
                h = (h ^ ((v >> 8) & 0xFF)) &* prime
                h = (h ^ ((v >> 16) & 0xFF)) &* prime
                h = (h ^ ((v >> 24) & 0xFF)) &* prime
                h = (h ^ ((v >> 32) & 0xFF)) &* prime
                h = (h ^ ((v >> 40) & 0xFF)) &* prime
                h = (h ^ ((v >> 48) & 0xFF)) &* prime
                h = (h ^ ((v >> 56) & 0xFF)) &* prime
            }
        }
        return h
    }

    public static func fnv1a64(tokens: [Int], mode: String) -> UInt64 {
        var hash: UInt64 = 0xcbf29ce484222325
        let prime: UInt64 = 0x100000001b3
        for byte in mode.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* prime
        }
        hash ^= UInt64(bitPattern: Int64(tokens.count))
        hash = hash &* prime
        for t in tokens {
            let v = UInt64(bitPattern: Int64(t))
            hash ^= (v & 0xFF); hash = hash &* prime
            hash ^= ((v >> 8) & 0xFF); hash = hash &* prime
            hash ^= ((v >> 16) & 0xFF); hash = hash &* prime
            hash ^= ((v >> 24) & 0xFF); hash = hash &* prime
        }
        return hash
    }
}

/// Tier-0 in-place GPU KV continuation tracker per partition slot (`0..<maxSlots`).
public struct QWResidentKVWatermarkTracker<Token: Equatable & Sendable>: Sendable {
    public var currentPartitionSlot: Int = 0
    public var currentPartitionTotalSlots: Int = 1
    public var residentPartitionTotalSlots: Int = 1
    public var historyBySlot: [Int: [Token]] = [:]
    public var hasMTPBySlot: [Int: Bool] = [:]

    public init() {}

    public var currentHistory: [Token] {
        get { historyBySlot[currentPartitionSlot, default: []] }
        set { historyBySlot[currentPartitionSlot] = newValue }
    }

    public var currentHasMTP: Bool {
        get { hasMTPBySlot[currentPartitionSlot, default: false] }
        set { hasMTPBySlot[currentPartitionSlot] = newValue }
    }

    /// Returns true if the GPU KV arena in `currentPartitionSlot` already holds `expectedTokens` in `0..<N`.
    public func canReuseInPlace(expectedTokens: ArraySlice<Token>, hasMTP: Bool) -> Bool {
        let N = expectedTokens.count
        let hist = currentHistory
        return currentPartitionTotalSlots == residentPartitionTotalSlots
            && currentHasMTP == hasMTP
            && hist.count >= N
            && hist.prefix(N) == expectedTokens
    }

    public mutating func recordRestored(expectedTokens: ArraySlice<Token>, hasMTP: Bool, inPlace: Bool) {
        let N = expectedTokens.count
        if inPlace {
            let curCount = currentHistory.count
            if curCount > N {
                historyBySlot[currentPartitionSlot]?.removeSubrange(N..<curCount)
            }
        } else {
            currentHistory = Array(expectedTokens)
            currentHasMTP = hasMTP
        }
    }

    public mutating func invalidateAll() {
        historyBySlot.removeAll(keepingCapacity: true)
        hasMTPBySlot.removeAll(keepingCapacity: true)
    }
}

/// Shared cold SSD directory helpers for `.qwcache` files.
public enum QWPrefixColdStore {
    public static func diskState(
        in dir: URL,
        filePrefix: String = "fn_entry_",
        pathExtension: String = "qwcache",
        syncQueue: DispatchQueue? = nil
    ) -> (entries: Int, bytes: Int)? {
        syncQueue?.sync {}
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.fileSizeKey]) else { return nil }
        var entries = 0
        var bytes = 0
        for item in items where item.lastPathComponent.hasPrefix(filePrefix)
            && item.pathExtension == pathExtension {
            entries += 1
            bytes += (try? item.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        }
        return (entries, bytes)
    }

    public static func makeRoom(
        in dir: URL,
        for bytes: Int,
        limit: Int,
        filePrefix: String = "fn_entry_",
        pathExtension: String = "qwcache"
    ) {
        guard limit > 0 else { return }
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey]) else { return }
        let files = items.filter { $0.lastPathComponent.hasPrefix(filePrefix)
            && $0.pathExtension == pathExtension }
        var total = files.reduce(0) { $0 + ((try? $1.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) }
        guard total + bytes > limit else { return }
        let oldest = files.sorted { a, b in
            let dateA = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let dateB = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return dateA < dateB
        }
        for file in oldest {
            let size = (try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            try? fm.removeItem(at: file)
            total -= size
            if total + bytes <= limit { break }
        }
    }
}

/// Unified Tier-1 slot-pinned warm RAM LRU store.
/// Protects the latest entry of each active request slot (`0..<maxSlots`) and shared base checkpoints
/// while evicting superseded intermediates and older unpinned entries in LRU order.
public final class QWSlotPinnedLRUStore<Key: Hashable & Sendable, Value: Sendable>: @unchecked Sendable {
    public struct StoredEntry: Sendable {
        public let value: Value
        public let charge: Int
        public let tokenCount: Int
        public let allowsSuffix: Bool
        public let rootHash: UInt64?
    }

    public static var defaultMaxSlots: Int { 8 }
    public static var defaultRootPrefixTokenCount: Int { 128 }

    public private(set) var maxSlots: Int
    public private(set) var byteLimit: Int
    public private(set) var entryLimit: Int
    public let rootPrefixTokenCount: Int

    public private(set) var entries: [Key: StoredEntry] = [:]
    public private(set) var order: [Key] = []
    public private(set) var chargedBytes: Int = 0
    public private(set) var evictions: Int = 0

    public private(set) var keySlot: [Key: Int] = [:]
    public private(set) var slotLastKey: [Int: Key] = [:]
    public private(set) var slotBaseKey: [Int: Key] = [:]
    public private(set) var slotLastTokens: [Int: [Int64]] = [:]
    public private(set) var slotRootHash: [Int: UInt64] = [:]
    public private(set) var slotLastTouch: [Int: UInt64] = [:]
    public private(set) var sharedBaseKeys: Set<Key> = []
    private var touchCounter: UInt64 = 0

    public var onEvict: ((Key, StoredEntry) -> Void)?

    public init(byteLimit: Int, entryLimit: Int = Int.max,
                maxSlots: Int = 8, rootPrefixTokenCount: Int = 128) {
        self.byteLimit = max(0, byteLimit)
        self.entryLimit = max(1, entryLimit)
        self.maxSlots = max(1, min(16, maxSlots))
        self.rootPrefixTokenCount = max(1, rootPrefixTokenCount)
    }

    public func setMaxSlots(_ count: Int) {
        self.maxSlots = max(1, min(16, count))
    }

    public func ensureByteLimit(atLeast minimumBytes: Int) {
        if minimumBytes > byteLimit {
            byteLimit = minimumBytes
        }
    }

    public func removeSharedBaseEntries() {
        for key in Array(sharedBaseKeys) {
            forget(key)
        }
    }

    public var isEnabled: Bool { byteLimit > 0 }
    public var count: Int { entries.count }
    public var state: (entries: Int, bytes: Int) { (entries.count, chargedBytes) }

    public func canHold(bytes: Int) -> Bool { isEnabled && bytes <= byteLimit }
    public func contains(_ key: Key) -> Bool { entries[key] != nil }

    public func isProtectedLastKey(_ key: Key, excludingSlot: Int? = nil) -> Bool {
        for (slot, lastKey) in slotLastKey where slot != excludingSlot {
            if lastKey == key && entries[lastKey] != nil {
                return true
            }
        }
        return false
    }

    public func isSharedBaseKey(_ key: Key) -> Bool {
        sharedBaseKeys.contains(key) && entries[key] != nil
    }

    public var hasSharedBase: Bool {
        sharedBaseKeys.contains { entries[$0] != nil }
    }

    public var sharedBaseBytes: Int {
        sharedBaseKeys.reduce(0) { $0 + (entries[$1]?.charge ?? 0) }
    }

    public var sharedBaseMaxTokens: Int {
        sharedBaseKeys.reduce(0) { max($0, entries[$1]?.tokenCount ?? 0) }
    }

    /// Returns the longest common prefix length (`>= rootPrefixTokenCount` and `< history.count`)
    /// between `history` and any active slot's `slotLastTokens` when `history` branches off from that slot.
    public func sharedBranchPrefixLength(for history: [Int64]) -> Int? {
        guard isEnabled, history.count > rootPrefixTokenCount else { return nil }
        var best = 0
        for (_, lastToks) in slotLastTokens {
            let limit = min(history.count, lastToks.count)
            var common = 0
            while common < limit && history[common] == lastToks[common] {
                common += 1
            }
            if common >= rootPrefixTokenCount && common < history.count && common < lastToks.count {
                best = max(best, common)
            }
        }
        return best >= rootPrefixTokenCount ? best : nil
    }

    /// Resolves the request slot (`0..<maxSlots`) for an incoming prompt history.
    public func resolveSlot(
        rootHash: UInt64,
        history: [Int64],
        isContinuationOf: (Key, Int) -> Bool
    ) -> Int {
        var activeContinuations: [(slot: Int, key: Key, count: Int)] = []
        for s in 0..<maxSlots {
            if let last = slotLastKey[s], let entry = entries[last],
               entry.tokenCount > 0, entry.tokenCount <= history.count {
                activeContinuations.append((s, last, entry.tokenCount))
            }
            if let base = slotBaseKey[s], base != slotLastKey[s], let entry = entries[base],
               entry.tokenCount > 0, entry.tokenCount <= history.count {
                activeContinuations.append((s, base, entry.tokenCount))
            }
        }
        activeContinuations.sort { $0.count > $1.count }
        for cand in activeContinuations where isContinuationOf(cand.key, cand.count) {
            return cand.slot
        }

        if history.count >= rootPrefixTokenCount {
            let rollbackSlots = (0..<maxSlots).filter { s in
                guard let last = slotLastKey[s], entries[last] != nil,
                      let lastToks = slotLastTokens[s],
                      history.count < lastToks.count,
                      lastToks.starts(with: history) else { return false }
                return true
            }
            if rollbackSlots.count == 1 {
                return rollbackSlots[0]
            }
        }

        for s in 0..<maxSlots {
            if let last = slotLastKey[s], entries[last] != nil { continue }
            return s
        }

        let sameRootSlots = (0..<maxSlots).filter { slotRootHash[$0] == rootHash }
        if let victim = sameRootSlots.min(by: { (slotLastTouch[$0] ?? 0) < (slotLastTouch[$1] ?? 0) }) {
            return victim
        }

        let shortSlots = (0..<maxSlots).filter { s in
            guard let last = slotLastKey[s], let entry = entries[last] else { return true }
            return entry.tokenCount < 1024
        }
        if let victim = shortSlots.min(by: { (slotLastTouch[$0] ?? 0) < (slotLastTouch[$1] ?? 0) }) {
            return victim
        }
        return (0..<maxSlots).min(by: { (slotLastTouch[$0] ?? 0) < (slotLastTouch[$1] ?? 0) }) ?? 0
    }

    /// Pre-evicts superseded entries of `slot` while retaining the slot's longest valid prefix checkpoint when room allows.
    public func prepareSlotForStore(
        slot: Int,
        history: [Int64],
        incomingBytes: Int,
        isPrefixEntry: (Key, Int) -> Bool
    ) {
        guard isEnabled else { return }
        let clampedSlot = max(0, min(maxSlots - 1, slot))
        slotLastTokens[clampedSlot] = history
        var keepBaseKey: Key? = nil
        var keepBaseTokens = 0
        if let baseKey = slotBaseKey[clampedSlot],
           let baseEntry = entries[baseKey],
           baseEntry.tokenCount > 0, baseEntry.tokenCount < history.count,
           isPrefixEntry(baseKey, baseEntry.tokenCount) {
            keepBaseKey = baseKey
            keepBaseTokens = baseEntry.tokenCount
        }
        for key in order where keySlot[key] == clampedSlot && !isSharedBaseKey(key) {
            guard let entry = entries[key],
                  entry.tokenCount > keepBaseTokens,
                  entry.tokenCount < history.count,
                  isPrefixEntry(key, entry.tokenCount) else { continue }
            keepBaseKey = key
            keepBaseTokens = entry.tokenCount
        }
        if let keepBaseKey {
            slotBaseKey[clampedSlot] = keepBaseKey
        }

        for key in Array(order) {
            if key == keepBaseKey || isSharedBaseKey(key) { continue }
            if keySlot[key] == clampedSlot {
                forget(key)
                continue
            }
            guard !isProtectedLastKey(key, excludingSlot: clampedSlot),
                  keySlot[key] == nil,
                  let entry = entries[key],
                  entry.tokenCount > 0, entry.tokenCount < history.count,
                  isPrefixEntry(key, entry.tokenCount) else { continue }
            forget(key)
        }

        if let baseKey = keepBaseKey, !isSharedBaseKey(baseKey), chargedBytes + incomingBytes > byteLimit {
            forget(baseKey)
        }
        makeRoom(for: incomingBytes, activeSlot: clampedSlot)
    }

    /// Two-pass eviction: first evicts non-protected non-shared entries, then short (<1024) or unprotected non-shared entries.
    /// Pinned shared-base entries (`isSharedBaseKey`) are never evicted by normal slot insertions.
    public func makeRoom(for incomingBytes: Int, activeSlot: Int? = nil) {
        guard isEnabled, incomingBytes <= byteLimit else { return }
        while entries.count >= entryLimit || chargedBytes + incomingBytes > byteLimit {
            guard let victim = order.first(where: {
                !isProtectedLastKey($0, excludingSlot: activeSlot) && !isSharedBaseKey($0)
            }) else {
                break
            }
            evictKey(victim)
        }
        while entries.count >= entryLimit || chargedBytes + incomingBytes > byteLimit {
            let shortVictim = order.first { key in
                !isSharedBaseKey(key) && (keySlot[key] != activeSlot && (entries[key]?.tokenCount ?? 0) < 1024)
            }
            guard let victim = shortVictim ?? order.first(where: { !isSharedBaseKey($0) }) else { break }
            evictKey(victim)
        }
    }

    @discardableResult
    public func evictOldest(excludingSlot: Int? = nil) -> Bool {
        guard let victim = order.first(where: {
            !isProtectedLastKey($0, excludingSlot: excludingSlot) && !isSharedBaseKey($0)
        }) ?? order.first(where: { !isSharedBaseKey($0) }) else { return false }
        evictKey(victim)
        return true
    }

    public func evictMatchingPrefixes(
        maxTokenCount: Int,
        matchesPrefix: (Key, Int) -> Bool
    ) {
        guard isEnabled else { return }
        for key in Array(order) {
            if isSharedBaseKey(key) || isProtectedLastKey(key) { continue }
            guard let entry = entries[key], entry.tokenCount > 0, entry.tokenCount < maxTokenCount else { continue }
            if matchesPrefix(key, entry.tokenCount) {
                evictKey(key)
            }
        }
    }

    @discardableResult
    public func store(
        key: Key,
        value: Value,
        charge: Int,
        tokenCount: Int,
        allowsSuffix: Bool = true,
        slot: Int? = nil,
        rootHash: UInt64? = nil,
        history: [Int64]? = nil,
        isSharedBase: Bool = false
    ) -> Bool {
        guard isEnabled, charge <= byteLimit else { return false }
        forget(key)
        makeRoom(for: charge, activeSlot: slot)
        guard entries.count < entryLimit, chargedBytes + charge <= byteLimit else { return false }
        entries[key] = StoredEntry(value: value, charge: charge, tokenCount: tokenCount, allowsSuffix: allowsSuffix, rootHash: rootHash)
        order.append(key)
        chargedBytes += charge
        if isSharedBase {
            sharedBaseKeys.insert(key)
            return true
        }
        if let slot {
            let clampedSlot = max(0, min(maxSlots - 1, slot))
            keySlot[key] = clampedSlot
            slotLastKey[clampedSlot] = key
            if let history {
                slotLastTokens[clampedSlot] = history
            }
            if let base = slotBaseKey[clampedSlot], let baseEntry = entries[base] {
                if tokenCount < baseEntry.tokenCount {
                    slotBaseKey[clampedSlot] = key
                }
            } else {
                slotBaseKey[clampedSlot] = key
            }
            if let rootHash {
                slotRootHash[clampedSlot] = rootHash
            }
            touchCounter &+= 1
            slotLastTouch[clampedSlot] = touchCounter
        }
        return true
    }

    public func value(for key: Key, touchOnHit: Bool = true) -> StoredEntry? {
        guard let entry = entries[key] else { return nil }
        if touchOnHit {
            touch(key)
        }
        return entry
    }

    public func touch(_ key: Key) {
        guard entries[key] != nil else { return }
        order.removeAll { $0 == key }
        order.append(key)
        if let s = keySlot[key] {
            touchCounter &+= 1
            slotLastTouch[s] = touchCounter
        }
    }

    public func touchSlot(for key: Key) {
        if let s = keySlot[key] {
            touchCounter &+= 1
            slotLastTouch[s] = touchCounter
        }
    }

    public func forget(_ key: Key) {
        if let entry = entries.removeValue(forKey: key) {
            chargedBytes -= entry.charge
        }
        order.removeAll { $0 == key }
        sharedBaseKeys.remove(key)
        if let s = keySlot.removeValue(forKey: key) {
            if slotLastKey[s] == key { slotLastKey.removeValue(forKey: s) }
            if slotBaseKey[s] == key { slotBaseKey.removeValue(forKey: s) }
        }
    }

    private func evictKey(_ key: Key) {
        guard let entry = entries[key] else { return }
        forget(key)
        evictions += 1
        onEvict?(key, entry)
    }
}
