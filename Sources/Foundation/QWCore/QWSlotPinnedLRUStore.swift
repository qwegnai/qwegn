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

    private let lock = NSRecursiveLock()
    private var _maxSlots: Int
    private var _byteLimit: Int
    private var _entryLimit: Int
    public let rootPrefixTokenCount: Int

    private var _entries: [Key: StoredEntry] = [:]
    private var _order: [Key] = []
    private var _chargedBytes: Int = 0
    private var _evictions: Int = 0

    private var _keySlot: [Key: Int] = [:]
    private var _slotLastKey: [Int: Key] = [:]
    private var _slotBaseKey: [Int: Key] = [:]
    private var _slotLastTokens: [Int: [Int64]] = [:]
    private var _slotRootHash: [Int: UInt64] = [:]
    private var _slotLastTouch: [Int: UInt64] = [:]
    private var _sharedBaseKeys: Set<Key> = []
    private var _sharedBranchKeys: Set<Key> = []
    private var touchCounter: UInt64 = 0
    private var _onEvict: ((Key, StoredEntry) -> Void)?
    private var _onRemove: ((Key) -> Void)?

    public var maxSlots: Int { lock.withLock { _maxSlots } }
    public var byteLimit: Int { lock.withLock { _byteLimit } }
    public var entryLimit: Int { lock.withLock { _entryLimit } }
    public var entries: [Key: StoredEntry] { lock.withLock { _entries } }
    public var order: [Key] { lock.withLock { _order } }
    public var chargedBytes: Int { lock.withLock { _chargedBytes } }
    public var evictions: Int { lock.withLock { _evictions } }
    public var sharedBaseKeys: Set<Key> { lock.withLock { _sharedBaseKeys } }
    public var onEvict: ((Key, StoredEntry) -> Void)? {
        get { lock.withLock { _onEvict } }
        set { lock.withLock { _onEvict = newValue } }
    }
    public var onRemove: ((Key) -> Void)? {
        get { lock.withLock { _onRemove } }
        set { lock.withLock { _onRemove = newValue } }
    }

    public init(byteLimit: Int, entryLimit: Int = Int.max,
                maxSlots: Int = 8, rootPrefixTokenCount: Int = 128) {
        self._byteLimit = max(0, byteLimit)
        self._entryLimit = max(1, entryLimit)
        self._maxSlots = max(1, min(16, maxSlots))
        self.rootPrefixTokenCount = max(1, rootPrefixTokenCount)
    }

    public func setMaxSlots(_ count: Int) {
        lock.withLock { _maxSlots = max(1, min(16, count)) }
    }

    public func ensureByteLimit(atLeast minimumBytes: Int) {
        lock.withLock {
            if minimumBytes > _byteLimit { _byteLimit = minimumBytes }
        }
    }

    /// Updates byte/entry budgets, purging everything on zero byte limit or evicting excess entries.
    public func reconfigure(byteLimit: Int, entryLimit: Int? = nil) {
        let pending = lock.withLock { () -> ([(Key, StoredEntry)], [Key], ((Key, StoredEntry) -> Void)?, ((Key) -> Void)?) in
            _byteLimit = max(0, byteLimit)
            if let entryLimit {
                _entryLimit = max(1, entryLimit)
            }
            var evicted: [(Key, StoredEntry)] = []
            var removed: [Key] = []
            if _byteLimit == 0 {
                for key in Array(_order) {
                    if let item = evictKeyLocked(key, removed: &removed) { evicted.append(item) }
                }
                clearMetadataLocked()
                return (evicted, removed, _onEvict, _onRemove)
            }
            while _entries.count > _entryLimit || _chargedBytes > _byteLimit {
                guard let victim = _order.first(where: {
                    !isProtectedLastKeyLocked($0) && !isSharedBaseKeyLocked($0)
                }) else { break }
                if let item = evictKeyLocked(victim, removed: &removed) { evicted.append(item) }
            }
            while _entries.count > _entryLimit || _chargedBytes > _byteLimit {
                guard let victim = _order.first(where: { !isSharedBaseKeyLocked($0) }) else { break }
                if let item = evictKeyLocked(victim, removed: &removed) { evicted.append(item) }
            }
            while _entries.count > _entryLimit || _chargedBytes > _byteLimit {
                guard let victim = _order.first else { break }
                if let item = evictKeyLocked(victim, removed: &removed) { evicted.append(item) }
            }
            return (evicted, removed, _onEvict, _onRemove)
        }
        notifyRemovals(pending.1, callback: pending.3)
        notifyEvictions(pending.0, callback: pending.2)
    }

    public func removeAll() {
        let pending = lock.withLock { () -> ([Key], ((Key) -> Void)?) in
            let removedKeys = _onRemove != nil ? Array(_entries.keys) : []
            clearMetadataLocked()
            return (removedKeys, _onRemove)
        }
        notifyRemovals(pending.0, callback: pending.1)
    }

    private func clearMetadataLocked() {
        _entries.removeAll()
        _order.removeAll()
        _chargedBytes = 0
        _keySlot.removeAll()
        _slotLastKey.removeAll()
        _slotBaseKey.removeAll()
        _slotLastTokens.removeAll()
        _slotRootHash.removeAll()
        _slotLastTouch.removeAll()
        _sharedBaseKeys.removeAll()
        _sharedBranchKeys.removeAll()
    }

    public func removeSharedBaseEntries() {
        let pending = lock.withLock { () -> ([Key], ((Key) -> Void)?) in
            var removed: [Key] = []
            for key in Array(_sharedBaseKeys) {
                if let k = forgetLocked(key) { removed.append(k) }
            }
            return (removed, _onRemove)
        }
        notifyRemovals(pending.0, callback: pending.1)
    }

    public var isEnabled: Bool { lock.withLock { _byteLimit > 0 } }
    public var count: Int { lock.withLock { _entries.count } }
    public var state: (entries: Int, bytes: Int) { lock.withLock { (_entries.count, _chargedBytes) } }

    public func canHold(bytes: Int) -> Bool { lock.withLock { _byteLimit > 0 && bytes <= _byteLimit } }
    public func canHoldIncludingShared(bytes: Int) -> Bool {
        lock.withLock {
            let shared = _sharedBaseKeys.reduce(0) { $0 + (_entries[$1]?.charge ?? 0) }
            return _byteLimit > 0 && (shared + bytes <= _byteLimit)
        }
    }
    public func contains(_ key: Key) -> Bool { lock.withLock { _entries[key] != nil } }

    public func snapshotOrderedEntries() -> (order: [Key], entries: [Key: StoredEntry], sharedBaseKeys: Set<Key>) {
        lock.withLock { (_order, _entries, _sharedBaseKeys) }
    }

    public func isProtectedLastKey(_ key: Key, excludingSlot: Int? = nil) -> Bool {
        lock.withLock { isProtectedLastKeyLocked(key, excludingSlot: excludingSlot) }
    }

    private func isProtectedLastKeyLocked(_ key: Key, excludingSlot: Int? = nil) -> Bool {
        for (slot, lastKey) in _slotLastKey where slot != excludingSlot {
            if lastKey == key && _entries[lastKey] != nil { return true }
        }
        return false
    }

    public func isSharedBaseKey(_ key: Key) -> Bool { lock.withLock { isSharedBaseKeyLocked(key) } }
    private func isSharedBaseKeyLocked(_ key: Key) -> Bool { _sharedBaseKeys.contains(key) && _entries[key] != nil }

    public var hasSharedBase: Bool { lock.withLock { _sharedBaseKeys.contains { _entries[$0] != nil } } }

    public var sharedBaseBytes: Int {
        lock.withLock { _sharedBaseKeys.reduce(0) { $0 + (_entries[$1]?.charge ?? 0) } }
    }

    /// Returns the longest common prefix length (`>= rootPrefixTokenCount` and `< history.count`)
    /// between `history` and any active slot's `slotLastTokens` when `history` branches off from that slot.
    public func sharedBranchPrefixLength(for history: [Int64]) -> Int? {
        lock.withLock {
            guard _byteLimit > 0, history.count > rootPrefixTokenCount else { return nil }
            var best = 0
            for (_, lastToks) in _slotLastTokens {
                let limit = min(history.count, lastToks.count)
                var common = 0
                while common < limit && history[common] == lastToks[common] { common += 1 }
                if common >= rootPrefixTokenCount && common < history.count && common < lastToks.count {
                    best = max(best, common)
                }
            }
            return best >= rootPrefixTokenCount ? best : nil
        }
    }

    /// Resolves the request slot (`0..<maxSlots`) for an incoming prompt history.
    public func resolveSlot(
        rootHash: UInt64,
        history: [Int64],
        isContinuationOf: (Key, Int) -> Bool
    ) -> Int {
        lock.withLock {
            var activeContinuations: [(slot: Int, key: Key, count: Int)] = []
            for s in 0..<_maxSlots {
                if let last = _slotLastKey[s], let entry = _entries[last],
                   entry.tokenCount > 0, entry.tokenCount <= history.count {
                    activeContinuations.append((s, last, entry.tokenCount))
                }
                if let base = _slotBaseKey[s], base != _slotLastKey[s], let entry = _entries[base],
                   entry.tokenCount > 0, entry.tokenCount <= history.count {
                    activeContinuations.append((s, base, entry.tokenCount))
                }
            }
            activeContinuations.sort { $0.count > $1.count }
            for cand in activeContinuations where isContinuationOf(cand.key, cand.count) { return cand.slot }

            if history.count >= rootPrefixTokenCount {
                let rollbackSlots = (0..<_maxSlots).filter { s in
                    guard let last = _slotLastKey[s], _entries[last] != nil,
                          let lastToks = _slotLastTokens[s],
                          history.count < lastToks.count,
                          lastToks.starts(with: history) else { return false }
                    return true
                }
                if rollbackSlots.count == 1 { return rollbackSlots[0] }
            }

            for s in 0..<_maxSlots {
                if let last = _slotLastKey[s], _entries[last] != nil { continue }
                return s
            }

            let sameRootSlots = (0..<_maxSlots).filter { _slotRootHash[$0] == rootHash }
            if let victim = sameRootSlots.min(by: { (_slotLastTouch[$0] ?? 0) < (_slotLastTouch[$1] ?? 0) }) {
                return victim
            }

            let shortSlots = (0..<_maxSlots).filter { s in
                guard let last = _slotLastKey[s], let entry = _entries[last] else { return true }
                return entry.tokenCount < 1024
            }
            if let victim = shortSlots.min(by: { (_slotLastTouch[$0] ?? 0) < (_slotLastTouch[$1] ?? 0) }) {
                return victim
            }
            return (0..<_maxSlots).min(by: { (_slotLastTouch[$0] ?? 0) < (_slotLastTouch[$1] ?? 0) }) ?? 0
        }
    }

    /// Pre-evicts superseded entries of `slot` while retaining the slot's longest valid prefix checkpoint when room allows.
    public func prepareSlotForStore(
        slot: Int,
        history: [Int64],
        incomingBytes: Int,
        isPrefixEntry: (Key, Int) -> Bool
    ) {
        let pending = lock.withLock { () -> ([(Key, StoredEntry)], [Key], ((Key, StoredEntry) -> Void)?, ((Key) -> Void)?) in
            guard _byteLimit > 0 else { return ([], [], nil, nil) }
            let clampedSlot = max(0, min(_maxSlots - 1, slot))
            _slotLastTokens[clampedSlot] = history
            var keepBaseKey: Key? = nil
            var keepBaseTokens = 0
            if let baseKey = _slotBaseKey[clampedSlot],
               let baseEntry = _entries[baseKey],
               baseEntry.tokenCount > 0, baseEntry.tokenCount < history.count,
               isPrefixEntry(baseKey, baseEntry.tokenCount) {
                keepBaseKey = baseKey
                keepBaseTokens = baseEntry.tokenCount
            }
            for key in _order where _keySlot[key] == clampedSlot && !isSharedBaseKeyLocked(key) {
                guard let entry = _entries[key],
                      entry.tokenCount > keepBaseTokens,
                      entry.tokenCount < history.count,
                      isPrefixEntry(key, entry.tokenCount) else { continue }
                keepBaseKey = key
                keepBaseTokens = entry.tokenCount
            }
            if let keepBaseKey { _slotBaseKey[clampedSlot] = keepBaseKey }

            var removed: [Key] = []
            for key in Array(_order) {
                if key == keepBaseKey || isSharedBaseKeyLocked(key) { continue }
                if _keySlot[key] == clampedSlot {
                    if let k = forgetLocked(key) { removed.append(k) }
                    continue
                }
                guard !isProtectedLastKeyLocked(key, excludingSlot: clampedSlot),
                      !_sharedBranchKeys.contains(key),
                      _keySlot[key] == nil,
                      let entry = _entries[key],
                      entry.tokenCount > 0, entry.tokenCount < history.count,
                      isPrefixEntry(key, entry.tokenCount) else { continue }
                if let k = forgetLocked(key) { removed.append(k) }
            }

            if let baseKey = keepBaseKey, !isSharedBaseKeyLocked(baseKey), _chargedBytes + incomingBytes > _byteLimit {
                if let k = forgetLocked(baseKey) { removed.append(k) }
            }
            let evicted = makeRoomLocked(for: incomingBytes, activeSlot: clampedSlot, removed: &removed)
            return (evicted, removed, _onEvict, _onRemove)
        }
        notifyRemovals(pending.1, callback: pending.3)
        notifyEvictions(pending.0, callback: pending.2)
    }

    /// Two-pass eviction: first evicts non-protected non-shared entries, then short (<1024) or unprotected non-shared entries.
    /// Pinned shared-base entries (`isSharedBaseKey`) are never evicted by normal slot insertions.
    public func makeRoom(for incomingBytes: Int, activeSlot: Int? = nil) {
        let pending = lock.withLock { () -> ([(Key, StoredEntry)], [Key], ((Key, StoredEntry) -> Void)?, ((Key) -> Void)?) in
            var removed: [Key] = []
            let evicted = makeRoomLocked(for: incomingBytes, activeSlot: activeSlot, removed: &removed)
            return (evicted, removed, _onEvict, _onRemove)
        }
        notifyRemovals(pending.1, callback: pending.3)
        notifyEvictions(pending.0, callback: pending.2)
    }

    private func makeRoomLocked(for incomingBytes: Int, activeSlot: Int?, removed: inout [Key]) -> [(Key, StoredEntry)] {
        guard _byteLimit > 0, incomingBytes <= _byteLimit else { return [] }
        var evicted: [(Key, StoredEntry)] = []
        while _entries.count >= _entryLimit || _chargedBytes + incomingBytes > _byteLimit {
            guard let victim = _order.first(where: {
                !isProtectedLastKeyLocked($0, excludingSlot: activeSlot) && !isSharedBaseKeyLocked($0)
            }) else { break }
            if let item = evictKeyLocked(victim, removed: &removed) { evicted.append(item) }
        }
        while _entries.count >= _entryLimit || _chargedBytes + incomingBytes > _byteLimit {
            let shortVictim = _order.first { key in
                !isSharedBaseKeyLocked(key) && (_keySlot[key] != activeSlot && (_entries[key]?.tokenCount ?? 0) < 1024)
            }
            guard let victim = shortVictim ?? _order.first(where: { !isSharedBaseKeyLocked($0) }) else { break }
            if let item = evictKeyLocked(victim, removed: &removed) { evicted.append(item) }
        }
        return evicted
    }

    @discardableResult
    public func evictOldest(excludingSlot: Int? = nil) -> Bool {
        let pending = lock.withLock { () -> ((Key, StoredEntry)?, [Key], ((Key, StoredEntry) -> Void)?, ((Key) -> Void)?) in
            guard let victim = _order.first(where: {
                !isProtectedLastKeyLocked($0, excludingSlot: excludingSlot) && !isSharedBaseKeyLocked($0)
            }) ?? _order.first(where: { !isSharedBaseKeyLocked($0) }) else { return (nil, [], nil, nil) }
            var removed: [Key] = []
            return (evictKeyLocked(victim, removed: &removed), removed, _onEvict, _onRemove)
        }
        notifyRemovals(pending.1, callback: pending.3)
        guard let item = pending.0 else { return false }
        pending.2?(item.0, item.1)
        return true
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
        isSharedBase: Bool = false,
        isBranchCheckpoint: Bool = false
    ) -> Bool {
        let pending = lock.withLock { () -> (Bool, [(Key, StoredEntry)], [Key], ((Key, StoredEntry) -> Void)?, ((Key) -> Void)?) in
            guard _byteLimit > 0, charge <= _byteLimit else { return (false, [], [], nil, nil) }
            var removed: [Key] = []
            if let k = forgetLocked(key) { removed.append(k) }
            let evicted = makeRoomLocked(for: charge, activeSlot: slot, removed: &removed)
            guard _entries.count < _entryLimit, _chargedBytes + charge <= _byteLimit else {
                return (false, evicted, removed, _onEvict, _onRemove)
            }
            _entries[key] = StoredEntry(value: value, charge: charge, tokenCount: tokenCount, allowsSuffix: allowsSuffix, rootHash: rootHash)
            _order.append(key)
            _chargedBytes += charge
            if isSharedBase {
                _sharedBaseKeys.insert(key)
                return (true, evicted, removed, _onEvict, _onRemove)
            }
            if isBranchCheckpoint {
                _sharedBranchKeys.insert(key)
            }
            if let slot {
                let clampedSlot = max(0, min(_maxSlots - 1, slot))
                _keySlot[key] = clampedSlot
                _slotLastKey[clampedSlot] = key
                if let history { _slotLastTokens[clampedSlot] = history }
                if let base = _slotBaseKey[clampedSlot], let baseEntry = _entries[base] {
                    if tokenCount < baseEntry.tokenCount { _slotBaseKey[clampedSlot] = key }
                } else {
                    _slotBaseKey[clampedSlot] = key
                }
                if let rootHash { _slotRootHash[clampedSlot] = rootHash }
                touchCounter &+= 1
                _slotLastTouch[clampedSlot] = touchCounter
            }
            return (true, evicted, removed, _onEvict, _onRemove)
        }
        notifyRemovals(pending.2, callback: pending.4)
        notifyEvictions(pending.1, callback: pending.3)
        return pending.0
    }

    public func value(for key: Key, touchOnHit: Bool = true) -> StoredEntry? {
        lock.withLock {
            guard let entry = _entries[key] else { return nil }
            if touchOnHit { touchLocked(key) }
            return entry
        }
    }

    private func touchLocked(_ key: Key) {
        guard _entries[key] != nil else { return }
        _order.removeAll { $0 == key }
        _order.append(key)
        if let s = _keySlot[key] {
            touchCounter &+= 1
            _slotLastTouch[s] = touchCounter
        }
    }

    public func forget(_ key: Key) {
        let pending = lock.withLock { () -> (Key?, ((Key) -> Void)?) in
            (forgetLocked(key), _onRemove)
        }
        if let removedKey = pending.0 {
            pending.1?(removedKey)
        }
    }

    private func forgetLocked(_ key: Key) -> Key? {
        let removed = _entries.removeValue(forKey: key)
        if let entry = removed { _chargedBytes -= entry.charge }
        _order.removeAll { $0 == key }
        _sharedBaseKeys.remove(key)
        _sharedBranchKeys.remove(key)
        if let s = _keySlot.removeValue(forKey: key) {
            if _slotLastKey[s] == key { _slotLastKey.removeValue(forKey: s) }
            if _slotBaseKey[s] == key { _slotBaseKey.removeValue(forKey: s) }
        }
        return removed != nil ? key : nil
    }

    private func evictKeyLocked(_ key: Key, removed: inout [Key]) -> (Key, StoredEntry)? {
        guard let entry = _entries[key] else { return nil }
        let slot = _keySlot[key]
        if let k = forgetLocked(key) { removed.append(k) }
        if let slot, _slotLastKey[slot] == nil, _slotBaseKey[slot] == nil, !_keySlot.values.contains(slot) {
            _slotLastTokens.removeValue(forKey: slot)
            _slotRootHash.removeValue(forKey: slot)
            _slotLastTouch.removeValue(forKey: slot)
        }
        _evictions += 1
        return (key, entry)
    }

    private func notifyRemovals(_ keys: [Key], callback: ((Key) -> Void)?) {
        guard let callback, !keys.isEmpty else { return }
        for key in keys { callback(key) }
    }

    private func notifyEvictions(_ evicted: [(Key, StoredEntry)], callback: ((Key, StoredEntry) -> Void)?) {
        guard let callback, !evicted.isEmpty else { return }
        for (key, entry) in evicted { callback(key, entry) }
    }
}
