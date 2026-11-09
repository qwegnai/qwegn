// An L1 hot-row cache in front of the PLE n-gram table's row source.
//
// The n-gram ids are EXPECTED to be steeply skewed, so the same rows are
// read over and over; caching them removes the pread() and the decode.
// That premise is not yet measured — see specs/30-flashnext-architecture.md
// §10 item 8. The cache breaks even at a ~50% hit rate.

import Foundation

/// One cached row and the clock reading of its last use.
/// A **class**, so the hit path refreshes the stamp with one field write
/// instead of writing the entry back into the dictionary.
final class FNNgramRowCacheEntry {
    let id: Int
    let row: [Float]
    var used: UInt64

    init(id: Int, row: [Float], used: UInt64) {
        self.id = id
        self.row = row
        self.used = used
    }
}

/// The cache refuses a row whose size is not the one it was built for: a
/// longer row would be under-charged against the budget.
public enum FNNgramRowCacheError: Error, CustomStringConvertible {
    case rowSizeMismatch(expected: Int, got: Int)

    public var description: String {
        switch self {
        case let .rowSizeMismatch(expected, got):
            return "ngram cache row is \(got) bytes, the cache was built for \(expected)"
        }
    }
}

/// One stripe of the cache: its own lock, its own rows, its own budget.
/// A **class**, not a struct — a struct stripe puts every stripe's mutable
/// state in one array's buffer, so two concurrent element writes conflict.
final class FNNgramRowCacheStripe: @unchecked Sendable {
    let lock = NSLock()
    var rows: [Int: FNNgramRowCacheEntry] = [:]
    var bytes = 0
    var hits = 0
    var misses = 0
    private var clock: UInt64 = 0

    /// The next clock reading. The caller holds the lock.
    func nextStamp() -> UInt64 {
        clock &+= 1
        return clock
    }

    /// The least recently used entry, if any. The caller holds the lock.
    /// The scan is over one stripe only, and a stripe holds a bounded number
    /// of rows, so the scan stays cheap.
    func leastRecentlyUsed() -> FNNgramRowCacheEntry? {
        var victim: FNNgramRowCacheEntry?
        var oldest = UInt64.max
        for entry in rows.values where entry.used < oldest {
            oldest = entry.used
            victim = entry
        }
        return victim
    }
}

/// An L1 hot-row cache in front of an `FNNgramRows` source.
/// Locking is **striped** — one `NSLock` per stripe — because `FNNgramEmbedding.embedLast` reads rows from
/// `DispatchQueue.concurrentPerform`; a single mutex would serialise them.
public final class FNNgramRowCache: @unchecked Sendable {
    /// How many rows one stripe aims to hold. The eviction victim is found by
    /// scanning a stripe, so a stripe must not grow without bound.
    static let targetRowsPerStripe = 32
    /// The most stripes the cache will ever build, whatever the budget.
    static let maxStripeCount = 1 << 16

    private let capacityBytes: Int
    private let rowBytes: Int
    private let stripes: [FNNgramRowCacheStripe]
    private let stripeBudget: Int

    /// - Parameters: `capacityBytes` is the cache's total row budget, `rowBytes` the bytes one row occupies
    ///   (`headDim * 4`), and `maxStripes` the most locks to spread the rows over. The budget is **global**:
    ///   each stripe gets `capacityBytes / stripeCount`, so the cache never exceeds it.
    public init(capacityBytes: Int, rowBytes: Int, maxStripes: Int = 4096) {
        precondition(rowBytes > 0, "the cache needs a positive row size")
        self.capacityBytes = max(0, capacityBytes)
        self.rowBytes = rowBytes
        // A stripe that cannot hold one row is useless, so there are never
        // more stripes than the budget has rows.
        let budgetRows = max(1, capacityBytes / rowBytes)
        let wanted = min(max(1, min(maxStripes, FNNgramRowCache.maxStripeCount)),
                         max(1, budgetRows / FNNgramRowCache.targetRowsPerStripe))
        // The largest power of two <= wanted, so the stripe index is a mask
        // and not a division.
        let count = 1 << (wanted.bitWidth - 1 - wanted.leadingZeroBitCount)
        self.stripes = (0..<count).map { _ in FNNgramRowCacheStripe() }
        self.stripeBudget = max(1, capacityBytes / count)
    }

    private func stripe(for id: Int) -> FNNgramRowCacheStripe {
        let h = (UInt(truncatingIfNeeded: id) &* 0x9E37_79B9) >> 32
        return stripes[Int(h & UInt(stripes.count - 1))]
    }

    /// The cached row, or nil on a miss.
    public func value(for id: Int) -> [Float]? {
        let s = stripe(for: id)
        s.lock.lock()
        defer { s.lock.unlock() }
        guard let entry = s.rows[id] else {
            s.misses += 1
            return nil
        }
        s.hits += 1
        entry.used = s.nextStamp()
        return entry.row
    }

    /// Cache a row. A row that is already cached is left alone.
    /// The row must be exactly the size the cache was built for.
    public func store(_ id: Int, _ row: [Float]) {
        precondition(row.count * MemoryLayout<Float>.size == rowBytes,
                     "the cache was built for \(rowBytes)-byte rows, got \(row.count * 4)")
        let s = stripe(for: id)
        s.lock.lock()
        defer { s.lock.unlock() }
        guard s.rows[id] == nil else { return }
        while s.bytes + rowBytes > stripeBudget, let victim = s.leastRecentlyUsed() {
            s.rows[victim.id] = nil
            s.bytes -= rowBytes
        }
        guard s.bytes + rowBytes <= stripeBudget else { return }
        s.rows[id] = FNNgramRowCacheEntry(id: id, row: row, used: s.nextStamp())
        s.bytes += rowBytes
    }

    /// Cache a row, refusing one of the wrong size instead of trapping.
    public func storeChecked(_ id: Int, _ row: [Float]) throws {
        let got = row.count * MemoryLayout<Float>.size
        guard got == rowBytes else {
            throw FNNgramRowCacheError.rowSizeMismatch(expected: rowBytes, got: got)
        }
        store(id, row)
    }

    /// Drop every cached row and reset the hit/miss counters.
    public func clear() {
        for s in stripes {
            s.lock.lock()
            s.rows.removeAll(keepingCapacity: true)
            s.bytes = 0
            s.hits = 0
            s.misses = 0
            s.lock.unlock()
        }
    }

    public var hitCount: Int {
        var total = 0
        for s in stripes {
            s.lock.lock()
            total += s.hits
            s.lock.unlock()
        }
        return total
    }

    public var missCount: Int {
        var total = 0
        for s in stripes {
            s.lock.lock()
            total += s.misses
            s.lock.unlock()
        }
        return total
    }

    /// The share of reads that were served from the cache. It is a ratio of
    /// two separate snapshots, so a read racing with the counters can move
    /// it by one — the counters only ever grow.
    public var hitRate: Double {
        let hits = hitCount
        let total = hits + missCount
        return total == 0 ? 0 : Double(hits) / Double(total)
    }

    /// The row payload the cache holds, in bytes.
    public var approximateBytes: Int {
        var total = 0
        for s in stripes {
            s.lock.lock()
            total += s.bytes
            s.lock.unlock()
        }
        return total
    }

    public var storedRowCount: Int {
        var total = 0
        for s in stripes {
            s.lock.lock()
            total += s.rows.count
            s.lock.unlock()
        }
        return total
    }

    /// The cache's total budget, in bytes.
    public var budgetBytes: Int { capacityBytes }

    /// Whether the cache can hold anything at all. A model without a PLE
    /// layer gets a cache with no budget, and a miss must not then pay for a
    /// row copy that will be thrown away.
    public var isEnabled: Bool { capacityBytes > 0 }

    /// How many locks the rows are spread over. Derived from the budget, so
    /// that no single stripe grows large enough to make an eviction scan
    /// expensive.
    public var stripeCount: Int { stripes.count }
}

/// An `FNNgramRows` that serves from a cache and falls through to the source.
public final class FNNgramCachedRows: FNNgramRows, @unchecked Sendable {
    public let headDim: Int
    public let cache: FNNgramRowCache
    private let backing: any FNNgramRows

    public init(backing: any FNNgramRows, capacityBytes: Int) {
        self.backing = backing
        self.headDim = backing.headDim
        self.cache = FNNgramRowCache(capacityBytes: capacityBytes,
                                     rowBytes: backing.headDim * MemoryLayout<Float>.size)
    }

    public func row(_ id: Int) -> [Float] {
        if let hit = cache.value(for: id) { return hit }
        let fresh = backing.row(id)
        cache.store(id, fresh)
        return fresh
    }

    public func copyRow(_ id: Int, to dst: UnsafeMutablePointer<Float>) {
        if let hit = cache.value(for: id) {
            hit.withUnsafeBufferPointer { src in
                dst.update(from: src.baseAddress!, count: headDim)
            }
            return
        }
        backing.copyRow(id, to: dst)
        // A cache with no budget stores nothing, so do not pay for a copy
        // that would be thrown away.
        guard cache.isEnabled else { return }
        let n = headDim
        let fresh = [Float](unsafeUninitializedCapacity: n) { out, count in
            out.baseAddress!.update(from: dst, count: n)
            count = n
        }
        do {
            try cache.storeChecked(id, fresh)
        } catch {
            // A row that is not the cache's row size is a broken contract.
            // The read already returned the right data, so the row is simply
            // not cached rather than the process going down over it.
        }
    }
}
