import XCTest
import Foundation
@testable import QWFlashNext

final class FNNgramRowCacheTests: XCTestCase {
    /// A row source that counts reads, so a cache hit is observable.
    private final class CountingRows: FNNgramRows, @unchecked Sendable {
        let headDim: Int
        private let lock = NSLock()
        private var reads_ = 0

        init(headDim: Int = 4) { self.headDim = headDim }

        var reads: Int {
            lock.lock(); defer { lock.unlock() }
            return reads_
        }

        func row(_ id: Int) -> [Float] {
            lock.lock(); reads_ += 1; lock.unlock()
            return [Float](repeating: Float(id), count: headDim)
        }

        func copyRow(_ id: Int, to dst: UnsafeMutablePointer<Float>) {
            lock.lock(); reads_ += 1; lock.unlock()
            for i in 0..<headDim { dst[i] = Float(id) }
        }
    }

    func testAMissReadsTheBackingSource() {
        let rows = CountingRows()
        let cached = FNNgramCachedRows(backing: rows, capacityBytes: 1 << 20)
        XCTAssertEqual(cached.row(3), [3, 3, 3, 3])
        XCTAssertEqual(rows.reads, 1)
    }

    func testASecondReadIsServedFromTheCache() {
        let rows = CountingRows()
        let cached = FNNgramCachedRows(backing: rows, capacityBytes: 1 << 20)
        _ = cached.row(3)
        _ = cached.row(3)
        XCTAssertEqual(rows.reads, 1, "the second read must not touch the source")
    }

    func testDifferentIdsDoNotCollide() {
        let rows = CountingRows()
        let cached = FNNgramCachedRows(backing: rows, capacityBytes: 1 << 20)
        XCTAssertEqual(cached.row(1), [1, 1, 1, 1])
        XCTAssertEqual(cached.row(2), [2, 2, 2, 2])
        XCTAssertEqual(rows.reads, 2)
    }

    func testHitAndMissCountersAreReported() {
        let rows = CountingRows()
        let cached = FNNgramCachedRows(backing: rows, capacityBytes: 1 << 20)
        _ = cached.row(1)
        _ = cached.row(1)
        XCTAssertEqual(cached.cache.hitCount, 1)
        XCTAssertEqual(cached.cache.missCount, 1)
    }

    func testTheCacheStaysWithinItsByteBudget() {
        let rows = CountingRows(headDim: 160)
        let cached = FNNgramCachedRows(backing: rows, capacityBytes: 64 * 1024)
        for id in 0..<10_000 { _ = cached.row(id) }
        XCTAssertLessThanOrEqual(cached.cache.approximateBytes, 64 * 1024)
    }

    func testConcurrentReadsAreSafe() {
        let rows = CountingRows(headDim: 160)
        let cached = FNNgramCachedRows(backing: rows, capacityBytes: 1 << 20)
        DispatchQueue.concurrentPerform(iterations: 64) { i in
            _ = cached.row(i % 32)
        }
        XCTAssertEqual(cached.row(0), [Float](repeating: 0, count: 160))
    }

    func testTheCacheNeverExceedsItsBudgetHoweverManyStripesFill() {
        // The budget is global and split across the stripes. A stripe that
        // fills up must not be able to push the cache over the budget.
        for capacity in [1 << 10, 4 << 10, 64 << 10, 1 << 20] {
            let rows = CountingRows(headDim: 160)
            let cached = FNNgramCachedRows(backing: rows, capacityBytes: capacity)
            for id in 0..<5_000 { _ = cached.row(id) }
            XCTAssertLessThanOrEqual(cached.cache.approximateBytes, capacity,
                                     "budget \(capacity) was exceeded")
        }
    }

    func testTheLeastRecentlyUsedRowIsTheOneEvicted() {
        let cache = FNNgramRowCache(capacityBytes: 2 * 4 * 4, rowBytes: 4 * 4, maxStripes: 1)
        cache.store(1, [1, 1, 1, 1])
        cache.store(2, [2, 2, 2, 2])
        _ = cache.value(for: 1)          // 1 is now the most recent
        cache.store(3, [3, 3, 3, 3])     // so 2 is the victim
        XCTAssertNotNil(cache.value(for: 1))
        XCTAssertNil(cache.value(for: 2))
        XCTAssertNotNil(cache.value(for: 3))
    }

    func testChurnEvictsTheOldestAndKeepsTheNewest() {
        // The stripe holds two rows. Five distinct ids cycle through it, so
        // only the two most recent may survive.
        let cache = FNNgramRowCache(capacityBytes: 2 * 16, rowBytes: 16, maxStripes: 1)
        for id in 0..<5 {
            cache.store(id, [Float](repeating: Float(id), count: 4))
        }
        XCTAssertEqual(cache.storedRowCount, 2)
        XCTAssertEqual(cache.approximateBytes, 32)
        XCTAssertNil(cache.value(for: 0))
        XCTAssertNil(cache.value(for: 1))
        XCTAssertNil(cache.value(for: 2))
        XCTAssertNotNil(cache.value(for: 3))
        XCTAssertNotNil(cache.value(for: 4))
    }

    func testARepeatedIdSurvivesTheChurnAroundIt() {
        // The row that keeps being read must never be the victim, however
        // much else comes through.
        let cache = FNNgramRowCache(capacityBytes: 2 * 16, rowBytes: 16, maxStripes: 1)
        cache.store(7, [7, 7, 7, 7])
        for id in 0..<50 {
            cache.store(id, [Float](repeating: Float(id), count: 4))
            _ = cache.value(for: 7)
        }
        XCTAssertNotNil(cache.value(for: 7), "the hot row must not be evicted")
    }

    func testTheStripeCountBoundsTheEvictionScan() {
        // The eviction victim is found by scanning ONE stripe, so the stripe
        // count has to grow with the budget or a stripe ends up holding
        // thousands of rows and an eviction costs more than the read it saves.
        let rows = CountingRows(headDim: 160)
        let cached = FNNgramCachedRows(backing: rows, capacityBytes: 1 << 20)
        let budgetRows = (1 << 20) / (160 * MemoryLayout<Float>.size)
        XCTAssertGreaterThanOrEqual(cached.cache.stripeCount * 64, budgetRows,
                                    "the stripes must be numerous enough to bound a stripe's rows")
    }

    func testTheStripeCountIsCapped() {
        // An unbounded stripe count would allocate an unbounded number of
        // stripe objects for a large budget.
        let cache = FNNgramRowCache(capacityBytes: 1 << 30, rowBytes: 4, maxStripes: .max)
        XCTAssertLessThanOrEqual(cache.stripeCount, 1 << 16)
    }

    func testARowOfTheWrongLengthIsRefused() {
        let cache = FNNgramRowCache(capacityBytes: 1 << 20, rowBytes: 16, maxStripes: 1)
        // A row that is not the configured size would be under-charged and
        // could push the cache past its budget.
        XCTAssertThrowsError(try cache.storeChecked(1, [1, 2, 3]))
    }

    func testClearingEmptiesTheCacheAndItsCounters() {
        let rows = CountingRows()
        let cached = FNNgramCachedRows(backing: rows, capacityBytes: 1 << 20)
        _ = cached.row(1)
        _ = cached.row(1)
        XCTAssertEqual(cached.cache.hitCount, 1)
        cached.cache.clear()
        XCTAssertEqual(cached.cache.storedRowCount, 0)
        XCTAssertEqual(cached.cache.approximateBytes, 0)
        XCTAssertEqual(cached.cache.hitCount, 0)
        XCTAssertEqual(cached.cache.missCount, 0)
        _ = cached.row(1)
        XCTAssertEqual(rows.reads, 2, "the source must be read again after a clear")
    }

    func testTheCachedRowIsTheOneTheSourceReturned() {
        let rows = CountingRows(headDim: 3)
        let cached = FNNgramCachedRows(backing: rows, capacityBytes: 1 << 20)
        XCTAssertEqual(cached.row(7), [7, 7, 7])
        XCTAssertEqual(cached.row(7), [7, 7, 7])
        XCTAssertEqual(cached.cache.storedRowCount, 1)
    }

    func testCopyRowIsServedFromTheCacheToo() {
        let rows = CountingRows(headDim: 8)
        let cached = FNNgramCachedRows(backing: rows, capacityBytes: 1 << 20)
        var buf = [Float](repeating: -1, count: 8)
        buf.withUnsafeMutableBufferPointer { cached.copyRow(5, to: $0.baseAddress!) }
        XCTAssertEqual(buf, [Float](repeating: 5, count: 8))
        buf.withUnsafeMutableBufferPointer { cached.copyRow(5, to: $0.baseAddress!) }
        XCTAssertEqual(rows.reads, 1, "the second copy must not touch the source")
        XCTAssertEqual(buf, [Float](repeating: 5, count: 8))
    }

    func testTheHitRateOnASkewedPatternIsHigh() {
        // The cache's whole premise is that the n-gram ids repeat. This
        // asserts the cache does its job on a skewed pattern; it says nothing
        // about whether the REAL ids are skewed (see spec 30 §10 item 8).
        let rows = CountingRows(headDim: 160)
        let cached = FNNgramCachedRows(backing: rows, capacityBytes: 1 << 20)
        let ids = FNNgramRowCachePerfTests.skewedIds(count: 12_240, span: 2_048, hot: 256)
        for id in ids { _ = cached.row(id) }
        XCTAssertGreaterThan(cached.cache.hitRate, 0.8,
                             "a skewed pattern must hit the cache most of the time")
    }
}

/// A row source backed by a real file, so a measurement includes the `pread`
/// the cache exists to avoid. It mirrors the production reader: the row is
/// stored BF16 on disk and decoded on the way out, using the space past it.
private final class FNCacheFileRows: FNNgramRows, @unchecked Sendable {
    let headDim: Int
    private let fd: Int32
    private let rowBytes: Int

    init(headDim: Int, rowCount: Int, url: URL) throws {
        self.headDim = headDim
        self.rowBytes = headDim * 2
        var table = [UInt16](repeating: 0, count: rowCount * headDim)
        for i in 0..<rowCount {
            for j in 0..<headDim { table[i * headDim + j] = UInt16(truncatingIfNeeded: i % 1000) }
        }
        let data = table.withUnsafeBytes { Data(bytes: $0.baseAddress!, count: $0.count) }
        try data.write(to: url)
        let opened = Darwin.open(url.path, O_RDONLY)
        guard opened >= 0 else { throw NSError(domain: "FNCacheFileRows", code: 1) }
        self.fd = opened
    }

    func closeFile() { if fd >= 0 { Darwin.close(fd) } }

    func copyRow(_ id: Int, to dst: UnsafeMutablePointer<Float>) {
        let n = headDim
        let off = off_t(id * rowBytes)
        let raw = UnsafeMutableRawPointer(dst).assumingMemoryBound(to: UInt16.self).advanced(by: n)
        let got = pread(fd, raw, rowBytes, off)
        precondition(got == rowBytes, "short read at row \(id)")
        for i in 0..<n {
            dst[i] = Float(bitPattern: UInt32(UInt16(littleEndian: raw[i])) << 16)
        }
    }

    func row(_ id: Int) -> [Float] {
        [Float](unsafeUninitializedCapacity: headDim) { out, count in
            copyRow(id, to: out.baseAddress!)
            count = headDim
        }
    }
}

final class FNNgramRowCachePerfTests: XCTestCase {
    /// A deterministic generator. The quoted figures have to be reproducible,
    /// so the access pattern must not change from run to run.
    private struct Seeded: RandomNumberGenerator {
        private var state: UInt64

        init(seed: UInt64) {
            state = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        }

        mutating func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    /// The same skewed access pattern the premise rests on: a hot subset
    /// carries nine tenths of the traffic and the tail is nearly uniform.
    static func skewedIds(count: Int, span: Int, hot: Int) -> [Int] {
        var rng = Seeded(seed: 0x5EED_C0DE)
        let hotIds = (0..<hot).map { _ in Int.random(in: 0..<span, using: &rng) }
        var ids: [Int] = []
        ids.reserveCapacity(count)
        for i in 0..<count {
            ids.append(i % 10 < 9 ? hotIds[i % hotIds.count] : Int.random(in: 0..<span, using: &rng))
        }
        return ids
    }

    /// The working set FITS inside the cache, so every read after the first
    /// is a hit. This is the cache's BEST case, not the production regime.
    func testNgramRowCachePerformanceHotPath() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["QW_PERF_GATE"] == "1",
                          "perf gate — run via `make perf`, not the unit path")
        let headDim = 160
        let rowCount = 2_048
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("fn-ngram-cache-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let source = try FNCacheFileRows(headDim: headDim, rowCount: rowCount, url: tmp)
        defer { source.closeFile() }

        let ids = FNNgramRowCachePerfTests.skewedIds(count: 12_240, span: rowCount, hot: 256)
        var buf = [Float](repeating: 0, count: headDim)
        func sweep(_ source: any FNNgramRows) -> Double {
            let t0 = DispatchTime.now().uptimeNanoseconds
            for id in ids {
                buf.withUnsafeMutableBufferPointer { source.copyRow(id, to: $0.baseAddress!) }
            }
            let t1 = DispatchTime.now().uptimeNanoseconds
            return Double(t1 - t0) / Double(ids.count)
        }

        // Warm the file's pages first. Without this the FIRST sweep pays the
        // page faults for the whole table and every later sweep looks
        // impossibly fast next to it.
        _ = sweep(source)

        let cached = FNNgramCachedRows(backing: source, capacityBytes: 1 << 20)
        // Interleave the arms and keep the best of each, so a machine that
        // drifts moves both sides together. The FIRST cached pass is the fill
        // pass (the cache is empty); the later ones are all hits.
        var rawNs = Double.infinity
        var fillNs = Double.infinity
        var steadyNs = Double.infinity
        for round in 0..<3 {
            rawNs = min(rawNs, sweep(source))
            let pass = sweep(cached)
            if round == 0 { fillNs = pass } else { steadyNs = min(steadyNs, pass) }
        }
        let hitRate = cached.cache.hitRate

        print(String(format: "ngram cache hot path (%dD BF16 rows, file-backed): raw %.1f ns/read, "
                      + "cached fill pass %.1f ns/read, cached steady state %.1f ns/read, "
                      + "cumulative hit rate %.1f%%",
                      headDim, rawNs, fillNs, steadyNs, hitRate * 100))
        XCTAssertLessThan(steadyNs, rawNs, "a cache hit must cost less than the file read")
        #if DEBUG
        let minSpeedup = 1.5
        #else
        let minSpeedup = 4.0
        #endif
        XCTAssertLessThan(steadyNs * minSpeedup, rawNs,
                          "the cache must be at least \(Int(minSpeedup))x faster than the raw read")
    }

    /// The working set does NOT fit, so most reads miss and pay the whole miss
    /// path: the `pread`, the decode, a fresh row allocation and the eviction
    /// scan. This is the cache's WORST case and where it stops paying off.
    func testNgramRowCachePerformanceMissPath() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["QW_PERF_GATE"] == "1",
                          "perf gate — run via `make perf`, not the unit path")
        let headDim = 160
        let rowCount = 8_192
        let rowBytes = headDim * MemoryLayout<Float>.size
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("fn-ngram-miss-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let source = try FNCacheFileRows(headDim: headDim, rowCount: rowCount, url: tmp)
        defer { source.closeFile() }

        // Every id is distinct, so nothing is ever reused.
        let ids = Array(0..<rowCount)
        var buf = [Float](repeating: 0, count: headDim)
        func sweep(_ source: any FNNgramRows, _ ids: [Int]) -> Double {
            let t0 = DispatchTime.now().uptimeNanoseconds
            for id in ids {
                buf.withUnsafeMutableBufferPointer { source.copyRow(id, to: $0.baseAddress!) }
            }
            let t1 = DispatchTime.now().uptimeNanoseconds
            return Double(t1 - t0) / Double(ids.count)
        }

        // A cache big enough for the whole working set: the second sweep is
        // every hit, which is the cache's hit cost. The budget has to be
        // generous: the rows are split across stripes and spread unevenly.
        let big = FNNgramCachedRows(backing: source, capacityBytes: rowCount * rowBytes * 4)
        // A cache far smaller than the working set: almost every read misses.
        let small = FNNgramCachedRows(backing: source, capacityBytes: 64 * rowBytes)

        _ = sweep(source, ids)
        _ = sweep(big, ids)
        // Interleave the three arms and keep the best of each: a miss does
        // strictly more work than a raw read, so a miss that measured
        // cheaper than the raw read would mean the measurement is broken.
        var rawNs = Double.infinity
        var hitNs = Double.infinity
        var missNs = Double.infinity
        for _ in 0..<3 {
            rawNs = min(rawNs, sweep(source, ids))
            hitNs = min(hitNs, sweep(big, ids))
            missNs = min(missNs, sweep(small, ids))
        }

        // The hit rate at which the cache breaks even against the raw read.
        let breakEven = (missNs - rawNs) / max(0.001, missNs - hitNs)
        print(String(format: "ngram cache miss path: raw %.1f ns/read, hit %.1f ns/read, "
                      + "miss %.1f ns/read, break-even hit rate %.0f%%",
                      rawNs, hitNs, missNs, breakEven * 100))

        XCTAssertLessThan(hitNs, rawNs, "a hit must cost less than the raw read")
        XCTAssertGreaterThan(missNs, rawNs, "a miss must cost more than the raw read")
        XCTAssertLessThan(breakEven, 1.0, "the cache must be able to pay for itself at some hit rate")
        #if !DEBUG
        XCTAssertLessThan(breakEven, 0.75,
                          "the cache must break even below a 75% hit rate")
        #endif
    }
}
