// QWKVCachePagePressureTests.swift — Tests universal onPagePressure eviction and deadlock safety.
import Foundation
import Metal
import XCTest
@testable import QWKV
import QWCore
import QWHAL
import QWAlloc

private final class SharedPrefixHolder: @unchecked Sendable {
    private let lock = NSLock()
    var prefixes: [QWKVSharedPrefix] = []
    var calls: Int = 0

    func push(_ prefix: QWKVSharedPrefix) {
        lock.withLock { prefixes.append(prefix) }
    }

    func evictOldest() -> Bool {
        let victim: QWKVSharedPrefix? = lock.withLock {
            guard !prefixes.isEmpty else { return nil }
            return prefixes.removeFirst()
        }
        return victim != nil
    }

    func incrementAndReturnTrue() -> Bool {
        lock.withLock { calls += 1 }
        return true
    }

    func callCount() -> Int {
        lock.withLock { calls }
    }
}

final class QWKVCachePagePressureTests: XCTestCase {
    private static let layers = [0, 1]

    override func setUpWithError() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.supportsSimdgroupMMA == true,
            "QWKVCache GPU arena tests require bare-metal Apple Silicon GPU"
        )
    }

    private func makeTightCache(pagesPerPool: Int) throws -> (QWKVCache, QWArena) {
        let hal = try QWHal(geometry: .qwen38_27b)
        let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: .qwen38_27b)
        let poolBytes = pageBytes * pagesPerPool * Self.layers.count * 2
        let tables = QWKVCache.tableSlotBytes * 2 * Self.layers.count * QWKVCache.maxSeqs
        let arena = try QWArena(hal: hal, weights: 0, activations: 0,
                                kv: poolBytes + tables, gdnStates: 0)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: Self.layers,
                                  mode: .f16, geometry: .qwen38_27b)
        return (cache, arena)
    }

    func testAllocateEvictsSharedPrefixViaDeinitOutsideLock() throws {
        let (cache, arena) = try makeTightCache(pagesPerPool: 2)
        let holder = SharedPrefixHolder()
        cache.onPagePressure = { holder.evictOldest() }

        let d1 = try cache.allocate(promptTokens: 256)
        try cache.append(d1, positions: 0..<256, layer: 0, kind: .k,
                         src: [UInt16](repeating: 0x3C00, count: 256 * QWModelGeometry.qwen38_27b.kvChannels))
        holder.push(try cache.captureSharedPrefix(d1))
        cache.finish(d1)

        let d2 = try cache.allocate(promptTokens: 256)
        try cache.append(d2, positions: 0..<256, layer: 0, kind: .k,
                         src: [UInt16](repeating: 0x3C00, count: 256 * QWModelGeometry.qwen38_27b.kvChannels))
        holder.push(try cache.captureSharedPrefix(d2))
        cache.finish(d2)

        XCTAssertEqual(cache.freePageCount, 0)
        // Non-cached admission triggers onPagePressure -> QWKVSharedPrefix.deinit -> releaseSharedPrefix.
        let nonCached = try cache.allocate(promptTokens: 512)
        XCTAssertEqual(cache.freePageCount, 0)
        cache.finish(nonCached)
        XCTAssertEqual(cache.freePageCount, cache.totalPages)
        _ = arena
    }

    func testAdoptGrowPrepareWritesAndAppendTriggerPagePressure() throws {
        let (cache, arena) = try makeTightCache(pagesPerPool: 2)
        let holder = SharedPrefixHolder()
        cache.onPagePressure = { holder.evictOldest() }

        let d1 = try cache.allocate(promptTokens: 200)
        try cache.append(d1, positions: 0..<200, layer: 0, kind: .k,
                         src: [UInt16](repeating: 0x3C00, count: 200 * QWModelGeometry.qwen38_27b.kvChannels))
        let basePrefix = try cache.captureSharedPrefix(d1)
        cache.finish(d1)

        let d2 = try cache.allocate(promptTokens: 100)
        try cache.append(d2, positions: 0..<100, layer: 0, kind: .k,
                         src: [UInt16](repeating: 0x3C00, count: 100 * QWModelGeometry.qwen38_27b.kvChannels))
        holder.push(try cache.captureSharedPrefix(d2))
        cache.finish(d2)
        XCTAssertEqual(cache.freePageCount, 0)

        // 1. allocate(adopting:) needing a future page evicts d2 via onPagePressure.
        let adopter = try cache.allocate(adopting: basePrefix, promptTokens: 300)
        XCTAssertEqual(cache.freePageCount, 0)
        cache.finish(adopter)

        // Re-fill the free page with another cached prefix.
        let d3 = try cache.allocate(promptTokens: 100)
        try cache.append(d3, positions: 0..<100, layer: 0, kind: .k,
                         src: [UInt16](repeating: 0x3C00, count: 100 * QWModelGeometry.qwen38_27b.kvChannels))
        holder.push(try cache.captureSharedPrefix(d3))
        cache.finish(d3)

        // 2. prepareWrites COW on shared tail page evicts d3.
        let cowSeq = try cache.allocate(adopting: basePrefix, promptTokens: 200)
        try cache.prepareWrites(cowSeq, positions: 199..<200)

        // 3. Re-fill via grow? Now cowSeq owns 2 pages? No, basePrefix held 1 + cowSeq replaced 1 = 2 pages.
        cache.finish(cowSeq)
        let d4 = try cache.allocate(promptTokens: 100)
        try cache.append(d4, positions: 0..<100, layer: 0, kind: .k,
                         src: [UInt16](repeating: 0x3C00, count: 100 * QWModelGeometry.qwen38_27b.kvChannels))
        holder.push(try cache.captureSharedPrefix(d4))
        cache.finish(d4)

        let growSeq = try cache.allocate(adopting: basePrefix, promptTokens: 200)
        // Releasing basePrefix is not done; growing beyond 256 needs d4 evicted.
        try cache.grow(growSeq, to: 300)
        cache.finish(growSeq)

        // 4. append COW on partial tail page evicts via onPagePressure too.
        let d5 = try cache.allocate(promptTokens: 100)
        try cache.append(d5, positions: 0..<100, layer: 0, kind: .k,
                         src: [UInt16](repeating: 0x3C00, count: 100 * QWModelGeometry.qwen38_27b.kvChannels))
        holder.push(try cache.captureSharedPrefix(d5))
        cache.finish(d5)
        let appendSeq = try cache.allocate(adopting: basePrefix, promptTokens: 200)
        try cache.append(appendSeq, positions: 199..<200, layer: 0, kind: .k,
                         src: [UInt16](repeating: 0x3C01, count: QWModelGeometry.qwen38_27b.kvChannels))
        cache.finish(appendSeq)

        cache.releaseSharedPrefix(basePrefix)
        XCTAssertEqual(cache.freePageCount, cache.totalPages)
        _ = arena
    }

    func testOnPagePressureBoundedWhenHandlerNeverFreesPages() throws {
        let (cache, arena) = try makeTightCache(pagesPerPool: 1)
        let live = try cache.allocate(promptTokens: 256)
        let holder = SharedPrefixHolder()
        cache.onPagePressure = { holder.incrementAndReturnTrue() }
        XCTAssertThrowsError(try cache.allocate(promptTokens: 256)) { error in
            guard case QWKVError.outOfPages = error else {
                return XCTFail("expected outOfPages, got \(error)")
            }
        }
        let calls = holder.callCount()
        XCTAssertGreaterThanOrEqual(calls, 1024)
        XCTAssertLessThanOrEqual(calls, max(1024, cache.totalPages) + 1)
        cache.finish(live)
        _ = arena
    }
}
