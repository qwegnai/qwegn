import XCTest
import Metal
import Foundation
@testable import QWKV
import QWCore
import QWHAL
import QWAlloc

private final class QWKVSharedPrefixBox: @unchecked Sendable {
    private let lock = NSLock()
    private var value: QWKVSharedPrefix?

    init(_ value: QWKVSharedPrefix) { self.value = value }

    func borrow() -> QWKVSharedPrefix? {
        lock.lock(); defer { lock.unlock() }
        return value
    }

    func releaseAndDrop(from cache: QWKVCache) {
        lock.lock()
        let prefix = value
        value = nil
        lock.unlock()
        if let prefix { cache.releaseSharedPrefix(prefix) }
    }

    func drop() {
        lock.lock()
        value = nil
        lock.unlock()
    }
}

private final class QWKVConcurrentFailures: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [String] = []

    func append(_ value: String) {
        lock.lock(); defer { lock.unlock() }
        values.append(value)
    }

    var all: [String] {
        lock.lock(); defer { lock.unlock() }
        return values
    }
}

/// C10 paged-KV gate (specs/10-qwkv.md): exact page accounting (RSS ==
/// pages×bytes, zero fragmentation), device page-table rewrite-in-place
/// (binds never move), burst-vs-streamed byte parity (spec: attention
final class QWKVCacheTests: XCTestCase {

    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()
    static let layers = [3, 7, 11, 15]        // a small attn-layer subset (geometry says 16; tests use 4 for speed)

    // deterministic f16 "K/V row" generator — same input → same bytes both
    // burst and streamed paths (the parity object).
    static func row(_ tok: Int, _ e: Int) -> UInt16 {
        var x = UInt16(truncatingIfNeeded: (tok &* 2654435761) &+ (e &* 40503) &+ 7)
        x = UInt16(truncatingIfNeeded: Int(x) &* 2246822519 &+ 31)
        return 0x3800 | (x & 0x7FF)          // ±[1,2) f16
    }
    static func kvSlice(from tok0: Int, count: Int) -> [UInt16] {
        let re = QWKVCache.kvHeads * QWKVCache.headDim
        var out = [UInt16](repeating: 0, count: count * re)
        for t in 0..<count { for e in 0..<re { out[t * re + e] = row(tok0 + t, e) } }
        return out
    }

    func makeCache(tokensCeiling: Int = 4096, mode: QWKVMode = .f16) throws -> (QWKVCache, QWArena) {
        // kv region sized for: pools + tables. 4 layers × 2 kinds.
        let pageBytes = QWKVCache.pageBytes(mode: mode)
        let needPages = (tokensCeiling + 255) / 256
        let poolBytes = pageBytes * needPages * 2                // headroom ×2
        let tables = QWKVCache.tableSlotBytes * 2 * Self.layers.count * QWKVCache.maxSeqs
        let arena = try QWArena(hal: Self.hal, weights: 0, activations: 0,
                                kv: poolBytes * Self.layers.count + tables, gdnStates: 0)
        let cache = try QWKVCache(hal: Self.hal, arena: arena, layers: Self.layers, mode: mode, geometry: .qwen38_27b)
        return (cache, arena)
    }

    func testPrivatePrefixSurvivesDonorFinishAndTwoIndependentAdopters() throws {
        let (cache, arena) = try makeCache(tokensCeiling: 1024)
        let free = cache.freePageCount
        for count in [0, 1, 255, 256, 257] {
            let donor = try cache.allocate(promptTokens: max(1, count))
            let original = Self.kvSlice(from: 0, count: count)
            for layer in Self.layers {
                for kind in QWKVKind.allCases {
                    try cache.append(donor, positions: 0..<count, layer: layer, kind: kind, src: original)
                }
            }
            let cost = count * QWKVCache.rowBytes * Self.layers.count * 2
            if cost > 0 {
                XCTAssertThrowsError(try cache.capturePrivatePrefix(donor, maxBytes: cost - 1))
            }
            let prefix = try cache.capturePrivatePrefix(donor, maxBytes: cost)
            XCTAssertEqual(prefix.memoryBytes, cost)
            XCTAssertEqual(prefix.tokenCount, count)
            cache.finish(donor)
            XCTAssertEqual(cache.freePageCount, free, "immutable bytes must not pin donor pages")
            let first = try cache.allocate(promptTokens: count + 1)
            let second = try cache.allocate(promptTokens: count + 1)
            try cache.adoptPrivatePrefix(prefix, into: first)
            try cache.adoptPrivatePrefix(prefix, into: second)
            for layer in Self.layers {
                for kind in QWKVKind.allCases {
                    XCTAssertEqual(readKV(cache: cache, h: first, layer: layer, kind: kind, tokens: count), original)
                    XCTAssertEqual(readKV(cache: cache, h: second, layer: layer, kind: kind, tokens: count), original)
                    XCTAssertTrue(Set(cache.pageIDs(first, layer: layer, kind: kind))
                        .isDisjoint(with: cache.pageIDs(second, layer: layer, kind: kind)))
                    try cache.append(first, positions: count..<(count + 1), layer: layer, kind: kind,
                                     src: Self.kvSlice(from: 999, count: 1))
                    XCTAssertEqual(readKV(cache: cache, h: second, layer: layer, kind: kind, tokens: count), original)
                }
            }
            cache.finish(first); cache.finish(second)
            XCTAssertEqual(cache.freePageCount, free)
        }
        _ = arena
    }

    func testPrivatePrefixRejectsForeignNonemptyAndShortDestinations() throws {
        let (cache, arena) = try makeCache(tokensCeiling: 1024)
        let donor = try cache.allocate(promptTokens: 257)
        cache.markWritten(donor, upTo: 257)
        let prefix = try cache.capturePrivatePrefix(donor, maxBytes: 8 << 20)
        cache.finish(donor)
        let short = try cache.allocate(promptTokens: 1)
        XCTAssertThrowsError(try cache.adoptPrivatePrefix(prefix, into: short))
        XCTAssertEqual(cache.written(short), 0)
        try cache.grow(short, to: 257)
        cache.markWritten(short, upTo: 1)
        XCTAssertThrowsError(try cache.adoptPrivatePrefix(prefix, into: short))
        XCTAssertEqual(cache.written(short), 1)
        cache.finish(short)
        XCTAssertThrowsError(try cache.adoptPrivatePrefix(prefix, into: short))
        XCTAssertThrowsError(try cache.capturePrivatePrefix(short, maxBytes: 8 << 20))
        let (foreign, foreignArena) = try makeCache(tokensCeiling: 1024)
        let other = try foreign.allocate(promptTokens: 257)
        XCTAssertThrowsError(try foreign.adoptPrivatePrefix(prefix, into: other))
        XCTAssertThrowsError(try foreign.capturePrivatePrefix(donor, maxBytes: 8 << 20))
        XCTAssertEqual(foreign.written(other), 0)
        foreign.finish(other)
        _ = arena; _ = foreignArena
    }

    func testSharedPrefixTwoAdoptersCowAndFinishedDonorBoundaries() throws {
        let (cache, arena) = try makeCache(tokensCeiling: 1024)
        let total = cache.totalPages
        for count in [0, 1, 255, 256, 257] {
            let donor = try cache.allocate(promptTokens: max(1, count))
            let original = Self.kvSlice(from: 0, count: count)
            if count > 0 {
                for layer in Self.layers {
                    for kind in QWKVKind.allCases {
                        try cache.append(donor, positions: 0..<count, layer: layer, kind: kind, src: original)
                    }
                }
            }
            let prefix = try cache.captureSharedPrefix(donor)
            let prefixPages = (count + QWKVCache.pageTokens - 1) / QWKVCache.pageTokens
            let donorIDs = Dictionary(uniqueKeysWithValues: Self.layers.flatMap { layer in
                QWKVKind.allCases.map { kind in
                    ("\(layer):\(kind.rawValue)", cache.pageIDs(donor, layer: layer, kind: kind))
                }
            })
            let first = try cache.allocate(adopting: prefix, promptTokens: count + 1)
            let second = try cache.allocate(adopting: prefix, promptTokens: count + 1)

            for layer in Self.layers {
                for kind in QWKVKind.allCases {
                    let key = "\(layer):\(kind.rawValue)"
                    let sharedIDs = Array(donorIDs[key]!.prefix(prefixPages))
                    XCTAssertEqual(Array(cache.pageIDs(first, layer: layer, kind: kind).prefix(prefixPages)),
                                   sharedIDs, "first adopter did not use the cached pages")
                    XCTAssertEqual(Array(cache.pageIDs(second, layer: layer, kind: kind).prefix(prefixPages)),
                                   sharedIDs, "second adopter did not use the cached pages")
                    XCTAssertEqual(readKV(cache: cache, h: first, layer: layer, kind: kind, tokens: count), original)
                    XCTAssertEqual(readKV(cache: cache, h: second, layer: layer, kind: kind, tokens: count), original)
                }
            }

            if count >= 1 {
                let replacement = Self.kvSlice(from: 999, count: 1)
                for layer in Self.layers {
                    for kind in QWKVKind.allCases {
                        try cache.append(first, positions: count..<(count + 1), layer: layer, kind: kind,
                                         src: replacement)
                        XCTAssertEqual(Array(readKV(cache: cache, h: first, layer: layer, kind: kind,
                                                   tokens: count + 1).suffix(QWKVCache.kvHeads * QWKVCache.headDim)),
                                       replacement, "append after COW did not write the requested bytes")
                        XCTAssertEqual(readKV(cache: cache, h: first, layer: layer, kind: kind, tokens: count),
                                       original, "COW changed bytes in the retained prefix")
                        XCTAssertEqual(readKV(cache: cache, h: second, layer: layer, kind: kind, tokens: count),
                                       original, "one adopter changed another adopter's prefix")
                        XCTAssertEqual(readKV(cache: cache, h: donor, layer: layer, kind: kind, tokens: count),
                                       original, "one adopter changed the donor's prefix")
                        let tailPage = (count - 1) / QWKVCache.pageTokens
                        let firstIDs = cache.pageIDs(first, layer: layer, kind: kind)
                        let donorPages = donorIDs["\(layer):\(kind.rawValue)"]!
                        if count % QWKVCache.pageTokens != 0 {
                            XCTAssertNotEqual(firstIDs[tailPage], donorPages[tailPage],
                                              "partial shared tail must COW before append")
                        }
                    }
                }
            }

            cache.finish(donor)
            let pagesNeeded = max(1, (count + 1 + QWKVCache.pageTokens - 1) / QWKVCache.pageTokens)
            let copiedTail = count > 0 && count % QWKVCache.pageTokens != 0 ? 1 : 0
            let expectedRetained = (prefixPages + 2 * (pagesNeeded - prefixPages) + copiedTail)
                * Self.layers.count * 2
            XCTAssertEqual(cache.freePageCount, total - expectedRetained,
                           "prefix and adopters must retain physical pages after donor finish")
            cache.finish(first)
            cache.finish(second)
            if prefixPages > 0 {
                XCTAssertLessThan(cache.freePageCount, total, "prefix owner must retain pages after adopters finish")
            } else {
                XCTAssertEqual(cache.freePageCount, total, "empty prefix owns no physical pages")
            }
            cache.releaseSharedPrefix(prefix)
            XCTAssertEqual(cache.freePageCount, total, "last shared reference must return every page")
        }
        _ = arena
    }

    func testSharedPrefixAdmissionUsesOnlyFuturePagesAndRefusesWithoutMutation() throws {
        let (cache, arena) = try makeCache(tokensCeiling: 257)
        let total = cache.totalPages
        let donor = try cache.allocate(promptTokens: 1)
        try cache.append(donor, positions: 0..<1, layer: Self.layers[0], kind: .k,
                         src: Self.kvSlice(from: 0, count: 1))
        let prefix = try cache.captureSharedPrefix(donor)
        let freeBefore = cache.freePageCount

        XCTAssertThrowsError(try cache.allocate(promptTokens: 257),
                             "private admission needs two pages when only one is free")
        XCTAssertEqual(cache.freePageCount, freeBefore, "failed private admission leaked a page")
        let foreignSetup = try makeCache(tokensCeiling: 1)
        let (foreign, foreignArena) = foreignSetup
        XCTAssertThrowsError(try foreign.allocate(adopting: prefix, promptTokens: 257))
        XCTAssertEqual(foreign.freePageCount, foreign.totalPages, "foreign validation allocated pages")

        let admitted = try cache.allocate(adopting: prefix, promptTokens: 257)
        XCTAssertEqual(cache.freePageCount, 0, "shared admission should allocate only the one future page")
        XCTAssertEqual(cache.pageIDs(admitted, layer: Self.layers[0], kind: .k)[0],
                       cache.pageIDs(donor, layer: Self.layers[0], kind: .k)[0])
        let idsBeforeInvalidWrite = cache.pageIDs(admitted, layer: Self.layers[0], kind: .k)
        XCTAssertThrowsError(try cache.prepareWrites(admitted, positions: 512..<513))
        XCTAssertEqual(cache.pageIDs(admitted, layer: Self.layers[0], kind: .k), idsBeforeInvalidWrite)
        XCTAssertEqual(cache.freePageCount, 0, "bounds refusal changed allocation state")

        cache.finish(donor)
        cache.finish(admitted)
        cache.releaseSharedPrefix(prefix)
        XCTAssertEqual(cache.freePageCount, total)
        XCTAssertThrowsError(try cache.allocate(adopting: prefix, promptTokens: 257),
                              "released prefixes must be stale")
        XCTAssertEqual(cache.freePageCount, total, "stale-prefix refusal allocated pages")
        _ = arena; _ = foreignArena
    }

    func testSharedPrefixAdoptsIntoPreallocatedHandleTransactionally() throws {
        let (cache, arena) = try makeCache(tokensCeiling: 1024)
        let total = cache.totalPages
        let donor = try cache.allocate(promptTokens: 257)
        try cache.append(donor, positions: 0..<257, layer: Self.layers[0], kind: .k,
                         src: Self.kvSlice(from: 0, count: 257))
        let prefix = try cache.captureSharedPrefix(donor)
        let short = try cache.allocate(promptTokens: 1)
        let shortIDs = cache.pageIDs(short, layer: Self.layers[0], kind: .k)
        let beforeRefusal = cache.freePageCount
        XCTAssertThrowsError(try cache.adoptSharedPrefix(prefix, into: short))
        XCTAssertEqual(cache.pageIDs(short, layer: Self.layers[0], kind: .k), shortIDs)
        XCTAssertEqual(cache.freePageCount, beforeRefusal, "short shared adoption changed page ownership")
        cache.finish(short)

        let consumer = try cache.allocate(promptTokens: 512)
        try cache.adoptSharedPrefix(prefix, into: consumer)
        for layer in Self.layers {
            for kind in QWKVKind.allCases {
                XCTAssertEqual(Array(cache.pageIDs(consumer, layer: layer, kind: kind).prefix(2)),
                               Array(cache.pageIDs(donor, layer: layer, kind: kind).prefix(2)))
            }
        }
        try cache.append(consumer, positions: 257..<258, layer: Self.layers[0], kind: .k,
                         src: Self.kvSlice(from: 999, count: 1))
        XCTAssertNotEqual(cache.pageIDs(consumer, layer: Self.layers[0], kind: .k)[1],
                          cache.pageIDs(donor, layer: Self.layers[0], kind: .k)[1],
                          "preallocated adopter did not COW its partial tail")
        cache.finish(donor)
        cache.finish(consumer)
        cache.releaseSharedPrefix(prefix)
        XCTAssertEqual(cache.freePageCount, total)
        _ = arena
    }

    func testSharedPrefixAndLegacySnapshotPinsReleaseExactly() throws {
        let (cache, arena) = try makeCache(tokensCeiling: 1024)
        let total = cache.totalPages
        let donor = try cache.allocate(promptTokens: 257)
        try cache.append(donor, positions: 0..<257, layer: Self.layers[0], kind: .k,
                         src: Self.kvSlice(from: 0, count: 257))
        let snapshot = try cache.snapshot(donor)
        let prefix = try cache.captureSharedPrefix(donor)
        try cache.rollback(donor, by: 1)
        cache.releaseSharedPrefix(prefix)
        let pinnedFree = cache.freePageCount
        XCTAssertLessThan(pinnedFree, total, "snapshot or surviving first page should retain pages")
        cache.release(snapshot)
        XCTAssertEqual(cache.freePageCount, total - Self.layers.count * 2,
                       "legacy pinned pages should release while the live first page remains")
        cache.finish(donor)
        XCTAssertEqual(cache.freePageCount, total)
        _ = arena
    }

    func testSharedPrefixRetainReleaseChurnReturnsPages() throws {
        let (cache, arena) = try makeCache(tokensCeiling: 512)
        let total = cache.totalPages
        for _ in 0..<40 {
            let donor = try cache.allocate(promptTokens: 257)
            try cache.append(donor, positions: 0..<257, layer: Self.layers[0], kind: .v,
                             src: Self.kvSlice(from: 0, count: 257))
            let prefix = try cache.captureSharedPrefix(donor)
            let consumer = try cache.allocate(adopting: prefix, promptTokens: 300)
            cache.finish(donor)
            cache.finish(consumer)
            cache.releaseSharedPrefix(prefix)
            XCTAssertEqual(cache.freePageCount, total, "shared-page churn leaked a page")
        }
        _ = arena
    }

    func testSharedPrefixReleaseAndDeinitSerializeWithPoolLifecycle() throws {
        let (cache, arena) = try makeCache(tokensCeiling: 2048)
        let totalPages = cache.totalPages

        for explicitRelease in [true, false] {
            let donor = try cache.allocate(promptTokens: 257)
            try cache.append(donor, positions: 0..<257, layer: Self.layers[0], kind: .k,
                             src: Self.kvSlice(from: 0, count: 257))
            let (prefixBox, consumer) = try {
                let prefix = try cache.captureSharedPrefix(donor)
                let consumer = try cache.allocate(adopting: prefix, promptTokens: 258)
                return (QWKVSharedPrefixBox(prefix), consumer)
            }()

            let failures = QWKVConcurrentFailures()
            let ready = DispatchGroup()
            let done = DispatchGroup()
            let start = DispatchSemaphore(value: 0)
            let workerCount = 5

            for worker in 0..<workerCount {
                ready.enter()
                done.enter()
                DispatchQueue.global().async {
                    ready.leave()
                    start.wait()
                    defer { done.leave() }

                    switch worker {
                    case 0:
                        for _ in 0..<12 {
                            do {
                                let handle = try cache.allocate(promptTokens: 1)
                                defer { cache.finish(handle) }
                                try cache.grow(handle, to: 513)
                            } catch {
                                failures.append("ordinary allocate/grow: \(error)")
                            }
                        }
                    case 1:
                        defer { cache.finish(consumer) }
                        do {
                            try cache.grow(consumer, to: 513)
                            try cache.prepareWrites(consumer, positions: 256..<257)
                        } catch {
                            failures.append("consumer grow/COW: \(error)")
                        }
                    case 2:
                        cache.finish(donor)
                    case 3:
                        for _ in 0..<24 {
                            guard let prefix = prefixBox.borrow() else { continue }
                            do {
                                let handle = try cache.allocate(adopting: prefix, promptTokens: 258)
                                defer { cache.finish(handle) }
                                try cache.grow(handle, to: 513)
                                try cache.prepareWrites(handle, positions: 256..<257)
                            } catch QWKVError.badArgument(let message)
                                where message.contains("shared prefix") {
                                // Release may win the cache lock after borrow and
                                // before admission validates the prefix.
                            } catch {
                                failures.append("racing shared admission/COW: \(error)")
                            }
                        }
                    default:
                        if explicitRelease {
                            prefixBox.releaseAndDrop(from: cache)
                        } else {
                            prefixBox.drop()
                        }
                        for _ in 0..<12 {
                            do {
                                let handle = try cache.allocate(promptTokens: 1)
                                defer { cache.finish(handle) }
                                try cache.grow(handle, to: 513)
                            } catch {
                                failures.append("release-side allocate/grow: \(error)")
                            }
                        }
                    }
                }
            }

            ready.wait()
            for _ in 0..<workerCount { start.signal() }
            done.wait()
            prefixBox.drop()
            cache.finish(donor)
            cache.finish(consumer)

            XCTAssertTrue(failures.all.isEmpty,
                          "concurrent cache operations failed: \(failures.all.joined(separator: "; "))")
            XCTAssertEqual(cache.freePageCount, totalPages,
                           "prefix release/deinit race leaked or double-freed physical pages")

            // Sequential churn beyond the table-slot count proves every
            // sequence identity used by concurrent workers was recycled.
            for _ in 0..<(QWKVCache.maxSeqs * 2) {
                let handle = try cache.allocate(promptTokens: 1)
                cache.finish(handle)
            }
            XCTAssertEqual(cache.freePageCount, totalPages,
                           "sequence-ID recycle churn changed page accounting")
        }
        _ = arena
    }

    /// Page ids are pool-local AND per (layer, kind): allocate must hand out
    /// distinct ids within each pool and mirror them into the device slot.
    func testAllocatePageAccounting() throws {
        let (cache, arena) = try makeCache(tokensCeiling: 2048)
        let total = cache.totalPages
        let h = try cache.allocate(promptTokens: 1000)          // 4 pages/layer
        let perLayer = (1000 + 255) / 256
        for l in Self.layers {
            let kIDs = cache.pageIDs(h, layer: l, kind: .k)
            let vIDs = cache.pageIDs(h, layer: l, kind: .v)
            XCTAssertEqual(kIDs.count, perLayer, "layer \(l) K pages")
            XCTAssertEqual(Set(kIDs).count, perLayer, "K ids distinct")
            XCTAssertEqual(Set(vIDs).count, perLayer, "V ids distinct")
        }
        // RSS law: consumed == total − free == perLayer × 2 kinds × layers
        let consumed = total - cache.freePageCount
        XCTAssertEqual(consumed, perLayer * 2 * Self.layers.count)
        _ = arena
    }

    /// Device page-table slot rewrite-in-place: layerViews' table base MUST
    /// be stable across grow (kernel bindings never relocate).
    func testTableBaseStableAcrossGrow() throws {
        let (cache, _) = try makeCache(tokensCeiling: 2048)
        let h = try cache.allocate(promptTokens: 256)
        let l = Self.layers[1]
        let before = try cache.layerViews(h, layer: l)
        let base0K = before.k.table.region.base, base0V = before.v.table.region.base
        try cache.grow(h, to: 1200)
        let after = try cache.layerViews(h, layer: l)
        XCTAssertEqual(after.k.table.region.base, base0K, "K table base moved on grow")
        XCTAssertEqual(after.v.table.region.base, base0V, "V table base moved on grow")
        XCTAssertEqual(after.k.table.shape.dims[0], (1200 + 255) / 256, "grown K table length")
    }

    /// Burst-vs-streamed byte parity (spec: attention outputs MUST be
    /// identical whether KV was written in one prefill burst or streamed 1
    /// token at a time). Proven at the page bytes the C09 kernel reads, using
    func testBurstVsStreamedParity() throws {
        let (cacheA, arenaA) = try makeCache(tokensCeiling: 1024)
        let (cacheB, arenaB) = try makeCache(tokensCeiling: 1024)
        let l = Self.layers[2]
        let N = 600
        let full = Self.kvSlice(from: 0, count: N)

        let a = try cacheA.allocate(promptTokens: N)
        try cacheA.grow(a, to: N)
        try cacheA.append(a, positions: 0..<N, layer: l, kind: .k, src: full)

        let b = try cacheB.allocate(promptTokens: 1)
        try cacheB.grow(b, to: N)
        for t in 0..<N {
            try cacheB.append(b, positions: t..<(t + 1), layer: l, kind: .k,
                              src: Self.kvSlice(from: t, count: 1))
        }
        let bytesA = readKV(cache: cacheA, h: a, layer: l, kind: .k, tokens: N)
        let bytesB = readKV(cache: cacheB, h: b, layer: l, kind: .k, tokens: N)
        XCTAssertEqual(bytesA, bytesB, "burst vs streamed K bytes diverged")
        _ = arenaA; _ = arenaB
    }

    func readKV(cache: QWKVCache, h: QWKVHandle, layer: Int,
                kind: QWKVKind, tokens: Int) -> [UInt16] {
        let re = QWKVCache.kvHeads * QWKVCache.headDim
        let pageBytes = QWKVCache.pageBytes(mode: .f16)
        let ids = cache.pageIDs(h, layer: layer, kind: kind)
        let views = (try? cache.layerViews(h, layer: layer))!
        let pool = kind == .k ? views.k.pool : views.v.pool
        var out = [UInt16](repeating: 0, count: tokens * re)
        for t in 0..<tokens {
            let pg = t / QWKVCache.pageTokens, off = t % QWKVCache.pageTokens
            let base = Int(ids[pg]) * pageBytes + off * re * 2
            pool.buffer.contents().advanced(by: pool.base + base)
                .withMemoryRebound(to: UInt16.self, capacity: re) { p in
                    for e in 0..<re { out[t * re + e] = p[e] }
                }
        }
        return out
    }

    /// Rollback (MTP reject): whole pages past the new end return to their
    /// pool; writtenTokens tracks the new high-water.
    func testRollbackFreesWholePages() throws {
        let (cache, _) = try makeCache(tokensCeiling: 2048)
        let h = try cache.allocate(promptTokens: 1)
        try cache.grow(h, to: 1000)
        let consumed1 = cache.totalPages - cache.freePageCount
        // write so rollback has real state; then reject 300 tokens
        try cache.append(h, positions: 0..<1000, layer: Self.layers[0], kind: .k,
                         src: Self.kvSlice(from: 0, count: 1000))
        try cache.rollback(h, by: 300)                        // → 700 tokens
        let keep = (700 + 255) / 256
        for l in Self.layers {
            XCTAssertEqual(cache.pageIDs(h, layer: l, kind: .k).count, keep, "layer \(l) K pages after rollback")
        }
        XCTAssertLessThan(cache.totalPages - cache.freePageCount, consumed1, "rollback freed pages")
    }

    /// Snapshot pins pages a rollback would otherwise free; release returns
    /// them and accounting goes back to the live-table total (exactness law).
    func testSnapshotPinAccounting() throws {
        let (cache, _) = try makeCache(tokensCeiling: 2048)
        let h = try cache.allocate(promptTokens: 1)
        try cache.grow(h, to: 700)
        try cache.append(h, positions: 0..<700, layer: Self.layers[0], kind: .k,
                         src: Self.kvSlice(from: 0, count: 700))   // cursor = 700
        let live = cache.totalPages - cache.freePageCount
        let snap = try cache.snapshot(h)
        try cache.rollback(h, by: 500)                        // → 200 tokens
        let afterRollback = cache.totalPages - cache.freePageCount
        // pinned pages NOT reclaimed while the snapshot is alive
        XCTAssertEqual(afterRollback, live, "pinned pages held during rollback")
        cache.release(snap)
        // after release, live == fresh-table total for 200 tokens (each layer
        // keeps ceil(200/256)=1 page × 2 kinds × layers)
        let keep = (200 + 255) / 256
        XCTAssertEqual(cache.totalPages - cache.freePageCount, keep * 2 * Self.layers.count,
                       "exact accounting after pin release")
    }

    /// Out-of-range kernel table reads see page 0, never a stale id: uploadSlot
    /// zeroes the tail of each device slot past the current id count.
    func testTableTailZeroedOnUpload() throws {
        let (cache, _) = try makeCache(tokensCeiling: 1024)
        let h = try cache.allocate(promptTokens: 256)         // 1 page/layer
        let l = Self.layers[0]
        let views = try cache.layerViews(h, layer: l)
        let ptr = views.k.table.region.pointer.assumingMemoryBound(to: Int32.self)
        let ids = cache.pageIDs(h, layer: l, kind: .k)
        XCTAssertEqual(ptr[0], ids[0], "first id mirrored")
        for i in ids.count..<1024 { XCTAssertEqual(ptr[i], 0, "tail slot \(i) not zero") }
    }

    /// Spec ceiling: a single sequence may not exceed maxTokensPerSeq.
    func testTokenCeilingGuard() throws {
        // pools get real headroom so init succeeds; the ceiling guard must
        // fire in allocate() itself (prompt > one 4 KiB table slot of ids).
        let (cache, _) = try makeCache(tokensCeiling: 1024)
        XCTAssertThrowsError(try cache.allocate(promptTokens: QWKVCache.maxTokensPerSeq + 256)) { err in
            XCTAssertTrue("\(err)".contains("ceiling"), "guard must reject past the slot ceiling, got \(err)")
        }
    }

    func testModeBytesMath() {
        // f16: 64 KB/token/layer (spec ROOFLINE); q8 halves it.
        XCTAssertEqual(QWKVCache.kvBytes(tokens: 1, layers: 1, mode: .f16), 4 * 256 * 2 * 2)
        XCTAssertEqual(QWKVCache.kvBytes(tokens: 1, layers: 1, mode: .q8) * 2,
                       QWKVCache.kvBytes(tokens: 1, layers: 1, mode: .f16),
                       "q8 halves KV")
        XCTAssertEqual(QWKVCache.kvBytes(tokens: 262144, layers: 16, mode: .f16) / 1024 / 1024 / 1024, 16,
                       "262K fp16 KV ≈ 16.8 GB budget anchor")
    }

    /// THE SOAK LAW (C21 M1, spec-21): a finished session returns EVERY page
    /// to its pool and its seq id recycles. Without finish(), a 10k-session
    /// soak dies at session 65 (maxSeqs table slots exhausted) and leaks a
    func testFinishReturnsPagesAndRecyclesSeqIds() throws {
        let (cache, _) = try makeCache(tokensCeiling: 2048)
        let total = cache.totalPages
        // one cycle: allocate 300 tokens (2 pages/layer), write, finish
        let h = try cache.allocate(promptTokens: 300)
        let perLayer = 2
        XCTAssertEqual(total - cache.freePageCount, perLayer * 2 * Self.layers.count)
        cache.finish(h)
        XCTAssertEqual(cache.freePageCount, total, "finish must return EVERY page")
        // recycled id: next allocate reuses the freed seq id (FIFO free-list,
        // QWState's removeFirst law — the FIRST freed id comes back first)
        let seqOfH = h.seq
        let h2 = try cache.allocate(promptTokens: 256)
        XCTAssertEqual(h2.seq, seqOfH, "freed seq id must recycle")
        // a SECOND concurrent seq gets a DIFFERENT id (recycle never aliases
        // a live seq — the guard seqs[id] == nil still rules)
        let h3 = try cache.allocate(promptTokens: 256)
        XCTAssertNotEqual(h3.seq, h2.seq)
        cache.finish(h2); cache.finish(h3)
        XCTAssertEqual(cache.freePageCount, total)
        // churn: 3× maxSeqs cycles — pages exact at every step, ids recycle,
        // no "seq id ≥ maxSeqs" throw ever fires
        for _ in 0..<(QWKVCache.maxSeqs * 3) {
            let hh = try cache.allocate(promptTokens: 300)
            try cache.grow(hh, to: 600)                    // exercise grow on a recycled id too
            cache.finish(hh)
            XCTAssertEqual(cache.freePageCount, total, "page leak mid-churn")
        }
        // finish is idempotent (double-finish = no-op, no negative accounting)
        let hd = try cache.allocate(promptTokens: 256)
        cache.finish(hd); cache.finish(hd)
        XCTAssertEqual(cache.freePageCount, total)
    }

    /// finish() zeroes the released seq's DEVICE table slots: a recycled id
    /// must never inherit a stale binding (uploadSlot's tail-zero invariant,
    /// generalized to the whole slot).
    func testFinishZeroesDeviceTableSlots() throws {
        let (cache, _) = try makeCache(tokensCeiling: 2048)
        let h = try cache.allocate(promptTokens: 600)
        let l = Self.layers[0]
        let ids = cache.pageIDs(h, layer: l, kind: .k)
        XCTAssertFalse(ids.isEmpty)
        // read the device slot bytes directly (shared storage → contents())
        let views = try cache.layerViews(h, layer: l)
        let slotPtr = views.k.table.region.buffer.contents()
            .advanced(by: views.k.table.region.base)
            .assumingMemoryBound(to: Int32.self)
        XCTAssertEqual(Array(UnsafeBufferPointer(start: slotPtr, count: ids.count)), ids,
                       "device slot must mirror the table before finish")
        cache.finish(h)
        // the slot for the (now freed) seq id must be ALL ZERO (count = full 4 KiB)
        let full = QWKVCache.tableSlotBytes / 4
        var nonzero = 0
        for i in 0..<full where slotPtr[i] != 0 { nonzero += 1 }
        XCTAssertEqual(nonzero, 0, "finish must zero the whole device slot (stale-binding law)")
    }

    /// Verifies that zero-copy shared prefix caching, COW page splitting, private
    /// prefix capture/adoption, and plane serialization work seamlessly across
    /// all KV quantization modes (f16, q8, q4) with zero memory/page leaks.
    func testSharedAndPrivatePrefixAcrossAllKVModes() throws {
        for mode in QWKVMode.allCases {
            let (cache, _) = try makeCache(tokensCeiling: 2048, mode: mode)
            let total = cache.totalPages
            let count = 257
            let donor = try cache.allocate(promptTokens: count)
            let original = Self.kvSlice(from: 0, count: count)
            for layer in Self.layers {
                for kind in QWKVKind.allCases {
                    try cache.append(donor, positions: 0..<count, layer: layer, kind: kind, src: original)
                }
            }

            // 1. Capture Shared Prefix
            let shared = try cache.captureSharedPrefix(donor)
            XCTAssertEqual(shared.mode, mode)
            XCTAssertEqual(shared.tokenCount, count)
            let prefixPages = (count + QWKVCache.pageTokens - 1) / QWKVCache.pageTokens
            XCTAssertEqual(shared.pageCount, prefixPages * Self.layers.count * 2)
            XCTAssertEqual(shared.memoryBytes, shared.pageCount * cache.bytesPerPage)

            // 2. Allocate Adopting
            let adopter1 = try cache.allocate(adopting: shared, promptTokens: count + 1)
            let adopter2 = try cache.allocate(adopting: shared, promptTokens: count + 1)
            for layer in Self.layers {
                for kind in QWKVKind.allCases {
                    let donorIDs = cache.pageIDs(donor, layer: layer, kind: kind)
                    let a1IDs = cache.pageIDs(adopter1, layer: layer, kind: kind)
                    let a2IDs = cache.pageIDs(adopter2, layer: layer, kind: kind)
                    XCTAssertEqual(Array(a1IDs.prefix(prefixPages)), Array(donorIDs.prefix(prefixPages)))
                    XCTAssertEqual(Array(a2IDs.prefix(prefixPages)), Array(donorIDs.prefix(prefixPages)))
                }
            }

            // 3. COW on adopter1
            let replacement = Self.kvSlice(from: 100, count: 1)
            for layer in Self.layers {
                for kind in QWKVKind.allCases {
                    try cache.append(adopter1, positions: count..<(count + 1), layer: layer, kind: kind, src: replacement)
                    let a1IDs = cache.pageIDs(adopter1, layer: layer, kind: kind)
                    let a2IDs = cache.pageIDs(adopter2, layer: layer, kind: kind)
                    // The partial tail page was privatized on adopter1, unaffected on adopter2
                    let tailPage = (count - 1) / QWKVCache.pageTokens
                    XCTAssertNotEqual(a1IDs[tailPage], a2IDs[tailPage])
                }
            }

            // 4. Capture & Adopt Private Prefix
            let privateCost = cache.serializedPlaneBytes(tokenCount: count) * Self.layers.count * 2
            let privatePrefix = try cache.capturePrivatePrefix(donor, maxBytes: privateCost)
            XCTAssertEqual(privatePrefix.mode, mode)
            XCTAssertEqual(privatePrefix.tokenCount, count)
            XCTAssertEqual(privatePrefix.memoryBytes, privateCost)

            let privateAdopter = try cache.allocate(promptTokens: count)
            try cache.adoptPrivatePrefix(privatePrefix, into: privateAdopter)
            XCTAssertEqual(cache.written(privateAdopter), count)

            // 5. Export shared prefix planes and reconstruct private prefix
            let exportedPlanes = cache.exportSharedPrefixPlanes(shared)
            XCTAssertEqual(exportedPlanes.count, Self.layers.count * 2)
            let reconstructed = cache.makePrivatePrefix(tokenCount: count, planes: exportedPlanes)
            XCTAssertEqual(reconstructed.mode, mode)
            XCTAssertEqual(reconstructed.tokenCount, count)

            // 6. Cleanup and verify page leaks = 0
            cache.finish(donor)
            cache.finish(adopter1)
            cache.finish(adopter2)
            cache.finish(privateAdopter)
            cache.releaseSharedPrefix(shared)
            XCTAssertEqual(cache.freePageCount, total, "mode \(mode) must return all pages with 0 leak")
        }
    }
}
