// QWInkling2PrefixCacheStoreTests.swift — Tests QWSlotPinnedLRUStore-backed QWInkling2PrefixCache behaviors.
import Foundation
import Metal
import XCTest
@testable import QWEngine

final class QWInkling2PrefixCacheStoreTests: XCTestCase {
    private func makeEntry(
        tokens: [Int],
        salt: String? = nil,
        finalNormRow: Data? = nil,
        dflashState: Data? = nil,
        convData: Data = Data(repeating: 1, count: 16),
        bytes: Int = 64
    ) -> QWInkling2PrefixCache.Entry {
        QWInkling2PrefixCache.Entry(
            key: QWInkling2PrefixCache.Key(salt: salt, tokens: tokens),
            convData: convData,
            visitSlices: [],
            dflashState: dflashState,
            finalNormRow: finalNormRow,
            byteCount: bytes + (finalNormRow?.count ?? 0) + (dflashState?.count ?? 0)
        )
    }

    func testCallbackOrderingOnEvictBeforeOnRetainAndSkipOversize() {
        let cache = QWInkling2PrefixCache(warmByteLimit: 150, maxEntries: 4)
        var events: [String] = []
        cache.onEvict = { tokens, freed in
            events.append("evict:\(tokens):\(freed)")
        }
        cache.onRetain = { tokens, charged, total, limit in
            events.append("retain:\(tokens):\(charged):\(total):\(limit)")
        }
        cache.onSkipOversize = { tokens, charge, limit in
            events.append("skip:\(tokens):\(charge):\(limit)")
        }

        cache.store(entry: makeEntry(tokens: [1, 2], bytes: 64))
        cache.store(entry: makeEntry(tokens: [3, 4, 5], bytes: 64))
        cache.store(entry: makeEntry(tokens: [6, 7, 8, 9], bytes: 64))
        cache.store(entry: makeEntry(tokens: [10, 11], bytes: 200))

        XCTAssertEqual(events, [
            "retain:2:64:64:150",
            "retain:3:64:128:150",
            "evict:2:64",
            "retain:4:64:128:150",
            "skip:2:200:150"
        ])
    }

    func testExactMatchWithoutFinalNormRowDoesNotTouchLRUWhenFallingBack() {
        let cache = QWInkling2PrefixCache(warmByteLimit: 256, maxEntries: 2)
        let prefix = makeEntry(tokens: [1, 2], finalNormRow: Data(repeating: 5, count: 8), bytes: 64)
        let exactNoNorm = makeEntry(tokens: [1, 2, 3], finalNormRow: nil, bytes: 64)
        cache.store(entry: prefix)
        cache.store(entry: exactNoNorm)

        // Exact hit has finalNormRow == nil -> falls back to [1, 2] and touches [1, 2], leaving [1, 2, 3] oldest.
        let hit = cache.matchLongestPrefix(tokens: [1, 2, 3], salt: nil, limit: 3, allowExact: true)
        XCTAssertEqual(hit?.key.tokens, [1, 2])

        // Storing a third entry with maxEntries=2 must evict [1, 2, 3], not [1, 2].
        var evictedTokens: [Int] = []
        cache.onEvict = { tokens, _ in evictedTokens.append(tokens) }
        cache.store(entry: makeEntry(tokens: [9, 10, 11, 12], finalNormRow: Data(repeating: 3, count: 8), bytes: 64))

        XCTAssertEqual(evictedTokens, [3])
        XCTAssertNotNil(cache.matchLongestPrefix(tokens: [1, 2], salt: nil, limit: 2, allowExact: true))
    }

    func testReconfigureShrinkGrowZeroPurgeAndClear() {
        let cache = QWInkling2PrefixCache(warmByteLimit: 400, maxEntries: 4)
        var evicted: [Int] = []
        cache.onEvict = { tokens, _ in evicted.append(tokens) }

        cache.store(entry: makeEntry(tokens: [1], bytes: 100))
        cache.store(entry: makeEntry(tokens: [2, 3], bytes: 100))
        cache.store(entry: makeEntry(tokens: [4, 5, 6], bytes: 100))
        XCTAssertEqual(cache.state?.entries, 3)
        XCTAssertEqual(cache.state?.bytes, 300)
        XCTAssertEqual(cache.state?.limit, 400)

        cache.reconfigure(warmByteLimit: 200, maxEntries: 2)
        XCTAssertEqual(evicted, [1])
        XCTAssertEqual(cache.state?.entries, 2)
        XCTAssertEqual(cache.state?.bytes, 200)
        XCTAssertEqual(cache.state?.limit, 200)

        cache.reconfigure(warmByteLimit: 500, maxEntries: 6)
        cache.store(entry: makeEntry(tokens: [7, 8, 9, 10], bytes: 250))
        XCTAssertEqual(cache.state?.entries, 3)
        XCTAssertEqual(cache.state?.bytes, 450)
        XCTAssertEqual(cache.state?.limit, 500)

        cache.reconfigure(warmByteLimit: 0, maxEntries: 6)
        XCTAssertFalse(cache.isEnabled)
        XCTAssertNil(cache.state)
        XCTAssertEqual(evicted, [1, 2, 3, 4])

        cache.reconfigure(warmByteLimit: 500, maxEntries: 6)
        cache.store(entry: makeEntry(tokens: [11, 12], finalNormRow: Data(repeating: 1, count: 4), bytes: 64))
        XCTAssertEqual(cache.state?.entries, 1)

        evicted.removeAll()
        cache.clear()
        XCTAssertEqual(evicted, [])
        XCTAssertEqual(cache.state?.entries, 0)
        XCTAssertEqual(cache.state?.bytes, 0)
        XCTAssertNil(cache.matchLongestPrefix(tokens: [11, 12], salt: nil, limit: 2, allowExact: true))
    }

    func testDFlashStateRoundTripThroughCacheAndRestore() throws {
        let cache = QWInkling2PrefixCache(warmByteLimit: 4096, maxEntries: 4)
        let dflashBlob = Data((0..<48).map { UInt8($0 & 0xFF) })
        let convBytes = Data((0..<32).map { UInt8(($0 + 7) & 0xFF) })
        let normRow = Data([0xAA, 0xBB, 0xCC, 0xDD])
        let entry = makeEntry(
            tokens: [10, 20, 30, 40],
            salt: "sys-a",
            finalNormRow: normRow,
            dflashState: dflashBlob,
            convData: convBytes,
            bytes: 32
        )
        cache.store(entry: entry)

        let matched = cache.matchLongestPrefix(tokens: [10, 20, 30, 40, 50], salt: "sys-a", limit: 5, allowExact: false)
        XCTAssertEqual(matched?.dflashState, dflashBlob)
        XCTAssertEqual(matched?.finalNormRow, normRow)
        XCTAssertEqual(matched?.convData, convBytes)
        XCTAssertNil(cache.matchLongestPrefix(tokens: [10, 20, 30, 40], salt: "other", limit: 4, allowExact: true))

        guard let device = MTLCreateSystemDefaultDevice(),
              let slab = device.makeBuffer(length: convBytes.count, options: .storageModeShared),
              let restoredEntry = matched else {
            return
        }
        let slot = Ink2SlotState(
            slotIndex: 0, capacity: 16, visitKVs: [],
            convStatesSlab: slab, residentStateBytes: convBytes.count, stateBuffers: [slab]
        )
        let outcome = QWInkling2PrefixCache.restore(entry: restoredEntry, into: slot, kvModeRaw: 0)
        XCTAssertTrue(outcome.baseOk)
        XCTAssertFalse(outcome.dflashOk)
        let restoredConv = Data(bytes: slab.contents(), count: convBytes.count)
        XCTAssertEqual(restoredConv, convBytes)
    }
}
