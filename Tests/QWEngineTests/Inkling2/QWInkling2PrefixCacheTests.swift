// QWInkling2PrefixCacheTests.swift — Tests exact-prompt hits, finalNormRow gating, and entry limits.
import Foundation
import XCTest
@testable import QWEngine

final class QWInkling2PrefixCacheTests: XCTestCase {
    private func makeEntry(tokens: [Int], salt: String? = nil, finalNormRow: Data? = nil, bytes: Int = 64) -> QWInkling2PrefixCache.Entry {
        QWInkling2PrefixCache.Entry(
            key: QWInkling2PrefixCache.Key(salt: salt, tokens: tokens),
            convData: Data(repeating: 1, count: 16),
            visitSlices: [],
            finalNormRow: finalNormRow,
            byteCount: bytes + (finalNormRow?.count ?? 0),
            tick: 0
        )
    }

    func testExactMatchRequiresFinalNormRowAndFallsBackToPrefix() {
        let cache = QWInkling2PrefixCache(warmByteLimit: 1024 * 1024, maxEntries: 32)
        let prefixEntry = makeEntry(tokens: [1, 2, 3], finalNormRow: Data(repeating: 7, count: 8))
        let noNormExact = makeEntry(tokens: [1, 2, 3, 4, 5], finalNormRow: nil)
        cache.store(entry: prefixEntry)
        cache.store(entry: noNormExact)

        let fallback = cache.matchLongestPrefix(tokens: [1, 2, 3, 4, 5], salt: nil, limit: 5, allowExact: true)
        XCTAssertEqual(fallback?.key.tokens, [1, 2, 3])

        let withNormExact = makeEntry(tokens: [1, 2, 3, 4, 5], finalNormRow: Data(repeating: 9, count: 8))
        cache.store(entry: withNormExact)

        let exact = cache.matchLongestPrefix(tokens: [1, 2, 3, 4, 5], salt: nil, limit: 5, allowExact: true)
        XCTAssertEqual(exact?.key.tokens, [1, 2, 3, 4, 5])
        XCTAssertEqual(exact?.finalNormRow, Data(repeating: 9, count: 8))

        let grown = cache.matchLongestPrefix(tokens: [1, 2, 3, 4, 5, 6], salt: nil, limit: 6, allowExact: true)
        XCTAssertEqual(grown?.key.tokens, [1, 2, 3, 4, 5])

        let disallowedExact = cache.matchLongestPrefix(tokens: [1, 2, 3, 4, 5], salt: nil, limit: 4, allowExact: true)
        XCTAssertEqual(disallowedExact?.key.tokens, [1, 2, 3])
    }

    func testReconfigureRetainsMoreThanEightEntriesUpToConfiguredLimit() {
        let cache = QWInkling2PrefixCache(warmByteLimit: 1024 * 1024, maxEntries: 8)
        cache.reconfigure(warmByteLimit: 1024 * 1024, maxEntries: 32)
        for i in 1...16 {
            cache.store(entry: makeEntry(tokens: [100 + i, 200 + i], finalNormRow: Data(repeating: 2, count: 8)))
        }
        XCTAssertEqual(cache.state?.entries, 16)
        XCTAssertEqual(cache.matchLongestPrefix(tokens: [101, 201], salt: nil, limit: 2, allowExact: true)?.key.tokens, [101, 201])
    }
}
