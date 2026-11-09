import XCTest
@testable import QWEngine

final class QWEnginePrefixStoreTests: XCTestCase {
    func testExactPolicyLongestEligiblePrefixAndLRU() throws {
        let store = try QWEnginePrefixStore<Int>(byteLimit: 128, entryLimit: 3)
        XCTAssertTrue(try store.insert(tokens: [11], mode: "a", value: 1, payloadBytes: 8, allowsSuffix: true))
        XCTAssertTrue(try store.insert(tokens: [11, 22], mode: "a", value: 2, payloadBytes: 8, allowsSuffix: true))
        XCTAssertTrue(try store.insert(tokens: [11, 22, 33], mode: "a", value: 3, payloadBytes: 8))
        XCTAssertEqual(try store.lookup(tokens: [11, 22, 33, 44], mode: "a")?.value, 2)
        XCTAssertNil(try store.lookup(tokens: [11, 22], mode: "b"))
        XCTAssertNil(try store.lookup(tokens: [12, 22], mode: "a"))
        XCTAssertEqual(try store.lookup(tokens: [11, 22, 33], mode: "a")?.value, 3)
        XCTAssertTrue(try store.insert(tokens: [44], mode: "a", value: 4, payloadBytes: 8))
        XCTAssertNil(try store.lookup(tokens: [11], mode: "a"), "least recent exact entry was evicted")
        XCTAssertEqual(store.evictions, 1)
        XCTAssertLessThanOrEqual(store.chargedBytes, store.byteLimit)
    }

    func testChargeReplacementOversizeAndRefusalHaveNoEvictionSideEffects() throws {
        let store = try QWEnginePrefixStore<Int>(byteLimit: 32)
        XCTAssertTrue(try store.insert(tokens: [11], mode: "a", value: 1, payloadBytes: 8))
        XCTAssertEqual(store.chargedBytes, 17)
        XCTAssertTrue(try store.insert(tokens: [11], mode: "a", value: 2, payloadBytes: 16))
        XCTAssertEqual(store.chargedBytes, 25)
        XCTAssertFalse(try store.insert(tokens: [22], mode: "a", value: 3, payloadBytes: Int.max))
        XCTAssertThrowsError(try store.insert(tokens: [-1], mode: "a", value: 3, payloadBytes: 0))
        XCTAssertThrowsError(try store.insert(tokens: [22], mode: "a", value: 3, payloadBytes: -1))
        XCTAssertThrowsError(try store.lookup(tokens: [], mode: "a"))
        XCTAssertEqual(store.chargedBytes, 25)
        XCTAssertEqual(store.count, 1)
        XCTAssertEqual(store.evictions, 0)
        XCTAssertEqual(try store.lookup(tokens: [11], mode: "a")?.value, 2)
        XCTAssertTrue(store.evictOldest())
        XCTAssertEqual(store.chargedBytes, 0)
        XCTAssertFalse(store.evictOldest())
        XCTAssertThrowsError(try QWEnginePrefixStore<Int>(byteLimit: 0))
    }

    func testPolicyBytesAreChargedAndOversizedReplacementPreservesEntry() throws {
        let store = try QWEnginePrefixStore<Int>(byteLimit: 16)
        XCTAssertTrue(try store.insert(tokens: [11], mode: "policy", value: 1, payloadBytes: 2))
        XCTAssertEqual(store.chargedBytes, 16)
        XCTAssertEqual(try store.keyBytes(tokens: [11], mode: "policy"), 14)
        XCTAssertFalse(try store.insert(tokens: [11], mode: "policy", value: 2, payloadBytes: 3))
        XCTAssertFalse(try store.insert(tokens: [22], mode: String(repeating: "x", count: 17),
                                       value: 3, payloadBytes: 0))
        XCTAssertEqual(try store.lookup(tokens: [11], mode: "policy")?.value, 1)
        XCTAssertEqual(store.chargedBytes, 16)
        XCTAssertEqual(store.evictions, 0)
    }

    func testIncrementalAgentPromptsMatchLongestPrefix() throws {
        let store = try QWEnginePrefixStore<String>(byteLimit: 1024, entryLimit: 10)
        let mode = "chunked:nax_x64"

        // Turn 1: System prompt + User request 1
        let turn1Tokens = Array(1...50)
        XCTAssertNil(try store.lookup(tokens: turn1Tokens, mode: mode))
        XCTAssertTrue(try store.insert(tokens: turn1Tokens, mode: mode, value: "turn1_state", payloadBytes: 64, allowsSuffix: true))

        // Turn 2: Turn 1 + Assistant response 1 + User request 2
        let turn2Tokens = Array(1...90)
        let hitTurn2 = try store.lookup(tokens: turn2Tokens, mode: mode)
        XCTAssertNotNil(hitTurn2)
        XCTAssertEqual(hitTurn2?.tokenCount, 50, "Turn 2 must match Turn 1 prefix of 50 tokens")
        XCTAssertEqual(hitTurn2?.value, "turn1_state")
        XCTAssertTrue(try store.insert(tokens: turn2Tokens, mode: mode, value: "turn2_state", payloadBytes: 96, allowsSuffix: true))

        // Turn 3: Turn 2 + Assistant response 2 + User request 3
        let turn3Tokens = Array(1...140)
        let hitTurn3 = try store.lookup(tokens: turn3Tokens, mode: mode)
        XCTAssertNotNil(hitTurn3)
        XCTAssertEqual(hitTurn3?.tokenCount, 90, "Turn 3 must match Turn 2 prefix of 90 tokens (longest prefix)")
        XCTAssertEqual(hitTurn3?.value, "turn2_state")
    }
}
