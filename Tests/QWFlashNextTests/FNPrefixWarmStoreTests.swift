import XCTest
@testable import QWFlashNext

/// The warm tier of the Flash-Next prefix cache: the RAM holds the hot
/// prefixes, the disk answers the ones it does not, and both stay inside
/// the limits the settings gave.
final class FNPrefixWarmStoreTests: XCTestCase {
    /// The name the disk uses for a saved prefix.
    private func entryName(tokens: [Int64], kvMode: FNKVStorageMode = .q4,
                            hasMTP: Bool = false) -> String {
        let hash = FNGPUEngine.fnPrefixHash(tokens: tokens[...], kvMode: kvMode, hasMTP: hasMTP)
        return String(format: "fn_entry_%016llx_%d.qwcache", hash, tokens.count)
    }

    func testTheTierCountsWhatItHolds() {
        let store = FNPrefixWarmStore(byteLimit: 1024)
        store.store(name: "fn_entry_0000000000000001_4.qwcache", blob: Data(repeating: 1, count: 100))
        store.store(name: "fn_entry_0000000000000002_8.qwcache", blob: Data(repeating: 2, count: 200))
        XCTAssertEqual(store.state.entries, 2)
        XCTAssertEqual(store.state.bytes, 300)
    }

    /// The RAM is the hot tier: the oldest entry goes when the limit is
    /// reached, so the newest prefixes are the ones it keeps.
    func testTheTierDropsItsOldestEntryWhenTheLimitIsReached() {
        let store = FNPrefixWarmStore(byteLimit: 250)
        store.store(name: "fn_entry_0000000000000001_4.qwcache", blob: Data(repeating: 1, count: 100))
        store.store(name: "fn_entry_0000000000000002_8.qwcache", blob: Data(repeating: 2, count: 100))
        store.store(name: "fn_entry_0000000000000003_12.qwcache", blob: Data(repeating: 3, count: 100))
        XCTAssertEqual(store.state.bytes, 200)
        XCTAssertNil(store.blob(name: "fn_entry_0000000000000001_4.qwcache"))
        XCTAssertNotNil(store.blob(name: "fn_entry_0000000000000002_8.qwcache"))
        XCTAssertNotNil(store.blob(name: "fn_entry_0000000000000003_12.qwcache"))
    }

    /// A hit makes an entry the newest one: the RAM keeps what is hot.
    func testTheTierKeepsWhatIsUsed() {
        let store = FNPrefixWarmStore(byteLimit: 250)
        store.store(name: "fn_entry_0000000000000001_4.qwcache", blob: Data(repeating: 1, count: 100))
        store.store(name: "fn_entry_0000000000000002_8.qwcache", blob: Data(repeating: 2, count: 100))
        _ = store.blob(name: "fn_entry_0000000000000001_4.qwcache")
        store.store(name: "fn_entry_0000000000000003_12.qwcache", blob: Data(repeating: 3, count: 100))
        XCTAssertNotNil(store.blob(name: "fn_entry_0000000000000001_4.qwcache"))
        XCTAssertNil(store.blob(name: "fn_entry_0000000000000002_8.qwcache"))
    }

    /// A prefix the RAM cannot hold lives on the disk, not in a tier
    /// that would break its own limit. It must not take a prefix that
    /// does fit with it.
    func testAPrefixBiggerThanTheLimitIsNotKept() {
        let store = FNPrefixWarmStore(byteLimit: 100)
        store.store(name: "fn_entry_0000000000000001_4.qwcache", blob: Data(repeating: 1, count: 100))
        store.store(name: "fn_entry_0000000000000002_8.qwcache", blob: Data(repeating: 2, count: 101))
        XCTAssertEqual(store.state.entries, 1)
        XCTAssertEqual(store.state.bytes, 100)
        XCTAssertNil(store.blob(name: "fn_entry_0000000000000002_8.qwcache"))
        XCTAssertNotNil(store.blob(name: "fn_entry_0000000000000001_4.qwcache"))
    }

    /// A limit of zero is no tier: the RAM holds nothing at all.
    func testATierWithoutRoomIsOff() {
        let store = FNPrefixWarmStore(byteLimit: 0)
        store.store(name: "fn_entry_0000000000000001_4.qwcache", blob: Data(repeating: 1, count: 10))
        XCTAssertFalse(store.isEnabled)
        XCTAssertEqual(store.state.entries, 0)
        XCTAssertNil(store.blob(name: "fn_entry_0000000000000001_4.qwcache"))
    }

    /// The tier answers with the longest prefix of this prompt it holds,
    /// and with nothing else.
    func testTheTierFindsTheLongestPrefixOfThePrompt() {
        let store = FNPrefixWarmStore(byteLimit: 4096)
        let prompt: [Int64] = [11, 22, 33, 44, 55, 66]
        store.store(name: entryName(tokens: Array(prompt.prefix(2))), blob: Data(repeating: 1, count: 10))
        store.store(name: entryName(tokens: Array(prompt.prefix(4))), blob: Data(repeating: 2, count: 10))
        let match = store.longestMatch(history: prompt, kvMode: .q4, hasMTP: false)
        XCTAssertEqual(match?.tokens, 4)
        XCTAssertEqual(match?.name, entryName(tokens: Array(prompt.prefix(4))))
    }

    /// A prefix of another prompt is not a prefix of this one, however
    /// many tokens it holds.
    func testTheTierDoesNotAnswerForAnotherPrompt() {
        let store = FNPrefixWarmStore(byteLimit: 4096)
        let other: [Int64] = [99, 98, 97, 96]
        store.store(name: entryName(tokens: other), blob: Data(repeating: 1, count: 10))
        let match = store.longestMatch(history: [11, 22, 33, 44], kvMode: .q4, hasMTP: false)
        XCTAssertNil(match)
    }

    /// The tier names are the ones the server counts.
    func testTheTierNamesAreTheOnesTheServerCounts() {
        XCTAssertEqual(FNPrefixCacheTier.warmRAM.rawValue, "warm_ram")
        XCTAssertEqual(FNPrefixCacheTier.coldSSD.rawValue, "cold_ssd")
    }

    /// The warm limit is the switch: with no room in the RAM there is no
    /// cache at all, and the cold tier needs room of its own.
    func testTheWarmLimitDecidesWhetherTheCacheRuns() {
        XCTAssertFalse(FNPrefixCacheTiers(warmByteLimit: 0, ssdByteLimit: 1024).isEnabled)
        XCTAssertEqual(FNPrefixCacheTiers(warmByteLimit: 0, ssdByteLimit: 1024).ssdByteLimit, 0)
        XCTAssertTrue(FNPrefixCacheTiers(warmByteLimit: 1024, ssdByteLimit: 0).isEnabled)
        XCTAssertEqual(FNPrefixCacheTiers(warmByteLimit: 1024, ssdByteLimit: 0).ssdByteLimit, 0)
        XCTAssertEqual(FNPrefixCacheTiers(warmByteLimit: 1024, ssdByteLimit: 2048).ssdByteLimit, 2048)
    }

    /// The cold tier is the overflow: it stays inside the limit, and the
    /// oldest prefixes go first.
    func testTheColdTierMakesRoomForTheNewPrefix() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("fn-cache-room-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let names = ["fn_entry_0000000000000001_4.qwcache",
                     "fn_entry_0000000000000002_8.qwcache",
                     "fn_entry_0000000000000003_12.qwcache"]
        for (index, name) in names.enumerated() {
            let url = dir.appendingPathComponent(name)
            try Data(repeating: 1, count: 100).write(to: url)
            let date = Date(timeIntervalSince1970: 1_000 + TimeInterval(index))
            try FileManager.default.setAttributes([.modificationDate: date], ofItemAtPath: url.path)
        }
        // A directory that does not hold the cache's files is not counted.
        try Data(repeating: 1, count: 500).write(to: dir.appendingPathComponent("other.txt"))

        // Three prefixes of 100 bytes and a new one of 100 bytes do not
        // fit under 250 bytes: the two oldest go, the newest stays.
        FNGPUEngine.makeRoom(in: dir, for: 100, limit: 250)

        let left = try FileManager.default.contentsOfDirectory(atPath: dir.path).sorted()
        XCTAssertEqual(left, ["fn_entry_0000000000000003_12.qwcache", "other.txt"])
    }

    /// Each request slot's last entry is protected from eviction while the
    /// engine is running; older/non-last entries ("the rest") are evicted first.
    func testLastEntryPerRequestSlotIsProtectedWhileRestAreEvicted() {
        let store = FNPrefixWarmStore(byteLimit: 350)
        var evictedTokens: [Int] = []
        store.onEvict = { tokens, _ in evictedTokens.append(tokens) }

        let slot0Turn1 = (0..<160).map { Int64(1000 + $0) }
        let slot0Turn2 = slot0Turn1 + (0..<40).map { Int64(2000 + $0) }
        let slot0Root = FNPrefixWarmStore.rootHash(history: slot0Turn1, kvMode: .q8, hasMTP: true)
        let s0 = store.resolveSlot(rootHash: slot0Root, history: slot0Turn1, kvMode: .q8, hasMTP: true)

        let name0_1 = entryName(tokens: slot0Turn1, kvMode: .q8, hasMTP: true)
        store.prepareSlotForStore(slot: s0, rootHash: slot0Root, history: slot0Turn1,
                                  incomingBytes: 100, kvMode: .q8, hasMTP: true)
        store.store(name: name0_1, blob: Data(repeating: 1, count: 100), slot: s0, rootHash: slot0Root)

        // Advance Slot 0 to Turn 2: Turn 1 stays as the evictable base entry, Turn 2 becomes the protected last entry.
        let name0_2 = entryName(tokens: slot0Turn2, kvMode: .q8, hasMTP: true)
        store.prepareSlotForStore(slot: s0, rootHash: slot0Root, history: slot0Turn2,
                                  incomingBytes: 100, kvMode: .q8, hasMTP: true)
        store.store(name: name0_2, blob: Data(repeating: 2, count: 100), slot: s0, rootHash: slot0Root)
        XCTAssertTrue(store.isProtectedLastEntry(name0_2))
        XCTAssertFalse(store.isProtectedLastEntry(name0_1))
        XCTAssertEqual(evictedTokens, [], "In-place slot progression must not fire false eviction callbacks")

        // Touch Turn 1 so it is MORE recently used than Turn 2 in raw LRU order,
        // verifying that slot-last protection overrides raw LRU and keeps Turn 2!
        _ = store.blob(name: name0_1)

        // Now Slot 1 arrives needing 200 bytes (total would be 400 > 350 limit).
        // Slot 0's non-last entry (name0_1) must be evicted, while Slot 0's last entry (name0_2) is preserved.
        let slot1Turn1 = (0..<180).map { Int64(5000 + $0) }
        let slot1Root = FNPrefixWarmStore.rootHash(history: slot1Turn1, kvMode: .q8, hasMTP: true)
        let s1 = store.resolveSlot(rootHash: slot1Root, history: slot1Turn1, kvMode: .q8, hasMTP: true)
        XCTAssertNotEqual(s0, s1)
        let name1_1 = entryName(tokens: slot1Turn1, kvMode: .q8, hasMTP: true)
        store.prepareSlotForStore(slot: s1, rootHash: slot1Root, history: slot1Turn1,
                                  incomingBytes: 200, kvMode: .q8, hasMTP: true)
        store.store(name: name1_1, blob: Data(repeating: 3, count: 200), slot: s1, rootHash: slot1Root)

        XCTAssertNil(store.blob(name: name0_1), "Older non-last entry must be evicted first")
        XCTAssertNotNil(store.blob(name: name0_2), "Slot 0 last entry must not be evicted")
        XCTAssertNotNil(store.blob(name: name1_1), "Slot 1 last entry must be stored")
    }

    /// Five concurrent sessions (1 main agent + 4 subagents) sharing an identical 1,024-token
    /// system prompt must receive 5 distinct slots (0..<5) instead of colliding on Slot 0,
    /// with zero evictions and 100% cache hit rate on subsequent turns.
    func testConcurrentSubagentsSharingSystemPromptGetDistinctSlotsAndZeroEvictions() {
        let store = FNPrefixWarmStore(byteLimit: 10_000, maxSlots: 8)
        var evictedCount = 0
        store.onEvict = { _, _ in evictedCount += 1 }

        let sharedSystemPrompt: [Int64] = (0..<1024).map { Int64(100 + $0) }
        let root = FNPrefixWarmStore.rootHash(history: sharedSystemPrompt, kvMode: .f16, hasMTP: true)

        var assignedSlots: [Int] = []
        var turn1Histories: [[Int64]] = []
        for agentIdx in 0..<5 {
            let userTurn: [Int64] = (0..<128).map { Int64(10_000 * (agentIdx + 1) + $0) }
            let h1 = sharedSystemPrompt + userTurn
            turn1Histories.append(h1)

            let slot = store.resolveSlot(rootHash: root, history: h1, kvMode: .f16, hasMTP: true)
            assignedSlots.append(slot)
            let name1 = entryName(tokens: h1, kvMode: .f16, hasMTP: true)
            store.prepareSlotForStore(slot: slot, rootHash: root, history: h1,
                                      incomingBytes: 500, kvMode: .f16, hasMTP: true)
            store.store(name: name1, blob: Data(repeating: UInt8(agentIdx + 1), count: 500),
                        slot: slot, rootHash: root, history: h1)
        }

        XCTAssertEqual(Set(assignedSlots).count, 5, "All 5 concurrent sessions sharing a system prompt must get distinct slots")
        XCTAssertEqual(assignedSlots, [0, 1, 2, 3, 4])
        XCTAssertEqual(evictedCount, 0, "No evictions may occur when 5 concurrent sessions populate their Turn 1 slots")

        // Turn 2 for all 5 concurrent sessions: each must hit its own Turn 1 prefix and stay in its assigned slot.
        for agentIdx in 0..<5 {
            let h1 = turn1Histories[agentIdx]
            let turn2Delta: [Int64] = (0..<96).map { Int64(90_000 * (agentIdx + 1) + $0) }
            let h2 = h1 + turn2Delta

            let match = store.longestMatch(history: h2, kvMode: .f16, hasMTP: true)
            XCTAssertEqual(match?.tokens, h1.count, "Session \(agentIdx) Turn 2 must match its full Turn 1 checkpoint")

            let slot2 = store.resolveSlot(rootHash: root, history: h2, kvMode: .f16, hasMTP: true)
            XCTAssertEqual(slot2, assignedSlots[agentIdx], "Session \(agentIdx) Turn 2 must resolve to its existing slot")

            let name2 = entryName(tokens: h2, kvMode: .f16, hasMTP: true)
            store.prepareSlotForStore(slot: slot2, rootHash: root, history: h2,
                                      incomingBytes: 600, kvMode: .f16, hasMTP: true)
            store.store(name: name2, blob: Data(repeating: UInt8(10 + agentIdx), count: 600),
                        slot: slot2, rootHash: root, history: h2)
        }
        XCTAssertEqual(evictedCount, 0)
    }

    /// A shared base checkpoint (`isSharedBase: true`) for `system + tools` is preserved across
    /// multi-turn slot progressions so newly spawned subagents always hit the base prefix on Turn 1.
    func testSharedBaseCheckpointIsProtectedAndServesBranchingSubagents() {
        let store = FNPrefixWarmStore(byteLimit: 10_000, maxSlots: 8)
        let sharedBase: [Int64] = (0..<512).map { Int64(500 + $0) }
        let baseName = entryName(tokens: sharedBase, kvMode: .f16, hasMTP: true)
        store.store(name: baseName, blob: Data(repeating: 0xAA, count: 400),
                    slot: nil, rootHash: nil, history: sharedBase, isSharedBase: true)
        XCTAssertTrue(store.isSharedBaseEntry(baseName))
        XCTAssertTrue(store.contains(name: baseName))

        let root = FNPrefixWarmStore.rootHash(history: sharedBase, kvMode: .f16, hasMTP: true)
        for agentIdx in 0..<5 {
            let t1 = sharedBase + (0..<100).map { Int64(20_000 * (agentIdx + 1) + $0) }
            // Turn 1 of every subagent must hit the 512-token shared base checkpoint!
            let baseMatch = store.longestMatch(history: t1, kvMode: .f16, hasMTP: true)
            XCTAssertEqual(baseMatch?.tokens, sharedBase.count)
            XCTAssertEqual(baseMatch?.name, baseName)

            let s = store.resolveSlot(rootHash: root, history: t1, kvMode: .f16, hasMTP: true)
            let n1 = entryName(tokens: t1, kvMode: .f16, hasMTP: true)
            store.prepareSlotForStore(slot: s, rootHash: root, history: t1,
                                      incomingBytes: 500, kvMode: .f16, hasMTP: true)
            store.store(name: n1, blob: Data(repeating: 1, count: 500), slot: s, rootHash: root, history: t1)

            // Advance each subagent through Turn 2 and Turn 3 (which evicts superseded per-slot intermediates).
            let t2 = t1 + (0..<80).map { Int64(30_000 * (agentIdx + 1) + $0) }
            let n2 = entryName(tokens: t2, kvMode: .f16, hasMTP: true)
            store.prepareSlotForStore(slot: s, rootHash: root, history: t2,
                                      incomingBytes: 550, kvMode: .f16, hasMTP: true)
            store.store(name: n2, blob: Data(repeating: 2, count: 550), slot: s, rootHash: root, history: t2)

            let t3 = t2 + (0..<80).map { Int64(40_000 * (agentIdx + 1) + $0) }
            let n3 = entryName(tokens: t3, kvMode: .f16, hasMTP: true)
            store.prepareSlotForStore(slot: s, rootHash: root, history: t3,
                                      incomingBytes: 600, kvMode: .f16, hasMTP: true)
            store.store(name: n3, blob: Data(repeating: 3, count: 600), slot: s, rootHash: root, history: t3)
        }

        // A 6th subagent spawning later must still find the shared base checkpoint intact.
        let subagent6Turn1 = sharedBase + (0..<120).map { Int64(99_000 + $0) }
        let match6 = store.longestMatch(history: subagent6Turn1, kvMode: .f16, hasMTP: true)
        XCTAssertEqual(match6?.tokens, sharedBase.count)
        XCTAssertEqual(match6?.name, baseName)
    }
}
