import XCTest
@testable import QWEngine
@testable import QWKV
import QWHAL
import Foundation

final class QWTwoTierCacheTests: XCTestCase {
    func testCacheIOReportsFailuresAndPreservesBestEffortResults() {
        final class Events: @unchecked Sendable {
            var values: [QWPrefixCacheEvent] = []
        }
        enum Failure: Error { case unavailable }
        let events = Events()
        let onEvent: @Sendable (QWPrefixCacheEvent) -> Void = { events.values.append($0) }
        for operation in [QWPrefixCacheIOOperation.write, .restore] {
            let result: Int? = QWCacheIO.attempt(operation, onEvent: onEvent) {
                throw Failure.unavailable
            }
            XCTAssertNil(result)
        }
        XCTAssertEqual(events.values.count, 2)
        if case .ioFailure(let operation, let message) = events.values[0] {
            XCTAssertEqual(operation, .write)
            XCTAssertTrue(message.contains("unavailable"))
        } else {
            XCTFail("expected write failure")
        }
        if case .ioFailure(let operation, _) = events.values[1] {
            XCTAssertEqual(operation, .restore)
        } else {
            XCTFail("expected restore failure")
        }
        let value = QWCacheIO.attempt(.restore, onEvent: onEvent) { 42 }
        XCTAssertEqual(value, 42)
        XCTAssertEqual(events.values.count, 2)
    }

    func testTruncatedDiskReadIsReportedAsRestoreFailure() throws {
        final class Events: @unchecked Sendable {
            var value: QWPrefixCacheEvent?
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try Data([1]).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let io = try QWDispatchIO(path: url.path, mode: .readOnly, directIO: false)
        defer { io.close() }
        let events = Events()
        var buffer = [UInt8](repeating: 0, count: 8)
        let result: Void? = QWCacheIO.attempt(.restore, onEvent: { events.value = $0 }) {
            try buffer.withUnsafeMutableBytes { raw in
                try QWCacheDiskManager.readExactly(io, offset: 0, into: raw.baseAddress!, length: raw.count)
            }
        }
        XCTAssertNil(result)
        guard case .ioFailure(.restore, let message) = events.value else {
            return XCTFail("expected a restore failure event")
        }
        XCTAssertTrue(message.contains("short cache read"))
    }

    func testCacheConfigDefaults() {
        let config = QWCacheConfig()
        XCTAssertEqual(config.warmByteLimit, 4 * 1024 * 1024 * 1024)
        XCTAssertEqual(config.ssdByteLimit, 10 * 1024 * 1024 * 1024)
        XCTAssertTrue(config.cacheDirectory.path.contains(".qwg/kv-cache"))
        XCTAssertTrue(config.wipeOnStart)
        XCTAssertEqual(config.entryLimit, 32)
    }

    func testDiskManagerWriteLookupReadEvictLifecycle() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw_test_cache_\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Set 64 KB limit so eviction triggers after 1-2 entries
        let config = QWCacheConfig(
            warmByteLimit: 1024 * 1024,
            ssdByteLimit: 64 * 1024,
            entryLimit: 10,
            cacheDirectory: tempDir,
            wipeOnStart: true
        )
        let diskManager = QWCacheDiskManager(config: config)

        // 1. Prepare dummy prefix state
        let roundRecord = QWRoundRecord(
            pendingToken: 42,
            consumedCount: 16,
            outputCount: 1,
            trunkHiddenSeed: nil,
            mtpPhysicalRows: 0,
            absolutePosition: 16,
            samplerDrawIndex: 0
        )
        let recData = Data(repeating: 1, count: 128)
        let ringData = Data(repeating: 2, count: 64)
        let dummyState = QWEnginePrefixState(
            position: 16,
            generated: 1,
            roundRecord: roundRecord,
            recData: recData,
            ringData: ringData,
            trunkHiddenData: nil,
            trunkHiddenValid: false,
            stochastic: nil
        )

        let tokens1 = [101, 102, 103, 104]
        let logits1 = Data(repeating: 7, count: 256)
        let plane1 = Data(repeating: 0xAA, count: 1024)

        // 2. Write entry 1
        let entry1File = try diskManager.write(
            tokens: tokens1,
            mode: "test_mode",
            nextToken: 105,
            state: dummyState,
            logits: logits1,
            trunkPlanes: [plane1],
            mtpPlanes: [],
            allowsSuffix: true
        )
        XCTAssertNotNil(entry1File)
        XCTAssertTrue(FileManager.default.fileExists(atPath: entry1File!.path))
        XCTAssertEqual(diskManager.entryCount, 1)
        XCTAssertGreaterThan(diskManager.chargedBytes, 0)

        // 3. Lookup entry 1 (exact match)
        let hit1 = diskManager.lookup(tokens: tokens1, mode: "test_mode")
        XCTAssertNotNil(hit1)
        XCTAssertEqual(hit1?.matchedTokens, 4)
        XCTAssertEqual(hit1?.metadata.nextToken, 105)

        // 4. Lookup entry 1 with longer prompt (suffix match)
        let hitPrefix = diskManager.lookup(tokens: [101, 102, 103, 104, 105, 106], mode: "test_mode")
        XCTAssertNotNil(hitPrefix)
        XCTAssertEqual(hitPrefix?.matchedTokens, 4)

        // 5. Read and deserialize entry 1
        let payload1 = try diskManager.read(hit: hit1!)
        XCTAssertEqual(payload1.nextToken, 105)
        XCTAssertEqual(payload1.state.position, 16)
        XCTAssertEqual(payload1.state.roundRecord.pendingToken, 42)
        XCTAssertEqual(payload1.logits, logits1)
        XCTAssertEqual(payload1.trunkPlanes.count, 1)
        XCTAssertEqual(payload1.trunkPlanes[0], plane1)

        let damaged = try FileHandle(forWritingTo: entry1File!)
        try damaged.seek(toOffset: 8)
        try damaged.write(contentsOf: Data("[".utf8))
        try damaged.close()
        XCTAssertThrowsError(try diskManager.read(hit: hit1!))

        // 6. Test LRU Eviction by writing large entries exceeding the 64 KB limit
        let bigPlane = Data(repeating: 0xBB, count: 40 * 1024)
        let tokens2 = [201, 202, 203]
        _ = try diskManager.write(
            tokens: tokens2,
            mode: "test_mode",
            nextToken: 204,
            state: dummyState,
            logits: logits1,
            trunkPlanes: [bigPlane],
            mtpPlanes: [],
            allowsSuffix: false
        )

        let tokens3 = [301, 302, 303]
        _ = try diskManager.write(
            tokens: tokens3,
            mode: "test_mode",
            nextToken: 304,
            state: dummyState,
            logits: logits1,
            trunkPlanes: [bigPlane],
            mtpPlanes: [],
            allowsSuffix: false
        )

        // Entry 1 should have been evicted due to LRU disk budget pressure
        let evictedHit = diskManager.lookup(tokens: tokens1, mode: "test_mode")
        XCTAssertNil(evictedHit, "Oldest entry 1 must be evicted to satisfy SSD byte limit")

        // 7. Test Startup Wipe
        let newManager = QWCacheDiskManager(config: config)
        XCTAssertEqual(newManager.entryCount, 0, "Wipe on start must clean existing files")
        let wipedHit = newManager.lookup(tokens: tokens3, mode: "test_mode")
        XCTAssertNil(wipedHit)
    }

    func testStandaloneQWCacheRoundTripAndSharedBaseStoreLifecycle() throws {
        let store = try QWEnginePrefixStore<Int>(byteLimit: 1024, entryLimit: 8, vocab: 1000, maxSlots: 4)
        XCTAssertTrue(try store.insert(tokens: [1, 2, 3, 4], mode: "chunked", value: 11, payloadBytes: 128,
                                       allowsSuffix: true, isSharedBase: true))
        store.ensureByteLimit(atLeast: 4096)
        XCTAssertEqual(store.byteLimit, 4096)
        XCTAssertNotNil(try store.lookup(tokens: [1, 2, 3, 4, 5], mode: "chunked"))
        store.removeSharedBaseEntries()
        XCTAssertNil(try store.lookup(tokens: [1, 2, 3, 4, 5], mode: "chunked"))
    }

    func testSharedBasePinningAndMultiTurnSlotAdvancement() throws {
        let store = try QWEnginePrefixStore<Int>(byteLimit: 4000, entryLimit: 8, vocab: 1000, maxSlots: 4)
        let baseTokens = Array(1...64)
        XCTAssertTrue(try store.insert(tokens: baseTokens, mode: "chunked", value: 100, payloadBytes: 2500,
                                       allowsSuffix: true, isSharedBase: true))
        XCTAssertTrue(store.hasSharedBase)
        XCTAssertEqual(store.sharedBasePrefix(matching: baseTokens + [5, 6], mode: "chunked")?.value, 100)

        // Attempting to insert an entry that would require evicting the pinned shared base must fail
        // rather than evicting the mounted .qwcache base.
        XCTAssertFalse(try store.insert(tokens: baseTokens + [5, 6], mode: "chunked", value: 101,
                                        payloadBytes: 2000, allowsSuffix: false))
        XCTAssertEqual(try store.lookup(tokens: baseTokens + [9], mode: "chunked")?.tokenCount, 64,
                       "Pinned shared base must never be evicted by slot pressure")

        // Give headroom for multi-turn slot checkpoints and verify slotBaseKey advances across turns.
        store.ensureByteLimit(atLeast: 5000)
        // Turn 1: intermediate checkpoint before <think> (99), then final ending after <think>
        XCTAssertTrue(try store.insert(tokens: baseTokens + [10, 11], mode: "chunked", value: 201,
                                       payloadBytes: 100, allowsSuffix: true))
        XCTAssertTrue(try store.insert(tokens: baseTokens + [10, 11, 99, 12], mode: "chunked", value: 202,
                                       payloadBytes: 100, allowsSuffix: true))

        // Turn 2 strips <think> (99) so prompt starts with baseTokens + [10, 11, 12, 20, 21]
        let turn2Lookup = try store.lookup(tokens: baseTokens + [10, 11, 12, 20, 21, 99, 22], mode: "chunked")
        XCTAssertEqual(turn2Lookup?.tokenCount, 66)
        XCTAssertEqual(turn2Lookup?.value, 201)

        // Turn 2 stores its own intermediate checkpoint and final checkpoint in the SAME slot
        XCTAssertTrue(try store.insert(tokens: baseTokens + [10, 11, 12, 20, 21], mode: "chunked", value: 301,
                                       payloadBytes: 100, allowsSuffix: true))
        XCTAssertTrue(try store.insert(tokens: baseTokens + [10, 11, 12, 20, 21, 99, 22], mode: "chunked", value: 302,
                                       payloadBytes: 100, allowsSuffix: true))

        // Turn 3 strips <think> (99) from Turn 2 and must match Turn 2's 69-token intermediate checkpoint
        let turn3Lookup = try store.lookup(tokens: baseTokens + [10, 11, 12, 20, 21, 22, 30], mode: "chunked")
        XCTAssertEqual(turn3Lookup?.tokenCount, 69,
                       "slotBaseKey must advance to the longest valid prefix checkpoint across turns")
        XCTAssertEqual(turn3Lookup?.value, 301)
        XCTAssertEqual(store.count, 3, "Pinned base + slotBaseKey + slotLastKey must not leak superseded entries")
    }
}
