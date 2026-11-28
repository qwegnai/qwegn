// QWSlotPinnedLRUStoreReconfigureTests.swift — Tests reconfigure shrink/grow/zero-purge and removeAll.
import Foundation
import XCTest
@testable import QWCore

final class QWSlotPinnedLRUStoreReconfigureTests: XCTestCase {
    func testReconfigureShrinkRespectsStrictLimitAndTwoPassPriorityOutsideLock() {
        let store = QWSlotPinnedLRUStore<String, Int>(byteLimit: 1000, entryLimit: 10, maxSlots: 4, rootPrefixTokenCount: 4)
        var evicted: [(String, Int)] = []
        store.onEvict = { key, entry in
            // Verify callback runs outside the lock so re-entrant reads do not deadlock.
            XCTAssertFalse(store.contains(key))
            evicted.append((key, entry.charge))
        }

        // Unslotted (non-protected), slot-0 protected, and shared-base entries.
        XCTAssertTrue(store.store(key: "plain-old", value: 1, charge: 100, tokenCount: 10))
        XCTAssertTrue(store.store(key: "slot0-last", value: 2, charge: 100, tokenCount: 12, slot: 0, history: Array(0..<12)))
        XCTAssertTrue(store.store(key: "shared-base", value: 3, charge: 100, tokenCount: 8, isSharedBase: true))
        XCTAssertTrue(store.store(key: "plain-new", value: 4, charge: 100, tokenCount: 14))
        XCTAssertEqual(store.count, 4)
        XCTAssertEqual(store.chargedBytes, 400)

        // Exact boundary (entryLimit: 4, byteLimit: 400) must not evict (strict >).
        store.reconfigure(byteLimit: 400, entryLimit: 4)
        XCTAssertEqual(store.count, 4)
        XCTAssertTrue(evicted.isEmpty)

        // Shrink to 2 entries / 200 bytes: evicts plain-old and plain-new first; slot0-last & shared-base remain.
        store.reconfigure(byteLimit: 200, entryLimit: 2)
        XCTAssertEqual(evicted.map(\.0), ["plain-old", "plain-new"])
        XCTAssertEqual(store.count, 2)
        XCTAssertEqual(store.chargedBytes, 200)
        XCTAssertTrue(store.contains("slot0-last"))
        XCTAssertTrue(store.contains("shared-base"))

        // Shrink further to byteLimit: 100 -> pass 2 evicts slot0-last; shared-base remains pinned.
        store.reconfigure(byteLimit: 100, entryLimit: 2)
        XCTAssertEqual(evicted.map(\.0), ["plain-old", "plain-new", "slot0-last"])
        XCTAssertEqual(store.count, 1)
        XCTAssertTrue(store.contains("shared-base"))
    }

    func testReconfigureGrowRetainsExistingAndAcceptsLargerEntries() {
        let store = QWSlotPinnedLRUStore<String, Int>(byteLimit: 200, entryLimit: 2, maxSlots: 2, rootPrefixTokenCount: 4)
        XCTAssertTrue(store.store(key: "a", value: 1, charge: 100, tokenCount: 5))
        XCTAssertTrue(store.store(key: "b", value: 2, charge: 100, tokenCount: 6))
        XCTAssertFalse(store.store(key: "too-big", value: 3, charge: 250, tokenCount: 7))

        store.reconfigure(byteLimit: 600, entryLimit: 5)
        XCTAssertEqual(store.byteLimit, 600)
        XCTAssertEqual(store.entryLimit, 5)
        XCTAssertEqual(store.count, 2)
        XCTAssertTrue(store.store(key: "c", value: 4, charge: 250, tokenCount: 7))
        XCTAssertEqual(store.count, 3)
        XCTAssertEqual(store.chargedBytes, 450)
    }

    func testReconfigureZeroPurgesAllIncludingSharedAndFiresOnEvictOutsideLock() {
        let store = QWSlotPinnedLRUStore<String, Int>(byteLimit: 512, entryLimit: 8, maxSlots: 2, rootPrefixTokenCount: 4)
        var evicted: [String] = []
        store.onEvict = { key, _ in
            XCTAssertFalse(store.isEnabled)
            XCTAssertEqual(store.count, 0)
            evicted.append(key)
        }
        XCTAssertTrue(store.store(key: "slot0", value: 1, charge: 64, tokenCount: 8, slot: 0, history: Array(0..<8)))
        XCTAssertTrue(store.store(key: "shared", value: 2, charge: 64, tokenCount: 4, isSharedBase: true))

        store.reconfigure(byteLimit: 0, entryLimit: 4)
        XCTAssertFalse(store.isEnabled)
        XCTAssertEqual(store.byteLimit, 0)
        XCTAssertEqual(store.entryLimit, 4)
        XCTAssertEqual(store.count, 0)
        XCTAssertEqual(store.chargedBytes, 0)
        XCTAssertFalse(store.hasSharedBase)
        XCTAssertFalse(store.isProtectedLastKey("slot0"))
        XCTAssertEqual(Set(evicted), Set(["slot0", "shared"]))
    }

    func testRemoveAllClearsEntriesMetadataAndFiresOnRemoveWithoutOnEvict() {
        let store = QWSlotPinnedLRUStore<String, Int>(byteLimit: 512, entryLimit: 8, maxSlots: 2, rootPrefixTokenCount: 4)
        var evicted = 0
        var removed: Set<String> = []
        store.onEvict = { _, _ in evicted += 1 }
        store.onRemove = { key in removed.insert(key) }

        let h0: [Int64] = [1, 2, 3, 4, 5, 6]
        XCTAssertTrue(store.store(key: "k1", value: 1, charge: 64, tokenCount: 6, slot: 0, rootHash: 42, history: h0, isBranchCheckpoint: true))
        XCTAssertTrue(store.store(key: "k2", value: 2, charge: 64, tokenCount: 4, isSharedBase: true))
        XCTAssertEqual(store.sharedBranchPrefixLength(for: [1, 2, 3, 4, 7, 8]), 4)

        store.removeAll()
        XCTAssertEqual(evicted, 0)
        XCTAssertEqual(removed, Set(["k1", "k2"]))
        XCTAssertEqual(store.count, 0)
        XCTAssertEqual(store.chargedBytes, 0)
        XCTAssertTrue(store.order.isEmpty)
        XCTAssertFalse(store.hasSharedBase)
        XCTAssertFalse(store.isProtectedLastKey("k1"))
        XCTAssertNil(store.sharedBranchPrefixLength(for: [1, 2, 3, 4, 7, 8]))
    }

    func testReconfigureShrinkFallbackEvictsSharedBasesWhenBudgetRequires() {
        let store = QWSlotPinnedLRUStore<String, Int>(byteLimit: 500, entryLimit: 10, maxSlots: 2, rootPrefixTokenCount: 4)
        var evicted: [String] = []
        var removed: [String] = []
        store.onEvict = { key, _ in evicted.append(key) }
        store.onRemove = { key in removed.append(key) }

        XCTAssertTrue(store.store(key: "shared-old", value: 1, charge: 120, tokenCount: 4, isSharedBase: true))
        XCTAssertTrue(store.store(key: "shared-new", value: 2, charge: 120, tokenCount: 4, isSharedBase: true))
        XCTAssertTrue(store.store(key: "slot0-last", value: 3, charge: 100, tokenCount: 8, slot: 0, history: Array(0..<8)))
        XCTAssertEqual(store.chargedBytes, 340)

        // Shrink below shared-base footprint: pass 2 evicts slot0-last, pass 3 evicts shared-old to fit 120 B.
        store.reconfigure(byteLimit: 150, entryLimit: 4)
        XCTAssertEqual(store.count, 1)
        XCTAssertEqual(store.chargedBytes, 120)
        XCTAssertFalse(store.contains("slot0-last"))
        XCTAssertFalse(store.contains("shared-old"))
        XCTAssertTrue(store.contains("shared-new"))
        XCTAssertEqual(evicted, ["slot0-last", "shared-old"])
        XCTAssertEqual(removed, ["slot0-last", "shared-old"])

        // Entry-limit shrink with shared bases only: pass 3 brings count down to entryLimit.
        XCTAssertTrue(store.store(key: "shared-a", value: 4, charge: 10, tokenCount: 4, isSharedBase: true))
        XCTAssertTrue(store.store(key: "shared-b", value: 5, charge: 10, tokenCount: 4, isSharedBase: true))
        XCTAssertEqual(store.count, 3)
        store.reconfigure(byteLimit: 150, entryLimit: 1)
        XCTAssertEqual(store.count, 1)
        XCTAssertLessThanOrEqual(store.chargedBytes, 150)
        XCTAssertTrue(store.contains("shared-b"))
    }

    func testOnRemoveCallbacksRunOutsideLockWithoutCrossThreadDeadlock() {
        let store = QWSlotPinnedLRUStore<String, Int>(byteLimit: 512, entryLimit: 8, maxSlots: 2, rootPrefixTokenCount: 4)
        let queue = DispatchQueue(label: "qw.lru.reentrant.test")
        var observedOutsideLock: [String] = []
        store.onRemove = { key in
            // Cross-thread read deadlocks if the store lock is held during onRemove.
            let isStillPresent = queue.sync { store.contains(key) }
            XCTAssertFalse(isStillPresent)
            observedOutsideLock.append(key)
        }

        XCTAssertTrue(store.store(key: "a", value: 1, charge: 64, tokenCount: 4))
        XCTAssertTrue(store.store(key: "b", value: 2, charge: 64, tokenCount: 4))
        XCTAssertTrue(store.store(key: "c", value: 3, charge: 64, tokenCount: 4, isSharedBase: true))

        store.forget("a")
        XCTAssertEqual(observedOutsideLock, ["a"])

        store.removeSharedBaseEntries()
        XCTAssertEqual(observedOutsideLock, ["a", "c"])

        store.removeAll()
        XCTAssertEqual(observedOutsideLock, ["a", "c", "b"])
    }
}
