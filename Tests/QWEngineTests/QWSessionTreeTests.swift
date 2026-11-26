// QWSessionTreeTests.swift — Unit tests for QWSessionTree COW forks, Radix pin refcounts, restore & crash journal (G4).

import Foundation
import QWCore
@testable import QWEngine
import XCTest

final class QWSessionTreeTests: XCTestCase {
    func testCheckpointForkRestorePruneAndCrashJournalRecovery() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-sess-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let journal = dir.appendingPathComponent("tree.jsonl")

        let tree = QWSessionTree(journalURL: journal)
        let root = try tree.checkpoint(
            label: "root",
            modelName: "qwen3.5-9b",
            kvMode: "q4",
            cacheSalt: "tenantA",
            tokenIds: [1, 2, 3, 4],
            recurrentCheckpointId: "rcpt_0",
            metadata: ["topic": "plan"]
        )
        XCTAssertEqual(tree.pinRefCount(tokenIds: [1, 2, 3, 4], kvMode: "q4", cacheSalt: "tenantA"), 1)

        // O(1) COW fork increments pin refcount without duplicating KV entries.
        let children = try tree.fork(from: root.id, count: 3, labels: ["a", "b", "c"])
        XCTAssertEqual(children.count, 3)
        XCTAssertEqual(children.map(\.prefixHash), [root.prefixHash, root.prefixHash, root.prefixHash])
        XCTAssertEqual(tree.pinRefCount(tokenIds: [1, 2, 3, 4], kvMode: "q4", cacheSalt: "tenantA"), 4)

        let lineage = try tree.lineage(of: children[1].id)
        XCTAssertEqual(lineage.map(\.id), [root.id, children[1].id])

        // Restore reports kvCacheHit when probe confirms prefix residency, else needsReplay.
        let hitRestore = try tree.restore(nodeId: children[0].id) { toks, mode in
            toks == [1, 2, 3, 4] && mode == "q4|salt=tenantA"
        }
        XCTAssertTrue(hitRestore.kvCacheHit)
        XCTAssertFalse(hitRestore.needsReplay)

        let missRestore = try tree.restore(nodeId: children[0].id) { _, _ in false }
        XCTAssertFalse(missRestore.kvCacheHit)
        XCTAssertTrue(missRestore.needsReplay)

        // Non-leaf prune refuses without cascade=true.
        XCTAssertThrowsError(try tree.prune(nodeId: root.id, cascade: false)) { err in
            XCTAssertEqual(err as? QWSessionTreeError, .cannotPruneWithChildren(root.id))
        }

        let removedLeaf = try tree.prune(nodeId: children[2].id, cascade: false)
        XCTAssertEqual(removedLeaf, [children[2].id])
        XCTAssertEqual(tree.pinRefCount(tokenIds: [1, 2, 3, 4], kvMode: "q4", cacheSalt: "tenantA"), 3)

        // Check 0o600 journal permissions and append truncated trailing corruption to verify crash recovery.
        let attrs = try FileManager.default.attributesOfItem(atPath: journal.path)
        XCTAssertEqual((attrs[.posixPermissions] as? NSNumber)?.intValue, 0o600)

        let h = try FileHandle(forWritingTo: journal)
        try h.seekToEnd()
        try h.write(contentsOf: Data("{\"truncated_partial".utf8))
        try h.close()

        let reloaded = QWSessionTree(journalURL: journal)
        XCTAssertEqual(reloaded.stats().nodeCount, 3) // root + children[0], children[1]
        XCTAssertNotNil(reloaded.node(id: root.id))
        XCTAssertNil(reloaded.node(id: children[2].id))
        XCTAssertEqual(reloaded.pinRefCount(tokenIds: [1, 2, 3, 4], kvMode: "q4", cacheSalt: "tenantA"), 3)

        // Cascade prune tears down the entire subtree and drops pin refcount to 0.
        _ = try reloaded.prune(nodeId: root.id, cascade: true)
        XCTAssertEqual(reloaded.stats().nodeCount, 0)
        XCTAssertEqual(reloaded.pinRefCount(tokenIds: [1, 2, 3, 4], kvMode: "q4", cacheSalt: "tenantA"), 0)
    }

    func testPrefixStoreNonMutatingHasCachedPrefixProbe() throws {
        let store = try QWEnginePrefixStore<Int>(byteLimit: 4096, entryLimit: 8, vocab: 1000)
        XCTAssertFalse(store.hasCachedPrefix(tokens: [1, 2, 3], mode: "f16"))
        _ = try store.insert(tokens: [1, 2, 3], mode: "f16", value: 42, payloadBytes: 64)
        XCTAssertTrue(store.hasCachedPrefix(tokens: [1, 2, 3], mode: "f16"))
        XCTAssertFalse(store.hasCachedPrefix(tokens: [1, 2, 3], mode: "q4"))
        XCTAssertFalse(store.hasCachedPrefix(tokens: [1, 2, 3, 4], mode: "f16"))
    }
}
