import XCTest
import Foundation
@testable import QWFlashNext

/// The decode stack's commit order is what keeps the GPU busy while the PLE
/// rows are read from the SSD, so the order is pinned here rather than left
/// to be re-derived from the code.
final class FNGPUDecodeStackOrderTests: XCTestCase {
    /// The order the OLD code produced, kept as the reference: each chunk
    /// waits for the embedding it consumes, then commits.
    static func legacyOrder(chunkHasPLE: [Bool]) -> [FNGPUDecodeStack.Step] {
        var steps: [FNGPUDecodeStack.Step] = []
        var waited = false
        for hasPLE in chunkHasPLE {
            if hasPLE, !waited {
                steps.append(.waitForPLE)
                waited = true
            }
            steps.append(.commit)
        }
        return steps
    }

    /// The order the new code produces, flattened.
    static func flatOrder(chunkHasPLE: [Bool]) -> [FNGPUDecodeStack.Step] {
        FNGPUDecodeStack.stepOrder(chunkHasPLE: chunkHasPLE).flatMap { $0 }
    }

    func testTheCutKeepsThePrePLEChunkFirst() {
        XCTAssertEqual(FNGPUDecodeStack.decodeCuts(slotCount: 48), [0, 1, 8, 48])
    }

    func testASmallStackDegradesToOneChunk() {
        XCTAssertEqual(FNGPUDecodeStack.decodeCuts(slotCount: 4), [0, 4])
    }

    func testThePrePLEChunkIsCommittedBeforeTheWait() {
        // The chunk that does not consume the embedding is committed FIRST,
        // so the GPU starts on it while the SSD reads are still in flight.
        let steps = FNGPUDecodeStack.stepOrder(chunkHasPLE: [false, true, false])
        XCTAssertEqual(steps, [[.commit], [.waitForPLE, .commit], [.commit]])
    }

    func testTheWaitPrecedesTheCommitOfTheChunkThatConsumesTheEmbedding() {
        let flat = Self.flatOrder(chunkHasPLE: [false, true, false])
        guard let waitIndex = flat.firstIndex(of: .waitForPLE) else {
            return XCTFail("the wait must be in the order")
        }
        guard let firstCommit = flat.firstIndex(of: .commit) else {
            return XCTFail("the order must commit a chunk")
        }
        XCTAssertLessThan(firstCommit, waitIndex, "the pre-PLE chunk must be committed before the wait")
        XCTAssertEqual(flat[waitIndex + 1], .commit, "the wait must precede a commit")
        XCTAssertEqual(flat, [.commit, .waitForPLE, .commit, .commit])
    }

    func testAPLEChunkInTheFirstChunkCollapsesTheOverlap() {
        // This is why the cut may not move: a wait before the first commit
        // leaves the GPU with nothing to overlap the SSD read with.
        let steps = FNGPUDecodeStack.stepOrder(chunkHasPLE: [true, false])
        XCTAssertEqual(steps, [[.waitForPLE, .commit], [.commit]])
        XCTAssertEqual(steps.first?.first, .waitForPLE)
    }

    func testTheRealGeometryKeepsThePLELayerOutOfTheFirstChunk() {
        // The real model has its PLE layer at index 1 (`ple_layer_ids = [2]`),
        // and the first chunk holds only layer 0.
        let cuts = FNGPUDecodeStack.decodeCuts(slotCount: 48)
        let pleIndex = 1
        XCTAssertGreaterThan(pleIndex, cuts[0], "the first chunk must not carry the PLE layer")
        XCTAssertGreaterThanOrEqual(pleIndex, cuts[1], "the PLE layer must start the second chunk")
        XCTAssertLessThan(pleIndex, cuts[2], "the PLE layer must be in the second chunk")
    }

    func testOnlyOneWaitIsEverNeeded() {
        // There is one PLE layer, so there is one wait. A second would be a
        // second chance to block a commit that does not depend on it.
        let steps = FNGPUDecodeStack.stepOrder(chunkHasPLE: [false, true, true, false])
        XCTAssertEqual(steps.flatMap { $0 }.filter { $0 == .waitForPLE }.count, 1)
    }

    func testEveryChunkCommitsExactlyOnce() {
        // Each chunk's step list ends with its one commit, so the command
        // buffer is ended exactly once per chunk and never left open.
        for chunks in 1...9 {
            for flags in allPatterns(count: chunks) {
                let steps = FNGPUDecodeStack.stepOrder(chunkHasPLE: flags)
                XCTAssertEqual(steps.count, chunks)
                for (index, chunkSteps) in steps.enumerated() {
                    XCTAssertEqual(chunkSteps.filter { $0 == .commit }.count, 1,
                                   "chunk \(index) of \(flags) must commit exactly once")
                    XCTAssertEqual(chunkSteps.last, .commit, "chunk \(index) of \(flags) must end on its commit")
                }
            }
        }
    }

    func testTheOrderMatchesTheOldCodeForEveryChunkLayout() {
        // The change must be a refactor, not a behaviour change: for every
        // layout of chunks with and without the PLE layer, the new order is
        // the old order.
        for chunks in 1...9 {
            for flags in allPatterns(count: chunks) {
                XCTAssertEqual(Self.flatOrder(chunkHasPLE: flags),
                               Self.legacyOrder(chunkHasPLE: flags),
                               "the order changed for \(flags)")
            }
        }
    }

    /// Every way of putting the PLE layer in a layout of `count` chunks.
    private func allPatterns(count: Int) -> [[Bool]] {
        (0..<(1 << count)).map { bits in
            (0..<count).map { bits & (1 << $0) != 0 }
        }
    }

    func testVerifyRollbackRingRetainsEveryAcceptedState() {
        for rows in 1...8 {
            XCTAssertTrue(FNGPUDecodeStack.supportsRollbackWindow(rows))
            for accepted in 0..<rows {
                var ring = [Int](repeating: -1, count: 8)
                ring[0] = 0
                var parity = 0
                for position in 1...rows {
                    parity = (parity + 1) & 7
                    ring[parity] = position
                }
                let rejected = rows - 1 - accepted
                XCTAssertEqual(ring[(parity - rejected) & 7], accepted + 1)
            }
        }
        XCTAssertFalse(FNGPUDecodeStack.supportsRollbackWindow(0))
        XCTAssertFalse(FNGPUDecodeStack.supportsRollbackWindow(9))
    }

    func testDynamicWideContextCopyProposesUpToSevenDrafts() {
        let block: [Int64] = [10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27]
        let history: [Int64] = block + [99] + [10, 11, 12, 13, 14, 15]
        let prop = FNContextCopyDrafter.proposeDrafts(
            history: history,
            mode: .hybrid,
            maxDrafts: FNGPUSessionData.defaultMTPDrafts,
            draftCapacity: FNGPUSessionData.draftCapacity,
            mtpDraft: 20,
            mtpChain: [20, 21, 22],
            mtpChainPos: history.count,
            mtpActive: true,
            relaxedMargin: 0.25,
            quadraticSeeding: true
        )
        XCTAssertEqual(prop.drafts, [20, 21, 22, 23, 24, 25, 26])
        XCTAssertFalse(prop.mtpActive)
        XCTAssertEqual(prop.mtpSourcedDrafts, 0)

        let novelHistory: [Int64] = [1, 2, 3, 4, 5, 6, 7, 8]
        let novelProp = FNContextCopyDrafter.proposeDrafts(
            history: novelHistory,
            mode: .hybrid,
            maxDrafts: FNGPUSessionData.defaultMTPDrafts,
            draftCapacity: FNGPUSessionData.draftCapacity,
            mtpDraft: 42,
            mtpChain: [42, 43, 44, 45, 46],
            mtpChainPos: novelHistory.count,
            mtpActive: true,
            relaxedMargin: 0.25,
            quadraticSeeding: true
        )
        XCTAssertEqual(novelProp.drafts, [42, 43, 44])
        XCTAssertTrue(novelProp.mtpActive)
        XCTAssertEqual(novelProp.mtpSourcedDrafts, 3)
    }

    func testContextCopyDrafterReturnsMultiTokenMatch() {
        let history: [Int64] = [10, 20, 30, 40, 50, 60, 70, 99, 10, 20, 30, 40]
        let match = FNContextCopyDrafter.drafts(history: history, maxCount: 3, minMatchLen: 3)
        XCTAssertNotNil(match)
        XCTAssertEqual(match?.matchLen, 4)
        XCTAssertEqual(match?.tokens, [50, 60, 70])
        XCTAssertEqual(FNContextCopyDrafter.draft(history: history, minMatchLen: 4), 50)
        let prefix: [Int64] = [10, 20, 30, 40, 50, 60, 70, 99, 10, 20, 30]
        let appended = FNContextCopyDrafter.drafts(history: prefix, appendedToken: 40, maxCount: 3, minMatchLen: 3)
        XCTAssertEqual(appended?.matchLen, 4)
        XCTAssertEqual(appended?.tokens, [50, 60, 70])

        let counter = [Int64](repeating: 0, count: 2050) + [10, 1, 10, 2, 10, 1, 0, 10, 1, 1, 10]
        XCTAssertNil(FNContextCopyDrafter.drafts(history: counter, appendedToken: 1, maxCount: 2, minMatchLen: 2))

        let consistentShort = [Int64](repeating: 0, count: 2050) + [10, 1, 5, 99, 10, 1, 5, 88, 10]
        let shortMatch = FNContextCopyDrafter.drafts(history: consistentShort, appendedToken: 1, maxCount: 2, minMatchLen: 2)
        XCTAssertEqual(shortMatch?.matchLen, 2)
        XCTAssertEqual(shortMatch?.tokens.first, 5)
    }

    func testContextCopyDrafterPeriodicDrafts() {
        let history: [Int64] = [1, 2, 3, 1, 2, 3]
        let drafts = FNContextCopyDrafter.periodicDrafts(history: history, maxCount: 3)
        XCTAssertEqual(drafts, [1, 2, 3])
        XCTAssertEqual(FNContextCopyDrafter.periodicDraft(history: history), 1)
    }
}
