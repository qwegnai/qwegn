import XCTest
import Metal
@testable import QWEngine

final class QWGreedyGenerationLoopTests: XCTestCase {
    func testBudgetsAndCommittedBatchAccounting() throws {
        for budget in 0...8 {
            var next = 11, budgets: [Int] = []
            let output = try QWGreedyGenerationLoop.run(vocab: QWTestGeometry.g27.vocab, 
                first: 10, maxNew: budget, stopIDs: [], onToken: nil,
                ordinary: { _ in XCTFail("unexpected ordinary path"); return 0 },
                committedRound: { remaining in
                    budgets.append(remaining)
                    let count = min(3, remaining)
                    let tokens = Array(next..<(next + count))
                    next += count
                    return QWGreedyRoundDecision(acceptedDrafts: 2, tokens: tokens, stopped: false)
                })
            XCTAssertEqual(output.tokens, Array(10..<(10 + budget)))
            XCTAssertEqual(output.stepMs.count, output.stepTokenCounts.count)
            XCTAssertEqual(output.stepTokenCounts.reduce(0, +), max(0, budget - 1))
            XCTAssertTrue(output.stepMs.allSatisfy { $0 >= 0 && $0.isFinite })
            XCTAssertEqual(budgets.first, budget > 1 ? budget - 1 : nil)
        }
    }

    func testSpeculativeCallbacksDeliverOnlyCommittedBatches() throws {
        for cancelAfter in 1...7 {
            var delivered: [Int] = [], committed: [Int] = [], rounds = 0
            let result = try QWGreedyGenerationLoop.run(vocab: QWTestGeometry.g27.vocab, 
                first: 10, maxNew: 7, stopIDs: [],
                onToken: { token in
                    if token != 10 { XCTAssertTrue(committed.contains(token), "callback ran before round commit") }
                    delivered.append(token)
                    return delivered.count < cancelAfter
                },
                ordinary: { _ in XCTFail("speculative callbacks must not use ordinary decode"); return 0 },
                committedRound: { _ in
                    rounds += 1
                    let start = 11 + (rounds - 1) * 3
                    let tokens = Array(start..<min(start + 3, 17))
                    committed.append(contentsOf: tokens)
                    return QWGreedyRoundDecision(acceptedDrafts: 2, tokens: tokens, stopped: false)
                },
                streamsCommittedRounds: true)
            XCTAssertEqual(result.tokens, delivered)
            let expectedCount: Int
            if cancelAfter == 1 {
                expectedCount = 1
            } else {
                expectedCount = min(7, 1 + ((cancelAfter - 2) / 3 + 1) * 3)
            }
            XCTAssertEqual(delivered.count, expectedCount)
            XCTAssertEqual(rounds, max(0, (expectedCount - 1 + 2) / 3))
            XCTAssertEqual(result.stepTokenCounts.reduce(0, +), expectedCount - 1)
        }
    }

    func testOrdinaryCallbackStillUsesOrdinaryDecodeByDefault() throws {
        var delivered: [Int] = [], consumed: [Int] = []
        let result = try QWGreedyGenerationLoop.run(vocab: QWTestGeometry.g27.vocab, 
            first: 10, maxNew: 5, stopIDs: [],
            onToken: { token in delivered.append(token); return delivered.count < 3 },
            ordinary: { token in consumed.append(token); return token + 1 },
            committedRound: { _ in XCTFail("committed rounds require explicit streaming opt-in"); throw Probe.failure })
        XCTAssertEqual(result.tokens, [10, 11, 12])
        XCTAssertEqual(delivered, result.tokens)
        XCTAssertEqual(consumed, [10, 11])
    }

    private enum Probe: Error { case failure }

    func testEmptyStoppedRoundPreservesEmittedPrefix() throws {
        var rounds = 0, stoppedToken: Int?
        let result = try QWGreedyGenerationLoop.run(vocab: QWTestGeometry.g27.vocab, 
            first: 10, maxNew: 7, stopIDs: [11], onToken: nil,
            ordinary: { _ in XCTFail("unexpected ordinary path"); return 0 },
            committedRound: { remaining in
                rounds += 1
                XCTAssertEqual(remaining, 6)
                return try QWGreedyRoundDecision.make(geometry: QWTestGeometry.g27, 
                    drafts: [11, 12], targetTokens: [11, 12, 13], maxNew: remaining, stopIDs: [11])
            },
            onStopToken: { stoppedToken = $0 })
        XCTAssertEqual(result.tokens, [10])
        XCTAssertEqual(result.stepTokenCounts, [0])
        XCTAssertEqual(result.stepMs.count, 1)
        XCTAssertEqual(rounds, 1)
        XCTAssertEqual(stoppedToken, 11)
    }

    func testFailureIsNotReturnedAsSuccessfulPartialOutput() throws {
        var rounds = 0
        XCTAssertThrowsError(try QWGreedyGenerationLoop.run(vocab: QWTestGeometry.g27.vocab, 
            first: 10, maxNew: 8, stopIDs: [], onToken: nil, ordinary: { $0 + 1 },
            committedRound: { _ in
                rounds += 1
                if rounds == 2 { throw Probe.failure }
                return QWGreedyRoundDecision(acceptedDrafts: 1, tokens: [11, 12], stopped: false)
            }))
        XCTAssertEqual(rounds, 2)
    }

    func testStopAndMalformedRoundRefusal() throws {
        let empty = try QWGreedyGenerationLoop.run(vocab: QWTestGeometry.g27.vocab, first: 10, maxNew: 3, stopIDs: [10], onToken: nil,
                                                   ordinary: { _ in XCTFail(); return 0 }, committedRound: nil)
        XCTAssertTrue(empty.tokens.isEmpty)
        let stopped = try QWGreedyGenerationLoop.run(vocab: QWTestGeometry.g27.vocab, 
            first: 10, maxNew: 8, stopIDs: [13], onToken: nil, ordinary: { $0 + 1 },
            committedRound: { _ in
                QWGreedyRoundDecision(acceptedDrafts: 2, tokens: [11, 12], stopped: true, stoppedToken: 13)
            })
        XCTAssertEqual(stopped.tokens, [10, 11, 12])
        XCTAssertEqual(stopped.stepTokenCounts, [2])
        for decision in [
            QWGreedyRoundDecision(acceptedDrafts: 0, tokens: [], stopped: false),
            QWGreedyRoundDecision(acceptedDrafts: 0, tokens: [13], stopped: true, stoppedToken: 13),
            QWGreedyRoundDecision(acceptedDrafts: 2, tokens: [11, 12, 14], stopped: false),
            QWGreedyRoundDecision(acceptedDrafts: 0, tokens: [-1], stopped: false)
        ] {
            XCTAssertThrowsError(try QWGreedyGenerationLoop.run(vocab: QWTestGeometry.g27.vocab, 
                first: 10, maxNew: 3, stopIDs: [13], onToken: nil, ordinary: { $0 + 1 },
                committedRound: { _ in decision }))
        }
    }
    func testSpeculationBackoffFallsBackToSerialOnLowAcceptance() throws {
        var trace: [String] = [], next = 11
        let backoff = QWSpeculationBackoff(drafts: 3, minAcceptance: 0.35, window: 2,
                                           initialCooldown: 3, maxCooldown: 6)
        let result = try QWGreedyGenerationLoop.run(vocab: QWTestGeometry.g27.vocab,
            first: 10, maxNew: 30, stopIDs: [], onToken: nil,
            ordinary: { token in trace.append("o"); XCTAssertEqual(token, next - 1); next += 1; return next - 1 },
            committedRound: { remaining in
                trace.append("r")
                let tokens = Array(next..<(next + min(1, remaining)))
                next += tokens.count
                return QWGreedyRoundDecision(acceptedDrafts: 0, tokens: tokens, stopped: false)
            }, backoff: backoff)
        XCTAssertEqual(result.tokens, Array(10..<40))
        XCTAssertEqual(trace.joined(), "rrooo" + "rroooooo" + "rroooooo" + "rroooooo")
    }

    func testSpeculationBackoffKeepsSpeculatingOnHighAcceptance() throws {
        var next = 11
        let result = try QWGreedyGenerationLoop.run(vocab: QWTestGeometry.g27.vocab,
            first: 10, maxNew: 41, stopIDs: [], onToken: nil,
            ordinary: { _ in XCTFail("high acceptance must not back off"); return 0 },
            committedRound: { remaining in
                let tokens = Array(next..<(next + min(4, remaining)))
                next += tokens.count
                return QWGreedyRoundDecision(acceptedDrafts: tokens.count - 1, tokens: tokens, stopped: false)
            }, backoff: QWSpeculationBackoff(drafts: 3))
        XCTAssertEqual(result.tokens, Array(10..<51))
    }

    func testSpeculationBackoffArmsOnlyForLongContext() {
        XCTAssertNil(QWSpeculationBackoff.armed(position: 131_071, drafts: 3))
        XCTAssertNil(QWSpeculationBackoff.armed(position: 1_000_000, drafts: 0))
        XCTAssertEqual(QWSpeculationBackoff.armed(position: 131_072, drafts: 3)?.drafts, 3)
    }

    func testSpeculationBackoffStateMachineAndColdEventShutdown() throws {
        var backoff = QWSpeculationBackoff(drafts: 4, minAcceptance: 0.35, window: 2, initialCooldown: 4, maxCooldown: 16)
        XCTAssertEqual(backoff.recommendedDrafts, 4)
        backoff.recordSpeculativeRound(acceptedDrafts: 2)
        backoff.recordSpeculativeRound(acceptedDrafts: 2)
        XCTAssertEqual(backoff.serialLeft, 0)
        XCTAssertEqual(backoff.recommendedDrafts, 3)

        backoff.recordSpeculativeRound(acceptedDrafts: 0)
        backoff.recordSpeculativeRound(acceptedDrafts: 0)
        XCTAssertEqual(backoff.serialLeft, 4)
        XCTAssertEqual(backoff.recommendedDrafts, 2)
        backoff.recordSerialStep()
        XCTAssertEqual(backoff.serialLeft, 3)

        if let device = MTLCreateSystemDefaultDevice(),
           let pipeline = ColdEventPipeline(device: device) {
            pipeline.shutdown()
            pipeline.shutdown()
        }
    }

    func testResidencyBudgetHonorsPolicyReserveAndLiveHeadroom() {
        let gib = 1 << 30
        XCTAssertEqual(QWEngineDriver.effectiveOSReserveGiB(policy: .maxResident, explicitReserveGiB: nil), 16.0)
        XCTAssertEqual(QWEngineDriver.effectiveOSReserveGiB(policy: .safe, explicitReserveGiB: nil), 24.0)
        XCTAssertEqual(QWEngineDriver.effectiveOSReserveGiB(policy: .maxResident, explicitReserveGiB: 20.0), 20.0)

        let maxResCappedByLive = QWEngineDriver.residencyBudgetBytes(
            policy: .maxResident,
            osReserveGiB: nil,
            arenaCap: 120 * gib,
            liveReclaimableBytes: 90 * gib
        )
        XCTAssertEqual(maxResCappedByLive, 86 * gib)

        let safeCappedByLive = QWEngineDriver.residencyBudgetBytes(
            policy: .safe,
            osReserveGiB: nil,
            arenaCap: 120 * gib,
            liveReclaimableBytes: 90 * gib
        )
        XCTAssertEqual(safeCappedByLive, 84 * gib)
    }
}
