import XCTest
import QWHead
import QWMTP
@testable import QWEngine

final class QWGreedyRoundDecisionTests: XCTestCase {
    func testSampledGumbelRejectionUsesTargetWinnerAndFiveDraftBoundary() throws {
        let sampler = QWSampler(config: QWSamplerConfig(temperature: 1, topK: 1, topP: 1,
                                                        uniformVersion: .upper24))
        let draft = try sampler.distribution(logits: [10, 0, 0])
        let target = try sampler.distribution(logits: [0, 10, 0])
        let stream = QWMTPRandomStream(seed: 23)
        try stream.beginGumbelRound(drafts: 1, vocabulary: 3)
        let proposals = [try stream.sampleGumbelProposal(draft)]
        let result = try stream.resolveGumbelRound(proposals: proposals,
                                                   targetDistributions: [target, draft])
        XCTAssertEqual(result.accepted, 0)
        XCTAssertEqual(result.finalToken, 1)
        XCTAssertEqual(try QWGreedyRoundDecision.sampled(geometry: QWTestGeometry.g27, proposals: proposals, result: result,
            maxNew: 2, stopIDs: []).tokens, [1])

        let drafts = [11, 12, 13, 14, 15]
        let decision = try QWGreedyRoundDecision.make(geometry: QWTestGeometry.g27, drafts: drafts,
            targetTokens: drafts + [16], maxNew: 6)
        XCTAssertEqual(decision.acceptedDrafts, 5)
        XCTAssertEqual(decision.tokens, drafts + [16])
        XCTAssertEqual(try decision.retainedPairRows(drafts: 5), 5)
    }

    func testEveryAcceptanceAndBudgetBoundary() throws {
        for drafts in [[11], [11, 12]] {
            for accepted in 0...drafts.count {
                var targets = drafts + [13]
                if accepted < drafts.count { targets[accepted] = 99 }
                let full = Array(drafts.prefix(accepted)) + [targets[accepted]]
                for budget in 0...4 {
                    let decision = try QWGreedyRoundDecision.make(geometry: QWTestGeometry.g27, drafts: drafts, targetTokens: targets, maxNew: budget)
                    XCTAssertEqual(decision.acceptedDrafts, accepted)
                    XCTAssertEqual(decision.tokens, Array(full.prefix(budget)))
                    XCTAssertEqual(decision.retainedRows, min(budget, accepted + 1))
                    XCTAssertEqual(decision.pendingToken, full.prefix(budget).last)
                    XCTAssertEqual(try decision.retainedPairRows(drafts: drafts.count),
                                   min(min(budget, accepted + 1), drafts.count))
                    for delivered in 0...decision.tokens.count {
                        let cut = try decision.truncating(to: delivered)
                        XCTAssertEqual(cut.retainedRows, delivered)
                        XCTAssertEqual(cut.tokens, Array(full.prefix(delivered)))
                        XCTAssertEqual(try cut.retainedPairRows(drafts: drafts.count), min(delivered, drafts.count))
                    }
                }
            }
        }
    }

    func testStopTokensAreNotEmittedOrConsumedAsPending() throws {
        for stop in [11, 12, 13] {
            let decision = try QWGreedyRoundDecision.make(geometry: QWTestGeometry.g27, drafts: [11, 12], targetTokens: [11, 12, 13],
                                                         maxNew: 3, stopIDs: [stop])
            XCTAssertTrue(decision.stopped)
            XCTAssertEqual(decision.tokens, Array([11, 12, 13].prefix(while: { $0 != stop })))
            XCTAssertEqual(decision.retainedRows, decision.tokens.count)
        }
        let limited = try QWGreedyRoundDecision.make(geometry: QWTestGeometry.g27, drafts: [11, 12], targetTokens: [11, 12, 13],
                                                    maxNew: 1, stopIDs: [12])
        XCTAssertFalse(limited.stopped)
        XCTAssertEqual(limited.tokens, [11])
    }

    func testInvalidInputsRefuseDecisions() throws {
        XCTAssertThrowsError(try QWGreedyRoundDecision.make(geometry: QWTestGeometry.g27, drafts: [], targetTokens: [1], maxNew: 1))
        XCTAssertThrowsError(try QWGreedyRoundDecision.make(geometry: QWTestGeometry.g27, drafts: [1], targetTokens: [1], maxNew: 1))
        XCTAssertThrowsError(try QWGreedyRoundDecision.make(geometry: QWTestGeometry.g27, drafts: [1], targetTokens: [1, 2], maxNew: -1))
        XCTAssertThrowsError(try QWGreedyRoundDecision.make(geometry: QWTestGeometry.g27, drafts: [-1], targetTokens: [1, 2], maxNew: 1))
        let decision = try QWGreedyRoundDecision.make(geometry: QWTestGeometry.g27, drafts: [1], targetTokens: [1, 2], maxNew: 2)
        XCTAssertThrowsError(try decision.truncating(to: 3))
    }
}
