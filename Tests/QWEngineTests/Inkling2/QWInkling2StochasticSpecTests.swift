// QWInkling2StochasticSpecTests.swift — Tests for Inkling2 DFlash/DSpark QWSpeculativeRoundResolver migration.

import Foundation
import XCTest
import QWCore
@testable import QWEngine
import QWHead
import QWMTP

final class QWInkling2StochasticSpecTests: XCTestCase {
    func testLoopPenaltyHistoryMatchesIncrementalRecordGeneratedTokenForPenalty() {
        let vocab = 256
        let baseRecent = [10, 20, 30, 40, 10, 20, 30, 40]
        var baseActive: [Int] = []
        var checkRecent: [Int] = []
        for t in baseRecent {
            QWInkling2Runtime.recordGeneratedTokenForPenalty(
                t, vocab: vocab, recentGenerated: &checkRecent, activeLoopTokens: &baseActive
            )
        }

        let emptyPrefix = QWInkling2Runtime.loopPenaltyHistory(
            baseRecent: baseRecent, baseActive: baseActive, draftPrefix: [][...], vocab: vocab
        )
        XCTAssertEqual(emptyPrefix, baseActive)

        let draftPrefix = [10, 20, 30, 40]
        let combined = QWInkling2Runtime.loopPenaltyHistory(
            baseRecent: baseRecent, baseActive: baseActive, draftPrefix: draftPrefix[...], vocab: vocab
        )
        var refRecent = baseRecent
        var refActive = baseActive
        for t in draftPrefix {
            QWInkling2Runtime.recordGeneratedTokenForPenalty(
                t, vocab: vocab, recentGenerated: &refRecent, activeLoopTokens: &refActive
            )
        }
        XCTAssertEqual(combined, refActive)
        XCTAssertEqual(combined, [10])
    }

    func testSeededDFlashAndStochasticSpecRoundResolverIsDeterministicAndExact() throws {
        let vocab = 6
        let draftLogits: [Float] = [0.2, 1.5, 0.9, -0.5, 1.1, -1.0]
        let targetLogits0: [Float] = [0.4, 1.2, 1.0, 0.2, 0.9, -0.8]
        let targetLogits1: [Float] = [1.7, 0.3, -0.2, 0.8, 0.1, -1.2]

        let cfg = QWSamplerConfig(
            temperature: 0.85, topK: 4, topP: 0.9, seed: 777,
            uniformVersion: .upper24, hasExplicitSeed: true
        )
        let sampler = QWSampler(config: cfg)
        let draftDist = try sampler.distribution(logits: draftLogits)

        func runOnce(seed: UInt64) throws -> QWSpeculativeRoundResolver.Resolution {
            let stream = QWMTPRandomStream(seed: seed ^ 0x4446_4C53_9E37_79B9, drawIndex: 1)
            let draftStream = QWMTPRandomStream(
                seed: stream.seed, drawIndex: try stream.reserveDraws(UInt64(2 * vocab))
            )
            try draftStream.beginGumbelRound(drafts: 1, vocabulary: vocab, maxDrafts: 7)
            let gProp = try draftStream.sampleGumbelProposal(draftDist)
            let proposals: [QWSpeculativeRoundResolver.DraftProposal] = [
                .distribution(token: gProp.token, distribution: gProp.distribution),
            ]
            return try QWSpeculativeRoundResolver.resolve(
                draftProposals: proposals,
                stream: stream,
                isGreedy: false,
                maxNew: 4,
                stopIDs: []
            ) { row, _, _ in
                try sampler.distribution(logits: row == 0 ? targetLogits0 : targetLogits1)
            }
        }

        let first = try runOnce(seed: cfg.seed)
        let second = try runOnce(seed: cfg.seed)
        XCTAssertEqual(first, second, "seeded DFlash/DSpark QWSpeculativeRoundResolver path must be deterministic")
    }

    func testRelaxedGreedySamplerGateMatchesLogitMarginAndProbabilityFloor() throws {
        let vocab = 4
        // Mode is token 0 (logit 3.0); draft is token 1 (logit 2.7 -> delta 0.3).
        let rowLogits: [Float] = [3.0, 2.7, -5.0, -5.0]
        let relaxedSampler = QWSampler(config: QWSamplerConfig(
            temperature: 1.0, topK: vocab, topP: 1.0, uniformVersion: .upper24
        ))

        let acceptRes = try QWSpeculativeRoundResolver.resolve(
            draftProposals: [.greedyToken(1)],
            stream: nil,
            isGreedy: true,
            maxNew: 2,
            margin: 0.4,
            minProbability: 0.3
        ) { row, _, _ in
            if row == 0 {
                return try relaxedSampler.distribution(logits: rowLogits)
            }
            return try QWSamplingDistribution.sparse(vocabulary: vocab, ids: [0], weights: [1.0])
        }
        XCTAssertEqual(acceptRes.acceptedDrafts, 1)
        XCTAssertEqual(acceptRes.tokens, [1, 0])

        let rejectFloor = try QWSpeculativeRoundResolver.resolve(
            draftProposals: [.greedyToken(1)],
            stream: nil,
            isGreedy: true,
            maxNew: 2,
            margin: 0.4,
            minProbability: 0.5
        ) { _, _, _ in
            try relaxedSampler.distribution(logits: rowLogits)
        }
        XCTAssertEqual(rejectFloor.acceptedDrafts, 0)
        XCTAssertEqual(rejectFloor.tokens, [0])
    }

    func testDSparkDraftProposalsAdapterConvertsGumbelProposals() throws {
        let dist = try QWSamplingDistribution.sparse(vocabulary: 8, ids: [1, 3, 5], weights: [0.2, 0.5, 0.3])
        let stream = QWMTPRandomStream(seed: 99)
        try stream.beginGumbelRound(drafts: 2, vocabulary: 8, maxDrafts: 7)
        let p0 = try stream.sampleGumbelProposal(dist)
        let p1 = try stream.sampleGumbelProposal(dist)
        let adapted: [QWSpeculativeRoundResolver.DraftProposal] = [p0, p1].map {
            .distribution(token: $0.token, distribution: $0.distribution)
        }
        XCTAssertEqual(adapted.map(\.token), [p0.token, p1.token])
        if case .distribution(let tok, let d) = adapted[0] {
            XCTAssertEqual(tok, p0.token)
            XCTAssertEqual(d.survivorIds, dist.survivorIds)
        } else {
            XCTFail("expected .distribution DraftProposal")
        }
    }
}
