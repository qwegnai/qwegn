// QWSpeculativeRoundResolverTests.swift — Exactness, lazy cutoff, constraint,
// and adaptive draft policy unit tests for QWSpeculativeRoundResolver & QWAdaptiveDraftPolicy.

import XCTest
import QWCore
import QWHead
@testable import QWMTP

private final class MockGrammarConstraint: QWTokenConstraint, @unchecked Sendable {
    let allowedByStep: [[Int]?]
    let allowsSpec: Bool
    let boundaryToken: Int?
    private(set) var step: Int

    init(allowedByStep: [[Int]?], allowsSpeculation: Bool = true, boundaryToken: Int? = nil, step: Int = 0) {
        self.allowedByStep = allowedByStep
        self.allowsSpec = allowsSpeculation
        self.boundaryToken = boundaryToken
        self.step = step
    }

    var allowsSpeculation: Bool { allowsSpec }
    var isUnconstrainedNow: Bool { step < allowedByStep.count ? allowedByStep[step] == nil : true }
    var speculationBoundaryToken: Int? { boundaryToken }

    func allowedTokens() throws -> [Int]? {
        step < allowedByStep.count ? allowedByStep[step] : nil
    }

    func accept(_ token: Int) throws {
        if let allowed = try allowedTokens(), !QWTokenMaskScratch.sortedTokens(allowed, contains: token) {
            throw QWTokenConstraintError.rejectedToken(token)
        }
        step += 1
    }

    func makeSpeculativeProbe() -> (any QWTokenConstraint)? {
        MockGrammarConstraint(allowedByStep: allowedByStep, allowsSpeculation: allowsSpec,
                              boundaryToken: boundaryToken, step: step)
    }
}

final class QWSpeculativeRoundResolverTests: XCTestCase {
    private func makeDist(_ logits: [Float], temp: Float = 1.0, topK: Int? = nil, topP: Float = 1.0) throws -> QWSamplingDistribution {
        let cfg = QWSamplerConfig(temperature: temp, topK: topK ?? logits.count, topP: topP, uniformVersion: .upper24)
        return try QWSampler(config: cfg).distribution(logits: logits)
    }

    private func assertExactMarginal(
        targetDist: QWSamplingDistribution,
        trials: Int = 6000,
        maxChiSquared: Double = 30.0,
        proposalForSeed: (UInt64) throws -> QWSpeculativeRoundResolver.DraftProposal
    ) throws {
        var counts = [Int](repeating: 0, count: targetDist.vocabulary)
        for s in 0..<trials {
            let seed = UInt64(10_000 + s)
            let stream = QWMTPRandomStream(seed: seed)
            let prop = try proposalForSeed(seed)
            let res = try QWSpeculativeRoundResolver.resolve(
                draftProposals: [prop],
                stream: stream,
                maxNew: 1,
                targetDistribution: { row, _, _ in
                    XCTAssertEqual(row, 0)
                    return targetDist
                }
            )
            XCTAssertEqual(res.tokens.count, 1)
            counts[res.tokens[0]] += 1
        }
        let allowedSet = Set(targetDist.survivorIds)
        for v in 0..<targetDist.vocabulary where !allowedSet.contains(v) {
            XCTAssertEqual(counts[v], 0, "token \(v) outside target support must have zero mass")
        }
        var chiSq = 0.0
        for (tok, prob) in zip(targetDist.survivorIds, targetDist.survivorProbs) where prob > 1e-4 {
            let expected = Double(trials) * prob
            let diff = Double(counts[tok]) - expected
            chiSq += (diff * diff) / expected
        }
        XCTAssertLessThan(chiSq, maxChiSquared, "LC marginal diverged from p (chi2=\(chiSq))")
    }

    func testExactMarginalPureTemperatureTopKTopPAsymmetricAndSmallDraftVocab() throws {
        let targetLogits: [Float] = [1.8, 0.9, 0.2, -0.5, 1.3, -1.2]
        let asymDraftLogits: [Float] = [-1.0, 2.2, 1.5, 0.4, -0.8, 0.1]

        let pPure = try makeDist(targetLogits, temp: 0.85)
        let qAsym = try makeDist(asymDraftLogits, temp: 1.1, topK: 4)
        try assertExactMarginal(targetDist: pPure) { seed in
            let u = Double(QWPhilox.u(0, seed: seed ^ 0xA5A5, version: .upper24))
            let tok = try qAsym.sampleToken(uniform: u, version: .upper24)
            return .sparse(token: tok, ids: qAsym.survivorIds, probs: qAsym.survivorProbs)
        }

        let pTopK = try makeDist(targetLogits, temp: 0.9, topK: 3)
        try assertExactMarginal(targetDist: pTopK) { seed in
            let u = Double(QWPhilox.u(0, seed: seed ^ 0x5A5A, version: .upper24))
            let tok = try qAsym.sampleToken(uniform: u, version: .upper24)
            return .distribution(token: tok, distribution: qAsym)
        }

        let pTopP = try makeDist(targetLogits, temp: 0.9, topK: 6, topP: 0.72)
        try assertExactMarginal(targetDist: pTopP) { seed in
            let u = Double(QWPhilox.u(0, seed: seed ^ 0x1357, version: .upper24))
            let tok = try qAsym.sampleToken(uniform: u, version: .upper24)
            return .sparse(token: tok, ids: qAsym.survivorIds, probs: qAsym.survivorProbs)
        }

        // draftVocab (3) < targetVocab (6): target tokens 3..5 have q(v) = 0.
        let smallDraft = try makeDist(Array(asymDraftLogits.prefix(3)), temp: 1.0)
        try assertExactMarginal(targetDist: pPure) { seed in
            let u = Double(QWPhilox.u(0, seed: seed ^ 0x2468, version: .upper24))
            let tok = try smallDraft.sampleToken(uniform: u, version: .upper24)
            return .sparse(token: tok, ids: smallDraft.survivorIds, probs: smallDraft.survivorProbs)
        }

        // Greedy point-mass draft proposal still yields exact target p(v).
        try assertExactMarginal(targetDist: pTopK) { seed in
            .greedyToken(Int(seed % 6))
        }
    }

    func testExactMarginalFullLogZOverflowFallback() throws {
        let targetLogits: [Float] = [2.0, 0.5, -0.3, 1.4, -1.1]
        let draftLogits: [Float] = [-0.4, 1.9, 0.8, -1.5, 0.2]
        let pTarget = try makeDist(targetLogits, temp: 0.8, topK: 3, topP: 0.9)
        let qUnfiltered = try makeDist(draftLogits, temp: 1.0)
        let maxL = draftLogits.max()!
        let lse = Float(Foundation.log(draftLogits.reduce(0.0) { $0 + Foundation.exp(Double($1 - maxL)) }))

        try draftLogits.withUnsafeBufferPointer { buf in
            try assertExactMarginal(targetDist: pTarget) { seed in
                let u = Double(QWPhilox.u(0, seed: seed ^ 0x7777, version: .upper24))
                let tok = try qUnfiltered.sampleToken(uniform: u, version: .upper24)
                return .fullLogZ(token: tok, temperature: 1.0, topK: draftLogits.count, topP: 1.0,
                                 logits: buf, maxLogit: maxL, logSumExp: lse)
            }
            // Also verify lazy QWSampler fallback when topK < vocab and no precomputed logSumExp is passed.
            let qTrunc = try makeDist(draftLogits, temp: 1.0, topK: 3, topP: 0.9)
            try assertExactMarginal(targetDist: pTarget) { seed in
                let u = Double(QWPhilox.u(0, seed: seed ^ 0x8888, version: .upper24))
                let tok = try qTrunc.sampleToken(uniform: u, version: .upper24)
                return .fullLogZ(token: tok, temperature: 1.0, topK: 3, topP: 0.9, logits: buf)
            }
        }
    }

    func testConstrainedSpeculationZeroGrammarEscape() throws {
        let targetLogits: [Float] = [3.5, 1.0, 0.8, 0.6, 2.5, -1.0]
        let draftLogits: [Float] = [4.0, 0.2, 0.3, 0.1, 0.5, -2.0]
        let allowed0 = [1, 2, 4]
        let sampler = QWSampler(config: QWSamplerConfig(temperature: 0.9, topK: 6, topP: 1.0, uniformVersion: .upper24))
        let qDist = try makeDist(draftLogits, temp: 1.0)
        let expectedDist = try targetLogits.withUnsafeBufferPointer { lBuf in
            try allowed0.withUnsafeBufferPointer { aBuf in
                try sampler.constrainedDistribution(logits: lBuf, allowed: aBuf)
            }
        }
        let constraint = MockGrammarConstraint(allowedByStep: [allowed0, nil])
        var counts = [Int](repeating: 0, count: 6)
        let trials = 4000
        for s in 0..<trials {
            let seed = UInt64(50_000 + s)
            let u = Double(QWPhilox.u(0, seed: seed ^ 0x9999, version: .upper24))
            let draftTok = try qDist.sampleToken(uniform: u, version: .upper24)
            let stream = QWMTPRandomStream(seed: seed)
            let res = try QWSpeculativeRoundResolver.resolve(
                draftProposals: [.distribution(token: draftTok, distribution: qDist)],
                stream: stream,
                maxNew: 1,
                constraint: constraint,
                targetDistribution: { _, _, probe in
                    let allowed = try probe?.allowedTokens() ?? [0, 1, 2, 3, 4, 5]
                    return try targetLogits.withUnsafeBufferPointer { lBuf in
                        try allowed.withUnsafeBufferPointer { aBuf in
                            try sampler.constrainedDistribution(logits: lBuf, allowed: aBuf)
                        }
                    }
                }
            )
            let out = res.tokens[0]
            XCTAssertTrue(allowed0.contains(out), "grammar escape token \(out)")
            counts[out] += 1
        }
        var chiSq = 0.0
        for (tok, prob) in zip(expectedDist.survivorIds, expectedDist.survivorProbs) {
            let exp = Double(trials) * prob
            let d = Double(counts[tok]) - exp
            chiSq += (d * d) / exp
        }
        XCTAssertLessThan(chiSq, 25.0)
    }

    func testEarlyRejectionLazyCutoffNeverEvaluatesTailRows() throws {
        let pAccept = try QWSamplingDistribution.sparse(vocabulary: 4, ids: [1], weights: [1.0])
        let pReject = try QWSamplingDistribution.sparse(vocabulary: 4, ids: [2], weights: [1.0])
        let proposals: [QWSpeculativeRoundResolver.DraftProposal] = [
            .greedyToken(1), // row 0: accepted (p(1) == 1)
            .greedyToken(0), // row 1: rejected (p(0) == 0, emits 2)
            .greedyToken(3), // row 2: MUST NOT be evaluated
            .greedyToken(3), // row 3: MUST NOT be evaluated
        ]
        var evaluatedRows: [Int] = []
        let stream = QWMTPRandomStream(seed: 42)
        let res = try QWSpeculativeRoundResolver.resolve(
            draftProposals: proposals,
            stream: stream,
            maxNew: 5,
            targetDistribution: { row, prefix, _ in
                evaluatedRows.append(row)
                XCTAssertEqual(Array(prefix), Array([1, 0, 3, 3].prefix(row)))
                return row == 0 ? pAccept : pReject
            }
        )
        XCTAssertEqual(evaluatedRows, [0, 1])
        XCTAssertEqual(res.acceptedDrafts, 1)
        XCTAssertEqual(res.tokens, [1, 2])
        XCTAssertEqual(stream.drawsConsumed, 9, "fixed stride 2*D+1 = 9 draws consumed")
    }

    func testRelaxedMarginAcceptsWithinMarginButRespectsConstraints() throws {
        let pTarget = try QWSamplingDistribution.sparse(vocabulary: 4, ids: [0, 1], weights: [0.6, 0.4])
        let res = try QWSpeculativeRoundResolver.resolve(
            draftProposals: [.greedyToken(1)],
            stream: QWMTPRandomStream(seed: 1),
            maxNew: 1,
            margin: 0.5, // log(0.4) - log(0.6) = -0.405 >= -0.5
            targetDistribution: { _, _, _ in pTarget }
        )
        XCTAssertEqual(res.acceptedDrafts, 1)
        XCTAssertEqual(res.tokens, [1])

        let forbidden = try QWSpeculativeRoundResolver.resolve(
            draftProposals: [.greedyToken(2)],
            stream: QWMTPRandomStream(seed: 1),
            maxNew: 1,
            constraint: MockGrammarConstraint(allowedByStep: [[0, 1]]),
            margin: 5.0,
            targetDistribution: { _, _, _ in pTarget }
        )
        XCTAssertEqual(forbidden.acceptedDrafts, 0)
        XCTAssertTrue([0, 1].contains(forbidden.tokens[0]))
    }

    func testGPUFastPathAvoidsTargetDistributionAndHandlesBoundaryAndStop() throws {
        var targetCalls = 0
        let stream = QWMTPRandomStream(seed: 99)
        let gpu = QWGPUVerifyResolution(resolved: true, acceptedCount: 2, tokens: [10, 20, 30])
        let res = try QWSpeculativeRoundResolver.resolve(
            draftProposals: [.greedyToken(10), .greedyToken(20)],
            gpuResolution: gpu,
            stream: stream,
            maxNew: 5,
            stopIDs: [30],
            targetDistribution: { _, _, _ in
                targetCalls += 1
                return try QWSamplingDistribution.sparse(vocabulary: 64, ids: [0], weights: [1.0])
            }
        )
        XCTAssertEqual(targetCalls, 0)
        XCTAssertTrue(res.usedGPUFastPath)
        XCTAssertEqual(res.tokens, [10, 20])
        XCTAssertTrue(res.stopped)
        XCTAssertEqual(res.stoppedToken, 30)
        XCTAssertEqual(stream.drawsConsumed, 5)

        let sparseBuf = QWSparseProposalBuffer(
            token: 1, isSparse: true, overflow: false,
            ids: [0, 1], probs: [0.25, 0.75]
        )
        let prop = try QWSpeculativeRoundResolver.DraftProposal.fromSparseBuffer(sparseBuf)
        XCTAssertEqual(prop.token, 1)
    }

    func testLCRandomStreamFixedStrideAndCancellation() throws {
        let stream = QWMTPRandomStream(seed: 7, drawIndex: 10)
        let window = try stream.beginLCRound(drafts: 3)
        XCTAssertEqual(window.start, 10)
        XCTAssertEqual(window.stride, 7)
        XCTAssertEqual(stream.drawsConsumed, 17)
        XCTAssertTrue(stream.isRoundActive)
        XCTAssertThrowsError(try stream.frozenState())
        XCTAssertThrowsError(try stream.nextUniformUpper24())

        try stream.completeLCRound()
        XCTAssertFalse(stream.isRoundActive)
        XCTAssertEqual(try stream.frozenState().drawsConsumed, 17)

        _ = try stream.beginLCRound(drafts: 2)
        stream.cancelActiveRound()
        XCTAssertFalse(stream.isRoundActive)
        XCTAssertEqual(stream.drawsConsumed, 22, "cancelled LC window draws stay burned")
        let u = try stream.nextUniformUpper24()
        XCTAssertEqual(u, Double(QWPhilox.u(22, seed: 7, version: .upper24)))
    }

    func testAdaptiveDraftPolicySparseMoEStochasticCapAndGreedyTimingInvariance() {
        XCTAssertEqual(
            QWAdaptiveDraftPolicy.neuralDepthCap(
                isStochastic: true, stageProfiling: false, hasDeviceEmbedding: true, numExperts: 288),
            1
        )
        XCTAssertEqual(
            QWAdaptiveDraftPolicy.neuralDepthCap(
                isStochastic: false, stageProfiling: false, hasDeviceEmbedding: true, numExperts: 288),
            1
        )
        XCTAssertNil(
            QWAdaptiveDraftPolicy.neuralDepthCap(
                isStochastic: true, stageProfiling: false, hasDeviceEmbedding: true, numExperts: 0)
        )
        XCTAssertNil(
            QWAdaptiveDraftPolicy.neuralDepthCap(
                isStochastic: true, stageProfiling: true, hasDeviceEmbedding: true, numExperts: 288)
        )

        var greedyPolicy = QWAdaptiveDraftPolicy(mode: .autoregressiveChain)
        var cleanPolicy = QWAdaptiveDraftPolicy(mode: .autoregressiveChain)
        for round in 0..<8 {
            let p1 = greedyPolicy.selectDraftCount(maxDrafts: 4)
            let p2 = cleanPolicy.selectDraftCount(maxDrafts: 4)
            XCTAssertEqual(p1, p2)
            let noisyMs = (round % 2 == 0) ? 900.0 : 1.0
            greedyPolicy.observeRound(draftCount: p1, accepted: 1, roundMs: noisyMs, isGreedy: true)
            greedyPolicy.observeChain(passes: max(1, p1 - 1), ms: noisyMs, isGreedy: true)
            cleanPolicy.observeAcceptance(draftCount: p2, accepted: 1)
        }
        XCTAssertEqual(greedyPolicy.selectDraftCount(maxDrafts: 4), cleanPolicy.selectDraftCount(maxDrafts: 4))
        XCTAssertEqual(greedyPolicy.expectedTokens(drafts: 3), cleanPolicy.expectedTokens(drafts: 3), accuracy: 1e-12)

        var greedyBlock = QWAdaptiveDraftPolicy(mode: .blockDiffusion(isSparseMoE: true))
        var seededBlock = QWAdaptiveDraftPolicy(mode: .blockDiffusion(isSparseMoE: true))
        var cleanBlock = QWAdaptiveDraftPolicy(mode: .blockDiffusion(isSparseMoE: true))
        for round in 0..<6 {
            let d1 = greedyBlock.selectDraftCount(maxDrafts: 7)
            let d2 = seededBlock.selectDraftCount(maxDrafts: 7)
            let d3 = cleanBlock.selectDraftCount(maxDrafts: 7)
            XCTAssertEqual(d1, d3)
            XCTAssertEqual(d2, d3)
            let noisyMs = (round % 2 == 0) ? 900.0 : 1.0
            greedyBlock.observeRound(draftCount: d1, accepted: 1, roundMs: noisyMs, isGreedy: true)
            seededBlock.observeRound(draftCount: d2, accepted: 1, roundMs: noisyMs, isGreedy: false, hasExplicitSeed: true)
            cleanBlock.observeAcceptance(draftCount: d3, accepted: 1)
        }
        XCTAssertEqual(greedyBlock.selectDraftCount(maxDrafts: 7), cleanBlock.selectDraftCount(maxDrafts: 7))
        XCTAssertEqual(seededBlock.selectDraftCount(maxDrafts: 7), cleanBlock.selectDraftCount(maxDrafts: 7))
    }
}
