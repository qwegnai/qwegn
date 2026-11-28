// QWTrunkRuntimeSupportTests.swift — Unit tests for shared QWTrunkRuntimeSupport helpers.
import Foundation
import XCTest
import QWCore
import QWDecision
import QWHead
@testable import QWEngine

final class QWTrunkRuntimeSupportTests: XCTestCase {
    private enum TestSentinelError: Error, Equatable {
        case boom
    }

    private final class MockTokenConstraint: QWTokenConstraint, @unchecked Sendable {
        var allowed: [Int]?
        private(set) var acceptedTokens: [Int] = []

        init(allowed: [Int]?) {
            self.allowed = allowed
        }

        func allowedTokens() throws -> [Int]? { allowed }
        func accept(_ token: Int) throws { acceptedTokens.append(token) }
    }

    func testRepetitionPenaltyWindowWraparoundAndOutOfRangeGuard() {
        XCTAssertEqual(QWTrunkRuntimeSupport.repetitionPenaltyWindow, 256)
        var recent: [Int] = []
        var loopTokens: [Int] = []
        let vocab = 512

        QWTrunkRuntimeSupport.recordGeneratedTokenForPenalty(
            -1, vocab: vocab, recentGenerated: &recent, activeLoopTokens: &loopTokens
        )
        QWTrunkRuntimeSupport.recordGeneratedTokenForPenalty(
            vocab, vocab: vocab, recentGenerated: &recent, activeLoopTokens: &loopTokens
        )
        XCTAssertTrue(recent.isEmpty)
        XCTAssertTrue(loopTokens.isEmpty)

        for tok in 0..<265 {
            QWTrunkRuntimeSupport.recordGeneratedTokenForPenalty(
                tok, vocab: vocab, recentGenerated: &recent, activeLoopTokens: &loopTokens
            )
        }
        XCTAssertEqual(recent.count, 256)
        XCTAssertEqual(recent, Array(9..<265))
        XCTAssertTrue(loopTokens.isEmpty)
    }

    func testRepetitionLoopDetectionAndSlideOut() {
        var recent: [Int] = []
        var loopTokens: [Int] = []
        let vocab = 1024

        for _ in 0..<4 {
            QWTrunkRuntimeSupport.recordGeneratedTokenForPenalty(
                42, vocab: vocab, recentGenerated: &recent, activeLoopTokens: &loopTokens
            )
        }
        XCTAssertEqual(loopTokens, [42])

        // Sliding 256 distinct tokens through the window evicts the period-1 repeat.
        for tok in 100..<356 {
            QWTrunkRuntimeSupport.recordGeneratedTokenForPenalty(
                tok, vocab: vocab, recentGenerated: &recent, activeLoopTokens: &loopTokens
            )
        }
        XCTAssertEqual(recent.count, 256)
        XCTAssertTrue(loopTokens.isEmpty)
    }

    func testSampleTokenGreedyConstrainedAndStochastic() throws {
        let logits: [Float] = [5.0, 2.0, 4.0, 3.0]
        var scratch: [Int] = []

        let greedySampler = QWSampler(
            config: QWSamplerConfig(temperature: 0, topK: 1, topP: 1, repetitionPenalty: 2.0)
        )
        let unpenalized = try logits.withUnsafeBufferPointer { buf in
            try QWTrunkRuntimeSupport.sampleToken(
                sampler: greedySampler, logitsRow: buf, penaltyHistory: [],
                constraint: nil, allowedScratch: &scratch
            )
        }
        XCTAssertEqual(unpenalized, 0)

        let penalized = try logits.withUnsafeBufferPointer { buf in
            try QWTrunkRuntimeSupport.sampleToken(
                sampler: greedySampler, logitsRow: buf, penaltyHistory: [0],
                constraint: nil, allowedScratch: &scratch
            )
        }
        XCTAssertEqual(penalized, 2)

        let constrained = MockTokenConstraint(allowed: [1, 3])
        let constrainedTok = try logits.withUnsafeBufferPointer { buf in
            try QWTrunkRuntimeSupport.sampleToken(
                sampler: greedySampler, logitsRow: buf, penaltyHistory: [],
                constraint: constrained, allowedScratch: &scratch
            )
        }
        XCTAssertEqual(constrainedTok, 3)
        XCTAssertEqual(scratch, [1, 3])

        let unconstrained = MockTokenConstraint(allowed: nil)
        let fallbackTok = try logits.withUnsafeBufferPointer { buf in
            try QWTrunkRuntimeSupport.sampleToken(
                sampler: greedySampler, logitsRow: buf, penaltyHistory: [],
                constraint: unconstrained, allowedScratch: &scratch
            )
        }
        XCTAssertEqual(fallbackTok, 0)

        let stochConfig = QWSamplerConfig(
            temperature: 0.8, topK: 4, topP: 0.95, seed: 12345, repetitionPenalty: 1.0
        )
        let s1 = QWSampler(config: stochConfig)
        let s2 = QWSampler(config: stochConfig)
        let t1 = try logits.withUnsafeBufferPointer { buf in
            try QWTrunkRuntimeSupport.sampleToken(
                sampler: s1, logitsRow: buf, penaltyHistory: [],
                constraint: nil, allowedScratch: &scratch
            )
        }
        let t2 = try logits.withUnsafeBufferPointer { buf in
            try s2.sample(logits: buf, repetitionHistory: []).token
        }
        XCTAssertEqual(t1, t2)
    }

    func testExtractDecisionSlotOutputSparseFullAndOutOfRange() {
        let logits: [Float] = [1.5, -2.25, 0.75, 4.0]
        logits.withUnsafeBufferPointer { buf in
            guard let ptr = buf.baseAddress else {
                return XCTFail("expected baseAddress")
            }

            let sparse = QWTrunkRuntimeSupport.extractDecisionSlotOutput(
                logitsRow: ptr, vocab: logits.count, candidateTokenIds: [-1, 1, 3, 4, 99]
            )
            XCTAssertNil(sparse.fullLogits)
            XCTAssertEqual(sparse.tokenLogits, [1: -2.25, 3: 4.0])

            let allOutOfRange = QWTrunkRuntimeSupport.extractDecisionSlotOutput(
                logitsRow: ptr, vocab: logits.count, candidateTokenIds: [-2, 4, 10]
            )
            XCTAssertNil(allOutOfRange.fullLogits)
            XCTAssertEqual(allOutOfRange.tokenLogits, [:])

            let full = QWTrunkRuntimeSupport.extractDecisionSlotOutput(
                logitsRow: ptr, vocab: logits.count, candidateTokenIds: []
            )
            XCTAssertEqual(full.tokenLogits, [:])
            XCTAssertEqual(full.fullLogits, logits)
        }
    }

    func testWithGPUTurnArbitratorNilPresentAndThrowing() throws {
        let nilResult = QWTrunkRuntimeSupport.withGPUTurn(arbitrator: nil) { 42 }
        XCTAssertEqual(nilResult, 42)

        let arb = QWGPUTurnArbitrator()
        let activeDuringCall = QWTrunkRuntimeSupport.withGPUTurn(arbitrator: arb) {
            arb.activePrimarySteps
        }
        XCTAssertEqual(activeDuringCall, 1)
        XCTAssertEqual(arb.activePrimarySteps, 0)

        XCTAssertThrowsError(
            try QWTrunkRuntimeSupport.withGPUTurn(arbitrator: arb) { () throws -> Int in
                XCTAssertEqual(arb.activePrimarySteps, 1)
                throw TestSentinelError.boom
            }
        ) { err in
            XCTAssertEqual(err as? TestSentinelError, .boom)
        }
        XCTAssertEqual(arb.activePrimarySteps, 0)
    }

    func testFormatSpeculativeSummary() {
        XCTAssertNil(
            QWTrunkRuntimeSupport.formatSpeculativeSummary(
                label: "dflash2", accepted: 0, proposals: 0, stepTokenCounts: []
            )
        )
        XCTAssertNil(
            QWTrunkRuntimeSupport.formatSpeculativeSummary(
                label: "mtp", accepted: 1, proposals: -1, stepTokenCounts: [1, 2]
            )
        )
        XCTAssertEqual(
            QWTrunkRuntimeSupport.formatSpeculativeSummary(
                label: "dflash2", accepted: 3, proposals: 5, stepTokenCounts: [2, 3, 1]
            ),
            "dflash2=3/5, rounds=3, mean=2.00 tok/rnd"
        )
        XCTAssertEqual(
            QWTrunkRuntimeSupport.formatSpeculativeSummary(
                label: "mtp", accepted: 1, proposals: 2, stepTokenCounts: []
            ),
            "mtp=1/2, rounds=1, mean=0.00 tok/rnd"
        )
    }
}
