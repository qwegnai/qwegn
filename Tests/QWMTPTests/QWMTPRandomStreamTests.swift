import XCTest
import QWHead
@testable import QWMTP

final class QWMTPRandomStreamTests: XCTestCase {
    private func distribution(_ token: Int) throws -> QWSamplingDistribution {
        var logits: [Float] = [0, 0, 0]
        logits[token] = 10
        return try QWSampler(config: QWSamplerConfig(temperature: 1, topK: 1, topP: 1,
                                                    uniformVersion: .upper24)).distribution(logits: logits)
    }

    func testProposalThenDecisionOrderAndActualConsumption() throws {
        let q0 = try distribution(0), q1 = try distribution(1), other = try distribution(2)
        let bonus = try QWSampler(config: QWSamplerConfig(temperature: 1, topK: 3, topP: 1))
            .distribution(logits: [0, 1, 2])
        for targets in [[other, q1, bonus], [q0, other, bonus], [q0, q1, bonus]] {
            let stream = QWMTPRandomStream(seed: 137)
            try stream.beginRound(drafts: 2)
            let proposals = try [stream.sampleProposal(q0), stream.sampleProposal(q1)]
            XCTAssertEqual(proposals.map(\.token), [0, 1])
            XCTAssertEqual(stream.drawsConsumed, 2)
            let uniforms = (2...4).map { Double(QWPhilox.u(UInt64($0), seed: 137, version: .upper24)) }
            let expected = try QWMTP.checkedAcceptRound(draftDistributions: [q0, q1],
                targetDistributions: targets, drafts: [0, 1], uniforms: uniforms)
            let result = try stream.accept(proposals: proposals, targetDistributions: targets)
            XCTAssertEqual(result, expected)
            XCTAssertEqual(stream.drawsConsumed, 2 + UInt64(result.nDrawn))
            XCTAssertFalse(stream.isRoundActive)
            XCTAssertThrowsError(try stream.accept(proposals: proposals, targetDistributions: targets))
            XCTAssertEqual(stream.drawsConsumed, 2 + UInt64(result.nDrawn))
        }
    }

    func testReceiptsRejectForeignReorderedStaleAndPartialWindows() throws {
        let q = try distribution(0)
        let stream = QWMTPRandomStream(seed: 7), foreign = QWMTPRandomStream(seed: 7)
        XCTAssertThrowsError(try stream.sampleProposal(q))
        try stream.beginRound(drafts: 2)
        try foreign.beginRound(drafts: 2)
        let first = try stream.sampleProposal(q)
        XCTAssertThrowsError(try stream.accept(proposals: [first], targetDistributions: [q, q]))
        let second = try stream.sampleProposal(q)
        let wrong = try [foreign.sampleProposal(q), foreign.sampleProposal(q)]
        XCTAssertThrowsError(try stream.beginRound(drafts: 1))
        XCTAssertThrowsError(try stream.sampleProposal(q))
        XCTAssertThrowsError(try stream.accept(proposals: wrong, targetDistributions: [q, q, q]))
        XCTAssertThrowsError(try stream.accept(proposals: [second, first], targetDistributions: [q, q, q]))
        XCTAssertEqual(stream.drawsConsumed, 2)
        _ = try stream.accept(proposals: [first, second], targetDistributions: [q, q, q])
        XCTAssertEqual(stream.drawsConsumed, 5)
        try stream.beginRound(drafts: 2)
        _ = try stream.sampleProposal(q); _ = try stream.sampleProposal(q)
        XCTAssertThrowsError(try stream.accept(proposals: [first, second], targetDistributions: [q, q, q]))
        XCTAssertEqual(stream.drawsConsumed, 7)
        try stream.discardRound()
    }

    func testDiscardBurnsOnlyExistingDrawsAndInvalidDecisionCanRetry() throws {
        let q = try distribution(0)
        let stream = QWMTPRandomStream(seed: 17)
        try stream.beginRound(drafts: 2)
        _ = try stream.sampleProposal(q)
        try stream.discardRound()
        XCTAssertEqual(stream.drawsConsumed, 1)
        XCTAssertThrowsError(try stream.discardRound())
        try stream.beginRound(drafts: 1)
        let proposal = try stream.sampleProposal(q)
        XCTAssertThrowsError(try stream.accept(proposals: [proposal], targetDistributions: []))
        XCTAssertEqual(stream.drawsConsumed, 2)
        XCTAssertTrue(stream.isRoundActive)
        let decision = try stream.accept(proposals: [proposal], targetDistributions: [q, q])
        XCTAssertEqual(decision.accepted, 1)
        XCTAssertEqual(stream.drawsConsumed, 4)
        try stream.beginRound(drafts: 1)
        try stream.discardRound()
        XCTAssertEqual(stream.drawsConsumed, 4)
    }

    func testCounterCapacityRefusesBeforeAnyRoundDraw() throws {
        let q = try distribution(0)
        for count in 1...2 {
            let maximum = UInt64(2 * count + 1)
            let refused = QWMTPRandomStream(seed: 0, drawIndex: UInt64.max - maximum + 1)
            XCTAssertThrowsError(try refused.beginRound(drafts: count))
            XCTAssertEqual(refused.drawsConsumed, UInt64.max - maximum + 1)
            XCTAssertFalse(refused.isRoundActive)
            let last = QWMTPRandomStream(seed: 0, drawIndex: UInt64.max - maximum)
            try last.beginRound(drafts: count)
            var proposals: [QWMTPProposal] = []
            for _ in 0..<count { proposals.append(try last.sampleProposal(q)) }
            _ = try last.accept(proposals: proposals, targetDistributions: Array(repeating: q, count: count + 1))
            XCTAssertEqual(last.drawsConsumed, UInt64.max)
            XCTAssertThrowsError(try last.beginRound(drafts: count))
        }
        for count in [0, -1, 3, Int.max] {
            XCTAssertThrowsError(try QWMTPRandomStream(seed: 0).beginRound(drafts: count))
        }

    }

    func testGumbelRoundSharesRowCountersAndBurnsCompleteWindow() throws {
        let rows = try [
            QWSampler(config: QWSamplerConfig(temperature: 1, topK: 3, topP: 1,
                                               uniformVersion: .upper24)).distribution(logits: [0, 1, 2]),
            QWSampler(config: QWSamplerConfig(temperature: 1, topK: 2, topP: 1,
                                               uniformVersion: .upper24)).distribution(logits: [2, 0, 1]),
            QWSampler(config: QWSamplerConfig(temperature: 1, topK: 3, topP: 1,
                                               uniformVersion: .upper24)).distribution(logits: [1, 2, 0])
        ]
        let stream = QWMTPRandomStream(seed: 137, drawIndex: 11)
        try stream.beginGumbelRound(drafts: 2, vocabulary: 3)
        let proposals = try [stream.sampleGumbelProposal(rows[0]), stream.sampleGumbelProposal(rows[1])]
        XCTAssertEqual(proposals.map(\.token), [
            try QWMTP.checkedGumbelWinner(distribution: rows[0], seed: 137, offset: 11),
            try QWMTP.checkedGumbelWinner(distribution: rows[1], seed: 137, offset: 14)
        ])
        XCTAssertEqual(stream.drawsConsumed, 17)
        let targets = [rows[2], rows[0], rows[1]]
        let expected = try QWMTP.checkedGumbelCouplingRound(draftDistributions: rows.prefix(2).map { $0 },
            targetDistributions: targets, seed: 137, offset: 11)
        let result = try stream.resolveGumbelRound(proposals: proposals, targetDistributions: targets)
        XCTAssertEqual(result, expected)
        XCTAssertEqual(stream.drawsConsumed, 20)
        XCTAssertFalse(stream.isRoundActive)
        XCTAssertThrowsError(try stream.resolveGumbelRound(proposals: proposals, targetDistributions: targets))

        try stream.beginGumbelRound(drafts: 5, vocabulary: 3)
        let partial = try stream.sampleGumbelProposal(rows[0])
        stream.cancelActiveRound()
        XCTAssertEqual(stream.drawsConsumed, 23, "successfully sampled row counters stay burned on cancellation")
        XCTAssertThrowsError(try stream.beginGumbelRound(drafts: 6, vocabulary: 3))
        XCTAssertThrowsError(try stream.beginGumbelRound(drafts: 1, vocabulary: 0))
        XCTAssertEqual(partial.token, try QWMTP.checkedGumbelWinner(
            distribution: rows[0], seed: 137, offset: 20))
    }

    func testTargetDrawFrozenCloneAndActiveRefusals() throws {
        let q = try QWSampler(config: QWSamplerConfig(temperature: 1, topK: 3, topP: 1))
            .distribution(logits: [0, 1, 2])
        let stream = QWMTPRandomStream(seed: 117)
        XCTAssertEqual(try stream.sampleTarget(q), try q.sampleToken(
            uniform: Double(QWPhilox.u(0, seed: 117, version: .upper24))))
        let frozen = try stream.frozenState()
        let clone = QWMTPRandomStream(cloning: frozen)
        XCTAssertEqual(try stream.sampleTarget(q), try clone.sampleTarget(q))
        XCTAssertEqual(stream.drawsConsumed, 2)
        try stream.beginRound(drafts: 1)
        XCTAssertThrowsError(try stream.sampleTarget(q))
        XCTAssertThrowsError(try stream.frozenState())
        _ = try stream.sampleProposal(q)
        stream.cancelActiveRound()
        stream.cancelActiveRound()
        XCTAssertEqual(stream.drawsConsumed, 3)
        XCTAssertFalse(stream.isRoundActive)
        XCTAssertEqual(frozen.drawsConsumed, 1)
        XCTAssertThrowsError(try QWMTPRandomStream(seed: 0, drawIndex: UInt64.max).sampleTarget(q))
    }
}
