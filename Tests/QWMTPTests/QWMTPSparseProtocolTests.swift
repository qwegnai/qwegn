import XCTest
import QWHead
@testable import QWMTP

final class QWMTPSparseProtocolTests: XCTestCase {
    private func distribution(_ logits: [Float], topK: Int = 3) throws -> QWSamplingDistribution {
        try QWSampler(config: QWSamplerConfig(temperature: 1, topK: topK, topP: 1,
                                             uniformVersion: .upper24)).distribution(logits: logits)
    }

    private func dense(_ row: QWSamplingDistribution, vocabulary: Int) -> [Double] {
        var result = [Double](repeating: 0, count: vocabulary)
        for (token, value) in zip(row.survivorIds, row.survivorProbs) { result[token] = value }
        return result
    }

    func testSparseMatchesDenseForRejectionAndBonus() throws {
        let q = try [distribution([3, -9, 1, -9, 0]), distribution([-9, 0, -9, 3, 1])]
        let p = try [distribution([0, -9, 1, -9, 3]), distribution([-9, 0, -9, 3, 1]),
                     distribution([1, -9, 0, -9, 3])]
        for n in 1...2 {
            let drafts = Array([0, 3].prefix(n))
            for accept in [0.0, 0.13, 0.77, 0.99] {
                for commit in [0.0, 0.21, 0.61, 0.99, Double(Float(1).nextDown)] {
                    let uniforms = [accept, commit, 0.73]
                    let sparse = try QWMTP.checkedAcceptRound(draftDistributions: Array(q.prefix(n)),
                        targetDistributions: Array(p.prefix(n + 1)), drafts: drafts, uniforms: uniforms)
                    let reference = try QWMTP.checkedAcceptRound(draftProbs: q.prefix(n).map { dense($0, vocabulary: 5) },
                        targetRows: p.prefix(n + 1).map { dense($0, vocabulary: 5) },
                        drafts: drafts, uniforms: uniforms)
                    XCTAssertEqual(sparse, reference)
                }
            }
        }
    }

    func testDisjointSupportsAndInvalidRequests() throws {
        let q = try distribution([10, 0, 0], topK: 1)
        let p = try distribution([0, 0, 10], topK: 1)
        let result = try QWMTP.checkedAcceptRound(draftDistributions: [q], targetDistributions: [p, p],
            drafts: [0], uniforms: [0, 0])
        XCTAssertEqual(result.accepted, 0)
        XCTAssertEqual(result.finalToken, 2)
        XCTAssertEqual(result.nDrawn, 2)
        for uniforms in [[], [0], [Double.nan, 0], [1, 0], [-0.1, 0]] {
            XCTAssertThrowsError(try QWMTP.checkedAcceptRound(draftDistributions: [q],
                targetDistributions: [p, p], drafts: [0], uniforms: uniforms))
        }
        for token in [-1, 1, 3] {
            XCTAssertThrowsError(try QWMTP.checkedAcceptRound(draftDistributions: [q],
                targetDistributions: [p, p], drafts: [token], uniforms: [0, 0]))
        }
        XCTAssertThrowsError(try QWMTP.checkedAcceptRound(draftDistributions: [q],
            targetDistributions: [p, p], drafts: [0], uniforms: [0, 0], offset: -1))
        XCTAssertThrowsError(try QWMTP.checkedAcceptRound(draftDistributions: [],
            targetDistributions: [p], drafts: [], uniforms: [0]))
        let foreignVocabulary = try distribution([10, 0], topK: 1)
        XCTAssertThrowsError(try QWMTP.checkedAcceptRound(draftDistributions: [foreignVocabulary],
            targetDistributions: [p, p], drafts: [0], uniforms: [0, 0]))
    }

    func testEqualSupportsTinyResidualAndDrawOffset() throws {
        let q = try distribution([0, 0, 0])
        let p = try distribution([0.00001, 0, 0])
        let equal = try QWMTP.checkedAcceptRound(draftDistributions: [q, q],
            targetDistributions: [q, q, q], drafts: [0, 1], uniforms: [0.99, 0.99, 0])
        XCTAssertEqual(equal.accepted, 2)
        XCTAssertEqual(equal.finalToken, 0)
        XCTAssertEqual(equal.nDrawn, 3)
        let uniforms = [0.999999, 0.7]
        let tiny = try QWMTP.checkedAcceptRound(draftDistributions: [q], targetDistributions: [p, p],
            drafts: [1], uniforms: [Double.nan] + uniforms, offset: 1)
        let reference = try QWMTP.checkedAcceptRound(draftProbs: [dense(q, vocabulary: 3)],
            targetRows: [dense(p, vocabulary: 3), dense(p, vocabulary: 3)], drafts: [1], uniforms: uniforms)
        XCTAssertEqual(tiny, reference)
        XCTAssertEqual(tiny.accepted, 0)
        XCTAssertEqual(tiny.finalToken, 0)
    }

    func testGumbelCouplingMatchesSharedNoiseReference() throws {
        let draft: [[Float]] = [[0.25, 1.5, -0.5], [2, -1, 0.5]]
        let target: [[Float]] = [[1.25, 0.75, -0.5], [0.5, 1.5, -0.25], [-1, 2, 0.25]]
        let seed: UInt64 = 137
        let offset: UInt64 = 11
        let temperature: Float = 0.8

        func winner(_ logits: [Float], row: Int) -> Int {
            let base = offset + UInt64(row * logits.count)
            return logits.indices.max { lhs, rhs in
                let lu = max(Double(QWPhilox.u(base + UInt64(lhs), seed: seed, version: .upper24)),
                             1.0 / 16_777_216.0)
                let ru = max(Double(QWPhilox.u(base + UInt64(rhs), seed: seed, version: .upper24)),
                             1.0 / 16_777_216.0)
                let ls = Double(logits[lhs]) / Double(temperature) - log(-log(lu))
                let rs = Double(logits[rhs]) / Double(temperature) - log(-log(ru))
                return ls == rs ? lhs > rhs : ls < rs
            }!
        }

        let actual = try QWMTP.checkedGumbelCouplingRound(
            draftLogits: draft, targetLogits: target, seed: seed, offset: offset,
            temperature: temperature)
        var accepted = 0
        var expectedDrafts: [Int] = []
        for row in draft.indices {
            let draftToken = winner(draft[row], row: row)
            let targetToken = winner(target[row], row: row)
            expectedDrafts.append(draftToken)
            if draftToken != targetToken {
                XCTAssertEqual(actual, QWMTP.GumbelRoundResult(accepted: row, finalToken: targetToken,
                                                               draftTokens: expectedDrafts))
                return
            }
            accepted += 1
        }
        XCTAssertEqual(actual, QWMTP.GumbelRoundResult(accepted: accepted,
            finalToken: winner(target[accepted], row: accepted),
            draftTokens: expectedDrafts))
        XCTAssertEqual(try QWMTP.checkedGumbelCouplingRound(
            winnerPairs: [[1, 0], [2, 2]], bonusTarget: 1, vocabulary: 3),
            QWMTP.GumbelRoundResult(accepted: 0, finalToken: 0, draftTokens: [1]))
        XCTAssertThrowsError(try QWMTP.checkedGumbelCouplingRound(
            winnerPairs: [[1]], bonusTarget: 0, vocabulary: 3))
    }

    func testGumbelCouplingRejectsMalformedRowsAndCounters() {
        XCTAssertThrowsError(try QWMTP.checkedGumbelCouplingRound(
            draftLogits: [[0, 1]], targetLogits: [[0, .infinity], [0, 1]], seed: 0))
        XCTAssertThrowsError(try QWMTP.checkedGumbelCouplingRound(
            draftLogits: [[0, 1]], targetLogits: [[0, 1], [0]], seed: 0))
        XCTAssertThrowsError(try QWMTP.checkedGumbelCouplingRound(
            draftLogits: [[0, 1]], targetLogits: [[0, 1], [0, 1]], seed: 0,
            offset: .max))
        XCTAssertThrowsError(try QWMTP.checkedGumbelCouplingRound(
            draftLogits: [[0, 1]], targetLogits: [[0, 1], [0, 1]], seed: 0,
            temperature: 0))
    }

    func testGumbelCouplingUsesFilteredSamplerSupports() throws {
        let drafts = try [
            distribution([4, -3, 1, -4], topK: 3),
            distribution([-2, 3, -1, 1], topK: 2),
            distribution([0, -2, 4, 1], topK: 4),
            distribution([3, -1, 0, 2], topK: 3),
            distribution([-2, 2, 0, 1], topK: 4)
        ]
        let targets = try [
            distribution([3, 0, 1, -2], topK: 3),
            distribution([0, 3, 1, -2], topK: 3),
            distribution([1, -1, 3, 0], topK: 2),
            distribution([2, 0, 1, 3], topK: 4),
            distribution([0, 1, 2, 3], topK: 2),
            distribution([2, 0, 3, 1], topK: 4)
        ]
        let seed: UInt64 = 713, offset: UInt64 = 41
        let result = try QWMTP.checkedGumbelCouplingRound(draftDistributions: drafts,
            targetDistributions: targets, seed: seed, offset: offset)

        func winner(_ row: QWSamplingDistribution, rowIndex: Int) -> Int {
            let base = offset + UInt64(rowIndex * row.vocabulary)
            return zip(row.survivorIds, row.survivorProbs).max { lhs, rhs in
                func score(_ item: (Int, Double)) -> Double {
                    let u = max(Double(QWPhilox.u(base + UInt64(item.0), seed: seed, version: .upper24)),
                                1.0 / 16_777_216.0)
                    return log(item.1) - log(-log(u))
                }
                let left = score(lhs), right = score(rhs)
                return left == right ? lhs.0 > rhs.0 : left < right
            }!.0
        }
        var pairs: [[Int]] = []
        for row in drafts.indices {
            pairs.append([winner(drafts[row], rowIndex: row), winner(targets[row], rowIndex: row)])
        }
        let expected = try QWMTP.checkedGumbelCouplingRound(winnerPairs: pairs,
            bonusTarget: winner(targets[5], rowIndex: 5), vocabulary: 4)
        XCTAssertEqual(result, expected)
        XCTAssertThrowsError(try QWMTP.checkedGumbelCouplingRound(
            draftDistributions: [drafts[0]], targetDistributions: [targets[0], targets[1]],
            seed: seed, offset: .max))
        let otherVocabulary = try distribution([3, 1], topK: 2)
        XCTAssertThrowsError(try QWMTP.checkedGumbelCouplingRound(
            draftDistributions: [otherVocabulary], targetDistributions: [targets[0], targets[1]],
            seed: seed))
    }

    func testGumbelWinnerPreservesFilteredCategoricalMarginal() throws {
        let row = try distribution([0.7, 0.2, -0.4, -2], topK: 4)
        var counts = [Int](repeating: 0, count: row.vocabulary)
        let samples = 4096
        for seed in 0..<samples {
            counts[try QWMTP.checkedGumbelWinner(distribution: row, seed: UInt64(seed), offset: 19)] += 1
        }
        var chiSquared = 0.0
        for (token, probability) in zip(row.survivorIds, row.survivorProbs) {
            let expected = Double(samples) * probability
            let difference = Double(counts[token]) - expected
            chiSquared += difference * difference / expected
        }
        XCTAssertLessThanOrEqual(chiSquared, 16.2663, "Gumbel-max marginal should match transformed sampler")
        XCTAssertEqual(counts.reduce(0, +), samples)
    }
}
