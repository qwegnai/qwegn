import XCTest
@testable import QWMTP

final class QWMTPCheckedProtocolTests: XCTestCase {
    func testCheckedReferencePreservesDecisionAndDrawLaws() throws {
        let q = [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2]]
        let p = [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2], [0.1, 0.1, 0.8]]
        for uniforms in [[0.9, 0.2, 0.5], [0.1, 0.9, 0.5], [0.1, 0.1, 0.5]] {
            let expected = QWMTP.acceptRound(draftProbs: q, targetRows: p, drafts: [0, 1], uniforms: uniforms)
            let actual = try QWMTP.checkedAcceptRound(draftProbs: q, targetRows: p, drafts: [0, 1], uniforms: uniforms)
            XCTAssertEqual(actual, expected)
            XCTAssertEqual(actual.nDrawn, actual.accepted < 2 ? actual.accepted + 2 : 3)
        }
        let all = try QWMTP.checkedAcceptRound(draftProbs: [[0.5, 0.5]],
            targetRows: [[0.5, 0.5], [0.25, 0.75]], drafts: [0], uniforms: [0.99, 0.25])
        XCTAssertEqual(all.accepted, 1)
        XCTAssertEqual(all.finalToken, 1, "inverse CDF is strict at boundaries")
    }

    func testInvalidRowsSupportAndWindowsThrow() {
        let q = [[0.5, 0.5]], p = [[0.5, 0.5], [0.5, 0.5]]
        for uniforms in [[], [0.1], [.nan, 0.1], [1, 0.1], [-0.1, 0.1]] {
            XCTAssertThrowsError(try QWMTP.checkedAcceptRound(draftProbs: q, targetRows: p,
                drafts: [0], uniforms: uniforms))
        }
        for row in [[Double.nan, 0.5], [0.1, 0.1], [-0.5, 1.5], [1]] {
            XCTAssertThrowsError(try QWMTP.checkedAcceptRound(draftProbs: [row], targetRows: p,
                drafts: [0], uniforms: [0.1, 0.1]))
        }
        XCTAssertThrowsError(try QWMTP.checkedAcceptRound(draftProbs: [[0, 1]], targetRows: p,
            drafts: [0], uniforms: [0.1, 0.1]))
        XCTAssertThrowsError(try QWMTP.checkedAcceptRound(draftProbs: q, targetRows: p,
            drafts: [2], uniforms: [0.1, 0.1]))
        XCTAssertThrowsError(try QWMTP.checkedAcceptRound(draftProbs: q, targetRows: p,
            drafts: [0], uniforms: [0.1, 0.1], offset: -1))
    }

    func testRejectionIdentityOnSmallNormalizedTables() throws {
        let q = [0.6, 0.3, 0.1], p = [0.2, 0.4, 0.4]
        let rejectedMass = zip(p, q).reduce(0.0) { $0 + max(0, $1.1 - $1.0) }
        let positive = zip(p, q).map { max(0, $0 - $1) }
        let mass = positive.reduce(0, +)
        for i in p.indices {
            let accepted = min(q[i], p[i])
            XCTAssertEqual(accepted + rejectedMass * positive[i] / mass, p[i], accuracy: 1e-15)
        }
        let result = try QWMTP.checkedAcceptRound(draftProbs: [q], targetRows: [p, p],
            drafts: [0], uniforms: [0.9, 0.4])
        XCTAssertEqual(result.accepted, 0)
        XCTAssertEqual(result.nDrawn, 2)
        XCTAssertEqual(result.finalToken, 2)
    }
}
