import XCTest
@testable import QWHead

final class QWConstrainedSamplingTests: XCTestCase {
    func testGreedyPicksBestAllowedWithoutConsumingDraw() throws {
        let sampler = QWSampler(config: QWSamplerConfig(temperature: 0, topK: 1, topP: 1))
        let logits: [Float] = [9, 1, 5, 5, 7]
        let token = try logits.withUnsafeBufferPointer { try sampler.constrainedToken(logits: $0, allowed: [1, 2, 3]) }
        XCTAssertEqual(token, 2, "ties go to the lowest allowed id")
        XCTAssertEqual(sampler.drawsConsumed, 0)
    }

    func testGreedyAppliesRepetitionPenalty() throws {
        let sampler = QWSampler(config: QWSamplerConfig(temperature: 0, topK: 1, topP: 1, repetitionPenalty: 2))
        let logits: [Float] = [0, 6, 4]
        let token = try logits.withUnsafeBufferPointer {
            try sampler.constrainedToken(logits: $0, allowed: [1, 2], repetitionHistory: [1])
        }
        XCTAssertEqual(token, 2)
    }

    func testSampledStaysInSupportAndConsumesOneDrawEach() throws {
        let sampler = QWSampler(config: QWSamplerConfig(temperature: 1, topK: 64, topP: 1, seed: 7))
        var logits = [Float](repeating: 0, count: 100)
        logits[0] = 50
        let allowed = [3, 10, 42, 99]
        for draw in 1...50 {
            let token = try logits.withUnsafeBufferPointer { try sampler.constrainedToken(logits: $0, allowed: allowed) }
            XCTAssertTrue(allowed.contains(token))
            XCTAssertEqual(sampler.drawsConsumed, UInt64(draw))
        }
    }

    func testSampledIsDeterministicForSeed() throws {
        let logits = (0..<64).map { Float($0 % 7) * 0.3 }
        let allowed = Array(stride(from: 1, to: 64, by: 3))
        func run() throws -> [Int] {
            let sampler = QWSampler(config: QWSamplerConfig(temperature: 0.8, topK: 8, topP: 0.9, seed: 11))
            return try (0..<20).map { _ in
                try logits.withUnsafeBufferPointer { try sampler.constrainedToken(logits: $0, allowed: allowed) }
            }
        }
        XCTAssertEqual(try run(), try run())
    }

    func testTopKOneIsTheBestAllowed() throws {
        let sampler = QWSampler(config: QWSamplerConfig(temperature: 1, topK: 1, topP: 1, seed: 3))
        let logits: [Float] = [10, 2, 3, 1]
        let token = try logits.withUnsafeBufferPointer { try sampler.constrainedToken(logits: $0, allowed: [1, 2, 3]) }
        XCTAssertEqual(token, 2)
    }

    func testRejectsBadSupport() {
        let sampler = QWSampler(config: QWSamplerConfig(temperature: 0, topK: 1, topP: 1))
        let logits: [Float] = [1, 2, 3]
        XCTAssertThrowsError(try logits.withUnsafeBufferPointer { try sampler.constrainedToken(logits: $0, allowed: []) })
        XCTAssertThrowsError(try logits.withUnsafeBufferPointer { try sampler.constrainedToken(logits: $0, allowed: [2, 1]) })
        XCTAssertThrowsError(try logits.withUnsafeBufferPointer { try sampler.constrainedToken(logits: $0, allowed: [3]) })
        let bad: [Float] = [1, .nan]
        XCTAssertThrowsError(try bad.withUnsafeBufferPointer { try sampler.constrainedToken(logits: $0, allowed: [1]) })
    }
}
