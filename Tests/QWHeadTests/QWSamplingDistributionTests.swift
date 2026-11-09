import XCTest
import Metal
import QWHAL
@testable import QWHead

final class QWSamplingDistributionTests: XCTestCase {
    func testBorrowedRowsAndScratchReuseKeepReceiptsImmutable() throws {
        let config = QWSamplerConfig(temperature: 1, topK: 64, topP: 1, uniformVersion: .upper24)
        let sampler = QWSampler(config: config)
        var logits = [Float](repeating: 0, count: 248320)
        let retained = try logits.withUnsafeBufferPointer { try sampler.distribution(logits: $0) }
        XCTAssertEqual(retained.survivorIds, Array(0..<64))
        XCTAssertEqual(retained.survivorProbs, Array(repeating: 1.0 / 64, count: 64))
        for iteration in 0..<8 {
            logits[iteration] = Float(iteration + 1)
            let borrowed = try logits.withUnsafeBufferPointer { try sampler.distribution(logits: $0) }
            let owned = try QWSampler(config: config).distribution(logits: logits)
            XCTAssertEqual(borrowed.survivorIds, owned.survivorIds)
            XCTAssertEqual(borrowed.survivorProbs.map(\.bitPattern), owned.survivorProbs.map(\.bitPattern))
            for uniform in [0, 0.5, Double(1).nextDown] {
                XCTAssertEqual(try borrowed.sampleToken(uniform: uniform), try owned.sampleToken(uniform: uniform))
            }
        }
        _ = try sampler.distribution(logits: [0, -1000])
        XCTAssertEqual(retained.survivorIds, Array(0..<64))
        XCTAssertEqual(retained.survivorProbs, Array(repeating: 1.0 / 64, count: 64))
        XCTAssertEqual(sampler.drawsConsumed, 0)
        let empty = UnsafeBufferPointer<Float>(start: nil, count: 0)
        XCTAssertThrowsError(try sampler.distribution(logits: empty))
    }

    func testVersionedEndpointsAndCounterBoundaries() {
        XCTAssertEqual(QWPhilox.uniform(UInt32.max, version: .legacyRounded32), 1)
        XCTAssertEqual(QWPhilox.uniform(0, version: .upper24), 0)
        XCTAssertEqual(QWPhilox.uniform(UInt32.max, version: .upper24), Float(1).nextDown)
        for n: UInt64 in [0, 1, UInt64(UInt32.max), UInt64(UInt32.max) + 1, UInt64.max] {
            let words = QWPhilox.philox4((UInt32(truncatingIfNeeded: n), UInt32(truncatingIfNeeded: n >> 32), 0, 0), 42)
            XCTAssertEqual(QWPhilox.u(n, seed: 42, version: .upper24),
                           QWPhilox.uniform(words.0, version: .upper24))
        }
        for word: UInt32 in [0, 1, 255, 256, 0x7fffffff, 0xffffff00, UInt32.max] {
            let uniform = QWPhilox.uniform(word, version: .upper24)
            XCTAssertGreaterThanOrEqual(uniform, 0)
            XCTAssertLessThan(uniform, 1)
        }
    }

    func testDistributionConstructionDoesNotConsumeDraws() throws {
        let config = QWSamplerConfig(temperature: 1, topK: 2, topP: 1, seed: 7)
        let sampler = QWSampler(config: config)
        let distribution = try sampler.distribution(logits: [0, 0, -1])
        XCTAssertEqual(distribution.survivorIds, [0, 1])
        XCTAssertEqual(distribution.survivorProbs, [0.5, 0.5])
        XCTAssertEqual(sampler.drawsConsumed, 0)
        XCTAssertEqual(try distribution.sample(uniform: 0).token, 0)
        XCTAssertEqual(try distribution.sample(uniform: 0.5).token, 1)
        XCTAssertEqual(try distribution.sample(uniform: 0.5, version: .legacyRounded32).token, 0)
        XCTAssertEqual(try distribution.sample(uniform: Double(1).nextDown).token, 1)
        let expected = try distribution.sample(uniform: Double(QWPhilox.u(0, seed: 7)),
                                               version: .legacyRounded32)
        let result = try sampler.sample(logits: [0, 0, -1])
        XCTAssertEqual(result.token, expected.token)
        XCTAssertEqual(result.logprob, expected.logprob)
        XCTAssertEqual(sampler.drawsConsumed, 1)
        _ = try sampler.distribution(logits: [5, 0, -1])
        XCTAssertEqual(sampler.drawsConsumed, 1)
        XCTAssertEqual(distribution.survivorProbs, [0.5, 0.5], "retained proposal must remain immutable")
    }

    func testRepetitionPenaltyAdjustsOnlyPreviouslySeenTokenLogits() throws {
        let config = QWSamplerConfig(temperature: 1, topK: 3, topP: 1,
                                     repetitionPenalty: 2)
        let sampler = QWSampler(config: config)
        let distribution = try sampler.distribution(logits: [2, 1, -2],
                                                    repetitionHistory: [0, 0])
        XCTAssertEqual(distribution.survivorProbs[0], distribution.survivorProbs[1], accuracy: 1e-6)
        let sample = try sampler.sample(logits: [2, 1, -2], repetitionHistory: [0])
        XCTAssertEqual(sample.survivorProbs[0], sample.survivorProbs[1], accuracy: 1e-6)

        let negative = try sampler.distribution(logits: [0, -2],
                                                repetitionHistory: [1])
        let unpenalized = try QWSampler(config: .init(temperature: 1, topK: 2, topP: 1))
            .distribution(logits: [0, -2])
        XCTAssertLessThan(negative.survivorProbs[1], unpenalized.survivorProbs[1])
        XCTAssertThrowsError(try sampler.distribution(logits: [0], repetitionHistory: [1]))
    }

    func testThePenalisedGreedyTokenRefusesInvalidContracts() throws {
        // The greedy path with a repetition penalty is a contract too: an
        // invalid penalty or a broken row throws, it never answers with an
        // unpenalised token.
        let sampler = QWSampler(config: .init(temperature: 1, topK: 2, topP: 1,
                                              repetitionPenalty: 2))
        let row: [Float] = [4, 3, 2]
        let token = try sampler.penalisedGreedyToken(logits: row, repetitionHistory: [2])
        XCTAssertEqual(token, 0, "the seen token is penalised, so the argmax moves")
        let broken = QWSampler(config: .init(temperature: 1, topK: 2, topP: 1,
                                              repetitionPenalty: 0))
        XCTAssertThrowsError(try broken.penalisedGreedyToken(logits: row, repetitionHistory: []))
        let nan = QWSampler(config: .init(temperature: 1, topK: 2, topP: 1,
                                          repetitionPenalty: 2))
        XCTAssertThrowsError(try nan.penalisedGreedyToken(logits: [Float(1), .nan, Float(2)], repetitionHistory: []))
        XCTAssertThrowsError(try nan.penalisedGreedyToken(logits: row, repetitionHistory: [9]))
    }

    func testInvalidInputsThrowWithoutAdvancingStream() throws {
        let sampler = QWSampler(config: .init())
        for logits: [Float] in [[], [.nan], [.infinity], [-.infinity]] {
            XCTAssertThrowsError(try sampler.sample(logits: logits))
            XCTAssertEqual(sampler.drawsConsumed, 0)
        }
        for config in [
            QWSamplerConfig(temperature: 0), .init(temperature: .nan),
            .init(temperature: Float.leastNonzeroMagnitude), .init(topK: 0),
            .init(topP: 0), .init(topP: 1.1), .init(topP: .nan),
            .init(repetitionPenalty: 0), .init(repetitionPenalty: .nan)
        ] {
            sampler.config = config
            XCTAssertThrowsError(try sampler.distribution(logits: [0, 1]))
        }
        sampler.config = .init(topK: 64, topP: 1)
        let distribution = try sampler.distribution(logits: [0, -1000])
        XCTAssertEqual(distribution.survivorIds, [0])
        XCTAssertEqual(try distribution.sample(uniform: 0).token, 0)
        for u in [Double.nan, -Double.infinity, -0.1, 1, 1.1] {
            XCTAssertThrowsError(try distribution.sample(uniform: u))
        }
        XCTAssertEqual(sampler.drawsConsumed, 0)
    }
}
