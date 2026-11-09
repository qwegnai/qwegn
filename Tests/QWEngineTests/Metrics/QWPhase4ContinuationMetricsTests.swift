import Foundation
import XCTest
import QWCore
@testable import QWEngine

struct QWPhase4ContinuationSample: Codable {
    let adoptionMs: Double
    let decodeMs: Double
    let stepMs: [Double]
    let stepTokenCounts: [Int]
    var roundProfiles: [QWMTPRoundProfile]? = nil

    func validate() throws {
        guard adoptionMs.isFinite, adoptionMs > 0, decodeMs.isFinite, decodeMs > 0,
              !stepMs.isEmpty, stepMs.count == stepTokenCounts.count,
              stepMs.allSatisfy({ $0.isFinite && $0 > 0 }),
              stepTokenCounts.allSatisfy({ (1...6).contains($0) }),
              stepTokenCounts.reduce(0, +) == 31 else {
            throw QWEngineDriverError.state("invalid frozen-history continuation timing")
        }
        if let roundProfiles {
            guard roundProfiles.count == stepTokenCounts.count else {
                throw QWEngineDriverError.state("continuation profile count differs from step count")
            }
            for (index, pair) in zip(roundProfiles, stepTokenCounts).enumerated() {
                let (profile, count) = pair
                guard (1...5).contains(profile.drafts), profile.retainedRows == count,
                      let accepted = profile.acceptedDrafts, (0...profile.drafts).contains(accepted),
                      count <= accepted + 1,
                      (index == roundProfiles.count - 1 || count == accepted + 1),
                      let tokens = profile.draftTokens,
                      tokens.count == profile.drafts,
                      tokens.allSatisfy({ (0..<QWModelGeometry.qwen38_27b.vocab).contains($0) }) else {
                    throw QWEngineDriverError.state("invalid continuation proposal or acceptance profile")
                }
                for stage in [profile.draft, profile.verification, profile.resolution] {
                    _ = try QWMTPStageTiming(hostMs: stage.hostMs, gpuMs: stage.gpuMs,
                                             commandBuffers: stage.commandBuffers)
                }
            }
        }
    }
}

struct QWPhase4ContinuationSummary: Codable {
    let measured: Int
    let adoptionP50Ms: Double, adoptionP95Ms: Double
    let decodeP50Ms: Double, decodeP95Ms: Double
    let usefulTokensPerSecondP50: Double
    let stepP95Ms: Double

    init(_ samples: [QWPhase4ContinuationSample]) throws {
        guard samples.count == 30 else {
            throw QWEngineDriverError.state("continuation timing requires exactly thirty measured samples")
        }
        for sample in samples { try sample.validate() }
        func percentile(_ values: [Double], _ p: Double) -> Double {
            values.sorted()[max(0, Int(ceil(Double(values.count) * p)) - 1)]
        }
        measured = samples.count
        adoptionP50Ms = percentile(samples.map(\.adoptionMs), 0.5)
        adoptionP95Ms = percentile(samples.map(\.adoptionMs), 0.95)
        decodeP50Ms = percentile(samples.map(\.decodeMs), 0.5)
        decodeP95Ms = percentile(samples.map(\.decodeMs), 0.95)
        usefulTokensPerSecondP50 = percentile(samples.map { 31000 / $0.decodeMs }, 0.5)
        stepP95Ms = percentile(samples.flatMap(\.stepMs), 0.95)
    }
}

final class QWPhase4ContinuationMetricsTests: XCTestCase {
    func testRoundProfilesPreserveAcceptanceBeforeTailTruncation() throws {
        let stage = try QWMTPStageTiming(hostMs: 3, gpuMs: 2, commandBuffers: 1)
        let round = QWMTPRoundProfile(drafts: 2, retainedRows: 1,
            draft: stage, verification: stage, resolution: stage,
            draftTokens: [11, 12], acceptedDrafts: 2)
        let rejected = QWMTPRoundProfile(drafts: 2, retainedRows: 1,
            draft: stage, verification: stage, resolution: stage,
            draftTokens: [11, 12], acceptedDrafts: 0)
        let sample = QWPhase4ContinuationSample(adoptionMs: 2, decodeMs: 1000,
            stepMs: Array(repeating: 9, count: 31), stepTokenCounts: Array(repeating: 1, count: 31),
            roundProfiles: Array(repeating: rejected, count: 30) + [round])
        XCTAssertNoThrow(try sample.validate())
        let decoded = try JSONDecoder().decode(QWPhase4ContinuationSample.self,
            from: JSONEncoder().encode(sample))
        XCTAssertEqual(decoded.roundProfiles?.last?.acceptedDrafts, 2)
        XCTAssertEqual(decoded.roundProfiles?.last?.draftTokens, [11, 12])
        XCTAssertThrowsError(try QWPhase4ContinuationSample(adoptionMs: 2, decodeMs: 1000,
            stepMs: sample.stepMs, stepTokenCounts: sample.stepTokenCounts,
            roundProfiles: [round]).validate())
        let invalid = QWMTPRoundProfile(drafts: 2, retainedRows: 1,
            draft: stage, verification: stage, resolution: stage,
            draftTokens: [11], acceptedDrafts: 3)
        XCTAssertThrowsError(try QWPhase4ContinuationSample(adoptionMs: 2, decodeMs: 1000,
            stepMs: sample.stepMs, stepTokenCounts: sample.stepTokenCounts,
            roundProfiles: Array(repeating: invalid, count: 31)).validate())
    }

    func testCachedPendingTokenIsExcludedAndExactSamplesRequired() throws {
        let sample = QWPhase4ContinuationSample(adoptionMs: 2, decodeMs: 1000,
            stepMs: Array(repeating: 1000.0 / 31, count: 31),
            stepTokenCounts: Array(repeating: 1, count: 31))
        let summary = try QWPhase4ContinuationSummary(Array(repeating: sample, count: 30))
        XCTAssertEqual(summary.usefulTokensPerSecondP50, 31)
        XCTAssertEqual(summary.adoptionP50Ms, 2)
        XCTAssertThrowsError(try QWPhase4ContinuationSummary(Array(repeating: sample, count: 29)))
        XCTAssertThrowsError(try QWPhase4ContinuationSample(adoptionMs: 0, decodeMs: 1000,
            stepMs: sample.stepMs, stepTokenCounts: sample.stepTokenCounts).validate())
        XCTAssertThrowsError(try QWPhase4ContinuationSample(adoptionMs: 1, decodeMs: 1000,
            stepMs: Array(repeating: 1, count: 32), stepTokenCounts: Array(repeating: 1, count: 32)).validate())
    }
}
