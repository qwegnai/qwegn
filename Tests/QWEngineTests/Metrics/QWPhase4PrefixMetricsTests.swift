import Darwin
import Foundation
import XCTest
@testable import QWEngine

struct QWPhase4PrefixSample: Codable {
    let wallMs: [Double]
    let ttftMs: [Double]
    let rssBytes: [UInt64]
    let chargedBytes: Int

    func validate() throws {
        guard wallMs.count == 4, ttftMs.count == 4, rssBytes.count == 4,
              wallMs.allSatisfy({ $0.isFinite && $0 > 0 }),
              ttftMs.allSatisfy({ $0.isFinite && $0 > 0 }),
              zip(wallMs, ttftMs).allSatisfy({ pair in pair.0 >= pair.1 }),
              rssBytes.allSatisfy({ $0 > 0 }), chargedBytes > 0, chargedBytes <= 512 << 20 else {
            throw QWEngineDriverError.state("invalid prefix-cache sample")
        }
    }

    static func rssBytesNow() throws -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard status == KERN_SUCCESS else {
            throw QWEngineDriverError.state("prefix-cache RSS observation failed: \(status)")
        }
        return UInt64(info.resident_size)
    }
}

struct QWPhase4PrefixSummary: Codable {
    let measured: Int
    let wallP50Ms: [Double], wallP95Ms: [Double]
    let ttftP50Ms: [Double], ttftP95Ms: [Double]
    let rssPeakBytes: [UInt64]
    let chargedPeakBytes: Int
    let freshWallDrift: Double, freshTTFTDrift: Double
    let hitWallSpeedup: Double, hitTTFTSpeedup: Double

    init(_ samples: [QWPhase4PrefixSample]) throws {
        guard samples.count >= 30 else { throw QWEngineDriverError.state("prefix comparison requires thirty samples") }
        for sample in samples { try sample.validate() }
        func percentile(_ values: [Double], _ p: Double) -> Double {
            values.sorted()[max(0, Int(ceil(Double(values.count) * p)) - 1)]
        }
        measured = samples.count
        wallP50Ms = (0..<4).map { index in percentile(samples.map { $0.wallMs[index] }, 0.5) }
        wallP95Ms = (0..<4).map { index in percentile(samples.map { $0.wallMs[index] }, 0.95) }
        ttftP50Ms = (0..<4).map { index in percentile(samples.map { $0.ttftMs[index] }, 0.5) }
        ttftP95Ms = (0..<4).map { index in percentile(samples.map { $0.ttftMs[index] }, 0.95) }
        rssPeakBytes = (0..<4).map { index in samples.map { $0.rssBytes[index] }.reduce(0, max) }
        chargedPeakBytes = samples.map(\.chargedBytes).reduce(0, max)
        freshWallDrift = abs(wallP50Ms[3] / wallP50Ms[0] - 1)
        freshTTFTDrift = abs(ttftP50Ms[3] / ttftP50Ms[0] - 1)
        hitWallSpeedup = min(wallP50Ms[0], wallP50Ms[3]) / wallP50Ms[2]
        hitTTFTSpeedup = min(ttftP50Ms[0], ttftP50Ms[3]) / ttftP50Ms[2]
    }
}

final class QWPhase4PrefixMetricsTests: XCTestCase {
    func testExactFourModesAndMeasuredThreshold() throws {
        let sample = QWPhase4PrefixSample(wallMs: [100, 110, 10, 102], ttftMs: [99, 100, 9, 100],
                                        rssBytes: [1000, 2000, 2000, 1500], chargedBytes: 100)
        let summary = try QWPhase4PrefixSummary(Array(repeating: sample, count: 30))
        XCTAssertEqual(summary.wallP50Ms, sample.wallMs)
        XCTAssertEqual(summary.hitWallSpeedup, 10)
        XCTAssertEqual(summary.freshWallDrift, 0.02, accuracy: 1e-12)
        XCTAssertEqual(summary.rssPeakBytes, sample.rssBytes)
        XCTAssertThrowsError(try QWPhase4PrefixSummary(Array(repeating: sample, count: 29)))
        XCTAssertThrowsError(try QWPhase4PrefixSample(wallMs: [1], ttftMs: [1], rssBytes: [1],
                                                   chargedBytes: 1).validate())
        XCTAssertThrowsError(try QWPhase4PrefixSample(wallMs: [1, 2, 3, 4], ttftMs: [1, 3, 3, 4],
                                                   rssBytes: [1, 1, 1, 1], chargedBytes: 1).validate())
        XCTAssertGreaterThan(try QWPhase4PrefixSample.rssBytesNow(), 0)
    }
}
