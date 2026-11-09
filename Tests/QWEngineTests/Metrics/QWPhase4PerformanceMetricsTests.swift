import Foundation
import Darwin
import XCTest
import QWCli
@testable import QWEngine

final class QWPhase4PerformanceReport {
    private let handle: FileHandle
    private let recordPrefix: String

    init(path: String, recordPrefix: String = "PHASE4_PERFORMANCE") throws {
        guard !recordPrefix.isEmpty,
              recordPrefix.utf8.allSatisfy({ (65...90).contains($0) || (48...57).contains($0) || $0 == 95 }) else {
            throw QWEngineDriverError.state("invalid Phase 4 report prefix")
        }
        guard path.hasPrefix("/") else {
            throw QWEngineDriverError.state("Phase 4 report must be an absolute new file path")
        }
        let descriptor = open(path, O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else {
            throw QWEngineDriverError.state("cannot create exclusive Phase 4 report: \(String(cString: strerror(errno)))")
        }
        handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        self.recordPrefix = recordPrefix
    }

    func write<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(value)
        guard let json = String(data: data, encoding: .utf8) else {
            throw QWEngineDriverError.state("Phase 4 report encoding is not UTF8")
        }
        let line = "\(recordPrefix) \(json)"
        try handle.write(contentsOf: Data((line + "\n").utf8))
        return line
    }

    func close() throws {
        try handle.synchronize()
        try handle.close()
    }

    func echo(_ line: String) throws {
        let data = Data((line + "\n").utf8)
        guard line.hasPrefix(recordPrefix + " "), !line.contains("\n"), data.count <= 4096 else {
            throw QWEngineDriverError.state("only short complete report records can be echoed")
        }
        guard fflush(stdout) == 0 else {
            throw QWEngineDriverError.state("cannot flush report stdout: \(String(cString: strerror(errno)))")
        }
        // Swift print can split a record across writes while XCTest writes stderr.
        var written: Int
        repeat {
            written = data.withUnsafeBytes { Darwin.write(STDOUT_FILENO, $0.baseAddress, $0.count) }
        } while written < 0 && errno == EINTR
        guard written == data.count else {
            throw QWEngineDriverError.state("report stdout write failed or was partial: \(written)/\(data.count)")
        }
    }
}

struct QWPhase4PerformanceSample: Codable {
    let totalMs: Double
    let prefillMs: Double
    let decodeMs: Double
    let emittedTokens: Int
    let stepMs: [Double]
    let stepTokenCounts: [Int]

    func validate() throws {
        guard emittedTokens > 1, totalMs.isFinite, totalMs > 0,
              prefillMs.isFinite, prefillMs > 0, decodeMs.isFinite, decodeMs > 0,
              !stepMs.isEmpty, stepMs.count == stepTokenCounts.count,
              stepMs.allSatisfy({ $0.isFinite && $0 > 0 }),
              stepTokenCounts.allSatisfy({ $0 > 0 }),
              stepTokenCounts.reduce(0, +) == emittedTokens - 1 else {
            throw QWEngineDriverError.state("invalid Phase 4 performance sample")
        }
    }
}

struct QWPhase4PerformanceSummary: Codable {
    let samples: Int
    let totalP50Ms: Double
    let totalP95Ms: Double
    let prefillP50Ms: Double
    let prefillP95Ms: Double
    let decodeP50Ms: Double
    let decodeP95Ms: Double
    let stepP50Ms: Double
    let stepP95Ms: Double
    let decodeTokensPerSecondP50: Double
    let decodeTokensPerSecondP95: Double
    let requestTokensPerSecondP50: Double
    let usefulTokensPerStep: Double

    init(_ values: [QWPhase4PerformanceSample]) throws {
        guard !values.isEmpty else { throw QWEngineDriverError.state("empty Phase 4 performance samples") }
        for value in values { try value.validate() }
        func percentile(_ numbers: [Double], _ q: Double) -> Double {
            numbers.sorted()[max(0, Int(ceil(Double(numbers.count) * q)) - 1)]
        }
        samples = values.count
        totalP50Ms = percentile(values.map(\.totalMs), 0.50)
        totalP95Ms = percentile(values.map(\.totalMs), 0.95)
        prefillP50Ms = percentile(values.map(\.prefillMs), 0.50)
        prefillP95Ms = percentile(values.map(\.prefillMs), 0.95)
        decodeP50Ms = percentile(values.map(\.decodeMs), 0.50)
        decodeP95Ms = percentile(values.map(\.decodeMs), 0.95)
        stepP50Ms = percentile(values.flatMap(\.stepMs), 0.50)
        stepP95Ms = percentile(values.flatMap(\.stepMs), 0.95)
        decodeTokensPerSecondP50 = percentile(values.map { Double($0.emittedTokens - 1) * 1000 / $0.decodeMs }, 0.50)
        decodeTokensPerSecondP95 = percentile(values.map { Double($0.emittedTokens - 1) * 1000 / $0.decodeMs }, 0.95)
        requestTokensPerSecondP50 = percentile(values.map { Double($0.emittedTokens) * 1000 / $0.totalMs }, 0.50)
        usefulTokensPerStep = Double(values.reduce(0) { $0 + $1.emittedTokens - 1 }) /
            Double(values.reduce(0) { $0 + $1.stepMs.count })
    }
}

final class QWPhase4PerformanceMetricsTests: XCTestCase {
    func testSeparateProfileRecordsAndInvalidPrefixes() throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-phase4-profile-\(UUID().uuidString).jsonl").path
        for prefix in ["", "bad", "PROFILE\nINJECTED", "PROFILE RECORD"] {
            XCTAssertThrowsError(try QWPhase4PerformanceReport(path: path, recordPrefix: prefix))
            XCTAssertFalse(FileManager.default.fileExists(atPath: path))
        }
        defer {
            do { try FileManager.default.removeItem(atPath: path) }
            catch { XCTFail("profile fixture cleanup failed: \(error)") }
        }
        let writer = try QWPhase4PerformanceReport(path: path, recordPrefix: "PHASE_PROFILE")
        let line = try writer.write(["kind": "completed"])
        try writer.close()
        XCTAssertTrue(line.hasPrefix("PHASE_PROFILE "))
        XCTAssertEqual(try String(contentsOfFile: path, encoding: .utf8), line + "\n")
    }

    func testReportIsDedicatedExclusiveAndPreservesLongRecords() throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-phase4-report-\(UUID().uuidString).jsonl").path
        defer {
            do { try FileManager.default.removeItem(atPath: path) }
            catch { XCTFail("report fixture cleanup failed: \(error)") }
        }
        let writer = try QWPhase4PerformanceReport(path: path)
        XCTAssertThrowsError(try QWPhase4PerformanceReport(path: path))
        XCTAssertThrowsError(try QWPhase4PerformanceReport(path: "relative.jsonl"))
        let payload = ["payload": String(repeating: "a", count: 100_000)]
        let first = try writer.write(payload)
        XCTAssertThrowsError(try writer.echo(first), "long records stay in the exclusive report")
        XCTAssertThrowsError(try writer.echo("INVALID"))
        let second = try writer.write(["kind": "completed"])
        try writer.close()
        XCTAssertEqual(try String(contentsOfFile: path, encoding: .utf8), first + "\n" + second + "\n")
        let json = Data(first.dropFirst("PHASE4_PERFORMANCE ".count).utf8)
        XCTAssertEqual(try JSONDecoder().decode([String: String].self, from: json), payload)
    }

    func testUsefulThroughputExcludesFirstTokenAndPreservesRoundLatency() throws {
        let sample = QWPhase4PerformanceSample(totalMs: 1500, prefillMs: 500, decodeMs: 1000,
            emittedTokens: 4, stepMs: [600, 400], stepTokenCounts: [2, 1])
        let summary = try QWPhase4PerformanceSummary([sample])
        XCTAssertEqual(summary.decodeTokensPerSecondP50, 3)
        XCTAssertEqual(summary.requestTokensPerSecondP50, 4 / 1.5)
        XCTAssertEqual(summary.usefulTokensPerStep, 1.5)
        XCTAssertEqual(summary.stepP95Ms, 600)
        XCTAssertEqual(summary.samples, 1)
        let decoded = try JSONDecoder().decode(QWPhase4PerformanceSummary.self,
            from: JSONEncoder().encode(summary))
        XCTAssertEqual(decoded.decodeTokensPerSecondP50, 3)
    }

    func testNearestRankLatencyPercentiles() throws {
        let values = (1...30).map { index in
            QWPhase4PerformanceSample(totalMs: Double(index + 1), prefillMs: 1,
                decodeMs: Double(index), emittedTokens: 2,
                stepMs: [Double(index)], stepTokenCounts: [1])
        }
        let summary = try QWPhase4PerformanceSummary(values)
        XCTAssertEqual(summary.decodeP50Ms, 15)
        XCTAssertEqual(summary.decodeP95Ms, 29)
    }

    func testRejectsInvalidTimingAndOutputAccounting() throws {
        XCTAssertThrowsError(try QWPhase4PerformanceSummary([]))
        for sample in [
            QWPhase4PerformanceSample(totalMs: .nan, prefillMs: 1, decodeMs: 1,
                emittedTokens: 2, stepMs: [1], stepTokenCounts: [1]),
            QWPhase4PerformanceSample(totalMs: 2, prefillMs: 1, decodeMs: 0,
                emittedTokens: 2, stepMs: [1], stepTokenCounts: [1]),
            QWPhase4PerformanceSample(totalMs: 2, prefillMs: 1, decodeMs: 1,
                emittedTokens: 4, stepMs: [1], stepTokenCounts: [1])
        ] {
            XCTAssertThrowsError(try QWPhase4PerformanceSummary([sample]))
        }
    }
}
