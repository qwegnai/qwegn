import XCTest
import Foundation
@testable import QWQuant
@testable import QWFlashNext

/// Checks the two row paths against the **real** n-gram table: 320,001,536
/// rows in 128 shards, in a machine-local Hugging Face snapshot, so these
/// skip when the table is not on this machine.
final class FNNgramShardRowsRealTableTests: XCTestCase {
    struct Assets {
        let snapshot: URL
        let sidecar: URL
        let index: URL
    }

    /// The real table, or nil when this machine does not have it.
    static func assets() -> Assets? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let root = home.appendingPathComponent(
            ".cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots")
        guard let items = try? FileManager.default.contentsOfDirectory(atPath: root.path),
              let first = items.sorted().first(where: { !$0.hasPrefix(".") }) else { return nil }
        let snapshot = root.appendingPathComponent(first)
        let sidecar = home.appendingPathComponent(
            ".qwg/models/Qwen3.8-Flash-Next-hq4g32/assets/ple_ngram_locations.json")
        guard FileManager.default.fileExists(atPath: sidecar.path),
              let indexName = try? FileManager.default.contentsOfDirectory(atPath: snapshot.path)
                  .sorted().first(where: { $0.hasSuffix(".index.json") }) else { return nil }
        return Assets(snapshot: snapshot, sidecar: sidecar, index: snapshot.appendingPathComponent(indexName))
    }

    /// Ids that sit on the shard boundaries and at the ends of the table.
    static let boundaryIDs = [0, 1, 159, 2_500_011, 2_500_012, 2_500_013,
                              317_501_523, 317_501_524, 320_001_535]

    /// 200 scattered ids from a seeded generator, so the sample repeats.
    static func scatteredIDs(count: Int, rowCount: Int) -> [Int] {
        var seed: UInt64 = 0x5EED_C0DE
        var ids: [Int] = []
        ids.reserveCapacity(count)
        for _ in 0..<count {
            seed = seed &+ 0x9E37_79B9_7F4A_7C15
            var z = seed
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            ids.append(Int((z ^ (z >> 31)) % UInt64(rowCount)))
        }
        return ids
    }

    func testBothPathsReturnTheSameRows() throws {
        guard let assets = Self.assets() else {
            throw XCTSkip("the real n-gram table is not on this machine")
        }
        // Scoped the way the loader scopes it: only the sidecar's shards.
        let wanted = try FNNgramShardRows.tensorNames(in: assets.sidecar)
        let container = try QWSafeTensorsContainer(url: assets.index, onlyTensors: wanted)
        let header = try FNNgramShardRows(container: container, locations: assets.sidecar)
        let pread = try FNNgramShardRows(locations: assets.sidecar, snapshot: assets.snapshot)
        XCTAssertEqual(header.headDim, 160)
        XCTAssertEqual(header.rowCount, 320_001_536)
        XCTAssertEqual(header.rowCount, pread.rowCount)
        for id in Self.boundaryIDs + Self.scatteredIDs(count: 200, rowCount: header.rowCount) {
            XCTAssertEqual(header.row(id), pread.row(id), "row \(id) differs between the two paths")
        }
    }
}

/// The row read is what Task 2.4 changed, so it is gated on the real table:
/// the header path must beat the `pread` path by a wide margin. The floor
/// is well under the measured ratio, but far above parity.
final class FNNgramShardRowsPerfTests: XCTestCase {
    func testNgramShardRowPathPerformance() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["QW_PERF_GATE"] == "1",
                          "perf gate — run via `make perf`, not the unit path")
        guard let assets = FNNgramShardRowsRealTableTests.assets() else {
            throw XCTSkip("the real n-gram table is not on this machine")
        }
        let wanted = try FNNgramShardRows.tensorNames(in: assets.sidecar)
        let setupStart = Date()
        let container = try QWSafeTensorsContainer(url: assets.index, onlyTensors: wanted)
        let setupMs = Date().timeIntervalSince(setupStart) * 1000
        let header = try FNNgramShardRows(container: container, locations: assets.sidecar)
        let pread = try FNNgramShardRows(locations: assets.sidecar, snapshot: assets.snapshot)
        let ids = FNNgramShardRowsRealTableTests.scatteredIDs(count: 200, rowCount: header.rowCount)
        var buf = [Float](repeating: 0, count: header.headDim)

        func measureNs(_ rows: FNNgramShardRows) -> Double {
            let start = Date()
            for id in ids {
                buf.withUnsafeMutableBufferPointer { p in
                    rows.copyRow(id, to: p.baseAddress!)
                }
            }
            return Date().timeIntervalSince(start) * 1e9 / Double(ids.count)
        }

        // Warm the pages, then interleave the two arms so a burst of load
        // hits both the same way.
        _ = measureNs(header)
        _ = measureNs(pread)
        var headerSamples: [Double] = []
        var preadSamples: [Double] = []
        for _ in 0..<10 {
            headerSamples.append(measureNs(header))
            preadSamples.append(measureNs(pread))
        }
        let headerNs = headerSamples.sorted()[headerSamples.count / 2]
        let preadNs = preadSamples.sorted()[preadSamples.count / 2]
        print("PERF header=\(Int(headerNs))ns pread=\(Int(preadNs))ns ratio=\(preadNs / headerNs) setup=\(String(format: "%.1f", setupMs))ms shards=\(wanted.count)")
        XCTAssertGreaterThan(preadNs / headerNs, 1.8,
                             "the header path must stay clearly faster than the pread path")
    }
}
