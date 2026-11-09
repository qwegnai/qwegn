import XCTest
import Foundation
@testable import QWQuant
@testable import QWFlashNext

final class FNNgramShardRowsTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qwg-ple-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    /// Writes a safetensors file whose PLE n-gram tensor holds `rows` rows of
    /// `headDim` BF16 values, row r filled with `Float(r)`.
    ///
    /// Returns the tensor's absolute offset in the file, which is what the
    /// locations sidecar calls `base_off`: the data region starts after the
    /// header, exactly as in a real checkpoint.
    @discardableResult
    private func writePleShard(name: String, rows: Int, headDim: Int,
                                tensorName: String = "model.ple.ple_embedding.ngram_embedding.weight",
                                dtype: String = "BF16") throws -> (url: URL, baseOffset: Int) {
        var payload = Data()
        for r in 0..<rows {
            for _ in 0..<headDim {
                let bits = UInt16(truncatingIfNeeded: Float(r).bitPattern >> 16)
                payload.append(UInt8(bits & 0xFF))
                payload.append(UInt8(bits >> 8))
            }
        }
        let header: [String: Any] = [
            tensorName: ["dtype": dtype, "shape": [rows, headDim],
                         "data_offsets": [0, payload.count]]
        ]
        let headerJSON = try JSONSerialization.data(withJSONObject: header)
        var out = Data()
        out.append(contentsOf: withUnsafeBytes(of: UInt64(headerJSON.count).littleEndian) { Array($0) })
        out.append(headerJSON)
        out.append(payload)
        let url = dir.appendingPathComponent(name)
        try out.write(to: url)
        return (url, 8 + headerJSON.count)
    }

    /// The locations sidecar for one shard.
    private func writeLocations(shards: [[String: Any]],
                                totalRows: Int? = nil) throws -> URL {
        let url = dir.appendingPathComponent("ple_ngram_locations.json")
        let rows = shards.reduce(0) { acc, s in acc + ((s["shape"] as? [Int])?.first ?? 0) }
        let obj: [String: Any] = ["shards": shards, "row_bytes": 320,
                                  "total_rows": totalRows ?? rows, "version": 1]
        try JSONSerialization.data(withJSONObject: obj).write(to: url)
        return url
    }

    /// Writes two shards and the index that binds them, and returns the
    /// index's URL and each shard's absolute tensor offset. The second
    /// shard's lane count is a parameter, so a test can make them disagree.
    private func writeTwoShardIndex(secondHeadDim: Int) throws -> (url: URL, firstBase: Int, secondBase: Int) {
        let first = try writePleShard(name: "model-00001.safetensors", rows: 4, headDim: 160,
                                      tensorName: "model.ple.ngram_embedding.shard_0.weight")
        let second = try writePleShard(name: "model-00002.safetensors", rows: 3, headDim: secondHeadDim,
                                       tensorName: "model.ple.ngram_embedding.shard_1.weight")
        let index = dir.appendingPathComponent("model.safetensors.index.json")
        let map = ["weight_map": [
            "model.ple.ngram_embedding.shard_0.weight": "model-00001.safetensors",
            "model.ple.ngram_embedding.shard_1.weight": "model-00002.safetensors",
        ]]
        try JSONSerialization.data(withJSONObject: map).write(to: index)
        return (index, first.baseOffset, second.baseOffset)
    }

    func testReadsARowStraightOutOfTheSafetensorsPayload() throws {
        let shard = try writePleShard(name: "model.safetensors", rows: 4, headDim: 160)
        let container = try QWSafeTensorsContainer(url: shard.url)
        let locations = try writeLocations(shards: [[
            "key": "model.ple.ple_embedding.ngram_embedding.weight",
            "first_row": 0, "shape": [4, 160], "row_bytes": 320, "base_off": shard.baseOffset,
        ]])
        let rows = try FNNgramShardRows(container: container, locations: locations)
        XCTAssertEqual(rows.headDim, 160)
        XCTAssertEqual(rows.rowCount, 4)
        XCTAssertEqual(rows.row(2)[0], 2.0)
        XCTAssertEqual(rows.row(3)[159], 3.0)
    }

    func testTheRowCountComesFromTheTensorShape() throws {
        let shard = try writePleShard(name: "model.safetensors", rows: 7, headDim: 160)
        let container = try QWSafeTensorsContainer(url: shard.url)
        let locations = try writeLocations(shards: [[
            "key": "model.ple.ple_embedding.ngram_embedding.weight",
            "first_row": 0, "shape": [7, 160], "row_bytes": 320, "base_off": shard.baseOffset,
        ]])
        let rows = try FNNgramShardRows(container: container, locations: locations)
        XCTAssertEqual(rows.rowCount, 7)
    }

    func testAMissingTensorFailsLoud() throws {
        let shard = try writePleShard(name: "model.safetensors", rows: 4, headDim: 160)
        let container = try QWSafeTensorsContainer(url: shard.url)
        let locations = try writeLocations(shards: [[
            "key": "model.nope", "first_row": 0, "shape": [4, 160], "row_bytes": 320,
            "base_off": shard.baseOffset,
        ]])
        XCTAssertThrowsError(try FNNgramShardRows(container: container, locations: locations))
    }

    func testAShapeThatDisagreesWithTheSidecarIsRefused() throws {
        // The header is the authority on the shape. A sidecar that says
        // otherwise is stale, and reading it would read the wrong rows.
        let shard = try writePleShard(name: "model.safetensors", rows: 4, headDim: 160)
        let container = try QWSafeTensorsContainer(url: shard.url)
        let locations = try writeLocations(shards: [[
            "key": "model.ple.ple_embedding.ngram_embedding.weight",
            "first_row": 0, "shape": [9, 160], "row_bytes": 320, "base_off": shard.baseOffset,
        ]])
        XCTAssertThrowsError(try FNNgramShardRows(container: container, locations: locations))
    }

    func testARowSizeThatDisagreesWithTheLanesIsRefused() throws {
        let shard = try writePleShard(name: "model.safetensors", rows: 4, headDim: 160)
        let container = try QWSafeTensorsContainer(url: shard.url)
        let locations = try writeLocations(shards: [[
            "key": "model.ple.ple_embedding.ngram_embedding.weight",
            "first_row": 0, "shape": [4, 160], "row_bytes": 128, "base_off": shard.baseOffset,
        ]])
        XCTAssertThrowsError(try FNNgramShardRows(container: container, locations: locations))
    }

    func testSeveralShardsAreReadThroughTheirOwnTensors() throws {
        // The real table is 128 shards, each with its own tensor name and
        // its own row range.
        let first = try writePleShard(name: "model-00001.safetensors", rows: 4, headDim: 160,
                                      tensorName: "model.ple.ngram_embedding.shard_0.weight")
        let second = try writePleShard(name: "model-00002.safetensors", rows: 3, headDim: 160,
                                       tensorName: "model.ple.ngram_embedding.shard_1.weight")
        let firstBase = first.baseOffset
        let secondBase = second.baseOffset
        let index = dir.appendingPathComponent("model.safetensors.index.json")
        let map = ["weight_map": [
            "model.ple.ngram_embedding.shard_0.weight": "model-00001.safetensors",
            "model.ple.ngram_embedding.shard_1.weight": "model-00002.safetensors",
        ]]
        try JSONSerialization.data(withJSONObject: map).write(to: index)
        let container = try QWSafeTensorsContainer(url: index)
        let locations = try writeLocations(shards: [
            ["key": "model.ple.ngram_embedding.shard_0.weight", "first_row": 0,
             "shape": [4, 160], "row_bytes": 320, "base_off": firstBase],
            ["key": "model.ple.ngram_embedding.shard_1.weight", "first_row": 4,
             "shape": [3, 160], "row_bytes": 320, "base_off": secondBase],
        ])
        let rows = try FNNgramShardRows(container: container, locations: locations)
        XCTAssertEqual(rows.rowCount, 7)
        XCTAssertEqual(rows.row(0)[0], 0.0)
        XCTAssertEqual(rows.row(3)[0], 3.0)
        // Row 4 is the first row of the second shard, whose payload is its own.
        XCTAssertEqual(rows.row(4)[0], 0.0)
        XCTAssertEqual(rows.row(6)[0], 2.0)
    }

    /// The message the refusal carried, so a test can prove that the check
    /// it means fired and not some other one.
    private func refusal(_ build: () throws -> FNNgramShardRows) -> String? {
        do {
            _ = try build()
            return nil
        } catch {
            return "\(error)"
        }
    }

    func testAShardWithADifferentRowSizeIsRefused() throws {
        // A shard whose rows are narrower than the table's would make a row
        // read run off the end of its own tensor. Each shard agrees with its
        // own header, so only the table-wide check can catch this.
        let shards = try! writeTwoShardIndex(secondHeadDim: 80)
        let container = try QWSafeTensorsContainer(url: shards.url)
        let locations = try writeLocations(shards: [
            ["key": "model.ple.ngram_embedding.shard_0.weight", "first_row": 0,
             "shape": [4, 160], "row_bytes": 320, "base_off": shards.firstBase],
            ["key": "model.ple.ngram_embedding.shard_1.weight", "first_row": 4,
             "shape": [3, 80], "row_bytes": 160, "base_off": shards.secondBase],
        ])
        let message = refusal { try FNNgramShardRows(container: container, locations: locations) }
        XCTAssertNotNil(message)
        XCTAssertTrue(message!.contains("bytes per row"), "refused for the wrong reason: \(message ?? "")")
    }

    func testAGapInTheRowRangeIsRefused() throws {
        // The shards must tile the table. A gap would make the row search
        // pick the wrong shard and read another shard's bytes.
        let shards = try! writeTwoShardIndex(secondHeadDim: 160)
        let container = try QWSafeTensorsContainer(url: shards.url)
        let locations = try writeLocations(shards: [
            ["key": "model.ple.ngram_embedding.shard_0.weight", "first_row": 0,
             "shape": [4, 160], "row_bytes": 320, "base_off": shards.firstBase],
            ["key": "model.ple.ngram_embedding.shard_1.weight", "first_row": 9,
             "shape": [3, 160], "row_bytes": 320, "base_off": shards.secondBase],
        ])
        let message = refusal { try FNNgramShardRows(container: container, locations: locations) }
        XCTAssertTrue(message?.contains("expects row") == true, "refused for the wrong reason: \(message ?? "")")
    }

    func testATotalRowCountThatDisagreesIsRefused() throws {
        let shard = try writePleShard(name: "model.safetensors", rows: 4, headDim: 160)
        let container = try QWSafeTensorsContainer(url: shard.url)
        let locations = try writeLocations(shards: [
            ["key": "model.ple.ple_embedding.ngram_embedding.weight", "first_row": 0,
             "shape": [4, 160], "row_bytes": 320, "base_off": shard.baseOffset],
        ], totalRows: 9)
        let message = refusal { try FNNgramShardRows(container: container, locations: locations) }
        XCTAssertTrue(message?.contains("its shards hold") == true, "refused for the wrong reason: \(message ?? "")")
    }

    func testASidecarDTypeThatDisagreesWithTheHeaderIsRefused() throws {
        let shard = try writePleShard(name: "model.safetensors", rows: 4, headDim: 160)
        let container = try QWSafeTensorsContainer(url: shard.url)
        let locations = try writeLocations(shards: [
            ["key": "model.ple.ple_embedding.ngram_embedding.weight", "dtype": "F16",
             "first_row": 0, "shape": [4, 160], "row_bytes": 320, "base_off": shard.baseOffset],
        ])
        let message = refusal { try FNNgramShardRows(container: container, locations: locations) }
        XCTAssertTrue(message?.contains("locations JSON") == true, "refused for the wrong reason: \(message ?? "")")
    }

    func testAnEmptyShardListIsRefused() throws {
        let shard = try writePleShard(name: "model.safetensors", rows: 4, headDim: 160)
        let container = try QWSafeTensorsContainer(url: shard.url)
        let locations = try writeLocations(shards: [])
        let message = refusal { try FNNgramShardRows(container: container, locations: locations) }
        XCTAssertTrue(message?.contains("no shards") == true, "refused for the wrong reason: \(message ?? "")")
    }

    func testTwoShardsNamingOneTensorAreRefused() throws {
        // Both row ranges would be served from the same payload: in bounds,
        // and wrong.
        let shard = try writePleShard(name: "model.safetensors", rows: 4, headDim: 160)
        let container = try QWSafeTensorsContainer(url: shard.url)
        let locations = try writeLocations(shards: [
            ["key": "model.ple.ple_embedding.ngram_embedding.weight", "first_row": 0,
             "shape": [4, 160], "row_bytes": 320, "base_off": shard.baseOffset],
            ["key": "model.ple.ple_embedding.ngram_embedding.weight", "first_row": 4,
             "shape": [3, 160], "row_bytes": 320, "base_off": shard.baseOffset],
        ])
        let message = refusal { try FNNgramShardRows(container: container, locations: locations) }
        XCTAssertTrue(message?.contains("twice") == true, "refused for the wrong reason: \(message ?? "")")
    }

    func testASidecarWithoutAnOffsetIsRefused() throws {
        // Without `base_off` there is nothing to cross-check the header
        // against, so the sidecar is not trusted.
        let shard = try writePleShard(name: "model.safetensors", rows: 4, headDim: 160)
        let container = try QWSafeTensorsContainer(url: shard.url)
        let locations = try writeLocations(shards: [
            ["key": "model.ple.ple_embedding.ngram_embedding.weight", "first_row": 0,
             "shape": [4, 160], "row_bytes": 320],
        ])
        let message = refusal { try FNNgramShardRows(container: container, locations: locations) }
        XCTAssertTrue(message?.contains("base_off") == true, "refused for the wrong reason: \(message ?? "")")
    }

    func testAMalformedSidecarIsRefusedRatherThanFatal() throws {
        // A sidecar that is not the expected shape must be refused, not trap.
        let url = dir.appendingPathComponent("ple_ngram_locations.json")
        try Data("{\"shards\": \"not a list\"}".utf8).write(to: url)
        let shard = try writePleShard(name: "model.safetensors", rows: 4, headDim: 160)
        let container = try QWSafeTensorsContainer(url: shard.url)
        let message = refusal { try FNNgramShardRows(container: container, locations: url) }
        XCTAssertTrue(message?.contains("shard list") == true, "refused for the wrong reason: \(message ?? "")")
    }

    func testTheLegacyPathChecksTheRowSizeAgainstTheLanes() throws {
        // The legacy path decodes `headDim` lanes out of every row, so a row
        // that claims a different lane count would overflow its scratch.
        let shard = try writePleShard(name: "model.safetensors", rows: 4, headDim: 160)
        let locations = try writeLocations(shards: [
            ["file": shard.url.lastPathComponent, "first_row": 0,
             "shape": [4, 160], "row_bytes": 640, "base_off": shard.baseOffset],
        ])
        let message = refusal { try FNNgramShardRows(locations: locations, snapshot: dir) }
        XCTAssertTrue(message?.contains("BF16 lanes") == true, "refused for the wrong reason: \(message ?? "")")
    }

    func testTheContainerIsScopedToTheTensorsTheSidecarNames() throws {
        // A checkpoint has 131 shards and the sidecar names 33 of them. The
        // reader must not map the other 98.
        _ = try writePleShard(name: "model-00001.safetensors", rows: 4, headDim: 160,
                              tensorName: "model.ple.ngram_embedding.shard_0.weight")
        let other = try writePleShard(name: "model-00002.safetensors", rows: 4, headDim: 160,
                                       tensorName: "model.layers.0.mlp.weight")
        let index = dir.appendingPathComponent("model.safetensors.index.json")
        let map = ["weight_map": [
            "model.ple.ngram_embedding.shard_0.weight": "model-00001.safetensors",
            "model.layers.0.mlp.weight": "model-00002.safetensors",
        ]]
        try JSONSerialization.data(withJSONObject: map).write(to: index)

        let scoped = try QWSafeTensorsContainer(url: index, onlyTensors: ["model.ple.ngram_embedding.shard_0.weight"])
        XCTAssertNotNil(scoped.tensor("model.ple.ngram_embedding.shard_0.weight"))
        XCTAssertNil(scoped.tensor("model.layers.0.mlp.weight"), "the container must not hold an unnamed tensor")
        XCTAssertEqual(scoped.shardURL(of: "model.ple.ngram_embedding.shard_0.weight")?.lastPathComponent,
                       "model-00001.safetensors")
        XCTAssertTrue(FileManager.default.fileExists(atPath: other.url.path))
    }

    func testAScopedContainerRefusesATensorItDoesNotHold() throws {
        _ = try writePleShard(name: "model-00001.safetensors", rows: 4, headDim: 160)
        let index = dir.appendingPathComponent("model.safetensors.index.json")
        let map = ["weight_map": ["model.ple.ngram_embedding.shard_0.weight": "model-00001.safetensors"]]
        try JSONSerialization.data(withJSONObject: map).write(to: index)
        XCTAssertThrowsError(try QWSafeTensorsContainer(url: index, onlyTensors: ["model.missing.weight"]))
    }
}
