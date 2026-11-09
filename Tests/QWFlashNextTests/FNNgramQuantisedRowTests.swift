import XCTest
import Foundation
import CryptoKit
@testable import QWQuant
@testable import QWFlashNext

/// The n-gram row reader, against the tables the **Python** writer wrote.
/// The golden bytes and the expected rows are both the writer's own, so the
/// reader and the writer cannot drift apart without a test going red.
final class FNNgramQuantisedRowTests: XCTestCase {
    /// One row format, as the golden fixture records it.
    struct Format {
        let name: String
        let version: UInt16
        let dtype: UInt32
        let rowBytes: Int
        let rowsPerSector: Int
        let file: String
        let expected: [Float]
    }

    static let fixtures = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().appendingPathComponent("fixtures")

    /// The golden fixture: the formats, their tables and the writer's rows.
    func golden() throws -> (headDim: Int, rows: Int, formats: [Format]) {
        let raw = try Data(contentsOf: Self.fixtures.appendingPathComponent("ngram_rows.json"))
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: raw) as? [String: Any])
        let headDim = try XCTUnwrap(root["headDim"] as? Int)
        let rows = try XCTUnwrap(root["rows"] as? Int)
        let list = try XCTUnwrap(root["formats"] as? [[String: Any]])
        var formats: [Format] = []
        for entry in list {
            let name = try XCTUnwrap(entry["name"] as? String)
            let version = try XCTUnwrap(entry["version"] as? Int)
            let dtype = try XCTUnwrap(entry["dtype"] as? Int)
            let rowBytes = try XCTUnwrap(entry["rowBytes"] as? Int)
            let rowsPerSector = try XCTUnwrap(entry["rowsPerSector"] as? Int)
            let file = try XCTUnwrap(entry["file"] as? String)
            let payload = try XCTUnwrap(Data(base64Encoded: try XCTUnwrap(entry["expected"] as? String)))
            let sha = try XCTUnwrap(entry["sha256"] as? String)
            XCTAssertEqual(SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined(),
                           sha, "\(name): the golden rows are not the ones the fixture names")
            formats.append(Format(name: name, version: UInt16(version), dtype: UInt32(dtype),
                                  rowBytes: rowBytes, rowsPerSector: rowsPerSector, file: file,
                                  expected: try Self.floats(payload)))
        }
        return (headDim, rows, formats)
    }

    /// The payload's float32 values, little-endian, as the writer wrote them.
    static func floats(_ payload: Data) throws -> [Float] {
        var out = [Float](repeating: 0, count: payload.count / 4)
        let bytes = [UInt8](payload)
        for i in 0..<out.count {
            let o = i * 4
            let bits = UInt32(bytes[o]) | (UInt32(bytes[o + 1]) << 8)
                | (UInt32(bytes[o + 2]) << 16) | (UInt32(bytes[o + 3]) << 24)
            out[i] = Float(bitPattern: bits)
        }
        return out
    }

    /// The table the writer wrote, opened by the reader.
    func table(_ format: Format) throws -> FNNgramBinRows {
        try FNNgramBinRows(url: Self.fixtures.appendingPathComponent(format.file))
    }

    /// Compares a row with the writer's, value by value.
    func assertRow(_ got: [Float], _ want: [Float], _ label: String) {
        XCTAssertEqual(got.count, want.count, "\(label): the row holds a different count")
        guard got.count == want.count else { return }
        for i in 0..<got.count where got[i] != want[i] {
            XCTFail("\(label): value \(i) is \(got[i]) (0x\(String(got[i].bitPattern, radix: 16))), "
                    + "the writer's is \(want[i]) (0x\(String(want[i].bitPattern, radix: 16)))")
            return
        }
    }

    func testTheFixtureNamesTheThreeFormats() throws {
        let g = try golden()
        XCTAssertEqual(g.headDim, 160)
        XCTAssertEqual(g.formats.map { $0.name }, ["bf16", "q8", "q4"])
        XCTAssertEqual(g.formats.map { $0.rowBytes }, [320, 170, 100])
        XCTAssertEqual(g.formats.map { $0.rowsPerSector }, [12, 24, 40])
        XCTAssertEqual(g.formats.map { $0.version }, [1, 2, 2])
        for f in g.formats {
            XCTAssertEqual(f.expected.count, g.rows * g.headDim, "\(f.name): the golden rows are short")
        }
    }

    func testTheBF16RowsAreTheWritersRows() throws {
        let g = try golden()
        let format = g.formats[0]
        let rows = try table(format)
        XCTAssertEqual(rows.dtype, .bf16)
        XCTAssertEqual(rows.headDim, g.headDim)
        XCTAssertEqual(rows.rowBytes, format.rowBytes)
        XCTAssertEqual(rows.rowsPerSector, format.rowsPerSector)
        XCTAssertEqual(rows.entryCount, g.rows)
        for id in 0..<g.rows {
            assertRow(rows.row(id), Array(format.expected[(id * g.headDim)..<((id + 1) * g.headDim)]),
                      "\(format.name) row \(id)")
        }
    }

    func testTheQ8RowsAreTheWritersRows() throws {
        let g = try golden()
        let format = g.formats[1]
        let rows = try table(format)
        XCTAssertEqual(rows.dtype, .q8g32)
        XCTAssertEqual(rows.headDim, g.headDim)
        XCTAssertEqual(rows.rowBytes, format.rowBytes)
        XCTAssertEqual(rows.rowsPerSector, format.rowsPerSector)
        XCTAssertEqual(rows.entryCount, g.rows)
        for id in 0..<g.rows {
            assertRow(rows.row(id), Array(format.expected[(id * g.headDim)..<((id + 1) * g.headDim)]),
                      "\(format.name) row \(id)")
        }
    }

    func testTheQ4RowsAreTheWritersRows() throws {
        let g = try golden()
        let format = g.formats[2]
        let rows = try table(format)
        XCTAssertEqual(rows.dtype, .q4g32)
        XCTAssertEqual(rows.headDim, g.headDim)
        XCTAssertEqual(rows.rowBytes, format.rowBytes)
        XCTAssertEqual(rows.rowsPerSector, format.rowsPerSector)
        XCTAssertEqual(rows.entryCount, g.rows)
        for id in 0..<g.rows {
            assertRow(rows.row(id), Array(format.expected[(id * g.headDim)..<((id + 1) * g.headDim)]),
                      "\(format.name) row \(id)")
        }
    }

    func testMappedRowsMatchPreadAndWriterAcrossFormats() throws {
        let golden = try golden()
        for format in golden.formats {
            let pread = try table(format)
            let mapped = try FNNgramBinRows(url: Self.fixtures.appendingPathComponent(format.file), mapped: true)
            for id in 0..<golden.rows {
                let expected = Array(format.expected[(id * golden.headDim)..<((id + 1) * golden.headDim)])
                assertRow(mapped.row(id), expected, "\(format.name) mapped row \(id)")
                assertRow(mapped.row(id), pread.row(id), "\(format.name) pread parity \(id)")
            }
        }
    }

    func testMappedReaderReportsMissingFile() {
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        XCTAssertThrowsError(try FNNgramBinRows(url: missing, mapped: true))
    }

    func testTheQ4RowIsNotRotated() throws {
        // A Q4 n-gram row is stored in the original domain: a looked-up row
        // is added to the residual stream, so the inverse transform must not
        // be applied. The two answers differ, so the golden tells them apart.
        let g = try golden()
        let rows = try table(g.formats[2])
        let got = rows.row(2)
        var rotated = got
        rotated.withUnsafeMutableBufferPointer { buf in
            let base = buf.baseAddress!
            for i in stride(from: 0, to: g.headDim, by: 32) {
                QWQuant.fwht32OrthoBlock(base.advanced(by: i))
            }
        }
        XCTAssertNotEqual(got, rotated, "the rotated and the plain form are the same answer")
    }

    func testTheRowsAcrossASectorBoundaryAreRight() throws {
        // Every format packs a different number of rows to a sector, so the
        // row that starts the second sector is a different one each time.
        let g = try golden()
        for format in g.formats {
            let rows = try table(format)
            let boundary = format.rowsPerSector
            for id in [boundary - 1, boundary, g.rows - 1] {
                assertRow(rows.row(id), Array(format.expected[(id * g.headDim)..<((id + 1) * g.headDim)]),
                          "\(format.name) row \(id)")
            }
        }
    }

    func testAGroupWithAZeroScaleDecodesToZero() throws {
        // The writer stores zero codes and a zero scale for a group with
        // no values, so the row reads back as zero and not as noise.
        let g = try golden()
        let q8 = try table(g.formats[1])
        let row0 = q8.row(0)
        XCTAssertEqual(row0, [Float](repeating: 0, count: g.headDim))
        let row1 = q8.row(1)
        XCTAssertEqual(Array(row1[0..<32]), [Float](repeating: 0, count: 32))
        XCTAssertFalse(row1[32..<64].allSatisfy { $0 == 0 },
                       "the group with a subnormal scale must not read as zero")
    }

    func testTheReaderDecodesIntoADestinationWithNoSlack() throws {
        // The destination holds the row's values and nothing else: there is
        // no room for a scratch area. The old reader staged the bytes in the
        // caller's buffer and read past the bytes it fetched on a quantised row.
        let g = try golden()
        for index in 0..<g.formats.count {
            let format = g.formats[index]
            let rows = try table(format)
            var buf = [Float](repeating: 0, count: rows.headDim)
            buf.withUnsafeMutableBufferPointer { p in
                rows.copyRow(0, to: p.baseAddress!)
            }
            assertRow(buf, Array(format.expected[0..<rows.headDim]), "\(format.name) row 0")
        }
    }

    func testTheReaderDoesNotWritePastTheRow() throws {
        // A canary one float beyond the row: a reader that writes past the
        // row's own values must trip it. The row's values must also be the
        // writer's, so a reader that scribbles inside the row is caught too.
        let g = try golden()
        for index in 0..<g.formats.count {
            let format = g.formats[index]
            let rows = try table(format)
            let n = rows.headDim
            var buf = [Float](repeating: -1, count: n + 1)
            buf.withUnsafeMutableBufferPointer { p in
                rows.copyRow(0, to: p.baseAddress!)
            }
            XCTAssertEqual(buf[n], -1, "\(format.name): the row wrote past its own end")
            assertRow(Array(buf[0..<n]), Array(format.expected[0..<n]),
                      "\(format.name) row 0")
        }
    }

    func testAnUnknownDTypeIsStillRefused() throws {
        // The reader serves three formats and refuses everything else by
        // name. A file that claims a fourth dtype is not read at all.
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ngram-refusal-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("ngram.bin")
        var bytes = Self.header(dtype: 9)
        bytes += [UInt8](repeating: 0, count: 4096)
        try Data(bytes).write(to: url)
        XCTAssertThrowsError(try FNNgramBinRows(url: url)) {
            XCTAssertEqual($0 as? QWNgramFormatError, .unsupportedDType(9))
        }
    }

    /// A canonical 4096-byte QNGRAM header, matching `pack_ngram.py`.
    static func header(dtype: UInt32) -> [UInt8] {
        var h = [UInt8](repeating: 0, count: QWNgramFormat.headerBytes)
        let magic: [UInt8] = [0x51, 0x4E, 0x47, 0x52, 0x41, 0x4D, 0x00, 0x00]
        for (i, b) in magic.enumerated() { h[i] = b }
        func put16(_ o: Int, _ v: UInt16) { h[o] = UInt8(v & 0xFF); h[o + 1] = UInt8(v >> 8) }
        func put32(_ o: Int, _ v: UInt32) {
            for i in 0..<4 { h[o + i] = UInt8((v >> (8 * UInt32(i))) & 0xFF) }
        }
        func put64(_ o: Int, _ v: UInt64) {
            for i in 0..<8 { h[o + i] = UInt8((v >> (8 * UInt64(i))) & 0xFF) }
        }
        put16(8, 2); put16(10, 0); put32(12, dtype)
        put64(16, 25); put16(24, 160); put16(26, 170)
        put32(28, 0); put64(32, 4096)
        h[44] = 64
        return h
    }
}
