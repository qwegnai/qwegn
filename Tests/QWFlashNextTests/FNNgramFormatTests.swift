import XCTest
import Foundation
@testable import QWQuant

final class FNNgramFormatTests: XCTestCase {
    /// Builds a canonical 4096-byte QNGRAM header, matching `pack_ngram.py` exactly.
    private func header(version: UInt16,
                        dtype: UInt32,
                        entryCount: UInt64,
                        headDim: UInt16,
                        rowBytes: UInt16,
                        l1Hot: UInt32 = 0) -> [UInt8] {
        var h = [UInt8](repeating: 0, count: 4096)
        let magic: [UInt8] = [0x51, 0x4E, 0x47, 0x52, 0x41, 0x4D, 0x00, 0x00]
        for (i, b) in magic.enumerated() { h[i] = b }
        func put16(_ off: Int, _ v: UInt16) { h[off] = UInt8(v & 0xFF); h[off + 1] = UInt8(v >> 8) }
        func put32(_ off: Int, _ v: UInt32) {
            for i in 0..<4 { h[off + i] = UInt8((v >> (8 * UInt32(i))) & 0xFF) }
        }
        func put64(_ off: Int, _ v: UInt64) {
            for i in 0..<8 { h[off + i] = UInt8((v >> (8 * UInt64(i))) & 0xFF) }
        }
        put16(8, version); put16(10, 0); put32(12, dtype)
        put64(16, entryCount); put16(24, headDim); put16(26, rowBytes)
        put32(28, l1Hot); put64(32, 4096)
        h[44] = 64
        return h
    }

    func testParsesTheCanonicalV1Header() throws {
        let h = try QWNgramFormat.parse(header(version: 1, dtype: 0, entryCount: 25, headDim: 160, rowBytes: 320))
        XCTAssertEqual(h.version, 1)
        XCTAssertEqual(h.dtype, .bf16)
        XCTAssertEqual(h.entryCount, 25)
        XCTAssertEqual(h.headDim, 160)
        XCTAssertEqual(h.rowBytes, 320)
        XCTAssertEqual(h.dataOffset, 4096)
    }

    func testParsesTheV2Q8Header() throws {
        let h = try QWNgramFormat.parse(header(version: 2, dtype: 1, entryCount: 25, headDim: 160, rowBytes: 170))
        XCTAssertEqual(h.version, 2)
        XCTAssertEqual(h.dtype, .q8g32)
        XCTAssertEqual(h.rowBytes, 170)
        XCTAssertEqual(try QWNgramFormat.rowsPerSector(rowBytes: 170), 24)
    }

    func testRejectsAnUnknownVersion() {
        XCTAssertThrowsError(try QWNgramFormat.parse(header(version: 7, dtype: 0, entryCount: 1, headDim: 160, rowBytes: 320))) {
            XCTAssertEqual($0 as? QWNgramFormatError, .unsupportedVersion(7))
        }
    }

    func testRejectsAnUnknownDType() {
        XCTAssertThrowsError(try QWNgramFormat.parse(header(version: 2, dtype: 9, entryCount: 1, headDim: 160, rowBytes: 170))) {
            XCTAssertEqual($0 as? QWNgramFormatError, .unsupportedDType(9))
        }
    }

    func testRejectsBadMagic() {
        var h = header(version: 1, dtype: 0, entryCount: 1, headDim: 160, rowBytes: 320)
        h[0] = 0x00
        XCTAssertThrowsError(try QWNgramFormat.parse(h)) {
            XCTAssertEqual($0 as? QWNgramFormatError, .badMagic)
        }
    }

    func testRowSizesFollowTheGroupLaw() throws {
        XCTAssertEqual(try QWNgramFormat.rowBytes(dtype: .bf16, headDim: 160), 320)
        XCTAssertEqual(try QWNgramFormat.rowBytes(dtype: .q8g32, headDim: 160), 170)
        XCTAssertEqual(try QWNgramFormat.rowBytes(dtype: .q4g32, headDim: 160), 100)
    }

    func testAQuantisedRowIsNeverLargerThanItsBf16Row() throws {
        // Law L8: the quantised formats are size-preserving with respect to
        // their own group law. They shrink; they never inflate.
        let bf16 = try QWNgramFormat.rowBytes(dtype: .bf16, headDim: 160)
        XCTAssertLessThan(try QWNgramFormat.rowBytes(dtype: .q8g32, headDim: 160), bf16)
        XCTAssertLessThan(try QWNgramFormat.rowBytes(dtype: .q4g32, headDim: 160), bf16)
    }

    func testRejectsARowSizeThatDisagreesWithTheDType() {
        // A v2 Q8 header that claims the BF16 row size: the reader must
        // refuse it, not read a 170-byte row as 320.
        XCTAssertThrowsError(try QWNgramFormat.parse(header(version: 2, dtype: 1, entryCount: 25, headDim: 160, rowBytes: 320))) {
            XCTAssertEqual($0 as? QWNgramFormatError, .badRowSize(320))
        }
    }

    func testRejectsAHeadDimThatIsNotAWholeNumberOfGroups() {
        XCTAssertThrowsError(try QWNgramFormat.rowBytes(dtype: .q8g32, headDim: 100))
    }

    func testRejectsAZeroHeadDim() {
        // A zero head dim would give a zero row size, and a zero row size
        // would make `rowsPerSector` divide by zero on the read path. The
        // header must be refused, not carried.
        XCTAssertThrowsError(try QWNgramFormat.rowBytes(dtype: .bf16, headDim: 0))
        XCTAssertThrowsError(try QWNgramFormat.parse(header(version: 1, dtype: 0, entryCount: 25, headDim: 0, rowBytes: 0))) {
            XCTAssertEqual($0 as? QWNgramFormatError, .badRowSize(0))
        }
    }

    func testTheRowsPerSectorIsNeverZeroForAValidRowSize() throws {
        for dtype in QWNgramDType.allCases {
            for headDim in [32, 64, 160, 512] {
                let bytes = try QWNgramFormat.rowBytes(dtype: dtype, headDim: headDim)
                XCTAssertGreaterThan(try QWNgramFormat.rowsPerSector(rowBytes: bytes), 0)
            }
        }
    }

    func testRejectsARowLargerThanASector() {
        // A row that does not fit a sector would leave zero rows per
        // sector, and every row index would divide by zero.
        XCTAssertThrowsError(try QWNgramFormat.rowsPerSector(rowBytes: 4097))
        XCTAssertThrowsError(try QWNgramFormat.rowsPerSector(rowBytes: 0))
        // A header claiming a head dim whose BF16 row is twice a sector.
        XCTAssertThrowsError(try QWNgramFormat.parse(header(version: 1, dtype: 0, entryCount: 25, headDim: 4096, rowBytes: 8192))) {
            XCTAssertEqual($0 as? QWNgramFormatError, .badRowSize(8192))
        }
    }

    func testTheWriterRefusesToBuildAHeaderTheReaderWouldReject() {
        // A writer that could emit an invalid header is a footgun: it would
        // produce a file the reader refuses, or worse, one it misreads.
        XCTAssertThrowsError(try QWNgramFormat.makeHeader(version: 1, dtype: .q8g32,
                                                           entryCount: 25, headDim: 160,
                                                           rowBytes: 170, l1Hot: 0))
        XCTAssertThrowsError(try QWNgramFormat.makeHeader(version: 2, dtype: .bf16,
                                                           entryCount: 25, headDim: 160,
                                                           rowBytes: 320, l1Hot: 0))
        XCTAssertThrowsError(try QWNgramFormat.makeHeader(version: 7, dtype: .bf16,
                                                           entryCount: 25, headDim: 160,
                                                           rowBytes: 320, l1Hot: 0))
        XCTAssertThrowsError(try QWNgramFormat.makeHeader(version: 1, dtype: .bf16,
                                                           entryCount: 25, headDim: 160,
                                                           rowBytes: 170, l1Hot: 0))
        XCTAssertThrowsError(try QWNgramFormat.makeHeader(version: 1, dtype: .bf16,
                                                           entryCount: 0, headDim: 160,
                                                           rowBytes: 320, l1Hot: 0))
        XCTAssertThrowsError(try QWNgramFormat.makeHeader(version: 1, dtype: .bf16,
                                                           entryCount: 25, headDim: 160,
                                                           rowBytes: 320, l1Hot: 26))
        XCTAssertThrowsError(try QWNgramFormat.makeHeader(version: 1, dtype: .bf16,
                                                           entryCount: 25, headDim: 160,
                                                           rowBytes: 320, l1Hot: 0,
                                                           note: "too short"))
    }

    func testAnEmptyNoteRoundTripsAsNoNote() throws {
        let bytes = try QWNgramFormat.makeHeader(version: 1, dtype: .bf16, entryCount: 25,
                                                 headDim: 160, rowBytes: 320, l1Hot: 0)
        let h = try QWNgramFormat.parse(bytes)
        XCTAssertEqual(h.note, "")
        // The note field is still 64 bytes wide; an all-zero one means the
        // producer wrote no note.
        XCTAssertEqual(bytes[44], 64)
    }

    func testRejectsAShortBufferWithItsOwnError() {
        // A short buffer is not a bad magic; it deserves its own error.
        XCTAssertThrowsError(try QWNgramFormat.parse([UInt8](repeating: 0, count: 100))) {
            XCTAssertEqual($0 as? QWNgramFormatError, .shortHeader)
        }
    }

    func testTheWriterAndTheReaderAgree() throws {
        let bytes = try QWNgramFormat.makeHeader(version: 2, dtype: .q4g32, entryCount: 25,
                                             headDim: 160, rowBytes: 100, l1Hot: 7)
        let h = try QWNgramFormat.parse(bytes)
        XCTAssertEqual(h.version, 2)
        XCTAssertEqual(h.dtype, .q4g32)
        XCTAssertEqual(h.entryCount, 25)
        XCTAssertEqual(h.headDim, 160)
        XCTAssertEqual(h.rowBytes, 100)
        XCTAssertEqual(h.l1Hot, 7)
        XCTAssertEqual(h.dataOffset, 4096)
    }

    func testRejectsAVersionThatCannotCarryItsDType() {
        // v1 is BF16 only; v2 carries a quantised format. A file that breaks
        // the pairing is malformed, and the reader must say so.
        XCTAssertThrowsError(try QWNgramFormat.parse(header(version: 1, dtype: 1, entryCount: 25, headDim: 160, rowBytes: 170))) {
            guard case .badHeader = $0 as? QWNgramFormatError else {
                return XCTFail("expected badHeader, got \($0)")
            }
        }
        XCTAssertThrowsError(try QWNgramFormat.parse(header(version: 2, dtype: 0, entryCount: 25, headDim: 160, rowBytes: 320))) {
            guard case .badHeader = $0 as? QWNgramFormatError else {
                return XCTFail("expected badHeader, got \($0)")
            }
        }
    }

    func testTheHeaderBytesAreTheProducersHeaderBytes() throws {
        // The cross-language contract. `Scripts/quantize/test_pack.py` asserts
        // these same hex strings against the Python writer, so the two
        // implementations cannot drift apart without a test going red.
        func hex(_ bytes: [UInt8]) -> String {
            bytes.map { String(format: "%02x", $0) }.joined()
        }
        let v1 = try QWNgramFormat.makeHeader(version: 1, dtype: .bf16, entryCount: 25,
                                          headDim: 160, rowBytes: 320, l1Hot: 0)
        XCTAssertEqual(hex(Array(v1[0..<45])),
                       "514e4752414d000001000000000000001900000000000000a00040010000000000100000000000000000000040")
        let v2q8 = try QWNgramFormat.makeHeader(version: 2, dtype: .q8g32, entryCount: 25,
                                            headDim: 160, rowBytes: 170, l1Hot: 0)
        XCTAssertEqual(hex(Array(v2q8[0..<45])),
                       "514e4752414d000002000000010000001900000000000000a000aa000000000000100000000000000000000040")
        let v2q4 = try QWNgramFormat.makeHeader(version: 2, dtype: .q4g32, entryCount: 25,
                                            headDim: 160, rowBytes: 100, l1Hot: 0)
        XCTAssertEqual(hex(Array(v2q4[0..<45])),
                       "514e4752414d000002000000020000001900000000000000a00064000000000000100000000000000000000040")
    }

    func testTheHeaderLayoutIsTheProducersLayout() throws {
        // The byte offsets are the contract with `pack_ngram.py`: magic@0,
        // version@8, flags@10, dtype@12, entryCount@16, headDim@24,
        // rowBytes@26, l1Hot@28, dataOffset@32.
        let bytes = try QWNgramFormat.makeHeader(version: 2, dtype: .q8g32, entryCount: 25,
                                             headDim: 160, rowBytes: 170, l1Hot: 3)
        XCTAssertEqual(Array(bytes[0..<8]), [0x51, 0x4E, 0x47, 0x52, 0x41, 0x4D, 0x00, 0x00])
        XCTAssertEqual(UInt16(bytes[8]) | (UInt16(bytes[9]) << 8), 2)
        XCTAssertEqual(UInt16(bytes[10]) | (UInt16(bytes[11]) << 8), 0)
        XCTAssertEqual(UInt32(bytes[12]) | (UInt32(bytes[13]) << 8)
                        | (UInt32(bytes[14]) << 16) | (UInt32(bytes[15]) << 24), 1)
        XCTAssertEqual(UInt64(bytes[16]) | (UInt64(bytes[17]) << 8), 25)
        XCTAssertEqual(UInt16(bytes[24]) | (UInt16(bytes[25]) << 8), 160)
        XCTAssertEqual(UInt16(bytes[26]) | (UInt16(bytes[27]) << 8), 170)
        XCTAssertEqual(UInt32(bytes[28]), 3)
        XCTAssertEqual((0..<8).reduce(UInt64(0)) { $0 | (UInt64(bytes[32 + $1]) << (8 * UInt64($1))) }, 4096)
    }
}
