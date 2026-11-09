// The QNGRAM on-disk contract, in one place: the header layout, the row
// formats, and the law that ties a row's size to its group structure.

import Foundation

/// The row formats a QNGRAM table can carry.
public enum QWNgramDType: UInt32, Sendable, Equatable, CaseIterable {
    case bf16 = 0
    case q8g32 = 1
    case q4g32 = 2

    /// The bytes one row of `headDim` values takes.
    /// Size-preserving (Law L8): `groups × (payload + scale)`, never more.
    public func rowBytes(headDim: Int) throws -> Int {
        guard headDim > 0 else { throw QWNgramFormatError.badRowSize(headDim) }
        switch self {
        case .bf16:
            return headDim * 2
        case .q8g32:
            guard headDim % QWNgramFormat.group == 0 else {
                throw QWNgramFormatError.badRowSize(headDim)
            }
            return (headDim / QWNgramFormat.group) * (32 + 2)
        case .q4g32:
            guard headDim % QWNgramFormat.group == 0 else {
                throw QWNgramFormatError.badRowSize(headDim)
            }
            return (headDim / QWNgramFormat.group) * 20
        }
    }
}

public enum QWNgramFormatError: Error, Equatable, CustomStringConvertible {
    case badMagic
    case shortHeader
    case unsupportedVersion(UInt16)
    case unsupportedDType(UInt32)
    case badRowSize(Int)
    case badHeader(String)

    public var description: String {
        switch self {
        case .badMagic: return "ngram.bin: bad magic"
        case .shortHeader: return "ngram.bin: header shorter than one sector"
        case .unsupportedVersion(let v): return "ngram.bin: unsupported version \(v)"
        case .unsupportedDType(let d): return "ngram.bin: unsupported dtype \(d)"
        case .badRowSize(let n): return "ngram.bin: row size \(n) does not follow the group law"
        case .badHeader(let detail): return "ngram.bin: bad header: \(detail)"
        }
    }
}

public struct QWNgramHeader: Sendable, Equatable {
    public let version: UInt16
    public let flags: UInt16
    public let dtype: QWNgramDType
    public let entryCount: Int
    public let headDim: Int
    public let rowBytes: Int
    public let l1Hot: Int
    public let dataOffset: Int
    /// The producer's sha256 of the rows, hex, when it wrote one.
    public let note: String
}

/// The QNGRAM on-disk contract, shared byte for byte with the Python writer
/// in `Scripts/quantize/pack_ngram.py`. v1 is BF16 only; v2 adds the two
/// quantised row formats.
public enum QWNgramFormat {
    public static let magic: UInt64 = 0x0000_4D41_5247_4E51  // "QNGRAM\0\0"
    public static let sectorSize = 4096
    public static let headerBytes = 4096
    public static let group = 32
    /// The bytes the producer's sha256 note occupies.
    public static let noteBytes = 64

    public static func parse(_ bytes: [UInt8]) throws -> QWNgramHeader {
        guard bytes.count >= headerBytes else { throw QWNgramFormatError.shortHeader }
        func u16(_ o: Int) -> UInt16 { UInt16(bytes[o]) | (UInt16(bytes[o + 1]) << 8) }
        func u32(_ o: Int) -> UInt32 {
            (0..<4).reduce(UInt32(0)) { $0 | (UInt32(bytes[o + $1]) << (8 * UInt32($1))) }
        }
        func u64(_ o: Int) -> UInt64 {
            (0..<8).reduce(UInt64(0)) { $0 | (UInt64(bytes[o + $1]) << (8 * UInt64($1))) }
        }
        guard u64(0) == magic else { throw QWNgramFormatError.badMagic }
        let version = u16(8)
        guard version == 1 || version == 2 else { throw QWNgramFormatError.unsupportedVersion(version) }
        let dtypeRaw = u32(12)
        guard let dtype = QWNgramDType(rawValue: dtypeRaw) else {
            throw QWNgramFormatError.unsupportedDType(dtypeRaw)
        }
        // The pairing is strict: v1 is BF16 only, v2 carries a quantised
        // format. A v2 file holding BF16 rows is malformed, not "v1 in
        // disguise", and a v1 file holding quantised rows is worse.
        if version == 1, dtype != .bf16 {
            throw QWNgramFormatError.badHeader(
                "version 1 cannot carry dtype \(dtypeRaw)")
        }
        if version == 2, dtype == .bf16 {
            throw QWNgramFormatError.badHeader(
                "version 2 must carry a quantised dtype")
        }
        let headDim = Int(u16(24))
        let rowBytes = Int(u16(26))
        guard rowBytes == (try dtype.rowBytes(headDim: headDim)) else {
            throw QWNgramFormatError.badRowSize(rowBytes)
        }
        // A row must fit in a sector, or the table has zero rows per sector
        // and every row index divides by zero.
        _ = try rowsPerSector(rowBytes: rowBytes)
        let entryCount = Int(u64(16))
        let l1Hot = Int(u32(28))
        let dataOffset = Int(u64(32))
        guard entryCount > 0 else { throw QWNgramFormatError.badHeader("entryCount is zero") }
        guard l1Hot <= entryCount else {
            throw QWNgramFormatError.badHeader("l1Hot \(l1Hot) exceeds entryCount \(entryCount)")
        }
        guard dataOffset >= sectorSize, dataOffset % sectorSize == 0 else {
            throw QWNgramFormatError.badHeader("dataOffset \(dataOffset) is not sector-aligned")
        }
        let noteLen = Int(bytes[44])
        var note = ""
        if noteLen == noteBytes {
            // The note field is a fixed 64 bytes; an all-zero one means the
            // producer wrote no note.
            let raw = bytes[45..<(45 + noteBytes)]
            if raw.contains(where: { $0 != 0 }) {
                note = String(decoding: raw, as: UTF8.self)
            }
        }
        return QWNgramHeader(version: version,
                             flags: u16(10),
                             dtype: dtype,
                             entryCount: entryCount,
                             headDim: headDim,
                             rowBytes: rowBytes,
                             l1Hot: l1Hot,
                             dataOffset: dataOffset,
                             note: note)
    }

    /// Builds a canonical 4096-byte header.
    /// The Swift writer and the Python writer must produce identical bytes.
    public static func makeHeader(version: UInt16,
                                  dtype: QWNgramDType,
                                  entryCount: UInt64,
                                  headDim: UInt16,
                                  rowBytes: UInt16,
                                  l1Hot: UInt32,
                                  note: String = "") throws -> [UInt8] {
        // A writer that could emit a header the reader refuses would be a
        // footgun, so it applies the reader's own rules first.
        guard version == 1 || version == 2 else {
            throw QWNgramFormatError.unsupportedVersion(version)
        }
        if version == 1, dtype != .bf16 {
            throw QWNgramFormatError.badHeader(
                "version 1 cannot carry dtype \(dtype.rawValue)")
        }
        if version == 2, dtype == .bf16 {
            throw QWNgramFormatError.badHeader(
                "version 2 must carry a quantised dtype")
        }
        guard rowBytes == (try dtype.rowBytes(headDim: Int(headDim))) else {
            throw QWNgramFormatError.badRowSize(Int(rowBytes))
        }
        guard entryCount > 0 else {
            throw QWNgramFormatError.badHeader("entryCount is zero")
        }
        guard l1Hot <= entryCount else {
            throw QWNgramFormatError.badHeader(
                "l1Hot \(l1Hot) exceeds entryCount \(entryCount)")
        }
        let noteBytes = Array(note.utf8)
        guard noteBytes.isEmpty || noteBytes.count == self.noteBytes else {
            throw QWNgramFormatError.badHeader(
                "note must be \(self.noteBytes) bytes or empty")
        }
        var h = [UInt8](repeating: 0, count: headerBytes)
        func put16(_ o: Int, _ v: UInt16) { h[o] = UInt8(v & 0xFF); h[o + 1] = UInt8(v >> 8) }
        func put32(_ o: Int, _ v: UInt32) {
            for i in 0..<4 { h[o + i] = UInt8((v >> (8 * UInt32(i))) & 0xFF) }
        }
        func put64(_ o: Int, _ v: UInt64) {
            for i in 0..<8 { h[o + i] = UInt8((v >> (8 * UInt64(i))) & 0xFF) }
        }
        put64(0, magic)
        put16(8, version); put16(10, 0); put32(12, dtype.rawValue)
        put64(16, entryCount); put16(24, headDim); put16(26, rowBytes)
        put32(28, l1Hot); put64(32, UInt64(headerBytes))
        h[44] = UInt8(self.noteBytes)
        for (i, b) in noteBytes.enumerated() { h[45 + i] = b }
        return h
    }

    public static func rowBytes(dtype: QWNgramDType, headDim: Int) throws -> Int {
        try dtype.rowBytes(headDim: headDim)
    }

    /// The rows that fit in one sector.
    /// A row must fit: a table is sector-packed, so a row larger than a
    /// sector leaves zero rows per sector, and every row index divides by zero.
    public static func rowsPerSector(rowBytes: Int) throws -> Int {
        guard rowBytes > 0, rowBytes <= sectorSize else {
            throw QWNgramFormatError.badRowSize(rowBytes)
        }
        return sectorSize / rowBytes
    }
}

extension QWNgramFormat {
    /// Decodes one packed row of `headDim` values into `dst`.
    /// The bytes are the producer's, in the producer's own domain: a row is
    /// added to the residual stream, so no transform is applied to it.
    public static func decodeRow(dtype: QWNgramDType,
                                  from bytes: UnsafeRawPointer,
                                  headDim: Int,
                                  into dst: UnsafeMutablePointer<Float>) {
        precondition(headDim > 0, "ngram.bin: a row must hold at least one value")
        switch dtype {
        case .bf16:
            for i in 0..<headDim {
                dst[i] = Float(bitPattern: UInt32(QWQuant.u16le(bytes, i * 2)) << 16)
            }
        case .q8g32:
            precondition(headDim % group == 0, "ngram.bin: a Q8 row holds whole groups")
            for g in 0..<(headDim / group) {
                let base = g * (group + 2)
                let scale = QWQuant.f16tof32Bits(QWQuant.u16le(bytes, base + group))
                for k in 0..<group {
                    let code = Int8(bitPattern: bytes.loadUnaligned(fromByteOffset: base + k,
                                                                    as: UInt8.self))
                    dst[g * group + k] = Float(code) * scale
                }
            }
        case .q4g32:
            // The block holds `QWQuant.group` values and the container's
            // group is the same number. A divergence would mis-read the
            // row, so it is checked here rather than assumed.
            precondition(group == QWQuant.group, "ngram.bin: the Q4 block group must match the row group")
            precondition(headDim % group == 0, "ngram.bin: a Q4 row holds whole groups")
            for g in 0..<(headDim / group) {
                QWQuant.dequantHQ4Group(bytes.advanced(by: g * QWQuant.blockBytes),
                                        into: dst.advanced(by: g * group))
            }
        }
    }
}
