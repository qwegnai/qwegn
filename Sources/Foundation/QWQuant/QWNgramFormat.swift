// The QNGRAM on-disk contract, in one place: the header layout, the row
// formats, and the law that ties a row's size to its group structure.

import Foundation

/// The row formats a QNGRAM table can carry.
public enum QWNgramDType: UInt32, Sendable, Equatable, CaseIterable {
    case bf16 = 0
    case q8g32 = 1
    case q4g32 = 2
    case vq2g32 = 3

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
        case .vq2g32:
            guard headDim % QWNgramFormat.group == 0 else {
                throw QWNgramFormatError.badRowSize(headDim)
            }
            return (headDim / QWNgramFormat.group) * 18
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
    /// Widened `[256 * 2]` Float32 e-means codebook for v3 `vq2g32` (`[]` for v1/v2).
    public let codebook: [Float]

    public init(version: UInt16, flags: UInt16, dtype: QWNgramDType, entryCount: Int,
                headDim: Int, rowBytes: Int, l1Hot: Int, dataOffset: Int,
                note: String, codebook: [Float] = []) {
        self.version = version
        self.flags = flags
        self.dtype = dtype
        self.entryCount = entryCount
        self.headDim = headDim
        self.rowBytes = rowBytes
        self.l1Hot = l1Hot
        self.dataOffset = dataOffset
        self.note = note
        self.codebook = codebook
    }
}

/// The QNGRAM on-disk contract, shared byte for byte with `Scripts/quantize/pack_ngram.py`.
/// v1 is BF16, v2 carries scalar group-quantised rows, and v3 carries VQ2_G32 + 1 KiB e-means LUT.
public enum QWNgramFormat {
    public static let magic: UInt64 = 0x0000_4D41_5247_4E51  // "QNGRAM\0\0"
    public static let sectorSize = 4096
    public static let headerBytes = 4096
    public static let group = 32
    /// The bytes the producer's sha256 note occupies.
    public static let noteBytes = 64
    public static let vq2CodebookOffset = 1024
    public static let vq2CodebookBytes = 1024
    public static let vq2CodebookScalars = 512

    private static func validateVQ2Codebook(_ rawU16: [UInt16]) throws -> [Float] {
        guard rawU16.count == vq2CodebookScalars else {
            throw QWNgramFormatError.badHeader("v3 codebook must hold 512 f16 values")
        }
        guard rawU16[0] == 0 && rawU16[1] == 0 else {
            throw QWNgramFormatError.badHeader("v3 codebook[0] must be canonical +0.0")
        }
        var out = [Float](repeating: 0, count: vq2CodebookScalars)
        var anyLive = false
        for (i, bits) in rawU16.enumerated() {
            let v = QWQuant.f16tof32Bits(bits)
            guard v.isFinite else {
                throw QWNgramFormatError.badHeader("non-finite value in v3 codebook")
            }
            out[i] = v
            if i >= 2 && v != 0 { anyLive = true }
        }
        guard anyLive else {
            throw QWNgramFormatError.badHeader("v3 codebook is all zero")
        }
        return out
    }

    private static func checkVersionPairing(version: UInt16, dtype: QWNgramDType) throws {
        guard version == 1 || version == 2 || version == 3 else {
            throw QWNgramFormatError.unsupportedVersion(version)
        }
        if version == 1, dtype != .bf16 {
            throw QWNgramFormatError.badHeader("version 1 cannot carry dtype \(dtype.rawValue)")
        }
        if version == 2, (dtype == .bf16 || dtype == .vq2g32) {
            throw QWNgramFormatError.badHeader("version 2 cannot carry dtype \(dtype.rawValue)")
        }
        if version == 3, dtype != .vq2g32 {
            throw QWNgramFormatError.badHeader("version 3 must carry dtype vq2g32")
        }
    }

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
        guard version == 1 || version == 2 || version == 3 else {
            throw QWNgramFormatError.unsupportedVersion(version)
        }
        let dtypeRaw = u32(12)
        guard let dtype = QWNgramDType(rawValue: dtypeRaw) else {
            throw QWNgramFormatError.unsupportedDType(dtypeRaw)
        }
        try checkVersionPairing(version: version, dtype: dtype)
        let codebook: [Float]
        if dtype == .vq2g32 {
            let raw = (0..<vq2CodebookScalars).map { u16(vq2CodebookOffset + $0 * 2) }
            codebook = try validateVQ2Codebook(raw)
        } else {
            codebook = []
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
                             note: note,
                             codebook: codebook)
    }

    /// Builds a canonical 4096-byte header matching the Python writer byte for byte.
    public static func makeHeader(version: UInt16,
                                  dtype: QWNgramDType,
                                  entryCount: UInt64,
                                  headDim: UInt16,
                                  rowBytes: UInt16,
                                  l1Hot: UInt32,
                                  note: String = "",
                                  codebook: [UInt16] = []) throws -> [UInt8] {
        try checkVersionPairing(version: version, dtype: dtype)
        if dtype == .vq2g32 {
            _ = try validateVQ2Codebook(codebook)
        } else if !codebook.isEmpty {
            throw QWNgramFormatError.badHeader("codebook is only valid for v3 vq2g32")
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
        if dtype == .vq2g32 {
            for (i, u) in codebook.enumerated() { put16(vq2CodebookOffset + i * 2, u) }
        }
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
                                  codebook: UnsafePointer<Float>? = nil,
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
            precondition(group == QWQuant.group, "ngram.bin: the Q4 block group must match the row group")
            precondition(headDim % group == 0, "ngram.bin: a Q4 row holds whole groups")
            for g in 0..<(headDim / group) {
                QWQuant.dequantHQ4Group(bytes.advanced(by: g * QWQuant.blockBytes),
                                        into: dst.advanced(by: g * group))
            }
        case .vq2g32:
            precondition(headDim % group == 0, "ngram.bin: a VQ2 row holds whole groups")
            guard let cb = codebook else {
                preconditionFailure("ngram.bin: VQ2 decode requires its e-means codebook")
            }
            let pairs = group / 2
            let stride = pairs + 2
            for g in 0..<(headDim / group) {
                let base = g * stride
                let scale = QWQuant.f16tof32Bits(QWQuant.u16le(bytes, base + pairs))
                for p in 0..<pairs {
                    let code = Int(bytes.loadUnaligned(fromByteOffset: base + p, as: UInt8.self))
                    let outIdx = g * group + 2 * p
                    dst[outIdx] = cb[code * 2] * scale
                    dst[outIdx + 1] = cb[code * 2 + 1] * scale
                }
            }
        }
    }
}
