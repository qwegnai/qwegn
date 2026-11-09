// BenchSuites.swift — the shapes, the banks and the host references.
//
// Every candidate is VERIFIED before it is timed: the bench computes the
// same product on the host and refuses to report a time for a kernel that
// disagrees with it. A number for a kernel that computes the wrong thing is
// worse than no number at all.

import Foundation
import Metal
import QWQuant

/// One measured configuration.
struct BenchPoint {
    /// The row-count bucket — the GEMV's output-row count (its N extent).
    let nBucket: Int
    let op: String
    let bits: Int
    let group: Int
    let candidate: String
    let gpuMilliseconds: Double
}

/// A candidate kernel and the shape of the work it is handed.
enum BenchCandidate {
    /// One of the size-preserving family's decode GEMVs, on our one plane.
    case family(bits: Int, group: Int)
    /// The incumbent's 4-bit G32 GEMV, on the very same blocks.
    case incumbentQ4G32
    /// MLX's tile GEMM, on MLX's three planes.
    case mlxNAX(bits: Int, group: Int)

    /// The kernel name, exactly as the metallib spells it.
    var name: String {
        switch self {
        case .family(let bits, let group): return "qwg_gemv_aq\(bits)_g\(group)_v4"
        case .incumbentQ4G32: return "qwg_gemv_q4g_v4"
        case .mlxNAX: return "qwg_gemm_mlx_affine_nax"
        }
    }

    /// The operation the candidate serves. The engine only ever asks for
    /// "gemv"; the MLX candidate answers to its own name, because the engine
    /// cannot serve it — it holds MLX's three planes, not ours.
    var op: String {
        switch self {
        case .family, .incumbentQ4G32: return "gemv"
        case .mlxNAX: return "mlx_nax_gemm"
        }
    }

    /// The quantisation width the candidate reads.
    var bits: Int {
        switch self {
        case .family(let bits, _): return bits
        case .incumbentQ4G32: return 4
        case .mlxNAX(let bits, _): return bits
        }
    }

    /// The quantisation group the candidate reads.
    var group: Int {
        switch self {
        case .family(_, let group): return group
        case .incumbentQ4G32: return 32
        case .mlxNAX(_, let group): return group
        }
    }

    /// The bank the candidate reads. The candidates that share a key share
    /// their buffers, so two kernels race on the very same bytes. The
    /// incumbent's 4-bit G32 has a key of its own: a different container.
    var bankKey: String {
        switch self {
        case .family(let bits, let group):
            return "aq|\(bits)|\(group)"
        case .incumbentQ4G32:
            return "hq4|4|32"
        case .mlxNAX(let bits, let group):
            return "mlx|\(bits)|\(group)"
        }
    }
}

/// The weights behind a bank, regenerated from a seed so a bank can be built
/// for any row count without holding one.
enum LogicalWeights {
    /// The seed of one group of one row.
    static func seed(row: Int, group: Int) -> UInt64 {
        splitmix64(UInt64(bitPattern: Int64(row)) &* 0x0000_0001_0000_03B3
                   &+ UInt64(group) &* 0x9E37_79B9_7F4A_7C15)
    }

    /// The group's codes, in `0 ..< 2^bits`.
    static func codes(row: Int, group: Int, bits: Int, count: Int) -> [UInt32] {
        let s = seed(row: row, group: group)
        let mask = UInt64((1 << bits) - 1)
        var out = [UInt32](repeating: 0, count: count)
        for i in 0..<count { out[i] = UInt32(splitmix64(s &+ UInt64(i)) & mask) }
        return out
    }

    /// The group's scale: an f16 the format can hold, never zero.
    static func scale(row: Int, group: Int) -> Float {
        let s = splitmix64(seed(row: row, group: group) &+ 0x5DEE_CE66_D)
        let unit = Double(s % 1_000_000) / 1_000_000.0
        return Float(Float16(unit * 0.02 + 0.001))
    }

    /// The bias that centres the codes: the family's law is
    /// `value = code * scale + bias`, so the codes span
    /// `-2^(bits-1) .. 2^(bits-1)-1` times the scale.
    static func bias(bits: Int, scale: Float) -> Float {
        -Float(1 << (bits - 1)) * scale
    }

    /// The bias the INCUMBENT's 4-bit G32 container stores: the magic's own
    /// fold, `bias - 1024 * scale`, so that `(1024 + code) * scale + bias`
    /// is the same number as the family's `code * scale + bias`.
    static func incumbentBias(scale: Float) -> Float {
        -1032.0 * scale
    }

    @inline(__always)
    private static func splitmix64(_ x: UInt64) -> UInt64 {
        var z = x &+ 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// One row's host result: the value, and the magnitude the error against it
/// is measured. A dot product of thousands of terms can land near zero, so
/// the magnitude — not the value — is what the tolerance rides on.
struct BenchReference {
    let value: Float
    let magnitude: Float
}

/// The shapes the bench sweeps.
enum BenchSuites {
    /// The row-count buckets: the GEMV's output rows, the only extent a
    /// decode GEMV has. Every bucket is a multiple of 64, and the largest
    /// ones are the model's own projection widths.
    static let nBuckets = [64, 256, 1024, 4096, 14336]

    /// The activation width the bench holds fixed.
    static let defaultCols = 4096

    /// The families' own pairs: every one the family supports.
    static var pairs: [(bits: Int, group: Int)] {
        QWAQBlock.supported.map { (bits: $0.bits, group: $0.group) }
    }

    /// The candidates that serve the "gemv" operation.
    static func gemvCandidates(bits: Int, group: Int) -> [BenchCandidate] {
        var out: [BenchCandidate] = [.family(bits: bits, group: group)]
        if bits == 4 && group == 32 { out.append(.incumbentQ4G32) }
        return out
    }

    /// The MLX candidate, on its own operation. The kernel is one
    /// instantiation — 4-bit, group 64 — so no other pair may be asked for.
    static func mlxCandidates(bits: Int, group: Int) -> [BenchCandidate] {
        precondition(bits == 4 && group == 64,
                     "the MLX candidate kernel is instantiated for 4-bit group 64 only")
        return [.mlxNAX(bits: bits, group: group)]
    }
}

/// Our one-plane bank: the family's blocks, row after row. The reference row
/// reads the bank back with the SAME host reader the family implements, so
/// the bench catches a GPU/host disagreement, never a wrong reader.
enum AQBanks {
    /// Builds the bank for a `(bits, group)` pair.
    static func make(bits: Int, group: Int, rows: Int, cols: Int) -> Data {
        let groups = cols / group
        let bytesPerGroup = QWAQBlock.bytes(bits: bits, group: group)
        var bytes = [UInt8](repeating: 0, count: rows * groups * bytesPerGroup)
        for r in 0..<rows {
            for g in 0..<groups {
                let scale = LogicalWeights.scale(row: r, group: g)
                let codes = LogicalWeights.codes(row: r, group: g, bits: bits, count: group)
                let bias = LogicalWeights.bias(bits: bits, scale: scale)
                let block = try! QWAQBlock.pack(bits: bits, group: group, values: codes,
                                               scale: scale, bias: bias)
                let off = (r * groups + g) * bytesPerGroup
                bytes.replaceSubrange(off..<(off + bytesPerGroup), with: block)
            }
        }
        return Data(bytes)
    }

    /// The host reference for one row: the sum of the groups' dots, in the
    /// arithmetic the device uses.
    static func referenceRow(bits: Int, group: Int, row: Int, cols: Int,
                              bank: Data, x: [Float]) -> BenchReference {
        let groups = cols / group
        let bytesPerGroup = QWAQBlock.bytes(bits: bits, group: group)
        var value: Float = 0
        var magnitude: Float = 0
        bank.withUnsafeBytes { raw in
            for g in 0..<groups {
                let off = (row * groups + g) * bytesPerGroup
                let slice = UnsafeRawBufferPointer(start: raw.baseAddress! + off,
                                                   count: bytesPerGroup)
                let groupX = Array(x[(g * group)..<(g + 1) * group])
                value += QWAQBlock.dot(group: slice, bits: bits, group: group, x: groupX)
                magnitude += magnitudeOfGroup(bits: bits, group: group, slice: slice, x: groupX)
            }
        }
        return BenchReference(value: value, magnitude: magnitude)
    }

    /// The magnitude of one group's dot: the sum of its absolute terms.
    private static func magnitudeOfGroup(bits: Int, group: Int,
                                         slice: UnsafeRawBufferPointer, x: [Float]) -> Float {
        let (scale, bias) = QWAQBlock.header(slice)
        let payload = UnsafeRawBufferPointer(rebasing: slice.dropFirst(4))
        var total: Float = 0
        for i in 0..<group {
            let v = Float(QWAQBlock.readBits(payload, bits: bits, index: i))
            total += abs((v * scale + bias) * x[i])
        }
        return total
    }

    /// The INCUMBENT's container for the same logical weights: the magic's
    /// fold in the header and the de-interleaved nibbles in the payload.
    static func makeIncumbent4G32(rows: Int, cols: Int) -> Data {
        let groups = cols / 32
        let blockBytes = 20
        var bytes = [UInt8](repeating: 0, count: rows * groups * blockBytes)
        for r in 0..<rows {
            for g in 0..<groups {
                let scale = LogicalWeights.scale(row: r, group: g)
                let codes = LogicalWeights.codes(row: r, group: g, bits: 4, count: 32)
                let bias = Float(Float16(LogicalWeights.incumbentBias(scale: scale)))
                var body = [UInt8](repeating: 0, count: 16)
                for i in 0..<32 {
                    let word = i / 8
                    let shift = UInt32(QWAQProducer.hq4NibbleBitOffset(i % 8))
                    var current = UInt32(body[word * 4]) | (UInt32(body[word * 4 + 1]) << 8)
                        | (UInt32(body[word * 4 + 2]) << 16) | (UInt32(body[word * 4 + 3]) << 24)
                    current |= (codes[i] & 0xF) << shift
                    body[word * 4] = UInt8(current & 0xFF)
                    body[word * 4 + 1] = UInt8((current >> 8) & 0xFF)
                    body[word * 4 + 2] = UInt8((current >> 16) & 0xFF)
                    body[word * 4 + 3] = UInt8((current >> 24) & 0xFF)
                }
                let off = (r * groups + g) * blockBytes
                bytes[off] = UInt8(Float16(scale).bitPattern & 0xFF)
                bytes[off + 1] = UInt8(Float16(scale).bitPattern >> 8)
                bytes[off + 2] = UInt8(Float16(bias).bitPattern & 0xFF)
                bytes[off + 3] = UInt8(Float16(bias).bitPattern >> 8)
                for i in 0..<16 { bytes[off + 4 + i] = body[i] }
            }
        }
        return Data(bytes)
    }

    /// The host reference for one row of the incumbent's container: the
    /// magic's form, `(1024 + code) * scale + bias`, rounded to f16 — the
    /// device folds the code into a half and runs a **half** FMA.
    static func incumbentReferenceRow(row: Int, cols: Int, bank: Data, x: [Float]) -> BenchReference {
        let groups = cols / 32
        var value: Float = 0
        var magnitude: Float = 0
        bank.withUnsafeBytes { raw in
            for g in 0..<groups {
                let off = row * groups * 20 + g * 20
                let slice = UnsafeRawBufferPointer(start: raw.baseAddress! + off, count: 20)
                let scale = Float(Float16(bitPattern: UInt16(slice[0]) | (UInt16(slice[1]) << 8)))
                let bias = Float(Float16(bitPattern: UInt16(slice[2]) | (UInt16(slice[3]) << 8)))
                for w in 0..<4 {
                    let word = slice.loadUnaligned(fromByteOffset: 4 + w * 4, as: UInt32.self)
                    for j in 0..<8 {
                        let nib = word >> UInt32(QWAQProducer.hq4NibbleBitOffset(j)) & 0xF
                        let v = Float(Float16((1024.0 + Float(nib)) * scale + bias))
                        // The block's own slice of the activations: block `g`
                        // covers columns g * 32 ..< g * 32 + 32.
                        let product = v * x[g * 32 + w * 8 + j]
                        value += product
                        magnitude += abs(product)
                    }
                }
            }
        }
        return BenchReference(value: value, magnitude: magnitude)
    }
}

/// MLX's three planes for the same logical weights: the packed codes, one
/// scale per group and one bias per group.
enum MLXBanks {
    /// MLX packs 4 bits densely: byte `b` holds elements `2b` and `2b+1`,
    /// the low nibble first.
    static func make(bits: Int, group: Int, rows: Int, cols: Int) -> (w: Data, scales: [Float], biases: [Float]) {
        let groups = cols / group
        let payload = group * bits / 8
        var bytes = [UInt8](repeating: 0, count: rows * groups * payload)
        var scales = [Float](repeating: 0, count: rows * groups)
        var biases = [Float](repeating: 0, count: rows * groups)
        for r in 0..<rows {
            for g in 0..<groups {
                let scale = LogicalWeights.scale(row: r, group: g)
                let codes = LogicalWeights.codes(row: r, group: g, bits: bits, count: group)
                let offset = r * groups + g
                scales[offset] = scale
                // The device reads the f16 the bias is stored as, so the
                // bank holds that value and the reference uses it.
                biases[offset] = Float(Float16(-Float(1 << (bits - 1)) * scale))
                let blockOffset = offset * payload
                for i in 0..<group {
                    let byte = blockOffset + i / 2
                    let nibble = UInt8(codes[i] & 0xF)
                    bytes[byte] = (i % 2 == 0) ? (bytes[byte] | nibble)
                                               : (bytes[byte] | (nibble << 4))
                }
            }
        }
        return (Data(bytes), scales, biases)
    }

    /// The host reference for one row: the plain affine dot. MLX dequantises
    /// into an f16 tile before the multiply, so the reference rounds the
    /// weight the same way.
    static func referenceRow(bits: Int, group: Int, row: Int, cols: Int,
                              w: Data, scales: [Float], biases: [Float], x: [Float]) -> BenchReference {
        let groups = cols / group
        let payload = group * bits / 8
        var value: Float = 0
        var magnitude: Float = 0
        w.withUnsafeBytes { raw in
            for g in 0..<groups {
                let scale = scales[row * groups + g]
                let bias = biases[row * groups + g]
                let blockOffset = (row * groups + g) * payload
                for i in 0..<group {
                    let byte = raw[blockOffset + i / 2]
                    let code = (i % 2 == 0) ? Float(byte & 0x0F) : Float(byte >> 4)
                    let weight = Float(Float16(code * scale + bias))
                    value += weight * x[g * group + i]
                    magnitude += abs(weight * x[g * group + i])
                }
            }
        }
        return BenchReference(value: value, magnitude: magnitude)
    }
}
