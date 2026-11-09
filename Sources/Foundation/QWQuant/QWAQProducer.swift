// QWAQProducer.swift — the two producers: the BF16 quantiser and the MLX
// repacker.
//
// They are the host side of the contract the Metal writers implement, and
// the bytes they produce are the incumbent Python packer's, exactly.

import Foundation

/// The load-time producers for the size-preserving affine-quantised family.
public enum QWAQProducer {
    /// The incumbent's `Block_HQ4_G32` encoder, on the host: 32 weights in,
    /// 20 bytes out, byte for byte what `Scripts/quantize/hq4.py` writes.
    public static func hq4G32Block(_ values: [Float]) -> Data {
        precondition(values.count == 32, "a HQ4_G32 block holds 32 weights")
        var v = values
        v.withUnsafeMutableBufferPointer { QWQuant.fwht32OrthoBlock($0.baseAddress!) }
        var amax: Float = 0
        for x in v { amax = max(amax, abs(x)) }
        let scale = Float16(min(amax / Float(7.5), Float(63.46875)))
        let bias = Float16(Float(-1032.0) * Float(scale))
        let exactScale = Float(scale)
        let exactBias = Float(bias)
        let safeScale = exactScale > 0 ? exactScale : 1
        var codes = [UInt32](repeating: 0, count: 32)
        for i in 0..<32 {
            if exactScale > 0 {
                let f = ((v[i] - exactBias) / safeScale) - Float(1023.5)
                codes[i] = UInt32(min(max(floor(f), 0), 15))
            } else {
                codes[i] = 8
            }
        }
        var body = [UInt8](repeating: 0, count: 16)
        for i in 0..<32 {
            let word = i / 8
            let shift = UInt32(4 * ((i % 8) >> 1) + 16 * ((i % 8) & 1))
            var current = UInt32(body[word * 4]) | (UInt32(body[word * 4 + 1]) << 8)
                | (UInt32(body[word * 4 + 2]) << 16) | (UInt32(body[word * 4 + 3]) << 24)
            current |= (codes[i] & 0xF) << shift
            body[word * 4] = UInt8(current & 0xFF)
            body[word * 4 + 1] = UInt8((current >> 8) & 0xFF)
            body[word * 4 + 2] = UInt8((current >> 16) & 0xFF)
            body[word * 4 + 3] = UInt8((current >> 24) & 0xFF)
        }
        var out = Data()
        out.append(contentsOf: withUnsafeBytes(of: Float16(scale).bitPattern.littleEndian) { Array($0) })
        out.append(contentsOf: withUnsafeBytes(of: Float16(bias).bitPattern.littleEndian) { Array($0) })
        out.append(contentsOf: body)
        return out
    }

    /// The incumbent's `Block_HQ4_G32` encoder, from codes: 32 codes in,
    /// 20 bytes out, with the magic's fold in the header and the
    /// de-interleaved nibbles in the payload.
    public static func hq4G32Block(codes: [UInt32], scale: Float) -> Data {
        precondition(codes.count == 32, "a HQ4_G32 block holds 32 codes")
        let storedScale = Float16(scale)
        let folded = Float16(-1032.0 * Float(storedScale))
        var body = [UInt8](repeating: 0, count: 16)
        for i in 0..<32 {
            let word = i / 8
            let shift = UInt32(hq4NibbleBitOffset(i % 8))
            var current = UInt32(body[word * 4]) | (UInt32(body[word * 4 + 1]) << 8)
                | (UInt32(body[word * 4 + 2]) << 16) | (UInt32(body[word * 4 + 3]) << 24)
            current |= (codes[i] & 0xF) << shift
            body[word * 4] = UInt8(current & 0xFF)
            body[word * 4 + 1] = UInt8((current >> 8) & 0xFF)
            body[word * 4 + 2] = UInt8((current >> 16) & 0xFF)
            body[word * 4 + 3] = UInt8((current >> 24) & 0xFF)
        }
        var out = Data()
        out.append(contentsOf: withUnsafeBytes(of: storedScale.bitPattern.littleEndian) { Array($0) })
        out.append(contentsOf: withUnsafeBytes(of: folded.bitPattern.littleEndian) { Array($0) })
        out.append(contentsOf: body)
        return out
    }

    /// Can the header hold this scale and this bias exactly? A value f16
    /// cannot hold is refused, not rounded: a rounded bias is a different
    /// weight. f16 SUBNORMALS are exact and are accepted.
    public static func headerHolds(scale: Float, bias: Float) -> Bool {
        Float(Float16(scale)) == scale && Float(Float16(bias)) == bias
    }

    /// One group of an MLX affine tensor, repacked into our one plane. The
    /// payload is copied verbatim; the header holds the group's own scale and
    /// bias. A header f16 cannot hold exactly is REFUSED BY NAME, not rounded.
    public static func repackMLXGroup(bits: Int, group: Int, packed: [UInt8],
                                       scale: Float, bias: Float) throws -> Data {
        let payload = group * bits / 8
        precondition(packed.count == payload, "a group holds \(payload) payload bytes")
        guard headerHolds(scale: scale, bias: bias) else {
            throw QWAQProducerError.headerNotRepresentable(scale: scale, bias: bias)
        }
        var out = Data()
        out.append(contentsOf: withUnsafeBytes(of: Float16(scale).bitPattern.littleEndian) { Array($0) })
        out.append(contentsOf: withUnsafeBytes(of: Float16(bias).bitPattern.littleEndian) { Array($0) })
        out.append(contentsOf: packed)
        return out
    }

    /// The incumbent's `Block_HQ4_G32` nibble layout: element `8w + j` sits
    /// at bit offset `4 * (j >> 1) + 16 * (j & 1)` of the word.
    public static func hq4NibbleBitOffset(_ j: Int) -> Int { 4 * (j >> 1) + 16 * (j & 1) }
}

/// What the family's producers refuse, by name.
public enum QWAQProducerError: Error, CustomStringConvertible {
    /// The block's f16 header cannot hold this scale or this bias exactly.
    /// Rounding it would decode to a different weight, so the group is
    /// refused instead.
    case headerNotRepresentable(scale: Float, bias: Float)

    public var description: String {
        switch self {
        case let .headerNotRepresentable(scale, bias):
            return "the f16 header cannot hold scale \(scale) and bias \(bias) exactly"
        }
    }
}
