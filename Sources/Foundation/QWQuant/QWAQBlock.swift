// QWAQBlock.swift — the size law for the size-preserving affine-quantised
// block family (Law L8). A (bits, group) pair occupies exactly its native
// bytes, so a repack never inflates DRAM traffic.
//
// The full layout — the bit stream and the scale/bias header — is written
// down in specs/04-qwquant.md.

import Foundation

/// The size law for the size-preserving affine-quantised block family.
/// One group is an f16 scale, an f16 bias, and a payload of
/// `group * bits / 8` bytes. See specs/04-qwquant.md.
public enum QWAQBlock {
    /// The supported `(bits, group)` pairs. (4, 32) is a group of 32, not the
    /// incumbent's `Block_HQ4_G32`: the payload is a bit stream here, not the
    /// incumbent's word layout, and the header is not folded.
    public static let supported: [(bits: Int, group: Int)] = [
        (2, 32), (2, 64), (2, 128), (2, 256),
        (3, 32), (3, 64), (3, 128), (3, 256),
        (4, 32), (4, 64), (4, 128), (4, 256),
        (6, 32), (6, 64), (6, 128), (6, 256),
        (8, 32), (8, 64), (8, 128), (8, 256),
    ]

    /// The `(bits, group)` pairs supported by rotated `HQ{2,3,4,6,8}_G{32,64,128,256}` blocks.
    public static let hqSupported: [(bits: Int, group: Int)] = [
        (2, 32), (2, 64), (2, 128), (2, 256),
        (3, 32), (3, 64), (3, 128), (3, 256),
        (4, 32), (4, 64), (4, 128), (4, 256),
        (6, 32), (6, 64), (6, 128), (6, 256),
        (8, 32), (8, 64), (8, 128), (8, 256),
    ]

    /// The payload bytes of one group, without the scale and bias.
    public static func payloadBytes(bits: Int, group: Int) -> Int {
        precondition(group % 32 == 0 && group > 0, "group must be a positive multiple of 32")
        precondition(bits > 0 && bits <= 8, "bits must be 1...8")
        return group * bits / 8
    }

    /// Bytes per group for `(bits, group)`: the payload plus the scale and
    /// the bias.
    public static func bytes(bits: Int, group: Int) -> Int {
        payloadBytes(bits: bits, group: group) + 4
    }

    /// The same size law, without trapping on a pair outside the family: a
    /// caller that accepts hostile input asks here first, so a bad pair is a
    /// refusal and not a crash. A pair outside `supported` has NO size.
    public static func bytesIfSupported(bits: Int, group: Int) -> Int? {
        guard isSupported(bits: bits, group: group) else { return nil }
        return bytes(bits: bits, group: group)
    }

    /// The bytes one row of an `(n, k)` bank of `(bits, group)` blocks takes.
    public static func rowBytes(bits: Int, group: Int, k: Int) -> Int {
        (k / group) * bytes(bits: bits, group: group)
    }

    /// The container's name, for diagnostics. It names the family's
    /// container only: the incumbent's `Block_HQ4_G32` keeps its own name,
    /// and the two are told apart by the basis, not by this string.
    public static func formatName(bits: Int, group: Int) -> String {
        "AQ\(bits)_G\(group)"
    }

    /// Bits per weight, including the scale and bias overhead.
    public static func bitsPerWeight(bits: Int, group: Int) -> Double {
        Double(bytes(bits: bits, group: group) * 8) / Double(group)
    }

    /// Is this a pair the family supports?
    public static func isSupported(bits: Int, group: Int) -> Bool {
        supported.contains { $0.bits == bits && $0.group == group }
    }

    /// Is this a pair the rotated `HQ{2,3,4,6,8}_G{32,64,128,256}` block family supports?
    public static func isHQSupported(bits: Int, group: Int) -> Bool {
        hqSupported.contains { $0.bits == bits && $0.group == group }
    }

    /// True when either the standard `AQ` family or the rotated `HQ` family supports `(bits, group)`.
    public static func isBlockSupported(bits: Int, group: Int) -> Bool {
        isSupported(bits: bits, group: group) || isHQSupported(bits: bits, group: group)
    }

    /// Bytes per group if supported by either `AQ` or `HQ` block layouts.
    public static func bytesIfBlockSupported(bits: Int, group: Int) -> Int? {
        guard isBlockSupported(bits: bits, group: group) else { return nil }
        return bytes(bits: bits, group: group)
    }

    /// Unpacks one group into `out`: the host reference for the family's
    /// arithmetic. It matches the device's twins to fp32 precision — see
    /// `specs/04-qwquant.md`.
    public static func unpack(group bytes: UnsafeRawBufferPointer,
                              bits: Int,
                              group: Int,
                              into out: inout [Float]) {
        precondition(bytes.count >= self.bytes(bits: bits, group: group), "group block is short")
        let (scale, bias) = header(bytes)
        let payload = UnsafeRawBufferPointer(rebasing: bytes.dropFirst(4))
        out = []
        out.reserveCapacity(group)
        for i in 0..<group {
            let code = Double(readBits(payload, bits: bits, index: i))
            out.append(Float(code * Double(scale) + Double(bias)))
        }
    }

    /// The block's f16 scale and bias, read little-endian — the same order
    /// `pack` writes them in, on every host.
    public static func header(_ bytes: UnsafeRawBufferPointer) -> (scale: Float, bias: Float) {
        let scaleBits = UInt16(bytes[0]) | (UInt16(bytes[1]) << 8)
        let biasBits = UInt16(bytes[2]) | (UInt16(bytes[3]) << 8)
        return (Float(Float16(bitPattern: scaleBits)), Float(Float16(bitPattern: biasBits)))
    }

    /// The integer at `index` in a continuous little-endian bit stream.
    public static func readBits(_ payload: UnsafeRawBufferPointer, bits: Int, index: Int) -> UInt32 {
        let bit = index * bits
        let byte = bit >> 3
        let off = UInt32(bit & 7)
        var v = UInt32(payload[byte]) >> off
        let taken = 8 - Int(off)
        if taken < bits {
            v |= UInt32(payload[byte + 1]) << UInt32(taken)
        }
        return v & ((1 << UInt32(bits)) - 1)
    }

    /// Packs one group: f16 scale, f16 bias, then the payload. A header f16
    /// cannot hold exactly is REFUSED, not rounded — see specs/04-qwquant.md.
    public static func pack(bits: Int, group: Int, values: [UInt32],
                            scale: Float = 1.0, bias: Float = 0.0) throws -> Data {
        precondition(values.count == group, "a group holds \(group) values")
        guard QWAQProducer.headerHolds(scale: scale, bias: bias) else {
            throw QWAQProducerError.headerNotRepresentable(scale: scale, bias: bias)
        }
        return packRoundedHeader(bits: bits, group: group, values: values, scale: scale, bias: bias)
    }

    /// Packs one group whose header is ROUNDED to f16 on purpose, so the
    /// block decodes to a different weight than the scale says. The format's
    /// own precision probe, not a way to write a bank.
    public static func packRoundedHeader(bits: Int, group: Int, values: [UInt32],
                                         scale: Float = 1.0, bias: Float = 0.0) -> Data {
        precondition(values.count == group, "a group holds \(group) values")
        let payload = payloadBytes(bits: bits, group: group)
        var out = Data()
        out.append(contentsOf: withUnsafeBytes(of: Float16(scale).bitPattern.littleEndian) { Array($0) })
        out.append(contentsOf: withUnsafeBytes(of: Float16(bias).bitPattern.littleEndian) { Array($0) })
        var body = [UInt8](repeating: 0, count: payload)
        for (i, v) in values.enumerated() {
            var bit = i * bits
            let end = bit + bits
            while bit < end {
                let byte = bit >> 3
                let off = UInt32(bit & 7)
                let take = min(8 - Int(off), end - bit)
                let mask = (UInt32(1) << UInt32(take)) - 1
                // A value that spans a byte boundary gives its high bits
                // to the second byte, so the value's own offset moves.
                let inValue = UInt32(bit - i * bits)
                let chunk = (v >> inValue) & mask
                body[byte] = UInt8((UInt32(body[byte]) & ~(mask << off)) | (chunk << off))
                bit += take
            }
        }
        out.append(contentsOf: body)
        return out
    }

    /// The dot product of one group against `x`, in the form the device
    /// kernels compute it: the weight of every element, built from its code,
    /// the group's scale and the group's bias. The host reference.
    public static func dot(group bytes: UnsafeRawBufferPointer,
                           bits: Int,
                           group: Int,
                           x: [Float]) -> Float {
        precondition(x.count >= group, "activation vector is short")
        let (scale, bias) = header(bytes)
        let payload = UnsafeRawBufferPointer(rebasing: bytes.dropFirst(4))
        var acc = 0.0 as Float
        for i in 0..<group {
            let v = Float(readBits(payload, bits: bits, index: i))
            acc += (v * scale + bias) * x[i]
        }
        return acc
    }
}
