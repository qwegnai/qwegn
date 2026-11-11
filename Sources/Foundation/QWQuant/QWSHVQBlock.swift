// QWSHVQBlock.swift — Size law and 0-ULP host reference unpacker for the
// Signed Hadamard e-means Vector Quantized family (SHVQ{2,3,4}_G{32,64,128,256}).
// Each group stores a 2-byte f16 scale header (zero bias) plus group*bits/8 payload bytes.

import Foundation

/// Signed Hadamard `e-means` (`arXiv:2611.09501`) vector-quantized block family.
public enum QWSHVQBlock {
    /// Supported `(bits, group)` pairs for `SHVQ`.
    public static let supported: [(bits: Int, group: Int)] = [
        (2, 32), (2, 64), (2, 128), (2, 256),
        (3, 32), (3, 64), (3, 128), (3, 256),
        (4, 32), (4, 64), (4, 128), (4, 256),
    ]

    /// True when `(bits, group)` is a valid `SHVQ` pair.
    public static func isSupported(bits: Int, group: Int) -> Bool {
        supported.contains { $0.bits == bits && $0.group == group }
    }

    /// Payload bytes of one group, excluding the 2-byte `f16` scale header.
    public static func payloadBytes(bits: Int, group: Int) -> Int {
        precondition(group % 32 == 0 && group > 0, "group must be a positive multiple of 32")
        precondition((2...4).contains(bits), "SHVQ bits must be 2...4")
        return (group * bits) / 8
    }

    /// Total bytes per group (`2`-byte `f16` scale + payload).
    public static func bytes(bits: Int, group: Int) -> Int {
        2 + payloadBytes(bits: bits, group: group)
    }

    /// Total bytes per group if `(bits, group)` is supported; otherwise `nil`.
    public static func bytesIfSupported(bits: Int, group: Int) -> Int? {
        guard isSupported(bits: bits, group: group) else { return nil }
        return bytes(bits: bits, group: group)
    }

    /// Bytes per row of an `(n, k)` bank of `(bits, group)` `SHVQ` blocks.
    public static func rowBytes(bits: Int, group: Int, k: Int) -> Int {
        (k / group) * bytes(bits: bits, group: group)
    }

    /// Canonical format name (`SHVQ{bits}_G{group}`).
    public static func formatName(bits: Int, group: Int) -> String {
        "SHVQ\(bits)_G\(group)"
    }

    /// Effective bits per weight including the 2-byte `f16` scale header.
    public static func bitsPerWeight(bits: Int, group: Int) -> Double {
        Double(bytes(bits: bits, group: group) * 8) / Double(group)
    }

    /// Reads the little-endian `f16` group scale widened to `Float`.
    @inline(__always)
    public static func scale(_ bytes: UnsafeRawBufferPointer) -> Float {
        let bits = UInt16(bytes[0]) | (UInt16(bytes[1]) << 8)
        return Float(Float16(bitPattern: bits))
    }

    /// Unpacks one rotated-domain `SHVQ` group into `out` at 0-ULP parity with Python & Metal.
    public static func unpack(
        group bytes: UnsafeRawBufferPointer,
        bits: Int,
        group: Int,
        into out: inout [Float]
    ) {
        let needed = self.bytes(bits: bits, group: group)
        precondition(bytes.count >= needed, "SHVQ group block is short")
        let sc = scale(bytes)
        let payload = UnsafeRawBufferPointer(rebasing: bytes.dropFirst(2))
        let nOct = group / 8
        out.removeAll(keepingCapacity: true)
        out.reserveCapacity(group)
        if bits == 2 {
            for oct in 0..<nOct {
                let signs = UInt16(payload[2 * oct])
                let lutBase = Int(payload[2 * oct + 1]) * 8
                for j in 0..<8 {
                    let signBit = ((signs >> j) & 1) << 15
                    let wBits = QWSHVQTables.lut2Bits[lutBase + j] | signBit
                    out.append(Float(Float16(bitPattern: wBits)) * sc)
                }
            }
        } else if bits == 3 {
            for oct in 0..<nOct {
                let signs = UInt16(payload[3 * oct])
                let lut0 = Int(payload[3 * oct + 1]) * 4
                let lut1 = Int(payload[3 * oct + 2]) * 4
                for j in 0..<4 {
                    let s0 = ((signs >> j) & 1) << 15
                    out.append(Float(Float16(bitPattern: QWSHVQTables.lut3Bits[lut0 + j] | s0)) * sc)
                }
                for j in 0..<4 {
                    let s1 = ((signs >> (j + 4)) & 1) << 15
                    out.append(Float(Float16(bitPattern: QWSHVQTables.lut3Bits[lut1 + j] | s1)) * sc)
                }
            }
        } else {
            for sub in 0..<(nOct * 2) {
                let u16 = UInt16(payload[2 * sub]) | (UInt16(payload[2 * sub + 1]) << 8)
                let idx12 = Int(u16 & 0x0FFF)
                let signs4 = (u16 >> 12) & 0x000F
                let packed = QWSHVQTables.lut4PackedU8[idx12]
                for j in 0..<4 {
                    let stepIdx = Int((packed >> (j * 8)) & 0xFF)
                    let sBit = ((signs4 >> j) & 1) << 15
                    let wBits = QWSHVQTables.stepMap4Bits[stepIdx] | sBit
                    out.append(Float(Float16(bitPattern: wBits)) * sc)
                }
            }
        }
    }

    /// Host reference dot product of one rotated-domain `SHVQ` group against `x`.
    public static func dot(
        group bytes: UnsafeRawBufferPointer,
        bits: Int,
        group: Int,
        x: [Float]
    ) -> Float {
        precondition(x.count >= group, "activation vector is short")
        var unpacked = [Float]()
        unpack(group: bytes, bits: bits, group: group, into: &unpacked)
        var acc: Float = 0.0
        for i in 0..<group {
            acc += unpacked[i] * x[i]
        }
        return acc
    }
}
