// IEEE-754 round-to-nearest-even float-to-bfloat16 conversion and helpers.

import Foundation

public enum QWBF16 {
    /// Converts a 32-bit float to bfloat16 bit pattern using round-to-nearest-even (IEEE 754 RNE).
    @inline(__always)
    public static func fromFloatRNE(_ val: Float) -> UInt16 {
        let bits = val.bitPattern
        // If NaN, quiet the NaN and preserve high bits.
        if (bits & 0x7F80_0000) == 0x7F80_0000 && (bits & 0x007F_FFFF) != 0 {
            return UInt16(truncatingIfNeeded: (bits >> 16) | 0x0040)
        }
        // Round to nearest even: lsb is bit 16, round bit is bit 15, sticky bits are bits 14..0.
        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: rounded >> 16)
    }

    /// Converts a bfloat16 bit pattern back to a 32-bit float.
    @inline(__always)
    public static func toFloat(_ u16: UInt16) -> Float {
        Float(bitPattern: UInt32(u16) << 16)
    }

    /// True when `bits` is a supported MLX affine quantization bit-width (2, 3, 4, 5, 6, or 8).
    @inline(__always)
    public static func isSupportedMLXAffineBits(_ bits: Int) -> Bool {
        bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8
    }

    /// Unpacks one quantized integer code at column `col` from an MLX `.u32`-packed row.
    @inline(__always)
    public static func unpackMLXAffineQ(rowWords: UnsafePointer<UInt32>, col: Int, bits: Int) -> Float {
        switch bits {
        case 2, 4, 8:
            let valsPerWord = 32 / bits
            let mask: UInt32 = (1 << bits) - 1
            let word = UInt32(littleEndian: rowWords[col / valsPerWord])
            let shift = (col % valsPerWord) * bits
            return Float((word >> shift) & mask)
        case 3:
            let w = UnsafeRawPointer(rowWords).assumingMemoryBound(to: UInt8.self).advanced(by: (col >> 3) * 3)
            let u24 = UInt32(w[0]) | (UInt32(w[1]) << 8) | (UInt32(w[2]) << 16)
            return Float((u24 >> ((col & 7) * 3)) & 0x7)
        case 6:
            let w = UnsafeRawPointer(rowWords).assumingMemoryBound(to: UInt8.self).advanced(by: (col >> 2) * 3)
            let u24 = UInt32(w[0]) | (UInt32(w[1]) << 8) | (UInt32(w[2]) << 16)
            return Float((u24 >> ((col & 3) * 6)) & 0x3F)
        case 5:
            let w = UnsafeRawPointer(rowWords).assumingMemoryBound(to: UInt8.self).advanced(by: (col >> 3) * 5)
            let u40 = UInt64(w[0]) | (UInt64(w[1]) << 8) | (UInt64(w[2]) << 16)
                | (UInt64(w[3]) << 24) | (UInt64(w[4]) << 32)
            return Float((u40 >> ((col & 7) * 5)) & 0x1F)
        default:
            return 0
        }
    }
}
