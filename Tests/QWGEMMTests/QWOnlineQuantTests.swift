import XCTest
import Foundation
@testable import QWQuant

/// The load-time producers: the BF16 quantiser and the MLX repacker.
/// The contract is the bytes: what the incumbent packer writes, and MLX's
/// codes without moving one.
final class QWOnlineQuantTests: XCTestCase {
    func testTheRepackerPreservesTheByteCountExactly() {
        // MLX affine 4-bit G64: 32 bytes of nibbles + 4 bytes of scale and
        // bias per group. Ours is the same 36 bytes in one plane.
        XCTAssertEqual(QWAQBlock.bytes(bits: 4, group: 64), 36)
        // And 4-bit G32 must NOT be used as a substitute — that would be 20
        // bytes and would change the bpw (Law L8).
        XCTAssertNotEqual(QWAQBlock.bytes(bits: 4, group: 32), 36)
    }

    func testTheRepackerKeepsEveryDequantisedValue() throws {
        // MLX: value = scale * q + bias. Ours: the same, exactly — the
        // header holds the group's OWN scale and bias, with no fold. bf16's
        // 8 significand bits fit f16's 11, so the header holds them exactly.
        let scale = bf16(0.01)
        let bias = bf16(-0.5)
        for (bits, group) in [(2, 64), (3, 64), (4, 64), (6, 64), (8, 64)] {
            let maxCode = UInt32((1 << bits) - 1)
            let codes = (0..<group).map { UInt8($0 % Int(maxCode + 1)) }
            let payload = mlxPayload(bits: bits, group: group, codes: codes)
            let block = try QWAQProducer.repackMLXGroup(bits: bits, group: group, packed: payload,
                                                        scale: scale, bias: bias)
            var values: [Float] = []
            block.withUnsafeBytes { QWAQBlock.unpack(group: $0, bits: bits, group: group, into: &values) }
            for i in 0..<group {
                let mlx = scale * Float(codes[i]) + bias
                XCTAssertEqual(Double(values[i]), Double(mlx), accuracy: 1e-6,
                               "\(bits)-bit element \(i)")
            }
        }
    }

    /// The header is f16, and a value f16 cannot hold exactly is refused
    /// rather than rounded: a rounded bias is a different weight.
    func testAHeaderValueF16CannotHoldIsRefused() {
        // A bf16 value in f16's normal range fits exactly.
        XCTAssertTrue(QWAQProducer.headerHolds(scale: bf16(0.01), bias: bf16(-0.5)))
        // Below f16's smallest normal: the value survives only as a
        // subnormal, and not always.
        XCTAssertFalse(QWAQProducer.headerHolds(scale: bf16(8.568e-08), bias: bf16(-1.0e-06)))
        // Above f16's largest: it becomes an infinity.
        XCTAssertFalse(QWAQProducer.headerHolds(scale: 70000, bias: 0))
        // A plain f32 value with more than 11 significand bits does not fit
        // either — the checkpoint's values are bf16, but ours are not.
        XCTAssertFalse(QWAQProducer.headerHolds(scale: 0.01, bias: -0.5))
    }

    /// An f16 SUBNORMAL is exact, so the rule must accept it: the real
    /// `Qwen3-8B-4bit` checkpoint has 87,030 scale and bias values below f16's
    /// smallest normal, and every one of them round-trips exactly.
    func testASubnormalHeaderIsAcceptedBecauseItIsExact() {
        // 916 units of 2^-24 — below the smallest normal, exactly on the grid.
        let subnormal = Float(916) / Float(16_777_216)
        XCTAssertLessThan(subnormal, Float(1) / Float(16_384),
                          "the value must be below f16's smallest normal")
        XCTAssertTrue(QWAQProducer.headerHolds(scale: subnormal, bias: -subnormal))
        // Half a unit off the grid is not representable, and is refused.
        XCTAssertFalse(QWAQProducer.headerHolds(scale: Float(916.5) / Float(16_777_216), bias: 0))
    }

    /// The host twin REFUSES BY NAME — it throws a named error, it does not
    /// trap and it does not round.
    func testTheRepackerRefusesByNameRatherThanTrapping() {
        let payload = [UInt8](repeating: 0, count: 24)
        XCTAssertThrowsError(
            try QWAQProducer.repackMLXGroup(bits: 3, group: 64, packed: payload,
                                            scale: 0.01, bias: -0.5)
        ) { error in
            guard case let QWAQProducerError.headerNotRepresentable(scale, bias) = error else {
                return XCTFail("the refusal must name the header: \(error)")
            }
            XCTAssertEqual(scale, 0.01)
            XCTAssertEqual(bias, -0.5)
        }
    }

    func testTheGroupIdMapsFlatScaleArraysOntoInterleavedRows() throws {
        // The repacker reads group g of row r from mlxScales[r * groups + g].
        // A wrong index silently gives a group another group's scale, so the
        // mapping is checked against the bytes that come out.
        let rows = 2, groups = 3, group = 64
        var scales: [Float] = []
        for r in 0..<rows {
            for g in 0..<groups { scales.append(bf16(Float(100 * r + g + 1) * 0.01)) }
        }
        for r in 0..<rows {
            for g in 0..<groups {
                let block = try QWAQProducer.repackMLXGroup(bits: 8, group: group,
                                                            packed: [UInt8](repeating: 0, count: group),
                                                            scale: scales[r * groups + g], bias: 0)
                let bytes = [UInt8](block)
                let stored = Float(Float16(bitPattern: UInt16(bytes[0]) | (UInt16(bytes[1]) << 8)))
                XCTAssertEqual(stored, Float(Float16(scales[r * groups + g])),
                               "row \(r) group \(g) must carry group \(r * groups + g)'s scale")
            }
        }
    }

    /// A bf16 value, as a checkpoint writes it: the value rounded to bf16's
    /// 8 significand bits. f16 holds 11, so these fit exactly.
    private func bf16(_ v: Float) -> Float { QWBF16.toFloat(QWBF16.fromFloatRNE(v)) }

    /// MLX's payload for a code array: one continuous little-endian bit
    /// stream, `bits` bits per code — the same bytes ours are.
    private func mlxPayload(bits: Int, group: Int, codes: [UInt8]) -> [UInt8] {
        let payload = group * bits / 8
        var out = [UInt8](repeating: 0, count: payload)
        for (i, code) in codes.enumerated() {
            let bit = i * bits
            let byte = bit >> 3
            let off = UInt32(bit & 7)
            let take = min(8 - Int(off), bits)
            let mask = UInt32((1 << take) - 1)
            out[byte] = UInt8(UInt32(out[byte]) | ((UInt32(code) & mask) << off))
            if take < bits {
                // The code's high bits start the NEXT byte, at its bit 0.
                out[byte + 1] = UInt8(UInt32(out[byte + 1]) | ((UInt32(code) >> UInt32(take)) << 0))
            }
        }
        return out
    }

    func testEveryWidthsPayloadIsCopiedVerbatim() throws {
        // One continuous little-endian bit stream in MLX and in ours, for
        // EVERY width — so the repack is a copy, and no code moves.
        for (bits, group) in [(2, 64), (3, 64), (4, 64), (6, 64), (8, 64)] {
            let payload = group * bits / 8
            let packed = (0..<payload).map { UInt8(($0 * 7 + 3) % 256) }
            let block = try QWAQProducer.repackMLXGroup(bits: bits, group: group, packed: packed,
                                                        scale: 1, bias: 0)
            XCTAssertEqual(block.count, payload + 4, "\(bits)-bit group")
            XCTAssertEqual([UInt8](block.dropFirst(4)), packed, "\(bits)-bit payload")
        }
    }

    func testTheQuantiserUsesTheIncumbentsScaleRule() {
        // The scale is `min(amax / 7.5, 63.46875)` on the ROTATED weights,
        // and the bias is `-1032 * scale`. The rotation is the packer's
        // FWHT-32, so the amax is taken after it.
        var values = [Float](repeating: 0, count: 32)
        values[0] = 1
        let block = QWAQProducer.hq4G32Block(values)
        let bytes = [UInt8](block)
        let scale = Float(Float16(bitPattern: UInt16(bytes[0]) | (UInt16(bytes[1]) << 8)))
        let bias = Float(Float16(bitPattern: UInt16(bytes[2]) | (UInt16(bytes[3]) << 8)))
        // One unit in one slot rotates to 1/sqrt(32) in every slot.
        let amax = 1.0 / Float(32.0).squareRoot()
        XCTAssertEqual(scale, Float(Float16(min(amax / Float(7.5), Float(63.46875)))), accuracy: 1e-6)
        XCTAssertEqual(bias, Float(Float16(Float(-1032.0) * scale)), accuracy: 1e-4)
    }

    func testTheQuantiserWritesTheDeInterleavedCodes() {
        // A constant group rotates to one non-zero slot, so the codes are
        // 15 for that slot and 8 for the rest. Which nibble holds which
        // element is the de-interleave, and it is what the device reads.
        let values = [Float](repeating: 1, count: 32)
        let block = QWAQProducer.hq4G32Block(values)
        let bytes = [UInt8](block)
        for w in 0..<4 {
            let word = UInt32(bytes[4 + w * 4]) | (UInt32(bytes[5 + w * 4]) << 8)
                | (UInt32(bytes[6 + w * 4]) << 16) | (UInt32(bytes[7 + w * 4]) << 24)
            // The code of element 8w+j sits at 4*(j>>1) + 16*(j&1).
            for j in 0..<8 {
                let nib = (word >> UInt32(QWAQProducer.hq4NibbleBitOffset(j))) & 0xF
                let want: UInt32 = (w == 0 && j == 0) ? 15 : 8
                XCTAssertEqual(nib, want, "word \(w) element \(j)")
            }
        }
    }
}
