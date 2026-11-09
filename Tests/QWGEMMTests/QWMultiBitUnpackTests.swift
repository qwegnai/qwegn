import XCTest
import Foundation
@testable import QWQuant

/// The size law is the whole point of Law L8: a (bits, group) pair must
/// occupy exactly its native bytes, so a repack never inflates DRAM traffic.
final class QWMultiBitUnpackTests: XCTestCase {
    func testNativeByteSizesArePreserved() {
        XCTAssertEqual(QWAQBlock.bytes(bits: 2, group: 64), 20)
        XCTAssertEqual(QWAQBlock.bytes(bits: 3, group: 64), 28)
        XCTAssertEqual(QWAQBlock.bytes(bits: 4, group: 32), 20)
        XCTAssertEqual(QWAQBlock.bytes(bits: 4, group: 64), 36)
        XCTAssertEqual(QWAQBlock.bytes(bits: 4, group: 128), 68)
        XCTAssertEqual(QWAQBlock.bytes(bits: 6, group: 64), 52)
        XCTAssertEqual(QWAQBlock.bytes(bits: 8, group: 64), 68)
    }

    func testBitsPerWeightNeverExceedsTheNativeRate() {
        // A 4-bit G64 checkpoint is 4.50 bpw; widening it to G32 (5.00 bpw)
        // would inflate traffic by 11%. The helper must refuse to do that.
        XCTAssertEqual(QWAQBlock.bitsPerWeight(bits: 4, group: 64), 4.5, accuracy: 0.001)
        XCTAssertEqual(QWAQBlock.bitsPerWeight(bits: 4, group: 128), 4.25, accuracy: 0.001)
        XCTAssertEqual(QWAQBlock.bitsPerWeight(bits: 2, group: 64), 2.5, accuracy: 0.001)
    }

    func testTheFourBitG32BlockIsTheIncumbentsSIZE() {
        // 4-bit G32 is 20 bytes, the incumbent's `Block_HQ4_G32` size — but
        // not its content: the header here is the group's own scale and bias,
        // and the payload is a bit stream, not the incumbent's word layout.
        XCTAssertEqual(QWAQBlock.bytes(bits: 4, group: 32), QWQuant.blockBytes)
    }

    func testEveryWidthHasOneArithmeticAndNoMagic() {
        // The block means `code * scale + bias` for EVERY width. The
        // incumbent's magic form would need the bias folded against the
        // scale: on a real checkpoint that is 0.44 of a step off.
        let values = (0..<64).map { UInt32($0 % 16) }
        for (bits, group) in [(2, 64), (3, 64), (4, 64), (6, 64), (8, 64)] {
            var out: [Float] = []
            makeBlock(bits: bits, group: group, values: values, scale: 1, bias: -1032)
                .withUnsafeBytes { QWAQBlock.unpack(group: $0, bits: bits, group: group, into: &out) }
            for i in 0..<group {
                let code = Double(values[i] & UInt32((1 << bits) - 1))
                XCTAssertEqual(Double(out[i]), code - 1032.0, accuracy: 1e-3,
                               "\(bits)-bit element \(i)")
            }
        }
    }

    func testTheBlockLayoutIsScaleThenBiasThenPayload() {
        // A block is `half scale; half bias; payload` — the same shape as
        // `Block_HQ4_G32`, so the bytes move the same way.
        let block = makeBlock(bits: 4, group: 32, values: (0..<32).map { UInt32($0 % 16) },
                              scale: 0.5, bias: -1)
        XCTAssertEqual(block.count, 20)
        let bytes = [UInt8](block)
        XCTAssertEqual(UInt16(bytes[0]) | (UInt16(bytes[1]) << 8),
                       Float16(0.5).bitPattern, "the scale is the first f16")
        XCTAssertEqual(UInt16(bytes[2]) | (UInt16(bytes[3]) << 8),
                       Float16(-1).bitPattern, "the bias is the second f16")
    }

    func testEverySupportedPairIsItsOwnNativeSize() {
        for (bits, group) in QWAQBlock.supported {
            XCTAssertEqual(QWAQBlock.bytes(bits: bits, group: group),
                           QWAQBlock.payloadBytes(bits: bits, group: group) + 4)
            XCTAssertTrue(QWAQBlock.isSupported(bits: bits, group: group))
        }
    }

    func testEveryWidthPacksAsOneBitStream() {
        // One continuous little-endian bit stream, for every width — the
        // bytes MLX writes, so the repack is a copy.
        for (bits, group) in [(2, 64), (3, 64), (4, 64), (6, 64), (8, 64)] {
            let maxValue = (1 << bits) - 1
            let values = (0..<group).map { UInt32($0 % (maxValue + 1)) }
            let block = makeBlock(bits: bits, group: group, values: values)
            var out: [Float] = []
            block.withUnsafeBytes { QWAQBlock.unpack(group: $0, bits: bits, group: group, into: &out) }
            XCTAssertEqual(out.count, group, "\(bits)-bit group")
            for i in 0..<group {
                XCTAssertEqual(Double(out[i]), Double(values[i]), accuracy: 1e-3, "\(bits)-bit element \(i)")
            }
        }
    }

    func testTheIncumbentsWordLayoutStaysTheIncumbents() {
        // The incumbent stores the even elements of a word in its low 16
        // bits and the odd ones in its high 16 bits. The family does not use
        // that layout — but the incumbent's packer does, so its table is
        // pinned here, where the packer can find it.
        XCTAssertEqual((0..<8).map { QWAQProducer.hq4NibbleBitOffset($0) },
                       [0, 16, 4, 20, 8, 24, 12, 28])
    }

    func testTheScaleAndBiasAreTheBlockHeader() {
        // scale 2, bias 3: the value is n * 2 + 3.
        let values = (0..<64).map { UInt32($0) }
        let block = makeBlock(bits: 8, group: 64, values: values, scale: 2, bias: 3)
        var out: [Float] = []
        block.withUnsafeBytes { QWAQBlock.unpack(group: $0, bits: 8, group: 64, into: &out) }
        XCTAssertEqual(Double(out[0]), 3.0, accuracy: 1e-3)
        XCTAssertEqual(Double(out[1]), 5.0, accuracy: 1e-3)
    }

    func testTheDotIsTheWeightOfEveryElement() {
        // The dot is the plain one: every element's weight, times its
        // activation, summed.
        for (bits, group) in [(4, 64), (8, 64)] {
            let values = (0..<group).map { UInt32($0 % ((1 << bits) - 1 + 1)) }
            let block = makeBlock(bits: bits, group: group, values: values, scale: 2, bias: 3)
            var x = [Float](repeating: 0, count: group)
            for i in 0..<group { x[i] = Float(i % 7) - 3.0 }
            let got = block.withUnsafeBytes { QWAQBlock.dot(group: $0, bits: bits, group: group, x: x) }
            var expected = 0.0
            for i in 0..<group {
                expected += (Double(values[i]) * 2.0 + 3.0) * Double(x[i])
            }
            XCTAssertEqual(Double(got), Double(expected), accuracy: 1e-3, "\(bits)-bit dot")
        }
    }

    func testTheRepackerCopiesThePayloadAndKeepsTheGroupsOwnScaleAndBias() throws {
        // The repack is a copy: MLX's payload bytes are the family's payload
        // bytes, and the header is the group's own scale and bias — no fold.
        let payload = (0..<24).map { UInt8($0 & 0xFF) }
        let block = try QWAQProducer.repackMLXGroup(bits: 3, group: 64, packed: payload,
                                                    scale: 0.03125, bias: -0.125)
        XCTAssertEqual(block.count, 28)
        let bytes = [UInt8](block)
        XCTAssertEqual(Array(bytes[4...]), payload)
        var out: [Float] = []
        block.withUnsafeBytes { QWAQBlock.unpack(group: $0, bits: 3, group: 64, into: &out) }
        XCTAssertEqual(Double(out[0]), 0.0 * 0.03125 - 0.125, accuracy: 1e-6)
    }

    /// A block in the family's layout: f16 scale, f16 bias, then the payload.
    private func makeBlock(bits: Int, group: Int, values: [UInt32],
                           scale: Float = 1.0, bias: Float = 0.0) -> Data {
        try! QWAQBlock.pack(bits: bits, group: group, values: values, scale: scale, bias: bias)
    }
}
