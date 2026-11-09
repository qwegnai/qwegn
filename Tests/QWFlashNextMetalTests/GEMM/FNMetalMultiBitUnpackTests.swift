import XCTest
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWQuant

/// The size-preserving affine-quantised GEMV family against the host
/// reference, and the 4-bit G32 kernel against the incumbent — bit for bit.
final class FNMetalMultiBitUnpackTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    /// A bank of `(bits, group)` blocks: `n` rows of `k / group` groups.
    /// `realistic` selects the packer's own scale rule — `amax / 7.5`, whose
    /// f16 rounding is inexact — so the device's fp16 rounding is exercised.
    private func makeBank(bits: Int, group: Int, n: Int, k: Int,
                           realistic: Bool = false) -> Data {
        let groups = k / group
        var bank = Data()
        for r in 0..<n {
            for g in 0..<groups {
                let maxValue = UInt32((1 << bits) - 1)
                let values = (0..<group).map { i in
                    UInt32((i + r * 7 + g * 13) % Int(maxValue + 1))
                }
                var scale: Float = 0.03125
                var bias: Float = -0.5
                if realistic {
                    scale = 0.3 / 7.5
                    // The 4-bit widths carry the packer's bias law; the
                    // others carry a plain offset, and the device reads
                    // whatever bias it is given.
                    bias = bits == 4 ? -1032.0 * scale : -0.5 * scale
                }
                // The packer's own scale is inexact in f16, so the realistic
                // bank asks for the rounded header on purpose; the exact
                // bank goes through the family's writer, which refuses.
                let block = realistic
                    ? QWAQBlock.packRoundedHeader(bits: bits, group: group, values: values,
                                                  scale: scale, bias: bias)
                    : try! QWAQBlock.pack(bits: bits, group: group, values: values,
                                          scale: scale, bias: bias)
                bank.append(block)
            }
        }
        return bank
    }

    /// The activation vector, deterministic and small enough to survive f16.
    /// The family is a decode-shaped GEMV: one vector of `k` activations
    /// against `n` weight rows.
    private func makeActivations(k: Int) -> [Float] {
        var seed: UInt32 = 0x5EED_C0DE
        return (0..<k).map { _ in
            seed = seed &* 1664525 &+ 1013904223
            return Float(Int32(truncatingIfNeeded: seed >> 8) % 512) / 512.0 - 0.5
        }
    }

    private func step(_ f: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
        try f(enc)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "step failed: \(String(describing: cb.error))")
    }

    /// The host's answer for one row, over the activations the device sees.
    /// `magnitude` is the sum of the absolute products — the natural scale of
    /// the dot, and the yardstick a relative tolerance must use.
    private func hostRow(bits: Int, group: Int, row: Int, k: Int,
                         bank: Data, x: [Float], f32Activations: Bool) -> (value: Float, magnitude: Float) {
        let groups = k / group
        let bytesPerGroup = QWAQBlock.bytes(bits: bits, group: group)
        var acc = 0.0 as Float
        var magnitude = 0.0 as Float
        for g in 0..<groups {
            let offset = (row * groups + g) * bytesPerGroup
            let slice = bank.subdata(in: offset..<(offset + bytesPerGroup))
            let xs = (0..<group).map { f32Activations ? x[g * group + $0] : Float(Float16(x[g * group + $0])) }
            acc += slice.withUnsafeBytes { QWAQBlock.dot(group: $0, bits: bits, group: group, x: xs) }
            var weights: [Float] = []
            slice.withUnsafeBytes { QWAQBlock.unpack(group: $0, bits: bits, group: group, into: &weights) }
            for i in 0..<group { magnitude += abs(weights[i] * xs[i]) }
        }
        return (acc, magnitude)
    }

    private func runFamily(bits: Int, group: Int, n: Int, k: Int,
                            realistic: Bool, f32Activations: Bool) throws -> [Float] {
        let bank = makeBank(bits: bits, group: group, n: n, k: k, realistic: realistic)
        let xs = makeActivations(k: k)
        let bytes = bank.count
        let arena = try QWArena(hal: Self.hal, weights: bytes + (8 << 20),
                                activations: (8 << 20), kv: 0, gdnStates: 0)
        let w = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
        arena.copyIn(w, from: bank)
        let act = arena.activationsRegion()
        let x32 = try act.allocate(QWTensorShape([k], .f32))
        arena.write(x32, from: xs)
        let x16 = try act.allocate(QWTensorShape([k], .f16))
        let y = try act.allocate(QWTensorShape([n], .f32))
        if !f32Activations {
            try step { enc in try QWGemm.encodeCast(enc, hal: Self.hal, x32: x32, y16: x16, count: k) }
        }
        try step { enc in
            try QWGemmAQ.encode(enc, hal: Self.hal, blocks: w,
                                x: f32Activations ? x32 : x16, y: y,
                                n: n, k: k, bits: bits, group: group,
                                f32Activations: f32Activations)
        }
        return arena.read(y, as: Float.self, count: n)
    }

    func testTheFamilyMatchesTheHostReference() throws {
        // The f32 twin computes in fp32, so it agrees with the host to fp32
        // precision. The f16 twin rounds every weight to fp16, so it agrees
        // to fp16 precision — one ulp of fp16 is 2^-11.
        for realistic in [false, true] {
            for (bits, group) in QWAQBlock.supported {
                let n = 4, k = group * 2
                let bank = makeBank(bits: bits, group: group, n: n, k: k, realistic: realistic)
                let xs = makeActivations(k: k)
                for f32Activations in [false, true] {
                    let got = try runFamily(bits: bits, group: group, n: n, k: k,
                                            realistic: realistic, f32Activations: f32Activations)
                    let tolerance = f32Activations ? 1e-5 : 1e-3
                    for r in 0..<n {
                        let ref = hostRow(bits: bits, group: group, row: r, k: k,
                                          bank: bank, x: xs, f32Activations: f32Activations)
                        let scale = max(abs(Double(ref.value)), Double(ref.magnitude), 1e-6)
                        let rel = abs(Double(got[r]) - Double(ref.value)) / scale
                        XCTAssertLessThan(rel, tolerance,
                                          "\(bits)-bit group \(group) row \(r) realistic \(realistic): got \(got[r]), want \(ref.value)")
                    }
                }
            }
        }
    }

}
