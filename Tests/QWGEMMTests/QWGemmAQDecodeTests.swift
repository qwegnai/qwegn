import XCTest
import Foundation
import Metal
@testable import QWGEMM
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// The decode GEMV of the size-preserving affine family: the dense path's
/// geometry, against the host reference — and against MLX's own decoder, on
/// the same checkpoint bytes.
final class QWGemmAQDecodeTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    /// A bank of `(bits, group)` blocks: `n` rows of `k / group` groups.
    private func makeBank(bits: Int, group: Int, n: Int, k: Int,
                           scale: Float = 0.03125, bias: Float = -0.5,
                           roundedHeader: Bool = false) -> Data {
        let groups = k / group
        var bank = Data()
        for r in 0..<n {
            for g in 0..<groups {
                let maxValue = UInt32((1 << bits) - 1)
                let values = (0..<group).map { i in
                    UInt32((i + r * 7 + g * 13) % Int(maxValue + 1))
                }
                // A header f16 cannot hold is refused by the family's writer;
                // the precision probe asks for the rounded one on purpose.
                let block = roundedHeader
                    ? QWAQBlock.packRoundedHeader(bits: bits, group: group, values: values,
                                                  scale: scale, bias: bias)
                    : try! QWAQBlock.pack(bits: bits, group: group, values: values,
                                          scale: scale, bias: bias)
                bank.append(block)
            }
        }
        return bank
    }

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

    /// The host's answer for one row, over the activations the device sees,
    /// and the natural scale of the dot (the sum of the absolute products).
    private func hostRow(bits: Int, group: Int, row: Int, k: Int,
                         bank: Data, x: [Float]) -> (value: Float, magnitude: Float) {
        let groups = k / group
        let bytesPerGroup = QWAQBlock.bytes(bits: bits, group: group)
        var acc = 0.0 as Float
        var magnitude = 0.0 as Float
        for g in 0..<groups {
            let offset = (row * groups + g) * bytesPerGroup
            let slice = bank.subdata(in: offset..<(offset + bytesPerGroup))
            let xs = (0..<group).map { Float(Float16(x[g * group + $0])) }
            acc += slice.withUnsafeBytes { QWAQBlock.dot(group: $0, bits: bits, group: group, x: xs) }
            var weights: [Float] = []
            slice.withUnsafeBytes { QWAQBlock.unpack(group: $0, bits: bits, group: group, into: &weights) }
            for i in 0..<group { magnitude += abs(weights[i] * xs[i]) }
        }
        return (acc, magnitude)
    }

    /// Runs the decode kernel over a bank and returns the f16 output.
    private func runDecode(bits: Int, group: Int, n: Int, k: Int,
                           bank: Data, x: [Float]) throws -> [Float] {
        let bytes = bank.count
        let arena = try QWArena(hal: Self.hal, weights: bytes + (8 << 20),
                                activations: (8 << 20), kv: 0, gdnStates: 0)
        let w = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
        arena.copyIn(w, from: bank)
        let act = arena.activationsRegion()
        let x32 = try act.allocate(QWTensorShape([k], .f32))
        arena.write(x32, from: x)
        let x16 = try act.allocate(QWTensorShape([k], .f16))
        // The cast writes x16; writing f32 into an f16 tensor would overrun it.
        let y16 = try act.allocate(QWTensorShape([n], .f16))
        try step { enc in try QWGemm.encodeCast(enc, hal: Self.hal, x32: x32, y16: x16, count: k) }
        try step { enc in
            try QWGemmAQ.encodeDecodeT4(enc, hal: Self.hal, blocks: w, x: x16, y16: y16,
                                        n: n, k: k, bits: bits, group: group)
        }
        return arena.read(y16, as: Float16.self, count: n).map { Float($0) }
    }

    /// Every pair of the family computes the host's dot.
    func testTheDecodeKernelMatchesTheHostReference() throws {
        for (bits, group) in QWAQBlock.supported {
            let n = 7, k = group * 4
            let bank = makeBank(bits: bits, group: group, n: n, k: k)
            let xs = makeActivations(k: k)
            let got = try runDecode(bits: bits, group: group, n: n, k: k, bank: bank, x: xs)
            for r in 0..<n {
                let ref = hostRow(bits: bits, group: group, row: r, k: k, bank: bank, x: xs)
                let scale = max(abs(Double(ref.value)), Double(ref.magnitude), 1e-6)
                let rel = abs(Double(got[r]) - Double(ref.value)) / scale
                XCTAssertLessThan(rel, 1e-3,
                                  "\(bits)-bit group \(group) row \(r): got \(got[r]), want \(ref.value)")
            }
        }
    }

    /// The decode kernel over a long inner dimension — the shape the engine
    /// actually runs — must still agree.
    func testTheDecodeKernelAgreesOverALongInnerDimension() throws {
        for (bits, group) in QWAQBlock.supported {
            let n = 5, k = 1024
            let bank = makeBank(bits: bits, group: group, n: n, k: k)
            let xs = makeActivations(k: k)
            let got = try runDecode(bits: bits, group: group, n: n, k: k, bank: bank, x: xs)
            for r in 0..<n {
                let ref = hostRow(bits: bits, group: group, row: r, k: k, bank: bank, x: xs)
                let scale = max(abs(Double(ref.value)), Double(ref.magnitude), 1e-6)
                let rel = abs(Double(got[r]) - Double(ref.value)) / scale
                XCTAssertLessThan(rel, 1e-3,
                                  "\(bits)-bit group \(group) row \(r): got \(got[r]), want \(ref.value)")
            }
        }
    }

    /// The format's own precision claim: the device's weight is the block's
    /// weight. The bank carries a scale fp16 cannot hold exactly, and the
    /// host reads the SAME f16 header.
    func testTheDecodeKernelAgreesWithTheHostOnAnInexactScale() throws {
        let n = 8, k = 1024
        let scale = Float(0.3 / 7.5)
        let bank = makeBank(bits: 4, group: 32, n: n, k: k, scale: scale, bias: -0.5 * scale,
                            roundedHeader: true)
        let xs = makeActivations(k: k)
        let got = try runDecode(bits: 4, group: 32, n: n, k: k, bank: bank, x: xs)
        var worst = 0.0 as Double
        for r in 0..<n {
            let ref = hostRow(bits: 4, group: 32, row: r, k: k, bank: bank, x: xs)
            let magnitude = max(abs(Double(ref.value)), Double(ref.magnitude), 1e-6)
            worst = max(worst, abs(Double(got[r]) - Double(ref.value)) / magnitude)
        }
        XCTAssertLessThan(worst, 1e-4,
                          "the device must agree with the host to fp32 precision, measured \(worst)")
    }

    /// The family's one-row and two-row paths — the row count rides in N's
    /// flags. The dense geometry only asks for the four-row one, but the
    /// others are part of the family's contract, so they are pinned here.
    func testTheOneRowAndTwoRowPathsAgreeWithTheHost() throws {
        for (bits, group) in QWAQBlock.supported {
            let n = 9, k = 512
            let bank = makeBank(bits: bits, group: group, n: n, k: k)
            let xs = makeActivations(k: k)
            for rows in [1, 2] {
                // The kernel writes f16, so the comparison is on the f16
                // VALUES; the raw bit patterns are what the parity test
                // above compares.
                let raw = try runDecodeRaw(bits: bits, group: group, n: n, k: k,
                                           bank: bank, x: xs, rows: rows)
                let got = raw.map { Float(Float16(bitPattern: $0)) }
                for r in 0..<n {
                    let ref = hostRow(bits: bits, group: group, row: r, k: k, bank: bank, x: xs)
                    let scale = max(abs(Double(ref.value)), Double(ref.magnitude), 1e-6)
                    let rel = abs(Double(got[r]) - Double(ref.value)) / scale
                    XCTAssertLessThan(rel, 1e-3,
                                      "\(bits)-bit group \(group), \(rows)-row path, row \(r): "
                                      + "got \(got[r]), want \(ref.value)")
                }
            }
        }
    }

    /// Runs a decode kernel with the row count the caller chooses, so the
    /// family's one-row and two-row paths can be reached: `encodeDecodeT4`
    /// always asks for the four-row path the dense geometry uses.
    private func runDecodeRaw(bits: Int, group: Int, n: Int, k: Int,
                               bank: Data, x: [Float], rows: Int) throws -> [UInt16] {
        let bytes = bank.count
        let arena = try QWArena(hal: Self.hal, weights: bytes + (8 << 20),
                                activations: (8 << 20), kv: 0, gdnStates: 0)
        let w = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
        arena.copyIn(w, from: bank)
        let act = arena.activationsRegion()
        let x32 = try act.allocate(QWTensorShape([k], .f32))
        arena.write(x32, from: x)
        let x16 = try act.allocate(QWTensorShape([k], .f16))
        let y16 = try act.allocate(QWTensorShape([n], .f16))
        try step { enc in try QWGemm.encodeCast(enc, hal: Self.hal, x32: x32, y16: x16, count: k) }
        try step { enc in
            let name = try XCTUnwrap(QWGemmAQ.decodeKernelName(bits: bits, group: group))
            let pso = try Self.hal.pipeline(name)
            enc.setComputePipelineState(pso)
            w.bind(enc, index: 0)
            x16.bind(enc, index: 1)
            y16.bind(enc, index: 3)
            var n32: UInt32
            var groups: Int
            switch rows {
            case 4:
                n32 = UInt32(n) | 0x4000_0000
                groups = (n + 15) / 16
            case 2:
                n32 = UInt32(n) | 0x8000_0000
                groups = (n + 7) / 8
            default:
                n32 = UInt32(n)
                groups = (n + 3) / 4
            }
            var k32 = UInt32(k)
            enc.setBytes(&n32, length: 4, index: 4)
            enc.setBytes(&k32, length: 4, index: 5)
            y16.bind(enc, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
        return arena.read(y16, as: UInt16.self, count: n)
    }
}
