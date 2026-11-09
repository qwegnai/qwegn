import XCTest
import Foundation
import Metal
@testable import QWGEMM
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// The MLP gate/up twin of the size-preserving affine family: the
/// incumbent's `qwg_gemv_q4g_silu_pair_dx_h16` geometry over the family's
/// blocks, against the host reference.
final class QWAQMLPTwinTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    /// The row-count flag the kernel reads its mode from.
    private enum Mode { case block, pair, plain }

    private func makeBank(bits: Int, group: Int, rows: Int, k: Int,
                           scale: Float = 0.00390625, bias: Float = -0.0078125) -> Data {
        var bank = Data()
        let maxValue = UInt32((1 << bits) - 1)
        for r in 0..<rows {
            for g in 0..<(k / group) {
                let values = (0..<group).map { i in
                    UInt32((i + r * 7 + g * 13) % Int(maxValue + 1))
                }
                bank.append(try! QWAQBlock.pack(bits: bits, group: group, values: values,
                                                scale: scale, bias: bias))
            }
        }
        return bank
    }

    private func makeActivations(k: Int) -> [Float] {
        var seed: UInt32 = 0x5EED_C0DE
        return (0..<k).map { _ in
            seed = seed &* 1664525 &+ 1013904223
            return Float(Int32(truncatingIfNeeded: seed >> 8) % 512) / 2048.0 - 0.25
        }
    }

    private func step(_ f: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
        try f(enc)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "step failed: \(String(describing: cb.error))")
    }

    /// One row's dot, over the activations the device saw.
    private func hostRow(bits: Int, group: Int, row: Int, k: Int,
                         bank: Data, x: [Float]) -> Float {
        let groups = k / group
        let bytesPerGroup = QWAQBlock.bytes(bits: bits, group: group)
        var acc = 0.0 as Float
        for g in 0..<groups {
            let offset = (row * groups + g) * bytesPerGroup
            let slice = bank.subdata(in: offset..<(offset + bytesPerGroup))
            let xs = (0..<group).map { Float(Float16(x[g * group + $0])) }
            acc += slice.withUnsafeBytes { QWAQBlock.dot(group: $0, bits: bits, group: group, x: xs) }
        }
        return acc
    }

    /// Runs the twin in one of the three modes the row-count flags select.
    private func runMLP(bits: Int, group: Int, intermediate: Int, k: Int,
                        bank: Data, x: [Float], mode: Mode) throws -> [Float] {
        let bytes = bank.count
        let arena = try QWArena(hal: Self.hal, weights: bytes + (8 << 20),
                                activations: (8 << 20), kv: 0, gdnStates: 0)
        let w = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
        arena.copyIn(w, from: bank)
        let act = arena.activationsRegion()
        let x32 = try act.allocate(QWTensorShape([k], .f32))
        arena.write(x32, from: x)
        let x16 = try act.allocate(QWTensorShape([k], .f16))
        let h16 = try act.allocate(QWTensorShape([intermediate], .f16))
        try step { enc in try QWGemm.encodeCast(enc, hal: Self.hal, x32: x32, y16: x16, count: k) }

        let rows = intermediate * 2
        let view = try XCTUnwrap(QWWeightView(blocks: w, n: rows, k: k,
                                               bits: bits, groupSize: group, isRotated: false))
        try step { enc in
            let name = try XCTUnwrap(QWGemmAQ.siluPairKernelName(bits: bits, group: group))
            let pso = try Self.hal.pipeline(name)
            enc.setComputePipelineState(pso)
            view.blocks.bind(enc, index: 0)
            x16.bind(enc, index: 1)
            h16.bind(enc, index: 8)
            var n32 = UInt32(rows)
            var width = 0
            switch mode {
            case .block:
                n32 |= 0x4000_0000
                width = intermediate / 32
            case .pair:
                n32 |= 0x8000_0000
                width = (intermediate + 7) / 8
            case .plain:
                width = (intermediate + 3) / 4
            }
            var k32 = UInt32(k)
            enc.setBytes(&n32, length: 4, index: 4)
            enc.setBytes(&k32, length: 4, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: width, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        return arena.read(h16, as: Float16.self, count: intermediate).map { Float($0) }
    }

    /// The twin's silu·up epilogue, against the host's.
    func testTheMLPTwinMatchesTheHostReference() throws {
        let intermediate = 64, k = 256
        for (bits, group) in QWAQBlock.supported {
            let bank = makeBank(bits: bits, group: group, rows: intermediate * 2, k: k)
            let xs = makeActivations(k: k)
            for mode in [Mode.block, .pair, .plain] {
                let got = try runMLP(bits: bits, group: group, intermediate: intermediate,
                                     k: k, bank: bank, x: xs, mode: mode)
                for p in 0..<intermediate {
                    let gate = hostRow(bits: bits, group: group, row: p, k: k, bank: bank, x: xs)
                    let up = hostRow(bits: bits, group: group, row: intermediate + p,
                                     k: k, bank: bank, x: xs)
                    let ref = (gate / (1.0 as Float + exp(-gate))) * up
                    let magnitude = abs(gate) * abs(up)
                    let ref16 = Float(Float16(ref))
                    let tol = abs(ref16) * 0.0009765625 + magnitude * 1e-4 + 1e-6
                    XCTAssertLessThanOrEqual(abs(got[p] - ref16), tol,
                                            "\(bits)-bit group \(group), \(mode), row \(p): "
                                            + "got \(got[p]), want \(ref)")
                }
            }
        }
    }

    /// The block mode shares one activation load across eight weight rows,
    /// so it must give the same answers as the pair mode — and the plain
    /// mode, which the driver never takes, must agree with both.
    func testTheMLPTwinsModesAgreeWithEachOther() throws {
        let intermediate = 64, k = 256
        for (bits, group) in QWAQBlock.supported {
            let bank = makeBank(bits: bits, group: group, rows: intermediate * 2, k: k)
            let xs = makeActivations(k: k)
            let block = try runMLP(bits: bits, group: group, intermediate: intermediate,
                                   k: k, bank: bank, x: xs, mode: .block)
            let pair = try runMLP(bits: bits, group: group, intermediate: intermediate,
                                  k: k, bank: bank, x: xs, mode: .pair)
            let plain = try runMLP(bits: bits, group: group, intermediate: intermediate,
                                   k: k, bank: bank, x: xs, mode: .plain)
            for p in 0..<intermediate {
                let scale = max(abs(Double(block[p])), 1e-6)
                XCTAssertLessThan(abs(Double(pair[p]) - Double(block[p])) / scale, 1e-3,
                                  "\(bits)-bit group \(group), row \(p): the pair mode and the "
                                  + "block mode must agree")
                XCTAssertLessThan(abs(Double(plain[p]) - Double(block[p])) / scale, 1e-3,
                                  "\(bits)-bit group \(group), row \(p): the plain mode and the "
                                  + "block mode must agree")
            }
        }
    }
}
