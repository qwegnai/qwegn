import XCTest
import Foundation
import Metal
@testable import QWGEMM
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// The head twin of the size-preserving affine family: the incumbent's
/// `qwg_head_gemv_q4g_argmax` epilogue over the family's blocks, against
/// the host reference.
final class QWAQHeadTwinTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

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
        var seed: UInt32 = 0xC0FF_EE01
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

    /// Runs the head twin and the final key reduction, and returns the
    /// logits and the token.
    private func runHead(bits: Int, group: Int, rows: Int, k: Int,
                         bank: Data, x: [Float], writeLogits: Bool) throws -> (logits: [Float], token: Int32) {
        let bytes = bank.count
        let slots = (rows + 31) / 32
        let arena = try QWArena(hal: Self.hal, weights: bytes + (8 << 20),
                                activations: (8 << 20), kv: 0, gdnStates: 0)
        let w = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
        arena.copyIn(w, from: bank)
        let act = arena.activationsRegion()
        let x32 = try act.allocate(QWTensorShape([k], .f32))
        arena.write(x32, from: x)
        let x16 = try act.allocate(QWTensorShape([k], .f16))
        let y = try act.allocate(QWTensorShape([rows], .f32))
        let keysHi = try act.allocate(QWTensorShape([slots], .i32))
        let keysLo = try act.allocate(QWTensorShape([slots], .i32))
        let token = try act.allocate(QWTensorShape([1], .i32))
        try step { enc in try QWGemm.encodeCast(enc, hal: Self.hal, x32: x32, y16: x16, count: k) }

        let view = try XCTUnwrap(QWWeightView(blocks: w, n: rows, k: k,
                                               bits: bits, groupSize: group, isRotated: false))
        try step { enc in
            try QWGemmAQ.encodeHeadArgmax(enc, hal: Self.hal, view: view, x: x16, y: y,
                                          keysHi: keysHi, keysLo: keysLo,
                                          writeLogits: writeLogits)
        }
        try step { enc in
            let pso = try Self.hal.pipeline("qwg_head_argmax_fin")
            enc.setComputePipelineState(pso)
            keysHi.bind(enc, index: 0)
            keysLo.bind(enc, index: 1)
            token.bind(enc, index: 2)
            var nTG = UInt32(slots)
            enc.setBytes(&nTG, length: 4, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        }
        return (arena.read(y, as: Float.self, count: rows), arena.read(token, as: Int32.self, count: 1)[0])
    }

    /// The twin's logits, row by row, against the host's dots.
    func testTheHeadTwinMatchesTheHostReference() throws {
        let rows = 96, k = 256
        for (bits, group) in QWAQBlock.supported {
            let bank = makeBank(bits: bits, group: group, rows: rows, k: k)
            let xs = makeActivations(k: k)
            let got = try runHead(bits: bits, group: group, rows: rows, k: k,
                                  bank: bank, x: xs, writeLogits: true)
            for r in 0..<rows {
                let ref = hostRow(bits: bits, group: group, row: r, k: k, bank: bank, x: xs)
                let scale = max(abs(Double(ref.value)), Double(ref.magnitude), 1e-6)
                let rel = abs(Double(got.logits[r]) - Double(ref.value)) / scale
                XCTAssertLessThan(rel, 1e-4,
                                  "\(bits)-bit group \(group), row \(r): got \(got.logits[r]), "
                                  + "want \(ref.value)")
            }
        }
    }

    /// The head's own law: the highest logit wins, and a tie goes to the
    /// LOWEST id. Rows 0 and 1 carry the largest codes, so they are the
    /// maximum and they are equal to each other — the tiebreak is real.
    func testTheHeadTwinPicksTheHighestLogitAndBreaksTiesLow() throws {
        let rows = 96, k = 256
        for (bits, group) in QWAQBlock.supported {
            let bank = makeTiedBank(bits: bits, group: group, rows: rows, k: k)
            let xs = makePositiveActivations(k: k)
            let got = try runHead(bits: bits, group: group, rows: rows, k: k,
                                  bank: bank, x: xs, writeLogits: false)
            var best = 0 as Float, bestRow = 0
            for r in 0..<rows {
                let ref = hostRow(bits: bits, group: group, row: r, k: k, bank: bank, x: xs)
                if ref.value > best {
                    best = ref.value
                    bestRow = r
                }
            }
            XCTAssertEqual(bestRow, 0,
                           "\(bits)-bit group \(group): the tied pair must be the maximum")
            XCTAssertEqual(got.token, Int32(0),
                           "\(bits)-bit group \(group): a tie must go to the LOWEST id")
        }
    }

    /// A bank whose first two rows carry the largest codes — equal to each
    /// other — and whose other rows carry the smallest.
    private func makeTiedBank(bits: Int, group: Int, rows: Int, k: Int) -> Data {
        let maxValue = UInt32((1 << bits) - 1)
        var bank = Data()
        for r in 0..<rows {
            let code = r < 2 ? maxValue : 0
            for _ in 0..<(k / group) {
                let values = [UInt32](repeating: code, count: group)
                bank.append(try! QWAQBlock.pack(bits: bits, group: group, values: values,
                                                scale: 0.00390625, bias: -0.0078125))
            }
        }
        return bank
    }

    /// Activations that are all positive, so the largest codes are the
    /// largest logit.
    private func makePositiveActivations(k: Int) -> [Float] {
        var seed: UInt32 = 0xC0FF_EE01
        return (0..<k).map { _ in
            seed = seed &* 1664525 &+ 1013904223
            return 0.05 + Float(Int32(truncatingIfNeeded: seed >> 8) % 512) / 2560.0
        }
    }
}
