import XCTest
import Foundation
import Metal
@testable import QWGEMM
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// The size-preserving affine family's prefill twins: the incumbent's two
/// prefill GEMMs over the family's own container, against a host reference
/// the test reads for itself.
final class QWAQBatchedGemmTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    /// One shape of the prefill: `m` tokens, `n` weight rows, `k` columns.
    private struct Shape {
        let m: Int
        let n: Int
        let k: Int
    }

    /// The small shape, and the shape the engine's prefill runs: `k` a
    /// multiple of 256 and `n` a multiple of 64.
    private static let shapes = [
        Shape(m: 64, n: 64, k: 256),
        Shape(m: 128, n: 128, k: 512),
    ]

    /// The name of the prefill twin for a pair: the incumbent's two
    /// geometries, over the family's container.
    private func kernelName(bits: Int, group: Int, bm128: Bool) -> String {
        let base = "qwg_gemm_aq\(bits)_g\(group)_h16"
        return bm128 ? base + "_bm128" : base
    }

    /// A bank of the family's blocks: `n` rows of `k / group` groups. Every
    /// element gets its own code, so no two columns of a row agree — a
    /// kernel that read the wrong column would answer differently.
    private func makeBank(bits: Int, group: Int, n: Int, k: Int,
                           scale: Float = 0.03125, bias: Float = -0.5,
                           code: ((Int) -> UInt32)? = nil) -> Data {
        let groups = k / group
        let maxCode = UInt32((1 << bits) - 1)
        var seed: UInt32 = 0x9E37_79B9
        var bank = Data()
        for r in 0..<n {
            for g in 0..<groups {
                let values = (0..<group).map { i -> UInt32 in
                    if let code = code { return code(r * k + g * group + i) }
                    seed = seed &* 1664525 &+ 1013904223
                    return (seed >> 9) % (maxCode + 1)
                }
                bank.append(try! QWAQBlock.pack(bits: bits, group: group, values: values,
                                                 scale: scale, bias: bias))
            }
        }
        return bank
    }

    /// The code at `index` of a group's payload: the `bits`-wide integer at
    /// that bit of the continuous little-endian bit stream. The test reads
    /// the stream itself, so the reference owes the implementation nothing.
    private func code(at index: Int, bits: Int, payload: [UInt8]) -> UInt32 {
        var value: UInt32 = 0
        for b in 0..<bits {
            let bit = index * bits + b
            let set = UInt32((payload[bit >> 3] >> UInt32(bit & 7)) & 1)
            value |= set << UInt32(b)
        }
        return value
    }

    /// One row's weights, in the container's own arithmetic: the code's
    /// scale and the group's bias, with the group's OWN header — no fold.
    private func rowWeights(bits: Int, group: Int, bank: Data, row: Int, k: Int) -> [Float] {
        let groups = k / group
        let groupBytes = group * bits / 8 + 4
        let bytes = [UInt8](bank)
        var out = [Float](repeating: 0, count: k)
        for g in 0..<groups {
            let base = (row * groups + g) * groupBytes
            let scale = Double(Float16(bitPattern: UInt16(bytes[base]) | (UInt16(bytes[base + 1]) << 8)))
            let bias = Double(Float16(bitPattern: UInt16(bytes[base + 2]) | (UInt16(bytes[base + 3]) << 8)))
            let payload = Array(bytes[(base + 4)..<(base + groupBytes)])
            for i in 0..<group {
                let c = Double(code(at: i, bits: bits, payload: payload))
                // The kernel rounds the weight to f16 before the fp32 dot.
                out[g * group + i] = Float(Float16(c * scale + bias))
            }
        }
        return out
    }

    /// The host's answer for one row, and the natural scale of the dot.
    private func hostRow(weights: ArraySlice<Float>,
                         x: ArraySlice<Float>) -> (value: Float, magnitude: Float) {
        var acc: Float = 0
        var magnitude: Float = 0
        for (w, xv) in zip(weights, x) {
            let product = w * xv
            acc += product
            magnitude += abs(product)
        }
        return (acc, magnitude)
    }

    /// The activations, as the device sees them: `m` rows of `k` values.
    private func makeActivations(m: Int, k: Int) -> [Float] {
        var seed: UInt32 = 0x5EED_C0DE
        var out = [Float]()
        out.reserveCapacity(m * k)
        for _ in 0..<(m * k) {
            seed = seed &* 1664525 &+ 1013904223
            out.append(Float(Int32(truncatingIfNeeded: seed >> 8) % 512) / 512.0 - 0.5)
        }
        return out
    }

    private func step(_ f: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
        try f(enc)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "step failed: \(String(describing: cb.error))")
    }

    /// Runs one prefill twin over a bank and returns the f16 output.
    private func runGemm(kernel: String, bm: Int, m: Int, n: Int, k: Int,
                         bank: Data, x: [Float]) throws -> [Float] {
        let bytes = bank.count
        let arena = try QWArena(hal: Self.hal, weights: bytes + (8 << 20),
                                activations: (16 << 20), kv: 0, gdnStates: 0)
        let blocks = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
        arena.copyIn(blocks, from: bank)
        let act = arena.activationsRegion()
        let x32 = try act.allocate(QWTensorShape([m * k], .f32))
        arena.write(x32, from: x)
        let x16 = try act.allocate(QWTensorShape([m * k], .f16))
        let y16 = try act.allocate(QWTensorShape([m * n], .f16))
        try step { enc in try QWGemm.encodeCast(enc, hal: Self.hal, x32: x32, y16: x16, count: m * k) }
        try step { enc in
            let pso = try Self.hal.pipeline(kernel)
            enc.setComputePipelineState(pso)
            blocks.bind(enc, index: 0)
            x16.bind(enc, index: 1)
            y16.bind(enc, index: 2)
            y16.bind(enc, index: 3)
            var m32 = UInt32(m), n32 = UInt32(n), k32 = UInt32(k)
            enc.setBytes(&m32, length: 4, index: 4)
            enc.setBytes(&n32, length: 4, index: 5)
            enc.setBytes(&k32, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: (n + 63) / 64, height: (m + bm - 1) / bm, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
        return arena.read(y16, as: Float16.self, count: m * n).map { Float($0) }
    }

    /// Every pair of the family, over both shapes and both geometries: the
    /// prefill twin must agree with the host's own arithmetic.
    func testThePrefillTwinsMatchTheHostReference() throws {
        var worst: [String: Double] = [:]
        for (bits, group) in QWAQBlock.supported {
            for shape in Self.shapes {
                let m = shape.m, n = shape.n, k = shape.k
                let bank = makeBank(bits: bits, group: group, n: n, k: k)
                let x = makeActivations(m: m, k: k)
                let weights = (0..<n).map { rowWeights(bits: bits, group: group, bank: bank, row: $0, k: k) }
                for bm128 in [false, true] {
                    let kernel = kernelName(bits: bits, group: group, bm128: bm128)
                    let got = try runGemm(kernel: kernel, bm: bm128 ? 128 : 64,
                                         m: m, n: n, k: k, bank: bank, x: x)
                    for token in 0..<m {
                        let activations = x[(token * k)..<((token + 1) * k)].map { Float(Float16($0)) }
                        for row in 0..<n {
                            let ref = hostRow(weights: weights[row][...], x: activations[...])
                            let scale = max(abs(Double(ref.value)), Double(ref.magnitude), 1e-6)
                            let rel = abs(Double(got[token * n + row]) - Double(ref.value)) / scale
                            let key = "\(bits)-bit group \(group)"
                            worst[key] = max(worst[key] ?? 0, rel)
                            XCTAssertLessThan(rel, 1e-3,
                                              "\(key), \(kernel), token \(token), row \(row): "
                                              + "got \(got[token * n + row]), want \(ref.value)")
                        }
                    }
                }
            }
        }
        for (pair, error) in worst.sorted(by: { $0.key < $1.key }) {
            print("[QWAQBatchedGemm] \(pair): worst relative error \(error)")
        }
    }

    /// Folding -1024 * scale into an f16 bias would round away 0.25 here.
    func testAffineBiasSurvivesLargeMagicExponentFold() throws {
        let foldedBias = Float16(Float(0.25) - Float(1024))
        XCTAssertEqual(foldedBias, -1024)
        XCTAssertEqual(Float16(Float(1025) + Float(foldedBias)), 1)
        let m = 64, n = 64, k = 256
        let bank = makeBank(bits: 4, group: 64, n: n, k: k, scale: 1.0, bias: 0.25) { _ in 1 }
        var activations = [Float](repeating: 0, count: m * k)
        for row in 0..<m { activations[row * k] = 1 }
        for bm128 in [false, true] {
            let kernel = kernelName(bits: 4, group: 64, bm128: bm128)
            let got = try runGemm(kernel: kernel, bm: bm128 ? 128 : 64,
                                  m: m, n: n, k: k, bank: bank, x: activations)
            XCTAssertTrue(got.allSatisfy { $0 == 1.25 }, "\(kernel) lost the affine bias")
        }
    }

    func testNaxF32PreservesAffineBias() throws {
        let m = 64, n = 64, k = 256
        let bank = makeBank(bits: 4, group: 64, n: n, k: k, scale: 1.0, bias: 0.25) { _ in 1 }
        let arena = try QWArena(hal: Self.hal, weights: bank.count + (1 << 20),
                                activations: 8 << 20, kv: 0, gdnStates: 0)
        let blocks = try arena.weightsRegion().allocate(QWTensorShape([bank.count / 2], .f16))
        arena.copyIn(blocks, from: bank)
        let weight = try XCTUnwrap(QWWeightView(blocks: blocks, n: n, k: k, bits: 4,
                                                groupSize: 64, isRotated: false))
        let act = arena.activationsRegion()
        let x32 = try act.allocate(QWTensorShape([m * k], .f32))
        let x16 = try act.allocate(QWTensorShape([m * k], .f16))
        let out = try act.allocate(QWTensorShape([m * n], .f32))
        var activations = [Float](repeating: 0, count: m * k)
        for row in 0..<m { activations[row * k] = 1 }
        arena.write(x32, from: activations)
        try step { enc in try QWGemm.encodeCast(enc, hal: Self.hal, x32: x32, y16: x16, count: m * k) }
        try step { enc in
            try QWGemm.encodeNaxF32(enc, hal: Self.hal, weight: weight, x16: x16, y16: x16, y32: out, rows: m)
        }
        XCTAssertTrue(arena.read(out, as: Float.self, count: m * n).allSatisfy { $0 == 1.25 })
    }

    /// The fragment's COLUMN mapping, pinned. A bank whose weights are one at
    /// the second half of a chunk's columns and zero elsewhere must answer
    /// with the activations of exactly those columns, and not with another's.
    func testTheFragmentReadsTheColumnsItClaims() throws {
        let m = 64, n = 64, k = 256
        let marked: Set<Int> = [16, 17, 18, 19]
        for (bits, group) in QWAQBlock.supported {
            let bank = makeBank(bits: bits, group: group, n: n, k: k, scale: 1.0, bias: 0.0) { column in
                marked.contains(column % 32) ? 1 : 0
            }
            // Distinct, increasing activations: any other column set sums to
            // another number.
            let x = (0..<(m * k)).map { Float(($0 % k) + 1) / 64.0 }
            var want: Float = 0
            var magnitude: Float = 0
            for column in 0..<k where marked.contains(column % 32) {
                want += Float(Float16(x[column]))
                magnitude += abs(Float(Float16(x[column])))
            }
            for bm128 in [false, true] {
                let kernel = kernelName(bits: bits, group: group, bm128: bm128)
                let got = try runGemm(kernel: kernel, bm: bm128 ? 128 : 64,
                                     m: m, n: n, k: k, bank: bank, x: x)
                for token in 0..<m {
                    for row in 0..<n {
                        let value = got[token * n + row]
                        let scale = max(abs(Double(want)), Double(magnitude), 1e-6)
                        let rel = abs(Double(value) - Double(want)) / scale
                        XCTAssertLessThan(rel, 1e-3,
                                          "\(bits)-bit group \(group), \(kernel), token \(token), "
                                          + "row \(row): got \(value), want \(want) — the fragment "
                                          + "must read the columns sc.x+0..3 and sc.x+16..19")
                    }
                }
            }
        }
    }

    /// The prefill's tail: fewer tokens than a tile holds, so the kernel
    /// takes its unaligned-row path.
    func testThePrefillTwinsAgreeOnFewTokens() throws {
        let m = 3, n = 64, k = 256
        for (bits, group) in QWAQBlock.supported {
            let bank = makeBank(bits: bits, group: group, n: n, k: k)
            let x = makeActivations(m: m, k: k)
            let weights = (0..<n).map { rowWeights(bits: bits, group: group, bank: bank, row: $0, k: k) }
            for bm128 in [false, true] {
                let kernel = kernelName(bits: bits, group: group, bm128: bm128)
                let got = try runGemm(kernel: kernel, bm: bm128 ? 128 : 64,
                                     m: m, n: n, k: k, bank: bank, x: x)
                for token in 0..<m {
                    let activations = x[(token * k)..<((token + 1) * k)].map { Float(Float16($0)) }
                    for row in 0..<n {
                        let ref = hostRow(weights: weights[row][...], x: activations[...])
                        let scale = max(abs(Double(ref.value)), Double(ref.magnitude), 1e-6)
                        let rel = abs(Double(got[token * n + row]) - Double(ref.value)) / scale
                        XCTAssertLessThan(rel, 1e-3,
                                          "\(bits)-bit group \(group), \(kernel), token \(token), "
                                          + "row \(row): got \(got[token * n + row]), want \(ref.value)")
                    }
                }
            }
        }
    }
}
