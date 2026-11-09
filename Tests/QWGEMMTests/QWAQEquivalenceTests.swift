import XCTest
import Foundation
import Metal
@testable import QWGEMM
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// The plan's acceptance gate for the size-preserving affine family: the
/// checkpoint's own bytes, repacked on the GPU and read by the family's
/// decode kernel, must give what MLX's own decoder gives.
final class QWAQEquivalenceTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    /// The dtype the checkpoint wrote its scales in: 0 is bf16, 1 is f16.
    private enum ScaleDType: UInt32 { case bf16 = 0, f16 = 1 }

    private func step(_ f: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
        try f(enc)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "step failed: \(String(describing: cb.error))")
    }

    private func makeActivations(k: Int) -> [Float] {
        var seed: UInt32 = 0x5EED_C0DE
        return (0..<k).map { _ in
            seed = seed &* 1664525 &+ 1013904223
            return Float(Int32(truncatingIfNeeded: seed >> 8) % 512) / 512.0 - 0.5
        }
    }

    /// The codes of one row, and MLX's payload bytes for them: one
    /// continuous little-endian bit stream, `bits` bits per code.
    private func mlxRow(codes: [UInt32], bits: Int, k: Int) -> [UInt8] {
        let bytes = k * bits / 8
        var out = [UInt8](repeating: 0, count: bytes)
        for (i, code) in codes.enumerated() {
            let bit = i * bits
            let byte = bit >> 3
            let off = UInt32(bit & 7)
            let take = min(8 - Int(off), bits)
            let mask = UInt32((1 << take) - 1)
            out[byte] = UInt8(UInt32(out[byte]) | ((code & mask) << off))
            if take < bits {
                // The code's high bits start the NEXT byte, at its bit 0.
                out[byte + 1] = UInt8(UInt32(out[byte + 1]) | ((code >> UInt32(take)) << 0))
            }
        }
        return out
    }

    /// One checkpoint: the payload, the scales, the biases, in the dtype the
    /// checkpoint wrote them in.
    private func makeCheckpoint(bits: Int, group: Int, n: Int, k: Int,
                                dtype: ScaleDType) -> (payload: Data, scales: [UInt16], biases: [UInt16]) {
        let groups = k / group
        var seed: UInt32 = 0x1357_9BDF
        var payload = Data()
        var scales: [UInt16] = []
        var biases: [UInt16] = []
        for _ in 0..<(n * groups) {
            // A scale and a bias in the range a real quantiser writes: small,
            // and of opposite signs.
            let s = Float(Int32(truncatingIfNeeded: seed >> 8) % 2000) / 20000.0 + 0.001
            let b = -Float(Int32(truncatingIfNeeded: seed >> 20) % 2000) / 2000.0
            seed = seed &* 1664525 &+ 1013904223
            scales.append(dtype == .bf16 ? QWBF16.fromFloatRNE(s) : Float16(s).bitPattern)
            biases.append(dtype == .bf16 ? QWBF16.fromFloatRNE(b) : Float16(b).bitPattern)
        }
        for _ in 0..<n {
            let codes: [UInt32] = (0..<k).map { _ in
                seed = seed &* 1664525 &+ 1013904223
                return UInt32((seed >> 8) % UInt32(1 << bits))
            }
            payload.append(contentsOf: mlxRow(codes: codes, bits: bits, k: k))
        }
        return (payload, scales, biases)
    }

    /// The scale and the bias a checkpoint value stands for.
    private func value(_ raw: UInt16, dtype: ScaleDType) -> Float {
        dtype == .bf16 ? QWBF16.toFloat(raw) : Float(Float16(bitPattern: raw))
    }

    /// The family's bank, written by the GPU repacker, and the number of
    /// groups whose header the repacker refused.
    private func repack(bits: Int, group: Int, n: Int, k: Int,
                        checkpoint: (payload: Data, scales: [UInt16], biases: [UInt16]),
                        dtype: ScaleDType, arena: QWArena) throws -> (bank: QWTensorValue, refusals: Int32) {
        let groups = k / group
        let bankBytes = n * groups * QWAQBlock.bytes(bits: bits, group: group)
        let bank = try arena.weightsRegion().allocate(QWTensorShape([bankBytes / 2], .f16))
        let payload = try arena.weightsRegion()
            .allocate(QWTensorShape([checkpoint.payload.count / 2], .f16))
        arena.copyIn(payload, from: checkpoint.payload)
        let scales = try arena.weightsRegion()
            .allocate(QWTensorShape([checkpoint.scales.count], .f16))
        var scaleData = Data()
        for s in checkpoint.scales { scaleData.append(contentsOf: [UInt8(s & 0xFF), UInt8(s >> 8)]) }
        arena.copyIn(scales, from: scaleData)
        let biases = try arena.weightsRegion()
            .allocate(QWTensorShape([checkpoint.biases.count], .f16))
        var biasData = Data()
        for b in checkpoint.biases { biasData.append(contentsOf: [UInt8(b & 0xFF), UInt8(b >> 8)]) }
        arena.copyIn(biases, from: biasData)
        // The flag channel is the kernel's own: one i32 word, zeroed by the
        // caller, holding the number of groups the f16 header could not hold.
        let flags = try arena.weightsRegion().allocate(QWTensorShape([1], .i32))
        arena.write(flags, from: [Int32(0)])

        try step { enc in
            let pso = try Self.hal.pipeline("qwg_repack_mlx_affine_to_aq_m5")
            enc.setComputePipelineState(pso)
            payload.bind(enc, index: 0)
            scales.bind(enc, index: 1)
            biases.bind(enc, index: 2)
            bank.bind(enc, index: 3)
            var rows = UInt32(n), cols = UInt32(k), b = UInt32(bits), g = UInt32(group)
            var dt = dtype.rawValue
            enc.setBytes(&rows, length: 4, index: 4)
            enc.setBytes(&cols, length: 4, index: 5)
            enc.setBytes(&b, length: 4, index: 6)
            enc.setBytes(&g, length: 4, index: 7)
            enc.setBytes(&dt, length: 4, index: 8)
            flags.bind(enc, index: 9)
            let total = n * groups
            let tpg = 64
            enc.dispatchThreadgroups(MTLSize(width: (total + tpg - 1) / tpg, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
        }
        let refusals = arena.read(flags, as: Int32.self, count: 1)[0]
        return (bank, refusals)
    }

    /// The gate: MLX's own decoder, on the same bytes, against the GPU.
    func testTheRepackedBankAgreesWithMLXsOwnDecoder() throws {
        let n = 8, k = 512
        for (bits, group) in QWAQBlock.supported {
            for dtype in [ScaleDType.bf16, .f16] {
                let checkpoint = makeCheckpoint(bits: bits, group: group, n: n, k: k, dtype: dtype)
                let arena = try QWArena(hal: Self.hal, weights: (8 << 20),
                                        activations: (8 << 20), kv: 0, gdnStates: 0)
                let (bank, refusals) = try repack(bits: bits, group: group, n: n, k: k,
                                                  checkpoint: checkpoint, dtype: dtype, arena: arena)
                XCTAssertEqual(refusals, 0,
                               "a scale or a bias the f16 header cannot hold exactly must be refused")

                let act = arena.activationsRegion()
                let x32 = try act.allocate(QWTensorShape([k], .f32))
                let xs = makeActivations(k: k)
                arena.write(x32, from: xs)
                let x16 = try act.allocate(QWTensorShape([k], .f16))
                let y16 = try act.allocate(QWTensorShape([n], .f16))
                try step { enc in try QWGemm.encodeCast(enc, hal: Self.hal, x32: x32, y16: x16, count: k) }
                try step { enc in
                    try QWGemmAQ.encodeDecodeT4(enc, hal: Self.hal, blocks: bank, x: x16, y16: y16,
                                                n: n, k: k, bits: bits, group: group)
                }
                let got = arena.read(y16, as: Float16.self, count: n).map { Float($0) }

                // MLX's own answer: its decoder's codes, its scales and
                // biases, over the activations the device saw.
                let groups = k / group
                let payload = [UInt8](checkpoint.payload)
                var worst = 0.0 as Double
                var worstBeyondOneUlp = 0.0 as Double
                for r in 0..<n {
                    let rowWords = payload[(r * k * bits / 8)...]
                    var ref = 0.0 as Double
                    var magnitude = 0.0 as Double
                    rowWords.withUnsafeBytes { raw in
                        let words = raw.bindMemory(to: UInt32.self)
                        for c in 0..<k {
                            let code = Double(QWBF16.unpackMLXAffineQ(rowWords: words.baseAddress!,
                                                                    col: c, bits: bits))
                            let s = Double(value(checkpoint.scales[r * groups + c / group], dtype: dtype))
                            let b = Double(value(checkpoint.biases[r * groups + c / group], dtype: dtype))
                            let w = code * s + b
                            ref += w * Double(Float16(xs[c]))
                            magnitude += abs(w * Double(Float16(xs[c])))
                        }
                    }
                    // The engine's output is f16, so the gate is the output
                    // format's own resolution: the device's f16 answer must
                    // be the reference's f16 answer, to within one ulp.
                    let ref16 = Double(Float16(ref))
                    let ulp = abs(ref16) * 0.0009765625   // f16 keeps 11 significand bits
                    worstBeyondOneUlp = max(worstBeyondOneUlp, abs(Double(got[r]) - ref16) - ulp)
                    let scale = max(abs(ref), magnitude, 1e-6)
                    worst = max(worst, abs(Double(got[r]) - ref) / scale)
                }
                XCTAssertLessThanOrEqual(worstBeyondOneUlp, 1e-6,
                                        "\(bits)-bit group \(group), \(dtype): the device's answer must "
                                        + "be the reference's at f16 resolution")
                XCTAssertLessThan(worst, 1e-3,
                                  "\(bits)-bit group \(group), \(dtype): the repacked bank must agree "
                                  + "with MLX's decoder to 1e-3, measured \(worst)")
            }
        }
    }

    /// The refusal branch itself: a header f16 cannot hold EXACTLY must be
    /// counted, and the offending block must be POISONED — not left holding
    /// the silently-rounded header, which would decode to a different weight.
    func testTheRepackerRefusesAndPoisonsAHeaderF16CannotHold() throws {
        let bits = 3, group = 64, n = 2, k = 128
        let checkpoint = makeCheckpoint(bits: bits, group: group, n: n, k: k, dtype: .bf16)
        // 2^16 is exact in bf16 and NOT in f16: f16's largest is 65504, so
        // the header would become an infinity.
        var scales = checkpoint.scales
        scales[0] = QWBF16.fromFloatRNE(65536.0)
        let lossy = (payload: checkpoint.payload, scales: scales, biases: checkpoint.biases)
        let arena = try QWArena(hal: Self.hal, weights: (8 << 20),
                                activations: (8 << 20), kv: 0, gdnStates: 0)
        let (bank, refusals) = try repack(bits: bits, group: group, n: n, k: k,
                                          checkpoint: lossy, dtype: .bf16, arena: arena)
        XCTAssertEqual(refusals, 1,
                       "exactly the one group whose header f16 cannot hold must be refused")
        // The refused group's header is NaN, so a caller that ignores the
        // count cannot read a rounded weight.
        let groups = k / group
        let bytesPerGroup = QWAQBlock.bytes(bits: bits, group: group)
        let raw = arena.read(bank, as: UInt16.self, count: n * groups * bytesPerGroup / 2)
        XCTAssertTrue(Float16(bitPattern: raw[0]).isNaN,
                      "the refused group's header must be poisoned, not rounded")
    }
}
