import XCTest
import Foundation
import Metal
@testable import QWGEMM
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// The GDN in-projection twins of the size-preserving affine family, over
/// the family's blocks: the decode twin and the prefill twin, each checked
/// against a host reference that reproduces its OWN arithmetic.
final class QWAQGDNTwinTests: XCTestCase {
    override func setUpWithError() throws {
        try XCTSkipUnless(QWDeviceCaps.current?.isMetal4TensorCapable == true, "AQ GDN prefill twin requires Metal 4 tensor hardware")
    }

    /// The bank's header: f16-exact (the format refuses a header it cannot
    /// hold) but NOT a power of two, so the dequantised weights are generally
    /// not f16-exact — a power-of-two scale would hide the narrowing.
    private static let scale: Float = 1025.0 / 4096.0
    private static let bias: Float = -684.0 / 4096.0

    /// A small hybrid geometry: one GDN layer and one attention layer, with
    /// the GDN's channel counts small enough to host-check.
    private static func hybrid() throws -> QWModelGeometry {
        let text: [String: Any] = [
            "model_type": "qwen3_5_text",
            "hidden_size": 256,
            "vocab_size": 64,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "full_attention_interval": 2,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "intermediate_size": 512,
            "partial_rotary_factor": 1.0,
            "rope_theta": 1e7,
            "eos_token_id": 2,
            "linear_key_head_dim": 32,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
        ]
        let data = try JSONSerialization.data(withJSONObject: ["text_config": text])
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    private func makeBank(bits: Int, group: Int, rows: Int, k: Int) -> Data {
        var bank = Data()
        let maxValue = UInt32((1 << bits) - 1)
        for r in 0..<rows {
            for g in 0..<(k / group) {
                let values = (0..<group).map { i in
                    UInt32((i + r * 7 + g * 13) % Int(maxValue + 1))
                }
                bank.append(try! QWAQBlock.pack(bits: bits, group: group, values: values,
                                                scale: Self.scale, bias: Self.bias))
            }
        }
        return bank
    }

    private func makeActivations(rows: Int, k: Int) -> [Float] {
        var seed: UInt32 = 0x5EED_C0DE
        return (0..<(rows * k)).map { _ in
            seed = seed &* 1664525 &+ 1013904223
            return Float(Int32(truncatingIfNeeded: seed >> 8) % 512) / 2048.0 - 0.25
        }
    }

    private func step(_ hal: QWHal,
                      _ f: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
        try f(enc)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "step failed: \(String(describing: cb.error))")
    }

    /// One row's dot in f32 — the DECODE twin's arithmetic: the weight built
    /// from its code, the activation in f16, the sum in f32. Also the magnitude
    /// of the terms, the honest denominator for a relative gate.
    private func hostRow(bits: Int, group: Int, row: Int, k: Int,
                         bank: Data, x: [Float]) -> (value: Float, magnitude: Float) {
        let groups = k / group
        let bytesPerGroup = QWAQBlock.bytes(bits: bits, group: group)
        var value: Float = 0
        var magnitude: Float = 0
        for g in 0..<groups {
            let offset = (row * groups + g) * bytesPerGroup
            let slice = bank.subdata(in: offset..<(offset + bytesPerGroup))
            let xs = (0..<group).map { Float(Float16(x[g * group + $0])) }
            value += slice.withUnsafeBytes { QWAQBlock.dot(group: $0, bits: bits, group: group, x: xs) }
            magnitude += slice.withUnsafeBytes { raw -> Float in
                let (scale, bias) = QWAQBlock.header(raw)
                let payload = UnsafeRawBufferPointer(rebasing: raw.dropFirst(4))
                var m: Float = 0
                for i in 0..<group {
                    let code = Float(QWAQBlock.readBits(payload, bits: bits, index: i))
                    m += abs((code * scale + bias) * xs[i])
                }
                return m
            }
        }
        return (value, magnitude)
    }

    /// The same row the way the PREFILL computes it: the weight narrowed to
    /// f16 for the MMA, the activation in f16, the products and the sum in
    /// f32. The narrowing is the ONLY difference between this and `hostRow`.
    private func hostRowNarrowed(bits: Int, group: Int, row: Int, k: Int,
                                  bank: Data, x: [Float]) -> Float {
        let groups = k / group
        let bytesPerGroup = QWAQBlock.bytes(bits: bits, group: group)
        var acc: Float = 0
        for g in 0..<groups {
            let offset = (row * groups + g) * bytesPerGroup
            let slice = bank.subdata(in: offset..<(offset + bytesPerGroup))
            slice.withUnsafeBytes { raw in
                let (scale, bias) = QWAQBlock.header(raw)
                let payload = UnsafeRawBufferPointer(rebasing: raw.dropFirst(4))
                for i in 0..<group {
                    let code = Float(QWAQBlock.readBits(payload, bits: bits, index: i))
                    let w = Float(Float16(code * scale + bias))
                    acc += w * Float(Float16(x[g * group + i]))
                }
            }
        }
        return acc
    }

    /// The family's gate: the relative error, normalised by the result's own
    /// magnitude OR the magnitude of its terms, whichever is larger — a row's
    /// result may be small through cancellation while its terms are not.
    private func relativeError(_ got: Float, _ want: Float, magnitude: Float) -> Double {
        let scale = max(abs(Double(want)), Double(magnitude), 1e-6)
        return abs(Double(got) - Double(want)) / scale
    }

    /// One GDN bank, its activations and the arena that holds them.
    private struct Fixture {
        let hal: QWHal
        let arena: QWArena
        let bank: Data
        let x: [Float]
        let bits: Int
        let group: Int
        let n: Int
        let k: Int
        let aRow0: Int
        let bRow0: Int
    }

    private func makeFixture(geometry: QWModelGeometry, bits: Int, group: Int,
                             tokens: Int) throws -> Fixture {
        let hal = try QWHal(geometry: geometry)
        let k = geometry.hidden
        let n = geometry.gdnTotalN
        let bank = makeBank(bits: bits, group: group, rows: n, k: k)
        let x = makeActivations(rows: tokens, k: k)
        let arena = try QWArena(hal: hal, weights: bank.count + (8 << 20),
                                activations: (16 << 20), kv: 0, gdnStates: 0)
        return Fixture(hal: hal, arena: arena, bank: bank, x: x, bits: bits, group: group,
                       n: n, k: k,
                       aRow0: geometry.gdnQKVChannels + geometry.gdnZChannels,
                       bRow0: geometry.gdnQKVChannels + geometry.gdnZChannels + geometry.gdnVHeads)
    }

    /// The decode twin's output: the rows, and the GDN's two f32 companions.
    private struct DecodeResult {
        var y16: [Float16]
        var y32: [Float]
        var betas: [Float]
        var glog: [Float]
    }

    private func runDecode(_ f: Fixture, aLog: [Float], dtBias: [Float]) throws -> DecodeResult {
        let arena = f.arena
        let act = arena.activationsRegion()
        let x32 = try act.allocate(QWTensorShape([f.k], .f32))
        arena.write(x32, from: f.x)
        let x16 = try act.allocate(QWTensorShape([f.k], .f16))
        let y16 = try act.allocate(QWTensorShape([f.n], .f16))
        let y32 = try act.allocate(QWTensorShape([f.n], .f32))
        let alog = try act.allocate(QWTensorShape([aLog.count], .f32))
        arena.write(alog, from: aLog)
        let dtb = try act.allocate(QWTensorShape([dtBias.count], .f32))
        arena.write(dtb, from: dtBias)
        let betas = try act.allocate(QWTensorShape([aLog.count], .f32))
        let glog = try act.allocate(QWTensorShape([aLog.count], .f32))
        try step(f.hal) { enc in try QWGemm.encodeCast(enc, hal: f.hal, x32: x32, y16: x16, count: f.k) }

        let w = try arena.weightsRegion().allocate(QWTensorShape([f.bank.count / 2], .f16))
        arena.copyIn(w, from: f.bank)
        let view = try XCTUnwrap(QWWeightView(blocks: w, n: f.n, k: f.k,
                                              bits: f.bits, groupSize: f.group, isRotated: false))
        try step(f.hal) { enc in
            let name = try XCTUnwrap(QWGemmAQ.gdnInProjDecodeKernelName(bits: f.bits, group: f.group))
            enc.setComputePipelineState(try f.hal.pipeline(name))
            view.blocks.bind(enc, index: 0)
            x16.bind(enc, index: 1)
            y32.bind(enc, index: 3)
            enc.setUInt32s(f.n | 0x4000_0000, f.k, startIndex: 4)
            y16.bind(enc, index: 8)
            alog.bind(enc, index: 9)
            dtb.bind(enc, index: 10)
            betas.bind(enc, index: 11)
            glog.bind(enc, index: 12)
            enc.dispatchThreadgroups(MTLSize(width: (f.n + 15) / 16, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
        return DecodeResult(y16: arena.read(y16, as: Float16.self, count: f.n),
                            y32: arena.read(y32, as: Float.self, count: f.n),
                            betas: arena.read(betas, as: Float.self, count: aLog.count),
                            glog: arena.read(glog, as: Float.self, count: aLog.count))
    }

    /// The prefill twin's output: the QKV rows, the Z rows and the f32
    /// companions of the A and B rows.
    private struct PrefillResult {
        var qkv16: [Float16]
        var z16: [Float16]
        var y32: [Float]
    }

    private func runPrefill(_ f: Fixture, tokens: Int,
                            qkv: Int, zCh: Int) throws -> PrefillResult {
        let arena = f.arena
        let act = arena.activationsRegion()
        let x32 = try act.allocate(QWTensorShape([tokens * f.k], .f32))
        arena.write(x32, from: f.x)
        let x16 = try act.allocate(QWTensorShape([tokens * f.k], .f16))
        let qkv16 = try act.allocate(QWTensorShape([tokens * qkv], .f16))
        let z16 = try act.allocate(QWTensorShape([tokens * zCh], .f16))
        let y32 = try act.allocate(QWTensorShape([tokens * f.n], .f32))
        try step(f.hal) { enc in
            try QWGemm.encodeCast(enc, hal: f.hal, x32: x32, y16: x16, count: tokens * f.k)
        }

        let w = try arena.weightsRegion().allocate(QWTensorShape([f.bank.count / 2], .f16))
        arena.copyIn(w, from: f.bank)
        let view = try XCTUnwrap(QWWeightView(blocks: w, n: f.n, k: f.k,
                                              bits: f.bits, groupSize: f.group, isRotated: false))
        try step(f.hal) { enc in
            let name = try XCTUnwrap(QWGemmAQ.gdnInProjPrefillKernelName(bits: f.bits, group: f.group))
            enc.setComputePipelineState(try f.hal.pipeline(name))
            enc.bindTensors(view.blocks, x16, qkv16, y32, startIndex: 0)
            enc.setUInt32s(tokens, f.n, f.k, startIndex: 4)
            z16.bind(enc, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: (f.n + 63) / 64, height: (tokens + 63) / 64, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
        return PrefillResult(qkv16: arena.read(qkv16, as: Float16.self, count: tokens * qkv),
                             z16: arena.read(z16, as: Float16.self, count: tokens * zCh),
                             y32: arena.read(y32, as: Float.self, count: tokens * f.n))
    }

    /// The decode twin: the GDN's in-projection over one row of activations.
    func testTheGDNDecodeTwinMatchesTheHostReference() throws {
        let geometry = try Self.hybrid()
        for (bits, group) in QWAQBlock.supported {
            let f = try makeFixture(geometry: geometry, bits: bits, group: group, tokens: 1)
            let aLog = (0..<geometry.gdnVHeads).map { Float($0) * 0.01 - 0.5 }
            let dtBias = (0..<geometry.gdnVHeads).map { Float($0) * 0.02 - 0.3 }
            let got = try runDecode(f, aLog: aLog, dtBias: dtBias)
            XCTAssertEqual(f.n, f.bRow0 + geometry.gdnVHeads)
            for row in 0..<f.n {
                let ref = hostRow(bits: bits, group: group, row: row, k: f.k, bank: f.bank, x: f.x)
                let err = relativeError(Float(got.y16[row]), ref.value, magnitude: ref.magnitude)
                XCTAssertLessThan(err, 1e-3,
                                  "\(bits)-bit group \(group), row \(row): got \(got.y16[row]), want \(ref.value)")
                if row >= f.aRow0 {
                    let err32 = relativeError(got.y32[row], ref.value, magnitude: ref.magnitude)
                    XCTAssertLessThan(err32, 1e-3,
                                      "\(bits)-bit group \(group), f32 row \(row)")
                }
            }
            // The A rows: the gate, and the f32 companion the GDN reads.
            for h in 0..<geometry.gdnVHeads {
                let ref = hostRow(bits: bits, group: group, row: f.aRow0 + h, k: f.k, bank: f.bank, x: f.x)
                let sp = log(1 + exp(ref.value + dtBias[h]))
                XCTAssertEqual(Double(got.glog[h]), Double(-exp(aLog[h]) * sp), accuracy: 1e-5)
            }
            // The B rows: the beta.
            for h in 0..<geometry.gdnVHeads {
                let ref = hostRow(bits: bits, group: group, row: f.bRow0 + h, k: f.k, bank: f.bank, x: f.x)
                XCTAssertEqual(Double(got.betas[h]), Double(1 / (1 + exp(-ref.value))), accuracy: 1e-5)
            }
        }
    }

    /// The prefill twin: the same in-projection over several tokens at once.
    func testTheGDNPrefillTwinMatchesTheHostReference() throws {
        let geometry = try Self.hybrid()
        let tokens = 4
        for (bits, group) in QWAQBlock.supported {
            let f = try makeFixture(geometry: geometry, bits: bits, group: group, tokens: tokens)
            let got = try runPrefill(f, tokens: tokens,
                                     qkv: geometry.gdnQKVChannels, zCh: geometry.gdnZChannels)
            for t in 0..<tokens {
                let row = Array(f.x[(t * f.k)..<((t + 1) * f.k)])
                for c in 0..<geometry.gdnQKVChannels {
                    let ref = hostRow(bits: bits, group: group, row: c, k: f.k, bank: f.bank, x: row)
                    let want = hostRowNarrowed(bits: bits, group: group, row: c, k: f.k, bank: f.bank, x: row)
                    let err = relativeError(Float(got.qkv16[t * geometry.gdnQKVChannels + c]),
                                            want, magnitude: ref.magnitude)
                    XCTAssertLessThan(err, 1e-3,
                                      "\(bits)-bit group \(group), token \(t), qkv column \(c)")
                }
                for c in 0..<geometry.gdnZChannels {
                    let ref = hostRow(bits: bits, group: group, row: geometry.gdnQKVChannels + c,
                                      k: f.k, bank: f.bank, x: row)
                    let want = hostRowNarrowed(bits: bits, group: group, row: geometry.gdnQKVChannels + c,
                                              k: f.k, bank: f.bank, x: row)
                    let err = relativeError(Float(got.z16[t * geometry.gdnZChannels + c]),
                                            want, magnitude: ref.magnitude)
                    XCTAssertLessThan(err, 1e-3,
                                      "\(bits)-bit group \(group), token \(t), z column \(c)")
                }
                for c in (geometry.gdnQKVChannels + geometry.gdnZChannels)..<f.n {
                    let ref = hostRow(bits: bits, group: group, row: c, k: f.k, bank: f.bank, x: row)
                    let want = hostRowNarrowed(bits: bits, group: group, row: c, k: f.k, bank: f.bank, x: row)
                    let err = relativeError(got.y32[t * f.n + c], want, magnitude: ref.magnitude)
                    XCTAssertLessThan(err, 1e-3,
                                      "\(bits)-bit group \(group), token \(t), ab column \(c)")
                }
            }
        }
    }

    /// The two paths must agree, and the agreement must be MEASURED rather
    /// than assumed: the bank's weights are not f16-exact, so the prefill's
    /// narrowing is a real difference, and it must stay inside its bound.
    func testTheDecodeAndPrefillTwinsAgreeWithinTheNarrowing() throws {
        let geometry = try Self.hybrid()
        let qkv = geometry.gdnQKVChannels
        let zCh = geometry.gdnZChannels
        for (bits, group) in QWAQBlock.supported {
            let f = try makeFixture(geometry: geometry, bits: bits, group: group, tokens: 1)
            let aLog = (0..<geometry.gdnVHeads).map { Float($0) * 0.01 - 0.5 }
            let dtBias = (0..<geometry.gdnVHeads).map { Float($0) * 0.02 - 0.3 }
            let decode = try runDecode(f, aLog: aLog, dtBias: dtBias)
            let prefill = try runPrefill(f, tokens: 1, qkv: qkv, zCh: zCh)
            let hasNarrowedWeight = (0..<min(group, 1 << bits)).contains { code in
                let weight = Float(code) * Self.scale + Self.bias
                return Float(Float16(weight)) != weight
            }
            XCTAssertTrue(hasNarrowedWeight, "\(bits)-bit group \(group): fixture has no narrowed weight")
            var narrowingIsActive = false
            for c in 0..<qkv {
                let ref = hostRow(bits: bits, group: group, row: c, k: f.k, bank: f.bank, x: f.x)
                let narrow = hostRowNarrowed(bits: bits, group: group, row: c, k: f.k, bank: f.bank, x: f.x)
                narrowingIsActive = narrowingIsActive || narrow != ref.value
                let diff = abs(Double(Float(decode.y16[c])) - Double(Float(prefill.qkv16[c])))
                XCTAssertLessThanOrEqual(diff, bound(value: ref.value, magnitude: ref.magnitude),
                                         "\(bits)-bit group \(group), qkv column \(c): the paths differ by \(diff)")
            }
            for c in 0..<zCh {
                let ref = hostRow(bits: bits, group: group, row: qkv + c, k: f.k, bank: f.bank, x: f.x)
                let narrow = hostRowNarrowed(bits: bits, group: group, row: qkv + c, k: f.k, bank: f.bank, x: f.x)
                narrowingIsActive = narrowingIsActive || narrow != ref.value
                let diff = abs(Double(Float(decode.y16[qkv + c])) - Double(Float(prefill.z16[c])))
                XCTAssertLessThanOrEqual(diff, bound(value: ref.value, magnitude: ref.magnitude),
                                         "\(bits)-bit group \(group), z column \(c): the paths differ by \(diff)")
            }
            // The narrowing is ACTIVE: the two paths are not the same number.
            // If the bank's weights were f16-exact this would be zero and the
            // comparison above would be vacuous.
            XCTAssertTrue(narrowingIsActive,
                          "\(bits)-bit group \(group): the bank's weights are f16-exact, "
                          + "so the test cannot tell the two paths apart")
        }
    }

    /// How far the two paths may differ: one f16 ulp of the result, plus the
    /// narrowing's own bound — each weight's relative error is at most 2^-11,
    /// and a row's is at most that times the row's magnitude.
    private func bound(value: Float, magnitude: Float) -> Double {
        let ulp16 = abs(Double(value)) * 0.0009765625
        return ulp16 + Double(magnitude) * 0.00048828125
    }
}
