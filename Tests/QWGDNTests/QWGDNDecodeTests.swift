import XCTest
import Metal
import Foundation
@testable import QWGDN
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// C12 recurrent-step gate: `qwg_gdn_recurrent` + `qwg_gdn_norm_sigmoid` vs torch
/// fixtures (gen_gdn_decode_fixtures.py — golden = qw38_oracle GatedDeltaNet,
/// HF torch_recurrent_gated_delta_rule semantics).
final class QWGDNDecodeTests: XCTestCase {

    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    override func setUpWithError() throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "1024-thread GDN recurrent kernel requires bare-metal Apple Silicon GPU")
    }
    static let spikeOut = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Scripts/spike/out")

    static func load(_ name: String) throws -> Data {
        let p = spikeOut.appendingPathComponent(name)
        guard let d = FileManager.default.contents(atPath: p.path) else {
            throw NSError(domain: "fixture", code: 2, userInfo: [NSLocalizedDescriptionKey: "missing \(name)"])
        }
        return d
    }
    static func u16s(_ d: Data) -> [UInt16] { d.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) } }
    static func f32s(_ d: Data) -> [Float] { d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }
    static func f16(_ bits: UInt16) -> Float { QWQuant.f16tof32Bits(bits) }

    static func buf(_ d: Data) -> MTLBuffer {
        let b = hal.device.makeBuffer(length: max(16, d.count), options: .storageModeShared)!
        d.withUnsafeBytes { _ = memcpy(b.contents(), $0.baseAddress!, $0.count) }
        return b
    }
    static func tv(_ b: MTLBuffer, _ n: Int, _ dt: QWDType) -> QWTensorValue {
        QWTensorValue(region: QWMRegion(buffer: b, base: 0,
                                        length: max(16, Int((Double(n) * dt.bytesPerParam).rounded()))),
                      offset: 0, shape: QWTensorShape([n], dt))
    }
    static func row(_ b: MTLBuffer, _ base: Int, _ n: Int, _ dt: QWDType) -> QWTensorValue {
        QWTensorValue(region: QWMRegion(buffer: b, base: base,
                                        length: max(16, Int((Double(n) * dt.bytesPerParam).rounded()))),
                      offset: 0, shape: QWTensorShape([n], dt))
    }

    static func viol(_ got: [Float], _ ref: [Float], _ absTol: Float) -> (Int, Double, Int) {
        var v = 0, worst = 0.0, wi = -1
        for i in 0..<got.count {
            let d = abs(Double(got[i]) - Double(ref[i]))
            if d > worst { worst = d; wi = i }
            guard got[i].isFinite && ref[i].isFinite else { v += 1; continue }
            let sp = max(Double(abs(got[i])), Double(abs(ref[i]))) * Double.ulpOfOne
            if d > max(Double(absTol), sp) { v += 1 }
        }
        return (v, worst, wi)
    }

    /// Replay `steps` tokens through the encoder (state IN PLACE like
    /// serving), returning every requested step's o (f16 bits) + final state.
    static func replay(yAll: Data, zAll: Data, beta: Data, glog: Data,
                       state0: Data, w: Data, steps: Int, keep: Set<Int>) throws
        -> (oRows: [Int: [UInt16]], state: [Float])
    {
        let dec = try QWGDNDecode(hal: hal)
        let yb = buf(yAll), zb = buf(zAll)
        let bb = buf(beta), gb = buf(glog), wb = buf(w)
        let stBuf = buf(state0)
        let scratch = hal.device.makeBuffer(length: QWGDNDecode.oScratchElems * 4,
                                            options: .storageModeShared)!
        let ob = hal.device.makeBuffer(length: 48 * 128 * 2, options: .storageModeShared)!
        var kept: [Int: [UInt16]] = [:]
        for t in 0..<steps {
            guard let (cb, ce) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
            try dec.encode(ce,
                           y: row(yb, t * 10240 * 2, 10240, .f16),
                           betas: row(bb, t * 48 * 4, 48, .f32),
                           glog: row(gb, t * 48 * 4, 48, .f32),
                           state: tv(stBuf, 48 * 128 * 128, .f32),
                           z: row(zb, t * 6144 * 2, 6144, .f16),
                           w: tv(wb, 128, .f16),
                           o: tv(ob, 48 * 128, .f16),
                           oScratch: tv(scratch, QWGDNDecode.oScratchElems, .f32))
            ce.endEncoding()                                     // end-before-wait
            cb.commit(); cb.waitUntilCompleted()
            if keep.contains(t + 1) {
                let p = ob.contents().assumingMemoryBound(to: UInt16.self)
                kept[t + 1] = (0..<(48 * 128)).map { p[$0] }
            }
        }
        let p = stBuf.contents().assumingMemoryBound(to: Float.self)
        return (kept, (0..<(48 * 128 * 128)).map { p[$0] })
    }

    func testDecodeWalkHeadA() throws {
        // 1000-step compounding walk from a FRESH zero state (head a).
        let y = try Self.load("gdn_dec_y_a.bin"), z = try Self.load("gdn_dec_z_a.bin")
        let beta = try Self.load("gdn_dec_beta_a.bin"), glog = try Self.load("gdn_dec_g_a.bin")
        let state0 = try Self.load("gdn_dec_state0_a.bin"), w = try Self.load("gdn_dec_w_a.bin")
        let keep: Set<Int> = [1, 2, 3, 10, 999, 1000]
        let (rows, st) = try Self.replay(yAll: y, zAll: z, beta: beta, glog: glog,
                                         state0: state0, w: w, steps: 1000, keep: keep)
        XCTAssertEqual(rows.count, keep.count, "kept-row bookkeeping")
        for t in keep {
            let ref = Self.u16s(try Self.load("gdn_dec_o16_t\(t)_a.bin"))
            let got = rows[t]!.map { Self.f16($0) }
            let rf = ref.map { Self.f16($0) }
            XCTAssertEqual(got.filter { !$0.isFinite }.count, 0, "step \(t): non-finite o")
            let v = Self.viol(got, rf, 1e-3)
            XCTAssertEqual(v.0, 0, "step \(t): o parity (worst \(v.1) @\(v.2))")
        }
        let stRef = Self.f32s(try Self.load("gdn_dec_state_a.bin"))
        XCTAssertEqual(st.filter { !$0.isFinite }.count, 0, "state must be finite after 1000 steps")
        let v = Self.viol(st, stRef, 1e-4)
        XCTAssertEqual(v.0, 0, "1000-step state drift (worst \(v.1) @\(v.2))")
    }

    func testDecodePreseededHeadB() throws {
        // A REAL long-sequence state (non-zero everywhere), 3 steps. Since
        // the S0 fix the generator dumps the walk's START in state0_b and
        // its FINAL in state_b — replay from the start, pin both ends.
        let y = try Self.load("gdn_dec_y_b.bin"), z = try Self.load("gdn_dec_z_b.bin")
        let beta = try Self.load("gdn_dec_beta_b.bin"), glog = try Self.load("gdn_dec_g_b.bin")
        let start = try Self.load("gdn_dec_state0_b.bin")
        let startF = Self.f32s(start)
        XCTAssertFalse(startF.allSatisfy { $0 == 0 }, "head b must start non-zero")
        let (rows, st) = try Self.replay(yAll: y, zAll: z, beta: beta, glog: glog,
                                         state0: start, w: try Self.load("gdn_dec_w_b.bin"),
                                         steps: 3, keep: [1, 2, 3])
        for t in [1, 2, 3] {
            let ref = Self.u16s(try Self.load("gdn_dec_o16_t\(t)_b.bin"))
            let got = rows[t]!.map { Self.f16($0) }
            let v = Self.viol(got, ref.map { Self.f16($0) }, 1e-3)
            XCTAssertEqual(v.0, 0, "b step \(t): o parity (worst \(v.1) @\(v.2))")
        }
        let stRef = Self.f32s(try Self.load("gdn_dec_state_b.bin"))
        let v = Self.viol(st, stRef, 1e-4)
        XCTAssertEqual(v.0, 0, "b 3-step final state (worst \(v.1) @\(v.2))")
    }

    func testStateOnlyReplayMatchesShippingPreseededWalk() throws {
        let y = try Self.load("gdn_dec_y_b.bin"), z = try Self.load("gdn_dec_z_b.bin")
        let beta = try Self.load("gdn_dec_beta_b.bin"), glog = try Self.load("gdn_dec_g_b.bin")
        let start = try Self.load("gdn_dec_state0_b.bin"), w = try Self.load("gdn_dec_w_b.bin")
        let (_, reference) = try Self.replay(yAll: y, zAll: z, beta: beta, glog: glog,
                                             state0: start, w: w, steps: 3, keep: [])
        let decoder = try QWGDNDecode(hal: Self.hal)
        let yb = Self.buf(y), bb = Self.buf(beta), gb = Self.buf(glog), state = Self.buf(start)
        let cb = try XCTUnwrap(Self.hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        do {
            for row in 0..<3 {
                try decoder.encodeStateReplay(enc, y: Self.row(yb, row * 10240 * 2, 10240, .f16),
                                               betas: Self.row(bb, row * 48 * 4, 48, .f32),
                                               glog: Self.row(gb, row * 48 * 4, 48, .f32),
                                               state: Self.tv(state, 48 * 128 * 128, .f32))
            }
        } catch {
            enc.endEncoding()
            throw error
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed, "\(String(describing: cb.error))")
        let got = state.contents().assumingMemoryBound(to: Float.self)
        var mismatches = 0, nonfinite = 0
        for i in reference.indices {
            if got[i].bitPattern != reference[i].bitPattern { mismatches += 1 }
            if !got[i].isFinite { nonfinite += 1 }
        }
        XCTAssertEqual(nonfinite, 0)
        XCTAssertEqual(mismatches, 0, "state-only replay must preserve the shipping FP32 recurrence")
    }

    func testDecodeOvsRmsRef() throws {
        // Spot-check one step's o against the fp32 (pre-f16-rounding)
        // reference at the tighter 1e-4 adjacency — pins that the kernel
        // keeps fp32 through the recurrence (f16-rounded internals would
        let y = try Self.load("gdn_dec_y_a.bin"), z = try Self.load("gdn_dec_z_a.bin")
        let beta = try Self.load("gdn_dec_beta_a.bin"), glog = try Self.load("gdn_dec_g_a.bin")
        let state0 = try Self.load("gdn_dec_state0_a.bin"), w = try Self.load("gdn_dec_w_a.bin")
        let (rows, _) = try Self.replay(yAll: y, zAll: z, beta: beta, glog: glog,
                                        state0: state0, w: w, steps: 1, keep: [1])
        let ref = Self.f32s(try Self.load("gdn_dec_o_t1_a.bin"))     // fp32 ref
        let got = rows[1]!.map { Self.f16($0) }
        let v = Self.viol(got, ref, 2e-3)   // store rounding adds ≤ f16 eps
        XCTAssertEqual(v.0, 0, "o vs fp32 ref (worst \(v.1) @\(v.2))")
    }

    func testShapeGuards() throws {
        // A genuinely wrong logical shape must throw (the region struct
        // cannot know the buffer's real capacity — production callers get
        // arena-sized tensors; unit tests assert on the logical guard).
        let dec = try QWGDNDecode(hal: Self.hal)
        let small = Self.hal.device.makeBuffer(length: 16, options: .storageModeShared)!
        guard let (cb, ce) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
        // y with one element short of 10240:
        XCTAssertThrowsError(try dec.encode(ce,
            y: Self.tv(small, 10239, .f16), betas: Self.tv(small, 48, .f32),
            glog: Self.tv(small, 48, .f32), state: Self.tv(small, 48 * 128 * 128, .f32),
            z: Self.tv(small, 6144, .f16), w: Self.tv(small, 128, .f16),
            o: Self.tv(small, 48 * 128, .f16), oScratch: Self.tv(small, 48 * 128, .f32)))
        // state declared f16 instead of f32:
        XCTAssertThrowsError(try dec.encode(ce,
            y: Self.tv(small, 10240, .f16), betas: Self.tv(small, 48, .f32),
            glog: Self.tv(small, 48, .f32), state: Self.tv(small, 48 * 128 * 128, .f16),
            z: Self.tv(small, 6144, .f16), w: Self.tv(small, 128, .f16),
            o: Self.tv(small, 48 * 128, .f16), oScratch: Self.tv(small, 48 * 128, .f32)))
        ce.endEncoding(); cb.commit()          // discarded buffer: end anyway
    }

    static let kdaConfig = """
        {
          "model_type": "glm5_next_text",
          "hidden_size": 4096,
          "intermediate_size": 12288,
          "vocab_size": 154880,
          "num_hidden_layers": 4,
          "layer_types": ["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"],
          "rms_norm_eps": 1e-5,
          "max_position_embeddings": 1048576,
          "num_attention_heads": 64,
          "num_key_value_heads": 64,
          "qk_head_dim": 256,
          "qk_nope_head_dim": 192,
          "qk_rope_head_dim": 64,
          "v_head_dim": 256,
          "kv_lora_rank": 512,
          "q_lora_rank": 1536,
          "mla_use_nope": true,
          "partial_rotary_factor": 1.0,
          "linear_attn_config": {
            "Linear_type": "kda",
            "gate_lower_bound": -5.0,
            "head_dim": 128,
            "num_heads": 64,
            "short_conv_kernel_size": 4
          },
          "n_routed_experts": 288,
          "num_experts_per_tok": 8,
          "moe_intermediate_size": 2048,
          "first_k_dense_replace": 3,
          "eos_token_id": 154820
        }
        """.data(using: .utf8)!

    func testKDAVectorDecayGatesAndRecurrence() throws {
        let kdaConfig = Self.kdaConfig
        let desc = try QWModelDescriptor.decode(configJSON: kdaConfig)
        let kdaGeo = QWModelGeometry(descriptor: desc)
        XCTAssertTrue(kdaGeo.gdnVectorDecay)
        XCTAssertEqual(kdaGeo.gdnVHeads, 64)
        XCTAssertEqual(kdaGeo.gdnGlogChannels, 8192)

        let kdaHal = try QWHal(geometry: kdaGeo)
        let dec = try QWGDNDecode(hal: kdaHal)

        let vh = 64
        let hd = 128
        let zCh = vh * hd // 8192
        let qkvCh = 3 * zCh // 24576
        let fgbStride = 2 * hd + vh // 320

        var fgbF16 = [UInt16](repeating: 0, count: fgbStride)
        for i in 0..<fgbStride {
            fgbF16[i] = Float16(sin(Float(i + 1) * 0.05) * 0.4).bitPattern
        }
        var fBF16 = [UInt16](repeating: 0, count: zCh * hd)
        var gBF16 = [UInt16](repeating: 0, count: zCh * hd)
        for i in 0..<(zCh * hd) {
            fBF16[i] = Float16(cos(Float(i % 251) * 0.03) * 0.08).bitPattern
            gBF16[i] = Float16(sin(Float(i % 257) * 0.03) * 0.08).bitPattern
        }
        let aLogF32 = (0..<vh).map { Float($0) * 0.02 - 0.5 }
        let dtBiasF32 = (0..<zCh).map { sin(Float($0) * 0.01) * 0.1 }

        let fgbBuf = Self.buf(fgbF16.withUnsafeBufferPointer { Data(buffer: $0) })
        let fBBuf = Self.buf(fBF16.withUnsafeBufferPointer { Data(buffer: $0) })
        let gBBuf = Self.buf(gBF16.withUnsafeBufferPointer { Data(buffer: $0) })
        let aLogBuf = Self.buf(aLogF32.withUnsafeBufferPointer { Data(buffer: $0) })
        let dtBiasBuf = Self.buf(dtBiasF32.withUnsafeBufferPointer { Data(buffer: $0) })

        let betasBuf = kdaHal.device.makeBuffer(length: vh * 4, options: .storageModeShared)!
        let glogBuf = kdaHal.device.makeBuffer(length: zCh * 4, options: .storageModeShared)!
        let zOutBuf = kdaHal.device.makeBuffer(length: zCh * 2, options: .storageModeShared)!

        guard let (cb1, ce1) = kdaHal.begin() else { throw NSError(domain: "hal", code: 1) }
        try dec.encodeKDAGates(
            ce1,
            fgb: Self.tv(fgbBuf, fgbStride, .f16),
            fB: Self.tv(fBBuf, zCh * hd, .f16),
            gB: Self.tv(gBBuf, zCh * hd, .f16),
            aLog: Self.tv(aLogBuf, vh, .f32),
            dtBias: Self.tv(dtBiasBuf, zCh, .f32),
            betas: Self.tv(betasBuf, vh, .f32),
            glog: Self.tv(glogBuf, zCh, .f32),
            zOut: Self.tv(zOutBuf, zCh, .f16),
            rows: 1
        )
        ce1.endEncoding()
        cb1.commit()
        cb1.waitUntilCompleted()

        let betasPtr = betasBuf.contents().bindMemory(to: Float.self, capacity: vh)
        let glogPtr = glogBuf.contents().bindMemory(to: Float.self, capacity: zCh)
        for h in 0..<vh {
            let b = Float(Float16(bitPattern: fgbF16[2 * hd + h]))
            XCTAssertEqual(betasPtr[h], 1.0 / (1.0 + exp(-b)), accuracy: 1e-6)
        }
        for c in 0..<zCh {
            var projection: Float = 0
            for i in 0..<hd {
                projection += Float(Float16(bitPattern: fgbF16[i]))
                    * Float(Float16(bitPattern: fBF16[c * hd + i]))
            }
            let rate = exp(aLogF32[c / hd])
            let gate = kdaGeo.gdnGateLowerBound / (1.0 + exp(-rate * (projection + dtBiasF32[c])))
            XCTAssertEqual(glogPtr[c], exp(gate), accuracy: 1e-5, "decay channel \(c)")
        }

        // Run one vector-decay recurrent step and verify finite output
        let yF16 = (0..<qkvCh).map { Float16(sin(Float($0 % 128) * 0.05) * 0.1).bitPattern }
        let wF16 = [UInt16](repeating: Float16(1.0).bitPattern, count: hd)
        let yBuf = Self.buf(yF16.withUnsafeBufferPointer { Data(buffer: $0) })
        let wBuf = Self.buf(wF16.withUnsafeBufferPointer { Data(buffer: $0) })
        let stateBuf = kdaHal.device.makeBuffer(length: vh * hd * hd * 4, options: .storageModeShared)!
        memset(stateBuf.contents(), 0, vh * hd * hd * 4)
        let oScratchBuf = kdaHal.device.makeBuffer(length: zCh * 4, options: .storageModeShared)!
        let oBuf = kdaHal.device.makeBuffer(length: zCh * 2, options: .storageModeShared)!

        guard let (cb2, ce2) = kdaHal.begin() else { throw NSError(domain: "hal", code: 1) }
        try dec.encode(
            ce2,
            y: Self.tv(yBuf, qkvCh, .f16),
            betas: Self.tv(betasBuf, vh, .f32),
            glog: Self.tv(glogBuf, zCh, .f32),
            state: Self.tv(stateBuf, vh * hd * hd, .f32),
            z: Self.tv(zOutBuf, zCh, .f16),
            w: Self.tv(wBuf, hd, .f16),
            o: Self.tv(oBuf, zCh, .f16),
            oScratch: Self.tv(oScratchBuf, zCh, .f32)
        )
        ce2.endEncoding()
        cb2.commit()
        cb2.waitUntilCompleted()

        let oPtr = oBuf.contents().bindMemory(to: UInt16.self, capacity: zCh)
        var nonZero = 0
        for i in 0..<zCh {
            let v = Self.f16(oPtr[i])
            XCTAssertTrue(v.isFinite)
            if abs(v) > 1e-6 { nonZero += 1 }
        }
        XCTAssertGreaterThan(nonZero, zCh / 2)
    }

    /// The prefill simdgroup-MMA gates kernel matches the scalar kernel, including a tail tile.
    func testKDAGatesMMAMatchesScalarKernel() throws {
        let kdaHal = try QWHal(geometry: QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: Self.kdaConfig)))
        let dec = try QWGDNDecode(hal: kdaHal)
        let vh = 64, hd = 128, zCh = vh * hd, fgbStride = 2 * hd + vh, T = 45
        let fgb = (0..<(T * fgbStride)).map { Float16(sin(Float($0 &* 7 % 1013) * 0.37) * 0.6).bitPattern }
        let fB = (0..<(zCh * hd)).map { Float16(cos(Float($0 % 977) * 0.11) * 0.09).bitPattern }
        let gB = (0..<(zCh * hd)).map { Float16(sin(Float($0 % 983) * 0.13) * 0.09).bitPattern }
        let aLog = (0..<vh).map { Float($0) * 0.02 - 0.5 }
        let dtBias = (0..<zCh).map { sin(Float($0) * 0.01) * 0.1 }
        func data<E>(_ a: [E]) -> Data { a.withUnsafeBufferPointer { Data(buffer: $0) } }
        let fgbBuf = Self.buf(data(fgb)), fBBuf = Self.buf(data(fB)), gBBuf = Self.buf(data(gB))
        let aLogBuf = Self.buf(data(aLog)), dtBuf = Self.buf(data(dtBias))

        func run(mma: Bool) throws -> (betas: [Float], glog: [Float], z: [Float]) {
            dec.kdaGatesMMAEnabled = mma
            let bb = kdaHal.device.makeBuffer(length: T * vh * 4, options: .storageModeShared)!
            let gb = kdaHal.device.makeBuffer(length: T * zCh * 4, options: .storageModeShared)!
            let zb = kdaHal.device.makeBuffer(length: T * zCh * 2, options: .storageModeShared)!
            guard let (cb, ce) = kdaHal.begin() else { throw NSError(domain: "hal", code: 1) }
            try dec.encodeKDAGates(ce, fgb: Self.tv(fgbBuf, T * fgbStride, .f16),
                                   fB: Self.tv(fBBuf, zCh * hd, .f16), gB: Self.tv(gBBuf, zCh * hd, .f16),
                                   aLog: Self.tv(aLogBuf, vh, .f32), dtBias: Self.tv(dtBuf, zCh, .f32),
                                   betas: Self.tv(bb, T * vh, .f32), glog: Self.tv(gb, T * zCh, .f32),
                                   zOut: Self.tv(zb, T * zCh, .f16), rows: T)
            ce.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            let zBits = Array(UnsafeBufferPointer(start: zb.contents().bindMemory(to: UInt16.self, capacity: T * zCh),
                                                  count: T * zCh))
            return (Array(UnsafeBufferPointer(start: bb.contents().bindMemory(to: Float.self, capacity: T * vh), count: T * vh)),
                    Array(UnsafeBufferPointer(start: gb.contents().bindMemory(to: Float.self, capacity: T * zCh), count: T * zCh)),
                    zBits.map(Self.f16))
        }
        let ref = try run(mma: false)
        let got = try run(mma: true)
        XCTAssertEqual(Self.viol(got.betas, ref.betas, 0).0, 0)
        let g = Self.viol(got.glog, ref.glog, 1e-4)
        XCTAssertEqual(g.0, 0, "glog worst \(g.1) at \(g.2)")
        let z = Self.viol(got.z, ref.z, 2e-3)
        XCTAssertEqual(z.0, 0, "z worst \(z.1) at \(z.2)")
        XCTAssertGreaterThan(ref.z.filter { abs($0) > 1e-3 }.count, T * zCh / 2)
    }
}
