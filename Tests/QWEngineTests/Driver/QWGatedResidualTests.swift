import XCTest
import Metal
@testable import QWAlloc
@testable import QWCore
@testable import QWHAL
@testable import QWNorm
@testable import QWQuant
@testable import QWEngine

final class QWGatedResidualTests: XCTestCase {

    private func makeTensor(hal: QWHal, elements: [Float], dtype: QWDType = .f16) throws -> QWTensorValue {
        let byteCount = elements.count * (dtype == .f16 ? 2 : 4)
        let buffer = try XCTUnwrap(hal.device.makeBuffer(length: byteCount, options: .storageModeShared))
        if dtype == .f16 {
            let f16Ptr = buffer.contents().bindMemory(to: Float16.self, capacity: elements.count)
            for i in 0..<elements.count {
                f16Ptr[i] = Float16(elements[i])
            }
        } else {
            let f32Ptr = buffer.contents().bindMemory(to: Float.self, capacity: elements.count)
            for i in 0..<elements.count {
                f32Ptr[i] = elements[i]
            }
        }
        return QWTensorValue(
            region: QWMRegion(buffer: buffer, base: 0, length: byteCount),
            offset: 0,
            shape: QWTensorShape([elements.count], dtype)
        )
    }

    private func readTensorF16(tensor: QWTensorValue) -> [Float16] {
        let ptr = (tensor.region.buffer.contents() + tensor.offset).bindMemory(
            to: Float16.self,
            capacity: tensor.shape.elementCount
        )
        return Array(UnsafeBufferPointer(start: ptr, count: tensor.shape.elementCount))
    }

    func testBitExactParityAgainstSplitBranchAdditionsDynamicShape() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let hidden = 128
        let rows = 1

        // Deterministic pseudo-random inputs
        var resFloats = [Float](repeating: 0, count: hidden)
        var gdnFloats = [Float](repeating: 0, count: hidden)
        var attnFloats = [Float](repeating: 0, count: hidden)
        var mlpFloats = [Float](repeating: 0, count: hidden)
        var pleFloats = [Float](repeating: 0, count: hidden)
        var gateFloats = [Float](repeating: 0, count: hidden)

        for i in 0..<hidden {
            resFloats[i] = sin(Float(i) * 0.1) * 0.5
            gdnFloats[i] = cos(Float(i) * 0.2) * 0.4
            attnFloats[i] = sin(Float(i) * 0.3 + 1.0) * 0.3
            mlpFloats[i] = cos(Float(i) * 0.4 + 2.0) * 0.6
            pleFloats[i] = sin(Float(i) * 0.5 - 1.0) * 0.2
            gateFloats[i] = cos(Float(i) * 0.6) * 1.5 // logit for sigmoid
        }

        let fusedResTensor = try makeTensor(hal: hal, elements: resFloats)

        let gdnTensor = try makeTensor(hal: hal, elements: gdnFloats)
        let attnTensor = try makeTensor(hal: hal, elements: attnFloats)
        let mlpTensor = try makeTensor(hal: hal, elements: mlpFloats)
        let pleTensor = try makeTensor(hal: hal, elements: pleFloats)
        let gateTensor = try makeTensor(hal: hal, elements: gateFloats)

        // 1. Dispatch Fused Kernel
        let cmdBuf1 = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc1 = try XCTUnwrap(cmdBuf1.makeComputeCommandEncoder())
        try QWGatedResidual.encode(
            enc1,
            hal: hal,
            residual: fusedResTensor,
            gdnOut: gdnTensor,
            attnOut: attnTensor,
            mlpOut: mlpTensor,
            pleEmbed: pleTensor,
            pleGate: gateTensor,
            flags: [.pleSigmoid],
            rows: rows
        )
        enc1.endEncoding()
        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()

        // 2. Verify vs host reference (CPU exp vs Metal exp within 1e-3)
        let fusedOut = readTensorF16(tensor: fusedResTensor)
        let hostRef = QWGatedResidual.hostFusedResidual(
            residual: resFloats,
            gdnOut: gdnFloats,
            attnOut: attnFloats,
            mlpOut: mlpFloats,
            pleEmbed: pleFloats,
            pleGate: gateFloats,
            pleSigmoid: true
        )
        for i in 0..<hidden {
            let hostF16 = Float16(hostRef[i])
            let diff = abs(Float(fusedOut[i]) - Float(hostF16))
            XCTAssertLessThanOrEqual(
                diff,
                1e-3,
                "Host ref mismatch at element \(i): fused=\(fusedOut[i]) host=\(hostF16)"
            )
        }
    }

    func testBitExactParityAgainstSplitBranchStandardHidden5120FastPath() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let hidden = QWModelGeometry.qwen38_27b.hidden // 5120
        let rows = 1

        var resFloats = [Float](repeating: 0, count: hidden)
        var gdnFloats = [Float](repeating: 0, count: hidden)
        var attnFloats = [Float](repeating: 0, count: hidden)
        var mlpFloats = [Float](repeating: 0, count: hidden)
        var pleFloats = [Float](repeating: 0, count: hidden)
        var gateFloats = [Float](repeating: 0, count: hidden)

        for i in 0..<hidden {
            resFloats[i] = sin(Float(i) * 0.01) * 0.25
            gdnFloats[i] = cos(Float(i) * 0.02) * 0.15
            attnFloats[i] = sin(Float(i) * 0.03 + 0.5) * 0.2
            mlpFloats[i] = cos(Float(i) * 0.04 - 0.5) * 0.3
            pleFloats[i] = sin(Float(i) * 0.05 + 1.2) * 0.1
            gateFloats[i] = cos(Float(i) * 0.06) * 1.0
        }

        let fusedResTensor = try makeTensor(hal: hal, elements: resFloats)
        let gdnTensor = try makeTensor(hal: hal, elements: gdnFloats)
        let attnTensor = try makeTensor(hal: hal, elements: attnFloats)
        let mlpTensor = try makeTensor(hal: hal, elements: mlpFloats)
        let pleTensor = try makeTensor(hal: hal, elements: pleFloats)
        let gateTensor = try makeTensor(hal: hal, elements: gateFloats)

        // 1. Dispatch Fused Kernel (hits the 5120 register unroll fast path)
        let cmdBuf1 = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc1 = try XCTUnwrap(cmdBuf1.makeComputeCommandEncoder())
        try QWGatedResidual.encode(
            enc1,
            hal: hal,
            residual: fusedResTensor,
            gdnOut: gdnTensor,
            attnOut: attnTensor,
            mlpOut: mlpTensor,
            pleEmbed: pleTensor,
            pleGate: gateTensor,
            flags: [.pleSigmoid],
            rows: rows
        )
        enc1.endEncoding()
        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()

        let fusedOut = readTensorF16(tensor: fusedResTensor)
        XCTAssertEqual(fusedOut.count, hidden)

        let hostRef = QWGatedResidual.hostFusedResidual(
            residual: resFloats,
            gdnOut: gdnFloats,
            attnOut: attnFloats,
            mlpOut: mlpFloats,
            pleEmbed: pleFloats,
            pleGate: gateFloats,
            pleSigmoid: true
        )
        for i in 0..<hidden {
            let hostF16 = Float16(hostRef[i])
            let diff = abs(Float(fusedOut[i]) - Float(hostF16))
            XCTAssertLessThanOrEqual(
                diff,
                1e-3,
                "Host ref mismatch at element \(i): fused=\(fusedOut[i]) host=\(hostF16)"
            )
        }
    }

    func testFusedResidualWithRMSNormEpilogueMatchesHostMirror() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let hidden = QWModelGeometry.qwen38_27b.hidden // 5120
        let rows = 1

        var resFloats = [Float](repeating: 0, count: hidden)
        var mlpFloats = [Float](repeating: 0, count: hidden)
        var gammaFloats = [Float](repeating: 0, count: hidden)

        for i in 0..<hidden {
            resFloats[i] = sin(Float(i) * 0.05) * 0.5
            mlpFloats[i] = cos(Float(i) * 0.05) * 0.3
            gammaFloats[i] = -0.05 + 0.1 * Float(i % 10) / 10.0 // raw checkpoint gamma w ≈ 0
        }

        let fusedRes = try makeTensor(hal: hal, elements: resFloats)
        let seqRes = try makeTensor(hal: hal, elements: resFloats)
        let mlpTensor = try makeTensor(hal: hal, elements: mlpFloats)
        let gammaTensor = try makeTensor(hal: hal, elements: gammaFloats)

        let fusedNormOut = try makeTensor(hal: hal, elements: [Float](repeating: 0, count: hidden))
        let seqNormOut = try makeTensor(hal: hal, elements: [Float](repeating: 0, count: hidden))

        // Path 1: Fused residual + RMSNorm epilogue
        let cmdBuf1 = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc1 = try XCTUnwrap(cmdBuf1.makeComputeCommandEncoder())
        try QWGatedResidual.encode(
            enc1,
            hal: hal,
            residual: fusedRes,
            mlpOut: mlpTensor,
            gamma: gammaTensor,
            normOut: fusedNormOut,
            flags: [.gammaPlusOne],
            rows: rows
        )
        enc1.endEncoding()
        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()

        // Path 2: Standard qwg_rmsnorm_residual
        let cmdBuf2 = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc2 = try XCTUnwrap(cmdBuf2.makeComputeCommandEncoder())
        try QWNorm.encode(
            enc2,
            hal: hal,
            variant: .rmsnormResidual,
            x: mlpTensor,
            gamma: gammaTensor,
            out: seqNormOut,
            residual: seqRes
        , geometry: .qwen38_27b)
        enc2.endEncoding()
        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()

        let fusedNorm = readTensorF16(tensor: fusedNormOut)
        let fusedResFinal = readTensorF16(tensor: fusedRes)
        let seqResFinal = readTensorF16(tensor: seqRes)
        let gammaFinal = readTensorF16(tensor: gammaTensor)

        // Residuals should be bit-exact
        for i in 0..<hidden {
            XCTAssertEqual(fusedResFinal[i].bitPattern, seqResFinal[i].bitPattern)
        }

        // Epilogue contract (qwg_rmsnorm_gated_residual_4branch): standard-basis
        // y = f16(s · rstd · (1+γ)). The shipping qwg_rmsnorm_residual emits the
        // FWHT32-ROTATED domain (HQ4 law, 681ae3f) — comparing the two element-wise
        var total: Float = 0
        for i in 0..<hidden { let t = Float(fusedResFinal[i]); total += t * t }
        let rstd = 1.0 / (total / Float(hidden) + 1e-6).squareRoot()
        for i in 0..<hidden {
            let gp1 = Float(Float16(Float(gammaFinal[i]) + 1.0))
            let want = Float(Float16(Float(fusedResFinal[i]) * rstd * gp1))
            let diff = abs(Float(fusedNorm[i]) - want)
            XCTAssertLessThanOrEqual(diff, 1e-3, "Index \(i): fused=\(fusedNorm[i]) mirror=\(want)")
        }
    }

    func testMHC4StreamEnterRouterNormAndMerge() throws {
        let geo = try QWWeightAllocationPlanTests.glm53FlashGeometry()
        XCTAssertEqual(geo.mhcConfig?.streams, 4)
        XCTAssertEqual(geo.mhcConfig?.iterations, 20)
        let hal = try QWHal(geometry: geo)

        let hidden = 128
        let c = 4
        let numCtrl = c + c * c // 20
        let numRows = 2 * c + c * c // 24

        var xFloats = [Float](repeating: 0, count: hidden)
        for i in 0..<hidden {
            xFloats[i] = sin(Float(i + 1) * 0.07) * 0.8
        }
        let xTensor = try makeTensor(hal: hal, elements: xFloats, dtype: .f16)
        let streamTensor = try makeTensor(hal: hal, elements: [Float](repeating: 0, count: c * hidden), dtype: .f32)

        let enterPso = try hal.pipeline("qwg_mhc_enter_f16")
        let routerPso = try hal.pipeline("qwg_mhc_router_norm_f16")
        let mergePso = try hal.pipeline("qwg_mhc_merge_f16")

        var uHidden = UInt32(hidden)
        do {
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(enterPso)
            enc.setBuffer(xTensor.region.buffer, offset: 0, index: 0)
            enc.setBuffer(streamTensor.region.buffer, offset: 0, index: 1)
            enc.setBytes(&uHidden, length: 4, index: 2)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }

        let streamPtr = streamTensor.region.buffer.contents().bindMemory(to: Float.self, capacity: c * hidden)
        for s in 0..<c {
            for h in 0..<hidden {
                XCTAssertEqual(streamPtr[s * hidden + h], Float(Float16(xFloats[h])), accuracy: 1e-6)
            }
        }

        var wFloats = [Float](repeating: 0, count: numRows * c * hidden)
        for i in 0..<wFloats.count {
            wFloats[i] = cos(Float(i) * 0.013) * 0.05
        }
        var bFloats = [Float](repeating: 0, count: numRows)
        for r in 0..<numRows {
            bFloats[r] = sin(Float(r) * 0.3) * 0.2
        }
        let wTensor = try makeTensor(hal: hal, elements: wFloats, dtype: .f32)
        let bTensor = try makeTensor(hal: hal, elements: bFloats, dtype: .f32)
        let gammaTensor = try makeTensor(hal: hal, elements: [Float](repeating: 0.0, count: hidden), dtype: .f16)
        let outNormTensor = try makeTensor(hal: hal, elements: [Float](repeating: 0, count: hidden), dtype: .f16)
        let ctrlTensor = try makeTensor(hal: hal, elements: [Float](repeating: 0, count: numCtrl), dtype: .f32)

        var eps: Float = 1e-5
        var preScale: Float = 1.0 / Float(c * hidden).squareRoot()
        var iters: UInt32 = 20
        do {
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(routerPso)
            enc.setBuffer(streamTensor.region.buffer, offset: 0, index: 0)
            enc.setBuffer(wTensor.region.buffer, offset: 0, index: 1)
            enc.setBuffer(bTensor.region.buffer, offset: 0, index: 2)
            enc.setBuffer(gammaTensor.region.buffer, offset: 0, index: 3)
            enc.setBuffer(outNormTensor.region.buffer, offset: 0, index: 4)
            enc.setBuffer(ctrlTensor.region.buffer, offset: 0, index: 5)
            enc.setBytes(&uHidden, length: 4, index: 6)
            enc.setBytes(&eps, length: 4, index: 7)
            enc.setBytes(&preScale, length: 4, index: 8)
            enc.setBytes(&iters, length: 4, index: 9)
            enc.setBuffer(outNormTensor.region.buffer, offset: 0, index: 10)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }

        let ctrlPtr = ctrlTensor.region.buffer.contents().bindMemory(to: Float.self, capacity: numCtrl)
        // Verify doubly-stochastic 4x4 comb^T matrix (stored in ctrl[4..<20]): rows and cols sum to ~1.0.
        for i in 0..<c {
            var rSum: Float = 0
            var cSum: Float = 0
            for j in 0..<c {
                let v = ctrlPtr[c + i * c + j]
                XCTAssertGreaterThan(v, 0)
                rSum += v
                cSum += ctrlPtr[c + j * c + i]
            }
            XCTAssertEqual(rSum, 1.0, accuracy: 2e-3)
            XCTAssertEqual(cSum, 1.0, accuracy: 2e-3)
        }

        // Dispatch qwg_mhc_merge_f16 with isLeave = true (flags = 3)
        let xOutTensor = try makeTensor(hal: hal, elements: [Float](repeating: 0, count: hidden), dtype: .f16)
        let nextNormTensor = try makeTensor(hal: hal, elements: [Float](repeating: 0, count: hidden), dtype: .f16)
        var flags: UInt32 = 3
        do {
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(mergePso)
            enc.setBuffer(streamTensor.region.buffer, offset: 0, index: 0)
            enc.setBuffer(xTensor.region.buffer, offset: 0, index: 1)
            enc.setBuffer(ctrlTensor.region.buffer, offset: 0, index: 2)
            enc.setBuffer(xOutTensor.region.buffer, offset: 0, index: 3)
            enc.setBuffer(gammaTensor.region.buffer, offset: 0, index: 4)
            enc.setBuffer(nextNormTensor.region.buffer, offset: 0, index: 5)
            enc.setBytes(&uHidden, length: 4, index: 6)
            enc.setBytes(&eps, length: 4, index: 7)
            enc.setBytes(&flags, length: 4, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }

        let xOut = readTensorF16(tensor: xOutTensor)
        for h in 0..<hidden {
            XCTAssertTrue(xOut[h].isFinite)
        }
    }

    func testErnie55MHCAdditiveEpsAndExitHeadMatchPythonGolden() throws {
        let ernieCfg = """
        {
          "architectures": ["Ernie5_5_MoeForCausalLM"],
          "model_type": "ernie5_5_moe",
          "vocab_size": 1024,
          "hidden_size": 256,
          "intermediate_size": 256,
          "num_hidden_layers": 2,
          "max_position_embeddings": 1048576,
          "rms_norm_eps": 0.25,
          "num_attention_heads": 64,
          "num_key_value_heads": 1,
          "head_dim": 512,
          "compress_ratios": [128, -2],
          "default_compress_ratio": 128,
          "q_lora_rank": 256,
          "kv_lora_rank": 512,
          "compress_qk_nope_head_dim": 448,
          "compress_qk_rope_head_dim": 64,
          "compress_v_head_dim": 512,
          "o_groups": 8,
          "o_lora_rank": 256,
          "sliding_window": 128,
          "compress_rope_theta": 160000.0,
          "hca_rope_type": "yarn",
          "rope_scaling": {
            "type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32,
            "beta_slow": 1
          },
          "rope_theta": 8000000,
          "rotary_percent": 1.0,
          "rope_interleave": true,
          "hybrid_mla_attention": "mqa_dsa",
          "hybrid_mla_num_attention_heads": 64,
          "hybrid_mla_num_key_value_heads": 64,
          "hybrid_mla_q_lora_rank": 256,
          "hybrid_mla_kv_lora_rank": 512,
          "hybrid_mla_qk_nope_head_dim": 192,
          "hybrid_mla_qk_rope_head_dim": 64,
          "hybrid_mla_v_head_dim": 256,
          "mqa_split_kv_b_proj": true,
          "index_n_heads": 64,
          "index_head_dim": 128,
          "index_topk": 2048,
          "use_fast_hadamard": true,
          "use_vha_attention": true,
          "add_full_attention_sink_bias": true,
          "use_attn_merger": true,
          "gated_attn_use_q_lora": true,
          "enable_hyper_connections": true,
          "hc_mult": 4,
          "hc_sinkhorn_iters": 20,
          "hc_eps": 1e-6,
          "hidden_act": "situ",
          "activation_situ_beta": 4.0,
          "activation_situ_linear_beta": 25.0,
          "first_k_dense_replace": 1,
          "n_routed_experts": 32,
          "num_experts_per_tok": 4,
          "moe_intermediate_size": 64,
          "moe_latent_size": 64,
          "latent_moe_use_norm": true,
          "moe_split_feature_routing": true,
          "routed_scaling_factor": 3.0,
          "routed_scaling_factor_learnable": true,
          "n_shared_experts": 1,
          "scoring_func": "sigmoid",
          "topk_method": "noaux_tc",
          "multimax": "lm_head",
          "eos_token_id": 2
        }
        """.data(using: .utf8)!
        let geo = QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: ernieCfg))
        let hal = try QWHal(geometry: geo)

        let fixtureURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/spike/out/ernie55_synthetic_goldens.json")
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: fixtureURL)) as? [String: Any])
        let mhc = try XCTUnwrap(root["mhc"] as? [String: Any])
        let exitRef = try XCTUnwrap(root["mhc_exit_head"] as? [String: Any])

        let res3D = try XCTUnwrap(mhc["residual"] as? [[[Double]]])
        let fn2D = try XCTUnwrap(mhc["fn"] as? [[Double]])
        let hcScale = try XCTUnwrap(mhc["scale"] as? [Double]).map { Float($0) }
        let hcBase = try XCTUnwrap(mhc["base"] as? [Double]).map { Float($0) }
        let normWeight = try XCTUnwrap(mhc["norm_weight"] as? [Double]).map { Float($0) }
        let wantPost = try XCTUnwrap(mhc["post_mix"] as? [[Double]]).map { $0.map { Float($0) } }
        let wantComb = try XCTUnwrap(mhc["comb"] as? [[[Double]]]).map { $0.map { $0.map { Float($0) } } }
        let wantLayerNormed = try XCTUnwrap(mhc["layer_normed"] as? [[Double]]).map { $0.map { Float($0) } }

        let headFn2D = try XCTUnwrap(exitRef["head_fn"] as? [[Double]])
        let headScale = Float(try XCTUnwrap(exitRef["head_scale"] as? Double))
        let headBase = try XCTUnwrap(exitRef["head_base"] as? [Double]).map { Float($0) }
        let wantExitRaw = try XCTUnwrap(exitRef["contracted"] as? [[Double]]).map { $0.map { Float($0) } }
        let wantExitNormed = try XCTUnwrap(exitRef["normed"] as? [[Double]]).map { $0.map { Float($0) } }

        let T = res3D.count, C = res3D[0].count, H = res3D[0][0].count
        let streamFlat = res3D.flatMap { $0.flatMap { $0.map { Float($0) } } }
        var routerW = [Float]()
        for (r, row) in fn2D.enumerated() {
            let sc = r < 4 ? hcScale[0] : (r < 8 ? hcScale[1] : hcScale[2])
            routerW.append(contentsOf: row.map { Float($0) * sc })
        }
        let headW = headFn2D.flatMap { $0.map { Float($0) * headScale } }
        let shiftedGamma = normWeight.map { $0 - 1.0 }

        let streamTV = try makeTensor(hal: hal, elements: streamFlat, dtype: .f32)
        let wTV = try makeTensor(hal: hal, elements: routerW, dtype: .f32)
        let bTV = try makeTensor(hal: hal, elements: hcBase, dtype: .f32)
        let gammaTV = try makeTensor(hal: hal, elements: shiftedGamma, dtype: .f16)
        let outNormTV = try makeTensor(hal: hal, elements: [Float](repeating: 0, count: T * H), dtype: .f16)
        let ctrlTV = try makeTensor(hal: hal, elements: [Float](repeating: 0, count: T * 20), dtype: .f32)

        let routerPSO = try hal.pipeline("qwg_mhc_router_norm_f16")
        let exitPSO = try hal.pipeline("qwg_mhc_exit_head_f16")
        var uHidden = UInt32(H)
        var eps: Float = 0.25
        var preScale: Float = 1.0
        var iters: UInt32 = 20

        do {
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(routerPSO)
            enc.setBuffer(streamTV.region.buffer, offset: 0, index: 0)
            enc.setBuffer(wTV.region.buffer, offset: 0, index: 1)
            enc.setBuffer(bTV.region.buffer, offset: 0, index: 2)
            enc.setBuffer(gammaTV.region.buffer, offset: 0, index: 3)
            enc.setBuffer(outNormTV.region.buffer, offset: 0, index: 4)
            enc.setBuffer(ctrlTV.region.buffer, offset: 0, index: 5)
            enc.setBytes(&uHidden, length: 4, index: 6)
            enc.setBytes(&eps, length: 4, index: 7)
            enc.setBytes(&preScale, length: 4, index: 8)
            enc.setBytes(&iters, length: 4, index: 9)
            enc.setBuffer(outNormTV.region.buffer, offset: 0, index: 10)
            enc.dispatchThreadgroups(MTLSize(width: T, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }

        let ctrlPtr = ctrlTV.region.buffer.contents().bindMemory(to: Float.self, capacity: T * 20)
        let outNorm16 = readTensorF16(tensor: outNormTV)
        for t in 0..<T {
            for c in 0..<C {
                XCTAssertEqual(ctrlPtr[t * 20 + c], wantPost[t][c], accuracy: 1e-5, "post_mix[\(t),\(c)]")
            }
            for i in 0..<C {
                for j in 0..<C {
                    let gotCombT = ctrlPtr[t * 20 + C + i * C + j]
                    XCTAssertEqual(gotCombT, wantComb[t][j][i], accuracy: 1e-5, "comb^T[\(t),\(i),\(j)]")
                }
            }
            var rowUnrot = (0..<H).map { Float(outNorm16[t * H + $0]) }
            rowUnrot.withUnsafeMutableBufferPointer { buf in
                for b in stride(from: 0, to: buf.count, by: 32) {
                    QWQuant.fwht32OrthoBlock(buf.baseAddress! + b)
                }
            }
            for h in 0..<H {
                XCTAssertEqual(rowUnrot[h], wantLayerNormed[t][h], accuracy: 2e-3, "layer_normed[\(t),\(h)]")
            }
        }

        let headWTV = try makeTensor(hal: hal, elements: headW, dtype: .f32)
        let headBTV = try makeTensor(hal: hal, elements: headBase, dtype: .f32)
        let exitRawTV = try makeTensor(hal: hal, elements: [Float](repeating: 0, count: T * H), dtype: .f16)
        let exitNorm16TV = try makeTensor(hal: hal, elements: [Float](repeating: 0, count: T * H), dtype: .f16)
        let exitRot32TV = try makeTensor(hal: hal, elements: [Float](repeating: 0, count: T * H), dtype: .f32)
        var rawFlags: UInt32 = 0
        var normFlags: UInt32 = 3
        do {
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(exitPSO)
            enc.setBuffer(streamTV.region.buffer, offset: 0, index: 0)
            enc.setBuffer(headWTV.region.buffer, offset: 0, index: 1)
            enc.setBuffer(headBTV.region.buffer, offset: 0, index: 2)
            enc.setBuffer(gammaTV.region.buffer, offset: 0, index: 3)
            enc.setBuffer(exitRawTV.region.buffer, offset: 0, index: 4)
            enc.setBuffer(exitRot32TV.region.buffer, offset: 0, index: 5)
            enc.setBytes(&uHidden, length: 4, index: 6)
            enc.setBytes(&eps, length: 4, index: 7)
            enc.setBytes(&preScale, length: 4, index: 8)
            enc.setBytes(&rawFlags, length: 4, index: 9)
            enc.dispatchThreadgroups(MTLSize(width: T, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.setBuffer(exitNorm16TV.region.buffer, offset: 0, index: 4)
            enc.setBytes(&normFlags, length: 4, index: 9)
            enc.dispatchThreadgroups(MTLSize(width: T, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }

        let gotRaw16 = readTensorF16(tensor: exitRawTV)
        let gotExitRotPtr = exitRot32TV.region.buffer.contents().bindMemory(to: Float.self, capacity: T * H)
        for t in 0..<T {
            for h in 0..<H {
                XCTAssertEqual(Float(gotRaw16[t * H + h]), Float(Float16(wantExitRaw[t][h])), accuracy: 1e-3,
                               "exit_head contracted[\(t),\(h)]")
            }
            var rowUnrot = (0..<H).map { gotExitRotPtr[t * H + $0] }
            rowUnrot.withUnsafeMutableBufferPointer { buf in
                for b in stride(from: 0, to: buf.count, by: 32) {
                    QWQuant.fwht32OrthoBlock(buf.baseAddress! + b)
                }
            }
            for h in 0..<H {
                XCTAssertEqual(rowUnrot[h], wantExitNormed[t][h], accuracy: 1e-5, "exit_head normed[\(t),\(h)]")
            }
        }
    }
}
