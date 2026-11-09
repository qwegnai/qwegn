import XCTest
import QWAlloc
import QWCore
@testable import QWEngine

final class QWWeightAllocationPlanTests: XCTestCase {
    /// The compiled-in 27B geometry — a stub, not an installed pack.
    static func geometry27B() throws -> QWModelGeometry {
        QWModelGeometry.qwen38_27b
    }

    func testShippingBudgetIncludesAlignmentAndRope() throws {
        let plan = try QWWeightAllocationPlan(geometry: try Self.geometry27B(), ropeRows: 8192)
        XCTAssertEqual(plan.shapes.count, 611)
        XCTAssertEqual(plan.bytes, 17_769_322_496)
        XCTAssertEqual(plan.shapes.first,
                       QWTensorShape([QWModelGeometry.qwen38_27b.vocab * QWModelGeometry.qwen38_27b.hidden], .f16))
        XCTAssertEqual(plan.shapes.last, QWTensorShape([8192, 64], .f32))
        // 48-element FP32 vectors need 256 bytes each, not 192.
        XCTAssertEqual(plan.shapes.filter { $0 == QWTensorShape([48], .f32) }.count, 96)
        let moreRows = try QWWeightAllocationPlan(geometry: try Self.geometry27B(), ropeRows: 8193)
        XCTAssertEqual(moreRows.bytes - plan.bytes, 256)
        XCTAssertEqual(plan.bytes % 256, 0)
        XCTAssertLessThan(plan.bytes, 46_000_000_000)
    }

    func testLoaderRequestsMustMatchExactly() throws {
        let plan = try QWWeightAllocationPlan(geometry: try Self.geometry27B(), ropeRows: 8192)
        for (index, shape) in plan.shapes.enumerated() {
            XCTAssertNoThrow(try plan.check(shape, at: index))
        }
        XCTAssertThrowsError(try plan.check(QWTensorShape([1], .f16), at: 0))
        XCTAssertThrowsError(try plan.check(plan.shapes[0], at: plan.shapes.count))
        XCTAssertThrowsError(try plan.check(plan.shapes[0], at: -1))
        XCTAssertNoThrow(try plan.finish(requests: plan.shapes.count, usedBytes: plan.bytes))
        XCTAssertThrowsError(try plan.finish(requests: plan.shapes.count - 1, usedBytes: plan.bytes))
        XCTAssertThrowsError(try plan.finish(requests: plan.shapes.count, usedBytes: plan.bytes - 256))
        XCTAssertThrowsError(try plan.finish(requests: plan.shapes.count, usedBytes: plan.bytes + 256))
    }

    func testQ4gHeadPlanUsesInterleavedPlaneBeforeLayerWeights() throws {
        let f16Plan = try QWWeightAllocationPlan(geometry: try Self.geometry27B(), ropeRows: 8192)
        let q4gPlan = try QWWeightAllocationPlan(geometry: try Self.geometry27B(), ropeRows: 8192, headQ4g: true)
        let groups = QWModelGeometry.qwen38_27b.vocab * (QWModelGeometry.qwen38_27b.hidden / 32)
        XCTAssertEqual(q4gPlan.shapes.first, QWTensorShape([groups * 10], .f16))
        XCTAssertEqual(q4gPlan.shapes.count, f16Plan.shapes.count)
        XCTAssertLessThan(q4gPlan.bytes, f16Plan.bytes)
        for (index, shape) in q4gPlan.shapes.enumerated() {
            XCTAssertNoThrow(try q4gPlan.check(shape, at: index))
        }
        XCTAssertNoThrow(try q4gPlan.finish(requests: q4gPlan.shapes.count,
                                             usedBytes: q4gPlan.bytes))
    }

    func testInvalidRopeRowsAreRefused() {
        XCTAssertThrowsError(try QWWeightAllocationPlan(geometry: try Self.geometry27B(), ropeRows: 0))
        XCTAssertThrowsError(try QWWeightAllocationPlan(geometry: try Self.geometry27B(), ropeRows: -1))
        XCTAssertThrowsError(try QWWeightAllocationPlan(geometry: try Self.geometry27B(), ropeRows: Int.max))
        XCTAssertThrowsError(try QWWeightAllocationPlan(geometry: try Self.geometry27B(), ropeRows: Int.max / 256 - 1))
    }

    func testMTPBudgetWhenEnabled() throws {
        let plan0 = try QWWeightAllocationPlan(geometry: try Self.geometry27B(), ropeRows: 8192, mtpDrafts: 0)
        XCTAssertEqual(plan0.shapes.count, 611)
        XCTAssertEqual(plan0.bytes, 17_769_322_496)

        let plan1 = try QWWeightAllocationPlan(geometry: try Self.geometry27B(), ropeRows: 8192, mtpDrafts: 1)
        XCTAssertEqual(plan1.shapes.count, 623)
        XCTAssertEqual(plan1.bytes, 18_034_797_056)
        XCTAssertEqual(plan1.bytes - plan0.bytes, 265_474_560)

        let plan2 = try QWWeightAllocationPlan(geometry: try Self.geometry27B(), ropeRows: 8192, mtpDrafts: 2)
        XCTAssertEqual(plan2.shapes.count, 623)
        XCTAssertEqual(plan2.bytes, 18_034_797_056)

        for (index, shape) in plan1.shapes.enumerated() {
            XCTAssertNoThrow(try plan1.check(shape, at: index))
        }
        XCTAssertNoThrow(try plan1.finish(requests: plan1.shapes.count, usedBytes: plan1.bytes))
    }

    func testInvalidMTPDraftsAreRefused() {
        XCTAssertThrowsError(try QWWeightAllocationPlan(geometry: try Self.geometry27B(), ropeRows: 8192, mtpDrafts: -1))
        XCTAssertThrowsError(try QWWeightAllocationPlan(geometry: try Self.geometry27B(), ropeRows: 8192, mtpDrafts: 6))
    }

    func testMTPConfigValidation() throws {
        XCTAssertEqual(try QWMTPConfig(drafts: 0).drafts, 0)
        XCTAssertEqual(try QWMTPConfig(drafts: 1).drafts, 1)
        XCTAssertEqual(try QWMTPConfig(drafts: 2).drafts, 2)
        XCTAssertEqual(try QWMTPConfig(drafts: 3).drafts, 3)
        XCTAssertEqual(try QWMTPConfig(drafts: 4).drafts, 4)
        XCTAssertEqual(try QWMTPConfig(drafts: 5).drafts, 5)
        XCTAssertThrowsError(try QWMTPConfig(drafts: -1))
        XCTAssertThrowsError(try QWMTPConfig(drafts: 6))
    }

    func testRoundRecordInitialization() {
        var rec = QWRoundRecord()
        XCTAssertNil(rec.pendingToken)
        XCTAssertEqual(rec.consumedCount, 0)
        XCTAssertEqual(rec.outputCount, 0)
        XCTAssertNil(rec.trunkHiddenSeed)
        XCTAssertEqual(rec.mtpPhysicalRows, 0)
        XCTAssertEqual(rec.absolutePosition, 0)
        XCTAssertEqual(rec.samplerDrawIndex, 0)

        rec.pendingToken = 42
        rec.consumedCount = 10
        rec.outputCount = 5
        rec.trunkHiddenSeed = [1.0, 2.0]
        rec.mtpPhysicalRows = 2
        rec.absolutePosition = 12
        rec.samplerDrawIndex = 7

        XCTAssertEqual(rec.pendingToken, 42)
        XCTAssertEqual(rec.consumedCount, 10)
        XCTAssertEqual(rec.outputCount, 5)
        XCTAssertEqual(rec.trunkHiddenSeed, [1.0, 2.0])
        XCTAssertEqual(rec.mtpPhysicalRows, 2)
        XCTAssertEqual(rec.absolutePosition, 12)
        XCTAssertEqual(rec.samplerDrawIndex, 7)
    }

    func testEngineConfigMTPDrafts() {
        var cfg = QWEngineConfig()
        XCTAssertEqual(cfg.mtpDrafts, 0)
        cfg.mtpDrafts = 1
        XCTAssertEqual(cfg.mtpDrafts, 1)
        cfg.mtpDrafts = 2
        XCTAssertEqual(cfg.mtpDrafts, 2)
    }

    static func glm53FlashGeometry(residentExperts: Int? = nil) throws -> QWModelGeometry {
        let glmConfig = """
        {
          "architectures": ["Glm5NextForConditionalGeneration"],
          "model_type": "glm5_next",
          "text_config": {
            "model_type": "glm5_next_text",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "vocab_size": 154880,
            "num_hidden_layers": 45,
            "num_nextn_predict_layers": 1,
            "layer_types": [
              "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
              "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
              "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
              "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
              "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
              "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
              "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
              "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
              "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
              "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
              "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
              "linear_attention"
            ],
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
            "rope_parameters": {
              "partial_rotary_factor": 1.0,
              "rope_theta": 1000000,
              "rope_type": "default"
            },
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
            "n_shared_experts": 1,
            "first_k_dense_replace": 3,
            "moe_layer_freq": 1,
            "routed_scaling_factor": 2.5,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "mlp_clamp": true,
            "swiglu_limits": 10.0,
            "norm_topk_prob": true,
            "index_n_heads": 32,
            "index_head_dim": 128,
            "index_topk": 2048,
            "index_kpool_compress": true,
            "index_kpool_always_select_tail": true,
            "index_share_for_mtp_iteration": true,
            "indexer_rope_interleave": true,
            "mhc": true,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "eos_token_id": [154820, 154827, 154829]
          }
        }
        """.data(using: .utf8)!
        let desc = try QWModelDescriptor.decode(configJSON: glmConfig)
        let geo = QWModelGeometry(descriptor: desc)
        if let k = residentExperts {
            return geo.withResidentExperts(k)
        }
        return geo
    }

    func testGlm53FlashResidencyTiersAndColdStaging() throws {
        let geo288 = try Self.glm53FlashGeometry(residentExperts: 288)
        XCTAssertEqual(geo288.effectiveResidentExperts, 288)
        XCTAssertEqual(geo288.coldExpertStagingSlots, 0)
        XCTAssertEqual(geo288.allocatedExpertSlots, 288)

        let geo216 = try Self.glm53FlashGeometry(residentExperts: 216)
        XCTAssertEqual(geo216.effectiveResidentExperts, 216)
        XCTAssertEqual(geo216.coldExpertStagingSlots, 8)
        XCTAssertEqual(geo216.allocatedExpertSlots, 224)

        let geo96 = try Self.glm53FlashGeometry(residentExperts: 96)
        XCTAssertEqual(geo96.effectiveResidentExperts, 96)
        XCTAssertEqual(geo96.coldExpertStagingSlots, 8)
        XCTAssertEqual(geo96.allocatedExpertSlots, 104)

        let fmt = QWWeightFormat.hqBlock(
            bits: 4, groupSize: 32,
            expertBits: 2, expertGroupSize: 128,
            denseIncumbent: true, expertIncumbent: false
        )
        let plan288 = try QWWeightAllocationPlan(
            geometry: geo288, ropeRows: 8192, headQ4g: true, format: fmt
        )
        let plan216 = try QWWeightAllocationPlan(
            geometry: geo216, ropeRows: 8192, headQ4g: true, format: fmt
        )
        let plan96 = try QWWeightAllocationPlan(
            geometry: geo96, ropeRows: 8192, headQ4g: true, format: fmt
        )

        XCTAssertEqual(plan288.bytes % 256, 0)
        XCTAssertEqual(plan216.bytes % 256, 0)
        XCTAssertEqual(plan96.bytes % 256, 0)

        let gib = Double(1024 * 1024 * 1024)
        XCTAssertLessThan(Double(plan288.bytes) / gib, 90.25)
        XCTAssertLessThan(Double(plan216.bytes) / gib, 69.51)
        XCTAssertLessThan(Double(plan96.bytes) / gib, 36.61)
        XCTAssertLessThan(plan216.bytes, plan288.bytes)
        XCTAssertLessThan(plan96.bytes, plan216.bytes)

        for (index, shape) in plan96.shapes.enumerated() {
            XCTAssertNoThrow(try plan96.check(shape, at: index))
        }
        XCTAssertNoThrow(try plan96.finish(requests: plan96.shapes.count, usedBytes: plan96.bytes))
    }
}
