// QWModelDescriptorPinningTests.swift — Pinning gate for geometryDigestV2Hex across all 12 model families.
import Foundation
import XCTest
@testable import QWCore

final class QWModelDescriptorPinningTests: XCTestCase {
    private func decodeString(_ json: String) throws -> QWModelDescriptor {
        try QWModelDescriptor.decode(configJSON: Data(json.utf8))
    }

    private func decodeJSON(_ obj: [String: Any]) throws -> QWModelDescriptor {
        try QWModelDescriptor.decode(configJSON: try JSONSerialization.data(withJSONObject: obj))
    }

    func testAll14FamiliesKeepBitIdenticalGeometryDigestV2AndTraitDefaults() throws {
        XCTAssertEqual(QWModelDescriptor.Family.allCases.count, 19)
        XCTAssertEqual(
            QWModelDescriptor.qwen38_27b.geometryDigestV2Hex,
            "9d6adc557a95b06707e353ac6379c9daa0ab1e597a86d779fbe4a0cd9e1b2a36"
        )
        let qwen38Dense = try decodeJSON([
            "model_type": "qwen3_5",
            "text_config": QWStubModelConfig.dense27BText(),
        ])
        XCTAssertEqual(qwen38Dense.family, .qwen38Dense)
        XCTAssertEqual(qwen38Dense.geometryDigestV2Hex,
                       "361588e9211e446d81c5dd3a6e3dd2ad7800c0339151fe4c9797accb8d09ce4a")
        XCTAssertEqual(qwen38Dense.maxSpeculativeDrafts, 15)
        XCTAssertFalse(qwen38Dense.supportsExpertStreaming)

        let qwen38MoE = try decodeString("""
        {
          "model_type": "qwen3_5_moe",
          "text_config": {
            "model_type": "qwen3_5_moe_text", "hidden_size": 2048, "vocab_size": 248320,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "full_attention_interval": 4, "rms_norm_eps": 1e-6, "max_position_embeddings": 262144,
            "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
            "attn_output_gate": true, "partial_rotary_factor": 0.25, "rope_theta": 10000000,
            "linear_key_head_dim": 128, "linear_num_key_heads": 16,
            "linear_num_value_heads": 32, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
            "num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512, "eos_token_id": 248044,
            "rope_parameters": {"mrope_section": [11, 11, 10], "rope_theta": 10000000, "partial_rotary_factor": 0.25}
          }
        }
        """)
        XCTAssertEqual(qwen38MoE.family, .qwen38MoE)
        XCTAssertEqual(qwen38MoE.geometryDigestV2Hex,
                       "35afc124918de872910b26824aefdf66548dfc9edd0e02bb169a03308721dc6d")
        XCTAssertEqual(qwen38MoE.maxSpeculativeDrafts, 15)
        XCTAssertTrue(qwen38MoE.supportsExpertStreaming)

        let flashNext = try QWModelDescriptor.decode(configJSON: try QWStubModelConfig.flashNextJSON())
        XCTAssertEqual(flashNext.family, .qwen38FlashNext)
        XCTAssertEqual(flashNext.geometryDigestV2Hex,
                       "66a7c788fc07f0e914b92b580bd1badcf03469b5c15ae44152d8758aa194acf1")
        XCTAssertEqual(flashNext.maxSpeculativeDrafts, 3)
        XCTAssertFalse(flashNext.supportsExpertStreaming)

        let loopLayers = (0..<32).map { ($0 + 1) % 4 == 0 ? "full_attention" : "linear_attention" }
        let senseNova = try decodeJSON([
            "model_type": "sensenova_loop",
            "loop_settings": ["start": 4, "end": 28, "eval_count": 2, "counts": [2]],
            "mhc": ["start": 4, "end": 28, "streams": 2, "iterations": 20],
            "text_config": [
                "model_type": "qwen3_5_text", "hidden_size": 4096, "intermediate_size": 12288,
                "vocab_size": 248320, "num_hidden_layers": 32, "full_attention_interval": 4,
                "layer_types": loopLayers,
                "rms_norm_eps": 1e-6, "max_position_embeddings": 524288,
                "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
                "attn_output_gate": true, "partial_rotary_factor": 0.25,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_conv_kernel_dim": 4,
                "mtp_num_hidden_layers": 0, "eos_token_id": 248044,
                "rope_parameters": [
                    "mrope_interleaved": true, "mrope_section": [11, 11, 10],
                    "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                    "rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 262144,
                ],
            ] as [String: Any],
        ])
        XCTAssertEqual(senseNova.family, .senseNovaLoop)
        XCTAssertEqual(senseNova.geometryDigestV2Hex,
                       "5d4f52da359180c21a32bb0b1d757611b06469cc4a8b1b1d0dbbed81a3ce4a4c")
        XCTAssertEqual(senseNova.maxSpeculativeDrafts, 15)

        let glmLayers = (0..<44).map { ($0 + 1) % 4 == 0 ? "deepseek_sparse_attention" : "linear_attention" }
            + ["linear_attention"]
        let glmText: [String: Any] = [
            "model_type": "glm5_next_text", "hidden_size": 4096, "intermediate_size": 12288,
            "vocab_size": 154880, "num_hidden_layers": 45, "num_nextn_predict_layers": 1,
            "layer_types": glmLayers,
            "rms_norm_eps": 1e-5, "max_position_embeddings": 1048576,
            "num_attention_heads": 64, "num_key_value_heads": 64,
            "qk_head_dim": 256, "qk_nope_head_dim": 192, "qk_rope_head_dim": 64,
            "v_head_dim": 256, "kv_lora_rank": 512, "q_lora_rank": 1536, "mla_use_nope": true,
            "rope_parameters": ["partial_rotary_factor": 1.0, "rope_theta": 1000000, "rope_type": "default"],
            "linear_attn_config": [
                "Linear_type": "kda", "gate_lower_bound": -5.0, "head_dim": 128,
                "num_heads": 64, "short_conv_kernel_size": 4,
            ],
            "n_routed_experts": 288, "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
            "n_shared_experts": 1, "first_k_dense_replace": 3, "moe_layer_freq": 1,
            "routed_scaling_factor": 2.5, "scoring_func": "sigmoid", "topk_method": "noaux_tc",
            "mlp_clamp": true, "swiglu_limits": 10.0, "norm_topk_prob": true,
            "index_n_heads": 32, "index_head_dim": 128, "index_topk": 2048,
            "index_kpool_compress": true, "index_kpool_always_select_tail": true,
            "index_share_for_mtp_iteration": true, "indexer_rope_interleave": true,
            "mhc": true, "hc_mult": 4, "hc_sinkhorn_iters": 20, "eos_token_id": [154820, 154827, 154829],
        ]
        let glm5 = try decodeJSON(["model_type": "glm5_next", "text_config": glmText])
        XCTAssertEqual(glm5.family, .glm5Next)
        XCTAssertEqual(glm5.geometryDigestV2Hex,
                       "c08e803c885f2fbdbbadeca64d18d9744676c66411ce19907be6fa70aa29c0cb")
        XCTAssertTrue(glm5.supportsExpertStreaming)

        let ernie = try decodeString("""
        {
          "model_type": "ernie5_5_moe", "hidden_size": 4096, "intermediate_size": 14336,
          "vocab_size": 201216, "num_hidden_layers": 43, "num_nextn_predict_layers": 1,
          "mtp_shared_last_layer": true,
          "compress_ratios": [
            128, 128, 128, 128, 128, 128, 128, -2,
            128, 128, 128, 128, 128, 128, -2,
            128, 128, 128, 128, 128, 128, -2,
            128, 128, 128, 128, 128, 128, -2,
            128, 128, 128, 128, 128, 128, -2,
            128, 128, 128, 128, 128, 128, -2, -2
          ],
          "rms_norm_eps": 1e-6, "max_position_embeddings": 1048576,
          "num_attention_heads": 64, "num_key_value_heads": 1, "head_dim": 512,
          "qk_nope_head_dim": 448, "qk_rope_head_dim": 64, "v_head_dim": 512,
          "q_lora_rank": 1024, "kv_lora_rank": 512, "o_groups": 8, "o_lora_rank": 1024,
          "sliding_window": 128, "compress_rope_theta": 160000.0, "hca_rope_type": "yarn",
          "rope_scaling": {"type": "yarn", "factor": 16, "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1},
          "rope_theta": 8000000, "rotary_percent": 1.0, "rope_interleave": true,
          "hybrid_mla_attention": "mqa_dsa", "hybrid_mla_num_attention_heads": 64,
          "hybrid_mla_q_lora_rank": 1024, "hybrid_mla_kv_lora_rank": 512,
          "hybrid_mla_qk_nope_head_dim": 192, "hybrid_mla_qk_rope_head_dim": 64, "hybrid_mla_v_head_dim": 256,
          "mqa_split_kv_b_proj": true, "index_n_heads": 64, "index_head_dim": 128, "index_topk": 2048,
          "use_fast_hadamard": true, "use_vha_attention": true, "add_full_attention_sink_bias": true,
          "use_attn_merger": true, "use_gated_attn": true, "gated_attn_use_q_lora": true,
          "enable_hyper_connections": true, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
          "hidden_act": "situ", "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0,
          "first_k_dense_replace": 1, "n_routed_experts": 512, "num_experts_per_tok": 10,
          "moe_intermediate_size": 2048, "moe_latent_size": 2048, "latent_moe_use_norm": true,
          "moe_split_feature_routing": true, "routed_scaling_factor": 3.0, "routed_scaling_factor_learnable": true,
          "n_shared_experts": 1, "scoring_func": "sigmoid", "topk_method": "noaux_tc",
          "multimax": "lm_head", "eos_token_id": 2
        }
        """)
        XCTAssertEqual(ernie.family, .ernie55MoE)
        XCTAssertEqual(ernie.geometryDigestV2Hex,
                       "91a5e033af397d9be8d23ae54964f6ac0a087c28e17cfbc1e1930e250c253897")
        XCTAssertTrue(ernie.supportsExpertStreaming)

        let maxwell = try decodeString("""
        {
          "model_type": "maxwell_moe", "hidden_size": 2048, "intermediate_size": 6144,
          "vocab_size": 200704, "num_hidden_layers": 45, "mlp_only_layers": [0, 1, 2],
          "num_experts": 256, "num_experts_per_tok": 12, "moe_intermediate_size": 512,
          "shared_expert_intermediate_size": 1024, "use_shared_expert_gate": false,
          "moe_router_enable_expert_bias": true, "router_score_function": "sigmoid",
          "num_attention_heads": 32, "num_key_value_heads": 4, "attention_head_size": 128,
          "qk_layernorm": true, "attention_variant": "block_sparse_gqa",
          "block_sparse_gqa_indexer_n_heads": 4, "block_sparse_gqa_indexer_head_dim": 128,
          "block_sparse_gqa_block_size": 128, "block_sparse_gqa_prefix_window_size": 1024,
          "block_sparse_gqa_local_window_size": 1024, "block_sparse_gqa_num_remote_blocks": 16,
          "enable_hyper_connections": true, "num_residual_streams": 4, "mhc_sinkhorn_iterations": 20,
          "mtp_num_layers": 1, "rms_norm_eps": 1e-6, "rope_theta": 20000000,
          "partial_rotary_factor": 1.0, "max_position_embeddings": 524288, "eos_token_id": 2
        }
        """)
        XCTAssertEqual(maxwell.family, .maxwellMoE)
        XCTAssertEqual(maxwell.geometryDigestV2Hex,
                       "e9008cee1cc96881fb02ef8fd0455aba7ee25f70cdb283594a53ce0d562ce740")
        XCTAssertTrue(maxwell.supportsExpertStreaming)

        let qwen3Dense = try decodeString("""
        {"model_type":"qwen3","hidden_size":4096,"vocab_size":151936,"num_hidden_layers":36,"num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"intermediate_size":12288,"max_position_embeddings":40960,"rms_norm_eps":1e-6,"rope_theta":1000000.0,"eos_token_id":151645}
        """)
        XCTAssertEqual(qwen3Dense.family, .qwen3Dense)
        XCTAssertEqual(qwen3Dense.geometryDigestV2Hex,
                       "5ec169ab79c5efb1c4789ebe0e017135c7e581c38fb610ce28493ccb1b7ea28c")

        let qwen3MoE = try decodeString("""
        {"model_type":"qwen3_moe","hidden_size":2048,"vocab_size":151936,"num_hidden_layers":4,"num_attention_heads":16,"num_key_value_heads":4,"head_dim":128,"intermediate_size":6144,"num_experts":128,"num_experts_per_tok":8,"moe_intermediate_size":768,"max_position_embeddings":32768,"rms_norm_eps":1e-6,"rope_theta":1000000.0,"eos_token_id":151645}
        """)
        XCTAssertEqual(qwen3MoE.family, .qwen3MoE)
        XCTAssertEqual(qwen3MoE.geometryDigestV2Hex,
                       "5d5073f73acf39f80b0389c3aa575cff7cf2624593f8f654dc3d8f37c85d3d4e")
        XCTAssertTrue(qwen3MoE.supportsExpertStreaming)

        let qwen2 = try decodeString("""
        {"model_type":"qwen2","hidden_size":3584,"vocab_size":152064,"num_hidden_layers":28,"num_attention_heads":28,"num_key_value_heads":4,"intermediate_size":18944,"max_position_embeddings":32768,"rms_norm_eps":1e-6,"rope_theta":1000000.0,"eos_token_id":151645}
        """)
        XCTAssertEqual(qwen2.family, .qwen2Dense)
        XCTAssertEqual(qwen2.geometryDigestV2Hex,
                       "1ec42336a22dadc1c4767f0e1cd29b4d97a0ef33cf1c759fa1cf6eaa77c83bd6")

        let qwen2QKV = try decodeString("""
        {"model_type":"qwen2","hidden_size":3584,"vocab_size":152064,"num_hidden_layers":28,"num_attention_heads":28,"num_key_value_heads":4,"intermediate_size":18944,"max_position_embeddings":32768,"rms_norm_eps":1e-6,"rope_theta":1000000.0,"eos_token_id":151645,"qkv_bias":true}
        """)
        let qwen2Attn = try decodeString("""
        {"model_type":"qwen2","hidden_size":3584,"vocab_size":152064,"num_hidden_layers":28,"num_attention_heads":28,"num_key_value_heads":4,"intermediate_size":18944,"max_position_embeddings":32768,"rms_norm_eps":1e-6,"rope_theta":1000000.0,"eos_token_id":151645,"attention_bias":true}
        """)
        XCTAssertEqual(qwen2QKV.geometryDigestV2Hex, qwen2.geometryDigestV2Hex)
        XCTAssertEqual(qwen2Attn.geometryDigestV2Hex, qwen2.geometryDigestV2Hex)
        XCTAssertNotEqual(qwen2QKV.geometryDigestV3Hex, qwen2.geometryDigestV3Hex)
        XCTAssertNotEqual(qwen2Attn.geometryDigestV3Hex, qwen2.geometryDigestV3Hex)
        XCTAssertNotEqual(qwen2Attn.geometryDigestV3Hex, qwen2QKV.geometryDigestV3Hex)

        let llama = try decodeString("""
        {"model_type":"llama","hidden_size":4096,"vocab_size":128256,"num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8,"intermediate_size":14336,"max_position_embeddings":131072,"rms_norm_eps":1e-5,"rope_theta":500000.0,"eos_token_id":[128001,128008,128009]}
        """)
        XCTAssertEqual(llama.family, .llamaDense)
        XCTAssertEqual(llama.geometryDigestV2Hex,
                       "8a609a66be69cb3d9ca67703858c2caeeb9acf233fe0d1e2f4e20e82c3a32dc4")

        let mistral = try decodeString("""
        {"model_type":"mistral","hidden_size":4096,"vocab_size":32000,"num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8,"intermediate_size":14336,"max_position_embeddings":32768,"rms_norm_eps":1e-5,"rope_theta":10000.0,"eos_token_id":2}
        """)
        XCTAssertEqual(mistral.family, .mistralDense)
        XCTAssertEqual(mistral.geometryDigestV2Hex,
                       "c15bc571c7e01b69e94efe9adf2b6c9c465c10fb4f9ea1e53b8fe8b54b96dcf6")

        // Standard unlocked features keep geometryDigestV2 immutable while separating V3.
        let mistralSW = try decodeString("""
        {"model_type":"mistral","hidden_size":4096,"vocab_size":32000,"num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8,"intermediate_size":14336,"max_position_embeddings":32768,"sliding_window":4096,"rope_scaling":{"rope_type":"linear","factor":2.0},"rms_norm_eps":1e-5,"rope_theta":10000.0,"eos_token_id":2}
        """)
        XCTAssertEqual(mistralSW.geometryDigestV2Hex, mistral.geometryDigestV2Hex)
        XCTAssertNotEqual(mistralSW.geometryDigestV3Hex, mistral.geometryDigestV3Hex)
        XCTAssertEqual(mistralSW.slidingWindow, 4096)
        XCTAssertEqual(mistralSW.ropeScaling, .linear(factor: 2.0))
        XCTAssertFalse(qwen3Dense.usesNativeSpeculative)
        XCTAssertTrue(QWModelDescriptor.qwen38_27b.usesNativeSpeculative)
        XCTAssertEqual(qwen3Dense.maxSpeculativeDrafts, 15)
        XCTAssertEqual(QWModelFamilyRegistry.shared.spec(forModelType: "mistral")?.isOfficialSupported, false)
    }

    func testAllTwelveFamiliesSpecPredicatesHaveConsistentTraits() {
        for fnName in ["qwen3_5", "qwen3_5_moe", "qwen4_exp", "sensenova_loop",
                       "glm5_next", "ernie5_5_moe", "maxwell_moe", "inkling2_loop", "bailing_moe_v3",
                       "gigachat4", "qwen3", "qwen3_moe", "qwen2", "llama", "mistral",
                       "lfm2", "lfm3", "d1_omni", "modernbert"] {
            XCTAssertNotNil(QWModelFamilyRegistry.shared.spec(forModelType: fnName), fnName)
        }
        let fnSpec = QWModelFamilyRegistry.shared.spec(forModelType: "qwen4_exp")
        XCTAssertEqual(fnSpec?.maxSpeculativeDrafts, 3)
        XCTAssertEqual(fnSpec?.supportsExpertStreaming, false)
    }
}
