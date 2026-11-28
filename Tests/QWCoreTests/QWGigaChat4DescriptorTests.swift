// QWGigaChat4DescriptorTests.swift — Tests GigaChat4 family decoding, capabilities, and V2/V3 digest parity.
import Foundation
import XCTest
@testable import QWCore

final class QWGigaChat4DescriptorTests: XCTestCase {
    private static let canonicalDigestV2 = "6aa3f3e504094dc6aa5cf32623fd59f7b24917da480c5528987202f320c2b92d"

    private static func syntheticGigaChat4Config() -> [String: Any] {
        [
            "architectures": ["GigaChat4ForCausalLM"],
            "auto_map": [
                "AutoConfig": "configuration_gigachat4.GigaChat4Config",
                "AutoModelForCausalLM": "modeling_gigachat4.GigaChat4ForCausalLM",
            ],
            "id2label": ["0": "LABEL_0", "1": "LABEL_1"],
            "label2id": ["LABEL_0": 0, "LABEL_1": 1],
            "max_length": 20,
            "model_type": "gigachat4",
            "hidden_size": 2560,
            "intermediate_size": 8192,
            "vocab_size": 128256,
            "num_hidden_layers": 32,
            "full_attention_layers": [3, 7, 11, 15, 19, 23, 27, 31],
            "linear_attention_type": "gated_deltanet",
            "linear_gating_type": "gated_rmsnorm_sigmoid",
            "linear_sigmoid_gate_scale": 2.0,
            "linear_attn_o_norm_eps": 1e-6,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "head_dim": 64,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "qk_head_dim": 192,
            "v_head_dim": 128,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 2048,
            "norm_type": "GatedNorm",
            "indexer_norm_type": "LlamaRMSNorm",
            "layernorm_type": "pre_post",
            "layernorm_gating_weight": 2.0,
            "gated_attention": true,
            "use_mla_scaling_factor": true,
            "first_k_dense_replace": 1,
            "n_routed_experts": 192,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "n_shared_experts": 1,
            "routed_scaling_factor": 2.5,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "norm_topk_prob": true,
            "use_shared_expert_sigmoid": false,
            "num_nextn_predict_layers": 1,
            "nextn_is_sparse": false,
            "nextn_is_linear": false,
            "final_logit_softcapping": 30.0,
            "rms_norm_eps": 1e-6,
            "rope_theta": 100000.0,
            "max_position_embeddings": 1048576,
            "eos_token_id": 2,
            "rope_parameters": [
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 32768,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "rope_theta": 100000.0,
            ] as [String: Any],
        ]
    }

    func testGigaChat4CapabilitiesCausalGuardAndCanonicalDigestV2() throws {
        let data = try JSONSerialization.data(withJSONObject: Self.syntheticGigaChat4Config())
        let d = try QWModelDescriptor.decode(configJSON: data)

        XCTAssertEqual(d.family, .gigachat4)
        XCTAssertEqual(d.role, .causalLM, "HF boilerplate id2label + max_length=20 must not misclassify ForCausalLM")
        XCTAssertNil(d.decisionConfig)
        XCTAssertEqual(d.maxPositionEmbeddings, 1048576)
        XCTAssertEqual(d.headDim, 192)
        XCTAssertEqual(d.v2DigestHeadDim, 64)
        XCTAssertEqual(d.geometryDigestV2Hex, Self.canonicalDigestV2)

        XCTAssertEqual(
            d.gatedNormConfig,
            QWModelDescriptor.GatedNormConfig(rank: 16, gatingWeight: 2.0, zeroCentered: false)
        )
        XCTAssertNil(d.normKind)
        XCTAssertTrue(d.useSandwichNorm)
        XCTAssertEqual(
            d.gdnGatingConfig,
            QWModelDescriptor.GDNGatingConfig(outputNormAct: .sigmoid, outputGateScale: 2.0, outputNormEps: 1e-6)
        )
        XCTAssertEqual(d.finalLogitSoftcapping, 30.0)

        XCTAssertEqual(d.layerTypes.count, 32)
        XCTAssertEqual(d.fullAttentionInterval, 4)
        for idx in 0..<32 {
            let want = ((idx + 1) % 4 == 0) ? "full_attention" : "linear_attention"
            XCTAssertEqual(d.layerTypes[idx], want, "layer \(idx)")
        }

        let mla = try XCTUnwrap(d.mlaConfig)
        XCTAssertEqual(mla.qLoraRank, 1536)
        XCTAssertEqual(mla.kvLoraRank, 512)
        XCTAssertEqual(mla.qkNopeHeadDim, 128)
        XCTAssertEqual(mla.qkRopeHeadDim, 64)
        XCTAssertEqual(mla.vHeadDim, 128)
        XCTAssertEqual(mla.alphaQ, Float((2560.0 / 1536.0).squareRoot()), accuracy: 1e-6)
        XCTAssertEqual(mla.alphaKV, Float((2560.0 / 512.0).squareRoot()), accuracy: 1e-6)
        XCTAssertTrue(mla.indexerKNormIsRMS)
        XCTAssertEqual(mla.softmaxScale, 0.13086058, accuracy: 1e-6)
        XCTAssertEqual(mla.outputGateMode, .elementWiseFromHidden)
        XCTAssertEqual(mla.rope?.interleave, true)
        XCTAssertTrue(d.ropeInterleave)

        let moe = try XCTUnwrap(d.moeRoutingConfig)
        XCTAssertEqual(d.firstKDenseReplace, 1)
        XCTAssertFalse(d.sharedExpertHasGate)
        XCTAssertEqual(moe.sharedExpertHasGate, false)
        XCTAssertTrue(moe.enableExpertBias)
        XCTAssertEqual(moe.scoreFunction, "sigmoid")

        let mtp = try XCTUnwrap(d.mtpConfig)
        XCTAssertEqual(mtp.layers, 1)
        XCTAssertFalse(mtp.isMoE)
        XCTAssertEqual(mtp.intermediateSize, 8192)
        XCTAssertFalse(mtp.attentionIsSparse)
        XCTAssertFalse(mtp.attentionIsLinear)
        XCTAssertTrue(mtp.sharedEmbedAndHead)

        XCTAssertEqual(
            d.specialTokens,
            QWModelSpecialTokens(
                eosTokenId: 2,
                imStartTokenId: 128000,
                imEndTokenId: 128001,
                thinkTokenId: 128014,
                endThinkTokenId: 128015,
                extraStopTokenIds: [2, 128001],
                usesChatTemplateIds: true
            )
        )

        let geo = QWModelGeometry(descriptor: d)
        XCTAssertEqual(geo.headDim, 192)
        XCTAssertEqual(geo.gatedNormConfig, d.gatedNormConfig)
        XCTAssertTrue(geo.useSandwichNorm)
        XCTAssertEqual(geo.gdnGatingConfig, d.gdnGatingConfig)
        XCTAssertEqual(geo.finalLogitSoftcapping, 30.0)
        XCTAssertEqual(geo.mlaConfig, d.mlaConfig)
        XCTAssertEqual(geo.mtpConfig, d.mtpConfig)
    }

    func testZeroCenteredGatedNormAndV3DigestSensitivity() throws {
        var base = Self.syntheticGigaChat4Config()
        let dBase = try QWModelDescriptor.decode(configJSON: try JSONSerialization.data(withJSONObject: base))

        base["norm_type"] = "ZeroCenteredGatedNorm"
        let dZero = try QWModelDescriptor.decode(configJSON: try JSONSerialization.data(withJSONObject: base))
        XCTAssertEqual(dZero.gatedNormConfig?.zeroCentered, true)
        XCTAssertEqual(dZero.geometryDigestV2Hex, dBase.geometryDigestV2Hex)
        XCTAssertNotEqual(dZero.geometryDigestV3Hex, dBase.geometryDigestV3Hex)
    }

    func testDecodeDrafterResolvesDecoupledMLAHeadDimAndGatedNorm() throws {
        let drafterObj: [String: Any] = [
            "architectures": ["GigaChat4DFlash2DraftModel"],
            "model_type": "gigachat4",
            "hidden_size": 2560,
            "intermediate_size": 8192,
            "vocab_size": 128256,
            "max_position_embeddings": 1048576,
            "rms_norm_eps": 1e-6,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "num_hidden_layers": 5,
            "norm_type": "GatedNorm",
            "layernorm_type": "pre_post",
            "layernorm_gating_weight": 2.0,
            "gated_attention": true,
            "use_mla_scaling_factor": true,
            "dflash_config": [
                "block_size": 8,
                "selector_rank": 256,
                "selector_top_k": 16,
                "attention_mode": "mla",
                "final_logit_softcapping": 30.0,
            ] as [String: Any],
        ]
        let dd = try QWModelDescriptor.decodeDrafter(
            configJSON: try JSONSerialization.data(withJSONObject: drafterObj)
        )
        XCTAssertEqual(dd.family, .gigachat4)
        XCTAssertEqual(dd.headDim, 192)
        XCTAssertEqual(dd.gatedNormConfig?.rank, 16)
        XCTAssertTrue(dd.useSandwichNorm)
        XCTAssertEqual(dd.finalLogitSoftcapping, 30.0)
        XCTAssertEqual(dd.mlaConfig?.outputGateMode, .elementWiseFromHidden)
    }

    func testRealSnapshotConfigMatchesCanonicalDigestV2IfPresent() throws {
        let snap = NSHomeDirectory() + "/.cache/huggingface/hub/models--ai-sage--GigaChat4-37B-A4B/"
            + "snapshots/1bf9a4b2b8d7d0f970e48d6458be77d49e07bcfc"
        let cfgPath = snap + "/config.json"
        try XCTSkipUnless(FileManager.default.fileExists(atPath: cfgPath), "local GigaChat4 snapshot absent")
        let d = try QWModelDescriptor.decode(configJSON: Data(contentsOf: URL(fileURLWithPath: cfgPath)))
        XCTAssertEqual(d.geometryDigestV2Hex, Self.canonicalDigestV2)
        XCTAssertEqual(d.role, .causalLM)
        XCTAssertEqual(d.headDim, 192)

        let dflashPath = snap + "/dflash_draft/config.json"
        if FileManager.default.fileExists(atPath: dflashPath) {
            let dd = try QWModelDescriptor.decodeDrafter(
                configJSON: Data(contentsOf: URL(fileURLWithPath: dflashPath))
            )
            XCTAssertEqual(dd.headDim, 192)
            XCTAssertEqual(dd.finalLogitSoftcapping, 30.0)
        }
    }
}
