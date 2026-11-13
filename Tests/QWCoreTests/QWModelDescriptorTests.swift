import XCTest
@testable import QWCore

/// §2 — QWModelDescriptor gate (flash-next plan):
///  (b) the v2 geometry digest is canonical, stable and model-binding,
final class QWModelDescriptorTests: XCTestCase {





    func testControl27BLegacyV1CharacterIdentity() throws {
        // QWQuant.geometryHashString builds this exact format from QWModelGeometry;
        // pin the literal so the descriptor seam can never drift it.
        let control = QWModelDescriptor.qwen38_27b
        XCTAssertEqual(control.legacyGeoV1String,
                       "qwg-geo-v1:5120|248320|17408|64|24|4|256|10240|48")
        XCTAssertEqual(control.verifyInternalConsistency(), [])
        XCTAssertEqual(control.family, .qwen38Dense)
        XCTAssertEqual(control.fullAttentionLayers.count, 16)
        XCTAssertEqual(control.specialTokens.defaultStopTokenIds,
                       [QWModelGeometry.qwen38_27b.endOfTextTokenId, QWModelGeometry.qwen38_27b.imEndTokenId])
        XCTAssertEqual(control.specialTokens.endThinkTokenId, QWModelGeometry.qwen38_27b.endThinkTokenId)
    }

    func testModelFamilyRegistryDynamicRegistration() throws {
        let registry = QWModelFamilyRegistry.shared
        XCTAssertTrue(registry.isSupported(modelType: "qwen3_5"))
        XCTAssertTrue(registry.isSupported(modelType: "qwen4_exp_text"))
        XCTAssertFalse(registry.isSupported(modelType: "custom_exp_v1"))

        let customSpec = QWModelFamilySpec(
            modelType: "custom_exp_v1",
            family: .qwen38Dense,
            defaultAttnOutputGate: true,
            defaultExpertGateType: nil,
            isOfficialSupported: true
        )
        registry.register(customSpec)
        defer { registry.unregister(modelType: "custom_exp_v1") }

        XCTAssertTrue(registry.isSupported(modelType: "custom_exp_v1"))
        let customConfig = """
        {"model_type":"custom_exp_v1","hidden_size":5120,"vocab_size":248320,
         "num_hidden_layers":4,"layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
         "full_attention_interval":4,"rms_norm_eps":1e-6,"max_position_embeddings":262144,
         "num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,
         "partial_rotary_factor":0.25,"linear_key_head_dim":128,"linear_num_key_heads":16,
         "linear_num_value_heads":48,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
         "eos_token_id":199999,"intermediate_size":17408}
        """.data(using: .utf8)!
        let decoded = try QWModelDescriptor.decode(configJSON: customConfig)
        XCTAssertEqual(decoded.family, .qwen38Dense)
        XCTAssertTrue(decoded.attnOutputGate)
        XCTAssertEqual(decoded.specialTokens.eosTokenId, 199999)
        XCTAssertTrue(decoded.specialTokens.defaultStopTokenIds.contains(199999))
    }

    func testTheHybridGeometryDigestIsPinned() throws {
        // The shipped packs carry this digest and QWPack compares against it,
        // so a silent change here would invalidate every pack. The golden
        // config is the real one, not a stub.
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let golden = repoRoot.appendingPathComponent("Scripts/oracle/config_golden.json")
        let decoded = try QWModelDescriptor.decode(configJSON: try Data(contentsOf: golden))
        XCTAssertEqual(decoded.geometryDigestV2Hex,
                       "361588e9211e446d81c5dd3a6e3dd2ad7800c0339151fe4c9797accb8d09ce4a")
        XCTAssertEqual(decoded.verifyInternalConsistency(), [])
    }

    func testMalformedConfigsFailClosed() throws {
        // cadence violation: full attention at wrong slot
        let badCadence = """
        {"model_type":"qwen3_5","hidden_size":5120,"vocab_size":248320,
         "num_hidden_layers":4,"layer_types":["full_attention","linear_attention","linear_attention","linear_attention"],
         "full_attention_interval":4,"rms_norm_eps":1e-6,"max_position_embeddings":262144,
         "num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,
         "partial_rotary_factor":0.25,"linear_key_head_dim":128,"linear_num_key_heads":16,
         "linear_num_value_heads":48,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
         "eos_token_id":248044,"intermediate_size":17408}
        """.data(using: .utf8)!
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: badCadence)) { err in
            guard case QWModelDescriptor.DescriptorError.invariant(let m) = err else {
                return XCTFail("expected invariant error, got \(err)")
            }
            XCTAssertTrue(m.contains("cadence"), m)
        }
    }

    func testGlm53FlashDescriptorDecodeAndConsistency() throws {
        let glmConfig = """
        {
          "architectures": ["Glm5NextForConditionalGeneration"],
          "model_type": "glm5_next",
          "image_token_id": 154842,
          "video_token_id": 154843,
          "vision_start_token_id": 154838,
          "vision_end_token_id": 154839,
          "vision_config": {
            "depth": 24,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "out_hidden_size": 4096,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 4096
          },
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
        let d = try QWModelDescriptor.decode(configJSON: glmConfig)
        XCTAssertEqual(d.verifyInternalConsistency(), [])
        XCTAssertEqual(d.family, .qwen38Dense)
        XCTAssertEqual(d.modelType, "glm5_next_text")
        XCTAssertEqual(d.envelopeModelType, "glm5_next")
        XCTAssertFalse(d.usesPlusOneRMSNorm)
        XCTAssertFalse(d.usesChatTemplateIds)
        XCTAssertEqual(d.hidden, 4096)
        XCTAssertEqual(d.vocab, 154880)
        XCTAssertEqual(d.denseIntermediate, 12288)
        XCTAssertEqual(d.numLayers, 45)
        XCTAssertEqual(d.fullAttentionLayers.count, 11)
        XCTAssertEqual(d.firstKDenseReplace, 3)
        XCTAssertFalse(d.isMoELayer(0))
        XCTAssertFalse(d.isMoELayer(1))
        XCTAssertFalse(d.isMoELayer(2))
        XCTAssertTrue(d.isMoELayer(3))
        XCTAssertTrue(d.isMoELayer(44))
        XCTAssertEqual(d.numExperts, 288)
        XCTAssertEqual(d.expertsPerTok, 8)
        XCTAssertEqual(d.moeIntermediate, 2048)
        XCTAssertEqual(d.sharedExpertIntermediate, 2048)
        XCTAssertFalse(d.sharedExpertHasGate)
        XCTAssertEqual(d.routedScalingFactor, 2.5)
        XCTAssertEqual(d.swigluLimit, 10.0)
        XCTAssertTrue(d.hasBiasCorrectedRouter)
        XCTAssertEqual(d.mhcConfig, QWModelDescriptor.MHCConfig(start: 0, end: 45, streams: 4, iterations: 20))
        XCTAssertTrue(d.mlaUseNoPE)
        XCTAssertEqual(d.kvLoraRank, 512)
        XCTAssertEqual(d.qLoraRank, 1536)
        XCTAssertTrue(d.indexKPoolCompress)
        XCTAssertTrue(d.indexKPoolAlwaysSelectTail)
        XCTAssertTrue(d.indexShareForMTP)
        XCTAssertTrue(d.indexerRopeInterleave)
        XCTAssertEqual(d.indexerHeads, 32)
        XCTAssertEqual(d.indexerHeadDim, 128)
        XCTAssertEqual(d.indexerBudget, 2048)
        XCTAssertEqual(d.indexerCompressRatio, 4)
        XCTAssertTrue(d.gdnVectorDecay)
        XCTAssertEqual(d.gdnQKHeads, 64)
        XCTAssertEqual(d.gdnVHeads, 64)
        XCTAssertEqual(d.gdnHeadDim, 128)
        XCTAssertEqual(d.gdnLowRankDim, 128)
        XCTAssertEqual(d.gdnGateLowerBound, -5.0)
        XCTAssertEqual(d.withResidentExperts(96).effectiveResidentExperts, 96)
    }

    func testErnie55FlashDescriptorDecodeGeometryLSAAndPythonDigestParity() throws {
        let ernieConfig = """
        {
          "architectures": ["Ernie5_5_MoeForCausalLM"],
          "model_type": "ernie5_5_moe",
          "hidden_size": 4096,
          "intermediate_size": 14336,
          "vocab_size": 201216,
          "num_hidden_layers": 43,
          "num_nextn_predict_layers": 1,
          "mtp_shared_last_layer": true,
          "compress_ratios": [
            128, 128, 128, 128, 128, 128, 128, -2,
            128, 128, 128, 128, 128, 128, -2,
            128, 128, 128, 128, 128, 128, -2,
            128, 128, 128, 128, 128, 128, -2,
            128, 128, 128, 128, 128, 128, -2,
            128, 128, 128, 128, 128, 128, -2,
            -2
          ],
          "rms_norm_eps": 1e-6,
          "max_position_embeddings": 1048576,
          "num_attention_heads": 64,
          "num_key_value_heads": 1,
          "head_dim": 512,
          "qk_nope_head_dim": 448,
          "qk_rope_head_dim": 64,
          "v_head_dim": 512,
          "q_lora_rank": 1024,
          "kv_lora_rank": 512,
          "o_groups": 8,
          "o_lora_rank": 1024,
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
          "hybrid_mla_q_lora_rank": 1024,
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
          "use_gated_attn": true,
          "gated_attn_use_q_lora": true,
          "enable_hyper_connections": true,
          "hc_mult": 4,
          "hc_sinkhorn_iters": 20,
          "hc_eps": 1e-6,
          "hidden_act": "situ",
          "activation_situ_beta": 4.0,
          "activation_situ_linear_beta": 25.0,
          "first_k_dense_replace": 1,
          "n_routed_experts": 512,
          "num_experts_per_tok": 10,
          "moe_intermediate_size": 2048,
          "moe_latent_size": 2048,
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

        let d = try QWModelDescriptor.decode(configJSON: ernieConfig)
        XCTAssertEqual(d.verifyInternalConsistency(), [])
        XCTAssertEqual(d.family, .qwen38Dense)
        XCTAssertEqual(d.modelType, "ernie5_5_moe")
        XCTAssertFalse(d.usesPlusOneRMSNorm)
        XCTAssertTrue(d.attnOutputGate)
        XCTAssertFalse(d.usesChatTemplateIds)
        XCTAssertEqual(d.compressedAttentionLayers.count, 37)
        XCTAssertEqual(d.fullAttentionLayers, [7, 14, 21, 28, 35, 42])
        XCTAssertEqual(d.linearAttentionLayers, [])
        XCTAssertEqual(d.fullAttentionInterval, 0)
        XCTAssertTrue(d.isNextNLayerDSA(0))
        XCTAssertEqual(d.activationKind, .situ(beta: 4.0, linearBeta: 25.0))
        XCTAssertEqual(d.moeLatentSize, 2048)
        XCTAssertTrue(d.latentMoeUseNorm)
        XCTAssertTrue(d.moeSplitFeatureRouting)
        XCTAssertTrue(d.routedScalingFactorLearnable)
        XCTAssertTrue(d.multimaxHead)
        XCTAssertTrue(d.mtpSharedLastLayer)
        XCTAssertEqual(d.mhcConfig, QWModelDescriptor.MHCConfig(start: 0, end: 43, streams: 4, iterations: 20, hcEps: 1e-6, hasExitHead: true))

        let geo = QWModelGeometry(descriptor: d)
        XCTAssertEqual(geo.moeRoutedDim, 2048)
        XCTAssertEqual(geo.hcaLayerCount, 37)
        XCTAssertEqual(geo.mlaLayerCount, 6)
        XCTAssertEqual(geo.gdnLayerCount, 0)
        XCTAssertEqual(geo.indexerPoolShift, 0)
        XCTAssertEqual(geo.hcaPoolShift, 7)
        XCTAssertTrue(geo.usesDecoupledRopeMLA)
        XCTAssertTrue(geo.usesLatentOrCompressedAttention)
        XCTAssertTrue(geo.supportsGroupedNAXMoEPrefill)
        XCTAssertEqual(d.lsaKeyProjectionSpec(layer: 0)?.kind, .hcaCompressWKV)
        XCTAssertEqual(d.lsaKeyProjectionSpec(layer: 0)?.sliceDim, 448)
        XCTAssertEqual(d.lsaInputDim, 448)
        XCTAssertEqual(d.lsaHeadSliceStart, 0)
        XCTAssertEqual(d.lsaHeadSliceDim, 448)
        XCTAssertNil(d.lsaKeyProjectionSpec(layer: 7))

        // Pin Python emit_pack_identity.py == Swift geometryDigestV2Hex byte-for-byte.
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let py = repoRoot.appendingPathComponent(".venv-oracle/bin/python")
        let script = repoRoot.appendingPathComponent("Scripts/oracle/emit_pack_identity.py")
        if FileManager.default.isExecutableFile(atPath: py.path) {
            let tmpCfg = FileManager.default.temporaryDirectory.appendingPathComponent("ernie55_cfg_\(UUID().uuidString).json")
            let tmpOut = FileManager.default.temporaryDirectory.appendingPathComponent("ernie55_id_\(UUID().uuidString).json")
            defer {
                try? FileManager.default.removeItem(at: tmpCfg)
                try? FileManager.default.removeItem(at: tmpOut)
            }
            try ernieConfig.write(to: tmpCfg)
            let proc = Process()
            proc.executableURL = py
            proc.arguments = [script.path, tmpCfg.path, tmpOut.path]
            proc.standardOutput = Pipe()
            proc.standardError = Pipe()
            try proc.run()
            proc.waitUntilExit()
            XCTAssertEqual(proc.terminationStatus, 0)
            let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: tmpOut)) as? [String: Any])
            XCTAssertEqual(obj["_canonical_string"] as? String, d.geometryDigestV2)
            XCTAssertEqual(obj["geometry_digest"] as? String, d.geometryDigestV2Hex)
            XCTAssertEqual(obj["model_id"] as? String, "ernie5_5_moe")
        }
    }

}
