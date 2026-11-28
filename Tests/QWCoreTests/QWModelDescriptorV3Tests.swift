// QWModelDescriptorV3Tests.swift — Tests for geometryDigestV3 sensitivity,
// speculative drafter role decoding, and live ~/.qwg/models validation.
import XCTest
import Foundation
@testable import QWCore

final class QWModelDescriptorV3Tests: XCTestCase {

    func testGeometryDigestV3Sensitivity() throws {
        let base = QWModelDescriptor.qwen38_27b
        XCTAssertNotEqual(base.geometryDigestV3Hex, base.geometryDigestV2Hex)

        let retypedFamily = QWModelDescriptor(
            family: .senseNovaLoop, modelType: base.modelType, capabilities: base.capabilities,
            hidden: base.hidden, vocab: base.vocab, denseIntermediate: base.denseIntermediate,
            numLayers: base.numLayers, layerTypes: base.layerTypes, fullAttentionInterval: base.fullAttentionInterval,
            rmsEps: base.rmsEps, maxPositionEmbeddings: base.maxPositionEmbeddings,
            qHeads: base.qHeads, kvHeads: base.kvHeads, headDim: base.headDim,
            attnOutputGate: base.attnOutputGate, partialRotaryFactor: base.partialRotaryFactor, ropeTheta: base.ropeTheta,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil, indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: nil, expertsPerTok: nil, moeIntermediate: nil, sharedExpertIntermediate: nil,
            expertGateType: nil, hcCount: nil, hcLowrank: nil,
            gdnQKHeads: base.gdnQKHeads, gdnVHeads: base.gdnVHeads, gdnHeadDim: base.gdnHeadDim,
            gdnQKVChannels: base.gdnQKVChannels, gdnZChannels: base.gdnZChannels, gdnConvKernel: base.gdnConvKernel,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil, ngramSize: nil, headsPerNgram: nil,
            ngramVocabSizeBase: nil, makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: base.mtpLayers, mtpUseDedicatedEmbeddings: base.mtpUseDedicatedEmbeddings, eosTokenId: base.eosTokenId,
            visPosEmbeds: 1024, gdnVectorDecay: true, role: base.role)
        XCTAssertEqual(base.geometryDigestV2Hex, retypedFamily.geometryDigestV2Hex)
        XCTAssertNotEqual(base.geometryDigestV3Hex, retypedFamily.geometryDigestV3Hex)

        let withMLAAndVision = QWModelDescriptor(
            family: .qwen38Dense, modelType: base.modelType, capabilities: base.capabilities,
            hidden: base.hidden, vocab: base.vocab, denseIntermediate: base.denseIntermediate,
            numLayers: base.numLayers, layerTypes: base.layerTypes, fullAttentionInterval: base.fullAttentionInterval,
            rmsEps: base.rmsEps, maxPositionEmbeddings: base.maxPositionEmbeddings,
            qHeads: base.qHeads, kvHeads: base.kvHeads, headDim: base.headDim,
            attnOutputGate: base.attnOutputGate, partialRotaryFactor: base.partialRotaryFactor, ropeTheta: base.ropeTheta,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil, indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: nil, expertsPerTok: nil, moeIntermediate: nil, sharedExpertIntermediate: nil,
            expertGateType: nil, hcCount: nil, hcLowrank: nil,
            gdnQKHeads: base.gdnQKHeads, gdnVHeads: base.gdnVHeads, gdnHeadDim: base.gdnHeadDim,
            gdnQKVChannels: base.gdnQKVChannels, gdnZChannels: base.gdnZChannels, gdnConvKernel: base.gdnConvKernel,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil, ngramSize: nil, headsPerNgram: nil,
            ngramVocabSizeBase: nil, makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: base.mtpLayers, mtpUseDedicatedEmbeddings: base.mtpUseDedicatedEmbeddings, eosTokenId: base.eosTokenId,
            visDepth: 24,
            loopSettings: .init(start: 2, end: 8, evalCount: 2),
            mhcConfig: .init(start: 0, end: 4, streams: 4, iterations: 20),
            kvLoraRank: 512)
        XCTAssertNotEqual(base.geometryDigestV3Hex, withMLAAndVision.geometryDigestV3Hex)
    }

    func testDecodeDrafterAssignsSpeculativeDrafterRole() throws {
        let drafterJSON = """
        {
          "architectures": ["DFlashDraftModel"],
          "model_type": "qwen3",
          "hidden_size": 1024,
          "vocab_size": 151936,
          "num_hidden_layers": 5,
          "num_attention_heads": 16,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "intermediate_size": 3072,
          "rms_norm_eps": 1e-6,
          "dflash_config": {"mask_token_id": 1},
          "eos_token_id": 151645
        }
        """.data(using: .utf8)!
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: drafterJSON)) { err in
            guard case QWModelDescriptor.DescriptorError.drafterModel = err else {
                return XCTFail("expected drafterModel error, got \(err)")
            }
        }
        let drafter = try QWModelDescriptor.decodeDrafter(configJSON: drafterJSON)
        XCTAssertEqual(drafter.role, .speculativeDrafter)
        XCTAssertEqual(drafter.family, .qwen3Dense)
        XCTAssertEqual(drafter.visDepth, 0)
        XCTAssertEqual(drafter.maxSpeculativeDrafts, 0)
    }

    func testLiveInstalledConfigsDecodeUniqueFamilyAndRole() throws {
        let inCI = ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true"
        try XCTSkipIf(inCI, "requires bare-metal ~/.qwg/models")
        let modelsDir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".qwg/models")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: modelsDir.path), "~/.qwg/models absent")
        let entries = try FileManager.default.contentsOfDirectory(at: modelsDir, includingPropertiesForKeys: [.isDirectoryKey])
        var checked = 0
        for dir in entries where (try? dir.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
            let assetsCfg = dir.appendingPathComponent("assets/config.json")
            let rootCfg = dir.appendingPathComponent("config.json")
            let cfgURL = FileManager.default.fileExists(atPath: assetsCfg.path) ? assetsCfg : rootCfg
            guard FileManager.default.fileExists(atPath: cfgURL.path) else { continue }
            let data = try Data(contentsOf: cfgURL)
            // Skip packs from unmerged feature branches whose model_type is not in this build's registry.
            if Self.hasUnregisteredModelType(in: data) { continue }
            let name = dir.lastPathComponent.lowercased()
            if name.contains("dflash") || name.contains("dspine") {
                let d = try QWModelDescriptor.decodeDrafter(configJSON: data)
                XCTAssertEqual(d.role, .speculativeDrafter, dir.lastPathComponent)
                XCTAssertEqual(d.verifyInternalConsistency(), [], dir.lastPathComponent)
            } else {
                let d = try QWModelDescriptor.decode(configJSON: data)
                XCTAssertNotEqual(d.role, .speculativeDrafter, dir.lastPathComponent)
                XCTAssertEqual(d.verifyInternalConsistency(), [], dir.lastPathComponent)
            }
            checked += 1
        }
        XCTAssertGreaterThan(checked, 0)
    }

    func testSenseNovaLoopFinalRopeParametersYaRNDecode() throws {
        let loopFinalJSON = """
        {
          "architectures": ["SenseNovaLoopForConditionalGeneration"],
          "model_type": "sensenova_loop",
          "model_name": "SenseNova-Loop-9B",
          "image_token_id": 248056,
          "video_token_id": 248057,
          "vision_start_token_id": 248053,
          "vision_end_token_id": 248054,
          "loop_settings": { "start": 4, "end": 28, "eval_count": 2, "counts": [2] },
          "mhc": { "start": 4, "end": 28, "streams": 2, "iterations": 20, "fixed_identity": false, "pre_scale": 1.0 },
          "vision_config": {
            "depth": 27, "hidden_size": 1152, "intermediate_size": 4304, "num_heads": 16,
            "num_position_embeddings": 2304, "out_hidden_size": 4096, "patch_size": 16,
            "spatial_merge_size": 2, "temporal_patch_size": 2
          },
          "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 4096, "intermediate_size": 12288, "vocab_size": 248320,
            "num_hidden_layers": 32, "full_attention_interval": 4,
            "layer_types": [
              "linear_attention", "linear_attention", "linear_attention", "full_attention",
              "linear_attention", "linear_attention", "linear_attention", "full_attention",
              "linear_attention", "linear_attention", "linear_attention", "full_attention",
              "linear_attention", "linear_attention", "linear_attention", "full_attention",
              "linear_attention", "linear_attention", "linear_attention", "full_attention",
              "linear_attention", "linear_attention", "linear_attention", "full_attention",
              "linear_attention", "linear_attention", "linear_attention", "full_attention",
              "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ],
            "rms_norm_eps": 1e-6, "max_position_embeddings": 524288,
            "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
            "attn_output_gate": true, "partial_rotary_factor": 0.25,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_conv_kernel_dim": 4,
            "mtp_num_hidden_layers": 0, "eos_token_id": 248044,
            "rope_parameters": {
              "mrope_interleaved": true, "mrope_section": [11, 11, 10],
              "partial_rotary_factor": 0.25, "rope_theta": 10000000,
              "rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 262144
            }
          }
        }
        """.data(using: .utf8)!

        let d = try QWModelDescriptor.decode(configJSON: loopFinalJSON)
        XCTAssertEqual(d.verifyInternalConsistency(), [])
        XCTAssertEqual(d.family, .senseNovaLoop)
        XCTAssertEqual(d.modelType, "qwen3_5_text")
        XCTAssertEqual(d.envelopeModelType, "sensenova_loop")
        XCTAssertEqual(d.maxPositionEmbeddings, 524_288)
        XCTAssertEqual(d.mtpLayers, 0)
        XCTAssertFalse(d.capabilities.contains(.mtp))
        XCTAssertEqual(d.executionPlan.count, 56)
        XCTAssertEqual(d.yarnRopeConfig?.factor, 2.0)
        XCTAssertEqual(d.yarnRopeConfig?.originalMaxPositionEmbeddings, 262_144)

        let geom = QWModelGeometry(descriptor: d)
        XCTAssertEqual(geom.originalMaxPositionEmbeddings, 262_144)
        XCTAssertEqual(geom.yarnRopeConfig?.factor, 2.0)
    }

    private static func hasUnregisteredModelType(in data: Data) -> Bool {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        let rawType = (obj["text_config"] as? [String: Any])?["model_type"] as? String
            ?? obj["model_type"] as? String
        guard let modelType = rawType, !modelType.isEmpty else { return false }
        return QWModelFamilyRegistry.shared.spec(forModelType: modelType) == nil
    }
}
