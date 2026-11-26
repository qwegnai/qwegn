// QWDecisionConfigTests.swift — Tests for LFM2/LFM3/LFM2-Encoder/ModernBERT decoding, .aimodel discovery, UMA sidecar admission, and GPU preemption.
import XCTest
import Foundation
@testable import QWCore

final class QWDecisionConfigTests: XCTestCase {
    func testDecodesRealD1_2BConfig() throws {
        let json = """
        {
          "architectures": ["D1Model"],
          "bos_token_id": 124894,
          "eos_token_id": 124900,
          "image_token_id": 125030,
          "model_type": "lfm3_vl",
          "pad_token_id": 124893,
          "text_config": {
            "bos_token_id": 124894,
            "conv_L_cache": 7,
            "conv_bias": false,
            "eos_token_id": 124900,
            "head_dim": 128,
            "hidden_size": 2048,
            "intermediate_size": 7680,
            "layer_types": ["conv", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv"],
            "max_position_embeddings": 128000,
            "model_type": "lfm3",
            "norm_eps": 1e-5,
            "num_attention_heads": 16,
            "num_hidden_layers": 8,
            "num_key_value_heads": 4,
            "rope_parameters": {
              "full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1000000.0, "rope_type": "default"},
              "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
            },
            "sliding_window": 256,
            "tie_word_embeddings": true,
            "use_sandwich_norm": true,
            "vocab_size": 128000
          },
          "tie_word_embeddings": true,
          "video_token_id": 125029,
          "vision_config": {
            "conv_kernel_size": 3,
            "hidden_size": 1024,
            "intermediate_size": 2560,
            "model_type": "lfm3_vl_vision",
            "norm_eps": 1e-5,
            "num_attention_heads": 16,
            "num_hidden_layers": 14,
            "patch_size": 16,
            "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "axial"},
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
          },
          "projector_bias": true,
          "projector_hidden_size": 4096,
          "projector_layer_norm_eps": 1e-6,
          "d1_codes": 2,
          "d1_attention": "set",
          "d1_numbers": true,
          "d1_set_size": 16,
          "d1_vision_max_pixels": 1638400
        }
        """
        let desc = try QWModelDescriptor.decode(configJSON: Data(json.utf8))
        XCTAssertEqual(desc.family, .lfm3)
        XCTAssertEqual(desc.role, .multimodalConditionalLM)
        XCTAssertTrue(desc.capabilities.contains(.vision))
        XCTAssertTrue(desc.capabilities.contains(.decision))
        XCTAssertTrue(desc.attnOutputGate)
        XCTAssertEqual(desc.attnGateMode, .separateHeadWiseSigmoid)
        XCTAssertTrue(desc.hasQKNorm)
        XCTAssertFalse(desc.hasMropeSection)
        XCTAssertEqual(desc.shortConvConfig, ShortConvConfig(kernelSize: 7, hasBias: false, isCentered: false, stateKey: "conv"))
        XCTAssertTrue(desc.useSandwichNorm)
        XCTAssertEqual(desc.normKind, .rmsNorm(eps: 1e-5, hasUnitOffset: false))
        XCTAssertEqual(desc.perKindRoPEConfig?.entry(forLayerKind: "full_attention")?.partialRotaryFactor, 0.5)
        XCTAssertEqual(desc.perKindRoPEConfig?.entry(forLayerKind: "sliding_attention")?.ropeTheta, 10000.0)
        XCTAssertEqual(desc.perKindRoPEConfig?.zeroTailPartialRotation, true)
        XCTAssertEqual(desc.layerSlidingWindow(4), 256)
        XCTAssertEqual(desc.layerSlidingWindow(2), 0)
        XCTAssertEqual(desc.visDepth, 14)
        XCTAssertEqual(desc.visHeads, 16)
        XCTAssertEqual(desc.visTemporal, 2)
        XCTAssertEqual(desc.visPosEmbeds, 0)
        XCTAssertEqual(desc.visionTowerConfig?.mergerIntermediateDim, 4096)
        XCTAssertEqual(desc.visionTowerConfig?.projectorNormEps ?? 0, 1e-6, accuracy: 1e-9)

        guard let dc = desc.decisionConfig else { return XCTFail("expected decisionConfig") }
        XCTAssertEqual(dc.topology, .packedSet(maxSetSize: 16))
        XCTAssertEqual(dc.visionMaxPixels, 1638400)
        XCTAssertTrue(dc.hasStatePointer)
        XCTAssertTrue(dc.hasGridReadout)
        XCTAssertTrue(dc.usesNumbersThousandthsGrid)
        if case .turnVerbalizer(let codes, let numbers, let packSet, let setSize, _, _, let lead, let chatML) = dc.framing {
            XCTAssertEqual(codes, 2)
            XCTAssertTrue(numbers)
            XCTAssertTrue(packSet)
            XCTAssertEqual(setSize, 16)
            XCTAssertEqual(lead, "<think></think>")
            XCTAssertTrue(chatML)
        } else {
            XCTFail("expected turnVerbalizer framing")
        }

        // Pin Python emit_pack_identity.py == Swift geometryDigestV2Hex byte-for-byte.
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let script = repoRoot.appendingPathComponent("Scripts/oracle/emit_pack_identity.py")
        if FileManager.default.fileExists(atPath: script.path) {
            let tmpCfg = FileManager.default.temporaryDirectory.appendingPathComponent("d1_2b_cfg_\(UUID().uuidString).json")
            let tmpOut = FileManager.default.temporaryDirectory.appendingPathComponent("d1_2b_id_\(UUID().uuidString).json")
            defer {
                try? FileManager.default.removeItem(at: tmpCfg)
                try? FileManager.default.removeItem(at: tmpOut)
            }
            try Data(json.utf8).write(to: tmpCfg)
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
            proc.arguments = [script.path, tmpCfg.path, tmpOut.path]
            let errPipe = Pipe()
            proc.standardOutput = Pipe()
            proc.standardError = errPipe
            try proc.run()
            proc.waitUntilExit()
            let errText = String(data: errPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            XCTAssertEqual(proc.terminationStatus, 0, "emit_pack_identity stderr: \(errText)")
            if proc.terminationStatus == 0 {
                let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: tmpOut)) as? [String: Any])
                XCTAssertEqual(obj["_canonical_string"] as? String, desc.geometryDigestV2)
                XCTAssertEqual(obj["geometry_digest"] as? String, desc.geometryDigestV2Hex)
                XCTAssertEqual(obj["model_id"] as? String, "lfm3")
            }
        }
    }

    func testDecodesRealD1_3BConfigAndLFM2AutoAdjustFF() throws {
        let json = """
        {
          "architectures": ["Lfm2VlForConditionalGeneration"],
          "auto_map": {"AutoModel": "modeling_d1.D1Model"},
          "eos_token_id": 124900,
          "image_token_id": 124907,
          "model_type": "lfm2_vl",
          "text_config": {
            "architectures": ["Lfm2ForCausalLM"],
            "block_auto_adjust_ff_dim": false,
            "block_multiple_of": 256,
            "conv_L_cache": 3,
            "conv_bias": false,
            "eos_token_id": 124900,
            "hidden_size": 2048,
            "intermediate_size": 10752,
            "layer_types": ["conv", "conv", "full_attention", "conv"],
            "max_position_embeddings": 32768,
            "model_type": "lfm2",
            "norm_eps": 1e-5,
            "num_attention_heads": 32,
            "num_hidden_layers": 4,
            "num_key_value_heads": 8,
            "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
            "tie_word_embeddings": true,
            "vocab_size": 128000
          },
          "vision_config": {
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "model_type": "siglip2_vision_model",
            "num_attention_heads": 16,
            "num_hidden_layers": 27,
            "num_patches": 256,
            "patch_size": 16
          }
        }
        """
        let desc = try QWModelDescriptor.decode(configJSON: Data(json.utf8))
        XCTAssertEqual(desc.family, .lfm2)
        XCTAssertEqual(desc.headDim, 64)
        XCTAssertEqual(desc.denseIntermediate, 10752)
        XCTAssertEqual(desc.visDepth, 27)
        XCTAssertEqual(desc.visPosEmbeds, 256)
        XCTAssertTrue(desc.capabilities.contains(.decision))
        XCTAssertEqual(desc.decisionConfig?.topology, .causal(supportsPrefixCache: true))

        // Verify block_auto_adjust_ff_dim: true adjustment (2 * 3072 / 3 = 2048).
        let adjustedJSON = json.replacingOccurrences(of: "\"block_auto_adjust_ff_dim\": false", with: "\"block_auto_adjust_ff_dim\": true")
            .replacingOccurrences(of: "\"intermediate_size\": 10752", with: "\"intermediate_size\": 3072")
        let adjusted = try QWModelDescriptor.decode(configJSON: Data(adjustedJSON.utf8))
        XCTAssertEqual(adjusted.denseIntermediate, 2048)
    }

    func testDecodesRealD1Omni600MEncoderConfig() throws {
        let json = """
        {
          "architectures": ["D1OmniModel"],
          "model_type": "d1_omni",
          "max_length": 16384,
          "image_text_length": 896,
          "audio_text_length": 15360,
          "head_layers": 2,
          "temperatures": {"choice:3-5": 1.3998981714248657, "noul:2": 1.666},
          "bos_token_id": 1,
          "pad_token_id": 0,
          "text_config": {
            "vocab_size": 65536,
            "hidden_size": 1024,
            "intermediate_size": 6656,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "layer_types": ["conv", "conv", "full_attention", "conv"],
            "norm_eps": 1e-5,
            "conv_L_cache": 3,
            "max_position_embeddings": 128000,
            "rope_theta": 1000000.0
          },
          "vision_config": {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "model_type": "siglip2_vision_model",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "num_patches": 256,
            "patch_size": 16
          }
        }
        """
        let desc = try QWModelDescriptor.decode(configJSON: Data(json.utf8))
        XCTAssertEqual(desc.family, .lfm2Encoder)
        XCTAssertEqual(desc.role, .decisionEncoder)
        XCTAssertEqual(desc.headDim, 64)
        XCTAssertEqual(desc.shortConvConfig, ShortConvConfig(kernelSize: 3, hasBias: false, isCentered: true, stateKey: "conv"))
        guard let dc = desc.decisionConfig else { return XCTFail("expected decisionConfig") }
        XCTAssertEqual(dc.topology, .bidirectionalEncoder(isolateMediaPrefix: true))
        XCTAssertEqual(dc.inlineTemperatures["choice:3-5"], 1.3998981714248657)
        if case .optionMarkers(let layout, _, _, let maxLen, let imgLen, let audLen) = dc.framing {
            XCTAssertEqual(layout, .stateQuestionOptions)
            XCTAssertEqual(maxLen, 16384)
            XCTAssertEqual(imgLen, 896)
            XCTAssertEqual(audLen, 15360)
        } else {
            XCTFail("expected optionMarkers framing")
        }
    }

    func testDecodesModernBERTSequenceClassification() throws {
        let json = """
        {
          "architectures": ["ModernBertForSequenceClassification"],
          "model_type": "modernbert",
          "hidden_size": 768,
          "vocab_size": 50368,
          "num_hidden_layers": 6,
          "num_attention_heads": 12,
          "intermediate_size": 1152,
          "max_position_embeddings": 8192,
          "norm_eps": 1e-5,
          "norm_bias": false,
          "global_rope_theta": 160000.0,
          "local_rope_theta": 10000.0,
          "local_attention": 128,
          "global_attn_every_n_layers": 3,
          "classifier_pooling": "mean",
          "id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"}
        }
        """
        let desc = try QWModelDescriptor.decode(configJSON: Data(json.utf8))
        XCTAssertEqual(desc.family, .modernBert)
        XCTAssertEqual(desc.role, .decisionEncoder)
        XCTAssertEqual(desc.normKind, .layerNorm(eps: 1e-5, hasBias: false))
        XCTAssertEqual(desc.layerTypes, ["full_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention"])
        XCTAssertEqual(desc.layerSlidingWindow(0), 0)
        XCTAssertEqual(desc.layerSlidingWindow(1), 128)
        XCTAssertEqual(desc.perKindRoPEConfig?.entry(forLayerKind: "full_attention")?.ropeTheta, 160000.0)
        XCTAssertEqual(desc.perKindRoPEConfig?.entry(forLayerKind: "sliding_attention")?.ropeTheta, 10000.0)
        if case .pooledClassifier(let pool, let n, let map) = desc.decisionConfig?.head {
            XCTAssertEqual(pool, .meanNonPad)
            XCTAssertEqual(n, 3)
            XCTAssertEqual(map[1], "neutral")
        } else {
            XCTFail("expected pooledClassifier head")
        }
    }

    func testExistingDigestPreservationAndWithDecisionSensitivity() {
        let base = QWModelDescriptor.qwen38_27b
        let modified = base.withDecisionConfig(QWDecisionModelConfig.synthesizedVerbalizer(vocabSize: base.vocab))
        XCTAssertEqual(base.geometryDigestV2Hex, modified.geometryDigestV2Hex)
        XCTAssertNotEqual(base.geometryDigestV3Hex, modified.geometryDigestV3Hex)
    }

    func testCoreAIBundleDiscoveryAndSidecarAdmission() throws {
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory.appendingPathComponent("qw_coreai_test_\(UUID().uuidString)")
        let bundle = tmp.appendingPathComponent("qwegn-edge-ane")
        let aimodel = bundle.appendingPathComponent("rl_agent.aimodel")
        try fm.createDirectory(at: aimodel, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tmp) }
        try "{\"sequence_length\": 1024}".write(to: bundle.appendingPathComponent("combined_provenance.json"), atomically: true, encoding: .utf8)

        let root = QWModelRoot(url: bundle)
        XCTAssertEqual(root.kind, .coreAIModel)
        XCTAssertEqual(QWModelRoot.canonicalModelName(for: aimodel), "qwegn-edge-ane")

        let entries = QWModelCatalog.scan(directory: tmp.path)
        XCTAssertEqual(entries.count, 1)
        XCTAssertTrue(entries[0].supported)
        XCTAssertTrue(entries[0].isCoreAIBundle)
        XCTAssertTrue(entries[0].isDecisionCapable)
        XCTAssertEqual(entries[0].nativeWindow, 1024)

        let snap128 = QWDeviceMemorySnapshot(physicalBytes: 128 << 30, reclaimableBytes: 90 << 30)
        let ok = QWResidencyAdmission.admitDecisionSidecar(
            primaryWiredBytes: 70 << 30,
            sidecarEstimateBytes: 4 << 30,
            requestedWarmCacheBytes: 16 << 30,
            requestedSessions: 4,
            osReserveGiB: 16.0,
            snapshot: snap128
        )
        XCTAssertTrue(ok.admitted)
        XCTAssertGreaterThan(ok.warmCache.admittedWarmBytes, 0)

        let refused = QWResidencyAdmission.admitDecisionSidecar(
            primaryWiredBytes: 110 << 30,
            sidecarEstimateBytes: 8 << 30,
            requestedWarmCacheBytes: 4 << 30,
            requestedSessions: 2,
            osReserveGiB: 16.0,
            snapshot: snap128
        )
        XCTAssertFalse(refused.admitted)
        XCTAssertEqual(refused.warmCache.admittedWarmBytes, 0)
    }

    func testGPUTurnArbitratorStepPreemption() async {
        let arb = QWGPUTurnArbitrator()
        await arb.acquirePrimaryStep()
        XCTAssertEqual(arb.activePrimarySteps, 1)

        let events = EventBox()
        let sidecarTask = Task {
            await arb.runHighPrioritySidecar {
                await events.append("sidecar-run")
            }
        }
        try? await Task.sleep(for: .milliseconds(15))
        await events.append("primary-release")
        arb.releasePrimaryStep()
        await sidecarTask.value

        let primary2 = Task {
            await arb.withPrimaryStep {
                await events.append("primary-2")
            }
        }
        await primary2.value
        let log = await events.snapshot()
        XCTAssertEqual(log, ["primary-release", "sidecar-run", "primary-2"])
        XCTAssertGreaterThanOrEqual(arb.totalPreemptionsServed, 1)
    }
}

private actor EventBox {
    private var log: [String] = []
    func append(_ s: String) { log.append(s) }
    func snapshot() -> [String] { log }
}
