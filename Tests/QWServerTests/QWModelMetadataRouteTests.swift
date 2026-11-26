// QWModelMetadataRouteTests.swift — Tests for /v1/models, /v1/models/:id, /api/show, and template thinking probes.

import Foundation
import XCTest
@testable import QWCore
@testable import QWEngine
@testable import QWServer

final class QWModelMetadataRouteTests: XCTestCase {
    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-model-meta-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func writeJSON(_ obj: Any, to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
        try data.write(to: url)
    }

    private func makeSyntheticPack(
        root: URL,
        name: String,
        chatTemplate: String? = nil,
        jinjaFile: String? = nil,
        maxNewTokens: Int? = nil,
        maxPositionEmbeddings: Int = 262_144,
        hasVision: Bool = false
    ) throws -> URL {
        let packDir = root.appendingPathComponent(name, isDirectory: true)
        let assetsDir = packDir.appendingPathComponent("assets", isDirectory: true)
        try FileManager.default.createDirectory(at: assetsDir, withIntermediateDirectories: true)
        var config: [String: Any] = [
            "model_type": "qwen3",
            "vocab_size": 151_936,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "max_position_embeddings": maxPositionEmbeddings,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "eos_token_id": 151_645,
        ]
        if hasVision {
            config["vision_config"] = [
                "depth": 4,
                "hidden_size": 1024,
                "intermediate_size": 2048,
                "num_heads": 16,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "num_position_embeddings": 2304,
                "out_hidden_size": 2048,
            ]
        }
        try writeJSON(config, to: assetsDir.appendingPathComponent("config.json"))
        let cfgData = try Data(contentsOf: assetsDir.appendingPathComponent("config.json"))
        _ = try QWModelDescriptor.decode(configJSON: cfgData)
        if let chatTemplate {
            try writeJSON(["chat_template": chatTemplate], to: assetsDir.appendingPathComponent("tokenizer_config.json"))
        }
        if let jinjaFile {
            try jinjaFile.write(to: assetsDir.appendingPathComponent("chat_template.jinja"), atomically: true, encoding: .utf8)
        }
        if let maxNewTokens {
            try writeJSON(["max_new_tokens": maxNewTokens], to: assetsDir.appendingPathComponent("generation_config.json"))
        }
        return packDir
    }

    func testModelObjectRoundTripAndOptionalOmitting() throws {
        let minimal = ModelObject(id: "plain-model", loaded: false, created: nil)
        let minData = try JSONEncoder().encode(minimal)
        let minJSON = try XCTUnwrap(JSONSerialization.jsonObject(with: minData) as? [String: Any])
        XCTAssertEqual(minJSON["id"] as? String, "plain-model")
        XCTAssertEqual(minJSON["owned_by"] as? String, "qwegine")
        XCTAssertEqual(minJSON["loaded"] as? Bool, false)
        XCTAssertNil(minJSON["context_length"])
        XCTAssertNil(minJSON["reasoning"])

        let full = ModelObject(
            id: "vision-thinker",
            loaded: true,
            created: 1_700_000_000,
            contextLength: 262_144,
            maxModelLen: 262_144,
            maxOutputTokens: 65_536,
            maxCompletionTokens: 65_536,
            inputModalities: ["text", "image"],
            outputModalities: ["text"],
            architecture: ModelObject.Architecture(inputModalities: ["text", "image"], outputModalities: ["text"]),
            meta: ModelObject.Meta(nCtx: 262_144, nCtxTrain: 1_048_576),
            capabilities: ["mtp", "vision"],
            reasoning: true,
            supportedReasoningLevels: ["none", "low", "medium", "high", "xhigh"],
            defaultReasoningLevel: "xhigh"
        )
        let fullData = try JSONEncoder().encode(full)
        let decoded = try JSONDecoder().decode(ModelObject.self, from: fullData)
        XCTAssertEqual(decoded, full)
    }

    func testDetectSupportsThinkingTagsAcrossTemplateFormats() throws {
        let tmp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tmp) }

        let qwenPack = try makeSyntheticPack(root: tmp, name: "qwen", chatTemplate: "{{ msg }}<think>\n")
        let erniePack = try makeSyntheticPack(root: tmp, name: "ernie", jinjaFile: "<ernie:think>ok</ernie:think>")
        let inklingPack = try makeSyntheticPack(
            root: tmp,
            name: "inkling",
            chatTemplate: "<|message_model|><|content_thinking|>ok"
        )
        let plainPack = try makeSyntheticPack(root: tmp, name: "plain", chatTemplate: "{{ msg }}")
        let missingPack = tmp.appendingPathComponent("does-not-exist").path

        XCTAssertEqual(QWChatTemplate.detectSupportsThinkingTags(atPackPath: qwenPack.path), true)
        XCTAssertEqual(QWChatTemplate.detectSupportsThinkingTags(atPackPath: erniePack.path), true)
        XCTAssertEqual(QWChatTemplate.detectSupportsThinkingTags(atPackPath: inklingPack.path), true)
        XCTAssertEqual(QWChatTemplate.detectSupportsThinkingTags(atPackPath: plainPack.path), false)
        XCTAssertNil(QWChatTemplate.detectSupportsThinkingTags(atPackPath: missingPack))
    }

    func testModelsEndpointsAndOllamaShowPublishHonestPerModelMetadata() async throws {
        let tmp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tmp) }

        let packVisionThink = try makeSyntheticPack(
            root: tmp,
            name: "Alpha-Vision-Think",
            chatTemplate: "<think></think>",
            maxNewTokens: 65_536,
            maxPositionEmbeddings: 262_144,
            hasVision: true
        )
        let packTextOnly = try makeSyntheticPack(
            root: tmp,
            name: "Beta-Text-1M",
            chatTemplate: "{{ prompt }}",
            maxNewTokens: nil,
            maxPositionEmbeddings: 1_048_576,
            hasVision: false
        )
        let packVisionNoCap = try makeSyntheticPack(
            root: tmp,
            name: "Gamma-Vision-128K",
            chatTemplate: "<ernie:think></ernie:think>",
            maxNewTokens: nil,
            maxPositionEmbeddings: 128_000,
            hasVision: true
        )

        let settingsURL = tmp.appendingPathComponent("settings.json")
        var settings = QWServerSettings(
            modelPackPath: packVisionThink.path,
            modelName: "Alpha-Vision-Think",
            maxContext: 262_144,
            maxTokens: 65_536,
            reasoningEffort: "high",
            loadVision: true
        )
        settings.setAlias("Alpha-Alias", for: "Alpha-Vision-Think")
        try QWSettingsManager.shared.saveSettings(settings, to: settingsURL)

        let profileStore = QWSettingsManager.shared.profileStore(forSettingsURL: settingsURL)
        try profileStore.save(QWKnobProfile(
            modelId: "Beta-Text-1M",
            values: ["contextTokens": .int(524_288), "maxTokens": .int(16_384)]
        ))

        let discovered: [QWServerSettings.DiscoveredModel] = [
            QWServerSettings.DiscoveredModel(
                name: "Alpha-Vision-Think",
                packPath: packVisionThink.path,
                family: "qwen3.8-flash-next",
                nativeWindow: 262_144,
                supported: true,
                capabilities: ["mtp", "vision"],
                hasVision: true
            ),
            QWServerSettings.DiscoveredModel(
                name: "Beta-Text-1M",
                packPath: packTextOnly.path,
                family: "ernie5_5_moe",
                nativeWindow: 1_048_576,
                supported: true,
                capabilities: ["moe"],
                hasVision: false
            ),
            QWServerSettings.DiscoveredModel(
                name: "Gamma-Vision-128K",
                packPath: packVisionNoCap.path,
                family: "lfm3",
                nativeWindow: 128_000,
                supported: true,
                capabilities: ["decision", "vision"],
                isDecisionCapable: true,
                hasVision: true
            ),
        ]

        let port = 19096
        let config = QWEngineLifecycleActor.Config(
            packPath: packVisionThink.path,
            modelName: "Alpha-Vision-Think",
            maxContext: 262_144,
            maxTokens: 65_536,
            reasoningEffort: "high",
            loadVision: true,
            sourceSummary: ["maxTokens": "config-file"]
        )
        let logger = QWServerLogger(level: .none)
        let actor = QWEngineLifecycleActor(config: config, logger: logger, configPath: settingsURL.path)
        let server = QWHttpServer(
            actor: actor,
            host: "127.0.0.1",
            port: port,
            logger: logger,
            finderRevealRunner: nil,
            modelCatalogScanner: { discovered },
            settingsFileURL: settingsURL
        )

        try server.start()
        defer { server.stop() }
        try await Task.sleep(nanoseconds: 100_000_000)

        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let (listData, listResp) = try await URLSession.shared.data(from: baseURL.appendingPathComponent("v1/models"))
        XCTAssertEqual((listResp as? HTTPURLResponse)?.statusCode, 200)
        let listJSON = try XCTUnwrap(JSONSerialization.jsonObject(with: listData) as? [String: Any])
        let items = try XCTUnwrap(listJSON["data"] as? [[String: Any]])

        func entry(_ id: String) throws -> [String: Any] {
            try XCTUnwrap(items.first { ($0["id"] as? String) == id }, "Missing model '\(id)'")
        }

        let alpha = try entry("Alpha-Alias")
        XCTAssertEqual(alpha["context_length"] as? Int, 262_144)
        XCTAssertEqual(alpha["max_model_len"] as? Int, 262_144)
        XCTAssertEqual(alpha["max_output_tokens"] as? Int, 65_536)
        XCTAssertEqual(alpha["max_completion_tokens"] as? Int, 65_536)
        XCTAssertEqual(alpha["input_modalities"] as? [String], ["text", "image"])
        XCTAssertEqual((alpha["architecture"] as? [String: Any])?["input_modalities"] as? [String], ["text", "image"])
        XCTAssertEqual(alpha["reasoning"] as? Bool, true)
        XCTAssertEqual(alpha["default_reasoning_level"] as? String, "xhigh")
        XCTAssertEqual(alpha["supported_reasoning_levels"] as? [String], ["none", "low", "medium", "high", "xhigh"])

        let beta = try entry("Beta-Text-1M")
        XCTAssertEqual(beta["context_length"] as? Int, 524_288)
        XCTAssertEqual(beta["max_output_tokens"] as? Int, 16_384)
        XCTAssertEqual((beta["meta"] as? [String: Any])?["n_ctx"] as? Int, 524_288)
        XCTAssertEqual((beta["meta"] as? [String: Any])?["n_ctx_train"] as? Int, 1_048_576)
        XCTAssertEqual(beta["input_modalities"] as? [String], ["text"])
        XCTAssertEqual(beta["reasoning"] as? Bool, false)
        XCTAssertNil(beta["supported_reasoning_levels"])

        let gamma = try entry("Gamma-Vision-128K")
        XCTAssertEqual(gamma["context_length"] as? Int, 128_000)
        XCTAssertEqual(gamma["max_output_tokens"] as? Int, 128_000)
        XCTAssertEqual(gamma["input_modalities"] as? [String], ["text", "image"])
        XCTAssertEqual(gamma["reasoning"] as? Bool, true)

        let (singleData, singleResp) = try await URLSession.shared.data(
            from: baseURL.appendingPathComponent("v1/models/Beta-Text-1M")
        )
        XCTAssertEqual((singleResp as? HTTPURLResponse)?.statusCode, 200)
        let singleJSON = try XCTUnwrap(JSONSerialization.jsonObject(with: singleData) as? [String: Any])
        XCTAssertEqual(singleJSON["context_length"] as? Int, 524_288)
        XCTAssertEqual(singleJSON["max_output_tokens"] as? Int, 16_384)

        var showReq = URLRequest(url: baseURL.appendingPathComponent("api/show"))
        showReq.httpMethod = "POST"
        showReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        showReq.httpBody = try JSONSerialization.data(withJSONObject: ["model": "Alpha-Vision-Think"])
        let (showData, showResp) = try await URLSession.shared.data(for: showReq)
        XCTAssertEqual((showResp as? HTTPURLResponse)?.statusCode, 200)
        let showJSON = try XCTUnwrap(JSONSerialization.jsonObject(with: showData) as? [String: Any])
        XCTAssertEqual(showJSON["capabilities"] as? [String], ["completion", "vision", "thinking"])
        XCTAssertEqual((showJSON["model_info"] as? [String: Any])?["context_length"] as? Int, 262_144)
        XCTAssertEqual(showJSON["parameters"] as? String, "num_ctx 262144")
    }
}
