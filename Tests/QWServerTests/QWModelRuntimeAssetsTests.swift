import Foundation
import XCTest
import QWCore
import QWEngine
import QWHAL
@testable import QWServer
import QWToken
import QWVision

final class QWModelRuntimeAssetsTests: XCTestCase {
    private func makeModelDirectory(
        _ generationConfig: String? = nil,
        template: String? = "{% for message in messages %}{{ message.content }}{% endfor %}",
        tokenizerConfig: String? = nil
    ) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let assets = root.appendingPathComponent("assets", isDirectory: true)
        try FileManager.default.createDirectory(at: assets, withIntermediateDirectories: true)
        try Data().write(to: root.appendingPathComponent("model.qwp"))
        try Data("{}".utf8).write(to: assets.appendingPathComponent("tokenizer.json"))
        if let generationConfig {
            try Data(generationConfig.utf8)
                .write(to: assets.appendingPathComponent("generation_config.json"))
        }
        if let template {
            try Data(template.utf8).write(to: assets.appendingPathComponent("chat_template.jinja"))
        }
        if let tokenizerConfig {
            try Data(tokenizerConfig.utf8)
                .write(to: assets.appendingPathComponent("tokenizer_config.json"))
        }
        return root
    }

    func testGenerationDefaultsApplyPerFieldBelowServerOverrides() throws {
        let root = try makeModelDirectory("""
        {
          "do_sample": true,
          "temperature": 0.35,
          "top_p": 0.82,
          "top_k": 17,
          "repetition_penalty": 1.12,
          "max_new_tokens": 72
        }
        """)
        defer { try? FileManager.default.removeItem(at: root) }

        let resolved = try QWServerConfigurationResolver.resolve(
            cliConfigPath: root.appendingPathComponent("missing-settings.json").path,
            cliPack: root.appendingPathComponent("model.qwp").path,
            cliTemperature: 0.2,
            environment: ["QWG_TOP_K": "23"])

        XCTAssertEqual(resolved.temperature, 0.2)
        XCTAssertEqual(resolved.topP, 0.82)
        XCTAssertEqual(resolved.topK, 23)
        XCTAssertEqual(resolved.repetitionPenalty, 1.12)
        XCTAssertEqual(resolved.maxTokens, 72)
        XCTAssertNil(resolved.modelMaxLength)
        XCTAssertEqual(resolved.sourceSummary["temperature"], "cli")
        XCTAssertEqual(resolved.sourceSummary["topP"], "model-config")
        XCTAssertEqual(resolved.sourceSummary["topK"], "env")
    }

    func testHFMaxLengthIsPreservedForPromptRelativeResolution() throws {
        let root = try makeModelDirectory(#"{"max_length": 2048}"#)
        defer { try? FileManager.default.removeItem(at: root) }

        let resolved = try QWServerConfigurationResolver.resolve(
            cliConfigPath: root.appendingPathComponent("missing-settings.json").path,
            cliPack: root.appendingPathComponent("model.qwp").path,
            environment: [:])

        XCTAssertEqual(resolved.modelMaxLength, 2048)
        XCTAssertEqual(resolved.maxTokens, 512)
    }

    func testOnlyExplicitSettingsKeysOverrideModelDefaults() throws {
        let root = try makeModelDirectory(#"{"temperature":0.4,"top_p":0.82,"top_k":17}"#)
        defer { try? FileManager.default.removeItem(at: root) }
        let settingsURL = root.appendingPathComponent("settings.json")
        try Data(#"{"topP":0.61}"#.utf8).write(to: settingsURL)

        let resolved = try QWServerConfigurationResolver.resolve(
            cliConfigPath: settingsURL.path,
            cliPack: root.appendingPathComponent("model.qwp").path,
            environment: [:])

        XCTAssertEqual(resolved.temperature, 0.4)
        XCTAssertEqual(resolved.topP, 0.61)
        XCTAssertEqual(resolved.topK, 17)
        XCTAssertEqual(resolved.sourceSummary["topP"], "config-file")
    }

    func testInvalidSettingsAreRejectedRatherThanUsingDefaults() throws {
        let root = try makeModelDirectory(#"{"temperature":0.4}"#)
        defer { try? FileManager.default.removeItem(at: root) }
        let settingsURL = root.appendingPathComponent("settings.json")
        for contents in [#"{"host":"10.0.0.1","temperature":"invalid"}"#,
                         #"{"host":"10.0.0.1","port":"invalid"}"#, "[]", "{"] {
            try Data(contents.utf8).write(to: settingsURL)
            XCTAssertThrowsError(try QWServerConfigurationResolver.resolve(
                cliConfigPath: settingsURL.path,
                cliPack: root.appendingPathComponent("model.qwp").path,
                environment: [:]), "accepted \(contents)")
        }
        XCTAssertThrowsError(try QWServerConfigurationResolver.resolve(
            cliConfigPath: root.path, cliPack: root.appendingPathComponent("model.qwp").path,
            environment: [:]))
    }

    func testIntegerSettingsAreAcceptedAsDoubles() throws {
        let root = try makeModelDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let settingsURL = root.appendingPathComponent("settings.json")
        try Data(#"{"temperature":1}"#.utf8).write(to: settingsURL)
        let resolved = try QWServerConfigurationResolver.resolve(
            cliConfigPath: settingsURL.path,
            cliPack: root.appendingPathComponent("model.qwp").path,
            environment: [:])
        XCTAssertEqual(resolved.temperature, 1)
        XCTAssertEqual(resolved.sourceSummary["temperature"], "config-file")
    }

    func testDoSampleFalseUsesGreedyAndMaxNewTokensOverridesMaxLength() throws {
        let root = try makeModelDirectory(#"{"do_sample":false,"temperature":0.8,"max_new_tokens":12,"max_length":2048}"#)
        defer { try? FileManager.default.removeItem(at: root) }

        let resolved = try QWServerConfigurationResolver.resolve(
            cliConfigPath: root.appendingPathComponent("missing-settings.json").path,
            cliPack: root.appendingPathComponent("model.qwp").path,
            environment: [:])

        XCTAssertEqual(resolved.temperature, 0)
        XCTAssertEqual(resolved.maxTokens, 12)
        XCTAssertNil(resolved.modelMaxLength)
    }

    func testInvalidGenerationDefaultIsRejected() throws {
        let root = try makeModelDirectory(#"{"top_p":1.5}"#)
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertThrowsError(try QWServerConfigurationResolver.resolve(
            cliConfigPath: root.appendingPathComponent("missing-settings.json").path,
            cliPack: root.appendingPathComponent("model.qwp").path,
            environment: [:]))
    }

    func testNullGenerationDefaultsAreTreatedAsUnset() throws {
        let root = try makeModelDirectory(#"{"temperature":null,"max_new_tokens":null,"max_length":null}"#)
        defer { try? FileManager.default.removeItem(at: root) }

        let resolved = try QWServerConfigurationResolver.resolve(
            cliConfigPath: root.appendingPathComponent("missing-settings.json").path,
            cliPack: root.appendingPathComponent("model.qwp").path,
            environment: [:])

        XCTAssertEqual(resolved.temperature, 0.7)
        XCTAssertEqual(resolved.maxTokens, 512)
        XCTAssertNil(resolved.modelMaxLength)
    }

    func testInvalidBundledTemplateIsRejected() throws {
        let root = try makeModelDirectory(template: "{% if messages %}")
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertThrowsError(try QWServerConfigurationResolver.resolve(
            cliConfigPath: root.appendingPathComponent("missing-settings.json").path,
            cliPack: root.appendingPathComponent("model.qwp").path,
            environment: [:]))
    }

    func testBundledTemplateRendersMessages() throws {
        let root = try makeModelDirectory(
            template: "{% for message in messages %}<|{{ message.role }}|>{{ message.content }}{% endfor %}")
        defer { try? FileManager.default.removeItem(at: root) }

        let assets = try QWModelRuntimeAssets.load(
            packPath: root.appendingPathComponent("model.qwp").path)
        let rendered = try assets.chatTemplate.render(messages: [
            .object([("role", .string("user")), ("content", .string("hello"))])
        ])

        XCTAssertEqual(rendered, "<|user|>hello")
    }

    func testDefaultNamedTokenizerTemplateAndSpecialTokenContext() throws {
        let root = try makeModelDirectory(
            template: nil,
            tokenizerConfig: """
            {
              "bos_token": {"content": "<bos>"},
              "chat_template": [
                {"name": "tool_use", "template": "ignored"},
                {"name": "default", "template": "{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}"}
              ]
            }
            """)
        defer { try? FileManager.default.removeItem(at: root) }

        let assets = try QWModelRuntimeAssets.load(
            packPath: root.appendingPathComponent("model.qwp").path)
        let rendered = try assets.chatTemplate.render(messages: [
            .object([("role", .string("user")), ("content", .string("hello"))])
        ])

        XCTAssertEqual(rendered, "<bos>hello")
    }

    func testAmbiguousNamedTokenizerTemplatesAreRejected() throws {
        let root = try makeModelDirectory(
            template: nil,
            tokenizerConfig: """
            {"chat_template":[
              {"name":"one","template":"one"},
              {"name":"two","template":"two"}
            ]}
            """)
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertThrowsError(try QWModelRuntimeAssets.load(
            packPath: root.appendingPathComponent("model.qwp").path))
    }

    func testConflictingTemplateSourcesAreRejected() throws {
        let root = try makeModelDirectory(
            template: "bundled",
            tokenizerConfig: #"{"chat_template":"tokenizer config"}"#)
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertThrowsError(try QWModelRuntimeAssets.load(
            packPath: root.appendingPathComponent("model.qwp").path))
    }

    func testModelBackendRegistryResolvesRegisteredModuleBuilder() throws {
        let registry = QWModelBackendRegistry(registrations: [])
        XCTAssertTrue(registry.registeredModules.isEmpty)

        registry.register(module: DenseModule.self) { config, descriptor, _ in
            let params = QWEngineParametersInfo(
                modelPackPath: config.packPath,
                modelName: config.modelName,
                modelDescriptor: descriptor,
                headStorage: "HQ4_G32_M5",
                contextBudget: config.maxContext,
                kvMode: config.kvMode,
                mtpDrafts: 0,
                mtpBackend: "standard",
                gdnPrefillMode: config.gdnPrefill,
                prefillScratchEviction: config.evictPrefillScratch
            )
            return StubModelBackend(descriptor: descriptor, parametersInfo: params)
        }
        XCTAssertEqual(registry.registeredModules.count, 1)
        XCTAssertTrue(registry.registeredModules.contains { $0 == DenseModule.self })

        let config = QWEngineLifecycleActor.Config(packPath: "/tmp/stub.qwp", modelName: "stub")
        let backend = try registry.loadBackend(
            config: config,
            descriptor: .qwen38_27b,
            logger: .shared
        )
        XCTAssertEqual(backend.family, .qwen38Dense)
        XCTAssertEqual(backend.parametersInfo.modelName, "stub")
    }

    func testEverySupportedModuleHasADefaultBackendBuilder() throws {
        let registry = QWModelBackendRegistry()
        for family in QWModuleRegistry.supportedFamilies {
            let module = try XCTUnwrap(QWModuleRegistry.moduleType(for: family))
            XCTAssertNotNil(registry.builder(for: module), family.rawValue)
        }
        XCTAssertNil(registry.builder(for: String.self))
    }

    func testUnifiedBackendForwardsResolvedPolicyAndEngineResult() throws {
        let engine = RecordingInferenceEngine()
        var speculation = QWSpeculativeParameters(mtpDrafts: 2, mtpBackend: "variational",
                                                   mtpDraftStrategy: "ngram")
        speculation.draftStrategyIsResolved = true
        let backend = QWUnifiedModelBackend(
            engine: engine, descriptor: .qwen38_27b,
            parametersInfo: QWEngineParametersInfo(modelPackPath: "/tmp/stub", modelName: "stub"),
            resolvedSpeculation: speculation)
        let request = QWModelGenerationRequest(
            promptTokens: [1, 2], maxNewTokens: 3, sampling: QWSamplingParameters(seed: 42),
            stopTokenIds: [9])
        let result = try backend.generate(request: request,
                                          config: QWEngineLifecycleActor.Config(packPath: "/tmp/stub"),
                                          onToken: nil)
        XCTAssertEqual(result.tokens, [7])
        XCTAssertEqual(result.activeContextCapacity, 128)
        XCTAssertEqual(engine.received?.prompt, [1, 2])
        XCTAssertEqual(engine.received?.speculative, speculation)
        XCTAssertTrue(backend.capabilities.contains(.speculative))
    }
}

private final class RecordingInferenceEngine: QWInferenceEngine, @unchecked Sendable {
    var received: QWGenerationRequest?
    var capabilities: Set<QWKnobCapability> { [.speculative] }
    var hasPrefixCache: Bool { false }
    var prefixCache: QWPrefixCacheState { QWPrefixCacheState(isConfigured: false) }
    var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)?
    var report: QWEngineReport {
        QWEngineReport(capacityLog: "", arenaBytes: 0, weightUsedBytes: 0,
                       weightReservedBytes: 0, contextCapacity: 128)
    }
    var lastTurn: QWGenerationResult? { nil }
    func makeConversation() -> QWConversation? { nil }
    func generate(request: QWGenerationRequest, onToken: ((Int) -> Bool)?) throws -> QWGenerationResult {
        received = request
        return QWGenerationResult(tokens: [7], stepMs: [], stepTokenCounts: [])
    }
}

private struct StubModelBackend: QWModelBackend {
    let descriptor: QWModelDescriptor
    let parametersInfo: QWEngineParametersInfo

    var family: QWModelDescriptor.Family { descriptor.family }
    var visionBackend: (any QWVisionBackend)? { nil }
    var hardwareCapabilities: QWDeviceCaps? { nil }
    var prefixCache: QWPrefixCacheState { QWPrefixCacheState(isConfigured: false) }
    var capabilities: Set<QWKnobCapability> { [] }

    func generate(
        request: QWModelGenerationRequest,
        config: QWEngineLifecycleActor.Config,
        onToken: ((Int) -> Bool)?
    ) throws -> QWModelCompletionMetrics {
        QWModelCompletionMetrics(
            tokens: [],
            prefillSec: 0.01,
            decodeSec: 0.01,
            firstTokenSec: 0.01,
            activeContextCapacity: config.maxContext
        )
    }
}
