import XCTest
@testable import QWServer
import QWCore

final class QWSettingsTests: XCTestCase {
    func testSettingsDefaultsAndCustomValues() {
        let defaults = QWServerSettings()
        XCTAssertEqual(defaults.host, "127.0.0.1")
        XCTAssertEqual(defaults.port, 8080)
        XCTAssertEqual(defaults.maxContext, 4096)
        XCTAssertEqual(defaults.warmRAMLimitGB, 4.0)
        XCTAssertEqual(defaults.ssdCacheLimitGB, 10.0)
        XCTAssertEqual(defaults.logDir, "~/.qwg/logs")
        XCTAssertEqual(defaults.logFileName, "qwg.log")
        XCTAssertFalse(defaults.runAtStartup)
        XCTAssertTrue(defaults.autoStartServer)
        XCTAssertTrue(defaults.expandedLogFilePath.hasSuffix(".qwg/logs/qwg.log"))

        var custom = defaults
        custom.port = 9090
        custom.runAtStartup = true
        custom.warmRAMLimitGB = 8.0
        custom.logLevel = "debug"
        XCTAssertEqual(custom.port, 9090)
        XCTAssertTrue(custom.runAtStartup)
        XCTAssertEqual(custom.warmRAMLimitGB, 8.0)
        XCTAssertEqual(custom.logLevel, "debug")
    }

    func testSettingsSerializationAndPersistence() throws {
        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let fileURL = tmpDir.appendingPathComponent("settings.json")
        let manager = QWSettingsManager()

        var settings = manager.loadSettings(from: fileURL)
        XCTAssertEqual(settings.port, 8080)

        settings.port = 8888
        settings.runAtStartup = true
        settings.modelName = "Qwen3.8-27B-Custom"
        try manager.saveSettings(settings, to: fileURL)

        XCTAssertTrue(FileManager.default.fileExists(atPath: fileURL.path))
        manager.invalidateCache()

        let reloaded = manager.loadSettings(from: fileURL)
        XCTAssertEqual(reloaded.port, 8888)
        XCTAssertTrue(reloaded.runAtStartup)
        XCTAssertEqual(reloaded.modelName, "Qwen3.8-27B-Custom")
    }

    final class CapturedBox: @unchecked Sendable {
        var line: String?
    }

    func testLoggerFileSinkAndRingBuffer() {
        let logger = QWServerLogger(level: .debug)
        let tmpLog = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-test-\(UUID().uuidString)")
            .appendingPathComponent("test.log")

        logger.configureFileSink(path: tmpLog.path)
        defer {
            try? FileManager.default.removeItem(at: tmpLog.deletingLastPathComponent())
        }

        let box = CapturedBox()
        let subId = logger.subscribe { line in
            box.line = line
        }

        logger.log(.info, "Test message from QWSettingsTests")
        XCTAssertNotNil(box.line)
        XCTAssertTrue(box.line!.contains("Test message from QWSettingsTests"))

        let recent = logger.getRecentLogs()
        XCTAssertFalse(recent.isEmpty)
        XCTAssertTrue(recent.last!.contains("Test message from QWSettingsTests"))

        logger.unsubscribe(subId)
    }

    func testEngineInfoResponseSchema() throws {
        let hw = QWEngineInfoResponse.HardwareFacts(
            modelIdentifier: "Mac16,6",
            cpuBrand: "Apple M5 Max",
            physicalCores: 16,
            logicalCores: 16,
            physicalMemoryBytes: 128 * 1024 * 1024 * 1024,
            physicalMemoryGB: 128.0,
            metalDeviceName: "Apple M5 Max",
            metalVersion: "Metal 4 (Tensor)",
            isTensorCapable: true,
            dramBwGBs: 585.0
        )
        let params = QWEngineInfoResponse.ServerParameters(
            modelName: "Qwen3.8-27B",
            modelPackPath: "/models/qwen38.qwp",
            headStorage: "Q4g",
            contextBudget: 32768,
            kvMode: "f16",
            mtpDrafts: 3,
            mtpBackend: "standard",
            gdnPrefillMode: "chunked",
            prefillScratchEviction: "auto",
            hostLogicalCores: 16,
            isLoaded: true
        )
        let cache = QWEngineInfoResponse.CacheConfigInfo(
            warmLimitBytes: 4 * 1024 * 1024 * 1024,
            ssdLimitBytes: 10 * 1024 * 1024 * 1024,
            cacheDir: "/tmp/cache"
        )
        let resp = QWEngineInfoResponse(
            status: "loaded",
            hardware: hw,
            parameters: params,
            cacheConfig: cache,
            lifecycle: QWEngineLifecycleInfo(phase: .loaded, since: Date(), model: "qwen38", detail: nil),
            endpoints: ["metrics": "GET /v1/metrics"]
        )

        let data = try JSONEncoder().encode(resp)
        let decoded = try JSONDecoder().decode(QWEngineInfoResponse.self, from: data)
        XCTAssertEqual(decoded.hardware.metalVersion, "Metal 4 (Tensor)")
        XCTAssertEqual(decoded.parameters.headStorage, "Q4g")
        XCTAssertEqual(decoded.parameters.hostLogicalCores, 16)
        XCTAssertEqual(decoded.cacheConfig.warmLimitBytes, 4 * 1024 * 1024 * 1024)
    }

    func testModelAliasAndDiscoveredModels() {
        var settings = QWServerSettings()
        XCTAssertEqual(settings.modelAlias, "")

        settings.modelAlias = "qwen-custom-alias"
        XCTAssertEqual(settings.modelAlias, "qwen-custom-alias")

        let hasLocalModels = ((try? FileManager.default.contentsOfDirectory(atPath: QWPaths.modelsDirectoryPath)) ?? [])
            .contains { !$0.hasPrefix(".") }
        let scanned = QWServerSettings.scanModelsDirectory()
        if hasLocalModels {
            XCTAssertFalse(scanned.isEmpty)
        }

        let model = DiscoveredModel(name: "test-model", packPath: "/path/to/test.qwp")
        XCTAssertEqual(model.id, "test-model")
        XCTAssertEqual(model.packPath, "/path/to/test.qwp")
    }

    func testHardwareUtilizationChipIdentifier() {
        let sampler = QWSystemMetricsSampler()
        let sample = sampler.sample()
        XCTAssertFalse(sample.chipIdentifier.isEmpty)
        XCTAssertTrue(sample.memoryTotalBytes > 0)
        XCTAssertTrue(sample.memoryTotalGB > 0)
    }

    func testConfigurationResolverPrecedenceHierarchy() throws {
        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let configFileURL = tmpDir.appendingPathComponent("custom-settings.json")
        var fileSettings = QWServerSettings()
        fileSettings.port = 8181
        fileSettings.host = "10.0.0.1"
        fileSettings.modelPackPath = "/file/path/model.qwp"
        fileSettings.maxContext = 8192
        fileSettings.kvMode = "q8"
        fileSettings.modelName = "FileModel"
        fileSettings.modelAlias = "file-alias"
        try QWSettingsManager.shared.saveSettings(fileSettings, to: configFileURL)

        // 1. Config file values are used when CLI and Env are absent
        let configOnly = try QWServerConfigurationResolver.resolve(
            cliConfigPath: configFileURL.path,
            environment: [:]
        )
        XCTAssertEqual(configOnly.port, 8181)
        XCTAssertEqual(configOnly.host, "10.0.0.1")
        XCTAssertEqual(configOnly.modelPackPath, "/file/path/model.qwp")
        XCTAssertEqual(configOnly.maxContext, 8192)
        XCTAssertEqual(configOnly.kvMode, "q8")
        XCTAssertEqual(configOnly.modelName, "FileModel")
        XCTAssertEqual(configOnly.modelAlias, "file-alias")

        // 2. Environment variables override config file values
        let envOverrides = try QWServerConfigurationResolver.resolve(
            cliConfigPath: configFileURL.path,
            environment: [
                "QWG_PORT": "9292",
                "QWG_HOST": "0.0.0.0",
                "QWG_PACK": "/env/path/model.qwp",
                "QWG_MAX_CONTEXT": "16384",
                "QWG_MODEL_ALIAS": "env-alias",
            ]
        )
        XCTAssertEqual(envOverrides.port, 9292)
        XCTAssertEqual(envOverrides.host, "0.0.0.0")
        XCTAssertEqual(envOverrides.modelPackPath, "/env/path/model.qwp")
        XCTAssertEqual(envOverrides.maxContext, 16384)
        XCTAssertEqual(envOverrides.modelAlias, "env-alias")
        XCTAssertEqual(envOverrides.kvMode, "q8") // Unchanged in env, falls back to config file

        // 3. CLI options override both Environment variables and Config file
        let cliOverrides = try QWServerConfigurationResolver.resolve(
            cliConfigPath: configFileURL.path,
            cliPack: "/cli/path/model.qwp",
            cliHost: "192.168.1.5",
            cliPort: 9999,
            cliModelAlias: "cli-alias",
            cliMaxContext: 32768,
            cliKVMode: "q4",
            environment: [
                "QWG_PORT": "9292",
                "QWG_HOST": "0.0.0.0",
                "QWG_PACK": "/env/path/model.qwp",
            ]
        )
        XCTAssertEqual(cliOverrides.port, 9999)
        XCTAssertEqual(cliOverrides.host, "192.168.1.5")
        XCTAssertEqual(cliOverrides.modelPackPath, "/cli/path/model.qwp")
        XCTAssertEqual(cliOverrides.modelAlias, "cli-alias")
        XCTAssertEqual(cliOverrides.maxContext, 32768)
        XCTAssertEqual(cliOverrides.kvMode, "q4")
    }

    func testSamplingParametersPersistenceAndResolution() throws {
        // 1. Defaults verification
        let defaults = QWServerSettings()
        XCTAssertEqual(defaults.temperature, 0.7)
        XCTAssertEqual(defaults.topP, 0.9)
        XCTAssertEqual(defaults.topK, 40)
        XCTAssertEqual(defaults.maxTokens, 512)
        XCTAssertEqual(defaults.repetitionPenalty, 1.05)
        XCTAssertEqual(defaults.seed, -1)
        XCTAssertTrue(defaults.enableThinking)
        XCTAssertEqual(defaults.reasoningEffort, "xhigh")
        XCTAssertEqual(defaults.systemPrompt, "")
        XCTAssertTrue(defaults.lazyLoad)

        // 2. Backward compatibility with older settings.json without sampling keys
        let legacyJson = """
        {
            "port": 8080,
            "host": "127.0.0.1",
            "modelName": "LegacyModel",
            "kvMode": "f16"
        }
        """.data(using: .utf8)!
        let legacyDecoded = try JSONDecoder().decode(QWServerSettings.self, from: legacyJson)
        XCTAssertEqual(legacyDecoded.modelName, "LegacyModel")
        XCTAssertEqual(legacyDecoded.temperature, 0.7)
        XCTAssertEqual(legacyDecoded.topP, 0.9)
        XCTAssertEqual(legacyDecoded.topK, 40)
        XCTAssertEqual(legacyDecoded.repetitionPenalty, 1.05)
        XCTAssertTrue(legacyDecoded.enableThinking)
        XCTAssertEqual(legacyDecoded.reasoningEffort, "xhigh")

        // 3. Custom values round-trip serialization
        var custom = defaults
        custom.temperature = 0.2
        custom.topP = 0.95
        custom.topK = 20
        custom.maxTokens = 2048
        custom.repetitionPenalty = 1.15
        custom.seed = 42
        custom.enableThinking = false
        custom.reasoningEffort = "low"
        custom.systemPrompt = "You are a code specialist."
        custom.lazyLoad = false

        let encoded = try JSONEncoder().encode(custom)
        let reloaded = try JSONDecoder().decode(QWServerSettings.self, from: encoded)
        XCTAssertEqual(reloaded.temperature, 0.2)
        XCTAssertEqual(reloaded.topP, 0.95)
        XCTAssertEqual(reloaded.topK, 20)
        XCTAssertEqual(reloaded.maxTokens, 2048)
        XCTAssertEqual(reloaded.repetitionPenalty, 1.15)
        XCTAssertEqual(reloaded.seed, 42)
        XCTAssertFalse(reloaded.enableThinking)
        XCTAssertEqual(reloaded.reasoningEffort, "low")
        XCTAssertEqual(reloaded.systemPrompt, "You are a code specialist.")
        XCTAssertFalse(reloaded.lazyLoad)

        // 4. Configuration resolver precedence for sampling parameters
        let resolved = try QWServerConfigurationResolver.resolve(
            cliTemperature: 0.0,
            cliTopP: 0.85,
            cliTopK: 10,
            cliMaxTokens: 4096,
            cliRepetitionPenalty: 1.1,
            cliSeed: 1234,
            cliEnableThinking: false,
            cliReasoningEffort: "medium",
            cliSystemPrompt: "CLI system prompt",
            cliLazy: false,
            environment: [
                "QWG_TEMPERATURE": "0.5",
                "QWG_ENABLE_THINKING": "true"
            ]
        )
        // CLI options take precedence over environment
        XCTAssertEqual(resolved.temperature, 0.0)
        XCTAssertEqual(resolved.topP, 0.85)
        XCTAssertEqual(resolved.topK, 10)
        XCTAssertEqual(resolved.maxTokens, 4096)
        XCTAssertEqual(resolved.repetitionPenalty, 1.1)
        XCTAssertEqual(resolved.seed, 1234)
        XCTAssertFalse(resolved.enableThinking)
        XCTAssertEqual(resolved.reasoningEffort, "medium")
        XCTAssertEqual(resolved.systemPrompt, "CLI system prompt")
        XCTAssertFalse(resolved.lazy)
    }

    func testDFlash2SettingsAndResolverPrecedence() throws {
        var settings = QWServerSettings()
        XCTAssertEqual(settings.dflashDraftPackPath, "")
        XCTAssertEqual(settings.dflashSelectorTopK, 16)

        settings.mtpBackend = "dflash2"
        settings.mtpDrafts = 7
        settings.dflashDraftPackPath = "/models/dflash2"
        settings.dflashSelectorTopK = 8

        let encoded = try JSONEncoder().encode(settings)
        let decoded = try JSONDecoder().decode(QWServerSettings.self, from: encoded)
        XCTAssertEqual(decoded.mtpBackend, "dflash2")
        XCTAssertEqual(decoded.mtpDrafts, 7)
        XCTAssertEqual(decoded.dflashDraftPackPath, "/models/dflash2")
        XCTAssertEqual(decoded.dflashSelectorTopK, 8)

        let resolvedBlock = try QWServerConfigurationResolver.resolve(
            cliDFlashDraftPack: "/cli/dflash2",
            cliDFlashBlockSize: 8,
            cliDFlashTopK: 4,
            environment: [:]
        )
        XCTAssertEqual(resolvedBlock.mtpBackend, "dflash2")
        XCTAssertEqual(resolvedBlock.mtpDrafts, 7)
        XCTAssertEqual(resolvedBlock.dflashDraftPackPath, "/cli/dflash2")
        XCTAssertEqual(resolvedBlock.dflashSelectorTopK, 4)
    }

    func testMultiModelAliasesPersistencePruningAndResolution() throws {
        var settings = QWServerSettings()
        let models = [
            DiscoveredModel(name: "Synthetic-27B", packPath: "/tmp/models/Synthetic-27B"),
            DiscoveredModel(name: "Synthetic-Flash-Next", packPath: "/tmp/models/Synthetic-Flash-Next")
        ]

        // 1. Assign per-model aliases
        settings.setAlias("qwen-27b", for: "Synthetic-27B")
        settings.setAlias("flash-next", for: "Synthetic-Flash-Next")
        settings.modelName = "Synthetic-27B"
        settings.modelPackPath = "/tmp/models/Synthetic-27B"
        settings.syncActiveModelAlias()

        XCTAssertEqual(settings.alias(for: "Synthetic-27B"), "qwen-27b")
        XCTAssertEqual(settings.alias(for: "Synthetic-Flash-Next"), "flash-next")
        XCTAssertEqual(settings.modelAlias, "qwen-27b")

        // 2. Resolve by alias, model name, or path
        let byAlias = settings.resolveModelReference("flash-next", discovered: models)
        XCTAssertEqual(byAlias.packPath, "/tmp/models/Synthetic-Flash-Next")
        XCTAssertEqual(byAlias.modelName, "Synthetic-Flash-Next")
        XCTAssertEqual(byAlias.alias, "flash-next")

        let byName = settings.resolveModelReference("Synthetic-27B", discovered: models)
        XCTAssertEqual(byName.packPath, "/tmp/models/Synthetic-27B")
        XCTAssertEqual(byName.modelName, "Synthetic-27B")
        XCTAssertEqual(byName.alias, "qwen-27b")

        // 3. Round-trip persistence of modelAliases dictionary
        let data = try JSONEncoder().encode(settings)
        var reloaded = try JSONDecoder().decode(QWServerSettings.self, from: data)
        XCTAssertEqual(reloaded.alias(for: "Synthetic-27B"), "qwen-27b")
        XCTAssertEqual(reloaded.alias(for: "Synthetic-Flash-Next"), "flash-next")

        // 4. Prune removed models from modelAliases
        let remainingModels = [
            DiscoveredModel(name: "Synthetic-27B", packPath: "/tmp/models/Synthetic-27B")
        ]
        let didPrune = reloaded.pruneRemovedModels(discovered: remainingModels)
        XCTAssertTrue(didPrune)
        XCTAssertEqual(reloaded.alias(for: "Synthetic-27B"), "qwen-27b")
        XCTAssertEqual(reloaded.alias(for: "Synthetic-Flash-Next"), "")
        XCTAssertFalse(reloaded.pruneRemovedModels(discovered: remainingModels))
    }

    func testTheAliasOfAnActiveModelUnderACustomNameSurvivesThePrune() throws {
        let installed = [
            DiscoveredModel(name: "Synthetic-Flash-Next", packPath: "/tmp/models/Synthetic-Flash-Next")
        ]
        var settings = QWServerSettings()
        // The engine serves the model under a name of the user's
        // choosing; the pack says which model that name means.
        settings.modelName = "Qwen3"
        settings.modelPackPath = "/tmp/models/Synthetic-Flash-Next"
        settings.setAlias("uat-alias", for: "Qwen3")
        settings.syncActiveModelAlias()
        XCTAssertEqual(settings.modelAlias, "uat-alias")

        // The pack is installed: the alias belongs to a model that is
        // there, and the prune must not take it.
        XCTAssertFalse(settings.pruneRemovedModels(discovered: installed))
        XCTAssertEqual(settings.modelAlias, "uat-alias")
        XCTAssertEqual(settings.alias(for: "Qwen3"), "uat-alias")

        // The pack is gone from the models directory: the alias has
        // nothing left to belong to.
        settings.modelPackPath = QWPaths.modelsDirectoryPath + "/Synthetic-Gone"
        XCTAssertTrue(settings.pruneRemovedModels(discovered: installed))
        XCTAssertEqual(settings.modelAlias, "")
        XCTAssertEqual(settings.alias(for: "Qwen3"), "")
    }

    func testVariationalMTPSettingsAndResetToTopPerformanceDefaults() throws {
        var settings = QWServerSettings()
        XCTAssertEqual(settings.mtpBackend, "variational")
        XCTAssertEqual(settings.mtpDrafts, 3)
        XCTAssertEqual(settings.mtpRelaxedMargin, 1.5, accuracy: 1e-5)
        XCTAssertEqual(settings.mtpRelaxedMinProb, 0.2, accuracy: 1e-5)
        XCTAssertEqual(settings.mtpDraftStrategy, "hybrid")
        XCTAssertTrue(settings.mtpQuadraticSeeding)

        // Mutate speculative and runtime settings away from optimal defaults
        settings.modelName = "Qwen3.5-35B-A3B"
        settings.modelPackPath = "/models/Qwen3.5-35B-A3B"
        settings.setAlias("fast-moe", for: "Qwen3.5-35B-A3B")
        settings.syncActiveModelAlias()
        settings.mtpBackend = "standard"
        settings.mtpDrafts = 0
        settings.mtpRelaxedMargin = 0.0
        settings.mtpRelaxedMinProb = 0.35
        settings.mtpDraftStrategy = "ngram"
        settings.mtpQuadraticSeeding = false
        settings.mtpAdaptiveDepth = true
        settings.mtpEntropyThreshold = 3.5
        settings.temperature = 1.4
        settings.kvMode = "q4"

        let data = try JSONEncoder().encode(settings)
        var decoded = try JSONDecoder().decode(QWServerSettings.self, from: data)
        XCTAssertEqual(decoded.mtpBackend, "standard")
        XCTAssertEqual(decoded.mtpDrafts, 0)
        XCTAssertEqual(decoded.mtpRelaxedMargin, 0.0, accuracy: 1e-5)
        XCTAssertEqual(decoded.mtpRelaxedMinProb, 0.35, accuracy: 1e-5)
        XCTAssertEqual(decoded.mtpDraftStrategy, "ngram")
        XCTAssertFalse(decoded.mtpQuadraticSeeding)
        XCTAssertTrue(decoded.mtpAdaptiveDepth)
        XCTAssertEqual(decoded.mtpEntropyThreshold, 3.5, accuracy: 1e-5)

        // Reset to top-performance defaults for a native-MTP descriptor
        let descriptor = QWModelDescriptor.qwen38_27b
        decoded.resetToTopPerformanceDefaults(for: descriptor)

        XCTAssertEqual(decoded.mtpBackend, "variational")
        XCTAssertEqual(decoded.mtpDrafts, 3)
        XCTAssertEqual(decoded.mtpRelaxedMargin, 1.5, accuracy: 1e-5)
        XCTAssertEqual(decoded.mtpRelaxedMinProb, 0.2, accuracy: 1e-5)
        XCTAssertEqual(decoded.mtpDraftStrategy, "hybrid")
        XCTAssertTrue(decoded.mtpQuadraticSeeding)
        XCTAssertFalse(decoded.mtpAdaptiveDepth)
        XCTAssertEqual(decoded.mtpEntropyThreshold, 0.7, accuracy: 1e-5)
        XCTAssertEqual(decoded.temperature, 0.7, accuracy: 1e-5)
        XCTAssertEqual(decoded.kvMode, "q4")
        XCTAssertEqual(decoded.lsaMode, "auto")
        // Identity & aliases must be preserved across Reset to Defaults
        XCTAssertEqual(decoded.modelName, "Qwen3.5-35B-A3B")
        XCTAssertEqual(decoded.modelPackPath, "/models/Qwen3.5-35B-A3B")
        XCTAssertEqual(decoded.alias(for: "Qwen3.5-35B-A3B"), "fast-moe")

        // Verify CLI / Env resolution for Variational MTP knobs
        let resolved = try QWServerConfigurationResolver.resolve(
            cliMTPRelaxedMargin: 1.75,
            cliMTPRelaxedMinProb: 0.3,
            cliMTPDraftStrategy: "mtp",
            cliMTPQuadraticSeeding: false,
            environment: [
                "QWG_MTP_RELAXED_MARGIN": "0.5",
                "QWG_MTP_RELAXED_MIN_PROB": "0.1",
                "QWG_MTP_DRAFT_STRATEGY": "ngram",
                "QWG_MTP_QUADRATIC_SEEDING": "true"
            ]
        )
        XCTAssertEqual(resolved.mtpRelaxedMargin, 1.75, accuracy: 1e-5)
        XCTAssertEqual(resolved.mtpRelaxedMinProb, 0.3, accuracy: 1e-5)
        XCTAssertEqual(resolved.mtpDraftStrategy, "mtp")
        XCTAssertFalse(resolved.mtpQuadraticSeeding)
        let fromEnv = try QWServerConfigurationResolver.resolve(
            environment: ["QWG_MTP_RELAXED_MIN_PROB": "0.1"])
        XCTAssertEqual(fromEnv.mtpRelaxedMinProb, 0.1, accuracy: 1e-5)
        XCTAssertEqual(fromEnv.speculativeParameters.mtpRelaxedMinProb, 0.1, accuracy: 1e-5)

        let legacy = try JSONDecoder().decode(QWServerSettings.self, from: Data("{}".utf8))
        XCTAssertEqual(legacy.mtpRelaxedMinProb, 0.2, accuracy: 1e-5)
    }

    func testPerModelProfileAndModelConfigDefaultAdoption() throws {
        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        // Model A: no MTP, custom temperature 0.25 saved in profile
        let modelADir = tmpDir.appendingPathComponent("ModelA", isDirectory: true)
        let assetsA = modelADir.appendingPathComponent("assets", isDirectory: true)
        try FileManager.default.createDirectory(at: assetsA, withIntermediateDirectories: true)
        try Data().write(to: modelADir.appendingPathComponent("model.qwp"))
        try Data("{}".utf8).write(to: assetsA.appendingPathComponent("tokenizer.json"))
        try Data("{% for message in messages %}{{ message.content }}{% endfor %}".utf8)
            .write(to: assetsA.appendingPathComponent("chat_template.jinja"))
        try Data("""
        {
          "model_type": "qwen3_5",
          "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248320,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mtp_num_hidden_layers": 0
          }
        }
        """.utf8).write(to: assetsA.appendingPathComponent("config.json"))

        // Model B: SenseNova-Loop with MTP=1 and generation_config.json (temperature=1.0, top_p=0.95, top_k=20)
        let modelBDir = tmpDir.appendingPathComponent("SenseNova-Loop-9B-Preview", isDirectory: true)
        let assetsB = modelBDir.appendingPathComponent("assets", isDirectory: true)
        try FileManager.default.createDirectory(at: assetsB, withIntermediateDirectories: true)
        try Data().write(to: modelBDir.appendingPathComponent("model.qwp"))
        try Data("{}".utf8).write(to: assetsB.appendingPathComponent("tokenizer.json"))
        try Data("{% for message in messages %}{{ message.content }}{% endfor %}".utf8)
            .write(to: assetsB.appendingPathComponent("chat_template.jinja"))
        try Data("""
        {
          "model_type": "sensenova_loop",
          "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 32,
            "full_attention_interval": 4,
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
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {
              "mrope_interleaved": true,
              "mrope_section": [11, 11, 10],
              "partial_rotary_factor": 0.25,
              "rope_theta": 10000000,
              "rope_type": "default"
            },
            "eos_token_id": 248044,
            "vocab_size": 248320,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-6,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mtp_num_hidden_layers": 1,
            "loop_settings": {"start": 4, "end": 28, "count": 2},
            "mhc": {"enabled": true, "start": 4, "end": 28, "num_steps": 3}
          }
        }
        """.utf8).write(to: assetsB.appendingPathComponent("config.json"))
        try Data("""
        {
          "do_sample": true,
          "temperature": 1.0,
          "top_p": 0.95,
          "top_k": 20
        }
        """.utf8).write(to: assetsB.appendingPathComponent("generation_config.json"))

        let settingsURL = tmpDir.appendingPathComponent("settings.json")
        var settingsA = QWServerSettings()
        settingsA.modelName = "ModelA"
        settingsA.modelPackPath = modelADir.path
        settingsA.speculativeDecodingEnabled = false
        settingsA.mtpDrafts = 0
        settingsA.temperature = 0.25
        settingsA.maxContext = 16384
        try QWSettingsManager.shared.saveSettings(settingsA, to: settingsURL)

        // Verify per-model profile was written for ModelA
        let profileA = try XCTUnwrap(QWSettingsManager.shared.loadProfile(forModel: "ModelA", from: settingsURL))
        XCTAssertEqual(profileA.values["temperature"], .double(0.25))
        XCTAssertEqual(profileA.values["contextTokens"], .int(16384))

        // Resolve for ModelB (which has NO profile yet): must adopt ModelB's config.json + generation_config.json
        // and enable MTP speculative decoding by default, ignoring ModelA's saved settings!
        let resolvedB = try QWServerConfigurationResolver.resolve(
            cliConfigPath: settingsURL.path,
            cliPack: modelBDir.path,
            cliModelName: "SenseNova-Loop-9B-Preview",
            environment: [:]
        )
        XCTAssertTrue(resolvedB.speculativeDecodingEnabled)
        XCTAssertEqual(resolvedB.mtpDrafts, 3)
        XCTAssertEqual(resolvedB.maxContext, 131072)
        XCTAssertEqual(resolvedB.temperature, 1.0, accuracy: 1e-5)
        XCTAssertEqual(resolvedB.topP, 0.95, accuracy: 1e-5)
        XCTAssertEqual(resolvedB.topK, 20)

        // Switching back to ModelA restores ModelA's dedicated profile
        let (effectiveA, restoredProfileA) = QWSettingsManager.shared.effectiveSettings(
            forModel: "ModelA",
            packPath: modelADir.path,
            base: settingsA,
            from: settingsURL
        )
        XCTAssertNotNil(restoredProfileA)
        XCTAssertEqual(effectiveA.temperature, 0.25, accuracy: 1e-5)
        XCTAssertEqual(effectiveA.maxContext, 16384)
        XCTAssertFalse(effectiveA.speculativeDecodingEnabled)
    }
}
