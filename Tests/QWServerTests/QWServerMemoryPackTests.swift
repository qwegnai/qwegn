// QWServerMemoryPackTests.swift — Tests for QWServer memory pack mounting,
// agent system-prompt auto-discovery, and kvMode enforcement/restoration.

import Foundation
import QWCore
import QWEngine
import QWHAL
import QWKV
import QWToken
import QWVision
import XCTest
@testable import QWServer

private final class MockMemoryBackend: QWModelBackend, @unchecked Sendable {
    let descriptor: QWModelDescriptor = .qwen38_27b
    var family: QWModelDescriptor.Family { descriptor.family }
    let parametersInfo: QWEngineParametersInfo
    let visionBackend: (any QWVisionBackend)? = nil
    let hardwareCapabilities: QWDeviceCaps? = nil
    var prefixCache: QWPrefixCacheState = QWPrefixCacheState(
        isConfigured: true, warmBytes: 0, ssdBytes: 0, warmEntries: 0, ssdEntries: 0
    )
    let capabilities: Set<QWKnobCapability> = [.lsa]
    var mountedURLs: [URL] = []
    var unmountCount: Int = 0
    let stubTokens: [Int]
    let stubBytes: Int

    init(config: QWEngineLifecycleActor.Config, stubTokens: [Int], stubBytes: Int) {
        self.stubTokens = stubTokens
        self.stubBytes = stubBytes
        self.parametersInfo = QWEngineParametersInfo(
            modelPackPath: config.packPath,
            modelName: config.modelName,
            modelDescriptor: .qwen38_27b,
            headStorage: "HQ4_G32_M5",
            visionStatus: "none",
            contextBudget: config.maxContext,
            kvMode: config.kvMode,
            mtpDrafts: 0,
            mtpBackend: "variational",
            gdnPrefillMode: "chunked",
            prefillScratchEviction: "auto",
            prefillChunkRows: 512,
            weightContainerNote: nil
        )
    }

    func generate(
        request: QWModelGenerationRequest,
        config: QWEngineLifecycleActor.Config,
        onToken: ((Int) -> Bool)?
    ) throws -> QWModelCompletionMetrics {
        QWModelCompletionMetrics(
            tokens: [1],
            prefillSec: 0.01,
            decodeSec: 0.01,
            firstTokenSec: 0.01,
            activeContextCapacity: config.maxContext,
            cacheHitTokens: stubTokens.count,
            cacheTier: "warm_ram",
            cacheActive: true,
            warmCacheBytes: stubBytes,
            ssdCacheBytes: 0
        )
    }

    func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) {
        mountedURLs.append(fileURL)
        prefixCache = QWPrefixCacheState(
            isConfigured: true, warmBytes: stubBytes, ssdBytes: 0, warmEntries: 1, ssdEntries: 0
        )
        return (stubTokens, stubBytes)
    }

    func unmountSharedBasePrefixes() {
        unmountCount += 1
        prefixCache = QWPrefixCacheState(
            isConfigured: true, warmBytes: 0, ssdBytes: 0, warmEntries: 0, ssdEntries: 0
        )
    }
}

final class QWServerMemoryPackTests: XCTestCase {

    private func makeMinimalTokenizerAndTemplate() throws -> (QWTokenizer, QWChatTemplate) {
        var vocab: [String: Int] = ["<|im_start|>": 0, "<|im_end|>": 1]
        for b in 0..<256 {
            vocab[String(format: "<0x%02X>", b)] = 2 + b
        }
        let root: [String: Any] = [
            "version": "1.0",
            "pre_tokenizer": [
                "type": "Sequence",
                "pretokenizers": [
                    ["type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": "always", "split": false]
                ]
            ],
            "model": [
                "type": "BPE",
                "byte_fallback": true,
                "vocab": vocab,
                "merges": [] as [String]
            ],
            "added_tokens": [
                ["id": 0, "content": "<|im_start|>", "special": true],
                ["id": 1, "content": "<|im_end|>", "special": true]
            ]
        ]
        let vocabData = try JSONSerialization.data(withJSONObject: root)
        let tok = try QWTokenizer(json: vocabData)
        let tpl = try QWChatTemplate(source: "{% for m in messages %}{{ m.content }}{% endfor %}")
        return (tok, tpl)
    }

    func testMountEnforcesKVModeAndContextAndUnmountRestoresOriginalConfig() async throws {
        let fm = FileManager.default
        let tempRoot = fm.temporaryDirectory.appendingPathComponent("qwg_srv_mem_\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: tempRoot, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tempRoot) }

        let qwgDir = QWMemoryManifest.qwgDirectoryURL(in: tempRoot)
        try fm.createDirectory(at: qwgDir, withIntermediateDirectories: true)
        let cacheFile = qwgDir.appendingPathComponent("demo_test-model_q4.qwcache")
        try Data([0x51, 0x57, 0x43, 0x48]).write(to: cacheFile)

        let artifact = QWMemoryManifest.CacheArtifact(
            modelName: "test-model",
            geometryDigest: QWModelDescriptor.qwen38_27b.geometryDigestV2Hex,
            kvMode: "q4",
            lsaMode: "auto",
            lsaSidecarDigest: nil,
            tokenCount: 8192,
            cacheFileName: cacheFile.lastPathComponent,
            cacheFileBytes: 4_194_304
        )
        let manifest = QWMemoryManifest(
            packName: "demo-repo",
            rootPath: tempRoot.path,
            clusterBlockTokens: 512,
            files: [
                QWMemoryManifest.FileEntry(
                    relativePath: "README.md",
                    tier: 0,
                    sha256: "abc",
                    mtimeEpochSec: 1000,
                    byteCount: 128,
                    tokenOffset: 0,
                    tokenCount: 64
                )
            ],
            caches: [artifact]
        )
        try manifest.save(to: tempRoot)

        let (tok, tpl) = try makeMinimalTokenizerAndTemplate()
        let stubTokens = Array(repeating: 2, count: 8192)
        let initialConfig = QWEngineLifecycleActor.Config(
            packPath: "/dummy/Qwen3.8-27B-hq4g32.qwp",
            modelName: "test-model",
            maxContext: 4096,
            kvMode: .f16,
            host: "127.0.0.1",
            port: 0
        )

        let actor = QWEngineLifecycleActor(
            config: initialConfig,
            logger: QWServerLogger(level: .none),
            load: { cfg, _, _ in
                let backend = MockMemoryBackend(config: cfg, stubTokens: stubTokens, stubBytes: 4_194_304)
                return QWEngineLifecycleActor.LoadedState(modelBackend: backend, tokenizer: tok, chatTemplate: tpl)
            }
        )

        XCTAssertEqual(actor.getConfig().kvMode, .f16)
        XCTAssertEqual(actor.getConfig().maxContext, 4096)
        XCTAssertNil(actor.getMountedMemoryPackInfo())

        let mountedInfo = try await actor.mountMemoryPack(at: tempRoot.path)
        XCTAssertEqual(mountedInfo.packName, "demo-repo")
        XCTAssertEqual(mountedInfo.enforcedKVMode, "q4")
        XCTAssertTrue(mountedInfo.enforcesKVMode)
        XCTAssertEqual(mountedInfo.tokenCount, 8192)
        XCTAssertEqual(actor.getConfig().kvMode, .q4)
        XCTAssertGreaterThanOrEqual(actor.getConfig().maxContext, 8192 + 16384)
        XCTAssertEqual(actor.getMountedMemoryPackInfo()?.enforcedKVMode, "q4")

        await actor.unmountMemoryPack()
        XCTAssertNil(actor.getMountedMemoryPackInfo())
        XCTAssertEqual(actor.getConfig().kvMode, .f16)
        XCTAssertEqual(actor.getConfig().maxContext, 4096)
    }

    func testAutoMountFromAgentSystemPromptPath() async throws {
        let fm = FileManager.default
        let tempRoot = fm.temporaryDirectory.appendingPathComponent("qwg_srv_auto_\(UUID().uuidString)", isDirectory: true)
        let subDir = tempRoot.appendingPathComponent("Sources/Sub", isDirectory: true)
        try fm.createDirectory(at: subDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tempRoot) }

        let qwgDir = QWMemoryManifest.qwgDirectoryURL(in: tempRoot)
        try fm.createDirectory(at: qwgDir, withIntermediateDirectories: true)
        let cacheFile = qwgDir.appendingPathComponent("auto_test-model_q4.qwcache")
        try Data([0x51, 0x57, 0x43, 0x48]).write(to: cacheFile)

        let artifact = QWMemoryManifest.CacheArtifact(
            modelName: "test-model",
            geometryDigest: QWModelDescriptor.qwen38_27b.geometryDigestV2Hex,
            kvMode: "q4",
            lsaMode: "auto",
            tokenCount: 2048,
            cacheFileName: cacheFile.lastPathComponent,
            cacheFileBytes: 1_048_576
        )
        let manifest = QWMemoryManifest(
            packName: "auto-repo",
            rootPath: tempRoot.path,
            files: [],
            caches: [artifact]
        )
        try manifest.save(to: tempRoot)

        let (tok, tpl) = try makeMinimalTokenizerAndTemplate()
        let stubTokens = Array(repeating: 3, count: 2048)
        let initialConfig = QWEngineLifecycleActor.Config(
            packPath: "/dummy/Qwen3.8-27B-hq4g32.qwp",
            modelName: "test-model",
            maxContext: 4096,
            kvMode: .f16,
            host: "127.0.0.1",
            port: 0
        )

        let actor = QWEngineLifecycleActor(
            config: initialConfig,
            logger: QWServerLogger(level: .none),
            load: { cfg, _, _ in
                let backend = MockMemoryBackend(config: cfg, stubTokens: stubTokens, stubBytes: 1_048_576)
                return QWEngineLifecycleActor.LoadedState(modelBackend: backend, tokenizer: tok, chatTemplate: tpl)
            }
        )

        let req = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data("""
            {
              "model": "test-model",
              "messages": [
                { "role": "system", "content": "Workspace directory: \(subDir.path)" },
                { "role": "user", "content": "Hello" }
              ]
            }
            """.utf8)
        )

        let prefixTokens = await actor.autoMountAndPrepareMemoryPrefix(for: req)
        XCTAssertEqual(prefixTokens.count, 2048)
        XCTAssertEqual(actor.getConfig().kvMode, .q4)
        XCTAssertEqual(actor.getMountedMemoryPackInfo()?.packName, "auto-repo")
        XCTAssertEqual(actor.getMountedMemoryPackInfo()?.enforcedKVMode, "q4")
    }

    func testDeterministicMultiTurnVaultRecallPreservesTurn1PrefixInChatTurnPreparer() throws {
        let fm = FileManager.default
        let tempRoot = fm.temporaryDirectory.appendingPathComponent("qwg_srv_vault_\(UUID().uuidString)", isDirectory: true)
        let srcDir = tempRoot.appendingPathComponent("Sources", isDirectory: true)
        try fm.createDirectory(at: srcDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tempRoot) }

        try "public struct VaultAlpha { public func runAlpha() {} }\n"
            .write(to: srcDir.appendingPathComponent("VaultAlpha.swift"), atomically: true, encoding: .utf8)
        try "public struct VaultBeta { public func runBeta() {} }\n"
            .write(to: srcDir.appendingPathComponent("VaultBeta.swift"), atomically: true, encoding: .utf8)

        let manifest = QWMemoryManifest(
            packName: "vault-repo",
            rootPath: tempRoot.path,
            files: [
                QWMemoryManifest.FileEntry(
                    relativePath: "Sources/VaultAlpha.swift",
                    tier: 1,
                    sha256: "a",
                    mtimeEpochSec: 1,
                    byteCount: 56,
                    tokenOffset: -1,
                    tokenCount: 20,
                    inCortex: false,
                    symbols: ["VaultAlpha", "runAlpha"]
                ),
                QWMemoryManifest.FileEntry(
                    relativePath: "Sources/VaultBeta.swift",
                    tier: 1,
                    sha256: "b",
                    mtimeEpochSec: 1,
                    byteCount: 54,
                    tokenOffset: -1,
                    tokenCount: 20,
                    inCortex: false,
                    symbols: ["VaultBeta", "runBeta"]
                )
            ]
        )

        let (tok, tpl) = try makeMinimalTokenizerAndTemplate()
        let cfg = QWEngineLifecycleActor.Config(
            packPath: "/dummy/Qwen3.8-27B-hq4g32.qwp",
            modelName: "test-model",
            maxContext: 32768,
            kvMode: .q4,
            host: "127.0.0.1",
            port: 0
        )

        let turn1Req = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data("""
            {
              "model": "test-model",
              "messages": [
                { "role": "user", "content": "Explain runAlpha in VaultAlpha.swift" }
              ]
            }
            """.utf8)
        )
        let prep1 = try QWChatTurnPreparer.preparePrompt(
            request: turn1Req,
            config: cfg,
            chatTemplate: tpl,
            tokenizer: tok,
            descriptor: .qwen38_27b,
            supportsTokenConstraints: false,
            memoryManifest: manifest
        )
        XCTAssertTrue(prep1.wire.contains("[VAULT FILE: Sources/VaultAlpha.swift]"))
        XCTAssertFalse(prep1.wire.contains("[VAULT FILE: Sources/VaultBeta.swift]"))

        let turn2Req = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data("""
            {
              "model": "test-model",
              "messages": [
                { "role": "user", "content": "Explain runAlpha in VaultAlpha.swift" },
                { "role": "assistant", "content": "runAlpha is a method on VaultAlpha." },
                { "role": "user", "content": "Now explain runBeta in VaultBeta.swift" }
              ]
            }
            """.utf8)
        )
        let prep2 = try QWChatTurnPreparer.preparePrompt(
            request: turn2Req,
            config: cfg,
            chatTemplate: tpl,
            tokenizer: tok,
            descriptor: .qwen38_27b,
            supportsTokenConstraints: false,
            memoryManifest: manifest
        )
        XCTAssertTrue(prep2.wire.hasPrefix(prep1.wire))
        XCTAssertTrue(prep2.wire.contains("[VAULT FILE: Sources/VaultBeta.swift]"))

        // Simulate the agent patching VaultAlpha.swift on disk mid-session;
        // Turn 1's historical prefix in Turn 3 MUST remain 100% byte-identical.
        try "public struct VaultAlpha { public func runAlpha() { print(42) } }\n"
            .write(to: srcDir.appendingPathComponent("VaultAlpha.swift"), atomically: true, encoding: .utf8)

        let turn3Req = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data("""
            {
              "model": "test-model",
              "messages": [
                { "role": "user", "content": "Explain runAlpha in VaultAlpha.swift" },
                { "role": "assistant", "content": "runAlpha is a method on VaultAlpha." },
                { "role": "user", "content": "Now explain runBeta in VaultBeta.swift" },
                {
                  "role": "assistant",
                  "content": "Checking files.",
                  "tool_calls": [
                    {
                      "id": "call_1",
                      "type": "function",
                      "function": {
                        "name": "patch",
                        "arguments": "{\\"path\\": \\"Sources/VaultAlpha.swift\\"}"
                      }
                    }
                  ]
                },
                { "role": "tool", "tool_call_id": "call_1", "content": "Patched VaultAlpha.swift" }
              ]
            }
            """.utf8)
        )
        let prep3 = try QWChatTurnPreparer.preparePrompt(
            request: turn3Req,
            config: cfg,
            chatTemplate: tpl,
            tokenizer: tok,
            descriptor: .qwen38_27b,
            supportsTokenConstraints: false,
            memoryManifest: manifest
        )
        XCTAssertTrue(prep3.wire.hasPrefix(prep1.wire))
        XCTAssertTrue(prep3.wire.hasPrefix(prep2.wire))
        XCTAssertFalse(prep3.wire.contains("<|memory_delta|>"))
    }

    func testAutonomousAgentToolStepRecallsVaultFileWhenUserPromptIsGeneric() throws {
        let fm = FileManager.default
        let tempRoot = fm.temporaryDirectory.appendingPathComponent("qwg_srv_tool_vault_\(UUID().uuidString)", isDirectory: true)
        let srcDir = tempRoot.appendingPathComponent("Sources", isDirectory: true)
        try fm.createDirectory(at: srcDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tempRoot) }

        try "public struct DriverResidency { public static func estimateContextMemory() -> Int { 1024 } }\n"
            .write(to: srcDir.appendingPathComponent("DriverResidency.swift"), atomically: true, encoding: .utf8)

        let manifest = QWMemoryManifest(
            packName: "tool-vault-repo",
            rootPath: tempRoot.path,
            files: [
                QWMemoryManifest.FileEntry(
                    relativePath: "Sources/DriverResidency.swift",
                    tier: 1,
                    sha256: "drv1",
                    mtimeEpochSec: 1,
                    byteCount: 93,
                    tokenOffset: -1,
                    tokenCount: 28,
                    inCortex: false,
                    symbols: ["DriverResidency", "estimateContextMemory"]
                )
            ]
        )

        let (tok, tpl) = try makeMinimalTokenizerAndTemplate()
        let cfg = QWEngineLifecycleActor.Config(
            packPath: "/dummy/Qwen3.8-27B-hq4g32.qwp",
            modelName: "test-model",
            maxContext: 32768,
            kvMode: .q4,
            host: "127.0.0.1",
            port: 0
        )

        let req = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data("""
            {
              "model": "test-model",
              "messages": [
                { "role": "user", "content": "some new commits landed verify and continue where interrupted" },
                {
                  "role": "assistant",
                  "content": "Checking estimateContextMemory in DriverResidency.swift",
                  "tool_calls": [
                    {
                      "id": "call_grep",
                      "type": "function",
                      "function": {
                        "name": "terminal",
                        "arguments": "{\\"command\\": \\"git status\\"}"
                      }
                    }
                  ]
                },
                { "role": "tool", "tool_call_id": "call_grep", "content": "On branch master" }
              ]
            }
            """.utf8)
        )
        let prepared = try QWChatTurnPreparer.preparePrompt(
            request: req,
            config: cfg,
            chatTemplate: tpl,
            tokenizer: tok,
            descriptor: .qwen38_27b,
            supportsTokenConstraints: false,
            memoryManifest: manifest
        )
        XCTAssertTrue(prepared.wire.contains("[VAULT FILE: Sources/DriverResidency.swift]"))
        XCTAssertEqual(prepared.messages.last?.get("tool_call_id"), .string("call_grep"))
    }
}
