// QWBackendFactsTests.swift — Unit tests for QWBackendFacts.headStorageLabel across pack formats.

import Foundation
import QWCore
import QWEngine
@testable import QWServer
import XCTest

final class QWBackendFactsTests: XCTestCase {
    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-backend-facts-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func writeJSON(_ obj: Any, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
        try data.write(to: url)
    }

    private func makeQWP1Pack(at dir: URL, headEntry: [String: Any]?) throws {
        let config: [String: Any] = [
            "model_type": "qwen3",
            "vocab_size": 151_936,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "eos_token_id": 151_645,
        ]
        let assetsDir = dir.appendingPathComponent("assets", isDirectory: true)
        try FileManager.default.createDirectory(at: assetsDir, withIntermediateDirectories: true)
        let cfgURL = assetsDir.appendingPathComponent("config.json")
        try writeJSON(config, to: cfgURL)
        let desc = try QWModelDescriptor.decode(configJSON: Data(contentsOf: cfgURL))

        var tensors: [String: Any] = [:]
        if let headEntry {
            tensors["lm_head.weight"] = headEntry
        }
        let header: [String: Any] = [
            "provenance": [
                "model_id": desc.family.rawValue,
                "geometry_digest": desc.geometryDigestV3Hex,
            ],
            "tensors": tensors,
        ]
        let hdrData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var file = Data("QWP1".utf8)
        var hdrLen = UInt64(hdrData.count).littleEndian
        withUnsafeBytes(of: &hdrLen) { (raw: UnsafeRawBufferPointer) in
            file.append(contentsOf: raw)
        }
        file.append(Data(count: 64))
        file.append(hdrData)
        try file.write(to: dir.appendingPathComponent("model.qwp"))
    }

    func testHeadStorageLabelAcrossQuantAndRawPackEntries() throws {
        let root = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }

        let cases: [(name: String, entry: [String: Any]?, expected: String)] = [
            ("shvq3", ["shape": [151_936, 256], "hq4_off": 64, "hq": ["bits": 3, "group": 64, "vq": "shvq1"]], "SHVQ3_G64"),
            ("aq4", ["shape": [151_936, 256], "hq4_off": 64, "hq": ["bits": 4, "group": 32, "vq": "aq"]], "AQ4_G32"),
            ("hq8", ["shape": [151_936, 256], "hq4_off": 64, "hq": ["bits": 8, "group": 32]], "HQ8_G32"),
            ("hq4_explicit", ["shape": [151_936, 256], "hq4_off": 64, "hq": ["bits": 4, "group": 32]], "HQ4_G32"),
            ("hq4_incumbent", ["shape": [151_936, 256], "hq4_off": 64], "HQ4_G32_M5"),
            ("hq8_planar", ["shape": [151_936, 256], "hq8_off": 64, "hq8_s_off": 128], "HQ8_G32_M5"),
            ("bf16", ["shape": [151_936, 256], "raw_off": 64, "kind": "bf16"], "BF16 (verbatim)"),
            ("raw_wins_over_hq4", ["shape": [151_936, 256], "raw_off": 64, "hq4_off": 128], "BF16 (verbatim)"),
            ("missing_head", nil, "not found in the pack"),
        ]

        for item in cases {
            let dir = root.appendingPathComponent(item.name, isDirectory: true)
            try makeQWP1Pack(at: dir, headEntry: item.entry)
            XCTAssertEqual(QWBackendFacts.headStorageLabel(packPath: dir.path), item.expected, item.name)
        }

        let stDir = root.appendingPathComponent("safetensors_pack", isDirectory: true)
        try FileManager.default.createDirectory(at: stDir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: stDir.appendingPathComponent("model.safetensors"))
        XCTAssertEqual(QWBackendFacts.headStorageLabel(packPath: stDir.path), "HQ4_G32_M5 (online GPU)")

        let missingDir = root.appendingPathComponent("nonexistent", isDirectory: true)
        XCTAssertEqual(QWBackendFacts.headStorageLabel(packPath: missingDir.path), "pack metadata unreadable")
    }

    func testHeadStorageLabelPrefersEngineReportFormatAndFallsBackToPack() throws {
        let root = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let shvqDir = root.appendingPathComponent("shvq_pack", isDirectory: true)
        try makeQWP1Pack(
            at: shvqDir,
            headEntry: ["shape": [151_936, 256], "hq4_off": 64, "hq": ["bits": 4, "group": 32, "vq": "shvq1"]]
        )

        let nilReportEngine = StubHeadEngine(weightFormat: nil)
        XCTAssertEqual(
            QWBackendFacts.headStorageLabel(engine: nilReportEngine, packPath: shvqDir.path),
            "SHVQ4_G32"
        )
        let hq4Engine = StubHeadEngine(weightFormat: "HQ4_G32")
        XCTAssertEqual(
            QWBackendFacts.headStorageLabel(engine: hq4Engine, packPath: shvqDir.path),
            "HQ4_G32_M5 (online GPU)"
        )
        let shvqEngine = StubHeadEngine(weightFormat: "SHVQ4_G32")
        XCTAssertEqual(
            QWBackendFacts.headStorageLabel(engine: shvqEngine, packPath: shvqDir.path),
            "SHVQ4_G32 (native, GPU)"
        )
    }
}

private final class StubHeadEngine: QWInferenceEngine, @unchecked Sendable {
    let weightFormat: String?
    init(weightFormat: String?) { self.weightFormat = weightFormat }

    var capabilities: Set<QWKnobCapability> { [] }
    var hasPrefixCache: Bool { false }
    var prefixCache: QWPrefixCacheState { QWPrefixCacheState(isConfigured: false) }
    var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)?
    var report: QWEngineReport {
        QWEngineReport(
            capacityLog: "", arenaBytes: 0, weightUsedBytes: 0,
            weightReservedBytes: 0, contextCapacity: 128, weightFormat: weightFormat
        )
    }
    var lastTurn: QWGenerationResult? { nil }
    func makeConversation() -> QWConversation? { nil }
    func generate(request: QWGenerationRequest, onToken: ((Int) -> Bool)?) throws -> QWGenerationResult {
        QWGenerationResult(tokens: [], stepMs: [], stepTokenCounts: [])
    }
}
