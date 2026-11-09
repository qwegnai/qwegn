import XCTest
@testable import QWCore

final class QWModelCatalogTests: XCTestCase {
    private func tempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("qwcat-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func testFolderWithoutConfigIsUnsupportedWithReason() throws {
        let root = try tempDir()
        try FileManager.default.createDirectory(at: root.appendingPathComponent("model-a"), withIntermediateDirectories: true)
        let entries = QWModelCatalog.scan(directory: root.path)
        XCTAssertEqual(entries.map(\.name), ["model-a"])
        XCTAssertFalse(entries[0].supported)
        XCTAssertEqual(entries[0].reason, "no config.json")
    }

    func testUnregisteredModelTypeIsNotSilentlyFlashNext() throws {
        let root = try tempDir()
        let assets = root.appendingPathComponent("drafter/assets")
        try FileManager.default.createDirectory(at: assets, withIntermediateDirectories: true)
        try Data(#"{"model_type":"qwen3","num_hidden_layers":5}"#.utf8).write(to: assets.appendingPathComponent("config.json"))
        let entry = QWModelCatalog.classify(name: "drafter", path: root.appendingPathComponent("drafter").path)
        XCTAssertFalse(entry.supported)
        XCTAssertNil(entry.family)
        XCTAssertEqual(entry.modelType, "qwen3")
    }

    func testUnknownModelTypeThrowsInDescriptor() {
        let json = Data(#"{"model_type":"not_a_real_family"}"#.utf8)
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: json))
    }

    func testInstalledModelsAreClassifiedFromTheirConfigs() throws {
        // A stub model tree, not the user's ~/.qwg/models: the unit suite must
        // run on a machine with no models installed.
        let root = try tempDir()
        let dense = root.appendingPathComponent("dense-stub")
        let moe = root.appendingPathComponent("moe-stub")
        for dir in [dense, moe] {
            try FileManager.default.createDirectory(at: dir.appendingPathComponent("assets"),
                                                    withIntermediateDirectories: true)
        }
        try QWStubModelConfig.dense27BJSON()
            .write(to: dense.appendingPathComponent("assets/config.json"))
        try QWStubModelConfig.flashNextJSON()
            .write(to: moe.appendingPathComponent("assets/config.json"))
        let entries = QWModelCatalog.scan(directory: root.path)
        XCTAssertEqual(entries.count, 2)
        for entry in entries where entry.supported {
            XCTAssertNotNil(entry.family, entry.name)
            XCTAssertNotNil(entry.nativeWindow, entry.name)
        }
        let byName = Dictionary(uniqueKeysWithValues: entries.map { ($0.name, $0) })
        XCTAssertEqual(byName["dense-stub"]?.family, .qwen38Dense)
        XCTAssertEqual(byName["moe-stub"]?.family, .qwen38FlashNext)
        XCTAssertEqual(byName["dense-stub"]?.nativeWindow, 262144)
    }

    func testDFlashDrafterModelIsRejectedAsPrimaryModel() throws {
        let root = try tempDir()
        let dflashDir = root.appendingPathComponent("Qwen3.8-27B-DFlash2-hq4g32")
        try FileManager.default.createDirectory(at: dflashDir.appendingPathComponent("assets"),
                                                withIntermediateDirectories: true)
        let dflashConfig = Data("""
        {
          "architectures": ["DFlash2DraftModel"],
          "model_type": "qwen3",
          "hidden_size": 5120,
          "intermediate_size": 17408,
          "num_hidden_layers": 5,
          "num_attention_heads": 40,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "vocab_size": 248320,
          "max_position_embeddings": 262144,
          "rms_norm_eps": 1e-6,
          "dflash_config": {
            "mask_token_id": 248070,
            "block_size": 8,
            "target_layer_ids": [5, 19, 33, 47, 61]
          }
        }
        """.utf8)
        try dflashConfig.write(to: dflashDir.appendingPathComponent("assets/config.json"))

        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: dflashConfig)) { error in
            guard case QWModelDescriptor.DescriptorError.drafterModel(let msg) = error else {
                XCTFail("Expected DescriptorError.drafterModel, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("DFlash2 drafter model"), "Got: \(msg)")
        }

        let entry = QWModelCatalog.classify(name: "Qwen3.8-27B-DFlash2-hq4g32", path: dflashDir.path)
        XCTAssertFalse(entry.supported)
        XCTAssertNil(entry.family)
        XCTAssertTrue(entry.reason?.lowercased().contains("drafter model") == true, "Got reason: \(String(describing: entry.reason))")
    }
}
