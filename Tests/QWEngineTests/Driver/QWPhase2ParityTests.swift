import XCTest
import Foundation
@testable import QWCore
@testable import QWQuant
@testable import QWEngine

/// The Phase 2 contract in one place: the hybrid models must not move, and
/// the new schemas must decode.
final class QWPhase2ParityTests: XCTestCase {
    static let fixtures = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().appendingPathComponent("Fixtures")

    /// A flat standard config for the family, with the shape the decoder needs.
    func config(_ extra: [String: Any]) -> Data {
        var base: [String: Any] = [
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 4096,
            "vocab_size": 151936,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 12288,
            "max_position_embeddings": 40960,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "eos_token_id": 151645,
            "tie_word_embeddings": false,
        ]
        for (k, v) in extra { base[k] = v }
        return try! JSONSerialization.data(withJSONObject: base)
    }

    func testTheHybridFamiliesAreUnchanged() throws {
        for type in ["qwen3_5", "qwen3_5_text", "qwen4_exp", "qwen4_exp_text"] {
            XCTAssertNotNil(QWModelFamilyRegistry.shared.spec(forModelType: type),
                            "\(type) must still be registered")
        }
    }

    func testTheStandardFamiliesDecode() throws {
        // Each family is decoded by the **standard** decoder, not just
        // looked up in the registry: a family that is registered but cannot
        // decode is not served.
        let families: [(String, [String: Any])] = [
            ("qwen3", [:]),
            ("qwen3_moe", ["num_experts": 64, "num_experts_per_tok": 8,
                            "moe_intermediate_size": 768]),
            ("qwen2", [:]),
            ("llama", [:]),
            ("mistral", [:]),
        ]
        for (type, extra) in families {
            var fields = extra
            fields["model_type"] = type
            let decoded = try? QWStandardConfigDecoder.decode(config(fields))
            XCTAssertNotNil(decoded, "\(type) must decode")
            XCTAssertNotNil(QWModelFamilyRegistry.shared.spec(forModelType: type),
                            "\(type) must be registered")
        }
    }

    func testEveryStandardFamilyHasAModule() {
        for family in [QWModelDescriptor.Family.qwen3Dense,
                       .qwen3MoE, .qwen2Dense, .llamaDense, .mistralDense] {
            XCTAssertNotNil(QWModuleRegistry.moduleType(for: family),
                            "\(family.rawValue) must have a module")
        }
    }

    func testTheStandardMoEKernelIsServed() {
        let key = QWEnginePlan.LayerKernelKey(attention: .fullAttention, mlp: .moe, ple: false)
        XCTAssertNotNil(QWLayerKernelRegistry.kernel(for: key))
    }

    func testTheWeightSourcePreferenceIsHonoured() throws {
        // The preference decides which container the engine reads, so the
        // resolution is asserted — not the number of enum cases.
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-pref-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data([0]).write(to: dir.appendingPathComponent("model.qwp"))
        try Data([0]).write(to: dir.appendingPathComponent("model.safetensors"))

        let auto = QWModelRoot(url: dir, preference: .auto)
        XCTAssertEqual(auto.kind, .qwp1, "auto prefers the packed container")
        XCTAssertEqual(auto.weights.lastPathComponent, "model.qwp")
        XCTAssertEqual(QWModelRoot(url: dir, preference: .qwp).kind, .qwp1)
        let safetensors = QWModelRoot(url: dir, preference: .safetensors)
        XCTAssertEqual(safetensors.kind, .safetensorsSingle,
                       "the safetensors preference must not take the pack")
        XCTAssertEqual(safetensors.weights.lastPathComponent, "model.safetensors")

        // A preference the directory cannot satisfy names no container.
        let onlyPack = dir.appendingPathComponent("only")
        try FileManager.default.createDirectory(at: onlyPack, withIntermediateDirectories: true)
        try Data([0]).write(to: onlyPack.appendingPathComponent("model.qwp"))
        XCTAssertNil(QWModelRoot(url: onlyPack, preference: .safetensors).kind,
                     "the safetensors preference must refuse a pack-only directory")
    }

    func testTheQNGRAMSizeLawHolds() throws {
        // Q8_G32: 24 rows per 4096-byte sector. Q4_G32: 40.
        XCTAssertEqual(try QWNgramFormat.rowsPerSector(rowBytes: 170), 24)
        XCTAssertEqual(try QWNgramFormat.rowsPerSector(rowBytes: 100), 40)
    }

    func testTheQwen3FamiliesAreOfficiallySupported() throws {
        // The flag is a runnability claim, and the audit behind it is in
        // the repo: the published-config test decides every checkpoint.
        for type in ["qwen3", "qwen3_moe"] {
            let spec = try XCTUnwrap(QWModelFamilyRegistry.shared.spec(forModelType: type))
            XCTAssertTrue(spec.isOfficialSupported, "\(type) must carry the support flag")
        }
        // Mistral waits for the sliding window the engine has not got.
        let mistral = try XCTUnwrap(QWModelFamilyRegistry.shared.spec(forModelType: "mistral"))
        XCTAssertFalse(mistral.isOfficialSupported, "mistral waits for the window")
    }

    func testTheSparseStandardModuleDeclaresTheRouterDoor() {
        // The router is the sparse path's own door. This is about the
        // **standard** path: the hybrid dense module carries the router
        // door for the 27B's own MoE layers, and that is not in question.
        XCTAssertTrue(Qwen3MoEModule.requiredDoors.contains("qwg_router_gemm"))
        XCTAssertFalse(Qwen3DenseModule.requiredDoors.contains("qwg_router_gemm"))
        XCTAssertFalse(StandardModule.requiredDoors.contains("qwg_router_gemm"))
    }

    func testTheStandardDecoderRefusesTheFiveNamedFeatures() throws {
        // A key that changes what the engine runs is refused by name, and
        // honoured when it asks for what the engine does.
        let refused: [(String, Any, String)] = [
            ("architectures", ["Qwen3ForGuardModel"], "Qwen3ForGuardModel"),
            ("auto_map", "modeling_qwen3.MyModel", "auto_map"),
            ("use_qk_norm", false, "use_qk_norm"),
            ("qkv_bias", true, "qkv_bias"),
            ("output_router_logits", true, "output_router_logits"),
        ]
        for (key, value, named) in refused {
            XCTAssertThrowsError(try QWStandardConfigDecoder.decode(config([key: value])),
                                 "\(key) must be refused by name") { error in
                XCTAssertTrue("\(error)".contains(named),
                              "\(key) was refused, but not for '\(named)': \(error)")
            }
        }
        // The same keys, asked for in the way the engine serves them.
        XCTAssertNoThrow(try QWStandardConfigDecoder.decode(config([:])))
        XCTAssertNoThrow(try QWStandardConfigDecoder.decode(config(["use_qk_norm": true])))
        XCTAssertNoThrow(try QWStandardConfigDecoder.decode(config(["qkv_bias": false])))
    }
}
