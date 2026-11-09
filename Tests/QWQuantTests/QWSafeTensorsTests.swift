import XCTest
import Foundation
@testable import QWQuant

final class QWSafeTensorsTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qwg-st-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    /// Writes a minimal single-file safetensors: 8-byte header length + JSON header + payload.
    private func writeSafetensors(_ name: String, tensors: [(String, [Int], String, [UInt8])]) throws {
        var header: [String: Any] = [:]
        var payload = Data()
        for (tname, shape, dtype, bytes) in tensors {
            header[tname] = ["dtype": dtype, "shape": shape,
                             "data_offsets": [payload.count, payload.count + bytes.count]]
            payload.append(contentsOf: bytes)
        }
        let headerJSON = try JSONSerialization.data(withJSONObject: header)
        var out = Data()
        out.append(contentsOf: withUnsafeBytes(of: UInt64(headerJSON.count).littleEndian) { Array($0) })
        out.append(headerJSON)
        out.append(payload)
        try out.write(to: dir.appendingPathComponent(name))
    }

    /// Writes a safetensors file from a raw header and payload.
    private func writeRaw(_ name: String, headerJSON: String, payload: [UInt8]) throws {
        let header = Data(headerJSON.utf8)
        var out = Data()
        out.append(contentsOf: withUnsafeBytes(of: UInt64(header.count).littleEndian) { Array($0) })
        out.append(header)
        out.append(contentsOf: payload)
        try out.write(to: dir.appendingPathComponent(name))
    }

    func testSingleFileHeaderIsParsed() throws {
        try writeSafetensors("model.safetensors", tensors: [
            ("model.embed_tokens.weight", [4, 8], "F16", Array(repeating: 0, count: 64)),
            ("model.layers.0.self_attn.q_proj.weight", [8, 8], "F32", Array(repeating: 0, count: 256)),
        ])
        let c = try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors"))
        XCTAssertEqual(c.tensors.count, 2)
        let q = c.tensor("model.layers.0.self_attn.q_proj.weight")
        XCTAssertEqual(q?.dtype, .f32)
        XCTAssertEqual(q?.shape, [8, 8])
        XCTAssertEqual(q?.byteLength, 256)
        XCTAssertEqual(q?.rows, 8)
        XCTAssertEqual(q?.cols, 8)
    }

    func testShardIndexResolvesAcrossFiles() throws {
        try writeSafetensors("model-00001-of-00002.safetensors", tensors: [
            ("model.embed_tokens.weight", [4, 8], "F16", Array(repeating: 0, count: 64)),
        ])
        try writeSafetensors("model-00002-of-00002.safetensors", tensors: [
            ("model.layers.0.self_attn.q_proj.weight", [8, 8], "F32", Array(repeating: 0, count: 256)),
        ])
        let index: [String: Any] = ["metadata": ["total_size": 320],
                                    "weight_map": ["model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                                                   "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00002.safetensors"]]
        try JSONSerialization.data(withJSONObject: index)
            .write(to: dir.appendingPathComponent("model.safetensors.index.json"))

        let c = try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors.index.json"))
        XCTAssertNotNil(c.tensor("model.embed_tokens.weight"))
        XCTAssertNotNil(c.tensor("model.layers.0.self_attn.q_proj.weight"))
        XCTAssertEqual(c.shards.count, 2)
    }

    func testCanonicalNamesBridgeTheLanguageModelPrefix() throws {
        try writeSafetensors("model.safetensors", tensors: [
            ("model.language_model.layers.0.mlp.gate_proj.weight", [8, 8], "F16", Array(repeating: 0, count: 128)),
        ])
        let c = try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors"))
        XCTAssertNotNil(c.tensor("model.language_model.layers.0.mlp.gate_proj.weight"))
        XCTAssertNotNil(c.tensor("model.layers.0.mlp.gate_proj.weight"),
                        "the canonical short name must resolve too")
        // The alias resolves to the same bytes.
        let short = try c.withBytes(of: "model.layers.0.mlp.gate_proj.weight") { $0.count }
        let long = try c.withBytes(of: "model.language_model.layers.0.mlp.gate_proj.weight") { $0.count }
        XCTAssertEqual(short, long)
        // The tensor reports the file's own name, whichever way it is reached.
        XCTAssertEqual(c.tensor("model.layers.0.mlp.gate_proj.weight")?.name,
                       "model.language_model.layers.0.mlp.gate_proj.weight")
    }

    func testTheAliasesAreNotInTheTensorList() throws {
        // A consumer that loads every weight must not load one twice: the
        // canonical bridge is a LOOKUP convenience, not a second tensor.
        try writeSafetensors("model.safetensors", tensors: [
            ("model.language_model.layers.0.mlp.gate_proj.weight", [8, 8], "F16", Array(repeating: 0, count: 128)),
        ])
        let c = try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors"))
        XCTAssertEqual(c.tensors.count, 1)
        XCTAssertNotNil(c.tensors["model.language_model.layers.0.mlp.gate_proj.weight"])
        XCTAssertNil(c.tensors["model.layers.0.mlp.gate_proj.weight"])
    }

    func testAnAmbiguousCanonicalNameIsNotAliased() throws {
        // Two different tensors claim one canonical name. The alias is not
        // created, so the lookup fails loudly instead of returning whichever
        // tensor happened to come first out of a dictionary.
        try writeSafetensors("model.safetensors", tensors: [
            ("model.language_model.thing.weight", [2, 2], "F16", Array(repeating: 1, count: 8)),
            ("language_model.model.thing.weight", [2, 2], "F16", Array(repeating: 2, count: 8)),
        ])
        let c = try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors"))
        XCTAssertEqual(c.tensors.count, 2)
        XCTAssertNil(c.tensor("model.thing.weight"),
                     "an ambiguous canonical name must not resolve to a random tensor")
        // Both originals still resolve to their own bytes.
        let a = try c.withBytes(of: "model.language_model.thing.weight") { $0[0] }
        let b = try c.withBytes(of: "language_model.model.thing.weight") { $0[0] }
        XCTAssertEqual(a, 1)
        XCTAssertEqual(b, 2)
    }

    func testBytesCanBeReadZeroCopy() throws {
        let payload: [UInt8] = (0..<32).map { UInt8($0) }
        try writeSafetensors("model.safetensors", tensors: [("w", [4, 8], "I8", payload)])
        let c = try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors"))
        let first = try c.withBytes(of: "w") { $0[0] }
        XCTAssertEqual(first, 0)
        let last = try c.withBytes(of: "w") { $0[31] }
        XCTAssertEqual(last, 31)
    }

    func testBadHeaderFailsLoud() throws {
        try Data("not a safetensors file".utf8).write(to: dir.appendingPathComponent("model.safetensors"))
        XCTAssertThrowsError(try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors")))
    }

    // MARK: - The offset law and the fails

    func testTheBytesLandAfterTheHeader() throws {
        // The offset law: `data_offsets` are relative to the data region,
        // which starts at 8 + header length. Reading at 8+offset displaces
        // every tensor by the header size and raises no error.
        let payload: [UInt8] = (0..<16).map { UInt8(0xA0 + $0) }
        try writeSafetensors("model.safetensors", tensors: [("w", [4, 4], "I8", payload)])
        let c = try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors"))
        let bytes = try c.withBytes(of: "w") { Array($0) }
        XCTAssertEqual(bytes, payload)
        let range = try XCTUnwrap(c.byteRange(of: "w"))
        XCTAssertEqual(range.count, 16)
    }

    func testAnUnknownDTypeIsRefusedByName() throws {
        try writeSafetensors("model.safetensors", tensors: [("w", [4, 4], "I4", Array(repeating: 0, count: 16))])
        XCTAssertThrowsError(try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors"))) { error in
            XCTAssertEqual(error as? QWSafeTensorsError, .unsupportedDType("I4"))
        }
    }

    func testATensorThatRunsPastTheEndOfTheFileIsRefused() throws {
        // A header that points past EOF is a silent misread waiting to
        // happen, so it is refused when the header is parsed.
        let header: [String: Any] = [
            "w": ["dtype": "I8", "shape": [4, 4], "data_offsets": [0, 4096]],
        ]
        let headerJSON = try JSONSerialization.data(withJSONObject: header)
        var out = Data()
        out.append(contentsOf: withUnsafeBytes(of: UInt64(headerJSON.count).littleEndian) { Array($0) })
        out.append(headerJSON)
        out.append(Data(repeating: 0, count: 16))
        try out.write(to: dir.appendingPathComponent("model.safetensors"))
        XCTAssertThrowsError(try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors")))
    }

    func testATensorWhoseBytesDisagreeWithItsShapeIsRefused() throws {
        // 32 bytes for a [4,4] I8 tensor is 16 elements — the header lies.
        try writeSafetensors("model.safetensors", tensors: [("w", [4, 4], "I8", Array(repeating: 0, count: 32))])
        XCTAssertThrowsError(try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors")))
    }

    func testAMissingTensorIsNamed() throws {
        try writeSafetensors("model.safetensors", tensors: [("w", [4, 4], "I8", Array(repeating: 0, count: 16))])
        let c = try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors"))
        XCTAssertThrowsError(try c.withBytes(of: "nope") { $0.count }) { error in
            XCTAssertEqual(error as? QWSafeTensorsError, .missingTensor("nope"))
        }
    }

    // MARK: - A hostile header must throw, never trap

    func testAnAbsurdHeaderLengthThrowsInsteadOfTrapping() throws {
        // UInt64.max as the header length. `Int(headerLength)` would trap.
        let url = dir.appendingPathComponent("model.safetensors")
        var out = Data()
        out.append(contentsOf: withUnsafeBytes(of: UInt64.max) { Array($0) })
        out.append(Data("{".utf8))
        try out.write(to: url)
        XCTAssertThrowsError(try QWSafeTensorsContainer(url: url))
    }

    func testAShapeThatOverflowsThrowsInsteadOfTrapping() throws {
        // The element count would overflow Int64. The multiplication must be
        // checked, not allowed to trap.
        try writeRaw("model.safetensors",
                     headerJSON: "{\"w\":{\"dtype\":\"I32\",\"shape\":[2,9223372036854775807],\"data_offsets\":[0,16]}}",
                     payload: Array(repeating: 0, count: 16))
        XCTAssertThrowsError(try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors")))
    }

    func testADimensionThatOverflowsThrowsInsteadOfTrapping() throws {
        try writeRaw("model.safetensors",
                     headerJSON: "{\"w\":{\"dtype\":\"I32\",\"shape\":[9223372036854775807,4],\"data_offsets\":[0,16]}}",
                     payload: Array(repeating: 0, count: 16))
        XCTAssertThrowsError(try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors")))
    }

    func testAHeaderThatIsNotAnObjectIsRefused() throws {
        // Valid JSON, but not an object: it must not become an empty
        // checkpoint that reports every tensor as missing.
        try writeRaw("model.safetensors", headerJSON: "[1,2,3]", payload: [])
        XCTAssertThrowsError(try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors")))
    }

    func testOverlappingTensorRangesAreRefused() throws {
        // Two tensors claiming the same four bytes is a silent misread.
        try writeRaw("model.safetensors",
                     headerJSON: "{\"a\":{\"dtype\":\"I8\",\"shape\":[4],\"data_offsets\":[0,4]},"
                                + "\"b\":{\"dtype\":\"I8\",\"shape\":[4],\"data_offsets\":[0,4]}}",
                     payload: Array(repeating: 0, count: 4))
        XCTAssertThrowsError(try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors")))
    }

    func testATensorThatLeavesAHoleIsRefused() throws {
        try writeRaw("model.safetensors",
                     headerJSON: "{\"a\":{\"dtype\":\"I8\",\"shape\":[4],\"data_offsets\":[0,4]},"
                                + "\"b\":{\"dtype\":\"I8\",\"shape\":[4],\"data_offsets\":[8,12]}}",
                     payload: Array(repeating: 0, count: 12))
        XCTAssertThrowsError(try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors")))
    }

    func testTrailingBytesPastTheDataRegionAreRefused() throws {
        try writeRaw("model.safetensors",
                     headerJSON: "{\"a\":{\"dtype\":\"I8\",\"shape\":[4],\"data_offsets\":[0,4]}}",
                     payload: Array(repeating: 0, count: 8))
        XCTAssertThrowsError(try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors")))
    }

    func testATensorTheIndexNamesButTheShardLacksIsRefused() throws {
        // The index and the shard disagree: that is a corrupt checkpoint, and
        // it must not look like a complete one.
        try writeSafetensors("shard.safetensors", tensors: [
            ("present", [4, 4], "I8", Array(repeating: 0, count: 16)),
        ])
        let index: [String: Any] = ["weight_map": ["present": "shard.safetensors",
                                                    "ghost": "shard.safetensors"]]
        try JSONSerialization.data(withJSONObject: index)
            .write(to: dir.appendingPathComponent("model.safetensors.index.json"))
        XCTAssertThrowsError(try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors.index.json")))
    }

    // MARK: - The dtypes the spec defines

    func testTheSpecDTypesAreAllKnown() throws {
        // The whole safetensors dtype list. A missing one would refuse a
        // whole real checkpoint — the U32 case is a real MLX checkpoint.
        let spec = ["BOOL", "F4", "F6_E2M3", "F6_E3M2", "U8", "I8", "F8_E5M2", "F8_E4M3",
                    "F8_E8M0", "F8_E4M3FNUZ", "F8_E5M2FNUZ", "I16", "U16", "F16", "BF16",
                    "I32", "U32", "F32", "C64", "F64", "I64", "U64"]
        for raw in spec {
            XCTAssertNotNil(QWSafeTensorsDType(rawValue: raw), raw)
        }
        XCTAssertEqual(QWSafeTensorsDType.bf16.bitSize, 16)
        XCTAssertEqual(QWSafeTensorsDType.f32.bitSize, 32)
        XCTAssertEqual(QWSafeTensorsDType.f64.bitSize, 64)
        XCTAssertEqual(QWSafeTensorsDType.u32.bitSize, 32)
        XCTAssertEqual(QWSafeTensorsDType.f4.bitSize, 4)
        XCTAssertEqual(QWSafeTensorsDType.f6E2M3.bitSize, 6)
    }

    func testAU32TensorIsRead() throws {
        try writeSafetensors("model.safetensors", tensors: [
            ("w", [2, 2], "U32", Array(repeating: 7, count: 16)),
        ])
        let c = try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors"))
        XCTAssertEqual(c.tensor("w")?.dtype, .u32)
        let v = try c.withBytes(of: "w") { raw -> UInt32 in raw.loadUnaligned(as: UInt32.self) }
        XCTAssertEqual(v, 0x0707_0707)          // four 0x07 bytes, little endian
    }

    func testASubByteDTypeIsMeasuredInBits() throws {
        // F4 is 4 bits per element: eight elements are four bytes.
        try writeSafetensors("model.safetensors", tensors: [
            ("w", [8], "F4", Array(repeating: 0, count: 4)),
        ])
        let c = try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors"))
        XCTAssertEqual(c.tensor("w")?.dtype, .f4)
        XCTAssertEqual(c.tensor("w")?.byteLength, 4)
    }

    // MARK: - The shapes the spec allows

    func testAZeroDimensionalTensorIsRead() throws {
        // shape [] is legal: one element of the dtype.
        try writeSafetensors("model.safetensors", tensors: [
            ("w", [], "I64", Array(repeating: 0, count: 8)),
        ])
        let c = try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors"))
        XCTAssertEqual(c.tensor("w")?.shape, [])
        XCTAssertEqual(c.tensor("w")?.byteLength, 8)
        XCTAssertEqual(c.tensor("w")?.elementCount, 1)
    }

    func testAZeroSizedTensorIsRead() throws {
        try writeSafetensors("model.safetensors", tensors: [
            ("w", [2, 0], "I32", []),
        ])
        let c = try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors"))
        XCTAssertEqual(c.tensor("w")?.shape, [2, 0])
        XCTAssertEqual(c.tensor("w")?.byteLength, 0)
    }

    // MARK: - The sharded offset law

    func testTheShardedOffsetLawLandsOnTheRightBytes() throws {
        // Each shard has its own header length, so each shard's data region
        // starts somewhere else. This is the 93 KB bug's exact shape.
        let aPayload: [UInt8] = (0..<8).map { UInt8(0xA0 + $0) }
        let bPayload: [UInt8] = (0..<8).map { UInt8(0xB0 + $0) }
        try writeSafetensors("model-00001-of-00002.safetensors", tensors: [
            ("a", [8], "I8", aPayload),
        ])
        try writeSafetensors("model-00002-of-00002.safetensors", tensors: [
            ("b", [8], "I8", bPayload),
        ])
        let index: [String: Any] = ["weight_map": ["a": "model-00001-of-00002.safetensors",
                                                    "b": "model-00002-of-00002.safetensors"]]
        try JSONSerialization.data(withJSONObject: index)
            .write(to: dir.appendingPathComponent("model.safetensors.index.json"))
        let c = try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors.index.json"))
        XCTAssertEqual(try c.withBytes(of: "a") { Array($0) }, aPayload)
        XCTAssertEqual(try c.withBytes(of: "b") { Array($0) }, bPayload)
        XCTAssertNotEqual(c.shardURL(of: "a"), c.shardURL(of: "b"))
    }
}
