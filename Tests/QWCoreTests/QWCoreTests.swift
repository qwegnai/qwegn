import XCTest
@testable import QWCore

/// C01 gate (specs/01-qwcore.md):
///  (a) derived channel counts re-derive from head counts,
///  (b) geometry matches the committed HF-config golden,
final class QWCoreTests: XCTestCase {

    func goldenURL() -> URL? {
        var dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<8 {
            let cand = dir.appendingPathComponent("Scripts/oracle/geometry_golden.json")
            if FileManager.default.fileExists(atPath: cand.path) { return cand }
            dir = dir.deletingLastPathComponent()
        }
        return nil
    }

    func testDerivedIdentities() {
        XCTAssertEqual(QWModelGeometry.qwen38_27b.verifyInternalConsistency(), [],
                       "derived constants must re-derive from head counts")
    }

    func testMatchesHFGolden() throws {
        guard let url = goldenURL() else {
            return XCTFail("geometry_golden.json not reachable from test source tree")
        }
        let data = try Data(contentsOf: url)
        let g = try JSONDecoder().decode([String: JSONValue].self, from: data)
        func i(_ k: String) -> Int { g[k]?.intValue ?? -1 }
        XCTAssertEqual(i("hidden"), QWModelGeometry.qwen38_27b.hidden)
        XCTAssertEqual(i("vocab"), QWModelGeometry.qwen38_27b.vocab)
        XCTAssertEqual(i("intermediate"), QWModelGeometry.qwen38_27b.intermediate)
        XCTAssertEqual(i("num_layers"), QWModelGeometry.qwen38_27b.numLayers)
        XCTAssertEqual(i("full_attention_interval"), QWModelGeometry.qwen38_27b.fullAttentionInterval)
        XCTAssertEqual(i("q_heads"), QWModelGeometry.qwen38_27b.qHeads)
        XCTAssertEqual(i("kv_heads"), QWModelGeometry.qwen38_27b.kvHeads)
        XCTAssertEqual(i("head_dim"), QWModelGeometry.qwen38_27b.headDim)
        XCTAssertEqual(i("q_proj_out"), QWModelGeometry.qwen38_27b.qProjOut)
        XCTAssertEqual(i("rotary_dims"), QWModelGeometry.qwen38_27b.rotaryDims)
        XCTAssertEqual(i("max_position_embeddings"), QWModelGeometry.qwen38_27b.maxPositionEmbeddings)
        XCTAssertEqual(i("gdn_qk_heads"), QWModelGeometry.qwen38_27b.gdnQKHeads)
        XCTAssertEqual(i("gdn_v_heads"), QWModelGeometry.qwen38_27b.gdnVHeads)
        XCTAssertEqual(i("gdn_head_dim"), QWModelGeometry.qwen38_27b.gdnHeadDim)
        XCTAssertEqual(i("gdn_qkv_channels"), QWModelGeometry.qwen38_27b.gdnQKVChannels)
        XCTAssertEqual(i("gdn_z_channels"), QWModelGeometry.qwen38_27b.gdnZChannels)
        XCTAssertEqual(i("gdn_conv_kernel"), QWModelGeometry.qwen38_27b.gdnConvKernel)
        XCTAssertEqual(i("num_mtp_layers"), QWModelGeometry.qwen38_27b.mtpLayers)
        XCTAssertEqual(g["mamba_ssm_dtype"]?.stringValue, "float32")
        XCTAssertEqual(g["attn_output_gate"]?.boolValue, true)
        XCTAssertEqual(g["mrope_section"]?.arrayValue?.compactMap(\.intValue), [11, 11, 10])
        XCTAssertEqual(g["rope_theta"]?.doubleValue, 1e7)
        XCTAssertEqual(g["rms_eps"]?.doubleValue ?? 0, 1e-6, accuracy: 1e-12)
    }

    func testLayerCadence() {
        var full = 0
        for l in 0..<QWModelGeometry.qwen38_27b.numLayers {
            let isFull = (l == 3 || l == 7 || l == 63 || (l + 1) % 4 == 0)
            XCTAssertEqual(QWModelGeometry.qwen38_27b.isFullAttention(l), isFull, "layer \(l)")
            if isFull { full += 1 }
        }
        XCTAssertEqual(full, 16)
    }

    func testDTypeSizes() {
        XCTAssertEqual(QWDType.q4g.bytesPerParam, 0.625, accuracy: 1e-9)
        XCTAssertEqual(QWDType.hq4_g32.bytesPerParam, 0.625, accuracy: 1e-9)
        XCTAssertEqual(QWDType.q4g.groupSize, 32)
        XCTAssertEqual(QWDType.q8_0.bytesPerParam, 1.0625, accuracy: 1e-9)
        // HQ4_G32_M5 model-size sanity: 26.89e9 text params ≈ 16.81 GB
        let textParams = 26.89e9
        let gb = textParams * QWDType.q4g.bytesPerParam / 1e9
        XCTAssertEqual(gb, 16.81, accuracy: 0.15)
    }
}

/// Minimal JSON value for the golden (no external deps; C01 stays dependency-free).
enum JSONValue: Decodable {
    case string(String), number(Double), bool(Bool), array([JSONValue]), object([String: JSONValue]), null
    init(from d: Decoder) throws {
        if let s = try? d.singleValueContainer().decode(String.self) { self = .string(s); return }
        if let n = try? d.singleValueContainer().decode(Double.self) { self = .number(n); return }
        if let b = try? d.singleValueContainer().decode(Bool.self) { self = .bool(b); return }
        if let a = try? d.singleValueContainer().decode([JSONValue].self) { self = .array(a); return }
        if let o = try? d.singleValueContainer().decode([String: JSONValue].self) { self = .object(o); return }
        self = .null
    }
    var intValue: Int? { if case .number(let d) = self { return Int(d) }; return nil }
    var doubleValue: Double? { if case .number(let d) = self { return d }; return nil }
    var boolValue: Bool? { if case .bool(let b) = self { return b }; return nil }
    var stringValue: String? { if case .string(let s) = self { return s }; return nil }
    var arrayValue: [JSONValue]? { if case .array(let a) = self { return a }; return nil }
}
