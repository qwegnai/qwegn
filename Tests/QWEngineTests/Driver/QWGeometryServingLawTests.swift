// QWGeometryServingLawTests.swift — the geometry's serving law: which
// kernels may serve a model, and which KV mode it may carry. The law is
// ONE predicate (the 27B's shape), so the writer, the compressed pool and
// the decode cannot disagree about it.
import XCTest
@testable import QWEngine
import QWCore
import QWKV

final class QWGeometryServingLawTests: XCTestCase {
    /// A geometry from a standard config, with the given head width and
    /// kv-head count.
    private func geometry(headDim: Int, qHeads: Int = 8, kvHeads: Int = 2) throws
        -> QWModelGeometry {
        let config: [String: Any] = [
            "model_type": "llama",
            "hidden_size": 1024,
            "num_hidden_layers": 1,
            "num_attention_heads": qHeads,
            "num_key_value_heads": kvHeads,
            "head_dim": headDim,
            "intermediate_size": 256,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "eos_token_id": 2,
            "vocab_size": 64,
        ]
        let data = try JSONSerialization.data(withJSONObject: config)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    /// A narrow head is served across f16, q8, and q4 as long as headDim is a multiple of 32 <= 256.
    func testANarrowHeadIsSupportedAcrossKVModes() throws {
        let g = try geometry(headDim: 128)
        XCTAssertNoThrow(try QWEngineDriver.validateGeometry(g, kvMode: .f16))
        for mode in [QWKVMode.q8, .q4] {
            XCTAssertNoThrow(try QWEngineDriver.validateGeometry(g, kvMode: mode))
        }
    }

    /// A 256-half head at another kv-head count is also supported across KV modes.
    func testA256HeadAtAnotherKvCountIsSupportedAcrossKVModes() throws {
        let g = try geometry(headDim: 256, qHeads: 16, kvHeads: 8)
        XCTAssertFalse(g.is27BShape)
        XCTAssertFalse(g.is27BKVShape)
        XCTAssertNoThrow(try QWEngineDriver.validateGeometry(g, kvMode: .f16))
        XCTAssertNoThrow(try QWEngineDriver.validateGeometry(g, kvMode: .q8))
        XCTAssertNoThrow(try QWEngineDriver.validateGeometry(g, kvMode: .q4))
    }

    /// The 27B itself keeps every mode: the compressed layouts are its own.
    func testThe27BShapeKeepsEveryMode() throws {
        let g = QWModelGeometry.qwen38_27b
        XCTAssertTrue(g.is27BShape)
        XCTAssertTrue(g.is27BKVShape)
        for mode in [QWKVMode.f16, .q8, .q4] {
            XCTAssertNoThrow(try QWEngineDriver.validateGeometry(g, kvMode: mode))
        }
    }

    /// A head the lane mapping cannot serve is refused: it covers a head in
    /// groups of 8 halves, and the writers cover 256 of them.
    func testAnUnservableHeadWidthIsRefused() throws {
        let narrow = try geometry(headDim: 128)
        XCTAssertNoThrow(try QWEngineDriver.validateGeometry(narrow, kvMode: .f16))
        // head_dim 100 is refused by the loader before it can reach here.
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: try JSONSerialization
            .data(withJSONObject: ["model_type": "llama", "hidden_size": 1024,
                                   "num_hidden_layers": 1, "num_attention_heads": 8,
                                   "num_key_value_heads": 2, "head_dim": 100,
                                   "intermediate_size": 256,
                                   "max_position_embeddings": 128,
                                   "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
                                   "eos_token_id": 2, "vocab_size": 64])))
        // A head wider than 256 halves would be half-written.
        let wide = try geometry(headDim: 512, qHeads: 2, kvHeads: 1)
        XCTAssertThrowsError(try QWEngineDriver.validateGeometry(wide, kvMode: .f16)) { e in
            guard case QWEngineDriverError.unsupportedModel = e else {
                return XCTFail("expected unsupportedModel, got \(e)")
            }
            XCTAssertTrue("\(e)".contains("256"), "the refusal names the limit: \(e)")
        }
    }
}
