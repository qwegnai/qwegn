import XCTest
import Foundation
@testable import QWQuant
import QWCore

final class QWLSASidecarTests: XCTestCase {
    func testSidecarWriteReadPeekRoundTripAndDigestGuard() throws {
        let desc = QWModelGeometry.qwen38_27b.descriptor
        let inDim = desc.lsaInputDim
        let indexDim = 64
        let basis = QWLSASidecar.synthesizeDeterministicOrthogonalBasis(
            inDim: inDim,
            indexDim: indexDim,
            seed: 42
        )
        XCTAssertEqual(basis.count, indexDim * inDim)

        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qlsa-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let sidecarURL = tmpDir.appendingPathComponent("model.qlsa")
        let prov = QWLSASidecar.Header.Provenance(
            modelId: "Qwen3.8-27B",
            geometryDigest: desc.geometryDigestV2Hex,
            method: "svd",
            indexDim: indexDim,
            recommendedBudget: 2048
        )
        var layerTensors: [Int: (wq: [UInt16], wk: [UInt16])] = [:]
        for l in desc.fullAttentionLayers {
            layerTensors[l] = (wq: basis, wk: basis)
        }
        try QWLSASidecar.write(
            to: sidecarURL,
            provenance: prov,
            layerTensors: layerTensors,
            wqShape: [indexDim, inDim],
            wkShape: [indexDim, inDim]
        )

        let peeked = try QWLSASidecar.peekInfo(url: sidecarURL, descriptor: desc)
        XCTAssertTrue(peeked.geometryMatches)
        XCTAssertEqual(peeked.method, "svd")
        XCTAssertEqual(peeked.indexDim, indexDim)
        XCTAssertEqual(peeked.recommendedBudget, 2048)

        let loaded = try QWLSASidecar(url: sidecarURL, descriptor: desc)
        let firstLayer = desc.fullAttentionLayers[0]
        let readWQ = try loaded.readF16Tensor("layers.\(firstLayer).wq_idx", expectedShape: [indexDim, inDim])
        XCTAssertEqual(readWQ, basis)

        // Mismatched geometry descriptor must be rejected
        let wrongConfigJSON = """
        {
          "model_type": "qwen3",
          "hidden_size": 4096,
          "intermediate_size": 12288,
          "vocab_size": 151936,
          "num_hidden_layers": 36,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "rms_norm_eps": 1e-6,
          "max_position_embeddings": 32768,
          "rope_theta": 1000000.0,
          "eos_token_id": 151645
        }
        """.data(using: .utf8)!
        let wrongDesc = try QWModelDescriptor.decode(configJSON: wrongConfigJSON)
        let wrongPeek = try QWLSASidecar.peekInfo(url: sidecarURL, descriptor: wrongDesc)
        XCTAssertFalse(wrongPeek.geometryMatches)
        XCTAssertThrowsError(try QWLSASidecar(url: sidecarURL, descriptor: wrongDesc))
    }

    func testComputeTopLeftSingularVectorsOrthonormality() throws {
        let rows = 128
        let cols = 256
        let rank = 16
        var mat = [Float](repeating: 0, count: rows * cols)
        for r in 0..<rows {
            for c in 0..<cols {
                mat[r * cols + c] = sin(Float(r * 7 + c * 3) * 0.05) + Float(r == c ? 2.0 : 0.0)
            }
        }
        let f16Basis = try QWLSASidecar.computeTopLeftSingularVectorsF16(
            matrix: mat,
            rows: rows,
            cols: cols,
            rank: rank
        )
        XCTAssertEqual(f16Basis.count, rank * rows)
        let f32Basis = f16Basis.map { Float(Float16(bitPattern: $0)) }
        for i in 0..<rank {
            for j in i..<rank {
                var dot: Float = 0
                for c in 0..<rows {
                    dot += f32Basis[i * rows + c] * f32Basis[j * rows + c]
                }
                if i == j {
                    XCTAssertEqual(dot, 1.0, accuracy: 1e-2, "row \(i) norm^2 != 1")
                } else {
                    XCTAssertEqual(dot, 0.0, accuracy: 1e-2, "rows (\(i),\(j)) not orthogonal")
                }
            }
        }
    }

    func testSynthesizeFromSafeTensorsAndParallelMultiLayerDeterminism() throws {
        let configJSON = """
        {
          "model_type": "qwen3",
          "hidden_size": 64,
          "intermediate_size": 128,
          "vocab_size": 256,
          "num_hidden_layers": 2,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 64,
          "rms_norm_eps": 1e-6,
          "max_position_embeddings": 4096,
          "rope_theta": 1000000.0,
          "eos_token_id": 1
        }
        """.data(using: .utf8)!
        let desc = try QWModelDescriptor.decode(configJSON: configJSON)
        let kvRows = desc.kvHeads * desc.headDim
        let cols = desc.hidden

        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qlsa-st-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }
        try configJSON.write(to: tmpDir.appendingPathComponent("config.json"))

        var header: [String: Any] = [:]
        var payload = Data()
        for layer in 0..<2 {
            var kProj = [Float](repeating: 0, count: kvRows * cols)
            for r in 0..<kvRows {
                for c in 0..<cols {
                    kProj[r * cols + c] = sin(Float((layer + 1) * (r * 5 + c * 3)) * 0.04)
                }
            }
            let bytes = kProj.withUnsafeBytes { Array($0) }
            let name = "model.layers.\(layer).self_attn.k_proj.weight"
            header[name] = [
                "dtype": "F32",
                "shape": [kvRows, cols],
                "data_offsets": [payload.count, payload.count + bytes.count],
            ]
            payload.append(contentsOf: bytes)
        }
        let headerData = try JSONSerialization.data(withJSONObject: header)
        var fileData = Data()
        fileData.append(contentsOf: withUnsafeBytes(of: UInt64(headerData.count).littleEndian) { Array($0) })
        fileData.append(headerData)
        fileData.append(payload)
        try fileData.write(to: tmpDir.appendingPathComponent("model.safetensors"))

        let run1 = try QWLSASidecar.synthesizeFromModel(packPath: tmpDir.path, descriptor: desc, indexDim: 8)
        let run2 = try QWLSASidecar.synthesizeFromModel(packPath: tmpDir.path, descriptor: desc, indexDim: 8)
        XCTAssertEqual(run1.count, 2)
        XCTAssertEqual(run1[0]?.wq, run2[0]?.wq)
        XCTAssertEqual(run1[1]?.wq, run2[1]?.wq)
        XCTAssertEqual(run1[0]?.wq.count, 8 * desc.lsaInputDim)
    }

    func testSynthesizeHCACompressWKVSlicesNoPESubspaceWithoutPlusOneNorm() throws {
        let configJSON = """
        {
          "model_type": "ernie5_5_moe",
          "hidden_size": 128,
          "intermediate_size": 256,
          "vocab_size": 256,
          "num_hidden_layers": 2,
          "compress_ratios": [128, -2],
          "num_attention_heads": 4,
          "num_key_value_heads": 1,
          "head_dim": 64,
          "qk_nope_head_dim": 48,
          "qk_rope_head_dim": 16,
          "q_lora_rank": 64,
          "kv_lora_rank": 64,
          "o_groups": 2,
          "o_lora_rank": 64,
          "sliding_window": 128,
          "compress_rope_theta": 160000.0,
          "hybrid_mla_num_attention_heads": 4,
          "hybrid_mla_num_key_value_heads": 4,
          "hybrid_mla_q_lora_rank": 64,
          "hybrid_mla_kv_lora_rank": 64,
          "hybrid_mla_qk_nope_head_dim": 48,
          "hybrid_mla_qk_rope_head_dim": 16,
          "hybrid_mla_v_head_dim": 64,
          "index_n_heads": 4,
          "index_head_dim": 32,
          "index_topk": 64,
          "first_k_dense_replace": 2,
          "rms_norm_eps": 1e-6,
          "max_position_embeddings": 4096,
          "eos_token_id": 2
        }
        """.data(using: .utf8)!
        let desc = try QWModelDescriptor.decode(configJSON: configJSON)
        XCTAssertEqual(desc.lsaInputDim, 48)
        XCTAssertEqual(desc.lsaKeyProjectionSpec(layer: 0)?.kind, .hcaCompressWKV)
        XCTAssertNil(desc.lsaKeyProjectionSpec(layer: 1))

        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qlsa-hca-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }
        try configJSON.write(to: tmpDir.appendingPathComponent("config.json"))

        var header: [String: Any] = [:]
        var payload = Data()
        var wkv = [Float](repeating: 0, count: 64 * 128)
        for r in 0..<64 {
            for c in 0..<128 {
                wkv[r * 128 + c] = (r < 48) ? sin(Float(r * 7 + c * 3) * 0.05) : 999.0
            }
        }
        let wkvBytes = wkv.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in Array(raw) }
        let wkvName = "model.layers.0.self_attn.core_attention.compressor.linear_wkv.weight"
        header[wkvName] = [
            "dtype": "F32", "shape": [64, 128], "data_offsets": [0, wkvBytes.count]
        ]
        payload.append(contentsOf: wkvBytes)
        let ln = [Float](repeating: 1.0, count: 128)
        let lnBytes = ln.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in Array(raw) }
        header["model.layers.0.input_layernorm.weight"] = [
            "dtype": "F32", "shape": [128], "data_offsets": [payload.count, payload.count + lnBytes.count]
        ]
        payload.append(contentsOf: lnBytes)

        let headerData = try JSONSerialization.data(withJSONObject: header)
        var fileData = Data()
        fileData.append(contentsOf: withUnsafeBytes(of: UInt64(headerData.count).littleEndian) { Array($0) })
        fileData.append(headerData)
        fileData.append(payload)
        try fileData.write(to: tmpDir.appendingPathComponent("model.safetensors"))

        let synthesized = try QWLSASidecar.synthesizeFromModel(packPath: tmpDir.path, descriptor: desc, indexDim: 8)
        XCTAssertEqual(synthesized.count, 1)
        XCTAssertNotNil(synthesized[0])
        XCTAssertNil(synthesized[1])
        XCTAssertEqual(synthesized[0]?.wk.count, 8 * 48)

        let sidecarURL = tmpDir.appendingPathComponent("model.qlsa")
        let prov = QWLSASidecar.Header.Provenance(
            modelId: "ernie5_5_moe", geometryDigest: desc.geometryDigestV2Hex,
            method: "svd", indexDim: 8, recommendedBudget: 2048
        )
        try QWLSASidecar.write(
            to: sidecarURL, provenance: prov, layerTensors: synthesized,
            wqShape: [8, 128], wkShape: [8, 48]
        )
        let loaded = try QWLSASidecar(url: sidecarURL, descriptor: desc)
        let readWK = try loaded.readF16Tensor("layers.0.wk_idx", expectedShape: [8, 48])
        XCTAssertEqual(readWK, synthesized[0]?.wk)
    }
}
