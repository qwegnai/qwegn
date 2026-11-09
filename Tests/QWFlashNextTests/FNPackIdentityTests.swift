import XCTest
import Foundation
@testable import QWCore
@testable import QWQuant
@testable import QWFlashNext

// Synthetic Flash-Next descriptor and packed N-gram binary table (`FNNgramBinRows`) tests.
// Uses only synthetic in-memory/temp-file fixtures (no real model packs).
final class FNPackIdentityTests: XCTestCase {
    func testFlashNextGoldenDescriptorDecodesCleanly() throws {
        let layerTypes = (0..<48).map { ($0 + 1) % 4 == 0 ? "\"full_attention\"" : "\"linear_attention\"" }
            .joined(separator: ",")
        let config = """
        {
          "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 2560,
            "vocab_size": 248320,
            "num_hidden_layers": 48,
            "layer_types": [\(layerTypes)],
            "full_attention_interval": 4,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262144,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000.0,
            "output_gate_type": "sigmoid",
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 1024,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 16,
            "indexer_head_dim": 128,
            "indexer_kv_heads": 1,
            "indexer_n_heads": 16,
            "hc_count": 4,
            "hc_lowrank": 320,
            "ple_layer_ids": [1, 2, 3],
            "ple_embed_dim": 2560,
            "ple_conv_kernel_size": 2,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "ngram_vocab_size_base": 409600,
            "make_ngram_vocab_size_divisible_by": 100,
            "split_ngram_parts": 2,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false,
            "eos_token_id": 248044
          }
        }
        """.data(using: .utf8)!
        let descriptor = try QWModelDescriptor.decode(configJSON: config)
        XCTAssertEqual(descriptor.family, .qwen38FlashNext)
        XCTAssertEqual(descriptor.numLayers, 48)
        XCTAssertEqual(descriptor.hidden, 2560)
        XCTAssertEqual(descriptor.hcCount, 4)
        XCTAssertEqual(descriptor.hcLowrank, 320)
        XCTAssertEqual(descriptor.pleEmbedDim, 2560)
    }

    func testNgramBinRowsRefusesATableTooShortForItsRows() throws {
        // The header claims 25 rows; the file must hold them. A truncated
        // table must be refused at open, not fail on the first read past
        // the end.
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("fn-ngram-short-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let entryCount = 25
        let header = try QWNgramFormat.makeHeader(version: 1, dtype: .bf16,
                                                  entryCount: UInt64(entryCount),
                                                  headDim: 160, rowBytes: 320, l1Hot: 0)
        // One sector of rows, but the header claims 25 rows spread over 3.
        var data = Data(count: 4096 + 4096)
        data.withUnsafeMutableBytes { raw in
            let p = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for (i, b) in header.enumerated() { p[i] = b }
        }
        let url = tmpDir.appendingPathComponent("ngram.bin")
        try data.write(to: url)
        XCTAssertThrowsError(try FNNgramBinRows(url: url)) { error in
            let ns = error as NSError
            XCTAssertEqual(ns.domain, "FNNgramBinRows")
            XCTAssertEqual(ns.code, 6)
        }
    }

    func testNgramBinRowsReadsSyntheticSectorAlignedTable() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("fn-ngram-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let binURL = tmpDir.appendingPathComponent("ngram.bin")
        let headDim = 160
        let rowBytes = headDim * 2 // 320 bytes
        let rowsPerSector = 12
        let entryCount = 25 // spans 3 sectors

        // Canonical QNGRAM header — must match Scripts/quantize/pack_ngram.py
        // byte for byte: magic@0, version@8, flags@10, dtype@12,
        // entryCount@16, headDim@24, rowBytes@26, l1_hot@28, dataOffset@32.
        let header = try QWNgramFormat.makeHeader(version: 1,
                                                  dtype: .bf16,
                                                  entryCount: UInt64(entryCount),
                                                  headDim: UInt16(headDim),
                                                  rowBytes: UInt16(rowBytes),
                                                  l1Hot: 0)

        var data = Data(count: 4096 + 3 * 4096)
        data.withUnsafeMutableBytes { raw in
            let p = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for (i, b) in header.enumerated() { p[i] = b }

            for rowIdx in 0..<entryCount {
                let sector = rowIdx / rowsPerSector
                let slot = rowIdx % rowsPerSector
                let rowOff = 4096 + sector * 4096 + slot * rowBytes
                for col in 0..<headDim {
                    let f = Float(rowIdx * 1000 + col) * 0.25
                    let bf16 = UInt16(f.bitPattern >> 16).littleEndian
                    raw.storeBytes(of: bf16, toByteOffset: rowOff + col * 2, as: UInt16.self)
                }
            }
        }
        try data.write(to: binURL)

        let loader = try FNNgramRowsLoader.open(assetsURL: tmpDir)
        XCTAssertEqual(loader.headDim, headDim)
        for rowIdx in [0, 11, 12, 24] {
            let got = loader.row(rowIdx)
            XCTAssertEqual(got.count, headDim)
            for col in 0..<headDim {
                let f = Float(rowIdx * 1000 + col) * 0.25
                let expected = Float(bitPattern: UInt32(UInt16(f.bitPattern >> 16)) << 16)
                XCTAssertEqual(got[col], expected, accuracy: 1e-6)
            }
        }
    }
}
