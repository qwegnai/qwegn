// QWAttnWave2Tests.swift — Wave 2 regression tests across PagedPrefill, MLA,
// general decode splits > 64, batched sharedPrefix=false, MBSA clamps, and HCA sub2.

import Foundation
import Metal
import XCTest
@testable import QWAttn
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWKV
import QWQuant

final class QWAttnWave2Tests: XCTestCase {
    private static func makeF16Buf(_ hal: QWHal, _ vals: [Float]) -> any MTLBuffer {
        let bits = vals.map { Float16($0).bitPattern }
        let buf = hal.device.makeBuffer(length: max(16, bits.count * 2), options: .storageModeShared)!
        bits.withUnsafeBufferPointer { src in
            if let base = src.baseAddress {
                buf.contents().copyMemory(from: base, byteCount: bits.count * 2)
            }
        }
        return buf
    }

    private static func tvF16(_ buf: any MTLBuffer, _ count: Int) -> QWTensorValue {
        QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length),
                      offset: 0, shape: QWTensorShape([count], .f16))
    }

    private static func det(_ seed: Int) -> Float {
        let m = (seed &* 2654435761) & 0xFFFF
        return Float(Int(m % 2001) - 1000) / 1000.0
    }

    private static func gqaGeometry(qHeads: Int, kvHeads: Int, headDim: Int) throws -> QWModelGeometry {
        let cfg: [String: Any] = [
            "model_type": "llama",
            "hidden_size": kvHeads * headDim * 4,
            "num_hidden_layers": 1,
            "num_attention_heads": qHeads,
            "num_key_value_heads": kvHeads,
            "head_dim": headDim,
            "intermediate_size": 256,
            "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "eos_token_id": 2,
            "vocab_size": 64,
        ]
        let data = try JSONSerialization.data(withJSONObject: cfg)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    private static func mlaGeometry(indexTopK: Int = 2048) throws -> QWModelGeometry {
        let cfg = """
        {
          "model_type": "glm5_next_text",
          "hidden_size": 4096,
          "intermediate_size": 12288,
          "vocab_size": 154880,
          "num_hidden_layers": 4,
          "layer_types": ["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"],
          "rms_norm_eps": 1e-5,
          "max_position_embeddings": 1048576,
          "num_attention_heads": 64,
          "num_key_value_heads": 64,
          "qk_head_dim": 256,
          "qk_nope_head_dim": 192,
          "qk_rope_head_dim": 64,
          "v_head_dim": 256,
          "kv_lora_rank": 512,
          "q_lora_rank": 1536,
          "mla_use_nope": true,
          "partial_rotary_factor": 1.0,
          "linear_attn_config": {
            "Linear_type": "kda",
            "gate_lower_bound": -5.0,
            "head_dim": 128,
            "num_heads": 64,
            "short_conv_kernel_size": 4
          },
          "n_routed_experts": 288,
          "num_experts_per_tok": 8,
          "moe_intermediate_size": 2048,
          "first_k_dense_replace": 3,
          "index_n_heads": 32,
          "index_head_dim": 128,
          "index_topk": \(indexTopK),
          "index_kpool_compress": true,
          "index_kpool_always_select_tail": true,
          "index_share_for_mtp_iteration": true,
          "indexer_rope_interleave": true,
          "eos_token_id": 154820
        }
        """.data(using: .utf8)!
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: cfg))
    }

    func testPagedPrefillRejectsGQARatioAbove16() throws {
        let geo = try Self.gqaGeometry(qHeads: 36, kvHeads: 2, headDim: 128)
        XCTAssertEqual(geo.gqaRatio, 18)
        let hal = try QWHal(geometry: geo)
        let prefill = try QWAttnPagedPrefill(hal: hal, geometry: geo)

        let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: geo)
        let tables = QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs
        let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: pageBytes * 2 + tables, gdnStates: 4096)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: [0], mode: .f16, geometry: geo)
        let seq = try cache.allocate(promptTokens: 32)
        let views = try cache.layerViews(seq, layer: 0)

        let T = 32
        let qgate = Self.tvF16(Self.makeF16Buf(hal, [Float](repeating: 0, count: T * geo.qProjOut)), T * geo.qProjOut)
        let out = Self.tvF16(Self.makeF16Buf(hal, [Float](repeating: 0, count: T * geo.qHeads * geo.headDim)),
                             T * geo.qHeads * geo.headDim)
        let bListBuf = try XCTUnwrap(hal.device.makeBuffer(length: T * 4 * 4, options: .storageModeShared))
        let bCountBuf = try XCTUnwrap(hal.device.makeBuffer(length: T * 4, options: .storageModeShared))
        let bList = QWTensorValue(region: QWMRegion(buffer: bListBuf, base: 0, length: bListBuf.length),
                                  offset: 0, shape: QWTensorShape([T * 4], .i32))
        let bCount = QWTensorValue(region: QWMRegion(buffer: bCountBuf, base: 0, length: bCountBuf.length),
                                   offset: 0, shape: QWTensorShape([T], .i32))

        guard let (cb1, enc1) = hal.begin() else { XCTFail("hal.begin"); return }
        XCTAssertThrowsError(try prefill.encode(enc1, qgate: qgate, kv: views, out: out, pos: 0, T: T)) { err in
            guard case QWAttnPagedPrefillError.badShape = err else {
                return XCTFail("Expected badShape from encode, got \(err)")
            }
        }
        hal.end(cb1, enc1)

        guard let (cb2, enc2) = hal.begin() else { XCTFail("hal.begin"); return }
        XCTAssertThrowsError(
            try prefill.encodeSparse(enc2, qgate: qgate, kv: views, out: out,
                                     blockList: bList, blockCount: bCount, pos: 0, T: T, listStride: 4)
        ) { err in
            guard case QWAttnPagedPrefillError.badShape = err else {
                return XCTFail("Expected badShape from encodeSparse, got \(err)")
            }
        }
        hal.end(cb2, enc2)
    }

    private static func ernieMlaDsaGeometry(window: Int, indexTopK: Int) throws -> QWModelGeometry {
        let cfg = """
        {
          "architectures": ["Ernie5_5_MoeForCausalLM"], "model_type": "ernie5_5_moe",
          "vocab_size": 1024, "hidden_size": 4096, "intermediate_size": 4096,
          "num_hidden_layers": 2, "max_position_embeddings": 1048576, "rms_norm_eps": 1e-5,
          "num_attention_heads": 64, "num_key_value_heads": 1, "head_dim": 512,
          "compress_ratios": [16, -2], "default_compress_ratio": 16,
          "q_lora_rank": 1536, "kv_lora_rank": 512,
          "compress_qk_nope_head_dim": 448, "compress_qk_rope_head_dim": 64, "compress_v_head_dim": 512,
          "o_groups": 4, "o_lora_rank": 512, "sliding_window": \(window),
          "compress_rope_theta": 160000.0, "hca_rope_type": "yarn",
          "rope_scaling": { "type": "yarn", "factor": 16, "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1 },
          "rope_theta": 10000.0, "rotary_percent": 1.0, "rope_interleave": true,
          "hybrid_mla_attention": "mqa_dsa", "hybrid_mla_num_attention_heads": 64, "hybrid_mla_num_key_value_heads": 64,
          "hybrid_mla_q_lora_rank": 1536, "hybrid_mla_kv_lora_rank": 512,
          "hybrid_mla_qk_nope_head_dim": 192, "hybrid_mla_qk_rope_head_dim": 64, "hybrid_mla_v_head_dim": 256,
          "mqa_split_kv_b_proj": true, "index_n_heads": 32, "index_head_dim": 128, "index_topk": \(indexTopK),
          "use_fast_hadamard": true, "use_vha_attention": true, "add_full_attention_sink_bias": true,
          "use_gated_attn": true, "use_attn_merger": true, "gated_attn_use_q_lora": true, "eos_token_id": 2
        }
        """.data(using: .utf8)!
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: cfg))
    }

    func testMLAAttnOperatorRejectsTopKPlusWindowAbove2560() throws {
        let geo = try Self.ernieMlaDsaGeometry(window: 600, indexTopK: 2048)
        let hal = try QWHal(geometry: geo)
        XCTAssertThrowsError(
            try QWMLAAttnOperator(hal: hal, geometry: geo, numAttnLayers: 1, maxContext: 512, maxPrefillChunk: 16)
        ) { err in
            guard case QWAttnError.badShape = err else {
                return XCTFail("Expected QWAttnError.badShape for topK+window=2648 > 2560, got \(err)")
            }
        }
    }

    func testAttnDecodeGeneralSplitsAbove64MultiHeadNoCrossHeadCorruption() throws {
        let geo = try QWAttnDecodeGeneralTests.tinyGeometry()
        XCTAssertEqual(geo.qHeads, 8)
        XCTAssertEqual(geo.kvHeads, 2)
        XCTAssertGreaterThan(
            QWAttnDecode.decodePartialsBytes(geometry: geo, maxSplits: 72),
            QWAttnDecode.decodePartialsBytes(geometry: geo, maxSplits: 64)
        )

        let hal = try QWHal(geometry: geo)
        let C = 144
        let splits = 72
        let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: geo)
        let pBytes = QWAttnDecode.decodePartialsBytes(geometry: geo, maxSplits: splits)
        let tables = QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs
        let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: pageBytes * 2 + tables, gdnStates: pBytes + 4096)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: [0], mode: .f16, geometry: geo)
        let h = try cache.allocate(promptTokens: C + 1)

        let data = QWAttnDecodeGeneralTests.tinyData(geometry: geo, keys: C)
        try cache.append(h, positions: 0..<C, layer: 0, kind: .k, src: data.kRows.flatMap { $0 })
        try cache.append(h, positions: 0..<C, layer: 0, kind: .v, src: data.vRows.flatMap { $0 })
        let views = try cache.layerViews(h, layer: 0)

        let g = arena.gdnRegion()
        let pOff = try g.allocate(bytes: pBytes)
        let partials = QWMRegion(buffer: g.region.buffer, base: g.region.base + pOff, length: pBytes)
        let qTV = QWAttnDecodeGeneralTests.tv(QWAttnDecodeGeneralTests.buf(hal, data.qRows), geo.qProjOut)
        let kNewTV = QWAttnDecodeGeneralTests.tv(QWAttnDecodeGeneralTests.buf(hal, data.knew), geo.kvHeads * geo.headDim)
        let vNewTV = QWAttnDecodeGeneralTests.tv(QWAttnDecodeGeneralTests.buf(hal, data.vnew), geo.kvHeads * geo.headDim)

        let dec = try QWAttnDecode(hal: hal, kvMode: .f16, geometry: geo)
        func run(s: Int) throws -> [Float] {
            let outBuf = try XCTUnwrap(hal.device.makeBuffer(length: geo.qHeads * geo.headDim * 2, options: .storageModeShared))
            guard let (cb, enc) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
            try dec.encode(enc, qgate: qTV, kv: views, knew: kNewTV, vnew: vNewTV,
                           partials: partials, out: QWAttnDecodeGeneralTests.tv(outBuf, geo.qHeads * geo.headDim),
                           C: C, splits: s)
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
            let p = outBuf.contents().assumingMemoryBound(to: UInt16.self)
            return (0..<(geo.qHeads * geo.headDim)).map { QWQuant.f16tof32Bits(p[$0]) }
        }

        let ref = try run(s: 1)
        let got = try run(s: splits)
        for i in 0..<ref.count {
            XCTAssertTrue(got[i].isFinite, "non-finite at \(i)")
            XCTAssertLessThan(abs(got[i] - ref[i]), 3e-3, "splits=72 corrupted general decode at \(i)")
        }
    }

    func testAttnDecodeBatchedSharedPrefixFalsePreservesCacheAndMatchesSequentialDecode() throws {
        let geo = try QWAttnDecodeGeneralTests.tinyGeometry()
        let hal = try QWHal(geometry: geo)
        let cBase = 16
        let T = 4
        let splits = 2
        let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: geo)
        let pBytes = QWAttnDecode.decodePartialsBytes(geometry: geo, maxSplits: max(QWAttnDecode.maxSplits, splits)) * T
        let tables = QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs
        let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: pageBytes * 2 + tables, gdnStates: pBytes + 4096)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: [0], mode: .f16, geometry: geo)
        let h = try cache.allocate(promptTokens: cBase + T + 4)

        let rowElems = geo.kvHeads * geo.headDim
        var pastK = [UInt16](), pastV = [UInt16]()
        for j in 0..<cBase {
            for e in 0..<rowElems {
                pastK.append(Float16(Self.det(j * 97 + e * 3) * 0.4).bitPattern)
                pastV.append(Float16(Self.det(j * 131 + e * 5) * 0.5).bitPattern)
            }
        }
        try cache.append(h, positions: 0..<cBase, layer: 0, kind: .k, src: pastK)
        try cache.append(h, positions: 0..<cBase, layer: 0, kind: .v, src: pastV)

        // Seed [cBase ..< cBase + T) with known sentinels so batched encode cannot mutate cache slots.
        let sentinel: UInt16 = 0x3C01
        try cache.append(h, positions: cBase..<(cBase + T), layer: 0, kind: .k,
                         src: [UInt16](repeating: sentinel, count: T * rowElems))
        try cache.append(h, positions: cBase..<(cBase + T), layer: 0, kind: .v,
                         src: [UInt16](repeating: sentinel, count: T * rowElems))
        let views = try cache.layerViews(h, layer: 0)

        let qAll = (0..<(T * geo.qProjOut)).map { Float16(Self.det(1000 + $0) * 0.4).bitPattern }
        let kNewAll = (0..<(T * rowElems)).map { Float16(Self.det(2000 + $0) * 0.4).bitPattern }
        let vNewAll = (0..<(T * rowElems)).map { Float16(Self.det(3000 + $0) * 0.5).bitPattern }

        let g = arena.gdnRegion()
        let pOff = try g.allocate(bytes: pBytes)
        let partials = QWMRegion(buffer: g.region.buffer, base: g.region.base + pOff, length: pBytes)

        let dec = try QWAttnDecode(hal: hal, kvMode: .f16, geometry: geo)
        let outBatchBuf = try XCTUnwrap(hal.device.makeBuffer(length: T * geo.qHeads * geo.headDim * 2, options: .storageModeShared))
        let qBuf = QWAttnDecodeGeneralTests.buf(hal, qAll)
        let kBuf = QWAttnDecodeGeneralTests.buf(hal, kNewAll)
        let vBuf = QWAttnDecodeGeneralTests.buf(hal, vNewAll)
        guard let (cbB, encB) = hal.begin() else { XCTFail("hal.begin"); return }
        try dec.encodeBatched(
            encB,
            qgate: QWAttnDecodeGeneralTests.tv(qBuf, T * geo.qProjOut),
            kv: views,
            knew: QWAttnDecodeGeneralTests.tv(kBuf, T * rowElems),
            vnew: QWAttnDecodeGeneralTests.tv(vBuf, T * rowElems),
            partials: partials,
            out: QWAttnDecodeGeneralTests.tv(outBatchBuf, T * geo.qHeads * geo.headDim),
            cBase: cBase, T: T, splits: splits, sharedPrefix: false)
        hal.end(cbB, encB)
        cbB.waitUntilCompleted()
        XCTAssertNil(cbB.error)

        // Compare each token sequentially by writing only rows [0 ..< cBase + t] via normal decode.
        let outElems = geo.qHeads * geo.headDim
        let batchPtr = outBatchBuf.contents().assumingMemoryBound(to: UInt16.self)
        for t in 0..<T {
            if t > 0 {
                let s = (t - 1) * rowElems
                let e = t * rowElems
                try cache.append(h, positions: (cBase + t - 1)..<(cBase + t), layer: 0, kind: .k, src: Array(kNewAll[s..<e]))
                try cache.append(h, positions: (cBase + t - 1)..<(cBase + t), layer: 0, kind: .v, src: Array(vNewAll[s..<e]))
            }
            let outRefBuf = try XCTUnwrap(hal.device.makeBuffer(length: outElems * 2, options: .storageModeShared))
            guard let (cbR, encR) = hal.begin() else { XCTFail("hal.begin"); return }
            let qSlice = QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length),
                                       offset: t * geo.qProjOut * 2, shape: QWTensorShape([geo.qProjOut], .f16))
            let kSlice = QWTensorValue(region: QWMRegion(buffer: kBuf, base: 0, length: kBuf.length),
                                       offset: t * rowElems * 2, shape: QWTensorShape([rowElems], .f16))
            let vSlice = QWTensorValue(region: QWMRegion(buffer: vBuf, base: 0, length: vBuf.length),
                                       offset: t * rowElems * 2, shape: QWTensorShape([rowElems], .f16))
            try dec.encode(encR, qgate: qSlice, kv: views, knew: kSlice, vnew: vSlice,
                           partials: partials, out: QWAttnDecodeGeneralTests.tv(outRefBuf, outElems),
                           C: cBase + t, splits: splits)
            hal.end(cbR, encR)
            cbR.waitUntilCompleted()
            XCTAssertNil(cbR.error)
            let refPtr = outRefBuf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<outElems {
                let a = QWQuant.f16tof32Bits(batchPtr[t * outElems + i])
                let b = QWQuant.f16tof32Bits(refPtr[i])
                XCTAssertLessThan(abs(a - b), 3e-3, "batched vs seq mismatch at t=\(t) i=\(i)")
            }
        }
    }

    func testPagedPrefillF16ExactPrecisionWithHighDynamicRangeKeysAtPosPlusTGe192() throws {
        let geo = QWModelGeometry.qwen38_27b
        let hal = try QWHal(geometry: geo)
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable, "Paged NAX prefill requires Apple10+ NAX")

        let pos = 192
        let T = 32
        let ceiling = pos + T + 64
        let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: geo)
        let needPages = (ceiling + 255) / 256
        let pBytes = QWAttnDecode.decodePartialsBytes(geometry: geo)
        let tables = QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs
        let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: pageBytes * needPages * 2 + tables, gdnStates: pBytes)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: [3], mode: .f16, geometry: geo)
        let h = try cache.allocate(promptTokens: ceiling)

        // Outliers on d=0 force coarse INT8 quantization if i8 path were erroneously used for .f16 mode.
        let kvElems = geo.kvHeads * geo.headDim
        var kData = [UInt16](repeating: 0, count: (pos + T) * kvElems)
        var vData = [UInt16](repeating: 0, count: (pos + T) * kvElems)
        for row in 0..<(pos + T) {
            for kh in 0..<geo.kvHeads {
                for d in 0..<geo.headDim {
                    let idx = row * kvElems + kh * geo.headDim + d
                    kData[idx] = Float16(d == 0 ? (row % 2 == 0 ? 48.0 : -48.0) : Self.det(row * 43 + d * 7) * 0.15).bitPattern
                    vData[idx] = Float16(Self.det(row * 79 + kh * 19 + d * 11) * 0.4).bitPattern
                }
            }
        }
        try cache.append(h, positions: 0..<(pos + T), layer: 3, kind: .k, src: kData)
        try cache.append(h, positions: 0..<(pos + T), layer: 3, kind: .v, src: vData)
        let views = try cache.layerViews(h, layer: 3)

        var qData = [UInt16](repeating: 0, count: T * geo.qProjOut)
        for t in 0..<T {
            for qh in 0..<geo.qHeads {
                let base = t * geo.qProjOut + qh * 2 * geo.headDim
                for d in 0..<geo.headDim {
                    qData[base + d] = Float16(d == 0 ? 0.01 : Self.det(t * 29 + qh * 17 + d) * 0.2).bitPattern
                    qData[base + geo.headDim + d] = Float16(0.5).bitPattern
                }
            }
        }

        let outElems = T * geo.qHeads * geo.headDim
        let qBuf = QWAttnDecodeGeneralTests.buf(hal, qData)
        let outPagedBuf = try XCTUnwrap(hal.device.makeBuffer(length: outElems * 2, options: .storageModeShared))
        let prefill = try QWAttnPagedPrefill(hal: hal, geometry: geo)
        guard let (cbP, encP) = hal.begin() else { XCTFail("hal.begin"); return }
        try prefill.encode(encP, qgate: Self.tvF16(qBuf, T * geo.qProjOut), kv: views,
                           out: Self.tvF16(outPagedBuf, outElems), pos: pos, T: T)
        hal.end(cbP, encP)
        cbP.waitUntilCompleted()
        XCTAssertNil(cbP.error)

        let decode = try QWAttnDecode(hal: hal, kvMode: .f16, geometry: geo)
        let g = arena.gdnRegion()
        let pOff = try g.allocate(bytes: pBytes)
        let partials = QWMRegion(buffer: g.region.buffer, base: g.region.base + pOff, length: pBytes)
        let pagedPtr = outPagedBuf.contents().assumingMemoryBound(to: UInt16.self)

        for t in [0, 15, T - 1] {
            let tokPos = pos + t
            let qRow = QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length),
                                     offset: t * geo.qProjOut * 2, shape: QWTensorShape([geo.qProjOut], .f16))
            let ks = tokPos * kvElems
            let kRow = Self.tvF16(QWAttnDecodeGeneralTests.buf(hal, Array(kData[ks..<(ks + kvElems)])), kvElems)
            let vRow = Self.tvF16(QWAttnDecodeGeneralTests.buf(hal, Array(vData[ks..<(ks + kvElems)])), kvElems)
            let outRefBuf = try XCTUnwrap(hal.device.makeBuffer(length: geo.qHeads * geo.headDim * 2, options: .storageModeShared))
            guard let (cbD, encD) = hal.begin() else { XCTFail("hal.begin"); return }
            try decode.encode(encD, qgate: qRow, kv: views, knew: kRow, vnew: vRow,
                              partials: partials, out: Self.tvF16(outRefBuf, geo.qHeads * geo.headDim),
                              C: tokPos, splits: QWAttnDecode.splits(forKeys: tokPos + 1))
            hal.end(cbD, encD)
            cbD.waitUntilCompleted()
            XCTAssertNil(cbD.error)
            let refPtr = outRefBuf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<(geo.qHeads * geo.headDim) {
                let diff = abs(QWQuant.f16tof32Bits(pagedPtr[t * geo.qHeads * geo.headDim + i]) - QWQuant.f16tof32Bits(refPtr[i]))
                XCTAssertLessThan(diff, 5e-3, "FP16 precision degraded at pos+t=\(tokPos) i=\(i)")
            }
        }
    }

    func testMBSASelectBlocksClampsRemoteBlocksAbove32() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true", "GPU MBSA test skipped on CI")
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .standard)
        let pso = try hal.pipeline("qwg_mbsa_select_blocks")

        let kvHeads = 2, headDim = 64, blockSize = 128, pos0 = 6400
        let prefixBlks = 2, localBlks = 2, remoteBlks = 40, listStride = 48
        let totalBlocks = (pos0 / blockSize) + 1
        let qBuf = Self.makeF16Buf(hal, (0..<(kvHeads * headDim)).map { _ in 0.25 })
        let kBuf = Self.makeF16Buf(hal, (0..<(totalBlocks * kvHeads * headDim)).map { Self.det($0) * 0.5 })
        let outList = try XCTUnwrap(hal.device.makeBuffer(length: kvHeads * listStride * 4, options: .storageModeShared))
        let outCount = try XCTUnwrap(hal.device.makeBuffer(length: kvHeads * 4, options: .storageModeShared))

        var p0: (UInt32, UInt32, UInt32, UInt32) = (UInt32(pos0), UInt32(kvHeads), UInt32(headDim), UInt32(blockSize))
        var p1: (UInt32, UInt32, UInt32, UInt32) = (UInt32(prefixBlks), UInt32(localBlks), UInt32(remoteBlks), UInt32(listStride))
        guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(kBuf, offset: 0, index: 1)
        enc.setBuffer(outList, offset: 0, index: 2)
        enc.setBuffer(outCount, offset: 0, index: 3)
        enc.setBytes(&p0, length: 16, index: 4)
        enc.setBytes(&p1, length: 16, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: kvHeads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let counts = outCount.contents().assumingMemoryBound(to: Int32.self)
        let list = outList.contents().assumingMemoryBound(to: Int32.self)
        for h in 0..<kvHeads {
            let c = Int(counts[h])
            XCTAssertGreaterThan(c, 0)
            XCTAssertLessThanOrEqual(c, prefixBlks + localBlks + 32)
            for i in 0..<c {
                XCTAssertGreaterThanOrEqual(list[h * listStride + i], 0)
                XCTAssertLessThan(list[h * listStride + i], Int32(totalBlocks))
            }
        }
    }

    func testMLALatentPrepZeroVectorQ8AndQ4KVAreFiniteNotNaN() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.supportsSimdgroupMMA == true,
            "MLA GPU test requires bare-metal Apple Silicon"
        )
        let geo = try Self.mlaGeometry()
        let hal = try QWHal(geometry: geo)

        for kvMode in [QWKVMode.q8, QWKVMode.q4] {
            let pageBytes = QWKVCache.pageBytes(mode: kvMode, geometry: geo)
            let tables = QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs
            let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: pageBytes * 2 + tables, gdnStates: 4096)
            let cache = try QWKVCache(hal: hal, arena: arena, layers: [3], mode: kvMode, geometry: geo)
            let seq = try cache.allocate(promptTokens: 16)
            let kvViews = try cache.layerViews(seq, layer: 3)
            let op = try QWMLAAttnOperator(hal: hal, geometry: geo, numAttnLayers: 1, maxContext: 64, maxPrefillChunk: 8)

            // Zero latent and zero ln1 force 0.0 group scales inside KV pool quantization.
            memset(op.region.buffer.contents() + op.mlaLatentScratch.offset, 0, 4 * op.latentStride * 2)
            memset(op.region.buffer.contents() + op.mlaQFullScratch.offset, 0, 4 * op.qFullStride * 2)

            let ln1 = Self.tvF16(Self.makeF16Buf(hal, [Float](repeating: 0, count: 4 * geo.hidden)), 4 * geo.hidden)
            let qANormW = Self.tvF16(Self.makeF16Buf(hal, [Float](repeating: 1, count: geo.qLoraRank)), geo.qLoraRank)
            let kvANormW = Self.tvF16(Self.makeF16Buf(hal, [Float](repeating: 1, count: geo.kvLoraRank)), geo.kvLoraRank)
            let kNormW = Self.tvF16(Self.makeF16Buf(hal, [Float](repeating: 1, count: geo.indexerHeadDim)), geo.indexerHeadDim)
            let kNormB = Self.tvF16(Self.makeF16Buf(hal, [Float](repeating: 0, count: geo.indexerHeadDim)), geo.indexerHeadDim)
            let wGateW = Self.tvF16(Self.makeF16Buf(hal, [Float](repeating: 0, count: (geo.indexerHeads + 4) * geo.hidden)),
                                    (geo.indexerHeads + 4) * geo.hidden)
            let ape = Self.tvF16(Self.makeF16Buf(hal, [Float](repeating: 0, count: 4 * geo.indexerHeadDim)), 4 * geo.indexerHeadDim)
            let kvBRows = geo.qHeads * (geo.qkNopeHeadDim + geo.vHeadDim)
            let kvBGroups = kvBRows * (geo.kvLoraRank / 32)
            let kvBBuf = try XCTUnwrap(hal.device.makeBuffer(length: kvBGroups * 20, options: .storageModeShared))
            memset(kvBBuf.contents(), 0, kvBBuf.length)
            let kvBTV = QWTensorValue(region: QWMRegion(buffer: kvBBuf, base: 0, length: kvBBuf.length),
                                      offset: 0, shape: QWTensorShape([kvBGroups * 10], .f16))
            let kvBView = try XCTUnwrap(QWWeightView(blocks: kvBTV, n: kvBRows, k: geo.kvLoraRank,
                                                     bits: 4, groupSize: 32, isRotated: true))
            let mlaW = QWMLAAttnOperator.LayerWeights(
                qANormW: qANormW, kvANormW: kvANormW, kNormW: kNormW, kNormB: kNormB,
                wGateWeights: wGateW, ape: ape, qBProj: kvBView, kvBProj: kvBView)
            let attOut = Self.tvF16(Self.makeF16Buf(hal, [Float](repeating: 0, count: 4 * geo.qHeads * geo.vHeadDim)),
                                    4 * geo.qHeads * geo.vHeadDim)

            guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
            op.encodeLatentPrepAndPool(enc, ln1: ln1, mlaW: mlaW, kv: kvViews, fullAttnLayer: 0, pos0: 0, rows: 4)
            op.encodeScoreSelectAbsorbAttendAndProjUV(enc, mlaW: mlaW, kv: kvViews, attOut: attOut, fullAttnLayer: 0, pos0: 0, rows: 4)
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)

            let outPtr = attOut.region.buffer.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<(4 * geo.qHeads * geo.vHeadDim) {
                let v = QWQuant.f16tof32Bits(outPtr[i])
                XCTAssertTrue(v.isFinite, "Zero KV mode \(kvMode) produced NaN/Inf at \(i)")
            }
        }
    }

    func testHCAGroupedWoASub2SHVQ15AndSHVQ1Parity() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let oGroups = 2, oLoraRank = 128, groupIn = 256, oGateOffset = 64
        let qUpStride = oGateOffset + oGroups * oLoraRank
        let wRows = oGroups * oLoraRank

        for (bits, group) in [(15, 64), (1, 64)] {
            let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
            var seed: UInt32 = 0x1357_2468 ^ UInt32(bits * 31)
            var packed = Data(count: wRows * (groupIn / group) * gBytes)
            var rawUnpacked = [Float16](repeating: 0, count: wRows * groupIn)
            packed.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                guard let base = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                var scratch = [Float]()
                for r in 0..<wRows {
                    for g in 0..<(groupIn / group) {
                        let bOff = (r * (groupIn / group) + g) * gBytes
                        let sc = Float16(0.05 * (1.0 + 0.1 * Float((r + g) % 5))).bitPattern
                        base[bOff] = UInt8(sc & 0xFF)
                        base[bOff + 1] = UInt8(sc >> 8)
                        for b in 2..<gBytes {
                            seed = seed &* 1664525 &+ 1013904223
                            base[bOff + b] = UInt8((seed >> 16) & 0xFF)
                        }
                        QWSHVQBlock.unpack(
                            group: UnsafeRawBufferPointer(start: base.advanced(by: bOff), count: gBytes),
                            bits: bits, group: group, into: &scratch)
                        for i in 0..<group { rawUnpacked[r * groupIn + g * group + i] = Float16(scratch[i]) }
                    }
                }
            }

            let T = 18
            let arena = try QWArena(hal: hal, weights: max(8192, packed.count + rawUnpacked.count * 2 + 4096),
                                    activations: 16 << 20, kv: 0, gdnStates: 0)
            let wr = arena.weightsRegion(), ar = arena.activationsRegion()
            let wSub2 = try wr.allocate(QWTensorShape([max(1, (packed.count + 1) / 2)], .f16))
            let wRaw = try wr.allocate(QWTensorShape([rawUnpacked.count], .f16))
            arena.copyIn(wSub2, from: packed)
            arena.write(wRaw, from: rawUnpacked)

            let zIn = try ar.allocate(QWTensorShape([T * oGroups * groupIn], .f16))
            let qUp = try ar.allocate(QWTensorShape([T * qUpStride], .f16))
            let outRaw = try ar.allocate(QWTensorShape([T * wRows], .f16))
            let outGemv = try ar.allocate(QWTensorShape([T * wRows], .f16))
            let outMR = try ar.allocate(QWTensorShape([T * wRows], .f16))
            let outMMA = try ar.allocate(QWTensorShape([T * wRows], .f16))
            arena.write(zIn, from: (0..<(T * oGroups * groupIn)).map { Float16(Self.det($0) * 0.25) })
            arena.write(qUp, from: (0..<(T * qUpStride)).map { Float16(Self.det(5000 + $0) * 0.5) })

            func run(_ name: String, _ w: QWTensorValue, _ dst: QWTensorValue, _ inc: Int, _ rowsT: Int, _ grid: MTLSize, _ tg: Int) throws {
                let pso = try hal.pipeline(name)
                var prm = QWHCAAttnOperator.GroupedWoAParams(
                    T: UInt32(rowsT), oGroups: UInt32(oGroups), oLoraRank: UInt32(oLoraRank),
                    groupIn: UInt32(groupIn), qUpStride: UInt32(qUpStride), oGateOffset: UInt32(oGateOffset),
                    bits: UInt32(bits), group: UInt32(group), isIncumbent: UInt32(inc))
                guard let (cb, enc) = hal.begin() else { throw XCTSkip("hal.begin") }
                enc.setComputePipelineState(pso)
                zIn.bind(enc, index: 0)
                w.bind(enc, index: 1)
                qUp.bind(enc, index: 2)
                dst.bind(enc, index: 3)
                enc.setBytes(&prm, length: MemoryLayout<QWHCAAttnOperator.GroupedWoAParams>.stride, index: 4)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
                hal.end(cb, enc)
                cb.waitUntilCompleted()
                XCTAssertNil(cb.error)
            }

            let tMR = 6
            let gGemv = MTLSize(width: oLoraRank / 32, height: oGroups, depth: T)
            let gMR = MTLSize(width: oLoraRank / 32, height: oGroups, depth: 1)
            let gMMA = MTLSize(width: oGroups, height: (T + 31) / 32, depth: oLoraRank / 128)
            try run("qwg_hca_grouped_wo_a_gemv", wRaw, outRaw, 3, T, gGemv, 256)
            try run("qwg_hca_grouped_wo_a_gemv_sub2", wSub2, outGemv, 2, T, gGemv, 256)
            try run("qwg_hca_grouped_wo_a_gemv_mr_sub2", wSub2, outMR, 2, tMR, gMR, 256)
            try run("qwg_hca_grouped_wo_a_mma_sub2", wSub2, outMMA, 2, T, gMMA, 128)

            let rawVals = arena.read(outRaw, as: Float16.self, count: T * wRows)
            let gemvVals = arena.read(outGemv, as: Float16.self, count: T * wRows)
            let mrVals = arena.read(outMR, as: Float16.self, count: tMR * wRows)
            let mmaVals = arena.read(outMMA, as: Float16.self, count: T * wRows)
            for i in 0..<(tMR * wRows) {
                XCTAssertEqual(mrVals[i].bitPattern, gemvVals[i].bitPattern, "bits=\(bits) mr != gemv at \(i)")
            }
            for i in 0..<rawVals.count {
                XCTAssertLessThan(abs(Float(gemvVals[i]) - Float(rawVals[i])), 4e-3, "bits=\(bits) sub2 vs raw at \(i)")
                XCTAssertLessThan(abs(Float(mmaVals[i]) - Float(rawVals[i])), 5e-3, "bits=\(bits) mma_sub2 vs raw at \(i)")
            }
            if hal.caps.isMetal4TensorCapable {
                let outNAX = try ar.allocate(QWTensorShape([T * wRows], .f16))
                let gNAX = MTLSize(width: oLoraRank / 64, height: (T + 63) / 64, depth: oGroups)
                try run("qwg_hca_grouped_wo_a_nax_sub2", wSub2, outNAX, 2, T, gNAX, 128)
                let naxVals = arena.read(outNAX, as: Float16.self, count: T * wRows)
                for i in 0..<rawVals.count {
                    XCTAssertLessThan(abs(Float(naxVals[i]) - Float(rawVals[i])), 5e-3, "bits=\(bits) nax_sub2 vs raw at \(i)")
                }
            }
        }
    }
}
