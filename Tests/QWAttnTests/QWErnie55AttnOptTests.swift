// QWErnie55AttnOptTests.swift — Parity tests for Wave 1.2 MLA-DSA & Wave 1.4 HCA upgrades.
import Foundation
import Metal
import XCTest
@testable import QWAlloc
@testable import QWAttn
@testable import QWCore
@testable import QWGEMM
@testable import QWHAL
@testable import QWKV
@testable import QWQuant

final class QWErnie55AttnOptTests: XCTestCase {
    private static func detRand(_ count: Int, seed: UInt64, scale: Float = 0.25) -> [Float] {
        var s = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
        return (0..<count).map { _ in
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27
            let r = s &* 0x2545_F491_4F6C_DD1D
            let u = Float(r & 0xFFFF) / 65535.0
            return (u * 2.0 - 1.0) * scale
        }
    }

    private static func makeF16Tensor(_ hal: QWHal, shape: [Int], values: [Float]) -> QWTensorValue {
        let h = values.map { Float16($0) }, bytes = max(256, (h.count * 2 + 255) & ~255)
        let buf = hal.device.makeBuffer(length: bytes, options: .storageModeShared)!
        h.withUnsafeBytes { (r: UnsafeRawBufferPointer) in if let s = r.baseAddress, !r.isEmpty { buf.contents().copyMemory(from: s, byteCount: r.count) } }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: bytes), offset: 0, shape: QWTensorShape(shape, .f16))
    }
    private static func makeF32Tensor(_ hal: QWHal, shape: [Int], values: [Float]) -> QWTensorValue {
        let bytes = max(256, (values.count * 4 + 255) & ~255), buf = hal.device.makeBuffer(length: bytes, options: .storageModeShared)!
        values.withUnsafeBytes { (r: UnsafeRawBufferPointer) in if let s = r.baseAddress, !r.isEmpty { buf.contents().copyMemory(from: s, byteCount: r.count) } }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: bytes), offset: 0, shape: QWTensorShape(shape, .f32))
    }
    private static func makeI32Tensor(_ hal: QWHal, shape: [Int], values: [Int32]) -> QWTensorValue {
        let bytes = max(256, (values.count * 4 + 255) & ~255), buf = hal.device.makeBuffer(length: bytes, options: .storageModeShared)!
        values.withUnsafeBytes { (r: UnsafeRawBufferPointer) in if let s = r.baseAddress, !r.isEmpty { buf.contents().copyMemory(from: s, byteCount: r.count) } }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: bytes), offset: 0, shape: QWTensorShape(shape, .i32))
    }

    private static func readF16(_ t: QWTensorValue, count: Int) -> [Float] {
        let p = t.region.buffer.contents().advanced(by: t.region.base + t.offset).assumingMemoryBound(to: UInt16.self)
        return [Float](unsafeUninitializedCapacity: count) { dst, initCnt in
            guard let dp = dst.baseAddress else { initCnt = 0; return }
            var i = 0; while i < count { dp[i] = Float(Float16(bitPattern: p[i])); i += 1 }
            initCnt = count
        }
    }

    private static func readF32(_ t: QWTensorValue, count: Int) -> [Float] {
        let p = t.region.buffer.contents().advanced(by: t.region.base + t.offset).assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: p, count: count))
    }

    private static func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        return a.withUnsafeBufferPointer { aB in
            b.withUnsafeBufferPointer { bB -> Float in
                guard let ap = aB.baseAddress, let bp = bB.baseAddress else { return 0 }
                var m: Float = 0, i = 0
                while i < aB.count { if !(ap[i].isFinite && bp[i].isFinite) { return .infinity }; let d = abs(ap[i] - bp[i]); if d > m { m = d }; i += 1 }
                return m
            }
        }
    }

    private static func packTestWeights(hal: QWHal, inc: Int, bits: Int, group: Int,
                                        rows: Int, cols: Int, seed: UInt64) -> QWTensorValue {
        if inc == 3 || bits == 16 {
            return makeF16Tensor(hal, shape: [rows * cols], values: detRand(rows * cols, seed: seed, scale: 0.2))
        }
        let nGroups = cols / group
        let rowBytes: Int
        if inc == 1 {
            rowBytes = (cols / 32) * 20
        } else if inc == 2 {
            let sBytes = bits == 1 ? 4 : (bits == 15 ? 6 : (bits == 2 ? 8 : (bits == 3 ? 12 : 16)))
            rowBytes = nGroups * (2 + (group / 32) * sBytes)
        } else {
            rowBytes = nGroups * (4 + (group * bits) / 8)
        }
        var bytes = Data(count: max(256, (rows * rowBytes + 1) & ~1))
        var rng = seed == 0 ? 0x55AA_55AA_1234_5678 : seed
        func nextByte() -> UInt8 {
            rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27
            return UInt8(truncatingIfNeeded: (rng &* 0x2545_F491_4F6C_DD1D) >> 8)
        }
        bytes.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            let gBytes = rowBytes / nGroups
            for r in 0..<rows {
                for g in 0..<nGroups {
                    let gp = ptr.advanced(by: r * rowBytes + g * gBytes)
                    if inc == 2 {
                        let sc = Float16(0.03 + Float((r + g) % 5) * 0.01)
                        UnsafeMutableRawPointer(gp).storeBytes(of: sc.bitPattern, as: UInt16.self)
                        for b in 2..<gBytes { gp[b] = nextByte() }
                    } else {
                        let sc: Float16
                        let bi: Float16
                        if inc == 1 {
                            sc = Float16(0.03 + Float((r + g) % 5) * 0.005)
                            bi = Float16(-1031.5 * Float(sc))
                        } else if bits == 8 {
                            sc = Float16(0.002 + Float((r + g) % 5) * 0.0005)
                            bi = Float16(-127.5 * Float(sc))
                        } else {
                            sc = Float16(0.03 + Float((r + g) % 5) * 0.005)
                            bi = Float16(-7.5 * Float(sc))
                        }
                        UnsafeMutableRawPointer(gp).storeBytes(of: sc.bitPattern, as: UInt16.self)
                        UnsafeMutableRawPointer(gp.advanced(by: 2)).storeBytes(of: bi.bitPattern, as: UInt16.self)
                        for b in 4..<gBytes { gp[b] = nextByte() }
                    }
                }
            }
        }
        let buf = hal.device.makeBuffer(length: bytes.count, options: .storageModeShared)!
        bytes.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let src = raw.baseAddress { buf.contents().copyMemory(from: src, byteCount: bytes.count) }
        }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: bytes.count),
                             offset: 0, shape: QWTensorShape([bytes.count / 2], .f16))
    }

    private static func makeErnieGeometry(qHeads: Int = 64, hidden: Int = 256,
                                          topK: Int = 32, window: Int = 16,
                                          vHeadDim: Int = 512, oLoraRank: Int = 32) throws -> QWModelGeometry {
        let cfg = """
        {
          "architectures": ["Ernie5_5_MoeForCausalLM"], "model_type": "ernie5_5_moe",
          "vocab_size": 1024, "hidden_size": \(hidden), "intermediate_size": \(hidden),
          "num_hidden_layers": 2, "max_position_embeddings": 1048576, "rms_norm_eps": 1e-5,
          "num_attention_heads": \(qHeads), "num_key_value_heads": 1, "head_dim": 512,
          "compress_ratios": [128, -2], "default_compress_ratio": 128,
          "q_lora_rank": 128, "kv_lora_rank": 512,
          "compress_qk_nope_head_dim": 448, "compress_qk_rope_head_dim": 64, "compress_v_head_dim": 512,
          "o_groups": 16, "o_lora_rank": \(oLoraRank), "sliding_window": \(window),
          "compress_rope_theta": 160000.0, "hca_rope_type": "yarn",
          "rope_scaling": { "type": "yarn", "factor": 16, "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1 },
          "rope_theta": 1000000.0, "rotary_percent": 1.0, "rope_interleave": true,
          "hybrid_mla_attention": "mqa_dsa", "hybrid_mla_num_attention_heads": \(qHeads), "hybrid_mla_num_key_value_heads": \(qHeads),
          "hybrid_mla_q_lora_rank": 128, "hybrid_mla_kv_lora_rank": 512,
          "hybrid_mla_qk_nope_head_dim": 256, "hybrid_mla_qk_rope_head_dim": 64, "hybrid_mla_v_head_dim": \(vHeadDim),
          "mqa_split_kv_b_proj": true, "index_n_heads": 64, "index_head_dim": 128, "index_topk": \(topK),
          "use_fast_hadamard": true, "use_vha_attention": true, "add_full_attention_sink_bias": true,
          "use_gated_attn": true, "use_attn_merger": true, "gated_attn_use_q_lora": true, "eos_token_id": 2
        }
        """.data(using: .utf8)!
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: cfg))
    }

    private static func fwht32(_ v: inout [Float], base: Int) {
        let invSqrt32: Float = 1.0 / sqrt(32.0)
        var h = 1
        while h < 32 {
            var i = 0
            while i < 32 {
                for j in i..<(i + h) {
                    let a = v[base + j], b = v[base + j + h]
                    v[base + j] = a + b
                    v[base + j + h] = a - b
                }
                i += h * 2
            }
            h <<= 1
        }
        for j in 0..<32 { v[base + j] *= invSqrt32 }
    }

    func testMLADSA8KiBSparseAndSplitMatchMMAReference() throws {
        let geo = try Self.makeErnieGeometry(topK: 48, window: 16)
        let hal = try QWHal(geometry: geo)
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable, "MLA-DSA NAX requires Metal 4")

        let psoSparse = try hal.pipeline("qwg_mla_dsa_sparse_attn_nax")
        let psoSparseH32 = try hal.pipeline("qwg_mla_dsa_sparse_attn_nax_h32")
        let psoSparseH64 = try hal.pipeline("qwg_mla_dsa_sparse_attn_nax_h64")
        let psoSplit = try hal.pipeline("qwg_mla_dsa_sparse_attn_nax_split")
        let psoCombine = try hal.pipeline("qwg_mla_dsa_sparse_attn_combine")
        let psoMMA = try hal.pipeline("qwg_mla_dsa_sparse_attn_mma")

        let qHeads = 64, kvRank = 512, ropeDim = 64, nopeDim = 256
        let qFullStride = qHeads * (nopeDim + ropeDim)
        let qRopeOffset = qHeads * nopeDim
        let rows = 33, pos0 = 64, topK = 48, window = 16, listStride = 128

        let qAbs = Self.makeF16Tensor(hal, shape: [rows * qHeads * kvRank], values: Self.detRand(rows * qHeads * kvRank, seed: 0x101))
        let mlaQFull = Self.makeF16Tensor(hal, shape: [rows * qFullStride], values: Self.detRand(rows * qFullStride, seed: 0x102))
        let kPe = Self.makeF16Tensor(hal, shape: [256 * ropeDim], values: Self.detRand(256 * ropeDim, seed: 0x103))
        let sink = Self.makeF32Tensor(hal, shape: [qHeads], values: (0..<qHeads).map { Float($0 % 5) * 0.2 - 0.8 })
        var blkArr = [Int32](repeating: 0, count: rows * listStride)
        for t in 0..<rows {
            let seqLen = pos0 + t + 1
            for s in 0..<(topK + window) { blkArr[t * listStride + s] = Int32((t * 17 + s * 7) % seqLen) }
        }
        let blkList = Self.makeI32Tensor(hal, shape: [rows * listStride], values: blkArr)
        let zeroOut = [Float](repeating: 0, count: rows * qHeads * kvRank)
        let outRef = Self.makeF16Tensor(hal, shape: [rows * qHeads * kvRank], values: zeroOut)
        let outCand = Self.makeF16Tensor(hal, shape: [rows * qHeads * kvRank], values: zeroOut)
        let partO = Self.makeF16Tensor(hal, shape: [8 * 4 * qHeads * kvRank], values: [Float](repeating: 0, count: 8 * 4 * qHeads * kvRank))
        let partML = Self.makeF32Tensor(hal, shape: [8 * 4 * qHeads * 2], values: [Float](repeating: 0, count: 8 * 4 * qHeads * 2))

        for kvMode: UInt32 in [0, 1, 2] {
            let mlaKV = kvMode == 0 ? QWKVMode.f16 : (kvMode == 1 ? .q8 : .q4)
            let pBytes = QWKVCache.pageBytes(mode: mlaKV, geometry: geo)
            let tables = QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs
            let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: pBytes * 2 + tables, gdnStates: 4096)
            let cache = try QWKVCache(hal: hal, arena: arena, layers: [1], mode: mlaKV, geometry: geo)
            let seq = try cache.allocate(promptTokens: pos0 + rows)
            let kv = try cache.layerViews(seq, layer: 1)

            // Seed deterministic valid KV page bytes (avoiding NaN scales).
            for pool in [kv.k.pool, kv.v.pool] {
                if kvMode == 0 {
                    let p = pool.buffer.contents().advanced(by: pool.base).assumingMemoryBound(to: Float16.self)
                    let vals = Self.detRand(256 * 256, seed: 0x301 + UInt64(kvMode))
                    for i in 0..<vals.count { p[i] = Float16(vals[i]) }
                } else if kvMode == 1 {
                    let base = pool.buffer.contents().advanced(by: pool.base)
                    let qPtr = base.assumingMemoryBound(to: Int8.self)
                    let scPtr = base.advanced(by: 65536).assumingMemoryBound(to: Float16.self)
                    for r in 0..<256 {
                        scPtr[r] = Float16(0.02 + Float(r % 7) * 0.005)
                        for c in 0..<256 { qPtr[r * 256 + c] = Int8(((r * 13 + c * 7) % 127) - 63) }
                    }
                } else {
                    let base = pool.buffer.contents().advanced(by: pool.base)
                    let qPtr = base.assumingMemoryBound(to: UInt8.self)
                    let sbPtr = base.advanced(by: 32768).assumingMemoryBound(to: UInt16.self)
                    for r in 0..<256 {
                        for g in 0..<2 {
                            sbPtr[r * 4 + g * 2] = Float16(0.05).bitPattern
                            sbPtr[r * 4 + g * 2 + 1] = Float16(-0.35).bitPattern
                        }
                        for c in 0..<128 { qPtr[r * 128 + c] = UInt8((r * 11 + c * 5) & 0xFF) }
                    }
                }
            }

            // 1. Sparse 8 KiB NAX + Split vs MMA reference.
            var sparseP = (T: UInt32(rows), pos0: UInt32(pos0), qHeads: UInt32(qHeads),
                           headDim: UInt32(nopeDim + ropeDim), kvLoraRank: UInt32(kvRank),
                           topK: UInt32(topK), windowTokens: UInt32(window),
                           listStride: UInt32(listStride), kvMode: kvMode,
                           ropeDim: UInt32(ropeDim), qRopeOffset: UInt32(qRopeOffset),
                           qFullStride: UInt32(qFullStride), hasSink: UInt32(1), softmaxScale: Float(0.0))
            guard let (cb1, e1) = hal.begin() else { throw XCTSkip("no cmdbuf") }
            e1.setComputePipelineState(psoMMA)
            qAbs.bind(e1, index: 0); kv.k.pool.bindTo(e1, index: 1); kv.v.pool.bindTo(e1, index: 2)
            kv.k.table.bind(e1, index: 3); kv.v.table.bind(e1, index: 4); blkList.bind(e1, index: 5)
            outRef.bind(e1, index: 6)
            withUnsafeBytes(of: &sparseP) { (r: UnsafeRawBufferPointer) in e1.setBytes(r.baseAddress!, length: r.count, index: 7) }
            mlaQFull.bind(e1, index: 8); kPe.bind(e1, index: 9); sink.bind(e1, index: 10)
            e1.dispatchThreadgroups(MTLSize(width: rows, height: qHeads / 16, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            e1.setComputePipelineState(psoSparse)
            outCand.bind(e1, index: 6)
            e1.dispatchThreadgroups(MTLSize(width: rows, height: qHeads / 16, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            hal.end(cb1, e1); cb1.waitUntilCompleted(); XCTAssertNil(cb1.error)
            let dSparse = Self.maxAbsDiff(Self.readF16(outRef, count: rows * qHeads * kvRank), Self.readF16(outCand, count: rows * qHeads * kvRank))
            XCTAssertLessThan(dSparse, 5e-3, "8KiB sparse NAX kvMode=\(kvMode) diff=\(dSparse)")

            for (label, psoH, gY, tptg) in [("h32", psoSparseH32, qHeads / 32, 256), ("h64", psoSparseH64, qHeads / 64, 512)] {
                guard let (cbH, eH) = hal.begin() else { throw XCTSkip("no cmdbuf") }
                eH.setComputePipelineState(psoH)
                qAbs.bind(eH, index: 0); kv.k.pool.bindTo(eH, index: 1); kv.v.pool.bindTo(eH, index: 2)
                kv.k.table.bind(eH, index: 3); kv.v.table.bind(eH, index: 4); blkList.bind(eH, index: 5)
                outCand.bind(eH, index: 6)
                withUnsafeBytes(of: &sparseP) { (r: UnsafeRawBufferPointer) in eH.setBytes(r.baseAddress!, length: r.count, index: 7) }
                mlaQFull.bind(eH, index: 8); kPe.bind(eH, index: 9); sink.bind(eH, index: 10)
                eH.dispatchThreadgroups(MTLSize(width: rows, height: gY, depth: 1), threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1))
                hal.end(cbH, eH); cbH.waitUntilCompleted(); XCTAssertNil(cbH.error)
                let dH = Self.maxAbsDiff(Self.readF16(outRef, count: rows * qHeads * kvRank), Self.readF16(outCand, count: rows * qHeads * kvRank))
                XCTAssertLessThan(dH, 5e-3, "DSA shared \(label) kvMode=\(kvMode) diff=\(dH)")
            }

            // 2. Split-K 8 KiB NAX vs MMA on 8 rows.
            var sp8 = sparseP; sp8.T = 8
            var nSp: UInt32 = 4
            guard let (cb3, e3) = hal.begin() else { throw XCTSkip("no cmdbuf") }
            e3.setComputePipelineState(psoSplit)
            qAbs.bind(e3, index: 0); kv.k.pool.bindTo(e3, index: 1); kv.v.pool.bindTo(e3, index: 2)
            kv.k.table.bind(e3, index: 3); kv.v.table.bind(e3, index: 4); blkList.bind(e3, index: 5)
            partO.bind(e3, index: 6)
            withUnsafeBytes(of: &sp8) { (r: UnsafeRawBufferPointer) in e3.setBytes(r.baseAddress!, length: r.count, index: 7) }
            partML.bind(e3, index: 8); mlaQFull.bind(e3, index: 9); kPe.bind(e3, index: 10); sink.bind(e3, index: 11)
            e3.dispatchThreadgroups(MTLSize(width: 8, height: qHeads / 16, depth: 4), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            e3.memoryBarrier(scope: .buffers)
            e3.setComputePipelineState(psoCombine)
            partO.bind(e3, index: 0); partML.bind(e3, index: 1); outCand.bind(e3, index: 2)
            withUnsafeBytes(of: &sp8) { (r: UnsafeRawBufferPointer) in e3.setBytes(r.baseAddress!, length: r.count, index: 3) }
            withUnsafeBytes(of: &nSp) { (r: UnsafeRawBufferPointer) in e3.setBytes(r.baseAddress!, length: r.count, index: 4) }
            sink.bind(e3, index: 5)
            e3.dispatchThreadgroups(MTLSize(width: 8 * qHeads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            hal.end(cb3, e3); cb3.waitUntilCompleted(); XCTAssertNil(cb3.error)
            let dSplit = Self.maxAbsDiff(Self.readF16(outRef, count: 8 * qHeads * kvRank), Self.readF16(outCand, count: 8 * qHeads * kvRank))
            XCTAssertLessThan(dSplit, 5e-3, "Split 8KiB kvMode=\(kvMode) diff=\(dSplit)")

            if kvMode == 2 {
                let modes: [(String, UInt32, UInt32, UInt32, UInt32)] = [
                    ("densePrefix", UInt32(0), UInt32(topK), UInt32(ropeDim), UInt32(nopeDim + ropeDim)),
                    ("denseStream", UInt32(pos0), UInt32(0), UInt32(ropeDim), UInt32(nopeDim + ropeDim)),
                    ("noRope", UInt32(pos0), UInt32(topK), UInt32(0), UInt32(nopeDim)),
                ]
                for (dLbl, dPos0, dTopK, dRope, dHDim) in modes {
                    var dp = sparseP
                    dp.pos0 = dPos0; dp.topK = dTopK; dp.ropeDim = dRope; dp.headDim = dHDim
                    for (hLbl, psoH, gY, tptg) in [("h32", psoSparseH32, qHeads / 32, 256), ("h64", psoSparseH64, qHeads / 64, 512)] {
                        guard let (cbD, eD) = hal.begin() else { throw XCTSkip("no cmdbuf") }
                        eD.setComputePipelineState(psoSparse)
                        qAbs.bind(eD, index: 0); kv.k.pool.bindTo(eD, index: 1); kv.v.pool.bindTo(eD, index: 2)
                        kv.k.table.bind(eD, index: 3); kv.v.table.bind(eD, index: 4); blkList.bind(eD, index: 5)
                        outRef.bind(eD, index: 6)
                        withUnsafeBytes(of: &dp) { (r: UnsafeRawBufferPointer) in eD.setBytes(r.baseAddress!, length: r.count, index: 7) }
                        mlaQFull.bind(eD, index: 8); kPe.bind(eD, index: 9); sink.bind(eD, index: 10)
                        eD.dispatchThreadgroups(MTLSize(width: rows, height: qHeads / 16, depth: 1),
                                                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                        eD.setComputePipelineState(psoH)
                        outCand.bind(eD, index: 6)
                        eD.dispatchThreadgroups(MTLSize(width: rows, height: gY, depth: 1),
                                                threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1))
                        hal.end(cbD, eD); cbD.waitUntilCompleted(); XCTAssertNil(cbD.error)
                        let dd = Self.maxAbsDiff(Self.readF16(outRef, count: rows * qHeads * kvRank),
                                                 Self.readF16(outCand, count: rows * qHeads * kvRank))
                        XCTAssertLessThan(dd, 5e-3, "DSA shared \(hLbl) \(dLbl) diff=\(dd)")
                    }
                }
            }
        }
    }

    func testMLAAbsorbQAndProjUVNAXAllContainersAnd512D() throws {
        let geo = try Self.makeErnieGeometry()
        let hal = try QWHal(geometry: geo)
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable, "MLA proj NAX requires Metal 4")

        let pAbsNAX = try hal.pipeline("qwg_mla_absorb_q_nax")
        let pAbsNAXSub2 = try hal.pipeline("qwg_mla_absorb_q_nax_sub2")
        let pAbsScalar = try hal.pipeline("qwg_mla_absorb_q")
        let pAbsScalarSub2 = try hal.pipeline("qwg_mla_absorb_q_sub2")
        let pUVNAX = try hal.pipeline("qwg_mla_proj_uv_nax")
        let pUVNAXSub2 = try hal.pipeline("qwg_mla_proj_uv_nax_sub2")
        let pUVScalar = try hal.pipeline("qwg_mla_proj_uv")
        let pUVScalarSub2 = try hal.pipeline("qwg_mla_proj_uv_sub2")

        let T = 18, qHeads = 8, nopeDim = 256, kvRank = 512
        for vHeadDim in [128, 512] {
            let qFullStride = qHeads * (nopeDim + 64)
            let qFull = Self.makeF16Tensor(hal, shape: [T * qFullStride], values: Self.detRand(T * qFullStride, seed: 0x401))
            let uLat = Self.makeF16Tensor(hal, shape: [T * qHeads * kvRank], values: Self.detRand(T * qHeads * kvRank, seed: 0x402))
            let formats: [(String, Int, Int, Int, Bool)] = [
                ("F16", 3, 16, 32, false),
                ("HQ4_G32", 1, 4, 32, false),
                ("AQ8_G32", 0, 8, 32, false),
                ("AQ4_G64", 0, 4, 64, false),
                ("SHVQ3_G64", 2, 3, 64, false),
                ("SHVQ1_G64", 2, 1, 64, true),
            ]
            for (name, inc, bits, group, sub2) in formats {
                let wRows = qHeads * (nopeDim + vHeadDim)
                let w = Self.packTestWeights(hal: hal, inc: inc, bits: bits, group: group,
                                             rows: wRows, cols: kvRank, seed: 0x500 + UInt64(vHeadDim + bits))
                let skipFwht = (vHeadDim == 512)
                var prm: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) = (
                    UInt32(T), UInt32(qHeads), UInt32(nopeDim), UInt32(vHeadDim), UInt32(kvRank),
                    UInt32(qFullStride), UInt32(bits), UInt32(group), UInt32(inc) | (skipFwht ? 0x10000 : 0)
                )
                let zeroAbs = [Float](repeating: 0, count: T * qHeads * kvRank)
                let zeroOut = [Float](repeating: 0, count: T * qHeads * vHeadDim)
                let qAbsRef = Self.makeF16Tensor(hal, shape: [T * qHeads * kvRank], values: zeroAbs)
                let qAbsCand = Self.makeF16Tensor(hal, shape: [T * qHeads * kvRank], values: zeroAbs)
                let outRef = Self.makeF16Tensor(hal, shape: [T * qHeads * vHeadDim], values: zeroOut)
                let outCand = Self.makeF16Tensor(hal, shape: [T * qHeads * vHeadDim], values: zeroOut)

                guard let (cb, e) = hal.begin() else { throw XCTSkip("no cmdbuf") }
                e.setComputePipelineState(sub2 ? pAbsNAXSub2 : pAbsNAX)
                qFull.bind(e, index: 0); w.bind(e, index: 1); qAbsCand.bind(e, index: 2)
                withUnsafeBytes(of: &prm) { (r: UnsafeRawBufferPointer) in e.setBytes(r.baseAddress!, length: r.count, index: 3) }
                e.dispatchThreadgroups(MTLSize(width: (kvRank + 63) / 64, height: (T + 63) / 64, depth: qHeads), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                e.setComputePipelineState(sub2 ? pUVNAXSub2 : pUVNAX)
                uLat.bind(e, index: 0); w.bind(e, index: 1); outCand.bind(e, index: 2)
                withUnsafeBytes(of: &prm) { (r: UnsafeRawBufferPointer) in e.setBytes(r.baseAddress!, length: r.count, index: 3) }
                e.dispatchThreadgroups(MTLSize(width: (vHeadDim + 63) / 64, height: (T + 63) / 64, depth: qHeads), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                if inc != 3 {
                    e.setComputePipelineState(sub2 ? pAbsScalarSub2 : pAbsScalar)
                    qFull.bind(e, index: 0); w.bind(e, index: 1); qAbsRef.bind(e, index: 2)
                    withUnsafeBytes(of: &prm) { (r: UnsafeRawBufferPointer) in e.setBytes(r.baseAddress!, length: r.count, index: 3) }
                    e.dispatchThreadgroups(MTLSize(width: qHeads, height: (T + 7) / 8, depth: 1), threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
                    e.setComputePipelineState(sub2 ? pUVScalarSub2 : pUVScalar)
                    uLat.bind(e, index: 0); w.bind(e, index: 1); outRef.bind(e, index: 2)
                    withUnsafeBytes(of: &prm) { (r: UnsafeRawBufferPointer) in e.setBytes(r.baseAddress!, length: r.count, index: 3) }
                    e.dispatchThreadgroups(MTLSize(width: qHeads, height: (T + 7) / 8, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
                hal.end(cb, e); cb.waitUntilCompleted(); XCTAssertNil(cb.error)

                if inc == 3 {
                    let qf = Self.readF16(qFull, count: T * qFullStride)
                    let uF = Self.readF16(uLat, count: T * qHeads * kvRank)
                    let wf = Self.readF16(w, count: wRows * kvRank)
                    var cpuAbs = [Float](repeating: 0, count: T * qHeads * kvRank)
                    var cpuUV = [Float](repeating: 0, count: T * qHeads * vHeadDim)
                    for t in 0..<T {
                        for h in 0..<qHeads {
                            let hOff = h * (nopeDim + vHeadDim)
                            for n in 0..<kvRank {
                                var s: Float = 0
                                for k in 0..<nopeDim { s += qf[t * qFullStride + h * nopeDim + k] * wf[(hOff + k) * kvRank + n] }
                                cpuAbs[(t * qHeads + h) * kvRank + n] = s
                            }
                            for n in 0..<vHeadDim {
                                var s: Float = 0
                                for k in 0..<kvRank { s += uF[(t * qHeads + h) * kvRank + k] * wf[(hOff + nopeDim + n) * kvRank + k] }
                                cpuUV[(t * qHeads + h) * vHeadDim + n] = s
                            }
                            if !skipFwht {
                                for blk in stride(from: 0, to: vHeadDim, by: 32) { Self.fwht32(&cpuUV, base: (t * qHeads + h) * vHeadDim + blk) }
                            }
                        }
                    }
                    XCTAssertLessThan(Self.maxAbsDiff(cpuAbs, Self.readF16(qAbsCand, count: cpuAbs.count)), 5e-3, "\(name) absorb F16")
                    XCTAssertLessThan(Self.maxAbsDiff(cpuUV, Self.readF16(outCand, count: cpuUV.count)), 5e-3, "\(name) projUV F16")
                } else {
                    XCTAssertLessThan(Self.maxAbsDiff(Self.readF16(qAbsRef, count: T * qHeads * kvRank),
                                                      Self.readF16(qAbsCand, count: T * qHeads * kvRank)), 5e-3, "\(name) vD=\(vHeadDim) absorb")
                    XCTAssertLessThan(Self.maxAbsDiff(Self.readF16(outRef, count: T * qHeads * vHeadDim),
                                                      Self.readF16(outCand, count: T * qHeads * vHeadDim)), 5e-3, "\(name) vD=\(vHeadDim) projUV")
                }
            }
        }
    }

    func testFlashIndexerSelectHeadsMatchesPyTorchReference() throws {
        let geo = try Self.makeErnieGeometry(hidden: 128, topK: 16, window: 8)
        let hal = try QWHal(geometry: geo)
        let pFlash = try hal.pipeline("qwg_mla_flashindexer_select_heads")
        let pIdxScore = try hal.pipeline("qwg_mla_indexer_score_select")

        let T = 35, hidden = 128, fullH = 64, selK = 8, idxDim = 128, qCh = 256
        let qFullStride = qCh + fullH * idxDim

        let ln1Arr = Self.detRand(T * hidden, seed: 0x701, scale: 0.4)
        var gateWArr = Self.detRand(fullH * hidden, seed: 0x702, scale: 0.3)
        // Force deterministic exact tie on row 0 heads 3 and 9 to check lowest-index tiebreak.
        for k in 0..<hidden { gateWArr[9 * hidden + k] = gateWArr[3 * hidden + k] }
        var biasArr = Self.detRand(fullH, seed: 0x703, scale: 0.5)
        biasArr[9] = biasArr[3]
        let idxWArr = Self.detRand(T * fullH, seed: 0x704, scale: 1.0)
        let qFullArr = Self.detRand(T * qFullStride, seed: 0x705, scale: 0.5)

        let ln1 = Self.makeF16Tensor(hal, shape: [T * hidden], values: ln1Arr)
        let gateW = Self.makeF16Tensor(hal, shape: [fullH * hidden], values: gateWArr)
        let bias = Self.makeF32Tensor(hal, shape: [fullH], values: biasArr)
        let idxW = Self.makeF32Tensor(hal, shape: [T * fullH], values: idxWArr)
        let mlaQ = Self.makeF16Tensor(hal, shape: [T * qFullStride], values: qFullArr)
        let qSelOut = Self.makeF16Tensor(hal, shape: [T * selK * idxDim], values: [Float](repeating: 0, count: T * selK * idxDim))
        let wSelOut = Self.makeF32Tensor(hal, shape: [T * selK], values: [Float](repeating: 0, count: T * selK))

        var fiP = (UInt32(T), UInt32(hidden), UInt32(fullH), UInt32(selK), UInt32(idxDim), UInt32(qCh), UInt32(qFullStride))
        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no cmdbuf") }
        enc.setComputePipelineState(pFlash)
        ln1.bind(enc, index: 0); gateW.bind(enc, index: 1); bias.bind(enc, index: 2)
        idxW.bind(enc, index: 3); mlaQ.bind(enc, index: 4); qSelOut.bind(enc, index: 5); wSelOut.bind(enc, index: 6)
        withUnsafeBytes(of: &fiP) { (r: UnsafeRawBufferPointer) in enc.setBytes(r.baseAddress!, length: r.count, index: 7) }
        enc.dispatchThreadgroups(MTLSize(width: (T + 31) / 32, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        hal.end(cb, enc); cb.waitUntilCompleted(); XCTAssertNil(cb.error)

        let gpuQSel = Self.readF16(qSelOut, count: T * selK * idxDim)
        let gpuWSel = Self.readF32(wSelOut, count: T * selK)
        let ln1F = ln1Arr.map { Float(Float16($0)) }
        let gwF = gateWArr.map { Float(Float16($0)) }
        let qfF = qFullArr.map { Float(Float16($0)) }

        var refWSel = [Float](repeating: 0, count: T * selK), refQSel = [Float](repeating: 0, count: T * selK * idxDim)
        for t in 0..<T {
            var gates = [Float](repeating: 0, count: fullH), scores = [Float](repeating: 0, count: fullH)
            for h in 0..<fullH {
                var dot: Float = 0
                for k in 0..<hidden { dot += ln1F[t * hidden + k] * gwF[h * hidden + k] }
                gates[h] = dot; scores[h] = dot + biasArr[h]
            }
            for k in 0..<selK {
                var bestH = 0, bestS = -Float.infinity
                for h in 0..<fullH where scores[h] > bestS { bestS = scores[h]; bestH = h }
                scores[bestH] = -Float.infinity
                refWSel[t * selK + k] = idxWArr[t * fullH + bestH] * gates[bestH]
                for d in 0..<idxDim { refQSel[(t * selK + k) * idxDim + d] = qfF[t * qFullStride + qCh + bestH * idxDim + d] }
            }
        }
        XCTAssertLessThan(Self.maxAbsDiff(refWSel, gpuWSel), 2e-2, "FlashIndexer w_sel parity")
        XCTAssertEqual(refQSel, gpuQSel, "FlashIndexer q_sel exact head gather parity")

        // Verify 8-head indexer scorer (`qChannels=0, idxHeads=8`) consumes compact `qSelOut`/`wSelOut`.
        let maxBlocks = 128, topK = 16, listStride = 64
        let pooled = Self.makeF16Tensor(hal, shape: [maxBlocks * idxDim], values: Self.detRand(maxBlocks * idxDim, seed: 0x706))
        let scoreScratch = Self.makeF32Tensor(hal, shape: [T * maxBlocks], values: [Float](repeating: 0, count: T * maxBlocks))
        let blkList = Self.makeI32Tensor(hal, shape: [T * listStride], values: [Int32](repeating: -1, count: T * listStride))
        let blkCnt = Self.makeI32Tensor(hal, shape: [T], values: [Int32](repeating: 0, count: T))
        var idxP: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) = (UInt32(T), 256, 0, UInt32(selK), UInt32(idxDim), UInt32(topK), UInt32(listStride), UInt32(maxBlocks))
        guard let (cb2, enc2) = hal.begin() else { throw XCTSkip("no cmdbuf") }
        enc2.setComputePipelineState(pIdxScore)
        qSelOut.bind(enc2, index: 0); wSelOut.bind(enc2, index: 1); pooled.bind(enc2, index: 2)
        scoreScratch.bind(enc2, index: 3); blkList.bind(enc2, index: 4); blkCnt.bind(enc2, index: 5)
        withUnsafeBytes(of: &idxP) { (r: UnsafeRawBufferPointer) in enc2.setBytes(r.baseAddress!, length: r.count, index: 6) }
        enc2.dispatchThreadgroups(MTLSize(width: T, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        hal.end(cb2, enc2); cb2.waitUntilCompleted(); XCTAssertNil(cb2.error)
        XCTAssertEqual(Int(blkCnt.region.buffer.contents().assumingMemoryBound(to: Int32.self)[0]), topK)
    }

    func testHCAVHAInvRopeNAXFloatOsWoAAndHalf4QPrep() throws {
        let geo = try Self.makeErnieGeometry()
        let hal = try QWHal(geometry: geo)
        let pQPrep = try hal.pipeline("qwg_hca_q_prep")
        for qH in [20, 64] {
            let T = 17, headDim = 512, nopeDim = 448, ropeDim = 64, oRank = 32, oGroups = 16, qUpStride = qH * headDim + oGroups * oRank, pos0 = 123, rmsEps: Float = 1e-5, mscale: Float = 1.25
            let inArr = Self.detRand(T * qUpStride, seed: 0x801 + UInt64(qH)), qBuf = Self.makeF16Tensor(hal, shape: [T * qUpStride], values: inArr)
            let invArr = (0..<(ropeDim / 2)).map { i in pow(Float(1_000_000.0), -Float(2 * i) / Float(ropeDim)) }
            let invBuf = Self.makeF32Tensor(hal, shape: [ropeDim / 2], values: invArr)
            var qp: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, Float, Float) = (UInt32(T), UInt32(pos0), UInt32(qH), UInt32(headDim), UInt32(nopeDim), UInt32(ropeDim), UInt32(qUpStride), 0, rmsEps, mscale)
            guard let (cb, e) = hal.begin() else { throw XCTSkip("no cmdbuf") }
            e.setComputePipelineState(pQPrep); qBuf.bind(e, index: 0); invBuf.bind(e, index: 1)
            withUnsafeBytes(of: &qp) { (r: UnsafeRawBufferPointer) in e.setBytes(r.baseAddress!, length: r.count, index: 2) }
            e.dispatchThreadgroups(MTLSize(width: (qH + 7) / 8, height: T, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            hal.end(cb, e); cb.waitUntilCompleted(); XCTAssertNil(cb.error)
            let got = Self.readF16(qBuf, count: T * qUpStride)
            var cpu = inArr.map { Float(Float16($0)) }
            for t in 0..<T {
                for h in 0..<qH {
                    let base = t * qUpStride + h * headDim
                    var sq: Float = 0
                    for d in 0..<headDim { sq += cpu[base + d] * cpu[base + d] }
                    let invR = 1.0 / sqrt(sq / Float(headDim) + rmsEps)
                    for d in 0..<nopeDim { cpu[base + d] *= invR }
                    for i in 0..<(ropeDim / 2) {
                        let ev = cpu[base + nopeDim + 2 * i] * invR
                        let od = cpu[base + nopeDim + 2 * i + 1] * invR
                        let ang = Float(pos0 + t) * invArr[i]
                        let c = cos(ang) * mscale, s = sin(ang) * mscale
                        cpu[base + nopeDim + 2 * i] = ev * c - od * s
                        cpu[base + nopeDim + 2 * i + 1] = od * c + ev * s
                    }
                }
            }
            XCTAssertLessThan(Self.maxAbsDiff(cpu, got), 3e-3, "q_prep half4 qHeads=\(qH)")
        }

        try XCTSkipUnless(hal.caps.isMetal4TensorCapable, "HCA VHA / WoA NAX requires Metal 4")

        // 2. qwg_hca_vha_invrope_nax (BLOCK=128, register H_32) across hasVHA & skipFwht vs qwg_hca_vha_invrope.
        let aRows = 19, qHeads = 64, headDim = 512
        let pVhaRef = try hal.pipeline("qwg_hca_vha_invrope")
        let pVhaNAX = try hal.pipeline("qwg_hca_vha_invrope_nax")
        let attIn = Self.makeF16Tensor(hal, shape: [aRows * qHeads * headDim], values: Self.detRand(aRows * qHeads * headDim, seed: 0x901))
        let vhaMix = Self.makeF16Tensor(hal, shape: [qHeads * qHeads], values: Self.detRand(qHeads * qHeads, seed: 0x904, scale: 0.2))
        let ropeInv = Self.makeF32Tensor(hal, shape: [32], values: (0..<32).map { i in pow(Float(1_000_000.0), -Float(2 * i) / 64.0) })
        let zRef = Self.makeF16Tensor(hal, shape: [aRows * qHeads * headDim], values: [Float](repeating: 0, count: aRows * qHeads * headDim))
        let zNAX = Self.makeF16Tensor(hal, shape: [aRows * qHeads * headDim], values: [Float](repeating: 0, count: aRows * qHeads * headDim))
        for hasVHAFlag: UInt32 in [0, 1, 0x10001] {
            var vP: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, Float, UInt32) = (
                UInt32(aRows), 77, UInt32(qHeads), UInt32(headDim), 448, 64, 0.85, hasVHAFlag
            )
            guard let (cbV, eV) = hal.begin() else { throw XCTSkip("no cmdbuf") }
            eV.setComputePipelineState(pVhaRef)
            attIn.bind(eV, index: 0); vhaMix.bind(eV, index: 1); ropeInv.bind(eV, index: 2); zRef.bind(eV, index: 3)
            withUnsafeBytes(of: &vP) { (r: UnsafeRawBufferPointer) in eV.setBytes(r.baseAddress!, length: r.count, index: 4) }
            eV.dispatchThreadgroups(MTLSize(width: headDim / 32, height: aRows, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            eV.setComputePipelineState(pVhaNAX)
            zNAX.bind(eV, index: 3)
            eV.dispatchThreadgroups(MTLSize(width: headDim / 128, height: aRows, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            hal.end(cbV, eV); cbV.waitUntilCompleted(); XCTAssertNil(cbV.error)
            XCTAssertLessThan(Self.maxAbsDiff(Self.readF16(zRef, count: aRows * qHeads * headDim),
                                              Self.readF16(zNAX, count: aRows * qHeads * headDim)), 5e-3, "VHA NAX flag=\(hasVHAFlag)")
        }

        // 3. HCAWoANaxTg parity: isolated t16 BM=16 (T=8,16) & BM=64 (T=19) across AQ8_G32, HQ4_G32, SHVQ1_G64.
        let pWoAMMA = try hal.pipeline("qwg_hca_grouped_wo_a_mma"), pWoAMMASub2 = try hal.pipeline("qwg_hca_grouped_wo_a_mma_sub2")
        let pWoANAX = try hal.pipeline("qwg_hca_grouped_wo_a_nax"), pWoANAXSub2 = try hal.pipeline("qwg_hca_grouped_wo_a_nax_sub2")
        let pWoANAXT16 = try hal.pipeline("qwg_hca_grouped_wo_a_nax_t16"), pWoANAXT16Sub2 = try hal.pipeline("qwg_hca_grouped_wo_a_nax_t16_sub2")
        let oGroups = 16, oRank = 128, groupIn = 2048
        let qGate = Self.makeF16Tensor(hal, shape: [aRows * (qHeads * headDim + oGroups * oRank)],
                                       values: Self.detRand(aRows * (qHeads * headDim + oGroups * oRank), seed: 0x905))
        let zZero = [Float](repeating: 0, count: aRows * oGroups * oRank)
        let zgRef = Self.makeF16Tensor(hal, shape: [aRows * oGroups * oRank], values: zZero)
        let zgNAX = Self.makeF16Tensor(hal, shape: [aRows * oGroups * oRank], values: zZero)
        for (inc, bits, group, sub2) in [(0, 8, 32, false), (1, 4, 32, false), (2, 1, 64, true)] {
            let woW = Self.packTestWeights(hal: hal, inc: inc, bits: bits, group: group, rows: oGroups * oRank, cols: groupIn, seed: 0x906 + UInt64(bits))
            for woARows in [8, 16, aRows] {
                var wP: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) = (
                    UInt32(woARows), UInt32(oGroups), UInt32(oRank), UInt32(groupIn),
                    UInt32(qHeads * headDim + oGroups * oRank), UInt32(qHeads * headDim), UInt32(bits), UInt32(group), UInt32(inc)
                )
                let isT16 = woARows <= 16
                let gNAX = MTLSize(width: oRank / 64, height: isT16 ? (woARows + 15) / 16 : (woARows + 63) / 64, depth: oGroups)
                guard let (cbW, eW) = hal.begin() else { throw XCTSkip("no cmdbuf") }
                eW.setComputePipelineState(sub2 ? pWoAMMASub2 : pWoAMMA)
                zRef.bind(eW, index: 0); woW.bind(eW, index: 1); qGate.bind(eW, index: 2); zgRef.bind(eW, index: 3)
                withUnsafeBytes(of: &wP) { (r: UnsafeRawBufferPointer) in eW.setBytes(r.baseAddress!, length: r.count, index: 4) }
                eW.dispatchThreadgroups(MTLSize(width: oGroups, height: (woARows + 31) / 32, depth: oRank / 128), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                eW.setComputePipelineState(isT16 ? (sub2 ? pWoANAXT16Sub2 : pWoANAXT16) : (sub2 ? pWoANAXSub2 : pWoANAX))
                zgNAX.bind(eW, index: 3)
                eW.dispatchThreadgroups(gNAX, threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                hal.end(cbW, eW); cbW.waitUntilCompleted(); XCTAssertNil(cbW.error)
                XCTAssertLessThan(Self.maxAbsDiff(Self.readF16(zgRef, count: woARows * oGroups * oRank), Self.readF16(zgNAX, count: woARows * oGroups * oRank)),
                                  5e-3, "WoA T=\(woARows) inc=\(inc) bits=\(bits)")
            }
        }

        // 4. Isolated SHVQ4 DirectLUT HCA wo_a: BM=16 t16 (T=8,16 shvq4_g32/g64) & BM=64 (T=19 shvq4_g64).
        let pWoANAX4 = try hal.pipeline("qwg_hca_grouped_wo_a_nax_shvq4"), pWoANAXT164 = try hal.pipeline("qwg_hca_grouped_wo_a_nax_t16_shvq4")
        for (woARows, shvqGrp) in [(8, 32), (8, 64), (16, 64), (aRows, 64)] {
            let wo4 = Self.packTestWeights(hal: hal, inc: 2, bits: 4, group: shvqGrp, rows: oGroups * oRank, cols: groupIn, seed: 0x90A + UInt64(shvqGrp))
            var p4: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) = (
                UInt32(woARows), UInt32(oGroups), UInt32(oRank), UInt32(groupIn),
                UInt32(qHeads * headDim + oGroups * oRank), UInt32(qHeads * headDim), 4, UInt32(shvqGrp), 2
            )
            let isT16 = woARows <= 16
            let gNAX = MTLSize(width: oRank / 64, height: isT16 ? (woARows + 15) / 16 : (woARows + 63) / 64, depth: oGroups)
            guard let (cb4, e4) = hal.begin() else { throw XCTSkip("no cmdbuf") }
            e4.setComputePipelineState(isT16 ? pWoANAXT16 : pWoANAX)
            zRef.bind(e4, index: 0); wo4.bind(e4, index: 1); qGate.bind(e4, index: 2); zgRef.bind(e4, index: 3)
            withUnsafeBytes(of: &p4) { (r: UnsafeRawBufferPointer) in e4.setBytes(r.baseAddress!, length: r.count, index: 4) }
            e4.dispatchThreadgroups(gNAX, threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            e4.setComputePipelineState(isT16 ? pWoANAXT164 : pWoANAX4)
            zgNAX.bind(e4, index: 3)
            e4.dispatchThreadgroups(gNAX, threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            hal.end(cb4, e4); cb4.waitUntilCompleted(); XCTAssertNil(cb4.error)
            XCTAssertLessThan(Self.maxAbsDiff(Self.readF16(zgRef, count: woARows * oGroups * oRank), Self.readF16(zgNAX, count: woARows * oGroups * oRank)),
                              1e-4, "nax_shvq4 T=\(woARows) g=\(shvqGrp)")
        }
    }

    func testOperatorEndToEndMLANaxProjFlashIndexerAndHCAVhaNax() throws {
        let geo = try Self.makeErnieGeometry(qHeads: 64, hidden: 256, topK: 32, window: 16, vHeadDim: 512)
        let hal = try QWHal(geometry: geo)
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable, "NAX operator end-to-end requires Metal 4")

        // 1. MLA operator across denseLim=48 at pos0=32, rows=32 + NAX absorb/proj + FlashIndexer.
        let pBytes = QWKVCache.pageBytes(mode: .q4, geometry: geo)
        let tables = QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs
        let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: pBytes * 2 + tables, gdnStates: 4096)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: [1], mode: .q4, geometry: geo)
        let seq = try cache.allocate(promptTokens: 96)
        let kv = try cache.layerViews(seq, layer: 1)

        let mla = try QWMLAAttnOperator(hal: hal, geometry: geo, numAttnLayers: 1, maxContext: 256, maxPrefillChunk: 32)
        let qNorm = Self.makeF16Tensor(hal, shape: [128], values: [Float](repeating: 1, count: 128))
        let kvNorm = Self.makeF16Tensor(hal, shape: [512], values: [Float](repeating: 1, count: 512))
        let kNormW = Self.makeF16Tensor(hal, shape: [128], values: [Float](repeating: 1, count: 128))
        let kNormB = Self.makeF16Tensor(hal, shape: [128], values: [Float](repeating: 0, count: 128))
        let wGate = Self.makeF16Tensor(hal, shape: [64 * 256], values: Self.detRand(64 * 256, seed: 0xA01))
        let kvBData = Self.packTestWeights(hal: hal, inc: 1, bits: 4, group: 32, rows: 64 * (256 + 512), cols: 512, seed: 0xA02)
        let kvB = QWWeightView(blocks: kvBData, n: 64 * (256 + 512), k: 512, containerKind: .incumbentHQ4G32)!
        let qBData = Self.packTestWeights(hal: hal, inc: 1, bits: 4, group: 32, rows: mla.qFullStride, cols: 128, seed: 0xA03)
        let qB = QWWeightView(blocks: qBData, n: mla.qFullStride, k: 128, containerKind: .incumbentHQ4G32)!
        let vhaMix = Self.makeF16Tensor(hal, shape: [64 * 64], values: Self.detRand(64 * 64, seed: 0xA04, scale: 0.1))
        let sink = Self.makeF32Tensor(hal, shape: [64], values: [Float](repeating: -0.5, count: 64))
        let fGate = Self.makeF16Tensor(hal, shape: [64 * 256], values: Self.detRand(64 * 256, seed: 0xA05, scale: 0.2))
        let fBias = Self.makeF32Tensor(hal, shape: [64], values: Self.detRand(64, seed: 0xA06, scale: 0.5))
        let mlaW = QWMLAAttnOperator.LayerWeights(qANormW: qNorm, kvANormW: kvNorm, kNormW: kNormW, kNormB: kNormB, wGateWeights: wGate, ape: nil, qBProj: qB, kvBProj: kvB, vhaMix: vhaMix, attnSink: sink, hasOGate: true, flashGateW: fGate, flashBias: fBias)
        let ln1 = Self.makeF16Tensor(hal, shape: [32 * 256], values: Self.detRand(32 * 256, seed: 0xA07))
        let zAtt = [Float](repeating: 0, count: 32 * 64 * 512)
        let attOutA = Self.makeF16Tensor(hal, shape: [32 * 64 * 512], values: zAtt), attOutB = Self.makeF16Tensor(hal, shape: [32 * 64 * 512], values: zAtt)
        func runMLA(projNax: Bool, flash: Bool, out: QWTensorValue) throws {
            mla.projNAXEnabled = projNax; mla.flashIndexerEnabled = flash
            let lp = mla.region.pointer.advanced(by: mla.mlaLatentScratch.offset).assumingMemoryBound(to: Float16.self)
            for (i, v) in Self.detRand(32 * mla.latentStride, seed: 0xA08).enumerated() { lp[i] = Float16(v) }
            let qp = mla.region.pointer.advanced(by: mla.mlaQFullScratch.offset).assumingMemoryBound(to: Float16.self)
            for (i, v) in Self.detRand(32 * mla.qFullStride, seed: 0xA09).enumerated() { qp[i] = Float16(v) }
            guard let (cb, e) = hal.begin() else { throw XCTSkip("no cmdbuf") }
            mla.encodeLatentPrepAndPool(e, ln1: ln1, mlaW: mlaW, kv: kv, fullAttnLayer: 0, pos0: 0, rows: 32)
            mla.encodeLatentPrepAndPool(e, ln1: ln1, mlaW: mlaW, kv: kv, fullAttnLayer: 0, pos0: 32, rows: 32)
            mla.encodeScoreSelectAbsorbAttendAndProjUV(e, mlaW: mlaW, kv: kv, attOut: out, fullAttnLayer: 0, pos0: 32, rows: 32, ln1: ln1)
            hal.end(cb, e); cb.waitUntilCompleted(); XCTAssertNil(cb.error)
        }
        try runMLA(projNax: false, flash: false, out: attOutA); try runMLA(projNax: true, flash: false, out: attOutB)
        let mlaValsA = Self.readF16(attOutA, count: 32 * 64 * 512), mlaValsB = Self.readF16(attOutB, count: 32 * 64 * 512)
        XCTAssertTrue(mlaValsA.allSatisfy(\.isFinite) && mlaValsA.contains(where: { abs($0) > 1e-4 }), "MLA output non-trivial")
        XCTAssertLessThan(Self.maxAbsDiff(mlaValsA, mlaValsB), 1e-2, "MLA proj NAX toggle parity")
        try runMLA(projNax: true, flash: true, out: attOutB)
        XCTAssertEqual(Int(mla.region.pointer.advanced(by: mla.blockCountScratch.offset).assumingMemoryBound(to: Int32.self)[31]), 48)

        let geoHCA = try Self.makeErnieGeometry(qHeads: 64, hidden: 256, topK: 32, window: 16, vHeadDim: 512, oLoraRank: 64), halHCA = try QWHal(geometry: geoHCA)
        let hca = try QWHCAAttnOperator(hal: halHCA, geometry: geoHCA, numHcaLayers: 1, maxContext: 512, maxPrefillChunk: 32)
        let woA = QWWeightView(blocks: Self.packTestWeights(hal: halHCA, inc: 1, bits: 4, group: 32, rows: 16 * 64, cols: (64 / 16) * 512, seed: 0xB01), n: 16 * 64, k: (64 / 16) * 512, containerKind: .incumbentHQ4G32)!
        let hcaW = QWHCAAttnOperator.LayerWeights(qNormW: Self.makeF16Tensor(halHCA, shape: [128], values: [Float](repeating: 1, count: 128)), kvNormW: Self.makeF16Tensor(halHCA, shape: [512], values: [Float](repeating: 1, count: 512)), compNormW: Self.makeF16Tensor(halHCA, shape: [512], values: [Float](repeating: 1, count: 512)), ape: Self.makeF32Tensor(halHCA, shape: [128, 512], values: [Float](repeating: 0.01, count: 128 * 512)), woA: woA, vhaMix: Self.makeF16Tensor(halHCA, shape: [64 * 64], values: Self.detRand(64 * 64, seed: 0xA04, scale: 0.1)), attnSink: Self.makeF32Tensor(halHCA, shape: [64], values: [Float](repeating: -0.5, count: 64)))
        let hOutA = Self.makeF16Tensor(halHCA, shape: [18 * hca.zOutStride], values: [Float](repeating: 0, count: 18 * hca.zOutStride)), hOutB = Self.makeF16Tensor(halHCA, shape: [18 * hca.zOutStride], values: [Float](repeating: 0, count: 18 * hca.zOutStride))
        for t in [hca.layers[0].compKV, hca.layers[0].swaKV] {
            let p = hca.region.pointer.advanced(by: t.offset).assumingMemoryBound(to: Float16.self)
            for (i, v) in Self.detRand(t.shape.elementCount, seed: 0xB02).enumerated() { p[i] = Float16(v) }
        }
        func runHCA(rows: Int, vhaNax: Bool, woANax: Bool, dst: QWTensorValue) throws {
            hca.hcaVhaNAXEnabled = vhaNax; hca.hcaWoANAXEnabled = woANax
            let qp = hca.region.pointer.advanced(by: hca.qUpGateScratch.offset).assumingMemoryBound(to: Float16.self)
            for (i, v) in Self.detRand(rows * hca.qUpStride, seed: 0xB03).enumerated() { qp[i] = Float16(v) }
            guard let (cb, e) = halHCA.begin() else { throw XCTSkip("no cmdbuf") }
            hca.encodeQPrepAttendVHAAndWoA(e, hcaW: hcaW, zGatedOut: dst, hcaLayer: 0, pos0: 260, rows: rows)
            halHCA.end(cb, e); cb.waitUntilCompleted(); XCTAssertNil(cb.error)
        }
        try runHCA(rows: 18, vhaNax: false, woANax: false, dst: hOutA); try runHCA(rows: 18, vhaNax: true, woANax: true, dst: hOutB)
        XCTAssertLessThan(Self.maxAbsDiff(Self.readF16(hOutA, count: 18 * hca.zOutStride), Self.readF16(hOutB, count: 18 * hca.zOutStride)), 1e-2, "HCA VHA+WoA NAX T=18")
        try runHCA(rows: 8, vhaNax: false, woANax: false, dst: hOutA); try runHCA(rows: 8, vhaNax: false, woANax: true, dst: hOutB)
        XCTAssertLessThan(Self.maxAbsDiff(Self.readF16(hOutA, count: 8 * hca.zOutStride), Self.readF16(hOutB, count: 8 * hca.zOutStride)), 5e-3, "HCA WoA NAX T=8 BM=16")
    }

    func testHQ8G32EncodeNaxF32BM16ParityVsAQGemv() throws {
        let hal = try QWHal(geometry: try Self.makeErnieGeometry())
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable, "AQ8 NAX BM=16 requires Metal 4")
        let n = 256, k = 512, maxRows = 16
        let blk = Self.packTestWeights(hal: hal, inc: 0, bits: 8, group: 32, rows: n, cols: k, seed: 0xC01)
        let hq8W = QWWeightView(blocks: blk, n: n, k: k, bits: 8, groupSize: 32, isRotated: true, isIncumbentContainer: false, isSHVQ: false)!
        let xAll = Self.makeF16Tensor(hal, shape: [maxRows * k], values: Self.detRand(maxRows * k, seed: 0xC02, scale: 0.25))
        let yRef = Self.makeF16Tensor(hal, shape: [maxRows * n], values: [Float](repeating: 0, count: maxRows * n))
        let yNax = Self.makeF16Tensor(hal, shape: [maxRows * n], values: [Float](repeating: 0, count: maxRows * n))
        let pGemv = try hal.pipeline(QWGemmAQ.decodeKernelName(bits: 8, group: 32, isSHVQ: false)!)
        for rows in [4, 8, 12, 16] {
            let xSub = xAll.slice(byteOffset: 0, count: rows * k)
            let yRefSub = yRef.slice(byteOffset: 0, count: rows * n)
            let yNaxSub = yNax.slice(byteOffset: 0, count: rows * n)
            guard let (cb, e) = hal.begin() else { throw XCTSkip("no cmdbuf") }
            e.setComputePipelineState(pGemv)
            hq8W.blocks.bind(e, index: 0)
            var kU = UInt32(k)
            e.setBytes(&kU, length: 4, index: 5)
            var r0 = 0
            while r0 < rows {
                let c = (rows - r0 > 8 && rows - r0 - 8 == 1) ? 5 : min(8, rows - r0)
                let xc = xSub.slice(byteOffset: r0 * k * 2, count: c * k)
                let yc = yRefSub.slice(byteOffset: r0 * n * 2, count: c * n)
                xc.bind(e, index: 1); yc.bind(e, index: 3); yc.bind(e, index: 8)
                var nU = UInt32(n) | 0x4000_0000 | (UInt32(c) << 24)
                e.setBytes(&nU, length: 4, index: 4)
                e.dispatchThreadgroups(MTLSize(width: (n + 15) / 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                r0 += c
            }
            try QWGemm.encodeNaxF32(e, hal: hal, weight: hq8W, x16: xSub, y16: xSub, y32: yNaxSub, rows: rows)
            hal.end(cb, e); cb.waitUntilCompleted(); XCTAssertNil(cb.error)
            XCTAssertLessThan(Self.maxAbsDiff(Self.readF16(yRefSub, count: rows * n), Self.readF16(yNaxSub, count: rows * n)), 2e-2, "HQ8 BM=16 T=\(rows)")
        }
    }

    func testHCAGroupedWoAT4087AQ8AndHQ4InformationalBenchmark() throws {
        let hal = try QWHal(geometry: try Self.makeErnieGeometry())
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable, "HCA wo_a NAX requires Metal 4")
        let T = 4087, oGroups = 8, oRank = 1024, groupIn = 512, outCols = oGroups * oRank
        let pMMA = try hal.pipeline("qwg_hca_grouped_wo_a_mma"), pNAX = try hal.pipeline("qwg_hca_grouped_wo_a_nax")
        func makeBuf(_ count: Int, seed: UInt64?) -> QWTensorValue {
            let bytes = max(256, (count * 2 + 255) & ~255)
            let buf = hal.device.makeBuffer(length: bytes, options: .storageModeShared)!
            if let s0 = seed {
                let lut = Self.detRand(4096, seed: s0, scale: 0.25).map { Float16($0) }
                let dst = buf.contents().assumingMemoryBound(to: Float16.self)
                lut.withUnsafeBufferPointer { lB in
                    guard let lp = lB.baseAddress else { return }
                    var i = 0
                    while i + 4096 <= count { memcpy(dst + i, lp, 8192); i += 4096 }
                    while i < count { dst[i] = lp[i & 4095]; i += 1 }
                }
            }
            return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: bytes), offset: 0, shape: QWTensorShape([count], .f16))
        }
        let zIn = makeBuf(T * oGroups * groupIn, seed: 0xD01), qGate = makeBuf(T * outCols, seed: 0xD02)
        let zgRef = makeBuf(T * outCols, seed: nil), zgNAX = makeBuf(T * outCols, seed: nil)
        var best: [String: Double] = [:]
        for (lbl, inc, bits, group) in [("AQ8", 0, 8, 32), ("HQ4", 1, 4, 32)] {
            let woW = Self.packTestWeights(hal: hal, inc: inc, bits: bits, group: group, rows: outCols, cols: groupIn, seed: 0xD03 + UInt64(bits))
            var wP: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) = (
                UInt32(T), UInt32(oGroups), UInt32(oRank), UInt32(groupIn), UInt32(outCols), 0, UInt32(bits), UInt32(group), UInt32(inc))
            func launch(_ pso: any MTLComputePipelineState, dst: QWTensorValue, grid: MTLSize) throws -> Double {
                guard let (cb, e) = hal.begin() else { throw XCTSkip("no cmdbuf") }
                e.setComputePipelineState(pso)
                zIn.bind(e, index: 0); woW.bind(e, index: 1); qGate.bind(e, index: 2); dst.bind(e, index: 3)
                withUnsafeBytes(of: &wP) { (r: UnsafeRawBufferPointer) in e.setBytes(r.baseAddress!, length: r.count, index: 4) }
                e.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                hal.end(cb, e); cb.waitUntilCompleted(); XCTAssertNil(cb.error)
                return (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
            }
            let gMMA = MTLSize(width: oGroups, height: (T + 31) / 32, depth: oRank / 128)
            let gNAX = MTLSize(width: oRank / 64, height: (T + 63) / 64, depth: oGroups)
            _ = try launch(pMMA, dst: zgRef, grid: gMMA); _ = try launch(pNAX, dst: zgNAX, grid: gNAX)
            let pA = zgRef.region.pointer.assumingMemoryBound(to: UInt16.self), pB = zgNAX.region.pointer.assumingMemoryBound(to: UInt16.self)
            var maxDiff: Float = 0, i = 0, lim = T * outCols
            while i < lim {
                let ua = pA[i], ub = pB[i]
                if ua != ub || (ua & 0x7C00) == 0x7C00 {
                    let fa = Float(Float16(bitPattern: ua)), fb = Float(Float16(bitPattern: ub))
                    if !(fa.isFinite && fb.isFinite) { maxDiff = .infinity; break }
                    maxDiff = max(maxDiff, abs(fa - fb))
                }
                i += 1
            }
            XCTAssertLessThan(maxDiff, 5e-3, "T=4087 wo_a \(lbl) parity diff=\(maxDiff)")
            var minMs = Double.greatestFiniteMagnitude
            for iter in 0..<3 { let ms = try launch(pNAX, dst: zgNAX, grid: gNAX); if iter >= 1 { minMs = min(minMs, ms) } }
            best[lbl] = minMs
        }
        print(String(format: "[QWErnie55AttnOptTests] T=4087 wo_a minGPU: AQ8=%.3f ms vs HQ4=%.3f ms", best["AQ8"]!, best["HQ4"]!))
    }
}
