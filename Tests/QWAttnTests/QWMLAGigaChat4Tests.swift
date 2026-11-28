// QWMLAGigaChat4Tests.swift — Parity tests for GigaChat4 Decoupled-RoPE MLA-DSA:
// GatedLatentNorm, alphaQ/alphaKV, indexer RMS k_norm, softmaxScale, oGateInQKV, denseOnly, and B=2..8 batch decode.
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

final class QWMLAGigaChat4Tests: XCTestCase {
    private func skipUnlessMetal() throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true"
                || QWDeviceCaps.current?.familyApple9 != true,
            "requires bare-metal Apple Silicon GPU"
        )
    }

    private static func makeGC4LikeGeometry(
        hidden: Int = 128,
        qHeads: Int = 16,
        qLora: Int = 64,
        kvLora: Int = 512,
        nopeDim: Int = 192,
        ropeDim: Int = 64,
        vDim: Int = 256,
        idxHeads: Int = 8,
        idxDim: Int = 128,
        idxTopK: Int = 8,
        window: Int = 4,
        rmsEps: Float = 1e-6,
        ropeTheta: Double = 80_000.0
    ) throws -> QWModelGeometry {
        let cfg = """
        {
          "architectures": ["Ernie5_5_MoeForCausalLM"], "model_type": "ernie5_5_moe",
          "vocab_size": 1024, "hidden_size": \(hidden), "intermediate_size": \(hidden),
          "num_hidden_layers": 2, "max_position_embeddings": 131072, "rms_norm_eps": \(rmsEps),
          "num_attention_heads": \(qHeads), "num_key_value_heads": 1, "head_dim": 512,
          "compress_ratios": [4, -2], "default_compress_ratio": 4,
          "q_lora_rank": 128, "kv_lora_rank": 512,
          "compress_qk_nope_head_dim": 448, "compress_qk_rope_head_dim": 64, "compress_v_head_dim": 512,
          "o_groups": 2, "o_lora_rank": 128, "sliding_window": \(window),
          "compress_rope_theta": 160000.0, "hca_rope_type": "yarn",
          "rope_theta": \(ropeTheta), "rotary_percent": 1.0, "rope_interleave": true,
          "hybrid_mla_attention": "mqa_dsa", "hybrid_mla_num_attention_heads": \(qHeads), "hybrid_mla_num_key_value_heads": \(qHeads),
          "hybrid_mla_q_lora_rank": \(qLora), "hybrid_mla_kv_lora_rank": \(kvLora),
          "hybrid_mla_qk_nope_head_dim": \(nopeDim), "hybrid_mla_qk_rope_head_dim": \(ropeDim), "hybrid_mla_v_head_dim": \(vDim),
          "mqa_split_kv_b_proj": true, "index_n_heads": \(idxHeads), "index_head_dim": \(idxDim), "index_topk": \(idxTopK),
          "use_fast_hadamard": true, "use_vha_attention": false, "add_full_attention_sink_bias": false,
          "use_gated_attn": true, "use_attn_merger": false, "gated_attn_use_q_lora": false, "eos_token_id": 2
        }
        """.data(using: .utf8)!
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: cfg))
    }

    private static func bf16Bits(_ value: Float) -> UInt16 {
        let u = value.bitPattern
        if (u & 0x7FFF_FFFF) > 0x7F80_0000 { return UInt16((u >> 16) | 0x0040) }
        let lsb = (u >> 16) & 1
        return UInt16(truncatingIfNeeded: (u &+ 0x7FFF &+ lsb) >> 16)
    }

    private static func bf16Round(_ value: Float) -> Float {
        Float(bitPattern: UInt32(bf16Bits(value)) << 16)
    }

    private static func fwht32InPlace(_ v: inout [Float]) {
        v.withUnsafeMutableBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            for b in stride(from: 0, to: buf.count, by: 32) {
                QWQuant.fwht32OrthoBlock(base + b)
            }
        }
    }

    private static func fwht128InPlace(_ v: inout [Float]) {
        var h = 1
        while h < 128 {
            var i = 0
            while i < 128 {
                for j in i..<(i + h) {
                    let a = v[j], b = v[j + h]
                    v[j] = a + b
                    v[j + h] = a - b
                }
                i += h * 2
            }
            h <<= 1
        }
        let sc: Float = 0.08838834764831845
        for i in 0..<128 { v[i] *= sc }
    }

    private static func detRand(_ count: Int, seed: UInt64, scale: Float = 0.5) -> [Float] {
        var s = seed
        return (0..<count).map { _ in
            s = s &* 6364136223846793005 &+ 1442695040888963407
            return (Float(Int32(truncatingIfNeeded: s >> 33)) / Float(Int32.max)) * scale
        }
    }

    private static func makeF16(_ hal: QWHal, _ shape: [Int], _ values: [Float]) -> QWTensorValue {
        let bits = values.map { Float16($0).bitPattern }
        let buf = hal.device.makeBuffer(length: max(16, bits.count * 2), options: .storageModeShared)!
        bits.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let base = raw.baseAddress, !raw.isEmpty { buf.contents().copyMemory(from: base, byteCount: raw.count) }
        }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length), offset: 0, shape: QWTensorShape(shape, .f16))
    }

    private static func makeBF16(_ hal: QWHal, _ shape: [Int], _ values: [Float]) -> QWTensorValue {
        let bits = values.map { bf16Bits($0) }
        let buf = hal.device.makeBuffer(length: max(16, bits.count * 2), options: .storageModeShared)!
        bits.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let base = raw.baseAddress, !raw.isEmpty { buf.contents().copyMemory(from: base, byteCount: raw.count) }
        }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length), offset: 0, shape: QWTensorShape(shape, .bf16))
    }

    private static func packHQ8G32(_ hal: QWHal, flat: [Float], n: Int, k: Int, rotate: Bool) -> QWWeightView {
        var w = flat
        if rotate {
            for r in 0..<n {
                var row = Array(w[(r * k)..<((r + 1) * k)])
                fwht32InPlace(&row)
                w.replaceSubrange((r * k)..<((r + 1) * k), with: row)
            }
        }
        let groups = k / 32
        let gBytes = QWAQBlock.bytes(bits: 8, group: 32)
        var bank = Data(count: n * groups * gBytes)
        bank.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let gOff = (r * groups + g) * gBytes
                    let start = r * k + g * 32
                    var mn = Float.infinity, mx = -Float.infinity
                    for i in 0..<32 {
                        mn = min(mn, w[start + i])
                        mx = max(mx, w[start + i])
                    }
                    let scH = Float16(max((mx - mn) / 255.0, 1e-8))
                    let biH = Float16(mn)
                    ptr[gOff + 0] = UInt8(scH.bitPattern & 0xFF)
                    ptr[gOff + 1] = UInt8(scH.bitPattern >> 8)
                    ptr[gOff + 2] = UInt8(biH.bitPattern & 0xFF)
                    ptr[gOff + 3] = UInt8(biH.bitPattern >> 8)
                    for i in 0..<32 {
                        let q = min(255.0, max(0.0, ((w[start + i] - Float(biH)) / Float(scH)).rounded()))
                        ptr[gOff + 4 + i] = UInt8(q)
                    }
                }
            }
        }
        let buf = hal.device.makeBuffer(length: max(16, bank.count), options: .storageModeShared)!
        bank.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let base = raw.baseAddress { buf.contents().copyMemory(from: base, byteCount: bank.count) }
        }
        let tv = QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length), offset: 0, shape: QWTensorShape([bank.count / 2], .f16))
        return QWWeightView(blocks: tv, n: n, k: k, bits: 8, groupSize: 32, isRotated: rotate, isIncumbentContainer: false, isSHVQ: false)!
    }

    private static func cpuGatedNorm(_ x: [Float], w: [Float], gu: [Float], gd: [Float], rank: Int, gw: Float, eps: Float) -> [Float] {
        let d = x.count
        var sumSq: Float = 0
        for v in x { sumSq += v * v }
        let inv = 1.0 / sqrt(sumSq / Float(d) + eps)
        var h = [Float](repeating: 0, count: d)
        for i in 0..<d { h[i] = bf16Round(x[i] * inv * w[i]) }
        var g1 = [Float](repeating: 0, count: rank)
        for r in 0..<rank {
            var dot: Float = 0
            for i in 0..<d { dot += gu[r * d + i] * h[i] }
            g1[r] = bf16Round(dot / (1.0 + exp(-dot)))
        }
        var y = [Float](repeating: 0, count: d)
        for i in 0..<d {
            var dot: Float = 0
            for r in 0..<rank { dot += gd[i * rank + r] * g1[r] }
            let g2 = bf16Round(1.0 / (1.0 + exp(-dot)))
            y[i] = bf16Round(h[i] * gw * g2)
        }
        return y
    }

    func testLatentPrepGatedNormAlphaQAlphaKVAndRMSIndexerKNorm() throws {
        try skipUnlessMetal()
        let geo = try Self.makeGC4LikeGeometry(qLora: 64, kvLora: 512, ropeDim: 64, idxHeads: 8, idxDim: 128)
        let hal = try QWHal(geometry: geo)
        let rot = hal.weightBasis.rotatesActivations
        let qL = 64, kvL = 512, rD = 64, iD = 128, rank = 16
        let alphaQ: Float = 0.25, alphaKV: Float = 0.25
        let rawStride = qL + kvL + rD + iD

        let qaW = Self.detRand(qL, seed: 10, scale: 1.0).map { Self.bf16Round(0.8 + $0) }
        let qaGU = Self.detRand(rank * qL, seed: 11, scale: 0.35).map(Self.bf16Round)
        let qaGD = Self.detRand(qL * rank, seed: 12, scale: 0.35).map(Self.bf16Round)
        let kvaW = Self.detRand(kvL, seed: 13, scale: 1.0).map { Self.bf16Round(0.8 + $0) }
        let kvaGU = Self.detRand(rank * kvL, seed: 14, scale: 0.2).map(Self.bf16Round)
        let kvaGD = Self.detRand(kvL * rank, seed: 15, scale: 0.2).map(Self.bf16Round)
        let knW = Self.detRand(iD, seed: 16, scale: 0.5).map { Float(Float16(0.9 + $0)) }
        let knB = Self.detRand(iD, seed: 17, scale: 2.0)

        let qbRows = 16 * (192 + 64) + 8 * 128
        let kvbRows = 16 * (192 + 256)
        let mlaW = QWMLAAttnOperator.LayerWeights(
            qANormW: Self.makeBF16(hal, [qL], qaW),
            kvANormW: Self.makeBF16(hal, [kvL], kvaW),
            kNormW: Self.makeF16(hal, [iD], knW),
            kNormB: Self.makeF16(hal, [iD], knB),
            wGateWeights: Self.makeF16(hal, [8, 128], Self.detRand(8 * 128, seed: 18)),
            ape: nil,
            qBProj: Self.packHQ8G32(hal, flat: Self.detRand(qbRows * qL, seed: 19, scale: 0.1), n: qbRows, k: qL, rotate: rot),
            kvBProj: Self.packHQ8G32(hal, flat: Self.detRand(kvbRows * kvL, seed: 20, scale: 0.1), n: kvbRows, k: kvL, rotate: rot),
            hasOGate: false,
            qAGatedNorm: .init(weight: Self.makeBF16(hal, [qL], qaW), gateUp: Self.makeBF16(hal, [rank, qL], qaGU), gateDown: Self.makeBF16(hal, [qL, rank], qaGD), rank: rank, gatingWeight: 2.0),
            kvAGatedNorm: .init(weight: Self.makeBF16(hal, [kvL], kvaW), gateUp: Self.makeBF16(hal, [rank, kvL], kvaGU), gateDown: Self.makeBF16(hal, [kvL, rank], kvaGD), rank: rank, gatingWeight: 2.0),
            alphaQ: alphaQ,
            alphaKV: alphaKV,
            indexerKNormIsRMS: true
        )

        let latVals = Self.detRand(rawStride, seed: 21, scale: 1.4).map { Float(Float16($0)) }
        let ln1Tensor = Self.makeF16(hal, [1, 128], Self.detRand(128, seed: 22))
        let (invFreq, _) = QWModelDescriptor.computeRopeInvFreq(rotaryDim: rD, base: 80_000.0, yarn: nil)

        for kvMode in [QWKVMode.f16, QWKVMode.q8, QWKVMode.q4] {
            let pBytes = QWKVCache.pageBytes(mode: kvMode, geometry: geo)
            let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: pBytes * 2 + QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs, gdnStates: 4096)
            let cache = try QWKVCache(hal: hal, arena: arena, layers: [1], mode: kvMode, geometry: geo)
            let seq = try cache.allocate(promptTokens: 8)
            let kv = try cache.layerViews(seq, layer: 1)
            let op = try QWMLAAttnOperator(hal: hal, geometry: geo, numAttnLayers: 1, maxContext: 128, maxPrefillChunk: 8)

            let lp = op.region.pointer.advanced(by: op.mlaLatentScratch.offset).assumingMemoryBound(to: UInt16.self)
            for i in 0..<rawStride { lp[i] = Float16(latVals[i]).bitPattern }

            guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
            op.encodeLatentPrepAndPool(enc, ln1: ln1Tensor, mlaW: mlaW, kv: kv, fullAttnLayer: 0, pos0: 3, rows: 1)
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)

            var refQ = Self.cpuGatedNorm(Array(latVals[0..<qL]), w: qaW, gu: qaGU, gd: qaGD, rank: rank, gw: 2.0, eps: 1e-6).map { $0 * alphaQ }
            if rot { Self.fwht32InPlace(&refQ) }
            let gotQPtr = op.region.pointer.advanced(by: op.qResidScratch.offset).assumingMemoryBound(to: UInt16.self)
            for i in 0..<qL {
                XCTAssertEqual(Float(Float16(bitPattern: gotQPtr[i])), refQ[i], accuracy: 2e-3, "qResid[\(i)] kv=\(kvMode)")
            }

            var refKV = Self.cpuGatedNorm(Array(latVals[qL..<(qL + kvL)]), w: kvaW, gu: kvaGU, gd: kvaGD, rank: rank, gw: 2.0, eps: 1e-6).map { $0 * alphaKV }
            if rot { Self.fwht32InPlace(&refKV) }
            if kvMode == .f16 {
                let kPool = kv.k.pool.buffer.contents().advanced(by: kv.k.pool.base).assumingMemoryBound(to: Float16.self)
                let vPool = kv.v.pool.buffer.contents().advanced(by: kv.v.pool.base).assumingMemoryBound(to: Float16.self)
                for c in 0..<256 {
                    XCTAssertEqual(Float(kPool[3 * 256 + c]), refKV[c], accuracy: 2e-3)
                    XCTAssertEqual(Float(vPool[3 * 256 + c]), refKV[256 + c], accuracy: 2e-3)
                }
            }

            let wk = Array(latVals[(qL + kvL + rD)..<rawStride])
            var wkSq: Float = 0
            for v in wk { wkSq += v * v }
            let wkInv = 1.0 / sqrt(wkSq / Float(iD) + 1e-6)
            var refKIdx = (0..<iD).map { wk[$0] * wkInv * knW[$0] }
            let halfR = rD / 2
            for j in 0..<halfR {
                let ang = 3.0 * invFreq[j]
                let cs = cos(ang), sn = sin(ang)
                let x0 = refKIdx[j], x1 = refKIdx[j + halfR]
                refKIdx[j] = x0 * cs - x1 * sn
                refKIdx[j + halfR] = x1 * cs + x0 * sn
            }
            Self.fwht128InPlace(&refKIdx)
            let pPool = op.region.pointer.advanced(by: op.layers[0].pooledKeys.offset).assumingMemoryBound(to: UInt16.self)
            for d in 0..<iD {
                XCTAssertEqual(Float(Float16(bitPattern: pPool[3 * iD + d])), refKIdx[d], accuracy: 3e-3, "kIdx[\(d)]")
            }
        }
    }

    func testSoftmaxScaleAcrossAllDSAKernels() throws {
        try skipUnlessMetal()
        let geo = try Self.makeGC4LikeGeometry(qHeads: 64, qLora: 64, kvLora: 512, nopeDim: 192, ropeDim: 64, vDim: 256, idxHeads: 8, idxDim: 128, idxTopK: 16, window: 8)
        let hal = try QWHal(geometry: geo)
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable, "NAX DSA kernels require Metal 4")

        let qHeads = 64, kvRank = 512, ropeDim = 64, nopeDim = 192
        let qFullStride = qHeads * (nopeDim + ropeDim)
        let qRopeOffset = qHeads * nopeDim
        let rows = 8, pos0 = 32, topK = 16, window = 8, listStride = 64
        let scaleOverride: Float = 0.13086058

        let qAbs = Self.makeF16(hal, [rows * qHeads * kvRank], Self.detRand(rows * qHeads * kvRank, seed: 101, scale: 0.9))
        let qFull = Self.makeF16(hal, [rows * qFullStride], Self.detRand(rows * qFullStride, seed: 102, scale: 0.9))
        let kPe = Self.makeF16(hal, [128 * ropeDim], Self.detRand(128 * ropeDim, seed: 103, scale: 0.9))
        let sink = Self.makeF16(hal, [qHeads], [Float](repeating: 0, count: qHeads))
        var blkArr = [Int32](repeating: 0, count: rows * listStride)
        for t in 0..<rows {
            let seqLen = pos0 + t + 1
            for s in 0..<(topK + window) { blkArr[t * listStride + s] = Int32((t * 11 + s * 5) % seqLen) }
        }
        let blkBuf = hal.device.makeBuffer(length: blkArr.count * 4, options: .storageModeShared)!
        blkArr.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in blkBuf.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count) }
        let blkList = QWTensorValue(region: QWMRegion(buffer: blkBuf, base: 0, length: blkBuf.length), offset: 0, shape: QWTensorShape([blkArr.count], .i32))

        let pBytes = QWKVCache.pageBytes(mode: .f16, geometry: geo)
        let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: pBytes * 2 + QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs, gdnStates: 4096)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: [1], mode: .f16, geometry: geo)
        let seq = try cache.allocate(promptTokens: pos0 + rows)
        let kv = try cache.layerViews(seq, layer: 1)
        for (idx, pool) in [kv.k.pool, kv.v.pool].enumerated() {
            let p = pool.buffer.contents().advanced(by: pool.base).assumingMemoryBound(to: Float16.self)
            let vals = Self.detRand(128 * 256, seed: UInt64(200 + idx), scale: 0.85)
            for i in 0..<vals.count { p[i] = Float16(vals[i]) }
        }

        let outScalar = Self.makeF16(hal, [rows * qHeads * kvRank], [Float](repeating: 0, count: rows * qHeads * kvRank))
        let outDefault = Self.makeF16(hal, [rows * qHeads * kvRank], [Float](repeating: 0, count: rows * qHeads * kvRank))
        let outCand = Self.makeF16(hal, [rows * qHeads * kvRank], [Float](repeating: 0, count: rows * qHeads * kvRank))
        let partO = Self.makeF16(hal, [rows * 4 * qHeads * kvRank], [Float](repeating: 0, count: rows * 4 * qHeads * kvRank))
        let partMLBuf = hal.device.makeBuffer(length: rows * 4 * qHeads * 8, options: .storageModeShared)!
        let partML = QWTensorValue(region: QWMRegion(buffer: partMLBuf, base: 0, length: partMLBuf.length), offset: 0, shape: QWTensorShape([rows * 4 * qHeads * 2], .f32))

        func makeParams(_ smScale: Float) -> QWMLAAttnOperator.DSASparseAttnParams {
            QWMLAAttnOperator.DSASparseAttnParams(
                T: UInt32(rows), pos0: UInt32(pos0), qHeads: UInt32(qHeads),
                headDim: UInt32(nopeDim + ropeDim), kvLoraRank: UInt32(kvRank),
                topK: UInt32(topK), windowTokens: UInt32(window), listStride: UInt32(listStride),
                kvMode: 0, ropeDim: UInt32(ropeDim), qRopeOffset: UInt32(qRopeOffset),
                qFullStride: UInt32(qFullStride), hasSink: 0, softmaxScale: smScale
            )
        }

        var pCustom = makeParams(scaleOverride)
        var pDef = makeParams(0.0)
        let psoScalar = try hal.pipeline("qwg_mla_dsa_sparse_attn")
        let psoMMA = try hal.pipeline("qwg_mla_dsa_sparse_attn_mma")
        let psoNAX = try hal.pipeline("qwg_mla_dsa_sparse_attn_nax")
        let psoH32 = try hal.pipeline("qwg_mla_dsa_sparse_attn_nax_h32")
        let psoH64 = try hal.pipeline("qwg_mla_dsa_sparse_attn_nax_h64")
        let psoSplit = try hal.pipeline("qwg_mla_dsa_sparse_attn_nax_split")
        let psoCombine = try hal.pipeline("qwg_mla_dsa_sparse_attn_combine")

        guard let (cb0, e0) = hal.begin() else { XCTFail("hal.begin"); return }
        e0.setComputePipelineState(psoScalar)
        qAbs.bind(e0, index: 0); kv.k.pool.bindTo(e0, index: 1); kv.v.pool.bindTo(e0, index: 2)
        kv.k.table.bind(e0, index: 3); kv.v.table.bind(e0, index: 4); blkList.bind(e0, index: 5)
        outScalar.bind(e0, index: 6)
        e0.setBytes(&pCustom, length: MemoryLayout<QWMLAAttnOperator.DSASparseAttnParams>.stride, index: 7)
        qFull.bind(e0, index: 8); kPe.bind(e0, index: 9); sink.bind(e0, index: 10)
        e0.dispatchThreadgroups(MTLSize(width: rows * qHeads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        outDefault.bind(e0, index: 6)
        e0.setBytes(&pDef, length: MemoryLayout<QWMLAAttnOperator.DSASparseAttnParams>.stride, index: 7)
        e0.dispatchThreadgroups(MTLSize(width: rows * qHeads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        hal.end(cb0, e0)
        cb0.waitUntilCompleted()
        XCTAssertNil(cb0.error)

        let totalElems = rows * qHeads * kvRank
        func readOut(_ tv: QWTensorValue) -> [Float] {
            let p = tv.region.pointer.assumingMemoryBound(to: UInt16.self)
            return (0..<totalElems).map { Float(Float16(bitPattern: p[$0])) }
        }
        let refVals = readOut(outScalar)
        let defVals = readOut(outDefault)
        let vsDefDiff = zip(refVals, defVals).map { abs($0 - $1) }.max() ?? 0
        XCTAssertGreaterThan(vsDefDiff, 1e-2, "softmaxScaleOverride must visibly alter attention distribution vs default rsqrt(256)=0.0625")

        for (label, pso, gY, tptg) in [("mma", psoMMA, qHeads / 16, 128), ("nax", psoNAX, qHeads / 16, 128), ("h32", psoH32, qHeads / 32, 256), ("h64", psoH64, qHeads / 64, 512)] {
            guard let (cb, e) = hal.begin() else { XCTFail("hal.begin"); return }
            e.setComputePipelineState(pso)
            qAbs.bind(e, index: 0); kv.k.pool.bindTo(e, index: 1); kv.v.pool.bindTo(e, index: 2)
            kv.k.table.bind(e, index: 3); kv.v.table.bind(e, index: 4); blkList.bind(e, index: 5)
            outCand.bind(e, index: 6)
            e.setBytes(&pCustom, length: MemoryLayout<QWMLAAttnOperator.DSASparseAttnParams>.stride, index: 7)
            qFull.bind(e, index: 8); kPe.bind(e, index: 9); sink.bind(e, index: 10)
            e.dispatchThreadgroups(MTLSize(width: rows, height: gY, depth: 1), threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1))
            hal.end(cb, e)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
            let diff = zip(refVals, readOut(outCand)).map { abs($0 - $1) }.max() ?? 0
            XCTAssertLessThan(diff, 4e-3, "\(label) softmaxScale diff \(diff)")
        }

        var nSp: UInt32 = 4
        guard let (cbS, eS) = hal.begin() else { XCTFail("hal.begin"); return }
        eS.setComputePipelineState(psoSplit)
        qAbs.bind(eS, index: 0); kv.k.pool.bindTo(eS, index: 1); kv.v.pool.bindTo(eS, index: 2)
        kv.k.table.bind(eS, index: 3); kv.v.table.bind(eS, index: 4); blkList.bind(eS, index: 5)
        partO.bind(eS, index: 6)
        eS.setBytes(&pCustom, length: MemoryLayout<QWMLAAttnOperator.DSASparseAttnParams>.stride, index: 7)
        partML.bind(eS, index: 8); qFull.bind(eS, index: 9); kPe.bind(eS, index: 10); sink.bind(eS, index: 11)
        eS.dispatchThreadgroups(MTLSize(width: rows, height: qHeads / 16, depth: 4), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        eS.memoryBarrier(scope: .buffers)
        eS.setComputePipelineState(psoCombine)
        partO.bind(eS, index: 0); partML.bind(eS, index: 1); outCand.bind(eS, index: 2)
        eS.setBytes(&pCustom, length: MemoryLayout<QWMLAAttnOperator.DSASparseAttnParams>.stride, index: 3)
        eS.setBytes(&nSp, length: 4, index: 4)
        sink.bind(eS, index: 5)
        eS.dispatchThreadgroups(MTLSize(width: rows * qHeads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        hal.end(cbS, eS)
        cbS.waitUntilCompleted()
        XCTAssertNil(cbS.error)
        let splitDiff = zip(refVals, readOut(outCand)).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(splitDiff, 4e-3, "nax_split+combine softmaxScale diff \(splitDiff)")
    }

    func testFusedOGateInQKVStandaloneOGateDenseOnlyAndBatchedDecodeB8() throws {
        try skipUnlessMetal()
        let H = 128, qHeads = 16, qL = 64, kvL = 512, nopeD = 192, ropeD = 64, vD = 256, idxH = 8, idxD = 128
        let geo = try Self.makeGC4LikeGeometry(hidden: H, qHeads: qHeads, qLora: qL, kvLora: kvL, nopeDim: nopeD, ropeDim: ropeD, vDim: vD, idxHeads: idxH, idxDim: idxD, idxTopK: 4, window: 4)
        let hal = try QWHal(geometry: geo)
        let rot = hal.weightBasis.rotatesActivations
        let outCols = qHeads * vD

        let qbRows = qHeads * (nopeD + ropeD) + idxH * idxD
        let kvbRows = qHeads * (nopeD + vD)
        let qBProj = Self.packHQ8G32(hal, flat: Self.detRand(qbRows * qL, seed: 301, scale: 0.12), n: qbRows, k: qL, rotate: rot)
        let kvBProj = Self.packHQ8G32(hal, flat: Self.detRand(kvbRows * kvL, seed: 302, scale: 0.12), n: kvbRows, k: kvL, rotate: rot)
        let gateProj = Self.packHQ8G32(hal, flat: Self.detRand(outCols * H, seed: 303, scale: 0.25), n: outCols, k: H, rotate: rot)

        let qaW = Self.makeF16(hal, [qL], Self.detRand(qL, seed: 304, scale: 0.4).map { 0.8 + $0 })
        let kvaW = Self.makeF16(hal, [kvL], Self.detRand(kvL, seed: 305, scale: 0.4).map { 0.8 + $0 })
        let knW = Self.makeF16(hal, [idxD], Self.detRand(idxD, seed: 306, scale: 0.4).map { 0.8 + $0 })
        let knB = Self.makeF16(hal, [idxD], [Float](repeating: 0, count: idxD))
        let wgW = Self.makeF16(hal, [idxH, H], Self.detRand(idxH * H, seed: 307, scale: 0.2))

        let wStandalone = QWMLAAttnOperator.LayerWeights(
            qANormW: qaW, kvANormW: kvaW, kNormW: knW, kNormB: knB, wGateWeights: wgW, ape: nil,
            qBProj: qBProj, kvBProj: kvBProj, hasOGate: true, oGateProj: gateProj,
            alphaQ: 0.25, alphaKV: 0.25, indexerKNormIsRMS: true, softmaxScaleOverride: 0.13086058,
            oGateInQKV: false
        )
        let wFused = QWMLAAttnOperator.LayerWeights(
            qANormW: qaW, kvANormW: kvaW, kNormW: knW, kNormB: knB, wGateWeights: wgW, ape: nil,
            qBProj: qBProj, kvBProj: kvBProj, hasOGate: true, oGateProj: nil,
            alphaQ: 0.25, alphaKV: 0.25, indexerKNormIsRMS: true, softmaxScaleOverride: 0.13086058,
            oGateInQKV: true
        )

        let T = 4
        let ln1 = Self.makeF16(hal, [T, H], Self.detRand(T * H, seed: 308, scale: 0.6))
        let gateProjected = Self.makeF16(hal, [T, outCols], [Float](repeating: 0, count: T * outCols))
        guard let (cbG, encG) = hal.begin() else { XCTFail("hal.begin"); return }
        try QWGemmAQ.encodeDecodeT4(encG, hal: hal, blocks: gateProj.blocks, x: ln1, y16: gateProjected, n: outCols, k: H, bits: 8, group: 32, rows: T, isSHVQ: false)
        hal.end(cbG, encG)
        cbG.waitUntilCompleted()

        let baseLatStride = qL + kvL + ropeD + idxD
        let fusedLatStride = baseLatStride + outCols
        let baseLatVals = Self.detRand(T * baseLatStride, seed: 309, scale: 0.7)
        let gatePtr = gateProjected.region.pointer.assumingMemoryBound(to: UInt16.self)

        let pBytes = QWKVCache.pageBytes(mode: .f16, geometry: geo)
        let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: pBytes * 16 + QWKVCache.tableSlotBytes * 4 * QWKVCache.maxSeqs, gdnStates: 4096)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: [0, 1], mode: .f16, geometry: geo)
        let seqA = try cache.allocate(promptTokens: 16)
        let seqB = try cache.allocate(promptTokens: 16)

        let opStand = try QWMLAAttnOperator(hal: hal, geometry: geo, numAttnLayers: 1, maxContext: 128, maxPrefillChunk: 8, oGateInQKV: false)
        let opFused = try QWMLAAttnOperator(hal: hal, geometry: geo, numAttnLayers: 1, maxContext: 128, maxPrefillChunk: 8, oGateInQKV: true)

        let pS = opStand.region.pointer.advanced(by: opStand.mlaLatentScratch.offset).assumingMemoryBound(to: UInt16.self)
        let pF = opFused.region.pointer.advanced(by: opFused.mlaLatentScratch.offset).assumingMemoryBound(to: UInt16.self)
        for t in 0..<T {
            for c in 0..<baseLatStride {
                let bits = Float16(baseLatVals[t * baseLatStride + c]).bitPattern
                pS[t * baseLatStride + c] = bits
                pF[t * fusedLatStride + c] = bits
            }
            for c in 0..<outCols {
                pF[t * fusedLatStride + baseLatStride + c] = gatePtr[t * outCols + c]
            }
        }

        func runFullEncode(_ op: QWMLAAttnOperator, _ enc: any MTLComputeCommandEncoder,
                           ln1: QWTensorValue, mlaW: QWMLAAttnOperator.LayerWeights,
                           kv: (k: QWKVLayerView, v: QWKVLayerView), attOut: QWTensorValue,
                           layer: Int, pos0: Int, rows: Int) throws {
            op.encodeLatentPrepAndPool(enc, ln1: ln1, mlaW: mlaW, kv: kv, fullAttnLayer: layer, pos0: pos0, rows: rows)
            let qStride = op.qFullStride(for: mlaW, state: op.layers[layer])
            try op.encodeMLAWeightViewH16(
                enc, x: op.qResidScratch.slice(byteOffset: 0, count: rows * qL, dtype: .f16),
                weight: mlaW.qBProj,
                y16: op.mlaQFullScratch.slice(byteOffset: 0, count: rows * qStride, dtype: .f16),
                rows: rows
            )
            op.encodeScoreSelectAbsorbAttendAndProjUV(
                enc, mlaW: mlaW, kv: kv, attOut: attOut, fullAttnLayer: layer, pos0: pos0, rows: rows, ln1: ln1
            )
        }

        let outStand = Self.makeF16(hal, [T, outCols], [Float](repeating: 0, count: T * outCols))
        let outFused = Self.makeF16(hal, [T, outCols], [Float](repeating: 0, count: T * outCols))
        guard let (cb1, enc1) = hal.begin() else { XCTFail("hal.begin"); return }
        try runFullEncode(opStand, enc1, ln1: ln1, mlaW: wStandalone, kv: try cache.layerViews(seqA, layer: 0), attOut: outStand, layer: 0, pos0: 0, rows: T)
        try runFullEncode(opFused, enc1, ln1: ln1, mlaW: wFused, kv: try cache.layerViews(seqB, layer: 0), attOut: outFused, layer: 0, pos0: 0, rows: T)
        hal.end(cb1, enc1)
        cb1.waitUntilCompleted()
        XCTAssertNil(cb1.error)

        let uStand = Array(UnsafeBufferPointer(start: outStand.region.pointer.assumingMemoryBound(to: UInt16.self), count: T * outCols))
        let uFused = Array(UnsafeBufferPointer(start: outFused.region.pointer.assumingMemoryBound(to: UInt16.self), count: T * outCols))
        XCTAssertEqual(uStand, uFused, "Fused oGateInQKV must be 0-ULP bit-identical to standalone oGateProj")

        // Dense-only Layer-32 MTP + liveDSASessionStashBytes check.
        let opMTP = try QWMLAAttnOperator(
            hal: hal, geometry: geo, numAttnLayers: 2, maxContext: 128, maxPrefillChunk: 8,
            oGateInQKV: true, mtpDenseOnly: true
        )
        XCTAssertTrue(opMTP.layers[0].hasIndexer)
        XCTAssertFalse(opMTP.layers[1].hasIndexer)
        let expectedStash = 8 * idxD * 2 + 8 * ropeD * 2 + 5 * ropeD * 2
        XCTAssertEqual(opMTP.liveSessionStashBytes(trunkPosition: 8, mtpLayerIndex: 1, mtpPosition: 5), expectedStash)

        let qbDenseRows = qHeads * (nopeD + ropeD)
        let wDenseMTP = QWMLAAttnOperator.LayerWeights(
            qANormW: qaW, kvANormW: kvaW, kNormW: knW, kNormB: knB, wGateWeights: wgW, ape: nil,
            qBProj: Self.packHQ8G32(hal, flat: Self.detRand(qbDenseRows * qL, seed: 310, scale: 0.12), n: qbDenseRows, k: qL, rotate: rot),
            kvBProj: kvBProj, hasOGate: true, alphaQ: 0.25, alphaKV: 0.25,
            softmaxScaleOverride: 0.13086058, denseOnly: true, oGateInQKV: true
        )
        let denseLatStride = qL + kvL + ropeD + outCols
        let pMTP = opMTP.region.pointer.advanced(by: opMTP.mlaLatentScratch.offset).assumingMemoryBound(to: UInt16.self)
        for i in 0..<(2 * denseLatStride) { pMTP[i] = Float16(sin(Float(i) * 0.07) * 0.4).bitPattern }
        let outDense = Self.makeF16(hal, [2, outCols], [Float](repeating: 0, count: 2 * outCols))
        let ln1Two = Self.makeF16(hal, [2, H], Self.detRand(2 * H, seed: 311, scale: 0.5))
        guard let (cbM, encM) = hal.begin() else { XCTFail("hal.begin"); return }
        try runFullEncode(opMTP, encM, ln1: ln1Two, mlaW: wDenseMTP, kv: try cache.layerViews(seqA, layer: 1), attOut: outDense, layer: 1, pos0: 0, rows: 2)
        hal.end(cbM, encM)
        cbM.waitUntilCompleted()
        XCTAssertNil(cbM.error)
        let denseVals = (0..<(2 * outCols)).map { Float(Float16(bitPattern: outDense.region.pointer.assumingMemoryBound(to: UInt16.self)[$0])) }
        XCTAssertTrue(denseVals.allSatisfy(\.isFinite) && denseVals.contains { abs($0) > 1e-4 })
        let stashBlob = opMTP.captureLiveSessionStash(trunkPosition: 2, mtpLayerIndex: 1, mtpPosition: 2)
        XCTAssertEqual(stashBlob.count, opMTP.liveSessionStashBytes(trunkPosition: 2, mtpLayerIndex: 1, mtpPosition: 2))
        XCTAssertTrue(opMTP.restoreLiveSessionStash(stashBlob, trunkPosition: 2, mtpLayerIndex: 1, mtpPosition: 2))

        // Batched decode B=2, 4, 8 sessions at distinct positions vs serial B=1.
        for B in [2, 4, 8] {
            let opBatch = try QWMLAAttnOperator(hal: hal, geometry: geo, numAttnLayers: 1, maxContext: 128, maxPrefillChunk: 8, oGateInQKV: true)
            let opSer = try QWMLAAttnOperator(hal: hal, geometry: geo, numAttnLayers: 1, maxContext: 128, maxPrefillChunk: 8, oGateInQKV: true)
            let arenaB = try QWArena(hal: hal, weights: 0, activations: 0, kv: pBytes * B * 4 + QWKVCache.tableSlotBytes * 2 * max(QWKVCache.maxSeqs, B * 2), gdnStates: 4096)
            let cacheBat = try QWKVCache(hal: hal, arena: arenaB, layers: [0], mode: .f16, geometry: geo)
            let arenaS = try QWArena(hal: hal, weights: 0, activations: 0, kv: pBytes * B * 4 + QWKVCache.tableSlotBytes * 2 * max(QWKVCache.maxSeqs, B * 2), gdnStates: 4096)
            let cacheSer = try QWKVCache(hal: hal, arena: arenaS, layers: [0], mode: .f16, geometry: geo)

            var batSessions: [QWMLAAttnOperator.SessionState] = []
            var serSessions: [QWMLAAttnOperator.SessionState] = []
            var batSeqs: [QWKVHandle] = []
            var serSeqs: [QWKVHandle] = []
            var positions: [Int] = []

            for b in 0..<B {
                let promptLen = 3 + (b % 4)
                positions.append(promptLen)
                let sb = try opBatch.makeSessionState(maxContext: 128)
                let ss = try opSer.makeSessionState(maxContext: 128)
                let qvb = try cacheBat.allocate(promptTokens: promptLen + 2)
                let qvs = try cacheSer.allocate(promptTokens: promptLen + 2)
                let preLn1 = Self.makeF16(hal, [promptLen, H], Self.detRand(promptLen * H, seed: UInt64(400 + b), scale: 0.5))
                let preLat = Self.detRand(promptLen * fusedLatStride, seed: UInt64(500 + b), scale: 0.5)
                let preOut = Self.makeF16(hal, [promptLen, outCols], [Float](repeating: 0, count: promptLen * outCols))

                for (opX, sessX, kvX) in [(opBatch, sb, try cacheBat.layerViews(qvb, layer: 0)), (opSer, ss, try cacheSer.layerViews(qvs, layer: 0))] {
                    let stepSlots = (0..<promptLen).map { t in QWMLAAttnOperator.DecodeBatchSlot(kv: kvX, position: t, state: sessX.layers[0]) }
                    for t in 0..<promptLen {
                        let p1 = opX.region.pointer.advanced(by: opX.mlaLatentScratch.offset).assumingMemoryBound(to: UInt16.self)
                        for c in 0..<fusedLatStride { p1[c] = Float16(preLat[t * fusedLatStride + c]).bitPattern }
                        guard let (cbT, encT) = hal.begin() else { XCTFail("hal.begin"); return }
                        try opX.encodeDecodeBatch(encT, ln1: preLn1.slice(byteOffset: t * H * 2, count: H, dtype: .f16), mlaW: wFused, slots: [stepSlots[t]], attOut: preOut.slice(byteOffset: t * outCols * 2, count: outCols, dtype: .f16))
                        hal.end(cbT, encT)
                        cbT.waitUntilCompleted()
                    }
                }
                batSessions.append(sb)
                serSessions.append(ss)
                batSeqs.append(qvb)
                serSeqs.append(qvs)
            }

            let stepLn1 = Self.makeF16(hal, [B, H], Self.detRand(B * H, seed: UInt64(600 + B), scale: 0.6))
            let stepLat = Self.detRand(B * fusedLatStride, seed: UInt64(700 + B), scale: 0.6)
            let pBat = opBatch.region.pointer.advanced(by: opBatch.mlaLatentScratch.offset).assumingMemoryBound(to: UInt16.self)
            for i in 0..<stepLat.count { pBat[i] = Float16(stepLat[i]).bitPattern }

            let outBatch = Self.makeF16(hal, [B, outCols], [Float](repeating: 0, count: B * outCols))
            let outSerial = Self.makeF16(hal, [B, outCols], [Float](repeating: 0, count: B * outCols))
            let slotsB = try (0..<B).map { b in
                QWMLAAttnOperator.DecodeBatchSlot(kv: try cacheBat.layerViews(batSeqs[b], layer: 0), position: positions[b], state: batSessions[b].layers[0])
            }

            guard let (cbB, encB) = hal.begin() else { XCTFail("hal.begin"); return }
            try opBatch.encodeDecodeBatch(encB, ln1: stepLn1, mlaW: wFused, slots: slotsB, attOut: outBatch)
            hal.end(cbB, encB)
            cbB.waitUntilCompleted()
            XCTAssertNil(cbB.error)

            for b in 0..<B {
                let pS1 = opSer.region.pointer.advanced(by: opSer.mlaLatentScratch.offset).assumingMemoryBound(to: UInt16.self)
                for c in 0..<fusedLatStride { pS1[c] = Float16(stepLat[b * fusedLatStride + c]).bitPattern }
                let slot1 = QWMLAAttnOperator.DecodeBatchSlot(kv: try cacheSer.layerViews(serSeqs[b], layer: 0), position: positions[b], state: serSessions[b].layers[0])
                guard let (cbS1, encS1) = hal.begin() else { XCTFail("hal.begin"); return }
                try opSer.encodeDecodeBatch(encS1, ln1: stepLn1.slice(byteOffset: b * H * 2, count: H, dtype: .f16), mlaW: wFused, slots: [slot1], attOut: outSerial.slice(byteOffset: b * outCols * 2, count: outCols, dtype: .f16))
                hal.end(cbS1, encS1)
                cbS1.waitUntilCompleted()
                XCTAssertNil(cbS1.error)
            }

            let gotB = (0..<(B * outCols)).map { Float(Float16(bitPattern: outBatch.region.pointer.assumingMemoryBound(to: UInt16.self)[$0])) }
            let gotS = (0..<(B * outCols)).map { Float(Float16(bitPattern: outSerial.region.pointer.assumingMemoryBound(to: UInt16.self)[$0])) }
            let maxBatchDiff = zip(gotB, gotS).map { abs($0 - $1) }.max() ?? 0
            XCTAssertLessThan(maxBatchDiff, 2e-3, "encodeDecodeBatch B=\(B) vs serial B=1 diff \(maxBatchDiff)")
        }
    }
}
