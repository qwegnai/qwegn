// QWErnie55AttnTests.swift — Golden parity, quantized-KV, e-means coarse pruning,
// and `.qwcache` snapshot tests for ERNIE-5.5 Decoupled-RoPE MLA-DSA and HCA.

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

final class QWErnie55AttnTests: XCTestCase {
    static let fixtureURL: URL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Scripts/spike/out/ernie55_synthetic_goldens.json")

    private struct ErnieGoldenFile: Decodable {
        struct YarnSection: Decodable {
            let inv_freq: [Float]
            let x: [[Float]]
            let positions: [Int]
            let fwd: [[Float]]
            let roundtrip: [[Float]]
        }
        struct MLADSASection: Decodable {
            let T: Int
            let hidden: Int
            let q_heads: Int
            let idx_heads: Int
            let idx_dim: Int
            let q_lora: Int
            let kv_lora: Int
            let nope_dim: Int
            let rope_dim: Int
            let v_dim: Int
            let topk: Int
            let window: Int
            let rms_eps: Float
            let rope_theta: Double
            let ln1: [[Float]]
            let latent: [[Float]]
            let qa_w: [Float]
            let kva_w: [Float]
            let kn_w: [Float]
            let kn_b: [Float]
            let wg_w: [[Float]]
            let qb_a: [[Float]]
            let qb_b: [[Float]]
            let kvb_a: [[Float]]
            let kvb_b: [[Float]]
            let vha_m: [[Float]]
            let sink: [Float]
            let selected: [[Int]]
            let out: [[Float]]
            let out_hq8_f16: [[Float]]
            let out_hq8_q4: [[Float]]
        }
        struct HCASection: Decodable {
            let T: Int
            let q_heads: Int
            let o_groups: Int
            let o_lora: Int
            let head_dim: Int
            let nope_dim: Int
            let rope_dim: Int
            let q_lora: Int
            let compress_ratio: Int
            let window: Int
            let rms_eps: Float
            let fused_in: [[Float]]
            let q_norm_w: [Float]
            let kv_norm_w: [Float]
            let comp_norm_w: [Float]
            let ape: [[Float]]
            let qup_a: [[Float]]
            let qup_b: [[Float]]
            let woa_a: [[Float]]
            let woa_b: [[Float]]
            let vha_m: [[Float]]
            let sink: [Float]
            let comp_kv: [[Float]]
            let out: [[Float]]
        }
        let yarn_rope: YarnSection
        let mla_dsa: MLADSASection
        let hca: HCASection
    }

    private func loadGoldens() throws -> ErnieGoldenFile {
        let data = try Data(contentsOf: Self.fixtureURL)
        return try JSONDecoder().decode(ErnieGoldenFile.self, from: data)
    }

    private static func fwht32InPlace(_ v: inout [Float]) {
        v.withUnsafeMutableBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            for b in stride(from: 0, to: buf.count, by: 32) {
                QWQuant.fwht32OrthoBlock(base + b)
            }
        }
    }

    private static func matmulLowRank(_ a: [[Float]], _ b: [[Float]]) -> [Float] {
        let rows = a.count
        let rank = a[0].count
        let cols = b[0].count
        var out = [Float](repeating: 0, count: rows * cols)
        for r in 0..<rows {
            for p in 0..<rank {
                let av = Float(Float16(a[r][p]))
                for c in 0..<cols {
                    out[r * cols + c] += av * Float(Float16(b[p][c]))
                }
            }
            for c in 0..<cols {
                out[r * cols + c] = Float(Float16(out[r * cols + c]))
            }
        }
        return out
    }

    private static func makeTensorF16(hal: QWHal, shape: [Int], values: [Float]) -> QWTensorValue {
        let bits = values.map { QWQuant.f32tof16Bits($0) }
        let byteCount = max(16, bits.count * 2)
        let buf = hal.device.makeBuffer(length: byteCount, options: .storageModeShared)!
        bits.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let base = raw.baseAddress, !raw.isEmpty {
                buf.contents().copyMemory(from: base, byteCount: raw.count)
            }
        }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length),
                             offset: 0, shape: QWTensorShape(shape, .f16))
    }

    private static func makeTensorF32(hal: QWHal, shape: [Int], values: [Float]) -> QWTensorValue {
        let byteCount = max(16, values.count * 4)
        let buf = hal.device.makeBuffer(length: byteCount, options: .storageModeShared)!
        values.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let base = raw.baseAddress, !raw.isEmpty {
                buf.contents().copyMemory(from: base, byteCount: raw.count)
            }
        }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length),
                             offset: 0, shape: QWTensorShape(shape, .f32))
    }

    private static func packHQ8G32(hal: QWHal, flat: [Float], n: Int, k: Int, rotate: Bool) -> QWWeightView {
        var w = flat
        if rotate {
            for r in 0..<n {
                var row = Array(w[(r * k)..<((r + 1) * k)])
                fwht32InPlace(&row)
                w.replaceSubrange((r * k)..<((r + 1) * k), with: row)
            }
        }
        let group = 32
        let groups = k / group
        let gBytes = QWAQBlock.bytes(bits: 8, group: group)
        var bank = Data(count: n * groups * gBytes)
        bank.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let gOff = (r * groups + g) * gBytes
                    let start = r * k + g * group
                    var mn = Float.infinity
                    var mx = -Float.infinity
                    for i in 0..<group {
                        let v = w[start + i]
                        if v < mn { mn = v }
                        if v > mx { mx = v }
                    }
                    let scH = Float16(max((mx - mn) / 255.0, 1e-8))
                    let biH = Float16(mn)
                    let scBits = scH.bitPattern
                    let biBits = biH.bitPattern
                    ptr[gOff + 0] = UInt8(scBits & 0xFF)
                    ptr[gOff + 1] = UInt8(scBits >> 8)
                    ptr[gOff + 2] = UInt8(biBits & 0xFF)
                    ptr[gOff + 3] = UInt8(biBits >> 8)
                    let scF = Float(scH)
                    let biF = Float(biH)
                    for i in 0..<group {
                        let q = min(255.0, max(0.0, ((w[start + i] - biF) / scF).rounded()))
                        ptr[gOff + 4 + i] = UInt8(q)
                    }
                }
            }
        }
        let buf = hal.device.makeBuffer(length: max(16, bank.count), options: .storageModeShared)!
        bank.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let base = raw.baseAddress {
                buf.contents().copyMemory(from: base, byteCount: bank.count)
            }
        }
        let tv = QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length),
                               offset: 0, shape: QWTensorShape([bank.count / 2], .f16))
        return QWWeightView(blocks: tv, n: n, k: k, bits: 8, groupSize: 32,
                            isRotated: rotate, isIncumbentContainer: false, isSHVQ: false)!
    }

    private static func encodeRowsH16(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                      a: QWTensorValue, w: QWWeightView, c: QWTensorValue, m: Int) throws {
        for r in 0..<m {
            try QWGemmAQ.encodeDecodeT4(
                enc, hal: hal, blocks: w.blocks,
                x: a.slice(byteOffset: r * w.k * 2, count: w.k),
                y16: c.slice(byteOffset: r * w.n * 2, count: w.n),
                n: w.n, k: w.k, bits: w.bits, group: w.groupSize, isSHVQ: w.isSHVQ)
        }
    }

    private static func makeErnieGeometry(
        hidden: Int,
        qHeads: Int,
        hcaQLora: Int,
        oGroups: Int,
        oLora: Int,
        compressRatio: Int,
        window: Int,
        mlaQLora: Int,
        mlaKvLora: Int,
        mlaNope: Int,
        mlaRope: Int,
        mlaVDim: Int,
        idxHeads: Int,
        idxDim: Int,
        idxTopK: Int,
        rmsEps: Float,
        ropeTheta: Double
    ) throws -> QWModelGeometry {
        let cfg = """
        {
          "architectures": ["Ernie5_5_MoeForCausalLM"], "model_type": "ernie5_5_moe",
          "vocab_size": 1024, "hidden_size": \(hidden), "intermediate_size": \(hidden),
          "num_hidden_layers": 2, "max_position_embeddings": 1048576, "rms_norm_eps": \(rmsEps),
          "num_attention_heads": \(qHeads), "num_key_value_heads": 1, "head_dim": 512,
          "compress_ratios": [\(compressRatio), -2], "default_compress_ratio": \(compressRatio),
          "q_lora_rank": \(hcaQLora), "kv_lora_rank": 512,
          "compress_qk_nope_head_dim": 448, "compress_qk_rope_head_dim": 64, "compress_v_head_dim": 512,
          "o_groups": \(oGroups), "o_lora_rank": \(oLora), "sliding_window": \(window),
          "compress_rope_theta": 160000.0, "hca_rope_type": "yarn",
          "rope_scaling": { "type": "yarn", "factor": 16, "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1 },
          "rope_theta": \(ropeTheta), "rotary_percent": 1.0, "rope_interleave": true,
          "hybrid_mla_attention": "mqa_dsa", "hybrid_mla_num_attention_heads": \(qHeads), "hybrid_mla_num_key_value_heads": \(qHeads),
          "hybrid_mla_q_lora_rank": \(mlaQLora), "hybrid_mla_kv_lora_rank": \(mlaKvLora),
          "hybrid_mla_qk_nope_head_dim": \(mlaNope), "hybrid_mla_qk_rope_head_dim": \(mlaRope), "hybrid_mla_v_head_dim": \(mlaVDim),
          "mqa_split_kv_b_proj": true, "index_n_heads": \(idxHeads), "index_head_dim": \(idxDim), "index_topk": \(idxTopK),
          "use_fast_hadamard": true, "use_vha_attention": true, "add_full_attention_sink_bias": true,
          "use_gated_attn": true, "use_attn_merger": true, "gated_attn_use_q_lora": true, "eos_token_id": 2
        }
        """.data(using: .utf8)!
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: cfg))
    }

    func testYarnRopeInvFreqAndRoundtripMatchesPythonOracle() throws {
        let g = try loadGoldens().yarn_rope
        let yarn = QWModelDescriptor.YarnRopeConfig(
            factor: 16.0, originalMaxPositionEmbeddings: 65536, betaFast: 32.0, betaSlow: 1.0)
        let (invFreq, mscale) = QWModelDescriptor.computeRopeInvFreq(
            rotaryDim: 64, base: 160_000.0, yarn: yarn)
        XCTAssertEqual(mscale, 1.0, accuracy: 1e-6)
        XCTAssertEqual(invFreq.count, g.inv_freq.count)
        for i in 0..<invFreq.count {
            XCTAssertEqual(invFreq[i], g.inv_freq[i], accuracy: 1e-6, "inv_freq[\(i)] mismatch")
        }
        for row in 0..<g.x.count {
            let pos = Float(g.positions[row])
            for pair in 0..<32 {
                let even = g.x[row][2 * pair]
                let odd = g.x[row][2 * pair + 1]
                let ang = pos * invFreq[pair]
                let c = cos(ang) * mscale
                let s = sin(ang) * mscale
                let fwdE = even * c - odd * s
                let fwdO = odd * c + even * s
                XCTAssertEqual(fwdE, g.fwd[row][2 * pair], accuracy: 2e-4)
                XCTAssertEqual(fwdO, g.fwd[row][2 * pair + 1], accuracy: 2e-4)
                let invE = fwdE * c + fwdO * s
                let invO = fwdO * c - fwdE * s
                XCTAssertEqual(invE, g.roundtrip[row][2 * pair], accuracy: 3e-4)
                XCTAssertEqual(invO, g.roundtrip[row][2 * pair + 1], accuracy: 3e-4)
            }
        }
    }

    func testCoarsePoolIndexPoolShiftZero64HeadsAndWindowExclusion() throws {
        let geo = try Self.makeErnieGeometry(
            hidden: 128, qHeads: 16, hcaQLora: 128, oGroups: 2, oLora: 128,
            compressRatio: 4, window: 128, mlaQLora: 64, mlaKvLora: 512,
            mlaNope: 192, mlaRope: 64, mlaVDim: 256, idxHeads: 64, idxDim: 128,
            idxTopK: 256, rmsEps: 1e-6, ropeTheta: 8_000_000.0)
        let hal = try QWHal(geometry: geo)
        let op = try QWMLAAttnOperator(
            hal: hal, geometry: geo, numAttnLayers: 1, maxContext: 20_480, maxPrefillChunk: 32,
            lsaConfig: QWLSAConfig(mode: .auto))
        XCTAssertTrue(op.isDecoupledDSA)
        XCTAssertEqual(op.poolShift, 0)
        XCTAssertEqual(op.windowTokens, 128)
        XCTAssertTrue(op.emeansCoarseEnabled)

        let totalTokens = 17_000
        let needleTokens = [913, 7411, 14_802]
        let ls = op.layers[0]
        let basePtr = op.region.pointer

        var state: UInt64 = 0x1234_5678_9ABC_DEF0
        func nextRand() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: state >> 33)) / Float(Int32.max)
        }
        var needleDir = (0..<128).map { _ in nextRand() }
        let nNorm = sqrt(needleDir.reduce(0) { $0 + $1 * $1 })
        for d in 0..<128 { needleDir[d] /= nNorm }

        let pPtr = basePtr.advanced(by: ls.pooledKeys.offset).bindMemory(to: UInt16.self, capacity: totalTokens * 128)
        for tok in 0..<totalTokens {
            let isNeedle = needleTokens.contains(tok)
            for d in 0..<128 {
                let v = isNeedle ? (needleDir[d] * 3.0) : (nextRand() * 0.25)
                pPtr[tok * 128 + d] = QWQuant.f32tof16Bits(v)
            }
        }

        let qPtr = basePtr.advanced(by: op.mlaQFullScratch.offset).bindMemory(to: UInt16.self, capacity: op.qFullStride)
        for i in 0..<op.qFullStride { qPtr[i] = 0 }
        for h in 0..<64 {
            for d in 0..<128 {
                qPtr[op.qChannels + h * 128 + d] = QWQuant.f32tof16Bits(needleDir[d])
            }
        }
        let wPtr = basePtr.advanced(by: op.idxWeightsScratch.offset).bindMemory(to: Float.self, capacity: 64)
        for h in 0..<64 { wPtr[h] = 1.0 / 8.0 }

        guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
        op.updateCoarseClusters(enc, ls: ls, fullAttnLayer: 0, pos0: 0, rows: totalTokens)
        op.encodeIndexerSelection(enc, ls: ls, pos0: totalTokens - 1, rows: 1, probeClusters: 480, windowBlocks: 128)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let count = basePtr.advanced(by: op.blockCountScratch.offset).assumingMemoryBound(to: Int32.self).pointee
        XCTAssertEqual(Int(count), 256 + 128)
        let listPtr = basePtr.advanced(by: op.blockListScratch.offset).bindMemory(to: Int32.self, capacity: Int(count))
        let selected = Array(UnsafeBufferPointer(start: listPtr, count: Int(count))).map(Int.init)
        let tailWindow = Array(selected.suffix(128))
        XCTAssertEqual(tailWindow, Array((totalTokens - 128)..<totalTokens))
        for nt in needleTokens {
            XCTAssertTrue(selected.prefix(256).contains(nt), "Expected needle token \(nt) in top-256 e-means selection")
        }
    }

    func testMLADecoupledRoPEDSAMatchesPythonOracleAndSnapshot() throws {
        let g = try loadGoldens().mla_dsa
        let geo = try Self.makeErnieGeometry(
            hidden: g.hidden, qHeads: g.q_heads, hcaQLora: 128, oGroups: 2, oLora: 128,
            compressRatio: 4, window: g.window, mlaQLora: g.q_lora, mlaKvLora: g.kv_lora,
            mlaNope: g.nope_dim, mlaRope: g.rope_dim, mlaVDim: g.v_dim,
            idxHeads: g.idx_heads, idxDim: g.idx_dim, idxTopK: g.topk,
            rmsEps: g.rms_eps, ropeTheta: g.rope_theta)
        let hal = try QWHal(geometry: geo)
        let rot = hal.weightBasis.rotatesActivations

        let qbRows = g.q_heads * (g.nope_dim + g.rope_dim + g.v_dim) + g.idx_heads * g.idx_dim
        let qbFlat = Self.matmulLowRank(g.qb_a, g.qb_b)
        let kvbRows = g.q_heads * (g.nope_dim + g.v_dim)
        let kvbFlat = Self.matmulLowRank(g.kvb_a, g.kvb_b)
        let qBProj = Self.packHQ8G32(hal: hal, flat: qbFlat, n: qbRows, k: g.q_lora, rotate: rot)
        let kvBProj = Self.packHQ8G32(hal: hal, flat: kvbFlat, n: kvbRows, k: g.kv_lora, rotate: rot)

        var wgFlat = g.wg_w.flatMap { $0 }
        if rot {
            for r in 0..<g.idx_heads {
                var row = Array(wgFlat[(r * g.hidden)..<((r + 1) * g.hidden)])
                Self.fwht32InPlace(&row)
                wgFlat.replaceSubrange((r * g.hidden)..<((r + 1) * g.hidden), with: row)
            }
        }
        let mlaW = QWMLAAttnOperator.LayerWeights(
            qANormW: Self.makeTensorF16(hal: hal, shape: [g.q_lora], values: g.qa_w),
            kvANormW: Self.makeTensorF16(hal: hal, shape: [g.kv_lora], values: g.kva_w),
            kNormW: Self.makeTensorF16(hal: hal, shape: [g.idx_dim], values: g.kn_w),
            kNormB: Self.makeTensorF16(hal: hal, shape: [g.idx_dim], values: g.kn_b),
            wGateWeights: Self.makeTensorF16(hal: hal, shape: [g.idx_heads, g.hidden], values: wgFlat),
            ape: nil,
            qBProj: qBProj,
            kvBProj: kvBProj,
            vhaMix: Self.makeTensorF16(hal: hal, shape: [g.q_heads, g.q_heads], values: g.vha_m.flatMap { $0 }),
            attnSink: Self.makeTensorF32(hal: hal, shape: [g.q_heads], values: g.sink),
            hasOGate: true
        )

        var ln1Flat = [Float]()
        for row in g.ln1 {
            var r = row
            if rot { Self.fwht32InPlace(&r) }
            ln1Flat.append(contentsOf: r)
        }
        let outCols = g.q_heads * g.v_dim

        for kvMode in [QWKVMode.f16, QWKVMode.q4] {
            let pageBytes = QWKVCache.pageBytes(mode: kvMode, geometry: geo)
            let tables = QWKVCache.tableSlotBytes * 2 * 1 * QWKVCache.maxSeqs
            let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: pageBytes * 4 + tables, gdnStates: 4096)
            let cache = try QWKVCache(hal: hal, arena: arena, layers: [1], mode: kvMode, geometry: geo)

            for mmaFlag in [false, true] {
                if mmaFlag && !hal.caps.isMetal4TensorCapable { continue }
                let seq = try cache.allocate(promptTokens: 32)
                let kvViews = try cache.layerViews(seq, layer: 1)
                let op = try QWMLAAttnOperator(hal: hal, geometry: geo, numAttnLayers: 1, maxContext: 512, maxPrefillChunk: 32)
                op.sparseAttnMMAEnabled = mmaFlag

                let latFlat = g.latent.flatMap { $0 }
                let latPtr = op.region.pointer.advanced(by: op.mlaLatentScratch.offset)
                    .bindMemory(to: UInt16.self, capacity: latFlat.count)
                for i in 0..<latFlat.count { latPtr[i] = QWQuant.f32tof16Bits(latFlat[i]) }

                let ln1Tensor = Self.makeTensorF16(hal: hal, shape: [g.T, g.hidden], values: ln1Flat)
                let attOut = Self.makeTensorF16(hal: hal, shape: [g.T, outCols], values: [Float](repeating: 0, count: g.T * outCols))

                guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
                op.encodeLatentPrepAndPool(enc, ln1: ln1Tensor, mlaW: mlaW, kv: kvViews, fullAttnLayer: 0, pos0: 0, rows: g.T)
                try Self.encodeRowsH16(enc, hal: hal, a: op.qResidScratch, w: mlaW.qBProj, c: op.mlaQFullScratch, m: g.T)
                op.encodeScoreSelectAbsorbAttendAndProjUV(
                    enc, mlaW: mlaW, kv: kvViews, attOut: attOut, fullAttnLayer: 0, pos0: 0, rows: g.T)
                hal.end(cb, enc)
                cb.waitUntilCompleted()
                XCTAssertNil(cb.error)

                let outPtr = attOut.region.pointer.advanced(by: attOut.offset)
                    .bindMemory(to: UInt16.self, capacity: g.T * outCols)
                let expectedOut = (kvMode == .q4) ? g.out_hq8_q4 : (rot ? g.out_hq8_f16 : g.out)
                var maxErr: Float = 0
                for t in 0..<g.T {
                    var row = (0..<outCols).map { Float(Float16(bitPattern: outPtr[t * outCols + $0])) }
                    if rot { Self.fwht32InPlace(&row) }
                    for c in 0..<outCols {
                        maxErr = max(maxErr, abs(row[c] - expectedOut[t][c]))
                    }
                }
                let tol: Float = (kvMode == .f16) ? 2.5e-3 : 3.5e-3
                XCTAssertLessThan(maxErr, tol, "MLA-DSA maxErr \(maxErr) exceeded \(tol) (kv=\(kvMode), mma=\(mmaFlag))")

                if kvMode == .f16 && !mmaFlag {
                    let listBase = op.region.pointer.advanced(by: op.blockListScratch.offset)
                        .bindMemory(to: Int32.self, capacity: g.T * op.listStride)
                    for t in (g.topk + g.window)..<g.T {
                        let got = (0..<(g.topk + g.window)).map { Int(listBase[t * op.listStride + $0]) }
                        XCTAssertEqual(Set(got), Set(g.selected[t]), "DSA selected mismatch at row \(t)")
                        XCTAssertEqual(Array(got.suffix(g.window)), Array(g.selected[t].suffix(g.window)))
                    }

                    let snapPos = 5
                    let snap = op.captureIndexerState(position: snapPos)
                    XCTAssertEqual(snap.count, op.indexerStateBytes(position: snapPos))
                    memset(op.region.pointer.advanced(by: op.layers[0].pooledKeys.offset), 0x5A, snapPos * g.idx_dim * 2)
                    XCTAssertTrue(op.restoreIndexerState(snap, position: snapPos))

                    for stepT in snapPos..<g.T {
                        let rowLat = g.latent[stepT]
                        for i in 0..<rowLat.count { latPtr[i] = QWQuant.f32tof16Bits(rowLat[i]) }
                        let stepLn1 = Self.makeTensorF16(
                            hal: hal, shape: [1, g.hidden],
                            values: Array(ln1Flat[(stepT * g.hidden)..<((stepT + 1) * g.hidden)]))
                        let stepOut = Self.makeTensorF16(hal: hal, shape: [1, outCols], values: [Float](repeating: 0, count: outCols))
                        guard let (cb2, enc2) = hal.begin() else { XCTFail("hal.begin"); return }
                        op.encodeLatentPrepAndPool(enc2, ln1: stepLn1, mlaW: mlaW, kv: kvViews, fullAttnLayer: 0, pos0: stepT, rows: 1)
                        try Self.encodeRowsH16(enc2, hal: hal, a: op.qResidScratch, w: mlaW.qBProj, c: op.mlaQFullScratch, m: 1)
                        op.encodeScoreSelectAbsorbAttendAndProjUV(
                            enc2, mlaW: mlaW, kv: kvViews, attOut: stepOut, fullAttnLayer: 0, pos0: stepT, rows: 1)
                        hal.end(cb2, enc2)
                        cb2.waitUntilCompleted()
                        XCTAssertNil(cb2.error)

                        let sPtr = stepOut.region.pointer.advanced(by: stepOut.offset).bindMemory(to: UInt16.self, capacity: outCols)
                        var sRow = (0..<outCols).map { Float(Float16(bitPattern: sPtr[$0])) }
                        if rot { Self.fwht32InPlace(&sRow) }
                        for c in 0..<outCols {
                            XCTAssertEqual(sRow[c], expectedOut[stepT][c], accuracy: 2.5e-3, "Snapshot decode mismatch at step \(stepT)")
                        }
                    }
                }
                cache.finish(seq)
            }
        }
    }

    func testMLADSAHardwareNAXPrefillMatchesMMA() throws {
        let g = try loadGoldens().mla_dsa
        let geo = try Self.makeErnieGeometry(
            hidden: g.hidden, qHeads: g.q_heads, hcaQLora: 128, oGroups: 2, oLora: 128,
            compressRatio: 4, window: g.window, mlaQLora: g.q_lora, mlaKvLora: g.kv_lora,
            mlaNope: g.nope_dim, mlaRope: g.rope_dim, mlaVDim: g.v_dim,
            idxHeads: g.idx_heads, idxDim: g.idx_dim, idxTopK: g.topk,
            rmsEps: g.rms_eps, ropeTheta: g.rope_theta)
        let hal = try QWHal(geometry: geo)
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable, "Metal 4 NAX tensor required")
        let rot = hal.weightBasis.rotatesActivations

        let rows = 16
        let qbRows = g.q_heads * (g.nope_dim + g.rope_dim + g.v_dim) + g.idx_heads * g.idx_dim
        let qBProj = Self.packHQ8G32(hal: hal, flat: Self.matmulLowRank(g.qb_a, g.qb_b), n: qbRows, k: g.q_lora, rotate: rot)
        let kvBProj = Self.packHQ8G32(hal: hal, flat: Self.matmulLowRank(g.kvb_a, g.kvb_b),
                                      n: g.q_heads * (g.nope_dim + g.v_dim), k: g.kv_lora, rotate: rot)
        var wgFlat = g.wg_w.flatMap { $0 }
        if rot {
            for r in 0..<g.idx_heads {
                var row = Array(wgFlat[(r * g.hidden)..<((r + 1) * g.hidden)])
                Self.fwht32InPlace(&row)
                wgFlat.replaceSubrange((r * g.hidden)..<((r + 1) * g.hidden), with: row)
            }
        }
        let mlaW = QWMLAAttnOperator.LayerWeights(
            qANormW: Self.makeTensorF16(hal: hal, shape: [g.q_lora], values: g.qa_w),
            kvANormW: Self.makeTensorF16(hal: hal, shape: [g.kv_lora], values: g.kva_w),
            kNormW: Self.makeTensorF16(hal: hal, shape: [g.idx_dim], values: g.kn_w),
            kNormB: Self.makeTensorF16(hal: hal, shape: [g.idx_dim], values: g.kn_b),
            wGateWeights: Self.makeTensorF16(hal: hal, shape: [g.idx_heads, g.hidden], values: wgFlat),
            ape: nil, qBProj: qBProj, kvBProj: kvBProj,
            vhaMix: Self.makeTensorF16(hal: hal, shape: [g.q_heads, g.q_heads], values: g.vha_m.flatMap { $0 }),
            attnSink: Self.makeTensorF32(hal: hal, shape: [g.q_heads], values: g.sink),
            hasOGate: true
        )

        let pageBytes = QWKVCache.pageBytes(mode: .q4, geometry: geo)
        let tables = QWKVCache.tableSlotBytes * 2 * 1 * QWKVCache.maxSeqs
        let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: pageBytes * 4 + tables, gdnStates: 4096)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: [1], mode: .q4, geometry: geo)
        let outCols = g.q_heads * g.v_dim

        var ln1Rows = [Float]()
        var latRows = [Float]()
        for t in 0..<rows {
            var r = g.ln1[t % g.T]
            if rot { Self.fwht32InPlace(&r) }
            ln1Rows.append(contentsOf: r)
            latRows.append(contentsOf: g.latent[t % g.T])
        }

        func run(useMMA: Bool) throws -> [Float] {
            let seq = try cache.allocate(promptTokens: 32)
            defer { cache.finish(seq) }
            let kvViews = try cache.layerViews(seq, layer: 1)
            let op = try QWMLAAttnOperator(hal: hal, geometry: geo, numAttnLayers: 1, maxContext: 512, maxPrefillChunk: 32)
            op.sparseAttnMMAEnabled = useMMA
            let latPtr = op.region.pointer.advanced(by: op.mlaLatentScratch.offset)
                .bindMemory(to: UInt16.self, capacity: latRows.count)
            for i in 0..<latRows.count { latPtr[i] = QWQuant.f32tof16Bits(latRows[i]) }
            let ln1T = Self.makeTensorF16(hal: hal, shape: [rows, g.hidden], values: ln1Rows)
            let outT = Self.makeTensorF16(hal: hal, shape: [rows, outCols], values: [Float](repeating: 0, count: rows * outCols))
            guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return [] }
            op.encodeLatentPrepAndPool(enc, ln1: ln1T, mlaW: mlaW, kv: kvViews, fullAttnLayer: 0, pos0: 0, rows: rows)
            try Self.encodeRowsH16(enc, hal: hal, a: op.qResidScratch, w: mlaW.qBProj, c: op.mlaQFullScratch, m: rows)
            op.encodeScoreSelectAbsorbAttendAndProjUV(
                enc, mlaW: mlaW, kv: kvViews, attOut: outT, fullAttnLayer: 0, pos0: 0, rows: rows)
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
            let p = outT.region.pointer.advanced(by: outT.offset).bindMemory(to: UInt16.self, capacity: rows * outCols)
            return (0..<(rows * outCols)).map { Float(Float16(bitPattern: p[$0])) }
        }

        let scalarOut = try run(useMMA: false)
        let naxOut = try run(useMMA: true)
        var maxDiff: Float = 0
        for i in 0..<scalarOut.count { maxDiff = max(maxDiff, abs(scalarOut[i] - naxOut[i])) }
        XCTAssertLessThan(maxDiff, 4e-3, "NAX vs scalar q4-KV MLA-DSA maxDiff \(maxDiff)")
    }

    func testHCAMatchesPythonOracleRawAndQuantizedWoAWithSnapshot() throws {
        let g = try loadGoldens().hca
        let geo = try Self.makeErnieGeometry(
            hidden: 256, qHeads: g.q_heads, hcaQLora: g.q_lora, oGroups: g.o_groups, oLora: g.o_lora,
            compressRatio: g.compress_ratio, window: g.window, mlaQLora: 64, mlaKvLora: 512,
            mlaNope: 192, mlaRope: 64, mlaVDim: 256, idxHeads: 64, idxDim: 128,
            idxTopK: 4, rmsEps: g.rms_eps, ropeTheta: 8_000_000.0)
        let hal = try QWHal(geometry: geo)
        let rot = hal.weightBasis.rotatesActivations

        let qupRows = g.q_heads * g.head_dim + g.o_groups * g.o_lora
        let wQUpGate = Self.packHQ8G32(hal: hal, flat: Self.matmulLowRank(g.qup_a, g.qup_b),
                                       n: qupRows, k: g.q_lora, rotate: rot)
        let groupIn = (g.q_heads / g.o_groups) * g.head_dim
        let woaFlat = Self.matmulLowRank(g.woa_a, g.woa_b)
        let woAHQ8 = Self.packHQ8G32(hal: hal, flat: woaFlat, n: g.o_groups * g.o_lora, k: groupIn, rotate: rot)
        let woARaw = Self.makeTensorF16(hal: hal, shape: [g.o_groups * g.o_lora, groupIn], values: woaFlat)

        let qNormT = Self.makeTensorF16(hal: hal, shape: [g.q_lora], values: g.q_norm_w)
        let kvNormT = Self.makeTensorF16(hal: hal, shape: [g.head_dim], values: g.kv_norm_w)
        let compNormT = Self.makeTensorF16(hal: hal, shape: [g.head_dim], values: g.comp_norm_w)
        let apeT = Self.makeTensorF32(hal: hal, shape: [g.compress_ratio, g.head_dim], values: g.ape.flatMap { $0 })
        let vhaT = Self.makeTensorF16(hal: hal, shape: [g.q_heads, g.q_heads], values: g.vha_m.flatMap { $0 })
        let sinkT = Self.makeTensorF32(hal: hal, shape: [g.q_heads], values: g.sink)

        let wRaw = QWHCAAttnOperator.LayerWeights(
            qNormW: qNormT, kvNormW: kvNormT, compNormW: compNormT, ape: apeT,
            wQUpGate: wQUpGate, woA: nil, woARawF16: woARaw, vhaMix: vhaT, attnSink: sinkT)
        let wQuant = QWHCAAttnOperator.LayerWeights(
            qNormW: qNormT, kvNormW: kvNormT, compNormW: compNormT, ape: apeT,
            wQUpGate: wQUpGate, woA: woAHQ8, woARawF16: nil, vhaMix: vhaT, attnSink: sinkT)

        for (label, lw) in [("rawF16", wRaw), ("HQ8_G32", wQuant)] {
            let op = try QWHCAAttnOperator(hal: hal, geometry: geo, numHcaLayers: 1, maxContext: 256, maxPrefillChunk: 32)
            let inFlat = g.fused_in.flatMap { $0 }
            let inPtr = op.region.pointer.advanced(by: op.fusedInScratch.offset)
                .bindMemory(to: UInt16.self, capacity: inFlat.count)
            for i in 0..<inFlat.count { inPtr[i] = QWQuant.f32tof16Bits(inFlat[i]) }

            guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
            op.encodeInPrepAndCompress(enc, hcaW: lw, hcaLayer: 0, pos0: 0, rows: g.T)
            try Self.encodeRowsH16(enc, hal: hal, a: op.qrScratch, w: wQUpGate, c: op.qUpGateScratch, m: g.T)
            op.encodeQPrepAttendVHAAndWoA(
                enc, hcaW: lw, zGatedOut: op.zGatedScratch, hcaLayer: 0, pos0: 0, rows: g.T, skipExitRotation: true)
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)

            let cPtr = op.region.pointer.advanced(by: op.layers[0].compKV.offset)
                .bindMemory(to: UInt16.self, capacity: g.comp_kv.count * g.head_dim)
            for m in 0..<g.comp_kv.count {
                for d in 0..<g.head_dim {
                    let got = Float(Float16(bitPattern: cPtr[m * g.head_dim + d]))
                    XCTAssertEqual(got, g.comp_kv[m][d], accuracy: 2e-3, "\(label) compKV[\(m)][\(d)]")
                }
            }

            let zPtr = op.region.pointer.advanced(by: op.zGatedScratch.offset)
                .bindMemory(to: UInt16.self, capacity: g.T * op.zOutStride)
            var maxErr: Float = 0
            for t in 0..<g.T {
                for c in 0..<op.zOutStride {
                    let got = Float(Float16(bitPattern: zPtr[t * op.zOutStride + c]))
                    maxErr = max(maxErr, abs(got - g.out[t][c]))
                }
            }
            XCTAssertLessThan(maxErr, 6e-3, "\(label) HCA zGated maxErr \(maxErr)")
        }

        // Grouped wo_a MMA path (rows = 18 >= 16) vs GEMV step-by-step parity + `.qwcache` snapshot roundtrip.
        let rowsMMA = 18
        let opMMA = try QWHCAAttnOperator(hal: hal, geometry: geo, numHcaLayers: 1, maxContext: 256, maxPrefillChunk: 32)
        let inPtrM = opMMA.region.pointer.advanced(by: opMMA.fusedInScratch.offset)
            .bindMemory(to: UInt16.self, capacity: rowsMMA * opMMA.inStride)
        for t in 0..<rowsMMA {
            let src = g.fused_in[t % g.T]
            for c in 0..<opMMA.inStride { inPtrM[t * opMMA.inStride + c] = QWQuant.f32tof16Bits(src[c]) }
        }
        guard let (cbM, encM) = hal.begin() else { XCTFail("hal.begin"); return }
        opMMA.encodeInPrepAndCompress(encM, hcaW: wQuant, hcaLayer: 0, pos0: 0, rows: rowsMMA)
        try Self.encodeRowsH16(encM, hal: hal, a: opMMA.qrScratch, w: wQUpGate, c: opMMA.qUpGateScratch, m: rowsMMA)
        opMMA.encodeQPrepAttendVHAAndWoA(
            encM, hcaW: wQuant, zGatedOut: opMMA.zGatedScratch, hcaLayer: 0, pos0: 0, rows: rowsMMA)
        hal.end(cbM, encM)
        cbM.waitUntilCompleted()
        XCTAssertNil(cbM.error)
        let mmaOutPtr = opMMA.region.pointer.advanced(by: opMMA.zGatedScratch.offset)
            .bindMemory(to: UInt16.self, capacity: rowsMMA * opMMA.zOutStride)
        let mmaFull = Array(UnsafeBufferPointer(start: mmaOutPtr, count: rowsMMA * opMMA.zOutStride))

        let snapPos = 6
        let opSnap = try QWHCAAttnOperator(hal: hal, geometry: geo, numHcaLayers: 1, maxContext: 256, maxPrefillChunk: 32)
        let sInPtr = opSnap.region.pointer.advanced(by: opSnap.fusedInScratch.offset)
            .bindMemory(to: UInt16.self, capacity: rowsMMA * opSnap.inStride)
        for t in 0..<snapPos {
            let src = g.fused_in[t % g.T]
            for c in 0..<opSnap.inStride { sInPtr[t * opSnap.inStride + c] = QWQuant.f32tof16Bits(src[c]) }
        }
        guard let (cbS0, encS0) = hal.begin() else { XCTFail("hal.begin"); return }
        opSnap.encodeInPrepAndCompress(encS0, hcaW: wQuant, hcaLayer: 0, pos0: 0, rows: snapPos)
        hal.end(cbS0, encS0)
        cbS0.waitUntilCompleted()

        let stateData = opSnap.captureState(position: snapPos)
        XCTAssertEqual(stateData.count, opSnap.stateBytes(position: snapPos))
        memset(opSnap.region.pointer.advanced(by: opSnap.layers[0].compKV.offset), 0xA5, 2 * g.head_dim * 2)
        memset(opSnap.region.pointer.advanced(by: opSnap.layers[0].swaKV.offset), 0x5A, opSnap.ringCap * g.head_dim * 2)
        XCTAssertFalse(opSnap.restoreState(stateData.dropLast(), position: snapPos))
        XCTAssertTrue(opSnap.restoreState(stateData, position: snapPos))

        for t in snapPos..<rowsMMA {
            let src = g.fused_in[t % g.T]
            for c in 0..<opSnap.inStride { sInPtr[c] = QWQuant.f32tof16Bits(src[c]) }
            guard let (cbS, encS) = hal.begin() else { XCTFail("hal.begin"); return }
            opSnap.encodeInPrepAndCompress(encS, hcaW: wQuant, hcaLayer: 0, pos0: t, rows: 1)
            try Self.encodeRowsH16(encS, hal: hal, a: opSnap.qrScratch, w: wQUpGate, c: opSnap.qUpGateScratch, m: 1)
            opSnap.encodeQPrepAttendVHAAndWoA(
                encS, hcaW: wQuant, zGatedOut: opSnap.zGatedScratch, hcaLayer: 0, pos0: t, rows: 1)
            hal.end(cbS, encS)
            cbS.waitUntilCompleted()
            XCTAssertNil(cbS.error)

            let stepPtr = opSnap.region.pointer.advanced(by: opSnap.zGatedScratch.offset)
                .bindMemory(to: UInt16.self, capacity: opSnap.zOutStride)
            var maxStepDiff: Float = 0
            for c in 0..<opSnap.zOutStride {
                let got = Float(Float16(bitPattern: stepPtr[c]))
                let exp = Float(Float16(bitPattern: mmaFull[t * opSnap.zOutStride + c]))
                maxStepDiff = max(maxStepDiff, abs(got - exp))
            }
            XCTAssertLessThan(maxStepDiff, 3e-3, "HCA snapshot GEMV vs MMA diff \(maxStepDiff) at t=\(t)")
        }
    }

    func testHCALSAEmeansPruningRecallsNeedlesAboveBudget() throws {
        let geo = try Self.makeErnieGeometry(
            hidden: 256, qHeads: 16, hcaQLora: 128, oGroups: 2, oLora: 128,
            compressRatio: 4, window: 4, mlaQLora: 64, mlaKvLora: 512,
            mlaNope: 192, mlaRope: 64, mlaVDim: 256, idxHeads: 64, idxDim: 128,
            idxTopK: 4, rmsEps: 1e-6, ropeTheta: 8_000_000.0)
        let hal = try QWHal(geometry: geo)
        let lsaCfg = QWLSAConfig(mode: .svd, budget: 32, clusterBlock: QWLSAConfig.emeansClusterBlock)
        let op = try QWHCAAttnOperator(
            hal: hal, geometry: geo, numHcaLayers: 1, maxContext: 18_432, maxPrefillChunk: 32,
            lsaConfig: lsaCfg, indexDim: 64, lsaBudgetBlocksOverride: 8)
        XCTAssertTrue(op.lsaAllocated)
        XCTAssertEqual(op.compBudget, 9)

        let basis = QWLSASidecar.synthesizeDeterministicOrthogonalBasis(inDim: 448, indexDim: 64, seed: 777)
        let wqT = Self.makeTensorF16(hal: hal, shape: [64, 448], values: basis.map { Float(Float16(bitPattern: $0)) })
        let ls = op.layers[0]
        let pKeys = try XCTUnwrap(ls.pooledKeys)
        let bList = try XCTUnwrap(op.blockListScratch)
        let bCount = try XCTUnwrap(op.blockCountScratch)

        let totalPools = 4200
        let needlePools = [311, 2190, 3985]
        var state: UInt64 = 0xCAFE_BABE_1122_3344
        func nextRand() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: state >> 33)) / Float(Int32.max)
        }

        // Build query direction in 448D NoPE space and compute its projected 64D unit landmark.
        var qDir448 = (0..<448).map { _ in nextRand() }
        let qNorm = sqrt(qDir448.reduce(0) { $0 + $1 * $1 })
        for d in 0..<448 { qDir448[d] /= qNorm }
        var qProj64 = [Float](repeating: 0, count: 64)
        for i in 0..<64 {
            var acc: Float = 0
            for d in 0..<448 { acc += Float(Float16(bitPattern: basis[i * 448 + d])) * qDir448[d] }
            qProj64[i] = acc
        }
        let qpNorm = sqrt(qProj64.reduce(0) { $0 + $1 * $1 })
        for i in 0..<64 { qProj64[i] /= qpNorm }

        let pPtr = op.region.pointer.advanced(by: pKeys.offset).bindMemory(to: UInt16.self, capacity: totalPools * 64)
        for m in 0..<totalPools {
            if needlePools.contains(m) {
                for i in 0..<64 { pPtr[m * 64 + i] = QWQuant.f32tof16Bits(qProj64[i]) }
            } else {
                let row = (0..<64).map { _ in nextRand() }
                let rn = max(1e-6, sqrt(row.reduce(0) { $0 + $1 * $1 }))
                for i in 0..<64 { pPtr[m * 64 + i] = QWQuant.f32tof16Bits(row[i] / rn * 0.6) }
            }
        }

        let qUpPtr = op.region.pointer.advanced(by: op.qUpGateScratch.offset)
            .bindMemory(to: UInt16.self, capacity: op.qUpStride)
        for i in 0..<op.qUpStride { qUpPtr[i] = 0 }
        for h in 0..<op.qHeads {
            for d in 0..<448 { qUpPtr[h * op.headDim + d] = QWQuant.f32tof16Bits(qDir448[d]) }
        }

        var qP = QWHCAAttnOperator.QPrepParams(
            T: 1, pos0: UInt32(totalPools * 4), qHeads: UInt32(op.qHeads),
            headDim: UInt32(op.headDim), nopeDim: UInt32(op.nopeDim), ropeDim: UInt32(op.ropeDim),
            qUpStride: UInt32(op.qUpStride), indexDim: 64, rmsEps: 1e-6, mscale: 1.0)

        guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
        op.updateCoarseClusters(enc, ls: ls, hcaLayer: 0, pos0: 0, rows: totalPools * 4)
        op.encodeLSASelection(enc, ls: ls, wqIdx: wqT, qPrepParams: &qP, pos0: totalPools * 4, rows: 1)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        XCTAssertEqual(op.clusteredMacros[0], 4)

        let cnt = Int(op.region.pointer.advanced(by: bCount.offset).assumingMemoryBound(to: Int32.self).pointee)
        XCTAssertEqual(cnt, 9)
        let lPtr = op.region.pointer.advanced(by: bList.offset).bindMemory(to: Int32.self, capacity: cnt)
        let selected = Array(UnsafeBufferPointer(start: lPtr, count: cnt)).map(Int.init)
        XCTAssertEqual(selected.first, 0, "Sink block 0 must always be retained")
        for np in needlePools {
            XCTAssertTrue(selected.contains(np), "Expected HCA LSA e-means to recall compressed needle \(np), got \(selected)")
        }
    }
}
