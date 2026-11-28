// QWMLAAttnOperatorTests.swift — Unit and GPU parity tests for NoPE MLA
// (`QWMLAAttnOperator`) and Key-Pool(4) sparse indexer kernels.

import XCTest
import Metal
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc
import QWKV
import QWQuant
import QWGEMM

final class QWMLAAttnOperatorTests: XCTestCase {
    override func setUpWithError() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.supportsSimdgroupMMA == true,
            "MLA/DSA GPU operator tests require bare-metal Apple Silicon GPU"
        )
    }

    static func makeBuffer<T: BitwiseCopyable>(_ values: [T], device: any MTLDevice) -> any MTLBuffer {
        let byteCount = values.count * MemoryLayout<T>.stride
        let buffer = device.makeBuffer(length: max(16, byteCount), options: .storageModeShared)!
        values.withUnsafeBufferPointer { source in
            if let baseAddress = source.baseAddress {
                buffer.contents().copyMemory(from: baseAddress, byteCount: byteCount)
            }
        }
        return buffer
    }

    func testBufferCopiesShortAndEmptyInputs() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let values: [UInt16] = [1, 23, 456]
        let short = Self.makeBuffer(values, device: device)
        XCTAssertEqual(short.length, 16)
        let copied = short.contents().bindMemory(to: UInt16.self, capacity: values.count)
        XCTAssertEqual(Array(UnsafeBufferPointer(start: copied, count: values.count)), values)
        XCTAssertEqual(Self.makeBuffer([UInt16](), device: device).length, 16)
    }

    static func mlaGeometry() throws -> QWModelGeometry {
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
          "index_topk": 2048,
          "index_kpool_compress": true,
          "index_kpool_always_select_tail": true,
          "index_share_for_mtp_iteration": true,
          "indexer_rope_interleave": true,
          "eos_token_id": 154820
        }
        """.data(using: .utf8)!
        let desc = try QWModelDescriptor.decode(configJSON: cfg)
        return QWModelGeometry(descriptor: desc)
    }

    private func makeTensor(hal: QWHal, f16Values: [Float]) -> QWTensorValue {
        let bits = f16Values.map { QWQuant.f32tof16Bits($0) }
        let b = Self.makeBuffer(bits, device: hal.device)
        return QWTensorValue(region: QWMRegion(buffer: b, base: 0, length: b.length),
                             offset: 0, shape: QWTensorShape([f16Values.count], .f16))
    }

    private func makeTensorF32(hal: QWHal, f32Values: [Float]) -> QWTensorValue {
        let b = Self.makeBuffer(f32Values, device: hal.device)
        return QWTensorValue(region: QWMRegion(buffer: b, base: 0, length: b.length),
                             offset: 0, shape: QWTensorShape([f32Values.count], .f32))
    }

    func testIndexerStateCaptureRestoresPoolsAndOpenBlockRows() throws {
        let geo = try Self.mlaGeometry()
        let hal = try QWHal(geometry: geo)
        let op = try QWMLAAttnOperator(hal: hal, geometry: geo, numAttnLayers: 2, maxContext: 512, maxPrefillChunk: 16)
        let rowBytes = op.idxDim * 2
        let base = op.region.pointer
        func fill(_ seed: UInt8) {
            for ls in op.layers {
                for t in [ls.kPending, ls.gatePending, ls.pooledKeys] {
                    let p = base.advanced(by: t.offset).assumingMemoryBound(to: UInt8.self)
                    for i in 0..<t.bytes { p[i] = UInt8(truncatingIfNeeded: i &* 31 &+ Int(seed) &+ t.offset) }
                }
            }
        }
        func bytes(_ t: QWTensorValue, row: Int) -> [UInt8] {
            let p = base.advanced(by: t.offset + row * rowBytes).assumingMemoryBound(to: UInt8.self)
            return Array(UnsafeBufferPointer(start: p, count: rowBytes))
        }
        let position = 4 * 37 + 3
        fill(1)
        let expected = op.layers.map { ls in
            (pools: (0..<(position / 4)).map { bytes(ls.pooledKeys, row: $0) },
             k: (148..<position).map { bytes(ls.kPending, row: $0 & (op.ringCap - 1)) },
             g: (148..<position).map { bytes(ls.gatePending, row: $0 & (op.ringCap - 1)) })
        }
        let data = op.captureIndexerState(position: position)
        XCTAssertEqual(data.count, op.indexerStateBytes(position: position))
        fill(99)
        XCTAssertFalse(op.restoreIndexerState(data.dropLast(), position: position))
        XCTAssertTrue(op.restoreIndexerState(data, position: position))
        for (l, ls) in op.layers.enumerated() {
            XCTAssertEqual((0..<(position / 4)).map { bytes(ls.pooledKeys, row: $0) }, expected[l].pools)
            XCTAssertEqual((148..<position).map { bytes(ls.kPending, row: $0 & (op.ringCap - 1)) }, expected[l].k)
            XCTAssertEqual((148..<position).map { bytes(ls.gatePending, row: $0 & (op.ringCap - 1)) }, expected[l].g)
        }
    }

    func testMLALatentPrepPoolKeysAndSparseAttention() throws {
        let geo = try Self.mlaGeometry()
        XCTAssertTrue(geo.descriptor.mlaUseNoPE)
        XCTAssertEqual(geo.kvChannels, 256)
        let hal = try QWHal(geometry: geo)

        let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: geo)
        let poolBytes = pageBytes * 8
        let tables = QWKVCache.tableSlotBytes * 2 * 1 * QWKVCache.maxSeqs
        let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: poolBytes + tables, gdnStates: 4096)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: [3], mode: .f16, geometry: geo)
        let seq = try cache.allocate(promptTokens: 64)
        let kvViews = try cache.layerViews(seq, layer: 3)

        let op = try QWMLAAttnOperator(hal: hal, geometry: geo, numAttnLayers: 1, maxContext: 512, maxPrefillChunk: 16)
        let rows = 8

        let latentPtr = (op.region.buffer.contents() + op.mlaLatentScratch.offset)
            .bindMemory(to: UInt16.self, capacity: rows * op.latentStride)
        for i in 0..<(rows * op.latentStride) {
            latentPtr[i] = QWQuant.f32tof16Bits(sin(Float(i % 251) * 0.05) * 0.3)
        }
        let qFullPtr = (op.region.buffer.contents() + op.mlaQFullScratch.offset)
            .bindMemory(to: UInt16.self, capacity: rows * op.qFullStride)
        for i in 0..<(rows * op.qFullStride) {
            qFullPtr[i] = QWQuant.f32tof16Bits(cos(Float(i % 257) * 0.04) * 0.25)
        }

        let ln1 = makeTensor(hal: hal, f16Values: (0..<(rows * geo.hidden)).map { sin(Float($0 % 199) * 0.03) * 0.2 })
        let qANormW = makeTensor(hal: hal, f16Values: [Float](repeating: 1.0, count: geo.qLoraRank))
        let kvANormW = makeTensor(hal: hal, f16Values: [Float](repeating: 1.0, count: geo.kvLoraRank))
        let kNormW = makeTensor(hal: hal, f16Values: [Float](repeating: 1.0, count: geo.indexerHeadDim))
        let kNormB = makeTensor(hal: hal, f16Values: [Float](repeating: 0.0, count: geo.indexerHeadDim))
        let wGateW = makeTensor(hal: hal, f16Values: (0..<((geo.indexerHeads + 4) * geo.hidden)).map { sin(Float($0 % 97) * 0.02) * 0.05 })
        let ape = makeTensor(hal: hal, f16Values: (0..<(4 * geo.indexerHeadDim)).map { cos(Float($0) * 0.05) * 0.1 })

        // Create non-zero HQ4_G32 blocks for kvBProj [64 * (192 + 256), 512] = [28672, 512]
        let kvBRows = geo.qHeads * (geo.qkNopeHeadDim + geo.vHeadDim)
        let kvBGroups = kvBRows * (geo.kvLoraRank / 32)
        var kvBWords = [UInt16](repeating: 0, count: kvBGroups * 10)
        for g in 0..<kvBGroups {
            kvBWords[g * 10 + 0] = QWQuant.f32tof16Bits(0.02) // d
            kvBWords[g * 10 + 1] = QWQuant.f32tof16Bits(-0.15) // m
            for w in 2..<10 {
                kvBWords[g * 10 + w] = UInt16((g + w) & 0xFFFF)
            }
        }
        let kvBBuf = hal.device.makeBuffer(bytes: kvBWords, length: kvBWords.count * 2, options: .storageModeShared)!
        let kvBTV = QWTensorValue(region: QWMRegion(buffer: kvBBuf, base: 0, length: kvBBuf.length),
                                  offset: 0, shape: QWTensorShape([kvBGroups * 10], .f16))
        let kvBView = try XCTUnwrap(QWWeightView(blocks: kvBTV, n: kvBRows, k: geo.kvLoraRank,
                                                 bits: 4, groupSize: 32, isRotated: true))

        let mlaW = QWMLAAttnOperator.LayerWeights(
            qANormW: qANormW,
            kvANormW: kvANormW,
            kNormW: kNormW,
            kNormB: kNormB,
            wGateWeights: wGateW,
            ape: ape,
            qBProj: kvBView,
            kvBProj: kvBView
        )

        let attOut = makeTensor(hal: hal, f16Values: [Float](repeating: 0, count: rows * geo.qHeads * geo.vHeadDim))

        guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
        op.encodeLatentPrepAndPool(enc, ln1: ln1, mlaW: mlaW, kv: kvViews, fullAttnLayer: 0, pos0: 0, rows: rows)
        op.encodeScoreSelectAbsorbAttendAndProjUV(enc, mlaW: mlaW, kv: kvViews, attOut: attOut, fullAttnLayer: 0, pos0: 0, rows: rows)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let countsPtr = (op.region.buffer.contents() + op.blockCountScratch.offset)
            .bindMemory(to: Int32.self, capacity: rows)
        // Rows 0..3 have 0 completed 4-token blocks before/at them, rows 3..6 have 1, row 7 has 2
        XCTAssertEqual(countsPtr[3], 1)
        XCTAssertEqual(countsPtr[7], 2)

        let outPtr = attOut.region.buffer.contents().bindMemory(to: UInt16.self, capacity: rows * geo.qHeads * geo.vHeadDim)
        var nonZero = 0
        for i in 0..<(rows * geo.qHeads * geo.vHeadDim) {
            let v = QWQuant.f16tof32Bits(outPtr[i])
            XCTAssertTrue(v.isFinite)
            if abs(v) > 1e-6 { nonZero += 1 }
        }
        XCTAssertGreaterThan(nonZero, (rows * geo.qHeads * geo.vHeadDim) / 2)
    }

    /// The MMA and NAX sparse-attention kernels must match the scalar kernel on the f16 latent pool,
    /// covering the all-pools path, the selected-list path and ragged causal tails.
    func testSparseAttnMMAMatchesScalarKernel() throws {
        let geo = try Self.mlaGeometry()
        let hal = try QWHal(geometry: geo)
        let scalar = try hal.pipeline("qwg_mla_sparse_attn")
        let mma = try hal.pipeline("qwg_mla_sparse_attn_mma")
        // nSplit: 0 = per-head scalar grid, 1 = 16-head MMA grid, >1 = split-K NAX + combine.
        var variants = [(scalar, 0), (mma, 1)]
        let combine = try hal.pipeline("qwg_mla_sparse_attn_combine")
        if hal.caps.isMetal4TensorCapable {
            let split = try hal.pipeline("qwg_mla_sparse_attn_nax_split")
            variants += [(try hal.pipeline("qwg_mla_sparse_attn_nax"), 1), (split, 7), (split, 16)]
        }
        let qHeads = 64, rank = 512, halfRank = 256, pages = 6
        var rng = SystemRandomNumberGenerator()
        func rnd(_ n: Int, _ s: Float) -> [UInt16] {
            (0..<n).map { _ in QWQuant.f32tof16Bits(Float.random(in: -s...s, using: &rng)) }
        }
        func buf<T: BitwiseCopyable>(_ a: [T]) -> any MTLBuffer {
            Self.makeBuffer(a, device: hal.device)
        }
        let kPool = buf(rnd(pages * 256 * halfRank, 1.0))
        let vPool = buf(rnd(pages * 256 * halfRank, 1.0))
        let kTable = buf([Int32]([3, 0, 5, 1, 4, 2]))
        let vTable = buf([Int32]([1, 4, 2, 5, 0, 3]))

        // (pos0, T, topKPools): all-pools, selected-list with ragged tails, long context.
        for (pos0, rows, topK) in [(0, 37, 512), (60, 9, 512), (200, 9, 512), (500, 9, 512), (500, 9, 16), (1300, 5, 64)] {
            let q = buf(rnd(rows * qHeads * rank, 0.5))
            var list = [Int32](repeating: 0, count: rows * topK)
            for t in 0..<rows {
                let nPools = (pos0 + t + 1) >> 2
                let picks = Array((0..<max(nPools, 1)).shuffled(using: &rng).prefix(topK))
                for (i, v) in picks.enumerated() { list[t * topK + i] = Int32(v) }
            }
            let blockList = buf(list)
            var params: [UInt32] = [UInt32(rows), UInt32(pos0), UInt32(qHeads), 192, UInt32(rank),
                                    UInt32(topK), UInt32(topK), 0]
            var outs: [[UInt16]] = []
            for (pso, nSplit) in variants {
                let out = hal.device.makeBuffer(length: rows * qHeads * rank * 2, options: .storageModeShared)!
                let partO = hal.device.makeBuffer(length: rows * max(1, nSplit) * qHeads * rank * 2, options: .storageModeShared)!
                let partML = hal.device.makeBuffer(length: rows * max(1, nSplit) * qHeads * 8, options: .storageModeShared)!
                guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
                enc.setComputePipelineState(pso)
                for (i, b) in [q, kPool, vPool, kTable, vTable, blockList, nSplit > 1 ? partO : out].enumerated() {
                    enc.setBuffer(b, offset: 0, index: i)
                }
                enc.setBytes(&params, length: params.count * 4, index: 7)
                if nSplit > 1 {
                    enc.setBuffer(partML, offset: 0, index: 8)
                    enc.dispatchThreadgroups(MTLSize(width: rows, height: qHeads / 16, depth: nSplit),
                                             threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                    var n = UInt32(nSplit)
                    enc.setComputePipelineState(combine)
                    for (i, b) in [partO, partML, out].enumerated() { enc.setBuffer(b, offset: 0, index: i) }
                    enc.setBytes(&params, length: params.count * 4, index: 3)
                    enc.setBytes(&n, length: 4, index: 4)
                    enc.dispatchThreadgroups(MTLSize(width: rows * qHeads, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                } else if nSplit == 1 {
                    enc.dispatchThreadgroups(MTLSize(width: rows, height: qHeads / 16, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                } else {
                    enc.dispatchThreadgroups(MTLSize(width: qHeads * rows, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
                hal.end(cb, enc)
                cb.waitUntilCompleted()
                XCTAssertNil(cb.error)
                let ptr = out.contents().bindMemory(to: UInt16.self, capacity: rows * qHeads * rank)
                outs.append(Array(UnsafeBufferPointer(start: ptr, count: rows * qHeads * rank)))
            }
            // CPU reference on a subset of heads.
            let qp = q.contents().bindMemory(to: UInt16.self, capacity: rows * qHeads * rank)
            let kp = kPool.contents().bindMemory(to: UInt16.self, capacity: pages * 256 * halfRank)
            let vp = vPool.contents().bindMemory(to: UInt16.self, capacity: pages * 256 * halfRank)
            let kt: [Int] = [3, 0, 5, 1, 4, 2], vt: [Int] = [1, 4, 2, 5, 0, 3]
            func latent(_ j: Int, _ c: Int) -> Float {
                let off = j & 255
                return c < halfRank ? QWQuant.f16tof32Bits(kp[(kt[j >> 8] * 256 + off) * halfRank + c])
                                    : QWQuant.f16tof32Bits(vp[(vt[j >> 8] * 256 + off) * halfRank + c - halfRank])
            }
            var worst = 0, worstErr: Float = 0
            for i in 0..<outs[0].count {
                let e = abs(QWQuant.f16tof32Bits(outs[0][i]) - QWQuant.f16tof32Bits(outs[1][i]))
                if e > worstErr { worstErr = e; worst = i }
            }
            let worstHead = (worst / rank) % qHeads
            let heads = Set([0, 21, 63, worstHead]).sorted()
            var cpuErr = [Float](repeating: 0, count: outs.count)
            for t in 0..<rows {
                let pq = pos0 + t, nPools = (pq + 1) >> 2, cnt = min(topK, nPools)
                var toks: [Int] = []
                for b in 0..<cnt {
                    let base = (nPools <= topK ? b : Int(list[t * topK + b])) * 4
                    toks += [base, base + 1, base + 2, base + 3]
                }
                toks += Array((nPools * 4)..<(pq + 1))
                for h in heads {
                    let qo = (t * qHeads + h) * rank
                    let sc = toks.map { j in (0..<rank).reduce(Float(0)) { $0 + QWQuant.f16tof32Bits(qp[qo + $1]) * latent(j, $1) } / Float(192).squareRoot() }
                    let m = sc.max()!
                    let w = sc.map { exp($0 - m) }
                    let l = w.reduce(0, +)
                    for c in stride(from: 0, to: rank, by: 7) {
                        var ref: Float = 0
                        for (i, j) in toks.enumerated() { ref += w[i] * latent(j, c) }
                        ref /= l
                        for k in 0..<outs.count { cpuErr[k] = max(cpuErr[k], abs(QWQuant.f16tof32Bits(outs[k][qo + c]) - ref)) }
                    }
                }
            }
            XCTAssertLessThan(cpuErr[0], 0.01, "scalar vs cpu pos0=\(pos0)")
            for k in 1..<outs.count {
                XCTAssertLessThan(cpuErr[k], 0.01, "variant \(k) vs cpu pos0=\(pos0)")
                var maxErr: Float = 0, maxRef: Float = 0
                for i in 0..<outs[0].count {
                    let a = QWQuant.f16tof32Bits(outs[0][i]), b = QWQuant.f16tof32Bits(outs[k][i])
                    XCTAssertTrue(b.isFinite)
                    maxErr = max(maxErr, abs(a - b))
                    maxRef = max(maxRef, abs(a))
                }
                XCTAssertGreaterThan(maxRef, 0.05, "pos0=\(pos0)")
                XCTAssertLessThan(maxErr, 0.01 * max(maxRef, 1), "variant \(k) pos0=\(pos0) T=\(rows) topK=\(topK)")
            }
        }
    }

    /// The MMA absorb kernel must match the scalar HQ4_G32 path (nope 256, rank 512).
    func testAbsorbQMMAMatchesScalarKernel() throws {
        let hal = try QWHal(geometry: try Self.mlaGeometry())
        let scalar = try hal.pipeline("qwg_mla_absorb_q")
        let mma = try hal.pipeline("qwg_mla_absorb_q_mma")
        let qHeads = 64, nope = 256, vDim = 256, rank = 512, qFullStride = qHeads * nope + 64
        var rng = SystemRandomNumberGenerator()
        // HQ4_G32 block: half scale, half bias, 4 x u32 nibbles (10 x u16).
        let nBlocks = qHeads * (nope + vDim) * (rank / 32)
        var words = [UInt16](repeating: 0, count: nBlocks * 10)
        for b in 0..<nBlocks {
            let scale = Float.random(in: 0.002...0.02, using: &rng)
            words[b * 10] = QWQuant.f32tof16Bits(scale)
            words[b * 10 + 1] = QWQuant.f32tof16Bits(-1024 * scale - 8 * scale)
            for w in 2..<10 { words[b * 10 + w] = UInt16.random(in: 0...UInt16.max, using: &rng) }
        }
        let wBuf = hal.device.makeBuffer(bytes: words, length: words.count * 2, options: .storageModeShared)!
        for rows in [16, 45, 100] {
            let qVals = (0..<(rows * qFullStride)).map { _ in QWQuant.f32tof16Bits(Float.random(in: -1...1, using: &rng)) }
            let qBuf = hal.device.makeBuffer(bytes: qVals, length: qVals.count * 2, options: .storageModeShared)!
            var params: [UInt32] = [UInt32(rows), UInt32(qHeads), UInt32(nope), UInt32(vDim), UInt32(rank),
                                    UInt32(qFullStride), 4, 32, 1]
            var outs: [[UInt16]] = []
            for useMMA in [false, true] {
                let out = hal.device.makeBuffer(length: rows * qHeads * rank * 2, options: .storageModeShared)!
                guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
                enc.setComputePipelineState(useMMA ? mma : scalar)
                enc.setBuffer(qBuf, offset: 0, index: 0)
                enc.setBuffer(wBuf, offset: 0, index: 1)
                enc.setBuffer(out, offset: 0, index: 2)
                enc.setBytes(&params, length: params.count * 4, index: 3)
                if useMMA {
                    enc.dispatchThreadgroups(MTLSize(width: qHeads, height: (rows + 31) / 32, depth: rank / 128),
                                             threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                } else {
                    enc.dispatchThreadgroups(MTLSize(width: qHeads, height: (rows + 7) / 8, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
                }
                hal.end(cb, enc)
                cb.waitUntilCompleted()
                XCTAssertNil(cb.error)
                let ptr = out.contents().bindMemory(to: UInt16.self, capacity: rows * qHeads * rank)
                outs.append(Array(UnsafeBufferPointer(start: ptr, count: rows * qHeads * rank)))
            }
            var maxErr: Float = 0, maxRef: Float = 0
            for i in 0..<outs[0].count {
                let a = QWQuant.f16tof32Bits(outs[0][i]), b = QWQuant.f16tof32Bits(outs[1][i])
                maxErr = max(maxErr, abs(a - b))
                maxRef = max(maxRef, abs(a))
            }
            XCTAssertGreaterThan(maxRef, 0.1, "rows=\(rows)")
            XCTAssertLessThan(maxErr, 2e-3 * maxRef, "rows=\(rows)")
        }
    }

    /// The MMA value up-projection must match the scalar HQ4_G32 path, including the H_32 rotation.
    func testProjUVMMAMatchesScalarKernel() throws {
        let hal = try QWHal(geometry: try Self.mlaGeometry())
        let scalar = try hal.pipeline("qwg_mla_proj_uv")
        let mma = try hal.pipeline("qwg_mla_proj_uv_mma")
        let qHeads = 64, nope = 256, vDim = 256, rank = 512
        var rng = SystemRandomNumberGenerator()
        let nBlocks = qHeads * (nope + vDim) * (rank / 32)
        var words = [UInt16](repeating: 0, count: nBlocks * 10)
        for b in 0..<nBlocks {
            let scale = Float.random(in: 0.002...0.02, using: &rng)
            words[b * 10] = QWQuant.f32tof16Bits(scale)
            words[b * 10 + 1] = QWQuant.f32tof16Bits(-1024 * scale - 8 * scale)
            for w in 2..<10 { words[b * 10 + w] = UInt16.random(in: 0...UInt16.max, using: &rng) }
        }
        let wBuf = hal.device.makeBuffer(bytes: words, length: words.count * 2, options: .storageModeShared)!
        for rows in [16, 45, 100] {
            let uVals = (0..<(rows * qHeads * rank)).map { _ in QWQuant.f32tof16Bits(Float.random(in: -1...1, using: &rng)) }
            let uBuf = hal.device.makeBuffer(bytes: uVals, length: uVals.count * 2, options: .storageModeShared)!
            var params: [UInt32] = [UInt32(rows), UInt32(qHeads), UInt32(nope), UInt32(vDim), UInt32(rank),
                                    UInt32(qHeads * nope), 4, 32, 1]
            let count = rows * qHeads * vDim
            var outs: [[UInt16]] = []
            for useMMA in [false, true] {
                let out = hal.device.makeBuffer(length: count * 2, options: .storageModeShared)!
                guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
                enc.setComputePipelineState(useMMA ? mma : scalar)
                enc.setBuffer(uBuf, offset: 0, index: 0)
                enc.setBuffer(wBuf, offset: 0, index: 1)
                enc.setBuffer(out, offset: 0, index: 2)
                enc.setBytes(&params, length: params.count * 4, index: 3)
                if useMMA {
                    enc.dispatchThreadgroups(MTLSize(width: qHeads, height: (rows + 31) / 32, depth: vDim / 128),
                                             threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                } else {
                    enc.dispatchThreadgroups(MTLSize(width: qHeads, height: (rows + 7) / 8, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
                hal.end(cb, enc)
                cb.waitUntilCompleted()
                XCTAssertNil(cb.error)
                let ptr = out.contents().bindMemory(to: UInt16.self, capacity: count)
                outs.append(Array(UnsafeBufferPointer(start: ptr, count: count)))
            }
            var maxErr: Float = 0, maxRef: Float = 0
            for i in 0..<count {
                let a = QWQuant.f16tof32Bits(outs[0][i]), b = QWQuant.f16tof32Bits(outs[1][i])
                maxErr = max(maxErr, abs(a - b))
                maxRef = max(maxRef, abs(a))
            }
            XCTAssertGreaterThan(maxRef, 0.1, "rows=\(rows)")
            XCTAssertLessThan(maxErr, 2e-3 * maxRef, "rows=\(rows)")
        }
    }

    func testSparseAttnNAXH32BitExactMatchesH16AcrossKVModes() throws {
        let hal = try QWHal(geometry: try Self.mlaGeometry())
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable, "Metal 4 NAX tensor required")
        let pMLAH16 = try hal.pipeline("qwg_mla_sparse_attn_nax")
        let pMLAH32 = try hal.pipeline("qwg_mla_sparse_attn_nax_h32")
        let rows = 33, pos0 = 260, qHeads = 64, headDim = 192, rank = 512, nPages = 2
        let halfRank = rank / 2
        var s: UInt64 = 0x5EED_9876_1234_ABCD
        func nextF(_ scale: Float) -> Float {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: s >> 33)) / Float(Int32.max) * scale
        }
        let qAbsBuf = Self.makeBuffer((0..<(rows * qHeads * rank)).map { _ in QWQuant.f32tof16Bits(nextF(0.3)) }, device: hal.device)
        let tableBuf = Self.makeBuffer([Int32(0), Int32(1)], device: hal.device)
        let poolStride = 32
        let mlaBlk = Self.makeBuffer((0..<(rows * poolStride)).map { Int32($0 % 60) }, device: hal.device)
        func makePool(mode: Int) -> any MTLBuffer {
            let pBytes = mode == 0 ? 256 * halfRank * 2 : (mode == 1 ? 66048 : 34816)
            var bytes = [UInt8](repeating: 0, count: nPages * pBytes)
            for p in 0..<nPages {
                let base = p * pBytes
                if mode == 0 {
                    for i in 0..<(256 * halfRank) {
                        let w = QWQuant.f32tof16Bits(nextF(0.35))
                        bytes[base + i * 2] = UInt8(w & 0xFF); bytes[base + i * 2 + 1] = UInt8(w >> 8)
                    }
                } else if mode == 1 {
                    for i in 0..<65536 { s = s &* 1664525 &+ 1013904223; bytes[base + i] = UInt8((s >> 16) & 0xFF) }
                    for r in 0..<256 {
                        let w = QWQuant.f32tof16Bits(0.015)
                        bytes[base + 65536 + r * 2] = UInt8(w & 0xFF); bytes[base + 65536 + r * 2 + 1] = UInt8(w >> 8)
                    }
                } else {
                    for i in 0..<32768 { s = s &* 1664525 &+ 1013904223; bytes[base + i] = UInt8((s >> 16) & 0xFF) }
                    for r in 0..<512 {
                        let sc = QWQuant.f32tof16Bits(0.03125), bi = QWQuant.f32tof16Bits(-0.25)
                        let off = base + 32768 + r * 4
                        bytes[off] = UInt8(sc & 0xFF); bytes[off + 1] = UInt8(sc >> 8)
                        bytes[off + 2] = UInt8(bi & 0xFF); bytes[off + 3] = UInt8(bi >> 8)
                    }
                }
            }
            return Self.makeBuffer(bytes, device: hal.device)
        }
        for kvMode in 0...2 {
            let kPool = makePool(mode: kvMode), vPool = makePool(mode: kvMode)
            let outCount = rows * qHeads * halfRank
            var mlaP = QWMLAAttnOperator.SparseAttnParams(
                T: UInt32(rows), pos0: UInt32(pos0), qHeads: UInt32(qHeads), headDim: UInt32(headDim),
                kvLoraRank: UInt32(rank), topKPools: UInt32(poolStride), listStride: UInt32(poolStride),
                kvMode: UInt32(kvMode))
            func runMLA(_ pso: any MTLComputePipelineState, grid: MTLSize, tg: Int) -> [UInt16] {
                let out = hal.device.makeBuffer(length: outCount * 2, options: .storageModeShared)!
                guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return [] }
                enc.setComputePipelineState(pso)
                for (i, b) in [qAbsBuf, kPool, vPool, tableBuf, tableBuf, mlaBlk, out].enumerated() { enc.setBuffer(b, offset: 0, index: i) }
                enc.setBytes(&mlaP, length: MemoryLayout<QWMLAAttnOperator.SparseAttnParams>.stride, index: 7)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
                hal.end(cb, enc); cb.waitUntilCompleted(); XCTAssertNil(cb.error)
                let p = out.contents().bindMemory(to: UInt16.self, capacity: outCount)
                return Array(UnsafeBufferPointer(start: p, count: outCount))
            }
            let mlaRef = runMLA(pMLAH16, grid: MTLSize(width: rows, height: qHeads / 16, depth: 1), tg: 128)
            let mlaH32 = runMLA(pMLAH32, grid: MTLSize(width: qHeads / 32, height: rows, depth: 1), tg: 256)
            XCTAssertEqual(mlaH32, mlaRef, "MLA h32 vs h16 must be bit-exact kvMode=\(kvMode)")
        }
    }

    func testLiveSessionStashPreservesFullRingAndMTPLayerAcrossMod4AndDSA() throws {
        let geo = try Self.mlaGeometry()
        let hal = try QWHal(geometry: geo)
        let op = try QWMLAAttnOperator(hal: hal, geometry: geo, numAttnLayers: 3, maxContext: 512, maxPrefillChunk: 16)
        let base = op.region.pointer

        func fillDeterministic(seed: UInt8) {
            for (idx, ls) in op.layers.enumerated() {
                for (tIdx, t) in [ls.pooledKeys, ls.kPending, ls.gatePending].enumerated() {
                    let p = base.advanced(by: t.offset).assumingMemoryBound(to: UInt8.self)
                    for i in 0..<t.bytes {
                        p[i] = UInt8(truncatingIfNeeded: i &* 17 &+ Int(seed) &+ idx &* 31 &+ tIdx * 7)
                    }
                }
            }
        }
        func snapshotBytes(_ t: QWTensorValue, byteCount: Int) -> [UInt8] {
            let p = base.advanced(by: t.offset).assumingMemoryBound(to: UInt8.self)
            return Array(UnsafeBufferPointer(start: p, count: byteCount))
        }

        let rowBytes = op.idxDim * 2
        let ringBytes = op.ringCap * rowBytes
        for (trunkPos, mtpPos) in [(16, 11), (19, 14)] {
            fillDeterministic(seed: UInt8(trunkPos))
            let expected = op.layers.enumerated().map { (i, ls) -> ([UInt8], [UInt8], [UInt8]) in
                let p = (i == 2) ? mtpPos : trunkPos
                return (snapshotBytes(ls.pooledKeys, byteCount: (p / 4) * rowBytes),
                        snapshotBytes(ls.kPending, byteCount: ringBytes),
                        snapshotBytes(ls.gatePending, byteCount: ringBytes))
            }
            let stash = op.captureLiveSessionStash(trunkPosition: trunkPos, mtpLayerIndex: 2, mtpPosition: mtpPos)
            XCTAssertEqual(stash.count, op.liveSessionStashBytes(trunkPosition: trunkPos, mtpLayerIndex: 2, mtpPosition: mtpPos))
            fillDeterministic(seed: 0xAA)
            XCTAssertTrue(op.restoreLiveSessionStash(stash, trunkPosition: trunkPos, mtpLayerIndex: 2, mtpPosition: mtpPos))
            for (i, ls) in op.layers.enumerated() {
                let p = (i == 2) ? mtpPos : trunkPos
                XCTAssertEqual(snapshotBytes(ls.pooledKeys, byteCount: (p / 4) * rowBytes), expected[i].0)
                XCTAssertEqual(snapshotBytes(ls.kPending, byteCount: ringBytes), expected[i].1)
                XCTAssertEqual(snapshotBytes(ls.gatePending, byteCount: ringBytes), expected[i].2)
            }
        }

        // Also check decoupled DSA live session stash (including kPeCache + mtpLayerIndex).
        let dsaCfg = """
        {
          "architectures": ["Ernie5_5_MoeForCausalLM"], "model_type": "ernie5_5_moe",
          "vocab_size": 1024, "hidden_size": 128, "intermediate_size": 128,
          "num_hidden_layers": 2, "max_position_embeddings": 1048576, "rms_norm_eps": 1e-6,
          "num_attention_heads": 16, "num_key_value_heads": 1, "head_dim": 512,
          "compress_ratios": [4, -2], "default_compress_ratio": 4,
          "q_lora_rank": 128, "kv_lora_rank": 512,
          "compress_qk_nope_head_dim": 448, "compress_qk_rope_head_dim": 64, "compress_v_head_dim": 512,
          "o_groups": 2, "o_lora_rank": 128, "sliding_window": 128,
          "compress_rope_theta": 160000.0, "hca_rope_type": "yarn",
          "rope_scaling": { "type": "yarn", "factor": 16, "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1 },
          "rope_theta": 8000000.0, "rotary_percent": 1.0, "rope_interleave": true,
          "hybrid_mla_attention": "mqa_dsa", "hybrid_mla_num_attention_heads": 16, "hybrid_mla_num_key_value_heads": 16,
          "hybrid_mla_q_lora_rank": 64, "hybrid_mla_kv_lora_rank": 512,
          "hybrid_mla_qk_nope_head_dim": 192, "hybrid_mla_qk_rope_head_dim": 64, "hybrid_mla_v_head_dim": 256,
          "mqa_split_kv_b_proj": true, "index_n_heads": 64, "index_head_dim": 128, "index_topk": 256,
          "use_fast_hadamard": true, "use_vha_attention": true, "add_full_attention_sink_bias": true,
          "use_gated_attn": true, "use_attn_merger": true, "gated_attn_use_q_lora": true, "eos_token_id": 2
        }
        """.data(using: .utf8)!
        let dsaGeo = QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: dsaCfg))
        let dsaOp = try QWMLAAttnOperator(hal: hal, geometry: dsaGeo, numAttnLayers: 3, maxContext: 512, maxPrefillChunk: 32)
        XCTAssertTrue(dsaOp.isDecoupledDSA)
        let dsaBase = dsaOp.region.pointer
        for (i, ls) in dsaOp.layers.enumerated() {
            for t in [ls.pooledKeys, ls.kPeCache].compactMap({ $0 }) {
                let p = dsaBase.advanced(by: t.offset).assumingMemoryBound(to: UInt8.self)
                for j in 0..<t.bytes { p[j] = UInt8(truncatingIfNeeded: (j + i * 13) & 0xFF) }
            }
        }
        let dsaStash = dsaOp.captureLiveSessionStash(trunkPosition: 16, mtpLayerIndex: 2, mtpPosition: 10)
        let expectedPools = (0..<3).map { i -> [UInt8] in
            let pos = (i == 2) ? 10 : 16
            let p = dsaBase.advanced(by: dsaOp.layers[i].pooledKeys.offset).assumingMemoryBound(to: UInt8.self)
            return Array(UnsafeBufferPointer(start: p, count: pos * dsaOp.idxDim * 2))
        }
        for ls in dsaOp.layers {
            for t in [ls.pooledKeys, ls.kPeCache].compactMap({ $0 }) {
                memset(dsaBase.advanced(by: t.offset), 0, t.bytes)
            }
        }
        XCTAssertTrue(dsaOp.restoreLiveSessionStash(dsaStash, trunkPosition: 16, mtpLayerIndex: 2, mtpPosition: 10))
        for i in 0..<3 {
            let pos = (i == 2) ? 10 : 16
            let p = dsaBase.advanced(by: dsaOp.layers[i].pooledKeys.offset).assumingMemoryBound(to: UInt8.self)
            XCTAssertEqual(Array(UnsafeBufferPointer(start: p, count: pos * dsaOp.idxDim * 2)), expectedPools[i])
        }
    }
}
