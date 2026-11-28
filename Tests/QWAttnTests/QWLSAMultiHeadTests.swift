import Foundation
import Metal
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc
import QWQuant
import XCTest

/// Verifies Full-Slice Q4 multi-head direct LSA kernels, CPU parity, clipping, snapshots, and fallback.
final class QWLSAMultiHeadTests: XCTestCase {
    private static func makeMHGeometry(
        qHeads: Int = 24, kvHeads: Int = 8, headDim: Int = 128,
        partialRotary: Double = 1.0, inklingClip: Float? = nil
    ) throws -> QWModelGeometry {
        var textDict: [String: Any] = [
            "model_type": inklingClip != nil ? "inkling2_loop_text" : "qwen3",
            "hidden_size": 256,
            "intermediate_size": 512,
            "vocab_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": qHeads,
            "num_key_value_heads": kvHeads,
            "head_dim": headDim,
            "partial_rotary_factor": partialRotary,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 65536,
            "rope_theta": 1000000.0,
            "eos_token_id": 1,
        ]
        var rootDict: [String: Any] = textDict
        if let clip = inklingClip {
            textDict["vocab_size"] = 256
            textDict["unpadded_vocab_size"] = 256
            textDict["n_routed_experts"] = 4
            textDict["num_experts_per_tok"] = 2
            textDict["sconv_kernel_size"] = 4
            textDict["d_rel"] = 16
            textDict["rel_extent"] = 1024
            textDict["log_scaling_n_floor"] = 8192
            textDict["log_scaling_alpha"] = 0.1
            textDict["use_value_norm_no_gamma"] = true
            textDict["qk_l2_clip_threshold"] = Double(clip)
            textDict["rel_l2_clip_threshold"] = 8.0
            textDict["local_layer_ids"] = [0]
            textDict["sliding_window_size"] = 1024
            textDict["n_layers_per_attn_res_block"] = 2
            textDict["attn_res_normalize_values"] = true
            textDict["final_logit_softcapping"] = 30.0
            textDict["mlp_gate_softcap"] = 4.0
            textDict["mlp_up_softcap"] = 25.0
            textDict["route_scale"] = 8.0
            textDict["loop_config"] = [
                "core_start": 0,
                "core_end": 2,
                "num_passes": 1,
                "gate_rank": 64,
                "is_varloop": false,
            ]
            rootDict = ["model_type": "inkling2_loop", "text_config": textDict]
        }
        let data = try JSONSerialization.data(withJSONObject: rootDict)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    private static func f32ToBF16(_ v: Float) -> UInt16 {
        var bits = v.bitPattern
        if (bits & 0x7F80_0000) == 0x7F80_0000 { return UInt16(truncatingIfNeeded: bits >> 16) }
        bits = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: bits >> 16)
    }

    private static func bf16ToF32(_ b: UInt16) -> Float { Float(bitPattern: UInt32(b) << 16) }

    private static func makeBF16Buffer(_ hal: QWHal, _ vals: [Float], label: String) -> QWTensorValue {
        let bf16 = vals.map(f32ToBF16)
        let buf = hal.device.makeBuffer(bytes: bf16, length: bf16.count * 2, options: .storageModeShared)!
        buf.label = label
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length),
                             offset: 0, shape: QWTensorShape([bf16.count], .bf16))
    }

    private static func makeFP16Buffer(_ hal: QWHal, _ vals: [Float], label: String) -> QWTensorValue {
        let f16 = vals.map { Float16($0).bitPattern }
        let buf = hal.device.makeBuffer(bytes: f16, length: f16.count * 2, options: .storageModeShared)!
        buf.label = label
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length),
                             offset: 0, shape: QWTensorShape([f16.count], .f16))
    }

    private static func readRowBlocks(_ op: QWLSAOperator, row: Int) -> [Int] {
        let base = op.region.pointer
        let cntPtr = base.advanced(by: op.blockCountScratch.offset).assumingMemoryBound(to: UInt32.self)
        let cnt = Int(cntPtr[row])
        let listPtr = base.advanced(by: op.blockListScratch.offset).assumingMemoryBound(to: Int32.self)
        return (0..<cnt).map { Int(listPtr[row * op.listStride + $0]) }
    }

    private struct Q4Block {
        let mnH: Float
        let scH: Float
        let nibs: [Float]
    }

    private static func quantizeQ4Block(_ mean: UnsafePointer<Float>, count: Int) -> Q4Block {
        var mnF: Float = .greatestFiniteMagnitude, mxF: Float = -.greatestFiniteMagnitude
        var i = 0
        while i < count {
            let v = mean[i]
            if v < mnF { mnF = v }
            if v > mxF { mxF = v }
            i += 1
        }
        if count == 0 { mnF = 0; mxF = 0 }
        let mnH = Float(Float16(mnF))
        let scH = Float(Float16(max(0, mxF - mnF) / 15.0))
        let invSc: Float = scH > 0 ? (1.0 / scH) : 0
        let nibs = [Float](unsafeUninitializedCapacity: count) { dst, initCnt in
            guard let dp = dst.baseAddress else { initCnt = 0; return }
            var d = 0
            while d < count {
                let qv = Foundation.rint(Double((mean[d] - mnH) * invSc))
                dp[d] = Float(min(15.0, max(0.0, qv)))
                d += 1
            }
            initCnt = count
        }
        return Q4Block(mnH: mnH, scH: scH, nibs: nibs)
    }

    private static func buildPooledQ4Blocks(clippedK: [[[Float]]], totalBlocks: Int) -> [[Q4Block]] {
        let kvHeads = clippedK[0].count
        let sliceDim = clippedK[0][0].count
        var out = [[Q4Block]]()
        out.reserveCapacity(totalBlocks)
        withUnsafeTemporaryAllocation(of: Float.self, capacity: sliceDim) { mTmp in
            guard let mp = mTmp.baseAddress else { return }
            var blk = 0
            while blk < totalBlocks {
                let k0 = clippedK[blk * 4], k1 = clippedK[blk * 4 + 1], k2 = clippedK[blk * 4 + 2], k3 = clippedK[blk * 4 + 3]
                var heads = [Q4Block]()
                heads.reserveCapacity(kvHeads)
                var g = 0
                while g < kvHeads {
                    k0[g].withUnsafeBufferPointer { b0 in
                        k1[g].withUnsafeBufferPointer { b1 in
                            k2[g].withUnsafeBufferPointer { b2 in
                                k3[g].withUnsafeBufferPointer { b3 in
                                    guard let p0 = b0.baseAddress, let p1 = b1.baseAddress,
                                          let p2 = b2.baseAddress, let p3 = b3.baseAddress else { return }
                                    var d = 0
                                    while d < sliceDim {
                                        mp[d] = 0.25 * (p0[d] + p1[d] + p2[d] + p3[d])
                                        d += 1
                                    }
                                    heads.append(quantizeQ4Block(mp, count: sliceDim))
                                }
                            }
                        }
                    }
                    g += 1
                }
                out.append(heads)
                blk += 1
            }
        }
        return out
    }

    private static func buildCoarseQ4Blocks(fineQ4: [[Q4Block]], totalCoarse: Int) -> [[Q4Block]] {
        let kvHeads = fineQ4[0].count
        let sliceDim = fineQ4[0][0].nibs.count
        var out = [[Q4Block]]()
        out.reserveCapacity(totalCoarse)
        withUnsafeTemporaryAllocation(of: Float.self, capacity: sliceDim) { mTmp in
            guard let mp = mTmp.baseAddress else { return }
            var cBlk = 0
            while cBlk < totalCoarse {
                var heads = [Q4Block]()
                heads.reserveCapacity(kvHeads)
                var g = 0
                while g < kvHeads {
                    var d = 0
                    while d < sliceDim { mp[d] = 0; d += 1 }
                    var off = 0
                    while off < 4 {
                        let b = fineQ4[cBlk * 4 + off][g]
                        b.nibs.withUnsafeBufferPointer { nB in
                            guard let np = nB.baseAddress else { return }
                            var j = 0
                            while j < sliceDim { mp[j] += b.mnH + b.scH * np[j]; j += 1 }
                        }
                        off += 1
                    }
                    d = 0
                    while d < sliceDim { mp[d] *= 0.25; d += 1 }
                    heads.append(quantizeQ4Block(mp, count: sliceDim))
                    g += 1
                }
                out.append(heads)
                cBlk += 1
            }
        }
        return out
    }

    private static func rankTopK(
        candIds: [Int], candScores: [[Float]], lseIndices: [Int], sinkRow: [Float]?, qHeads: Int, k: Int
    ) -> [Int] {
        var lse = [Float](repeating: 0, count: qHeads)
        for qh in 0..<qHeads {
            var m: Float = sinkRow?[qh] ?? -1e30
            for idx in lseIndices { m = max(m, candScores[qh][idx]) }
            var s: Float = sinkRow.map { exp($0[qh] - m) } ?? 0
            for idx in lseIndices { s += exp(candScores[qh][idx] - m) }
            lse[qh] = m + log(max(1e-30, s))
        }
        var cands: [(id: Int, u: UInt32)] = []
        cands.reserveCapacity(candIds.count)
        for (i, id) in candIds.enumerated() {
            var pSum: Float = 0
            for qh in 0..<qHeads { pSum += exp(candScores[qh][i] - lse[qh]) }
            let clamped = min(Float(qHeads), max(0, pSum))
            let bits = clamped.bitPattern
            let u16: UInt32 = bits <= 0x3380_0000 ? 0 : min(65535, (bits - 0x3380_0000) >> 12)
            cands.append((id: id, u: u16))
        }
        return cands.sorted { $0.u != $1.u ? $0.u > $1.u : $0.id > $1.id }.prefix(k).map(\.id).sorted()
    }

    private static func cpuReferenceSelectRow(
        qRow: [[Float]],
        clippedK: [[[Float]]],
        pos: Int,
        budgetBlocks: Int = 128,
        windowBlocks: Int = 128,
        sinkRow: [Float]? = nil,
        coarseCap: Int? = nil
    ) -> [Int] {
        let resolvedCoarseCap = coarseCap ?? min(4096, max(2048, budgetBlocks * 2))
        let qHeads = qRow.count
        let kvHeads = clippedK[0].count
        let sliceDim = clippedK[0][0].count
        let gPerKV = qHeads / kvHeads
        let totalBlocks = (pos + 1) / 4
        if totalBlocks <= 1 + budgetBlocks + windowBlocks { return Array(0..<totalBlocks) }
        let middleEnd = totalBlocks - windowBlocks
        let numMid = middleEnd - 1
        let kEff = min(budgetBlocks, numMid)
        let fineQ4 = buildPooledQ4Blocks(clippedK: clippedK, totalBlocks: totalBlocks)
        let qSum = qRow.map { $0.reduce(Float(0), +) }
        let scaleInv = 1.0 / sqrt(Float(sliceDim))

        let qFlat = qRow.flatMap { $0 }
        func evalDeq(poolQ4: [[Q4Block]], blkIds: [Int]) -> [[Float]] {
            let nBlks = blkIds.count
            var out = (0..<qHeads).map { _ in [Float](repeating: 0, count: nBlks) }
            qFlat.withUnsafeBufferPointer { qB in
                guard let qp = qB.baseAddress else { return }
                var i = 0
                while i < nBlks {
                    let blkRow = poolQ4[blkIds[i]]
                    var g = 0
                    while g < kvHeads {
                        let qb = blkRow[g]
                        qb.nibs.withUnsafeBufferPointer { nB in
                            guard let np = nB.baseAddress else { return }
                            var sub = 0
                            while sub < gPerKV {
                                let qh = g * gPerKV + sub
                                let qHead = qp + (qh * sliceDim)
                                var dotN: Float = 0, d = 0
                                while d < sliceDim { dotN += qHead[d] * np[d]; d += 1 }
                                let raw = min(60000.0, max(-60000.0, (qb.scH * dotN + qb.mnH * qSum[qh]) * scaleInv))
                                out[qh][i] = Float(Float16(raw))
                                sub += 1
                            }
                        }
                        g += 1
                    }
                    i += 1
                }
            }
            return out
        }

        let numCoarseMid = (middleEnd + 3) / 4
        if numMid > resolvedCoarseCap * 4 && numCoarseMid > resolvedCoarseCap {
            let totalCoarse = totalBlocks / 4
            let coarseQ4 = buildCoarseQ4Blocks(fineQ4: fineQ4, totalCoarse: totalCoarse)
            let cScores = evalDeq(poolQ4: coarseQ4, blkIds: Array(0..<totalCoarse))
            let topC = rankTopK(
                candIds: Array(0..<numCoarseMid), candScores: cScores,
                lseIndices: Array(0..<totalCoarse), sinkRow: sinkRow, qHeads: qHeads, k: resolvedCoarseCap)
            var fineCands: [Int] = []
            for c in topC {
                for fb in max(1, 4 * c)..<min(middleEnd, 4 * c + 4) { fineCands.append(fb) }
            }
            let s2Items = fineCands + [0] + Array(middleEnd..<totalBlocks)
            let fScores = evalDeq(poolQ4: fineQ4, blkIds: s2Items)
            let picked = rankTopK(
                candIds: fineCands, candScores: fScores,
                lseIndices: Array(0..<s2Items.count), sinkRow: sinkRow, qHeads: qHeads, k: kEff)
            return [0] + picked + Array(middleEnd..<totalBlocks)
        }

        // Put middle items 1..<middleEnd first so candIds index cleanly into allScores.
        let ordered = Array(1..<middleEnd) + [0] + Array(middleEnd..<totalBlocks)
        let allScores = evalDeq(poolQ4: fineQ4, blkIds: ordered)
        let picked = rankTopK(
            candIds: Array(1..<middleEnd), candScores: allScores,
            lseIndices: Array(0..<ordered.count), sinkRow: sinkRow, qHeads: qHeads, k: kEff)
        return [0] + picked + Array(middleEnd..<totalBlocks)
    }

    func testSingleQHeadNeedleAndSnapshotsWithRegionParity() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal device required")
        let hal = QWLSATests.hal
        let geom = try Self.makeMHGeometry(qHeads: 24, kvHeads: 8, headDim: 128, inklingClip: 22.627417)
        let cfg = QWLSAConfig(mode: .svd, budget: 512, clusterBlock: 512)
        let op1 = try XCTUnwrap(QWLSAOperator.makeIfEnabled(
            hal: hal, geometry: geom, lsaConfig: cfg, packPath: "", pack: nil,
            attnLayers: [0], maxContext: 8192, maxPrefillChunk: 512))
        let op2 = try QWLSAOperator(
            hal: hal, geometry: geom, config: cfg, indexDim: 64,
            layerWeights: [.init(physicalLayer: 5, wqIdxF16: [], wkIdxF16: [], isHeadFactorized: true)],
            maxContext: 8192, maxPrefillChunk: 512)

        XCTAssertTrue(op1.isMultiHeadDirect)
        XCTAssertEqual(op1.qIndexStride, 24 * 128)
        XCTAssertEqual(op1.pooledMetaBytes, 32)
        XCTAssertEqual(op1.pooledRowBytes, 544)
        let estBytes = QWLSAOperator.estimateRegionBytes(
            geometry: geom, lsaConfig: cfg, attnLayers: [0],
            maxContext: 8192, maxPrefillChunk: 512, indexDim: 64, isMultiHeadDirect: true)
        XCTAssertEqual(op1.region.length, estBytes)

        let totalTok = 4608
        var qAll = [Float](repeating: 0, count: totalTok * 24 * 128)
        var kAll = [Float](repeating: 0, count: totalTok * 8 * 128)
        for t in 0..<totalTok {
            for g in 0..<8 {
                let d0 = (t * 7 + g * 3) % 120
                kAll[(t * 8 + g) * 128 + d0] = 1.0
                kAll[(t * 8 + g) * 128 + d0 + 1] = 0.25
            }
            for qh in 0..<24 { qAll[(t * 24 + qh) * 128 + ((t + qh) % 12)] = 1.0 }
        }
        let bStar = 150
        for tok in (bStar * 4)..<(bStar * 4 + 4) { kAll[(tok * 8 + 5) * 128 + 127] = 18.0 }
        for qTok in [4095, 4607] { qAll[(qTok * 24 + 17) * 128 + 127] = 18.0 }

        func runRange(_ op: QWLSAOperator, start: Int, count: Int, selectLastChunk: Bool = false) {
            for s0 in stride(from: start, to: start + count, by: 512) {
                let n = min(512, start + count - s0)
                let qSlice = Array(qAll[(s0 * 24 * 128)..<((s0 + n) * 24 * 128)])
                let kSlice = Array(kAll[(s0 * 8 * 128)..<((s0 + n) * 8 * 128)])
                let qTV = Self.makeBF16Buffer(hal, qSlice, label: "q_mh")
                let kTV = Self.makeBF16Buffer(hal, kSlice, label: "k_mh")
                guard let (cb, enc) = hal.begin() else { return XCTFail("cb") }
                op.encodeIndexAndClusterBF16(enc, qIn: qTV, kIn: kTV, fullAttnLayer: 0,
                                             pos0: s0, rows: n, qRowStride: 24 * 128, qMode: .allRows)
                if selectLastChunk && s0 + n == start + count {
                    op.encodeSelectBlocks(enc, fullAttnLayer: 0, pos0: s0, rows: n, tiled32: false)
                }
                hal.end(cb, enc)
                cb.waitUntilCompleted()
            }
        }

        runRange(op1, start: 0, count: 4096, selectLastChunk: true)
        let blocksAt4096 = Self.readRowBlocks(op1, row: 511)
        XCTAssertEqual(blocksAt4096.count, 1 + 128 + 128)
        XCTAssertEqual(blocksAt4096.first, 0)
        XCTAssertEqual(blocksAt4096, blocksAt4096.sorted())
        XCTAssertTrue(blocksAt4096.contains(bStar), "must select single-head needle block \(bStar)")

        let fullSnap = op1.captureIndexerState(position: 4096)
        XCTAssertTrue(op2.isValidIndexerState(fullSnap, position: 4096))
        XCTAssertTrue(op2.restoreIndexerState(fullSnap, position: 4096))

        op1.setPinnedBaseState(fullSnap, position: 4096)
        op2.setPinnedBaseState(fullSnap, position: 4096)
        runRange(op1, start: 4096, count: 512, selectLastChunk: true)
        let expectedAt4608 = Self.readRowBlocks(op1, row: 511)
        XCTAssertTrue(expectedAt4608.contains(bStar))

        let deltaSnap = op1.captureIndexerState(position: 4608)
        XCTAssertEqual(deltaSnap.count, op1.deltaIndexerStateBytes(basePos512: 4096, position: 4608))
        XCTAssertTrue(op2.isValidIndexerState(deltaSnap, position: 4608))
        XCTAssertTrue(op2.restoreIndexerState(deltaSnap, position: 4608))

        let qTVLast = Self.makeBF16Buffer(hal, Array(qAll[(4607 * 24 * 128)...]), label: "ql")
        let kTVLast = Self.makeBF16Buffer(hal, Array(kAll[(4607 * 8 * 128)...]), label: "kl")
        guard let (cb, enc) = hal.begin() else { return XCTFail("cb") }
        op2.encodeIndexAndClusterBF16(enc, qIn: qTVLast, kIn: kTVLast, fullAttnLayer: 0,
                                      pos0: 4607, rows: 1, qRowStride: 24 * 128, qMode: .allRows)
        op2.encodeSelectBlocks(enc, fullAttnLayer: 0, pos0: 4607, rows: 1)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertEqual(Self.readRowBlocks(op2, row: 0), expectedAt4608)

        // Verify non-16-aligned snapshot capture & delta round-trip (position=4614 => mod 16=6, mod 4=2).
        let qTail6 = Self.makeBF16Buffer(hal, [Float](repeating: 0.2, count: 6 * 24 * 128), label: "qt6")
        let kTail6 = Self.makeBF16Buffer(hal, [Float](repeating: 0.1, count: 6 * 8 * 128), label: "kt6")
        guard let (cb6, enc6) = hal.begin() else { return XCTFail("cb") }
        op1.encodeIndexAndClusterBF16(enc6, qIn: qTail6, kIn: kTail6, fullAttnLayer: 0,
                                      pos0: 4608, rows: 6, qRowStride: 24 * 128, qMode: .allRows)
        hal.end(cb6, enc6)
        cb6.waitUntilCompleted()
        let delta4614 = op1.captureIndexerState(position: 4614)
        XCTAssertEqual(delta4614.count, op1.deltaIndexerStateBytes(basePos512: 4096, position: 4614))
        XCTAssertTrue(op2.isValidIndexerState(delta4614, position: 4614))
        XCTAssertTrue(op2.restoreIndexerState(delta4614, position: 4614))
        let full4614 = op1.captureIndexerState(position: 4614, forceFull: true)
        XCTAssertEqual(full4614.count, op1.fullIndexerStateBytes(position: 4614))
        XCTAssertTrue(op2.isValidIndexerState(full4614, position: 4614))
        XCTAssertTrue(op2.restoreIndexerState(full4614, position: 4614))
    }

    func testFullSliceQ4CPUParityPartialRopeClippingAndFallback() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal device required")
        let hal = QWLSATests.hal

        // Case A: Partial RoPE headDim=256 (sliceStart=64, sliceDim=192), FP16 gated inputs, kL2Clip=0, tiled32 prefill.
        let geom192 = try Self.makeMHGeometry(qHeads: 16, kvHeads: 4, headDim: 256, partialRotary: 0.25)
        let cfg = QWLSAConfig(mode: .svd, budget: 512, clusterBlock: 512)
        let op192 = try QWLSAOperator(
            hal: hal, geometry: geom192, config: cfg, indexDim: 64,
            layerWeights: [.init(physicalLayer: 0, wqIdxF16: [], wkIdxF16: [], isHeadFactorized: true)],
            maxContext: 4096, maxPrefillChunk: 512)
        XCTAssertEqual(op192.sliceStart, 64)
        XCTAssertEqual(op192.sliceDim, 192)
        let tokA = 2048
        let qStride = geom192.descriptor.attnOutputGate ? 2 : 1
        var qGate = [Float](repeating: 0, count: tokA * geom192.qProjOut)
        var kIn = [Float](repeating: 0, count: tokA * geom192.kvChannels)
        for t in 0..<tokA {
            for g in 0..<4 {
                for d in 0..<256 {
                    kIn[(t * 4 + g) * 256 + d] = sin(Float((t + 1) * (d + 3) + g * 17) * 0.17)
                }
            }
            for qh in 0..<16 {
                for d in 0..<256 {
                    qGate[(t * 16 * qStride + qh * qStride) * 256 + d] =
                        cos(Float((t + 5) * (d + 1) + qh * 11) * 0.21)
                }
            }
        }
        for s0 in stride(from: 0, to: tokA, by: 512) {
            let qTV = Self.makeFP16Buffer(hal, Array(qGate[(s0 * geom192.qProjOut)..<((s0 + 512) * geom192.qProjOut)]), label: "qp")
            let kTV = Self.makeFP16Buffer(hal, Array(kIn[(s0 * geom192.kvChannels)..<((s0 + 512) * geom192.kvChannels)]), label: "kp")
            guard let (cb, enc) = hal.begin() else { return XCTFail("cb") }
            let isLast = (s0 == 1536)
            op192.encodeIndexAndCluster(enc, qgate: qTV, kIn: kTV, fullAttnLayer: 0, pos0: s0, rows: 512,
                                        qMode: isLast ? .tiled32 : .none)
            if isLast { op192.encodeSelectBlocks(enc, fullAttnLayer: 0, pos0: s0, rows: 512, tiled32: true) }
            hal.end(cb, enc)
            cb.waitUntilCompleted()
        }
        let clippedK192: [[[Float]]] = (0..<tokA).map { t in
            (0..<4).map { g in
                (64..<256).map { d in Float(Float16(kIn[(t * 4 + g) * 256 + d])) }
            }
        }
        for refRow in [31, 511] {
            let t = 1536 + refRow
            let qRow: [[Float]] = (0..<16).map { qh in
                (64..<256).map { d in Float(Float16(qGate[(t * 16 * qStride + qh * qStride) * 256 + d])) }
            }
            let expected = Self.cpuReferenceSelectRow(qRow: qRow, clippedK: clippedK192, pos: t)
            XCTAssertEqual(Self.readRowBlocks(op192, row: refRow), expected, "CPU parity mismatch at tiled row \(refRow)")
        }

        // Case B: BF16 + kL2Clip=22.627417 with norm>clip + massive zero-tie fallback.
        let clipVal: Float = 22.627417
        let geom128 = try Self.makeMHGeometry(qHeads: 24, kvHeads: 8, headDim: 128, inklingClip: clipVal)
        let op128 = try QWLSAOperator(
            hal: hal, geometry: geom128, config: cfg, indexDim: 64,
            layerWeights: [.init(physicalLayer: 5, wqIdxF16: [], wkIdxF16: [], isHeadFactorized: true)],
            maxContext: 24_576, maxPrefillChunk: 512)
        let tokB = 2048
        var qB = [Float](repeating: 0, count: tokB * 24 * 128)
        var kB = [Float](repeating: 0, count: tokB * 8 * 128)
        for t in 0..<tokB {
            for g in 0..<8 {
                for d in 0..<128 {
                    kB[(t * 8 + g) * 128 + d] = (d == 0 ? 55.0 : 0.0) + 4.0 * sin(Float(t * 7 + g * 19 + d) * 0.13)
                }
            }
            for qh in 0..<24 {
                for d in 0..<128 { qB[(t * 24 + qh) * 128 + d] = 1.5 * cos(Float(t * 3 + qh * 5 + d) * 0.17) }
            }
        }
        for s0 in stride(from: 0, to: tokB, by: 512) {
            let qTV = Self.makeBF16Buffer(hal, Array(qB[(s0 * 24 * 128)..<((s0 + 512) * 24 * 128)]), label: "qb")
            let kTV = Self.makeBF16Buffer(hal, Array(kB[(s0 * 8 * 128)..<((s0 + 512) * 8 * 128)]), label: "kb")
            guard let (cb, enc) = hal.begin() else { return XCTFail("cb") }
            op128.encodeIndexAndClusterBF16(enc, qIn: qTV, kIn: kTV, fullAttnLayer: 0, pos0: s0, rows: 512,
                                            qRowStride: 24 * 128, qMode: .allRows)
            if s0 == 1536 { op128.encodeSelectBlocks(enc, fullAttnLayer: 0, pos0: s0, rows: 512, tiled32: false) }
            hal.end(cb, enc)
            cb.waitUntilCompleted()
        }
        let clippedK128: [[[Float]]] = (0..<tokB).map { t in
            (0..<8).map { g in
                let raw = (0..<128).map { d in Self.bf16ToF32(Self.f32ToBF16(kB[(t * 8 + g) * 128 + d])) }
                let normSq = raw.reduce(Float(0)) { $0 + $1 * $1 }
                let scK = min(clipVal / (sqrt(normSq) + 1e-6), 1.0)
                return raw.map { Float(Float16($0 * scK)) }
            }
        }
        let qRowB: [[Float]] = (0..<24).map { qh in
            (0..<128).map { d in Float(Float16(Self.bf16ToF32(Self.f32ToBF16(qB[(2047 * 24 + qh) * 128 + d])))) }
        }
        XCTAssertEqual(
            Self.readRowBlocks(op128, row: 511),
            Self.cpuReferenceSelectRow(qRow: qRowB, clippedK: clippedK128, pos: 2047))

        // Massive zero-tie fallback (18,000 zero tokens => !fast path) + zero-at-rest verification.
        let tokZero = 18_000
        let zeroQ = Self.makeBF16Buffer(hal, [Float](repeating: 0, count: 512 * 24 * 128), label: "zq")
        let zeroK = Self.makeBF16Buffer(hal, [Float](repeating: 0, count: 512 * 8 * 128), label: "zk")
        for s0 in stride(from: 0, to: tokZero, by: 512) {
            let n = min(512, tokZero - s0)
            guard let (cb, enc) = hal.begin() else { return XCTFail("cb") }
            op128.encodeIndexAndClusterBF16(enc, qIn: zeroQ, kIn: zeroK, fullAttnLayer: 0,
                                            pos0: s0, rows: n, qRowStride: 24 * 128, qMode: .allRows)
            if s0 + n == tokZero { op128.encodeSelectBlocks(enc, fullAttnLayer: 0, pos0: s0, rows: n, tiled32: false) }
            hal.end(cb, enc)
            cb.waitUntilCompleted()
        }
        let totB = tokZero / 4
        let midEndB = totB - 128
        let expectedFallback = [0] + Array((midEndB - 128)..<midEndB) + Array(midEndB..<totB)
        let lastRow = tokZero - (((tokZero - 1) / 512) * 512) - 1
        XCTAssertEqual(Self.readRowBlocks(op128, row: lastRow), expectedFallback)

        let histPtr = op128.region.pointer.advanced(by: op128.selHist.offset).assumingMemoryBound(to: UInt32.self)
        let cntPtr = op128.region.pointer.advanced(by: op128.selCount.offset).assumingMemoryBound(to: UInt32.self)
        for i in 0..<512 { XCTAssertEqual(histPtr[i], 0) }
        XCTAssertEqual(cntPtr[0], 0)

        // Mixed isGt + heavy-tie !fast fallback check (20,000 tokens => 4,871 middle blocks > SORT_N=4096).
        let tokMix = 20_000
        let totMix = tokMix / 4
        let midEndMix = totMix - 128
        let highBlocks = Array(stride(from: 50, to: 4800, by: 160)) // 30 scattered blocks across chunks
        let highSet = Set(highBlocks)
        let lastChunkStart = ((tokMix - 1) / 512) * 512
        let lastChunkRows = tokMix - lastChunkStart
        var mixQArr = [Float](repeating: 0, count: lastChunkRows * 24 * 128)
        mixQArr[(lastChunkRows - 1) * 24 * 128] = 12.0
        let mixQ = Self.makeBF16Buffer(hal, mixQArr, label: "mq")
        for s0 in stride(from: 0, to: tokMix, by: 512) {
            let n = min(512, tokMix - s0)
            let bStart = s0 / 4, bEnd = (s0 + n) / 4
            let kBuf: QWTensorValue
            if (bStart..<bEnd).contains(where: { highSet.contains($0) }) {
                var kArr = [Float](repeating: 0, count: n * 8 * 128)
                for b in bStart..<bEnd where highSet.contains(b) {
                    for off in 0..<4 { kArr[((b - bStart) * 4 + off) * 8 * 128] = 12.0 }
                }
                kBuf = Self.makeBF16Buffer(hal, kArr, label: "mk")
            } else {
                kBuf = zeroK
            }
            let isLast = (s0 + n == tokMix)
            guard let (cb, enc) = hal.begin() else { return XCTFail("cb") }
            op128.encodeIndexAndClusterBF16(enc, qIn: isLast ? mixQ : zeroQ, kIn: kBuf, fullAttnLayer: 0,
                                            pos0: s0, rows: n, qRowStride: 24 * 128,
                                            qMode: isLast ? .allRows : .none)
            if isLast { op128.encodeSelectBlocks(enc, fullAttnLayer: 0, pos0: s0, rows: n, tiled32: false) }
            hal.end(cb, enc)
            cb.waitUntilCompleted()
        }
        let nonHigh = (1..<midEndMix).filter { !highSet.contains($0) }
        let expectedMix = [0] + (highBlocks + Array(nonHigh.suffix(128 - highBlocks.count))).sorted()
            + Array(midEndMix..<totMix)
        XCTAssertEqual(Self.readRowBlocks(op128, row: lastChunkRows - 1), expectedMix)
        for i in 0..<512 { XCTAssertEqual(histPtr[i], 0) }
        XCTAssertEqual(cntPtr[0], 0)
    }

    func testTwoStageCoarseToFineParitySinkSuppressionAndFallback() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal device required")
        let hal = QWLSATests.hal
        let geom = try Self.makeMHGeometry(qHeads: 24, kvHeads: 8, headDim: 128, inklingClip: 22.627417)
        let cfg = QWLSAConfig(mode: .svd, budget: 512, clusterBlock: 512)
        let op = try QWLSAOperator(
            hal: hal, geometry: geom, config: cfg, indexDim: 64,
            layerWeights: [.init(physicalLayer: 5, wqIdxF16: [], wkIdxF16: [], isHeadFactorized: true)],
            maxContext: 36_864, maxPrefillChunk: 512)
        XCTAssertEqual(op.region.length, QWLSAOperator.estimateRegionBytes(
            geometry: geom, lsaConfig: cfg, attnLayers: [0], maxContext: 36_864,
            maxPrefillChunk: 512, indexDim: 64, isMultiHeadDirect: true))

        // Stream 34,000 tokens (8500 blocks, numMid=8371 > 8192, numCoarseMid=2093 > 2048 => 2-stage!).
        let totalTok = 34_000
        let bgPeriod = 64
        let bgPattern: [[[Float]]] = (0..<bgPeriod).map { p in
            (0..<8).map { g in
                var vec = [Float](repeating: 0, count: 128)
                vec[(p * 5 + g * 11) % 60] = 2.0 + 0.25 * Float(p % 7)
                vec[64 + ((p + g) % 32)] = -1.5
                return vec.map { Float(Float16(Self.bf16ToF32(Self.f32ToBF16($0)))) }
            }
        }
        var clippedK = [[[Float]]](repeating: [], count: totalTok)
        for t in 0..<totalTok { clippedK[t] = bgPattern[t & (bgPeriod - 1)] }
        for t in 2000..<2016 { clippedK[t][0][120] = 10.0 }      // Distractor coarse block 125 (fine 500..503)
        for t in 4800..<4816 { clippedK[t][5][121] = 11.0 }      // Needle coarse block 300 (fine 1200..1203)
        for t in 8000..<8640 { clippedK[t][1][122] = 8.0 }       // 160 medium blocks (fine 2000..<2160)

        var qQuery = [[Float]](repeating: [Float](repeating: 0, count: 128), count: 24)
        for qh in 0..<24 { qQuery[qh][qh % 16] = 1.0 }
        qQuery[0][120] = 12.0   // Distractor head 0 (logit +10.61)
        qQuery[3][122] = 8.0    // Medium head 3 (160 blocks, prob ~0.00625 each)
        qQuery[17][121] = 10.0  // Needle head 17 (logit +9.72)

        var sinkRow = [Float](repeating: 0.0, count: 24)
        sinkRow[0] = 22.0 // Learned sink suppresses distractor head 0 (~1.1e-5 << 0.00625) when sink != nil!
        let flatQ64 = Array(repeating: qQuery.flatMap { $0 }, count: 64).flatMap { $0 }
        let qPref64 = Self.makeBF16Buffer(hal, flatQ64, label: "qp64")
        let qDec1 = Self.makeBF16Buffer(hal, qQuery.flatMap { $0 }, label: "qd1")
        let zeroQ512 = Self.makeBF16Buffer(hal, [Float](repeating: 0, count: 512 * 24 * 128), label: "zq512")
        let sink64 = Self.makeBF16Buffer(hal, Array(repeating: sinkRow, count: 64).flatMap { $0 }, label: "sk64")
        let sink1 = Self.makeBF16Buffer(hal, sinkRow, label: "sk1")

        let bgChunkFlat = (0..<512).flatMap { t in bgPattern[t & (bgPeriod - 1)].flatMap { $0 } }
        let bgChunkTV = Self.makeBF16Buffer(hal, bgChunkFlat, label: "kbg")
        for s0 in stride(from: 0, to: totalTok - 64, by: 512) {
            let n = min(512, (totalTok - 64) - s0)
            let needsCustom = (s0..<s0 + n).contains(where: {
                (2000..<2016).contains($0) || (4800..<4816).contains($0) || (8000..<8640).contains($0)
            })
            let kTV: QWTensorValue
            if needsCustom {
                var kArr = [Float]()
                kArr.reserveCapacity(n * 8 * 128)
                for t in s0..<(s0 + n) {
                    for g in 0..<8 { kArr.append(contentsOf: clippedK[t][g]) }
                }
                kTV = Self.makeBF16Buffer(hal, kArr, label: "kp")
            } else {
                kTV = bgChunkTV
            }
            guard let (cb, enc) = hal.begin() else { return XCTFail("cb") }
            op.encodeIndexAndClusterBF16(enc, qIn: zeroQ512, kIn: kTV, fullAttnLayer: 0,
                                         pos0: s0, rows: n, qRowStride: 24 * 128, qMode: .none)
            hal.end(cb, enc)
            cb.waitUntilCompleted()
        }

        // Prefill last 64 tokens [33_936 ..< 34_000) with tiled32 + sink and assert exact CPU parity.
        let pStart = totalTok - 64
        let kTail64 = Self.makeBF16Buffer(hal, (pStart..<totalTok).flatMap { t in clippedK[t].flatMap { $0 } }, label: "kt")
        guard let (cbP, encP) = hal.begin() else { return XCTFail("cb") }
        op.encodeIndexAndClusterBF16(encP, qIn: qPref64, kIn: kTail64, fullAttnLayer: 0,
                                     pos0: pStart, rows: 64, qRowStride: 24 * 128, qMode: .tiled32)
        op.encodeSelectBlocks(encP, fullAttnLayer: 0, pos0: pStart, rows: 64, tiled32: true, sink: sink64)
        hal.end(cbP, encP)
        cbP.waitUntilCompleted()

        let qRowF16 = qQuery.map { row in row.map { Float(Float16(Self.bf16ToF32(Self.f32ToBF16($0)))) } }
        let sinkF32 = sinkRow.map { Self.bf16ToF32(Self.f32ToBF16($0)) }
        for refRow in [31, 63] {
            let expected = Self.cpuReferenceSelectRow(
                qRow: qRowF16, clippedK: clippedK, pos: pStart + refRow, sinkRow: sinkF32)
            XCTAssertEqual(Self.readRowBlocks(op, row: refRow), expected, "2-stage tiled32 mismatch at row \(refRow)")
        }

        // Decode T=1 at pos=33_999 with sink vs without sink: needle block 1200 must win with sink & match CPU!
        let lastK1 = Self.makeBF16Buffer(hal, clippedK[totalTok - 1].flatMap { $0 }, label: "kd1")
        guard let (cbD, encD) = hal.begin() else { return XCTFail("cb") }
        op.encodeIndexAndClusterBF16(encD, qIn: qDec1, kIn: lastK1, fullAttnLayer: 0,
                                     pos0: totalTok - 1, rows: 1, qRowStride: 24 * 128, qMode: .allRows)
        op.encodeSelectBlocks(encD, fullAttnLayer: 0, pos0: totalTok - 1, rows: 1, tiled32: false, sink: sink1)
        hal.end(cbD, encD)
        cbD.waitUntilCompleted()
        let gotWithSink = Self.readRowBlocks(op, row: 0)
        XCTAssertTrue(gotWithSink.contains(1200), "needle block 1200 must be selected with sink suppression")
        XCTAssertFalse(gotWithSink.contains(500), "distractor block 500 must be suppressed by sink=+22")
        XCTAssertEqual(gotWithSink, Self.cpuReferenceSelectRow(
            qRow: qRowF16, clippedK: clippedK, pos: totalTok - 1, sinkRow: sinkF32))

        // Decode without sink must match CPU sink=nil (distractor head 0 dominates when sink isn't accounted for).
        guard let (cbNS, encNS) = hal.begin() else { return XCTFail("cb") }
        op.encodeSelectBlocks(encNS, fullAttnLayer: 0, pos0: totalTok - 1, rows: 1, tiled32: false, sink: nil)
        hal.end(cbNS, encNS)
        cbNS.waitUntilCompleted()
        let gotNoSink = Self.readRowBlocks(op, row: 0)
        XCTAssertTrue(gotNoSink.contains(500))
        XCTAssertEqual(gotNoSink, Self.cpuReferenceSelectRow(
            qRow: qRowF16, clippedK: clippedK, pos: totalTok - 1, sinkRow: nil))

        // Re-run a 128-row (4 tiled32 units, bqTile=4) prefill tail and verify exact CPU parity.
        let pStart128 = totalTok - 128
        let flatQ128 = Array(repeating: qQuery.flatMap { $0 }, count: 128).flatMap { $0 }
        let sink128 = Self.makeBF16Buffer(hal, Array(repeating: sinkRow, count: 128).flatMap { $0 }, label: "sk128")
        let qPref128 = Self.makeBF16Buffer(hal, flatQ128, label: "qp128")
        let kTail128 = Self.makeBF16Buffer(
            hal, (pStart128..<totalTok).flatMap { t in clippedK[t].flatMap { $0 } }, label: "kt128")
        guard let (cbP4, encP4) = hal.begin() else { return XCTFail("cb") }
        op.encodeIndexAndClusterBF16(encP4, qIn: qPref128, kIn: kTail128, fullAttnLayer: 0,
                                     pos0: pStart128, rows: 128, qRowStride: 24 * 128, qMode: .tiled32)
        op.encodeSelectBlocks(encP4, fullAttnLayer: 0, pos0: pStart128, rows: 128, tiled32: true, sink: sink128)
        hal.end(cbP4, encP4)
        cbP4.waitUntilCompleted()
        for refRow in [31, 63, 127] {
            let expected = Self.cpuReferenceSelectRow(
                qRow: qRowF16, clippedK: clippedK, pos: pStart128 + refRow, sinkRow: sinkF32)
            XCTAssertEqual(Self.readRowBlocks(op, row: refRow), expected, "bqTile=4 mismatch at row \(refRow)")
        }

        // 2-stage !fast fallback check (34,000 zero tokens => coarse_expand keeps 2048 -> ~8190 fine > SORT_N=4096).
        let zeroK512 = Self.makeBF16Buffer(hal, [Float](repeating: 0, count: 512 * 8 * 128), label: "zk512")
        for s0 in stride(from: 0, to: totalTok, by: 512) {
            let n = min(512, totalTok - s0)
            guard let (cbZ, encZ) = hal.begin() else { return XCTFail("cb") }
            op.encodeIndexAndClusterBF16(encZ, qIn: zeroQ512, kIn: zeroK512, fullAttnLayer: 0,
                                         pos0: s0, rows: n, qRowStride: 24 * 128, qMode: .allRows)
            if s0 + n == totalTok {
                op.encodeSelectBlocks(encZ, fullAttnLayer: 0, pos0: s0, rows: n, tiled32: false, sink: nil)
            }
            hal.end(cbZ, encZ)
            cbZ.waitUntilCompleted()
        }
        let totB = totalTok / 4
        let midEndB = totB - 128
        let expected2StageFallback = [0] + Array((midEndB - 128)..<midEndB) + Array(midEndB..<totB)
        let lastZeroRow = totalTok - (((totalTok - 1) / 512) * 512) - 1
        XCTAssertEqual(Self.readRowBlocks(op, row: lastZeroRow), expected2StageFallback)
    }

    func testAdaptiveLongContextBudgetAndExplicitOverride() throws {
        XCTAssertEqual(QWLSAConfig.adaptiveBudget(forMaxContext: 8192), 2048)
        XCTAssertEqual(QWLSAConfig.adaptiveBudget(forMaxContext: 65_536), 2048)
        XCTAssertEqual(QWLSAConfig.adaptiveBudget(forMaxContext: 98_304), 4096)
        XCTAssertEqual(QWLSAConfig.adaptiveBudget(forMaxContext: 131_072), 4096)
        XCTAssertEqual(QWLSAConfig.adaptiveBudget(forMaxContext: 196_608), 8192)
        XCTAssertEqual(QWLSAConfig.adaptiveBudget(forMaxContext: 262_144), 8192)
        XCTAssertEqual(QWLSAConfig.adaptiveBudget(forMaxContext: 393_216), 8192)
        XCTAssertEqual(QWLSAConfig.adaptiveBudget(forMaxContext: 524_288), 8192)
        XCTAssertEqual(QWLSAConfig.adaptiveBudget(forMaxContext: 786_432), 16_384)
        XCTAssertEqual(QWLSAConfig.adaptiveBudget(forMaxContext: 1_048_576), 16_384)

        let autoDefault = QWLSAConfig(mode: .auto)
        XCTAssertEqual(autoDefault.resolvingAdaptiveBudget(forMaxContext: 65_536).budget, 2048)
        XCTAssertEqual(autoDefault.resolvingAdaptiveBudget(forMaxContext: 131_072).budget, 4096)
        XCTAssertEqual(autoDefault.resolvingAdaptiveBudget(forMaxContext: 262_144).budget, 8192)
        XCTAssertEqual(autoDefault.resolvingAdaptiveBudget(forMaxContext: 524_288).budget, 8192)
        XCTAssertEqual(autoDefault.resolvingAdaptiveBudget(forMaxContext: 1_048_576).budget, 16_384)
        XCTAssertEqual(QWLSAConfig(mode: .off).resolvingAdaptiveBudget(forMaxContext: 1_048_576).budget, 2048)

        let explicit2k = QWLSAConfig(mode: .auto, budget: 2048, explicitBudget: true)
        XCTAssertEqual(explicit2k.resolvingAdaptiveBudget(forMaxContext: 1_048_576).budget, 2048)
        let explicit1k = QWLSAConfig(mode: .auto, budget: 1024)
        XCTAssertEqual(explicit1k.resolvingAdaptiveBudget(forMaxContext: 1_048_576).budget, 1024)

        let decodedDefault = try JSONDecoder().decode(QWLSAConfig.self, from: JSONEncoder().encode(autoDefault))
        XCTAssertEqual(decodedDefault.resolvingAdaptiveBudget(forMaxContext: 262_144).budget, 8192)
        XCTAssertEqual(decodedDefault.resolvingAdaptiveBudget(forMaxContext: 524_288).budget, 8192)
        XCTAssertEqual(decodedDefault.resolvingAdaptiveBudget(forMaxContext: 1_048_576).budget, 16_384)
        let decoded1k = try JSONDecoder().decode(QWLSAConfig.self, from: JSONEncoder().encode(explicit1k))
        XCTAssertEqual(decoded1k.resolvingAdaptiveBudget(forMaxContext: 1_048_576).budget, 1024)

        let geom = try Self.makeMHGeometry(qHeads: 24, kvHeads: 8, headDim: 128, inklingClip: 22.627417)
        let estAuto1M = QWLSAOperator.estimateRegionBytes(
            geometry: geom, lsaConfig: autoDefault, attnLayers: [0],
            maxContext: 1_048_576, maxPrefillChunk: 512, indexDim: 64, isMultiHeadDirect: true)
        let est16k1M = QWLSAOperator.estimateRegionBytes(
            geometry: geom, lsaConfig: QWLSAConfig(mode: .auto, budget: 16_384), attnLayers: [0],
            maxContext: 1_048_576, maxPrefillChunk: 512, indexDim: 64, isMultiHeadDirect: true)
        let est2k1M = QWLSAOperator.estimateRegionBytes(
            geometry: geom, lsaConfig: explicit2k, attnLayers: [0],
            maxContext: 1_048_576, maxPrefillChunk: 512, indexDim: 64, isMultiHeadDirect: true)
        XCTAssertEqual(estAuto1M, est16k1M)
        XCTAssertGreaterThan(estAuto1M, est2k1M)

        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal device required")
        let op256k = try QWLSAOperator(
            hal: QWLSATests.hal, geometry: geom, config: QWLSAConfig(mode: .svd), indexDim: 64,
            layerWeights: [.init(physicalLayer: 5, wqIdxF16: [], wkIdxF16: [], isHeadFactorized: true)],
            maxContext: 262_144, maxPrefillChunk: 512)
        XCTAssertEqual(op256k.config.budget, 8192)
        XCTAssertEqual(op256k.mhCoarseCap, 4096)
        XCTAssertEqual(op256k.region.length, QWLSAOperator.estimateRegionBytes(
            geometry: geom, lsaConfig: QWLSAConfig(mode: .svd), attnLayers: [0],
            maxContext: 262_144, maxPrefillChunk: 512, indexDim: 64, isMultiHeadDirect: true))
    }
}
