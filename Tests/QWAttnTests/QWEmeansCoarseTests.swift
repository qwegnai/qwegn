// QWEmeansCoarseTests.swift — Unit, needle-recall, snapshot, and paired GPU A/B tests
// for capacity-balanced e-means (arXiv:2611.09501) `QWCoarsePoolIndex` across LSA and MLA.

import XCTest
import Metal
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWQuant

final class QWEmeansCoarseTests: XCTestCase {
    static let hal: QWHal = QWLSATests.hal
    static let geometry = QWLSATests.geometry

    private static func makeLSA(maxContext: Int, budget: Int, clusterBlock: Int) throws -> QWLSAOperator {
        let cfg = QWLSAConfig(mode: .svd, budget: budget, clusterBlock: clusterBlock)
        let inDim = geometry.lsaInputDim
        let basis = QWLSASidecar.synthesizeDeterministicOrthogonalBasis(inDim: inDim, indexDim: 64, seed: 101)
        let lw = QWLSAOperator.LayerWeights(physicalLayer: 3, wqIdxF16: basis, wkIdxF16: basis)
        return try QWLSAOperator(
            hal: hal, geometry: geometry, config: cfg, indexDim: 64,
            layerWeights: [lw], maxContext: maxContext, maxPrefillChunk: 512)
    }

    private static func synthTopicHaystack(
        numPools: Int,
        dim: Int,
        rmsScale: Float,
        needlePools: [Int],
        seed: UInt64 = 0xBEEF_CAFE
    ) -> (poolsF16: [UInt16], needleDirs: [[Float]]) {
        var state = seed
        func nextNormal() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let u1 = max(1e-6, Float((state >> 33) & 0x7FFF_FFFF) / Float(0x8000_0000))
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let u2 = Float((state >> 33) & 0x7FFF_FFFF) / Float(0x8000_0000)
            return sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
        }
        let nTopics = 128
        var topics = [[Float]]()
        topics.reserveCapacity(nTopics)
        for _ in 0..<nTopics {
            var v = (0..<dim).map { _ in nextNormal() }
            let n = max(1e-6, sqrt(v.reduce(0) { $0 + $1 * $1 }))
            for d in 0..<dim { v[d] /= n }
            topics.append(v)
        }
        var out = [UInt16](repeating: 0, count: numPools * dim)
        var p = 0
        while p < numPools {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let tIdx = Int((state >> 48) % UInt64(nTopics))
            let topic = topics[tIdx]
            let span = min(32, numPools - p)
            for s in 0..<span {
                var row = [Float](repeating: 0, count: dim)
                var sq: Float = 0
                for d in 0..<dim {
                    let v = topic[d] + 0.35 * nextNormal()
                    row[d] = v
                    sq += v * v
                }
                let gain = rmsScale / max(1e-6, sqrt(sq))
                for d in 0..<dim {
                    out[(p + s) * dim + d] = Float16(row[d] * gain).bitPattern
                }
            }
            p += span
        }
        var needleDirs = [[Float]]()
        for np in needlePools {
            var dir = (0..<dim).map { _ in nextNormal() }
            let n = max(1e-6, sqrt(dir.reduce(0) { $0 + $1 * $1 }))
            for d in 0..<dim { dir[d] /= n }
            needleDirs.append(dir)
            for off in 0..<2 where (np + off) < numPools {
                var row = [Float](repeating: 0, count: dim)
                var sq: Float = 0
                for d in 0..<dim {
                    let v = dir[d] + 0.03 * nextNormal()
                    row[d] = v
                    sq += v * v
                }
                let gain = rmsScale / max(1e-6, sqrt(sq))
                for d in 0..<dim {
                    out[(np + off) * dim + d] = Float16(row[d] * gain).bitPattern
                }
            }
        }
        return (out, needleDirs)
    }

    func testEmeansCluster1024CSRPartitionAndDeterminism() throws {
        let op = try Self.makeLSA(maxContext: 8192, budget: 512, clusterBlock: 4096)
        XCTAssertTrue(op.isEmeansCoarse)
        let numPools = 2048
        let (pools, _) = Self.synthTopicHaystack(numPools: numPools, dim: 64, rmsScale: 1.0, needlePools: [300, 1400])
        let ls = op.layers[0]
        let basePtr = op.region.pointer
        pools.withUnsafeBytes { raw in
            basePtr.advanced(by: ls.pooledKeys.offset).copyMemory(from: raw.baseAddress!, byteCount: raw.count)
        }

        let pipes = try QWCoarsePoolIndex.Pipelines(hal: Self.hal)
        let csr = QWCoarsePoolIndex.LayerCSR(
            centroids: ls.centroids, memberPools: ls.bucketMembers,
            clusterSpans: ls.bucketSizes, maxMacros: op.maxSuperblocks)

        func runCluster() throws -> ([UInt16], [UInt16], [UInt16]) {
            guard let (cb, enc) = Self.hal.begin() else { XCTFail("hal.begin"); return ([], [], []) }
            QWCoarsePoolIndex.encodeClusterMacros(
                enc, pso: pipes.cluster1024, pooledKeys: ls.pooledKeys,
                csr: csr, startMacro: 0, numMacros: 2, indexDim: 64)
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
            let cPtr = basePtr.advanced(by: ls.centroids.offset).assumingMemoryBound(to: UInt16.self)
            let mPtr = basePtr.advanced(by: ls.bucketMembers.offset).assumingMemoryBound(to: UInt16.self)
            let sPtr = basePtr.advanced(by: ls.bucketSizes.offset).assumingMemoryBound(to: UInt16.self)
            return (
                Array(UnsafeBufferPointer(start: cPtr, count: 128 * 64)),
                Array(UnsafeBufferPointer(start: mPtr, count: 2048)),
                Array(UnsafeBufferPointer(start: sPtr, count: 128 * 2))
            )
        }

        let r1 = try runCluster()
        let r2 = try runCluster()
        XCTAssertEqual(r1.0, r2.0, "centroids must be deterministic across runs")
        XCTAssertEqual(r1.1, r2.1, "memberPools CSR permutation must be deterministic")
        XCTAssertEqual(r1.2, r2.2, "clusterSpans CSR offsets must be deterministic")

        for m in 0..<2 {
            var expectedOff = 0
            var seen = Set<Int>()
            var nonEmpty = 0
            for j in 0..<64 {
                let off = Int(r1.2[(m * 64 + j) * 2])
                let cnt = Int(r1.2[(m * 64 + j) * 2 + 1])
                XCTAssertEqual(off, expectedOff, "CSR offset mismatch at macro \(m) cluster \(j)")
                expectedOff += cnt
                if cnt > 0 { nonEmpty += 1 }
                for k in 0..<cnt {
                    let pool = Int(r1.1[m * 1024 + off + k])
                    XCTAssertTrue((0..<1024).contains(pool))
                    XCTAssertTrue(seen.insert(pool).inserted, "duplicate pool \(pool) in macro \(m)")
                }
                var normSq: Float = 0
                for d in 0..<64 {
                    let v = Float(Float16(bitPattern: r1.0[(m * 64 + j) * 64 + d]))
                    normSq += v * v
                }
                if cnt > 0 {
                    XCTAssertEqual(sqrt(normSq), 1.0, accuracy: 2e-2, "centroid \(j) must be unit-normalized")
                }
            }
            XCTAssertEqual(expectedOff, 1024)
            XCTAssertEqual(seen.count, 1024)
            XCTAssertGreaterThanOrEqual(nonEmpty, 48, "capacity balancing must keep clusters populated")
        }
    }

    func testLSASnapshotCompressionAndCOWDeltaRoundTrip() throws {
        let wardOp = try Self.makeLSA(maxContext: 16384, budget: 512, clusterBlock: 512)
        let emOp = try Self.makeLSA(maxContext: 16384, budget: 512, clusterBlock: 4096)

        let pos = 16384
        let pooledOnly = (pos / 4) * 64 * 2
        let wardClusterOverhead = wardOp.fullIndexerStateBytes(position: pos) - pooledOnly
        let emClusterOverhead = emOp.fullIndexerStateBytes(position: pos) - pooledOnly
        XCTAssertEqual(wardClusterOverhead, 32 * 8512)
        XCTAssertEqual(emClusterOverhead, 4 * 10496)
        XCTAssertGreaterThan(Double(wardClusterOverhead) / Double(emClusterOverhead), 6.48)

        let chunk = 512
        let q = QWAttnPagedPrefillTests.synthQGate(n: chunk, seed: 0x1111)
        let k = QWAttnPagedPrefillTests.synthKV(n: chunk, seed: 0x2222)
        let qb = Self.hal.device.makeBuffer(bytes: q, length: q.count * 2, options: .storageModeShared)!
        let kb = Self.hal.device.makeBuffer(bytes: k, length: k.count * 2, options: .storageModeShared)!
        let qTV = QWTensorValue(region: QWMRegion(buffer: qb, base: 0, length: qb.length),
                                offset: 0, shape: QWTensorShape([chunk * Self.geometry.qProjOut], .f16))
        let kTV = QWTensorValue(region: QWMRegion(buffer: kb, base: 0, length: kb.length),
                                offset: 0, shape: QWTensorShape([chunk * Self.geometry.kvChannels], .f16))

        func stepPrefill(_ op: QWLSAOperator, from p0: Int, to p1: Int) {
            var cur = p0
            while cur < p1 {
                guard let (cb, enc) = Self.hal.begin() else { return }
                op.encodeIndexAndCluster(enc, qgate: qTV, kIn: kTV, fullAttnLayer: 0, pos0: cur, rows: chunk)
                Self.hal.end(cb, enc)
                cb.waitUntilCompleted()
                cur += chunk
            }
        }

        stepPrefill(emOp, from: 0, to: 8192)
        let baseSnap = emOp.captureIndexerState(position: 8192, forceFull: true)
        emOp.setPinnedBaseState(baseSnap, position: 8192)
        stepPrefill(emOp, from: 8192, to: 13312)
        let deltaSnap = emOp.captureIndexerState(position: 13312)
        XCTAssertLessThan(deltaSnap.count, emOp.fullIndexerStateBytes(position: 13312))
        XCTAssertTrue(emOp.isValidIndexerState(deltaSnap, position: 13312))

        guard let (cb1, enc1) = Self.hal.begin() else { XCTFail("hal.begin"); return }
        emOp.encodeIndexAndCluster(enc1, qgate: qTV, kIn: kTV, fullAttnLayer: 0, pos0: 13312, rows: 4)
        emOp.encodeSelectBlocks(enc1, fullAttnLayer: 0, pos0: 13312, rows: 4)
        Self.hal.end(cb1, enc1)
        cb1.waitUntilCompleted()
        let basePtr = emOp.region.pointer
        let cntPtr = basePtr.advanced(by: emOp.blockCountScratch.offset).assumingMemoryBound(to: Int32.self)
        let lstPtr = basePtr.advanced(by: emOp.blockListScratch.offset).assumingMemoryBound(to: Int32.self)
        let refCounts = Array(UnsafeBufferPointer(start: cntPtr, count: 4))
        let refLists = Array(UnsafeBufferPointer(start: lstPtr, count: 4 * emOp.listStride))

        let emRestored = try Self.makeLSA(maxContext: 16384, budget: 512, clusterBlock: 4096)
        emRestored.setPinnedBaseState(baseSnap, position: 8192)
        XCTAssertTrue(emRestored.restoreIndexerState(deltaSnap, position: 13312))
        guard let (cb2, enc2) = Self.hal.begin() else { XCTFail("hal.begin"); return }
        emRestored.encodeIndexAndCluster(enc2, qgate: qTV, kIn: kTV, fullAttnLayer: 0, pos0: 13312, rows: 4)
        emRestored.encodeSelectBlocks(enc2, fullAttnLayer: 0, pos0: 13312, rows: 4)
        Self.hal.end(cb2, enc2)
        cb2.waitUntilCompleted()
        let rBase = emRestored.region.pointer
        let rCnt = Array(UnsafeBufferPointer(
            start: rBase.advanced(by: emRestored.blockCountScratch.offset).assumingMemoryBound(to: Int32.self), count: 4))
        let rLst = Array(UnsafeBufferPointer(
            start: rBase.advanced(by: emRestored.blockListScratch.offset).assumingMemoryBound(to: Int32.self),
            count: 4 * emRestored.listStride))
        XCTAssertEqual(rCnt, refCounts)
        XCTAssertEqual(rLst, refLists)
    }

    func testLSAThreeNeedleRecallAt64K() throws {
        let totalTokens = 65_536
        let numPools = totalTokens / 4
        let needles = [1500, 7400, 13200]
        let op = try Self.makeLSA(maxContext: totalTokens + 1024, budget: 2048, clusterBlock: 4096)
        let (pools, needleDirs) = Self.synthTopicHaystack(
            numPools: numPools, dim: 64, rmsScale: 1.0, needlePools: needles)
        let ls = op.layers[0]
        let basePtr = op.region.pointer
        pools.withUnsafeBytes { raw in
            basePtr.advanced(by: ls.pooledKeys.offset).copyMemory(from: raw.baseAddress!, byteCount: raw.count)
        }
        let pipes = try QWCoarsePoolIndex.Pipelines(hal: Self.hal)
        let csr = QWCoarsePoolIndex.LayerCSR(
            centroids: ls.centroids, memberPools: ls.bucketMembers,
            clusterSpans: ls.bucketSizes, maxMacros: op.maxSuperblocks)
        guard let (cbC, encC) = Self.hal.begin() else { XCTFail("hal.begin"); return }
        QWCoarsePoolIndex.encodeClusterMacros(
            encC, pso: pipes.cluster1024, pooledKeys: ls.pooledKeys,
            csr: csr, startMacro: 0, numMacros: numPools / 1024, indexDim: 64)
        Self.hal.end(cbC, encC)
        cbC.waitUntilCompleted()
        XCTAssertNil(cbC.error)

        let qPtr = basePtr.advanced(by: op.qIdxScratch.offset).assumingMemoryBound(to: UInt16.self)
        for (r, dir) in needleDirs.enumerated() {
            for d in 0..<64 { qPtr[r * 64 + d] = Float16(dir[d]).bitPattern }
        }
        guard let (cbS, encS) = Self.hal.begin() else { XCTFail("hal.begin"); return }
        op.encodeSelectBlocks(encS, fullAttnLayer: 0, pos0: totalTokens - 3, rows: 3)
        Self.hal.end(cbS, encS)
        cbS.waitUntilCompleted()
        XCTAssertNil(cbS.error)

        let cntPtr = basePtr.advanced(by: op.blockCountScratch.offset).assumingMemoryBound(to: UInt32.self)
        let lstPtr = basePtr.advanced(by: op.blockListScratch.offset).assumingMemoryBound(to: Int32.self)
        for (r, np) in needles.enumerated() {
            let cnt = Int(cntPtr[r])
            XCTAssertEqual(cnt, 1 + 512 + 128)
            let selected = Array(UnsafeBufferPointer(start: lstPtr + r * op.listStride, count: cnt))
            for i in 1..<selected.count {
                XCTAssertLessThan(selected[i - 1], selected[i], "selected blocks must be strictly ascending")
            }
            XCTAssertEqual(selected[0], 0, "sink block 0 must be preserved")
            XCTAssertTrue(selected.contains(Int32(np)), "missing needle pool \(np) in row \(r)")
            XCTAssertTrue(selected.contains(Int32(np + 1)), "missing needle pool \(np + 1) in row \(r)")
        }
    }

    private static func mlaCoarseGeometry() throws -> QWModelGeometry {
        let cfg = """
        {
          "text_config": {
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
        }
        """.data(using: .utf8)!
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: cfg))
    }

    func testMLACoarseIndexThreeNeedleRecallAt64K() throws {
        let geo = try Self.mlaCoarseGeometry()
        let hal = try QWHal(geometry: geo)
        let totalTokens = 65_536
        let numPools = totalTokens / 4
        let needles = [1800, 8100, 14500]
        let mla = try QWMLAAttnOperator(
            hal: hal, geometry: geo, numAttnLayers: 1,
            maxContext: totalTokens + 1024, maxPrefillChunk: 64,
            lsaConfig: QWLSAConfig(mode: .svd, budget: 2048))
        let csr = try XCTUnwrap(mla.layers[0].coarseCSR)
        let (pools, needleDirs) = Self.synthTopicHaystack(
            numPools: numPools, dim: 128, rmsScale: sqrt(128.0), needlePools: needles, seed: 0xABCD_1234)
        let basePtr = mla.region.pointer
        pools.withUnsafeBytes { raw in
            basePtr.advanced(by: mla.layers[0].pooledKeys.offset).copyMemory(from: raw.baseAddress!, byteCount: raw.count)
        }

        let pipes = try QWCoarsePoolIndex.Pipelines(hal: hal)
        guard let (cbC, encC) = hal.begin() else { XCTFail("hal.begin"); return }
        QWCoarsePoolIndex.encodeClusterMacros(
            encC, pso: pipes.cluster1024, pooledKeys: mla.layers[0].pooledKeys,
            csr: csr, startMacro: 0, numMacros: numPools / 1024, indexDim: 128)
        hal.end(cbC, encC)
        cbC.waitUntilCompleted()
        XCTAssertNil(cbC.error)

        let qChannels = geo.qHeads * mla.nopeDim
        let qFullPtr = basePtr.advanced(by: mla.mlaQFullScratch.offset).assumingMemoryBound(to: UInt16.self)
        let idxWPtr = basePtr.advanced(by: mla.idxWeightsScratch.offset).assumingMemoryBound(to: Float.self)
        for r in 0..<3 {
            for h in 0..<32 {
                idxWPtr[r * 32 + h] = (h == 0) ? 1.0 : 0.05
                for d in 0..<128 {
                    let val: Float = (h == 0) ? needleDirs[r][d] : 0.0
                    qFullPtr[r * mla.qFullStride + qChannels + h * 128 + d] = Float16(val).bitPattern
                }
            }
        }

        let psoCoarse = try hal.pipeline("qwg_emeans_mla_score_coarse")
        let psoCands = try hal.pipeline("qwg_emeans_mla_score_cands")
        let psoFallback = try hal.pipeline("qwg_emeans_mla_flat_fallback")
        let cBuf = hal.device.makeBuffer(length: 16 << 20, options: .storageModeShared)!
        memset(cBuf.contents(), 0, cBuf.length)
        let cReg = QWMRegion(buffer: cBuf, base: 0, length: cBuf.length)
        let cScratch = QWCoarsePoolIndex.Scratch(
            cScore: QWTensorValue(region: cReg, offset: 0, shape: QWTensorShape([4, csr.maxClusters], .i32)),
            uHist: QWTensorValue(region: cReg, offset: 65536, shape: QWTensorShape([4, 512], .i32)),
            candState: QWTensorValue(region: cReg, offset: 98304, shape: QWTensorShape([8], .i32)),
            candPools: QWTensorValue(region: cReg, offset: 131072, shape: QWTensorShape([4, 10240], .i32)),
            candU: QWTensorValue(region: cReg, offset: 4194304, shape: QWTensorShape([4, 10240], .i32)),
            maxBatchRows: 4, clusterStride: csr.maxClusters, candCap: 10240)
        var prm = QWCoarsePoolIndex.SelectParams(
            T: 3, pos0: totalTokens - 3, indexDim: 128, budgetBlocks: mla.topKPools,
            windowBlocks: 0, hasSink: false, writeCount: true, listStride: mla.listStride,
            rowStep: 1, maxRow: 3, clusterStride: csr.maxClusters, probeClusters: 320,
            candCap: 10240, qChannels: qChannels, idxHeads: 32)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let numClusters = (numPools / 1024) * 64
        guard let (cbS, encS) = hal.begin() else { XCTFail("hal.begin"); return }
        let plen = MemoryLayout<QWCoarsePoolIndex.SelectParams>.stride
        encS.setComputePipelineState(psoCoarse)
        mla.mlaQFullScratch.bind(encS, index: 0)
        mla.idxWeightsScratch.bind(encS, index: 1)
        csr.centroids.bind(encS, index: 2)
        csr.clusterSpans.bind(encS, index: 3)
        cScratch.cScore.bind(encS, index: 4)
        cScratch.uHist.bind(encS, index: 5)
        encS.setBytes(&prm, length: plen, index: 6)
        encS.dispatchThreadgroups(MTLSize(width: (numClusters + 63) / 64, height: 3, depth: 1), threadsPerThreadgroup: tg)
        QWCoarsePoolIndex.encodeHistCollectScoreFinalize(
            encS, pipelines: pipes, csr: csr, scratch: cScratch,
            blockList: mla.blockListScratch, blockCount: mla.blockCountScratch, params: &prm,
            numClusters: numClusters, maxMiddlePools: numPools,
            encodeScoreCands: { enc, maxCand, p in
                enc.setComputePipelineState(psoCands)
                mla.mlaQFullScratch.bind(enc, index: 0)
                mla.idxWeightsScratch.bind(enc, index: 1)
                mla.layers[0].pooledKeys.bind(enc, index: 2)
                cScratch.candState.bind(enc, index: 3)
                cScratch.candPools.bind(enc, index: 4)
                cScratch.candU.bind(enc, index: 5)
                enc.setBytes(&p, length: plen, index: 6)
                enc.dispatchThreadgroups(MTLSize(width: (maxCand + 63) / 64, height: 3, depth: 1), threadsPerThreadgroup: tg)
            },
            encodeFlatFallback: { enc, p in
                enc.setComputePipelineState(psoFallback)
                mla.mlaQFullScratch.bind(enc, index: 0)
                mla.idxWeightsScratch.bind(enc, index: 1)
                mla.layers[0].pooledKeys.bind(enc, index: 2)
                cScratch.candState.bind(enc, index: 3)
                mla.blockListScratch.bind(enc, index: 4)
                mla.blockCountScratch.bind(enc, index: 5)
                enc.setBytes(&p, length: plen, index: 6)
                enc.dispatchThreadgroups(MTLSize(width: 3, height: 1, depth: 1), threadsPerThreadgroup: tg)
            }
        )
        hal.end(cbS, encS)
        cbS.waitUntilCompleted()
        XCTAssertNil(cbS.error)

        let cntPtr = basePtr.advanced(by: mla.blockCountScratch.offset).assumingMemoryBound(to: Int32.self)
        let lstPtr = basePtr.advanced(by: mla.blockListScratch.offset).assumingMemoryBound(to: Int32.self)
        for (r, np) in needles.enumerated() {
            let cnt = Int(cntPtr[r])
            XCTAssertEqual(cnt, mla.topKPools)
            let selected = Array(UnsafeBufferPointer(start: lstPtr + r * mla.listStride, count: cnt))
            XCTAssertTrue(selected.contains(Int32(np)), "MLA coarse missed needle \(np) at row \(r)")
            XCTAssertTrue(selected.contains(Int32(np + 1)), "MLA coarse missed needle \(np + 1) at row \(r)")
        }
        let snap = mla.captureIndexerState(position: 8192)
        XCTAssertTrue(mla.restoreIndexerState(snap, position: 8192))
    }

    func testPairedPerfAB_LSA_and_MLA() throws {
        guard ProcessInfo.processInfo.environment["QW_PERF_GATE"] == "1" else {
            throw XCTSkip("Set QW_PERF_GATE=1 to run paired interleaved LSA & MLA coarse A/B benchmarks")
        }
        let maxT = 1_048_576
        let rungs = [65_536, 131_072, 262_144, 524_288, 1_048_576]
        let wardOp = try Self.makeLSA(maxContext: maxT + 1024, budget: 2048, clusterBlock: 512)
        let emOp = try Self.makeLSA(maxContext: maxT + 1024, budget: 2048, clusterBlock: 4096)
        let chunk = 512
        let q = QWAttnPagedPrefillTests.synthQGate(n: chunk, seed: 0x3333)
        let k = QWAttnPagedPrefillTests.synthKV(n: chunk, seed: 0x4444)
        let qb = Self.hal.device.makeBuffer(bytes: q, length: q.count * 2, options: .storageModeShared)!
        let kb = Self.hal.device.makeBuffer(bytes: k, length: k.count * 2, options: .storageModeShared)!
        let qTV = QWTensorValue(region: QWMRegion(buffer: qb, base: 0, length: qb.length),
                                offset: 0, shape: QWTensorShape([chunk * Self.geometry.qProjOut], .f16))
        let kTV = QWTensorValue(region: QWMRegion(buffer: kb, base: 0, length: kb.length),
                                offset: 0, shape: QWTensorShape([chunk * Self.geometry.kvChannels], .f16))
        var filled = 0
        func fillBoth(to T: Int) {
            while filled < T {
                guard let (cb, enc) = Self.hal.begin() else { return }
                var n = 0
                while filled < T && n < 64 {
                    wardOp.encodeIndexAndCluster(enc, qgate: qTV, kIn: kTV, fullAttnLayer: 0, pos0: filled, rows: chunk)
                    emOp.encodeIndexAndCluster(enc, qgate: qTV, kIn: kTV, fullAttnLayer: 0, pos0: filled, rows: chunk)
                    enc.memoryBarrier(scope: .buffers)
                    filled += chunk; n += 1
                }
                Self.hal.end(cb, enc)
                cb.waitUntilCompleted()
            }
        }
        func gpuMs(_ hal: QWHal, iters: Int, _ body: (any MTLComputeCommandEncoder) -> Void) -> Double {
            guard let (cb, enc) = hal.begin() else { return .nan }
            for _ in 0..<iters { body(enc); enc.memoryBarrier(scope: .buffers) }
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            return (cb.gpuEndTime - cb.gpuStartTime) * 1e3 / Double(iters)
        }
        for T in rungs {
            fillBoth(to: T)
            var wardDec = [Double](), emDec = [Double](), wardPre = [Double](), emPre = [Double]()
            for r in 0..<8 {
                if r % 2 == 0 {
                    wardDec.append(gpuMs(Self.hal, iters: 20) { wardOp.encodeSelectBlocks($0, fullAttnLayer: 0, pos0: T - 1, rows: 1) })
                    emDec.append(gpuMs(Self.hal, iters: 20) { emOp.encodeSelectBlocks($0, fullAttnLayer: 0, pos0: T - 1, rows: 1) })
                    wardPre.append(gpuMs(Self.hal, iters: 5) { wardOp.encodeSelectBlocks($0, fullAttnLayer: 0, pos0: T - 512, rows: 512, tiled32: true) })
                    emPre.append(gpuMs(Self.hal, iters: 5) { emOp.encodeSelectBlocks($0, fullAttnLayer: 0, pos0: T - 512, rows: 512, tiled32: true) })
                } else {
                    emDec.append(gpuMs(Self.hal, iters: 20) { emOp.encodeSelectBlocks($0, fullAttnLayer: 0, pos0: T - 1, rows: 1) })
                    wardDec.append(gpuMs(Self.hal, iters: 20) { wardOp.encodeSelectBlocks($0, fullAttnLayer: 0, pos0: T - 1, rows: 1) })
                    emPre.append(gpuMs(Self.hal, iters: 5) { emOp.encodeSelectBlocks($0, fullAttnLayer: 0, pos0: T - 512, rows: 512, tiled32: true) })
                    wardPre.append(gpuMs(Self.hal, iters: 5) { wardOp.encodeSelectBlocks($0, fullAttnLayer: 0, pos0: T - 512, rows: 512, tiled32: true) })
                }
            }
            let md = { (xs: [Double]) -> Double in let s = xs.sorted(); return s[s.count / 2] }
            print(String(format: "LSA-PAIRED T=%7d | dec ward=%.3f(p50=%.3f)ms em=%.3f(p50=%.3f)ms (%.2fx) | pre512 ward=%.3f(p50=%.3f)ms em=%.3f(p50=%.3f)ms (%.2fx)",
                         T, wardDec.min()!, md(wardDec), emDec.min()!, md(emDec), wardDec.min()! / emDec.min()!,
                         wardPre.min()!, md(wardPre), emPre.min()!, md(emPre), wardPre.min()! / emPre.min()!))
        }

        let geo = try Self.mlaCoarseGeometry()
        let mlaHal = try QWHal(geometry: geo)
        let mla = try QWMLAAttnOperator(
            hal: mlaHal, geometry: geo, numAttnLayers: 1,
            maxContext: maxT + 1024, maxPrefillChunk: 32,
            lsaConfig: QWLSAConfig(mode: .svd, budget: 2048))
        let csr = try XCTUnwrap(mla.layers[0].coarseCSR)
        let mlaBase = mla.region.pointer
        let pkPtr = mlaBase.advanced(by: mla.layers[0].pooledKeys.offset).assumingMemoryBound(to: UInt16.self)
        let qfPtr = mlaBase.advanced(by: mla.mlaQFullScratch.offset).assumingMemoryBound(to: UInt16.self)
        let iwPtr = mlaBase.advanced(by: mla.idxWeightsScratch.offset).assumingMemoryBound(to: Float.self)
        var s: UInt32 = 0x1357_9BDF
        for i in 0..<(mla.maxBlocks * 128) {
            s = s &* 1664525 &+ 1013904223
            pkPtr[i] = Float16(Float(Int32(bitPattern: s)) * 4e-10).bitPattern
        }
        for i in 0..<(32 * mla.qFullStride) {
            s = s &* 1664525 &+ 1013904223
            qfPtr[i] = Float16(Float(Int32(bitPattern: s)) * 4e-10).bitPattern
        }
        for i in 0..<(32 * 32) {
            s = s &* 1664525 &+ 1013904223
            iwPtr[i] = 0.02 + Float(s & 0xFF) * 0.001
        }
        let pipes = try QWCoarsePoolIndex.Pipelines(hal: mlaHal)
        let psoSplitScore = try mlaHal.pipeline("qwg_mla_indexer_score_mma")
        let psoSplitSel = try mlaHal.pipeline("qwg_mla_indexer_select")
        let psoMlaCoarse = try mlaHal.pipeline("qwg_emeans_mla_score_coarse")
        let psoMlaCands = try mlaHal.pipeline("qwg_emeans_mla_score_cands")
        let psoMlaFallback = try mlaHal.pipeline("qwg_emeans_mla_flat_fallback")
        let cBuf = mlaHal.device.makeBuffer(length: 16 << 20, options: .storageModeShared)!
        memset(cBuf.contents(), 0, cBuf.length)
        let cReg = QWMRegion(buffer: cBuf, base: 0, length: cBuf.length)
        let cScratch = QWCoarsePoolIndex.Scratch(
            cScore: QWTensorValue(region: cReg, offset: 0, shape: QWTensorShape([32, csr.maxClusters], .i32)),
            uHist: QWTensorValue(region: cReg, offset: 2_097_152, shape: QWTensorShape([32, 512], .i32)),
            candState: QWTensorValue(region: cReg, offset: 2_228_224, shape: QWTensorShape([64], .i32)),
            candPools: QWTensorValue(region: cReg, offset: 2_359_296, shape: QWTensorShape([32, 10240], .i32)),
            candU: QWTensorValue(region: cReg, offset: 3_670_016, shape: QWTensorShape([32, 10240], .i32)),
            maxBatchRows: 32, clusterStride: csr.maxClusters, candCap: 10240)
        guard let (cbM, encM) = mlaHal.begin() else { return }
        QWCoarsePoolIndex.encodeClusterMacros(
            encM, pso: pipes.cluster1024, pooledKeys: mla.layers[0].pooledKeys,
            csr: csr, startMacro: 0, numMacros: csr.maxMacros, indexDim: 128)
        mlaHal.end(cbM, encM)
        cbM.waitUntilCompleted()

        let qChannels = geo.qHeads * mla.nopeDim
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        func runFlatMLA(_ enc: any MTLComputeCommandEncoder, pos0: Int, rows: Int) {
            var p: [UInt32] = [UInt32(rows), UInt32(pos0), UInt32(qChannels), 32, 128,
                               UInt32(mla.topKPools), UInt32(mla.listStride), UInt32(mla.maxBlocks)]
            mla.mlaQFullScratch.bind(enc, index: 0)
            mla.idxWeightsScratch.bind(enc, index: 1)
            mla.layers[0].pooledKeys.bind(enc, index: 2)
            mla.scoreScratch.bind(enc, index: 3)
            mla.blockListScratch.bind(enc, index: 4)
            mla.blockCountScratch.bind(enc, index: 5)
            enc.setBytes(&p, length: p.count * 4, index: 6)
            enc.setComputePipelineState(psoSplitScore)
            enc.dispatchThreadgroups(MTLSize(width: ((pos0 + rows) / 4 + 63) / 64, height: rows, depth: 1), threadsPerThreadgroup: tg)
            enc.setComputePipelineState(psoSplitSel)
            enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: tg)
        }
        func runEmeansMLA(_ enc: any MTLComputeCommandEncoder, pos0: Int, rows: Int) {
            let nClusters = (((pos0 + rows) / 4) >> 10) * 64
            let maxPools = (pos0 + rows) / 4
            var ep = QWCoarsePoolIndex.SelectParams(
                T: rows, pos0: pos0, indexDim: 128, budgetBlocks: mla.topKPools,
                windowBlocks: 0, hasSink: false, writeCount: true, listStride: mla.listStride,
                rowStep: 1, maxRow: rows, clusterStride: csr.maxClusters, probeClusters: 320,
                candCap: 10240, qChannels: qChannels, idxHeads: 32)
            let plen = MemoryLayout<QWCoarsePoolIndex.SelectParams>.stride
            enc.setComputePipelineState(psoMlaCoarse)
            mla.mlaQFullScratch.bind(enc, index: 0)
            mla.idxWeightsScratch.bind(enc, index: 1)
            csr.centroids.bind(enc, index: 2)
            csr.clusterSpans.bind(enc, index: 3)
            cScratch.cScore.bind(enc, index: 4)
            cScratch.uHist.bind(enc, index: 5)
            enc.setBytes(&ep, length: plen, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: (nClusters + 63) / 64, height: rows, depth: 1), threadsPerThreadgroup: tg)
            QWCoarsePoolIndex.encodeHistCollectScoreFinalize(
                enc, pipelines: pipes, csr: csr, scratch: cScratch,
                blockList: mla.blockListScratch, blockCount: mla.blockCountScratch, params: &ep,
                numClusters: nClusters, maxMiddlePools: maxPools,
                encodeScoreCands: { e, maxCand, prm in
                    e.setComputePipelineState(psoMlaCands)
                    mla.mlaQFullScratch.bind(e, index: 0)
                    mla.idxWeightsScratch.bind(e, index: 1)
                    mla.layers[0].pooledKeys.bind(e, index: 2)
                    cScratch.candState.bind(e, index: 3)
                    cScratch.candPools.bind(e, index: 4)
                    cScratch.candU.bind(e, index: 5)
                    e.setBytes(&prm, length: plen, index: 6)
                    e.dispatchThreadgroups(MTLSize(width: (maxCand + 63) / 64, height: rows, depth: 1), threadsPerThreadgroup: tg)
                },
                encodeFlatFallback: { e, prm in
                    e.setComputePipelineState(psoMlaFallback)
                    mla.mlaQFullScratch.bind(e, index: 0)
                    mla.idxWeightsScratch.bind(e, index: 1)
                    mla.layers[0].pooledKeys.bind(e, index: 2)
                    cScratch.candState.bind(e, index: 3)
                    mla.blockListScratch.bind(e, index: 4)
                    mla.blockCountScratch.bind(e, index: 5)
                    e.setBytes(&prm, length: plen, index: 6)
                    e.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: tg)
                }
            )
        }
        for T in rungs {
            var flatDec = [Double](), emDec = [Double](), flatPre = [Double](), emPre = [Double]()
            for r in 0..<8 {
                if r % 2 == 0 {
                    flatDec.append(gpuMs(mlaHal, iters: 10) { runFlatMLA($0, pos0: T - 1, rows: 1) })
                    emDec.append(gpuMs(mlaHal, iters: 10) { runEmeansMLA($0, pos0: T - 1, rows: 1) })
                    flatPre.append(gpuMs(mlaHal, iters: 3) { runFlatMLA($0, pos0: T - 32, rows: 32) })
                    emPre.append(gpuMs(mlaHal, iters: 3) { runEmeansMLA($0, pos0: T - 32, rows: 32) })
                } else {
                    emDec.append(gpuMs(mlaHal, iters: 10) { runEmeansMLA($0, pos0: T - 1, rows: 1) })
                    flatDec.append(gpuMs(mlaHal, iters: 10) { runFlatMLA($0, pos0: T - 1, rows: 1) })
                    emPre.append(gpuMs(mlaHal, iters: 3) { runEmeansMLA($0, pos0: T - 32, rows: 32) })
                    flatPre.append(gpuMs(mlaHal, iters: 3) { runFlatMLA($0, pos0: T - 32, rows: 32) })
                }
            }
            let md = { (xs: [Double]) -> Double in let s = xs.sorted(); return s[s.count / 2] }
            print(String(format: "MLA-PAIRED T=%7d | dec flat=%.3f(p50=%.3f)ms em=%.3f(p50=%.3f)ms (%.2fx) | pre32 flat=%.3f(p50=%.3f)ms em=%.3f(p50=%.3f)ms (%.2fx)",
                         T, flatDec.min()!, md(flatDec), emDec.min()!, md(emDec), flatDec.min()! / emDec.min()!,
                         flatPre.min()!, md(flatPre), emPre.min()!, md(emPre), flatPre.min()! / emPre.min()!))
        }
    }
}
