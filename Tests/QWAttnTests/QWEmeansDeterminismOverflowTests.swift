// QWEmeansDeterminismOverflowTests.swift — GPU tests for deterministic cluster tie-breaking,
// on-device exact flat candidate-overflow fallback, <320K Stage-1 bypass, and opt-in MLA coarse mode.

import XCTest
import Metal
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWQuant

final class QWEmeansDeterminismOverflowTests: XCTestCase {
    static let hal: QWHal = QWLSATests.hal
    static let geometry = QWLSATests.geometry

    private static func makeLSA(maxContext: Int, budget: Int, clusterBlock: Int) throws -> QWLSAOperator {
        let cfg = QWLSAConfig(mode: .svd, budget: budget, clusterBlock: clusterBlock)
        let inDim = geometry.lsaInputDim
        let basis = QWLSASidecar.synthesizeDeterministicOrthogonalBasis(inDim: inDim, indexDim: 64, seed: 202)
        let lw = QWLSAOperator.LayerWeights(physicalLayer: 3, wqIdxF16: basis, wkIdxF16: basis)
        return try QWLSAOperator(
            hal: hal, geometry: geometry, config: cfg, indexDim: 64,
            layerWeights: [lw], maxContext: maxContext, maxPrefillChunk: 128)
    }

    func testEqualScoreTieDeterminismAcrossSimdgroupsAndChunks() throws {
        let numMacros = 6
        let numClusters = numMacros * 64 // 384 clusters -> spans 2 chunks of 256 in collect
        let numPools = numMacros * 1024  // 6144 pools
        let pos0 = numPools * 4 + 512
        let hal = Self.hal
        let pipes = try QWCoarsePoolIndex.Pipelines(hal: hal)
        let psoCoarse = try hal.pipeline("qwg_emeans_lsa_score_coarse")
        let psoCands = try hal.pipeline("qwg_emeans_lsa_score_cands")
        let psoFallback = try hal.pipeline("qwg_emeans_lsa_flat_fallback")

        let buf = hal.device.makeBuffer(length: 16 << 20, options: .storageModeShared)!
        memset(buf.contents(), 0, buf.length)
        let reg = QWMRegion(buffer: buf, base: 0, length: buf.length)
        let qIdx = QWTensorValue(region: reg, offset: 0, shape: QWTensorShape([1, 64], .f16))
        let pooled = QWTensorValue(region: reg, offset: 4096, shape: QWTensorShape([numPools + 256, 64], .f16))
        let centroids = QWTensorValue(region: reg, offset: 1_048_576, shape: QWTensorShape([numClusters, 64], .f16))
        let members = QWTensorValue(region: reg, offset: 1_179_648, shape: QWTensorShape([numPools / 2], .i32))
        let spans = QWTensorValue(region: reg, offset: 1_245_184, shape: QWTensorShape([numClusters], .i32))
        let cScore = QWTensorValue(region: reg, offset: 1_310_720, shape: QWTensorShape([1, numClusters], .i32))
        let uHist = QWTensorValue(region: reg, offset: 1_376_256, shape: QWTensorShape([1, 512], .i32))
        let candState = QWTensorValue(region: reg, offset: 1_441_792, shape: QWTensorShape([2], .i32))
        let candPools = QWTensorValue(region: reg, offset: 1_507_328, shape: QWTensorShape([1, 4096], .i32))
        let candU = QWTensorValue(region: reg, offset: 1_638_400, shape: QWTensorShape([1, 4096], .i32))
        let blockList = QWTensorValue(region: reg, offset: 1_769_472, shape: QWTensorShape([1, 512], .i32))
        let blockCount = QWTensorValue(region: reg, offset: 1_835_008, shape: QWTensorShape([1], .i32))

        let ptr = reg.pointer
        let qPtr = ptr.advanced(by: qIdx.offset).assumingMemoryBound(to: UInt16.self)
        qPtr[0] = Float16(1.0).bitPattern
        let cPtr = ptr.advanced(by: centroids.offset).assumingMemoryBound(to: UInt16.self)
        let mPtr = ptr.advanced(by: members.offset).assumingMemoryBound(to: UInt16.self)
        let sPtr = ptr.advanced(by: spans.offset).assumingMemoryBound(to: UInt16.self)
        let pPtr = ptr.advanced(by: pooled.offset).assumingMemoryBound(to: UInt16.self)

        // Give EVERY cluster the exact same centroid score (0.5) so all 384 clusters tie at thresh16.
        for b in 0..<numClusters {
            cPtr[b * 64] = Float16(0.5).bitPattern
            let localOff = (b % 64) * 16
            sPtr[b * 2] = UInt16(localOff)
            sPtr[b * 2 + 1] = 16
        }
        for m in 0..<numMacros {
            for i in 0..<1024 {
                mPtr[m * 1024 + i] = UInt16(i)
                let globalPool = m * 1024 + i
                // Distinct deterministic pool scores inside clusters 0..<70 vs 70..<384.
                let clusterId = m * 64 + (i / 16)
                let val: Float = clusterId < 70 ? (0.8 - Float(globalPool) * 1e-5) : 0.99
                pPtr[globalPool * 64] = Float16(val).bitPattern
            }
        }

        let csr = QWCoarsePoolIndex.LayerCSR(centroids: centroids, memberPools: members, clusterSpans: spans, maxMacros: numMacros)
        let scratch = QWCoarsePoolIndex.Scratch(
            cScore: cScore, uHist: uHist, candState: candState, candPools: candPools, candU: candU,
            maxBatchRows: 1, clusterStride: numClusters, candCap: 4096)

        var reference: [Int32]? = nil
        for _ in 0..<6 {
            var prm = QWCoarsePoolIndex.SelectParams(
                T: 1, pos0: pos0, indexDim: 64, budgetBlocks: 64, windowBlocks: 128,
                hasSink: true, writeCount: true, listStride: 512, rowStep: 1, maxRow: 1,
                clusterStride: numClusters, probeClusters: 70, candCap: 4096)
            guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
            let plen = MemoryLayout<QWCoarsePoolIndex.SelectParams>.stride
            let tpg = MTLSize(width: 256, height: 1, depth: 1)
            enc.setComputePipelineState(psoCoarse)
            qIdx.bind(enc, index: 0)
            csr.centroids.bind(enc, index: 1)
            csr.clusterSpans.bind(enc, index: 2)
            scratch.cScore.bind(enc, index: 3)
            scratch.uHist.bind(enc, index: 4)
            enc.setBytes(&prm, length: plen, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: (numClusters + 255) / 256, height: 1, depth: 1), threadsPerThreadgroup: tpg)
            QWCoarsePoolIndex.encodeHistCollectScoreFinalize(
                enc, pipelines: pipes, csr: csr, scratch: scratch,
                blockList: blockList, blockCount: blockCount, params: &prm,
                numClusters: numClusters, maxMiddlePools: numPools,
                encodeScoreCands: { enc, maxCand, p in
                    enc.setComputePipelineState(psoCands)
                    qIdx.bind(enc, index: 0)
                    pooled.bind(enc, index: 1)
                    scratch.candState.bind(enc, index: 2)
                    scratch.candPools.bind(enc, index: 3)
                    scratch.candU.bind(enc, index: 4)
                    enc.setBytes(&p, length: plen, index: 5)
                    enc.dispatchThreadgroups(MTLSize(width: (maxCand + 255) / 256, height: 1, depth: 1), threadsPerThreadgroup: tpg)
                },
                encodeFlatFallback: { enc, p in
                    enc.setComputePipelineState(psoFallback)
                    qIdx.bind(enc, index: 0)
                    pooled.bind(enc, index: 1)
                    scratch.candState.bind(enc, index: 2)
                    blockList.bind(enc, index: 3)
                    blockCount.bind(enc, index: 4)
                    enc.setBytes(&p, length: plen, index: 5)
                    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tpg)
                }
            )
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)

            let cnt = Int(ptr.advanced(by: blockCount.offset).assumingMemoryBound(to: UInt32.self)[0])
            let lst = Array(UnsafeBufferPointer(
                start: ptr.advanced(by: blockList.offset).assumingMemoryBound(to: Int32.self), count: cnt))
            if let ref = reference {
                XCTAssertEqual(lst, ref, "equal-score coarse cluster ties must resolve bit-identically across runs")
            } else {
                reference = lst
                // Clusters 0..<70 correspond to pools 0..<1120; higher-id tied clusters (70+) must NOT leak in.
                for pId in lst.dropFirst().prefix(64) {
                    XCTAssertLessThan(pId, 1120, "ascending cluster tie-break must pick clusters 0..<70 deterministically")
                }
            }
        }
    }

    func testCandidateOverflowFallsBackToExactFlatTopKWithoutLoss() throws {
        let numMacros = 2
        let numClusters = numMacros * 64
        let numPools = numMacros * 1024 // 2048 pools
        let pos0 = (numPools + 128) * 4
        let smallCandCap = 128 // Force overflow: 70 clusters * 16 pools = 1120 > 128!
        let hal = Self.hal
        let pipes = try QWCoarsePoolIndex.Pipelines(hal: hal)
        let psoCoarse = try hal.pipeline("qwg_emeans_lsa_score_coarse")
        let psoCands = try hal.pipeline("qwg_emeans_lsa_score_cands")
        let psoFallback = try hal.pipeline("qwg_emeans_lsa_flat_fallback")

        let buf = hal.device.makeBuffer(length: 8 << 20, options: .storageModeShared)!
        memset(buf.contents(), 0, buf.length)
        let reg = QWMRegion(buffer: buf, base: 0, length: buf.length)
        let qIdx = QWTensorValue(region: reg, offset: 0, shape: QWTensorShape([1, 64], .f16))
        let pooled = QWTensorValue(region: reg, offset: 4096, shape: QWTensorShape([numPools + 256, 64], .f16))
        let centroids = QWTensorValue(region: reg, offset: 1_048_576, shape: QWTensorShape([numClusters, 64], .f16))
        let members = QWTensorValue(region: reg, offset: 1_114_112, shape: QWTensorShape([numPools / 2], .i32))
        let spans = QWTensorValue(region: reg, offset: 1_179_648, shape: QWTensorShape([numClusters], .i32))
        let cScore = QWTensorValue(region: reg, offset: 1_245_184, shape: QWTensorShape([1, numClusters], .i32))
        let uHist = QWTensorValue(region: reg, offset: 1_310_720, shape: QWTensorShape([1, 512], .i32))
        let candState = QWTensorValue(region: reg, offset: 1_376_256, shape: QWTensorShape([2], .i32))
        let candPools = QWTensorValue(region: reg, offset: 1_441_792, shape: QWTensorShape([1, smallCandCap], .i32))
        let candU = QWTensorValue(region: reg, offset: 1_507_328, shape: QWTensorShape([1, smallCandCap], .i32))
        let blockList = QWTensorValue(region: reg, offset: 1_572_864, shape: QWTensorShape([1, 512], .i32))
        let blockCount = QWTensorValue(region: reg, offset: 1_638_400, shape: QWTensorShape([1], .i32))

        let ptr = reg.pointer
        ptr.advanced(by: qIdx.offset).assumingMemoryBound(to: UInt16.self)[0] = Float16(1.0).bitPattern
        let cPtr = ptr.advanced(by: centroids.offset).assumingMemoryBound(to: UInt16.self)
        let mPtr = ptr.advanced(by: members.offset).assumingMemoryBound(to: UInt16.self)
        let sPtr = ptr.advanced(by: spans.offset).assumingMemoryBound(to: UInt16.self)
        let pPtr = ptr.advanced(by: pooled.offset).assumingMemoryBound(to: UInt16.self)

        for b in 0..<numClusters {
            cPtr[b * 64] = Float16(0.75).bitPattern
            sPtr[b * 2] = UInt16((b % 64) * 16)
            sPtr[b * 2 + 1] = 16
        }
        let lateNeedles = Array(1900..<1916)
        for m in 0..<numMacros {
            for i in 0..<1024 {
                mPtr[m * 1024 + i] = UInt16(i)
                let pool = m * 1024 + i
                let score: Float = lateNeedles.contains(pool) ? (2.0 + Float(pool - 1900) * 0.01) : 0.1
                pPtr[pool * 64] = Float16(score).bitPattern
            }
        }

        let csr = QWCoarsePoolIndex.LayerCSR(centroids: centroids, memberPools: members, clusterSpans: spans, maxMacros: numMacros)
        let scratch = QWCoarsePoolIndex.Scratch(
            cScore: cScore, uHist: uHist, candState: candState, candPools: candPools, candU: candU,
            maxBatchRows: 1, clusterStride: numClusters, candCap: smallCandCap)
        var prm = QWCoarsePoolIndex.SelectParams(
            T: 1, pos0: pos0, indexDim: 64, budgetBlocks: 32, windowBlocks: 128,
            hasSink: true, writeCount: true, listStride: 512, rowStep: 1, maxRow: 1,
            clusterStride: numClusters, probeClusters: 70, candCap: smallCandCap)

        guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
        let plen = MemoryLayout<QWCoarsePoolIndex.SelectParams>.stride
        let tpg = MTLSize(width: 256, height: 1, depth: 1)
        enc.setComputePipelineState(psoCoarse)
        qIdx.bind(enc, index: 0)
        csr.centroids.bind(enc, index: 1)
        csr.clusterSpans.bind(enc, index: 2)
        scratch.cScore.bind(enc, index: 3)
        scratch.uHist.bind(enc, index: 4)
        enc.setBytes(&prm, length: plen, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tpg)
        QWCoarsePoolIndex.encodeHistCollectScoreFinalize(
            enc, pipelines: pipes, csr: csr, scratch: scratch,
            blockList: blockList, blockCount: blockCount, params: &prm,
            numClusters: numClusters, maxMiddlePools: numPools,
            encodeScoreCands: { enc, maxCand, p in
                enc.setComputePipelineState(psoCands)
                qIdx.bind(enc, index: 0)
                pooled.bind(enc, index: 1)
                scratch.candState.bind(enc, index: 2)
                scratch.candPools.bind(enc, index: 3)
                scratch.candU.bind(enc, index: 4)
                enc.setBytes(&p, length: plen, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: (maxCand + 255) / 256, height: 1, depth: 1), threadsPerThreadgroup: tpg)
            },
            encodeFlatFallback: { enc, p in
                enc.setComputePipelineState(psoFallback)
                qIdx.bind(enc, index: 0)
                pooled.bind(enc, index: 1)
                scratch.candState.bind(enc, index: 2)
                blockList.bind(enc, index: 3)
                blockCount.bind(enc, index: 4)
                enc.setBytes(&p, length: plen, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tpg)
            }
        )
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let cnt = Int(ptr.advanced(by: blockCount.offset).assumingMemoryBound(to: UInt32.self)[0])
        XCTAssertEqual(cnt, 1 + 32 + 128)
        let selected = Set(UnsafeBufferPointer(
            start: ptr.advanced(by: blockList.offset).assumingMemoryBound(to: Int32.self), count: cnt))
        for np in lateNeedles {
            XCTAssertTrue(selected.contains(Int32(np)), "on-device overflow fallback must recover late needle \(np) past candCap")
        }
        // candState must be restored to zero-at-rest after fallback completes.
        let cs = ptr.advanced(by: candState.offset).assumingMemoryBound(to: UInt32.self)
        XCTAssertEqual(cs[0], 0)
        XCTAssertEqual(cs[1], 0)
    }

    func testBelowPruneThresholdBypassesStage1AndMLADefaultsExact() throws {
        let shortBg = QWCoarsePoolIndex.batchGeometry(
            pos0: 16_384, units: 1, budgetBlocks: 512, windowBlocks: 128,
            hasSink: true, writeCount: true, probeClusters: 320)
        XCTAssertEqual(shortBg.selectionMode, .exactFlat)
        XCTAssertFalse(shortBg.shouldPrune)

        let longBg = QWCoarsePoolIndex.batchGeometry(
            pos0: 65_536, units: 1, budgetBlocks: 512, windowBlocks: 128,
            hasSink: true, writeCount: true, probeClusters: 320)
        XCTAssertEqual(longBg.selectionMode, .approximateCoarse)
        XCTAssertTrue(longBg.shouldPrune)

        let op = try Self.makeLSA(maxContext: 20_480, budget: 2048, clusterBlock: 4096)
        let ls = op.layers[0]
        let ptr = op.region.pointer
        let pPtr = ptr.advanced(by: ls.pooledKeys.offset).assumingMemoryBound(to: UInt16.self)
        let qPtr = ptr.advanced(by: op.qIdxScratch.offset).assumingMemoryBound(to: UInt16.self)
        qPtr[0] = Float16(1.0).bitPattern
        for m in 0..<4096 {
            pPtr[m * 64] = Float16(Float(m % 997) * 0.001).bitPattern
        }
        func runSelect() -> [Int32] {
            guard let (cb, enc) = Self.hal.begin() else { return [] }
            op.encodeSelectBlocks(enc, fullAttnLayer: 0, pos0: 16_383, rows: 1)
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            let cnt = Int(ptr.advanced(by: op.blockCountScratch.offset).assumingMemoryBound(to: UInt32.self)[0])
            return Array(UnsafeBufferPointer(
                start: ptr.advanced(by: op.blockListScratch.offset).assumingMemoryBound(to: Int32.self), count: cnt))
        }
        let clean = runSelect()
        // Poison centroids with max-finite half values: Stage-1 bypass must ignore centroids completely at 16K!
        let cPtr = ptr.advanced(by: ls.centroids.offset).assumingMemoryBound(to: UInt16.self)
        for i in 0..<(op.maxBuckets * 64) { cPtr[i] = 0x7BFF }
        let poisoned = runSelect()
        XCTAssertEqual(poisoned, clean, "<41K LSA must bypass Stage-1 centroid scoring completely")

        if ProcessInfo.processInfo.environment["QW_MLA_EMEANS"] == nil {
            let mlaCfg = """
            {"text_config":{"model_type":"glm5_next_text","hidden_size":4096,"intermediate_size":12288,"vocab_size":154880,"num_hidden_layers":4,"layer_types":["linear_attention","linear_attention","linear_attention","deepseek_sparse_attention"],"rms_norm_eps":1e-5,"max_position_embeddings":65536,"num_attention_heads":64,"num_key_value_heads":64,"qk_head_dim":256,"qk_nope_head_dim":192,"qk_rope_head_dim":64,"v_head_dim":256,"kv_lora_rank":512,"q_lora_rank":1536,"mla_use_nope":true,"partial_rotary_factor":1.0,"linear_attn_config":{"Linear_type":"kda","gate_lower_bound":-5.0,"head_dim":128,"num_heads":64,"short_conv_kernel_size":4},"n_routed_experts":288,"num_experts_per_tok":8,"moe_intermediate_size":2048,"first_k_dense_replace":3,"index_n_heads":32,"index_head_dim":128,"index_topk":2048,"index_kpool_compress":true,"index_kpool_always_select_tail":true,"index_share_for_mtp_iteration":true,"indexer_rope_interleave":true,"eos_token_id":154820}}
            """.data(using: .utf8)!
            let mlaGeo = QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: mlaCfg))
            let mlaHal = try QWHal(geometry: mlaGeo)
            let mlaOff = try QWMLAAttnOperator(
                hal: mlaHal, geometry: mlaGeo, numAttnLayers: 1, maxContext: 65_536, maxPrefillChunk: 32,
                lsaConfig: QWLSAConfig(mode: .off))
            XCTAssertFalse(mlaOff.emeansCoarseEnabled, "lsaMode=off must select exact flat MLA scoring")
            let mlaAuto = try QWMLAAttnOperator(
                hal: mlaHal, geometry: mlaGeo, numAttnLayers: 1, maxContext: 65_536, maxPrefillChunk: 32,
                lsaConfig: QWLSAConfig(mode: .auto))
            XCTAssertTrue(mlaAuto.emeansCoarseEnabled, "lsaMode=auto must enable MLA e-means coarse pruning (<41K flat bypass)")
        }
    }
}
