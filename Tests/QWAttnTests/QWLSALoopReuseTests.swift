// QWLSALoopReuseTests.swift — Tests for cross-loop LSA physical-layer sharing and block reuse.

import Foundation
import Metal
import XCTest
@testable import QWAttn
@testable import QWCore
import QWHAL
import QWAlloc
import QWQuant

private extension QWModelGeometry {
    func lsaLoopReusePlan(
        attnLayers: [Int],
        crossLoopReuse: Bool,
        stride: Int = 0
    ) -> QWLSALoopReusePlan {
        let plan = executionPlan
        let physLayers = attnLayers.map { v in
            plan.indices.contains(v) ? plan[v] : v
        }
        return QWLSALoopReusePlan.build(physicalLayers: physLayers, enabled: crossLoopReuse, stride: stride)
    }
}

final class QWLSALoopReuseTests: XCTestCase {
    private func makeLoopDescriptor(
        numLayers: Int = 24,
        loopStart: Int = 6,
        loopEnd: Int = 18,
        evalCount: Int = 3,
        fullInterval: Int = 6
    ) -> QWModelDescriptor {
        let layerTypes = (0..<numLayers).map { ($0 + 1) % fullInterval == 0 ? "full_attention" : "linear_attention" }
        return QWModelDescriptor(
            family: .senseNovaLoop, modelType: "sensenova_loop", capabilities: [],
            hidden: 256, vocab: 1024, denseIntermediate: 512,
            numLayers: numLayers, layerTypes: layerTypes, fullAttentionInterval: fullInterval,
            rmsEps: 1e-6, maxPositionEmbeddings: 65_536,
            qHeads: 8, kvHeads: 2, headDim: 64, attnOutputGate: true,
            partialRotaryFactor: 1.0, ropeTheta: 10_000.0,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: nil, expertsPerTok: nil, moeIntermediate: nil,
            sharedExpertIntermediate: nil, expertGateType: nil, hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 4, gdnVHeads: 4, gdnHeadDim: 64,
            gdnQKVChannels: 768, gdnZChannels: 256, gdnConvKernel: 4,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 0, mtpUseDedicatedEmbeddings: false, eosTokenId: 2,
            loopSettings: QWModelDescriptor.LoopSettings(start: loopStart, end: loopEnd, evalCount: evalCount)
        )
    }

    func testExecutionPlanLoopReuseMetadataAndEnvParsing() {
        let desc = makeLoopDescriptor(numLayers: 24, loopStart: 6, loopEnd: 18, evalCount: 3, fullInterval: 6)
        let geo = QWModelGeometry(descriptor: desc)
        let fullVisits = (0..<geo.numVirtualLayers).filter { geo.isVirtualLayerFullAttention($0) }
        XCTAssertEqual(fullVisits.map { geo.executionPlan[$0] }, [5, 11, 17, 11, 17, 11, 17, 23])

        let offPlan = geo.lsaLoopReusePlan(attnLayers: fullVisits, crossLoopReuse: false)
        XCTAssertFalse(offPlan.hasReuse)
        XCTAssertEqual(offPlan.reuseSlotCount, 0)
        XCTAssertEqual(offPlan.sourceForFullAttnLayer, Array(0..<8))

        let onPlan = geo.lsaLoopReusePlan(attnLayers: fullVisits, crossLoopReuse: true, stride: 0)
        XCTAssertTrue(onPlan.hasReuse)
        XCTAssertEqual(onPlan.sourceForFullAttnLayer, [0, 1, 2, 1, 2, 1, 2, 7])
        XCTAssertEqual(onPlan.isFollower, [false, false, false, true, true, true, true, false])
        XCTAssertEqual(onPlan.uniqueLeaderIndices, [0, 1, 2, 7])
        XCTAssertEqual(onPlan.reuseSlotCount, 2)
        XCTAssertEqual(onPlan.reuseCacheSlotForLeader[1], 0)
        XCTAssertEqual(onPlan.reuseCacheSlotForLeader[2], 1)
        XCTAssertNil(onPlan.reuseCacheSlotForLeader[0])
        XCTAssertNil(onPlan.reuseCacheSlotForLeader[7])

        // Configurable stride=2 recomputes every 2 visits across 4 loop passes.
        let stridePlan = QWLSALoopReusePlan.build(
            physicalLayers: [11, 11, 11, 11], enabled: true, stride: 2
        )
        XCTAssertEqual(stridePlan.sourceForFullAttnLayer, [0, 0, 2, 2])
        XCTAssertEqual(stridePlan.reuseSlotCount, 2)

        // Non-looped model stays untouched even with reuse enabled.
        let flatPlan = QWLSALoopReusePlan.build(physicalLayers: [0, 1, 2, 3], enabled: true)
        XCTAssertFalse(flatPlan.hasReuse)
        XCTAssertEqual(flatPlan.reuseSlotCount, 0)

        // Environment parsing and explicit CLI override precedence.
        XCTAssertEqual(QWLSAConfig.parseLoopShareEnv("1")?.enabled, true)
        XCTAssertEqual(QWLSAConfig.parseLoopShareEnv("1")?.stride, 0)
        XCTAssertEqual(QWLSAConfig.parseLoopShareEnv("stride:2")?.stride, 2)
        XCTAssertEqual(QWLSAConfig.parseLoopShareEnv("3")?.stride, 3)
        XCTAssertEqual(QWLSAConfig.parseLoopShareEnv("off")?.enabled, false)

        let autoEnv = QWLSAConfig(mode: .auto).resolvingLoopShareEnvironment([QWLSAConfig.loopShareEnvKey: "stride:2"])
        XCTAssertTrue(autoEnv.crossLoopReuse)
        XCTAssertEqual(autoEnv.loopShareStride, 2)

        let cliDisabled = QWLSAConfig(mode: .auto, crossLoopReuse: false)
            .resolvingLoopShareEnvironment([QWLSAConfig.loopShareEnvKey: "1"])
        XCTAssertFalse(cliDisabled.crossLoopReuse)
    }

    func testCrossLoopBlockReuseSharesInterleavedSelectionsAndRoundTripsSnapshot() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal device required")
        let geo = QWModelGeometry.qwen38_27b
        let hal = try QWHal(geometry: geo)
        let inDim = geo.lsaInputDim
        let physSeq = [5, 11, 17, 11, 17, 11, 17, 23]
        var basisCache: [Int: QWLSAOperator.LayerWeights] = [:]
        let weights = physSeq.map { phys -> QWLSAOperator.LayerWeights in
            if let cached = basisCache[phys] { return cached }
            let wq = QWLSASidecar.synthesizeDeterministicOrthogonalBasis(
                inDim: inDim, indexDim: 64, seed: UInt64(phys * 10 + 1))
            let wk = QWLSASidecar.synthesizeDeterministicOrthogonalBasis(
                inDim: inDim, indexDim: 64, seed: UInt64(phys * 10 + 2))
            let lw = QWLSAOperator.LayerWeights(physicalLayer: phys, wqIdxF16: wq, wkIdxF16: wk)
            basisCache[phys] = lw
            return lw
        }
        let opRef = try QWLSAOperator(
            hal: hal, geometry: geo,
            config: QWLSAConfig(mode: .svd, budget: 512, clusterBlock: 512, crossLoopReuse: false),
            layerWeights: weights, maxContext: 4096, maxPrefillChunk: 32
        )
        let opReuse = try QWLSAOperator(
            hal: hal, geometry: geo,
            config: QWLSAConfig(mode: .svd, budget: 512, clusterBlock: 512, crossLoopReuse: true),
            layerWeights: weights, maxContext: 4096, maxPrefillChunk: 32
        )
        XCTAssertLessThan(opReuse.region.buffer.length, opRef.region.buffer.length)
        XCTAssertLessThan(opReuse.fullIndexerStateBytes(position: 2056), opRef.fullIndexerStateBytes(position: 2056))
        XCTAssertTrue(opReuse.isSelectionFollower(fullAttnLayer: 3))
        XCTAssertFalse(opReuse.isSelectionFollower(fullAttnLayer: 1))
        XCTAssertEqual(opReuse.blockList(forFullAttnLayer: 3).offset, opReuse.blockList(forFullAttnLayer: 1).offset)
        XCTAssertNotEqual(opReuse.blockList(forFullAttnLayer: 2).offset, opReuse.blockList(forFullAttnLayer: 1).offset)

        let totalPos = 2052
        let qElems = totalPos * geo.qProjOut
        let kElems = totalPos * geo.kvChannels
        let sinLUT = (0..<97).map { Float16(sin(Double($0) * 0.13) * 0.4).bitPattern }
        let cosLUT = (0..<89).map { Float16(cos(Double($0) * 0.17) * 0.4).bitPattern }
        let qBuf = hal.device.makeBuffer(length: qElems * 2, options: .storageModeShared)!
        let kBuf = hal.device.makeBuffer(length: kElems * 2, options: .storageModeShared)!
        sinLUT.withUnsafeBufferPointer { lut in
            guard let lp = lut.baseAddress else { return }
            let qp = qBuf.contents().assumingMemoryBound(to: UInt16.self)
            var i = 0, m = 0
            while i < qElems {
                qp[i] = lp[m]
                m += 1; if m == 97 { m = 0 }
                i += 1
            }
        }
        cosLUT.withUnsafeBufferPointer { lut in
            guard let lp = lut.baseAddress else { return }
            let kp = kBuf.contents().assumingMemoryBound(to: UInt16.self)
            var i = 0, m = 0
            while i < kElems {
                kp[i] = lp[m]
                m += 1; if m == 89 { m = 0 }
                i += 1
            }
        }

        func runOperator(_ op: QWLSAOperator) throws -> ([Int], Int, [Int], Int) {
            let chunk = 512
            for pos0 in stride(from: 0, to: 2048, by: chunk) {
                let qSlice = QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length),
                                           offset: pos0 * geo.qProjOut * 2,
                                           shape: QWTensorShape([chunk, geo.qProjOut], .f16))
                let kSlice = QWTensorValue(region: QWMRegion(buffer: kBuf, base: 0, length: kBuf.length),
                                           offset: pos0 * geo.kvChannels * 2,
                                           shape: QWTensorShape([chunk, geo.kvChannels], .f16))
                let (cb, enc) = try XCTUnwrap(hal.begin())
                for f in 0..<physSeq.count {
                    op.encodeIndexAndCluster(enc, qgate: qSlice, kIn: kSlice, fullAttnLayer: f,
                                             pos0: pos0, rows: chunk, qMode: .none)
                }
                hal.end(cb, enc); cb.waitUntilCompleted(); XCTAssertNil(cb.error)
            }
            let tailQ = QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length),
                                      offset: 2048 * geo.qProjOut * 2,
                                      shape: QWTensorShape([4, geo.qProjOut], .f16))
            let tailK = QWTensorValue(region: QWMRegion(buffer: kBuf, base: 0, length: kBuf.length),
                                      offset: 2048 * geo.kvChannels * 2,
                                      shape: QWTensorShape([4, geo.kvChannels], .f16))
            func readList(_ tvList: QWTensorValue, _ tvCount: QWTensorValue) -> ([Int], Int) {
                let cnt = Int(tvCount.region.pointer.advanced(by: tvCount.offset)
                    .assumingMemoryBound(to: Int32.self)[0])
                let ptr = tvList.region.pointer.advanced(by: tvList.offset).assumingMemoryBound(to: Int32.self)
                return ((0..<cnt).map { Int(ptr[$0]) }, cnt)
            }
            let (cb, enc) = try XCTUnwrap(hal.begin())
            for f in 0..<2 {
                op.encodeIndexAndCluster(enc, qgate: tailQ, kIn: tailK, fullAttnLayer: f,
                                         pos0: 2048, rows: 4, qMode: .allRows)
            }
            op.encodeSelectBlocks(enc, fullAttnLayer: 1, pos0: 2048, rows: 4)
            hal.end(cb, enc); cb.waitUntilCompleted(); XCTAssertNil(cb.error)
            let (l1, c1) = readList(op.blockList(forFullAttnLayer: 1), op.blockCount(forFullAttnLayer: 1))

            // Interleaved physical layer 17 (f=2) + follower layer 11 visits (f=3, f=5).
            let (cb2, enc2) = try XCTUnwrap(hal.begin())
            for f in [2, 3, 4, 5] {
                op.encodeIndexAndCluster(enc2, qgate: tailQ, kIn: tailK, fullAttnLayer: f,
                                         pos0: 2048, rows: 4, qMode: .allRows)
                op.encodeSelectBlocks(enc2, fullAttnLayer: f, pos0: 2048, rows: 4)
            }
            hal.end(cb2, enc2); cb2.waitUntilCompleted(); XCTAssertNil(cb2.error)
            let (l3, c3) = readList(op.blockList(forFullAttnLayer: 3), op.blockCount(forFullAttnLayer: 3))
            return (l1, c1, l3, c3)
        }

        let (refL1, refC1, _, _) = try runOperator(opRef)
        let (reuseL1, reuseC1, reuseL3, reuseC3) = try runOperator(opReuse)
        XCTAssertEqual(reuseC1, refC1)
        XCTAssertEqual(reuseL1, refL1, "leader pass must match non-reused baseline bit-identically")
        XCTAssertEqual(reuseC3, reuseC1)
        XCTAssertEqual(reuseL3, reuseL1, "follower visit must preserve leader selections across interleaved layers")

        let snap = opReuse.captureIndexerState(position: totalPos)
        let opRestored = try QWLSAOperator(
            hal: hal, geometry: geo,
            config: QWLSAConfig(mode: .svd, budget: 512, clusterBlock: 512, crossLoopReuse: true),
            layerWeights: weights, maxContext: 4096, maxPrefillChunk: 32
        )
        XCTAssertTrue(opRestored.restoreIndexerState(snap, position: totalPos))
    }
}
