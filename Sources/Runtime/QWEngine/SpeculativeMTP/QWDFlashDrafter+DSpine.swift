// QWDFlashDrafter+DSpine.swift — DSpine adjacent causal code bridge and
// exact-sender last-write refinement tail extension for QWDFlashDrafter.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWHead
import QWLoad
import QWMTP
import QWQuant

struct DSpineBridgeWeights {
    let readWeights: [QWTensorValue]
    let writeWeights: [QWTensorValue]
    let packedGatePrev: QWTensorValue
    let gateBias: QWTensorValue
    let codebook: QWTensorValue
}

struct DSpineBridgeScratch {
    let bridgeRms: QWTensorValue
    let bridgeProj: QWTensorValue
    let bridgeCodes: QWTensorValue
    let bridgePacked: QWTensorValue
    let bridgeMsg: QWTensorValue
    let propXPost4: QWTensorValue
    let refineMsg: QWTensorValue
    let refineStates: QWTensorValue
    let refineNorm: QWTensorValue
    let refineRot: QWTensorValue
    let refineTable: QWTensorValue
}

struct DSpinePipelines {
    let psoBridgeState: any MTLComputePipelineState
    let psoBridgeWriteAdd: any MTLComputePipelineState
    let psoRefineCodeToMsg: any MTLComputePipelineState
    let psoRefineWrite: any MTLComputePipelineState
    let psoRefineScores: any MTLComputePipelineState
    let psoRefineScoresSub2: any MTLComputePipelineState
    let psoRefineChain: any MTLComputePipelineState

    init(hal: QWHal) throws {
        self.psoBridgeState = try hal.pipeline("qwg_dspine_bridge_state")
        self.psoBridgeWriteAdd = try hal.pipeline("qwg_dspine_bridge_write_add")
        self.psoRefineCodeToMsg = try hal.pipeline("qwg_dspine_refine_code_to_msg")
        self.psoRefineWrite = try hal.pipeline("qwg_dspine_refine_write")
        self.psoRefineScores = try hal.pipeline("qwg_dspine_refine_scores")
        self.psoRefineScoresSub2 = try hal.pipeline("qwg_dspine_refine_scores_sub2")
        self.psoRefineChain = try hal.pipeline("qwg_dspine_refine_chain")
    }
}

extension QWDFlashDrafter {
    static func loadDSpineBridgeWeights(
        config: QWDFlashConfig,
        wSub: QWSubAllocator,
        loadF16: (String, Int) throws -> QWTensorValue,
        loadRawBF16: (String, Int) throws -> QWTensorValue
    ) throws -> DSpineBridgeWeights {
        let H = config.hiddenSize
        let V = config.vocabSize
        let C = config.bridgeCodeSize
        let R = config.bridgeRank
        let L = config.numLayers

        let earlyAll = try loadF16("selected_prefix_bridge.early_read_weights", (L - 1) * C * H)
        let lastRead = try loadF16("selected_prefix_bridge.read_weight", C * H)
        var readViews: [QWTensorValue] = []
        let readLayerBytes = C * H * 2
        for l in 0..<(L - 1) {
            readViews.append(QWTensorValue(
                region: earlyAll.region,
                offset: earlyAll.offset + l * readLayerBytes,
                shape: QWTensorShape([C * H], .f16)
            ))
        }
        readViews.append(lastRead)

        let writeAll = try loadF16("selected_prefix_bridge.feature_write_weights", L * H * R)
        var writeViews: [QWTensorValue] = []
        let writeLayerBytes = H * R * 2
        for l in 0..<L {
            writeViews.append(QWTensorValue(
                region: writeAll.region,
                offset: writeAll.offset + l * writeLayerBytes,
                shape: QWTensorShape([H * R], .f16)
            ))
        }

        let packedGatePrev = try wSub.allocate(QWTensorShape([3 * R * C], .f16))
        let gateW = try loadF16("selected_prefix_bridge.gate.weight", R * C)
        let predGateW = try loadF16("selected_prefix_bridge.predecessor_gate.weight", R * C)
        let prevW = try loadF16("selected_prefix_bridge.prev.weight", R * C)
        let partBytes = R * C * 2
        let dstBase = packedGatePrev.region.buffer.contents()
            .advanced(by: packedGatePrev.region.base + packedGatePrev.offset)
        for (idx, part) in [gateW, predGateW, prevW].enumerated() {
            let srcPtr = part.region.buffer.contents().advanced(by: part.region.base + part.offset)
            memcpy(dstBase.advanced(by: idx * partBytes), srcPtr, partBytes)
        }

        let gateBias = try loadF16("selected_prefix_bridge.gate.bias", R)
        let codebook = try loadRawBF16("selected_prefix_bridge.codebook", V * C)

        return DSpineBridgeWeights(
            readWeights: readViews,
            writeWeights: writeViews,
            packedGatePrev: packedGatePrev,
            gateBias: gateBias,
            codebook: codebook
        )
    }

    static func allocateDSpineScratch(
        config: QWDFlashConfig,
        sSub: QWSubAllocator,
        maxB: Int,
        maxD: Int
    ) throws -> DSpineBridgeScratch {
        let H = config.hiddenSize
        let C = config.bridgeCodeSize
        let R = config.bridgeRank
        return DSpineBridgeScratch(
            bridgeRms: try sSub.allocate(QWTensorShape([maxB], .f32)),
            bridgeProj: try sSub.allocate(QWTensorShape([maxD * C], .f32)),
            bridgeCodes: try sSub.allocate(QWTensorShape([maxB * C], .f16)),
            bridgePacked: try sSub.allocate(QWTensorShape([maxB * 3 * R], .f32)),
            bridgeMsg: try sSub.allocate(QWTensorShape([maxD * R], .f16)),
            propXPost4: try sSub.allocate(QWTensorShape([maxD * H], .f32)),
            refineMsg: try sSub.allocate(QWTensorShape([maxD * 16 * R], .f16)),
            refineStates: try sSub.allocate(QWTensorShape([maxD * 16 * H], .f32)),
            refineNorm: try sSub.allocate(QWTensorShape([maxD * 16 * H], .f16)),
            refineRot: try sSub.allocate(QWTensorShape([maxD * 16 * H], .f16)),
            refineTable: try sSub.allocate(QWTensorShape([maxD * 16 * 16], .f32))
        )
    }

    func encodeDSpineBridgeLayer(
        _ enc: any MTLComputeCommandEncoder,
        layer: Int,
        anchorToken: Int,
        rows M: Int,
        bw: DSpineBridgeWeights,
        bs: DSpineBridgeScratch
    ) {
        let D = M - 1
        let H = config.hiddenSize
        let C = config.bridgeCodeSize
        let R = config.bridgeRank

        encodeRMSNorm(
            enc, x: propX, gamma: propDown, y: propNorm, yRot: bs.bridgeRms,
            rows: M, mode: 4, epsOverride: config.bridgeEps
        )

        let propNormDraft = QWTensorValue(
            region: propNorm.region, offset: propNorm.offset + H * 2,
            shape: QWTensorShape([D * H], .f16)
        )
        encodeF16GemvBatched(enc, w: bw.readWeights[layer], x: propNormDraft, y: bs.bridgeProj, n: C, k: H, m: D)

        var anchorI = Int32(anchorToken), dimC = UInt32(C), bEps = config.bridgeEps, mode0: UInt32 = 0
        enc.setComputePipelineState(dspinePSO.psoBridgeState)
        bs.bridgeProj.bind(enc, index: 0)
        bw.codebook.bind(enc, index: 1)
        bs.bridgeCodes.bind(enc, index: 2)
        enc.setBytes(&anchorI, length: 4, index: 3)
        enc.setBytes(&dimC, length: 4, index: 4)
        enc.setBytes(&bEps, length: 4, index: 5)
        enc.setBytes(&mode0, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: M, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        encodeF16GemvBatched(enc, w: bw.packedGatePrev, x: bs.bridgeCodes, y: bs.bridgePacked, n: 3 * R, k: C, m: M)

        var dimR = UInt32(R), mode1: UInt32 = 1
        enc.setComputePipelineState(dspinePSO.psoBridgeState)
        bs.bridgePacked.bind(enc, index: 0)
        bw.gateBias.bind(enc, index: 1)
        bs.bridgeMsg.bind(enc, index: 2)
        enc.setBytes(&anchorI, length: 4, index: 3)
        enc.setBytes(&dimR, length: 4, index: 4)
        enc.setBytes(&bEps, length: 4, index: 5)
        enc.setBytes(&mode1, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: D, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        let propXDraft = QWTensorValue(
            region: propX.region, offset: propX.offset + H * 4,
            shape: QWTensorShape([D * H], .f32)
        )
        let rmsDraft = QWTensorValue(
            region: bs.bridgeRms.region, offset: bs.bridgeRms.offset + 4,
            shape: QWTensorShape([D], .f32)
        )
        let isLast = (layer == config.numLayers - 1)
        let rawOut = isLast ? bs.propXPost4 : propXDraft

        var hU = UInt32(H), rU = UInt32(R), dU = UInt32(D)
        enc.setComputePipelineState(dspinePSO.psoBridgeWriteAdd)
        bw.writeWeights[layer].bind(enc, index: 0)
        bs.bridgeMsg.bind(enc, index: 1)
        propXDraft.bind(enc, index: 2)
        rmsDraft.bind(enc, index: 3)
        rawOut.bind(enc, index: 4)
        enc.setBytes(&hU, length: 4, index: 5)
        enc.setBytes(&rU, length: 4, index: 6)
        enc.setBytes(&dU, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: (H + 3) / 4, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

        if !isLast {
            encodeRMSNorm(enc, x: propX, gamma: layers[layer + 1].ln1, y: propNorm, yRot: propRot, rows: M, mode: 2)
        }
    }

    func encodeGateUpMatmul(
        _ enc: any MTLComputeCommandEncoder,
        weight: QWWeightView,
        input: QWTensorValue,
        output: QWTensorValue,
        rows: Int
    ) {
        if !QWMTPBatchedMLP.usesQ4GMKernel(weight) {
            encodeAQGateUpMatmul(enc, weight: weight, input: input, output: output, rows: rows)
            return
        }
        let H = config.hiddenSize
        let I = config.intermediateSize
        if let guPSO = psoGateUpFwht32Batched[rows] {
            enc.setComputePipelineState(guPSO)
            weight.blocks.bind(enc, index: 0)
            input.bind(enc, index: 1)
            output.bind(enc, index: 3)
            var guN = UInt32(2 * I) | (weight.isIncumbentContainer ? 0 : 0x8000_0000)
            var guK = UInt32(H)
            enc.setBytes(&guN, length: 4, index: 4)
            enc.setBytes(&guK, length: 4, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: (I + 31) / 32, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        } else if (9...16).contains(rows) {
            let c0 = rows / 2, c1 = rows - c0
            encodeGateUpMatmul(enc, weight: weight, input: input.slice(byteOffset: 0, count: c0 * H),
                               output: output.slice(byteOffset: 0, count: c0 * I), rows: c0)
            encodeGateUpMatmul(enc, weight: weight, input: input.slice(byteOffset: c0 * H * 2, count: c1 * H),
                               output: output.slice(byteOffset: c0 * I * 2, count: c1 * I), rows: c1)
        }
    }

    func encodeDSpineRefinementTail(
        _ enc: any MTLComputeCommandEncoder,
        draftCount D: Int,
        topK: Int,
        headStorage: QWHeadStorage,
        bw: DSpineBridgeWeights,
        bs: DSpineBridgeScratch,
        stochastic: Bool
    ) {
        let H = config.hiddenSize
        let C = config.bridgeCodeSize
        let R = config.bridgeRank
        var hU = UInt32(H), rU = UInt32(R), cU = UInt32(C), topKU = UInt32(topK), dU = UInt32(D)

        enc.setComputePipelineState(dspinePSO.psoRefineCodeToMsg)
        bw.packedGatePrev.bind(enc, index: 0)
        bw.codebook.bind(enc, index: 1)
        candidatesOut.bind(enc, index: 2)
        bs.bridgePacked.bind(enc, index: 3)
        bw.gateBias.bind(enc, index: 4)
        bs.refineMsg.bind(enc, index: 5)
        enc.setBytes(&rU, length: 4, index: 6)
        enc.setBytes(&cU, length: 4, index: 7)
        enc.setBytes(&topKU, length: 4, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: (R + 3) / 4, height: D, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

        let propXDraft = QWTensorValue(
            region: propX.region, offset: propX.offset + H * 4,
            shape: QWTensorShape([D * H], .f32)
        )
        let rmsDraft = QWTensorValue(
            region: bs.bridgeRms.region, offset: bs.bridgeRms.offset + 4,
            shape: QWTensorShape([D], .f32)
        )
        enc.setComputePipelineState(dspinePSO.psoRefineWrite)
        bw.writeWeights[config.numLayers - 1].bind(enc, index: 0)
        bs.refineMsg.bind(enc, index: 1)
        propXDraft.bind(enc, index: 2)
        rmsDraft.bind(enc, index: 3)
        bs.refineStates.bind(enc, index: 4)
        enc.setBytes(&hU, length: 4, index: 5)
        enc.setBytes(&rU, length: 4, index: 6)
        enc.setBytes(&topKU, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: (H + 3) / 4, height: D, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

        encodeRMSNorm(
            enc, x: bs.refineStates, gamma: finalNorm,
            y: bs.refineNorm, yRot: bs.refineRot, rows: D * 16, mode: 2
        )

        var headMode: UInt32
        var useSub2Refine = false
        switch headStorage {
        case .f16(let hTensor):
            headMode = 0
            enc.setComputePipelineState(dspinePSO.psoRefineScores)
            hTensor.bind(enc, index: 0)
            bs.refineNorm.bind(enc, index: 1)
        case .q4g(let hw):
            if hw.view.isSHVQ {
                headMode = 0x8000_0000 | UInt32(hw.view.bits) | (UInt32(hw.view.groupSize) << 8)
                useSub2Refine = hw.view.bits == 1 || hw.view.bits == 15
            } else {
                headMode = hw.view.isIncumbentContainer ? 1 : 2
            }
            enc.setComputePipelineState(useSub2Refine ? dspinePSO.psoRefineScoresSub2 : dspinePSO.psoRefineScores)
            hw.view.blocks.bind(enc, index: 0)
            bs.refineRot.bind(enc, index: 1)
        }
        candidatesOut.bind(enc, index: 2)
        bs.refineTable.bind(enc, index: 3)
        enc.setBytes(&hU, length: 4, index: 4)
        enc.setBytes(&topKU, length: 4, index: 5)
        enc.setBytes(&headMode, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: 16, height: D, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        if !stochastic {
            enc.setComputePipelineState(dspinePSO.psoRefineChain)
            bs.refineTable.bind(enc, index: 0)
            candidatesOut.bind(enc, index: 1)
            draftTokensOut.bind(enc, index: 2)
            enc.setBytes(&dU, length: 4, index: 3)
            enc.setBytes(&topKU, length: 4, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
    }

    func selectDSpineStochastic(
        owner: QWStochasticSession,
        draftCount D: Int,
        topK: Int,
        bs: DSpineBridgeScratch
    ) throws -> (drafts: [Int], proposals: [QWMTPGumbelProposal]) {
        let V = config.vocabSize
        let temp = max(Double(owner.config.temperature), 1e-6)
        let tablePtr = bs.refineTable.region.buffer.contents()
            .advanced(by: bs.refineTable.region.base + bs.refineTable.offset)
            .assumingMemoryBound(to: Float.self)
        let cPtr = candidatesOut.region.buffer.contents()
            .advanced(by: candidatesOut.region.base + candidatesOut.offset)
            .assumingMemoryBound(to: Int32.self)

        var senderIdx = 0
        var drafts: [Int] = []
        var proposals: [QWMTPGumbelProposal] = []
        for pos in 0..<D {
            var scores = [Double](repeating: 0, count: topK)
            var ids = [Int](repeating: 0, count: topK)
            var maxScore = -Double.infinity
            let rowBase = (pos * 16 + senderIdx) * 16
            for k in 0..<topK {
                let cand = Int(cPtr[pos * 16 + k])
                let s = Double(tablePtr[rowBase + k])
                ids[k] = cand
                scores[k] = s
                if s > maxScore { maxScore = s }
            }
            let weights = scores.map { exp(($0 - maxScore) / temp) }
            let dist = try QWSamplingDistribution.sparse(vocabulary: V, ids: ids, weights: weights)
            let proposal = try owner.stream.sampleGumbelProposal(dist)
            if let chosenK = ids.firstIndex(of: proposal.token) {
                senderIdx = chosenK
            }
            drafts.append(proposal.token)
            proposals.append(proposal)
        }
        return (drafts, proposals)
    }
}
