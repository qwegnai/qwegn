// QWInkling2Forward.swift — Exact 48-visit Metal 4 forward pass for Inkling-2.
import Foundation
import Metal
import QWAlloc
import QWAttn
import QWCore
import QWGEMM
import QWHAL
import QWHead

final class QWInkling2Forward: @unchecked Sendable {
    struct EmbedP { var T, H, V: UInt32; var eps: Float }
    struct NormP { var rows, dim, inIsF32: UInt32; var eps: Float }
    struct RouterP {
        var T, H, numE, topK: UInt32
        var routeScale: Float
        var moeSpecBudget: UInt32 = 0
    }
    struct SplitAttnProjP { var T, totalCols, qDim, gateDim, rDim, sinkDim, kvDim, hasKV: UInt32 }
    struct AttnResP {
        var T, H, nBlocks, numQ, blockStride, partialIsF32, normalizeValues: UInt32; var eps: Float
        var slots: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
    }
    struct BankP {
        var T, H, nStates, numQ, stateStride: UInt32; var eps: Float
        var slots: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
    }
    struct CtrlP { var T, H, rank: UInt32; var eps: Float }
    struct SConvP { var T, C, nKV, kSize, updState, resMode: UInt32; var invSqrt: Float }
    struct QKVRP {
        var T, nH, nKV, headDim, dRel, isSliding, posOff, ringCap, kvMode, nFloor: UInt32
        var qkClip, relClip, logAlpha, rmsEps: Float
    }
    struct TreeDepthsP {
        var depths: (
            UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
            UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32
        )
    }
    struct AttnP {
        var T, nH, nKV, headDim, dRel, sliding, relExtent, posOff, ringCap, kvMode, kvLen, splits: UInt32
    }

    let hal: QWHal
    let weights: QWInkling2Weights
    let session: QWInkling2Session
    let attnSinkRel: QWAttnSinkRel
    let lsaOperator: QWLSAOperator?
    let globalVisitToAttnIndex: [Int: Int]
    let pEmbed: any MTLComputePipelineState
    let pNorm: any MTLComputePipelineState
    let pCopyBF16: any MTLComputePipelineState
    let pRouter: any MTLComputePipelineState
    let pRouterSelect: any MTLComputePipelineState
    let pAttnRes: any MTLComputePipelineState
    let pAttnResCtrlNorm: any MTLComputePipelineState
    let pAttnResPackPairBatch: any MTLComputePipelineState
    let pBankNorm: any MTLComputePipelineState
    let pBankAttend: any MTLComputePipelineState
    let pCtrl: any MTLComputePipelineState
    let pCtrlPackPair: any MTLComputePipelineState
    let pCtrlApply: any MTLComputePipelineState
    let pCtrlBlendNorm: any MTLComputePipelineState
    let pCtrlPackPairBatch: any MTLComputePipelineState
    let pCtrlDownGemvBatch: any MTLComputePipelineState
    let pCtrlDownNaxBatch: (any MTLComputePipelineState)?
    let pCtrlFgBlendNaxBatch: (any MTLComputePipelineState)?
    let pCtrlApplyBatch: any MTLComputePipelineState
    let pCtrlRmsNormBatch: any MTLComputePipelineState
    let pCtrlBlendNormBatch: any MTLComputePipelineState
    let pSConvStat: any MTLComputePipelineState
    let pSConvStatTap: any MTLComputePipelineState
    let pSConvStatTreeTap: any MTLComputePipelineState
    let pSConvDyn: any MTLComputePipelineState
    let pSConvDynTap: any MTLComputePipelineState
    let pSConvDynTreeTap: any MTLComputePipelineState
    let pSConvCommit: any MTLComputePipelineState
    let pQKVR: any MTLComputePipelineState
    let pQKVRTree: any MTLComputePipelineState
    let pQKVRSConvStaticFused: any MTLComputePipelineState
    let pQKVRSConvStaticTreeFused: any MTLComputePipelineState
    let pSConvCommitKVPair: any MTLComputePipelineState
    let pGQA: any MTLComputePipelineState
    let pGQASplit: any MTLComputePipelineState
    let pF16ToBF16: any MTLComputePipelineState
    let pF32ToBF16: any MTLComputePipelineState
    let pSoftcapF32: any MTLComputePipelineState
    let pBF16ToF16FWHT32: any MTLComputePipelineState
    let pSplitAttnProj: any MTLComputePipelineState

    init(
        hal: QWHal,
        weights: QWInkling2Weights,
        session: QWInkling2Session,
        lsaOperator: QWLSAOperator? = nil
    ) throws {
        self.hal = hal
        self.weights = weights
        self.session = session
        self.attnSinkRel = try QWAttnSinkRel(hal: hal)
        self.lsaOperator = lsaOperator
        var visitMap: [Int: Int] = [:]
        var nextGlobalIdx = 0
        for (visit, lIdx) in session.executionPlan.enumerated() where !session.ink2.isSlidingLayer(lIdx) {
            visitMap[visit] = nextGlobalIdx
            nextGlobalIdx += 1
        }
        self.globalVisitToAttnIndex = visitMap
        self.pEmbed = try hal.pipeline("qwg_ink2_embed_norm")
        self.pNorm = try hal.pipeline("qwg_ink2_rms_norm")
        self.pCopyBF16 = try hal.pipeline("qwg_ink2_copy_bf16")
        self.pRouter = try hal.pipeline("qwg_ink2_router_topk")
        self.pRouterSelect = try hal.pipeline("qwg_ink2_router_select_topk")
        self.pAttnRes = try hal.pipeline("qwg_ink2_attn_res")
        self.pAttnResCtrlNorm = try hal.pipeline("qwg_ink2_attn_res_ctrl_norm")
        self.pAttnResPackPairBatch = try hal.pipeline("qwg_ink2_attn_res_pack_pair_batch")
        self.pBankNorm = try hal.pipeline("qwg_ink2_loop_bank_norm")
        self.pBankAttend = try hal.pipeline("qwg_ink2_loop_bank_attend")
        self.pCtrl = try hal.pipeline("qwg_ink2_msg_controller")
        self.pCtrlPackPair = try hal.pipeline("qwg_ink2_ctrl_pack_pair")
        self.pCtrlApply = try hal.pipeline("qwg_ink2_ctrl_apply_f32")
        self.pCtrlBlendNorm = try hal.pipeline("qwg_ink2_ctrl_blend_norm")
        self.pCtrlPackPairBatch = try hal.pipeline("qwg_ink2_ctrl_pack_pair_batch")
        self.pCtrlDownGemvBatch = try hal.pipeline("qwg_ink2_ctrl_down_gemv_batch")
        if hal.caps.isMetal4TensorCapable {
            self.pCtrlDownNaxBatch = try hal.pipeline("qwg_ink2_ctrl_down_nax_batch")
            self.pCtrlFgBlendNaxBatch = try hal.pipeline("qwg_ink2_ctrl_fg_blend_nax_batch")
        } else {
            self.pCtrlDownNaxBatch = nil
            self.pCtrlFgBlendNaxBatch = nil
        }
        self.pCtrlApplyBatch = try hal.pipeline("qwg_ink2_ctrl_apply_f32_batch")
        self.pCtrlRmsNormBatch = try hal.pipeline("qwg_ink2_ctrl_rms_norm_batch")
        self.pCtrlBlendNormBatch = try hal.pipeline("qwg_ink2_ctrl_blend_norm_batch")
        self.pSConvStat = try hal.pipeline("qwg_ink2_sconv_static")
        self.pSConvStatTap = try hal.pipeline("qwg_ink2_sconv_static_tap")
        self.pSConvStatTreeTap = try hal.pipeline("qwg_ink2_sconv_static_tree_tap")
        self.pSConvDyn = try hal.pipeline("qwg_ink2_sconv_dynamic")
        self.pSConvDynTap = try hal.pipeline("qwg_ink2_sconv_dynamic_tap")
        self.pSConvDynTreeTap = try hal.pipeline("qwg_ink2_sconv_dynamic_tree_tap")
        self.pSConvCommit = try hal.pipeline("qwg_ink2_sconv_commit")
        self.pQKVR = try hal.pipeline("qwg_ink2_qkvr_prep_kv_write")
        self.pQKVRTree = try hal.pipeline("qwg_ink2_qkvr_prep_kv_write_tree")
        self.pQKVRSConvStaticFused = try hal.pipeline("qwg_ink2_qkvr_sconv_static_fused")
        self.pQKVRSConvStaticTreeFused = try hal.pipeline("qwg_ink2_qkvr_sconv_static_tree_fused")
        self.pSConvCommitKVPair = try hal.pipeline("qwg_ink2_sconv_commit_kv_pair")
        self.pGQA = try hal.pipeline("qwg_ink2_gqa_attn")
        self.pGQASplit = try hal.pipeline("qwg_ink2_gqa_split_combine")
        self.pF16ToBF16 = try hal.pipeline("qwg_ink2_f16_to_bf16")
        self.pF32ToBF16 = try hal.pipeline("qwg_ink2_f32_to_bf16")
        self.pSoftcapF32 = try hal.pipeline("qwg_ink2_softcap_f32")
        self.pBF16ToF16FWHT32 = try hal.pipeline("qwg_ink2_bf16_to_f16_fwht32")
        self.pSplitAttnProj = try hal.pipeline("qwg_ink2_split_attn_proj")
        try QWGemmFP.prewarm(hal: hal)
        for name in [
            "qwg_moe_group_pairs_f32",
            "qwg_moe_build_active_buckets",
            "qwg_moe_down_reduce_combine_f32",
            "qwg_moe_down_reduce_combine_f32_nofma",
            "qwg_moe_down_reduce_combine_bf16",
            "qwg_head_argmax_batched_fin"
        ] {
            _ = try hal.pipeline(name)
        }
        if weights.isQuantizedQWP {
            _ = try hal.pipeline("qwg_gemv_q4g_dx_bf16_t4", isAQ4: false)
            _ = try hal.pipeline("qwg_gemv_q4g_dx_bf16_t4", isAQ4: true)
            let hasSoftcap = (session.ink2.mlpGateSoftcap > 0 || session.ink2.mlpUpSoftcap > 0)
            var warmedCompact = Set<String>()
            for lw in weights.layers {
                if case .qwp(let gu, let dn) = lw.experts {
                    if let n = QWGemm.compactGateUpKernelName(for: gu.containerKind, hasSoftcap: hasSoftcap) { warmedCompact.insert(n) }
                    if let n = QWGemm.compactDownKernelName(for: dn.containerKind) { warmedCompact.insert(n) }
                }
            }
            for name in warmedCompact {
                _ = try hal.pipeline(name)
            }
        }
        for m in 1...8 {
            _ = try hal.pipeline("qwg_head_bf16_argmax_m\(m)")
        }
        _ = try hal.pipeline("qwg_head_bf16_argmax_m16")
    }

    func slotTuple(_ slots: [Int]) -> (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) {
        let s: [UInt32] = (0..<8).map { $0 < slots.count ? UInt32(slots[$0]) : 0 }
        return (s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7])
    }

    func encodeNorm(
        _ enc: any MTLComputeCommandEncoder,
        inBF16: any MTLBuffer, inBF16Off: Int,
        inF32: any MTLBuffer, inF32Off: Int,
        inIsF32: Bool,
        w: Ink2Slice,
        outBF16: any MTLBuffer, outOff: Int,
        rows: Int, dim: Int, eps: Float
    ) {
        var np = NormP(rows: UInt32(rows), dim: UInt32(dim), inIsF32: inIsF32 ? 1 : 0, eps: eps)
        enc.setComputePipelineState(pNorm)
        enc.setBuffer(inBF16, offset: inBF16Off, index: 0)
        enc.setBuffer(inF32, offset: inF32Off, index: 1)
        enc.setBuffer(w.buffer, offset: w.offset, index: 2)
        enc.setBuffer(outBF16, offset: outOff, index: 3)
        enc.setBytes(&np, length: MemoryLayout<NormP>.stride, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    @inline(__always)
    private static func f32ToBF16(_ v: Float) -> UInt16 {
        var bits = v.bitPattern
        if (bits & 0x7F80_0000) == 0x7F80_0000 { return UInt16(truncatingIfNeeded: bits >> 16) }
        bits = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: bits >> 16)
    }

    func forward(
        tokens: [Int],
        posOffset: Int,
        runHeadRows: Int,
        greedyHead: Bool = false,
        updateState: Bool = true,
        captureTapBuf: (any MTLBuffer)? = nil,
        tapVisits: [Int] = [],
        recordVerifyConv: Bool = false,
        treePlan: Ink2TreeVerifyPlan? = nil,
        visualSpans: [QWVisualSpan] = [],
        slot: Ink2SlotState? = nil,
        useLSA: Bool = true,
        forceDenseAttention: Bool = false,
        profiler: QWInkling2Profiler? = nil,
        moeSpecBudget: Int = 0
    ) throws {
        let activeSlot = slot ?? session.primarySlotState
        let useSlotLSA = useLSA && activeSlot.slotIndex == 0
        let t = tokens.count
        guard t > 0, t <= session.maxChunk, runHeadRows >= 0, runHeadRows <= 16 else {
            throw QWInkling2Error.badState("invalid forward chunk t=\(t) maxChunk=\(session.maxChunk) runHeadRows=\(runHeadRows)")
        }
        if recordVerifyConv && t > 16 {
            throw QWInkling2Error.badState("recordVerifyConv supports t <= 16 (got \(t))")
        }
        if let treePlan, (t != treePlan.nodesCount || t > 16) {
            throw QWInkling2Error.badState("treePlan node count \(treePlan.nodesCount) != tokens \(t)")
        }
        let d = session.descriptor
        let ink = session.ink2
        let H = d.hidden
        let moeI = d.moeIntermediate ?? 1152
        let nExp = d.numExperts ?? 128
        let topK = d.expertsPerTok ?? 8
        let eps = d.rmsEps
        let xBytes = t * H * 2
        let thStride = t * H

        let tokPtr = session.tokIdsBuf.contents().assumingMemoryBound(to: Int32.self)
        for i in 0..<t { tokPtr[i] = Int32(tokens[i]) }

        var visRanges: [(chunkRow: Int, rows: Int, srcOff: Int)] = []
        if !visualSpans.isEmpty {
            let chunkEnd = posOffset + t
            let dstPtr = session.oOutBuf.contents().assumingMemoryBound(to: UInt16.self)
            for span in visualSpans {
                let sEnd = span.tokenOffset + span.tokenCount
                let ovStart = max(posOffset, span.tokenOffset)
                let ovEnd = min(chunkEnd, sEnd)
                guard ovEnd > ovStart else { continue }
                let rowCount = ovEnd - ovStart
                let chunkRow = ovStart - posOffset
                let spanRelRow = ovStart - span.tokenOffset
                let dstBase = chunkRow * H
                let srcBase = spanRelRow * H
                for i in 0..<(rowCount * H) {
                    dstPtr[dstBase + i] = Self.f32ToBF16(span.embeddings[srcBase + i])
                }
                visRanges.append((chunkRow: chunkRow, rows: rowCount, srcOff: dstBase * 2))
            }
        }

        var tapMap: [Int: Int] = [:]
        if captureTapBuf != nil {
            for (slot, vIdx) in tapVisits.enumerated() { tapMap[vIdx] = slot }
        }
        let finalTapSlot = tapMap[session.executionPlan.count]
        let needFinalNorm = runHeadRows > 0 || finalTapSlot != nil

        let hostStartNs = profiler?.beginHostTimer() ?? 0
        let wallStartNs = DispatchTime.now().uptimeNanoseconds
        func nextCB() -> (any MTLCommandBuffer)? {
            (profiler == nil)
                ? (hal.queue.makeCommandBufferWithUnretainedReferences() ?? hal.queue.makeCommandBuffer())
                : hal.queue.makeCommandBuffer()
        }
        guard var cb = nextCB() else {
            throw QWInkling2Error.badState("failed to create command buffer")
        }
        var pendingCBs: [any MTLCommandBuffer] = []
        var encOpt: (any MTLComputeCommandEncoder)? = nil
        func flushCommandBufferSlice() throws {
            profiler?.endActiveStage(encoder: &encOpt)
            encOpt?.endEncoding()
            encOpt = nil
            cb.commit()
            if profiler != nil {
                cb.waitUntilCompleted()
                if let err = cb.error {
                    throw QWInkling2Error.badState("Metal execution failed: \(err.localizedDescription)")
                }
                profiler?.finishCommandBuffer(cb, tokens: 0)
            } else {
                pendingCBs.append(cb)
            }
            guard let nextCB = nextCB() else {
                throw QWInkling2Error.badState("failed to create command buffer slice")
            }
            cb = nextCB
        }
        func encoder(_ stage: Ink2Stage? = nil) throws -> any MTLComputeCommandEncoder {
            if let stage, let profiler {
                profiler.beginStage(stage, on: cb, encoder: &encOpt, hal: hal)
            }
            if let e = encOpt { return e }
            guard let e = cb.makeComputeCommandEncoder() else {
                throw QWInkling2Error.badState("failed to create compute encoder")
            }
            hal.touchKeepAlive(e)
            encOpt = e
            return e
        }
        var (curXBuf, curXOff) = initialXTarget(ink: ink, xBytes: xBytes)
        do {
            let e0 = try encoder(.embed)
            var ep = EmbedP(T: UInt32(t), H: UInt32(H), V: UInt32(d.vocab), eps: eps)
            e0.setComputePipelineState(pEmbed)
            e0.setBuffer(session.tokIdsBuf, offset: 0, index: 0)
            e0.setBuffer(weights.embedTokens.buffer, offset: weights.embedTokens.offset, index: 1)
            e0.setBuffer(weights.embedNorm.buffer, offset: weights.embedNorm.offset, index: 2)
            e0.setBuffer(curXBuf, offset: curXOff, index: 3)
            e0.setBytes(&ep, length: MemoryLayout<EmbedP>.stride, index: 4)
            e0.dispatchThreadgroups(MTLSize(width: t, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            if !visRanges.isEmpty {
                for vr in visRanges {
                    encodeCopyBF16(
                        e0, src: session.oOutBuf, srcOff: vr.srcOff,
                        dst: curXBuf, dstOff: curXOff + vr.chunkRow * H * 2, count: vr.rows * H
                    )
                }
            }
        }

        var blocksSlots: [Int] = []
        var preludeBlocksCount = 0
        var lastBankStateCount = 0
        var bankValid = false

        for visit in 0..<session.executionPlan.count {
            if t >= 2048 && visit > 0 && visit % 24 == 0 {
                try flushCommandBufferSlice()
            }
            let lIdx = session.executionPlan[visit]
            let lw = weights.layers[lIdx]
            let kv = activeSlot.visitKVs[visit]
            let cBase = visit * 4

            try encodeLoopCoreTransition(
                visit: visit, lIdx: lIdx, t: t, H: H, thStride: thStride, eps: eps, ink: ink,
                curXBuf: &curXBuf, curXOff: &curXOff, blocksSlots: &blocksSlots,
                preludeBlocksCount: &preludeBlocksCount, lastBankStateCount: &lastBankStateCount,
                bankValid: &bankValid, beginEncoder: encoder
            )

            let useBank = bankValid && lIdx >= ink.loopCoreStart && lIdx < ink.loopCoreEnd
            if let tapBuf = captureTapBuf, let tapSlot = tapMap[visit] {
                let tEnc = try encoder(.loopResCtrlNorm)
                encodeCopyBF16(tEnc, src: curXBuf, srcOff: curXOff,
                               dst: tapBuf, dstOff: tapSlot * xBytes, count: thStride)
            }
            if useBank {
                let bEnc = try encoder(.loopResCtrlNorm)
                encodeLoopBankAttend(
                    bEnc, lIdx: lIdx, t: t, H: H, thStride: thStride,
                    lastBankStateCount: lastBankStateCount, eps: eps, ink: ink
                )
            }

            let enc = try encoder(.loopResCtrlNorm)
            let (kNormOff, vNormOff) = try encodePreAttnCtrlNorm(
                enc, lIdx: lIdx, lw: lw, useBank: useBank, t: t, H: H, thStride: thStride,
                eps: eps, ink: ink, blocksSlots: blocksSlots, curXBuf: curXBuf, curXOff: curXOff
            )

            // Attention projections
            let attnProjEnc = try encoder(.denseAttnProj)
            let proj = try encodeAttnProjections(
                attnProjEnc, lw: lw, t: t, kNormOff: kNormOff, vNormOff: vNormOff
            )

            let updConv = updateState && !recordVerifyConv && treePlan == nil
            let writeLSAKConv = !lw.isSliding && useSlotLSA && (lsaOperator.map {
                !$0.isSelectionFollower(fullAttnLayer: globalVisitToAttnIndex[visit] ?? -1)
            } ?? false)
            try encodeKVSConvAndQKVR(
                lw: lw, kv: kv, proj: proj, convSlotBase: cBase,
                convSlab: activeSlot.convStatesSlab,
                kNormOff: kNormOff, vNormOff: vNormOff,
                t: t, posOffset: posOffset,
                updateState: updConv, recordVerifyConv: recordVerifyConv,
                writeKConv: writeLSAKConv, treePlan: treePlan,
                beginEncoder: encoder
            )

            let faEnc = try encoder(.flashAttn)
            try encodeAttentionBlock(
                faEnc, visit: visit, lw: lw, kv: kv, proj: proj, t: t, posOffset: posOffset,
                useLSA: useSlotLSA, forceDenseAttention: forceDenseAttention, treePlan: treePlan
            )

            let oEnc = try encoder(.denseAttnProj)
            try encodeLinear(oEnc, weight: lw.oProjW, xBuf: session.attnOutBuf, yBuf: session.oOutBuf, m: t)

            let isBoundary = ink.isAttnResBlockBoundary(lIdx)
            if isBoundary {
                let nextSlot = blocksSlots.count
                if !(curXBuf === session.blocksRingBuf && curXOff == nextSlot * xBytes) {
                    let bEnc = try encoder(.loopResCtrlNorm)
                    encodeCopyBF16(bEnc, src: curXBuf, srcOff: curXOff,
                                   dst: session.blocksRingBuf, dstOff: nextSlot * xBytes, count: thStride)
                }
                blocksSlots.append(nextSlot)
            }

            let enc2 = try encoder(.sconv)
            encodeSConvStaticTap(
                enc2, xBF16: session.oOutBuf, weight: lw.attnSConv, convSlot: cBase + 2,
                convSlab: activeSlot.convStatesSlab,
                resBF16: curXBuf, resBF16Off: curXOff, resF32: session.dummyF,
                outBF16: session.dummyB, outF32: session.resF32Buf,
                t: t, c: H, resMode: isBoundary ? 1 : 2, denom: 5.0, updateState: updConv,
                recordVerifyConv: recordVerifyConv, treePlan: treePlan
            )

            let resEnc = try encoder(.loopResCtrlNorm)
            try encodePostAttnCtrlNorm(
                resEnc, lw: lw, useBank: useBank, t: t, H: H, thStride: thStride,
                eps: eps, ink: ink, blocksSlots: blocksSlots
            )

            encOpt = try encodeMoEBlock(
                lw: lw, t: t, H: H, nExp: nExp, topK: topK, moeI: moeI, ink: ink,
                profiler: profiler, cb: cb, beginEncoder: encoder, moeSpecBudget: moeSpecBudget
            )

            let (nextBuf, nextOff) = nextXTarget(
                nextVisit: visit + 1, blocksSlots: blocksSlots, ink: ink, xBytes: xBytes
            )

            let enc3 = try encoder(.sconv)
            encodeSConvStaticTap(
                enc3, xBF16: session.moeOutBuf, weight: lw.mlpSConv, convSlot: cBase + 3,
                convSlab: activeSlot.convStatesSlab,
                resBF16: session.dummyB, resF32: session.resF32Buf,
                outBF16: nextBuf, outBF16Off: nextOff, outF32: session.dummyF,
                t: t, c: H, resMode: 3, denom: 5.0, updateState: updConv,
                recordVerifyConv: recordVerifyConv, treePlan: treePlan
            )
            curXBuf = nextBuf
            curXOff = nextOff
        }

        if needFinalNorm {
            try encodeFinalNormAndHead(
                t: t, H: H, thStride: thStride, xBytes: xBytes, eps: eps,
                blocksSlots: blocksSlots, partialBF16: curXBuf, partialBF16Off: curXOff,
                captureTapBuf: captureTapBuf, finalTapSlot: finalTapSlot,
                runHeadRows: runHeadRows, greedyHead: greedyHead, beginEncoder: encoder
            )
        }

        profiler?.endActiveStage(encoder: &encOpt)
        encOpt?.endEncoding()
        profiler?.endHostTimer(hostStartNs)
        cb.commit()
        var gpuSeconds = 0.0
        for prev in pendingCBs {
            prev.waitUntilCompleted()
            if let err = prev.error {
                throw QWInkling2Error.badState("Metal execution failed: \(err.localizedDescription)")
            }
            gpuSeconds += max(0.0, prev.gpuEndTime - prev.gpuStartTime)
        }
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw QWInkling2Error.badState("Metal execution failed: \(err.localizedDescription)")
        }
        gpuSeconds += max(0.0, cb.gpuEndTime - cb.gpuStartTime)
        let wallSeconds = Double(DispatchTime.now().uptimeNanoseconds &- wallStartNs) * 1e-9
        QWTrunkRuntimeSupport.recordGPUBusy(gpuSeconds: gpuSeconds, wallSeconds: wallSeconds)
        profiler?.finishCommandBuffer(cb, tokens: t)
    }
}
