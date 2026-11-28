// QWInkling2Forward+Attn.swift — Generalized QWAttnSinkRel NAX/LSA prefill and Split-KV GQA decode.
import Foundation
import Metal
import QWAlloc
import QWAttn
import QWCore
import QWHAL
import QWKV

extension QWInkling2Forward {
    func encodeAttentionBlock(
        _ enc: any MTLComputeCommandEncoder,
        visit: Int,
        lw: QWInkling2Weights.LayerWeights,
        kv: Ink2VisitKV,
        proj: Ink2AttnProjViews,
        t: Int,
        posOffset: Int,
        qPrepOff: Int = 0,
        rPrepOff: Int = 0,
        attnOutOff: Int = 0,
        useLSA: Bool = true,
        forceDenseAttention: Bool = false,
        treePlan: Ink2TreeVerifyPlan? = nil
    ) throws {
        let cfg = lw.isSliding ? session.slidingAttnConfig : session.globalAttnConfig
        let kvLen = posOffset + t
        let qTV = session.qPrepTV.slice(byteOffset: qPrepOff, count: t * cfg.nH * cfg.headDim, dtype: .bf16)
        let rTV = session.rPrepTV.slice(byteOffset: rPrepOff, count: t * cfg.nH * cfg.dRel, dtype: .bf16)
        let outTV = session.attnOutTV.slice(byteOffset: attnOutOff, count: t * cfg.nH * cfg.headDim, dtype: .bf16)
        let sinkTV = QWTensorValue(
            region: QWMRegion(buffer: proj.sinkBuf, base: 0, length: proj.sinkBuf.length),
            offset: proj.sinkOff,
            shape: QWTensorShape([t * cfg.nH], .bf16)
        )
        let gateTV = QWTensorValue(
            region: QWMRegion(buffer: proj.gateBuf, base: 0, length: proj.gateBuf.length),
            offset: proj.gateOff,
            shape: QWTensorShape([t * cfg.nH * cfg.headDim], .bf16)
        )

        if useLSA, !lw.isSliding, let lsa = lsaOperator, let fullAttnIdx = globalVisitToAttnIndex[visit] {
            let isPrefill = t > QWAttnSinkRel.maxDecodeBatchTokens
            let sparseActive = !forceDenseAttention && (isPrefill
                ? (kvLen > lsa.config.denseBypassThreshold)
                : (posOffset >= lsa.config.denseBypassThreshold))
            let useNaxSparse = isPrefill && hal.caps.supportsSimdgroupMMA
                && cfg.headDim == 128 && (cfg.nH / cfg.nKV) <= 16
            let reuseBlocks = lsa.isSelectionFollower(fullAttnLayer: fullAttnIdx)
            if !reuseBlocks {
                let qMode: QWLSAOperator.IndexPrepQMode = !sparseActive
                    ? .none
                    : (useNaxSparse ? .tiled32 : .allRows)
                let kConvTV = QWTensorValue(
                    region: QWMRegion(buffer: session.kConvBuf, base: 0, length: session.kConvBuf.length),
                    offset: 0,
                    shape: QWTensorShape([t * cfg.nKV * cfg.headDim], .bf16)
                )
                lsa.encodeIndexAndClusterBF16(
                    enc, qIn: qTV, kIn: kConvTV,
                    fullAttnLayer: fullAttnIdx, pos0: posOffset, rows: t,
                    qRowStride: cfg.nH * cfg.headDim, qMode: qMode
                )
                if sparseActive {
                    lsa.encodeSelectBlocks(
                        enc, fullAttnLayer: fullAttnIdx, pos0: posOffset, rows: t,
                        tiled32: useNaxSparse, sink: sinkTV
                    )
                }
            }
            if sparseActive {
                let blockListTV = lsa.blockList(forFullAttnLayer: fullAttnIdx)
                let blockCountTV = lsa.blockCount(forFullAttnLayer: fullAttnIdx)
                if useNaxSparse {
                    try attnSinkRel.encodePrefillSparse(
                        enc, config: cfg,
                        q: qTV, r: rTV, relProj: lw.relProjTV,
                        sink: sinkTV, gate: gateTV, kv: kv.kvCacheView,
                        out: outTV,
                        blockList: blockListTV, blockCount: blockCountTV,
                        listStride: lsa.listStride,
                        T: t, posOffset: posOffset, kvLen: kvLen, perTileBlockList: false
                    )
                    return
                } else if !isPrefill {
                    try attnSinkRel.encodeDecodeSparse(
                        enc, config: cfg,
                        q: qTV, r: rTV, relProj: lw.relProjTV,
                        sink: sinkTV, gate: gateTV, kv: kv.kvCacheView,
                        partials: session.partialsView, out: outTV,
                        blockList: blockListTV, blockCount: blockCountTV,
                        listStride: lsa.listStride,
                        T: t, posOffset: posOffset, kvLen: kvLen, splits: nil,
                        treeMeta: treePlan?.attnTreeMeta
                    )
                    return
                }
            }
        }

        if t > QWAttnSinkRel.maxDecodeBatchTokens {
            if hal.caps.supportsSimdgroupMMA && cfg.headDim == 128 && (cfg.nH / cfg.nKV) <= 16 {
                try attnSinkRel.encodePrefill(
                    enc, config: cfg,
                    q: qTV, r: rTV, relProj: lw.relProjTV,
                    sink: sinkTV, gate: gateTV, kv: kv.kvCacheView,
                    out: outTV, T: t, posOffset: posOffset, kvLen: kvLen
                )
                return
            }
            let apBase = AttnP(
                T: UInt32(t), nH: UInt32(cfg.nH), nKV: UInt32(cfg.nKV),
                headDim: UInt32(cfg.headDim), dRel: UInt32(cfg.dRel),
                sliding: UInt32(cfg.slidingWindow), relExtent: UInt32(cfg.relExtent),
                posOff: UInt32(posOffset), ringCap: UInt32(kv.ringCap),
                kvMode: session.kvModeRaw, kvLen: UInt32(kvLen), splits: 1
            )
            var ap = apBase
            enc.setComputePipelineState(pGQA)
            enc.setBuffer(session.qPrepBuf, offset: qPrepOff, index: 0)
            enc.setBuffer(session.rPrepBuf, offset: rPrepOff, index: 1)
            enc.setBuffer(lw.relProj.buffer, offset: lw.relProj.offset, index: 2)
            enc.setBuffer(proj.sinkBuf, offset: proj.sinkOff, index: 3)
            enc.setBuffer(proj.gateBuf, offset: proj.gateOff, index: 4)
            enc.setBuffer(kv.kCache, offset: 0, index: 5)
            enc.setBuffer(kv.vCache, offset: 0, index: 6)
            enc.setBuffer(kv.kMeta, offset: 0, index: 7)
            enc.setBuffer(kv.vMeta, offset: 0, index: 8)
            enc.setBuffer(session.attnOutBuf, offset: attnOutOff, index: 9)
            enc.setBuffer(session.partOBuf, offset: 0, index: 10)
            enc.setBuffer(session.partMLBuf, offset: 0, index: 11)
            enc.setBytes(&ap, length: MemoryLayout<AttnP>.stride, index: 12)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: cfg.nH, depth: t),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            return
        }

        try attnSinkRel.encodeDecode(
            enc, config: cfg,
            q: qTV, r: rTV, relProj: lw.relProjTV,
            sink: sinkTV, gate: gateTV, kv: kv.kvCacheView,
            partials: session.partialsView, out: outTV,
            T: t, posOffset: posOffset, kvLen: kvLen, splits: nil,
            treeMeta: treePlan?.attnTreeMeta
        )
    }
}
