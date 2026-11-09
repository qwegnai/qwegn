import Foundation
import Metal
import QWCore
import QWQuant
import QWHAL
import QWAlloc
import QWGEMM
import QWGDN

// Multi-session batched layer passes (B active decode slots sharing weight projections).
extension FNGPULayerPasses {
    /// Multi-session batched PLE over `B` active slots (`1 <= B <= 4`), sharing weight projections while stepping each slot's PLE history.
    static func pleBatch(_ enc: any MTLComputeCommandEncoder, hal: QWHal, layerIndex: Int, p: FNGPUPLE, s: FNGPUScratch,
                         stream: QWTensorValue, out: QWTensorValue, activeSlots: [Int],
                         slotStates: inout [FNBatchSlotState]) throws {
        let B = activeSlots.count, W = s.width, H = s.hidden, E = p.embedDim
        let e = s.pleEmbed
        let rotEmbed: QWTensorValue
        if hal.weightBasis.rotatesActivations {
            try QWGemm.encodeH32F32(enc, hal: hal, x: e, out: s.rot32, rows: B, hidden: E)
            barrier(enc)
            rotEmbed = s.rot32
        } else {
            rotEmbed = e
        }
        try QWGemm.encodeFnHCNorm(enc, hal: hal, x: stream, weight: p.normQuery, xn: s.pleQueryN, rows: B, width: W, hidden: H, eps: p.eps)
        try denseProj(enc, hal: hal, s: s, weight: p.keyValueBank, x: rotEmbed, y: s.pleKV, rows: B)
        barrier(enc)
        try pleBatchKeyGate(enc, hal: hal, p: p, s: s, rows: B, strided: true)
        barrier(enc)
        try QWGemm.encodeFnHCNorm(enc, hal: hal, x: s.pleGV, weight: p.normConv, xn: s.pleGVN, rows: B, width: W, hidden: H, eps: p.eps)
        barrier(enc)
        for b in 0..<B {
            let slotIdx = activeSlots[b]
            let pleGV = s.pleGV.slice(byteOffset: b * W * 4, count: W, dtype: .f32)
            let pleGVN = s.pleGVN.slice(byteOffset: b * W * 4, count: W, dtype: .f32)
            let pleOut = s.pleOut.slice(byteOffset: b * W * 4, count: W, dtype: .f32)
            let par = slotStates[slotIdx].pleParity[layerIndex] & QWFourSlotRollbackRing.ringMask
            let parNext = (par + 1) & QWFourSlotRollbackRing.ringMask
            let histPair = slotStates[slotIdx].pleHistory[layerIndex]
            try QWGemm.encodeFnPLEConv(enc, hal: hal, gv: pleGV, gvNorm: pleGVN, convWeight: p.convW, out: pleOut, rows: 1,
                                       width: W, kernel: p.kernel, dilation: p.dilation, hist: histPair[par])
            try QWGemm.encodeFnPLEHistory(enc, hal: hal, oldHist: histPair[par], gvNorm: pleGVN, newHist: histPair[parNext],
                                          width: W, historyRows: p.historyRows, rows: 1)
            slotStates[slotIdx].pleParity[layerIndex] = parNext
        }
        barrier(enc)
        try QWGemm.encodeAddF32(enc, hal: hal, a: stream, b: s.pleOut, out: out, count: B * W)
        barrier(enc)
    }

    /// Multi-session batched GDN over `B` active slots (`1 <= B <= 4`), sharing `inProjBA`, `qkvZ`, and `out` projections.
    static func gdnBatch(_ enc: any MTLComputeCommandEncoder, ctx: FNGPUContext, layerIndex: Int, g: FNGPUGDNMixer, s: FNGPUScratch,
                         out: QWTensorValue, activeSlots: [Int], slotStates: inout [FNBatchSlotState]) throws {
        let hal = ctx.hal, B = activeSlots.count
        let CONV = g.convDim, VH = g.vHeads, VALUE = g.vHeads * g.vDim
        guard let qkvZ = g.qkvZ else { return }
        try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: g.inProjBA, x: s.rot32, y: s.gdnBA, rows: B)
        try denseProj(enc, hal: hal, s: s, weight: qkvZ, x: s.rot32, y: s.qkvZ32, rows: B)
        barrier(enc)
        let xStride = CONV + VALUE
        let baStride = 2 * VH
        for b in 0..<B {
            let slotIdx = activeSlots[b]
            let p0 = slotStates[slotIdx].gdnConvParity[layerIndex] & QWFourSlotRollbackRing.ringMask
            let p1 = (p0 + 1) & QWFourSlotRollbackRing.ringMask
            slotStates[slotIdx].gdnConvParity[layerIndex] = p1
            let convPair = slotStates[slotIdx].gdnConvState[layerIndex]
            let xRow = view(s.qkvZ32, byteOffset: b * xStride * 4, count: xStride, dtype: .f32)
            let bRow = view(s.gdnBA, byteOffset: b * baStride * 4, count: baStride, dtype: .f32)
            let aRow = view(s.gdnBA, byteOffset: (b * baStride + VH) * 4, count: VH, dtype: .f32)
            let yRow = view(s.yDense32, byteOffset: b * CONV * 4, count: CONV, dtype: .f32)
            let glogRow = view(s.glog, byteOffset: b * VH * 4, count: VH, dtype: .f32)
            let betaRow = view(s.beta, byteOffset: b * VH * 4, count: VH, dtype: .f32)
            try QWGDNF32.encodeConvPrepF32(enc, hal: hal, x: xRow, w: g.conv,
                                           stIn: convPair[p0], stOut: convPair[p1],
                                           b: bRow, a: aRow, aLog: g.aLog, dtBias: g.dtBias,
                                           y: yRow, glog: glogRow, beta: betaRow,
                                           rows: 1, xStride: xStride, baStride: baStride)
        }
        barrier(enc)
        for b in 0..<B {
            let slotIdx = activeSlots[b]
            let sp = slotStates[slotIdx].gdnStateParity[layerIndex] & QWFourSlotRollbackRing.ringMask
            let st = slotStates[slotIdx].gdnStates[layerIndex][sp]
            let yRow = view(s.yDense32, byteOffset: b * CONV * 4, count: CONV, dtype: .f32)
            let glogRow = view(s.glog, byteOffset: b * VH * 4, count: VH, dtype: .f32)
            let betaRow = view(s.beta, byteOffset: b * VH * 4, count: VH, dtype: .f32)
            let zRow = view(s.qkvZ32, byteOffset: (b * xStride + CONV) * 4, count: VALUE, dtype: .f32)
            let outNormRow = view(s.outNorm32, byteOffset: b * VALUE * 4, count: VALUE, dtype: .f32)
            try QWGDNF32.encodeRecurrentRowsF32(enc, hal: hal, y: yRow, betas: betaRow, glog: glogRow,
                                                state: st, stateRow0: st, z: zRow, zStride: xStride,
                                                normWeight: g.normWeight, out: outNormRow, rows: 1)
        }
        barrier(enc)
        try denseProj(enc, hal: hal, s: s, weight: g.out, x: s.outNorm32, y: out, rows: B)
        barrier(enc)
    }

    /// Multi-session batched full-attention decode over `B` active slots (`1 <= B <= 4`), sharing `a.bank` and `a.oProj`.
    static func attentionDecodeBatch(_ enc: any MTLComputeCommandEncoder, ctx: FNGPUContext, a: FNGPUAttnMixer,
                                     c: FNGPUAttnDecodeState, out: QWTensorValue,
                                     activeSlots: [Int], positions: [Int], totalSlots: Int) throws {
        let hal = ctx.hal, B = activeSlots.count
        let NH = a.heads, NKV = a.kvHeads, D = a.headDim, ROT = a.rotaryDim, ID = a.indexHeadDim
        let NI = a.indexHeads, BK = a.blockTopK, BN = a.bankRows
        try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: a.bank, x: c.rot32, y: c.src, rows: B)
        barrier(enc)
        var anyPool = false
        var anySelect = false
        for b in 0..<B {
            c.selectPartition(slot: activeSlots[b], totalSlots: totalSlots, mixer: a)
            let pos = positions[b]
            let needSelect = (pos + 1) / a.compress > a.blockTopK
            if needSelect { anySelect = true }
            if (pos + 1) % a.compress == 0 { anyPool = true }
            let rSrc = view(c.src, byteOffset: b * BN * 4, count: BN, dtype: .f32)
            let rQ = view(c.q, byteOffset: b * NH * D * 4, count: NH * D, dtype: .f32)
            let rGate = view(c.gateF, byteOffset: b * NH * D * 4, count: NH * D, dtype: .f32)
            let rIdxQ = view(c.idxQ, byteOffset: b * NI * ID * 4, count: NI * ID, dtype: .f32)
            let cosP = view(c.cos, byteOffset: pos * ROT * 4, count: ROT, dtype: .f32)
            let sinP = view(c.sin, byteOffset: pos * ROT * 4, count: ROT, dtype: .f32)
            let idxKView = view(c.indexK, byteOffset: pos * ID * 2, count: ID, dtype: .f16)
            let kv = c.kvViews(pos: pos, count: 1, kvHeads: NKV, headDim: D)
            try QWGemm.encodeFnAttnPrepF32(enc, hal: hal, src: rSrc, qNorm: a.qNorm, kNorm: a.kNorm, cos: cosP, sin: sinP,
                                           q: rQ, k: kv.k, v: kv.v,
                                           gate: rGate, rows: 1, heads: NH, kvHeads: NKV, rowStride: a.bankRows,
                                           idxK: needSelect ? nil : idxKView, ikOff: a.ikOff,
                                           kvMode: c.kvMode, kMeta: kv.kMeta, vMeta: kv.vMeta)
            if needSelect {
                try QWGemm.encodeFnIndexPrep(enc, hal: hal, src: rSrc, iqNorm: a.iqNorm, cos: cosP, sin: sinP, idxQ: rIdxQ,
                                             idxK: idxKView, rows: 1, rowStride: a.bankRows, iqOff: a.iqOff, ikOff: a.ikOff)
            }
        }
        barrier(enc)
        if anyPool {
            for b in 0..<B {
                let pos = positions[b]
                guard (pos + 1) % a.compress == 0 else { continue }
                c.selectPartition(slot: activeSlots[b], totalSlots: totalSlots, mixer: a)
                let blk = (pos + 1) / a.compress - 1
                try QWGemm.encodeFnIndexPool(enc, hal: hal,
                                             idxK: view(c.indexK, byteOffset: blk * a.compress * ID * 2, count: a.compress * ID, dtype: .f16),
                                             ikNorm: a.ikNorm,
                                             bcos: view(c.bcos, byteOffset: blk * ROT * 4, count: ROT, dtype: .f32),
                                             bsin: view(c.bsin, byteOffset: blk * ROT * 4, count: ROT, dtype: .f32),
                                             pooled: view(c.pooled, byteOffset: blk * ID * 2, count: ID, dtype: .f16),
                                             blocks: 1, compress: a.compress)
            }
            if anySelect { barrier(enc) }
        }
        if anySelect {
            let maxBlocks = max(1, c.qsaScores.shape.elementCount / max(1, c.rows))
            for b in 0..<B {
                let pos = positions[b]
                guard (pos + 1) / a.compress > a.blockTopK else { continue }
                c.selectPartition(slot: activeSlots[b], totalSlots: totalSlots, mixer: a)
                let rIdxQ = view(c.idxQ, byteOffset: b * NI * ID * 4, count: NI * ID, dtype: .f32)
                let rList = view(c.list, byteOffset: b * BK * 4, count: BK, dtype: .i32)
                let rScores = view(c.qsaScores, byteOffset: b * maxBlocks * 4, count: maxBlocks, dtype: .i32)
                try QWGemm.encodeFnQSASelect(enc, hal: hal, idxQ: rIdxQ, pooled: c.pooled, list: rList, rows: 1,
                                             blockTopK: a.blockTopK, pos0: pos, scores: rScores)
            }
            barrier(enc)
        }
        for b in 0..<B {
            c.selectPartition(slot: activeSlots[b], totalSlots: totalSlots, mixer: a)
            let pos = positions[b]
            let rQ = view(c.q, byteOffset: b * NH * D * 4, count: NH * D, dtype: .f32)
            let rGate = view(c.gateF, byteOffset: b * NH * D * 4, count: NH * D, dtype: .f32)
            let rList = view(c.list, byteOffset: b * BK * 4, count: BK, dtype: .i32)
            let rAttn = view(c.attnOut, byteOffset: b * NH * D * 4, count: NH * D, dtype: .f32)
            try QWGemm.encodeFnAttnSparseGatedF32(enc, hal: hal, q: rQ, k: c.k, v: c.v, gate: rGate, list: rList,
                                                  out: rAttn, rows: 1, heads: NH, kvHeads: NKV, pos0: pos,
                                                  kvMode: c.kvMode, kMeta: c.kMeta, vMeta: c.vMeta)
        }
        barrier(enc)
        try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: a.oProj, x: c.attnOut, y: out, rows: B)
        barrier(enc)
    }

    /// Multi-session batched decoder layer over `B` active slots (`1 <= B <= 4`).
    public static func layerBatch(_ enc: any MTLComputeCommandEncoder, ctx: FNGPUContext, layer L: FNGPULayerWeights,
                                  s: FNGPUScratch, stream: QWTensorValue, out: QWTensorValue,
                                  attnState: FNGPUAttnDecodeState?, activeSlots: [Int], positions: [Int],
                                  totalSlots: Int, slotStates: inout [FNBatchSlotState],
                                  inInject: (base: QWTensorValue, src: QWTensorValue, scale: QWTensorValue)? = nil,
                                  deferOutInject: Bool = false) throws {
        let hal = ctx.hal, B = activeSlots.count
        var h = stream
        var attnInject = inInject
        if let p = L.ple {
            if let inj = attnInject {
                try QWGemm.encodeFnInject(enc, hal: hal, base: inj.base, src: inj.src, inj: inj.scale,
                                          out: stream, rows: B, hidden: s.hidden, hc: s.hc)
                barrier(enc)
                attnInject = nil
            }
            try pleBatch(enc, hal: hal, layerIndex: L.index, p: p, s: s, stream: stream, out: s.h,
                         activeSlots: activeSlots, slotStates: &slotStates)
            h = s.h
        }
        try hcMix(enc, hal: hal, hc: L.attnHC, s: s, stream: h, mixed: s.mixedA, inj: s.injA,
                  rotMixed: attnState?.rot32,
                  injectBase: attnInject?.base, injectSrc: attnInject?.src, injectScale: attnInject?.scale,
                  rows: B)
        switch L.mixer {
        case .gdn(let g):
            try gdnBatch(enc, ctx: ctx, layerIndex: L.index, g: g, s: s, out: s.oG,
                         activeSlots: activeSlots, slotStates: &slotStates)
        case .attention(let a):
            if let c = attnState {
                try attentionDecodeBatch(enc, ctx: ctx, a: a, c: c, out: s.oG,
                                         activeSlots: activeSlots, positions: positions, totalSlots: totalSlots)
            }
        }
        try hcMix(enc, hal: hal, hc: L.mlpHC, s: s, stream: s.h2, mixed: s.mixedB, inj: s.injB,
                  injectBase: h, injectSrc: s.oG, injectScale: s.injA, rows: B)
        try moe(enc, ctx: ctx, m: L.moe, s: s, mixed: s.mixedB, out: s.moeOut, rows: B)
        if !deferOutInject {
            try QWGemm.encodeFnInject(enc, hal: hal, base: s.h2, src: s.moeOut, inj: s.injB, out: out, rows: B, hidden: s.hidden, hc: s.hc)
            barrier(enc)
        }
    }
}
