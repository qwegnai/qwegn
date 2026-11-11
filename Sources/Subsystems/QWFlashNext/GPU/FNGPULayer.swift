import Foundation
import Metal
import QWCore
import QWQuant
import QWHAL
import QWAlloc
import QWGEMM
import QWGDN

/// Encodes one FlashNext decoder layer on the GPU. Each phase is barrier-separated; per-token loops never interleave
/// dependent stages (the lesson from the GDN out_proj mismatch).
public enum FNGPULayerPasses {
    public static func view(_ t: QWTensorValue, byteOffset: Int, count: Int, dtype: QWDType) -> QWTensorValue {
        t.slice(byteOffset: byteOffset, count: count, dtype: dtype)
    }

    static func barrier(_ enc: any MTLComputeCommandEncoder) { enc.memoryBarrier(scope: .buffers) }

    /// Dense HQ4 projection over T rows using the shared QWGemm.encodeDenseProjF32 dispatcher.
    static func denseProj(_ enc: any MTLComputeCommandEncoder, hal: QWHal, s: FNGPUScratch, weight: QWWeightView,
                           x: QWTensorValue, y: QWTensorValue, rows: Int,
                           epi: UInt32 = 0, scale: Float = 1,
                           x16Precast: QWTensorValue? = nil) throws {
        try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: weight, x: x, y: y,
                                      naxX16: s.naxX16, naxY16: s.naxY16, maxTileWidth: s.width,
                                      rows: rows, epi: epi, scale: scale, x16Precast: x16Precast)
    }

    public static func ple(_ enc: any MTLComputeCommandEncoder, hal: QWHal, p: FNGPUPLE, s: FNGPUScratch,
                           stream: QWTensorValue, out: QWTensorValue, embed: QWTensorValue? = nil) throws {
        let T = s.T, W = s.width, H = s.hidden, C = s.hc, E = p.embedDim
        let e = embed ?? s.pleEmbed
        let rotEmbed: QWTensorValue
        if hal.weightBasis.rotatesActivations {
            try QWGemm.encodeH32F32(enc, hal: hal, x: e, out: s.rot32, rows: T, hidden: E)
            barrier(enc)
            rotEmbed = s.rot32
        } else {
            rotEmbed = e
        }
        if T <= 8 {
            try QWGemm.encodeFnHCNorm(enc, hal: hal, x: stream, weight: p.normQuery, xn: s.pleQueryN, rows: T, width: W, hidden: H, eps: p.eps)
            try denseProj(enc, hal: hal, s: s, weight: p.keyValueBank, x: rotEmbed, y: s.pleKV, rows: T)
            barrier(enc)
            try pleBatchKeyGate(enc, hal: hal, p: p, s: s, rows: T, strided: true)
            barrier(enc)
            try QWGemm.encodeFnHCNorm(enc, hal: hal, x: s.pleGV, weight: p.normConv, xn: s.pleGVN, rows: T, width: W, hidden: H, eps: p.eps)
            barrier(enc)
            for t in 0..<T {
                let pleGV = s.pleGV.slice(byteOffset: t * W * 4, count: W, dtype: .f32)
                let pleGVN = s.pleGVN.slice(byteOffset: t * W * 4, count: W, dtype: .f32)
                let pleOut = s.pleOut.slice(byteOffset: t * W * 4, count: W, dtype: .f32)
                let p0 = p.parity & QWFourSlotRollbackRing.ringMask
                let p1 = (p0 + 1) & QWFourSlotRollbackRing.ringMask
                let hist = p.history[p0]
                try QWGemm.encodeFnPLEConv(enc, hal: hal, gv: pleGV, gvNorm: pleGVN, convWeight: p.convW, out: pleOut, rows: 1,
                                           width: W, kernel: p.kernel, dilation: p.dilation, hist: hist)
                try QWGemm.encodeFnPLEHistory(enc, hal: hal, oldHist: hist, gvNorm: pleGVN, newHist: p.history[p1],
                                              width: W, historyRows: p.historyRows, rows: 1)
                p.parity = p1
                barrier(enc)
            }
            try QWGemm.encodeAddF32(enc, hal: hal, a: stream, b: s.pleOut, out: out, count: T * W)
            barrier(enc)
            return
        }
        try QWGemm.encodeFnHCNorm(enc, hal: hal, x: stream, weight: p.normQuery, xn: s.pleQueryN, rows: T, width: W, hidden: H, eps: p.eps)
        barrier(enc)
        let x16Pre: QWTensorValue?
        if T >= 64 && W % 64 == 0 && H % 64 == 0 && E % 64 == 0 && E <= s.width {
            let count = T * E
            let xHi = s.naxX16.slice(byteOffset: 0, count: count, dtype: .f16)
            try QWGemm.encodeCast(enc, hal: hal, x32: rotEmbed, y16: xHi, count: count)
            barrier(enc)
            x16Pre = xHi
        } else {
            x16Pre = nil
        }
        try denseProj(enc, hal: hal, s: s, weight: p.keyBank, x: rotEmbed, y: s.pleK, rows: T, x16Precast: x16Pre)
        try denseProj(enc, hal: hal, s: s, weight: p.valueBank, x: rotEmbed, y: s.pleV, rows: T, x16Precast: x16Pre)
        barrier(enc)
        try QWGemm.encodeFnHCNorm(enc, hal: hal, x: s.pleK, weight: p.normKey, xn: s.pleKeyN, rows: T, width: W, hidden: H, eps: p.eps)
        barrier(enc)
        try QWGemm.encodeFnPLEGate(enc, hal: hal, keyNorm: s.pleKeyN, queryNorm: s.pleQueryN, vals: s.pleV, gv: s.pleGV,
                                   rows: T, hidden: H, hc: C)
        barrier(enc)
        try QWGemm.encodeFnHCNorm(enc, hal: hal, x: s.pleGV, weight: p.normConv, xn: s.pleGVN, rows: T, width: W, hidden: H, eps: p.eps)
        barrier(enc)
        let p0 = p.parity & QWFourSlotRollbackRing.ringMask
        let p1 = (p0 + 1) & QWFourSlotRollbackRing.ringMask
        let hist = p.history[p0]
        try QWGemm.encodeFnPLEConv(enc, hal: hal, gv: s.pleGV, gvNorm: s.pleGVN, convWeight: p.convW, out: s.pleOut, rows: T,
                                   width: W, kernel: p.kernel, dilation: p.dilation, hist: hist)
        try QWGemm.encodeFnPLEHistory(enc, hal: hal, oldHist: hist, gvNorm: s.pleGVN, newHist: p.history[p1],
                                      width: W, historyRows: p.historyRows, rows: T)
        p.parity = p1
        barrier(enc)
        try QWGemm.encodeAddF32(enc, hal: hal, a: stream, b: s.pleOut, out: out, count: T * W)
        barrier(enc)
    }

    /// HC mixer: normalized stream -> low-rank gate -> mixed (H), rotated mixed (H), and injection (C).
    public static func hcMix(_ enc: any MTLComputeCommandEncoder, hal: QWHal, hc: FNGPUHC, s: FNGPUScratch,
                             stream: QWTensorValue, mixed: QWTensorValue, inj: QWTensorValue,
                             rotMixed: QWTensorValue? = nil,
                             injectBase: QWTensorValue? = nil, injectSrc: QWTensorValue? = nil,
                             injectScale: QWTensorValue? = nil,
                             rows: Int? = nil) throws {
        let T = rows ?? s.T, W = s.width, H = s.hidden, C = s.hc
        let rotOut = rotMixed ?? s.rot32
        let useNaxDown = T >= 64 && hc.down.n % 64 == 0 && hc.down.k % 64 == 0 && hc.down.k <= s.width
        let rot16Pre: QWTensorValue? = useNaxDown ? s.naxX16 : nil
        if T <= 8,
           try QWGemm.encodeFnHCNormDownInjectDecode(enc, hal: hal, downWeight: hc.down, x: injectBase ?? stream,
                                                     src: injectSrc, inj: injectScale,
                                                     xOut: injectBase != nil ? stream : nil,
                                                     normWeight: hc.norm, xn: s.hcXN, yLow: s.hcLow,
                                                     injWeight: hc.injectBF16, yInj: inj, width: W, hidden: H,
                                                     eps: hc.eps, hc: C, scale: Float(C), rows: T) {
            barrier(enc)
            try QWGemm.encodeFnHCUpMixRot(enc, hal: hal, upWeight: hc.up, hcLow: s.hcLow, xn: s.hcXN,
                                          mixed: mixed, rotMixed: rotOut, rows: T, hidden: H, hc: C)
            barrier(enc)
            return
        }
        if let base = injectBase, let src = injectSrc, let scale = injectScale {
            try QWGemm.encodeFnHCInjectNormRot(enc, hal: hal, base: base, src: src, inj: scale, xOut: stream,
                                               weight: hc.norm, xn: s.hcXN, rot: s.hcRotW, rows: T,
                                               width: W, hidden: H, eps: hc.eps, rot16: rot16Pre, rinvOnly: T > 8)
        } else {
            try QWGemm.encodeFnHCNormRot(enc, hal: hal, x: stream, weight: hc.norm, xn: s.hcXN, rot: s.hcRotW, rows: T,
                                         width: W, hidden: H, eps: hc.eps, rot16: rot16Pre, rinvOnly: T > 8)
        }
        barrier(enc)
        if T <= 8 {
            try QWGemm.encodeFnHCDownInjectDecode(enc, hal: hal, downWeight: hc.down, xRot: s.hcRotW, yLow: s.hcLow,
                                                  injWeight: hc.injectBF16, xXn: s.hcXN, yInj: inj,
                                                  hc: C, scale: Float(C), rows: T)
            barrier(enc)
            try QWGemm.encodeFnHCUpMixRot(enc, hal: hal, upWeight: hc.up, hcLow: s.hcLow, xn: s.hcXN,
                                          mixed: mixed, rotMixed: rotOut, rows: T, hidden: H, hc: C)
            barrier(enc)
        } else {
            let useNaxUp = T >= 64 && hc.up.n % 64 == 0 && hc.up.k % 64 == 0 && hc.up.k <= s.width
            let gateOut = useNaxUp ? view(s.hcGate, byteOffset: 0, count: T * W, dtype: .f16) : s.hcGate
            if useNaxDown {
                try denseProj(enc, hal: hal, s: s, weight: hc.down, x: s.hcRotW, y: s.hcLow, rows: T, epi: 0, scale: 1, x16Precast: s.naxX16)
                barrier(enc)
                try QWGemm.encodeFnHCLowEpilogue(enc, hal: hal, yLow: s.hcLow, rows: T, lowrank: hc.down.n, scale: Float(C), out16: useNaxUp ? s.naxX16 : nil)
                barrier(enc)
                try denseProj(enc, hal: hal, s: s, weight: hc.up, x: s.hcLow, y: gateOut, rows: T, epi: useNaxUp ? 0 : 2, scale: 1, x16Precast: useNaxUp ? s.naxX16 : nil)
            } else {
                try denseProj(enc, hal: hal, s: s, weight: hc.down, x: s.hcRotW, y: s.hcLow, rows: T, epi: 3, scale: Float(C))
                barrier(enc)
                try denseProj(enc, hal: hal, s: s, weight: hc.up, x: s.hcLow, y: gateOut, rows: T, epi: useNaxUp ? 0 : 2, scale: 1)
            }
            barrier(enc)
            let skipRot32 = T >= 64 && rotMixed == nil && H % 64 == 0 && H <= s.width
            try QWGemm.encodeFnHCMixRot(enc, hal: hal, gate: gateOut, xn: s.hcXN, mixed: mixed, rotMixed: rotOut,
                                        rows: T, hidden: H, hc: C, rotMixed16: T > 8 ? s.naxX16 : nil,
                                        applySigmoid: useNaxUp, skipRot32: skipRot32, injWeight: hc.injectBF16, injOut: inj,
                                        xnRecompute: (stream, hc.norm))
            barrier(enc)
        }
    }

    static func gdn(_ enc: any MTLComputeCommandEncoder, ctx: FNGPUContext, g: FNGPUGDNMixer, s: FNGPUScratch,
                    mixed: QWTensorValue, out: QWTensorValue) throws {
        let hal = ctx.hal, T = s.T, H = s.hidden
        let CONV = g.convDim, VH = g.vHeads, VALUE = g.vHeads * g.vDim
        // s.rot32 is already populated by hcMix; for T <= 8 inProjBA and qkvZ keep the T = 1
        // kernel family (rows-parameterized V4 GEMVs — bit-identical per-row reduction order).
        if T <= 8, let qkvZ = g.qkvZ {
            try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: g.inProjBA, x: s.rot32, y: s.gdnBA, rows: T)
            try denseProj(enc, hal: hal, s: s, weight: qkvZ, x: s.rot32, y: s.qkvZ32, rows: T)
        } else if let qkvZ = g.qkvZ {
            try denseProj(enc, hal: hal, s: s, weight: qkvZ, x: s.rot32, y: s.qkvZ32, rows: T,
                          x16Precast: T >= 64 ? s.naxX16 : nil)
            try QWGemm.encodeF32Gemv(enc, hal: hal, weight: g.inProjB, x: mixed, y: s.gdnB, rows: T, n: VH, k: H, epilogue: 0, scale: 1)
            try QWGemm.encodeF32Gemv(enc, hal: hal, weight: g.inProjA, x: mixed, y: s.gdnA, rows: T, n: VH, k: H, epilogue: 0, scale: 1)
        } else {
            try QWGemm.encodeF32Gemv(enc, hal: hal, weight: g.inProjB, x: mixed, y: s.gdnB, rows: T, n: VH, k: H, epilogue: 0, scale: 1)
            try QWGemm.encodeF32Gemv(enc, hal: hal, weight: g.inProjA, x: mixed, y: s.gdnA, rows: T, n: VH, k: H, epilogue: 0, scale: 1)
            barrier(enc)
            try denseProj(enc, hal: hal, s: s, weight: g.qkv, x: s.rot32, y: s.qkv32, rows: T,
                          x16Precast: T >= 64 ? s.naxX16 : nil)
            try denseProj(enc, hal: hal, s: s, weight: g.z, x: s.rot32, y: s.z32, rows: T,
                          x16Precast: T >= 64 ? s.naxX16 : nil)
        }
        barrier(enc)
        if T <= 8 {
            let p0 = g.convParity & QWFourSlotRollbackRing.ringMask
            g.convParity = (p0 + T) & QWFourSlotRollbackRing.ringMask
            let interleaved = g.qkvZ != nil
            let xStride = interleaved ? (CONV + VALUE) : CONV
            let baStride = interleaved ? 2 * VH : VH
            let aOff = (interleaved ? VH : s.maxT * VH) * 4
            let zOff = (interleaved ? CONV : s.maxT * CONV) * 4
            try QWGDNF32.encodeConvPrepF32(enc, hal: hal, x: s.qkvZ32, w: g.conv,
                                           stIn: g.convState[p0],
                                           stOut: g.convState[(p0 + 1) & QWFourSlotRollbackRing.ringMask],
                                           b: s.gdnBA,
                                           a: view(s.gdnBA, byteOffset: aOff, count: T * baStride - (aOff / 4), dtype: .f32),
                                           aLog: g.aLog, dtBias: g.dtBias,
                                           y: s.yDense32, glog: s.glog, beta: s.beta,
                                           rows: T, xStride: xStride, baStride: baStride,
                                           ringBase: g.convState[0], ringParity: p0)
            barrier(enc)
            if T == 1 {
                let stIn = g.states[g.stateParity & QWFourSlotRollbackRing.ringMask]
                try QWGDNF32.encodeRecurrentRowsF32(enc, hal: hal, y: s.yDense32, betas: s.beta, glog: s.glog,
                                                    state: stIn,
                                                    z: view(s.qkvZ32, byteOffset: zOff, count: VALUE, dtype: .f32),
                                                    zStride: interleaved ? CONV + VALUE : VALUE,
                                                    normWeight: g.normWeight, out: s.outNorm32, rows: 1)
            } else {
                let sp0 = g.stateParity & QWFourSlotRollbackRing.ringMask
                g.stateParity = (sp0 + T) & QWFourSlotRollbackRing.ringMask
                try QWGDNF32.encodeRecurrentRowsF32(enc, hal: hal, y: s.yDense32, betas: s.beta, glog: s.glog,
                                                    state: g.states[sp0],
                                                    z: view(s.qkvZ32, byteOffset: zOff, count: VALUE, dtype: .f32),
                                                    zStride: interleaved ? CONV + VALUE : VALUE,
                                                    normWeight: g.normWeight, out: s.outNorm32, rows: T,
                                                    ringBase: g.states[0], ringParity: sp0)
            }
            barrier(enc)
        } else {
            let p0 = g.convParity & QWFourSlotRollbackRing.ringMask
            let p1 = (p0 + 1) & QWFourSlotRollbackRing.ringMask
            let st = g.convState[p0], stNext = g.convState[p1]
            g.convParity = p1
            let interleaved = g.qkvZ != nil
            let xIn = interleaved ? s.qkvZ32 : s.qkv32
            let xStride = interleaved ? (CONV + VALUE) : CONV
            let zIn = interleaved ? view(s.qkvZ32, byteOffset: CONV * 4, count: T * (CONV + VALUE) - CONV, dtype: .f32) : s.z32
            let zStride = interleaved ? (CONV + VALUE) : VALUE
            try QWGDNF32.encodeConvPrepSeqF32(enc, hal: hal, x: xIn, w: g.conv, stIn: st, stOut: stNext,
                                              b: s.gdnB, a: s.gdnA, aLog: g.aLog, dtBias: g.dtBias,
                                              y: s.yDense32, glog: s.glog, beta: s.beta, rows: T, xStride: xStride)
            barrier(enc)
            try QWGDNF32.encodeRecurrentSeqF32(enc, hal: hal, y: s.yDense32, betas: s.beta, glog: s.glog,
                                               state: g.state, z: zIn, normWeight: g.normWeight, out: s.outNorm32,
                                               tokens: T, convDim: CONV, vHeads: VH, zStride: zStride,
                                               out16: T >= 64 ? s.naxX16 : nil)
            barrier(enc)
        }
        try denseProj(enc, hal: hal, s: s, weight: g.out, x: s.outNorm32, y: out, rows: T,
                      x16Precast: T >= 64 ? s.naxX16 : nil)
        barrier(enc)
    }

    /// The prefill attention over T rows, in f32, with the K/V written straight into the decode cache so the
    /// following decode steps read the prompt. The batched kernels take a per-row position (pos0 + row).
    static func attentionPrefill(_ enc: any MTLComputeCommandEncoder, ctx: FNGPUContext, a: FNGPUAttnMixer,
                                 c: FNGPUAttnDecodeState, s: FNGPUScratch, pos: Int, out: QWTensorValue,
                                 cacheOnly: Bool = false, lastRowOnly: Bool = false) throws {
        let hal = ctx.hal, T = s.T
        let NH = a.heads, NKV = a.kvHeads, D = a.headDim, ROT = a.rotaryDim, ID = a.indexHeadDim
        try denseProj(enc, hal: hal, s: s, weight: a.bank, x: s.rot32, y: s.src, rows: T,
                      x16Precast: T >= 64 ? s.naxX16 : nil)
        barrier(enc)
        let cos = view(c.cos, byteOffset: pos * ROT * 4, count: T * ROT, dtype: .f32)
        let sin = view(c.sin, byteOffset: pos * ROT * 4, count: T * ROT, dtype: .f32)
        let needSelect = (pos + T) / a.compress > a.blockTopK
        let selectStart = (lastRowOnly && T > 1) ? (T - 1) : max(0, a.blockTopK * a.compress - pos)
        let selectRows = T - selectStart, idxKView = view(c.indexK, byteOffset: pos * ID * 2, count: T * ID, dtype: .f16)
        let kv = c.kvViews(pos: pos, count: T, kvHeads: NKV, headDim: D)
        let prepIdxK: QWTensorValue? = (needSelect && !cacheOnly && selectStart == 0) ? nil : idxKView
        try QWGemm.encodeFnAttnPrepF32(enc, hal: hal, src: s.src, qNorm: a.qNorm, kNorm: a.kNorm, cos: cos, sin: sin,
                                       q: s.q32, k: kv.k, v: kv.v, gate: s.gateF32, rows: T, heads: NH, kvHeads: NKV,
                                       rowStride: a.bankRows, idxK: prepIdxK, ikOff: a.ikOff,
                                       kvMode: c.kvMode, kMeta: kv.kMeta, vMeta: kv.vMeta)
        if needSelect && !cacheOnly && selectRows > 0 {
            let srcSel = view(s.src, byteOffset: selectStart * a.bankRows * 4, count: selectRows * a.bankRows, dtype: .f32)
            let cosSel = view(c.cos, byteOffset: (pos + selectStart) * ROT * 4, count: selectRows * ROT, dtype: .f32)
            let sinSel = view(c.sin, byteOffset: (pos + selectStart) * ROT * 4, count: selectRows * ROT, dtype: .f32)
            let idxQSel = view(s.idxQ, byteOffset: selectStart * a.indexHeads * ID * 4, count: selectRows * a.indexHeads * ID, dtype: .f32)
            let idxKSel = view(c.indexK, byteOffset: (pos + selectStart) * ID * 2, count: selectRows * ID, dtype: .f16)
            try QWGemm.encodeFnIndexPrep(enc, hal: hal, src: srcSel, iqNorm: a.iqNorm, cos: cosSel, sin: sinSel,
                                         idxQ: idxQSel, idxK: idxKSel, rows: selectRows, rowStride: a.bankRows, iqOff: a.iqOff, ikOff: a.ikOff)
        }
        barrier(enc)
        let b0 = pos / a.compress, b1 = (pos + T) / a.compress, newBlocks = b1 - b0
        if newBlocks > 0 {
            try QWGemm.encodeFnIndexPool(enc, hal: hal,
                                         idxK: view(c.indexK, byteOffset: b0 * a.compress * ID * 2, count: newBlocks * a.compress * ID, dtype: .f16),
                                         ikNorm: a.ikNorm,
                                         bcos: view(c.bcos, byteOffset: b0 * ROT * 4, count: newBlocks * ROT, dtype: .f32),
                                         bsin: view(c.bsin, byteOffset: b0 * ROT * 4, count: newBlocks * ROT, dtype: .f32),
                                         pooled: view(c.pooled, byteOffset: b0 * ID * 2, count: newBlocks * ID, dtype: .f16),
                                         blocks: newBlocks, compress: a.compress)
            if let clusters = c.lsaClusters {
                _ = try QWGemm.encodeFnLSAClusterRange(enc, hal: hal, pooled: c.pooled, clusters: clusters,
                                                       pos0: pos, count: T, indexDim: ID)
            }
            if needSelect { barrier(enc) }
        }
        if cacheOnly { barrier(enc); return }
        if lastRowOnly && T > 1 {
            let r = T - 1
            let posLast = pos + r
            let qLast = view(s.q32, byteOffset: r * NH * D * 4, count: NH * D, dtype: .f32)
            let gateLast = view(s.gateF32, byteOffset: r * NH * D * 4, count: NH * D, dtype: .f32)
            let listLast = view(s.list, byteOffset: r * a.blockTopK * 4, count: a.blockTopK, dtype: .i32)
            let attnLast = view(s.attnOut32, byteOffset: r * NH * D * 4, count: NH * D, dtype: .f32)
            let outLast = view(out, byteOffset: r * s.hidden * 4, count: s.hidden, dtype: .f32)
            if needSelect {
                let idxQLast = view(s.idxQ, byteOffset: r * a.indexHeads * ID * 4, count: a.indexHeads * ID, dtype: .f32)
                try QWGemm.encodeFnQSASelect(enc, hal: hal, idxQ: idxQLast, pooled: c.pooled, list: listLast, rows: 1,
                                             blockTopK: a.blockTopK, pos0: posLast, scores: s.qsaScores,
                                             lsaClusters: c.lsaClusters)
                barrier(enc)
            }
            try QWGemm.encodeFnAttnSparseGatedF32(enc, hal: hal, q: qLast, k: c.k, v: c.v, gate: gateLast, list: listLast,
                                                  out: attnLast, rows: 1, heads: NH, kvHeads: NKV, pos0: posLast,
                                                  kvMode: c.kvMode, kMeta: c.kMeta, vMeta: c.vMeta)
            barrier(enc)
            try denseProj(enc, hal: hal, s: s, weight: a.oProj, x: attnLast, y: outLast, rows: 1)
            barrier(enc)
            return
        }
        if needSelect && selectRows > 0 {
            let idxQSel = view(s.idxQ, byteOffset: selectStart * a.indexHeads * ID * 4, count: selectRows * a.indexHeads * ID, dtype: .f32)
            let listSel = view(s.list, byteOffset: selectStart * a.blockTopK * 4, count: selectRows * a.blockTopK, dtype: .i32)
            try QWGemm.encodeFnQSASelect(enc, hal: hal, idxQ: idxQSel, pooled: c.pooled, list: listSel, rows: selectRows,
                                         blockTopK: a.blockTopK, pos0: pos + selectStart, scores: s.qsaScores,
                                         lsaClusters: c.lsaClusters)
            barrier(enc)
        }
        try QWGemm.encodeFnAttnSparseGatedF32(enc, hal: hal, q: s.q32, k: c.k, v: c.v, gate: s.gateF32, list: s.list,
                                              out: s.attnOut32, rows: T, heads: NH, kvHeads: NKV, pos0: pos,
                                              kvMode: c.kvMode, kMeta: c.kMeta, vMeta: c.vMeta,
                                              out16: T >= 64 ? s.naxX16 : nil)
        barrier(enc)
        try denseProj(enc, hal: hal, s: s, weight: a.oProj, x: s.attnOut32, y: out, rows: T,
                      x16Precast: T >= 64 ? s.naxX16 : nil)
        barrier(enc)
    }

    /// One decode-path attention row (`row` of a row-tiled state; row 0 for T = 1). The K/V and index
    /// caches are absolute-indexed by `pos`; the new K/V goes at `pos`, QSA + sparse attention read `pos0`.
    /// Every scratch buffer is sliced to this row; output is the o_proj row f32 [hidden].
    static func attentionDecode(_ enc: any MTLComputeCommandEncoder, ctx: FNGPUContext, a: FNGPUAttnMixer,
                                c: FNGPUAttnDecodeState, mixed: QWTensorValue, pos: Int, out: QWTensorValue,
                                row: Int = 0) throws {
        _ = mixed
        let hal = ctx.hal
        let NH = a.heads, NKV = a.kvHeads, D = a.headDim, ROT = a.rotaryDim, ID = a.indexHeadDim
        let NI = a.indexHeads, BK = a.blockTopK, BN = a.bankRows, H = c.hidden
        let rRot = view(c.rot32, byteOffset: row * H * 4, count: H, dtype: .f32)
        let rSrc = view(c.src, byteOffset: row * BN * 4, count: BN, dtype: .f32)
        let rQ = view(c.q, byteOffset: row * NH * D * 4, count: NH * D, dtype: .f32)
        let rGate = view(c.gateF, byteOffset: row * NH * D * 4, count: NH * D, dtype: .f32)
        let rIdxQ = view(c.idxQ, byteOffset: row * NI * ID * 4, count: NI * ID, dtype: .f32)
        let rList = view(c.list, byteOffset: row * BK * 4, count: BK, dtype: .i32)
        let rAttn = view(c.attnOut, byteOffset: row * NH * D * 4, count: NH * D, dtype: .f32)
        let rOut = view(out, byteOffset: row * H * 4, count: H, dtype: .f32)
        let cosP = view(c.cos, byteOffset: pos * ROT * 4, count: ROT, dtype: .f32)
        let sinP = view(c.sin, byteOffset: pos * ROT * 4, count: ROT, dtype: .f32)
        try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: a.bank, x: rRot, y: rSrc)
        barrier(enc)
        let needSelect = (pos + 1) / a.compress > a.blockTopK
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
        barrier(enc)
        if (pos + 1) % a.compress == 0 {
            let b = (pos + 1) / a.compress - 1
            try QWGemm.encodeFnIndexPool(enc, hal: hal,
                                         idxK: view(c.indexK, byteOffset: b * a.compress * ID * 2, count: a.compress * ID, dtype: .f16),
                                         ikNorm: a.ikNorm,
                                         bcos: view(c.bcos, byteOffset: b * ROT * 4, count: ROT, dtype: .f32),
                                         bsin: view(c.bsin, byteOffset: b * ROT * 4, count: ROT, dtype: .f32),
                                         pooled: view(c.pooled, byteOffset: b * ID * 2, count: ID, dtype: .f16),
                                         blocks: 1, compress: a.compress)
            if let clusters = c.lsaClusters {
                _ = try QWGemm.encodeFnLSAClusterRange(enc, hal: hal, pooled: c.pooled, clusters: clusters,
                                                       pos0: pos, count: 1, indexDim: ID)
            }
            if needSelect { barrier(enc) }
        }
        if needSelect {
            let maxBlocks = max(1, c.qsaScores.shape.elementCount / max(1, c.rows))
            let rScores = view(c.qsaScores, byteOffset: row * maxBlocks * 4, count: maxBlocks, dtype: .i32)
            try QWGemm.encodeFnQSASelect(enc, hal: hal, idxQ: rIdxQ, pooled: c.pooled, list: rList, rows: 1,
                                         blockTopK: a.blockTopK, pos0: pos, scores: rScores,
                                         lsaClusters: c.lsaClusters)
            barrier(enc)
        }
        try QWGemm.encodeFnAttnSparseGatedF32(enc, hal: hal, q: rQ, k: c.k, v: c.v, gate: rGate, list: rList,
                                              out: rAttn, rows: 1, heads: NH, kvHeads: NKV, pos0: pos,
                                              kvMode: c.kvMode, kMeta: c.kMeta, vMeta: c.vMeta)
        barrier(enc)
        try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: a.oProj, x: rAttn, y: rOut)
        barrier(enc)
    }

    /// Verify rows of a full-attention layer: rows t = 0..<T sit at positions pos + t. Projections, K/V/index
    /// writes, sparse attention and o_proj run once for all rows; index pooling runs per completed block.
    static func attentionDecodeRows(_ enc: any MTLComputeCommandEncoder, ctx: FNGPUContext, a: FNGPUAttnMixer,
                                    c: FNGPUAttnDecodeState, pos: Int, out: QWTensorValue, rows T: Int) throws {
        let needFirst = (pos + 1) / a.compress > a.blockTopK, needLast = (pos + T) / a.compress > a.blockTopK
        if T == 1 || (needFirst != needLast) {
            for t in 0..<T {
                try attentionDecode(enc, ctx: ctx, a: a, c: c, mixed: out, pos: pos + t, out: out, row: t)
            }
            return
        }
        let needSelect = needFirst
        let hal = ctx.hal
        let NH = a.heads, NKV = a.kvHeads, D = a.headDim, ROT = a.rotaryDim, ID = a.indexHeadDim
        let cosView = view(c.cos, byteOffset: pos * ROT * 4, count: T * ROT, dtype: .f32)
        let sinView = view(c.sin, byteOffset: pos * ROT * 4, count: T * ROT, dtype: .f32)
        let idxKView = view(c.indexK, byteOffset: pos * ID * 2, count: T * ID, dtype: .f16)
        let kv = c.kvViews(pos: pos, count: T, kvHeads: NKV, headDim: D)
        try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: a.bank, x: c.rot32, y: c.src, rows: T)
        barrier(enc)
        try QWGemm.encodeFnAttnPrepF32(enc, hal: hal, src: c.src, qNorm: a.qNorm, kNorm: a.kNorm,
                                       cos: cosView, sin: sinView,
                                       q: c.q, k: kv.k, v: kv.v,
                                       gate: c.gateF, rows: T, heads: NH, kvHeads: NKV, rowStride: a.bankRows,
                                       idxK: needSelect ? nil : idxKView,
                                       ikOff: a.ikOff,
                                       kvMode: c.kvMode, kMeta: kv.kMeta, vMeta: kv.vMeta)
        if needSelect {
            try QWGemm.encodeFnIndexPrep(enc, hal: hal, src: c.src, iqNorm: a.iqNorm, cos: cosView, sin: sinView,
                                         idxQ: c.idxQ, idxK: idxKView, rows: T, rowStride: a.bankRows,
                                         iqOff: a.iqOff, ikOff: a.ikOff)
        }
        barrier(enc)
        for t in 0..<T where (pos + t + 1) % a.compress == 0 {
            let b = (pos + t + 1) / a.compress - 1
            try QWGemm.encodeFnIndexPool(enc, hal: hal,
                                         idxK: view(c.indexK, byteOffset: b * a.compress * ID * 2, count: a.compress * ID, dtype: .f16),
                                         ikNorm: a.ikNorm,
                                         bcos: view(c.bcos, byteOffset: b * ROT * 4, count: ROT, dtype: .f32),
                                         bsin: view(c.bsin, byteOffset: b * ROT * 4, count: ROT, dtype: .f32),
                                         pooled: view(c.pooled, byteOffset: b * ID * 2, count: ID, dtype: .f16),
                                         blocks: 1, compress: a.compress)
            if let clusters = c.lsaClusters {
                _ = try QWGemm.encodeFnLSAClusterRange(enc, hal: hal, pooled: c.pooled, clusters: clusters,
                                                       pos0: pos + t, count: 1, indexDim: ID)
            }
            barrier(enc)
        }
        if needSelect {
            try QWGemm.encodeFnQSASelect(enc, hal: hal, idxQ: c.idxQ, pooled: c.pooled, list: c.list, rows: T,
                                         blockTopK: a.blockTopK, pos0: pos, scores: c.qsaScores,
                                         lsaClusters: c.lsaClusters)
            barrier(enc)
        }
        try QWGemm.encodeFnAttnSparseGatedF32(enc, hal: hal, q: c.q, k: c.k, v: c.v, gate: c.gateF, list: c.list,
                                              out: c.attnOut, rows: T, heads: NH, kvHeads: NKV, pos0: pos,
                                              kvMode: c.kvMode, kMeta: c.kMeta, vMeta: c.vMeta)
        barrier(enc)
        try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: a.oProj, x: c.attnOut, y: out, rows: T)
        barrier(enc)
    }

    static func moe(_ enc: inout any MTLComputeCommandEncoder, ctx: FNGPUContext, m: FNGPUMoE, s: FNGPUScratch,
                    mixed: QWTensorValue, out: QWTensorValue, rows: Int? = nil,
                    cb: (any MTLCommandBuffer)? = nil) throws {
        let T = rows ?? s.T
        let weights = QWMoEOperator.Weights(
            router: m.gateAndSharedBF16,
            decodeRouterOverride: QWGemm.encodeFnRouterBF16Decode,
            gateUp: m.gateUp,
            down: m.down,
            sharedGateUp: m.sharedGateUp,
            sharedDown: m.sharedDown,
            sharedGateScalar: m.sharedGateScalar,
            hidden: s.hidden,
            experts: m.experts,
            topK: m.topK,
            intermediate: m.intermediate,
            sharedIntermediate: m.sharedIntermediate,
            normTopK: true,
            reuseExpertsAcrossTokens: false
        )
        let scratch = QWMoEOperator.Scratch(
            xRot32: s.rot32, xPlain32: mixed,
            logits: s.logits, idx: s.idx, wts: s.wRoute,
            gu32: s.gu32, act32: s.act32,
            guSh32: s.guSh32, actSh32: s.actSh32,
            shared: s.sh, gate: s.sg,
            expOff: s.expOff, sortedPair: s.sortedPair,
            sortedTok: s.sortedTok, invPerm: s.invPerm,
            tileE: s.tileE, tileR: s.tileR, tileCount: s.tileCount,
            gu16: s.gu, dnHi: s.dnHi, dnAcc: s.dnAcc,
            naxX16: s.naxX16, naxY16: s.naxY16, maxTileWidth: s.width
        )
        let rotateFn: ((inout any MTLComputeCommandEncoder) throws -> Void)? =
            cb.map { buf in { try ctx.hal.rotateEncoder(buf, &$0) } }
        try QWMoEOperator.encodeResident(
            &enc, hal: ctx.hal, weights: weights, scratch: scratch,
            input: .f32PreRotated(plain: mixed, rotated: s.rot32, prefillPrecast16: T >= 64 ? s.naxX16 : nil),
            out32: out, rows: T, supportsGroupedNaxPrefill: true, concurrentBarriers: true,
            rotateEncoder: rotateFn
        )
    }

    /// One decoder layer for a prefill chunk of T tokens. The attention layers run the cached decode attention per token,
    /// which fills the KV and index caches the following decode steps read; every other stage is batched over the T rows.
    public static func layerPrefill(_ enc: inout any MTLComputeCommandEncoder, cb: any MTLCommandBuffer,
                                     ctx: FNGPUContext, layer L: FNGPULayerWeights,
                                     s: FNGPUScratch, stream: QWTensorValue, out: QWTensorValue,
                                     attnState: FNGPUAttnDecodeState?, pos: Int, pleEmbed: QWTensorValue? = nil,
                                     inInject: (base: QWTensorValue, src: QWTensorValue, scale: QWTensorValue)? = nil,
                                     deferOutInject: Bool = false) throws {
        let hal = ctx.hal, T = s.T, H = s.hidden
        var h = stream
        var attnInject = inInject
        if let p = L.ple {
            if let inj = attnInject {
                try QWHCOperator.encodeInject(enc, hal: hal, base: inj.base, src: inj.src, scale: inj.scale,
                                              out: stream, rows: T, hidden: H, hc: s.hc)
                attnInject = nil
            }
            try QWPLEOperator.encode(enc, hal: hal, p: p, s: s, stream: stream, out: s.h, embed: pleEmbed)
            h = s.h
        }
        try QWHCOperator.encodeMix(enc, hal: hal, hc: L.attnHC, s: s, stream: h, mixed: s.mixedA, inj: s.injA,
                                   injectBase: attnInject?.base, injectSrc: attnInject?.src, injectScale: attnInject?.scale)
        switch L.mixer {
        case .gdn(let g):
            try gdn(enc, ctx: ctx, g: g, s: s, mixed: s.mixedA, out: s.oG)
        case .attention(let a):
            if let c = attnState {
                try QWQSAOperator.encodePrefill(enc, ctx: ctx, a: a, c: c, s: s, pos: pos, out: s.oG)
            }
        }
        try QWHCOperator.encodeMix(enc, hal: hal, hc: L.mlpHC, s: s, stream: s.h2, mixed: s.mixedB, inj: s.injB,
                                   injectBase: h, injectSrc: s.oG, injectScale: s.injA)
        try moe(&enc, ctx: ctx, m: L.moe, s: s, mixed: s.mixedB, out: s.moeOut, cb: cb)
        if !deferOutInject {
            try QWHCOperator.encodeInject(enc, hal: hal, base: s.h2, src: s.moeOut, scale: s.injB,
                                          out: out, rows: T, hidden: H, hc: s.hc)
        }
    }

    /// One decoder layer: stream in (f32 [T, W]) -> stream out. PLE (when present) runs first, then attention-side HC and mixer,
    /// injection, MLP-side HC and MoE, second injection. With `attnState` set, attention is the cached decode step at `pos`.
    public static func layer(_ enc: any MTLComputeCommandEncoder, ctx: FNGPUContext, layer L: FNGPULayerWeights,
                             s: FNGPUScratch, stream: QWTensorValue, out: QWTensorValue,
                             attnState: FNGPUAttnDecodeState? = nil, pos: Int = 0,
                             inInject: (base: QWTensorValue, src: QWTensorValue, scale: QWTensorValue)? = nil,
                             deferOutInject: Bool = false) throws {
        let hal = ctx.hal
        var h = stream
        var attnInject = inInject
        if let p = L.ple {
            if let inj = attnInject {
                try QWHCOperator.encodeInject(enc, hal: hal, base: inj.base, src: inj.src, scale: inj.scale,
                                              out: stream, rows: s.T, hidden: s.hidden, hc: s.hc)
                attnInject = nil
            }
            try QWPLEOperator.encode(enc, hal: hal, p: p, s: s, stream: stream, out: s.h)
            h = s.h
        }
        try QWHCOperator.encodeMix(enc, hal: hal, hc: L.attnHC, s: s, stream: h, mixed: s.mixedA, inj: s.injA,
                                   rotMixed: attnState?.rot32,
                                   injectBase: attnInject?.base, injectSrc: attnInject?.src, injectScale: attnInject?.scale)
        switch L.mixer {
        case .gdn(let g):
            try gdn(enc, ctx: ctx, g: g, s: s, mixed: s.mixedA, out: s.oG)
        case .attention(let a):
            if let c = attnState {
                // One decode row per token: row t attends causally at absolute position pos + t.
                try QWQSAOperator.encodeDecodeRows(enc, ctx: ctx, a: a, c: c, pos: pos, out: s.oG, rows: s.T)
            }
        }
        try QWHCOperator.encodeMix(enc, hal: hal, hc: L.mlpHC, s: s, stream: s.h2, mixed: s.mixedB, inj: s.injB,
                                   injectBase: h, injectSrc: s.oG, injectScale: s.injA)
        var mEnc = enc
        try moe(&mEnc, ctx: ctx, m: L.moe, s: s, mixed: s.mixedB, out: s.moeOut)
        if !deferOutInject {
            try QWHCOperator.encodeInject(enc, hal: hal, base: s.h2, src: s.moeOut, scale: s.injB,
                                          out: out, rows: s.T, hidden: s.hidden, hc: s.hc)
        }
    }
}
