// QWDenseLayerKernels.swift — The dense engine's layer kinds.
//
// The branches that used to live in the decode and prefill loops. The code
// is the same; what changed is where it is decided: the plan resolves the
// kind once, and the loop looks the kernel up.

import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWMLP
import QWHead
import QWQuant
import QWState

/// The part of a dense layer every kind shares: the residual norm and the
/// MLP. The attention is the kind's own.
class DenseLayerKernel: @unchecked Sendable {
    /// The kind's attention, for one decode step.
    func encodeAttention(_ context: QWLayerDecodeContext) throws {
        throw QWEngineDriverError.state("a dense layer kernel must encode its attention")
    }

    /// The kind's attention, for one prefill chunk.
    func encodeAttentionPrefill(_ context: QWLayerPrefillContext) throws {
        throw QWEngineDriverError.state("a dense layer kernel must encode its attention")
    }

    /// The layer's MLP for one decode step: the dense SwiGLU. A sparse-MoE
    /// layer overrides it — the rest of the layer is shared.
    func encodeMLP(_ context: QWLayerDecodeContext) throws {
        let driver = context.driver
        let enc = context.enc
        let sc = driver.sc
        let geometry = driver.geometry
        let view = context.weights.gu.view
        let fuseFwht = !view.isIncumbentContainer && geometry.intermediate > 0 && (geometry.intermediate % 32 == 0)
        // MLP (M=1 route: silu_pair_dx_h16 + fwht32)
        driver.mlpSiluPairH16(enc, x: sc.ln2, gs: sc.gs, wQ4: view,
                              h16: sc.gu16, fuseFwht32: fuseFwht)
        if !fuseFwht {
            driver.encodeFwht32Fast(enc, x: sc.gu16)
        }
        try driver.gemvH16T4(enc, n: geometry.hidden, k: geometry.intermediate, x: sc.gu16, gs: sc.gsI,
                      wQ4: context.weights.down, y16: sc.dn16)
    }

    /// The layer's MLP for one prefill chunk: the dense SwiGLU.
    func encodeMLPPrefill(_ context: QWLayerPrefillContext) throws {
        let driver = context.driver
        let enc = context.enc
        let psc = driver.psc
        let geometry = driver.geometry
        let lw = context.weights
        let H = geometry.hidden
        let I = geometry.intermediate
        let T = context.rows
        driver.gemmBatchedH16(enc, m: T, n: 2 * I, k: H,
                       x: driver.prefixRows(psc.ln2, rows: T, cols: H, dt: .f16),
                       wQ4: lw.gu.view,
                       y16: driver.prefixRows(psc.guRaw16, rows: T, cols: 2 * I, dt: .f16))
        driver.encodeSiluMulF16(enc,
                         gu: driver.prefixRows(psc.guRaw16, rows: T, cols: 2 * I, dt: .f16),
                         out: driver.prefixRows(psc.gu16, rows: T, cols: I, dt: .f16),
                         m: T, i: I)
        driver.gemmBatchedH16(enc, m: T, n: H, k: I,
                       x: driver.prefixRows(psc.gu16, rows: T, cols: I, dt: .f16),
                       wQ4: lw.down,
                       y16: driver.prefixRows(psc.dn16, rows: T, cols: H, dt: .f16))
    }

    /// Stage label for QW_DECODE_PROFILE (the concrete layer-kernel type).
    static var profileKind: String { String(describing: self) }

    func encodeDecode(_ context: QWLayerDecodeContext) throws {
        let driver = context.driver
        var enc = driver.activeEnc ?? context.enc
        let sc = driver.sc
        let geometry = driver.geometry
        let l = context.layer
        let lw = context.weights
        let plan = geometry.executionPlan
        let p = plan.indices.contains(l) ? plan[l] : l
        let nextP = (l + 1 < plan.count) ? plan[l + 1] : nil
        let nextGamma = nextP.map { driver.layersW[$0].ln1 } ?? driver.finalNorm

        if let mhc = lw.mhc, let mhcCfg = geometry.mhcConfig,
           let mhcStream = sc.mhcStream, let mhcControls = sc.mhcControls {
            let prevP = l > 0 ? (plan.indices.contains(l - 1) ? plan[l - 1] : (l - 1)) : -1
            let hasDeferredPrevFFN = (p != mhcCfg.start) && (prevP >= mhcCfg.start && prevP < mhcCfg.end - 1)
            driver.profileStage("mhc.router_norm")
            enc = driver.activeEnc ?? context.enc
            if p == mhcCfg.start {
                try driver.encodeMHCEnter(enc, x: sc.x, stream: mhcStream, rows: 1)
            }
            try driver.encodeMHCRouterNorm(enc, stream: mhcStream, w: mhc.attnW, b: mhc.attnB,
                                           gamma: lw.ln1, controls: mhcControls, out: sc.ln1,
                                           rows: 1, mergeBranchOut: hasDeferredPrevFFN ? sc.dn16 : nil)
            driver.profileStage("attn." + Self.profileKind)
            try encodeAttention(context)
            driver.profileStage("mhc.router_norm")
            enc = driver.activeEnc ?? context.enc
            try driver.encodeMHCRouterNorm(enc, stream: mhcStream, w: mhc.ffnW, b: mhc.ffnB,
                                           gamma: lw.ln2, controls: mhcControls, out: sc.ln2,
                                           rows: 1, mergeBranchOut: sc.o16)
            driver.profileStage("mlp." + Self.profileKind)
            try encodeMLP(context)
            driver.profileStage("mhc.merge")
            enc = driver.activeEnc ?? context.enc
            let isLeave = (p == mhcCfg.end - 1)
            let nextIsMHC = nextP.map { $0 >= mhcCfg.start && $0 < mhcCfg.end } ?? false
            let deferPostFFN = !isLeave && nextIsMHC && (nextP != mhcCfg.start)
            if !deferPostFFN {
                if isLeave, mhcCfg.hasExitHead, let exitH = driver.mhcExitHead {
                    try driver.encodeMHCMerge(enc, stream: mhcStream, branchOut: sc.dn16,
                                              controls: mhcControls, xOut: sc.x,
                                              nextGamma: nextGamma, nextNormOut: sc.ln1,
                                              rows: 1, isLeave: false, sinkhornIters: mhcCfg.iterations,
                                              hasNextNorm: false)
                    try driver.encodeMHCExitHead(enc, stream: mhcStream, headW: exitH.headW, headB: exitH.headB,
                                                 normGamma: driver.finalNorm, outNorm: sc.ln1,
                                                 rows: 1, applyNorm: true)
                } else {
                    try driver.encodeMHCMerge(enc, stream: mhcStream, branchOut: sc.dn16,
                                              controls: mhcControls, xOut: sc.x,
                                              nextGamma: nextGamma, nextNormOut: sc.ln1,
                                              rows: 1, isLeave: isLeave, sinkhornIters: mhcCfg.iterations,
                                              hasNextNorm: isLeave)
                }
            }
        } else {
            // ln1 for layer 0 ONLY: the plain norm. For l ≥ 1 the stream
            // already arrives normalized — rung 2 of layer l−1 emitted
            // rmsnorm(x)·γ_ln1(l) folded into its fused residual launch
            // (exactly the oracle's `x = x + attn+mlp` then
            // `input_layernorm(x)`; a second norm would double-norm).
            if l == 0 {
                try driver.encodeRMSNormFast(enc, x: sc.x, gamma: lw.ln1, out: sc.ln1)
            }
            try encodeAttention(context)
            enc = driver.activeEnc ?? context.enc
            // ── shared tail (leg-2c rung order, parity-proven) ─────────────
            try driver.encodeRMSNormResidualFast(enc, x: sc.o16, gamma: lw.ln2, out: sc.ln2, residual: sc.x)
            try encodeMLP(context)
            enc = driver.activeEnc ?? context.enc
            // Rung 2: x := half(x + mlp) IN PLACE; the emitted norm uses the
            // NEXT layer's ln1 gamma (final norm at l = 63) — the oracle's
            // `x = x + mlp(...)` followed by the next layer's
            try driver.encodeRMSNormResidualFast(enc, x: sc.dn16, gamma: nextGamma, out: sc.ln1, residual: sc.x)
        }
        if let dflashDrafter = driver.dflashDrafter, p < dflashDrafter.targetLayerSlotMap.count {
            let slot = dflashDrafter.targetLayerSlotMap[p]
            if slot >= 0 {
                dflashDrafter.encodeCaptureTargetHidden(
                    enc, source: sc.x, srcRowOffset: 0, dstRowOffset: 0, rows: 1, slot: slot)
            }
        }
    }

    func encodePrefillChunk(_ context: QWLayerPrefillContext) throws {
        let driver = context.driver
        var enc = driver.activeEnc ?? context.enc
        let psc = driver.psc
        let geometry = driver.geometry
        let l = context.layer
        let lw = context.weights
        let H = geometry.hidden
        let T = context.rows
        let plan = geometry.executionPlan
        let p = plan.indices.contains(l) ? plan[l] : l
        let nextP = (l + 1 < plan.count) ? plan[l + 1] : nil
        let nextGamma = nextP.map { driver.layersW[$0].ln1 } ?? driver.finalNorm

        if let mhc = lw.mhc, let mhcCfg = geometry.mhcConfig,
           let mhcStream = psc.mhcStream, let mhcControls = psc.mhcControls {
            let activeX = driver.prefixRows(psc.x, rows: T, cols: H, dt: .f16)
            let activeLn1 = driver.prefixRows(psc.ln1, rows: T, cols: H, dt: .f16)
            let activeLn2 = driver.prefixRows(psc.ln2, rows: T, cols: H, dt: .f16)
            let activeO16 = driver.prefixRows(psc.o16, rows: T, cols: H, dt: .f16)
            let activeDn16 = driver.prefixRows(psc.dn16, rows: T, cols: H, dt: .f16)
            let prevP = l > 0 ? (plan.indices.contains(l - 1) ? plan[l - 1] : (l - 1)) : -1
            let hasDeferredPrevFFN = (p != mhcCfg.start) && (prevP >= mhcCfg.start && prevP < mhcCfg.end - 1)
            driver.profileStage("mhc.router_norm")
            enc = driver.activeEnc ?? context.enc
            if p == mhcCfg.start {
                try driver.encodeMHCEnter(enc, x: activeX, stream: mhcStream, rows: T)
            }
            try driver.encodeMHCRouterNorm(enc, stream: mhcStream, w: mhc.attnW, b: mhc.attnB,
                                           gamma: lw.ln1, controls: mhcControls, out: activeLn1,
                                           rows: T, mergeBranchOut: hasDeferredPrevFFN ? activeDn16 : nil)
            driver.profileStage("attn." + Self.profileKind)
            try encodeAttentionPrefill(context)
            driver.profileStage("mhc.router_norm")
            enc = driver.activeEnc ?? context.enc
            try driver.encodeMHCRouterNorm(enc, stream: mhcStream, w: mhc.ffnW, b: mhc.ffnB,
                                           gamma: lw.ln2, controls: mhcControls, out: activeLn2,
                                           rows: T, mergeBranchOut: activeO16)
            driver.profileStage("mlp." + Self.profileKind)
            try encodeMLPPrefill(context)
            driver.profileStage("mhc.merge")
            enc = driver.activeEnc ?? context.enc
            let isLeave = (p == mhcCfg.end - 1)
            let nextIsMHC = nextP.map { $0 >= mhcCfg.start && $0 < mhcCfg.end } ?? false
            let deferPostFFN = !isLeave && nextIsMHC && (nextP != mhcCfg.start)
            if !deferPostFFN {
                if isLeave, mhcCfg.hasExitHead, let exitH = driver.mhcExitHead {
                    try driver.encodeMHCMerge(enc, stream: mhcStream, branchOut: activeDn16,
                                              controls: mhcControls, xOut: activeX,
                                              nextGamma: nextGamma, nextNormOut: activeLn1,
                                              rows: T, isLeave: false, sinkhornIters: mhcCfg.iterations,
                                              hasNextNorm: false)
                    try driver.encodeMHCExitHead(enc, stream: mhcStream, headW: exitH.headW, headB: exitH.headB,
                                                 normGamma: driver.finalNorm, outNorm: activeLn1,
                                                 rows: T, applyNorm: true)
                } else {
                    try driver.encodeMHCMerge(enc, stream: mhcStream, branchOut: activeDn16,
                                              controls: mhcControls, xOut: activeX,
                                              nextGamma: nextGamma, nextNormOut: activeLn1,
                                              rows: T, isLeave: isLeave, sinkhornIters: mhcCfg.iterations,
                                              hasNextNorm: isLeave)
                }
            }
        } else {
            if l == 0 {
                try driver.encodeRMSNormFast(enc,
                                  x: driver.prefixRows(psc.x, rows: T, cols: H, dt: .f16),
                                  gamma: lw.ln1,
                                  out: driver.prefixRows(psc.ln1, rows: T, cols: H, dt: .f16),
                                  rows: T)
            }
            try encodeAttentionPrefill(context)
            enc = driver.activeEnc ?? context.enc
            try driver.encodeRMSNormResidualFast(enc,
                                      x: driver.prefixRows(psc.o16, rows: T, cols: H, dt: .f16),
                                      gamma: lw.ln2,
                                      out: driver.prefixRows(psc.ln2, rows: T, cols: H, dt: .f16),
                                      residual: driver.prefixRows(psc.x, rows: T, cols: H, dt: .f16),
                                      rows: T, rowParallel: false)
            try encodeMLPPrefill(context)
            enc = driver.activeEnc ?? context.enc
            try driver.encodeRMSNormResidualFast(enc,
                                      x: driver.prefixRows(psc.dn16, rows: T, cols: H, dt: .f16),
                                      gamma: nextGamma,
                                      out: driver.prefixRows(psc.ln1, rows: T, cols: H, dt: .f16),
                                      residual: driver.prefixRows(psc.x, rows: T, cols: H, dt: .f16),
                                      rows: T, rowParallel: false)
        }
        if let dflashDrafter = driver.dflashDrafter, p < dflashDrafter.targetLayerSlotMap.count {
            let slot = dflashDrafter.targetLayerSlotMap[p]
            if slot >= 0 {
                let capRows = min(T, dflashDrafter.config.maxCacheSize)
                dflashDrafter.encodeCaptureTargetHidden(
                    enc, source: psc.x, srcRowOffset: T - capRows, dstRowOffset: 0, rows: capRows, slot: slot)
            }
        }
    }
}

/// The GDN layer: linear attention with a gated norm.
final class GDNLayerKernel: DenseLayerKernel, QWLayerKernel, @unchecked Sendable {
    static let key = QWEnginePlan.LayerKernelKey(attention: .gdn, mlp: .dense, ple: false)

    static func make() -> any QWLayerKernel { GDNLayerKernel() }

    override func encodeAttention(_ context: QWLayerDecodeContext) throws {
        let driver = context.driver
        var enc = context.enc
        func mark(_ stage: String) { driver.profileStage(stage); if let a = driver.activeEnc { enc = a } }
        let sc = driver.sc
        let geometry = driver.geometry
        let l = context.layer
        let w = context.weights.gdn!
        let H = geometry.hidden
        if geometry.gdnVectorDecay, let fgbProj = w.fgbProj, let fgb16 = sc.fgb16,
           let fB = w.fB, let gB = w.gB {
            try driver.gemvH16T4(enc, n: geometry.gdnQKVChannels, k: H, x: sc.ln1, gs: sc.gs,
                                 wQ4: w.inProj, y16: sc.qkv16)
            mark("kda.fgb_proj")
            try driver.gemvH16T4(enc, n: fgbProj.n, k: H, x: sc.ln1, gs: sc.gs,
                                 wQ4: fgbProj, y16: fgb16)
            mark("kda.gates")
            try driver.gdnDecode.encodeKDAGates(
                enc, fgb: fgb16, fB: fB, gB: gB, aLog: w.aLog, dtBias: w.dtBias,
                betas: sc.betas, glog: sc.glog, zOut: sc.z16, rows: 1)
        } else {
            try driver.gemvH16GdnInProjT4(enc, n: geometry.gdnTotalN, k: H, x: sc.ln1, gs: sc.gs, wQ4: w.inProj,
                                   y16: sc.gdnY16, y32: sc.gdnY,
                                   aLog: w.aLog, dtBias: w.dtBias, betas: sc.betas, glog: sc.glog)
        }
        mark("gdn.conv")
        // C11 conv + headnorm fused for T=1:
        let gi = driver.gdnIndexMap[l]
        try driver.gdnConv.encodeT1WithHeadnorm(enc, x: sc.qkv16, w: w.conv, bias: nil,
                                                 stIn: context.session.ring[gi], y: sc.convY,
                                                 stOut: context.session.ring[gi])
        mark("gdn.recurrent")
        // C12 recurrent + gated norm
        try driver.gdnDecode.encode(enc, y: sc.convY, betas: sc.betas, glog: sc.glog,
                                     state: context.session.rec[gi], z: sc.z16, w: w.normW,
                                     o: sc.oOut, oScratch: sc.oScr,
                                     outGroupSum: nil)
        mark("gdn.out_proj")
        // out_proj (K = zC)
        try driver.gemvH16T4(enc, n: H, k: geometry.gdnZChannels, x: sc.oOut, gs: sc.gs48,
                      wQ4: w.out, y16: sc.o16)
    }

    override func encodeAttentionPrefill(_ context: QWLayerPrefillContext) throws {
        let driver = context.driver
        var enc = context.enc
        func mark(_ stage: String) {
            driver.profileStage(stage)
            if let a = driver.activeEnc { enc = a }
        }
        let psc = driver.psc
        let geometry = driver.geometry
        let l = context.layer
        let w = context.weights.gdn!
        let T = context.rows
        let pos = context.position
        let H = geometry.hidden
        let qkvC = geometry.gdnQKVChannels, zC = geometry.gdnZChannels
        let gdnVH = geometry.gdnVHeads
        let glogC = geometry.gdnGlogChannels
        let gi = driver.gdnIndexMap[l]   // M6 FIX: single source of truth
        if geometry.gdnVectorDecay, let fgbProj = w.fgbProj, let fgb16 = psc.fgb16,
           let fB = w.fB, let gB = w.gB {
            let activeLn1 = driver.prefixRows(psc.ln1, rows: T, cols: H, dt: .f16)
            mark("kda.in_proj")
            driver.gemmBatchedH16(enc, m: T, n: qkvC, k: H, x: activeLn1, wQ4: w.inProj,
                                  y16: driver.prefixRows(psc.qkv16, rows: T, cols: qkvC, dt: .f16))
            mark("kda.fgb_proj")
            driver.gemmBatchedH16(enc, m: T, n: fgbProj.n, k: H, x: activeLn1, wQ4: fgbProj,
                                  y16: driver.prefixRows(fgb16, rows: T, cols: fgbProj.n, dt: .f16))
            mark("kda.gates")
            try driver.gdnDecode.encodeKDAGates(
                enc,
                fgb: driver.prefixRows(fgb16, rows: T, cols: fgbProj.n, dt: .f16),
                fB: fB, gB: gB, aLog: w.aLog, dtBias: w.dtBias,
                betas: driver.prefixRows(psc.betas, rows: T, cols: gdnVH, dt: .f32),
                glog: driver.prefixRows(psc.glog, rows: T, cols: glogC, dt: .f32),
                zOut: driver.prefixRows(psc.z16, rows: T, cols: zC, dt: .f16),
                rows: T)
            mark("kda.conv")
            try driver.gdnConv.encodePrefillWithHeadnorm(
                enc,
                x: driver.prefixRows(psc.qkv16, rows: T, cols: qkvC, dt: .f16),
                w: w.conv, bias: nil,
                stIn: pos == 0 ? driver.gdnConv.zeroRing : context.session.ring[gi],
                y: driver.prefixRows(psc.convY, rows: T, cols: qkvC, dt: .f16),
                stOut: context.session.ring[gi],
                T: T)
        } else {
            driver.gemmBatchedH16GdnInProj(enc, m: T, n: geometry.gdnTotalN, k: H,
                                    x: driver.prefixRows(psc.ln1, rows: T, cols: H, dt: .f16),
                                    wQ4: w.inProj,
                                    qkv16: psc.qkv16, z16: psc.z16, y32: psc.gdnY)
            try driver.gdnConv.encodePrefillWithHeadnorm(enc,
                                                          x: driver.prefixRows(psc.qkv16, rows: T, cols: qkvC, dt: .f16),
                                                          w: w.conv, bias: nil,
                                                          stIn: pos == 0 ? driver.gdnConv.zeroRing : context.session.ring[gi],
                                                          y: driver.prefixRows(psc.convY, rows: T, cols: qkvC, dt: .f16),
                                                          stOut: context.session.ring[gi],
                                                          T: T,
                                                          gdnY32: psc.gdnY, aLog: w.aLog, dtBias: w.dtBias,
                                                          betas: driver.prefixRows(psc.betas, rows: T, cols: gdnVH, dt: .f32),
                                                          glog: driver.prefixRows(psc.glog, rows: T, cols: glogC, dt: .f32))
        }
        mark("gdn.recurrence")
        switch context.mode {
        case .chunked:
            if T <= 4096 {
                try driver.gdnPrefill.encodeRegRes(enc,
                                                   y: driver.prefixRows(psc.convY, rows: T, cols: qkvC, dt: .f16),
                                                   betas: driver.prefixRows(psc.betas, rows: T, cols: gdnVH, dt: .f32),
                                                   glog: driver.prefixRows(psc.glog, rows: T, cols: glogC, dt: .f32),
                                                   z: driver.prefixRows(psc.z16, rows: T, cols: zC, dt: .f16),
                                                   w: w.normW,
                                                   state: context.session.rec[gi],
                                                   o: driver.prefixRows(psc.oOut, rows: T, cols: zC, dt: .f16),
                                                   oScratch: driver.prefixRows(psc.oScr, rows: T, cols: zC, dt: .f32),
                                                   T: T)
            } else {
                try driver.gdnPrefill.encode(enc,
                                             y: driver.prefixRows(psc.convY, rows: T, cols: qkvC, dt: .f16),
                                             betas: driver.prefixRows(psc.betas, rows: T, cols: gdnVH, dt: .f32),
                                             glog: driver.prefixRows(psc.glog, rows: T, cols: glogC, dt: .f32),
                                             z: driver.prefixRows(psc.z16, rows: T, cols: zC, dt: .f16),
                                             w: w.normW,
                                             state: context.session.rec[gi],
                                             o: driver.prefixRows(psc.oOut, rows: T, cols: zC, dt: .f16),
                                             U: driver.prefixRows(psc.U, rows: T, cols: zC, dt: .f32),
                                             W: driver.prefixRows(psc.W, rows: T, cols: zC, dt: .f32),
                                             A: driver.prefixRows(psc.A, rows: T, cols: zC, dt: .f32),
                                             B: driver.prefixRows(psc.B, rows: T, cols: zC, dt: .f32),
                                             oScratch: driver.prefixRows(psc.oScr, rows: T, cols: zC, dt: .f32),
                                             T: T, L: min(64, T),
                                             solveBackend: driver.gdnSolveBackend)
            }
        case .naive:
            for t in 0..<T {
                let yRow = driver.stridedRow(psc.convY, row: t, cols: qkvC, dt: .f16)
                let bRow = driver.stridedRow(psc.betas, row: t, cols: gdnVH, dt: .f32)
                let gRow = driver.stridedRow(psc.glog, row: t, cols: glogC, dt: .f32)
                let zRow = driver.stridedRow(psc.z16, row: t, cols: zC, dt: .f16)
                let oRow = driver.stridedRow(psc.oOut, row: t, cols: zC, dt: .f16)
                let sRow = driver.stridedRow(psc.oScr, row: t, cols: zC, dt: .f32)
                try driver.gdnDecode.encode(enc, y: yRow, betas: bRow, glog: gRow,
                                            state: context.session.rec[gi], z: zRow, w: w.normW,
                                            o: oRow, oScratch: sRow)
            }
        case .regres:
            // EXP-9 sub-item 1: 192-TG zero-barrier register-resident
            // recurrence + qwg_gdn_prefill_norm.
            try driver.gdnPrefill.encodeRegRes(enc,
                                               y: driver.prefixRows(psc.convY, rows: T, cols: qkvC, dt: .f16),
                                               betas: driver.prefixRows(psc.betas, rows: T, cols: gdnVH, dt: .f32),
                                               glog: driver.prefixRows(psc.glog, rows: T, cols: glogC, dt: .f32),
                                               z: driver.prefixRows(psc.z16, rows: T, cols: zC, dt: .f16),
                                               w: w.normW,
                                               state: context.session.rec[gi],
                                               o: driver.prefixRows(psc.oOut, rows: T, cols: zC, dt: .f16),
                                               oScratch: driver.prefixRows(psc.oScr, rows: T, cols: zC, dt: .f32),
                                               T: T)
        }
        mark("gdn.out_proj")
        driver.gemmBatchedH16(enc, m: T, n: H, k: zC,
                       x: driver.prefixRows(psc.oOut, rows: T, cols: zC, dt: .f16),
                       wQ4: w.out,
                       y16: driver.prefixRows(psc.o16, rows: T, cols: H, dt: .f16))
    }
}

/// The full-attention layer: paged KV attention with rope.
final class FullAttentionLayerKernel: DenseLayerKernel, QWLayerKernel, @unchecked Sendable {
    static let key = QWEnginePlan.LayerKernelKey(attention: .fullAttention, mlp: .dense, ple: false)

    static func make() -> any QWLayerKernel { FullAttentionLayerKernel() }

    override func encodeAttention(_ context: QWLayerDecodeContext) throws {
        let driver = context.driver
        var enc = context.enc
        func mark(_ stage: String) { driver.profileStage(stage); if let a = driver.activeEnc { enc = a } }
        let sc = driver.sc
        let geometry = driver.geometry
        let l = context.layer
        let pos = context.position
        let w = context.weights.attn!
        let H = geometry.hidden
        let fullAttnIdx = driver.attnIndexMap[l]
        let views = context.session.attnKVViews[fullAttnIdx]
        if let mla = driver.mlaOperator, let mlaW = w.mla {
            let latRow = driver.prefixRows(mla.mlaLatentScratch, rows: 1, cols: mla.latentStride, dt: .f16)
            let qResRow = driver.prefixRows(mla.qResidScratch, rows: 1, cols: geometry.qLoraRank, dt: .f16)
            let qFullRow = driver.prefixRows(mla.mlaQFullScratch, rows: 1, cols: mla.qFullStride, dt: .f16)
            try driver.gemvH16T4(enc, n: w.qkvProj.n, k: H, x: sc.ln1, gs: sc.gs, wQ4: w.qkvProj, y16: latRow)
            mark("mla.latent_prep")
            mla.encodeLatentPrepAndPool(
                enc, ln1: sc.ln1,
                mlaW: mlaW, kv: views,
                fullAttnLayer: fullAttnIdx, pos0: pos, rows: 1)
            mark("mla.q_b_proj")
            try driver.gemvH16T4(enc, n: mlaW.qBProj.n, k: geometry.qLoraRank, x: qResRow, gs: sc.gs, wQ4: mlaW.qBProj, y16: qFullRow)
            mark("mla.attend")
            mla.encodeScoreSelectAbsorbAttendAndProjUV(
                enc, mlaW: mlaW, kv: views,
                attOut: sc.attOut,
                fullAttnLayer: fullAttnIdx, pos0: pos, rows: 1,
                stage: QWEngineDriver.decodeProfileEnabled
                    ? { name in driver.profileStage(name); return driver.activeEnc } : nil)
            mark("mla.o_proj")
            try driver.gemvH16T4(enc, n: H, k: geometry.qHeads * geometry.vHeadDim,
                          x: sc.attOut, gs: sc.gs48, wQ4: w.o, y16: sc.o16)
            return
        }
        try driver.gemvH16T4(enc, n: geometry.attnTotalN, k: H, x: sc.ln1, gs: sc.gs, wQ4: w.qkvProj, y16: sc.attnY16)
        // C07 fused per-head norm + partial rope (position [pos]).
        try driver.ropeApply.encodeQK(enc, hal: driver.hal, q: sc.q16, k: sc.k16,
                                      wq: w.wq, wk: w.wkTiled,
                                      qOut: sc.qR, kOut: sc.kR,
                                      positionStart: pos, count: 1)
        // C09 split-K decode: attends the new row from knew/vnew AND
        // materialises it into the page slot in the same launch.
        let attnDecode = driver.attnDecode!
        if let lsa = driver.lsaOperator {
            lsa.encodeIndexAndCluster(enc, qgate: sc.qR, kIn: sc.kR,
                                      fullAttnLayer: fullAttnIdx, pos0: pos, rows: 1)
            if pos >= driver.lsaConfig.denseBypassThreshold {
                lsa.encodeSelectBlocks(enc, fullAttnLayer: fullAttnIdx, pos0: pos, rows: 1)
                let activeKeys = min(pos + 1, driver.lsaConfig.alignedBudget + 640)
                try attnDecode.encodeSparse(
                    enc, qgate: sc.qR, kv: views,
                    knew: sc.kR, vnew: sc.v16,
                    partials: sc.partials, out: sc.attOut,
                    blockList: lsa.blockListScratch,
                    blockCount: lsa.blockCountScratch,
                    C: pos, splits: QWAttnDecode.splits(forKeys: activeKeys),
                    listStride: lsa.listStride)
            } else {
                try attnDecode.encode(enc, qgate: sc.qR, kv: views,
                                      knew: sc.kR, vnew: sc.v16,
                                      partials: sc.partials, out: sc.attOut,
                                      C: pos, splits: QWAttnDecode.splits(forKeys: pos + 1))
            }
        } else {
            try attnDecode.encode(enc, qgate: sc.qR, kv: views,
                                  knew: sc.kR, vnew: sc.v16,
                                  partials: sc.partials, out: sc.attOut,
                                  C: pos, splits: QWAttnDecode.splits(forKeys: pos + 1))
        }
        // o_proj (K = qHeads·headDim = 6144 — the C08 law, NOT 5120)
        try driver.gemvH16T4(enc, n: H, k: geometry.qHeads * geometry.headDim,
                      x: sc.attOut, gs: sc.gs48, wQ4: w.o, y16: sc.o16)
    }

    override func encodeAttentionPrefill(_ context: QWLayerPrefillContext) throws {
        let driver = context.driver
        let enc = context.enc
        let psc = driver.psc
        let geometry = driver.geometry
        let l = context.layer
        let w = context.weights.attn!
        let T = context.rows
        let pos = context.position
        let H = geometry.hidden
        let qOut = geometry.qProjOut, qC = geometry.qChannels, kvC = geometry.kvChannels
        let fullAttnIdx = driver.attnIndexMap[l]
        let views = context.session.attnKVViews[fullAttnIdx]
        if let mla = driver.mlaOperator, let mlaW = w.mla {
            let vCols = w.o.k
            let activeLn1 = driver.prefixRows(psc.ln1, rows: T, cols: H, dt: .f16)
            let latRows = driver.prefixRows(mla.mlaLatentScratch, rows: T, cols: mla.latentStride, dt: .f16)
            let qResRows = driver.prefixRows(mla.qResidScratch, rows: T, cols: geometry.qLoraRank, dt: .f16)
            let qFullRows = driver.prefixRows(mla.mlaQFullScratch, rows: T, cols: mla.qFullStride, dt: .f16)
            let attOutRows = driver.prefixRows(psc.attOut, rows: T, cols: vCols, dt: .f16)
            var enc = enc
            func mark(_ stage: String) -> (any MTLComputeCommandEncoder)? {
                driver.profileStage(stage)
                if let a = driver.activeEnc { enc = a }
                return driver.activeEnc
            }
            _ = mark("mla.qkv_a_proj")
            driver.gemmBatchedH16(enc, m: T, n: w.qkvProj.n, k: H, x: activeLn1, wQ4: w.qkvProj, y16: latRows)
            _ = mark("mla.latent_prep")
            mla.encodeLatentPrepAndPool(
                enc, ln1: activeLn1,
                mlaW: mlaW, kv: views,
                fullAttnLayer: fullAttnIdx, pos0: pos, rows: T)
            _ = mark("mla.q_b_proj")
            driver.gemmBatchedH16(enc, m: T, n: mlaW.qBProj.n, k: geometry.qLoraRank, x: qResRows, wQ4: mlaW.qBProj, y16: qFullRows)
            mla.encodeScoreSelectAbsorbAttendAndProjUV(
                enc, mlaW: mlaW, kv: views,
                attOut: attOutRows,
                fullAttnLayer: fullAttnIdx, pos0: pos, rows: T,
                stage: QWEngineDriver.prefillProfileEnabled ? mark : nil)
            _ = mark("mla.o_proj")
            enc = driver.activeEnc ?? enc
            driver.gemmBatchedH16(enc, m: T, n: H, k: vCols,
                           x: attOutRows,
                           wQ4: w.o,
                           y16: driver.prefixRows(psc.o16, rows: T, cols: H, dt: .f16))
            return
        }
        driver.gemmBatchedH16(enc, m: T, n: geometry.attnTotalN, k: H,
                       x: driver.prefixRows(psc.ln1, rows: T, cols: H, dt: .f16),
                       wQ4: w.qkvProj,
                       y16: driver.prefixRows(psc.attnY16, rows: T, cols: geometry.attnTotalN, dt: .f16))
        try driver.ropeApply.encodeExtractQK(enc, hal: driver.hal,
                                              attnY: driver.prefixRows(psc.attnY16, rows: T, cols: geometry.attnTotalN, dt: .f16),
                                              qOut: driver.prefixRows(psc.attQR16, rows: T, cols: qOut, dt: .f16),
                                              kOut: driver.prefixRows(psc.attKR16, rows: T, cols: kvC, dt: .f16),
                                              vOut: driver.prefixRows(psc.attV16, rows: T, cols: kvC, dt: .f16),
                                              wq: w.wq, wk: w.wkTiled,
                                              positionStart: pos, count: T)
        if let lsa = driver.lsaOperator {
            lsa.encodeIndexAndCluster(
                enc,
                qgate: driver.prefixRows(psc.attQR16, rows: T, cols: qOut, dt: .f16),
                kIn: driver.prefixRows(psc.attKR16, rows: T, cols: kvC, dt: .f16),
                fullAttnLayer: fullAttnIdx, pos0: pos, rows: T)
        }
        // Pad to the SELECTED prefill backend's tile (32 rows, 64 for
        // the quad kernel) — a hardcoded 32 crashed quad at S=32. Only the
        // 27B's NAX prefill reads whole tiles; the standard path does not.
        let padTile = driver.attnPrefill.tileRows
        let sp = (T + padTile - 1) / padTile * padTile
        if sp > T && QWEngineDriver.zeroPadServes(geometry: geometry) {
            try driver.encodeZeroPadAttn(enc, q: psc.attQR16, k: psc.attKR16, v: psc.attV16, t: T, sp: sp)
        }
        if let lsa = driver.lsaOperator, pos + T > driver.lsaConfig.denseBypassThreshold {
            driver.encodePagedKVWrite(enc, kv: views,
                               k: driver.prefixRows(psc.attKR16, rows: T, cols: kvC, dt: .f16),
                               v: driver.prefixRows(psc.attV16, rows: T, cols: kvC, dt: .f16),
                               startPos: pos, t: T)
            let useNAX = driver.geometry.supportsNAXPagedAttnPrefill
            lsa.encodeSelectBlocks(enc, fullAttnLayer: fullAttnIdx, pos0: pos, rows: T, tiled32: useNAX)
            if !useNAX {
                try driver.pagedAttnPrefillGeneral.encodeSparse(
                    enc,
                    qgate: driver.prefixRows(psc.attQR16, rows: sp, cols: qOut, dt: .f16),
                    kv: views,
                    out: driver.prefixRows(psc.attOut, rows: T, cols: qC, dt: .f16),
                    blockList: lsa.blockListScratch,
                    blockCount: lsa.blockCountScratch,
                    pos: pos, T: T, listStride: lsa.listStride)
            } else {
                try driver.pagedAttnPrefill.encodeSparse(
                    enc,
                    qgate: driver.prefixRows(psc.attQR16, rows: sp, cols: qOut, dt: .f16),
                    kv: views,
                    out: driver.prefixRows(psc.attOut, rows: T, cols: qC, dt: .f16),
                    blockList: lsa.blockListScratch,
                    blockCount: lsa.blockCountScratch,
                    pos: pos, T: T, listStride: lsa.listStride)
            }
        } else if !driver.geometry.supportsNAXPagedAttnPrefill {
            // General scalar prefill fallback for unsupported attention shapes.
            driver.encodePagedKVWrite(enc, kv: views,
                               k: driver.prefixRows(psc.attKR16, rows: T, cols: kvC, dt: .f16),
                               v: driver.prefixRows(psc.attV16, rows: T, cols: kvC, dt: .f16),
                               startPos: pos, t: T)
            try driver.pagedAttnPrefillGeneral.encode(enc,
                                               qgate: driver.prefixRows(psc.attQR16, rows: sp, cols: qOut, dt: .f16),
                                               kv: views,
                                               out: driver.prefixRows(psc.attOut, rows: T, cols: qC, dt: .f16),
                                               pos: pos, T: T)
        } else if pos == 0 && T < 128 && driver.geometry.is27BShape {
            try driver.attnPrefill.encode(enc,
                                          qgate: driver.prefixRows(psc.attQR16, rows: sp, cols: qOut, dt: .f16),
                                          k: driver.prefixRows(psc.attKR16, rows: sp, cols: kvC, dt: .f16),
                                          v: driver.prefixRows(psc.attV16, rows: sp, cols: kvC, dt: .f16),
                                          out: driver.prefixRows(psc.attOut, rows: T, cols: qC, dt: .f16),
                                          S: T)
            driver.encodePagedKVWrite(enc, kv: views,
                               k: driver.prefixRows(psc.attKR16, rows: T, cols: kvC, dt: .f16),
                               v: driver.prefixRows(psc.attV16, rows: T, cols: kvC, dt: .f16),
                               startPos: pos, t: T)
        } else {
            // NAX paged prefill: write KV pool, then attend over all cached tokens.
            driver.encodePagedKVWrite(enc, kv: views,
                               k: driver.prefixRows(psc.attKR16, rows: T, cols: kvC, dt: .f16),
                               v: driver.prefixRows(psc.attV16, rows: T, cols: kvC, dt: .f16),
                               startPos: pos, t: T)
            try driver.pagedAttnPrefill.encode(enc,
                                               qgate: driver.prefixRows(psc.attQR16, rows: sp, cols: qOut, dt: .f16),
                                               kv: views,
                                               out: driver.prefixRows(psc.attOut, rows: T, cols: qC, dt: .f16),
                                               pos: pos, T: T)
        }
        driver.gemmBatchedH16(enc, m: T, n: H, k: qC,
                       x: driver.prefixRows(psc.attOut, rows: T, cols: qC, dt: .f16),
                       wQ4: w.o,
                       y16: driver.prefixRows(psc.o16, rows: T, cols: H, dt: .f16))
    }
}
