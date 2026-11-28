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
import QWMTP
import QWQuant

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
        if context.weights.mhc == nil {
            driver.profileStage("mlp.dense")
        }
        let enc = context.enc
        let sc = driver.sc
        let geometry = driver.geometry
        let view = context.weights.gu.view
        let fuseFwht = !view.isIncumbentContainer && geometry.intermediate > 0 && (geometry.intermediate % 32 == 0)
        // MLP (M=1 route: silu_pair_dx_h16 + fwht32)
        let plan = geometry.executionPlan
        let p = plan.indices.contains(context.layer) ? plan[context.layer] : context.layer
        let clamp = geometry.routedSwiGLUClamp(layer: p)
        try driver.mlpSiluPairH16(enc, x: sc.ln2, gs: sc.gs, wQ4: view,
                                  h16: sc.gu16, fuseFwht32: fuseFwht, swigluClamp: clamp)
        if !fuseFwht {
            driver.encodeFwht32Fast(enc, x: sc.gu16)
        }
        try driver.gemvH16T4(enc, n: geometry.hidden, k: geometry.intermediate, x: sc.gu16, gs: sc.gsI,
                      wQ4: context.weights.down, y16: sc.dn16)
    }

    /// The layer's MLP for one prefill chunk: the dense SwiGLU.
    func encodeMLPPrefill(_ context: QWLayerPrefillContext) throws {
        let driver = context.driver
        if context.weights.mhc == nil {
            driver.profileStage("mlp.dense")
        }
        let enc = context.enc
        let psc = driver.psc
        let geometry = driver.geometry
        let lw = context.weights
        let H = geometry.hidden
        let I = geometry.intermediate
        let T = context.rows
        let plan = geometry.executionPlan
        let p = plan.indices.contains(context.layer) ? plan[context.layer] : context.layer
        let clamp = geometry.routedSwiGLUClamp(layer: p)
        if (I & 31) == 0 && driver.hal.caps.supportsSimdgroupMMA {
            try driver.gemmBatchedH16SwiGLU(
                enc, m: T, i: I, k: H,
                x: driver.prefixRows(psc.ln2, rows: T, cols: H, dt: .f16),
                wQ4: lw.gu.view,
                out16: driver.prefixRows(psc.gu16, rows: T, cols: I, dt: .f16),
                swigluClamp: clamp)
        } else {
            try driver.gemmBatchedH16(enc, m: T, n: 2 * I, k: H,
                               x: driver.prefixRows(psc.ln2, rows: T, cols: H, dt: .f16),
                               wQ4: lw.gu.view,
                               y16: driver.prefixRows(psc.guRaw16, rows: T, cols: 2 * I, dt: .f16))
            try driver.encodeSiluMulF16(enc,
                             gu: driver.prefixRows(psc.guRaw16, rows: T, cols: 2 * I, dt: .f16),
                             out: driver.prefixRows(psc.gu16, rows: T, cols: I, dt: .f16),
                             m: T, i: I,
                             swigluClamp: clamp)
        }
        try driver.gemmBatchedH16(enc, m: T, n: H, k: I,
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
            let moeBridge = driver.canFusePreMoENorm(weights: lw, rows: 1) ? sc.moe.map {
                (xRot32: $0.xRot32, xPlain32: $0.xPlain32, skipRot32: false)
            } : nil
            try driver.encodeMHCRouterNorm(enc, stream: mhcStream, w: mhc.ffnW, b: mhc.ffnB,
                                           gamma: lw.ln2, controls: mhcControls, out: sc.ln2,
                                           rows: 1, mergeBranchOut: sc.o16, moeBridge: moeBridge)
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
        } else if let gn = lw.gatedNorms, geometry.useSandwichNorm {
            let nextGated = nextP.flatMap { driver.layersW[$0].gatedNorms?.input } ?? driver.finalGatedNorm
            try encodeDecodeSandwichGated(context, gatedNorms: gn, nextGatedNorm: nextGated)
            enc = driver.activeEnc ?? context.enc
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
            let moeBridge = driver.canFusePreMoENorm(weights: lw, rows: 1) ? sc.moe.map {
                (xRot32: $0.xRot32, xPlain32: $0.xPlain32, skipRot32: false)
            } : nil
            try driver.encodeRMSNormResidualFast(enc, x: sc.o16, gamma: lw.ln2, out: sc.ln2, residual: sc.x, moeBridge: moeBridge)
            try encodeMLP(context)
            enc = driver.activeEnc ?? context.enc
            // Rung 2: x := half(x + mlp) IN PLACE; the emitted norm uses the
            // NEXT layer's ln1 gamma (final norm at l = 63) — the oracle's
            // `x = x + mlp(...)` followed by the next layer's
            try driver.encodeRMSNormResidualFast(enc, x: sc.dn16, gamma: nextGamma, out: sc.ln1, residual: sc.x)
        }
        if let dflashDrafter = driver.dflashDrafter {
            let captureIdx = dflashDrafter.usesVirtualLayerCapture ? l : p
            guard captureIdx < dflashDrafter.targetLayerSlotMap.count else { return }
            let slot = dflashDrafter.targetLayerSlotMap[captureIdx]
            if slot >= 0 {
                if let _ = lw.mhc, let mhcCfg = geometry.mhcConfig,
                   let mhcStream = sc.mhcStream, let mhcControls = sc.mhcControls {
                    let isLeave = (p == mhcCfg.end - 1)
                    let nextIsMHC = nextP.map { $0 >= mhcCfg.start && $0 < mhcCfg.end } ?? false
                    let deferPostFFN = !isLeave && nextIsMHC && (nextP != mhcCfg.start)
                    dflashDrafter.encodeCaptureHCTargetHidden(
                        enc, base: mhcStream, srcRowOffset: 0, dstRowOffset: 0, rows: 1, slot: slot,
                        hc: mhcCfg.streams,
                        deferredInject: deferPostFFN ? (src: sc.dn16, inj: mhcControls, isMHC: true) : nil)
                } else {
                    dflashDrafter.encodeCaptureTargetHidden(
                        enc, source: sc.x, srcRowOffset: 0, dstRowOffset: 0, rows: 1, slot: slot)
                }
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
            let moeBridge = driver.canFusePreMoENorm(weights: lw, rows: T) ? psc.moe.map {
                (xRot32: driver.prefixRows($0.xRot32, rows: T, cols: H, dt: .f32),
                 xPlain32: driver.prefixRows($0.xPlain32, rows: T, cols: H, dt: .f32),
                 skipRot32: driver.canSkipPreMoERot32(weights: lw, scratch: $0, rows: T))
            } : nil
            try driver.encodeMHCRouterNorm(enc, stream: mhcStream, w: mhc.ffnW, b: mhc.ffnB,
                                           gamma: lw.ln2, controls: mhcControls, out: activeLn2,
                                           rows: T, mergeBranchOut: activeO16, moeBridge: moeBridge)
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
        } else if let gn = lw.gatedNorms, geometry.useSandwichNorm {
            let nextGated = nextP.flatMap { driver.layersW[$0].gatedNorms?.input } ?? driver.finalGatedNorm
            try encodePrefillSandwichGated(context, gatedNorms: gn, nextGatedNorm: nextGated)
            enc = driver.activeEnc ?? context.enc
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
            let moeBridge = driver.canFusePreMoENorm(weights: lw, rows: T) ? psc.moe.map {
                (xRot32: driver.prefixRows($0.xRot32, rows: T, cols: H, dt: .f32),
                 xPlain32: driver.prefixRows($0.xPlain32, rows: T, cols: H, dt: .f32),
                 skipRot32: driver.canSkipPreMoERot32(weights: lw, scratch: $0, rows: T))
            } : nil
            try driver.encodeRMSNormResidualFast(enc,
                                      x: driver.prefixRows(psc.o16, rows: T, cols: H, dt: .f16),
                                      gamma: lw.ln2,
                                      out: driver.prefixRows(psc.ln2, rows: T, cols: H, dt: .f16),
                                      residual: driver.prefixRows(psc.x, rows: T, cols: H, dt: .f16),
                                      rows: T, rowParallel: false, moeBridge: moeBridge)
            try encodeMLPPrefill(context)
            enc = driver.activeEnc ?? context.enc
            try driver.encodeRMSNormResidualFast(enc,
                                      x: driver.prefixRows(psc.dn16, rows: T, cols: H, dt: .f16),
                                      gamma: nextGamma,
                                      out: driver.prefixRows(psc.ln1, rows: T, cols: H, dt: .f16),
                                      residual: driver.prefixRows(psc.x, rows: T, cols: H, dt: .f16),
                                      rows: T, rowParallel: false)
        }
        if let dflashDrafter = driver.dflashDrafter, let cap = context.dflashCapture {
            let captureIdx = dflashDrafter.usesVirtualLayerCapture ? l : p
            guard captureIdx < dflashDrafter.targetLayerSlotMap.count else { return }
            let slot = dflashDrafter.targetLayerSlotMap[captureIdx]
            if slot >= 0 {
                if let _ = lw.mhc, let mhcCfg = geometry.mhcConfig,
                   let mhcStream = psc.mhcStream, let mhcControls = psc.mhcControls {
                    let isLeave = (p == mhcCfg.end - 1)
                    let nextIsMHC = nextP.map { $0 >= mhcCfg.start && $0 < mhcCfg.end } ?? false
                    let deferPostFFN = !isLeave && nextIsMHC && (nextP != mhcCfg.start)
                    let activeDn16 = driver.prefixRows(psc.dn16, rows: T, cols: H, dt: .f16)
                    dflashDrafter.encodeCaptureHCTargetHidden(
                        enc, base: mhcStream, srcRowOffset: cap.localStart, dstRowOffset: 0,
                        rows: cap.captureRows, slot: slot, hc: mhcCfg.streams,
                        deferredInject: deferPostFFN ? (src: activeDn16, inj: mhcControls, isMHC: true) : nil)
                } else {
                    dflashDrafter.encodeCaptureTargetHidden(
                        enc, source: psc.x, srcRowOffset: cap.localStart, dstRowOffset: 0,
                        rows: cap.captureRows, slot: slot)
                }
            }
        }
    }

    /// Batched decode trunk pass for B=2...4 independent sessions (including mHC and short-context LSA).
    func encodeBatchDecode(
        _ enc: any MTLComputeCommandEncoder,
        driver: QWEngineDriver,
        sessions: [QWEngineSession],
        layer l: Int,
        weights lw: QWEngineDriver.LayerW
    ) throws {
        var curEnc = driver.activeEnc ?? enc
        let psc = driver.psc
        let geometry = driver.geometry
        let H = geometry.hidden
        let B = sessions.count
        let plan = geometry.executionPlan
        let p = plan.indices.contains(l) ? plan[l] : l
        let nextP = (l + 1 < plan.count) ? plan[l + 1] : nil
        if let gn = lw.gatedNorms, geometry.useSandwichNorm, lw.mhc == nil {
            let nextGated = nextP.flatMap { driver.layersW[$0].gatedNorms?.input } ?? driver.finalGatedNorm
            try encodeBatchDecodeSandwichGated(
                curEnc, driver: driver, sessions: sessions,
                layer: l, physicalLayer: p, weights: lw,
                gatedNorms: gn, nextGatedNorm: nextGated)
            return
        }
        let nextGamma = nextP.map { driver.layersW[$0].ln1 } ?? driver.finalNorm

        let activeX = driver.prefixRows(psc.x, rows: B, cols: H, dt: .f16)
        let activeLn1 = driver.prefixRows(psc.ln1, rows: B, cols: H, dt: .f16)
        let activeLn2 = driver.prefixRows(psc.ln2, rows: B, cols: H, dt: .f16)
        let activeO16 = driver.prefixRows(psc.o16, rows: B, cols: H, dt: .f16)

        if let mhc = lw.mhc, let mhcCfg = geometry.mhcConfig,
           let mhcStream = psc.mhcStream, let mhcControls = psc.mhcControls {
            let prevP = l > 0 ? (plan.indices.contains(l - 1) ? plan[l - 1] : (l - 1)) : -1
            let hasDeferredPrevFFN = (p != mhcCfg.start) && (prevP >= mhcCfg.start && prevP < mhcCfg.end - 1)
            let prevDnSource = (prevP >= 0 && driver.layersW[prevP].moe != nil)
                ? psc.dn16
                : try driver.sharedBatchMLP().downRows
            if p == mhcCfg.start {
                try driver.encodeMHCEnter(curEnc, x: activeX, stream: mhcStream, rows: B)
                curEnc.memoryBarrier(scope: .buffers)
            }
            try driver.encodeMHCRouterNorm(
                curEnc, stream: mhcStream, w: mhc.attnW, b: mhc.attnB,
                gamma: lw.ln1, controls: mhcControls, out: activeLn1,
                rows: B, mergeBranchOut: hasDeferredPrevFFN ? driver.prefixRows(prevDnSource, rows: B, cols: H, dt: .f16) : nil)
            curEnc.memoryBarrier(scope: .buffers)
        } else if l == 0 {
            try driver.encodeRMSNormFast(curEnc, x: activeX, gamma: lw.ln1, out: activeLn1, rows: B)
            curEnc.memoryBarrier(scope: .buffers)
        }
        if lw.kind == .gdn || lw.kind == .gdnMoe {
            try GDNLayerKernel.encodeAttentionBatch(
                driver: driver, enc: curEnc, sessions: sessions, layer: l, weights: lw)
        } else {
            try FullAttentionLayerKernel.encodeAttentionBatch(
                driver: driver, enc: curEnc, sessions: sessions, layer: l, weights: lw)
        }
        curEnc = driver.activeEnc ?? curEnc
        let moeBridge = driver.canFusePreMoENorm(weights: lw, rows: B) ? psc.moe.map {
            (xRot32: driver.prefixRows($0.xRot32, rows: B, cols: H, dt: .f32),
             xPlain32: driver.prefixRows($0.xPlain32, rows: B, cols: H, dt: .f32),
             skipRot32: false)
        } : nil
        if let mhc = lw.mhc, geometry.mhcConfig != nil,
           let mhcStream = psc.mhcStream, let mhcControls = psc.mhcControls {
            try driver.encodeMHCRouterNorm(
                curEnc, stream: mhcStream, w: mhc.ffnW, b: mhc.ffnB,
                gamma: lw.ln2, controls: mhcControls, out: activeLn2,
                rows: B, mergeBranchOut: activeO16, moeBridge: moeBridge)
        } else {
            try driver.encodeRMSNormResidualFast(
                curEnc, x: activeO16, gamma: lw.ln2, out: activeLn2, residual: activeX,
                rows: B, rowParallel: false, moeBridge: moeBridge)
        }
        curEnc.memoryBarrier(scope: .buffers)
        let dnSource: QWTensorValue
        if lw.moe != nil {
            let mlpContext = QWLayerPrefillContext(
                driver: driver, enc: curEnc, session: sessions[0],
                position: sessions[0].position, layer: l, weights: lw, rows: B, mode: .chunked)
            try encodeMLPPrefill(mlpContext)
            curEnc = driver.activeEnc ?? curEnc
            dnSource = psc.dn16
        } else {
            let batchMLP = try driver.sharedBatchMLP()
            curEnc.useResource(batchMLP.downRows.region.buffer, usage: [.read, .write])
            curEnc.useResource(batchMLP.sums.region.buffer, usage: [.read, .write])
            let activeMlpRows = driver.prefixRows(psc.gu16, rows: B, cols: geometry.intermediate, dt: .f16)
            let fuseFwht = !lw.gu.view.isIncumbentContainer && geometry.intermediate > 0 && (geometry.intermediate % 32 == 0)
            try batchMLP.encodeGateUp(
                curEnc,
                input: activeLn2,
                weight: lw.gu.view,
                output: activeMlpRows,
                activeRows: B,
                fuseFwht32: fuseFwht,
                swigluClamp: geometry.routedSwiGLUClamp(layer: p))
            if !fuseFwht {
                driver.encodeFwht32Fast(curEnc, x: activeMlpRows)
                curEnc.memoryBarrier(scope: .buffers)
            }
            try batchMLP.encodeDown(curEnc, input: activeMlpRows, weight: lw.down, activeRows: B)
            dnSource = batchMLP.downRows
        }
        curEnc.memoryBarrier(scope: .buffers)
        if let _ = lw.mhc, let mhcCfg = geometry.mhcConfig,
           let mhcStream = psc.mhcStream, let mhcControls = psc.mhcControls {
            let isLeave = (p == mhcCfg.end - 1)
            let nextIsMHC = nextP.map { $0 >= mhcCfg.start && $0 < mhcCfg.end } ?? false
            let deferPostFFN = !isLeave && nextIsMHC && (nextP != mhcCfg.start)
            if !deferPostFFN {
                let activeDn = driver.prefixRows(dnSource, rows: B, cols: H, dt: .f16)
                if isLeave, mhcCfg.hasExitHead, let exitH = driver.mhcExitHead {
                    try driver.encodeMHCMerge(
                        curEnc, stream: mhcStream, branchOut: activeDn, controls: mhcControls,
                        xOut: activeX, nextGamma: nextGamma, nextNormOut: activeLn1,
                        rows: B, isLeave: false, sinkhornIters: mhcCfg.iterations, hasNextNorm: false)
                    curEnc.memoryBarrier(scope: .buffers)
                    try driver.encodeMHCExitHead(
                        curEnc, stream: mhcStream, headW: exitH.headW, headB: exitH.headB,
                        normGamma: driver.finalNorm, outNorm: activeLn1, rows: B, applyNorm: true)
                } else {
                    try driver.encodeMHCMerge(
                        curEnc, stream: mhcStream, branchOut: activeDn, controls: mhcControls,
                        xOut: activeX, nextGamma: nextGamma, nextNormOut: activeLn1,
                        rows: B, isLeave: isLeave, sinkhornIters: mhcCfg.iterations, hasNextNorm: isLeave)
                }
                curEnc.memoryBarrier(scope: .buffers)
            }
        } else {
            try driver.encodeRMSNormResidualFast(
                curEnc,
                x: driver.prefixRows(dnSource, rows: B, cols: H, dt: .f16),
                gamma: nextGamma,
                out: activeLn1,
                residual: activeX,
                rows: B)
            curEnc.memoryBarrier(scope: .buffers)
        }
    }
}

