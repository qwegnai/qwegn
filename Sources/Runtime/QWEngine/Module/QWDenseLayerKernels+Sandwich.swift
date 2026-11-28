// QWDenseLayerKernels+Sandwich.swift — Fused 2-dispatch/layer pre_post GatedNorm
// sandwich execution across decode, prefill, and B=2..8 batched decode.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWNorm

extension QWEngineDriver {
    var hasSandwichGatedNorm: Bool {
        geometry.useSandwichNorm && finalGatedNorm != nil
    }
}

extension DenseLayerKernel {
    func encodeDecodeSandwichGated(
        _ context: QWLayerDecodeContext,
        gatedNorms gn: QWEngineDriver.GatedLayerNorms,
        nextGatedNorm: QWGatedNormWeights?
    ) throws {
        let driver = context.driver
        var enc = driver.activeEnc ?? context.enc
        let sc = driver.sc
        let l = context.layer
        let lw = context.weights
        if l == 0 {
            driver.profileStage("norm.sandwich")
            enc = driver.activeEnc ?? context.enc
            try QWNorm.encodeGatedRMSNorm(
                enc, hal: driver.hal, x: sc.x, weights: gn.input, out: sc.ln1)
        }
        try encodeAttention(context)
        driver.profileStage("norm.sandwich")
        enc = driver.activeEnc ?? context.enc
        let moeBridge = driver.canFusePreMoENorm(weights: lw, rows: 1) ? sc.moe.map {
            (xRot32: $0.xRot32, xPlain32: $0.xPlain32, skipRot32: false)
        } : nil
        try QWNorm.encodeGatedSandwichResidual(
            enc, hal: driver.hal,
            branchOut: sc.o16, residual: sc.x,
            postWeights: gn.postSelfAttn, nextPreWeights: gn.postAttn, out: sc.ln2,
            xRot32: moeBridge?.xRot32, xPlain32: moeBridge?.xPlain32,
            skipRot32: moeBridge?.skipRot32 ?? false)
        try encodeMLP(context)
        driver.profileStage("norm.sandwich")
        enc = driver.activeEnc ?? context.enc
        try QWNorm.encodeGatedSandwichResidual(
            enc, hal: driver.hal,
            branchOut: sc.dn16, residual: sc.x,
            postWeights: gn.postFFN, nextPreWeights: nextGatedNorm, out: sc.ln1)
    }

    func encodePrefillSandwichGated(
        _ context: QWLayerPrefillContext,
        gatedNorms gn: QWEngineDriver.GatedLayerNorms,
        nextGatedNorm: QWGatedNormWeights?
    ) throws {
        let driver = context.driver
        var enc = driver.activeEnc ?? context.enc
        let psc = driver.psc
        let H = driver.geometry.hidden
        let T = context.rows
        let l = context.layer
        let lw = context.weights
        let activeX = driver.prefixRows(psc.x, rows: T, cols: H, dt: .f16)
        let activeLn1 = driver.prefixRows(psc.ln1, rows: T, cols: H, dt: .f16)
        let activeLn2 = driver.prefixRows(psc.ln2, rows: T, cols: H, dt: .f16)
        let activeO16 = driver.prefixRows(psc.o16, rows: T, cols: H, dt: .f16)
        let activeDn16 = driver.prefixRows(psc.dn16, rows: T, cols: H, dt: .f16)
        if l == 0 {
            driver.profileStage("norm.sandwich")
            enc = driver.activeEnc ?? context.enc
            try QWNorm.encodeGatedRMSNorm(
                enc, hal: driver.hal, x: activeX, weights: gn.input, out: activeLn1)
        }
        try encodeAttentionPrefill(context)
        driver.profileStage("norm.sandwich")
        enc = driver.activeEnc ?? context.enc
        let moeBridge = driver.canFusePreMoENorm(weights: lw, rows: T) ? psc.moe.map {
            (xRot32: driver.prefixRows($0.xRot32, rows: T, cols: H, dt: .f32),
             xPlain32: driver.prefixRows($0.xPlain32, rows: T, cols: H, dt: .f32),
             skipRot32: driver.canSkipPreMoERot32(weights: lw, scratch: $0, rows: T))
        } : nil
        try QWNorm.encodeGatedSandwichResidual(
            enc, hal: driver.hal,
            branchOut: activeO16, residual: activeX,
            postWeights: gn.postSelfAttn, nextPreWeights: gn.postAttn, out: activeLn2,
            xRot32: moeBridge?.xRot32, xPlain32: moeBridge?.xPlain32,
            skipRot32: moeBridge?.skipRot32 ?? false)
        try encodeMLPPrefill(context)
        driver.profileStage("norm.sandwich")
        enc = driver.activeEnc ?? context.enc
        try QWNorm.encodeGatedSandwichResidual(
            enc, hal: driver.hal,
            branchOut: activeDn16, residual: activeX,
            postWeights: gn.postFFN, nextPreWeights: nextGatedNorm, out: activeLn1)
    }

    func encodeBatchDecodeSandwichGated(
        _ enc: any MTLComputeCommandEncoder,
        driver: QWEngineDriver,
        sessions: [QWEngineSession],
        layer l: Int,
        physicalLayer p: Int,
        weights lw: QWEngineDriver.LayerW,
        gatedNorms gn: QWEngineDriver.GatedLayerNorms,
        nextGatedNorm: QWGatedNormWeights?
    ) throws {
        var curEnc = driver.activeEnc ?? enc
        let psc = driver.psc
        let geometry = driver.geometry
        let H = geometry.hidden
        let B = sessions.count
        let activeX = driver.prefixRows(psc.x, rows: B, cols: H, dt: .f16)
        let activeLn1 = driver.prefixRows(psc.ln1, rows: B, cols: H, dt: .f16)
        let activeLn2 = driver.prefixRows(psc.ln2, rows: B, cols: H, dt: .f16)
        let activeO16 = driver.prefixRows(psc.o16, rows: B, cols: H, dt: .f16)
        if l == 0 {
            try QWNorm.encodeGatedRMSNorm(
                curEnc, hal: driver.hal, x: activeX, weights: gn.input, out: activeLn1)
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
        try QWNorm.encodeGatedSandwichResidual(
            curEnc, hal: driver.hal,
            branchOut: activeO16, residual: activeX,
            postWeights: gn.postSelfAttn, nextPreWeights: gn.postAttn, out: activeLn2,
            xRot32: moeBridge?.xRot32, xPlain32: moeBridge?.xPlain32,
            skipRot32: moeBridge?.skipRot32 ?? false)
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
            let fuseFwht = !lw.gu.view.isIncumbentContainer
                && geometry.intermediate > 0 && (geometry.intermediate % 32 == 0)
            try batchMLP.encodeGateUp(
                curEnc, input: activeMlpRows.shape.elementCount > 0 ? activeLn2 : activeLn2,
                weight: lw.gu.view, output: activeMlpRows, activeRows: B,
                fuseFwht32: fuseFwht, swigluClamp: geometry.routedSwiGLUClamp(layer: p))
            if !fuseFwht {
                driver.encodeFwht32Fast(curEnc, x: activeMlpRows)
                curEnc.memoryBarrier(scope: .buffers)
            }
            try batchMLP.encodeDown(curEnc, input: activeMlpRows, weight: lw.down, activeRows: B)
            dnSource = batchMLP.downRows
        }
        curEnc.memoryBarrier(scope: .buffers)
        try QWNorm.encodeGatedSandwichResidual(
            curEnc, hal: driver.hal,
            branchOut: driver.prefixRows(dnSource, rows: B, cols: H, dt: .f16),
            residual: activeX,
            postWeights: gn.postFFN, nextPreWeights: nextGatedNorm, out: activeLn1)
        curEnc.memoryBarrier(scope: .buffers)
    }
}
