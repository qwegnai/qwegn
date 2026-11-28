// QWHybridLayerKernels+BatchMLA.swift — B=2..8 multi-session batched attention
// decode for both MLA/DSA and standard GQA layers.
import Foundation
import Metal
import QWAlloc
import QWAttn
import QWCore
import QWHAL
import QWKV
import QWNorm

extension FullAttentionLayerKernel {
    static func encodeAttentionBatch(
        driver: QWEngineDriver,
        enc: any MTLComputeCommandEncoder,
        sessions: [QWEngineSession],
        layer l: Int,
        weights lw: QWEngineDriver.LayerW
    ) throws {
        let B = sessions.count
        let psc = driver.psc
        let geometry = driver.geometry
        let w = lw.attn!
        let H = geometry.hidden
        let fullAttnIdx = driver.attnIndexMap[l]
        let batchMLP = try driver.sharedBatchMLP()
        enc.useResource(batchMLP.downRows.region.buffer, usage: [.read, .write])
        enc.useResource(batchMLP.sums.region.buffer, usage: [.read, .write])

        if let mla = driver.mlaOperator, let mlaW = w.mla {
            let activeLn1 = driver.prefixRows(psc.ln1, rows: B, cols: H, dt: .f16)
            let latOut = driver.prefixRows(mla.mlaLatentScratch, rows: B, cols: w.qkvProj.n, dt: .f16)
            let vCols = w.o.k
            let activeAttOut = driver.prefixRows(psc.attOut, rows: B, cols: vCols, dt: .f16)
            try batchMLP.encodeProjection(
                enc, input: activeLn1, weight: w.qkvProj, output: latOut, activeRows: B)
            enc.memoryBarrier(scope: .buffers)
            var slots: [QWMLAAttnOperator.DecodeBatchSlot] = []
            slots.reserveCapacity(B)
            for b in 0..<B {
                let sess = sessions[b]
                guard let mlaState = sess.mlaBatchState, fullAttnIdx < mlaState.layers.count else {
                    throw QWEngineDriverError.state("missing per-session MLA batch state")
                }
                slots.append(QWMLAAttnOperator.DecodeBatchSlot(
                    kv: sess.attnKVViews[fullAttnIdx],
                    position: sess.position,
                    state: mlaState.layers[fullAttnIdx],
                    mropePositions: sess.mropeSlice(
                        pos: sess.position, count: 1, mropeLayout: mla.mropeLayout)
                ))
            }
            try mla.encodeDecodeBatch(
                enc, ln1: activeLn1, mlaW: mlaW, slots: slots, attOut: activeAttOut,
                encodeBatchProj: { projEnc, x, weight, y16, rows in
                    try batchMLP.encodeProjection(
                        projEnc, input: x, weight: weight, output: y16, activeRows: rows)
                    projEnc.memoryBarrier(scope: .buffers)
                })
            enc.memoryBarrier(scope: .buffers)
            try batchMLP.encodeProjection(
                enc, input: activeAttOut, weight: w.o,
                output: driver.prefixRows(psc.o16, rows: B, cols: H, dt: .f16),
                activeRows: B)
            enc.memoryBarrier(scope: .buffers)
            if let oBias = w.oBias {
                try QWNorm.encodeAddBiasF16(
                    enc, hal: driver.hal,
                    x: driver.prefixRows(psc.o16, rows: B, cols: H, dt: .f16),
                    bias: oBias, rows: B, cols: H)
                enc.memoryBarrier(scope: .buffers)
            }
            return
        }

        let qOut = geometry.qProjOut, qC = geometry.qChannels, kvC = geometry.kvChannels
        guard let attnDecode = driver.attnDecode else {
            throw QWEngineDriverError.state("batched attention requires attnDecode")
        }
        try batchMLP.encodeProjection(
            enc,
            input: driver.prefixRows(psc.ln1, rows: B, cols: H, dt: .f16),
            weight: w.qkvProj,
            output: driver.prefixRows(psc.attnY16, rows: B, cols: geometry.attnTotalN, dt: .f16),
            activeRows: B)
        enc.memoryBarrier(scope: .buffers)
        let layerSW = geometry.virtualLayerSlidingWindow(l)
        for b in 0..<B {
            let sess = sessions[b]
            let pos = sess.position
            let views = sess.attnKVViews[fullAttnIdx]
            let yRow = driver.stridedRow(psc.attnY16, row: b, cols: geometry.attnTotalN, dt: .f16)
            let qRow = driver.stridedRow(psc.attQR16, row: b, cols: qOut, dt: .f16)
            let kRow = driver.stridedRow(psc.attKR16, row: b, cols: kvC, dt: .f16)
            let vRow = driver.stridedRow(psc.attV16, row: b, cols: kvC, dt: .f16)
            let oRow = driver.stridedRow(psc.attOut, row: b, cols: qC, dt: .f16)
            try driver.ropeApply.encodeExtractQK(
                enc, hal: driver.hal,
                attnY: yRow, qOut: qRow, kOut: kRow, vOut: vRow,
                wq: w.wq, wk: w.wkTiled, qkvBias: w.qkvBias,
                positionStart: max(0, pos + sess.roundRecord.mropeDelta), count: 1)
            enc.memoryBarrier(scope: .buffers)
            let activeKeys = layerSW > 0 ? min(pos + 1, layerSW) : (pos + 1)
            try attnDecode.encode(
                enc, qgate: qRow, kv: views, knew: kRow, vnew: vRow,
                partials: driver.sc.partials, out: oRow,
                C: pos, splits: QWAttnDecode.splits(forKeys: activeKeys, kvHeads: geometry.kvHeads, gpuCores: driver.hal.caps.gpuCores),
                slidingWindow: layerSW)
            enc.memoryBarrier(scope: .buffers)
        }
        try batchMLP.encodeProjection(
            enc,
            input: driver.prefixRows(psc.attOut, rows: B, cols: qC, dt: .f16),
            weight: w.o,
            output: driver.prefixRows(psc.o16, rows: B, cols: H, dt: .f16),
            activeRows: B)
        enc.memoryBarrier(scope: .buffers)
        if let oBias = w.oBias {
            try QWNorm.encodeAddBiasF16(
                enc, hal: driver.hal,
                x: driver.prefixRows(psc.o16, rows: B, cols: H, dt: .f16),
                bias: oBias, rows: B, cols: H)
            enc.memoryBarrier(scope: .buffers)
        }
    }
}
