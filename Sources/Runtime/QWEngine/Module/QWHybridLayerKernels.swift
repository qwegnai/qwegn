// QWHybridLayerKernels.swift — GDN linear-attention and full-attention layer kernels.

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
        if geometry.gdnVectorDecay, let fgbProj = w.fgbProj, let fgb16 = sc.fgb16 {
            if context.weights.mhc == nil { mark("kda.in_proj") }
            try driver.gemvH16T4(enc, n: geometry.gdnQKVChannels, k: H, x: sc.ln1, gs: sc.gs,
                                 wQ4: w.inProj, y16: sc.qkv16)
            mark("kda.fgb_proj")
            try driver.gemvH16T4(enc, n: fgbProj.n, k: H, x: sc.ln1, gs: sc.gs,
                                 wQ4: fgbProj, y16: fgb16)
            mark("kda.gates")
            if let fB = w.fB, let gB = w.gB {
                try driver.gdnDecode.encodeKDAGates(
                    enc, fgb: fgb16, fB: fB, gB: gB, aLog: w.aLog, dtBias: w.dtBias,
                    betas: sc.betas, glog: sc.glog, zOut: sc.z16, rows: 1)
            } else {
                try driver.gdnDecode.encodeKDAGatesDirect(
                    enc, fgb: fgb16, ln1: sc.ln1, bProjF16: w.bProjF16,
                    aLog: w.aLog, dtBias: w.dtBias,
                    betas: sc.betas, glog: sc.glog, zOut: sc.z16, rows: 1)
            }
        } else {
            if context.weights.mhc == nil { mark("gdn.in_proj") }
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
        if geometry.gdnVectorDecay, let fgbProj = w.fgbProj, let fgb16 = psc.fgb16 {
            let activeLn1 = driver.prefixRows(psc.ln1, rows: T, cols: H, dt: .f16)
            mark("kda.in_proj")
            try driver.gemmBatchedH16(enc, m: T, n: qkvC, k: H, x: activeLn1, wQ4: w.inProj,
                                      y16: driver.prefixRows(psc.qkv16, rows: T, cols: qkvC, dt: .f16))
            mark("kda.fgb_proj")
            try driver.gemmBatchedH16(enc, m: T, n: fgbProj.n, k: H, x: activeLn1, wQ4: fgbProj,
                                      y16: driver.prefixRows(fgb16, rows: T, cols: fgbProj.n, dt: .f16))
            mark("kda.gates")
            let fgbRows = driver.prefixRows(fgb16, rows: T, cols: fgbProj.n, dt: .f16)
            let betasRows = driver.prefixRows(psc.betas, rows: T, cols: gdnVH, dt: .f32)
            let glogRows = driver.prefixRows(psc.glog, rows: T, cols: glogC, dt: .f32)
            let zOutRows = driver.prefixRows(psc.z16, rows: T, cols: zC, dt: .f16)
            if let fB = w.fB, let gB = w.gB {
                try driver.gdnDecode.encodeKDAGates(
                    enc, fgb: fgbRows, fB: fB, gB: gB, aLog: w.aLog, dtBias: w.dtBias,
                    betas: betasRows, glog: glogRows, zOut: zOutRows, rows: T)
            } else {
                try driver.gdnDecode.encodeKDAGatesDirect(
                    enc, fgb: fgbRows, ln1: activeLn1, bProjF16: w.bProjF16,
                    aLog: w.aLog, dtBias: w.dtBias,
                    betas: betasRows, glog: glogRows, zOut: zOutRows, rows: T)
            }
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
            if context.weights.mhc == nil { mark("gdn.in_proj") }
            driver.gemmBatchedH16GdnInProj(enc, m: T, n: geometry.gdnTotalN, k: H,
                                    x: driver.prefixRows(psc.ln1, rows: T, cols: H, dt: .f16),
                                    wQ4: w.inProj,
                                    qkv16: psc.qkv16, z16: psc.z16, y32: psc.gdnY)
            if context.weights.mhc == nil { mark("gdn.conv") }
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
            if T <= 4096 || geometry.gdnVectorDecay || psc.U.shape.elementCount < T * zC {
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
        try driver.gemmBatchedH16(enc, m: T, n: H, k: zC,
                       x: driver.prefixRows(psc.oOut, rows: T, cols: zC, dt: .f16),
                       wQ4: w.out,
                       y16: driver.prefixRows(psc.o16, rows: T, cols: H, dt: .f16))
    }

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
        let w = lw.gdn!
        let H = geometry.hidden
        let qkvC = geometry.gdnQKVChannels, zC = geometry.gdnZChannels
        let gdnVH = geometry.gdnVHeads, glogC = geometry.gdnGlogChannels
        let gi = driver.gdnIndexMap[l]
        let activeLn1 = driver.prefixRows(psc.ln1, rows: B, cols: H, dt: .f16)
        let batchMLP = try driver.sharedBatchMLP()
        enc.useResource(batchMLP.downRows.region.buffer, usage: [.read, .write])
        enc.useResource(batchMLP.sums.region.buffer, usage: [.read, .write])
        if geometry.gdnVectorDecay, let fgbProj = w.fgbProj, let fgb16 = psc.fgb16 {
            try batchMLP.encodeProjection(enc, input: activeLn1, weight: w.inProj,
                                          output: driver.prefixRows(psc.qkv16, rows: B, cols: qkvC, dt: .f16),
                                          activeRows: B)
            try batchMLP.encodeProjection(enc, input: activeLn1, weight: fgbProj,
                                          output: driver.prefixRows(fgb16, rows: B, cols: fgbProj.n, dt: .f16),
                                          activeRows: B)
            enc.memoryBarrier(scope: .buffers)
            let fgbRows = driver.prefixRows(fgb16, rows: B, cols: fgbProj.n, dt: .f16)
            let betasRows = driver.prefixRows(psc.betas, rows: B, cols: gdnVH, dt: .f32)
            let glogRows = driver.prefixRows(psc.glog, rows: B, cols: glogC, dt: .f32)
            let zOutRows = driver.prefixRows(psc.z16, rows: B, cols: zC, dt: .f16)
            if let fB = w.fB, let gB = w.gB {
                try driver.gdnDecode.encodeKDAGates(
                    enc, fgb: fgbRows, fB: fB, gB: gB, aLog: w.aLog, dtBias: w.dtBias,
                    betas: betasRows, glog: glogRows, zOut: zOutRows, rows: B)
            } else {
                try driver.gdnDecode.encodeKDAGatesDirect(
                    enc, fgb: fgbRows, ln1: activeLn1, bProjF16: w.bProjF16,
                    aLog: w.aLog, dtBias: w.dtBias,
                    betas: betasRows, glog: glogRows, zOut: zOutRows, rows: B)
            }
            enc.memoryBarrier(scope: .buffers)
            let qkvBatch = driver.prefixRows(psc.qkv16, rows: B, cols: qkvC, dt: .f16)
            let convBatch = driver.prefixRows(psc.convY, rows: B, cols: qkvC, dt: .f16)
            let oBatch = driver.prefixRows(psc.oOut, rows: B, cols: zC, dt: .f16)
            try driver.gdnConv.encodeBatchT1WithHeadnorm(
                enc, x: qkvBatch, xStride: qkvC, w: w.conv, bias: nil,
                y: convBatch, yStride: qkvC, rows: B,
                stateAt: { sessions[$0].ring[gi] })
            enc.memoryBarrier(scope: .buffers)
            try driver.gdnDecode.encodeBatch(
                enc, y: convBatch, yStride: qkvC,
                betas: betasRows, betasStride: gdnVH,
                glog: glogRows, glogStride: glogC,
                z: zOutRows, zStride: zC, w: w.normW,
                o: oBatch, oStride: zC, rows: B,
                stateAt: { sessions[$0].rec[gi] })
            enc.memoryBarrier(scope: .buffers)
        } else {
            let gdnBuf16 = QWTensorValue(region: psc.gdnY.region, offset: psc.gdnY.offset,
                                         shape: QWTensorShape([B * geometry.gdnTotalN], .f16))
            let controlsBuf = QWTensorValue(region: psc.dn32.region, offset: psc.dn32.offset,
                                            shape: QWTensorShape([B * 2 * gdnVH], .f32))
            let view = w.inProj
            if view.isIncumbentContainer || (view.bits == 4 && view.groupSize == 32 && !view.isSHVQ) {
                let isAQ4 = !view.isIncumbentContainer
                let rowPaired = B >= 4
                let name = rowPaired ? "qwm_gdn_projection_rp_m\(B)" : "qwm_gdn_projection_m\(B)"
                let pso = try driver.hal.pipeline(name, isAQ4: isAQ4)
                enc.setComputePipelineState(pso)
                view.blocks.bind(enc, index: 0)
                activeLn1.bind(enc, index: 1)
                gdnBuf16.bind(enc, index: 3)
                var nU = UInt32(geometry.gdnTotalN) | (view.isIncumbentContainer ? 0 : 0x8000_0000)
                var kU = UInt32(H)
                enc.setBytes(&nU, length: 4, index: 4)
                enc.setBytes(&kU, length: 4, index: 5)
                w.aLog.bind(enc, index: 9)
                w.dtBias.bind(enc, index: 10)
                controlsBuf.bind(enc, index: 11)
                enc.dispatchThreadgroups(MTLSize(width: (geometry.gdnTotalN + 3) / 4, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: rowPaired ? 64 : 128, height: 1, depth: 1))
            } else {
                guard let name = QWGemmAQ.gdnInProjDecodeKernelName(bits: view.bits, group: view.groupSize, isSHVQ: view.isSHVQ) else {
                    throw QWEngineDriverError.state("no batched GDN decode kernel for \(view.bits)-bit g\(view.groupSize)")
                }
                let pso = try driver.hal.pipeline(name)
                enc.setComputePipelineState(pso)
                view.blocks.bind(enc, index: 0)
                activeLn1.bind(enc, index: 1)
                gdnBuf16.bind(enc, index: 3)
                var nU = UInt32(geometry.gdnTotalN) | 0x4000_0000 | (UInt32(B) << 24)
                var kU = UInt32(H)
                enc.setBytes(&nU, length: 4, index: 4)
                enc.setBytes(&kU, length: 4, index: 5)
                gdnBuf16.bind(enc, index: 8)
                w.aLog.bind(enc, index: 9)
                w.dtBias.bind(enc, index: 10)
                controlsBuf.bind(enc, index: 11)
                controlsBuf.bind(enc, index: 12)
                enc.dispatchThreadgroups(MTLSize(width: (geometry.gdnTotalN + 15) / 16, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
            enc.memoryBarrier(scope: .buffers)
            let gdnStride = geometry.gdnTotalN
            let baStride = 2 * gdnVH
            let convBatch = driver.prefixRows(psc.convY, rows: B, cols: qkvC, dt: .f16)
            let oBatch = driver.prefixRows(psc.oOut, rows: B, cols: zC, dt: .f16)
            let zBase = QWTensorValue(region: gdnBuf16.region, offset: gdnBuf16.offset + qkvC * 2,
                                      shape: QWTensorShape([(B - 1) * gdnStride + zC], .f16))
            let glogBase = QWTensorValue(region: controlsBuf.region, offset: controlsBuf.offset + gdnVH * 4,
                                         shape: QWTensorShape([(B - 1) * baStride + glogC], .f32))
            try driver.gdnConv.encodeBatchT1WithHeadnorm(
                enc, x: gdnBuf16, xStride: gdnStride, w: w.conv, bias: nil,
                y: convBatch, yStride: qkvC, rows: B,
                stateAt: { sessions[$0].ring[gi] })
            enc.memoryBarrier(scope: .buffers)
            try driver.gdnDecode.encodeBatch(
                enc, y: convBatch, yStride: qkvC,
                betas: controlsBuf, betasStride: baStride,
                glog: glogBase, glogStride: baStride,
                z: zBase, zStride: gdnStride, w: w.normW,
                o: oBatch, oStride: zC, rows: B,
                stateAt: { sessions[$0].rec[gi] })
            enc.memoryBarrier(scope: .buffers)
        }
        try batchMLP.encodeProjection(
            enc,
            input: driver.prefixRows(psc.oOut, rows: B, cols: zC, dt: .f16),
            weight: w.out,
            output: driver.prefixRows(psc.o16, rows: B, cols: H, dt: .f16),
            activeRows: B)
        enc.memoryBarrier(scope: .buffers)
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
            let latRow = driver.prefixRows(mla.mlaLatentScratch, rows: 1, cols: w.qkvProj.n, dt: .f16)
            let qFullRow = driver.prefixRows(mla.mlaQFullScratch, rows: 1, cols: mlaW.qBProj.n, dt: .f16)
            let mropePos = context.session.mropeSlice(pos: pos, count: 1, mropeLayout: mla.mropeLayout)
            mark("mla.qkv_a_proj")
            try driver.gemvH16T4(enc, n: w.qkvProj.n, k: H, x: sc.ln1, gs: sc.gs, wQ4: w.qkvProj, y16: latRow)
            mark("mla.latent_prep")
            mla.encodeLatentPrepAndPool(
                enc, ln1: sc.ln1,
                mlaW: mlaW, kv: views,
                fullAttnLayer: fullAttnIdx, pos0: pos, rows: 1,
                mropePositions: mropePos)
            mark("mla.q_b_proj")
            let qBInput = mla.isDirectQ
                ? sc.ln1
                : driver.prefixRows(mla.qResidScratch, rows: 1, cols: mlaW.qBProj.k, dt: .f16)
            try driver.gemvH16T4(enc, n: mlaW.qBProj.n, k: mlaW.qBProj.k, x: qBInput, gs: sc.gs, wQ4: mlaW.qBProj, y16: qFullRow)
            mark("mla.attend")
            mla.encodeScoreSelectAbsorbAttendAndProjUV(
                enc, mlaW: mlaW, kv: views,
                attOut: sc.attOut,
                fullAttnLayer: fullAttnIdx, pos0: pos, rows: 1,
                ln1: sc.ln1,
                mropePositions: mropePos,
                stage: QWEngineDriver.decodeProfileEnabled
                    ? { name in driver.profileStage(name); return driver.activeEnc } : nil)
            mark("mla.o_proj")
            try driver.gemvH16T4(enc, n: H, k: geometry.qHeads * geometry.vHeadDim,
                          x: sc.attOut, gs: sc.gs48, wQ4: w.o, y16: sc.o16)
            return
        }
        if context.weights.mhc == nil {
            mark("attn.gqa")
        }
        let qkvCols = w.qkvProj.n
        let attnYSlice = driver.prefixRows(sc.attnY16, rows: 1, cols: qkvCols, dt: .f16)
        try driver.gemvH16T4(enc, n: qkvCols, k: H, x: sc.ln1, gs: sc.gs, wQ4: w.qkvProj, y16: attnYSlice)
        let ropePos = max(0, pos + context.session.roundRecord.mropeDelta)
        let turnTable = context.session.mropeTurnTableSlice(pos: pos, count: 1, rotaryDims: geometry.rotaryDims)
        // C07 fused per-head norm + partial rope (position [pos]).
        if let qkvBias = w.qkvBias {
            try driver.ropeApply.encodeExtractQK(enc, hal: driver.hal,
                                                  attnY: attnYSlice,
                                                  qOut: sc.qR, kOut: sc.kR, vOut: sc.v16,
                                                  wq: w.wq, wk: w.wkTiled,
                                                  qkvBias: qkvBias,
                                                  positionStart: ropePos, count: 1,
                                                  inRowCols: qkvCols,
                                                  customTable: turnTable)
        } else {
            try driver.ropeApply.encodeQK(enc, hal: driver.hal, q: sc.q16, k: sc.k16,
                                          wq: w.wq, wk: w.wkTiled,
                                          qOut: sc.qR, kOut: sc.kR,
                                          positionStart: ropePos, count: 1,
                                          customTable: turnTable)
        }
        if let mbsa = driver.mbsaOperator, let mbsaW = w.mbsa, pos + 1 <= mbsa.maxContext {
            let idxCols = geometry.mbsaIdxCols
            if let idxProj = mbsaW.qkvIdxProj {
                try driver.gemvH16T4(
                    enc, n: idxCols, k: H,
                    x: sc.ln1, gs: sc.gs, wQ4: idxProj,
                    y16: driver.prefixRows(mbsa.idxQKVScratch, rows: 1, cols: idxCols, dt: .f16))
                mbsa.encodeIndexPrep(enc, weights: mbsaW, ropeTable: driver.ropeApply.tableTensor,
                                     layerIndex: fullAttnIdx, pos0: pos, rows: 1)
            } else {
                let idxSlice = QWTensorValue(
                    region: sc.attnY16.region,
                    offset: sc.attnY16.offset + geometry.attnTotalN * 2,
                    shape: QWTensorShape([idxCols], .f16))
                mbsa.encodeIndexPrep(enc, weights: mbsaW, ropeTable: driver.ropeApply.tableTensor,
                                     layerIndex: fullAttnIdx, pos0: pos, rows: 1,
                                     qkvSource: idxSlice, qkvRowStrideCols: qkvCols)
            }
        }
        // C09 split-K decode: attends the new row from knew/vnew AND
        // materialises it into the page slot in the same launch.
        let attnDecode = driver.attnDecode!
        let layerSW = geometry.virtualLayerSlidingWindow(l)
        if let mbsa = driver.mbsaOperator, w.mbsa != nil,
           pos >= mbsa.config.denseEquivalenceTokens, pos + 1 <= mbsa.maxContext {
            driver.encodePagedKVWrite(enc, kv: views, k: sc.kR, v: sc.v16, startPos: pos, t: 1)
            enc.memoryBarrier(scope: .buffers)
            mbsa.encodeSelectAndAttend(enc, qgate: sc.qR, kv: views, out: sc.attOut,
                                       layerIndex: fullAttnIdx, pos0: pos, rows: 1,
                                       partials: sc.partials)
        } else if layerSW == 0, let lsa = driver.lsaOperator,
                  !(context.session.batchedDecodeTouched && pos < driver.lsaConfig.denseBypassThreshold) {
            let isSparse = pos >= driver.lsaConfig.denseBypassThreshold
            lsa.encodeIndexAndCluster(enc, qgate: sc.qR, kIn: sc.kR,
                                      fullAttnLayer: fullAttnIdx, pos0: pos, rows: 1,
                                      qMode: isSparse ? .allRows : .none)
            if isSparse {
                lsa.encodeSelectBlocks(enc, fullAttnLayer: fullAttnIdx, pos0: pos, rows: 1)
                let activeKeys = min(pos + 1, driver.lsaConfig.alignedBudget + 640)
                try attnDecode.encodeSparse(
                    enc, qgate: sc.qR, kv: views,
                    knew: sc.kR, vnew: sc.v16,
                    partials: sc.partials, out: sc.attOut,
                    blockList: lsa.blockList(forFullAttnLayer: fullAttnIdx),
                    blockCount: lsa.blockCount(forFullAttnLayer: fullAttnIdx),
                    C: pos, splits: QWAttnDecode.splits(forKeys: activeKeys, kvHeads: geometry.kvHeads, gpuCores: driver.hal.caps.gpuCores),
                    listStride: lsa.listStride)
            } else {
                try attnDecode.encode(enc, qgate: sc.qR, kv: views,
                                      knew: sc.kR, vnew: sc.v16,
                                      partials: sc.partials, out: sc.attOut,
                                      C: pos, splits: QWAttnDecode.splits(forKeys: pos + 1, kvHeads: geometry.kvHeads, gpuCores: driver.hal.caps.gpuCores),
                                      slidingWindow: layerSW)
            }
        } else {
            let activeKeys = layerSW > 0 ? min(pos + 1, layerSW) : (pos + 1)
            try attnDecode.encode(enc, qgate: sc.qR, kv: views,
                                  knew: sc.kR, vnew: sc.v16,
                                  partials: sc.partials, out: sc.attOut,
                                  C: pos, splits: QWAttnDecode.splits(forKeys: activeKeys, kvHeads: geometry.kvHeads, gpuCores: driver.hal.caps.gpuCores),
                                  slidingWindow: layerSW)
        }
        // o_proj (K = qHeads·headDim = 6144 — the C08 law, NOT 5120)
        try driver.gemvH16T4(enc, n: H, k: geometry.qHeads * geometry.headDim,
                      x: sc.attOut, gs: sc.gs48, wQ4: w.o, y16: sc.o16)
        if let oBias = w.oBias {
            try QWNorm.encodeAddBiasF16(enc, hal: driver.hal, x: sc.o16, bias: oBias, rows: 1, cols: H)
        }
    }

    override func encodeAttentionPrefill(_ context: QWLayerPrefillContext) throws {
        let driver = context.driver
        var enc = context.enc
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
            let latRows = driver.prefixRows(mla.mlaLatentScratch, rows: T, cols: w.qkvProj.n, dt: .f16)
            let qFullRows = driver.prefixRows(mla.mlaQFullScratch, rows: T, cols: mlaW.qBProj.n, dt: .f16)
            let attOutRows = driver.prefixRows(psc.attOut, rows: T, cols: vCols, dt: .f16)
            var enc = enc
            func mark(_ stage: String) -> (any MTLComputeCommandEncoder)? {
                driver.profileStage(stage)
                if let a = driver.activeEnc { enc = a }
                return driver.activeEnc
            }
            let mropePos = context.session.mropeSlice(pos: pos, count: T, mropeLayout: mla.mropeLayout)
            _ = mark("mla.qkv_a_proj")
            try driver.gemmBatchedH16(enc, m: T, n: w.qkvProj.n, k: H, x: activeLn1, wQ4: w.qkvProj, y16: latRows)
            _ = mark("mla.latent_prep")
            mla.encodeLatentPrepAndPool(
                enc, ln1: activeLn1,
                mlaW: mlaW, kv: views,
                fullAttnLayer: fullAttnIdx, pos0: pos, rows: T,
                mropePositions: mropePos)
            _ = mark("mla.q_b_proj")
            let qBInput = mla.isDirectQ
                ? activeLn1
                : driver.prefixRows(mla.qResidScratch, rows: T, cols: mlaW.qBProj.k, dt: .f16)
            try driver.gemmBatchedH16(enc, m: T, n: mlaW.qBProj.n, k: mlaW.qBProj.k, x: qBInput, wQ4: mlaW.qBProj, y16: qFullRows)
            mla.encodeScoreSelectAbsorbAttendAndProjUV(
                enc, mlaW: mlaW, kv: views,
                attOut: attOutRows,
                fullAttnLayer: fullAttnIdx, pos0: pos, rows: T,
                ln1: activeLn1,
                mropePositions: mropePos,
                stage: QWEngineDriver.prefillProfileEnabled ? mark : nil)
            _ = mark("mla.o_proj")
            enc = driver.activeEnc ?? enc
            try driver.gemmBatchedH16(enc, m: T, n: H, k: vCols,
                           x: attOutRows,
                           wQ4: w.o,
                           y16: driver.prefixRows(psc.o16, rows: T, cols: H, dt: .f16))
            return
        }
        if context.weights.mhc == nil {
            driver.profileStage("attn.gqa")
            enc = driver.activeEnc ?? enc
        }
        let qkvCols = w.qkvProj.n
        let attnYSlice = driver.prefixRows(psc.attnY16, rows: T, cols: qkvCols, dt: .f16)
        try driver.gemmBatchedH16(enc, m: T, n: qkvCols, k: H,
                       x: driver.prefixRows(psc.ln1, rows: T, cols: H, dt: .f16),
                       wQ4: w.qkvProj,
                       y16: attnYSlice)
        let prefillRopePos = max(0, pos + context.session.roundRecord.mropeDelta)
        let prefillTurnTable = context.session.mropeTurnTableSlice(pos: pos, count: T, rotaryDims: geometry.rotaryDims)
        try driver.ropeApply.encodeExtractQK(enc, hal: driver.hal,
                                              attnY: attnYSlice,
                                              qOut: driver.prefixRows(psc.attQR16, rows: T, cols: qOut, dt: .f16),
                                              kOut: driver.prefixRows(psc.attKR16, rows: T, cols: kvC, dt: .f16),
                                              vOut: driver.prefixRows(psc.attV16, rows: T, cols: kvC, dt: .f16),
                                              wq: w.wq, wk: w.wkTiled,
                                              qkvBias: w.qkvBias,
                                              positionStart: prefillRopePos, count: T,
                                              inRowCols: qkvCols,
                                              customTable: prefillTurnTable)
        if let mbsa = driver.mbsaOperator, let mbsaW = w.mbsa, pos + T <= mbsa.maxContext {
            let idxCols = geometry.mbsaIdxCols
            if let idxProj = mbsaW.qkvIdxProj {
                try driver.gemmBatchedH16(
                    enc, m: T, n: idxCols, k: H,
                    x: driver.prefixRows(psc.ln1, rows: T, cols: H, dt: .f16),
                    wQ4: idxProj,
                    y16: driver.prefixRows(mbsa.idxQKVScratch, rows: T, cols: idxCols, dt: .f16))
                enc.memoryBarrier(scope: .buffers)
                mbsa.encodeIndexPrep(enc, weights: mbsaW, ropeTable: driver.ropeApply.tableTensor,
                                     layerIndex: fullAttnIdx, pos0: pos, rows: T)
            } else {
                let idxSlice = QWTensorValue(
                    region: psc.attnY16.region,
                    offset: psc.attnY16.offset + geometry.attnTotalN * 2,
                    shape: QWTensorShape([T * qkvCols - geometry.attnTotalN], .f16))
                mbsa.encodeIndexPrep(enc, weights: mbsaW, ropeTable: driver.ropeApply.tableTensor,
                                     layerIndex: fullAttnIdx, pos0: pos, rows: T,
                                     qkvSource: idxSlice, qkvRowStrideCols: qkvCols)
            }
        }
        let layerSW = geometry.virtualLayerSlidingWindow(l)
        let useNAX = driver.hal.caps.supportsSimdgroupMMA
            && driver.geometry.supportsNAXPagedAttnPrefill
        if layerSW == 0, let lsa = driver.lsaOperator {
            let sparseActive = (pos + T) > driver.lsaConfig.denseBypassThreshold
            let qMode: QWLSAOperator.IndexPrepQMode = !sparseActive
                ? .none
                : (useNAX ? .tiled32 : .allRows)
            lsa.encodeIndexAndCluster(
                enc,
                qgate: driver.prefixRows(psc.attQR16, rows: T, cols: qOut, dt: .f16),
                kIn: driver.prefixRows(psc.attKR16, rows: T, cols: kvC, dt: .f16),
                fullAttnLayer: fullAttnIdx, pos0: pos, rows: T, qMode: qMode)
        }
        // Pad to the SELECTED prefill backend's tile (32 rows, 64 for
        // the quad kernel) — a hardcoded 32 crashed quad at S=32. Only the
        // 27B's NAX prefill reads whole tiles; the standard path does not.
        let padTile = driver.attnPrefill.tileRows
        let sp = (T + padTile - 1) / padTile * padTile
        if sp > T && QWEngineDriver.zeroPadServes(geometry: geometry) {
            try driver.encodeZeroPadAttn(enc, q: psc.attQR16, k: psc.attKR16, v: psc.attV16, t: T, sp: sp)
        }
        if let mbsa = driver.mbsaOperator, w.mbsa != nil,
           pos + T > mbsa.config.denseEquivalenceTokens, pos + T <= mbsa.maxContext {
            driver.encodePagedKVWrite(enc, kv: views,
                               k: driver.prefixRows(psc.attKR16, rows: T, cols: kvC, dt: .f16),
                               v: driver.prefixRows(psc.attV16, rows: T, cols: kvC, dt: .f16),
                               startPos: pos, t: T)
            enc.memoryBarrier(scope: .buffers)
            mbsa.encodeSelectAndAttend(
                enc,
                qgate: driver.prefixRows(psc.attQR16, rows: T, cols: qOut, dt: .f16),
                kv: views,
                out: driver.prefixRows(psc.attOut, rows: T, cols: qC, dt: .f16),
                layerIndex: fullAttnIdx, pos0: pos, rows: T,
                partials: T == 1 ? driver.sc.partials : (T <= 8 ? driver.mtpSharedAttentionPartials : nil))
        } else if layerSW == 0, let lsa = driver.lsaOperator, pos + T > driver.lsaConfig.denseBypassThreshold {
            driver.encodePagedKVWrite(enc, kv: views,
                               k: driver.prefixRows(psc.attKR16, rows: T, cols: kvC, dt: .f16),
                               v: driver.prefixRows(psc.attV16, rows: T, cols: kvC, dt: .f16),
                               startPos: pos, t: T)
            lsa.encodeSelectBlocks(enc, fullAttnLayer: fullAttnIdx, pos0: pos, rows: T, tiled32: useNAX)
            let blockListTV = lsa.blockList(forFullAttnLayer: fullAttnIdx)
            let blockCountTV = lsa.blockCount(forFullAttnLayer: fullAttnIdx)
            if !useNAX {
                try driver.pagedAttnPrefillGeneral.encodeSparse(
                    enc,
                    qgate: driver.prefixRows(psc.attQR16, rows: sp, cols: qOut, dt: .f16),
                    kv: views,
                    out: driver.prefixRows(psc.attOut, rows: T, cols: qC, dt: .f16),
                    blockList: blockListTV,
                    blockCount: blockCountTV,
                    pos: pos, T: T, listStride: lsa.listStride)
            } else {
                try driver.pagedAttnPrefill.encodeSparse(
                    enc,
                    qgate: driver.prefixRows(psc.attQR16, rows: sp, cols: qOut, dt: .f16),
                    kv: views,
                    out: driver.prefixRows(psc.attOut, rows: T, cols: qC, dt: .f16),
                    blockList: blockListTV,
                    blockCount: blockCountTV,
                    pos: pos, T: T, listStride: lsa.listStride)
            }
        } else if !useNAX {
            // General scalar prefill fallback for unsupported attention shapes.
            driver.encodePagedKVWrite(enc, kv: views,
                               k: driver.prefixRows(psc.attKR16, rows: T, cols: kvC, dt: .f16),
                               v: driver.prefixRows(psc.attV16, rows: T, cols: kvC, dt: .f16),
                               startPos: pos, t: T)
            try driver.pagedAttnPrefillGeneral.encode(enc,
                                               qgate: driver.prefixRows(psc.attQR16, rows: sp, cols: qOut, dt: .f16),
                                               kv: views,
                                               out: driver.prefixRows(psc.attOut, rows: T, cols: qC, dt: .f16),
                                               pos: pos, T: T, slidingWindow: layerSW)
        } else if pos == 0 && T < 128 && driver.geometry.matchesHead256GQA6Kernel
                    && driver.geometry.attnOutputGate && layerSW == 0 {
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
                                               pos: pos, T: T, slidingWindow: layerSW)
        }
        try driver.gemmBatchedH16(enc, m: T, n: H, k: qC,
                       x: driver.prefixRows(psc.attOut, rows: T, cols: qC, dt: .f16),
                       wQ4: w.o,
                       y16: driver.prefixRows(psc.o16, rows: T, cols: H, dt: .f16))
        if let oBias = w.oBias {
            try QWNorm.encodeAddBiasF16(
                enc, hal: driver.hal,
                x: driver.prefixRows(psc.o16, rows: T, cols: H, dt: .f16),
                bias: oBias, rows: T, cols: H)
        }
    }
}
