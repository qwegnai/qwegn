// QWInkling2Forward+MoE.swift — Fused BF16 projections and QWGemmFP MoE dispatch for Inkling-2.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL

struct Ink2AttnProjViews {
    let qBuf: any MTLBuffer
    let qOff: Int
    let gateBuf: any MTLBuffer
    let gateOff: Int
    let rBuf: any MTLBuffer
    let rOff: Int
    let sinkBuf: any MTLBuffer
    let sinkOff: Int
    let kBuf: any MTLBuffer
    let kOff: Int
    let vBuf: any MTLBuffer
    let vOff: Int
}

extension QWInkling2Forward {
    func encodeFPGemm(
        _ enc: any MTLComputeCommandEncoder,
        bank: QWGemmFP.BF16Bank,
        xBuf: any MTLBuffer, xOff: Int = 0,
        yBuf: any MTLBuffer, yOff: Int = 0,
        m: Int,
        outIsF32: Bool = false,
        softcap: Float = 0.0
    ) throws {
        let xSlice = QWFPBufferSlice(buffer: xBuf, offset: xOff, bytes: max(0, xBuf.length - xOff))
        let ySlice = QWFPBufferSlice(buffer: yBuf, offset: yOff, bytes: max(0, yBuf.length - yOff))
        try QWGemmFP.encodeBF16(
            enc, hal: hal, weight: bank, x: xSlice, y: ySlice,
            m: m, inDtype: .bf16, outDtype: outIsF32 ? .f32 : .bf16,
            softcap: softcap, minNaxRows: 17
        )
    }

    func encodeRotateToF16(
        _ enc: any MTLComputeCommandEncoder,
        srcBF16: any MTLBuffer, srcOff: Int = 0,
        dst: any MTLBuffer, dstOff: Int = 0,
        count: Int,
        rotate: Bool,
        outIsF32: Bool = false
    ) {
        guard count > 0 else { return }
        var cnt = UInt32(count)
        var flags: UInt32 = (rotate ? 1 : 0) | (outIsF32 ? 2 : 0)
        enc.setComputePipelineState(pBF16ToF16FWHT32)
        enc.setBuffer(srcBF16, offset: srcOff, index: 0)
        enc.setBuffer(dst, offset: dstOff, index: 1)
        enc.setBytes(&cnt, length: 4, index: 2)
        enc.setBytes(&flags, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: (count + 255) / 256, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func encodeF16ToBF16(
        _ enc: any MTLComputeCommandEncoder,
        src: any MTLBuffer, srcOff: Int = 0,
        dst: any MTLBuffer, dstOff: Int = 0,
        count: Int
    ) {
        guard count > 0 else { return }
        var cnt = UInt32(count)
        enc.setComputePipelineState(pF16ToBF16)
        enc.setBuffer(src, offset: srcOff, index: 0)
        enc.setBuffer(dst, offset: dstOff, index: 1)
        enc.setBytes(&cnt, length: 4, index: 2)
        let threads = (count + 3) / 4
        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, threads), height: 1, depth: 1))
    }

    func encodeF32ToBF16(
        _ enc: any MTLComputeCommandEncoder,
        src: any MTLBuffer, srcOff: Int = 0,
        dst: any MTLBuffer, dstOff: Int = 0,
        count: Int,
        softcap: Float = 0.0
    ) {
        guard count > 0 else { return }
        var cnt = UInt32(count)
        var cap = softcap
        enc.setComputePipelineState(pF32ToBF16)
        enc.setBuffer(src, offset: srcOff, index: 0)
        enc.setBuffer(dst, offset: dstOff, index: 1)
        enc.setBytes(&cnt, length: 4, index: 2)
        enc.setBytes(&cap, length: 4, index: 3)
        let threads = (count + 3) / 4
        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, threads), height: 1, depth: 1))
    }

    func encodeSoftcapF32(
        _ enc: any MTLComputeCommandEncoder,
        data: any MTLBuffer, offset: Int = 0,
        count: Int, softcap: Float
    ) {
        guard count > 0, softcap > 0 else { return }
        var cnt = UInt32(count)
        var cap = softcap
        enc.setComputePipelineState(pSoftcapF32)
        enc.setBuffer(data, offset: offset, index: 0)
        enc.setBytes(&cnt, length: 4, index: 1)
        enc.setBytes(&cap, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, count), height: 1, depth: 1))
    }

    func encodeLinear(
        _ enc: any MTLComputeCommandEncoder,
        weight: Ink2LinearW,
        xBuf: any MTLBuffer, xOff: Int = 0,
        yBuf: any MTLBuffer, yOff: Int = 0,
        m: Int,
        xAlreadyRotatedF16: Bool = false,
        x16ByteOffset: Int = 0
    ) throws {
        guard m > 0 else { return }
        switch weight {
        case .bf16(let bank):
            try encodeFPGemm(enc, bank: bank, xBuf: xBuf, xOff: xOff, yBuf: yBuf, yOff: yOff, m: m)
        case .quant(let view):
            if !xAlreadyRotatedF16 {
                encodeRotateToF16(enc, srcBF16: xBuf, srcOff: xOff, dst: session.qwpF16InBuf,
                                  count: m * view.k, rotate: view.isRotated, outIsF32: false)
            }
            let x16TV = session.qwpF16InTV.slice(byteOffset: xAlreadyRotatedF16 ? x16ByteOffset : 0,
                                                 count: m * view.k, dtype: .f16)
            let yBF16TV = QWTensorValue(region: QWMRegion(buffer: yBuf, base: 0, length: yBuf.length),
                                        offset: yOff, shape: QWTensorShape([m * view.n], .bf16))
            if m > 16 && hal.caps.supportsSimdgroupMMA
                && ((view.n % 64) == 0 || !view.isIncumbentContainer) && (view.k % 64) == 0 {
                try QWGemm.encodeNaxF32(enc, hal: hal, weight: view, x16: x16TV, y16: x16TV, y32: yBF16TV, rows: m)
                return
            }
            if view.isSHVQ && m <= 16 {
                try QWGemmAQ.encodeDecodeBF16(
                    enc, hal: hal, blocks: view.blocks, x: x16TV, yBF16: yBF16TV,
                    n: view.n, k: view.k, bits: view.bits, group: view.groupSize,
                    rows: m, isSHVQ: true
                )
                return
            }
            if !view.isSHVQ && view.bits == 4 && view.groupSize == 32 && m <= 16 {
                try QWGemm.encodeHQ4DecodeBF16(
                    enc, hal: hal, weight: view, x16: x16TV, yBF16: yBF16TV, rows: m
                )
                return
            }
            if !view.isIncumbentContainer {
                let maxRowsPerChunk = max(1, session.qwpF32OutElems / max(1, view.n))
                var r0 = 0
                while r0 < m {
                    let chunk = min(maxRowsPerChunk, m - r0)
                    let xSub = x16TV.slice(byteOffset: r0 * view.k * 2, count: chunk * view.k, dtype: .f16)
                    let ySub = session.qwpF32OutTV.slice(byteOffset: 0, count: chunk * view.n, dtype: .f32)
                    try QWGemmAQ.encode(enc, hal: hal, blocks: view.blocks, x: xSub, y: ySub,
                                        n: view.n, k: view.k, bits: view.bits, group: view.groupSize,
                                        rows: chunk, isSHVQ: view.isSHVQ)
                    encodeF32ToBF16(enc, src: session.qwpF32OutBuf, dst: yBuf,
                                    dstOff: yOff + r0 * view.n * 2, count: chunk * view.n, softcap: 0)
                    r0 += chunk
                }
                return
            }
            let plan = QWGemm.plan(QWGemmShape(m: 1, n: view.n, k: view.k), dtype: .q4g, epi: .none, outDtype: .f16)
            let maxRows = max(1, session.qwpF16OutTV.shape.elementCount / max(1, view.n))
            var r0 = 0
            while r0 < m {
                let chunk = min(maxRows, m - r0)
                for rr in 0..<chunk {
                    let xSub = x16TV.slice(byteOffset: (r0 + rr) * view.k * 2, count: view.k, dtype: .f16)
                    let ySub = session.qwpF16OutTV.slice(byteOffset: rr * view.n * 2, count: view.n, dtype: .f16)
                    try QWGemm.encodeH16(enc, hal: hal, plan: plan, x: xSub, wQ4: view, y16: ySub)
                }
                encodeF16ToBF16(enc, src: session.qwpF16OutBuf, dst: yBuf,
                                dstOff: yOff + r0 * view.n * 2, count: chunk * view.n)
                r0 += chunk
            }
        }
    }

    func encodeLinearF32(
        _ enc: any MTLComputeCommandEncoder,
        weight: Ink2LinearW,
        xBuf: any MTLBuffer, xOff: Int = 0,
        yF32Buf: any MTLBuffer, yOff: Int = 0,
        m: Int,
        xIsF32: Bool = false
    ) throws {
        guard m > 0 else { return }
        let yTV = QWTensorValue(region: QWMRegion(buffer: yF32Buf, base: 0, length: yF32Buf.length),
                                offset: yOff, shape: QWTensorShape([m * weight.n], .f32))
        switch weight {
        case .bf16(let bank):
            let xSlice = QWFPBufferSlice(buffer: xBuf, offset: xOff, bytes: max(0, xBuf.length - xOff))
            try QWGemmFP.encodeBF16(enc, hal: hal, weight: bank, x: xSlice, y: QWFPBufferSlice(yTV),
                                    m: m, inDtype: xIsF32 ? .f32 : .bf16, outDtype: .f32, minNaxRows: 17)
        case .quant(let view):
            if xIsF32 {
                encodeF32ToBF16(enc, src: xBuf, srcOff: xOff, dst: session.qwpF16OutBuf, count: m * view.k, softcap: 0)
                encodeRotateToF16(enc, srcBF16: session.qwpF16OutBuf, dst: session.qwpF16InBuf,
                                  count: m * view.k, rotate: view.isRotated, outIsF32: false)
            } else {
                encodeRotateToF16(enc, srcBF16: xBuf, srcOff: xOff, dst: session.qwpF16InBuf,
                                  count: m * view.k, rotate: view.isRotated, outIsF32: false)
            }
            let x16TV = session.qwpF16InTV.slice(byteOffset: 0, count: m * view.k, dtype: .f16)
            if m > 16 && hal.caps.supportsSimdgroupMMA
                && ((view.n % 64) == 0 || !view.isIncumbentContainer) && (view.k % 64) == 0 {
                try QWGemm.encodeNaxF32(enc, hal: hal, weight: view, x16: x16TV, y16: x16TV, y32: yTV, rows: m)
                return
            }
            if !view.isIncumbentContainer {
                try QWGemmAQ.encode(enc, hal: hal, blocks: view.blocks, x: x16TV, y: yTV,
                                    n: view.n, k: view.k, bits: view.bits, group: view.groupSize,
                                    rows: m, isSHVQ: view.isSHVQ)
                return
            }
            let plan = QWGemm.plan(QWGemmShape(m: 1, n: view.n, k: view.k), dtype: .q4g, epi: .none)
            for rr in 0..<m {
                let xSub = x16TV.slice(byteOffset: rr * view.k * 2, count: view.k, dtype: .f16)
                let ySub = yTV.slice(byteOffset: rr * view.n * 4, count: view.n, dtype: .f32)
                try QWGemm.encode(enc, hal: hal, plan: plan, x: xSub, wQ4: view, y: ySub)
            }
        }
    }

    private func encodeFusedQuantProj(
        _ enc: any MTLComputeCommandEncoder,
        view: QWWeightView,
        rows t: Int,
        xAlreadyRotatedF16: Bool = false
    ) throws {
        let yBF16 = QWTensorValue(
            region: QWMRegion(buffer: session.fusedAttnProjBuf, base: 0, length: session.fusedAttnProjBuf.length),
            offset: 0, shape: QWTensorShape([t * view.n], .bf16)
        )
        if !xAlreadyRotatedF16 {
            encodeRotateToF16(enc, srcBF16: session.normBuf, dst: session.qwpF16InBuf,
                              count: t * view.k, rotate: view.isRotated)
        }
        let x16 = session.qwpF16InTV.slice(byteOffset: 0, count: t * view.k, dtype: .f16)
        if t > 16 && hal.caps.supportsSimdgroupMMA
            && ((view.n % 64) == 0 || !view.isIncumbentContainer) && (view.k % 64) == 0 {
            try QWGemm.encodeNaxF32(enc, hal: hal, weight: view, x16: x16, y16: x16, y32: yBF16, rows: t)
            return
        }
        if view.isSHVQ && t <= 16 {
            try QWGemmAQ.encodeDecodeBF16(
                enc, hal: hal, blocks: view.blocks, x: x16, yBF16: yBF16,
                n: view.n, k: view.k, bits: view.bits, group: view.groupSize,
                rows: t, isSHVQ: true
            )
        } else if !view.isSHVQ && view.bits == 4 && view.groupSize == 32 && t <= 16 {
            try QWGemm.encodeHQ4DecodeBF16(enc, hal: hal, weight: view, x16: x16, yBF16: yBF16, rows: t)
        } else if !view.isIncumbentContainer {
            let maxRows = max(1, session.qwpF32OutElems / max(1, view.n))
            var r0 = 0
            while r0 < t {
                let chunk = min(maxRows, t - r0)
                let xSub = x16.slice(byteOffset: r0 * view.k * 2, count: chunk * view.k, dtype: .f16)
                let ySub = session.qwpF32OutTV.slice(byteOffset: 0, count: chunk * view.n, dtype: .f32)
                try QWGemmAQ.encode(enc, hal: hal, blocks: view.blocks, x: xSub, y: ySub,
                                    n: view.n, k: view.k, bits: view.bits, group: view.groupSize,
                                    rows: chunk, isSHVQ: view.isSHVQ)
                encodeF32ToBF16(enc, src: session.qwpF32OutBuf, dst: session.fusedAttnProjBuf,
                                dstOff: r0 * view.n * 2, count: chunk * view.n, softcap: 0)
                r0 += chunk
            }
        } else {
            let plan = QWGemm.plan(QWGemmShape(m: 1, n: view.n, k: view.k),
                                   dtype: .q4g, epi: .none, outDtype: .f16)
            for rr in 0..<t {
                let xSub = x16.slice(byteOffset: rr * view.k * 2, count: view.k, dtype: .f16)
                let ySub = session.qwpF16OutTV.slice(byteOffset: rr * view.n * 2, count: view.n, dtype: .f16)
                try QWGemm.encodeH16(enc, hal: hal, plan: plan, x: xSub, wQ4: view, y16: ySub)
            }
            encodeF16ToBF16(enc, src: session.qwpF16OutBuf, dst: session.fusedAttnProjBuf, count: t * view.n)
        }
    }

    private func finalizeFusedAttnProj(
        _ enc: any MTLComputeCommandEncoder,
        lw: QWInkling2Weights.LayerWeights,
        t: Int,
        hasKV: Bool,
        kNormOff: Int,
        vNormOff: Int,
        kvAlreadyRotatedF16: Bool = false
    ) throws -> Ink2AttnProjViews {
        let lo = lw.projLayout
        if t == 1 {
            if !hasKV {
                try encodeLinear(enc, weight: lw.kProjW, xBuf: session.normBuf, xOff: kNormOff,
                                 yBuf: session.kProjBuf, m: 1,
                                 xAlreadyRotatedF16: kvAlreadyRotatedF16,
                                 x16ByteOffset: kvAlreadyRotatedF16 ? kNormOff : 0)
                try encodeLinear(enc, weight: lw.vProjW, xBuf: session.normBuf, xOff: vNormOff,
                                 yBuf: session.vProjBuf, m: 1,
                                 xAlreadyRotatedF16: kvAlreadyRotatedF16,
                                 x16ByteOffset: kvAlreadyRotatedF16 ? vNormOff : 0)
            }
            return Ink2AttnProjViews(
                qBuf: session.fusedAttnProjBuf, qOff: lo.qOutByteOffset,
                gateBuf: session.fusedAttnProjBuf, gateOff: lo.gateOutByteOffset,
                rBuf: session.fusedAttnProjBuf, rOff: lo.rOutByteOffset,
                sinkBuf: session.fusedAttnProjBuf, sinkOff: lo.sinkOutByteOffset,
                kBuf: hasKV ? session.fusedAttnProjBuf : session.kProjBuf,
                kOff: hasKV ? lo.kOutByteOffset : 0,
                vBuf: hasKV ? session.fusedAttnProjBuf : session.vProjBuf,
                vOff: hasKV ? lo.vOutByteOffset : 0
            )
        }
        let totalCols = hasKV ? lo.totalRows : lo.qGateRSinkRows
        var sp = QWInkling2Forward.SplitAttnProjP(
            T: UInt32(t), totalCols: UInt32(totalCols),
            qDim: UInt32(lo.qDim), gateDim: UInt32(lo.qDim),
            rDim: UInt32(lo.rDim), sinkDim: UInt32(lo.sinkDim),
            kvDim: UInt32(lo.kvDim), hasKV: hasKV ? 1 : 0
        )
        enc.setComputePipelineState(pSplitAttnProj)
        enc.setBuffer(session.fusedAttnProjBuf, offset: 0, index: 0)
        enc.setBuffer(session.qProjBuf, offset: 0, index: 1)
        enc.setBuffer(session.gateBuf, offset: 0, index: 2)
        enc.setBuffer(session.rProjBuf, offset: 0, index: 3)
        enc.setBuffer(session.sinkBuf, offset: 0, index: 4)
        enc.setBuffer(session.kProjBuf, offset: 0, index: 5)
        enc.setBuffer(session.vProjBuf, offset: 0, index: 6)
        enc.setBytes(&sp, length: MemoryLayout<QWInkling2Forward.SplitAttnProjP>.stride, index: 7)
        let gridW = (totalCols + 7) / 8
        enc.dispatchThreads(MTLSize(width: gridW, height: t, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, gridW), height: 1, depth: 1))
        if !hasKV {
            try encodeLinear(enc, weight: lw.kProjW, xBuf: session.normBuf, xOff: kNormOff,
                             yBuf: session.kProjBuf, m: t,
                             xAlreadyRotatedF16: kvAlreadyRotatedF16,
                             x16ByteOffset: kvAlreadyRotatedF16 ? kNormOff : 0)
            try encodeLinear(enc, weight: lw.vProjW, xBuf: session.normBuf, xOff: vNormOff,
                             yBuf: session.vProjBuf, m: t,
                             xAlreadyRotatedF16: kvAlreadyRotatedF16,
                             x16ByteOffset: kvAlreadyRotatedF16 ? vNormOff : 0)
        }
        return Ink2AttnProjViews(
            qBuf: session.qProjBuf, qOff: 0,
            gateBuf: session.gateBuf, gateOff: 0,
            rBuf: session.rProjBuf, rOff: 0,
            sinkBuf: session.sinkBuf, sinkOff: 0,
            kBuf: session.kProjBuf, kOff: 0,
            vBuf: session.vProjBuf, vOff: 0
        )
    }

    func encodeAttnProjections(
        _ enc: any MTLComputeCommandEncoder,
        lw: QWInkling2Weights.LayerWeights,
        t: Int,
        kNormOff: Int,
        vNormOff: Int
    ) throws -> Ink2AttnProjViews {
        let fusedY = QWFPBufferSlice(
            buffer: session.fusedAttnProjBuf, offset: 0, bytes: session.fusedAttnProjBuf.length
        )
        let x0 = QWFPBufferSlice(buffer: session.normBuf, offset: 0, bytes: session.normBuf.length)
        if lw.isSliding, kNormOff == 0, vNormOff == 0 {
            if let qgrskv = lw.qgrskvBank {
                try QWGemmFP.encodeBF16(enc, hal: hal, weight: qgrskv, x: x0, y: fusedY,
                                        m: t, inDtype: .bf16, outDtype: .bf16, minNaxRows: 17)
                return try finalizeFusedAttnProj(enc, lw: lw, t: t, hasKV: true, kNormOff: kNormOff, vNormOff: vNormOff)
            }
            if let qgrskv = lw.qgrskvView {
                try encodeFusedQuantProj(enc, view: qgrskv, rows: t)
                return try finalizeFusedAttnProj(enc, lw: lw, t: t, hasKV: true, kNormOff: kNormOff, vNormOff: vNormOff)
            }
        }
        if let qgrs = lw.qgrsBank {
            try QWGemmFP.encodeBF16(enc, hal: hal, weight: qgrs, x: x0, y: fusedY,
                                    m: t, inDtype: .bf16, outDtype: .bf16, minNaxRows: 17)
            return try finalizeFusedAttnProj(enc, lw: lw, t: t, hasKV: false, kNormOff: kNormOff, vNormOff: vNormOff)
        }
        if let qgrs = lw.qgrsView {
            let canBatch3NormRot = (t <= 16)
                && kNormOff == t * qgrs.k * 2
                && vNormOff == 2 * t * qgrs.k * 2
                && {
                    if case .quant(let kv) = lw.kProjW, case .quant(let vv) = lw.vProjW {
                        return kv.k == qgrs.k && vv.k == qgrs.k
                            && kv.isRotated == qgrs.isRotated && vv.isRotated == qgrs.isRotated
                    }
                    return false
                }()
            if canBatch3NormRot {
                encodeRotateToF16(enc, srcBF16: session.normBuf, dst: session.qwpF16InBuf,
                                  count: 3 * t * qgrs.k, rotate: qgrs.isRotated)
            }
            try encodeFusedQuantProj(enc, view: qgrs, rows: t, xAlreadyRotatedF16: canBatch3NormRot)
            return try finalizeFusedAttnProj(enc, lw: lw, t: t, hasKV: false, kNormOff: kNormOff, vNormOff: vNormOff,
                                             kvAlreadyRotatedF16: canBatch3NormRot)
        }

        let sharedRot: Bool? = {
            guard case .quant(let qv) = lw.qProjW,
                  case .quant(let rv) = lw.rProjW,
                  case .quant(let sv) = lw.sinkProjW,
                  case .quant(let gv) = lw.gateProjW,
                  qv.isRotated == rv.isRotated,
                  qv.isRotated == sv.isRotated,
                  qv.isRotated == gv.isRotated else { return nil }
            return qv.isRotated
        }()
        let batchAll3 = sharedRot != nil && t <= 16
            && kNormOff == t * lw.qProjW.k * 2 && vNormOff == 2 * t * lw.qProjW.k * 2
            && {
                if case .quant(let kv) = lw.kProjW, case .quant(let vv) = lw.vProjW {
                    return kv.k == lw.qProjW.k && vv.k == lw.qProjW.k
                        && kv.isRotated == sharedRot && vv.isRotated == sharedRot
                }
                return false
            }()
        if let rot = sharedRot {
            encodeRotateToF16(enc, srcBF16: session.normBuf, dst: session.qwpF16InBuf,
                              count: (batchAll3 ? 3 : 1) * t * lw.qProjW.k, rotate: rot)
        }
        let shared = sharedRot != nil
        try encodeLinear(enc, weight: lw.qProjW, xBuf: session.normBuf, yBuf: session.qProjBuf, m: t,
                         xAlreadyRotatedF16: shared)
        try encodeLinear(enc, weight: lw.rProjW, xBuf: session.normBuf, yBuf: session.rProjBuf, m: t,
                         xAlreadyRotatedF16: shared)
        try encodeLinear(enc, weight: lw.sinkProjW, xBuf: session.normBuf, yBuf: session.sinkBuf, m: t,
                         xAlreadyRotatedF16: shared)
        try encodeLinear(enc, weight: lw.gateProjW, xBuf: session.normBuf, yBuf: session.gateBuf, m: t,
                         xAlreadyRotatedF16: shared)
        let kShared = batchAll3 || (shared && kNormOff == 0 && {
            if case .quant(let kv) = lw.kProjW { return kv.isRotated == sharedRot }
            return false
        }())
        let vShared = batchAll3 || (shared && vNormOff == 0 && {
            if case .quant(let vv) = lw.vProjW { return vv.isRotated == sharedRot }
            return false
        }())
        try encodeLinear(enc, weight: lw.kProjW, xBuf: session.normBuf, xOff: kNormOff,
                         yBuf: session.kProjBuf, m: t, xAlreadyRotatedF16: kShared,
                         x16ByteOffset: batchAll3 ? kNormOff : 0)
        try encodeLinear(enc, weight: lw.vProjW, xBuf: session.normBuf, xOff: vNormOff,
                         yBuf: session.vProjBuf, m: t, xAlreadyRotatedF16: vShared,
                         x16ByteOffset: batchAll3 ? vNormOff : 0)
        return Ink2AttnProjViews(
            qBuf: session.qProjBuf, qOff: 0,
            gateBuf: session.gateBuf, gateOff: 0,
            rBuf: session.rProjBuf, rOff: 0,
            sinkBuf: session.sinkBuf, sinkOff: 0,
            kBuf: session.kProjBuf, kOff: 0,
            vBuf: session.vProjBuf, vOff: 0
        )
    }

    func encodeMoEBlock(
        lw: QWInkling2Weights.LayerWeights,
        t: Int,
        H: Int,
        nExp: Int,
        topK: Int,
        moeI: Int,
        ink: QWModelDescriptor.Inkling2Config,
        profiler: QWInkling2Profiler?,
        cb: any MTLCommandBuffer,
        beginEncoder: (Ink2Stage) throws -> any MTLComputeCommandEncoder,
        moeSpecBudget: Int = 0
    ) throws -> any MTLComputeCommandEncoder {
        let rtEnc = try beginEncoder(.moeRouteBucket)
        let useUnionPrune = moeSpecBudget > 0 && t > 1 && t <= 16 && topK <= 16
            && t * topK <= 256 && moeSpecBudget < t * topK
        var rtp = RouterP(T: UInt32(t), H: UInt32(H), numE: UInt32(nExp),
                          topK: UInt32(topK), routeScale: ink.routeScale,
                          moeSpecBudget: useUnionPrune ? 0 : UInt32(max(0, moeSpecBudget)))
        if nExp <= 512 {
            let normRowSlice = QWFPBufferSlice(buffer: session.normBuf, offset: 0, bytes: t * H * 2)
            try QWGemmFP.encodeBF16(
                rtEnc, hal: hal, weight: lw.routerGateBank,
                x: normRowSlice, y: session.routerLogitsSlice,
                m: t, inDtype: .bf16, outDtype: .f32, minNaxRows: 17
            )
            rtEnc.setComputePipelineState(pRouterSelect)
            rtEnc.setBuffer(session.routerLogitsF32Buf, offset: 0, index: 0)
            rtEnc.setBuffer(lw.routerBias.buffer, offset: lw.routerBias.offset, index: 1)
            rtEnc.setBuffer(session.moeTopIdsBuf, offset: 0, index: 2)
            rtEnc.setBuffer(session.moeTopWBuf, offset: 0, index: 3)
            rtEnc.setBytes(&rtp, length: MemoryLayout<RouterP>.stride, index: 4)
            rtEnc.dispatchThreadgroups(MTLSize(width: t, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        } else {
            rtEnc.setComputePipelineState(pRouter)
            rtEnc.setBuffer(session.normBuf, offset: 0, index: 0)
            rtEnc.setBuffer(lw.routerGate.buffer, offset: lw.routerGate.offset, index: 1)
            rtEnc.setBuffer(lw.routerBias.buffer, offset: lw.routerBias.offset, index: 2)
            rtEnc.setBuffer(session.moeTopIdsBuf, offset: 0, index: 3)
            rtEnc.setBuffer(session.moeTopWBuf, offset: 0, index: 4)
            rtEnc.setBytes(&rtp, length: MemoryLayout<RouterP>.stride, index: 5)
            rtEnc.dispatchThreadgroups(MTLSize(width: t, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        if useUnionPrune {
            var unionP = RouterP(T: UInt32(t), H: UInt32(H), numE: UInt32(nExp),
                                 topK: UInt32(topK), routeScale: ink.routeScale,
                                 moeSpecBudget: UInt32(moeSpecBudget))
            rtEnc.setComputePipelineState(try hal.pipeline("qwg_ink2_router_union_prune"))
            rtEnc.setBuffer(session.moeTopIdsBuf, offset: 0, index: 0)
            rtEnc.setBuffer(session.moeTopWBuf, offset: 0, index: 1)
            rtEnc.setBytes(&unionP, length: MemoryLayout<RouterP>.stride, index: 2)
            rtEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }

        var moeEnc = try beginEncoder(.moeExperts)
        let profilerRef = profiler
        let rotateForMoE: (inout any MTLComputeCommandEncoder) throws -> Void = { current in
            guard let profilerRef else {
                try self.hal.rotateEncoder(cb, &current)
                return
            }
            var tmp: (any MTLComputeCommandEncoder)? = current
            profilerRef.endActiveStage(encoder: &tmp)
            tmp?.endEncoding()
            tmp = nil
            profilerRef.beginStage(.moeExperts, on: cb, encoder: &tmp, hal: self.hal)
            if let existing = tmp {
                current = existing
            } else {
                guard let next = cb.makeComputeCommandEncoder() else {
                    throw QWInkling2Error.badState("failed to recreate compute encoder")
                }
                self.hal.touchKeepAlive(next)
                current = next
            }
        }

        if case .qwp(let gateUp, let down) = lw.experts {
            if hal.caps.supportsSimdgroupMMA && t > 16 {
                encodeRotateToF16(moeEnc, srcBF16: session.normBuf, dst: session.qwpF16InBuf,
                                  count: t * H, rotate: gateUp.isRotated, outIsF32: false)
                let xInSlice = QWFPBufferSlice(buffer: session.qwpF16InBuf, offset: 0, bytes: t * H * 2)
                try QWGemm.encodeMoEBankRoutedCombine(
                    &moeEnc, hal: hal,
                    x: xInSlice, xDtype: .f16, interDtype: .f16,
                    gateUp: lw.experts.gateUpBank, up: lw.experts.upBank, down: lw.experts.downBank,
                    idx: session.moeTopIdsTV, wts: session.moeTopWTV,
                    shared: nil, sharedDtype: .bf16, gate: nil,
                    out: session.moeOutSlice, outDtype: .bf16,
                    expOff: session.moeOffsetsTV, sortedPair: session.moeSortedTV,
                    sortedTok: session.moeSortedTokTV, invPerm: session.moeInvPermTV,
                    tileE: session.moeTileETV, tileR: session.moeTileRTV, tileCount: session.moeTileCountTV,
                    activeExperts: nil, activeIndirect: nil,
                    guBuf: session.moePrefillGUF16TV, actBuf: session.moeActF16TV, dnAccBuf: session.moeSlotOutF16TV,
                    hidden: H, experts: nExp, topK: topK, intermediate: moeI, rows: t,
                    gateSoftcap: ink.mlpGateSoftcap, upSoftcap: ink.mlpUpSoftcap,
                    supportsGroupedNaxPrefill: true, concurrentBarriers: false,
                    rotateEncoder: rotateForMoE
                )
            } else {
                let actTV = session.qwpMoEActF32TV.slice(byteOffset: 0, count: t * topK * moeI, dtype: .f32)
                let out32TV = session.qwpMoEOutF32TV.slice(byteOffset: 0, count: t * H, dtype: .f32)
                let idsSubTV = session.moeTopIdsTV.slice(byteOffset: 0, count: t * topK, dtype: .i32)
                let wtsSubTV = session.moeTopWTV.slice(byteOffset: 0, count: t * topK, dtype: .f32)
                let moeOutBF16TV = QWTensorValue(
                    region: QWMRegion(buffer: session.moeOutBuf, base: 0, length: session.moeOutBuf.length),
                    offset: 0, shape: QWTensorShape([t * H], .bf16)
                )
                let compactSlotCapOk = t * topK * H <= session.qwpMoEGUF32TV.shape.elementCount
                encodeRotateToF16(moeEnc, srcBF16: session.normBuf, dst: session.qwpMoERotF32Buf,
                                  count: t * H, rotate: gateUp.isRotated, outIsF32: true)
                let xRotTV = session.qwpMoERotF32TV.slice(byteOffset: 0, count: t * H, dtype: .f32)
                if t >= 2 && QWGemm.supportsCompactQWP(gateUp: gateUp, down: down) && compactSlotCapOk {
                    let dnSlotTV = session.qwpMoEGUF32TV.slice(byteOffset: 0, count: t * topK * H, dtype: .f32)
                    try QWGemm.encodeMoEQWPCompactRoutedCombine(
                        moeEnc, hal: hal,
                        x: xRotTV, gateUp: gateUp, down: down,
                        ids: idsSubTV, weights: wtsSubTV,
                        shared: out32TV, gate: wtsSubTV, out: out32TV,
                        expertOff: session.moeOffsetsTV,
                        activeExperts: session.moeActiveExpertsTV,
                        activeIndirect: session.moeActiveArgsTV,
                        sortedPairIdx: session.moeSortedTV,
                        sortedTok: session.moeSortedTokTV,
                        actBuf: actTV, dnAccBuf: dnSlotTV,
                        hidden: H, experts: nExp, routes: topK, intermediate: moeI, tokens: t,
                        out16: moeOutBF16TV,
                        gateSoftcap: ink.mlpGateSoftcap, upSoftcap: ink.mlpUpSoftcap,
                        applyFwht: (moeI % 32 == 0) && down.isRotated,
                        skipShared: true
                    )
                    return moeEnc
                }
                let guTV = session.qwpMoEGUF32TV.slice(byteOffset: 0, count: t * topK * 2 * moeI, dtype: .f32)
                let fuseGateUp = (moeI % 32 == 0) && (down.isRotated == (weights.weightBasis == .rotated))
                if fuseGateUp {
                    try QWGemm.encodeMoEGateUpF32R1(
                        moeEnc, hal: hal, x: xRotTV, bank: gateUp, ids: idsSubTV, out: actTV,
                        rowsPerExpert: 2 * moeI, routes: topK, tokens: t, fuseSiluFwht: true,
                        reuseExpertsAcrossTokens: true,
                        gateSoftcap: ink.mlpGateSoftcap, upSoftcap: ink.mlpUpSoftcap
                    )
                } else {
                    try QWGemm.encodeMoEGateUpF32R1(
                        moeEnc, hal: hal, x: xRotTV, bank: gateUp, ids: idsSubTV, out: guTV,
                        rowsPerExpert: 2 * moeI, routes: topK, tokens: t, fuseSiluFwht: false
                    )
                    try QWGemmFP.encodeSwiGLUSoftcap(
                        moeEnc, hal: hal, gu: QWFPBufferSlice(guTV), act: QWFPBufferSlice(actTV),
                        rows: t * topK, intermediate: moeI,
                        gateCap: ink.mlpGateSoftcap, upCap: ink.mlpUpSoftcap,
                        inDtype: .f32, outDtype: .f32, rotateFwht32: down.isRotated
                    )
                }
                try QWGemm.encodeMoEDownSharedCombineF32(
                    moeEnc, hal: hal, x: actTV, bank: down,
                    rowsPerRoute: H, routes: topK, weights: wtsSubTV,
                    shared: out32TV, gate: wtsSubTV, out: out32TV,
                    ids: idsSubTV, tokens: t, out16: moeOutBF16TV, reuseExpertsAcrossTokens: false, skipShared: true
                )
            }
            return moeEnc
        }

        let canGrouped = hal.caps.supportsSimdgroupMMA
        let useGroupedNax = canGrouped && t > 16
        try QWGemm.encodeMoEBankRoutedCombine(
            &moeEnc, hal: hal,
            x: session.normSlice, xDtype: .bf16, interDtype: .bf16,
            gateUp: lw.experts.gateUpBank, up: lw.experts.upBank, down: lw.experts.downBank,
            idx: session.moeTopIdsTV, wts: session.moeTopWTV,
            shared: nil, sharedDtype: .bf16, gate: nil,
            out: session.moeOutSlice, outDtype: .bf16,
            expOff: session.moeOffsetsTV, sortedPair: session.moeSortedTV,
            sortedTok: session.moeSortedTokTV, invPerm: session.moeInvPermTV,
            tileE: useGroupedNax ? session.moeTileETV : nil,
            tileR: useGroupedNax ? session.moeTileRTV : nil,
            tileCount: useGroupedNax ? session.moeTileCountTV : nil,
            activeExperts: useGroupedNax ? nil : session.moeActiveExpertsTV,
            activeIndirect: useGroupedNax ? nil : session.moeActiveArgsTV,
            guBuf: session.moePrefillGUTV, actBuf: session.moeActTV, dnAccBuf: session.moeSlotOutTV,
            hidden: H, experts: nExp, topK: topK, intermediate: moeI, rows: t,
            gateSoftcap: ink.mlpGateSoftcap, upSoftcap: ink.mlpUpSoftcap,
            supportsGroupedNaxPrefill: canGrouped, concurrentBarriers: false,
            rotateEncoder: (t > 1) ? rotateForMoE : nil,
            minNaxRows: 17
        )
        return moeEnc
    }
}
