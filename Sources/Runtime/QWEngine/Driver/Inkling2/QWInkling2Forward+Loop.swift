// QWInkling2Forward+Loop.swift — Zero-blit copies, fused AttnRes/Norm, fast controllers, and GPU head dispatch.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWHead

struct Ink2AttnResCtrlNormP {
    var T, H, nBlocks, numQ, numOut, blockStride, partialIsF32, normalizeValues: UInt32
    var useCtrl, rank, rotateOut, outIsF16, outStride, msgStride: UInt32
    var attnResEps, rmsEps: Float
    var slotIds: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
    var normWOffsets: (UInt32, UInt32, UInt32, UInt32)
    var ctrlDownOffsets: (UInt32, UInt32, UInt32, UInt32)
    var ctrlForgetOffsets: (UInt32, UInt32, UInt32, UInt32)
    var ctrlGainOffsets: (UInt32, UInt32, UInt32, UInt32)
}

struct Ink2CtrlBatchP {
    var T, H, rank, thStride: UInt32
    var eps: Float
    var numBranches, msgBaseSlot: UInt32
    var pad1: UInt32 = 0
    var arIdx: (UInt32, UInt32, UInt32, UInt32)
    var downElemOff: (UInt32, UInt32, UInt32, UInt32)
    var forgetElemOff: (UInt32, UInt32, UInt32, UInt32)
    var gainElemOff: (UInt32, UInt32, UInt32, UInt32)
    var normElemOff: (UInt32, UInt32, UInt32, UInt32)
}

struct Ink2AttnResPackPairBatchP {
    var T, H, rank, thStride: UInt32
    var nBlocks, blockStride, partialIsF32, normalizeValues: UInt32
    var arEps, rmsEps: Float
    var numBranches, msgBaseSlot: UInt32
    var arIdx: (UInt32, UInt32, UInt32, UInt32)
    var slotIds: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
}

extension QWInkling2Forward {
    func initialXTarget(ink: QWModelDescriptor.Inkling2Config, xBytes: Int) -> (any MTLBuffer, Int) {
        guard let firstL = session.executionPlan.first else { return (session.xBuf, 0) }
        if firstL == ink.loopCoreStart, firstL == 0 { return (session.preludeBuf, 0) }
        if firstL != ink.loopCoreStart, ink.isAttnResBlockBoundary(firstL),
           (xBytes & 3) == 0, xBytes <= session.blocksRingBuf.length {
            return (session.blocksRingBuf, 0)
        }
        return (session.xBuf, 0)
    }

    func nextXTarget(
        nextVisit: Int, blocksSlots: [Int], ink: QWModelDescriptor.Inkling2Config, xBytes: Int
    ) -> (any MTLBuffer, Int) {
        guard nextVisit < session.executionPlan.count else { return (session.xBuf, 0) }
        let nextL = session.executionPlan[nextVisit]
        if nextL == ink.loopCoreStart {
            return (nextVisit == nextL ? session.preludeBuf : session.xBuf, 0)
        }
        if ink.isAttnResBlockBoundary(nextL), blocksSlots.count < 6,
           (xBytes & 3) == 0, (blocksSlots.count + 1) * xBytes <= session.blocksRingBuf.length {
            return (session.blocksRingBuf, blocksSlots.count * xBytes)
        }
        return (session.xBuf, 0)
    }

    func encodeCopyBF16(
        _ enc: any MTLComputeCommandEncoder,
        src: any MTLBuffer, srcOff: Int = 0,
        dst: any MTLBuffer, dstOff: Int = 0,
        count: Int
    ) {
        guard count > 0 else { return }
        var cntU = UInt32(count)
        enc.setComputePipelineState(pCopyBF16)
        enc.setBuffer(src, offset: srcOff, index: 0)
        enc.setBuffer(dst, offset: dstOff, index: 1)
        enc.setBytes(&cntU, length: 4, index: 2)
        let total = (count + 7) / 8
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, total), height: 1, depth: 1))
    }

    func encodeAttnResCtrlNorm0(
        _ enc: any MTLComputeCommandEncoder,
        t: Int, H: Int, thStride: Int,
        blocksSlots: [Int],
        partialBF16: any MTLBuffer, partialBF16Off: Int = 0,
        partialF32: any MTLBuffer, partialIsF32: Bool,
        query: Ink2Slice, numQ: Int, nWeights: [Ink2Slice], eps: Float,
        ink: QWModelDescriptor.Inkling2Config
    ) {
        let numOut = nWeights.count
        let canFused = blocksSlots.count <= 7 && H <= 4096
            && (1...4).contains(numQ) && (1...4).contains(numOut)
            && nWeights.allSatisfy { $0.buffer === nWeights[0].buffer && ($0.offset & 1) == 0 }
        if canFused {
            let noff: [UInt32] = (0..<4).map { $0 < numOut ? UInt32(nWeights[$0].offset / 2) : 0 }
            let z4: (UInt32, UInt32, UInt32, UInt32) = (0, 0, 0, 0)
            var p = Ink2AttnResCtrlNormP(
                T: UInt32(t), H: UInt32(H), nBlocks: UInt32(blocksSlots.count),
                numQ: UInt32(numQ), numOut: UInt32(numOut), blockStride: UInt32(thStride),
                partialIsF32: partialIsF32 ? 1 : 0,
                normalizeValues: ink.attnResNormalizeValues ? 1 : 0,
                useCtrl: 0, rank: 0, rotateOut: 0, outIsF16: 0,
                outStride: UInt32(thStride), msgStride: UInt32(thStride),
                attnResEps: 1e-6, rmsEps: eps,
                slotIds: slotTuple(blocksSlots),
                normWOffsets: (noff[0], noff[1], noff[2], noff[3]),
                ctrlDownOffsets: z4, ctrlForgetOffsets: z4, ctrlGainOffsets: z4
            )
            enc.setComputePipelineState(pAttnResCtrlNorm)
            enc.setBuffer(session.blocksRingBuf, offset: 0, index: 0)
            enc.setBuffer(partialBF16, offset: partialBF16Off, index: 1)
            enc.setBuffer(partialF32, offset: 0, index: 2)
            enc.setBuffer(query.buffer, offset: query.offset, index: 3)
            enc.setBuffer(session.dummyF, offset: 0, index: 4)
            enc.setBuffer(session.dummyB, offset: 0, index: 5)
            enc.setBuffer(session.dummyB, offset: 0, index: 6)
            enc.setBuffer(session.dummyB, offset: 0, index: 7)
            enc.setBuffer(nWeights[0].buffer, offset: 0, index: 8)
            enc.setBuffer(session.normBuf, offset: 0, index: 9)
            enc.setBytes(&p, length: MemoryLayout<Ink2AttnResCtrlNormP>.stride, index: 10)
            // Kernel iterates outIdx internally and derives t from tg.y (or tg.x when t == 1).
            enc.dispatchThreadgroups(MTLSize(width: 1, height: t, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            return
        }

        var arp = AttnResP(
            T: UInt32(t), H: UInt32(H), nBlocks: UInt32(blocksSlots.count),
            numQ: UInt32(numQ), blockStride: UInt32(thStride),
            partialIsF32: partialIsF32 ? 1 : 0, normalizeValues: ink.attnResNormalizeValues ? 1 : 0,
            eps: 1e-6, slots: slotTuple(blocksSlots)
        )
        enc.setComputePipelineState(pAttnRes)
        enc.setBuffer(session.blocksRingBuf, offset: 0, index: 0)
        enc.setBuffer(partialBF16, offset: partialBF16Off, index: 1)
        enc.setBuffer(partialF32, offset: 0, index: 2)
        enc.setBuffer(query.buffer, offset: query.offset, index: 3)
        enc.setBuffer(session.attnResOutBuf, offset: 0, index: 4)
        enc.setBytes(&arp, length: MemoryLayout<AttnResP>.stride, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: numQ, height: t, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        let rowBytes = thStride * 2
        for i in 0..<numOut {
            let arOff = min(i, numQ - 1) * thStride * 4
            encodeNorm(enc, inBF16: session.dummyB, inBF16Off: 0,
                       inF32: session.attnResOutBuf, inF32Off: arOff, inIsF32: true,
                       w: nWeights[i], outBF16: session.normBuf, outOff: i * rowBytes,
                       rows: t, dim: H, eps: eps)
        }
    }

    func encodeControllerBranch(
        _ enc: any MTLComputeCommandEncoder,
        xOff: Int, msgOff: Int, ctrl: QWInkling2Weights.ControllerW,
        normW: Ink2Slice, outNOff: Int, ctrlOutOff: Int,
        t: Int, H: Int, rank: Int, eps: Float
    ) throws {
        var cp = CtrlP(T: UInt32(t), H: UInt32(H), rank: UInt32(rank), eps: eps)
        enc.setComputePipelineState(pCtrlPackPair)
        enc.setBuffer(session.attnResOutBuf, offset: xOff, index: 0)
        enc.setBuffer(session.msgOutBuf, offset: msgOff, index: 1)
        enc.setBuffer(session.ctrlPairBuf, offset: 0, index: 2)
        enc.setBytes(&cp, length: MemoryLayout<CtrlP>.stride, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: t, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        try encodeLinearF32(enc, weight: ctrl.downW, xBuf: session.ctrlPairBuf, xOff: 0,
                            yF32Buf: session.ctrlHidBuf, yOff: 0, m: t)
        if t <= 16, let forget = ctrl.forget, let gain = ctrl.gain {
            enc.setComputePipelineState(pCtrlApply)
            enc.setBuffer(session.attnResOutBuf, offset: xOff, index: 0)
            enc.setBuffer(session.msgOutBuf, offset: msgOff, index: 1)
            enc.setBuffer(session.ctrlHidBuf, offset: 0, index: 2)
            enc.setBuffer(forget.buffer, offset: forget.offset, index: 3)
            enc.setBuffer(gain.buffer, offset: gain.offset, index: 4)
            enc.setBuffer(session.ctrlOutBuf, offset: ctrlOutOff, index: 5)
            enc.setBytes(&cp, length: MemoryLayout<CtrlP>.stride, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: (H + 31) / 32, height: t, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encodeNorm(enc, inBF16: session.dummyB, inBF16Off: 0,
                       inF32: session.ctrlOutBuf, inF32Off: ctrlOutOff,
                       inIsF32: true, w: normW, outBF16: session.normBuf, outOff: outNOff,
                       rows: t, dim: H, eps: eps)
        } else {
            try encodeLinearF32(enc, weight: ctrl.forgetGainW, xBuf: session.ctrlHidBuf, xOff: 0,
                                yF32Buf: session.ctrlOutBuf, yOff: 0, m: t, xIsF32: true)
            enc.setComputePipelineState(pCtrlBlendNorm)
            enc.setBuffer(session.attnResOutBuf, offset: xOff, index: 0)
            enc.setBuffer(session.msgOutBuf, offset: msgOff, index: 1)
            enc.setBuffer(session.ctrlOutBuf, offset: 0, index: 2)
            enc.setBuffer(normW.buffer, offset: normW.offset, index: 3)
            enc.setBuffer(session.normBuf, offset: outNOff, index: 4)
            enc.setBytes(&cp, length: MemoryLayout<CtrlP>.stride, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: t, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    func encodeControllerBranchesBatch(
        _ enc: any MTLComputeCommandEncoder,
        branches: [(arSlot: Int, msgSlot: Int, ctrl: QWInkling2Weights.ControllerW, normW: Ink2Slice)],
        msgBaseSlot: Int,
        t: Int, H: Int, rank: Int, eps: Float,
        attnResInputs: (blocksSlots: [Int], partialBF16: any MTLBuffer, partialBF16Off: Int,
                        partialF32: any MTLBuffer, partialIsF32: Bool, query: Ink2Slice,
                        numQ: Int, normalizeValues: Bool)? = nil
    ) throws {
        func encodeSeparateAttnRes() {
            guard let ar = attnResInputs else { return }
            var arp = AttnResP(
                T: UInt32(t), H: UInt32(H), nBlocks: UInt32(ar.blocksSlots.count),
                numQ: UInt32(ar.numQ), blockStride: UInt32(t * H),
                partialIsF32: ar.partialIsF32 ? 1 : 0,
                normalizeValues: ar.normalizeValues ? 1 : 0,
                eps: 1e-6, slots: slotTuple(ar.blocksSlots)
            )
            enc.setComputePipelineState(pAttnRes)
            enc.setBuffer(session.blocksRingBuf, offset: 0, index: 0)
            enc.setBuffer(ar.partialBF16, offset: ar.partialBF16Off, index: 1)
            enc.setBuffer(ar.partialF32, offset: 0, index: 2)
            enc.setBuffer(ar.query.buffer, offset: ar.query.offset, index: 3)
            enc.setBuffer(session.attnResOutBuf, offset: 0, index: 4)
            enc.setBytes(&arp, length: MemoryLayout<AttnResP>.stride, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: ar.numQ, height: t, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        let nb = branches.count
        guard (1...3).contains(nb),
              branches.enumerated().allSatisfy({ $0.element.msgSlot == msgBaseSlot + $0.offset }),
              let down0 = branches[0].ctrl.down,
              let fg0 = branches[0].ctrl.forget,
              branches.allSatisfy({
                  guard let d = $0.ctrl.down, let f = $0.ctrl.forget, let g = $0.ctrl.gain else { return false }
                  return d.buffer === down0.buffer && (d.offset & 1) == 0
                      && f.buffer === fg0.buffer && (f.offset & 1) == 0
                      && g.buffer === fg0.buffer && (g.offset & 1) == 0
                      && $0.normW.buffer === branches[0].normW.buffer && ($0.normW.offset & 1) == 0
              }) else {
            encodeSeparateAttnRes()
            let thStride = t * H
            for (i, b) in branches.enumerated() {
                try encodeControllerBranch(
                    enc, xOff: b.arSlot * thStride * 4, msgOff: b.msgSlot * thStride * 4,
                    ctrl: b.ctrl, normW: b.normW, outNOff: i * thStride * 2,
                    ctrlOutOff: i * thStride * 4, t: t, H: H, rank: rank, eps: eps
                )
            }
            return
        }

        func u4(_ fn: (Int) -> UInt32) -> (UInt32, UInt32, UInt32, UInt32) {
            (0 < nb ? fn(0) : 0, 1 < nb ? fn(1) : 0, 2 < nb ? fn(2) : 0, 0)
        }
        var bp = Ink2CtrlBatchP(
            T: UInt32(t), H: UInt32(H), rank: UInt32(rank), thStride: UInt32(t * H),
            eps: eps, numBranches: UInt32(nb), msgBaseSlot: UInt32(msgBaseSlot),
            arIdx: u4 { UInt32(branches[$0].arSlot) },
            downElemOff: u4 { UInt32(branches[$0].ctrl.down!.offset / 2) },
            forgetElemOff: u4 { UInt32(branches[$0].ctrl.forget!.offset / 2) },
            gainElemOff: u4 { UInt32(branches[$0].ctrl.gain!.offset / 2) },
            normElemOff: u4 { UInt32(branches[$0].normW.offset / 2) }
        )
        let nbU = nb
        if let ar = attnResInputs, ar.blocksSlots.count <= 7, H <= 4096 {
            var fp = Ink2AttnResPackPairBatchP(
                T: UInt32(t), H: UInt32(H), rank: UInt32(rank), thStride: UInt32(t * H),
                nBlocks: UInt32(ar.blocksSlots.count), blockStride: UInt32(t * H),
                partialIsF32: ar.partialIsF32 ? 1 : 0, normalizeValues: ar.normalizeValues ? 1 : 0,
                arEps: 1e-6, rmsEps: eps, numBranches: UInt32(nb), msgBaseSlot: UInt32(msgBaseSlot),
                arIdx: bp.arIdx, slotIds: slotTuple(ar.blocksSlots)
            )
            enc.setComputePipelineState(pAttnResPackPairBatch)
            enc.setBuffer(session.blocksRingBuf, offset: 0, index: 0)
            enc.setBuffer(ar.partialBF16, offset: ar.partialBF16Off, index: 1)
            enc.setBuffer(ar.partialF32, offset: 0, index: 2)
            enc.setBuffer(ar.query.buffer, offset: ar.query.offset, index: 3)
            enc.setBuffer(session.msgOutBuf, offset: 0, index: 4)
            enc.setBuffer(session.attnResOutBuf, offset: 0, index: 5)
            enc.setBuffer(session.ctrlPairBuf, offset: 0, index: 6)
            enc.setBytes(&fp, length: MemoryLayout<Ink2AttnResPackPairBatchP>.stride, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: t, height: nbU, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        } else {
            encodeSeparateAttnRes()
            enc.setComputePipelineState(pCtrlPackPairBatch)
            enc.setBuffer(session.attnResOutBuf, offset: 0, index: 0)
            enc.setBuffer(session.msgOutBuf, offset: 0, index: 1)
            enc.setBuffer(session.ctrlPairBuf, offset: 0, index: 2)
            enc.setBytes(&bp, length: MemoryLayout<Ink2CtrlBatchP>.stride, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: t, height: nbU, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }

        if t <= 16 {
            enc.setComputePipelineState(pCtrlDownGemvBatch)
            enc.setBuffer(session.ctrlPairBuf, offset: 0, index: 0)
            enc.setBuffer(down0.buffer, offset: 0, index: 1)
            enc.setBuffer(session.ctrlHidBuf, offset: 0, index: 2)
            enc.setBytes(&bp, length: MemoryLayout<Ink2CtrlBatchP>.stride, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: (rank + 3) / 4, height: nbU, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

            let canFastApply64 = (rank == 64)
                && branches.allSatisfy { ($0.ctrl.forget!.offset & 7) == 0 && ($0.ctrl.gain!.offset & 7) == 0 }
            enc.setComputePipelineState(pCtrlApplyBatch)
            enc.setBuffer(session.attnResOutBuf, offset: 0, index: 0)
            enc.setBuffer(session.msgOutBuf, offset: 0, index: 1)
            enc.setBuffer(session.ctrlHidBuf, offset: 0, index: 2)
            enc.setBuffer(fg0.buffer, offset: 0, index: 3)
            enc.setBuffer(session.ctrlOutBuf, offset: 0, index: 4)
            enc.setBytes(&bp, length: MemoryLayout<Ink2CtrlBatchP>.stride, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: canFastApply64 ? ((H + 63) / 64) : ((H + 31) / 32), height: t, depth: nbU),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )

            enc.setComputePipelineState(pCtrlRmsNormBatch)
            enc.setBuffer(session.ctrlOutBuf, offset: 0, index: 0)
            enc.setBuffer(branches[0].normW.buffer, offset: 0, index: 1)
            enc.setBuffer(session.normBuf, offset: 0, index: 2)
            enc.setBytes(&bp, length: MemoryLayout<Ink2CtrlBatchP>.stride, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: t, height: nbU, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        } else if let pDownNax = pCtrlDownNaxBatch, let pFgBlendNax = pCtrlFgBlendNaxBatch,
                  hal.caps.isMetal4TensorCapable {
            enc.setComputePipelineState(pDownNax)
            enc.setBuffer(session.ctrlPairBuf, offset: 0, index: 0)
            enc.setBuffer(down0.buffer, offset: 0, index: 1)
            enc.setBuffer(session.ctrlHidBuf, offset: 0, index: 2)
            enc.setBytes(&bp, length: MemoryLayout<Ink2CtrlBatchP>.stride, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: (rank + 63) / 64, height: (t + 63) / 64, depth: nbU),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

            enc.setComputePipelineState(pFgBlendNax)
            enc.setBuffer(session.attnResOutBuf, offset: 0, index: 0)
            enc.setBuffer(session.msgOutBuf, offset: 0, index: 1)
            enc.setBuffer(session.ctrlHidBuf, offset: 0, index: 2)
            enc.setBuffer(fg0.buffer, offset: 0, index: 3)
            enc.setBuffer(session.ctrlOutBuf, offset: 0, index: 4)
            enc.setBytes(&bp, length: MemoryLayout<Ink2CtrlBatchP>.stride, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: (H + 31) / 32, height: (t + 63) / 64, depth: nbU),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

            enc.setComputePipelineState(pCtrlRmsNormBatch)
            enc.setBuffer(session.ctrlOutBuf, offset: 0, index: 0)
            enc.setBuffer(branches[0].normW.buffer, offset: 0, index: 1)
            enc.setBuffer(session.normBuf, offset: 0, index: 2)
            enc.setBytes(&bp, length: MemoryLayout<Ink2CtrlBatchP>.stride, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: t, height: nbU, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        } else {
            for (i, b) in branches.enumerated() {
                try encodeLinearF32(
                    enc, weight: b.ctrl.downW,
                    xBuf: session.ctrlPairBuf, xOff: i * t * 2 * H * 2,
                    yF32Buf: session.ctrlHidBuf, yOff: i * t * rank * 4, m: t
                )
            }
            for (i, b) in branches.enumerated() {
                try encodeLinearF32(
                    enc, weight: b.ctrl.forgetGainW,
                    xBuf: session.ctrlHidBuf, xOff: i * t * rank * 4,
                    yF32Buf: session.ctrlOutBuf, yOff: i * t * 2 * H * 4, m: t, xIsF32: true
                )
            }
            enc.setComputePipelineState(pCtrlBlendNormBatch)
            enc.setBuffer(session.attnResOutBuf, offset: 0, index: 0)
            enc.setBuffer(session.msgOutBuf, offset: 0, index: 1)
            enc.setBuffer(session.ctrlOutBuf, offset: 0, index: 2)
            enc.setBuffer(branches[0].normW.buffer, offset: 0, index: 3)
            enc.setBuffer(session.normBuf, offset: 0, index: 4)
            enc.setBytes(&bp, length: MemoryLayout<Ink2CtrlBatchP>.stride, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: t, height: nbU, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    func encodeFinalNormAndHead(
        t: Int, H: Int, thStride: Int, xBytes: Int, eps: Float,
        blocksSlots: [Int], partialBF16: any MTLBuffer, partialBF16Off: Int,
        captureTapBuf: (any MTLBuffer)?, finalTapSlot: Int?,
        runHeadRows: Int, greedyHead: Bool,
        beginEncoder: (Ink2Stage?) throws -> any MTLComputeCommandEncoder
    ) throws {
        let ef = try beginEncoder(.loopResCtrlNorm)
        encodeAttnResCtrlNorm0(
            ef, t: t, H: H, thStride: thStride, blocksSlots: blocksSlots,
            partialBF16: partialBF16, partialBF16Off: partialBF16Off,
            partialF32: session.dummyF, partialIsF32: false,
            query: weights.finalInputQuery, numQ: 1, nWeights: [weights.finalNorm],
            eps: eps, ink: session.ink2
        )
        if let tapBuf = captureTapBuf, let slot = finalTapSlot {
            encodeCopyBF16(ef, src: session.normBuf, srcOff: 0,
                           dst: tapBuf, dstOff: slot * xBytes, count: thStride)
        }
        if runHeadRows > 0 {
            let efHead = try beginEncoder(.finalHead)
            let headStartRow = max(0, t - runHeadRows)
            try encodeHeadProjection(
                efHead, headStartRow: headStartRow, runHeadRows: runHeadRows, greedyHead: greedyHead
            )
        }
    }

    private func encodeHeadProjection(
        _ efHead: any MTLComputeCommandEncoder,
        headStartRow: Int,
        runHeadRows: Int,
        greedyHead: Bool
    ) throws {
        let H = session.descriptor.hidden
        let V = session.ink2.unpaddedVocabSize
        let cap = session.ink2.finalLogitSoftcapping
        switch weights.lmHeadStorage {
        case .bf16(let headTV):
            if greedyHead {
                let xHeadTV = QWTensorValue(
                    region: QWMRegion(buffer: session.normBuf, base: 0, length: session.normBuf.length),
                    offset: headStartRow * H * 2,
                    shape: QWTensorShape([runHeadRows * H], .bf16)
                )
                try QWHead.encodeBatchedGreedyArgmax(
                    efHead, hal: hal,
                    storage: .bf16(headTV),
                    x: xHeadTV,
                    keysHi: session.headKeysHiTV,
                    keysLo: session.headKeysLoTV,
                    tokens: session.headTokensTV,
                    geometry: session.geometry,
                    M: runHeadRows, N: V, softcap: cap
                )
            } else {
                guard let headBank = weights.lmHeadBank else {
                    throw QWInkling2Error.badState("missing BF16 lmHeadBank")
                }
                try encodeFPGemm(
                    efHead, bank: headBank,
                    xBuf: session.normBuf, xOff: headStartRow * H * 2,
                    yBuf: session.logitsF32Buf, m: runHeadRows,
                    outIsF32: true, softcap: cap
                )
            }
        case .q4g(let hw):
            encodeRotateToF16(
                efHead, srcBF16: session.normBuf, srcOff: headStartRow * H * 2,
                dst: session.qwpF16InBuf, count: runHeadRows * H,
                rotate: hw.view.isRotated, outIsF32: false
            )
            let x16HeadTV = session.qwpF16InTV.slice(byteOffset: 0, count: runHeadRows * H, dtype: .f16)
            if greedyHead {
                try QWHead.encodeBatchedGreedyArgmax(
                    efHead, hal: hal,
                    storage: weights.lmHeadStorage,
                    x: x16HeadTV,
                    keysHi: session.headKeysHiTV,
                    keysLo: session.headKeysLoTV,
                    tokens: session.headTokensTV,
                    geometry: session.geometry,
                    M: runHeadRows, N: V, softcap: cap
                )
            } else {
                let logitsTV = session.logitsF32TV.slice(byteOffset: 0, count: runHeadRows * V, dtype: .f32)
                if runHeadRows == 1 {
                    try QWHead.encodeQ4g(efHead, hal: hal, weight: hw, x: x16HeadTV, y: logitsTV)
                } else {
                    try QWHead.encodeBatched(efHead, hal: hal, view: hw.view, x: x16HeadTV, y: logitsTV, m: runHeadRows)
                }
                if cap > 0 {
                    encodeSoftcapF32(efHead, data: session.logitsF32Buf, count: runHeadRows * V, softcap: cap)
                }
            }
        case .f16:
            throw QWInkling2Error.badState("unsupported .f16 lmHeadStorage for Inkling-2")
        }
    }

    /// Recomputes final head logits/argmax from a cached final-norm row on exact prompt cache hits.
    func runHeadFromFinalNorm(
        finalNormRow: Data,
        greedyHead: Bool,
        profiler: QWInkling2Profiler? = nil
    ) throws {
        let rowBytes = session.descriptor.hidden * 2
        guard finalNormRow.count == rowBytes else {
            throw QWInkling2Error.badState("invalid cached finalNormRow bytes=\(finalNormRow.count) expected=\(rowBytes)")
        }
        let dst = session.normBuf.contents()
        withExtendedLifetime(session.normBuf) {
            finalNormRow.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                dst.copyMemory(from: base, byteCount: rowBytes)
            }
        }
        let hostStartNs = profiler?.beginHostTimer() ?? 0
        let cbOpt = (profiler == nil)
            ? (hal.queue.makeCommandBufferWithUnretainedReferences() ?? hal.queue.makeCommandBuffer())
            : hal.queue.makeCommandBuffer()
        guard let cb = cbOpt else {
            throw QWInkling2Error.badState("failed to create head command buffer for exact cache hit")
        }
        var encOpt: (any MTLComputeCommandEncoder)? = nil
        profiler?.beginStage(.finalHead, on: cb, encoder: &encOpt, hal: hal)
        if encOpt == nil {
            guard let plain = cb.makeComputeCommandEncoder() else {
                throw QWInkling2Error.badState("failed to create compute encoder for exact cache hit")
            }
            hal.touchKeepAlive(plain)
            encOpt = plain
        }
        if let activeEnc = encOpt {
            try encodeHeadProjection(activeEnc, headStartRow: 0, runHeadRows: 1, greedyHead: greedyHead)
        }
        profiler?.endActiveStage(encoder: &encOpt)
        encOpt?.endEncoding()
        profiler?.endHostTimer(hostStartNs)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw QWInkling2Error.badState("Metal execution failed on exact cache hit: \(err.localizedDescription)")
        }
        profiler?.finishCommandBuffer(cb, tokens: 1)
    }

    func encodeLoopCoreTransition(
        visit: Int,
        lIdx: Int,
        t: Int,
        H: Int,
        thStride: Int,
        eps: Float,
        ink: QWModelDescriptor.Inkling2Config,
        curXBuf: inout any MTLBuffer,
        curXOff: inout Int,
        blocksSlots: inout [Int],
        preludeBlocksCount: inout Int,
        lastBankStateCount: inout Int,
        bankValid: inout Bool,
        beginEncoder: (Ink2Stage) throws -> any MTLComputeCommandEncoder
    ) throws {
        guard lIdx == ink.loopCoreStart else { return }
        if visit == lIdx {
            if !(curXBuf === session.preludeBuf && curXOff == 0) {
                let cEnc = try beginEncoder(.loopResCtrlNorm)
                encodeCopyBF16(cEnc, src: curXBuf, srcOff: curXOff,
                               dst: session.preludeBuf, dstOff: 0, count: thStride)
                curXBuf = session.preludeBuf
                curXOff = 0
            }
            preludeBlocksCount = blocksSlots.count
        } else {
            let tailSlots = Array(blocksSlots.dropFirst(preludeBlocksCount + 1))
            let nStates = tailSlots.count + 1
            lastBankStateCount = nStates
            var bp = BankP(T: UInt32(t), H: UInt32(H), nStates: UInt32(nStates), numQ: 4,
                           stateStride: UInt32(thStride), eps: eps, slots: slotTuple(tailSlots))
            let enc = try beginEncoder(.loopResCtrlNorm)
            enc.setComputePipelineState(pBankNorm)
            enc.setBuffer(session.blocksRingBuf, offset: 0, index: 0)
            enc.setBuffer(curXBuf, offset: curXOff, index: 1)
            enc.setBuffer(session.bankBuf, offset: 0, index: 2)
            enc.setBytes(&bp, length: MemoryLayout<BankP>.stride, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: nStates, height: t, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            bankValid = true
            curXBuf = session.preludeBuf
            curXOff = 0
            blocksSlots = Array(blocksSlots.prefix(preludeBlocksCount))
        }
    }

    func encodeLoopBankAttend(
        _ enc: any MTLComputeCommandEncoder,
        lIdx: Int,
        t: Int,
        H: Int,
        thStride: Int,
        lastBankStateCount: Int,
        eps: Float,
        ink: QWModelDescriptor.Inkling2Config
    ) {
        let mqOff = weights.messageQueries.offset + (lIdx - ink.loopCoreStart) * 4 * H * 4
        var bp = BankP(T: UInt32(t), H: UInt32(H), nStates: UInt32(lastBankStateCount), numQ: 4,
                       stateStride: UInt32(thStride), eps: eps, slots: slotTuple([]))
        enc.setComputePipelineState(pBankAttend)
        enc.setBuffer(session.bankBuf, offset: 0, index: 0)
        enc.setBuffer(weights.messageQueries.buffer, offset: mqOff, index: 1)
        enc.setBuffer(session.msgOutBuf, offset: 0, index: 2)
        enc.setBytes(&bp, length: MemoryLayout<BankP>.stride, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: 4, height: t, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func encodePreAttnCtrlNorm(
        _ enc: any MTLComputeCommandEncoder,
        lIdx: Int,
        lw: QWInkling2Weights.LayerWeights,
        useBank: Bool,
        t: Int,
        H: Int,
        thStride: Int,
        eps: Float,
        ink: QWModelDescriptor.Inkling2Config,
        blocksSlots: [Int],
        curXBuf: any MTLBuffer,
        curXOff: Int
    ) throws -> (kNormOff: Int, vNormOff: Int) {
        let normBytes = thStride * 2
        var kNormOff = 0, vNormOff = 0
        if lIdx == 0, case .sliding(let inNorm) = lw.attnNorm {
            encodeNorm(enc, inBF16: curXBuf, inBF16Off: curXOff, inF32: session.dummyF, inF32Off: 0,
                       inIsF32: false, w: inNorm, outBF16: session.normBuf, outOff: 0,
                       rows: t, dim: H, eps: eps)
        } else if let aQ = lw.attnInputQuery {
            let numQ = lw.isSliding ? 1 : 3
            if !useBank || lw.controllers == nil {
                let nWeights: [Ink2Slice] = switch lw.attnNorm {
                case .sliding(let w): lw.isSliding ? [w] : [w, w, w]
                case .global(let qW, let kW, let vW): [qW, kW, vW]
                }
                if nWeights.count == 3 {
                    kNormOff = normBytes
                    vNormOff = 2 * normBytes
                }
                encodeAttnResCtrlNorm0(
                    enc, t: t, H: H, thStride: thStride, blocksSlots: blocksSlots,
                    partialBF16: curXBuf, partialBF16Off: curXOff,
                    partialF32: session.dummyF, partialIsF32: false,
                    query: aQ, numQ: numQ, nWeights: nWeights, eps: eps, ink: ink
                )
            } else if let ctrlSet = lw.controllers {
                kNormOff = normBytes
                vNormOff = 2 * normBytes
                let nWeights: [Ink2Slice] = switch lw.attnNorm {
                case .sliding(let w): [w, w, w]
                case .global(let qW, let kW, let vW): [qW, kW, vW]
                }
                let cs = [ctrlSet.attn, ctrlSet.k, ctrlSet.v]
                try encodeControllerBranchesBatch(
                    enc,
                    branches: (0..<3).map { i in
                        (arSlot: min(i, numQ - 1), msgSlot: i, ctrl: cs[i], normW: nWeights[i])
                    },
                    msgBaseSlot: 0, t: t, H: H, rank: ink.loopGateRank, eps: eps,
                    attnResInputs: (
                        blocksSlots: blocksSlots, partialBF16: curXBuf, partialBF16Off: curXOff,
                        partialF32: session.dummyF, partialIsF32: false,
                        query: aQ, numQ: numQ, normalizeValues: ink.attnResNormalizeValues
                    )
                )
            }
        }
        return (kNormOff, vNormOff)
    }

    func encodePostAttnCtrlNorm(
        _ enc: any MTLComputeCommandEncoder,
        lw: QWInkling2Weights.LayerWeights,
        useBank: Bool,
        t: Int,
        H: Int,
        thStride: Int,
        eps: Float,
        ink: QWModelDescriptor.Inkling2Config,
        blocksSlots: [Int]
    ) throws {
        if !useBank || lw.controllers == nil {
            encodeAttnResCtrlNorm0(
                enc, t: t, H: H, thStride: thStride, blocksSlots: blocksSlots,
                partialBF16: session.dummyB, partialBF16Off: 0,
                partialF32: session.resF32Buf, partialIsF32: true,
                query: lw.mlpInputQuery, numQ: 1, nWeights: [lw.postAttnNorm],
                eps: eps, ink: ink
            )
        } else if let mCtrl = lw.controllers?.mlp {
            try encodeControllerBranchesBatch(
                enc,
                branches: [(arSlot: 0, msgSlot: 3, ctrl: mCtrl, normW: lw.postAttnNorm)],
                msgBaseSlot: 3, t: t, H: H, rank: ink.loopGateRank, eps: eps,
                attnResInputs: (
                    blocksSlots: blocksSlots, partialBF16: session.dummyB, partialBF16Off: 0,
                    partialF32: session.resF32Buf, partialIsF32: true,
                    query: lw.mlpInputQuery, numQ: 1, normalizeValues: ink.attnResNormalizeValues
                )
            )
        }
    }
}
