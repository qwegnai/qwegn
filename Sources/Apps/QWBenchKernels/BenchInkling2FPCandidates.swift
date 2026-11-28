// BenchInkling2FPCandidates.swift — Candidate Wave 1-3 kernel encoders for --only-ink2-fp.
import Foundation
import Metal
import QWHAL

struct Ink2MoENaxP {
    var experts, n, k, yStride, yColOff, inDtype, outDtype, flags: UInt32
}

struct Ink2MoECompactP {
    var n, k, topK, numExperts, wExpertRows, upRowBias, inDtype, outDtype, flags: UInt32
    var gateCap, upCap: Float
}

struct Ink2SwiGLUP {
    var m, i: UInt32
    var gateCap, upCap: Float
    var flags: UInt32
}

struct Ink2MoECombineP {
    var hidden, topK, tokens, inDtype, outDtype, sharedDtype, flags: UInt32
}

enum Ink2FPCandidates {
    final class MoECandidateBuffers {
        let sortedTok: any MTLBuffer
        let invPerm: any MTLBuffer
        let activeExperts: any MTLBuffer
        let activeArgs: any MTLBuffer
        let tileE: any MTLBuffer
        let tileR: any MTLBuffer
        let tileCount: any MTLBuffer
        let prefillGU: any MTLBuffer
        let isMXFP8: Bool
        let eCount: Int
        let h: Int
        let inter: Int
        let topK: Int
        let gateCap: Float
        let upCap: Float
        let gw: any MTLBuffer, gSc: any MTLBuffer, gS2: any MTLBuffer
        let uw: any MTLBuffer, uSc: any MTLBuffer, uS2: any MTLBuffer
        let dw: any MTLBuffer, dSc: any MTLBuffer, dS2: any MTLBuffer
        let xBuf: any MTLBuffer, topIdsBuf: any MTLBuffer, topWBuf: any MTLBuffer
        let offsetsBuf: any MTLBuffer, sortedBuf: any MTLBuffer
        let moeActBuf: any MTLBuffer, slotOutBuf: any MTLBuffer, outBuf: any MTLBuffer

        init(
            dev: any MTLDevice, isMXFP8: Bool, eCount: Int, h: Int, inter: Int, topK: Int, maxT: Int,
            gateCap: Float, upCap: Float,
            gw: any MTLBuffer, gSc: any MTLBuffer, gS2: any MTLBuffer,
            uw: any MTLBuffer, uSc: any MTLBuffer, uS2: any MTLBuffer,
            dw: any MTLBuffer, dSc: any MTLBuffer, dS2: any MTLBuffer,
            xBuf: any MTLBuffer, topIdsBuf: any MTLBuffer, topWBuf: any MTLBuffer,
            offsetsBuf: any MTLBuffer, sortedBuf: any MTLBuffer,
            moeActBuf: any MTLBuffer, slotOutBuf: any MTLBuffer, outBuf: any MTLBuffer
        ) throws {
            let maxPairs = maxT * topK
            let maxTiles = min(maxPairs, min(eCount, maxPairs) + (maxPairs + 63) / 64)
            let guSlices = max(1, (maxTiles + max(1, 2560 / ((2 * inter + 63) / 64)) - 1) / max(1, 2560 / ((2 * inter + 63) / 64)))
            let dnSlices = max(1, (maxTiles + max(1, 2560 / ((h + 63) / 64)) - 1) / max(1, 2560 / ((h + 63) / 64)))
            self.sortedTok = try BenchRunner.buffer(dev, bytes: maxPairs * 4)
            self.invPerm = try BenchRunner.buffer(dev, bytes: maxPairs * 4)
            self.activeExperts = try BenchRunner.buffer(dev, bytes: eCount * 4)
            self.activeArgs = try BenchRunner.buffer(dev, bytes: 48)
            self.tileE = try BenchRunner.buffer(dev, bytes: max(1, maxTiles) * 4)
            self.tileR = try BenchRunner.buffer(dev, bytes: max(1, maxTiles) * 4)
            self.tileCount = try BenchRunner.buffer(dev, bytes: (4 + (guSlices + dnSlices) * 3) * 4)
            self.prefillGU = try BenchRunner.buffer(dev, bytes: maxPairs * 2 * inter * 2)
            self.isMXFP8 = isMXFP8; self.eCount = eCount; self.h = h; self.inter = inter; self.topK = topK
            self.gateCap = gateCap; self.upCap = upCap
            self.gw = gw; self.gSc = gSc; self.gS2 = gS2
            self.uw = uw; self.uSc = uSc; self.uS2 = uS2
            self.dw = dw; self.dSc = dSc; self.dS2 = dS2
            self.xBuf = xBuf; self.topIdsBuf = topIdsBuf; self.topWBuf = topWBuf
            self.offsetsBuf = offsetsBuf; self.sortedBuf = sortedBuf
            self.moeActBuf = moeActBuf; self.slotOutBuf = slotOutBuf; self.outBuf = outBuf
        }
    }

    private static func resolve(_ hal: QWHal, _ name: String) -> (any MTLComputePipelineState)? {
        guard hal.metallib.functionNames.contains(name) else { return nil }
        return try? hal.pipeline(name)
    }

    static func makeMoEEncoder(
        hal: QWHal, bufs: MoECandidateBuffers, t: Int, forceNAX: Bool? = nil
    ) -> (candName: String, encode: ((any MTLCommandBuffer, inout any MTLComputeCommandEncoder) -> Void)?) {
        let isPrefill = forceNAX ?? (t > 8)
        let useNAX = hal.caps.isMetal4TensorCapable
        let prefillKind = useNAX ? "nax" : "sg"
        let candName = isPrefill
            ? (bufs.isMXFP8 ? "qwg_moe_\(prefillKind)_gemm_mxfp8_g32" : "qwg_moe_\(prefillKind)_gemm_nvfp4_g16")
            : (bufs.isMXFP8 ? "qwg_moe_compact_mxfp8_gate_up_swiglu" : "qwg_moe_compact_nvfp4_gate_up_swiglu")
        guard let pGrp = resolve(hal, "qwg_moe_group_pairs_f32"),
              let pComb = resolve(hal, "qwg_moe_down_reduce_combine_bf16") else { return (candName, nil) }
        let pairs = t * bufs.topK
        let tg256 = MTLSize(width: 256, height: 1, depth: 1)

        if isPrefill {
            guard hal.caps.supportsSimdgroupMMA,
                  let pNaxGU = resolve(hal, candName),
                  let pNaxDn = resolve(hal, candName),
                  let pSwiGLU = resolve(hal, "qwg_swiglu_softcap_f16") else { return (candName, nil) }
            let guCols = (bufs.inter + 63) / 64, dnCols = (bufs.h + 63) / 64
            let maxTiles = min(pairs, min(bufs.eCount, pairs) + (pairs + 63) / 64)
            let chunk = max(1, 2560 / guCols), dnChunk = max(1, 2560 / dnCols)
            let guSlices = (maxTiles + chunk - 1) / chunk
            let downIndirectOff = 16 + guSlices * 12
            var writeWord = UInt32(1 | ((guCols & 0x7FFF) << 1) | ((dnCols & 0xFFFF) << 16))
            return (candName, { _, enc in
                var eU = UInt32(bufs.eCount), pU = UInt32(pairs), rU = UInt32(bufs.topK)
                var wInv: UInt32 = 1
                enc.setComputePipelineState(pGrp)
                enc.setBuffer(bufs.topIdsBuf, offset: 0, index: 0)
                enc.setBuffer(bufs.offsetsBuf, offset: 0, index: 1)
                enc.setBuffer(bufs.sortedBuf, offset: 0, index: 2)
                enc.setBuffer(bufs.sortedTok, offset: 0, index: 3)
                enc.setBytes(&eU, length: 4, index: 4); enc.setBytes(&pU, length: 4, index: 5); enc.setBytes(&rU, length: 4, index: 6)
                enc.setBuffer(bufs.invPerm, offset: 0, index: 7); enc.setBytes(&wInv, length: 4, index: 8)
                enc.setBuffer(bufs.tileE, offset: 0, index: 9); enc.setBuffer(bufs.tileR, offset: 0, index: 10)
                enc.setBuffer(bufs.tileCount, offset: 0, index: 11); enc.setBytes(&writeWord, length: 4, index: 12)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg256)
                enc.memoryBarrier(scope: .buffers)

                for (isUp, yColOff) in [(false, UInt32(0)), (true, UInt32(bufs.inter))] {
                    let w = isUp ? bufs.uw : bufs.gw, sc = isUp ? bufs.uSc : bufs.gSc, s2 = isUp ? bufs.uS2 : bufs.gS2
                    var np = Ink2MoENaxP(experts: eU, n: UInt32(bufs.inter), k: UInt32(bufs.h), yStride: UInt32(2 * bufs.inter),
                                         yColOff: yColOff, inDtype: 1, outDtype: 1, flags: bufs.isMXFP8 ? 2 : 3)
                    enc.setComputePipelineState(pNaxGU)
                    if bufs.isMXFP8 {
                        enc.setBuffer(w, offset: 0, index: 0); enc.setBuffer(sc, offset: 0, index: 1)
                        enc.setBuffer(bufs.xBuf, offset: 0, index: 2); enc.setBuffer(bufs.prefillGU, offset: 0, index: 3)
                        enc.setBuffer(bufs.offsetsBuf, offset: 0, index: 4); enc.setBuffer(bufs.tileE, offset: 0, index: 5)
                        enc.setBuffer(bufs.tileR, offset: 0, index: 6); enc.setBuffer(bufs.tileCount, offset: 0, index: 7)
                        enc.setBytes(&np, length: MemoryLayout<Ink2MoENaxP>.stride, index: 8)
                        enc.setBuffer(bufs.sortedTok, offset: 0, index: 10)
                    } else {
                        enc.setBuffer(w, offset: 0, index: 0); enc.setBuffer(sc, offset: 0, index: 1); enc.setBuffer(s2, offset: 0, index: 2)
                        enc.setBuffer(bufs.xBuf, offset: 0, index: 3); enc.setBuffer(bufs.prefillGU, offset: 0, index: 4)
                        enc.setBuffer(bufs.offsetsBuf, offset: 0, index: 5); enc.setBuffer(bufs.tileE, offset: 0, index: 6)
                        enc.setBuffer(bufs.tileR, offset: 0, index: 7); enc.setBuffer(bufs.tileCount, offset: 0, index: 8)
                        enc.setBytes(&np, length: MemoryLayout<Ink2MoENaxP>.stride, index: 9)
                        enc.setBuffer(bufs.sortedTok, offset: 0, index: 11)
                    }
                    dispatchChunks(enc, bufs.tileCount, 16, maxTiles, chunk, bufs.isMXFP8 ? 9 : 10)
                }
                enc.memoryBarrier(scope: .buffers)

                var sp = Ink2SwiGLUP(m: UInt32(pairs), i: UInt32(bufs.inter), gateCap: bufs.gateCap, upCap: bufs.upCap, flags: 3)
                enc.setComputePipelineState(pSwiGLU)
                enc.setBuffer(bufs.prefillGU, offset: 0, index: 0); enc.setBuffer(bufs.moeActBuf, offset: 0, index: 1)
                enc.setBytes(&sp, length: MemoryLayout<Ink2SwiGLUP>.stride, index: 2)
                let nx = max(1, (bufs.inter + 7) / 8)
                enc.dispatchThreads(MTLSize(width: nx, height: pairs, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: min(256, nx), height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)

                var dp = Ink2MoENaxP(experts: eU, n: UInt32(bufs.h), k: UInt32(bufs.inter), yStride: UInt32(bufs.h),
                                     yColOff: 0, inDtype: 1, outDtype: 1, flags: bufs.isMXFP8 ? 0 : 1)
                enc.setComputePipelineState(pNaxDn)
                if bufs.isMXFP8 {
                    enc.setBuffer(bufs.dw, offset: 0, index: 0); enc.setBuffer(bufs.dSc, offset: 0, index: 1)
                    enc.setBuffer(bufs.moeActBuf, offset: 0, index: 2); enc.setBuffer(bufs.slotOutBuf, offset: 0, index: 3)
                    enc.setBuffer(bufs.offsetsBuf, offset: 0, index: 4); enc.setBuffer(bufs.tileE, offset: 0, index: 5)
                    enc.setBuffer(bufs.tileR, offset: 0, index: 6); enc.setBuffer(bufs.tileCount, offset: 0, index: 7)
                    enc.setBytes(&dp, length: MemoryLayout<Ink2MoENaxP>.stride, index: 8)
                    enc.setBuffer(bufs.sortedTok, offset: 0, index: 10)
                } else {
                    enc.setBuffer(bufs.dw, offset: 0, index: 0); enc.setBuffer(bufs.dSc, offset: 0, index: 1); enc.setBuffer(bufs.dS2, offset: 0, index: 2)
                    enc.setBuffer(bufs.moeActBuf, offset: 0, index: 3); enc.setBuffer(bufs.slotOutBuf, offset: 0, index: 4)
                    enc.setBuffer(bufs.offsetsBuf, offset: 0, index: 5); enc.setBuffer(bufs.tileE, offset: 0, index: 6)
                    enc.setBuffer(bufs.tileR, offset: 0, index: 7); enc.setBuffer(bufs.tileCount, offset: 0, index: 8)
                    enc.setBytes(&dp, length: MemoryLayout<Ink2MoENaxP>.stride, index: 9)
                    enc.setBuffer(bufs.sortedTok, offset: 0, index: 11)
                }
                dispatchChunks(enc, bufs.tileCount, downIndirectOff, maxTiles, dnChunk, bufs.isMXFP8 ? 9 : 10)
                enc.memoryBarrier(scope: .buffers)
                encodeCombine(enc, pComb, bufs: bufs, t: t)
            })
        }

        let dnCand = bufs.isMXFP8 ? "qwg_moe_compact_mxfp8_down" : "qwg_moe_compact_nvfp4_down"
        guard let pBuild = resolve(hal, "qwg_moe_build_active_buckets"),
              let pGU = resolve(hal, candName),
              let pDn = resolve(hal, dnCand) else { return (candName, nil) }
        return (candName, { _, enc in
            var eU = UInt32(bufs.eCount), pU = UInt32(pairs), rU = UInt32(bufs.topK)
            var wInv: UInt32 = 1, wZero: UInt32 = 0
            enc.setComputePipelineState(pGrp)
            enc.setBuffer(bufs.topIdsBuf, offset: 0, index: 0); enc.setBuffer(bufs.offsetsBuf, offset: 0, index: 1)
            enc.setBuffer(bufs.sortedBuf, offset: 0, index: 2); enc.setBuffer(bufs.sortedTok, offset: 0, index: 3)
            enc.setBytes(&eU, length: 4, index: 4); enc.setBytes(&pU, length: 4, index: 5); enc.setBytes(&rU, length: 4, index: 6)
            enc.setBuffer(bufs.invPerm, offset: 0, index: 7); enc.setBytes(&wInv, length: 4, index: 8)
            enc.setBuffer(bufs.tileE, offset: 0, index: 9); enc.setBuffer(bufs.tileR, offset: 0, index: 10)
            enc.setBuffer(bufs.tileCount, offset: 0, index: 11); enc.setBytes(&wZero, length: 4, index: 12)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg256)
            enc.memoryBarrier(scope: .buffers)

            var guGX = UInt32((bufs.inter + 31) / 32), dnGX = UInt32((bufs.h + 31) / 32)
            enc.setComputePipelineState(pBuild)
            enc.setBuffer(bufs.offsetsBuf, offset: 0, index: 0); enc.setBuffer(bufs.activeExperts, offset: 0, index: 1)
            enc.setBuffer(bufs.activeArgs, offset: 0, index: 2); enc.setBytes(&eU, length: 4, index: 3)
            enc.setBytes(&guGX, length: 4, index: 4); enc.setBytes(&dnGX, length: 4, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg256)
            enc.memoryBarrier(scope: .buffers)

            var cpGU = Ink2MoECompactP(n: UInt32(bufs.inter), k: UInt32(bufs.h), topK: rU, numExperts: eU,
                                       wExpertRows: UInt32(bufs.inter), upRowBias: 0, inDtype: 1, outDtype: 1,
                                       flags: bufs.isMXFP8 ? 2 : 3, gateCap: bufs.gateCap, upCap: bufs.upCap)
            enc.setComputePipelineState(pGU)
            if bufs.isMXFP8 {
                enc.setBuffer(bufs.xBuf, offset: 0, index: 0); enc.setBuffer(bufs.gw, offset: 0, index: 1); enc.setBuffer(bufs.gSc, offset: 0, index: 2)
                enc.setBuffer(bufs.uw, offset: 0, index: 3); enc.setBuffer(bufs.uSc, offset: 0, index: 4)
                enc.setBuffer(bufs.offsetsBuf, offset: 0, index: 5); enc.setBuffer(bufs.activeExperts, offset: 0, index: 6)
                enc.setBuffer(bufs.activeArgs, offset: 0, index: 7); enc.setBuffer(bufs.sortedTok, offset: 0, index: 8)
                enc.setBuffer(bufs.sortedBuf, offset: 0, index: 9); enc.setBuffer(bufs.moeActBuf, offset: 0, index: 10)
                enc.setBytes(&cpGU, length: MemoryLayout<Ink2MoECompactP>.stride, index: 11)
            } else {
                enc.setBuffer(bufs.xBuf, offset: 0, index: 0); enc.setBuffer(bufs.gw, offset: 0, index: 1)
                enc.setBuffer(bufs.gSc, offset: 0, index: 2); enc.setBuffer(bufs.gS2, offset: 0, index: 3)
                enc.setBuffer(bufs.uw, offset: 0, index: 4); enc.setBuffer(bufs.uSc, offset: 0, index: 5)
                enc.setBuffer(bufs.uS2, offset: 0, index: 6); enc.setBuffer(bufs.offsetsBuf, offset: 0, index: 7)
                enc.setBuffer(bufs.activeExperts, offset: 0, index: 8); enc.setBuffer(bufs.activeArgs, offset: 0, index: 9)
                enc.setBuffer(bufs.sortedTok, offset: 0, index: 10); enc.setBuffer(bufs.sortedBuf, offset: 0, index: 11)
                enc.setBuffer(bufs.moeActBuf, offset: 0, index: 12); enc.setBytes(&cpGU, length: MemoryLayout<Ink2MoECompactP>.stride, index: 13)
            }
            enc.dispatchThreadgroups(indirectBuffer: bufs.activeArgs, indirectBufferOffset: 16, threadsPerThreadgroup: tg256)
            enc.memoryBarrier(scope: .buffers)

            var cpDn = Ink2MoECompactP(n: UInt32(bufs.h), k: UInt32(bufs.inter), topK: rU, numExperts: eU,
                                       wExpertRows: UInt32(bufs.h), upRowBias: 0, inDtype: 1, outDtype: 1,
                                       flags: bufs.isMXFP8 ? 2 : 3, gateCap: 0, upCap: 0)
            enc.setComputePipelineState(pDn)
            if bufs.isMXFP8 {
                enc.setBuffer(bufs.moeActBuf, offset: 0, index: 0); enc.setBuffer(bufs.dw, offset: 0, index: 1)
                enc.setBuffer(bufs.dSc, offset: 0, index: 2); enc.setBuffer(bufs.offsetsBuf, offset: 0, index: 3)
                enc.setBuffer(bufs.activeExperts, offset: 0, index: 4); enc.setBuffer(bufs.activeArgs, offset: 0, index: 5)
                enc.setBuffer(bufs.sortedBuf, offset: 0, index: 6); enc.setBuffer(bufs.slotOutBuf, offset: 0, index: 7)
                enc.setBytes(&cpDn, length: MemoryLayout<Ink2MoECompactP>.stride, index: 8)
            } else {
                enc.setBuffer(bufs.moeActBuf, offset: 0, index: 0); enc.setBuffer(bufs.dw, offset: 0, index: 1)
                enc.setBuffer(bufs.dSc, offset: 0, index: 2); enc.setBuffer(bufs.dS2, offset: 0, index: 3)
                enc.setBuffer(bufs.offsetsBuf, offset: 0, index: 4); enc.setBuffer(bufs.activeExperts, offset: 0, index: 5)
                enc.setBuffer(bufs.activeArgs, offset: 0, index: 6); enc.setBuffer(bufs.sortedBuf, offset: 0, index: 7)
                enc.setBuffer(bufs.slotOutBuf, offset: 0, index: 8); enc.setBytes(&cpDn, length: MemoryLayout<Ink2MoECompactP>.stride, index: 9)
            }
            enc.dispatchThreadgroups(indirectBuffer: bufs.activeArgs, indirectBufferOffset: 32, threadsPerThreadgroup: tg256)
            enc.memoryBarrier(scope: .buffers)
            encodeCombine(enc, pComb, bufs: bufs, t: t)
        })
    }

    static func makeHeadArgmaxEncoder(
        hal: QWHal, m: Int, v: Int, k: Int, softcap: Float,
        wBuf: any MTLBuffer, xBuf: any MTLBuffer,
        keysHi: any MTLBuffer, keysLo: any MTLBuffer, tokBuf: any MTLBuffer
    ) -> ((any MTLCommandBuffer, inout any MTLComputeCommandEncoder) -> Void)? {
        guard let pArg = resolve(hal, "qwg_head_bf16_argmax_m\(m)"),
              let pFin = resolve(hal, "qwg_head_argmax_batched_fin") else { return nil }
        let rowsPerTG = (m <= 4) ? 16 : 8
        let slots = (v + rowsPerTG - 1) / rowsPerTG
        return { _, enc in
            var nU = UInt32(v), kU = UInt32(k), cap = softcap, slotsU = UInt32(slots), mU = UInt32(m)
            enc.setComputePipelineState(pArg)
            enc.setBuffer(wBuf, offset: 0, index: 0); enc.setBuffer(xBuf, offset: 0, index: 1)
            enc.setBuffer(keysHi, offset: 0, index: 2); enc.setBuffer(keysLo, offset: 0, index: 3)
            enc.setBytes(&nU, length: 4, index: 4); enc.setBytes(&kU, length: 4, index: 5); enc.setBytes(&cap, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: slots, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(pFin)
            enc.setBuffer(keysHi, offset: 0, index: 0); enc.setBuffer(keysLo, offset: 0, index: 1)
            enc.setBuffer(tokBuf, offset: 0, index: 2); enc.setBytes(&slotsU, length: 4, index: 3); enc.setBytes(&mU, length: 4, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: m, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        }
    }

    private static func dispatchChunks(
        _ enc: any MTLComputeCommandEncoder, _ buf: any MTLBuffer,
        _ baseOff: Int, _ maxTiles: Int, _ chunk: Int, _ wordIdx: Int
    ) {
        var base = 0, slice = 0
        while base < maxTiles {
            var word = UInt32(base) << 8
            enc.setBytes(&word, length: 4, index: wordIdx)
            enc.dispatchThreadgroups(indirectBuffer: buf, indirectBufferOffset: baseOff + slice * 12,
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            base += min(chunk, maxTiles - base); slice += 1
        }
    }

    private static func encodeCombine(
        _ enc: any MTLComputeCommandEncoder, _ pComb: any MTLComputePipelineState,
        bufs: MoECandidateBuffers, t: Int
    ) {
        var cp = Ink2MoECombineP(hidden: UInt32(bufs.h), topK: UInt32(bufs.topK), tokens: UInt32(t),
                                 inDtype: 1, outDtype: 1, sharedDtype: 1, flags: 2)
        enc.setComputePipelineState(pComb)
        enc.setBuffer(bufs.slotOutBuf, offset: 0, index: 0); enc.setBuffer(bufs.invPerm, offset: 0, index: 1)
        enc.setBuffer(bufs.topWBuf, offset: 0, index: 2); enc.setBuffer(bufs.outBuf, offset: 0, index: 3)
        enc.setBuffer(bufs.topWBuf, offset: 0, index: 4); enc.setBuffer(bufs.outBuf, offset: 0, index: 5)
        enc.setBytes(&cp, length: MemoryLayout<Ink2MoECombineP>.stride, index: 6)
        enc.setBuffer(bufs.topWBuf, offset: 0, index: 7)
        let gx = (((bufs.h + 3) / 4) + 255) / 256
        enc.dispatchThreadgroups(MTLSize(width: gx, height: t, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}
