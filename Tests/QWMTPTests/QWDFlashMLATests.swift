// QWDFlashMLATests.swift — 0-ULP CPU-vs-Metal parity tests for Matrix-Absorbed MLA
// DFlash 2 kernels (GatedNorm, sandwich norm, conv mode 2, SwiGLU limit=10, c_KV/k_pe
// prep, W_UK absorb, MLA block attention, and W_UV+gate).
import Darwin
import Foundation
import Metal
import XCTest
@testable import QWMTP
import QWCore
import QWHAL
import QWQuant

final class QWDFlashMLATests: XCTestCase {
    static func bf16Bits(_ v: Float) -> UInt16 {
        let u = v.bitPattern
        if (u & 0x7F80_0000) == 0x7F80_0000 { return UInt16(truncatingIfNeeded: u >> 16) }
        let rounded = u &+ 0x7FFF &+ ((u >> 16) & 1)
        return UInt16(truncatingIfNeeded: rounded >> 16)
    }

    static func bf16ToF32(_ b: UInt16) -> Float { Float(bitPattern: UInt32(b) << 16) }
    static func bf16Round(_ v: Float) -> Float { bf16ToF32(bf16Bits(v)) }

    static func cpuFWHT32(_ block: [Float]) -> [Float] {
        precondition(block.count == 32)
        var x = block
        var s = 1
        while s < 32 {
            var next = x
            for i in 0..<32 {
                let p = x[i ^ s]
                next[i] = ((i & s) == 0) ? (x[i] + p) : (p - x[i])
            }
            x = next
            s <<= 1
        }
        return x.map { $0 * 0.17677669529663689 }
    }

    static func cpuSimdSum32(_ vals: [Float]) -> Float {
        precondition(vals.count == 32)
        var v = vals
        var s = 16
        while s >= 1 {
            for i in 0..<s { v[i] += v[i + s] }
            s >>= 1
        }
        return v[0]
    }

    static func cpuGatedNormRow(
        x: [Float], normW: [UInt16], gateUp: [UInt16], gateDown: [UInt16],
        eps: Float, gatingWeight: Float, scale: Float, rank: Int, zeroCentered: Bool, tptg: Int
    ) -> [Float] {
        let H = x.count, nSg = tptg / 32
        var sgSums = [Float](repeating: 0, count: 32)
        for sg in 0..<nSg {
            var laneAcc = [Float](repeating: 0, count: 32)
            for lane in 0..<32 {
                var sq: Float = 0
                var i = sg * 32 + lane
                while i < H { sq += x[i] * x[i]; i += tptg }
                laneAcc[lane] = sq
            }
            sgSums[sg] = cpuSimdSum32(laneAcc)
        }
        let rstd = 1.0 / sqrtf(cpuSimdSum32(sgSums) * (Float(1.0) / Float(H)) + eps)
        var h = [Float](repeating: 0, count: H)
        for i in 0..<H {
            let w = bf16ToF32(normW[i]) + (zeroCentered ? 1.0 : 0.0)
            h[i] = bf16Round(x[i] * rstd * w)
        }
        var g1 = [Float](repeating: 0, count: rank)
        for r in 0..<rank {
            var laneAcc = [Float](repeating: 0, count: 32)
            for lane in 0..<32 {
                var acc: Float = 0
                var c = lane
                while c < H { acc += bf16ToF32(gateUp[r * H + c]) * h[c]; c += 32 }
                laneAcc[lane] = acc
            }
            let s = cpuSimdSum32(laneAcc)
            g1[r] = bf16Round(s / (1.0 + expf(-s)))
        }
        var y = [Float](repeating: 0, count: H)
        for i in 0..<H {
            var gdAcc: Float = 0
            for r in 0..<rank { gdAcc += bf16ToF32(gateDown[i * rank + r]) * g1[r] }
            let g2 = bf16Round(1.0 / (1.0 + expf(-gdAcc)))
            var yi = bf16Round(h[i] * gatingWeight * g2)
            if scale != 1.0 { yi = bf16Round(yi * scale) }
            y[i] = yi
        }
        return y
    }

    func sharedBuffer<T>(_ dev: any MTLDevice, _ values: [T]) throws -> any MTLBuffer {
        let bytes = max(16, values.count * MemoryLayout<T>.stride)
        let buf = try XCTUnwrap(dev.makeBuffer(length: bytes, options: .storageModeShared))
        let dst = buf.contents()
        withExtendedLifetime(buf) {
            values.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                if let src = raw.baseAddress, raw.count > 0 {
                    dst.copyMemory(from: src, byteCount: raw.count)
                }
            }
        }
        return buf
    }

    func zeroBuffer(_ dev: any MTLDevice, byteCount: Int) throws -> any MTLBuffer {
        let buf = try XCTUnwrap(dev.makeBuffer(length: max(16, byteCount), options: .storageModeShared))
        memset(buf.contents(), 0, buf.length)
        return buf
    }

    private struct GatedNormTestParams {
        var eps, gatingWeight, scale: Float
        var hidden, rank, zeroCentered, mode, hasNextPreNorm: UInt32
    }

    func testGatedRMSNormAndSandwich0ULPParity() throws {
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)
        let dev = hal.device
        let psoNormRot = try hal.pipeline("qwg_dflash_gated_rmsnorm", fcDrafterRotated: true, fcHeadRotated: true)
        let psoNormStd = try hal.pipeline("qwg_dflash_gated_rmsnorm", fcDrafterRotated: false, fcHeadRotated: false)
        let psoSandRot = try hal.pipeline("qwg_dflash_gated_rmsnorm_sandwich", fcDrafterRotated: true, fcHeadRotated: true)

        let H = 2560, rank = 16, rows = 3
        let eps: Float = 1e-6, gatingWeight: Float = 2.0
        var xF32 = [Float](repeating: 0, count: rows * H)
        var xF16 = [UInt16](repeating: 0, count: rows * H)
        for i in 0..<xF32.count {
            let v = Float(((i * 7 + 3) % 17) - 8) * 0.25
            xF32[i] = v
            xF16[i] = Float16(v).bitPattern
        }
        let normW = (0..<H).map { Self.bf16Bits(0.75 + Float(($0 % 5) - 2) * 0.125) }
        let preW = (0..<H).map { Self.bf16Bits(1.0 + Float(($0 % 7) - 3) * 0.0625) }
        let gateUp = (0..<(rank * H)).map { Self.bf16Bits(Float((($0 * 11 + 5) % 9) - 4) * 0.03125) }
        let gateDown = (0..<(H * rank)).map { Self.bf16Bits(Float((($0 * 13 + 1) % 7) - 3) * 0.0625) }
        let preGU = (0..<(rank * H)).map { Self.bf16Bits(Float((($0 * 5 + 2) % 9) - 4) * 0.03125) }
        let preGD = (0..<(H * rank)).map { Self.bf16Bits(Float((($0 * 3 + 4) % 7) - 3) * 0.0625) }

        let x32Buf = try sharedBuffer(dev, xF32), x16Buf = try sharedBuffer(dev, xF16)
        let wBuf = try sharedBuffer(dev, normW), guBuf = try sharedBuffer(dev, gateUp), gdBuf = try sharedBuffer(dev, gateDown)
        let preWBuf = try sharedBuffer(dev, preW), preGUBuf = try sharedBuffer(dev, preGU), preGDBuf = try sharedBuffer(dev, preGD)

        for (rot, pso) in [(true, psoNormRot), (false, psoNormStd)] {
            let yBuf = try zeroBuffer(dev, byteCount: rows * H * 2)
            let yRotBuf = try zeroBuffer(dev, byteCount: rows * H * 2)
            let yRot32Buf = try zeroBuffer(dev, byteCount: rows * H * 4)
            var p5 = GatedNormTestParams(
                eps: eps, gatingWeight: gatingWeight, scale: 1.0,
                hidden: UInt32(H), rank: UInt32(rank), zeroCentered: 0, mode: 5, hasNextPreNorm: 0
            )
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(pso)
            enc.setBuffer(x32Buf, offset: 0, index: 0)
            enc.setBuffer(wBuf, offset: 0, index: 1)
            enc.setBuffer(guBuf, offset: 0, index: 2)
            enc.setBuffer(gdBuf, offset: 0, index: 3)
            enc.setBuffer(yBuf, offset: 0, index: 4)
            enc.setBuffer(yRotBuf, offset: 0, index: 5)
            enc.setBuffer(yRot32Buf, offset: 0, index: 6)
            enc.setBytes(&p5, length: MemoryLayout<GatedNormTestParams>.stride, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()

            let yPtr = yBuf.contents().assumingMemoryBound(to: UInt16.self)
            let yRotPtr = yRotBuf.contents().assumingMemoryBound(to: UInt16.self)
            let yRot32Ptr = yRot32Buf.contents().assumingMemoryBound(to: UInt32.self)
            for r in 0..<rows {
                let refY = Self.cpuGatedNormRow(
                    x: Array(xF32[(r * H)..<((r + 1) * H)]), normW: normW, gateUp: gateUp, gateDown: gateDown,
                    eps: eps, gatingWeight: gatingWeight, scale: 1.0, rank: rank, zeroCentered: false, tptg: 256
                )
                for c in 0..<H { XCTAssertEqual(yPtr[r * H + c], Float16(refY[c]).bitPattern) }
                for blk in 0..<(H / 32) {
                    let slice = Array(refY[(blk * 32)..<((blk + 1) * 32)])
                    let refR = rot ? Self.cpuFWHT32(slice) : slice
                    for k in 0..<32 {
                        let idx = r * H + blk * 32 + k
                        XCTAssertEqual(yRotPtr[idx], Float16(refR[k]).bitPattern)
                        XCTAssertEqual(yRot32Ptr[idx], refR[k].bitPattern)
                    }
                }
            }
        }

        let propXInit = xF32.map { $0 * 0.5 }
        let propXBuf = try sharedBuffer(dev, propXInit)
        let outPreBuf = try zeroBuffer(dev, byteCount: rows * H * 2)
        var ps = GatedNormTestParams(
            eps: eps, gatingWeight: gatingWeight, scale: 1.0,
            hidden: UInt32(H), rank: UInt32(rank), zeroCentered: 0, mode: 1, hasNextPreNorm: 1
        )
        let cbS = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let encS = try XCTUnwrap(cbS.makeComputeCommandEncoder())
        encS.setComputePipelineState(psoSandRot)
        encS.setBuffer(propXBuf, offset: 0, index: 0)
        encS.setBuffer(x16Buf, offset: 0, index: 1)
        encS.setBuffer(wBuf, offset: 0, index: 2)
        encS.setBuffer(guBuf, offset: 0, index: 3)
        encS.setBuffer(gdBuf, offset: 0, index: 4)
        encS.setBuffer(preWBuf, offset: 0, index: 5)
        encS.setBuffer(preGUBuf, offset: 0, index: 6)
        encS.setBuffer(preGDBuf, offset: 0, index: 7)
        encS.setBuffer(outPreBuf, offset: 0, index: 8)
        encS.setBytes(&ps, length: MemoryLayout<GatedNormTestParams>.stride, index: 9)
        encS.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encS.endEncoding()
        cbS.commit()
        cbS.waitUntilCompleted()

        let gotXPtr = propXBuf.contents().assumingMemoryBound(to: UInt32.self)
        let gotPrePtr = outPreBuf.contents().assumingMemoryBound(to: UInt16.self)
        for r in 0..<rows {
            let bRow = (0..<H).map { Float(Float16(bitPattern: xF16[r * H + $0])) }
            let postY = Self.cpuGatedNormRow(
                x: bRow, normW: normW, gateUp: gateUp, gateDown: gateDown,
                eps: eps, gatingWeight: gatingWeight, scale: 1.0, rank: rank, zeroCentered: false, tptg: 256
            )
            var nextX = [Float](repeating: 0, count: H)
            for c in 0..<H {
                nextX[c] = propXInit[r * H + c] + postY[c]
                XCTAssertEqual(gotXPtr[r * H + c], nextX[c].bitPattern)
            }
            let preY = Self.cpuGatedNormRow(
                x: nextX, normW: preW, gateUp: preGU, gateDown: preGD,
                eps: eps, gatingWeight: gatingWeight, scale: 1.0, rank: rank, zeroCentered: false, tptg: 256
            )
            for blk in 0..<(H / 32) {
                let rotPre = Self.cpuFWHT32(Array(preY[(blk * 32)..<((blk + 1) * 32)]))
                for k in 0..<32 {
                    XCTAssertEqual(gotPrePtr[r * H + blk * 32 + k], Float16(rotPre[k]).bitPattern)
                }
            }
        }
    }

    func testConvMode2AndSwiGLUClamp10ExactParity() throws {
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)
        let dev = hal.device
        let psoConv = try hal.pipeline("qwg_dflash_conv_apply", fcDrafterRotated: true, fcHeadRotated: true)
        let psoSiluRot = try hal.pipeline("qwg_dflash_silu_mul_f16", fcDrafterRotated: true, fcHeadRotated: true)

        let H = 2560, convGroups = 160, M = 4
        let xVals = (0..<(M * H)).map { Float16((Float((($0 * 7) % 12) - 6) + 0.5) * 0.125).bitPattern }
        let baseVals = (0..<(4 * H)).map { Float16((Float((($0 * 3 + 1) % 8) - 4) + 0.5) * 0.25).bitPattern }
        let dynVals = (0..<(M * 4 * convGroups)).map { (Float((($0 * 5 + 2) % 10) - 5) + 0.5) * 0.0625 }

        let xBuf = try sharedBuffer(dev, xVals), dynBuf = try sharedBuffer(dev, dynVals), baseBuf = try sharedBuffer(dev, baseVals)
        let out2Buf = try zeroBuffer(dev, byteCount: M * H * 2)

        let I = 64
        var guVals = [UInt16](repeating: 0, count: M * 2 * I)
        let testGates: [Float] = [-14.0, -4.0, 2.5, 10.0, 14.5, -0.5, 8.0, 19.0]
        let testUps: [Float] = [-16.0, -10.0, -3.5, 4.0, 10.0, 15.5, -12.0, 1.25]
        for m in 0..<M {
            for c in 0..<I {
                guVals[m * 2 * I + c] = Float16(testGates[(m + c) % testGates.count]).bitPattern
                guVals[m * 2 * I + I + c] = Float16(testUps[(m * 3 + c) % testUps.count]).bitPattern
            }
        }
        let guBuf = try sharedBuffer(dev, guVals), siluOutBuf = try zeroBuffer(dev, byteCount: M * I * 2)

        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        var mode2: UInt32 = 2, hiddenU4 = UInt32(H / 8), groupsU = UInt32(convGroups)
        enc.setComputePipelineState(psoConv)
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(dynBuf, offset: 0, index: 1)
        enc.setBuffer(baseBuf, offset: 0, index: 2)
        enc.setBuffer(out2Buf, offset: 0, index: 3)
        enc.setBytes(&mode2, length: 4, index: 4)
        enc.setBytes(&hiddenU4, length: 4, index: 5)
        enc.setBytes(&groupsU, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: M, height: (H / 8 + 31) / 32, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        var mPacked = UInt32(M) | (UInt32(Float16(10.0).bitPattern) << 16), iU = UInt32(I)
        enc.setComputePipelineState(psoSiluRot)
        enc.setBuffer(guBuf, offset: 0, index: 0)
        enc.setBuffer(siluOutBuf, offset: 0, index: 1)
        enc.setBytes(&mPacked, length: 4, index: 2)
        enc.setBytes(&iU, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: M, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let gotConv = out2Buf.contents().assumingMemoryBound(to: UInt16.self)
        let groupSize = H / convGroups
        for t in 0..<M {
            for h in 0..<H {
                let g = h / groupSize
                let xc = Float(Float16(bitPattern: xVals[t * H + h]))
                let xp = t > 0 ? Float(Float16(bitPattern: xVals[(t - 1) * H + h])) : 0
                let d1Off = t * 4 * convGroups + 2 * convGroups
                let w0 = Float(Float16(bitPattern: baseVals[2 * H + h])) + dynVals[d1Off + g]
                let w1 = Float(Float16(bitPattern: baseVals[3 * H + h])) + dynVals[d1Off + convGroups + g]
                XCTAssertEqual(gotConv[t * H + h], Float16(xc * w0 + (t > 0 ? xp * w1 : 0)).bitPattern)
            }
        }

        let gotSilu = siluOutBuf.contents().assumingMemoryBound(to: UInt16.self)
        for m in 0..<M {
            var unrot = [Float](repeating: 0, count: I)
            for c in 0..<I {
                let g = min(Float(Float16(bitPattern: guVals[m * 2 * I + c])), 10.0)
                let u = min(max(Float(Float16(bitPattern: guVals[m * 2 * I + I + c])), -10.0), 10.0)
                unrot[c] = (g / (1.0 + expf(-g))) * u
            }
            for blk in 0..<(I / 32) {
                let rot = Self.cpuFWHT32(Array(unrot[(blk * 32)..<((blk + 1) * 32)]))
                for k in 0..<32 {
                    XCTAssertEqual(gotSilu[m * I + blk * 32 + k], Float16(rot[k]).bitPattern)
                }
            }
        }
    }

    private struct LatentParams {
        var eps, gatingWeight, alphaQ, alphaKV, ropeMscale: Float
        var qLoraRank, kvLoraRank, ropeDim, gateDim: UInt32
        var normRank, zeroCentered, ropeInterleave, posOffset, maxCacheSize: UInt32
    }

    private struct QBParams {
        var qHeads, nopeDim, ropeDim, vHeadDim, kvLoraRank, mRows, posOffset, ropeInterleave: UInt32
        var ropeMscale: Float
        var quantKind, bits, group, qkvStride, gateOffset, gatedAttention, pad: UInt32
    }

    private struct AttnParams {
        var qHeads, nopeDim, ropeDim, kvLoraRank, newOffset, mRows, maxCacheSize, isCausal: UInt32
        var softmaxScale: Float
        var pad0: UInt32 = 0, pad1: UInt32 = 0, pad2: UInt32 = 0
    }

    func testMLAKVCtxLatentAbsorbAttnAndProjUVGate0ULPParity() throws {
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)
        let dev = hal.device
        let psoCtx = try hal.pipeline("qwg_dflash_mla_kv_ctx_prep", fcDrafterRotated: false, fcHeadRotated: false)
        let psoLat = try hal.pipeline("qwg_dflash_mla_latent_prep", fcDrafterRotated: false, fcHeadRotated: false)
        let psoAbs = try hal.pipeline("qwg_dflash_mla_q_absorb")
        let psoAttn = try hal.pipeline("qwg_dflash_mla_attn")
        let psoUV = try hal.pipeline("qwg_dflash_mla_proj_uv_gate", fcDrafterRotated: true, fcHeadRotated: true)

        let qHeads = 4, nopeDim = 128, ropeDim = 64, vDim = 128, qLora = 256, kvLora = 512, normRank = 16
        let gateDim = qHeads * vDim, qkvStride = qLora + kvLora + ropeDim + gateDim
        let qFullStride = qHeads * (nopeDim + ropeDim)
        let maxCache = 4, S = 3, M = 3, ctxOffset = 2, newOffset = ctxOffset + S
        let eps: Float = 1e-6, gatingW: Float = 2.0
        let alphaQ = Float((2560.0 / 1536.0).squareRoot()), alphaKV = Float((2560.0 / 512.0).squareRoot())
        let yarn = QWModelDescriptor.YarnRopeConfig(
            factor: 32.0, originalMaxPositionEmbeddings: 32768,
            betaFast: 32.0, betaSlow: 1.0, mscale: 1.0, mscaleAllDim: 1.0
        )
        let (invFreq, ropeMscale) = QWModelDescriptor.computeRopeInvFreq(rotaryDim: ropeDim, base: 100_000.0, yarn: yarn)
        let mAll = 0.1 * Foundation.log(32.0) + 1.0
        let softmaxScale = Float((1.0 / 192.0.squareRoot()) * mAll * mAll)

        let kvCtxIn = (0..<(S * (kvLora + ropeDim))).map {
            Float16((Float((($0 * 5 + 1) % 8) - 4) + 0.5) * 0.5).bitPattern
        }
        var qkvProp = [UInt16](repeating: 0, count: M * qkvStride)
        for m in 0..<M {
            let base = m * qkvStride
            for i in 0..<(qLora + kvLora + ropeDim) {
                qkvProp[base + i] = Float16((Float(((m * 13 + i * 7 + 2) % 8) - 4) + 0.5) * 0.5).bitPattern
            }
            for i in 0..<gateDim {
                qkvProp[base + qLora + kvLora + ropeDim + i] = Float16(0.0).bitPattern
            }
        }
        let qFullInit = (0..<(M * qFullStride)).map { Float16((Float((($0 * 3 + 4) % 8) - 4) + 0.5) * 0.25).bitPattern }
        let qNW = (0..<qLora).map { Self.bf16Bits(1.0 + Float($0 % 3) * 0.25) }
        let qGU = (0..<(normRank * qLora)).map { Self.bf16Bits(Float((($0 * 3) % 5) - 2) * 0.0625) }
        let qGD = (0..<(qLora * normRank)).map { Self.bf16Bits(Float((($0 * 7) % 5) - 2) * 0.0625) }
        let kvNW = (0..<kvLora).map { Self.bf16Bits(1.25 + Float($0 % 4) * 0.25) }
        let kvGU = (0..<(normRank * kvLora)).map { Self.bf16Bits(Float((($0 * 5) % 5) - 2) * 0.0625) }
        let kvGD = (0..<(kvLora * normRank)).map { Self.bf16Bits(Float((($0 * 9) % 5) - 2) * 0.0625) }
        let kvB = (0..<(qHeads * (nopeDim + vDim) * kvLora)).map {
            Self.bf16Bits((Float((($0 * 11 + 3) % 4) - 2) + 0.5) * 0.25)
        }

        let kvCtxBuf = try sharedBuffer(dev, kvCtxIn), qkvBuf = try sharedBuffer(dev, qkvProp)
        let qFullBuf = try sharedBuffer(dev, qFullInit), invBuf = try sharedBuffer(dev, invFreq), kvBBuf = try sharedBuffer(dev, kvB)
        let qNWBuf = try sharedBuffer(dev, qNW), qGUBuf = try sharedBuffer(dev, qGU), qGDBuf = try sharedBuffer(dev, qGD)
        let kvNWBuf = try sharedBuffer(dev, kvNW), kvGUBuf = try sharedBuffer(dev, kvGU), kvGDBuf = try sharedBuffer(dev, kvGD)

        let kCacheBuf = try zeroBuffer(dev, byteCount: maxCache * kvLora * 2)
        let vCacheBuf = try zeroBuffer(dev, byteCount: maxCache * ropeDim * 2)
        let qLatBuf = try zeroBuffer(dev, byteCount: M * qLora * 2)
        let propCKVBuf = try zeroBuffer(dev, byteCount: M * kvLora * 2)
        let propKPeBuf = try zeroBuffer(dev, byteCount: M * ropeDim * 2)
        let qAbsBuf = try zeroBuffer(dev, byteCount: M * qHeads * kvLora * 2)
        let uLatBuf = try zeroBuffer(dev, byteCount: M * qHeads * kvLora * 2)
        let attnOutBuf = try zeroBuffer(dev, byteCount: M * qHeads * vDim * 2)

        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        var lpCtx = LatentParams(
            eps: eps, gatingWeight: gatingW, alphaQ: alphaQ, alphaKV: alphaKV, ropeMscale: ropeMscale,
            qLoraRank: UInt32(qLora), kvLoraRank: UInt32(kvLora), ropeDim: UInt32(ropeDim),
            gateDim: UInt32(gateDim), normRank: UInt32(normRank), zeroCentered: 0, ropeInterleave: 1,
            posOffset: UInt32(ctxOffset), maxCacheSize: UInt32(maxCache)
        )
        enc.setComputePipelineState(psoCtx)
        enc.setBuffer(kvCtxBuf, offset: 0, index: 0); enc.setBuffer(kvNWBuf, offset: 0, index: 1)
        enc.setBuffer(kvGUBuf, offset: 0, index: 2); enc.setBuffer(kvGDBuf, offset: 0, index: 3)
        enc.setBuffer(invBuf, offset: 0, index: 4); enc.setBuffer(kCacheBuf, offset: 0, index: 5)
        enc.setBuffer(vCacheBuf, offset: 0, index: 6)
        enc.setBytes(&lpCtx, length: MemoryLayout<LatentParams>.stride, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: S, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

        var lpProp = lpCtx; lpProp.posOffset = UInt32(newOffset)
        enc.setComputePipelineState(psoLat)
        enc.setBuffer(qkvBuf, offset: 0, index: 0); enc.setBuffer(qNWBuf, offset: 0, index: 1)
        enc.setBuffer(qGUBuf, offset: 0, index: 2); enc.setBuffer(qGDBuf, offset: 0, index: 3)
        enc.setBuffer(kvNWBuf, offset: 0, index: 4); enc.setBuffer(kvGUBuf, offset: 0, index: 5)
        enc.setBuffer(kvGDBuf, offset: 0, index: 6); enc.setBuffer(invBuf, offset: 0, index: 7)
        enc.setBuffer(qLatBuf, offset: 0, index: 8); enc.setBuffer(propCKVBuf, offset: 0, index: 9)
        enc.setBuffer(propKPeBuf, offset: 0, index: 10)
        enc.setBytes(&lpProp, length: MemoryLayout<LatentParams>.stride, index: 11)
        enc.dispatchThreadgroups(MTLSize(width: M, height: 2, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        var qbp = QBParams(
            qHeads: UInt32(qHeads), nopeDim: UInt32(nopeDim), ropeDim: UInt32(ropeDim),
            vHeadDim: UInt32(vDim), kvLoraRank: UInt32(kvLora), mRows: UInt32(M),
            posOffset: UInt32(newOffset), ropeInterleave: 1, ropeMscale: ropeMscale,
            quantKind: 3, bits: 16, group: 1, qkvStride: UInt32(qkvStride),
            gateOffset: UInt32(qLora + kvLora + ropeDim), gatedAttention: 1, pad: 0
        )
        enc.setComputePipelineState(psoAbs)
        enc.setBuffer(qFullBuf, offset: 0, index: 0); enc.setBuffer(kvBBuf, offset: 0, index: 1)
        enc.setBuffer(invBuf, offset: 0, index: 2); enc.setBuffer(qAbsBuf, offset: 0, index: 3)
        enc.setBytes(&qbp, length: MemoryLayout<QBParams>.stride, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: kvLora / 128, height: qHeads, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        var ap = AttnParams(
            qHeads: UInt32(qHeads), nopeDim: UInt32(nopeDim), ropeDim: UInt32(ropeDim),
            kvLoraRank: UInt32(kvLora), newOffset: UInt32(newOffset), mRows: UInt32(M),
            maxCacheSize: UInt32(maxCache), isCausal: 0, softmaxScale: softmaxScale
        )
        enc.setComputePipelineState(psoAttn)
        enc.setBuffer(qAbsBuf, offset: 0, index: 0); enc.setBuffer(qFullBuf, offset: 0, index: 1)
        enc.setBuffer(kCacheBuf, offset: 0, index: 2); enc.setBuffer(vCacheBuf, offset: 0, index: 3)
        enc.setBuffer(propCKVBuf, offset: 0, index: 4); enc.setBuffer(propKPeBuf, offset: 0, index: 5)
        enc.setBuffer(uLatBuf, offset: 0, index: 6)
        enc.setBytes(&ap, length: MemoryLayout<AttnParams>.stride, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: qHeads, height: M, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        enc.setComputePipelineState(psoUV)
        enc.setBuffer(uLatBuf, offset: 0, index: 0); enc.setBuffer(kvBBuf, offset: 0, index: 1)
        enc.setBuffer(qkvBuf, offset: 0, index: 2); enc.setBuffer(attnOutBuf, offset: 0, index: 3)
        enc.setBytes(&qbp, length: MemoryLayout<QBParams>.stride, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: vDim / 32, height: qHeads, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        func cpuInterleavedRope(_ src: [UInt16], pos: Int) -> [UInt16] {
            var out = [UInt16](repeating: 0, count: ropeDim)
            for p in 0..<(ropeDim / 2) {
                let ang = Float(pos) * invFreq[p]
                let c = cosf(ang) * ropeMscale, s = sinf(ang) * ropeMscale
                let x0 = Float(Float16(bitPattern: src[2 * p])), x1 = Float(Float16(bitPattern: src[2 * p + 1]))
                out[2 * p] = Float16(x0 * c - x1 * s).bitPattern
                out[2 * p + 1] = Float16(x1 * c + x0 * s).bitPattern
            }
            return out
        }

        var refKCache = [UInt16](repeating: 0, count: maxCache * kvLora)
        var refVCache = [UInt16](repeating: 0, count: maxCache * ropeDim)
        for s in 0..<S {
            let pos = ctxOffset + s, slot = pos % maxCache, base = s * (kvLora + ropeDim)
            let cNorm = Self.cpuGatedNormRow(
                x: (0..<kvLora).map { Float(Float16(bitPattern: kvCtxIn[base + $0])) },
                normW: kvNW, gateUp: kvGU, gateDown: kvGD,
                eps: eps, gatingWeight: gatingW, scale: alphaKV, rank: normRank, zeroCentered: false, tptg: 128
            )
            for i in 0..<kvLora { refKCache[slot * kvLora + i] = Float16(cNorm[i]).bitPattern }
            let peOut = cpuInterleavedRope(Array(kvCtxIn[(base + kvLora)..<(base + kvLora + ropeDim)]), pos: pos)
            for i in 0..<ropeDim { refVCache[slot * ropeDim + i] = peOut[i] }
        }
        let gotKCache = kCacheBuf.contents().assumingMemoryBound(to: UInt16.self)
        let gotVCache = vCacheBuf.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<refKCache.count { XCTAssertEqual(gotKCache[i], refKCache[i]) }
        for i in 0..<refVCache.count { XCTAssertEqual(gotVCache[i], refVCache[i]) }

        let gotQLat = qLatBuf.contents().assumingMemoryBound(to: UInt16.self)
        let gotPropCKV = propCKVBuf.contents().assumingMemoryBound(to: UInt16.self)
        let gotPropKPe = propKPeBuf.contents().assumingMemoryBound(to: UInt16.self)
        let gotQAbs = qAbsBuf.contents().assumingMemoryBound(to: UInt16.self)
        var refPropCKV = [UInt16](repeating: 0, count: M * kvLora)
        var refPropKPe = [UInt16](repeating: 0, count: M * ropeDim)
        for m in 0..<M {
            let rowOff = m * qkvStride
            let qNorm = Self.cpuGatedNormRow(
                x: (0..<qLora).map { Float(Float16(bitPattern: qkvProp[rowOff + $0])) },
                normW: qNW, gateUp: qGU, gateDown: qGD,
                eps: eps, gatingWeight: gatingW, scale: alphaQ, rank: normRank, zeroCentered: false, tptg: 256
            )
            for i in 0..<qLora { XCTAssertEqual(gotQLat[m * qLora + i], Float16(qNorm[i]).bitPattern) }
            let cNorm = Self.cpuGatedNormRow(
                x: (0..<kvLora).map { Float(Float16(bitPattern: qkvProp[rowOff + qLora + $0])) },
                normW: kvNW, gateUp: kvGU, gateDown: kvGD,
                eps: eps, gatingWeight: gatingW, scale: alphaKV, rank: normRank, zeroCentered: false, tptg: 256
            )
            for i in 0..<kvLora {
                let bits = Float16(cNorm[i]).bitPattern
                refPropCKV[m * kvLora + i] = bits
                XCTAssertEqual(gotPropCKV[m * kvLora + i], bits)
            }
            let peOut = cpuInterleavedRope(Array(qkvProp[(rowOff + qLora + kvLora)..<(rowOff + qLora + kvLora + ropeDim)]), pos: newOffset + m)
            for i in 0..<ropeDim {
                refPropKPe[m * ropeDim + i] = peOut[i]
                XCTAssertEqual(gotPropKPe[m * ropeDim + i], peOut[i])
            }
        }

        var refQAbs = [UInt16](repeating: 0, count: M * qHeads * kvLora)
        var refQRope = [UInt16](repeating: 0, count: M * qHeads * ropeDim)
        for m in 0..<M {
            for h in 0..<qHeads {
                let ropeOff = m * qFullStride + qHeads * nopeDim + h * ropeDim
                let rOut = cpuInterleavedRope(Array(qFullInit[ropeOff..<(ropeOff + ropeDim)]), pos: newOffset + m)
                for i in 0..<ropeDim { refQRope[(m * qHeads + h) * ropeDim + i] = rOut[i] }
                let headRowBase = h * (nopeDim + vDim)
                for col in 0..<kvLora {
                    var acc: Float = 0
                    for r in 0..<nopeDim {
                        acc = fma(Float(Float16(bitPattern: qFullInit[m * qFullStride + h * nopeDim + r])),
                                  Self.bf16ToF32(kvB[(headRowBase + r) * kvLora + col]), acc)
                    }
                    let bits = Float16(acc).bitPattern
                    refQAbs[(m * qHeads + h) * kvLora + col] = bits
                    XCTAssertEqual(gotQAbs[(m * qHeads + h) * kvLora + col], bits)
                }
            }
        }

        let gotULat = uLatBuf.contents().assumingMemoryBound(to: UInt16.self)
        let gotAttnOut = attnOutBuf.contents().assumingMemoryBound(to: UInt16.self)
        for m in 0..<M {
            for h in 0..<qHeads {
                let absQ = newOffset + m, startK = absQ > maxCache ? (absQ - maxCache) : 0
                var maxLogit: Float = -3.4e38, lsum: Float = 0, hasLogit = false
                var oAcc = [Float](repeating: 0, count: kvLora)
                func stepKey(ckv: [UInt16], kpe: [UInt16]) {
                    var laneDots = [Float](repeating: 0, count: 32)
                    for lane in 0..<32 {
                        var d: Float = 0
                        for c in 0..<(kvLora / 128) {
                            for k in 0..<4 {
                                let idx = c * 128 + lane * 4 + k
                                d += Float(Float16(bitPattern: refQAbs[(m * qHeads + h) * kvLora + idx])) * softmaxScale
                                    * Float(Float16(bitPattern: ckv[idx]))
                            }
                        }
                        for r in 0..<(ropeDim / 32) {
                            let idx = r * 32 + lane
                            d += Float(Float16(bitPattern: refQRope[(m * qHeads + h) * ropeDim + idx])) * softmaxScale
                                * Float(Float16(bitPattern: kpe[idx]))
                        }
                        laneDots[lane] = d
                    }
                    let s = Self.cpuSimdSum32(laneDots)
                    let newMax = hasLogit ? max(maxLogit, s) : s
                    let corr = hasLogit ? expf(maxLogit - newMax) : 0.0, prob = expf(s - newMax)
                    lsum = lsum * corr + prob
                    for i in 0..<kvLora { oAcc[i] = oAcc[i] * corr + Float(Float16(bitPattern: ckv[i])) * prob }
                    maxLogit = newMax; hasLogit = true
                }
                for absK in startK..<newOffset {
                    let slot = absK % maxCache
                    stepKey(ckv: Array(refKCache[(slot * kvLora)..<((slot + 1) * kvLora)]),
                            kpe: Array(refVCache[(slot * ropeDim)..<((slot + 1) * ropeDim)]))
                }
                for pIdx in 0..<M {
                    stepKey(ckv: Array(refPropCKV[(pIdx * kvLora)..<((pIdx + 1) * kvLora)]),
                            kpe: Array(refPropKPe[(pIdx * ropeDim)..<((pIdx + 1) * ropeDim)]))
                }
                let invL = lsum > 0 ? (1.0 / lsum) : 0.0
                var refU = [Float](repeating: 0, count: kvLora)
                for i in 0..<kvLora {
                    let uBits = Float16(oAcc[i] * invL).bitPattern
                    XCTAssertEqual(gotULat[(m * qHeads + h) * kvLora + i], uBits)
                    refU[i] = Float(Float16(bitPattern: uBits))
                }
                let headRowBase = h * (nopeDim + vDim) + nopeDim
                let gateBase = m * qkvStride + qLora + kvLora + ropeDim + h * vDim
                for vBlk in 0..<(vDim / 32) {
                    var blockVals = [Float](repeating: 0, count: 32)
                    for lv in 0..<32 {
                        let row = headRowBase + vBlk * 32 + lv
                        var lanePart = [Float](repeating: 0, count: 16)
                        for sIdx in 0..<(kvLora / 32) {
                            var d: Float = 0
                            for k in 0..<32 {
                                let col = sIdx * 32 + k
                                d += Self.bf16ToF32(kvB[row * kvLora + col]) * refU[col]
                            }
                            lanePart[sIdx % 16] += d
                        }
                        var s = 1
                        while s <= 8 {
                            var nxt = lanePart
                            for i in 0..<16 { nxt[i] = lanePart[i] + lanePart[i ^ s] }
                            lanePart = nxt; s <<= 1
                        }
                        let g = Float(Float16(bitPattern: qkvProp[gateBase + vBlk * 32 + lv]))
                        blockVals[lv] = lanePart[0] * (1.0 / (1.0 + expf(-g)))
                    }
                    let rotVals = Self.cpuFWHT32(blockVals)
                    for lv in 0..<32 {
                        XCTAssertEqual(gotAttnOut[m * (qHeads * vDim) + h * vDim + vBlk * 32 + lv],
                                       Float16(rotVals[lv]).bitPattern)
                    }
                }
            }
        }
    }
}
