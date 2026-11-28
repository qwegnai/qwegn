// QWDFlashKernelParityTests.swift — CPU-vs-GPU 0-ULP parity tests for generalized
// DFlash kernels (H=2560 HC capture, RMSNorm + mode 5, conv groups=160, dynamic top-16)
// and MoE/HC config discovery + adaptive policy tests.
import Darwin
import Foundation
import Metal
import XCTest
@testable import QWMTP
import QWCore
import QWHAL
import QWQuant

final class QWDFlashKernelParityTests: XCTestCase {
    private static func cpuFWHT32(_ block: [Float]) -> [Float] {
        precondition(block.count == 32)
        var s = block
        for lane in 0..<4 {
            let b = lane * 8
            for i in stride(from: 0, to: 8, by: 2) {
                let x = s[b + i], y = s[b + i + 1]
                s[b + i] = x + y; s[b + i + 1] = x - y
            }
            for i in stride(from: 0, to: 8, by: 4) {
                let a0 = s[b + i], a1 = s[b + i + 1], b0 = s[b + i + 2], b1 = s[b + i + 3]
                s[b + i] = a0 + b0; s[b + i + 1] = a1 + b1
                s[b + i + 2] = a0 - b0; s[b + i + 3] = a1 - b1
            }
            for i in 0..<4 {
                let x = s[b + i], y = s[b + i + 4]
                s[b + i] = x + y; s[b + i + 4] = x - y
            }
        }
        var s4 = s
        for lane in 0..<4 {
            let neg1 = (lane & 1) != 0
            for i in 0..<8 {
                let v = s[lane * 8 + i], p = s[(lane ^ 1) * 8 + i]
                s4[lane * 8 + i] = neg1 ? (p - v) : (v + p)
            }
        }
        let invSqrt32: Float = 0.17677669529663689
        var out = [Float](repeating: 0, count: 32)
        for lane in 0..<4 {
            let neg2 = (lane & 2) != 0
            for i in 0..<8 {
                let v = s4[lane * 8 + i], p = s4[(lane ^ 2) * 8 + i]
                out[lane * 8 + i] = (neg2 ? (p - v) : (v + p)) * invSqrt32
            }
        }
        return out
    }

    private func sharedBuffer<T>(_ dev: any MTLDevice, _ values: [T]) throws -> any MTLBuffer {
        let bytes = max(16, values.count * MemoryLayout<T>.stride)
        let buf = try XCTUnwrap(dev.makeBuffer(length: bytes, options: .storageModeShared))
        _ = values.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            memcpy(buf.contents(), raw.baseAddress!, raw.count)
        }
        return buf
    }

    private func zeroBuffer(_ dev: any MTLDevice, byteCount: Int) throws -> any MTLBuffer {
        let buf = try XCTUnwrap(dev.makeBuffer(length: max(16, byteCount), options: .storageModeShared))
        memset(buf.contents(), 0, buf.length)
        return buf
    }

    private func bf16Bits(_ x: Float) -> UInt16 {
        UInt16(truncatingIfNeeded: x.bitPattern >> 16)
    }

    private func f32FromBF16(_ b: UInt16) -> Float {
        Float(bitPattern: UInt32(b) << 16)
    }

    func testCaptureHCTargetHidden2560ExactParity() throws {
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)
        let dev = hal.device
        let pso = try hal.pipeline("qwg_dflash_capture_hc_hidden")

        let H = 2560
        let hc = 4
        let totalSlots = 5
        let srcRows = 4
        let dstRows = 5
        let srcRowOffset = 1
        let dstRowOffset = 2
        let slot = 3
        let activeRows = 2

        var baseVals = [Float](repeating: 0, count: srcRows * hc * H)
        for i in 0..<baseVals.count {
            baseVals[i] = Float(((i * 13 + 7) % 17) - 8) * 0.125
        }
        var srcF32 = [Float](repeating: 0, count: srcRows * H)
        for i in 0..<srcF32.count {
            srcF32[i] = Float(((i * 5 + 3) % 11) - 5) * 0.25
        }
        var injVals = [Float](repeating: 0, count: srcRows * hc)
        for i in 0..<injVals.count {
            injVals[i] = Float(((i * 3 + 1) % 7) - 3) * 0.5
        }
        var srcF16 = [UInt16](repeating: 0, count: srcRows * H)
        for i in 0..<srcF16.count {
            srcF16[i] = Float16(Float(((i * 7 + 1) % 9) - 4) * 0.25).bitPattern
        }
        var mhcCtrl = [Float](repeating: 0, count: srcRows * (hc + hc * hc))
        for r in 0..<srcRows {
            let off = r * (hc + hc * hc)
            for c in 0..<hc { mhcCtrl[off + c] = 0.25 * Float(c + 1) }
            for i in 0..<hc {
                for j in 0..<hc {
                    mhcCtrl[off + hc + i * hc + j] = (i == j) ? 0.5 : 0.125
                }
            }
        }

        let baseBuf = try sharedBuffer(dev, baseVals)
        let srcF32Buf = try sharedBuffer(dev, srcF32)
        let injBuf = try sharedBuffer(dev, injVals)
        let srcF16Buf = try sharedBuffer(dev, srcF16)
        let ctrlBuf = try sharedBuffer(dev, mhcCtrl)

        let cases: [(flags: UInt32, srcBuf: any MTLBuffer, injBuf: any MTLBuffer)] = [
            (0, baseBuf, baseBuf),
            (1, srcF32Buf, injBuf),
            (2, srcF16Buf, ctrlBuf)
        ]
        for (flags, sBuf, iBuf) in cases {
            let dstBuf = try zeroBuffer(dev, byteCount: dstRows * totalSlots * H * 2)
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(pso)
            enc.setBuffer(baseBuf, offset: 0, index: 0)
            enc.setBuffer(sBuf, offset: 0, index: 1)
            enc.setBuffer(iBuf, offset: 0, index: 2)
            enc.setBuffer(dstBuf, offset: 0, index: 3)
            var sOff = UInt32(srcRowOffset), dOff = UInt32(dstRowOffset), slotU = UInt32(slot)
            var hiddenU4 = UInt32(H / 8), totalSlotsU = UInt32(totalSlots)
            var hcU = UInt32(hc), flagsU = flags
            enc.setBytes(&sOff, length: 4, index: 4)
            enc.setBytes(&dOff, length: 4, index: 5)
            enc.setBytes(&slotU, length: 4, index: 6)
            enc.setBytes(&hiddenU4, length: 4, index: 7)
            enc.setBytes(&totalSlotsU, length: 4, index: 8)
            enc.setBytes(&hcU, length: 4, index: 9)
            enc.setBytes(&flagsU, length: 4, index: 10)
            enc.dispatchThreadgroups(MTLSize(width: activeRows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()

            let gotPtr = dstBuf.contents().assumingMemoryBound(to: UInt16.self)
            let invHC = Float(1.0) / Float(hc)
            for row in 0..<activeRows {
                let sRow = srcRowOffset + row
                let dRow = dstRowOffset + row
                var mean = [Float](repeating: 0, count: H)
                for h in 0..<H {
                    if flags == 2 {
                        let b = Float(Float16(bitPattern: srcF16[sRow * H + h]))
                        let ctrlOff = sRow * (hc + hc * hc)
                        var sumS: Float = 0
                        for i in 0..<hc {
                            var ns = mhcCtrl[ctrlOff + i] * b
                            for j in 0..<hc {
                                ns = fma(mhcCtrl[ctrlOff + hc + i * hc + j], baseVals[(sRow * hc + j) * H + h], ns)
                            }
                            sumS += ns
                        }
                        mean[h] = sumS * invHC
                    } else if flags == 1 {
                        var sum: Float = 0
                        for c in 0..<hc {
                            sum += baseVals[(sRow * hc + c) * H + h] + srcF32[sRow * H + h] * injVals[sRow * hc + c]
                        }
                        mean[h] = sum * invHC
                    } else {
                        var sum: Float = 0
                        for c in 0..<hc { sum += baseVals[(sRow * hc + c) * H + h] }
                        mean[h] = sum * invHC
                    }
                }
                let dstBase = (dRow * totalSlots + slot) * H
                for blk in 0..<(H / 32) {
                    let ref = Self.cpuFWHT32(Array(mean[(blk * 32)..<((blk + 1) * 32)]))
                    for k in 0..<32 {
                        let idx = blk * 32 + k
                        let refBits = Float16(ref[k]).bitPattern
                        XCTAssertEqual(gotPtr[dstBase + idx], refBits,
                                       "HC capture flag \(flags) row \(row) idx \(idx) mismatch")
                    }
                }
            }
        }
    }

    func testStagePropBF16Parity() throws {
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)
        let dev = hal.device
        let pso = try hal.pipeline("qwg_dflash_stage_prop_bf16")

        let H = 2560
        let V = 32
        let rows = 6
        let anchorToken = 19
        let maskToken = 7
        var table = [UInt16](repeating: 0, count: V * H)
        for t in 0..<V {
            for h in 0..<H {
                table[t * H + h] = bf16Bits(Float(((t * 17 + h * 3) % 29) - 14) * 0.25)
            }
        }
        let tableBuf = try sharedBuffer(dev, table)
        let outBuf = try zeroBuffer(dev, byteCount: rows * H * 4)

        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        enc.setComputePipelineState(pso)
        enc.setBuffer(tableBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var aI = Int32(anchorToken), mI = Int32(maskToken), hU = UInt32(H), vU = UInt32(V)
        enc.setBytes(&aI, length: 4, index: 2)
        enc.setBytes(&mI, length: 4, index: 3)
        enc.setBytes(&hU, length: 4, index: 4)
        enc.setBytes(&vU, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: (H + 255) / 256, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let outPtr = outBuf.contents().assumingMemoryBound(to: UInt32.self)
        for r in 0..<rows {
            let tok = (r == 0) ? anchorToken : maskToken
            for h in 0..<H {
                let refBits = f32FromBF16(table[tok * H + h]).bitPattern
                XCTAssertEqual(outPtr[r * H + h], refBits, "stage_prop_bf16 row \(r) h \(h) mismatch")
            }
        }

        // Negative / out-of-range tokens must clamp cleanly to token 0.
        let clampOutBuf = try zeroBuffer(dev, byteCount: 2 * H * 4)
        let cbClamp = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let encClamp = try XCTUnwrap(cbClamp.makeComputeCommandEncoder())
        var badAnchor: Int32 = -7, badMask = Int32(V + 11)
        encClamp.setComputePipelineState(pso)
        encClamp.setBuffer(tableBuf, offset: 0, index: 0)
        encClamp.setBuffer(clampOutBuf, offset: 0, index: 1)
        encClamp.setBytes(&badAnchor, length: 4, index: 2)
        encClamp.setBytes(&badMask, length: 4, index: 3)
        encClamp.setBytes(&hU, length: 4, index: 4)
        encClamp.setBytes(&vU, length: 4, index: 5)
        encClamp.dispatchThreadgroups(MTLSize(width: (H + 255) / 256, height: 2, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encClamp.endEncoding()
        cbClamp.commit()
        cbClamp.waitUntilCompleted()
        let clampPtr = clampOutBuf.contents().assumingMemoryBound(to: UInt32.self)
        for r in 0..<2 {
            for h in 0..<H {
                XCTAssertEqual(clampPtr[r * H + h], f32FromBF16(table[h]).bitPattern)
            }
        }
    }

    func testGeneralizedRMSNormH2560AndMode5Parity() throws {
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)
        let dev = hal.device
        let pso = try hal.pipeline("qwg_dflash_rmsnorm")

        let H = 2560
        let rows = 2
        // Choose exact +/-2.0 pattern per row so meanSq = 4.0 and rstd = 0.5 exactly (0 ULP).
        var xF32 = [Float](repeating: 0, count: rows * H)
        var gamma = [UInt16](repeating: 0, count: H)
        for r in 0..<rows {
            for h in 0..<H {
                xF32[r * H + h] = ((h + r) % 2 == 0) ? 2.0 : -2.0
            }
        }
        for h in 0..<H {
            gamma[h] = Float16(Float(((h * 5 + 1) % 9) - 4) * 0.25).bitPattern
        }

        let xBuf = try sharedBuffer(dev, xF32)
        let gBuf = try sharedBuffer(dev, gamma)
        let yBuf = try zeroBuffer(dev, byteCount: rows * H * 2)
        let yRotBuf = try zeroBuffer(dev, byteCount: rows * H * 2)
        let yRot32Buf = try zeroBuffer(dev, byteCount: rows * H * 4)

        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        enc.setComputePipelineState(pso)
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(gBuf, offset: 0, index: 1)
        enc.setBuffer(yBuf, offset: 0, index: 2)
        enc.setBuffer(yRotBuf, offset: 0, index: 3)
        var eps: Float = 0.0, modeU: UInt32 = 5, hiddenU = UInt32(H)
        enc.setBytes(&eps, length: 4, index: 4)
        enc.setBytes(&modeU, length: 4, index: 5)
        enc.setBytes(&hiddenU, length: 4, index: 6)
        enc.setBuffer(yRot32Buf, offset: 0, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let yPtr = yBuf.contents().assumingMemoryBound(to: UInt16.self)
        let yRotPtr = yRotBuf.contents().assumingMemoryBound(to: UInt16.self)
        let yRot32Ptr = yRot32Buf.contents().assumingMemoryBound(to: UInt32.self)

        for r in 0..<rows {
            var normed = [Float](repeating: 0, count: H)
            for h in 0..<H {
                normed[h] = xF32[r * H + h] * 0.5 * Float(Float16(bitPattern: gamma[h]))
                XCTAssertEqual(yPtr[r * H + h], Float16(normed[h]).bitPattern, "y mismatch row \(r) h \(h)")
            }
            for blk in 0..<(H / 32) {
                let refRot = Self.cpuFWHT32(Array(normed[(blk * 32)..<((blk + 1) * 32)]))
                for k in 0..<32 {
                    let idx = blk * 32 + k
                    XCTAssertEqual(yRotPtr[r * H + idx], Float16(refRot[k]).bitPattern, "yRot mismatch row \(r) idx \(idx)")
                    XCTAssertEqual(yRot32Ptr[r * H + idx], refRot[k].bitPattern, "yRot32 mismatch row \(r) idx \(idx)")
                }
            }
        }

        // Standard basis leaves yRot and yRot32 unrotated.
        let stdHal = try QWHal(geometry: .qwen38_27b, weightBasis: .standard)
        let stdPso = try stdHal.pipeline("qwg_dflash_rmsnorm")
        let stdYRot32 = try zeroBuffer(stdHal.device, byteCount: rows * H * 4)
        let cbStd = try XCTUnwrap(stdHal.queue.makeCommandBuffer())
        let encStd = try XCTUnwrap(cbStd.makeComputeCommandEncoder())
        encStd.setComputePipelineState(stdPso)
        encStd.setBuffer(xBuf, offset: 0, index: 0)
        encStd.setBuffer(gBuf, offset: 0, index: 1)
        encStd.setBuffer(yBuf, offset: 0, index: 2)
        encStd.setBuffer(yRotBuf, offset: 0, index: 3)
        encStd.setBytes(&eps, length: 4, index: 4)
        encStd.setBytes(&modeU, length: 4, index: 5)
        encStd.setBytes(&hiddenU, length: 4, index: 6)
        encStd.setBuffer(stdYRot32, offset: 0, index: 7)
        encStd.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encStd.endEncoding()
        cbStd.commit()
        cbStd.waitUntilCompleted()

        let stdPtr = stdYRot32.contents().assumingMemoryBound(to: UInt32.self)
        for r in 0..<rows {
            for h in 0..<H {
                let ref = xF32[r * H + h] * 0.5 * Float(Float16(bitPattern: gamma[h]))
                XCTAssertEqual(stdPtr[r * H + h], ref.bitPattern, "standard yRot32 mismatch")
            }
        }
    }

    func testGeneralizedConvApplyGroups160Parity() throws {
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)
        let dev = hal.device
        let pso = try hal.pipeline("qwg_dflash_conv_apply")

        let H = 2560
        let convGroups = 160
        let M = 4
        var xVals = [UInt16](repeating: 0, count: M * H)
        var baseVals = [UInt16](repeating: 0, count: 4 * H)
        var dynVals = [Float](repeating: 0, count: M * 4 * convGroups)
        var resInit = [Float](repeating: 0, count: M * H)

        for i in 0..<xVals.count { xVals[i] = Float16(Float(((i * 7) % 13) - 6) * 0.125).bitPattern }
        for i in 0..<baseVals.count { baseVals[i] = Float16(Float(((i * 3 + 1) % 9) - 4) * 0.25).bitPattern }
        for i in 0..<dynVals.count { dynVals[i] = Float(((i * 5 + 2) % 11) - 5) * 0.125 }
        for i in 0..<resInit.count { resInit[i] = Float(((i * 11) % 7) - 3) * 0.5 }

        let xBuf = try sharedBuffer(dev, xVals)
        let dynBuf = try sharedBuffer(dev, dynVals)
        let baseBuf = try sharedBuffer(dev, baseVals)
        let out0Buf = try zeroBuffer(dev, byteCount: M * H * 2)
        let res1Buf = try sharedBuffer(dev, resInit)

        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        var hiddenU4 = UInt32(H / 8), groupsU = UInt32(convGroups)
        var mode0: UInt32 = 0, mode1: UInt32 = 1

        enc.setComputePipelineState(pso)
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(dynBuf, offset: 0, index: 1)
        enc.setBuffer(baseBuf, offset: 0, index: 2)
        enc.setBuffer(out0Buf, offset: 0, index: 3)
        enc.setBytes(&mode0, length: 4, index: 4)
        enc.setBytes(&hiddenU4, length: 4, index: 5)
        enc.setBytes(&groupsU, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: M, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        enc.setComputePipelineState(pso)
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(dynBuf, offset: 0, index: 1)
        enc.setBuffer(baseBuf, offset: 0, index: 2)
        enc.setBuffer(res1Buf, offset: 0, index: 3)
        enc.setBytes(&mode1, length: 4, index: 4)
        enc.setBytes(&hiddenU4, length: 4, index: 5)
        enc.setBytes(&groupsU, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: M, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let got0 = out0Buf.contents().assumingMemoryBound(to: UInt16.self)
        let got1 = res1Buf.contents().assumingMemoryBound(to: UInt32.self)
        let groupSize = H / convGroups
        for t in 0..<M {
            var raw0 = [Float](repeating: 0, count: H)
            for h in 0..<H {
                let g = h / groupSize
                let xc = Float(Float16(bitPattern: xVals[t * H + h]))
                let xp = t > 0 ? Float(Float16(bitPattern: xVals[(t - 1) * H + h])) : 0
                let d0Off = t * 4 * convGroups
                let w0_0 = Float(Float16(bitPattern: baseVals[0 * H + h])) + dynVals[d0Off + g]
                let w0_1 = Float(Float16(bitPattern: baseVals[1 * H + h])) + dynVals[d0Off + convGroups + g]
                raw0[h] = xc * w0_0 + (t > 0 ? xp * w0_1 : 0)

                let d1Off = d0Off + 2 * convGroups
                let w1_0 = Float(Float16(bitPattern: baseVals[2 * H + h])) + dynVals[d1Off + g]
                let w1_1 = Float(Float16(bitPattern: baseVals[3 * H + h])) + dynVals[d1Off + convGroups + g]
                let ref1 = resInit[t * H + h] + (xc * w1_0 + (t > 0 ? xp * w1_1 : 0))
                XCTAssertEqual(got1[t * H + h], ref1.bitPattern, "conv mode 1 t \(t) h \(h) mismatch")
            }
            for blk in 0..<(H / 32) {
                let rot0 = Self.cpuFWHT32(Array(raw0[(blk * 32)..<((blk + 1) * 32)]))
                for k in 0..<32 {
                    let idx = blk * 32 + k
                    XCTAssertEqual(got0[t * H + idx], Float16(rot0[k]).bitPattern,
                                   "conv mode 0 t \(t) idx \(idx) mismatch")
                }
            }
        }
    }

    func testDynamicVocabTop16Stage1And2Parity() throws {
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)
        let dev = hal.device
        let psoS1 = try hal.pipeline("qwg_dflash_top16_stage1")
        let psoS2 = try hal.pipeline("qwg_dflash_top16_stage2")

        let D = 3
        let topK = 16
        for vocab in [65_536, 248_320, 393_216] {
            var logits = [Float](repeating: -20.0, count: D * vocab)
            var expected = [[(id: Int, val: Float)]](repeating: [], count: D)
            for r in 0..<D {
                // Place distinct peaks across chunk boundaries + ties in id order.
                var chosen: [Int: Float] = [:]
                for rank in 0..<24 {
                    let id = min(vocab - 1, ((rank * 15_731 + r * 997 + 13) % vocab))
                    let val: Float = (rank < 2) ? 50.0 : Float(24 - rank) * 1.25 + Float(r) * 0.0625
                    if chosen[id] == nil {
                        chosen[id] = val
                        logits[r * vocab + id] = val
                    }
                }
                expected[r] = Array(chosen.map { (id: $0.key, val: $0.value) }.sorted {
                    if $0.val != $1.val { return $0.val > $1.val }
                    return $0.id < $1.id
                }.prefix(topK))
            }

            let logBuf = try sharedBuffer(dev, logits)
            let pValBuf = try zeroBuffer(dev, byteCount: D * 256 * 16 * 4)
            let pIdBuf = try zeroBuffer(dev, byteCount: D * 256 * 16 * 4)
            let uBuf = try zeroBuffer(dev, byteCount: D * 16 * 4)
            let cBuf = try zeroBuffer(dev, byteCount: D * 16 * 4)

            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            var vocabU = UInt32(vocab), topKU = UInt32(topK)
            enc.setComputePipelineState(psoS1)
            enc.setBuffer(logBuf, offset: 0, index: 0)
            enc.setBuffer(pValBuf, offset: 0, index: 1)
            enc.setBuffer(pIdBuf, offset: 0, index: 2)
            enc.setBytes(&vocabU, length: 4, index: 3)
            enc.setBytes(&topKU, length: 4, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: 256, height: D, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            enc.setComputePipelineState(psoS2)
            enc.setBuffer(pValBuf, offset: 0, index: 0)
            enc.setBuffer(pIdBuf, offset: 0, index: 1)
            enc.setBuffer(uBuf, offset: 0, index: 2)
            enc.setBuffer(cBuf, offset: 0, index: 3)
            enc.setBytes(&topKU, length: 4, index: 4)
            enc.setBytes(&vocabU, length: 4, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: D, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()

            let uPtr = uBuf.contents().assumingMemoryBound(to: UInt32.self)
            let cPtr = cBuf.contents().assumingMemoryBound(to: Int32.self)
            for r in 0..<D {
                for k in 0..<topK {
                    let ref = expected[r][k]
                    XCTAssertEqual(Int(cPtr[r * 16 + k]), ref.id, "top16 id vocab=\(vocab) r=\(r) k=\(k)")
                    XCTAssertEqual(uPtr[r * 16 + k], ref.val.bitPattern, "top16 val vocab=\(vocab) r=\(r) k=\(k)")
                }
            }
        }
    }

    func testConfigDiscoveryAndSparseMoEAdaptivePolicy() throws {
        let tmpRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg_dflash_disc_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpRoot) }

        let packDir = tmpRoot.appendingPathComponent("Qwen3.8-Flash-Next-DFlash2-shvq3g64", isDirectory: true)
        let assetsDir = packDir.appendingPathComponent("assets", isDirectory: true)
        try FileManager.default.createDirectory(at: assetsDir, withIntermediateDirectories: true)

        let cfgJSON = """
        {
          "architectures": ["DFlash2DraftModel"],
          "model_type": "qwen3",
          "hidden_size": 2560,
          "intermediate_size": 7680,
          "num_hidden_layers": 5,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "vocab_size": 248320,
          "block_size": 8,
          "dflash_config": {
            "mask_token_id": 248077,
            "target_layer_ids": [5, 15, 25, 35, 45]
          }
        }
        """
        try cfgJSON.write(to: assetsDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        let hdrJSON = """
        {
          "provenance": {"source": "unit-test"},
          "tensors": {
            "fc.weight": {"shape": [2560, 12800], "hq4_off": 64, "hq": {"bits": 3, "group": 64, "vq": "shvq1"}},
            "hidden_norm.weight": {"shape": [2560], "raw_off": 1024},
            "norm.weight": {"shape": [2560], "raw_off": 2048},
            "layers.0.self_attn.q_proj.weight": {"shape": [4096, 2560], "hq4_off": 4096, "hq": {"bits": 3, "group": 64, "vq": "shvq1"}}
          }
        }
        """
        var qwp = Data("QWP1".utf8)
        let hdrBytes = Data(hdrJSON.utf8)
        let len = UInt64(hdrBytes.count)
        for shift in stride(from: 0, to: 64, by: 8) {
            qwp.append(UInt8(truncatingIfNeeded: len >> shift))
        }
        qwp.append(Data(count: 64))
        qwp.append(hdrBytes)
        try qwp.write(to: packDir.appendingPathComponent("model.qwp"))

        let moeHCDesc = QWModelDescriptor(
            family: .qwen38FlashNext, modelType: "qwen4_exp",
            capabilities: [.moe],
            hidden: 2560, vocab: 248320,
            denseIntermediate: nil, numLayers: 48,
            layerTypes: (0..<48).map { ($0 + 1) % 4 == 0 ? "full_attention" : "linear_attention" },
            fullAttentionInterval: 4, rmsEps: 1e-6,
            maxPositionEmbeddings: 262_144,
            qHeads: 24, kvHeads: 2, headDim: 256,
            attnOutputGate: false, partialRotaryFactor: 0.25, ropeTheta: 1e7,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: 512, expertsPerTok: 10, moeIntermediate: 640,
            sharedExpertIntermediate: 640, expertGateType: nil,
            hcCount: 4, hcLowrank: 320,
            gdnQKHeads: 16, gdnVHeads: 48, gdnHeadDim: 128,
            gdnQKVChannels: 10_240, gdnZChannels: 6144, gdnConvKernel: 4,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 1, mtpUseDedicatedEmbeddings: false, eosTokenId: 248_044
        )
        let discovered = QWDFlashConfig.defaultPackPath(for: moeHCDesc, modelsDirectory: tmpRoot)
        XCTAssertEqual(
            URL(fileURLWithPath: discovered).resolvingSymlinksInPath().path,
            packDir.resolvingSymlinksInPath().path
        )
        XCTAssertNoThrow(try QWDFlashConfig.validateDrafter(
            targetDescriptor: moeHCDesc, packPath: discovered, drafts: 7, selectorTopK: 16
        ))

        var policy = QWDFlashAdaptivePolicy(isSparseMoE: true)
        XCTAssertEqual(policy.selectDraftCount(maxDrafts: 7), 3)
        policy.observeRound(draftCount: 3, accepted: 1, roundMs: 31.0)
        XCTAssertTrue([2, 3, 5, 7].contains(policy.selectDraftCount(maxDrafts: 7)))
        XCTAssertEqual(QWDFlashDrafter.siluThreadgroups(i: 7680), 8)
        XCTAssertEqual(QWDFlashDrafter.rowPairedMinimumRows, 4)
    }

    func testGeneralizedRoPEAndAttn16Heads4KVParity() throws {
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)
        let dev = hal.device
        let psoKVCtx = try hal.pipeline("qwg_dflash_rope_kv_ctx")
        let psoQKProp = try hal.pipeline("qwg_dflash_rope_qk_prop")
        let psoAttn = try hal.pipeline("qwg_dflash_attn")

        let qHeads = 16
        let kvHeads = 4
        let headDim = 128
        let qDim = qHeads * headDim
        let kvDim = kvHeads * headDim
        let qkvStride = qDim + 2 * kvDim
        let maxCache = 8
        let S = 2
        let M = 2
        let cacheOffset = 3

        var kvCtxIn = [UInt16](repeating: 0, count: S * 2 * kvDim)
        for i in 0..<kvCtxIn.count { kvCtxIn[i] = Float16(Float(((i * 5 + 1) % 7) - 3) * 0.25).bitPattern }
        var qkvProp = [UInt16](repeating: 0, count: M * qkvStride)
        for i in 0..<qkvProp.count { qkvProp[i] = Float16(Float(((i * 3 + 2) % 9) - 4) * 0.125).bitPattern }
        let wOne = [UInt16](repeating: Float16(1.0).bitPattern, count: headDim)
        var invFreq = [Float](repeating: 0, count: headDim / 2)
        for i in 0..<(headDim / 2) { invFreq[i] = 1.0 / powf(10_000.0, Float(2 * i) / Float(headDim)) }

        let kvCtxBuf = try sharedBuffer(dev, kvCtxIn)
        let wkBuf = try sharedBuffer(dev, wOne)
        let wqBuf = try sharedBuffer(dev, wOne)
        let invFreqBuf = try sharedBuffer(dev, invFreq)
        let kCacheBuf = try zeroBuffer(dev, byteCount: maxCache * kvDim * 2)
        let vCacheBuf = try zeroBuffer(dev, byteCount: maxCache * kvDim * 2)
        let qkvBuf = try sharedBuffer(dev, qkvProp)
        let attnOutBuf = try zeroBuffer(dev, byteCount: M * qDim * 2)

        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        var cOffU = UInt32(0), epsF: Float = 1e-6, maxCacheU = UInt32(maxCache)
        var qHeadsU = UInt32(qHeads), kvHeadsU = UInt32(kvHeads)
        enc.setComputePipelineState(psoKVCtx)
        enc.setBuffer(kvCtxBuf, offset: 0, index: 0)
        enc.setBuffer(wkBuf, offset: 0, index: 1)
        enc.setBuffer(invFreqBuf, offset: 0, index: 2)
        enc.setBuffer(kCacheBuf, offset: 0, index: 3)
        enc.setBuffer(vCacheBuf, offset: 0, index: 4)
        enc.setBytes(&cOffU, length: 4, index: 5)
        enc.setBytes(&epsF, length: 4, index: 6)
        enc.setBytes(&maxCacheU, length: 4, index: 7)
        enc.setBytes(&kvHeadsU, length: 4, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: S, height: kvHeads, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        var pOffU = UInt32(cacheOffset)
        enc.setComputePipelineState(psoQKProp)
        enc.setBuffer(qkvBuf, offset: 0, index: 0)
        enc.setBuffer(wqBuf, offset: 0, index: 1)
        enc.setBuffer(wkBuf, offset: 0, index: 2)
        enc.setBuffer(invFreqBuf, offset: 0, index: 3)
        enc.setBytes(&pOffU, length: 4, index: 4)
        enc.setBytes(&epsF, length: 4, index: 5)
        enc.setBytes(&qHeadsU, length: 4, index: 6)
        enc.setBytes(&kvHeadsU, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: M, height: qHeads + kvHeads, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        var newOffU = UInt32(S), mU = UInt32(M), isCausalU = UInt32(0)
        enc.setComputePipelineState(psoAttn)
        enc.setBuffer(qkvBuf, offset: 0, index: 0)
        enc.setBuffer(kCacheBuf, offset: 0, index: 1)
        enc.setBuffer(vCacheBuf, offset: 0, index: 2)
        enc.setBuffer(attnOutBuf, offset: 0, index: 3)
        enc.setBytes(&newOffU, length: 4, index: 4)
        enc.setBytes(&mU, length: 4, index: 5)
        enc.setBytes(&maxCacheU, length: 4, index: 6)
        enc.setBytes(&isCausalU, length: 4, index: 7)
        enc.setBytes(&qHeadsU, length: 4, index: 8)
        enc.setBytes(&kvHeadsU, length: 4, index: 9)
        enc.dispatchThreadgroups(MTLSize(width: qHeads, height: M, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        // Verify V cache slots 0..<S and finite non-zero attention outputs across all 16 heads.
        let vPtr = vCacheBuf.contents().assumingMemoryBound(to: UInt16.self)
        for s in 0..<S {
            for i in 0..<kvDim {
                XCTAssertEqual(vPtr[s * kvDim + i], kvCtxIn[s * 2 * kvDim + kvDim + i])
            }
        }
        let outPtr = attnOutBuf.contents().assumingMemoryBound(to: UInt16.self)
        var nonZero = 0
        for i in 0..<(M * qDim) {
            let v = Float(Float16(bitPattern: outPtr[i]))
            XCTAssertTrue(v.isFinite)
            if v != 0 { nonZero += 1 }
        }
        XCTAssertGreaterThan(nonZero, M * qDim / 2)
    }

    func testTop16SelectorNaNAndCandidateBoundsHardening() throws {
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)
        let dev = hal.device
        let psoS1 = try hal.pipeline("qwg_dflash_top16_stage1")
        let psoS2 = try hal.pipeline("qwg_dflash_top16_stage2")
        let psoSel = try hal.pipeline("qwg_dflash_selector_greedy")

        let vocab = 8, topK = 16, D = 2
        var logits = [Float](repeating: -.infinity, count: D * vocab)
        logits[0] = .nan; logits[3] = 4.5; logits[5] = .infinity; logits[vocab + 2] = 2.0
        let logBuf = try sharedBuffer(dev, logits)
        let pValBuf = try zeroBuffer(dev, byteCount: D * 256 * 16 * 4)
        let pIdBuf = try zeroBuffer(dev, byteCount: D * 256 * 16 * 4)
        let uBuf = try zeroBuffer(dev, byteCount: D * 16 * 4)
        let cBuf = try zeroBuffer(dev, byteCount: D * 16 * 4)
        let hProjBuf = try sharedBuffer(dev, [Float](repeating: 0.1, count: D * 256))
        let bookBuf = try sharedBuffer(dev, [UInt16](repeating: bf16Bits(0.25), count: vocab * 256))
        let outTokBuf = try zeroBuffer(dev, byteCount: D * 4)

        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        var vocabU = UInt32(vocab), topKU = UInt32(topK), badAnchor: Int32 = -1, draftU = UInt32(D)
        var softcapF: Float = 0.0, softcapModeU: UInt32 = 0, rankU = UInt32(256)
        enc.setComputePipelineState(psoS1)
        enc.setBuffer(logBuf, offset: 0, index: 0)
        enc.setBuffer(pValBuf, offset: 0, index: 1)
        enc.setBuffer(pIdBuf, offset: 0, index: 2)
        enc.setBytes(&vocabU, length: 4, index: 3)
        enc.setBytes(&topKU, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 256, height: D, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.setComputePipelineState(psoS2)
        enc.setBuffer(pValBuf, offset: 0, index: 0)
        enc.setBuffer(pIdBuf, offset: 0, index: 1)
        enc.setBuffer(uBuf, offset: 0, index: 2)
        enc.setBuffer(cBuf, offset: 0, index: 3)
        enc.setBytes(&topKU, length: 4, index: 4)
        enc.setBytes(&vocabU, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: D, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.setComputePipelineState(psoSel)
        enc.setBuffer(hProjBuf, offset: 0, index: 0)
        enc.setBuffer(uBuf, offset: 0, index: 1)
        enc.setBuffer(cBuf, offset: 0, index: 2)
        enc.setBuffer(bookBuf, offset: 0, index: 3)
        enc.setBuffer(bookBuf, offset: 0, index: 4)
        enc.setBuffer(outTokBuf, offset: 0, index: 5)
        enc.setBytes(&badAnchor, length: 4, index: 6)
        enc.setBytes(&draftU, length: 4, index: 7)
        enc.setBytes(&topKU, length: 4, index: 8)
        enc.setBytes(&vocabU, length: 4, index: 9)
        enc.setBytes(&softcapF, length: 4, index: 10)
        enc.setBytes(&softcapModeU, length: 4, index: 11)
        enc.setBytes(&rankU, length: 4, index: 12)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let uPtr = uBuf.contents().assumingMemoryBound(to: Float.self)
        let cPtr = cBuf.contents().assumingMemoryBound(to: Int32.self)
        let dPtr = outTokBuf.contents().assumingMemoryBound(to: Int32.self)
        XCTAssertEqual(cPtr[0], 3)
        XCTAssertEqual(cPtr[16], 2)
        for i in 0..<(D * 16) {
            XCTAssertTrue(uPtr[i].isFinite)
            XCTAssertTrue((0..<vocab).contains(Int(cPtr[i])))
        }
        XCTAssertEqual(dPtr[0], 3)
        XCTAssertEqual(dPtr[1], 2)

        XCTAssertEqual(QWDFlashDrafter.clampToken(-3, vocab: 16), 0)
        XCTAssertEqual(QWDFlashDrafter.clampToken(0x7fff_ffff, vocab: 16), 0)
        XCTAssertEqual(QWDFlashDrafter.clampToken(5, vocab: 16), 5)
        let dist = try QWDFlashDrafter.buildSparseDistribution(
            vocab: 16,
            candidates: [(-1, 9.0), (0x7fff_ffff, 9.0), (4, .nan), (7, 2.0)],
            temperature: 0.7)
        XCTAssertEqual(dist.survivorIds, [7])
        let allNonFinite = try QWDFlashDrafter.buildSparseDistribution(
            vocab: 16, candidates: [(4, .nan), (7, -.infinity)], temperature: 0.7)
        XCTAssertEqual(allNonFinite.survivorIds, [4, 7])
        let emptyDist = try QWDFlashDrafter.buildSparseDistribution(
            vocab: 16, candidates: [(-1, 1.0), (99, 2.0)], temperature: 0.7)
        XCTAssertEqual(emptyDist.survivorIds, [0])
    }
}
