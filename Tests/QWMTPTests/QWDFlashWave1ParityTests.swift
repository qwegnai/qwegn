// QWDFlashWave1ParityTests.swift — Synthetic parity tests for unrotated DFlash
// capture (`FC_DRAFTER_ROTATED = false`) and dynamic `selectorRank` (`!= 256`).
import Foundation
import Metal
import XCTest
@testable import QWMTP
import QWCore
import QWHAL

final class QWDFlashWave1ParityTests: XCTestCase {
    private static func bf16Bits(_ x: Float) -> UInt16 {
        UInt16(truncatingIfNeeded: x.bitPattern >> 16)
    }

    private static func bf16ToF32(_ b: UInt16) -> Float {
        Float(bitPattern: UInt32(b) << 16)
    }

    private func sharedBuffer<T>(_ dev: any MTLDevice, _ values: [T]) throws -> any MTLBuffer {
        let bytes = max(16, values.count * MemoryLayout<T>.stride)
        let buf = try XCTUnwrap(dev.makeBuffer(length: bytes, options: .storageModeShared))
        let dst = buf.contents()
        withExtendedLifetime(buf) {
            values.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                if let base = raw.baseAddress, raw.count > 0 {
                    dst.copyMemory(from: base, byteCount: raw.count)
                }
            }
        }
        return buf
    }

    private func zeroBuffer(_ dev: any MTLDevice, byteCount: Int) throws -> any MTLBuffer {
        let buf = try XCTUnwrap(dev.makeBuffer(length: max(16, byteCount), options: .storageModeShared))
        memset(buf.contents(), 0, buf.length)
        return buf
    }

    func testCaptureTargetAndHCHiddenUnrotatedFC() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.supportsSimdgroupMMA == true,
            "Metal capture parity requires bare-metal Apple Silicon GPU"
        )
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)
        let dev = hal.device
        let psoTarget = try hal.pipeline("qwg_dflash_capture_target_hidden", fcDrafterRotated: false, fcHeadRotated: false)
        let psoHC = try hal.pipeline("qwg_dflash_capture_hc_hidden", fcDrafterRotated: false, fcHeadRotated: false)

        let H = 256, totalSlots = 3, slot = 1, srcRows = 3, dstRows = 4
        let srcOff = 1, dstOff = 1, rows = 2, hc = 4

        var srcF16 = [UInt16](repeating: 0, count: srcRows * H)
        for i in 0..<srcF16.count {
            srcF16[i] = Float16(Float(((i * 11 + 5) % 23) - 11) * 0.25).bitPattern
        }
        let srcBuf = try sharedBuffer(dev, srcF16)
        let dstTargetBuf = try zeroBuffer(dev, byteCount: dstRows * totalSlots * H * 2)

        var baseVals = [Float](repeating: 0, count: srcRows * hc * H)
        for i in 0..<baseVals.count {
            baseVals[i] = Float(((i * 13 + 3) % 19) - 9) * 0.125
        }
        var injSrc = [Float](repeating: 0, count: srcRows * H)
        for i in 0..<injSrc.count {
            injSrc[i] = Float(((i * 7 + 1) % 13) - 6) * 0.25
        }
        var injScales = [Float](repeating: 0, count: srcRows * hc)
        for i in 0..<injScales.count {
            injScales[i] = Float(((i * 3 + 2) % 5) + 1) * 0.25
        }
        let baseBuf = try sharedBuffer(dev, baseVals)
        let injSrcBuf = try sharedBuffer(dev, injSrc)
        let injScaleBuf = try sharedBuffer(dev, injScales)
        let dstHCMeanBuf = try zeroBuffer(dev, byteCount: dstRows * totalSlots * H * 2)
        let dstHCInjectBuf = try zeroBuffer(dev, byteCount: dstRows * totalSlots * H * 2)

        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        var sOffU = UInt32(srcOff), dOffU = UInt32(dstOff), slotU = UInt32(slot)
        var hU4 = UInt32(H / 8), tsU = UInt32(totalSlots), hcU = UInt32(hc)

        enc.setComputePipelineState(psoTarget)
        enc.setBuffer(srcBuf, offset: 0, index: 0)
        enc.setBuffer(dstTargetBuf, offset: 0, index: 1)
        enc.setBytes(&sOffU, length: 4, index: 2)
        enc.setBytes(&dOffU, length: 4, index: 3)
        enc.setBytes(&slotU, length: 4, index: 4)
        enc.setBytes(&hU4, length: 4, index: 5)
        enc.setBytes(&tsU, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        for (flagsVal, outBuf) in [(UInt32(0), dstHCMeanBuf), (UInt32(1), dstHCInjectBuf)] {
            var flagsU = flagsVal
            enc.setComputePipelineState(psoHC)
            enc.setBuffer(baseBuf, offset: 0, index: 0)
            enc.setBuffer(flagsVal == 0 ? baseBuf : injSrcBuf, offset: 0, index: 1)
            enc.setBuffer(flagsVal == 0 ? baseBuf : injScaleBuf, offset: 0, index: 2)
            enc.setBuffer(outBuf, offset: 0, index: 3)
            enc.setBytes(&sOffU, length: 4, index: 4)
            enc.setBytes(&dOffU, length: 4, index: 5)
            enc.setBytes(&slotU, length: 4, index: 6)
            enc.setBytes(&hU4, length: 4, index: 7)
            enc.setBytes(&tsU, length: 4, index: 8)
            enc.setBytes(&hcU, length: 4, index: 9)
            enc.setBytes(&flagsU, length: 4, index: 10)
            enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let tPtr = dstTargetBuf.contents().assumingMemoryBound(to: UInt16.self)
        let mPtr = dstHCMeanBuf.contents().assumingMemoryBound(to: UInt16.self)
        let iPtr = dstHCInjectBuf.contents().assumingMemoryBound(to: UInt16.self)
        for r in 0..<rows {
            let sRow = srcOff + r
            let dBase = (dstOff + r) * totalSlots * H + slot * H
            for h in 0..<H {
                XCTAssertEqual(tPtr[dBase + h], srcF16[sRow * H + h],
                               "unrotated target capture mismatch at r=\(r) h=\(h)")
                var sum0: Float = 0, sum1: Float = 0
                for c in 0..<hc {
                    let b = baseVals[(sRow * hc + c) * H + h]
                    sum0 += b
                    sum1 += b + injSrc[sRow * H + h] * injScales[sRow * hc + c]
                }
                XCTAssertEqual(mPtr[dBase + h], Float16(sum0 / Float(hc)).bitPattern,
                               "unrotated HC mean mismatch at r=\(r) h=\(h)")
                XCTAssertEqual(iPtr[dBase + h], Float16(sum1 / Float(hc)).bitPattern,
                               "unrotated HC inject mismatch at r=\(r) h=\(h)")
            }
        }
    }

    func testSelectorGreedyDynamicRank() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.supportsSimdgroupMMA == true,
            "Metal selector parity requires bare-metal Apple Silicon GPU"
        )
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)
        let dev = hal.device
        let psoSel = try hal.pipeline("qwg_dflash_selector_greedy")

        for rank in [128, 384] {
            let vocab = 6, topK = 3, D = 2, anchor = 1
            var unary = [Float](repeating: -10.0, count: D * 16)
            var cands = [Int32](repeating: -1, count: D * 16)
            unary[0] = 2.0; unary[1] = 1.5; unary[2] = 0.5
            cands[0] = 2; cands[1] = 3; cands[2] = 4
            unary[16 + 0] = 1.0; unary[16 + 1] = 1.2; unary[16 + 2] = 0.8
            cands[16 + 0] = 0; cands[16 + 1] = 4; cands[16 + 2] = 5

            var hProj = [Float](repeating: 0, count: D * rank)
            for p in 0..<D {
                for r in 0..<rank {
                    hProj[p * rank + r] = Float(((p * 37 + r * 13) % 19) - 9) * 0.05
                }
            }
            var pred = [UInt16](repeating: 0, count: vocab * rank)
            var succ = [UInt16](repeating: 0, count: vocab * rank)
            for v in 0..<vocab {
                for r in 0..<rank {
                    pred[v * rank + r] = Self.bf16Bits(Float(((v * 11 + r * 5) % 17) - 8) * 0.1)
                    succ[v * rank + r] = Self.bf16Bits(Float(((v * 7 + r * 3) % 15) - 7) * 0.1)
                }
            }

            // CPU greedy reference over exact rank.
            var expected = [Int]()
            var curPred = anchor
            for p in 0..<D {
                var bestScore: Float = -.greatestFiniteMagnitude
                var bestTok = 0
                for k in 0..<topK {
                    let cand = Int(cands[p * 16 + k])
                    var edge: Float = 0
                    for r in 0..<rank {
                        let pv = Self.bf16ToF32(pred[curPred * rank + r]) * hProj[p * rank + r]
                        let sv = Self.bf16ToF32(succ[cand * rank + r])
                        edge += pv * sv
                    }
                    let score = unary[p * 16 + k] + edge
                    if score > bestScore {
                        bestScore = score
                        bestTok = cand
                    }
                }
                expected.append(bestTok)
                curPred = bestTok
            }

            let hBuf = try sharedBuffer(dev, hProj)
            let uBuf = try sharedBuffer(dev, unary)
            let cBuf = try sharedBuffer(dev, cands)
            let pBuf = try sharedBuffer(dev, pred)
            let sBuf = try sharedBuffer(dev, succ)
            let outBuf = try zeroBuffer(dev, byteCount: D * 4)

            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            var anchorI = Int32(anchor), dU = UInt32(D), tkU = UInt32(topK)
            var vU = UInt32(vocab), capF: Float = 0, modeU: UInt32 = 0, rankU = UInt32(rank)
            enc.setComputePipelineState(psoSel)
            enc.setBuffer(hBuf, offset: 0, index: 0)
            enc.setBuffer(uBuf, offset: 0, index: 1)
            enc.setBuffer(cBuf, offset: 0, index: 2)
            enc.setBuffer(pBuf, offset: 0, index: 3)
            enc.setBuffer(sBuf, offset: 0, index: 4)
            enc.setBuffer(outBuf, offset: 0, index: 5)
            enc.setBytes(&anchorI, length: 4, index: 6)
            enc.setBytes(&dU, length: 4, index: 7)
            enc.setBytes(&tkU, length: 4, index: 8)
            enc.setBytes(&vU, length: 4, index: 9)
            enc.setBytes(&capF, length: 4, index: 10)
            enc.setBytes(&modeU, length: 4, index: 11)
            enc.setBytes(&rankU, length: 4, index: 12)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)

            let dPtr = outBuf.contents().assumingMemoryBound(to: Int32.self)
            XCTAssertEqual((0..<D).map { Int(dPtr[$0]) }, expected, "selector rank=\(rank) mismatch")
        }
    }
}
