// QWInkling2DSparkParityTests.swift — GPU vs Float64 CPU parity for Inkling-2 DSpark kernels & conv rollback.
import Foundation
import Metal
import XCTest
import QWCore
@testable import QWEngine
import QWHAL

private extension QWInkling2Session {
    func snapshotPreVerifyConvStates() {
        guard let preBuf = convPreVerifyBuf else { return }
        memcpy(preBuf.contents(), convStatesSlab.contents(), convStatesSlab.length)
    }
}

final class QWInkling2DSparkParityTests: XCTestCase {
    private typealias R = QWInk2TestRef

    private struct MixP { var T, H, tapPlaneStride: UInt32; var eps, a0, a1, a2, a3: Float }
    private struct CtxKVP { var T, nKV, headDim, isSliding, posOff, ringCap: UInt32; var eps, ropeTheta: Float }
    private struct AddNormP { var rows, dim, hasRes, writeRes: UInt32; var eps: Float }
    private struct PropQKRP { var B, nH, nKV, headDim, posOff: UInt32; var eps, ropeTheta: Float }
    private struct AttnP { var B, nH, nKV, headDim, ctxStart, ctxEnd, sliding, ringCap: UInt32; var scale: Float }
    private struct MarkovCtxP { var count, H, projRows, rank: UInt32 }
    private struct MarkovStepP { var stepIdx, H, projRows, rank, resetState: UInt32 }
    private struct LogitsP { var stepIdx, V, rank, partCount: UInt32; var softcap: Float }
    private struct SConvVerifyCommitP {
        var numVisits, kvDim, H, kept, stateKvStride, stateHStride, verifyKvStride, verifyHStride: UInt32
    }

    private func makeHAL() throws -> QWHal {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal device required")
        return try QWHal(geometry: QWModelGeometry.qwen38_27b, weightBasis: .standard)
    }

    private func buf<T>(_ hal: QWHal, _ array: [T]) -> any MTLBuffer {
        let len = max(256, array.count * MemoryLayout<T>.stride)
        let b = hal.device.makeBuffer(length: len, options: .storageModeShared)!
        if !array.isEmpty { _ = array.withUnsafeBytes { memcpy(b.contents(), $0.baseAddress!, $0.count) } }
        return b
    }

    private func emptyBuf(_ hal: QWHal, bytes: Int) -> any MTLBuffer {
        let b = hal.device.makeBuffer(length: max(256, bytes), options: .storageModeShared)!
        memset(b.contents(), 0, b.length)
        return b
    }

    private func runCmd(_ hal: QWHal, _ body: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = hal.begin() else { XCTFail("cannot begin CB"); return }
        try body(enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { XCTFail("GPU error: \(err)") }
    }

    func testMixNormAndAddNormParity() throws {
        let hal = try makeHAL()
        let T = 3, H = 256, stride = 5, eps: Float = 1e-6
        let taps = R.randomBF16Floats(count: 4 * stride * H, seed: 201)
        let nw = R.randomBF16Floats(count: H, seed: 202, scale: 1.5)
        let a: (Float, Float, Float, Float) = (0.4, 0.3, 0.2, 0.1)
        let outMix = emptyBuf(hal, bytes: T * H * 2)
        var mp = MixP(T: UInt32(T), H: UInt32(H), tapPlaneStride: UInt32(stride * H),
                      eps: eps, a0: a.0, a1: a.1, a2: a.2, a3: a.3)
        let pMix = try hal.pipeline("qwg_ink2_dspark_mix_norm")
        try runCmd(hal) { enc in
            enc.setComputePipelineState(pMix)
            enc.setBuffer(buf(hal, R.bf16Array(taps)), offset: 0, index: 0)
            enc.setBuffer(buf(hal, R.bf16Array(nw)), offset: 0, index: 1)
            enc.setBuffer(outMix, offset: 0, index: 2)
            enc.setBytes(&mp, length: MemoryLayout<MixP>.stride, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: T, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        let gotMix = R.floatsFromBF16Buffer(outMix, count: T * H)
        for t in 0..<T {
            var mixed = [Double](repeating: 0, count: H)
            var sq = 0.0
            for d in 0..<H {
                let ps = stride * H
                let m = Double(taps[t * H + d]) * Double(a.0) + Double(taps[ps + t * H + d]) * Double(a.1)
                    + Double(taps[2 * ps + t * H + d]) * Double(a.2) + Double(taps[3 * ps + t * H + d]) * Double(a.3)
                mixed[d] = m
                sq += m * m
            }
            let inv = 1.0 / sqrt(sq / Double(H) + Double(eps))
            for d in 0..<H {
                let want = R.roundBF16(mixed[d] * inv * Double(nw[d]))
                XCTAssertLessThanOrEqual(abs(gotMix[t * H + d] - want), 2e-2)
            }
        }

        // add_norm + silu_mul
        let xF = R.randomBF16Floats(count: T * H, seed: 203)
        let rF = R.randomBF16Floats(count: T * H, seed: 204)
        let resB = buf(hal, R.bf16Array(rF))
        let normB = emptyBuf(hal, bytes: T * H * 2)
        var ap = AddNormP(rows: UInt32(T), dim: UInt32(H), hasRes: 1, writeRes: 1, eps: eps)
        let pAdd = try hal.pipeline("qwg_ink2_dspark_add_norm")
        try runCmd(hal) { enc in
            enc.setComputePipelineState(pAdd)
            enc.setBuffer(buf(hal, R.bf16Array(xF)), offset: 0, index: 0)
            enc.setBuffer(resB, offset: 0, index: 1)
            enc.setBuffer(buf(hal, R.bf16Array(nw)), offset: 0, index: 2)
            enc.setBuffer(normB, offset: 0, index: 3)
            enc.setBytes(&ap, length: MemoryLayout<AddNormP>.stride, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: T, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        let gotRes = R.floatsFromBF16Buffer(resB, count: T * H)
        let gotNrm = R.floatsFromBF16Buffer(normB, count: T * H)
        for t in 0..<T {
            var sq = 0.0
            for d in 0..<H { let s = Double(xF[t * H + d]) + Double(rF[t * H + d]); sq += s * s }
            let inv = 1.0 / sqrt(sq / Double(H) + Double(eps))
            for d in 0..<H {
                let s = Double(xF[t * H + d]) + Double(rF[t * H + d])
                XCTAssertLessThanOrEqual(abs(gotRes[t * H + d] - R.roundBF16(s)), 1e-2)
                XCTAssertLessThanOrEqual(abs(gotNrm[t * H + d] - R.roundBF16(s * inv * Double(nw[d]))), 2e-2)
            }
        }
    }

    func testCtxKVRopePropQKRopeAndBidirectionalAttnParity() throws {
        let hal = try makeHAL()
        let nH = 6, nKV = 2, headDim = 64, ctxLen = 4, B = 8, ringCap = 12
        let eps: Float = 1e-6, theta: Float = 1e6
        let kvDim = nKV * headDim, qDim = nH * headDim

        let kCtx = R.randomBF16Floats(count: ctxLen * kvDim, seed: 301)
        let vCtx = R.randomBF16Floats(count: ctxLen * kvDim, seed: 302)
        let kNormCtx = R.randomBF16Floats(count: headDim, seed: 303, scale: 1.5)
        let qProp = R.randomBF16Floats(count: B * qDim, seed: 304)
        let qNorm = R.randomBF16Floats(count: headDim, seed: 305, scale: 1.5)
        let kProp = R.randomBF16Floats(count: B * kvDim, seed: 306)
        let kNorm = R.randomBF16Floats(count: headDim, seed: 307, scale: 1.5)
        let vProp = R.randomBF16Floats(count: B * kvDim, seed: 308)

        let kCache = emptyBuf(hal, bytes: ringCap * kvDim * 2)
        let vCache = emptyBuf(hal, bytes: ringCap * kvDim * 2)
        let qBuf = buf(hal, R.bf16Array(qProp))
        let kPBuf = buf(hal, R.bf16Array(kProp))
        let attnOutGQA = emptyBuf(hal, bytes: B * qDim * 2)

        var cp = CtxKVP(T: UInt32(ctxLen), nKV: UInt32(nKV), headDim: UInt32(headDim),
                        isSliding: 1, posOff: 10, ringCap: UInt32(ringCap), eps: eps, ropeTheta: theta)
        var qkr = PropQKRP(B: UInt32(B), nH: UInt32(nH), nKV: UInt32(nKV), headDim: UInt32(headDim),
                           posOff: UInt32(10 + ctxLen), eps: eps, ropeTheta: theta)
        var ap = AttnP(B: UInt32(B), nH: UInt32(nH), nKV: UInt32(nKV), headDim: UInt32(headDim),
                       ctxStart: 10, ctxEnd: UInt32(10 + ctxLen), sliding: 6, ringCap: UInt32(ringCap),
                       scale: 1.0 / Float(headDim).squareRoot())

        let pCtx = try hal.pipeline("qwg_ink2_dspark_ctx_kv_write")
        let pProp = try hal.pipeline("qwg_ink2_dspark_prop_qk_rope")
        let pAttnGQA = try hal.pipeline("qwg_ink2_dspark_attn_gqa")
        try runCmd(hal) { enc in
            enc.setComputePipelineState(pCtx)
            enc.setBuffer(buf(hal, R.bf16Array(kCtx)), offset: 0, index: 0)
            enc.setBuffer(buf(hal, R.bf16Array(vCtx)), offset: 0, index: 1)
            enc.setBuffer(buf(hal, R.bf16Array(kNormCtx)), offset: 0, index: 2)
            enc.setBuffer(kCache, offset: 0, index: 3)
            enc.setBuffer(vCache, offset: 0, index: 4)
            enc.setBytes(&cp, length: MemoryLayout<CtxKVP>.stride, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: nKV, height: ctxLen, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

            enc.setComputePipelineState(pProp)
            enc.setBuffer(qBuf, offset: 0, index: 0)
            enc.setBuffer(buf(hal, R.bf16Array(qNorm)), offset: 0, index: 1)
            enc.setBuffer(kPBuf, offset: 0, index: 2)
            enc.setBuffer(buf(hal, R.bf16Array(kNorm)), offset: 0, index: 3)
            enc.setBytes(&qkr, length: MemoryLayout<PropQKRP>.stride, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: nH + nKV, height: B, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

            enc.setComputePipelineState(pAttnGQA)
            enc.setBuffer(qBuf, offset: 0, index: 0)
            enc.setBuffer(kPBuf, offset: 0, index: 1)
            enc.setBuffer(buf(hal, R.bf16Array(vProp)), offset: 0, index: 2)
            enc.setBuffer(kCache, offset: 0, index: 3)
            enc.setBuffer(vCache, offset: 0, index: 4)
            enc.setBuffer(attnOutGQA, offset: 0, index: 5)
            enc.setBytes(&ap, length: MemoryLayout<AttnP>.stride, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: nKV, height: B, depth: (nH / nKV + 3) / 4),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }

        let gotQ = R.floatsFromBF16Buffer(qBuf, count: B * qDim)
        let gotKP = R.floatsFromBF16Buffer(kPBuf, count: B * kvDim)
        let gotKC = R.floatsFromBF16Buffer(kCache, count: ringCap * kvDim)
        let gotOGQA = R.floatsFromBF16Buffer(attnOutGQA, count: B * qDim)

        // CPU reference bidirectional attention over prepared Q/K + V
        let group = nH / nKV
        let scale = 1.0 / sqrt(Double(headDim))
        for b in 0..<B {
            for h in 0..<nH {
                let kvh = h / group
                var scores: [Double] = []
                var vs: [[Double]] = []
                for absP in 10..<(10 + ctxLen) {
                    let slot = absP % ringCap
                    var dot = 0.0
                    var vr = [Double](repeating: 0, count: headDim)
                    for d in 0..<headDim {
                        dot += Double(gotQ[(b * nH + h) * headDim + d]) * Double(gotKC[(slot * nKV + kvh) * headDim + d])
                        vr[d] = Double(vCtx[((absP - 10) * nKV + kvh) * headDim + d])
                    }
                    scores.append(dot * scale); vs.append(vr)
                }
                for p in 0..<B {
                    var dot = 0.0
                    var vr = [Double](repeating: 0, count: headDim)
                    for d in 0..<headDim {
                        dot += Double(gotQ[(b * nH + h) * headDim + d]) * Double(gotKP[(p * nKV + kvh) * headDim + d])
                        vr[d] = Double(vProp[(p * nKV + kvh) * headDim + d])
                    }
                    scores.append(dot * scale); vs.append(vr)
                }
                let mx = scores.max() ?? 0
                let ex = scores.map { exp($0 - mx) }
                let sm = ex.reduce(0, +)
                for d in 0..<headDim {
                    var acc = 0.0
                    for i in 0..<ex.count { acc += (ex[i] / sm) * vs[i][d] }
                    let want = R.roundBF16(acc)
                    let idx = (b * nH + h) * headDim + d
                    XCTAssertLessThanOrEqual(abs(gotOGQA[idx] - want), 2e-2)
                }
            }
        }
    }

    func testMarkovHeadAQ8Stage1AndMultiSliceGQAParity() throws {
        let hal = try makeHAL()
        let nH = 10, nKV = 2, headDim = 64, ctxLen = 3, B = 4, ringCap = 8
        let kvDim = nKV * headDim, qDim = nH * headDim
        let qP = R.randomBF16Floats(count: B * qDim, seed: 501)
        let kP = R.randomBF16Floats(count: B * kvDim, seed: 502)
        let vP = R.randomBF16Floats(count: B * kvDim, seed: 503)
        let kC = R.randomBF16Floats(count: ringCap * kvDim, seed: 504)
        let vC = R.randomBF16Floats(count: ringCap * kvDim, seed: 505)
        let outGQA = emptyBuf(hal, bytes: B * qDim * 2)
        var ap = AttnP(B: UInt32(B), nH: UInt32(nH), nKV: UInt32(nKV), headDim: UInt32(headDim),
                       ctxStart: 2, ctxEnd: UInt32(2 + ctxLen), sliding: 4, ringCap: UInt32(ringCap),
                       scale: 1.0 / Float(headDim).squareRoot())
        let pAttnGQA = try hal.pipeline("qwg_ink2_dspark_attn_gqa")
        try runCmd(hal) { enc in
            enc.setComputePipelineState(pAttnGQA)
            enc.setBuffer(buf(hal, R.bf16Array(qP)), offset: 0, index: 0)
            enc.setBuffer(buf(hal, R.bf16Array(kP)), offset: 0, index: 1)
            enc.setBuffer(buf(hal, R.bf16Array(vP)), offset: 0, index: 2)
            enc.setBuffer(buf(hal, R.bf16Array(kC)), offset: 0, index: 3)
            enc.setBuffer(buf(hal, R.bf16Array(vC)), offset: 0, index: 4)
            enc.setBuffer(outGQA, offset: 0, index: 5)
            enc.setBytes(&ap, length: MemoryLayout<AttnP>.stride, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: nKV, height: B, depth: (nH / nKV + 3) / 4),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        let gGQA = R.floatsFromBF16Buffer(outGQA, count: B * qDim)
        let group = nH / nKV, scale = 1.0 / sqrt(Double(headDim))
        for b in 0..<B {
            for h in 0..<nH {
                let kvh = h / group
                var scores = [Double](), vs = [[Double]]()
                for absP in 2..<(2 + ctxLen) {
                    let slot = absP % ringCap
                    var dot = 0.0, vr = [Double](repeating: 0, count: headDim)
                    for d in 0..<headDim {
                        dot += Double(qP[(b * nH + h) * headDim + d]) * Double(kC[(slot * nKV + kvh) * headDim + d])
                        vr[d] = Double(vC[(slot * nKV + kvh) * headDim + d])
                    }
                    scores.append(dot * scale); vs.append(vr)
                }
                for p in 0..<B {
                    var dot = 0.0, vr = [Double](repeating: 0, count: headDim)
                    for d in 0..<headDim {
                        dot += Double(qP[(b * nH + h) * headDim + d]) * Double(kP[(p * nKV + kvh) * headDim + d])
                        vr[d] = Double(vP[(p * nKV + kvh) * headDim + d])
                    }
                    scores.append(dot * scale); vs.append(vr)
                }
                let mx = scores.max() ?? 0, ex = scores.map { exp($0 - mx) }, sm = ex.reduce(0, +)
                for d in 0..<headDim {
                    let want = R.roundBF16((0..<ex.count).reduce(0.0) { $0 + (ex[$1] / sm) * vs[$1][d] })
                    XCTAssertLessThanOrEqual(abs(gGQA[(b * nH + h) * headDim + d] - want), 2e-2)
                }
            }
        }

        let count = 3, Rdim = 64, V = 512, softcap: Float = 30.0
        let latF = R.randomBF16Floats(count: count * Rdim, seed: 510)
        let baseF = R.randomBF16Floats(count: count * V, seed: 511, scale: 2.0)
        let foldWF = R.randomBF16Floats(count: V * Rdim, seed: 512, scale: 0.5)
        let foldBF = R.randomBF16Floats(count: V, seed: 513, scale: 0.3)
        let nGroups = Rdim / 32, rowBytes = nGroups * 36
        var packed = [UInt8](repeating: 0, count: V * rowBytes)
        var deqW = [Double](repeating: 0, count: V * Rdim)
        packed.withUnsafeMutableBytes { raw in
            let p = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for v in 0..<V {
                for g in 0..<nGroups {
                    let gOff = v * Rdim + g * 32
                    let slice = (0..<32).map { Double(foldWF[gOff + $0]) }
                    let mn = slice.min()!, mx = slice.max()!
                    let minF16 = Float16(mn), scaleF16 = Float16(max(0.0, mx - Double(minF16)) / 255.0)
                    let minD = Double(minF16), scaleD = Double(scaleF16), inv = scaleD > 0 ? 1.0 / scaleD : 0.0
                    var sb = (scaleF16.bitPattern, minF16.bitPattern)
                    let gDst = p.advanced(by: v * rowBytes + g * 36)
                    memcpy(gDst, &sb, 4)
                    for j in 0..<32 {
                        let q = UInt8(clamping: scaleD > 0 ? Int(((slice[j] - minD) * inv).rounded(.toNearestOrEven)) : 0)
                        gDst[4 + j] = q
                        deqW[gOff + j] = scaleD * Double(q) + minD
                    }
                }
            }
        }
        let foldWB = buf(hal, packed), foldBB = buf(hal, R.bf16Array(foldBF)), baseB = buf(hal, R.bf16Array(baseF))
        let pVals = emptyBuf(hal, bytes: 1024 * 4), pIdxs = emptyBuf(hal, bytes: 1024 * 4)
        let draftToks = buf(hal, [Int32](repeating: 0, count: count + 1))
        let pL1 = try hal.pipeline("qwg_ink2_dspark_markov_logits_aq8_stage1")
        let pL2 = try hal.pipeline("qwg_ink2_dspark_markov_argmax_final")
        let partCount = min(1024, (V + 7) / 8)
        for step in 0..<count {
            let latB = buf(hal, R.bf16Array(Array(latF[(step * Rdim)..<((step + 1) * Rdim)])))
            var lp = LogitsP(stepIdx: UInt32(step), V: UInt32(V), rank: UInt32(Rdim), partCount: UInt32(partCount), softcap: softcap)
            try runCmd(hal) { enc in
                enc.setComputePipelineState(pL1)
                enc.setBuffer(baseB, offset: 0, index: 0)
                enc.setBuffer(latB, offset: 0, index: 1)
                enc.setBuffer(foldWB, offset: 0, index: 2)
                enc.setBuffer(foldBB, offset: 0, index: 3)
                enc.setBuffer(pVals, offset: 0, index: 4)
                enc.setBuffer(pIdxs, offset: 0, index: 5)
                enc.setBytes(&lp, length: MemoryLayout<LogitsP>.stride, index: 6)
                enc.dispatchThreadgroups(MTLSize(width: partCount, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.setComputePipelineState(pL2)
                enc.setBuffer(pVals, offset: 0, index: 0)
                enc.setBuffer(pIdxs, offset: 0, index: 1)
                enc.setBuffer(draftToks, offset: 0, index: 2)
                enc.setBytes(&lp, length: MemoryLayout<LogitsP>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
        }
        let tokPtr = draftToks.contents().assumingMemoryBound(to: Int32.self)
        let gpuToks = (1...count).map { Int(tokPtr[$0]) }
        var cpuToks: [Int] = []
        for s in 0..<count {
            var bestV = -Double.infinity, bestI = Int.max
            for v in 0..<V {
                var d = Double(foldBF[v])
                for r in 0..<Rdim { d += Double(latF[s * Rdim + r]) * deqW[v * Rdim + r] }
                var comb = Double(R.roundBF16(Double(baseF[s * V + v]) + Double(R.roundBF16(d))))
                comb = Double(R.roundBF16(tanh(comb / Double(softcap)) * Double(softcap)))
                if comb > bestV || (comb == bestV && v < bestI) { bestV = comb; bestI = v }
            }
            cpuToks.append(bestI)
        }
        XCTAssertEqual(gpuToks, cpuToks)
    }

    func testMarkovHeadFullPipelineAndArgmaxParity() throws {
        let hal = try makeHAL()
        let count = 3, H = 64, Rdim = 32, projRows = 96, V = 512, softcap: Float = 30.0
        let xF = R.randomBF16Floats(count: count * H, seed: 401)
        let fRnnF = R.randomBF16Floats(count: H, seed: 402)
        let rnnWF = R.randomBF16Floats(count: projRows * (2 * Rdim + 2 * H), seed: 403, scale: 0.4)
        let rnnBF = R.randomBF16Floats(count: projRows, seed: 404, scale: 0.2)
        let w1F = R.randomBF16Floats(count: V * Rdim, seed: 405, scale: 0.5)
        let baseF = R.randomBF16Floats(count: count * V, seed: 406, scale: 2.0)
        let foldWF = R.randomBF16Floats(count: V * Rdim, seed: 407, scale: 0.5)
        let foldBF = R.randomBF16Floats(count: V, seed: 408, scale: 0.3)

        let ctxOut = emptyBuf(hal, bytes: count * projRows * 4)
        let stateOut = emptyBuf(hal, bytes: Rdim * 2)
        let latOut = emptyBuf(hal, bytes: Rdim * 2)
        let draftToks = buf(hal, [Int32(17), 0, 0, 0])
        let pVals = emptyBuf(hal, bytes: 1024 * 4)
        let pIdxs = emptyBuf(hal, bytes: 1024 * 4)
        let rnnWB = buf(hal, R.bf16Array(rnnWF))
        let w1B = buf(hal, R.bf16Array(w1F))
        let baseB = buf(hal, R.bf16Array(baseF))
        let foldWB = buf(hal, R.bf16Array(foldWF))
        let foldBB = buf(hal, R.bf16Array(foldBF))

        let pCtx = try hal.pipeline("qwg_ink2_dspark_markov_ctx")
        let pStep = try hal.pipeline("qwg_ink2_dspark_markov_step_state")
        let pL1 = try hal.pipeline("qwg_ink2_dspark_markov_logits_stage1")
        let pL2 = try hal.pipeline("qwg_ink2_dspark_markov_argmax_final")
        let partCount = min(1024, (V + 7) / 8)

        try runCmd(hal) { enc in
            var mcp = MarkovCtxP(count: UInt32(count), H: UInt32(H), projRows: UInt32(projRows), rank: UInt32(Rdim))
            enc.setComputePipelineState(pCtx)
            enc.setBuffer(buf(hal, R.bf16Array(xF)), offset: 0, index: 0)
            enc.setBuffer(buf(hal, R.bf16Array(fRnnF)), offset: 0, index: 1)
            enc.setBuffer(rnnWB, offset: 0, index: 2)
            enc.setBuffer(buf(hal, R.bf16Array(rnnBF)), offset: 0, index: 3)
            enc.setBuffer(ctxOut, offset: 0, index: 4)
            enc.setBytes(&mcp, length: MemoryLayout<MarkovCtxP>.stride, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: (projRows + 7) / 8, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            for step in 0..<count {
                var msp = MarkovStepP(stepIdx: UInt32(step), H: UInt32(H), projRows: UInt32(projRows),
                                      rank: UInt32(Rdim), resetState: step == 0 ? 1 : 0)
                enc.setComputePipelineState(pStep)
                enc.setBuffer(draftToks, offset: 0, index: 0)
                enc.setBuffer(w1B, offset: 0, index: 1)
                enc.setBuffer(rnnWB, offset: 0, index: 2)
                enc.setBuffer(ctxOut, offset: 0, index: 3)
                enc.setBuffer(stateOut, offset: 0, index: 4)
                enc.setBuffer(latOut, offset: 0, index: 5)
                enc.setBytes(&msp, length: MemoryLayout<MarkovStepP>.stride, index: 6)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

                var lp = LogitsP(stepIdx: UInt32(step), V: UInt32(V), rank: UInt32(Rdim),
                                 partCount: UInt32(partCount), softcap: softcap)
                enc.setComputePipelineState(pL1)
                enc.setBuffer(baseB, offset: 0, index: 0)
                enc.setBuffer(latOut, offset: 0, index: 1)
                enc.setBuffer(foldWB, offset: 0, index: 2)
                enc.setBuffer(foldBB, offset: 0, index: 3)
                enc.setBuffer(pVals, offset: 0, index: 4)
                enc.setBuffer(pIdxs, offset: 0, index: 5)
                enc.setBytes(&lp, length: MemoryLayout<LogitsP>.stride, index: 6)
                enc.dispatchThreadgroups(MTLSize(width: partCount, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

                enc.setComputePipelineState(pL2)
                enc.setBuffer(pVals, offset: 0, index: 0)
                enc.setBuffer(pIdxs, offset: 0, index: 1)
                enc.setBuffer(draftToks, offset: 0, index: 2)
                enc.setBytes(&lp, length: MemoryLayout<LogitsP>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
        }

        // CPU reference
        let tokPtr = draftToks.contents().assumingMemoryBound(to: Int32.self)
        let gpuToks = (1...count).map { Int(tokPtr[$0]) }
        var state = [Double](repeating: 0, count: Rdim)
        var prev = 17
        let wStride = 2 * Rdim + 2 * H
        var cpuToks: [Int] = []
        for s in 0..<count {
            var ctx = [Double](repeating: 0, count: projRows)
            for r in 0..<projRows {
                var acc = Double(rnnBF[r])
                for k in 0..<H {
                    acc += Double(xF[s * H + k]) * Double(rnnWF[r * wStride + 2 * Rdim + k])
                    acc += Double(fRnnF[k]) * Double(rnnWF[r * wStride + 2 * Rdim + H + k])
                }
                ctx[r] = acc
            }
            var raw = [Double](repeating: 0, count: projRows)
            for r in 0..<projRows {
                var acc = ctx[r]
                for k in 0..<Rdim {
                    acc += state[k] * Double(rnnWF[r * wStride + k])
                    acc += Double(w1F[prev * Rdim + k]) * Double(rnnWF[r * wStride + Rdim + k])
                }
                raw[r] = Double(R.roundBF16(acc))
            }
            var latent = [Double](repeating: 0, count: Rdim)
            for r in 0..<Rdim {
                let g = Double(R.roundBF16(1.0 / (1.0 + exp(-raw[r]))))
                let c = Double(R.roundBF16(tanh(raw[r + Rdim])))
                let o = Double(R.roundBF16(tanh(raw[r + 2 * Rdim])))
                state[r] = Double(R.roundBF16(g * state[r] + (1.0 - g) * c))
                latent[r] = o
            }
            var bestVal = -Double.infinity, bestIdx = Int.max
            for v in 0..<V {
                var d = Double(foldBF[v])
                for r in 0..<Rdim { d += latent[r] * Double(foldWF[v * Rdim + r]) }
                let bias = Double(R.roundBF16(d))
                var comb = Double(R.roundBF16(Double(baseF[s * V + v]) + bias))
                comb = Double(R.roundBF16(tanh(comb / Double(softcap)) * Double(softcap)))
                if comb > bestVal || (comb == bestVal && v < bestIdx) { bestVal = comb; bestIdx = v }
            }
            prev = bestIdx
            cpuToks.append(bestIdx)
        }
        XCTAssertEqual(gpuToks, cpuToks)
    }

    func testCommitVerifyConvStatesExactRollback() throws {
        let hal = try makeHAL()
        let dev = hal.device
        let pCommit = try hal.pipeline("qwg_ink2_sconv_verify_commit")
        let desc = QWModelDescriptor(
            family: .inkling2Loop, modelType: "inkling2", capabilities: [.moe],
            hidden: 16, vocab: 64, denseIntermediate: 32, numLayers: 2,
            layerTypes: ["sliding_attention", "full_attention"], fullAttentionInterval: 2,
            rmsEps: 1e-6, maxPositionEmbeddings: 128, qHeads: 2, kvHeads: 1, headDim: 8,
            attnOutputGate: true, partialRotaryFactor: 1.0, ropeTheta: 1e6,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil, numExperts: 4, expertsPerTok: 2,
            moeIntermediate: 8, sharedExpertIntermediate: nil, expertGateType: nil,
            hcCount: nil, hcLowrank: nil, gdnQKHeads: 0, gdnVHeads: 0, gdnHeadDim: 0,
            gdnQKVChannels: 0, gdnZChannels: 0, gdnConvKernel: 0,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 0, mtpUseDedicatedEmbeddings: false, eosTokenId: 200006,
            loopSettings: QWModelDescriptor.LoopSettings(start: 0, end: 1, evalCount: 1, counts: [1])
        )
        let ink = QWModelDescriptor.Inkling2Config(
            unpaddedVocabSize: 64, dRel: 4, relExtent: 16, qBias: false, oBias: false,
            logScalingNFloor: 8, logScalingAlpha: 0.1, useValueNormNoGamma: true,
            qkL2ClipThreshold: 22.6, relL2ClipThreshold: 8.0, localLayerIds: [0],
            sconvKernelSize: 4, finalLogitSoftcapping: 30.0, swaHeadDim: 8, swaQHeads: 2, swaKVHeads: 1,
            slidingWindowSize: 16, layersPerAttnResBlock: 1, attnResNormalizeValues: true,
            mlpGateSoftcap: 4.0, mlpUpSoftcap: 25.0, routeScale: 1.0, useGateBias: true,
            loopCoreStart: 0, loopCoreEnd: 1, loopNumPasses: 1, loopGateRank: 4, isVarloop: false,
            defaultQuantMode: .bf16, defaultQuantBits: 16, defaultQuantGroupSize: 32,
            perLayerExpertQuant: [:], dspark: nil, vision: nil, padTokenId: 0
        )
        let session = try QWInkling2Session(device: dev, descriptor: desc, ink2: ink,
                                            capacity: 64, kvModeRaw: 0, enableSpeculative: true)
        for kept in [1, 2, 3, 8] {
            let slabPtr = session.convStatesSlab.contents()
            for (cIdx, stOff) in session.convStateOffsets.enumerated() {
                let ch = session.convVerifyChannels[cIdx]
                let ptr = slabPtr.advanced(by: stOff).assumingMemoryBound(to: Float.self)
                for slot in 0..<3 {
                    for c in 0..<ch { ptr[slot * ch + c] = Float(100 + slot * 10 + c) }
                }
            }
            session.snapshotPreVerifyConvStates()
            let vPtr = session.convVerifyInputsBuf!.contents()
            for cIdx in session.convStateOffsets.indices {
                let ch = session.convVerifyChannels[cIdx]
                let xPtr = vPtr.advanced(by: session.convVerifyOffsets[cIdx]).assumingMemoryBound(to: UInt16.self)
                for r in 0..<8 {
                    for c in 0..<ch { xPtr[r * ch + c] = R.bf16Bits(Float((r + 1) * 16 + c)) }
                }
            }
            session.commitVerifyConvStates(kept: kept)
            let p0 = slabPtr.advanced(by: session.convStateOffsets[0]).assumingMemoryBound(to: Float.self)
            let ch0 = session.convVerifyChannels[0]
            if kept == 1 {
                XCTAssertEqual(p0[0], 110); XCTAssertEqual(p0[ch0], 120); XCTAssertEqual(p0[2 * ch0], 16)
            } else if kept == 2 {
                XCTAssertEqual(p0[0], 120); XCTAssertEqual(p0[ch0], 16); XCTAssertEqual(p0[2 * ch0], 32)
            } else if kept == 3 {
                XCTAssertEqual(p0[0], 16); XCTAssertEqual(p0[ch0], 32); XCTAssertEqual(p0[2 * ch0], 48)
            } else {
                XCTAssertEqual(p0[0], 96); XCTAssertEqual(p0[ch0], 112); XCTAssertEqual(p0[2 * ch0], 128)
            }
            let floatCount = session.convStatesSlab.length / MemoryLayout<Float>.stride
            let fPtr = slabPtr.assumingMemoryBound(to: Float.self)
            let cpuSlab = Array(UnsafeBufferPointer(start: fPtr, count: floatCount))
            memcpy(slabPtr, session.convPreVerifyBuf!.contents(), session.convStatesSlab.length)
            let kvDim = desc.kvHeads * desc.headDim, H = desc.hidden
            let totalCh = 2 * kvDim + 2 * H
            var cp = SConvVerifyCommitP(
                numVisits: UInt32(session.executionPlan.count), kvDim: UInt32(kvDim), H: UInt32(H), kept: UInt32(kept),
                stateKvStride: UInt32(session.convStateKvStrideElems), stateHStride: UInt32(session.convStateHStrideElems),
                verifyKvStride: UInt32(session.convVerifyKvStrideElems), verifyHStride: UInt32(session.convVerifyHStrideElems)
            )
            try runCmd(hal) { enc in
                enc.setComputePipelineState(pCommit)
                enc.setBuffer(session.convStatesSlab, offset: 0, index: 0)
                enc.setBuffer(session.convVerifyInputsBuf!, offset: 0, index: 1)
                enc.setBytes(&cp, length: MemoryLayout<SConvVerifyCommitP>.stride, index: 2)
                enc.dispatchThreads(MTLSize(width: totalCh, height: session.executionPlan.count, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: min(256, totalCh), height: 1, depth: 1))
            }
            for i in 0..<floatCount { XCTAssertEqual(fPtr[i], cpuSlab[i], "mismatch at float[\(i)] kept=\(kept)") }
        }
    }
}
