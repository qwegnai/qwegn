// QWSHVQSub2GemmTests.swift — GPU parity tests for SHVQ1_G64 and SHVQ15_G64
// across SwiGLU pair, head argmax, GDN inproj, MoE (f32 + NAX), HC, MLA, and DSpine.
import Foundation
import Metal
import XCTest
@testable import QWGEMM
import QWAlloc
import QWCore
import QWHAL
import QWQuant

final class QWSHVQSub2GemmTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private struct MLAAbsorbParamsPod {
        var T: UInt32
        var qHeads: UInt32
        var nopeDim: UInt32
        var vHeadDim: UInt32
        var kvLoraRank: UInt32
        var qFullStride: UInt32
        var bits: UInt32
        var group: UInt32
        var isIncumbent: UInt32
    }

    private func makeSHVQBank(bits: Int, group: Int, n: Int, k: Int, scale: Float = 0.0625) -> Data {
        let groups = k / group
        let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
        var seed: UInt32 = 0x9E37_79B9 ^ UInt32(bits * 131 + group)
        var bank = Data(count: n * groups * gBytes)
        bank.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let base = (r * groups + g) * gBytes
                    let sc = Float16(scale * (1.0 + 0.125 * Float((r + g) % 5))).bitPattern
                    ptr[base] = UInt8(sc & 0xFF)
                    ptr[base + 1] = UInt8(sc >> 8)
                    for b in 2..<gBytes {
                        seed = seed &* 1664525 &+ 1013904223
                        ptr[base + b] = UInt8((seed >> 16) & 0xFF)
                    }
                }
            }
        }
        return bank
    }

    private func unpackBankRows(bits: Int, group: Int, n: Int, k: Int, bank: Data) -> [[Float]] {
        let groups = k / group
        let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
        var rows = [[Float]]()
        rows.reserveCapacity(n)
        for r in 0..<n {
            var row = [Float]()
            row.reserveCapacity(k)
            var tmp = [Float]()
            for g in 0..<groups {
                let off = (r * groups + g) * gBytes
                let sub = bank.subdata(in: off..<(off + gBytes))
                sub.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                    QWSHVQBlock.unpack(group: raw, bits: bits, group: group, into: &tmp)
                }
                row.append(contentsOf: tmp)
            }
            rows.append(row)
        }
        return rows
    }

    private func makeActivations(count: Int, seedInit: UInt32) -> [Float] {
        var seed = seedInit
        return (0..<count).map { _ in
            seed = seed &* 1664525 &+ 1013904223
            return Float(Int32(truncatingIfNeeded: seed >> 8) % 512) / 512.0 - 0.5
        }
    }

    private func step(_ f: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
        try f(enc)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "GPU step failed: \(String(describing: cb.error))")
    }

    private func dotRef(w: [Float], x: ArraySlice<Float>, roundWeightsToF16: Bool = false) -> (val: Float, mag: Float) {
        var acc: Float = 0, mag: Float = 0
        for (wi, xi) in zip(w, x) {
            let ww = roundWeightsToF16 ? Float(Float16(wi)) : wi
            let p = ww * xi
            acc += p
            mag += abs(p)
        }
        return (acc, mag)
    }

    func testSiluPairHeadArgmaxAndGDNParityForSHVQ1AndSHVQ15() throws {
        for (bits, group) in [(1, 64), (15, 64)] {
            let inter = 64, k = 256, nRows = inter * 2
            let bank = makeSHVQBank(bits: bits, group: group, n: nRows, k: k)
            let wRows = unpackBankRows(bits: bits, group: group, n: nRows, k: k, bank: bank)
            let arena = try QWArena(hal: Self.hal, weights: bank.count + (2 << 20),
                                    activations: (8 << 20), kv: 0, gdnStates: 0)
            let wBuf = try arena.weightsRegion().allocate(QWTensorShape([(bank.count + 1) / 2], .f16))
            arena.copyIn(wBuf, from: bank)
            let view = try XCTUnwrap(QWWeightView(blocks: wBuf, n: nRows, k: k, bits: bits,
                                                  groupSize: group, isRotated: true, isSHVQ: true))
            let act = arena.activationsRegion()
            let siluName = try XCTUnwrap(QWGemmAQ.siluPairKernelName(bits: bits, group: group, isSHVQ: true))
            let siluPSO = try Self.hal.pipeline(siluName)

            for m in [1, 3] {
                let xRaw = makeActivations(count: m * k, seedInit: UInt32(0x2100 + m + bits))
                let x16 = try act.allocate(QWTensorShape([m * k], .f16))
                let out16 = try act.allocate(QWTensorShape([m * inter], .f16))
                arena.write(x16, from: xRaw.map { Float16($0) })
                try step { enc in
                    enc.setComputePipelineState(siluPSO)
                    wBuf.bind(enc, index: 0)
                    x16.bind(enc, index: 1)
                    var nFlag: UInt32 = m == 1
                        ? (UInt32(nRows) | 0x4000_0000)
                        : (UInt32(nRows) | 0x4000_0000 | (UInt32(m) << 24))
                    var kU = UInt32(k)
                    enc.setBytes(&nFlag, length: 4, index: 4)
                    enc.setBytes(&kU, length: 4, index: 5)
                    out16.bind(enc, index: 8)
                    enc.dispatchThreadgroups(MTLSize(width: (inter + 31) / 32, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
                let got = arena.read(out16, as: Float16.self, count: m * inter).map { Float($0) }
                for t in 0..<m {
                    let xSlice = xRaw[(t * k)..<((t + 1) * k)].map { Float(Float16($0)) }[...]
                    for r in 0..<inter {
                        let gVal = dotRef(w: wRows[r], x: xSlice).val
                        let uVal = dotRef(w: wRows[inter + r], x: xSlice).val
                        let want = (gVal / (1.0 + exp(-gVal))) * uVal
                        XCTAssertEqual(got[t * inter + r], Float(Float16(want)), accuracy: 2e-3,
                                       "silu_pair SHVQ\(bits)_G\(group) m=\(m) t=\(t) r=\(r)")
                    }
                }
            }

            // Head GEMV + argmax parity
            let xHead = makeActivations(count: k, seedInit: UInt32(0x3300 + bits))
            let xHead16 = try act.allocate(QWTensorShape([k], .f16))
            let yHead32 = try act.allocate(QWTensorShape([nRows], .f32))
            let slots = (nRows + 31) / 32
            let kHi = try act.allocate(QWTensorShape([slots], .f32))
            let kLo = try act.allocate(QWTensorShape([slots], .i32))
            let tokOut = try act.allocate(QWTensorShape([1], .i32))
            arena.write(xHead16, from: xHead.map { Float16($0) })
            try step { enc in
                try QWGemmAQ.encodeHeadArgmax(enc, hal: Self.hal, view: view, x: xHead16, y: yHead32,
                                              keysHi: kHi, keysLo: kLo, writeLogits: true)
                enc.memoryBarrier(scope: .buffers)
                try QWGemm.encodeArgmaxFin(enc, hal: Self.hal, keysHi: kHi, keysLo: kLo,
                                           token: tokOut, slots: slots, rows: 1)
            }
            let gotHead = arena.read(yHead32, as: Float.self, count: nRows)
            let gotTok = arena.read(tokOut, as: Int32.self, count: 1)[0]
            let xHeadSlice = xHead.map { Float(Float16($0)) }[...]
            var bestRow = 0, bestVal = -Float.infinity
            for r in 0..<nRows {
                let ref = dotRef(w: wRows[r], x: xHeadSlice).val
                XCTAssertEqual(gotHead[r], ref, accuracy: 2e-3)
                if ref > bestVal { bestVal = ref; bestRow = r }
            }
            XCTAssertEqual(Int(gotTok), bestRow)

            // GDN decode & prefill parity on QKV rows (< aRow0 => pure y16 = half(vx))
            let gdnDecName = try XCTUnwrap(QWGemmAQ.gdnInProjDecodeKernelName(bits: bits, group: group, isSHVQ: true))
            let gdnPreName = try XCTUnwrap(QWGemmAQ.gdnInProjPrefillKernelName(bits: bits, group: group, isSHVQ: true))
            let yGdnDec = try act.allocate(QWTensorShape([nRows], .f16))
            let dummy32 = try act.allocate(QWTensorShape([nRows], .f32))
            try step { enc in
                let pso = try Self.hal.pipeline(gdnDecName)
                enc.setComputePipelineState(pso)
                wBuf.bind(enc, index: 0)
                xHead16.bind(enc, index: 1)
                dummy32.bind(enc, index: 3)
                var nFlag = UInt32(nRows) | 0x4000_0000
                var kU = UInt32(k)
                enc.setBytes(&nFlag, length: 4, index: 4)
                enc.setBytes(&kU, length: 4, index: 5)
                yGdnDec.bind(enc, index: 8)
                dummy32.bind(enc, index: 9)
                dummy32.bind(enc, index: 10)
                dummy32.bind(enc, index: 11)
                dummy32.bind(enc, index: 12)
                enc.dispatchThreadgroups(MTLSize(width: (nRows + 15) / 16, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
            let gotGdnDec = arena.read(yGdnDec, as: Float16.self, count: nRows).map { Float($0) }
            for r in 0..<nRows {
                let ref = dotRef(w: wRows[r], x: xHeadSlice).val
                XCTAssertEqual(gotGdnDec[r], Float(Float16(ref)), accuracy: 2e-3)
            }

            if Self.hal.caps.isMetal4TensorCapable {
                let mPre = 8
                let qkvStride = Self.hal.geometry.gdnQKVChannels
                let xPreRaw = makeActivations(count: mPre * k, seedInit: UInt32(0x4400 + bits))
                let xPre16 = try act.allocate(QWTensorShape([mPre * k], .f16))
                let yGdnPre = try act.allocate(QWTensorShape([mPre * qkvStride], .f16))
                arena.write(xPre16, from: xPreRaw.map { Float16($0) })
                try step { enc in
                    let pso = try Self.hal.pipeline(gdnPreName)
                    enc.setComputePipelineState(pso)
                    wBuf.bind(enc, index: 0)
                    xPre16.bind(enc, index: 1)
                    yGdnPre.bind(enc, index: 2)
                    dummy32.bind(enc, index: 3)
                    var mU = UInt32(mPre), nU = UInt32(nRows), kU = UInt32(k)
                    enc.setBytes(&mU, length: 4, index: 4)
                    enc.setBytes(&nU, length: 4, index: 5)
                    enc.setBytes(&kU, length: 4, index: 6)
                    dummy32.bind(enc, index: 7)
                    dummy32.bind(enc, index: 8)
                    dummy32.bind(enc, index: 9)
                    dummy32.bind(enc, index: 10)
                    enc.dispatchThreadgroups(MTLSize(width: (nRows + 63) / 64, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                }
                let gotGdnPre = arena.read(yGdnPre, as: Float16.self, count: mPre * qkvStride).map { Float($0) }
                for t in 0..<mPre {
                    let sl = xPreRaw[(t * k)..<((t + 1) * k)].map { Float(Float16($0)) }[...]
                    for r in 0..<nRows {
                        let ref = dotRef(w: wRows[r], x: sl, roundWeightsToF16: true).val
                        XCTAssertEqual(gotGdnPre[t * qkvStride + r], Float(Float16(ref)), accuracy: 3e-3)
                    }
                }
            }
        }
    }

    func testMoEF32AndIndirectNAXParityForSHVQ1AndSHVQ15() throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "Grouped NAX MoE requires Metal 4 tensor hardware")
        let experts = 8, tokens = 16, routes = 2, pairs = tokens * routes
        let hidden = 128, inter = 128, guRows = inter * 2
        let tilePlan = QWGemm.MoETilePlan(experts: experts, pairs: pairs, gateUpRows: guRows, downRows: hidden)

        for (bits, group) in [(1, 64), (15, 64)] {
            let guData = makeSHVQBank(bits: bits, group: group, n: experts * guRows, k: hidden)
            let dnData = makeSHVQBank(bits: bits, group: group, n: experts * hidden, k: inter)
            let arena = try QWArena(hal: Self.hal, weights: guData.count + dnData.count + (2 << 20),
                                    activations: (16 << 20), kv: 0, gdnStates: 0)
            let wr = arena.weightsRegion(), ar = arena.activationsRegion()
            let guBuf = try wr.allocate(QWTensorShape([(guData.count + 1) / 2], .f16))
            let dnBuf = try wr.allocate(QWTensorShape([(dnData.count + 1) / 2], .f16))
            arena.copyIn(guBuf, from: guData)
            arena.copyIn(dnBuf, from: dnData)
            let guView = try XCTUnwrap(QWWeightView(blocks: guBuf, n: experts * guRows, k: hidden,
                                                    bits: bits, groupSize: group, isRotated: true, isSHVQ: true))
            let dnView = try XCTUnwrap(QWWeightView(blocks: dnBuf, n: experts * hidden, k: inter,
                                                    bits: bits, groupSize: group, isRotated: true, isSHVQ: true))

            let xeVals = makeActivations(count: tokens * hidden, seedInit: UInt32(0x6000 + bits))
            let xe16 = try ar.allocate(QWTensorShape([tokens * hidden], .f16))
            let xe32 = try ar.allocate(QWTensorShape([tokens * hidden], .f32))
            arena.write(xe16, from: xeVals.map { Float16($0) })
            arena.write(xe32, from: xeVals.map { Float(Float16($0)) })

            let ids: [Int32] = (0..<pairs).map { Int32(($0 * 3 + 1) % experts) }
            let idsT = try ar.allocate(QWTensorShape([pairs], .i32))
            arena.write(idsT, from: ids)
            let wRoute = try ar.allocate(QWTensorShape([pairs], .f32))
            arena.write(wRoute, from: [Float](repeating: 0.5, count: pairs))
            let shared = try ar.allocate(QWTensorShape([tokens * hidden], .f32))
            let gate = try ar.allocate(QWTensorShape([tokens], .f32))
            arena.write(shared, from: [Float](repeating: 0, count: tokens * hidden))
            arena.write(gate, from: [Float](repeating: 0, count: tokens))

            let actRef = try ar.allocate(QWTensorShape([pairs * inter], .f32))
            let outRef = try ar.allocate(QWTensorShape([tokens * hidden], .f32))
            try step { enc in
                try QWGemm.encodeMoEGateUpF32R1(enc, hal: Self.hal, x: xe32, bank: guView, ids: idsT,
                                                out: actRef, rowsPerExpert: guRows, routes: routes,
                                                tokens: tokens, fuseSiluFwht: true,
                                                reuseExpertsAcrossTokens: false)
                enc.memoryBarrier(scope: .buffers)
                try QWGemm.encodeMoEDownSharedCombineF32(enc, hal: Self.hal, x: actRef, bank: dnView,
                                                         rowsPerRoute: hidden, routes: routes,
                                                         weights: wRoute, shared: shared, gate: gate,
                                                         out: outRef, ids: idsT, tokens: tokens,
                                                         reuseExpertsAcrossTokens: false)
            }

            let expOff = try ar.allocate(QWTensorShape([experts + 1], .i32))
            let sortedPair = try ar.allocate(QWTensorShape([pairs], .i32))
            let sortedTok = try ar.allocate(QWTensorShape([pairs], .i32))
            let invPerm = try ar.allocate(QWTensorShape([pairs], .i32))
            let tileE = try ar.allocate(QWTensorShape([tilePlan.maxTiles], .i32))
            let tileR = try ar.allocate(QWTensorShape([tilePlan.maxTiles], .i32))
            let tcIndirect = try ar.allocate(QWTensorShape([tilePlan.scratchInts], .i32))
            let gu16 = try ar.allocate(QWTensorShape([pairs * guRows], .f16))
            let dnHi = try ar.allocate(QWTensorShape([pairs * inter], .f16))
            let dnAcc = try ar.allocate(QWTensorShape([pairs * hidden], .f32))
            let outInd = try ar.allocate(QWTensorShape([tokens * hidden], .f32))

            guard let pair = Self.hal.begin() else { throw XCTSkip("no command buffer") }
            let cb = pair.0
            var enc = pair.1
            try QWGemm.encodeMoEGroupPairsF32(enc, hal: Self.hal, ids: idsT, expertOff: expOff,
                                              sortedPairIdx: sortedPair, sortedTok: sortedTok,
                                              experts: experts, totalPairs: pairs, routes: routes,
                                              invPerm: invPerm, tileE: tileE, tileR: tileR,
                                              tileCount: tcIndirect, tilePlan: tilePlan)
            try Self.hal.rotateEncoder(cb, &enc)
            try QWGemm.encodeMoEGateUpNax(enc, hal: Self.hal, xe: xe16, bank: guView,
                                          expertOff: expOff, tileE: tileE, tileR: tileR,
                                          tileCount: tcIndirect, gu16: gu16, rowsPerExpert: guRows,
                                          tilePlan: tilePlan, sortedTok: sortedTok)
            enc.memoryBarrier(scope: .buffers)
            try QWGemm.encodeSiluMulSplitF16(enc, hal: Self.hal, gu: gu16, outHi: dnHi, outLo: dnHi,
                                             rows: pairs, intermediate: inter)
            enc.memoryBarrier(scope: .buffers)
            try QWGemm.encodeMoENaxSplitF32(enc, hal: Self.hal, xHi: dnHi, xLo: dnHi, bank: dnView,
                                            expertOff: expOff, tileE: tileE, tileR: tileR,
                                            tileCount: tcIndirect, y32: dnAcc, rowsPerExpert: hidden,
                                            tilePlan: tilePlan)
            enc.memoryBarrier(scope: .buffers)
            try QWGemm.encodeMoEDownReduceCombineF32(enc, hal: Self.hal, dnAcc: dnAcc, invPerm: invPerm,
                                                     weights: wRoute, shared: shared, gate: gate,
                                                     out: outInd, hidden: hidden, routes: routes, tokens: tokens)
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)

            let ref = arena.read(outRef, as: Float.self, count: tokens * hidden)
            let got = arena.read(outInd, as: Float.self, count: tokens * hidden)
            var sumSq = 0.0, refSq = 0.0
            for i in 0..<ref.count {
                let d = Double(got[i]) - Double(ref[i])
                sumSq += d * d
                refSq += Double(ref[i]) * Double(ref[i])
            }
            let rel = (sumSq / max(refSq, 1e-30)).squareRoot()
            XCTAssertLessThan(rel, 5e-3, "SHVQ\(bits)_G\(group) MoE NAX vs F32 rel=\(rel)")
        }
    }

    func testMLAAndDSpineAndHCKernelsForSHVQ1AndSHVQ15() throws {
        for (bits, group) in [(1, 64), (15, 64)] {
            // 1. DSpine refine_scores parity vs CPU unpack
            let hDim = 128, vocab = 32, topK = 4, dSteps = 2
            let headBank = makeSHVQBank(bits: bits, group: group, n: vocab, k: hDim)
            let headRows = unpackBankRows(bits: bits, group: group, n: vocab, k: hDim, bank: headBank)
            let arena = try QWArena(hal: Self.hal, weights: headBank.count + (2 << 20),
                                    activations: (8 << 20), kv: 0, gdnStates: 0)
            let wr = arena.weightsRegion(), ar = arena.activationsRegion()
            let hBuf = try wr.allocate(QWTensorShape([(headBank.count + 1) / 2], .f16))
            arena.copyIn(hBuf, from: headBank)

            let actVals = makeActivations(count: dSteps * 16 * hDim, seedInit: UInt32(0x7100 + bits))
            let refineAct = try ar.allocate(QWTensorShape([dSteps * 16 * hDim], .f16))
            let cands = try ar.allocate(QWTensorShape([dSteps * 16], .i32))
            let table = try ar.allocate(QWTensorShape([dSteps * 16 * 16], .f32))
            arena.write(refineAct, from: actVals.map { Float16($0) })
            var candVals = [Int32](repeating: 0, count: dSteps * 16)
            for d in 0..<dSteps {
                for k in 0..<topK { candVals[d * 16 + k] = Int32((d * 7 + k * 5 + 3) % vocab) }
            }
            arena.write(cands, from: candVals)
            try step { enc in
                let pso = try Self.hal.pipeline("qwg_dspine_refine_scores_sub2")
                enc.setComputePipelineState(pso)
                hBuf.bind(enc, index: 0)
                refineAct.bind(enc, index: 1)
                cands.bind(enc, index: 2)
                table.bind(enc, index: 3)
                var hU = UInt32(hDim), topKU = UInt32(topK)
                var headMode = UInt32(0x8000_0000) | UInt32(bits) | (UInt32(group) << 8)
                enc.setBytes(&hU, length: 4, index: 4)
                enc.setBytes(&topKU, length: 4, index: 5)
                enc.setBytes(&headMode, length: 4, index: 6)
                enc.dispatchThreadgroups(MTLSize(width: 16, height: dSteps, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            }
            let gotTable = arena.read(table, as: Float.self, count: dSteps * 16 * 16)
            for d in 0..<dSteps {
                let kLimit = d == 0 ? 1 : topK
                for kCurr in 0..<topK {
                    let cand = Int(candVals[d * 16 + kCurr])
                    for kp in 0..<kLimit {
                        let base = (d * 16 + kp) * hDim
                        let xSlice = actVals[base..<(base + hDim)].map { Float(Float16($0)) }[...]
                        let want = dotRef(w: headRows[cand], x: xSlice).val
                        XCTAssertEqual(gotTable[(d * 16 + kp) * 16 + kCurr], want, accuracy: 2e-3)
                    }
                }
            }

            // 2. MLA qwg_mla_absorb_q parity (isIncumbent = 2)
            let tRows = 2, qHeads = 2, nopeDim = 64, vHeadDim = 64, kvLoraRank = 128
            let kvBRows = qHeads * (nopeDim + vHeadDim)
            let kvBBank = makeSHVQBank(bits: bits, group: group, n: kvBRows, k: kvLoraRank)
            let kvBUnpacked = unpackBankRows(bits: bits, group: group, n: kvBRows, k: kvLoraRank, bank: kvBBank)
            let kvBBuf = try wr.allocate(QWTensorShape([(kvBBank.count + 1) / 2], .f16))
            arena.copyIn(kvBBuf, from: kvBBank)
            let qStride = qHeads * nopeDim
            let qVals = makeActivations(count: tRows * qStride, seedInit: UInt32(0x8200 + bits))
            let qFull = try ar.allocate(QWTensorShape([tRows * qStride], .f16))
            let qAbs = try ar.allocate(QWTensorShape([tRows * qHeads * kvLoraRank], .f16))
            arena.write(qFull, from: qVals.map { Float16($0) })
            try step { enc in
                let pso = try Self.hal.pipeline("qwg_mla_absorb_q_sub2")
                enc.setComputePipelineState(pso)
                qFull.bind(enc, index: 0)
                kvBBuf.bind(enc, index: 1)
                qAbs.bind(enc, index: 2)
                var p = MLAAbsorbParamsPod(T: UInt32(tRows), qHeads: UInt32(qHeads), nopeDim: UInt32(nopeDim),
                                           vHeadDim: UInt32(vHeadDim), kvLoraRank: UInt32(kvLoraRank),
                                           qFullStride: UInt32(qStride), bits: UInt32(bits),
                                           group: UInt32(group), isIncumbent: 2)
                enc.setBytes(&p, length: MemoryLayout<MLAAbsorbParamsPod>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: qHeads, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
            }
            let gotAbs = arena.read(qAbs, as: Float16.self, count: tRows * qHeads * kvLoraRank).map { Float($0) }
            for t in 0..<tRows {
                for h in 0..<qHeads {
                    let rowBase = h * (nopeDim + vHeadDim)
                    for c in 0..<kvLoraRank {
                        var sum: Float = 0
                        for r in 0..<nopeDim {
                            let qv = Float(Float16(qVals[t * qStride + h * nopeDim + r]))
                            sum += qv * kvBUnpacked[rowBase + r][c]
                        }
                        XCTAssertEqual(gotAbs[(t * qHeads + h) * kvLoraRank + c], Float(Float16(sum)), accuracy: 3e-3)
                    }
                }
            }
            let attOut = try ar.allocate(QWTensorShape([tRows * qHeads * vHeadDim], .f16))
            try step { enc in
                let pso = try Self.hal.pipeline("qwg_mla_proj_uv_sub2")
                enc.setComputePipelineState(pso)
                qAbs.bind(enc, index: 0)
                kvBBuf.bind(enc, index: 1)
                attOut.bind(enc, index: 2)
                var p = MLAAbsorbParamsPod(T: UInt32(tRows), qHeads: UInt32(qHeads), nopeDim: UInt32(nopeDim),
                                           vHeadDim: UInt32(vHeadDim), kvLoraRank: UInt32(kvLoraRank),
                                           qFullStride: UInt32(qStride), bits: UInt32(bits),
                                           group: UInt32(group), isIncumbent: 2)
                enc.setBytes(&p, length: MemoryLayout<MLAAbsorbParamsPod>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: qHeads, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            let gotUV = arena.read(attOut, as: Float16.self, count: tRows * qHeads * vHeadDim).map { Float($0) }
            XCTAssertTrue(gotUV.allSatisfy { $0.isFinite })

            // 3. Hyper-connection down_inject kernel parity vs CPU unpack
            let nDown = 8, kHC = 128, hcStreams = 4
            let dnHCBank = makeSHVQBank(bits: bits, group: group, n: nDown, k: kHC)
            let dnHCRows = unpackBankRows(bits: bits, group: group, n: nDown, k: kHC, bank: dnHCBank)
            let dnHCBuf = try wr.allocate(QWTensorShape([(dnHCBank.count + 1) / 2], .f16))
            arena.copyIn(dnHCBuf, from: dnHCBank)
            let xRotVals = makeActivations(count: kHC, seedInit: UInt32(0x9300 + bits))
            let xRotT = try ar.allocate(QWTensorShape([kHC], .f32))
            let yLowT = try ar.allocate(QWTensorShape([nDown], .f32))
            let injWT = try ar.allocate(QWTensorShape([hcStreams * kHC], .f16))
            let xXnT = try ar.allocate(QWTensorShape([kHC], .f32))
            let yInjT = try ar.allocate(QWTensorShape([hcStreams], .f32))
            arena.write(xRotT, from: xRotVals)
            arena.write(injWT, from: [UInt16](repeating: 0, count: hcStreams * kHC))
            arena.write(xXnT, from: [Float](repeating: 0, count: kHC))
            let hcDownName = try XCTUnwrap(QWGemmAQ.hcDownInjectF32KernelName(bits: bits, group: group, isSHVQ: true))
            try step { enc in
                let pso = try Self.hal.pipeline(hcDownName)
                enc.setComputePipelineState(pso)
                dnHCBuf.bind(enc, index: 0)
                xRotT.bind(enc, index: 1)
                yLowT.bind(enc, index: 2)
                injWT.bind(enc, index: 3)
                xXnT.bind(enc, index: 4)
                yInjT.bind(enc, index: 5)
                var nU = UInt32(nDown), kU = UInt32(kHC), hcU = UInt32(hcStreams)
                var scaleF: Float = 2.0
                enc.setBytes(&nU, length: 4, index: 6)
                enc.setBytes(&kU, length: 4, index: 7)
                enc.setBytes(&scaleF, length: 4, index: 8)
                enc.setBytes(&hcU, length: 4, index: 9)
                enc.dispatchThreadgroups(MTLSize(width: (nDown + 3) / 4 + hcStreams, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
            let gotLow = arena.read(yLowT, as: Float.self, count: nDown)
            for r in 0..<nDown {
                let z = dotRef(w: dnHCRows[r], x: xRotVals[...]).val / 2.0
                let want = z / (1.0 + exp(-z))
                XCTAssertEqual(gotLow[r], want, accuracy: 2e-3)
            }

            // 4. Hyper-connection up_mix_h32 parity vs CPU unpack
            let hiddenHC = 64, lowRank = 64
            let upBank = makeSHVQBank(bits: bits, group: group, n: hcStreams * hiddenHC, k: lowRank)
            let upRows = unpackBankRows(bits: bits, group: group, n: hcStreams * hiddenHC, k: lowRank, bank: upBank)
            let upBuf = try wr.allocate(QWTensorShape([(upBank.count + 1) / 2], .f16))
            arena.copyIn(upBuf, from: upBank)
            let hcLowVals = makeActivations(count: lowRank, seedInit: UInt32(0xA400 + bits))
            let xnVals = makeActivations(count: hcStreams * hiddenHC, seedInit: UInt32(0xB500 + bits))
            let hcLowT = try ar.allocate(QWTensorShape([lowRank], .f32))
            let xnT = try ar.allocate(QWTensorShape([hcStreams * hiddenHC], .f32))
            let mixedT = try ar.allocate(QWTensorShape([hiddenHC], .f32))
            let rotMixedT = try ar.allocate(QWTensorShape([hiddenHC], .f32))
            arena.write(hcLowT, from: hcLowVals)
            arena.write(xnT, from: xnVals)
            let hcUpName = try XCTUnwrap(QWGemmAQ.hcUpMixH32F32KernelName(bits: bits, group: group, isSHVQ: true))
            try step { enc in
                let pso = try Self.hal.pipeline(hcUpName)
                enc.setComputePipelineState(pso)
                upBuf.bind(enc, index: 0)
                hcLowT.bind(enc, index: 1)
                xnT.bind(enc, index: 2)
                mixedT.bind(enc, index: 3)
                rotMixedT.bind(enc, index: 4)
                var hU = UInt32(hiddenHC), hcU = UInt32(hcStreams), rU = UInt32(lowRank)
                enc.setBytes(&hU, length: 4, index: 5)
                enc.setBytes(&hcU, length: 4, index: 6)
                enc.setBytes(&rU, length: 4, index: 7)
                enc.dispatchThreadgroups(MTLSize(width: hiddenHC / 32, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            var lowRot = hcLowVals
            lowRot.withUnsafeMutableBufferPointer { buf in
                for off in stride(from: 0, to: lowRank, by: 32) {
                    QWQuant.fwht32OrthoBlock(buf.baseAddress! + off)
                }
            }
            let gotMixed = arena.read(mixedT, as: Float.self, count: hiddenHC)
            for h in 0..<hiddenHC {
                var sum: Float = 0
                for c in 0..<hcStreams {
                    let u = dotRef(w: upRows[c * hiddenHC + h], x: lowRot[...]).val
                    let g = 1.0 / (1.0 + exp(-u))
                    sum += g * xnVals[c * hiddenHC + h]
                }
                XCTAssertEqual(gotMixed[h], sum / Float(hcStreams), accuracy: 2e-3)
            }
        }
    }
}
