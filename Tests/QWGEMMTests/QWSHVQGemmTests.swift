// QWSHVQGemmTests.swift — GPU parity and micro-benchmark tests for the full
// SHVQ{2,3,4}_G{32,64,128,256} Metal kernel family (decode GEMV, f32 GEMV/BM,
// head argmax, batched NAX GEMM, and MoE gateup/down/NAX).
import Foundation
import Metal
import XCTest
@testable import QWGEMM
import QWAlloc
import QWCore
import QWHAL
import QWQuant

final class QWSHVQGemmTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private func makeSHVQBank(bits: Int, group: Int, n: Int, k: Int, scale: Float = 0.0625) -> Data {
        let groups = k / group
        let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
        var seed: UInt32 = 0x9E37_79B9 ^ UInt32(bits * 131 + group)
        var bank = Data()
        bank.reserveCapacity(n * groups * gBytes)
        for r in 0..<n {
            for g in 0..<groups {
                var block = [UInt8](repeating: 0, count: gBytes)
                let sc = Float16(scale * (1.0 + 0.125 * Float((r + g) % 5))).bitPattern
                block[0] = UInt8(sc & 0xFF)
                block[1] = UInt8(sc >> 8)
                for b in 2..<gBytes {
                    seed = seed &* 1664525 &+ 1013904223
                    block[b] = UInt8((seed >> 16) & 0xFF)
                }
                bank.append(contentsOf: block)
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
                sub.withUnsafeBytes { QWSHVQBlock.unpack(group: $0, bits: bits, group: group, into: &tmp) }
                row.append(contentsOf: tmp)
            }
            rows.append(row)
        }
        return rows
    }

    private func makeActivations(count: Int, seedInit: UInt32 = 0x5EED_C0DE) -> [Float] {
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
        var acc: Float = 0
        var mag: Float = 0
        for (wi, xi) in zip(w, x) {
            let ww = roundWeightsToF16 ? Float(Float16(wi)) : wi
            let p = ww * xi
            acc += p
            mag += abs(p)
        }
        return (acc, mag)
    }

    func testDecodeT4AndMultiRowMatchHostAcrossAll12Pairs() throws {
        for (bits, group) in QWSHVQBlock.supported {
            let n = 19, k = max(512, group * 2)
            let bank = makeSHVQBank(bits: bits, group: group, n: n, k: k)
            let wRows = unpackBankRows(bits: bits, group: group, n: n, k: k, bank: bank)
            let arena = try QWArena(hal: Self.hal, weights: bank.count + (2 << 20),
                                    activations: (4 << 20), kv: 0, gdnStates: 0)
            let wBuf = try arena.weightsRegion().allocate(QWTensorShape([(bank.count + 1) / 2], .f16))
            arena.copyIn(wBuf, from: bank)
            let act = arena.activationsRegion()
            // 1. Test encodeDecodeT4 (_dx_h16_t4, f16 output)
            let xRaw1 = makeActivations(count: k, seedInit: UInt32(97 + bits))
            let x32_1 = try act.allocate(QWTensorShape([k], .f32))
            let x16_1 = try act.allocate(QWTensorShape([k], .f16))
            let y16_1 = try act.allocate(QWTensorShape([n], .f16))
            arena.write(x32_1, from: xRaw1)
            try step { enc in
                try QWGemm.encodeCast(enc, hal: Self.hal, x32: x32_1, y16: x16_1, count: k)
                try QWGemmAQ.encodeDecodeT4(enc, hal: Self.hal, blocks: wBuf, x: x16_1, y16: y16_1,
                                            n: n, k: k, bits: bits, group: group, isSHVQ: true)
            }
            let gotT4 = arena.read(y16_1, as: Float16.self, count: n).map { Float($0) }
            let x1Slice = xRaw1.map { Float(Float16($0)) }[...]
            for r in 0..<n {
                let ref = dotRef(w: wRows[r], x: x1Slice)
                let denom = max(Double(abs(ref.val)), Double(ref.mag), 1e-5)
                let rel = abs(Double(gotT4[r]) - Double(ref.val)) / denom
                XCTAssertLessThan(rel, 1.5e-3, "decodeT4 SHVQ\(bits)_G\(group) r=\(r): got \(gotT4[r]) want \(ref.val)")
            }

            // 2. Test QWGemmAQ.encode (_v4 multi-row f32 output, m=3)
            let m = 3
            let xRaw = makeActivations(count: m * k, seedInit: UInt32(m * 97 + bits))
            let x32 = try act.allocate(QWTensorShape([m * k], .f32))
            let x16 = try act.allocate(QWTensorShape([m * k], .f16))
            let y32 = try act.allocate(QWTensorShape([m * n], .f32))
            arena.write(x32, from: xRaw)
            try step { enc in
                try QWGemm.encodeCast(enc, hal: Self.hal, x32: x32, y16: x16, count: m * k)
                try QWGemmAQ.encode(enc, hal: Self.hal, blocks: wBuf, x: x16, y: y32,
                                    n: n, k: k, bits: bits, group: group, rows: m, isSHVQ: true)
            }
            let got = arena.read(y32, as: Float.self, count: m * n)
            for t in 0..<m {
                let xSlice = xRaw[(t * k)..<((t + 1) * k)].map { Float(Float16($0)) }[...]
                for r in 0..<n {
                    let ref = dotRef(w: wRows[r], x: xSlice)
                    let denom = max(Double(abs(ref.val)), Double(ref.mag), 1e-5)
                    let rel = abs(Double(got[t * n + r]) - Double(ref.val)) / denom
                    XCTAssertLessThan(rel, 1e-4,
                                      "SHVQ\(bits)_G\(group) m=\(m) t=\(t) r=\(r): got \(got[t * n + r]) want \(ref.val)")
                }
            }
        }
    }

    func testF32GemvV4AndBMAndArgmaxMatchHostAcrossAll12Pairs() throws {
        for (bits, group) in QWSHVQBlock.supported {
            let n = 32, k = max(512, group * 2)
            let bank = makeSHVQBank(bits: bits, group: group, n: n, k: k)
            let wRows = unpackBankRows(bits: bits, group: group, n: n, k: k, bank: bank)
            let arena = try QWArena(hal: Self.hal, weights: bank.count + (2 << 20),
                                    activations: (8 << 20), kv: 0, gdnStates: 0)
            let wBuf = try arena.weightsRegion().allocate(QWTensorShape([(bank.count + 1) / 2], .f16))
            arena.copyIn(wBuf, from: bank)
            let weight = try XCTUnwrap(QWWeightView(blocks: wBuf, n: n, k: k, bits: bits,
                                                    groupSize: group, isRotated: true, isSHVQ: true))
            let act = arena.activationsRegion()

            // 1. Test encodeHQ4F32V4 for m=1 and m=3 (fused multi-row m4 path)
            for m in [1, 3] {
                let x = makeActivations(count: m * k, seedInit: UInt32(0x1234 + m + group))
                let xT = try act.allocate(QWTensorShape([m * k], .f32))
                let yT = try act.allocate(QWTensorShape([m * n], .f32))
                arena.write(xT, from: x)
                try step { enc in
                    try QWGemm.encodeHQ4F32V4(enc, hal: Self.hal, weight: weight, x: xT, y: yT, rows: m)
                }
                let got = arena.read(yT, as: Float.self, count: m * n)
                for t in 0..<m {
                    let xSlice = x[(t * k)..<((t + 1) * k)]
                    for r in 0..<n {
                        let ref = dotRef(w: wRows[r], x: xSlice)
                        let denom = max(Double(abs(ref.val)), Double(ref.mag), 1e-5)
                        let rel = abs(Double(got[t * n + r]) - Double(ref.val)) / denom
                        XCTAssertLessThan(rel, 1e-4,
                                          "f32_v4 SHVQ\(bits)_G\(group) m=\(m) t=\(t) r=\(r): rel=\(rel)")
                    }
                }
            }

            // 2. Test encodeHQ4F32BM for m=8
            let mBM = 8
            let xBM = makeActivations(count: mBM * k, seedInit: UInt32(0x7777 + bits))
            let xBMT = try act.allocate(QWTensorShape([mBM * k], .f32))
            let yBMT = try act.allocate(QWTensorShape([mBM * n], .f32))
            arena.write(xBMT, from: xBM)
            try step { enc in
                try QWGemm.encodeHQ4F32BM(enc, hal: Self.hal, weight: weight, x: xBMT, y: yBMT, rows: mBM)
            }
            let gotBM = arena.read(yBMT, as: Float.self, count: mBM * n)
            for t in 0..<mBM {
                let xSlice = xBM[(t * k)..<((t + 1) * k)]
                for r in 0..<n {
                    let ref = dotRef(w: wRows[r], x: xSlice)
                    let denom = max(Double(abs(ref.val)), Double(ref.mag), 1e-5)
                    let rel = abs(Double(gotBM[t * n + r]) - Double(ref.val)) / denom
                    XCTAssertLessThan(rel, 1e-4, "f32_bm SHVQ\(bits)_G\(group) t=\(t) r=\(r): rel=\(rel)")
                }
            }

            // 3. Test encodeHQ4F32V4Argmax + encodeArgmaxFin (m=1 and m=2 am2 path)
            for mAm in [1, 2] {
                let slots = QWGemm.blockArgmaxSlots(n)
                let xAm = makeActivations(count: mAm * k, seedInit: UInt32(0xBEEF + mAm * 31 + bits))
                let xAmT = try act.allocate(QWTensorShape([mAm * k], .f32))
                let yAmT = try act.allocate(QWTensorShape([mAm * n], .f32))
                let kHi = try act.allocate(QWTensorShape([mAm * slots], .f32))
                let kLo = try act.allocate(QWTensorShape([mAm * slots], .i32))
                let tokOut = try act.allocate(QWTensorShape([mAm], .i32))
                arena.write(xAmT, from: xAm)
                try step { enc in
                    try QWGemm.encodeHQ4F32V4Argmax(enc, hal: Self.hal, weight: weight, x: xAmT, y: yAmT,
                                                    keysHi: kHi, keysLo: kLo, rows: mAm)
                    enc.memoryBarrier(scope: .buffers)
                    try QWGemm.encodeArgmaxFin(enc, hal: Self.hal, keysHi: kHi, keysLo: kLo,
                                               token: tokOut, slots: slots, rows: mAm)
                }
                let gotToks = arena.read(tokOut, as: Int32.self, count: mAm)
                for t in 0..<mAm {
                    let xSlice = xAm[(t * k)..<((t + 1) * k)]
                    var bestRow = 0
                    var bestVal = -Float.infinity
                    for r in 0..<n {
                        let v = dotRef(w: wRows[r], x: xSlice).val
                        if v > bestVal { bestVal = v; bestRow = r }
                    }
                    XCTAssertEqual(Int(gotToks[t]), bestRow, "argmax SHVQ\(bits)_G\(group) m=\(mAm) t=\(t)")
                }
            }
        }
    }

    func testPrefillNaxH16AndNaxF32MatchHostAcrossAll12Pairs() throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "SHVQ NAX prefill requires Metal 4 tensor hardware")
        for (bits, group) in QWSHVQBlock.supported {
            let n = 64, k = 256
            let bank = makeSHVQBank(bits: bits, group: group, n: n, k: k)
            let wRows = unpackBankRows(bits: bits, group: group, n: n, k: k, bank: bank)
            let arena = try QWArena(hal: Self.hal, weights: bank.count + (2 << 20),
                                    activations: (8 << 20), kv: 0, gdnStates: 0)
            let wBuf = try arena.weightsRegion().allocate(QWTensorShape([(bank.count + 1) / 2], .f16))
            arena.copyIn(wBuf, from: bank)
            let weight = try XCTUnwrap(QWWeightView(blocks: wBuf, n: n, k: k, bits: bits,
                                                    groupSize: group, isRotated: true, isSHVQ: true))
            let act = arena.activationsRegion()
            for m in [3, 64] {
                let x = makeActivations(count: m * k, seedInit: UInt32(0x4242 + m + bits * 17))
                let x32 = try act.allocate(QWTensorShape([m * k], .f32))
                let x16 = try act.allocate(QWTensorShape([m * k], .f16))
                let y16 = try act.allocate(QWTensorShape([m * n], .f16))
                let y32 = try act.allocate(QWTensorShape([m * n], .f32))
                arena.write(x32, from: x)
                try step { enc in
                    try QWGemm.encodeCast(enc, hal: Self.hal, x32: x32, y16: x16, count: m * k)
                    enc.memoryBarrier(scope: .buffers)
                    try QWGemm.encodeNaxF32(enc, hal: Self.hal, weight: weight,
                                            x16: x16, y16: x16, y32: y32, rows: m)
                }
                let got32 = arena.read(y32, as: Float.self, count: m * n)
                for t in 0..<m {
                    let xSlice = x[(t * k)..<((t + 1) * k)].map { Float(Float16($0)) }[...]
                    for r in 0..<n {
                        let ref = dotRef(w: wRows[r], x: xSlice, roundWeightsToF16: true)
                        let denom = max(Double(abs(ref.val)), Double(ref.mag), 1e-5)
                        let rel = abs(Double(got32[t * n + r]) - Double(ref.val)) / denom
                        XCTAssertLessThan(rel, 1e-3,
                                          "nax_f32 SHVQ\(bits)_G\(group) m=\(m) t=\(t) r=\(r): got \(got32[t * n + r]) want \(ref.val)")
                    }
                }
                // Also verify H16 batched paths (BM64 & BM128)
                if m == 64 {
                    for bm128 in [false, true] {
                        let kName = try XCTUnwrap(
                            QWGemmAQ.gemmKernelName(bits: bits, group: group, bm128: bm128, isSHVQ: true)
                        )
                        let bm = bm128 ? 128 : 64
                        try step { enc in
                            let pso = try Self.hal.pipeline(kName)
                            enc.setComputePipelineState(pso)
                            wBuf.bind(enc, index: 0)
                            x16.bind(enc, index: 1)
                            y16.bind(enc, index: 2)
                            y16.bind(enc, index: 3)
                            var m32 = UInt32(m), n32 = UInt32(n), k32 = UInt32(k)
                            enc.setBytes(&m32, length: 4, index: 4)
                            enc.setBytes(&n32, length: 4, index: 5)
                            enc.setBytes(&k32, length: 4, index: 6)
                            enc.dispatchThreadgroups(
                                MTLSize(width: (n + 63) / 64, height: (m + bm - 1) / bm, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
                            )
                        }
                        let got16 = arena.read(y16, as: Float16.self, count: m * n).map { Float($0) }
                        for t in 0..<m {
                            let xSlice = x[(t * k)..<((t + 1) * k)].map { Float(Float16($0)) }[...]
                            for r in 0..<n {
                                let ref = dotRef(w: wRows[r], x: xSlice, roundWeightsToF16: true)
                                let denom = max(Double(abs(ref.val)), Double(ref.mag), 1e-5)
                                let rel = abs(Double(got16[t * n + r]) - Double(ref.val)) / denom
                                XCTAssertLessThan(rel, 1.5e-3, "\(kName) t=\(t) r=\(r): rel=\(rel)")
                            }
                        }
                    }
                }
            }
        }
    }

    func testMoEGateUpAndDownCombineMatchHostAcrossBitWidths() throws {
        for (bits, group) in [(2, 64), (3, 64), (4, 32), (4, 64)] {
            let experts = 3
            let inter = 64
            let hidden = 256
            let gateUpRowsPerExp = inter * 2
            let guBank = makeSHVQBank(bits: bits, group: group, n: experts * gateUpRowsPerExp, k: hidden)
            let guWeights = unpackBankRows(bits: bits, group: group, n: experts * gateUpRowsPerExp, k: hidden, bank: guBank)
            let dnBank = makeSHVQBank(bits: bits, group: group, n: experts * hidden, k: inter)
            let dnWeights = unpackBankRows(bits: bits, group: group, n: experts * hidden, k: inter, bank: dnBank)

            let arena = try QWArena(hal: Self.hal, weights: guBank.count + dnBank.count + (2 << 20),
                                    activations: (4 << 20), kv: 0, gdnStates: 0)
            let wReg = arena.weightsRegion()
            let guBuf = try wReg.allocate(QWTensorShape([(guBank.count + 1) / 2], .f16))
            let dnBuf = try wReg.allocate(QWTensorShape([(dnBank.count + 1) / 2], .f16))
            arena.copyIn(guBuf, from: guBank)
            arena.copyIn(dnBuf, from: dnBank)
            let guView = try XCTUnwrap(QWWeightView(blocks: guBuf, n: experts * gateUpRowsPerExp, k: hidden,
                                                    bits: bits, groupSize: group, isRotated: true, isSHVQ: true))
            let dnView = try XCTUnwrap(QWWeightView(blocks: dnBuf, n: experts * hidden, k: inter,
                                                    bits: bits, groupSize: group, isRotated: true, isSHVQ: true))

            let act = arena.activationsRegion()
            let routes = 2
            let xVec = makeActivations(count: hidden, seedInit: 0xABCD)
            let xT = try act.allocate(QWTensorShape([hidden], .f32))
            let idsT = try act.allocate(QWTensorShape([routes], .i32))
            let guOut = try act.allocate(QWTensorShape([routes * gateUpRowsPerExp], .f32))
            arena.write(xT, from: xVec)
            let chosen: [Int32] = [2, 0]
            arena.write(idsT, from: chosen)

            try step { enc in
                try QWGemm.encodeMoEGateUpF32R1(enc, hal: Self.hal, x: xT, bank: guView, ids: idsT, out: guOut,
                                                rowsPerExpert: gateUpRowsPerExp, routes: routes, tokens: 1,
                                                fuseSiluFwht: false)
            }
            let gotGU = arena.read(guOut, as: Float.self, count: routes * gateUpRowsPerExp)
            for rt in 0..<routes {
                let e = Int(chosen[rt])
                for r in 0..<gateUpRowsPerExp {
                    let ref = dotRef(w: guWeights[e * gateUpRowsPerExp + r], x: xVec[...])
                    let denom = max(Double(abs(ref.val)), Double(ref.mag), 1e-5)
                    let rel = abs(Double(gotGU[rt * gateUpRowsPerExp + r]) - Double(ref.val)) / denom
                    XCTAssertLessThan(rel, 1e-4, "moe_gateup SHVQ\(bits)_G\(group) rt=\(rt) r=\(r): rel=\(rel)")
                }
            }

            // Test MoE down + shared combine
            let interIn = makeActivations(count: routes * inter, seedInit: 0xDCBA)
            let interT = try act.allocate(QWTensorShape([routes * inter], .f32))
            let rwT = try act.allocate(QWTensorShape([routes], .f32))
            let sharedT = try act.allocate(QWTensorShape([hidden], .f32))
            let gateT = try act.allocate(QWTensorShape([1], .f32))
            let dnOutT = try act.allocate(QWTensorShape([hidden], .f32))
            let routeWeights: [Float] = [0.625, 0.375]
            let sharedVec = makeActivations(count: hidden, seedInit: 0x5555)
            let gateVal: [Float] = [0.5]
            arena.write(interT, from: interIn)
            arena.write(rwT, from: routeWeights)
            arena.write(sharedT, from: sharedVec)
            arena.write(gateT, from: gateVal)

            try step { enc in
                try QWGemm.encodeMoEDownSharedCombineF32(
                    enc, hal: Self.hal, x: interT, bank: dnView, rowsPerRoute: hidden, routes: routes,
                    weights: rwT, shared: sharedT, gate: gateT, out: dnOutT, ids: idsT, tokens: 1
                )
            }
            let gotDn = arena.read(dnOutT, as: Float.self, count: hidden)
            for h in 0..<hidden {
                var want = 0.5 * sharedVec[h]
                var mag = abs(want)
                for rt in 0..<routes {
                    let e = Int(chosen[rt])
                    let ref = dotRef(w: dnWeights[e * hidden + h],
                                     x: interIn[(rt * inter)..<((rt + 1) * inter)])
                    want += routeWeights[rt] * ref.val
                    mag += routeWeights[rt] * ref.mag
                }
                let denom = max(Double(abs(want)), Double(mag), 1e-5)
                let rel = abs(Double(gotDn[h]) - Double(want)) / denom
                XCTAssertLessThan(rel, 1e-4, "moe_down SHVQ\(bits)_G\(group) h=\(h): rel=\(rel)")
            }
        }
    }

    func testSHVQ2And3SRAMStagingNonMultipleTailAndFusedGateUpParity() throws {
        for (bits, group) in [(2, 32), (2, 64), (3, 32), (3, 64)] {
            let nTail = 22, k = 128
            let bank = makeSHVQBank(bits: bits, group: group, n: nTail, k: k)
            let wRows = unpackBankRows(bits: bits, group: group, n: nTail, k: k, bank: bank)
            let arena = try QWArena(hal: Self.hal, weights: bank.count + (2 << 20),
                                    activations: (4 << 20), kv: 0, gdnStates: 0)
            let wBuf = try arena.weightsRegion().allocate(QWTensorShape([(bank.count + 1) / 2], .f16))
            arena.copyIn(wBuf, from: bank)
            let weight = try XCTUnwrap(QWWeightView(blocks: wBuf, n: nTail, k: k, bits: bits,
                                                    groupSize: group, isRotated: true, isSHVQ: true))
            let act = arena.activationsRegion()
            for m in [1, 3] {
                let x = makeActivations(count: m * k, seedInit: UInt32(0x3100 + m + bits * 11))
                let xT = try act.allocate(QWTensorShape([m * k], .f32))
                let yT = try act.allocate(QWTensorShape([m * nTail], .f32))
                arena.write(xT, from: x)
                try step { enc in
                    try QWGemm.encodeHQ4F32V4(enc, hal: Self.hal, weight: weight, x: xT, y: yT, rows: m)
                }
                let got = arena.read(yT, as: Float.self, count: m * nTail)
                for t in 0..<m {
                    let xSlice = x[(t * k)..<((t + 1) * k)]
                    for r in 0..<nTail {
                        let ref = dotRef(w: wRows[r], x: xSlice)
                        let denom = max(Double(abs(ref.val)), Double(ref.mag), 1e-5)
                        XCTAssertLessThan(abs(Double(got[t * nTail + r]) - Double(ref.val)) / denom, 1e-4)
                    }
                }
            }
            for mAm in [1, 2] {
                let slots = QWGemm.blockArgmaxSlots(nTail)
                let xAm = makeActivations(count: mAm * k, seedInit: UInt32(0x5100 + mAm + bits))
                let xAmT = try act.allocate(QWTensorShape([mAm * k], .f32))
                let yAmT = try act.allocate(QWTensorShape([mAm * nTail], .f32))
                let kHi = try act.allocate(QWTensorShape([mAm * slots], .f32))
                let kLo = try act.allocate(QWTensorShape([mAm * slots], .i32))
                let tokOut = try act.allocate(QWTensorShape([mAm], .i32))
                arena.write(xAmT, from: xAm)
                try step { enc in
                    try QWGemm.encodeHQ4F32V4Argmax(enc, hal: Self.hal, weight: weight, x: xAmT, y: yAmT,
                                                    keysHi: kHi, keysLo: kLo, rows: mAm)
                    enc.memoryBarrier(scope: .buffers)
                    try QWGemm.encodeArgmaxFin(enc, hal: Self.hal, keysHi: kHi, keysLo: kLo,
                                               token: tokOut, slots: slots, rows: mAm)
                }
                let gotToks = arena.read(tokOut, as: Int32.self, count: mAm)
                for t in 0..<mAm {
                    let xSlice = xAm[(t * k)..<((t + 1) * k)]
                    var bestRow = 0, bestVal = -Float.infinity
                    for r in 0..<nTail {
                        let v = dotRef(w: wRows[r], x: xSlice).val
                        if v > bestVal { bestVal = v; bestRow = r }
                    }
                    XCTAssertEqual(Int(gotToks[t]), bestRow)
                }
            }

            let experts = 2, inter = 64, guRows = inter * 2, tokens = 2, routes = 2
            let guBank = makeSHVQBank(bits: bits, group: group, n: experts * guRows, k: k)
            let dnBank = makeSHVQBank(bits: bits, group: group, n: experts * nTail, k: k)
            let dnRows = unpackBankRows(bits: bits, group: group, n: experts * nTail, k: k, bank: dnBank)
            let moeArena = try QWArena(hal: Self.hal, weights: guBank.count + dnBank.count + (2 << 20),
                                       activations: (4 << 20), kv: 0, gdnStates: 0)
            let mw = moeArena.weightsRegion(), ma = moeArena.activationsRegion()
            let guBuf = try mw.allocate(QWTensorShape([(guBank.count + 1) / 2], .f16))
            let dnBuf = try mw.allocate(QWTensorShape([(dnBank.count + 1) / 2], .f16))
            moeArena.copyIn(guBuf, from: guBank)
            moeArena.copyIn(dnBuf, from: dnBank)
            let guView = try XCTUnwrap(QWWeightView(blocks: guBuf, n: experts * guRows, k: k,
                                                    bits: bits, groupSize: group, isRotated: true, isSHVQ: true))
            let dnView = try XCTUnwrap(QWWeightView(blocks: dnBuf, n: experts * nTail, k: k,
                                                    bits: bits, groupSize: group, isRotated: true, isSHVQ: true))
            let xTok = makeActivations(count: tokens * k, seedInit: 0x7711)
            let xTokT = try ma.allocate(QWTensorShape([tokens * k], .f32))
            let idsT = try ma.allocate(QWTensorShape([tokens * routes], .i32))
            let guUnfusedT = try ma.allocate(QWTensorShape([tokens * routes * guRows], .f32))
            let actRefT = try ma.allocate(QWTensorShape([tokens * routes * inter], .f32))
            let actFusedT = try ma.allocate(QWTensorShape([tokens * routes * inter], .f32))
            moeArena.write(xTokT, from: xTok)
            moeArena.write(idsT, from: [Int32(1), 0, 1, 0])
            try step { enc in
                try QWGemm.encodeMoEGateUpF32R1(enc, hal: Self.hal, x: xTokT, bank: guView, ids: idsT,
                                                out: guUnfusedT, rowsPerExpert: guRows, routes: routes,
                                                tokens: tokens, fuseSiluFwht: false)
                enc.memoryBarrier(scope: .buffers)
                try QWGemm.encodeSiluMulF32(enc, hal: Self.hal, gu: guUnfusedT, act: actRefT,
                                            rows: tokens * routes, intermediate: inter)
                try QWGemm.encodeMoEGateUpF32R1(enc, hal: Self.hal, x: xTokT, bank: guView, ids: idsT,
                                                out: actFusedT, rowsPerExpert: guRows, routes: routes,
                                                tokens: tokens, fuseSiluFwht: true)
            }
            let refAct = moeArena.read(actRefT, as: Float.self, count: tokens * routes * inter)
            let gotAct = moeArena.read(actFusedT, as: Float.self, count: tokens * routes * inter)
            for i in 0..<refAct.count {
                XCTAssertEqual(gotAct[i], refAct[i], accuracy: 1e-4, "fused gateup SHVQ\(bits)_G\(group) i=\(i)")
            }

            let interIn = makeActivations(count: routes * k, seedInit: 0x8822)
            let interT = try ma.allocate(QWTensorShape([routes * k], .f32))
            let rwT = try ma.allocate(QWTensorShape([routes], .f32))
            let shT = try ma.allocate(QWTensorShape([nTail], .f32))
            let gateT = try ma.allocate(QWTensorShape([1], .f32))
            let dnOutT = try ma.allocate(QWTensorShape([nTail], .f32))
            let shVec = makeActivations(count: nTail, seedInit: 0x9933)
            moeArena.write(interT, from: interIn)
            moeArena.write(rwT, from: [Float(0.75), 0.25])
            moeArena.write(shT, from: shVec)
            moeArena.write(gateT, from: [Float(0.5)])
            try step { enc in
                try QWGemm.encodeMoEDownSharedCombineF32(
                    enc, hal: Self.hal, x: interT, bank: dnView, rowsPerRoute: nTail, routes: routes,
                    weights: rwT, shared: shT, gate: gateT, out: dnOutT, ids: idsT, tokens: 1)
            }
            let gotDn = moeArena.read(dnOutT, as: Float.self, count: nTail)
            for h in 0..<nTail {
                let r0 = dotRef(w: dnRows[1 * nTail + h], x: interIn[0..<k])
                let r1 = dotRef(w: dnRows[0 * nTail + h], x: interIn[k..<(2 * k)])
                let want = 0.5 * shVec[h] + 0.75 * r0.val + 0.25 * r1.val
                let denom = max(Double(abs(want)), Double(0.75 * r0.mag + 0.25 * r1.mag), 1e-5)
                XCTAssertLessThan(abs(Double(gotDn[h]) - Double(want)) / denom, 1e-4)
            }
        }
    }

    func testGroupedMoEIndirectDispatchMatchesReferenceAndEliminatesEmptySlices() throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "Grouped NAX MoE requires Metal 4 tensor hardware")
        let experts = 64, tokens = 64, routes = 4
        let pairs = tokens * routes
        let hidden = 128, inter = 2048, guRows = inter * 2
        let tilePlan = QWGemm.MoETilePlan(experts: experts, pairs: pairs, gateUpRows: guRows, downRows: hidden)
        XCTAssertEqual(tilePlan.guSlices, 2)
        XCTAssertEqual(tilePlan.dnSlices, 1)

        func fastBank(bits: Int, group: Int, n: Int, k: Int, seed: UInt32) -> Data {
            let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
            let groups = k / group
            var d = Data(count: n * groups * gBytes)
            d.withUnsafeMutableBytes { raw in
                guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                var s = seed
                for r in 0..<n {
                    for g in 0..<groups {
                        let base = (r * groups + g) * gBytes
                        let sc = Float16(0.03125 * Float(1 + (r + g) % 3)).bitPattern
                        ptr[base] = UInt8(sc & 0xFF)
                        ptr[base + 1] = UInt8(sc >> 8)
                        for b in 2..<gBytes {
                            s = s &* 1664525 &+ 1013904223
                            ptr[base + b] = UInt8((s >> 16) & 0xFF)
                        }
                    }
                }
            }
            return d
        }

        for (bits, group) in [(2, 64), (3, 64), (4, 64)] {
            let guData = fastBank(bits: bits, group: group, n: experts * guRows, k: hidden, seed: 0x1111_2222)
            let dnData = fastBank(bits: bits, group: group, n: experts * hidden, k: inter, seed: 0x3333_4444)
            let arena = try QWArena(hal: Self.hal, weights: guData.count + dnData.count + (2 << 20),
                                    activations: (32 << 20), kv: 0, gdnStates: 0)
            let wr = arena.weightsRegion(), ar = arena.activationsRegion()
            let guBuf = try wr.allocate(QWTensorShape([(guData.count + 1) / 2], .f16))
            let dnBuf = try wr.allocate(QWTensorShape([(dnData.count + 1) / 2], .f16))
            arena.copyIn(guBuf, from: guData)
            arena.copyIn(dnBuf, from: dnData)
            let guView = try XCTUnwrap(QWWeightView(blocks: guBuf, n: experts * guRows, k: hidden,
                                                    bits: bits, groupSize: group, isRotated: true, isSHVQ: true))
            let dnView = try XCTUnwrap(QWWeightView(blocks: dnBuf, n: experts * hidden, k: inter,
                                                    bits: bits, groupSize: group, isRotated: true, isSHVQ: true))

            let xe32 = try ar.allocate(QWTensorShape([tokens * hidden], .f32))
            let xe16 = try ar.allocate(QWTensorShape([tokens * hidden], .f16))
            let xeVals = makeActivations(count: tokens * hidden, seedInit: 0xA1B2)
            arena.write(xe32, from: xeVals)
            arena.write(xe16, from: xeVals.map { Float16($0) })
            arena.write(xe32, from: xeVals.map { Float(Float16($0)) })

            // Distinct skewed experts per token: 5 active experts out of 64 -> 5 active tiles (slice 1 = 0).
            var skewedIds = [Int32](repeating: 0, count: pairs)
            for t in 0..<tokens {
                skewedIds[t * routes + 0] = 2
                skewedIds[t * routes + 1] = 17
                skewedIds[t * routes + 2] = 51
                skewedIds[t * routes + 3] = (t < 46) ? 9 : 33
            }
            let idsT = try ar.allocate(QWTensorShape([pairs], .i32))
            arena.write(idsT, from: skewedIds)

            let wRoute = try ar.allocate(QWTensorShape([pairs], .f32))
            arena.write(wRoute, from: [Float](repeating: 0.25, count: pairs))
            let shared = try ar.allocate(QWTensorShape([tokens * hidden], .f32))
            let gate = try ar.allocate(QWTensorShape([tokens], .f32))
            arena.write(shared, from: [Float](repeating: 0, count: tokens * hidden))
            arena.write(gate, from: [Float](repeating: 0, count: tokens))

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

            let tcWords = arena.read(tcIndirect, as: UInt32.self, count: tilePlan.scratchInts)
            // Expert 2: 64 pairs (1 tile), Expert 9: 30 pairs (1 tile), Expert 17: 64 pairs (1 tile), Expert 51: 98 pairs (2 tiles) -> 5 active tiles!
            XCTAssertEqual(tcWords[0], 5, "4 skewed experts (64, 30, 64, 98 pairs) produce 1 + 1 + 1 + 2 = 5 active tiles")
            XCTAssertEqual(Array(tcWords[4..<7]), [UInt32(tilePlan.guColTiles), 5, 1], "gu slice 0 launches exact active tiles")
            XCTAssertEqual(Array(tcWords[7..<10]), [0, 0, 0], "gu slice 1 launches 0 empty threadgroups")
            XCTAssertEqual(Array(tcWords[10..<13]), [UInt32(tilePlan.dnColTiles), 5, 1], "dn slice 0 launches exact active tiles")

            let ref = arena.read(outRef, as: Float.self, count: tokens * hidden)
            let got = arena.read(outInd, as: Float.self, count: tokens * hidden)
            var sumSq = 0.0, refSq = 0.0
            for i in 0..<ref.count {
                let d = Double(got[i]) - Double(ref[i])
                sumSq += d * d
                refSq += Double(ref[i]) * Double(ref[i])
            }
            let rel = (sumSq / max(refSq, 1e-30)).squareRoot()
            XCTAssertLessThan(rel, 5e-3, "SHVQ\(bits)_G\(group) indirect grouped NAX vs f32 reference rel=\(rel)")
        }
    }
}
