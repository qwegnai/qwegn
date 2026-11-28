// QWFPQuantKernelParityTests.swift — Parity tests for BF16 NAX/GEMV, NVFP4_G16,
// MXFP8_G32 zero-LUT unpack, grouped-NAX prefill, active-bucket decode, and softcap SwiGLU.
import Foundation
import Metal
import XCTest
@testable import QWGEMM
import QWAlloc
import QWCore
import QWHAL

final class QWFPQuantKernelParityTests: XCTestCase {
    private func makeHal() throws -> QWHal {
        do {
            return try QWHal(geometry: .qwen38_27b)
        } catch {
            throw XCTSkip("Metal HAL unavailable: \(error)")
        }
    }

    private func sharedSlice(
        _ hal: QWHal, bytes: Int, label: String = "test"
    ) throws -> (any MTLBuffer, QWFPBufferSlice) {
        let len = max(bytes, 64)
        guard let buf = hal.device.makeBuffer(length: len, options: .storageModeShared) else {
            throw XCTSkip("MTLBuffer allocation failed for \(label)")
        }
        return (buf, QWFPBufferSlice(buffer: buf, offset: 0, bytes: len))
    }

    private func makeSlice<T>(
        _ hal: QWHal, values: [T]
    ) throws -> (any MTLBuffer, QWFPBufferSlice) {
        let byteLen = max(values.count * MemoryLayout<T>.stride, 64)
        let (buf, slice) = try sharedSlice(hal, bytes: byteLen)
        values.withUnsafeBytes { raw in
            if let src = raw.baseAddress, raw.count > 0 {
                buf.contents().copyMemory(from: src, byteCount: raw.count)
            }
        }
        return (buf, slice)
    }

    private func runOnGPU(
        _ hal: QWHal,
        _ body: (inout any MTLComputeCommandEncoder, @escaping (inout any MTLComputeCommandEncoder) throws -> Void) throws -> Void
    ) throws {
        guard let (cb, initial) = hal.begin() else {
            throw XCTSkip("Failed to open Metal command buffer")
        }
        var enc = initial
        let rotator: (inout any MTLComputeCommandEncoder) throws -> Void = { current in
            try hal.rotateEncoder(cb, &current)
        }
        try body(&enc, rotator)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        if let err = cb.error {
            XCTFail("GPU command buffer error: \(err)")
        }
    }

    private func bf16Bits(_ v: Float) -> UInt16 {
        var bits = v.bitPattern
        if (bits & 0x7F80_0000) == 0x7F80_0000 { return UInt16(truncatingIfNeeded: bits >> 16) }
        bits = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: bits >> 16)
    }

    private func f32FromBF16(_ u: UInt16) -> Float { Float(bitPattern: UInt32(u) << 16) }
    private func roundBF16(_ v: Double) -> Float { f32FromBF16(bf16Bits(Float(v))) }

    private func fp8E4M3FN(_ b: UInt8) -> Double {
        let s: Double = (b & 0x80) != 0 ? -1.0 : 1.0
        let e = Int((b >> 3) & 0xF), m = Int(b & 0x7)
        if e == 15 && m == 7 { return .nan }
        if e == 0 { return s * pow(2.0, -6.0) * (Double(m) / 8.0) }
        return s * pow(2.0, Double(e - 7)) * (1.0 + Double(m) / 8.0)
    }

    private func ue8m0(_ b: UInt8) -> Double {
        if b == 255 { return .nan }
        if b == 0 { return pow(2.0, -127.0) }
        return pow(2.0, Double(Int(b) - 127))
    }

    private let fp4Table: [Double] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]

    private func softcapSwiGLU(_ g: Double, _ u: Double, gateCap: Double, upCap: Double) -> Double {
        let sig = 1.0 / (1.0 + exp(-g))
        let gc = gateCap > 0 ? (tanh(g / gateCap) * gateCap * sig) : (g * sig)
        let uc = upCap > 0 ? (tanh(u / upCap) * upCap) : u
        return gc * uc
    }

    private struct PackedNVFP4 {
        let weights: [UInt32]
        let scales: [UInt8]
        let s2: [Float]
        let ref: [Double]
    }

    private func packNVFP4(experts: Int, rows: Int, k: Int, seed: UInt64, withS2: Bool) -> PackedNVFP4 {
        var rng = seed
        func nextByte() -> UInt8 {
            rng = rng &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return UInt8((rng >> 56) & 0xFF)
        }
        let scPool: [UInt8] = [0x30, 0x34, 0x38, 0x3C, 0x40, 0xB4, 0xBC, 0x28]
        let groups = k / 16
        var weights = [UInt32](repeating: 0, count: experts * rows * groups * 2)
        var scales = [UInt8](repeating: 0, count: experts * rows * groups)
        var s2 = [Float](repeating: 1.0, count: experts)
        var ref = [Double](repeating: 0, count: experts * rows * k)
        for e in 0..<experts {
            let es2 = withS2 ? Double(Float(0.25 + 0.125 * Double((e % 4) + 1))) : 1.0
            s2[e] = Float(es2)
            for r in 0..<rows {
                for g in 0..<groups {
                    let scB = scPool[Int(nextByte()) % scPool.count]
                    scales[(e * rows + r) * groups + g] = scB
                    let scV = fp8E4M3FN(scB)
                    var w0: UInt32 = 0, w1: UInt32 = 0
                    for j in 0..<16 {
                        let nib = Int(nextByte() & 0xF)
                        if j < 8 { w0 |= UInt32(nib) << (j * 4) } else { w1 |= UInt32(nib) << ((j - 8) * 4) }
                        ref[((e * rows + r) * k) + g * 16 + j] = fp4Table[nib] * scV * es2
                    }
                    let base = ((e * rows + r) * groups + g) * 2
                    weights[base] = w0
                    weights[base + 1] = w1
                }
            }
        }
        return PackedNVFP4(weights: weights, scales: scales, s2: s2, ref: ref)
    }

    private struct PackedMXFP8 {
        let weights: [UInt32]
        let scales: [UInt8]
        let ref: [Double]
    }

    private func packMXFP8(experts: Int, rows: Int, k: Int, seed: UInt64) -> PackedMXFP8 {
        var rng = seed
        func nextByte() -> UInt8 {
            rng = rng &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return UInt8((rng >> 56) & 0xFF)
        }
        let wPool: [UInt8] = [0x00, 0x04, 0x20, 0x30, 0x38, 0x3C, 0x40, 0x44, 0xB8, 0xBC, 0xC0, 0x84]
        let e8Pool: [UInt8] = [125, 126, 127, 128, 129]
        let groups = k / 32
        var weights = [UInt32](repeating: 0, count: experts * rows * groups * 8)
        var scales = [UInt8](repeating: 0, count: experts * rows * groups)
        var ref = [Double](repeating: 0, count: experts * rows * k)
        for e in 0..<experts {
            for r in 0..<rows {
                for g in 0..<groups {
                    let sb = e8Pool[Int(nextByte()) % e8Pool.count]
                    scales[(e * rows + r) * groups + g] = sb
                    let sv = ue8m0(sb)
                    let base = ((e * rows + r) * groups + g) * 8
                    for wi in 0..<8 {
                        var word: UInt32 = 0
                        for bi in 0..<4 {
                            let wb = wPool[Int(nextByte()) % wPool.count]
                            word |= UInt32(wb) << (bi * 8)
                            ref[((e * rows + r) * k) + g * 32 + wi * 4 + bi] = fp8E4M3FN(wb) * sv
                        }
                        weights[base + wi] = word
                    }
                }
            }
        }
        return PackedMXFP8(weights: weights, scales: scales, ref: ref)
    }

    func testFP8E4M3FNAll256AndUE8M0Parity() throws {
        let hal = try makeHal()
        let n = 256, k = 32
        var wWords = [UInt32](repeating: 0, count: n * 8)
        for b in 0..<256 { wWords[b * 8] = UInt32(b) }
        let scales = [UInt8](repeating: 127, count: n)
        let xVals = [Float](repeating: 1.0, count: k)
        let (_, wS) = try makeSlice(hal, values: wWords)
        let (_, scS) = try makeSlice(hal, values: scales)
        let (_, xS) = try makeSlice(hal, values: xVals)
        let (_, expOffS) = try makeSlice(hal, values: [Int32(0), Int32(1)])
        let (_, actExpS) = try makeSlice(hal, values: [Int32(0)])
        let (_, actArgS) = try makeSlice(hal, values: [UInt32](repeating: 0, count: 12))
        let (_, spS) = try makeSlice(hal, values: [Int32(0)])
        let (outBuf, outS) = try sharedSlice(hal, bytes: n * 4)

        let bank = QWGemmFP.MXFP8Bank(weights: wS, scales: scS, experts: 1, rowsPerExpert: n, k: k)
        let plan = QWGemmFP.MoEActiveBucketPlan(experts: 1, intermediate: k, hidden: n)
        try runOnGPU(hal) { enc, rot in
            try QWGemmFP.encodeMoEBuildActiveBuckets(enc, hal: hal, expertOff: expOffS, activeExperts: actExpS, indirectArgs: actArgS, plan: plan)
            try rot(&enc)
            try QWGemmFP.encodeMoECompactMXFP8Down(enc, hal: hal, actIn: xS, down: bank, expertOff: expOffS,
                                                  activeExperts: actExpS, activeArgs: actArgS, sortedPairIdx: spS,
                                                  slotOut: outS, topK: 1, inDtype: .f32, outDtype: .f32, sortedOrder: true)
        }
        let got = outBuf.contents().assumingMemoryBound(to: Float.self)
        for b in 0..<256 {
            let expected = fp8E4M3FN(UInt8(b))
            if expected.isNaN {
                XCTAssertTrue(got[b].isNaN, "FP8 byte 0x\(String(b, radix: 16)) must unpack to NaN")
            } else {
                XCTAssertEqual(Double(got[b]), expected, accuracy: 1e-6, "FP8 byte 0x\(String(b, radix: 16)) mismatch")
            }
        }

        // Verify UE8M0 edge bytes: 0 -> 2^-127, 1 -> 2^-126, 125..129.
        let e8Bytes: [UInt8] = [0, 1, 125, 126, 127, 128, 129, 255]
        var e8Words = [UInt32](repeating: 0, count: e8Bytes.count * 8)
        for r in 0..<e8Bytes.count { e8Words[r * 8] = 0x38 }
        let (e8WBuf, e8WS) = try makeSlice(hal, values: e8Words)
        _ = e8WBuf
        let (_, e8ScS) = try makeSlice(hal, values: e8Bytes)
        let (e8OutBuf, e8OutS) = try sharedSlice(hal, bytes: e8Bytes.count * 4)
        let e8Bank = QWGemmFP.MXFP8Bank(weights: e8WS, scales: e8ScS, experts: 1, rowsPerExpert: e8Bytes.count, k: k)
        let e8Plan = QWGemmFP.MoEActiveBucketPlan(experts: 1, intermediate: k, hidden: e8Bytes.count)
        try runOnGPU(hal) { enc, rot in
            try QWGemmFP.encodeMoEBuildActiveBuckets(enc, hal: hal, expertOff: expOffS, activeExperts: actExpS, indirectArgs: actArgS, plan: e8Plan)
            try rot(&enc)
            try QWGemmFP.encodeMoECompactMXFP8Down(enc, hal: hal, actIn: xS, down: e8Bank, expertOff: expOffS,
                                                  activeExperts: actExpS, activeArgs: actArgS, sortedPairIdx: spS,
                                                  slotOut: e8OutS, topK: 1, inDtype: .f32, outDtype: .f32, sortedOrder: true)
        }
        let e8Got = e8OutBuf.contents().assumingMemoryBound(to: Float.self)
        for (idx, sb) in e8Bytes.enumerated() {
            let want = ue8m0(sb)
            if want.isNaN {
                XCTAssertTrue(e8Got[idx].isNaN, "UE8M0=255 must unpack to NaN")
            } else if sb == 0 {
                // Apple GPU compute shaders flush IEEE FP32 subnormal operands (2^-127) to zero under FTZ.
                let got = Double(e8Got[idx])
                XCTAssertTrue(got == 0.0 || abs(got - want) <= 1e-42, "UE8M0=0 subnormal mismatch: \(got)")
            } else {
                XCTAssertEqual(Double(e8Got[idx]), want, accuracy: max(want * 1e-5, 1e-42), "UE8M0=\(sb) mismatch")
            }
        }
    }

    func testBF16NAXAndGemvParityAcrossLayouts() throws {
        let hal = try makeHal()
        let n = 80, k = 96
        let resF32 = (0..<(70 * n)).map { Float($0 % 17 - 8) * 0.05 }
        let wF32 = (0..<(n * k)).map { roundBF16(sin(Double($0) * 0.13) * 0.4) }
        let wBF16 = wF32.map(bf16Bits)
        let (_, wS) = try makeSlice(hal, values: wBF16)
        let weight = QWGemmFP.BF16Bank(weights: wS, n: n, k: k)

        if hal.caps.isMetal4TensorCapable {
            for m in [20, 70] {
                let xF32 = (0..<(m * k)).map { roundBF16(cos(Double($0) * 0.11) * 0.35) }
                let xBF16 = xF32.map(bf16Bits)
                let (_, xBS) = try makeSlice(hal, values: xBF16)
                let (_, xFS) = try makeSlice(hal, values: xF32)
                let (yBBuf, yBS) = try makeSlice(hal, values: [UInt16](repeating: 0, count: m * n))
                let (yFBuf, yFS) = try makeSlice(hal, values: Array(resF32.prefix(m * n)))
                let cap: Float = m == 20 ? 15.0 : 0.0
                try runOnGPU(hal) { enc, _ in
                    try QWGemmFP.encodeBF16NAX(enc, hal: hal, weight: weight, x: xBS, y: yBS, m: m,
                                               inDtype: .bf16, outDtype: .bf16, addResidual: false, softcap: cap)
                    try QWGemmFP.encodeBF16NAX(enc, hal: hal, weight: weight, x: xFS, y: yFS, m: m,
                                               inDtype: .f32, outDtype: .f32, addResidual: true, softcap: 0)
                }
                let gotB = yBBuf.contents().assumingMemoryBound(to: UInt16.self)
                let gotF = yFBuf.contents().assumingMemoryBound(to: Float.self)
                for row in 0..<m {
                    for col in 0..<n {
                        var sum = 0.0
                        for j in 0..<k { sum += Double(xF32[row * k + j]) * Double(wF32[col * k + j]) }
                        let idx = row * n + col
                        let refB = cap > 0 ? tanh(sum / Double(cap)) * Double(cap) : sum
                        XCTAssertEqual(Double(f32FromBF16(gotB[idx])), Double(roundBF16(refB)), accuracy: 0.06, "NAX bf16 m=\(m) [\(row),\(col)]")
                        XCTAssertEqual(Double(gotF[idx]), sum + Double(resF32[idx]), accuracy: 0.02, "NAX f32+res m=\(m) [\(row),\(col)]")
                    }
                }
            }
        }

        // Row-paired GEMV rows2 and rows4 across M = 1...8 and unaligned dims.
        let unalignedN = 19, unalignedK = 45
        let uwF32 = (0..<(unalignedN * unalignedK)).map { roundBF16(sin(Double($0) * 0.17) * 0.5) }
        let (_, uwS) = try makeSlice(hal, values: uwF32.map(bf16Bits))
        let uWeight = QWGemmFP.BF16Bank(weights: uwS, n: unalignedN, k: unalignedK)
        for rowsSG in [2, 4] {
            for m in [1, 2, 3, 4, 5, 6, 7, 8] {
                let xF16 = (0..<(m * unalignedK)).map { Float16(sin(Double($0) * 0.21) * 0.4) }
                let (_, xS) = try makeSlice(hal, values: xF16.map(\.bitPattern))
                let resBF16 = (0..<(m * unalignedN)).map { bf16Bits(Float($0 % 5) * 0.1) }
                let (yBuf, yS) = try makeSlice(hal, values: resBF16)
                try runOnGPU(hal) { enc, _ in
                    try QWGemmFP.encodeBF16Gemv(enc, hal: hal, weight: uWeight, x: xS, y: yS, m: m,
                                                inDtype: .f16, outDtype: .bf16,
                                                addResidual: true, softcap: 12.0, rowsPerSimdgroup: rowsSG)
                }
                let got = yBuf.contents().assumingMemoryBound(to: UInt16.self)
                for row in 0..<m {
                    for col in 0..<unalignedN {
                        var sum = Double(f32FromBF16(resBF16[row * unalignedN + col]))
                        for j in 0..<unalignedK { sum += Double(xF16[row * unalignedK + j]) * Double(uwF32[col * unalignedK + j]) }
                        let ref = tanh(sum / 12.0) * 12.0
                        XCTAssertEqual(Double(f32FromBF16(got[row * unalignedN + col])), Double(roundBF16(ref)),
                                       accuracy: 0.04, "GEMV rows\(rowsSG) m=\(m) [\(row),\(col)]")
                    }
                }
            }
        }
    }

    func testNVFP4UnpackAndCompactMoEParity() throws {
        let hal = try makeHal()
        let experts = 6, topK = 2, tokens = 5, hidden = 64, inter = 32
        let pairs = tokens * topK
        let gatePack = packNVFP4(experts: experts, rows: inter, k: hidden, seed: 0xA11CE, withS2: true)
        let upPack = packNVFP4(experts: experts, rows: inter, k: hidden, seed: 0xB0B00, withS2: true)
        let downPack = packNVFP4(experts: experts, rows: hidden, k: inter, seed: 0xCAFE, withS2: false)

        let (_, gWS) = try makeSlice(hal, values: gatePack.weights)
        let (_, gScS) = try makeSlice(hal, values: gatePack.scales)
        let (_, gS2S) = try makeSlice(hal, values: gatePack.s2)
        let (_, uWS) = try makeSlice(hal, values: upPack.weights)
        let (_, uScS) = try makeSlice(hal, values: upPack.scales)
        let (_, uS2S) = try makeSlice(hal, values: upPack.s2)
        let (_, dWS) = try makeSlice(hal, values: downPack.weights)
        let (_, dScS) = try makeSlice(hal, values: downPack.scales)

        let gateBank = QWGemmFP.NVFP4Bank(weights: gWS, scales: gScS, scales2: gS2S, experts: experts, rowsPerExpert: inter, k: hidden)
        let upBank = QWGemmFP.NVFP4Bank(weights: uWS, scales: uScS, scales2: uS2S, experts: experts, rowsPerExpert: inter, k: hidden)
        let downBank = QWGemmFP.NVFP4Bank(weights: dWS, scales: dScS, scales2: nil, experts: experts, rowsPerExpert: hidden, k: inter)

        let xF32 = (0..<(tokens * hidden)).map { roundBF16(sin(Double($0) * 0.09) * 0.5) }
        let xBF16 = xF32.map(bf16Bits)
        // Intentionally skip experts 0 and 4 to test active bucket compaction.
        let idVals: [Int32] = [1, 3, 2, 1, 5, 2, 1, 3, 2, 5]
        let wtVals: [Float] = [0.6, 0.4, 0.7, 0.3, 0.5, 0.5, 0.8, 0.2, 0.35, 0.65]

        let arena = try QWArena(hal: hal, weights: 1 << 16, activations: 4 << 20, kv: 0, gdnStates: 0)
        let ar = arena.activationsRegion()
        let xTV = try ar.allocate(QWTensorShape([tokens, hidden], .bf16)); arena.write(xTV, from: xBF16)
        let idsTV = try ar.allocate(QWTensorShape([pairs], .i32)); arena.write(idsTV, from: idVals)
        let wtsTV = try ar.allocate(QWTensorShape([pairs], .f32)); arena.write(wtsTV, from: wtVals)
        let expOffTV = try ar.allocate(QWTensorShape([experts + 1], .i32))
        let sortedPairTV = try ar.allocate(QWTensorShape([pairs], .i32))
        let sortedTokTV = try ar.allocate(QWTensorShape([pairs], .i32))
        let invPermTV = try ar.allocate(QWTensorShape([pairs], .i32))
        let actExpTV = try ar.allocate(QWTensorShape([experts], .i32))
        let actIndTV = try ar.allocate(QWTensorShape([12], .i32))
        let actOutTV = try ar.allocate(QWTensorShape([pairs * inter], .bf16))
        let dnAccTV = try ar.allocate(QWTensorShape([pairs * hidden], .bf16))
        let outTV = try ar.allocate(QWTensorShape([tokens, hidden], .f32))

        for (gateCap, upCap) in [(Float(0), Float(0)), (Float(10), Float(8))] {
            let bucketPlan = QWGemmFP.MoEActiveBucketPlan(experts: experts, intermediate: inter, hidden: hidden)
            try runOnGPU(hal) { enc, rot in
                try QWGemm.encodeMoEGroupPairsF32(enc, hal: hal, ids: idsTV, expertOff: expOffTV,
                                                  sortedPairIdx: sortedPairTV, sortedTok: sortedTokTV,
                                                  experts: experts, totalPairs: pairs, routes: topK, invPerm: invPermTV)
                enc.memoryBarrier(scope: .buffers)
                try QWGemmFP.encodeMoEBuildActiveBuckets(
                    enc, hal: hal, expertOff: QWFPBufferSlice(expOffTV),
                    activeExperts: QWFPBufferSlice(actExpTV), indirectArgs: QWFPBufferSlice(actIndTV), plan: bucketPlan)
                try rot(&enc)
                try QWGemmFP.encodeMoECompactNVFP4GateUpSwiGLU(
                    enc, hal: hal, x: QWFPBufferSlice(xTV), gate: gateBank, up: upBank,
                    expertOff: QWFPBufferSlice(expOffTV), activeExperts: QWFPBufferSlice(actExpTV),
                    activeArgs: QWFPBufferSlice(actIndTV), sortedTok: QWFPBufferSlice(sortedTokTV),
                    sortedPairIdx: QWFPBufferSlice(sortedPairTV), actOut: QWFPBufferSlice(actOutTV),
                    intermediate: inter, topK: topK, inDtype: .bf16, outDtype: .bf16,
                    gateCap: gateCap, upCap: upCap, sortedOrder: true)
                enc.memoryBarrier(scope: .buffers)
                try QWGemmFP.encodeMoECompactNVFP4Down(
                    enc, hal: hal, actIn: QWFPBufferSlice(actOutTV), down: downBank,
                    expertOff: QWFPBufferSlice(expOffTV), activeExperts: QWFPBufferSlice(actExpTV),
                    activeArgs: QWFPBufferSlice(actIndTV), sortedPairIdx: QWFPBufferSlice(sortedPairTV),
                    slotOut: QWFPBufferSlice(dnAccTV), topK: topK, inDtype: .bf16, outDtype: .bf16, sortedOrder: true)
                enc.memoryBarrier(scope: .buffers)
                try QWGemmFP.encodeMoEDownReduceCombineBF16(
                    enc, hal: hal, dnAcc: QWFPBufferSlice(dnAccTV), invPerm: QWFPBufferSlice(invPermTV),
                    weights: QWFPBufferSlice(wtsTV), shared: nil, gate: nil, out: QWFPBufferSlice(outTV),
                    hidden: hidden, topK: topK, tokens: tokens, inDtype: .bf16, outDtype: .f32)
            }
            let actCount = arena.read(actIndTV, as: UInt32.self, count: 12)
            XCTAssertEqual(actCount[0], 4, "Only 4 of 6 experts should be active")
            XCTAssertEqual(actCount[4], UInt32(bucketPlan.gateUpGridX))
            XCTAssertEqual(actCount[5], 4)
            XCTAssertEqual(actCount[8], UInt32(bucketPlan.downGridX))
            XCTAssertEqual(arena.read(actExpTV, as: Int32.self, count: 4), [1, 2, 3, 5])

            let got = arena.read(outTV, as: Float.self, count: tokens * hidden)
            for t in 0..<tokens {
                var tokOut = [Double](repeating: 0, count: hidden)
                for r in 0..<topK {
                    let e = Int(idVals[t * topK + r]), w = Double(wtVals[t * topK + r])
                    var act = [Double](repeating: 0, count: inter)
                    for i in 0..<inter {
                        var gSum = 0.0, uSum = 0.0
                        for j in 0..<hidden {
                            let xv = Double(xF32[t * hidden + j])
                            gSum += xv * gatePack.ref[(e * inter + i) * hidden + j]
                            uSum += xv * upPack.ref[(e * inter + i) * hidden + j]
                        }
                        act[i] = Double(roundBF16(softcapSwiGLU(gSum, uSum, gateCap: Double(gateCap), upCap: Double(upCap))))
                    }
                    for h in 0..<hidden {
                        var dn = 0.0
                        for i in 0..<inter { dn += act[i] * downPack.ref[(e * hidden + h) * inter + i] }
                        tokOut[h] += w * Double(roundBF16(dn))
                    }
                }
                for h in 0..<hidden {
                    XCTAssertEqual(Double(got[t * hidden + h]), tokOut[h], accuracy: 0.08, "NVFP4 compact cap=\(gateCap) [\(t),\(h)]")
                }
            }
        }
    }

    func testMoENaxPrefillNVFP4AndMXFP8Parity() throws {
        let hal = try makeHal()
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable, "Metal 4 NAX tensor ops required")
        // Heterogeneous counts hitting empty, <=16, <=32, <=64, 64+<=16 tail compaction.
        let counts = [0, 1, 16, 17, 32, 66]
        let experts = counts.count, topK = 3, hidden = 64, inter = 64
        let totalPairs = counts.reduce(0, +)
        let tokens = totalPairs / topK
        precondition(totalPairs % topK == 0)

        var idVals = [Int32]()
        for (e, c) in counts.enumerated() { for _ in 0..<c { idVals.append(Int32(e)) } }
        let wtVals = (0..<totalPairs).map { _ in Float(1.0 / Double(topK)) }
        let xF16 = (0..<(tokens * hidden)).map { Float16(sin(Double($0) * 0.07) * 0.25) }
        let shF32 = (0..<(tokens * hidden)).map { Float($0 % 11 - 5) * 0.03 }
        let gateVals = [Float](repeating: 0.75, count: tokens)

        let arena = try QWArena(hal: hal, weights: 1 << 16, activations: 8 << 20, kv: 0, gdnStates: 0)
        let ar = arena.activationsRegion()
        let xTV = try ar.allocate(QWTensorShape([tokens, hidden], .f16)); arena.write(xTV, from: xF16.map(\.bitPattern))
        let idsTV = try ar.allocate(QWTensorShape([totalPairs], .i32)); arena.write(idsTV, from: idVals)
        let wtsTV = try ar.allocate(QWTensorShape([totalPairs], .f32)); arena.write(wtsTV, from: wtVals)
        let shTV = try ar.allocate(QWTensorShape([tokens, hidden], .f32)); arena.write(shTV, from: shF32)
        let gateTV = try ar.allocate(QWTensorShape([tokens], .f32)); arena.write(gateTV, from: gateVals)
        let tilePlan = QWGemm.MoETilePlan(experts: experts, pairs: totalPairs, gateUpRows: 2 * inter, downRows: hidden)
        let expOffTV = try ar.allocate(QWTensorShape([experts + 1], .i32))
        let sortedPairTV = try ar.allocate(QWTensorShape([totalPairs], .i32))
        let sortedTokTV = try ar.allocate(QWTensorShape([totalPairs], .i32))
        let invPermTV = try ar.allocate(QWTensorShape([totalPairs], .i32))
        let tileETV = try ar.allocate(QWTensorShape([tilePlan.maxTiles], .i32))
        let tileRTV = try ar.allocate(QWTensorShape([tilePlan.maxTiles], .i32))
        let tileCountTV = try ar.allocate(QWTensorShape([tilePlan.scratchInts], .i32))
        let guTV = try ar.allocate(QWTensorShape([totalPairs, 2 * inter], .f16))
        let actTV = try ar.allocate(QWTensorShape([totalPairs, inter], .f16))
        let dnAccTV = try ar.allocate(QWTensorShape([totalPairs, hidden], .f16))
        let outTV = try ar.allocate(QWTensorShape([tokens, hidden], .f32))

        // 1) NVFP4_G16 concatenated gateUp + SwiGLU softcap epilogue + down NAX.
        let nvGU = packNVFP4(experts: experts, rows: 2 * inter, k: hidden, seed: 0x1111, withS2: true)
        let nvDn = packNVFP4(experts: experts, rows: hidden, k: inter, seed: 0x2222, withS2: true)
        let (_, nvgW) = try makeSlice(hal, values: nvGU.weights)
        let (_, nvgSc) = try makeSlice(hal, values: nvGU.scales)
        let (_, nvgS2) = try makeSlice(hal, values: nvGU.s2)
        let (_, nvdW) = try makeSlice(hal, values: nvDn.weights)
        let (_, nvdSc) = try makeSlice(hal, values: nvDn.scales)
        let (_, nvdS2) = try makeSlice(hal, values: nvDn.s2)
        let guBankNV = QWMoEExpertBank.nvfp4(QWGemmFP.NVFP4Bank(
            weights: nvgW, scales: nvgSc, scales2: nvgS2, experts: experts, rowsPerExpert: 2 * inter, k: hidden))
        let dnBankNV = QWMoEExpertBank.nvfp4(QWGemmFP.NVFP4Bank(
            weights: nvdW, scales: nvdSc, scales2: nvdS2, experts: experts, rowsPerExpert: hidden, k: inter))

        try runOnGPU(hal) { enc, rot in
            try QWGemm.encodeMoEBankRoutedCombine(
                &enc, hal: hal, x: QWFPBufferSlice(xTV), xDtype: .f16, interDtype: .f16,
                gateUp: guBankNV, up: nil, down: dnBankNV, idx: idsTV, wts: wtsTV,
                shared: QWFPBufferSlice(shTV), sharedDtype: .f32, gate: QWFPBufferSlice(gateTV),
                out: QWFPBufferSlice(outTV), outDtype: .f32,
                expOff: expOffTV, sortedPair: sortedPairTV, sortedTok: sortedTokTV, invPerm: invPermTV,
                tileE: tileETV, tileR: tileRTV, tileCount: tileCountTV,
                activeExperts: nil, activeIndirect: nil,
                guBuf: guTV, actBuf: actTV, dnAccBuf: dnAccTV,
                hidden: hidden, experts: experts, topK: topK, intermediate: inter, rows: tokens,
                gateSoftcap: 8.0, upSoftcap: 6.0, supportsGroupedNaxPrefill: true,
                concurrentBarriers: true, rotateEncoder: rot)
        }
        let gotNV = arena.read(outTV, as: Float.self, count: tokens * hidden)
        for t in 0..<tokens {
            var refTok = (0..<hidden).map { 0.75 * Double(shF32[t * hidden + $0]) }
            var magTok = (0..<hidden).map { abs(0.75 * Double(shF32[t * hidden + $0])) }
            for r in 0..<topK {
                let e = Int(idVals[t * topK + r]), w = Double(wtVals[t * topK + r])
                var act = [Double](repeating: 0, count: inter)
                for i in 0..<inter {
                    var gs = 0.0, us = 0.0
                    for j in 0..<hidden {
                        let xv = Double(xF16[t * hidden + j])
                        gs += xv * nvGU.ref[(e * 2 * inter + i) * hidden + j]
                        us += xv * nvGU.ref[(e * 2 * inter + inter + i) * hidden + j]
                    }
                    act[i] = Double(Float16(softcapSwiGLU(gs, us, gateCap: 8.0, upCap: 6.0)))
                }
                for h in 0..<hidden {
                    var dn = 0.0
                    for i in 0..<inter { dn += act[i] * nvDn.ref[(e * hidden + h) * inter + i] }
                    let term = w * Double(Float16(dn))
                    refTok[h] += term
                    magTok[h] += abs(term)
                }
            }
            for h in 0..<hidden {
                XCTAssertEqual(Double(gotNV[t * hidden + h]), refTok[h], accuracy: max(0.05, magTok[h] * 8e-3), "NVFP4 NAX [\(t),\(h)]")
            }
        }

        // 2) MXFP8_G32 grouped NAX prefill with standard SwiGLU (gateCap == 0).
        let mxGU = packMXFP8(experts: experts, rows: 2 * inter, k: hidden, seed: 0x3333)
        let mxDn = packMXFP8(experts: experts, rows: hidden, k: inter, seed: 0x4444)
        let (_, mxgW) = try makeSlice(hal, values: mxGU.weights)
        let (_, mxgSc) = try makeSlice(hal, values: mxGU.scales)
        let (_, mxdW) = try makeSlice(hal, values: mxDn.weights)
        let (_, mxdSc) = try makeSlice(hal, values: mxDn.scales)
        let guBankMX = QWMoEExpertBank.mxfp8(QWGemmFP.MXFP8Bank(
            weights: mxgW, scales: mxgSc, experts: experts, rowsPerExpert: 2 * inter, k: hidden))
        let dnBankMX = QWMoEExpertBank.mxfp8(QWGemmFP.MXFP8Bank(
            weights: mxdW, scales: mxdSc, experts: experts, rowsPerExpert: hidden, k: inter))

        try runOnGPU(hal) { enc, rot in
            try QWGemm.encodeMoEBankRoutedCombine(
                &enc, hal: hal, x: QWFPBufferSlice(xTV), xDtype: .f16, interDtype: .f16,
                gateUp: guBankMX, up: nil, down: dnBankMX, idx: idsTV, wts: wtsTV,
                shared: nil, sharedDtype: .f16, gate: nil,
                out: QWFPBufferSlice(outTV), outDtype: .f32,
                expOff: expOffTV, sortedPair: sortedPairTV, sortedTok: sortedTokTV, invPerm: invPermTV,
                tileE: tileETV, tileR: tileRTV, tileCount: tileCountTV,
                activeExperts: nil, activeIndirect: nil,
                guBuf: guTV, actBuf: actTV, dnAccBuf: dnAccTV,
                hidden: hidden, experts: experts, topK: topK, intermediate: inter, rows: tokens,
                gateSoftcap: 0, upSoftcap: 0, supportsGroupedNaxPrefill: true,
                concurrentBarriers: true, rotateEncoder: rot)
        }
        let gotMX = arena.read(outTV, as: Float.self, count: tokens * hidden)
        for t in 0..<tokens {
            var refTok = [Double](repeating: 0, count: hidden)
            var magTok = [Double](repeating: 0, count: hidden)
            for r in 0..<topK {
                let e = Int(idVals[t * topK + r]), w = Double(wtVals[t * topK + r])
                var act = [Double](repeating: 0, count: inter)
                for i in 0..<inter {
                    var gs = 0.0, us = 0.0
                    for j in 0..<hidden {
                        let xv = Double(xF16[t * hidden + j])
                        gs += xv * Double(roundBF16(mxGU.ref[(e * 2 * inter + i) * hidden + j]))
                        us += xv * Double(roundBF16(mxGU.ref[(e * 2 * inter + inter + i) * hidden + j]))
                    }
                    act[i] = Double(Float16(softcapSwiGLU(gs, us, gateCap: 0, upCap: 0)))
                }
                for h in 0..<hidden {
                    var dn = 0.0
                    for i in 0..<inter { dn += act[i] * Double(roundBF16(mxDn.ref[(e * hidden + h) * inter + i])) }
                    let term = w * Double(Float16(dn))
                    refTok[h] += term
                    magTok[h] += abs(term)
                }
            }
            for h in 0..<hidden {
                XCTAssertEqual(Double(gotMX[t * hidden + h]), refTok[h], accuracy: max(0.08, magTok[h] * 2.5e-2), "MXFP8 NAX [\(t),\(h)]")
            }
        }
    }

    func testActiveBucketsSparseMXFP8AndQWMoEOperatorFPE2E() throws {
        let hal = try makeHal()
        let experts = 32, topK = 2, hidden = 64, inter = 32, sharedInter = 32
        let rW = (0..<(experts * hidden)).map { roundBF16(cos(Double($0) * 0.23) * 0.4) }
        let shGUW = (0..<(2 * sharedInter * hidden)).map { roundBF16(sin(Double($0) * 0.13) * 0.3) }
        let shDnW = (0..<(hidden * sharedInter)).map { roundBF16(cos(Double($0) * 0.19) * 0.3) }
        let (_, rS) = try makeSlice(hal, values: rW.map(bf16Bits))
        let (_, shGUS) = try makeSlice(hal, values: shGUW.map(bf16Bits))
        let (_, shDnS) = try makeSlice(hal, values: shDnW.map(bf16Bits))

        let mxGU = packMXFP8(experts: experts, rows: 2 * inter, k: hidden, seed: 0x5555)
        let mxDn = packMXFP8(experts: experts, rows: hidden, k: inter, seed: 0x6666)
        let (_, guW) = try makeSlice(hal, values: mxGU.weights)
        let (_, guSc) = try makeSlice(hal, values: mxGU.scales)
        let (_, dnW) = try makeSlice(hal, values: mxDn.weights)
        let (_, dnSc) = try makeSlice(hal, values: mxDn.scales)

        let fpWeights = QWMoEOperator.FPWeights(
            router: QWGemmFP.BF16Bank(weights: rS, n: experts, k: hidden),
            gateUp: .mxfp8(QWGemmFP.MXFP8Bank(weights: guW, scales: guSc, experts: experts, rowsPerExpert: 2 * inter, k: hidden)),
            down: .mxfp8(QWGemmFP.MXFP8Bank(weights: dnW, scales: dnSc, experts: experts, rowsPerExpert: hidden, k: inter)),
            sharedGateUp: QWGemmFP.BF16Bank(weights: shGUS, n: 2 * sharedInter, k: hidden),
            sharedDown: QWGemmFP.BF16Bank(weights: shDnS, n: hidden, k: sharedInter),
            hidden: hidden, experts: experts, topK: topK,
            intermediate: inter, sharedIntermediate: sharedInter,
            gateSoftcap: 12.0, upSoftcap: 10.0, normTopK: true)

        for tokens in [3, 12] {
            let pairs = tokens * topK
            let tilePlan = QWGemm.MoETilePlan(experts: experts, pairs: pairs, gateUpRows: 2 * inter, downRows: hidden)
            let arena = try QWArena(hal: hal, weights: 1 << 16, activations: 4 << 20, kv: 0, gdnStates: 0)
            let ar = arena.activationsRegion()
            let scratch = QWMoEOperator.FPScratch(
                logits: try ar.allocate(QWTensorShape([tokens, experts], .f32)),
                idx: try ar.allocate(QWTensorShape([tokens, topK], .i32)),
                wts: try ar.allocate(QWTensorShape([tokens, topK], .f32)),
                expOff: try ar.allocate(QWTensorShape([experts + 1], .i32)),
                sortedPair: try ar.allocate(QWTensorShape([pairs], .i32)),
                sortedTok: try ar.allocate(QWTensorShape([pairs], .i32)),
                invPerm: try ar.allocate(QWTensorShape([pairs], .i32)),
                tileE: try ar.allocate(QWTensorShape([tilePlan.maxTiles], .i32)),
                tileR: try ar.allocate(QWTensorShape([tilePlan.maxTiles], .i32)),
                tileCount: try ar.allocate(QWTensorShape([tilePlan.scratchInts], .i32)),
                activeExperts: try ar.allocate(QWTensorShape([experts], .i32)),
                activeIndirect: try ar.allocate(QWTensorShape([12], .i32)),
                gu: try ar.allocate(QWTensorShape([pairs, 2 * inter], .bf16)),
                act: try ar.allocate(QWTensorShape([pairs, inter], .bf16)),
                dnAcc: try ar.allocate(QWTensorShape([pairs, hidden], .bf16)),
                guSh: try ar.allocate(QWTensorShape([tokens, 2 * sharedInter], .bf16)),
                actSh: try ar.allocate(QWTensorShape([tokens, sharedInter], .bf16)),
                shared: try ar.allocate(QWTensorShape([tokens, hidden], .bf16)),
                gate: nil, interDtype: .bf16)

            let xF32 = (0..<(tokens * hidden)).map { roundBF16(sin(Double($0) * 0.14) * 0.35) }
            let (_, xS) = try makeSlice(hal, values: xF32.map(bf16Bits))
            let (outBuf, outS) = try sharedSlice(hal, bytes: tokens * hidden * 2)
            try runOnGPU(hal) { enc, rot in
                try QWMoEOperator.encodeFPResident(
                    &enc, hal: hal, weights: fpWeights, scratch: scratch,
                    input: xS, inDtype: .bf16, out: outS, outDtype: .bf16, rows: tokens,
                    supportsGroupedNaxPrefill: hal.caps.supportsSimdgroupMMA,
                    concurrentBarriers: true, rotateEncoder: rot)
            }
            let gotIdx = arena.read(scratch.idx, as: Int32.self, count: pairs)
            let gotWts = arena.read(scratch.wts, as: Float.self, count: pairs)
            let gotOut = outBuf.contents().assumingMemoryBound(to: UInt16.self)

            for t in 0..<tokens {
                var tokRef = [Double](repeating: 0, count: hidden)
                var magRef = [Double](repeating: 0, count: hidden)
                var l1Inner = [Double](repeating: 0, count: hidden)
                var shAct = [Double](repeating: 0, count: sharedInter)
                for i in 0..<sharedInter {
                    var gs = 0.0, us = 0.0
                    for j in 0..<hidden {
                        let xv = Double(xF32[t * hidden + j])
                        gs += xv * Double(shGUW[i * hidden + j])
                        us += xv * Double(shGUW[(sharedInter + i) * hidden + j])
                    }
                    let gf = Double(roundBF16(gs)), uf = Double(roundBF16(us))
                    shAct[i] = Double(roundBF16(softcapSwiGLU(gf, uf, gateCap: 12.0, upCap: 10.0)))
                }
                for h in 0..<hidden {
                    var sSum = 0.0, sL1 = 0.0
                    for i in 0..<sharedInter {
                        let p = shAct[i] * Double(shDnW[h * sharedInter + i])
                        sSum += p; sL1 += abs(p)
                    }
                    let shB = Double(roundBF16(sSum))
                    tokRef[h] = shB
                    magRef[h] = abs(shB)
                    l1Inner[h] = sL1
                }
                for r in 0..<topK {
                    let e = Int(gotIdx[t * topK + r]), w = Double(gotWts[t * topK + r])
                    var act = [Double](repeating: 0, count: inter)
                    for i in 0..<inter {
                        var gs = 0.0, us = 0.0
                        for j in 0..<hidden {
                            let xv = Double(xF32[t * hidden + j])
                            gs += xv * Double(roundBF16(mxGU.ref[(e * 2 * inter + i) * hidden + j]))
                            us += xv * Double(roundBF16(mxGU.ref[(e * 2 * inter + inter + i) * hidden + j]))
                        }
                        let gf = Double(roundBF16(gs)), uf = Double(roundBF16(us))
                        act[i] = Double(roundBF16(softcapSwiGLU(gf, uf, gateCap: 12.0, upCap: 10.0)))
                    }
                    for h in 0..<hidden {
                        var dn = 0.0, dL1 = 0.0
                        for i in 0..<inter {
                            let p = act[i] * Double(roundBF16(mxDn.ref[(e * hidden + h) * inter + i]))
                            dn += p; dL1 += abs(p)
                        }
                        let term = w * Double(roundBF16(dn))
                        tokRef[h] += term
                        magRef[h] += abs(term)
                        l1Inner[h] += w * dL1
                    }
                }
                for h in 0..<hidden {
                    let want = Double(roundBF16(tokRef[h]))
                    let tol = max(0.25, max(magRef[h] * 3e-2, l1Inner[h] * 1.5e-2))
                    XCTAssertEqual(Double(f32FromBF16(gotOut[t * hidden + h])), want,
                                   accuracy: tol,
                                   "QWMoEOperator.encodeFPResident T=\(tokens) [\(t),\(h)]")
                }
            }
        }
    }

    func testFusedNVFP4SplitGateUpScales2ParityAcrossPrefillAndDecode() throws {
        let hal = try makeHal()
        let experts = 8, topK = 2, hidden = 64, inter = 32, tokens = 10, totalPairs = tokens * topK
        let arena = try QWArena(hal: hal, weights: 1 << 16, activations: 8 << 20, kv: 0, gdnStates: 0)
        let ar = arena.activationsRegion()
        let gPack = packNVFP4(experts: experts, rows: inter, k: hidden, seed: 0x7711, withS2: true)
        let uPack = packNVFP4(experts: experts, rows: inter, k: hidden, seed: 0x8822, withS2: true)
        let dPack = packNVFP4(experts: experts, rows: hidden, k: inter, seed: 0x9933, withS2: true)
        var guW: [UInt32] = [], guSc: [UInt8] = []
        let wExp = inter * (hidden / 8), sExp = inter * (hidden / 16)
        for e in 0..<experts {
            guW.append(contentsOf: gPack.weights[(e * wExp)..<((e + 1) * wExp)])
            guW.append(contentsOf: uPack.weights[(e * wExp)..<((e + 1) * wExp)])
            guSc.append(contentsOf: gPack.scales[(e * sExp)..<((e + 1) * sExp)])
            guSc.append(contentsOf: uPack.scales[(e * sExp)..<((e + 1) * sExp)])
        }
        let uRefScaled = uPack.ref.map { $0 * 1.7 }
        let guBank = QWMoEExpertBank.nvfp4(QWGemmFP.NVFP4Bank(
            weights: try makeSlice(hal, values: guW).1, scales: try makeSlice(hal, values: guSc).1,
            scales2: try makeSlice(hal, values: gPack.s2 + uPack.s2.map { $0 * 1.7 }).1,
            experts: experts, rowsPerExpert: 2 * inter, k: hidden, splitGateUpScales2: true))
        let dnBank = QWMoEExpertBank.nvfp4(QWGemmFP.NVFP4Bank(
            weights: try makeSlice(hal, values: dPack.weights).1, scales: try makeSlice(hal, values: dPack.scales).1,
            scales2: try makeSlice(hal, values: dPack.s2).1, experts: experts, rowsPerExpert: hidden, k: inter))

        let xF16 = (0..<(tokens * hidden)).map { Float16(sin(Double($0) * 0.21) * 0.4) }
        let idVals = (0..<totalPairs).map { Int32(($0 * 3 + $0 / topK) % experts) }
        let wtVals = (0..<totalPairs).map { Float($0 % 2 == 0 ? 0.65 : 0.35) }
        let xTV = try ar.allocate(QWTensorShape([tokens, hidden], .f16)); arena.write(xTV, from: xF16.map(\.bitPattern))
        let idsTV = try ar.allocate(QWTensorShape([totalPairs], .i32)); arena.write(idsTV, from: idVals)
        let wtsTV = try ar.allocate(QWTensorShape([totalPairs], .f32)); arena.write(wtsTV, from: wtVals)
        let expOffTV = try ar.allocate(QWTensorShape([experts + 1], .i32))
        let sortedPairTV = try ar.allocate(QWTensorShape([totalPairs], .i32))
        let sortedTokTV = try ar.allocate(QWTensorShape([totalPairs], .i32))
        let invPermTV = try ar.allocate(QWTensorShape([totalPairs], .i32))
        let tilePlan = QWGemm.MoETilePlan(experts: experts, pairs: totalPairs, gateUpRows: 2 * inter, downRows: hidden)
        let tileETV = try ar.allocate(QWTensorShape([max(1, tilePlan.maxTiles)], .i32))
        let tileRTV = try ar.allocate(QWTensorShape([max(1, tilePlan.maxTiles)], .i32))
        let tileCountTV = try ar.allocate(QWTensorShape([tilePlan.scratchInts], .i32))
        let actExpertsTV = try ar.allocate(QWTensorShape([experts], .i32))
        let actArgsTV = try ar.allocate(QWTensorShape([QWGemmFP.MoEActiveBucketPlan.indirectBufferBytes / 4], .i32))
        let guTV = try ar.allocate(QWTensorShape([totalPairs, 2 * inter], .f16))
        let actTV = try ar.allocate(QWTensorShape([totalPairs, inter], .f16))
        let dnAccTV = try ar.allocate(QWTensorShape([totalPairs, hidden], .f16))
        let outTV = try ar.allocate(QWTensorShape([tokens, hidden], .f32))

        for curT in hal.caps.supportsSimdgroupMMA ? [1, 4, tokens] : [1, 4] {
            try runOnGPU(hal) { enc, rot in
                try QWGemm.encodeMoEBankRoutedCombine(
                    &enc, hal: hal, x: QWFPBufferSlice(xTV), xDtype: .f16, interDtype: .f16,
                    gateUp: guBank, up: nil, down: dnBank, idx: idsTV, wts: wtsTV,
                    shared: nil, sharedDtype: .f16, gate: nil, out: QWFPBufferSlice(outTV), outDtype: .f32,
                    expOff: expOffTV, sortedPair: sortedPairTV, sortedTok: sortedTokTV, invPerm: invPermTV,
                    tileE: curT > 8 ? tileETV : nil, tileR: curT > 8 ? tileRTV : nil,
                    tileCount: curT > 8 ? tileCountTV : nil,
                    activeExperts: curT == 1 ? nil : actExpertsTV, activeIndirect: curT == 1 ? nil : actArgsTV,
                    guBuf: guTV, actBuf: actTV, dnAccBuf: dnAccTV,
                    hidden: hidden, experts: experts, topK: topK, intermediate: inter, rows: curT,
                    gateSoftcap: 8.0, upSoftcap: 6.0, supportsGroupedNaxPrefill: true,
                    concurrentBarriers: true, rotateEncoder: rot)
            }
            let got = arena.read(outTV, as: Float.self, count: curT * hidden)
            for t in 0..<curT {
                var ref = [Double](repeating: 0, count: hidden), mag = [Double](repeating: 0, count: hidden)
                for r in 0..<topK {
                    let e = Int(idVals[t * topK + r]), w = Double(wtVals[t * topK + r])
                    let act = (0..<inter).map { i -> Double in
                        let gs = (0..<hidden).reduce(0.0) { $0 + Double(xF16[t * hidden + $1]) * gPack.ref[(e * inter + i) * hidden + $1] }
                        let us = (0..<hidden).reduce(0.0) { $0 + Double(xF16[t * hidden + $1]) * uRefScaled[(e * inter + i) * hidden + $1] }
                        return Double(Float16(softcapSwiGLU(gs, us, gateCap: 8.0, upCap: 6.0)))
                    }
                    for h in 0..<hidden {
                        let term = w * Double(Float16((0..<inter).reduce(0.0) { $0 + act[$1] * dPack.ref[(e * hidden + h) * inter + $1] }))
                        ref[h] += term; mag[h] += abs(term)
                    }
                }
                for h in 0..<hidden {
                    XCTAssertEqual(Double(got[t * hidden + h]), ref[h], accuracy: max(0.05, mag[h] * 8e-3),
                                   "splitS2 curT=\(curT) [\(t),\(h)]")
                }
            }
        }
    }
}
