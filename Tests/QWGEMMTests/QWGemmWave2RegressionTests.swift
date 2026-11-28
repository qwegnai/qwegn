// QWGemmWave2RegressionTests.swift — Wave 2 regression gates for Router, SHVQ MoE, MoE NAX, and QWGemmAQ.
import Foundation
import Metal
import XCTest
@testable import QWGEMM
import QWAlloc
import QWCore
import QWHAL
import QWQuant

final class QWGemmWave2RegressionTests: XCTestCase {
    private func makeHal(basis: QWWeightBasis = .rotated) throws -> QWHal {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("no Metal device") }
        return try QWHal(geometry: .qwen38_27b, weightBasis: basis)
    }

    private func makeHQ4Bank(n: Int, k: Int, seedInit: UInt32 = 0x1357_2468) -> Data {
        let groups = k / 32
        var seed = seedInit
        var out = Data(count: n * groups * 20)
        out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let base = (r * groups + g) * 20
                    let scF16 = Float16(0.02 * (1.0 + 0.1 * Float((r + g) % 7)))
                    let sc = scF16.bitPattern
                    let bi = Float16(-1031.5 * Float(scF16)).bitPattern
                    ptr[base] = UInt8(sc & 0xFF)
                    ptr[base + 1] = UInt8(sc >> 8)
                    ptr[base + 2] = UInt8(bi & 0xFF)
                    ptr[base + 3] = UInt8(bi >> 8)
                    for b in 0..<16 {
                        seed = seed &* 1664525 &+ 1013904223
                        ptr[base + 4 + b] = UInt8((seed >> 16) & 0xFF)
                    }
                }
            }
        }
        return out
    }

    private func makeAQBank(bits: Int, group: Int, n: Int, k: Int, seedInit: UInt32 = 17) throws -> Data {
        var seed = seedInit
        var bank = Data()
        let mask = UInt32((1 << bits) - 1)
        for r in 0..<n {
            for g in 0..<(k / group) {
                let vals = (0..<group).map { idx -> UInt32 in
                    seed = seed &* 1664525 &+ 1013904223
                    return ((seed >> 12) ^ UInt32(r * 7 + g * 3 + idx)) & mask
                }
                bank.append(try QWAQBlock.pack(bits: bits, group: group, values: vals, scale: 0.03125, bias: -0.25))
            }
        }
        return bank
    }

    private func makeSHVQBank(bits: Int, group: Int, n: Int, k: Int, seedInit: UInt32 = 0x9E37_79B9) -> Data {
        let groups = k / group
        let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
        var seed = seedInit
        var bank = Data(count: n * groups * gBytes)
        bank.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let base = (r * groups + g) * gBytes
                    let sc = Float16(0.05 * (1.0 + 0.1 * Float((r + g) % 5))).bitPattern
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

    private func unpackSHVQRMat(bits: Int, group: Int, rows: Int, cols: Int, bank: Data) -> [Float] {
        let groups = cols / group
        let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
        var out = [Float]()
        out.reserveCapacity(rows * cols)
        var tmp = [Float]()
        for r in 0..<rows {
            for g in 0..<groups {
                let off = (r * groups + g) * gBytes
                let slice = bank.subdata(in: off..<(off + gBytes))
                slice.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                    QWSHVQBlock.unpack(group: raw, bits: bits, group: group, into: &tmp)
                }
                out.append(contentsOf: tmp)
            }
        }
        return out
    }

    func testRouterF16MatchesCPURefAndNaNRouteSafe() throws {
        let hal = try makeHal()
        let n = 64, k = 64
        for rows in [1, 4, 16, 128] {
            let arena = try QWArena(hal: hal, weights: n * k * 4 + 4096, activations: rows * (n + k) * 8 + 4096, kv: 0, gdnStates: 0)
            let wF16Vals = (0..<(n * k)).map { Float16(Float(($0 * 13) % 29 - 14) * 0.0625) }
            let xVals = (0..<(rows * k)).map { Float(($0 * 7) % 19 - 9) * 0.125 }
            let wT = try arena.weightsRegion().allocate(QWTensorShape([n, k], .f16))
            arena.write(wT, from: wF16Vals)
            let xT = try arena.activationsRegion().allocate(QWTensorShape([rows, k], .f32))
            arena.write(xT, from: xVals)
            let yT = try arena.activationsRegion().allocate(QWTensorShape([rows, n], .f32))

            guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
            try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: wT, x: xT, y: yT, rows: rows, n: n, k: k)
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)

            let got = arena.read(yT, as: Float.self, count: rows * n)
            for r in 0..<rows {
                for c in 0..<n {
                    var ref: Float = 0
                    for i in 0..<k { ref += Float(wF16Vals[c * k + i]) * xVals[r * k + i] }
                    XCTAssertEqual(got[r * n + c], ref, accuracy: 1e-4, "rows=\(rows) (\(r),\(c))")
                }
            }
        }

        // All-NaN router logits with latentLabels + scaleParam + groups must cleanly output -1 / 0.0 without OOB.
        let e = 32, topK = 4, r = 2
        let arena = try QWArena(hal: hal, weights: 0, activations: 64 << 10, kv: 0, gdnStates: 0)
        let act = arena.activationsRegion()
        let logitsT = try act.allocate(QWTensorShape([r * e], .f32))
        arena.write(logitsT, from: [Float](repeating: .nan, count: r * e))
        let idxT = try act.allocate(QWTensorShape([r * topK], .i32))
        let wtsT = try act.allocate(QWTensorShape([r * topK], .f32))
        let labelsT = try act.allocate(QWTensorShape([e], .i32))
        arena.write(labelsT, from: (0..<e).map { Int32($0 % 2) })
        let scaleParamT = try act.allocate(QWTensorShape([e], .f32))
        arena.write(scaleParamT, from: [Float](repeating: 1.0, count: e))
        let groupIdsT = try act.allocate(QWTensorShape([r], .i32))
        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
        try QWGemm.encodeFnMoeRoute(enc, hal: hal, logits: logitsT, idx: idxT, wts: wtsT,
                                    rows: r, experts: e, topK: topK, normTopK: true,
                                    routedScaleParam: scaleParamT, latentLabels: labelsT,
                                    groupIds: groupIdsT, numGroups: 2)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        let ids = arena.read(idxT, as: Int32.self, count: r * topK)
        let wts = arena.read(wtsT, as: Float.self, count: r * topK)
        for v in ids { XCTAssertEqual(v, -1, "all-NaN route id must be -1, got \(v)") }
        for w in wts { XCTAssertEqual(w, 0.0, "all-NaN route weight must be 0, got \(w)") }
    }

    func testRouterNaxColumnBoundsGuardOnNonMultipleOf64() throws {
        let hal = try makeHal()
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable, "qwg_router_nax requires Metal 4")
        let rows = 128, n = 72, paddedN = 128, k = 64
        let arena = try QWArena(hal: hal, weights: n * k * 2 + 4096, activations: (rows * (k + paddedN)) * 4 + 4096, kv: 0, gdnStates: 0)
        let wFloat = (0..<(n * k)).map { Float(($0 * 11) % 23 - 11) * 0.05 }
        let wBF16 = wFloat.map(QWBF16.fromFloatRNE)
        let xVals = (0..<(rows * k)).map { Float(($0 * 5) % 17 - 8) * 0.1 }
        let wT = try arena.weightsRegion().allocate(QWTensorShape([n, k], .bf16))
        arena.write(wT, from: wBF16)
        let xT = try arena.activationsRegion().allocate(QWTensorShape([rows, k], .f32))
        arena.write(xT, from: xVals)
        let yT = try arena.activationsRegion().allocate(QWTensorShape([rows * paddedN], .f32))
        let sentinel: Float = -9999.0
        arena.write(yT, from: [Float](repeating: sentinel, count: rows * paddedN))

        let pso = try hal.pipeline("qwg_router_nax")
        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
        enc.setComputePipelineState(pso)
        enc.bindTensors(wT, xT, yT, startIndex: 0)
        enc.setUInt32s(rows, n, k, startIndex: 3)
        wT.bind(enc, index: 7)
        yT.bind(enc, index: 8)
        enc.setUInt32s(0, startIndex: 9)
        enc.dispatchThreadgroups(MTLSize(width: 2, height: (rows + 63) / 64, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        let got = arena.read(yT, as: Float.self, count: rows * n)
        for row in 0..<rows {
            for col in 0..<n {
                var ref: Float = 0
                for c in 0..<k { ref += QWBF16.toFloat(wBF16[col * k + c]) * xVals[row * k + c] }
                XCTAssertEqual(got[row * n + col], ref, accuracy: 2e-2, "row=\(row) col=\(col)")
            }
        }
        // Tail past rows * n must keep sentinel untouched.
        let tail = arena.read(yT.slice(byteOffset: rows * n * 4, count: rows * (paddedN - n), dtype: .f32),
                              as: Float.self, count: rows * (paddedN - n))
        XCTAssertEqual(tail.first, sentinel)
        XCTAssertEqual(tail.last, sentinel)
    }

    func testSHVQGroupLatentMoEUpperBitsAndPromoteMasking() throws {
        let hal = try makeHal()
        let bits = 4, group = 64, e = 4, t = 2, ng = 2, realE = 2, inter = 64, lat = 64
        let guData = makeSHVQBank(bits: bits, group: group, n: e * 2 * inter, k: lat, seedInit: 0x22)
        let dnData = makeSHVQBank(bits: bits, group: group, n: e * lat, k: inter, seedInit: 0x77)
        let guMat = unpackSHVQRMat(bits: bits, group: group, rows: e * 2 * inter, cols: lat, bank: guData)
        let dnMat = unpackSHVQRMat(bits: bits, group: group, rows: e * lat, cols: inter, bank: dnData)

        let arena = try QWArena(hal: hal, weights: guData.count + dnData.count + 4096,
                                activations: 1 << 20, kv: 0, gdnStates: 0)
        let guBuf = try arena.weightsRegion().allocate(QWTensorShape([guData.count / 2], .f16))
        let dnBuf = try arena.weightsRegion().allocate(QWTensorShape([dnData.count / 2], .f16))
        arena.copyIn(guBuf, from: guData)
        arena.copyIn(dnBuf, from: dnData)
        let guView = try XCTUnwrap(QWWeightView(blocks: guBuf, n: e * 2 * inter, k: lat,
                                                bits: bits, groupSize: group, isRotated: true,
                                                isIncumbentContainer: false, isSHVQ: true))
        let dnView = try XCTUnwrap(QWWeightView(blocks: dnBuf, n: e * lat, k: inter,
                                                bits: bits, groupSize: group, isRotated: true,
                                                isIncumbentContainer: false, isSHVQ: true))

        // Pack (gStride=2 << 23) | (gIdx << 16) | expert.
        let packedIds: [Int32] = [
            Int32((2 << 23) | (0 << 16) | 1), Int32((2 << 23) | (1 << 16) | 2),
            Int32((2 << 23) | (0 << 16) | 3), Int32((2 << 23) | (1 << 16) | 0),
        ]
        let act = arena.activationsRegion()
        let xLat = try act.allocate(QWTensorShape([t * ng * lat], .f32))
        let xVals = (0..<(t * ng * lat)).map { Float(($0 * 9) % 15 - 7) * 0.125 }
        arena.write(xLat, from: xVals)
        let idsT = try act.allocate(QWTensorShape([t * realE], .i32))
        arena.write(idsT, from: packedIds)
        let wtsT = try act.allocate(QWTensorShape([t * realE], .f32))
        arena.write(wtsT, from: [Float](repeating: 0.5, count: t * realE))
        let guOut = try act.allocate(QWTensorShape([t * realE * 2 * inter], .f32))
        let actOut = try act.allocate(QWTensorShape([t * realE * inter], .f32))
        let zLatOut = try act.allocate(QWTensorShape([t * ng * lat], .f32))
        let sharedZeros = try act.allocate(QWTensorShape([t * ng * lat], .f32))
        let gateZeros = try act.allocate(QWTensorShape([t], .f32))

        // Verify qwmoe_promote preserves upper bits in qwg_moe_resident_ids.
        let slotMap = try act.allocate(QWTensorShape([e], .i32))
        arena.write(slotMap, from: [Int32(0), Int32(1), Int32(-1), Int32(2)])
        let resIds = try act.allocate(QWTensorShape([t * realE], .i32))

        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
        enc.setComputePipelineState(try hal.pipeline("qwg_moe_resident_ids"))
        enc.bindTensors(idsT, slotMap, resIds, startIndex: 0)
        enc.setUInt32s(t * realE, e, startIndex: 3)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        try QWGemm.encodeMoEGateUpF32R1(enc, hal: hal, x: xLat, bank: guView, ids: idsT,
                                         out: guOut, rowsPerExpert: 2 * inter,
                                         routes: realE, tokens: t, fuseSiluFwht: false)
        enc.memoryBarrier(scope: .buffers)
        try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guOut, act: actOut,
                                    rows: t * realE, intermediate: inter)
        enc.memoryBarrier(scope: .buffers)
        try QWGemm.encodeMoEDownSharedCombineF32(enc, hal: hal, x: actOut, bank: dnView,
                                                  rowsPerRoute: lat, routes: realE,
                                                  weights: wtsT, shared: sharedZeros,
                                                  gate: gateZeros, out: zLatOut,
                                                  ids: idsT, tokens: t,
                                                  skipShared: true, numGroups: ng,
                                                  tokenIndexedX: false)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let promGot = arena.read(resIds, as: Int32.self, count: 4)
        XCTAssertEqual(promGot, [
            Int32((2 << 23) | (0 << 16) | 1),
            Int32(-1),
            Int32(bitPattern: UInt32((2 << 23) | (0 << 16) | 2)),
            Int32((2 << 23) | (1 << 16) | 0),
        ])

        let zGot = arena.read(zLatOut, as: Float.self, count: t * ng * lat)
        for tok in 0..<t {
            for grp in 0..<ng {
                var want = [Float](repeating: 0, count: lat)
                for r in 0..<realE {
                    let raw = UInt32(bitPattern: packedIds[tok * realE + r])
                    if ((raw >> 16) & 0x7F) != grp { continue }
                    let expId = Int(raw & 0xFFFF)
                    let xOff = (tok * ng + grp) * lat
                    var gateA = [Float](repeating: 0, count: inter)
                    for i in 0..<inter {
                        var g: Float = 0, u: Float = 0
                        for col in 0..<lat {
                            g += guMat[(expId * 2 * inter + i) * lat + col] * xVals[xOff + col]
                            u += guMat[(expId * 2 * inter + inter + i) * lat + col] * xVals[xOff + col]
                        }
                        gateA[i] = (g / (1.0 + exp(-g))) * u
                    }
                    gateA.withUnsafeMutableBufferPointer { ptr in
                        guard let base = ptr.baseAddress else { return }
                        for b in stride(from: 0, to: inter, by: 32) {
                            QWQuant.fwht32OrthoBlock(base + b)
                        }
                    }
                    for row in 0..<lat {
                        var sum: Float = 0
                        for i in 0..<inter { sum += dnMat[(expId * lat + row) * inter + i] * gateA[i] }
                        want[row] += 0.5 * sum
                    }
                }
                for row in 0..<lat {
                    XCTAssertEqual(zGot[(tok * ng + grp) * lat + row], want[row], accuracy: 1e-4, "tok=\(tok) grp=\(grp) row=\(row)")
                }
            }
        }
    }

    func testMoEOperatorTightNaxScratchAndNon64SharedDownRunsSafely() throws {
        let hal = try makeHal()
        let h = 64, e = 32, topK = 4, inter = 64, sharedInter = 32, rows = 64
        let guData = makeHQ4Bank(n: e * 2 * inter, k: h)
        let dnData = makeHQ4Bank(n: e * h, k: inter)
        let shGuData = makeHQ4Bank(n: 2 * sharedInter, k: h)
        let shDnData = makeHQ4Bank(n: h, k: sharedInter)
        let routerData = (0..<(e * h)).map { QWBF16.fromFloatRNE(Float(($0 * 7) % 13 - 6) * 0.05) }
        let gateWData = (0..<h).map { Float(($0 * 5) % 9 - 4) * 0.05 }

        let totalW = guData.count + dnData.count + shGuData.count + shDnData.count + routerData.count * 2 + gateWData.count * 4
        let arena = try QWArena(hal: hal, weights: totalW + (1 << 20), activations: 32 << 20, kv: 0, gdnStates: 0)
        func allocView(_ d: Data, _ n: Int, _ k: Int) throws -> QWWeightView {
            let t = try arena.weightsRegion().allocate(QWTensorShape([d.count / 2], .f16))
            arena.copyIn(t, from: d)
            return try XCTUnwrap(QWWeightView(blocks: t, n: n, k: k, isRotated: true))
        }
        let routerT = try arena.weightsRegion().allocate(QWTensorShape([e, h], .bf16))
        arena.write(routerT, from: routerData)
        let shGateScalar = try arena.weightsRegion().allocate(QWTensorShape([h], .f32))
        arena.write(shGateScalar, from: gateWData)

        let weights = QWMoEOperator.Weights(
            router: routerT,
            gateUp: try allocView(guData, e * 2 * inter, h),
            down: try allocView(dnData, e * h, inter),
            sharedGateUp: try allocView(shGuData, 2 * sharedInter, h),
            sharedDown: try allocView(shDnData, h, sharedInter),
            sharedGateScalar: shGateScalar,
            hidden: h, experts: e, topK: topK,
            intermediate: inter, sharedIntermediate: sharedInter
        )

        let act = arena.activationsRegion()
        let xPlain = try act.allocate(QWTensorShape([rows * h], .f32))
        let xRot = try act.allocate(QWTensorShape([rows * h], .f32))
        let xVals = (0..<(rows * h)).map { Float(($0 * 11) % 21 - 10) * 0.05 }
        arena.write(xPlain, from: xVals)
        arena.write(xRot, from: xVals)
        let pairs = rows * topK
        let plan = QWGemm.MoETilePlan(experts: e, pairs: pairs, gateUpRows: 2 * inter, downRows: h)
        // Intentionally size naxX16 to T * H (strictly smaller than T * (H + sharedInter))
        // so shAct16 tail aliasing MUST be refused when prefillPrecast16 is nil.
        let naxX16 = try act.allocate(QWTensorShape([rows * h], .f16))
        let scratch = QWMoEOperator.Scratch(
            xRot32: try act.allocate(QWTensorShape([rows * h], .f32)),
            xPlain32: try act.allocate(QWTensorShape([rows * h], .f32)),
            logits: try act.allocate(QWTensorShape([rows * e], .f32)),
            idx: try act.allocate(QWTensorShape([pairs], .i32)),
            wts: try act.allocate(QWTensorShape([pairs], .f32)),
            gu32: try act.allocate(QWTensorShape([pairs * 2 * inter], .f32)),
            act32: try act.allocate(QWTensorShape([pairs * inter], .f32)),
            guSh32: try act.allocate(QWTensorShape([rows * 2 * sharedInter], .f32)),
            actSh32: try act.allocate(QWTensorShape([rows * sharedInter], .f32)),
            shared: try act.allocate(QWTensorShape([rows * h], .f32)),
            gate: try act.allocate(QWTensorShape([rows], .f32)),
            expOff: try act.allocate(QWTensorShape([e + 1], .i32)),
            sortedPair: try act.allocate(QWTensorShape([pairs], .i32)),
            sortedTok: try act.allocate(QWTensorShape([pairs], .i32)),
            invPerm: try act.allocate(QWTensorShape([pairs], .i32)),
            tileE: try act.allocate(QWTensorShape([pairs], .i32)),
            tileR: try act.allocate(QWTensorShape([pairs], .i32)),
            tileCount: try act.allocate(QWTensorShape([plan.scratchInts], .i32)),
            gu16: try act.allocate(QWTensorShape([pairs * 2 * inter], .f16)),
            dnHi: try act.allocate(QWTensorShape([pairs * inter], .f16)),
            dnAcc: try act.allocate(QWTensorShape([pairs * h], .f32)),
            naxX16: naxX16,
            naxY16: try act.allocate(QWTensorShape([rows * h], .f16)),
            maxTileWidth: 64
        )
        let out32 = try act.allocate(QWTensorShape([rows * h], .f32))
        let out16 = try act.allocate(QWTensorShape([rows * h], .f16))
        guard let (cb, rawEnc) = hal.begin(.concurrent) else { throw XCTSkip("no command buffer") }
        var enc: any MTLComputeCommandEncoder = rawEnc
        var encEnded = false
        defer { if !encEnded { enc.endEncoding() } }
        try QWMoEOperator.encodeResident(
            &enc, hal: hal, weights: weights, scratch: scratch,
            input: .f32PreRotated(plain: xPlain, rotated: xRot, prefillPrecast16: nil),
            out32: out32, out16: out16, rows: rows,
            supportsGroupedNaxPrefill: hal.caps.supportsSimdgroupMMA,
            concurrentBarriers: true,
            rotateEncoder: { current in
                current.endEncoding()
                guard let next = cb.makeComputeCommandEncoder(dispatchType: .concurrent) else {
                    throw XCTSkip("failed to rotate compute encoder")
                }
                current = next
            }
        )
        hal.end(cb, enc)
        encEnded = true
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        for v in arena.read(out32, as: Float.self, count: rows * h) { XCTAssertTrue(v.isFinite) }

        // Router shared-SwiGLU guard: IS=12 (% 8 != 0) must still leave valid route IDs.
        let rIdx = try act.allocate(QWTensorShape([4 * topK], .i32))
        let rWts = try act.allocate(QWTensorShape([4 * topK], .f32))
        let guNon8 = try act.allocate(QWTensorShape([4 * 24], .f32))
        let actNon8 = try act.allocate(QWTensorShape([4 * 12], .f32))
        guard let (cb2, enc2) = hal.begin() else { throw XCTSkip("no command buffer") }
        try QWGemm.encodeFnMoeRoute(enc2, hal: hal, logits: scratch.logits, idx: rIdx, wts: rWts,
                                    rows: 4, experts: e, topK: topK, normTopK: true,
                                    guSh: guNon8, actSh: actNon8, sharedIntermediate: 12)
        hal.end(cb2, enc2)
        cb2.waitUntilCompleted()
        XCTAssertNil(cb2.error)
        for id in arena.read(rIdx, as: Int32.self, count: 4 * topK) {
            XCTAssertTrue((0..<Int32(e)).contains(id))
        }
    }

    func testGemmSiluPairRejectsNonIncumbentAndAQHandlesLargeRowsAndGuards() throws {
        let hal = try makeHal()
        let n = 64, k = 64
        let aqBank = try makeAQBank(bits: 4, group: 32, n: n, k: k)
        let arena = try QWArena(hal: hal, weights: aqBank.count + 4096, activations: 16 * (n + k) * 4 + 4096, kv: 0, gdnStates: 0)
        let wBuf = try arena.weightsRegion().allocate(QWTensorShape([aqBank.count / 2], .f16))
        arena.copyIn(wBuf, from: aqBank)
        let nonInc = try XCTUnwrap(QWWeightView(blocks: wBuf, n: n, k: k, bits: 4, groupSize: 32,
                                                isRotated: false, isIncumbentContainer: false))

        let act = arena.activationsRegion()
        let x16 = try act.allocate(QWTensorShape([10 * k], .f16))
        arena.write(x16, from: (0..<(10 * k)).map { Float16(Float(($0 * 5) % 13 - 6) * 0.125) })
        let yMulti16 = try act.allocate(QWTensorShape([10 * n], .f32))
        let yRef16 = try act.allocate(QWTensorShape([10 * n], .f32))
        let yOut16 = try act.allocate(QWTensorShape([n], .f16))

        // QWGemm.encode must reject non-incumbent non-.f16 .siluMulPair rather than silently passing.
        let planPair = QWGemm.plan(.init(m: 1, n: n, k: k), dtype: .q4g, epi: .siluMulPair)
        if let (_, enc) = hal.begin() {
            defer { enc.endEncoding() }
            XCTAssertThrowsError(try QWGemm.encode(enc, hal: hal, plan: planPair, x: x16, wQ4: nonInc, y: yMulti16))
            XCTAssertThrowsError(try QWGemmAQ.encode(enc, hal: hal, blocks: wBuf, x: x16, y: yMulti16,
                                                     n: 1 << 24, k: k, bits: 4, group: 32, rows: 2))
            XCTAssertThrowsError(try QWGemmAQ.encodeDecodeT4(enc, hal: hal, blocks: wBuf, x: x16, y16: yOut16,
                                                             n: 1 << 24, k: k, bits: 4, group: 32))
        }

        // QWGemm.encode .none with M=2 skinny plan routes non-incumbent across rows=2.
        let planM2 = QWGemm.plan(.init(m: 2, n: n, k: k), dtype: .q4g, epi: .none)
        let yPlan2 = try act.allocate(QWTensorShape([2 * n], .f32))
        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
        try QWGemm.encode(enc, hal: hal, plan: planM2, x: x16, wQ4: nonInc, y: yPlan2)
        // QWGemmAQ rows=10 (> 8) must match per-row rows=1.
        try QWGemmAQ.encode(enc, hal: hal, blocks: wBuf, x: x16, y: yMulti16, n: n, k: k, bits: 4, group: 32, rows: 10)
        for r in 0..<10 {
            try QWGemmAQ.encode(enc, hal: hal, blocks: wBuf,
                                x: x16.slice(byteOffset: r * k * 2, count: k, dtype: .f16),
                                y: yRef16.slice(byteOffset: r * n * 4, count: n, dtype: .f32),
                                n: n, k: k, bits: 4, group: 32, rows: 1)
        }
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        XCTAssertEqual(arena.read(yPlan2, as: Float.self, count: 2 * n),
                       Array(arena.read(yRef16, as: Float.self, count: 2 * n)))
        XCTAssertEqual(arena.read(yMulti16, as: Float.self, count: 10 * n),
                       arena.read(yRef16, as: Float.self, count: 10 * n))
    }

    func testLatentNormFwhtNonMultipleOf32ConvergesUniformly() throws {
        let hal = try makeHal(basis: .rotated)
        let rows = 2, latent = 40
        let arena = try QWArena(hal: hal, weights: 1024, activations: 64 << 10, kv: 0, gdnStates: 0)
        let act = arena.activationsRegion()
        let x = try act.allocate(QWTensorShape([rows * latent], .f32))
        let out = try act.allocate(QWTensorShape([rows * latent], .f32))
        let gamma = try act.allocate(QWTensorShape([latent], .f32))
        arena.write(x, from: (0..<(rows * latent)).map { Float(($0 % 9) - 4) * 0.25 })
        arena.write(gamma, from: [Float](repeating: 0.0, count: latent))

        let pso = try hal.pipeline("qwg_latent_norm_h32_f32")
        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
        enc.setComputePipelineState(pso)
        enc.bindTensors(x, gamma, out, out, startIndex: 0)
        enc.setUInt32s(rows, latent, 1, startIndex: 4)
        var eps: Float = 1e-6
        enc.setBytes(&eps, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        let got = arena.read(out, as: Float.self, count: rows * latent)
        for v in got { XCTAssertTrue(v.isFinite) }
    }
}
