// QWGemmApple9SGTests.swift — Parity tests for Apple9 simdgroup_half8x8 GEMM, SwiGLU, GDN, and MoE kernels.
import Foundation
import Metal
import XCTest
@testable import QWGEMM
import QWAlloc
import QWCore
import QWHAL
import QWQuant

final class QWGemmApple9SGTests: XCTestCase {
    private func makeHal(basis: QWWeightBasis = .rotated, geometry: QWModelGeometry = .qwen38_27b) throws -> QWHal {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true", "Requires Apple Silicon GPU")
        return try QWHal(geometry: geometry, weightBasis: basis)
    }

    private func makeBuf<T>(_ hal: QWHal, _ values: [T]) throws -> (any MTLBuffer, QWTensorValue, QWFPBufferSlice) {
        let len = max(values.count * MemoryLayout<T>.stride, 64)
        guard let buf = hal.device.makeBuffer(length: len, options: .storageModeShared) else {
            throw XCTSkip("MTLBuffer allocation failed")
        }
        values.withUnsafeBytes { raw in
            if let base = raw.baseAddress, raw.count > 0 {
                buf.contents().copyMemory(from: base, byteCount: raw.count)
            }
        }
        let dt: QWDType = MemoryLayout<T>.stride == 4 ? .f32 : .f16
        let tv = QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: len), offset: 0,
                               shape: QWTensorShape([max(1, values.count)], dt))
        return (buf, tv, QWFPBufferSlice(buffer: buf, offset: 0, bytes: len))
    }

    private func makeHQ4Data(rows: Int, k: Int, seedInit: UInt32 = 0x2468_ACE0) -> Data {
        var seed = seedInit
        func nextU() -> UInt32 { seed = seed &* 1664525 &+ 1013904223; return seed }
        let groups = k / 32
        var out = Data(count: rows * groups * 20)
        out.withUnsafeMutableBytes { raw in
            let p = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for b in 0..<(rows * groups) {
                let off = b * 20
                let sc = Float(Float16(0.01 * Float(1 + b % 5)))
                let scB = Float16(sc).bitPattern
                let biB = Float16(-1031.5 * sc).bitPattern
                p[off] = UInt8(scB & 0xFF); p[off + 1] = UInt8(scB >> 8)
                p[off + 2] = UInt8(biB & 0xFF); p[off + 3] = UInt8(biB >> 8)
                for j in 4..<20 { p[off + j] = UInt8((nextU() >> 16) & 0xFF) }
            }
        }
        return out
    }

    private func makeAQData(bits: Int, group: Int, rows: Int, k: Int) throws -> Data {
        var seed: UInt32 = 0x1357_2468
        func nextU() -> UInt32 { seed = seed &* 1664525 &+ 1013904223; return seed }
        let maxC = UInt32((1 << bits) - 1)
        var data = Data()
        for r in 0..<rows {
            for g in 0..<(k / group) {
                let vals = (0..<group).map { _ in (nextU() >> 12) % (maxC + 1) }
                try data.append(QWAQBlock.pack(bits: bits, group: group, values: vals,
                                               scale: 0.015625 * Float(1 + (r + g) % 4), bias: -0.25))
            }
        }
        return data
    }

    private func makeSHVQData(bits: Int, group: Int, rows: Int, k: Int) -> Data {
        var seed: UInt32 = 0x9988_7766
        func nextU() -> UInt32 { seed = seed &* 1664525 &+ 1013904223; return seed }
        let gb = QWSHVQBlock.bytes(bits: bits, group: group)
        let groups = k / group
        var out = Data(count: rows * groups * gb)
        out.withUnsafeMutableBytes { raw in
            let p = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for b in 0..<(rows * groups) {
                let off = b * gb
                let sc = Float16(0.025 * Float(1 + b % 5)).bitPattern
                p[off] = UInt8(sc & 0xFF); p[off + 1] = UInt8(sc >> 8)
                for j in 2..<gb { p[off + j] = UInt8((nextU() >> 16) & 0xFF) }
            }
        }
        return out
    }

    private func weightTensor(_ hal: QWHal, _ data: Data) throws -> QWTensorValue {
        let len = max(data.count, 64)
        let buf = hal.device.makeBuffer(length: len, options: .storageModeShared)!
        data.withUnsafeBytes { raw in
            if let base = raw.baseAddress { buf.contents().copyMemory(from: base, byteCount: data.count) }
        }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: len), offset: 0,
                             shape: QWTensorShape([max(1, data.count / 2)], .f16))
    }

    func testDenseSGAndF32ParityAcrossContainersAndShapes() throws {
        let hal = try makeHal()
        let n = 96, k = 128
        let mSizes = [7, 16, 25, 32, 48, 64, 96]
        let maxM = mSizes.max()!
        let xVals = (0..<(maxM * k)).map { Float16(sin(Double($0) * 0.07) * 0.4) }
        let (_, xTV, _) = try makeBuf(hal, xVals)

        struct Case { let label: String; let data: Data; let sg: String; let sgF32: String; let b: Int; let g: Int; let shvq: Bool; let inc: Bool }
        let shvqN = 128 // Aligned N for HQ4 reference compare
        let cases: [Case] = [
            Case(label: "HQ4", data: makeHQ4Data(rows: shvqN, k: k), sg: "qwg_gemm_q4g_sg", sgF32: "qwg_gemm_q4g_sg_f32", b: 4, g: 32, shvq: false, inc: true),
            Case(label: "AQ4_G64", data: try makeAQData(bits: 4, group: 64, rows: n, k: k), sg: "qwg_gemm_aq4_g64_sg", sgF32: "qwg_gemm_aq4_g64_sg_f32", b: 4, g: 64, shvq: false, inc: false),
            Case(label: "AQ8_G32", data: try makeAQData(bits: 8, group: 32, rows: n, k: k), sg: "qwg_gemm_aq8_g32_sg", sgF32: "qwg_gemm_aq8_g32_sg_f32", b: 8, g: 32, shvq: false, inc: false),
            Case(label: "SHVQ1_G64", data: makeSHVQData(bits: 1, group: 64, rows: n, k: k), sg: "qwg_gemm_shvq1_g64_sg", sgF32: "qwg_gemm_shvq1_g64_sg_f32", b: 1, g: 64, shvq: true, inc: false),
            Case(label: "SHVQ15_G64", data: makeSHVQData(bits: 15, group: 64, rows: n, k: k), sg: "qwg_gemm_shvq15_g64_sg", sgF32: "qwg_gemm_shvq15_g64_sg_f32", b: 15, g: 64, shvq: true, inc: false),
            Case(label: "SHVQ2_G64", data: makeSHVQData(bits: 2, group: 64, rows: n, k: k), sg: "qwg_gemm_shvq2_g64_sg", sgF32: "qwg_gemm_shvq2_g64_sg_f32", b: 2, g: 64, shvq: true, inc: false),
            Case(label: "SHVQ3_G64", data: makeSHVQData(bits: 3, group: 64, rows: n, k: k), sg: "qwg_gemm_shvq3_g64_sg", sgF32: "qwg_gemm_shvq3_g64_sg_f32", b: 3, g: 64, shvq: true, inc: false),
            Case(label: "SHVQ4_G32", data: makeSHVQData(bits: 4, group: 32, rows: n, k: k), sg: "qwg_gemm_shvq4_g32_sg", sgF32: "qwg_gemm_shvq4_g32_sg_f32", b: 4, g: 32, shvq: true, inc: false),
        ]

        for c in cases {
            let rowsN = c.inc ? shvqN : n
            let wTV = try weightTensor(hal, c.data)
            let wView = try XCTUnwrap(QWWeightView(
                blocks: wTV, n: rowsN, k: k,
                containerKind: c.inc ? .incumbentHQ4G32
                    : (c.shvq ? .shvq(bits: c.b, group: c.g) : .rotatedHQ(bits: c.b, group: c.g))))
            let pSG = try hal.pipeline(c.sg)
            let pF32 = try hal.pipeline(c.sgF32)
            let (refBuf, refTV, _) = try makeBuf(hal, [Float](repeating: 0, count: maxM * rowsN))
            let (out16Buf, out16TV, _) = try makeBuf(hal, [Float16](repeating: 0, count: maxM * rowsN))
            let (out32Buf, out32TV, _) = try makeBuf(hal, [Float](repeating: 0, count: maxM * rowsN))
            let (_, xRef32TV, _) = try makeBuf(hal, xVals.map { Float($0) })

            for m in mSizes {
                let bm = m <= 16 ? 16 : (m <= 32 ? 32 : 64)
                let mode = m <= 16 ? 0x2000_0000 : (m <= 32 ? 0x4000_0000 : 0)
                let (cb, enc) = try XCTUnwrap(hal.begin())
                var ended = false
                defer { if !ended { enc.endEncoding() } }
                if c.inc {
                    try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: wView,
                                              x: xRef32TV.slice(byteOffset: 0, count: m * k, dtype: .f32),
                                              y: refTV.slice(byteOffset: 0, count: m * rowsN, dtype: .f32),
                                              rows: m)
                } else {
                    try QWGemmAQ.encode(enc, hal: hal, blocks: wTV,
                                        x: xTV.slice(byteOffset: 0, count: m * k, dtype: .f16),
                                        y: refTV.slice(byteOffset: 0, count: m * rowsN, dtype: .f32),
                                        n: rowsN, k: k, bits: c.b, group: c.g, rows: m, isSHVQ: c.shvq)
                }
                enc.setComputePipelineState(pSG)
                enc.bindTensors(wTV, xTV, out16TV, out16TV, startIndex: 0)
                enc.setUInt32s(m | mode, rowsN, k, startIndex: 4)
                enc.setUInt32s(k, rowsN, startIndex: 9)
                enc.dispatchThreadgroups(MTLSize(width: (rowsN + 63) / 64, height: (m + bm - 1) / bm, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

                enc.setComputePipelineState(pF32)
                enc.bindTensors(wTV, xTV, xTV, out32TV, startIndex: 0)
                enc.setUInt32s(m | mode, rowsN, k, startIndex: 4)
                enc.dispatchThreadgroups(MTLSize(width: (rowsN + 63) / 64, height: (m + bm - 1) / bm, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                ended = true
                hal.end(cb, enc); cb.waitUntilCompleted()
                XCTAssertNil(cb.error)

                let rPtr = refBuf.contents().assumingMemoryBound(to: Float.self)
                let hPtr = out16Buf.contents().assumingMemoryBound(to: Float16.self)
                let fPtr = out32Buf.contents().assumingMemoryBound(to: Float.self)
                var maxD32: Float = 0
                for idx in 0..<(m * rowsN) {
                    let tol16: Float = max(2e-2, 2e-3 * abs(rPtr[idx]))
                    XCTAssertLessThan(abs(Float(hPtr[idx]) - rPtr[idx]), tol16, "\(c.label) sg M=\(m)")
                    maxD32 = max(maxD32, abs(fPtr[idx] - rPtr[idx]))
                }
                XCTAssertLessThan(maxD32, 2e-2, "\(c.label) sg_f32 M=\(m) maxD32=\(maxD32)")
            }
        }
    }

    func testSwigluAndGdnInProjSGParity() throws {
        let hal = try makeHal(basis: .rotated, geometry: .qwen38_27b)
        let iDim = 96, k = 128, m = 25
        let xVals = (0..<(m * k)).map { Float16(cos(Double($0) * 0.05) * 0.35) }
        let (_, xTV, _) = try makeBuf(hal, xVals)
        let formats: [(String, Data, String, String)] = [
            ("HQ4", makeHQ4Data(rows: 2 * iDim, k: k), "qwg_gemm_q4g_sg_f32", "qwg_gemm_q4g_sg_swiglu"),
            ("AQ4", try makeAQData(bits: 4, group: 64, rows: 2 * iDim, k: k), "qwg_gemm_aq4_g64_sg_f32", "qwg_gemm_aq4_g64_sg_swiglu"),
            ("SHVQ4", makeSHVQData(bits: 4, group: 32, rows: 2 * iDim, k: k), "qwg_gemm_shvq4_g32_sg_f32", "qwg_gemm_shvq4_g32_sg_swiglu"),
        ]
        for (label, wData, gName, sName) in formats {
            let wTV = try weightTensor(hal, wData)
            let pG = try hal.pipeline(gName), pS = try hal.pipeline(sName)
            let (_, guTV, _) = try makeBuf(hal, [Float](repeating: 0, count: m * 2 * iDim))
            let (refBuf, refTV, _) = try makeBuf(hal, [Float](repeating: 0, count: m * iDim))
            let (fusBuf, fusTV, _) = try makeBuf(hal, [Float16](repeating: 0, count: m * iDim))
            let (cb, enc) = try XCTUnwrap(hal.begin())
            enc.setComputePipelineState(pG)
            enc.bindTensors(wTV, xTV, xTV, guTV, startIndex: 0)
            enc.setUInt32s(m | 0x4000_0000, 2 * iDim, k, startIndex: 4)
            enc.dispatchThreadgroups(MTLSize(width: (2 * iDim + 63) / 64, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guTV, act: refTV, rows: m, intermediate: iDim)

            enc.setComputePipelineState(pS)
            enc.bindTensors(wTV, xTV, fusTV, startIndex: 0)
            enc.setUInt32s(m | 0x4000_0000, iDim, k, k, iDim, startIndex: 3)
            enc.dispatchThreadgroups(MTLSize(width: iDim / 32, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            hal.end(cb, enc); cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
            let rp = refBuf.contents().assumingMemoryBound(to: Float.self)
            let fp = fusBuf.contents().assumingMemoryBound(to: Float16.self)
            for idx in 0..<(m * iDim) {
                let tol: Float = max(2e-2, 2e-3 * abs(rp[idx]))
                XCTAssertLessThan(abs(Float(fp[idx]) - rp[idx]), tol, "\(label) sg_swiglu mismatch")
            }
        }

        // GDN in-proj split test on matching qwen38_27b geometry
        let geom = QWModelGeometry.qwen38_27b
        let gdnHal = try makeHal(basis: .rotated, geometry: geom)
        let nTotal = geom.gdnTotalN, qkvC = geom.gdnQKVChannels, zC = geom.gdnZChannels, abC = max(1, nTotal - qkvC - zC)
        let gm = 19, gk = 128
        let gxVals = (0..<(gm * gk)).map { Float16(sin(Double($0) * 0.09) * 0.3) }
        let (_, gxTV, _) = try makeBuf(gdnHal, gxVals)
        let gwTV = try weightTensor(gdnHal, makeHQ4Data(rows: nTotal, k: gk))
        let (fullBuf, fullTV, _) = try makeBuf(gdnHal, [Float16](repeating: 0, count: gm * nTotal))
        let (qkvBuf, qkvTV, _) = try makeBuf(gdnHal, [Float16](repeating: 0, count: gm * qkvC))
        let (zBuf, zTV, _) = try makeBuf(gdnHal, [Float16](repeating: 0, count: gm * zC))
        let (baBuf, baTV, _) = try makeBuf(gdnHal, [Float](repeating: 0, count: gm * abC))
        let (cb2, enc2) = try XCTUnwrap(gdnHal.begin())
        enc2.setComputePipelineState(try gdnHal.pipeline("qwg_gemm_q4g_sg"))
        enc2.bindTensors(gwTV, gxTV, fullTV, fullTV, startIndex: 0)
        enc2.setUInt32s(gm | 0x4000_0000, nTotal, gk, startIndex: 4); enc2.setUInt32s(gk, nTotal, startIndex: 9)
        enc2.dispatchThreadgroups(MTLSize(width: (nTotal + 63) / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc2.setComputePipelineState(try gdnHal.pipeline("qwg_gemm_q4g_sg_gdn_inproj"))
        enc2.bindTensors(gwTV, gxTV, qkvTV, baTV, startIndex: 0)
        enc2.setUInt32s(gm, nTotal, gk, startIndex: 4); zTV.bind(enc2, index: 7)
        enc2.dispatchThreadgroups(MTLSize(width: (nTotal + 63) / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        gdnHal.end(cb2, enc2); cb2.waitUntilCompleted()
        XCTAssertNil(cb2.error)
        let fP = fullBuf.contents().assumingMemoryBound(to: Float16.self)
        let qP = qkvBuf.contents().assumingMemoryBound(to: Float16.self)
        let zP = zBuf.contents().assumingMemoryBound(to: Float16.self)
        let bP = baBuf.contents().assumingMemoryBound(to: Float.self)
        for r in 0..<gm {
            for c in 0..<qkvC { XCTAssertEqual(Float(qP[r * qkvC + c]), Float(fP[r * nTotal + c]), accuracy: 1e-4) }
            for c in 0..<zC { XCTAssertEqual(Float(zP[r * zC + c]), Float(fP[r * nTotal + qkvC + c]), accuracy: 1e-4) }
            for c in 0..<abC { XCTAssertEqual(bP[r * abC + c], Float(fP[r * nTotal + qkvC + zC + c]), accuracy: 2e-2) }
        }
    }

    func testGroupedMoEAndFPOperatorsSGParity() throws {
        let hal = try makeHal()
        let E = 4, K = 2, T = 20, H = 128, I = 64
        let pairs = T * K
        let plan = QWGemm.MoETilePlan(experts: E, pairs: pairs, gateUpRows: 2 * I, downRows: H)
        var ids = [Int32](repeating: 0, count: pairs)
        for i in 0..<pairs { ids[i] = Int32((i * 3) % E) }
        let xVals = (0..<(T * H)).map { Float16(sin(Double($0) * 0.13) * 0.25) }
        let (_, idsTV, _) = try makeBuf(hal, ids)
        let (_, expOffTV, _) = try makeBuf(hal, [Int32](repeating: 0, count: E + 1))
        let (_, sortedPairTV, _) = try makeBuf(hal, [Int32](repeating: 0, count: pairs))
        let (_, sortedTokTV, _) = try makeBuf(hal, [Int32](repeating: 0, count: pairs))
        let (_, tileETV, _) = try makeBuf(hal, [Int32](repeating: 0, count: plan.maxTiles))
        let (_, tileRTV, _) = try makeBuf(hal, [Int32](repeating: 0, count: plan.maxTiles))
        let (_, tileCountTV, _) = try makeBuf(hal, [Int32](repeating: 0, count: plan.scratchInts))
        let (_, xTV, _) = try makeBuf(hal, xVals)
        let wGU = try weightTensor(hal, makeHQ4Data(rows: E * 2 * I, k: H, seedInit: 0x3333))
        let wDN = try weightTensor(hal, makeHQ4Data(rows: E * H, k: I, seedInit: 0x4444))
        let (actBuf, actTV, _) = try makeBuf(hal, [Float16](repeating: 0, count: pairs * I))
        let (dnBuf, dnTV, _) = try makeBuf(hal, [Float](repeating: 0, count: pairs * H))

        let pBuild = try hal.pipeline("qwg_moe_group_pairs_f32")
        let pGUSg = try hal.pipeline("qwg_moe_sg_swiglu_q4g")
        let pDNSg = try hal.pipeline("qwg_moe_sg_split_f32")
        let (cb, enc) = try XCTUnwrap(hal.begin())
        enc.setComputePipelineState(pBuild)
        enc.bindTensors(idsTV, expOffTV, sortedPairTV, sortedTokTV, startIndex: 0)
        enc.setUInt32s(E, pairs, K, startIndex: 4)
        sortedPairTV.bind(enc, index: 7); enc.setUInt32s(0, startIndex: 8)
        enc.bindTensors(tileETV, tileRTV, tileCountTV, startIndex: 9)
        enc.setUInt32s(plan.packedWriteWord, startIndex: 12)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        enc.setComputePipelineState(pGUSg)
        enc.bindTensors(wGU, xTV, actTV, expOffTV, tileETV, tileRTV, tileCountTV, startIndex: 0)
        enc.setPod(QWNaxGroupedParams(experts: UInt32(E), n: UInt32(2 * I), k: UInt32(H), blocksPerRow: UInt32(H / 32), rowTiles: 0, colTiles: UInt32(plan.guColTiles)), index: 7)
        sortedTokTV.bind(enc, index: 9); enc.setUInt32s(1, startIndex: 10); wGU.bind(enc, index: 11)
        QWGemm.dispatchTileChunks(enc, indirectBuffer: tileCountTV.region.buffer, baseByteOffset: tileCountTV.region.base + 16, colTiles: plan.guColTiles, maxTiles: plan.maxTiles, flags: 0, flagsIndex: 8)
        enc.memoryBarrier(scope: .buffers)

        enc.setComputePipelineState(pDNSg)
        enc.bindTensors(wDN, actTV, actTV, dnTV, expOffTV, tileETV, tileRTV, tileCountTV, startIndex: 0)
        enc.setPod(QWNaxGroupedParams(experts: UInt32(E), n: UInt32(H), k: UInt32(I), blocksPerRow: UInt32(I / 32), rowTiles: 0, colTiles: UInt32(plan.dnColTiles)), index: 8)
        wDN.bind(enc, index: 10)
        QWGemm.dispatchTileChunks(enc, indirectBuffer: tileCountTV.region.buffer, baseByteOffset: tileCountTV.region.base + plan.downIndirectByteOffset, colTiles: plan.dnColTiles, maxTiles: plan.maxTiles, flags: 0, flagsIndex: 9)
        hal.end(cb, enc); cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        let aPtr = actBuf.contents().assumingMemoryBound(to: Float16.self)
        let dPtr = dnBuf.contents().assumingMemoryBound(to: Float.self)
        XCTAssertTrue((0..<(pairs * I)).allSatisfy { aPtr[$0].isFinite })
        XCTAssertTrue((0..<(pairs * H)).allSatisfy { dPtr[$0].isFinite })

        // Verify BF16 SG clamp (> 65504) produces finite clamped output matching CPU
        let bfM = 48, bfN = 64, bfK = 64
        var xLarge = [Float](repeating: 0.5, count: bfM * bfK)
        xLarge[0] = 100_000.0
        var wLarge = [UInt16](repeating: UInt16(0x3F80), count: bfN * bfK) // BF16 1.0
        wLarge[0] = UInt16(0x4780) // BF16 65536.0 (> 65504)
        let (_, wBS, wBSL) = try makeBuf(hal, wLarge)
        _ = wBS
        let (_, _, xLargeSL) = try makeBuf(hal, xLarge)
        let (yBFBuf, _, yBFSL) = try makeBuf(hal, [Float](repeating: 0, count: bfM * bfN))
        struct SGBF16Params { var m, n, k, strideX, strideY, inIsF32, outIsF32, addResidual: UInt32; var softcap: Float }
        let params = SGBF16Params(m: UInt32(bfM), n: UInt32(bfN), k: UInt32(bfK),
                                  strideX: UInt32(bfK), strideY: UInt32(bfN),
                                  inIsF32: 1, outIsF32: 1, addResidual: 0, softcap: 0)
        let (cb3, enc3) = try XCTUnwrap(hal.begin())
        enc3.setComputePipelineState(try hal.pipeline("qwg_gemm_bf16_sg_bm64"))
        wBSL.bind(enc3, index: 0); xLargeSL.bind(enc3, index: 1); xLargeSL.bind(enc3, index: 2)
        yBFSL.bind(enc3, index: 3); yBFSL.bind(enc3, index: 4)
        enc3.setPod(params, index: 5)
        enc3.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        hal.end(cb3, enc3); cb3.waitUntilCompleted()
        XCTAssertNil(cb3.error)
        let bfOut = yBFBuf.contents().assumingMemoryBound(to: Float.self)
        XCTAssertTrue(bfOut[0].isFinite, "clamped BF16->FP16 SG must stay finite")
        XCTAssertEqual(bfOut[bfN + 1], Float(bfK) * 0.5, accuracy: 0.1)

        // Verify NVFP4_G16 & MXFP8_G32 grouped MoE SG kernels against deterministic FP reference
        let fpE = 2, fpM = 18, fpN = 64, fpK = 64
        let fpPlan = QWGemm.MoETilePlan(experts: fpE, pairs: fpM, gateUpRows: 2 * fpN, downRows: fpN)
        let fpIds = (0..<fpM).map { Int32($0 % fpE) }
        let (_, fpIdsTV, _) = try makeBuf(hal, fpIds)
        let (_, fpExpOffTV, _) = try makeBuf(hal, [Int32](repeating: 0, count: fpE + 1))
        let (_, fpSP, _) = try makeBuf(hal, [Int32](repeating: 0, count: fpM))
        let (_, fpST, _) = try makeBuf(hal, [Int32](repeating: 0, count: fpM))
        let (_, fpTE, _) = try makeBuf(hal, [Int32](repeating: 0, count: fpPlan.maxTiles))
        let (_, fpTR, _) = try makeBuf(hal, [Int32](repeating: 0, count: fpPlan.maxTiles))
        let (_, fpTC, _) = try makeBuf(hal, [Int32](repeating: 0, count: fpPlan.scratchInts))
        let fpX = (0..<(fpM * fpK)).map { Float16(0.25 * Float(($0 % 5) - 2)) }
        let (_, _, fpXS) = try makeBuf(hal, fpX)
        // NVFP4 nibble 0x22 -> 1.0, scale 0x38 -> 1.0, s2 = 0.5 => weight 0.5
        let nvW = [UInt32](repeating: 0x2222_2222, count: fpE * fpN * (fpK / 8))
        let nvSc = [UInt8](repeating: 0x38, count: fpE * fpN * (fpK / 16))
        let (_, _, nvWS) = try makeBuf(hal, nvW)
        let (_, _, nvScS) = try makeBuf(hal, nvSc)
        let (_, _, nvS2S) = try makeBuf(hal, [Float](repeating: 0.5, count: fpE))
        let (nvOutBuf, _, nvOutS) = try makeBuf(hal, [Float](repeating: 0, count: fpM * fpN))
        // MXFP8 byte 0x38 -> 1.0, UE8M0 127 -> 1.0 => weight 1.0
        let mxW = [UInt32](repeating: 0x3838_3838, count: fpE * fpN * (fpK / 4))
        let mxSc = [UInt8](repeating: 127, count: fpE * fpN * (fpK / 32))
        let (_, _, mxWS) = try makeBuf(hal, mxW)
        let (_, _, mxScS) = try makeBuf(hal, mxSc)
        let (mxOutBuf, _, mxOutS) = try makeBuf(hal, [Float](repeating: 0, count: fpM * fpN))

        let (cb4, enc4) = try XCTUnwrap(hal.begin())
        try QWGemm.encodeMoEGroupPairsF32(enc4, hal: hal, ids: fpIdsTV, expertOff: fpExpOffTV,
                                          sortedPairIdx: fpSP, sortedTok: fpST, experts: fpE,
                                          totalPairs: fpM, routes: 1, tileE: fpTE, tileR: fpTR,
                                          tileCount: fpTC, tilePlan: fpPlan)
        enc4.memoryBarrier(scope: .buffers)
        enc4.setComputePipelineState(try hal.pipeline("qwg_moe_sg_gemm_nvfp4_g16"))
        nvWS.bind(enc4, index: 0); nvScS.bind(enc4, index: 1); nvS2S.bind(enc4, index: 2)
        fpXS.bind(enc4, index: 3); nvOutS.bind(enc4, index: 4)
        enc4.bindTensors(fpExpOffTV, fpTE, fpTR, fpTC, startIndex: 5)
        enc4.setPod(QWGemmFPMoENaxParams(experts: UInt32(fpE), n: UInt32(fpN), k: UInt32(fpK), yStride: UInt32(fpN), yColOff: 0, inDtype: 0, outDtype: 2, flags: 3), index: 9)
        fpST.bind(enc4, index: 11)
        QWGemm.dispatchTileChunks(enc4, indirectBuffer: fpTC.region.buffer, baseByteOffset: fpTC.region.base + 16, colTiles: fpPlan.dnColTiles, maxTiles: fpPlan.maxTiles, flags: 0, flagsIndex: 10)

        enc4.setComputePipelineState(try hal.pipeline("qwg_moe_sg_gemm_mxfp8_g32"))
        mxWS.bind(enc4, index: 0); mxScS.bind(enc4, index: 1)
        fpXS.bind(enc4, index: 2); mxOutS.bind(enc4, index: 3)
        enc4.bindTensors(fpExpOffTV, fpTE, fpTR, fpTC, startIndex: 4)
        enc4.setPod(QWGemmFPMoENaxParams(experts: UInt32(fpE), n: UInt32(fpN), k: UInt32(fpK), yStride: UInt32(fpN), yColOff: 0, inDtype: 0, outDtype: 2, flags: 2), index: 8)
        fpST.bind(enc4, index: 10)
        QWGemm.dispatchTileChunks(enc4, indirectBuffer: fpTC.region.buffer, baseByteOffset: fpTC.region.base + 16, colTiles: fpPlan.dnColTiles, maxTiles: fpPlan.maxTiles, flags: 0, flagsIndex: 9)
        hal.end(cb4, enc4); cb4.waitUntilCompleted()
        XCTAssertNil(cb4.error)
        let stHost = fpST.region.buffer.contents().advanced(by: fpST.region.base).assumingMemoryBound(to: Int32.self)
        let nvPtr = nvOutBuf.contents().assumingMemoryBound(to: Float.self)
        let mxPtr = mxOutBuf.contents().assumingMemoryBound(to: Float.self)
        for slot in 0..<fpM {
            let tok = Int(stHost[slot])
            let rowSum = (0..<fpK).reduce(Float(0)) { $0 + Float(fpX[tok * fpK + $1]) }
            XCTAssertEqual(nvPtr[slot * fpN], rowSum * 0.5, accuracy: 1e-2)
            XCTAssertEqual(mxPtr[slot * fpN], rowSum, accuracy: 1e-2)
        }
    }

    func testNaxRowChunkTiers() {
        XCTAssertEqual(QWGemm.naxRowChunk(rows: 512, k: 4096, gpuCores: 40), 512)
        XCTAssertEqual(QWGemm.naxRowChunk(rows: 4096, k: 17408, gpuCores: 40), 704)
        XCTAssertEqual(QWGemm.naxRowChunk(rows: 4096, k: 17408, gpuCores: 16), 320)
        XCTAssertEqual(QWGemm.naxRowChunk(rows: 4096, k: 17408, gpuCores: 8), 128)
        XCTAssertEqual(QWGemm.naxRowChunk(rows: 2048, k: 4096, gpuCores: 16), 1536)
        XCTAssertEqual(QWGemm.naxRowChunk(rows: 2048, k: 4096, gpuCores: 8), 768)
    }
}
