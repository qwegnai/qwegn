// QWMLAGeneralizedTests.swift — GPU parity tests for generalized NoPE MLA
// across SHVQ{1,15,2,3,4}, AQ/HQ{3,4,8}, and paged q8/q4 KV (MMA, NAX, split-K, and decode).

import Foundation
import Metal
import XCTest
@testable import QWAttn
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWKV
import QWQuant

final class QWMLAGeneralizedTests: XCTestCase {
    private func makeBuffer<T: BitwiseCopyable>(_ values: [T], device: any MTLDevice) -> any MTLBuffer {
        let byteCount = values.count * MemoryLayout<T>.stride
        let buffer = device.makeBuffer(length: max(16, byteCount), options: .storageModeShared)!
        values.withUnsafeBufferPointer { src in
            if let base = src.baseAddress {
                buffer.contents().copyMemory(from: base, byteCount: byteCount)
            }
        }
        return buffer
    }

    private func makeDataBuffer(_ data: Data, device: any MTLDevice) -> any MTLBuffer {
        let buffer = device.makeBuffer(length: max(16, data.count), options: .storageModeShared)!
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let base = raw.baseAddress {
                buffer.contents().copyMemory(from: base, byteCount: raw.count)
            }
        }
        return buffer
    }

    private func fwht32(_ x: inout [Float], base: Int) {
        let invSqrt32 = Float(1.0 / 32.0).squareRoot()
        var stride = 1
        while stride < 32 {
            var i = 0
            while i < 32 {
                for j in 0..<stride {
                    let a = x[base + i + j]
                    let b = x[base + i + j + stride]
                    x[base + i + j] = a + b
                    x[base + i + j + stride] = a - b
                }
                i += stride * 2
            }
            stride *= 2
        }
        for i in 0..<32 { x[base + i] *= invSqrt32 }
    }

    private func makeSHVQBank(bits: Int, group: Int, n: Int, k: Int) -> (Data, [[Float]]) {
        let groups = k / group
        let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
        var seed: UInt32 = 0x9E37_79B9 ^ UInt32(bits * 131 + group)
        var bank = Data()
        bank.reserveCapacity(n * groups * gBytes)
        var unpacked = [[Float]]()
        unpacked.reserveCapacity(n)
        for r in 0..<n {
            var row = [Float]()
            row.reserveCapacity(k)
            var tmp = [Float]()
            for g in 0..<groups {
                var block = [UInt8](repeating: 0, count: gBytes)
                let sc = Float16(0.03125 * (1.0 + 0.125 * Float((r + g) % 5))).bitPattern
                block[0] = UInt8(sc & 0xFF)
                block[1] = UInt8(sc >> 8)
                for b in 2..<gBytes {
                    seed = seed &* 1664525 &+ 1013904223
                    block[b] = UInt8((seed >> 16) & 0xFF)
                }
                block.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                    QWSHVQBlock.unpack(group: raw, bits: bits, group: group, into: &tmp)
                }
                row.append(contentsOf: tmp)
                bank.append(contentsOf: block)
            }
            unpacked.append(row)
        }
        return (bank, unpacked)
    }

    private func makeAQBank(bits: Int, group: Int, n: Int, k: Int) -> (Data, [[Float]]) {
        let groups = k / group
        let payBytes = (group * bits) / 8
        let gBytes = 4 + payBytes
        var seed: UInt32 = 0xA5A5_1234 ^ UInt32(bits * 97 + group)
        var bank = Data()
        bank.reserveCapacity(n * groups * gBytes)
        var unpacked = [[Float]]()
        unpacked.reserveCapacity(n)
        let qMask = UInt32((1 << bits) - 1)
        for r in 0..<n {
            var row = [Float]()
            row.reserveCapacity(k)
            for g in 0..<groups {
                var block = [UInt8](repeating: 0, count: gBytes)
                let scVal: Float = 0.015625 * (1.0 + 0.125 * Float((r + g) % 4))
                let biVal: Float = -scVal * Float(1 << (bits - 1))
                let scBits = Float16(scVal).bitPattern
                let biBits = Float16(biVal).bitPattern
                block[0] = UInt8(scBits & 0xFF); block[1] = UInt8(scBits >> 8)
                block[2] = UInt8(biBits & 0xFF); block[3] = UInt8(biBits >> 8)
                for b in 4..<gBytes {
                    seed = seed &* 1664525 &+ 1013904223
                    block[b] = UInt8((seed >> 16) & 0xFF)
                }
                let scF = Float(Float16(bitPattern: scBits))
                let biF = Float(Float16(bitPattern: biBits))
                for i in 0..<group {
                    let bitPos = i * bits
                    let byteIdx = 4 + (bitPos >> 3)
                    let shift = bitPos & 7
                    var raw = UInt32(block[byteIdx])
                    if byteIdx + 1 < gBytes { raw |= UInt32(block[byteIdx + 1]) << 8 }
                    let q = (raw >> shift) & qMask
                    row.append(Float(q) * scF + biF)
                }
                bank.append(contentsOf: block)
            }
            unpacked.append(row)
        }
        return (bank, unpacked)
    }

    private func makeQuantKVPool(pages: Int, halfRank: Int, kvMode: UInt32, seedInit: UInt32) -> (Data, [[Float]]) {
        var seed = seedInit
        let totalRows = pages * 256
        var decoded = Array(repeating: [Float](repeating: 0, count: halfRank), count: totalRows)
        if kvMode == 1 {
            let pageBytes = 65536 + 512
            var bytes = [UInt8](repeating: 0, count: pages * pageBytes)
            for pg in 0..<pages {
                let pBase = pg * pageBytes
                for off in 0..<256 {
                    seed = seed &* 1664525 &+ 1013904223
                    let scH = Float16(0.01 + 0.005 * Float(seed & 7))
                    let scBits = scH.bitPattern
                    bytes[pBase + 65536 + off * 2] = UInt8(scBits & 0xFF)
                    bytes[pBase + 65536 + off * 2 + 1] = UInt8(scBits >> 8)
                    let scF = Float(scH)
                    for c in 0..<halfRank {
                        seed = seed &* 1664525 &+ 1013904223
                        let q = Int8(truncatingIfNeeded: Int32(seed >> 24) - 128)
                        bytes[pBase + off * halfRank + c] = UInt8(bitPattern: q)
                        decoded[pg * 256 + off][c] = Float(Float16(Float(q) * scF))
                    }
                }
            }
            return (Data(bytes), decoded)
        } else {
            let pageBytes = 32768 + 2048
            var bytes = [UInt8](repeating: 0, count: pages * pageBytes)
            for pg in 0..<pages {
                let pBase = pg * pageBytes
                for off in 0..<256 {
                    var scF = [Float](repeating: 0, count: 2)
                    var mnF = [Float](repeating: 0, count: 2)
                    for g in 0..<2 {
                        seed = seed &* 1664525 &+ 1013904223
                        let scH = Float16(0.04 + 0.01 * Float(seed & 3))
                        let mnH = Float16(-0.3 - 0.05 * Float((seed >> 4) & 3))
                        let metaOff = pBase + 32768 + (off * 2 + g) * 4
                        bytes[metaOff + 0] = UInt8(scH.bitPattern & 0xFF)
                        bytes[metaOff + 1] = UInt8(scH.bitPattern >> 8)
                        bytes[metaOff + 2] = UInt8(mnH.bitPattern & 0xFF)
                        bytes[metaOff + 3] = UInt8(mnH.bitPattern >> 8)
                        scF[g] = Float(scH)
                        mnF[g] = Float(mnH)
                    }
                    for b in 0..<(halfRank / 2) {
                        seed = seed &* 1664525 &+ 1013904223
                        let packed = UInt8((seed >> 16) & 0xFF)
                        bytes[pBase + off * 128 + b] = packed
                        let g = (2 * b) >> 7
                        let n0 = Float(packed & 0x0F)
                        let n1 = Float(packed >> 4)
                        decoded[pg * 256 + off][2 * b] = Float(Float16(n0 * scF[g] + mnF[g]))
                        decoded[pg * 256 + off][2 * b + 1] = Float(Float16(n1 * scF[g] + mnF[g]))
                    }
                }
            }
            return (Data(bytes), decoded)
        }
    }

    func testSparseAttnGeneralizedKVModesMatchScalarAndCPU() throws {
        let geo = try QWMLAAttnOperatorTests.mlaGeometry()
        let hal = try QWHal(geometry: geo)
        let scalar = try hal.pipeline("qwg_mla_sparse_attn")
        let mma = try hal.pipeline("qwg_mla_sparse_attn_mma")
        let combine = try hal.pipeline("qwg_mla_sparse_attn_combine")
        var variants: [(any MTLComputePipelineState, Int)] = [(scalar, 0), (mma, 1)]
        if hal.caps.isMetal4TensorCapable {
            let split = try hal.pipeline("qwg_mla_sparse_attn_nax_split")
            variants += [(try hal.pipeline("qwg_mla_sparse_attn_nax"), 1), (split, 7), (split, 16)]
        }

        let qHeads = 64, rank = 512, halfRank = 256, pages = 6
        let kt = [3, 0, 5, 1, 4, 2], vt = [1, 4, 2, 5, 0, 3]
        let kTable = makeBuffer(kt.map(Int32.init), device: hal.device)
        let vTable = makeBuffer(vt.map(Int32.init), device: hal.device)

        for kvMode: UInt32 in [1, 2] {
            let (kData, kDec) = makeQuantKVPool(pages: pages, halfRank: halfRank, kvMode: kvMode, seedInit: 0x1111_2222 + kvMode)
            let (vData, vDec) = makeQuantKVPool(pages: pages, halfRank: halfRank, kvMode: kvMode, seedInit: 0x3333_4444 + kvMode)
            let kPool = makeDataBuffer(kData, device: hal.device)
            let vPool = makeDataBuffer(vData, device: hal.device)
            func latent(_ j: Int, _ c: Int) -> Float {
                let off = j & 255
                return c < halfRank ? kDec[kt[j >> 8] * 256 + off][c] : vDec[vt[j >> 8] * 256 + off][c - halfRank]
            }

            var seed: UInt32 = 0x5566_7788 ^ kvMode
            for (pos0, rows, topK) in [(0, 37, 512), (500, 9, 16), (1300, 2, 64), (1300, 1, 64)] {
                let qVals: [UInt16] = (0..<(rows * qHeads * rank)).map { _ in
                    seed = seed &* 1664525 &+ 1013904223
                    let v = Float(Int32(truncatingIfNeeded: seed >> 16) - 32768) / 65536.0
                    return QWQuant.f32tof16Bits(v)
                }
                let q = makeBuffer(qVals, device: hal.device)
                var list = [Int32](repeating: 0, count: rows * topK)
                for t in 0..<rows {
                    let nPools = max(1, (pos0 + t + 1) >> 2)
                    let cnt = min(topK, nPools)
                    for i in 0..<cnt { list[t * topK + i] = Int32((i * 7 + t) % nPools) }
                }
                let blockList = makeBuffer(list, device: hal.device)
                var params: [UInt32] = [UInt32(rows), UInt32(pos0), UInt32(qHeads), 192, UInt32(rank),
                                        UInt32(topK), UInt32(topK), kvMode]
                var outs = [[UInt16]]()
                for (pso, nSplit) in variants {
                    let out = hal.device.makeBuffer(length: rows * qHeads * rank * 2, options: .storageModeShared)!
                    let partO = hal.device.makeBuffer(length: rows * max(1, nSplit) * qHeads * rank * 2, options: .storageModeShared)!
                    let partML = hal.device.makeBuffer(length: rows * max(1, nSplit) * qHeads * 8, options: .storageModeShared)!
                    guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
                    enc.setComputePipelineState(pso)
                    for (idx, b) in [q, kPool, vPool, kTable, vTable, blockList, nSplit > 1 ? partO : out].enumerated() {
                        enc.setBuffer(b, offset: 0, index: idx)
                    }
                    enc.setBytes(&params, length: params.count * 4, index: 7)
                    if nSplit > 1 {
                        enc.setBuffer(partML, offset: 0, index: 8)
                        enc.dispatchThreadgroups(MTLSize(width: rows, height: qHeads / 16, depth: nSplit),
                                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                        var n = UInt32(nSplit)
                        enc.setComputePipelineState(combine)
                        for (idx, b) in [partO, partML, out].enumerated() { enc.setBuffer(b, offset: 0, index: idx) }
                        enc.setBytes(&params, length: params.count * 4, index: 3)
                        enc.setBytes(&n, length: 4, index: 4)
                        enc.dispatchThreadgroups(MTLSize(width: rows * qHeads, height: 1, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                    } else if nSplit == 1 {
                        enc.dispatchThreadgroups(MTLSize(width: rows, height: qHeads / 16, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                    } else {
                        enc.dispatchThreadgroups(MTLSize(width: qHeads * rows, height: 1, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    }
                    hal.end(cb, enc)
                    cb.waitUntilCompleted()
                    XCTAssertNil(cb.error)
                    let ptr = out.contents().bindMemory(to: UInt16.self, capacity: rows * qHeads * rank)
                    outs.append(Array(UnsafeBufferPointer(start: ptr, count: rows * qHeads * rank)))
                }

                for t in [0, rows - 1] {
                    let pq = pos0 + t, nPools = (pq + 1) >> 2, cnt = min(topK, nPools)
                    var toks = [Int]()
                    for b in 0..<cnt {
                        let base = (nPools <= topK ? b : Int(list[t * topK + b])) * 4
                        toks += [base, base + 1, base + 2, base + 3]
                    }
                    toks += Array((nPools * 4)..<(pq + 1))
                    for h in [0, 17, 63] {
                        let qo = (t * qHeads + h) * rank
                        let sc = toks.map { j in
                            (0..<rank).reduce(Float(0)) { $0 + QWQuant.f16tof32Bits(qVals[qo + $1]) * latent(j, $1) }
                                / Float(192).squareRoot()
                        }
                        let m = sc.max()!
                        let w = sc.map { exp($0 - m) }
                        let l = w.reduce(0, +)
                        for c in stride(from: 0, to: rank, by: 13) {
                            var ref: Float = 0
                            for (i, j) in toks.enumerated() { ref += w[i] * latent(j, c) }
                            ref /= l
                            for k in 0..<outs.count {
                                let err = abs(QWQuant.f16tof32Bits(outs[k][qo + c]) - ref)
                                XCTAssertLessThan(err, 0.012, "kvMode=\(kvMode) variant=\(k) pos0=\(pos0) rows=\(rows)")
                            }
                        }
                    }
                }
                for k in 1..<outs.count {
                    var maxErr: Float = 0, maxRef: Float = 0
                    for i in 0..<outs[0].count {
                        let a = QWQuant.f16tof32Bits(outs[0][i]), b = QWQuant.f16tof32Bits(outs[k][i])
                        XCTAssertTrue(b.isFinite)
                        maxErr = max(maxErr, abs(a - b))
                        maxRef = max(maxRef, abs(a))
                    }
                    XCTAssertGreaterThan(maxRef, 0.03)
                    XCTAssertLessThan(maxErr, 0.012 * max(maxRef, 1.0),
                                      "kvMode=\(kvMode) variant=\(k) pos0=\(pos0) rows=\(rows)")
                }
            }
        }
    }

    func testAbsorbQAndProjUVGeneralizedSHVQAndAQMatchCPUAndScalar() throws {
        let hal = try QWHal(geometry: try QWMLAAttnOperatorTests.mlaGeometry())
        let absorbScalar = try hal.pipeline("qwg_mla_absorb_q")
        let absorbScalarSub2 = try hal.pipeline("qwg_mla_absorb_q_sub2")
        let absorbMMA = try hal.pipeline("qwg_mla_absorb_q_mma")
        let absorbMMASub2 = try hal.pipeline("qwg_mla_absorb_q_mma_sub2")
        let projScalar = try hal.pipeline("qwg_mla_proj_uv")
        let projScalarSub2 = try hal.pipeline("qwg_mla_proj_uv_sub2")
        let projMMA = try hal.pipeline("qwg_mla_proj_uv_mma")
        let projMMASub2 = try hal.pipeline("qwg_mla_proj_uv_mma_sub2")

        let qHeads = 8, nope = 256, vDim = 256, rank = 512, qFullStride = qHeads * nope + 64
        let totalRows = qHeads * (nope + vDim)
        let specs: [(isIncumbent: UInt32, bits: Int, group: Int)] = [
            (2, 4, 32), (2, 3, 64), (2, 2, 64), (2, 15, 64), (2, 1, 64),
            (0, 4, 32), (0, 3, 64), (0, 8, 64),
        ]

        var seed: UInt32 = 0xCAFE_BABE
        for spec in specs {
            let isSub2 = spec.isIncumbent == 2 && (spec.bits == 1 || spec.bits == 15)
            let (bankData, wRows) = spec.isIncumbent == 2
                ? makeSHVQBank(bits: spec.bits, group: spec.group, n: totalRows, k: rank)
                : makeAQBank(bits: spec.bits, group: spec.group, n: totalRows, k: rank)
            let wBuf = makeDataBuffer(bankData, device: hal.device)

            for rows in [1, 2, 5, 16, 37] {
                let qVals: [UInt16] = (0..<(rows * qFullStride)).map { _ in
                    seed = seed &* 1664525 &+ 1013904223
                    return QWQuant.f32tof16Bits(Float(Int32(truncatingIfNeeded: seed >> 16) - 32768) / 65536.0)
                }
                let uVals: [UInt16] = (0..<(rows * qHeads * rank)).map { _ in
                    seed = seed &* 1664525 &+ 1013904223
                    return QWQuant.f32tof16Bits(Float(Int32(truncatingIfNeeded: seed >> 16) - 32768) / 65536.0)
                }
                let qBuf = makeBuffer(qVals, device: hal.device)
                let uBuf = makeBuffer(uVals, device: hal.device)
                var params: [UInt32] = [UInt32(rows), UInt32(qHeads), UInt32(nope), UInt32(vDim), UInt32(rank),
                                        UInt32(qFullStride), UInt32(spec.bits), UInt32(spec.group), spec.isIncumbent]

                let mmaModes = rows >= 16 ? [false, true] : [false]
                var absOuts = [[UInt16]](), uvOuts = [[UInt16]]()
                for useMMA in mmaModes {
                    let aOut = hal.device.makeBuffer(length: rows * qHeads * rank * 2, options: .storageModeShared)!
                    let pOut = hal.device.makeBuffer(length: rows * qHeads * vDim * 2, options: .storageModeShared)!
                    guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
                    enc.setComputePipelineState(useMMA ? (isSub2 ? absorbMMASub2 : absorbMMA)
                                                       : (isSub2 ? absorbScalarSub2 : absorbScalar))
                    enc.setBuffer(qBuf, offset: 0, index: 0)
                    enc.setBuffer(wBuf, offset: 0, index: 1)
                    enc.setBuffer(aOut, offset: 0, index: 2)
                    enc.setBytes(&params, length: params.count * 4, index: 3)
                    if useMMA {
                        enc.dispatchThreadgroups(MTLSize(width: qHeads, height: (rows + 31) / 32, depth: rank / 128),
                                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                    } else {
                        enc.dispatchThreadgroups(MTLSize(width: qHeads, height: (rows + 7) / 8, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
                    }
                    enc.setComputePipelineState(useMMA ? (isSub2 ? projMMASub2 : projMMA)
                                                       : (isSub2 ? projScalarSub2 : projScalar))
                    enc.setBuffer(uBuf, offset: 0, index: 0)
                    enc.setBuffer(wBuf, offset: 0, index: 1)
                    enc.setBuffer(pOut, offset: 0, index: 2)
                    enc.setBytes(&params, length: params.count * 4, index: 3)
                    if useMMA {
                        enc.dispatchThreadgroups(MTLSize(width: qHeads, height: (rows + 31) / 32, depth: vDim / 128),
                                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                    } else {
                        enc.dispatchThreadgroups(MTLSize(width: qHeads, height: (rows + 7) / 8, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    }
                    hal.end(cb, enc)
                    cb.waitUntilCompleted()
                    XCTAssertNil(cb.error)
                    let aPtr = aOut.contents().bindMemory(to: UInt16.self, capacity: rows * qHeads * rank)
                    let pPtr = pOut.contents().bindMemory(to: UInt16.self, capacity: rows * qHeads * vDim)
                    absOuts.append(Array(UnsafeBufferPointer(start: aPtr, count: rows * qHeads * rank)))
                    uvOuts.append(Array(UnsafeBufferPointer(start: pPtr, count: rows * qHeads * vDim)))
                }

                for t in [0, rows - 1] {
                    for h in [0, qHeads - 1] {
                        let headBase = h * (nope + vDim)
                        for c in stride(from: 0, to: rank, by: 31) {
                            var ref: Float = 0
                            for r in 0..<nope {
                                ref += QWQuant.f16tof32Bits(qVals[t * qFullStride + h * nope + r]) * wRows[headBase + r][c]
                            }
                            for out in absOuts {
                                let got = QWQuant.f16tof32Bits(out[(t * qHeads + h) * rank + c])
                                XCTAssertLessThan(abs(got - ref), 0.015, "absorb spec=\(spec) rows=\(rows)")
                            }
                        }
                        var uvVec = [Float](repeating: 0, count: vDim)
                        let uBase = (t * qHeads + h) * rank
                        for r in 0..<vDim {
                            var s: Float = 0
                            let wRow = wRows[headBase + nope + r]
                            for c in 0..<rank {
                                s += QWQuant.f16tof32Bits(uVals[uBase + c]) * wRow[c]
                            }
                            uvVec[r] = s
                        }
                        for b in stride(from: 0, to: vDim, by: 32) { fwht32(&uvVec, base: b) }
                        for r in stride(from: 0, to: vDim, by: 17) {
                            for out in uvOuts {
                                let got = QWQuant.f16tof32Bits(out[t * qHeads * vDim + h * vDim + r])
                                XCTAssertLessThan(abs(got - uvVec[r]), 0.015, "projUV spec=\(spec) rows=\(rows)")
                            }
                        }
                    }
                }
                if absOuts.count == 2 {
                    var maxAbsErr: Float = 0, maxAbsRef: Float = 0
                    for i in 0..<absOuts[0].count {
                        let a = QWQuant.f16tof32Bits(absOuts[0][i]), b = QWQuant.f16tof32Bits(absOuts[1][i])
                        maxAbsErr = max(maxAbsErr, abs(a - b)); maxAbsRef = max(maxAbsRef, abs(a))
                    }
                    XCTAssertLessThan(maxAbsErr, 4e-3 * max(maxAbsRef, 1.0), "absorb MMA spec=\(spec) rows=\(rows)")
                    var maxUvErr: Float = 0, maxUvRef: Float = 0
                    for i in 0..<uvOuts[0].count {
                        let a = QWQuant.f16tof32Bits(uvOuts[0][i]), b = QWQuant.f16tof32Bits(uvOuts[1][i])
                        maxUvErr = max(maxUvErr, abs(a - b)); maxUvRef = max(maxUvRef, abs(a))
                    }
                    XCTAssertLessThan(maxUvErr, 4e-3 * max(maxUvRef, 1.0), "projUV MMA spec=\(spec) rows=\(rows)")
                }
            }
        }
    }

    func testOperatorEndToEndSHVQ4AndQ4KVMatchesScalarFallback() throws {
        let geo = try QWMLAAttnOperatorTests.mlaGeometry()
        let hal = try QWHal(geometry: geo)
        let pageBytes = QWKVCache.pageBytes(mode: .q4, geometry: geo)
        let tables = QWKVCache.tableSlotBytes * 2 * 1 * QWKVCache.maxSeqs
        let kvRows = geo.qHeads * (geo.qkNopeHeadDim + geo.vHeadDim)
        let (bankData, _) = makeSHVQBank(bits: 4, group: 32, n: kvRows, k: geo.kvLoraRank)
        let wBuf = makeDataBuffer(bankData, device: hal.device)
        let wTV = QWTensorValue(region: QWMRegion(buffer: wBuf, base: 0, length: wBuf.length),
                                offset: 0, shape: QWTensorShape([(bankData.count + 1) / 2], .f16))
        let kvBView = try XCTUnwrap(QWWeightView(blocks: wTV, n: kvRows, k: geo.kvLoraRank,
                                                 bits: 4, groupSize: 32, isRotated: true, isSHVQ: true))

        func f16Tensor(_ vals: [Float]) -> QWTensorValue {
            let b = makeBuffer(vals.map { QWQuant.f32tof16Bits($0) }, device: hal.device)
            return QWTensorValue(region: QWMRegion(buffer: b, base: 0, length: b.length),
                                 offset: 0, shape: QWTensorShape([vals.count], .f16))
        }

        let maxRows = 24
        let ln1 = f16Tensor((0..<(maxRows * geo.hidden)).map { sin(Float($0 % 199) * 0.03) * 0.2 })
        let qANormW = f16Tensor([Float](repeating: 1.0, count: geo.qLoraRank))
        let kvANormW = f16Tensor([Float](repeating: 1.0, count: geo.kvLoraRank))
        let kNormW = f16Tensor([Float](repeating: 1.0, count: geo.indexerHeadDim))
        let kNormB = f16Tensor([Float](repeating: 0.0, count: geo.indexerHeadDim))
        let wGateW = f16Tensor((0..<((geo.indexerHeads + 4) * geo.hidden)).map { sin(Float($0 % 97) * 0.02) * 0.05 })
        let ape = f16Tensor((0..<(4 * geo.indexerHeadDim)).map { cos(Float($0) * 0.05) * 0.1 })
        let mlaW = QWMLAAttnOperator.LayerWeights(qANormW: qANormW, kvANormW: kvANormW, kNormW: kNormW,
                                                  kNormB: kNormB, wGateWeights: wGateW, ape: ape,
                                                  qBProj: kvBView, kvBProj: kvBView)

        func runPipeline(mmaEnabled: Bool) throws -> ([UInt16], [UInt16]) {
            let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: pageBytes * 4 + tables, gdnStates: 4096)
            let cache = try QWKVCache(hal: hal, arena: arena, layers: [3], mode: .q4, geometry: geo)
            let seq = try cache.allocate(promptTokens: 64)
            let kvViews = try cache.layerViews(seq, layer: 3)
            let op = try QWMLAAttnOperator(hal: hal, geometry: geo, numAttnLayers: 1, maxContext: 512, maxPrefillChunk: 32)
            op.sparseAttnMMAEnabled = mmaEnabled

            let prefillRows = 20
            let decodeRows = 2
            func seedInputs(rows: Int, salt: Int) {
                let latPtr = (op.region.buffer.contents() + op.mlaLatentScratch.offset)
                    .bindMemory(to: UInt16.self, capacity: rows * op.latentStride)
                for i in 0..<(rows * op.latentStride) {
                    latPtr[i] = QWQuant.f32tof16Bits(sin(Float((i + salt) % 251) * 0.05) * 0.3)
                }
                let qPtr = (op.region.buffer.contents() + op.mlaQFullScratch.offset)
                    .bindMemory(to: UInt16.self, capacity: rows * op.qFullStride)
                for i in 0..<(rows * op.qFullStride) {
                    qPtr[i] = QWQuant.f32tof16Bits(cos(Float((i + salt) % 257) * 0.04) * 0.25)
                }
            }

            let prefillOut = f16Tensor([Float](repeating: 0, count: prefillRows * geo.qHeads * geo.vHeadDim))
            seedInputs(rows: prefillRows, salt: 0)
            guard let (cb1, enc1) = hal.begin() else { XCTFail("hal.begin"); return ([], []) }
            op.encodeLatentPrepAndPool(enc1, ln1: ln1, mlaW: mlaW, kv: kvViews, fullAttnLayer: 0, pos0: 0, rows: prefillRows)
            op.encodeScoreSelectAbsorbAttendAndProjUV(enc1, mlaW: mlaW, kv: kvViews, attOut: prefillOut,
                                                      fullAttnLayer: 0, pos0: 0, rows: prefillRows)
            hal.end(cb1, enc1)
            cb1.waitUntilCompleted()
            XCTAssertNil(cb1.error)

            let decodeOut = f16Tensor([Float](repeating: 0, count: decodeRows * geo.qHeads * geo.vHeadDim))
            seedInputs(rows: decodeRows, salt: 997)
            guard let (cb2, enc2) = hal.begin() else { XCTFail("hal.begin"); return ([], []) }
            op.encodeLatentPrepAndPool(enc2, ln1: ln1, mlaW: mlaW, kv: kvViews, fullAttnLayer: 0, pos0: prefillRows, rows: decodeRows)
            op.encodeScoreSelectAbsorbAttendAndProjUV(enc2, mlaW: mlaW, kv: kvViews, attOut: decodeOut,
                                                      fullAttnLayer: 0, pos0: prefillRows, rows: decodeRows)
            hal.end(cb2, enc2)
            cb2.waitUntilCompleted()
            XCTAssertNil(cb2.error)

            let pCount = prefillRows * geo.qHeads * geo.vHeadDim
            let dCount = decodeRows * geo.qHeads * geo.vHeadDim
            let pPtr = prefillOut.region.buffer.contents().bindMemory(to: UInt16.self, capacity: pCount)
            let dPtr = decodeOut.region.buffer.contents().bindMemory(to: UInt16.self, capacity: dCount)
            return (Array(UnsafeBufferPointer(start: pPtr, count: pCount)),
                    Array(UnsafeBufferPointer(start: dPtr, count: dCount)))
        }

        let (refPrefill, refDecode) = try runPipeline(mmaEnabled: false)
        let (optPrefill, optDecode) = try runPipeline(mmaEnabled: true)
        for (label, pair) in [("prefill", (refPrefill, optPrefill)), ("decode", (refDecode, optDecode))] {
            var maxErr: Float = 0, maxRef: Float = 0
            for i in 0..<pair.0.count {
                let a = QWQuant.f16tof32Bits(pair.0[i]), b = QWQuant.f16tof32Bits(pair.1[i])
                XCTAssertTrue(b.isFinite)
                maxErr = max(maxErr, abs(a - b))
                maxRef = max(maxRef, abs(a))
            }
            XCTAssertGreaterThan(maxRef, 0.02, label)
            XCTAssertLessThan(maxErr, 0.015 * max(maxRef, 1.0), label)
        }
    }
}
