// QWHCANAXAttnTests.swift — Apple10+ NAX MQA HCA attention (`qwg_hca_attn_nax`)
// parity against `qwg_hca_attn_mma` across unsplit prefill, Split-K + combine, and sparse LSA.

import Foundation
import Metal
import XCTest
@testable import QWAttn
import QWAlloc
import QWCore
import QWHAL

final class QWHCANAXAttnTests: XCTestCase {
    private static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private func makeData(count: Int, seed: UInt64, scale: Float = 0.25) -> [Float] {
        var s = seed
        return (0..<count).map { _ in
            s = s &* 6364136223846793005 &+ 1442695040888963407
            let v = Float(Int32(truncatingIfNeeded: s >> 33)) / Float(Int32.max)
            return v * scale
        }
    }

    private func runCase(
        rows T: Int, pos0: Int, qHeads: Int, compressRatio: Int, slidingWindow: Int,
        nSplit: Int, hasSink: Bool, useBlockList: Bool, compBudget: Int, maxComp: Int
    ) throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "qwg_hca_attn_nax requires Apple10+ NAX")
        let psoMMA = try Self.hal.pipeline("qwg_hca_attn_mma")
        let psoNAX = try Self.hal.pipeline("qwg_hca_attn_nax")
        let psoCombine = try Self.hal.pipeline("qwg_hca_attn_combine")

        let headDim = 512
        let ringCap = 256
        let qUpStride = qHeads * headDim + 64
        let listStride = max(8, compBudget + 4)
        let arena = try QWArena(hal: Self.hal, weights: 1 << 20, activations: 64 << 20, kv: 0, gdnStates: 0)
        let ar = arena.activationsRegion()

        let qUp = try ar.allocate(QWTensorShape([T * qUpStride], .f16))
        let compKV = try ar.allocate(QWTensorShape([maxComp * headDim], .f16))
        let swaKV = try ar.allocate(QWTensorShape([ringCap * headDim], .f16))
        let bList = try ar.allocate(QWTensorShape([T * listStride], .i32))
        let bCount = try ar.allocate(QWTensorShape([T], .i32))
        let sink = try ar.allocate(QWTensorShape([qHeads], .f32))
        let outMMA = try ar.allocate(QWTensorShape([T * qHeads * headDim], .f16))
        let outNAX = try ar.allocate(QWTensorShape([T * qHeads * headDim], .f16))
        let partO = try ar.allocate(QWTensorShape([T * max(1, nSplit) * qHeads * headDim], .f16))
        let partML = try ar.allocate(QWTensorShape([T * max(1, nSplit) * qHeads * 2], .f32))

        arena.write(qUp, from: makeData(count: T * qUpStride, seed: 0x1111, scale: 0.35).map { Float16($0) })
        arena.write(compKV, from: makeData(count: maxComp * headDim, seed: 0x2222, scale: 0.30).map { Float16($0) })
        arena.write(swaKV, from: makeData(count: ringCap * headDim, seed: 0x3333, scale: 0.30).map { Float16($0) })
        arena.write(sink, from: (0..<qHeads).map { h in h % 3 == 0 ? Float(-1e30) : (Float(h % 5) - 2.0) * 0.6 })

        var counts = [Int32](repeating: Int32(compBudget), count: T)
        var lists = [Int32](repeating: 0, count: T * listStride)
        for t in 0..<T {
            let totalComp = (pos0 + t + 1) / compressRatio
            counts[t] = Int32(min(compBudget, totalComp))
            for k in 0..<Int(counts[t]) { lists[t * listStride + k] = Int32((k * 3 + t) % max(1, totalComp)) }
        }
        arena.write(bCount, from: counts)
        arena.write(bList, from: lists)

        var params = QWHCAAttnOperator.AttnParams(
            T: UInt32(T), pos0: UInt32(pos0), qHeads: UInt32(qHeads), headDim: UInt32(headDim),
            qUpStride: UInt32(qUpStride), compressRatio: UInt32(compressRatio),
            slidingWindow: UInt32(slidingWindow), ringMask: UInt32(ringCap - 1),
            compBudget: UInt32(compBudget), listStride: UInt32(listStride),
            useBlockList: useBlockList ? 1 : 0, hasSink: hasSink ? 1 : 0
        )

        func runKernel(_ pso: any MTLComputePipelineState, hg: Int, threads: Int, dst: QWTensorValue) throws {
            guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
            enc.setComputePipelineState(pso)
            qUp.bind(enc, index: 0)
            compKV.bind(enc, index: 1)
            swaKV.bind(enc, index: 2)
            bList.bind(enc, index: 3)
            bCount.bind(enc, index: 4)
            sink.bind(enc, index: 5)
            (nSplit > 1 ? partO : dst).bind(enc, index: 6)
            partML.bind(enc, index: 7)
            enc.setBytes(&params, length: MemoryLayout<QWHCAAttnOperator.AttnParams>.stride, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: T, height: (qHeads + hg - 1) / hg, depth: nSplit),
                                     threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
            if nSplit > 1 {
                enc.memoryBarrier(scope: .buffers)
                var nSplitU = UInt32(nSplit)
                enc.setComputePipelineState(psoCombine)
                partO.bind(enc, index: 0)
                partML.bind(enc, index: 1)
                sink.bind(enc, index: 2)
                dst.bind(enc, index: 3)
                enc.setBytes(&params, length: MemoryLayout<QWHCAAttnOperator.AttnParams>.stride, index: 4)
                enc.setBytes(&nSplitU, length: 4, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: T * qHeads, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
        }

        try runKernel(psoMMA, hg: 16, threads: 128, dst: outMMA)
        try runKernel(psoNAX, hg: 32, threads: 256, dst: outNAX)

        let ref = arena.read(outMMA, as: Float16.self, count: T * qHeads * headDim)
        let got = arena.read(outNAX, as: Float16.self, count: T * qHeads * headDim)
        var maxDiff: Float = 0
        for i in 0..<ref.count {
            maxDiff = max(maxDiff, abs(Float(got[i]) - Float(ref[i])))
        }
        XCTAssertLessThan(maxDiff, 2e-3,
                          "HCA NAX vs MMA diff=\(maxDiff) T=\(T) nSplit=\(nSplit) sparse=\(useBlockList)")
    }

    func testHCAATTNNAXMatchesMMAUnsplitSplitAndSparse() throws {
        // Unsplit prefill (non-multiple-of-32 slots + learned sink).
        try runCase(rows: 5, pos0: 260, qHeads: 32, compressRatio: 128, slidingWindow: 128,
                    nSplit: 1, hasSink: true, useBlockList: false, compBudget: 16, maxComp: 8)
        // Split-K decode + combine (32 heads + sink).
        try runCase(rows: 2, pos0: 600, qHeads: 32, compressRatio: 128, slidingWindow: 128,
                    nSplit: 4, hasSink: true, useBlockList: false, compBudget: 16, maxComp: 16)
        // Sparse LSA blockList + odd 16-head remainder coverage.
        try runCase(rows: 3, pos0: 900, qHeads: 16, compressRatio: 128, slidingWindow: 128,
                    nSplit: 4, hasSink: true, useBlockList: true, compBudget: 3, maxComp: 16)
    }
}
