// QWMLAIndexerSplitTests.swift — GPU parity of the split MLA indexer vs the fused kernel.

import XCTest
import Metal
@testable import QWAttn
import QWCore
import QWHAL
import QWQuant

/// The split indexer (pool-parallel MMA scoring + per-row select) must reproduce the
/// fused score+select kernel: same scores up to fp32 accumulation order, same top-K set.
final class QWMLAIndexerSplitTests: XCTestCase {
    func testSplitScoreSelectMatchesFused() throws {
        let geo = try QWMLAAttnOperatorTests.mlaGeometry()
        let hal = try QWHal(geometry: geo)
        let fused = try hal.pipeline("qwg_mla_indexer_score_select")
        let score = try hal.pipeline("qwg_mla_indexer_score_mma")
        let select = try hal.pipeline("qwg_mla_indexer_select")
        let heads = 32, dim = 128, qChannels = 64, qStride = qChannels + heads * dim
        let maxBlocks = 1024, topK = 64
        var rng = SystemRandomNumberGenerator()
        func rnd(_ n: Int, _ s: Float) -> [UInt16] {
            (0..<n).map { _ in QWQuant.f32tof16Bits(Float.random(in: -s...s, using: &rng)) }
        }
        func buf<T: BitwiseCopyable>(_ a: [T]) -> any MTLBuffer {
            QWMLAAttnOperatorTests.makeBuffer(a, device: hal.device)
        }
        // (pos0, T): identity rows below top-K, a row crossing it, and a long context with a ragged pool tile.
        for (pos0, rows) in [(200, 70), (3000, 5), (4011, 3)] {
            let q = buf(rnd(rows * qStride, 0.5))
            let w = buf((0..<(rows * heads)).map { _ in Float.random(in: 0...1, using: &rng) })
            let keys = buf(rnd(maxBlocks * dim, 1.0))
            var params: [UInt32] = [UInt32(rows), UInt32(pos0), UInt32(qChannels), UInt32(heads), UInt32(dim),
                                    UInt32(topK), UInt32(topK), UInt32(maxBlocks)]
            var results: [(scores: [Float], lists: [[Int32]])] = []
            for split in [false, true] {
                let scores = buf([Float](repeating: -1, count: rows * maxBlocks))
                let list = buf([Int32](repeating: -1, count: rows * topK))
                let count = buf([Int32](repeating: -1, count: rows))
                guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
                for (i, b) in [q, w, keys, scores, list, count].enumerated() { enc.setBuffer(b, offset: 0, index: i) }
                enc.setBytes(&params, length: params.count * 4, index: 6)
                let tg = MTLSize(width: 256, height: 1, depth: 1)
                if split {
                    enc.setComputePipelineState(score)
                    let maxPools = (pos0 + rows) / 4
                    enc.dispatchThreadgroups(MTLSize(width: (maxPools + 63) / 64, height: rows, depth: 1), threadsPerThreadgroup: tg)
                    enc.setComputePipelineState(select)
                } else {
                    enc.setComputePipelineState(fused)
                }
                enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: tg)
                hal.end(cb, enc)
                cb.waitUntilCompleted()
                XCTAssertNil(cb.error)
                let sp = scores.contents().bindMemory(to: Float.self, capacity: rows * maxBlocks)
                let lp = list.contents().bindMemory(to: Int32.self, capacity: rows * topK)
                let cp = count.contents().bindMemory(to: Int32.self, capacity: rows)
                var lists: [[Int32]] = []
                for t in 0..<rows {
                    XCTAssertEqual(Int(cp[t]), min(topK, (pos0 + t + 1) / 4), "count t=\(t)")
                    lists.append(Array(UnsafeBufferPointer(start: lp + t * topK, count: Int(cp[t]))))
                }
                results.append((Array(UnsafeBufferPointer(start: sp, count: rows * maxBlocks)), lists))
            }
            for t in 0..<rows {
                let nPools = (pos0 + t + 1) / 4
                if nPools <= topK {
                    XCTAssertEqual(results[0].lists[t], results[1].lists[t], "identity t=\(t)")
                    continue
                }
                var worst: Float = 0, peak: Float = 0
                for i in 0..<nPools {
                    let a = results[0].scores[t * maxBlocks + i], b = results[1].scores[t * maxBlocks + i]
                    worst = max(worst, abs(a - b))
                    peak = max(peak, abs(a))
                }
                XCTAssertGreaterThan(peak, 0.1, "pos0=\(pos0) t=\(t)")
                XCTAssertLessThan(worst, 1e-3 * peak, "pos0=\(pos0) t=\(t)")
                let overlap = Set(results[0].lists[t]).intersection(results[1].lists[t]).count
                XCTAssertGreaterThanOrEqual(overlap, topK - 2, "pos0=\(pos0) t=\(t)")
            }
        }
    }
}
