import XCTest
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWGEMM

/// The expert-sorted grouped MoE (MLX's SwitchGLU shape) against the per-pair routed path: same result, but each
/// expert's weight slab is read once for every pair routed to it instead of once per (token, route) pair.
final class FNMetalMoEGroupTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private func makeBank(_ n: Int, _ k: Int) -> Data {
        var d = Data(capacity: n * (k / 32) * 20)
        var rng: UInt64 = 0x9E3779B97F4A7C15
        func next() -> UInt32 {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17
            return UInt32(truncatingIfNeeded: rng >> 32)
        }
        for _ in 0..<(n * (k / 32)) {
            d.append(Data([UInt8(0x1F), 0x2C, 0, 0]))    // scale 0x2C1F, bias 0
            for _ in 0..<4 { d.append(Data([UInt8(next() & 0xFF), UInt8((next() >> 8) & 0xFF),
                                            UInt8((next() >> 16) & 0xFF), UInt8((next() >> 24) & 0xFF)])) }
        }
        return d
    }

    /// experts=32, routes=4, tokens=64, intermediate=64 -> rowsPerExpert = 128, K = 256.
    func testGroupedVsPerPair() throws {
        let experts = 32, routes = 4, tokens = 64, inter = 64, k = 256
        let rowsPerExpert = 2 * inter
        let n = experts * rowsPerExpert
        let bytes = n * (k / 32) * 20
        let pairs = tokens * routes
        let arena = try QWArena(hal: Self.hal, weights: bytes + (8 << 20),
                                activations: (256 << 20), kv: 0, gdnStates: 0)
        let data = makeBank(n, k)
        let w = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
        arena.copyIn(w, from: data)
        let view = QWWeightView(blocks: w, n: n, k: k, isRotated: true)!

        let act = arena.activationsRegion()
        let x = try act.allocate(QWTensorShape([tokens * k], .f32))
        var seed: UInt32 = 777
        var xs = [Float](repeating: 0, count: tokens * k)
        for i in 0..<xs.count { seed = seed &* 1664525 &+ 1013904223; xs[i] = Float(Int32(truncatingIfNeeded: seed)) * 1e-6 }
        arena.copyIn(x, from: xs.withUnsafeBufferPointer { Data(buffer: $0) })

        // Routing: a skewed but deterministic expert choice per (token, route).
        let ids = try act.allocate(QWTensorShape([pairs], .i32))
        var idv = [Int32](repeating: 0, count: pairs)
        var rs: UInt32 = 99
        for t in 0..<tokens { for r in 0..<routes { rs = rs &* 1664525 &+ 1013904223; idv[t * routes + r] = Int32((rs >> 8) % UInt32(experts)) } }
        arena.copyIn(ids, from: idv.withUnsafeBufferPointer { Data(buffer: $0) })

        let yPair = try act.allocate(QWTensorShape([pairs * rowsPerExpert], .f32))
        let yGroup = try act.allocate(QWTensorShape([pairs * rowsPerExpert], .f32))
        let expOff = try act.allocate(QWTensorShape([experts + 1], .i32))
        let sortedPair = try act.allocate(QWTensorShape([pairs], .i32))
        let sortedTok = try act.allocate(QWTensorShape([pairs], .i32))

        func run(_ grouped: Bool) throws -> Double {
            var best = Double.infinity
            for _ in 0..<3 {
                guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
                let t0 = CACurrentMediaTime()
                if grouped {
                    try QWGemm.encodeMoEGroupPairsF32(enc, hal: Self.hal, ids: ids, expertOff: expOff,
                                                     sortedPairIdx: sortedPair, sortedTok: sortedTok,
                                                     experts: experts, totalPairs: pairs, routes: routes)
                    Self.hal.end(cb, enc)
                    cb.waitUntilCompleted()
                    XCTAssertNil(cb.error, "pair sort failed: \(String(describing: cb.error))")
                    guard let (cb2, enc2) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
                    enc2.setComputePipelineState(try Self.hal.pipeline("qwg_moe_gateup_group_f32"))
                    enc2.bindTensors(view.blocks, x, yGroup, expOff, sortedPair, sortedTok, startIndex: 0)
                    enc2.setUInt32s(rowsPerExpert, view.k, startIndex: 6)
                    enc2.dispatchThreadgroups(MTLSize(width: experts, height: (rowsPerExpert + 31) / 32, depth: 1),
                                              threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    Self.hal.end(cb2, enc2)
                    cb2.waitUntilCompleted()
                    XCTAssertNil(cb2.error, "grouped gate/up failed: \(String(describing: cb2.error))")
                    best = min(best, CACurrentMediaTime() - t0)
                } else {
                    try QWGemm.encodeMoEGateUpF32R1(enc, hal: Self.hal, x: x, bank: view, ids: ids, out: yPair,
                                                   rowsPerExpert: rowsPerExpert, routes: routes, tokens: tokens)
                    Self.hal.end(cb, enc)
                    cb.waitUntilCompleted()
                    XCTAssertNil(cb.error, "per-pair gate/up failed: \(String(describing: cb.error))")
                    best = min(best, CACurrentMediaTime() - t0)
                }
            }
            return best
        }
        let tPair = try run(false)
        let tGroup = try run(true)

        func read(_ t: QWTensorValue) -> [Float] {
            Array(UnsafeBufferPointer(start: arena.cpuPointer(of: t).bindMemory(to: Float.self, capacity: pairs * rowsPerExpert),
                                      count: pairs * rowsPerExpert))
        }
        let a = read(yPair), b = read(yGroup)
        var maxAbs = 0.0, sumSq = 0.0, refSq = 0.0
        for i in 0..<a.count {
            let e = abs(Double(a[i]) - Double(b[i]))
            if e > maxAbs { maxAbs = e }
            sumSq += e * e
            refSq += Double(a[i]) * Double(a[i])
        }
        let rel = (sumSq / max(refSq, 1e-30)).squareRoot()
        // Weight traffic: the per-pair path reads the slab once per (token, route) pair; the grouped path reads it
        // once per expert (one pass over the pairs of that expert).
        let slab = Double(bytes)
        let pairTraffic = slab * Double(pairs)
        let groupTraffic = slab * Double(experts)
        print(String(format: "FN-MOEGROUP pairs=%d: per-pair %.2f ms (%.3f GB) | grouped %.2f ms (%.3f GB) | speedup %.2fx | rel err %.2e max abs %.2e",
                     pairs, tPair * 1000, pairTraffic / 1e9, tGroup * 1000, groupTraffic / 1e9, tPair / tGroup, rel, maxAbs))
        XCTAssertLessThan(rel, 1e-5, "grouped MoE must match the per-pair path")
        XCTAssertLessThan(groupTraffic, pairTraffic, "the grouped path must move less weight traffic")
    }
}
