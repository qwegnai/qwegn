import XCTest
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWGEMM

/// The NAX/MPP grouped gate/up (f16 operands, fp32 accumulate; the default path, QW_DISABLE_NAX=1 selects f32)
/// against the per-pair f32 routed path. Tolerance is the golden bound for the f16 path.
final class FNMetalMoENaxTests: XCTestCase {
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

    func testNaxVsPerPair() throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "NAX MoE requires Metal 4 tensor hardware")
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
        // Activation scale keeps the f16 output in range.
        for i in 0..<xs.count { seed = seed &* 1664525 &+ 1013904223; xs[i] = Float(Int32(truncatingIfNeeded: seed)) * 1e-9 }
        arena.copyIn(x, from: xs.withUnsafeBufferPointer { Data(buffer: $0) })

        let ids = try act.allocate(QWTensorShape([pairs], .i32))
        var idv = [Int32](repeating: 0, count: pairs)
        var rs: UInt32 = 99
        for t in 0..<tokens { for r in 0..<routes { rs = rs &* 1664525 &+ 1013904223; idv[t * routes + r] = Int32((rs >> 8) % UInt32(experts)) } }
        arena.copyIn(ids, from: idv.withUnsafeBufferPointer { Data(buffer: $0) })

        let yPair = try act.allocate(QWTensorShape([pairs * rowsPerExpert], .f32))
        let expOff = try act.allocate(QWTensorShape([experts + 1], .i32))
        let sortedPair = try act.allocate(QWTensorShape([pairs], .i32))
        let sortedTok = try act.allocate(QWTensorShape([pairs], .i32))
        let x16 = try act.allocate(QWTensorShape([tokens * k], .f16))
        let xs16 = xs.map { Float16($0) }
        arena.copyIn(x16, from: xs16.withUnsafeBufferPointer { Data(buffer: $0) })
        let tilePlan = QWGemm.MoETilePlan(experts: experts, pairs: pairs, gateUpRows: rowsPerExpert, downRows: 64)
        let tileE = try act.allocate(QWTensorShape([pairs], .i32))
        let tileR = try act.allocate(QWTensorShape([pairs], .i32))
        let tileCount = try act.allocate(QWTensorShape([tilePlan.scratchInts], .i32))
        let gu16 = try act.allocate(QWTensorShape([pairs * rowsPerExpert], .f16))

        func step(_ f: (any MTLComputeCommandEncoder) throws -> Void) throws {
            guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
            try f(enc)
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error, "step failed: \(String(describing: cb.error))")
        }

        try step { enc in try QWGemm.encodeMoEGroupPairsF32(enc, hal: Self.hal, ids: ids, expertOff: expOff,
                                                           sortedPairIdx: sortedPair, sortedTok: sortedTok,
                                                           experts: experts, totalPairs: pairs, routes: routes,
                                                           tileE: tileE, tileR: tileR, tileCount: tileCount,
                                                           tilePlan: tilePlan) }
        try step { enc in try QWGemm.encodeMoEGateUpF32R1(enc, hal: Self.hal, x: x, bank: view, ids: ids, out: yPair,
                                                         rowsPerExpert: rowsPerExpert, routes: routes, tokens: tokens) }
        try step { enc in try QWGemm.encodeMoEGateUpNax(enc, hal: Self.hal, xe: x16, bank: view, expertOff: expOff, tileE: tileE,
                                                       tileR: tileR, tileCount: tileCount, gu16: gu16,
                                                       rowsPerExpert: rowsPerExpert, tilePlan: tilePlan, sortedTok: sortedTok) }

        let a = Array(UnsafeBufferPointer(start: arena.cpuPointer(of: yPair).bindMemory(to: Float.self, capacity: pairs * rowsPerExpert),
                                          count: pairs * rowsPerExpert))
        let sp = Array(UnsafeBufferPointer(start: arena.cpuPointer(of: sortedPair).bindMemory(to: Int32.self, capacity: pairs),
                                           count: pairs))
        let guPtr = arena.cpuPointer(of: gu16).bindMemory(to: Float16.self, capacity: pairs * rowsPerExpert)
        var b = [Float](repeating: 0, count: pairs * rowsPerExpert)
        for s in 0..<pairs {
            let p = Int(sp[s])
            for c in 0..<rowsPerExpert {
                b[p * rowsPerExpert + c] = Float(guPtr[s * rowsPerExpert + c])
            }
        }
        var sumSq = 0.0, refSq = 0.0, maxAbs = 0.0
        for i in 0..<a.count {
            let e = abs(Double(a[i]) - Double(b[i]))
            if e > maxAbs { maxAbs = e }
            sumSq += e * e
            refSq += Double(a[i]) * Double(a[i])
        }
        let rel = (sumSq / max(refSq, 1e-30)).squareRoot()
        print(String(format: "FN-MOENAX pairs=%d: rel err %.2e max abs %.2e", pairs, rel, maxAbs))
        XCTAssertLessThan(rel, 5e-3, "NAX gate/up must match the per-pair f32 path within the golden f16 bound")
    }
}
