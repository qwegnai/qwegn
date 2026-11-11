import XCTest
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWGEMM

/// The NAX/MPP grouped down with a hi/lo split activation (fp32 accumulate) against the f32 grouped down it replaces.
final class FNMetalMoEDownNaxTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private func makeBank(_ n: Int, _ k: Int) -> Data {
        var d = Data(capacity: n * (k / 32) * 20)
        var rng: UInt64 = 0x7F4A7C159E3779B9
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

    func testSplitDownVsF32() throws {
        let experts = 32, routes = 4, tokens = 64, hidden = 128, inter = 64
        let pairs = tokens * routes
        let n = experts * hidden
        let bytes = n * (inter / 32) * 20
        let arena = try QWArena(hal: Self.hal, weights: bytes + (8 << 20), activations: (256 << 20), kv: 0, gdnStates: 0)
        let w = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
        arena.copyIn(w, from: makeBank(n, inter))
        let view = QWWeightView(blocks: w, n: n, k: inter, isRotated: true)!
        let act = arena.activationsRegion()
        let a32 = try act.allocate(QWTensorShape([pairs * inter], .f32))
        var seed: UInt32 = 999
        var xs = [Float](repeating: 0, count: pairs * inter)
        for i in 0..<xs.count { seed = seed &* 1664525 &+ 1013904223; xs[i] = Float(Int32(truncatingIfNeeded: seed)) * 1e-9 }
        arena.copyIn(a32, from: xs.withUnsafeBufferPointer { Data(buffer: $0) })
        let ids = try act.allocate(QWTensorShape([pairs], .i32))
        var idv = [Int32](repeating: 0, count: pairs)
        var rs: UInt32 = 5
        for t in 0..<tokens { for r in 0..<routes { rs = rs &* 1664525 &+ 1013904223; idv[t * routes + r] = Int32((rs >> 8) % UInt32(experts)) } }
        arena.copyIn(ids, from: idv.withUnsafeBufferPointer { Data(buffer: $0) })
        let expOff = try act.allocate(QWTensorShape([experts + 1], .i32))
        let sortedPair = try act.allocate(QWTensorShape([pairs], .i32))
        let sortedTok = try act.allocate(QWTensorShape([pairs], .i32))
        let invPerm = try act.allocate(QWTensorShape([pairs], .i32))
        let hi = try act.allocate(QWTensorShape([pairs * inter], .f16))
        let lo = try act.allocate(QWTensorShape([pairs * inter], .f16))
        let tilePlan = QWGemm.MoETilePlan(experts: experts, pairs: pairs, gateUpRows: 64, downRows: hidden)
        let tileE = try act.allocate(QWTensorShape([pairs], .i32))
        let tileR = try act.allocate(QWTensorShape([pairs], .i32))
        let tileCount = try act.allocate(QWTensorShape([tilePlan.scratchInts], .i32))
        let ySorted = try act.allocate(QWTensorShape([pairs * hidden], .f32))
        let wRoute = try act.allocate(QWTensorShape([pairs], .f32))
        let wVals = [Float](repeating: 0.25, count: pairs)
        arena.copyIn(wRoute, from: wVals.withUnsafeBufferPointer { Data(buffer: $0) })
        let shared = try act.allocate(QWTensorShape([tokens * hidden], .f32))
        let gate = try act.allocate(QWTensorShape([tokens], .f32))
        let zeroShared = [Float](repeating: 0, count: tokens * hidden)
        let zeroGate = [Float](repeating: 0, count: tokens)
        arena.copyIn(shared, from: zeroShared.withUnsafeBufferPointer { Data(buffer: $0) })
        arena.copyIn(gate, from: zeroGate.withUnsafeBufferPointer { Data(buffer: $0) })
        let yRef = try act.allocate(QWTensorShape([tokens * hidden], .f32))
        let yNax = try act.allocate(QWTensorShape([tokens * hidden], .f32))

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
                                                           invPerm: invPerm, tileE: tileE, tileR: tileR,
                                                           tileCount: tileCount, tilePlan: tilePlan) }
        let sp = Array(UnsafeBufferPointer(start: arena.cpuPointer(of: sortedPair).bindMemory(to: Int32.self, capacity: pairs),
                                           count: pairs))
        var hiVals = [Float16](repeating: 0, count: pairs * inter)
        var loVals = [Float16](repeating: 0, count: pairs * inter)
        for s in 0..<pairs {
            let p = Int(sp[s])
            for c in 0..<inter {
                let v = xs[p * inter + c]
                let h = Float16(v)
                hiVals[s * inter + c] = h
                loVals[s * inter + c] = Float16(v - Float(h))
            }
        }
        arena.copyIn(hi, from: hiVals.withUnsafeBufferPointer { Data(buffer: $0) })
        arena.copyIn(lo, from: loVals.withUnsafeBufferPointer { Data(buffer: $0) })
        try step { enc in try QWGemm.encodeMoEDownSharedCombineF32(enc, hal: Self.hal, x: a32, bank: view,
                                                                  rowsPerRoute: hidden, routes: routes,
                                                                  weights: wRoute, shared: shared, gate: gate,
                                                                  out: yRef, ids: ids, tokens: tokens) }
        try step { enc in try QWGemm.encodeMoENaxSplitF32(enc, hal: Self.hal, xHi: hi, xLo: lo, bank: view, expertOff: expOff,
                                                         tileE: tileE, tileR: tileR, tileCount: tileCount, y32: ySorted,
                                                         rowsPerExpert: hidden, tilePlan: tilePlan) }
        try step { enc in try QWGemm.encodeMoEDownReduceCombineF32(enc, hal: Self.hal, dnAcc: ySorted, invPerm: invPerm,
                                                                  weights: wRoute, shared: shared, gate: gate,
                                                                  out: yNax, hidden: hidden, routes: routes, tokens: tokens) }

        func read(_ t: QWTensorValue) -> [Float] {
            Array(UnsafeBufferPointer(start: arena.cpuPointer(of: t).bindMemory(to: Float.self, capacity: tokens * hidden),
                                      count: tokens * hidden))
        }
        let a = read(yRef), b = read(yNax)
        var sumSq = 0.0, refSq = 0.0, maxAbs = 0.0
        for i in 0..<a.count {
            let e = abs(Double(a[i]) - Double(b[i]))
            if e > maxAbs { maxAbs = e }
            sumSq += e * e; refSq += Double(a[i]) * Double(a[i])
        }
        let rel = (sumSq / max(refSq, 1e-30)).squareRoot()
        print(String(format: "FN-MOEDOWN pairs=%d: rel err %.2e max abs %.2e", pairs, rel, maxAbs))
        XCTAssertLessThan(rel, 5e-3, "split NAX down must match the f32 grouped down within the golden bound")
    }
}
