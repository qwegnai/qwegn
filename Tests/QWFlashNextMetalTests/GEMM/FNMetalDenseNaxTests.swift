import XCTest
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWFlashNext

/// The dense row-batched NAX/MPP projections (Single-Pass f16 default and Hi/Lo Split f16 greedy fallback)
/// against the f32 batched GEMM they replace. Prefill-sized rows (>= 64).
final class FNMetalDenseNaxTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private func makeBank(_ n: Int, _ k: Int) -> Data {
        var d = Data(capacity: n * (k / 32) * 20)
        var rng: UInt64 = 0x2545F4914F6CDD1D
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

    func testDenseNaxVsF32() throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "NAX GEMM requires Metal 4 tensor hardware")
        let rows = 128, k = 2560, n = 1280
        let bytes = n * (k / 32) * 20
        let arena = try QWArena(hal: Self.hal, weights: bytes + (8 << 20), activations: (256 << 20), kv: 0, gdnStates: 0)
        let w = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
        arena.copyIn(w, from: makeBank(n, k))
        let view = QWWeightView(blocks: w, n: n, k: k, isRotated: true)!
        let act = arena.activationsRegion()
        let x = try act.allocate(QWTensorShape([rows * k], .f32))
        var seed: UInt32 = 4242
        var xs = [Float](repeating: 0, count: rows * k)
        for i in 0..<xs.count { seed = seed &* 1664525 &+ 1013904223; xs[i] = Float(Int32(truncatingIfNeeded: seed)) * 1e-9 }
        arena.copyIn(x, from: xs.withUnsafeBufferPointer { Data(buffer: $0) })
        let x16 = try act.allocate(QWTensorShape([rows * k], .f16))
        let y16 = try act.allocate(QWTensorShape([rows * k], .f16))
        let yRef = try act.allocate(QWTensorShape([rows * n], .f32))
        let yNaxSingle = try act.allocate(QWTensorShape([rows * n], .f32))

        func step(_ f: (any MTLComputeCommandEncoder) throws -> Void) throws {
            guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
            try f(enc)
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error, "step failed: \(String(describing: cb.error))")
        }
        try step { enc in try QWGemm.encodeHQ4F32BM(enc, hal: Self.hal, weight: view, x: x, y: yRef, rows: rows) }
        try step { enc in try QWGemm.encodeCast(enc, hal: Self.hal, x32: x, y16: x16, count: rows * k) }
        try step { enc in try QWGemm.encodeNaxF32(enc, hal: Self.hal, weight: view, x16: x16, y16: y16, y32: yNaxSingle, rows: rows) }

        func read(_ t: QWTensorValue) -> [Float] {
            Array(UnsafeBufferPointer(start: arena.cpuPointer(of: t).bindMemory(to: Float.self, capacity: rows * n), count: rows * n))
        }
        let a = read(yRef), bSingle = read(yNaxSingle)
        func relErr(_ u: [Float], _ v: [Float]) -> Double {
            var sumSq = 0.0, refSq = 0.0
            for i in 0..<u.count { let e = Double(u[i]) - Double(v[i]); sumSq += e * e; refSq += Double(u[i]) * Double(u[i]) }
            return (sumSq / max(refSq, 1e-30)).squareRoot()
        }
        let relSingle = relErr(a, bSingle)
        print(String(format: "FN-DENSENAX rows=%d N=%d K=%d: single rel err %.2e", rows, n, k, relSingle))
        XCTAssertLessThan(relSingle, 5e-3, "single-pass dense NAX must match the f32 batched GEMM within the golden f16 bound")

        let stochastic = QWGemm.configureDenseNax(temperature: 0.7, topP: 0.9, topK: 40)
        XCTAssertEqual(stochastic.mode, .singlePass)
        XCTAssertTrue(stochastic.logMessage.contains("Single-Pass Dense NAX"))

        let greedyTemp = QWGemm.configureDenseNax(temperature: 0.0, topP: 0.9, topK: 40)
        XCTAssertEqual(greedyTemp.mode, .singlePass)
        XCTAssertTrue(greedyTemp.logMessage.contains("Single-Pass Dense NAX"))

        let greedyTopK = QWGemm.configureDenseNax(temperature: 0.7, topP: 0.9, topK: 1)
        XCTAssertEqual(greedyTopK.mode, .singlePass)
        XCTAssertTrue(greedyTopK.logMessage.contains("Single-Pass Dense NAX"))
    }
}
