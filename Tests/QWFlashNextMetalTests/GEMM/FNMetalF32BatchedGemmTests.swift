import XCTest
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWGEMM

/// The weight-stationary batched HQ4 GEMM (64 token rows x 32 output rows per threadgroup) against the
/// per-row GEMV: same f32 result, but each weight slab is read once per 64 rows instead of once per row.
final class FNMetalF32BatchedGemmTests: XCTestCase {
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

    func testBatchedVsPerRow() throws {
        let n = 10240, k = 2560            // the GDN qkv projection shape
        let bytes = n * (k / 32) * 20
        let arena = try QWArena(hal: Self.hal, weights: bytes + (8 << 20), activations: 512 << 20, kv: 0, gdnStates: 0)
        let data = makeBank(n, k)
        let w = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
        arena.copyIn(w, from: data)
        let view = QWWeightView(blocks: w, n: n, k: k, isRotated: true)!
        let act = arena.activationsRegion()
        let rows = 512
        let x = try act.allocate(QWTensorShape([rows * k], .f32))
        var seed: UInt32 = 12345
        var xs = [Float](repeating: 0, count: rows * k)
        for i in 0..<xs.count { seed = seed &* 1664525 &+ 1013904223; xs[i] = Float(Int32(truncatingIfNeeded: seed)) * 1e-6 }
        arena.copyIn(x, from: xs.withUnsafeBufferPointer { Data(buffer: $0) })
        let yRow = try act.allocate(QWTensorShape([rows * n], .f32))
        let yBm = try act.allocate(QWTensorShape([rows * n], .f32))

        func run(_ bm: Bool) throws -> Double {
            var best = Double.infinity
            for _ in 0..<3 {
                guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
                let t0 = CACurrentMediaTime()
                if bm {
                    try QWGemm.encodeHQ4F32BM(enc, hal: Self.hal, weight: view, x: x, y: yBm, rows: rows)
                } else {
                    try QWGemm.encodeHQ4F32V4(enc, hal: Self.hal, weight: view, x: x, y: yRow, rows: rows)
                }
                Self.hal.end(cb, enc)
                cb.waitUntilCompleted()
                XCTAssertNil(cb.error, "GPU command failed: \(String(describing: cb.error))")
                best = min(best, CACurrentMediaTime() - t0)
            }
            return best
        }
        let tRow = try run(false)
        let tBm = try run(true)

        func read(_ t: QWTensorValue) -> [Float] {
            Array(UnsafeBufferPointer(start: arena.cpuPointer(of: t).bindMemory(to: Float.self, capacity: rows * n), count: rows * n))
        }
        let a = read(yRow), b = read(yBm)
        var maxAbs = 0.0, sumSq = 0.0, refSq = 0.0
        for i in 0..<a.count { let e = abs(Double(a[i]) - Double(b[i])); if e > maxAbs { maxAbs = e }; sumSq += e * e; refSq += Double(a[i]) * Double(a[i]) }
        let rel = (sumSq / max(refSq, 1e-30)).squareRoot()
        let rowTraffic = Double(bytes) * Double(rows)
        let bmTraffic = Double(bytes) * Double((rows + 63) / 64)
        print(String(format: "FN-BM rows=%d: per-row %.2f ms (%.2f GB weight) | batched %.2f ms (%.2f GB weight) | speedup %.2fx | rel err %.2e max abs %.2e",
                     rows, tRow * 1000, rowTraffic / 1e9, tBm * 1000, bmTraffic / 1e9, tRow / tBm, rel, maxAbs))
        XCTAssertLessThan(rel, 1e-5, "batched GEMM must match the per-row GEMV")
        XCTAssertLessThan(tBm, tRow, "batched GEMM must be faster than the per-row GEMV")
    }
}
