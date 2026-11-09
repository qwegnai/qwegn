import XCTest
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWGEMM

/// Measures the GDN projection kernel (HQ4 weights, f32 activations) against its two operands.
final class FNMetalGDNProjBenchTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private func makeBank(_ n: Int, _ k: Int) -> Data {
        var d = Data(capacity: n * (k / 32) * 20)
        let scale = UInt16(0x211F)
        let word = UInt32(0x11111111)
        for _ in 0..<(n * (k / 32)) {
            d.append(Data([UInt8(scale & 0xFF), UInt8(scale >> 8), 0, 0]))
            for _ in 0..<4 { d.append(Data([
                UInt8(word & 0xFF), UInt8((word >> 8) & 0xFF), UInt8((word >> 16) & 0xFF), UInt8(word >> 24)])) }
        }
        return d
    }

    func testGDNProjectionBandwidth() throws {
        let n = 10240, k = 2560            // the GDN qkv projection shape
        let bytes = n * (k / 32) * 20
        let arena = try QWArena(hal: Self.hal, weights: bytes + (8 << 20), activations: 512 << 20, kv: 0, gdnStates: 0)
        let data = makeBank(n, k)
        let w = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
        arena.copyIn(w, from: data)
        let view = QWWeightView(blocks: w, n: n, k: k, isRotated: true)!
        let act = arena.activationsRegion()
        for rows in [1, 32, 512] {
            let x = try act.allocate(QWTensorShape([rows * k], .f32))
            arena.copyIn(x, from: (0..<(rows * k)).map { Float($0 % 13) * 0.01 }
                .withUnsafeBufferPointer { Data(buffer: $0) })
            let y = try act.allocate(QWTensorShape([rows * n], .f32))
            var best = Double.infinity
            for _ in 0..<3 {
                guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
                let t0 = CACurrentMediaTime()
                try QWGemm.encodeHQ4F32V4(enc, hal: Self.hal, weight: view, x: x, y: y, rows: rows)
                Self.hal.end(cb, enc)
                cb.waitUntilCompleted()
                best = min(best, CACurrentMediaTime() - t0)
            }
            let weightBytes = Double(bytes) * Double(rows)   // the weight is re-read per row
            let xBytes = Double(rows * k * 4)
            print(String(format: "GDNPROJ rows=%d: %.2f ms | weight traffic %.2f GB -> %.0f GB/s | x traffic %.2f GB -> %.0f GB/s",
                         rows, best * 1000, weightBytes / 1e9, weightBytes / 1e9 / best,
                         xBytes / 1e9, xBytes / 1e9 / best))
        }
    }
}
