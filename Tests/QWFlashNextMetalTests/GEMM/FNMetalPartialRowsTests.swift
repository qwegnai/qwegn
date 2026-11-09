import XCTest
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWGEMM

/// Partial-tile parity: an R-row GEMM must equal R single-row GEMMs, for every row count and epilogue
/// the Flash-Next prefill uses. Covers the 4-row tile kernel (V4) and the 64-row batched kernel (BM).
final class FNMetalPartialRowsTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    /// Builds an HQ4_G32 bank (scale, bias, four packed words per 32 columns) with finite values.
    private func makeBank(_ n: Int, _ k: Int) -> Data {
        var d = Data(capacity: n * (k / 32) * 20)
        let scale = UInt16(0x211F)          // f16 0.01
        let word = UInt32(0x11111111)
        for _ in 0..<(n * (k / 32)) {
            d.append(Data([UInt8(scale & 0xFF), UInt8(scale >> 8), 0, 0]))
            for _ in 0..<4 { d.append(Data([
                UInt8(word & 0xFF), UInt8((word >> 8) & 0xFF), UInt8((word >> 16) & 0xFF), UInt8(word >> 24)])) }
        }
        return d
    }

    func testPartialRowsMatchSingleRow() throws {
        let n = 256, k = 2560
        let bytes = n * (k / 32) * 20
        let rowsList = [2, 3, 5, 7, 19, 31, 32, 33]
        let epis: [(UInt32, Float)] = [(0, 1), (2, 1), (3, 4)]
        let maxRows = rowsList.max()!
        let arena = try QWArena(hal: Self.hal, weights: bytes + (4 << 20), activations: 64 << 20, kv: 0, gdnStates: 0)
        let w = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
        arena.copyIn(w, from: makeBank(n, k))
        let view = QWWeightView(blocks: w, n: n, k: k, isRotated: true)!
        let act = arena.activationsRegion()
        let x = try act.allocate(QWTensorShape([maxRows * k], .f32))
        let xs = (0..<(maxRows * k)).map { Float(($0 % 17)) * 0.05 - 0.4 }
        arena.copyIn(x, from: xs.withUnsafeBufferPointer { Data(buffer: $0) })
        let y = try act.allocate(QWTensorShape([maxRows * n], .f32))
        let ref = try act.allocate(QWTensorShape([maxRows * n], .f32))
        var failures: [String] = []
        for (epi, scale) in epis {
            for rows in rowsList {
                for kernel in ["V4", "BM"] {
                    guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
                    func run(_ xin: QWTensorValue, _ yout: QWTensorValue, _ r: Int) throws {
                        if kernel == "V4" {
                            try QWGemm.encodeHQ4F32V4(enc, hal: Self.hal, weight: view, x: xin, y: yout, rows: r, epi: epi, scale: scale)
                        } else {
                            try QWGemm.encodeHQ4F32BM(enc, hal: Self.hal, weight: view, x: xin, y: yout, rows: r, epi: epi, scale: scale)
                        }
                    }
                    try run(x, y, rows)
                    // Reference: the batched BM kernel with one row (the semantics 32+ row prefill relies on).
                    for r in 0..<rows {
                        try QWGemm.encodeHQ4F32BM(enc, hal: Self.hal, weight: view,
                                                  x: x.slice(byteOffset: r * k * 4, count: k, dtype: .f32),
                                                  y: ref.slice(byteOffset: r * n * 4, count: n, dtype: .f32),
                                                  rows: 1, epi: epi, scale: scale)
                    }
                    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                    let pa = arena.cpuPointer(of: y).assumingMemoryBound(to: Float.self)
                    let pb = arena.cpuPointer(of: ref).assumingMemoryBound(to: Float.self)
                    var worst: Float = 0, peak: Float = 1e-6
                    for i in 0..<(rows * n) {
                        worst = max(worst, abs(pa[i] - pb[i])); peak = max(peak, abs(pb[i]))
                    }
                    if worst / peak > 1e-4 { failures.append("\(kernel) rows=\(rows) epi=\(epi): rel=\(worst / peak)") }
                }
            }
        }
        XCTAssertTrue(failures.isEmpty, "partial-row mismatch:\n" + failures.joined(separator: "\n"))
    }
}
