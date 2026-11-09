import XCTest
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWQuant

/// The 4-row HQ4 tile kernel must match the 1-row kernel for every shape, including multi-row batches.
final class FNMetalV4TileTests: XCTestCase {
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

    func testV4TileMatchesR1ForEveryShape() throws {
        let shapes = [(n: 64, k: 2560), (n: 16384, k: 2560), (n: 2560, k: 6144), (n: 320, k: 10240),
                      (n: 10240, k: 320), (n: 64, k: 64), (n: 16, k: 32), (n: 100, k: 2560), (n: 512, k: 512)]
        let rows = 4
        var failures: [String] = []
        for shape in shapes {
            let (n, k) = (shape.n, shape.k)
            let bytes = n * (k / 32) * 20
            let arena = try QWArena(hal: Self.hal, weights: bytes + (4 << 20), activations: 64 << 20, kv: 0, gdnStates: 0)
            let data = makeBank(n, k)
            let w = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
            arena.copyIn(w, from: data)
            let view = QWWeightView(blocks: w, n: n, k: k, isRotated: true)!
            let act = arena.activationsRegion()
            let x = try act.allocate(QWTensorShape([rows * k], .f32))
            let xs = (0..<(rows * k)).map { Float(($0 % 17)) * 0.05 - 0.4 }
            arena.copyIn(x, from: xs.withUnsafeBufferPointer { Data(buffer: $0) })
            let y4 = try act.allocate(QWTensorShape([rows * n], .f32))
            let y1 = try act.allocate(QWTensorShape([rows * n], .f32))
            guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
            try QWGemm.encodeHQ4F32V4(enc, hal: Self.hal, weight: view, x: x, y: y4, rows: rows)
            for r in 0..<rows {
                let xRow = QWTensorValue(region: x.region, offset: x.offset + r * k * 4, shape: QWTensorShape([k], .f32))
                let yRow = QWTensorValue(region: y1.region, offset: y1.offset + r * n * 4, shape: QWTensorShape([n], .f32))
                try QWGemm.encodeHQ4F32V4(enc, hal: Self.hal, weight: view, x: xRow, y: yRow, rows: 1)
            }
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            let pa = arena.cpuPointer(of: y4).assumingMemoryBound(to: Float.self)
            let pb = arena.cpuPointer(of: y1).assumingMemoryBound(to: Float.self)
            let a = (0..<(rows * n)).map { pa[$0] }
            let b = (0..<(rows * n)).map { pb[$0] }
            var worst: Float = 0
            for i in 0..<a.count { worst = max(worst, abs(a[i] - b[i])) }
            let scale = max(1e-6, b.map { abs($0) }.max() ?? 1)
            if worst / scale > 1e-4 { failures.append("n=\(n) k=\(k): rel=\(worst / scale)") }
        }
        XCTAssertTrue(failures.isEmpty, "v4 tile mismatch: " + failures.joined(separator: ", "))
    }

    func testAQV4TileAndArgmaxMatchR1ForEverySupportedPair() throws {
        let n = 512, k = 256, rows = 4
        let slots = QWGemm.blockArgmaxSlots(n)
        for pair in QWAQBlock.supported {
            let (bits, group) = (pair.bits, pair.group)
            let groups = k / group
            let maxVal = UInt32((1 << bits) - 1)
            var bank = Data()
            for r in 0..<n {
                for g in 0..<groups {
                    let values = (0..<group).map { i in UInt32((i + r * 5 + g * 11) % Int(maxVal + 1)) }
                    let block = try QWAQBlock.pack(bits: bits, group: group, values: values,
                                                   scale: 0.03125, bias: -0.25)
                    bank.append(block)
                }
            }
            let arena = try QWArena(hal: Self.hal, weights: bank.count + (2 << 20),
                                    activations: 16 << 20, kv: 0, gdnStates: 0)
            let w = try arena.weightsRegion().allocate(QWTensorShape([bank.count / 2], .f16))
            arena.copyIn(w, from: bank)
            let view = QWWeightView(blocks: w, n: n, k: k, bits: bits, groupSize: group, isRotated: false)!
            let act = arena.activationsRegion()
            let x = try act.allocate(QWTensorShape([rows * k], .f32))
            let xs = (0..<(rows * k)).map { i -> Float in
                let r = i / k, c = i % k
                return Float(((c * (r + 3) + r * 19) % 29) - 14) * 0.05
            }
            arena.write(x, from: xs)
            let yV4 = try act.allocate(QWTensorShape([rows * n], .f32))
            let yAm = try act.allocate(QWTensorShape([rows * n], .f32))
            let keysHi = try act.allocate(QWTensorShape([rows * slots], .f32))
            let keysLo = try act.allocate(QWTensorShape([rows * slots], .i32))
            let tokOut = try act.allocate(QWTensorShape([rows], .i32))

            guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
            try QWGemm.encodeHQ4F32V4(enc, hal: Self.hal, weight: view, x: x, y: yV4, rows: rows)
            try QWGemm.encodeHQ4F32V4Argmax(enc, hal: Self.hal, weight: view, x: x, y: yAm,
                                            keysHi: keysHi, keysLo: keysLo, rows: rows)
            enc.memoryBarrier(scope: .buffers)
            try QWGemm.encodeArgmaxFin(enc, hal: Self.hal, keysHi: keysHi, keysLo: keysLo,
                                       token: tokOut, slots: slots, rows: rows)
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)

            let gotV4 = arena.read(yV4, as: Float.self, count: rows * n)
            let gotToks = arena.read(tokOut, as: Int32.self, count: rows)
            XCTAssertTrue(gotV4.allSatisfy { $0.isFinite }, "AQ\(bits)_G\(group) v4 has non-finite output")
            var cpuRef = [Float](repeating: 0, count: rows * n)
            var worstV4ReferenceError = 0.0
            for index in 0..<(rows * n) {
                let token = index / n, output = index % n
                var exact = 0.0, magnitude = 0.0
                for c in 0..<k {
                    let g = c / group, i = c % group
                    let code = (i + output * 5 + g * 11) % Int(maxVal + 1)
                    let term = (Double(code) * 0.03125 - 0.25) * Double(xs[token * k + c])
                    exact += term
                    magnitude += abs(term)
                }
                cpuRef[index] = Float(exact)
                let denominator = max(magnitude, 1e-6)
                worstV4ReferenceError = max(worstV4ReferenceError, abs(Double(gotV4[index]) - exact) / denominator)
            }
            // Absolute-product scaling remains meaningful for cancelling dot products.
            let referenceTolerance = 64 * Double(Float.ulpOfOne)
            XCTAssertLessThan(worstV4ReferenceError, referenceTolerance, "AQ\(bits)_G\(group) v4 vs CPU")
            let scale = max(1e-6, cpuRef.map { abs($0) }.max() ?? 1)

            for r in 0..<rows {
                var bestIdx = 0
                var bestVal = -Float.infinity
                for col in 0..<n {
                    let v = cpuRef[r * n + col]
                    if v > bestVal { bestVal = v; bestIdx = col }
                }
                // Low-bit synthetic banks repeat rows, so exact logit ties are resolved by v4-vs-ref rounding.
                let tok = Int(gotToks[r])
                XCTAssertTrue(tok >= 0 && tok < n, "AQ\(bits)_G\(group) row \(r) argmax out of range")
                if tok >= 0 && tok < n {
                    XCTAssertLessThanOrEqual(bestVal - cpuRef[r * n + tok], 1e-4 * scale,
                                             "AQ\(bits)_G\(group) row \(r) argmax mismatch (best \(bestIdx), got \(tok))")
                }
            }
        }
    }
}
