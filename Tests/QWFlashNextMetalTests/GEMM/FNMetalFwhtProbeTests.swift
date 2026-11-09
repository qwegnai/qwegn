import XCTest
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
@testable import QWFlashNext

// Probe: qwg_fwht32_f16 via QWGemm.encodeFwht32 on a known f16 vector, against a CPU orthonormal FWHT.
final class FNMetalFwhtProbeTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    func testFwht32RotatesAnF16Vector() throws {
        let n = 6400   // same size as the MoE chain (K*I), 256-thread path
        let input: [Float16] = (0..<n).map { Float16(Float(($0 * 37) % 19) - 9) / 4 }
        let arena = try QWArena(hal: Self.hal, weights: 1024, activations: n * 2 + 4096, kv: 0, gdnStates: 0)
        let v = try arena.activationsRegion().allocate(QWTensorShape([n], .f16))
        arena.copyIn(v, from: input.withUnsafeBufferPointer { Data(buffer: $0) })

        guard let (cb, enc) = Self.hal.begin() else { return XCTFail("no command buffer") }
        try QWGemm.encodeFwht32(enc, hal: Self.hal, x: v, count: n)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let got = Array(UnsafeBufferPointer(start: arena.cpuPointer(of: v).bindMemory(to: Float16.self, capacity: n), count: n)).map { Float($0) }
        // CPU orthonormal FWHT per group of 32 (self-inverse).
        var want = input.map { Float($0) }
        for g in stride(from: 0, to: n, by: 32) {
            var h = 1
            while h < 32 {
                var i0 = 0
                while i0 < 32 {
                    for j in 0..<h {
                        let a = want[g + i0 + j], b = want[g + i0 + j + h]
                        want[g + i0 + j] = a + b
                        want[g + i0 + j + h] = a - b
                    }
                    i0 += 2 * h
                }
                h *= 2
            }
        }
        for i in 0..<n { want[i] /= Float(32).squareRoot() }
        var maxErr: Float = 0
        for i in 0..<n { maxErr = max(maxErr, abs(got[i] - want[i])) }
        print("FNFwhtProbe in[0..3]=\(input[0..<4].map { Float($0) }) got[0..3]=\(got[0..<4]) want[0..3]=\(want[0..<4]) maxErr=\(maxErr)")
        XCTAssertLessThan(maxErr, 1e-2)
    }
}
