// QWWeightBasisNormTests.swift — the activation basis.
//
// The engine's own weights are quantised in the ROTATED basis, and the
// activations are rotated to match. A checkpoint kept in its own bytes
// holds weights in the STANDARD basis, and its activations must NOT be
// rotated. This proves the switch is real: the two bases' outputs differ
// by exactly the engine's own transform, and nothing else.
import XCTest
import Metal
@testable import QWNorm
import QWCore
import QWHAL
import QWAlloc
import QWQuant

final class QWWeightBasisNormTests: XCTestCase {
    static let H = QWModelGeometry.qwen38_27b.hidden
    /// The two HALs, one per basis. A missing library is a SKIP with a
    /// reason, not a `try!` that aborts the process: the standard library is
    /// a second build, and a checkout that has not built it should say so.
    static func hal(_ basis: QWWeightBasis) throws -> QWHal {
        do { return try QWHal(geometry: .qwen38_27b, weightBasis: basis) }
        catch { throw XCTSkip("the \(basis) metallib is not built: \(error)") }
    }

    /// Deterministic, non-degenerate values: no all-zero row, no constant.
    static func inputs(count: Int) -> [Float] {
        var seed: UInt32 = 0x1234_5678
        return (0..<count).map { _ in
            seed = seed &* 1664525 &+ 1013904223
            return Float(Int32(truncatingIfNeeded: seed >> 8) % 1024) / 1024.0 - 0.5
        }
    }

    func runNorm(_ hal: QWHal, x: QWTensorValue, gamma: QWTensorValue,
                 out: QWTensorValue) throws {
        guard let (cb, enc) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
        do {
            try QWNorm.encode(enc, hal: hal, variant: .rmsnorm, x: x, gamma: gamma,
                              out: out, residual: nil, rowScalesOut: nil,
                              geometry: .qwen38_27b)
        } catch {
            enc.endEncoding()
            throw error
        }
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
    }

    /// The engine's own 32-point transform, in place. It runs on the
    /// ROTATED hal on purpose: the transform kernel is gated on the basis
    /// too, so only the rotated basis performs the transform at all.
    func rotate(_ hal: QWHal, _ t: QWTensorValue, count: Int) throws {
        guard let (cb, enc) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
        enc.setComputePipelineState(try hal.pipeline("qwg_fwht32_f16"))
        t.bind(enc, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: count / 32, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
    }

    /// The two bases differ by exactly the transform, and the standard one
    /// is the plain norm.
    func testTheTwoBasesDifferByExactlyTheTransform() throws {
        let rows = 2
        let n = rows * Self.H
        let arena = try QWArena(hal: try Self.hal(.rotated), weights: 0,
                                activations: n * 2 * 4 + Self.H * 2 + 8192, kv: 0, gdnStates: 0)
        let a = arena.activationsRegion()
        let x = try a.allocate(QWTensorShape([n], .f16))
        let gamma = try a.allocate(QWTensorShape([Self.H], .f16))
        let yRot = try a.allocate(QWTensorShape([n], .f16))
        let yPlain = try a.allocate(QWTensorShape([n], .f16))
        let scratch = try a.allocate(QWTensorShape([n], .f16))
        arena.write(x, from: Self.inputs(count: n).map { Float16($0) })
        arena.write(gamma, from: Self.inputs(count: Self.H).map { Float16($0) })

        try runNorm(try Self.hal(.rotated), x: x, gamma: gamma, out: yRot)
        try runNorm(try Self.hal(.standard), x: x, gamma: gamma, out: yPlain)

        // The two must actually differ — otherwise the gate would be a
        // no-op and the comparison below would prove nothing.
        let rotPtr = arena.cpuPointer(of: yRot).assumingMemoryBound(to: UInt16.self)
        let plainPtr = arena.cpuPointer(of: yPlain).assumingMemoryBound(to: UInt16.self)
        var differing = 0
        for i in 0..<n where rotPtr[i] != plainPtr[i] { differing += 1 }
        XCTAssertGreaterThan(differing, n / 2,
                             "the two bases produced the same numbers: the gate does nothing")

        // Rotating the plain output must give the rotated one.
        let dst = arena.cpuPointer(of: scratch).assumingMemoryBound(to: UInt16.self)
        for i in 0..<n { dst[i] = plainPtr[i] }
        try rotate(try Self.hal(.rotated), scratch, count: n)

        var worst = 0.0
        var magnitude = 0.0
        for i in 0..<n {
            let rotated = Float(QWQuant.f16tof32Bits(rotPtr[i]))
            let transformed = Float(QWQuant.f16tof32Bits(dst[i]))
            magnitude = max(magnitude, abs(Double(rotated)))
            worst = max(worst, abs(Double(rotated) - Double(transformed)))
        }
        XCTAssertLessThan(worst / max(magnitude, 1e-6), 2e-3,
                          "the rotated output is not the transform of the plain one")
    }

    /// The transform kernel is gated too: on the standard basis it leaves
    /// the buffer alone.
    func testTheTransformKernelIsGatedOnTheBasis() throws {
        let n = 256
        let arena = try QWArena(hal: try Self.hal(.rotated), weights: 0,
                                activations: n * 2 * 2 + 4096, kv: 0, gdnStates: 0)
        let a = arena.activationsRegion()
        let t = try a.allocate(QWTensorShape([n], .f16))
        let scratch = try a.allocate(QWTensorShape([n], .f16))
        arena.write(t, from: Self.inputs(count: n).map { Float16($0) })
        let src = arena.cpuPointer(of: t).assumingMemoryBound(to: UInt16.self)
        let dst = arena.cpuPointer(of: scratch).assumingMemoryBound(to: UInt16.self)
        for i in 0..<n { dst[i] = src[i] }

        try rotate(try Self.hal(.standard), scratch, count: n)
        var changed = 0
        for i in 0..<n where dst[i] != src[i] { changed += 1 }
        XCTAssertEqual(changed, 0,
                       "the standard basis must not rotate: the transform kernel is gated")
    }
}
