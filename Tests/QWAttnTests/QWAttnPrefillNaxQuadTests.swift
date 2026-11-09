import XCTest
import Metal
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// EXP-9 sub-item 2 parity gate: `qwg_attn_prefill_nax_quad` (new, opt-in,
/// all-4-simdgroups-participate, BQ=64/threadgroup, direct-store epilogue)
/// vs the SHIPPING `qwg_attn_prefill_nax_x64` kernel (BQ=32/threadgroup,
final class QWAttnPrefillNaxQuadTests: XCTestCase {

    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    // MARK: - SplitMix64 deterministic PRNG (same family as the GDN regres harness)
    struct SplitMix64 {
        var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func next() -> UInt64 {
            state = state &+ 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        mutating func nextUnit() -> Float {
            // (0,1) uniform, avoids exact 0/1 to keep logs/divisions finite.
            Float(next() >> 11) * (1.0 / 9007199254740992.0) * 0.999998 + 0.000001
        }
        mutating func nextNormalish() -> Float {
            // crude Irwin-Hall(4)-ish: cheap, deterministic, bounded, no NaN risk.
            var s: Float = 0
            for _ in 0..<4 { s += nextUnit() }
            return (s - 2.0) * 0.9
        }
    }

    static func f16Bits(_ v: Float) -> UInt16 { QWQuant.f32tof16Bits(v) }
    static func f16(_ bits: UInt16) -> Float { QWQuant.f16tof32Bits(bits) }

    /// Synthesize one case's raw (unpadded) buffers: qgate [S][12288],
    /// k/v [S][1024], all f16. Q/K rows are l2-normed over headDim (matches
    /// the engine's norming contract used elsewhere in this repo), V and
    static func synth(S: Int, seed: UInt64) -> (qgate: [UInt16], k: [UInt16], v: [UInt16]) {
        var rng = SplitMix64(seed: seed)
        let H = QWModelGeometry.qwen38_27b.headDim, QH = QWModelGeometry.qwen38_27b.qHeads, KVH = QWModelGeometry.qwen38_27b.kvHeads
        var qgate = [UInt16](repeating: 0, count: S * QWModelGeometry.qwen38_27b.qProjOut)
        var k = [UInt16](repeating: 0, count: S * KVH * H)
        var v = [UInt16](repeating: 0, count: S * KVH * H)
        for row in 0..<S {
            for h in 0..<QH {
                var q = (0..<H).map { _ in rng.nextNormalish() }
                let n = sqrtf(q.reduce(0) { $0 + $1 * $1 }) + 1e-6
                for d in 0..<H { q[d] /= n }
                let base = row * QWModelGeometry.qwen38_27b.qProjOut + h * (2 * H)
                for d in 0..<H { qgate[base + d] = f16Bits(q[d]) }
                for d in 0..<H { qgate[base + H + d] = f16Bits(rng.nextNormalish() * 0.5) } // gate logits
            }
            for h in 0..<KVH {
                var kk = (0..<H).map { _ in rng.nextNormalish() }
                let n = sqrtf(kk.reduce(0) { $0 + $1 * $1 }) + 1e-6
                for d in 0..<H { kk[d] /= n }
                let kbase = row * KVH * H + h * H
                for d in 0..<H { k[kbase + d] = f16Bits(kk[d]) }
                let vbase = kbase
                for d in 0..<H { v[vbase + d] = f16Bits(rng.nextNormalish() * 0.3) }
            }
        }
        return (qgate, k, v)
    }

    static func u16Buffer(_ a: [UInt16], padElems: Int) -> MTLBuffer {
        let b = hal.device.makeBuffer(length: max(16, (a.count + padElems) * 2), options: .storageModeShared)!
        memset(b.contents(), 0, (a.count + padElems) * 2)   // zero pad rows (v5+ kernel law)
        _ = a.withUnsafeBufferPointer { src in
            memcpy(b.contents(), src.baseAddress!, a.count * 2)
        }
        return b
    }
    static func readOut(_ b: MTLBuffer, _ n: Int) -> [Float] {
        let p = b.contents().assumingMemoryBound(to: UInt16.self)
        return (0..<n).map { f16(p[$0]) }
    }

    /// Dispatch one kernel directly (bypassing QWAttnPrefill so each arm can
    /// use its OWN tileRows-dependent padding and grid, matching the exact
    /// pattern `QWAttnPrefillTests.runPrefill(..., kernel:)` already uses for
    static func run(kernel: String, tileRows: Int,
                     qgate: [UInt16], k: [UInt16], v: [UInt16], S: Int) throws -> [Float] {
        let pso = try hal.pipeline(kernel)
        let qgateBuf = u16Buffer(qgate, padElems: tileRows * QWModelGeometry.qwen38_27b.qProjOut)
        let kBuf = u16Buffer(k, padElems: tileRows * QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim)
        let vBuf = u16Buffer(v, padElems: tileRows * QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim)
        let outLen = S * QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim
        let outBuf = hal.device.makeBuffer(length: outLen * 2, options: .storageModeShared)!
        memset(outBuf.contents(), 0, outLen * 2)
        var sU = UInt32(S)
        guard let (cb, enc) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
        enc.setComputePipelineState(pso)
        enc.setBuffer(qgateBuf, offset: 0, index: 0)
        enc.setBuffer(kBuf, offset: 0, index: 1)
        enc.setBuffer(vBuf, offset: 0, index: 2)
        enc.setBuffer(outBuf, offset: 0, index: 3)
        enc.setBytes(&sU, length: 4, index: 4)
        let tiles = (S + tileRows - 1) / tileRows
        enc.dispatchThreadgroups(MTLSize(width: tiles * QWModelGeometry.qwen38_27b.qHeads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return readOut(outBuf, outLen)
    }

    /// 1-ulp adjacency (C06 law), local copy — matches QWAttnPrefillTests.
    static func parityViolations(_ got: [Float], _ ref: [Float], absTol: Float = 1e-3)
        -> (violations: Int, worstAbs: Double, worstIdx: Int) {
        func spacing(_ v: Float) -> Double {
            let e = Int((v.bitPattern >> 23) & 0xFF) - 127 - 10
            return e >= 0 ? Double(1 << e) : 1.0 / Double(1 << -e)
        }
        var viol = 0, worst = -1.0, worstIdx = -1
        for i in 0..<got.count {
            let d = abs(Double(got[i]) - Double(ref[i]))
            if d > worst { worst = d; worstIdx = i }
            guard d > Double(absTol) else { continue }
            if d / min(spacing(got[i]), spacing(ref[i])) > 1.0001 { viol += 1 }
        }
        return (viol, worst, worstIdx)
    }

    /// Covers: single short tile (7), exact OLD tile boundary (32), exact
    /// NEW tile boundary (64), one row past the new tile (65), a ragged
    /// multi-tile size exercising BOTH kernels' tails simultaneously (100),
    func testQuadVsX64Parity() throws {
        for S in [7, 32, 64, 65, 100, 128, 512] {
            let (qg, k, v) = Self.synth(S: S, seed: 0xA11CE_D00D ^ UInt64(S))
            let ref = try Self.run(kernel: "qwg_attn_prefill_nax_x64", tileRows: 32, qgate: qg, k: k, v: v, S: S)
            let got = try Self.run(kernel: "qwg_attn_prefill_nax_quad", tileRows: 64, qgate: qg, k: k, v: v, S: S)
            XCTAssertEqual(got.filter { !$0.isFinite }.count, 0, "quad S=\(S): non-finite output")
            let p = Self.parityViolations(got, ref)
            XCTAssertEqual(p.violations, 0, "quad S=\(S): parity vs nax_x64 (worst \(p.worstAbs) @\(p.worstIdx))")
        }
    }

    /// Causal law, re-asserted directly against the NEW kernel (mirrors
    /// QWAttnPrefillTests.testCausalLaw but targets qwg_attn_prefill_nax_quad):
    /// corrupting future k/v rows must not change earlier rows' output.
    func testQuadCausalLaw() throws {
        let S = 64
        let (qg, k, v) = Self.synth(S: S, seed: 0xC0FFEE)
        let clean = try Self.run(kernel: "qwg_attn_prefill_nax_quad", tileRows: 64, qgate: qg, k: k, v: v, S: S)
        var kM = k, vM = v
        for i in (32 * QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim)..<(S * QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim) {
            kM[i] = kM[i] &+ 0x0100
            vM[i] = vM[i] &+ 0x0080
        }
        let mut = try Self.run(kernel: "qwg_attn_prefill_nax_quad", tileRows: 64, qgate: qg, k: kM, v: vM, S: S)
        var diffs = 0
        let rowW = QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim
        for r in 0..<32 {
            for col in 0..<rowW where clean[r * rowW + col] != mut[r * rowW + col] { diffs += 1 }
        }
        XCTAssertEqual(diffs, 0, "quad causal law: mutating future k/v must not change rows 0..<32")
    }

    /// Cross-tile-width boundary: a 65-row case must show row 64 (the one
    /// row that falls in the new kernel's SECOND threadgroup, tile0=64)
    /// matching between a 65-row run and a 128-row run's row 64 ONLY up to
    func testQuadTileBoundaryConsistency() throws {
        let (qg65, k65, v65) = Self.synth(S: 65, seed: 0x5EED)
        let got65 = try Self.run(kernel: "qwg_attn_prefill_nax_quad", tileRows: 64, qgate: qg65, k: k65, v: v65, S: 65)
        // Build a 128-row case whose first 65 rows are byte-identical to the
        // 65-row case (same seed → same PRNG stream prefix).
        let (qg128, k128, v128) = Self.synth(S: 128, seed: 0x5EED)
        let got128 = try Self.run(kernel: "qwg_attn_prefill_nax_quad", tileRows: 64, qgate: qg128, k: k128, v: v128, S: 128)
        let rowW = QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim
        var diffs = 0
        for col in 0..<rowW where got65[64 * rowW + col] != got128[64 * rowW + col] { diffs += 1 }
        XCTAssertEqual(diffs, 0, "row 64 must be identical across S=65 and S=128 (causal law across tile0=64 boundary)")
    }
}
