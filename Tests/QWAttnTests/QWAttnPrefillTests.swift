import XCTest
import Metal
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// C08 prefill kernel gate: `qwg_attn_prefill` vs torch fixtures
/// (gen_attn_fixtures.py — fp32 oracle, f16 engine-storage law). Parity law:
/// C06 1-ulp adjacency @ absTol 1e-3. Cases: S ∈ {7,16,32,128,512} (tile
final class QWAttnPrefillTests: XCTestCase {

    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    static let spikeOut = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Scripts/spike/out")

    static func load(_ name: String) throws -> Data {
        let p = spikeOut.appendingPathComponent(name)
        guard let d = FileManager.default.contents(atPath: p.path) else {
            throw NSError(domain: "fixture", code: 2, userInfo: [NSLocalizedDescriptionKey: "missing \(name)"])
        }
        return d
    }
    static func buf(_ d: Data) -> MTLBuffer {
        let b = hal.device.makeBuffer(length: max(16, d.count), options: .storageModeShared)!
        d.withUnsafeBytes { _ = memcpy(b.contents(), $0.baseAddress!, $0.count) }
        return b
    }
    /// Collective fragment loads cannot clamp per-row → inputs carry +32 zero
    /// PAD ROWS (flash-attention padding law; tail rows S̃=0 masked by
    /// causality, tail O masked at store). makeBuffer is zero-filled.
    static func padded(_ d: Data, rowBytes: Int, padRows: Int = 32) -> MTLBuffer {
        let b = hal.device.makeBuffer(length: max(16, d.count + padRows * rowBytes), options: .storageModeShared)!
        d.withUnsafeBytes { _ = memcpy(b.contents(), $0.baseAddress!, $0.count) }
        return b
    }
    /// v5 kernel requires ZEROED pad rows (staged P̂ reads pbuf beyond kEnd;
    /// garbage there leaks exp() into dim columns ≥ kEnd−a0 for near-tail
    /// rows). Tests that measure the old garbage-tolerant contract must opt
    static func padded(_ d: Data, rowBytes: Int, zeroPad: Bool, padRows: Int = 32) -> MTLBuffer {
        let extra = padRows * rowBytes
        let b = hal.device.makeBuffer(length: max(16, d.count + extra), options: .storageModeShared)!
        memset(b.contents(), 0, d.count + extra)
        d.withUnsafeBytes { _ = memcpy(b.contents(), $0.baseAddress!, $0.count) }
        return b
    }
    static func u16(_ b: MTLBuffer, _ n: Int) -> [UInt16] {
        let p = b.contents().assumingMemoryBound(to: UInt16.self)
        return (0..<n).map { p[$0] }
    }
    static func f16(_ bits: UInt16) -> Float { QWQuant.f16tof32Bits(bits) }
    static func bits(_ v: Float) -> UInt16 { QWQuant.f32tof16Bits(v) }
    static func readF(_ bits: [UInt16]) -> [Float] { bits.map { f16($0) } }

    /// 1-ulp adjacency (C06 law) — local copy (test targets are independent).
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

    /// Full prefill: qgate/k/v [S][…] f16 → out [S][6144] f16.
    /// grid.x = ⌈S/32⌉·24 workgroups × 128 threads (4 simdgroups × 8 rows).
    static func runPrefill(qgate: Data, k: Data, v: Data, S: Int) throws -> [UInt16] {
        try runPrefill(qgate: qgate, k: k, v: v, S: S, kernel: "qwg_attn_prefill")
    }

    /// Same dispatch, explicit kernel name (the NAX swap gate runs BOTH arms
    /// through identical fixtures: the shipping v5.1 kernel and the NAX
    /// tensor-ops rewrite must each hold the C06 parity law vs the CPU goldens).
    static func runPrefill(qgate: Data, k: Data, v: Data, S: Int, kernel: String) throws -> [UInt16] {
        let outLen = S * 6144 * 2                    // o-proj input width 6144 f16
        let out = hal.device.makeBuffer(length: outLen, options: .storageModeShared)!
        var sU = UInt32(S)
        let pso = try hal.pipeline(kernel)
        let tileRows = 32
        guard let (cb, enc) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
        enc.setComputePipelineState(pso)
        enc.setBuffer(padded(qgate, rowBytes: 12288 * 2, zeroPad: true, padRows: tileRows), offset: 0, index: 0)
        enc.setBuffer(padded(k, rowBytes: 1024 * 2, zeroPad: true, padRows: tileRows), offset: 0, index: 1)
        enc.setBuffer(padded(v, rowBytes: 1024 * 2, zeroPad: true, padRows: tileRows), offset: 0, index: 2)
        enc.setBuffer(out, offset: 0, index: 3)
        enc.setBytes(&sU, length: 4, index: 4)
        let tiles = (S + tileRows - 1) / tileRows
        let tgThreads = 128
        // WORKGROUPS × 128 threads (dispatchThreads would dispatch 24 THREADS
        // total and leave 3.5 simdgroups inactive — the zero-strip bug).
        enc.dispatchThreadgroups(MTLSize(width: tiles * QWModelGeometry.qwen38_27b.qHeads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
        enc.endEncoding()                                        // end-before-wait
        cb.commit()
        cb.waitUntilCompleted()
        return u16(out, S * 6144)
    }

    static let cases: [(S: Int, tag: String)] = [
        (7, ""), (16, ""), (32, ""), (128, ""), (512, ""),
        (32, "_adv30"), (32, "_ties"),
    ]

    func testPrefillParity() throws {
        for c in Self.cases {
            let qg = try Self.load("attn_qgate_S\(c.S)\(c.tag).bin")
            let k = try Self.load("attn_k_S\(c.S)\(c.tag).bin")
            let v = try Self.load("attn_v_S\(c.S)\(c.tag).bin")
            let refData = try Self.load("attn_o_S\(c.S)\(c.tag).bin")
            let refBits = [UInt16](unsafeBytes: refData)
            let ref = Self.readF(refBits)
            let got = Self.readF(try Self.runPrefill(qgate: qg, k: k, v: v, S: c.S))

            XCTAssertEqual(got.filter { !$0.isFinite }.count, 0, "S=\(c.S)\(c.tag): non-finite output")
            let v2 = Self.parityViolations(got, ref)
            XCTAssertEqual(v2.violations, 0,
                "S=\(c.S)\(c.tag): parity (worst \(v2.worstAbs) @\(v2.worstIdx))")
        }
    }

    /// The NAX tensor-ops arm (qwg_attn_prefill_nax) must hold the SAME
    /// C06 parity vs the CPU goldens as v5.1 — the swap ships only when this
    /// is 0 violations. (S≤512 fixtures; the engine's maxChunk is 4096 and
    func testPrefillParityNax() throws {
        for c in Self.cases {
            let qg = try Self.load("attn_qgate_S\(c.S)\(c.tag).bin")
            let k = try Self.load("attn_k_S\(c.S)\(c.tag).bin")
            let v = try Self.load("attn_v_S\(c.S)\(c.tag).bin")
            let ref = Self.readF([UInt16](unsafeBytes: try Self.load("attn_o_S\(c.S)\(c.tag).bin")))
            let got = Self.readF(try Self.runPrefill(qgate: qg, k: k, v: v, S: c.S,
                                                     kernel: "qwg_attn_prefill_nax"))
            XCTAssertEqual(got.filter { !$0.isFinite }.count, 0, "NAX S=\(c.S)\(c.tag): non-finite")
            let p = Self.parityViolations(got, ref)
            XCTAssertEqual(p.violations, 0, "NAX S=\(c.S)\(c.tag): parity (worst \(p.worstAbs) @\(p.worstIdx))")
        }
    }

    func testPrefillParityNaxX64() throws {
        do {
            _ = try Self.hal.pipeline("qwg_attn_prefill_nax_x64")
        } catch {
            throw XCTSkip("nax_x64 kernel entrypoint not landed yet")
        }
        for c in Self.cases {
            let qg = try Self.load("attn_qgate_S\(c.S)\(c.tag).bin")
            let k = try Self.load("attn_k_S\(c.S)\(c.tag).bin")
            let v = try Self.load("attn_v_S\(c.S)\(c.tag).bin")
            let ref = Self.readF([UInt16](unsafeBytes: try Self.load("attn_o_S\(c.S)\(c.tag).bin")))
            let got = Self.readF(try Self.runPrefill(qgate: qg, k: k, v: v, S: c.S,
                                                     kernel: "qwg_attn_prefill_nax_x64"))
            XCTAssertEqual(got.filter { !$0.isFinite }.count, 0, "NAX-X64 S=\(c.S)\(c.tag): non-finite")
            let p = Self.parityViolations(got, ref)
            XCTAssertEqual(p.violations, 0, "NAX-X64 S=\(c.S)\(c.tag): parity (worst \(p.worstAbs) @\(p.worstIdx))")
        }
    }

    func testCausalLaw() throws {
        // THE causal property, tested directly: future keys must not affect
        // past outputs. Corrupt k̂/v̂ rows 16..31 (scale ×1000 — nothing
        // subtle) and require the S=32 output rows 0..15 to be BIT-IDENTICAL
        let qg = try Self.load("attn_qgate_S32.bin")
        let k = try Self.load("attn_k_S32.bin"), v = try Self.load("attn_v_S32.bin")
        let clean = try Self.runPrefill(qgate: qg, k: k, v: v, S: 32)
        var kM = [UInt8](k), vM = [UInt8](v)
        let rowBytes = 1024 * 2
        for i in (16 * rowBytes)..<(32 * rowBytes) { kM[i] = kM[i] &+ 77; vM[i] = vM[i] &+ 39 }
        let mut = try Self.runPrefill(qgate: qg, k: Data(kM), v: Data(vM), S: 32)
        var diffs = 0
        for r in 0..<16 {
            for col in 0..<6144 where clean[r * 6144 + col] != mut[r * 6144 + col] { diffs += 1 }
        }
        XCTAssertEqual(diffs, 0, "causal law: mutating future k/v must not change rows 0..15")
    }
}

private extension Array where Element == UInt16 {
    init(unsafeBytes data: Data) {
        self = Array(data.withUnsafeBytes { $0.bindMemory(to: UInt16.self) })
    }
}
