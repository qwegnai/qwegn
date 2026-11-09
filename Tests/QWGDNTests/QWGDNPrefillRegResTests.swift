import XCTest
import Metal
import Foundation
@testable import QWGDN
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// EXP-9 sub-item 1 (OPT-IN, NOT shipping default) — parity for
/// `qwg_gdn_prefill_regres`, the register-resident alternative prefill
/// recurrence. Self-contained: generates its own deterministic synthetic
final class QWGDNPrefillRegResTests: XCTestCase {

    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    struct SplitMix64 {
        var s: UInt64
        mutating func next() -> UInt64 {
            s = s &+ 0x9E3779B97F4A7C15
            var z = s
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        mutating func u() -> Double { Double(next() >> 11) / Double(1 << 53) }
    }

    static func buf(_ bytes: Int) -> MTLBuffer { hal.device.makeBuffer(length: max(16, bytes), options: .storageModeShared)! }
    static func tv(_ b: MTLBuffer, _ n: Int, _ dt: QWDType) -> QWTensorValue {
        QWTensorValue(region: QWMRegion(buffer: b, base: 0,
                                        length: max(16, Int((Double(n) * dt.bytesPerParam).rounded()))),
                      offset: 0, shape: QWTensorShape([n], dt))
    }
    static func rowTV(_ b: MTLBuffer, totalBytes: Int, rowOffsetBytes: Int, n: Int, dt: QWDType) -> QWTensorValue {
        QWTensorValue(region: QWMRegion(buffer: b, base: 0, length: totalBytes),
                      offset: rowOffsetBytes, shape: QWTensorShape([n], dt))
    }

    /// Fixture scenario: synthesizes q̂/k̂ (l2-normed per the norming
    /// contract), v, beta, glog, z, w, and an optional nonzero initial
    /// state, with the requested decay/beta bands.
    struct Scenario {
        let T: Int
        let seed: UInt64
        let decayLogLo: Double, decayLogHi: Double   // glog band (<=0)
        let betaLo: Double, betaHi: Double
        let nonzeroState: Bool
    }

    static func l2normed(_ rng: inout SplitMix64, dim: Int, scale: Double) -> [Float] {
        var v = (0..<dim).map { _ in Float(-1.0 + 2.0 * rng.u()) }
        var ss: Float = 0
        for x in v { ss += x * x }
        let inv = 1.0 / Foundation.sqrt(max(ss, 1e-12))
        for i in v.indices { v[i] *= inv * Float(scale) }
        return v
    }

    /// Builds the three f16/f32 input buffers (y, betas, glog, z, w) and an
    /// optional nonzero initial state buffer for `T` tokens, 48 v-heads /
    /// 16 key-heads / Dk=Dv=128 — SAME layout `QWGDNPrefill`/`QWGDNDecode`
    static func makeInputs(_ sc: Scenario) -> (y: [UInt16], beta: [Float], glog: [Float],
                                                z: [UInt16], w: [UInt16], state0: [Float]) {
        var rng = SplitMix64(s: sc.seed)
        var y = [UInt16](repeating: 0, count: sc.T * 10240)
        var beta = [Float](repeating: 0, count: sc.T * 48)
        var glog = [Float](repeating: 0, count: sc.T * 48)
        var z = [UInt16](repeating: 0, count: sc.T * 6144)
        for t in 0..<sc.T {
            for i in 0..<16 {
                let qh = l2normed(&rng, dim: 128, scale: pow(128.0, -0.5))
                let kh = l2normed(&rng, dim: 128, scale: 1.0)
                for d in 0..<128 {
                    y[t * 10240 + i * 128 + d] = QWQuant.f32tof16Bits(qh[d])
                    y[t * 10240 + 2048 + i * 128 + d] = QWQuant.f32tof16Bits(kh[d])
                }
            }
            for h in 0..<48 {
                for d in 0..<128 {
                    y[t * 10240 + 4096 + h * 128 + d] = QWQuant.f32tof16Bits(Float(-0.5 + rng.u()))
                }
                beta[t * 48 + h] = Float(sc.betaLo + (sc.betaHi - sc.betaLo) * rng.u())
                glog[t * 48 + h] = Float(sc.decayLogLo + (sc.decayLogHi - sc.decayLogLo) * rng.u())
            }
            for e in 0..<6144 { z[t * 6144 + e] = QWQuant.f32tof16Bits(Float(-1.0 + 2.0 * rng.u())) }
        }
        var w = [UInt16](repeating: 0, count: 128)
        for d in 0..<128 { w[d] = QWQuant.f32tof16Bits(Float(0.8 + 0.4 * rng.u())) }
        var state0 = [Float](repeating: 0, count: 48 * 128 * 128)
        if sc.nonzeroState {
            for i in state0.indices { state0[i] = Float(-0.3 + 0.6 * rng.u()) }
        }
        return (y, beta, glog, z, w, state0)
    }

    static func viol(_ got: [Float], _ ref: [Float], _ absTol: Float, f16Spacing: Bool = false) -> (Int, Double, Int) {
        var v = 0, worst = 0.0, wi = -1
        let eps = f16Spacing ? 1.0 / 1024.0 : Double.ulpOfOne
        for i in 0..<got.count {
            let d = abs(Double(got[i]) - Double(ref[i]))
            if d > worst { worst = d; wi = i }
            guard got[i].isFinite && ref[i].isFinite else { v += 1; continue }
            let sp = max(Double(abs(got[i])), Double(abs(ref[i]))) * eps
            if d > max(Double(absTol), sp) { v += 1 }
        }
        return (v, worst, wi)
    }

    /// Runs the register-resident path. Returns (o bits [T][6144], final
    /// state f32 [48*128*128]).
    static func runRegRes(y: [UInt16], beta: [Float], glog: [Float], z: [UInt16], w: [UInt16],
                          state0: [Float], T: Int) throws -> (o: [UInt16], state: [Float]) {
        let eng = try QWGDNPrefill(hal: hal)
        let yb = buf(y.count * 2); yb.contents().copyMemory(from: y, byteCount: y.count * 2)
        let bb = buf(beta.count * 4); bb.contents().copyMemory(from: beta, byteCount: beta.count * 4)
        let gb = buf(glog.count * 4); gb.contents().copyMemory(from: glog, byteCount: glog.count * 4)
        let zb = buf(z.count * 2); zb.contents().copyMemory(from: z, byteCount: z.count * 2)
        let wb = buf(w.count * 2); wb.contents().copyMemory(from: w, byteCount: w.count * 2)
        let stBuf = buf(state0.count * 4); stBuf.contents().copyMemory(from: state0, byteCount: state0.count * 4)
        let osb = buf(T * 6144 * 4); memset(osb.contents(), 0, osb.length)
        let ob = buf(T * 6144 * 2); memset(ob.contents(), 0, ob.length)
        guard let (cb, ce) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
        try eng.encodeRegRes(ce, y: tv(yb, T * 10240, .f16), betas: tv(bb, T * 48, .f32),
                             glog: tv(gb, T * 48, .f32), z: tv(zb, T * 6144, .f16),
                             w: tv(wb, 128, .f16), state: tv(stBuf, 48 * 128 * 128, .f32),
                             o: tv(ob, T * 6144, .f16), oScratch: tv(osb, T * 6144, .f32), T: T)
        ce.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error { throw e }
        let po = ob.contents().assumingMemoryBound(to: UInt16.self)
        let ps = stBuf.contents().assumingMemoryBound(to: Float.self)
        return ((0..<(T * 6144)).map { po[$0] }, (0..<(48 * 128 * 128)).map { ps[$0] })
    }

    /// Runs the certified streamed-decode oracle (C12, one dispatch/token).
    static func runStreamed(y: [UInt16], beta: [Float], glog: [Float], z: [UInt16], w: [UInt16],
                            state0: [Float], T: Int) throws -> (o: [UInt16], state: [Float]) {
        let dec = try QWGDNDecode(hal: hal)
        let yb = buf(y.count * 2); yb.contents().copyMemory(from: y, byteCount: y.count * 2)
        let bb = buf(beta.count * 4); bb.contents().copyMemory(from: beta, byteCount: beta.count * 4)
        let gb = buf(glog.count * 4); gb.contents().copyMemory(from: glog, byteCount: glog.count * 4)
        let zb = buf(z.count * 2); zb.contents().copyMemory(from: z, byteCount: z.count * 2)
        let wb = buf(w.count * 2); wb.contents().copyMemory(from: w, byteCount: w.count * 2)
        let stBuf = buf(state0.count * 4); stBuf.contents().copyMemory(from: state0, byteCount: state0.count * 4)
        let oBuf = buf(T * 6144 * 2); memset(oBuf.contents(), 0, oBuf.length)
        let oScratchBuf = buf(48 * 128 * 4)
        let oScratchTV = tv(oScratchBuf, 48 * 128, .f32)
        let wTV = tv(wb, 128, .f16)
        let stateTV = tv(stBuf, 48 * 128 * 128, .f32)
        for t in 0..<T {
            guard let (cb, ce) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
            let yRow = rowTV(yb, totalBytes: y.count * 2, rowOffsetBytes: t * 10240 * 2, n: 10240, dt: .f16)
            let bRow = rowTV(bb, totalBytes: beta.count * 4, rowOffsetBytes: t * 48 * 4, n: 48, dt: .f32)
            let gRow = rowTV(gb, totalBytes: glog.count * 4, rowOffsetBytes: t * 48 * 4, n: 48, dt: .f32)
            let zRow = rowTV(zb, totalBytes: z.count * 2, rowOffsetBytes: t * 6144 * 2, n: 6144, dt: .f16)
            let oRow = rowTV(oBuf, totalBytes: T * 6144 * 2, rowOffsetBytes: t * 6144 * 2, n: 6144, dt: .f16)
            try dec.encode(ce, y: yRow, betas: bRow, glog: gRow, state: stateTV,
                          z: zRow, w: wTV, o: oRow, oScratch: oScratchTV)
            hal.end(cb, ce)
            cb.waitUntilCompleted()
            if let e = cb.error { throw e }
        }
        let po = oBuf.contents().assumingMemoryBound(to: UInt16.self)
        let ps = stBuf.contents().assumingMemoryBound(to: Float.self)
        return ((0..<(T * 6144)).map { po[$0] }, (0..<(48 * 128 * 128)).map { ps[$0] })
    }

    /// Runs the certified shipping chunked prefill (C13 / MTILDE).
    static func runChunked(y: [UInt16], beta: [Float], glog: [Float], z: [UInt16], w: [UInt16],
                          state0: [Float], T: Int, L: Int) throws -> (o: [UInt16], state: [Float]) {
        let eng = try QWGDNPrefill(hal: hal)
        let yb = buf(y.count * 2); yb.contents().copyMemory(from: y, byteCount: y.count * 2)
        let bb = buf(beta.count * 4); bb.contents().copyMemory(from: beta, byteCount: beta.count * 4)
        let gb = buf(glog.count * 4); gb.contents().copyMemory(from: glog, byteCount: glog.count * 4)
        let zb = buf(z.count * 2); zb.contents().copyMemory(from: z, byteCount: z.count * 2)
        let wb = buf(w.count * 2); wb.contents().copyMemory(from: w, byteCount: w.count * 2)
        let stBuf = buf(state0.count * 4); stBuf.contents().copyMemory(from: state0, byteCount: state0.count * 4)
        let uw = QWGDNPrefill.uwabElems(T: T)
        func zeros(_ n: Int, _ dt: QWDType) -> MTLBuffer {
            let b = buf(Int((Double(n) * dt.bytesPerParam).rounded())); memset(b.contents(), 0, b.length); return b
        }
        let Ub = zeros(uw, .f32), Wb = zeros(uw, .f32), Ab = zeros(uw, .f32), Bb = zeros(uw, .f32)
        let osb = zeros(QWGDNPrefill.oScratchElems(T: T), .f32)
        let ob = zeros(T * 6144, .f16)
        guard let (cb, ce) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
        try eng.encode(ce, y: tv(yb, T * 10240, .f16), betas: tv(bb, T * 48, .f32),
                      glog: tv(gb, T * 48, .f32), z: tv(zb, T * 6144, .f16),
                      w: tv(wb, 128, .f16), state: tv(stBuf, 48 * 128 * 128, .f32),
                      o: tv(ob, T * 6144, .f16),
                      U: tv(Ub, uw, .f32), W: tv(Wb, uw, .f32),
                      A: tv(Ab, uw, .f32), B: tv(Bb, uw, .f32),
                      oScratch: tv(osb, QWGDNPrefill.oScratchElems(T: T), .f32),
                      T: T, L: L)
        ce.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error { throw e }
        let po = ob.contents().assumingMemoryBound(to: UInt16.self)
        let ps = stBuf.contents().assumingMemoryBound(to: Float.self)
        return ((0..<(T * 6144)).map { po[$0] }, (0..<(48 * 128 * 128)).map { ps[$0] })
    }

    static func compare(_ tag: String, _ a: (o: [UInt16], state: [Float]), _ b: (o: [UInt16], state: [Float])) {
        let af = a.o.map { QWQuant.f16tof32Bits($0) }, bf = b.o.map { QWQuant.f16tof32Bits($0) }
        XCTAssertEqual(af.filter { !$0.isFinite }.count, 0, "\(tag): o non-finite (lhs)")
        XCTAssertEqual(bf.filter { !$0.isFinite }.count, 0, "\(tag): o non-finite (rhs)")
        XCTAssertEqual(a.state.filter { !$0.isFinite }.count, 0, "\(tag): state non-finite (lhs)")
        XCTAssertEqual(b.state.filter { !$0.isFinite }.count, 0, "\(tag): state non-finite (rhs)")
        let vo = viol(af, bf, 1e-3, f16Spacing: true)
        XCTAssertEqual(vo.0, 0, "\(tag): o parity mismatch (worst \(vo.1) @\(vo.2) / \(af.count) entries)")
        let vs = viol(a.state, b.state, 1e-4)
        XCTAssertEqual(vs.0, 0, "\(tag): state parity mismatch (worst \(vs.1) @\(vs.2) / \(a.state.count) entries)")
    }

    // MARK: - Scenarios (EXP-9 sub-item 1 ladder)

    func testZeroStateModerateDecay() throws {
        let sc = Scenario(T: 37, seed: 1, decayLogLo: -1.0, decayLogHi: -0.05,
                          betaLo: 0.3, betaHi: 0.9, nonzeroState: false)
        let (y, beta, glog, z, w, st0) = Self.makeInputs(sc)
        let r = try Self.runRegRes(y: y, beta: beta, glog: glog, z: z, w: w, state0: st0, T: sc.T)
        let d = try Self.runStreamed(y: y, beta: beta, glog: glog, z: z, w: w, state0: st0, T: sc.T)
        let c = try Self.runChunked(y: y, beta: beta, glog: glog, z: z, w: w, state0: st0, T: sc.T, L: 32)
        Self.compare("zeroState-vs-streamed", r, d)
        Self.compare("zeroState-vs-chunked", r, c)
    }

    func testNonzeroInitialState() throws {
        let sc = Scenario(T: 65, seed: 2, decayLogLo: -0.8, decayLogHi: -0.02,
                          betaLo: 0.4, betaHi: 0.95, nonzeroState: true)
        let (y, beta, glog, z, w, st0) = Self.makeInputs(sc)
        let r = try Self.runRegRes(y: y, beta: beta, glog: glog, z: z, w: w, state0: st0, T: sc.T)
        let d = try Self.runStreamed(y: y, beta: beta, glog: glog, z: z, w: w, state0: st0, T: sc.T)
        let c = try Self.runChunked(y: y, beta: beta, glog: glog, z: z, w: w, state0: st0, T: sc.T, L: 64)
        Self.compare("nonzeroState-vs-streamed", r, d)
        Self.compare("nonzeroState-vs-chunked", r, c)
    }

    func testRaggedLengths() throws {
        for T in [1, 3, 17, 130] {
            let sc = Scenario(T: T, seed: 100 + UInt64(T), decayLogLo: -0.9, decayLogHi: -0.03,
                              betaLo: 0.3, betaHi: 0.9, nonzeroState: T % 2 == 0)
            let (y, beta, glog, z, w, st0) = Self.makeInputs(sc)
            let r = try Self.runRegRes(y: y, beta: beta, glog: glog, z: z, w: w, state0: st0, T: T)
            let d = try Self.runStreamed(y: y, beta: beta, glog: glog, z: z, w: w, state0: st0, T: T)
            Self.compare("ragged-T\(T)-vs-streamed", r, d)
        }
    }

    func testBetaNearOne() throws {
        let sc = Scenario(T: 48, seed: 7, decayLogLo: -0.6, decayLogHi: -0.01,
                          betaLo: 0.97, betaHi: 0.999, nonzeroState: true)
        let (y, beta, glog, z, w, st0) = Self.makeInputs(sc)
        let r = try Self.runRegRes(y: y, beta: beta, glog: glog, z: z, w: w, state0: st0, T: sc.T)
        let d = try Self.runStreamed(y: y, beta: beta, glog: glog, z: z, w: w, state0: st0, T: sc.T)
        Self.compare("betaNearOne-vs-streamed", r, d)
    }

    func testStrongDecay() throws {
        // glog very negative: e^glog -> ~0, state should decay hard every step.
        let sc = Scenario(T: 48, seed: 9, decayLogLo: -20.0, decayLogHi: -5.0,
                          betaLo: 0.2, betaHi: 0.8, nonzeroState: true)
        let (y, beta, glog, z, w, st0) = Self.makeInputs(sc)
        let r = try Self.runRegRes(y: y, beta: beta, glog: glog, z: z, w: w, state0: st0, T: sc.T)
        let d = try Self.runStreamed(y: y, beta: beta, glog: glog, z: z, w: w, state0: st0, T: sc.T)
        Self.compare("strongDecay-vs-streamed", r, d)
    }

    /// Resumed sequence: a single T=96 regres pass must match a two-call
    /// regres walk (T1=40 state-carried into T2=56) splitting the SAME
    /// token stream — this is the resumed-sequence/state-ownership contract
    func testResumedSequenceMatchesSingleShot() throws {
        let T = 96, T1 = 40, T2 = 56
        let sc = Scenario(T: T, seed: 42, decayLogLo: -0.7, decayLogHi: -0.02,
                          betaLo: 0.3, betaHi: 0.9, nonzeroState: true)
        let (y, beta, glog, z, w, st0) = Self.makeInputs(sc)

        let whole = try Self.runRegRes(y: y, beta: beta, glog: glog, z: z, w: w, state0: st0, T: T)

        // split call 1: first T1 tokens, carries state0 -> state1
        let y1 = Array(y[0..<(T1 * 10240)]), b1 = Array(beta[0..<(T1 * 48)]), g1 = Array(glog[0..<(T1 * 48)])
        let z1 = Array(z[0..<(T1 * 6144)])
        let part1 = try Self.runRegRes(y: y1, beta: b1, glog: g1, z: z1, w: w, state0: st0, T: T1)

        // split call 2: remaining T2 tokens, resumes from part1.state
        let y2 = Array(y[(T1 * 10240)..<(T * 10240)]), b2 = Array(beta[(T1 * 48)..<(T * 48)])
        let g2 = Array(glog[(T1 * 48)..<(T * 48)]), z2 = Array(z[(T1 * 6144)..<(T * 6144)])
        let part2 = try Self.runRegRes(y: y2, beta: b2, glog: g2, z: z2, w: w, state0: part1.state, T: T2)

        let resumedO = part1.o + part2.o
        Self.compare("resumed-vs-singleShot", (o: resumedO, state: part2.state), whole)
    }
}
