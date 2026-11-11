import XCTest
import Metal
import Foundation
@testable import QWGDN
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// C13 chunked-prefill gate: `qwg_gdn_prefill_{solve,scan,norm}` vs torch
/// fixtures (gen_gdn_prefill_fixtures.py — golden = the NAIVE fp32 walk, the
/// exact admission framing of spec-13: "state + output vs naive C12 path").
final class QWGDNPrefillTests: XCTestCase {

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
    static func u16s(_ d: Data) -> [UInt16] { d.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) } }
    static func f32s(_ d: Data) -> [Float] { d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }
    static func f16(_ bits: UInt16) -> Float { QWQuant.f16tof32Bits(bits) }

    static func buf(_ d: Data) -> MTLBuffer {
        let b = hal.device.makeBuffer(length: max(16, d.count), options: .storageModeShared)!
        d.withUnsafeBytes { _ = memcpy(b.contents(), $0.baseAddress!, $0.count) }
        return b
    }
    static func tv(_ b: MTLBuffer, _ n: Int, _ dt: QWDType) -> QWTensorValue {
        QWTensorValue(region: QWMRegion(buffer: b, base: 0,
                                        length: max(16, Int((Double(n) * dt.bytesPerParam).rounded()))),
                      offset: 0, shape: QWTensorShape([n], dt))
    }

    /// Adjacency law: f16 STORES compare at 1-ulp-of-f16 spacing (value·2⁻¹⁰;
    /// Double.ulpOfOne is the FP64 machine eps and turns a legal store-rounding
    /// tie at |v|≈4 into a "violation" — exactly the 3.90625e-3 @|v|=4 seen at
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

    /// One prefill of T tokens (L chunk) → o rows (f16 bits, [T][6144]) +
    /// final state (f32, in-place buffer read back).
    static func prefill(y: Data, beta: Data, glog: Data, z: Data, w: Data,
                        state0: Data, T: Int, L: Int) throws -> (o: [UInt16], state: [Float]) {
        let eng = try QWGDNPrefill(hal: hal)
        let yb = buf(y), bb = buf(beta), gb = buf(glog), zb = buf(z), wb = buf(w)
        let stBuf = buf(state0)
        let uw = QWGDNPrefill.uwabElems(T: T)
        func zeros(_ n: Int, _ dt: QWDType) -> MTLBuffer {
            let b = hal.device.makeBuffer(length: max(16, Int((Double(n) * dt.bytesPerParam).rounded())),
                                          options: .storageModeShared)!
            memset(b.contents(), 0, b.length)
            return b
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
        ce.endEncoding()                                          // end-before-wait
        cb.commit(); cb.waitUntilCompleted()
        let po = ob.contents().assumingMemoryBound(to: UInt16.self)
        let ps = stBuf.contents().assumingMemoryBound(to: Float.self)
        return ((0..<(T * 6144)).map { po[$0] }, (0..<(48 * 128 * 128)).map { ps[$0] })
    }

    static func checkHead(_ tag: String, _ T: Int, L: Int) throws {
        let y = try load("gdn_pre_y_\(tag).bin"), z = try load("gdn_pre_z_\(tag).bin")
        let beta = try load("gdn_pre_beta_\(tag).bin"), glog = try load("gdn_pre_g_\(tag).bin")
        let state0 = try load("gdn_pre_state0_\(tag).bin"), w = try load("gdn_pre_w_\(tag).bin")
        let (oBits, st) = try prefill(y: y, beta: beta, glog: glog, z: z, w: w,
                                      state0: state0, T: T, L: L)
        // which o-rows the fixture dumped
        let dumped = [1, 2, 63, 64, 65, 66, 128, 129, 130, T - 1, T].filter { $0 > 0 && $0 <= T }
        var ov = 0, worst = 0.0, wi = -1   // (avoid shadowing static viol())
        for t in dumped {
            let ref = u16s(try load("gdn_pre_o16_t\(t)_\(tag).bin"))
            for h in 0..<48 {
                let off = (t - 1) * 6144 + h * 128      // t is 1-based; o 0-based
                for d in 0..<128 {
                    let g = f16(oBits[off + d]), r = f16(ref[h * 128 + d])   // golden [48,128] h-major
                    let e = abs(Double(g) - Double(r))
                    if e > worst { worst = e; wi = off + d }
                    let sp = max(abs(Double(g)), abs(Double(r))) * (1.0 / 1024.0)   // f16 1-ulp
                    if !(g.isFinite && r.isFinite) || e > max(1e-3, sp) { ov += 1 }
                }
            }
        }
        XCTAssertEqual(ov, 0, "\(tag)/L\(L): o parity (worst \(worst) @\(wi))")
        // state @1e-4 adjacency over the WHOLE walk (spec gate @1K)
        let stRef = f32s(try load("gdn_pre_state_\(tag).bin"))
        let v = viol(st, stRef, 1e-4)
        XCTAssertEqual(st.filter { !$0.isFinite }.count, 0, "\(tag)/L\(L): state non-finite")
        XCTAssertEqual(v.0, 0, "\(tag)/L\(L): state parity (worst \(v.1) @\(v.2))")
    }

    func testPrefillParityRagged() throws { try Self.checkHead("r", 130, L: 64) }
    func testPrefillParity1K() throws { try Self.checkHead("k", 1024, L: 64) }

    func testPrefillChunk128Guard() throws {
        // L=128 exceeds the shipped kernel's 64×64 solve tile → must throw
        // (the spec sweeps L∈{64,128}; a 128 variant is a separate kernel).
        let eng = try QWGDNPrefill(hal: Self.hal)
        let small = Self.hal.device.makeBuffer(length: 16, options: .storageModeShared)!
        guard let (cb, ce) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
        XCTAssertThrowsError(try eng.encode(ce,
            y: Self.tv(small, 10240, .f16), betas: Self.tv(small, 48, .f32),
            glog: Self.tv(small, 48, .f32), z: Self.tv(small, 6144, .f16),
            w: Self.tv(small, 128, .f16), state: Self.tv(small, 48 * 128 * 128, .f32),
            o: Self.tv(small, 6144, .f16), U: Self.tv(small, 6144, .f32),
            W: Self.tv(small, 6144, .f32), A: Self.tv(small, 6144, .f32),
            B: Self.tv(small, 6144, .f32), oScratch: Self.tv(small, 6144, .f32),
            T: 128, L: 128))
        ce.endEncoding(); cb.commit()
    }

    func testChunkedVsStreamedPrefillParity() throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "Streamed GDN decode requires bare-metal Apple Silicon GPU")
        // STATE-CLOBBER DETECTOR & MULTI-TOKEN PARITY LAW:
        // Comparing chunked K1/K2/K3 scan against sequential C12 streamed decode (T=1 x N).
        // Using ragged fixture (T=130, L=64):
        let tag = "r", T = 130, L = 64
        let y = try Self.load("gdn_pre_y_\(tag).bin")
        let beta = try Self.load("gdn_pre_beta_\(tag).bin")
        let glog = try Self.load("gdn_pre_g_\(tag).bin")
        let z = try Self.load("gdn_pre_z_\(tag).bin")
        let w = try Self.load("gdn_pre_w_\(tag).bin")
        let state0 = try Self.load("gdn_pre_state0_\(tag).bin")

        // 1. Chunked prefill (K1/K2/K3)
        let (oChunkedBits, stChunked) = try Self.prefill(y: y, beta: beta, glog: glog, z: z, w: w,
                                                         state0: state0, T: T, L: L)

        // 2. Streamed decode (C12 token-by-token)
        let dec = try QWGDNDecode(hal: Self.hal)
        let yb = Self.buf(y), bb = Self.buf(beta), gb = Self.buf(glog), zb = Self.buf(z), wb = Self.buf(w)
        let stBuf = Self.buf(state0)
        let oStreamedBuf = Self.hal.device.makeBuffer(length: T * 6144 * 2, options: .storageModeShared)!
        let oScratchBuf = Self.hal.device.makeBuffer(length: 48 * 128 * 4, options: .storageModeShared)!
        let oScratchTV = Self.tv(oScratchBuf, 48 * 128, .f32)
        let wTV = Self.tv(wb, 128, .f16)
        let stateTV = Self.tv(stBuf, 48 * 128 * 128, .f32)

        for t in 0..<T {
            guard let (cb, ce) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
            let yRow = QWTensorValue(region: QWMRegion(buffer: yb, base: 0, length: T * 10240 * 2),
                                     offset: t * 10240 * 2, shape: QWTensorShape([10240], .f16))
            let bRow = QWTensorValue(region: QWMRegion(buffer: bb, base: 0, length: T * 48 * 4),
                                     offset: t * 48 * 4, shape: QWTensorShape([48], .f32))
            let gRow = QWTensorValue(region: QWMRegion(buffer: gb, base: 0, length: T * 48 * 4),
                                     offset: t * 48 * 4, shape: QWTensorShape([48], .f32))
            let zRow = QWTensorValue(region: QWMRegion(buffer: zb, base: 0, length: T * 6144 * 2),
                                     offset: t * 6144 * 2, shape: QWTensorShape([6144], .f16))
            let oRow = QWTensorValue(region: QWMRegion(buffer: oStreamedBuf, base: 0, length: T * 6144 * 2),
                                     offset: t * 6144 * 2, shape: QWTensorShape([6144], .f16))

            try dec.encode(ce, y: yRow, betas: bRow, glog: gRow, state: stateTV,
                           z: zRow, w: wTV, o: oRow, oScratch: oScratchTV)
            Self.hal.end(cb, ce)
            cb.waitUntilCompleted()
        }

        let pStreamedSt = stBuf.contents().assumingMemoryBound(to: Float.self)
        let stStreamed = (0..<(48 * 128 * 128)).map { pStreamedSt[$0] }
        let pStreamedO = oStreamedBuf.contents().assumingMemoryBound(to: UInt16.self)
        let oStreamedBits = (0..<(T * 6144)).map { pStreamedO[$0] }

        // Compare state: <= 1e-4 tolerance
        let vState = Self.viol(stChunked, stStreamed, 1e-4)
        XCTAssertEqual(vState.0, 0, "Chunked vs streamed final state parity mismatch (worst \(vState.1) @\(vState.2))")

        // Compare o outputs:
        let dumped = [1, 2, 63, 64, 65, 66, 128, 129, 130].filter { $0 <= T }
        var ov = 0, worstO = 0.0, wiO = -1
        for t in dumped {
            for h in 0..<48 {
                let off = (t - 1) * 6144 + h * 128
                for d in 0..<128 {
                    let cVal = Self.f16(oChunkedBits[off + d])
                    let sVal = Self.f16(oStreamedBits[off + d])
                    let diff = abs(Double(cVal) - Double(sVal))
                    if diff > worstO { worstO = diff; wiO = off + d }
                    let sp = max(abs(Double(cVal)), abs(Double(sVal))) * (1.0 / 1024.0)
                    if !(cVal.isFinite && sVal.isFinite) || diff > max(1e-3, sp) { ov += 1 }
                }
            }
        }
        XCTAssertEqual(ov, 0, "Chunked vs streamed o output parity mismatch (worst \(worstO) @\(wiO))")
    }

    /// Explicit A/B parity test between the scalar forward-substitution solve
    /// and the Phase 1 Nilpotent Triangular Inversion solve.
    func testNilpotentVsScalarSolveParity() throws {
        let tag = "r"
        let T = 130, L = 64
        let y = try Self.load("gdn_pre_y_\(tag).bin"), z = try Self.load("gdn_pre_z_\(tag).bin")
        let beta = try Self.load("gdn_pre_beta_\(tag).bin"), glog = try Self.load("gdn_pre_g_\(tag).bin")
        let state0 = try Self.load("gdn_pre_state0_\(tag).bin"), w = try Self.load("gdn_pre_w_\(tag).bin")

        func runPrefill(solve: QWGDNPrefill.SolveBackend) throws -> (o: [UInt16], state: [Float]) {
            let eng = try QWGDNPrefill(hal: Self.hal, solveBackend: solve)
            let yb = Self.buf(y), bb = Self.buf(beta), gb = Self.buf(glog), zb = Self.buf(z), wb = Self.buf(w)
            let stBuf = Self.buf(state0)
            let uw = QWGDNPrefill.uwabElems(T: T)
            func zeros(_ n: Int, _ dt: QWDType) -> MTLBuffer {
                let b = Self.hal.device.makeBuffer(length: max(16, Int((Double(n) * dt.bytesPerParam).rounded())),
                                                   options: .storageModeShared)!
                memset(b.contents(), 0, b.length)
                return b
            }
            let Ub = zeros(uw, .f32), Wb = zeros(uw, .f32), Ab = zeros(uw, .f32), Bb = zeros(uw, .f32)
            let osb = zeros(QWGDNPrefill.oScratchElems(T: T), .f32)
            let ob = zeros(T * 6144, .f16)
            guard let (cb, ce) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
            try eng.encode(ce, y: Self.tv(yb, T * 10240, .f16), betas: Self.tv(bb, T * 48, .f32),
                           glog: Self.tv(gb, T * 48, .f32), z: Self.tv(zb, T * 6144, .f16),
                           w: Self.tv(wb, 128, .f16), state: Self.tv(stBuf, 48 * 128 * 128, .f32),
                           o: Self.tv(ob, T * 6144, .f16),
                           U: Self.tv(Ub, uw, .f32), W: Self.tv(Wb, uw, .f32),
                           A: Self.tv(Ab, uw, .f32), B: Self.tv(Bb, uw, .f32),
                           oScratch: Self.tv(osb, QWGDNPrefill.oScratchElems(T: T), .f32),
                           T: T, L: L)
            ce.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
            let po = ob.contents().assumingMemoryBound(to: UInt16.self)
            let ps = stBuf.contents().assumingMemoryBound(to: Float.self)
            return ((0..<(T * 6144)).map { po[$0] }, (0..<(48 * 128 * 128)).map { ps[$0] })
        }

        let scalarRes = try runPrefill(solve: .scalar)
        let nilpotentRes = try runPrefill(solve: .nilpotent)

        // Compare state parity (fp32)
        var maxStateDiff: Float = 0.0
        for i in 0..<scalarRes.state.count {
            let diff = abs(scalarRes.state[i] - nilpotentRes.state[i])
            if diff > maxStateDiff { maxStateDiff = diff }
        }
        XCTAssertLessThan(maxStateDiff, 1e-4, "Nilpotent vs scalar solve state diff \(maxStateDiff) exceeds 1e-4")

        // Compare output parity (f16 store spacing per C06 adjacency law)
        var ov = 0, worstO = 0.0
        for i in 0..<scalarRes.o.count {
            let sVal = Self.f16(scalarRes.o[i])
            let nVal = Self.f16(nilpotentRes.o[i])
            let diff = abs(Double(sVal) - Double(nVal))
            if diff > worstO { worstO = diff }
            let sp = max(abs(Double(sVal)), abs(Double(nVal))) * (1.0 / 1024.0)
            if !(sVal.isFinite && nVal.isFinite) || diff > max(1e-3, sp) { ov += 1 }
        }
        XCTAssertEqual(ov, 0, "Nilpotent vs scalar solve output violations (\(ov) viol, worst \(worstO))")
    }
}
