import XCTest
import Metal
import Foundation
@testable import QWGDN
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// C13 bisect ladder — each stage validated against a fp64-validated
/// reference before the next runs:
///   rung 1  K1 (`qwg_gdn_prefill_solve`) → U/Wmat/A/B (validated: the
final class QWGDNPrefillProbeTests: XCTestCase {

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
    static func f32s(_ d: Data) -> [Float] { d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }
    static func u16s(_ d: Data) -> [UInt16] { d.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) } }
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

    static func viol(_ got: [Double], _ ref: [Double], _ absTol: Double) -> (Int, Double, Int) {
        var v = 0, worst = 0.0, wi = -1
        for i in 0..<got.count {
            let d = abs(got[i] - ref[i])
            if d > worst { worst = d; wi = i }
            guard got[i].isFinite && ref[i].isFinite else { v += 1; continue }
            let sp = max(abs(got[i]), abs(ref[i])) * Double.ulpOfOne
            if d > max(absTol, sp) { v += 1 }
        }
        return (v, worst, wi)
    }

    private struct ProbeRun {
        let oScratch: [Float]
        let state: [Float]
    }

    private func scanKernelName(_ backend: QWGDNPrefill.ScanBackend) -> String {
        switch backend {
        case .scalar: return "qwg_gdn_prefill_scan"
        case .nax: return "qwg_gdn_prefill_scan_nax"
        case .hybrid: return "qwg_gdn_prefill_scan_hybrid"
        }
    }

    private func runPrefill(tag: String, T: Int, backend: QWGDNPrefill.ScanBackend) throws -> ProbeRun {
        let eng = try QWGDNPrefill(hal: Self.hal, scanBackend: backend)
        XCTAssertEqual(eng.kernelNameScan, scanKernelName(backend))

        let y = try Self.load("gdn_pre_y_\(tag).bin")
        let beta = try Self.load("gdn_pre_beta_\(tag).bin")
        let glog = try Self.load("gdn_pre_g_\(tag).bin")
        let z = try Self.load("gdn_pre_z_\(tag).bin")
        let w = try Self.load("gdn_pre_w_\(tag).bin")
        let st0 = try Self.load("gdn_pre_state0_\(tag).bin")

        let yb = Self.buf(y), bb = Self.buf(beta), gb = Self.buf(glog)
        let zb = Self.buf(z), wb = Self.buf(w), st = Self.buf(st0)
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
                       w: Self.tv(wb, 128, .f16), state: Self.tv(st, 48 * 128 * 128, .f32),
                       o: Self.tv(ob, T * 6144, .f16),
                       U: Self.tv(Ub, uw, .f32), W: Self.tv(Wb, uw, .f32),
                       A: Self.tv(Ab, uw, .f32), B: Self.tv(Bb, uw, .f32),
                       oScratch: Self.tv(osb, QWGDNPrefill.oScratchElems(T: T), .f32),
                       T: T, L: 64)
        ce.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let os = Self.f32s(Data(bytes: osb.contents(), count: osb.length))
        let stf = Self.f32s(Data(bytes: st.contents(), count: st.length))
        XCTAssertEqual(os.count, T * 6144, "oScratch size mismatch for \(tag)")
        XCTAssertEqual(stf.count, 48 * 128 * 128, "state size mismatch for \(tag)")
        return ProbeRun(oScratch: os, state: stf)
    }

    func testPrefillBisectLadder() throws {
        try XCTSkipIf(!FileManager.default.fileExists(
            atPath: Self.spikeOut.appendingPathComponent("gdn_probe_y.bin").path),
            "diagnostic probe fixtures are not produced by make init (C13 bisect aid)")
        let T = 130, L = 64
        let eng = try QWGDNPrefill(hal: Self.hal)
        let y = try Self.load("gdn_probe_y.bin"), beta = try Self.load("gdn_probe_beta.bin")
        let glog = try Self.load("gdn_probe_g.bin"), z = try Self.load("gdn_probe_z.bin")
        let w = try Self.load("gdn_probe_w.bin"), st0 = try Self.load("gdn_probe_state0.bin")
        XCTAssertEqual(w.count, 256, "norm weight w must be the 256-byte file (case collision!)")
        let yb = Self.buf(y), bb = Self.buf(beta), gb = Self.buf(glog)
        let zb = Self.buf(z), wb = Self.buf(w), st = Self.buf(st0)
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
                       w: Self.tv(wb, 128, .f16), state: Self.tv(st, 48 * 128 * 128, .f32),
                       o: Self.tv(ob, T * 6144, .f16),
                       U: Self.tv(Ub, uw, .f32), W: Self.tv(Wb, uw, .f32),
                       A: Self.tv(Ab, uw, .f32), B: Self.tv(Bb, uw, .f32),
                       oScratch: Self.tv(osb, QWGDNPrefill.oScratchElems(T: T), .f32),
                       T: T, L: L)
        ce.endEncoding()
        cb.commit(); cb.waitUntilCompleted()

        // ── rung 1: K1 → U/Wmat/A/B ──
        for (name, buf) in [("U", Ub), ("Wmat", Wb), ("A", Ab), ("B", Bb)] {
            let got = Self.f32s(Data(bytes: buf.contents(), count: buf.length)).map(Double.init)
            let ref = Self.f32s(try Self.load("gdn_probe_\(name).bin")).map(Double.init)
            let v = Self.viol(got, ref, 1e-4)
            XCTAssertEqual(v.0, 0, "\(name): K1 parity (worst \(v.1) @\(v.2))")
        }

        // ── rung 2: K2 → pre-norm o_scr vs the validated chain ──
        let gotOS = Self.f32s(Data(bytes: osb.contents(), count: osb.length)).map(Double.init)
        let refOS = Self.f32s(try Self.load("gdn_probe_oscr.bin")).map(Double.init)
        let v2 = Self.viol(gotOS, refOS, 1e-3)
        XCTAssertEqual(v2.0, 0, "o_scr: K2 parity (worst \(v2.1) @\(v2.2))")

        // ── rung 2b: K3 self-consistency recomputed in Swift from the
        // kernel's OWN o_scr (zero fixture trust) ──
        let obits = Self.u16s(Data(bytes: ob.contents(), count: ob.length))
        let zbits = Self.u16s(z)
        let wbits = Self.u16s(w)
        var nv = 0, nw = 0.0, nwi = -1, worstRow = -1
        for row in 0..<(T * 48) {
            let t = row / 48, h = row % 48
            var sumsq = 0.0
            for dd in 0..<128 { let v = Double(gotOS[row * 128 + dd]); sumsq += v * v }
            let rms = 1.0 / (sumsq / 128.0 + 1e-6).squareRoot()
            var refs = [Double](repeating: 0, count: 128)
            for dd in 0..<128 {
                let o = Double(gotOS[row * 128 + dd])
                let zn = Double(QWQuant.f16tof32Bits(zbits[t * 6144 + h * 128 + dd]))
                let wv = Double(QWQuant.f16tof32Bits(wbits[dd]))
                refs[dd] = o * rms * wv * (zn / (1.0 + exp(-zn)))
            }
            for blk in 0..<4 {                       // orthonormal FWHT-32 per block
                var s = 1
                while s < 32 {
                    for i in 0..<32 where i & s == 0 {
                        let a = refs[blk * 32 + i], b = refs[blk * 32 + i + s]
                        refs[blk * 32 + i] = a + b; refs[blk * 32 + i + s] = a - b
                    }
                    s <<= 1
                }
                for i in 0..<32 { refs[blk * 32 + i] *= 0.17677669529663689 }
            }
            for dd in 0..<128 {
                let refv = refs[dd]
                let gotv = Double(QWQuant.f16tof32Bits(obits[row * 128 + dd]))
                let e = abs(gotv - refv)
                if e > nw { nw = e; nwi = row * 128 + dd; worstRow = row }
                if e > max(1e-2, abs(refv) * 2e-3) { nv += 1 }
            }
        }
        XCTAssertEqual(nv, 0, "K3 self-consistency (worst \(nw) @\(nwi) row~\(worstRow / 48))")

        // ── rung 3: final o vs the r-fixture goldens ──
        for t in [1, 2, 63, 64, 65, 66, 128, 129, 130] {
            let ref = Self.u16s(try Self.load("gdn_pre_o16_t\(t)_r.bin"))
            var g: [Double] = [], r: [Double] = []
            for h in 0..<48 {
                let off = (t - 1) * 6144 + h * 128
                for d in 0..<128 {
                    g.append(Double(QWQuant.f16tof32Bits(obits[off + d])))
                    r.append(Double(QWQuant.f16tof32Bits(ref[h * 128 + d])))   // golden is [48,128] h-major (NOT ref[d] — that compared every head to head 0)
                }
            }
            let v3 = Self.viol(g, r, 4e-3)
            XCTAssertEqual(v3.0, 0, "kernel-o vs golden t=\(t) (worst \(v3.1) @\(v3.2))")
        }

        // ── rung 4: final state (in-place buffer) vs the chunked golden ──
        let stGot = Self.f32s(Data(bytes: st.contents(), count: st.length)).map(Double.init)
        let stRef = Self.f32s(try Self.load("gdn_probe_state.bin")).map(Double.init)
        XCTAssertEqual(stGot.filter { !$0.isFinite }.count, 0, "state non-finite")
        let v4 = Self.viol(stGot, stRef, 1e-4)
        XCTAssertEqual(v4.0, 0, "state: full-pipeline parity (worst \(v4.1) @\(v4.2))")
    }

    func testPrefillParityNaxK2() throws {
        for (tag, T) in [("r", 130), ("k", 1024)] {
            let ship = try runPrefill(tag: tag, T: T, backend: .scalar)
            let nax = try runPrefill(tag: tag, T: T, backend: .nax)

            let oShip = ship.oScratch.map(Double.init)
            let oNax = nax.oScratch.map(Double.init)
            let stShip = ship.state.map(Double.init)
            let stNax = nax.state.map(Double.init)

            let vo = Self.viol(oNax, oShip, 1e-3)
            XCTAssertEqual(vo.0, 0, "\(tag): oScratch ship-vs-nax parity (worst \(vo.1) @\(vo.2))")

            let vs = Self.viol(stNax, stShip, 1e-4)
            XCTAssertEqual(vs.0, 0, "\(tag): state ship-vs-nax parity (worst \(vs.1) @\(vs.2))")
        }
    }

    func testPrefillParityHybridK2() throws {
        for (tag, T) in [("r", 130), ("k", 1024)] {
            let ship = try runPrefill(tag: tag, T: T, backend: .scalar)
            let hybrid = try runPrefill(tag: tag, T: T, backend: .hybrid)

            let oShip = ship.oScratch.map(Double.init)
            let oHybrid = hybrid.oScratch.map(Double.init)
            let stShip = ship.state.map(Double.init)
            let stHybrid = hybrid.state.map(Double.init)

            let vo = Self.viol(oHybrid, oShip, 1e-3)
            XCTAssertEqual(vo.0, 0, "\(tag): oScratch ship-vs-hybrid parity (worst \(vo.1) @\(vo.2))")

            let vs = Self.viol(stHybrid, stShip, 1e-4)
            XCTAssertEqual(vs.0, 0, "\(tag): state ship-vs-hybrid parity (worst \(vs.1) @\(vs.2))")
        }
    }
}

