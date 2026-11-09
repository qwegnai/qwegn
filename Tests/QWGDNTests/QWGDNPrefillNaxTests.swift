import XCTest
import Metal
import Foundation
@testable import QWGDN
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// NAX scan-arm gate: `qwg_gdn_prefill_scan_nax` (simdgroup_matrix route) vs
/// the shipping `qwg_gdn_prefill_scan` — compare-vs-shipping is the law.
/// Arms run the FULL pipeline (K1 solve + scan arm + K3 norm) on the fixture
final class QWGDNPrefillNaxTests: XCTestCase {

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

    /// One prefill of T tokens (L chunk) with the given scan kernel name →
    /// o rows (f16 bits, [T][6144]) + final state (f32, in-place read back).
    static func prefill(scanName: String, y: Data, beta: Data, glog: Data, z: Data, w: Data,
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
                       T: T, L: L, kernelName: scanName)
        ce.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        let po = ob.contents().assumingMemoryBound(to: UInt16.self)
        let ps = stBuf.contents().assumingMemoryBound(to: Float.self)
        return ((0..<(T * 6144)).map { po[$0] }, (0..<(48 * 128 * 128)).map { ps[$0] })
    }

    /// o adjacency (f16 1-ulp spacing law — shipping store class).
    static func violO(_ got: [UInt16], _ ref: [UInt16]) -> (Int, Double, Int) {
        var v = 0, worst = 0.0, wi = -1
        for i in 0..<got.count {
            let g = Double(f16(got[i])), r = Double(f16(ref[i]))
            let d = abs(g - r)
            if d > worst { worst = d; wi = i }
            if !(g.isFinite && r.isFinite) { v += 1; continue }
            let sp = max(abs(g), abs(r)) * (1.0 / 1024.0)   // f16 1-ulp (signed law)
            if d > max(1e-3, sp) { v += 1 }
        }
        return (v, worst, wi)
    }
    /// state adjacency (strict fp32 relative law, 1e-4 rung).
    static func violS(_ got: [Float], _ ref: [Float]) -> (Int, Double, Int) {
        var v = 0, worst = 0.0, wi = -1
        for i in 0..<got.count {
            let d = abs(Double(got[i]) - Double(ref[i]))
            if d > worst { worst = d; wi = i }
            guard got[i].isFinite && ref[i].isFinite else { v += 1; continue }
            let sp = max(abs(Double(got[i])), abs(Double(ref[i]))) * Double.ulpOfOne
            if d > max(1e-4, sp) { v += 1 }
        }
        return (v, worst, wi)
    }

    /// Wall ms of ONE full prefill (warmup 1, then timed run), for both arms.
    static func timePrefill(scanName: String, y: Data, beta: Data, glog: Data, z: Data,
                            w: Data, state0: Data, T: Int, L: Int, reps: Int) throws -> Double {
        var best = Double.infinity
        for _ in 0..<reps {
            let t0 = DispatchTime.now()
            _ = try prefill(scanName: scanName, y: y, beta: beta, glog: glog, z: z,
                            w: w, state0: state0, T: T, L: L)
            let dt = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e6
            if dt < best { best = dt }
        }
        return best
    }

    func testScanParityNax() throws {
        let shipName = "qwg_gdn_prefill_scan"
        let naxName = "qwg_gdn_prefill_scan_nax"
        for (tag, T) in [("k", 1024), ("r", 130)] {
            let y = try Self.load("gdn_pre_y_\(tag).bin"), z = try Self.load("gdn_pre_z_\(tag).bin")
            let beta = try Self.load("gdn_pre_beta_\(tag).bin"), glog = try Self.load("gdn_pre_g_\(tag).bin")
            let state0 = try Self.load("gdn_pre_state0_\(tag).bin"), w = try Self.load("gdn_pre_w_\(tag).bin")
            // shipping arm ×2 (determinism control) + nax arm
            let (oShip, stShip) = try Self.prefill(scanName: shipName, y: y, beta: beta, glog: glog,
                                              z: z, w: w, state0: state0, T: T, L: 64)
            let (oShip2, stShip2) = try Self.prefill(scanName: shipName, y: y, beta: beta, glog: glog,
                                                z: z, w: w, state0: state0, T: T, L: 64)
            XCTAssertEqual(oShip, oShip2, "\(tag): shipping arm NOT deterministic (rerun differs)")
            XCTAssertEqual(stShip, stShip2, "\(tag): shipping state NOT deterministic")
            let (oNax, stNax) = try Self.prefill(scanName: naxName, y: y, beta: beta, glog: glog,
                                            z: z, w: w, state0: state0, T: T, L: 64)
            // o vs shipping (all rows, full width)
            let vo = Self.violO(oNax, oShip)
            XCTAssertEqual(vo.0, 0, "\(tag): nax-vs-ship o parity (worst \(vo.1) @\(vo.2))")
            XCTAssertEqual(stNax.filter { !$0.isFinite }.count, 0, "\(tag): nax state non-finite")
            // state vs the FIXTURE golden (1e-4 rung) + vs shipping arm
            let stRef = try Self.load("gdn_pre_state_\(tag).bin").withUnsafeBytes {
                Array($0.bindMemory(to: Float.self))
            }
            let vg = Self.violS(stNax, stRef)
            XCTAssertEqual(vg.0, 0, "\(tag): nax state vs fixture golden (worst \(vg.1) @\(vg.2))")
            let vs = Self.violS(stNax, stShip)
            XCTAssertEqual(vs.0, 0, "\(tag): nax state vs shipping arm (worst \(vs.1) @\(vs.2))")
            print("parity \(tag): o worst \(vo.1) (\(vo.0) viol), state vs golden worst \(vg.1) (\(vg.0) viol)")
        }
        // honest timing, min-of-5, T=1024 (both arms)
        let y = try Self.load("gdn_pre_y_k.bin"), z = try Self.load("gdn_pre_z_k.bin")
        let beta = try Self.load("gdn_pre_beta_k.bin"), glog = try Self.load("gdn_pre_g_k.bin")
        let state0 = try Self.load("gdn_pre_state0_k.bin"), w = try Self.load("gdn_pre_w_k.bin")
        let tShip = try Self.timePrefill(scanName: shipName, y: y, beta: beta, glog: glog, z: z,
                                    w: w, state0: state0, T: 1024, L: 64, reps: 5)
        let tNax = try Self.timePrefill(scanName: naxName, y: y, beta: beta, glog: glog, z: z,
                                   w: w, state0: state0, T: 1024, L: 64, reps: 5)
        print(String(format: "TIMING @T=1024 min-of-5: ship %.3f ms | nax %.3f ms (ratio %.2fx)",
                     tShip, tNax, tShip / tNax))
    }
}
