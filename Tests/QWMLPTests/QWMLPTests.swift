import XCTest
import Metal
import Foundation
@testable import QWMLP
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWQuant

/// C14 — SwiGLU FFN parity (spec 14: ≤5e-4 f16-out at all M incl. MTP M=8).
/// Fixtures (gen_mlp_fixtures.py): q4-g128 planes + fp32 refs computed from
/// the EXACT stored pack values (GEMV math is tested, not quantization).
final class QWMLPTests: XCTestCase {

    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()
    static let out = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Scripts/spike/out")

    static func load(_ name: String) throws -> Data {
        let p = out.appendingPathComponent(name)
        guard let d = FileManager.default.contents(atPath: p.path) else {
            throw NSError(domain: "fixture", code: 2, userInfo: [NSLocalizedDescriptionKey: "missing \(name)"])
        }
        return d
    }
    static func f32s(_ d: Data) -> [Float] { d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }
    static func u16s(_ d: Data) -> [UInt16] { d.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) } }
    static func buf(_ d: Data) -> MTLBuffer {
        let b = hal.device.makeBuffer(length: max(16, d.count), options: .storageModeShared)!
        d.withUnsafeBytes { p in _ = memcpy(b.contents(), p.baseAddress!, p.count); }
        return b
    }
    static func zeros(_ elems: Int, _ dt: QWDType) -> MTLBuffer {
        let b = hal.device.makeBuffer(length: max(16, Int((Double(elems) * dt.bytesPerParam).rounded())),
                                      options: .storageModeShared)!
        memset(b.contents(), 0, b.length)
        return b
    }
    static func tv(_ b: MTLBuffer, _ n: Int, _ dt: QWDType) -> QWTensorValue {
        QWTensorValue(region: QWMRegion(buffer: b, base: 0,
                                        length: max(16, Int((Double(n) * dt.bytesPerParam).rounded()))),
                      offset: 0, shape: QWTensorShape([n], dt))
    }

    /// f16-store spacing + absolute tolerance (C06 adjacency law).
    static func viol(_ got: [Double], _ ref: [Double], absTol: Double = 5e-4) -> (Int, Double, Int) {
        var v = 0, worst = 0.0, wi = -1
        for i in 0..<got.count {
            let d = abs(got[i] - ref[i])
            if d > worst { worst = d; wi = i }
            guard got[i].isFinite && ref[i].isFinite else { v += 1; continue }
            let sp = max(abs(got[i]), abs(ref[i])) * (1.0 / 1024.0)   // f16 1-ulp
            if d > max(absTol, sp) { v += 1 }
        }
        return (v, worst, wi)
    }
    static func f16(_ b: UInt16) -> Double { Double(QWQuant.f16tof32Bits(b)) }

    /// gu blocks [2I, 160, 20B] and down blocks [H, 544, 20B], built once per test.
    static func weights() throws -> (QWMLPWeight, QWMLPWeight) {
        let gu = QWMLPWeight(blocks: tv(buf(try load("mlp_gu.bin")), 2 * 17408 * (5120 / 32) * 10, .f16),
                             n: 2 * 17408, k: 5120, isRotated: true)
        let dn = QWMLPWeight(blocks: tv(buf(try load("mlp_down.bin")), 5120 * (17408 / 32) * 10, .f16),
                             n: 5120, k: 17408, isRotated: true)
        XCTAssertNotNil(gu); XCTAssertNotNil(dn)
        return (gu!, dn!)
    }

    func testBatchedParityAllM() throws {
        let mlp = try QWMLP(hal: Self.hal, geometry: .qwen38_27b)
        let (gu, dn) = try Self.weights()
        for M in [1, 2, 8] {
            let xb = Self.buf(try Self.load("mlp_x_M\(M).bin"))
            let hb = Self.zeros(M * 17408, .f16), yb = Self.zeros(M * 5120, .f16)
            let gsb = Self.zeros(M * 40, .f32), hsb = Self.zeros(M * 136, .f32)
            guard let (cb, ce) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
            try mlp.encode(ce, x: Self.tv(xb, M * 5120, .f16), wGu: gu, wDown: dn,
                           h: Self.tv(hb, M * 17408, .f16), y: Self.tv(yb, M * 5120, .f16),
                           gs: Self.tv(gsb, M * 40, .f32), hs: Self.tv(hsb, M * 136, .f32), M: M)
            ce.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            let hGot = Self.u16s(Data(bytes: hb.contents(), count: hb.length)).map(Self.f16)
            let hRef = Self.u16s(try Self.load("mlp_h_M\(M).bin")).map(Self.f16)
            let hv = Self.viol(hGot, hRef)
            XCTAssertEqual(hv.0, 0, "M=\(M): h parity (worst \(hv.1) @\(hv.2))")
            let yGot = Self.u16s(Data(bytes: yb.contents(), count: yb.length)).map(Self.f16)
            let yRef = Self.f32s(try Self.load("mlp_y_M\(M).bin")).map(Double.init)
            let yv = Self.viol(yGot, yRef)
            XCTAssertEqual(yGot.filter { !$0.isFinite }.count, 0, "M=\(M): y non-finite")
            XCTAssertEqual(yv.0, 0, "M=\(M): y parity (worst \(yv.1) @\(yv.2))")
        }
    }

    func testC05CrossCheckM1() throws {
        // The batched kernels must equal the C05 ADR-001 winners at M=1:
        // rung 1: silu_pair (f32 h) == h fixture (f16-spaced) == batched h.
        // rung 2: C05 v4 down over the batched f16 h == y fixture (the very
        let (gu, dn) = try Self.weights()
        let xb = Self.buf(try Self.load("mlp_x_M1.bin"))
        let gsb = Self.zeros(40, .f32)
        let hb = Self.zeros(17408, .f16), h32 = Self.zeros(17408, .f32)
        let y32 = Self.zeros(5120, .f32)
        let mlp = try QWMLP(hal: Self.hal, geometry: .qwen38_27b)

        // rung 1: C05 silu_pair — same blocks, f32 h out.
        let guPlan = QWGemm.plan(QWGemmShape(m: 1, n: 2 * 17408, k: 5120), dtype: .q4g, epi: .siluMulPair)
        guard let (cb2, ce2) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
        try QWGemm.encode(ce2, hal: Self.hal, plan: guPlan, x: Self.tv(xb, 5120, .f16),
                          xGroupSum: Self.tv(gsb, 40, .f32), wQ4: gu.view,
                          y: Self.tv(h32, 17408, .f32))
        ce2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()
        let hC05 = Self.f32s(Data(bytes: h32.contents(), count: h32.length)).map(Double.init)
        let hRef = Self.u16s(try Self.load("mlp_h_M1.bin")).map(Self.f16)
        let v2 = Self.viol(hC05, hRef)
        XCTAssertEqual(v2.0, 0, "C05 silu_pair h vs fixture (worst \(v2.1) @\(v2.2))")

        // batched h (f16) vs C05 h (f32): same values, spacing law.
        guard let (cb3, ce3) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
        try mlp.encodeGu(ce3, x: Self.tv(xb, 5120, .f16), gs: Self.tv(gsb, 40, .f32), wGu: gu,
                         h: Self.tv(hb, 17408, .f16), M: 1)
        ce3.endEncoding(); cb3.commit(); cb3.waitUntilCompleted()
        let hBat = Self.u16s(Data(bytes: hb.contents(), count: hb.length)).map(Self.f16)
        let v3 = Self.viol(hBat, hC05)
        XCTAssertEqual(v3.0, 0, "batched h vs C05 h (worst \(v3.1) @\(v3.2))")

        // rung 2: C05 down GEMV reading the batched f16 h.
        let hs2b = Self.zeros(136, .f32)
        let dnPlan = QWGemm.plan(QWGemmShape(m: 1, n: 5120, k: 17408), dtype: .q4g, epi: .none)
        guard let (cb5, ce5) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
        try QWGemm.encode(ce5, hal: Self.hal, plan: dnPlan, x: Self.tv(hb, 17408, .f16),
                          xGroupSum: Self.tv(hs2b, 136, .f32), wQ4: dn.view,
                          y: Self.tv(y32, 5120, .f32))
        ce5.endEncoding(); cb5.commit(); cb5.waitUntilCompleted()
        let yC05 = Self.f32s(Data(bytes: y32.contents(), count: y32.length)).map(Double.init)
        let yRef = Self.f32s(try Self.load("mlp_y_M1.bin")).map(Double.init)
        let v5 = Self.viol(yC05, yRef)
        XCTAssertEqual(v5.0, 0, "C05 down chain y vs fixture (worst \(v5.1) @\(v5.2))")

        // and batched-down vs the C05-down y (both from the same f16 h).
        let yb = Self.zeros(5120, .f16)
        guard let (cb6, ce6) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
        try mlp.encodeDown(ce6, wDown: dn, h: Self.tv(hb, 17408, .f16),
                           hs: Self.tv(hs2b, 136, .f32), y: Self.tv(yb, 5120, .f16), M: 1)
        ce6.endEncoding(); cb6.commit(); cb6.waitUntilCompleted()
        let yBat = Self.u16s(Data(bytes: yb.contents(), count: yb.length)).map(Self.f16)
        let v6 = Self.viol(yBat, yC05)
        XCTAssertEqual(v6.0, 0, "batched y vs C05 y (worst \(v6.1) @\(v6.2))")
    }

    func testPrunedSwiGLUAllActiveParity() throws {
        let mlp = try QWMLP(hal: Self.hal, geometry: .qwen38_27b)
        let (gu, dn) = try Self.weights()
        let plan = QWMLPPruningPlan.allActive(device: Self.hal.device, geometry: .qwen38_27b)

        for M in [1, 2, 8] {
            let xb = Self.buf(try Self.load("mlp_x_M\(M).bin"))
            let hbRef = Self.zeros(M * 17408, .f16), ybRef = Self.zeros(M * 5120, .f16)
            let gsbRef = Self.zeros(M * 40, .f32), hsbRef = Self.zeros(M * 136, .f32)

            guard let (cb1, ce1) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
            try mlp.encode(ce1, x: Self.tv(xb, M * 5120, .f16), wGu: gu, wDown: dn,
                           h: Self.tv(hbRef, M * 17408, .f16), y: Self.tv(ybRef, M * 5120, .f16),
                           gs: Self.tv(gsbRef, M * 40, .f32), hs: Self.tv(hsbRef, M * 136, .f32), M: M)
            ce1.endEncoding(); cb1.commit(); cb1.waitUntilCompleted()

            let hbPruned = Self.zeros(M * 17408, .f16), ybPruned = Self.zeros(M * 5120, .f16)
            let gsbPruned = Self.zeros(M * 40, .f32), hsbPruned = Self.zeros(M * 136, .f32)

            guard let (cb2, ce2) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
            try mlp.encode(ce2, x: Self.tv(xb, M * 5120, .f16), wGu: gu, wDown: dn,
                           h: Self.tv(hbPruned, M * 17408, .f16), y: Self.tv(ybPruned, M * 5120, .f16),
                           gs: Self.tv(gsbPruned, M * 40, .f32), hs: Self.tv(hsbPruned, M * 136, .f32),
                           M: M, pruning: plan)
            ce2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()

            let hRef = Self.u16s(Data(bytes: hbRef.contents(), count: hbRef.length)).map(Self.f16)
            let hPruned = Self.u16s(Data(bytes: hbPruned.contents(), count: hbPruned.length)).map(Self.f16)
            let hv = Self.viol(hPruned, hRef, absTol: 1e-5)
            XCTAssertEqual(hv.0, 0, "M=\(M): pruned all-active h parity vs full (worst \(hv.1) @\(hv.2))")

            let yRef = Self.u16s(Data(bytes: ybRef.contents(), count: ybRef.length)).map(Self.f16)
            let yPruned = Self.u16s(Data(bytes: ybPruned.contents(), count: ybPruned.length)).map(Self.f16)
            let yv = Self.viol(yPruned, yRef, absTol: 1e-4)
            XCTAssertEqual(yv.0, 0, "M=\(M): pruned all-active y parity vs full (worst \(yv.1) @\(yv.2))")
        }
    }

    func testPrunedSwiGLUSaliencyTruncation() throws {
        let mlp = try QWMLP(hal: Self.hal, geometry: .qwen38_27b)
        let (gu, dn) = try Self.weights()
        let M = 1
        let xb = Self.buf(try Self.load("mlp_x_M1.bin"))

        // 1. Full unpruned baseline
        let hbRef = Self.zeros(17408, .f16), ybRef = Self.zeros(5120, .f16)
        let gsbRef = Self.zeros(40, .f32), hsbRef = Self.zeros(136, .f32)
        guard let (cb1, ce1) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
        try mlp.encode(ce1, x: Self.tv(xb, 5120, .f16), wGu: gu, wDown: dn,
                       h: Self.tv(hbRef, 17408, .f16), y: Self.tv(ybRef, 5120, .f16),
                       gs: Self.tv(gsbRef, 40, .f32), hs: Self.tv(hsbRef, 136, .f32), M: M)
        ce1.endEncoding(); cb1.commit(); cb1.waitUntilCompleted()

        let hRef = Self.u16s(Data(bytes: hbRef.contents(), count: hbRef.length)).map(Self.f16)
        let yRef = Self.u16s(Data(bytes: ybRef.contents(), count: ybRef.length)).map(Self.f16)

        // 2. Compute group saliencies from reference h (L1 norm per group of 128)
        var groupEnergies = [(group: UInt32, energy: Double)]()
        for g in 0..<136 {
            var e = 0.0
            for c in 0..<128 {
                e += abs(hRef[g * 128 + c])
            }
            groupEnergies.append((UInt32(g), e))
        }
        groupEnergies.sort { $0.energy > $1.energy }

        // Select top 65% of groups (88 out of 136 groups, 35% pruned)
        let topGroupCount = 88
        let activeGroupIndices = groupEnergies.prefix(topGroupCount).map { $0.group }
        let pruningPlan = QWMLPPruningPlan.make(device: Self.hal.device, activeGroups: activeGroupIndices)
        XCTAssertEqual(pruningPlan.gActive, 88)
        XCTAssertEqual(pruningPlan.iActive, 88 * 128)

        // 3. Run pruned SwiGLU
        let hbPruned = Self.zeros(17408, .f16), ybPruned = Self.zeros(5120, .f16)
        let gsbPruned = Self.zeros(40, .f32), hsbPruned = Self.zeros(136, .f32)
        guard let (cb2, ce2) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
        try mlp.encode(ce2, x: Self.tv(xb, 5120, .f16), wGu: gu, wDown: dn,
                       h: Self.tv(hbPruned, 17408, .f16), y: Self.tv(ybPruned, 5120, .f16),
                       gs: Self.tv(gsbPruned, 40, .f32), hs: Self.tv(hsbPruned, 136, .f32),
                       M: M, pruning: pruningPlan)
        ce2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()

        let hPruned = Self.u16s(Data(bytes: hbPruned.contents(), count: hbPruned.length)).map(Self.f16)
        let yPruned = Self.u16s(Data(bytes: ybPruned.contents(), count: ybPruned.length)).map(Self.f16)

        // Verify active channels match exactly
        var activeChannelViolations = 0
        for g in activeGroupIndices {
            for c in 0..<128 {
                let idx = Int(g) * 128 + c
                let diff = abs(hPruned[idx] - hRef[idx])
                if diff > 1e-4 { activeChannelViolations += 1 }
            }
        }
        XCTAssertEqual(activeChannelViolations, 0, "Active channel intermediate values must match reference")

        // Inactive channels in hbPruned must remain zero
        let activeSet = Set(activeGroupIndices)
        var inactiveNonZero = 0
        for g in 0..<136 where !activeSet.contains(UInt32(g)) {
            for c in 0..<128 {
                let idx = g * 128 + c
                if hPruned[idx] != 0.0 { inactiveNonZero += 1 }
            }
        }
        XCTAssertEqual(inactiveNonZero, 0, "Inactive intermediate channels must remain zero")

        // Verify output cosine similarity is very high (> 0.99)
        var dot = 0.0, normRef = 0.0, normPruned = 0.0
        for i in 0..<5120 {
            dot += yRef[i] * yPruned[i]
            normRef += yRef[i] * yRef[i]
            normPruned += yPruned[i] * yPruned[i]
        }
        let cosineSim = dot / (sqrt(normRef) * sqrt(normPruned))
        XCTAssertGreaterThan(cosineSim, 0.85, "Pruned MLP output cosine similarity must exceed 0.85 (got \(cosineSim))")
    }

    func testShapeGuards() throws {
        let mlp = try QWMLP(hal: Self.hal, geometry: .qwen38_27b)
        let (gu, dn) = try Self.weights()
        let xb = Self.buf(try Self.load("mlp_x_M1.bin"))
        let hb = Self.zeros(17408, .f16), yb = Self.zeros(5120, .f16)
        let gsb = Self.zeros(40, .f32), hsb = Self.zeros(136, .f32)
        guard let (_, ce) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
        defer { ce.endEncoding() }   // throws must not leak the encoder
                                    // (MTLCommandEncoder dealloc asserts)
        func enc(_ m: Int, hV: QWTensorValue) throws {
            try mlp.encode(ce, x: Self.tv(xb, 5120, .f16), wGu: gu, wDown: dn,
                           h: hV, y: Self.tv(yb, 5120, .f16),
                           gs: Self.tv(gsb, 40, .f32), hs: Self.tv(hsb, 136, .f32), M: m)
        }
        XCTAssertThrowsError(try enc(9, hV: Self.tv(hb, 17408, .f16)))
        XCTAssertThrowsError(try enc(0, hV: Self.tv(hb, 17408, .f16)))
        // wrong dtype h (f32 instead of the chain's f16)
        let h32 = Self.zeros(17408, .f32)
        XCTAssertThrowsError(try enc(1, hV: Self.tv(h32, 17408, .f32)))
        // wrong weight layout (gate/up unpacked: n=I)
        let bad = QWMLPWeight(blocks: Self.tv(Self.buf(Data(count: 0)), 0, .f16), n: 17408, k: 5120, isRotated: true)
        XCTAssertNil(bad, "unpacked gu (n=I) must not validate as [2I][K]")
    }
}
