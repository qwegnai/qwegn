import XCTest
import Metal
@testable import QWGEMM
import QWCore
import QWHAL
import QWAlloc

/// C05 gate (specs/05-qwgemm.md):
///  1. engine kernels reproduce the numpy q4 dequant-GEMV reference (fixtures
///     shared with the spike race — same bytes, same math, now through the
final class QWGEMMTests: XCTestCase {

    static let K = 5120
    static let fixtureRoot: URL = {
        var d = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<8 {
            let c = d.appendingPathComponent("Scripts/spike/out")
            if FileManager.default.fileExists(atPath: c.path) { return c }
            d = d.deletingLastPathComponent()
        }
        fatalError("Scripts/spike/out not found — run Scripts/oracle/gen_s1_q4_fixtures.py")
    }()
    static func fixture(_ name: String) -> Data {
        try! Data(contentsOf: fixtureRoot.appendingPathComponent(name))
    }

    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()   // finds Sources/QWMetal/kernel.metallib

    struct QFixture {
        let blocks: Data, x: Data, gs: Data, y: Data
        let n: Int
    }
    static func loadFixture(_ n: Int) -> QFixture {
        func need(_ suffix: String, _ exp: Int) -> Data {
            let d = fixture("s1_q4_N\(n)\(suffix)")
            precondition(d.count == exp, "fixture \(suffix) size \(d.count) ≠ \(exp)")
            return d
        }
        let g32 = K / 32
        return QFixture(blocks: need(".bin", n * g32 * 20), x: need("_x.bin", K * 2),
                        gs: need("_gs.bin", 40 * 4), y: need("_y.bin", n * 4), n: n)
    }

    /// Everything in ONE arena; run the engine door; return GPU milliseconds
    /// (median of `iters`, warmup 10) and y as [Float].
    struct RunResult { let y: [Float]; let p50ms: Double }
    func runEngine(_ f: QFixture, epi: QWEpilogue, iters: Int = 0) throws -> RunResult {
        let rows = epi == .siluMulPair ? f.n / 2 : f.n
        let g32 = Self.K / 32
        let arena = try QWArena(hal: Self.hal,
                                weights: f.blocks.count + 256,
                                activations: rows * 4 + Self.K * 2 + (Self.K / 128) * 4 + 3 * 256,
                                kv: 0, gdnStates: 0)
        let wreg = arena.weightsRegion(), act = arena.activationsRegion()
        let blkV = try wreg.allocate(QWTensorShape([f.n * g32 * 10], .f16))
        arena.copyIn(blkV, from: f.blocks)

        let xV  = try act.allocate(QWTensorShape([Self.K], .f16))
        let gsV = try act.allocate(QWTensorShape([Self.K / 128], .f32))
        let yV  = try act.allocate(QWTensorShape([rows], .f32))
        arena.copyIn(xV, from: f.x); arena.copyIn(gsV, from: f.gs)

        guard let w = QWWeightView(blocks: blkV, n: f.n, k: Self.K, isRotated: true) else {
            return try XCTUnwrap(nil as RunResult?, "weight view rejected fixture")
        }
        let plan = QWGemm.plan(QWGemmShape(m: 1, n: f.n, k: Self.K), dtype: .q4g, epi: epi)
        var times: [Double] = []
        let total = max(1, 10 + iters)
        for i in 0..<total {
            guard let (cb, enc) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
            try QWGemm.encode(enc, hal: Self.hal, plan: plan, x: xV, xGroupSum: gsV, wQ4: w, y: yV)
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            if i >= 10 { times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000) }
        }
        times.sort()
        let y = Array(UnsafeBufferPointer(
            start: arena.cpuPointer(of: yV).bindMemory(to: Float.self, capacity: rows), count: rows))
        return RunResult(y: y, p50ms: times.isEmpty ? -1 : times[times.count / 2])
    }

    // MARK: gates

    func testDispatchTableIsPureTable() {
        let p1 = QWGemm.plan(QWGemmShape(m: 1, n: 10240, k: 5120), dtype: .q4g, epi: .none)
        XCTAssertEqual(p1.backend, .handQ4gV4)
        XCTAssertEqual(p1.threadsPerTG, 32)
        let p2 = QWGemm.plan(QWGemmShape(m: 1, n: QWModelGeometry.qwen38_27b.vocab, k: 5120), dtype: .q4g, epi: .none)
        XCTAssertEqual(p2.family, .f2LmHeadGemv)
        XCTAssertEqual(p2.backend, .handQ4gV4)
        let p3 = QWGemm.plan(QWGemmShape(m: 1, n: 12288, k: 5120), dtype: .q4g, epi: .siluMulPair)
        XCTAssertEqual(p3.backend, .handQ4gSiluPair)
        let p4 = QWGemm.plan(QWGemmShape(m: 1024, n: 12288, k: 5120), dtype: .q4g, epi: .none)
        XCTAssertEqual(p4.backend, .mpsGraphBmmF16)  // F3 — EXP-1 race winner (bench_f3 2026-10-05)
        XCTAssertEqual(QWGemm.plan(QWGemmShape(m: 1024, n: 12288, k: 5100), dtype: .q4g,
                                   epi: .none).backend, .blocked)  // ragged K stays blocked
        // f16 escape hatch resolves to its own kernel
        let p5 = QWGemm.plan(QWGemmShape(m: 1, n: 6144, k: 5120), dtype: .f16, epi: .none)
        XCTAssertEqual(p5.kernel, "qwg_gemv_f16_v4")
    }

    func testEngineGemvMatchesNumpyQ4Ref() throws {
        for n in [6144, 17408] {
            let f = Self.loadFixture(n)
            let r = try runEngine(f, epi: .none)
            let ref = f.y.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            var bad = 0, maxabs = 0.0
            for i in 0..<n where abs(r.y[i] - ref[i]) > 2e-2 {
                bad += 1; maxabs = max(maxabs, abs(Double(r.y[i] - ref[i])))
            }
            XCTAssertEqual(bad, 0, "engine q4g GEMV N=\(n): \(bad) off ref (max \(maxabs))")
        }
    }

    func testBlockedFamiliesThrowCleanly() throws {
        let f = Self.loadFixture(6144)
        let g32 = Self.K / 32
        let arena = try QWArena(hal: Self.hal, weights: f.blocks.count &+ 768,
                                activations: 1 << 20, kv: 0, gdnStates: 0)
        let wreg = arena.weightsRegion(), act = arena.activationsRegion()
        let blkV = try wreg.allocate(QWTensorShape([f.n * g32 * 10], .f16))
        let xV  = try act.allocate(QWTensorShape([Self.K], .f16))
        let gsV = try act.allocate(QWTensorShape([Self.K / 128], .f32))
        let yV  = try act.allocate(QWTensorShape([f.n], .f32))
        guard let w = QWWeightView(blocks: blkV, n: f.n, k: Self.K, isRotated: true)
        else { return XCTFail("view") }
        let plan = QWGemm.plan(QWGemmShape(m: 1024, n: f.n, k: Self.K), dtype: .q4g, epi: .none)
        guard let (cb, enc) = Self.hal.begin() else { return XCTFail("hal begin") }
        XCTAssertThrowsError(try QWGemm.encode(enc, hal: Self.hal, plan: plan,
                                               x: xV, xGroupSum: gsV, wQ4: w, y: yV)) { err in
            guard case QWGemmError.f3Blocked = err else { return XCTFail("\(err)") }
        }
        Self.hal.end(cb, enc)
    }

    /// Gate 2 smoke (authoritative ≤5% lives in bench_gemm): the fused
    /// gate‖up path must not blow up vs plain on this box.
    // MARK: F5 skinny window (C16 race fixtures — Scripts/oracle/gen_f5_fixtures.py)

    struct F5Shape { let name: String, n: Int, k: Int }
    static let f5Shapes: [F5Shape] = [
        .init(name: "f5q",  n: 12288, k: 5120),   // q_proj  (Q‖gate)
        .init(name: "f5kv", n: 1024,  k: 5120),   // k/v_proj
        .init(name: "f5o",  n: 5120,  k: 6144),   // o_proj  (M2-only window)
        .init(name: "f5fc", n: 5120,  k: 10240),  // mtp fc  (v7: engine K ⇒ legal)
        .init(name: "f5gu", n: 34816, k: 5120),   // gate‖up (plain dot; silu raced in bench)
        .init(name: "f5dn", n: 5120,  k: 17408),  // down    (v7: x 209 KB ≤ 256 KB ⇒ legal)
        .init(name: "f5hd", n: 61440, k: 5120),   // lm_head surrogate
    ]
    static func f5Data(_ s: String) -> Data { fixture(s) }

    /// C06/C07 f16 binade spacing (same law/impl as QWMTPTests.f16Spacing).
    static func f16Spacing(_ v: Float) -> Double {
        let e = Int((v.bitPattern >> 23) & 0xFF) - 127 - 10
        return e >= 0 ? Double(1 << e) : 1.0 / Double(1 << -e)
    }

    /// Parity law (C14 pattern): absolute ≤5e-4 OR f16-benade adjacency
    /// (values here are f32 y but yref came from f16-rounded weights +
    /// f32 accum; kernel order differs per-lane ⇒ last-ulp deltas land
    func testF5SkinnyParityAndWindow() throws {
        func viol(_ got: [Float], _ ref: [Float]) -> (Int, Double) {
            var v = 0; var worst = 0.0
            for i in 0..<got.count {
                let d = abs(Double(got[i]) - Double(ref[i]))
                if d > worst { worst = d }
                if !(d <= 5e-4 || d <= min(Self.f16Spacing(got[i]), Self.f16Spacing(ref[i]))) { v += 1 }
            }
            return (v, worst)
        }
        for sh in Self.f5Shapes {
            for M in [2, 3] {
                let plan = QWGemm.plan(QWGemmShape(m: M, n: sh.n, k: sh.k), dtype: .q4g, epi: .none)
                // v7 window: M∈{2,3}, x = M·K·2 B ≤ maxConstantBytes (256 KB
                // clamp) ⇒ EVERY engine shape here is legal (worst 209 KB).
                let legal = M * sh.k * 2 <= QWGemm.maxConstantBytes
                if !legal {
                    XCTAssertEqual(plan.backend, .blocked, "\(sh.name) M=\(M) must block (window)")
                    continue
                }
                XCTAssertEqual(plan.backend, .handQ4gSkinny, "\(sh.name) M=\(M)")
                let blocks = Self.f5Data("\(sh.name).bin")
                let x      = Self.f5Data("\(sh.name)_x_M\(M).bin")
                let gs     = Self.f5Data("\(sh.name)_gs_M\(M).bin")
                let y      = Self.f5Data("\(sh.name)_y_M\(M).bin")
                let g32    = sh.k / 32
                XCTAssertEqual(blocks.count, sh.n * g32 * 20)
                let arena = try QWArena(hal: Self.hal, weights: blocks.count + 256,
                                        activations: x.count + gs.count + y.count + 3 * 256,
                                        kv: 0, gdnStates: 0)
                let wreg = arena.weightsRegion(), act = arena.activationsRegion()
                let blkV = try wreg.allocate(QWTensorShape([sh.n * g32 * 10], .f16))
                arena.copyIn(blkV, from: blocks)
                let xV  = try act.allocate(QWTensorShape([M * sh.k], .f16))
                let gsV = try act.allocate(QWTensorShape([M * sh.k / 128], .f32))
                let yV  = try act.allocate(QWTensorShape([M * sh.n], .f32))
                arena.copyIn(xV, from: x); arena.copyIn(gsV, from: gs)
                let wOpt = QWWeightView(blocks: blkV, n: sh.n, k: sh.k, isRotated: true)
                guard let w = wOpt else {
                    XCTFail("weight view rejected \(sh.name)"); continue
                }
                guard let (cb, enc) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
                try QWGemm.encode(enc, hal: Self.hal, plan: plan, x: xV, xGroupSum: gsV, wQ4: w, y: yV)
                Self.hal.end(cb, enc); cb.waitUntilCompleted()
                let got = Array(UnsafeBufferPointer(
                    start: arena.cpuPointer(of: yV).bindMemory(to: Float.self, capacity: M * sh.n),
                    count: M * sh.n))
                let ref = y.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                let (v, worst) = viol(got, ref)
                XCTAssertEqual(v, 0, "F5 \(sh.name) M=\(M): \(v) violations, worst \(worst)")
            }
        }
    }

    func testF5BlockedThrowsCleanly() throws {
        // window-outside shapes throw badWeights (NOT f3Blocked) — the C14
        // regime message; encode allocates nothing before the throw.
        // Sizes must satisfy QWWeightView (n·G32·20 block bytes) so the view
        let n = 5120, k = 17408, g32 = k / 32, g128 = k / 128, M = 4   // M≥4: F1×M is the measured winner
        let arena = try QWArena(hal: Self.hal, weights: n * g32 * 20 + 256,
                                activations: M * k * 2 + M * g128 * 4 + n * M * 4 + 4 * 256,
                                kv: 0, gdnStates: 0)
        let act = arena.activationsRegion(), wreg = arena.weightsRegion()
        let xV  = try act.allocate(QWTensorShape([M * k], .f16))      // M=4 sized
        let gsV = try act.allocate(QWTensorShape([M * g128], .f32))
        let yV  = try act.allocate(QWTensorShape([M * n], .f32))
        let blkV = try wreg.allocate(QWTensorShape([n * g32 * 10], .f16))
        let wOpt = QWWeightView(blocks: blkV, n: n, k: k, isRotated: true)
        guard let w = wOpt else { XCTFail("weight view rejected"); return }
        let plan = QWGemm.plan(QWGemmShape(m: M, n: n, k: k), dtype: .q4g, epi: .none)
        XCTAssertEqual(plan.backend, .blocked)     // M=4: outside the F5 window
        guard let (cb, enc) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
        XCTAssertThrowsError(try QWGemm.encode(enc, hal: Self.hal, plan: plan, x: xV,
                                               xGroupSum: gsV, wQ4: w, y: yV)) { e in
            XCTAssertTrue(e is QWGemmError, "wrong error type: \(e)")
            if case QWGemmError.badWeights(let m)? = e as? QWGemmError {
                XCTAssertTrue(m.contains("F5 window"), "message: \(m)")
            } else { XCTFail("expected badWeights, got \(e)") }
        }
        Self.hal.end(cb, enc)   // end() ALREADY commits (QWHal.end = endEncoding+commit)
        cb.waitUntilCompleted() // the throw preceded any bind — encoder closed clean
    }

    func testSiluPairEpilogueOverheadSmoke() throws {
        let n = 17408
        let f = Self.loadFixture(n)
        let plain = try runEngine(f, epi: .none, iters: 30)
        let pair  = try runEngine(f, epi: .siluMulPair, iters: 30)
        XCTAssertEqual(pair.y.count, n / 2)
        print(String(format: "silu_pair smoke: plain %.3f ms vs fused %.3f ms (Δ %.1f%%)",
                     plain.p50ms, pair.p50ms, (pair.p50ms / plain.p50ms - 1) * 100))
        XCTAssertLessThan(pair.p50ms, plain.p50ms * 1.25,
                          "fused silu_pair smoke margin blown (>25% over plain)")
    }
}
