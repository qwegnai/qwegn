// QWMTPTests — C16 golden bindings. The protocol core (accept / residual
// resample / draw-count law) replays the torch golden bit-exactly before
// any GPU leg exists; the scan law and draw law bind directly.
import XCTest
import Metal
@testable import QWMTP
@testable import QWRope   // internal(tableTensor:hal:nullSlab:) — the fixture
                           // table IS the oracle's rows; QWRopeTables(rows:, geometry: .qwen38_27b)
                           // computes its own (real-position) table.
// The fc GEMV leg routes through the shipped C05 door (QWGemm) with the
// C15 weight-view type; both are transitive deps of QWMTP by design —
// the draft leg reuses the engine's proven planes, never a private copy.
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWHead
import QWNorm
import QWQuant
import QWRope
import QWKV
import QWAttn
import QWMLP

final class QWMTPTests: XCTestCase {

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
    static func f64s(_ d: Data) -> [Double] { d.withUnsafeBytes { Array($0.bindMemory(to: Double.self)) } }
    static func i32s(_ d: Data) -> [Int32] { d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) } }

    // GPU-leg helpers — identical to the C15 harness (same buffers, same
    // QWTensorValue view; the fc leg must ride the proven door).
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()
    static func f32s(_ d: Data) -> [Float] { d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }
    static func f16s(_ d: Data) -> [UInt16] { d.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) } }
    static func readF(_ b: [UInt16]) -> [Float] { b.map { QWQuant.f16tof32Bits($0) } }
    static func u16row(_ b: MTLBuffer, _ byteOffset: Int, _ n: Int) -> [UInt16] {
        let p = b.contents().advanced(by: byteOffset).assumingMemoryBound(to: UInt16.self)
        return (0..<n).map { p[$0] }
    }
    /// C06/C07 parity law, one element: gap ≤ absTol, OR ≤ 1-ulp of the
    /// f16 BINADE (spacing(v) = 2^(exp(v)−10), the C07 parityViolations
    /// implementation — f16 ulp, NOT Double.ulpOfOne; values are exactly
    static func f16Spacing(_ v: Float) -> Double {
        let e = Int((v.bitPattern >> 23) & 0xFF) - 127 - 10
        return e >= 0 ? Double(1 << e) : 1.0 / Double(1 << -e)
    }
    static func adjacent(_ got: Float, _ ref: Float) -> Bool {
        guard got.isFinite, ref.isFinite else { return got == ref }
        let d = abs(Double(got) - Double(ref))
        if d <= 1e-3 { return true }
        return d <= min(f16Spacing(got), f16Spacing(ref))
    }
    static func buf(_ d: Data) -> MTLBuffer {
        let b = hal.device.makeBuffer(length: max(16, d.count), options: .storageModeShared)!
        d.withUnsafeBytes { p in _ = memcpy(b.contents(), p.baseAddress!, p.count) }
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

    struct Geo: Decodable { let H, KVH, HD, NL, V, ISZ, P, N, R: Int }
    struct RoundsFile: Decodable { let uni_consumed: Int
        enum CodingKeys: String, CodingKey { case uni_consumed } }

    /// Full golden replay: per round feed drafts + p_d + p_t + the exact
    /// uniform window; accepted / finalToken / nDrawn must match the
    /// golden EXACTLY and the stream must end fully consumed.
    func testGoldenReplay() throws {
        let rounds = try Self.i32s(Self.load("mtpA_rounds.bin"))      // [R,N]
        let accG = try Self.i32s(Self.load("mtpA_accepted.bin"))
        let commG = try Self.i32s(Self.load("mtpA_committed.bin"))
        let drawnG = try Self.i32s(Self.load("mtpA_drawn.bin"))
        let uni = try Self.f64s(Self.load("mtpA_uni.bin"))
        let pd = try Self.f64s(Self.load("mtpA_pd.bin"))              // [R,N,V]
        let pt = try Self.f64s(Self.load("mtpA_pt.bin"))              // [R,N+1,V]
        let geo = try JSONDecoder().decode(Geo.self, from: try Self.load("mtpA_geom.json"))
        let rf = try JSONDecoder().decode(RoundsFile.self, from: try Self.load("mtpA_rounds.json"))
        let (R, N, V) = (geo.R, geo.N, geo.V)
        XCTAssertEqual(rounds.count, R * N)
        XCTAssertEqual(uni.count, rf.uni_consumed)

        var ui = 0
        var accMismatch = 0, tokMismatch = 0
        for r in 0..<R {
            let dp = (0..<N).map { Array(pd[(r * N + $0) * V ..< (r * N + $0 + 1) * V]) }
            let tp = (0...N).map { Array(pt[(r * (N + 1) + $0) * V ..< (r * (N + 1) + $0 + 1) * V]) }
            let d = (0..<N).map { Int(rounds[r * N + $0]) }
            // STREAM LAW (gen_mtp_fixtures docstring): per round the draft
            // leg consumed N uniforms FIRST (sampling d_1..d_N), then the
            // accept leg consumed nDrawn. The replay skips the draft slots
            let res = QWMTP.acceptRound(draftProbs: dp, targetRows: tp, drafts: d,
                                        uniforms: uni, offset: ui + N)
            if res.accepted != Int(accG[r]) { accMismatch += 1 }
            if res.finalToken != Int(commG[r]) { tokMismatch += 1 }
            XCTAssertEqual(res.nDrawn, Int(drawnG[r]), "draws law r\(r)")
            ui += N + res.nDrawn           // draft slots + accept slots
        }
        XCTAssertEqual(accMismatch, 0, "accept-lens vs golden")
        XCTAssertEqual(tokMismatch, 0, "committed tokens vs golden")
        XCTAssertEqual(ui, uni.count, "uniform stream fully consumed exactly")
    }

    /// Draw-count law property: N=2 ⇒ acc+2 draws on reject, 3 on all-
    /// accept. Seeded random rows.
    func testDrawCountLaw() {
        var seed: UInt64 = 42
        func u() -> Double { seed &+= 0x9E3779B97F4A7C15
            var z = seed &* 0xff51afd7ed558ccd
            z ^= z >> 33; z &*= 0xc4ceb9fe1a85ec53; z ^= z >> 33
            return Double(z >> 11) / Double(1 << 53) }
        func row(_ scale: Double) -> [Double] {
            var r = (0..<64).map { _ in (u() - 0.5) * scale }
            let mx = r.max()!
            r = r.map { exp($0 - mx) }
            let s = r.reduce(0, +)
            return r.map { $0 / s }
        }
        for _ in 0..<2000 {
            let pd = [row(2), row(2)]
            let pt = [row(2), row(2), row(2)]
            let d = [Int.random(in: 0..<64), Int.random(in: 0..<64)]
            let uni = (0..<4).map { _ in u() }
            let res = QWMTP.acceptRound(draftProbs: pd, targetRows: pt, drafts: d,
                                        uniforms: uni)
            XCTAssertEqual(res.nDrawn, res.accepted < 2 ? res.accepted + 2 : 3)
        }
    }

    /// Scan law: ascending strict-cumsum; boundary tokens + fp tail.
    func testScanLaw() {
        let p: [Double] = [0.5, 0.25, 0.125, 0.125]
        p.withUnsafeBufferPointer { b in
            XCTAssertEqual(QWMTP.scan(b, 0.4), 0)
            XCTAssertEqual(QWMTP.scan(b, 0.5), 1)          // strict: 0.5 is NOT < 0.5
            XCTAssertEqual(QWMTP.scan(b, 0.74), 1)
            XCTAssertEqual(QWMTP.scan(b, 0.76), 2)
            XCTAssertEqual(QWMTP.scan(b, 0.999), 3)
            XCTAssertEqual(QWMTP.scan(b, 0.9999999999999999), 3)
        }
    }

    // ── fc GEMV (draft leg): [5120,10240] q4g is an F1-family shape —
    //    ships on the C05 ADR-001 winner via QWGemm.encode (reuse law;
    //    QWHead.encode guards K==hidden and serves only the head plane).
    //    Fixtures from the golden round-0 state: x = cat[e‖h] f16, so the
    //    parity pins the [e-block ‖ h-block] wiring order too — swapping
    //    halves changes every output (enorm/hnorm weights differ). ──

    func testFcGemvParity() throws {
        let N = 5120, K = 10240
        let x = try Self.buf(Self.load("mtpfc_x.bin"))
        let yref = Self.f32s(try Self.load("mtpfc_y.bin"))
        let w = try XCTUnwrap(QWHeadWeight(
            blocks: Self.tv(Self.buf(try Self.load("mtpfc_w.bin")), N * (K / 32) * 10, .f16),
            n: N, k: K, isRotated: true))
        let plan = QWGemm.plan(QWGemmShape(m: 1, n: N, k: K), dtype: .q4g, epi: .none)
        XCTAssertEqual(plan.backend, .handQ4gV4, "fc must route the F1 winner")

        let y = Self.zeros(N, .f32)
        guard let (cb, enc) = Self.hal.begin() else { XCTFail("cmdbuffer"); return }
        try QWGemm.encode(enc, hal: Self.hal, plan: plan, x: Self.tv(x, K, .f16),
                          wQ4: w.view, y: Self.tv(y, N, .f32))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let got = y.contents().bindMemory(to: Float.self, capacity: N)
        var worst = 0.0, wi = -1
        for i in 0..<N {
            let d = abs(Double(got[i]) - Double(yref[i]))
            if d > worst { worst = d; wi = i }
        }
        XCTAssertLessThanOrEqual(worst, 1e-3, "fc GEMV: worst \(worst) @\(wi)")
    }

    // ── LEG 2a: the chain's seam + norm rungs through the SHIPPED doors.
    // The GEMV doors emit f32 (accumulate law); every downstream door takes
    // f16 storage — the seam cast is an explicit launch, never a CPU
    func testChainRungs() throws {
        func ref(_ f: String) throws -> [Float] {           // f16-stored ref
            Self.f16s(try Self.load(f)).map { QWQuant.f16tof32Bits($0) }
        }
        var viol = (cast: 0, enorm: 0, hnorm: 0)

        // rung 0: f32 → f16 seam cast
        let src = Self.buf(try Self.load("mtpc_zprev.bin"))            // f32 [5120]
        let dst = Self.zeros(5120, .f16)
        guard let (cb, enc) = Self.hal.begin() else { XCTFail("cmdbuffer"); return }
        try QWGemm.encodeCast(enc, hal: Self.hal, x32: Self.tv(src, 5120, .f32),
                              y16: Self.tv(dst, 5120, .f16))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let castGot = Self.readF(Self.u16row(dst, 0, 5120))
        let castRef = try ref("mtpc_cast_f16.bin")
        for i in 0..<5120 where !Self.adjacent(castGot[i], castRef[i]) { viol.cast += 1 }

        // rung 1: enorm(embed row f16) — plain rmsnorm door
        func norm(_ inF: QWTensorValue, _ wF: String, _ refF: String, _ variant: QWNormVariant) throws -> Int {
            let g = Self.buf(try Self.load(wF))
            let out = Self.zeros(5120, .f16)
            guard let (cb, enc) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
            try QWNorm.encode(enc, hal: Self.hal, variant: variant,
                              x: inF, gamma: Self.tv(g, 5120, .f16), out: Self.tv(out, 5120, .f16), geometry: .qwen38_27b)
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            let got = Self.readF(Self.u16row(out, 0, 5120)), r = try ref(refF)
            var v = 0
            for i in 0..<5120 where !Self.adjacent(got[i], r[i]) { v += 1 }
            return v
        }
        let e = Self.buf(try Self.load("mtpc_e.bin"))
        viol.enorm = try norm(Self.tv(e, 5120, .f16), "mtpc_enorm_w.bin", "mtpc_enorm_ref.bin", .rmsnorm)
        // rung 2: hnorm consumes the CAST output (in rotated H_32 basis via .rmsnormFromRot)
        let z16 = QWTensorValue(region: QWMRegion(buffer: dst, base: 0, length: 10240),
                                offset: 0, shape: QWTensorShape([5120], .f16))
        viol.hnorm = try norm(z16, "mtpc_hnorm_w.bin", "mtpc_hnorm_ref.bin", .rmsnormFromRot)

        XCTAssertEqual(viol.cast, 0, "seam cast adjacency")
        XCTAssertEqual(viol.enorm, 0, "enorm adjacency")
        XCTAssertEqual(viol.hnorm, 0, "hnorm adjacency (fed by the cast seam)")
    }

    // ── LEG 2b: fc → cast → ln1 → q/k GEMV → norm+rope door. Device-only
    // seams (each door's f16 output feeds the next door's f16 input with no
    // CPU bounce); the fixture chain (gen_mtp_chain2_fixtures) mirrors the
    func testDraftChain2b() throws {
        func ref16(_ f: String) throws -> [Float] { Self.readF(Self.f16s(try Self.load(f))) }
        func ref32(_ f: String) throws -> [Float] { Self.f32s(try Self.load(f)) }
        func plane(_ base: String, _ n: Int, _ k: Int) throws -> QWWeightView {
            try XCTUnwrap(QWHeadWeight(
                blocks: Self.tv(Self.buf(try Self.load("\(base).bin")), n * (k / 32) * 10, .f16),
                n: n, k: k, isRotated: true)).view
        }
        let H = 5120, K = 10240, QG = 12288, KK = 1024

        // rung 3: fc GEMV — proj (f16 fixture) → x1 f32
        let proj = Self.buf(try Self.load("mtpc_proj.bin"))
        let gs = Self.zeros(K / 128, .f32)
        let x1_32 = Self.zeros(H, .f32), x1_16 = Self.zeros(H, .f16)
        guard let (cb, enc) = Self.hal.begin() else { XCTFail("cmdbuffer"); return }
        try QWGemm.encode(enc, hal: Self.hal,
                          plan: QWGemm.plan(QWGemmShape(m: 1, n: H, k: K), dtype: .q4g, epi: .none),
                          x: Self.tv(proj, K, .f16), xGroupSum: Self.tv(gs, K / 128, .f32),
                          wQ4: try plane("mtpc_fc", H, K), y: Self.tv(x1_32, H, .f32))
        try QWGemm.encodeCast(enc, hal: Self.hal, x32: Self.tv(x1_32, H, .f32),
                              y16: Self.tv(x1_16, H, .f16))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let f32viol: (Int, Double) = {
            let g = Self.f32s(Data(bytes: x1_32.contents(), count: H * 4))
            let r = try! ref32("mtpc_x1_f32.bin")
            var v = 0, wst = 0.0
            for i in 0..<H { let d = abs(Double(g[i]) - Double(r[i])); if d > 1e-3 { v += 1 }; wst = max(wst, d) }
            return (v, wst)
        }()
        XCTAssertEqual(f32viol.0, 0, "fc GEMV f32 parity (worst \(f32viol.1))")

        // rung 4: ln1 (plain rmsnorm on the f16 seam output — device-fed)
        let x1h16 = Self.zeros(H, .f16)
        let ln1 = Self.buf(try Self.load("mtpc_ln1.bin"))
        guard let (cb2, e2) = Self.hal.begin() else { XCTFail("cmdbuffer"); return }
        try QWNorm.encode(e2, hal: Self.hal, variant: .rmsnorm, x: Self.tv(x1_16, H, .f16),
                          gamma: Self.tv(ln1, H, .f16), out: Self.tv(x1h16, H, .f16), geometry: .qwen38_27b)
        e2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()

        // rungs 5-7: q/k/v GEMVs read x1h from ln1's DEVICE output (no CPU
        // re-upload — the whole chain runs device-resident), cast each f32
        // to f16, then the rope door (per-head norm + partial rope, pos
        let tab = Self.buf(try Self.load("mtpc_rope_tab.bin"))            // f32 [3][64]
        let wqT = Self.buf(try Self.load("mtpc_wq.bin"))                   // [256]
        let wkT = Self.buf(try Self.load("mtpc_wk.bin"))                   // [1024]
        let rope = QWRopeApply(tableTensor: Self.tv(tab, 3 * 64, .f32),
                               hal: Self.hal, nullSlab: Self.zeros(16, .f16), geometry: .qwen38_27b)
        let gsq = Self.zeros(H / 128, .f32)
        let qg32 = Self.zeros(QG, .f32), qg16 = Self.zeros(QG, .f16)
        let k32 = Self.zeros(KK, .f32), k16 = Self.zeros(KK, .f16)
        let v32 = Self.zeros(KK, .f32), v16 = Self.zeros(KK, .f16)
        let qo = Self.zeros(QG, .f16), ko = Self.zeros(KK, .f16)
        let p1 = QWGemm.plan(QWGemmShape(m: 1, n: QG, k: H), dtype: .q4g, epi: .none)
        let p2 = QWGemm.plan(QWGemmShape(m: 1, n: KK, k: H), dtype: .q4g, epi: .none)
        guard let (cb3, e3) = Self.hal.begin() else { XCTFail("cmdbuffer"); return }
        try QWGemm.encode(e3, hal: Self.hal, plan: p1, x: Self.tv(x1h16, H, .f16),
                          xGroupSum: Self.tv(gsq, H / 128, .f32),
                          wQ4: try plane("mtpc_q", QG, H), y: Self.tv(qg32, QG, .f32))
        try QWGemm.encodeCast(e3, hal: Self.hal, x32: Self.tv(qg32, QG, .f32), y16: Self.tv(qg16, QG, .f16))
        try QWGemm.encode(e3, hal: Self.hal, plan: p2, x: Self.tv(x1h16, H, .f16),
                          xGroupSum: Self.tv(gsq, H / 128, .f32),
                          wQ4: try plane("mtpc_k", KK, H), y: Self.tv(k32, KK, .f32))
        try QWGemm.encodeCast(e3, hal: Self.hal, x32: Self.tv(k32, KK, .f32), y16: Self.tv(k16, KK, .f16))
        try QWGemm.encode(e3, hal: Self.hal, plan: p2, x: Self.tv(x1h16, H, .f16),
                          xGroupSum: Self.tv(gsq, H / 128, .f32),
                          wQ4: try plane("mtpc_v", KK, H), y: Self.tv(v32, KK, .f32))
        try QWGemm.encodeCast(e3, hal: Self.hal, x32: Self.tv(v32, KK, .f32), y16: Self.tv(v16, KK, .f16))
        // the rope door: q [12288] f16 | k [1024] f16 | wq | wk → qOut, kOut
        try rope.encodeQK(e3, hal: Self.hal,
                          q: Self.tv(qg16, QG, .f16), k: Self.tv(k16, KK, .f16),
                          wq: Self.tv(wqT, 256, .f16), wk: Self.tv(wkT, KK, .f16),
                          qOut: Self.tv(qo, QG, .f16), kOut: Self.tv(ko, KK, .f16),
                          normOut: nil, positions: [0])
        e3.endEncoding(); cb3.commit(); cb3.waitUntilCompleted()

        func viol(_ got: [Float], _ ref: [Float], _ tag: String, absTol: Double = 1e-2, maxUlp: Int = 64) {
            var v = 0, wst = 0.0, wi = -1, worstRatio = 0.0
            for i in 0..<got.count {
                let d = abs(Double(got[i]) - Double(ref[i]))
                if d > wst { wst = d; wi = i }
                guard got[i].isFinite, ref[i].isFinite else { v += 1; continue }
                if d <= absTol { continue }
                let sp = min(Self.f16Spacing(got[i]), Self.f16Spacing(ref[i]))
                worstRatio = max(worstRatio, d / sp)
                if d > sp * Double(maxUlp) { v += 1 }
            }
            XCTAssertEqual(v, 0, "\(tag): \(v) violations (worst \(wst) @\(wi), ratio \(worstRatio) ulp)")
        }
        viol(Self.readF(Self.u16row(qo, 0, QG)), try ref16("mtpc_qdoor_ref.bin"), "rope q door")
        viol(Self.readF(Self.u16row(ko, 0, KK)), try ref16("mtpc_knew_ref.bin"), "rope k door")
        viol(Self.readF(Self.u16row(v16, 0, KK)), try ref16("mtpc_vnew_ref.bin"), "vnew cast")
    }

    // ── LEG 2c: the draft step's BACK HALF, all shipped doors: C09 decode
    // on the MTP OWN pair-row cache (seed rows + the new row the kernel
    // writes itself) → o_proj GEMV → rmsnormResidual → C05 silu_pair + fwht32
    func testDraftChain2c() throws {
        func ref16(_ f: String) throws -> [Float] { Self.readF(Self.f16s(try Self.load(f))) }
        func ref32(_ f: String) throws -> [Float] { Self.f32s(try Self.load(f)) }
        func plane(_ base: String, _ n: Int, _ k: Int) throws -> QWWeightView {
            try XCTUnwrap(QWHeadWeight(
                blocks: Self.tv(Self.buf(try Self.load("\(base).bin")), n * (k / 32) * 10, .f16),
                n: n, k: k, isRotated: true)).view
        }
        func viol(_ got: [Float], _ ref: [Float], _ tag: String, absTol: Double = 1e-2, maxUlp: Int = 64) {
            var v = 0, wst = 0.0, wi = -1, ratio = 0.0
            for i in 0..<got.count {
                let d = abs(Double(got[i]) - Double(ref[i]))
                if d > wst { wst = d; wi = i }
                guard got[i].isFinite, ref[i].isFinite else { v += 1; continue }
                if d <= absTol { continue }
                let sp = min(Self.f16Spacing(got[i]), Self.f16Spacing(ref[i]))
                ratio = max(ratio, d / sp)
                if d > sp * Double(maxUlp) { v += 1 }
            }
            XCTAssertEqual(v, 0, "\(tag): \(v) violations (worst \(wst) @\(wi), ratio \(ratio) ulp)")
        }
        let H = 5120, QG = 12288, KK = 1024, I = 1024      // I = shrunk-real isize
        let P = 16, C = P - 1                              // C keys in cache; new row = C+1-th

        func env(_ ceiling: Int) throws -> (QWKVCache, QWArena, QWMRegion) {
            let pageBytes = QWKVCache.pageBytes(mode: .f16)
            let poolBytes = pageBytes * ((ceiling + 255) / 256) * 2
            let tables = QWKVCache.tableSlotBytes * 2 * 1 * QWKVCache.maxSeqs
            let pBytes = QWAttnDecode.decodePartialsBytes(geometry: .qwen38_27b)
            let arena = try QWArena(hal: Self.hal, weights: 0, activations: 0,
                                    kv: poolBytes + tables, gdnStates: pBytes)
            let cache = try QWKVCache(hal: Self.hal, arena: arena, layers: [3], mode: .f16, geometry: .qwen38_27b)
            let g = arena.gdnRegion()
            let off = try g.allocate(bytes: pBytes)
            return (cache, arena, QWMRegion(buffer: g.region.buffer, base: g.region.base + off, length: pBytes))
        }
        let seedK = Self.readF(Self.f16s(try Self.load("mtpc_seedkq_k.bin")))   // [4,15,256]
        let seedV = Self.readF(Self.f16s(try Self.load("mtpc_seedkq_v.bin")))
        func tokenRows(_ kv: [Float]) -> [UInt16] {
            var out = [UInt16](repeating: 0, count: (P - 1) * KK)
            for t in 0..<(P - 1) {
                for g in 0..<4 {
                    for d in 0..<256 {
                        out[t * KK + g * 256 + d] = QWQuant.f32tof16Bits(kv[(g * (P - 1) + t) * 256 + d])
                    }
                }
            }
            return out
        }
        let (cache, arena, part) = try env(C + 260)
        let h = try cache.allocate(promptTokens: C + 256)
        try cache.append(h, positions: 0..<(P - 1), layer: 3, kind: .k, src: tokenRows(seedK))
        try cache.append(h, positions: 0..<(P - 1), layer: 3, kind: .v, src: tokenRows(seedV))
        let qgD = Self.buf(try Self.load("mtpc_qdoor_ref.bin"))
        let kn = Self.buf(try Self.load("mtpc_knew_ref.bin"))
        let vn = Self.buf(try Self.load("mtpc_vnew_ref.bin"))
        let attOut = Self.zeros(6144, .f16)
        let dec = try QWAttnDecode(hal: Self.hal, geometry: .qwen38_27b)
        let views = try cache.layerViews(h, layer: 3)
        guard let (cbA, eA) = Self.hal.begin() else { XCTFail("cmdbuffer"); return }
        try dec.encode(eA, qgate: Self.tv(qgD, QG, .f16), kv: views,
                       knew: Self.tv(kn, KK, .f16), vnew: Self.tv(vn, KK, .f16),
                       partials: part, out: Self.tv(attOut, 6144, .f16),
                       C: C, splits: QWAttnDecode.splits(forKeys: C + 1))
        eA.endEncoding(); cbA.commit(); cbA.waitUntilCompleted()
        viol(Self.readF(Self.u16row(attOut, 0, 6144)), try ref16("mtpc_att_out.bin"), "attn decode")

        // ── o_proj + fused (res1, ln2) via rmsnormResidual.
        let o32 = Self.zeros(H, .f32)
        let gsO = Self.zeros(6144 / 128, .f32)
        guard let (cbB, eB) = Self.hal.begin() else { XCTFail("cmdbuffer"); return }
        try QWGemm.encode(eB, hal: Self.hal,
                          plan: QWGemm.plan(QWGemmShape(m: 1, n: H, k: 6144), dtype: .q4g, epi: .none),
                          x: Self.tv(attOut, 6144, .f16), xGroupSum: Self.tv(gsO, 48, .f32),
                          wQ4: try plane("mtpc_o", H, 6144), y: Self.tv(o32, H, .f32))
        let o16 = Self.zeros(H, .f16)
        try QWGemm.encodeCast(eB, hal: Self.hal, x32: Self.tv(o32, H, .f32), y16: Self.tv(o16, H, .f16))
        eB.endEncoding(); cbB.commit(); cbB.waitUntilCompleted()

        let res1buf = Self.buf(try Self.load("mtpc_x1.bin"))        // x1 f16 (2a rung)
        let ln2out = Self.zeros(H, .f16)
        let ln2w = Self.buf(try Self.load("mtpc_ln2.bin"))
        guard let (cbC, eC) = Self.hal.begin() else { XCTFail("cmdbuffer"); return }
        try QWNorm.encode(eC, hal: Self.hal, variant: .rmsnormResidual,
                          x: Self.tv(o16, H, .f16), gamma: Self.tv(ln2w, H, .f16),
                          out: Self.tv(ln2out, H, .f16), residual: Self.tv(res1buf, H, .f16), geometry: .qwen38_27b)
        eC.endEncoding(); cbC.commit(); cbC.waitUntilCompleted()
        viol(Self.readF(Self.u16row(res1buf, 0, H)), try ref16("mtpc_res1.bin"), "res1 (fused)")
        viol(Self.readF(Self.u16row(ln2out, 0, H)), try ref16("mtpc_ln2a.bin"), "ln2 (fused)")

        // ── MLP M=1 on the shrunk-real planes: silu_pair (gate‖up rows) →
        //    h f32 → cast → fwht32 → down GEMV → fused (res2, post_norm).
        let gs2 = Self.zeros(H / 128, .f32)
        let h32 = Self.zeros(I, .f32)
        guard let (cbD, eD) = Self.hal.begin() else { XCTFail("cmdbuffer"); return }
        try QWGemm.encode(eD, hal: Self.hal,
                          plan: QWGemm.plan(QWGemmShape(m: 1, n: 2 * I, k: H), dtype: .q4g, epi: .siluMulPair),
                          x: Self.tv(ln2out, H, .f16), xGroupSum: Self.tv(gs2, H / 128, .f32),
                          wQ4: try plane("mtpc_gu", 2 * I, H), y: Self.tv(h32, I, .f32))
        let h16 = Self.zeros(I, .f16)
        try QWGemm.encodeCast(eD, hal: Self.hal, x32: Self.tv(h32, I, .f32), y16: Self.tv(h16, I, .f16))
        try QWGemm.encodeFwht32(eD, hal: Self.hal, x: Self.tv(h16, I, .f16))
        eD.endEncoding(); cbD.commit(); cbD.waitUntilCompleted()
        viol(Self.readF(Self.u16row(h16, 0, I)), try ref16("mtpc_h.bin"), "mlp h", absTol: 2e-2)

        let gsI = Self.zeros(I / 128, .f32)
        let dn32 = Self.zeros(H, .f32)
        guard let (cbE, eE) = Self.hal.begin() else { XCTFail("cmdbuffer"); return }
        try QWGemm.encode(eE, hal: Self.hal,
                          plan: QWGemm.plan(QWGemmShape(m: 1, n: H, k: I), dtype: .q4g, epi: .none),
                          x: Self.tv(h16, I, .f16), xGroupSum: Self.tv(gsI, I / 128, .f32),
                          wQ4: try plane("mtpc_dn", H, I), y: Self.tv(dn32, H, .f32))
        let dn16 = Self.zeros(H, .f16)
        try QWGemm.encodeCast(eE, hal: Self.hal, x32: Self.tv(dn32, H, .f32), y16: Self.tv(dn16, H, .f16))
        eE.endEncoding(); cbE.commit(); cbE.waitUntilCompleted()

        let res2buf = Self.buf(try Self.load("mtpc_res1.bin"))       // res1 f16
        let zstep = Self.zeros(H, .f16)
        let pnw = Self.buf(try Self.load("mtpc_postnorm.bin"))
        guard let (cbF, eF) = Self.hal.begin() else { XCTFail("cmdbuffer"); return }
        try QWNorm.encode(eF, hal: Self.hal, variant: .rmsnormResidual,
                          x: Self.tv(dn16, H, .f16), gamma: Self.tv(pnw, H, .f16),
                          out: Self.tv(zstep, H, .f16), residual: Self.tv(res2buf, H, .f16), geometry: .qwen38_27b)
        eF.endEncoding(); cbF.commit(); cbF.waitUntilCompleted()
        viol(Self.readF(Self.u16row(res2buf, 0, H)), try ref16("mtpc_res2.bin"), "res2 (fused)")
        viol(Self.readF(Self.u16row(zstep, 0, H)), try ref16("mtpc_z_step.bin"), "post_norm")

        // ── SHARED trunk head → logits f32 → argmax door == q4 draft-0.
        let z16 = QWTensorValue(region: QWMRegion(buffer: zstep, base: 0, length: H * 2),
                                offset: 0, shape: QWTensorShape([H], .f16))
        let gsH = Self.zeros(H / 128, .f32)
        let logits = Self.zeros(4096, .f32)
        guard let (cbG, eG) = Self.hal.begin() else { XCTFail("cmdbuffer"); return }
        try QWGemm.encode(eG, hal: Self.hal,
                          plan: QWGemm.plan(QWGemmShape(m: 1, n: 4096, k: H), dtype: .q4g, epi: .none),
                          x: z16, xGroupSum: Self.tv(gsH, H / 128, .f32),
                          wQ4: try plane("mtpc_head", 4096, H), y: Self.tv(logits, 4096, .f32))
        eG.endEncoding(); cbG.commit(); cbG.waitUntilCompleted()
        viol(Self.f32s(Data(bytes: logits.contents(), count: 4096 * 4)), try ref32("mtpc_logits.bin"), "head logits")

        let head = try QWHead(hal: Self.hal, geometry: .qwen38_27b)
        let nTG = QWHead.argmaxSlots(4096)
        let kh = Self.zeros(nTG, .i32), kl = Self.zeros(nTG, .i32), tok = Self.zeros(1, .i32)
        guard let (cbH, eH) = Self.hal.begin() else { XCTFail("cmdbuffer"); return }
        try head.encodeArgmax(eH, y: Self.tv(logits, 4096, .f32),
                              keysHi: Self.tv(kh, nTG, .i32), keysLo: Self.tv(kl, nTG, .i32),
                              token: Self.tv(tok, 1, .i32), N: 4096)
eH.endEncoding(); cbH.commit(); cbH.waitUntilCompleted()
        let want = Self.i32s(try Self.load("mtpc_dft0_q4.bin"))[0]
        XCTAssertEqual(tok.contents().load(as: UInt32.self), UInt32(bitPattern: want),
                       "draft-0 argmax vs q4-plane golden")
        _ = arena   // retention
    }

    func testGPUGumbelArgmaxMatchesPhiloxReference() throws {
        let draft: [Float] = [0.25, 1.5, -0.5, -2]
        let target: [Float] = [1.25, 0.75, -0.5, -2]
        let vocabulary = UInt32(draft.count)
        let seed: UInt64 = 137
        let position: UInt64 = 3
        let temperature: Float = 0.75
        func floatBuffer(_ values: [Float]) throws -> MTLBuffer {
            let buffer = try XCTUnwrap(Self.hal.device.makeBuffer(
                length: values.count * MemoryLayout<Float>.size, options: .storageModeShared))
            values.withUnsafeBytes { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            return buffer
        }
        func winner(_ logits: [Float]) -> Int32 {
            let base = position * UInt64(vocabulary)
            return Int32(logits.indices.max { lhs, rhs in
                let lu = max(Double(QWPhilox.u(base + UInt64(lhs), seed: seed, version: .upper24)),
                             1.0 / 16_777_216.0)
                let ru = max(Double(QWPhilox.u(base + UInt64(rhs), seed: seed, version: .upper24)),
                             1.0 / 16_777_216.0)
                let ls = Double(logits[lhs]) / Double(temperature) - log(-log(lu))
                let rs = Double(logits[rhs]) / Double(temperature) - log(-log(ru))
                return ls == rs ? lhs > rhs : ls < rs
            }!)
        }
        let draftBuffer = try floatBuffer(draft), targetBuffer = try floatBuffer(target)
        let output = try XCTUnwrap(Self.hal.device.makeBuffer(length: 8, options: .storageModeShared))
        memset(output.contents(), 0, output.length)
        let pipeline = try Self.hal.pipeline("qwg_gumbel_sample_argmax")
        let commandBuffer = try XCTUnwrap(Self.hal.queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(draftBuffer, offset: 0, index: 0)
        encoder.setBuffer(targetBuffer, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var vocabularyValue = vocabulary
        var seedValue = seed, positionValue = position
        var temperatureValue = temperature
        encoder.setBytes(&vocabularyValue, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&seedValue, length: MemoryLayout<UInt64>.size, index: 4)
        encoder.setBytes(&positionValue, length: MemoryLayout<UInt64>.size, index: 5)
        encoder.setBytes(&temperatureValue, length: MemoryLayout<Float>.size, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertEqual(commandBuffer.status, .completed, "\(String(describing: commandBuffer.error))")
        let winners = output.contents().assumingMemoryBound(to: Int32.self)
        XCTAssertEqual(winners[0], winner(draft))
        XCTAssertEqual(winners[1], winner(target))
    }

    func testVariationalRelaxedGreedyAndStochasticAcceptance() throws {
        // 1. Greedy acceptance: margin = 0 matches strict greedy; margin = 1.5 accepts near-argmax drafts.
        let targetLogits: [[Float]] = [
            [1.0, 5.0, 4.2, -1.0], // argmax = 1 (5.0), token 2 = 4.2 (gap 0.8 <= 1.5)
            [0.5, 2.0, 6.0, 3.0],  // argmax = 2 (6.0), token 1 = 2.0 (gap 4.0 > 1.5)
            [9.0, 1.0, 0.0, -2.0]  // bonus row argmax = 0
        ]
        let targetTokens = [1, 2, 0]
        let drafts = [2, 1]

        let strictGreedy = try QWMTP.checkedRelaxedGreedyRound(
            drafts: drafts,
            targetTokens: targetTokens,
            targetLogits: targetLogits,
            vocabulary: 4,
            margin: 0.0
        )
        XCTAssertEqual(strictGreedy.accepted, 0)
        XCTAssertEqual(strictGreedy.finalToken, 1)

        let relaxedGreedy = try QWMTP.checkedRelaxedGreedyRound(
            drafts: drafts,
            targetTokens: targetTokens,
            targetLogits: targetLogits,
            vocabulary: 4,
            margin: 1.5
        )
        XCTAssertEqual(relaxedGreedy.accepted, 1)
        XCTAssertEqual(relaxedGreedy.finalToken, 2) // argmax of row 1 where draft[1]=1 was rejected

        // 1b. Probability floor: row 0's draft has p_target ~= 0.306 — kept at 0.2, refused at 0.35.
        let flooredLow = try QWMTP.checkedRelaxedGreedyRound(
            drafts: drafts, targetTokens: targetTokens, targetLogits: targetLogits,
            vocabulary: 4, margin: 1.5, minProbability: 0.2)
        XCTAssertEqual(flooredLow.accepted, 1)
        let flooredHigh = try QWMTP.checkedRelaxedGreedyRound(
            drafts: drafts, targetTokens: targetTokens, targetLogits: targetLogits,
            vocabulary: 4, margin: 1.5, minProbability: 0.35)
        XCTAssertEqual(flooredHigh.accepted, 0)
        XCTAssertEqual(flooredHigh.finalToken, 1)
        XCTAssertTrue(QWMTP.clearsProbabilityFloor(targetLogits[0], token: 2, minProbability: 0))
        XCTAssertFalse(QWMTP.clearsProbabilityFloor(targetLogits[0], token: 2, minProbability: 1))
        XCTAssertThrowsError(try QWMTP.checkedRelaxedGreedyRound(
            drafts: drafts, targetTokens: targetTokens, targetLogits: targetLogits,
            vocabulary: 4, margin: 1.5, minProbability: 1.5))

        // 2. Relaxed Gumbel coupling: margin = 0 matches strict Gumbel coupling; margin > 0 accepts within margin.
        let draftDists = [
            try QWSamplingDistribution.sparse(vocabulary: 4, ids: [0, 1, 2, 3], weights: [0.05, 0.15, 0.75, 0.05]),
            try QWSamplingDistribution.sparse(vocabulary: 4, ids: [0, 1, 2, 3], weights: [0.10, 0.70, 0.10, 0.10])
        ]
        let targetDists = [
            try QWSamplingDistribution.sparse(vocabulary: 4, ids: [0, 1, 2, 3], weights: [0.05, 0.50, 0.40, 0.05]),
            try QWSamplingDistribution.sparse(vocabulary: 4, ids: [0, 1, 2, 3], weights: [0.05, 0.45, 0.45, 0.05]),
            try QWSamplingDistribution.sparse(vocabulary: 4, ids: [0, 1, 2, 3], weights: [0.80, 0.10, 0.05, 0.05])
        ]
        let gumbelStrict = try QWMTP.checkedRelaxedGumbelCouplingRound(
            draftDistributions: draftDists,
            targetDistributions: targetDists,
            seed: 42,
            offset: 10,
            margin: 0.0
        )
        let gumbelWide = try QWMTP.checkedRelaxedGumbelCouplingRound(
            draftDistributions: draftDists,
            targetDistributions: targetDists,
            seed: 42,
            offset: 10,
            margin: 10.0
        )
        XCTAssertGreaterThanOrEqual(gumbelWide.accepted, gumbelStrict.accepted)
        XCTAssertEqual(gumbelWide.accepted, 2)

        // 3. Relaxed Leviathan acceptance: margin = 0 matches strict acceptRound; margin > 0 increases acceptance.
        let uniforms: [Double] = [0.99, 0.99, 0.10, 0.10] // high uniforms reject strict Leviathan when tp < dp
        let strictLev = try QWMTP.checkedRelaxedAcceptRound(
            draftDistributions: draftDists,
            targetDistributions: targetDists,
            drafts: [2, 1],
            uniforms: uniforms,
            margin: 0.0
        )
        let relaxedLev = try QWMTP.checkedRelaxedAcceptRound(
            draftDistributions: draftDists,
            targetDistributions: targetDists,
            drafts: [2, 1],
            uniforms: uniforms,
            margin: 1.0
        )
        XCTAssertEqual(strictLev.accepted, 0)
        XCTAssertEqual(relaxedLev.accepted, 2)
        XCTAssertEqual(relaxedLev.finalToken, 0)
    }

    func testGPUVariationalArgmaxFinKernel() throws {
        // M = 4 rows, vocab = 8, slots = 1: rows 0..1 accept drafts within margin 1.5,
        // row 2 rejects (gap 5.0 > 1.5 -> argmax 5), and row 3 is the bonus row (argmax 7).
        let rows = 4
        let vocab = 8
        let slots = 1
        var logits = [Float](repeating: -10.0, count: rows * vocab)
        logits[0 * vocab + 1] = 10.0
        logits[0 * vocab + 2] = 9.2
        logits[1 * vocab + 3] = 10.0
        logits[1 * vocab + 4] = 9.5
        logits[2 * vocab + 5] = 10.0
        logits[2 * vocab + 6] = 5.0
        logits[3 * vocab + 7] = 8.0

        func sortable(_ v: Float) -> UInt32 {
            let raw = v.bitPattern
            let bits = (raw & 0x7fffffff) == 0 ? 0 : raw
            return (bits & 0x80000000) != 0 ? ~bits : (bits ^ 0x80000000)
        }
        let keysHi: [UInt32] = [sortable(10.0), sortable(10.0), sortable(10.0), sortable(8.0)]
        let keysLo: [UInt32] = [~UInt32(1), ~UInt32(3), ~UInt32(5), ~UInt32(7)]
        // draftTok layout matches Flash-Next tokenIn: [committed, draft0, draft1, draft2]
        let draftTok: [UInt32] = [0, 2, 4, 6]

        let logitsBuf = Self.buf(logits.withUnsafeBytes { Data($0) })
        let hiBuf = Self.buf(keysHi.withUnsafeBytes { Data($0) })
        let loBuf = Self.buf(keysLo.withUnsafeBytes { Data($0) })
        let draftsBuf = Self.buf(draftTok.withUnsafeBytes { Data($0) })
        let outTokBuf = Self.zeros(rows, .i32)

        guard let (cb, enc) = Self.hal.begin() else { XCTFail("cmdbuffer"); return }
        try QWGemm.encodeArgmaxFin(
            enc,
            hal: Self.hal,
            keysHi: Self.tv(hiBuf, rows * slots, .i32),
            keysLo: Self.tv(loBuf, rows * slots, .i32),
            token: Self.tv(outTokBuf, rows, .i32),
            slots: slots,
            rows: rows,
            variational: QWVariationalAcceptance(
                logits: Self.tv(logitsBuf, rows * vocab, .f32),
                draftTokens: Self.tv(draftsBuf, rows, .i32),
                vocab: vocab,
                totalRows: rows,
                margin: 1.5
            )
        )
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let gotToks = outTokBuf.contents().bindMemory(to: UInt32.self, capacity: rows)
        XCTAssertEqual(gotToks[0], 2, "Row 0 outputs accepted draftTok[1] = 2 within margin 1.5")
        XCTAssertEqual(gotToks[1], 4, "Row 1 outputs accepted draftTok[2] = 4 within margin 1.5")
        XCTAssertEqual(gotToks[2], 5, "Row 2 outputs strict argmax 5 after rejecting draftTok[3] = 6")
        XCTAssertEqual(gotToks[3], 7, "Row 3 (bonus row) outputs strict argmax 7")
    }

    func testGPUVariationalArgmaxFinMinProbFloor() throws {
        // Same rows as the margin test; with p_min = 0.35 row 0's draft (p ≈ 0.31) falls below the
        // floor while row 1's draft (p ≈ 0.38) clears it, although both are within margin 1.5.
        let rows = 4
        let vocab = 8
        let slots = 1
        var logits = [Float](repeating: -10.0, count: rows * vocab)
        logits[0 * vocab + 1] = 10.0
        logits[0 * vocab + 2] = 9.2
        logits[1 * vocab + 3] = 10.0
        logits[1 * vocab + 4] = 9.5
        logits[2 * vocab + 5] = 10.0
        logits[2 * vocab + 6] = 5.0
        logits[3 * vocab + 7] = 8.0

        func sortable(_ v: Float) -> UInt32 {
            let raw = v.bitPattern
            let bits = (raw & 0x7fffffff) == 0 ? 0 : raw
            return (bits & 0x80000000) != 0 ? ~bits : (bits ^ 0x80000000)
        }
        let keysHi: [UInt32] = [sortable(10.0), sortable(10.0), sortable(10.0), sortable(8.0)]
        let keysLo: [UInt32] = [~UInt32(1), ~UInt32(3), ~UInt32(5), ~UInt32(7)]
        let draftTok: [UInt32] = [0, 2, 4, 6]

        let logitsBuf = Self.buf(logits.withUnsafeBytes { Data($0) })
        let hiBuf = Self.buf(keysHi.withUnsafeBytes { Data($0) })
        let loBuf = Self.buf(keysLo.withUnsafeBytes { Data($0) })
        let draftsBuf = Self.buf(draftTok.withUnsafeBytes { Data($0) })
        let outTokBuf = Self.zeros(rows, .i32)

        guard let (cb, enc) = Self.hal.begin() else { XCTFail("cmdbuffer"); return }
        try QWGemm.encodeArgmaxFin(
            enc,
            hal: Self.hal,
            keysHi: Self.tv(hiBuf, rows * slots, .i32),
            keysLo: Self.tv(loBuf, rows * slots, .i32),
            token: Self.tv(outTokBuf, rows, .i32),
            slots: slots,
            rows: rows,
            variational: QWVariationalAcceptance(
                logits: Self.tv(logitsBuf, rows * vocab, .f32),
                draftTokens: Self.tv(draftsBuf, rows, .i32),
                vocab: vocab,
                totalRows: rows,
                margin: 1.5,
                minLogProb: QWVariationalAcceptance.minLogProb(forMinProbability: 0.35)
            )
        )
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let gotToks = outTokBuf.contents().bindMemory(to: UInt32.self, capacity: rows)
        XCTAssertEqual(gotToks[0], 1, "Row 0 rejects draft 2 (p ≈ 0.31 < 0.35) and keeps argmax 1")
        XCTAssertEqual(gotToks[1], 4, "Row 1 accepts draft 4 (p ≈ 0.38 >= 0.35)")
        XCTAssertEqual(gotToks[2], 5, "Row 2 rejects draft 6 by margin")
        XCTAssertEqual(gotToks[3], 7, "Row 3 (bonus row) outputs strict argmax 7")
    }
}
