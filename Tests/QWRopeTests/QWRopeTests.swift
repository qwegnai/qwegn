import XCTest
import Metal
@testable import QWRope
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// C07 gate (specs/07-qwrope.md):
///  1. table parity vs torch fixture: default ≤ 1e-5, YaRN ≤ 1e-5 (spec law)
///  2. fused apply parity vs fixture outputs (f16 1-ulp adjacency law, C06)
final class QWRopeTests: XCTestCase {

    static let qSlot = QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim * 2      // 12288
    static let kAll = QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim          // 1024
    static let fixtureRoot: URL = {
        var d = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<8 {
            let c = d.appendingPathComponent("Scripts/spike/out")
            if FileManager.default.fileExists(atPath: c.path) { return c }
            d = d.deletingLastPathComponent()
        }
        fatalError("run Scripts/oracle/gen_rope_fixtures.py")
    }()
    static func fixture(_ n: String) -> Data { try! Data(contentsOf: fixtureRoot.appendingPathComponent(n)) }
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()


    static func f16(_ bits: UInt16) -> Float { QWQuant.f16tof32Bits(bits) }
    static func floats(_ d: Data) -> [Float] { d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }
    static func f16Vals(_ d: Data) -> [Float] {
        d.withUnsafeBytes { raw in (0..<d.count / 2).map { f16(raw.load(fromByteOffset: $0 * 2, as: UInt16.self)) } }
    }

    /// 1-ulp adjacency count (C06 law: f16 outputs compare bit-adjacent, not raw abs)
    static func parityViolations(_ got: [Float], _ ref: [Float], absTol: Float = 1e-3)
        -> (violations: Int, worstAbs: Double, worstIdx: Int, worstRatio: Double) {
        func spacing(_ v: Float) -> Double {
            let e = Int((v.bitPattern >> 23) & 0xFF) - 127 - 10
            return e >= 0 ? Double(1 << e) : 1.0 / Double(1 << -e)
        }
        var viol = 0, worst = -1, worstAbs = 0.0, worstRatio = 0.0
        for i in 0..<got.count {
            let d = abs(Double(got[i]) - Double(ref[i]))
            if d > worstAbs { worstAbs = d; worst = i }
            guard d > Double(absTol) else { continue }
            let r = d / min(spacing(got[i]), spacing(ref[i]))
            worstRatio = max(worstRatio, r)
            if r > 1.0001 { viol += 1 }
        }
        return (viol, worstAbs, worst, worstRatio)
    }

    /// max abs diff of one 64-float table row against fixture row p (split out
    /// — the inline 3-arg max/abs overload cost Swift >5 s to type-check).
    func tableRowWorst(_ row: [Float], _ c: [Float], _ s: [Float], _ p: Int) -> Double {
        var w = 0.0
        for i in 0..<32 {
            let d1 = abs(row[i] - c[p * 64 + i]); let d2 = abs(row[32 + i] - s[p * 64 + i])
            w = max(w, Double(max(d1, d2)))
        }
        return w
    }

    // MARK: table parity

    func testTableParityDefault() throws {
        // fixture pos_ids = randperm(262144)[:512] → build rows up to max pos
        let posProbe = Self.floats(Self.fixture("rope_pos_ids.bin"))
        let t = QWRopeTables(rows: Int(posProbe.max() ?? 0) + 1, yarn: nil, geometry: .qwen38_27b)
        let pos = posProbe
        let refC = Self.floats(Self.fixture("rope_plain_cos.bin"))   // [512,64]
        let refS = Self.floats(Self.fixture("rope_plain_sin.bin"))
        var worst = 0.0
        for p in 0..<512 {
            // engine table is indexed by position row; fixture rows are
            // pos_ids order → build our row for pos[p] via direct cos check.
            let row = t.cosSinRow(Int(pos[p]))
            worst = max(worst, tableRowWorst(row, refC, refS, p))
        }
        XCTAssertLessThanOrEqual(worst, 1e-5, "default table parity (spec ≤ 1e-5)")
    }

    func testTableParityYarn() throws {
        let posProbe = Self.floats(Self.fixture("rope_pos_ids.bin"))
        let t = QWRopeTables(rows: Int(posProbe.max() ?? 0) + 1, yarn: QWYaRNConfig(factor: 4), geometry: .qwen38_27b)
        let refC = Self.floats(Self.fixture("rope_yarn4_cos.bin"))
        let refS = Self.floats(Self.fixture("rope_yarn4_sin.bin"))
        let pos = posProbe
        var worst = 0.0
        for p in 0..<512 {
            let row = t.cosSinRow(Int(pos[p]))
            worst = max(worst, tableRowWorst(row, refC, refS, p))
        }
        XCTAssertLessThanOrEqual(worst, 1e-5, "YaRN table parity")
    }

    func testYarnNeedle1M() throws {
        // extrapolation sanity (spec: needle at simulated 1M position)
        let t = QWRopeTables(rows: 1_000_001, yarn: QWYaRNConfig(factor: 4), geometry: .qwen38_27b)
        let refC = Self.floats(Self.fixture("rope_yarn4_cos_1m.bin"))   // [2,64] cat[f,f]
        let refS = Self.floats(Self.fixture("rope_yarn4_sin_1m.bin"))
        // engine row = [cos0..31 | sin0..31]; HF files each duplicate (cat[f,f])
        // → compare cos-half vs cos-file, sin-half vs sin-file (the other two
        // table tests already do this; needle must too — comparing sin-half
        // against the cos file's duplicate gave a bogus |sin−cos|≈1.6 diff).
        for (j, p) in [999_000, 1_000_000].enumerated() {
            let row = t.cosSinRow(p)
            var worst = 0.0; var widx = -1
            for i in 0..<32 {
                let dc = abs(row[i] - refC[j * 64 + i]), ds = abs(row[32 + i] - refS[j * 64 + i])
                if Double(dc) > worst { worst = Double(dc); widx = i }
                if Double(ds) > worst { worst = Double(ds); widx = 32 + i }
            }
            XCTAssertLessThanOrEqual(worst, 1e-5, "1M needle row \(p) widx \(widx)")
        }
        // table build cost is the cold budget line: ≤ 200 ms for 262 K rows
    }

    func testTableBuildBudget() throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable && ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true",
                          "Cold table build timing gate requires bare-metal hardware")
        let t0 = DispatchTime.now()
        _ = QWRopeTables(rows: 262_144, yarn: nil, geometry: .qwen38_27b)
        let ms = Double(t0.distance(to: .now()).nanosecondsOrDie()) / 1e6
        XCTAssertLessThanOrEqual(ms, 200, "cold table build budget 262K rows")
    }

    // MARK: apply parity

    struct IO {
        let S: Int
        let arena: QWArena
        let q, k, wq, wk, qo, ko, norm, table: QWTensorValue
        let nullSlab: MTLBuffer      // instance-stored: MTLBuffer is not Sendable
    }

    func makeIO(_ S: Int) throws -> IO {
        // activations: q in/out + k in/out + staged norm (full q‖gate+k size)
        // + weights wq/wk + table rows [S,64] f32 + slack (all in activations).
        let act = S * Self.qSlot * 2 + S * Self.kAll * 2          // q in, k in
                  + S * Self.qSlot * 2 + S * Self.kAll * 2        // q out, k out
                  + S * (Self.qSlot + Self.kAll) * 2              // staged norm
                  + QWModelGeometry.qwen38_27b.headDim * 2 + QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim * 2
                  + S * 64 * 4 + 32 * 256
        let arena = try QWArena(hal: Self.hal, weights: 0, activations: act, kv: 0, gdnStates: 0)
        let a = arena.activationsRegion()
        let q = try a.allocate(QWTensorShape([S * Self.qSlot], .f16))
        let k = try a.allocate(QWTensorShape([S * Self.kAll], .f16))
        let wq = try a.allocate(QWTensorShape([QWModelGeometry.qwen38_27b.headDim], .f16))
        let wk = try a.allocate(QWTensorShape([QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim], .f16))
        let qo = try a.allocate(QWTensorShape([S * Self.qSlot], .f16))
        let ko = try a.allocate(QWTensorShape([S * Self.kAll], .f16))
        let norm = try a.allocate(QWTensorShape([S * (Self.qSlot + Self.kAll)], .f16))
        // table for this call: positions 0…S-1 → rows [S,64] f32
        let table = try a.allocate(QWTensorShape([S, 64], .f32))
        arena.copyIn(q, from: Self.fixture("rope_q_in_S\(S).bin"))
        arena.copyIn(k, from: Self.fixture("rope_k_in_S\(S).bin"))
        arena.copyIn(wq, from: Self.fixture("rope_qw_S\(S).bin"))
        arena.copyIn(wk, from: Self.fixture("rope_kw_S\(S).bin"))
        _ = table   // table bytes are bound by run() (fixture rows, pos order)
        let slab = Self.hal.device.makeBuffer(length: 16, options: .storageModeShared)
        return IO(S: S, arena: arena, q: q, k: k, wq: wq, wk: wk, qo: qo, ko: ko, norm: norm, table: table, nullSlab: slab!)
    }

    func run(_ io: IO, writeNorm: Bool) throws {
        let tables = QWRopeTables(rows: io.S, yarn: nil, geometry: .qwen38_27b)   // positions 0…S-1
        // rebuild table rows with the FIXTURE's exact bytes so the parity
        // question is ONLY the kernel, never the table math (table parity has
        // its own tests above).
        let c = Self.floats(Self.fixture("rope_table_cos_S\(io.S).bin"))
        let s = Self.floats(Self.fixture("rope_table_sin_S\(io.S).bin"))
        var rows = [Float](repeating: 0, count: io.S * 64)
        for p in 0..<io.S {
            for i in 0..<32 {
                rows[p * 64 + i] = c[p * 64 + i]
                rows[p * 64 + 32 + i] = s[p * 64 + i]
            }
        }
        rows.withUnsafeBytes { io.arena.copyIn(io.table, from: Data($0)) }
        _ = tables
        let rope = QWRopeApply(tableTensor: io.table, hal: Self.hal,
                               nullSlab: io.nullSlab, geometry: .qwen38_27b)
        guard let (cb, enc) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
        do {
            try rope.encodeQK(enc, hal: Self.hal,
                              q: io.q, k: io.k, wq: io.wq, wk: io.wk,
                              qOut: io.qo, kOut: io.ko,
                              normOut: writeNorm ? io.norm : nil,
                              positions: Array(0..<io.S))
        } catch { enc.endEncoding(); throw error }   // end-on-throw; NEVER wait uncommitted
        Self.hal.end(cb, enc)                        // commit FIRST…
        cb.waitUntilCompleted()                      // …then wait
    }

    func testApplyParityDecode() throws {
        for S in [1, 8, 64] {
            let io = try makeIO(S)
            // POISON the outputs: 0x7C00 halves = ~65504·pattern. If the
            // kernel never writes a slot, parity "worst" will show poison.
            poison(io, io.qo, S * Self.qSlot); poison(io, io.ko, S * Self.kAll)
            poison(io, io.norm, S * (Self.qSlot + Self.kAll))
            try run(io, writeNorm: true)
            let qGot = readF16(io, io.qo, S * Self.qSlot)
            let kGot = readF16(io, io.ko, S * Self.kAll)
            let qRef = Self.f16Vals(Self.fixture("rope_q_out_S\(S).bin"))
            let kRef = Self.f16Vals(Self.fixture("rope_k_out_S\(S).bin"))
            let vq = Self.parityViolations(qGot, qRef)
            if vq.violations > 0, S == 1 {
                let p = io.arena.cpuPointer(of: io.qo).assumingMemoryBound(to: UInt16.self)
                let hex = (0..<16).map { String(format: "%04X", p[$0]) }.joined(separator: " ")
                let xin = io.arena.cpuPointer(of: io.q).assumingMemoryBound(to: UInt16.self)
                let hexIn = (0..<16).map { String(format: "%04X", xin[$0]) }.joined(separator: " ")
                let np_ = io.arena.cpuPointer(of: io.norm).assumingMemoryBound(to: UInt16.self)
                let hexN = (0..<16).map { String(format: "%04X", np_[$0]) }.joined(separator: " ")
                print("S1 qo hex[0:16]   \(hex)")
                print("S1 q  hex[0:16]   \(hexIn)")
                print("S1 n  hex[0:16]   \(hexN)")
                print("S1 poison65504 in qo: \(qGot.filter { $0 == 65504 }.count), zeros: \(qGot.filter { $0 == 0 }.count) of \(qGot.count)")
            }
            XCTAssertEqual(vq.violations, 0, "S=\(S) q parity: worst \(vq.worstAbs) @\(vq.worstIdx) ratio \(vq.worstRatio)")
            let vko = Self.parityViolations(kGot, kRef)
            XCTAssertEqual(vko.violations, 0, "S=\(S) k parity: worst \(vko.worstAbs) @\(vko.worstIdx) ratio \(vko.worstRatio)")
            // staged norm (the C08 head-norm hook): q rows land at Q slots of
            // each interleaved pair (slot head = 2·row); k rows at their base.
            let nGot = readNorm(io, S)
            let nRef = Self.f16Vals(Self.fixture("rope_q_norm_S\(S).bin"))   // f16 engine law
            var qSlots: [Float] = []
            for p in 0..<S { for row in 0..<24 {   // compact [p·768 + r·32] uint4s
                let b = p * 6144 + row * 256
                qSlots += nGot[b ..< b + 256] } }
            let vn = Self.parityViolations(qSlots, nRef)
            if vn.violations > 0, S <= 8 {
                var rope = 0, pass_ = 0, heads = Set<Int>()
                for i in 0..<qSlots.count where abs(Double(qSlots[i]) - Double(nRef[i])) > 1e-3 {
                    if i % 256 < 64 { rope += 1 } else { pass_ += 1 }
                    heads.insert(i / 256)
                }
                print("norm viol S=\(S): rope-dims \(rope) passthrough \(pass_) heads \(heads.sorted().prefix(8))")
                let i = (0..<qSlots.count).first { abs(Double(qSlots[$0]) - Double(nRef[$0])) > 0.003 } ?? -1
                if i >= 0 { print("worst@\(i): got \(String(format: "%.8f", qSlots[i])) ref \(String(format: "%.8f", nRef[i]))") }
            }
            XCTAssertEqual(vn.violations, 0, "S=\(S) staged norm: worst \(vn.worstAbs) @\(vn.worstIdx) ratio \(vn.worstRatio)")
            let knRef = Self.f16Vals(Self.fixture("rope_k_norm_S\(S).bin"))  // f16 engine law
            var kSlots: [Float] = []
            for p in 0..<S { for row in 0..<4 {   // kBase = S·6144, [p·1024 + r·256]
                let base = S * 6144 + p * 1024 + row * 256
                kSlots += nGot[base ..< base + 256] } }
            let vk = Self.parityViolations(kSlots, knRef)
            XCTAssertEqual(vk.violations, 0, "S=\(S) k staged norm: worst \(vk.worstAbs) @\(vk.worstIdx)")
        }
    }

    func testExtractRoPEParity() throws {
        // EXP-8 landing #2: test that fused encodeExtractQK matches encodeQK bit-for-bit
        // and correctly extracts V.
        let S = 8
        let io = try makeIO(S)
        try run(io, writeNorm: false)
        let qRef = readF16(io, io.qo, S * Self.qSlot)
        let kRef = readF16(io, io.ko, S * Self.kAll)

        // Construct packed attnY [S, 14336]
        let totalN = 14336
        let qSlot = Self.qSlot   // 12288
        let kAll = Self.kAll     // 1024
        let vAll = Self.kAll     // 1024
        let act = S * (totalN + qSlot + kAll + vAll) * 2 + 65536
        let testArena = try QWArena(hal: Self.hal, weights: 0, activations: act, kv: 0, gdnStates: 0)
        let a = testArena.activationsRegion()
        let attnY = try a.allocate(QWTensorShape([S * totalN], .f16))
        let qFused = try a.allocate(QWTensorShape([S * qSlot], .f16))
        let kFused = try a.allocate(QWTensorShape([S * kAll], .f16))
        let vFused = try a.allocate(QWTensorShape([S * vAll], .f16))

        let qInRaw = io.arena.cpuPointer(of: io.q).assumingMemoryBound(to: UInt16.self)
        let kInRaw = io.arena.cpuPointer(of: io.k).assumingMemoryBound(to: UInt16.self)
        let yRaw = testArena.cpuPointer(of: attnY).assumingMemoryBound(to: UInt16.self)

        var expectedV: [Float] = []
        for s in 0..<S {
            // copy Q into attnY
            for i in 0..<qSlot {
                yRaw[s * totalN + i] = qInRaw[s * qSlot + i]
            }
            // copy K into attnY
            for i in 0..<kAll {
                yRaw[s * totalN + qSlot + i] = kInRaw[s * kAll + i]
            }
            // synthetic V into attnY
            for i in 0..<vAll {
                let val = Float((s * vAll + i) % 1000) * 0.1
                let bits = Self.f16bits(val)
                yRaw[s * totalN + qSlot + kAll + i] = bits
                expectedV.append(Self.f16(bits))
            }
        }

        // Run fused extractQK
        let apply = QWRopeApply(tableTensor: io.table, hal: Self.hal, nullSlab: io.nullSlab, geometry: .qwen38_27b)
        guard let (cb, enc) = Self.hal.begin() else { XCTFail("hal.begin"); return }
        try apply.encodeExtractQK(enc, hal: Self.hal,
                                  attnY: attnY,
                                  qOut: qFused, kOut: kFused, vOut: vFused,
                                  wq: io.wq, wk: io.wk,
                                  positionStart: 0, count: S)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()

        let qGot = readF16(testArena, qFused, S * qSlot)
        let kGot = readF16(testArena, kFused, S * kAll)
        let vGot = readF16(testArena, vFused, S * vAll)

        let vq = Self.parityViolations(qGot, qRef)
        XCTAssertEqual(vq.violations, 0, "fused Q parity vs encodeQK: worst \(vq.worstAbs) @\(vq.worstIdx)")
        let vk = Self.parityViolations(kGot, kRef)
        XCTAssertEqual(vk.violations, 0, "fused K parity vs encodeQK: worst \(vk.worstAbs) @\(vk.worstIdx)")
        let vv = Self.parityViolations(vGot, expectedV)
        XCTAssertEqual(vv.violations, 0, "fused V extraction parity: worst \(vv.worstAbs) @\(vv.worstIdx)")
    }

    func testMropeAxisSelection() throws {
        // HF recomposition_frequencies: pair i takes axis i%3 of pos (t,h,w).
        // Text rows (all equal) must equal the scalar row exactly; distinct
        // (t,h,w) must cycle t→h→w per pair.
        let tb = QWRopeTables(rows: 10, yarn: nil, geometry: .qwen38_27b)   // cosSinRow(7/9/5) must exist — rows:4 trapped the precondition (SIGILL)
        let row = tb.cosSinRowMrope((t: 7, h: 7, w: 7))
        let scalar = tb.cosSinRow(7)
        var worst = 0.0
        for i in 0..<64 { worst = max(worst, Double(abs(row[i] - scalar[i]))) }
        XCTAssertEqual(worst, 0, "text-only mrope must equal scalar row")
        // distinct axes: pair 0 uses t=9, pair 1 uses h=3, pair 2 uses w=5
        let d = tb.cosSinRowMrope((t: 9, h: 3, w: 5))
        let i0 = tb.cosSinRow(9), i1 = tb.cosSinRow(3), i2 = tb.cosSinRow(5)
        XCTAssertEqual(d[0], i0[0]); XCTAssertEqual(d[1], i1[1]); XCTAssertEqual(d[2], i2[2])
        XCTAssertEqual(d[32], i0[32]); XCTAssertEqual(d[33], i1[33]); XCTAssertEqual(d[34], i2[34])
    }

    func poison(_ io: IO, _ t: QWTensorValue, _ n: Int) {
        let p = io.arena.cpuPointer(of: t).assumingMemoryBound(to: UInt16.self)
        for i in 0..<n { p[i] = 0x7C00 }      // f16 65504
    }
    static func f16bits(_ f: Float) -> UInt16 { QWQuant.f32tof16Bits(f) }

    func dump(_ label: String, _ got: [Float], _ ref: [Float], _ idx: [Int]) {
        var line = "\(label): "
        for i in idx { line += "[\(i) g=\(String(format: "%.4f", got[i])) r=\(String(format: "%.4f", ref[i]))] " }
        print(line)
    }

    func readF16(_ io: IO, _ t: QWTensorValue, _ n: Int) -> [Float] {
        readF16(io.arena, t, n)
    }
    func readF16(_ arena: QWArena, _ t: QWTensorValue, _ n: Int) -> [Float] {
        let p = arena.cpuPointer(of: t).assumingMemoryBound(to: UInt16.self)
        return (0..<n).map { Self.f16(p[$0]) }
    }
    func readNorm(_ io: IO, _ S: Int) -> [Float] { readF16(io, io.norm, S * (Self.qSlot + Self.kAll)) }
}

extension DispatchTimeInterval {
    func nanosecondsOrDie() -> Int { if case .nanoseconds(let n) = self { return n }
        if case .microseconds(let n) = self { return n * 1_000 }
        if case .milliseconds(let n) = self { return n * 1_000_000 }
        if case .seconds(let n) = self { return n * 1_000_000_000 }
        fatalError("units") }
}
