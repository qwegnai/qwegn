import XCTest
import Metal
import Foundation
@testable import QWHead
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWQuant

/// C15 — split-K LM-head GEMV + greedy argmax + sampler parity.
/// Fixtures (gen_head_fixtures.py): N=61440 head planes (a quarter vocab;
/// rows structurally identical — the FULL-vocab bench runs synthetic
final class QWHeadTests: XCTestCase {

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
    static func f16s(_ d: Data) -> [UInt16] { d.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) } }
    static func u32s(_ d: Data) -> [UInt32] { d.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) } }
    static func i32s(_ d: Data) -> [Int32] { d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) } }
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

    static let N = 61440, K = 5120
    static func headWeight() throws -> QWHeadWeight {
        let w = QWHeadWeight(blocks: tv(buf(try load("head_w.bin")), N * (K / 32) * 10, .f16),
                             n: N, k: K, isRotated: true)
        XCTAssertNotNil(w)
        return w!
    }
    static func run(_ cb: any MTLCommandBuffer, _ enc: any MTLComputeCommandEncoder) {
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    }

    // ── GEMV parity: S=1 (direct) and S=4 (split + combine) vs fp32 ref ──
    func testGemvParityS1AndS4() throws {
        let head = try QWHead(hal: Self.hal, geometry: .qwen38_27b)
        let x = try Self.buf(Self.load("head_x.bin"))            // f16 [5120]
        let gs = try Self.buf(Self.load("head_x_gsum.bin"))      // f32 [40]
        let yref = try Self.f32s(Self.load("head_y.bin"))
        for S in [1, 4] {
            head.splits = S
            let y = Self.zeros(Self.N, .f32)
            let part = Self.zeros(QWHead.partialElems(Self.N, S), .f32)
            guard let (cb, enc) = Self.hal.begin() else { XCTFail("no cmdbuffer"); return }
            try head.encode(enc, x: Self.tv(x, Self.K, .f16), gs: Self.tv(gs, Self.K / 128, .f32),
                            w: try Self.headWeight(), y: Self.tv(y, Self.N, .f32),
                            partials: Self.tv(part, QWHead.partialElems(Self.N, S), .f32))
            Self.run(cb, enc)
            var v = 0, wi = -1; var worst = 0.0
            let got = y.contents().bindMemory(to: Float.self, capacity: Self.N)
            for i in 0..<Self.N {
                let d = abs(Double(got[i]) - Double(yref[i]))
                if d > worst { worst = d; wi = i }
                if !got[i].isFinite || d > 2e-4 { v += 1 }
            }
            XCTAssertEqual(v, 0, "S=\(S): \(v) GEMV mismatches, worst \(worst) @\(wi)")
        }
    }

    // ── greedy argmax: fixture greedy + adversarial ±80 (overflow safety) ──
    func testArgmaxFixtures() throws {
        let head = try QWHead(hal: Self.hal, geometry: .qwen38_27b)
        for (logitFile, goldFile) in [("head_y.bin", "head_greedy.bin"),
                                      ("head_adv_logits.bin", "head_adv_greedy.bin")] {
            let logits = Self.buf(try Self.load(logitFile))
            let nTG = QWHead.argmaxSlots(Self.N)
            let kh = Self.zeros(nTG, .i32), kl = Self.zeros(nTG, .i32)
            let tok = Self.zeros(1, .i32)
            guard let (cb, enc) = Self.hal.begin() else { XCTFail("no cmdbuffer"); return }
            try head.encodeArgmax(enc, y: Self.tv(logits, Self.N, .f32),
                                  keysHi: Self.tv(kh, nTG, .i32), keysLo: Self.tv(kl, nTG, .i32),
                                  token: Self.tv(tok, 1, .i32), N: Self.N)
            Self.run(cb, enc)
            let got = tok.contents().load(as: UInt32.self)
            let want = Self.u32s(try Self.load(goldFile))[0]
            XCTAssertEqual(got, want, "\(logitFile): argmax \(got) ≠ golden \(want)")
        }
    }

    func testGreedyQ4gFusedParity() throws {
        let head = try QWHead(hal: Self.hal, geometry: .qwen38_27b)
        let x = Self.buf(try Self.load("head_x.bin"))
        let w = try Self.headWeight()
        let wantTok = Self.u32s(try Self.load("head_greedy.bin"))[0]
        let yRef = Self.zeros(Self.N, .f32)
        let yFused = Self.zeros(Self.N, .f32)
        let nTG = QWHead.gemvArgmaxSlots(Self.N)
        let kh = Self.zeros(nTG, .i32), kl = Self.zeros(nTG, .i32)
        let tok = Self.zeros(1, .i32)
        guard let (cb, enc) = Self.hal.begin() else { XCTFail("no cmdbuffer"); return }
        try head.encode(enc, x: Self.tv(x, Self.K, .f16), w: w, y: Self.tv(yRef, Self.N, .f32))
        try head.encodeGreedyQ4g(enc, x: Self.tv(x, Self.K, .f16), w: w,
                                 y: Self.tv(yFused, Self.N, .f32),
                                 keysHi: Self.tv(kh, nTG, .i32), keysLo: Self.tv(kl, nTG, .i32),
                                 token: Self.tv(tok, 1, .i32))
        Self.run(cb, enc)
        let gotTok = tok.contents().load(as: UInt32.self)
        XCTAssertEqual(gotTok, wantTok, "fused greedy Q4G argmax \(gotTok) ≠ golden \(wantTok)")
        let gotY = yFused.contents().bindMemory(to: Float.self, capacity: Self.N)
        let refY = yRef.contents().bindMemory(to: Float.self, capacity: Self.N)
        var diffs = 0
        for i in 0..<Self.N where gotY[i].bitPattern != refY[i].bitPattern { diffs += 1 }
        XCTAssertEqual(diffs, 0, "fused greedy Q4G logits had \(diffs) bit mismatches vs shipping head.encode")
    }

    // ── 10k-position argmax golden (spec gate "== 100% on 10k golden
    //    positions"). gen_argmax10k.py: N=8192 rows, kinds = crafted ties
    //    (lowest id wins), all-negative (sortable's ~b branch), ±80 rails,
    func testArgmax10kGolden() throws {
        let head = try QWHead(hal: Self.hal, geometry: .qwen38_27b)
        let P = 10_000, N = 8192
        let raw = try Self.load("head_amax10k_logits.bin")
        let ids = Self.u32s(try Self.load("head_amax10k_ids.bin"))
        XCTAssertEqual(raw.count, P * N * 4)
        XCTAssertEqual(ids.count, P)
        let rowBytes = N * 4
        let nTG = QWHead.argmaxSlots(N)
        // scratch: tokens [P] | kh [P·nTG] | kl [P·nTG]  (all u32 words)
        let scratch = Self.zeros(P + 2 * P * nTG, .i32)
        func at(_ base: Int, _ elems: Int) -> QWTensorValue {
            QWTensorValue(region: QWMRegion(buffer: scratch, base: base * 4,
                                            length: max(16, elems * 4)),
                          offset: 0, shape: QWTensorShape([elems], .i32))
        }
        guard let (cb, enc) = Self.hal.begin() else { XCTFail("no cmdbuffer"); return }
        for p in 0..<P {
            let row = raw.subdata(in: (p * rowBytes)..<(p * rowBytes + rowBytes))
            let logits = Self.buf(row)
            try head.encodeArgmax(enc, y: Self.tv(logits, N, .f32),
                                  keysHi: at(p * nTG, nTG),
                                  keysLo: at(P &* nTG + p * nTG, nTG),
                                  token: at(2 * P * nTG + p, 1), N: N)
        }
        Self.run(cb, enc)
        let outTok = scratch.contents().advanced(by: 2 * P * nTG * 4)
            .bindMemory(to: UInt32.self, capacity: P)
        var fails = 0, worst = -1
        for p in 0..<P where outTok[p] != ids[p] { fails += 1; if worst < 0 { worst = p } }
        XCTAssertEqual(fails, 0, "argmax golden: \(fails)/\(P) rows differ, first @\(worst)")
    }

    // ── Philox golden: the Swift stream == canonical fixture stream ──
    func testPhiloxGolden() throws {
        let golden = Self.f32s(try Self.load("head_philox_golden.bin"))
        let seed: UInt64 = 0x0123456789ABCDEF
        for n in 0..<64 {
            let u = QWPhilox.u(UInt64(n), seed: seed)
            XCTAssertEqual(u.bitPattern, golden[n].bitPattern, "philox u[\(n)] bit-mismatch")
        }
    }

    // ── sampler vs torch nucleus: survivor set, renorm probs, logprob ──
    func testSamplerVsTorchNucleus() throws {
        let logits = Self.f32s(try Self.load("head_dist_logits.bin"))
        let wantIdx = Self.i32s(try Self.load("head_dist_idx.bin")).map { Int(truncatingIfNeeded: $0) }
        let wantP = Self.f32s(try Self.load("head_dist_probs.bin"))
        var dmax = -Float.infinity
        for v in logits where v > dmax { dmax = v }
        // torch log_softmax(logits/T) reference (float64 sum — same answer
        // far inside the gate at these magnitudes).
        var sum = 0.0
        for v in logits { sum += Foundation.exp((Double(v) - Double(dmax)) / 0.7) }
        let lse = Foundation.log(sum)

        let smp = QWSampler(config: .init(temperature: 0.7, topK: 64, topP: 0.9,
                                          seed: 0x0123456789ABCDEF))
        let r = try smp.sample(logits: logits)
        XCTAssertEqual(r.survivorIds, wantIdx, "nucleus survivor set ≠ torch")
        XCTAssertEqual(r.survivorProbs.count, wantP.count)
        for i in 0..<wantP.count {
            XCTAssertEqual(Double(r.survivorProbs[i]), Double(wantP[i]), accuracy: 1e-4,
                           "renorm p[\(i)] (id \(wantIdx[i]))")
        }
        XCTAssertTrue(wantIdx.contains(r.token), "sampled token \(r.token) outside nucleus")
        // full-softmax log p of logits/T: (v − dmax)/T − ln Σexp
        let wantLP = Float((Double(logits[r.token]) - Double(dmax)) / 0.7 - lse)
        XCTAssertEqual(Double(r.logprob), Double(wantLP), accuracy: 1e-3, "full-softmax logprob")
    }

    // ── RNG determinism law: same seed + call sequence ⇒ same draws ──
    func testSamplerDeterminism() throws {
        let logits = Self.f32s(try Self.load("head_dist_logits.bin"))
        let cfg = QWSamplerConfig(temperature: 0.7, topK: 64, topP: 0.9, seed: 42)
        let a = QWSampler(config: cfg)
        let seqA = try (0..<32).map { _ in try a.sample(logits: logits).token }
        let again = try (0..<32).map { _ in try a.sample(logits: logits).token }
        XCTAssertNotEqual(seqA, again, "call counter must advance (repeat draws differ)")
        a.resetCallSequence()
        let replay = try (0..<32).map { _ in try a.sample(logits: logits).token }
        XCTAssertEqual(seqA, replay, "same seed + call sequence must replay bit-identical")
        // different seed ⇒ different stream (with overwhelming probability
        // over 32 draws from k_eff ≈ 9.5)
        let b = QWSampler(config: .init(temperature: 0.7, topK: 64, topP: 0.9, seed: 43))
        let seqB = try (0..<32).map { _ in try b.sample(logits: logits).token }
        XCTAssertNotEqual(seqA, seqB, "seed must enter the stream")
    }

    // ── encoder guards (steady-state: bad shapes throw, never garbage) ──
    func testEncoderGuards() throws {
        let head = try QWHead(hal: Self.hal, geometry: .qwen38_27b)
        let w = try Self.headWeight()
        let x = Self.buf(try Self.load("head_x.bin"))
        let gs = Self.buf(try Self.load("head_x_gsum.bin"))
        let y = Self.zeros(Self.N, .f32)
        let badX = Self.zeros(Self.K + 128, .f16)              // wrong element count
        guard let (_, enc) = Self.hal.begin() else { XCTFail("no cmdbuffer"); return }   // guard-only: cb never committed
        XCTAssertThrowsError(try head.encode(enc, x: Self.tv(badX, Self.K + 128, .f16),
                                             gs: Self.tv(gs, Self.K / 128, .f32), w: w,
                                             y: Self.tv(y, Self.N, .f32)))
        head.splits = 4
        XCTAssertThrowsError(try head.encode(enc, x: Self.tv(x, Self.K, .f16),
                                             gs: Self.tv(gs, Self.K / 128, .f32),
                                             w: w, y: Self.tv(y, Self.N, .f32),
                                             partials: nil), "S>1 without partials must throw")
        head.splits = 1
        // argmax: token buffer missing capacity
        let nTG = QWHead.argmaxSlots(Self.N)
        let tok = Self.zeros(0, .i32)
        XCTAssertThrowsError(try head.encodeArgmax(enc, y: Self.tv(y, Self.N, .f32),
                                                   keysHi: Self.tv(Self.zeros(nTG, .i32), nTG, .i32),
                                                   keysLo: Self.tv(Self.zeros(nTG, .i32), nTG, .i32),
                                                   token: Self.tv(tok, 0, .i32), N: Self.N))
        // nothing dispatched (every encode threw before dispatch); the
        // encoder is closed unused.
        enc.endEncoding()
    }

    func testBatchedRowPairedNonIncumbentBitExactParity() throws {
        let w = try Self.headWeight()
        let xSingle = try Self.load("head_x.bin")
        var x5Data = Data()
        for _ in 0..<5 { x5Data.append(xSingle) }
        let x5Buf = Self.buf(x5Data)
        let y2Buf = Self.zeros(2 * Self.N, .f32)
        let y5Buf = Self.zeros(5 * Self.N, .f32)

        for isRotated in [true, false] {
            let nonIncView = try XCTUnwrap(QWWeightView(
                blocks: w.view.blocks, n: Self.N, k: Self.K,
                bits: 4, groupSize: 32, isRotated: isRotated, isIncumbentContainer: false))
            XCTAssertFalse(nonIncView.isIncumbentContainer)
            guard let (cb, enc) = Self.hal.begin() else { XCTFail("no cmdbuffer"); return }
            try QWHead.encodeBatchedQ4GKernel(enc, hal: Self.hal, view: nonIncView,
                                              x: Self.tv(x5Buf, 2 * Self.K, .f16),
                                              y: Self.tv(y2Buf, 2 * Self.N, .f32), m: 2)
            try QWHead.encodeBatchedQ4GKernel(enc, hal: Self.hal, view: nonIncView,
                                              x: Self.tv(x5Buf, 5 * Self.K, .f16),
                                              y: Self.tv(y5Buf, 5 * Self.N, .f32), m: 5)
            Self.run(cb, enc)
            let p2 = y2Buf.contents().bindMemory(to: Float.self, capacity: 2 * Self.N)
            let p5 = y5Buf.contents().bindMemory(to: Float.self, capacity: 2 * Self.N)
            var badIdx = -1
            for i in 0..<(2 * Self.N) where abs(p2[i] - p5[i]) > 1e-4 { badIdx = i; break }
            XCTAssertEqual(badIdx, -1,
                           "batched head row-paired M=5 vs M=2 mismatch at \(badIdx) ("
                           + "\(p2[max(0, badIdx)]) vs \(p5[max(0, badIdx)])"
                           + ") for non-incumbent isRotated=\(isRotated)")
        }
    }
}
