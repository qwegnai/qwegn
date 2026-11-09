// QWAttnPagedPrefillGeneralTests.swift — the standard path's paged
// prefill: one scalar kernel that attends a chunk of T queries at `pos`
// against all the keys in [0, pos+T), reading the K/V out of the paged KV
// pool through the page table. The oracle is an fp32 host reference under
// the 1-ulp adjacency law (C06); the output carries the 32-point rotation
// (C109), so the oracle carries it too.
import XCTest
import Metal
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc
import QWKV
import QWQuant

final class QWAttnPagedPrefillGeneralTests: XCTestCase {
    static let layers = [3]

    // ── data helpers ───────────────────────────────────────────────────────

    static func f16(_ bits: UInt16) -> Float { QWQuant.f16tof32Bits(bits) }
    static func h16(_ v: Float) -> UInt16 { Float16(v).bitPattern }

    /// A deterministic value in [-1, 1] — the oracle's data, exactly f16.
    static func det(_ seed: Int) -> Float {
        let m = (seed &* 2654435761) & 0xFFFF
        return Float(Int(m % 2001) - 1000) / 1000.0
    }

    /// 1-ulp adjacency (C06) on f16-derived values: the f16 binade spacing.
    static func parityViolations(_ got: [Float], _ ref: [Float], absTol: Float = 1e-3)
        -> (violations: Int, worstAbs: Double, worstIdx: Int) {
        func spacing(_ v: Float) -> Double {
            let e = Int((v.bitPattern >> 23) & 0xFF) - 127 - 10
            return e >= 0 ? Double(1 << e) : 1.0 / Double(1 << -e)
        }
        var viol = 0, worst = 0.0, worstIdx = -1
        for i in 0..<got.count {
            let d = abs(Double(got[i]) - Double(ref[i]))
            if d > worst { worst = d; worstIdx = i }
            guard got[i].isFinite && ref[i].isFinite else { viol += 1; continue }
            if d > max(Double(absTol), min(spacing(got[i]), spacing(ref[i]))) { viol += 1 }
        }
        return (viol, worst, worstIdx)
    }

    /// 1 f16 ulp of the reference value — the law for a value STORED as f16.
    static func f16UlpViolations(_ got: [Float], _ ref: [Float], absTol: Float = 1e-3)
        -> (violations: Int, worstAbs: Double, worstIdx: Int) {
        var viol = 0, worst = 0.0, worstIdx = -1
        for i in 0..<got.count {
            let d = abs(Double(got[i]) - Double(ref[i]))
            if d > worst { worst = d; worstIdx = i }
            guard got[i].isFinite && ref[i].isFinite else { viol += 1; continue }
            if d > max(Double(absTol), Double(abs(ref[i])) * 0.0009765625) { viol += 1 }
        }
        return (viol, worst, worstIdx)
    }

    /// A one-layer llama: 1024 hidden, 8 q heads, 2 kv heads, a 128-half
    /// head. `gated` turns on the attention-output gate, the Q‖gate packing
    /// the 27B's family uses.
    static func tinyGeometry(gated: Bool = false) throws -> QWModelGeometry {
        var config: [String: Any] = [
            "model_type": "llama",
            "hidden_size": 1024,
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 256,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "eos_token_id": 2,
            "vocab_size": 64,
        ]
        if gated { config["attn_output_gate"] = true }
        let data = try JSONSerialization.data(withJSONObject: config)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    /// The oracle's data: the pool's K/V rows and the q rows. The first
    /// element of every K row carries the key index, so the scores separate
    /// and the softmax's maximum is unique.
    struct Fixture {
        var kRows: [[UInt16]]
        var vRows: [[UInt16]]
        var qRows: [UInt16]
        var rowElems: Int
        var headDim: Int
        var qHeads: Int
        var kvHeads: Int
    }

    static func fixture(geometry: QWModelGeometry, keys: Int, queries: Int) -> Fixture {
        let rowElems = geometry.kvHeads * geometry.headDim
        var kRows: [[UInt16]] = []
        var vRows: [[UInt16]] = []
        for j in 0..<keys {
            var k = [UInt16](repeating: 0, count: rowElems)
            var v = [UInt16](repeating: 0, count: rowElems)
            for h in 0..<geometry.kvHeads {
                for d in 0..<geometry.headDim {
                    let e = h * geometry.headDim + d
                    k[e] = h16(d == 0 ? 4.0 + Float(j) : det(j &* 131 &+ e &* 7) * 0.05)
                    v[e] = h16(det(j &* 197 &+ e &* 11) * 0.5)
                }
            }
            kRows.append(k)
            vRows.append(v)
        }
        // The q rows: the q, and (when the family gates) the gate that
        // follows it in the same head's block.
        let stride = geometry.attnOutputGate ? 2 * geometry.headDim : geometry.headDim
        var qRows = [UInt16](repeating: 0, count: queries * geometry.qHeads * stride)
        for r in 0..<queries {
            for h in 0..<geometry.qHeads {
                for d in 0..<geometry.headDim {
                    let e = (r * geometry.qHeads + h) * stride + d
                    qRows[e] = h16(d == 0 ? 1.0 : det(r &* 313 &+ e &* 13) * 0.3)
                    if geometry.attnOutputGate {
                        qRows[e + geometry.headDim] =
                            h16(det(r &* 577 &+ e &* 19) * 0.4)
                    }
                }
            }
        }
        return Fixture(kRows: kRows, vRows: vRows, qRows: qRows, rowElems: rowElems,
                       headDim: geometry.headDim, qHeads: geometry.qHeads,
                       kvHeads: geometry.kvHeads)
    }

    /// One head's attention over the keys [0, count): the UN-NORMALISED
    /// softmax sum, with the running maximum and the sum.
    static func reference(q: [Float], k: [[Float]], v: [[Float]], kvh: Int,
                          headDim: Int, count: Int)
        -> (o: [Float], m: Float, l: Float, gap: Float) {
        let scale = Float(1.0 / Double(headDim).squareRoot())
        var scores = [Float](repeating: 0, count: count)
        for j in 0..<count {
            var dot: Float = 0
            for d in 0..<headDim { dot += q[d] * k[j][kvh * headDim + d] }
            scores[j] = dot * scale
        }
        let m = scores.max() ?? 0
        var second = -Float.greatestFiniteMagnitude
        for s in scores where s != m { second = max(second, s) }
        var l: Float = 0
        var o = [Float](repeating: 0, count: headDim)
        for j in 0..<count {
            let p = exp(scores[j] - m)
            l += p
            for d in 0..<headDim { o[d] += p * v[j][kvh * headDim + d] }
        }
        return (o, m, l, scores.isEmpty ? 0 : m - second)
    }

    /// The output for one head: the softmax sum divided by its sum, the gate
    /// (when the family has one), then the 32-point normalised rotation over
    /// each group of 32 halves.
    static func rotated(_ o: [Float], l: Float, headDim: Int, gate: [Float]?) -> [Float] {
        let linv = l > 0 ? 1.0 / l : 0.0
        var h = [Float](repeating: 0, count: headDim)
        for d in 0..<headDim {
            let g = gate.map { $0[d] } ?? 1.0
            h[d] = Float(Float16(o[d] * linv * g))
        }
        let invSqrt32 = Float(1.0 / 32.0.squareRoot())
        var out = [Float](repeating: 0, count: headDim)
        for g in stride(from: 0, to: headDim, by: 32) {
            for i in 0..<32 {
                var acc: Float = 0
                for j in 0..<32 {
                    acc += (i & j).nonzeroBitCount % 2 == 0 ? h[g + j] : -h[g + j]
                }
                out[g + i] = Float(Float16(acc * invSqrt32))
            }
        }
        return out
    }


    /// A pool of `pages` pages whose page table is a **permutation**: the
    /// logical page p lives in the physical page `perm[p]`, and no page maps
    /// to itself — so a kernel that ignores the table reads the WRONG page.
    static func permutedPool(hal: QWHal, geometry: QWModelGeometry,
                              rows: [[UInt16]], pages: Int, perm: [Int32])
        throws -> QWKVLayerView {
        precondition(perm.count == pages, "the permutation covers every page")
        let rowElems = geometry.kvHeads * geometry.headDim
        let pageElems = QWKVCache.pageTokens * rowElems
        let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: geometry)
        let buf = hal.device.makeBuffer(length: pages * pageBytes, options: .storageModeShared)!
        let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
        for p in 0..<pages {
            let phys = Int(perm[p])
            for s in 0..<QWKVCache.pageTokens {
                let tok = p * QWKVCache.pageTokens + s
                guard tok < rows.count else { continue }
                let at = phys * pageElems + s * rowElems
                for e in 0..<rowElems { dst[at + e] = rows[tok][e] }
            }
        }
        let tab = hal.device.makeBuffer(length: 4 * pages, options: .storageModeShared)!
        let tp = tab.contents().bindMemory(to: Int32.self, capacity: pages)
        for p in 0..<pages { tp[p] = perm[p] }
        return QWKVLayerView(pool: QWMRegion(buffer: buf, base: 0, length: buf.length),
                             table: QWTensorValue(
                                region: QWMRegion(buffer: tab, base: 0, length: tab.length),
                                offset: 0, shape: QWTensorShape([pages], .i32)),
                             pageBytes: pageBytes, mode: .f16)
    }

    /// A derangement of `pages` — no page maps to itself — so a kernel that
    /// reads the pool without the table's indirection reads the WRONG page.
    /// `shift` picks WHICH one, so the K and V tables can differ.
    static func derangement(_ pages: Int, _ shift: Int) -> [Int32] {
        precondition(pages >= 2, "a derangement needs at least two pages")
        let m = shift % pages
        let s = m == 0 ? 1 : m
        return (0..<pages).map { Int32(($0 + s) % pages) }
    }

    // ── the law ────────────────────────────────────────────────────────────

    /// The general paged prefill matches the fp32 host reference: causal over
    /// [0, pos+qi], read through the page table, at a 128-half head.
    func testTheGeneralPagedPrefillMatchesAHostReference() throws {
        // 520 keys → three pages, so the page table's indirection is real.
        try checkMatchesTheOracle(try Self.tinyGeometry(), pos: 512, T: 8)
    }

    /// The gated family: the Q‖gate packing, with the sigmoid gate applied
    /// before the rotation — the 27B's law, on a narrow head.
    func testTheGeneralPagedPrefillAppliesTheGate() throws {
        let geometry = try Self.tinyGeometry(gated: true)
        XCTAssertTrue(geometry.attnOutputGate)
        try checkMatchesTheOracle(geometry, pos: 512, T: 8)
    }

    /// What the scalar kernel costs: a 512-query chunk over a 512-token
    /// context, measured on the GPU and reported. No assertion on the time —
    /// the number is the measurement, and the machine is noisy.
    func testTheGeneralPagedPrefillThroughput() throws {
        let geometry = try Self.tinyGeometry()
        let hal = try QWHal(geometry: geometry)
        let pos = 512, T = 512
        let data = Self.fixture(geometry: geometry, keys: pos + T, queries: T)
        let views = try Self.permutedViews(hal: hal, geometry: geometry,
                                           kRows: data.kRows, vRows: data.vRows,
                                           keys: pos + T)
        let op = try QWAttnPagedPrefillGeneral(hal: hal, geometry: geometry)
        let samples = try measure(op: op, hal: hal, geometry: geometry,
                                  data: data, views: views, pos: pos, T: T)
        let median = samples[samples.count / 2]
        // The attention's work: the QK products and the PV products, over
        // every (query, key) pair the causal mask admits.
        var pairs = 0
        for qi in 0..<T { pairs += pos + qi + 1 }
        let flops = 4.0 * Double(geometry.qHeads) * Double(geometry.headDim) * Double(pairs)
        let tflops = flops / (median * 1e-3) / 1e12
        print("testTheGeneralPagedPrefillThroughput: T=\(T) pos=\(pos) head=\(geometry.headDim) "
              + "median=\(String(format: "%.3f", median)) ms "
              + "(min=\(String(format: "%.3f", samples.first!)) max=\(String(format: "%.3f", samples.last!))) "
              + "=\(String(format: "%.3f", tflops)) TFLOP/s — the scalar first cut")
    }

    /// The long-context cliff: the same 512-query chunk at a growing context.
    /// The scalar shape re-reads the K/V for every query, so the cost grows
    /// with the context — the wall of the first cut, reported (spec 08).
    func testTheGeneralPagedPrefillCostsGrowWithTheContext() throws {
        let geometry = try Self.tinyGeometry()
        let hal = try QWHal(geometry: geometry)
        let op = try QWAttnPagedPrefillGeneral(hal: hal, geometry: geometry)
        let T = 512
        var rows: [(pos: Int, ms: Double, usPerToken: Double)] = []
        for pos in [0, 512, 2048, 8192, 32768] {
            let data = Self.fixture(geometry: geometry, keys: pos + T, queries: T)
            let views = try Self.permutedViews(hal: hal, geometry: geometry,
                                               kRows: data.kRows, vRows: data.vRows,
                                               keys: pos + T)
            let samples = try measure(op: op, hal: hal, geometry: geometry,
                                      data: data, views: views, pos: pos, T: T)
            let median = samples[samples.count / 2]
            // The cost of the context's LAST token: the chunk's attention per
            // token, at the context's own depth.
            rows.append((pos, median, median * 1000.0 / Double(T)))
        }
        for r in rows {
            print("testTheGeneralPagedPrefillCostsGrowWithTheContext: pos=\(r.pos) "
                  + "median=\(String(format: "%.3f", r.ms)) ms "
                  + "=\(String(format: "%.2f", r.usPerToken)) µs/token")
        }
        // The cliff: the last token's cost at a 32k context against the same
        // chunk at an empty one. The scalar shape re-reads the pool per
        // query, so the ratio tracks the context's depth.
        let empty = rows[0].usPerToken
        let deep = rows[rows.count - 1].usPerToken
        XCTAssertGreaterThan(deep, empty * 16,
            "the cliff is real: a 32k context costs far more per token than an empty one")
    }

    // ── measurement ────────────────────────────────────────────────────────

    /// The K and V pools, each with its own derangement of the page table.
    static func permutedViews(hal: QWHal, geometry: QWModelGeometry,
                              kRows: [[UInt16]], vRows: [[UInt16]], keys: Int)
        throws -> (k: QWKVLayerView, v: QWKVLayerView) {
        let pages = (keys + QWKVCache.pageTokens - 1) / QWKVCache.pageTokens
        precondition(pages >= 2, "the fixture needs several pages")
        return (try permutedPool(hal: hal, geometry: geometry, rows: kRows,
                                 pages: pages, perm: derangement(pages, 1)),
                try permutedPool(hal: hal, geometry: geometry, rows: vRows,
                                 pages: pages, perm: derangement(pages, 2)))
    }

    /// The GPU time of the chunk, in milliseconds: the median of the samples
    /// after the warm-up runs.
    func measure(op: QWAttnPagedPrefillGeneral, hal: QWHal, geometry: QWModelGeometry,
                 data: QWAttnPagedPrefillGeneralTests.Fixture,
                 views: (k: QWKVLayerView, v: QWKVLayerView),
                 pos: Int, T: Int) throws -> [Double] {
        let outHalves = T * geometry.qHeads * geometry.headDim
        let out = hal.device.makeBuffer(length: outHalves * 2, options: .storageModeShared)!
        let qBuf = hal.device.makeBuffer(length: data.qRows.count * 2, options: .storageModeShared)!
        data.qRows.withUnsafeBytes { _ = memcpy(qBuf.contents(), $0.baseAddress!, $0.count) }
        let qgate = QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length),
                                   offset: 0, shape: QWTensorShape([data.qRows.count], .f16))
        let outT = QWTensorValue(region: QWMRegion(buffer: out, base: 0, length: out.length),
                                 offset: 0, shape: QWTensorShape([outHalves], .f16))
        var samples: [Double] = []
        for run in 0..<10 {
            guard let (cb, enc) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
            try op.encode(enc, qgate: qgate, kv: views, out: outT, pos: pos, T: T)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            XCTAssertEqual(cb.status, .completed, "\(String(describing: cb.error))")
            // The first runs are the warm-up; the machine's clocks settle.
            if run >= 3 { samples.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3) }
        }
        samples.sort()
        return samples
    }

    private func checkMatchesTheOracle(_ geometry: QWModelGeometry, pos: Int, T: Int) throws {
        let hal = try QWHal(geometry: geometry)
        let keys = pos + T
        let data = Self.fixture(geometry: geometry, keys: keys, queries: T)

        // Several pages, and the K and V tables carry DIFFERENT permutations —
        // so a kernel that ignores the table, or that swaps K and V, reads
        // the wrong rows and the oracle disagrees.
        let views = try Self.permutedViews(hal: hal, geometry: geometry,
                                           kRows: data.kRows, vRows: data.vRows,
                                           keys: pos + T)

        let op = try QWAttnPagedPrefillGeneral(hal: hal, geometry: geometry)
        let outHalves = T * geometry.qHeads * geometry.headDim
        let out = hal.device.makeBuffer(length: outHalves * 2, options: .storageModeShared)!
        let qBuf = hal.device.makeBuffer(length: data.qRows.count * 2, options: .storageModeShared)!
        data.qRows.withUnsafeBytes { _ = memcpy(qBuf.contents(), $0.baseAddress!, $0.count) }
        guard let (cb, enc) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
        try op.encode(enc,
                      qgate: QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length),
                                           offset: 0,
                                           shape: QWTensorShape([data.qRows.count], .f16)),
                      kv: views,
                      out: QWTensorValue(region: QWMRegion(buffer: out, base: 0, length: out.length),
                                         offset: 0, shape: QWTensorShape([outHalves], .f16)),
                      pos: pos, T: T)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed, "\(String(describing: cb.error))")

        let kAll = data.kRows.map { row in row.map { Self.f16($0) } }
        let vAll = data.vRows.map { row in row.map { Self.f16($0) } }
        let outPtr = out.contents().assumingMemoryBound(to: UInt16.self)
        let gqa = geometry.qHeads / geometry.kvHeads
        let stride = geometry.attnOutputGate ? 2 * geometry.headDim : geometry.headDim
        for qi in 0..<T {
            for h in 0..<geometry.qHeads {
                let kvh = h / gqa
                let q = (0..<geometry.headDim).map {
                    Self.f16(data.qRows[(qi * geometry.qHeads + h) * stride + $0])
                }
                // The gate, when the family has one: the sigmoid of the
                // second half of the head's block.
                let gate: [Float]? = geometry.attnOutputGate
                    ? (0..<geometry.headDim).map {
                        let g = Self.f16(data.qRows[(qi * geometry.qHeads + h) * stride
                                                   + geometry.headDim + $0])
                        return 1.0 / (1.0 + exp(-g))
                    }
                    : nil
                // The query at `pos+qi` sees the keys [0, pos+qi].
                let ref = Self.reference(q: q, k: kAll, v: vAll, kvh: kvh,
                                         headDim: geometry.headDim, count: pos + qi + 1)
                XCTAssertGreaterThan(ref.gap, 1e-3,
                    "query \(qi) head \(h): the oracle's top two scores are within rounding")
                let got = (0..<geometry.headDim).map {
                    Self.f16(outPtr[(qi * geometry.qHeads + h) * geometry.headDim + $0])
                }
                let want = Self.rotated(ref.o, l: ref.l, headDim: geometry.headDim, gate: gate)
                let v = Self.f16UlpViolations(got, want)
                XCTAssertEqual(v.violations, 0,
                               "query \(qi) head \(h): output (worst \(v.worstAbs) @\(v.worstIdx))")
            }
        }
    }
}
