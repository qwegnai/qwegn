// QWAttnDecodeGeneralTests.swift — the standard path's decode: a head
// narrower than 256 halves. The general kernel keeps the frozen decode's
// lane mapping and MASKS the lanes past the head's own width (the head
// window), so the simdgroup collectives see the head's own elements only.
// Three laws: the fp32 host reference at a 128-half head (C06's 1-ulp
// adjacency); the general decode IS the frozen decode at the 27B's shape;
// and the combine writes the head's halves ONLY.
import XCTest
import Metal
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc
import QWKV
import QWQuant

final class QWAttnDecodeGeneralTests: XCTestCase {
    static let layers = [3]                       // one attn layer
    static let maxSplits = QWAttnDecode.maxSplits

    // ── data helpers ───────────────────────────────────────────────────────

    static func f16(_ bits: UInt16) -> Float { QWQuant.f16tof32Bits(bits) }
    static func h16(_ v: Float) -> UInt16 { Float16(v).bitPattern }

    /// A deterministic value in [-1, 1] — the oracle's data, exactly f16.
    static func det(_ seed: Int) -> Float {
        let m = (seed &* 2654435761) & 0xFFFF
        return Float(Int(m % 2001) - 1000) / 1000.0
    }

    static func buf(_ hal: QWHal, _ halves: [UInt16]) -> MTLBuffer {
        let b = hal.device.makeBuffer(length: max(16, halves.count * 2), options: .storageModeShared)!
        halves.withUnsafeBytes { _ = memcpy(b.contents(), $0.baseAddress!, $0.count) }
        return b
    }

    static func tv(_ b: MTLBuffer, _ n: Int) -> QWTensorValue {
        QWTensorValue(region: QWMRegion(buffer: b, base: 0, length: b.length),
                      offset: 0, shape: QWTensorShape([n], .f16))
    }

    /// 1-ulp adjacency (C06 law) on f16-derived values: the f16 binade
    /// spacing (value·2⁻¹⁰), the house law for a value whose inputs are f16.
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

    /// A one-layer llama: 1024 hidden, 8 q heads, 2 kv heads, a 128-half head
    /// (GQA 4 — the frozen decode's 12-key staging assumes GQA 6).
    static func tinyGeometry() throws -> QWModelGeometry {
        let config: [String: Any] = [
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
        let data = try JSONSerialization.data(withJSONObject: config)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    /// The oracle's data for one tiny model: the pool's K/V rows, the new
    /// row, and the q rows. The first element of every K row carries the key
    /// index, so the scores separate and the softmax's maximum is unique.
    struct TinyData {
        var kRows: [[UInt16]]
        var vRows: [[UInt16]]
        var knew: [UInt16]
        var vnew: [UInt16]
        var qRows: [UInt16]
        var rowElems: Int
        var headDim: Int
        var qHeads: Int
        var kvHeads: Int
    }

    static func tinyData(geometry: QWModelGeometry, keys: Int) -> TinyData {
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
        var knew = [UInt16](repeating: 0, count: rowElems)
        var vnew = [UInt16](repeating: 0, count: rowElems)
        for h in 0..<geometry.kvHeads {
            for d in 0..<geometry.headDim {
                let e = h * geometry.headDim + d
                knew[e] = h16(d == 0 ? 4.0 + Float(keys) : det(keys &* 131 &+ e &* 7) * 0.05)
                vnew[e] = h16(det(keys &* 197 &+ e &* 11) * 0.5)
            }
        }
        var qRows = [UInt16](repeating: 0, count: geometry.qHeads * geometry.headDim)
        for h in 0..<geometry.qHeads {
            for d in 0..<geometry.headDim {
                qRows[h * geometry.headDim + d] =
                    h16(d == 0 ? 1.0 : det(h &* 97 &+ d &* 13) * 0.3)
            }
        }
        return TinyData(kRows: kRows, vRows: vRows, knew: knew, vnew: vnew, qRows: qRows,
                        rowElems: rowElems, headDim: geometry.headDim,
                        qHeads: geometry.qHeads, kvHeads: geometry.kvHeads)
    }

    /// The host reference: one q head's attention over the keys [0, C]. The
    /// result is the UN-NORMALISED softmax sum (the kernel's partial row),
    /// with the running maximum m and the sum l.
    static func reference(q: [Float], k: [[Float]], v: [[Float]], kvh: Int,
                          headDim: Int) -> (o: [Float], m: Float, l: Float, gap: Float) {
        let scale = Float(1.0 / Double(headDim).squareRoot())
        var scores = [Float](repeating: 0, count: k.count)
        for j in 0..<k.count {
            var dot: Float = 0
            for d in 0..<headDim { dot += q[d] * k[j][kvh * headDim + d] }
            scores[j] = dot * scale
        }
        let m = scores.max() ?? 0
        // The gap to the runner-up: the oracle's precondition — if the top
        // two scores are within rounding, the two sides may pick different
        // maxima and the comparison would be meaningless.
        var second = -Float.greatestFiniteMagnitude
        for s in scores where s != m { second = max(second, s) }
        var l: Float = 0
        var o = [Float](repeating: 0, count: headDim)
        for j in 0..<k.count {
            let p = exp(scores[j] - m)
            l += p
            for d in 0..<headDim { o[d] += p * v[j][kvh * headDim + d] }
        }
        return (o, m, l, scores.isEmpty ? 0 : m - second)
    }

    /// The combine's output for one head: the partials' O divided by L, then
    /// the 32-point normalised FWHT over each group of 32 halves — the
    /// packer's weight-side reparameterisation (C109), kept by every path.
    static func combined(_ o: [Float], l: Float, headDim: Int) -> [Float] {
        let linv = l > 0 ? 1.0 / l : 0.0
        var h = [Float](repeating: 0, count: headDim)
        for d in 0..<headDim { h[d] = Float(Float16(o[d] * linv)) }
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

    /// 1 f16 ulp of the reference value (the relative f16 spacing) — the law
    /// for a value that is STORED as f16.
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

    // ── environment ────────────────────────────────────────────────────────

    /// The cache + the partials scratch for a sequence up to `tokensCeiling`.
    static func makeEnv(hal: QWHal, geometry: QWModelGeometry, layers: [Int],
                        tokensCeiling: Int) throws -> (QWKVCache, QWArena, QWMRegion) {
        let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: geometry)
        let needPages = (tokensCeiling + 255) / 256
        let poolBytes = pageBytes * needPages * 2
        let tables = QWKVCache.tableSlotBytes * 2 * layers.count * QWKVCache.maxSeqs
        let pBytes = QWAttnDecode.decodePartialsBytes(geometry: geometry)
        let arena = try QWArena(hal: hal, weights: 0, activations: 0,
                                kv: poolBytes * layers.count + tables, gdnStates: pBytes)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: layers, mode: .f16,
                                  geometry: geometry)
        let g = arena.gdnRegion()
        let off = try g.allocate(bytes: pBytes)
        let region = QWMRegion(buffer: g.region.buffer, base: g.region.base + off, length: pBytes)
        return (cache, arena, region)
    }

    /// The partials row of one (head, split): [O headDim][m][l].
    static func partialRow(_ region: QWMRegion, head: Int, split: Int,
                            geometry: QWModelGeometry) -> (o: [Float], m: Float, l: Float) {
        let stride = geometry.headDim + 2
        let base = region.base + (head * maxSplits + split) * stride * 4
        let p = region.buffer.contents().advanced(by: base).assumingMemoryBound(to: Float.self)
        return ((0..<geometry.headDim).map { p[$0] },
                p[geometry.headDim], p[geometry.headDim + 1])
    }

    // ── law 1: the host reference at a 128-half head ───────────────────────

    func testTheGeneralDecodeMatchesAHostReferenceAt128() throws {
        let geometry = try Self.tinyGeometry()
        let hal = try QWHal(geometry: geometry)
        let C = 40                                  // 5 full phases of 8 keys
        let data = Self.tinyData(geometry: geometry, keys: C)

        let (cache, _, partials) = try Self.makeEnv(
            hal: hal, geometry: geometry, layers: Self.layers, tokensCeiling: C + 1)
        let handle = try cache.allocate(promptTokens: C + 1)
        try cache.append(handle, positions: 0..<C, layer: Self.layers[0], kind: .k,
                         src: data.kRows.flatMap { $0 })
        try cache.append(handle, positions: 0..<C, layer: Self.layers[0], kind: .v,
                         src: data.vRows.flatMap { $0 })

        let dec = try QWAttnDecode(hal: hal, geometry: geometry)
        let views = try cache.layerViews(handle, layer: Self.layers[0])
        // The output buffer is LONGER than the tensor: the combine must not
        // write past the last head's halves (law 3).
        let outHalves = geometry.qHeads * geometry.headDim
        let out = hal.device.makeBuffer(length: (outHalves + 64) * 2, options: .storageModeShared)!
        let sentinel: UInt16 = 0xBEEF
        for i in 0..<(outHalves + 64) {
            out.contents().advanced(by: i * 2).assumingMemoryBound(to: UInt16.self)[0] = sentinel
        }
        let qBuf = Self.buf(hal, data.qRows)
        let kBuf = Self.buf(hal, data.knew)
        let vBuf = Self.buf(hal, data.vnew)
        guard let (cb, enc) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
        try dec.encode(enc,
                       qgate: Self.tv(qBuf, geometry.qProjOut),
                       kv: views,
                       knew: Self.tv(kBuf, data.rowElems),
                       vnew: Self.tv(vBuf, data.rowElems),
                       partials: partials,
                       out: Self.tv(out, outHalves),
                       C: C, splits: 1)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed, "\(String(describing: cb.error))")

        // The oracle: the pool's rows plus the new row.
        let kAll = (0..<(C + 1)).map { t -> [Float] in
            (0..<data.rowElems).map { Self.f16(t < C ? data.kRows[t][$0] : data.knew[$0]) }
        }
        let vAll = (0..<(C + 1)).map { t -> [Float] in
            (0..<data.rowElems).map { Self.f16(t < C ? data.vRows[t][$0] : data.vnew[$0]) }
        }
        let partialsBuf = partials
        for h in 0..<geometry.qHeads {
            let kvh = h / (geometry.qHeads / geometry.kvHeads)
            let q = (0..<geometry.headDim).map { Self.f16(data.qRows[h * geometry.headDim + $0]) }
            let ref = Self.reference(q: q, k: kAll, v: vAll, kvh: kvh, headDim: geometry.headDim)
            XCTAssertGreaterThan(ref.gap, 1e-3,
                "head \(h): the oracle's top two scores are within rounding (\(ref.gap))")
            let got = Self.partialRow(partialsBuf, head: h, split: 0, geometry: geometry)
            let v = Self.parityViolations(got.o, ref.o)
            XCTAssertEqual(v.violations, 0,
                           "head \(h): partials (worst \(v.worstAbs) @\(v.worstIdx))")
            XCTAssertEqual(got.m, ref.m, accuracy: 1e-3, "head \(h): m")
            XCTAssertEqual(got.l, ref.l, accuracy: max(1e-2, ref.l * 1e-3), "head \(h): l")

            // The combine's output for this head: the partials normalised by
            // L, then the rotation.
            let outPtr = out.contents()
                .advanced(by: h * geometry.headDim * 2).assumingMemoryBound(to: UInt16.self)
            let gotOut = (0..<geometry.headDim).map { Self.f16(outPtr[$0]) }
            let refOut = Self.combined(got.o, l: got.l, headDim: geometry.headDim)
            let w = Self.f16UlpViolations(gotOut, refOut)
            XCTAssertEqual(w.violations, 0,
                           "head \(h): combined output (worst \(w.worstAbs) @\(w.worstIdx))")
        }

        // Law 3: the bytes past the tensor's end stay untouched — the
        // combine's head window stops at the head's own width.
        let tail = out.contents().advanced(by: outHalves * 2).assumingMemoryBound(to: UInt16.self)
        for i in 0..<64 {
            XCTAssertEqual(tail[i], sentinel, "the combine wrote past the last head")
        }

        // The new row's K/V landed in the pool — the epilogue's write stops
        // at the head's own width, so the row's halves past it are NOT
        // touched by the head's writer (the row's other head owns them).
        let rowElems = data.rowElems
        for (kind, fixture) in [(QWKVKind.k, data.knew), (QWKVKind.v, data.vnew)] {
            let ids = cache.pageIDs(handle, layer: Self.layers[0], kind: kind)
            let pg = C / 256, off = C % 256
            let views = try cache.layerViews(handle, layer: Self.layers[0])
            let base = (kind == .k ? views.k : views.v).pool
            let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: geometry)
            let got = (0..<rowElems).map { i in
                base.buffer.contents()
                    .advanced(by: base.base + Int(ids[pg]) * pageBytes + (off * rowElems + i) * 2)
                    .assumingMemoryBound(to: UInt16.self)[0]
            }
            XCTAssertEqual(got, fixture, "\(kind): the new row was not materialised into the page slot")
        }
    }

    // ── law 2: the general decode IS the frozen decode at the 27B's shape ──

    /// The 27B's geometry, on the repo's own decode fixtures: the general
    /// kernels must leave the frozen ones' bytes unchanged.
    func testTheGeneralDecodeMatchesTheFrozenDecodeAt256() throws {
        let geometry = QWModelGeometry.qwen38_27b
        let hal = try QWHal(geometry: geometry)
        let T = 257, C = T - 1
        let dir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/spike/out")
        func load(_ name: String) throws -> [UInt16] {
            let p = dir.appendingPathComponent(name)
            let d = try Data(contentsOf: p)
            return d.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
        }
        let qgate = try load("attn_qgate_dec_t257.bin")
        let knew = try load("attn_knew_t257.bin")
        let vnew = try load("attn_vnew_t257.bin")
        let kcache = try load("attn_kcache_t257.bin")
        let vcache = try load("attn_vcache_t257.bin")

        var outs: [[UInt16]] = []
        for forceGeneral in [false, true] {
            let (cache, _, partials) = try Self.makeEnv(
                hal: hal, geometry: geometry, layers: Self.layers, tokensCeiling: T)
            let handle = try cache.allocate(promptTokens: T)
            try cache.append(handle, positions: 0..<C, layer: Self.layers[0], kind: .k, src: kcache)
            try cache.append(handle, positions: 0..<C, layer: Self.layers[0], kind: .v, src: vcache)

            let dec = try QWAttnDecode(hal: hal, geometry: geometry, forceGeneral: forceGeneral)
            let views = try cache.layerViews(handle, layer: Self.layers[0])
            let outHalves = geometry.qHeads * geometry.headDim
            let out = hal.device.makeBuffer(length: outHalves * 2, options: .storageModeShared)!
            guard let (cb, enc) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
            try dec.encode(enc,
                           qgate: Self.tv(Self.buf(hal, qgate), geometry.qProjOut),
                           kv: views,
                           knew: Self.tv(Self.buf(hal, knew), geometry.kvHeads * geometry.headDim),
                           vnew: Self.tv(Self.buf(hal, vnew), geometry.kvHeads * geometry.headDim),
                           partials: partials,
                           out: Self.tv(out, outHalves),
                           C: C, splits: QWAttnDecode.splits(forKeys: C + 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            XCTAssertEqual(cb.status, .completed, "\(String(describing: cb.error))")
            outs.append(Array(UnsafeBufferPointer(
                start: out.contents().assumingMemoryBound(to: UInt16.self), count: outHalves)))
        }
        let got = outs[0].map { Self.f16($0) }
        let ref = outs[1].map { Self.f16($0) }
        XCTAssertEqual(got.count, ref.count)
        let v = Self.parityViolations(got, ref)
        XCTAssertEqual(v.violations, 0,
                       "general vs frozen at 256 (worst \(v.worstAbs) @\(v.worstIdx))")
    }
}
