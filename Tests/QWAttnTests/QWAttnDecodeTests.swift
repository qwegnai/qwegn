import XCTest
import Metal
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc
import QWKV
import QWQuant

/// C09 decode-kernel gate: `qwg_attn_decode`(+combine) vs torch fixtures
/// (gen_attn_decode_fixtures.py — fp32 oracle, f16 engine storage law; the
/// cache rows are the PAST T−1 keys, the new row is attended from knew/vnew
final class QWAttnDecodeTests: XCTestCase {

    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()
    static let layers = [3]                                 // one attn layer

    static let spikeOut = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Scripts/spike/out")

    static func u16s(_ d: Data) -> [UInt16] {
        d.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
    }
    static func load(_ name: String) throws -> Data {
        let p = spikeOut.appendingPathComponent(name)
        guard let d = FileManager.default.contents(atPath: p.path) else {
            throw NSError(domain: "fixture", code: 2, userInfo: [NSLocalizedDescriptionKey: "missing \(name)"])
        }
        return d
    }
    static func buf(_ d: Data) -> MTLBuffer {
        let b = hal.device.makeBuffer(length: max(16, d.count), options: .storageModeShared)!
        d.withUnsafeBytes { _ = memcpy(b.contents(), $0.baseAddress!, $0.count) }
        return b
    }
    static func f16(_ bits: UInt16) -> Float { QWQuant.f16tof32Bits(bits) }
    static func readF(_ b: [UInt16]) -> [Float] { b.map { f16($0) } }
    static func u16row(_ b: MTLBuffer, byteOffset: Int, _ n: Int) -> [UInt16] {
        let p = b.contents().advanced(by: byteOffset).assumingMemoryBound(to: UInt16.self)
        return (0..<n).map { p[$0] }
    }

    /// 1-ulp adjacency (C06 law) — local copy (test targets are independent).
    static func parityViolations(_ got: [Float], _ ref: [Float], absTol: Float = 1e-3)
        -> (violations: Int, worstAbs: Double, worstIdx: Int) {
        var viol = 0, worst = 0.0, worstIdx = -1
        for i in 0..<got.count {
            let d = abs(Double(got[i]) - Double(ref[i]))
            if d > worst { worst = d; worstIdx = i }
            guard got[i].isFinite && ref[i].isFinite else { viol += 1; continue }
            let spacing = max(Double(abs(got[i])), Double(abs(ref[i]))) * Double.ulpOfOne
            if d > max(Double(absTol), spacing) { viol += 1 }
        }
        return (viol, worst, worstIdx)
    }

    /// Cache + partials scratch arena for a sequence up to `tokensCeiling`.
    /// LAW: the cache pools claim the whole kv region at init — the decode
    /// partials scratch therefore rides the arena's GDN region (production
    static func makeEnv(tokensCeiling: Int) throws -> (QWKVCache, QWArena, QWMRegion) {
        let pageBytes = QWKVCache.pageBytes(mode: .f16)
        let needPages = (tokensCeiling + 255) / 256
        let poolBytes = pageBytes * needPages * 2
        let tables = QWKVCache.tableSlotBytes * 2 * layers.count * QWKVCache.maxSeqs
        let pBytes = QWAttnDecode.decodePartialsBytes(geometry: .qwen38_27b)
        let arena = try QWArena(hal: hal, weights: 0, activations: 0,
                                kv: poolBytes * layers.count + tables, gdnStates: pBytes)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: layers, mode: .f16, geometry: .qwen38_27b)
        let g = arena.gdnRegion()
        let off = try g.allocate(bytes: pBytes)
        let region = QWMRegion(buffer: g.region.buffer, base: g.region.base + off, length: pBytes)
        return (cache, arena, region)
    }

    static func tv(_ b: MTLBuffer, _ n: Int) -> QWTensorValue {
        QWTensorValue(region: QWMRegion(buffer: b, base: 0, length: max(16, n * 2)),
                      offset: 0, shape: QWTensorShape([n], .f16))
    }

    /// One full decode step (both launches) → out [6144] f16 bits.
    static func runDecode(cache: QWKVCache, handle: QWKVHandle, layer: Int,
                          qgate: Data, knew: Data, vnew: Data,
                          partials: QWMRegion, C: Int, splits: Int) throws -> [UInt16] {
        let dec = try QWAttnDecode(hal: hal, geometry: .qwen38_27b)
        let out = hal.device.makeBuffer(length: 6144 * 2, options: .storageModeShared)!
        let views = try cache.layerViews(handle, layer: layer)
        guard let (cb, enc) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
        try dec.encode(enc,
                       qgate: tv(buf(qgate), QWModelGeometry.qwen38_27b.qProjOut),
                       kv: views,
                       knew: tv(buf(knew), QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim),
                       vnew: tv(buf(vnew), QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim),
                       partials: partials,
                       out: tv(out, 6144),
                       C: C, splits: splits)
        enc.endEncoding()                                        // end-before-wait
        cb.commit()
        cb.waitUntilCompleted()
        return u16row(out, byteOffset: 0, 6144)
    }

    static let cases: [(T: Int, tag: String)] = [
        (2, ""), (257, ""), (1000, ""), (2049, ""), (4321, ""),
        (1000, "_adv30"), (257, "_ties"),
    ]

    func testDecodeParity() throws {
        for c in Self.cases {
            let C = c.T - 1
            let qgate = try Self.load("attn_qgate_dec_t\(c.T)\(c.tag).bin")
            let knew = try Self.load("attn_knew_t\(c.T)\(c.tag).bin")
            let vnew = try Self.load("attn_vnew_t\(c.T)\(c.tag).bin")
            let kcache = try Self.load("attn_kcache_t\(c.T)\(c.tag).bin")
            let vcache = try Self.load("attn_vcache_t\(c.T)\(c.tag).bin")
            let refBits = Self.u16s(try Self.load("attn_o_t\(c.T)\(c.tag).bin"))
            let ref = Self.readF(refBits)

            let (cache, _, part) = try Self.makeEnv(tokensCeiling: c.T + 256)
            let h = try cache.allocate(promptTokens: c.T)
            // burst-fill K and V caches with the PAST rows (row C — the new
            // token — is the KERNEL's job; never pre-filled by tests)
            try cache.append(h, positions: 0..<C, layer: QWAttnDecodeTests.layers[0], kind: .k,
                             src: Self.u16s(kcache))
            try cache.append(h, positions: 0..<C, layer: QWAttnDecodeTests.layers[0], kind: .v,
                             src: Self.u16s(vcache))

            let splits = QWAttnDecode.splits(forKeys: C + 1)
            let got = Self.readF(try Self.runDecode(cache: cache, handle: h, layer: QWAttnDecodeTests.layers[0],
                                                    qgate: qgate, knew: knew, vnew: vnew,
                                                    partials: part, C: C, splits: splits))
            XCTAssertEqual(got.filter { !$0.isFinite }.count, 0, "T=\(c.T)\(c.tag): non-finite out")
            let v = Self.parityViolations(got, ref)
            XCTAssertEqual(v.violations, 0,
                "T=\(c.T)\(c.tag): parity (worst \(v.worstAbs) @\(v.worstIdx)) splits=\(splits)")
        }
    }

    func testSplitsVariants() throws {
        // T=4321 (4321 keys): splits ∈ {1, 8, 64} — 64 forces ragged tails
        // (per=68 ⇒ 64·68=4352 > 4321: empty ranges must stay neutral).
        let T = 4321, C = T - 1
        let qgate = try Self.load("attn_qgate_dec_t4321.bin")
        let knew = try Self.load("attn_knew_t4321.bin")
        let vnew = try Self.load("attn_vnew_t4321.bin")
        let refBits = Self.u16s(try Self.load("attn_o_t4321.bin"))
        let ref = Self.readF(refBits)
        let kcache = Self.u16s(try Self.load("attn_kcache_t4321.bin"))
        let vcache = Self.u16s(try Self.load("attn_vcache_t4321.bin"))
        for sp in [1, 8, 64] {
            let (cache, _, part) = try Self.makeEnv(tokensCeiling: T + 256)
            let h = try cache.allocate(promptTokens: T)
            try cache.append(h, positions: 0..<C, layer: QWAttnDecodeTests.layers[0], kind: .k, src: kcache)
            try cache.append(h, positions: 0..<C, layer: QWAttnDecodeTests.layers[0], kind: .v, src: vcache)
            let got = Self.readF(try Self.runDecode(cache: cache, handle: h, layer: QWAttnDecodeTests.layers[0],
                                                    qgate: qgate, knew: knew, vnew: vnew,
                                                    partials: part, C: C, splits: sp))
            let v = Self.parityViolations(got, ref)
            XCTAssertEqual(v.violations, 0, "splits=\(sp): parity (worst \(v.worstAbs) @\(v.worstIdx))")
        }
    }

    func testNewTokenWriteReadback() throws {
        // "decode never launches a standalone KV-write": after one step the
        // NEW row (C) of both pools must byte-equal the knew/vnew fixtures.
        let T = 257, C = T - 1
        let knewData = try Self.load("attn_knew_t257.bin")
        let vnewData = try Self.load("attn_vnew_t257.bin")
        let kcache = Self.u16s(try Self.load("attn_kcache_t257.bin"))
        let vcache = Self.u16s(try Self.load("attn_vcache_t257.bin"))
        let (cache, _, part) = try Self.makeEnv(tokensCeiling: T + 256)
        let h = try cache.allocate(promptTokens: T)
        try cache.append(h, positions: 0..<C, layer: QWAttnDecodeTests.layers[0], kind: .k, src: kcache)
        try cache.append(h, positions: 0..<C, layer: QWAttnDecodeTests.layers[0], kind: .v, src: vcache)
        _ = try Self.runDecode(cache: cache, handle: h, layer: QWAttnDecodeTests.layers[0],
                               qgate: try Self.load("attn_qgate_dec_t257.bin"),
                               knew: knewData, vnew: vnewData, partials: part, C: C, splits: 2)
        let pageBytes = QWKVCache.pageBytes(mode: .f16)
        let rowElems = QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim       // 1024 halves
        for (kind, fixture) in [(QWKVKind.k, Self.u16s(knewData)),
                                (QWKVKind.v, Self.u16s(vnewData))] {
            let ids = cache.pageIDs(h, layer: QWAttnDecodeTests.layers[0], kind: kind)
            let pg = C / 256, off = C % 256
            let pool = try cache.layerViews(h, layer: QWAttnDecodeTests.layers[0])
            let base = (kind == .k ? pool.k : pool.v).pool
            let got = Self.u16row(base.buffer,
                                  byteOffset: base.base + Int(ids[pg]) * pageBytes + off * rowElems * 2,
                                  rowElems)
            XCTAssertEqual(got, fixture, "\(kind) new-token row not materialised into page slot")
        }
    }

    func testStreamVsBurstMidStream() throws {
        // spec-09 mid-stream law: KV written by prefill (burst) vs extended
        // token-by-token (streamed) must produce IDENTICAL decode outputs.
        let T = 1000, C = T - 1
        let qgate = try Self.load("attn_qgate_dec_t1000.bin")
        let knew = try Self.load("attn_knew_t1000.bin")
        let vnew = try Self.load("attn_vnew_t1000.bin")
        let refBits = Self.u16s(try Self.load("attn_o_t1000.bin"))
        let ref = Self.readF(refBits)
        let kcache = Self.u16s(try Self.load("attn_kcache_t1000.bin"))
        let vcache = Self.u16s(try Self.load("attn_vcache_t1000.bin"))
        let rowElems = QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim

        var outs: [[UInt16]] = []
        for streamed in [false, true] {
            let (cache, _, part) = try Self.makeEnv(tokensCeiling: T + 256)
            let h = try cache.allocate(promptTokens: streamed ? 1 : T)
            if streamed {
                try cache.grow(h, to: T)
                for t in 0..<C {
                    let s = Array(kcache[(t * rowElems)..<((t + 1) * rowElems)])
                    try cache.append(h, positions: t..<(t + 1), layer: QWAttnDecodeTests.layers[0], kind: .k, src: s)
                    let sv = Array(vcache[(t * rowElems)..<((t + 1) * rowElems)])
                    try cache.append(h, positions: t..<(t + 1), layer: QWAttnDecodeTests.layers[0], kind: .v, src: sv)
                }
            } else {
                try cache.append(h, positions: 0..<C, layer: QWAttnDecodeTests.layers[0], kind: .k, src: kcache)
                try cache.append(h, positions: 0..<C, layer: QWAttnDecodeTests.layers[0], kind: .v, src: vcache)
            }
            outs.append(try Self.runDecode(cache: cache, handle: h, layer: QWAttnDecodeTests.layers[0],
                                           qgate: qgate, knew: knew, vnew: vnew,
                                           partials: part, C: C, splits: 1))
        }
        XCTAssertEqual(outs[0], outs[1], "streamed ≠ burst decode output")
        let v = Self.parityViolations(Self.readF(outs[0]), ref)
        XCTAssertEqual(v.violations, 0, "mid-stream parity (worst \(v.worstAbs) @\(v.worstIdx))")
    }

    func testPageIndirectionTransparency() throws {
        // Shuffle physical pages (snapshot pins, rollback, grow re-allocates
        // fresh pages, refill rows) — the output must not notice: proves the
        // kernel follows the int32 table, not contiguity assumptions.
        let T = 1000, C = T - 1
        let qgate = try Self.load("attn_qgate_dec_t1000.bin")
        let knew = try Self.load("attn_knew_t1000.bin")
        let vnew = try Self.load("attn_vnew_t1000.bin")
        let refBits = Self.u16s(try Self.load("attn_o_t1000.bin"))
        let ref = Self.readF(refBits)
        let kcache = Self.u16s(try Self.load("attn_kcache_t1000.bin"))
        let vcache = Self.u16s(try Self.load("attn_vcache_t1000.bin"))

        let (cache, _, part) = try Self.makeEnv(tokensCeiling: T + 1024)
        let h = try cache.allocate(promptTokens: T)
        try cache.append(h, positions: 0..<C, layer: QWAttnDecodeTests.layers[0], kind: .k, src: kcache)
        try cache.append(h, positions: 0..<C, layer: QWAttnDecodeTests.layers[0], kind: .v, src: vcache)
        let idsBefore = cache.pageIDs(h, layer: QWAttnDecodeTests.layers[0], kind: .k)

        let snap = try cache.snapshot(h)
        try cache.rollback(h, by: 512)                          // drop tail pages
        try cache.grow(h, to: T)                                // grab NEW pages (pinned hold)
        let tailStart = C - 512
        try cache.append(h, positions: tailStart..<C, layer: QWAttnDecodeTests.layers[0], kind: .k,
                         src: Array(kcache[(tailStart * 1024)..<(C * 1024)]))
        try cache.append(h, positions: tailStart..<C, layer: QWAttnDecodeTests.layers[0], kind: .v,
                         src: Array(vcache[(tailStart * 1024)..<(C * 1024)]))
        cache.release(snap)
        let idsAfter = cache.pageIDs(h, layer: QWAttnDecodeTests.layers[0], kind: .k)
        XCTAssertNotEqual(idsBefore, idsAfter, "test must actually move pages (fixture premise)")

        let got = Self.readF(try Self.runDecode(cache: cache, handle: h, layer: QWAttnDecodeTests.layers[0],
                                                qgate: qgate, knew: knew, vnew: vnew,
                                                partials: part, C: C, splits: 1))
        let v = Self.parityViolations(got, ref)
        XCTAssertEqual(v.violations, 0, "indirection (worst \(v.worstAbs) @\(v.worstIdx))")
    }

    func testLongContext1MSplitsAndCombineParity() throws {
        XCTAssertEqual(QWAttnDecode.splits(forKeys: 262_144), 64)
        XCTAssertEqual(QWAttnDecode.splits(forKeys: 524_288), 128)
        XCTAssertEqual(QWAttnDecode.splits(forKeys: 1_048_576), 256)

        let T = 4321, C = T - 1
        let qgate = try Self.load("attn_qgate_dec_t4321.bin")
        let knew = try Self.load("attn_knew_t4321.bin")
        let vnew = try Self.load("attn_vnew_t4321.bin")
        let ref = Self.readF(Self.u16s(try Self.load("attn_o_t4321.bin")))
        let kcache = Self.u16s(try Self.load("attn_kcache_t4321.bin"))
        let vcache = Self.u16s(try Self.load("attn_vcache_t4321.bin"))

        let (cache, _, _) = try Self.makeEnv(tokensCeiling: T + 256)
        let h = try cache.allocate(promptTokens: T)
        try cache.append(h, positions: 0..<C, layer: QWAttnDecodeTests.layers[0], kind: .k, src: kcache)
        try cache.append(h, positions: 0..<C, layer: QWAttnDecodeTests.layers[0], kind: .v, src: vcache)

        let longPartialsBytes = QWAttnDecode.decodePartialsBytes(
            geometry: .qwen38_27b, maxSplits: QWAttnDecode.maxLongContextSplits)
        let pBuf = try XCTUnwrap(Self.hal.device.makeBuffer(length: longPartialsBytes, options: .storageModeShared))
        let longPart = QWMRegion(buffer: pBuf, base: 0, length: longPartialsBytes)

        for sp in [128, 256] {
            let got = Self.readF(try Self.runDecode(cache: cache, handle: h, layer: QWAttnDecodeTests.layers[0],
                                                    qgate: qgate, knew: knew, vnew: vnew,
                                                    partials: longPart, C: C, splits: sp))
            let v = Self.parityViolations(got, ref)
            XCTAssertEqual(v.violations, 0, "splits=\(sp): parity (worst \(v.worstAbs) @\(v.worstIdx))")
        }
    }
}
