import XCTest
import Metal
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc
import QWKV
import QWQuant

/// EXP-9 sub-item 3 parity gate: `qwg_attn_decode_batched` +
/// `qwg_attn_decode_combine_batched` (new, opt-in, ONE dispatch pair for a
/// T-query continuation chunk) vs the SHIPPING per-token loop (T separate
final class QWAttnDecodeBatchedTests: XCTestCase {

    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()
    static let layer = 3

    struct SplitMix64 {
        var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func next() -> UInt64 {
            state = state &+ 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        mutating func nextUnit() -> Float {
            Float(next() >> 11) * (1.0 / 9007199254740992.0) * 0.999998 + 0.000001
        }
        mutating func nextNormalish() -> Float {
            var s: Float = 0
            for _ in 0..<4 { s += nextUnit() }
            return (s - 2.0) * 0.9
        }
    }

    static func f16Bits(_ v: Float) -> UInt16 { QWQuant.f32tof16Bits(v) }
    static func f16(_ bits: UInt16) -> Float { QWQuant.f16tof32Bits(bits) }

    /// Synthesize `n` rows of l2-normed-per-head q̂/k̂ (matches the engine's
    /// norming contract) plus small-valued v and gate; returns raw f16 bits.
    static func synthRows(n: Int, seed: UInt64, qHeads: Int, headDim: Int) -> [UInt16] {
        var rng = SplitMix64(seed: seed)
        var out = [UInt16](repeating: 0, count: n * qHeads * 2 * headDim / (qHeads == QWModelGeometry.qwen38_27b.qHeads ? 1 : 1))
        // (kept simple: caller always passes qHeads == QWModelGeometry.qwen38_27b.qHeads or kvHeads)
        out = [UInt16](repeating: 0, count: n * qHeads * 2 * headDim)
        for row in 0..<n {
            for h in 0..<qHeads {
                var q = (0..<headDim).map { _ in rng.nextNormalish() }
                let norm = sqrtf(q.reduce(0) { $0 + $1 * $1 }) + 1e-6
                for d in 0..<headDim { q[d] /= norm }
                let base = row * qHeads * 2 * headDim + h * (2 * headDim)
                for d in 0..<headDim { out[base + d] = f16Bits(q[d]) }
                for d in 0..<headDim { out[base + headDim + d] = f16Bits(rng.nextNormalish() * 0.5) }
            }
        }
        return out
    }
    static func synthKV(n: Int, seed: UInt64) -> [UInt16] {
        var rng = SplitMix64(seed: seed)
        let H = QWModelGeometry.qwen38_27b.headDim, KVH = QWModelGeometry.qwen38_27b.kvHeads
        var out = [UInt16](repeating: 0, count: n * KVH * H)
        for row in 0..<n {
            for h in 0..<KVH {
                var k = (0..<H).map { _ in rng.nextNormalish() }
                let norm = sqrtf(k.reduce(0) { $0 + $1 * $1 }) + 1e-6
                for d in 0..<H { k[d] /= norm }
                let base = row * KVH * H + h * H
                for d in 0..<H { out[base + d] = f16Bits(k[d]) }
            }
        }
        return out
    }
    static func synthV(n: Int, seed: UInt64) -> [UInt16] {
        var rng = SplitMix64(seed: seed)
        let H = QWModelGeometry.qwen38_27b.headDim, KVH = QWModelGeometry.qwen38_27b.kvHeads
        var out = [UInt16](repeating: 0, count: n * KVH * H)
        for i in 0..<out.count { out[i] = f16Bits(rng.nextNormalish() * 0.3) }
        return out
    }

    static func buf(_ a: [UInt16]) -> MTLBuffer {
        let b = hal.device.makeBuffer(length: max(16, a.count * 2), options: .storageModeShared)!
        a.withUnsafeBufferPointer { _ = memcpy(b.contents(), $0.baseAddress!, a.count * 2) }
        return b
    }
    static func tv(_ b: MTLBuffer, _ n: Int) -> QWTensorValue {
        QWTensorValue(region: QWMRegion(buffer: b, base: 0, length: max(16, n * 2)),
                      offset: 0, shape: QWTensorShape([n], .f16))
    }
    static func readOut(_ b: MTLBuffer, _ n: Int) -> [Float] {
        let p = b.contents().assumingMemoryBound(to: UInt16.self)
        return (0..<n).map { f16(p[$0]) }
    }

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

    /// Fresh cache+arena for a sequence with `pos` ALREADY-cached rows and
    /// room for `pos+T` total. Pre-fills rows [0,pos) with synthetic k/v
    /// (the "prefix"); rows [pos,pos+T) are left for the kernels to write
    static func makeEnv(pos: Int, T: Int, splits: Int, newRows: (k: [UInt16], v: [UInt16])? = nil,
                        streamPrefix: Bool = false) throws
        -> (QWKVCache, QWKVHandle, QWMRegion) {
        let ceiling = pos + T + 256
        let pageBytes = QWKVCache.pageBytes(mode: .f16)
        let needPages = (ceiling + 255) / 256
        let poolBytes = pageBytes * needPages * 2
        let tables = QWKVCache.tableSlotBytes * 2 * 1 * QWKVCache.maxSeqs
        let pBytes = QWAttnDecode.decodePartialsBytesBatched(T: max(T, 1), geometry: .qwen38_27b)
        let arena = try QWArena(hal: hal, weights: 0, activations: 0,
                                kv: poolBytes + tables, gdnStates: pBytes)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: [layer], mode: .f16, geometry: .qwen38_27b)
        let h = try cache.allocate(promptTokens: ceiling)
        if streamPrefix {
            for start in stride(from: 0, to: pos, by: 1024) {
                let count = min(1024, pos - start)
                let keys = synthKV(n: count, seed: 0x1111 &+ UInt64(start))
                let values = synthV(n: count, seed: 0x2222 &+ UInt64(start))
                try cache.append(h, positions: start..<(start + count), layer: layer, kind: .k, src: keys)
                try cache.append(h, positions: start..<(start + count), layer: layer, kind: .v, src: values)
            }
        } else if pos > 0 {
            let kPrefix = synthKV(n: pos, seed: 0x1111 &+ UInt64(pos))
            let vPrefix = synthV(n: pos, seed: 0x2222 &+ UInt64(pos))
            try cache.append(h, positions: 0..<pos, layer: layer, kind: .k, src: kPrefix)
            try cache.append(h, positions: 0..<pos, layer: layer, kind: .v, src: vPrefix)
        }
        if let nr = newRows, T > 0 {
            try cache.append(h, positions: pos..<(pos + T), layer: layer, kind: .k, src: nr.k)
            try cache.append(h, positions: pos..<(pos + T), layer: layer, kind: .v, src: nr.v)
        }
        let g = arena.gdnRegion()
        let off = try g.allocate(bytes: pBytes)
        let region = QWMRegion(buffer: g.region.buffer, base: g.region.base + off, length: pBytes)
        return (cache, h, region)
    }

    /// Reference leg: T separate `encode(...)` calls (today's shipping
    /// per-token Swift loop, bit-for-bit what the driver executes).
    static func runPerToken(pos: Int, T: Int, qgate: [UInt16], knew: [UInt16], vnew: [UInt16],
                            splits: Int) throws -> [Float] {
        let (cache, h, singlePart) = try makeEnv(pos: pos, T: T, splits: splits, newRows: (knew, vnew))
        let dec = try QWAttnDecode(hal: hal, geometry: .qwen38_27b)
        var outAll = [Float](repeating: 0, count: T * QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim)
        for t in 0..<T {
            let views = try cache.layerViews(h, layer: layer)
            let qRow = Array(qgate[(t * QWModelGeometry.qwen38_27b.qProjOut)..<((t + 1) * QWModelGeometry.qwen38_27b.qProjOut)])
            let kRow = Array(knew[(t * QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim)..<((t + 1) * QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim)])
            let vRow = Array(vnew[(t * QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim)..<((t + 1) * QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim)])
            let out = hal.device.makeBuffer(length: QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim * 2, options: .storageModeShared)!
            guard let (cb, enc) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
            try dec.encode(enc, qgate: tv(buf(qRow), QWModelGeometry.qwen38_27b.qProjOut), kv: views,
                           knew: tv(buf(kRow), QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim),
                           vnew: tv(buf(vRow), QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim),
                           partials: singlePart, out: tv(out, QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim),
                           C: pos + t, splits: splits)
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            let rowF = readOut(out, QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim)
            for i in 0..<rowF.count { outAll[t * QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim + i] = rowF[i] }
        }
        return outAll
    }

    /// New leg: ONE `encodeBatched(...)` dispatch pair for the whole chunk.
    static func runBatched(pos: Int, T: Int, qgate: [UInt16], knew: [UInt16], vnew: [UInt16],
                           splits: Int, sharedPrefix: Bool = false) throws -> [Float] {
        let (cache, h, part) = try makeEnv(pos: pos, T: T, splits: splits, newRows: (knew, vnew))
        let dec = try QWAttnDecode(hal: hal, sharedPrefixRows: sharedPrefix ? T : nil, geometry: .qwen38_27b)
        let views = try cache.layerViews(h, layer: layer)
        let out = hal.device.makeBuffer(length: T * QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim * 2, options: .storageModeShared)!
        guard let (cb, enc) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
        try dec.encodeBatched(enc, qgate: tv(buf(qgate), T * QWModelGeometry.qwen38_27b.qProjOut), kv: views,
                              knew: tv(buf(knew), T * QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim),
                              vnew: tv(buf(vnew), T * QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim),
                              partials: part, out: tv(out, T * QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim),
                              cBase: pos, T: T, splits: splits, sharedPrefix: sharedPrefix)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        return readOut(out, T * QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim)
    }

    func testSharedPrefixB2B3ParityAtCausalAndSplitBoundaries() throws {
        for T in [2, 3] {
            for pos in [0, 1, 10, 127, 254, 255, 511, 1024] {
                let q = Self.synthRows(n: T, seed: 0xA51, qHeads: QWModelGeometry.qwen38_27b.qHeads,
                                      headDim: QWModelGeometry.qwen38_27b.headDim)
                let k = Self.synthKV(n: T, seed: 0xB62), v = Self.synthV(n: T, seed: 0xC73)
                for splits in [1, 8, 64] {
                    let ref = try Self.runPerToken(pos: pos, T: T, qgate: q, knew: k, vnew: v, splits: splits)
                    let got = try Self.runBatched(pos: pos, T: T, qgate: q, knew: k, vnew: v,
                                                 splits: splits, sharedPrefix: true)
                    let parity = Self.parityViolations(got, ref, absTol: 5e-4)
                    XCTAssertEqual(parity.violations, 0, "B=\(T) C=\(pos) splits=\(splits) worst=\(parity.worstAbs)")
                    func orderedBits(_ value: Float) -> Int {
                        let bits = Self.f16Bits(value)
                        return bits & 0x8000 == 0 ? 0x8000 + Int(bits) : 0x8000 - Int(bits & 0x7FFF)
                    }
                    let worstULP = zip(got, ref).map { abs(orderedBits($0.0) - orderedBits($0.1)) }.max() ?? 0
                    XCTAssertEqual(worstULP, 0, "exact FP16 parity B=\(T) C=\(pos) splits=\(splits)")
                }
            }
        }
    }

    func testSharedPrefixExactParityWithRMSScaleInputs() throws {
        for T in [2, 3] {
            for pos in [10, 254, 1024] {
                for seed in [UInt64(0xA51), 0xE83, 0x199] {
                    let q = Self.synthRows(n: T, seed: seed, qHeads: QWModelGeometry.qwen38_27b.qHeads,
                                          headDim: QWModelGeometry.qwen38_27b.headDim).enumerated().map { index, bits in
                        Self.f16Bits(Self.f16(bits) * (index % 512 < 256 ? 16 : 1))
                    }
                    let k = Self.synthKV(n: T, seed: seed &+ 1).map { Self.f16Bits(Self.f16($0) * 16) }
                    let v = Self.synthV(n: T, seed: seed &+ 2)
                    let splits = QWAttnDecode.splits(forKeys: pos + T)
                    let ref = try Self.runPerToken(pos: pos, T: T, qgate: q, knew: k, vnew: v, splits: splits)
                    let got = try Self.runBatched(pos: pos, T: T, qgate: q, knew: k, vnew: v,
                                                 splits: splits, sharedPrefix: true)
                    let mismatches = zip(got, ref).filter { $0.0 != $0.1 || !$0.0.isFinite || !$0.1.isFinite }.count
                    XCTAssertEqual(mismatches, 0, "RMS-scale exact output B=\(T) C=\(pos) seed=\(seed)")
                }
            }
        }
    }

    func testSharedPrefixComponentTiming() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["QW_PERF_GATE"] == "1",
                          "perf gate — run via `make perf`, not the unit path")
        for pos in [1024, 32768] {
            for T in [2, 3] {
                let splits = QWAttnDecode.splits(forKeys: pos + T)
                let q = Self.synthRows(n: T, seed: 0xA51, qHeads: QWModelGeometry.qwen38_27b.qHeads, headDim: QWModelGeometry.qwen38_27b.headDim)
                let k = Self.synthKV(n: T, seed: 0xB62), v = Self.synthV(n: T, seed: 0xC73)
                let (cache, handle, partials) = try Self.makeEnv(pos: pos, T: T, splits: splits,
                                                              newRows: (k, v), streamPrefix: true)
                let views = try cache.layerViews(handle, layer: Self.layer)
                let decode = try QWAttnDecode(hal: Self.hal, sharedPrefixRows: T, geometry: .qwen38_27b)
                let qv = Self.tv(Self.buf(q), q.count), kv = Self.tv(Self.buf(k), k.count), vv = Self.tv(Self.buf(v), v.count)
                let n = T * QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim
                let refBuffer = try XCTUnwrap(Self.hal.device.makeBuffer(length: n * 2, options: .storageModeShared))
                let sharedBuffer = try XCTUnwrap(Self.hal.device.makeBuffer(length: n * 2, options: .storageModeShared))
                var host = [[Double](), [Double]()], gpu = [[Double](), [Double]()]
                func row(_ value: QWTensorValue, _ index: Int, _ columns: Int) -> QWTensorValue {
                    QWTensorValue(region: value.region, offset: value.offset + index * columns * 2,
                                  shape: QWTensorShape([columns], .f16))
                }
                for iteration in 0..<40 {
                    for arm in (iteration % 2 == 0 ? [0, 1] : [1, 0]) {
                        let started = DispatchTime.now().uptimeNanoseconds
                        let cb = try XCTUnwrap(Self.hal.queue.makeCommandBuffer())
                        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
                        let output = Self.tv(arm == 0 ? refBuffer : sharedBuffer, n)
                        func encode() throws {
                            defer { enc.endEncoding() }
                            if arm == 1 {
                                try decode.encodeBatched(enc, qgate: qv, kv: views, knew: kv, vnew: vv,
                                    partials: partials, out: output, cBase: pos, T: T, splits: splits, sharedPrefix: true)
                            } else {
                                for query in 0..<T {
                                    try decode.encode(enc, qgate: row(qv, query, QWModelGeometry.qwen38_27b.qProjOut), kv: views,
                                        knew: row(kv, query, 1024), vnew: row(vv, query, 1024), partials: partials,
                                        out: row(output, query, 6144), C: pos + query,
                                        splits: QWAttnDecode.splits(forKeys: pos + query + 1))
                                    enc.memoryBarrier(scope: .buffers)
                                }
                            }
                        }
                        try encode()
                        cb.commit()
                        cb.waitUntilCompleted()
                        guard cb.status == .completed else {
                            throw QWAttnDecodeError.badShape("component GPU error: \(String(describing: cb.error))")
                        }
                        if iteration >= 10 {
                            host[arm].append(Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000)
                            let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000
                            guard ms > 0 && ms.isFinite else {
                                throw QWAttnDecodeError.badShape("component GPU timestamps unavailable")
                            }
                            gpu[arm].append(ms)
                        }
                    }
                    let parity = Self.parityViolations(Self.readOut(sharedBuffer, n), Self.readOut(refBuffer, n), absTol: 5e-4)
                    XCTAssertEqual(parity.violations, 0, "B=\(T) C=\(pos) worst=\(parity.worstAbs)")
                    let mismatches = zip(Self.readOut(sharedBuffer, n), Self.readOut(refBuffer, n))
                        .filter { $0.0 != $0.1 || !$0.0.isFinite || !$0.1.isFinite }.count
                    XCTAssertEqual(mismatches, 0, "exact component output B=\(T) C=\(pos) iteration=\(iteration)")
                }
                for (name, times) in [("host", host), ("gpu", gpu)] {
                    let ref = times[0].sorted(), shared = times[1].sorted()
                    print("SHARED_ATTN_COMPONENT C=\(pos) B=\(T) \(name) reference p50/p95=\(ref[14])/\(ref[28]) ms shared=\(shared[14])/\(shared[28]) ms gain=\(ref[14] / shared[14])x; 10 discarded/30 measured, alternating order; not L1 qualification")
                }
                cache.finish(handle)
            }
        }
    }

    func testSharedPrefixRejectsInvalidConfigurationAndStorage() throws {
        XCTAssertThrowsError(try QWAttnDecode(hal: Self.hal, sharedPrefixRows: 1, geometry: .qwen38_27b))
        let T = 2, pos = 10, splits = 8
        let (cache, handle, partials) = try Self.makeEnv(pos: pos, T: T, splits: splits)
        let views = try cache.layerViews(handle, layer: Self.layer)
        let q = Self.tv(Self.buf([UInt16](repeating: 0, count: T * 12288)), T * 12288)
        let k = Self.tv(Self.buf([UInt16](repeating: 0, count: T * 1024)), T * 1024)
        let output = Self.tv(Self.buf([UInt16](repeating: 0, count: T * 6144)), T * 6144)
        let decoder = try QWAttnDecode(hal: Self.hal, sharedPrefixRows: T, geometry: .qwen38_27b)
        let unconfigured = try QWAttnDecode(hal: Self.hal, geometry: .qwen38_27b)
        let cb = try XCTUnwrap(Self.hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        defer { enc.endEncoding(); cache.finish(handle) }
        func rejected(_ decoder: QWAttnDecode, _ query: QWTensorValue, _ scratch: QWMRegion,
                      _ out: QWTensorValue, _ count: Int = 2) {
            XCTAssertThrowsError(try decoder.encodeBatched(enc, qgate: query, kv: views, knew: k, vnew: k,
                partials: scratch, out: out, cBase: pos, T: count, splits: splits, sharedPrefix: true))
        }
        rejected(unconfigured, q, partials, output)
        rejected(decoder, q, partials, output, 3)
        rejected(decoder, q, QWMRegion(buffer: partials.buffer, base: partials.base, length: 4), output)
        rejected(decoder, q, partials, QWTensorValue(region: q.region, offset: q.offset,
                                                   shape: output.shape))
        rejected(decoder, QWTensorValue(region: q.region, offset: 2, shape: q.shape), partials, output)
    }

    func testSharedPrefixCausalMaskExcludesLargeFutureValues() throws {
        for T in [2, 3] {
            let q = [UInt16](repeating: 0, count: T * 12288)
            let k = [UInt16](repeating: 0, count: T * 1024)
            let v = (0..<T).flatMap { row in
                [UInt16](repeating: Self.f16Bits(powf(100, Float(row))), count: 1024)
            }
            let got = try Self.runBatched(pos: 1, T: T, qgate: q, knew: k, vnew: v, splits: 8, sharedPrefix: true)
            let expected = try Self.runPerToken(pos: 1, T: T, qgate: q, knew: k, vnew: v, splits: 8)
            XCTAssertEqual(got, expected, "prewritten future rows must not enter earlier queries")
            // Outputs live in the FWHT32-rotated domain (HQ4 law, 681ae3f):
            // legitimate values reach ~sqrt(32) x the pre-rotation bound of 0.39,
            // so the tripwire sits at 3.0 — a future-row leak (mixing v=100)
            // would land the whole block in the tens, far above this ceiling.
            XCTAssertTrue(got.prefix(6144).allSatisfy { $0.isFinite && abs($0) < 3 },
                          "the first query must not see future values 100 or 10000")
        }
    }

    func runCase(pos: Int, T: Int, tag: String) throws {
        let qgate = Self.synthRows(n: T, seed: 0xA5A5 ^ UInt64(pos) ^ (UInt64(T) << 20),
                                   qHeads: QWModelGeometry.qwen38_27b.qHeads, headDim: QWModelGeometry.qwen38_27b.headDim)
        let knew = Self.synthKV(n: T, seed: 0xB6B6 ^ UInt64(pos))
        let vnew = Self.synthV(n: T, seed: 0xC7C7 ^ UInt64(pos))
        let splits = QWAttnDecode.splits(forKeys: pos + T)
        let ref = try Self.runPerToken(pos: pos, T: T, qgate: qgate, knew: knew, vnew: vnew, splits: splits)
        let got = try Self.runBatched(pos: pos, T: T, qgate: qgate, knew: knew, vnew: vnew, splits: splits)
        XCTAssertEqual(got.filter { !$0.isFinite }.count, 0, "\(tag): non-finite output")
        let p = Self.parityViolations(got, ref)
        XCTAssertEqual(p.violations, 0, "\(tag): parity vs per-token loop (worst \(p.worstAbs) @\(p.worstIdx))")
    }

    /// Plain continuation: a mid-sequence chunk, no page-boundary interplay.
    func testContinuation() throws { try runCase(pos: 10, T: 8, tag: "continuation pos=10 T=8") }

    /// Ragged chunk sizes (not powers of 2, not dividing the 256-row page size).
    func testRaggedChunk() throws {
        for T in [1, 3, 7, 13] { try runCase(pos: 5, T: T, tag: "ragged T=\(T)") }
    }

    /// Page-boundary crossing: pos chosen so [pos, pos+T) straddles the
    /// 256-row page boundary (prefix ends just before 256, chunk crosses it).
    func testPageBoundaryCrossing() throws { try runCase(pos: 250, T: 12, tag: "page-boundary pos=250 T=12") }

    /// "Growth": TWO sequential batched chunks against the SAME growing
    /// cache (chunk 2's prefix includes chunk 1's newly-written rows) must
    /// match the per-token reference computed over the SAME two-chunk growth.
    func testSequentialGrowth() throws {
        let pos0 = 20, T1 = 6, T2 = 9
        let qgate1 = Self.synthRows(n: T1, seed: 0xD1, qHeads: QWModelGeometry.qwen38_27b.qHeads, headDim: QWModelGeometry.qwen38_27b.headDim)
        let knew1 = Self.synthKV(n: T1, seed: 0xD2), vnew1 = Self.synthV(n: T1, seed: 0xD3)
        let qgate2 = Self.synthRows(n: T2, seed: 0xD4, qHeads: QWModelGeometry.qwen38_27b.qHeads, headDim: QWModelGeometry.qwen38_27b.headDim)
        let knew2 = Self.synthKV(n: T2, seed: 0xD5), vnew2 = Self.synthV(n: T2, seed: 0xD6)

        // Reference: per-token loop across BOTH chunks on one growing cache.
        let splits1 = QWAttnDecode.splits(forKeys: pos0 + T1)
        let (cacheR, hR, partR) = try Self.makeEnv(pos: pos0, T: T1 + T2, splits: 64)
        // Mirror the driver's call order: `encodePagedKVWrite` for a chunk
        // runs ONCE, before that chunk's attention dispatches (batched or
        // per-token) — pre-write chunk 1's rows now, chunk 2's rows later.
        try cacheR.append(hR, positions: pos0..<(pos0 + T1), layer: Self.layer, kind: .k, src: knew1)
        try cacheR.append(hR, positions: pos0..<(pos0 + T1), layer: Self.layer, kind: .v, src: vnew1)
        let decR = try QWAttnDecode(hal: Self.hal, geometry: .qwen38_27b)
        var ref = [Float](repeating: 0, count: (T1 + T2) * QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim)
        func step(_ t: Int, _ pos: Int, _ qg: [UInt16], _ kn: [UInt16], _ vn: [UInt16], _ splits: Int) throws {
            let views = try cacheR.layerViews(hR, layer: Self.layer)
            let out = Self.hal.device.makeBuffer(length: QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim * 2, options: .storageModeShared)!
            guard let (cb, enc) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
            try decR.encode(enc, qgate: Self.tv(Self.buf(qg), QWModelGeometry.qwen38_27b.qProjOut), kv: views,
                            knew: Self.tv(Self.buf(kn), QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim),
                            vnew: Self.tv(Self.buf(vn), QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim),
                            partials: partR, out: Self.tv(out, QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim),
                            C: pos, splits: splits)
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            let rowF = Self.readOut(out, QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim)
            for i in 0..<rowF.count { ref[t * QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim + i] = rowF[i] }
        }
        for t in 0..<T1 {
            let s = QWModelGeometry.qwen38_27b.qProjOut, kvS = QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim
            try step(t, pos0 + t, Array(qgate1[(t*s)..<((t+1)*s)]), Array(knew1[(t*kvS)..<((t+1)*kvS)]), Array(vnew1[(t*kvS)..<((t+1)*kvS)]), splits1)
        }
        let splits2 = QWAttnDecode.splits(forKeys: pos0 + T1 + T2)
        try cacheR.append(hR, positions: (pos0 + T1)..<(pos0 + T1 + T2), layer: Self.layer, kind: .k, src: knew2)
        try cacheR.append(hR, positions: (pos0 + T1)..<(pos0 + T1 + T2), layer: Self.layer, kind: .v, src: vnew2)
        for t in 0..<T2 {
            let s = QWModelGeometry.qwen38_27b.qProjOut, kvS = QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim
            try step(T1 + t, pos0 + T1 + t, Array(qgate2[(t*s)..<((t+1)*s)]), Array(knew2[(t*kvS)..<((t+1)*kvS)]), Array(vnew2[(t*kvS)..<((t+1)*kvS)]), splits2)
        }

        // New: TWO batched chunk calls on a SEPARATE, fresh growing cache.
        let (cacheB, hB, partB) = try Self.makeEnv(pos: pos0, T: T1 + T2, splits: 64)
        try cacheB.append(hB, positions: pos0..<(pos0 + T1), layer: Self.layer, kind: .k, src: knew1)
        try cacheB.append(hB, positions: pos0..<(pos0 + T1), layer: Self.layer, kind: .v, src: vnew1)
        let decB = try QWAttnDecode(hal: Self.hal, geometry: .qwen38_27b)
        let viewsB1 = try cacheB.layerViews(hB, layer: Self.layer)
        let out1 = Self.hal.device.makeBuffer(length: T1 * QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim * 2, options: .storageModeShared)!
        guard let (cb1, enc1) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
        try decB.encodeBatched(enc1, qgate: Self.tv(Self.buf(qgate1), T1 * QWModelGeometry.qwen38_27b.qProjOut), kv: viewsB1,
                               knew: Self.tv(Self.buf(knew1), T1 * QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim),
                               vnew: Self.tv(Self.buf(vnew1), T1 * QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim),
                               partials: partB, out: Self.tv(out1, T1 * QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim),
                               cBase: pos0, T: T1, splits: splits1)
        enc1.endEncoding(); cb1.commit(); cb1.waitUntilCompleted()

        let viewsB2 = try cacheB.layerViews(hB, layer: Self.layer)
        try cacheB.append(hB, positions: (pos0 + T1)..<(pos0 + T1 + T2), layer: Self.layer, kind: .k, src: knew2)
        try cacheB.append(hB, positions: (pos0 + T1)..<(pos0 + T1 + T2), layer: Self.layer, kind: .v, src: vnew2)
        let out2 = Self.hal.device.makeBuffer(length: T2 * QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim * 2, options: .storageModeShared)!
        guard let (cb2, enc2) = Self.hal.begin() else { throw NSError(domain: "hal", code: 1) }
        try decB.encodeBatched(enc2, qgate: Self.tv(Self.buf(qgate2), T2 * QWModelGeometry.qwen38_27b.qProjOut), kv: viewsB2,
                               knew: Self.tv(Self.buf(knew2), T2 * QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim),
                               vnew: Self.tv(Self.buf(vnew2), T2 * QWModelGeometry.qwen38_27b.kvHeads * QWModelGeometry.qwen38_27b.headDim),
                               partials: partB, out: Self.tv(out2, T2 * QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim),
                               cBase: pos0 + T1, T: T2, splits: splits2)
        enc2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()

        var got = Self.readOut(out1, T1 * QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim)
        got.append(contentsOf: Self.readOut(out2, T2 * QWModelGeometry.qwen38_27b.qHeads * QWModelGeometry.qwen38_27b.headDim))

        XCTAssertEqual(got.filter { !$0.isFinite }.count, 0, "growth: non-finite output")
        let p = Self.parityViolations(got, ref)
        XCTAssertEqual(p.violations, 0, "growth: parity vs per-token loop across two sequential chunks (worst \(p.worstAbs) @\(p.worstIdx))")
    }
}
