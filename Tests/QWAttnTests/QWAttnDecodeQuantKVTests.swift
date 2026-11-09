// 4c compressed-KV parity (q8 + q4).
//
// Same golden fixtures as `QWAttnDecodeTests` (Scripts/spike/out/attn_*.bin:
// Full note: docs/STYLE_NOTES.md#QWAttnDecodeQuantKVTests.swift.
import XCTest
import Metal
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc
import QWKV
import QWQuant

final class QWAttnDecodeQuantKVTests: XCTestCase {

    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()
    static let layers = [3]
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
    static func u16s(_ d: Data) -> [UInt16] {
        d.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
    }
    static func f16(_ bits: UInt16) -> Float { QWQuant.f16tof32Bits(bits) }
    static func readF(_ b: [UInt16]) -> [Float] { b.map { f16($0) } }
    static func buf(_ d: Data) -> MTLBuffer {
        let b = hal.device.makeBuffer(length: max(16, d.count), options: .storageModeShared)!
        d.withUnsafeBytes { _ = memcpy(b.contents(), $0.baseAddress!, $0.count) }
        return b
    }
    static func tv(_ b: MTLBuffer, _ n: Int) -> QWTensorValue {
        QWTensorValue(region: QWMRegion(buffer: b, base: 0, length: max(16, n * 2)),
                      offset: 0, shape: QWTensorShape([n], .f16))
    }

    /// Stride identities: Swift layout authority ≡ kernel #defines (the
    /// numbers in qwattn_decode.metal / qwgemm_batched.metal headers).
    func testLayoutIdentities() throws {
        XCTAssertEqual(QWKVCache.pageBytes(mode: .q8), 264192, "≡ QW_Q8_PGE")
        XCTAssertEqual(QWKVCompressedLayout(mode: .q8).dataBytesPerPage, 262144, "≡ QW_Q8_HDR")
        XCTAssertEqual(QWKVCache.pageBytes(mode: .q4), 139264, "≡ QW_Q4_PGE")
        XCTAssertEqual(QWKVCompressedLayout(mode: .q4).dataBytesPerPage, 131072, "≡ QW_Q4_HDR")
    }

    static func makeEnv(mode: QWKVMode, tokensCeiling: Int) throws -> (QWKVCache, QWArena, QWMRegion) {
        let pageBytes = QWKVCache.pageBytes(mode: mode)
        let needPages = (tokensCeiling + 255) / 256
        let poolBytes = pageBytes * needPages * 2
        let tables = QWKVCache.tableSlotBytes * 2 * layers.count * QWKVCache.maxSeqs
        let pBytes = QWAttnDecode.decodePartialsBytes(geometry: .qwen38_27b)
        let arena = try QWArena(hal: hal, weights: 0, activations: 0,
                                kv: poolBytes * layers.count + tables, gdnStates: pBytes)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: layers, mode: mode, geometry: .qwen38_27b)
        let g = arena.gdnRegion()
        let off = try g.allocate(bytes: pBytes)
        let region = QWMRegion(buffer: g.region.buffer, base: g.region.base + off, length: pBytes)
        return (cache, arena, region)
    }

    static func runDecode(cache: QWKVCache, handle: QWKVHandle, layer: Int,
                          qgate: Data, knew: Data, vnew: Data,
                          partials: QWMRegion, C: Int, splits: Int,
                          mode: QWKVMode = .f16) throws -> [UInt16] {
        // 4c: pool addressing is a pipeline VARIANT (compile-time template);
        // the decoder must be built for the pool's mode.
        let dec = try QWAttnDecode(hal: hal, kvMode: mode, geometry: .qwen38_27b)
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
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        _ = cb
        let p = out.contents().assumingMemoryBound(to: UInt16.self)
        return (0..<6144).map { p[$0] }
    }

    // MARK: - byte identity: the GPU row-C store ≡ Swift reference quantizer

    func testGPUStoredBytesMatchReferenceQuantizerQ8() throws {
        let T = 1000, C = T - 1
        let qgate = try Self.load("attn_qgate_dec_t\(T).bin")
        let knew = try Self.load("attn_knew_t\(T).bin")
        let vnew = try Self.load("attn_vnew_t\(T).bin")
        let kcache = try Self.u16s(Self.load("attn_kcache_t\(T).bin"))
        let vcache = try Self.u16s(Self.load("attn_vcache_t\(T).bin"))
        let (cache, _, part) = try Self.makeEnv(mode: .q8, tokensCeiling: T + 256)
        let h = try cache.allocate(promptTokens: T)
        try cache.append(h, positions: 0..<C, layer: Self.layers[0], kind: .k, src: kcache)
        try cache.append(h, positions: 0..<C, layer: Self.layers[0], kind: .v, src: vcache)
        let splits = QWAttnDecode.splits(forKeys: C + 1)
        _ = try Self.runDecode(cache: cache, handle: h, layer: Self.layers[0], qgate: qgate,
                               knew: knew, vnew: vnew, partials: part, C: C, splits: splits, mode: .q8)
        // Reference bytes for row C (K and V) from the f16 staging rows.
        let layout = QWKVCompressedLayout(mode: .q8)
        var mismatches = 0
        let pairs: [(row: [UInt16], kind: QWKVKind)] = [(Self.u16s(knew), .k), (Self.u16s(vnew), .v)]
        for (row, kind) in pairs {
            let views = try cache.layerViews(h, layer: Self.layers[0])
            let view = kind == .k ? views.k : views.v
            // Row C (writtenTokens-1): CPU append wrote it, the kernel epilogue
        // must have rewritten the SAME bytes — this is the CPU≡GPU gate.
        let page = C / 256, slot = C % 256
            let pageID = view.table.region.buffer.contents()
                .advanced(by: view.table.region.base + view.table.offset + page * 4)
                .assumingMemoryBound(to: Int32.self)[0]
            let pageBase = view.pool.buffer.contents()
                .advanced(by: view.pool.base + Int(pageID) * QWKVCache.pageBytes(mode: .q8))
            // Expected payload + header.
            var f = [Float](repeating: 0, count: 1024)
            var q = [Int8](repeating: 0, count: 1024)
            var s = [Float16](repeating: 0, count: 4)
            for i in 0..<1024 { f[i] = Float(Float16(bitPattern: row[i])) }
            try QWKVCompressedLayout.quantizeQ8(input: f, qOut: &q, scalesOut: &s)
            for i in 0..<1024 {
                let got = pageBase.advanced(by: slot * 1024 + i).assumingMemoryBound(to: Int8.self)[0]
                if got != q[i] { mismatches += 1 }
            }
            for hh in 0..<4 {
                let got = pageBase.advanced(by: layout.dataBytesPerPage + slot * 8 + hh * 2)
                    .assumingMemoryBound(to: Float16.self)[0]
                if got.bitPattern != s[hh].bitPattern { mismatches += 1 }
            }
        }
        XCTAssertEqual(mismatches, 0, "GPU q8 row-C store must match quantizeQ8 byte-for-byte")
    }

    func testGPUStoredBytesMatchReferenceQuantizerQ4() throws {
        let T = 1000, C = T - 1
        let qgate = try Self.load("attn_qgate_dec_t\(T).bin")
        let knew = try Self.load("attn_knew_t\(T).bin")
        let vnew = try Self.load("attn_vnew_t\(T).bin")
        let kcache = try Self.u16s(Self.load("attn_kcache_t\(T).bin"))
        let vcache = try Self.u16s(Self.load("attn_vcache_t\(T).bin"))
        let (cache, _, part) = try Self.makeEnv(mode: .q4, tokensCeiling: T + 256)
        let h = try cache.allocate(promptTokens: T)
        try cache.append(h, positions: 0..<C, layer: Self.layers[0], kind: .k, src: kcache)
        try cache.append(h, positions: 0..<C, layer: Self.layers[0], kind: .v, src: vcache)
        let splits = QWAttnDecode.splits(forKeys: C + 1)
        _ = try Self.runDecode(cache: cache, handle: h, layer: Self.layers[0], qgate: qgate,
                               knew: knew, vnew: vnew, partials: part, C: C, splits: splits, mode: .q4)
        let layout = QWKVCompressedLayout(mode: .q4)
        var mismatches = 0
        let pairs: [(row: [UInt16], kind: QWKVKind)] = [(Self.u16s(knew), .k), (Self.u16s(vnew), .v)]
        for (row, kind) in pairs {
            let views = try cache.layerViews(h, layer: Self.layers[0])
            let view = kind == .k ? views.k : views.v
            let page = C / 256, slot = C % 256
            let pageID = view.table.region.buffer.contents()
                .advanced(by: view.table.region.base + view.table.offset + page * 4)
                .assumingMemoryBound(to: Int32.self)[0]
            let pageBase = view.pool.buffer.contents()
                .advanced(by: view.pool.base + Int(pageID) * QWKVCache.pageBytes(mode: .q4))
            var f = [Float](repeating: 0, count: 1024)
            var q = [UInt8](repeating: 0, count: 512)
            var sc = [Float16](repeating: 0, count: 8)
            var mn = [Float16](repeating: 0, count: 8)
            for i in 0..<1024 { f[i] = Float(Float16(bitPattern: row[i])) }
            try QWKVCompressedLayout.quantizeQ4(input: f, qOut: &q, scalesOut: &sc, minsOut: &mn)
            for i in 0..<512 {
                let got = pageBase.advanced(by: slot * 512 + i).assumingMemoryBound(to: UInt8.self)[0]
                if got != q[i] { mismatches += 1 }
            }
            for g in 0..<8 {
                let hgot = pageBase.advanced(by: layout.dataBytesPerPage + slot * 32 + g * 4)
                    .assumingMemoryBound(to: Float16.self)
                if hgot[0].bitPattern != sc[g].bitPattern || hgot[1].bitPattern != mn[g].bitPattern {
                    mismatches += 1
                }
            }
        }
        XCTAssertEqual(mismatches, 0, "GPU q4 row-C store must match quantizeQ4 byte-for-byte")
    }

    // MARK: - output parity vs the f16 golden

    func testDecodedOutputParityQ8() throws {
        for T in [2, 257, 1000, 4321] {
            let C = T - 1
            let qgate = try Self.load("attn_qgate_dec_t\(T).bin")
            let knew = try Self.load("attn_knew_t\(T).bin")
            let vnew = try Self.load("attn_vnew_t\(T).bin")
            let kcache = try Self.u16s(Self.load("attn_kcache_t\(T).bin"))
            let vcache = try Self.u16s(Self.load("attn_vcache_t\(T).bin"))
            let ref = Self.readF(try Self.u16s(Self.load("attn_o_t\(T).bin")))
            let (cache, _, part) = try Self.makeEnv(mode: .q8, tokensCeiling: T + 256)
            let h = try cache.allocate(promptTokens: T)
            try cache.append(h, positions: 0..<C, layer: Self.layers[0], kind: .k, src: kcache)
            try cache.append(h, positions: 0..<C, layer: Self.layers[0], kind: .v, src: vcache)
            let splits = QWAttnDecode.splits(forKeys: C + 1)
            let gotBits = try Self.runDecode(cache: cache, handle: h, layer: Self.layers[0],
                                                  qgate: qgate, knew: knew, vnew: vnew,
                                                  partials: part, C: C, splits: splits, mode: .q8)
            let got = Self.readF(gotBits)
            XCTAssertEqual(got.filter { !$0.isFinite }.count, 0)
            var worst = 0.0
            for i in 0..<got.count { worst = max(worst, abs(Double(got[i]) - Double(ref[i]))) }
            // Band pinned from the first measured run: q8 worst across the
            // golden ladder 6e-4 (T=4321) … 2.4e-2 (T=2, where the only key
            // is quantised and softmax amplifies the error). 5e-2 = the
            // recorded envelope; drift beyond it is a regression, not a
            // tolerance to renegotiate.
            XCTAssertLessThan(worst, 5e-2, "q8 T=\(T): worst \(worst) vs f16 golden")
        }
    }

    func testDecodedOutputParityQ4() throws {
        for T in [2, 257, 1000, 4321] {
            let C = T - 1
            let qgate = try Self.load("attn_qgate_dec_t\(T).bin")
            let knew = try Self.load("attn_knew_t\(T).bin")
            let vnew = try Self.load("attn_vnew_t\(T).bin")
            let kcache = try Self.u16s(Self.load("attn_kcache_t\(T).bin"))
            let vcache = try Self.u16s(Self.load("attn_vcache_t\(T).bin"))
            let ref = Self.readF(try Self.u16s(Self.load("attn_o_t\(T).bin")))
            let (cache, _, part) = try Self.makeEnv(mode: .q4, tokensCeiling: T + 256)
            let h = try cache.allocate(promptTokens: T)
            try cache.append(h, positions: 0..<C, layer: Self.layers[0], kind: .k, src: kcache)
            try cache.append(h, positions: 0..<C, layer: Self.layers[0], kind: .v, src: vcache)
            let splits = QWAttnDecode.splits(forKeys: C + 1)
            let gotBits = try Self.runDecode(cache: cache, handle: h, layer: Self.layers[0],
                                                  qgate: qgate, knew: knew, vnew: vnew,
                                                  partials: part, C: C, splits: splits, mode: .q4)
            let got = Self.readF(gotBits)
            XCTAssertEqual(got.filter { !$0.isFinite }.count, 0)
            var worst = 0.0
            for i in 0..<got.count { worst = max(worst, abs(Double(got[i]) - Double(ref[i]))) }
            // q4 asymmetric group-128 (15-step grid): measured 8.9e-3
            // (T=4321) … 0.219 (T=2 single-quantised-key worst case).
            XCTAssertLessThan(worst, 0.35, "q4 T=\(T): worst \(worst) vs f16 golden")
        }
    }

    func testSharedPrefixBatchedAcceptsQuantizedKVModes() throws {
        let g = QWModelGeometry.qwen38_27b
        let B = 3, C = 16
        for mode in [QWKVMode.q8, QWKVMode.q4] {
            let pageBytes = QWKVCache.pageBytes(mode: mode)
            let poolBytes = pageBytes * 4
            let tables = QWKVCache.tableSlotBytes * 2 * Self.layers.count * QWKVCache.maxSeqs
            let pBytes = QWAttnDecode.decodePartialsBytesBatched(T: B, geometry: g)
            let arena = try QWArena(hal: Self.hal, weights: 0, activations: 0,
                                    kv: poolBytes * Self.layers.count + tables, gdnStates: pBytes)
            let cache = try QWKVCache(hal: Self.hal, arena: arena, layers: Self.layers, mode: mode, geometry: g)
            let h = try cache.allocate(promptTokens: C + B)
            let views = try cache.layerViews(h, layer: Self.layers[0])
            let gdn = arena.gdnRegion()
            let off = try gdn.allocate(bytes: pBytes)
            let part = QWMRegion(buffer: gdn.region.buffer, base: gdn.region.base + off, length: pBytes)

            let qgateBuf = Self.hal.device.makeBuffer(length: B * g.qProjOut * 2, options: .storageModeShared)!
            let knewBuf = Self.hal.device.makeBuffer(length: B * g.kvHeads * g.headDim * 2, options: .storageModeShared)!
            let vnewBuf = Self.hal.device.makeBuffer(length: B * g.kvHeads * g.headDim * 2, options: .storageModeShared)!
            let outBuf = Self.hal.device.makeBuffer(length: B * g.qHeads * g.headDim * 2, options: .storageModeShared)!

            let dec = try QWAttnDecode(hal: Self.hal, sharedPrefixRows: B, kvMode: mode, geometry: g)
            guard let (cb, enc) = Self.hal.begin() else { return XCTFail("begin nil") }
            try dec.encodeBatched(
                enc,
                qgate: Self.tv(qgateBuf, B * g.qProjOut),
                kv: views,
                knew: Self.tv(knewBuf, B * g.kvHeads * g.headDim),
                vnew: Self.tv(vnewBuf, B * g.kvHeads * g.headDim),
                partials: part,
                out: Self.tv(outBuf, B * g.qHeads * g.headDim),
                cBase: C, T: B, splits: 8, sharedPrefix: true)
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
        }
    }
}
