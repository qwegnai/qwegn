// QWKVGeneralWriterTests.swift — the standard path's KV writer: the pool's
// geometry comes from the model, and the writer fills exactly the head's
// halves. The frozen writer maps 32 lanes × 8 halves onto ONE head, so it
// serves a 256-half head only (C103).
import XCTest
import Metal
import Foundation
@testable import QWKV
import QWCore
import QWHAL
import QWAlloc

final class QWKVGeneralWriterTests: XCTestCase {
    /// The pool's bytes after a writer has filled `T` tokens of one page.
    private struct Pool {
        var k: [UInt16]
        var v: [UInt16]
    }

    /// A one-layer llama: 1024 hidden, 8 q heads, 2 kv heads, a 128-half head.
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

    /// The chunk's K bytes: a token's row element `e` has its own value, so
    /// one head's bytes differ from the next head's.
    static func kValue(t: Int, e: Int) -> UInt16 {
        UInt16(truncatingIfNeeded: t &* 977 &+ e &* 31 &+ 1)
    }

    /// The chunk's V bytes.
    static func vValue(t: Int, e: Int) -> UInt16 {
        UInt16(truncatingIfNeeded: t &* 613 &+ e &* 17 &+ 2)
    }

    /// The bytes a writer leaves in the pool, with everything it did not
    /// write still holding the sentinel.
    private func runWriter(_ pso: any MTLComputePipelineState, hal: QWHal,
                            geometry: QWModelGeometry, T: Int,
                            startPos: Int = 0, pages: Int = 1,
                            bindsQ8Mode: Bool = false) throws -> Pool {
        let rowElems = geometry.kvHeads * geometry.headDim
        let pageElems = QWKVCache.pageTokens * rowElems
        let sentinel: UInt16 = 0xBEEF
        func pool() throws -> MTLBuffer {
            let b = try XCTUnwrap(hal.device.makeBuffer(length: pageElems * 2 * pages,
                                                        options: .storageModeShared))
            let p = b.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<(pageElems * pages) { p[i] = sentinel }
            return b
        }
        let kpool = try pool(), vpool = try pool()
        let table = try XCTUnwrap(hal.device.makeBuffer(length: 4 * pages, options: .storageModeShared))
        let tp = table.contents().bindMemory(to: Int32.self, capacity: pages)
        for i in 0..<pages { tp[i] = Int32(i) }
        // The chunk's K/V rows: one row per token, kvHeads×headDim halves.
        var kIn = [UInt16](repeating: 0, count: T * rowElems)
        var vIn = [UInt16](repeating: 0, count: T * rowElems)
        for t in 0..<T {
            for e in 0..<rowElems {
                kIn[t * rowElems + e] = Self.kValue(t: t, e: e)
                vIn[t * rowElems + e] = Self.vValue(t: t, e: e)
            }
        }
        let kBuf = try XCTUnwrap(hal.device.makeBuffer(length: kIn.count * 2,
                                                       options: .storageModeShared))
        let vBuf = try XCTUnwrap(hal.device.makeBuffer(length: vIn.count * 2,
                                                       options: .storageModeShared))
        _ = kIn.withUnsafeBytes { memcpy(kBuf.contents(), $0.baseAddress!, $0.count) }
        _ = vIn.withUnsafeBytes { memcpy(vBuf.contents(), $0.baseAddress!, $0.count) }

        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        enc.setComputePipelineState(pso)
        enc.setBuffer(kpool, offset: 0, index: 0)
        enc.setBuffer(vpool, offset: 0, index: 1)
        enc.setBuffer(table, offset: 0, index: 2)
        enc.setBuffer(table, offset: 0, index: 3)
        enc.setBuffer(kBuf, offset: 0, index: 4)
        enc.setBuffer(vBuf, offset: 0, index: 5)
        enc.setUInt32s(startPos, T, startIndex: 6)
        if bindsQ8Mode {
            // The frozen writer declares buffer(8) = the q8 mode (0 = f16);
            // an unbound read of it is not something to rely on.
            let mode = try XCTUnwrap(hal.device.makeBuffer(length: 4, options: .storageModeShared))
            mode.contents().bindMemory(to: UInt32.self, capacity: 1)[0] = 0
            enc.setBuffer(mode, offset: 0, index: 8)
        }
        enc.dispatchThreadgroups(MTLSize(width: geometry.kvHeads, height: T, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed, "\(String(describing: cb.error))")
        func read(_ b: MTLBuffer) -> [UInt16] {
            let p = b.contents().assumingMemoryBound(to: UInt16.self)
            return (0..<(pageElems * pages)).map { p[$0] }
        }
        return Pool(k: read(kpool), v: read(vpool))
    }

    /// Each head receives its OWN bytes: a head's halves are its own, not
    /// the next head's.
    func testTheGeneralWriterFillsEachHeadWithItsOwnBytes() throws {
        let geometry = try Self.tinyGeometry()
        let hal = try QWHal(geometry: geometry)
        let pso = try hal.pipeline("qwg_kv_paged_write_general")
        let T = 3
        let rowElems = geometry.kvHeads * geometry.headDim
        let pool = try runWriter(pso, hal: hal, geometry: geometry, T: T)
        for t in 0..<T {
            for kvh in 0..<geometry.kvHeads {
                for d in 0..<geometry.headDim {
                    let e = kvh * geometry.headDim + d
                    let at = t * rowElems + e
                    XCTAssertEqual(pool.k[at], Self.kValue(t: t, e: e),
                                   "K token \(t) head \(kvh) dim \(d)")
                    XCTAssertEqual(pool.v[at], Self.vValue(t: t, e: e),
                                   "V token \(t) head \(kvh) dim \(d)")
                }
            }
        }
    }

    /// The head window: the lanes past a head's data write NOTHING — what a
    /// missing mask breaks, since the unmasked lanes write the next head's
    /// own bytes one row further on.
    func testTheGeneralWriterWritesNothingPastTheHead() throws {
        let geometry = try Self.tinyGeometry()
        let hal = try QWHal(geometry: geometry)
        let pso = try hal.pipeline("qwg_kv_paged_write_general")
        let T = 3
        let rowElems = geometry.kvHeads * geometry.headDim
        let pool = try runWriter(pso, hal: hal, geometry: geometry, T: T)
        // Everything past the written tokens is still the sentinel.
        let written = T * rowElems
        XCTAssertTrue(pool.k[written...].allSatisfy { $0 == 0xBEEF })
        XCTAssertTrue(pool.v[written...].allSatisfy { $0 == 0xBEEF })
    }

    /// At the 27B's geometry the general writer and the frozen one leave the
    /// same bytes — the general one is the frozen one with a head window.
    func testTheGeneralWriterMatchesTheFrozenWriterAt256() throws {
        let geometry = QWModelGeometry.qwen38_27b
        let hal = try QWHal(geometry: geometry)
        let general = try hal.pipeline("qwg_kv_paged_write_general")
        let frozen = try hal.pipeline("qwg_kv_paged_write")
        let T = 2
        let a = try runWriter(general, hal: hal, geometry: geometry, T: T)
        let b = try runWriter(frozen, hal: hal, geometry: geometry, T: T, bindsQ8Mode: true)
        XCTAssertEqual(a.k, b.k)
        XCTAssertEqual(a.v, b.v)
    }

    /// The pool's row and page follow the model's own geometry.
    func testThePoolLayoutFollowsTheGeometry() throws {
        let geometry = try Self.tinyGeometry()
        let rowElems = geometry.kvHeads * geometry.headDim
        XCTAssertEqual(rowElems, 2 * 128)
        XCTAssertEqual(QWKVCache.pageBytes(mode: .f16, geometry: geometry),
                       QWKVCache.pageTokens * rowElems * 2)
        XCTAssertEqual(QWKVCache.pageBytes(mode: .f16),
                       QWKVCache.pageBytes(mode: .f16, geometry: .qwen38_27b))
    }

    /// A chunk that crosses the page boundary lands on the right page and
    /// slot: the writer's page arithmetic is `tok >> 8` / `tok & 255`, and
    /// the pool's row stride is the geometry's.
    func testTheWriterLandsOnTheRightPageAndSlot() throws {
        let geometry = try Self.tinyGeometry()
        let hal = try QWHal(geometry: geometry)
        let pso = try hal.pipeline("qwg_kv_paged_write_general")
        let rowElems = geometry.kvHeads * geometry.headDim
        let pageElems = QWKVCache.pageTokens * rowElems
        let startPos = 253, T = 6
        let pool = try runWriter(pso, hal: hal, geometry: geometry, T: T,
                                 startPos: startPos, pages: 2)
        for t in 0..<T {
            let tok = startPos + t
            let page = tok / QWKVCache.pageTokens, slot = tok % QWKVCache.pageTokens
            for kvh in 0..<geometry.kvHeads {
                for d in 0..<geometry.headDim {
                    let e = kvh * geometry.headDim + d
                    let at = page * pageElems + slot * rowElems + e
                    XCTAssertEqual(pool.k[at], Self.kValue(t: t, e: e),
                                   "K token \(tok) head \(kvh) dim \(d)")
                    XCTAssertEqual(pool.v[at], Self.vValue(t: t, e: e),
                                   "V token \(tok) head \(kvh) dim \(d)")
                }
            }
        }
    }

    /// A cache on a narrow geometry: its page is the model's own, and the
    /// host-side f16 fill lands on the model's own row stride.
    func testANarrowGeometryCacheUsesItsOwnPageAndRow() throws {
        let geometry = try Self.tinyGeometry()
        let hal = try QWHal(geometry: geometry)
        let layers = [0]
        let rowElems = geometry.kvHeads * geometry.headDim
        let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: geometry)
        let tables = QWKVCache.tableSlotBytes * 2 * layers.count * QWKVCache.maxSeqs
        // One pool per kind, so the arena carries two pages per layer.
        let arena = try QWArena(hal: hal, weights: 0, activations: 0,
                                kv: pageBytes * 2 * layers.count + tables, gdnStates: 4096)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: layers, mode: .f16,
                                  geometry: geometry)
        XCTAssertEqual(cache.bytesPerPage, QWKVCache.pageTokens * rowElems * 2)
        XCTAssertEqual(cache.bytesPerPage, 256 * 256 * 2)

        let handle = try cache.allocate(promptTokens: 4)
        let rows = (0..<3).map { t in
            (0..<rowElems).map { e in Self.kValue(t: t, e: e) }
        }
        try cache.append(handle, positions: 0..<3, layer: layers[0], kind: .k,
                         src: rows.flatMap { $0 })
        let views = try cache.layerViews(handle, layer: layers[0])
        let pool = views.k.pool.buffer.contents()
            .advanced(by: views.k.pool.base).assumingMemoryBound(to: UInt16.self)
        for t in 0..<3 {
            for e in 0..<rowElems {
                XCTAssertEqual(pool[t * rowElems + e], Self.kValue(t: t, e: e),
                               "token \(t) element \(e)")
            }
        }
    }

    /// A compressed mode is the 27B's layout: a cache on any other geometry
    /// refuses it, and so does the host-side compressed fill.
    func testACompressedModeIsRefusedOffThe27BShape() throws {
        let geometry = try Self.tinyGeometry()
        let hal = try QWHal(geometry: geometry)
        let layers = [0]
        let tables = QWKVCache.tableSlotBytes * 2 * layers.count * QWKVCache.maxSeqs
        func arena(_ mode: QWKVMode) throws -> QWArena {
            let pageBytes = QWKVCache.pageBytes(mode: mode, geometry: .qwen38_27b)
            return try QWArena(hal: hal, weights: 0, activations: 0,
                               kv: pageBytes * 2 * layers.count + tables, gdnStates: 4096)
        }
        // The cache constructs on any geometry; the compressed layout is
        // enforced where data is written (QWKVCache.append), by name.
        let cache = try QWKVCache(hal: hal, arena: try arena(.q8), layers: layers,
                                  mode: .q8, geometry: geometry)
        let h = try cache.allocate(promptTokens: 1)
        let src = [UInt16](repeating: 0, count: geometry.kvHeads * geometry.headDim)
        XCTAssertThrowsError(try cache.append(h, positions: 0..<1, layer: 0, kind: .k, src: src)) { e in
            XCTAssertTrue("\(e)".contains("27B"), "the refusal names the law: \(e)")
        }
    }

    /// A head the lane mapping cannot serve is refused at the loader: the
    /// mapping covers a head in groups of 8 halves, and the rotation 32.
    func testAHeadWidthThatIsNotAMultipleOf32IsRefused() throws {
        let config: [String: Any] = [
            "model_type": "llama",
            "hidden_size": 1024,
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 100,
            "intermediate_size": 256,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "eos_token_id": 2,
            "vocab_size": 64,
        ]
        let data = try JSONSerialization.data(withJSONObject: config)
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: data)) { e in
            XCTAssertTrue("\(e)".contains("multiple of 32"), "the refusal names the law: \(e)")
        }
    }
}
