import XCTest
import Metal
import Foundation
@testable import QWKV
import QWCore
import QWHAL
import QWAlloc

/// Tests for Candidate 2: 1M context slot expansion (1024 -> 4096 slots)
/// and compressed KV cache (Q8/Q4 layout).
final class QWKV1MAndCompressedLayoutTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()
    static let testLayers = [3, 7]

    func testSlotExpansionConstantsFor1MContext() {
        // Table slot bytes must be 16 KiB (16384 bytes) to host 4096 Int32 page IDs.
        XCTAssertEqual(QWKVCache.tableSlotBytes, 16384, "16 KiB slot required for 4096 page IDs")
        // 4096 page slots * 256 tokens/page = 1,048,576 tokens (1M context ceiling).
        XCTAssertEqual(QWKVCache.maxTokensPerSeq, 1_048_576, "Max tokens per seq must equal exactly 1,048,576 (1M)")
    }

    func testAllocateCeilingGuardAt1MContext() throws {
        // Sizing a minimal cache to verify ceiling guard arithmetic
        let pageBytes = QWKVCache.pageBytes(mode: .f16)
        let tables = QWKVCache.tableSlotBytes * 2 * Self.testLayers.count * QWKVCache.maxSeqs
        let arena = try QWArena(hal: Self.hal, weights: 0, activations: 0,
                                kv: pageBytes * 4 * Self.testLayers.count + tables, gdnStates: 0)
        let cache = try QWKVCache(hal: Self.hal, arena: arena, layers: Self.testLayers, mode: .f16, geometry: .qwen38_27b)

        // Beyond 1,048,576 tokens must throw ceiling error
        XCTAssertThrowsError(try cache.allocate(promptTokens: QWKVCache.maxTokensPerSeq + 256)) { err in
            XCTAssertTrue("\(err)".contains("ceiling"), "Must reject prompt beyond 1M ceiling: \(err)")
        }
    }

    func testFinishZeroesExpandedDeviceSlot() throws {
        let pageBytes = QWKVCache.pageBytes(mode: .f16)
        let needPages = 4
        let poolBytes = pageBytes * needPages * 2
        let tables = QWKVCache.tableSlotBytes * 2 * Self.testLayers.count * QWKVCache.maxSeqs
        let arena = try QWArena(hal: Self.hal, weights: 0, activations: 0,
                                kv: poolBytes * Self.testLayers.count + tables, gdnStates: 0)
        let cache = try QWKVCache(hal: Self.hal, arena: arena, layers: Self.testLayers, mode: .f16, geometry: .qwen38_27b)

        let h = try cache.allocate(promptTokens: 512)
        let l = Self.testLayers[0]
        let views = try cache.layerViews(h, layer: l)
        let slotPtr = views.k.table.region.buffer.contents()
            .advanced(by: views.k.table.region.base)
            .assumingMemoryBound(to: Int32.self)

        cache.finish(h)
        // All 4096 Int32 entries in the 16 KiB device slot must be zeroed
        let totalSlots = QWKVCache.tableSlotBytes / 4
        XCTAssertEqual(totalSlots, 4096)
        var nonZeroCount = 0
        for i in 0..<totalSlots {
            if slotPtr[i] != 0 { nonZeroCount += 1 }
        }
        XCTAssertEqual(nonZeroCount, 0, "All 4096 table slot entries must be zeroed on finish")
    }

    func testKVCompressedModesAndByteMath() {
        // Test QWKVMode enum cases
        XCTAssertEqual(QWKVMode.f16.bytesPerElement, 2.0)
        XCTAssertEqual(QWKVMode.q8.bytesPerElement, 1.0)
        XCTAssertEqual(QWKVMode.q4.bytesPerElement, 0.5)

        // 1 token = 4 kv heads * 256 head dim = 1024 elements
        // FP16: 2048 bytes (no header). Compressed pages carry their C04
        // scale header INSIDE the page (the kernel addresses it at a fixed
        // payload offset), so pageBytes = payload + header — this is the
        // stride the q8/q4 decode kernels multiply page ids by (pinned by
        // QWAttnDecodeQuantKVTests.testLayoutIdentities vs the kernel
        // #defines). Payload-only math for budgeting stays in kvBytes below.
        XCTAssertEqual(QWKVCache.pageBytes(mode: .f16), 256 * 1024 * 2)
        XCTAssertEqual(QWKVCache.pageBytes(mode: .q8), 256 * 1024 * 1 + 256 * 4 * 2)      // +2048 B f16 scales
        XCTAssertEqual(QWKVCache.pageBytes(mode: .q4), 256 * 1024 / 2 + 2048 * 2 * 2)     // +8192 B {scale,min}

        // 1M tokens (1,048,576) across 16 layers:
        // FP16: 2 * 16 * 1,048,576 * 1024 * 2 = 64 GiB (68,719,476,736 bytes)
        // Q8:   2 * 16 * 1,048,576 * 1024 * 1 = 32 GiB (34,359,738,368 bytes)
        let oneMillion = 1_048_576
        XCTAssertEqual(QWKVCache.kvBytes(tokens: oneMillion, layers: 16, mode: .f16), 68_719_476_736)
        XCTAssertEqual(QWKVCache.kvBytes(tokens: oneMillion, layers: 16, mode: .q8), 34_359_738_368)
        XCTAssertEqual(QWKVCache.kvBytes(tokens: oneMillion, layers: 16, mode: .q4), 17_179_869_184)
    }

    func testCompressedKVPageLayoutMetadata() {
        // Layout metadata for a page of 256 tokens:
        let layoutF16 = QWKVCompressedLayout(mode: .f16)
        XCTAssertEqual(layoutF16.dataBytesPerPage, 524_288)
        XCTAssertEqual(layoutF16.scaleBytesPerPage, 0)
        XCTAssertEqual(layoutF16.totalBytesPerPage, 524_288)

        let layoutQ8 = QWKVCompressedLayout(mode: .q8)
        // 256 tokens * 4 heads * 256 bytes = 262,144 bytes data
        XCTAssertEqual(layoutQ8.dataBytesPerPage, 262_144)
        // Per head per row scale (fp16 = 2 bytes): 256 tokens * 4 heads * 2 bytes = 2048 bytes
        XCTAssertEqual(layoutQ8.scaleBytesPerPage, 2048)
        XCTAssertEqual(layoutQ8.totalBytesPerPage, 262_144 + 2048)

        let layoutQ4 = QWKVCompressedLayout(mode: .q4)
        // 256 tokens * 4 heads * 128 bytes = 131,072 bytes data (packed nibbles)
        XCTAssertEqual(layoutQ4.dataBytesPerPage, 131_072)
        // Group-128 scale and min (fp16 each = 4 bytes per group):
        // Each head has 256 elements = 2 groups. 4 heads = 8 groups per token.
        // 256 tokens * 8 groups * 4 bytes (scale + min) = 8192 bytes scales
        XCTAssertEqual(layoutQ4.scaleBytesPerPage, 8192)
        XCTAssertEqual(layoutQ4.totalBytesPerPage, 131_072 + 8192)
    }

    func testQ8QuantizeDequantizeRoundTrip() throws {
        // Deterministic test vector of 1024 Float values (1 token of 4 heads x 256 dims)
        var input = [Float](repeating: 0, count: 1024)
        for i in 0..<1024 {
            input[i] = sin(Float(i) * 0.1) * 2.5
        }

        var qData = [Int8](repeating: 0, count: 1024)
        var scales = [Float16](repeating: 0, count: 4)

        try QWKVCompressedLayout.quantizeQ8(input: input, qOut: &qData, scalesOut: &scales)

        var dequant = [Float](repeating: 0, count: 1024)
        try QWKVCompressedLayout.dequantizeQ8(qIn: qData, scalesIn: scales, out: &dequant)

        // Check max error <= scale / 127
        for h in 0..<4 {
            let sc = Float(scales[h])
            let maxTol = max(sc / 127.0 + 1e-4, 0.05)
            for d in 0..<256 {
                let idx = h * 256 + d
                XCTAssertLessThanOrEqual(abs(input[idx] - dequant[idx]), maxTol)
            }
        }
    }

    func testQ4QuantizeDequantizeRoundTrip() throws {
        // 1 token = 1024 elements (8 groups of 128)
        var input = [Float](repeating: 0, count: 1024)
        for i in 0..<1024 {
            input[i] = cos(Float(i) * 0.05) * 4.0 + 1.0
        }

        var qData = [UInt8](repeating: 0, count: 512) // packed nibbles
        var scales = [Float16](repeating: 0, count: 8)
        var mins = [Float16](repeating: 0, count: 8)

        try QWKVCompressedLayout.quantizeQ4(input: input, qOut: &qData, scalesOut: &scales, minsOut: &mins)

        var dequant = [Float](repeating: 0, count: 1024)
        try QWKVCompressedLayout.dequantizeQ4(qIn: qData, scalesIn: scales, minsIn: mins, out: &dequant)

        // In uniform quantization with bin width `sc`, max rounding error is 0.5 * sc
        for g in 0..<8 {
            let sc = Float(scales[g])
            let maxTol = sc * 0.5 + 0.05
            for d in 0..<128 {
                let idx = g * 128 + d
                XCTAssertLessThanOrEqual(abs(input[idx] - dequant[idx]), maxTol)
            }
        }
    }

    func testGeneralizedQ8AndQ4QuantizeDequantizeAcrossNon27BGeometries() throws {
        for (kvHeads, headDim) in [(8, 128), (4, 64), (2, 192)] {
            let total = kvHeads * headDim
            let groupsPerHead = max(1, (headDim + 127) / 128)
            let totalGroups = kvHeads * groupsPerHead
            var input = [Float](repeating: 0, count: total)
            for i in 0..<total {
                input[i] = sin(Float(i) * 0.07) * 3.0 - 0.5
            }

            var q8 = [Int8](repeating: 0, count: total)
            var s8 = [Float16](repeating: 0, count: kvHeads)
            try QWKVCompressedLayout.quantizeQ8(
                input: input, qOut: &q8, scalesOut: &s8, kvHeads: kvHeads, headDim: headDim
            )
            var deq8 = [Float](repeating: 0, count: total)
            try QWKVCompressedLayout.dequantizeQ8(
                qIn: q8, scalesIn: s8, out: &deq8, kvHeads: kvHeads, headDim: headDim
            )
            for h in 0..<kvHeads {
                let sc = Float(s8[h])
                let maxTol = max(sc / 127.0 + 1e-4, 0.05)
                for d in 0..<headDim {
                    let idx = h * headDim + d
                    XCTAssertLessThanOrEqual(abs(input[idx] - deq8[idx]), maxTol)
                }
            }

            var q4 = [UInt8](repeating: 0, count: total / 2)
            var s4 = [Float16](repeating: 0, count: totalGroups)
            var m4 = [Float16](repeating: 0, count: totalGroups)
            try QWKVCompressedLayout.quantizeQ4(
                input: input, qOut: &q4, scalesOut: &s4, minsOut: &m4, kvHeads: kvHeads, headDim: headDim
            )
            var deq4 = [Float](repeating: 0, count: total)
            try QWKVCompressedLayout.dequantizeQ4(
                qIn: q4, scalesIn: s4, minsIn: m4, out: &deq4, kvHeads: kvHeads, headDim: headDim
            )
            for h in 0..<kvHeads {
                for g in 0..<groupsPerHead {
                    let gIdx = h * groupsPerHead + g
                    let sc = Float(s4[gIdx])
                    let maxTol = sc * 0.5 + 0.05
                    let groupStart = g * 128
                    let groupLen = min(128, headDim - groupStart)
                    for d in 0..<groupLen {
                        let idx = h * headDim + groupStart + d
                        XCTAssertLessThanOrEqual(abs(input[idx] - deq4[idx]), maxTol)
                    }
                }
            }
        }
    }

    func testSplitKVSubAllocatorsPlaceKAndVInDisjointBuffers() throws {
        let pageBytes = QWKVCache.pageBytes(mode: .f16)
        let needPages = 4
        let kTotal = pageBytes * needPages * Self.testLayers.count
        let vTotal = pageBytes * needPages * Self.testLayers.count
        let tables = QWKVCache.tableSlotBytes * 2 * Self.testLayers.count * QWKVCache.maxSeqs
        let kBuf = try XCTUnwrap(Self.hal.device.makeBuffer(length: kTotal + tables / 2, options: .storageModeShared))
        let vBuf = try XCTUnwrap(Self.hal.device.makeBuffer(length: vTotal + tables / 2, options: .storageModeShared))
        let kSub = QWSubAllocator(region: QWMRegion(buffer: kBuf, base: 0, length: kBuf.length))
        let vSub = QWSubAllocator(region: QWMRegion(buffer: vBuf, base: 0, length: vBuf.length))
        let cache = try QWKVCache(hal: Self.hal, subAllocator: kSub, vSubAllocator: vSub,
                                  layers: Self.testLayers, mode: .f16, geometry: .qwen38_27b)
        let h = try cache.allocate(promptTokens: 512)
        defer { cache.finish(h) }
        for l in Self.testLayers {
            let views = try cache.layerViews(h, layer: l)
            XCTAssertTrue(views.k.pool.buffer === kBuf)
            XCTAssertTrue(views.v.pool.buffer === vBuf)
            XCTAssertFalse(views.k.pool.buffer === views.v.pool.buffer)
        }
    }
}
