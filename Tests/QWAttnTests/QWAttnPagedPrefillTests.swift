import XCTest
import Metal
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc
import QWKV
import QWQuant

final class QWAttnPagedPrefillTests: XCTestCase {
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

    static func synthQGate(n: Int, seed: UInt64) -> [UInt16] {
        var rng = SplitMix64(seed: seed)
        let qHeads = QWModelGeometry.qwen38_27b.qHeads
        let headDim = QWModelGeometry.qwen38_27b.headDim
        var out = [UInt16](repeating: 0, count: n * qHeads * 2 * headDim)
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

    func testPagedPrefillMatchesDecodeReferenceF16() throws {
        let pos = 64
        let T = 32
        let sp = (T + 31) / 32 * 32
        let ceiling = pos + T + 256
        let pageBytes = QWKVCache.pageBytes(mode: .f16)
        let needPages = (ceiling + 255) / 256
        let poolBytes = pageBytes * needPages * 2
        let tables = QWKVCache.tableSlotBytes * 2 * 1 * QWKVCache.maxSeqs
        let arena = try QWArena(hal: Self.hal, weights: 0, activations: 0,
                                kv: poolBytes + tables, gdnStates: QWAttnDecode.decodePartialsBytes(geometry: .qwen38_27b))
        let cache = try QWKVCache(hal: Self.hal, arena: arena, layers: [Self.layer], mode: .f16, geometry: .qwen38_27b)
        let h = try cache.allocate(promptTokens: ceiling)

        // Prepopulate past keys/values
        let kPrefix = Self.synthKV(n: pos, seed: 0x1234)
        let vPrefix = Self.synthV(n: pos, seed: 0x5678)
        try cache.append(h, positions: 0..<pos, layer: Self.layer, kind: .k, src: kPrefix)
        try cache.append(h, positions: 0..<pos, layer: Self.layer, kind: .v, src: vPrefix)

        // Current chunk tokens
        let kChunk = Self.synthKV(n: T, seed: 0x9ABC)
        let vChunk = Self.synthV(n: T, seed: 0xDEF0)
        try cache.append(h, positions: pos..<(pos + T), layer: Self.layer, kind: .k, src: kChunk)
        try cache.append(h, positions: pos..<(pos + T), layer: Self.layer, kind: .v, src: vChunk)

        let views = try cache.layerViews(h, layer: Self.layer)
        let qgateBits = Self.synthQGate(n: sp, seed: 0xFEED)
        let qgateBuf = Self.buf(qgateBits)
        let qgateTV = Self.tv(qgateBuf, sp * QWModelGeometry.qwen38_27b.qProjOut)

        let outPagedBuf = Self.hal.device.makeBuffer(length: T * 6144 * 2, options: .storageModeShared)!
        let outPagedTV = Self.tv(outPagedBuf, T * 6144)

        // 1. Run paged prefill
        let pagedPrefill = try QWAttnPagedPrefill(hal: Self.hal, geometry: .qwen38_27b)
        guard let (cbPaged, encPaged) = Self.hal.begin() else { XCTFail(); return }
        try pagedPrefill.encode(encPaged, qgate: qgateTV, kv: views, out: outPagedTV, pos: pos, T: T)
        encPaged.endEncoding()
        cbPaged.commit()
        cbPaged.waitUntilCompleted()
        let outPaged = Self.readOut(outPagedBuf, T * 6144)

        // 2. Run decode reference loop (one by one)
        let decode = try QWAttnDecode(hal: Self.hal, kvMode: .f16, geometry: .qwen38_27b)
        let partialsBuf = Self.hal.device.makeBuffer(length: QWAttnDecode.decodePartialsBytes(geometry: .qwen38_27b), options: .storageModeShared)!
        let partialsRegion = QWMRegion(buffer: partialsBuf, base: 0, length: QWAttnDecode.decodePartialsBytes(geometry: .qwen38_27b))
        let outRefBuf = Self.hal.device.makeBuffer(length: T * 6144 * 2, options: .storageModeShared)!

        for t in 0..<T {
            let tokPos = pos + t
            let qRow = QWTensorValue(region: qgateTV.region, offset: t * QWModelGeometry.qwen38_27b.qProjOut * 2,
                                     shape: QWTensorShape([QWModelGeometry.qwen38_27b.qProjOut], .f16))
            let kRow = QWTensorValue(region: Self.tv(Self.buf(Array(kChunk[(t * 1024)..<((t + 1) * 1024)])), 1024).region,
                                     offset: 0, shape: QWTensorShape([1024], .f16))
            let vRow = QWTensorValue(region: Self.tv(Self.buf(Array(vChunk[(t * 1024)..<((t + 1) * 1024)])), 1024).region,
                                     offset: 0, shape: QWTensorShape([1024], .f16))
            let outRow = QWTensorValue(region: QWMRegion(buffer: outRefBuf, base: 0, length: T * 6144 * 2),
                                       offset: t * 6144 * 2, shape: QWTensorShape([6144], .f16))
            guard let (cbDec, encDec) = Self.hal.begin() else { XCTFail(); return }
            try decode.encode(encDec, qgate: qRow, kv: views, knew: kRow, vnew: vRow,
                              partials: partialsRegion, out: outRow, C: tokPos,
                              splits: QWAttnDecode.splits(forKeys: tokPos + 1))
            encDec.endEncoding()
            cbDec.commit()
            cbDec.waitUntilCompleted()
        }
        let outRef = Self.readOut(outRefBuf, T * 6144)

        // Compare parity: all finite, max absolute delta <= 5e-3 (NAX fast::exp2 vs scalar exp)
        var maxDiff: Float = 0
        for i in 0..<outPaged.count {
            XCTAssertTrue(outPaged[i].isFinite, "paged output not finite at \(i)")
            XCTAssertTrue(outRef[i].isFinite, "ref output not finite at \(i)")
            let diff = abs(outPaged[i] - outRef[i])
            if diff > maxDiff { maxDiff = diff }
        }
        print("testPagedPrefillMatchesDecodeReferenceF16 maxDiff: \(maxDiff)")
        XCTAssertLessThan(maxDiff, 5e-3, "Paged prefill delta too large vs decode reference")
    }

    func testPagedPrefillRaggedAndMultiTileF16() throws {
        // Test T=65 (crosses 32 and 64 boundaries) at pos=96
        let pos = 96
        let T = 65
        let sp = (T + 31) / 32 * 32
        let ceiling = pos + T + 256
        let pageBytes = QWKVCache.pageBytes(mode: .f16)
        let needPages = (ceiling + 255) / 256
        let poolBytes = pageBytes * needPages * 2
        let tables = QWKVCache.tableSlotBytes * 2 * 1 * QWKVCache.maxSeqs
        let arena = try QWArena(hal: Self.hal, weights: 0, activations: 0,
                                kv: poolBytes + tables, gdnStates: QWAttnDecode.decodePartialsBytes(geometry: .qwen38_27b))
        let cache = try QWKVCache(hal: Self.hal, arena: arena, layers: [Self.layer], mode: .f16, geometry: .qwen38_27b)
        let h = try cache.allocate(promptTokens: ceiling)

        let kPrefix = Self.synthKV(n: pos, seed: 0x1111)
        let vPrefix = Self.synthV(n: pos, seed: 0x2222)
        try cache.append(h, positions: 0..<pos, layer: Self.layer, kind: .k, src: kPrefix)
        try cache.append(h, positions: 0..<pos, layer: Self.layer, kind: .v, src: vPrefix)

        let kChunk = Self.synthKV(n: T, seed: 0x3333)
        let vChunk = Self.synthV(n: T, seed: 0x4444)
        try cache.append(h, positions: pos..<(pos + T), layer: Self.layer, kind: .k, src: kChunk)
        try cache.append(h, positions: pos..<(pos + T), layer: Self.layer, kind: .v, src: vChunk)

        let views = try cache.layerViews(h, layer: Self.layer)
        let qgateBits = Self.synthQGate(n: sp, seed: 0x5555)
        let qgateBuf = Self.buf(qgateBits)
        let qgateTV = Self.tv(qgateBuf, sp * QWModelGeometry.qwen38_27b.qProjOut)

        let outPagedBuf = Self.hal.device.makeBuffer(length: T * 6144 * 2, options: .storageModeShared)!
        let outPagedTV = Self.tv(outPagedBuf, T * 6144)

        let pagedPrefill = try QWAttnPagedPrefill(hal: Self.hal, geometry: .qwen38_27b)
        guard let (cbPaged, encPaged) = Self.hal.begin() else { XCTFail(); return }
        try pagedPrefill.encode(encPaged, qgate: qgateTV, kv: views, out: outPagedTV, pos: pos, T: T)
        encPaged.endEncoding()
        cbPaged.commit()
        cbPaged.waitUntilCompleted()
        let outPaged = Self.readOut(outPagedBuf, T * 6144)

        for i in 0..<outPaged.count {
            XCTAssertTrue(outPaged[i].isFinite, "paged output not finite at \(i)")
        }
    }

    func testPagedPrefillMatchesDecodeReferenceQ8() throws {
        let pos = 64
        let T = 32
        let sp = (T + 31) / 32 * 32
        let ceiling = pos + T + 256
        let pageBytes = QWKVCache.pageBytes(mode: .q8)
        let needPages = (ceiling + 255) / 256
        let poolBytes = pageBytes * needPages * 2
        let tables = QWKVCache.tableSlotBytes * 2 * 1 * QWKVCache.maxSeqs
        let arena = try QWArena(hal: Self.hal, weights: 0, activations: 0,
                                kv: poolBytes + tables, gdnStates: QWAttnDecode.decodePartialsBytes(geometry: .qwen38_27b))
        let cache = try QWKVCache(hal: Self.hal, arena: arena, layers: [Self.layer], mode: .q8, geometry: .qwen38_27b)
        let h = try cache.allocate(promptTokens: ceiling)

        let kPrefix = Self.synthKV(n: pos, seed: 0x1234)
        let vPrefix = Self.synthV(n: pos, seed: 0x5678)
        try cache.append(h, positions: 0..<pos, layer: Self.layer, kind: .k, src: kPrefix)
        try cache.append(h, positions: 0..<pos, layer: Self.layer, kind: .v, src: vPrefix)

        let kChunk = Self.synthKV(n: T, seed: 0x9ABC)
        let vChunk = Self.synthV(n: T, seed: 0xDEF0)
        try cache.append(h, positions: pos..<(pos + T), layer: Self.layer, kind: .k, src: kChunk)
        try cache.append(h, positions: pos..<(pos + T), layer: Self.layer, kind: .v, src: vChunk)

        let views = try cache.layerViews(h, layer: Self.layer)
        let qgateBits = Self.synthQGate(n: sp, seed: 0xFEED)
        let qgateBuf = Self.buf(qgateBits)
        let qgateTV = Self.tv(qgateBuf, sp * QWModelGeometry.qwen38_27b.qProjOut)

        let outPagedBuf = Self.hal.device.makeBuffer(length: T * 6144 * 2, options: .storageModeShared)!
        let outPagedTV = Self.tv(outPagedBuf, T * 6144)

        let pagedPrefill = try QWAttnPagedPrefill(hal: Self.hal, geometry: .qwen38_27b)
        guard let (cbPaged, encPaged) = Self.hal.begin() else { XCTFail(); return }
        try pagedPrefill.encode(encPaged, qgate: qgateTV, kv: views, out: outPagedTV, pos: pos, T: T)
        encPaged.endEncoding()
        cbPaged.commit()
        cbPaged.waitUntilCompleted()
        let outPaged = Self.readOut(outPagedBuf, T * 6144)

        let decode = try QWAttnDecode(hal: Self.hal, kvMode: .q8, geometry: .qwen38_27b)
        let partialsBuf = Self.hal.device.makeBuffer(length: QWAttnDecode.decodePartialsBytes(geometry: .qwen38_27b), options: .storageModeShared)!
        let partialsRegion = QWMRegion(buffer: partialsBuf, base: 0, length: QWAttnDecode.decodePartialsBytes(geometry: .qwen38_27b))
        let outRefBuf = Self.hal.device.makeBuffer(length: T * 6144 * 2, options: .storageModeShared)!

        for t in 0..<T {
            let tokPos = pos + t
            let qRow = QWTensorValue(region: qgateTV.region, offset: t * QWModelGeometry.qwen38_27b.qProjOut * 2,
                                     shape: QWTensorShape([QWModelGeometry.qwen38_27b.qProjOut], .f16))
            let kRow = QWTensorValue(region: Self.tv(Self.buf(Array(kChunk[(t * 1024)..<((t + 1) * 1024)])), 1024).region,
                                     offset: 0, shape: QWTensorShape([1024], .f16))
            let vRow = QWTensorValue(region: Self.tv(Self.buf(Array(vChunk[(t * 1024)..<((t + 1) * 1024)])), 1024).region,
                                     offset: 0, shape: QWTensorShape([1024], .f16))
            let outRow = QWTensorValue(region: QWMRegion(buffer: outRefBuf, base: 0, length: T * 6144 * 2),
                                       offset: t * 6144 * 2, shape: QWTensorShape([6144], .f16))
            guard let (cbDec, encDec) = Self.hal.begin() else { XCTFail(); return }
            try decode.encode(encDec, qgate: qRow, kv: views, knew: kRow, vnew: vRow,
                              partials: partialsRegion, out: outRow, C: tokPos,
                              splits: QWAttnDecode.splits(forKeys: tokPos + 1))
            encDec.endEncoding()
            cbDec.commit()
            cbDec.waitUntilCompleted()
        }
        let outRef = Self.readOut(outRefBuf, T * 6144)

        var maxDiff: Float = 0
        for i in 0..<outPaged.count {
            XCTAssertTrue(outPaged[i].isFinite, "paged q8 output not finite at \(i)")
            XCTAssertTrue(outRef[i].isFinite, "ref q8 output not finite at \(i)")
            let diff = abs(outPaged[i] - outRef[i])
            if diff > maxDiff { maxDiff = diff }
        }
        print("testPagedPrefillMatchesDecodeReferenceQ8 maxDiff: \(maxDiff)")
        XCTAssertLessThan(maxDiff, 5e-3, "Paged prefill delta too large vs decode reference in Q8")
    }

    func testPagedPrefillMatchesDecodeReferenceQ4() throws {
        let pos = 64
        let T = 32
        let sp = (T + 31) / 32 * 32
        let ceiling = pos + T + 256
        let pageBytes = QWKVCache.pageBytes(mode: .q4)
        let needPages = (ceiling + 255) / 256
        let poolBytes = pageBytes * needPages * 2
        let tables = QWKVCache.tableSlotBytes * 2 * 1 * QWKVCache.maxSeqs
        let arena = try QWArena(hal: Self.hal, weights: 0, activations: 0,
                                kv: poolBytes + tables, gdnStates: QWAttnDecode.decodePartialsBytes(geometry: .qwen38_27b))
        let cache = try QWKVCache(hal: Self.hal, arena: arena, layers: [Self.layer], mode: .q4, geometry: .qwen38_27b)
        let h = try cache.allocate(promptTokens: ceiling)

        let kPrefix = Self.synthKV(n: pos, seed: 0x1234)
        let vPrefix = Self.synthV(n: pos, seed: 0x5678)
        try cache.append(h, positions: 0..<pos, layer: Self.layer, kind: .k, src: kPrefix)
        try cache.append(h, positions: 0..<pos, layer: Self.layer, kind: .v, src: vPrefix)

        let kChunk = Self.synthKV(n: T, seed: 0x9ABC)
        let vChunk = Self.synthV(n: T, seed: 0xDEF0)
        try cache.append(h, positions: pos..<(pos + T), layer: Self.layer, kind: .k, src: kChunk)
        try cache.append(h, positions: pos..<(pos + T), layer: Self.layer, kind: .v, src: vChunk)

        let views = try cache.layerViews(h, layer: Self.layer)
        let qgateBits = Self.synthQGate(n: sp, seed: 0xFEED)
        let qgateBuf = Self.buf(qgateBits)
        let qgateTV = Self.tv(qgateBuf, sp * QWModelGeometry.qwen38_27b.qProjOut)

        let outPagedBuf = Self.hal.device.makeBuffer(length: T * 6144 * 2, options: .storageModeShared)!
        let outPagedTV = Self.tv(outPagedBuf, T * 6144)

        let pagedPrefill = try QWAttnPagedPrefill(hal: Self.hal, geometry: .qwen38_27b)
        guard let (cbPaged, encPaged) = Self.hal.begin() else { XCTFail(); return }
        try pagedPrefill.encode(encPaged, qgate: qgateTV, kv: views, out: outPagedTV, pos: pos, T: T)
        encPaged.endEncoding()
        cbPaged.commit()
        cbPaged.waitUntilCompleted()
        let outPaged = Self.readOut(outPagedBuf, T * 6144)

        let decode = try QWAttnDecode(hal: Self.hal, kvMode: .q4, geometry: .qwen38_27b)
        let partialsBuf = Self.hal.device.makeBuffer(length: QWAttnDecode.decodePartialsBytes(geometry: .qwen38_27b), options: .storageModeShared)!
        let partialsRegion = QWMRegion(buffer: partialsBuf, base: 0, length: QWAttnDecode.decodePartialsBytes(geometry: .qwen38_27b))
        let outRefBuf = Self.hal.device.makeBuffer(length: T * 6144 * 2, options: .storageModeShared)!

        for t in 0..<T {
            let tokPos = pos + t
            let qRow = QWTensorValue(region: qgateTV.region, offset: t * QWModelGeometry.qwen38_27b.qProjOut * 2,
                                     shape: QWTensorShape([QWModelGeometry.qwen38_27b.qProjOut], .f16))
            let kRow = QWTensorValue(region: Self.tv(Self.buf(Array(kChunk[(t * 1024)..<((t + 1) * 1024)])), 1024).region,
                                     offset: 0, shape: QWTensorShape([1024], .f16))
            let vRow = QWTensorValue(region: Self.tv(Self.buf(Array(vChunk[(t * 1024)..<((t + 1) * 1024)])), 1024).region,
                                     offset: 0, shape: QWTensorShape([1024], .f16))
            let outRow = QWTensorValue(region: QWMRegion(buffer: outRefBuf, base: 0, length: T * 6144 * 2),
                                       offset: t * 6144 * 2, shape: QWTensorShape([6144], .f16))
            guard let (cbDec, encDec) = Self.hal.begin() else { XCTFail(); return }
            try decode.encode(encDec, qgate: qRow, kv: views, knew: kRow, vnew: vRow,
                              partials: partialsRegion, out: outRow, C: tokPos,
                              splits: QWAttnDecode.splits(forKeys: tokPos + 1))
            encDec.endEncoding()
            cbDec.commit()
            cbDec.waitUntilCompleted()
        }
        let outRef = Self.readOut(outRefBuf, T * 6144)

        var maxDiff: Float = 0
        for i in 0..<outPaged.count {
            XCTAssertTrue(outPaged[i].isFinite, "paged q4 output not finite at \(i)")
            XCTAssertTrue(outRef[i].isFinite, "ref q4 output not finite at \(i)")
            let diff = abs(outPaged[i] - outRef[i])
            if diff > maxDiff { maxDiff = diff }
        }
        print("testPagedPrefillMatchesDecodeReferenceQ4 maxDiff: \(maxDiff)")
        XCTAssertLessThan(maxDiff, 5e-3, "Paged prefill delta too large vs decode reference in Q4")
    }

    func testPagedPrefill35BShapePipelineCreation() throws {
        let configPath = ("~/.mlx-serve/models/ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit/config.json" as NSString).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: configPath) else {
            throw XCTSkip("needs the local Qwen3.6-35B config (not part of the repository)")
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        let desc = try QWModelDescriptor.decode(configJSON: data)
        let geo = QWModelGeometry(descriptor: desc)
        XCTAssertTrue(geo.supportsNAXPagedAttnPrefill)
        let hal35B = try QWHal(geometry: geo)
        let prefill = try QWAttnPagedPrefill(hal: hal35B, geometry: geo)
        XCTAssertEqual(prefill.geometry.headDim, 256)
        XCTAssertEqual(prefill.geometry.gqaRatio, 8)
    }

    func testPagedPrefillExecution35BShapeF16() throws {
        let configPath = ("~/.mlx-serve/models/ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit/config.json" as NSString).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: configPath) else {
            throw XCTSkip("needs the local Qwen3.6-35B config (not part of the repository)")
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        let desc = try QWModelDescriptor.decode(configJSON: data)
        let geo = QWModelGeometry(descriptor: desc)
        let hal35B = try QWHal(geometry: geo)
        let prefill = try QWAttnPagedPrefill(hal: hal35B, geometry: geo)

        let pos = 32
        let T = 32
        let sp = (T + 31) / 32 * 32
        let ceiling = pos + T + 256
        let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: geo)
        let needPages = (ceiling + 255) / 256
        let poolBytes = pageBytes * needPages * 2
        let tables = QWKVCache.tableSlotBytes * 2 * 1 * QWKVCache.maxSeqs
        let arena = try QWArena(hal: hal35B, weights: 0, activations: 0,
                                kv: poolBytes + tables, gdnStates: 0)
        let cache = try QWKVCache(hal: hal35B, arena: arena, layers: [0], mode: .f16, geometry: geo)
        let h = try cache.allocate(promptTokens: ceiling)

        let kvDim = geo.kvHeads * geo.headDim
        var rng = SplitMix64(seed: 0x42)
        var kData = [UInt16](repeating: 0, count: (pos + T) * kvDim)
        var vData = [UInt16](repeating: 0, count: (pos + T) * kvDim)
        for i in 0..<kData.count {
            kData[i] = Self.f16Bits(rng.nextNormalish() * 0.1)
            vData[i] = Self.f16Bits(rng.nextNormalish() * 0.1)
        }
        try cache.append(h, positions: 0..<(pos + T), layer: 0, kind: .k, src: kData)
        try cache.append(h, positions: 0..<(pos + T), layer: 0, kind: .v, src: vData)

        let views = try cache.layerViews(h, layer: 0)
        var qData = [UInt16](repeating: 0, count: sp * geo.qProjOut)
        for i in 0..<qData.count { qData[i] = Self.f16Bits(rng.nextNormalish() * 0.1) }
        let qBuf = hal35B.device.makeBuffer(bytes: qData, length: qData.count * 2, options: .storageModeShared)!
        let qTV = Self.tv(qBuf, qData.count)

        let outDim = T * geo.qHeads * geo.headDim
        let outBuf = hal35B.device.makeBuffer(length: outDim * 2, options: .storageModeShared)!
        let outTV = Self.tv(outBuf, outDim)

        guard let (cb, enc) = hal35B.begin() else { XCTFail(); return }
        try prefill.encode(enc, qgate: qTV, kv: views, out: outTV, pos: pos, T: T)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let out = Self.readOut(outBuf, outDim)
        XCTAssertEqual(out.count, outDim)
        for i in 0..<out.count {
            XCTAssertTrue(out[i].isFinite, "paged 35B output not finite at \(i)")
        }
    }

    func testPagedPrefillHeadDim128MatchesGeneralPrefillF16() throws {
        let cfg: [String: Any] = [
            "model_type": "qwen3",
            "hidden_size": 4096,
            "num_hidden_layers": 1,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 12288,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "eos_token_id": 2,
            "vocab_size": 64
        ]
        let desc = try QWModelDescriptor.decode(configJSON: JSONSerialization.data(withJSONObject: cfg))
        let geo = QWModelGeometry(descriptor: desc)
        XCTAssertTrue(geo.supportsNAXPagedAttnPrefill)
        XCTAssertFalse(geo.attnOutputGate)
        let hal128 = try QWHal(geometry: geo)
        let naxPrefill = try QWAttnPagedPrefill(hal: hal128, geometry: geo)
        let genPrefill = try QWAttnPagedPrefillGeneral(hal: hal128, geometry: geo)

        let pos = 16
        let T = 25
        let sp = (T + 31) / 32 * 32
        let ceiling = pos + T + 256
        let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: geo)
        let needPages = (ceiling + 255) / 256
        let poolBytes = pageBytes * needPages * 2
        let tables = QWKVCache.tableSlotBytes * 2 * 1 * QWKVCache.maxSeqs
        let arena = try QWArena(hal: hal128, weights: 0, activations: 0,
                                kv: poolBytes + tables, gdnStates: 0)
        let cache = try QWKVCache(hal: hal128, arena: arena, layers: [0], mode: .f16, geometry: geo)
        let h = try cache.allocate(promptTokens: ceiling)

        let kvDim = geo.kvHeads * geo.headDim
        var rng = SplitMix64(seed: 0x77)
        var kData = [UInt16](repeating: 0, count: (pos + T) * kvDim)
        var vData = [UInt16](repeating: 0, count: (pos + T) * kvDim)
        for i in 0..<kData.count {
            kData[i] = Self.f16Bits(rng.nextNormalish() * 0.1)
            vData[i] = Self.f16Bits(rng.nextNormalish() * 0.1)
        }
        try cache.append(h, positions: 0..<(pos + T), layer: 0, kind: .k, src: kData)
        try cache.append(h, positions: 0..<(pos + T), layer: 0, kind: .v, src: vData)

        let views = try cache.layerViews(h, layer: 0)
        var qData = [UInt16](repeating: 0, count: sp * geo.qProjOut)
        for i in 0..<(T * geo.qProjOut) { qData[i] = Self.f16Bits(rng.nextNormalish() * 0.1) }
        let qBuf = hal128.device.makeBuffer(bytes: qData, length: qData.count * 2, options: .storageModeShared)!
        let qTV = Self.tv(qBuf, qData.count)

        let outDim = T * geo.qHeads * geo.headDim
        let outNaxBuf = hal128.device.makeBuffer(length: outDim * 2, options: .storageModeShared)!
        let outGenBuf = hal128.device.makeBuffer(length: outDim * 2, options: .storageModeShared)!
        let outNaxTV = Self.tv(outNaxBuf, outDim)
        let outGenTV = Self.tv(outGenBuf, outDim)

        guard let (cb1, enc1) = hal128.begin() else { XCTFail(); return }
        try naxPrefill.encode(enc1, qgate: qTV, kv: views, out: outNaxTV, pos: pos, T: T)
        try genPrefill.encode(enc1, qgate: qTV, kv: views, out: outGenTV, pos: pos, T: T)
        enc1.endEncoding()
        cb1.commit()
        cb1.waitUntilCompleted()

        let outNax = Self.readOut(outNaxBuf, outDim)
        let outGen = Self.readOut(outGenBuf, outDim)
        var maxDiff: Float = 0
        for i in 0..<outDim {
            XCTAssertTrue(outNax[i].isFinite, "nax output not finite at \(i)")
            maxDiff = max(maxDiff, abs(outNax[i] - outGen[i]))
        }
        XCTAssertLessThan(maxDiff, 1e-3, "headDim=128 NAX prefill vs general prefill maxDiff=\(maxDiff)")
    }

    func testPagedPrefillOddGQARatioAndMHAMatchesGeneralPrefillF16() throws {
        for (qHeads, kvHeads) in [(28, 4), (8, 8)] {
            let cfg: [String: Any] = [
                "model_type": "qwen2",
                "hidden_size": qHeads * 128,
                "num_hidden_layers": 1,
                "num_attention_heads": qHeads,
                "num_key_value_heads": kvHeads,
                "head_dim": 128,
                "intermediate_size": 4096,
                "max_position_embeddings": 4096,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "eos_token_id": 2,
                "vocab_size": 64
            ]
            let desc = try QWModelDescriptor.decode(configJSON: JSONSerialization.data(withJSONObject: cfg))
            let geo = QWModelGeometry(descriptor: desc)
            XCTAssertTrue(geo.supportsNAXPagedAttnPrefill, "gqaRatio=\(geo.gqaRatio) should support NAX paged prefill")
            let halOdd = try QWHal(geometry: geo)
            let naxPrefill = try QWAttnPagedPrefill(hal: halOdd, geometry: geo)
            let genPrefill = try QWAttnPagedPrefillGeneral(hal: halOdd, geometry: geo)

            let pos = 32
            let T = 32
            let sp = (T + 31) / 32 * 32
            let ceiling = pos + T + 256
            let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: geo)
            let needPages = (ceiling + 255) / 256
            let poolBytes = pageBytes * needPages * 2
            let tables = QWKVCache.tableSlotBytes * 2 * 1 * QWKVCache.maxSeqs
            let arena = try QWArena(hal: halOdd, weights: 0, activations: 0,
                                    kv: poolBytes + tables, gdnStates: 0)
            let cache = try QWKVCache(hal: halOdd, arena: arena, layers: [0], mode: .f16, geometry: geo)
            let h = try cache.allocate(promptTokens: ceiling)

            let kvDim = geo.kvHeads * geo.headDim
            var rng = SplitMix64(seed: 0x99)
            var kData = [UInt16](repeating: 0, count: (pos + T) * kvDim)
            var vData = [UInt16](repeating: 0, count: (pos + T) * kvDim)
            for i in 0..<kData.count {
                kData[i] = Self.f16Bits(rng.nextNormalish() * 0.1)
                vData[i] = Self.f16Bits(rng.nextNormalish() * 0.1)
            }
            try cache.append(h, positions: 0..<(pos + T), layer: 0, kind: .k, src: kData)
            try cache.append(h, positions: 0..<(pos + T), layer: 0, kind: .v, src: vData)

            let views = try cache.layerViews(h, layer: 0)
            var qData = [UInt16](repeating: 0, count: sp * geo.qProjOut)
            for i in 0..<(T * geo.qProjOut) { qData[i] = Self.f16Bits(rng.nextNormalish() * 0.1) }
            let qBuf = halOdd.device.makeBuffer(bytes: qData, length: qData.count * 2, options: .storageModeShared)!
            let qTV = Self.tv(qBuf, qData.count)

            let outDim = T * geo.qHeads * geo.headDim
            let outNaxBuf = halOdd.device.makeBuffer(length: outDim * 2, options: .storageModeShared)!
            let outGenBuf = halOdd.device.makeBuffer(length: outDim * 2, options: .storageModeShared)!
            let outNaxTV = Self.tv(outNaxBuf, outDim)
            let outGenTV = Self.tv(outGenBuf, outDim)

            guard let (cb1, enc1) = halOdd.begin() else { XCTFail(); return }
            try naxPrefill.encode(enc1, qgate: qTV, kv: views, out: outNaxTV, pos: pos, T: T)
            try genPrefill.encode(enc1, qgate: qTV, kv: views, out: outGenTV, pos: pos, T: T)
            enc1.endEncoding()
            cb1.commit()
            cb1.waitUntilCompleted()

            let outNax = Self.readOut(outNaxBuf, outDim)
            let outGen = Self.readOut(outGenBuf, outDim)
            var maxDiff: Float = 0
            for i in 0..<outDim {
                XCTAssertTrue(outNax[i].isFinite, "nax output not finite at \(i) for gqaRatio=\(geo.gqaRatio)")
                maxDiff = max(maxDiff, abs(outNax[i] - outGen[i]))
            }
            XCTAssertLessThan(maxDiff, 1e-3, "gqaRatio=\(geo.gqaRatio) NAX prefill vs general maxDiff=\(maxDiff)")
        }
    }
}

