// QWAttnSWRingParityTests.swift — GPU bit-exactness of host-recycled SWA KV ring vs un-recycled masked reference.
import XCTest
import Metal
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc
import QWKV
import QWQuant

final class QWAttnSWRingParityTests: XCTestCase {
    private static func makeGeometry(layerTypes: [String], slidingWindow: Int?, maxPos: Int = 8192) throws -> QWModelGeometry {
        var cfg: [String: Any] = [
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_hidden_layers": layerTypes.count,
            "layer_types": layerTypes,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "intermediate_size": 2048,
            "max_position_embeddings": maxPos,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "eos_token_id": 2,
            "vocab_size": 256,
        ]
        if let slidingWindow { cfg["sliding_window"] = slidingWindow }
        let data = try JSONSerialization.data(withJSONObject: cfg)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    private static func f16Bits(_ v: Float) -> UInt16 { QWQuant.f32tof16Bits(v) }

    private static func synthKV(tokens: Int, geometry: QWModelGeometry) -> (k: [UInt16], v: [UInt16]) {
        let rowElems = geometry.kvChannels
        var k = [UInt16](repeating: 0, count: tokens * rowElems)
        var v = [UInt16](repeating: 0, count: tokens * rowElems)
        for t in 0..<tokens {
            for e in 0..<rowElems {
                k[t * rowElems + e] = f16Bits(sin(Float(t * 23 + e * 7)) * 0.4)
                v[t * rowElems + e] = f16Bits(cos(Float(t * 17 + e * 11)) * 0.4)
            }
        }
        return (k, v)
    }

    private static func synthQGate(tokens: Int, seedOffset: Int, geometry: QWModelGeometry) -> [UInt16] {
        let count = tokens * geometry.qProjOut
        return (0..<count).map { idx in
            let d = idx % geometry.headDim
            return f16Bits(d == 0 ? 1.2 : sin(Float((idx + seedOffset) * 19)) * 0.35)
        }
    }

    private static func slice(_ arr: [UInt16], rows: Range<Int>, cols: Int) -> [UInt16] {
        Array(arr[(rows.lowerBound * cols)..<(rows.upperBound * cols)])
    }

    func testRecycledSWAKVRingMatchesUnrecycledMaskedReferenceBitExactAcrossKVModes() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true", "Requires Metal HAL")
        let window = 512
        let chunk = 256
        let safetyMargin = 7 + QWKVCache.pageTokens
        let ringGeo = try Self.makeGeometry(layerTypes: ["sliding_attention"], slidingWindow: window, maxPos: 8192)
        let refGeo = try Self.makeGeometry(layerTypes: ["sliding_attention"], slidingWindow: window, maxPos: 8192)
        let hal = try QWHal(geometry: ringGeo)

        let totalPrefill = 4096
        let decodeSteps = 8
        let draftLen = 5
        let totalRows = totalPrefill + decodeSteps + draftLen
        let (allK, allV) = Self.synthKV(tokens: totalRows, geometry: ringGeo)
        let rowElems = ringGeo.kvChannels
        let qOut = ringGeo.qProjOut
        let outCols = ringGeo.qHeads * ringGeo.headDim

        for mode in [QWKVMode.f16, .q8, .q4] {
            let pageBytes = QWKVCache.pageBytes(mode: mode, geometry: ringGeo)
            let tables = QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs
            let capPages = ((window + chunk + safetyMargin + QWKVCache.pageTokens - 1) / QWKVCache.pageTokens) + 1
            let ringArena = try QWArena(hal: hal, weights: 0, activations: 0, kv: capPages * 2 * pageBytes + tables, gdnStates: 0)
            let ringCache = try QWKVCache(hal: hal, arena: ringArena, layers: [0], mode: mode, geometry: ringGeo)

            // Reference cache holds ALL 17 pages un-recycled with slidingWindow: 0 and explicit window mask.
            let refPages = (totalRows + QWKVCache.pageTokens - 1) / QWKVCache.pageTokens
            let refArena = try QWArena(hal: hal, weights: 0, activations: 0, kv: refPages * 2 * pageBytes + tables, gdnStates: 0)
            let refCache = try QWKVCache(hal: hal, arena: refArena, layers: [0], mode: mode, geometry: refGeo,
                                         layerSlidingWindows: [0])

            let ringH = try ringCache.allocate(promptTokens: totalPrefill)
            let refH = try refCache.allocate(promptTokens: totalPrefill)
            defer {
                ringCache.finish(ringH)
                refCache.finish(refH)
            }

            let genPrefill = try QWAttnPagedPrefillGeneral(hal: hal, geometry: ringGeo)
            let naxPrefill = hal.caps.supportsSimdgroupMMA
                ? try QWAttnPagedPrefill(hal: hal, geometry: ringGeo) : nil
            let decOp = try QWAttnDecode(hal: hal, kvMode: mode, geometry: ringGeo, forceGeneral: true)

            func submit(_ encode: (any MTLComputeCommandEncoder) throws -> Void) throws {
                guard let (cb, enc) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
                var ended = false
                defer { if !ended { enc.endEncoding() } }
                try encode(enc)
                enc.endEncoding()
                ended = true
                cb.commit()
                cb.waitUntilCompleted()
                if let err = cb.error { throw err }
            }

            let sp = (chunk + 31) / 32 * 32
            let qBuf = hal.device.makeBuffer(length: sp * qOut * 2, options: .storageModeShared)!
            let kBuf = hal.device.makeBuffer(length: rowElems * 2, options: .storageModeShared)!
            let vBuf = hal.device.makeBuffer(length: rowElems * 2, options: .storageModeShared)!
            let ringOut = hal.device.makeBuffer(length: chunk * outCols * 2, options: .storageModeShared)!
            let refOut = hal.device.makeBuffer(length: chunk * outCols * 2, options: .storageModeShared)!
            let partialsBuf = hal.device.makeBuffer(
                length: QWAttnDecode.decodePartialsBytes(geometry: ringGeo, maxSplits: QWAttnDecode.maxSplits),
                options: .storageModeShared)!
            let partialsReg = QWMRegion(buffer: partialsBuf, base: 0, length: partialsBuf.length)

            func runPrefillPair(pos: Int, count: Int) throws {
                let qBits = Self.synthQGate(tokens: sp, seedOffset: pos * 17, geometry: ringGeo)
                qBits.withUnsafeBytes { raw in _ = memcpy(qBuf.contents(), raw.baseAddress!, raw.count) }
                let qTV = QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length),
                                        offset: 0, shape: QWTensorShape([sp * qOut], .f16))
                let ringViews = try ringCache.layerViews(ringH, layer: 0)
                let refViews = try refCache.layerViews(refH, layer: 0)
                let ringOutTV = QWTensorValue(region: QWMRegion(buffer: ringOut, base: 0, length: count * outCols * 2),
                                              offset: 0, shape: QWTensorShape([count * outCols], .f16))
                let refOutTV = QWTensorValue(region: QWMRegion(buffer: refOut, base: 0, length: count * outCols * 2),
                                             offset: 0, shape: QWTensorShape([count * outCols], .f16))
                try submit { try genPrefill.encode($0, qgate: qTV, kv: ringViews, out: ringOutTV, pos: pos, T: count, slidingWindow: window) }
                try submit { try genPrefill.encode($0, qgate: qTV, kv: refViews, out: refOutTV, pos: pos, T: count, slidingWindow: window) }

                let pRing = ringOut.contents().assumingMemoryBound(to: UInt16.self)
                let pRef = refOut.contents().assumingMemoryBound(to: UInt16.self)
                for idx in 0..<(count * outCols) {
                    XCTAssertEqual(pRing[idx], pRef[idx], "prefill bit mismatch mode=\(mode) pos=\(pos) idx=\(idx)")
                }

                if let nax = naxPrefill {
                    try submit { try nax.encode($0, qgate: qTV, kv: ringViews, out: ringOutTV, pos: pos, T: count, slidingWindow: window) }
                    try submit { try nax.encode($0, qgate: qTV, kv: refViews, out: refOutTV, pos: pos, T: count, slidingWindow: window) }
                    for idx in 0..<(count * outCols) {
                        XCTAssertEqual(pRing[idx], pRef[idx], "NAX bit mismatch mode=\(mode) pos=\(pos) idx=\(idx)")
                    }
                }
            }

            // 1. Chunked 4096-token prefill across the ring pool.
            for cIdx in 0..<(totalPrefill / chunk) {
                let pos = cIdx * chunk
                let kSl = Self.slice(allK, rows: pos..<(pos + chunk), cols: rowElems)
                let vSl = Self.slice(allV, rows: pos..<(pos + chunk), cols: rowElems)
                try ringCache.append(ringH, positions: pos..<(pos + chunk), layer: 0, kind: .k, src: kSl)
                try ringCache.append(ringH, positions: pos..<(pos + chunk), layer: 0, kind: .v, src: vSl)
                try refCache.append(refH, positions: pos..<(pos + chunk), layer: 0, kind: .k, src: kSl)
                try refCache.append(refH, positions: pos..<(pos + chunk), layer: 0, kind: .v, src: vSl)
                try runPrefillPair(pos: pos, count: chunk)
                try ringCache.recycleExpiredWindowPages(ringH, currentPosition: pos + chunk, safetyMargin: safetyMargin)
                XCTAssertLessThanOrEqual(ringCache.allocatedPhysicalPages(ringH, layer: 0), capPages)
            }

            func runDecodeStep(pos: Int) throws {
                let qBits = Self.synthQGate(tokens: 1, seedOffset: pos * 31, geometry: ringGeo)
                qBits.withUnsafeBytes { raw in _ = memcpy(qBuf.contents(), raw.baseAddress!, raw.count) }
                let kSl = Self.slice(allK, rows: pos..<(pos + 1), cols: rowElems)
                let vSl = Self.slice(allV, rows: pos..<(pos + 1), cols: rowElems)
                kSl.withUnsafeBytes { raw in _ = memcpy(kBuf.contents(), raw.baseAddress!, raw.count) }
                vSl.withUnsafeBytes { raw in _ = memcpy(vBuf.contents(), raw.baseAddress!, raw.count) }

                let qTV = QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qOut * 2),
                                        offset: 0, shape: QWTensorShape([qOut], .f16))
                let kTV = QWTensorValue(region: QWMRegion(buffer: kBuf, base: 0, length: rowElems * 2),
                                        offset: 0, shape: QWTensorShape([rowElems], .f16))
                let vTV = QWTensorValue(region: QWMRegion(buffer: vBuf, base: 0, length: rowElems * 2),
                                        offset: 0, shape: QWTensorShape([rowElems], .f16))
                let ringOutTV = QWTensorValue(region: QWMRegion(buffer: ringOut, base: 0, length: outCols * 2),
                                              offset: 0, shape: QWTensorShape([outCols], .f16))
                let refOutTV = QWTensorValue(region: QWMRegion(buffer: refOut, base: 0, length: outCols * 2),
                                             offset: 0, shape: QWTensorShape([outCols], .f16))

                if ringH.tokenCapacity <= pos { try ringCache.grow(ringH, to: pos + QWKVCache.pageTokens) }
                try ringCache.prepareWrites(ringH, positions: pos..<(pos + 1))
                if refH.tokenCapacity <= pos { try refCache.grow(refH, to: pos + QWKVCache.pageTokens) }
                try refCache.prepareWrites(refH, positions: pos..<(pos + 1))

                let splits = QWAttnDecode.splits(forKeys: min(pos + 1, window))
                let ringViews = try ringCache.layerViews(ringH, layer: 0)
                let refViews = try refCache.layerViews(refH, layer: 0)

                try submit {
                    try decOp.encode($0, qgate: qTV, kv: ringViews, knew: kTV, vnew: vTV,
                                     partials: partialsReg, out: ringOutTV, C: pos, splits: splits, slidingWindow: window)
                }
                ringCache.markWritten(ringH, upTo: pos + 1)
                try ringCache.recycleExpiredWindowPages(ringH, currentPosition: pos + 1, safetyMargin: safetyMargin)

                try submit {
                    try decOp.encode($0, qgate: qTV, kv: refViews, knew: kTV, vnew: vTV,
                                     partials: partialsReg, out: refOutTV, C: pos, splits: splits, slidingWindow: window)
                }
                refCache.markWritten(refH, upTo: pos + 1)

                let pRing = ringOut.contents().assumingMemoryBound(to: UInt16.self)
                let pRef = refOut.contents().assumingMemoryBound(to: UInt16.self)
                for idx in 0..<outCols {
                    XCTAssertEqual(pRing[idx], pRef[idx], "decode bit mismatch mode=\(mode) pos=\(pos) idx=\(idx)")
                }
                XCTAssertLessThanOrEqual(ringCache.allocatedPhysicalPages(ringH, layer: 0), capPages)
            }

            // 2. Decode across page boundary + speculative draft write + rollback + continued decode.
            for step in 0..<decodeSteps {
                try runDecodeStep(pos: totalPrefill + step)
            }
            let draftStart = totalPrefill + decodeSteps
            if ringH.tokenCapacity < draftStart + draftLen { try ringCache.grow(ringH, to: draftStart + draftLen) }
            try ringCache.append(ringH, positions: draftStart..<(draftStart + draftLen), layer: 0, kind: .k,
                                 src: Self.slice(allK, rows: draftStart..<(draftStart + draftLen), cols: rowElems))
            try ringCache.append(ringH, positions: draftStart..<(draftStart + draftLen), layer: 0, kind: .v,
                                 src: Self.slice(allV, rows: draftStart..<(draftStart + draftLen), cols: rowElems))
            try ringCache.rollback(ringH, by: draftLen)
            try ringCache.recycleExpiredWindowPages(ringH, currentPosition: draftStart, safetyMargin: safetyMargin)
            for step in 0..<4 {
                try runDecodeStep(pos: draftStart + step)
            }
        }
    }

    func testPerLayerSlidingWindowDispatchDistinguishesSWAAndFullLayers() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true", "Requires Metal HAL")
        let geo = try Self.makeGeometry(layerTypes: ["sliding_attention", "full_attention"], slidingWindow: 64, maxPos: 1024)
        let hal = try QWHal(geometry: geo)
        let tokens = 128
        let (kAll, vAll) = Self.synthKV(tokens: tokens, geometry: geo)
        let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: geo)
        let tables = QWKVCache.tableSlotBytes * 2 * 2 * QWKVCache.maxSeqs
        let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: 8 * pageBytes + tables, gdnStates: 0)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: [0, 1], mode: .f16, geometry: geo)
        let h = try cache.allocate(promptTokens: tokens)
        defer { cache.finish(h) }
        for layer in [0, 1] {
            try cache.append(h, positions: 0..<tokens, layer: layer, kind: .k, src: kAll)
            try cache.append(h, positions: 0..<tokens, layer: layer, kind: .v, src: vAll)
        }

        let decOp = try QWAttnDecode(hal: hal, geometry: geo)
        let prefillGen = try QWAttnPagedPrefillGeneral(hal: hal, geometry: geo)
        let qBits = Self.synthQGate(tokens: 1, seedOffset: 77, geometry: geo)
        let qBuf = hal.device.makeBuffer(bytes: qBits, length: qBits.count * 2, options: .storageModeShared)!
        let kBuf = hal.device.makeBuffer(bytes: Self.slice(kAll, rows: (tokens - 1)..<tokens, cols: geo.kvChannels),
                                         length: geo.kvChannels * 2, options: .storageModeShared)!
        let vBuf = hal.device.makeBuffer(bytes: Self.slice(vAll, rows: (tokens - 1)..<tokens, cols: geo.kvChannels),
                                         length: geo.kvChannels * 2, options: .storageModeShared)!
        let outElems = geo.qHeads * geo.headDim
        let outSWA = hal.device.makeBuffer(length: outElems * 2, options: .storageModeShared)!
        let outFull = hal.device.makeBuffer(length: outElems * 2, options: .storageModeShared)!
        let partials = hal.device.makeBuffer(
            length: QWAttnDecode.decodePartialsBytes(geometry: geo, maxSplits: QWAttnDecode.maxSplits),
            options: .storageModeShared)!
        let qTV = QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length),
                                offset: 0, shape: QWTensorShape([geo.qProjOut], .f16))
        let kTV = QWTensorValue(region: QWMRegion(buffer: kBuf, base: 0, length: kBuf.length),
                                offset: 0, shape: QWTensorShape([geo.kvChannels], .f16))
        let vTV = QWTensorValue(region: QWMRegion(buffer: vBuf, base: 0, length: vBuf.length),
                                offset: 0, shape: QWTensorShape([geo.kvChannels], .f16))
        let pReg = QWMRegion(buffer: partials, base: 0, length: partials.length)

        let v0 = try cache.layerViews(h, layer: 0)
        let v1 = try cache.layerViews(h, layer: 1)
        guard let (cb1, enc1) = hal.begin() else { throw NSError(domain: "hal", code: 10) }
        try decOp.encode(enc1, qgate: qTV, kv: v0, knew: kTV, vnew: vTV, partials: pReg,
                         out: QWTensorValue(region: QWMRegion(buffer: outSWA, base: 0, length: outElems * 2),
                                            offset: 0, shape: QWTensorShape([outElems], .f16)),
                         C: tokens - 1, splits: 8, slidingWindow: 64)
        enc1.endEncoding(); cb1.commit(); cb1.waitUntilCompleted()

        guard let (cb2, enc2) = hal.begin() else { throw NSError(domain: "hal", code: 11) }
        try decOp.encode(enc2, qgate: qTV, kv: v1, knew: kTV, vnew: vTV, partials: pReg,
                         out: QWTensorValue(region: QWMRegion(buffer: outFull, base: 0, length: outElems * 2),
                                            offset: 0, shape: QWTensorShape([outElems], .f16)),
                         C: tokens - 1, splits: 8, slidingWindow: 0)
        enc2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()

        let swPtr = outSWA.contents().assumingMemoryBound(to: UInt16.self)
        let flPtr = outFull.contents().assumingMemoryBound(to: UInt16.self)
        let differs = (0..<outElems).contains { swPtr[$0] != flPtr[$0] }
        XCTAssertTrue(differs, "SWA (w=64) and Full (w=0) layers must produce distinct outputs beyond window")

        // Verify PagedPrefillGeneral per-layer slidingWindow override also diverges cleanly.
        guard let (cb3, enc3) = hal.begin() else { throw NSError(domain: "hal", code: 12) }
        try prefillGen.encode(enc3, qgate: qTV, kv: v0,
                              out: QWTensorValue(region: QWMRegion(buffer: outSWA, base: 0, length: outElems * 2),
                                                 offset: 0, shape: QWTensorShape([outElems], .f16)),
                              pos: tokens - 1, T: 1, slidingWindow: 64)
        enc3.endEncoding(); cb3.commit(); cb3.waitUntilCompleted()

        guard let (cb4, enc4) = hal.begin() else { throw NSError(domain: "hal", code: 13) }
        try prefillGen.encode(enc4, qgate: qTV, kv: v1,
                              out: QWTensorValue(region: QWMRegion(buffer: outFull, base: 0, length: outElems * 2),
                                                 offset: 0, shape: QWTensorShape([outElems], .f16)),
                              pos: tokens - 1, T: 1, slidingWindow: 0)
        enc4.endEncoding(); cb4.commit(); cb4.waitUntilCompleted()
        let prefillDiffers = (0..<outElems).contains { swPtr[$0] != flPtr[$0] }
        XCTAssertTrue(prefillDiffers, "Prefill SWA vs Full layer override must differ")
    }
}
