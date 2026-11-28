// QWAttnApple9MMATests.swift — Parity tests for Apple9 simdgroup_half8x8 Paged, Sink+Rel, and MLA Split-K kernels.
import Foundation
import Metal
import XCTest
@testable import QWAttn
import QWAlloc
import QWCore
import QWHAL
import QWKV
import QWQuant

final class QWAttnApple9MMATests: XCTestCase {
    private struct RNG {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &+ 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        mutating func nextSigned(_ scale: Float = 1.0) -> Float {
            let u = Float(next() >> 11) * (1.0 / 9007199254740992.0)
            return (u * 2.0 - 1.0) * scale
        }
    }

    private struct SparseAttnParams {
        var T: UInt32
        var pos0: UInt32
        var qHeads: UInt32
        var headDim: UInt32
        var kvLoraRank: UInt32
        var topKPools: UInt32
        var listStride: UInt32
        var kvMode: UInt32
    }

    private struct DSASparseAttnParams {
        var T: UInt32
        var pos0: UInt32
        var qHeads: UInt32
        var headDim: UInt32
        var kvLoraRank: UInt32
        var topK: UInt32
        var windowTokens: UInt32
        var listStride: UInt32
        var kvMode: UInt32
        var ropeDim: UInt32
        var qRopeOffset: UInt32
        var qFullStride: UInt32
        var hasSink: UInt32
        var softmaxScale: Float
    }

    override func setUpWithError() throws {
        try XCTSkipIf(
            MTLCreateSystemDefaultDevice() == nil || ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true",
            "Metal GPU required"
        )
    }

    private func makeF16Buf(_ device: any MTLDevice, _ vals: [Float]) -> any MTLBuffer {
        let bits = vals.map { QWQuant.f32tof16Bits($0) }
        let buf = device.makeBuffer(length: max(16, bits.count * 2), options: .storageModeShared)!
        bits.withUnsafeBytes { raw in
            if let base = raw.baseAddress, !raw.isEmpty {
                buf.contents().copyMemory(from: base, byteCount: raw.count)
            }
        }
        return buf
    }

    private func makeBF16TV(_ device: any MTLDevice, _ vals: [Float], dims: [Int]) -> QWTensorValue {
        let bits = vals.map { QWBF16.fromFloatRNE($0) }
        let bytes = max(16, bits.count * 2)
        let buf = device.makeBuffer(length: bytes, options: .storageModeShared)!
        bits.withUnsafeBytes { raw in
            if let base = raw.baseAddress, !raw.isEmpty {
                buf.contents().copyMemory(from: base, byteCount: raw.count)
            }
        }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: bytes),
                             offset: 0, shape: QWTensorShape(dims, .bf16))
    }

    private func readF16(_ buf: any MTLBuffer, count: Int) -> [Float] {
        let ptr = buf.contents().assumingMemoryBound(to: UInt16.self)
        return (0..<count).map { QWQuant.f16tof32Bits(ptr[$0]) }
    }

    private func dequantPagedKV(
        view: QWKVLayerView, mode: QWKVMode, count: Int, kvHeads: Int, headDim: Int
    ) -> [Float] {
        let pageTable = view.table.region.buffer.contents()
            .advanced(by: view.table.region.base + view.table.offset)
            .assumingMemoryBound(to: Int32.self)
        let poolPtr = view.pool.buffer.contents().advanced(by: view.pool.base)
        let pageBytes = view.pageBytes
        var out = [Float](repeating: 0, count: count * kvHeads * headDim)
        for tok in 0..<count {
            let pg = Int(pageTable[tok >> 8])
            let slot = tok & 255
            let pgBase = poolPtr.advanced(by: pg * pageBytes)
            for kvh in 0..<kvHeads {
                let dstBase = (tok * kvHeads + kvh) * headDim
                switch mode {
                case .f16:
                    let p = pgBase.assumingMemoryBound(to: UInt16.self)
                    for d in 0..<headDim {
                        out[dstBase + d] = QWQuant.f16tof32Bits(p[(slot * kvHeads + kvh) * headDim + d])
                    }
                case .q8:
                    let payOff = (slot * kvHeads + kvh) * headDim
                    let hdrBase = 256 * kvHeads * headDim
                    let p = pgBase.assumingMemoryBound(to: Int8.self)
                    let scPtr = pgBase.advanced(by: hdrBase).assumingMemoryBound(to: UInt16.self)
                    let sc = QWQuant.f16tof32Bits(scPtr[slot * kvHeads + kvh])
                    for d in 0..<headDim {
                        out[dstBase + d] = Float(Float16(Float(p[payOff + d]) * sc))
                    }
                case .q4:
                    let row4 = kvHeads * (headDim / 2)
                    let hdrBase = 256 * row4
                    let grpsPerHead = (headDim + 127) / 128
                    let grow = kvHeads * grpsPerHead * 4
                    let p = pgBase.assumingMemoryBound(to: UInt8.self)
                    let smPtr = pgBase.advanced(by: hdrBase + slot * grow).assumingMemoryBound(to: UInt16.self)
                    let rowPtr = p.advanced(by: slot * row4 + kvh * (headDim / 2))
                    for d in 0..<headDim {
                        let grp = kvh * grpsPerHead + (d >> 7)
                        let sc = QWQuant.f16tof32Bits(smPtr[grp * 2])
                        let mn = QWQuant.f16tof32Bits(smPtr[grp * 2 + 1])
                        let b = rowPtr[d >> 1]
                        let q = (d & 1) == 0 ? (b & 0x0F) : (b >> 4)
                        out[dstBase + d] = Float(Float16(Float(q) * sc + mn))
                    }
                }
            }
        }
        return out
    }

    private func readBF16(_ tv: QWTensorValue, count: Int) -> [Float] {
        let ptr = tv.region.buffer.contents().advanced(by: tv.region.base + tv.offset)
            .assumingMemoryBound(to: UInt16.self)
        return (0..<count).map { Float(bitPattern: UInt32(ptr[$0]) << 16) }
    }

    private func runCmd(_ hal: QWHal, _ encode: (any MTLComputeCommandEncoder) throws -> Void) throws {
        let (cb, enc) = try XCTUnwrap(hal.begin())
        var ended = false
        defer { if !ended { enc.endEncoding() } }
        try encode(enc)
        hal.end(cb, enc)
        ended = true
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "Metal error: \(String(describing: cb.error))")
    }

    private func makeStdGeometry(qHeads: Int = 12, kvHeads: Int = 4, headDim: Int = 128) throws -> QWModelGeometry {
        let cfg: [String: Any] = [
            "model_type": "qwen3", "hidden_size": qHeads * headDim, "num_hidden_layers": 1,
            "num_attention_heads": qHeads, "num_key_value_heads": kvHeads, "head_dim": headDim,
            "intermediate_size": 512, "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "eos_token_id": 2, "vocab_size": 128,
        ]
        let data = try JSONSerialization.data(withJSONObject: cfg)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    private func fwht32Row(_ vals: [Float]) -> [Float] {
        var out = vals
        let scale = Float(1.0 / 32.0.squareRoot())
        for blk in stride(from: 0, to: out.count, by: 32) {
            var h = 1
            while h < 32 {
                for i in stride(from: blk, to: blk + 32, by: h * 2) {
                    for j in i..<(i + h) {
                        let u = out[j], v = out[j + h]
                        out[j] = u + v
                        out[j + h] = u - v
                    }
                }
                h <<= 1
            }
            for i in blk..<(blk + 32) { out[i] *= scale }
        }
        return out
    }

    func testPagedPrefillDenseAndSparseMMAParity() throws {
        var rng = RNG(state: 0x4199A101)
        let geometries: [(String, QWModelGeometry, QWWeightBasis)] = [
            ("rotated_hd256", .qwen38_27b, .rotated),
            ("std_hd128", try makeStdGeometry(qHeads: 8, kvHeads: 2, headDim: 128), .standard),
        ]
        for (tag, geom, basis) in geometries {
            let hal = try QWHal(geometry: geom, weightBasis: basis)
            let apply = try QWAttnPagedPrefill(hal: hal, geometry: geom)
            let general = try QWAttnPagedPrefillGeneral(hal: hal, geometry: geom)
            let pos = 40, T = 24, sp = ((T + 31) / 32) * 32, ceiling = pos + T + 256
            let qOut = geom.qProjOut, qH = geom.qHeads, kvH = geom.kvHeads, hd = geom.headDim
            let gqa = qH / kvH
            let qStride = geom.attnOutputGate ? 2 : 1

            let qVals = (0..<(sp * qOut)).map { $0 < T * qOut ? Float(Float16(rng.nextSigned(0.25))) : 999.0 }
            let qBuf = makeF16Buf(hal.device, qVals)
            let qTV = QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length),
                                    offset: 0, shape: QWTensorShape([sp * qOut], .f16))
            let kSrc = (0..<(pos + T) * kvH * hd).map { _ in Float(Float16(rng.nextSigned(0.25))) }
            let vSrc = (0..<(pos + T) * kvH * hd).map { _ in Float(Float16(rng.nextSigned(0.3))) }

            for mode in [QWKVMode.f16, .q8, .q4] {
                let pageBytes = QWKVCache.pageBytes(mode: mode, geometry: geom)
                let needPages = (ceiling + 255) / 256
                let tables = QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs
                let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: pageBytes * needPages * 2 + tables, gdnStates: 4096)
                let cache = try QWKVCache(hal: hal, arena: arena, layers: [0], mode: mode, geometry: geom)
                let h = try cache.allocate(promptTokens: ceiling)
                try cache.append(h, positions: 0..<(pos + T), layer: 0, kind: .k, src: kSrc.map { QWQuant.f32tof16Bits($0) })
                try cache.append(h, positions: 0..<(pos + T), layer: 0, kind: .v, src: vSrc.map { QWQuant.f32tof16Bits($0) })
                let views = try cache.layerViews(h, layer: 0)

                let outGenBuf = makeF16Buf(hal.device, [Float](repeating: 0, count: T * qH * hd))
                let outGenTV = QWTensorValue(region: QWMRegion(buffer: outGenBuf, base: 0, length: outGenBuf.length),
                                             offset: 0, shape: QWTensorShape([T * qH * hd], .f16))

                let mmaSuffix = mode == .f16 ? "" : (mode == .q8 ? "_q8" : "_q4")
                let mmaPSO = try hal.pipeline("qwg_attn_prefill_paged_mma_pair\(mmaSuffix)")
                let outMMABuf = makeF16Buf(hal.device, [Float](repeating: 0, count: T * qH * hd))
                let outMMATV = QWTensorValue(region: QWMRegion(buffer: outMMABuf, base: 0, length: outMMABuf.length),
                                             offset: 0, shape: QWTensorShape([T * qH * hd], .f16))
                let outApplyBuf = makeF16Buf(hal.device, [Float](repeating: 0, count: T * qH * hd))
                let outApplyTV = QWTensorValue(region: QWMRegion(buffer: outApplyBuf, base: 0, length: outApplyBuf.length),
                                               offset: 0, shape: QWTensorShape([T * qH * hd], .f16))
                let sw = basis == .standard ? 32 : 0
                try runCmd(hal) { enc in
                    enc.setComputePipelineState(mmaPSO)
                    qTV.bind(enc, index: 0)
                    enc.setBuffer(views.k.pool.buffer, offset: views.k.pool.base, index: 1)
                    enc.setBuffer(views.v.pool.buffer, offset: views.v.pool.base, index: 2)
                    views.k.table.bind(enc, index: 3)
                    views.v.table.bind(enc, index: 4)
                    outMMATV.bind(enc, index: 5)
                    var posU = UInt32(pos), tU = UInt32(T), swU = UInt32(sw)
                    enc.setBytes(&posU, length: 4, index: 6)
                    enc.setBytes(&tU, length: 4, index: 7)
                    enc.setBytes(&swU, length: 4, index: 8)
                    enc.dispatchThreadgroups(MTLSize(width: kvH * ((T + 31) / 32), height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 64 * gqa, height: 1, depth: 1))
                }
                try runCmd(hal) { enc in
                    try apply.encode(enc, qgate: qTV, kv: views, out: outApplyTV, pos: pos, T: T, slidingWindow: sw)
                }
                try runCmd(hal) { enc in
                    try general.encode(enc, qgate: qTV, kv: views, out: outGenTV, pos: pos, T: T, slidingWindow: sw)
                }

                let gotMMA = readF16(outMMABuf, count: T * qH * hd)
                let gotApply = readF16(outApplyBuf, count: T * qH * hd)
                let gotGen = readF16(outGenBuf, count: T * qH * hd)
                XCTAssertTrue(gotMMA.allSatisfy(\.isFinite), "\(tag) \(mode) NaN/Inf")
                let maxErr = zip(gotMMA, gotGen).map { abs($0 - $1) }.max() ?? 999
                let maxApplyDiff = zip(gotApply, gotGen).map { abs($0 - $1) }.max() ?? 999
                if mode == .f16 {
                    let attnScale = 1.0 / Double(hd).squareRoot()
                    var maxCpuErr: Float = 0
                    for t in 0..<T {
                        let absR = pos + t
                        let minK = sw > 0 ? max(0, absR + 1 - sw) : 0
                        for head in 0..<qH {
                            let kvh = head / gqa
                            let qOff = t * qOut + head * qStride * hd
                            var scores = [Double](repeating: 0, count: absR - minK + 1)
                            for j in minK...absR {
                                var dot = 0.0
                                let kOff = j * kvH * hd + kvh * hd
                                for d in 0..<hd { dot += Double(qVals[qOff + d]) * Double(kSrc[kOff + d]) }
                                scores[j - minK] = dot * attnScale
                            }
                            let mx = scores.max()!
                            let w = scores.map { exp($0 - mx) }
                            let l = w.reduce(0, +)
                            var acc = [Float](repeating: 0, count: hd)
                            for (idx, j) in (minK...absR).enumerated() {
                                let vOff = j * kvH * hd + kvh * hd
                                for d in 0..<hd { acc[d] += Float(w[idx] / l) * vSrc[vOff + d] }
                            }
                            if geom.attnOutputGate {
                                for d in 0..<hd { acc[d] *= Float(1.0 / (1.0 + exp(-Double(qVals[qOff + hd + d])))) }
                            }
                            if basis == .rotated { acc = fwht32Row(acc) }
                            let oOff = (t * qH + head) * hd
                            for d in 0..<hd { maxCpuErr = max(maxCpuErr, abs(gotMMA[oOff + d] - acc[d])) }
                        }
                    }
                    XCTAssertLessThan(maxCpuErr, 1.5e-2, "\(tag) f16 cpu err=\(maxCpuErr)")
                }
                XCTAssertLessThan(maxErr, 2e-2, "\(tag) \(mode) dense MMA err=\(maxErr)")
                XCTAssertLessThan(maxApplyDiff, 2e-2, "\(tag) \(mode) apply err=\(maxApplyDiff)")

                // Sparse paged MMA (select odd-indexed blocks [1, 3, 5, ...] plus causal tail)
                let listStride = 8
                let totalBlocks = (pos + T) / 4
                let selBlocks = Swift.stride(from: 1, to: totalBlocks, by: 2).prefix(listStride).map { Int32($0) }
                var listArr = [Int32](repeating: 0, count: T * listStride)
                for row in 0..<T {
                    for (bi, blk) in selBlocks.enumerated() { listArr[row * listStride + bi] = blk }
                }
                let cntArr = [Int32](repeating: Int32(selBlocks.count), count: T)
                let listBuf = hal.device.makeBuffer(bytes: listArr, length: max(16, listArr.count * 4), options: .storageModeShared)!
                let cntBuf = hal.device.makeBuffer(bytes: cntArr, length: max(16, cntArr.count * 4), options: .storageModeShared)!
                let listTV = QWTensorValue(region: QWMRegion(buffer: listBuf, base: 0, length: listBuf.length),
                                           offset: 0, shape: QWTensorShape([T * listStride], .i32))
                let cntTV = QWTensorValue(region: QWMRegion(buffer: cntBuf, base: 0, length: cntBuf.length),
                                          offset: 0, shape: QWTensorShape([T], .i32))
                let hdTag = hd <= 128 ? "_hd128" : ""
                let spName = "qwg_attn_prefill_paged_mma_sparse_pair\(hdTag)\(mmaSuffix)"
                let spPSO = try hal.pipeline(spName)
                let outSpBuf = makeF16Buf(hal.device, [Float](repeating: 0, count: T * qH * hd))
                let outSpTV = QWTensorValue(region: QWMRegion(buffer: outSpBuf, base: 0, length: outSpBuf.length),
                                            offset: 0, shape: QWTensorShape([T * qH * hd], .f16))
                try runCmd(hal) { enc in
                    enc.setComputePipelineState(spPSO)
                    qTV.bind(enc, index: 0)
                    enc.setBuffer(views.k.pool.buffer, offset: views.k.pool.base, index: 1)
                    enc.setBuffer(views.v.pool.buffer, offset: views.v.pool.base, index: 2)
                    views.k.table.bind(enc, index: 3)
                    views.v.table.bind(enc, index: 4)
                    outSpTV.bind(enc, index: 5)
                    var posU = UInt32(pos), tU = UInt32(T), strU = UInt32(listStride)
                    enc.setBytes(&posU, length: 4, index: 6)
                    enc.setBytes(&tU, length: 4, index: 7)
                    listTV.bind(enc, index: 8)
                    cntTV.bind(enc, index: 9)
                    enc.setBytes(&strU, length: 4, index: 10)
                    enc.dispatchThreadgroups(MTLSize(width: kvH * ((T + 31) / 32), height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 64 * gqa, height: 1, depth: 1))
                }
                let gotSp = readF16(outSpBuf, count: T * qH * hd)
                let deqK = dequantPagedKV(view: views.k, mode: mode, count: pos + T, kvHeads: kvH, headDim: hd)
                let deqV = dequantPagedKV(view: views.v, mode: mode, count: pos + T, kvHeads: kvH, headDim: hd)
                var maxSpErr: Float = 0
                let attnScaleSp = 1.0 / Double(hd).squareRoot()
                for t in 0..<T {
                    let tile0 = (t / 32) * 32
                    let kEndTile = pos + min(tile0 + 32, T)
                    let complete = kEndTile >> 2
                    var keys = [Int]()
                    for b in selBlocks where Int(b) < complete {
                        keys += [Int(b) * 4, Int(b) * 4 + 1, Int(b) * 4 + 2, Int(b) * 4 + 3]
                    }
                    for j in (complete * 4)..<kEndTile { keys.append(j) }
                    let absR = pos + t
                    for head in 0..<qH {
                        let kvh = head / gqa
                        let qOff = t * qOut + head * qStride * hd
                        let vis = keys.filter { $0 <= absR && $0 < pos + T }
                        let scores = vis.map { j -> Double in
                            var dot = 0.0
                            let kOff = j * kvH * hd + kvh * hd
                            for d in 0..<hd { dot += Double(qVals[qOff + d]) * Double(deqK[kOff + d]) }
                            return dot * attnScaleSp
                        }
                        let mx = scores.max()!
                        let w = scores.map { exp($0 - mx) }
                        let l = w.reduce(0, +)
                        var acc = [Float](repeating: 0, count: hd)
                        for (idx, j) in vis.enumerated() {
                            let vOff = j * kvH * hd + kvh * hd
                            for d in 0..<hd { acc[d] += Float(w[idx] / l) * deqV[vOff + d] }
                        }
                        if geom.attnOutputGate {
                            for d in 0..<hd { acc[d] *= Float(1.0 / (1.0 + exp(-Double(qVals[qOff + hd + d])))) }
                        }
                        if basis == .rotated { acc = fwht32Row(acc) }
                        let oOff = (t * qH + head) * hd
                        for d in 0..<hd { maxSpErr = max(maxSpErr, abs(gotSp[oOff + d] - acc[d])) }
                    }
                }
                XCTAssertLessThan(maxSpErr, 1.5e-2, "\(tag) \(mode) sparse MMA err=\(maxSpErr)")
            }
        }
    }

    func testSinkRelDenseAndSparseMMAParityAndBF16Clamp() throws {
        var rng = RNG(state: 0x519CE042)
        let geom = try makeStdGeometry(qHeads: 8, kvHeads: 2, headDim: 128)
        let hal = try QWHal(geometry: geom, weightBasis: .standard)
        let op = try QWAttnSinkRel(hal: hal)
        let nH = 8, nKV = 2, hd = 128, T = 36, posOff = 48, sliding = 40, ringCap = 96, relExt = 24
        let totalKV = posOff + T

        for dRel in [16, 32] {
            for mode in [QWKVMode.f16, .q8, .q4] {
                let cfg = QWSinkRelConfig(
                    nH: nH, nKV: nKV, headDim: hd, dRel: dRel, relExtent: relExt,
                    slidingWindow: sliding, hasSink: true, hasRelBias: true, hasOutGate: true, kvMode: mode
                )
                let qVals = (0..<(T * nH * hd)).map { _ in Float(bitPattern: UInt32(QWBF16.fromFloatRNE(rng.nextSigned(0.3))) << 16) }
                let rVals = (0..<(T * nH * dRel)).map { _ in Float(bitPattern: UInt32(QWBF16.fromFloatRNE(rng.nextSigned(0.3))) << 16) }
                let relProjVals = (0..<(dRel * relExt)).map { _ in Float(bitPattern: UInt32(QWBF16.fromFloatRNE(rng.nextSigned(0.2))) << 16) }
                let sinkVals = (0..<(T * nH)).map { _ in Float(bitPattern: UInt32(QWBF16.fromFloatRNE(rng.nextSigned(0.5))) << 16) }
                let gateVals = (0..<(T * nH * hd)).map { _ in Float(bitPattern: UInt32(QWBF16.fromFloatRNE(rng.nextSigned(0.8))) << 16) }

                let kRaw = (0..<(totalKV * nKV * hd)).map { _ in Float(Float16(rng.nextSigned(0.25))) }
                let vRaw = (0..<(totalKV * nKV * hd)).map { _ in Float(Float16(rng.nextSigned(0.25))) }
                let rb = QWAttnSinkRel.kvRowBytes(mode: mode, nKV: nKV, headDim: hd)
                let payB = max(16, ringCap * rb.payload), metaB = max(16, ringCap * max(4, rb.meta))
                let kBuf = hal.device.makeBuffer(length: payB, options: .storageModeShared)!
                let vBuf = hal.device.makeBuffer(length: payB, options: .storageModeShared)!
                let kmBuf = hal.device.makeBuffer(length: metaB, options: .storageModeShared)!
                let vmBuf = hal.device.makeBuffer(length: metaB, options: .storageModeShared)!
                func fillStream(_ src: [Float], _ pay: any MTLBuffer, _ meta: any MTLBuffer) -> [Float] {
                    var deq = [Float](repeating: 0, count: totalKV * nKV * hd)
                    for p in 0..<totalKV {
                        let slot = p % ringCap
                        for kvh in 0..<nKV {
                            let sOff = (p * nKV + kvh) * hd, rh = slot * nKV + kvh
                            if mode == .f16 {
                                let d = pay.contents().assumingMemoryBound(to: UInt16.self)
                                for i in 0..<hd { d[rh * hd + i] = Float16(src[sOff + i]).bitPattern; deq[sOff + i] = src[sOff + i] }
                            } else if mode == .q8 {
                                let mx = (0..<hd).map { abs(src[sOff + $0]) }.max() ?? 0
                                let sc16 = Float16(max(mx / 127.0, 6.103515625e-5))
                                meta.contents().assumingMemoryBound(to: UInt16.self)[rh] = sc16.bitPattern
                                let d = pay.contents().assumingMemoryBound(to: Int8.self)
                                for i in 0..<hd {
                                    let q = Int8(max(-127, min(127, Int((src[sOff + i] / Float(sc16)).rounded(.toNearestOrEven)))))
                                    d[rh * hd + i] = q; deq[sOff + i] = Float(Float16(Float(q)) * sc16)
                                }
                            } else {
                                let slice = (0..<hd).map { src[sOff + $0] }
                                let mn16 = Float16(slice.min() ?? 0)
                                let sc16 = Float16(max(((slice.max() ?? 0) - Float(mn16)) / 15.0, 6.103515625e-5))
                                let mp = meta.contents().assumingMemoryBound(to: UInt16.self)
                                mp[rh * 2] = sc16.bitPattern; mp[rh * 2 + 1] = mn16.bitPattern
                                let d = pay.contents().assumingMemoryBound(to: UInt8.self)
                                for i in 0..<(hd / 2) {
                                    let q0 = max(0, min(15, Int(((slice[i * 2] - Float(mn16)) / Float(sc16)).rounded(.toNearestOrEven))))
                                    let q1 = max(0, min(15, Int(((slice[i * 2 + 1] - Float(mn16)) / Float(sc16)).rounded(.toNearestOrEven))))
                                    d[rh * (hd / 2) + i] = UInt8(q0 | (q1 << 4))
                                    deq[sOff + i * 2] = Float(Float16(Float(q0)) * sc16 + mn16)
                                    deq[sOff + i * 2 + 1] = Float(Float16(Float(q1)) * sc16 + mn16)
                                }
                            }
                        }
                    }
                    return deq
                }
                let kDeq = fillStream(kRaw, kBuf, kmBuf)
                let vDeq = fillStream(vRaw, vBuf, vmBuf)
                let kv = QWSinkRelKVCache(kCache: QWMRegion(buffer: kBuf, base: 0, length: payB),
                                          vCache: QWMRegion(buffer: vBuf, base: 0, length: payB),
                                          kMeta: mode == .f16 ? nil : QWMRegion(buffer: kmBuf, base: 0, length: metaB),
                                          vMeta: mode == .f16 ? nil : QWMRegion(buffer: vmBuf, base: 0, length: metaB),
                                          ringCap: ringCap, mode: mode)
                let qTV = makeBF16TV(hal.device, qVals, dims: [T, nH, hd])
                let rTV = makeBF16TV(hal.device, rVals, dims: [T, nH, dRel])
                let relTV = makeBF16TV(hal.device, relProjVals, dims: [dRel, relExt])
                let sinkTV = makeBF16TV(hal.device, sinkVals, dims: [T, nH])
                let gateTV = makeBF16TV(hal.device, gateVals, dims: [T, nH, hd])
                let outTV = makeBF16TV(hal.device, [Float](repeating: 0, count: T * nH * hd), dims: [T, nH, hd])
                let sfx = mode == .f16 ? "f16" : (mode == .q8 ? "q8" : "q4")
                let mmaPSO = try hal.pipeline("qwg_attn_prefill_sink_rel_mma_\(sfx)")
                var params = op.makeMetalParams(config: cfg, T: T, posOffset: posOff, kvLen: totalKV, ringCap: ringCap, splits: 1)

                try runCmd(hal) { enc in
                    enc.setComputePipelineState(mmaPSO)
                    qTV.bind(enc, index: 0); rTV.bind(enc, index: 1); relTV.bind(enc, index: 2)
                    sinkTV.bind(enc, index: 3); gateTV.bind(enc, index: 4)
                    enc.setBuffer(kBuf, offset: 0, index: 5); enc.setBuffer(vBuf, offset: 0, index: 6)
                    enc.setBuffer(kmBuf, offset: 0, index: 7); enc.setBuffer(vmBuf, offset: 0, index: 8)
                    outTV.bind(enc, index: 9)
                    enc.setBytes(&params, length: MemoryLayout<QWAttnSinkRel.MetalParams>.stride, index: 10)
                    enc.dispatchThreadgroups(MTLSize(width: (T + 31) / 32, height: nKV, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 64 * (nH / nKV), height: 1, depth: 1))
                }
                let got = readBF16(outTV, count: T * nH * hd)
                var maxErr: Float = 0
                let sc = 1.0 / Double(hd).squareRoot()
                for t in 0..<T {
                    let qPos = posOff + t, minK = max(0, qPos + 1 - sliding)
                    for h in 0..<nH {
                        let kvh = h / (nH / nKV)
                        var scores = [Double]()
                        for j in minK...qPos {
                            var dot = 0.0
                            for d in 0..<hd { dot += Double(qVals[(t * nH + h) * hd + d]) * Double(kDeq[(j * nKV + kvh) * hd + d]) }
                            var s = dot * sc
                            let dist = qPos - j
                            if dist < relExt {
                                for dr in 0..<dRel { s += Double(rVals[(t * nH + h) * dRel + dr]) * Double(relProjVals[dr * relExt + dist]) }
                            }
                            scores.append(s)
                        }
                        let sSink = Double(sinkVals[t * nH + h])
                        let mAll = max(sSink, scores.max() ?? -1e30)
                        var wSum = exp(sSink - mAll)
                        var acc = [Double](repeating: 0, count: hd)
                        for (idx, j) in (minK...qPos).enumerated() {
                            let w = exp(scores[idx] - mAll); wSum += w
                            for d in 0..<hd { acc[d] += w * Double(vDeq[(j * nKV + kvh) * hd + d]) }
                        }
                        for d in 0..<hd {
                            let g = Double(gateVals[(t * nH + h) * hd + d])
                            let ref = Float((acc[d] / wSum) * (1.0 / (1.0 + exp(-g))))
                            maxErr = max(maxErr, abs(got[(t * nH + h) * hd + d] - ref))
                        }
                    }
                }
                XCTAssertLessThan(maxErr, 2e-2, "sink_rel dense dRel=\(dRel) \(mode) err=\(maxErr)")

                // Sparse sink_rel MMA kernel verification (all blocks present -> matches dense!)
                let spTiles = (T + 31) / 32
                let stride = (totalKV / 4) + 1
                var bList = [Int32](repeating: 0, count: spTiles * stride)
                var bCnt = [Int32](repeating: 0, count: spTiles)
                for tile in 0..<spTiles {
                    let kEnd = min(posOff + min((tile + 1) * 32, T), totalKV)
                    let complete = kEnd >> 2
                    for b in 0..<complete { bList[tile * stride + b] = Int32(b) }
                    bCnt[tile] = Int32(complete)
                }
                let blBuf = hal.device.makeBuffer(bytes: bList, length: max(16, bList.count * 4), options: .storageModeShared)!
                let bcBuf = hal.device.makeBuffer(bytes: bCnt, length: max(16, bCnt.count * 4), options: .storageModeShared)!
                let blTV = QWTensorValue(region: QWMRegion(buffer: blBuf, base: 0, length: blBuf.length),
                                         offset: 0, shape: QWTensorShape([bList.count], .i32))
                let bcTV = QWTensorValue(region: QWMRegion(buffer: bcBuf, base: 0, length: bcBuf.length),
                                         offset: 0, shape: QWTensorShape([bCnt.count], .i32))
                let outSpTV = makeBF16TV(hal.device, [Float](repeating: 0, count: T * nH * hd), dims: [T, nH, hd])
                let spPSO = try hal.pipeline("qwg_attn_prefill_sink_rel_sparse_mma_\(sfx)")
                var spMeta: (UInt32, UInt32, UInt32, UInt32) = (UInt32(stride), 0, 0, 0)
                try runCmd(hal) { enc in
                    enc.setComputePipelineState(spPSO)
                    qTV.bind(enc, index: 0); rTV.bind(enc, index: 1); relTV.bind(enc, index: 2)
                    sinkTV.bind(enc, index: 3); gateTV.bind(enc, index: 4)
                    enc.setBuffer(kBuf, offset: 0, index: 5); enc.setBuffer(vBuf, offset: 0, index: 6)
                    enc.setBuffer(kmBuf, offset: 0, index: 7); enc.setBuffer(vmBuf, offset: 0, index: 8)
                    outSpTV.bind(enc, index: 9)
                    enc.setBytes(&params, length: MemoryLayout<QWAttnSinkRel.MetalParams>.stride, index: 10)
                    blTV.bind(enc, index: 11); bcTV.bind(enc, index: 12)
                    enc.setBytes(&spMeta, length: 16, index: 13)
                    enc.dispatchThreadgroups(MTLSize(width: spTiles, height: nKV, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 64 * (nH / nKV), height: 1, depth: 1))
                }
                let gotSp = readBF16(outSpTV, count: T * nH * hd)
                let diffSp = zip(got, gotSp).map { abs($0 - $1) }.max() ?? 999
                XCTAssertLessThan(diffSp, 1e-2, "sink_rel sparse dRel=\(dRel) \(mode) diff=\(diffSp)")

                // Also ensure encodePrefill/encodePrefillSparse succeed without error on current HAL
                try runCmd(hal) { enc in
                    try op.encodePrefill(enc, config: cfg, q: qTV, r: rTV, relProj: relTV,
                                         sink: sinkTV, gate: gateTV, kv: kv, out: outSpTV, T: T, posOffset: posOff)
                }
            }
        }

        // Verify BF16 > 65504 clamping never produces Inf/NaN
        let extQ = [Float](repeating: 1e6, count: 16 * nH * hd)
        let extK = [Float](repeating: 0.01, count: 16 * nKV * hd)
        let extKBuf = makeF16Buf(hal.device, extK)
        let extOut = makeBF16TV(hal.device, [Float](repeating: 0, count: 16 * nH * hd), dims: [16, nH, hd])
        let extQTV = makeBF16TV(hal.device, extQ, dims: [16, nH, hd])
        let dummy = makeBF16TV(hal.device, [0], dims: [1])
        let extCfg = QWSinkRelConfig(nH: nH, nKV: nKV, headDim: hd)
        var extP = op.makeMetalParams(config: extCfg, T: 16, posOffset: 0, kvLen: 16, ringCap: 16, splits: 1)
        let f16PSO = try hal.pipeline("qwg_attn_prefill_sink_rel_mma_f16")
        try runCmd(hal) { enc in
            enc.setComputePipelineState(f16PSO)
            extQTV.bind(enc, index: 0); dummy.bind(enc, index: 1); dummy.bind(enc, index: 2)
            dummy.bind(enc, index: 3); dummy.bind(enc, index: 4)
            enc.setBuffer(extKBuf, offset: 0, index: 5); enc.setBuffer(extKBuf, offset: 0, index: 6)
            enc.setBuffer(extKBuf, offset: 0, index: 7); enc.setBuffer(extKBuf, offset: 0, index: 8)
            extOut.bind(enc, index: 9)
            enc.setBytes(&extP, length: MemoryLayout<QWAttnSinkRel.MetalParams>.stride, index: 10)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: nKV, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64 * (nH / nKV), height: 1, depth: 1))
        }
        let clampVals = readBF16(extOut, count: 16 * nH * hd)
        XCTAssertTrue(clampVals.allSatisfy { $0.isFinite && abs($0 - 0.01) < 1e-2 })
    }

    func testMLASparseAndDSASplitMMAParity() throws {
        let cfg = """
        {"model_type":"glm5_next_text","hidden_size":2048,"intermediate_size":4096,"vocab_size":1024,
         "num_hidden_layers":1,"layer_types":["deepseek_sparse_attention"],"rms_norm_eps":1e-5,
         "max_position_embeddings":4096,"num_attention_heads":16,"num_key_value_heads":16,
         "qk_head_dim":256,"qk_nope_head_dim":192,"qk_rope_head_dim":64,"v_head_dim":256,
         "kv_lora_rank":512,"q_lora_rank":1536,"mla_use_nope":true,"partial_rotary_factor":1.0,
         "index_n_heads":32,"index_head_dim":128,"index_topk":128,"eos_token_id":2}
        """.data(using: .utf8)!
        let geom = QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: cfg))
        let hal = try QWHal(geometry: geom)
        let mlaOp = try QWMLAAttnOperator(hal: hal, geometry: geom, numAttnLayers: 1, maxContext: 512, maxPrefillChunk: 16)
        XCTAssertGreaterThan(mlaOp.attnSplits, 1)

        var rng = RNG(state: 0x31A5917)
        let rows = 4, qHeads = 16, rank = 512, halfRank = 256, pos0 = 48, nSplit = 8
        let totalToks = pos0 + rows, topK = 16
        let kRaw = (0..<(totalToks * halfRank)).map { _ in Float(Float16(rng.nextSigned(0.25))) }
        let vRaw = (0..<(totalToks * halfRank)).map { _ in Float(Float16(rng.nextSigned(0.25))) }
        let qRaw = (0..<(rows * qHeads * rank)).map { _ in Float(Float16(rng.nextSigned(0.25))) }
        let qBuf = makeF16Buf(hal.device, qRaw)
        let kPoolBuf = makeF16Buf(hal.device, kRaw)
        let vPoolBuf = makeF16Buf(hal.device, vRaw)
        let ktBuf = hal.device.makeBuffer(bytes: [Int32](repeating: 0, count: 4), length: 16, options: .storageModeShared)!
        let vtBuf = hal.device.makeBuffer(bytes: [Int32](repeating: 0, count: 4), length: 16, options: .storageModeShared)!
        let blBuf = hal.device.makeBuffer(length: max(16, rows * topK * 4), options: .storageModeShared)!
        let partOBuf = hal.device.makeBuffer(length: rows * nSplit * qHeads * rank * 2, options: .storageModeShared)!
        let partMLBuf = hal.device.makeBuffer(length: rows * nSplit * qHeads * 8, options: .storageModeShared)!
        let outSplitBuf = makeF16Buf(hal.device, [Float](repeating: 0, count: rows * qHeads * rank))
        let outScalarBuf = makeF16Buf(hal.device, [Float](repeating: 0, count: rows * qHeads * rank))

        let splitPSO = try hal.pipeline("qwg_mla_sparse_attn_mma_split")
        let combinePSO = try hal.pipeline("qwg_mla_sparse_attn_combine")
        let scalarPSO = try hal.pipeline("qwg_mla_sparse_attn")
        var pNoPE = SparseAttnParams(T: UInt32(rows), pos0: UInt32(pos0), qHeads: UInt32(qHeads),
                                     headDim: 192, kvLoraRank: UInt32(rank), topKPools: UInt32(topK),
                                     listStride: UInt32(topK), kvMode: 0)
        try runCmd(hal) { enc in
            enc.setComputePipelineState(scalarPSO)
            for (i, b) in [qBuf, kPoolBuf, vPoolBuf, ktBuf, vtBuf, blBuf, outScalarBuf].enumerated() {
                enc.setBuffer(b, offset: 0, index: i)
            }
            enc.setBytes(&pNoPE, length: MemoryLayout<SparseAttnParams>.stride, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: qHeads * rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(splitPSO)
            for (i, b) in [qBuf, kPoolBuf, vPoolBuf, ktBuf, vtBuf, blBuf, partOBuf].enumerated() {
                enc.setBuffer(b, offset: 0, index: i)
            }
            enc.setBytes(&pNoPE, length: MemoryLayout<SparseAttnParams>.stride, index: 7)
            enc.setBuffer(partMLBuf, offset: 0, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: rows, height: qHeads / 16, depth: nSplit),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            var nSU = UInt32(nSplit)
            enc.setComputePipelineState(combinePSO)
            enc.setBuffer(partOBuf, offset: 0, index: 0)
            enc.setBuffer(partMLBuf, offset: 0, index: 1)
            enc.setBuffer(outSplitBuf, offset: 0, index: 2)
            enc.setBytes(&pNoPE, length: MemoryLayout<SparseAttnParams>.stride, index: 3)
            enc.setBytes(&nSU, length: 4, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: rows * qHeads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
        let sVals = readF16(outSplitBuf, count: rows * qHeads * rank)
        let refVals = readF16(outScalarBuf, count: rows * qHeads * rank)
        let errNoPE = zip(sVals, refVals).map { abs($0 - $1) }.max() ?? 999
        XCTAssertLessThan(errNoPE, 1.5e-2, "MLA mma_split err=\(errNoPE)")

        // DSA split with sink, empty splits (nSplit=16 vs 5 tokens), and ropeDim in [0, 64, 128]
        let dsaSplitPSO = try hal.pipeline("qwg_mla_dsa_sparse_attn_mma_split")
        let dsaCombinePSO = try hal.pipeline("qwg_mla_dsa_sparse_attn_combine")
        let dsaScalarPSO = try hal.pipeline("qwg_mla_dsa_sparse_attn")
        for ropeDim in [0, 64, 128] {
            let dsaRows = 2, dsaPos0 = 3, dsaSplits = 16
            let qFullStride = qHeads * max(1, ropeDim)
            let qRopeVals = (0..<(dsaRows * qFullStride)).map { _ in Float(Float16(rng.nextSigned(0.25))) }
            let kPeVals = (0..<((dsaPos0 + dsaRows) * max(1, ropeDim))).map { _ in Float(Float16(rng.nextSigned(0.25))) }
            let sinkF32 = (0..<qHeads).map { _ in Float(rng.nextSigned(0.5)) }
            let qFullBuf = makeF16Buf(hal.device, qRopeVals)
            let kPeBuf = makeF16Buf(hal.device, kPeVals)
            let sinkBuf = hal.device.makeBuffer(bytes: sinkF32, length: qHeads * 4, options: .storageModeShared)!
            let pOBuf = hal.device.makeBuffer(length: dsaRows * dsaSplits * qHeads * rank * 2, options: .storageModeShared)!
            let pMLBuf = hal.device.makeBuffer(length: dsaRows * dsaSplits * qHeads * 8, options: .storageModeShared)!
            let outDBuf = makeF16Buf(hal.device, [Float](repeating: 0, count: dsaRows * qHeads * rank))
            let outDRBuf = makeF16Buf(hal.device, [Float](repeating: 0, count: dsaRows * qHeads * rank))
            var pDSA = DSASparseAttnParams(
                T: UInt32(dsaRows), pos0: UInt32(dsaPos0), qHeads: UInt32(qHeads),
                headDim: UInt32(192 + ropeDim), kvLoraRank: UInt32(rank), topK: 0, windowTokens: 64,
                listStride: 1, kvMode: 0, ropeDim: UInt32(ropeDim), qRopeOffset: 0,
                qFullStride: UInt32(qFullStride), hasSink: 1,
                softmaxScale: ropeDim == 64 ? 0.1308608 : 0.0
            )
            try runCmd(hal) { enc in
                enc.setComputePipelineState(dsaScalarPSO)
                for (i, b) in [qBuf, kPoolBuf, vPoolBuf, ktBuf, vtBuf, blBuf, outDRBuf].enumerated() {
                    enc.setBuffer(b, offset: 0, index: i)
                }
                enc.setBytes(&pDSA, length: MemoryLayout<DSASparseAttnParams>.stride, index: 7)
                enc.setBuffer(qFullBuf, offset: 0, index: 8)
                enc.setBuffer(kPeBuf, offset: 0, index: 9)
                enc.setBuffer(sinkBuf, offset: 0, index: 10)
                enc.dispatchThreadgroups(MTLSize(width: qHeads * dsaRows, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(dsaSplitPSO)
                for (i, b) in [qBuf, kPoolBuf, vPoolBuf, ktBuf, vtBuf, blBuf, pOBuf].enumerated() {
                    enc.setBuffer(b, offset: 0, index: i)
                }
                enc.setBytes(&pDSA, length: MemoryLayout<DSASparseAttnParams>.stride, index: 7)
                enc.setBuffer(pMLBuf, offset: 0, index: 8)
                enc.setBuffer(qFullBuf, offset: 0, index: 9)
                enc.setBuffer(kPeBuf, offset: 0, index: 10)
                enc.setBuffer(sinkBuf, offset: 0, index: 11)
                enc.dispatchThreadgroups(MTLSize(width: dsaRows, height: qHeads / 16, depth: dsaSplits),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
                var nSU = UInt32(dsaSplits)
                enc.setComputePipelineState(dsaCombinePSO)
                enc.setBuffer(pOBuf, offset: 0, index: 0)
                enc.setBuffer(pMLBuf, offset: 0, index: 1)
                enc.setBuffer(outDBuf, offset: 0, index: 2)
                enc.setBytes(&pDSA, length: MemoryLayout<DSASparseAttnParams>.stride, index: 3)
                enc.setBytes(&nSU, length: 4, index: 4)
                enc.setBuffer(sinkBuf, offset: 0, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: dsaRows * qHeads, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
            let dsaGot = readF16(outDBuf, count: dsaRows * qHeads * rank)
            let dsaRef = readF16(outDRBuf, count: dsaRows * qHeads * rank)
            let errDSA = zip(dsaGot, dsaRef).map { abs($0 - $1) }.max() ?? 999
            XCTAssertLessThan(errDSA, 1.5e-2, "DSA mma_split ropeDim=\(ropeDim) err=\(errDSA)")
        }
    }
}
