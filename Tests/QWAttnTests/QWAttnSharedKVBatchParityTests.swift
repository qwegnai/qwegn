// QWAttnSharedKVBatchParityTests.swift — 0-ULP parity for 4-SG Sink+RelBias (T=2..4) across f16, q8, q4.
import Foundation
import Metal
import XCTest
@testable import QWAttn
import QWAlloc
import QWCore
import QWHAL
import QWKV
import QWQuant

final class QWAttnSharedKVBatchParityTests: XCTestCase {

    private struct SplitMix64 {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &+ 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        mutating func nextSigned(_ scale: Double = 1.0) -> Float {
            Float((Double(next() >> 11) / Double(1 << 53) * 2.0 - 1.0) * scale)
        }
    }

    private static func makeGeometry(
        qHeads: Int, kvHeads: Int, headDim: Int, gated: Bool = false, slidingWindow: Int? = nil
    ) throws -> QWModelGeometry {
        var cfg: [String: Any] = [
            "model_type": slidingWindow == nil ? "llama" : "mistral",
            "hidden_size": qHeads * headDim,
            "num_hidden_layers": 1,
            "num_attention_heads": qHeads,
            "num_key_value_heads": kvHeads,
            "head_dim": headDim,
            "intermediate_size": 256,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1000000.0,
            "eos_token_id": 2,
            "vocab_size": 128,
        ]
        if gated { cfg["attn_output_gate"] = true }
        if let slidingWindow { cfg["sliding_window"] = slidingWindow }
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(
            configJSON: try JSONSerialization.data(withJSONObject: cfg)
        ))
    }

    private static func synthBF16(count: Int, seed: UInt64, scale: Double) -> [UInt16] {
        var rng = SplitMix64(state: seed)
        return [UInt16](unsafeUninitializedCapacity: count) { dst, initialized in
            guard let out = dst.baseAddress else { initialized = 0; return }
            var i = 0
            while i < count { out[i] = QWBF16.fromFloatRNE(rng.nextSigned(scale)); i += 1 }
            initialized = count
        }
    }

    private static func makeU16Tensor(_ hal: QWHal, _ bits: [UInt16], dims: [Int], dtype: QWDType) -> QWTensorValue {
        let bytes = max(16, bits.count * 2)
        let buf = hal.device.makeBuffer(length: bytes, options: .storageModeShared)!
        memset(buf.contents(), 0, bytes)
        bits.withUnsafeBytes { raw in
            if let src = raw.baseAddress, !raw.isEmpty { memcpy(buf.contents(), src, raw.count) }
        }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: bytes),
                             offset: 0, shape: QWTensorShape(dims, dtype))
    }

    private static func emptyU16Tensor(_ hal: QWHal, dims: [Int], dtype: QWDType) -> QWTensorValue {
        let bytes = max(16, dims.reduce(1, *) * 2)
        let buf = hal.device.makeBuffer(length: bytes, options: .storageModeShared)!
        memset(buf.contents(), 0, bytes)
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: bytes),
                             offset: 0, shape: QWTensorShape(dims, dtype))
    }

    private static func readBits(_ tv: QWTensorValue, count: Int) -> [UInt16] {
        let ptr = tv.region.buffer.contents().advanced(by: tv.region.base + tv.offset)
            .assumingMemoryBound(to: UInt16.self)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    private static func buildSinkRelKV(
        hal: QWHal, mode: QWKVMode, kBits: [UInt16], vBits: [UInt16],
        totalTokens: Int, ringCap: Int, slidingWindow: Int, nKV: Int, headDim: Int
    ) -> QWSinkRelKVCache {
        let rowBytes = QWAttnSinkRel.kvRowBytes(mode: mode, nKV: nKV, headDim: headDim)
        let payBytes = max(16, ringCap * rowBytes.payload)
        let metaBytes = max(16, ringCap * max(4, rowBytes.meta))
        let kBuf = hal.device.makeBuffer(length: payBytes, options: .storageModeShared)!
        let vBuf = hal.device.makeBuffer(length: payBytes, options: .storageModeShared)!
        let kmBuf = hal.device.makeBuffer(length: metaBytes, options: .storageModeShared)!
        let vmBuf = hal.device.makeBuffer(length: metaBytes, options: .storageModeShared)!
        memset(kBuf.contents(), 0, payBytes)
        memset(vBuf.contents(), 0, payBytes)
        memset(kmBuf.contents(), 0, metaBytes)
        memset(vmBuf.contents(), 0, metaBytes)

        func write(_ src: [UInt16], pay: any MTLBuffer, meta: any MTLBuffer) {
            src.withUnsafeBufferPointer { sBuf in
                guard let sp = sBuf.baseAddress else { return }
                withUnsafeTemporaryAllocation(of: Float.self, capacity: headDim) { vTmp in
                    guard let vp = vTmp.baseAddress else { return }
                    let dstU16 = pay.contents().assumingMemoryBound(to: UInt16.self)
                    let dstI8 = pay.contents().assumingMemoryBound(to: Int8.self)
                    let dstU8 = pay.contents().assumingMemoryBound(to: UInt8.self)
                    let mPtr = meta.contents().assumingMemoryBound(to: UInt16.self)
                    var pos = 0
                    while pos < totalTokens {
                        let slot = slidingWindow > 0 ? (pos % ringCap) : pos
                        var kvh = 0
                        while kvh < nKV {
                            let srcOff = (pos * nKV + kvh) * headDim
                            let rowHead = slot * nKV + kvh
                            var d = 0, minV: Float = .greatestFiniteMagnitude, maxV: Float = -.greatestFiniteMagnitude, maxAbs: Float = 0
                            while d < headDim {
                                let fv = Float(Float16(Float(bitPattern: UInt32(sp[srcOff + d]) << 16)))
                                vp[d] = fv
                                if fv < minV { minV = fv }
                                if fv > maxV { maxV = fv }
                                let av = abs(fv)
                                if av > maxAbs { maxAbs = av }
                                d += 1
                            }
                            switch mode {
                            case .f16:
                                let rowOut = rowHead * headDim
                                d = 0
                                while d < headDim { dstU16[rowOut + d] = Float16(vp[d]).bitPattern; d += 1 }
                            case .q8:
                                let sc16 = Float16(max(maxAbs / 127.0, 6.103515625e-5))
                                let inv = Float(sc16) > 0 ? 1.0 / Float(sc16) : 0.0
                                mPtr[rowHead] = sc16.bitPattern
                                let rowOut = rowHead * headDim
                                d = 0
                                while d < headDim {
                                    dstI8[rowOut + d] = Int8(max(-127, min(127, Int((vp[d] * inv).rounded(.toNearestOrEven)))))
                                    d += 1
                                }
                            case .q4:
                                let mn16 = Float16(minV)
                                let mnF = Float(mn16)
                                let sc16 = Float16(max((maxV - mnF) / 15.0, 6.103515625e-5))
                                let inv = 1.0 / max(Float(sc16), 1e-7)
                                mPtr[rowHead * 2] = sc16.bitPattern
                                mPtr[rowHead * 2 + 1] = mn16.bitPattern
                                let halfDim = headDim / 2
                                let rowOut = rowHead * halfDim
                                var p = 0
                                while p < halfDim {
                                    let q0 = max(0, min(15, Int(((vp[p * 2] - mnF) * inv).rounded(.toNearestOrEven))))
                                    let q1 = max(0, min(15, Int(((vp[p * 2 + 1] - mnF) * inv).rounded(.toNearestOrEven))))
                                    dstU8[rowOut + p] = UInt8(q0 | (q1 << 4))
                                    p += 1
                                }
                            }
                            kvh += 1
                        }
                        pos += 1
                    }
                }
            }
        }
        write(kBits, pay: kBuf, meta: kmBuf)
        write(vBits, pay: vBuf, meta: vmBuf)
        return QWSinkRelKVCache(
            kCache: QWMRegion(buffer: kBuf, base: 0, length: payBytes),
            vCache: QWMRegion(buffer: vBuf, base: 0, length: payBytes),
            kMeta: mode == .f16 ? nil : QWMRegion(buffer: kmBuf, base: 0, length: metaBytes),
            vMeta: mode == .f16 ? nil : QWMRegion(buffer: vmBuf, base: 0, length: metaBytes),
            ringCap: ringCap, mode: mode
        )
    }

    func testSinkRelT2ToT4AndT8ZeroULPParityLinearAndTree() throws {
        let headDim = 128, dRel = 16, relExtent = 64
        let parents = [-1, 0, 0, 1]
        var depths = [UInt32](repeating: 0, count: parents.count)
        var ancMask = [UInt32](repeating: 0, count: parents.count)
        for i in 0..<parents.count {
            depths[i] = parents[i] < 0 ? 0 : (depths[parents[i]] + 1)
            ancMask[i] = (parents[i] < 0 ? 0 : ancMask[parents[i]]) | (UInt32(1) << i)
        }
        let treeFull = QWAttnSinkRel.TreeMeta(ancMask: ancMask, depths: depths)

        for (nKV, gqa) in [(4, 3), (4, 2), (2, 4), (2, 6)] {
            let nH = nKV * gqa
            let hal = try QWHal(geometry: try Self.makeGeometry(qHeads: nH, kvHeads: nKV, headDim: headDim))
            let op = try QWAttnSinkRel(hal: hal)

            for (sw, posOffset, ringCap) in [(1024, 1100, 1040), (0, 2050, 2064)] {
                let maxT = 8
                let totalKV = posOffset + maxT
                let qAll = Self.synthBF16(count: maxT * nH * headDim, seed: 801, scale: 2.0)
                let kAll = Self.synthBF16(count: totalKV * nKV * headDim, seed: 802, scale: 2.0)
                let vAll = Self.synthBF16(count: totalKV * nKV * headDim, seed: 803, scale: 0.9)
                let rAll = Self.synthBF16(count: maxT * nH * dRel, seed: 804, scale: 1.1)
                let relProj = Self.synthBF16(count: dRel * relExtent, seed: 805, scale: 0.25)
                let sinkAll = Self.synthBF16(count: maxT * nH, seed: 806, scale: 1.0)
                let gateAll = Self.synthBF16(count: maxT * nH * headDim, seed: 807, scale: 1.3)
                let rpTV = Self.makeU16Tensor(hal, relProj, dims: [dRel, relExtent], dtype: .bf16)
                let pSizes = QWAttnSinkRel.partialsBytes(
                    tokens: maxT, qHeads: nH, headDim: headDim, splits: QWAttnSinkRel.maxSplits
                )
                let parts = QWSinkRelPartials(
                    partO: QWMRegion(buffer: hal.device.makeBuffer(length: pSizes.partOBytes, options: .storageModeShared)!,
                                     base: 0, length: pSizes.partOBytes),
                    partML: QWMRegion(buffer: hal.device.makeBuffer(length: pSizes.partMLBytes, options: .storageModeShared)!,
                                      base: 0, length: pSizes.partMLBytes)
                )

                for mode in [QWKVMode.f16, .q8, .q4] {
                    let cfg = QWSinkRelConfig(
                        nH: nH, nKV: nKV, headDim: headDim, dRel: dRel, relExtent: relExtent,
                        slidingWindow: sw, hasSink: true, hasRelBias: true, hasOutGate: true, kvMode: mode
                    )
                    let kv = Self.buildSinkRelKV(
                        hal: hal, mode: mode, kBits: kAll, vBits: vAll,
                        totalTokens: totalKV, ringCap: ringCap, slidingWindow: sw, nKV: nKV, headDim: headDim
                    )

                    func run(
                        _ cnt: Int, _ qSlice: [UInt16], _ rSlice: [UInt16],
                        _ sSlice: [UInt16], _ gSlice: [UInt16], _ kvArg: QWSinkRelKVCache,
                        _ pos: Int, _ tree: QWAttnSinkRel.TreeMeta?
                    ) throws -> [UInt16] {
                        let oTV = Self.emptyU16Tensor(hal, dims: [cnt, nH, headDim], dtype: .bf16)
                        guard let (cb, enc) = hal.begin() else { XCTFail("begin failed"); return [] }
                        try op.encodeDecode(
                            enc, config: cfg,
                            q: Self.makeU16Tensor(hal, qSlice, dims: [cnt, nH, headDim], dtype: .bf16),
                            r: Self.makeU16Tensor(hal, rSlice, dims: [cnt, nH, dRel], dtype: .bf16),
                            relProj: rpTV,
                            sink: Self.makeU16Tensor(hal, sSlice, dims: [cnt, nH], dtype: .bf16),
                            gate: Self.makeU16Tensor(hal, gSlice, dims: [cnt, nH, headDim], dtype: .bf16),
                            kv: kvArg, partials: parts, out: oTV, T: cnt, posOffset: pos,
                            splits: nil, treeMeta: tree
                        )
                        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                        XCTAssertEqual(cb.status, .completed)
                        return Self.readBits(oTV, count: cnt * nH * headDim)
                    }

                    var t1Linear = [[UInt16]]()
                    for i in 0..<maxT {
                        let qo = i * nH * headDim, ro = i * nH * dRel, so = i * nH
                        t1Linear.append(try run(
                            1,
                            Array(qAll[qo..<(qo + nH * headDim)]),
                            Array(rAll[ro..<(ro + nH * dRel)]),
                            Array(sinkAll[so..<(so + nH)]),
                            Array(gateAll[qo..<(qo + nH * headDim)]),
                            kv, posOffset + i, nil
                        ))
                    }

                    for batchT in [2, 3, 4, 8] {
                        let outLin = try run(
                            batchT,
                            Array(qAll[0..<(batchT * nH * headDim)]),
                            Array(rAll[0..<(batchT * nH * dRel)]),
                            Array(sinkAll[0..<(batchT * nH)]),
                            Array(gateAll[0..<(batchT * nH * headDim)]),
                            kv, posOffset, nil
                        )
                        for i in 0..<batchT {
                            let row = Array(outLin[(i * nH * headDim)..<((i + 1) * nH * headDim)])
                            XCTAssertEqual(row, t1Linear[i],
                                           "0-ULP linear mismatch gqa=\(gqa) sw=\(sw) mode=\(mode) T=\(batchT) tok=\(i)")
                        }
                    }

                    for treeT in [1, 2, 3, 4] {
                        let meta = QWAttnSinkRel.TreeMeta(
                            ancMask: Array(treeFull.ancMask.prefix(treeT)),
                            depths: Array(treeFull.depths.prefix(treeT))
                        )
                        let outTree = try run(
                            treeT,
                            Array(qAll[0..<(treeT * nH * headDim)]),
                            Array(rAll[0..<(treeT * nH * dRel)]),
                            Array(sinkAll[0..<(treeT * nH)]),
                            Array(gateAll[0..<(treeT * nH * headDim)]),
                            kv, posOffset, meta
                        )
                        for i in 0..<treeT {
                            let qo = i * nH * headDim, ro = i * nH * dRel, so = i * nH
                            var path: [Int] = [], cur = i
                            while cur >= 0 { path.append(cur); cur = parents[cur] }
                            path.reverse()
                            var kSeq = Array(kAll[0..<(posOffset * nKV * headDim)])
                            var vSeq = Array(vAll[0..<(posOffset * nKV * headDim)])
                            for idx in path {
                                let s = (posOffset + idx) * nKV * headDim
                                kSeq.append(contentsOf: kAll[s..<(s + nKV * headDim)])
                                vSeq.append(contentsOf: vAll[s..<(s + nKV * headDim)])
                            }
                            let kvP = Self.buildSinkRelKV(
                                hal: hal, mode: mode, kBits: kSeq, vBits: vSeq,
                                totalTokens: posOffset + path.count, ringCap: ringCap,
                                slidingWindow: sw, nKV: nKV, headDim: headDim
                            )
                            let t1Tree = try run(
                                1,
                                Array(qAll[qo..<(qo + nH * headDim)]),
                                Array(rAll[ro..<(ro + nH * dRel)]),
                                Array(sinkAll[so..<(so + nH)]),
                                Array(gateAll[qo..<(qo + nH * headDim)]),
                                kvP, posOffset + path.count - 1, nil
                            )
                            let gotRow = Array(outTree[qo..<(qo + nH * headDim)])
                            XCTAssertEqual(gotRow, t1Tree,
                                           "0-ULP tree mismatch gqa=\(gqa) sw=\(sw) mode=\(mode) T=\(treeT) tok=\(i)")
                        }
                    }
                }
            }
        }
    }

    func testSinkRelSplitsTunedForBatch4SWAAndGlobal() {
        XCTAssertEqual(QWAttnSinkRel.splitSpan(ringCap: 1024, slidingWindow: 1024), 128)
        XCTAssertEqual(QWAttnSinkRel.splitSpan(ringCap: 8192, slidingWindow: 0), 256)
        XCTAssertGreaterThanOrEqual(
            QWAttnSinkRel.splits(forKeys: 1024, kvHeads: 4, batchTokens: 4, slidingWindow: 1024), 2
        )
        for keys in [2048, 4096, 8192, 16384, 32768] {
            let sp = QWAttnSinkRel.splits(forKeys: keys, kvHeads: 4, batchTokens: 4, slidingWindow: 0)
            XCTAssertGreaterThanOrEqual(sp, 2)
            XCTAssertLessThanOrEqual(sp, QWAttnSinkRel.maxSplits)
        }
    }
}
