// QWAttnSinkRelBatch8ParityTests.swift — Bit-exact parity for MAX_U=1 batch8 (T=5..8) vs T=1 reference.
import Foundation
import Metal
import XCTest
@testable import QWAttn
import QWAlloc
import QWCore
import QWHAL
import QWKV
import QWQuant

final class QWAttnSinkRelBatch8ParityTests: XCTestCase {
    private static func makeGeometry(qHeads: Int = 12, kvHeads: Int = 4, headDim: Int = 128) throws -> QWModelGeometry {
        let cfg: [String: Any] = [
            "model_type": "qwen3",
            "hidden_size": qHeads * headDim,
            "num_hidden_layers": 1,
            "num_attention_heads": qHeads,
            "num_key_value_heads": kvHeads,
            "head_dim": headDim,
            "intermediate_size": 512,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "eos_token_id": 2,
            "vocab_size": 128,
        ]
        let data = try JSONSerialization.data(withJSONObject: cfg)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

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
            let unit = Double(next() >> 11) / Double(1 << 53)
            return Float((unit * 2.0 - 1.0) * scale)
        }
    }

    private static func synthBF16(count: Int, seed: UInt64, scale: Double) -> [Float] {
        var rng = SplitMix64(state: seed)
        return (0..<count).map { _ in
            Float(bitPattern: UInt32(QWBF16.fromFloatRNE(rng.nextSigned(scale))) << 16)
        }
    }

    private static func makeBF16Tensor(_ hal: QWHal, _ vals: [Float], dims: [Int]) -> QWTensorValue {
        let bits = vals.map(QWBF16.fromFloatRNE)
        let bytes = max(16, bits.count * 2)
        let buf = hal.device.makeBuffer(length: bytes, options: .storageModeShared)!
        let dst = buf.contents()
        withExtendedLifetime(buf) {
            bits.withUnsafeBytes { raw in
                if let src = raw.baseAddress, !raw.isEmpty { dst.copyMemory(from: src, byteCount: raw.count) }
            }
        }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: bytes), offset: 0,
                             shape: QWTensorShape(dims, .bf16))
    }

    private static func emptyBF16Tensor(_ hal: QWHal, dims: [Int]) -> QWTensorValue {
        let bytes = max(16, dims.reduce(1, *) * 2)
        let buf = hal.device.makeBuffer(length: bytes, options: .storageModeShared)!
        memset(buf.contents(), 0, bytes)
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: bytes), offset: 0,
                             shape: QWTensorShape(dims, .bf16))
    }

    private static func readBF16Bits(_ tv: QWTensorValue, count: Int) -> [UInt16] {
        let ptr = tv.region.buffer.contents().advanced(by: tv.region.base + tv.offset)
            .assumingMemoryBound(to: UInt16.self)
        return (0..<count).map { ptr[$0] }
    }

    private static func buildKVCache(
        hal: QWHal, mode: QWKVMode, kRaw: [Float], vRaw: [Float],
        totalTokens: Int, ringCap: Int, slidingWindow: Int, nKV: Int, headDim: Int
    ) -> QWSinkRelKVCache {
        let rowBytes = QWAttnSinkRel.kvRowBytes(mode: mode, nKV: nKV, headDim: headDim)
        let payBytes = max(16, ringCap * rowBytes.payload)
        let metaBytes = max(16, ringCap * max(4, rowBytes.meta))
        let kBuf = hal.device.makeBuffer(length: payBytes, options: .storageModeShared)!
        let vBuf = hal.device.makeBuffer(length: payBytes, options: .storageModeShared)!
        let kmBuf = hal.device.makeBuffer(length: metaBytes, options: .storageModeShared)!
        let vmBuf = hal.device.makeBuffer(length: metaBytes, options: .storageModeShared)!
        memset(kBuf.contents(), 0, payBytes); memset(vBuf.contents(), 0, payBytes)
        memset(kmBuf.contents(), 0, metaBytes); memset(vmBuf.contents(), 0, metaBytes)

        func writeStream(src: [Float], payBuf: any MTLBuffer, metaBuf: any MTLBuffer) {
            for pos in 0..<totalTokens {
                let slot = slidingWindow > 0 ? (pos % ringCap) : pos
                for kvh in 0..<nKV {
                    let srcOff = (pos * nKV + kvh) * headDim
                    let rowHead = slot * nKV + kvh
                    let vals = (0..<headDim).map { Float(Float16(src[srcOff + $0])) }
                    switch mode {
                    case .f16:
                        let dst = payBuf.contents().assumingMemoryBound(to: UInt16.self)
                        for d in 0..<headDim { dst[rowHead * headDim + d] = Float16(vals[d]).bitPattern }
                    case .q8:
                        let sc16 = Float16(max((vals.map(abs).max() ?? 0) / 127.0, 6.103515625e-5))
                        let invSc = Float(sc16) > 0 ? (1.0 / Float(sc16)) : 0.0
                        let dst = payBuf.contents().assumingMemoryBound(to: Int8.self)
                        metaBuf.contents().assumingMemoryBound(to: UInt16.self)[rowHead] = sc16.bitPattern
                        for d in 0..<headDim {
                            dst[rowHead * headDim + d] = Int8(max(-127, min(127, Int((vals[d] * invSc).rounded(.toNearestOrEven)))))
                        }
                    case .q4:
                        let mn16 = Float16(vals.min() ?? 0)
                        let sc16 = Float16(max(((vals.max() ?? 0) - Float(mn16)) / 15.0, 6.103515625e-5))
                        let invSc = 1.0 / max(Float(sc16), 1e-7)
                        let dst = payBuf.contents().assumingMemoryBound(to: UInt8.self)
                        let meta = metaBuf.contents().assumingMemoryBound(to: UInt16.self)
                        meta[rowHead * 2] = sc16.bitPattern
                        meta[rowHead * 2 + 1] = mn16.bitPattern
                        for p in 0..<(headDim / 2) {
                            let q0 = max(0, min(15, Int(((vals[p * 2] - Float(mn16)) * invSc).rounded(.toNearestOrEven))))
                            let q1 = max(0, min(15, Int(((vals[p * 2 + 1] - Float(mn16)) * invSc).rounded(.toNearestOrEven))))
                            dst[rowHead * (headDim / 2) + p] = UInt8(q0 | (q1 << 4))
                        }
                    }
                }
            }
        }
        writeStream(src: kRaw, payBuf: kBuf, metaBuf: kmBuf)
        writeStream(src: vRaw, payBuf: vBuf, metaBuf: vmBuf)
        return QWSinkRelKVCache(
            kCache: QWMRegion(buffer: kBuf, base: 0, length: payBytes),
            vCache: QWMRegion(buffer: vBuf, base: 0, length: payBytes),
            kMeta: mode == .f16 ? nil : QWMRegion(buffer: kmBuf, base: 0, length: metaBytes),
            vMeta: mode == .f16 ? nil : QWMRegion(buffer: vmBuf, base: 0, length: metaBytes),
            ringCap: ringCap, mode: mode
        )
    }

    func testBatch8LinearAndTreeMatchT1BitExact() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && MTLCreateSystemDefaultDevice() != nil,
            "SinkRel batch8 Metal parity requires bare-metal Apple Silicon GPU"
        )
        let nH = 12, nKV = 2, headDim = 128, dRel = 16, relExtent = 64, posOffset = 300
        let hal = try QWHal(geometry: try Self.makeGeometry(qHeads: nH, kvHeads: nKV, headDim: headDim))
        let op = try QWAttnSinkRel(hal: hal)

        for tBatch in [5, 6, 7, 8] {
            var depths = [UInt32](repeating: 0, count: tBatch)
            var ancMask = [UInt32](repeating: 0, count: tBatch)
            for i in 0..<tBatch {
                let p = i == 0 ? -1 : (i / 2)
                depths[i] = p < 0 ? 0 : (depths[p] + 1)
                ancMask[i] = (p < 0 ? 0 : ancMask[p]) | (UInt32(1) << i)
            }
            let treeMeta = QWAttnSinkRel.TreeMeta(ancMask: ancMask, depths: depths)

            for (mode, sw, ringCap, splitsArg) in [
                (QWKVMode.f16, 0, 512, Optional<Int>.some(1)),
                (QWKVMode.q8, 128, 160, Optional<Int>.none),
                (QWKVMode.q4, 0, 512, Optional<Int>.none),
            ] {
                let cfg = QWSinkRelConfig(
                    nH: nH, nKV: nKV, headDim: headDim, dRel: dRel, relExtent: relExtent,
                    slidingWindow: sw, hasSink: true, hasRelBias: true, hasOutGate: true, kvMode: mode
                )
                let totalKV = posOffset + tBatch
                let q = Self.synthBF16(count: tBatch * nH * headDim, seed: UInt64(8100 + tBatch), scale: 2.0)
                let kRaw = Self.synthBF16(count: totalKV * nKV * headDim, seed: UInt64(8200 + tBatch), scale: 2.0)
                let vRaw = Self.synthBF16(count: totalKV * nKV * headDim, seed: UInt64(8300 + tBatch), scale: 0.9)
                let r = Self.synthBF16(count: tBatch * nH * dRel, seed: UInt64(8400 + tBatch), scale: 1.1)
                let relProj = Self.synthBF16(count: dRel * relExtent, seed: UInt64(8500 + tBatch), scale: 0.25)
                let sink = Self.synthBF16(count: tBatch * nH, seed: UInt64(8600 + tBatch), scale: 1.0)
                let gate = Self.synthBF16(count: tBatch * nH * headDim, seed: UInt64(8700 + tBatch), scale: 1.4)

                let kv = Self.buildKVCache(hal: hal, mode: mode, kRaw: kRaw, vRaw: vRaw,
                                           totalTokens: totalKV, ringCap: ringCap, slidingWindow: sw,
                                           nKV: nKV, headDim: headDim)
                let rpTV = Self.makeBF16Tensor(hal, relProj, dims: [dRel, relExtent])
                let pSizes = QWAttnSinkRel.partialsBytes(tokens: tBatch, qHeads: nH, headDim: headDim,
                                                         splits: QWAttnSinkRel.maxSplits)
                let parts = QWSinkRelPartials(
                    partO: QWMRegion(buffer: hal.device.makeBuffer(length: pSizes.partOBytes, options: .storageModeShared)!,
                                     base: 0, length: pSizes.partOBytes),
                    partML: QWMRegion(buffer: hal.device.makeBuffer(length: pSizes.partMLBytes, options: .storageModeShared)!,
                                      base: 0, length: pSizes.partMLBytes)
                )

                func run(_ cnt: Int, _ qi: [Float], _ ri: [Float], _ si: [Float], _ gi: [Float],
                         _ kvArg: QWSinkRelKVCache, _ pos: Int, _ tree: QWAttnSinkRel.TreeMeta?) throws -> [UInt16] {
                    let oTV = Self.emptyBF16Tensor(hal, dims: [cnt, nH, headDim])
                    guard let (cb, enc) = hal.begin() else { XCTFail("begin failed"); return [] }
                    try op.encodeDecode(
                        enc, config: cfg,
                        q: Self.makeBF16Tensor(hal, qi, dims: [cnt, nH, headDim]),
                        r: Self.makeBF16Tensor(hal, ri, dims: [cnt, nH, dRel]),
                        relProj: rpTV,
                        sink: Self.makeBF16Tensor(hal, si, dims: [cnt, nH]),
                        gate: Self.makeBF16Tensor(hal, gi, dims: [cnt, nH, headDim]),
                        kv: kvArg, partials: parts, out: oTV,
                        T: cnt, posOffset: pos, splits: splitsArg, treeMeta: tree
                    )
                    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                    XCTAssertNil(cb.error)
                    return Self.readBF16Bits(oTV, count: cnt * nH * headDim)
                }

                let outLin = try run(tBatch, q, r, sink, gate, kv, posOffset, nil)
                let outTree = try run(tBatch, q, r, sink, gate, kv, posOffset, treeMeta)

                for i in 0..<tBatch {
                    let qo = i * nH * headDim, ro = i * nH * dRel, so = i * nH
                    let qi = Array(q[qo..<(qo + nH * headDim)])
                    let ri = Array(r[ro..<(ro + nH * dRel)])
                    let si = Array(sink[so..<(so + nH)])
                    let gi = Array(gate[qo..<(qo + nH * headDim)])

                    let t1Lin = try run(1, qi, ri, si, gi, kv, posOffset + i, nil)
                    XCTAssertEqual(Array(outLin[qo..<(qo + nH * headDim)]), t1Lin,
                                   "batch8 linear T=\(tBatch) mode=\(mode) tok=\(i) must match T=1 bit-exact")

                    var path: [Int] = []
                    var cur = i
                    while cur >= 0 { path.append(cur); cur = cur == 0 ? -1 : (cur / 2) }
                    path.reverse()
                    var kSeq = Array(kRaw[0..<(posOffset * nKV * headDim)])
                    var vSeq = Array(vRaw[0..<(posOffset * nKV * headDim)])
                    for idx in path {
                        let s = (posOffset + idx) * nKV * headDim
                        kSeq.append(contentsOf: kRaw[s..<(s + nKV * headDim)])
                        vSeq.append(contentsOf: vRaw[s..<(s + nKV * headDim)])
                    }
                    let kvP = Self.buildKVCache(hal: hal, mode: mode, kRaw: kSeq, vRaw: vSeq,
                                                totalTokens: posOffset + path.count, ringCap: ringCap,
                                                slidingWindow: sw, nKV: nKV, headDim: headDim)
                    let t1Tree = try run(1, qi, ri, si, gi, kvP, posOffset + path.count - 1, nil)
                    XCTAssertEqual(Array(outTree[qo..<(qo + nH * headDim)]), t1Tree,
                                   "batch8 tree T=\(tBatch) mode=\(mode) node=\(i) must match T=1 bit-exact")
                }
            }
        }
    }
}
