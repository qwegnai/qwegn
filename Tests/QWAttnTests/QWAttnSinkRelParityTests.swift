// QWAttnSinkRelParityTests.swift — FP64 CPU reference parity for NAX sink+rel prefill
// and GQA-shared Split-KV decode/verify across f16, q8, and q4 KV cache modes.
import Foundation
import Metal
import XCTest
@testable import QWAttn
import QWAlloc
import QWCore
import QWHAL
import QWKV
import QWQuant

final class QWAttnSinkRelParityTests: XCTestCase {
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
        init(seed: UInt64) { state = seed }
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

    private static func bf16Bits(_ v: Float) -> UInt16 { QWBF16.fromFloatRNE(v) }
    private static func f32FromBF16(_ bits: UInt16) -> Float { Float(bitPattern: UInt32(bits) << 16) }
    private static func roundBF16(_ v: Double) -> Float { f32FromBF16(bf16Bits(Float(v))) }

    private static func synthBF16(count: Int, seed: UInt64, scale: Double) -> [Float] {
        var rng = SplitMix64(seed: seed)
        return (0..<count).map { _ in roundBF16(Double(rng.nextSigned(scale))) }
    }

    private static func makeBF16Tensor(_ hal: QWHal, _ vals: [Float], dims: [Int]) -> QWTensorValue {
        let bits = vals.map(bf16Bits)
        let bytes = max(16, bits.count * 2)
        let buf = hal.device.makeBuffer(length: bytes, options: .storageModeShared)!
        bits.withUnsafeBytes { raw in
            if let src = raw.baseAddress, !raw.isEmpty { memcpy(buf.contents(), src, raw.count) }
        }
        let region = QWMRegion(buffer: buf, base: 0, length: bytes)
        return QWTensorValue(region: region, offset: 0, shape: QWTensorShape(dims, .bf16))
    }

    private static func emptyBF16Tensor(_ hal: QWHal, dims: [Int]) -> QWTensorValue {
        let count = dims.reduce(1, *)
        let bytes = max(16, count * 2)
        let buf = hal.device.makeBuffer(length: bytes, options: .storageModeShared)!
        memset(buf.contents(), 0, bytes)
        let region = QWMRegion(buffer: buf, base: 0, length: bytes)
        return QWTensorValue(region: region, offset: 0, shape: QWTensorShape(dims, .bf16))
    }

    private static func readBF16(_ tv: QWTensorValue, count: Int) -> [Float] {
        let ptr = tv.region.buffer.contents().advanced(by: tv.region.base + tv.offset)
            .assumingMemoryBound(to: UInt16.self)
        return (0..<count).map { f32FromBF16(ptr[$0]) }
    }

    private static func buildKVCache(
        hal: QWHal,
        mode: QWKVMode,
        kRaw: [Float],
        vRaw: [Float],
        totalTokens: Int,
        ringCap: Int,
        slidingWindow: Int,
        nKV: Int,
        headDim: Int
    ) -> (kv: QWSinkRelKVCache, kDeq: [Float], vDeq: [Float]) {
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

        var kDeq = [Float](repeating: 0, count: totalTokens * nKV * headDim)
        var vDeq = [Float](repeating: 0, count: totalTokens * nKV * headDim)

        func writeStream(src: [Float], deq: inout [Float], payBuf: any MTLBuffer, metaBuf: any MTLBuffer) {
            for pos in 0..<totalTokens {
                let slot = slidingWindow > 0 ? (pos % ringCap) : pos
                for kvh in 0..<nKV {
                    let srcOff = (pos * nKV + kvh) * headDim
                    let rowHead = slot * nKV + kvh
                    let vals = (0..<headDim).map { Float(Float16(src[srcOff + $0])) }
                    switch mode {
                    case .f16:
                        let dst = payBuf.contents().assumingMemoryBound(to: UInt16.self)
                        for d in 0..<headDim {
                            dst[rowHead * headDim + d] = Float16(vals[d]).bitPattern
                            deq[srcOff + d] = vals[d]
                        }
                    case .q8:
                        let headMax = vals.map(abs).max() ?? 0
                        let sc16 = Float16(max(headMax / 127.0, 6.103515625e-5))
                        let invSc = Float(sc16) > 0 ? (1.0 / Float(sc16)) : 0.0
                        let dst = payBuf.contents().assumingMemoryBound(to: Int8.self)
                        metaBuf.contents().assumingMemoryBound(to: UInt16.self)[rowHead] = sc16.bitPattern
                        for d in 0..<headDim {
                            let q = Int((vals[d] * invSc).rounded(.toNearestOrEven))
                            let c = Int8(max(-127, min(127, q)))
                            dst[rowHead * headDim + d] = c
                            deq[srcOff + d] = Float(Float16(Float(c)) * sc16)
                        }
                    case .q4:
                        let headMin = vals.min() ?? 0
                        let headMax = vals.max() ?? 0
                        let mn16 = Float16(headMin)
                        let sc16 = Float16(max((headMax - Float(mn16)) / 15.0, 6.103515625e-5))
                        let invSc = 1.0 / max(Float(sc16), 1e-7)
                        let dst = payBuf.contents().assumingMemoryBound(to: UInt8.self)
                        let meta = metaBuf.contents().assumingMemoryBound(to: UInt16.self)
                        meta[rowHead * 2] = sc16.bitPattern
                        meta[rowHead * 2 + 1] = mn16.bitPattern
                        for p in 0..<(headDim / 2) {
                            let d0 = p * 2, d1 = p * 2 + 1
                            let q0 = max(0, min(15, Int(((vals[d0] - Float(mn16)) * invSc).rounded(.toNearestOrEven))))
                            let q1 = max(0, min(15, Int(((vals[d1] - Float(mn16)) * invSc).rounded(.toNearestOrEven))))
                            dst[rowHead * (headDim / 2) + p] = UInt8(q0 | (q1 << 4))
                            deq[srcOff + d0] = Float(Float16(Float(q0)) * sc16 + mn16)
                            deq[srcOff + d1] = Float(Float16(Float(q1)) * sc16 + mn16)
                        }
                    }
                }
            }
        }

        writeStream(src: kRaw, deq: &kDeq, payBuf: kBuf, metaBuf: kmBuf)
        writeStream(src: vRaw, deq: &vDeq, payBuf: vBuf, metaBuf: vmBuf)
        let kv = QWSinkRelKVCache(
            kCache: QWMRegion(buffer: kBuf, base: 0, length: payBytes),
            vCache: QWMRegion(buffer: vBuf, base: 0, length: payBytes),
            kMeta: mode == .f16 ? nil : QWMRegion(buffer: kmBuf, base: 0, length: metaBytes),
            vMeta: mode == .f16 ? nil : QWMRegion(buffer: vmBuf, base: 0, length: metaBytes),
            ringCap: ringCap,
            mode: mode
        )
        return (kv, kDeq, vDeq)
    }

    private static func cpuReferenceFP64(
        q: [Float], kDeq: [Float], vDeq: [Float],
        r: [Float]?, relProj: [Float]?, sink: [Float]?, gate: [Float]?,
        config: QWSinkRelConfig, T: Int, posOffset: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: T * config.nH * config.headDim)
        let scale = 1.0 / Double(config.headDim).squareRoot()
        let gqa = config.nH / config.nKV
        for t in 0..<T {
            let qPos = posOffset + t
            let minKey = (config.slidingWindow > 0 && qPos + 1 > config.slidingWindow)
                ? (qPos + 1 - config.slidingWindow) : 0
            for h in 0..<config.nH {
                let kvh = h / gqa
                var scores = [Double]()
                scores.reserveCapacity(qPos - minKey + 1)
                for j in minKey...qPos {
                    var dot = 0.0
                    let qOff = (t * config.nH + h) * config.headDim
                    let kOff = (j * config.nKV + kvh) * config.headDim
                    for d in 0..<config.headDim {
                        dot += Double(q[qOff + d]) * Double(kDeq[kOff + d])
                    }
                    var s = dot * scale
                    let dist = qPos - j
                    if config.hasRelBias, let r, let relProj, dist < config.relExtent {
                        let rOff = (t * config.nH + h) * config.dRel
                        for dr in 0..<config.dRel {
                            s += Double(r[rOff + dr]) * Double(relProj[dr * config.relExtent + dist])
                        }
                    }
                    scores.append(s)
                }
                let sSink = (config.hasSink ? sink.map { Double($0[t * config.nH + h]) } : nil)
                let maxKeyScore = scores.max() ?? -1e30
                let maxAll = sSink.map { max($0, maxKeyScore) } ?? maxKeyScore
                var sumW = sSink.map { exp($0 - maxAll) } ?? 0.0
                var acc = [Double](repeating: 0, count: config.headDim)
                for (idx, j) in (minKey...qPos).enumerated() {
                    let w = exp(scores[idx] - maxAll)
                    sumW += w
                    let vOff = (j * config.nKV + kvh) * config.headDim
                    for d in 0..<config.headDim {
                        acc[d] += w * Double(vDeq[vOff + d])
                    }
                }
                let oOff = (t * config.nH + h) * config.headDim
                for d in 0..<config.headDim {
                    var val = acc[d] / sumW
                    if config.hasOutGate, let gate {
                        let g = Double(gate[oOff + d])
                        val *= 1.0 / (1.0 + exp(-g))
                    }
                    out[oOff + d] = roundBF16(val)
                }
            }
        }
        return out
    }

    private func assertParity(
        _ got: [Float],
        _ want: [Float],
        absTol: Float,
        relTol: Float = 0.02,
        context: String
    ) {
        XCTAssertEqual(got.count, want.count, context)
        var worstAbs: Float = 0
        var worstIdx = -1
        for i in 0..<got.count {
            XCTAssertTrue(got[i].isFinite, "non-finite output at \(i) (\(context))")
            let diff = abs(got[i] - want[i])
            if diff > worstAbs { worstAbs = diff; worstIdx = i }
            let bound = max(absTol, abs(want[i]) * relTol)
            if diff > bound {
                XCTFail("\(context) mismatch at \(i): got=\(got[i]) want=\(want[i]) diff=\(diff) bound=\(bound)")
                return
            }
        }
        XCTAssertLessThanOrEqual(worstAbs, max(absTol, (want.map(abs).max() ?? 0) * relTol),
                                 "\(context) worstIdx=\(worstIdx)")
    }

    private func runPrefillCase(
        hal: QWHal,
        op: QWAttnSinkRel,
        mode: QWKVMode,
        nH: Int,
        nKV: Int,
        T: Int,
        posOffset: Int,
        dRel: Int,
        relExtent: Int,
        slidingWindow: Int,
        hasSink: Bool,
        hasRelBias: Bool,
        hasOutGate: Bool,
        seed: UInt64
    ) throws {
        let headDim = 128
        let totalKV = posOffset + T
        let ringCap = slidingWindow > 0 ? (slidingWindow + T) : totalKV
        let cfg = QWSinkRelConfig(
            nH: nH, nKV: nKV, headDim: headDim, dRel: dRel, relExtent: relExtent,
            slidingWindow: slidingWindow, hasSink: hasSink, hasRelBias: hasRelBias,
            hasOutGate: hasOutGate, kvMode: mode
        )
        let q = Self.synthBF16(count: T * nH * headDim, seed: seed &+ 1, scale: 1.5)
        let kRaw = Self.synthBF16(count: totalKV * nKV * headDim, seed: seed &+ 2, scale: 1.5)
        let vRaw = Self.synthBF16(count: totalKV * nKV * headDim, seed: seed &+ 3, scale: 0.8)
        let r = hasRelBias ? Self.synthBF16(count: T * nH * dRel, seed: seed &+ 4, scale: 1.2) : nil
        let relProj = hasRelBias ? Self.synthBF16(count: dRel * relExtent, seed: seed &+ 5, scale: 0.25) : nil
        let sink = hasSink ? Self.synthBF16(count: T * nH, seed: seed &+ 6, scale: 1.0) : nil
        let gate = hasOutGate ? Self.synthBF16(count: T * nH * headDim, seed: seed &+ 7, scale: 1.5) : nil

        let (kv, kDeq, vDeq) = Self.buildKVCache(
            hal: hal, mode: mode, kRaw: kRaw, vRaw: vRaw,
            totalTokens: totalKV, ringCap: ringCap, slidingWindow: slidingWindow, nKV: nKV, headDim: headDim
        )
        let qTV = Self.makeBF16Tensor(hal, q, dims: [T, nH, headDim])
        let rTV = r.map { Self.makeBF16Tensor(hal, $0, dims: [T, nH, dRel]) }
        let rpTV = relProj.map { Self.makeBF16Tensor(hal, $0, dims: [dRel, relExtent]) }
        let sTV = sink.map { Self.makeBF16Tensor(hal, $0, dims: [T, nH]) }
        let gTV = gate.map { Self.makeBF16Tensor(hal, $0, dims: [T, nH, headDim]) }
        let outTV = Self.emptyBF16Tensor(hal, dims: [T, nH, headDim])

        guard let (cb, enc) = hal.begin() else { XCTFail("begin failed"); return }
        try op.encodePrefill(
            enc, config: cfg, q: qTV, r: rTV, relProj: rpTV, sink: sTV, gate: gTV,
            kv: kv, out: outTV, T: T, posOffset: posOffset
        )
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed, "prefill failed: \(String(describing: cb.error))")

        let got = Self.readBF16(outTV, count: T * nH * headDim)
        let want = Self.cpuReferenceFP64(
            q: q, kDeq: kDeq, vDeq: vDeq, r: r, relProj: relProj, sink: sink, gate: gate,
            config: cfg, T: T, posOffset: posOffset
        )
        assertParity(
            got, want, absTol: 1.5e-2,
            context: "prefill mode=\(mode) T=\(T) pos=\(posOffset) sw=\(slidingWindow) ext=\(relExtent) sink=\(hasSink) rel=\(hasRelBias) gate=\(hasOutGate)"
        )
    }

    private func runDecodeCase(
        hal: QWHal,
        op: QWAttnSinkRel,
        mode: QWKVMode,
        nH: Int,
        nKV: Int,
        headDim: Int = 128,
        T: Int,
        posOffset: Int,
        dRel: Int,
        relExtent: Int,
        slidingWindow: Int,
        ringCap: Int,
        splits: Int,
        hasSink: Bool,
        hasRelBias: Bool,
        hasOutGate: Bool,
        seed: UInt64
    ) throws {
        let totalKV = posOffset + T
        let cfg = QWSinkRelConfig(
            nH: nH, nKV: nKV, headDim: headDim, dRel: dRel, relExtent: relExtent,
            slidingWindow: slidingWindow, hasSink: hasSink, hasRelBias: hasRelBias,
            hasOutGate: hasOutGate, kvMode: mode
        )
        let q = Self.synthBF16(count: T * nH * headDim, seed: seed &+ 1, scale: 1.5)
        let kRaw = Self.synthBF16(count: totalKV * nKV * headDim, seed: seed &+ 2, scale: 1.5)
        let vRaw = Self.synthBF16(count: totalKV * nKV * headDim, seed: seed &+ 3, scale: 0.8)
        let r = hasRelBias ? Self.synthBF16(count: T * nH * dRel, seed: seed &+ 4, scale: 1.2) : nil
        let relProj = hasRelBias ? Self.synthBF16(count: dRel * relExtent, seed: seed &+ 5, scale: 0.25) : nil
        let sink = hasSink ? Self.synthBF16(count: T * nH, seed: seed &+ 6, scale: 1.0) : nil
        let gate = hasOutGate ? Self.synthBF16(count: T * nH * headDim, seed: seed &+ 7, scale: 1.5) : nil

        let (kv, kDeq, vDeq) = Self.buildKVCache(
            hal: hal, mode: mode, kRaw: kRaw, vRaw: vRaw,
            totalTokens: totalKV, ringCap: ringCap, slidingWindow: slidingWindow, nKV: nKV, headDim: headDim
        )
        let qTV = Self.makeBF16Tensor(hal, q, dims: [T, nH, headDim])
        let rTV = r.map { Self.makeBF16Tensor(hal, $0, dims: [T, nH, dRel]) }
        let rpTV = relProj.map { Self.makeBF16Tensor(hal, $0, dims: [dRel, relExtent]) }
        let sTV = sink.map { Self.makeBF16Tensor(hal, $0, dims: [T, nH]) }
        let gTV = gate.map { Self.makeBF16Tensor(hal, $0, dims: [T, nH, headDim]) }
        let outTV = Self.emptyBF16Tensor(hal, dims: [T, nH, headDim])

        let pSizes = QWAttnSinkRel.partialsBytes(tokens: T, qHeads: nH, headDim: headDim, splits: splits)
        let pOBuf = hal.device.makeBuffer(length: max(16, pSizes.partOBytes), options: .storageModeShared)!
        let pMLBuf = hal.device.makeBuffer(length: max(16, pSizes.partMLBytes), options: .storageModeShared)!
        let partials = QWSinkRelPartials(
            partO: QWMRegion(buffer: pOBuf, base: 0, length: pOBuf.length),
            partML: QWMRegion(buffer: pMLBuf, base: 0, length: pMLBuf.length)
        )

        guard let (cb, enc) = hal.begin() else { XCTFail("begin failed"); return }
        try op.encodeDecode(
            enc, config: cfg, q: qTV, r: rTV, relProj: rpTV, sink: sTV, gate: gTV,
            kv: kv, partials: partials, out: outTV, T: T, posOffset: posOffset, splits: splits
        )
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed, "decode failed: \(String(describing: cb.error))")

        let got = Self.readBF16(outTV, count: T * nH * headDim)
        let want = Self.cpuReferenceFP64(
            q: q, kDeq: kDeq, vDeq: vDeq, r: r, relProj: relProj, sink: sink, gate: gate,
            config: cfg, T: T, posOffset: posOffset
        )
        assertParity(
            got, want, absTol: 1.5e-2,
            context: "decode mode=\(mode) M=\(T) pos=\(posOffset) sw=\(slidingWindow) splits=\(splits)"
        )
    }

    func testSplitsCalculationAndShapeValidation() throws {
        XCTAssertEqual(QWAttnSinkRel.splits(forKeys: 64, kvHeads: 8, slidingWindow: 0), 1)
        XCTAssertEqual(QWAttnSinkRel.splits(forKeys: 128, kvHeads: 8, slidingWindow: 1024), 1)
        let globalSplits = QWAttnSinkRel.splits(forKeys: 4096, kvHeads: 8, batchTokens: 1, slidingWindow: 0)
        XCTAssertGreaterThan(globalSplits, 1)
        XCTAssertEqual((globalSplits * 8) % 40, 0, "expected wave-aligned global split count")
        let swSplits = QWAttnSinkRel.splits(forKeys: 8192, kvHeads: 8, batchTokens: 1, slidingWindow: 1024)
        XCTAssertGreaterThan(swSplits, 1)
        XCTAssertEqual((swSplits * 8) % 40, 0, "expected wave-aligned sliding-window split count")

        let spBatch16 = QWAttnSinkRel.splits(forKeys: 4096, kvHeads: 8, batchTokens: 16, slidingWindow: 0)
        XCTAssertGreaterThan(spBatch16, 1)
        XCTAssertEqual((spBatch16 * 8) % 40, 0, "expected wave-aligned batch16 split count")

        let geometry = try Self.makeGeometry(qHeads: 12, kvHeads: 4, headDim: 128)
        let hal = try QWHal(geometry: geometry)
        let op = try QWAttnSinkRel(hal: hal)
        let qTV = Self.emptyBF16Tensor(hal, dims: [2, 12, 128])
        let outTV = Self.emptyBF16Tensor(hal, dims: [2, 12, 128])
        let (kv, _, _) = Self.buildKVCache(
            hal: hal, mode: .f16, kRaw: [Float](repeating: 0, count: 2 * 4 * 128),
            vRaw: [Float](repeating: 0, count: 2 * 4 * 128),
            totalTokens: 2, ringCap: 2, slidingWindow: 0, nKV: 4, headDim: 128
        )
        let badCfg = QWSinkRelConfig(nH: 12, nKV: 4, headDim: 128, hasSink: true, kvMode: .f16)
        guard let (cb, enc) = hal.begin() else { XCTFail("begin failed"); return }
        XCTAssertThrowsError(
            try op.encodeDecode(enc, config: badCfg, q: qTV, sink: nil, kv: kv, out: outTV, T: 2, posOffset: 0)
        )
        let tooManyRows = QWSinkRelConfig(nH: 12, nKV: 4, headDim: 128, kvMode: .f16)
        XCTAssertThrowsError(
            try op.encodeDecode(
                enc, config: tooManyRows, q: Self.emptyBF16Tensor(hal, dims: [17, 12, 128]),
                kv: kv, out: Self.emptyBF16Tensor(hal, dims: [17, 12, 128]), T: 17, posOffset: 0
            )
        )
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    func testPrefillParityShortAndMultiTileAcrossKVModes() throws {
        let geometry = try Self.makeGeometry(qHeads: 12, kvHeads: 4, headDim: 128)
        let hal = try QWHal(geometry: geometry)
        try XCTSkipUnless(
            hal.caps.supportsSimdgroupMMA,
            "Sink/rel MMA/NAX prefill requires Apple7+ hardware"
        )
        let op = try QWAttnSinkRel(hal: hal)

        for mode in [QWKVMode.f16, .q8, .q4] {
            try runPrefillCase(
                hal: hal, op: op, mode: mode, nH: 6, nKV: 2, T: 10, posOffset: 0,
                dRel: 16, relExtent: 32, slidingWindow: 0,
                hasSink: true, hasRelBias: true, hasOutGate: true, seed: 101
            )
            try runPrefillCase(
                hal: hal, op: op, mode: mode, nH: 12, nKV: 4, T: 65, posOffset: 35,
                dRel: 16, relExtent: 48, slidingWindow: 0,
                hasSink: true, hasRelBias: true, hasOutGate: true, seed: 202
            )
        }
    }

    func testPrefillRelSkipSlidingWindowAndOptionalFlags() throws {
        let geometry = try Self.makeGeometry(qHeads: 12, kvHeads: 4, headDim: 128)
        let hal = try QWHal(geometry: geometry)
        try XCTSkipUnless(
            hal.caps.supportsSimdgroupMMA,
            "Sink/rel MMA/NAX prefill requires Apple7+ hardware"
        )
        let op = try QWAttnSinkRel(hal: hal)

        // Exact relExtent skip boundary: minDist == relExtent (95 - 31 == 64) and minDist == relExtent - 1 (94 - 31 == 63).
        try runPrefillCase(
            hal: hal, op: op, mode: .f16, nH: 12, nKV: 4, T: 40, posOffset: 95,
            dRel: 16, relExtent: 64, slidingWindow: 0,
            hasSink: true, hasRelBias: true, hasOutGate: true, seed: 303
        )
        try runPrefillCase(
            hal: hal, op: op, mode: .f16, nH: 12, nKV: 4, T: 40, posOffset: 94,
            dRel: 16, relExtent: 64, slidingWindow: 0,
            hasSink: true, hasRelBias: true, hasOutGate: true, seed: 304
        )
        // Sliding window ring wrap + dRel=8 across all KV modes.
        for mode in [QWKVMode.f16, .q8, .q4] {
            try runPrefillCase(
                hal: hal, op: op, mode: mode, nH: 12, nKV: 4, T: 36, posOffset: 80,
                dRel: 8, relExtent: 24, slidingWindow: 24,
                hasSink: true, hasRelBias: true, hasOutGate: true, seed: 404
            )
        }
        // Optional features disabled and partial combinations.
        try runPrefillCase(
            hal: hal, op: op, mode: .f16, nH: 8, nKV: 2, T: 33, posOffset: 16,
            dRel: 0, relExtent: 0, slidingWindow: 0,
            hasSink: false, hasRelBias: false, hasOutGate: false, seed: 505
        )
        try runPrefillCase(
            hal: hal, op: op, mode: .q8, nH: 8, nKV: 2, T: 20, posOffset: 5,
            dRel: 0, relExtent: 0, slidingWindow: 12,
            hasSink: true, hasRelBias: false, hasOutGate: false, seed: 606
        )
    }

    func testDecodeAndVerifySplitKVParityAcrossKVModes() throws {
        let geometry = try Self.makeGeometry(qHeads: 16, kvHeads: 4, headDim: 128)
        let hal = try QWHal(geometry: geometry)
        let op = try QWAttnSinkRel(hal: hal)

        for mode in [QWKVMode.f16, .q8, .q4] {
            let sp1 = QWAttnSinkRel.splits(forKeys: 301, kvHeads: 4, batchTokens: 1, slidingWindow: 0)
            try runDecodeCase(
                hal: hal, op: op, mode: mode, nH: 12, nKV: 4, T: 1, posOffset: 300,
                dRel: 16, relExtent: 128, slidingWindow: 0, ringCap: 301, splits: sp1,
                hasSink: true, hasRelBias: true, hasOutGate: true, seed: 707
            )
            try runDecodeCase(
                hal: hal, op: op, mode: mode, nH: 12, nKV: 4, T: 1, posOffset: 48,
                dRel: 16, relExtent: 32, slidingWindow: 0, ringCap: 49, splits: 1,
                hasSink: false, hasRelBias: true, hasOutGate: false, seed: 808
            )
            let spSW = QWAttnSinkRel.splits(forKeys: 321, kvHeads: 4, batchTokens: 1, slidingWindow: 160)
            try runDecodeCase(
                hal: hal, op: op, mode: mode, nH: 12, nKV: 4, T: 1, posOffset: 320,
                dRel: 16, relExtent: 64, slidingWindow: 160, ringCap: 200, splits: spSW,
                hasSink: true, hasRelBias: true, hasOutGate: true, seed: 909
            )
            let spM8 = QWAttnSinkRel.splits(forKeys: 258, kvHeads: 4, batchTokens: 8, slidingWindow: 0)
            try runDecodeCase(
                hal: hal, op: op, mode: mode, nH: 16, nKV: 4, T: 8, posOffset: 250,
                dRel: 16, relExtent: 96, slidingWindow: 0, ringCap: 258, splits: spM8,
                hasSink: true, hasRelBias: true, hasOutGate: true, seed: 1010
            )
            let spM8SW = QWAttnSinkRel.splits(forKeys: 308, kvHeads: 4, batchTokens: 8, slidingWindow: 160)
            try runDecodeCase(
                hal: hal, op: op, mode: mode, nH: 16, nKV: 4, T: 8, posOffset: 300,
                dRel: 16, relExtent: 64, slidingWindow: 160, ringCap: 200, splits: spM8SW,
                hasSink: true, hasRelBias: true, hasOutGate: true, seed: 1111
            )
        }
        // GQA > 4 head chunking verification + T=2, 4, 16 linear batched decode.
        for tBatch in [2, 4, 16] {
            let sp = QWAttnSinkRel.splits(forKeys: 256 + tBatch, kvHeads: 4, batchTokens: tBatch, slidingWindow: 0)
            try runDecodeCase(
                hal: hal, op: op, mode: .f16, nH: 16, nKV: 4, T: tBatch, posOffset: 256,
                dRel: 16, relExtent: 64, slidingWindow: 0, ringCap: 256 + tBatch, splits: sp,
                hasSink: true, hasRelBias: true, hasOutGate: true, seed: UInt64(1300 + tBatch)
            )
        }
        let spG6 = QWAttnSinkRel.splits(forKeys: 208, kvHeads: 2, batchTokens: 16, slidingWindow: 0)
        try runDecodeCase(
            hal: hal, op: op, mode: .f16, nH: 12, nKV: 2, T: 16, posOffset: 192,
            dRel: 16, relExtent: 64, slidingWindow: 0, ringCap: 208, splits: spG6,
            hasSink: true, hasRelBias: true, hasOutGate: true, seed: 1212
        )
    }

    func testDecodeBatchedTreeParityAcrossKVModes() throws {
        let nH = 12, nKV = 4, headDim = 128, dRel = 16, relExtent = 64, T = 16, posOffset = 180
        let parents = [-1, 0, 0, 0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 10, 13]
        var depths = [UInt32](repeating: 0, count: T)
        var ancMask = [UInt32](repeating: 0, count: T)
        for i in 0..<T {
            if parents[i] < 0 {
                depths[i] = 0
                ancMask[i] = 1
            } else {
                let p = parents[i]
                depths[i] = depths[p] + 1
                ancMask[i] = ancMask[p] | (1 << UInt32(i))
            }
        }
        let meta = QWAttnSinkRel.TreeMeta(ancMask: ancMask, depths: depths)
        let totalKV = posOffset + T
        let geometry = try Self.makeGeometry(qHeads: nH, kvHeads: nKV, headDim: headDim)
        let hal = try QWHal(geometry: geometry)
        let op = try QWAttnSinkRel(hal: hal)

        for (mode, sliding, ringCap) in [(QWKVMode.f16, 0, totalKV), (.q8, 96, 128), (.q4, 0, totalKV)] {
            let cfg = QWSinkRelConfig(
                nH: nH, nKV: nKV, headDim: headDim, dRel: dRel, relExtent: relExtent,
                slidingWindow: sliding, hasSink: true, hasRelBias: true, hasOutGate: true, kvMode: mode
            )
            let q = Self.synthBF16(count: T * nH * headDim, seed: 4201, scale: 1.4)
            let kRaw = Self.synthBF16(count: totalKV * nKV * headDim, seed: 4202, scale: 1.4)
            let vRaw = Self.synthBF16(count: totalKV * nKV * headDim, seed: 4203, scale: 0.9)
            let r = Self.synthBF16(count: T * nH * dRel, seed: 4204, scale: 1.1)
            let relProj = Self.synthBF16(count: dRel * relExtent, seed: 4205, scale: 0.25)
            let sink = Self.synthBF16(count: T * nH, seed: 4206, scale: 1.0)
            let gate = Self.synthBF16(count: T * nH * headDim, seed: 4207, scale: 1.4)

            let (kv, kDeq, vDeq) = Self.buildKVCache(
                hal: hal, mode: mode, kRaw: kRaw, vRaw: vRaw,
                totalTokens: totalKV, ringCap: ringCap, slidingWindow: sliding, nKV: nKV, headDim: headDim
            )
            let qTV = Self.makeBF16Tensor(hal, q, dims: [T, nH, headDim])
            let rTV = Self.makeBF16Tensor(hal, r, dims: [T, nH, dRel])
            let rpTV = Self.makeBF16Tensor(hal, relProj, dims: [dRel, relExtent])
            let sTV = Self.makeBF16Tensor(hal, sink, dims: [T, nH])
            let gTV = Self.makeBF16Tensor(hal, gate, dims: [T, nH, headDim])
            let outTV = Self.emptyBF16Tensor(hal, dims: [T, nH, headDim])
            let splits = QWAttnSinkRel.splits(forKeys: totalKV, kvHeads: nKV, batchTokens: T, slidingWindow: sliding)
            let pSizes = QWAttnSinkRel.partialsBytes(tokens: T, qHeads: nH, headDim: headDim, splits: splits)
            let pOBuf = hal.device.makeBuffer(length: max(16, pSizes.partOBytes), options: .storageModeShared)!
            let pMLBuf = hal.device.makeBuffer(length: max(16, pSizes.partMLBytes), options: .storageModeShared)!
            let partials = QWSinkRelPartials(
                partO: QWMRegion(buffer: pOBuf, base: 0, length: pOBuf.length),
                partML: QWMRegion(buffer: pMLBuf, base: 0, length: pMLBuf.length)
            )
            guard let (cb, enc) = hal.begin() else { XCTFail("begin failed"); return }
            try op.encodeDecode(
                enc, config: cfg, q: qTV, r: rTV, relProj: rpTV, sink: sTV, gate: gTV,
                kv: kv, partials: partials, out: outTV, T: T, posOffset: posOffset,
                splits: splits, treeMeta: meta
            )
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            XCTAssertEqual(cb.status, .completed, "tree decode failed: \(String(describing: cb.error))")

            let got = Self.readBF16(outTV, count: T * nH * headDim)
            // Verify each tree node against an exact linearized single-step causal sequence containing only its ancestors.
            for node in 0..<T {
                var path: [Int] = []
                var cur = node
                while cur >= 0 { path.append(cur); cur = parents[cur] }
                path.reverse()
                let seqLen = posOffset + path.count
                var kSeq = Array(kDeq[0..<(posOffset * nKV * headDim)])
                var vSeq = Array(vDeq[0..<(posOffset * nKV * headDim)])
                for p in path {
                    let sK = (posOffset + p) * nKV * headDim
                    kSeq.append(contentsOf: kDeq[sK..<(sK + nKV * headDim)])
                    vSeq.append(contentsOf: vDeq[sK..<(sK + nKV * headDim)])
                }
                let qStart = node * nH * headDim
                let rStart = node * nH * dRel
                let sStart = node * nH
                let wantNode = Self.cpuReferenceFP64(
                    q: Array(q[qStart..<(qStart + nH * headDim)]),
                    kDeq: kSeq, vDeq: vSeq,
                    r: Array(r[rStart..<(rStart + nH * dRel)]),
                    relProj: relProj,
                    sink: Array(sink[sStart..<(sStart + nH)]),
                    gate: Array(gate[qStart..<(qStart + nH * headDim)]),
                    config: cfg, T: 1, posOffset: seqLen - 1
                )
                let gotNode = Array(got[qStart..<(qStart + nH * headDim)])
                assertParity(gotNode, wantNode, absTol: 1.5e-2, context: "tree mode=\(mode) sw=\(sliding) node=\(node)")
            }
        }
    }

    func testDecode8SGMatches1SGAndFP64Exact() throws {
        let nH = 24, nKV = 8, headDim = 128, dRel = 16, relExtent = 1024
        let geometry = try Self.makeGeometry(qHeads: nH, kvHeads: nKV, headDim: headDim)
        let hal = try QWHal(geometry: geometry)
        let op = try QWAttnSinkRel(hal: hal)

        for (posOffset, sw, sc, mode) in [
            (3, 1024, Double(1.5), QWKVMode.f16),
            (24, 1024, Double(5.0), QWKVMode.q4),
            (24, 0, Double(5.0), QWKVMode.q4),
            (180, 0, Double(4.0), QWKVMode.q8),
        ] {
            let totalKV = posOffset + 8
            let cfg = QWSinkRelConfig(
                nH: nH, nKV: nKV, headDim: headDim, dRel: dRel, relExtent: relExtent,
                slidingWindow: sw, hasSink: true, hasRelBias: true, hasOutGate: true, kvMode: mode
            )
            let q = Self.synthBF16(count: 8 * nH * headDim, seed: 5101, scale: sc)
            let kRaw = Self.synthBF16(count: totalKV * nKV * headDim, seed: 5102, scale: sc)
            let vRaw = Self.synthBF16(count: totalKV * nKV * headDim, seed: 5103, scale: 0.8)
            let r = Self.synthBF16(count: 8 * nH * dRel, seed: 5104, scale: 1.2)
            let relProj = Self.synthBF16(count: dRel * relExtent, seed: 5105, scale: 0.25)
            let sink = Self.synthBF16(count: 8 * nH, seed: 5106, scale: 1.0)
            let gate = Self.synthBF16(count: 8 * nH * headDim, seed: 5107, scale: 1.5)
            let (kv, kDeq, vDeq) = Self.buildKVCache(
                hal: hal, mode: mode, kRaw: kRaw, vRaw: vRaw,
                totalTokens: totalKV, ringCap: sw > 0 ? sw : totalKV, slidingWindow: sw, nKV: nKV, headDim: headDim
            )

            func evalDecode(T: Int, qSlice: [Float], rSlice: [Float], sSlice: [Float], gSlice: [Float], pos: Int) throws -> [Float] {
                let qTV = Self.makeBF16Tensor(hal, qSlice, dims: [T, nH, headDim])
                let rTV = Self.makeBF16Tensor(hal, rSlice, dims: [T, nH, dRel])
                let rpTV = Self.makeBF16Tensor(hal, relProj, dims: [dRel, relExtent])
                let sTV = Self.makeBF16Tensor(hal, sSlice, dims: [T, nH])
                let gTV = Self.makeBF16Tensor(hal, gSlice, dims: [T, nH, headDim])
                let outTV = Self.emptyBF16Tensor(hal, dims: [T, nH, headDim])
                let pSizes = QWAttnSinkRel.partialsBytes(tokens: T, qHeads: nH, headDim: headDim, splits: 1)
                let pOBuf = hal.device.makeBuffer(length: max(16, pSizes.partOBytes), options: .storageModeShared)!
                let pMLBuf = hal.device.makeBuffer(length: max(16, pSizes.partMLBytes), options: .storageModeShared)!
                let partials = QWSinkRelPartials(
                    partO: QWMRegion(buffer: pOBuf, base: 0, length: pOBuf.length),
                    partML: QWMRegion(buffer: pMLBuf, base: 0, length: pMLBuf.length)
                )
                guard let (cb, enc) = hal.begin() else { XCTFail("begin failed"); return [] }
                try op.encodeDecode(
                    enc, config: cfg, q: qTV, r: rTV, relProj: rpTV, sink: sTV, gate: gTV,
                    kv: kv, partials: partials, out: outTV, T: T, posOffset: pos, splits: 1
                )
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                return Self.readBF16(outTV, count: T * nH * headDim)
            }

            let out1 = try evalDecode(
                T: 1,
                qSlice: Array(q[0..<(nH * headDim)]), rSlice: Array(r[0..<(nH * dRel)]),
                sSlice: Array(sink[0..<nH]), gSlice: Array(gate[0..<(nH * headDim)]),
                pos: posOffset
            )
            let out8 = try evalDecode(T: 8, qSlice: q, rSlice: r, sSlice: sink, gSlice: gate, pos: posOffset)
            let want = Self.cpuReferenceFP64(
                q: Array(q[0..<(nH * headDim)]), kDeq: kDeq, vDeq: vDeq,
                r: Array(r[0..<(nH * dRel)]), relProj: relProj,
                sink: Array(sink[0..<nH]), gate: Array(gate[0..<(nH * headDim)]),
                config: cfg, T: 1, posOffset: posOffset
            )
            let row0Of8 = Array(out8[0..<(nH * headDim)])
            let max1Vs8 = zip(out1, row0Of8).map { abs($0 - $1) }.max() ?? 0
            XCTAssertEqual(max1Vs8, 0.0, "8-SG T=1 and 1-SG batched must be bit-identical at splits=1")
            assertParity(out1, want, absTol: 1.5e-2, context: "8SG vs FP64 mode=\(mode)")
            assertParity(out1, row0Of8, absTol: 1.5e-2, context: "8SG vs 1SG mode=\(mode)")
        }
    }

    func testDecodeMultiSplitT1MatchesBatchedAndTreeBitExact() throws {
        let nH = 12, nKV = 4, headDim = 128, dRel = 16, relExtent = 64, posOffset = 509, T = 8
        let parents = [-1, 0, 0, 1, 1, 2, 4]
        var depths = [UInt32](repeating: 0, count: parents.count)
        var ancMask = [UInt32](repeating: 0, count: parents.count)
        for i in 0..<parents.count {
            depths[i] = parents[i] < 0 ? 0 : (depths[parents[i]] + 1)
            ancMask[i] = (parents[i] < 0 ? 0 : ancMask[parents[i]]) | (UInt32(1) << i)
        }
        let treeMeta = QWAttnSinkRel.TreeMeta(ancMask: ancMask, depths: depths)
        let hal = try QWHal(geometry: try Self.makeGeometry(qHeads: nH, kvHeads: nKV, headDim: headDim))
        let op = try QWAttnSinkRel(hal: hal)
        for (sw, ringCap) in [(0, 2048), (256, 300)] {
            for mode in [QWKVMode.f16, .q8, .q4] {
                let cfg = QWSinkRelConfig(
                    nH: nH, nKV: nKV, headDim: headDim, dRel: dRel, relExtent: relExtent,
                    slidingWindow: sw, hasSink: true, hasRelBias: true, hasOutGate: true, kvMode: mode
                )
                let q = Self.synthBF16(count: T * nH * headDim, seed: 6101, scale: 3.5)
                let kRaw = Self.synthBF16(count: (posOffset + T) * nKV * headDim, seed: 6102, scale: 3.5)
                let vRaw = Self.synthBF16(count: (posOffset + T) * nKV * headDim, seed: 6103, scale: 0.9)
                let r = Self.synthBF16(count: T * nH * dRel, seed: 6104, scale: 1.2)
                let relProj = Self.synthBF16(count: dRel * relExtent, seed: 6105, scale: 0.25)
                let sink = Self.synthBF16(count: T * nH, seed: 6106, scale: 1.0)
                let gate = Self.synthBF16(count: T * nH * headDim, seed: 6107, scale: 1.4)
                let (kv, _, _) = Self.buildKVCache(
                    hal: hal, mode: mode, kRaw: kRaw, vRaw: vRaw,
                    totalTokens: posOffset + T, ringCap: ringCap, slidingWindow: sw, nKV: nKV, headDim: headDim
                )
                let rpTV = Self.makeBF16Tensor(hal, relProj, dims: [dRel, relExtent])
                let pSizes = QWAttnSinkRel.partialsBytes(tokens: T, qHeads: nH, headDim: headDim, splits: QWAttnSinkRel.maxSplits)
                let parts = QWSinkRelPartials(
                    partO: QWMRegion(buffer: hal.device.makeBuffer(length: pSizes.partOBytes, options: .storageModeShared)!, base: 0, length: pSizes.partOBytes),
                    partML: QWMRegion(buffer: hal.device.makeBuffer(length: pSizes.partMLBytes, options: .storageModeShared)!, base: 0, length: pSizes.partMLBytes)
                )
                func run(_ cnt: Int, _ qi: [Float], _ ri: [Float], _ si: [Float], _ gi: [Float], _ kvArg: QWSinkRelKVCache, _ pos: Int, _ tree: QWAttnSinkRel.TreeMeta?) throws -> [Float] {
                    let oTV = Self.emptyBF16Tensor(hal, dims: [cnt, nH, headDim])
                    guard let (cb, enc) = hal.begin() else { XCTFail("begin failed"); return [] }
                    try op.encodeDecode(
                        enc, config: cfg, q: Self.makeBF16Tensor(hal, qi, dims: [cnt, nH, headDim]),
                        r: Self.makeBF16Tensor(hal, ri, dims: [cnt, nH, dRel]), relProj: rpTV,
                        sink: Self.makeBF16Tensor(hal, si, dims: [cnt, nH]), gate: Self.makeBF16Tensor(hal, gi, dims: [cnt, nH, headDim]),
                        kv: kvArg, partials: parts, out: oTV, T: cnt, posOffset: pos, splits: nil, treeMeta: tree
                    )
                    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                    return Self.readBF16(oTV, count: cnt * nH * headDim)
                }
                let tCnt = parents.count
                let outLin = try run(T, q, r, sink, gate, kv, posOffset, nil)
                let outTree = try run(tCnt, Array(q[0..<(tCnt * nH * headDim)]), Array(r[0..<(tCnt * nH * dRel)]), Array(sink[0..<(tCnt * nH)]), Array(gate[0..<(tCnt * nH * headDim)]), kv, posOffset, treeMeta)
                for i in 0..<tCnt {
                    let qo = i * nH * headDim, ro = i * nH * dRel, so = i * nH
                    var path: [Int] = [], cur = i
                    while cur >= 0 { path.append(cur); cur = parents[cur] }
                    path.reverse()
                    var kSeq = Array(kRaw[0..<(posOffset * nKV * headDim)]), vSeq = Array(vRaw[0..<(posOffset * nKV * headDim)])
                    for idx in path {
                        let s = (posOffset + idx) * nKV * headDim
                        kSeq.append(contentsOf: kRaw[s..<(s + nKV * headDim)]); vSeq.append(contentsOf: vRaw[s..<(s + nKV * headDim)])
                    }
                    let (kvP, _, _) = Self.buildKVCache(hal: hal, mode: mode, kRaw: kSeq, vRaw: vSeq, totalTokens: posOffset + path.count, ringCap: ringCap, slidingWindow: sw, nKV: nKV, headDim: headDim)
                    let qi = Array(q[qo..<(qo + nH * headDim)]), ri = Array(r[ro..<(ro + nH * dRel)]), si = Array(sink[so..<(so + nH)]), gi = Array(gate[qo..<(qo + nH * headDim)])
                    let t1 = try run(1, qi, ri, si, gi, kvP, posOffset + path.count - 1, nil)
                    let t1L = try run(1, qi, ri, si, gi, kv, posOffset + i, nil)
                    XCTAssertEqual(zip(t1, Array(outTree[qo..<(qo + nH * headDim)])).map { abs($0 - $1) }.max() ?? 0, 0.0)
                    XCTAssertEqual(zip(t1L, Array(outLin[qo..<(qo + nH * headDim)])).map { abs($0 - $1) }.max() ?? 0, 0.0)
                }
            }
        }
    }
}
