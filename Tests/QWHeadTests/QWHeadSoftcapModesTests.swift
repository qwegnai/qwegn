// QWHeadSoftcapModesTests.swift — CPU-vs-Metal parity tests for `softcap = 30.0` across
// F16, BF16, HQ4_G32, HQ8/AQ8_G32, SHVQ4_G32, single/batched logits, and greedy argmax.
import Foundation
import Metal
import XCTest
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWHead
import QWQuant

final class QWHeadSoftcapModesTests: XCTestCase {
    private let hidden = 64
    private let vocab = 96
    private let cap: Float = 30.0

    private func skipUnlessMetal() throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true"
                || QWDeviceCaps.current?.familyApple9 != true,
            "requires bare-metal Apple Silicon GPU"
        )
    }

    private func makeGeometry() throws -> QWModelGeometry {
        let cfg = """
        {
          "model_type": "qwen3",
          "hidden_size": \(hidden),
          "intermediate_size": \(hidden * 2),
          "vocab_size": \(vocab),
          "num_hidden_layers": 1,
          "rms_norm_eps": 1e-6,
          "max_position_embeddings": 256,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 64,
          "rope_theta": 10000.0,
          "eos_token_id": 2
        }
        """.data(using: .utf8)!
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: cfg))
    }

    private func bf16Bits(_ value: Float) -> UInt16 {
        let u = value.bitPattern
        if (u & 0x7FFF_FFFF) > 0x7F80_0000 { return UInt16((u >> 16) | 0x0040) }
        let lsb = (u >> 16) & 1
        return UInt16(truncatingIfNeeded: (u &+ 0x7FFF &+ lsb) >> 16)
    }

    private func cpuSoftcap(_ v: Float, _ c: Float) -> Float {
        (c > 0) ? (tanh(min(max(v / c, -15.0), 15.0)) * c) : v
    }

    private func sortableU32(_ x: Float) -> UInt32 {
        let u = x.bitPattern
        if (u & 0x7FFF_FFFF) > 0x7F80_0000 { return 0 }
        return (u & 0x8000_0000) != 0 ? ~u : (u ^ 0x8000_0000)
    }

    private func cpuArgmax(_ row: [Float]) -> Int32 {
        var bestIdx = 0
        var bestHi = sortableU32(row[0])
        for i in 1..<row.count {
            let hi = sortableU32(row[i])
            if hi > bestHi {
                bestHi = hi
                bestIdx = i
            }
        }
        return Int32(bestIdx)
    }

    private func makeActivationRows(count m: Int, scale: Float = 1.0) -> [[Float16]] {
        var rows: [[Float16]] = []
        rows.reserveCapacity(m)
        for r in 0..<m {
            var row = [Float16](repeating: .zero, count: hidden)
            for c in 0..<hidden {
                let centered = (c * 7 + r * 13) % 29 - 14
                row[c] = Float16((Float(centered) / 8.0) * scale)
            }
            rows.append(row)
        }
        return rows
    }

    private func makeAQ8Bank() -> Data {
        var out = Data()
        for r in 0..<vocab {
            for g in 0..<(hidden / 32) {
                let vals = (0..<32).map { i in UInt32((i * 11 + r * 17 + g * 23) & 0xFF) }
                let sc: Float = (r % 9 == 0) ? 0.0625 : 0.03125
                let bi: Float = -128.0 * sc
                out.append(try! QWAQBlock.pack(bits: 8, group: 32, values: vals, scale: sc, bias: bi))
            }
        }
        return out
    }

    private func makeSHVQ4Bank() -> Data {
        let bBytes = QWSHVQBlock.bytes(bits: 4, group: 32)
        var out = Data(count: vocab * (hidden / 32) * bBytes)
        out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<vocab {
                for g in 0..<(hidden / 32) {
                    let off = (r * (hidden / 32) + g) * bBytes
                    let sc = Float16((r % 7 == 0) ? 1.75 : 0.65).bitPattern
                    ptr[off + 0] = UInt8(sc & 0xFF)
                    ptr[off + 1] = UInt8(sc >> 8)
                    for sub in 0..<8 {
                        let idx12 = UInt16((r * 37 + g * 53 + sub * 19) & 0x0FFF)
                        let signs4 = UInt16((r + g + sub) & 0x000F)
                        let u16 = idx12 | (signs4 << 12)
                        ptr[off + 2 + 2 * sub] = UInt8(u16 & 0xFF)
                        ptr[off + 2 + 2 * sub + 1] = UInt8(u16 >> 8)
                    }
                }
            }
        }
        return out
    }

    private func makeHQ4IncumbentBank() -> Data {
        let groups = hidden / 32
        var out = Data(count: vocab * groups * 20)
        out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let u16 = raw.baseAddress?.assumingMemoryBound(to: UInt16.self) else { return }
            for b in 0..<(vocab * groups) {
                let base = b * 10
                let sc16 = Float16(0.25 + Float(b % 7) * 0.08)
                let mnVal = -2.0 + Float(b % 5) * 0.3
                let bias16 = Float16(mnVal - 1024.0 * Float(sc16))
                u16[base + 0] = sc16.bitPattern
                u16[base + 1] = bias16.bitPattern
                for w in 2..<10 {
                    u16[base + w] = UInt16((b * 29 + w * 43) & 0xFFFF)
                }
            }
        }
        return out
    }

    func testF16AndBF16Softcap30SingleBatchedAndGreedyArgmax() throws {
        try skipUnlessMetal()
        let geo = try makeGeometry()
        let hal = try QWHal(geometry: geo)
        let head = try QWHead(hal: hal, geometry: geo)
        let maxM = 10
        let arena = try QWArena(hal: hal, weights: (2 << 20), activations: (2 << 20), kv: 256, gdnStates: 256)
        let wr = arena.weightsRegion()
        let ar = arena.activationsRegion()

        let wVals: [Float] = (0..<(vocab * hidden)).map { i in
            let r = i / hidden, c = i % hidden
            return sin(Float(r * 13 + c * 7 + 3) * 0.17) * ((r % 11 == 0) ? 4.5 : 1.3)
        }
        let wF16 = try wr.allocate(QWTensorShape([vocab * hidden], .f16))
        let wBF16 = try wr.allocate(QWTensorShape([vocab * hidden], .bf16))
        let pf16 = arena.cpuPointer(of: wF16).assumingMemoryBound(to: UInt16.self)
        let pbf16 = arena.cpuPointer(of: wBF16).assumingMemoryBound(to: UInt16.self)
        for i in 0..<wVals.count {
            pf16[i] = Float16(wVals[i]).bitPattern
            pbf16[i] = bf16Bits(wVals[i])
        }

        let xs = makeActivationRows(count: maxM, scale: 2.2)
        let xF16 = try ar.allocate(QWTensorShape([maxM * hidden], .f16))
        let px = arena.cpuPointer(of: xF16).assumingMemoryBound(to: Float16.self)
        for r in 0..<maxM {
            for c in 0..<hidden { px[r * hidden + c] = xs[r][c] }
        }
        let yGot = try ar.allocate(QWTensorShape([maxM * vocab], .f32))
        let slots = QWHead.batchedArgmaxSlots(vocab)
        let kh = try ar.allocate(QWTensorShape([maxM * slots], .i32))
        let kl = try ar.allocate(QWTensorShape([maxM * slots], .i32))
        let toks = try ar.allocate(QWTensorShape([maxM], .i32))

        for m in [1, 4, 10] {
            let xSlice = QWTensorValue(region: xF16.region, offset: xF16.offset, shape: QWTensorShape([m * hidden], .f16))
            let ySlice = QWTensorValue(region: yGot.region, offset: yGot.offset, shape: QWTensorShape([m * vocab], .f32))
            let tSlice = QWTensorValue(region: toks.region, offset: toks.offset, shape: QWTensorShape([m], .i32))

            for isBF16 in [false, true] {
                let storage: QWHeadStorage = isBF16 ? .bf16(wBF16) : .f16(wF16)
                guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
                try head.encode(enc, storage: storage, x: xSlice, y: ySlice, M: m, softcap: cap)
                try head.encodeBatchedGreedyArgmax(
                    enc, storage: storage, x: xSlice, keysHi: kh, keysLo: kl, tokens: tSlice, M: m, softcap: cap
                )
                hal.end(cb, enc)
                cb.waitUntilCompleted()
                XCTAssertNil(cb.error)

                let py = arena.cpuPointer(of: ySlice).assumingMemoryBound(to: Float.self)
                let pt = arena.cpuPointer(of: tSlice).assumingMemoryBound(to: Int32.self)
                for r in 0..<m {
                    var refRow = [Float](repeating: 0, count: vocab)
                    for v in 0..<vocab {
                        var dot: Float = 0
                        for c in 0..<hidden {
                            let wv: Float = isBF16
                                ? Float(bitPattern: UInt32(pbf16[v * hidden + c]) << 16)
                                : Float(Float16(bitPattern: pf16[v * hidden + c]))
                            dot += wv * Float(xs[r][c])
                        }
                        refRow[v] = cpuSoftcap(dot, cap)
                        XCTAssertLessThan(abs(py[r * vocab + v]), cap + 1e-4)
                        XCTAssertEqual(py[r * vocab + v], refRow[v], accuracy: 2e-3, "bf16=\(isBF16) m=\(m) r=\(r) v=\(v)")
                    }
                    XCTAssertEqual(pt[r], cpuArgmax(refRow), "argmax mismatch bf16=\(isBF16) m=\(m) r=\(r)")
                }
            }
        }
    }

    func testQ4GContainersHQ4AQ8AndSHVQ4Softcap30AndArgmax() throws {
        try skipUnlessMetal()
        let geo = try makeGeometry()
        let hal = try QWHal(geometry: geo)
        let head = try QWHead(hal: hal, geometry: geo)
        let maxM = 10
        let arena = try QWArena(hal: hal, weights: (2 << 20), activations: (2 << 20), kv: 256, gdnStates: 256)
        let wr = arena.weightsRegion()
        let ar = arena.activationsRegion()

        let hq4Data = makeHQ4IncumbentBank()
        let aq8Data = makeAQ8Bank()
        let shvqData = makeSHVQ4Bank()

        func uploadWeight(_ data: Data, bits: Int, group: Int, rotated: Bool, incumbent: Bool, shvq: Bool) throws -> QWHeadWeight {
            let tv = try wr.allocate(QWTensorShape([data.count / 2], .f16))
            arena.copyIn(tv, from: data)
            return try XCTUnwrap(QWHeadWeight(
                blocks: tv, n: vocab, k: hidden, bits: bits, groupSize: group,
                isRotated: rotated, isIncumbentContainer: incumbent, isSHVQ: shvq
            ))
        }

        let hq4W = try uploadWeight(hq4Data, bits: 4, group: 32, rotated: true, incumbent: true, shvq: false)
        let aq8W = try uploadWeight(aq8Data, bits: 8, group: 32, rotated: false, incumbent: false, shvq: false)
        let shvqW = try uploadWeight(shvqData, bits: 4, group: 32, rotated: true, incumbent: false, shvq: true)

        let xs = makeActivationRows(count: maxM, scale: 2.5)
        let xF16 = try ar.allocate(QWTensorShape([maxM * hidden], .f16))
        let px = arena.cpuPointer(of: xF16).assumingMemoryBound(to: Float16.self)
        for r in 0..<maxM {
            for c in 0..<hidden { px[r * hidden + c] = xs[r][c] }
        }
        let yRaw = try ar.allocate(QWTensorShape([maxM * vocab], .f32))
        let yCap = try ar.allocate(QWTensorShape([maxM * vocab], .f32))
        let yGreedy = try ar.allocate(QWTensorShape([vocab], .f32))
        let slots = QWHead.batchedArgmaxSlots(vocab)
        let kh = try ar.allocate(QWTensorShape([maxM * slots], .i32))
        let kl = try ar.allocate(QWTensorShape([maxM * slots], .i32))
        let toks = try ar.allocate(QWTensorShape([maxM], .i32))
        let tok1 = try ar.allocate(QWTensorShape([1], .i32))

        for (label, weight) in [("HQ4", hq4W), ("AQ8", aq8W), ("SHVQ4", shvqW)] {
            for m in [1, 4, 10] {
                let xSlice = QWTensorValue(region: xF16.region, offset: xF16.offset, shape: QWTensorShape([m * hidden], .f16))
                let yRawSlice = QWTensorValue(region: yRaw.region, offset: yRaw.offset, shape: QWTensorShape([m * vocab], .f32))
                let yCapSlice = QWTensorValue(region: yCap.region, offset: yCap.offset, shape: QWTensorShape([m * vocab], .f32))
                let tSlice = QWTensorValue(region: toks.region, offset: toks.offset, shape: QWTensorShape([m], .i32))

                guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
                try head.encode(enc, storage: .q4g(weight), x: xSlice, y: yRawSlice, M: m, softcap: 0.0)
                try head.encode(enc, storage: .q4g(weight), x: xSlice, y: yCapSlice, M: m, softcap: cap)
                try head.encodeBatchedGreedyArgmax(
                    enc, storage: .q4g(weight), x: xSlice, keysHi: kh, keysLo: kl, tokens: tSlice, M: m, softcap: cap
                )
                hal.end(cb, enc)
                cb.waitUntilCompleted()
                XCTAssertNil(cb.error)

                let pRaw = arena.cpuPointer(of: yRawSlice).assumingMemoryBound(to: Float.self)
                let pCap = arena.cpuPointer(of: yCapSlice).assumingMemoryBound(to: Float.self)
                let pTok = arena.cpuPointer(of: tSlice).assumingMemoryBound(to: Int32.self)

                for r in 0..<m {
                    var refRow = [Float](repeating: 0, count: vocab)
                    for v in 0..<vocab {
                        let expected = cpuSoftcap(pRaw[r * vocab + v], cap)
                        refRow[v] = expected
                        XCTAssertEqual(pCap[r * vocab + v], expected, accuracy: 2e-5, "\(label) m=\(m) r=\(r) v=\(v)")
                    }
                    XCTAssertEqual(pTok[r], cpuArgmax(refRow), "\(label) batched argmax m=\(m) r=\(r)")
                }
            }

            let x0 = QWTensorValue(region: xF16.region, offset: 0, shape: QWTensorShape([hidden], .f16))
            for writeLogits in [true, false] {
                memset(arena.cpuPointer(of: yGreedy), 0, vocab * 4)
                guard let (cbG, encG) = hal.begin() else { XCTFail("hal.begin"); return }
                try head.encodeGreedyQ4g(
                    encG, x: x0, w: weight, y: yGreedy, keysHi: kh, keysLo: kl,
                    token: tok1, writeLogits: writeLogits, softcap: cap
                )
                hal.end(cbG, encG)
                cbG.waitUntilCompleted()
                XCTAssertNil(cbG.error)

                let pRaw0 = arena.cpuPointer(of: yRaw).assumingMemoryBound(to: Float.self)
                let ref0 = (0..<vocab).map { cpuSoftcap(pRaw0[$0], cap) }
                let gotTok = arena.cpuPointer(of: tok1).assumingMemoryBound(to: Int32.self).pointee
                XCTAssertEqual(gotTok, cpuArgmax(ref0), "\(label) encodeGreedyQ4g writeLogits=\(writeLogits)")
                if writeLogits {
                    let pG = arena.cpuPointer(of: yGreedy).assumingMemoryBound(to: Float.self)
                    for v in 0..<vocab {
                        XCTAssertEqual(pG[v], ref0[v], accuracy: 2e-5)
                    }
                }
            }
        }
    }

    func testEncodeArgmaxSoftcap30SaturationTieBreakLowestId() throws {
        try skipUnlessMetal()
        let geo = try makeGeometry()
        let hal = try QWHal(geometry: geo)
        let head = try QWHead(hal: hal, geometry: geo)
        let n = 512
        let arena = try QWArena(hal: hal, weights: 256, activations: 64 << 10, kv: 256, gdnStates: 256)
        let ar = arena.activationsRegion()

        var vals = [Float](repeating: -10.0, count: n)
        vals[7] = 520.0
        vals[19] = 680.0
        vals[300] = 750.0
        let y = try ar.allocate(QWTensorShape([n], .f32))
        let py = arena.cpuPointer(of: y).assumingMemoryBound(to: Float.self)
        for i in 0..<n { py[i] = vals[i] }

        let nTG = QWHead.argmaxSlots(n)
        let kh = try ar.allocate(QWTensorShape([nTG], .i32))
        let kl = try ar.allocate(QWTensorShape([nTG], .i32))
        let tokUncapped = try ar.allocate(QWTensorShape([1], .i32))
        let tokCapped = try ar.allocate(QWTensorShape([1], .i32))

        guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
        try head.encodeArgmax(enc, y: y, keysHi: kh, keysLo: kl, token: tokUncapped, N: n, softcap: 0.0)
        try head.encodeArgmax(enc, y: y, keysHi: kh, keysLo: kl, token: tokCapped, N: n, softcap: cap)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let idUncapped = arena.cpuPointer(of: tokUncapped).assumingMemoryBound(to: Int32.self).pointee
        let idCapped = arena.cpuPointer(of: tokCapped).assumingMemoryBound(to: Int32.self).pointee
        XCTAssertEqual(idUncapped, 300, "Uncapped argmax must pick 750.0 at index 300")
        XCTAssertEqual(idCapped, 7, "Softcapped argmax must saturate 520/680/750 to 30.0 and tie-break to lowest index 7")
        XCTAssertEqual(py[19], 680.0, "encodeArgmax(softcap:) must not mutate caller logits in place")
    }
}
