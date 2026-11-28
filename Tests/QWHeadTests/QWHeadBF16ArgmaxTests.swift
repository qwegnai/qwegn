// QWHeadBF16ArgmaxTests.swift — Batched BF16 head GEMV & 2-stage deterministic argmax tests.
import Foundation
import Metal
import XCTest
import QWAlloc
import QWCore
import QWHAL
import QWHead

final class QWHeadBF16ArgmaxTests: XCTestCase {
    private func skipUnlessMetal() throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true"
                || QWDeviceCaps.current?.supportsSimdgroupMMA != true,
            "requires bare-metal Apple Silicon GPU"
        )
    }

    private func makeGeometry(hidden: Int, vocab: Int) -> QWModelGeometry {
        QWModelGeometry(descriptor: QWModelDescriptor(
            family: .qwen3Dense, modelType: "qwen3", capabilities: [],
            hidden: hidden, vocab: vocab, denseIntermediate: hidden * 2, numLayers: 1,
            layerTypes: ["full_attention"], fullAttentionInterval: 1, rmsEps: 1e-6,
            maxPositionEmbeddings: 256, qHeads: 2, kvHeads: 1, headDim: 64,
            attnOutputGate: false, partialRotaryFactor: 1.0, ropeTheta: 10000.0,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: nil, expertsPerTok: nil, moeIntermediate: nil,
            sharedExpertIntermediate: nil, expertGateType: nil,
            hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 0, gdnVHeads: 0, gdnHeadDim: 0,
            gdnQKVChannels: 0, gdnZChannels: 0, gdnConvKernel: 0,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 0, mtpUseDedicatedEmbeddings: false, eosTokenId: 2
        ))
    }

    private func bf16Bits(_ value: Float) -> UInt16 {
        let u = value.bitPattern
        if (u & 0x7FFF_FFFF) > 0x7F80_0000 {
            return UInt16((u >> 16) | 0x0040)
        }
        let lsb = (u >> 16) & 1
        return UInt16(truncatingIfNeeded: (u &+ 0x7FFF &+ lsb) >> 16)
    }

    private func bf16ToFloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    private func cpuSoftcap(_ x: Float, cap: Float) -> Float {
        (cap > 0) ? (tanh(min(max(x / cap, -15), 15)) * cap) : x
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

    private func runGPU(_ hal: QWHal, _ body: (any MTLComputeCommandEncoder) throws -> Void) throws {
        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        var ended = false
        defer { if !ended { enc.endEncoding() } }
        try body(enc)
        enc.endEncoding()
        ended = true
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed)
    }

    func testBatchedArgmaxSlotsContract() {
        for n in [1, 7, 8, 9, 15, 16, 17, 32, 33, 248_320] {
            let slots = QWHead.batchedArgmaxSlots(n)
            XCTAssertGreaterThanOrEqual(slots, (n + 7) / 8)
            XCTAssertGreaterThanOrEqual(slots, (n + 15) / 16)
            XCTAssertGreaterThanOrEqual(slots, (n + 31) / 32)
        }
    }

    func testBF16HeadProjectionParity() throws {
        try skipUnlessMetal()
        let hal = try QWHal(geometry: .qwen38_27b)
        let n = 37
        for hidden in [32, 36, 256, 2304] {
            let geo = makeGeometry(hidden: hidden, vocab: n)
            let maxM = 16
            let arena = try QWArena(
                hal: hal,
                weights: n * hidden * 2 + 512,
                activations: maxM * (hidden * 4 + n * 4) + 16384,
                kv: 64,
                gdnStates: 64
            )
            let w = try arena.weightsRegion().allocate(QWTensorShape([n * hidden], .bf16))
            let a = arena.activationsRegion()
            let xBF16 = try a.allocate(QWTensorShape([maxM * hidden], .bf16))
            let xF16 = try a.allocate(QWTensorShape([maxM * hidden], .f16))
            let y = try a.allocate(QWTensorShape([maxM * n], .f32))

            var wHost = [UInt16](repeating: 0, count: n * hidden)
            for r in 0..<n {
                for c in 0..<hidden {
                    let v = Float(((r * 13 + c * 7) % 23) - 11) * 0.03125
                    wHost[r * hidden + c] = bf16Bits(v)
                }
            }
            arena.write(w, from: wHost)

            var xBFHost = [UInt16](repeating: 0, count: maxM * hidden)
            var xF16Host = [Float16](repeating: 0, count: maxM * hidden)
            for m in 0..<maxM {
                for c in 0..<hidden {
                    let v = Float(((m * 17 + c * 5) % 19) - 9) * 0.0625
                    xBFHost[m * hidden + c] = bf16Bits(v)
                    xF16Host[m * hidden + c] = Float16(v)
                }
            }
            arena.write(xBF16, from: xBFHost)
            arena.write(xF16, from: xF16Host)

            let head = try QWHead(hal: hal, geometry: geo)
            for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16] {
                for useF16Input in [false, true] {
                    if hidden > 256 && m != 4 && m != 8 { continue }
                    for cap in [Float(0.0), Float(30.0)] {
                        let xm = (useF16Input ? xF16 : xBF16).slice(
                            byteOffset: 0, count: m * hidden, dtype: useF16Input ? .f16 : .bf16)
                        let ym = y.slice(byteOffset: 0, count: m * n, dtype: .f32)
                        try runGPU(hal) { enc in
                            if cap == 0 && !useF16Input && m == 4 {
                                try head.encode(enc, storage: .bf16(w), x: xm, y: ym, M: m, softcap: cap)
                            } else {
                                try QWHead.encodeBF16(enc, hal: hal, w: w, x: xm, y: ym,
                                                      geometry: geo, M: m, softcap: cap)
                            }
                        }
                        let gpu = arena.read(ym, as: Float.self, count: m * n)
                        for b in 0..<m {
                            for r in 0..<n {
                                var acc: Float = 0
                                for c in 0..<hidden {
                                    let xv = useF16Input
                                        ? Float(xF16Host[b * hidden + c])
                                        : bf16ToFloat(xBFHost[b * hidden + c])
                                    acc += bf16ToFloat(wHost[r * hidden + c]) * xv
                                }
                                let expected = cpuSoftcap(acc, cap: cap)
                                XCTAssertEqual(gpu[b * n + r], expected, accuracy: 2e-2,
                                               "H=\(hidden) M=\(m) b=\(b) r=\(r) f16=\(useF16Input) cap=\(cap)")
                            }
                        }
                    }
                }
            }
        }
    }

    func testBF16BatchedGreedyArgmaxParityAndTieBreak() throws {
        try skipUnlessMetal()
        let hal = try QWHal(geometry: .qwen38_27b)
        let n = 65
        for hidden in [36, 256] {
            let geo = makeGeometry(hidden: hidden, vocab: n)
            let maxM = 16
            let slots = QWHead.batchedArgmaxSlots(n)
            let arena = try QWArena(
                hal: hal,
                weights: n * hidden * 2 + 512,
                activations: maxM * (hidden * 4 + slots * 8 + 128) + 16384,
                kv: 64,
                gdnStates: 64
            )
            let w = try arena.weightsRegion().allocate(QWTensorShape([n * hidden], .bf16))
            let a = arena.activationsRegion()
            let xBF16 = try a.allocate(QWTensorShape([maxM * hidden], .bf16))
            let xF16 = try a.allocate(QWTensorShape([maxM * hidden], .f16))
            let keysHi = try a.allocate(QWTensorShape([maxM * slots], .i32))
            let keysLo = try a.allocate(QWTensorShape([maxM * slots], .i32))
            let tokens = try a.allocate(QWTensorShape([maxM], .i32))

            // Craft ties at indices (5, 19, 41), negative winners, and large values for softcap saturation.
            var wHost = [UInt16](repeating: 0, count: n * hidden)
            for r in 0..<n {
                let scale: Float
                if r == 5 || r == 19 || r == 41 {
                    scale = 2.0
                } else if r == 7 || r == 33 {
                    scale = 4.0
                } else if r == 11 || r == 29 {
                    scale = -1.0
                } else {
                    scale = -2.5
                }
                wHost[r * hidden] = bf16Bits(scale)
                for c in 1..<hidden {
                    wHost[r * hidden + c] = bf16Bits(Float((r + c) % 5 - 2) * 0.015625)
                }
            }
            arena.write(w, from: wHost)

            var xBFHost = [UInt16](repeating: 0, count: maxM * hidden)
            var xF16Host = [Float16](repeating: 0, count: maxM * hidden)
            for m in 0..<maxM {
                // Alternate positive/negative leading element to probe tie-breaks, zero ties, and saturation.
                let lead: Float = (m % 3 == 0) ? 1.0 : ((m % 3 == 1) ? -0.5 : 8.0)
                xBFHost[m * hidden] = bf16Bits(lead)
                xF16Host[m * hidden] = Float16(lead)
            }
            arena.write(xBF16, from: xBFHost)
            arena.write(xF16, from: xF16Host)

            for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16] {
                for useF16Input in [false, true] {
                    for cap in [Float(0.0), Float(0.05)] {
                        let xm = (useF16Input ? xF16 : xBF16).slice(
                            byteOffset: 0, count: m * hidden, dtype: useF16Input ? .f16 : .bf16)
                        let tokSlice = tokens.slice(byteOffset: 0, count: m, dtype: .i32)
                        try runGPU(hal) { enc in
                            try QWHead.encodeBatchedGreedyArgmax(
                                enc, hal: hal, storage: .bf16(w), x: xm,
                                keysHi: keysHi, keysLo: keysLo, tokens: tokSlice,
                                geometry: geo, M: m, softcap: cap
                            )
                        }
                        let got = arena.read(tokSlice, as: Int32.self, count: m)
                        for b in 0..<m {
                            var rowLogits = [Float](repeating: 0, count: n)
                            for r in 0..<n {
                                var acc: Float = 0
                                for c in 0..<hidden {
                                    let xv = useF16Input
                                        ? Float(xF16Host[b * hidden + c])
                                        : bf16ToFloat(xBFHost[b * hidden + c])
                                    acc += bf16ToFloat(wHost[r * hidden + c]) * xv
                                }
                                rowLogits[r] = cpuSoftcap(acc, cap: cap)
                            }
                            XCTAssertEqual(got[b], cpuArgmax(rowLogits),
                                           "argmax mismatch H=\(hidden) M=\(m) b=\(b) f16=\(useF16Input) cap=\(cap)")
                        }
                    }
                }
            }
        }
    }

    func testF16AndQ4GBatchedGreedyArgmaxParity() throws {
        try skipUnlessMetal()
        let hal = try QWHal(geometry: .qwen38_27b)
        let n = 49
        let hidden = 256
        let g32 = hidden / 32
        let geo = makeGeometry(hidden: hidden, vocab: n)
        let maxM = 16
        let slots = QWHead.batchedArgmaxSlots(n)
        let arena = try QWArena(
            hal: hal,
            weights: (n * hidden * 2) + (n * g32 * 20) + 1024,
            activations: maxM * (hidden * 2 + n * 4 + slots * 8) + 4096,
            kv: 64,
            gdnStates: 64
        )
        let wF16 = try arena.weightsRegion().allocate(QWTensorShape([n * hidden], .f16))
        let wBlk = try arena.weightsRegion().allocate(QWTensorShape([n * g32 * 10], .f16))
        let a = arena.activationsRegion()
        let xF16 = try a.allocate(QWTensorShape([maxM * hidden], .f16))
        let yRef = try a.allocate(QWTensorShape([maxM * n], .f32))
        let keysHi = try a.allocate(QWTensorShape([maxM * slots], .i32))
        let keysLo = try a.allocate(QWTensorShape([maxM * slots], .i32))
        let tokens = try a.allocate(QWTensorShape([maxM], .i32))

        var wF16Host = [Float16](repeating: 0, count: n * hidden)
        for r in 0..<n {
            let lead: Float = (r == 3 || r == 22 || r == 48) ? 2.5 : Float((r % 9) - 4) * 0.5
            wF16Host[r * hidden] = Float16(lead)
            for c in 1..<hidden {
                wF16Host[r * hidden + c] = Float16(Float((r * 7 + c * 3) % 11 - 5) * 0.03125)
            }
        }
        arena.write(wF16, from: wF16Host)

        var blkHost = [UInt16](repeating: 0, count: n * g32 * 10)
        for r in 0..<n {
            for g in 0..<g32 {
                let base = (r * g32 + g) * 10
                blkHost[base] = Float16(0.125).bitPattern
                blkHost[base + 1] = Float16(-0.5).bitPattern
                let nib = UInt16(((r + g) & 0xF) * 0x1111)
                for j in 2..<10 { blkHost[base + j] = nib ^ UInt16(j &* 0x0101) }
            }
        }
        arena.write(wBlk, from: blkHost)

        var xHost = [Float16](repeating: 0, count: maxM * hidden)
        for m in 0..<maxM {
            xHost[m * hidden] = Float16((m % 2 == 0) ? 1.5 : -1.0)
            for c in 1..<hidden {
                xHost[m * hidden + c] = Float16(Float((m + c) % 7 - 3) * 0.125)
            }
        }
        arena.write(xF16, from: xHost)

        let hqWeight = try XCTUnwrap(QWHeadWeight(
            blocks: wBlk, n: n, k: hidden, bits: 4, groupSize: 32,
            isRotated: true, isIncumbentContainer: true
        ))
        let aqWeight = try XCTUnwrap(QWHeadWeight(
            blocks: wBlk, n: n, k: hidden, bits: 4, groupSize: 32,
            isRotated: false, isIncumbentContainer: false
        ))
        let storages: [(String, QWHeadStorage)] = [
            ("f16", .f16(wF16)),
            ("hq4", .q4g(hqWeight)),
            ("aq4", .q4g(aqWeight)),
        ]

        let head = try QWHead(hal: hal, geometry: geo)
        for (label, storage) in storages {
            var m1Cap025 = [Int32](repeating: 0, count: maxM)
            let tokOne = tokens.slice(byteOffset: 0, count: 1, dtype: .i32)
            for b in 0..<maxM {
                let xOne = xF16.slice(byteOffset: b * hidden * 2, count: hidden, dtype: .f16)
                try runGPU(hal) { enc in
                    try QWHead.encodeBatchedGreedyArgmax(
                        enc, hal: hal, storage: storage, x: xOne,
                        keysHi: keysHi, keysLo: keysLo, tokens: tokOne,
                        geometry: geo, M: 1, softcap: 0.25
                    )
                }
                m1Cap025[b] = arena.read(tokOne, as: Int32.self, count: 1)[0]
            }
            for m in [1, 4, 5, 8, 9, 12, 16] {
                let xm = xF16.slice(byteOffset: 0, count: m * hidden, dtype: .f16)
                let ym = yRef.slice(byteOffset: 0, count: m * n, dtype: .f32)
                let tokSlice = tokens.slice(byteOffset: 0, count: m, dtype: .i32)

                try runGPU(hal) { enc in
                    try head.encode(enc, storage: storage, x: xm, y: ym, M: m)
                }
                let refLogits = arena.read(ym, as: Float.self, count: m * n)

                for cap in [Float(0.0), Float(0.5), Float(0.05)] {
                    try runGPU(hal) { enc in
                        try QWHead.encodeBatchedGreedyArgmax(
                            enc, hal: hal, storage: storage, x: xm,
                            keysHi: keysHi, keysLo: keysLo, tokens: tokSlice,
                            geometry: geo, M: m, softcap: cap
                        )
                    }
                    let got = arena.read(tokSlice, as: Int32.self, count: m)
                    for b in 0..<m {
                        let slice = (0..<n).map { cpuSoftcap(refLogits[b * n + $0], cap: cap) }
                        XCTAssertEqual(got[b], cpuArgmax(slice), "\(label) M=\(m) b=\(b) cap=\(cap)")
                    }
                }

                try runGPU(hal) { enc in
                    try QWHead.encodeBatchedGreedyArgmax(
                        enc, hal: hal, storage: storage, x: xm,
                        keysHi: keysHi, keysLo: keysLo, tokens: tokSlice,
                        geometry: geo, M: m, softcap: 0.25
                    )
                }
                let gotCap025 = arena.read(tokSlice, as: Int32.self, count: m)
                for b in 0..<m {
                    XCTAssertEqual(gotCap025[b], m1Cap025[b], "\(label) M=\(m) b=\(b) cap=0.25 vs M=1")
                }
            }
        }
    }

    func testBF16ValidationRejections() throws {
        try skipUnlessMetal()
        let hal = try QWHal(geometry: .qwen38_27b)
        let n = 16, hidden = 32
        let geo = makeGeometry(hidden: hidden, vocab: n)
        let arena = try QWArena(hal: hal, weights: n * hidden * 4 + 256, activations: 8192, kv: 64, gdnStates: 64)
        let wBF16 = try arena.weightsRegion().allocate(QWTensorShape([n * hidden], .bf16))
        let wF16 = try arena.weightsRegion().allocate(QWTensorShape([n * hidden], .f16))
        let a = arena.activationsRegion()
        let xBuf = try a.allocate(QWTensorShape([hidden + 8], .bf16))
        let xBF16 = xBuf.slice(byteOffset: 0, count: hidden, dtype: .bf16)
        let xI32 = try a.allocate(QWTensorShape([hidden], .i32))
        let yF32 = try a.allocate(QWTensorShape([n], .f32))
        let badKeys = try a.allocate(QWTensorShape([1], .f32))
        let keysHi = try a.allocate(QWTensorShape([QWHead.batchedArgmaxSlots(n)], .i32))
        let keysLo = try a.allocate(QWTensorShape([QWHead.batchedArgmaxSlots(n)], .i32))
        let tok = try a.allocate(QWTensorShape([1], .i32))

        try runGPU(hal) { enc in
            XCTAssertThrowsError(try QWHead.encodeBF16(enc, hal: hal, w: wF16, x: xBF16, y: yF32, geometry: geo))
            XCTAssertThrowsError(try QWHead.encodeBF16(enc, hal: hal, w: wBF16, x: xI32, y: yF32, geometry: geo))
            XCTAssertThrowsError(try QWHead.encodeBF16(enc, hal: hal, w: wBF16, x: xBF16, y: yF32, geometry: geo, M: 17))
            XCTAssertThrowsError(try QWHead.encodeBF16(enc, hal: hal, w: wBF16, x: xBF16, y: yF32, geometry: geo, N: n + 1))
            XCTAssertThrowsError(try QWHead.encodeBatchedGreedyArgmax(
                enc, hal: hal, storage: .bf16(wBF16), x: xBF16,
                keysHi: badKeys, keysLo: keysLo, tokens: tok, geometry: geo
            ))
            let unaligned = xBuf.slice(byteOffset: 2, count: hidden, dtype: .bf16)
            XCTAssertThrowsError(try QWHead.encodeBatchedGreedyArgmax(
                enc, hal: hal, storage: .bf16(wBF16), x: unaligned,
                keysHi: keysHi, keysLo: keysLo, tokens: tok, geometry: geo
            ))
        }
    }

    func testSHVQ4BatchedGreedyArgmaxParity() throws {
        try skipUnlessMetal()
        let hal = try QWHal(geometry: .qwen38_27b)
        let n = 67, hidden = 64, g32 = hidden / 32, maxM = 16
        let geo = makeGeometry(hidden: hidden, vocab: n)
        let slots = max(QWHead.batchedArgmaxSlots(n), (n + 31) / 32)
        let arena = try QWArena(
            hal: hal, weights: n * g32 * 18 + 1024,
            activations: maxM * (hidden * 2 + n * 4 + slots * 8) + 4096,
            kv: 64, gdnStates: 64
        )
        let wBlk = try arena.weightsRegion().allocate(QWTensorShape([n * g32 * 9], .f16))
        let a = arena.activationsRegion()
        let xF16 = try a.allocate(QWTensorShape([maxM * hidden], .f16))
        let yRef = try a.allocate(QWTensorShape([maxM * n], .f32))
        let keysHi = try a.allocate(QWTensorShape([maxM * slots], .i32))
        let keysLo = try a.allocate(QWTensorShape([maxM * slots], .i32))
        let tokens = try a.allocate(QWTensorShape([maxM], .i32))

        var blkHost = [UInt16](repeating: 0, count: n * g32 * 9)
        for r in 0..<n {
            for g in 0..<g32 {
                let base = (r * g32 + g) * 9
                blkHost[base] = Float16((r == 19) ? 0.5 : (0.0625 * Float((r % 5) + 1))).bitPattern
                for j in 1..<9 { blkHost[base + j] = UInt16(truncatingIfNeeded: (r &* 31 &+ g &* 7 &+ j) &* 0x101) }
            }
        }
        arena.write(wBlk, from: blkHost)
        var xHost = [Float16](repeating: 0, count: maxM * hidden)
        for m in 0..<maxM {
            for c in 0..<hidden { xHost[m * hidden + c] = Float16(Float((m * 3 + c) % 9 - 4) * 0.125) }
        }
        arena.write(xF16, from: xHost)

        let shvqWeight = try XCTUnwrap(QWHeadWeight(
            blocks: wBlk, n: n, k: hidden, bits: 4, groupSize: 32,
            isRotated: true, isIncumbentContainer: false, isSHVQ: true
        ))
        let storage = QWHeadStorage.q4g(shvqWeight)
        let head = try QWHead(hal: hal, geometry: geo)
        for m in [1, 4, 5, 8, 9, 12, 16] {
            let xm = xF16.slice(byteOffset: 0, count: m * hidden, dtype: .f16)
            let ym = yRef.slice(byteOffset: 0, count: m * n, dtype: .f32)
            let tokSlice = tokens.slice(byteOffset: 0, count: m, dtype: .i32)
            try runGPU(hal) { enc in try head.encode(enc, storage: storage, x: xm, y: ym, M: m) }
            let refLogits = arena.read(ym, as: Float.self, count: m * n)
            for cap in [Float(0.0), Float(30.0)] {
                try runGPU(hal) { enc in
                    try QWHead.encodeBatchedGreedyArgmax(
                        enc, hal: hal, storage: storage, x: xm,
                        keysHi: keysHi, keysLo: keysLo, tokens: tokSlice,
                        geometry: geo, M: m, softcap: cap
                    )
                }
                let got = arena.read(tokSlice, as: Int32.self, count: m)
                for b in 0..<m {
                    let slice = (0..<n).map { cpuSoftcap(refLogits[b * n + $0], cap: cap) }
                    XCTAssertEqual(got[b], cpuArgmax(slice), "shvq4 M=\(m) b=\(b) cap=\(cap)")
                }
            }
        }
    }
}
