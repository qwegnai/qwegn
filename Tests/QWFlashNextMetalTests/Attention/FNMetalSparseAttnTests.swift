import XCTest
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
@testable import QWFlashNext

/// Sparse gated attention (one query, selected 4-token blocks plus the causal tail) at decode depth, against a CPU
/// reference of the same math: scores over the selected slots, softmax, value sum, sigmoid gate, 32-point WHT.
final class FNMetalSparseAttnTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    func testSparseGatedAttnDecodeDepth() throws { try check(1, 300) }
    func testSparseGatedAttnPrefillRows() throws { try check(4, 0) }
    func testSparseGatedAttnPrefillDeep() throws { try check(8, 300) }
    func testSparseGatedAttnKVModesParity() throws {
        try checkKVMode(.f16, rows: 8, maxRelErr: 2e-3)
        try checkKVMode(.q8, rows: 8, maxRelErr: 1.5e-2)
        try checkKVMode(.q4, rows: 8, maxRelErr: 8e-2)
        try checkKVMode(.f16, rows: 16, maxRelErr: 2e-3)
        try checkKVMode(.q8, rows: 16, maxRelErr: 1.5e-2)
        try checkKVMode(.q4, rows: 16, maxRelErr: 8e-2)
        try check(16, 300, maxRelErr: 1.5e-3)
    }

    private func checkKVMode(_ mode: FNKVStorageMode, rows T: Int = 8, maxRelErr: Double) throws {
        let nh = 16, nkv = 4, D = 256, ROT = 64
        let rowStride = nh * 2 * D + 2 * nkv * D
        let arena = try QWArena(hal: Self.hal, weights: 1 << 20, activations: 32 << 20, kv: 0, gdnStates: 0)
        let act = arena.activationsRegion()
        var seed: UInt32 = 777
        func rnd() -> Float {
            seed = seed &* 1664525 &+ 1013904223
            return Float(Int32(truncatingIfNeeded: seed)) * 1e-9 * 0.5
        }
        let srcV = (0..<(T * rowStride)).map { _ in rnd() }
        let qnV = (0..<D).map { _ in rnd() * 0.1 }
        let knV = (0..<D).map { _ in rnd() * 0.1 }
        let cosV = (0..<(T * ROT)).map { _ in Float(1.0) }
        let sinV = (0..<(T * ROT)).map { _ in Float(0.0) }
        let listV = (0..<(T * 512)).map { Int32($0 % 512) }

        let src = try act.allocate(QWTensorShape([T * rowStride], .f32))
        let qNorm = try act.allocate(QWTensorShape([D], .f32))
        let kNorm = try act.allocate(QWTensorShape([D], .f32))
        let cosT = try act.allocate(QWTensorShape([T * ROT], .f32))
        let sinT = try act.allocate(QWTensorShape([T * ROT], .f32))
        let list = try act.allocate(QWTensorShape([T * 512], .i32))
        arena.write(src, from: srcV); arena.write(qNorm, from: qnV); arena.write(kNorm, from: knV)
        arena.write(cosT, from: cosV); arena.write(sinT, from: sinV); arena.write(list, from: listV)

        func runMode(_ m: FNKVStorageMode) throws -> [Float] {
            let q = try act.allocate(QWTensorShape([T * nh * D], .f32))
            let g = try act.allocate(QWTensorShape([T * nh * D], .f32))
            let out = try act.allocate(QWTensorShape([T * nh * D], .f32))
            let k = try act.allocate(QWTensorShape([T * nkv * D], m == .f32 ? .f32 : .f16))
            let v = try act.allocate(QWTensorShape([T * nkv * D], m == .f32 ? .f32 : .f16))
            let kMeta = try act.allocate(QWTensorShape([T * nkv * 4], .f16))
            let vMeta = try act.allocate(QWTensorShape([T * nkv * 4], .f16))
            guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
            try QWGemm.encodeFnAttnPrepF32(enc, hal: Self.hal, src: src, qNorm: qNorm, kNorm: kNorm,
                                           cos: cosT, sin: sinT, q: q, k: k, v: v, gate: g,
                                           rows: T, heads: nh, kvHeads: nkv, rowStride: rowStride,
                                           kvMode: m, kMeta: kMeta, vMeta: vMeta)
            enc.memoryBarrier(scope: .buffers)
            try QWGemm.encodeFnAttnSparseGatedF32(enc, hal: Self.hal, q: q, k: k, v: v, gate: g,
                                                  list: list, out: out, rows: T, heads: nh, kvHeads: nkv, pos0: 0,
                                                  kvMode: m, kMeta: kMeta, vMeta: vMeta)
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
            return arena.read(out, as: Float.self, count: T * nh * D)
        }
        let ref = try runMode(.f32)
        let got = try runMode(mode)
        var sumSq = 0.0, refSq = 0.0
        for i in 0..<ref.count {
            let e = Double(got[i]) - Double(ref[i])
            sumSq += e * e; refSq += Double(ref[i]) * Double(ref[i])
        }
        let rel = (sumSq / max(refSq, 1e-30)).squareRoot()
        XCTAssertLessThan(rel, maxRelErr, "KV mode \(mode) rel err \(rel) exceeded \(maxRelErr)")
    }
    func testQSASelectLongContextParity() throws {
        // ~470 positive scores against a 512 budget: the threshold lands on the zero-score ties.
        try checkQSASelect(rows: 2, pos0: 4000)
        try checkQSASelect(rows: 16, pos0: 8000)
        try checkQSASelect(rows: 2, pos0: 8192)
        try checkQSASelect(rows: 2, pos0: 131072)
        try checkQSASelect(rows: 2, pos0: 262144)
    }

    private func checkQSASelect(rows T: Int, pos0: Int) throws {
        let NI = 4, DI = 128, topK = 512
        let maxBlocks = (pos0 + T) / 4
        let arena = try QWArena(hal: Self.hal, weights: 1 << 20, activations: 128 << 20, kv: 0, gdnStates: 0)
        let act = arena.activationsRegion()
        var seed: UInt32 = 424242
        func rnd() -> Float {
            seed = seed &* 1664525 &+ 1013904223
            return Float(Int32(truncatingIfNeeded: seed)) * 1e-9
        }
        let qs = (0..<(T * NI * DI)).map { _ in rnd() }
        var ps = (0..<(maxBlocks * DI)).map { _ in Float(Float16(rnd())) }
        // Force a cluster of zero-score ties so tie-breaking (prefer higher block index) is exercised at 128K+.
        for b in stride(from: 0, to: min(2048, maxBlocks), by: 2) {
            for d in 0..<DI { ps[b * DI + d] = 0 }
        }
        let ps16 = ps.map { Float16($0).bitPattern }
        let idxQ = try act.allocate(QWTensorShape([T * NI * DI], .f32))
        let pooled = try act.allocate(QWTensorShape([maxBlocks * DI], .f16))
        let list = try act.allocate(QWTensorShape([T * topK], .i32))
        let scores = try act.allocate(QWTensorShape([T * maxBlocks], .i32))
        arena.write(idxQ, from: qs)
        arena.write(pooled, from: ps16)

        guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
        try QWGemm.encodeFnQSASelect(enc, hal: Self.hal, idxQ: idxQ, pooled: pooled, list: list,
                                     rows: T, blockTopK: topK, pos0: pos0, scores: scores)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        let got = arena.read(list, as: Int32.self, count: T * topK)
        let invSqrt = 1.0 / Float(DI).squareRoot()
        for t in 0..<T {
            let n = (pos0 + t + 1) / 4
            var scored: [(Int, Float)] = []
            scored.reserveCapacity(n)
            for b in 0..<n {
                var score: Float = 0
                for h in 0..<NI {
                    var dot: Float = 0
                    for d in 0..<DI {
                        dot += qs[(t * NI + h) * DI + d] * ps[b * DI + d]
                    }
                    score += max(dot, 0)
                }
                scored.append((b, score * invSqrt))
            }
            scored.sort { $0.1 > $1.1 || ($0.1 == $1.1 && $0.0 > $1.0) }
            let want = scored.prefix(min(topK, n)).map { Int32($0.0) }.sorted()
            let gotRow = Array(got[(t * topK)..<((t + 1) * topK)])
            XCTAssertEqual(gotRow, want, "QSA select mismatch at pos0=\(pos0) row=\(t)")
        }
    }

    private func check(_ T: Int, _ pos0: Int, maxRelErr: Double = 1e-4) throws {
        let nh = 16, nkv = 4, D = 256
        let cache = pos0 + T
        let arena = try QWArena(hal: Self.hal, weights: 8 << 20, activations: (64 << 20), kv: 0, gdnStates: 0)
        let act = arena.activationsRegion()
        var seed: UInt32 = 31337
        func rnd() -> Float {
            seed = seed &* 1664525 &+ 1013904223
            return Float(Int32(truncatingIfNeeded: seed)) * 1e-9   // uniform in about [-2, 2]
        }
        let qs = (0..<(T * nh * D)).map { _ in rnd() * 0.5 }
        let ks = (0..<(cache * nkv * D)).map { _ in rnd() * 0.5 }
        let vs = (0..<(cache * nkv * D)).map { _ in rnd() * 0.5 }
        let gs = (0..<(T * nh * D)).map { _ in rnd() * 0.5 }
        var listV = [Int32](repeating: 0, count: T * 512)
        for t in 0..<T {
            let nbT = min(512, (pos0 + t + 1) / 4)
            for i in 0..<nbT { listV[t * 512 + i] = Int32(i) }
        }
        let q = try act.allocate(QWTensorShape([T * nh * D], .f32))
        let k = try act.allocate(QWTensorShape([cache * nkv * D], .f32))
        let v = try act.allocate(QWTensorShape([cache * nkv * D], .f32))
        let g = try act.allocate(QWTensorShape([T * nh * D], .f32))
        let list = try act.allocate(QWTensorShape([T * 512], .i32))
        let out = try act.allocate(QWTensorShape([T * nh * D], .f32))
        arena.copyIn(q, from: qs.withUnsafeBufferPointer { Data(buffer: $0) })
        arena.copyIn(k, from: ks.withUnsafeBufferPointer { Data(buffer: $0) })
        arena.copyIn(v, from: vs.withUnsafeBufferPointer { Data(buffer: $0) })
        arena.copyIn(g, from: gs.withUnsafeBufferPointer { Data(buffer: $0) })
        arena.copyIn(list, from: listV.withUnsafeBufferPointer { Data(buffer: $0) })

        guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
        try QWGemm.encodeFnAttnSparseGatedF32(enc, hal: Self.hal, q: q, k: k, v: v, gate: g, list: list, out: out,
                                              rows: T, heads: nh, kvHeads: nkv, pos0: pos0)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "attention failed: \(String(describing: cb.error))")
        let got = Array(UnsafeBufferPointer(start: arena.cpuPointer(of: out).bindMemory(to: Float.self, capacity: T * nh * D), count: T * nh * D))

        // CPU reference.
        let scale = 1.0 / Float(D).squareRoot()
        var ref = [Float](repeating: 0, count: T * nh * D)
        let BLOCK = 4
        for t in 0..<T {
            let pq = pos0 + t
            let nb = (pq + 1) / BLOCK
            let cnt = min(512, nb)
            let tailStart = nb * BLOCK
            let slots = cnt * BLOCK + (pq + 1 - tailStart)
            for h in 0..<nh {
                let gk = h / (nh / nkv)
                var scores = [Float](repeating: 0, count: slots)
                var js = [Int](repeating: 0, count: slots)
                for slot in 0..<slots {
                    let j = slot < cnt * BLOCK ? Int(listV[t * 512 + slot / BLOCK]) * BLOCK + slot % BLOCK : tailStart + (slot - cnt * BLOCK)
                    js[slot] = j
                    var dot: Float = 0
                    for d in 0..<D { dot += qs[(t * nh + h) * D + d] * ks[(j * nkv + gk) * D + d] }
                    scores[slot] = dot * scale
                }
                let mx = scores.max() ?? 0
                var sum: Float = 0
                var o = [Float](repeating: 0, count: D)
                for slot in 0..<slots {
                    let p = exp(scores[slot] - mx)
                    sum += p
                    for d in 0..<D { o[d] += p * vs[(js[slot] * nkv + gk) * D + d] }
                }
                for d in 0..<D {
                    ref[(t * nh + h) * D + d] = (o[d] / sum) / (1 + exp(-gs[(t * nh + h) * D + d]))
                }
                // 32-point WHT per block, then the output scale.
                for b in stride(from: 0, to: D, by: 32) {
                    var w = Array(ref[((t * nh + h) * D + b)..<((t * nh + h) * D + b + 32)])
                    var s = 1
                    while s < 32 {
                        for i in 0..<32 where (i & s) == 0 {
                            let a = w[i], c = w[i + s]
                            w[i] = a + c; w[i + s] = a - c
                        }
                        s <<= 1
                    }
                    for i in 0..<32 { ref[(t * nh + h) * D + b + i] = w[i] * 0.17677669529663689 }
                }
            }
        }
        var sumSq = 0.0, refSq = 0.0
        for i in 0..<ref.count { let e = Double(got[i]) - Double(ref[i]); sumSq += e * e; refSq += Double(ref[i]) * Double(ref[i]) }
        let rel = (sumSq / max(refSq, 1e-30)).squareRoot()
        print(String(format: "FN-SPARSE T=%d pos0=%d: rel err %.2e", T, pos0, rel))
        XCTAssertLessThan(rel, maxRelErr, "sparse gated attention must match the CPU reference (T=\(T), pos0=\(pos0))")
    }

    func testAttnDecodeStateSlotPartitionIsolation() throws {
        let cap = 1024, nh = 16, nkv = 4, D = 256, idxH = 4, idxKV = 1, idxD = 128, rot = 64
        let bankRows = nh * 2 * D + 2 * nkv * D + idxH * idxD + idxKV * idxD
        let arena = try QWArena(hal: Self.hal, weights: 4 << 20, activations: 64 << 20,
                                kv: 128 << 20, gdnStates: 0)
        let dummyW = try arena.weightsRegion().allocate(QWTensorShape([256], .f16))
        let dummyView = QWWeightView(blocks: dummyW, n: 16, k: 32, isRotated: true)!
        let dummyNorm = try arena.activationsRegion().allocate(QWTensorShape([D], .f32))
        let mixer = FNGPUAttnMixer(
            bank: dummyView, oProj: dummyView,
            qNorm: dummyNorm, kNorm: dummyNorm, iqNorm: dummyNorm, ikNorm: dummyNorm,
            heads: nh, kvHeads: nkv, headDim: D, indexHeads: idxH, indexKVHeads: idxKV, indexHeadDim: idxD,
            rotaryDim: rot, compress: 4, blockTopK: 512, bankRows: bankRows,
            rope: FNRope(rotaryDim: rot, theta: 10000.0)
        )
        let state = try FNGPUAttnDecodeState(arena: arena, mixer: mixer, hidden: 2560, capacity: cap, kvMode: .f16)
        XCTAssertEqual(state.fullCapacity, cap)
        XCTAssertEqual(state.capacity, cap)

        var kOffsets: [Int] = []
        var idxOffsets: [Int] = []
        var poolOffsets: [Int] = []
        for s in 0..<4 {
            state.selectPartition(slot: s, totalSlots: 4, mixer: mixer)
            XCTAssertEqual(state.capacity, cap / 4)
            kOffsets.append(state.k.offset)
            idxOffsets.append(state.indexK.offset)
            poolOffsets.append(state.pooled.offset)
        }
        for s in 1..<4 {
            XCTAssertGreaterThan(kOffsets[s], kOffsets[s - 1])
            XCTAssertGreaterThan(idxOffsets[s], idxOffsets[s - 1])
            XCTAssertGreaterThan(poolOffsets[s], poolOffsets[s - 1])
        }
        state.selectPartition(slot: 0, totalSlots: 1, mixer: mixer)
        XCTAssertEqual(state.capacity, cap)
        XCTAssertEqual(state.k.offset, kOffsets[0])
    }

    func testQSAEmeansClusterThreeNeedleRecallAndFootprint() throws {
        let totalTokens = 81_920
        let numPools = totalTokens / 4
        let needles = [1900, 9600, 17400]
        let NI = 4, DI = 128, topK = 512
        let arena = try QWArena(hal: Self.hal, weights: 4 << 20, activations: 128 << 20, kv: 64 << 20, gdnStates: 0)
        let act = arena.activationsRegion()
        let dummyW = try arena.weightsRegion().allocate(QWTensorShape([256], .f16))
        let dummyView = QWWeightView(blocks: dummyW, n: 16, k: 32, isRotated: true)!
        let dummyNorm = try act.allocate(QWTensorShape([256], .f32))
        let mixer = FNGPUAttnMixer(
            bank: dummyView, oProj: dummyView,
            qNorm: dummyNorm, kNorm: dummyNorm, iqNorm: dummyNorm, ikNorm: dummyNorm,
            heads: 16, kvHeads: 4, headDim: 256, indexHeads: NI, indexKVHeads: 1, indexHeadDim: DI,
            rotaryDim: 64, compress: 4, blockTopK: topK, bankRows: 16 * 512 + 8 * 256 + 5 * 128,
            rope: FNRope(rotaryDim: 64, theta: 10000.0))
        let dummyState = try FNGPUAttnDecodeState(arena: arena, mixer: mixer, hidden: 2560, capacity: 1024, kvMode: .q4)

        let wardReg = try FNGPUAttnDecodeState.allocateLSAClusters(
            hal: Self.hal, kvCap: 262_144, indexHeadDim: DI,
            states: Array(repeating: (dummyState, nil), count: 13), clusterBlock: 512)
        let emReg = try FNGPUAttnDecodeState.allocateLSAClusters(
            hal: Self.hal, kvCap: 262_144, indexHeadDim: DI,
            states: Array(repeating: (dummyState, nil), count: 13), clusterBlock: 4096)
        XCTAssertLessThan(Double(emReg.length), Double(wardReg.length) * 0.25,
                          "e-means QSA cluster region must shrink per-layer cluster RAM dramatically vs Ward")

        _ = try FNGPUAttnDecodeState.allocateLSAClusters(
            hal: Self.hal, kvCap: totalTokens, indexHeadDim: DI, states: [(dummyState, nil)], clusterBlock: 4096)
        let clusters = try XCTUnwrap(dummyState.lsaClusters)
        XCTAssertTrue(clusters.isEmeans)

        var rng: UInt64 = 0x55AA_9911
        func nextN() -> Float {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let u1 = max(1e-6, Float((rng >> 33) & 0x7FFF_FFFF) / Float(0x8000_0000))
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let u2 = Float((rng >> 33) & 0x7FFF_FFFF) / Float(0x8000_0000)
            return sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
        }
        var topics = [[Float]]()
        for _ in 0..<128 {
            var v = (0..<DI).map { _ in nextN() }
            let n = max(1e-6, sqrt(v.reduce(0) { $0 + $1 * $1 }))
            for d in 0..<DI { v[d] /= n }
            topics.append(v)
        }
        var ps16 = [UInt16](repeating: 0, count: numPools * DI)
        var p = 0
        while p < numPools {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let topic = topics[Int((rng >> 48) % 128)]
            let span = min(32, numPools - p)
            for s in 0..<span {
                var row = [Float](repeating: 0, count: DI)
                var sq: Float = 0
                for d in 0..<DI { let v = topic[d] + 0.35 * nextN(); row[d] = v; sq += v * v }
                let gain = Float(DI).squareRoot() / max(1e-6, sqrt(sq))
                for d in 0..<DI { ps16[(p + s) * DI + d] = Float16(row[d] * gain).bitPattern }
            }
            p += span
        }
        var qs = [Float](repeating: 0, count: 3 * NI * DI)
        for (r, np) in needles.enumerated() {
            var dir = (0..<DI).map { _ in nextN() }
            let n = max(1e-6, sqrt(dir.reduce(0) { $0 + $1 * $1 }))
            for d in 0..<DI { dir[d] /= n }
            for off in 0..<2 {
                var row = [Float](repeating: 0, count: DI)
                var sq: Float = 0
                for d in 0..<DI { let v = dir[d] + 0.03 * nextN(); row[d] = v; sq += v * v }
                let gain = Float(DI).squareRoot() / max(1e-6, sqrt(sq))
                for d in 0..<DI { ps16[(np + off) * DI + d] = Float16(row[d] * gain).bitPattern }
            }
            for d in 0..<DI { qs[(r * NI + 0) * DI + d] = dir[d] }
        }

        let idxQ = try act.allocate(QWTensorShape([3 * NI * DI], .f32))
        let pooled = try act.allocate(QWTensorShape([numPools * DI], .f16))
        let list = try act.allocate(QWTensorShape([3 * topK], .i32))
        let scores = try act.allocate(QWTensorShape([3 * numPools], .i32))
        arena.write(idxQ, from: qs)
        arena.write(pooled, from: ps16)

        guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
        let didCluster = try QWGemm.encodeFnLSAClusterRange(
            enc, hal: Self.hal, pooled: pooled, clusters: clusters, pos0: 0, count: totalTokens, indexDim: DI)
        XCTAssertTrue(didCluster)
        enc.memoryBarrier(scope: .buffers)
        try QWGemm.encodeFnQSASelect(
            enc, hal: Self.hal, idxQ: idxQ, pooled: pooled, list: list,
            rows: 3, blockTopK: topK, pos0: totalTokens - 3, scores: scores, lsaClusters: clusters)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let got = arena.read(list, as: Int32.self, count: 3 * topK)
        for (r, np) in needles.enumerated() {
            let rowSel = Array(got[(r * topK)..<((r + 1) * topK)])
            for i in 1..<rowSel.count {
                XCTAssertLessThan(rowSel[i - 1], rowSel[i], "QSA e-means list must be strictly ascending")
            }
            XCTAssertTrue(rowSel.contains(Int32(np)), "QSA e-means missed needle \(np) at row \(r)")
            XCTAssertTrue(rowSel.contains(Int32(np + 1)), "QSA e-means missed needle \(np + 1) at row \(r)")
        }
    }

    func testQSAPairedPerfAB() throws {
        guard ProcessInfo.processInfo.environment["QW_PERF_GATE"] == "1" else {
            throw XCTSkip("Set QW_PERF_GATE=1 to run paired QSA flat vs Ward vs e-means A/B")
        }
        let maxT = 1_048_576
        let maxPools = maxT / 4
        let NI = 4, DI = 128, topK = 512
        let arena = try QWArena(hal: Self.hal, weights: 4 << 20, activations: 256 << 20, kv: 64 << 20, gdnStates: 0)
        let act = arena.activationsRegion()
        let dummyW = try arena.weightsRegion().allocate(QWTensorShape([256], .f16))
        let dummyView = QWWeightView(blocks: dummyW, n: 16, k: 32, isRotated: true)!
        let dummyNorm = try act.allocate(QWTensorShape([256], .f32))
        let mixer = FNGPUAttnMixer(
            bank: dummyView, oProj: dummyView,
            qNorm: dummyNorm, kNorm: dummyNorm, iqNorm: dummyNorm, ikNorm: dummyNorm,
            heads: 16, kvHeads: 4, headDim: 256, indexHeads: NI, indexKVHeads: 1, indexHeadDim: DI,
            rotaryDim: 64, compress: 4, blockTopK: topK, bankRows: 16 * 512 + 8 * 256 + 5 * 128,
            rope: FNRope(rotaryDim: 64, theta: 10000.0))
        let stWard = try FNGPUAttnDecodeState(arena: arena, mixer: mixer, hidden: 2560, capacity: 1024, kvMode: .q4)
        let stEm = try FNGPUAttnDecodeState(arena: arena, mixer: mixer, hidden: 2560, capacity: 1024, kvMode: .q4)
        _ = try FNGPUAttnDecodeState.allocateLSAClusters(
            hal: Self.hal, kvCap: maxT, indexHeadDim: DI, states: [(stWard, nil)], clusterBlock: 512)
        _ = try FNGPUAttnDecodeState.allocateLSAClusters(
            hal: Self.hal, kvCap: maxT, indexHeadDim: DI, states: [(stEm, nil)], clusterBlock: 4096)
        let wardC = try XCTUnwrap(stWard.lsaClusters)
        let emC = try XCTUnwrap(stEm.lsaClusters)

        let idxQ = try act.allocate(QWTensorShape([32 * NI * DI], .f32))
        let pooled = try act.allocate(QWTensorShape([maxPools * DI], .f16))
        let list = try act.allocate(QWTensorShape([32 * topK], .i32))
        let scores = try act.allocate(QWTensorShape([32 * maxPools], .i32))
        var seed: UInt32 = 0x2468_ACE1
        let qSeed = (0..<(32 * NI * DI)).map { _ -> Float in
            seed = seed &* 1664525 &+ 1013904223
            return Float(Int32(bitPattern: seed)) * 4e-10
        }
        let pSeed = (0..<(maxPools * DI)).map { _ -> UInt16 in
            seed = seed &* 1664525 &+ 1013904223
            return Float16(Float(Int32(bitPattern: seed)) * 4e-10).bitPattern
        }
        arena.write(idxQ, from: qSeed)
        arena.write(pooled, from: pSeed)
        guard let (cbInit, encInit) = Self.hal.begin() else { return }
        _ = try QWGemm.encodeFnLSAClusterRange(encInit, hal: Self.hal, pooled: pooled, clusters: wardC, pos0: 0, count: maxT, indexDim: DI)
        _ = try QWGemm.encodeFnLSAClusterRange(encInit, hal: Self.hal, pooled: pooled, clusters: emC, pos0: 0, count: maxT, indexDim: DI)
        Self.hal.end(cbInit, encInit)
        cbInit.waitUntilCompleted()

        func gpuMs(iters: Int, _ body: (any MTLComputeCommandEncoder) throws -> Void) throws -> Double {
            guard let (cb, enc) = Self.hal.begin() else { return .nan }
            for _ in 0..<iters { try body(enc); enc.memoryBarrier(scope: .buffers) }
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            return (cb.gpuEndTime - cb.gpuStartTime) * 1e3 / Double(iters)
        }
        func med(_ xs: [Double]) -> Double { let s = xs.sorted(); return s[s.count / 2] }
        for T in [131_072, 262_144, 524_288, 1_048_576] {
            var wardD = [Double](), flatD = [Double](), emD = [Double]()
            for r in 0..<8 {
                if r % 2 == 0 {
                    wardD.append(try gpuMs(iters: 10) {
                        try QWGemm.encodeFnQSASelect($0, hal: Self.hal, idxQ: idxQ, pooled: pooled, list: list,
                                                     rows: 1, blockTopK: topK, pos0: T - 1, scores: scores, lsaClusters: wardC)
                    })
                    flatD.append(try gpuMs(iters: 10) {
                        try QWGemm.encodeFnQSASelect($0, hal: Self.hal, idxQ: idxQ, pooled: pooled, list: list,
                                                     rows: 1, blockTopK: topK, pos0: T - 1, scores: scores, lsaClusters: nil)
                    })
                    emD.append(try gpuMs(iters: 10) {
                        try QWGemm.encodeFnQSASelect($0, hal: Self.hal, idxQ: idxQ, pooled: pooled, list: list,
                                                     rows: 1, blockTopK: topK, pos0: T - 1, scores: scores, lsaClusters: emC)
                    })
                } else {
                    emD.append(try gpuMs(iters: 10) {
                        try QWGemm.encodeFnQSASelect($0, hal: Self.hal, idxQ: idxQ, pooled: pooled, list: list,
                                                     rows: 1, blockTopK: topK, pos0: T - 1, scores: scores, lsaClusters: emC)
                    })
                    flatD.append(try gpuMs(iters: 10) {
                        try QWGemm.encodeFnQSASelect($0, hal: Self.hal, idxQ: idxQ, pooled: pooled, list: list,
                                                     rows: 1, blockTopK: topK, pos0: T - 1, scores: scores, lsaClusters: nil)
                    })
                    wardD.append(try gpuMs(iters: 10) {
                        try QWGemm.encodeFnQSASelect($0, hal: Self.hal, idxQ: idxQ, pooled: pooled, list: list,
                                                     rows: 1, blockTopK: topK, pos0: T - 1, scores: scores, lsaClusters: wardC)
                    })
                }
            }
            print(String(format: "QSA-PAIRED T=%7d | ward=%.3f(p50=%.3f)ms flat=%.3f(p50=%.3f)ms emeans=%.3f(p50=%.3f)ms | %.2fx vs ward, %.2fx vs flat",
                         T, wardD.min()!, med(wardD), flatD.min()!, med(flatD), emD.min()!, med(emD),
                         wardD.min()! / emD.min()!, flatD.min()! / emD.min()!))
        }
    }
}
