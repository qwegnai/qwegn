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
}
