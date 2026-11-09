import Foundation
import XCTest
import QWCore
import QWQuant
import QWHAL
import QWAlloc
import QWGEMM
@testable import QWFlashNext

final class FNMetalMTPPrefillCacheTests: XCTestCase {
    func testCacheOnlyAttentionMatchesFullPrefill() throws {
        for mode: FNKVStorageMode in [.f16, .q4] {
            try checkCacheParity(mode: mode)
        }
    }

    private func checkCacheParity(mode: FNKVStorageMode) throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let ctx = try FNGPUContext(hal: hal, doors: [], geometry: .qwen38_27b)
        let arena = try QWArena(hal: hal, weights: 4 << 20, activations: 128 << 20, kv: 0, gdnStates: 0)
        let hidden = 32, rows = 16, capacity = 2064
        let heads = 16, kvHeads = 4, headDim = 256, indexHeads = 4, indexDim = 128
        let bankRows = heads * 2 * headDim + 2 * kvHeads * headDim + indexHeads * indexDim + indexDim
        let block = QWAQProducer.hq4G32Block((0..<32).map { Float($0 - 16) / 128 })
        func weight(_ n: Int, _ k: Int) throws -> QWWeightView {
            let count = n * (k / 32)
            var data = Data(capacity: count * block.count)
            for _ in 0..<count { data.append(block) }
            let tensor = try arena.weightsRegion().allocate(QWTensorShape([data.count / 2], .f16))
            arena.copyIn(tensor, from: data)
            return try XCTUnwrap(QWWeightView(blocks: tensor, n: n, k: k, isRotated: true))
        }
        let bank = try weight(bankRows, hidden)
        let outWeight = try weight(hidden, heads * headDim)
        let norm = try arena.activationsRegion().allocate(QWTensorShape([headDim], .f32))
        arena.write(norm, from: [Float](repeating: 1, count: headDim))
        let attention = FNGPUAttnMixer(
            bank: bank, oProj: outWeight, qNorm: norm, kNorm: norm, iqNorm: norm, ikNorm: norm,
            heads: heads, kvHeads: kvHeads, headDim: headDim, indexHeads: indexHeads,
            indexKVHeads: 1, indexHeadDim: indexDim, rotaryDim: 64, compress: 4,
            blockTopK: 512, bankRows: bankRows, rope: FNRope(rotaryDim: 64, theta: 10000)
        )
        let moe = FNGPUMoE(gateAndSharedBF16: norm, gateUp: bank, down: bank,
                           sharedGateUp: bank, sharedDown: bank, sharedGateScalar: norm,
                           experts: 1, topK: 1, intermediate: 32, sharedIntermediate: 32)
        func scratch() throws -> FNGPUScratch {
            let s = try FNGPUScratch(arena: arena, T: rows, hidden: hidden, hc: 1, lowrank: 32,
                                     moe: moe, ple: nil, gdn: nil, attn: attention, maxCapacity: capacity)
            let inputs = (0..<(rows * hidden)).map { Float(($0 % 31) - 15) / 32 }
            arena.write(s.rot32, from: inputs)
            return s
        }
        let fullScratch = try scratch(), cacheScratch = try scratch(), lastScratch = try scratch()
        let full = try FNGPUAttnDecodeState(arena: arena, mixer: attention, hidden: hidden,
                                            capacity: capacity, kvMode: mode)
        let cache = try FNGPUAttnDecodeState(arena: arena, mixer: attention, hidden: hidden,
                                             capacity: capacity, kvMode: mode)
        let lastOnly = try FNGPUAttnDecodeState(arena: arena, mixer: attention, hidden: hidden,
                                                capacity: capacity, kvMode: mode)
        for state in [full, cache, lastOnly] {
            for tensor in [state.k, state.v, state.kMeta, state.vMeta, state.indexK, state.pooled] {
                memset(arena.cpuPointer(of: tensor), 0, tensor.bytes)
            }
        }
        func run(_ state: FNGPUAttnDecodeState, _ scratch: FNGPUScratch,
                 position: Int, cacheOnly: Bool, lastRowOnly: Bool = false) throws -> Double {
            let pair = try XCTUnwrap(hal.begin(.concurrent))
            try FNGPULayerPasses.attentionPrefill(pair.1, ctx: ctx, a: attention, c: state,
                                                  s: scratch, pos: position, out: scratch.oG,
                                                  cacheOnly: cacheOnly, lastRowOnly: lastRowOnly)
            hal.end(pair.0, pair.1)
            pair.0.waitUntilCompleted()
            XCTAssertNil(pair.0.error)
            return (pair.0.gpuEndTime - pair.0.gpuStartTime) * 1000
        }
        func bytes(_ tensor: QWTensorValue) -> Data {
            Data(bytes: arena.cpuPointer(of: tensor), count: tensor.bytes)
        }
        for position in [0, 2048] {
            _ = try run(full, fullScratch, position: position, cacheOnly: false)
            _ = try run(cache, cacheScratch, position: position, cacheOnly: true)
            _ = try run(lastOnly, lastScratch, position: position, cacheOnly: false, lastRowOnly: true)
            for (reference, actual) in zip(
                [full.k, full.v, full.kMeta, full.vMeta, full.indexK, full.pooled],
                [cache.k, cache.v, cache.kMeta, cache.vMeta, cache.indexK, cache.pooled]
            ) {
                XCTAssertEqual(bytes(reference), bytes(actual), "cache mismatch: mode \(mode), pos \(position)")
            }
            for (reference, actual) in zip(
                [full.k, full.v, full.kMeta, full.vMeta, full.indexK, full.pooled],
                [lastOnly.k, lastOnly.v, lastOnly.kMeta, lastOnly.vMeta, lastOnly.indexK, lastOnly.pooled]
            ) {
                XCTAssertEqual(bytes(reference), bytes(actual), "last-row cache mismatch: mode \(mode), pos \(position)")
            }
            let fullLast = arena.read(fullScratch.oG, as: Float.self, count: rows * hidden).suffix(hidden)
            let lastOut = arena.read(lastScratch.oG, as: Float.self, count: rows * hidden).suffix(hidden)
            for (a, b) in zip(fullLast, lastOut) {
                XCTAssertEqual(a, b, accuracy: 1e-4, "last-row output mismatch: mode \(mode), pos \(position)")
            }
        }
        if ProcessInfo.processInfo.environment["QW_PERF_GATE"] == "1" {
            var fullTimes: [Double] = [], cacheTimes: [Double] = []
            for sample in 0..<5 {
                if sample.isMultiple(of: 2) {
                    fullTimes.append(try run(full, fullScratch, position: 0, cacheOnly: false))
                    cacheTimes.append(try run(cache, cacheScratch, position: 0, cacheOnly: true))
                } else {
                    cacheTimes.append(try run(cache, cacheScratch, position: 0, cacheOnly: true))
                    fullTimes.append(try run(full, fullScratch, position: 0, cacheOnly: false))
                }
            }
            print("MTP attention GPU ms for \(mode): full \(fullTimes), cache-only \(cacheTimes)")
        }
    }
}
