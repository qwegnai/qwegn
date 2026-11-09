import Foundation
import XCTest
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
@testable import QWFlashNext

final class FNMetalPLEBatchTests: XCTestCase {
    func testStridedKeyNormAndGateMatchPerSlot() throws {
        for batch in [1, 2, 4] {
            try check(batch: batch, hidden: 256)
            try check(batch: batch, hidden: 2560)
        }
    }

    private func check(batch: Int, hidden: Int) throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let arena = try QWArena(hal: hal, weights: 1 << 20, activations: 64 << 20, kv: 0, gdnStates: 0)
        let act = arena.activationsRegion(), hc = 4, width = hidden * hc, stride = width + hidden
        func f32(_ count: Int) throws -> QWTensorValue {
            try act.allocate(QWTensorShape([count], .f32))
        }
        let kv = try f32(batch * stride), query = try f32(batch * width), weight = try f32(width)
        let refKey = try f32(batch * width), outKey = try f32(batch * width)
        let refGate = try f32(batch * width), outGate = try f32(batch * width)
        let kvValues = (0..<(batch * stride)).map { Float(($0 % 127) - 63) / 64 }
        let queryValues = (0..<(batch * width)).map { Float(($0 % 97) - 48) / 64 }
        let weights = (0..<width).map { Float(($0 % 17) - 8) / 128 }
        arena.write(kv, from: kvValues)
        arena.write(query, from: queryValues)
        arena.write(weight, from: weights)
        func run(strided: Bool) throws -> (gpu: Double, encode: Double) {
            let started = DispatchTime.now().uptimeNanoseconds
            let pair = try XCTUnwrap(hal.begin(.concurrent))
            let enc = pair.1, key = strided ? outKey : refKey, gate = strided ? outGate : refGate
            if strided {
                try QWGemm.encodeFnHCNorm(enc, hal: hal, x: kv, weight: weight, xn: key,
                                          rows: batch, width: width, hidden: hidden, eps: 1e-5, xStride: stride)
            } else {
                for row in 0..<batch {
                    let input = kv.slice(byteOffset: row * stride * 4, count: width, dtype: .f32)
                    let output = key.slice(byteOffset: row * width * 4, count: width, dtype: .f32)
                    try QWGemm.encodeFnHCNorm(enc, hal: hal, x: input, weight: weight, xn: output,
                                              rows: 1, width: width, hidden: hidden, eps: 1e-5)
                }
            }
            enc.memoryBarrier(scope: .buffers)
            if strided {
                let values = kv.slice(byteOffset: width * 4, count: (batch - 1) * stride + hidden, dtype: .f32)
                try QWGemm.encodeFnPLEGate(enc, hal: hal, keyNorm: key, queryNorm: query, vals: values, gv: gate,
                                           rows: batch, hidden: hidden, hc: hc, valStride: stride)
            } else {
                for row in 0..<batch {
                    let values = kv.slice(byteOffset: (row * stride + width) * 4, count: hidden, dtype: .f32)
                    let kn = key.slice(byteOffset: row * width * 4, count: width, dtype: .f32)
                    let qn = query.slice(byteOffset: row * width * 4, count: width, dtype: .f32)
                    let out = gate.slice(byteOffset: row * width * 4, count: width, dtype: .f32)
                    try QWGemm.encodeFnPLEGate(enc, hal: hal, keyNorm: kn, queryNorm: qn, vals: values, gv: out,
                                               rows: 1, hidden: hidden, hc: hc)
                }
            }
            let encodeMs = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
            hal.end(pair.0, enc)
            pair.0.waitUntilCompleted()
            XCTAssertNil(pair.0.error)
            return ((pair.0.gpuEndTime - pair.0.gpuStartTime) * 1000, encodeMs)
        }
        _ = try run(strided: false)
        _ = try run(strided: true)
        let keys = arena.read(refKey, as: Float.self, count: batch * width)
        let gates = arena.read(refGate, as: Float.self, count: batch * width)
        XCTAssertEqual(keys, arena.read(outKey, as: Float.self, count: batch * width))
        XCTAssertEqual(gates, arena.read(outGate, as: Float.self, count: batch * width))
        for row in 0..<batch {
            for group in 0..<hc {
                let offset = row * width + group * hidden
                let input = row * stride + group * hidden
                let sumSquares = (0..<hidden).reduce(0.0) { $0 + pow(Double(kvValues[input + $1]), 2) }
                let inverse = 1 / sqrt(sumSquares / Double(hidden) + 1e-5)
                var dot = 0.0
                for d in 0..<hidden {
                    let expected = Double(kvValues[input + d]) * inverse * (1 + Double(weights[group * hidden + d]))
                    XCTAssertEqual(Double(keys[offset + d]), expected, accuracy: 3e-5)
                    dot += Double(keys[offset + d]) * Double(queryValues[offset + d])
                }
                let scaled = dot / sqrt(Double(hidden))
                let transformed = (scaled > 0 ? 1.0 : (scaled < 0 ? -1.0 : 0.0)) * sqrt(max(abs(scaled), 1e-6))
                let sigmoid = 1 / (1 + exp(-transformed))
                for d in 0..<hidden {
                    let expected = sigmoid * Double(kvValues[row * stride + width + d])
                    XCTAssertEqual(Double(gates[offset + d]), expected, accuracy: 3e-5)
                }
            }
        }
        let dummyBlocks = try arena.weightsRegion().allocate(QWTensorShape([10], .f16))
        let dummy = try XCTUnwrap(QWWeightView(blocks: dummyBlocks, n: 1, k: 32, isRotated: true))
        let ple = FNGPUPLE(keyValueBank: dummy, keyBank: dummy, valueBank: dummy,
                           normKey: weight, normQuery: weight, normConv: weight, convW: weight,
                           embedDim: 1, kernel: 1, dilation: 1, eps: 1e-5, history: [])
        let moe = FNGPUMoE(gateAndSharedBF16: weight, gateUp: dummy, down: dummy,
                           sharedGateUp: dummy, sharedDown: dummy, sharedGateScalar: weight,
                           experts: 1, topK: 1, intermediate: 1, sharedIntermediate: 1)
        for strided in [false, true] {
            let scratch = try FNGPUScratch(arena: arena, T: batch, hidden: hidden, hc: hc, lowrank: 1,
                                           moe: moe, ple: nil, gdn: nil, attn: nil)
            arena.write(scratch.pleKV, from: kvValues)
            arena.write(scratch.pleQueryN, from: queryValues)
            let pair = try XCTUnwrap(hal.begin(.concurrent))
            try FNGPULayerPasses.pleBatchKeyGate(pair.1, hal: hal, p: ple, s: scratch,
                                                rows: batch, strided: strided)
            hal.end(pair.0, pair.1)
            pair.0.waitUntilCompleted()
            XCTAssertNil(pair.0.error)
            XCTAssertEqual(keys, arena.read(scratch.pleKeyN, as: Float.self, count: batch * width))
            XCTAssertEqual(gates, arena.read(scratch.pleGV, as: Float.self, count: batch * width))
        }
        if ProcessInfo.processInfo.environment["QW_PERF_GATE"] == "1", batch == 4, hidden == 2560 {
            var baselineGPU: [Double] = [], candidateGPU: [Double] = []
            var baselineEncode: [Double] = [], candidateEncode: [Double] = []
            for sample in 0..<10 {
                let baseline: (gpu: Double, encode: Double), candidate: (gpu: Double, encode: Double)
                if sample.isMultiple(of: 2) {
                    baseline = try run(strided: false)
                    candidate = try run(strided: true)
                } else {
                    candidate = try run(strided: true)
                    baseline = try run(strided: false)
                }
                baselineGPU.append(baseline.gpu); candidateGPU.append(candidate.gpu)
                baselineEncode.append(baseline.encode); candidateEncode.append(candidate.encode)
            }
            print("Batch PLE GPU ms: per-slot \(baselineGPU), strided \(candidateGPU)")
            print("Batch PLE CPU encode ms: per-slot \(baselineEncode), strided \(candidateEncode)")
        }
    }
}
