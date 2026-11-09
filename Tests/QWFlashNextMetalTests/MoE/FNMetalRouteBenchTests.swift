import XCTest
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWFlashNext

/// Route kernel alone: 48 dispatches (one per layer) in one command buffer, GPU time per dispatch.
final class FNMetalRouteBenchTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    func testRouteDispatchCost() throws {
        let E = 512, K = 10, layers = 48
        let arena = try QWArena(hal: Self.hal, weights: 1 << 20, activations: 4 << 20, kv: 0, gdnStates: 0)
        let act = arena.activationsRegion()
        let logits = try act.allocate(QWTensorShape([E], .f32))
        let idx = try act.allocate(QWTensorShape([K], .i32))
        let wts = try act.allocate(QWTensorShape([K], .f32))
        var seed: UInt32 = 7
        let lv = (0..<E).map { _ -> Float in seed = seed &* 1664525 &+ 1013904223; return Float(Int32(truncatingIfNeeded: seed)) * 1e-9 }
        arena.copyIn(logits, from: lv.withUnsafeBufferPointer { Data(buffer: $0) })
        var best = Double.infinity
        for _ in 0..<15 {
            guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
            for _ in 0..<layers {
                try QWGemm.encodeFnMoeRoute(enc, hal: Self.hal, logits: logits, idx: idx, wts: wts, rows: 1,
                                            experts: E, topK: K, normTopK: true)
            }
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000
            best = min(best, ms)
        }
        print(String(format: "FN-ROUTE-BENCH min of 15: 48 dispatches gpu %.3f ms = %.1f us per dispatch", best, best * 1000 / Double(layers)))
    }

    func testTrivialDispatchFloor() throws {
        // A near-empty kernel 48 times: the per-dispatch floor on this GPU for comparison.
        let arena = try QWArena(hal: Self.hal, weights: 1 << 20, activations: 4 << 20, kv: 0, gdnStates: 0)
        let act = arena.activationsRegion()
        let x = try act.allocate(QWTensorShape([2560], .f32))
        let y = try act.allocate(QWTensorShape([2560], .f32))
        var best = Double.infinity
        for _ in 0..<15 {
            guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
            for _ in 0..<48 {
                try QWGemm.encodeH32F32(enc, hal: Self.hal, x: x, out: y, rows: 1, hidden: 2560)
            }
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000
            best = min(best, ms)
        }
        print(String(format: "FN-FLOOR min of 15: 48 H32 dispatches gpu %.3f ms = %.1f us per dispatch", best, best * 1000 / 48))
    }
}
