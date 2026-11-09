import XCTest
import Metal
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWGDN
@testable import QWFlashNext

// Dense causal conv (qwg_gdn_conv, QWGDNConv.encode) GPU time per call, for comparison with the mlx-vlm / oMLX
// formulations timed in scratch/conv_mlx_bench.py. Gated: QW_BENCH_CONV=1. Each command buffer holds 50 independent
// dispatches; GPU time = (gpuEndTime - gpuStartTime) / 50, median over 20 buffers.
final class FNConvBenchTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    func testDenseConvGPUTime() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["QW_BENCH_CONV"] == "1", "bench gated by QW_BENCH_CONV=1")
        let conv = try QWGDNConv(hal: Self.hal, geometry: .qwen38_27b)
        let C = QWGDNConv.channels
        for T in [1, 512] {
            let sizes = QWGDNConv.sizes(T: T)
            let arena = try QWArena(hal: Self.hal, weights: 4096,
                                    activations: 2 * (sizes.xy + sizes.state + sizes.state + sizes.w + sizes.bias) * 2 + (1 << 20),
                                    kv: 0, gdnStates: 0)
            let act = arena.activationsRegion()
            let x = try act.allocate(QWTensorShape([sizes.xy], .f16))
            let y = try act.allocate(QWTensorShape([sizes.xy], .f16))
            let stIn = try act.allocate(QWTensorShape([sizes.state], .f16))
            let stOut = try act.allocate(QWTensorShape([sizes.state], .f16))
            let w = try act.allocate(QWTensorShape([sizes.w], .f16))
            let bias = try act.allocate(QWTensorShape([sizes.bias], .f16))
            var times = [Double]()
            for _ in 0..<20 {
                guard let (cb, enc) = Self.hal.begin() else { return XCTFail("no command buffer") }
                for _ in 0..<50 {
                    try conv.encode(enc, x: x, w: w, bias: bias, stIn: stIn, y: y, stOut: stOut, T: T)
                }
                Self.hal.end(cb, enc)
                cb.waitUntilCompleted()
                XCTAssertNil(cb.error, "GPU command failed: \(String(describing: cb.error))")
                times.append((cb.gpuEndTime - cb.gpuStartTime) * 1e6 / 50.0)
            }
            times.sort()
            print("FNConvBench dense qwg_gdn_conv T=\(T) C=\(C): median_us=\(times[times.count / 2]) min_us=\(times[0])")
        }
    }
}
