// dram_bw_probe.swift — measures the GPU's achievable DRAM read bandwidth with trivial streaming kernels.
//
//   swiftc -O Scripts/dram_bw_probe.swift -o /tmp/dram_bw_probe && /tmp/dram_bw_probe
//
// Kernel bandwidth claims should be compared against the SUSTAINED figure printed here (40 back-to-back
// dispatches per command buffer, best of 5), not the interface peak: a pure read loop is the upper bound
// any weight-streaming GEMV can reach on the machine as it is loaded at the time of measurement.
import Foundation
import Metal

let src = """
#include <metal_stdlib>
using namespace metal;
kernel void rd_grid(device const float4* a [[buffer(0)]], device float* o [[buffer(1)]], constant uint& n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]], uint gs [[threads_per_grid]]) {
    float4 s0 = 0, s1 = 0, s2 = 0, s3 = 0; uint i = gid;
    for (; i + 3 * gs < n; i += 4 * gs) { s0 += a[i]; s1 += a[i + gs]; s2 += a[i + 2 * gs]; s3 += a[i + 3 * gs]; }
    for (; i < n; i += gs) s0 += a[i];
    float4 s = s0 + s1 + s2 + s3; if (s.x == 1234.5f) o[gid] = s.y;
}
kernel void rd_block(device const float4* a [[buffer(0)]], device float* o [[buffer(1)]], constant uint& n [[buffer(2)]],
                     uint tg [[threadgroup_position_in_grid]], uint ntg [[threadgroups_per_grid]],
                     uint lid [[thread_index_in_threadgroup]], uint tpg [[threads_per_threadgroup]]) {
    uint per = (n + ntg - 1) / ntg; uint b = tg * per; uint e = min(n, b + per);
    float4 s0 = 0, s1 = 0; uint i = b + lid;
    for (; i + tpg < e; i += 2 * tpg) { s0 += a[i]; s1 += a[i + tpg]; }
    for (; i < e; i += tpg) s0 += a[i];
    float4 s = s0 + s1; if (s.x == 1234.5f) o[tg * tpg + lid] = s.y;
}
"""

guard let dev = MTLCreateSystemDefaultDevice(), let queue = dev.makeCommandQueue() else { fatalError("no Metal device") }
let lib = try dev.makeLibrary(source: src, options: nil)
let bytes = 2 << 30
guard let a = dev.makeBuffer(length: bytes, options: .storageModePrivate),
      let o = dev.makeBuffer(length: 1 << 26, options: .storageModePrivate) else { fatalError("alloc failed") }
do {
    let cb = queue.makeCommandBuffer()!, b = cb.makeBlitCommandEncoder()!
    b.fill(buffer: a, range: 0..<bytes, value: 1); b.endEncoding(); cb.commit(); cb.waitUntilCompleted()
}
var n = UInt32(bytes / 16)

/// Best-of-`reps` bandwidth in GB/s for `dispatches` back-to-back passes over the buffer in one command buffer.
func measure(_ pso: MTLComputePipelineState, tg: Int, grid: Int, dispatches: Int, reps: Int = 5) -> Double {
    var best = 0.0
    for _ in 0..<reps {
        let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(pso); e.setBuffer(a, offset: 0, index: 0); e.setBuffer(o, offset: 0, index: 1)
        e.setBytes(&n, length: 4, index: 2)
        for _ in 0..<dispatches {
            e.dispatchThreadgroups(MTLSize(width: grid, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        }
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        best = max(best, Double(dispatches) * Double(bytes) / (cb.gpuEndTime - cb.gpuStartTime) / 1e9)
    }
    return best
}

print(dev.name)
var burst = (0.0, ""), sustained = (0.0, "")
for name in ["rd_grid", "rd_block"] {
    let pso = try dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
    for tg in [256, 512, 1024] {
        for grid in [160, 320, 640, 1280] {
            let b = measure(pso, tg: tg, grid: grid, dispatches: 1)
            let s = measure(pso, tg: tg, grid: grid, dispatches: 40)
            let cfg = "\(name) tg=\(tg) grid=\(grid)"
            if b > burst.0 { burst = (b, cfg) }
            if s > sustained.0 { sustained = (s, cfg) }
            print(String(format: "%-26@ burst %6.1f GB/s   sustained %6.1f GB/s", cfg, b, s))
        }
    }
}
print(String(format: "BEST BURST     %.1f GB/s (%@)", burst.0, burst.1))
print(String(format: "BEST SUSTAINED %.1f GB/s (%@)", sustained.0, sustained.1))
