// BenchCoreScalingSuites.swift — Apple Silicon core-scaling & dispatch-tax suites (--only-core-scaling).
// Suite A measures UMA read BW & compute scaling vs active TG count across arithmetic intensities.
// Suite B measures per-dispatch boundary, memoryBarrier, and rotateEncoder overhead inside one CB.

import Foundation
import Metal
import QWHAL

enum CoreScalingSuites {
    struct ScalingRow {
        let intensity: String
        let tgCount: Int
        let gpuMs: Double
        let minMs: Double
        let maxMs: Double
        let gbps: Double
        let gflops: Double
        let pctOfPeak: Double
    }

    struct DispatchTaxRow {
        let mode: String
        let dispatches: Int
        let totalMs: Double
        let minMs: Double
        let maxMs: Double
        let usPerDispatch: Double
        let deltaUsVsSerial: Double
    }

    struct Report {
        let deviceName: String
        let scaling: [ScalingRow]
        let dispatchTax: [DispatchTaxRow]
    }

    private enum DispatchModeKind: String, CaseIterable {
        case serialNoBarrier = "serial_no_barrier"
        case serialBarrier = "serial_memory_barrier"
        case concurrentBarrier = "concurrent_memory_barrier"
        case serialRotate = "serial_rotate_encoder"
    }

    private struct IntensitySpec {
        let label: String
        let kernel: String
        let vec4Count: Int
        let flopsPerVec4: Int
    }

    private static let tgCounts = [1, 4, 8, 16, 24, 32, 40, 48, 64, 80, 120, 160]
    private static let threadsPerTG = 128
    private static let taxDispatches = 256

    private static let metalSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void bench_core_read_stream(
        device const float4* src [[buffer(0)]],
        device float* out [[buffer(1)]],
        constant uint& vec4Count [[buffer(2)]],
        uint gid [[thread_position_in_grid]],
        uint gsize [[threads_per_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint tgid [[threadgroup_position_in_grid]]
    ) {
        float4 acc = float4(0.0f);
        for (uint i = gid; i < vec4Count; i += gsize) {
            acc += src[i];
        }
        float s = acc.x + acc.y + acc.z + acc.w;
        s = simd_sum(s);
        threadgroup float part[4];
        if ((tid & 31u) == 0u) { part[tid >> 5u] = s; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) { out[tgid] = part[0] + part[1] + part[2] + part[3]; }
    }

    static inline float4 bench_fma_chain(float4 v, uint iters) {
        float4 a = v;
        float4 b = float4(1.0001f, 0.9999f, 1.0002f, 0.9998f);
        float4 c = float4(0.0001f, -0.0001f, 0.0002f, -0.0002f);
        for (uint k = 0u; k < iters; ++k) {
            a = fma(a, b, c);
        }
        return a;
    }

    kernel void bench_core_ai_8(
        device const float4* src [[buffer(0)]],
        device float* out [[buffer(1)]],
        constant uint& vec4Count [[buffer(2)]],
        uint gid [[thread_position_in_grid]],
        uint gsize [[threads_per_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint tgid [[threadgroup_position_in_grid]]
    ) {
        float4 acc = float4(0.0f);
        for (uint i = gid; i < vec4Count; i += gsize) {
            acc += bench_fma_chain(src[i], 16u);
        }
        float s = simd_sum(acc.x + acc.y + acc.z + acc.w);
        threadgroup float part[4];
        if ((tid & 31u) == 0u) { part[tid >> 5u] = s; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) { out[tgid] = part[0] + part[1] + part[2] + part[3]; }
    }

    kernel void bench_core_ai_32(
        device const float4* src [[buffer(0)]],
        device float* out [[buffer(1)]],
        constant uint& vec4Count [[buffer(2)]],
        uint gid [[thread_position_in_grid]],
        uint gsize [[threads_per_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint tgid [[threadgroup_position_in_grid]]
    ) {
        float4 acc = float4(0.0f);
        for (uint i = gid; i < vec4Count; i += gsize) {
            acc += bench_fma_chain(src[i], 64u);
        }
        float s = simd_sum(acc.x + acc.y + acc.z + acc.w);
        threadgroup float part[4];
        if ((tid & 31u) == 0u) { part[tid >> 5u] = s; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) { out[tgid] = part[0] + part[1] + part[2] + part[3]; }
    }

    kernel void bench_core_ai_1024(
        device const float4* src [[buffer(0)]],
        device float* out [[buffer(1)]],
        constant uint& vec4Count [[buffer(2)]],
        uint gid [[thread_position_in_grid]],
        uint gsize [[threads_per_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint tgid [[threadgroup_position_in_grid]]
    ) {
        float4 acc = float4(0.0f);
        for (uint i = gid; i < vec4Count; i += gsize) {
            acc += bench_fma_chain(src[i], 2048u);
        }
        float s = simd_sum(acc.x + acc.y + acc.z + acc.w);
        threadgroup float part[4];
        if ((tid & 31u) == 0u) { part[tid >> 5u] = s; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) { out[tgid] = part[0] + part[1] + part[2] + part[3]; }
    }

    kernel void bench_dispatch_nop(
        device uint* out [[buffer(0)]],
        constant uint& idx [[buffer(1)]],
        uint tid [[thread_index_in_threadgroup]]
    ) {
        if (tid == 0u) { out[idx & 255u] = idx; }
    }
    """

    static func run(hal: QWHal, samples: Int, repeats: Int) throws -> Report {
        let lib: any MTLLibrary
        do {
            lib = try hal.device.makeLibrary(source: metalSource, options: nil)
        } catch {
            throw BenchError.kernelFailed("bench_core_scaling", error.localizedDescription)
        }
        let rounds = max(6, samples)
        let reps = max(4, repeats)
        let scaling = try runCoreScaling(hal: hal, lib: lib, rounds: rounds, reps: reps)
        let tax = try runDispatchTax(hal: hal, lib: lib, rounds: rounds)
        return Report(deviceName: hal.caps.name, scaling: scaling, dispatchTax: tax)
    }

    private static func makePSO(_ hal: QWHal, _ lib: any MTLLibrary, _ name: String) throws -> any MTLComputePipelineState {
        guard let fn = lib.makeFunction(name: name) else {
            throw BenchError.kernelFailed(name, "missing from bench library")
        }
        return try hal.device.makeComputePipelineState(function: fn)
    }

    private static func runCoreScaling(
        hal: QWHal, lib: any MTLLibrary, rounds: Int, reps: Int
    ) throws -> [ScalingRow] {
        let specs: [IntensitySpec] = [
            IntensitySpec(label: "0_FLOP/B_read", kernel: "bench_core_read_stream",
                          vec4Count: 8_388_608, flopsPerVec4: 4),
            IntensitySpec(label: "8_FLOP/B", kernel: "bench_core_ai_8",
                          vec4Count: 8_388_608, flopsPerVec4: 128),
            IntensitySpec(label: "32_FLOP/B", kernel: "bench_core_ai_32",
                          vec4Count: 4_194_304, flopsPerVec4: 512),
            IntensitySpec(label: "1024_FLOP/B", kernel: "bench_core_ai_1024",
                          vec4Count: 131_072, flopsPerVec4: 16_384),
        ]
        let maxVec4 = specs.map(\.vec4Count).max() ?? 8_388_608
        let srcBuf = try BenchRunner.buffer(hal.device, bytes: maxVec4 * 16)
        let srcPtr = srcBuf.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<(maxVec4 * 4) { srcPtr[i] = Float((i % 97) + 1) * 1e-3 }
        let outBuf = try BenchRunner.buffer(hal.device, bytes: (tgCounts.max() ?? 160) * 4)

        var rows: [ScalingRow] = []
        for spec in specs {
            let pso = try makePSO(hal, lib, spec.kernel)
            let repCount = spec.kernel == "bench_core_ai_1024" ? min(reps, 4) : reps
            var rawStats: [(tg: Int, med: Double, lo: Double, hi: Double)] = []
            for tg in tgCounts {
                var nVec = UInt32(spec.vec4Count)
                let encodeOne: (any MTLComputeCommandEncoder) -> Void = { enc in
                    enc.setComputePipelineState(pso)
                    enc.setBuffer(srcBuf, offset: 0, index: 0)
                    enc.setBuffer(outBuf, offset: 0, index: 1)
                    enc.setBytes(&nVec, length: 4, index: 2)
                    enc.dispatchThreadgroups(MTLSize(width: tg, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: threadsPerTG, height: 1, depth: 1))
                }
                _ = try timeCompute(hal: hal, name: spec.kernel, mode: .serial, repeats: 1, flushCache: false) { _, enc in
                    encodeOne(enc)
                }
                var mins: [Double] = []
                mins.reserveCapacity(rounds)
                for _ in 0..<rounds {
                    let a = try timeCompute(hal: hal, name: spec.kernel, mode: .serial, repeats: repCount, flushCache: true) { _, enc in
                        encodeOne(enc)
                    }
                    let b = try timeCompute(hal: hal, name: spec.kernel, mode: .serial, repeats: repCount, flushCache: true) { _, enc in
                        encodeOne(enc)
                    }
                    mins.append(min(a, b))
                }
                let s = mins.sorted()
                rawStats.append((tg, s[s.count / 2], s.first!, s.last!))
            }
            let peakMs = rawStats.last?.med ?? 1.0
            let totalBytes = Double(spec.vec4Count) * 16.0
            let totalFlops = Double(spec.vec4Count) * Double(spec.flopsPerVec4)
            for item in rawStats {
                let sec = max(item.med / 1000.0, 1e-12)
                let gbps = totalBytes / sec / 1e9
                let gflops = totalFlops / sec / 1e9
                let pct = (peakMs / max(item.med, 1e-12)) * 100.0
                rows.append(ScalingRow(
                    intensity: spec.label, tgCount: item.tg, gpuMs: item.med,
                    minMs: item.lo, maxMs: item.hi, gbps: gbps, gflops: gflops, pctOfPeak: pct
                ))
            }
        }
        return rows
    }

    private static func runDispatchTax(
        hal: QWHal, lib: any MTLLibrary, rounds: Int
    ) throws -> [DispatchTaxRow] {
        let pso = try makePSO(hal, lib, "bench_dispatch_nop")
        let outBuf = try BenchRunner.buffer(hal.device, bytes: 256 * 4)
        let modes = DispatchModeKind.allCases
        var samplesByMode: [DispatchModeKind: [Double]] = [:]
        for m in modes {
            samplesByMode[m] = []
            _ = try encodeDispatchTaxPass(hal: hal, pso: pso, outBuf: outBuf, mode: m)
        }
        for r in 0..<rounds {
            let order = (r % 2 == 0) ? modes : modes.reversed()
            for m in order {
                let a = try encodeDispatchTaxPass(hal: hal, pso: pso, outBuf: outBuf, mode: m)
                let b = try encodeDispatchTaxPass(hal: hal, pso: pso, outBuf: outBuf, mode: m)
                samplesByMode[m, default: []].append(min(a, b))
            }
        }
        let baseMed = (samplesByMode[.serialNoBarrier] ?? [0]).sorted()[rounds / 2]
        let baseUs = (baseMed * 1000.0) / Double(taxDispatches)
        return modes.map { m in
            let s = (samplesByMode[m] ?? [0]).sorted()
            let med = s[s.count / 2]
            let us = (med * 1000.0) / Double(taxDispatches)
            return DispatchTaxRow(
                mode: m.rawValue, dispatches: taxDispatches, totalMs: med,
                minMs: s.first ?? med, maxMs: s.last ?? med,
                usPerDispatch: us, deltaUsVsSerial: us - baseUs
            )
        }
    }

    private static func encodeDispatchTaxPass(
        hal: QWHal, pso: any MTLComputePipelineState, outBuf: any MTLBuffer, mode: DispatchModeKind
    ) throws -> Double {
        let dispMode: QWHal.ComputeDispatchMode = (mode == .concurrentBarrier) ? .concurrent : .serial
        return try timeCompute(hal: hal, name: mode.rawValue, mode: dispMode, repeats: 1, flushCache: false) { cb, enc in
            for i in 0..<taxDispatches {
                var idx = UInt32(i)
                enc.setComputePipelineState(pso)
                enc.setBuffer(outBuf, offset: 0, index: 0)
                enc.setBytes(&idx, length: 4, index: 1)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                switch mode {
                case .serialNoBarrier:
                    break
                case .serialBarrier, .concurrentBarrier:
                    enc.memoryBarrier(scope: .buffers)
                case .serialRotate:
                    if i + 1 < taxDispatches { try hal.rotateEncoder(cb, &enc) }
                }
            }
        }
    }

    private static func timeCompute(
        hal: QWHal, name: String, mode: QWHal.ComputeDispatchMode, repeats: Int, flushCache: Bool,
        encode: (any MTLCommandBuffer, inout any MTLComputeCommandEncoder) throws -> Void
    ) throws -> Double {
        if flushCache { BenchRunner.flush(hal: hal) }
        guard let (cb, initEnc) = hal.beginUnretained(mode) else { throw BenchError.noTiming(name) }
        var enc = initEnc
        let nReps = max(1, repeats)
        for _ in 0..<nReps { try encode(cb, &enc) }
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        if let err = cb.error { throw BenchError.kernelFailed(name, String(describing: err)) }
        let s = cb.gpuStartTime, e = cb.gpuEndTime
        guard s > 0, e > s else { throw BenchError.noTiming(name) }
        return (e - s) * 1000.0 / Double(nReps)
    }

    static func renderMarkdown(_ report: Report, rounds: Int, reps: Int) -> String {
        var out = "# qwg-bench-kernels --only-core-scaling\n\n"
        out += "Device: `\(report.deviceName)` | Alternating min-of-2: \(rounds) rounds x \(reps) reps/CB.\n\n"
        out += "## Suite A: Active-TG Core Scaling\n\n"
        out += "| intensity | tgCount | gpu ms | range ms | GB/s | GFLOP/s | % of 160-TG peak |\n"
        out += "|---|---|---|---|---|---|---|\n"
        for r in report.scaling {
            out += "| \(r.intensity) | \(r.tgCount) | "
                + String(format: "%.4f", r.gpuMs) + " | "
                + String(format: "%.4f–%.4f", r.minMs, r.maxMs) + " | "
                + String(format: "%.1f", r.gbps) + " | "
                + String(format: "%.1f", r.gflops) + " | "
                + String(format: "%.1f%%", r.pctOfPeak) + " |\n"
        }
        out += "\n## Suite B: Dispatch, Barrier & Encoder Rotation Tax (N=\(taxDispatches))\n\n"
        out += "| mode | total ms | range ms | us/dispatch | delta us vs serial |\n"
        out += "|---|---|---|---|---|\n"
        for r in report.dispatchTax {
            out += "| `\(r.mode)` | "
                + String(format: "%.4f", r.totalMs) + " | "
                + String(format: "%.4f–%.4f", r.minMs, r.maxMs) + " | "
                + String(format: "%.3f", r.usPerDispatch) + " | "
                + String(format: "%+.3f", r.deltaUsVsSerial) + " |\n"
        }
        return out
    }
}
