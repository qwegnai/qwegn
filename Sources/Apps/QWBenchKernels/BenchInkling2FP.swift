// BenchInkling2FP.swift — Inkling-2 FP kernel shootout (--only-ink2-fp).
// Races baseline scalar Inkling-2 kernels against Wave 1-3 NAX/compact/GQA/batched-head
// kernels using alternating min-of-2 pairs (>= 6 rounds, >= 8 reps/CB) with CPU FP64 checks.

import Accelerate
import Foundation
import Metal
import QWCore
import QWHAL
import QWQuant

@_silgen_name("cblas_dgemm$NEWLAPACK")
private func ink2BenchDgemm(
    _ order: Int32, _ transA: Int32, _ transB: Int32,
    _ m: Int32, _ n: Int32, _ k: Int32,
    _ alpha: Double, _ a: UnsafePointer<Double>, _ lda: Int32,
    _ b: UnsafePointer<Double>, _ ldb: Int32,
    _ beta: Double, _ c: UnsafeMutablePointer<Double>, _ ldc: Int32
)

private struct Ink2DenseGemmP {
    var M, N, K, strideX, strideY, inIsF32, outIsF32, addResidual: UInt32
    var softcap: Float
}

private struct Ink2MoEP {
    var T, N, K, topK, numExperts, hidden: UInt32
    var gateCap, upCap: Float
}

private struct Ink2AttnP {
    var T, nH, nKV, headDim, dRel, sliding, relExtent, posOffset, ringCap, kvMode, kvLen, splits: UInt32
}

private struct Ink2SinkRelAttnP {
    var T, nH, nKV, headDim, dRel, sliding, relExtent, posOffset: UInt32
    var ringCap, kvMode, kvLen, splits, hasSink, hasRelBias, hasOutGate, reserved: UInt32
}

enum Ink2FPSuites {
    struct Row {
        let suite: String
        let shape: String
        let baseKernel: String
        let candKernel: String
        let baseMs: Double
        let baseRange: (Double, Double)
        let baseErr: Double
        let candMs: Double?
        let candRange: (Double, Double)?
        let candErr: Double?
        let note: String
    }

    private enum MoEFormat: String {
        case nvfp4 = "NVFP4_G16"
        case mxfp8 = "MXFP8_G32"
    }

    private static let bf16Tol = 1.5e-2
    private static let fp4E2M1: [Float] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    private static let fp8E4M3FN: [Float] = (0..<256).map { idx in
        let b = UInt8(idx)
        let sign: Float = (b & 0x80) != 0 ? -1.0 : 1.0
        let exp = Int((b >> 3) & 0x0F), mant = Int(b & 0x07)
        if exp == 0x0F && mant == 0x07 { return Float.nan }
        if exp == 0 { return sign * Float(mant) / 512.0 }
        let s: FloatingPointSign = (b & 0x80) != 0 ? .minus : .plus
        return Float(sign: s, exponent: exp - 7, significand: 1.0 + Float(mant) / 8.0)
    }

    static func run(hal: QWHal, samples: Int, repeats: Int) throws -> [Row] {
        let rounds = max(6, samples), reps = max(8, repeats)
        var rows: [Row] = []
        rows.append(contentsOf: try runBF16Dense(hal: hal, rounds: rounds, reps: reps))
        rows.append(contentsOf: try runMoESuite(hal: hal, format: .nvfp4, rounds: rounds, reps: reps))
        rows.append(contentsOf: try runMoESuite(hal: hal, format: .mxfp8, rounds: rounds, reps: reps))
        rows.append(contentsOf: try runAttnSuite(hal: hal, rounds: rounds, reps: reps))
        rows.append(contentsOf: try runHeadSuite(hal: hal, rounds: rounds, reps: reps))
        return rows
    }

    static func renderMarkdown(_ rows: [Row], rounds: Int, reps: Int) -> String {
        var out = "# qwg-bench-kernels --only-ink2-fp\n\n"
        out += "Alternating min-of-2 pairs: \(rounds) rounds x \(reps) reps/CB.\n\n"
        out += "| suite | shape | base | cand | base ms | cand ms | speedup | base err | cand err | status |\n"
        out += "|---|---|---|---|---|---|---|---|---|---|\n"
        for r in rows {
            let cms = r.candMs.map { String(format: "%.4f", $0) } ?? "-"
            let spd = r.candMs.map { String(format: "%.2fx", r.baseMs / max($0, 1e-9)) } ?? "-"
            let cerr = r.candErr.map { String(format: "%.1e", $0) } ?? "-"
            out += "| \(r.suite) | \(r.shape) | `\(r.baseKernel)` | `\(r.candKernel)` | "
                + String(format: "%.4f", r.baseMs) + " | \(cms) | \(spd) | "
                + String(format: "%.1e", r.baseErr) + " | \(cerr) | \(r.note) |\n"
        }
        return out
    }

    private static func resolvePipeline(_ hal: QWHal, _ name: String) -> (any MTLComputePipelineState)? {
        guard hal.metallib.functionNames.contains(name) else { return nil }
        return try? hal.pipeline(name)
    }

    private static func requirePipeline(_ hal: QWHal, _ name: String) throws -> any MTLComputePipelineState {
        guard let pso = resolvePipeline(hal, name) else {
            throw BenchError.kernelFailed(name, "missing from metallib")
        }
        return pso
    }

    private static func timeCB(
        hal: QWHal, name: String, repeats: Int,
        encode: (any MTLCommandBuffer, inout any MTLComputeCommandEncoder) throws -> Void
    ) throws -> Double {
        guard let (cb, initEnc) = hal.begin() else { throw BenchError.noTiming(name) }
        var enc = initEnc
        try encode(cb, &enc)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        if let err = cb.error { throw BenchError.kernelFailed(name, String(describing: err)) }

        BenchRunner.flush(hal: hal)
        guard let (cb2, initEnc2) = hal.begin() else { throw BenchError.noTiming(name) }
        var enc2 = initEnc2
        let nReps = max(1, repeats)
        for _ in 0..<nReps { try encode(cb2, &enc2) }
        hal.end(cb2, enc2)
        cb2.waitUntilCompleted()
        if let err = cb2.error { throw BenchError.kernelFailed(name, String(describing: err)) }
        let s = cb2.gpuStartTime, e = cb2.gpuEndTime
        guard s > 0, e > s else { throw BenchError.noTiming(name) }
        return (e - s) * 1000.0 / Double(nReps)
    }

    private static func raceMinOf2(
        hal: QWHal, suite: String, shape: String, baseName: String, candName: String,
        rounds: Int, reps: Int, candPresent: Bool = false, clearOut: () -> Void,
        encodeBase: (any MTLCommandBuffer, inout any MTLComputeCommandEncoder) throws -> Void,
        verifyBase: () throws -> Double,
        encodeCand: ((any MTLCommandBuffer, inout any MTLComputeCommandEncoder) throws -> Void)?,
        verifyCand: (() throws -> Double)?
    ) throws -> Row {
        clearOut()
        _ = try timeCB(hal: hal, name: baseName, repeats: 1, encode: encodeBase)
        let baseErr = try verifyBase()
        guard baseErr <= bf16Tol else { throw BenchError.verificationFailed(baseName, baseErr) }

        guard let encodeCand, let verifyCand else {
            var mins: [Double] = []
            mins.reserveCapacity(rounds)
            for _ in 0..<rounds {
                let a = try timeCB(hal: hal, name: baseName, repeats: reps, encode: encodeBase)
                let b = try timeCB(hal: hal, name: baseName, repeats: reps, encode: encodeBase)
                mins.append(min(a, b))
            }
            let sorted = mins.sorted()
            let note = candPresent ? "pending cand encoder" : "skipped (not in metallib)"
            return Row(suite: suite, shape: shape, baseKernel: baseName, candKernel: candName,
                       baseMs: sorted[sorted.count / 2], baseRange: (sorted.first!, sorted.last!),
                       baseErr: baseErr, candMs: nil, candRange: nil, candErr: nil, note: note)
        }

        clearOut()
        _ = try timeCB(hal: hal, name: candName, repeats: 1, encode: encodeCand)
        let candErr = try verifyCand()
        guard candErr <= bf16Tol else { throw BenchError.verificationFailed(candName, candErr) }

        var baseMins: [Double] = [], candMins: [Double] = []
        baseMins.reserveCapacity(rounds); candMins.reserveCapacity(rounds)
        for r in 0..<rounds {
            let b0: Double, b1: Double, c0: Double, c1: Double
            if r % 2 == 0 {
                b0 = try timeCB(hal: hal, name: baseName, repeats: reps, encode: encodeBase)
                c0 = try timeCB(hal: hal, name: candName, repeats: reps, encode: encodeCand)
                b1 = try timeCB(hal: hal, name: baseName, repeats: reps, encode: encodeBase)
                c1 = try timeCB(hal: hal, name: candName, repeats: reps, encode: encodeCand)
            } else {
                c0 = try timeCB(hal: hal, name: candName, repeats: reps, encode: encodeCand)
                b0 = try timeCB(hal: hal, name: baseName, repeats: reps, encode: encodeBase)
                c1 = try timeCB(hal: hal, name: candName, repeats: reps, encode: encodeCand)
                b1 = try timeCB(hal: hal, name: baseName, repeats: reps, encode: encodeBase)
            }
            baseMins.append(min(b0, b1)); candMins.append(min(c0, c1))
        }
        let bS = baseMins.sorted(), cS = candMins.sorted()
        return Row(suite: suite, shape: shape, baseKernel: baseName, candKernel: candName,
                   baseMs: bS[bS.count / 2], baseRange: (bS.first!, bS.last!), baseErr: baseErr,
                   candMs: cS[cS.count / 2], candRange: (cS.first!, cS.last!), candErr: candErr,
                   note: "ok")
    }

    @inline(__always)
    private static func splitmix64(_ x: UInt64) -> UInt64 {
        var z = x &+ 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    @inline(__always) private static func bf16F32(_ b: UInt16) -> Float { Float(bitPattern: UInt32(b) << 16) }
    @inline(__always) private static func f32BF16(_ v: Float) -> UInt16 { QWBF16.fromFloatRNE(v) }
    @inline(__always) private static func tanhSafe(_ x: Double) -> Double { tanh(min(max(x, -15.0), 15.0)) }

    @inline(__always)
    private static func ue8m0(_ b: UInt8) -> Float {
        if b == 0xFF { return Float.nan }
        if b == 0 { return Float(bitPattern: 0x0040_0000) }
        return Float(bitPattern: UInt32(b) << 23)
    }

    @inline(__always)
    private static func softcapSwiglu(_ g: Double, _ u: Double, gateCap: Float, upCap: Float) -> Double {
        let gc = Double(gateCap), uc = Double(upCap)
        let sg = (gc > 0 ? tanhSafe(g / gc) * gc : g) * (1.0 / (1.0 + exp(-g)))
        let su = uc > 0 ? tanhSafe(u / uc) * uc : u
        return Double(bf16F32(f32BF16(Float(sg * su))))
    }

    private static func fillBF16Words(_ buf: any MTLBuffer, count: Int, seed: UInt64, expBase: UInt64 = 0x7B) {
        let qCount = count >> 2
        let qPtr = buf.contents().bindMemory(to: UInt64.self, capacity: max(1, qCount))
        var s = seed
        for i in 0..<qCount {
            s = splitmix64(s)
            var packed: UInt64 = 0
            for lane in 0..<4 {
                let bits = (s >> (lane * 16))
                let sign = (bits & 1) << 15
                let exp = (expBase + ((bits >> 1) & 3)) << 7
                let mant = (bits >> 3) & 0x7F
                packed |= (sign | exp | mant) << (lane * 16)
            }
            qPtr[i] = packed
        }
        let u16 = buf.contents().bindMemory(to: UInt16.self, capacity: count)
        for i in (qCount << 2)..<count {
            s = splitmix64(s)
            let sign = (s & 1) << 15
            let exp = (expBase + ((s >> 1) & 3)) << 7
            u16[i] = UInt16(sign | exp | ((s >> 3) & 0x7F))
        }
    }

    private static func fillF16Words(_ buf: any MTLBuffer, count: Int, seed: UInt64, scale: Double = 0.5) {
        let ptr = buf.contents().bindMemory(to: UInt16.self, capacity: count)
        var s = seed
        for i in 0..<count {
            s = splitmix64(s)
            ptr[i] = Float16((Double(s & 0xFFFF) / 32767.5 - 1.0) * scale).bitPattern
        }
    }

    private static func bindBuffers(_ enc: any MTLComputeCommandEncoder, _ buffers: [any MTLBuffer]) {
        for (idx, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: idx) }
    }

    private static func verifyBF16Matmul(
        wBuf: any MTLBuffer, xBuf: any MTLBuffer, yBuf: any MTLBuffer, m: Int, n: Int, k: Int
    ) -> Double {
        let w = wBuf.contents().bindMemory(to: UInt16.self, capacity: n * k)
        let x = xBuf.contents().bindMemory(to: UInt16.self, capacity: m * k)
        let y = yBuf.contents().bindMemory(to: UInt16.self, capacity: m * n)
        var worst = 0.0
        for rowM in Array(Set([0, min(1, m - 1), m / 2, m - 1])).sorted() {
            for colN in Array(Set([0, min(1, n - 1), n / 2, n - 1])).sorted() {
                var val = 0.0, mag = 0.0
                for j in 0..<k {
                    let p = Double(bf16F32(w[colN * k + j])) * Double(bf16F32(x[rowM * k + j]))
                    val += p; mag += abs(p)
                }
                worst = max(worst, BenchRunner.relativeError(got: bf16F32(y[rowM * n + colN]),
                                                             reference: BenchReference(value: Float(val), magnitude: Float(mag))))
            }
        }
        return worst
    }

    private static func runBF16Dense(hal: QWHal, rounds: Int, reps: Int) throws -> [Row] {
        let dev = hal.device, n = 3072, k = 2304, maxM = 2048
        let wBuf = try BenchRunner.buffer(dev, bytes: n * k * 2)
        let xBuf = try BenchRunner.buffer(dev, bytes: maxM * k * 2)
        let yBuf = try BenchRunner.buffer(dev, bytes: maxM * n * 2)
        let dummyF = try BenchRunner.buffer(dev, bytes: 256)
        fillBF16Words(wBuf, count: n * k, seed: 0xBF16_0001, expBase: 0x7B)
        fillBF16Words(xBuf, count: maxM * k, seed: 0xBF16_0002, expBase: 0x7C)

        func encodeDense(_ enc: any MTLComputeCommandEncoder, pso: any MTLComputePipelineState,
                         m: Int, xOff: Int = 0, yOff: Int = 0, gridW: Int, gridH: Int, tgW: Int) {
            var gp = Ink2DenseGemmP(M: UInt32(m), N: UInt32(n), K: UInt32(k), strideX: UInt32(k), strideY: UInt32(n),
                                   inIsF32: 0, outIsF32: 0, addResidual: 0, softcap: 0.0)
            enc.setComputePipelineState(pso)
            enc.setBuffer(wBuf, offset: 0, index: 0)
            enc.setBuffer(xBuf, offset: xOff, index: 1)
            enc.setBuffer(dummyF, offset: 0, index: 2)
            enc.setBuffer(yBuf, offset: yOff, index: 3)
            enc.setBuffer(dummyF, offset: 0, index: 4)
            enc.setBytes(&gp, length: MemoryLayout<Ink2DenseGemmP>.stride, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: gridW, height: gridH, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgW, height: 1, depth: 1))
        }

        var rows: [Row] = []
        let gemmCandName = hal.caps.isMetal4TensorCapable ? "qwg_gemm_bf16_nax_bm32" : "qwg_gemm_bf16_sg_bm32"
        let pNax = resolvePipeline(hal, gemmCandName)
        let pR2M8 = try requirePipeline(hal, "qwg_gemv_bf16_rows2_m8")
        for m in [8, 64] {
            rows.append(try raceMinOf2(
                hal: hal, suite: "bf16_gemm", shape: "M=\(m) \(n)x\(k)",
                baseName: "qwg_gemv_bf16_rows2_m8", candName: gemmCandName, rounds: rounds, reps: reps,
                clearOut: { memset(yBuf.contents(), 0, m * n * 2) },
                encodeBase: { _, enc in
                    var row = 0
                    while row < m {
                        let sub = min(8, m - row)
                        encodeDense(enc, pso: pR2M8, m: sub, xOff: row * k * 2, yOff: row * n * 2,
                                    gridW: (n + 7) / 8, gridH: 1, tgW: 128)
                        row += sub
                    }
                },
                verifyBase: { verifyBF16Matmul(wBuf: wBuf, xBuf: xBuf, yBuf: yBuf, m: m, n: n, k: k) },
                encodeCand: pNax.map { p in { _, enc in encodeDense(enc, pso: p, m: m, gridW: (n + 63) / 64, gridH: (m + 31) / 32, tgW: 128) } },
                verifyCand: { verifyBF16Matmul(wBuf: wBuf, xBuf: xBuf, yBuf: yBuf, m: m, n: n, k: k) }
            ))
        }

        for m in 1...8 {
            let r4Name = "qwg_gemv_bf16_rows4_m\(m)", r2Name = "qwg_gemv_bf16_rows2_m\(m)"
            let pBase = try requirePipeline(hal, r4Name)
            let pCand = resolvePipeline(hal, r2Name)
            rows.append(try raceMinOf2(
                hal: hal, suite: "bf16_gemv", shape: "M=\(m) \(n)x\(k)",
                baseName: r4Name, candName: r2Name, rounds: rounds, reps: reps,
                clearOut: { memset(yBuf.contents(), 0, m * n * 2) },
                encodeBase: { _, enc in encodeDense(enc, pso: pBase, m: m, gridW: (n + 15) / 16, gridH: 1, tgW: 128) },
                verifyBase: { verifyBF16Matmul(wBuf: wBuf, xBuf: xBuf, yBuf: yBuf, m: m, n: n, k: k) },
                encodeCand: pCand.map { p in { _, enc in encodeDense(enc, pso: p, m: m, gridW: (n + 7) / 8, gridH: 1, tgW: 128) } },
                verifyCand: { verifyBF16Matmul(wBuf: wBuf, xBuf: xBuf, yBuf: yBuf, m: m, n: n, k: k) }
            ))
        }
        return rows
    }

    private static func fillMoEWeights(
        format: MoEFormat, wBuf: any MTLBuffer, scBuf: any MTLBuffer, s2Buf: (any MTLBuffer)?,
        experts: Int, n: Int, k: Int, seed: UInt64
    ) {
        let groups = (format == .nvfp4) ? (k >> 4) : (k >> 5)
        let wordsPerGrp = (format == .nvfp4) ? 2 : 8
        let wCount = experts * n * groups * wordsPerGrp
        let wPtr = wBuf.contents().bindMemory(to: UInt32.self, capacity: wCount)
        let sPtr = scBuf.contents().bindMemory(to: UInt8.self, capacity: experts * n * groups)
        if let s2Buf {
            let s2Ptr = s2Buf.contents().bindMemory(to: Float.self, capacity: experts)
            for e in 0..<experts { s2Ptr[e] = 0.0625 + Float(e & 7) * 0.03125 }
        }
        var s = seed
        for i in 0..<wCount {
            s = splitmix64(s)
            if format == .nvfp4 {
                wPtr[i] = UInt32(truncatingIfNeeded: s)
            } else {
                let b0 = UInt32(s & 0xBF), b1 = UInt32((s >> 8) & 0xBF)
                let b2 = UInt32((s >> 16) & 0xBF), b3 = UInt32((s >> 24) & 0xBF)
                wPtr[i] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
            }
        }
        for i in 0..<(experts * n * groups) {
            s = splitmix64(s)
            sPtr[i] = (format == .nvfp4) ? UInt8(0x28 + (s & 0x1F)) : UInt8(122 + (s & 0x07))
        }
    }

    private static func moeDotRow(
        format: MoEFormat, wPtr: UnsafePointer<UInt32>, scPtr: UnsafePointer<UInt8>, s2: Float,
        expert: Int, row: Int, rowsPerExp: Int, cols: Int, x: [Double]
    ) -> Double {
        let groups = (format == .nvfp4) ? (cols >> 4) : (cols >> 5)
        let wordsPerGrp = (format == .nvfp4) ? 2 : 8
        let wBase = (expert * rowsPerExp + row) * (groups * wordsPerGrp)
        let sBase = (expert * rowsPerExp + row) * groups
        var sum = 0.0
        for g in 0..<groups {
            if format == .nvfp4 {
                let w0 = wPtr[wBase + g * 2], w1 = wPtr[wBase + g * 2 + 1]
                let sc = Double(fp8E4M3FN[Int(scPtr[sBase + g])])
                let xOff = g * 16
                var gs = 0.0
                for j in 0..<8 {
                    gs += Double(fp4E2M1[Int((w0 >> (j * 4)) & 0xF)]) * x[xOff + j]
                    gs += Double(fp4E2M1[Int((w1 >> (j * 4)) & 0xF)]) * x[xOff + 8 + j]
                }
                sum += gs * sc
            } else {
                let sc = Double(ue8m0(scPtr[sBase + g]))
                let xOff = g * 32
                var gs = 0.0
                for wIdx in 0..<8 {
                    let word = wPtr[wBase + g * 8 + wIdx], k0 = xOff + wIdx * 4
                    gs += Double(fp8E4M3FN[Int(word & 0xFF)]) * x[k0]
                    gs += Double(fp8E4M3FN[Int((word >> 8) & 0xFF)]) * x[k0 + 1]
                    gs += Double(fp8E4M3FN[Int((word >> 16) & 0xFF)]) * x[k0 + 2]
                    gs += Double(fp8E4M3FN[Int((word >> 24) & 0xFF)]) * x[k0 + 3]
                }
                sum += gs * sc
            }
        }
        return (format == .nvfp4) ? (sum * Double(s2)) : sum
    }

    private static func verifyMoE(
        format: MoEFormat, t: Int, eCount: Int, h: Int, inter: Int, topK: Int,
        gateCap: Float, upCap: Float, bufs: [any MTLBuffer]
    ) -> Double {
        let xPtr = bufs[0].contents().bindMemory(to: UInt16.self, capacity: t * h)
        let idPtr = bufs[1].contents().bindMemory(to: Int32.self, capacity: t * topK)
        let twPtr = bufs[2].contents().bindMemory(to: Float.self, capacity: t * topK)
        let outPtr = bufs[3].contents().bindMemory(to: UInt16.self, capacity: t * h)
        @inline(__always) func u32p(_ b: any MTLBuffer) -> UnsafeMutablePointer<UInt32> { b.contents().bindMemory(to: UInt32.self, capacity: b.length / 4) }
        @inline(__always) func u8p(_ b: any MTLBuffer) -> UnsafeMutablePointer<UInt8> { b.contents().bindMemory(to: UInt8.self, capacity: b.length) }
        @inline(__always) func f32p(_ b: any MTLBuffer) -> UnsafeMutablePointer<Float> { b.contents().bindMemory(to: Float.self, capacity: eCount) }
        let (gW, gS, gS2) = (u32p(bufs[4]), u8p(bufs[5]), f32p(bufs[6]))
        let (uW, uS, uS2) = (u32p(bufs[7]), u8p(bufs[8]), f32p(bufs[9]))
        let (dW, dS, dS2) = (u32p(bufs[10]), u8p(bufs[11]), f32p(bufs[12]))

        let tokSamples = (t <= 1) ? [0] : [0, t - 1]
        let colSamples = Array(stride(from: 0, to: h, by: 256))
        var worst = 0.0
        var xRow = [Double](repeating: 0, count: h), act = [Double](repeating: 0, count: inter)

        for tok in tokSamples {
            for j in 0..<h { xRow[j] = Double(bf16F32(xPtr[tok * h + j])) }
            var colAcc = [Double](repeating: 0, count: colSamples.count)
            var colMag = [Double](repeating: 0, count: colSamples.count)
            for k in 0..<topK {
                let slot = tok * topK + k, expId = Int(idPtr[slot]), rw = Double(twPtr[slot])
                for i in 0..<inter {
                    let gv = moeDotRow(format: format, wPtr: gW, scPtr: gS, s2: gS2[expId],
                                       expert: expId, row: i, rowsPerExp: inter, cols: h, x: xRow)
                    let uv = moeDotRow(format: format, wPtr: uW, scPtr: uS, s2: uS2[expId],
                                       expert: expId, row: i, rowsPerExp: inter, cols: h, x: xRow)
                    act[i] = softcapSwiglu(gv, uv, gateCap: gateCap, upCap: upCap)
                }
                for (ci, d) in colSamples.enumerated() {
                    let dv = moeDotRow(format: format, wPtr: dW, scPtr: dS, s2: dS2[expId],
                                       expert: expId, row: d, rowsPerExp: h, cols: inter, x: act)
                    let term = rw * Double(bf16F32(f32BF16(Float(dv))))
                    colAcc[ci] += term; colMag[ci] += abs(term)
                }
            }
            for (ci, d) in colSamples.enumerated() {
                worst = max(worst, BenchRunner.relativeError(
                    got: bf16F32(outPtr[tok * h + d]),
                    reference: BenchReference(value: Float(colAcc[ci]), magnitude: Float(colMag[ci]))))
            }
        }
        return worst
    }

    private static func runMoESuite(hal: QWHal, format: MoEFormat, rounds: Int, reps: Int) throws -> [Row] {
        let dev = hal.device
        let eCount = 128, h = 2304, inter = 1152, topK = 8, maxT = 256
        let gateCap: Float = 4.0, upCap: Float = 25.0
        let grp = (format == .nvfp4) ? 16 : 32
        let guWBytes = (format == .nvfp4) ? (eCount * inter * (h / 2)) : (eCount * inter * h)
        let dnWBytes = (format == .nvfp4) ? (eCount * h * (inter / 2)) : (eCount * h * inter)

        let gw = try BenchRunner.buffer(dev, bytes: guWBytes), gSc = try BenchRunner.buffer(dev, bytes: eCount * inter * (h / grp))
        let uw = try BenchRunner.buffer(dev, bytes: guWBytes), uSc = try BenchRunner.buffer(dev, bytes: eCount * inter * (h / grp))
        let dw = try BenchRunner.buffer(dev, bytes: dnWBytes), dSc = try BenchRunner.buffer(dev, bytes: eCount * h * (inter / grp))
        let gS2 = try BenchRunner.buffer(dev, bytes: eCount * 4)
        let uS2 = try BenchRunner.buffer(dev, bytes: eCount * 4)
        let dS2 = try BenchRunner.buffer(dev, bytes: eCount * 4)
        fillMoEWeights(format: format, wBuf: gw, scBuf: gSc, s2Buf: gS2, experts: eCount, n: inter, k: h, seed: 0x4D01)
        fillMoEWeights(format: format, wBuf: uw, scBuf: uSc, s2Buf: uS2, experts: eCount, n: inter, k: h, seed: 0x4D02)
        fillMoEWeights(format: format, wBuf: dw, scBuf: dSc, s2Buf: dS2, experts: eCount, n: h, k: inter, seed: 0x4D03)

        let xBuf = try BenchRunner.buffer(dev, bytes: maxT * h * 2)
        fillBF16Words(xBuf, count: maxT * h, seed: 0x1234_5678, expBase: 0x7C)
        let topIdsBuf = try BenchRunner.buffer(dev, bytes: maxT * topK * 4)
        let topWBuf = try BenchRunner.buffer(dev, bytes: maxT * topK * 4)
        let idP = topIdsBuf.contents().bindMemory(to: Int32.self, capacity: maxT * topK)
        let twP = topWBuf.contents().bindMemory(to: Float.self, capacity: maxT * topK)
        for t in 0..<maxT {
            for k in 0..<topK {
                idP[t * topK + k] = Int32((t * 17 + k * 13) % eCount)
                twP[t * topK + k] = 0.5 + 0.125 * Float(k)
            }
        }

        let offsetsBuf = try BenchRunner.buffer(dev, bytes: (eCount + 1) * 4)
        let sortedBuf = try BenchRunner.buffer(dev, bytes: maxT * topK * 4)
        let moeActBuf = try BenchRunner.buffer(dev, bytes: maxT * topK * inter * 2)
        let slotOutBuf = try BenchRunner.buffer(dev, bytes: maxT * topK * h * 2)
        let outBuf = try BenchRunner.buffer(dev, bytes: maxT * h * 2)

        let verifyBufs = [xBuf, topIdsBuf, topWBuf, outBuf, gw, gSc, gS2, uw, uSc, uS2, dw, dSc, dS2]
        let candBufs = try Ink2FPCandidates.MoECandidateBuffers(
            dev: dev, isMXFP8: format == .mxfp8, eCount: eCount, h: h, inter: inter, topK: topK, maxT: maxT,
            gateCap: gateCap, upCap: upCap, gw: gw, gSc: gSc, gS2: gS2, uw: uw, uSc: uSc, uS2: uS2,
            dw: dw, dSc: dSc, dS2: dS2, xBuf: xBuf, topIdsBuf: topIdsBuf, topWBuf: topWBuf,
            offsetsBuf: offsetsBuf, sortedBuf: sortedBuf, moeActBuf: moeActBuf, slotOutBuf: slotOutBuf, outBuf: outBuf)
        var results: [Row] = []
        for (regime, t) in [("decode", 1), ("verify", 4), ("prefill", 256)] {
            let (compName, compEnc) = Ink2FPCandidates.makeMoEEncoder(hal: hal, bufs: candBufs, t: t, forceNAX: false)
            let (naxName, naxEnc) = Ink2FPCandidates.makeMoEEncoder(hal: hal, bufs: candBufs, t: t, forceNAX: true)
            guard let encodeBase = (t <= 8 ? compEnc : naxEnc) else { continue }
            let baseName = (t <= 8 ? compName : naxName)
            let encodeCand = (t <= 8 ? naxEnc : compEnc)
            let candName = (t <= 8 ? naxName : compName)
            let verifyFn = { verifyMoE(format: format, t: t, eCount: eCount, h: h, inter: inter, topK: topK, gateCap: gateCap, upCap: upCap, bufs: verifyBufs) }
            results.append(try raceMinOf2(
                hal: hal, suite: "moe_\(format.rawValue.lowercased())_\(regime)", shape: "E=\(eCount) H=\(h) I=\(inter) K=\(topK) T=\(t)",
                baseName: baseName, candName: candName, rounds: rounds, reps: reps,
                candPresent: encodeCand != nil, clearOut: { memset(outBuf.contents(), 0, t * h * 2) },
                encodeBase: encodeBase, verifyBase: verifyFn, encodeCand: encodeCand, verifyCand: encodeCand != nil ? verifyFn : nil))
        }
        return results
    }

    private static func verifyInk2Attn(
        t: Int, posOff: Int, nH: Int, nKV: Int, hd: Int, dRel: Int, relExtent: Int, bufs: [any MTLBuffer]
    ) -> Double {
        let (qBuf, rBuf, relProjBuf, sinkBuf) = (bufs[0], bufs[1], bufs[2], bufs[3])
        let (gateBuf, kBuf, vBuf, outBuf) = (bufs[4], bufs[5], bufs[6], bufs[7])
        let gqa = nH / nKV, qkScale = 1.0 / Double(hd).squareRoot()
        let qP = qBuf.contents().bindMemory(to: UInt16.self, capacity: t * nH * hd)
        let rP = rBuf.contents().bindMemory(to: UInt16.self, capacity: t * nH * dRel)
        let rpP = relProjBuf.contents().bindMemory(to: UInt16.self, capacity: dRel * relExtent)
        let sP = sinkBuf.contents().bindMemory(to: UInt16.self, capacity: t * nH)
        let gP = gateBuf.contents().bindMemory(to: UInt16.self, capacity: t * nH * hd)
        let kP = kBuf.contents().bindMemory(to: UInt16.self, capacity: (posOff + t) * nKV * hd)
        let vP = vBuf.contents().bindMemory(to: UInt16.self, capacity: (posOff + t) * nKV * hd)
        let oP = outBuf.contents().bindMemory(to: UInt16.self, capacity: t * nH * hd)

        let tokSamples = (t == 1) ? [0] : Array(Set([0, min(511, t - 1), min(1024, t - 1), t - 1])).sorted()
        var worst = 0.0
        var qVec = [Double](repeating: 0, count: hd), rVec = [Double](repeating: 0, count: dRel)

        for tok in tokSamples {
            let qPos = posOff + tok
            for h in [0, nH - 1] {
                let kvh = h / gqa, qOff = (tok * nH + h) * hd, rOff = (tok * nH + h) * dRel
                for d in 0..<hd { qVec[d] = Double(bf16F32(qP[qOff + d])) }
                for d in 0..<dRel { rVec[d] = Double(bf16F32(rP[rOff + d])) }

                var scores = [Double](repeating: 0, count: qPos + 1)
                let sinkVal = Double(bf16F32(sP[tok * nH + h]))
                var maxS = sinkVal
                for j in 0...qPos {
                    let kOff = (j * nKV + kvh) * hd
                    var dot = 0.0
                    for d in 0..<hd { dot += qVec[d] * Double(Float16(bitPattern: kP[kOff + d])) }
                    var sc = dot * qkScale
                    let dist = qPos - j
                    if dist < relExtent {
                        for d in 0..<dRel { sc += rVec[d] * Double(bf16F32(rpP[d * relExtent + dist])) }
                    }
                    scores[j] = sc
                    if sc > maxS { maxS = sc }
                }
                var lSum = exp(sinkVal - maxS)
                var probs = [Double](repeating: 0, count: qPos + 1)
                for j in 0...qPos { let p = exp(scores[j] - maxS); probs[j] = p; lSum += p }
                for d in [0, 64, 127] {
                    let sig = 1.0 / (1.0 + exp(-Double(bf16F32(gP[qOff + d]))))
                    var acc = 0.0, mag = 0.0
                    for j in 0...qPos {
                        let w = (probs[j] / lSum) * sig
                        let vv = Double(Float16(bitPattern: vP[(j * nKV + kvh) * hd + d]))
                        acc += w * vv; mag += abs(w * vv)
                    }
                    worst = max(worst, BenchRunner.relativeError(
                        got: bf16F32(oP[qOff + d]), reference: BenchReference(value: Float(acc), magnitude: Float(mag))))
                }
            }
        }
        return worst
    }

    private static func runAttnSuite(hal: QWHal, rounds: Int, reps: Int) throws -> [Row] {
        let dev = hal.device
        let nH = 24, nKV = 8, hd = 128, dRel = 16, relExtent = 1024, maxSeq = 4096, maxKV = 16384
        let qBuf = try BenchRunner.buffer(dev, bytes: maxSeq * nH * hd * 2)
        let rBuf = try BenchRunner.buffer(dev, bytes: maxSeq * nH * dRel * 2)
        let relProjBuf = try BenchRunner.buffer(dev, bytes: dRel * relExtent * 2)
        let sinkBuf = try BenchRunner.buffer(dev, bytes: maxSeq * nH * 2)
        let gateBuf = try BenchRunner.buffer(dev, bytes: maxSeq * nH * hd * 2)
        let kBuf = try BenchRunner.buffer(dev, bytes: maxKV * nKV * hd * 2)
        let vBuf = try BenchRunner.buffer(dev, bytes: maxKV * nKV * hd * 2)
        let kMeta = try BenchRunner.buffer(dev, bytes: 256), vMeta = try BenchRunner.buffer(dev, bytes: 256)
        let outBuf = try BenchRunner.buffer(dev, bytes: maxSeq * nH * hd * 2)
        let partO = try BenchRunner.buffer(dev, bytes: 16 * nH * hd * 4), partML = try BenchRunner.buffer(dev, bytes: 16 * nH * 8)

        fillBF16Words(qBuf, count: maxSeq * nH * hd, seed: 0xA770_0001, expBase: 0x7C)
        fillBF16Words(rBuf, count: maxSeq * nH * dRel, seed: 0xA770_0002, expBase: 0x7B)
        fillBF16Words(relProjBuf, count: dRel * relExtent, seed: 0xA770_0003, expBase: 0x7A)
        fillBF16Words(sinkBuf, count: maxSeq * nH, seed: 0xA770_0004, expBase: 0x7C)
        fillBF16Words(gateBuf, count: maxSeq * nH * hd, seed: 0xA770_0005, expBase: 0x7D)
        fillF16Words(kBuf, count: maxKV * nKV * hd, seed: 0xA770_0006, scale: 0.5)
        fillF16Words(vBuf, count: maxKV * nKV * hd, seed: 0xA770_0007, scale: 0.5)

        let pGQA = try requirePipeline(hal, "qwg_ink2_gqa_attn"), pSplit = try requirePipeline(hal, "qwg_ink2_gqa_split_combine")
        let pCombCand = resolvePipeline(hal, "qwg_attn_decode_sink_rel_combine")
        let vBufs = [qBuf, rBuf, relProjBuf, sinkBuf, gateBuf, kBuf, vBuf, outBuf]
        var results: [Row] = []
        for (label, t, posOff, kvLen, splits, isPrefill) in [
            ("attn_prefill_512", 512, 0, 512, 1, true),
            ("attn_prefill_4096", 4096, 0, 4096, 1, true),
            ("attn_decode_16k", 1, 16383, 16384, 8, false),
        ] {
            var ap = Ink2AttnP(T: UInt32(t), nH: UInt32(nH), nKV: UInt32(nKV), headDim: UInt32(hd),
                               dRel: UInt32(dRel), sliding: 0, relExtent: UInt32(relExtent),
                               posOffset: UInt32(posOff), ringCap: UInt32(kvLen), kvMode: 0,
                               kvLen: UInt32(kvLen), splits: UInt32(splits))
            let encodeBase: (any MTLCommandBuffer, inout any MTLComputeCommandEncoder) -> Void = { _, enc in
                enc.setComputePipelineState(pGQA)
                bindBuffers(enc, [qBuf, rBuf, relProjBuf, sinkBuf, gateBuf, kBuf, vBuf, kMeta, vMeta, outBuf, partO, partML])
                enc.setBytes(&ap, length: MemoryLayout<Ink2AttnP>.stride, index: 12)
                enc.dispatchThreadgroups(MTLSize(width: splits, height: nH, depth: t), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                if splits > 1 {
                    enc.memoryBarrier(scope: .buffers)
                    enc.setComputePipelineState(pSplit); bindBuffers(enc, [partO, partML, gateBuf, outBuf])
                    enc.setBytes(&ap, length: MemoryLayout<Ink2AttnP>.stride, index: 4)
                    enc.dispatchThreadgroups(MTLSize(width: nH, height: t, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                }
            }
            let prefillAttnName = hal.caps.isMetal4TensorCapable
                ? "qwg_attn_prefill_sink_rel_nax_f16"
                : "qwg_attn_prefill_sink_rel_mma_f16"
            let candName = isPrefill ? prefillAttnName : "qwg_attn_decode_sink_rel_gqa_f16"
            let spInit = Ink2SinkRelAttnP(
                T: UInt32(t), nH: UInt32(nH), nKV: UInt32(nKV), headDim: UInt32(hd), dRel: UInt32(dRel), sliding: 0,
                relExtent: UInt32(relExtent), posOffset: UInt32(posOff), ringCap: UInt32(kvLen), kvMode: 0,
                kvLen: UInt32(kvLen), splits: UInt32(splits), hasSink: 1, hasRelBias: 1, hasOutGate: 1, reserved: 0)
            let pCand = resolvePipeline(hal, candName)
            let encodeCand: ((any MTLCommandBuffer, inout any MTLComputeCommandEncoder) -> Void)? = pCand.flatMap { p in
                if isPrefill {
                    return { _, enc in
                        var sp = spInit; enc.setComputePipelineState(p)
                        bindBuffers(enc, [qBuf, rBuf, relProjBuf, sinkBuf, gateBuf, kBuf, vBuf, kMeta, vMeta, outBuf])
                        enc.setBytes(&sp, length: MemoryLayout<Ink2SinkRelAttnP>.stride, index: 10)
                        enc.dispatchThreadgroups(MTLSize(width: (t + 31) / 32, height: nKV, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 64 * (nH / nKV), height: 1, depth: 1))
                    }
                }
                guard splits <= 1 || pCombCand != nil else { return nil }
                return { _, enc in
                    var sp = spInit; enc.setComputePipelineState(p)
                    bindBuffers(enc, [qBuf, rBuf, relProjBuf, sinkBuf, gateBuf, kBuf, vBuf, kMeta, vMeta, outBuf, partO, partML])
                    enc.setBytes(&sp, length: MemoryLayout<Ink2SinkRelAttnP>.stride, index: 12)
                    enc.dispatchThreadgroups(MTLSize(width: splits, height: nKV, depth: t), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    if splits > 1, let pc = pCombCand {
                        enc.memoryBarrier(scope: .buffers); enc.setComputePipelineState(pc); bindBuffers(enc, [partO, partML, gateBuf, outBuf])
                        enc.setBytes(&sp, length: MemoryLayout<Ink2SinkRelAttnP>.stride, index: 4)
                        enc.dispatchThreadgroups(MTLSize(width: nH, height: t, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    }
                }
            }
            let verifyFn = { verifyInk2Attn(t: t, posOff: posOff, nH: nH, nKV: nKV, hd: hd, dRel: dRel, relExtent: relExtent, bufs: vBufs) }
            results.append(try raceMinOf2(
                hal: hal, suite: label, shape: "24Q/8KV hd=128 " + (isPrefill ? "seq=\(t)" : "kv=16K"),
                baseName: splits > 1 ? "qwg_ink2_gqa_attn+split" : "qwg_ink2_gqa_attn",
                candName: isPrefill ? candName : "\(candName)+combine", rounds: rounds, reps: reps,
                candPresent: pCand != nil, clearOut: { memset(outBuf.contents(), 0, t * nH * hd * 2) },
                encodeBase: encodeBase, verifyBase: verifyFn, encodeCand: encodeCand, verifyCand: encodeCand != nil ? verifyFn : nil
            ))
        }
        return results
    }

    private static func verifyHeadTokens(
        wBuf: any MTLBuffer, xBuf: any MTLBuffer, tokBuf: any MTLBuffer,
        m: Int, v: Int, k: Int, softcap: Float, bestIdx: [Int], bestVal: [Double]
    ) -> Double {
        let tokP = tokBuf.contents().bindMemory(to: Int32.self, capacity: m)
        let wP = wBuf.contents().bindMemory(to: UInt16.self, capacity: v * k)
        let xP = xBuf.contents().bindMemory(to: UInt16.self, capacity: m * k)
        var worst = 0.0
        for r in 0..<m {
            let got = Int(tokP[r])
            guard got >= 0 && got < v else { return Double.infinity }
            if got == bestIdx[r] { continue }
            var dot = 0.0
            for c in 0..<k { dot += Double(bf16F32(wP[got * k + c])) * Double(bf16F32(xP[r * k + c])) }
            let sc = Double(softcap)
            let gotLogit = sc > 0 ? tanhSafe(dot / sc) * sc : dot
            worst = max(worst, abs(bestVal[r] - gotLogit) / max(abs(bestVal[r]), 1.0))
        }
        return worst
    }

    private static func runHeadSuite(hal: QWHal, rounds: Int, reps: Int) throws -> [Row] {
        let dev = hal.device, v = 200058, k = 2304, maxM = 8
        let softcap: Float = 30.0
        let argmaxSlots = (v + 255) / 256
        let wBuf = try BenchRunner.buffer(dev, bytes: v * k * 2)
        let xBuf = try BenchRunner.buffer(dev, bytes: maxM * k * 2)
        let dummyF = try BenchRunner.buffer(dev, bytes: 256), dummyB = try BenchRunner.buffer(dev, bytes: 256)
        let logitsBuf = try BenchRunner.buffer(dev, bytes: maxM * v * 4)
        let keysHi = try BenchRunner.buffer(dev, bytes: maxM * argmaxSlots * 4)
        let keysLo = try BenchRunner.buffer(dev, bytes: maxM * argmaxSlots * 4)
        let tokBuf = try BenchRunner.buffer(dev, bytes: maxM * 4)
        fillBF16Words(wBuf, count: v * k, seed: 0xDEAD_0001, expBase: 0x7B)
        fillBF16Words(xBuf, count: maxM * k, seed: 0xDEAD_0002, expBase: 0x7C)

        let chunkRows = 4096
        var xF64 = [Double](repeating: 0, count: maxM * k)
        let xP = xBuf.contents().bindMemory(to: UInt16.self, capacity: maxM * k)
        for i in 0..<(maxM * k) { xF64[i] = Double(bf16F32(xP[i])) }
        let wP = wBuf.contents().bindMemory(to: UInt16.self, capacity: v * k)
        var wChunk = [Double](repeating: 0, count: chunkRows * k), yChunk = [Double](repeating: 0, count: maxM * chunkRows)
        var bestIdx = [Int](repeating: 0, count: maxM), bestVal = [Double](repeating: -Double.infinity, count: maxM)
        let sc = Double(softcap)
        var row0 = 0
        while row0 < v {
            let nChunk = min(chunkRows, v - row0)
            for i in 0..<(nChunk * k) { wChunk[i] = Double(bf16F32(wP[row0 * k + i])) }
            xF64.withUnsafeBufferPointer { xb in wChunk.withUnsafeBufferPointer { wb in
                yChunk.withUnsafeMutableBufferPointer { yb in
                    ink2BenchDgemm(101, 111, 112, Int32(maxM), Int32(nChunk), Int32(k), 1.0,
                                   xb.baseAddress!, Int32(k), wb.baseAddress!, Int32(k), 0.0, yb.baseAddress!, Int32(nChunk))
                }
            } }
            for r in 0..<maxM {
                for j in 0..<nChunk {
                    let lg = tanhSafe(yChunk[r * nChunk + j] / sc) * sc
                    if lg > bestVal[r] { bestVal[r] = lg; bestIdx[r] = row0 + j }
                }
            }
            row0 += nChunk
        }

        let pArg = try requirePipeline(hal, "qwg_head_argmax"), pFin = try requirePipeline(hal, "qwg_head_argmax_fin")
        let tg128 = MTLSize(width: 128, height: 1, depth: 1), tg256 = MTLSize(width: 256, height: 1, depth: 1), tg1024 = MTLSize(width: 1024, height: 1, depth: 1)
        var results: [Row] = []
        let candSlots = (v + 7) / 8
        let candHi = try BenchRunner.buffer(dev, bytes: maxM * candSlots * 4)
        let candLo = try BenchRunner.buffer(dev, bytes: maxM * candSlots * 4)
        for m in [1, 4, 8] {
            let gemvName = "qwg_gemv_bf16_rows2_m\(m)"
            let pGemv = try requirePipeline(hal, gemvName)
            var gp = Ink2DenseGemmP(M: UInt32(m), N: UInt32(v), K: UInt32(k), strideX: UInt32(k), strideY: UInt32(v),
                                   inIsF32: 0, outIsF32: 1, addResidual: 0, softcap: softcap)
            let encodeBase: (any MTLCommandBuffer, inout any MTLComputeCommandEncoder) -> Void = { _, enc in
                enc.setComputePipelineState(pGemv); bindBuffers(enc, [wBuf, xBuf, dummyF, dummyB, logitsBuf])
                enc.setBytes(&gp, length: MemoryLayout<Ink2DenseGemmP>.stride, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: (v + 7) / 8, height: 1, depth: 1), threadsPerThreadgroup: tg128)
                enc.memoryBarrier(scope: .buffers)
                for r in 0..<m {
                    var nU = UInt32(v), nTG = UInt32(argmaxSlots)
                    enc.setComputePipelineState(pArg)
                    enc.setBuffer(logitsBuf, offset: r * v * 4, index: 0)
                    enc.setBuffer(keysHi, offset: 0, index: 1); enc.setBuffer(keysLo, offset: 0, index: 2)
                    enc.setBytes(&nU, length: 4, index: 3)
                    enc.dispatchThreadgroups(MTLSize(width: argmaxSlots, height: 1, depth: 1), threadsPerThreadgroup: tg256)
                    enc.memoryBarrier(scope: .buffers)
                    enc.setComputePipelineState(pFin)
                    enc.setBuffer(keysHi, offset: 0, index: 0); enc.setBuffer(keysLo, offset: 0, index: 1)
                    enc.setBuffer(tokBuf, offset: r * 4, index: 2); enc.setBytes(&nTG, length: 4, index: 3)
                    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg1024)
                    if r + 1 < m { enc.memoryBarrier(scope: .buffers) }
                }
            }
            let candName = "qwg_head_bf16_argmax_m\(m)"
            let encodeCand = Ink2FPCandidates.makeHeadArgmaxEncoder(
                hal: hal, m: m, v: v, k: k, softcap: softcap,
                wBuf: wBuf, xBuf: xBuf, keysHi: candHi, keysLo: candLo, tokBuf: tokBuf)
            let verifyFn = { verifyHeadTokens(wBuf: wBuf, xBuf: xBuf, tokBuf: tokBuf, m: m, v: v, k: k, softcap: softcap, bestIdx: bestIdx, bestVal: bestVal) }
            results.append(try raceMinOf2(
                hal: hal, suite: "head_bf16_argmax", shape: "M=\(m) V=\(v) K=\(k)",
                baseName: "\(gemvName)+argmax", candName: candName, rounds: rounds, reps: reps,
                candPresent: encodeCand != nil, clearOut: { memset(tokBuf.contents(), 0xFF, m * 4) },
                encodeBase: encodeBase, verifyBase: verifyFn,
                encodeCand: encodeCand, verifyCand: encodeCand != nil ? verifyFn : nil
            ))
        }
        return results
    }
}
