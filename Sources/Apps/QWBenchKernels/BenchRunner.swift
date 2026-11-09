// BenchRunner.swift — the dispatch, the verification and the timing.
//
// A candidate is checked against the host at the shape it is about to be
// timed at, so the report never carries a number for a kernel that computes
// the wrong thing — and the check covers the shape, not a stand-in.

import Foundation
import Metal
import QWHAL
import QWQuant

/// Why a measurement could not be made.
enum BenchError: Error, CustomStringConvertible {
    case allocationFailed
    case kernelFailed(String, String)
    case noTiming(String)
    case verificationFailed(String, Double)

    var description: String {
        switch self {
        case .allocationFailed:
            return "could not allocate the bench buffers"
        case .kernelFailed(let k, let e):
            return "kernel \(k) failed: \(e)"
        case .noTiming(let k):
            return "kernel \(k) produced no GPU timing"
        case .verificationFailed(let k, let d):
            return "kernel \(k) disagreed with the host by \(d)"
        }
    }
}

/// The buffers one measurement needs, and the host reference they must match.
struct BenchWork {
    let weights: any MTLBuffer
    let scales: any MTLBuffer
    let biases: any MTLBuffer
    let x: any MTLBuffer
    let y: any MTLBuffer
    /// The host's answer for the same weights and the same activations.
    let reference: [BenchReference]
    /// The output is f16 for the MLX tile GEMM and f32 for the family.
    let yIsHalf: Bool
}

/// Builds the banks, dispatches the candidates and times them.
enum BenchRunner {
    /// The shape the quick verification runs at: small, and a multiple of 64.
    static let verifyRows = 64
    static let verifyCols = 256

    /// Builds the buffers and the host reference for one candidate.
    static func makeWork(hal: QWHal, candidate: BenchCandidate,
                         rows: Int, cols: Int) throws -> BenchWork {
        let device = hal.device
        let xs = makeActivations(cols: cols)
        switch candidate {
        case .family(let bits, let group):
            return try makeAQWork(device: device, bits: bits, group: group,
                                  rows: rows, cols: cols, xs: xs)
        case .incumbentQ4G32:
            return try makeIncumbentWork(device: device, rows: rows, cols: cols, xs: xs)
        case .mlxNAX(let bits, let group):
            return try makeMLXWork(device: device, bits: bits, group: group,
                                   rows: rows, cols: cols, xs: xs)
        }
    }

    /// Runs a candidate and returns the GPU milliseconds one dispatch took.
    /// The pipeline is warmed and the cache flushed first, so the number is
    /// a cold read — the regime a decode step is in.
    static func run(hal: QWHal, candidate: BenchCandidate, work: BenchWork,
                    rows: Int, cols: Int, repeats: Int = 1) throws -> Double {
        let pso = try hal.pipeline(candidate.name)
        // The family sweep and the operator suites share ONE cold-read
        // protocol: warm, flush, time. Two implementations of it would be
        // free to drift apart, and a bench that measures two regimes would
        // be comparing them.
        return try time(hal: hal, name: candidate.name,
                        warm: { enc in
                            encode(enc, pso: pso, candidate: candidate, work: work,
                                   rows: rows, cols: cols)
                        },
                        timed: { enc in
                            encode(enc, pso: pso, candidate: candidate, work: work,
                                   rows: rows, cols: cols)
                        },
                        repeats: repeats)
    }

    /// Encodes one candidate's dispatch into an encoder.
    static func encode(_ enc: any MTLComputeCommandEncoder,
                       pso: any MTLComputePipelineState, candidate: BenchCandidate,
                       work: BenchWork, rows: Int, cols: Int) {
        enc.setComputePipelineState(pso)
        switch candidate {
        case .family, .incumbentQ4G32:
            enc.setBuffer(work.weights, offset: 0, index: 0)
            enc.setBuffer(work.x, offset: 0, index: 1)
            enc.setBuffer(work.y, offset: 0, index: 3)
            var n32 = UInt32(rows), k32 = UInt32(cols)
            enc.setBytes(&n32, length: 4, index: 4)
            enc.setBytes(&k32, length: 4, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        case .mlxNAX:
            enc.setBuffer(work.weights, offset: 0, index: 0)
            enc.setBuffer(work.scales, offset: 0, index: 1)
            enc.setBuffer(work.biases, offset: 0, index: 2)
            enc.setBuffer(work.x, offset: 0, index: 3)
            enc.setBuffer(work.y, offset: 0, index: 4)
            var k32 = Int32(cols), n32 = Int32(rows), m32 = Int32(1)
            enc.setBytes(&k32, length: 4, index: 5)
            enc.setBytes(&n32, length: 4, index: 6)
            enc.setBytes(&m32, length: 4, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: (rows + 63) / 64, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
    }

    /// Evicts the GPU's cache, so the next dispatch reads its weights cold.
    /// The SDK imports `fillBuffer:range:value:` as `__fill`, because `fill`
    /// collides with `std::fill`.
    static func flush(hal: QWHal) {
        if flushBuffer == nil {
            flushBuffer = hal.device.makeBuffer(length: flushBytes, options: .storageModeShared)
        }
        guard let buffer = flushBuffer,
              let cb = hal.queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { return }
        blit.__fill(buffer, range: NSRange(location: 0, length: flushBytes), value: 0)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    /// The scratch buffer's size: large enough to evict the machine's L2.
    static let flushBytes = 256 << 20
    /// The bench is single-threaded: one process, one flush buffer.
    nonisolated(unsafe) private static var flushBuffer: (any MTLBuffer)?

    /// Checks a candidate's output against the work's host reference.
    static func verify(candidate: BenchCandidate, work: BenchWork, rows: Int) throws {
        var worst: Double = 0
        if work.yIsHalf {
            let got = work.y.contents().bindMemory(to: UInt16.self, capacity: rows)
            for r in 0..<rows {
                let v = Float(Float16(bitPattern: got[r]))
                worst = max(worst, relativeError(got: v, reference: work.reference[r]))
            }
        } else {
            let got = work.y.contents().bindMemory(to: Float.self, capacity: rows)
            for r in 0..<rows {
                worst = max(worst, relativeError(got: got[r], reference: work.reference[r]))
            }
        }
        guard worst <= 1e-3 else {
            // The first rows, side by side: a disagreement this size is a
            // mismatch of arithmetic, not of rounding.
            if !work.yIsHalf {
                let got = work.y.contents().bindMemory(to: Float.self, capacity: rows)
                for r in 0..<min(4, rows) {
                    print("  row \(r): got \(got[r]) reference \(work.reference[r].value) "
                          + "magnitude \(work.reference[r].magnitude)")
                }
            }
            throw BenchError.verificationFailed(candidate.name, worst)
        }
    }

    /// The relative error, against the magnitude of the row's terms. The
    /// value itself can be near zero, and the magnitude cannot.
    static func relativeError(got: Float, reference: BenchReference) -> Double {
        let scale = max(abs(Double(reference.value)), Double(reference.magnitude), 1e-6)
        return abs(Double(got) - Double(reference.value)) / scale
    }

    /// The activation vector: deterministic, and an f16 the format holds.
    static func makeActivations(cols: Int) -> [Float] {
        var out = [Float](repeating: 0, count: cols)
        for i in 0..<cols {
            let unit = Double(splitmix64(UInt64(i) &* 0x9E37_79B9_7F4A_7C15) % 2_000_001)
                / 1_000_000.0 - 1.0
            out[i] = Float(Float16(unit))
        }
        return out
    }

    /// Times one candidate's dispatches. The closure encodes them; the
    /// pipeline is warmed and the cache flushed first, so the number is a
    /// cold read — the regime a decode step is in.
    static func time(hal: QWHal, name: String,
                     warm: (any MTLComputeCommandEncoder) throws -> Void,
                     timed: (any MTLComputeCommandEncoder) throws -> Void,
                     repeats: Int = 1) throws -> Double {
        guard let (cb, enc) = hal.begin() else { throw BenchError.noTiming(name) }
        try warm(enc)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        if let error = cb.error { throw BenchError.kernelFailed(name, String(describing: error)) }
        flush(hal: hal)
        guard let (cb2, enc2) = hal.begin() else { throw BenchError.noTiming(name) }
        for _ in 0..<max(1, repeats) { try timed(enc2) }
        hal.end(cb2, enc2)
        cb2.waitUntilCompleted()
        if let error = cb2.error { throw BenchError.kernelFailed(name, String(describing: error)) }
        let start = cb2.gpuStartTime
        let end = cb2.gpuEndTime
        guard start > 0, end > start else { throw BenchError.noTiming(name) }
        return (end - start) * 1000.0 / Double(max(1, repeats))
    }

    /// The median of the timed samples — the middle one, not the mean, so a
    /// single scheduling hiccup cannot move the number.
    static func median(_ samples: [Double]) -> Double {
        let sorted = samples.sorted()
        return sorted[sorted.count / 2]
    }

    /// The worst relative error of an f16 output against the reference.
    static func worstError(_ buffer: any MTLBuffer, rows: Int,
                            reference: [BenchReference]) -> Double {
        let got = buffer.contents().bindMemory(to: UInt16.self, capacity: rows)
        var worst: Double = 0
        for r in 0..<rows {
            let v = Float(Float16(bitPattern: got[r]))
            worst = max(worst, relativeError(got: v, reference: reference[r]))
        }
        return worst
    }

    /// Makes a buffer the bench writes into.
    static func buffer(_ device: any MTLDevice, bytes: Int) throws -> any MTLBuffer {
        guard let buffer = device.makeBuffer(length: max(bytes, 4), options: .storageModeShared) else {
            throw BenchError.allocationFailed
        }
        return buffer
    }

    /// Fills a buffer with f16 values.
    static func fillF16(_ buffer: any MTLBuffer, _ values: [Float]) {
        let out = buffer.contents().bindMemory(to: UInt16.self, capacity: values.count)
        for i in 0..<values.count { out[i] = Float16(values[i]).bitPattern }
    }

    /// Fills a buffer with f32 values.
    static func fillF32(_ buffer: any MTLBuffer, _ values: [Float]) {
        let out = buffer.contents().bindMemory(to: Float.self, capacity: values.count)
        for i in 0..<values.count { out[i] = values[i] }
    }

    /// The normalised 32-point Fast Walsh-Hadamard transform, in place, over
    /// each group of 32 contiguous elements — the rotation the kernels apply,
    /// ending in the 1/sqrt(32) factor.
    static func fwht32(_ values: inout [Float]) {
        var base = 0
        while base < values.count {
            let end = min(base + 32, values.count)
            var s = 1
            while s < 32 {
                var i = base
                while i < end {
                    if i & s == 0, i + s < end {
                        let a = values[i], b = values[i + s]
                        values[i] = a + b
                        values[i + s] = a - b
                    }
                    i += 1
                }
                s <<= 1
            }
            base += 32
        }
        let scale = Float(0.17677669529663689)
        for i in 0..<values.count { values[i] *= scale }
    }

    private static func makeAQWork(device: any MTLDevice, bits: Int, group: Int,
                                   rows: Int, cols: Int, xs: [Float]) throws -> BenchWork {
        let bank = AQBanks.make(bits: bits, group: group, rows: rows, cols: cols)
        let weights = try makeBuffer(device: device, bytes: bank.count)
        bank.withUnsafeBytes { src in
            weights.contents().copyMemory(from: src.baseAddress!, byteCount: bank.count)
        }
        let x = try makeBuffer(device: device, bytes: cols * 2)
        let y = try makeBuffer(device: device, bytes: rows * 4)
        let xb = x.contents().bindMemory(to: UInt16.self, capacity: cols)
        for i in 0..<cols { xb[i] = Float16(xs[i]).bitPattern }
        var reference = [BenchReference](repeating: BenchReference(value: 0, magnitude: 0),
                                         count: rows)
        for r in 0..<rows {
            reference[r] = AQBanks.referenceRow(bits: bits, group: group, row: r,
                                                 cols: cols, bank: bank, x: xs)
        }
        let empty = try makeBuffer(device: device, bytes: 4)
        return BenchWork(weights: weights, scales: empty, biases: empty,
                         x: x, y: y, reference: reference, yIsHalf: false)
    }

    /// The incumbent's own container for the same logical weights: its own
    /// bytes, and the reference those bytes stand for.
    private static func makeIncumbentWork(device: any MTLDevice, rows: Int, cols: Int,
                                          xs: [Float]) throws -> BenchWork {
        let bank = AQBanks.makeIncumbent4G32(rows: rows, cols: cols)
        let weights = try makeBuffer(device: device, bytes: bank.count)
        bank.withUnsafeBytes { src in
            weights.contents().copyMemory(from: src.baseAddress!, byteCount: bank.count)
        }
        let x = try makeBuffer(device: device, bytes: cols * 2)
        let y = try makeBuffer(device: device, bytes: rows * 4)
        let xb = x.contents().bindMemory(to: UInt16.self, capacity: cols)
        for i in 0..<cols { xb[i] = Float16(xs[i]).bitPattern }
        var reference = [BenchReference](repeating: BenchReference(value: 0, magnitude: 0),
                                         count: rows)
        for r in 0..<rows {
            reference[r] = AQBanks.incumbentReferenceRow(row: r, cols: cols, bank: bank, x: xs)
        }
        let empty = try makeBuffer(device: device, bytes: 4)
        return BenchWork(weights: weights, scales: empty, biases: empty,
                         x: x, y: y, reference: reference, yIsHalf: false)
    }

    private static func makeMLXWork(device: any MTLDevice, bits: Int, group: Int,
                                    rows: Int, cols: Int, xs: [Float]) throws -> BenchWork {
        let (w, scales, biases) = MLXBanks.make(bits: bits, group: group,
                                                rows: rows, cols: cols)
        let weights = try makeBuffer(device: device, bytes: w.count)
        w.withUnsafeBytes { src in
            weights.contents().copyMemory(from: src.baseAddress!, byteCount: w.count)
        }
        let scaleBuf = try makeBuffer(device: device, bytes: scales.count * 2)
        let biasBuf = try makeBuffer(device: device, bytes: biases.count * 2)
        let sb = scaleBuf.contents().bindMemory(to: UInt16.self, capacity: scales.count)
        let bb = biasBuf.contents().bindMemory(to: UInt16.self, capacity: biases.count)
        for i in 0..<scales.count {
            sb[i] = Float16(scales[i]).bitPattern
            bb[i] = Float16(biases[i]).bitPattern
        }
        let x = try makeBuffer(device: device, bytes: cols * 2)
        let y = try makeBuffer(device: device, bytes: rows * 2)
        let xb = x.contents().bindMemory(to: UInt16.self, capacity: cols)
        for i in 0..<cols { xb[i] = Float16(xs[i]).bitPattern }
        var reference = [BenchReference](repeating: BenchReference(value: 0, magnitude: 0),
                                         count: rows)
        for r in 0..<rows {
            reference[r] = MLXBanks.referenceRow(bits: bits, group: group, row: r, cols: cols,
                                                 w: w, scales: scales, biases: biases, x: xs)
        }
        return BenchWork(weights: weights, scales: scaleBuf, biases: biasBuf,
                         x: x, y: y, reference: reference, yIsHalf: true)
    }

    private static func makeBuffer(device: any MTLDevice, bytes: Int) throws -> any MTLBuffer {
        guard let buffer = device.makeBuffer(length: max(bytes, 4), options: .storageModeShared) else {
            throw BenchError.allocationFailed
        }
        return buffer
    }

    @inline(__always)
    private static func splitmix64(_ x: UInt64) -> UInt64 {
        var z = x &+ 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
