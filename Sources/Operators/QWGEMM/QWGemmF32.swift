// QWGemmF32.swift — Shared f32 and NAX/MPP dense projection encoders, Walsh-Hadamard rotations, and SwiGLU/Add.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL

/// On-device Variational MTP relaxed acceptance (arXiv:2610.26693) for `encodeArgmaxFin`.
/// A draft that differs from the row argmax is accepted when its logit is within `margin` of the
/// maximum and, if `minLogProb` is finite, its softmax log-probability is at least `minLogProb`.
public struct QWVariationalAcceptance {
    public var logits: QWTensorValue
    public var draftTokens: QWTensorValue
    public var vocab: Int
    public var totalRows: Int
    public var margin: Float
    public var minLogProb: Float

    public init(logits: QWTensorValue, draftTokens: QWTensorValue, vocab: Int, totalRows: Int,
                margin: Float, minLogProb: Float = -.infinity) {
        self.logits = logits
        self.draftTokens = draftTokens
        self.vocab = vocab
        self.totalRows = totalRows
        self.margin = margin
        self.minLogProb = minLogProb
    }

    /// `log(minProbability)`, or `-inf` when the probability floor is off (`<= 0`).
    public static func minLogProb(forMinProbability p: Float) -> Float {
        p > 0 ? log(min(p, 1)) : -.infinity
    }
}

extension QWGemm {
    /// Dense NAX mode is Single-Pass (`qwg_gemm_q4g_nax_f32`) across both stochastic and greedy hyperparameters.
    public static let denseNaxMode: QWDenseNaxMode = .singlePass

    /// Resolves and applies the active Dense NAX kernel mode for the given sampling hyperparameters.
    @discardableResult
    public static func configureDenseNax(
        temperature: Double, topP: Double = 1.0, topK: Int = 0
    ) -> (mode: QWDenseNaxMode, logMessage: String) {
        let sampling = QWSamplingParameters(temperature: temperature, topP: topP, topK: topK)
        return (sampling.resolveDenseNaxMode(),
                sampling.denseNaxSelectionLogMessage(kernelName: "qwg_gemm_q4g_nax_f32"))
    }

    /// out f32 [rows, hidden] = rotate(x f32 [rows, hidden]) in the normalized Walsh-Hadamard domain.
    public static func encodeH32F32(_ enc: any MTLComputeCommandEncoder, hal: QWHal, x: QWTensorValue,
                                    out: QWTensorValue, rows: Int, hidden: Int) throws {
        enc.setComputePipelineState(try hal.pipeline("qwg_h32_f32"))
        enc.bindTensors(x, out, startIndex: 0)
        enc.setUInt32s(hidden, startIndex: 2)
        enc.dispatchThreadgroups(MTLSize(width: hidden / 32, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    /// silu(gate) * up followed by the 32-point WHT, f32. Requires intermediate % 32 == 0.
    public static func encodeSiluMulF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal, gu: QWTensorValue,
                                        act: QWTensorValue, rows: Int, intermediate: Int,
                                        act16: QWTensorValue? = nil) throws {
        precondition(intermediate % 32 == 0, "f32 silu-mul needs intermediate % 32 == 0")
        enc.setComputePipelineState(try hal.pipeline("qwg_silu_mul_f32"))
        enc.bindTensors(gu, act, startIndex: 0)
        enc.setUInt32s(rows, intermediate, startIndex: 2)
        (act16 ?? act).bind(enc, index: 4)
        enc.setUInt32s(act16 != nil ? 1 : 0, startIndex: 5)
        enc.dispatchThreads(MTLSize(width: intermediate / 8, height: rows, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    /// silu(gate) * up in f32 + 32-point WHT, emitting hi/lo f16 split halves (outHi + outLo).
    public static func encodeSiluMulSplitF16(_ enc: any MTLComputeCommandEncoder, hal: QWHal, gu: QWTensorValue,
                                             outHi: QWTensorValue, outLo: QWTensorValue,
                                             rows: Int, intermediate: Int) throws {
        enc.setComputePipelineState(try hal.pipeline("qwg_silu_mul_split_f16"))
        enc.bindTensors(gu, outHi, outLo, startIndex: 0)
        enc.setUInt32s(rows, intermediate, startIndex: 3)
        enc.dispatchThreads(MTLSize(width: (intermediate + 7) / 8, height: rows, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
    }

    /// out = a + b over count f32 elements.
    public static func encodeAddF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal, a: QWTensorValue, b: QWTensorValue,
                                    out: QWTensorValue, count: Int) throws {
        enc.setComputePipelineState(try hal.pipeline("qwg_add_f32"))
        enc.bindTensors(a, b, out, startIndex: 0)
        enc.setUInt32s(count, startIndex: 3)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Y[t, n] = epilogue(sum_k W[n, k] * X[t, k]); epilogue 0 none, 1 silu(y/scale), 2 sigmoid(y), 3 2*sigmoid(y/scale).
    public static func encodeF32Gemv(_ enc: any MTLComputeCommandEncoder, hal: QWHal, weight: QWTensorValue,
                                     x: QWTensorValue, y: QWTensorValue, rows: Int, n: Int, k: Int,
                                     epilogue: UInt32, scale: Float) throws {
        enc.bindTensors(weight, x, y, startIndex: 0)
        enc.setUInt32s(n, k, Int(epilogue), startIndex: 3)
        enc.setPod(scale, index: 6)
        if rows >= 4 {
            enc.setComputePipelineState(try hal.pipeline("qwg_f32_gemv_bm"))
            enc.setUInt32s(rows, startIndex: 7)
            let grid: MTLSize
            if rows >= 16 && n >= 16 && (k & 31) == 0 {
                grid = MTLSize(width: (n + 31) / 32, height: (rows + 31) / 32, depth: 1)
            } else if rows >= 16 && n < 16 && (k & 3) == 0 {
                grid = MTLSize(width: (rows + 31) / 32, height: (n + 3) / 4, depth: 1)
            } else {
                grid = MTLSize(width: (rows + 3) / 4, height: n, depth: 1)
            }
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            return
        }
        enc.setComputePipelineState(try hal.pipeline("qwg_f32_gemv"))
        enc.dispatchThreadgroups(MTLSize(width: n, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private static func encodeHQ4Gemv(_ enc: any MTLComputeCommandEncoder, hal: QWHal, kernel: String,
                                      weight: QWWeightView, x: QWTensorValue, y: QWTensorValue,
                                      rows: Int, rowsPerTG: Int, epi: UInt32, scale: Float,
                                      v4: Bool = false) throws {
        enc.setComputePipelineState(try hal.pipeline(kernel))
        if rows > 4 && epi != 3 && v4 {
            let xBytes = x.shape.dtype == .f16 ? 2 : 4
            let yBytes = y.shape.dtype == .f16 ? 2 : 4
            weight.blocks.bind(enc, index: 0)
            enc.setPod(scale, index: 6)
            var r0 = 0
            while r0 < rows {
                let r = min(4, rows - r0)
                let fuse = r >= 2
                let packedEpi = fuse ? (Int(epi) | (r << 8)) : Int(epi)
                x.slice(byteOffset: r0 * weight.k * xBytes, count: r * weight.k, dtype: x.shape.dtype).bind(enc, index: 1)
                y.slice(byteOffset: r0 * weight.n * yBytes, count: r * weight.n, dtype: y.shape.dtype).bind(enc, index: 2)
                enc.setUInt32s(weight.n, weight.k, packedEpi, startIndex: 3)
                enc.dispatchThreadgroups(MTLSize(width: (weight.n + rowsPerTG - 1) / rowsPerTG,
                                                 height: fuse ? 1 : r, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                r0 += r
            }
            return
        }
        enc.bindTensors(weight.blocks, x, y, startIndex: 0)
        let fuseMultiRow = (rows >= 2 && rows <= 4 && epi != 3 && v4)
        let packedEpi = (fuseMultiRow ? (Int(epi) | (rows << 8)) : Int(epi))
        enc.setUInt32s(weight.n, weight.k, packedEpi, startIndex: 3)
        enc.setPod(scale, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: (weight.n + rowsPerTG - 1) / rowsPerTG,
                                         height: fuseMultiRow ? 1 : rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// HQ4 rot-domain GEMV, f32 activations, 4 rows per simdgroup. epi: 0 none, 1 silu(y/scale), 2 sigmoid, 3 silu(y/scale) + H_32.
    public static func encodeHQ4F32V4(_ enc: any MTLComputeCommandEncoder, hal: QWHal, weight: QWWeightView,
                                      x: QWTensorValue, y: QWTensorValue, rows: Int = 1,
                                      epi: UInt32 = 0, scale: Float = 1) throws {
        let name: String
        if weight.isIncumbentContainer {
            name = "qwg_gemv_q4g_f32_v4"
        } else {
            guard let aqName = QWGemmAQ.f32V4KernelName(bits: weight.bits, group: weight.groupSize) else {
                throw QWGemmError.badWeights("no affine-quantised f32 v4 kernel for \(weight.bits)-bit group \(weight.groupSize)")
            }
            name = aqName
        }
        // The v4 body runs 16-lane halves (32 rows per threadgroup) when K / 32 is an odd multiple of 16.
        let halves = (weight.k / 32) % 32 == 16
        let rowsPerTG = (epi == 3 || halves) ? 32 : 16
        try encodeHQ4Gemv(enc, hal: hal, kernel: name,
                          weight: weight, x: x, y: y,
                          rows: rows, rowsPerTG: rowsPerTG, epi: epi, scale: scale, v4: true)
    }

    /// Threadgroups the v4-GEMV argmax epilogue writes one key pair per: (N + 15) / 16 (16 rows/TG).
    public static func blockArgmaxSlots(_ N: Int) -> Int { (N + 15) / 16 }

    /// HQ4 rot-domain GEMV with the greedy argmax folded into the epilogue: one sortable
    /// (logit, row) key pair per threadgroup (ties → lowest row; = qwg_head_argmax encoding).
    /// Follow with encodeArgmaxFin. epi: 0 none, 1 silu(y/scale), 2 sigmoid.
    public static func encodeHQ4F32V4Argmax(_ enc: any MTLComputeCommandEncoder, hal: QWHal, weight: QWWeightView,
                                            x: QWTensorValue, y: QWTensorValue,
                                            keysHi: QWTensorValue, keysLo: QWTensorValue,
                                            rows: Int = 1, epi: UInt32 = 0, scale: Float = 1) throws {
        let slots = blockArgmaxSlots(weight.n)
        precondition(keysHi.shape.elementCount >= rows * slots && keysLo.shape.elementCount >= rows * slots,
                     "v4 argmax key buffers need \(rows * slots) slots")
        if rows == 2 && epi == 0 {
            let name: String
            if weight.isIncumbentContainer {
                name = "qwg_gemv_q4g_f32_v4_am2"
            } else {
                guard let aqName = QWGemmAQ.f32V4Argmax2KernelName(bits: weight.bits, group: weight.groupSize) else {
                    throw QWGemmError.badWeights("no affine-quantised f32 v4_am2 kernel for \(weight.bits)-bit group \(weight.groupSize)")
                }
                name = aqName
            }
            enc.setComputePipelineState(try hal.pipeline(name))
            enc.bindTensors(weight.blocks, x, keysHi, keysLo, startIndex: 0)
            enc.setUInt32s(weight.n, weight.k, startIndex: 4)
            y.bind(enc, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: slots, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            return
        }
        let name: String
        if weight.isIncumbentContainer {
            name = "qwg_gemv_q4g_f32_v4_am"
        } else {
            guard let aqName = QWGemmAQ.f32V4ArgmaxKernelName(bits: weight.bits, group: weight.groupSize) else {
                throw QWGemmError.badWeights("no affine-quantised f32 v4_am kernel for \(weight.bits)-bit group \(weight.groupSize)")
            }
            name = aqName
        }
        enc.setComputePipelineState(try hal.pipeline(name))
        if rows > 4 {
            weight.blocks.bind(enc, index: 0)
            enc.setPod(scale, index: 6)
            var r0 = 0
            while r0 < rows {
                let r = min(4, rows - r0)
                let fuse = r > 1
                let packedEpi = fuse ? (Int(epi) | (r << 8)) : Int(epi)
                x.slice(byteOffset: r0 * weight.k * 4, count: r * weight.k, dtype: .f32).bind(enc, index: 1)
                y.slice(byteOffset: r0 * weight.n * 4, count: r * weight.n, dtype: .f32).bind(enc, index: 2)
                keysHi.slice(byteOffset: r0 * slots * 4, count: r * slots, dtype: keysHi.shape.dtype).bind(enc, index: 7)
                keysLo.slice(byteOffset: r0 * slots * 4, count: r * slots, dtype: keysLo.shape.dtype).bind(enc, index: 8)
                enc.setUInt32s(weight.n, weight.k, packedEpi, startIndex: 3)
                enc.dispatchThreadgroups(MTLSize(width: slots, height: fuse ? 1 : r, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                r0 += r
            }
            return
        }
        enc.bindTensors(weight.blocks, x, y, startIndex: 0)
        keysHi.bind(enc, index: 7)   // the kernel's key slots sit past the scalar constants (3-6)
        keysLo.bind(enc, index: 8)
        let fuseRows = rows > 1 && rows <= 4
        let packedEpi = fuseRows ? (Int(epi) | (rows << 8)) : Int(epi)
        enc.setUInt32s(weight.n, weight.k, packedEpi, startIndex: 3)
        enc.setPod(scale, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: slots, height: fuseRows ? 1 : rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Second argmax stage: folds per-threadgroup keys into the final token id, or dispatches
    /// `qwg_head_argmax_fin_variational` when `variational` has `margin > 0` and `rows > 1`.
    /// `minLogProb` gates relaxed acceptance on the draft's own log-probability (`-inf` disables it).
    public static func encodeArgmaxFin(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        keysHi: QWTensorValue,
        keysLo: QWTensorValue,
        token: QWTensorValue,
        slots: Int,
        rows: Int = 1,
        variational: QWVariationalAcceptance? = nil
    ) throws {
        precondition(keysHi.shape.elementCount >= rows * slots && keysLo.shape.elementCount >= rows * slots &&
                     token.shape.elementCount >= rows,
                     "argmax-fin buffers mismatch (slots=\(slots), rows=\(rows))")
        if let v = variational, v.margin > 0, rows > 1 {
            enc.setComputePipelineState(try hal.pipeline("qwg_head_argmax_fin_variational"))
            enc.bindTensors(keysHi, keysLo, token, startIndex: 0)
            enc.setUInt32s(slots, startIndex: 3)
            enc.bindTensors(v.logits, v.draftTokens, startIndex: 4)
            enc.setUInt32s(v.vocab, v.totalRows, startIndex: 6)
            enc.setPod(v.margin, index: 8)
            enc.setPod(v.minLogProb, index: 9)
        } else {
            enc.setComputePipelineState(try hal.pipeline("qwg_head_argmax_fin"))
            enc.bindTensors(keysHi, keysLo, token, startIndex: 0)
            enc.setUInt32s(slots, startIndex: 3)
        }
        enc.dispatchThreadgroups(MTLSize(width: 1, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    /// HQ4 rot-domain GEMM, f32 activations, weight-stationary: 64 token rows x 32 output rows per threadgroup.
    /// epi: 0 none, 1 silu(y / scale), 2 sigmoid, 3 silu(y / scale) + H_32.
    public static func encodeHQ4F32BM(_ enc: any MTLComputeCommandEncoder, hal: QWHal, weight: QWWeightView,
                                      x: QWTensorValue, y: QWTensorValue, rows: Int,
                                      epi: UInt32 = 0, scale: Float = 1) throws {
        let name: String
        if weight.isIncumbentContainer {
            name = "qwg_gemm_q4g_f32_bm"
        } else {
            guard let aqName = QWGemmAQ.f32BMKernelName(bits: weight.bits, group: weight.groupSize) else {
                throw QWGemmError.badWeights("no affine-quantised f32 bm kernel for \(weight.bits)-bit group \(weight.groupSize)")
            }
            name = aqName
        }
        enc.setComputePipelineState(try hal.pipeline(name))
        enc.bindTensors(weight.blocks, x, y, startIndex: 0)
        enc.setUInt32s(rows, weight.n, weight.k, Int(epi), startIndex: 3)
        enc.setPod(scale, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: (weight.n + 31) / 32, height: (rows + 63) / 64, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Embedding gather from a raw-bf16 table: out[c*hidden + d] = bf16(table[token*hidden + d])
    /// tiled over hc streams. The token id is read on-device (head token tensor), so the
    /// decode step needs no host staging of the input stream.
    public static func encodeEmbedBf16F32(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                          table: QWTensorValue, token: QWTensorValue, out: QWTensorValue,
                                          hidden: Int, hc: Int, rows: Int = 1) throws {
        enc.setComputePipelineState(try hal.pipeline("qwg_embed_bf16_f32"))
        enc.bindTensors(table, token, out, startIndex: 0)
        enc.setUInt32s(hidden, hc, startIndex: 3)
        enc.dispatchThreadgroups(MTLSize(width: (hidden * hc + 255) / 256, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Dense HQ4 GEMM on the NAX/MPP units with single-pass f16 activations: y32[rows, N] = x16 . W^T.
    public static func encodeNaxF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal, weight: QWWeightView,
                                    x16: QWTensorValue, y16: QWTensorValue, y32: QWTensorValue, rows: Int) throws {
        precondition(weight.n % 64 == 0 && weight.k % 64 == 0, "dense NAX needs N and K multiples of 64")
        // BM64 measures faster than BM128 at every M for both containers (M5 Max: 60-64 vs 26-56 TFLOPS).
        let name: String
        if weight.isIncumbentContainer {
            name = "qwg_gemm_q4g_nax_f32"
        } else {
            guard let aqName = QWGemmAQ.naxF32KernelName(bits: weight.bits, group: weight.groupSize) else {
                throw QWGemmError.badWeights("no affine-quantised nax f32 kernel for \(weight.bits)-bit group \(weight.groupSize)")
            }
            name = aqName
        }
        enc.setComputePipelineState(try hal.pipeline(name))
        enc.bindTensors(weight.blocks, x16, y16, y32, startIndex: 0)
        let outBytes = y32.shape.dtype == .f16 ? 2 : 4
        let flag = y32.shape.dtype == .f16 ? Int(bitPattern: 0x8000_0000) : 0
        enc.setUInt32s(weight.n, weight.k, startIndex: 5)
        let chunk = naxRowChunk(rows: rows, k: weight.k)
        var r0 = 0
        while r0 < rows {
            let r = min(chunk, rows - r0)
            if r0 > 0 {
                x16.slice(byteOffset: r0 * weight.k * 2, count: r * weight.k).bind(enc, index: 1)
                y32.slice(byteOffset: r0 * weight.n * outBytes, count: r * weight.n).bind(enc, index: 3)
            }
            enc.setUInt32s(r | flag, startIndex: 4)
            enc.dispatchThreadgroups(MTLSize(width: weight.n / 64, height: (r + 63) / 64, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            r0 += r
        }
    }

    /// Row chunk that keeps the f16 activation panel re-read by every column tile resident in the SLC.
    /// Unchunked, M x K panels beyond ~24 MB thrash to DRAM (M5 Max, M=8192: K=17408 25 -> 62 TFLOPS,
    /// K=6144 42 -> 62 TFLOPS chunked).
    public static func naxRowChunk(rows: Int, k: Int) -> Int {
        let budget = 24 << 20
        guard rows * k * 2 > budget else { return rows }
        return max(512, (budget / (k * 2)) / 64 * 64)
    }

    /// Unified dense HQ4 projection over `rows` f32 activations, automatically selecting Single-Pass NAX (`rows >= 64`),
    /// batched weight-stationary f32 GEMM (`rows >= 32`), or 4-row f32 GEMV.
    public static func encodeDenseProjF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal, weight: QWWeightView,
                                          x: QWTensorValue, y: QWTensorValue,
                                          naxX16: QWTensorValue, naxY16: QWTensorValue, maxTileWidth: Int,
                                          rows: Int, epi: UInt32 = 0, scale: Float = 1,
                                          x16Precast: QWTensorValue? = nil) throws {
        if rows >= 64 && epi == 0 && scale == 1 &&
           weight.n % 64 == 0 && weight.k % 64 == 0 && weight.k <= maxTileWidth {
            let count = rows * weight.k
            if let precast = x16Precast {
                let xHi = precast.slice(byteOffset: 0, count: count, dtype: .f16)
                try encodeNaxF32(enc, hal: hal, weight: weight, x16: xHi, y16: xHi, y32: y, rows: rows)
            } else {
                let xHi = naxX16.slice(byteOffset: 0, count: count, dtype: .f16)
                let xLo = naxY16.slice(byteOffset: 0, count: count, dtype: .f16)
                try encodeCast(enc, hal: hal, x32: x, y16: xHi, count: count)
                enc.memoryBarrier(scope: .buffers)
                try encodeNaxF32(enc, hal: hal, weight: weight, x16: xHi, y16: xLo, y32: y, rows: rows)
            }
        } else if rows >= 32 {
            try encodeHQ4F32BM(enc, hal: hal, weight: weight, x: x, y: y, rows: rows, epi: epi, scale: scale)
        } else {
            try encodeHQ4F32V4(enc, hal: hal, weight: weight, x: x, y: y, rows: rows, epi: epi, scale: scale)
        }
    }
}
