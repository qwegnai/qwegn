// QWGemmAQ.swift — the size-preserving affine-quantised GEMV family.
//
// A (bits, group) pair names one kernel, and the pair's native size is
// never widened to a denser group. The 4-bit G32 pair is the same SIZE as
// the incumbent's Block_HQ4_G32, but a different container: the payload is
// a bit stream and the header is not folded.

import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// The size-preserving affine-quantised and SHVQ GEMV family (Law L8).
public enum QWGemmAQ {
    private static func isSupported(bits: Int, group: Int, isSHVQ: Bool) -> Bool {
        isSHVQ ? QWSHVQBlock.isSupported(bits: bits, group: group) : QWAQBlock.isBlockSupported(bits: bits, group: group)
    }

    /// The kernel for a `(bits, group)` pair.
    public static func kernelName(bits: Int, group: Int, f32Activations: Bool = false, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        let base = isSHVQ ? "qwg_gemv_shvq\(bits)_g\(group)_v4" : "qwg_gemv_aq\(bits)_g\(group)_v4"
        return f32Activations ? base + "_f32" : base
    }

    /// The decode kernel for a `(bits, group)` pair: four output rows per simdgroup, f16 in and f16 out.
    public static func decodeKernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_gemv_shvq\(bits)_g\(group)_dx_h16_t4" : "qwg_gemv_aq\(bits)_g\(group)_dx_h16_t4"
    }

    /// The threads one decode threadgroup runs, and the rows it owns.
    public static let decodeThreadsPerThreadgroup = 128
    public static let decodeRowsPerThreadgroup = 16

    /// The MLP gate/up kernel for a `(bits, group)` pair.
    public static func siluPairKernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_gemv_shvq\(bits)_g\(group)_silu_pair_dx_h16" : "qwg_gemv_aq\(bits)_g\(group)_silu_pair_dx_h16"
    }

    /// The head's fused GEMV-and-argmax kernel for a `(bits, group)` pair.
    public static func headArgmaxKernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_head_gemv_shvq\(bits)_g\(group)_argmax" : "qwg_head_gemv_aq\(bits)_g\(group)_argmax"
    }

    /// The prefill GEMM for a `(bits, group)` pair.
    public static func gemmKernelName(bits: Int, group: Int, bm128: Bool = false, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        let base = isSHVQ ? "qwg_gemm_shvq\(bits)_g\(group)_h16" : "qwg_gemm_aq\(bits)_g\(group)_h16"
        return base + (bm128 ? "_bm128" : "")
    }

    /// The GDN in-projection decode kernel for a `(bits, group)` pair.
    public static func gdnInProjDecodeKernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_gemv_shvq\(bits)_g\(group)_dx_h16_gdn_inproj_t4" : "qwg_gemv_aq\(bits)_g\(group)_dx_h16_gdn_inproj_t4"
    }

    /// The GDN in-projection prefill kernel for a `(bits, group)` pair.
    public static func gdnInProjPrefillKernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_gemm_shvq\(bits)_g\(group)_h16_gdn_inproj" : "qwg_gemm_aq\(bits)_g\(group)_h16_gdn_inproj"
    }

    public static func f32BMKernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_gemm_shvq\(bits)_g\(group)_f32_bm" : "qwg_gemm_aq\(bits)_g\(group)_f32_bm"
    }

    public static func f32V4KernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_gemv_shvq\(bits)_g\(group)_f32_v4" : "qwg_gemv_aq\(bits)_g\(group)_f32_v4"
    }

    public static func f32V4ArgmaxKernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_gemv_shvq\(bits)_g\(group)_f32_v4_am" : "qwg_gemv_aq\(bits)_g\(group)_f32_v4_am"
    }

    public static func f32V4Argmax2KernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_gemv_shvq\(bits)_g\(group)_f32_v4_am2" : "qwg_gemv_aq\(bits)_g\(group)_f32_v4_am2"
    }

    public static func naxF32KernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_gemm_shvq\(bits)_g\(group)_nax_f32" : "qwg_gemm_aq\(bits)_g\(group)_nax_f32"
    }

    public static func moeGateUpF32R1KernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_moe_gateup_shvq\(bits)_g\(group)_f32_r1" : "qwg_moe_gateup_aq\(bits)_g\(group)_f32_r1"
    }

    public static func moeDownSharedCombineF32KernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_moe_down_shared_combine_shvq\(bits)_g\(group)_f32" : "qwg_moe_down_shared_combine_aq\(bits)_g\(group)_f32"
    }

    public static func moeNaxGemmKernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_moe_nax_gemm_shvq\(bits)_g\(group)" : "qwg_moe_nax_gemm_aq\(bits)_g\(group)"
    }

    public static func moeNaxSplitF32KernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_moe_nax_split_shvq\(bits)_g\(group)_f32" : "qwg_moe_nax_split_aq\(bits)_g\(group)_f32"
    }

    public static func hcUpMixH32F32KernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_hc_up_mix_h32_shvq\(bits)_g\(group)_f32" : "qwg_hc_up_mix_h32_aq\(bits)_g\(group)_f32"
    }

    public static func hcDownInjectF32KernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_hc_down_inject_shvq\(bits)_g\(group)_f32" : "qwg_hc_down_inject_aq\(bits)_g\(group)_f32"
    }

    /// The decode HC norm(+inject)+rotation+down+inject single-dispatch kernel for a `(bits, group)` down bank.
    public static func hcNormDownInjectF32KernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_hc_norm_down_inject_shvq\(bits)_g\(group)_f32" : "qwg_hc_norm_down_inject_aq\(bits)_g\(group)_f32"
    }

    /// The bytes one row of an `(n, k)` bank of `(bits, group)` blocks takes.
    public static func rowBytes(bits: Int, group: Int, k: Int, isSHVQ: Bool = false) -> Int? {
        let maybeBytes = isSHVQ
            ? QWSHVQBlock.bytesIfSupported(bits: bits, group: group)
            : QWAQBlock.bytesIfBlockSupported(bits: bits, group: group)
        guard let groupBytes = maybeBytes,
              k > 0, k % group == 0 else { return nil }
        return (k / group) * groupBytes
    }

    /// The embedding gather over the family's blocks: the token's row, decoded
    /// on the device. The pair rides in the constants, so ONE kernel serves
    /// every pair the family holds.
    public static func encodeEmbedAQ(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                      blocks: QWTensorValue, token: QWTensorValue,
                                      out: QWTensorValue, hidden: Int, hc: Int,
                                      bits: Int, group: Int, rows: Int = 1) throws {
        guard hidden > 0, hc > 0, rows > 0 else {
            throw QWGemmError.badWeights("the embedding gather needs a positive shape")
        }
        guard bits >= 1, bits <= 8, group % 32 == 0 else {
            throw QWGemmError.badWeights("no affine-quantised container for \(bits)-bit group \(group)")
        }
        guard hidden % group == 0 else {
            throw QWGemmError.badWeights("hidden must be a multiple of the group \(group)")
        }
        guard let rowBytes = QWAQBlock.bytesIfSupported(bits: bits, group: group)
                .map({ (hidden / group) * $0 }) else {
            throw QWGemmError.badWeights("no affine-quantised size for \(bits)-bit group \(group)")
        }
        guard blocks.bytes >= rowBytes else {
            throw QWGemmError.badWeights("the table holds \(blocks.bytes) bytes, \(rowBytes) needed")
        }
        guard token.shape.elementCount >= rows else {
            throw QWGemmError.badWeights("the token tensor holds \(token.shape.elementCount), \(rows) needed")
        }
        guard out.shape.elementCount >= rows * hidden * hc else {
            throw QWGemmError.badWeights("the output holds \(out.shape.elementCount) values, "
                                         + "\(rows * hidden * hc) needed")
        }
        let pso = try hal.pipeline("qwg_embed_aq_f32")
        enc.setComputePipelineState(pso)
        blocks.bind(enc, index: 0)
        token.bind(enc, index: 1)
        out.bind(enc, index: 2)
        var hiddenU = UInt32(hidden), hcU = UInt32(hc)
        var bitsU = UInt32(bits), groupU = UInt32(group)
        enc.setBytes(&hiddenU, length: 4, index: 3)
        enc.setBytes(&hcU, length: 4, index: 4)
        enc.setBytes(&bitsU, length: 4, index: 5)
        enc.setBytes(&groupU, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: (hidden + 255) / 256, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Encodes one GEMV over a bank of `(bits, group)` blocks: one
    /// threadgroup per row, 32 threads, so the row is uniform across the
    /// group. The dtypes and extents are checked, not assumed.
    public static func encode(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                               blocks: QWTensorValue, x: QWTensorValue, y: QWTensorValue,
                               n: Int, k: Int, bits: Int, group: Int,
                               f32Activations: Bool = false,
                               rows: Int = 1, isSHVQ: Bool = false) throws {
        guard let name = kernelName(bits: bits, group: group, f32Activations: f32Activations, isSHVQ: isSHVQ) else {
            throw QWGemmError.badWeights("no affine-quantised kernel for \(bits)-bit group \(group)")
        }
        guard k % group == 0, n > 0, rows > 0 else {
            throw QWGemmError.badWeights("k must be a multiple of \(group) and n positive")
        }
        guard let rowBytes = rowBytes(bits: bits, group: group, k: k, isSHVQ: isSHVQ) else {
            throw QWGemmError.badWeights("no affine-quantised size for \(bits)-bit group \(group)")
        }
        let needed = n * rowBytes
        guard blocks.bytes >= needed else {
            throw QWGemmError.badWeights("block bank holds \(blocks.bytes) bytes, \(needed) needed")
        }
        let wantX: QWDType = f32Activations ? .f32 : .f16
        guard x.shape.dtype == wantX else {
            throw QWGemmError.badWeights("activations must be \(wantX), got \(x.shape.dtype)")
        }
        guard x.shape.elementCount >= rows * k else {
            throw QWGemmError.badWeights("activations hold \(x.shape.elementCount), \(rows * k) needed")
        }
        guard y.shape.dtype == .f32 else {
            throw QWGemmError.badWeights("output must be f32, got \(y.shape.dtype)")
        }
        guard y.shape.elementCount >= rows * n else {
            throw QWGemmError.badWeights("output holds \(y.shape.elementCount), \(rows * n) needed")
        }
        let pso = try hal.pipeline(name)
        enc.setComputePipelineState(pso)
        blocks.bind(enc, index: 0)
        x.bind(enc, index: 1)
        y.bind(enc, index: 3)
        let useMultiRow = !f32Activations && rows >= 2 && rows <= 4 && n < (1 << 24)
        let useRows4 = !f32Activations && n < (1 << 30)
        var n32: UInt32
        if useMultiRow {
            n32 = UInt32(n) | 0x8000_0000 | (UInt32(rows) << 24)
        } else if useRows4 {
            n32 = UInt32(n) | 0x4000_0000
        } else {
            n32 = UInt32(n)
        }
        var k32 = UInt32(k)
        enc.setBytes(&n32, length: 4, index: 4)
        enc.setBytes(&k32, length: 4, index: 5)
        if useRows4 {
            let groups = (n + decodeRowsPerThreadgroup - 1) / decodeRowsPerThreadgroup
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: decodeThreadsPerThreadgroup,
                                                                   height: 1, depth: 1))
        } else {
            enc.dispatchThreadgroups(MTLSize(width: n, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
    }

    /// Encodes one decode GEMV over a bank of `(bits, group)` blocks, in the
    /// dense path's geometry: four output rows per simdgroup, f16
    /// activations in, f16 out. The geometry is the incumbent's `dx_h16_t4`.
    public static func encodeDecodeT4(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                      blocks: QWTensorValue, x: QWTensorValue,
                                      y16: QWTensorValue,
                                      n: Int, k: Int, bits: Int, group: Int,
                                      isSHVQ: Bool = false) throws {
        guard let name = decodeKernelName(bits: bits, group: group, isSHVQ: isSHVQ) else {
            throw QWGemmError.badWeights("no affine-quantised decode kernel for \(bits)-bit group \(group)")
        }
        guard k % group == 0, n > 0 else {
            throw QWGemmError.badWeights("k must be a multiple of \(group) and n positive")
        }
        // The row count rides in N's flags, so the count itself must leave
        // the top two bits alone: a bank of more than 2^30 rows would have
        // its count read back as a flag and the kernel would write the
        // wrong rows.
        guard n < (1 << 30) else {
            throw QWGemmError.badWeights("a bank of \(n) rows does not fit the row-count flags")
        }
        guard let rowBytes = rowBytes(bits: bits, group: group, k: k, isSHVQ: isSHVQ) else {
            throw QWGemmError.badWeights("no affine-quantised size for \(bits)-bit group \(group)")
        }
        let needed = n * rowBytes
        guard blocks.bytes >= needed else {
            throw QWGemmError.badWeights("block bank holds \(blocks.bytes) bytes, \(needed) needed")
        }
        guard x.shape.dtype == .f16 else {
            throw QWGemmError.badWeights("activations must be f16, got \(x.shape.dtype)")
        }
        guard x.shape.elementCount >= k else {
            throw QWGemmError.badWeights("activations hold \(x.shape.elementCount), \(k) needed")
        }
        guard y16.shape.dtype == .f16 else {
            throw QWGemmError.badWeights("output must be f16, got \(y16.shape.dtype)")
        }
        guard y16.shape.elementCount >= n else {
            throw QWGemmError.badWeights("output holds \(y16.shape.elementCount), \(n) needed")
        }
        let pso = try hal.pipeline(name)
        enc.setComputePipelineState(pso)
        blocks.bind(enc, index: 0)
        x.bind(enc, index: 1)
        y16.bind(enc, index: 3)
        var n32 = UInt32(n) | 0x4000_0000
        var k32 = UInt32(k)
        enc.setBytes(&n32, length: 4, index: 4)
        enc.setBytes(&k32, length: 4, index: 5)
        y16.bind(enc, index: 8)
        let groups = (n + decodeRowsPerThreadgroup - 1) / decodeRowsPerThreadgroup
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: decodeThreadsPerThreadgroup,
                                                               height: 1, depth: 1))
    }

    /// Encodes the head's fused GEMV-and-argmax over the family's blocks:
    /// the incumbent's `qwg_head_gemv_q4g_argmax` epilogue — the sortable key,
    /// the tiebreak, the threadgroup fold.
    public static func encodeHeadArgmax(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                         view: QWWeightView, x: QWTensorValue, y: QWTensorValue,
                                         keysHi: QWTensorValue, keysLo: QWTensorValue,
                                         writeLogits: Bool) throws {
        guard let name = headArgmaxKernelName(bits: view.bits, group: view.groupSize, isSHVQ: view.isSHVQ) else {
            throw QWGemmError.badWeights("no affine-quantised head kernel for "
                                         + "\(view.bits)-bit group \(view.groupSize)")
        }
        let n = view.n, k = view.k
        guard n > 0, k > 0, k % view.groupSize == 0 else {
            throw QWGemmError.badWeights("the head's rows and k are wrong")
        }
        // The row count rides in N's flags, so the count itself must leave
        // the top two bits alone.
        guard n < (1 << 30) else {
            throw QWGemmError.badWeights("a head of \(n) rows does not fit the row-count flags")
        }
        let slots = (n + 31) / 32
        guard x.shape.dtype == .f16, x.shape.elementCount >= k else {
            throw QWGemmError.badWeights("activations must be \(k) f16 values")
        }
        guard y.shape.dtype == .f32, y.shape.elementCount >= n else {
            throw QWGemmError.badWeights("the logits hold \(y.shape.elementCount), \(n) needed")
        }
        guard keysHi.shape.elementCount >= slots, keysLo.shape.elementCount >= slots else {
            throw QWGemmError.badWeights("the key scratch holds fewer than \(slots) slots")
        }
        var n32 = UInt32(n) | 0x8000_0000 | (writeLogits ? 0 : 0x4000_0000)
        var k32 = UInt32(k)
        let pso = try hal.pipeline(name)
        enc.setComputePipelineState(pso)
        view.blocks.bind(enc, index: 0)
        x.bind(enc, index: 1)
        y.bind(enc, index: 3)
        enc.setBytes(&n32, length: 4, index: 4)
        enc.setBytes(&k32, length: 4, index: 5)
        keysHi.bind(enc, index: 6)
        keysLo.bind(enc, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: slots, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}
