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
    public static func kernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_gemv_shvq\(bits)_g\(group)_v4" : "qwg_gemv_aq\(bits)_g\(group)_v4"
    }

    /// The decode kernel for a `(bits, group)` pair: four output rows per simdgroup, f16 in and f16 out.
    public static func decodeKernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_gemv_shvq\(bits)_g\(group)_dx_h16_t4" : "qwg_gemv_aq\(bits)_g\(group)_dx_h16_t4"
    }

    /// Low-register M<=4 SHVQ decode kernel specialization for `(bits, group)`.
    public static func decodeM4KernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSHVQ, isSupported(bits: bits, group: group, isSHVQ: true) else { return nil }
        return "qwg_gemv_shvq\(bits)_g\(group)_dx_h16_t4_m4"
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

    /// Low-register M<=4 SHVQ fused GEMV-and-argmax kernel specialization for `(bits, group)`.
    public static func headArgmaxM4KernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSHVQ, isSupported(bits: bits, group: group, isSHVQ: true) else { return nil }
        return "qwg_head_gemv_shvq\(bits)_g\(group)_argmax_m4"
    }

    /// Single-pass M=9..16 SHVQ fused GEMV-and-argmax kernel specialization for `(bits, group)`.
    public static func headArgmaxM16KernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSHVQ, isSupported(bits: bits, group: group, isSHVQ: true) else { return nil }
        return "qwg_head_gemv_shvq\(bits)_g\(group)_argmax_m16"
    }

    /// The prefill GEMM for a `(bits, group)` pair (`h16` NAX on Apple10+, `sg` on Apple9).
    public static func gemmKernelName(bits: Int, group: Int, isSHVQ: Bool = false, useNAX: Bool = true) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        let suffix = useNAX ? "h16" : "sg"
        return isSHVQ ? "qwg_gemm_shvq\(bits)_g\(group)_\(suffix)" : "qwg_gemm_aq\(bits)_g\(group)_\(suffix)"
    }

    /// The fused gate_up + SwiGLU + FWHT-32 prefill GEMM for a `(bits, group)` pair.
    public static func swigluGemmKernelName(bits: Int, group: Int, isSHVQ: Bool = false, useNAX: Bool = true) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        let suffix = useNAX ? "h16_swiglu" : "sg_swiglu"
        return isSHVQ ? "qwg_gemm_shvq\(bits)_g\(group)_\(suffix)" : "qwg_gemm_aq\(bits)_g\(group)_\(suffix)"
    }

    /// The GDN in-projection decode kernel for a `(bits, group)` pair.
    public static func gdnInProjDecodeKernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_gemv_shvq\(bits)_g\(group)_dx_h16_gdn_inproj_t4" : "qwg_gemv_aq\(bits)_g\(group)_dx_h16_gdn_inproj_t4"
    }

    /// The GDN in-projection prefill kernel for a `(bits, group)` pair.
    public static func gdnInProjPrefillKernelName(bits: Int, group: Int, isSHVQ: Bool = false, useNAX: Bool = true) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        let suffix = useNAX ? "h16_gdn_inproj" : "sg_gdn_inproj"
        return isSHVQ ? "qwg_gemm_shvq\(bits)_g\(group)_\(suffix)" : "qwg_gemm_aq\(bits)_g\(group)_\(suffix)"
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

    public static func naxF32KernelName(bits: Int, group: Int, isSHVQ: Bool = false, useNAX: Bool = true) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        let suffix = useNAX ? "nax_f32" : "sg_f32"
        return isSHVQ ? "qwg_gemm_shvq\(bits)_g\(group)_\(suffix)" : "qwg_gemm_aq\(bits)_g\(group)_\(suffix)"
    }

    public static func naxTG16F32KernelName(bits: Int, group: Int, isSHVQ: Bool) -> String? {
        guard isSHVQ, bits == 4, isSupported(bits: bits, group: group, isSHVQ: true) else { return nil }
        return "qwg_gemm_shvq4_g\(group)_nax_tg16_f32"
    }

    public static func moeGateUpF32R1KernelName(bits: Int, group: Int, isSHVQ: Bool = false, hasSoftcap: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        let suffix = hasSoftcap ? "_sc" : ""
        return isSHVQ ? "qwg_moe_gateup_shvq\(bits)_g\(group)_f32_r1\(suffix)"
            : "qwg_moe_gateup_aq\(bits)_g\(group)_f32_r1\(suffix)"
    }

    public static func moeDownSharedCombineF32KernelName(bits: Int, group: Int, isSHVQ: Bool = false) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        return isSHVQ ? "qwg_moe_down_shared_combine_shvq\(bits)_g\(group)_f32" : "qwg_moe_down_shared_combine_aq\(bits)_g\(group)_f32"
    }

    public static func moeNaxGemmKernelName(bits: Int, group: Int, isSHVQ: Bool = false, useNAX: Bool = true) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        let tag = useNAX ? "nax" : "sg"
        return isSHVQ ? "qwg_moe_\(tag)_gemm_shvq\(bits)_g\(group)" : "qwg_moe_\(tag)_gemm_aq\(bits)_g\(group)"
    }

    public static func moeNaxSplitF32KernelName(bits: Int, group: Int, isSHVQ: Bool = false, useNAX: Bool = true) -> String? {
        guard isSupported(bits: bits, group: group, isSHVQ: isSHVQ) else { return nil }
        let tag = useNAX ? "nax" : "sg"
        return isSHVQ ? "qwg_moe_\(tag)_split_shvq\(bits)_g\(group)_f32" : "qwg_moe_\(tag)_split_aq\(bits)_g\(group)_f32"
    }

    /// The fused grouped-MoE gate_up + SwiGLU + FWHT-32 kernel for a `(bits, group)` pair.
    public static func moeNaxSwigluKernelName(bits: Int, group: Int, isSHVQ: Bool = false, useNAX: Bool = true) -> String? {
        guard group == 32 || group == 64 else { return nil }
        let tag = useNAX ? "nax" : "sg"
        if isSHVQ {
            guard QWSHVQBlock.isSupported(bits: bits, group: group) else { return nil }
            return "qwg_moe_\(tag)_swiglu_shvq\(bits)_g\(group)"
        }
        guard bits == 4 || bits == 8, QWAQBlock.isBlockSupported(bits: bits, group: group) else { return nil }
        return "qwg_moe_\(tag)_swiglu_aq\(bits)_g\(group)"
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
                               rows: Int = 1, isSHVQ: Bool = false) throws {
        guard let name = kernelName(bits: bits, group: group, isSHVQ: isSHVQ) else {
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
        guard x.shape.dtype == .f16 else {
            throw QWGemmError.badWeights("activations must be f16, got \(x.shape.dtype)")
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
        guard n < (rows > 1 ? (1 << 24) : (1 << 30)) else {
            throw QWGemmError.badWeights("a bank of \(n) rows does not fit row-count flags")
        }
        let pso = try hal.pipeline(name)
        enc.setComputePipelineState(pso)
        blocks.bind(enc, index: 0)
        var k32 = UInt32(k)
        enc.setBytes(&k32, length: 4, index: 5)
        var rowOffset = 0
        while rowOffset < rows {
            let chunk = min(8, rows - rowOffset)
            let xChunk = (rowOffset == 0 && chunk == rows)
                ? x
                : x.slice(byteOffset: rowOffset * k * 2, count: chunk * k, dtype: .f16)
            let yChunk = (rowOffset == 0 && chunk == rows)
                ? y
                : y.slice(byteOffset: rowOffset * n * 4, count: chunk * n, dtype: .f32)
            xChunk.bind(enc, index: 1)
            yChunk.bind(enc, index: 3)
            var n32 = chunk >= 2
                ? (UInt32(n) | 0x8000_0000 | (UInt32(chunk) << 24))
                : (UInt32(n) | 0x4000_0000)
            enc.setBytes(&n32, length: 4, index: 4)
            let groups = (n + decodeRowsPerThreadgroup - 1) / decodeRowsPerThreadgroup
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: decodeThreadsPerThreadgroup,
                                                                   height: 1, depth: 1))
            rowOffset += chunk
        }
    }

    /// Encodes 1+ decode GEMV over a bank of `(bits, group)` blocks, in the
    /// dense path's geometry: four (or two for SHVQ rows=1 N<=8192 and rows=5..8)
    /// output rows per simdgroup, f16 activations in, f16 out.
    public static func encodeDecodeT4(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                      blocks: QWTensorValue, x: QWTensorValue,
                                      y16: QWTensorValue,
                                      n: Int, k: Int, bits: Int, group: Int,
                                      rows: Int = 1, isSHVQ: Bool = false) throws {
        guard let name = decodeKernelName(bits: bits, group: group, isSHVQ: isSHVQ) else {
            throw QWGemmError.badWeights("no affine-quantised decode kernel for \(bits)-bit group \(group)")
        }
        guard k % group == 0, n > 0, rows > 0 else {
            throw QWGemmError.badWeights("k must be a multiple of \(group) and n and rows positive")
        }
        // The row-count encoding reserves bits 24..29 for multi-row alongside bit 30,
        // so rows >= 2^24 would collide with mRows in dx_h16_t4.
        guard n < (1 << 24) else {
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
        guard x.shape.elementCount >= rows * k else {
            throw QWGemmError.badWeights("activations hold \(x.shape.elementCount), \(rows * k) needed")
        }
        guard y16.shape.dtype == .f16 else {
            throw QWGemmError.badWeights("output must be f16, got \(y16.shape.dtype)")
        }
        guard y16.shape.elementCount >= rows * n else {
            throw QWGemmError.badWeights("output holds \(y16.shape.elementCount), \(rows * n) needed")
        }
        let pso = try hal.pipeline(name)
        let psoM4 = (isSHVQ && rows >= 2) ? decodeM4KernelName(bits: bits, group: group, isSHVQ: true)
            .flatMap { try? hal.pipeline($0) } : nil
        var activeIsM4: Bool? = nil
        blocks.bind(enc, index: 0)
        var k32 = UInt32(k)
        enc.setBytes(&k32, length: 4, index: 5)
        var rowOffset = 0
        while rowOffset < rows {
            let rem = rows - rowOffset
            let chunk = (rem > 8 && rem - 8 == 1) ? 5 : min(8, rem)
            let useM4 = isSHVQ && (chunk >= 2 && chunk <= 4) && (psoM4 != nil)
            let useTwoRows = isSHVQ && ((chunk == 1 && n <= 8192) || (chunk >= 5 && chunk <= 8))
            if useM4 != activeIsM4 {
                if let m4 = psoM4, useM4 {
                    enc.setComputePipelineState(m4)
                } else {
                    enc.setComputePipelineState(pso)
                }
                activeIsM4 = useM4
            }
            let xChunk = (rowOffset == 0 && chunk == rows)
                ? x
                : x.slice(byteOffset: rowOffset * k * 2, count: chunk * k, dtype: .f16)
            let yChunk = (rowOffset == 0 && chunk == rows)
                ? y16
                : y16.slice(byteOffset: rowOffset * n * 2, count: chunk * n, dtype: .f16)
            xChunk.bind(enc, index: 1)
            yChunk.bind(enc, index: 3)
            yChunk.bind(enc, index: 8)
            var n32: UInt32
            if chunk == 1 {
                n32 = UInt32(n) | (useTwoRows ? 0x8000_0000 : 0x4000_0000)
            } else {
                n32 = UInt32(n) | (useTwoRows ? 0xC000_0000 : 0x4000_0000) | (UInt32(chunk) << 24)
            }
            enc.setBytes(&n32, length: 4, index: 4)
            let rowsPerTG = (!useM4 && useTwoRows) ? (decodeRowsPerThreadgroup / 2) : decodeRowsPerThreadgroup
            let groups = (n + rowsPerTG - 1) / rowsPerTG
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: decodeThreadsPerThreadgroup,
                                                                   height: 1, depth: 1))
            rowOffset += chunk
        }
    }

    /// Encodes 1..8-row SHVQ decode GEMV writing directly to `.bf16` output (`bfloat(float)` single rounding).
    public static func encodeDecodeBF16(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                        blocks: QWTensorValue, x: QWTensorValue,
                                        yBF16: QWTensorValue,
                                        n: Int, k: Int, bits: Int, group: Int,
                                        rows: Int = 1, isSHVQ: Bool = false) throws {
        guard isSHVQ else {
            throw QWGemmError.badWeights("encodeDecodeBF16 requires SHVQ blocks")
        }
        guard let name = decodeKernelName(bits: bits, group: group, isSHVQ: true) else {
            throw QWGemmError.badWeights("no SHVQ decode kernel for \(bits)-bit group \(group)")
        }
        guard k % group == 0, n > 0, rows > 0 else {
            throw QWGemmError.badWeights("k must be a multiple of \(group), n and rows positive")
        }
        guard n < (1 << 24) else {
            throw QWGemmError.badWeights("a bank of \(n) rows does not fit row-count flags")
        }
        guard let rowBytes = rowBytes(bits: bits, group: group, k: k, isSHVQ: true) else {
            throw QWGemmError.badWeights("no SHVQ size for \(bits)-bit group \(group)")
        }
        guard blocks.bytes >= n * rowBytes else {
            throw QWGemmError.badWeights("block bank holds \(blocks.bytes) bytes, \(n * rowBytes) needed")
        }
        guard x.shape.dtype == .f16, x.shape.elementCount >= rows * k else {
            throw QWGemmError.badWeights("activations must be f16 with >= \(rows * k) elements")
        }
        guard yBF16.shape.dtype == .bf16, yBF16.shape.elementCount >= rows * n else {
            throw QWGemmError.badWeights("output must be bf16 with >= \(rows * n) elements")
        }
        let pso = try hal.pipeline(name)
        let psoM4 = (rows >= 2) ? decodeM4KernelName(bits: bits, group: group, isSHVQ: true)
            .flatMap { try? hal.pipeline($0) } : nil
        var activeIsM4: Bool? = nil
        blocks.bind(enc, index: 0)
        var k32 = UInt32(k)
        enc.setBytes(&k32, length: 4, index: 5)
        var rowOffset = 0
        while rowOffset < rows {
            let rem = rows - rowOffset
            let chunk = (rem > 8 && rem - 8 == 1) ? 5 : min(8, rem)
            let useM4 = (chunk >= 2 && chunk <= 4) && (psoM4 != nil)
            let useTwoRows = (chunk == 1 && n <= 8192) || (chunk >= 5 && chunk <= 8)
            if useM4 != activeIsM4 {
                if let m4 = psoM4, useM4 {
                    enc.setComputePipelineState(m4)
                } else {
                    enc.setComputePipelineState(pso)
                }
                activeIsM4 = useM4
            }
            let xChunk = (rowOffset == 0 && chunk == rows)
                ? x
                : x.slice(byteOffset: rowOffset * k * 2, count: chunk * k, dtype: .f16)
            let yChunk = (rowOffset == 0 && chunk == rows)
                ? yBF16
                : yBF16.slice(byteOffset: rowOffset * n * 2, count: chunk * n, dtype: .bf16)
            xChunk.bind(enc, index: 1)
            yChunk.bind(enc, index: 3)
            yChunk.bind(enc, index: 8)
            var n32 = UInt32(n)
                | (useTwoRows ? 0xE000_0000 : 0x6000_0000)
                | (chunk >= 2 ? (UInt32(chunk) << 24) : 0)
            enc.setBytes(&n32, length: 4, index: 4)
            let rowsPerTG = (!useM4 && useTwoRows) ? (decodeRowsPerThreadgroup / 2) : decodeRowsPerThreadgroup
            let groups = (n + rowsPerTG - 1) / rowsPerTG
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: decodeThreadsPerThreadgroup,
                                                                   height: 1, depth: 1))
            rowOffset += chunk
        }
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

extension QWGemm {
    public static func moeNaxSwigluKernelName(for bank: QWWeightView, useNAX: Bool = true) -> String? {
        bank.isIncumbentContainer
            ? (useNAX ? "qwg_moe_nax_swiglu_q4g" : "qwg_moe_sg_swiglu_q4g")
            : QWGemmAQ.moeNaxSwigluKernelName(bits: bank.bits, group: bank.groupSize, isSHVQ: bank.isSHVQ, useNAX: useNAX)
    }

    public static func supportsMoEGateUpNaxSwiglu(_ bank: QWWeightView) -> Bool {
        moeNaxSwigluKernelName(for: bank) != nil
    }

    /// Fused grouped-MoE gate_up + SwiGLU + FWHT-32 emitting rotated FP16 directly into `out16`.
    public static func encodeMoEGateUpNaxSwiglu(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal, xe: QWTensorValue,
        bank: QWWeightView, expertOff: QWTensorValue, tileE: QWTensorValue,
        tileR: QWTensorValue, tileCount: QWTensorValue, out16: QWTensorValue,
        intermediate: Int, tilePlan: MoETilePlan,
        sortedTok: QWTensorValue? = nil,
        coldBank: QWWeightView? = nil,
        tileFilter: MoETileFilter = .all,
        swigluClamp: QWSwiGLUClampSpec? = nil
    ) throws {
        precondition(intermediate > 0 && intermediate % 32 == 0 && bank.k % 64 == 0,
                     "NAX MoE SwiGLU needs I multiple of 32 and K multiple of 64")
        let rowsPerExpert = 2 * intermediate
        guard tileCount.shape.elementCount >= 4 + tilePlan.guSlices * 3 else {
            throw QWGemmError.badShape("moe gateup swiglu nax: tileCount buffer too small for indirect slices")
        }
        if let cg = coldBank, cg.containerKind != bank.containerKind || cg.k != bank.k {
            throw QWGemmError.badWeights("moe gateup swiglu nax coldBank container/k mismatch")
        }
        guard let kernelName = moeNaxSwigluKernelName(for: bank, useNAX: hal.caps.isMetal4TensorCapable) else {
            throw QWGemmError.badWeights("no moe nax swiglu kernel for \(bank.formatName)")
        }
        let residentE = bank.n / rowsPerExpert
        let coldE = (coldBank?.n ?? 0) / rowsPerExpert
        let experts = residentE + coldE
        let packedExperts = coldBank != nil
            ? ((UInt32(residentE & 0xFFFF) << 16) | UInt32(experts & 0xFFFF))
            : UInt32(experts)
        let colTiles = intermediate / 32
        let grp = QWNaxGroupedParams(experts: packedExperts, n: UInt32(rowsPerExpert), k: UInt32(bank.k),
                                     blocksPerRow: UInt32(bank.k / max(1, bank.groupSize)),
                                     rowTiles: 0, colTiles: UInt32(colTiles))
        enc.setComputePipelineState(try hal.pipeline(kernelName, swigluClamp: swigluClamp))
        enc.bindTensors(bank.blocks, xe, out16, expertOff, tileE, tileR, tileCount, startIndex: 0)
        enc.setPod(grp, index: 7)
        enc.bindTensors(sortedTok ?? tileE, startIndex: 9)
        enc.setUInt32s(sortedTok != nil ? 1 : 0, startIndex: 10)
        (coldBank ?? bank).blocks.bind(enc, index: 11)
        dispatchTileChunks(enc, indirectBuffer: tileCount.region.buffer,
                           baseByteOffset: tileCount.region.base + tileCount.offset + 16,
                           colTiles: colTiles, maxTiles: tilePlan.maxTiles,
                           flags: tileFilter.rawValue, flagsIndex: 8)
    }

    public static func denseSwiGLUNaxKernelName(for weight: QWWeightView, useNAX: Bool = true) -> String? {
        weight.isIncumbentContainer
            ? (useNAX ? "qwg_gemm_q4g_h16_swiglu" : "qwg_gemm_q4g_sg_swiglu")
            : QWGemmAQ.swigluGemmKernelName(bits: weight.bits, group: weight.groupSize, isSHVQ: weight.isSHVQ, useNAX: useNAX)
    }

    public static func supportsDenseSwiGLUNax(_ weight: QWWeightView) -> Bool {
        denseSwiGLUNaxKernelName(for: weight) != nil
    }

    /// Fused dense gate_up + SwiGLU + FWHT-32 emitting rotated FP16 directly into `out16`.
    public static func encodeDenseSwiGLUNax(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        weight: QWWeightView, x16: QWTensorValue, out16: QWTensorValue,
        rows: Int, intermediate: Int,
        xStride: Int? = nil, yStride: Int? = nil,
        swigluClamp: QWSwiGLUClampSpec? = nil
    ) throws {
        guard rows > 0, intermediate > 0, intermediate % 32 == 0,
              weight.k > 0, weight.k % 64 == 0, weight.n == 2 * intermediate else {
            throw QWGemmError.badShape(
                "dense swiglu nax invalid shape M=\(rows) I=\(intermediate) w=(\(weight.n)x\(weight.k))")
        }
        guard x16.shape.dtype == .f16, out16.shape.dtype == .f16 else {
            throw QWGemmError.badShape("dense swiglu nax requires f16 input and output")
        }
        let useNAX = hal.caps.isMetal4TensorCapable
        guard let kernelName = denseSwiGLUNaxKernelName(for: weight, useNAX: useNAX) else {
            throw QWGemmError.badWeights("no dense swiglu nax kernel for \(weight.formatName)")
        }
        enc.setComputePipelineState(try hal.pipeline(kernelName, swigluClamp: swigluClamp))
        enc.bindTensors(weight.blocks, x16, out16, startIndex: 0)
        let xs = xStride ?? weight.k, ys = yStride ?? intermediate
        let chunk = naxRowChunk(rows: rows, k: weight.k, gpuCores: hal.caps.gpuCores)
        var r0 = 0
        while r0 < rows {
            let r = min(chunk, rows - r0)
            let use64 = useNAX && r <= 16 && (intermediate & 63) == 0
            let bm = r <= 16 ? 16 : (r <= 32 ? 32 : 64)
            let bn = use64 ? 64 : 32
            let modeBits = r <= 16 ? 0x2000_0000 : (r <= 32 ? 0x4000_0000 : 0)
            if r0 > 0 {
                x16.slice(byteOffset: r0 * xs * 2, count: r * xs).bind(enc, index: 1)
                out16.slice(byteOffset: r0 * ys * 2, count: r * ys).bind(enc, index: 2)
            }
            enc.setUInt32s(r | modeBits, intermediate, weight.k, xs, ys, startIndex: 3)
            enc.dispatchThreadgroups(MTLSize(width: intermediate / bn, height: (r + bm - 1) / bm, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            r0 += r
        }
    }
}
