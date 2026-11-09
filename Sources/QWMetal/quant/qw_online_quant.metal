// qw_online_quant.metal — the load-time GPU writers.
//
//   qwg_quantize_bf16_to_hq4_g32_m5  — BF16 weights -> Block_HQ4_G32 rows.
//   qwg_repack_mlx_affine_to_aq_m5    — an MLX affine checkpoint -> our plane.
//
// Both write into a caller-supplied buffer; no host round-trip. The formats
// are in specs/04-qwquant.md.

#include <metal_stdlib>
#include "quant/qw_quant_decls.metalh"
using namespace metal;

/// Quantise a BF16 weight matrix into Block_HQ4_G32 rows.
/// `src` is `rows * cols` BF16 values, row-major; `dst` is
/// `rows * (cols/32) * 20` bytes — the incumbent packer's bytes.
kernel void qwg_quantize_bf16_to_hq4_g32_m5(device const ushort* src [[buffer(0)]],
                                              device char* dst [[buffer(1)]],
                                              constant uint& rows [[buffer(4)]],
                                              constant uint& cols [[buffer(5)]],
                                              uint ti [[thread_position_in_grid]],
                                              uint tptg [[threads_per_grid]]) {
    const uint groups = cols / QW_HQ4_GROUP;
    const uint blocks = rows * groups;
    for (uint b = ti; b < blocks; b += tptg) {
        const uint r = b / groups;
        const uint g = b % groups;
        float v[QW_HQ4_GROUP];
        device const ushort* s = src + (size_t)r * cols + g * QW_HQ4_GROUP;
        for (uint i = 0; i < QW_HQ4_GROUP; ++i) { v[i] = qwg_bf16_to_f32(s[i]); }
        qwg_fwht32_ortho_f32(v);

        float amax = 0.0f;
        for (uint i = 0; i < QW_HQ4_GROUP; ++i) { amax = max(amax, abs(v[i])); }
        half scale;
        half bias;
        qwg_hq4_scale_bias(amax, scale, bias);
        const float exactScale = float(scale);
        const float exactBias = float(bias);
        const float safeScale = exactScale > 0.0f ? exactScale : 1.0f;

        uint codes[QW_HQ4_GROUP];
        for (uint i = 0; i < QW_HQ4_GROUP; ++i) {
            const float f = floor((v[i] - exactBias) / safeScale - 1023.5f);
            codes[i] = exactScale > 0.0f ? uint(clamp(f, 0.0f, 15.0f)) : 8u;
        }

        device char* out = dst + (size_t)b * QW_HQ4_BLOCK_BYTES;
        *(device half*)out = scale;
        *(device half*)(out + 2) = bias;
        for (uint w = 0; w < 4; ++w) {
            *(device uint*)(out + 4 + w * 4) = qwg_hq4_pack_word(codes + w * 8);
        }
    }
}

/// Reads one f16 or bf16 value, in the dtype the checkpoint used.
METAL_FUNC float qwg_read_f16_or_bf16(device const ushort* p, uint dtype, uint i) {
    const ushort bits = p[i];
    return (dtype == 0u) ? qwg_bf16_to_f32(bits) : float(as_type<half>(bits));
}

/// True when `stored = half(raw)` holds `raw` without overflow or normal-range loss.
/// In f16's subnormal range (`|raw| < 2^-14 = 6.103515625e-5f`), a bf16 header
/// rounds to the nearest 2^-24 subnormal step (~3e-8) while remaining finite.
METAL_FUNC bool qwg_f16_header_holds(float raw, half stored) {
    if (float(stored) == raw) return true;
    return isfinite(raw) && isfinite(float(stored)) && fabs(raw) < 6.103515625e-5f;
}

/// Repack an MLX affine checkpoint into our one plane, at the same byte
/// count. `scaleDType` is 0 for bf16 and 1 for f16. `flags` is output-only:
/// the caller zeroes it and reads the count. See specs/04-qwquant.md.
kernel void qwg_repack_mlx_affine_to_aq_m5(device const uchar* mlxW [[buffer(0)]],
                                          device const ushort* mlxScales [[buffer(1)]],
                                          device const ushort* mlxBiases [[buffer(2)]],
                                          device char* dst [[buffer(3)]],
                                          constant uint& rows [[buffer(4)]],
                                          constant uint& cols [[buffer(5)]],
                                          constant uint& bits [[buffer(6)]],
                                          constant uint& group [[buffer(7)]],
                                          constant uint& scaleDType [[buffer(8)]],
                                          device atomic_uint* flags [[buffer(9)]],
                                          uint ti [[thread_position_in_grid]],
                                          uint tptg [[threads_per_grid]]) {
    const uint groups = cols / group;
    const uint payload = group * bits / 8;
    const uint groupBytes = payload + 4;
    const uint total = rows * groups;
    // The f16 NaN, as a bit pattern: a poisoned header is not a number.
    const half poison = as_type<half>(ushort(0x7e00u));

    for (uint b = ti; b < total; b += tptg) {
        const uint r = b / groups;
        const uint g = b % groups;
        const uint groupId = r * groups + g;
        device const uchar* src = mlxW + (size_t)r * groups * payload + (size_t)g * payload;
        device char* out = dst + (size_t)b * groupBytes;

        const float rawScale = qwg_read_f16_or_bf16(mlxScales, scaleDType, groupId);
        const float rawBias = qwg_read_f16_or_bf16(mlxBiases, scaleDType, groupId);
        const half storedScale = half(rawScale);
        const half storedBias = half(rawBias);
        const bool exact = qwg_f16_header_holds(rawScale, storedScale)
                        && qwg_f16_header_holds(rawBias, storedBias);
        if (!exact) {
            atomic_fetch_add_explicit(flags, 1u, memory_order_relaxed);
        }
        *(device half*)out = exact ? storedScale : poison;
        *(device half*)(out + 2) = exact ? storedBias : poison;

        if ((payload & 3u) == 0u) {
            device const uint* src32 = (device const uint*)src;
            device uint* dst32 = (device uint*)(out + 4);
            const uint words = payload >> 2u;
            for (uint i = 0; i < words; ++i) { dst32[i] = src32[i]; }
        } else {
            device uchar* d = (device uchar*)(out + 4);
            for (uint i = 0; i < payload; ++i) { d[i] = src[i]; }
        }
    }
}

/// Repack 3D MLX switch_mlp `gate_proj` [experts, inter, cols] and `up_proj` [experts, inter, cols]
/// directly into an interleaved [experts, 2 * inter, cols] AQ{bits}_G{group} bank.
kernel void qwg_repack_mlx_affine_switch_gate_up_to_aq_m5(
    device const uchar* gateW [[buffer(0)]],
    device const ushort* gateScales [[buffer(1)]],
    device const ushort* gateBiases [[buffer(2)]],
    device const uchar* upW [[buffer(3)]],
    device const ushort* upScales [[buffer(4)]],
    device const ushort* upBiases [[buffer(5)]],
    device char* dst [[buffer(6)]],
    constant uint& experts [[buffer(7)]],
    constant uint& inter [[buffer(8)]],
    constant uint& cols [[buffer(9)]],
    constant uint& bits [[buffer(10)]],
    constant uint& group [[buffer(11)]],
    constant uint& scaleDType [[buffer(12)]],
    device atomic_uint* flags [[buffer(13)]],
    uint ti [[thread_position_in_grid]],
    uint tptg [[threads_per_grid]]) {
    const uint groups = cols / group;
    const uint payload = group * bits / 8;
    const uint groupBytes = payload + 4;
    const uint outRowsPerExpert = 2u * inter;
    const uint totalRows = experts * outRowsPerExpert;
    const uint total = totalRows * groups;
    const half poison = as_type<half>(ushort(0x7e00u));

    for (uint b = ti; b < total; b += tptg) {
        const uint outRow = b / groups;
        const uint g = b % groups;
        const uint e = outRow / outRowsPerExpert;
        const uint localRow = outRow % outRowsPerExpert;
        const bool isUp = localRow >= inter;
        const uint srcRow = e * inter + (isUp ? (localRow - inter) : localRow);
        const uint srcGroup = srcRow * groups + g;

        device const uchar* wBase = isUp ? upW : gateW;
        device const ushort* sBase = isUp ? upScales : gateScales;
        device const ushort* bBase = isUp ? upBiases : gateBiases;

        device const uchar* src = wBase + (size_t)srcGroup * payload;
        device char* out = dst + (size_t)b * groupBytes;

        const float rawScale = qwg_read_f16_or_bf16(sBase, scaleDType, srcGroup);
        const float rawBias = qwg_read_f16_or_bf16(bBase, scaleDType, srcGroup);
        const half storedScale = half(rawScale);
        const half storedBias = half(rawBias);
        const bool exact = qwg_f16_header_holds(rawScale, storedScale)
                        && qwg_f16_header_holds(rawBias, storedBias);
        if (!exact) {
            atomic_fetch_add_explicit(flags, 1u, memory_order_relaxed);
        }
        *(device half*)out = exact ? storedScale : poison;
        *(device half*)(out + 2) = exact ? storedBias : poison;

        if ((payload & 3u) == 0u) {
            device const uint* src32 = (device const uint*)src;
            device uint* dst32 = (device uint*)(out + 4);
            const uint words = payload >> 2u;
            for (uint i = 0; i < words; ++i) { dst32[i] = src32[i]; }
        } else {
            device uchar* d = (device uchar*)(out + 4);
            for (uint i = 0; i < payload; ++i) { d[i] = src[i]; }
        }
    }
}
