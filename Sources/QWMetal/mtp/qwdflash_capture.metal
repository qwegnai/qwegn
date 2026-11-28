// ── QWDFlash Capture — Target hidden capture and proposal staging kernels ────
// Captures 1-stream or HC-stream target residuals into FWHT-rotated FP16
// slots and stages BF16 token embeddings onto GPU proposal inputs.
#include <metal_stdlib>
#include "include/qw_dequant.metalh"
using namespace metal;

constant bool FC_DRAFTER_ROTATED [[function_constant(236)]];

struct DFlashCapH8 { half v[8]; } __attribute__((__packed__, __aligned__(16)));
static inline DFlashCapH8 asDFlashCapH8(uint4 w) { return *(thread DFlashCapH8*)&w; }
static inline float dflash_cap_bf16_to_f32(ushort b) { return as_type<float>(uint(b) << 16u); }

// Capture unrotated target residual [rows, H], rotate via fwht32,
// and scatter into [maxRows, totalSlots * H] at slot * hiddenU4.
kernel void qwg_dflash_capture_target_hidden(
        device const uint4* src      [[buffer(0)]],
        device uint4* dst            [[buffer(1)]],
        constant uint& srcRowOffset  [[buffer(2)]],
        constant uint& dstRowOffset  [[buffer(3)]],
        constant uint& slot          [[buffer(4)]],
        constant uint& hiddenU4      [[buffer(5)]],
        constant uint& totalSlots    [[buffer(6)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    const uint rowStrideU4 = totalSlots * hiddenU4;
    device const uint4* xr = src + (size_t)(srcRowOffset + tg) * hiddenU4;
    device uint4* yr = dst + (size_t)(dstRowOffset + tg) * rowStrideU4 + (size_t)slot * hiddenU4;
    for (uint u = ti; u < hiddenU4; u += 32u) {
        DFlashCapH8 hv = asDFlashCapH8(xr[u]);
        if (FC_DRAFTER_ROTATED) { fwht32_inplace(hv.v, ti); }
        yr[u] = *(thread uint4*)&hv;
    }
}

// Capture hc-stream FP32 residuals into targetHidden[dstRow, slot * hiddenU4].
// flags == 0: mean(base); flags & 1: FlashNext deferred f32 inject;
// flags & 2: mHC deferred f16 merge with controls [hc + hc*hc].
kernel void qwg_dflash_capture_hc_hidden(
        device const float* base     [[buffer(0)]],
        device const void* src       [[buffer(1)]],
        device const float* inj      [[buffer(2)]],
        device uint4* dst            [[buffer(3)]],
        constant uint& srcRowOffset  [[buffer(4)]],
        constant uint& dstRowOffset  [[buffer(5)]],
        constant uint& slot          [[buffer(6)]],
        constant uint& hiddenU4      [[buffer(7)]],
        constant uint& totalSlots    [[buffer(8)]],
        constant uint& hc            [[buffer(9)]],
        constant uint& flags         [[buffer(10)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    const uint hidden = hiddenU4 << 3u;
    const uint sRow = srcRowOffset + tg;
    const uint dRow = dstRowOffset + tg;
    const float invHC = 1.0f / float(hc);
    device const float* rowBase = base + (size_t)sRow * hc * hidden;
    device uint4* yr = dst + (size_t)dRow * totalSlots * hiddenU4 + (size_t)slot * hiddenU4;
    for (uint u = ti; u < hiddenU4; u += 32u) {
        float f[8];
        if ((flags & 2u) != 0u) {
            DFlashCapH8 bv = asDFlashCapH8(*((device const uint4*)src + (size_t)sRow * hiddenU4 + u));
            device const float* ctrl = inj + (size_t)sRow * (hc + hc * hc);
            const uint hBase = u << 3u;
            for (int h = 0; h < 8; ++h) {
                float b = float(bv.v[h]);
                float sumS = 0.0f;
                for (uint i = 0u; i < hc; ++i) {
                    float ns = ctrl[i] * b;
                    for (uint j = 0u; j < hc; ++j) {
                        ns = fma(ctrl[hc + i * hc + j], rowBase[(size_t)j * hidden + hBase + uint(h)], ns);
                    }
                    sumS += ns;
                }
                f[h] = sumS * invHC;
            }
        } else {
            float4 sum0 = float4(0.0f), sum1 = float4(0.0f);
            if ((flags & 1u) != 0u) {
                device const float4* sr = (device const float4*)((device const float*)src + (size_t)sRow * hidden) + u * 2u;
                float4 sv0 = sr[0], sv1 = sr[1];
                device const float* injRow = inj + (size_t)sRow * hc;
                for (uint c = 0u; c < hc; ++c) {
                    float sc = injRow[c];
                    device const float4* bc = (device const float4*)(rowBase + (size_t)c * hidden) + u * 2u;
                    sum0 += bc[0] + sv0 * sc;
                    sum1 += bc[1] + sv1 * sc;
                }
            } else {
                for (uint c = 0u; c < hc; ++c) {
                    device const float4* bc = (device const float4*)(rowBase + (size_t)c * hidden) + u * 2u;
                    sum0 += bc[0];
                    sum1 += bc[1];
                }
            }
            sum0 *= invHC;
            sum1 *= invHC;
            f[0] = sum0.x; f[1] = sum0.y; f[2] = sum0.z; f[3] = sum0.w;
            f[4] = sum1.x; f[5] = sum1.y; f[6] = sum1.z; f[7] = sum1.w;
        }
        DFlashCapH8 rv;
        if (FC_DRAFTER_ROTATED) {
            fwht32_f32_to_h8(f, rv.v, ti);
        } else {
            for (int h = 0; h < 8; ++h) rv.v[h] = half(f[h]);
        }
        yr[u] = *(thread uint4*)&rv;
    }
}

// Stage anchorToken (row 0) and maskToken (rows >= 1) from BF16 [vocab, hidden] table into FP32 propX.
kernel void qwg_dflash_stage_prop_bf16(
        device const ushort* table   [[buffer(0)]],
        device float* propX          [[buffer(1)]],
        constant int& anchorToken    [[buffer(2)]],
        constant int& maskToken      [[buffer(3)]],
        constant uint& hidden        [[buffer(4)]],
        constant uint& vocab         [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint h = gid.x;
    const uint row = gid.y;
    if (h >= hidden || vocab == 0u) return;
    const int rawTok = (row == 0u) ? anchorToken : maskToken;
    const uint tok = (rawTok >= 0 && rawTok < int(vocab) && rawTok != 0x7fffffff) ? uint(rawTok) : 0u;
    propX[(size_t)row * hidden + h] = dflash_cap_bf16_to_f32(table[(size_t)tok * hidden + h]);
}
