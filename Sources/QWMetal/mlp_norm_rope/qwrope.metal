// ── C07 QWRope — fused per-head norm + partial rope (spec specs/07-qwrope.md)
// ONE launch covers q AND k (spec: "≤ 2 launches total for norm+rope on q,k").
// grid = (S, 28): x = token, y = head row (0..23 q, 24..27 k). q rows use the
// Design note: docs/KERNEL_NOTES.md#qwrope.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_geometry_constants.metalh"
using namespace metal;

struct H8 { half v[8]; } __attribute__((__packed__, __aligned__(16)));
static inline H8 asH8(uint4 w) { return *(thread H8*)&w; }

kernel void qwg_rope_norm_qk(device const uint4* qIn      [[buffer(0)]],
                             device const uint4* kIn      [[buffer(1)]],
                             device uint4* qOut           [[buffer(2)]],
                             device uint4* kOut           [[buffer(3)]],
                             device const uint4* wq       [[buffer(4)]],
                             device const uint4* wk       [[buffer(5)]],
                             device const float* table    [[buffer(6)]],
                             device uint4* normOut        [[buffer(7)]],   // nullable
                             constant uint4& flags        [[buffer(8)]],
                             constant float& eps          [[buffer(9)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint ti [[thread_index_in_threadgroup]]) {
    const uint s = tg.x;
    const bool isQ = tg.y < QW_Q_HEADS;
    const uint row = isQ ? tg.y : (tg.y - QW_Q_HEADS);
    const uint head = isQ ? (row * QW_Q_STRIDE) : row;   // q slot = interleaved Q‖gate
    const size_t uOff = (isQ ? ((size_t)s * (QW_QPROJ_OUT / 8u) + (size_t)head * flags.x)
                            : ((size_t)s * (QW_KV_HEADS * QW_HEAD_DIM / 8u) + (size_t)row * flags.x));
    device const uint4* src = (isQ ? qIn : kIn) + uOff;
    device uint4* dst = (isQ ? qOut : kOut) + uOff;
    device const uint4* w = (isQ ? wq : (wk + (size_t)row * flags.x));
    if (flags.x != 32) return;                    // golden head only; Swift pre-checks

    uint4 xw = src[ti];                           // 8 halves of the head row
    H8 xv = asH8(xw), wv = asH8(w[ti]);
    float xn[8];
    float acc = 0.0f;
    for (int i = 0; i < 8; i++) { float v = float(xv.v[i]); acc += v * v; }
    float rstd = rsqrt(simd_sum(acc) / 256.0f + eps);
    for (int i = 0; i < 8; i++) xn[i] = float(xv.v[i]) * rstd * (float(wv.v[i]) + 1.0f);

    // uniform partner exchange (all 32 lanes participate; xor-4 ⇄ l±4)
    float4 lo = float4(xn[0], xn[1], xn[2], xn[3]);
    float4 hi = float4(xn[4], xn[5], xn[6], xn[7]);
    float4 plo = simd_shuffle_xor(lo, 4), phi = simd_shuffle_xor(hi, 4);

    if (flags.y) {                                 // staged head-norm (f16 store)
        H8 n16;
        for (int i = 0; i < 8; i++) n16.v[i] = half(xn[i]);
        const size_t nBase = isQ ? (size_t)s * 768
                                 : (size_t)flags.z + (size_t)s * 128;
        normOut[nBase + (size_t)row * 32 + ti] = *(thread uint4*)&n16;
    }

    // Gate halves of each interleaved [Q‖gate] slot are NEVER touched by
    // rope math but qOut is a complete buffer → pass them through with one
    // extra uint4 copy per lane (rope only needed the Q half; the ª saving
    // is in the MATH, not the byte count).
    if (isQ && QW_HAS_Q_GATE) { dst[flags.x + ti] = src[flags.x + ti]; }

    uint4 o;
    if (ti < QW_ROPE_DIMS / 8u) {                    // rope dims 0..(ropeDims-1)
        device const float* cs = table + (size_t)s * QW_ROPE_DIMS;
        const float pf[8] = { plo.x, plo.y, plo.z, plo.w, phi.x, phi.y, phi.z, phi.w };
        const bool firstHalf = ti < QW_ROPE_DIMS / 16u;
        H8 out8;
        for (int i = 0; i < 8; i++) {
            uint p = (ti * 8u + uint(i)) & (QW_ROPE_DIMS / 2u - 1u);
            float c = cs[p], sn = cs[QW_ROPE_DIMS / 2u + p];
            float rotv = firstHalf ? -pf[i] : pf[i];
            out8.v[i] = half(xn[i] * c + rotv * sn);
        }
        o = *(thread uint4*)&out8;
    } else {                                       // passthrough dims: single round
        H8 out8;
        for (int i = 0; i < 8; i++) out8.v[i] = half(xn[i]);
        o = *(thread uint4*)&out8;
    }
    dst[ti] = o;
}

// ── Fused Attention Extract + RoPE + RMSNorm + V copy (EXP-8 landing #2) ──
// Reads directly from packed attnY [S, 14336] (1792 uint4s/row).
// Eliminates qwg_extract_attn_qkv_f16 and intermediate attQ16/attK16 DRAM traffic.
kernel void qwg_extract_rope_norm_qk(device const uint4* attnY    [[buffer(0)]],
                                     device uint4* qOut           [[buffer(1)]],
                                     device uint4* kOut           [[buffer(2)]],
                                     device uint4* vOut           [[buffer(3)]],
                                     device const uint4* wq       [[buffer(4)]],
                                     device const uint4* wk       [[buffer(5)]],
                                     device const float* table    [[buffer(6)]],
                                     constant float& eps          [[buffer(7)]],
                                     uint2 tg [[threadgroup_position_in_grid]],
                                     uint ti [[thread_index_in_threadgroup]]) {
    const uint s = tg.x;
    const bool isQ = tg.y < QW_Q_HEADS;
    const uint row = isQ ? tg.y : (tg.y - QW_Q_HEADS);
    const uint head = isQ ? (row * QW_Q_STRIDE) : row;   // q slot = interleaved Q‖gate

    // attnY's row: q‖gate, then k, then v — in uint4s
    const size_t inRowU4 = (QW_QPROJ_OUT + 2u * QW_KV_HEADS * QW_HEAD_DIM) / 8u;
    const size_t inRow = (size_t)s * inRowU4;
    const size_t srcOff = isQ ? (inRow + head * 32u)
                              : (inRow + (size_t)QW_QPROJ_OUT / 8u + head * 32u);
    device const uint4* src = attnY + srcOff;

    const size_t dstOff = isQ ? ((size_t)s * QW_QPROJ_OUT / 8u + head * 32u)
                              : ((size_t)s * QW_KV_HEADS * QW_HEAD_DIM / 8u + head * 32u);
    device uint4* dst = (isQ ? qOut : kOut) + dstOff;
    device const uint4* w = (isQ ? wq : (wk + (size_t)row * 32u));

    uint4 xw = src[ti];                           // 8 halves of the head row
    H8 xv = asH8(xw), wv = asH8(w[ti]);
    float xn[8];
    float acc = 0.0f;
    for (int i = 0; i < 8; i++) { float v = float(xv.v[i]); acc += v * v; }
    float rstd = rsqrt(simd_sum(acc) / 256.0f + eps);
    for (int i = 0; i < 8; i++) xn[i] = float(xv.v[i]) * rstd * (float(wv.v[i]) + 1.0f);

    // uniform partner exchange (all 32 lanes participate; xor-4 ⇄ l±4)
    float4 lo = float4(xn[0], xn[1], xn[2], xn[3]);
    float4 hi = float4(xn[4], xn[5], xn[6], xn[7]);
    float4 plo = simd_shuffle_xor(lo, 4), phi = simd_shuffle_xor(hi, 4);

    if (isQ) {
        // Pass through gate half (32 uint4s after Q)
        dst[32 + ti] = src[32 + ti];
    } else {
        // The 4 K threadgroups also copy V to vOut (128 uint4s per token)
        vOut[(size_t)s * QW_KV_HEADS * QW_HEAD_DIM / 8u + head * 32u + ti] =
            attnY[inRow + (size_t)(QW_QPROJ_OUT + QW_KV_HEADS * QW_HEAD_DIM) / 8u + head * 32u + ti];
    }

    uint4 o;
    if (ti < QW_ROPE_DIMS / 8u) {                    // rope dims 0..(ropeDims-1)
        device const float* cs = table + (size_t)s * QW_ROPE_DIMS;
        const float pf[8] = { plo.x, plo.y, plo.z, plo.w, phi.x, phi.y, phi.z, phi.w };
        const bool firstHalf = ti < QW_ROPE_DIMS / 16u;
        H8 out8;
        for (int i = 0; i < 8; i++) {
            uint p = (ti * 8u + uint(i)) & (QW_ROPE_DIMS / 2u - 1u);
            float c = cs[p], sn = cs[QW_ROPE_DIMS / 2u + p];
            float rotv = firstHalf ? -pf[i] : pf[i];
            out8.v[i] = half(xn[i] * c + rotv * sn);
        }
        o = *(thread uint4*)&out8;
    } else {                                       // passthrough dims: single round
        H8 out8;
        for (int i = 0; i < 8; i++) out8.v[i] = half(xn[i]);
        o = *(thread uint4*)&out8;
    }
    dst[ti] = o;
}

