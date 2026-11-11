// qwhc_shvq.metal — Hyper-connection mixer kernels over SHVQ{2,3,4}_G{32,64,128,256}
// hc.up and hc.down weights, f32 end to end.

#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_dequant.metalh"
#include "qw_dequant_f32.metalh"
#include "qw_dequant_shvq.metalh"
#include "qwhc_common.metalh"
using namespace metal;

static inline float4 qwg_shvq_fwht4_f32(float4 v) {
    v = v.xxzz + v.yyww * float4(1.0f, -1.0f, 1.0f, -1.0f);
    return v.xyxy + v.zwzw * float4(1.0f, 1.0f, -1.0f, -1.0f);
}

static inline void qwg_shvq_fwht32_slice_f32(thread float4 (&lo)[4], thread float4 (&hi)[4]) {
    if (!QW_FWHT_ROTATED) return;
    #pragma unroll
    for (uint w = 0u; w < 4u; ++w) {
        const float4 a = qwg_shvq_fwht4_f32(lo[w]), b = qwg_shvq_fwht4_f32(hi[w]);
        lo[w] = a + b;
        hi[w] = a - b;
    }
    #pragma unroll
    for (uint w = 0u; w < 4u; w += 2u) {
        const float4 a = lo[w], b = lo[w + 1u], c = hi[w], d = hi[w + 1u];
        lo[w] = a + b; lo[w + 1u] = a - b;
        hi[w] = c + d; hi[w + 1u] = c - d;
    }
    #pragma unroll
    for (uint w = 0u; w < 2u; ++w) {
        const float4 a = lo[w], b = lo[w + 2u], c = hi[w], d = hi[w + 2u];
        lo[w] = (a + b) * 0.17677669529663689f; lo[w + 2u] = (a - b) * 0.17677669529663689f;
        hi[w] = (c + d) * 0.17677669529663689f; hi[w + 2u] = (c - d) * 0.17677669529663689f;
    }
}

template <int BITS, int GROUP>
static inline void qwg_hc_up_mix_h32_shvq_f32_body(device const char*  upBytes,
                                                   device const float* hcLow,
                                                   device const float* xn,
                                                   device float*       mixed,
                                                   device float*       rotMixed,
                                                   uint hidden, uint hc, uint R,
                                                   uint2 tg, uint sg, uint lane,
                                                   threadgroup float* sRot,
                                                   threadgroup float* sMix) {
    const uint b = tg.x, realHC = hc & 0xFFu, nBlocks = R >> 5u;
    const uint halfId = lane >> 4u, s = lane & 15u, h0 = b * 32u + sg * 4u;
    constexpr uint groupBytes = 2u + (uint(GROUP * BITS) >> 3u);
    constexpr uint slicesPerGroup = uint(GROUP) >> 5u;
    constexpr uint slicePayloadBytes = 4u * uint(BITS);
    const uint rowBytes = (R / uint(GROUP)) * groupBytes;
    const size_t hOff = size_t(s / slicesPerGroup) * groupBytes;
    const uint byteInGroup = (s & (slicesPerGroup - 1u)) * slicePayloadBytes;
    const uint t = tg.y;
    for (uint jb = sg; jb < nBlocks; jb += 8u) {
        sRot[jb * 32u + lane] = qwg_fwht32_lane_f32(hcLow[(size_t)t * R + jb * 32u + lane], lane);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 lo[4], hi[4];
    if (s < nBlocks) {
        threadgroup const float* xg = sRot + (s << 5u);
        #pragma unroll
        for (uint w = 0u; w < 4u; ++w) {
            threadgroup const float4* p4 = (threadgroup const float4*)(xg + w * 8u);
            lo[w] = p4[0]; hi[w] = p4[1];
        }
    }
    float4 prodEven = float4(0.0f), prodOdd = float4(0.0f);
    #pragma unroll
    for (uint p = 0u; p < 2u; ++p) {
        const uint c = halfId * 2u + p;
        float4 u = float4(0.0f);
        if (s < nBlocks && c < realHC) {
            device const char* r0p = upBytes + (size_t)(c * hidden + h0) * rowBytes;
            u.x = qwg_shvq_slice_dot<BITS>(r0p + hOff, byteInGroup, lo, hi);
            u.y = qwg_shvq_slice_dot<BITS>(r0p + rowBytes + hOff, byteInGroup, lo, hi);
            u.z = qwg_shvq_slice_dot<BITS>(r0p + 2u * rowBytes + hOff, byteInGroup, lo, hi);
            u.w = qwg_shvq_slice_dot<BITS>(r0p + 3u * rowBytes + hOff, byteInGroup, lo, hi);
        }
        u += simd_shuffle_down(u, 8); u += simd_shuffle_down(u, 4);
        u += simd_shuffle_down(u, 2); u += simd_shuffle_down(u, 1);
        if (s == 0u && c < realHC) {
            const float4 gate = 1.0f / (1.0f + exp(-u));
            const float4 xnc = *(device const float4*)(xn + ((size_t)t * realHC + c) * hidden + h0);
            if (p == 0u) prodEven = gate * xnc; else prodOdd = gate * xnc;
        }
    }
    const float4 p2 = simd_shuffle_down(prodEven, 16), p3 = simd_shuffle_down(prodOdd, 16);
    if (lane == 0u) {
        const float4 mv = (((prodEven + prodOdd) + p2) + p3) / float(realHC);
        *(device float4*)(mixed + (size_t)t * hidden + h0) = mv;
        sMix[sg * 4u + 0u] = mv.x; sMix[sg * 4u + 1u] = mv.y;
        sMix[sg * 4u + 2u] = mv.z; sMix[sg * 4u + 3u] = mv.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        rotMixed[(size_t)t * hidden + b * 32u + lane] = qwg_fwht32_lane_f32(sMix[lane], lane);
    }
}

template <int BITS, int GROUP>
static inline void qwg_hc_down_inject_shvq_f32_body(device const char*   downBytes,
                                                    device const float*  xRot,
                                                    device float*        yLow,
                                                    device const ushort* injW,
                                                    device const float*  xXn,
                                                    device float*        yInj,
                                                    uint NDown, uint K, float scale, uint CH,
                                                    uint2 tgp, uint sg, uint lane,
                                                    threadgroup float4* part4) {
    const uint tg = tgp.x, rw = tgp.y, realCH = CH & 0xFFu, nDownTG = (NDown + 3u) >> 2u;
    if (tg < nDownTG) {
        const uint row0 = tg * 4u;
        const float4 acc = qwg_gemv_shvq_rows4_part_f32<BITS, GROUP>(
            downBytes, xRot + rw * K, row0, NDown, K, sg * 32u + lane, 128u);
        qwg_hc_down_rows4_store(acc, yLow, NDown, rw, row0, scale, sg, lane, part4);
        return;
    }
    qwg_hc_inject_bf16_f32_impl(injW, xXn, yInj, K, scale, realCH, rw, tg - nDownTG, sg, lane, (threadgroup float*)part4);
}

template <int BITS, int GROUP>
static inline void qwg_hc_norm_down_inject_shvq_f32_body(
    device const char*   downBytes,
    device const float*  x,
    device const float*  src,
    device const float*  inj,
    device float*        xOut,
    device const float*  normW,
    device float*        xn,
    device float*        yLow,
    device const ushort* injW,
    device float*        yInj,
    uint NDown, uint K, uint hidden, float eps, float scale,
    uint CH, uint hasInject,
    uint2 tgp, uint sg, uint lane,
    threadgroup float4* part4,
    threadgroup float* sInv) {
    const uint tg = tgp.x, rw = tgp.y, realCH = CH & 0xFFu, nDownTG = (NDown + 3u) >> 2u;
    const uint nTG = nDownTG + realCH;
    const uint tid = sg * 32u + lane;
    const uint groups = K / hidden;
    const bool injecting = hasInject != 0u;
    device const float* xr = x + (size_t)rw * K;
    device const float* sr = src + (size_t)rw * hidden;
    device const float* ir = inj + (size_t)rw * groups;
    {
        const float4 sc4 = injecting ? float4(ir[0], groups > 1u ? ir[1] : 0.0f, groups > 2u ? ir[2] : 0.0f,
                                              groups > 3u ? ir[3] : 0.0f) : float4(0.0f);
        const uint hidden4 = hidden >> 2u;
        float4 ss4 = float4(0.0f);
        #pragma unroll(4)
        for (uint k4 = tid; k4 < (K >> 2u); k4 += 128u) {
            const uint g = k4 / hidden4;
            const float4 sel = float4(g == 0u, g == 1u, g == 2u, g == 3u);
            float4 v = ((device const float4*)xr)[k4];
            if (injecting) v += ((device const float4*)sr)[k4 - g * hidden4] * dot(sc4, sel);
            ss4 += sel * dot(v, v);
        }
        ss4 = simd_sum(ss4);
        if (lane == 0u) part4[sg] = ss4;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < groups) {
            const float4 t = (part4[0] + part4[1]) + (part4[2] + part4[3]);
            sInv[tid] = 1.0f / sqrt(t[tid] / float(hidden) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    {
        const uint K4 = K >> 2u;
        const uint chunk = (K4 + nTG - 1u) / nTG;
        const uint b0 = tg * chunk, b1 = min(b0 + chunk, K4);
        for (uint k4 = b0 + tid; k4 < b1; k4 += 128u) {
            const uint k = k4 << 2u, g = k / hidden;
            float4 v = ((device const float4*)xr)[k4];
            if (injecting) {
                v += ((device const float4*)sr)[(k - g * hidden) >> 2u] * ir[g];
                ((device float4*)(xOut + (size_t)rw * K))[k4] = v;
            }
            ((device float4*)(xn + (size_t)rw * K))[k4] = v * sInv[g] * (1.0f + ((device const float4*)normW)[k4]);
        }
    }

    if (tg < nDownTG) {
        const uint row0 = tg * 4u;
        constexpr uint groupBytes = 2u + (uint(GROUP * BITS) >> 3u);
        constexpr uint slicesPerGroup = uint(GROUP) >> 5u;
        constexpr uint slicePayloadBytes = 4u * uint(BITS);
        const uint rowBytes = (K / uint(GROUP)) * groupBytes;
        device const char* r0 = downBytes + (size_t)row0 * rowBytes;
        device const char* r1 = (row0 + 1u < NDown) ? (r0 + rowBytes) : r0;
        device const char* r2 = (row0 + 2u < NDown) ? (r0 + 2u * rowBytes) : r0;
        device const char* r3 = (row0 + 3u < NDown) ? (r0 + 3u * rowBytes) : r0;
        float4 acc = float4(0.0f);
        for (uint b = tid; b < (K >> 5u); b += 128u) {
            const uint col = b << 5u, g = col / hidden;
            const float inv = sInv[g];
            const float sc = injecting ? ir[g] : 0.0f;
            device const float4* x4 = (device const float4*)(xr + col);
            device const float4* s4 = (device const float4*)(sr + (col - g * hidden));
            device const float4* w4 = (device const float4*)(normW + col);
            float4 xlo[4], xhi[4];
            #pragma unroll
            for (uint w = 0u; w < 4u; ++w) {
                float4 a = x4[2u * w], c = x4[2u * w + 1u];
                if (injecting) { a += s4[2u * w] * sc; c += s4[2u * w + 1u] * sc; }
                xlo[w] = a * inv * (1.0f + w4[2u * w]);
                xhi[w] = c * inv * (1.0f + w4[2u * w + 1u]);
            }
            qwg_shvq_fwht32_slice_f32(xlo, xhi);
            const size_t groupOff = size_t(b / slicesPerGroup) * groupBytes;
            const uint byteInGroup = (b & (slicesPerGroup - 1u)) * slicePayloadBytes;
            acc.x += qwg_shvq_slice_dot<BITS>(r0 + groupOff, byteInGroup, xlo, xhi);
            acc.y += qwg_shvq_slice_dot<BITS>(r1 + groupOff, byteInGroup, xlo, xhi);
            acc.z += qwg_shvq_slice_dot<BITS>(r2 + groupOff, byteInGroup, xlo, xhi);
            acc.w += qwg_shvq_slice_dot<BITS>(r3 + groupOff, byteInGroup, xlo, xhi);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        qwg_hc_down_rows4_store(acc, yLow, NDown, rw, row0, scale, sg, lane, part4);
        return;
    }
    const uint n = tg - nDownTG;
    device const ushort4* wRow4 = (device const ushort4*)(injW + (size_t)n * K);
    float acc = 0.0f;
    for (uint k4 = tid; k4 < (K >> 2u); k4 += 128u) {
        const uint k = k4 << 2u, g = k / hidden;
        float4 v = ((device const float4*)xr)[k4];
        if (injecting) v += ((device const float4*)sr)[(k - g * hidden) >> 2u] * ir[g];
        v = v * sInv[g] * (1.0f + ((device const float4*)normW)[k4]);
        const ushort4 u = wRow4[k4];
        const float4 wv = float4(as_type<float>(uint(u.x) << 16u), as_type<float>(uint(u.y) << 16u),
                                 as_type<float>(uint(u.z) << 16u), as_type<float>(uint(u.w) << 16u));
        acc += dot(wv, v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc = simd_sum(acc);
    threadgroup float* partial = (threadgroup float*)part4;
    if (lane == 0u) partial[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        const float y = (partial[0] + partial[1]) + (partial[2] + partial[3]);
        yInj[rw * realCH + n] = 2.0f / (1.0f + exp(-(y / scale)));
    }
}

#define QW_SHVQ_HC_F32_KERNELS(BITS, GROUP)                                            \
    kernel void qwg_hc_norm_down_inject_shvq##BITS##_g##GROUP##_f32(                    \
        device const char*   downBytes [[buffer(0)]],                                  \
        device const float*  x         [[buffer(1)]],                                  \
        device const float*  src       [[buffer(2)]],                                  \
        device const float*  inj       [[buffer(3)]],                                  \
        device float*        xOut      [[buffer(4)]],                                  \
        device const float*  normW     [[buffer(5)]],                                  \
        device float*        xn        [[buffer(6)]],                                  \
        device float*        yLow      [[buffer(7)]],                                  \
        device const ushort* injW      [[buffer(8)]],                                  \
        device float*        yInj      [[buffer(9)]],                                  \
        constant uint&       NDown     [[buffer(10)]],                                 \
        constant uint&       K         [[buffer(11)]],                                 \
        constant uint&       hidden    [[buffer(12)]],                                 \
        constant float&      eps       [[buffer(13)]],                                 \
        constant float&      scale     [[buffer(14)]],                                 \
        constant uint&       CH        [[buffer(15)]],                                 \
        constant uint&       hasInject [[buffer(16)]],                                 \
        uint2 tgp  [[threadgroup_position_in_grid]],                                   \
        uint  sg   [[simdgroup_index_in_threadgroup]],                                 \
        uint  lane [[thread_index_in_simdgroup]]) {                                    \
        threadgroup float4 part4[16];                                                  \
        threadgroup float sInv[16];                                                    \
        qwg_hc_norm_down_inject_shvq_f32_body<BITS, GROUP>(downBytes, x, src, inj,     \
            xOut, normW, xn, yLow, injW, yInj, NDown, K, hidden, eps, scale, CH,       \
            hasInject, tgp, sg, lane, part4, sInv);                                    \
    }                                                                                  \
    kernel void qwg_hc_up_mix_h32_shvq##BITS##_g##GROUP##_f32(                          \
        device const char*  upBytes  [[buffer(0)]],                                    \
        device const float* hcLow    [[buffer(1)]],                                    \
        device const float* xn       [[buffer(2)]],                                    \
        device float*       mixed    [[buffer(3)]],                                    \
        device float*       rotMixed [[buffer(4)]],                                    \
        constant uint&      hidden   [[buffer(5)]],                                    \
        constant uint&      hc       [[buffer(6)]],                                    \
        constant uint&      R        [[buffer(7)]],                                    \
        uint2 tg   [[threadgroup_position_in_grid]],                                   \
        uint  sg   [[simdgroup_index_in_threadgroup]],                                 \
        uint  lane [[thread_index_in_simdgroup]]) {                                    \
        threadgroup float sRot[512];                                                   \
        threadgroup float sMix[32];                                                    \
        qwg_hc_up_mix_h32_shvq_f32_body<BITS, GROUP>(upBytes, hcLow, xn, mixed,        \
            rotMixed, hidden, hc, R, tg, sg, lane, sRot, sMix);                        \
    }                                                                                  \
    kernel void qwg_hc_down_inject_shvq##BITS##_g##GROUP##_f32(                         \
        device const char*   downBytes [[buffer(0)]],                                  \
        device const float*  xRot      [[buffer(1)]],                                  \
        device float*        yLow      [[buffer(2)]],                                  \
        device const ushort* injW      [[buffer(3)]],                                  \
        device const float*  xXn       [[buffer(4)]],                                  \
        device float*        yInj      [[buffer(5)]],                                  \
        constant uint&       NDown     [[buffer(6)]],                                  \
        constant uint&       K         [[buffer(7)]],                                  \
        constant float&      scale     [[buffer(8)]],                                  \
        constant uint&       CH        [[buffer(9)]],                                  \
        uint2 tgp  [[threadgroup_position_in_grid]],                                   \
        uint  sg   [[simdgroup_index_in_threadgroup]],                                 \
        uint  lane [[thread_index_in_simdgroup]]) {                                    \
        threadgroup float4 part4[4];                                                   \
        qwg_hc_down_inject_shvq_f32_body<BITS, GROUP>(downBytes, xRot, yLow, injW,     \
            xXn, yInj, NDown, K, scale, CH, tgp, sg, lane, part4);                     \
    }

QW_SHVQ_HC_F32_KERNELS(2, 32)
QW_SHVQ_HC_F32_KERNELS(2, 64)
QW_SHVQ_HC_F32_KERNELS(2, 128)
QW_SHVQ_HC_F32_KERNELS(2, 256)
QW_SHVQ_HC_F32_KERNELS(3, 32)
QW_SHVQ_HC_F32_KERNELS(3, 64)
QW_SHVQ_HC_F32_KERNELS(3, 128)
QW_SHVQ_HC_F32_KERNELS(3, 256)
QW_SHVQ_HC_F32_KERNELS(4, 32)
QW_SHVQ_HC_F32_KERNELS(4, 64)
QW_SHVQ_HC_F32_KERNELS(4, 128)
QW_SHVQ_HC_F32_KERNELS(4, 256)
