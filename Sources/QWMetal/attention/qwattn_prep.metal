// qwattn_prep.metal — Grouped projection output prep for query, key, value, and gate.
// Per token the TG order is nh q-heads, nkv k-heads, nkv v-heads, nh gate-heads; 256 threads = head dims.
// q and k: per-head RMSNorm ((1 + weight) scale) then partial RoPE (first 64 dims, half-split). v and gate: copies.
// Supports KV cache modes: f32 (0), f16 (1), q8 (2), q4 (3); optional raw indexK f16 [T, 128] copy when ikOff > 0.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_geometry_constants.metalh"
using namespace metal;

template <uint KV_MODE>
inline void qwg_attn_prep_impl(device const float* src,
                               device const float* qNorm,
                               device const float* kNorm,
                               device const float* cosT,
                               device const float* sinT,
                               device float*       qOut,
                               device uchar*       kOut,
                               device uchar*       vOut,
                               device float*       gOut,
                               uint T, uint nh, uint nkv, uint rowStride,
                               uint kOff, uint vOff,
                               device half*        idxK,
                               uint ikOff,
                               device half*        kMeta,
                               device half*        vMeta,
                               threadgroup float*  partial,
                               threadgroup float*  sMax,
                               threadgroup float*  staged,
                               uint tg, uint d) {
    constexpr uint D = 256u;
    constexpr uint ROT = 64u;
    constexpr uint HALF = ROT / 2u;
    const uint perTok = nh + 2u * nkv + nh;
    const uint t = tg / perTok;
    const uint r = tg - t * perTok;
    const uint role = (r < nh) ? 0u : (r < nh + nkv) ? 1u : (r < nh + 2u * nkv) ? 2u : 3u;
    const uint h = (role == 0u) ? r : (role == 1u) ? r - nh : (role == 2u) ? r - nh - nkv : r - nh - 2u * nkv;
    const uint rowBase = t * rowStride;
    const uint sg = d >> 5u, lane = d & 31u;
    if (role == 2u) {
        const float v = src[rowBase + vOff + h * D + d];
        if (ikOff > 0u && h == 0u && d < 128u) {
            idxK[t * 128u + d] = half(src[rowBase + ikOff + d]);
        }
        const uint seg = t * nkv + h;
        if constexpr (KV_MODE == 0u) {
            ((device float*)vOut)[seg * D + d] = v;
        } else if constexpr (KV_MODE == 1u) {
            ((device half*)vOut)[seg * D + d] = half(v);
        } else if constexpr (KV_MODE == 2u) {
            const float sm = simd_max(fabs(v));
            if (lane == 0u) partial[sg] = sm;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float mx = 0.0f;
            for (uint i = 0u; i < 8u; i++) mx = fmax(mx, partial[i]);
            const half sc16 = half(fmax(mx / 127.0f, 1e-8f));
            const float inv = 1.0f / float(sc16);
            ((device char*)vOut)[seg * D + d] = (char)(int)clamp(round(v * inv), -127.0f, 127.0f);
            if (d == 0u) vMeta[seg] = sc16;
        } else if constexpr (KV_MODE == 3u) {
            const float smin = simd_min(v), smax = simd_max(v);
            if (lane == 0u) { partial[sg] = smin; sMax[sg] = smax; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint grpBase = (d >> 7u) * 4u;
            float mn = 1e30f, mx = -1e30f;
            for (uint i = 0u; i < 4u; i++) {
                mn = fmin(mn, partial[grpBase + i]);
                mx = fmax(mx, sMax[grpBase + i]);
            }
            const half sc16 = half(fmax((mx - mn) / 15.0f, 1e-8f));
            const half mn16 = half(mn);
            const float inv = 1.0f / float(sc16);
            const uint q = (uint)clamp(round((v - float(mn16)) * inv), 0.0f, 15.0f);
            const uint qOdd = simd_shuffle_down(q, 1u);
            if ((d & 1u) == 0u) {
                vOut[seg * (D / 2u) + (d >> 1u)] = (uchar)(q | (qOdd << 4u));
            }
            if ((d & 127u) == 0u) {
                const uint mIdx = (seg * 2u + (d >> 7u)) * 2u;
                vMeta[mIdx] = sc16;
                vMeta[mIdx + 1u] = mn16;
            }
        }
        return;
    }
    if (role == 3u) {
        gOut[(t * nh + h) * D + d] = src[rowBase + h * 2u * D + D + d];
        return;
    }
    const float x = (role == 0u) ? src[rowBase + h * 2u * D + d] : src[rowBase + kOff + h * D + d];
    device const float* w = (role == 0u) ? qNorm : kNorm;
    const float ss = simd_sum(x * x);
    if (lane == 0u) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = partial[0] + partial[1] + partial[2] + partial[3]
                      + partial[4] + partial[5] + partial[6] + partial[7];
    const float inv = rsqrt(total / float(D) + QW_RMS_EPS);
    staged[d] = x * inv * (1.0f + w[d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = staged[d];
    if (d < ROT) {
        const float rot = (d < HALF) ? -staged[d + HALF] : staged[d - HALF];
        o = staged[d] * cosT[t * ROT + d] + rot * sinT[t * ROT + d];
    }
    if (role == 0u) {
        qOut[(t * nh + h) * D + d] = o;
        return;
    }
    const uint seg = t * nkv + h;
    if constexpr (KV_MODE == 0u) {
        ((device float*)kOut)[seg * D + d] = o;
    } else if constexpr (KV_MODE == 1u) {
        ((device half*)kOut)[seg * D + d] = half(o);
    } else if constexpr (KV_MODE == 2u) {
        const float sm = simd_max(fabs(o));
        if (lane == 0u) partial[sg] = sm;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mx = 0.0f;
        for (uint i = 0u; i < 8u; i++) mx = fmax(mx, partial[i]);
        const half sc16 = half(fmax(mx / 127.0f, 1e-8f));
        const float qinv = 1.0f / float(sc16);
        ((device char*)kOut)[seg * D + d] = (char)(int)clamp(round(o * qinv), -127.0f, 127.0f);
        if (d == 0u) kMeta[seg] = sc16;
    } else if constexpr (KV_MODE == 3u) {
        const float smin = simd_min(o), smax = simd_max(o);
        if (lane == 0u) { partial[sg] = smin; sMax[sg] = smax; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint grpBase = (d >> 7u) * 4u;
        float mn = 1e30f, mx = -1e30f;
        for (uint i = 0u; i < 4u; i++) {
            mn = fmin(mn, partial[grpBase + i]);
            mx = fmax(mx, sMax[grpBase + i]);
        }
        const half sc16 = half(fmax((mx - mn) / 15.0f, 1e-8f));
        const half mn16 = half(mn);
        const float qinv = 1.0f / float(sc16);
        const uint q = (uint)clamp(round((o - float(mn16)) * qinv), 0.0f, 15.0f);
        const uint qOdd = simd_shuffle_down(q, 1u);
        if ((d & 1u) == 0u) {
            kOut[seg * (D / 2u) + (d >> 1u)] = (uchar)(q | (qOdd << 4u));
        }
        if ((d & 127u) == 0u) {
            const uint mIdx = (seg * 2u + (d >> 7u)) * 2u;
            kMeta[mIdx] = sc16;
            kMeta[mIdx + 1u] = mn16;
        }
    }
}

#define DEFINE_ATTN_PREP_KERNEL(NAME, MODE) \
kernel void NAME(device const float* src       [[buffer(0)]], \
                 device const float* qNorm     [[buffer(1)]], \
                 device const float* kNorm     [[buffer(2)]], \
                 device const float* cosT      [[buffer(3)]], \
                 device const float* sinT      [[buffer(4)]], \
                 device float*       qOut      [[buffer(5)]], \
                 device uchar*       kOut      [[buffer(6)]], \
                 device uchar*       vOut      [[buffer(7)]], \
                 device float*       gOut      [[buffer(8)]], \
                 constant uint&      T         [[buffer(9)]], \
                 constant uint&      nh        [[buffer(10)]], \
                 constant uint&      nkv       [[buffer(11)]], \
                 constant uint&      rowStride [[buffer(12)]], \
                 constant uint&      kOff      [[buffer(13)]], \
                 constant uint&      vOff      [[buffer(14)]], \
                 device half*        idxK      [[buffer(15)]], \
                 constant uint&      ikOff     [[buffer(16)]], \
                 device half*        kMeta     [[buffer(17)]], \
                 device half*        vMeta     [[buffer(18)]], \
                 uint tg [[threadgroup_position_in_grid]], \
                 uint d  [[thread_position_in_threadgroup]]) { \
    threadgroup float partial[8]; \
    threadgroup float sMax[8]; \
    threadgroup float staged[256]; \
    qwg_attn_prep_impl<MODE>(src, qNorm, kNorm, cosT, sinT, qOut, kOut, vOut, gOut, \
                             T, nh, nkv, rowStride, kOff, vOff, idxK, ikOff, kMeta, vMeta, \
                             partial, sMax, staged, tg, d); \
}

DEFINE_ATTN_PREP_KERNEL(qwg_attn_prep_f32, 0u)
DEFINE_ATTN_PREP_KERNEL(qwg_attn_prep_f16, 1u)
DEFINE_ATTN_PREP_KERNEL(qwg_attn_prep_q8,  2u)
DEFINE_ATTN_PREP_KERNEL(qwg_attn_prep_q4,  3u)
