// qwgemm_shvq_dense.metal — Dense decode GEMV, general GEMV, and greedy head
// argmax twins for SHVQ{1,15,2,3,4}_G{32,64,128,256}.

#include <metal_stdlib>
#include <metal_simdgroup>
#define QW_SHVQ_ROUTED_DIRECT_LUT 1
#include "qw_dequant_shvq.metalh"
#include "qw_fastmath_safe.metalh"
using namespace metal;

#define QW_SHVQ_HEAD_C 8u

static inline uint qwg_shvq_sortable(float v) {
    return qw_sortable_f32_u32(v);
}

template <int BITS, int GROUP, uint ROWS, typename OutT = half>
static inline void qwg_gemv_shvq_dx_body(device const char* blocks,
                                         device const half* x,
                                         uint realN,
                                         constant uint& K,
                                         device OutT* y,
                                         uint tg,
                                         uint ti,
                                         uint tptg,
                                         threadgroup const ShvqTgLut<BITS> &tgLut) {
    const uint simdPerTG = max(1u, tptg >> 5u);
    const uint simdId = ti >> 5u;
    const uint lane = ti & 31u;
    const uint row0 = (tg * simdPerTG + simdId) * ROWS;
    if (row0 >= realN) return;
    float4 acc;
    if constexpr (BITS <= 3) {
        acc = qwg_gemv_shvq_rowsN_tg<BITS, GROUP, ROWS>(blocks, x, row0, realN, K, lane, tgLut);
    } else {
        (void)tgLut;
        acc = qwg_gemv_shvq_rowsN<BITS, GROUP, ROWS>(blocks, x, row0, realN, K, lane);
    }
    if (lane == 0u) {
        #pragma unroll
        for (uint r = 0u; r < ROWS; ++r) {
            const uint row = row0 + r;
            if (row < realN) y[row] = OutT(acc[r]);
        }
    }
}

template <int BITS, int GROUP, uint MAX_M>
static inline void qwg_head_gemv_shvq_argmax_m_body(
    device const char* blocks, device const half* x, device float* y,
    uint realN, uint K, uint mRows, uint row0, uint tg, uint ti, uint sg, uint lane,
    bool skipY, threadgroup const ShvqTgLut<BITS> &tgLut,
    threadgroup uint* sh, threadgroup uint* sl,
    device uint* keysHi, device uint* keysLo) {
    float4 acc[MAX_M];
    if (row0 < realN) {
        qwg_gemv_shvq_rows4_m_acc<BITS, GROUP, MAX_M>(
            blocks, x, row0, realN, K, mRows, lane, tgLut, acc);
    } else {
        #pragma unroll
        for (uint m = 0u; m < MAX_M; ++m) acc[m] = float4(-INFINITY);
    }
    if (lane == 0u && sg < QW_SHVQ_HEAD_C) {
        for (uint m = 0u; m < MAX_M; ++m) {
            if (m >= mRows) break;
            uint bh = qwg_shvq_sortable(-INFINITY), bl = 0u;
            const float4 am = acc[m];
            const float vals[4] = { am.x, am.y, am.z, am.w };
            #pragma unroll
            for (uint r = 0u; r < 4u; ++r) {
                const uint row = row0 + r;
                if (row < realN) {
                    const float v = vals[r];
                    if (!skipY) y[size_t(m) * realN + row] = v;
                    const uint kh = qwg_shvq_sortable(v);
                    const uint kl = ~row;
                    if (kh > bh || (kh == bh && kl > bl)) { bh = kh; bl = kl; }
                }
            }
            sh[m * QW_SHVQ_HEAD_C + sg] = bh;
            sl[m * QW_SHVQ_HEAD_C + sg] = bl;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ti < mRows) {
        const uint base = ti * QW_SHVQ_HEAD_C;
        uint bh = sh[base], bl = sl[base];
        for (uint q = 1u; q < QW_SHVQ_HEAD_C; ++q) {
            const uint kh = sh[base + q], kl = sl[base + q];
            if (kh > bh || (kh == bh && kl > bl)) { bh = kh; bl = kl; }
        }
        const uint slots = (realN + 31u) >> 5u;
        keysHi[size_t(ti) * slots + tg] = bh;
        keysLo[size_t(ti) * slots + tg] = bl;
    }
}

template <int BITS, int GROUP>
static inline void qwg_head_gemv_shvq_argmax_m16_body(
    device const char* blocks, device const half* x, device float* y,
    uint realN, uint K, uint mRows, uint row0, uint tg, uint ti, uint sg, uint lane,
    bool skipY, threadgroup const ShvqTgLut<BITS> &tgLut,
    threadgroup uint* sh, threadgroup uint* sl,
    device uint* keysHi, device uint* keysLo) {
    float2 acc[16];
    if (row0 < realN) {
        qwg_gemv_shvq_rows2_m16_acc<BITS, GROUP>(
            blocks, x, row0, realN, K, mRows, lane, tgLut, acc);
    } else {
        #pragma unroll
        for (uint m = 0u; m < 16u; ++m) acc[m] = float2(-INFINITY);
    }
    if (lane == 0u && sg < QW_SHVQ_HEAD_C) {
        for (uint m = 0u; m < 16u; ++m) {
            if (m >= mRows) break;
            uint bh = qwg_shvq_sortable(-INFINITY), bl = 0u;
            const float2 am = acc[m];
            const float vals[2] = { am.x, am.y };
            #pragma unroll
            for (uint r = 0u; r < 2u; ++r) {
                const uint row = row0 + r;
                if (row < realN) {
                    const float v = vals[r];
                    if (!skipY) y[size_t(m) * realN + row] = v;
                    const uint kh = qwg_shvq_sortable(v);
                    const uint kl = ~row;
                    if (kh > bh || (kh == bh && kl > bl)) { bh = kh; bl = kl; }
                }
            }
            sh[m * QW_SHVQ_HEAD_C + sg] = bh;
            sl[m * QW_SHVQ_HEAD_C + sg] = bl;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ti < mRows) {
        const uint base = ti * QW_SHVQ_HEAD_C;
        uint bh = sh[base], bl = sl[base];
        for (uint q = 1u; q < QW_SHVQ_HEAD_C; ++q) {
            const uint kh = sh[base + q], kl = sl[base + q];
            if (kh > bh || (kh == bh && kl > bl)) { bh = kh; bl = kl; }
        }
        const uint slots = (realN + 15u) >> 4u;
        keysHi[size_t(ti) * slots + tg] = bh;
        keysLo[size_t(ti) * slots + tg] = bl;
    }
}

template <int BITS, int GROUP, typename OutT = float, uint MAX_M = 8u>
METAL_FUNC void qwg_gemv_shvq_rows2_m(device const char* blocks,
                                      device const half* x,
                                      device OutT* y,
                                      uint row0,
                                      uint realN,
                                      uint K,
                                      uint mRows,
                                      uint lane,
                                      threadgroup const ShvqTgLut<BITS> &tgLut) {
    const uint nBlocks = K >> 5u;
    constexpr uint groupBytes = qwg_shvq_group_bytes(BITS, GROUP);
    const uint rowBytes = (K / uint(GROUP)) * groupBytes;
    device const char* rowBase = blocks + (size_t)row0 * rowBytes;
    const bool hasR1 = (row0 + 1u < realN);
    float2 acc[MAX_M];
    #pragma unroll
    for (uint m = 0u; m < MAX_M; ++m) acc[m] = float2(0.0f);
    for (uint g = lane; g < nBlocks; g += 32u) {
        const uint col = g << 5u;
        const size_t groupOff = size_t(col / uint(GROUP)) * groupBytes;
        const uint byteInGroup = qwg_shvq_byte_in_group<BITS, GROUP>(col);
        const float sc0 = qwg_shvq_eff_scale<BITS>(rowBase + groupOff);
        const float sc1 = hasR1 ? qwg_shvq_eff_scale<BITS>(rowBase + rowBytes + groupOff) : 0.0f;
        float dm0[MAX_M], dm1[MAX_M];
        #pragma unroll
        for (uint m = 0u; m < MAX_M; ++m) { dm0[m] = 0.0f; dm1[m] = 0.0f; }
        uint u0[4], u1[4];
        if constexpr (BITS == 15) {
            qwg_shvq15_slice_u12x4((device const uchar*)(rowBase + groupOff + 2u + byteInGroup), u0);
            if (hasR1) {
                qwg_shvq15_slice_u12x4((device const uchar*)(rowBase + rowBytes + groupOff + 2u + byteInGroup), u1);
            }
        }
        #pragma unroll
        for (uint w = 0u; w < 4u; ++w) {
            half4 hxLo[MAX_M], hxHi[MAX_M];
            #pragma unroll
            for (uint m = 0u; m < MAX_M; ++m) {
                if (m < mRows) qwg_load_half8(x + size_t(m) * K + col + w * 8u, hxLo[m], hxHi[m]);
            }
            float4 fLo0, fHi0;
            bool neg0 = false;
            if constexpr (BITS == 15) {
                neg0 = qwg_shvq15_oct_f4(u0[w], fLo0, fHi0);
            } else {
                half4 hLo0, hHi0;
                qwg_shvq_unpack_oct8_impl<BITS, true>(
                    (device const uchar*)(rowBase + groupOff + 2u + byteInGroup), w, &tgLut, hLo0, hHi0);
                fLo0 = float4(hLo0); fHi0 = float4(hHi0);
            }
            #pragma unroll
            for (uint m = 0u; m < MAX_M; ++m) {
                if (m < mRows) {
                    const float d0 = dot(fLo0, float4(hxLo[m])) + dot(fHi0, float4(hxHi[m]));
                    dm0[m] += neg0 ? -d0 : d0;
                }
            }
            if (hasR1) {
                float4 fLo1, fHi1;
                bool neg1 = false;
                if constexpr (BITS == 15) {
                    neg1 = qwg_shvq15_oct_f4(u1[w], fLo1, fHi1);
                } else {
                    half4 hLo1, hHi1;
                    qwg_shvq_unpack_oct8_impl<BITS, true>(
                        (device const uchar*)(rowBase + rowBytes + groupOff + 2u + byteInGroup), w, &tgLut, hLo1, hHi1);
                    fLo1 = float4(hLo1); fHi1 = float4(hHi1);
                }
                #pragma unroll
                for (uint m = 0u; m < MAX_M; ++m) {
                    if (m < mRows) {
                        const float d1 = dot(fLo1, float4(hxLo[m])) + dot(fHi1, float4(hxHi[m]));
                        dm1[m] += neg1 ? -d1 : d1;
                    }
                }
            }
        }
        #pragma unroll
        for (uint m = 0u; m < MAX_M; ++m) {
            if (m < mRows) {
                acc[m].x = fma(dm0[m], sc0, acc[m].x);
                if (hasR1) acc[m].y = fma(dm1[m], sc1, acc[m].y);
            }
        }
    }
    #pragma unroll
    for (uint m = 0u; m < MAX_M; ++m) {
        if (m < mRows) acc[m] = qwg_simd_reduce_down<32u>(acc[m]);
    }
    if (lane == 0u) {
        #pragma unroll
        for (uint m = 0u; m < MAX_M; ++m) {
            if (m < mRows) {
                device OutT* ym = y + (size_t)m * realN;
                ym[row0] = OutT(acc[m].x);
                if (hasR1) ym[row0 + 1u] = OutT(acc[m].y);
            }
        }
    }
}

#define QW_SHVQ_DENSE_KERNELS(BITS, GROUP)                                              \
    kernel void qwg_gemv_shvq##BITS##_g##GROUP##_dx_h16_t4(                             \
        device const char* blocks [[buffer(0)]],                                        \
        device const half* x      [[buffer(1)]],                                        \
        device float*      y32    [[buffer(3)]],                                        \
        constant uint&     N      [[buffer(4)]],                                        \
        constant uint&     K      [[buffer(5)]],                                        \
        device half*       y16    [[buffer(8)]],                                        \
        uint tg   [[threadgroup_position_in_grid]],                                     \
        uint ti   [[thread_index_in_threadgroup]],                                      \
        uint tptg [[threads_per_threadgroup]]) {                                        \
        (void)y32;                                                                      \
        threadgroup ShvqTgLut<BITS> tgLut;                                              \
        if constexpr (BITS <= 3) {                                                      \
            tgLut.load(ti, tptg);                                                       \
            threadgroup_barrier(mem_flags::mem_threadgroup);                            \
        }                                                                               \
        if ((N & 0x40000000u) != 0u) {                                                  \
            const bool outBF16 = (N & 0x20000000u) != 0u;                               \
            const bool twoRows = (N & 0x80000000u) != 0u;                               \
            const uint mRows = (N >> 24u) & 0x1Fu;                                      \
            const uint realN = N & 0x00FFFFFFu;                                         \
            if (mRows >= 2u && twoRows) {                                              \
                const uint simdPerTG = max(1u, tptg >> 5u);                             \
                const uint row0 = (tg * simdPerTG + (ti >> 5u)) * 2u;                   \
                if (row0 < realN) {                                                     \
                    if (outBF16) {                                                      \
                        qwg_gemv_shvq_rows2_m<BITS, GROUP, bfloat, 8u>(                 \
                            blocks, x, (device bfloat*)y16, row0, realN, K, mRows,      \
                            ti & 31u, tgLut);                                           \
                    } else {                                                            \
                        qwg_gemv_shvq_rows2_m<BITS, GROUP, half, 8u>(                   \
                            blocks, x, y16, row0, realN, K, mRows, ti & 31u, tgLut);    \
                    }                                                                   \
                }                                                                       \
            } else if (mRows >= 2u) {                                                   \
                const uint simdPerTG = max(1u, tptg >> 5u);                             \
                const uint row0 = (tg * simdPerTG + (ti >> 5u)) * 4u;                   \
                if (row0 < realN) {                                                     \
                    if (outBF16) {                                                      \
                        qwg_gemv_shvq_rows4_m<BITS, GROUP, bfloat, 8u>(                 \
                            blocks, x, (device bfloat*)y16, row0, realN, K, mRows,      \
                            ti & 31u, tgLut);                                           \
                    } else {                                                            \
                        qwg_gemv_shvq_rows4_m<BITS, GROUP, half, 8u>(                   \
                            blocks, x, y16, row0, realN, K, mRows, ti & 31u, tgLut);    \
                    }                                                                   \
                }                                                                       \
            } else if (outBF16) {                                                       \
                if (twoRows) {                                                          \
                    qwg_gemv_shvq_dx_body<BITS, GROUP, 2u, bfloat>(                     \
                        blocks, x, realN, K, (device bfloat*)y16, tg, ti, tptg, tgLut); \
                } else {                                                                \
                    qwg_gemv_shvq_dx_body<BITS, GROUP, 4u, bfloat>(                     \
                        blocks, x, realN, K, (device bfloat*)y16, tg, ti, tptg, tgLut); \
                }                                                                       \
            } else {                                                                    \
                qwg_gemv_shvq_dx_body<BITS, GROUP, 4u, half>(                           \
                    blocks, x, realN, K, y16, tg, ti, tptg, tgLut);                     \
            }                                                                           \
        } else if ((N & 0x80000000u) != 0u) {                                           \
            qwg_gemv_shvq_dx_body<BITS, GROUP, 2u>(blocks, x, N & 0x7FFFFFFFu,          \
                                                   K, y16, tg, ti, tptg, tgLut);        \
        } else {                                                                        \
            qwg_gemv_shvq_dx_body<BITS, GROUP, 1u>(blocks, x, N, K, y16, tg, ti, tptg, tgLut); \
        }                                                                               \
    }                                                                                   \
    kernel void qwg_gemv_shvq##BITS##_g##GROUP##_dx_h16_t4_m4(                          \
        device const char* blocks [[buffer(0)]],                                        \
        device const half* x      [[buffer(1)]],                                        \
        device float*      y32    [[buffer(3)]],                                        \
        constant uint&     N      [[buffer(4)]],                                        \
        constant uint&     K      [[buffer(5)]],                                        \
        device half*       y16    [[buffer(8)]],                                        \
        uint tg   [[threadgroup_position_in_grid]],                                     \
        uint ti   [[thread_index_in_threadgroup]],                                      \
        uint tptg [[threads_per_threadgroup]]) {                                        \
        (void)y32;                                                                      \
        threadgroup ShvqTgLut<BITS> tgLut;                                              \
        if constexpr (BITS <= 3) {                                                      \
            tgLut.load(ti, tptg);                                                       \
            threadgroup_barrier(mem_flags::mem_threadgroup);                            \
        }                                                                               \
        const bool outBF16 = (N & 0x20000000u) != 0u;                                   \
        const uint mRows = (N >> 24u) & 0x1Fu;                                          \
        const uint realN = N & 0x00FFFFFFu;                                             \
        const uint simdPerTG = max(1u, tptg >> 5u);                                     \
        const uint row0 = (tg * simdPerTG + (ti >> 5u)) * 4u;                           \
        if (row0 >= realN) return;                                                      \
        if (outBF16) {                                                                  \
            qwg_gemv_shvq_rows4_m<BITS, GROUP, bfloat, 4u>(                             \
                blocks, x, (device bfloat*)y16, row0, realN, K, mRows, ti & 31u, tgLut); \
        } else {                                                                        \
            qwg_gemv_shvq_rows4_m<BITS, GROUP, half, 4u>(                               \
                blocks, x, y16, row0, realN, K, mRows, ti & 31u, tgLut);                \
        }                                                                               \
    }                                                                                   \
    kernel void qwg_gemv_shvq##BITS##_g##GROUP##_v4(                                    \
        device const char* blocks [[buffer(0)]],                                        \
        device const half* x      [[buffer(1)]],                                        \
        device float*      y      [[buffer(3)]],                                        \
        constant uint&     N      [[buffer(4)]],                                        \
        constant uint&     K      [[buffer(5)]],                                        \
        uint tg   [[threadgroup_position_in_grid]],                                     \
        uint ti   [[thread_index_in_threadgroup]],                                      \
        uint tptg [[threads_per_threadgroup]]) {                                        \
        const uint simdId = ti >> 5u;                                                   \
        const uint lane = ti & 31u;                                                     \
        const uint simdPerTG = max(1u, tptg >> 5u);                                     \
        threadgroup ShvqTgLut<BITS> tgLut;                                              \
        if constexpr (BITS <= 3) {                                                      \
            tgLut.load(ti, tptg);                                                       \
            threadgroup_barrier(mem_flags::mem_threadgroup);                            \
        }                                                                               \
        if ((N & 0x80000000u) != 0u) {                                                  \
            const uint mRows = (N >> 24u) & 0x3Fu;                                      \
            const uint realN = N & 0x00FFFFFFu;                                         \
            const uint row0 = (tg * simdPerTG + simdId) * 4u;                           \
            if (row0 >= realN) return;                                                  \
            if (mRows <= 4u) {                                                          \
                qwg_gemv_shvq_rows4_m<BITS, GROUP, float, 4u>(                          \
                    blocks, x, y, row0, realN, K, mRows, lane, tgLut);                  \
            } else {                                                                    \
                qwg_gemv_shvq_rows4_m<BITS, GROUP, float, 8u>(                          \
                    blocks, x, y, row0, realN, K, mRows, lane, tgLut);                  \
            }                                                                           \
        } else if ((N & 0x40000000u) != 0u) {                                           \
            const uint realN = N & 0x3FFFFFFFu;                                         \
            const uint row0 = (tg * simdPerTG + simdId) * 4u;                           \
            if (row0 >= realN) return;                                                  \
            const float4 acc = qwg_gemv_shvq_rowsN_tg<BITS, GROUP, 4u>(                 \
                blocks, x, row0, realN, K, lane, tgLut);                                \
            if (lane == 0u) {                                                           \
                _Pragma("unroll")                                                       \
                for (uint r = 0u; r < 4u; ++r) {                                        \
                    if (row0 + r < realN) y[row0 + r] = acc[r];                         \
                }                                                                       \
            }                                                                           \
        } else {                                                                        \
            const uint row = tg * simdPerTG + simdId;                                   \
            if (row >= N) return;                                                       \
            const float v = qwg_gemv_shvq_rowsN_tg<BITS, GROUP, 1u>(                    \
                blocks, x, row, row + 1u, K, lane, tgLut).x;                            \
            if (lane == 0u) y[row] = v;                                                 \
        }                                                                               \
    }                                                                                   \
    kernel void qwg_head_gemv_shvq##BITS##_g##GROUP##_argmax_m4(                        \
        device const char* blocks [[buffer(0)]],                                        \
        device const half* x      [[buffer(1)]],                                        \
        device float*      y      [[buffer(3)]],                                        \
        constant uint&     N      [[buffer(4)]],                                        \
        constant uint&     K      [[buffer(5)]],                                        \
        device uint*       keysHi [[buffer(6)]],                                        \
        device uint*       keysLo [[buffer(7)]],                                        \
        uint tg [[threadgroup_position_in_grid]],                                       \
        uint ti [[thread_index_in_threadgroup]]) {                                      \
        const uint sg = ti >> 5u;                                                       \
        const uint lane = ti & 31u;                                                     \
        threadgroup ShvqTgLut<BITS> tgLut;                                              \
        if constexpr (BITS <= 3) {                                                      \
            tgLut.load(ti, QW_SHVQ_HEAD_C * 32u);                                       \
            threadgroup_barrier(mem_flags::mem_threadgroup);                            \
        }                                                                               \
        threadgroup uint sh[4u * QW_SHVQ_HEAD_C], sl[4u * QW_SHVQ_HEAD_C];              \
        const bool skipY = (N & 0x40000000u) != 0u;                                     \
        const uint mRows = (N >> 24u) & 0x1Fu;                                          \
        const uint realN = N & 0x00FFFFFFu;                                             \
        const uint row0 = (tg * QW_SHVQ_HEAD_C + sg) * 4u;                              \
        qwg_head_gemv_shvq_argmax_m_body<BITS, GROUP, 4u>(                              \
            blocks, x, y, realN, K, mRows, row0, tg, ti, sg, lane,                      \
            skipY, tgLut, sh, sl, keysHi, keysLo);                                      \
    }                                                                                   \
    kernel void qwg_head_gemv_shvq##BITS##_g##GROUP##_argmax_m16(                       \
        device const char* blocks [[buffer(0)]],                                        \
        device const half* x      [[buffer(1)]],                                        \
        device float*      y      [[buffer(3)]],                                        \
        constant uint&     N      [[buffer(4)]],                                        \
        constant uint&     K      [[buffer(5)]],                                        \
        device uint*       keysHi [[buffer(6)]],                                        \
        device uint*       keysLo [[buffer(7)]],                                        \
        uint tg [[threadgroup_position_in_grid]],                                       \
        uint ti [[thread_index_in_threadgroup]]) {                                      \
        const uint sg = ti >> 5u;                                                       \
        const uint lane = ti & 31u;                                                     \
        threadgroup ShvqTgLut<BITS> tgLut;                                              \
        if constexpr (BITS <= 3) {                                                      \
            tgLut.load(ti, QW_SHVQ_HEAD_C * 32u);                                       \
            threadgroup_barrier(mem_flags::mem_threadgroup);                            \
        }                                                                               \
        threadgroup uint sh[16u * QW_SHVQ_HEAD_C], sl[16u * QW_SHVQ_HEAD_C];            \
        const bool skipY = (N & 0x40000000u) != 0u;                                     \
        const uint mRows = (N >> 24u) & 0x1Fu;                                          \
        const uint realN = N & 0x00FFFFFFu;                                             \
        const uint row0 = (tg * QW_SHVQ_HEAD_C + sg) * 2u;                              \
        qwg_head_gemv_shvq_argmax_m16_body<BITS, GROUP>(                                \
            blocks, x, y, realN, K, mRows, row0, tg, ti, sg, lane,                      \
            skipY, tgLut, sh, sl, keysHi, keysLo);                                      \
    }                                                                                   \
    kernel void qwg_head_gemv_shvq##BITS##_g##GROUP##_argmax(                           \
        device const char* blocks [[buffer(0)]],                                        \
        device const half* x      [[buffer(1)]],                                        \
        device float*      y      [[buffer(3)]],                                        \
        constant uint&     N      [[buffer(4)]],                                        \
        constant uint&     K      [[buffer(5)]],                                        \
        device uint*       keysHi [[buffer(6)]],                                        \
        device uint*       keysLo [[buffer(7)]],                                        \
        uint tg [[threadgroup_position_in_grid]],                                       \
        uint ti [[thread_index_in_threadgroup]]) {                                      \
        const uint sg = ti >> 5u;                                                       \
        const uint lane = ti & 31u;                                                     \
        threadgroup ShvqTgLut<BITS> tgLut;                                              \
        if constexpr (BITS <= 3) {                                                      \
            tgLut.load(ti, QW_SHVQ_HEAD_C * 32u);                                       \
            threadgroup_barrier(mem_flags::mem_threadgroup);                            \
        }                                                                               \
        threadgroup uint sh[8u * QW_SHVQ_HEAD_C], sl[8u * QW_SHVQ_HEAD_C];              \
        const bool skipY = (N & 0x40000000u) != 0u;                                     \
        if ((N & 0x80000000u) != 0u) {                                                  \
            const uint mRows = (N >> 24u) & 0x1Fu;                                      \
            const uint realN = N & 0x00FFFFFFu;                                         \
            const uint row0 = (tg * QW_SHVQ_HEAD_C + sg) * 4u;                          \
            if (mRows >= 2u) {                                                          \
                qwg_head_gemv_shvq_argmax_m_body<BITS, GROUP, 8u>(                      \
                    blocks, x, y, realN, K, mRows, row0, tg, ti, sg, lane,              \
                    skipY, tgLut, sh, sl, keysHi, keysLo);                              \
                return;                                                                 \
            }                                                                           \
            const float4 vx = (row0 < realN)                                            \
                ? qwg_gemv_shvq_rowsN_tg<BITS, GROUP, 4u>(blocks, x, row0, realN, K, lane, tgLut) \
                : float4(-INFINITY);                                                    \
            if (lane == 0u) {                                                           \
                const float vals[4] = { vx.x, vx.y, vx.z, vx.w };                       \
                uint bh = qwg_shvq_sortable(-INFINITY), bl = 0u;                        \
                _Pragma("unroll")                                                       \
                for (uint r = 0u; r < 4u; ++r) {                                        \
                    const uint row = row0 + r;                                          \
                    if (row < realN) {                                                  \
                        const float v = vals[r];                                        \
                        if (!skipY) y[row] = v;                                         \
                        const uint kh = qwg_shvq_sortable(v);                           \
                        const uint kl = ~row;                                           \
                        if (kh > bh || (kh == bh && kl > bl)) { bh = kh; bl = kl; }     \
                    }                                                                   \
                }                                                                       \
                sh[sg] = bh;                                                            \
                sl[sg] = bl;                                                            \
            }                                                                           \
        } else {                                                                        \
            const uint realN = N & 0x3FFFFFFFu;                                         \
            const uint row = tg * QW_SHVQ_HEAD_C + sg;                                  \
            float vx = -INFINITY;                                                       \
            if (row < realN) {                                                          \
                vx = qwg_gemv_shvq_rowsN_tg<BITS, GROUP, 1u>(                           \
                    blocks, x, row, realN, K, lane, tgLut).x;                           \
            }                                                                           \
            if (lane == 0u) {                                                           \
                if (row < realN && !skipY) y[row] = vx;                                 \
                sh[sg] = qwg_shvq_sortable(vx);                                         \
                sl[sg] = (row < realN) ? ~row : 0u;                                     \
            }                                                                           \
        }                                                                               \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                \
        if (ti == 0u) {                                                                 \
            uint bh = sh[0], bl = sl[0];                                                \
            for (uint q = 1u; q < QW_SHVQ_HEAD_C; ++q) {                                \
                if (sh[q] > bh || (sh[q] == bh && sl[q] > bl)) { bh = sh[q]; bl = sl[q]; } \
            }                                                                           \
            keysHi[tg] = bh;                                                            \
            keysLo[tg] = bl;                                                            \
        }                                                                               \
    }

QW_SHVQ_DENSE_KERNELS(1, 32)
QW_SHVQ_DENSE_KERNELS(1, 64)
QW_SHVQ_DENSE_KERNELS(1, 128)
QW_SHVQ_DENSE_KERNELS(1, 256)
QW_SHVQ_DENSE_KERNELS(15, 32)
QW_SHVQ_DENSE_KERNELS(15, 64)
QW_SHVQ_DENSE_KERNELS(15, 128)
QW_SHVQ_DENSE_KERNELS(15, 256)
QW_SHVQ_DENSE_KERNELS(2, 32)
QW_SHVQ_DENSE_KERNELS(2, 64)
QW_SHVQ_DENSE_KERNELS(2, 128)
QW_SHVQ_DENSE_KERNELS(2, 256)
QW_SHVQ_DENSE_KERNELS(3, 32)
QW_SHVQ_DENSE_KERNELS(3, 64)
QW_SHVQ_DENSE_KERNELS(3, 128)
QW_SHVQ_DENSE_KERNELS(3, 256)
QW_SHVQ_DENSE_KERNELS(4, 32)
QW_SHVQ_DENSE_KERNELS(4, 64)
QW_SHVQ_DENSE_KERNELS(4, 128)
QW_SHVQ_DENSE_KERNELS(4, 256)
