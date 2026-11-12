// qwgemm_shvq_dense.metal — Dense decode GEMV, general GEMV, and greedy head
// argmax twins for SHVQ{1,15,2,3,4}_G{32,64,128,256}.

#include <metal_stdlib>
#include <metal_simdgroup>
#define QW_SHVQ_ROUTED_DIRECT_LUT 1
#include "qw_dequant_shvq.metalh"
using namespace metal;

#define QW_SHVQ_HEAD_C 8u

static inline uint qwg_shvq_sortable(float v) {
    const uint b = as_type<uint>(v);
    return (b & 0x80000000u) ? ~b : (b ^ 0x80000000u);
}

template <int BITS, int GROUP, uint ROWS>
static inline void qwg_gemv_shvq_dx_body(device const char* blocks,
                                         device const half* x,
                                         uint realN,
                                         constant uint& K,
                                         device half* y16,
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
            if (row < realN) y16[row] = half(acc[r]);
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
            const uint mRows = (N >> 24u) & 0x3Fu;                                      \
            if (mRows >= 2u) {                                                          \
                const uint realN = N & 0x00FFFFFFu;                                     \
                const uint simdPerTG = max(1u, tptg >> 5u);                             \
                const uint row0 = (tg * simdPerTG + (ti >> 5u)) * 4u;                   \
                if (row0 < realN) {                                                     \
                    qwg_gemv_shvq_rows4_m<BITS, GROUP, half>(                           \
                        blocks, x, y16, row0, realN, K, mRows, ti & 31u, tgLut);        \
                }                                                                       \
            } else {                                                                    \
                qwg_gemv_shvq_dx_body<BITS, GROUP, 4u>(blocks, x, N & 0x3FFFFFFFu,      \
                                                       K, y16, tg, ti, tptg, tgLut);    \
            }                                                                           \
        } else if ((N & 0x80000000u) != 0u) {                                           \
            qwg_gemv_shvq_dx_body<BITS, GROUP, 2u>(blocks, x, N & 0x7FFFFFFFu,          \
                                                   K, y16, tg, ti, tptg, tgLut);        \
        } else {                                                                        \
            qwg_gemv_shvq_dx_body<BITS, GROUP, 1u>(blocks, x, N, K, y16, tg, ti, tptg, tgLut); \
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
            qwg_gemv_shvq_rows4_m<BITS, GROUP>(blocks, x, y, row0, realN, K, mRows, lane, tgLut); \
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
    kernel void qwg_gemv_shvq##BITS##_g##GROUP##_v4_f32(                                \
        device const char*  blocks [[buffer(0)]],                                       \
        device const float* x      [[buffer(1)]],                                       \
        device float*       y      [[buffer(3)]],                                       \
        constant uint&      N      [[buffer(4)]],                                       \
        constant uint&      K      [[buffer(5)]],                                       \
        uint tg   [[threadgroup_position_in_grid]],                                     \
        uint ti   [[thread_index_in_threadgroup]],                                      \
        uint tptg [[threads_per_threadgroup]]) {                                        \
        threadgroup ShvqTgLut<BITS> tgLut;                                              \
        if constexpr (BITS <= 3) {                                                      \
            tgLut.load(ti, tptg);                                                       \
            threadgroup_barrier(mem_flags::mem_threadgroup);                            \
        }                                                                               \
        const uint simdId = ti >> 5u;                                                   \
        const uint lane = ti & 31u;                                                     \
        const uint row = tg * max(1u, tptg >> 5u) + simdId;                             \
        if (row >= N) return;                                                           \
        const float v = qwg_gemv_shvq_row_f32_tg<BITS, GROUP>(blocks, x, row, K, lane, tgLut); \
        if (lane == 0u) y[row] = v;                                                     \
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
        threadgroup uint sh[QW_SHVQ_HEAD_C], sl[QW_SHVQ_HEAD_C];                        \
        const bool skipY = (N & 0x40000000u) != 0u;                                     \
        if ((N & 0x80000000u) != 0u) {                                                  \
            const uint realN = N & 0x3FFFFFFFu;                                         \
            const uint row0 = (tg * QW_SHVQ_HEAD_C + sg) * 4u;                          \
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
