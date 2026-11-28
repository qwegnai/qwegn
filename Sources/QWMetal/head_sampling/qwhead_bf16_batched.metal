// qwhead_bf16_batched.metal — Batched BF16 LM-head GEMV (rows2/rows4) and
// fused 2-stage deterministic GPU greedy argmax (BF16/F16/HQ4/AQ4) for M=1..8.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_dequant.metalh"
#include "include/qw_fastmath_safe.metalh"
using namespace metal;

struct QWHeadH8BF { half v[8]; } __attribute__((__packed__, __aligned__(16)));

static inline float qw_head_softcap(float v, float cap) {
    return (cap > 0.0f) ? (precise::tanh(clamp(v / cap, -15.0f, 15.0f)) * cap) : v;
}

static inline float qw_head_dot4_fma(float4 a, float4 b, float acc) {
    return fma(a.w, b.w, fma(a.z, b.z, fma(a.y, b.y, fma(a.x, b.x, acc))));
}

static inline float qw_head_dot8_fma(float4 w0, float4 w1, float4 x0, float4 x1, float acc) {
    return qw_head_dot4_fma(w1, x1, qw_head_dot4_fma(w0, x0, acc));
}

static inline float4 qw_head_simd_sum4(float4 v) {
    v += simd_shuffle_down(v, 16u);
    v += simd_shuffle_down(v, 8u);
    v += simd_shuffle_down(v, 4u);
    v += simd_shuffle_down(v, 2u);
    v += simd_shuffle_down(v, 1u);
    return v;
}

static inline void qw_head_unpack_bf16_8(uint4 va, thread float4& lo, thread float4& hi) {
    lo = float4(as_type<bfloat>(ushort(va.x)), as_type<bfloat>(ushort(va.x >> 16u)),
                as_type<bfloat>(ushort(va.y)), as_type<bfloat>(ushort(va.y >> 16u)));
    hi = float4(as_type<bfloat>(ushort(va.z)), as_type<bfloat>(ushort(va.z >> 16u)),
                as_type<bfloat>(ushort(va.w)), as_type<bfloat>(ushort(va.w >> 16u)));
}

template<bool X_IS_F16>
static inline void qw_head_unpack_x8(uint4 xv, thread float4& lo, thread float4& hi) {
    if constexpr (X_IS_F16) {
        lo = float4(float2(as_type<half2>(xv.x)), float2(as_type<half2>(xv.y)));
        hi = float4(float2(as_type<half2>(xv.z)), float2(as_type<half2>(xv.w)));
    } else {
        qw_head_unpack_bf16_8(xv, lo, hi);
    }
}

template<bool X_IS_F16>
static inline float qw_head_load_x1(device const ushort* x, size_t idx) {
    if constexpr (X_IS_F16) {
        return float(as_type<half>(x[idx]));
    } else {
        return float(as_type<bfloat>(x[idx]));
    }
}

template<uint M, uint ROWS, bool X_IS_F16>
static inline void qw_head_bf16_dot_rows(
        device const ushort* W, device const ushort* x,
        uint row0, uint realN, uint K, uint lane,
        thread float (&out)[ROWS][M], uint mRows = M) {
    float acc[ROWS][M];
    #pragma unroll
    for (uint r = 0u; r < ROWS; ++r) {
        #pragma unroll
        for (uint m = 0u; m < M; ++m) acc[r][m] = 0.0f;
    }

    if ((K & 7u) == 0u) {
        const uint chunks = K >> 3u;
        device const uint4* wr[ROWS];
        #pragma unroll
        for (uint r = 0u; r < ROWS; ++r) {
            const uint row = (row0 + r < realN) ? (row0 + r) : row0;
            wr[r] = (device const uint4*)(W + (size_t)row * K);
        }
        for (uint c = lane; c < chunks; c += 32u) {
            if constexpr (ROWS == 4u) {
                float4 xLo[M], xHi[M];
                #pragma unroll
                for (uint m = 0u; m < M; ++m) {
                    if (m >= mRows) break;
                    const uint4 xv = ((device const uint4*)(x + (size_t)m * K))[c];
                    qw_head_unpack_x8<X_IS_F16>(xv, xLo[m], xHi[m]);
                }
                #pragma unroll
                for (uint r = 0u; r < ROWS; ++r) {
                    float4 wLo, wHi;
                    qw_head_unpack_bf16_8(wr[r][c], wLo, wHi);
                    #pragma unroll
                    for (uint m = 0u; m < M; ++m) {
                        if (m >= mRows) break;
                        acc[r][m] = qw_head_dot8_fma(wLo, wHi, xLo[m], xHi[m], acc[r][m]);
                    }
                }
            } else {
                float4 wLo[ROWS], wHi[ROWS];
                #pragma unroll
                for (uint r = 0u; r < ROWS; ++r) {
                    qw_head_unpack_bf16_8(wr[r][c], wLo[r], wHi[r]);
                }
                #pragma unroll
                for (uint m = 0u; m < M; ++m) {
                    if (m >= mRows) break;
                    const uint4 xv = ((device const uint4*)(x + (size_t)m * K))[c];
                    float4 xLo, xHi;
                    qw_head_unpack_x8<X_IS_F16>(xv, xLo, xHi);
                    #pragma unroll
                    for (uint r = 0u; r < ROWS; ++r) {
                        acc[r][m] = qw_head_dot8_fma(wLo[r], wHi[r], xLo, xHi, acc[r][m]);
                    }
                }
            }
        }
    } else {
        device const bfloat* wr[ROWS];
        #pragma unroll
        for (uint r = 0u; r < ROWS; ++r) {
            const uint row = (row0 + r < realN) ? (row0 + r) : row0;
            wr[r] = (device const bfloat*)(W + (size_t)row * K);
        }
        for (uint k = lane; k < K; k += 32u) {
            float wv[ROWS];
            #pragma unroll
            for (uint r = 0u; r < ROWS; ++r) wv[r] = float(wr[r][k]);
            #pragma unroll
            for (uint m = 0u; m < M; ++m) {
                if (m >= mRows) break;
                const float xv = qw_head_load_x1<X_IS_F16>(x, (size_t)m * K + k);
                #pragma unroll
                for (uint r = 0u; r < ROWS; ++r) acc[r][m] = fma(wv[r], xv, acc[r][m]);
            }
        }
    }

    #pragma unroll
    for (uint r = 0u; r < ROWS; ++r) {
        const float4 lo = qw_head_simd_sum4(float4(
            acc[r][0],
            (M >= 2u) ? acc[r][1] : 0.0f,
            (M >= 3u) ? acc[r][2] : 0.0f,
            (M >= 4u) ? acc[r][3] : 0.0f));
        out[r][0] = lo.x;
        if constexpr (M >= 2u) out[r][1] = lo.y;
        if constexpr (M >= 3u) out[r][2] = lo.z;
        if constexpr (M >= 4u) out[r][3] = lo.w;
        if constexpr (M > 4u) {
            if (mRows > 4u) {
                const float4 hi = qw_head_simd_sum4(float4(
                    acc[r][4],
                    (M >= 6u) ? acc[r][5] : 0.0f,
                    (M >= 7u) ? acc[r][6] : 0.0f,
                    (M >= 8u) ? acc[r][7] : 0.0f));
                out[r][4] = hi.x;
                if constexpr (M >= 6u) out[r][5] = hi.y;
                if constexpr (M >= 7u) out[r][6] = hi.z;
                if constexpr (M >= 8u) out[r][7] = hi.w;
            }
        }
        if constexpr (M > 8u) {
            if (mRows > 8u) {
                const float4 hi2 = qw_head_simd_sum4(float4(
                    acc[r][8],
                    (M >= 10u) ? acc[r][9] : 0.0f,
                    (M >= 11u) ? acc[r][10] : 0.0f,
                    (M >= 12u) ? acc[r][11] : 0.0f));
                out[r][8] = hi2.x;
                if constexpr (M >= 10u) out[r][9] = hi2.y;
                if constexpr (M >= 11u) out[r][10] = hi2.z;
                if constexpr (M >= 12u) out[r][11] = hi2.w;
            }
        }
        if constexpr (M > 12u) {
            if (mRows > 12u) {
                const float4 hi3 = qw_head_simd_sum4(float4(
                    acc[r][12],
                    (M >= 14u) ? acc[r][13] : 0.0f,
                    (M >= 15u) ? acc[r][14] : 0.0f,
                    (M >= 16u) ? acc[r][15] : 0.0f));
                out[r][12] = hi3.x;
                if constexpr (M >= 14u) out[r][13] = hi3.y;
                if constexpr (M >= 15u) out[r][14] = hi3.z;
                if constexpr (M >= 16u) out[r][15] = hi3.w;
            }
        }
    }
}

template<uint M, uint ROWS, bool DYN_M = false>
static inline void qw_head_bf16_gemv_body(
        device const ushort* W, device const ushort* x, device float* y,
        uint N, uint K, float softcap, uint tg, uint ti, uint tptg) {
    const uint realN = DYN_M ? (N & 0x00FFFFFFu) : (N & 0x3FFFFFFFu);
    const uint mRows = DYN_M ? min(M, (N >> 24u) & 0x1Fu) : M;
    const uint simdPerTG = max(1u, tptg >> 5u);
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    const uint row0 = (tg * simdPerTG + simd_id) * ROWS;
    if (row0 >= realN) return;

    float vals[ROWS][M];
    if ((N & 0x80000000u) != 0u) {
        qw_head_bf16_dot_rows<M, ROWS, true>(W, x, row0, realN, K, lane, vals, mRows);
    } else {
        qw_head_bf16_dot_rows<M, ROWS, false>(W, x, row0, realN, K, lane, vals, mRows);
    }

    if (lane != 0u) return;
    #pragma unroll
    for (uint r = 0u; r < ROWS; ++r) {
        const uint row = row0 + r;
        if (row < realN) {
            #pragma unroll
            for (uint m = 0u; m < M; ++m) {
                if (m >= mRows) break;
                y[(size_t)m * realN + row] = qw_head_softcap(vals[r][m], softcap);
            }
        }
    }
}

// All threads in the threadgroup enter together so the barrier stays uniform.
template<uint M, uint ROWS>
static inline void qw_head_publish_argmax(
        bool active, thread const float (&vals)[ROWS][M],
        uint row0, uint realN, uint nTG, float softcap,
        device uint* keysHi, device uint* keysLo,
        threadgroup uint* sh, threadgroup uint* sl,
        uint tg, uint ti, uint tptg, uint mRows = M) {
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    const uint simdPerTG = min(8u, max(1u, tptg >> 5u));

    if (lane == 0u && simd_id < 8u) {
        #pragma unroll
        for (uint m = 0u; m < M; ++m) {
            if (m >= mRows) break;
            uint bh = 0u;
            uint bl = 0u;
            if (active) {
                #pragma unroll
                for (uint r = 0u; r < ROWS; ++r) {
                    const uint row = row0 + r;
                    if (row < realN) {
                        const float v = qw_head_softcap(vals[r][m], softcap);
                        const uint kh = qw_sortable_f32_u32(v);
                        const uint kl = ~row;
                        if (r == 0u || kh > bh || (kh == bh && kl > bl)) {
                            bh = kh;
                            bl = kl;
                        }
                    }
                }
            }
            sh[simd_id * M + m] = bh;
            sl[simd_id * M + m] = bl;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (ti < mRows) {
        uint bh = sh[ti];
        uint bl = sl[ti];
        for (uint q = 1u; q < simdPerTG; ++q) {
            const uint kh = sh[q * M + ti];
            const uint kl = sl[q * M + ti];
            if (kh > bh || (kh == bh && kl > bl)) {
                bh = kh;
                bl = kl;
            }
        }
        keysHi[(size_t)ti * nTG + tg] = bh;
        keysLo[(size_t)ti * nTG + tg] = bl;
    }
}

template<uint M, uint ROWS, bool DYN_M = false>
static inline void qw_head_bf16_argmax_body(
        device const ushort* W, device const ushort* x,
        device uint* keysHi, device uint* keysLo,
        threadgroup uint* sh, threadgroup uint* sl,
        uint N, uint K, float softcap, uint tg, uint ti, uint tptg) {
    const uint realN = DYN_M ? (N & 0x00FFFFFFu) : (N & 0x3FFFFFFFu);
    const uint mRows = DYN_M ? min(M, (N >> 24u) & 0x1Fu) : M;
    const uint simdPerTG = max(1u, tptg >> 5u);
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    const uint row0 = (tg * simdPerTG + simd_id) * ROWS;
    const uint rowsPerTG = simdPerTG * ROWS;
    const uint nTG = (realN + rowsPerTG - 1u) / rowsPerTG;
    const bool active = (row0 < realN);

    float vals[ROWS][M];
    if (active) {
        if ((N & 0x80000000u) != 0u) {
            qw_head_bf16_dot_rows<M, ROWS, true>(W, x, row0, realN, K, lane, vals, mRows);
        } else {
            qw_head_bf16_dot_rows<M, ROWS, false>(W, x, row0, realN, K, lane, vals, mRows);
        }
    }
    qw_head_publish_argmax<M, ROWS>(active, vals, row0, realN, nTG, softcap,
                                    keysHi, keysLo, sh, sl, tg, ti, tptg, mRows);
}

template<uint M, uint ROWS>
static inline void qw_head_f16_dot_rows(
        device const half* W, device const half* x,
        uint row0, uint realN, uint K, uint lane,
        thread float (&out)[ROWS][M], uint mRows = M) {
    float acc[ROWS][M];
    #pragma unroll
    for (uint r = 0u; r < ROWS; ++r) {
        #pragma unroll
        for (uint m = 0u; m < M; ++m) acc[r][m] = 0.0f;
    }

    if ((K & 7u) == 0u) {
        const uint chunks = K >> 3u;
        device const QWHeadH8BF* wr[ROWS];
        #pragma unroll
        for (uint r = 0u; r < ROWS; ++r) {
            const uint row = (row0 + r < realN) ? (row0 + r) : row0;
            wr[r] = (device const QWHeadH8BF*)(W + (size_t)row * K);
        }
        device const QWHeadH8BF* xvPtr = (device const QWHeadH8BF*)x;
        for (uint c = lane; c < chunks; c += 32u) {
            if constexpr (ROWS == 4u) {
                QWHeadH8BF x8[M];
                #pragma unroll
                for (uint m = 0u; m < M; ++m) {
                    if (m >= mRows) break;
                    x8[m] = xvPtr[(size_t)m * chunks + c];
                }
                #pragma unroll
                for (uint r = 0u; r < ROWS; ++r) {
                    const QWHeadH8BF w8 = wr[r][c];
                    #pragma unroll
                    for (uint m = 0u; m < M; ++m) {
                        if (m >= mRows) break;
                        for (int i = 0; i < 8; ++i) {
                            acc[r][m] += float(w8.v[i]) * float(x8[m].v[i]);
                        }
                    }
                }
            } else {
                QWHeadH8BF w8[ROWS];
                #pragma unroll
                for (uint r = 0u; r < ROWS; ++r) w8[r] = wr[r][c];
                #pragma unroll
                for (uint m = 0u; m < M; ++m) {
                    if (m >= mRows) break;
                    const QWHeadH8BF xm = xvPtr[(size_t)m * chunks + c];
                    for (int i = 0; i < 8; ++i) {
                        const float xv = float(xm.v[i]);
                        #pragma unroll
                        for (uint r = 0u; r < ROWS; ++r) {
                            acc[r][m] += float(w8[r].v[i]) * xv;
                        }
                    }
                }
            }
        }
    } else {
        device const half* wr[ROWS];
        #pragma unroll
        for (uint r = 0u; r < ROWS; ++r) {
            const uint row = (row0 + r < realN) ? (row0 + r) : row0;
            wr[r] = W + (size_t)row * K;
        }
        for (uint k = lane; k < K; k += 32u) {
            float wv[ROWS];
            #pragma unroll
            for (uint r = 0u; r < ROWS; ++r) wv[r] = float(wr[r][k]);
            #pragma unroll
            for (uint m = 0u; m < M; ++m) {
                if (m >= mRows) break;
                const float xv = float(x[(size_t)m * K + k]);
                #pragma unroll
                for (uint r = 0u; r < ROWS; ++r) acc[r][m] += wv[r] * xv;
            }
        }
    }

    #pragma unroll
    for (uint r = 0u; r < ROWS; ++r) {
        const float4 lo = qw_head_simd_sum4(float4(
            acc[r][0],
            (M >= 2u) ? acc[r][1] : 0.0f,
            (M >= 3u) ? acc[r][2] : 0.0f,
            (M >= 4u) ? acc[r][3] : 0.0f));
        out[r][0] = lo.x;
        if constexpr (M >= 2u) out[r][1] = lo.y;
        if constexpr (M >= 3u) out[r][2] = lo.z;
        if constexpr (M >= 4u) out[r][3] = lo.w;
        if constexpr (M > 4u) {
            if (mRows > 4u) {
                const float4 hi = qw_head_simd_sum4(float4(
                    acc[r][4],
                    (M >= 6u) ? acc[r][5] : 0.0f,
                    (M >= 7u) ? acc[r][6] : 0.0f,
                    (M >= 8u) ? acc[r][7] : 0.0f));
                out[r][4] = hi.x;
                if constexpr (M >= 6u) out[r][5] = hi.y;
                if constexpr (M >= 7u) out[r][6] = hi.z;
                if constexpr (M >= 8u) out[r][7] = hi.w;
            }
        }
        if constexpr (M > 8u) {
            if (mRows > 8u) {
                const float4 hi2 = qw_head_simd_sum4(float4(
                    acc[r][8],
                    (M >= 10u) ? acc[r][9] : 0.0f,
                    (M >= 11u) ? acc[r][10] : 0.0f,
                    (M >= 12u) ? acc[r][11] : 0.0f));
                out[r][8] = hi2.x;
                if constexpr (M >= 10u) out[r][9] = hi2.y;
                if constexpr (M >= 11u) out[r][10] = hi2.z;
                if constexpr (M >= 12u) out[r][11] = hi2.w;
            }
        }
        if constexpr (M > 12u) {
            if (mRows > 12u) {
                const float4 hi3 = qw_head_simd_sum4(float4(
                    acc[r][12],
                    (M >= 14u) ? acc[r][13] : 0.0f,
                    (M >= 15u) ? acc[r][14] : 0.0f,
                    (M >= 16u) ? acc[r][15] : 0.0f));
                out[r][12] = hi3.x;
                if constexpr (M >= 14u) out[r][13] = hi3.y;
                if constexpr (M >= 15u) out[r][14] = hi3.z;
                if constexpr (M >= 16u) out[r][15] = hi3.w;
            }
        }
    }
}

template<uint M, uint ROWS, bool DYN_M = false>
static inline void qw_head_f16_argmax_body(
        device const half* W, device const half* x,
        device uint* keysHi, device uint* keysLo,
        threadgroup uint* sh, threadgroup uint* sl,
        uint N, uint K, float softcap, uint tg, uint ti, uint tptg) {
    const uint realN = DYN_M ? (N & 0x00FFFFFFu) : (N & 0x3FFFFFFFu);
    const uint mRows = DYN_M ? min(M, (N >> 24u) & 0x1Fu) : M;
    const uint simdPerTG = max(1u, tptg >> 5u);
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    const uint row0 = (tg * simdPerTG + simd_id) * ROWS;
    const uint rowsPerTG = simdPerTG * ROWS;
    const uint nTG = (realN + rowsPerTG - 1u) / rowsPerTG;
    const bool active = (row0 < realN);

    float vals[ROWS][M];
    if (active) {
        qw_head_f16_dot_rows<M, ROWS>(W, x, row0, realN, K, lane, vals, mRows);
    }
    qw_head_publish_argmax<M, ROWS>(active, vals, row0, realN, nTG, softcap,
                                    keysHi, keysLo, sh, sl, tg, ti, tptg, mRows);
}

template<uint M, uint ROWS = 2u, bool DYN_M = false>
static inline void qw_head_q4g_argmax_body(
        device const Block_HQ4_G32* blocks, device const half* x,
        device uint* keysHi, device uint* keysLo,
        threadgroup uint* sh, threadgroup uint* sl,
        uint N, uint K, float softcap, uint tg, uint ti, uint tptg) {
    const uint realN = DYN_M ? (N & 0x00FFFFFFu) : (N & 0x3FFFFFFFu);
    const uint mRows = DYN_M ? min(M, (N >> 24u) & 0x1Fu) : M;
    const uint simdPerTG = max(1u, tptg >> 5u);
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    const uint row0 = (tg * simdPerTG + simd_id) * ROWS;
    const uint rowsPerTG = simdPerTG * ROWS;
    const uint nTG = (realN + rowsPerTG - 1u) / rowsPerTG;
    const bool active = (row0 < realN);

    float vals[ROWS][M];
    if (active) {
        const uint nBlocks = K >> 5u;
        device const Block_HQ4_G32* const r0 = blocks + (size_t)row0 * nBlocks;
        if constexpr (ROWS == 2u) {
            device const Block_HQ4_G32* const rows[2] = { r0, (row0 + 1u < realN) ? (r0 + nBlocks) : r0 };
            if (QW_IS_AQ4) qwg_gemv_hq4_rows_r_m_t<M, 2u, true>(rows, x, K, lane, vals, mRows);
            else qwg_gemv_hq4_rows_r_m_t<M, 2u, false>(rows, x, K, lane, vals, mRows);
        } else {
            device const Block_HQ4_G32* const rows[1] = { r0 };
            if (QW_IS_AQ4) qwg_gemv_hq4_rows_r_m_t<M, 1u, true>(rows, x, K, lane, vals, mRows);
            else qwg_gemv_hq4_rows_r_m_t<M, 1u, false>(rows, x, K, lane, vals, mRows);
        }
    }
    qw_head_publish_argmax<M, ROWS>(active, vals, row0, realN, nTG, softcap,
                                    keysHi, keysLo, sh, sl, tg, ti, tptg, mRows);
}

#define QW_HEAD_BF16_ENTRY(M, ROWS) \
kernel void qwg_head_bf16_m##M( \
        device const ushort* W [[buffer(0)]], \
        device const ushort* x [[buffer(1)]], \
        device float* y [[buffer(3)]], \
        constant uint& N [[buffer(4)]], \
        constant uint& K [[buffer(5)]], \
        constant float& softcap [[buffer(6)]], \
        uint tg [[threadgroup_position_in_grid]], \
        uint ti [[thread_index_in_threadgroup]], \
        uint tptg [[threads_per_threadgroup]]) { \
    qw_head_bf16_gemv_body<M, ROWS>(W, x, y, N, K, softcap, tg, ti, tptg); \
}

#define QW_HEAD_BF16_ARGMAX_ENTRY(M, ROWS) \
kernel void qwg_head_bf16_argmax_m##M( \
        device const ushort* W [[buffer(0)]], \
        device const ushort* x [[buffer(1)]], \
        device uint* keysHi [[buffer(2)]], \
        device uint* keysLo [[buffer(3)]], \
        constant uint& N [[buffer(4)]], \
        constant uint& K [[buffer(5)]], \
        constant float& softcap [[buffer(6)]], \
        uint tg [[threadgroup_position_in_grid]], \
        uint ti [[thread_index_in_threadgroup]], \
        uint tptg [[threads_per_threadgroup]]) { \
    threadgroup uint sh[8 * M]; \
    threadgroup uint sl[8 * M]; \
    qw_head_bf16_argmax_body<M, ROWS>(W, x, keysHi, keysLo, sh, sl, N, K, softcap, tg, ti, tptg); \
}

#define QW_HEAD_F16_ARGMAX_ENTRY(M, ROWS) \
kernel void qwg_head_f16_argmax_m##M( \
        device const half* W [[buffer(0)]], \
        device const half* x [[buffer(1)]], \
        device uint* keysHi [[buffer(2)]], \
        device uint* keysLo [[buffer(3)]], \
        constant uint& N [[buffer(4)]], \
        constant uint& K [[buffer(5)]], \
        constant float& softcap [[buffer(6)]], \
        uint tg [[threadgroup_position_in_grid]], \
        uint ti [[thread_index_in_threadgroup]], \
        uint tptg [[threads_per_threadgroup]]) { \
    threadgroup uint sh[8 * M]; \
    threadgroup uint sl[8 * M]; \
    qw_head_f16_argmax_body<M, ROWS>(W, x, keysHi, keysLo, sh, sl, N, K, softcap, tg, ti, tptg); \
}

#define QW_HEAD_Q4G_ARGMAX_ENTRY(M) \
kernel void qwg_head_q4g_argmax_m##M( \
        device const Block_HQ4_G32* blocks [[buffer(0)]], \
        device const half* x [[buffer(1)]], \
        device uint* keysHi [[buffer(2)]], \
        device uint* keysLo [[buffer(3)]], \
        constant uint& N [[buffer(4)]], \
        constant uint& K [[buffer(5)]], \
        constant float& softcap [[buffer(6)]], \
        uint tg [[threadgroup_position_in_grid]], \
        uint ti [[thread_index_in_threadgroup]], \
        uint tptg [[threads_per_threadgroup]]) { \
    threadgroup uint sh[8 * M]; \
    threadgroup uint sl[8 * M]; \
    qw_head_q4g_argmax_body<M>(blocks, x, keysHi, keysLo, sh, sl, N, K, softcap, tg, ti, tptg); \
}

QW_HEAD_BF16_ENTRY(1, 4u)
QW_HEAD_BF16_ENTRY(2, 4u)
QW_HEAD_BF16_ENTRY(3, 4u)
QW_HEAD_BF16_ENTRY(4, 4u)
QW_HEAD_BF16_ENTRY(5, 2u)
QW_HEAD_BF16_ENTRY(6, 2u)
QW_HEAD_BF16_ENTRY(7, 2u)
QW_HEAD_BF16_ENTRY(8, 2u)

QW_HEAD_BF16_ARGMAX_ENTRY(1, 4u)
QW_HEAD_BF16_ARGMAX_ENTRY(2, 4u)
QW_HEAD_BF16_ARGMAX_ENTRY(3, 4u)
QW_HEAD_BF16_ARGMAX_ENTRY(4, 4u)
QW_HEAD_BF16_ARGMAX_ENTRY(5, 2u)
QW_HEAD_BF16_ARGMAX_ENTRY(6, 2u)
QW_HEAD_BF16_ARGMAX_ENTRY(7, 2u)
QW_HEAD_BF16_ARGMAX_ENTRY(8, 2u)

QW_HEAD_F16_ARGMAX_ENTRY(1, 4u)
QW_HEAD_F16_ARGMAX_ENTRY(2, 4u)
QW_HEAD_F16_ARGMAX_ENTRY(3, 4u)
QW_HEAD_F16_ARGMAX_ENTRY(4, 4u)
QW_HEAD_F16_ARGMAX_ENTRY(5, 2u)
QW_HEAD_F16_ARGMAX_ENTRY(6, 2u)
QW_HEAD_F16_ARGMAX_ENTRY(7, 2u)
QW_HEAD_F16_ARGMAX_ENTRY(8, 2u)

QW_HEAD_Q4G_ARGMAX_ENTRY(1)
QW_HEAD_Q4G_ARGMAX_ENTRY(2)
QW_HEAD_Q4G_ARGMAX_ENTRY(3)
QW_HEAD_Q4G_ARGMAX_ENTRY(4)
QW_HEAD_Q4G_ARGMAX_ENTRY(5)
QW_HEAD_Q4G_ARGMAX_ENTRY(6)
QW_HEAD_Q4G_ARGMAX_ENTRY(7)
QW_HEAD_Q4G_ARGMAX_ENTRY(8)

kernel void qwg_head_bf16_m16(
        device const ushort* W [[buffer(0)]],
        device const ushort* x [[buffer(1)]],
        device float* y [[buffer(3)]],
        constant uint& N [[buffer(4)]],
        constant uint& K [[buffer(5)]],
        constant float& softcap [[buffer(6)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]],
        uint tptg [[threads_per_threadgroup]]) {
    qw_head_bf16_gemv_body<16u, 2u, true>(W, x, y, N, K, softcap, tg, ti, tptg);
}

kernel void qwg_head_bf16_argmax_m16(
        device const ushort* W [[buffer(0)]],
        device const ushort* x [[buffer(1)]],
        device uint* keysHi [[buffer(2)]],
        device uint* keysLo [[buffer(3)]],
        constant uint& N [[buffer(4)]],
        constant uint& K [[buffer(5)]],
        constant float& softcap [[buffer(6)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]],
        uint tptg [[threads_per_threadgroup]]) {
    threadgroup uint sh[8 * 16];
    threadgroup uint sl[8 * 16];
    qw_head_bf16_argmax_body<16u, 2u, true>(W, x, keysHi, keysLo, sh, sl, N, K, softcap, tg, ti, tptg);
}

kernel void qwg_head_f16_argmax_m16(
        device const half* W [[buffer(0)]],
        device const half* x [[buffer(1)]],
        device uint* keysHi [[buffer(2)]],
        device uint* keysLo [[buffer(3)]],
        constant uint& N [[buffer(4)]],
        constant uint& K [[buffer(5)]],
        constant float& softcap [[buffer(6)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]],
        uint tptg [[threads_per_threadgroup]]) {
    threadgroup uint sh[8 * 16];
    threadgroup uint sl[8 * 16];
    qw_head_f16_argmax_body<16u, 2u, true>(W, x, keysHi, keysLo, sh, sl, N, K, softcap, tg, ti, tptg);
}

kernel void qwg_head_q4g_argmax_m16(
        device const Block_HQ4_G32* blocks [[buffer(0)]],
        device const half* x [[buffer(1)]],
        device uint* keysHi [[buffer(2)]],
        device uint* keysLo [[buffer(3)]],
        constant uint& N [[buffer(4)]],
        constant uint& K [[buffer(5)]],
        constant float& softcap [[buffer(6)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]],
        uint tptg [[threads_per_threadgroup]]) {
    threadgroup uint sh[8 * 16];
    threadgroup uint sl[8 * 16];
    qw_head_q4g_argmax_body<16u, 1u, true>(blocks, x, keysHi, keysLo, sh, sl, N, K, softcap, tg, ti, tptg);
}

static inline void qw_head_argmax_batched_fin_body(
        device const uint* keysHi,
        device const uint* keysLo,
        device int32_t*    tokens,
        threadgroup uint*  sh,
        threadgroup uint*  sl,
        uint nTG, uint M, float softcap,
        uint3 tg, uint tid, uint3 tptg3) {
    const uint row = (tg.y > 0u) ? tg.y : tg.x;
    const uint tptg = tptg3.x * tptg3.y * tptg3.z;
    if (row >= M) return;
    device const uint* rHi = keysHi + (size_t)row * nTG;
    device const uint* rLo = keysLo + (size_t)row * nTG;
    uint bh = 0u, bl = 0u;
    for (uint q = tid; q < nTG; q += tptg) {
        uint h = rHi[q];
        const uint l = rLo[q];
        if (softcap > 0.0f && l != 0u) {
            const uint raw = ((h & 0x80000000u) != 0u) ? (h ^ 0x80000000u) : ~h;
            h = qw_sortable_f32_u32(qw_head_softcap(as_type<float>(raw), softcap));
        }
        if (h > bh || (h == bh && l > bl)) {
            bh = h;
            bl = l;
        }
    }
    const uint maxHi = simd_max(bh);
    const uint maxLo = simd_max((bh == maxHi) ? bl : 0u);
    const uint lane = tid & 31u;
    const uint simd_id = tid >> 5u;
    if (lane == 0u && simd_id < 32u) {
        sh[simd_id] = maxHi;
        sl[simd_id] = maxLo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nSg = min(32u, max(1u, tptg >> 5u));
    if (tid == 0u) {
        uint bh2 = sh[0];
        uint bl2 = sl[0];
        for (uint q = 1u; q < nSg; ++q) {
            if (sh[q] > bh2 || (sh[q] == bh2 && sl[q] > bl2)) {
                bh2 = sh[q];
                bl2 = sl[q];
            }
        }
        tokens[row] = (nTG > 0u) ? int32_t(~bl2) : 0;
    }
}

kernel void qwg_head_argmax_batched_fin(
        device const uint* keysHi [[buffer(0)]],
        device const uint* keysLo [[buffer(1)]],
        device int32_t*    tokens [[buffer(2)]],
        constant uint&     nTG    [[buffer(3)]],
        constant uint&     M      [[buffer(4)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 tptg3 [[threads_per_threadgroup]]) {
    threadgroup uint sh[32];
    threadgroup uint sl[32];
    qw_head_argmax_batched_fin_body(keysHi, keysLo, tokens, sh, sl, nTG, M, 0.0f, tg, tid, tptg3);
}

kernel void qwg_head_argmax_batched_fin_softcap(
        device const uint* keysHi  [[buffer(0)]],
        device const uint* keysLo  [[buffer(1)]],
        device int32_t*    tokens  [[buffer(2)]],
        constant uint&     nTG     [[buffer(3)]],
        constant uint&     M       [[buffer(4)]],
        constant float&    softcap [[buffer(5)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 tptg3 [[threads_per_threadgroup]]) {
    threadgroup uint sh[32];
    threadgroup uint sl[32];
    qw_head_argmax_batched_fin_body(keysHi, keysLo, tokens, sh, sl, nTG, M, softcap, tg, tid, tptg3);
}
