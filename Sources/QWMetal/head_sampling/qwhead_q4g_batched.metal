// qwhead_q4g_batched.metal — batched HQ4_G32_M5 lm_head: ONE pass over the
// 20-byte Block_HQ4_G32 weight blocks feeds all M verify rows.
//
// BIT-PIN LAW: per-row math is the EXACT body of qwg_gemv_q4g_v4
// (qwg_dot_hq4_block + shuffle-down tree 16/8/4/2/1; lane 0 stores).
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_dequant.metalh"
using namespace metal;

template<uint M>
static inline void qwhead_q4g_batched(
        device const Block_HQ4_G32* blocks,
        constant const half* x,
        device float* y, uint N, uint K, uint tg, uint ti, uint tptg = 32u) {
    const bool isAQ4 = (N & 0x80000000u) != 0u;
    const uint realN = N & 0x7FFFFFFFu;
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    const uint row = tg * max(1u, tptg >> 5u) + simd_id;
    if (row >= realN) return;
    float acc[M];
    qwg_gemv_hq4_row_m<M>(blocks, x, row, K, lane, acc, isAQ4);
    if (lane == 0u) {
        #pragma unroll
        for (uint r = 0; r < M; r++) y[(size_t)r * realN + row] = acc[r];
    }
}

/// Two weight rows per simdgroup share each x load (bit-identical per row for HQ4 and AQ4).
template<uint M>
static inline void qwhead_q4g_batched_rp(
        device const Block_HQ4_G32* blocks,
        constant const half* x,
        device float* y, uint N, uint K, uint tg, uint ti, uint tptg) {
    const bool isAQ4 = (N & 0x80000000u) != 0u;
    const uint realN = N & 0x7FFFFFFFu;
    const uint row0 = (tg * (tptg >> 5u) + (ti >> 5u)) * 2u;
    if (row0 >= realN) return;
    const uint nBlocks = K >> 5u;
    device const Block_HQ4_G32* rows[2] = {blocks + (size_t)row0 * nBlocks,
                                           blocks + (size_t)min(row0 + 1u, realN - 1u) * nBlocks};
    float v[2][M];
    qwg_gemv_hq4_rows_r_m<M, 2>(rows, x, K, ti & 31u, v, isAQ4);
    if ((ti & 31u) != 0u) return;
    #pragma unroll
    for (uint r = 0; r < M; r++) {
        y[(size_t)r * realN + row0] = v[0][r];
        if (row0 + 1u < realN) y[(size_t)r * realN + row0 + 1u] = v[1][r];
    }
}

#define QW_Q4G_RP_ENTRY(M) \
kernel void qwg_head_q4g_rp_m##M( \
        device const Block_HQ4_G32* blocks [[buffer(0)]], \
        constant const half* x [[buffer(1)]], \
        device float* y [[buffer(3)]], \
        constant uint& N [[buffer(4)]], \
        constant uint& K [[buffer(5)]], \
        uint tg [[threadgroup_position_in_grid]], \
        uint ti [[thread_index_in_threadgroup]], \
        uint tptg [[threads_per_threadgroup]]) { \
    qwhead_q4g_batched_rp<M>(blocks, x, y, N, K, tg, ti, tptg); \
}
QW_Q4G_RP_ENTRY(4)
QW_Q4G_RP_ENTRY(5)
QW_Q4G_RP_ENTRY(6)
QW_Q4G_RP_ENTRY(7)
QW_Q4G_RP_ENTRY(8)

#define QW_Q4G_ENTRY(M) \
kernel void qwg_head_q4g_m##M( \
        device const Block_HQ4_G32* blocks [[buffer(0)]], \
        constant const half* x [[buffer(1)]], \
        device float* y [[buffer(3)]], \
        constant uint& N [[buffer(4)]], \
        constant uint& K [[buffer(5)]], \
        uint tg [[threadgroup_position_in_grid]], \
        uint ti [[thread_index_in_threadgroup]], \
        uint tptg [[threads_per_threadgroup]]) { \
    qwhead_q4g_batched<M>(blocks, x, y, N, K, tg, ti, tptg); \
}
QW_Q4G_ENTRY(2)
QW_Q4G_ENTRY(3)
QW_Q4G_ENTRY(4)
QW_Q4G_ENTRY(5)
QW_Q4G_ENTRY(6)
QW_Q4G_ENTRY(7)
QW_Q4G_ENTRY(8)
