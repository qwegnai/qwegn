// qwgemm_aq_gdn.metal — the size-preserving affine family's GDN in-projection
// decode twins: the incumbent's `qwg_gemv_q4g_dx_h16_gdn_inproj_t4` over the
// family's own blocks. The same geometry and the same epilogue; the family's
// arithmetic, and NO rotation — the family's weights carry none.

// buffers: 0 blocks | 1 x f16 | 3 y f32 | 4 N u32 | 5 K u32 | 8 y f16
// 9 aLog | 10 dtBias | 11 betas | 12 glog

#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_dequant_multi.metalh"
#include "include/qw_geometry_constants.metalh"
using namespace metal;

/// The GDN's row epilogue: the QKV and Z rows, the A and B rows, and their
/// f32 companions. The weights do not enter here, so both containers compute
/// the same thing from the same dot.
static inline void qwg_aq_gdn_inproj_store_row(uint row, float vx, device half* y16,
                                               device float* y32, device const float* aLog,
                                               device const float* dtBias, device float* betas,
                                               device float* glog) {
    const uint aRow0 = QW_GDN_QKV_CH + QW_GDN_Z_CH;
    const uint bRow0 = aRow0 + QW_GDN_V_HEADS;
    const uint totalN = bRow0 + QW_GDN_V_HEADS;
    if (row < aRow0) {
        y16[row] = half(vx);
    } else if (row < bRow0) {
        const uint h = row - aRow0;
        const float sp = log(1.0f + exp(vx + dtBias[h]));
        glog[h] = -exp(aLog[h]) * sp;
        if (y32) y32[row] = vx;
        y16[row] = half(vx);
    } else if (row < totalN) {
        const uint h = row - bRow0;
        betas[h] = 1.0f / (1.0f + exp(-vx));
        if (y32) y32[row] = vx;
        y16[row] = half(vx);
    }
}

/// Four rows per simdgroup, as the incumbent's `_t4` does.
template <int BITS, int GROUP>
static inline void qwg_gemv_aq_gdn_inproj_rows4_body(device const char* blocks,
                                                     device const half* x,
                                                     device float* y32,
                                                     uint realN,
                                                     constant uint& K,
                                                     device half* y16,
                                                     device const float* aLog,
                                                     device const float* dtBias,
                                                     device float* betas,
                                                     device float* glog,
                                                     uint tg,
                                                     uint ti,
                                                     uint tptg = 128u) {
    uint simd_per_tg = max(1u, tptg >> 5u);
    uint simd_id = ti >> 5u;
    uint lane = ti & 31u;
    uint row0 = (tg * simd_per_tg + simd_id) * 4u;
    if (row0 >= realN) return;
    float4 acc = qwg_gemv_aq_rowsN<BITS, GROUP, 4u>(blocks, x, row0, realN, K, lane);
    if (lane == 0u) {
        qwg_aq_gdn_inproj_store_row(row0, acc.x, y16, y32, aLog, dtBias, betas, glog);
        if (row0 + 1u < realN) qwg_aq_gdn_inproj_store_row(row0 + 1u, acc.y, y16, y32, aLog, dtBias, betas, glog);
        if (row0 + 2u < realN) qwg_aq_gdn_inproj_store_row(row0 + 2u, acc.z, y16, y32, aLog, dtBias, betas, glog);
        if (row0 + 3u < realN) qwg_aq_gdn_inproj_store_row(row0 + 3u, acc.w, y16, y32, aLog, dtBias, betas, glog);
    }
}

/// Multi-row (M = 2..4) GDN in-projection over 4 output rows per simdgroup, loading each weight block once.
template <int BITS, int GROUP>
static inline void qwg_gemv_aq_gdn_inproj_rows4_m_body(device const char* blocks,
                                                       device const half* x,
                                                       uint realN,
                                                       constant uint& K,
                                                       device half* y16,
                                                       device const float* aLog,
                                                       device const float* dtBias,
                                                       device float* controls,
                                                       uint mRows,
                                                       uint tg,
                                                       uint ti,
                                                       uint tptg = 128u) {
    const uint simd_per_tg = max(1u, tptg >> 5u);
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    const uint row0 = (tg * simd_per_tg + simd_id) * 4u;
    if (row0 >= realN) return;
    const uint nBlocks = K >> 5u;
    const uint groupBytes = GROUP * BITS / 8 + 4u;
    const uint rowBytes = (K / GROUP) * groupBytes;
    device const char* r0 = blocks + (size_t)row0 * rowBytes;
    device const char* r1 = (row0 + 1u < realN) ? (r0 + rowBytes) : r0;
    device const char* r2 = (row0 + 2u < realN) ? (r0 + 2u * rowBytes) : r0;
    device const char* r3 = (row0 + 3u < realN) ? (r0 + 3u * rowBytes) : r0;
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f), acc3 = float4(0.0f);
    for (uint g = lane; g < nBlocks; g += 32u) {
        const uint col = g << 5u;
        const size_t groupOff = size_t(col / GROUP) * groupBytes;
        const uint byteInGroup = (col % GROUP) * BITS / 8u;
        device const char* gp0 = r0 + groupOff;
        device const char* gp1 = r1 + groupOff;
        device const char* gp2 = r2 + groupOff;
        device const char* gp3 = r3 + groupOff;
        {
            half4 xlo0[4], xhi0[4], xlo1[4], xhi1[4];
            const float2 sumX = float2(
                qwg_aq_load_prescaled32<BITS>(x + col, xlo0, xhi0),
                qwg_aq_load_prescaled32<BITS>(x + (size_t)K + col, xlo1, xhi1));
            const float2 d0 = qwg_aq_slice_dot2<BITS>(gp0, byteInGroup, xlo0, xhi0, xlo1, xhi1, sumX);
            const float2 d1 = qwg_aq_slice_dot2<BITS>(gp1, byteInGroup, xlo0, xhi0, xlo1, xhi1, sumX);
            const float2 d2 = qwg_aq_slice_dot2<BITS>(gp2, byteInGroup, xlo0, xhi0, xlo1, xhi1, sumX);
            const float2 d3 = qwg_aq_slice_dot2<BITS>(gp3, byteInGroup, xlo0, xhi0, xlo1, xhi1, sumX);
            acc0 += float4(d0.x, d1.x, d2.x, d3.x);
            acc1 += float4(d0.y, d1.y, d2.y, d3.y);
        }
        if (mRows == 3u) {
            half4 xlo2[4], xhi2[4];
            const float sumX2 = qwg_aq_load_prescaled32<BITS>(x + (size_t)2u * K + col, xlo2, xhi2);
            acc2.x += qwg_aq_slice_dot<BITS>(gp0, byteInGroup, xlo2, xhi2, sumX2);
            acc2.y += qwg_aq_slice_dot<BITS>(gp1, byteInGroup, xlo2, xhi2, sumX2);
            acc2.z += qwg_aq_slice_dot<BITS>(gp2, byteInGroup, xlo2, xhi2, sumX2);
            acc2.w += qwg_aq_slice_dot<BITS>(gp3, byteInGroup, xlo2, xhi2, sumX2);
        } else if (mRows >= 4u) {
            half4 xlo2[4], xhi2[4], xlo3[4], xhi3[4];
            const float2 sumX23 = float2(
                qwg_aq_load_prescaled32<BITS>(x + (size_t)2u * K + col, xlo2, xhi2),
                qwg_aq_load_prescaled32<BITS>(x + (size_t)3u * K + col, xlo3, xhi3));
            const float2 d0 = qwg_aq_slice_dot2<BITS>(gp0, byteInGroup, xlo2, xhi2, xlo3, xhi3, sumX23);
            const float2 d1 = qwg_aq_slice_dot2<BITS>(gp1, byteInGroup, xlo2, xhi2, xlo3, xhi3, sumX23);
            const float2 d2 = qwg_aq_slice_dot2<BITS>(gp2, byteInGroup, xlo2, xhi2, xlo3, xhi3, sumX23);
            const float2 d3 = qwg_aq_slice_dot2<BITS>(gp3, byteInGroup, xlo2, xhi2, xlo3, xhi3, sumX23);
            acc2 += float4(d0.x, d1.x, d2.x, d3.x);
            acc3 += float4(d0.y, d1.y, d2.y, d3.y);
        }
    }
    acc0 += simd_shuffle_down(acc0, 16); acc1 += simd_shuffle_down(acc1, 16);
    acc0 += simd_shuffle_down(acc0, 8);  acc1 += simd_shuffle_down(acc1, 8);
    acc0 += simd_shuffle_down(acc0, 4);  acc1 += simd_shuffle_down(acc1, 4);
    acc0 += simd_shuffle_down(acc0, 2);  acc1 += simd_shuffle_down(acc1, 2);
    acc0 += simd_shuffle_down(acc0, 1);  acc1 += simd_shuffle_down(acc1, 1);
    if (mRows >= 3u) {
        acc2 += simd_shuffle_down(acc2, 16);
        acc2 += simd_shuffle_down(acc2, 8);
        acc2 += simd_shuffle_down(acc2, 4);
        acc2 += simd_shuffle_down(acc2, 2);
        acc2 += simd_shuffle_down(acc2, 1);
    }
    if (mRows >= 4u) {
        acc3 += simd_shuffle_down(acc3, 16);
        acc3 += simd_shuffle_down(acc3, 8);
        acc3 += simd_shuffle_down(acc3, 4);
        acc3 += simd_shuffle_down(acc3, 2);
        acc3 += simd_shuffle_down(acc3, 1);
    }
    if (lane == 0u) {
        const float4 accs[4] = { acc0, acc1, acc2, acc3 };
        const uint ctrlStride = 2u * QW_GDN_V_HEADS;
        for (uint m = 0u; m < mRows; ++m) {
            device half* ym = y16 + (size_t)m * realN;
            device float* bm = controls + (size_t)m * ctrlStride;
            device float* gm = bm + QW_GDN_V_HEADS;
            #pragma unroll
            for (uint r = 0u; r < 4u; ++r) {
                if (row0 + r < realN) {
                    qwg_aq_gdn_inproj_store_row(row0 + r, accs[m][r], ym, nullptr, aLog, dtBias, bm, gm);
                }
            }
        }
    }
}

/// One row per simdgroup, the incumbent's plain form — the fallback the
/// `_t4` kernel takes when the rows4 flag is clear. It uses the family's own
/// per-element arithmetic, like the rows4 body, so the file has ONE rounding
/// and the two shapes differ only in how they cover the rows.
template <int BITS, int GROUP>
static inline void qwg_gemv_aq_gdn_inproj_body(device const char* blocks,
                                               device const half* x,
                                               device float* y32,
                                               constant uint& N,
                                               constant uint& K,
                                               device half* y16,
                                               device const float* aLog,
                                               device const float* dtBias,
                                               device float* betas,
                                               device float* glog,
                                               uint tg,
                                               uint ti,
                                               uint tptg = 32u) {
    uint rows_per_tg = max(1u, tptg >> 5u);
    uint simd_id = ti >> 5u;
    uint lane = ti & 31u;
    uint row = tg * rows_per_tg + simd_id;
    if (row >= N) return;
    float4 acc = qwg_gemv_aq_rowsN<BITS, GROUP, 1u>(blocks, x, row, N, K, lane);
    if (lane == 0u) {
        qwg_aq_gdn_inproj_store_row(row, acc.x, y16, y32, aLog, dtBias, betas, glog);
    }
}

#define QW_AQ_GDN_DECODE_T4_KERNEL(BITS, GROUP, NAME)                  \
    kernel void NAME(device const char* blocks [[buffer(0)]],          \
                     device const half* x [[buffer(1)]],               \
                     device float* y32 [[buffer(3)]],                  \
                     constant uint& N [[buffer(4)]],                   \
                     constant uint& K [[buffer(5)]],                   \
                     device half* y16 [[buffer(8)]],                   \
                     device const float* aLog [[buffer(9)]],            \
                     device const float* dtBias [[buffer(10)]],         \
                     device float* betas [[buffer(11)]],                \
                     device float* glog [[buffer(12)]],                 \
                     uint tg [[threadgroup_position_in_grid]],           \
                     uint ti [[thread_index_in_threadgroup]],           \
                     uint tptg [[threads_per_threadgroup]]) {           \
        if ((N & 0x40000000u) != 0u) {                                 \
            const uint mRows = (N >> 24u) & 0x3Fu;                     \
            if (mRows >= 2u) {                                         \
                qwg_gemv_aq_gdn_inproj_rows4_m_body<BITS, GROUP>(      \
                    blocks, x, N & 0x00FFFFFFu, K, y16, aLog, dtBias,  \
                    betas, mRows, tg, ti, tptg);                       \
            } else {                                                   \
                qwg_gemv_aq_gdn_inproj_rows4_body<BITS, GROUP>(        \
                    blocks, x, y32, N & 0x3FFFFFFFu, K, y16, aLog, dtBias, \
                    betas, glog, tg, ti, tptg);                        \
            }                                                          \
        } else {                                                       \
            qwg_gemv_aq_gdn_inproj_body<BITS, GROUP>(                  \
                blocks, x, y32, N, K, y16, aLog, dtBias, betas, glog, tg, ti, tptg); \
        }                                                              \
    }

QW_AQ_GDN_DECODE_T4_KERNEL(2, 32, qwg_gemv_aq2_g32_dx_h16_gdn_inproj_t4)
QW_AQ_GDN_DECODE_T4_KERNEL(2, 64, qwg_gemv_aq2_g64_dx_h16_gdn_inproj_t4)
QW_AQ_GDN_DECODE_T4_KERNEL(2, 128, qwg_gemv_aq2_g128_dx_h16_gdn_inproj_t4)
QW_AQ_GDN_DECODE_T4_KERNEL(2, 256, qwg_gemv_aq2_g256_dx_h16_gdn_inproj_t4)
QW_AQ_GDN_DECODE_T4_KERNEL(3, 32, qwg_gemv_aq3_g32_dx_h16_gdn_inproj_t4)
QW_AQ_GDN_DECODE_T4_KERNEL(3, 64, qwg_gemv_aq3_g64_dx_h16_gdn_inproj_t4)
QW_AQ_GDN_DECODE_T4_KERNEL(3, 128, qwg_gemv_aq3_g128_dx_h16_gdn_inproj_t4)
QW_AQ_GDN_DECODE_T4_KERNEL(3, 256, qwg_gemv_aq3_g256_dx_h16_gdn_inproj_t4)
QW_AQ_GDN_DECODE_T4_KERNEL(4, 32, qwg_gemv_aq4_g32_dx_h16_gdn_inproj_t4)
QW_AQ_GDN_DECODE_T4_KERNEL(4, 64, qwg_gemv_aq4_g64_dx_h16_gdn_inproj_t4)
QW_AQ_GDN_DECODE_T4_KERNEL(4, 128, qwg_gemv_aq4_g128_dx_h16_gdn_inproj_t4)
QW_AQ_GDN_DECODE_T4_KERNEL(4, 256, qwg_gemv_aq4_g256_dx_h16_gdn_inproj_t4)
QW_AQ_GDN_DECODE_T4_KERNEL(6, 32, qwg_gemv_aq6_g32_dx_h16_gdn_inproj_t4)
QW_AQ_GDN_DECODE_T4_KERNEL(6, 64, qwg_gemv_aq6_g64_dx_h16_gdn_inproj_t4)
QW_AQ_GDN_DECODE_T4_KERNEL(6, 128, qwg_gemv_aq6_g128_dx_h16_gdn_inproj_t4)
QW_AQ_GDN_DECODE_T4_KERNEL(6, 256, qwg_gemv_aq6_g256_dx_h16_gdn_inproj_t4)
QW_AQ_GDN_DECODE_T4_KERNEL(8, 32, qwg_gemv_aq8_g32_dx_h16_gdn_inproj_t4)
QW_AQ_GDN_DECODE_T4_KERNEL(8, 64, qwg_gemv_aq8_g64_dx_h16_gdn_inproj_t4)
QW_AQ_GDN_DECODE_T4_KERNEL(8, 128, qwg_gemv_aq8_g128_dx_h16_gdn_inproj_t4)
QW_AQ_GDN_DECODE_T4_KERNEL(8, 256, qwg_gemv_aq8_g256_dx_h16_gdn_inproj_t4)
