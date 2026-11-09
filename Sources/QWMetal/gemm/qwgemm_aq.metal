// qwgemm_aq.metal — the size-preserving affine-quantised decode GEMV, in the
// dense path's geometry: the incumbent's `dx_h16_t4`, down to the register
// budget. The arithmetic is the family's — the per-element affine form
// `code * scale + bias` for every width, with the group's own scale and bias
// and no fold. specs/04-qwquant.md has the design and the protocol.

// buffers: 0 blocks | 1 x f16 | 3 y f32 | 4 N u32 | 5 K u32 | 8 y f16
// N carries the row count in its flags: 0x4000_0000 four rows,
// 0x8000_0000 two rows, otherwise one.

#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_dequant_multi.metalh"
using namespace metal;

/// The decode GEMV: `ROWS` output rows per simdgroup.
///
/// The kernel writes f16 only, as the incumbent's `_dx_h16_t4` does: the f32
/// output would share a buffer with the f16 one, and a kernel that wrote both
/// would overwrite its own answer.
template <int BITS, int GROUP, uint ROWS>
static inline void qwg_gemv_aq_body(device const char* blocks,
                                    device const half* x,
                                    uint realN,
                                    constant uint& K,
                                    device half* y16,
                                    uint tg,
                                    uint ti,
                                    uint tptg) {
    const uint simdPerTG = max(1u, tptg >> 5u);
    const uint simdId = ti >> 5u;
    const uint lane = ti & 31u;
    const uint row0 = (tg * simdPerTG + simdId) * ROWS;
    if (row0 >= realN) return;

    const float4 acc = qwg_gemv_aq_rowsN<BITS, GROUP, ROWS>(blocks, x, row0, realN, K, lane);
    if (lane == 0u) {
        #pragma unroll
        for (uint r = 0; r < ROWS; ++r) {
            const uint row = row0 + r;
            if (row < realN) {
                y16[row] = half(acc[r]);
            }
        }
    }
}

#define QW_AQ_GEMV_T4_KERNEL(BITS, GROUP, NAME)                                   \
    kernel void NAME(device const char* blocks [[buffer(0)]],                      \
                     device const half* x [[buffer(1)]],                           \
                     device float* y32 [[buffer(3)]],                             \
                     constant uint& N [[buffer(4)]],                               \
                     constant uint& K [[buffer(5)]],                               \
                     device half* y16 [[buffer(8)]],                               \
                     uint tg [[threadgroup_position_in_grid]],                     \
                     uint ti [[thread_index_in_threadgroup]],                     \
                     uint tptg [[threads_per_threadgroup]]) {                     \
        if ((N & 0x40000000u) != 0u) {                                            \
            const uint mRows = (N >> 24u) & 0x3Fu;                                \
            if (mRows >= 2u) {                                                    \
                const uint realN = N & 0x00FFFFFFu;                               \
                const uint simdPerTG = max(1u, tptg >> 5u);                       \
                const uint row0 = (tg * simdPerTG + (ti >> 5u)) * 4u;             \
                if (row0 < realN) {                                               \
                    qwg_gemv_aq_rows4_m<BITS, GROUP, half>(                       \
                        blocks, x, y16, row0, realN, K, mRows, ti & 31u);         \
                }                                                                 \
            } else {                                                              \
                qwg_gemv_aq_body<BITS, GROUP, 4u>(blocks, x, N & 0x3FFFFFFFu,     \
                                                 K, y16, tg, ti, tptg);           \
            }                                                                     \
        } else if ((N & 0x80000000u) != 0u) {                                     \
            qwg_gemv_aq_body<BITS, GROUP, 2u>(blocks, x, N & 0x7FFFFFFFu,         \
                                             K, y16, tg, ti, tptg);              \
        } else {                                                                 \
            qwg_gemv_aq_body<BITS, GROUP, 1u>(blocks, x, N, K, y16, tg, ti, tptg); \
        }                                                                        \
    }

QW_AQ_GEMV_T4_KERNEL(2, 32, qwg_gemv_aq2_g32_dx_h16_t4)
QW_AQ_GEMV_T4_KERNEL(2, 64, qwg_gemv_aq2_g64_dx_h16_t4)
QW_AQ_GEMV_T4_KERNEL(2, 128, qwg_gemv_aq2_g128_dx_h16_t4)
QW_AQ_GEMV_T4_KERNEL(2, 256, qwg_gemv_aq2_g256_dx_h16_t4)
QW_AQ_GEMV_T4_KERNEL(3, 32, qwg_gemv_aq3_g32_dx_h16_t4)
QW_AQ_GEMV_T4_KERNEL(3, 64, qwg_gemv_aq3_g64_dx_h16_t4)
QW_AQ_GEMV_T4_KERNEL(3, 128, qwg_gemv_aq3_g128_dx_h16_t4)
QW_AQ_GEMV_T4_KERNEL(3, 256, qwg_gemv_aq3_g256_dx_h16_t4)
QW_AQ_GEMV_T4_KERNEL(4, 32, qwg_gemv_aq4_g32_dx_h16_t4)
QW_AQ_GEMV_T4_KERNEL(4, 64, qwg_gemv_aq4_g64_dx_h16_t4)
QW_AQ_GEMV_T4_KERNEL(4, 128, qwg_gemv_aq4_g128_dx_h16_t4)
QW_AQ_GEMV_T4_KERNEL(4, 256, qwg_gemv_aq4_g256_dx_h16_t4)
QW_AQ_GEMV_T4_KERNEL(6, 32, qwg_gemv_aq6_g32_dx_h16_t4)
QW_AQ_GEMV_T4_KERNEL(6, 64, qwg_gemv_aq6_g64_dx_h16_t4)
QW_AQ_GEMV_T4_KERNEL(6, 128, qwg_gemv_aq6_g128_dx_h16_t4)
QW_AQ_GEMV_T4_KERNEL(6, 256, qwg_gemv_aq6_g256_dx_h16_t4)
QW_AQ_GEMV_T4_KERNEL(8, 32, qwg_gemv_aq8_g32_dx_h16_t4)
QW_AQ_GEMV_T4_KERNEL(8, 64, qwg_gemv_aq8_g64_dx_h16_t4)
QW_AQ_GEMV_T4_KERNEL(8, 128, qwg_gemv_aq8_g128_dx_h16_t4)
QW_AQ_GEMV_T4_KERNEL(8, 256, qwg_gemv_aq8_g256_dx_h16_t4)
