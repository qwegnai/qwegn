// qwgemm_multi.metal — the size-preserving affine-quantised GEMV family.
//
// One entry point per (bits, group) pair, plus its f32-activation twin.
// The names are explicit, not template instantiations: the kernel
// reachability gate reads the metallib's entry points, and a mangled
// template name is not a name Swift can ask for.
// buffers: 0 blocks | 1 x | 3 y f32 | 4 N u32 | 5 K u32

#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_dequant_multi.metalh"
using namespace metal;

// The geometry is the incumbent's `qwg_gemv_q4g_v4`: the threadgroup's
// simdgroups take one row each, and the lane splits the groups.

#define QWG_AQ_KERNEL(BITS, GROUP, NAME)                                       \
    kernel void NAME(device const char* blocks [[buffer(0)]],                   \
                     device const half* x [[buffer(1)]],                        \
                     device float* y [[buffer(3)]],                              \
                     constant uint& N [[buffer(4)]],                            \
                     constant uint& K [[buffer(5)]],                            \
                     uint tg [[threadgroup_position_in_grid]],                  \
                     uint ti [[thread_index_in_threadgroup]],                  \
                     uint tptg [[threads_per_threadgroup]]) {                 \
        const uint simd_id = ti >> 5u;                                         \
        const uint lane = ti & 31u;                                            \
        const uint simdPerTG = max(1u, tptg >> 5u);                            \
        if ((N & 0x80000000u) != 0u) {                                         \
            const uint mRows = (N >> 24u) & 0x3Fu;                             \
            const uint realN = N & 0x00FFFFFFu;                                \
            const uint row0 = (tg * simdPerTG + simd_id) * 4u;                 \
            if (row0 >= realN) return;                                         \
            qwg_gemv_aq_rows4_m<BITS, GROUP>(                                  \
                blocks, x, y, row0, realN, K, mRows, lane);                    \
        } else if ((N & 0x40000000u) != 0u) {                                  \
            const uint realN = N & 0x3FFFFFFFu;                                \
            const uint row0 = (tg * simdPerTG + simd_id) * 4u;                 \
            if (row0 >= realN) return;                                         \
            const float4 acc = qwg_gemv_aq_rowsN<BITS, GROUP, 4u>(             \
                blocks, x, row0, realN, K, lane);                              \
            if (lane == 0u) {                                                  \
                _Pragma("unroll")                                              \
                for (uint r = 0; r < 4u; ++r) {                                \
                    if (row0 + r < realN) { y[row0 + r] = acc[r]; }            \
                }                                                              \
            }                                                                  \
        } else {                                                               \
            const uint row = tg * simdPerTG + simd_id;                         \
            if (row >= N) return;                                              \
            const float v = qwg_aq_gemv_row_f16<BITS, GROUP>(                  \
                blocks, x, row, K, lane);                                      \
            if (lane == 0u) { y[row] = v; }                                    \
        }                                                                      \
    }

#define QWG_AQ_KERNEL_F32(BITS, GROUP, NAME)                                   \
    kernel void NAME(device const char* blocks [[buffer(0)]],                   \
                     device const float* x [[buffer(1)]],                       \
                     device float* y [[buffer(3)]],                              \
                     constant uint& N [[buffer(4)]],                            \
                     constant uint& K [[buffer(5)]],                            \
                     uint tg [[threadgroup_position_in_grid]],                  \
                     uint ti [[thread_index_in_threadgroup]],                  \
                     uint tptg [[threads_per_threadgroup]]) {                 \
        const uint simd_id = ti >> 5u;                                         \
        const uint lane = ti & 31u;                                            \
        const uint row = tg * max(1u, tptg >> 5u) + simd_id;                   \
        if (row >= N) return;                                                  \
        const float v = qwg_aq_gemv_row_f32<BITS, GROUP>(blocks, x, row, K, lane); \
        if (lane == 0u) { y[row] = v; }                                         \
    }

QWG_AQ_KERNEL(2, 32, qwg_gemv_aq2_g32_v4)
QWG_AQ_KERNEL(2, 64, qwg_gemv_aq2_g64_v4)
QWG_AQ_KERNEL(2, 128, qwg_gemv_aq2_g128_v4)
QWG_AQ_KERNEL(2, 256, qwg_gemv_aq2_g256_v4)
QWG_AQ_KERNEL(3, 32, qwg_gemv_aq3_g32_v4)
QWG_AQ_KERNEL(3, 64, qwg_gemv_aq3_g64_v4)
QWG_AQ_KERNEL(3, 128, qwg_gemv_aq3_g128_v4)
QWG_AQ_KERNEL(3, 256, qwg_gemv_aq3_g256_v4)
QWG_AQ_KERNEL(4, 32, qwg_gemv_aq4_g32_v4)
QWG_AQ_KERNEL(4, 64, qwg_gemv_aq4_g64_v4)
QWG_AQ_KERNEL(4, 128, qwg_gemv_aq4_g128_v4)
QWG_AQ_KERNEL(4, 256, qwg_gemv_aq4_g256_v4)
QWG_AQ_KERNEL(6, 32, qwg_gemv_aq6_g32_v4)
QWG_AQ_KERNEL(6, 64, qwg_gemv_aq6_g64_v4)
QWG_AQ_KERNEL(6, 128, qwg_gemv_aq6_g128_v4)
QWG_AQ_KERNEL(6, 256, qwg_gemv_aq6_g256_v4)
QWG_AQ_KERNEL(8, 32, qwg_gemv_aq8_g32_v4)
QWG_AQ_KERNEL(8, 64, qwg_gemv_aq8_g64_v4)
QWG_AQ_KERNEL(8, 128, qwg_gemv_aq8_g128_v4)
QWG_AQ_KERNEL(8, 256, qwg_gemv_aq8_g256_v4)

QWG_AQ_KERNEL_F32(2, 32, qwg_gemv_aq2_g32_v4_f32)
QWG_AQ_KERNEL_F32(2, 64, qwg_gemv_aq2_g64_v4_f32)
QWG_AQ_KERNEL_F32(2, 128, qwg_gemv_aq2_g128_v4_f32)
QWG_AQ_KERNEL_F32(2, 256, qwg_gemv_aq2_g256_v4_f32)
QWG_AQ_KERNEL_F32(3, 32, qwg_gemv_aq3_g32_v4_f32)
QWG_AQ_KERNEL_F32(3, 64, qwg_gemv_aq3_g64_v4_f32)
QWG_AQ_KERNEL_F32(3, 128, qwg_gemv_aq3_g128_v4_f32)
QWG_AQ_KERNEL_F32(3, 256, qwg_gemv_aq3_g256_v4_f32)
QWG_AQ_KERNEL_F32(4, 32, qwg_gemv_aq4_g32_v4_f32)
QWG_AQ_KERNEL_F32(4, 64, qwg_gemv_aq4_g64_v4_f32)
QWG_AQ_KERNEL_F32(4, 128, qwg_gemv_aq4_g128_v4_f32)
QWG_AQ_KERNEL_F32(4, 256, qwg_gemv_aq4_g256_v4_f32)
QWG_AQ_KERNEL_F32(6, 32, qwg_gemv_aq6_g32_v4_f32)
QWG_AQ_KERNEL_F32(6, 64, qwg_gemv_aq6_g64_v4_f32)
QWG_AQ_KERNEL_F32(6, 128, qwg_gemv_aq6_g128_v4_f32)
QWG_AQ_KERNEL_F32(6, 256, qwg_gemv_aq6_g256_v4_f32)
QWG_AQ_KERNEL_F32(8, 32, qwg_gemv_aq8_g32_v4_f32)
QWG_AQ_KERNEL_F32(8, 64, qwg_gemv_aq8_g64_v4_f32)
QWG_AQ_KERNEL_F32(8, 128, qwg_gemv_aq8_g128_v4_f32)
QWG_AQ_KERNEL_F32(8, 256, qwg_gemv_aq8_g256_v4_f32)
