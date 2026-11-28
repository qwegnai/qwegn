// qwgemm_aq_batched.metal — NAX prefill GEMM, GDN in-proj, and MoE twins for AQ.
// Instantiates compile-time <BITS, GROUP> kernels over shared qw_gemm_nax.metalh templates.
#if !defined(QW_DISABLE_NAX)
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_moe_tile.metalh"
#include "include/qw_gemm_nax.metalh"

#define QW_AQ_GEMM_KERNEL(BITS, GROUP, NAME)                                         \
    kernel void NAME(device const char* blocks [[buffer(0)]],                          \
                     device const half* x [[buffer(1)]],                               \
                     device half* y16 [[buffer(2)]],                                   \
                     device float* y32 [[buffer(3)]],                                  \
                     constant uint& M [[buffer(4)]], constant uint& N [[buffer(5)]],     \
                     constant uint& K [[buffer(6)]],                                   \
                     uint3 tg [[threadgroup_position_in_grid]],                        \
                     uint lid [[thread_index_in_threadgroup]],                        \
                     uint simd_gid [[simdgroup_index_in_threadgroup]],               \
                     uint simd_lid [[thread_index_in_simdgroup]]) {                  \
        qwg_gemm_h16_mode_dispatch<BITS, GROUP>(                                      \
            blocks, x, y16, y32, M, N, K, tg, lid, simd_gid, simd_lid);               \
    }

#define QW_AQ_GEMM_GDN_KERNEL(BITS, GROUP, NAME)                                     \
    kernel void NAME(device const char* blocks [[buffer(0)]],                          \
                     device const half* x [[buffer(1)]],                               \
                     device half* qkv16 [[buffer(2)]],                                 \
                     device float* y32 [[buffer(3)]],                                  \
                     constant uint& M [[buffer(4)]], constant uint& N [[buffer(5)]],     \
                     constant uint& K [[buffer(6)]],                                   \
                     device half* z16 [[buffer(7)]],                                   \
                     uint3 tg [[threadgroup_position_in_grid]],                        \
                     uint lid [[thread_index_in_threadgroup]],                        \
                     uint simd_gid [[simdgroup_index_in_threadgroup]],               \
                     uint simd_lid [[thread_index_in_simdgroup]]) {                  \
        qmm_t_nax_tgp_dual_impl<half, half, false, 64, 64, 64, 2, 2, true, true,       \
                                false, BITS, GROUP>(                                  \
            blocks, x, qkv16, y32, nullptr, int(K), int(N), int(M), tg, lid,          \
            simd_gid, simd_lid, z16);                                                 \
    }

#define QW_AQ_NAX_F32_KERNELS(BITS, GROUP)                                           \
    kernel void qwg_gemm_aq##BITS##_g##GROUP##_nax_f32(                               \
        device const char* w   [[buffer(0)]],                                          \
        device const half* xh  [[buffer(1)]],                                          \
        device const half* xl  [[buffer(2)]],                                          \
        device float*      y32 [[buffer(3)]],                                          \
        constant uint&     M   [[buffer(4)]],                                          \
        constant uint&     N   [[buffer(5)]],                                          \
        constant uint&     K   [[buffer(6)]],                                          \
        uint3 tg [[threadgroup_position_in_grid]],                                     \
        uint simd_gid [[simdgroup_index_in_threadgroup]]) {                            \
        qwg_gemm_nax_f32_dispatch<BITS, GROUP>(w, xh, xl, y32, M, N, K, tg, simd_gid); \
    }                                                                                 \
    kernel void qwg_moe_nax_gemm_aq##BITS##_g##GROUP(                                 \
        device const char*    w         [[buffer(0)]],                                 \
        device const half*    x         [[buffer(1)]],                                 \
        device half*          y         [[buffer(2)]],                                 \
        device const int32_t* expertOff [[buffer(3)]],                                 \
        device const int32_t* tileE     [[buffer(4)]],                                 \
        device const int32_t* tileR     [[buffer(5)]],                                 \
        device const int32_t* tileCount [[buffer(6)]],                                 \
        constant QWGemmGroupedParams& p [[buffer(7)]],                                 \
        constant uint&        tileBlock [[buffer(8)]],                                 \
        device const int32_t* sortedTok [[buffer(9)]],                                 \
        constant uint&        hasGather [[buffer(10)]],                                \
        device const char*    coldW     [[buffer(11)]],                                \
        uint3 tg [[threadgroup_position_in_grid]],                                     \
        uint lid [[thread_index_in_threadgroup]],                                     \
        uint simd_gid [[simdgroup_index_in_threadgroup]],                             \
        uint simd_lid [[thread_index_in_simdgroup]]) {                                \
        qwg_moe_nax_gemm_body<BITS, GROUP>(w, x, y, expertOff, tileE, tileR, tileCount, \
            p, tileBlock, sortedTok, hasGather, coldW, tg, lid, simd_gid, simd_lid);  \
    }                                                                                 \
    kernel void qwg_moe_nax_split_aq##BITS##_g##GROUP##_f32(                          \
        device const char*    w         [[buffer(0)]],                                 \
        device const half*    xh        [[buffer(1)]],                                 \
        device const half*    xl        [[buffer(2)]],                                 \
        device float*         y32       [[buffer(3)]],                                 \
        device const int32_t* expertOff [[buffer(4)]],                                 \
        device const int32_t* tileE     [[buffer(5)]],                                 \
        device const int32_t* tileR     [[buffer(6)]],                                 \
        device const int32_t* tileCount [[buffer(7)]],                                 \
        constant QWGemmGroupedParams& p [[buffer(8)]],                                 \
        constant uint&        tileBlock [[buffer(9)]],                                 \
        device const char*    coldW     [[buffer(10)]],                                \
        uint3 tg [[threadgroup_position_in_grid]],                                     \
        uint simd_gid [[simdgroup_index_in_threadgroup]]) {                            \
        qwg_moe_nax_split_f32_body<BITS, GROUP>(                                       \
            w, xh, xl, y32, expertOff, tileE, tileR, tileCount,                        \
            p, tileBlock, coldW, tg, simd_gid);                                        \
    }

#define QW_AQ_BATCHED_SUITE(BITS, GROUP)                                             \
    QW_AQ_GEMM_KERNEL(BITS, GROUP, qwg_gemm_aq##BITS##_g##GROUP##_h16)               \
    QW_AQ_GEMM_GDN_KERNEL(BITS, GROUP, qwg_gemm_aq##BITS##_g##GROUP##_h16_gdn_inproj) \
    QW_AQ_NAX_F32_KERNELS(BITS, GROUP)

QW_AQ_BATCHED_SUITE(2, 32)
QW_AQ_BATCHED_SUITE(2, 64)
QW_AQ_BATCHED_SUITE(2, 128)
QW_AQ_BATCHED_SUITE(2, 256)
QW_AQ_BATCHED_SUITE(3, 32)
QW_AQ_BATCHED_SUITE(3, 64)
QW_AQ_BATCHED_SUITE(3, 128)
QW_AQ_BATCHED_SUITE(3, 256)
QW_AQ_BATCHED_SUITE(4, 32)
QW_AQ_BATCHED_SUITE(4, 64)
QW_AQ_BATCHED_SUITE(4, 128)
QW_AQ_BATCHED_SUITE(4, 256)
QW_AQ_BATCHED_SUITE(6, 32)
QW_AQ_BATCHED_SUITE(6, 64)
QW_AQ_BATCHED_SUITE(6, 128)
QW_AQ_BATCHED_SUITE(6, 256)
QW_AQ_BATCHED_SUITE(8, 32)
QW_AQ_BATCHED_SUITE(8, 64)
QW_AQ_BATCHED_SUITE(8, 128)
QW_AQ_BATCHED_SUITE(8, 256)
#endif // !defined(QW_DISABLE_NAX)
