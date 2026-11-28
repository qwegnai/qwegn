// qwgemm_sg_aq.metal — Apple9 simdgroup_half8x8 dense, GDN, F32, and MoE kernels for AQ{2..8}_G{32..256}.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_gemm_sg.metalh"

#define QW_SG_AQ_SUITE(BITS, GROUP)                                                            \
    kernel void qwg_gemm_aq##BITS##_g##GROUP##_sg(                                             \
        device const char* blocks [[buffer(0)]],                                               \
        device const half* x      [[buffer(1)]],                                               \
        device half*       y16    [[buffer(2)]],                                               \
        device float*      y32    [[buffer(3)]],                                               \
        constant uint&     M      [[buffer(4)]],                                               \
        constant uint&     N      [[buffer(5)]],                                               \
        constant uint&     K      [[buffer(6)]],                                               \
        constant uint&     xStride [[buffer(9)]],                                              \
        constant uint&     yStride [[buffer(10)]],                                             \
        uint3 tg [[threadgroup_position_in_grid]],                                             \
        uint lid [[thread_index_in_threadgroup]],                                              \
        uint simd_gid [[simdgroup_index_in_threadgroup]]) {                                    \
        threadgroup QWSGTileMem tgMem;                                                         \
        qwg_gemm_sg_h16_dispatch<BITS, GROUP>(blocks, x, y16, y32, M, N, K,                    \
            int(xStride), int(yStride), tg, lid, simd_gid, tgMem);                             \
    }                                                                                          \
    kernel void qwg_gemm_aq##BITS##_g##GROUP##_sg_gdn_inproj(                                  \
        device const char* blocks [[buffer(0)]],                                               \
        device const half* x      [[buffer(1)]],                                               \
        device half*       qkv16  [[buffer(2)]],                                               \
        device float*      y32    [[buffer(3)]],                                               \
        constant uint&     M      [[buffer(4)]],                                               \
        constant uint&     N      [[buffer(5)]],                                               \
        constant uint&     K      [[buffer(6)]],                                               \
        device half*       z16    [[buffer(7)]],                                               \
        uint3 tg [[threadgroup_position_in_grid]],                                             \
        uint lid [[thread_index_in_threadgroup]],                                              \
        uint simd_gid [[simdgroup_index_in_threadgroup]]) {                                    \
        threadgroup QWSGTileMem tgMem;                                                         \
        qmm_t_sg_dual_impl<64, 64, 2, 2, half, true, true, false, BITS, GROUP>(                \
            blocks, x, qkv16, y32, int(K), int(N), int(M), int(K), int(N),                     \
            tg, lid, simd_gid, tgMem, z16);                                                    \
    }                                                                                          \
    kernel void qwg_gemm_aq##BITS##_g##GROUP##_sg_f32(                                         \
        device const char* blocks [[buffer(0)]],                                               \
        device const half* xh     [[buffer(1)]],                                               \
        device const half* xl     [[buffer(2)]],                                               \
        device float*      y32    [[buffer(3)]],                                               \
        constant uint&     M      [[buffer(4)]],                                               \
        constant uint&     N      [[buffer(5)]],                                               \
        constant uint&     K      [[buffer(6)]],                                               \
        uint3 tg [[threadgroup_position_in_grid]],                                             \
        uint lid [[thread_index_in_threadgroup]],                                              \
        uint simd_gid [[simdgroup_index_in_threadgroup]]) {                                    \
        (void)xl;                                                                              \
        threadgroup QWSGTileMem tgMem;                                                         \
        qwg_gemm_sg_f32_dispatch<BITS, GROUP>(blocks, xh, y32, M, N, K, tg, lid, simd_gid, tgMem); \
    }                                                                                          \
    kernel void qwg_moe_sg_gemm_aq##BITS##_g##GROUP(                                           \
        device const char*    w         [[buffer(0)]],                                         \
        device const half*    x         [[buffer(1)]],                                         \
        device half*          y         [[buffer(2)]],                                         \
        device const int32_t* expertOff [[buffer(3)]],                                         \
        device const int32_t* tileE     [[buffer(4)]],                                         \
        device const int32_t* tileR     [[buffer(5)]],                                         \
        device const int32_t* tileCount [[buffer(6)]],                                         \
        constant QWGemmGroupedParams& p [[buffer(7)]],                                         \
        constant uint&        tileBlock [[buffer(8)]],                                         \
        device const int32_t* sortedTok [[buffer(9)]],                                         \
        constant uint&        hasGather [[buffer(10)]],                                        \
        device const char*    coldW     [[buffer(11)]],                                        \
        uint3 tg [[threadgroup_position_in_grid]],                                             \
        uint lid [[thread_index_in_threadgroup]],                                              \
        uint simd_gid [[simdgroup_index_in_threadgroup]]) {                                    \
        threadgroup QWSGMoETileMem tgMem;                                                      \
        qwg_moe_sg_gemm_body<BITS, GROUP>(w, x, y, expertOff, tileE, tileR, tileCount,         \
            p, tileBlock, sortedTok, hasGather, coldW, tg, lid, simd_gid, tgMem);              \
    }                                                                                          \
    kernel void qwg_moe_sg_split_aq##BITS##_g##GROUP##_f32(                                    \
        device const char*    w         [[buffer(0)]],                                         \
        device const half*    xh        [[buffer(1)]],                                         \
        device const half*    xl        [[buffer(2)]],                                         \
        device float*         y32       [[buffer(3)]],                                         \
        device const int32_t* expertOff [[buffer(4)]],                                         \
        device const int32_t* tileE     [[buffer(5)]],                                         \
        device const int32_t* tileR     [[buffer(6)]],                                         \
        device const int32_t* tileCount [[buffer(7)]],                                         \
        constant QWGemmGroupedParams& p [[buffer(8)]],                                         \
        constant uint&        tileBlock [[buffer(9)]],                                         \
        device const char*    coldW     [[buffer(10)]],                                        \
        uint3 tg [[threadgroup_position_in_grid]],                                             \
        uint lid [[thread_index_in_threadgroup]],                                              \
        uint simd_gid [[simdgroup_index_in_threadgroup]]) {                                    \
        (void)xl;                                                                              \
        threadgroup QWSGMoETileMem tgMem;                                                      \
        qwg_moe_sg_split_f32_body<BITS, GROUP>(w, xh, y32, expertOff, tileE, tileR, tileCount, \
            p, tileBlock, coldW, tg, lid, simd_gid, tgMem);                                    \
    }

QW_SG_AQ_SUITE(2, 32)  QW_SG_AQ_SUITE(2, 64)  QW_SG_AQ_SUITE(2, 128)  QW_SG_AQ_SUITE(2, 256)
QW_SG_AQ_SUITE(3, 32)  QW_SG_AQ_SUITE(3, 64)  QW_SG_AQ_SUITE(3, 128)  QW_SG_AQ_SUITE(3, 256)
QW_SG_AQ_SUITE(4, 32)  QW_SG_AQ_SUITE(4, 64)  QW_SG_AQ_SUITE(4, 128)  QW_SG_AQ_SUITE(4, 256)
QW_SG_AQ_SUITE(6, 32)  QW_SG_AQ_SUITE(6, 64)  QW_SG_AQ_SUITE(6, 128)  QW_SG_AQ_SUITE(6, 256)
QW_SG_AQ_SUITE(8, 32)  QW_SG_AQ_SUITE(8, 64)  QW_SG_AQ_SUITE(8, 128)  QW_SG_AQ_SUITE(8, 256)
