// qwgemm_sg_shvq.metal — Apple9 simdgroup_half8x8 kernels for SHVQ{1,15,2,3,4}_G{32..256}.
// Isolates QW_SHVQ_ROUTED_DIRECT_LUT to this TU to honor Law 27 constant-LUT limits.
#define QW_SHVQ_ROUTED_DIRECT_LUT 1
#define QW_SG_ENABLE_SHVQ 1
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_gemm_sg.metalh"

#define QW_SG_SHVQ_IMPL(BITS, GROUP, HAS_TG, DECL_TG, PREP_TG, LUT_PTR)                        \
    kernel void qwg_gemm_shvq##BITS##_g##GROUP##_sg(                                           \
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
        DECL_TG                                                                                \
        PREP_TG                                                                                \
        qwg_gemm_sg_h16_dispatch<BITS, GROUP, true, HAS_TG>(blocks, x, y16, y32, M, N, K,      \
            int(xStride), int(yStride), tg, lid, simd_gid, tgMem, LUT_PTR);                    \
    }                                                                                          \
    kernel void qwg_gemm_shvq##BITS##_g##GROUP##_sg_gdn_inproj(                                \
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
        DECL_TG                                                                                \
        PREP_TG                                                                                \
        qmm_t_sg_dual_impl<64, 64, 2, 2, half, true, true, false, BITS, GROUP, true, HAS_TG>(  \
            blocks, x, qkv16, y32, int(K), int(N), int(M), int(K), int(N),                     \
            tg, lid, simd_gid, tgMem, z16, nullptr, LUT_PTR);                                  \
    }                                                                                          \
    kernel void qwg_gemm_shvq##BITS##_g##GROUP##_sg_f32(                                       \
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
        DECL_TG                                                                                \
        PREP_TG                                                                                \
        qwg_gemm_sg_f32_dispatch<BITS, GROUP, true, HAS_TG>(                                   \
            blocks, xh, y32, M, N, K, tg, lid, simd_gid, tgMem, LUT_PTR);                      \
    }                                                                                          \
    kernel void qwg_moe_sg_gemm_shvq##BITS##_g##GROUP(                                         \
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
        DECL_TG                                                                                \
        PREP_TG                                                                                \
        qwg_moe_sg_gemm_body<BITS, GROUP, true, HAS_TG>(w, x, y, expertOff, tileE, tileR,      \
            tileCount, p, tileBlock, sortedTok, hasGather, coldW,                              \
            tg, lid, simd_gid, tgMem, LUT_PTR);                                                \
    }                                                                                          \
    kernel void qwg_moe_sg_split_shvq##BITS##_g##GROUP##_f32(                                  \
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
        DECL_TG                                                                                \
        PREP_TG                                                                                \
        qwg_moe_sg_split_f32_body<BITS, GROUP, true, HAS_TG>(w, xh, y32, expertOff, tileE,     \
            tileR, tileCount, p, tileBlock, coldW, tg, lid, simd_gid, tgMem, LUT_PTR);         \
    }

#define QW_SG_SHVQ_TG(BITS, GROUP)                                                             \
    QW_SG_SHVQ_IMPL(BITS, GROUP, true,                                                         \
        threadgroup ShvqTgLut<BITS> tgLut;,                                                    \
        tgLut.load(lid, 128u); threadgroup_barrier(mem_flags::mem_threadgroup);,               \
        tgLut.ptr16())

#define QW_SG_SHVQ_DIRECT(BITS, GROUP) QW_SG_SHVQ_IMPL(BITS, GROUP, false, , , nullptr)

QW_SG_SHVQ_TG(1, 32)      QW_SG_SHVQ_TG(1, 64)      QW_SG_SHVQ_TG(1, 128)      QW_SG_SHVQ_TG(1, 256)
QW_SG_SHVQ_DIRECT(15, 32) QW_SG_SHVQ_DIRECT(15, 64) QW_SG_SHVQ_DIRECT(15, 128) QW_SG_SHVQ_DIRECT(15, 256)
QW_SG_SHVQ_TG(2, 32)      QW_SG_SHVQ_TG(2, 64)      QW_SG_SHVQ_TG(2, 128)      QW_SG_SHVQ_TG(2, 256)
QW_SG_SHVQ_DIRECT(3, 32)  QW_SG_SHVQ_DIRECT(3, 64)  QW_SG_SHVQ_DIRECT(3, 128)  QW_SG_SHVQ_DIRECT(3, 256)
QW_SG_SHVQ_DIRECT(4, 32)  QW_SG_SHVQ_DIRECT(4, 64)  QW_SG_SHVQ_DIRECT(4, 128)  QW_SG_SHVQ_DIRECT(4, 256)
