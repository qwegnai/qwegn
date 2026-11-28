// qwgemm_sg_hq4.metal — Apple9 simdgroup_half8x8 dense, GDN, F32, and grouped MoE kernels for Block_HQ4_G32.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_gemm_sg.metalh"

kernel void qwg_gemm_q4g_sg(device const Block_HQ4_G32* blocks [[buffer(0)]],
                            device const half*          x      [[buffer(1)]],
                            device half*                y16    [[buffer(2)]],
                            device float*               y32    [[buffer(3)]],
                            constant uint&              M      [[buffer(4)]],
                            constant uint&              N      [[buffer(5)]],
                            constant uint&              K      [[buffer(6)]],
                            constant uint&              xStride [[buffer(9)]],
                            constant uint&              yStride [[buffer(10)]],
                            uint3 tg       [[threadgroup_position_in_grid]],
                            uint  lid      [[thread_index_in_threadgroup]],
                            uint  simd_gid [[simdgroup_index_in_threadgroup]]) {
    threadgroup QWSGTileMem tgMem;
    qwg_gemm_sg_h16_dispatch((const device char*)blocks, x, y16, y32,
                             M, N, K, int(xStride), int(yStride), tg, lid, simd_gid, tgMem);
}

kernel void qwg_gemm_q4g_sg_gdn_inproj(device const Block_HQ4_G32* blocks [[buffer(0)]],
                                       device const half*          x      [[buffer(1)]],
                                       device half*                qkv16  [[buffer(2)]],
                                       device float*               y32    [[buffer(3)]],
                                       constant uint&              M      [[buffer(4)]],
                                       constant uint&              N      [[buffer(5)]],
                                       constant uint&              K      [[buffer(6)]],
                                       device half*                z16    [[buffer(7)]],
                                       uint3 tg       [[threadgroup_position_in_grid]],
                                       uint  lid      [[thread_index_in_threadgroup]],
                                       uint  simd_gid [[simdgroup_index_in_threadgroup]]) {
    threadgroup QWSGTileMem tgMem;
    qmm_t_sg_dual_impl<64, 64, 2, 2, half, true, true, false>(
        (const device char*)blocks, x, qkv16, y32,
        int(K), int(N), int(M), int(K), int(N), tg, lid, simd_gid, tgMem, z16);
}

kernel void qwg_gemm_q4g_sg_f32(device const Block_HQ4_G32* blocks [[buffer(0)]],
                                device const half*          xh     [[buffer(1)]],
                                device const half*          xl     [[buffer(2)]],
                                device float*               y32    [[buffer(3)]],
                                constant uint&              M      [[buffer(4)]],
                                constant uint&              N      [[buffer(5)]],
                                constant uint&              K      [[buffer(6)]],
                                uint3 tg       [[threadgroup_position_in_grid]],
                                uint  lid      [[thread_index_in_threadgroup]],
                                uint  simd_gid [[simdgroup_index_in_threadgroup]]) {
    (void)xl;
    threadgroup QWSGTileMem tgMem;
    qwg_gemm_sg_f32_dispatch((const device char*)blocks, xh, y32, M, N, K, tg, lid, simd_gid, tgMem);
}

kernel void qwg_moe_sg_gemm(device const Block_HQ4_G32* w [[buffer(0)]],
                            device const half* x [[buffer(1)]],
                            device half* y [[buffer(2)]],
                            device const int32_t* expertOff [[buffer(3)]],
                            device const int32_t* tileE [[buffer(4)]],
                            device const int32_t* tileR [[buffer(5)]],
                            device const int32_t* tileCount [[buffer(6)]],
                            constant QWGemmGroupedParams& p [[buffer(7)]],
                            constant uint& tileBlock [[buffer(8)]],
                            device const int32_t* sortedTok [[buffer(9)]],
                            constant uint& hasGather [[buffer(10)]],
                            device const Block_HQ4_G32* coldW [[buffer(11)]],
                            uint3 tg [[threadgroup_position_in_grid]],
                            uint lid [[thread_index_in_threadgroup]],
                            uint simd_gid [[simdgroup_index_in_threadgroup]]) {
    threadgroup QWSGMoETileMem tgMem;
    qwg_moe_sg_gemm_body((device const char*)w, x, y, expertOff, tileE, tileR, tileCount, p,
                         tileBlock, sortedTok, hasGather, (device const char*)coldW,
                         tg, lid, simd_gid, tgMem);
}

kernel void qwg_moe_sg_split_f32(device const Block_HQ4_G32* w [[buffer(0)]],
                                 device const half* xh [[buffer(1)]],
                                 device const half* xl [[buffer(2)]],
                                 device float* y32 [[buffer(3)]],
                                 device const int32_t* expertOff [[buffer(4)]],
                                 device const int32_t* tileE [[buffer(5)]],
                                 device const int32_t* tileR [[buffer(6)]],
                                 device const int32_t* tileCount [[buffer(7)]],
                                 constant QWGemmGroupedParams& p [[buffer(8)]],
                                 constant uint& tileBlock [[buffer(9)]],
                                 device const Block_HQ4_G32* coldW [[buffer(10)]],
                                 uint3 tg [[threadgroup_position_in_grid]],
                                 uint lid [[thread_index_in_threadgroup]],
                                 uint simd_gid [[simdgroup_index_in_threadgroup]]) {
    (void)xl;
    threadgroup QWSGMoETileMem tgMem;
    qwg_moe_sg_split_f32_body((device const char*)w, xh, y32, expertOff, tileE, tileR, tileCount,
                              p, tileBlock, (device const char*)coldW, tg, lid, simd_gid, tgMem);
}
