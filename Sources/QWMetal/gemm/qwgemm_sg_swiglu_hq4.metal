// qwgemm_sg_swiglu_hq4.metal — Apple9 simdgroup_half8x8 gate_up + SwiGLU + FWHT-32 for HQ4_G32.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_gemm_sg_swiglu.metalh"

kernel void qwg_gemm_q4g_sg_swiglu(device const Block_HQ4_G32* blocks  [[buffer(0)]],
                                   device const half*          x       [[buffer(1)]],
                                   device half*                out16   [[buffer(2)]],
                                   constant uint&              M       [[buffer(3)]],
                                   constant uint&              I       [[buffer(4)]],
                                   constant uint&              K       [[buffer(5)]],
                                   constant uint&              xStride [[buffer(6)]],
                                   constant uint&              yStride [[buffer(7)]],
                                   uint3 tg       [[threadgroup_position_in_grid]],
                                   uint  lid      [[thread_index_in_threadgroup]],
                                   uint  simd_gid [[simdgroup_index_in_threadgroup]],
                                   uint  simd_lid [[thread_index_in_simdgroup]]) {
    threadgroup QWSGSwigluMem tgMem;
    const uint mode = (M >> 29u) & 3u;
    const int realM = int(M & 0x1FFFFFFFu);
    if (mode == 1u) {
        qmm_t_sg_swiglu_impl<16, 1, 4>((const device char*)blocks, x, out16,
            int(K), int(I), realM, int(xStride), int(yStride), tg, lid, simd_gid, simd_lid, tgMem);
    } else if (mode == 2u) {
        qmm_t_sg_swiglu_impl<32, 2, 2>((const device char*)blocks, x, out16,
            int(K), int(I), realM, int(xStride), int(yStride), tg, lid, simd_gid, simd_lid, tgMem);
    } else {
        qmm_t_sg_swiglu_impl<64, 2, 2>((const device char*)blocks, x, out16,
            int(K), int(I), realM, int(xStride), int(yStride), tg, lid, simd_gid, simd_lid, tgMem);
    }
}

kernel void qwg_moe_sg_swiglu_q4g(device const Block_HQ4_G32*   w         [[buffer(0)]],
                                  device const half*            x         [[buffer(1)]],
                                  device half*                  y         [[buffer(2)]],
                                  device const int32_t*         expertOff [[buffer(3)]],
                                  device const int32_t*         tileE     [[buffer(4)]],
                                  device const int32_t*         tileR     [[buffer(5)]],
                                  device const int32_t*         tileCount [[buffer(6)]],
                                  constant QWGemmGroupedParams& p         [[buffer(7)]],
                                  constant uint&                tileBlock [[buffer(8)]],
                                  device const int32_t*         sortedTok [[buffer(9)]],
                                  constant uint&                hasGather [[buffer(10)]],
                                  device const Block_HQ4_G32*   coldW     [[buffer(11)]],
                                  uint3 tg       [[threadgroup_position_in_grid]],
                                  uint  lid      [[thread_index_in_threadgroup]],
                                  uint  simd_gid [[simdgroup_index_in_threadgroup]],
                                  uint  simd_lid [[thread_index_in_simdgroup]]) {
    threadgroup QWSGSwigluMem tgMem;
    qwg_moe_sg_swiglu_body<0, 0, false, false>((device const char*)w, x, y, expertOff, tileE,
        tileR, tileCount, p, tileBlock, sortedTok, hasGather, (device const char*)coldW,
        tg, lid, simd_gid, simd_lid, tgMem);
}
