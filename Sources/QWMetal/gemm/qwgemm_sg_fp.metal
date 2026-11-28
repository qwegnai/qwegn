// qwgemm_sg_fp.metal — Apple9 simdgroup_half8x8 kernels for BF16 GEMM, NVFP4_G16 & MXFP8_G32 MoE.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_moe_fp_sg.metalh"

kernel void qwg_gemm_bf16_sg_bm32(device const bfloat* W [[buffer(0)]],
                                  device const bfloat* xBF16 [[buffer(1)]],
                                  device const float* xF32 [[buffer(2)]],
                                  device bfloat* yBF16 [[buffer(3)]],
                                  device float* yF32 [[buffer(4)]],
                                  constant QWGemmFPBF16Params& p [[buffer(5)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint lid [[thread_index_in_threadgroup]],
                                  uint simd_gid [[simdgroup_index_in_threadgroup]]) {
    threadgroup QWGFPSgTileMem tgMem;
    qwg_gemm_bf16_sg_body<32>(W, xBF16, xF32, yBF16, yF32, p, tg, lid, simd_gid, tgMem);
}

kernel void qwg_gemm_bf16_sg_bm64(device const bfloat* W [[buffer(0)]],
                                  device const bfloat* xBF16 [[buffer(1)]],
                                  device const float* xF32 [[buffer(2)]],
                                  device bfloat* yBF16 [[buffer(3)]],
                                  device float* yF32 [[buffer(4)]],
                                  constant QWGemmFPBF16Params& p [[buffer(5)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint lid [[thread_index_in_threadgroup]],
                                  uint simd_gid [[simdgroup_index_in_threadgroup]]) {
    threadgroup QWGFPSgTileMem tgMem;
    qwg_gemm_bf16_sg_body<64>(W, xBF16, xF32, yBF16, yF32, p, tg, lid, simd_gid, tgMem);
}

kernel void qwg_moe_sg_gemm_nvfp4_g16(
    device const uint32_t* w [[buffer(0)]], device const uchar* sc [[buffer(1)]],
    device const float* s2 [[buffer(2)]], device const char* x [[buffer(3)]],
    device char* y [[buffer(4)]], device const int32_t* expertOff [[buffer(5)]],
    device const int32_t* tileE [[buffer(6)]], device const int32_t* tileR [[buffer(7)]],
    device const int32_t* tileCount [[buffer(8)]], constant QWGemmFPSgMoEParams& p [[buffer(9)]],
    constant uint& tileBlock [[buffer(10)]], device const int32_t* sortedTok [[buffer(11)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
    threadgroup QWGFPSgTgLut tgLut;
    threadgroup QWGFPSgTileMem tgMem;
    qwg_moe_sg_fp_dispatch<false>(w, sc, s2, x, y, expertOff, tileE, tileR, tileCount, p,
        tileBlock, sortedTok, tgLut, tgMem, tg, lid, simd_gid);
}

kernel void qwg_moe_sg_gemm_mxfp8_g32(
    device const uint32_t* w [[buffer(0)]], device const uchar* sc [[buffer(1)]],
    device const char* x [[buffer(2)]], device char* y [[buffer(3)]],
    device const int32_t* expertOff [[buffer(4)]], device const int32_t* tileE [[buffer(5)]],
    device const int32_t* tileR [[buffer(6)]], device const int32_t* tileCount [[buffer(7)]],
    constant QWGemmFPSgMoEParams& p [[buffer(8)]], constant uint& tileBlock [[buffer(9)]],
    device const int32_t* sortedTok [[buffer(10)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
    threadgroup QWGFPSgTgLut tgLut;
    threadgroup QWGFPSgTileMem tgMem;
    qwg_moe_sg_fp_dispatch<true>(w, sc, nullptr, x, y, expertOff, tileE, tileR, tileCount, p,
        tileBlock, sortedTok, tgLut, tgMem, tg, lid, simd_gid);
}

kernel void qwg_moe_sg_gemm_nvfp4_g16_swiglu(
    device const uint32_t* w [[buffer(0)]], device const uchar* sc [[buffer(1)]],
    device const float* s2 [[buffer(2)]], device const char* x [[buffer(3)]],
    device char* y [[buffer(4)]], device const int32_t* expertOff [[buffer(5)]],
    device const int32_t* tileE [[buffer(6)]], device const int32_t* tileR [[buffer(7)]],
    device const int32_t* tileCount [[buffer(8)]], constant QWGemmFPSgMoESwigluParams& p [[buffer(9)]],
    constant uint& tileBlock [[buffer(10)]], device const int32_t* sortedTok [[buffer(11)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
    threadgroup QWGFPSgTgLut tgLut;
    threadgroup QWGFPSgSwigluMem tgMem;
    qwg_moe_sg_fp_swiglu_dispatch<false>(w, sc, s2, x, y, expertOff, tileE, tileR, tileCount, p,
        tileBlock, sortedTok, tgLut, tgMem, tg, lid, simd_gid);
}

kernel void qwg_moe_sg_gemm_mxfp8_g32_swiglu(
    device const uint32_t* w [[buffer(0)]], device const uchar* sc [[buffer(1)]],
    device const char* x [[buffer(2)]], device char* y [[buffer(3)]],
    device const int32_t* expertOff [[buffer(4)]], device const int32_t* tileE [[buffer(5)]],
    device const int32_t* tileR [[buffer(6)]], device const int32_t* tileCount [[buffer(7)]],
    constant QWGemmFPSgMoESwigluParams& p [[buffer(8)]], constant uint& tileBlock [[buffer(9)]],
    device const int32_t* sortedTok [[buffer(10)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
    threadgroup QWGFPSgTgLut tgLut;
    threadgroup QWGFPSgSwigluMem tgMem;
    qwg_moe_sg_fp_swiglu_dispatch<true>(w, sc, nullptr, x, y, expertOff, tileE, tileR, tileCount, p,
        tileBlock, sortedTok, tgLut, tgMem, tg, lid, simd_gid);
}
