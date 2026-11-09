// qwmhc_rows.metal — prefill mHC router-norm: one threadgroup per QW_MHC_ROWS rows, so each
// f32 router weight load is shared by those rows (the fused kernel re-reads the whole router
// per row). Same math as qwg_mhc_router_norm_f16 (merge, router dots, controls, branch norm).
#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_geometry_constants.metalh"
#include "qw_mhc_common.metalh"
using namespace metal;

#define QW_MHC_ROWS 4u

template <uint C, uint R>
inline void qwg_mhc_router_norm_rows_impl(device float*       mhcStream,
                                          device const float* routerW,
                                          device const float* routerB,
                                          device const half*  normGamma,
                                          device half*        outNorm,
                                          device float*       mhcControls,
                                          uint                hidden,
                                          float               eps,
                                          float               preScale,
                                          uint                iterations,
                                          device const half*  mergeBranchOut,
                                          uint                rows,
                                          uint                tg,
                                          uint                sg,
                                          uint                lane,
                                          threadgroup float*  partSS,
                                          threadgroup float*  partDots,
                                          threadgroup float*  sPre) {
    constexpr uint NUM_CTRL = C + C * C;
    constexpr uint NUM_ROWS = 2u * C + C * C;
    const uint tid = sg * 32u + lane;
    const uint t0 = tg * R;
    const uint nRows = min(R, rows - t0);
    const uint width = C * hidden;
    const uint width4 = width >> 2u;
    const bool hasMerge = (iterations & 0x80000000u) != 0u;
    const uint realIterations = iterations & 0x7FFFFFFFu;

    if (hasMerge) {
        for (uint r = 0u; r < nRows; ++r) {
            const uint t = t0 + r;
            device float* sBase = mhcStream + (size_t)t * width;
            device const half* bo = mergeBranchOut + (size_t)t * hidden;
            device const float* prevCtrl = mhcControls + (size_t)t * NUM_CTRL;
            float post[C];
            float mix[C * C];
            #pragma unroll
            for (uint c = 0u; c < C; ++c) post[c] = prevCtrl[c];
            #pragma unroll
            for (uint m = 0u; m < C * C; ++m) mix[m] = prevCtrl[C + m];
            for (uint h = tid; h < hidden; h += 256u) {
                float v[C];
                #pragma unroll
                for (uint c = 0u; c < C; ++c) v[c] = sBase[c * hidden + h];
                const float b = float(bo[h]);
                #pragma unroll
                for (uint i = 0u; i < C; ++i) {
                    float ns = post[i] * b;
                    #pragma unroll
                    for (uint j = 0u; j < C; ++j) ns = fma(mix[i * C + j], v[j], ns);
                    sBase[i * hidden + h] = ns;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    float ss[R];
    float acc[R][NUM_ROWS];
    #pragma unroll
    for (uint r = 0u; r < R; ++r) {
        ss[r] = 0.0f;
        #pragma unroll
        for (uint q = 0u; q < NUM_ROWS; ++q) acc[r][q] = 0.0f;
    }
    device const float4* s4 = (device const float4*)(mhcStream + (size_t)t0 * width);
    for (uint k4 = tid; k4 < width4; k4 += 256u) {
        float4 v[R];
        #pragma unroll
        for (uint r = 0u; r < R; ++r) {
            v[r] = (r < nRows) ? s4[(size_t)r * width4 + k4] : float4(0.0f);
            ss[r] += dot(v[r], v[r]);
        }
        #pragma unroll
        for (uint q = 0u; q < NUM_ROWS; ++q) {
            const float4 w = ((device const float4*)(routerW + (size_t)q * width))[k4];
            #pragma unroll
            for (uint r = 0u; r < R; ++r) acc[r][q] += dot(v[r], w);
        }
    }
    #pragma unroll
    for (uint r = 0u; r < R; ++r) {
        ss[r] = simd_sum(ss[r]);
        #pragma unroll
        for (uint q = 0u; q < NUM_ROWS; ++q) acc[r][q] = simd_sum(acc[r][q]);
    }
    if (lane == 0u) {
        #pragma unroll
        for (uint r = 0u; r < R; ++r) {
            partSS[r * 8u + sg] = ss[r];
            #pragma unroll
            for (uint q = 0u; q < NUM_ROWS; ++q) partDots[(r * 8u + sg) * NUM_ROWS + q] = acc[r][q];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Lane 0 of simdgroup r runs row r's controls (rows in parallel).
    if (lane == 0u && sg < nRows) {
        const uint r = sg;
        float totSS = 0.0f;
        float tot[NUM_ROWS];
        #pragma unroll
        for (uint q = 0u; q < NUM_ROWS; ++q) tot[q] = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            totSS += partSS[r * 8u + i];
            #pragma unroll
            for (uint q = 0u; q < NUM_ROWS; ++q) tot[q] += partDots[(r * 8u + i) * NUM_ROWS + q];
        }
        qwg_mhc_controls<C>(tot, totSS, width, eps, preScale, realIterations, routerB,
                            mhcControls + (size_t)(t0 + r) * NUM_CTRL, sPre + r * 4u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = 0u; r < nRows; ++r) {
        const uint t = t0 + r;
        qwg_mhc_branch_norm<C>(mhcStream + (size_t)t * width, normGamma, outNorm + (size_t)t * hidden,
                               hidden, eps, sPre + r * 4u, tid, sg, lane, partSS);
        // partSS is rewritten by the next row's norm.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Grid: ceil(rows / QW_MHC_ROWS) threadgroups x 256 threads.
kernel void qwg_mhc_router_norm_rows_f16(device float*       mhcStream      [[buffer(0)]],
                                         device const float* routerW        [[buffer(1)]],
                                         device const float* routerB        [[buffer(2)]],
                                         device const half*  normGamma      [[buffer(3)]],
                                         device half*        outNorm        [[buffer(4)]],
                                         device float*       mhcControls    [[buffer(5)]],
                                         constant uint&      hidden         [[buffer(6)]],
                                         constant float&     eps            [[buffer(7)]],
                                         constant float&     preScale       [[buffer(8)]],
                                         constant uint&      iterations     [[buffer(9)]],
                                         device const half*  mergeBranchOut [[buffer(10)]],
                                         constant uint&      rows           [[buffer(11)]],
                                         uint tg   [[threadgroup_position_in_grid]],
                                         uint sg   [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partSS[QW_MHC_ROWS * 8u];
    threadgroup float partDots[QW_MHC_ROWS * 8u * 24u];
    threadgroup float sPre[QW_MHC_ROWS * 4u];
    if (QW_MHC_STREAMS == 4u) {
        qwg_mhc_router_norm_rows_impl<4u, QW_MHC_ROWS>(mhcStream, routerW, routerB, normGamma, outNorm,
                                                       mhcControls, hidden, eps, preScale, iterations,
                                                       mergeBranchOut, rows, tg, sg, lane,
                                                       partSS, partDots, sPre);
    } else {
        qwg_mhc_router_norm_rows_impl<2u, QW_MHC_ROWS>(mhcStream, routerW, routerB, normGamma, outNorm,
                                                       mhcControls, hidden, eps, preScale, iterations,
                                                       mergeBranchOut, rows, tg, sg, lane,
                                                       partSS, partDots, sPre);
    }
}
