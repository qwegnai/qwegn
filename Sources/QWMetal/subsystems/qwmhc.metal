// qwmhc.metal — Multi-stream Hyper-Connections (mHC) with Sinkhorn-Knopp doubly-stochastic mixing.
// Generalized via template <uint C> (C = 2 or C = 4), specialized by QW_MHC_STREAMS (FC 220).
#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_geometry_constants.metalh"
#include "qw_dequant.metalh"
#include "qw_dequant_f32.metalh"
#include "qw_dequant_multi.metalh"
#include "qw_mhc_common.metalh"
using namespace metal;

template <uint C>
inline void qwg_mhc_enter_f16_impl(device const half* x,
                                   device float*      mhcStream,
                                   uint               hidden,
                                   uint2              tg,
                                   uint               tid) {
    const uint h = tg.x * 256u + tid;
    const uint t = tg.y;
    if (h >= hidden) return;
    const float v = float(x[(size_t)t * hidden + h]);
    const size_t base = (size_t)t * (C * hidden);
    #pragma unroll
    for (uint c = 0u; c < C; ++c) {
        mhcStream[base + c * hidden + h] = v;
    }
}

kernel void qwg_mhc_enter_f16(device const half* x         [[buffer(0)]],
                              device float*      mhcStream [[buffer(1)]],
                              constant uint&     hidden    [[buffer(2)]],
                              uint2 tg  [[threadgroup_position_in_grid]],
                              uint  tid [[thread_index_in_threadgroup]]) {
    if (QW_MHC_STREAMS == 4u) {
        qwg_mhc_enter_f16_impl<4u>(x, mhcStream, hidden, tg, tid);
    } else {
        qwg_mhc_enter_f16_impl<2u>(x, mhcStream, hidden, tg, tid);
    }
}


template <uint C>
inline void qwg_mhc_router_norm_f16_impl(device float*       mhcStream,
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
                                         uint                tg,
                                         uint                sg,
                                         uint                lane,
                                         threadgroup float*  partSS,
                                         threadgroup float*  partDots,
                                         threadgroup float*  sPre) {
    constexpr uint NUM_CTRL = C + C * C;
    constexpr uint NUM_ROWS = 2u * C + C * C;
    const uint tid = sg * 32u + lane;
    const uint t = tg;
    const uint width = C * hidden;
    const uint width4 = width >> 2u;
    const bool hasMerge = (iterations & 0x80000000u) != 0u;
    const uint realIterations = iterations & 0x7FFFFFFFu;

    device float* sBase = mhcStream + (size_t)t * width;

    if (hasMerge) {
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
                for (uint j = 0u; j < C; ++j) {
                    ns = fma(mix[i * C + j], v[j], ns);
                }
                sBase[i * hidden + h] = ns;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    float ss = 0.0f;
    float acc[NUM_ROWS];
    #pragma unroll
    for (uint r = 0u; r < NUM_ROWS; ++r) acc[r] = 0.0f;

    device const float4* sr4 = (device const float4*)sBase;
    for (uint k4 = tid; k4 < width4; k4 += 256u) {
        const float4 v = sr4[k4];
        ss += dot(v, v);
        #pragma unroll
        for (uint r = 0u; r < NUM_ROWS; ++r) {
            device const float4* wr4 = (device const float4*)(routerW + (size_t)r * width);
            acc[r] += dot(v, wr4[k4]);
        }
    }

    ss = simd_sum(ss);
    #pragma unroll
    for (uint r = 0u; r < NUM_ROWS; ++r) {
        acc[r] = simd_sum(acc[r]);
    }

    if (lane == 0u) {
        partSS[sg] = ss;
        #pragma unroll
        for (uint r = 0u; r < NUM_ROWS; ++r) {
            partDots[sg * NUM_ROWS + r] = acc[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float totSS = 0.0f;
        float tot[NUM_ROWS];
        #pragma unroll
        for (uint r = 0u; r < NUM_ROWS; ++r) tot[r] = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            totSS += partSS[i];
            #pragma unroll
            for (uint r = 0u; r < NUM_ROWS; ++r) {
                tot[r] += partDots[i * NUM_ROWS + r];
            }
        }
        qwg_mhc_controls<C>(tot, totSS, width, eps, preScale, realIterations, routerB,
                            mhcControls + (size_t)t * NUM_CTRL, sPre);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    qwg_mhc_branch_norm<C>(sBase, normGamma, outNorm + (size_t)t * hidden, hidden, eps,
                           sPre, tid, sg, lane, partSS);
}

kernel void qwg_mhc_router_norm_f16(device float*       mhcStream      [[buffer(0)]],
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
                                    uint tg   [[threadgroup_position_in_grid]],
                                    uint sg   [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partSS[8];
    threadgroup float partDots[8 * 24];
    threadgroup float sPre[4];
    if (QW_MHC_STREAMS == 4u) {
        qwg_mhc_router_norm_f16_impl<4u>(mhcStream, routerW, routerB, normGamma, outNorm,
                                         mhcControls, hidden, eps, preScale, iterations,
                                         mergeBranchOut, tg, sg, lane, partSS, partDots, sPre);
    } else {
        qwg_mhc_router_norm_f16_impl<2u>(mhcStream, routerW, routerB, normGamma, outNorm,
                                         mhcControls, hidden, eps, preScale, iterations,
                                         mergeBranchOut, tg, sg, lane, partSS, partDots, sPre);
    }
}

// Split router-norm for decode/verify: the fused kernel reads the f32 router from one
// threadgroup per row; here ceil(hidden/64) threadgroups compute partial dots (with the
// deferred merge fused) and a per-row finalize runs the controls + branch-norm tail.
constant constexpr uint QW_MHC_PART_THREADS = 64u;
constant constexpr uint QW_MHC_PART_RG = 4u;
constant constexpr uint QW_MHC_FIN_SLICE = 256u;

template <uint C>
inline void qwg_mhc_router_partial_impl(device float*       mhcStream,
                                        device const float* routerW,
                                        device const float* mhcControls,
                                        device const half*  mergeBranchOut,
                                        device float*       partials,
                                        uint                hidden,
                                        uint                iterations,
                                        uint2               tg,
                                        uint                sg,
                                        uint                lane,
                                        threadgroup float*  red) {
    constexpr uint NUM_CTRL = C + C * C;
    constexpr uint NUM_ROWS = 2u * C + C * C;
    constexpr uint NUM_GRAM = C * (C + 1u) / 2u;
    constexpr uint NUM_OUT = NUM_ROWS + 1u + NUM_GRAM;
    constexpr uint NUM_SG = QW_MHC_PART_THREADS * QW_MHC_PART_RG / 32u;
    const uint tid = sg * 32u + lane;
    // Row group rg owns router rows rg, rg + RG, ...; group 0 also owns ss/Gram and the merge store.
    const uint rg = tid / QW_MHC_PART_THREADS;
    const uint p = tg.x;
    const uint t = tg.y;
    const uint nParts = (hidden + QW_MHC_PART_THREADS - 1u) / QW_MHC_PART_THREADS;
    const uint width = C * hidden;
    const uint h = p * QW_MHC_PART_THREADS + (tid % QW_MHC_PART_THREADS);
    const bool hasMerge = (iterations & 0x80000000u) != 0u;
    device float* sBase = mhcStream + (size_t)t * width;

    float v[C];
    #pragma unroll
    for (uint c = 0u; c < C; ++c) v[c] = 0.0f;
    if (h < hidden) {
        #pragma unroll
        for (uint c = 0u; c < C; ++c) v[c] = sBase[c * hidden + h];
        if (hasMerge) {
            device const float* prevCtrl = mhcControls + (size_t)t * NUM_CTRL;
            const float b = float(mergeBranchOut[(size_t)t * hidden + h]);
            float nv[C];
            #pragma unroll
            for (uint i = 0u; i < C; ++i) {
                float ns = prevCtrl[i] * b;
                #pragma unroll
                for (uint j = 0u; j < C; ++j) ns = fma(prevCtrl[C + i * C + j], v[j], ns);
                nv[i] = ns;
            }
            #pragma unroll
            for (uint i = 0u; i < C; ++i) v[i] = nv[i];
        }
    }
    if (hasMerge) {
        // Every row group must read the pre-merge stream before group 0 overwrites it.
        threadgroup_barrier(mem_flags::mem_device);
        if (rg == 0u && h < hidden) {
            #pragma unroll
            for (uint i = 0u; i < C; ++i) sBase[i * hidden + h] = v[i];
        }
    }

    float ss = 0.0f;
    float acc[NUM_ROWS];
    #pragma unroll
    for (uint c = 0u; c < C; ++c) ss = fma(v[c], v[c], ss);
    #pragma unroll
    for (uint r = 0u; r < NUM_ROWS; ++r) {
        float a = 0.0f;
        if (h < hidden && (r % QW_MHC_PART_RG) == rg) {
            device const float* wr = routerW + (size_t)r * width + h;
            #pragma unroll
            for (uint c = 0u; c < C; ++c) a = fma(v[c], wr[c * hidden], a);
        }
        acc[r] = ((r % QW_MHC_PART_RG) == rg) ? simd_sum(a) : 0.0f;
    }
    // Stream Gram entries let the finalize derive the branch RMS (pre^T G pre) without a
    // second full pass, so it can spread the branch norm over many threadgroups.
    float gram[NUM_GRAM];
    {
        uint k = 0u;
        #pragma unroll
        for (uint c = 0u; c < C; ++c) {
            #pragma unroll
            for (uint d = c; d < C; ++d) gram[k++] = (rg == 0u) ? simd_sum(v[c] * v[d]) : 0.0f;
        }
    }
    ss = (rg == 0u) ? simd_sum(ss) : 0.0f;
    if (lane == 0u) {
        red[sg * NUM_OUT] = ss;
        #pragma unroll
        for (uint r = 0u; r < NUM_ROWS; ++r) red[sg * NUM_OUT + 1u + r] = acc[r];
        #pragma unroll
        for (uint k = 0u; k < NUM_GRAM; ++k) red[sg * NUM_OUT + 1u + NUM_ROWS + k] = gram[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NUM_OUT) {
        float a = 0.0f;
        #pragma unroll
        for (uint i = 0u; i < NUM_SG; ++i) a += red[i * NUM_OUT + tid];
        partials[((size_t)t * nParts + p) * NUM_OUT + tid] = a;
    }
}

kernel void qwg_mhc_router_partial_f32(device float*       mhcStream      [[buffer(0)]],
                                       device const float* routerW        [[buffer(1)]],
                                       device const float* mhcControls    [[buffer(2)]],
                                       device const half*  mergeBranchOut [[buffer(3)]],
                                       device float*       partials       [[buffer(4)]],
                                       constant uint&      hidden         [[buffer(5)]],
                                       constant uint&      iterations     [[buffer(6)]],
                                       uint2 tg   [[threadgroup_position_in_grid]],
                                       uint  sg   [[simdgroup_index_in_threadgroup]],
                                       uint  lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[(QW_MHC_PART_THREADS * QW_MHC_PART_RG / 32u) * 35u];
    if (QW_MHC_STREAMS == 4u) {
        qwg_mhc_router_partial_impl<4u>(mhcStream, routerW, mhcControls, mergeBranchOut, partials,
                                        hidden, iterations, tg, sg, lane, red);
    } else {
        qwg_mhc_router_partial_impl<2u>(mhcStream, routerW, mhcControls, mergeBranchOut, partials,
                                        hidden, iterations, tg, sg, lane, red);
    }
}

template <uint C>
inline void qwg_mhc_router_finalize_impl(device const float* mhcStream,
                                         device const float* partials,
                                         device const float* routerB,
                                         device const half*  normGamma,
                                         device half*        outNorm,
                                         device float*       mhcControls,
                                         uint                hidden,
                                         float               eps,
                                         float               preScale,
                                         uint                iterations,
                                         uint2               tg,
                                         uint                sg,
                                         uint                lane,
                                         threadgroup float*  tots,
                                         threadgroup float*  sPre) {
    constexpr uint NUM_CTRL = C + C * C;
    constexpr uint NUM_ROWS = 2u * C + C * C;
    constexpr uint NUM_GRAM = C * (C + 1u) / 2u;
    constexpr uint NUM_OUT = NUM_ROWS + 1u + NUM_GRAM;
    const uint tid = sg * 32u + lane;
    const uint slice = tg.x;
    const uint t = tg.y;
    const bool writeCtrl = slice == 0u;
    const uint width = C * hidden;
    const uint nParts = (hidden + QW_MHC_PART_THREADS - 1u) / QW_MHC_PART_THREADS;
    device const float* part = partials + (size_t)t * nParts * NUM_OUT;
    // Each simdgroup reduces outputs sg, sg+8, ...; lanes stride over parts so loads issue in parallel.
    for (uint o = sg; o < NUM_OUT; o += 8u) {
        float a = 0.0f;
        for (uint p = lane; p < nParts; p += 32u) a += part[p * NUM_OUT + o];
        a = simd_sum(a);
        if (lane == 0u) tots[o] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Every slice threadgroup recomputes the (cheap) controls; only slice 0 stores them.
    if (C == 4u) {
        if (sg == 0u) {
            qwg_mhc_controls4_simd(tots, width, eps, preScale, iterations & 0x7FFFFFFFu, routerB,
                                   mhcControls + (size_t)t * NUM_CTRL, sPre, lane, writeCtrl);
        }
    } else if (tid == 0u) {
        float tot[NUM_ROWS];
        #pragma unroll
        for (uint r = 0u; r < NUM_ROWS; ++r) tot[r] = tots[1u + r];
        // C == 2 slices store identical controls; the redundant writes are benign.
        qwg_mhc_controls<C>(tot, tots[0], width, eps, preScale, iterations & 0x7FFFFFFFu, routerB,
                            mhcControls + (size_t)t * NUM_CTRL, sPre);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float pre[C];
    #pragma unroll
    for (uint c = 0u; c < C; ++c) pre[c] = sPre[c];
    float ssBranch = 0.0f;
    {
        uint k = 0u;
        #pragma unroll
        for (uint c = 0u; c < C; ++c) {
            #pragma unroll
            for (uint d = c; d < C; ++d) {
                const float g = tots[1u + NUM_ROWS + k++];
                ssBranch = fma((c == d ? 1.0f : 2.0f) * pre[c] * pre[d], g, ssBranch);
            }
        }
    }
    const float rstdBranch = rsqrt(max(ssBranch, 0.0f) / float(hidden) + eps);

    device const float* sBase = mhcStream + (size_t)t * width;
    device half* dst = outNorm + (size_t)t * hidden;
    const uint h = slice * QW_MHC_FIN_SLICE + tid;
    float bv = 0.0f;
    if (h < hidden) {
        #pragma unroll
        for (uint c = 0u; c < C; ++c) bv = fma(pre[c], sBase[c * hidden + h], bv);
    }
    float v = bv * rstdBranch * (1.0f + (h < hidden ? float(normGamma[h]) : 0.0f));
    v = qwg_fwht32_lane_f32(v, lane);
    if (h < hidden) dst[h] = half(v);
}

kernel void qwg_mhc_router_finalize_f16(device const float* mhcStream   [[buffer(0)]],
                                        device const float* partials    [[buffer(1)]],
                                        device const float* routerB     [[buffer(2)]],
                                        device const half*  normGamma   [[buffer(3)]],
                                        device half*        outNorm     [[buffer(4)]],
                                        device float*       mhcControls [[buffer(5)]],
                                        constant uint&      hidden      [[buffer(6)]],
                                        constant float&     eps         [[buffer(7)]],
                                        constant float&     preScale    [[buffer(8)]],
                                        constant uint&      iterations  [[buffer(9)]],
                                        uint2 tg  [[threadgroup_position_in_grid]],
                                        uint sg   [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tots[35];
    threadgroup float sPre[4];
    if (QW_MHC_STREAMS == 4u) {
        qwg_mhc_router_finalize_impl<4u>(mhcStream, partials, routerB, normGamma, outNorm, mhcControls,
                                         hidden, eps, preScale, iterations, tg, sg, lane,
                                         tots, sPre);
    } else {
        qwg_mhc_router_finalize_impl<2u>(mhcStream, partials, routerB, normGamma, outNorm, mhcControls,
                                         hidden, eps, preScale, iterations, tg, sg, lane,
                                         tots, sPre);
    }
}

template <uint C>
inline void qwg_mhc_merge_f16_impl(device float*       mhcStream,
                                   device const half*  branchOut,
                                   device const float* mhcControls,
                                   device half*        xOut,
                                   device const half*  nextGamma,
                                   device half*        nextNormOut,
                                   uint                hidden,
                                   float               eps,
                                   uint                flags,
                                   uint                tg,
                                   uint                sg,
                                   uint                lane,
                                   threadgroup float*  partSS) {
    constexpr uint NUM_CTRL = C + C * C;
    constexpr float INV_C = 1.0f / float(C);
    const uint tid = sg * 32u + lane;
    const uint t = tg;
    const uint width = C * hidden;
    device float* sBase = mhcStream + (size_t)t * width;
    device const half* bo = branchOut + (size_t)t * hidden;
    device const float* ctrl = mhcControls + (size_t)t * NUM_CTRL;

    float post[C];
    float mix[C * C];
    #pragma unroll
    for (uint c = 0u; c < C; ++c) post[c] = ctrl[c];
    #pragma unroll
    for (uint m = 0u; m < C * C; ++m) mix[m] = ctrl[C + m];

    const bool isLeave = (flags & 1u) != 0u;
    const bool hasNextNorm = (flags & 2u) != 0u;

    float ss = 0.0f;
    for (uint h = tid; h < hidden; h += 256u) {
        float v[C];
        #pragma unroll
        for (uint c = 0u; c < C; ++c) v[c] = sBase[c * hidden + h];
        const float b = float(bo[h]);
        float sumS = 0.0f;
        #pragma unroll
        for (uint i = 0u; i < C; ++i) {
            float ns = post[i] * b;
            #pragma unroll
            for (uint j = 0u; j < C; ++j) {
                ns = fma(mix[i * C + j], v[j], ns);
            }
            sBase[i * hidden + h] = ns;
            sumS += ns;
        }
        if (isLeave) {
            const half xh = half(INV_C * sumS);
            xOut[(size_t)t * hidden + h] = xh;
            const float xf = float(xh);
            ss += xf * xf;
        }
    }

    if (hasNextNorm) {
        ss = simd_sum(ss);
        if (lane == 0u) partSS[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float totSS = 0.0f;
        for (uint i = 0u; i < 8u; ++i) totSS += partSS[i];
        const float rstd = rsqrt(totSS / float(hidden) + eps);
        device half* dst = nextNormOut + (size_t)t * hidden;
        for (uint h = tid; h < hidden; h += 256u) {
            const float xf = float(xOut[(size_t)t * hidden + h]);
            float v = xf * rstd * (1.0f + float(nextGamma[h]));
            v = qwg_fwht32_lane_f32(v, lane);
            dst[h] = half(v);
        }
    }
}

kernel void qwg_mhc_merge_f16(device float*       mhcStream   [[buffer(0)]],
                              device const half*  branchOut   [[buffer(1)]],
                              device const float* mhcControls [[buffer(2)]],
                              device half*        xOut        [[buffer(3)]],
                              device const half*  nextGamma   [[buffer(4)]],
                              device half*        nextNormOut [[buffer(5)]],
                              constant uint&      hidden      [[buffer(6)]],
                              constant float&     eps         [[buffer(7)]],
                              constant uint&      flags       [[buffer(8)]],
                              uint tg   [[threadgroup_position_in_grid]],
                              uint sg   [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partSS[8];
    if (QW_MHC_STREAMS == 4u) {
        qwg_mhc_merge_f16_impl<4u>(mhcStream, branchOut, mhcControls, xOut, nextGamma,
                                   nextNormOut, hidden, eps, flags, tg, sg, lane, partSS);
    } else {
        qwg_mhc_merge_f16_impl<2u>(mhcStream, branchOut, mhcControls, xOut, nextGamma,
                                   nextNormOut, hidden, eps, flags, tg, sg, lane, partSS);
    }
}
