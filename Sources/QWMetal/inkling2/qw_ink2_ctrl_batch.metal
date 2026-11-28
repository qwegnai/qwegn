// qw_ink2_ctrl_batch.metal — Batched 1..3 branch loop controller kernels for Inkling-2.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "inkling2/qw_ink2_common.metalh"

using namespace metal;

#if !defined(QW_DISABLE_NAX)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#if !defined(__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__) || __ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ < 260200
#error "NAX cooperative_tensor kernels require -mmacosx-version-min=26.2 or newer (Apple MPP operand_index ABI)."
#endif
#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"

using namespace mlx::steel;
#endif // !defined(QW_DISABLE_NAX)

struct QWInk2CtrlBatchParams {
    uint T, H, rank, thStride;
    float eps;
    uint numBranches, msgBaseSlot, pad1;
    uint4 arIdx;
    uint4 downElemOff;
    uint4 forgetElemOff;
    uint4 gainElemOff;
    uint4 normElemOff;
};

struct QWInk2AttnResPackPairBatchParams {
    uint T, H, rank, thStride;
    uint nBlocks, blockStride, partialIsF32, normalizeValues;
    float arEps, rmsEps;
    uint numBranches, msgBaseSlot;
    uint4 arIdx;
    uint slotIds[8];
};

static inline void qw_ink2_batch_ar_reduce(
        float sq, float qd, uint H, float eps,
        threadgroup float* sgSq, threadgroup float* sgDot,
        threadgroup float* invRmsSh, threadgroup float* alphaSh,
        uint b, uint simd_id, uint lane) {
    sq = simd_sum(sq);
    qd = simd_sum(qd);
    if (lane == 0u) {
        sgSq[simd_id] = sq;
        sgDot[simd_id] = qd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        const float vSq = simd_sum((lane < 8u) ? sgSq[lane] : 0.0f);
        const float vDot = simd_sum((lane < 8u) ? sgDot[lane] : 0.0f);
        if (lane == 0u) {
            const float inv = rsqrt(vSq / float(H) + eps);
            invRmsSh[b] = inv;
            alphaSh[b] = (vDot * inv) * rsqrt(float(H));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/// Fused AttnRes + CtrlPackPair over 1..3 branches (`grid = (T, numBranches, 1)`).
kernel void qwg_ink2_attn_res_pack_pair_batch(device const bfloat* blocksBuf [[buffer(0)]],
                                              device const bfloat* partialBF16 [[buffer(1)]],
                                              device const float* partialF32 [[buffer(2)]],
                                              device const float* queryF32 [[buffer(3)]],
                                              device const float* msgF32 [[buffer(4)]],
                                              device float* attnResOutF32 [[buffer(5)]],
                                              device bfloat* pairBF16 [[buffer(6)]],
                                              constant QWInk2AttnResPackPairBatchParams& p [[buffer(7)]],
                                              uint2 tg [[threadgroup_position_in_grid]],
                                              uint tid [[thread_index_in_threadgroup]],
                                              uint simd_id [[simdgroup_index_in_threadgroup]],
                                              uint lane [[thread_index_in_simdgroup]]) {
    const uint t = tg.x, b = tg.y;
    if (t >= p.T || b >= p.numBranches || b >= 3u || p.H > 4096u) return;
    const uint q = p.arIdx[b];
    const uint nBlocks = min(p.nBlocks, 7u);
    const uint nCand = nBlocks + 1u;
    const uint nSteps = min(16u, (p.H + 255u) >> 8u);
    device const float* qRow = queryF32 + size_t(q) * p.H;

    threadgroup float sgSq[8], sgDot[8], invRmsSh[8], alphaSh[8];
    uint blkReg[7][8];
    float partReg[16];

    for (uint cand = 0u; cand < nBlocks; ++cand) {
        device const bfloat* bp = blocksBuf + size_t(p.slotIds[cand]) * p.blockStride + size_t(t) * p.H;
        float sq = 0.0f, qd = 0.0f;
        for (uint k = 0u; k < 8u; ++k) {
            const uint d0 = tid + (2u * k) * 256u;
            if (d0 >= p.H) break;
            const uint d1 = d0 + 256u;
            const bfloat b0 = bp[d0];
            const bfloat b1 = (d1 < p.H) ? bp[d1] : bfloat(0.0f);
            blkReg[cand][k] = uint(as_type<ushort>(b0)) | (uint(as_type<ushort>(b1)) << 16u);
            const float v0 = float(b0);
            sq += v0 * v0;
            qd += v0 * qRow[d0];
            if (d1 < p.H) {
                const float v1 = float(b1);
                sq += v1 * v1;
                qd += v1 * qRow[d1];
            }
        }
        qw_ink2_batch_ar_reduce(sq, qd, p.H, p.arEps, sgSq, sgDot, invRmsSh, alphaSh, cand, simd_id, lane);
    }

    {
        float sq = 0.0f, qd = 0.0f;
        device const float* pf = partialF32 + size_t(t) * p.H;
        device const bfloat* pb = partialBF16 + size_t(t) * p.H;
        for (uint k = 0u; k < nSteps; ++k) {
            const uint d = tid + (k << 8u);
            if (d < p.H) {
                const float v = (p.partialIsF32 != 0u) ? pf[d] : float(pb[d]);
                partReg[k] = v;
                sq += v * v;
                qd += v * qRow[d];
            }
        }
        qw_ink2_batch_ar_reduce(sq, qd, p.H, p.arEps, sgSq, sgDot, invRmsSh, alphaSh, nBlocks, simd_id, lane);
    }

    if (tid == 0u) {
        float mx = -1e30f;
        for (uint cand = 0u; cand < nCand; ++cand) mx = max(mx, alphaSh[cand]);
        float s = 0.0f;
        for (uint cand = 0u; cand < nCand; ++cand) {
            const float w = exp(alphaSh[cand] - mx);
            alphaSh[cand] = w;
            s += w;
        }
        const float invS = 1.0f / max(s, 1e-30f);
        for (uint cand = 0u; cand < nCand; ++cand) alphaSh[cand] *= invS;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool writeAr = true;
    for (uint j = 0u; j < b; ++j) {
        if (p.arIdx[j] == q) writeAr = false;
    }
    device float* arDst = attnResOutF32 + size_t(q * p.T + t) * p.H;
    device const float* mp = msgF32 + size_t((p.msgBaseSlot + b) * p.T + t) * p.H;
    float mv[16];
    float sqX = 0.0f, sqM = 0.0f;

    for (uint k = 0u; k < (nSteps + 1u) / 2u; ++k) {
        const uint d0 = tid + (2u * k) * 256u;
        const uint d1 = d0 + 256u;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (uint cand = 0u; cand < nBlocks; ++cand) {
            const uint pk = blkReg[cand][k];
            const float inv = invRmsSh[cand];
            const float a = alphaSh[cand];
            const float v0 = float(as_type<bfloat>(ushort(pk)));
            const float v1 = float(as_type<bfloat>(ushort(pk >> 16u)));
            acc0 += a * ((p.normalizeValues != 0u) ? (v0 * inv) : v0);
            acc1 += a * ((p.normalizeValues != 0u) ? (v1 * inv) : v1);
        }
        const float invP = invRmsSh[nBlocks];
        const float aP = alphaSh[nBlocks];
        if (d0 < p.H) {
            const float vp0 = partReg[2u * k];
            acc0 += aP * ((p.normalizeValues != 0u) ? (vp0 * invP) : vp0);
            partReg[2u * k] = acc0;
            if (writeAr) arDst[d0] = acc0;
            const float m0 = mp[d0];
            mv[2u * k] = m0;
            sqX += acc0 * acc0;
            sqM += m0 * m0;
        }
        if (d1 < p.H) {
            const float vp1 = partReg[2u * k + 1u];
            acc1 += aP * ((p.normalizeValues != 0u) ? (vp1 * invP) : vp1);
            partReg[2u * k + 1u] = acc1;
            if (writeAr) arDst[d1] = acc1;
            const float m1 = mp[d1];
            mv[2u * k + 1u] = m1;
            sqX += acc1 * acc1;
            sqM += m1 * m1;
        }
    }

    sqX = simd_sum(sqX);
    sqM = simd_sum(sqM);
    if (lane == 0u) {
        sgSq[simd_id] = sqX;
        sgDot[simd_id] = sqM;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        const float vx = simd_sum((lane < 8u) ? sgSq[lane] : 0.0f);
        const float vm = simd_sum((lane < 8u) ? sgDot[lane] : 0.0f);
        if (lane == 0u) {
            sgSq[0] = rsqrt(vx / float(p.H) + p.rmsEps);
            sgDot[0] = rsqrt(vm / float(p.H) + p.rmsEps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invX = sgSq[0], invM = sgDot[0];
    device bfloat* pairDst = pairBF16 + size_t(b * p.T + t) * (2u * p.H);
    for (uint k = 0u; k < nSteps; ++k) {
        const uint d = tid + (k << 8u);
        if (d < p.H) {
            pairDst[d] = qw_ink2_bf16(partReg[k] * invX);
            pairDst[p.H + d] = qw_ink2_bf16(mv[k] * invM);
        }
    }
}

inline float qw_ink2_batch_tanh_precise(float x) {
    return precise::tanh(clamp(x, -15.0f, 15.0f));
}

inline float qw_ink2_batch_dot4_init(float4 a, float4 b) {
    return fma(a.w, b.w, fma(a.z, b.z, fma(a.y, b.y, a.x * b.x)));
}

inline float qw_ink2_batch_dot8_fma(float4 w0, float4 w1, float4 x0, float4 x1, float acc) {
    const float a0 = fma(w0.w, x0.w, fma(w0.z, x0.z, fma(w0.y, x0.y, fma(w0.x, x0.x, acc))));
    return fma(w1.w, x1.w, fma(w1.z, x1.z, fma(w1.y, x1.y, fma(w1.x, x1.x, a0))));
}

kernel void qwg_ink2_ctrl_pack_pair_batch(device const float* xF32 [[buffer(0)]],
                                          device const float* msgF32 [[buffer(1)]],
                                          device bfloat* pairBF16 [[buffer(2)]],
                                          constant QWInk2CtrlBatchParams& p [[buffer(3)]],
                                          uint2 tg [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint simd_id [[simdgroup_index_in_threadgroup]],
                                          uint lane [[thread_index_in_simdgroup]]) {
    const uint t = tg.x, b = tg.y;
    if (t >= p.T || b >= p.numBranches || b >= 3u) return;
    const uint aSlot = p.arIdx[b];
    const uint mSlot = p.msgBaseSlot + b;
    device const float* xp = xF32 + size_t(aSlot * p.T + t) * p.H;
    device const float* mp = msgF32 + size_t(mSlot * p.T + t) * p.H;

    const uint nSteps = min(16u, (p.H + 255u) >> 8u);
    float xv[16], mv[16];
    float sqX = 0.0f, sqM = 0.0f;
    for (uint k = 0u; k < nSteps; ++k) {
        const uint d = tid + (k << 8u);
        if (d < p.H) {
            const float x0 = xp[d], m0 = mp[d];
            xv[k] = x0;
            mv[k] = m0;
            sqX += x0 * x0;
            sqM += m0 * m0;
        }
    }
    for (uint d = tid + 4096u; d < p.H; d += 256u) {
        const float x0 = xp[d], m0 = mp[d];
        sqX += x0 * x0;
        sqM += m0 * m0;
    }
    sqX = simd_sum(sqX);
    sqM = simd_sum(sqM);
    threadgroup float sgX[8], sgM[8];
    if (lane == 0u) {
        sgX[simd_id] = sqX;
        sgM[simd_id] = sqM;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        const float vx = simd_sum((lane < 8u) ? sgX[lane] : 0.0f);
        const float vm = simd_sum((lane < 8u) ? sgM[lane] : 0.0f);
        if (lane == 0u) {
            sgX[0] = rsqrt(vx / float(p.H) + p.eps);
            sgM[0] = rsqrt(vm / float(p.H) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invX = sgX[0], invM = sgM[0];
    device bfloat* dst = pairBF16 + size_t(b * p.T + t) * (2u * p.H);
    for (uint k = 0u; k < nSteps; ++k) {
        const uint d = tid + (k << 8u);
        if (d < p.H) {
            dst[d] = qw_ink2_bf16(xv[k] * invX);
            dst[p.H + d] = qw_ink2_bf16(mv[k] * invM);
        }
    }
    for (uint d = tid + 4096u; d < p.H; d += 256u) {
        dst[d] = qw_ink2_bf16(xp[d] * invX);
        dst[p.H + d] = qw_ink2_bf16(mp[d] * invM);
    }
}

kernel void qwg_ink2_ctrl_down_gemv_batch(device const bfloat* pairBF16 [[buffer(0)]],
                                          device const bfloat* weightSlab [[buffer(1)]],
                                          device float* hidF32 [[buffer(2)]],
                                          constant QWInk2CtrlBatchParams& p [[buffer(3)]],
                                          uint2 tg [[threadgroup_position_in_grid]],
                                          uint sgi [[simdgroup_index_in_threadgroup]],
                                          uint lane [[thread_index_in_simdgroup]]) {
    const uint r = tg.x * 4u + sgi;
    const uint b = tg.y;
    if (r >= p.rank || b >= p.numBranches || b >= 3u || p.T == 0u || p.T > 16u) return;

    const uint inK = 2u * p.H;
    device const bfloat* wRow = weightSlab + size_t(p.downElemOff[b]) + size_t(r) * inK;
    device const bfloat* xBase = pairBF16 + size_t(b * p.T) * inK;

    float acc[16];
    #pragma unroll
    for (uint m = 0u; m < 16u; ++m) acc[m] = 0.0f;

    const uint vecEnd = inK & ~7u;
    for (uint k = lane * 8u; k < vecEnd; k += 256u) {
        const ushort4 wu = *(device const ushort4*)(wRow + k);
        const float4 wLo = float4(as_type<bfloat>(wu.x), as_type<bfloat>(wu.y),
                                  as_type<bfloat>(wu.z), as_type<bfloat>(wu.w));
        const ushort4 wu2 = *(device const ushort4*)(wRow + k + 4u);
        const float4 wHi = float4(as_type<bfloat>(wu2.x), as_type<bfloat>(wu2.y),
                                  as_type<bfloat>(wu2.z), as_type<bfloat>(wu2.w));
        #pragma unroll
        for (uint m = 0u; m < 16u; ++m) {
            if (m < p.T) {
                device const bfloat* xm = xBase + size_t(m) * inK + k;
                const ushort4 xu = *(device const ushort4*)xm;
                const ushort4 xu2 = *(device const ushort4*)(xm + 4u);
                const float4 xLo = float4(as_type<bfloat>(xu.x), as_type<bfloat>(xu.y),
                                          as_type<bfloat>(xu.z), as_type<bfloat>(xu.w));
                const float4 xHi = float4(as_type<bfloat>(xu2.x), as_type<bfloat>(xu2.y),
                                          as_type<bfloat>(xu2.z), as_type<bfloat>(xu2.w));
                acc[m] = qw_ink2_batch_dot8_fma(wLo, wHi, xLo, xHi, acc[m]);
            }
        }
    }
    for (uint k = vecEnd + lane; k < inK; k += 32u) {
        const float wv = float(wRow[k]);
        #pragma unroll
        for (uint m = 0u; m < 16u; ++m) {
            if (m < p.T) acc[m] = fma(wv, float(xBase[size_t(m) * inK + k]), acc[m]);
        }
    }

    #pragma unroll
    for (uint m = 0u; m < 16u; ++m) {
        if (m < p.T) {
            const float s = simd_sum(acc[m]);
            if (lane == 0u) hidF32[size_t(b * p.T + m) * p.rank + r] = s;
        }
    }
}

kernel void qwg_ink2_ctrl_apply_f32_batch(device const float* xF32 [[buffer(0)]],
                                          device const float* msgF32 [[buffer(1)]],
                                          device const float* hidF32 [[buffer(2)]],
                                          device const bfloat* weightSlab [[buffer(3)]],
                                          device float* outF32 [[buffer(4)]],
                                          constant QWInk2CtrlBatchParams& p [[buffer(5)]],
                                          uint3 tg [[threadgroup_position_in_grid]],
                                          uint simd_id [[simdgroup_index_in_threadgroup]],
                                          uint lane [[thread_index_in_simdgroup]]) {
    const uint t = tg.y, b = tg.z;
    if (t >= p.T || b >= p.numBranches || b >= 3u) return;

    device const float* hp = hidF32 + size_t(b * p.T + t) * p.rank;
    device const bfloat* forgetBase = weightSlab + size_t(p.forgetElemOff[b]);
    device const bfloat* gainBase = weightSlab + size_t(p.gainElemOff[b]);
    const size_t xBase = size_t(p.arIdx[b] * p.T + t) * p.H;
    const size_t mBase = size_t((p.msgBaseSlot + b) * p.T + t) * p.H;
    const size_t oBase = size_t(b * p.T + t) * p.H;

    if (p.rank == 64u && ((size_t(hp) & 15u) == 0u)
        && ((size_t(forgetBase) & 7u) == 0u) && ((size_t(gainBase) & 7u) == 0u)) {
        const uint dBase = tg.x * 64u + simd_id * 8u;
        if (dBase >= p.H) return;
        const uint sub = lane >> 4u;
        const uint l16 = lane & 15u;
        const float4 h4 = *(device const float4*)(hp + l16 * 4u);
        #pragma unroll
        for (uint k = 0u; k < 4u; ++k) {
            const uint rr = dBase + k * 2u + sub;
            float2 fg = float2(0.0f);
            if (rr < p.H) {
                const ushort4 fu = *(device const ushort4*)(forgetBase + size_t(rr) * 64u + l16 * 4u);
                const ushort4 gu = *(device const ushort4*)(gainBase + size_t(rr) * 64u + l16 * 4u);
                const float4 ff = float4(as_type<bfloat>(fu.x), as_type<bfloat>(fu.y),
                                         as_type<bfloat>(fu.z), as_type<bfloat>(fu.w));
                const float4 gf = float4(as_type<bfloat>(gu.x), as_type<bfloat>(gu.y),
                                         as_type<bfloat>(gu.z), as_type<bfloat>(gu.w));
                fg.x = qw_ink2_batch_dot4_init(ff, h4);
                fg.y = qw_ink2_batch_dot4_init(gf, h4);
            }
            fg += simd_shuffle_xor(fg, ushort(8));
            fg += simd_shuffle_xor(fg, ushort(4));
            fg += simd_shuffle_xor(fg, ushort(2));
            fg += simd_shuffle_xor(fg, ushort(1));
            if (l16 == 0u && rr < p.H) {
                const float f = qw_ink2_batch_tanh_precise(fg.x);
                const float g = qw_ink2_batch_tanh_precise(fg.y);
                outF32[oBase + rr] = (1.0f - f) * xF32[xBase + rr] + g * msgF32[mBase + rr];
            }
        }
        return;
    }

    const uint d0 = tg.x * 32u + simd_id * 4u;
    if (d0 >= p.H) return;
    device const bfloat* fw[4];
    device const bfloat* gw[4];
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        const uint rr = (d0 + j < p.H) ? (d0 + j) : d0;
        fw[j] = forgetBase + size_t(rr) * p.rank;
        gw[j] = gainBase + size_t(rr) * p.rank;
    }

    float4 fAcc = float4(0.0f), gAcc = float4(0.0f);
    if ((p.rank & 1u) == 0u && ((size_t(hp) & 7u) == 0u)
        && ((size_t(forgetBase) & 3u) == 0u) && ((size_t(gainBase) & 3u) == 0u)) {
        for (uint r = lane * 2u; r < p.rank; r += 64u) {
            const float2 h2 = *(device const float2*)(hp + r);
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) {
                if (d0 + j < p.H) {
                    const ushort2 fu = *(device const ushort2*)(fw[j] + r);
                    const ushort2 gu = *(device const ushort2*)(gw[j] + r);
                    const float2 ff = float2(as_type<bfloat>(fu.x), as_type<bfloat>(fu.y));
                    const float2 gf = float2(as_type<bfloat>(gu.x), as_type<bfloat>(gu.y));
                    fAcc[j] += dot(h2, ff);
                    gAcc[j] += dot(h2, gf);
                }
            }
        }
    } else {
        for (uint r = lane; r < p.rank; r += 32u) {
            const float hr = hp[r];
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) {
                if (d0 + j < p.H) {
                    fAcc[j] += hr * float(fw[j][r]);
                    gAcc[j] += hr * float(gw[j][r]);
                }
            }
        }
    }

    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        const float fDot = simd_sum(fAcc[j]);
        const float gDot = simd_sum(gAcc[j]);
        if (lane == 0u && d0 + j < p.H) {
            const uint d = d0 + j;
            const float f = qw_ink2_batch_tanh_precise(fDot);
            const float g = qw_ink2_batch_tanh_precise(gDot);
            outF32[oBase + d] = (1.0f - f) * xF32[xBase + d] + g * msgF32[mBase + d];
        }
    }
}

kernel void qwg_ink2_ctrl_rms_norm_batch(device const float* inF32 [[buffer(0)]],
                                         device const bfloat* weightSlab [[buffer(1)]],
                                         device bfloat* outBF16 [[buffer(2)]],
                                         constant QWInk2CtrlBatchParams& p [[buffer(3)]],
                                         uint2 tg [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint simd_id [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    const uint t = tg.x, b = tg.y;
    if (t >= p.T || b >= p.numBranches || b >= 3u) return;
    const size_t base = size_t(b * p.T + t) * p.H;
    device const float* src = inF32 + base;
    device const bfloat* nw = weightSlab + size_t(p.normElemOff[b]);

    const uint nSteps = min(16u, (p.H + 255u) >> 8u);
    float vReg[16];
    float sq = 0.0f;
    for (uint k = 0u; k < nSteps; ++k) {
        const uint d = tid + (k << 8u);
        if (d < p.H) {
            const float v = src[d];
            vReg[k] = v;
            sq += v * v;
        }
    }
    for (uint d = tid + 4096u; d < p.H; d += 256u) {
        const float v = src[d];
        sq += v * v;
    }
    sq = simd_sum(sq);
    threadgroup float sgSums[8];
    if (lane == 0u) sgSums[simd_id] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        const float val = simd_sum((lane < 8u) ? sgSums[lane] : 0.0f);
        if (lane == 0u) sgSums[0] = rsqrt(val / float(p.H) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invRms = sgSums[0];
    device bfloat* dst = outBF16 + base;
    for (uint k = 0u; k < nSteps; ++k) {
        const uint d = tid + (k << 8u);
        if (d < p.H) dst[d] = qw_ink2_bf16((vReg[k] * invRms) * float(nw[d]));
    }
    for (uint d = tid + 4096u; d < p.H; d += 256u) {
        dst[d] = qw_ink2_bf16((src[d] * invRms) * float(nw[d]));
    }
}

kernel void qwg_ink2_ctrl_blend_norm_batch(device const float* xF32 [[buffer(0)]],
                                           device const float* msgF32 [[buffer(1)]],
                                           device const float* fgF32 [[buffer(2)]],
                                           device const bfloat* weightSlab [[buffer(3)]],
                                           device bfloat* outBF16 [[buffer(4)]],
                                           constant QWInk2CtrlBatchParams& p [[buffer(5)]],
                                           uint2 tg [[threadgroup_position_in_grid]],
                                           uint tid [[thread_index_in_threadgroup]],
                                           uint simd_id [[simdgroup_index_in_threadgroup]],
                                           uint lane [[thread_index_in_simdgroup]]) {
    const uint t = tg.x, b = tg.y;
    if (t >= p.T || b >= p.numBranches || b >= 3u) return;
    device const float* xp = xF32 + size_t(p.arIdx[b] * p.T + t) * p.H;
    device const float* mp = msgF32 + size_t((p.msgBaseSlot + b) * p.T + t) * p.H;
    device const float* fgp = fgF32 + size_t(b * p.T + t) * (2u * p.H);
    device const bfloat* nw = weightSlab + size_t(p.normElemOff[b]);

    const uint nSteps = min(16u, (p.H + 255u) >> 8u);
    float vReg[16];
    float sq = 0.0f;
    for (uint k = 0u; k < nSteps; ++k) {
        const uint d = tid + (k << 8u);
        if (d < p.H) {
            const float f = qw_ink2_batch_tanh_precise(fgp[d]);
            const float g = qw_ink2_batch_tanh_precise(fgp[p.H + d]);
            const float v = (1.0f - f) * xp[d] + g * mp[d];
            vReg[k] = v;
            sq += v * v;
        }
    }
    for (uint d = tid + 4096u; d < p.H; d += 256u) {
        const float f = qw_ink2_batch_tanh_precise(fgp[d]);
        const float g = qw_ink2_batch_tanh_precise(fgp[p.H + d]);
        const float v = (1.0f - f) * xp[d] + g * mp[d];
        sq += v * v;
    }
    sq = simd_sum(sq);
    threadgroup float sgSq[8];
    if (lane == 0u) sgSq[simd_id] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        const float totalSq = simd_sum((lane < 8u) ? sgSq[lane] : 0.0f);
        if (lane == 0u) sgSq[0] = rsqrt(totalSq / float(p.H) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invRms = sgSq[0];
    device bfloat* dst = outBF16 + size_t(b * p.T + t) * p.H;
    for (uint k = 0u; k < nSteps; ++k) {
        const uint d = tid + (k << 8u);
        if (d < p.H) dst[d] = qw_ink2_bf16((vReg[k] * invRms) * float(nw[d]));
    }
    for (uint d = tid + 4096u; d < p.H; d += 256u) {
        const float f = qw_ink2_batch_tanh_precise(fgp[d]);
        const float g = qw_ink2_batch_tanh_precise(fgp[p.H + d]);
        const float v = (1.0f - f) * xp[d] + g * mp[d];
        dst[d] = qw_ink2_bf16((v * invRms) * float(nw[d]));
    }
}

inline void qw_ink2_stage_bf16_k32(
    threadgroup ushort4* dst,
    device const bfloat* rowPtr,
    bool validRow,
    bool vec16Row,
    uint k0,
    uint colOff,
    uint K) {
    const uint c0 = k0 + colOff;
    if (!validRow || c0 >= K) {
        dst[0] = ushort4(0); dst[1] = ushort4(0);
        dst[2] = ushort4(0); dst[3] = ushort4(0);
        return;
    }
    if (vec16Row && (c0 + 16u <= K)) {
        const uint4 u0 = *(device const uint4*)(rowPtr + c0);
        const uint4 u1 = *(device const uint4*)(rowPtr + c0 + 8u);
        dst[0] = ushort4(ushort(u0.x), ushort(u0.x >> 16u), ushort(u0.y), ushort(u0.y >> 16u));
        dst[1] = ushort4(ushort(u0.z), ushort(u0.z >> 16u), ushort(u0.w), ushort(u0.w >> 16u));
        dst[2] = ushort4(ushort(u1.x), ushort(u1.x >> 16u), ushort(u1.y), ushort(u1.y >> 16u));
        dst[3] = ushort4(ushort(u1.z), ushort(u1.z >> 16u), ushort(u1.w), ushort(u1.w >> 16u));
        return;
    }
    for (uint v = 0u; v < 4u; ++v) {
        ushort4 s = ushort4(0);
        for (uint e = 0u; e < 4u; ++e) {
            const uint c = c0 + v * 4u + e;
            if (c < K) s[e] = as_type<ushort>(rowPtr[c]);
        }
        dst[v] = s;
    }
}

#if !defined(QW_DISABLE_NAX)
kernel void qwg_ink2_ctrl_down_nax_batch(device const bfloat* pairBF16 [[buffer(0)]],
                                          device const bfloat* weightSlab [[buffer(1)]],
                                          device float* hidF32 [[buffer(2)]],
                                          constant QWInk2CtrlBatchParams& p [[buffer(3)]],
                                          uint3 tg [[threadgroup_position_in_grid]],
                                          uint lid [[thread_index_in_threadgroup]],
                                          uint simd_gid [[simdgroup_index_in_threadgroup]]) {
    const uint nBase = tg.x * 64u, mBase = tg.y * 64u, b = tg.z;
    if (b >= p.numBranches || b >= 3u || mBase >= p.T || nBase >= p.rank) return;
    const short sgM = short(simd_gid / 2u), sgN = short(simd_gid % 2u);
    const uint m0 = mBase + uint(sgM * 32), n0 = nBase + uint(sgN * 32);
    const short sgp_sm = short(min(32, int(p.T) - int(m0)));
    const short sgp_sn = short(min(32, int(p.rank) - int(n0)));
    const bool sgActive = (sgp_sm > 0 && sgp_sn > 0);
    const bool alignedM = (sgp_sm == 32), alignedN = (sgp_sn == 32);

    const uint inK = 2u * p.H;
    device const bfloat* wBase = weightSlab + size_t(p.downElemOff[b]);
    device const bfloat* xRowPtr = pairBF16 + (size_t(b * p.T) + min(m0, p.T - 1u)) * inK;
    const uint bRow = nBase + (lid >> 1u);
    const bool validRow = (bRow < p.rank);
    const bool vec16W = ((inK & 7u) == 0u) && ((size_t(wBase) & 15u) == 0u);
    device const bfloat* wRow = wBase + size_t(validRow ? bRow : nBase) * inK;

    threadgroup ushort4 bShU4[64 * 9];
    threadgroup ushort4* dst = bShU4 + size_t(lid >> 1u) * 9u + size_t(lid & 1u) * 4u;
    NAXTile<float, 2, 2> Dtile;
    Dtile.clear();

    STEEL_PRAGMA_NO_UNROLL
    for (uint k0 = 0u; k0 < inK; k0 += 32u) {
        qw_ink2_stage_bf16_k32(dst, wRow, validRow, vec16W, k0, (lid & 1u) * 16u, inK);
        NAXTile<bfloat, 2, 2> Atile;
        if (sgActive) {
            if (alignedM && (k0 + 32u <= inK)) Atile.load(xRowPtr + k0, int(inK));
            else Atile.load_safe(xRowPtr + k0, int(inK), short2(min(32, int(inK - k0)), sgp_sm));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgActive) {
            NAXTile<bfloat, 2, 2> Btile;
            Btile.template load<bfloat, 36, 1>((threadgroup const bfloat*)(bShU4 + size_t(sgN * 32) * 9u));
            tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!sgActive) return;
    device float* hidPtr = hidF32 + size_t(b * p.T + m0) * p.rank + n0;
    if (alignedM && alignedN) Dtile.store(hidPtr, int(p.rank));
    else Dtile.store_safe(hidPtr, int(p.rank), short2(sgp_sn, sgp_sm));
}

kernel void qwg_ink2_ctrl_fg_blend_nax_batch(device const float* xF32 [[buffer(0)]],
                                              device const float* msgF32 [[buffer(1)]],
                                              device const float* hidF32 [[buffer(2)]],
                                              device const bfloat* weightSlab [[buffer(3)]],
                                              device float* outF32 [[buffer(4)]],
                                              constant QWInk2CtrlBatchParams& p [[buffer(5)]],
                                              uint3 tg [[threadgroup_position_in_grid]],
                                              uint lid [[thread_index_in_threadgroup]],
                                              uint simd_gid [[simdgroup_index_in_threadgroup]]) {
    const uint nBase = tg.x * 32u, mBase = tg.y * 64u, b = tg.z;
    if (b >= p.numBranches || b >= 3u || mBase >= p.T || nBase >= p.H) return;
    const uint m0 = mBase + simd_gid * 16u;
    const short sgp_sm = short(min(16, int(p.T) - int(m0)));
    const short sgp_sn = short(min(32, int(p.H) - int(nBase)));
    const bool sgActive = (sgp_sm > 0 && sgp_sn > 0);
    const short2 sc = BaseNAXFrag::get_coord();

    device const bfloat* fWBase = weightSlab + size_t(p.forgetElemOff[b]);
    device const bfloat* gWBase = weightSlab + size_t(p.gainElemOff[b]);
    const uint row = lid >> 1u;
    const uint localRow = row & 31u;
    const uint bRow = nBase + localRow;
    const bool validRow = (bRow < p.H);
    device const bfloat* wSel = (row >= 32u) ? gWBase : fWBase;
    const bool vec16W = ((p.rank & 7u) == 0u) && ((size_t(wSel) & 15u) == 0u);
    device const bfloat* wRow = wSel + size_t(validRow ? bRow : nBase) * p.rank;

    threadgroup ushort4 bShU4[64 * 9];
    threadgroup ushort4* dst = bShU4 + size_t(row) * 9u + size_t(lid & 1u) * 4u;
    device const float* hBase = hidF32 + size_t(b * p.T) * p.rank;
    const bool vec4H = ((p.rank & 3u) == 0u) && ((size_t(hidF32) & 15u) == 0u);

    NAXTile<float, 1, 2> Df, Dg;
    Df.clear();
    Dg.clear();

    STEEL_PRAGMA_NO_UNROLL
    for (uint k0 = 0u; k0 < p.rank; k0 += 32u) {
        qw_ink2_stage_bf16_k32(dst, wRow, validRow, vec16W, k0, (lid & 1u) * 16u, p.rank);
        NAXTile<bfloat, 1, 2> Atile;
        if (sgActive) {
            const_for_loop<0, 2, 1>([&](auto kc) {
                thread auto& af = Atile.template frag_at<0, kc.value>();
                const uint c = k0 + uint(kc.value * 16 + int(sc.x));
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; ++i) {
                    const uint r = m0 + uint(int(sc.y) + int(i) * 8);
                    if (vec4H && r < p.T && (c + 4u <= p.rank)) {
                        const float4 vf = *(device const float4*)(hBase + size_t(r) * p.rank + c);
                        af[i * 4 + 0] = qw_ink2_bf16(vf.x); af[i * 4 + 1] = qw_ink2_bf16(vf.y);
                        af[i * 4 + 2] = qw_ink2_bf16(vf.z); af[i * 4 + 3] = qw_ink2_bf16(vf.w);
                    } else {
                        STEEL_PRAGMA_UNROLL
                        for (short j = 0; j < 4; ++j) {
                            af[i * 4 + j] = (r < p.T && (c + uint(j)) < p.rank)
                                ? qw_ink2_bf16(hBase[size_t(r) * p.rank + c + uint(j)]) : bfloat(0.0f);
                        }
                    }
                }
            });
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgActive) {
            NAXTile<bfloat, 2, 2> Bforget, Bgain;
            Bforget.template load<bfloat, 36, 1>((threadgroup const bfloat*)bShU4);
            tile_matmad_nax(Df, Atile, metal::bool_constant<false>{}, Bforget, metal::bool_constant<true>{});
            Bgain.template load<bfloat, 36, 1>((threadgroup const bfloat*)(bShU4 + 32u * 9u));
            tile_matmad_nax(Dg, Atile, metal::bool_constant<false>{}, Bgain, metal::bool_constant<true>{});
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!sgActive) return;
    const size_t xSlotOff = size_t(p.arIdx[b] * p.T) * p.H;
    const size_t mSlotOff = size_t((p.msgBaseSlot + b) * p.T) * p.H;
    const size_t oSlotOff = size_t(b * p.T) * p.H;
    const bool vec4Out = ((p.H & 3u) == 0u) && ((size_t(xF32) & 15u) == 0u)
        && ((size_t(msgF32) & 15u) == 0u) && ((size_t(outF32) & 15u) == 0u);

    const_for_loop<0, 2, 1>([&](auto nn) {
        thread const auto& ff = Df.template frag_at<0, nn.value>();
        thread const auto& gf = Dg.template frag_at<0, nn.value>();
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; ++i) {
            const uint gr = m0 + uint(int(sc.y) + int(i) * 8);
            if (gr >= p.T) continue;
            const uint gc0 = nBase + uint(nn.value * 16 + int(sc.x));
            const size_t rowH = size_t(gr) * p.H + gc0;
            if (vec4Out && (gc0 + 4u <= p.H)) {
                const float4 xv = *(device const float4*)(xF32 + xSlotOff + rowH);
                const float4 mv = *(device const float4*)(msgF32 + mSlotOff + rowH);
                float4 ov;
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; ++j) {
                    const float f = qw_ink2_batch_tanh_precise(ff[i * 4 + j]);
                    const float g = qw_ink2_batch_tanh_precise(gf[i * 4 + j]);
                    ov[j] = (1.0f - f) * xv[j] + g * mv[j];
                }
                *(device float4*)(outF32 + oSlotOff + rowH) = ov;
            } else {
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; ++j) {
                    if (gc0 + uint(j) < p.H) {
                        const float f = qw_ink2_batch_tanh_precise(ff[i * 4 + j]);
                        const float g = qw_ink2_batch_tanh_precise(gf[i * 4 + j]);
                        outF32[oSlotOff + rowH + uint(j)] =
                            (1.0f - f) * xF32[xSlotOff + rowH + uint(j)] + g * msgF32[mSlotOff + rowH + uint(j)];
                    }
                }
            }
        }
    });
}
#endif // !defined(QW_DISABLE_NAX)
