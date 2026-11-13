// qwattn_hca_nax.metal — Joint MQA 512D HCA attention on Apple10+ NAX tensor MMA.
// 1 token x 32 heads (2 x 16-head groups x 4 dim simdgroups) per 256-thread TG;
// supports unsplit prefill with learned sinks, Split-K decode, and sparse LSA.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
using namespace metal;
using namespace mlx::steel;

#define HCA_NAX_BK  32u
#define HCA_NAX_DIM 512u
#define HCA_NAX_DSG 128u

struct HCAAttnParams {
    uint T;
    uint pos0;
    uint qHeads;
    uint headDim;
    uint qUpStride;
    uint compressRatio;
    uint slidingWindow;
    uint ringMask;
    uint compBudget;
    uint listStride;
    uint useBlockList;
    uint hasSink;
};

typedef BaseNAXFrag::dtype_frag_t<half>  hca_hfrag;
typedef BaseNAXFrag::dtype_frag_t<float> hca_ffrag;

// Grid: (T, (qHeads + 31) / 32, nSplit), Threadgroup: (256, 1, 1).
kernel void qwg_hca_attn_nax(
    device const half*     qUpGate    [[buffer(0)]],
    device const half*     compKV     [[buffer(1)]],
    device const half*     swaKV      [[buffer(2)]],
    device const int*      blockList  [[buffer(3)]],
    device const int*      blockCount [[buffer(4)]],
    device const float*    attnSink   [[buffer(5)]],
    device half*           outBuf     [[buffer(6)]],
    device float2*         partML     [[buffer(7)]],
    constant HCAAttnParams& p         [[buffer(8)]],
    uint3                  tg         [[threadgroup_position_in_grid]],
    uint3                  ntg        [[threadgroups_per_grid]],
    uint                   sg         [[simdgroup_index_in_threadgroup]],
    uint                   lane       [[thread_index_in_simdgroup]]
) {
    const uint t = tg.x;
    const uint hGroup0 = tg.y * 32u;
    const uint z = tg.z;
    const uint nSplit = ntg.z;
    if (t >= p.T || hGroup0 >= p.qHeads) return;

    const uint sg_h = sg >> 2u;
    const uint sg_d = sg & 3u;
    const uint h0 = hGroup0 + sg_h * 16u;
    threadgroup float sPart[2u * 4u * 16u * 32u];
    threadgroup float* part = sPart + sg_h * (4u * 16u * 32u);

    const uint pos = p.pos0 + t;
    const uint totalComp = (pos + 1u) / p.compressRatio;
    const bool sparseComp = (p.useBlockList != 0u) && (totalComp > p.compBudget);
    const uint nCompSlots = sparseComp ? uint(blockCount[t]) : totalComp;
    const uint swaStart = (pos + 1u > p.slidingWindow) ? (pos + 1u - p.slidingWindow) : 0u;
    const uint nSwaSlots = (pos + 1u) - swaStart;
    const uint slots = nCompSlots + nSwaSlots;
    const float scale2 = rsqrt(float(HCA_NAX_DIM)) * 1.4426950408889634f;

    uint sBegin = 0u, sEnd = slots;
    if (nSplit > 1u) {
        const uint chunk = (((slots + nSplit - 1u) / nSplit) + HCA_NAX_BK - 1u) & ~(HCA_NAX_BK - 1u);
        sBegin = min(slots, z * chunk);
        sEnd = min(slots, sBegin + chunk);
    }

    const short2 sc = BaseNAXFrag::get_coord();
    const short sm = sc.y, sn = sc.x;
    const uint dimCol = sg_d * HCA_NAX_DSG + uint(sn);

    hca_hfrag qf[8];
    {
        device const half* q0 = qUpGate + (size_t)t * p.qUpStride
            + (size_t)(h0 + uint(sm)) * HCA_NAX_DIM + dimCol;
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < 8; d++) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                const bool validH = (h0 + uint(sm + i * 8)) < p.qHeads;
                const half4 v = validH ? *(device const half4*)(q0 + (size_t)i * 8u * HCA_NAX_DIM + d * 16)
                                       : half4(0.0h);
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; j++) qf[d][i * 4 + j] = v[j];
            }
        }
    }

    hca_ffrag oacc[8];
    STEEL_PRAGMA_UNROLL
    for (short d = 0; d < 8; d++) oacc[d] = hca_ffrag(0.0f);
    float mrow[2] = {-1e30f, -1e30f};
    float lrow[2] = {0.0f, 0.0f};
    if (nSplit == 1u && p.hasSink != 0u) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            const uint h = h0 + uint(sm + i * 8);
            if (h < p.qHeads) {
                const float sk = attnSink[h];
                if (sk > -1e29f) {
                    mrow[i] = sk * 1.4426950408889634f;
                    lrow[i] = 1.0f;
                }
            }
        }
    }

    device const int* rowList = blockList + (size_t)t * p.listStride;
    for (uint s0 = sBegin; s0 < sEnd; s0 += HCA_NAX_BK) {
        hca_hfrag kf[2][8];
        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < 2; f++) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                const uint slot = s0 + uint(f * 16 + sm + i * 8);
                device const half* src = nullptr;
                if (slot < sEnd) {
                    if (slot < nCompSlots) {
                        const uint cIdx = sparseComp ? uint(rowList[slot]) : slot;
                        src = compKV + (size_t)cIdx * HCA_NAX_DIM + dimCol;
                    } else {
                        const uint swaPos = swaStart + (slot - nCompSlots);
                        src = swaKV + (size_t)(swaPos & p.ringMask) * HCA_NAX_DIM + dimCol;
                    }
                }
                STEEL_PRAGMA_UNROLL
                for (short d = 0; d < 8; d++) {
                    const half4 v = src ? *(device const half4*)(src + d * 16) : half4(0.0h);
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; j++) kf[f][d][i * 4 + j] = v[j];
                }
            }
        }

        hca_ffrag s[2] = {hca_ffrag(0.0f), hca_ffrag(0.0f)};
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < 8; d++) {
            BaseNAXFrag::mma(s[0], s[1], qf[d], metal::false_type{},
                             kf[0][d], kf[1][d], metal::true_type{});
        }
        if (s0 > sBegin) threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 16; e++) part[(sg_d * 16u + uint(e)) * 32u + lane] = s[e >> 3][e & 7];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 16; e++) {
            float v = part[uint(e) * 32u + lane];
            v += part[(16u + uint(e)) * 32u + lane];
            v += part[(32u + uint(e)) * 32u + lane];
            v += part[(48u + uint(e)) * 32u + lane];
            s[e >> 3][e & 7] = v;
        }

        float mnew[2] = {mrow[0], mrow[1]};
        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < 2; f++) {
            STEEL_PRAGMA_UNROLL
            for (short loc = 0; loc < 8; loc++) {
                const uint slot = s0 + uint(f * 16 + sn + (loc & 3));
                const float x = (slot < sEnd) ? s[f][loc] * scale2 : -1e30f;
                s[f][loc] = x;
                mnew[loc >> 2] = max(mnew[loc >> 2], x);
            }
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            mnew[i] = max(mnew[i], simd_shuffle_xor(mnew[i], ushort(1)));
            mnew[i] = max(mnew[i], simd_shuffle_xor(mnew[i], ushort(8)));
        }

        hca_hfrag pf[2];
        float psum[2] = {0.0f, 0.0f};
        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < 2; f++) {
            STEEL_PRAGMA_UNROLL
            for (short loc = 0; loc < 8; loc++) {
                const float m = mnew[loc >> 2];
                const float x = (m > -1e29f) ? fast::exp2(s[f][loc] - m) : 0.0f;
                pf[f][loc] = half(x);
                psum[loc >> 2] += x;
            }
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            psum[i] += simd_shuffle_xor(psum[i], ushort(1));
            psum[i] += simd_shuffle_xor(psum[i], ushort(8));
            const bool grew = (mnew[i] > mrow[i]) && (mrow[i] > -1e29f);
            const float corr = grew ? fast::exp2(mrow[i] - mnew[i]) : 1.0f;
            lrow[i] = lrow[i] * corr + psum[i];
            if (grew) {
                STEEL_PRAGMA_UNROLL
                for (short d = 0; d < 8; d++) {
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; j++) oacc[d][i * 4 + j] *= corr;
                }
            }
            mrow[i] = mnew[i];
        }

        // Gathered K frags are reused directly as the V operand (K = V).
        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < 2; f++) {
            STEEL_PRAGMA_UNROLL
            for (short pr = 0; pr < 4; pr++) {
                BaseNAXFrag::mma(oacc[2 * pr], oacc[2 * pr + 1], pf[f], metal::false_type{},
                                 kf[f][2 * pr], kf[f][2 * pr + 1], metal::false_type{});
            }
        }
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
        const uint h = h0 + uint(sm + i * 8);
        if (h >= p.qHeads) continue;
        const float l = lrow[i];
        const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        const size_t rowIdx = (nSplit > 1u) ? (((size_t)t * nSplit + z) * p.qHeads + h)
                                            : ((size_t)t * p.qHeads + h);
        if (nSplit > 1u && sg_d == 0u && sn == 0) {
            partML[rowIdx] = float2((l > 0.0f) ? (mrow[i] * 0.6931471805599453f) : -INFINITY, l);
        }
        device half* dst = outBuf + rowIdx * HCA_NAX_DIM + dimCol;
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < 8; d++) {
            *(device half4*)(dst + d * 16) = half4(
                half(oacc[d][i * 4 + 0] * inv), half(oacc[d][i * 4 + 1] * inv),
                half(oacc[d][i * 4 + 2] * inv), half(oacc[d][i * 4 + 3] * inv));
        }
    }
}
