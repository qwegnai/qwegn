// MQA-shared sparse latent attention on NAX tensor MMA (Apple10+, f16 KV only); same
// contract as qwg_mla_sparse_attn_mma. 1 token x 16 heads per threadgroup; simdgroup g
// owns latent dims [g*128, +128). Gathered K frags double as the V operand (K = V = c),
// so there is no threadgroup staging. Partial S is summed through a double-buffered slab
// (one barrier per 32 keys) in a fixed order. Decode rows use the split-K variant: each
// z-slice attends a 32-aligned slot range into partials merged by the combine kernel.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "qw_mla_params.metalh"
using namespace metal;
using namespace mlx::steel;

#define MLA_NAX_HG   16u
#define MLA_NAX_BK   32u
#define MLA_NAX_RANK 512u
#define MLA_NAX_DSG  128u

typedef BaseNAXFrag::dtype_frag_t<half>  mla_hfrag;
typedef BaseNAXFrag::dtype_frag_t<float> mla_ffrag;

template <bool SPLIT>
METAL_FUNC void qwg_mla_nax_attend(
    device const half* qAbs, device const half* kPool, device const half* vPool,
    device const int* kTable, device const int* vTable, device const int* blockList,
    device half* uLatOut, device float2* partML, constant MLASparseAttnParams& p,
    uint t, uint h0, uint z, uint nSplit, uint tid, uint sg, uint lane,
    threadgroup float* sPart, threadgroup uint* blkIdx
) {
    const uint halfRank = MLA_NAX_RANK / 2u;
    const uint pq = p.pos0 + t;
    const uint nPools = (pq + 1u) >> 2u;
    const uint cnt = min(min(p.topKPools, 1024u), nPools);
    const uint tailStart = nPools << 2u;
    const uint selSlots = cnt << 2u;
    const uint slots = selSlots + (pq + 1u - tailStart);
    const float scale2 = rsqrt(float(p.headDim)) * 1.4426950408889634f;
    uint sBegin = 0u, sEnd = slots;
    if (SPLIT) {
        const uint chunk = (((slots + nSplit - 1u) / nSplit) + MLA_NAX_BK - 1u) & ~(MLA_NAX_BK - 1u);
        sBegin = min(slots, z * chunk);
        sEnd = min(slots, sBegin + chunk);
    }

    if (nPools <= p.topKPools) {
        for (uint i = tid; i < cnt; i += 128u) blkIdx[i] = i << 2u;
    } else {
        device const int* rowList = blockList + (size_t)t * p.listStride;
        for (uint i = tid; i < cnt; i += 128u) blkIdx[i] = uint(rowList[i]) << 2u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short2 sc = BaseNAXFrag::get_coord();
    const short sm = sc.y, sn = sc.x;
    // Dims of this simdgroup live entirely in one half of the split latent (k | v pool).
    device const half* pool = (sg < 2u) ? kPool : vPool;
    device const int* table = (sg < 2u) ? kTable : vTable;
    const uint poolCol = (sg & 1u) * MLA_NAX_DSG + uint(sn);

    mla_hfrag qf[8];
    {
        device const half* q0 = qAbs + ((size_t)t * p.qHeads + h0 + uint(sm)) * MLA_NAX_RANK
                                + sg * MLA_NAX_DSG + uint(sn);
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < 8; d++) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                const half4 v = *(device const half4*)(q0 + (size_t)i * 8u * MLA_NAX_RANK + d * 16);
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; j++) qf[d][i * 4 + j] = v[j];
            }
        }
    }

    mla_ffrag oacc[8];
    STEEL_PRAGMA_UNROLL
    for (short d = 0; d < 8; d++) oacc[d] = mla_ffrag(0.0f);
    float mrow[2] = {-1e30f, -1e30f};
    float lrow[2] = {0.0f, 0.0f};

    uint buf = 0u;
    for (uint s0 = sBegin; s0 < sEnd; s0 += MLA_NAX_BK) {
        // Gather this lane's 4 key rows (keys kf*16 + sm + 8i) x 8 dim frags.
        mla_hfrag kf[2][8];
        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < 2; f++) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                const uint slot = s0 + uint(f * 16 + sm + i * 8);
                device const half* src = nullptr;
                if (slot < sEnd) {
                    const uint j = (slot < selSlots) ? (blkIdx[slot >> 2u] + (slot & 3u))
                                                     : (tailStart + (slot - selSlots));
                    const uint pg = uint(table[j >> 8u]);
                    src = pool + ((size_t)pg * 256u + (j & 255u)) * halfRank + poolCol;
                }
                STEEL_PRAGMA_UNROLL
                for (short d = 0; d < 8; d++) {
                    const half4 v = src ? *(device const half4*)(src + d * 16) : half4(0.0h);
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; j++) kf[f][d][i * 4 + j] = v[j];
                }
            }
        }

        // Partial S[16 heads x 32 keys] over this simdgroup's 128 dims.
        mla_ffrag s[2] = {mla_ffrag(0.0f), mla_ffrag(0.0f)};
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < 8; d++) {
            BaseNAXFrag::mma(s[0], s[1], qf[d], metal::false_type{},
                             kf[0][d], kf[1][d], metal::true_type{});
        }
        threadgroup float* part = sPart + buf * (4u * 16u * 32u);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 16; e++) part[(sg * 16u + uint(e)) * 32u + lane] = s[e >> 3][e & 7];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 16; e++) {
            float v = part[uint(e) * 32u + lane];
            v += part[(16u + uint(e)) * 32u + lane];
            v += part[(32u + uint(e)) * 32u + lane];
            v += part[(48u + uint(e)) * 32u + lane];
            s[e >> 3][e & 7] = v;
        }
        buf ^= 1u;

        // Scale (log2e folded) and mask slots past the row's key count.
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

        mla_hfrag pf[2];
        float psum[2] = {0.0f, 0.0f};
        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < 2; f++) {
            STEEL_PRAGMA_UNROLL
            for (short loc = 0; loc < 8; loc++) {
                const float x = fast::exp2(s[f][loc] - mnew[loc >> 2]);
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

        // O[16 x 128] += P[16 x 32] * c[32 x 128]: the gathered K frags are the V operand.
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
        const float inv = (lrow[i] > 0.0f) ? (1.0f / lrow[i]) : 0.0f;
        const size_t rowIdx = SPLIT ? ((size_t)t * nSplit + z) * p.qHeads + h0 + uint(sm + i * 8)
                                    : (size_t)t * p.qHeads + h0 + uint(sm + i * 8);
        if (SPLIT && sg == 0u && sn == 0) partML[rowIdx] = float2(mrow[i], lrow[i]);
        device half* dst = uLatOut + rowIdx * MLA_NAX_RANK + sg * MLA_NAX_DSG + uint(sn);
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < 8; d++) {
            *(device half4*)(dst + d * 16) = half4(half(oacc[d][i * 4 + 0] * inv), half(oacc[d][i * 4 + 1] * inv),
                                                   half(oacc[d][i * 4 + 2] * inv), half(oacc[d][i * 4 + 3] * inv));
        }
    }
}

// Grid: (T, qHeads / 16, 1), Threadgroup: (128, 1, 1).
kernel void qwg_mla_sparse_attn_nax(
    device const half*           qAbs      [[buffer(0)]],
    device const half*           kPool     [[buffer(1)]],
    device const half*           vPool     [[buffer(2)]],
    device const int*            kTable    [[buffer(3)]],
    device const int*            vTable    [[buffer(4)]],
    device const int*            blockList [[buffer(5)]],
    device half*                 uLatOut   [[buffer(6)]],
    constant MLASparseAttnParams& p        [[buffer(7)]],
    uint2                        tg        [[threadgroup_position_in_grid]],
    uint                         tid       [[thread_index_in_threadgroup]],
    uint                         sg        [[simdgroup_index_in_threadgroup]],
    uint                         lane      [[thread_index_in_simdgroup]]
) {
    if (tg.x >= p.T) return;
    threadgroup float sPart[2 * 4 * 16 * 32];
    threadgroup uint  blkIdx[1024];
    qwg_mla_nax_attend<false>(qAbs, kPool, vPool, kTable, vTable, blockList, uLatOut, nullptr, p,
                              tg.x, tg.y * MLA_NAX_HG, 0u, 1u, tid, sg, lane, sPart, blkIdx);
}

// Split-K partials: partO[(t*nSplit+z)*qHeads+h] (normalized, half) and partML = (log2 max, sum).
// Grid: (T, qHeads / 16, nSplit), Threadgroup: (128, 1, 1).
kernel void qwg_mla_sparse_attn_nax_split(
    device const half*           qAbs      [[buffer(0)]],
    device const half*           kPool     [[buffer(1)]],
    device const half*           vPool     [[buffer(2)]],
    device const int*            kTable    [[buffer(3)]],
    device const int*            vTable    [[buffer(4)]],
    device const int*            blockList [[buffer(5)]],
    device half*                 partO     [[buffer(6)]],
    constant MLASparseAttnParams& p        [[buffer(7)]],
    device float2*               partML    [[buffer(8)]],
    uint3                        tg        [[threadgroup_position_in_grid]],
    uint3                        ntg       [[threadgroups_per_grid]],
    uint                         tid       [[thread_index_in_threadgroup]],
    uint                         sg        [[simdgroup_index_in_threadgroup]],
    uint                         lane      [[thread_index_in_simdgroup]]
) {
    if (tg.x >= p.T) return;
    threadgroup float sPart[2 * 4 * 16 * 32];
    threadgroup uint  blkIdx[1024];
    qwg_mla_nax_attend<true>(qAbs, kPool, vPool, kTable, vTable, blockList, partO, partML, p,
                             tg.x, tg.y * MLA_NAX_HG, tg.z, ntg.z, tid, sg, lane, sPart, blkIdx);
}

// uLat[t,h] = sum_z w_z O_z / sum_z w_z with w_z = l_z * exp2(m_z - max m); 4 dims per thread.
// Grid: (T * qHeads, 1, 1), Threadgroup: (128, 1, 1).
kernel void qwg_mla_sparse_attn_combine(
    device const half*           partO     [[buffer(0)]],
    device const float2*         partML    [[buffer(1)]],
    device half*                 uLatOut   [[buffer(2)]],
    constant MLASparseAttnParams& p        [[buffer(3)]],
    constant uint&               nSplit    [[buffer(4)]],
    uint                         tg        [[threadgroup_position_in_grid]],
    uint                         tid       [[thread_index_in_threadgroup]]
) {
    const uint t = tg / p.qHeads;
    const uint h = tg - t * p.qHeads;
    if (t >= p.T) return;
    const size_t base = (size_t)t * nSplit * p.qHeads + h;
    float mMax = -1e30f;
    for (uint z = 0u; z < nSplit; ++z) {
        const float2 ml = partML[base + (size_t)z * p.qHeads];
        if (ml.y > 0.0f) mMax = max(mMax, ml.x);
    }
    float4 acc = float4(0.0f);
    float wSum = 0.0f;
    for (uint z = 0u; z < nSplit; ++z) {
        const size_t row = base + (size_t)z * p.qHeads;
        const float2 ml = partML[row];
        if (ml.y <= 0.0f) continue;
        const float w = ml.y * fast::exp2(ml.x - mMax);
        wSum += w;
        acc += w * float4(*(device const half4*)(partO + row * MLA_NAX_RANK + tid * 4u));
    }
    const float inv = (wSum > 0.0f) ? (1.0f / wSum) : 0.0f;
    *(device half4*)(uLatOut + ((size_t)t * p.qHeads + h) * MLA_NAX_RANK + tid * 4u) = half4(acc * inv);
}
