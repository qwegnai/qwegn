// qwattn_mla_indexer_split.metal — MLA indexer scoring parallel over pool tiles
// (simdgroup MMA) plus a per-row top-K select. Replaces the fused one-threadgroup-per-row
// score+select, whose serial scoring dominates decode at long context.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "../include/qw_mla_indexer.metalh"
using namespace metal;

#define MLA_IDX_POOLS 64u
#define MLA_IDX_HEADS 32u
#define MLA_IDX_DIM   128u
#define MLA_IDX_QS    (MLA_IDX_DIM + 8u)

// score[t,pool] = sum_h w[t,h] relu(q[t,h].k[pool]/sqrt(dim)); simdgroup g scores pools
// [g*8, +8) as S[32 heads x 8 pools] on 8x8 MMA; identity rows (pools <= top-K) skip.
// Grid: (ceil(maxPools / 64), T, 1), Threadgroup: (256, 1, 1). Requires 32 heads x 128 dims.
kernel void qwg_mla_indexer_score_mma(
    device const half*        mlaQFull     [[buffer(0)]],
    device const float*       idxWeights   [[buffer(1)]],
    device const half*        pooledKeys   [[buffer(2)]],
    device float*             scoreScratch [[buffer(3)]],
    constant MLAIndexerParams& p           [[buffer(6)]],
    uint2                     tg           [[threadgroup_position_in_grid]],
    uint                      tid          [[thread_index_in_threadgroup]],
    uint                      sg           [[simdgroup_index_in_threadgroup]],
    uint                      lane         [[thread_index_in_simdgroup]]
) {
    const uint t = tg.y;
    if (t >= p.T) return;
    const uint nPools = (p.pos0 + t + 1u) >> 2u;
    const uint pool0 = tg.x * MLA_IDX_POOLS;
    if (nPools <= p.topKPools || pool0 >= nPools) return;

    threadgroup half  shQ[MLA_IDX_HEADS * MLA_IDX_QS];
    threadgroup float shW[MLA_IDX_HEADS];
    threadgroup float shS[8u * MLA_IDX_HEADS * 8u];

    device const half* qRow = mlaQFull + (size_t)t * (p.qChannels + p.idxHeads * p.idxDim) + p.qChannels;
    for (uint i = tid; i < MLA_IDX_HEADS * MLA_IDX_DIM / 4u; i += 256u) {
        const uint h = i / (MLA_IDX_DIM / 4u), c = (i % (MLA_IDX_DIM / 4u)) * 4u;
        *(threadgroup half4*)(shQ + h * MLA_IDX_QS + c) = *(device const half4*)(qRow + h * MLA_IDX_DIM + c);
    }
    if (tid < MLA_IDX_HEADS) shW[tid] = idxWeights[(size_t)t * p.idxHeads + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint pBase = pool0 + sg * 8u;
    if (pBase >= nPools) return;

    simdgroup_float8x8 acc[4];
    #pragma unroll
    for (uint a = 0u; a < 4u; ++a) acc[a] = simdgroup_float8x8(0.0f);
    // pooledKeys rows are padded to a multiple of 8, so the 8-pool tile never reads OOB.
    device const half* kBase = pooledKeys + (size_t)pBase * MLA_IDX_DIM;
    for (uint d = 0u; d < MLA_IDX_DIM / 8u; ++d) {
        simdgroup_half8x8 kf;
        simdgroup_load(kf, kBase + d * 8u, MLA_IDX_DIM, ulong2(0, 0), true);
        #pragma unroll
        for (uint a = 0u; a < 4u; ++a) {
            simdgroup_half8x8 qf;
            simdgroup_load(qf, shQ + a * 8u * MLA_IDX_QS + d * 8u, MLA_IDX_QS);
            simdgroup_multiply_accumulate(acc[a], qf, kf, acc[a]);
        }
    }
    threadgroup float* s = shS + sg * (MLA_IDX_HEADS * 8u);
    #pragma unroll
    for (uint a = 0u; a < 4u; ++a) simdgroup_store(acc[a], s + a * 64u, 8u);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // lane -> pool (lane & 7), head octet (lane >> 3); reduce octets with xor 8, 16.
    const float scale = rsqrt(float(MLA_IDX_DIM));
    const uint pc = lane & 7u, h0 = (lane >> 3u) * 8u;
    float v = 0.0f;
    #pragma unroll
    for (uint h = 0u; h < 8u; ++h) v = fma(shW[h0 + h], max(0.0f, s[(h0 + h) * 8u + pc] * scale), v);
    v += simd_shuffle_xor(v, 8u);
    v += simd_shuffle_xor(v, 16u);
    if (lane < 8u && pBase + lane < nPools) {
        scoreScratch[(size_t)t * p.scoreStride + pBase + lane] = v;
    }
}

// Per-row top-K over scores from qwg_mla_indexer_score_mma.
// Grid: (T, 1, 1), Threadgroup: (256, 1, 1).
kernel void qwg_mla_indexer_select(
    device const float*       scoreScratch [[buffer(3)]],
    device int*               blockList    [[buffer(4)]],
    device int*               blockCount   [[buffer(5)]],
    constant MLAIndexerParams& p           [[buffer(6)]],
    uint                      t            [[threadgroup_position_in_grid]],
    uint                      tid          [[thread_index_in_threadgroup]]
) {
    if (t >= p.T) return;
    const uint nPools = (p.pos0 + t + 1u) >> 2u;
    device int* dstList = blockList + (size_t)t * p.listStride;
    if (nPools <= p.topKPools) {
        for (uint i = tid; i < nPools; i += 256u) dstList[i] = int(i);
        if (tid == 0u) blockCount[t] = int(nPools);
        return;
    }
    threadgroup atomic_uint hist[256];
    threadgroup uint tgState[4];
    qwg_mla_select_pools(scoreScratch + (size_t)t * p.scoreStride, nPools, p.topKPools,
                         dstList, blockCount + t, tid, hist, tgState);
}
