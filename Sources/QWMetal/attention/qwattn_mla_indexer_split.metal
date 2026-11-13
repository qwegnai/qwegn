// qwattn_mla_indexer_split.metal — MLA indexer scoring parallel over pool tiles
// (simdgroup MMA, 32H/64H) plus a per-row top-K select with optional trailing window.
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
// [g*8, +8) on 8x8 MMA in 32-head chunks; identity rows (pools <= topK + window) skip.
// Grid: (ceil(maxPools / 64), T, 1), Threadgroup: (256, 1, 1). Supports 32 or 64 heads.
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
    const uint idxHeads = qw_mla_idx_heads(p);
    const uint poolShift = qw_mla_idx_pool_shift(p);
    const uint winBlocks = qw_mla_idx_window_blocks(p);
    const uint totalPools = (p.pos0 + t + 1u) >> poolShift;
    const uint denseLim = p.topKPools + winBlocks;
    if (totalPools <= denseLim) return;
    const uint midPools = totalPools - winBlocks;
    const uint pool0 = tg.x * MLA_IDX_POOLS;
    if (pool0 >= midPools) return;

    threadgroup half  shQ[MLA_IDX_HEADS * MLA_IDX_QS];
    threadgroup float shW[64];
    threadgroup float shS[8u * MLA_IDX_HEADS * 8u];

    device const half* qRow = mlaQFull + (size_t)t * (p.qChannels + idxHeads * p.idxDim) + p.qChannels;
    if (tid < idxHeads) shW[tid] = idxWeights[(size_t)t * idxHeads + tid];

    const uint pBase = pool0 + sg * 8u;
    const bool sgActive = (pBase < midPools);
    const float scale = rsqrt(float(MLA_IDX_DIM));
    const uint pc = lane & 7u, h0 = (lane >> 3u) * 8u;
    const uint hChunks = idxHeads >> 5u;
    float v = 0.0f;

    for (uint hc = 0u; hc < hChunks; ++hc) {
        const uint hOff = hc * MLA_IDX_HEADS;
        for (uint i = tid; i < MLA_IDX_HEADS * MLA_IDX_DIM / 4u; i += 256u) {
            const uint h = i / (MLA_IDX_DIM / 4u), c = (i % (MLA_IDX_DIM / 4u)) * 4u;
            *(threadgroup half4*)(shQ + h * MLA_IDX_QS + c) = *(device const half4*)(qRow + (hOff + h) * MLA_IDX_DIM + c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgActive) {
            simdgroup_float8x8 acc[4];
            #pragma unroll
            for (uint a = 0u; a < 4u; ++a) acc[a] = simdgroup_float8x8(0.0f);
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

            float part = 0.0f;
            #pragma unroll
            for (uint h = 0u; h < 8u; ++h) {
                part = fma(shW[hOff + h0 + h], max(0.0f, s[(h0 + h) * 8u + pc] * scale), part);
            }
            part += simd_shuffle_xor(part, 8u);
            part += simd_shuffle_xor(part, 16u);
            v += part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgActive && lane < 8u && pBase + lane < midPools) {
        scoreScratch[(size_t)t * p.scoreStride + pBase + lane] = v;
    }
}

// Per-row top-K over scores from qwg_mla_indexer_score_mma, plus trailing window append.
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
    const uint poolShift = qw_mla_idx_pool_shift(p);
    const uint winBlocks = qw_mla_idx_window_blocks(p);
    const uint totalPools = (p.pos0 + t + 1u) >> poolShift;
    const uint denseLim = p.topKPools + winBlocks;
    device int* dstList = blockList + (size_t)t * p.listStride;
    if (totalPools <= denseLim) {
        for (uint i = tid; i < totalPools; i += 256u) dstList[i] = int(i);
        if (tid == 0u) blockCount[t] = int(totalPools);
        return;
    }
    const uint midPools = totalPools - winBlocks;
    threadgroup atomic_uint hist[256];
    threadgroup uint tgState[4];
    qwg_mla_select_pools(scoreScratch + (size_t)t * p.scoreStride, midPools, p.topKPools,
                         dstList, blockCount + t, tid, hist, tgState);
    if (winBlocks > 0u) {
        threadgroup_barrier(mem_flags::mem_device);
        for (uint w = tid; w < winBlocks; w += 256u) dstList[p.topKPools + w] = int(midPools + w);
        if (tid == 0u) blockCount[t] = int(p.topKPools + winBlocks);
    }
}
