// qwattn_mla_sparse.metal — Matrix-Absorbed Multi-head Latent Attention (NoPE MLA)
// with Key-Pool(4) Sparse Indexer in the rotated H_32 basis (H_32^T = H_32, H_32^2 = I).
// Stores H_32-rotated c_KV in the paged KV cache, absorbs W_UK into q_nope, attends over
// sparse top-K tokens, projects through W_UV, and rotates in-register for o_proj.

#include <metal_stdlib>
#include <metal_simdgroup>
#include "../include/qw_mla_dequant.metalh"
#include "../include/qw_geometry_constants.metalh"
using namespace metal;

METAL_FUNC float qwg_mla_read_kv_elem(device const uchar* poolBytes,
                                      uint pageID,
                                      uint pageOff,
                                      uint halfRank,
                                      uint col,
                                      uint kvMode) {
    if (kvMode == 0u) {
        device const half* hPool = (device const half*)poolBytes;
        return float(hPool[((size_t)pageID * 256u + pageOff) * halfRank + col]);
    } else if (kvMode == 1u) {
        const size_t dataBytes = 256ul * halfRank;
        const size_t pageBytes = dataBytes + 256ul * 2ul;
        device const uchar* pageBase = poolBytes + (size_t)pageID * pageBytes;
        device const char* qRow = (device const char*)(pageBase + (size_t)pageOff * halfRank);
        const half sc = *((device const half*)(pageBase + dataBytes + (size_t)pageOff * 2ul));
        return float(qRow[col]) * float(sc);
    } else {
        const uint groupsPerHalf = max(1u, halfRank >> 7u);
        const size_t dataBytes = 128ul * halfRank;
        const size_t pageBytes = dataBytes + 256ul * groupsPerHalf * 4ul;
        device const uchar* pageBase = poolBytes + (size_t)pageID * pageBytes;
        device const uchar* qRow = pageBase + (size_t)pageOff * (halfRank >> 1u);
        const uint g = col >> 7u;
        const half2 sm = *((device const half2*)(pageBase + dataBytes) + (size_t)pageOff * groupsPerHalf + g);
        const uchar b = qRow[col >> 1u];
        const uint nib = ((col & 1u) == 0u) ? uint(b & 0x0Fu) : uint(b >> 4u);
        return fma(float(nib), float(sm.x), float(sm.y));
    }
}

template <bool SUB2 = false>
METAL_FUNC float qwg_mla_read_weight(device const char* blocks,
                                     uint row,
                                     uint col,
                                     uint K,
                                     uint bits,
                                     uint group,
                                     uint isIncumbent) {
    if (isIncumbent == 1u) {
        device const Block_HQ4_G32* b = ((device const Block_HQ4_G32*)blocks)
            + (size_t)row * (K >> 5u) + (col >> 5u);
        const uint idx = col & 31u;
        const uint w = idx >> 3u;
        const uint sub = idx & 7u;
        const ushort2 raw = as_type<ushort2>(b->qs[w]);
        const ushort hw = ((sub & 1u) == 0u) ? raw.x : raw.y;
        const ushort nib = (hw >> ((sub >> 1u) * 4u)) & 0x000Fu;
        const half fp = as_type<half>(ushort(nib | 0x6400u));
        return float(fma(fp, b->scale, b->bias));
    } else if (isIncumbent == 2u) {
        const uint nGroups = K / group;
        if constexpr (SUB2) {
            const uint bytesPerGroup = qwg_shvq_group_bytes(int(bits), int(group));
            device const char* gp = blocks + ((size_t)row * nGroups + (col / group)) * bytesPerGroup;
            return qwg_shvq_read_elem_t<true>(gp, bits, col % group);
        } else {
            const uint bytesPerGroup = (group * bits) / 8u + 2u;
            device const char* gp = blocks + ((size_t)row * nGroups + (col / group)) * bytesPerGroup;
            return qwg_shvq_read_elem_t<false>(gp, bits, col % group);
        }
    } else {
        const uint nGroups = K / group;
        const uint bytesPerGroup = (group * bits) / 8u + 4u;
        device const char* gp = blocks + ((size_t)row * nGroups + (col / group)) * bytesPerGroup;
        const float scale = float(*(device const half*)gp);
        const float bias = float(*(device const half*)(gp + 2));
        const uint q = qwg_aq_read_bits((device const uchar*)(gp + 4), int(bits), int(col % group));
        return fma(float(q), scale, bias);
    }
}

// 4. Matrix-Absorbed Query Projection: q_abs[t, h, c] = sum_{r=0..255} q_nope[t, h, r] * W_{UK, h}[r, c].
// Grid: (qHeads, (T + 7) / 8, 1), Threadgroup: (512, 1, 1).
template <bool SUB2>
static inline void qwg_mla_absorb_q_impl(
    device const half*       mlaQFull,
    device const char*       kvBBlocks,
    device half*             qAbsOut,
    constant MLAAbsorbParams& p,
    uint3                    tgPos,
    uint                     c,
    threadgroup float (&shQ)[8][256],
    threadgroup ShvqMlaTgLut& tgLut
) {
    const uint h = tgPos.x;
    const uint tBase = tgPos.y * 8u;
    if (h >= p.qHeads || tBase >= p.T) return;
    const uint validM = min(8u, p.T - tBase);
    const uint inc = p.isIncumbent & 0xFFu;

    if (inc == 2u) {
        qw_mla_load_shvq_lut<SUB2>(tgLut, p.bits, c, 512u);
    }
    for (uint idx = c; idx < validM * p.nopeDim; idx += 512u) {
        const uint m = idx / p.nopeDim;
        const uint r = idx - m * p.nopeDim;
        shQ[m][r] = float(mlaQFull[(size_t)(tBase + m) * p.qFullStride + h * p.nopeDim + r]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint headRowBase = h * (p.nopeDim + p.vHeadDim);
    if (p.kvLoraRank == 512u && p.nopeDim >= 32u && p.nopeDim <= 256u && (p.nopeDim & 3u) == 0u
        && p.group >= 32u && (p.group & 31u) == 0u && inc <= 2u) {
        const uint rSlice = c >> 7u;
        const uint quadIdx = c & 127u;
        const uint sliceRows = p.nopeDim >> 2u;
        const uint r0 = rSlice * sliceRows;
        float4 acc4[8] = {
            float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f),
            float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)
        };
        qw_mla_absorb_quad_slice<SUB2>(kvBBlocks, headRowBase, r0, r0 + sliceRows, quadIdx * 4u,
                                       validM, p, shQ, tgLut, acc4);
        threadgroup float4* shPart = (threadgroup float4*)shQ;
        for (uint m = 0u; m < validM; ++m) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            shPart[c] = acc4[m];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (rSlice == 0u) {
                const float4 sum4 = (shPart[quadIdx] + shPart[quadIdx + 128u])
                    + (shPart[quadIdx + 256u] + shPart[quadIdx + 384u]);
                *(device half4*)(qAbsOut + ((size_t)(tBase + m) * p.qHeads + h) * 512u + quadIdx * 4u) = half4(sum4);
            }
        }
        return;
    }

    if (c >= p.kvLoraRank) return;

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint r = 0u; r < p.nopeDim; ++r) {
        const float w = qwg_mla_read_weight<SUB2>(kvBBlocks, headRowBase + r, c,
                                                  p.kvLoraRank, p.bits, p.group, inc);
        #pragma unroll
        for (uint m = 0u; m < 8u; ++m) {
            if (m < validM) {
                acc[m] = fma(shQ[m][r], w, acc[m]);
            }
        }
    }

    for (uint m = 0u; m < validM; ++m) {
        qAbsOut[((size_t)(tBase + m) * p.qHeads + h) * p.kvLoraRank + c] = half(acc[m]);
    }
}

kernel void qwg_mla_absorb_q(
    device const half*       mlaQFull   [[buffer(0)]],
    device const char*       kvBBlocks  [[buffer(1)]],
    device half*             qAbsOut    [[buffer(2)]],
    constant MLAAbsorbParams& p         [[buffer(3)]],
    uint3                    tgPos      [[threadgroup_position_in_grid]],
    uint                     c          [[thread_index_in_threadgroup]]
) {
    alignas(16) threadgroup float shQ[8][256];
    threadgroup ShvqMlaTgLut tgLut;
    qwg_mla_absorb_q_impl<false>(mlaQFull, kvBBlocks, qAbsOut, p, tgPos, c, shQ, tgLut);
}

kernel void qwg_mla_absorb_q_sub2(
    device const half*       mlaQFull   [[buffer(0)]],
    device const char*       kvBBlocks  [[buffer(1)]],
    device half*             qAbsOut    [[buffer(2)]],
    constant MLAAbsorbParams& p         [[buffer(3)]],
    uint3                    tgPos      [[threadgroup_position_in_grid]],
    uint                     c          [[thread_index_in_threadgroup]]
) {
    alignas(16) threadgroup float shQ[8][256];
    threadgroup ShvqMlaTgLut tgLut;
    qwg_mla_absorb_q_impl<true>(mlaQFull, kvBBlocks, qAbsOut, p, tgPos, c, shQ, tgLut);
}

// 5. Sparse Matrix-Absorbed Latent Attention over 4-token pools + causal tail.
// Grid: (qHeads * T, 1, 1), Threadgroup: (256, 1, 1).
kernel void qwg_mla_sparse_attn(
    device const half*           qAbs      [[buffer(0)]],
    device const half*           kPool     [[buffer(1)]],
    device const half*           vPool     [[buffer(2)]],
    device const int*            kTable    [[buffer(3)]],
    device const int*            vTable    [[buffer(4)]],
    device const int*            blockList [[buffer(5)]],
    device half*                 uLatOut   [[buffer(6)]],
    constant MLASparseAttnParams& p        [[buffer(7)]],
    uint                         tg        [[threadgroup_position_in_grid]],
    uint                         d         [[thread_index_in_threadgroup]],
    uint                         sg        [[simdgroup_index_in_threadgroup]],
    uint                         lane      [[thread_index_in_simdgroup]]
) {
    const uint t = tg / p.qHeads;
    const uint h = tg - t * p.qHeads;
    if (t >= p.T) return;

    threadgroup float score[4100];
    threadgroup uint blkIdx[1024];
    threadgroup float sgScratch[8];

    const uint pq = p.pos0 + t;
    const uint nPools = (pq + 1u) >> 2u;
    const uint cnt = min(min(p.topKPools, 1024u), nPools);
    const uint tailStart = nPools << 2u;
    const uint slots = (cnt << 2u) + (pq + 1u - tailStart);
    const float scale = rsqrt(float(p.headDim));

    if (nPools <= p.topKPools) {
        for (uint i = d; i < cnt; i += 256u) blkIdx[i] = i << 2u;
    } else {
        device const int* rowList = blockList + (size_t)t * p.listStride;
        for (uint i = d; i < cnt; i += 256u) blkIdx[i] = uint(rowList[i]) << 2u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint halfRank = p.kvLoraRank >> 1u; // 256
    device const half* qRow = qAbs + ((size_t)t * p.qHeads + h) * p.kvLoraRank;
    float qRegK[8];
    float qRegV[8];
    #pragma unroll
    for (uint c = 0u; c < 8u; ++c) {
        qRegK[c] = float(qRow[c * 32u + lane]);
        qRegV[c] = float(qRow[halfRank + c * 32u + lane]);
    }

    if (p.kvMode == 0u) {
        for (uint b = sg; b < cnt; b += 8u) {
            const uint baseJ = blkIdx[b];
            const uint pageSlot = baseJ >> 8u;
            const uint pageOff = baseJ & 255u;
            const uint pageK = uint(kTable[pageSlot]);
            const uint pageV = uint(vTable[pageSlot]);
            device const half* kBase = kPool + ((size_t)pageK * 256u + pageOff) * halfRank + lane;
            device const half* vBase = vPool + ((size_t)pageV * 256u + pageOff) * halfRank + lane;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            #pragma unroll
            for (uint c = 0u; c < 8u; ++c) {
                const uint colOff = c * 32u;
                const float qk = qRegK[c];
                const float qv = qRegV[c];
                s0 = fma(qk, float(kBase[colOff]), s0);
                s0 = fma(qv, float(vBase[colOff]), s0);
                s1 = fma(qk, float(kBase[halfRank + colOff]), s1);
                s1 = fma(qv, float(vBase[halfRank + colOff]), s1);
                s2 = fma(qk, float(kBase[halfRank * 2u + colOff]), s2);
                s2 = fma(qv, float(vBase[halfRank * 2u + colOff]), s2);
                s3 = fma(qk, float(kBase[halfRank * 3u + colOff]), s3);
                s3 = fma(qv, float(vBase[halfRank * 3u + colOff]), s3);
            }
            const float d0 = simd_sum(s0) * scale;
            const float d1 = simd_sum(s1) * scale;
            const float d2 = simd_sum(s2) * scale;
            const float d3 = simd_sum(s3) * scale;
            if (lane == 0u) {
                *(threadgroup float4*)(score + (b << 2u)) = float4(d0, d1, d2, d3);
            }
        }
        for (uint slot = (cnt << 2u) + sg; slot < slots; slot += 8u) {
            const uint j = tailStart + (slot - (cnt << 2u));
            const uint pageSlot = j >> 8u;
            const uint pageOff = j & 255u;
            const uint pageK = uint(kTable[pageSlot]);
            const uint pageV = uint(vTable[pageSlot]);
            device const half* kRow = kPool + ((size_t)pageK * 256u + pageOff) * halfRank + lane;
            device const half* vRow = vPool + ((size_t)pageV * 256u + pageOff) * halfRank + lane;
            float s0 = 0.0f;
            #pragma unroll
            for (uint c = 0u; c < 8u; ++c) {
                s0 = fma(qRegK[c], float(kRow[c * 32u]), s0);
                s0 = fma(qRegV[c], float(vRow[c * 32u]), s0);
            }
            const float dotVal = simd_sum(s0) * scale;
            if (lane == 0u) {
                score[slot] = dotVal;
            }
        }
    } else if (p.kvMode == 1u) {
        const size_t dataBytes = 256ul * halfRank;
        const size_t pageBytes = dataBytes + 512ul;
        device const uchar* kBytes = (device const uchar*)kPool;
        device const uchar* vBytes = (device const uchar*)vPool;
        for (uint b = sg; b < cnt; b += 8u) {
            const uint baseJ = blkIdx[b];
            const uint pageSlot = baseJ >> 8u;
            const uint pageOff = baseJ & 255u;
            const uint pageK = uint(kTable[pageSlot]);
            const uint pageV = uint(vTable[pageSlot]);
            device const uchar* kPage = kBytes + (size_t)pageK * pageBytes;
            device const uchar* vPage = vBytes + (size_t)pageV * pageBytes;
            device const char* kBase = (device const char*)(kPage + (size_t)pageOff * halfRank + lane);
            device const char* vBase = (device const char*)(vPage + (size_t)pageOff * halfRank + lane);
            const float4 scK = float4(*(device const packed_half4*)(kPage + dataBytes + (size_t)pageOff * 2ul));
            const float4 scV = float4(*(device const packed_half4*)(vPage + dataBytes + (size_t)pageOff * 2ul));
            float4 sK = float4(0.0f);
            float4 sV = float4(0.0f);
            #pragma unroll
            for (uint c = 0u; c < 8u; ++c) {
                const uint colOff = c * 32u;
                const float qk = qRegK[c];
                const float qv = qRegV[c];
                sK.x = fma(qk, float(kBase[colOff]), sK.x);
                sV.x = fma(qv, float(vBase[colOff]), sV.x);
                sK.y = fma(qk, float(kBase[halfRank + colOff]), sK.y);
                sV.y = fma(qv, float(vBase[halfRank + colOff]), sV.y);
                sK.z = fma(qk, float(kBase[halfRank * 2u + colOff]), sK.z);
                sV.z = fma(qv, float(vBase[halfRank * 2u + colOff]), sV.z);
                sK.w = fma(qk, float(kBase[halfRank * 3u + colOff]), sK.w);
                sV.w = fma(qv, float(vBase[halfRank * 3u + colOff]), sV.w);
            }
            const float4 sTot = sK * scK + sV * scV;
            const float d0 = simd_sum(sTot.x) * scale;
            const float d1 = simd_sum(sTot.y) * scale;
            const float d2 = simd_sum(sTot.z) * scale;
            const float d3 = simd_sum(sTot.w) * scale;
            if (lane == 0u) {
                *(threadgroup float4*)(score + (b << 2u)) = float4(d0, d1, d2, d3);
            }
        }
        for (uint slot = (cnt << 2u) + sg; slot < slots; slot += 8u) {
            const uint j = tailStart + (slot - (cnt << 2u));
            const uint pageSlot = j >> 8u;
            const uint pageOff = j & 255u;
            const uint pageK = uint(kTable[pageSlot]);
            const uint pageV = uint(vTable[pageSlot]);
            device const uchar* kPage = kBytes + (size_t)pageK * pageBytes;
            device const uchar* vPage = vBytes + (size_t)pageV * pageBytes;
            device const char* kRow = (device const char*)(kPage + (size_t)pageOff * halfRank + lane);
            device const char* vRow = (device const char*)(vPage + (size_t)pageOff * halfRank + lane);
            const float scK = float(*((device const half*)(kPage + dataBytes + (size_t)pageOff * 2ul)));
            const float scV = float(*((device const half*)(vPage + dataBytes + (size_t)pageOff * 2ul)));
            float sK = 0.0f;
            float sV = 0.0f;
            #pragma unroll
            for (uint c = 0u; c < 8u; ++c) {
                sK = fma(qRegK[c], float(kRow[c * 32u]), sK);
                sV = fma(qRegV[c], float(vRow[c * 32u]), sV);
            }
            const float dotVal = simd_sum(sK * scK + sV * scV) * scale;
            if (lane == 0u) {
                score[slot] = dotVal;
            }
        }
    } else {
        for (uint slot = sg; slot < slots; slot += 8u) {
            const uint j = (slot < (cnt << 2u))
                ? (blkIdx[slot >> 2u] + (slot & 3u))
                : (tailStart + (slot - (cnt << 2u)));
            const uint pageSlot = j >> 8u;
            const uint pageOff = j & 255u;
            const uint pageK = uint(kTable[pageSlot]);
            const uint pageV = uint(vTable[pageSlot]);
            float s0 = 0.0f;
            #pragma unroll
            for (uint c = 0u; c < 8u; ++c) {
                const uint col = c * 32u + lane;
                const float kv = qwg_mla_read_kv_elem((device const uchar*)kPool, pageK, pageOff, halfRank, col, p.kvMode);
                const float vv = qwg_mla_read_kv_elem((device const uchar*)vPool, pageV, pageOff, halfRank, col, p.kvMode);
                s0 = fma(qRegK[c], kv, s0);
                s0 = fma(qRegV[c], vv, s0);
            }
            const float dotVal = simd_sum(s0) * scale;
            if (lane == 0u) {
                score[slot] = dotVal;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localMax = -INFINITY;
    for (uint s = d; s < slots; s += 256u) {
        localMax = max(localMax, score[s]);
    }
    localMax = simd_max(localMax);
    if (lane == 0u) sgScratch[sg] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float vMax = (lane < 8u) ? sgScratch[lane] : -INFINITY;
        vMax = simd_max(vMax);
        if (lane == 0u) sgScratch[0] = vMax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float m = sgScratch[0];

    float localSum = 0.0f;
    for (uint s = d; s < slots; s += 256u) {
        const float w = exp(score[s] - m);
        score[s] = w;
        localSum += w;
    }
    localSum = simd_sum(localSum);
    // Every simdgroup must have read m from sgScratch[0] before it is reused for the sums.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) sgScratch[sg] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float vSum = (lane < 8u) ? sgScratch[lane] : 0.0f;
        vSum = simd_sum(vSum);
        if (lane == 0u) sgScratch[0] = (vSum > 0.0f) ? (1.0f / vSum) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invL = sgScratch[0];

    float accK = 0.0f;
    float accV = 0.0f;
    if (p.kvMode == 0u) {
        for (uint b = 0u; b < cnt; ++b) {
            const uint baseJ = blkIdx[b];
            const uint pageSlot = baseJ >> 8u;
            const uint pageOff = baseJ & 255u;
            const uint pageK = uint(kTable[pageSlot]);
            const uint pageV = uint(vTable[pageSlot]);
            const float4 w4 = *(threadgroup const float4*)(score + (b << 2u));
            device const half* kBase = kPool + ((size_t)pageK * 256u + pageOff) * halfRank + d;
            device const half* vBase = vPool + ((size_t)pageV * 256u + pageOff) * halfRank + d;
            accK = fma(w4.x, float(kBase[0]), accK);
            accK = fma(w4.y, float(kBase[halfRank]), accK);
            accK = fma(w4.z, float(kBase[halfRank * 2u]), accK);
            accK = fma(w4.w, float(kBase[halfRank * 3u]), accK);
            accV = fma(w4.x, float(vBase[0]), accV);
            accV = fma(w4.y, float(vBase[halfRank]), accV);
            accV = fma(w4.z, float(vBase[halfRank * 2u]), accV);
            accV = fma(w4.w, float(vBase[halfRank * 3u]), accV);
        }
        for (uint slot = (cnt << 2u); slot < slots; ++slot) {
            const uint j = tailStart + (slot - (cnt << 2u));
            const uint pageSlot = j >> 8u;
            const uint pageOff = j & 255u;
            const uint pageK = uint(kTable[pageSlot]);
            const uint pageV = uint(vTable[pageSlot]);
            const float w = score[slot];
            accK = fma(w, float(kPool[((size_t)pageK * 256u + pageOff) * halfRank + d]), accK);
            accV = fma(w, float(vPool[((size_t)pageV * 256u + pageOff) * halfRank + d]), accV);
        }
    } else if (p.kvMode == 1u) {
        const size_t dataBytes = 256ul * halfRank;
        const size_t pageBytes = dataBytes + 512ul;
        device const uchar* kBytes = (device const uchar*)kPool;
        device const uchar* vBytes = (device const uchar*)vPool;
        for (uint b = 0u; b < cnt; ++b) {
            const uint baseJ = blkIdx[b];
            const uint pageSlot = baseJ >> 8u;
            const uint pageOff = baseJ & 255u;
            const uint pageK = uint(kTable[pageSlot]);
            const uint pageV = uint(vTable[pageSlot]);
            device const uchar* kPage = kBytes + (size_t)pageK * pageBytes;
            device const uchar* vPage = vBytes + (size_t)pageV * pageBytes;
            const float4 w4 = *(threadgroup const float4*)(score + (b << 2u));
            const float4 wK4 = w4 * float4(*(device const packed_half4*)(kPage + dataBytes + (size_t)pageOff * 2ul));
            const float4 wV4 = w4 * float4(*(device const packed_half4*)(vPage + dataBytes + (size_t)pageOff * 2ul));
            device const char* kBase = (device const char*)(kPage + (size_t)pageOff * halfRank + d);
            device const char* vBase = (device const char*)(vPage + (size_t)pageOff * halfRank + d);
            accK = fma(wK4.x, float(kBase[0]), accK);
            accK = fma(wK4.y, float(kBase[halfRank]), accK);
            accK = fma(wK4.z, float(kBase[halfRank * 2u]), accK);
            accK = fma(wK4.w, float(kBase[halfRank * 3u]), accK);
            accV = fma(wV4.x, float(vBase[0]), accV);
            accV = fma(wV4.y, float(vBase[halfRank]), accV);
            accV = fma(wV4.z, float(vBase[halfRank * 2u]), accV);
            accV = fma(wV4.w, float(vBase[halfRank * 3u]), accV);
        }
        for (uint slot = (cnt << 2u); slot < slots; ++slot) {
            const uint j = tailStart + (slot - (cnt << 2u));
            const uint pageSlot = j >> 8u;
            const uint pageOff = j & 255u;
            const uint pageK = uint(kTable[pageSlot]);
            const uint pageV = uint(vTable[pageSlot]);
            device const uchar* kPage = kBytes + (size_t)pageK * pageBytes;
            device const uchar* vPage = vBytes + (size_t)pageV * pageBytes;
            const float w = score[slot];
            const float wK = w * float(*((device const half*)(kPage + dataBytes + (size_t)pageOff * 2ul)));
            const float wV = w * float(*((device const half*)(vPage + dataBytes + (size_t)pageOff * 2ul)));
            accK = fma(wK, float(((device const char*)kPage)[(size_t)pageOff * halfRank + d]), accK);
            accV = fma(wV, float(((device const char*)vPage)[(size_t)pageOff * halfRank + d]), accV);
        }
    } else {
        for (uint slot = 0u; slot < slots; ++slot) {
            const uint j = (slot < (cnt << 2u))
                ? (blkIdx[slot >> 2u] + (slot & 3u))
                : (tailStart + (slot - (cnt << 2u)));
            const uint pageSlot = j >> 8u;
            const uint pageOff = j & 255u;
            const uint pageK = uint(kTable[pageSlot]);
            const uint pageV = uint(vTable[pageSlot]);
            const float w = score[slot];
            accK = fma(w, qwg_mla_read_kv_elem((device const uchar*)kPool, pageK, pageOff, halfRank, d, p.kvMode), accK);
            accV = fma(w, qwg_mla_read_kv_elem((device const uchar*)vPool, pageV, pageOff, halfRank, d, p.kvMode), accV);
        }
    }

    device half* dstU = uLatOut + ((size_t)t * p.qHeads + h) * p.kvLoraRank;
    dstU[d] = half(accK * invL);
    dstU[halfRank + d] = half(accV * invL);
}

// 6. Value Up-Projection + Fused H_32 Rotation:
//    attOut[t, h, r] = fwht32(sum_{c=0..511} uLat[t, h, c] * W_{UV, h}[r, c]).
// Grid: (qHeads, (T + 7) / 8, 1), Threadgroup: (256, 1, 1).
template <bool SUB2>
static inline void qwg_mla_proj_uv_impl(
    device const half*       uLat,
    device const char*       kvBBlocks,
    device half*             attOut,
    constant MLAAbsorbParams& p,
    uint3                    tgPos,
    uint                     r,
    uint                     lane,
    threadgroup half (&shU)[8][512],
    threadgroup ShvqMlaTgLut& tgLut
) {
    const uint h = tgPos.x;
    const uint tBase = tgPos.y * 8u;
    if (h >= p.qHeads || tBase >= p.T) return;
    const uint validM = min(8u, p.T - tBase);
    const uint inc = p.isIncumbent & 0xFFu;
    const bool skipFwht = (p.isIncumbent & 0x10000u) != 0u;

    if (inc == 2u) {
        qw_mla_load_shvq_lut<SUB2>(tgLut, p.bits, r, 256u);
    }
    for (uint idx = r; idx < validM * p.kvLoraRank; idx += 256u) {
        const uint m = idx / p.kvLoraRank;
        const uint c = idx - m * p.kvLoraRank;
        shU[m][c] = uLat[((size_t)(tBase + m) * p.qHeads + h) * p.kvLoraRank + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (r >= p.vHeadDim) return;

    const uint outStride = p.qHeads * p.vHeadDim;
    if (inc == 1u && p.kvLoraRank == 512u && p.vHeadDim == 256u) {
        const uint sg = r >> 5u;
        const uint subRow = lane >> 4u;
        const uint blkIdx = lane & 15u;
        device const Block_HQ4_G32* headBlocks = ((device const Block_HQ4_G32*)kvBBlocks)
            + (size_t)(h * (p.nopeDim + 256u) + p.nopeDim + sg * 32u) * 16u;
        if (validM == 1u) {
            threadgroup const half* uBase0 = &shU[0][blkIdx * 32u];
            float myOut = 0.0f;
            for (uint pIdx = 0u; pIdx < 16u; ++pIdx) {
                const Block_HQ4_G32 blk = headBlocks[(pIdx * 2u + subRow) * 16u + blkIdx];
                float s = qwg_dot_hq4_block(blk, uBase0);
                s += simd_shuffle_xor(s, ushort(8));
                s += simd_shuffle_xor(s, ushort(4));
                s += simd_shuffle_xor(s, ushort(2));
                s += simd_shuffle_xor(s, ushort(1));
                const float r0 = simd_broadcast(s, ushort(0));
                const float r1 = simd_broadcast(s, ushort(16));
                if (lane == pIdx * 2u)      myOut = r0;
                if (lane == pIdx * 2u + 1u) myOut = r1;
            }
            const float rot = skipFwht ? myOut : qwg_fwht32_lane_f32(myOut, lane);
            attOut[(size_t)tBase * outStride + h * 256u + r] = half(rot);
            return;
        }
        if (validM == 2u) {
            threadgroup const half* uBase0 = &shU[0][blkIdx * 32u];
            threadgroup const half* uBase1 = &shU[1][blkIdx * 32u];
            float myOut0 = 0.0f;
            float myOut1 = 0.0f;
            for (uint pIdx = 0u; pIdx < 16u; ++pIdx) {
                const Block_HQ4_G32 blk = headBlocks[(pIdx * 2u + subRow) * 16u + blkIdx];
                float2 s = float2(qwg_dot_hq4_block(blk, uBase0),
                                  qwg_dot_hq4_block(blk, uBase1));
                s += simd_shuffle_xor(s, ushort(8));
                s += simd_shuffle_xor(s, ushort(4));
                s += simd_shuffle_xor(s, ushort(2));
                s += simd_shuffle_xor(s, ushort(1));
                const float2 r0 = simd_broadcast(s, ushort(0));
                const float2 r1 = simd_broadcast(s, ushort(16));
                if (lane == pIdx * 2u)      { myOut0 = r0.x; myOut1 = r0.y; }
                if (lane == pIdx * 2u + 1u) { myOut0 = r1.x; myOut1 = r1.y; }
            }
            const float rot0 = skipFwht ? myOut0 : qwg_fwht32_lane_f32(myOut0, lane);
            const float rot1 = skipFwht ? myOut1 : qwg_fwht32_lane_f32(myOut1, lane);
            attOut[(size_t)(tBase + 0u) * outStride + h * 256u + r] = half(rot0);
            attOut[(size_t)(tBase + 1u) * outStride + h * 256u + r] = half(rot1);
            return;
        }
        float myOut[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint pIdx = 0u; pIdx < 16u; ++pIdx) {
            const Block_HQ4_G32 blk = headBlocks[(pIdx * 2u + subRow) * 16u + blkIdx];
            const half2 s2 = half2(blk.scale);
            const half2 b2 = half2(blk.bias);
            half2 w01[4], w23[4], w45[4], w67[4];
            #pragma unroll
            for (uint w = 0u; w < 4u; ++w) {
                unpack_hq4_word8(blk.qs[w], s2, b2, w01[w], w23[w], w45[w], w67[w]);
            }
            for (uint m = 0u; m < validM; ++m) {
                threadgroup const half* xg = &shU[m][blkIdx * 32u];
                float s = 0.0f;
                #pragma unroll
                for (uint w = 0u; w < 4u; ++w) {
                    half4 lo, hi;
                    qwg_load_half8(xg + w * 8u, lo, hi);
                    s = fma(float(w01[w].x), float(lo.x), s);
                    s = fma(float(w01[w].y), float(lo.y), s);
                    s = fma(float(w23[w].x), float(lo.z), s);
                    s = fma(float(w23[w].y), float(lo.w), s);
                    s = fma(float(w45[w].x), float(hi.x), s);
                    s = fma(float(w45[w].y), float(hi.y), s);
                    s = fma(float(w67[w].x), float(hi.z), s);
                    s = fma(float(w67[w].y), float(hi.w), s);
                }
                s += simd_shuffle_xor(s, ushort(8));
                s += simd_shuffle_xor(s, ushort(4));
                s += simd_shuffle_xor(s, ushort(2));
                s += simd_shuffle_xor(s, ushort(1));
                const float r0 = simd_broadcast(s, ushort(0));
                const float r1 = simd_broadcast(s, ushort(16));
                if (lane == pIdx * 2u)      myOut[m] = r0;
                if (lane == pIdx * 2u + 1u) myOut[m] = r1;
            }
        }
        for (uint m = 0u; m < validM; ++m) {
            const float rot = skipFwht ? myOut[m] : qwg_fwht32_lane_f32(myOut[m], lane);
            attOut[(size_t)(tBase + m) * outStride + h * 256u + r] = half(rot);
        }
        return;
    } else if (p.kvLoraRank == 512u && p.vHeadDim == 256u && p.group >= 32u && (p.group & 31u) == 0u
               && ((inc == 2u && (SUB2 ? (p.bits == 1u || p.bits == 15u) : (p.bits >= 2u && p.bits <= 4u)))
                   || (!SUB2 && inc == 0u))) {
        const uint sg = r >> 5u;
        const uint subRow = lane >> 4u;
        const uint blkIdx = lane & 15u;
        const uint baseRow = h * (p.nopeDim + 256u) + p.nopeDim + sg * 32u;
        const uint col0 = blkIdx * 32u;
        if (validM == 1u) {
            float4 uLo0[4], uHi0[4];
            threadgroup const half* xg0 = &shU[0][col0];
            #pragma unroll
            for (uint w = 0u; w < 4u; ++w) {
                half4 lo, hi;
                qwg_load_half8(xg0 + w * 8u, lo, hi);
                uLo0[w] = float4(lo);
                uHi0[w] = float4(hi);
            }
            float myOut = 0.0f;
            for (uint pIdx = 0u; pIdx < 16u; ++pIdx) {
                float4 wLo[4], wHi[4];
                qw_mla_unpack_slice32_reg<SUB2>(kvBBlocks, baseRow + pIdx * 2u + subRow, col0, p, tgLut, wLo, wHi);
                float s = 0.0f;
                #pragma unroll
                for (uint w = 0u; w < 4u; ++w) {
                    s += dot(wLo[w], uLo0[w]) + dot(wHi[w], uHi0[w]);
                }
                s += simd_shuffle_xor(s, ushort(8));
                s += simd_shuffle_xor(s, ushort(4));
                s += simd_shuffle_xor(s, ushort(2));
                s += simd_shuffle_xor(s, ushort(1));
                const float r0 = simd_broadcast(s, ushort(0));
                const float r1 = simd_broadcast(s, ushort(16));
                if (lane == pIdx * 2u)      myOut = r0;
                if (lane == pIdx * 2u + 1u) myOut = r1;
            }
            const float rot = skipFwht ? myOut : qwg_fwht32_lane_f32(myOut, lane);
            attOut[(size_t)tBase * outStride + h * 256u + r] = half(rot);
            return;
        }
        float myOut[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint pIdx = 0u; pIdx < 16u; ++pIdx) {
            float4 wLo[4], wHi[4];
            qw_mla_unpack_slice32_reg<SUB2>(kvBBlocks, baseRow + pIdx * 2u + subRow, col0, p, tgLut, wLo, wHi);
            for (uint m = 0u; m < validM; ++m) {
                threadgroup const half* xg = &shU[m][col0];
                float s = 0.0f;
                #pragma unroll
                for (uint w = 0u; w < 4u; ++w) {
                    half4 lo, hi;
                    qwg_load_half8(xg + w * 8u, lo, hi);
                    s += dot(wLo[w], float4(lo)) + dot(wHi[w], float4(hi));
                }
                s += simd_shuffle_xor(s, ushort(8));
                s += simd_shuffle_xor(s, ushort(4));
                s += simd_shuffle_xor(s, ushort(2));
                s += simd_shuffle_xor(s, ushort(1));
                const float r0 = simd_broadcast(s, ushort(0));
                const float r1 = simd_broadcast(s, ushort(16));
                if (lane == pIdx * 2u)      myOut[m] = r0;
                if (lane == pIdx * 2u + 1u) myOut[m] = r1;
            }
        }
        for (uint m = 0u; m < validM; ++m) {
            const float rot = skipFwht ? myOut[m] : qwg_fwht32_lane_f32(myOut[m], lane);
            attOut[(size_t)(tBase + m) * outStride + h * 256u + r] = half(rot);
        }
        return;
    }

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const uint uvRow = h * (p.nopeDim + p.vHeadDim) + p.nopeDim + r;
    for (uint c = 0u; c < p.kvLoraRank; ++c) {
        const float w = qwg_mla_read_weight<SUB2>(kvBBlocks, uvRow, c,
                                                  p.kvLoraRank, p.bits, p.group, inc);
        #pragma unroll
        for (uint m = 0u; m < 8u; ++m) {
            if (m < validM) {
                acc[m] = fma(float(shU[m][c]), w, acc[m]);
            }
        }
    }

    for (uint m = 0u; m < validM; ++m) {
        const float rot = skipFwht ? acc[m] : qwg_fwht32_lane_f32(acc[m], lane);
        attOut[(size_t)(tBase + m) * outStride + h * p.vHeadDim + r] = half(rot);
    }
}

kernel void qwg_mla_proj_uv(
    device const half*       uLat      [[buffer(0)]],
    device const char*       kvBBlocks [[buffer(1)]],
    device half*             attOut    [[buffer(2)]],
    constant MLAAbsorbParams& p        [[buffer(3)]],
    uint3                    tgPos     [[threadgroup_position_in_grid]],
    uint                     r         [[thread_index_in_threadgroup]],
    uint                     lane      [[thread_index_in_simdgroup]]
) {
    threadgroup half shU[8][512];
    threadgroup ShvqMlaTgLut tgLut;
    qwg_mla_proj_uv_impl<false>(uLat, kvBBlocks, attOut, p, tgPos, r, lane, shU, tgLut);
}

kernel void qwg_mla_proj_uv_sub2(
    device const half*       uLat      [[buffer(0)]],
    device const char*       kvBBlocks [[buffer(1)]],
    device half*             attOut    [[buffer(2)]],
    constant MLAAbsorbParams& p        [[buffer(3)]],
    uint3                    tgPos     [[threadgroup_position_in_grid]],
    uint                     r         [[thread_index_in_threadgroup]],
    uint                     lane      [[thread_index_in_simdgroup]]
) {
    threadgroup half shU[8][512];
    threadgroup ShvqMlaTgLut tgLut;
    qwg_mla_proj_uv_impl<true>(uLat, kvBBlocks, attOut, p, tgPos, r, lane, shU, tgLut);
}
