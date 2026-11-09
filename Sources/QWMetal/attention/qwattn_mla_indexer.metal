// qwattn_mla_indexer.metal — NoPE MLA Latent Norms, Paged KV Cache Write,
// 4-Token Key-Pool Compression, and Indexer Top-K Pool Selection.
// Split from qwattn_mla_sparse.metal.

#include <metal_stdlib>
#include <metal_simdgroup>
#include "../include/qw_dequant_multi.metalh"
#include "../include/qw_geometry_constants.metalh"
#include "../include/qw_mla_indexer.metalh"
using namespace metal;

struct MLALatentPrepParams {
    uint T;
    uint pos0;
    uint hidden;
    uint qLoraRank;
    uint kvLoraRank;
    uint idxDim;
    uint idxHeads;
    uint ringMask;
    float rmsEps;
    float kNormEps;
    uint kvMode;
};

// 1. Fused Latent Norms (q_a_layernorm + kv_a_layernorm + k_norm), Paged KV Cache Write,
//    and Indexer Gate/Weights Projection.
// Grid: (T, 1, 1), Threadgroup: (256, 1, 1).
kernel void qwg_mla_latent_prep(
    device const half*          mlaLatent      [[buffer(0)]],
    device const half*          ln1            [[buffer(1)]],
    device const half*          qANormW        [[buffer(2)]],
    device const half*          kvANormW       [[buffer(3)]],
    device const half*          kNormW         [[buffer(4)]],
    device const half*          kNormB         [[buffer(5)]],
    device const half*          wGateWeights   [[buffer(6)]],
    device half*                qResidOut      [[buffer(7)]],
    device half*                kPool          [[buffer(8)]],
    device half*                vPool          [[buffer(9)]],
    device const int*           kTable         [[buffer(10)]],
    device const int*           vTable         [[buffer(11)]],
    device half*                kPending       [[buffer(12)]],
    device half*                gatePending    [[buffer(13)]],
    device float*               idxWeightsOut  [[buffer(14)]],
    constant MLALatentPrepParams& p            [[buffer(15)]],
    uint                        t              [[threadgroup_position_in_grid]],
    uint                        tid            [[thread_index_in_threadgroup]],
    uint                        sg             [[simdgroup_index_in_threadgroup]],
    uint                        lane           [[thread_index_in_simdgroup]]
) {
    if (t >= p.T) return;
    threadgroup float sgScratch[8];
    threadgroup half shX[4096];

    const uint pos = p.pos0 + t;
    const uint latentStride = p.qLoraRank + p.kvLoraRank + p.idxDim;
    device const half* rowLat = mlaLatent + (size_t)t * latentStride;

    // (a) q_a_layernorm + fwht32 -> qResidOut [T, qLoraRank]
    float sqQ = 0.0f;
    for (uint i = tid; i < p.qLoraRank; i += 256u) {
        const float v = float(rowLat[i]);
        sqQ = fma(v, v, sqQ);
    }
    sqQ = simd_sum(sqQ);
    if (lane == 0u) sgScratch[sg] = sqQ;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float s = (lane < 8u) ? sgScratch[lane] : 0.0f;
        s = simd_sum(s);
        if (lane == 0u) sgScratch[0] = rsqrt(s / float(p.qLoraRank) + p.rmsEps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invRmsQ = sgScratch[0];

    device half* dstQResid = qResidOut + (size_t)t * p.qLoraRank;
    for (uint base = sg * 32u; base < p.qLoraRank; base += 256u) {
        const uint idx = base + lane;
        const float g = float(qANormW[idx]);
        float v = float(rowLat[idx]) * invRmsQ * g;
        v = qwg_fwht32_lane_f32(v, lane);
        dstQResid[idx] = half(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // (b) kv_a_layernorm + fwht32 -> paged kPool (0..255) and vPool (256..511)
    device const half* kvLat = rowLat + p.qLoraRank;
    float sqKV = 0.0f;
    for (uint i = tid; i < p.kvLoraRank; i += 256u) {
        const float v = float(kvLat[i]);
        sqKV = fma(v, v, sqKV);
    }
    sqKV = simd_sum(sqKV);
    if (lane == 0u) sgScratch[sg] = sqKV;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float s = (lane < 8u) ? sgScratch[lane] : 0.0f;
        s = simd_sum(s);
        if (lane == 0u) sgScratch[0] = rsqrt(s / float(p.kvLoraRank) + p.rmsEps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invRmsKV = sgScratch[0];
    // The quantized-KV scale reductions below reuse sgScratch.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint pageSlot = pos >> 8u;
    const uint pageOff = pos & 255u;
    const uint pageK = uint(kTable[pageSlot]);
    const uint pageV = uint(vTable[pageSlot]);
    const uint halfRank = p.kvLoraRank >> 1u;

    for (uint base = 0u; base < p.kvLoraRank; base += 256u) {
        const uint idx = base + (sg * 32u + lane);
        const float g = float(kvANormW[idx]);
        float v = float(kvLat[idx]) * invRmsKV * g;
        v = qwg_fwht32_lane_f32(v, lane);
        const uint col = idx & (halfRank - 1u);
        device uchar* poolBytes = (base < halfRank) ? (device uchar*)kPool : (device uchar*)vPool;
        const uint pageID = (base < halfRank) ? pageK : pageV;

        if (p.kvMode == 0u) {
            device half* dst = ((device half*)poolBytes) + ((size_t)pageID * 256u + pageOff) * halfRank;
            dst[col] = half(v);
        } else if (p.kvMode == 1u) {
            float maxAbs = simd_max(fabs(v));
            if (lane == 0u) sgScratch[sg] = maxAbs;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0u) {
                float mVal = (lane < 8u) ? sgScratch[lane] : 0.0f;
                mVal = simd_max(mVal);
                if (lane == 0u) {
                    const half scH = half(max(mVal / 127.0f, 1e-8f));
                    sgScratch[0] = float(scH);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const half scH = half(sgScratch[0]);
            const float invSc = 1.0f / float(scH);
            const size_t dataBytes = 256ul * halfRank;
            const size_t pageBytes = dataBytes + 256ul * 2ul;
            device uchar* pageBase = poolBytes + (size_t)pageID * pageBytes;
            device char* qRow = (device char*)(pageBase + (size_t)pageOff * halfRank);
            qRow[col] = char(clamp(rint(v * invSc), -127.0f, 127.0f));
            if (col == 0u) {
                *((device half*)(pageBase + dataBytes + (size_t)pageOff * 2ul)) = scH;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            const uint grp = sg >> 2u;
            float minV = simd_min(v);
            float maxV = simd_max(v);
            if (lane == 0u) {
                shX[sg] = half(minV);
                sgScratch[sg] = maxV;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg < 2u && lane < 4u) {
                const uint sgIdx = sg * 4u + lane;
                float gMin = float(shX[sgIdx]);
                float gMax = sgScratch[sgIdx];
                for (uint off = 2u; off >= 1u; off >>= 1u) {
                    gMin = min(gMin, simd_shuffle_xor(gMin, ushort(off)));
                    gMax = max(gMax, simd_shuffle_xor(gMax, ushort(off)));
                }
                if (lane == 0u) {
                    const half scH = half(max((gMax - gMin) / 15.0f, 1e-8f));
                    const half mnH = half(gMin);
                    sgScratch[sg * 2u + 0u] = float(scH);
                    sgScratch[sg * 2u + 1u] = float(mnH);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const half scH = half(sgScratch[grp * 2u + 0u]);
            const half mnH = half(sgScratch[grp * 2u + 1u]);
            const uint qNib = uint(clamp(rint((v - float(mnH)) / float(scH)), 0.0f, 15.0f));
            shX[col] = half(qNib);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint groupsPerHalf = max(1u, halfRank >> 7u);
            const size_t dataBytes = 128ul * halfRank;
            const size_t pageBytes = dataBytes + 256ul * groupsPerHalf * 4ul;
            device uchar* pageBase = poolBytes + (size_t)pageID * pageBytes;
            device uchar* qRow = pageBase + (size_t)pageOff * (halfRank >> 1u);
            if (tid < (halfRank >> 1u)) {
                const uint q0 = uint(shX[2u * tid]);
                const uint q1 = uint(shX[2u * tid + 1u]);
                qRow[tid] = uchar((q0 & 0x0Fu) | ((q1 & 0x0Fu) << 4u));
            }
            if (tid < groupsPerHalf) {
                const half gSc = half(sgScratch[tid * 2u + 0u]);
                const half gMn = half(sgScratch[tid * 2u + 1u]);
                *((device half2*)(pageBase + dataBytes) + (size_t)pageOff * groupsPerHalf + tid) = half2(gSc, gMn);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // (c) indexer.k_norm (full LayerNorm with mean, var, weight, bias) -> kPending
    device const half* kRaw = rowLat + p.qLoraRank + p.kvLoraRank;
    const float xK = (tid < p.idxDim) ? float(kRaw[tid]) : 0.0f;
    float sumK = simd_sum(xK);
    if (lane == 0u) sgScratch[sg] = sumK;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float s = (lane < 8u) ? sgScratch[lane] : 0.0f;
        s = simd_sum(s);
        if (lane == 0u) sgScratch[0] = s / float(p.idxDim);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float meanK = sgScratch[0];

    const float diffK = (tid < p.idxDim) ? (xK - meanK) : 0.0f;
    float varK = simd_sum(diffK * diffK);
    // meanK must be read from sgScratch[0] by every simdgroup before it is overwritten.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) sgScratch[sg] = varK;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float s = (lane < 8u) ? sgScratch[lane] : 0.0f;
        s = simd_sum(s);
        if (lane == 0u) sgScratch[0] = rsqrt(s / float(p.idxDim) + p.kNormEps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invStdK = sgScratch[0];

    const uint ringSlot = pos & p.ringMask;
    if (tid < p.idxDim) {
        const float kn = diffK * invStdK * float(kNormW[tid]) + float(kNormB[tid]);
        kPending[(size_t)ringSlot * p.idxDim + tid] = half(kn);
    }
}

// 1b. gatePending[ringSlot] [idxDim] + idxWeightsOut[t] [idxHeads] from ln1[t]; one simdgroup
// per output row so the (idxDim + idxHeads) x hidden f16 weight spreads over many threadgroups.
// Grid: (ceil((idxDim + idxHeads) / 8), T), Threadgroup: (256, 1, 1).
kernel void qwg_mla_gate_weights_proj(
    device const half*          ln1            [[buffer(0)]],
    device const half*          wGateWeights   [[buffer(1)]],
    device half*                gatePending    [[buffer(2)]],
    device float*               idxWeightsOut  [[buffer(3)]],
    constant MLALatentPrepParams& p            [[buffer(4)]],
    uint2                       tg             [[threadgroup_position_in_grid]],
    uint                        sg             [[simdgroup_index_in_threadgroup]],
    uint                        lane           [[thread_index_in_simdgroup]]
) {
    const uint t = tg.y;
    const uint r = tg.x * 8u + sg;
    if (t >= p.T || r >= p.idxDim + p.idxHeads) return;
    device const half* xRow = ln1 + (size_t)t * p.hidden;
    device const half* wRow = wGateWeights + (size_t)r * p.hidden;
    float acc = 0.0f;
    for (uint k = lane * 4u; k < p.hidden; k += 128u) {
        const half4 w4 = *(device const half4*)(wRow + k);
        const half4 x4 = *(device const half4*)(xRow + k);
        acc += dot(float4(w4), float4(x4));
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        if (r < p.idxDim) {
            const uint ringSlot = (p.pos0 + t) & p.ringMask;
            gatePending[(size_t)ringSlot * p.idxDim + r] = half(acc);
        } else {
            idxWeightsOut[(size_t)t * p.idxHeads + (r - p.idxDim)] = acc;
        }
    }
}

// 2. 4-Token Key-Pool Compression (learned APE + gate softmax pooling).
// Grid: (numBlocks, 1, 1), Threadgroup: (128, 1, 1).
kernel void qwg_mla_pool_keys(
    device const half* kPending    [[buffer(0)]],
    device const half* gatePending [[buffer(1)]],
    device const half* ape         [[buffer(2)]],
    device half*       pooledKeys  [[buffer(3)]],
    constant uint&     startBlock  [[buffer(4)]],
    constant uint&     numBlocks   [[buffer(5)]],
    constant uint&     idxDim      [[buffer(6)]],
    constant uint&     ringMask    [[buffer(7)]],
    uint               tg          [[threadgroup_position_in_grid]],
    uint               d           [[thread_index_in_threadgroup]]
) {
    if (tg >= numBlocks || d >= idxDim) return;
    const uint b = startBlock + tg;
    const uint r0 = (b * 4u + 0u) & ringMask;
    const uint r1 = (b * 4u + 1u) & ringMask;
    const uint r2 = (b * 4u + 2u) & ringMask;
    const uint r3 = (b * 4u + 3u) & ringMask;

    const float l0 = float(gatePending[(size_t)r0 * idxDim + d]) + float(ape[0u * idxDim + d]);
    const float l1 = float(gatePending[(size_t)r1 * idxDim + d]) + float(ape[1u * idxDim + d]);
    const float l2 = float(gatePending[(size_t)r2 * idxDim + d]) + float(ape[2u * idxDim + d]);
    const float l3 = float(gatePending[(size_t)r3 * idxDim + d]) + float(ape[3u * idxDim + d]);

    const float m = max(max(l0, l1), max(l2, l3));
    const float e0 = exp(l0 - m);
    const float e1 = exp(l1 - m);
    const float e2 = exp(l2 - m);
    const float e3 = exp(l3 - m);
    const float invS = 1.0f / (e0 + e1 + e2 + e3);

    const half p0 = half(e0 * invS);
    const half p1 = half(e1 * invS);
    const half p2 = half(e2 * invS);
    const half p3 = half(e3 * invS);

    const half k0 = kPending[(size_t)r0 * idxDim + d];
    const half k1 = kPending[(size_t)r1 * idxDim + d];
    const half k2 = kPending[(size_t)r2 * idxDim + d];
    const half k3 = kPending[(size_t)r3 * idxDim + d];

    const float pooled = float(p0) * float(k0) + float(p1) * float(k1)
                       + float(p2) * float(k2) + float(p3) * float(k3);
    pooledKeys[(size_t)b * idxDim + d] = half(pooled);
}

// 3. Indexer Pool Scoring & Top-512 Selection.
// Grid: (T, 1, 1), Threadgroup: (256, 1, 1).
kernel void qwg_mla_indexer_score_select(
    device const half*        mlaQFull     [[buffer(0)]],
    device const float*       idxWeights   [[buffer(1)]],
    device const half*        pooledKeys   [[buffer(2)]],
    device float*             scoreScratch [[buffer(3)]],
    device int*               blockList    [[buffer(4)]],
    device int*               blockCount   [[buffer(5)]],
    constant MLAIndexerParams& p           [[buffer(6)]],
    uint                      t            [[threadgroup_position_in_grid]],
    uint                      tid          [[thread_index_in_threadgroup]],
    uint                      sg           [[simdgroup_index_in_threadgroup]],
    uint                      lane         [[thread_index_in_simdgroup]]
) {
    if (t >= p.T) return;
    const uint pq = p.pos0 + t;
    const uint nPools = (pq + 1u) >> 2u;
    device int* dstList = blockList + (size_t)t * p.listStride;

    if (nPools <= p.topKPools) {
        for (uint i = tid; i < nPools; i += 256u) {
            dstList[i] = int(i);
        }
        if (tid == 0u) {
            blockCount[t] = int(nPools);
        }
        return;
    }

    threadgroup half shQIdx[4096];
    threadgroup float shW[32];
    threadgroup atomic_uint hist[256];
    threadgroup uint tgState[4];

    const uint qFullStride = p.qChannels + p.idxHeads * p.idxDim;
    device const half* qIdxRow = mlaQFull + (size_t)t * qFullStride + p.qChannels;
    const uint totalQIdx = p.idxHeads * p.idxDim;
    for (uint i = tid; i < totalQIdx; i += 256u) {
        shQIdx[i] = qIdxRow[i];
    }
    if (tid < p.idxHeads) {
        shW[tid] = idxWeights[(size_t)t * p.idxHeads + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = rsqrt(float(p.idxDim));
    device float* rowScores = scoreScratch + (size_t)t * p.scoreStride;

    // Score each candidate pool by loading pooledKeys[poolIdx] once per simdgroup
    // and dotting against all 32 staged query heads in threadgroup SRAM.
    for (uint poolIdx = sg; poolIdx < nPools; poolIdx += 8u) {
        const float4 kp = float4(*(device const half4*)(pooledKeys + (size_t)poolIdx * 128u + lane * 4u));
        float totalScore = 0.0f;
        for (uint h = 0u; h < p.idxHeads; ++h) {
            const float4 qh = float4(*(threadgroup const half4*)(shQIdx + h * 128u + lane * 4u));
            const float dotVal = simd_sum(dot(qh, kp));
            totalScore = fma(shW[h], max(0.0f, dotVal * scale), totalScore);
        }
        if (lane == 0u) {
            rowScores[poolIdx] = totalScore;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    qwg_mla_select_pools(rowScores, nPools, p.topKPools, dstList, blockCount + t, tid, hist, tgState);
}
