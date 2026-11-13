// qwattn_hca.metal — Hybrid Compressed Attention (HCA) GPU kernels:
// fused input norms + SWA ring + compressor staging, 128:1 channel-wise softmax
// compression (+ optional Weight-SVD LSA key/query prep), Q norm/YaRN RoPE, and
// joint MQA 512D MMA attention (unsplit prefill + Split-K decode + combine).

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "../include/qw_weight_basis.metalh"
using namespace metal;

struct HCAInPrepParams {
    uint T;
    uint pos0;
    uint qLoraRank;
    uint headDim;
    uint nopeDim;
    uint ropeDim;
    uint compressRatio;
    uint ringMask;
    float rmsEps;
    float mscale;
};

struct HCACompressParams {
    uint startBlock;
    uint numBlocks;
    uint compressRatio;
    uint headDim;
    uint nopeDim;
    uint ropeDim;
    uint ringMask;
    uint indexDim;
    float rmsEps;
    float mscale;
    uint hasLSA;
};

struct HCAQPrepParams {
    uint T;
    uint pos0;
    uint qHeads;
    uint headDim;
    uint nopeDim;
    uint ropeDim;
    uint qUpStride;
    uint indexDim;
    float rmsEps;
    float mscale;
};

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

static inline uint qw_hca_row(uint l) { return ((l >> 4) & 1u) * 4u + ((l >> 2) & 1u) * 2u + ((l >> 1) & 1u); }
static inline uint qw_hca_col(uint l) { return ((l >> 3) & 1u) * 4u + (l & 1u) * 2u; }

// 1. Fused q_layernorm (+ H_32) + kv_layernorm (+ GPT-J YaRN RoPE -> swaKV) + compressor staging.
// Grid: (T, 1, 1), Threadgroup: (256, 1, 1).
kernel void qwg_hca_in_prep(
    device const half*     fusedIn       [[buffer(0)]],
    device const half*     qNormW        [[buffer(1)]],
    device const half*     kvNormW       [[buffer(2)]],
    device const float*    ape           [[buffer(3)]],
    device const float*    invFreq       [[buffer(4)]],
    device half*           qrOut         [[buffer(5)]],
    device half*           swaKV         [[buffer(6)]],
    device float*          compKvPend    [[buffer(7)]],
    device float*          compScorePend [[buffer(8)]],
    constant HCAInPrepParams& p          [[buffer(9)]],
    uint                   t             [[threadgroup_position_in_grid]],
    uint                   tid           [[thread_index_in_threadgroup]],
    uint                   sg            [[simdgroup_index_in_threadgroup]],
    uint                   lane          [[thread_index_in_simdgroup]]
) {
    if (t >= p.T) return;
    threadgroup float sgScratch[8];
    threadgroup float shKV[512];

    const uint pos = p.pos0 + t;
    const uint inStride = p.qLoraRank + 3u * p.headDim;
    device const half* rowIn = fusedIn + (size_t)t * inStride;

    float sqQ = 0.0f;
    for (uint i = tid; i < p.qLoraRank; i += 256u) {
        const float v = float(rowIn[i]);
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

    device half* dstQr = qrOut + (size_t)t * p.qLoraRank;
    for (uint base = sg * 32u; base < p.qLoraRank; base += 256u) {
        const uint idx = base + lane;
        const float v = float(rowIn[idx]) * invRmsQ * float(qNormW[idx]);
        dstQr[idx] = half(qwg_fwht32_lane_f32(v, lane));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const half* kvRaw = rowIn + p.qLoraRank;
    float sqKV = 0.0f;
    for (uint i = tid; i < p.headDim; i += 256u) {
        const float v = float(kvRaw[i]);
        sqKV = fma(v, v, sqKV);
    }
    sqKV = simd_sum(sqKV);
    if (lane == 0u) sgScratch[sg] = sqKV;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float s = (lane < 8u) ? sgScratch[lane] : 0.0f;
        s = simd_sum(s);
        if (lane == 0u) sgScratch[0] = rsqrt(s / float(p.headDim) + p.rmsEps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invRmsKV = sgScratch[0];

    for (uint i = tid; i < p.headDim; i += 256u) {
        shKV[i] = float(kvRaw[i]) * invRmsKV * float(kvNormW[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint halfRope = p.ropeDim >> 1u;
    if (tid < halfRope) {
        const uint base = p.nopeDim + 2u * tid;
        const float even = shKV[base], odd = shKV[base + 1u];
        const float ang = float(pos) * invFreq[tid];
        const float c = cos(ang) * p.mscale, s = sin(ang) * p.mscale;
        shKV[base] = even * c - odd * s;
        shKV[base + 1u] = odd * c + even * s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint ringSlot = pos & p.ringMask;
    device half* dstSwa = swaKV + (size_t)ringSlot * p.headDim;
    device float* dstCKv = compKvPend + (size_t)ringSlot * p.headDim;
    device float* dstCSc = compScorePend + (size_t)ringSlot * p.headDim;
    device const half* cKvIn = kvRaw + p.headDim;
    device const half* cScIn = cKvIn + p.headDim;
    device const float* apeRow = ape + (size_t)(pos % p.compressRatio) * p.headDim;

    for (uint i = tid; i < p.headDim; i += 256u) {
        dstSwa[i] = half(shKV[i]);
        dstCKv[i] = float(cKvIn[i]);
        dstCSc[i] = float(cScIn[i]) + apeRow[i];
    }
}

// 2. 128:1 channel-wise softmax compression + RMSNorm + optional Weight-SVD LSA + YaRN RoPE.
// Grid: (numBlocks, 1, 1), Threadgroup: (256, 1, 1).
kernel void qwg_hca_compress_blocks(
    device const float*       compKvPend    [[buffer(0)]],
    device const float*       compScorePend [[buffer(1)]],
    device const half*        compNormW     [[buffer(2)]],
    device const float*       invFreq       [[buffer(3)]],
    device const half*        wkIdx         [[buffer(4)]],
    device half*              compKV        [[buffer(5)]],
    device half*              pooledKeys    [[buffer(6)]],
    constant HCACompressParams& p           [[buffer(7)]],
    uint                      tg            [[threadgroup_position_in_grid]],
    uint                      tid           [[thread_index_in_threadgroup]],
    uint                      sg            [[simdgroup_index_in_threadgroup]],
    uint                      lane          [[thread_index_in_simdgroup]]
) {
    if (tg >= p.numBlocks) return;
    threadgroup float sgScratch[8];
    threadgroup float shC[512];
    threadgroup float shLsa[64];

    const uint m = p.startBlock + tg;
    const uint basePos = m * p.compressRatio;
    float sq = 0.0f;

    for (uint d = tid; d < p.headDim; d += 256u) {
        float mx = -INFINITY;
        for (uint r = 0u; r < p.compressRatio; ++r) {
            const uint slot = (basePos + r) & p.ringMask;
            mx = max(mx, compScorePend[(size_t)slot * p.headDim + d]);
        }
        float sumE = 0.0f, sumKV = 0.0f;
        for (uint r = 0u; r < p.compressRatio; ++r) {
            const uint slot = (basePos + r) & p.ringMask;
            const float e = exp(compScorePend[(size_t)slot * p.headDim + d] - mx);
            sumE += e;
            sumKV = fma(e, compKvPend[(size_t)slot * p.headDim + d], sumKV);
        }
        const float c = (sumE > 0.0f) ? (sumKV / sumE) : 0.0f;
        shC[d] = c;
        sq = fma(c, c, sq);
    }
    sq = simd_sum(sq);
    if (lane == 0u) sgScratch[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float s = (lane < 8u) ? sgScratch[lane] : 0.0f;
        s = simd_sum(s);
        if (lane == 0u) sgScratch[0] = rsqrt(s / float(p.headDim) + p.rmsEps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invRms = sgScratch[0];

    for (uint d = tid; d < p.headDim; d += 256u) {
        shC[d] *= invRms * float(compNormW[d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (p.hasLSA != 0u && p.indexDim > 0u && p.indexDim <= 64u) {
        for (uint i = sg; i < p.indexDim; i += 8u) {
            device const half* wRow = wkIdx + (size_t)i * p.nopeDim;
            float acc = 0.0f;
            for (uint d = lane; d < p.nopeDim; d += 32u) {
                acc = fma(float(wRow[d]), shC[d], acc);
            }
            acc = simd_sum(acc);
            if (lane == 0u) shLsa[i] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float nsq = 0.0f;
            for (uint i = lane; i < p.indexDim; i += 32u) nsq = fma(shLsa[i], shLsa[i], nsq);
            nsq = simd_sum(nsq);
            const float invN = rsqrt(max(nsq, 1e-12f));
            for (uint i = lane; i < p.indexDim; i += 32u) {
                pooledKeys[(size_t)m * p.indexDim + i] = half(shLsa[i] * invN);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint halfRope = p.ropeDim >> 1u;
    if (tid < halfRope) {
        const uint base = p.nopeDim + 2u * tid;
        const float even = shC[base], odd = shC[base + 1u];
        const float ang = float(basePos) * invFreq[tid];
        const float c = cos(ang) * p.mscale, s = sin(ang) * p.mscale;
        shC[base] = even * c - odd * s;
        shC[base + 1u] = odd * c + even * s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device half* dstC = compKV + (size_t)m * p.headDim;
    for (uint d = tid; d < p.headDim; d += 256u) {
        dstC[d] = half(shC[d]);
    }
}

// 3a. Per-head weightless Q RMSNorm + GPT-J YaRN RoPE in-place on qUpGate.
// Grid: (qHeads, T, 1), Threadgroup: (256, 1, 1).
kernel void qwg_hca_q_prep(
    device half*           qUpGate [[buffer(0)]],
    device const float*    invFreq [[buffer(1)]],
    constant HCAQPrepParams& p     [[buffer(2)]],
    uint2                  tg      [[threadgroup_position_in_grid]],
    uint                   tid     [[thread_index_in_threadgroup]],
    uint                   sg      [[simdgroup_index_in_threadgroup]],
    uint                   lane    [[thread_index_in_simdgroup]]
) {
    const uint h = tg.x, t = tg.y;
    if (h >= p.qHeads || t >= p.T) return;
    threadgroup float sgScratch[8];

    device half* qh = qUpGate + (size_t)t * p.qUpStride + (size_t)h * p.headDim;
    float sq = 0.0f;
    for (uint d = tid; d < p.headDim; d += 256u) {
        const float v = float(qh[d]);
        sq = fma(v, v, sq);
    }
    sq = simd_sum(sq);
    if (lane == 0u) sgScratch[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float s = (lane < 8u) ? sgScratch[lane] : 0.0f;
        s = simd_sum(s);
        if (lane == 0u) sgScratch[0] = rsqrt(s / float(p.headDim) + p.rmsEps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invRms = sgScratch[0];

    for (uint d = tid; d < p.nopeDim; d += 256u) {
        qh[d] = half(float(qh[d]) * invRms);
    }
    const uint halfRope = p.ropeDim >> 1u;
    if (tid < halfRope) {
        const uint base = p.nopeDim + 2u * tid;
        const float even = float(qh[base]) * invRms;
        const float odd = float(qh[base + 1u]) * invRms;
        const float ang = float(p.pos0 + t) * invFreq[tid];
        const float c = cos(ang) * p.mscale, s = sin(ang) * p.mscale;
        qh[base] = half(even * c - odd * s);
        qh[base + 1u] = half(odd * c + even * s);
    }
}

// 3b. Weight-SVD LSA query landmark projection across MQA NoPE heads (>262K tokens).
// Grid: (T, 1, 1), Threadgroup: (256, 1, 1).
kernel void qwg_hca_lsa_q_prep(
    device const half*     qUpGate [[buffer(0)]],
    device const half*     wqIdx   [[buffer(1)]],
    device half*           qIdxOut [[buffer(2)]],
    constant HCAQPrepParams& p     [[buffer(3)]],
    uint                   t       [[threadgroup_position_in_grid]],
    uint                   tid     [[thread_index_in_threadgroup]],
    uint                   sg      [[simdgroup_index_in_threadgroup]],
    uint                   lane    [[thread_index_in_simdgroup]]
) {
    if (t >= p.T) return;
    threadgroup float shQBar[512];
    threadgroup float shLsa[64];

    device const half* qRow = qUpGate + (size_t)t * p.qUpStride;
    const float invH = 1.0f / float(p.qHeads);
    for (uint d = tid; d < p.nopeDim; d += 256u) {
        float sum = 0.0f;
        for (uint h = 0u; h < p.qHeads; ++h) sum += float(qRow[(size_t)h * p.headDim + d]);
        shQBar[d] = sum * invH;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = sg; i < p.indexDim; i += 8u) {
        device const half* wRow = wqIdx + (size_t)i * p.nopeDim;
        float acc = 0.0f;
        for (uint d = lane; d < p.nopeDim; d += 32u) acc = fma(float(wRow[d]), shQBar[d], acc);
        acc = simd_sum(acc);
        if (lane == 0u) shLsa[i] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        float nsq = 0.0f;
        for (uint i = lane; i < p.indexDim; i += 32u) nsq = fma(shLsa[i], shLsa[i], nsq);
        nsq = simd_sum(nsq);
        const float invN = rsqrt(max(nsq, 1e-12f));
        for (uint i = lane; i < p.indexDim; i += 32u) {
            qIdxOut[(size_t)t * p.indexDim + i] = half(shLsa[i] * invN);
        }
    }
}

// 4. Joint MQA 512D attention over compressed summaries + SWA ring (unsplit prefill & Split-K decode).
// Grid: (T, qHeads / 16, nSplit), Threadgroup: (128, 1, 1).
#define HCA_HG   16u
#define HCA_TK   16u
#define HCA_DIM  512u
#define HCA_KS   520u

kernel void qwg_hca_attn_mma(
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
    uint                   tid        [[thread_index_in_threadgroup]],
    uint                   sg         [[simdgroup_index_in_threadgroup]],
    uint                   lane       [[thread_index_in_simdgroup]]
) {
    const uint t = tg.x;
    const uint h0 = tg.y * HCA_HG;
    const uint z = tg.z;
    const uint nSplit = ntg.z;
    if (t >= p.T || h0 >= p.qHeads) return;

    threadgroup half  ks[HCA_TK * HCA_KS];
    threadgroup float sbuf[HCA_HG * HCA_TK];
    threadgroup half  pbuf[HCA_HG * HCA_TK];
    threadgroup float corrH[HCA_HG];
    threadgroup float mH[HCA_HG];
    threadgroup float lH[HCA_HG];

    const uint pos = p.pos0 + t;
    const uint totalComp = (pos + 1u) / p.compressRatio;
    const bool sparseComp = (p.useBlockList != 0u) && (totalComp > p.compBudget);
    const uint nCompSlots = sparseComp ? uint(blockCount[t]) : totalComp;
    const uint swaStart = (pos + 1u > p.slidingWindow) ? (pos + 1u - p.slidingWindow) : 0u;
    const uint nSwaSlots = (pos + 1u) - swaStart;
    const uint slots = nCompSlots + nSwaSlots;
    const float scale = rsqrt(float(HCA_DIM));

    uint sBegin = 0u, sEnd = slots;
    if (nSplit > 1u) {
        const uint chunk = (((slots + nSplit - 1u) / nSplit) + HCA_TK - 1u) & ~(HCA_TK - 1u);
        sBegin = min(slots, z * chunk);
        sEnd = min(slots, sBegin + chunk);
    }

    if (tid < HCA_HG) {
        if (nSplit == 1u && p.hasSink != 0u) {
            const float sk = attnSink[h0 + tid];
            mH[tid] = (sk > -1e29f) ? sk : -INFINITY;
            lH[tid] = (sk > -1e29f) ? 1.0f : 0.0f;
        } else {
            mH[tid] = -INFINITY;
            lH[tid] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint rl = qw_hca_row(lane), cl = qw_hca_col(lane);
    const uint sHb = sg & 1u, sTb = sg >> 1u;
    device const half* qBase = qUpGate + (size_t)t * p.qUpStride + (size_t)(h0 + sHb * 8u) * HCA_DIM;
    device const int* rowList = blockList + (size_t)t * p.listStride;

    simdgroup_float8x8 oacc[2][16];
    #pragma unroll
    for (uint a = 0u; a < 2u; ++a) {
        #pragma unroll
        for (uint b = 0u; b < 16u; ++b) oacc[a][b] = simdgroup_float8x8(0.0f);
    }

    for (uint s0 = sBegin; s0 < sEnd; s0 += HCA_TK) {
        for (uint i = tid; i < HCA_TK * (HCA_DIM / 4u); i += 128u) {
            const uint r = i >> 7u;
            const uint c4 = (i & 127u) * 4u;
            const uint slot = s0 + r;
            half4 v = half4(0.0h);
            if (slot < sEnd) {
                if (slot < nCompSlots) {
                    const uint cIdx = sparseComp ? uint(rowList[slot]) : slot;
                    v = *(device const half4*)(compKV + (size_t)cIdx * HCA_DIM + c4);
                } else {
                    const uint swaPos = swaStart + (slot - nCompSlots);
                    v = *(device const half4*)(swaKV + (size_t)(swaPos & p.ringMask) * HCA_DIM + c4);
                }
            }
            *(threadgroup half4*)(ks + r * HCA_KS + c4) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            simdgroup_float8x8 sacc = simdgroup_float8x8(0.0f);
            for (uint d = 0u; d < HCA_DIM / 8u; ++d) {
                simdgroup_half8x8 qf, kf;
                simdgroup_load(qf, qBase + d * 8u, HCA_DIM);
                simdgroup_load(kf, ks + sTb * 8u * HCA_KS + d * 8u, HCA_KS, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sacc, qf, kf, sacc);
            }
            auto e = sacc.thread_elements();
            const uint hr = sHb * 8u + rl;
            #pragma unroll
            for (uint i = 0u; i < 2u; ++i) {
                const uint col = sTb * 8u + cl + i;
                sbuf[hr * HCA_TK + col] = (s0 + col < sEnd) ? e[i] * scale : -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            const uint hr = tid >> 3u, c0 = (tid & 7u) * 2u;
            const float a = sbuf[hr * HCA_TK + c0];
            const float b = sbuf[hr * HCA_TK + c0 + 1u];
            float mx = max(a, b);
            mx = max(mx, simd_shuffle_xor(mx, 1u));
            mx = max(mx, simd_shuffle_xor(mx, 2u));
            mx = max(mx, simd_shuffle_xor(mx, 4u));
            const float mOld = mH[hr];
            const float mNew = max(mOld, mx);
            const float pa = (mNew == -INFINITY) ? 0.0f : exp(a - mNew);
            const float pb = (mNew == -INFINITY) ? 0.0f : exp(b - mNew);
            pbuf[hr * HCA_TK + c0] = half(pa);
            pbuf[hr * HCA_TK + c0 + 1u] = half(pb);
            float sm = pa + pb;
            sm += simd_shuffle_xor(sm, 1u);
            sm += simd_shuffle_xor(sm, 2u);
            sm += simd_shuffle_xor(sm, 4u);
            const float corr = (mOld == -INFINITY) ? 0.0f : exp(mOld - mNew);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if ((tid & 7u) == 0u) {
                corrH[hr] = corr;
                mH[hr] = mNew;
                lH[hr] = lH[hr] * corr + sm;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            const float c0 = corrH[rl], c1 = corrH[8u + rl];
            simdgroup_half8x8 pf[2][2];
            #pragma unroll
            for (uint a = 0u; a < 2u; ++a) {
                #pragma unroll
                for (uint k = 0u; k < 2u; ++k)
                    simdgroup_load(pf[a][k], pbuf + a * 8u * HCA_TK + k * 8u, HCA_TK);
            }
            #pragma unroll
            for (uint b = 0u; b < 16u; ++b) {
                const uint col = sg * 128u + b * 8u;
                simdgroup_half8x8 kf0, kf1;
                simdgroup_load(kf0, ks + col, HCA_KS);
                simdgroup_load(kf1, ks + 8u * HCA_KS + col, HCA_KS);
                auto e0 = oacc[0][b].thread_elements();
                e0[0] *= c0; e0[1] *= c0;
                oacc[0][b].thread_elements() = e0;
                auto e1 = oacc[1][b].thread_elements();
                e1[0] *= c1; e1[1] *= c1;
                oacc[1][b].thread_elements() = e1;
                simdgroup_multiply_accumulate(oacc[0][b], pf[0][0], kf0, oacc[0][b]);
                simdgroup_multiply_accumulate(oacc[0][b], pf[0][1], kf1, oacc[0][b]);
                simdgroup_multiply_accumulate(oacc[1][b], pf[1][0], kf0, oacc[1][b]);
                simdgroup_multiply_accumulate(oacc[1][b], pf[1][1], kf1, oacc[1][b]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint a = 0u; a < 2u; ++a) {
        const uint hr = a * 8u + rl;
        const uint h = h0 + hr;
        const float l = lH[hr];
        const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        const size_t rowIdx = (nSplit > 1u) ? (((size_t)t * nSplit + z) * p.qHeads + h)
                                            : ((size_t)t * p.qHeads + h);
        if (nSplit > 1u && sg == 0u && cl == 0u) partML[rowIdx] = float2(mH[hr], l);
        device half* dst = outBuf + rowIdx * HCA_DIM;
        #pragma unroll
        for (uint b = 0u; b < 16u; ++b) {
            auto e = oacc[a][b].thread_elements();
            const uint col = sg * 128u + b * 8u + cl;
            *(device half2*)(dst + col) = half2(half(e[0] * inv), half(e[1] * inv));
        }
    }
}

// 4b. Split-K combine for HCA MQA attention with per-head sink.
// Grid: (T * qHeads, 1, 1), Threadgroup: (128, 1, 1).
kernel void qwg_hca_attn_combine(
    device const half*     partO    [[buffer(0)]],
    device const float2*   partML   [[buffer(1)]],
    device const float*    attnSink [[buffer(2)]],
    device half*           attOut   [[buffer(3)]],
    constant HCAAttnParams& p       [[buffer(4)]],
    constant uint&         nSplit   [[buffer(5)]],
    uint                   tg       [[threadgroup_position_in_grid]],
    uint                   tid      [[thread_index_in_threadgroup]]
) {
    const uint t = tg / p.qHeads;
    const uint h = tg - t * p.qHeads;
    if (t >= p.T) return;
    const size_t base = (size_t)t * nSplit * p.qHeads + h;
    float sink = -INFINITY;
    bool validSink = false;
    if (p.hasSink != 0u) {
        const float sk = attnSink[h];
        if (sk > -1e29f) { sink = sk; validSink = true; }
    }
    float mMax = sink;
    for (uint z = 0u; z < nSplit; ++z) {
        const float2 ml = partML[base + (size_t)z * p.qHeads];
        if (ml.y > 0.0f) mMax = max(mMax, ml.x);
    }
    float4 acc = float4(0.0f);
    float wSum = (validSink && mMax > -1e29f) ? exp(sink - mMax) : 0.0f;
    for (uint z = 0u; z < nSplit; ++z) {
        const size_t row = base + (size_t)z * p.qHeads;
        const float2 ml = partML[row];
        if (ml.y <= 0.0f) continue;
        const float w = ml.y * exp(ml.x - mMax);
        wSum += w;
        acc += w * float4(*(device const half4*)(partO + row * HCA_DIM + tid * 4u));
    }
    const float inv = (wSum > 0.0f) ? (1.0f / wSum) : 0.0f;
    *(device half4*)(attOut + ((size_t)t * p.qHeads + h) * HCA_DIM + tid * 4u) = half4(acc * inv);
}
