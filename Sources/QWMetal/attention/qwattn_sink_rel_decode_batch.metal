// qwattn_sink_rel_decode_batch.metal — KV-shared batched/tree Split-KV GQA decode kernels (T=2..16).
#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_attn_sink_rel.metalh"
#include "inkling2/qw_ink2_common.metalh"
using namespace metal;

METAL_FUNC void qw_attn_batch_score_tok(
    uint dist, uint gCount, bool mayRel, float qkScale,
    float4 kf, float4 vf, device const bfloat* relProj,
    constant QWSinkRelAttnParams& p, uint lane,
    const thread float4* q4, const thread float* rLane,
    thread float* m, thread float* l, thread float4* acc
) {
    #pragma clang fp contract(off) reassociate(off)
    const bool inRel = mayRel && (dist < p.relExtent);
    const float relW = (inRel && lane < p.dRel)
        ? float(relProj[size_t(lane) * p.relExtent + dist]) : 0.0f;
    #pragma unroll
    for (uint g = 0u; g < 4u; ++g) {
        if (g < gCount) {
            float score = simd_sum(qwg_sink_rel_dot4(q4[g], kf)) * qkScale;
            if (inRel) score += simd_sum(rLane[g] * relW);
            const float mNew = max(m[g], score);
            const float a = (m[g] > -1e29f) ? exp(m[g] - mNew) : 0.0f;
            const float bExp = exp(score - mNew);
            l[g] = l[g] * a + bExp;
            acc[g] = acc[g] * a + bExp * vf;
            m[g] = mNew;
        }
    }
}

template <uint MODE, uint MAX_U = 2u>
METAL_FUNC void qwg_attn_decode_sink_rel_gqa_batch_body(
    device const bfloat* qBF16, device const bfloat* rBF16, device const bfloat* relProj,
    device const bfloat* sinkBF16, device const bfloat* gateBF16,
    device const uchar* kCache, device const uchar* vCache,
    device const uchar* kMeta, device const uchar* vMeta,
    device bfloat* outBF16, device float* partO, device float2* partML,
    constant QWSinkRelAttnParams& p, constant QWSinkRelTreeMeta& tree,
    uint3 tg, uint sgi, uint lane,
    threadgroup half4 (*kSh)[33], threadgroup half4 (*vSh)[33]
) {
    #pragma clang fp contract(off) reassociate(off)
    const uint gqa = p.nH / p.nKV;
    const uint gChunks = (gqa + 3u) >> 2u;
    const uint sp = tg.x;
    const uint kvh = tg.y / gChunks;
    const uint g0 = (tg.y - kvh * gChunks) * 4u;
    const uint tGrpBase = (MAX_U == 1u) ? 0u : (tg.z * 16u);
    if (sp >= p.splits || kvh >= p.nKV || g0 >= gqa || tGrpBase >= p.T) return;

    const uint gCount = min(4u, gqa - g0);
    const uint headBase = kvh * gqa + g0;
    const uint dBase = lane * 4u;
    const bool activeD = (dBase < p.headDim);
    const float qkScale = rsqrt(float(p.headDim));
    const bool fastT8 = (MAX_U == 1u) || (p.T <= 8u && tGrpBase == 0u);
    const uint uEnd = fastT8 ? 1u : MAX_U;

    // Partition prefix keys by fixed blocks when splitSpan > 0 so T=1 and T>=2 match bit-for-bit.
    const uint prefixEnd = min(p.posOffset, p.kvLen);
    uint blockStart = 0u, blockEnd = 0u, kStart = 0u, kEnd = 0u;
    if (p.splitSpan > 0u) {
        const uint globalMin = (p.sliding > 0u && p.posOffset + 1u > p.sliding) ? (p.posOffset + 1u - p.sliding) : 0u;
        const uint firstBlock = min(globalMin, p.kvLen) / p.splitSpan;
        blockStart = (firstBlock + sp) * p.splitSpan;
        blockEnd = blockStart + p.splitSpan;
        kStart = max(blockStart, min(globalMin, prefixEnd));
        kEnd = min(blockEnd, prefixEnd);
    } else {
        uint prefixMinKey = 0xFFFFFFFFu;
        const uint tGrpEnd = min(p.T, tGrpBase + ((MAX_U == 1u) ? 8u : 16u));
        for (uint tk = tGrpBase; tk < tGrpEnd; ++tk) {
            const uint qp = (tree.isTree != 0u && tk < 16u) ? (p.posOffset + tree.depths[tk]) : (p.posOffset + tk);
            const uint mk = (p.sliding > 0u && qp + 1u > p.sliding) ? (qp + 1u - p.sliding) : 0u;
            prefixMinKey = min(prefixMinKey, mk);
        }
        if (prefixMinKey > prefixEnd) prefixMinKey = prefixEnd;
        const uint totalKeys = prefixEnd - prefixMinKey;
        const uint chunk = (totalKeys > 0u) ? ((totalKeys + p.splits - 1u) / p.splits) : 0u;
        kStart = prefixMinKey + sp * chunk;
        kEnd = min(prefixEnd, kStart + chunk);
    }

    const uint t0 = tGrpBase + sgi * MAX_U;
    bool tokValid[MAX_U];
    uint tokIdx[MAX_U], qPos[MAX_U], tokMin[MAX_U];
    float4 q4[MAX_U][4], acc[MAX_U][4];
    float rLane[MAX_U][4], m[MAX_U][4], l[MAX_U][4];

    #pragma unroll
    for (uint u = 0u; u < MAX_U; ++u) {
        const uint tok = fastT8 ? sgi : (t0 + u);
        tokIdx[u] = tok;
        tokValid[u] = (u < uEnd) && (tok < p.T) && (gCount > 0u);
        qPos[u] = (tokValid[u] && tree.isTree != 0u && tok < 16u)
            ? (p.posOffset + tree.depths[tok]) : (p.posOffset + tok);
        tokMin[u] = (p.sliding > 0u && qPos[u] + 1u > p.sliding) ? (qPos[u] + 1u - p.sliding) : 0u;
        const uint kvEndU = min(qPos[u] + 1u, p.kvLen);
        const bool hasSinkU = (p.splitSpan > 0u)
            ? ((p.hasSink != 0u) && (tokMin[u] >= blockStart) && (tokMin[u] < blockEnd) && (tokMin[u] < kvEndU))
            : ((sp == 0u) && (p.hasSink != 0u));
        #pragma unroll
        for (uint g = 0u; g < 4u; ++g) {
            if (tokValid[u] && g < gCount) {
                const uint h = headBase + g;
                q4[u][g] = activeD
                    ? float4(*(device const bfloat4*)(qBF16 + (size_t(tok) * p.nH + h) * p.headDim + dBase))
                    : float4(0.0f);
                rLane[u][g] = (p.hasRelBias != 0u && lane < p.dRel)
                    ? float(rBF16[(size_t(tok) * p.nH + h) * p.dRel + lane]) : 0.0f;
                m[u][g] = hasSinkU ? float(sinkBF16[size_t(tok) * p.nH + h]) : -1e30f;
                l[u][g] = hasSinkU ? 1.0f : 0.0f;
            } else {
                q4[u][g] = float4(0.0f); rLane[u][g] = 0.0f; m[u][g] = -1e30f; l[u][g] = 0.0f;
            }
            acc[u][g] = float4(0.0f);
        }
    }

    const bool mayRel = (p.hasRelBias != 0u) && (p.dRel > 0u) && (p.relExtent > 0u);
    for (uint j0 = kStart; j0 < kEnd; j0 += 16u) {
        const uint cnt = min(16u, kEnd - j0);
        #pragma unroll
        for (uint r = 0u; r < 2u; ++r) {
            const uint row = r * 8u + sgi;
            if (row < cnt && activeD) {
                const uint slot = (p.sliding > 0u) ? ((j0 + row) % p.ringCap) : (j0 + row);
                kSh[row][lane] = qwg_sink_rel_read_kv4<MODE>(lane, slot, kvh, p.nKV, p.headDim, kCache, kMeta);
                vSh[row][lane] = qwg_sink_rel_read_kv4<MODE>(lane, slot, kvh, p.nKV, p.headDim, vCache, vMeta);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint b = 0u; b < cnt; ++b) {
            const uint j = j0 + b;
            const float4 kf = activeD ? float4(kSh[b][lane]) : float4(0.0f);
            const float4 vf = activeD ? float4(vSh[b][lane]) : float4(0.0f);
            #pragma unroll
            for (uint u = 0u; u < MAX_U; ++u) {
                if (u >= uEnd) break;
                if (!tokValid[u] || j < tokMin[u]) continue;
                const uint dist = qPos[u] - j;
                qw_attn_batch_score_tok(
                    dist, gCount, mayRel, qkScale, kf, vf, relProj, p, lane,
                    q4[u], rLane[u], m[u], l[u], acc[u]);
            }
        }
        if (j0 + 16u < kEnd) threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint batchEnd = min(p.kvLen, p.posOffset + p.T);
    const bool enterBatch = (p.splitSpan > 0u)
        ? (p.posOffset < batchEnd && batchEnd > blockStart && p.posOffset < blockEnd)
        : (sp == 0u && p.posOffset < batchEnd);
    if (enterBatch) {
        if (kStart < kEnd) threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j0 = p.posOffset; j0 < batchEnd; j0 += 16u) {
            const uint cnt = min(16u, batchEnd - j0);
            #pragma unroll
            for (uint r = 0u; r < 2u; ++r) {
                const uint row = r * 8u + sgi;
                if (row < cnt && activeD) {
                    const uint slot = (p.sliding > 0u) ? ((j0 + row) % p.ringCap) : (j0 + row);
                    kSh[row][lane] = qwg_sink_rel_read_kv4<MODE>(lane, slot, kvh, p.nKV, p.headDim, kCache, kMeta);
                    vSh[row][lane] = qwg_sink_rel_read_kv4<MODE>(lane, slot, kvh, p.nKV, p.headDim, vCache, vMeta);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint b = 0u; b < cnt; ++b) {
                const uint j = j0 + b;
                const uint kRel = j - p.posOffset;
                const uint kLogical = (tree.isTree != 0u && kRel < 16u) ? (p.posOffset + tree.depths[kRel]) : j;
                if (p.splitSpan > 0u && (kLogical < blockStart || kLogical >= blockEnd)) continue;
                const float4 kf = activeD ? float4(kSh[b][lane]) : float4(0.0f);
                const float4 vf = activeD ? float4(vSh[b][lane]) : float4(0.0f);
                #pragma unroll
                for (uint u = 0u; u < MAX_U; ++u) {
                    if (u >= uEnd) break;
                    if (!tokValid[u]) continue;
                    const uint tok = tokIdx[u];
                    bool vis = false;
                    uint dist = 0u;
                    if (tree.isTree == 0u) {
                        vis = (j >= tokMin[u]) && (j <= qPos[u]);
                        if (vis) dist = qPos[u] - j;
                    } else {
                        vis = (tok < 16u && kRel < 16u) && ((tree.ancMask[tok] & (1u << kRel)) != 0u);
                        if (vis) {
                            dist = tree.depths[tok] - tree.depths[kRel];
                            vis = (p.sliding == 0u || dist < p.sliding);
                        }
                    }
                    if (!vis) continue;
                    qw_attn_batch_score_tok(
                        dist, gCount, mayRel, qkScale, kf, vf, relProj, p, lane,
                        q4[u], rLane[u], m[u], l[u], acc[u]);
                }
            }
            if (j0 + 16u < batchEnd) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    #pragma unroll
    for (uint u = 0u; u < MAX_U; ++u) {
        if (u >= uEnd) break;
        if (!tokValid[u]) continue;
        const uint tok = tokIdx[u];
        if (p.splits == 1u) {
            if (!activeD) continue;
            #pragma unroll
            for (uint g = 0u; g < 4u; ++g) {
                if (g < gCount) {
                    const uint h = headBase + g;
                    const size_t oBase = (size_t(tok) * p.nH + h) * p.headDim + dBase;
                    float4 res = acc[u][g] / max(l[u][g], 1e-30f);
                    if (p.hasOutGate != 0u) {
                        const float4 g4 = float4(*(device const bfloat4*)(gateBF16 + oBase));
                        res.x *= qw_ink2_sigmoid(g4.x); res.y *= qw_ink2_sigmoid(g4.y);
                        res.z *= qw_ink2_sigmoid(g4.z); res.w *= qw_ink2_sigmoid(g4.w);
                    }
                    *(device bfloat4*)(outBF16 + oBase) = bfloat4(
                        qw_ink2_bf16(res.x), qw_ink2_bf16(res.y), qw_ink2_bf16(res.z), qw_ink2_bf16(res.w)
                    );
                }
            }
        } else {
            #pragma unroll
            for (uint g = 0u; g < 4u; ++g) {
                if (g < gCount) {
                    const uint h = headBase + g;
                    const size_t mlIdx = (size_t(tok) * p.nH + h) * p.splits + sp;
                    if (activeD) *(device float4*)(partO + mlIdx * p.headDim + dBase) = acc[u][g];
                    if (lane == 0u) partML[mlIdx] = float2(m[u][g], l[u][g]);
                }
            }
        }
    }
}

template <uint MODE>
METAL_FUNC void qwg_attn_decode_sink_rel_gqa_batch4_body(
    device const bfloat* qBF16, device const bfloat* rBF16, device const bfloat* relProj,
    device const bfloat* sinkBF16, device const bfloat* gateBF16,
    device const uchar* kCache, device const uchar* vCache,
    device const uchar* kMeta, device const uchar* vMeta,
    device bfloat* outBF16, device float* partO, device float2* partML,
    constant QWSinkRelAttnParams& p, constant QWSinkRelTreeMeta& tree,
    uint3 tg, uint sgi, uint lane,
    threadgroup half4 (*kSh)[33], threadgroup half4 (*vSh)[33]
) {
    #pragma clang fp contract(off) reassociate(off)
    const uint gqa = p.nH / p.nKV;
    const uint gChunks = (gqa + 3u) >> 2u;
    const uint sp = tg.x;
    const uint kvh = tg.y / gChunks;
    const uint g0 = (tg.y - kvh * gChunks) * 4u;
    if (sp >= p.splits || kvh >= p.nKV || g0 >= gqa || p.T == 0u) return;

    const uint gCount = min(4u, gqa - g0);
    const uint headBase = kvh * gqa + g0;
    const uint dBase = lane * 4u;
    const bool activeD = (dBase < p.headDim);
    const float qkScale = rsqrt(float(p.headDim));

    const uint prefixEnd = min(p.posOffset, p.kvLen);
    uint blockStart = 0u, blockEnd = 0u, kStart = 0u, kEnd = 0u;
    if (p.splitSpan > 0u) {
        const uint globalMin = (p.sliding > 0u && p.posOffset + 1u > p.sliding) ? (p.posOffset + 1u - p.sliding) : 0u;
        const uint firstBlock = min(globalMin, p.kvLen) / p.splitSpan;
        blockStart = (firstBlock + sp) * p.splitSpan;
        blockEnd = blockStart + p.splitSpan;
        kStart = max(blockStart, min(globalMin, prefixEnd));
        kEnd = min(blockEnd, prefixEnd);
    } else {
        uint prefixMinKey = 0xFFFFFFFFu;
        const uint tEnd = min(p.T, 4u);
        for (uint tk = 0u; tk < tEnd; ++tk) {
            const uint qp = (tree.isTree != 0u && tk < 16u) ? (p.posOffset + tree.depths[tk]) : (p.posOffset + tk);
            const uint mk = (p.sliding > 0u && qp + 1u > p.sliding) ? (qp + 1u - p.sliding) : 0u;
            prefixMinKey = min(prefixMinKey, mk);
        }
        if (prefixMinKey > prefixEnd) prefixMinKey = prefixEnd;
        const uint totalKeys = prefixEnd - prefixMinKey;
        const uint chunk = (totalKeys > 0u) ? ((totalKeys + p.splits - 1u) / p.splits) : 0u;
        kStart = prefixMinKey + sp * chunk;
        kEnd = min(prefixEnd, kStart + chunk);
    }

    const uint tok = sgi;
    const bool tokValid = (tok < p.T);
    const uint qPos = (tokValid && tree.isTree != 0u && tok < 16u)
        ? (p.posOffset + tree.depths[tok]) : (p.posOffset + tok);
    const uint tokMin = (p.sliding > 0u && qPos + 1u > p.sliding) ? (qPos + 1u - p.sliding) : 0u;
    const uint kvEndTok = min(qPos + 1u, p.kvLen);
    const bool hasSinkTok = (p.splitSpan > 0u)
        ? ((p.hasSink != 0u) && (tokMin >= blockStart) && (tokMin < blockEnd) && (tokMin < kvEndTok))
        : ((sp == 0u) && (p.hasSink != 0u));

    float4 q4[4], acc[4];
    float rLane[4], m[4], l[4];
    #pragma unroll
    for (uint g = 0u; g < 4u; ++g) {
        if (tokValid && g < gCount) {
            const uint h = headBase + g;
            q4[g] = activeD
                ? float4(*(device const bfloat4*)(qBF16 + (size_t(tok) * p.nH + h) * p.headDim + dBase))
                : float4(0.0f);
            rLane[g] = (p.hasRelBias != 0u && lane < p.dRel)
                ? float(rBF16[(size_t(tok) * p.nH + h) * p.dRel + lane]) : 0.0f;
            m[g] = hasSinkTok ? float(sinkBF16[size_t(tok) * p.nH + h]) : -1e30f;
            l[g] = hasSinkTok ? 1.0f : 0.0f;
        } else {
            q4[g] = float4(0.0f); rLane[g] = 0.0f; m[g] = -1e30f; l[g] = 0.0f;
        }
        acc[g] = float4(0.0f);
    }

    const bool mayRel = (p.hasRelBias != 0u) && (p.dRel > 0u) && (p.relExtent > 0u);
    for (uint j0 = kStart; j0 < kEnd; j0 += 16u) {
        const uint cnt = min(16u, kEnd - j0);
        #pragma unroll
        for (uint r = 0u; r < 4u; ++r) {
            const uint row = r * 4u + sgi;
            if (row < cnt && activeD) {
                const uint slot = (p.sliding > 0u) ? ((j0 + row) % p.ringCap) : (j0 + row);
                kSh[row][lane] = qwg_sink_rel_read_kv4<MODE>(lane, slot, kvh, p.nKV, p.headDim, kCache, kMeta);
                vSh[row][lane] = qwg_sink_rel_read_kv4<MODE>(lane, slot, kvh, p.nKV, p.headDim, vCache, vMeta);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tokValid) {
            for (uint b = 0u; b < cnt; ++b) {
                const uint j = j0 + b;
                if (j < tokMin) continue;
                const float4 kf = activeD ? float4(kSh[b][lane]) : float4(0.0f);
                const float4 vf = activeD ? float4(vSh[b][lane]) : float4(0.0f);
                const uint dist = qPos - j;
                qw_attn_batch_score_tok(
                    dist, gCount, mayRel, qkScale, kf, vf, relProj, p, lane,
                    q4, rLane, m, l, acc);
            }
        }
        if (j0 + 16u < kEnd) threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint batchEnd = min(p.kvLen, p.posOffset + p.T);
    const bool enterBatch = (p.splitSpan > 0u)
        ? (p.posOffset < batchEnd && batchEnd > blockStart && p.posOffset < blockEnd)
        : (sp == 0u && p.posOffset < batchEnd);
    if (enterBatch) {
        if (kStart < kEnd) threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j0 = p.posOffset; j0 < batchEnd; j0 += 16u) {
            const uint cnt = min(16u, batchEnd - j0);
            if (sgi < cnt && activeD) {
                const uint jLoad = j0 + sgi;
                const uint slot = (p.sliding > 0u) ? (jLoad % p.ringCap) : jLoad;
                kSh[sgi][lane] = qwg_sink_rel_read_kv4<MODE>(lane, slot, kvh, p.nKV, p.headDim, kCache, kMeta);
                vSh[sgi][lane] = qwg_sink_rel_read_kv4<MODE>(lane, slot, kvh, p.nKV, p.headDim, vCache, vMeta);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tokValid) {
                for (uint b = 0u; b < cnt; ++b) {
                    const uint j = j0 + b;
                    const uint kRel = j - p.posOffset;
                    const uint kLogical = (tree.isTree != 0u && kRel < 16u) ? (p.posOffset + tree.depths[kRel]) : j;
                    if (p.splitSpan > 0u && (kLogical < blockStart || kLogical >= blockEnd)) continue;
                    bool vis = false;
                    uint dist = 0u;
                    if (tree.isTree == 0u) {
                        vis = (j >= tokMin) && (j <= qPos);
                        if (vis) dist = qPos - j;
                    } else {
                        vis = (tok < 16u && kRel < 16u) && ((tree.ancMask[tok] & (1u << kRel)) != 0u);
                        if (vis) {
                            dist = tree.depths[tok] - tree.depths[kRel];
                            vis = (p.sliding == 0u || dist < p.sliding);
                        }
                    }
                    if (!vis) continue;
                    const float4 kf = activeD ? float4(kSh[b][lane]) : float4(0.0f);
                    const float4 vf = activeD ? float4(vSh[b][lane]) : float4(0.0f);
                    qw_attn_batch_score_tok(
                        dist, gCount, mayRel, qkScale, kf, vf, relProj, p, lane,
                        q4, rLane, m, l, acc);
                }
            }
            if (j0 + 16u < batchEnd) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (!tokValid) return;
    if (p.splits == 1u) {
        if (!activeD) return;
        #pragma unroll
        for (uint g = 0u; g < 4u; ++g) {
            if (g < gCount) {
                const uint h = headBase + g;
                const size_t oBase = (size_t(tok) * p.nH + h) * p.headDim + dBase;
                float4 res = acc[g] / max(l[g], 1e-30f);
                if (p.hasOutGate != 0u) {
                    const float4 g4 = float4(*(device const bfloat4*)(gateBF16 + oBase));
                    res.x *= qw_ink2_sigmoid(g4.x); res.y *= qw_ink2_sigmoid(g4.y);
                    res.z *= qw_ink2_sigmoid(g4.z); res.w *= qw_ink2_sigmoid(g4.w);
                }
                *(device bfloat4*)(outBF16 + oBase) = bfloat4(
                    qw_ink2_bf16(res.x), qw_ink2_bf16(res.y), qw_ink2_bf16(res.z), qw_ink2_bf16(res.w)
                );
            }
        }
    } else {
        #pragma unroll
        for (uint g = 0u; g < 4u; ++g) {
            if (g < gCount) {
                const uint h = headBase + g;
                const size_t mlIdx = (size_t(tok) * p.nH + h) * p.splits + sp;
                if (activeD) *(device float4*)(partO + mlIdx * p.headDim + dBase) = acc[g];
                if (lane == 0u) partML[mlIdx] = float2(m[g], l[g]);
            }
        }
    }
}

#define QW_SINK_REL_DECODE_BATCH_ENTRY(SFX, MODE) \
kernel void qwg_attn_decode_sink_rel_gqa_batch##SFX( \
    device const bfloat* qBF16 [[buffer(0)]], device const bfloat* rBF16 [[buffer(1)]], \
    device const bfloat* relProj [[buffer(2)]], device const bfloat* sinkBF16 [[buffer(3)]], \
    device const bfloat* gateBF16 [[buffer(4)]], device const uchar* kCache [[buffer(5)]], \
    device const uchar* vCache [[buffer(6)]], device const uchar* kMeta [[buffer(7)]], \
    device const uchar* vMeta [[buffer(8)]], device bfloat* outBF16 [[buffer(9)]], \
    device float* partO [[buffer(10)]], device float2* partML [[buffer(11)]], \
    constant QWSinkRelAttnParams& p [[buffer(12)]], constant QWSinkRelTreeMeta& tree [[buffer(13)]], \
    uint3 tg [[threadgroup_position_in_grid]], uint sgi [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half4 kSh[16][33]; \
    threadgroup half4 vSh[16][33]; \
    qwg_attn_decode_sink_rel_gqa_batch_body<MODE, 2u>( \
        qBF16, rBF16, relProj, sinkBF16, gateBF16, kCache, vCache, kMeta, vMeta, \
        outBF16, partO, partML, p, tree, tg, sgi, lane, kSh, vSh); \
} \
kernel void qwg_attn_decode_sink_rel_gqa_batch8##SFX( \
    device const bfloat* qBF16 [[buffer(0)]], device const bfloat* rBF16 [[buffer(1)]], \
    device const bfloat* relProj [[buffer(2)]], device const bfloat* sinkBF16 [[buffer(3)]], \
    device const bfloat* gateBF16 [[buffer(4)]], device const uchar* kCache [[buffer(5)]], \
    device const uchar* vCache [[buffer(6)]], device const uchar* kMeta [[buffer(7)]], \
    device const uchar* vMeta [[buffer(8)]], device bfloat* outBF16 [[buffer(9)]], \
    device float* partO [[buffer(10)]], device float2* partML [[buffer(11)]], \
    constant QWSinkRelAttnParams& p [[buffer(12)]], constant QWSinkRelTreeMeta& tree [[buffer(13)]], \
    uint3 tg [[threadgroup_position_in_grid]], uint sgi [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half4 kSh[16][33]; \
    threadgroup half4 vSh[16][33]; \
    qwg_attn_decode_sink_rel_gqa_batch_body<MODE, 1u>( \
        qBF16, rBF16, relProj, sinkBF16, gateBF16, kCache, vCache, kMeta, vMeta, \
        outBF16, partO, partML, p, tree, tg, sgi, lane, kSh, vSh); \
} \
kernel void qwg_attn_decode_sink_rel_gqa_batch4##SFX( \
    device const bfloat* qBF16 [[buffer(0)]], device const bfloat* rBF16 [[buffer(1)]], \
    device const bfloat* relProj [[buffer(2)]], device const bfloat* sinkBF16 [[buffer(3)]], \
    device const bfloat* gateBF16 [[buffer(4)]], device const uchar* kCache [[buffer(5)]], \
    device const uchar* vCache [[buffer(6)]], device const uchar* kMeta [[buffer(7)]], \
    device const uchar* vMeta [[buffer(8)]], device bfloat* outBF16 [[buffer(9)]], \
    device float* partO [[buffer(10)]], device float2* partML [[buffer(11)]], \
    constant QWSinkRelAttnParams& p [[buffer(12)]], constant QWSinkRelTreeMeta& tree [[buffer(13)]], \
    uint3 tg [[threadgroup_position_in_grid]], uint sgi [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half4 kSh[16][33]; \
    threadgroup half4 vSh[16][33]; \
    qwg_attn_decode_sink_rel_gqa_batch4_body<MODE>( \
        qBF16, rBF16, relProj, sinkBF16, gateBF16, kCache, vCache, kMeta, vMeta, \
        outBF16, partO, partML, p, tree, tg, sgi, lane, kSh, vSh); \
}
QW_SINK_REL_DECODE_BATCH_ENTRY(_f16, 0u)
QW_SINK_REL_DECODE_BATCH_ENTRY(_q8, 1u)
QW_SINK_REL_DECODE_BATCH_ENTRY(_q4, 2u)
#undef QW_SINK_REL_DECODE_BATCH_ENTRY
