// qwattn_mla_mma_split.metal — Apple9+ simdgroup-MMA Split-K sparse MLA & DSA attention.
// Writes normalized log2-domain partials (partO, partML) consumed by qwg_mla*_combine.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "../include/qw_mla_dequant.metalh"

using namespace metal;

#define MLA_SPLIT_HG   16u
#define MLA_SPLIT_TK   16u
#define MLA_SPLIT_RANK 512u
#define MLA_SPLIT_KS   (MLA_SPLIT_RANK + 8u)
#define MLA_SPLIT_KPS  72u

static inline uint qw_mla_split_row(uint l) {
    return ((l >> 4) & 1u) * 4u + ((l >> 2) & 1u) * 2u + ((l >> 1) & 1u);
}

static inline uint qw_mla_split_col(uint l) {
    return ((l >> 3) & 1u) * 4u + (l & 1u) * 2u;
}

// Grid: (T, qHeads / 16, nSplit), Threadgroup: (128, 1, 1).
kernel void qwg_mla_sparse_attn_mma_split(
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
    const uint t = tg.x, h0 = tg.y * MLA_SPLIT_HG, z = tg.z, nSplit = ntg.z;
    if (t >= p.T) return;

    threadgroup half  ks[MLA_SPLIT_TK * MLA_SPLIT_KS];
    threadgroup float sbuf[MLA_SPLIT_HG * MLA_SPLIT_TK];
    threadgroup half  pbuf[MLA_SPLIT_HG * MLA_SPLIT_TK];
    threadgroup float corrH[MLA_SPLIT_HG];
    threadgroup float mH[MLA_SPLIT_HG];
    threadgroup float lH[MLA_SPLIT_HG];
    threadgroup uint  blkIdx[1024];

    const uint halfRank = MLA_SPLIT_RANK / 2u;
    const uint pq = p.pos0 + t;
    const uint nPools = (pq + 1u) >> 2u;
    const uint cnt = min(min(p.topKPools, 1024u), nPools);
    const uint tailStart = nPools << 2u;
    const uint slots = (cnt << 2u) + (pq + 1u - tailStart);
    const float scale2 = rsqrt(float(p.headDim)) * 1.4426950408889634f;

    const uint chunk = (((slots + nSplit - 1u) / nSplit) + 15u) & ~15u;
    const uint sBegin = min(slots, z * chunk);
    const uint sEnd = min(slots, sBegin + chunk);
    if (sBegin >= sEnd) {
        if (tid < MLA_SPLIT_HG) {
            partML[((size_t)t * nSplit + z) * p.qHeads + h0 + tid] = float2(-1e30f, 0.0f);
        }
        return;
    }

    if (nPools <= p.topKPools) {
        for (uint i = tid; i < cnt; i += 128u) blkIdx[i] = i << 2u;
    } else {
        device const int* rowList = blockList + (size_t)t * p.listStride;
        for (uint i = tid; i < cnt; i += 128u) blkIdx[i] = uint(rowList[i]) << 2u;
    }
    if (tid < MLA_SPLIT_HG) { mH[tid] = -1e30f; lH[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint rl = qw_mla_split_row(lane), cl = qw_mla_split_col(lane);
    const uint sHb = sg & 1u, sTb = sg >> 1u;
    device const half* qBase = qAbs + ((size_t)t * p.qHeads + h0 + sHb * 8u) * MLA_SPLIT_RANK;

    simdgroup_float8x8 oacc[2][16];
    for (uint a = 0u; a < 2u; ++a) {
        for (uint b = 0u; b < 16u; ++b) oacc[a][b] = simdgroup_float8x8(0.0f);
    }

    for (uint s0 = sBegin; s0 < sEnd; s0 += MLA_SPLIT_TK) {
        for (uint i = tid; i < MLA_SPLIT_TK * (MLA_SPLIT_RANK / 4u); i += 128u) {
            const uint r = i / (MLA_SPLIT_RANK / 4u);
            const uint c4 = (i - r * (MLA_SPLIT_RANK / 4u)) * 4u;
            const uint slot = s0 + r;
            half4 v = half4(0.0h);
            if (slot < sEnd) {
                const uint j = (slot < (cnt << 2u)) ? (blkIdx[slot >> 2u] + (slot & 3u))
                                                    : (tailStart + (slot - (cnt << 2u)));
                const uint pageSlot = j >> 8u, pageOff = j & 255u;
                if (c4 < halfRank) {
                    const uint pg = uint(kTable[pageSlot]);
                    v = qw_mla_load_kv_half4((device const uchar*)kPool, pg, pageOff, c4, p.kvMode);
                } else {
                    const uint pg = uint(vTable[pageSlot]);
                    v = qw_mla_load_kv_half4((device const uchar*)vPool, pg, pageOff, c4 - halfRank, p.kvMode);
                }
            }
            *(threadgroup half4*)(ks + r * MLA_SPLIT_KS + c4) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            simdgroup_float8x8 sacc = simdgroup_float8x8(0.0f);
            for (uint d = 0u; d < MLA_SPLIT_RANK / 8u; ++d) {
                simdgroup_half8x8 qf, kf;
                simdgroup_load(qf, qBase + d * 8u, MLA_SPLIT_RANK);
                simdgroup_load(kf, ks + sTb * 8u * MLA_SPLIT_KS + d * 8u, MLA_SPLIT_KS, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sacc, qf, kf, sacc);
            }
            auto e = sacc.thread_elements();
            const uint hr = sHb * 8u + rl;
            for (uint i = 0u; i < 2u; ++i) {
                const uint col = sTb * 8u + cl + i;
                sbuf[hr * MLA_SPLIT_TK + col] = (s0 + col < sEnd) ? e[i] * scale2 : -1e30f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            const uint hr = tid >> 3u, c0 = (tid & 7u) * 2u;
            const float a = sbuf[hr * MLA_SPLIT_TK + c0];
            const float b = sbuf[hr * MLA_SPLIT_TK + c0 + 1u];
            float mx = max(a, b);
            mx = max(mx, simd_shuffle_xor(mx, 1u));
            mx = max(mx, simd_shuffle_xor(mx, 2u));
            mx = max(mx, simd_shuffle_xor(mx, 4u));
            const float mOld = mH[hr];
            const float mNew = max(mOld, mx);
            const float pa = (mNew <= -1e29f || a <= -1e29f) ? 0.0f : fast::exp2(a - mNew);
            const float pb = (mNew <= -1e29f || b <= -1e29f) ? 0.0f : fast::exp2(b - mNew);
            pbuf[hr * MLA_SPLIT_TK + c0] = half(pa);
            pbuf[hr * MLA_SPLIT_TK + c0 + 1u] = half(pb);
            float sm = pa + pb;
            sm += simd_shuffle_xor(sm, 1u);
            sm += simd_shuffle_xor(sm, 2u);
            sm += simd_shuffle_xor(sm, 4u);
            const float corr = (mOld <= -1e29f) ? 0.0f : fast::exp2(mOld - mNew);
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
            for (uint a = 0u; a < 2u; ++a) {
                for (uint k = 0u; k < 2u; ++k) {
                    simdgroup_load(pf[a][k], pbuf + a * 8u * MLA_SPLIT_TK + k * 8u, MLA_SPLIT_TK);
                }
            }
            for (uint b = 0u; b < 16u; ++b) {
                const uint col = sg * 128u + b * 8u;
                simdgroup_half8x8 kf0, kf1;
                simdgroup_load(kf0, ks + col, MLA_SPLIT_KS);
                simdgroup_load(kf1, ks + 8u * MLA_SPLIT_KS + col, MLA_SPLIT_KS);
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

    const size_t mlBase = ((size_t)t * nSplit + z) * p.qHeads;
    if (tid < MLA_SPLIT_HG) partML[mlBase + h0 + tid] = float2(mH[tid], lH[tid]);
    for (uint a = 0u; a < 2u; ++a) {
        const uint h = h0 + a * 8u + rl;
        const float l = lH[a * 8u + rl];
        const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        device half* dst = partO + (mlBase + h) * MLA_SPLIT_RANK;
        for (uint b = 0u; b < 16u; ++b) {
            const auto e = oacc[a][b].thread_elements();
            const uint col = sg * 128u + b * 8u + cl;
            *(device half2*)(dst + col) = half2(half(e[0] * inv), half(e[1] * inv));
        }
    }
}

// Grid: (T, qHeads / 16, nSplit), Threadgroup: (128, 1, 1).
// attnSink is folded once in qwg_mla_dsa_sparse_attn_combine.
kernel void qwg_mla_dsa_sparse_attn_mma_split(
    device const half*              qAbs      [[buffer(0)]],
    device const half*              kPool     [[buffer(1)]],
    device const half*              vPool     [[buffer(2)]],
    device const int*               kTable    [[buffer(3)]],
    device const int*               vTable    [[buffer(4)]],
    device const int*               blockList [[buffer(5)]],
    device half*                    partO     [[buffer(6)]],
    constant MLADSASparseAttnParams& p        [[buffer(7)]],
    device float2*                  partML    [[buffer(8)]],
    device const half*              mlaQFull  [[buffer(9)]],
    device const half*              kPeCache  [[buffer(10)]],
    device const float*             attnSink  [[buffer(11)]],
    uint3                           tg        [[threadgroup_position_in_grid]],
    uint3                           ntg       [[threadgroups_per_grid]],
    uint                            tid       [[thread_index_in_threadgroup]],
    uint                            sg        [[simdgroup_index_in_threadgroup]],
    uint                            lane      [[thread_index_in_simdgroup]]
) {
    (void)attnSink;
    const uint t = tg.x, h0 = tg.y * MLA_SPLIT_HG, z = tg.z, nSplit = ntg.z;
    if (t >= p.T) return;

    threadgroup half  ks[MLA_SPLIT_TK * MLA_SPLIT_KS];
    threadgroup half  kps[MLA_SPLIT_TK * MLA_SPLIT_KPS];
    threadgroup float sbuf[MLA_SPLIT_HG * MLA_SPLIT_TK];
    threadgroup half  pbuf[MLA_SPLIT_HG * MLA_SPLIT_TK];
    threadgroup float corrH[MLA_SPLIT_HG];
    threadgroup float mH[MLA_SPLIT_HG];
    threadgroup float lH[MLA_SPLIT_HG];
    threadgroup uint  blkIdx[2560];

    const uint halfRank = MLA_SPLIT_RANK / 2u;
    const uint seqLen = p.pos0 + t + 1u;
    const bool isDenseStream = (p.topK == 0u);
    const uint startTok = isDenseStream
        ? ((p.windowTokens > 0u && seqLen > p.windowTokens) ? (seqLen - p.windowTokens) : 0u)
        : 0u;
    const uint denseLim = p.topK + p.windowTokens;
    const uint slots = isDenseStream ? (seqLen - startTok) : min(min(seqLen, denseLim), 2560u);
    const float baseScale = (p.softmaxScale > 0.0f) ? p.softmaxScale : rsqrt(float(p.headDim));
    const float scale2 = baseScale * 1.4426950408889634f;

    const uint chunk = (((slots + nSplit - 1u) / nSplit) + 15u) & ~15u;
    const uint sBegin = min(slots, z * chunk);
    const uint sEnd = min(slots, sBegin + chunk);
    if (sBegin >= sEnd) {
        if (tid < MLA_SPLIT_HG) {
            partML[((size_t)t * nSplit + z) * p.qHeads + h0 + tid] = float2(-1e30f, 0.0f);
        }
        return;
    }

    if (!isDenseStream) {
        if (seqLen <= denseLim) {
            for (uint i = tid; i < slots; i += 128u) blkIdx[i] = i;
        } else {
            device const int* rowList = blockList + (size_t)t * p.listStride;
            for (uint i = tid; i < slots; i += 128u) blkIdx[i] = min(uint(max(rowList[i], 0)), seqLen - 1u);
        }
    }
    if (tid < MLA_SPLIT_HG) { mH[tid] = -1e30f; lH[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint rl = qw_mla_split_row(lane), cl = qw_mla_split_col(lane);
    const uint sHb = sg & 1u, sTb = sg >> 1u;
    device const half* qBase = qAbs + ((size_t)t * p.qHeads + h0 + sHb * 8u) * MLA_SPLIT_RANK;
    device const half* qRopeBase = mlaQFull + (size_t)t * p.qFullStride + p.qRopeOffset
                                   + (size_t)(h0 + sHb * 8u) * p.ropeDim;

    simdgroup_float8x8 oacc[2][16];
    for (uint a = 0u; a < 2u; ++a) {
        for (uint b = 0u; b < 16u; ++b) oacc[a][b] = simdgroup_float8x8(0.0f);
    }

    for (uint s0 = sBegin; s0 < sEnd; s0 += MLA_SPLIT_TK) {
        for (uint i = tid; i < MLA_SPLIT_TK * (MLA_SPLIT_RANK / 4u); i += 128u) {
            const uint r = i / (MLA_SPLIT_RANK / 4u);
            const uint c4 = (i - r * (MLA_SPLIT_RANK / 4u)) * 4u;
            const uint slot = s0 + r;
            half4 v = half4(0.0h);
            if (slot < sEnd) {
                const uint j = isDenseStream ? (startTok + slot) : blkIdx[slot];
                const uint pageSlot = j >> 8u, pageOff = j & 255u;
                if (c4 < halfRank) {
                    const uint pg = uint(kTable[pageSlot]);
                    v = qw_mla_load_kv_half4((device const uchar*)kPool, pg, pageOff, c4, p.kvMode);
                } else {
                    const uint pg = uint(vTable[pageSlot]);
                    v = qw_mla_load_kv_half4((device const uchar*)vPool, pg, pageOff, c4 - halfRank, p.kvMode);
                }
            }
            *(threadgroup half4*)(ks + r * MLA_SPLIT_KS + c4) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 sacc = simdgroup_float8x8(0.0f);
        for (uint d = 0u; d < MLA_SPLIT_RANK / 8u; ++d) {
            simdgroup_half8x8 qf, kf;
            simdgroup_load(qf, qBase + d * 8u, MLA_SPLIT_RANK);
            simdgroup_load(kf, ks + sTb * 8u * MLA_SPLIT_KS + d * 8u, MLA_SPLIT_KS, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(sacc, qf, kf, sacc);
        }
        for (uint ropeOff = 0u; ropeOff < p.ropeDim; ropeOff += 64u) {
            const uint chunkCols = min(64u, p.ropeDim - ropeOff);
            const uint kpVecs = chunkCols >> 2u;
            for (uint i = tid; i < MLA_SPLIT_TK * kpVecs; i += 128u) {
                const uint r = i / kpVecs;
                const uint c4 = (i - r * kpVecs) * 4u;
                const uint slot = s0 + r;
                half4 rv = half4(0.0h);
                if (slot < sEnd) {
                    const uint j = isDenseStream ? (startTok + slot) : blkIdx[slot];
                    rv = *(device const half4*)(kPeCache + (size_t)j * p.ropeDim + ropeOff + c4);
                }
                *(threadgroup half4*)(kps + r * MLA_SPLIT_KPS + c4) = rv;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint ropeTiles = chunkCols >> 3u;
            for (uint d = 0u; d < ropeTiles; ++d) {
                simdgroup_half8x8 qf, kf;
                simdgroup_load(qf, qRopeBase + ropeOff + d * 8u, p.ropeDim);
                simdgroup_load(kf, kps + sTb * 8u * MLA_SPLIT_KPS + d * 8u, MLA_SPLIT_KPS, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sacc, qf, kf, sacc);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        {
            const auto e = sacc.thread_elements();
            const uint hr = sHb * 8u + rl;
            for (uint i = 0u; i < 2u; ++i) {
                const uint col = sTb * 8u + cl + i;
                sbuf[hr * MLA_SPLIT_TK + col] = (s0 + col < sEnd) ? e[i] * scale2 : -1e30f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            const uint hr = tid >> 3u, c0 = (tid & 7u) * 2u;
            const float a = sbuf[hr * MLA_SPLIT_TK + c0];
            const float b = sbuf[hr * MLA_SPLIT_TK + c0 + 1u];
            float mx = max(a, b);
            mx = max(mx, simd_shuffle_xor(mx, 1u));
            mx = max(mx, simd_shuffle_xor(mx, 2u));
            mx = max(mx, simd_shuffle_xor(mx, 4u));
            const float mOld = mH[hr];
            const float mNew = max(mOld, mx);
            const float pa = (mNew <= -1e29f || a <= -1e29f) ? 0.0f : fast::exp2(a - mNew);
            const float pb = (mNew <= -1e29f || b <= -1e29f) ? 0.0f : fast::exp2(b - mNew);
            pbuf[hr * MLA_SPLIT_TK + c0] = half(pa);
            pbuf[hr * MLA_SPLIT_TK + c0 + 1u] = half(pb);
            float sm = pa + pb;
            sm += simd_shuffle_xor(sm, 1u);
            sm += simd_shuffle_xor(sm, 2u);
            sm += simd_shuffle_xor(sm, 4u);
            const float corr = (mOld <= -1e29f) ? 0.0f : fast::exp2(mOld - mNew);
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
            for (uint a = 0u; a < 2u; ++a) {
                for (uint k = 0u; k < 2u; ++k) {
                    simdgroup_load(pf[a][k], pbuf + a * 8u * MLA_SPLIT_TK + k * 8u, MLA_SPLIT_TK);
                }
            }
            for (uint b = 0u; b < 16u; ++b) {
                const uint col = sg * 128u + b * 8u;
                simdgroup_half8x8 kf0, kf1;
                simdgroup_load(kf0, ks + col, MLA_SPLIT_KS);
                simdgroup_load(kf1, ks + 8u * MLA_SPLIT_KS + col, MLA_SPLIT_KS);
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

    const size_t mlBase = ((size_t)t * nSplit + z) * p.qHeads;
    if (tid < MLA_SPLIT_HG) partML[mlBase + h0 + tid] = float2(mH[tid], lH[tid]);
    for (uint a = 0u; a < 2u; ++a) {
        const uint h = h0 + a * 8u + rl;
        const float l = lH[a * 8u + rl];
        const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        device half* dst = partO + (mlBase + h) * MLA_SPLIT_RANK;
        for (uint b = 0u; b < 16u; ++b) {
            const auto e = oacc[a][b].thread_elements();
            const uint col = sg * 128u + b * 8u + cl;
            *(device half2*)(dst + col) = half2(half(e[0] * inv), half(e[1] * inv));
        }
    }
}
