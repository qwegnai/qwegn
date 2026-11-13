// qwattn_mla_mma.metal — simdgroup-MMA prefill kernels for NoPE MLA (see qwattn_mla_sparse.metal
// for the scalar reference kernels and the full pipeline): the absorbed query projection and
// the MQA-shared sparse latent attention.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "../include/qw_mla_dequant.metalh"
using namespace metal;

// simdgroup 8x8 accumulator fragment map: lane -> (row, first of two adjacent columns).
static inline uint qw_mla_row(uint l) { return ((l >> 4) & 1u) * 4u + ((l >> 2) & 1u) * 2u + ((l >> 1) & 1u); }
static inline uint qw_mla_col(uint l) { return ((l >> 3) & 1u) * 4u + (l & 1u) * 2u; }

// 4b. Absorbed query projection on simdgroup MMA for HQ4_G32, SHVQ, and AQ/HQ (nope 256, rank 512).
// Grid: (qHeads, (T + 31) / 32, kvLoraRank / 128), Threadgroup: (128, 1, 1).
#define MLA_ABS_TM 32u
#define MLA_ABS_TN 128u
#define MLA_ABS_TK 32u

template <bool SUB2>
static inline void qwg_mla_absorb_q_mma_impl(
    device const half*       mlaQFull,
    device const char*       kvBBlocks,
    device half*             qAbsOut,
    constant MLAAbsorbParams& p,
    uint3                    tg,
    uint                     tid,
    uint                     sg,
    uint                     lane,
    threadgroup half*        qs,
    threadgroup half*        ws,
    threadgroup ShvqMlaTgLut& tgLut
) {
    const uint h = tg.x;
    const uint tBase = tg.y * MLA_ABS_TM;
    const uint cBase = tg.z * MLA_ABS_TN;
    if (h >= p.qHeads || tBase >= p.T) return;
    const uint inc = p.isIncumbent & 0xFFu;

    if (inc == 2u && qw_mla_load_shvq_lut<SUB2>(tgLut, p.bits, tid, 128u)) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint qsStride = MLA_ABS_TK + 8u, wsStride = MLA_ABS_TN + 8u;

    simdgroup_float8x8 acc[4][4];
    #pragma unroll
    for (uint i = 0u; i < 4u; ++i) {
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
    }

    const uint headRowBase = h * (p.nopeDim + p.vHeadDim);
    const uint wRow = tid >> 2u, wBlk = tid & 3u;
    const uint qRow = tid >> 2u, qCol = (tid & 3u) * 8u;
    const bool qValid = tBase + qRow < p.T;
    device const half* qSrc = mlaQFull + (size_t)(tBase + qRow) * p.qFullStride + h * p.nopeDim + qCol;

    for (uint k0 = 0u; k0 < p.nopeDim; k0 += MLA_ABS_TK) {
        {
            half4 a = half4(0.0h), b = half4(0.0h);
            if (qValid) {
                a = *(device const half4*)(qSrc + k0);
                b = *(device const half4*)(qSrc + k0 + 4u);
            }
            *(threadgroup half4*)(qs + qRow * qsStride + qCol) = a;
            *(threadgroup half4*)(qs + qRow * qsStride + qCol + 4u) = b;
        }
        qw_mla_unpack_slice32<SUB2>(kvBBlocks, headRowBase + k0 + wRow, cBase + wBlk * 32u,
                                    p, tgLut, ws + wRow * wsStride + wBlk * 32u, 1u);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0u; kk < MLA_ABS_TK; kk += 8u) {
            simdgroup_half8x8 af[4], bf[4];
            #pragma unroll
            for (uint i = 0u; i < 4u; ++i) simdgroup_load(af[i], qs + i * 8u * qsStride + kk, qsStride);
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) simdgroup_load(bf[j], ws + kk * wsStride + sg * 32u + j * 8u, wsStride);
            #pragma unroll
            for (uint i = 0u; i < 4u; ++i) {
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j) simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint rl = qw_mla_row(lane), cl = qw_mla_col(lane);
    #pragma unroll
    for (uint i = 0u; i < 4u; ++i) {
        const uint t = tBase + i * 8u + rl;
        if (t >= p.T) continue;
        device half* dst = qAbsOut + ((size_t)t * p.qHeads + h) * p.kvLoraRank + cBase + sg * 32u + cl;
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            auto e = acc[i][j].thread_elements();
            *(device half2*)(dst + j * 8u) = half2(half(e[0]), half(e[1]));
        }
    }
}

kernel void qwg_mla_absorb_q_mma(
    device const half*       mlaQFull   [[buffer(0)]],
    device const char*       kvBBlocks  [[buffer(1)]],
    device half*             qAbsOut    [[buffer(2)]],
    constant MLAAbsorbParams& p         [[buffer(3)]],
    uint3                    tg         [[threadgroup_position_in_grid]],
    uint                     tid        [[thread_index_in_threadgroup]],
    uint                     sg         [[simdgroup_index_in_threadgroup]],
    uint                     lane       [[thread_index_in_simdgroup]]
) {
    threadgroup half qs[MLA_ABS_TM * (MLA_ABS_TK + 8u)];
    threadgroup half ws[MLA_ABS_TK * (MLA_ABS_TN + 8u)];
    threadgroup ShvqMlaTgLut tgLut;
    qwg_mla_absorb_q_mma_impl<false>(mlaQFull, kvBBlocks, qAbsOut, p, tg, tid, sg, lane, qs, ws, tgLut);
}

kernel void qwg_mla_absorb_q_mma_sub2(
    device const half*       mlaQFull   [[buffer(0)]],
    device const char*       kvBBlocks  [[buffer(1)]],
    device half*             qAbsOut    [[buffer(2)]],
    constant MLAAbsorbParams& p         [[buffer(3)]],
    uint3                    tg         [[threadgroup_position_in_grid]],
    uint                     tid        [[thread_index_in_threadgroup]],
    uint                     sg         [[simdgroup_index_in_threadgroup]],
    uint                     lane       [[thread_index_in_simdgroup]]
) {
    threadgroup half qs[MLA_ABS_TM * (MLA_ABS_TK + 8u)];
    threadgroup half ws[MLA_ABS_TK * (MLA_ABS_TN + 8u)];
    threadgroup ShvqMlaTgLut tgLut;
    qwg_mla_absorb_q_mma_impl<true>(mlaQFull, kvBBlocks, qAbsOut, p, tg, tid, sg, lane, qs, ws, tgLut);
}

// 5b. MQA-shared sparse latent attention on simdgroup MMA (f16/q8/q4 KV).
// Stages 16-token tiles of c_KV (512 dims) once in threadgroup SRAM across MLA_MMA_HG heads.
// Grid: (T, qHeads / MLA_MMA_HG, 1), Threadgroup: (128, 1, 1).
#define MLA_MMA_HG   16u
#define MLA_MMA_TK   16u
#define MLA_MMA_RANK 512u
#define MLA_MMA_KS   (MLA_MMA_RANK + 8u)

kernel void qwg_mla_sparse_attn_mma(
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
    const uint t = tg.x;
    const uint h0 = tg.y * MLA_MMA_HG;
    if (t >= p.T) return;

    threadgroup half  ks[MLA_MMA_TK * MLA_MMA_KS];
    threadgroup float sbuf[MLA_MMA_HG * MLA_MMA_TK];
    threadgroup half  pbuf[MLA_MMA_HG * MLA_MMA_TK];
    threadgroup float corrH[MLA_MMA_HG];
    threadgroup float mH[MLA_MMA_HG];
    threadgroup float lH[MLA_MMA_HG];
    threadgroup uint  blkIdx[1024];

    const uint halfRank = MLA_MMA_RANK / 2u;
    const uint pq = p.pos0 + t;
    const uint nPools = (pq + 1u) >> 2u;
    const uint cnt = min(min(p.topKPools, 1024u), nPools);
    const uint tailStart = nPools << 2u;
    const uint slots = (cnt << 2u) + (pq + 1u - tailStart);
    const float scale = rsqrt(float(p.headDim));

    if (nPools <= p.topKPools) {
        for (uint i = tid; i < cnt; i += 128u) blkIdx[i] = i << 2u;
    } else {
        device const int* rowList = blockList + (size_t)t * p.listStride;
        for (uint i = tid; i < cnt; i += 128u) blkIdx[i] = uint(rowList[i]) << 2u;
    }
    if (tid < MLA_MMA_HG) { mH[tid] = -INFINITY; lH[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint rl = qw_mla_row(lane), cl = qw_mla_col(lane);
    // S tile owned by this simdgroup: heads sHb*8.., tokens sTb*8..
    const uint sHb = sg & 1u, sTb = sg >> 1u;
    device const half* qBase = qAbs + ((size_t)t * p.qHeads + h0 + sHb * 8u) * MLA_MMA_RANK;
    // O columns owned by this simdgroup: dims sg*128 .. +128 for all MLA_MMA_HG heads.
    simdgroup_float8x8 oacc[2][16];
    #pragma unroll
    for (uint a = 0u; a < 2u; ++a) {
        #pragma unroll
        for (uint b = 0u; b < 16u; ++b) oacc[a][b] = simdgroup_float8x8(0.0f);
    }

    for (uint s0 = 0u; s0 < slots; s0 += MLA_MMA_TK) {
        // Stage 16 latent rows: 16 x 512 halves as half4 (4 per lane-chunk), zero past `slots`.
        for (uint i = tid; i < MLA_MMA_TK * (MLA_MMA_RANK / 4u); i += 128u) {
            const uint r = i / (MLA_MMA_RANK / 4u);
            const uint c4 = (i - r * (MLA_MMA_RANK / 4u)) * 4u;
            const uint slot = s0 + r;
            half4 v = half4(0.0h);
            if (slot < slots) {
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
            *(threadgroup half4*)(ks + r * MLA_MMA_KS + c4) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = q_abs (8 heads x 512) * c^T (512 x 8 tokens).
        {
            simdgroup_float8x8 sacc = simdgroup_float8x8(0.0f);
            for (uint d = 0u; d < MLA_MMA_RANK / 8u; ++d) {
                simdgroup_half8x8 qf, kf;
                simdgroup_load(qf, qBase + d * 8u, MLA_MMA_RANK);
                simdgroup_load(kf, ks + sTb * 8u * MLA_MMA_KS + d * 8u, MLA_MMA_KS, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sacc, qf, kf, sacc);
            }
            auto e = sacc.thread_elements();
            const uint hr = sHb * 8u + rl;
            #pragma unroll
            for (uint i = 0u; i < 2u; ++i) {
                const uint col = sTb * 8u + cl + i;
                sbuf[hr * MLA_MMA_TK + col] = (s0 + col < slots) ? e[i] * scale : -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax: 8 threads per head row, 2 tokens each.
        {
            const uint hr = tid >> 3u, c0 = (tid & 7u) * 2u;
            const float a = sbuf[hr * MLA_MMA_TK + c0];
            const float b = sbuf[hr * MLA_MMA_TK + c0 + 1u];
            float mx = max(a, b);
            mx = max(mx, simd_shuffle_xor(mx, 1u));
            mx = max(mx, simd_shuffle_xor(mx, 2u));
            mx = max(mx, simd_shuffle_xor(mx, 4u));
            const float mOld = mH[hr];
            const float mNew = max(mOld, mx);
            const float pa = exp(a - mNew), pb = exp(b - mNew);
            pbuf[hr * MLA_MMA_TK + c0] = half(pa);
            pbuf[hr * MLA_MMA_TK + c0 + 1u] = half(pb);
            float sm = pa + pb;
            sm += simd_shuffle_xor(sm, 1u);
            sm += simd_shuffle_xor(sm, 2u);
            sm += simd_shuffle_xor(sm, 4u);
            const float corr = (mOld == -INFINITY) ? 0.0f : exp(mOld - mNew);
            // Every lane reads mH/lH above before lane 0 of the group writes; the barrier
            // below orders these writes before the next tile's reads.
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if ((tid & 7u) == 0u) {
                corrH[hr] = corr;
                mH[hr] = mNew;
                lH[hr] = lH[hr] * corr + sm;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(corr) O + P (heads x 16 tokens) * c (16 tokens x 128 dims of this simdgroup).
        {
            const float c0 = corrH[rl], c1 = corrH[8u + rl];
            simdgroup_half8x8 pf[2][2];
            #pragma unroll
            for (uint a = 0u; a < 2u; ++a) {
                #pragma unroll
                for (uint k = 0u; k < 2u; ++k)
                    simdgroup_load(pf[a][k], pbuf + a * 8u * MLA_MMA_TK + k * 8u, MLA_MMA_TK);
            }
            #pragma unroll
            for (uint b = 0u; b < 16u; ++b) {
                const uint col = sg * 128u + b * 8u;
                simdgroup_half8x8 kf0, kf1;
                simdgroup_load(kf0, ks + col, MLA_MMA_KS);
                simdgroup_load(kf1, ks + 8u * MLA_MMA_KS + col, MLA_MMA_KS);
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
        const uint h = h0 + a * 8u + rl;
        const float l = lH[a * 8u + rl];
        const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        device half* dst = uLatOut + ((size_t)t * p.qHeads + h) * MLA_MMA_RANK;
        #pragma unroll
        for (uint b = 0u; b < 16u; ++b) {
            auto e = oacc[a][b].thread_elements();
            const uint col = sg * 128u + b * 8u + cl;
            *(device half2*)(dst + col) = half2(half(e[0] * inv), half(e[1] * inv));
        }
    }
}

// 6b. Value up-projection on simdgroup MMA for HQ4_G32, SHVQ, and AQ/HQ (rank 512, v 256).
// Grid: (qHeads, (T + 31) / 32, vHeadDim / 128), Threadgroup: (128, 1, 1).
template <bool SUB2>
static inline void qwg_mla_proj_uv_mma_impl(
    device const half*       uLat,
    device const char*       kvBBlocks,
    device half*             attOut,
    constant MLAAbsorbParams& p,
    uint3                    tg,
    uint                     tid,
    uint                     sg,
    uint                     lane,
    threadgroup half*        us,
    threadgroup half*        ws,
    threadgroup float*       os,
    threadgroup ShvqMlaTgLut& tgLut
) {
    const uint h = tg.x;
    const uint tBase = tg.y * MLA_ABS_TM;
    const uint cBase = tg.z * MLA_ABS_TN;
    if (h >= p.qHeads || tBase >= p.T) return;
    const uint inc = p.isIncumbent & 0xFFu;

    if (inc == 2u && qw_mla_load_shvq_lut<SUB2>(tgLut, p.bits, tid, 128u)) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint usStride = MLA_ABS_TK + 8u, wsStride = MLA_ABS_TN + 8u, osStride = MLA_ABS_TN + 4u;

    simdgroup_float8x8 acc[4][4];
    #pragma unroll
    for (uint i = 0u; i < 4u; ++i) {
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
    }

    const uint wRow = h * (p.nopeDim + p.vHeadDim) + p.nopeDim + cBase + tid;
    const uint uRow = tid >> 2u, uCol = (tid & 3u) * 8u;
    const bool uValid = tBase + uRow < p.T;
    device const half* uSrc = uLat + ((size_t)(tBase + uRow) * p.qHeads + h) * p.kvLoraRank + uCol;

    for (uint k0 = 0u; k0 < p.kvLoraRank; k0 += MLA_ABS_TK) {
        {
            half4 a = half4(0.0h), b = half4(0.0h);
            if (uValid) {
                a = *(device const half4*)(uSrc + k0);
                b = *(device const half4*)(uSrc + k0 + 4u);
            }
            *(threadgroup half4*)(us + uRow * usStride + uCol) = a;
            *(threadgroup half4*)(us + uRow * usStride + uCol + 4u) = b;
        }
        qw_mla_unpack_slice32<SUB2>(kvBBlocks, wRow, k0, p, tgLut, ws + tid, wsStride);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0u; kk < MLA_ABS_TK; kk += 8u) {
            simdgroup_half8x8 af[4], bf[4];
            #pragma unroll
            for (uint i = 0u; i < 4u; ++i) simdgroup_load(af[i], us + i * 8u * usStride + kk, usStride);
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) simdgroup_load(bf[j], ws + kk * wsStride + sg * 32u + j * 8u, wsStride);
            #pragma unroll
            for (uint i = 0u; i < 4u; ++i) {
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j) simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint i = 0u; i < 4u; ++i) {
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) simdgroup_store(acc[i][j], os + i * 8u * osStride + sg * 32u + j * 8u, osStride);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint outStride = p.qHeads * p.vHeadDim;
    const uint rows = min(MLA_ABS_TM, p.T - tBase);
    const bool skipFwht = (p.isIncumbent & 0x10000u) != 0u;
    for (uint m = 0u; m < rows; ++m) {
        const float raw = os[m * osStride + sg * 32u + lane];
        const float rot = skipFwht ? raw : qwg_fwht32_lane_f32(raw, lane);
        attOut[(size_t)(tBase + m) * outStride + h * p.vHeadDim + cBase + sg * 32u + lane] = half(rot);
    }
}

kernel void qwg_mla_proj_uv_mma(
    device const half*       uLat       [[buffer(0)]],
    device const char*       kvBBlocks  [[buffer(1)]],
    device half*             attOut     [[buffer(2)]],
    constant MLAAbsorbParams& p         [[buffer(3)]],
    uint3                    tg         [[threadgroup_position_in_grid]],
    uint                     tid        [[thread_index_in_threadgroup]],
    uint                     sg         [[simdgroup_index_in_threadgroup]],
    uint                     lane       [[thread_index_in_simdgroup]]
) {
    threadgroup half us[MLA_ABS_TM * (MLA_ABS_TK + 8u)];
    threadgroup half ws[MLA_ABS_TK * (MLA_ABS_TN + 8u)];
    threadgroup float os[MLA_ABS_TM * (MLA_ABS_TN + 4u)];
    threadgroup ShvqMlaTgLut tgLut;
    qwg_mla_proj_uv_mma_impl<false>(uLat, kvBBlocks, attOut, p, tg, tid, sg, lane, us, ws, os, tgLut);
}

kernel void qwg_mla_proj_uv_mma_sub2(
    device const half*       uLat       [[buffer(0)]],
    device const char*       kvBBlocks  [[buffer(1)]],
    device half*             attOut     [[buffer(2)]],
    constant MLAAbsorbParams& p         [[buffer(3)]],
    uint3                    tg         [[threadgroup_position_in_grid]],
    uint                     tid        [[thread_index_in_threadgroup]],
    uint                     sg         [[simdgroup_index_in_threadgroup]],
    uint                     lane       [[thread_index_in_simdgroup]]
) {
    threadgroup half us[MLA_ABS_TM * (MLA_ABS_TK + 8u)];
    threadgroup half ws[MLA_ABS_TK * (MLA_ABS_TN + 8u)];
    threadgroup float os[MLA_ABS_TM * (MLA_ABS_TN + 4u)];
    threadgroup ShvqMlaTgLut tgLut;
    qwg_mla_proj_uv_mma_impl<true>(uLat, kvBBlocks, attOut, p, tg, tid, sg, lane, us, ws, os, tgLut);
}

#define MLA_MMA_KPS 72u

kernel void qwg_mla_dsa_sparse_attn_mma(
    device const half*              qAbs      [[buffer(0)]],
    device const half*              kPool     [[buffer(1)]],
    device const half*              vPool     [[buffer(2)]],
    device const int*               kTable    [[buffer(3)]],
    device const int*               vTable    [[buffer(4)]],
    device const int*               blockList [[buffer(5)]],
    device half*                    uLatOut   [[buffer(6)]],
    constant MLADSASparseAttnParams& p        [[buffer(7)]],
    device const half*              mlaQFull  [[buffer(8)]],
    device const half*              kPeCache  [[buffer(9)]],
    device const float*             attnSink  [[buffer(10)]],
    uint2                           tg        [[threadgroup_position_in_grid]],
    uint                            tid       [[thread_index_in_threadgroup]],
    uint                            sg        [[simdgroup_index_in_threadgroup]],
    uint                            lane      [[thread_index_in_simdgroup]]
) {
    const uint t = tg.x;
    const uint h0 = tg.y * MLA_MMA_HG;
    if (t >= p.T) return;

    threadgroup half  ks[MLA_MMA_TK * MLA_MMA_KS];
    threadgroup half  kps[MLA_MMA_TK * MLA_MMA_KPS];
    threadgroup float sbuf[MLA_MMA_HG * MLA_MMA_TK];
    threadgroup half  pbuf[MLA_MMA_HG * MLA_MMA_TK];
    threadgroup float corrH[MLA_MMA_HG];
    threadgroup float mH[MLA_MMA_HG];
    threadgroup float lH[MLA_MMA_HG];
    threadgroup uint  blkIdx[2560];

    const uint halfRank = MLA_MMA_RANK / 2u;
    const uint seqLen = p.pos0 + t + 1u;
    const uint denseLim = p.topK + p.windowTokens;
    const uint slots = min(min(seqLen, denseLim), 2560u);
    const float scale = rsqrt(float(p.headDim));

    if (seqLen <= denseLim) {
        for (uint i = tid; i < slots; i += 128u) blkIdx[i] = i;
    } else {
        device const int* rowList = blockList + (size_t)t * p.listStride;
        for (uint i = tid; i < slots; i += 128u) blkIdx[i] = min(uint(max(rowList[i], 0)), seqLen - 1u);
    }
    if (tid < MLA_MMA_HG) {
        const float sk = (p.hasSink != 0u) ? attnSink[h0 + tid] : -INFINITY;
        mH[tid] = (sk > -1e29f) ? sk : -INFINITY;
        lH[tid] = (sk > -1e29f) ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint rl = qw_mla_row(lane), cl = qw_mla_col(lane);
    const uint sHb = sg & 1u, sTb = sg >> 1u;
    device const half* qBase = qAbs + ((size_t)t * p.qHeads + h0 + sHb * 8u) * MLA_MMA_RANK;
    device const half* qRopeBase = mlaQFull + (size_t)t * p.qFullStride + p.qRopeOffset + (size_t)(h0 + sHb * 8u) * p.ropeDim;
    simdgroup_float8x8 oacc[2][16];
    #pragma unroll
    for (uint a = 0u; a < 2u; ++a) {
        #pragma unroll
        for (uint b = 0u; b < 16u; ++b) oacc[a][b] = simdgroup_float8x8(0.0f);
    }

    for (uint s0 = 0u; s0 < slots; s0 += MLA_MMA_TK) {
        for (uint i = tid; i < MLA_MMA_TK * (MLA_MMA_RANK / 4u); i += 128u) {
            const uint r = i / (MLA_MMA_RANK / 4u);
            const uint c4 = (i - r * (MLA_MMA_RANK / 4u)) * 4u;
            const uint slot = s0 + r;
            half4 v = half4(0.0h);
            if (slot < slots) {
                const uint j = blkIdx[slot];
                const uint pageSlot = j >> 8u, pageOff = j & 255u;
                if (c4 < halfRank) {
                    const uint pg = uint(kTable[pageSlot]);
                    v = qw_mla_load_kv_half4((device const uchar*)kPool, pg, pageOff, c4, p.kvMode);
                } else {
                    const uint pg = uint(vTable[pageSlot]);
                    v = qw_mla_load_kv_half4((device const uchar*)vPool, pg, pageOff, c4 - halfRank, p.kvMode);
                }
            }
            *(threadgroup half4*)(ks + r * MLA_MMA_KS + c4) = v;
        }
        if (p.ropeDim == 64u) {
            for (uint i = tid; i < MLA_MMA_TK * 16u; i += 128u) {
                const uint r = i >> 4u, c4 = (i & 15u) * 4u;
                const uint slot = s0 + r;
                const half4 rv = (slot < slots)
                    ? *(device const half4*)(kPeCache + (size_t)blkIdx[slot] * 64u + c4)
                    : half4(0.0h);
                *(threadgroup half4*)(kps + r * MLA_MMA_KPS + c4) = rv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            simdgroup_float8x8 sacc = simdgroup_float8x8(0.0f);
            for (uint d = 0u; d < MLA_MMA_RANK / 8u; ++d) {
                simdgroup_half8x8 qf, kf;
                simdgroup_load(qf, qBase + d * 8u, MLA_MMA_RANK);
                simdgroup_load(kf, ks + sTb * 8u * MLA_MMA_KS + d * 8u, MLA_MMA_KS, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sacc, qf, kf, sacc);
            }
            if (p.ropeDim == 64u) {
                for (uint d = 0u; d < 8u; ++d) {
                    simdgroup_half8x8 qf, kf;
                    simdgroup_load(qf, qRopeBase + d * 8u, 64u);
                    simdgroup_load(kf, kps + sTb * 8u * MLA_MMA_KPS + d * 8u, MLA_MMA_KPS, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(sacc, qf, kf, sacc);
                }
            }
            auto e = sacc.thread_elements();
            const uint hr = sHb * 8u + rl;
            #pragma unroll
            for (uint i = 0u; i < 2u; ++i) {
                const uint col = sTb * 8u + cl + i;
                sbuf[hr * MLA_MMA_TK + col] = (s0 + col < slots) ? e[i] * scale : -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            const uint hr = tid >> 3u, c0 = (tid & 7u) * 2u;
            const float a = sbuf[hr * MLA_MMA_TK + c0];
            const float b = sbuf[hr * MLA_MMA_TK + c0 + 1u];
            float mx = max(a, b);
            mx = max(mx, simd_shuffle_xor(mx, 1u));
            mx = max(mx, simd_shuffle_xor(mx, 2u));
            mx = max(mx, simd_shuffle_xor(mx, 4u));
            const float mOld = mH[hr];
            const float mNew = max(mOld, mx);
            const float pa = exp(a - mNew), pb = exp(b - mNew);
            pbuf[hr * MLA_MMA_TK + c0] = half(pa);
            pbuf[hr * MLA_MMA_TK + c0 + 1u] = half(pb);
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
                    simdgroup_load(pf[a][k], pbuf + a * 8u * MLA_MMA_TK + k * 8u, MLA_MMA_TK);
            }
            #pragma unroll
            for (uint b = 0u; b < 16u; ++b) {
                const uint col = sg * 128u + b * 8u;
                simdgroup_half8x8 kf0, kf1;
                simdgroup_load(kf0, ks + col, MLA_MMA_KS);
                simdgroup_load(kf1, ks + 8u * MLA_MMA_KS + col, MLA_MMA_KS);
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
        const uint h = h0 + a * 8u + rl;
        const float l = lH[a * 8u + rl];
        const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        device half* dst = uLatOut + ((size_t)t * p.qHeads + h) * MLA_MMA_RANK;
        #pragma unroll
        for (uint b = 0u; b < 16u; ++b) {
            auto e = oacc[a][b].thread_elements();
            const uint col = sg * 128u + b * 8u + cl;
            *(device half2*)(dst + col) = half2(half(e[0] * inv), half(e[1] * inv));
        }
    }
}
