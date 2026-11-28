// qwattn_sink_rel_mma.metal — Apple9+ simdgroup_half8x8 dense and sparse prefill
// with optional attention sink, banded relative position bias, and output gate.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "qw_attn_sink_rel.metalh"
#include "inkling2/qw_ink2_common.metalh"

using namespace metal;

#define QW_SR_MMA_BM         32u
#define QW_SR_MMA_BK         32u
#define QW_SR_MMA_HDIM       128u
#define QW_SR_MMA_KV_STRIDE  136u
#define QW_SR_MMA_REL_STRIDE 68u

static inline uint qw_sr_mma_row(uint l) {
    return ((l >> 4) & 1u) * 4u + ((l >> 2) & 1u) * 2u + ((l >> 1) & 1u);
}

static inline uint qw_sr_mma_col(uint l) {
    return ((l >> 3) & 1u) * 4u + (l & 1u) * 2u;
}

static inline half qw_sr_clamp_bf16(bfloat v) {
    return half(clamp(float(v), -65504.0f, 65504.0f));
}

METAL_FUNC void qw_sr_mma_load_q_tile(
    device const bfloat* qBF16, uint rowBase, uint head, uint nH, uint T,
    uint colBase, uint rl, uint cl, thread simdgroup_half8x8* qf
) {
    for (uint bm = 0u; bm < 2u; bm++) {
        const uint r = rowBase + bm * 8u + rl;
        auto qe = qf[bm].thread_elements();
        if (r < T) {
            device const bfloat* src = qBF16 + (size_t(r) * nH + head) * QW_SR_MMA_HDIM + colBase + cl;
            qe[0] = qw_sr_clamp_bf16(src[0]);
            qe[1] = qw_sr_clamp_bf16(src[1]);
        } else {
            qe[0] = half(0.0h);
            qe[1] = half(0.0h);
        }
        qf[bm].thread_elements() = qe;
    }
}

METAL_FUNC void qw_sr_mma_stage_proj(
    device const bfloat* relProj, threadgroup bfloat* relStage, uint tid, uint kThreads,
    uint dr0, uint dRel, uint relExtent, int tileDStart, int tileDEnd
) {
    for (uint idx = tid; idx < 256u; idx += kThreads) {
        const uint drLoc = idx >> 4u, u4 = (idx & 15u) * 4u, dr = dr0 + drLoc;
        const int dist0 = tileDStart + int(u4);
        threadgroup ushort* dst = (threadgroup ushort*)(relStage + drLoc * QW_SR_MMA_REL_STRIDE + u4);
        if (dr < dRel && dist0 >= 0 && dist0 + 3 <= tileDEnd && ((dist0 & 3) == 0)
            && ((size_t(relProj + size_t(dr) * relExtent + size_t(dist0)) & 7u) == 0u)) {
            *(threadgroup ushort4*)dst = *(device const ushort4*)(relProj + size_t(dr) * relExtent + size_t(dist0));
        } else {
            for (uint j = 0u; j < 4u; ++j) {
                const int dist = dist0 + int(j);
                dst[j] = (dr < dRel && dist >= 0 && dist <= tileDEnd)
                    ? as_type<ushort>(relProj[size_t(dr) * relExtent + size_t(dist)]) : 0u;
            }
        }
    }
}

METAL_FUNC uint2 qw_sr_mma_sparse_range(
    device const int* myList, uint rawCnt, uint minKey, uint completeLimit
) {
    uint bBegin = 0u;
    if (minKey >= 4u) {
        const uint minBlk = minKey >> 2u;
        uint lo = 0u, hi = rawCnt;
        while (lo < hi) {
            const uint mid = (lo + hi) >> 1u;
            if (uint(myList[mid]) < minBlk) lo = mid + 1u; else hi = mid;
        }
        bBegin = lo;
    }
    uint lo = bBegin, hi = rawCnt;
    while (lo < hi) {
        const uint mid = (lo + hi) >> 1u;
        if (uint(myList[mid]) < completeLimit) lo = mid + 1u; else hi = mid;
    }
    return uint2(bBegin, lo);
}

METAL_FUNC uint qw_sr_mma_slot_key(
    uint slotIdx, uint bBegin, uint uCount, uint tailStart, uint totalSlots, device const int* myList
) {
    if (slotIdx >= totalSlots) return 0xFFFFFFFFu;
    return (slotIdx < uCount * 4u)
        ? (uint(myList[bBegin + (slotIdx >> 2u)]) * 4u + (slotIdx & 3u))
        : (tailStart + (slotIdx - uCount * 4u));
}

METAL_FUNC void qw_sr_mma_epilogue(
    thread simdgroup_float8x8 (*otile)[16], thread float* lrow,
    device const bfloat* gateBF16, device bfloat* outBF16, constant QWSinkRelAttnParams& p,
    uint rowBase, uint head, uint rl, uint cl
) {
    for (uint bm = 0u; bm < 2u; bm++) {
        const uint rOut = rowBase + bm * 8u + rl;
        if (rOut >= p.T) continue;
        const float rcp = (lrow[bm] > 0.0f) ? (1.0f / lrow[bm]) : 0.0f;
        for (uint d = 0u; d < 16u; d++) {
            const size_t oIdx = (size_t(rOut) * p.nH + head) * QW_SR_MMA_HDIM + d * 8u + cl;
            const auto e = otile[bm][d].thread_elements();
            float2 o2 = float2(e[0], e[1]) * rcp;
            if (p.hasOutGate != 0u) {
                const float2 g2 = float2(*(device const bfloat2*)(gateBF16 + oIdx));
                o2.x *= qw_ink2_sigmoid(g2.x);
                o2.y *= qw_ink2_sigmoid(g2.y);
            }
            *(device bfloat2*)(outBF16 + oIdx) = bfloat2(qw_ink2_bf16(o2.x), qw_ink2_bf16(o2.y));
        }
    }
}

template <uint MODE>
METAL_FUNC void qwg_attn_prefill_sink_rel_mma_body(
    device const bfloat* qBF16, device const bfloat* rBF16, device const bfloat* relProj,
    device const bfloat* sinkBF16, device const bfloat* gateBF16,
    device const uchar* kCache, device const uchar* vCache,
    device const uchar* kMeta, device const uchar* vMeta,
    device bfloat* outBF16, constant QWSinkRelAttnParams& p,
    uint tid, uint sgi, uint lane, uint2 tg,
    threadgroup half* kvblk, threadgroup half* vblk, threadgroup bfloat* relStage
) {
    const uint gqaRatio = p.nH / p.nKV, kThreads = 64u * gqaRatio;
    const uint tile0 = tg.x * QW_SR_MMA_BM, kvh = tg.y;
    if (tile0 >= p.T || kvh >= p.nKV) return;

    const uint sgiH = sgi >> 1u, sgiLocal = sgi & 1u;
    const uint head = kvh * gqaRatio + sgiH, rowBase = tile0 + sgiLocal * 16u;
    const uint kEnd = min(p.posOffset + min(tile0 + QW_SR_MMA_BM, p.T), p.kvLen);
    const uint kStartRaw = (p.sliding > 0u && (p.posOffset + tile0 + 1u) > p.sliding)
        ? ((p.posOffset + tile0 + 1u) - p.sliding) : 0u;
    const uint nStart = kStartRaw & ~(QW_SR_MMA_BK - 1u);

    constexpr float log2e = 1.4426950408889634f;
    const float qkScale2 = rsqrt(float(QW_SR_MMA_HDIM)) * log2e;
    const uint rl = qw_sr_mma_row(lane), cl = qw_sr_mma_col(lane);

    simdgroup_float8x8 otile[2][16];
    for (uint bm = 0u; bm < 2u; bm++) {
        for (uint d = 0u; d < 16u; d++) otile[bm][d] = simdgroup_float8x8(0.0f);
    }
    float mrow[2], lrow[2];
    for (uint bm = 0u; bm < 2u; bm++) {
        const uint rLoc = rowBase + bm * 8u + rl;
        if (p.hasSink != 0u && rLoc < p.T) {
            mrow[bm] = float(sinkBF16[size_t(rLoc) * p.nH + head]) * log2e;
            lrow[bm] = 1.0f;
        } else {
            mrow[bm] = -1e30f;
            lrow[bm] = 0.0f;
        }
    }

    for (uint n0 = nStart; n0 < kEnd; n0 += QW_SR_MMA_BK) {
        if (n0 != nStart) threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (MODE == 0u) {
            for (uint e = tid; e < 512u; e += kThreads) {
                const uint kr = e >> 4u, vecIdx8 = e & 15u, absKey = n0 + kr;
                const uint ob = kr * QW_SR_MMA_KV_STRIDE + (vecIdx8 << 3u);
                uint4 k8(0u), v8(0u);
                if (absKey >= kStartRaw && absKey < kEnd) {
                    const uint slot = (p.sliding > 0u) ? (absKey % p.ringCap) : absKey;
                    const size_t off = (size_t(slot) * p.nKV + kvh) * QW_SR_MMA_HDIM + (vecIdx8 << 3u);
                    k8 = *(device const uint4*)(((device const half*)kCache) + off);
                    v8 = *(device const uint4*)(((device const half*)vCache) + off);
                }
                *(threadgroup uint4*)(kvblk + ob) = k8;
                *(threadgroup uint4*)(vblk + ob) = v8;
            }
        } else {
            for (uint e = tid; e < 1024u; e += kThreads) {
                const uint kr = e >> 5u, vecIdx = e & 31u, absKey = n0 + kr;
                const uint ob = kr * QW_SR_MMA_KV_STRIDE + (vecIdx << 2u);
                half4 k4(0.0h), v4(0.0h);
                if (absKey >= kStartRaw && absKey < kEnd) {
                    const uint slot = (p.sliding > 0u) ? (absKey % p.ringCap) : absKey;
                    k4 = qwg_sink_rel_read_kv4<MODE>(vecIdx, slot, kvh, p.nKV, QW_SR_MMA_HDIM, kCache, kMeta);
                    v4 = qwg_sink_rel_read_kv4<MODE>(vecIdx, slot, kvh, p.nKV, QW_SR_MMA_HDIM, vCache, vMeta);
                }
                *(threadgroup half4*)(kvblk + ob) = k4;
                *(threadgroup half4*)(vblk + ob) = v4;
            }
        }

        const uint tileMinQ = p.posOffset + tile0, tileMaxK = n0 + 31u;
        const uint tileMinDist = (tileMinQ > tileMaxK) ? (tileMinQ - tileMaxK) : 0u;
        const bool tileHasRel = (p.hasRelBias != 0u) && (p.dRel > 0u) && (p.relExtent > 0u)
                              && (tileMinDist < p.relExtent);
        const int tileDStart = int(tileMinDist);
        const int tileDEnd = min(int(p.relExtent) - 1, int(p.posOffset + min(tile0 + QW_SR_MMA_BM, p.T)) - 1 - int(n0));
        if (tileHasRel) {
            qw_sr_mma_stage_proj(relProj, relStage, tid, kThreads, 0u, p.dRel, p.relExtent, tileDStart, tileDEnd);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 stile[2][4];
        for (uint bm = 0u; bm < 2u; bm++) {
            for (uint ik = 0u; ik < 4u; ik++) stile[bm][ik] = simdgroup_float8x8(0.0f);
        }
        for (uint d = 0u; d < 16u; d++) {
            simdgroup_half8x8 qf[2], kf[4];
            qw_sr_mma_load_q_tile(qBF16, rowBase, head, p.nH, p.T, d * 8u, rl, cl, qf);
            for (uint ik = 0u; ik < 4u; ik++) {
                simdgroup_load(kf[ik], kvblk + ik * 8u * QW_SR_MMA_KV_STRIDE + d * 8u,
                               QW_SR_MMA_KV_STRIDE, ulong2(0, 0), true);
            }
            for (uint bm = 0u; bm < 2u; bm++) {
                for (uint ik = 0u; ik < 4u; ik++) {
                    simdgroup_multiply_accumulate(stile[bm][ik], qf[bm], kf[ik], stile[bm][ik]);
                }
            }
        }

        if (tileHasRel) {
            for (uint bm = 0u; bm < 2u; bm++) {
                const uint qLoc = rowBase + bm * 8u + rl;
                const uint absRR = p.posOffset + qLoc;
                const device bfloat* rRow = (qLoc < p.T) ? (rBF16 + (size_t(qLoc) * p.nH + head) * p.dRel) : nullptr;
                for (uint ik = 0u; ik < 4u; ik++) {
                    auto se = stile[bm][ik].thread_elements();
                    se[0] *= qkScale2; se[1] *= qkScale2;
                    for (uint j = 0u; j < 2u; j++) {
                        const uint absCC = n0 + ik * 8u + cl + j;
                        if (rRow != nullptr && absRR >= absCC && (absRR - absCC) < p.relExtent) {
                            const int dist = int(absRR - absCC);
                            if (dist >= tileDStart && dist <= tileDEnd) {
                                float bias = 0.0f;
                                const uint lim0 = min(p.dRel, 16u);
                                for (uint dr = 0u; dr < lim0; dr++) {
                                    bias = fma(float(rRow[dr]),
                                               float(relStage[dr * QW_SR_MMA_REL_STRIDE + uint(dist - tileDStart)]),
                                               bias);
                                }
                                for (uint dr = 16u; dr < p.dRel; dr++) {
                                    bias = fma(float(rRow[dr]),
                                               float(relProj[size_t(dr) * p.relExtent + size_t(dist)]),
                                               bias);
                                }
                                se[j] += bias * log2e;
                            }
                        }
                    }
                    stile[bm][ik].thread_elements() = se;
                }
            }
        } else {
            for (uint bm = 0u; bm < 2u; bm++) {
                for (uint ik = 0u; ik < 4u; ik++) {
                    auto se = stile[bm][ik].thread_elements();
                    se[0] *= qkScale2; se[1] *= qkScale2;
                    stile[bm][ik].thread_elements() = se;
                }
            }
        }

        const uint maxRowPlus1 = p.posOffset + min(rowBase + 16u, p.T);
        const bool isUnmasked = (rowBase + 16u <= p.T)
            && (n0 + QW_SR_MMA_BK <= p.posOffset + rowBase) && (n0 + QW_SR_MMA_BK <= p.kvLen)
            && (p.sliding == 0u || maxRowPlus1 <= n0 + p.sliding);

        simdgroup_half8x8 pt[2][4];
        for (uint bm = 0u; bm < 2u; bm++) {
            float mnew = mrow[bm];
            const uint qLoc = rowBase + bm * 8u + rl;
            const uint absRR = p.posOffset + qLoc;
            for (uint ik = 0u; ik < 4u; ik++) {
                auto se = stile[bm][ik].thread_elements();
                if (!isUnmasked) {
                    const uint col0 = n0 + ik * 8u + cl;
                    const bool inW0 = (p.sliding == 0u) || (absRR < col0 + p.sliding);
                    const bool inW1 = (p.sliding == 0u) || (absRR < (col0 + 1u) + p.sliding);
                    if (!(qLoc < p.T && col0 <= absRR && col0 < p.kvLen && inW0)) se[0] = -1e30f;
                    if (!(qLoc < p.T && (col0 + 1u) <= absRR && (col0 + 1u) < p.kvLen && inW1)) se[1] = -1e30f;
                    stile[bm][ik].thread_elements() = se;
                }
                mnew = max(mnew, max(se[0], se[1]));
            }
            mnew = max(mnew, simd_shuffle_xor(mnew, ushort(1)));
            mnew = max(mnew, simd_shuffle_xor(mnew, ushort(8)));

            float psum = 0.0f;
            for (uint ik = 0u; ik < 4u; ik++) {
                const auto se = stile[bm][ik].thread_elements();
                const float x0 = (isUnmasked || se[0] > -1e29f) ? fast::exp2(se[0] - mnew) : 0.0f;
                const float x1 = (isUnmasked || se[1] > -1e29f) ? fast::exp2(se[1] - mnew) : 0.0f;
                psum += x0 + x1;
                auto pe = pt[bm][ik].thread_elements();
                pe[0] = half(x0); pe[1] = half(x1);
                pt[bm][ik].thread_elements() = pe;
            }
            psum += simd_shuffle_xor(psum, ushort(1));
            psum += simd_shuffle_xor(psum, ushort(8));

            const bool grew = (mnew > mrow[bm]);
            const float corr = (grew && mrow[bm] > -1e29f) ? fast::exp2(mrow[bm] - mnew) : (grew ? 0.0f : 1.0f);
            lrow[bm] = lrow[bm] * corr + psum;
            if (grew && mrow[bm] > -1e29f) {
                for (uint d = 0u; d < 16u; d++) {
                    auto oe = otile[bm][d].thread_elements();
                    oe[0] *= corr; oe[1] *= corr;
                    otile[bm][d].thread_elements() = oe;
                }
            }
            mrow[bm] = mnew;
        }

        for (uint ik = 0u; ik < 4u; ik++) {
            for (uint d = 0u; d < 16u; d++) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf, vblk + ik * 8u * QW_SR_MMA_KV_STRIDE + d * 8u, QW_SR_MMA_KV_STRIDE);
                simdgroup_multiply_accumulate(otile[0][d], pt[0][ik], vf, otile[0][d]);
                simdgroup_multiply_accumulate(otile[1][d], pt[1][ik], vf, otile[1][d]);
            }
        }
    }

    qw_sr_mma_epilogue(otile, lrow, gateBF16, outBF16, p, rowBase, head, rl, cl);
}

#define QW_SR_MMA_PREFILL_ENTRY(SFX, MODE) \
kernel void qwg_attn_prefill_sink_rel_mma##SFX( \
    device const bfloat* qBF16 [[buffer(0)]], device const bfloat* rBF16 [[buffer(1)]], \
    device const bfloat* relProj [[buffer(2)]], device const bfloat* sinkBF16 [[buffer(3)]], \
    device const bfloat* gateBF16 [[buffer(4)]], device const uchar* kCache [[buffer(5)]], \
    device const uchar* vCache [[buffer(6)]], device const uchar* kMeta [[buffer(7)]], \
    device const uchar* vMeta [[buffer(8)]], device bfloat* outBF16 [[buffer(9)]], \
    constant QWSinkRelAttnParams& p [[buffer(10)]], \
    uint tid [[thread_index_in_threadgroup]], uint sgi [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint2 tg [[threadgroup_position_in_grid]] \
) { \
    threadgroup half kvblk[QW_SR_MMA_BK * QW_SR_MMA_KV_STRIDE]; \
    threadgroup half vblk[QW_SR_MMA_BK * QW_SR_MMA_KV_STRIDE]; \
    threadgroup bfloat relStage[16u * QW_SR_MMA_REL_STRIDE]; \
    qwg_attn_prefill_sink_rel_mma_body<MODE>( \
        qBF16, rBF16, relProj, sinkBF16, gateBF16, kCache, vCache, kMeta, vMeta, outBF16, \
        p, tid, sgi, lane, tg, &kvblk[0], &vblk[0], &relStage[0]); \
}
QW_SR_MMA_PREFILL_ENTRY(_f16, 0u)
QW_SR_MMA_PREFILL_ENTRY(_q8, 1u)
QW_SR_MMA_PREFILL_ENTRY(_q4, 2u)
#undef QW_SR_MMA_PREFILL_ENTRY

struct QWSinkRelSparseMetaMMA {
    uint listStride;
    uint rowIndexed;
    uint pad0;
    uint pad1;
};

template <uint MODE>
METAL_FUNC void qwg_attn_prefill_sink_rel_sparse_mma_body(
    device const bfloat* qBF16, device const bfloat* rBF16, device const bfloat* relProj,
    device const bfloat* sinkBF16, device const bfloat* gateBF16,
    device const uchar* kCache, device const uchar* vCache,
    device const uchar* kMeta, device const uchar* vMeta,
    device bfloat* outBF16, constant QWSinkRelAttnParams& p,
    device const int* blockList, device const uint* blockCount, constant QWSinkRelSparseMetaMMA& sp,
    uint tid, uint sgi, uint lane, uint2 tg,
    threadgroup half* kvblk, threadgroup half* vblk, threadgroup uint* absKeySh
) {
    const uint gqaRatio = p.nH / p.nKV, kThreads = 64u * gqaRatio;
    const uint tile0 = tg.x * QW_SR_MMA_BM, kvh = tg.y;
    if (tile0 >= p.T || kvh >= p.nKV) return;

    const uint sgiH = sgi >> 1u, sgiLocal = sgi & 1u;
    const uint head = kvh * gqaRatio + sgiH, rowBase = tile0 + sgiLocal * 16u;
    const uint kEnd = min(p.posOffset + min(tile0 + QW_SR_MMA_BM, p.T), p.kvLen);
    const uint kStartRaw = (p.sliding > 0u && (p.posOffset + tile0 + 1u) > p.sliding)
        ? ((p.posOffset + tile0 + 1u) - p.sliding) : 0u;

    const uint listRow = (sp.rowIndexed != 0u) ? min(tile0 + 31u, p.T - 1u) : tg.x;
    const device int* myList = blockList + size_t(listRow) * sp.listStride;
    const uint totalCompleteBlocks = kEnd >> 2u;
    const uint2 range = qw_sr_mma_sparse_range(myList, blockCount[listRow], kStartRaw, totalCompleteBlocks);
    const uint bBegin = range.x, uCount = range.y - range.x;
    const uint tailStart = totalCompleteBlocks * 4u;
    const uint totalSlots = uCount * 4u + ((kEnd > tailStart) ? (kEnd - tailStart) : 0u);

    constexpr float log2e = 1.4426950408889634f;
    const float qkScale2 = rsqrt(float(QW_SR_MMA_HDIM)) * log2e;
    const uint rl = qw_sr_mma_row(lane), cl = qw_sr_mma_col(lane);

    float rReg[2][16];
    if (p.hasRelBias != 0u && p.dRel > 0u && p.dRel <= 16u) {
        for (uint bm = 0u; bm < 2u; bm++) {
            const uint qLoc = rowBase + bm * 8u + rl;
            const device bfloat* rPtr = (qLoc < p.T) ? (rBF16 + (size_t(qLoc) * p.nH + head) * p.dRel) : nullptr;
            for (uint dr = 0u; dr < 16u; dr++) {
                rReg[bm][dr] = (rPtr != nullptr && dr < p.dRel) ? float(rPtr[dr]) : 0.0f;
            }
        }
    }

    simdgroup_float8x8 otile[2][16];
    for (uint bm = 0u; bm < 2u; bm++) {
        for (uint d = 0u; d < 16u; d++) otile[bm][d] = simdgroup_float8x8(0.0f);
    }
    float mrow[2], lrow[2];
    for (uint bm = 0u; bm < 2u; bm++) {
        const uint rLoc = rowBase + bm * 8u + rl;
        const bool hasS = (p.hasSink != 0u && rLoc < p.T);
        mrow[bm] = hasS ? float(sinkBF16[size_t(rLoc) * p.nH + head]) * log2e : -1e30f;
        lrow[bm] = hasS ? 1.0f : 0.0f;
    }

    for (uint s0 = 0u; s0 < totalSlots; s0 += QW_SR_MMA_BK) {
        if (s0 != 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32u) absKeySh[tid] = qw_sr_mma_slot_key(s0 + tid, bBegin, uCount, tailStart, totalSlots, myList);

        const uint loadN = (MODE == 0u) ? 512u : 1024u;
        for (uint e = tid; e < loadN; e += kThreads) {
            const uint kr = (MODE == 0u) ? (e >> 4u) : (e >> 5u);
            const uint vecIdx = (MODE == 0u) ? (e & 15u) : (e & 31u);
            const uint absKey = qw_sr_mma_slot_key(s0 + kr, bBegin, uCount, tailStart, totalSlots, myList);
            const uint ob = kr * QW_SR_MMA_KV_STRIDE + ((MODE == 0u) ? (vecIdx << 3u) : (vecIdx << 2u));
            const bool validK = (absKey >= kStartRaw && absKey < kEnd);
            const uint slot = validK ? ((p.sliding > 0u) ? (absKey % p.ringCap) : absKey) : 0u;
            if constexpr (MODE == 0u) {
                const size_t off = (size_t(slot) * p.nKV + kvh) * QW_SR_MMA_HDIM + (vecIdx << 3u);
                *(threadgroup uint4*)(kvblk + ob) = validK ? *(device const uint4*)(((device const half*)kCache) + off) : uint4(0u);
                *(threadgroup uint4*)(vblk + ob) = validK ? *(device const uint4*)(((device const half*)vCache) + off) : uint4(0u);
            } else {
                *(threadgroup half4*)(kvblk + ob) = validK
                    ? qwg_sink_rel_read_kv4<MODE>(vecIdx, slot, kvh, p.nKV, QW_SR_MMA_HDIM, kCache, kMeta) : half4(0.0h);
                *(threadgroup half4*)(vblk + ob) = validK
                    ? qwg_sink_rel_read_kv4<MODE>(vecIdx, slot, kvh, p.nKV, QW_SR_MMA_HDIM, vCache, vMeta) : half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 stile[2][4];
        for (uint bm = 0u; bm < 2u; bm++) {
            for (uint ik = 0u; ik < 4u; ik++) stile[bm][ik] = simdgroup_float8x8(0.0f);
        }
        for (uint d = 0u; d < 16u; d++) {
            simdgroup_half8x8 qf[2], kf[4];
            qw_sr_mma_load_q_tile(qBF16, rowBase, head, p.nH, p.T, d * 8u, rl, cl, qf);
            for (uint ik = 0u; ik < 4u; ik++) {
                simdgroup_load(kf[ik], kvblk + ik * 8u * QW_SR_MMA_KV_STRIDE + d * 8u,
                               QW_SR_MMA_KV_STRIDE, ulong2(0, 0), true);
            }
            for (uint bm = 0u; bm < 2u; bm++) {
                for (uint ik = 0u; ik < 4u; ik++) {
                    simdgroup_multiply_accumulate(stile[bm][ik], qf[bm], kf[ik], stile[bm][ik]);
                }
            }
        }

        const uint stepCount = min(QW_SR_MMA_BK, totalSlots - s0);
        const uint stepMinKey = absKeySh[0], stepMaxKey = absKeySh[stepCount - 1u];
        const uint tileMinDist = (p.posOffset + tile0 > stepMaxKey) ? ((p.posOffset + tile0) - stepMaxKey) : 0u;
        const bool stepHasRel = (p.hasRelBias != 0u) && (p.dRel > 0u) && (p.relExtent > 0u) && (tileMinDist < p.relExtent);
        const uint maxRowPlus1 = p.posOffset + min(rowBase + 16u, p.T);
        const bool isUnmasked = (rowBase + 16u <= p.T)
            && (s0 + QW_SR_MMA_BK <= totalSlots) && (stepMaxKey <= p.posOffset + rowBase)
            && (stepMaxKey < p.kvLen) && (p.sliding == 0u || maxRowPlus1 <= stepMinKey + p.sliding);

        simdgroup_half8x8 pt[2][4];
        for (uint bm = 0u; bm < 2u; bm++) {
            float mnew = mrow[bm];
            const uint qLoc = rowBase + bm * 8u + rl;
            const uint absRR = p.posOffset + qLoc;
            for (uint ik = 0u; ik < 4u; ik++) {
                auto se = stile[bm][ik].thread_elements();
                se[0] *= qkScale2; se[1] *= qkScale2;
                if (!isUnmasked || stepHasRel) {
                    const uint colLocal = ik * 8u + cl;
                    for (uint j = 0u; j < 2u; j++) {
                        const uint slotC = s0 + colLocal + j;
                        const uint absCC = (slotC < totalSlots) ? absKeySh[colLocal + j] : 0xFFFFFFFFu;
                        const bool inWin = (p.sliding == 0u) || (absRR < absCC + p.sliding);
                        if (qLoc < p.T && slotC < totalSlots && absCC <= absRR && absCC < p.kvLen && inWin) {
                            if (stepHasRel && (absRR - absCC) < p.relExtent) {
                                const uint dist = absRR - absCC;
                                float bias = 0.0f;
                                const device bfloat* rPtr = (p.dRel > 16u)
                                    ? (rBF16 + (size_t(qLoc) * p.nH + head) * p.dRel) : nullptr;
                                for (uint dr = 0u; dr < p.dRel; ++dr) {
                                    const float rv = (p.dRel <= 16u) ? rReg[bm][dr] : float(rPtr[dr]);
                                    bias = fma(rv, float(relProj[size_t(dr) * p.relExtent + dist]), bias);
                                }
                                se[j] += bias * log2e;
                            }
                        } else {
                            se[j] = -1e30f;
                        }
                    }
                    stile[bm][ik].thread_elements() = se;
                }
                mnew = max(mnew, max(se[0], se[1]));
            }
            mnew = max(mnew, simd_shuffle_xor(mnew, ushort(1)));
            mnew = max(mnew, simd_shuffle_xor(mnew, ushort(8)));

            float psum = 0.0f;
            for (uint ik = 0u; ik < 4u; ik++) {
                const auto se = stile[bm][ik].thread_elements();
                const float x0 = (isUnmasked || se[0] > -1e29f) ? fast::exp2(se[0] - mnew) : 0.0f;
                const float x1 = (isUnmasked || se[1] > -1e29f) ? fast::exp2(se[1] - mnew) : 0.0f;
                psum += x0 + x1;
                auto pe = pt[bm][ik].thread_elements();
                pe[0] = half(x0); pe[1] = half(x1);
                pt[bm][ik].thread_elements() = pe;
            }
            psum += simd_shuffle_xor(psum, ushort(1));
            psum += simd_shuffle_xor(psum, ushort(8));

            const bool grew = (mnew > mrow[bm]);
            const float corr = (grew && mrow[bm] > -1e29f) ? fast::exp2(mrow[bm] - mnew) : (grew ? 0.0f : 1.0f);
            lrow[bm] = lrow[bm] * corr + psum;
            if (grew && mrow[bm] > -1e29f) {
                for (uint d = 0u; d < 16u; d++) {
                    auto oe = otile[bm][d].thread_elements();
                    oe[0] *= corr; oe[1] *= corr;
                    otile[bm][d].thread_elements() = oe;
                }
            }
            mrow[bm] = mnew;
        }

        for (uint ik = 0u; ik < 4u; ik++) {
            for (uint d = 0u; d < 16u; d++) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf, vblk + ik * 8u * QW_SR_MMA_KV_STRIDE + d * 8u, QW_SR_MMA_KV_STRIDE);
                simdgroup_multiply_accumulate(otile[0][d], pt[0][ik], vf, otile[0][d]);
                simdgroup_multiply_accumulate(otile[1][d], pt[1][ik], vf, otile[1][d]);
            }
        }
    }

    qw_sr_mma_epilogue(otile, lrow, gateBF16, outBF16, p, rowBase, head, rl, cl);
}

#define QW_SR_MMA_SPARSE_ENTRY(SFX, MODE) \
kernel void qwg_attn_prefill_sink_rel_sparse_mma##SFX( \
    device const bfloat* qBF16 [[buffer(0)]], device const bfloat* rBF16 [[buffer(1)]], \
    device const bfloat* relProj [[buffer(2)]], device const bfloat* sinkBF16 [[buffer(3)]], \
    device const bfloat* gateBF16 [[buffer(4)]], device const uchar* kCache [[buffer(5)]], \
    device const uchar* vCache [[buffer(6)]], device const uchar* kMeta [[buffer(7)]], \
    device const uchar* vMeta [[buffer(8)]], device bfloat* outBF16 [[buffer(9)]], \
    constant QWSinkRelAttnParams& p [[buffer(10)]], device const int* blockList [[buffer(11)]], \
    device const uint* blockCount [[buffer(12)]], constant QWSinkRelSparseMetaMMA& sp [[buffer(13)]], \
    uint tid [[thread_index_in_threadgroup]], uint sgi [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint2 tg [[threadgroup_position_in_grid]] \
) { \
    threadgroup half kvblk[QW_SR_MMA_BK * QW_SR_MMA_KV_STRIDE]; \
    threadgroup half vblk[QW_SR_MMA_BK * QW_SR_MMA_KV_STRIDE]; \
    threadgroup uint absKeySh[32]; \
    qwg_attn_prefill_sink_rel_sparse_mma_body<MODE>( \
        qBF16, rBF16, relProj, sinkBF16, gateBF16, kCache, vCache, kMeta, vMeta, outBF16, \
        p, blockList, blockCount, sp, tid, sgi, lane, tg, &kvblk[0], &vblk[0], &absKeySh[0]); \
}
QW_SR_MMA_SPARSE_ENTRY(_f16, 0u)
QW_SR_MMA_SPARSE_ENTRY(_q8, 1u)
QW_SR_MMA_SPARSE_ENTRY(_q4, 2u)
#undef QW_SR_MMA_SPARSE_ENTRY
