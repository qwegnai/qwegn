// qwattn_sink_rel_sparse.metal — Sparse NAX FlashAttention prefill and Split-KV GQA decode
// for LSA / e-means-4096 with optional sink, banded relative position bias, and output gate.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_attn_sink_rel.metalh"
#include "inkling2/qw_ink2_common.metalh"

using namespace metal;

#if !defined(QW_DISABLE_NAX)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#if !defined(__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__) || __ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ < 260200
#error "NAX cooperative_tensor kernels require -mmacosx-version-min=26.2 or newer (Apple MPP operand_index ABI)."
#endif
#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"

using namespace mlx::steel;

#define QW_SRS_BM        32u
#define QW_SRS_BK        32u
#define QW_SRS_KU        16u
#define QW_SRS_HDIM      128u
#define QW_SRS_KV_STRIDE 136u
#endif // !defined(QW_DISABLE_NAX)

struct QWSinkRelSparseMeta {
    uint listStride;
    uint rowIndexed;
    uint pad0;
    uint pad1;
};

METAL_FUNC uint2 qwg_sr_sparse_range(
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

METAL_FUNC uint qwg_sr_slot_key(
    uint slotIdx, uint bBegin, uint uCount, uint tailStart, uint totalSlots, device const int* myList
) {
    if (slotIdx >= totalSlots) return 0xFFFFFFFFu;
    return (slotIdx < uCount * 4u)
        ? (uint(myList[bBegin + (slotIdx >> 2u)]) * 4u + (slotIdx & 3u))
        : (tailStart + (slotIdx - uCount * 4u));
}

METAL_FUNC void qwg_sr_write_out(
    float4 o4, device const bfloat* gateBF16, device bfloat* outBF16, size_t oIdx, uint hasOutGate
) {
    if (hasOutGate != 0u) {
        const float4 g4 = float4(*(device const bfloat4*)(gateBF16 + oIdx));
        o4.x *= qw_ink2_sigmoid(g4.x); o4.y *= qw_ink2_sigmoid(g4.y);
        o4.z *= qw_ink2_sigmoid(g4.z); o4.w *= qw_ink2_sigmoid(g4.w);
    }
    *(device bfloat4*)(outBF16 + oIdx) = bfloat4(
        qw_ink2_bf16(o4.x), qw_ink2_bf16(o4.y), qw_ink2_bf16(o4.z), qw_ink2_bf16(o4.w));
}

#if !defined(QW_DISABLE_NAX)
METAL_FUNC void qwg_sr_load_q_frag(
    device const bfloat* qBF16, uint rowBase, uint head, uint nH, uint T,
    uint colBase, short2 sc, thread BaseNAXFrag::dtype_frag_t<half>& dst
) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
        const uint r = rowBase + uint(sc.y) + uint(i) * 8u;
        if (r < T) {
            const ushort4 raw = *(device const ushort4*)(
                qBF16 + (size_t(r) * nH + head) * QW_SRS_HDIM + colBase + uint(sc.x));
            dst[i * 4 + 0] = half(float(as_type<bfloat>(raw.x)));
            dst[i * 4 + 1] = half(float(as_type<bfloat>(raw.y)));
            dst[i * 4 + 2] = half(float(as_type<bfloat>(raw.z)));
            dst[i * 4 + 3] = half(float(as_type<bfloat>(raw.w)));
        } else {
            dst[i * 4 + 0] = half(0.0h); dst[i * 4 + 1] = half(0.0h);
            dst[i * 4 + 2] = half(0.0h); dst[i * 4 + 3] = half(0.0h);
        }
    }
}

template <uint MODE>
METAL_FUNC void qwg_attn_prefill_sink_rel_sparse_nax_body(
    device const bfloat* qBF16, device const bfloat* rBF16, device const bfloat* relProj,
    device const bfloat* sinkBF16, device const bfloat* gateBF16,
    device const uchar* kCache, device const uchar* vCache,
    device const uchar* kMeta, device const uchar* vMeta,
    device bfloat* outBF16, constant QWSinkRelAttnParams& p,
    device const int* blockList, device const uint* blockCount, constant QWSinkRelSparseMeta& sp,
    uint tid, uint sgi, uint lane, uint2 tg,
    threadgroup half* kvblk, threadgroup half* vblk, threadgroup uint* absKeySh
) {
    (void)lane;
    const uint gqaRatio = p.nH / p.nKV, kThreads = 64u * gqaRatio;
    const uint tile0 = tg.x * QW_SRS_BM, kvh = tg.y;
    if (tile0 >= p.T || kvh >= p.nKV) return;

    const uint sgiH = sgi >> 1u, sgiLocal = sgi & 1u;
    const uint head = kvh * gqaRatio + sgiH, rowBase = tile0 + sgiLocal * QW_SRS_KU;
    const uint kEnd = min(p.posOffset + min(tile0 + QW_SRS_BM, p.T), p.kvLen);
    const uint kStartRaw = (p.sliding > 0u && (p.posOffset + tile0 + 1u) > p.sliding)
        ? ((p.posOffset + tile0 + 1u) - p.sliding) : 0u;

    const uint listRow = (sp.rowIndexed != 0u) ? min(tile0 + 31u, p.T - 1u) : tg.x;
    const device int* myList = blockList + size_t(listRow) * sp.listStride;
    const uint totalCompleteBlocks = kEnd >> 2u;
    const uint2 range = qwg_sr_sparse_range(myList, blockCount[listRow], kStartRaw, totalCompleteBlocks);
    const uint bBegin = range.x, uCount = range.y - range.x;
    const uint tailStart = totalCompleteBlocks * 4u;
    const uint totalSlots = uCount * 4u + ((kEnd > tailStart) ? (kEnd - tailStart) : 0u);

    constexpr float log2e = 1.4426950408889634f;
    const float qkScale2 = rsqrt(float(QW_SRS_HDIM)) * log2e;
    const short2 sc = BaseNAXFrag::get_coord();
    const short sm = sc.y, sn = sc.x;

    NAXTile<half, 1, 1> qf[8];
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < 8; id++) {
        qwg_sr_load_q_frag(qBF16, rowBase, head, p.nH, p.T, uint(id) * QW_SRS_KU, sc, qf[id].frag_at(0, 0));
    }

    bfloat rReg[2][16];
    if (p.hasRelBias != 0u && p.dRel > 0u && p.dRel <= 16u) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            const uint qLoc = rowBase + uint(sm) + uint(i) * 8u;
            const device bfloat* rPtr = (qLoc < p.T) ? (rBF16 + (size_t(qLoc) * p.nH + head) * p.dRel) : nullptr;
            for (uint dr = 0u; dr < 16u; ++dr) {
                rReg[i][dr] = (rPtr != nullptr && dr < p.dRel) ? rPtr[dr] : bfloat(0.0f);
            }
        }
    }

    NAXTile<float, 1, 8> otile;
    otile.clear();
    float mrow[2], lrow[2];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
        const uint rLoc = rowBase + uint(sm) + uint(i) * 8u;
        const bool hasS = (p.hasSink != 0u && rLoc < p.T);
        mrow[i] = hasS ? float(sinkBF16[size_t(rLoc) * p.nH + head]) * log2e : -1e30f;
        lrow[i] = hasS ? 1.0f : 0.0f;
    }

    for (uint s0 = 0u; s0 < totalSlots; s0 += QW_SRS_BK) {
        if (s0 != 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
        NAXTile<half, 1, 2> pt;
        if (tid < 32u) absKeySh[tid] = qwg_sr_slot_key(s0 + tid, bBegin, uCount, tailStart, totalSlots, myList);

        const uint loadN = (MODE == 0u) ? 512u : 1024u;
        for (uint e = tid; e < loadN; e += kThreads) {
            const uint kr = (MODE == 0u) ? (e >> 4u) : (e >> 5u);
            const uint vecIdx = (MODE == 0u) ? (e & 15u) : (e & 31u);
            const uint absKey = qwg_sr_slot_key(s0 + kr, bBegin, uCount, tailStart, totalSlots, myList);
            const uint ob = kr * QW_SRS_KV_STRIDE + ((MODE == 0u) ? (vecIdx << 3u) : (vecIdx << 2u));
            const bool validK = (absKey >= kStartRaw && absKey < kEnd);
            const uint slot = validK ? ((p.sliding > 0u) ? (absKey % p.ringCap) : absKey) : 0u;
            if constexpr (MODE == 0u) {
                const size_t off = (size_t(slot) * p.nKV + kvh) * QW_SRS_HDIM + (vecIdx << 3u);
                *(threadgroup uint4*)(kvblk + ob) = validK ? *(device const uint4*)(((device const half*)kCache) + off) : uint4(0u);
                *(threadgroup uint4*)(vblk + ob) = validK ? *(device const uint4*)(((device const half*)vCache) + off) : uint4(0u);
            } else {
                *(threadgroup half4*)(kvblk + ob) = validK
                    ? qwg_sink_rel_read_kv4<MODE>(vecIdx, slot, kvh, p.nKV, QW_SRS_HDIM, kCache, kMeta) : half4(0.0h);
                *(threadgroup half4*)(vblk + ob) = validK
                    ? qwg_sink_rel_read_kv4<MODE>(vecIdx, slot, kvh, p.nKV, QW_SRS_HDIM, vCache, vMeta) : half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        NAXTile<float, 1, 2> stile;
        stile.clear();
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < 8; id++) {
            NAXTile<half, 2, 1> kt;
            kt.template load<half, QW_SRS_KV_STRIDE, 1>(kvblk + uint(id) * QW_SRS_KU);
            BaseNAXFrag::mma(stile.frag_at(0, 0), stile.frag_at(0, 1), qf[id].frag_at(0, 0), metal::false_type{},
                             kt.frag_at(0, 0), kt.frag_at(1, 0), metal::true_type{});
        }
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < 2; ik++) {
            thread auto& fg = stile.frag_at(0, ik);
            STEEL_PRAGMA_UNROLL
            for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) fg[loc] *= qkScale2;
        }

        const uint stepCount = min(QW_SRS_BK, totalSlots - s0);
        const uint stepMinKey = absKeySh[0], stepMaxKey = absKeySh[stepCount - 1u];
        const uint tileMinQ = p.posOffset + tile0;
        const uint tileMinDist = (tileMinQ > stepMaxKey) ? (tileMinQ - stepMaxKey) : 0u;
        const bool stepHasRel = (p.hasRelBias != 0u) && (p.dRel > 0u) && (p.relExtent > 0u) && (tileMinDist < p.relExtent);
        const uint maxRowPlus1 = p.posOffset + min(rowBase + 16u, p.T);
        const bool isUnmasked = (rowBase + 16u <= p.T)
            && (s0 + QW_SRS_BK <= totalSlots) && (stepMaxKey <= p.posOffset + rowBase)
            && (stepMaxKey < p.kvLen) && (p.sliding == 0u || maxRowPlus1 <= stepMinKey + p.sliding);

        if (!isUnmasked || stepHasRel) {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < 2; ik++) {
                thread auto& fg = stile.frag_at(0, ik);
                const uint colLocal = uint(ik * QW_SRS_KU) + uint(sn);
                STEEL_PRAGMA_UNROLL
                for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                    const uint i = uint(loc >> 2), j = uint(loc & 3);
                    const uint qLoc = rowBase + uint(sm) + i * 8u, slotC = s0 + colLocal + j;
                    const uint absCC = (slotC < totalSlots) ? absKeySh[colLocal + j] : 0xFFFFFFFFu;
                    const uint absRR = p.posOffset + qLoc;
                    const bool inWin = (p.sliding == 0u) || (absRR < absCC + p.sliding);
                    if (qLoc < p.T && slotC < totalSlots && absCC <= absRR && absCC < p.kvLen && inWin) {
                        if (stepHasRel && (absRR - absCC) < p.relExtent) {
                            const uint dist = absRR - absCC;
                            float bias = 0.0f;
                            const device bfloat* rPtr = (p.dRel > 16u)
                                ? (rBF16 + (size_t(qLoc) * p.nH + head) * p.dRel) : nullptr;
                            for (uint dr = 0u; dr < p.dRel; ++dr) {
                                const float rv = (p.dRel <= 16u) ? float(rReg[i][dr]) : float(rPtr[dr]);
                                bias = fma(rv, float(relProj[size_t(dr) * p.relExtent + dist]), bias);
                            }
                            fg[loc] += bias * log2e;
                        }
                    } else {
                        fg[loc] = -1e30f;
                    }
                }
            }
        }

        float mnew[2] = {mrow[0], mrow[1]};
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < 2; ik++) {
            thread auto& fg = stile.frag_at(0, ik);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < BaseNAXFrag::kElemRows; i++) {
                mnew[i] = max(mnew[i], max(max(fg[i * 4 + 0], fg[i * 4 + 1]), max(fg[i * 4 + 2], fg[i * 4 + 3])));
            }
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            mnew[i] = max(mnew[i], simd_shuffle_xor(mnew[i], ushort(1)));
            mnew[i] = max(mnew[i], simd_shuffle_xor(mnew[i], ushort(8)));
        }

        float psum[2] = {0.0f, 0.0f};
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < 2; ik++) {
            thread auto& sf = stile.frag_at(0, ik);
            thread auto& pf = pt.frag_at(0, ik);
            STEEL_PRAGMA_UNROLL
            for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                const short i = loc >> 2;
                const float x = (isUnmasked || sf[loc] > -1e29f) ? fast::exp2(sf[loc] - mnew[i]) : 0.0f;
                pf[loc] = half(x);
                psum[i] += x;
            }
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            psum[i] += simd_shuffle_xor(psum[i], ushort(1));
            psum[i] += simd_shuffle_xor(psum[i], ushort(8));
            const bool grew = (mnew[i] > mrow[i]);
            const float corr = (grew && mrow[i] > -1e29f) ? fast::exp2(mrow[i] - mnew[i]) : (grew ? 0.0f : 1.0f);
            lrow[i] = lrow[i] * corr + psum[i];
            if (grew && mrow[i] > -1e29f) {
                STEEL_PRAGMA_UNROLL
                for (short d = 0; d < 8; d++) {
                    thread auto& ofg = otile.frag_at(0, d);
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; j++) ofg[i * 4 + j] *= corr;
                }
            }
            mrow[i] = mnew[i];
        }

        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < 2; ik++) {
            STEEL_PRAGMA_UNROLL
            for (short pair = 0; pair < 4; pair++) {
                NAXTile<half, 1, 2> vt;
                vt.template load<half, QW_SRS_KV_STRIDE, 1>(
                    vblk + uint(ik * QW_SRS_KU) * QW_SRS_KV_STRIDE + uint(pair * 32));
                BaseNAXFrag::mma(otile.frag_at(0, 2 * pair), otile.frag_at(0, 2 * pair + 1),
                                 pt.frag_at(0, ik), metal::false_type{},
                                 vt.frag_at(0, 0), vt.frag_at(0, 1), metal::false_type{});
            }
        }
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
        const uint rOut = rowBase + uint(sm) + uint(i) * 8u;
        if (rOut >= p.T) continue;
        const float rcp = (lrow[i] > 0.0f) ? (1.0f / lrow[i]) : 0.0f;
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < 8; d++) {
            const size_t oIdx = (size_t(rOut) * p.nH + head) * QW_SRS_HDIM + uint(d) * QW_SRS_KU + uint(sn);
            thread auto& fg = otile.frag_at(0, d);
            const float4 o4 = float4(fg[i * 4 + 0], fg[i * 4 + 1], fg[i * 4 + 2], fg[i * 4 + 3]) * rcp;
            qwg_sr_write_out(o4, gateBF16, outBF16, oIdx, p.hasOutGate);
        }
    }
}
#endif // !defined(QW_DISABLE_NAX)

METAL_FUNC void qwg_sr_dec_score_tok(
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

template <uint MODE>
METAL_FUNC void qwg_attn_decode_sink_rel_sparse_gqa_body(
    device const bfloat* qBF16, device const bfloat* rBF16, device const bfloat* relProj,
    device const bfloat* sinkBF16, device const bfloat* gateBF16,
    device const uchar* kCache, device const uchar* vCache,
    device const uchar* kMeta, device const uchar* vMeta,
    device bfloat* outBF16, device float* partO, device float2* partML,
    constant QWSinkRelAttnParams& p, constant QWSinkRelTreeMeta& tree,
    device const int* blockList, device const uint* blockCount,
    uint3 tg, uint sgi, uint lane,
    threadgroup half4 (*kSh)[33], threadgroup half4 (*vSh)[33], threadgroup uint* jSh
) {
    #pragma clang fp contract(off) reassociate(off)
    const uint gqa = p.nH / p.nKV, gChunks = (gqa + 3u) >> 2u;
    const uint spSplit = tg.x, kvh = tg.y / gChunks, g0 = (tg.y - kvh * gChunks) * 4u, tGrpBase = tg.z * 16u;
    if (spSplit >= p.splits || kvh >= p.nKV || g0 >= gqa || tGrpBase >= p.T) return;

    const uint gCount = min(4u, gqa - g0), headBase = kvh * gqa + g0, dBase = lane * 4u;
    const bool activeD = (dBase < p.headDim);
    const float qkScale = rsqrt(float(p.headDim));
    const bool fastT4 = (p.T <= 4u && tGrpBase == 0u);
    const uint gSub = fastT4 ? ((sgi & 1u) * 2u) : 0u;
    const uint myGCount = (gSub < gCount) ? min(fastT4 ? 2u : 4u, gCount - gSub) : 0u;
    const uint headBaseLocal = headBase + gSub, uEnd = fastT4 ? 1u : 2u;

    const uint batchEnd = min(p.kvLen, p.posOffset + p.T);
    const uint tailStart = p.posOffset & ~3u;
    const uint globalMin = (p.sliding > 0u && p.posOffset + 1u > p.sliding) ? (p.posOffset + 1u - p.sliding) : 0u;
    const uint2 range = qwg_sr_sparse_range(blockList, blockCount[0], globalMin, p.posOffset >> 2u);
    const uint bBegin = range.x, uCount = range.y - range.x;
    const uint totalKeys = uCount * 4u + ((batchEnd > tailStart) ? (batchEnd - tailStart) : 0u);
    const uint chunk = (totalKeys > 0u) ? ((totalKeys + p.splits - 1u) / p.splits) : 0u;
    const uint sStart = min(spSplit * chunk, totalKeys), sEnd = min(sStart + chunk, totalKeys);

    const uint t0 = tGrpBase + sgi * 2u;
    bool tokValid[2];
    uint tokIdx[2], qPos[2], tokMin[2];
    float4 q4[2][4], acc[2][4];
    float rLane[2][4], m[2][4], l[2][4];
    const bool hasSinkHere = (spSplit == 0u) && (p.hasSink != 0u);

    #pragma unroll
    for (uint u = 0u; u < 2u; ++u) {
        const uint tok = fastT4 ? (sgi >> 1u) : (t0 + u);
        tokIdx[u] = tok;
        tokValid[u] = (u < uEnd) && (tok < p.T) && (myGCount > 0u);
        qPos[u] = (tokValid[u] && tree.isTree != 0u && tok < 16u)
            ? (p.posOffset + tree.depths[tok]) : (p.posOffset + tok);
        tokMin[u] = (p.sliding > 0u && qPos[u] + 1u > p.sliding) ? (qPos[u] + 1u - p.sliding) : 0u;
        #pragma unroll
        for (uint g = 0u; g < 4u; ++g) {
            const uint h = headBaseLocal + g;
            const bool ok = tokValid[u] && g < myGCount;
            q4[u][g] = (ok && activeD) ? float4(*(device const bfloat4*)(qBF16 + (size_t(tok) * p.nH + h) * p.headDim + dBase)) : float4(0.0f);
            rLane[u][g] = (ok && p.hasRelBias != 0u && lane < p.dRel) ? float(rBF16[(size_t(tok) * p.nH + h) * p.dRel + lane]) : 0.0f;
            m[u][g] = (ok && hasSinkHere) ? float(sinkBF16[size_t(tok) * p.nH + h]) : -1e30f;
            l[u][g] = (ok && hasSinkHere) ? 1.0f : 0.0f;
            acc[u][g] = float4(0.0f);
        }
    }

    const bool mayRel = (p.hasRelBias != 0u) && (p.dRel > 0u) && (p.relExtent > 0u);
    for (uint s0 = sStart; s0 < sEnd; s0 += 8u) {
        const uint slotLoad = s0 + sgi;
        if (slotLoad < sEnd) {
            const uint jLoad = qwg_sr_slot_key(slotLoad, bBegin, uCount, tailStart, totalKeys, blockList);
            if (lane == 0u) jSh[sgi] = jLoad;
            if (activeD && jLoad >= globalMin && jLoad < batchEnd) {
                const uint kvSlot = (p.sliding > 0u) ? (jLoad % p.ringCap) : jLoad;
                kSh[sgi][lane] = qwg_sink_rel_read_kv4<MODE>(lane, kvSlot, kvh, p.nKV, p.headDim, kCache, kMeta);
                vSh[sgi][lane] = qwg_sink_rel_read_kv4<MODE>(lane, kvSlot, kvh, p.nKV, p.headDim, vCache, vMeta);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint cnt = min(8u, sEnd - s0);
        for (uint b = 0u; b < cnt; ++b) {
            const uint j = jSh[b];
            if (j < globalMin || j >= batchEnd) continue;
            const float4 kf = activeD ? float4(kSh[b][lane]) : float4(0.0f);
            const float4 vf = activeD ? float4(vSh[b][lane]) : float4(0.0f);
            #pragma unroll
            for (uint u = 0u; u < 2u; ++u) {
                if (u >= uEnd || !tokValid[u]) continue;
                const uint tok = tokIdx[u];
                bool vis = false;
                uint dist = 0u;
                if (j < p.posOffset) {
                    vis = (j >= tokMin[u]);
                    if (vis) dist = qPos[u] - j;
                } else {
                    const uint kRel = j - p.posOffset;
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
                }
                if (vis) qwg_sr_dec_score_tok(dist, myGCount, mayRel, qkScale, kf, vf, relProj, p, lane, q4[u], rLane[u], m[u], l[u], acc[u]);
            }
        }
        if (s0 + 8u < sEnd) threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint u = 0u; u < 2u; ++u) {
        if (u >= uEnd || !tokValid[u]) continue;
        const uint tok = tokIdx[u];
        #pragma unroll
        for (uint g = 0u; g < 4u; ++g) {
            if (g >= myGCount) continue;
            const uint h = headBaseLocal + g;
            if (p.splits == 1u) {
                if (!activeD) continue;
                const size_t oBase = (size_t(tok) * p.nH + h) * p.headDim + dBase;
                qwg_sr_write_out(acc[u][g] / max(l[u][g], 1e-30f), gateBF16, outBF16, oBase, p.hasOutGate);
            } else {
                const size_t mlIdx = (size_t(tok) * p.nH + h) * p.splits + spSplit;
                if (activeD) *(device float4*)(partO + mlIdx * p.headDim + dBase) = acc[u][g];
                if (lane == 0u) partML[mlIdx] = float2(m[u][g], l[u][g]);
            }
        }
    }
}

#if !defined(QW_DISABLE_NAX)
#define QW_SR_PREFILL_SPARSE_ENTRY(SFX, MODE) \
kernel void qwg_attn_prefill_sink_rel_sparse_nax##SFX( \
    device const bfloat* qBF16 [[buffer(0)]], device const bfloat* rBF16 [[buffer(1)]], \
    device const bfloat* relProj [[buffer(2)]], device const bfloat* sinkBF16 [[buffer(3)]], \
    device const bfloat* gateBF16 [[buffer(4)]], device const uchar* kCache [[buffer(5)]], \
    device const uchar* vCache [[buffer(6)]], device const uchar* kMeta [[buffer(7)]], \
    device const uchar* vMeta [[buffer(8)]], device bfloat* outBF16 [[buffer(9)]], \
    constant QWSinkRelAttnParams& p [[buffer(10)]], device const int* blockList [[buffer(11)]], \
    device const uint* blockCount [[buffer(12)]], constant QWSinkRelSparseMeta& sp [[buffer(13)]], \
    uint tid [[thread_index_in_threadgroup]], uint sgi [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint2 tg [[threadgroup_position_in_grid]]) { \
    threadgroup half kvblk[QW_SRS_BK * QW_SRS_KV_STRIDE]; \
    threadgroup half vblk[QW_SRS_BK * QW_SRS_KV_STRIDE]; \
    threadgroup uint absKeySh[32]; \
    qwg_attn_prefill_sink_rel_sparse_nax_body<MODE>( \
        qBF16, rBF16, relProj, sinkBF16, gateBF16, kCache, vCache, kMeta, vMeta, outBF16, \
        p, blockList, blockCount, sp, tid, sgi, lane, tg, &kvblk[0], &vblk[0], &absKeySh[0]); \
}
QW_SR_PREFILL_SPARSE_ENTRY(_f16, 0u)
QW_SR_PREFILL_SPARSE_ENTRY(_q8, 1u)
QW_SR_PREFILL_SPARSE_ENTRY(_q4, 2u)
#undef QW_SR_PREFILL_SPARSE_ENTRY
#endif // !defined(QW_DISABLE_NAX)

#define QW_SR_DECODE_SPARSE_ENTRY(SFX, MODE) \
kernel void qwg_attn_decode_sink_rel_sparse_gqa##SFX( \
    device const bfloat* qBF16 [[buffer(0)]], device const bfloat* rBF16 [[buffer(1)]], \
    device const bfloat* relProj [[buffer(2)]], device const bfloat* sinkBF16 [[buffer(3)]], \
    device const bfloat* gateBF16 [[buffer(4)]], device const uchar* kCache [[buffer(5)]], \
    device const uchar* vCache [[buffer(6)]], device const uchar* kMeta [[buffer(7)]], \
    device const uchar* vMeta [[buffer(8)]], device bfloat* outBF16 [[buffer(9)]], \
    device float* partO [[buffer(10)]], device float2* partML [[buffer(11)]], \
    constant QWSinkRelAttnParams& p [[buffer(12)]], constant QWSinkRelTreeMeta& tree [[buffer(13)]], \
    device const int* blockList [[buffer(14)]], device const uint* blockCount [[buffer(15)]], \
    uint3 tg [[threadgroup_position_in_grid]], uint sgi [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half4 kSh[8][33]; \
    threadgroup half4 vSh[8][33]; \
    threadgroup uint jSh[8]; \
    qwg_attn_decode_sink_rel_sparse_gqa_body<MODE>( \
        qBF16, rBF16, relProj, sinkBF16, gateBF16, kCache, vCache, kMeta, vMeta, \
        outBF16, partO, partML, p, tree, blockList, blockCount, tg, sgi, lane, kSh, vSh, jSh); \
}
QW_SR_DECODE_SPARSE_ENTRY(_f16, 0u)
QW_SR_DECODE_SPARSE_ENTRY(_q8, 1u)
QW_SR_DECODE_SPARSE_ENTRY(_q4, 2u)
#undef QW_SR_DECODE_SPARSE_ENTRY
