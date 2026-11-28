// qwattn_sink_rel_nax.metal — Generalized NAX FlashAttention prefill
// with optional attention sink, banded relative position bias, and output gate (f16/q8/q4 KV).
#if !defined(QW_DISABLE_NAX)
#include <metal_stdlib>
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#if !defined(__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__) || __ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ < 260200
#error "NAX cooperative_tensor kernels require -mmacosx-version-min=26.2 or newer (Apple MPP operand_index ABI)."
#endif
#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "qw_attn_sink_rel.metalh"
#include "inkling2/qw_ink2_common.metalh"

using namespace metal;
using namespace mlx::steel;

#define QW_SR_BM          32u
#define QW_SR_BK          32u
#define QW_SR_KU          16u
#define QW_SR_HDIM        128u
#define QW_SR_KV_STRIDE   136u
#define QW_SR_REL_STRIDE  68u

METAL_FUNC void qwg_sink_rel_load_q_frag(
    device const bfloat* qBF16, uint rowBase, uint head, uint nH, uint T,
    uint colBase, short2 sc, thread BaseNAXFrag::dtype_frag_t<half>& dst
) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
        const uint r = rowBase + uint(sc.y) + uint(i) * 8u;
        if (r < T) {
            const device bfloat* src = qBF16 + (size_t(r) * nH + head) * QW_SR_HDIM + colBase + uint(sc.x);
            const ushort4 raw = *(device const ushort4*)src;
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

METAL_FUNC void qwg_sink_rel_load_r_frag(
    device const bfloat* rBF16, uint rowBase, uint head, uint nH, uint T,
    uint drBase, uint dRel, short2 sc, thread BaseNAXFrag::dtype_frag_t<bfloat>& dst
) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
        const uint r = rowBase + uint(sc.y) + uint(i) * 8u;
        if (r < T) {
            const device bfloat* rowPtr = rBF16 + (size_t(r) * nH + head) * dRel;
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) {
                const uint dr = drBase + uint(sc.x) + uint(j);
                dst[i * 4 + j] = (dr < dRel) ? rowPtr[dr] : bfloat(0.0f);
            }
        } else {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) dst[i * 4 + j] = bfloat(0.0f);
        }
    }
}

METAL_FUNC void qwg_sink_rel_stage_proj(
    device const bfloat* relProj, threadgroup bfloat* relStage, uint tid, uint kThreads,
    uint dr0, uint dRel, uint relExtent, int tileDStart, int tileDEnd
) {
    for (uint idx = tid; idx < 256u; idx += kThreads) {
        const uint drLoc = idx >> 4u, u4 = (idx & 15u) * 4u, dr = dr0 + drLoc;
        const int dist0 = tileDStart + int(u4);
        threadgroup ushort* dst = (threadgroup ushort*)(relStage + drLoc * QW_SR_REL_STRIDE + u4);
        if (dr < dRel && dist0 >= 0 && dist0 + 3 <= tileDEnd && ((dist0 & 3) == 0)
            && ((size_t(relProj + size_t(dr) * relExtent + size_t(dist0)) & 7u) == 0u)) {
            const ushort4 v4 = *(device const ushort4*)(relProj + size_t(dr) * relExtent + size_t(dist0));
            *(threadgroup ushort4*)dst = v4;
        } else {
            STEEL_PRAGMA_UNROLL
            for (uint j = 0u; j < 4u; ++j) {
                const int dist = dist0 + int(j);
                dst[j] = (dr < dRel && dist >= 0 && dist <= tileDEnd)
                    ? as_type<ushort>(relProj[size_t(dr) * relExtent + size_t(dist)]) : 0u;
            }
        }
    }
}

METAL_FUNC void qwg_sink_rel_mma_bias(
    threadgroup const bfloat* relStage, thread const BaseNAXFrag::dtype_frag_t<bfloat>& rf,
    uint colOff, bool needB2, thread NAXTile<float, 1, 3>& btile
) {
    NAXTile<bfloat, 1, 2> pt;
    pt.template load<bfloat, QW_SR_REL_STRIDE, 1>(relStage + colOff);
    BaseNAXFrag::mma(btile.frag_at(0, 0), btile.frag_at(0, 1), rf, metal::false_type{},
                     pt.frag_at(0, 0), pt.frag_at(0, 1), metal::false_type{});
    if (needB2) {
        NAXTile<bfloat, 1, 1> ptTail;
        ptTail.template load<bfloat, QW_SR_REL_STRIDE, 1>(relStage + colOff + 32u);
        BaseNAXFrag::dtype_frag_t<bfloat> zeroFrag(bfloat(0.0f));
        BaseNAXFrag::dtype_frag_t<float> b3Dummy(0.0f);
        BaseNAXFrag::mma(btile.frag_at(0, 2), b3Dummy, rf, metal::false_type{},
                         ptTail.frag_at(0, 0), zeroFrag, metal::false_type{});
    }
}

METAL_FUNC float4 qwg_sink_rel_fetch_chunk(
    int c, float4 b0, float4 b1, float4 b2, bool useB2, uint lane0
) {
    const uint uc = uint(clamp(c, 0, 11)), g = uc & 3u, blk = uc >> 2u;
    const ushort src = ushort(lane0 | (g & 1u) | ((g & 2u) << 2u));
    const float4 v0 = simd_shuffle(b0, src);
    const float4 v1 = simd_shuffle(b1, src);
    const float4 v2 = useB2 ? simd_shuffle(b2, src) : float4(0.0f);
    return (blk == 0u) ? v0 : ((blk == 1u) ? v1 : v2);
}

METAL_FUNC void qwg_sink_rel_apply_bias(
    thread const NAXTile<float, 1, 3>& btile, thread NAXTile<float, 1, 2>& stile,
    uint posOffset, uint rowBase, uint T, uint n0, int stripDStart, int stripDEnd,
    short2 sc, uint lane, float log2e
) {
    const uint lane0 = lane & ~9u;
    const bool needB2 = (stripDEnd - stripDStart) >= 32;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
        const float4 rB0(btile.frag_at(0, 0)[i * 4 + 0], btile.frag_at(0, 0)[i * 4 + 1],
                         btile.frag_at(0, 0)[i * 4 + 2], btile.frag_at(0, 0)[i * 4 + 3]);
        const float4 rB1(btile.frag_at(0, 1)[i * 4 + 0], btile.frag_at(0, 1)[i * 4 + 1],
                         btile.frag_at(0, 1)[i * 4 + 2], btile.frag_at(0, 1)[i * 4 + 3]);
        const float4 rB2(btile.frag_at(0, 2)[i * 4 + 0], btile.frag_at(0, 2)[i * 4 + 1],
                         btile.frag_at(0, 2)[i * 4 + 2], btile.frag_at(0, 2)[i * 4 + 3]);
        const uint qRowLocal = rowBase + uint(sc.y) + uint(i) * 8u;
        const int qRow = int(posOffset + qRowLocal);
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < 2; ik++) {
            const int u0 = qRow - int(n0 + uint(ik * 16) + uint(sc.x)) - stripDStart;
            const int cHi = (u0 >= 0) ? (u0 >> 2) : -1;
            const int cLo = ((u0 - 3) >= 0) ? ((u0 - 3) >> 2) : -1;
            const float4 vHi = qwg_sink_rel_fetch_chunk(cHi, rB0, rB1, rB2, needB2, lane0);
            const float4 vLo = (cLo == cHi)
                ? vHi : qwg_sink_rel_fetch_chunk(cLo, rB0, rB1, rB2, needB2, lane0);
            thread auto& fg = stile.frag_at(0, ik);
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) {
                const int u = u0 - int(j), dist = stripDStart + u;
                if (u >= 0 && dist <= stripDEnd && qRowLocal < T) {
                    const float bVal = ((u >> 2) == cHi) ? vHi[u & 3] : vLo[u & 3];
                    fg[i * 4 + j] += bVal * log2e;
                }
            }
        }
    }
}

template <uint MODE>
METAL_FUNC void qwg_attn_prefill_sink_rel_nax_body(
    device const bfloat* qBF16, device const bfloat* rBF16, device const bfloat* relProj,
    device const bfloat* sinkBF16, device const bfloat* gateBF16,
    device const uchar* kCache, device const uchar* vCache,
    device const uchar* kMeta, device const uchar* vMeta,
    device bfloat* outBF16, constant QWSinkRelAttnParams& p,
    uint tid, uint sgi, uint lane, uint2 tg,
    threadgroup half* kvblk, threadgroup half* vblk, threadgroup bfloat* relStage
) {
    const uint gqaRatio = p.nH / p.nKV, kThreads = 64u * gqaRatio;
    const uint tile0 = tg.x * QW_SR_BM, kvh = tg.y;
    if (tile0 >= p.T || kvh >= p.nKV) return;

    const uint sgiH = sgi >> 1u, sgiLocal = sgi & 1u;
    const uint head = kvh * gqaRatio + sgiH, rowBase = tile0 + sgiLocal * QW_SR_KU;
    const uint kEnd = min(p.posOffset + min(tile0 + QW_SR_BM, p.T), p.kvLen);
    const uint kStartRaw = (p.sliding > 0u && (p.posOffset + tile0 + 1u) > p.sliding)
        ? ((p.posOffset + tile0 + 1u) - p.sliding) : 0u;
    const uint nStart = kStartRaw & ~(QW_SR_BK - 1u);

    constexpr float log2e = 1.4426950408889634f;
    const float qkScale2 = rsqrt(float(QW_SR_HDIM)) * log2e;
    const short2 sc = BaseNAXFrag::get_coord();
    const short sm = sc.y, sn = sc.x;

    NAXTile<half, 1, 1> qf[8];
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < 8; id++) {
        qwg_sink_rel_load_q_frag(
            qBF16, rowBase, head, p.nH, p.T, uint(id) * QW_SR_KU, sc, qf[id].frag_at(0, 0)
        );
    }

    BaseNAXFrag::dtype_frag_t<bfloat> rf;
    if (p.hasRelBias != 0u && p.dRel > 0u && p.dRel <= 16u) {
        qwg_sink_rel_load_r_frag(rBF16, rowBase, head, p.nH, p.T, 0u, p.dRel, sc, rf);
    }

    NAXTile<float, 1, 8> otile;
    otile.clear();

    float mrow[2], lrow[2];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
        const uint rLoc = rowBase + uint(sm) + uint(i) * 8u;
        if (p.hasSink != 0u && rLoc < p.T) {
            mrow[i] = float(sinkBF16[size_t(rLoc) * p.nH + head]) * log2e;
            lrow[i] = 1.0f;
        } else {
            mrow[i] = -1e30f;
            lrow[i] = 0.0f;
        }
    }

    for (uint n0 = nStart; n0 < kEnd; n0 += QW_SR_BK) {
        if (n0 != nStart) threadgroup_barrier(mem_flags::mem_threadgroup);
        NAXTile<half, 1, 2> pt;

        if constexpr (MODE == 0u) {
            for (uint e = tid; e < 512u; e += kThreads) {
                const uint kr = e >> 4u, vecIdx8 = e & 15u, absKey = n0 + kr;
                const uint ob = kr * QW_SR_KV_STRIDE + (vecIdx8 << 3u);
                uint4 k8(0u), v8(0u);
                if (absKey >= kStartRaw && absKey < kEnd) {
                    const uint slot = (p.sliding > 0u) ? (absKey % p.ringCap) : absKey;
                    const size_t rowHead = size_t(slot) * p.nKV + kvh;
                    const size_t off = rowHead * QW_SR_HDIM + (vecIdx8 << 3u);
                    k8 = *(device const uint4*)(((device const half*)kCache) + off);
                    v8 = *(device const uint4*)(((device const half*)vCache) + off);
                }
                *(threadgroup uint4*)(kvblk + ob) = k8;
                *(threadgroup uint4*)(vblk + ob) = v8;
            }
        } else {
            for (uint e = tid; e < 1024u; e += kThreads) {
                const uint kr = e >> 5u, vecIdx = e & 31u, absKey = n0 + kr;
                const uint ob = kr * QW_SR_KV_STRIDE + (vecIdx << 2u);
                half4 k4(0.0h), v4(0.0h);
                if (absKey >= kStartRaw && absKey < kEnd) {
                    const uint slot = (p.sliding > 0u) ? (absKey % p.ringCap) : absKey;
                    k4 = qwg_sink_rel_read_kv4<MODE>(vecIdx, slot, kvh, p.nKV, QW_SR_HDIM, kCache, kMeta);
                    v4 = qwg_sink_rel_read_kv4<MODE>(vecIdx, slot, kvh, p.nKV, QW_SR_HDIM, vCache, vMeta);
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
        const int tileDEnd = min(int(p.relExtent) - 1, int(p.posOffset + min(tile0 + QW_SR_BM, p.T)) - 1 - int(n0));
        if (tileHasRel) {
            qwg_sink_rel_stage_proj(
                relProj, relStage, tid, kThreads, 0u, p.dRel, p.relExtent, tileDStart, tileDEnd
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        NAXTile<float, 1, 2> stile;
        stile.clear();
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < 8; id++) {
            NAXTile<half, 2, 1> kt;
            kt.template load<half, QW_SR_KV_STRIDE, 1>(kvblk + uint(id) * QW_SR_KU);
            BaseNAXFrag::mma(stile.frag_at(0, 0), stile.frag_at(0, 1), qf[id].frag_at(0, 0), metal::false_type{},
                             kt.frag_at(0, 0), kt.frag_at(1, 0), metal::true_type{});
        }
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < 2; ik++) {
            thread auto& fg = stile.frag_at(0, ik);
            STEEL_PRAGMA_UNROLL
            for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) fg[loc] *= qkScale2;
        }

        if (tileHasRel) {
            const int stripMinQ = int(p.posOffset + rowBase);
            const int stripMaxQ = int(p.posOffset + min(rowBase + QW_SR_KU, p.T)) - 1;
            const int stripDStart = (stripMinQ > int(n0 + 31u)) ? (stripMinQ - int(n0 + 31u)) : 0;
            const int stripDEnd = min(int(p.relExtent) - 1, stripMaxQ - int(n0));
            const bool stripHasRel = (rowBase < p.T) && (stripDStart <= stripDEnd);
            NAXTile<float, 1, 3> btile;
            btile.clear();
            if (stripHasRel) {
                const uint colOff = uint(stripDStart - tileDStart);
                const bool needB2 = (stripDEnd - stripDStart) >= 32;
                if (p.dRel <= 16u) {
                    qwg_sink_rel_mma_bias(relStage, rf, colOff, needB2, btile);
                } else {
                    BaseNAXFrag::dtype_frag_t<bfloat> rfk;
                    qwg_sink_rel_load_r_frag(rBF16, rowBase, head, p.nH, p.T, 0u, p.dRel, sc, rfk);
                    qwg_sink_rel_mma_bias(relStage, rfk, colOff, needB2, btile);
                }
            }
            for (uint dr0 = 16u; dr0 < p.dRel; dr0 += 16u) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                qwg_sink_rel_stage_proj(
                    relProj, relStage, tid, kThreads, dr0, p.dRel, p.relExtent, tileDStart, tileDEnd
                );
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (stripHasRel) {
                    const uint colOff = uint(stripDStart - tileDStart);
                    const bool needB2 = (stripDEnd - stripDStart) >= 32;
                    BaseNAXFrag::dtype_frag_t<bfloat> rfk;
                    qwg_sink_rel_load_r_frag(rBF16, rowBase, head, p.nH, p.T, dr0, p.dRel, sc, rfk);
                    qwg_sink_rel_mma_bias(relStage, rfk, colOff, needB2, btile);
                }
            }
            if (stripHasRel) {
                qwg_sink_rel_apply_bias(
                    btile, stile, p.posOffset, rowBase, p.T, n0, stripDStart, stripDEnd, sc, lane, log2e
                );
            }
        }

        const uint maxRowPlus1 = p.posOffset + min(rowBase + 16u, p.T);
        const bool isUnmasked = (rowBase + 16u <= p.T)
            && (n0 + QW_SR_BK <= p.posOffset + rowBase) && (n0 + QW_SR_BK <= p.kvLen)
            && (p.sliding == 0u || maxRowPlus1 <= n0 + p.sliding);
        if (!isUnmasked) {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < 2; ik++) {
                thread auto& fg = stile.frag_at(0, ik);
                const uint col0 = n0 + uint(ik * QW_SR_KU) + uint(sn);
                STEEL_PRAGMA_UNROLL
                for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                    const uint i = uint(loc >> 2), j = uint(loc & 3);
                    const uint qLoc = rowBase + uint(sm) + i * 8u;
                    const uint absCC = col0 + j, absRR = p.posOffset + qLoc;
                    const bool inWin = (p.sliding == 0u) || (absRR < absCC + p.sliding);
                    if (!(qLoc < p.T && absCC <= absRR && absCC < p.kvLen && inWin)) fg[loc] = -1e30f;
                }
            }
        }

        float mnew[2] = {mrow[0], mrow[1]};
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < 2; ik++) {
            thread auto& fg = stile.frag_at(0, ik);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < BaseNAXFrag::kElemRows; i++) {
                const float v = max(max(fg[i * 4 + 0], fg[i * 4 + 1]), max(fg[i * 4 + 2], fg[i * 4 + 3]));
                mnew[i] = max(mnew[i], v);
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
        }

        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
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
                vt.template load<half, QW_SR_KV_STRIDE, 1>(
                    vblk + uint(ik * QW_SR_KU) * QW_SR_KV_STRIDE + uint(pair * 32)
                );
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
            const uint col0 = uint(d) * QW_SR_KU + uint(sn);
            const size_t oIdx = (size_t(rOut) * p.nH + head) * QW_SR_HDIM + col0;
            thread auto& fg = otile.frag_at(0, d);
            float4 o4 = float4(fg[i * 4 + 0], fg[i * 4 + 1], fg[i * 4 + 2], fg[i * 4 + 3]) * rcp;
            if (p.hasOutGate != 0u) {
                const float4 g4 = float4(*(device const bfloat4*)(gateBF16 + oIdx));
                o4.x *= qw_ink2_sigmoid(g4.x);
                o4.y *= qw_ink2_sigmoid(g4.y);
                o4.z *= qw_ink2_sigmoid(g4.z);
                o4.w *= qw_ink2_sigmoid(g4.w);
            }
            *(device bfloat4*)(outBF16 + oIdx) = bfloat4(
                qw_ink2_bf16(o4.x), qw_ink2_bf16(o4.y), qw_ink2_bf16(o4.z), qw_ink2_bf16(o4.w)
            );
        }
    }
}

#define QW_SINK_REL_PREFILL_ENTRY(SFX, MODE) \
kernel void qwg_attn_prefill_sink_rel_nax##SFX( \
    device const bfloat* qBF16 [[buffer(0)]], device const bfloat* rBF16 [[buffer(1)]], \
    device const bfloat* relProj [[buffer(2)]], device const bfloat* sinkBF16 [[buffer(3)]], \
    device const bfloat* gateBF16 [[buffer(4)]], device const uchar* kCache [[buffer(5)]], \
    device const uchar* vCache [[buffer(6)]], device const uchar* kMeta [[buffer(7)]], \
    device const uchar* vMeta [[buffer(8)]], device bfloat* outBF16 [[buffer(9)]], \
    constant QWSinkRelAttnParams& p [[buffer(10)]], \
    uint tid [[thread_index_in_threadgroup]], uint sgi [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint2 tg [[threadgroup_position_in_grid]]) { \
    threadgroup half kvblk[QW_SR_BK * QW_SR_KV_STRIDE]; \
    threadgroup half vblk[QW_SR_BK * QW_SR_KV_STRIDE]; \
    threadgroup bfloat relStage[16u * QW_SR_REL_STRIDE]; \
    qwg_attn_prefill_sink_rel_nax_body<MODE>( \
        qBF16, rBF16, relProj, sinkBF16, gateBF16, kCache, vCache, kMeta, vMeta, outBF16, \
        p, tid, sgi, lane, tg, &kvblk[0], &vblk[0], &relStage[0]); \
}
QW_SINK_REL_PREFILL_ENTRY(_f16, 0u)
QW_SINK_REL_PREFILL_ENTRY(_q8, 1u)
QW_SINK_REL_PREFILL_ENTRY(_q4, 2u)
#undef QW_SINK_REL_PREFILL_ENTRY
#endif // !defined(QW_DISABLE_NAX)
