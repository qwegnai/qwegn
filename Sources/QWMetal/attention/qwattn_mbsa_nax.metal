// qwattn_mbsa_nax.metal — NAX sparse paged prefill and split-K GQA sparse decode for MBSA.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_geometry_constants.metalh"
#include "include/qw_dequant.metalh"
#include "include/qw_kv_paged.metalh"
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

#define QW_MBSA_KU       16u
#define QW_MBSA_BK       32u
#define QW_MBSA_NAX_HDIM 128u

// 1. M5 NAX Cooperative Tensor Sparse Paged Prefill (f16, q8, q4) over tiled32 union blocks + bitmasks.
template <uint MODE>
static inline void mbsa_prefill_paged_nax_body(
        const device half*  qgate,
        const device half*  kpool,
        const device half*  vpool,
        const device int*   ktab,
        const device int*   vtab,
        device half*        outp,
        constant uint&      pos,
        constant uint&      T,
        const device int*   tileBlockList,
        const device uint*  tileBlockMask,
        const device uint*  tileBlockCount,
        constant uint&      tileListStride,
        constant uint&      blockSize,
        uint tid, uint sgi, uint3 tg,
        threadgroup half* kvblk, threadgroup half* vblk) {
    const uint kThreads   = 64u * QW_GQA_RATIO;
    const uint num_groups = QW_KV_HEADS;
    const uint numTiles   = (T + 31u) >> 5u;
    if (T == 0u || tg.x >= numTiles * num_groups) return;
    const uint kvh        = tg.x % num_groups;
    // LPT tile ordering: schedule longest causal tiles (highest tileIdx) first.
    const uint tileIdx    = numTiles - 1u - (tg.x / num_groups);
    const uint tile0      = tileIdx * 32u;
    const uint headBase   = kvh * QW_GQA_RATIO;
    const uint kEnd       = pos + min(tile0 + 32u, T);

    const size_t tileSlot    = (size_t)tileIdx * num_groups + kvh;
    const uint uCount        = tileBlockCount[tileSlot];
    const device int* myList = tileBlockList + tileSlot * tileListStride;
    const device uint* myMask = tileBlockMask + tileSlot * tileListStride;

    const uint sgi_h     = sgi >> 1u;
    const uint sgi_local = sgi & 1u;
    const uint head      = headBase + sgi_h;
    const uint rowBase   = tile0 + sgi_local * QW_MBSA_KU;

    const float scale2 = QW_ATTN_SCALE * 1.4426950408889634f;
    constexpr short kU = 16;
    const short nq = short(QW_HEAD_DIM >> 4u);
    const short npair = short(QW_HEAD_DIM >> 5u);

    const device half* qdev = qgate + (uint64_t)rowBase * QW_QPROJ_OUT
                              + (uint64_t)head * QW_Q_STRIDE * QW_HEAD_DIM;

    const short2 sc = BaseNAXFrag::get_coord();
    const short sm = sc.y, sn = sc.x;

    NAXTile<half, 1, 1> Qf[16];
    const short validRows = short(clamp(int(T) - int(rowBase), 0, 16));
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < 16; id++) {
        if (id < nq) {
            if (validRows == 16)
                Qf[id].load(qdev + id * kU, int(QW_QPROJ_OUT));
            else
                Qf[id].load_safe(qdev + id * kU, int(QW_QPROJ_OUT), short2(kU, validRows));
        }
    }

    NAXTile<float, 1, 16> Otile;
    Otile.clear();

    float mrow[2], lrow[2];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) { mrow[i] = -1e30f; lrow[i] = 0.0f; }

    const uint kLoadVec8 = 4u * QW_HEAD_DIM;
    const uint kStageStride = QW_MBSA_NAX_HDIM;
    const uint VecPerRow = QW_HEAD_DIM >> 3u;
    const uint halfShift = sgi_local * 16u;

    for (uint bi = 0u; bi < uCount; ++bi) {
        const int rawBlk = myList[bi];
        if (rawBlk < 0) continue;
        const uint qMask = myMask[bi];
        if (qMask == 0u) continue;
        const uint bStart = uint(rawBlk) * blockSize;
        if (bStart >= kEnd) break;
        const uint subLimit = min(blockSize, kEnd - bStart);
        const uint halfMask = (qMask >> halfShift) & 0xFFFFu;

        for (uint sub = 0u; sub < subLimit; sub += QW_MBSA_BK) {
            const uint n0 = bStart + sub;
            const uint subValid = min(QW_MBSA_BK, subLimit - sub);
            for (uint e = tid; e < kLoadVec8; e += kThreads) {
                const uint kr = (QW_HEAD_DIM == 128u) ? (e >> 4u) : (e / VecPerRow);
                const uint dim0 = (QW_HEAD_DIM == 128u) ? ((e & 15u) << 3u) : ((e % VecPerRow) << 3u);
                const uint absKey = n0 + kr;
                const uint ob = kr * kStageStride + dim0;
                uint pk = 0u, pv = 0u;
                if (QW_HEAD_DIM == 128u) {
                    const ushort leader = ushort(tid & 16u);
                    if ((tid & 15u) == 0u && kr < subValid) {
                        const uint page = absKey >> 8u;
                        pk = (uint)ktab[page];
                        pv = (uint)vtab[page];
                    }
                    pk = simd_shuffle(pk, leader);
                    pv = simd_shuffle(pv, leader);
                } else if (kr < subValid) {
                    const uint page = absKey >> 8u;
                    pk = (uint)ktab[page];
                    pv = (uint)vtab[page];
                }
                if (kr < subValid) {
                    const uint slot = absKey & 255u;
                    if (MODE == 0u) {
                        const uint64_t rowOff = (uint64_t)slot * QW_ROW + (uint64_t)kvh * QW_HEAD_DIM + dim0;
                        device const half* sk = kpool + (uint64_t)pk * QW_PGEL + rowOff;
                        device const half* sv = vpool + (uint64_t)pv * QW_PGEL + rowOff;
                        *(threadgroup half4*)(kvblk + ob)      = *(device const half4*)sk;
                        *(threadgroup half4*)(kvblk + ob + 4u) = *(device const half4*)(sk + 4u);
                        *(threadgroup half4*)(vblk + ob)       = *(device const half4*)sv;
                        *(threadgroup half4*)(vblk + ob + 4u)  = *(device const half4*)(sv + 4u);
                    } else if (MODE == 1u) {
                        qwg_q8_load8((const device char*)kpool, (uint64_t)pk * QW_Q8_PGE, slot, kvh, dim0, kvblk, ob);
                        qwg_q8_load8((const device char*)vpool, (uint64_t)pv * QW_Q8_PGE, slot, kvh, dim0, vblk, ob);
                    } else {
                        qwg_q4_load8((const device uchar*)kpool, (uint64_t)pk * QW_Q4_PGE, slot, kvh, dim0, kvblk, ob);
                        qwg_q4_load8((const device uchar*)vpool, (uint64_t)pv * QW_Q4_PGE, slot, kvh, dim0, vblk, ob);
                    }
                } else {
                    *(threadgroup half4*)(kvblk + ob)      = half4(0.0h);
                    *(threadgroup half4*)(kvblk + ob + 4u) = half4(0.0h);
                    *(threadgroup half4*)(vblk + ob)       = half4(0.0h);
                    *(threadgroup half4*)(vblk + ob + 4u)  = half4(0.0h);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (halfMask != 0u) {
                NAXTile<float, 1, 2> Stile;
                Stile.clear();
                STEEL_PRAGMA_UNROLL
                for (short id = 0; id < 16; id++) {
                    if (id < nq) {
                        NAXTile<half, 2, 1> Kt;
                        Kt.template load<half, QW_MBSA_NAX_HDIM, 1>(kvblk + id * kU);
                        BaseNAXFrag::mma(Stile.frag_at(0, 0),
                                         Stile.frag_at(0, 1),
                                         Qf[id].frag_at(0, 0), metal::false_type{},
                                         Kt.frag_at(0, 0), Kt.frag_at(1, 0),
                                         metal::true_type{});
                    }
                }

                const bool isUnmasked = (halfMask == 0xFFFFu) && (subValid == QW_MBSA_BK) && (n0 + QW_MBSA_BK <= pos + rowBase);
                if (isUnmasked) {
                    STEEL_PRAGMA_UNROLL
                    for (short ik = 0; ik < 2; ik++) {
                        thread auto& fg = Stile.frag_at(0, ik);
                        STEEL_PRAGMA_UNROLL
                        for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) fg[loc] *= scale2;
                    }
                } else {
                    STEEL_PRAGMA_UNROLL
                    for (short ik = 0; ik < 2; ik++) {
                        thread auto& fg = Stile.frag_at(0, ik);
                        const uint colBase = (uint)(ik * kU) + (uint)sn;
                        STEEL_PRAGMA_UNROLL
                        for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                            const uint i = (uint)(loc >> 2), j = (uint)(loc & 3);
                            const uint cLocal = colBase + j;
                            const uint abs_cc = n0 + cLocal;
                            const uint qInTile = halfShift + (uint)sm + i * 8u;
                            const uint abs_rr = pos + tile0 + qInTile;
                            const bool qSel = ((qMask >> qInTile) & 1u) != 0u;
                            fg[loc] = (qSel && cLocal < subValid && tile0 + qInTile < T && abs_cc <= abs_rr)
                                ? (fg[loc] * scale2)
                                : (-1e30f);
                        }
                    }
                }

                float mnew[2];
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; i++) mnew[i] = mrow[i];
                STEEL_PRAGMA_UNROLL
                for (short ik = 0; ik < 2; ik++) {
                    thread auto& fg = Stile.frag_at(0, ik);
                    STEEL_PRAGMA_UNROLL
                    for (short i = 0; i < BaseNAXFrag::kElemRows; i++) {
                        float v = fg[i * 4 + 0] > fg[i * 4 + 1] ? fg[i * 4 + 0] : fg[i * 4 + 1];
                        v = v > fg[i * 4 + 2] ? v : fg[i * 4 + 2];
                        v = v > fg[i * 4 + 3] ? v : fg[i * 4 + 3];
                        mnew[i] = mnew[i] > v ? mnew[i] : v;
                    }
                }
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; i++) {
                    mnew[i] = max(mnew[i], simd_shuffle_xor(mnew[i], ushort(1)));
                    mnew[i] = max(mnew[i], simd_shuffle_xor(mnew[i], ushort(8)));
                }

                NAXTile<half, 1, 2> Pt;
                float psum[2] = {0.0f, 0.0f};
                if (isUnmasked) {
                    STEEL_PRAGMA_UNROLL
                    for (short ik = 0; ik < 2; ik++) {
                        thread auto& sf = Stile.frag_at(0, ik);
                        thread auto& pf = Pt.frag_at(0, ik);
                        STEEL_PRAGMA_UNROLL
                        for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                            const short i = loc >> 2;
                            const float x = fast::exp2(sf[loc] - mnew[i]);
                            pf[loc] = half(x);
                            psum[i] += x;
                        }
                    }
                } else {
                    STEEL_PRAGMA_UNROLL
                    for (short ik = 0; ik < 2; ik++) {
                        thread auto& sf = Stile.frag_at(0, ik);
                        thread auto& pf = Pt.frag_at(0, ik);
                        STEEL_PRAGMA_UNROLL
                        for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                            const short i = loc >> 2;
                            const float x = (sf[loc] > -1e29f) ? fast::exp2(sf[loc] - mnew[i]) : 0.0f;
                            pf[loc] = half(x);
                            psum[i] += x;
                        }
                    }
                }
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; i++) {
                    psum[i] += simd_shuffle_xor(psum[i], ushort(1));
                    psum[i] += simd_shuffle_xor(psum[i], ushort(8));
                }

                float corr[2];
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; i++) {
                    corr[i] = (mrow[i] > -1e29f) ? fast::exp2(mrow[i] - mnew[i]) : 0.0f;
                    lrow[i] = lrow[i] * corr[i] + psum[i];
                    mrow[i] = mnew[i];
                }
                if (corr[0] != 1.0f || corr[1] != 1.0f) {
                    STEEL_PRAGMA_UNROLL
                    for (short id = 0; id < 16; id++) {
                        if (id < nq) {
                            thread auto& of = Otile.frag_at(0, id);
                            STEEL_PRAGMA_UNROLL
                            for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) of[loc] *= corr[loc >> 2];
                        }
                    }
                }

                STEEL_PRAGMA_UNROLL
                for (short ik = 0; ik < 2; ik++) {
                    STEEL_PRAGMA_UNROLL
                    for (short idp = 0; idp < 8; idp++) {
                        if (idp < npair) {
                            const short id0 = idp * 2;
                            NAXTile<half, 1, 2> Vt;
                            Vt.template load<half, QW_MBSA_NAX_HDIM, 1>(vblk + (ik * kU) * kStageStride + id0 * kU);
                            BaseNAXFrag::mma(Otile.frag_at(0, id0),
                                             Otile.frag_at(0, id0 + 1),
                                             Pt.frag_at(0, ik), metal::false_type{},
                                             Vt.frag_at(0, 0), Vt.frag_at(0, 1),
                                             metal::false_type{});
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    float linv[2];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) linv[i] = (lrow[i] > 0.0f) ? (1.0f / lrow[i]) : 0.0f;

    const device half* gdev = qdev + QW_HEAD_DIM;
    device half* odev = outp + (uint64_t)rowBase * (QW_Q_HEADS * QW_HEAD_DIM)
                        + (uint64_t)head * QW_HEAD_DIM;

    STEEL_PRAGMA_UNROLL
    for (short idp = 0; idp < 8; idp++) {
        if (idp < npair) {
            const short id0 = idp * 2;
            const short id1 = id0 + 1;
            NAXTile<half, 1, 1> G0, G1;
            if (QW_HAS_Q_GATE) {
                if (validRows == 16) {
                    G0.load(gdev + id0 * kU, int(QW_QPROJ_OUT));
                    G1.load(gdev + id1 * kU, int(QW_QPROJ_OUT));
                } else {
                    G0.load_safe(gdev + id0 * kU, int(QW_QPROJ_OUT), short2(kU, validRows));
                    G1.load_safe(gdev + id1 * kU, int(QW_QPROJ_OUT), short2(kU, validRows));
                }
            }
            thread auto& of0 = Otile.frag_at(0, id0);
            thread auto& of1 = Otile.frag_at(0, id1);
            thread auto& gf0 = G0.frag_at(0, 0);
            thread auto& gf1 = G1.frag_at(0, 0);

            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                float w[8];
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; j++) {
                    const short loc = i * 4 + j;
                    const float g0 = QW_HAS_Q_GATE ? (1.0f / (1.0f + fast::exp(-float(gf0[loc])))) : 1.0f;
                    const float g1 = QW_HAS_Q_GATE ? (1.0f / (1.0f + fast::exp(-float(gf1[loc])))) : 1.0f;
                    w[j]     = of0[loc] * linv[i] * g0;
                    w[4 + j] = of1[loc] * linv[i] * g1;
                }
                if (QW_FWHT_ROTATED) {
                    const float a0 = w[0] + w[1], a1 = w[0] - w[1];
                    const float a2 = w[2] + w[3], a3 = w[2] - w[3];
                    const float b0 = a0 + a2,     b2 = a0 - a2;
                    const float b1 = a1 + a3,     b3 = a1 - a3;
                    const float c0 = w[4] + w[5], c1 = w[4] - w[5];
                    const float c2 = w[6] + w[7], c3 = w[6] - w[7];
                    const float d0 = c0 + c2,     d2 = c0 - c2;
                    const float d1 = c1 + c3,     d3 = c1 - c3;
                    w[0] = b0; w[1] = b1; w[2] = b2; w[3] = b3;
                    w[4] = d0; w[5] = d1; w[6] = d2; w[7] = d3;
                    const bool hi4  = (sn & 4) != 0;
                    const bool hi8  = (sn & 8) != 0;
                    STEEL_PRAGMA_UNROLL
                    for (short k = 0; k < 8; k++) {
                        const float p4 = simd_shuffle_xor(w[k], ushort(1));
                        w[k] = hi4 ? (p4 - w[k]) : (w[k] + p4);
                    }
                    STEEL_PRAGMA_UNROLL
                    for (short k = 0; k < 8; k++) {
                        const float p8 = simd_shuffle_xor(w[k], ushort(8));
                        w[k] = hi8 ? (p8 - w[k]) : (w[k] + p8);
                    }
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; j++) {
                        const float u = w[j], v = w[4 + j];
                        w[j]     = (u + v) * 0.17677669529663689f;
                        w[4 + j] = (u - v) * 0.17677669529663689f;
                    }
                }
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; j++) {
                    const short loc = i * 4 + j;
                    gf0[loc] = half(w[j]);
                    gf1[loc] = half(w[4 + j]);
                }
            }
            if (validRows == 16) {
                G0.store(odev + id0 * kU, int(QW_Q_HEADS * QW_HEAD_DIM));
                G1.store(odev + id1 * kU, int(QW_Q_HEADS * QW_HEAD_DIM));
            } else if (validRows > 0) {
                G0.store_safe(odev + id0 * kU, int(QW_Q_HEADS * QW_HEAD_DIM), short2(kU, validRows));
                G1.store_safe(odev + id1 * kU, int(QW_Q_HEADS * QW_HEAD_DIM), short2(kU, validRows));
            }
        }
    }
}

#define QW_MBSA_NAX_PREFILL_ENTRY(SFX, MODE) \
kernel void qwg_mbsa_prefill_paged_nax##SFX( \
        const device half*  qgate          [[buffer(0)]], \
        const device half*  kpool          [[buffer(1)]], \
        const device half*  vpool          [[buffer(2)]], \
        const device int*   ktab           [[buffer(3)]], \
        const device int*   vtab           [[buffer(4)]], \
        device half*        outp           [[buffer(5)]], \
        constant uint&      pos            [[buffer(6)]], \
        constant uint&      T              [[buffer(7)]], \
        const device int*   tileBlockList  [[buffer(8)]], \
        const device uint*  tileBlockMask  [[buffer(9)]], \
        const device uint*  tileBlockCount [[buffer(10)]], \
        constant uint&      tileListStride [[buffer(11)]], \
        constant uint&      blockSize      [[buffer(12)]], \
        uint tid [[thread_index_in_threadgroup]], \
        uint sgi [[simdgroup_index_in_threadgroup]], \
        uint3 tg [[threadgroup_position_in_grid]]) { \
    threadgroup half kvblk[QW_MBSA_BK * QW_MBSA_NAX_HDIM]; \
    threadgroup half vblk[QW_MBSA_BK * QW_MBSA_NAX_HDIM]; \
    mbsa_prefill_paged_nax_body<MODE>(qgate, kpool, vpool, ktab, vtab, outp, pos, T, \
                                      tileBlockList, tileBlockMask, tileBlockCount, \
                                      tileListStride, blockSize, tid, sgi, tg, kvblk, vblk); \
}
QW_MBSA_NAX_PREFILL_ENTRY(_f16, 0u)
QW_MBSA_NAX_PREFILL_ENTRY(_q8,  1u)
QW_MBSA_NAX_PREFILL_ENTRY(_q4,  2u)
#undef QW_MBSA_NAX_PREFILL_ENTRY
#endif // !defined(QW_DISABLE_NAX)
#define QW_MBSA_DEC_HDIM 256u

// 2. Split-K GQA-fused MBSA Sparse Decode & Verify (rows = 1..8) over paged KV pools.
// Register-only 32-point normalized FWHT across 8 lanes (4 halves/lane); identity in STANDARD basis.
METAL_FUNC void mbsa_fwht32_h4(thread half (&v)[4], uint lane) {
    if (!QW_FWHT_ROTATED) return;
    float f[4];
    #pragma unroll
    for (uint i = 0u; i < 4u; i++) f[i] = float(v[i]);
    const float a0 = f[0] + f[1], a1 = f[0] - f[1];
    const float a2 = f[2] + f[3], a3 = f[2] - f[3];
    f[0] = a0 + a2; f[2] = a0 - a2;
    f[1] = a1 + a3; f[3] = a1 - a3;
    const bool neg1 = (lane & 1u) != 0u;
    const bool neg2 = (lane & 2u) != 0u;
    const bool neg4 = (lane & 4u) != 0u;
    #pragma unroll
    for (uint i = 0u; i < 4u; i++) {
        const float p1 = simd_shuffle_xor(f[i], ushort(1));
        f[i] = neg1 ? (p1 - f[i]) : (f[i] + p1);
    }
    #pragma unroll
    for (uint i = 0u; i < 4u; i++) {
        const float p2 = simd_shuffle_xor(f[i], ushort(2));
        f[i] = neg2 ? (p2 - f[i]) : (f[i] + p2);
    }
    constexpr float inv_sqrt32 = 0.17677669529663689f;
    #pragma unroll
    for (uint i = 0u; i < 4u; i++) {
        const float p4 = simd_shuffle_xor(f[i], ushort(4));
        f[i] = (neg4 ? (p4 - f[i]) : (f[i] + p4)) * inv_sqrt32;
    }
    #pragma unroll
    for (uint i = 0u; i < 4u; i++) v[i] = half(f[i]);
}

template <uint MODE, uint ELEMS>
static inline void mbsa_decode_sparse_impl(
        const device half* qgate, device half* kpool, device half* vpool,
        const device int* ktab, const device int* vtab,
        const device int* blockList, const device int* blockCount,
        device float* partials, device half* outp, uint4 params,
        uint tid, uint3 tg,
        threadgroup half* stageK, threadgroup half* stageV,
        threadgroup uint* sAbsKey, threadgroup uint* sPk, threadgroup uint* sPv) {
    constexpr uint stageStride = (ELEMS == 4u) ? 128u : QW_MBSA_DEC_HDIM;
    const uint kvh = tg.x, qi = tg.z;
    const uint sg = tid >> 5u, lane = tid & 31u;
    const uint head = kvh * QW_GQA_RATIO + sg;
    const uint d0 = lane * ELEMS;
    const bool live = d0 < QW_HEAD_DIM;
    const uint pos = params.x + qi, blockSize = params.y, listStride = params.z, splits = params.w;

    const device half* qrow = qgate + (uint64_t)qi * QW_QPROJ_OUT + (uint64_t)head * QW_Q_STRIDE * QW_HEAD_DIM;
    float qv[ELEMS];
    if (live) {
        for (uint i = 0u; i < ELEMS; i++) qv[i] = float(qrow[d0 + i]);
    } else {
        for (uint i = 0u; i < ELEMS; i++) qv[i] = 0.0f;
    }

    const size_t slotBase = ((size_t)qi * QW_KV_HEADS + kvh) * listStride;
    const uint nBlks = uint(blockCount[(size_t)qi * QW_KV_HEADS + kvh]);
    const uint totalSlots = nBlks * blockSize;
    const uint per = ((totalSlots + splits - 1u) / splits + 15u) & ~15u;
    const uint s0 = min(tg.y * per, totalSlots);
    const uint s1 = min(s0 + per, totalSlots);

    float m = -1e30f, lsum = 0.0f;
    float oacc[ELEMS];
    for (uint i = 0u; i < ELEMS; i++) oacc[i] = 0.0f;

    #define MBSA_MERGE(s, vv) { \
        const float _mn = max(m, s); \
        const float _c  = (m > -1e29f) ? ((s > m) ? metal::exp(m - s) : 1.0f) : 0.0f; \
        const float _p  = metal::exp(s - _mn); \
        lsum = lsum * _c + _p; \
        for (uint _i = 0u; _i < ELEMS; _i++) oacc[_i] = oacc[_i] * _c + _p * float(vv[_i]); \
        m = _mn; \
    }
    #define MBSA_DOT(kk) ({ \
        float _pd = 0.0f; \
        for (uint _i = 0u; _i < ELEMS; _i++) _pd += qv[_i] * float(kk[_i]); \
        simd_sum(_pd) * QW_ATTN_SCALE; })

    const uint kThreads = QW_GQA_RATIO * 32u;
    const uint vecPerRow = QW_HEAD_DIM >> 3u;
    const uint totalVecs = 16u * vecPerRow;

    uint r = s0;
    for (; r + 16u <= s1; r += 16u) {
        if (tid < 16u) {
            const uint slotIdx = r + tid;
            const int rawBlk = blockList[slotBase + (slotIdx / blockSize)];
            const uint absKey = (rawBlk >= 0) ? (uint(rawBlk) * blockSize + (slotIdx % blockSize)) : 0xFFFFFFFFu;
            const bool ok = (rawBlk >= 0) && (absKey <= pos);
            sAbsKey[tid] = ok ? absKey : 0xFFFFFFFFu;
            if (ok) {
                const uint page = absKey >> 8u;
                sPk[tid] = (uint)ktab[page];
                sPv[tid] = (uint)vtab[page];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint e = tid; e < totalVecs; e += kThreads) {
            const uint keyIdx = (QW_HEAD_DIM == 128u) ? (e >> 4u) : (e / vecPerRow);
            const uint col = (QW_HEAD_DIM == 128u) ? ((e & 15u) << 3u) : ((e % vecPerRow) << 3u);
            const uint absKey = sAbsKey[keyIdx];
            threadgroup half* dk = &stageK[keyIdx * stageStride + col];
            threadgroup half* dv = &stageV[keyIdx * stageStride + col];
            if (absKey != 0xFFFFFFFFu) {
                const uint slot = absKey & 255u;
                const uint pk = sPk[keyIdx], pv = sPv[keyIdx];
                if (MODE == 0u) {
                    const uint64_t off = (uint64_t)slot * QW_ROW + (uint64_t)kvh * QW_HEAD_DIM + col;
                    *(threadgroup uint4*)dk = *(const device uint4*)(kpool + (uint64_t)pk * QW_PGEL + off);
                    *(threadgroup uint4*)dv = *(const device uint4*)(vpool + (uint64_t)pv * QW_PGEL + off);
                } else {
                    half ka[8], va[8];
                    QW_KV_LOAD8(ka, kpool, pk, slot, kvh, col, MODE)
                    QW_KV_LOAD8(va, vpool, pv, slot, kvh, col, MODE)
                    *(threadgroup half4*)dk        = half4(ka[0], ka[1], ka[2], ka[3]);
                    *(threadgroup half4*)(dk + 4u) = half4(ka[4], ka[5], ka[6], ka[7]);
                    *(threadgroup half4*)dv        = half4(va[0], va[1], va[2], va[3]);
                    *(threadgroup half4*)(dv + 4u) = half4(va[4], va[5], va[6], va[7]);
                }
            } else {
                *(threadgroup uint4*)dk = uint4(0u);
                *(threadgroup uint4*)dv = uint4(0u);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float s16[16];
        for (uint j = 0u; j < 16u; j++) {
            alignas(16) half kk[ELEMS];
            if (live) {
                if constexpr (ELEMS == 4u) {
                    *(thread half4*)kk = *(threadgroup const half4*)&stageK[j * stageStride + d0];
                } else {
                    *(thread uint4*)kk = *(threadgroup const uint4*)&stageK[j * stageStride + d0];
                }
            } else {
                for (uint i = 0u; i < ELEMS; i++) kk[i] = half(0.0h);
            }
            const float sc = MBSA_DOT(kk);
            s16[j] = (sAbsKey[j] != 0xFFFFFFFFu) ? sc : -1e30f;
        }
        float bm = s16[0];
        for (uint j = 1u; j < 16u; j++) bm = max(bm, s16[j]);
        if (bm > -1e29f) {
            const float Mb = max(m, bm);
            float lp = 0.0f, op[ELEMS];
            for (uint i = 0u; i < ELEMS; i++) op[i] = 0.0f;
            for (uint j = 0u; j < 16u; j++) {
                const float pj = (s16[j] > -1e29f) ? metal::exp(s16[j] - Mb) : 0.0f;
                lp += pj;
                if (pj > 0.0f && live) {
                    alignas(16) half vv[ELEMS];
                    if constexpr (ELEMS == 4u) {
                        *(thread half4*)vv = *(threadgroup const half4*)&stageV[j * stageStride + d0];
                    } else {
                        *(thread uint4*)vv = *(threadgroup const uint4*)&stageV[j * stageStride + d0];
                    }
                    for (uint i = 0u; i < ELEMS; i++) op[i] += pj * float(vv[i]);
                }
            }
            const float cb = (m > -1e29f) ? metal::exp(m - Mb) : 0.0f;
            lsum = lsum * cb + lp;
            for (uint i = 0u; i < ELEMS; i++) oacc[i] = oacc[i] * cb + op[i];
            m = Mb;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (; r < s1; r++) {
        const int rawBlk = blockList[slotBase + (r / blockSize)];
        if (rawBlk < 0) continue;
        const uint j = uint(rawBlk) * blockSize + (r % blockSize);
        if (j > pos) continue;
        const uint page = j >> 8u, slot = j & 255u;
        const uint pk = (uint)ktab[page], pv = (uint)vtab[page];
        alignas(16) half kk[ELEMS], vv[ELEMS];
        if (live) {
            if (MODE == 0u) {
                const uint64_t rowOff = (uint64_t)slot * QW_ROW + (uint64_t)kvh * QW_HEAD_DIM + d0;
                if constexpr (ELEMS == 4u) {
                    *(thread half4*)kk = *(const device half4*)(kpool + (uint64_t)pk * QW_PGEL + rowOff);
                    *(thread half4*)vv = *(const device half4*)(vpool + (uint64_t)pv * QW_PGEL + rowOff);
                } else {
                    *(thread uint4*)kk = *(const device uint4*)(kpool + (uint64_t)pk * QW_PGEL + rowOff);
                    *(thread uint4*)vv = *(const device uint4*)(vpool + (uint64_t)pv * QW_PGEL + rowOff);
                }
            } else if constexpr (ELEMS == 4u) {
                half k8[8], v8[8];
                const uint d8 = d0 & ~7u;
                const uint sub = (lane & 1u) * 4u;
                QW_KV_LOAD8(k8, kpool, pk, slot, kvh, d8, MODE)
                QW_KV_LOAD8(v8, vpool, pv, slot, kvh, d8, MODE)
                for (uint i = 0u; i < 4u; i++) { kk[i] = k8[sub + i]; vv[i] = v8[sub + i]; }
            } else {
                QW_KV_LOAD8(kk, kpool, pk, slot, kvh, d0, MODE)
                QW_KV_LOAD8(vv, vpool, pv, slot, kvh, d0, MODE)
            }
        } else {
            for (uint i = 0u; i < ELEMS; i++) { kk[i] = half(0.0h); vv[i] = half(0.0h); }
        }
        MBSA_MERGE(MBSA_DOT(kk), vv)
    }

    #undef MBSA_MERGE
    #undef MBSA_DOT

    if (splits > 1u) {
        const uint splitStride = max(64u, splits);
        device float* pb = partials
            + (uint64_t)qi * (QW_Q_HEADS * splitStride * (QW_HEAD_DIM + 2u))
            + (head * splitStride + tg.y) * (QW_HEAD_DIM + 2u);
        if (live) {
            for (uint i = 0u; i < ELEMS; i++) pb[d0 + i] = oacc[i];
        }
        if (lane == 0u) {
            pb[QW_HEAD_DIM] = m;
            pb[QW_HEAD_DIM + 1u] = lsum;
        }
    } else if constexpr (ELEMS == 4u) {
        alignas(16) half hv[4];
        if (live) {
            float gf[4];
            for (uint i = 0u; i < 4u; i++) gf[i] = 1.0f;
            if (QW_HAS_Q_GATE) {
                const device half* g4 = qrow + QW_HEAD_DIM + d0;
                for (uint i = 0u; i < 4u; i++) gf[i] = 1.0f / (1.0f + metal::exp(-float(g4[i])));
            }
            const float invL = (lsum > 0.0f) ? (1.0f / lsum) : 0.0f;
            for (uint i = 0u; i < 4u; i++) hv[i] = half(oacc[i] * invL * gf[i]);
        } else {
            for (uint i = 0u; i < 4u; i++) hv[i] = half(0.0h);
        }
        mbsa_fwht32_h4(hv, lane);
        if (live) {
            device half* ov = outp + (uint64_t)qi * (QW_Q_HEADS * QW_HEAD_DIM) + head * QW_HEAD_DIM + d0;
            *(device half4*)ov = *(thread const half4*)hv;
        }
    } else if (live) {
        float gf[8];
        for (uint i = 0u; i < 8u; i++) gf[i] = 1.0f;
        if (QW_HAS_Q_GATE) {
            const device half* g4 = qrow + QW_HEAD_DIM + d0;
            for (uint i = 0u; i < 8u; i++) gf[i] = 1.0f / (1.0f + metal::exp(-float(g4[i])));
        }
        const float invL = (lsum > 0.0f) ? (1.0f / lsum) : 0.0f;
        alignas(16) half hv[8];
        for (uint i = 0u; i < 8u; i++) hv[i] = half(oacc[i] * invL * gf[i]);
        fwht32_inplace(hv, lane);
        device half* ov = outp + (uint64_t)qi * (QW_Q_HEADS * QW_HEAD_DIM) + head * QW_HEAD_DIM + d0;
        *(device uint4*)ov = *(thread const uint4*)hv;
    }
}

template <uint MODE>
static inline void mbsa_decode_sparse_body(
        const device half* qgate, device half* kpool, device half* vpool,
        const device int* ktab, const device int* vtab,
        const device int* blockList, const device int* blockCount,
        device float* partials, device half* outp, uint4 params,
        uint tid, uint3 tg,
        threadgroup half* stageKV,
        threadgroup uint* sAbsKey, threadgroup uint* sPk, threadgroup uint* sPv) {
    if (QW_HEAD_DIM <= 128u) {
        mbsa_decode_sparse_impl<MODE, 4u>(
            qgate, kpool, vpool, ktab, vtab, blockList, blockCount,
            partials, outp, params, tid, tg,
            stageKV, stageKV + 16u * 128u, sAbsKey, sPk, sPv);
    } else {
        mbsa_decode_sparse_impl<MODE, 8u>(
            qgate, kpool, vpool, ktab, vtab, blockList, blockCount,
            partials, outp, params, tid, tg,
            stageKV, stageKV + 16u * QW_MBSA_DEC_HDIM, sAbsKey, sPk, sPv);
    }
}

#define QW_MBSA_DECODE_SPARSE_ENTRY(SFX, MODE) \
kernel void qwg_mbsa_decode_sparse##SFX( \
        const device half* qgate      [[buffer(0)]], \
        device half*       kpool      [[buffer(1)]], \
        device half*       vpool      [[buffer(2)]], \
        const device int*  ktab       [[buffer(3)]], \
        const device int*  vtab       [[buffer(4)]], \
        const device int*  blockList  [[buffer(5)]], \
        const device int*  blockCount [[buffer(6)]], \
        device float*      partials   [[buffer(7)]], \
        device half*       outp       [[buffer(8)]], \
        constant uint4&    params     [[buffer(9)]], \
        threadgroup half*  stageKV    [[threadgroup(0)]], \
        uint tid [[thread_index_in_threadgroup]], \
        uint3 tg [[threadgroup_position_in_grid]]) { \
    threadgroup uint sAbsKey[16], sPk[16], sPv[16]; \
    mbsa_decode_sparse_body<MODE>(qgate, kpool, vpool, ktab, vtab, blockList, blockCount, \
                                  partials, outp, params, tid, tg, stageKV, sAbsKey, sPk, sPv); \
}
QW_MBSA_DECODE_SPARSE_ENTRY(_f16, 0u)
QW_MBSA_DECODE_SPARSE_ENTRY(_q8,  1u)
QW_MBSA_DECODE_SPARSE_ENTRY(_q4,  2u)
#undef QW_MBSA_DECODE_SPARSE_ENTRY
