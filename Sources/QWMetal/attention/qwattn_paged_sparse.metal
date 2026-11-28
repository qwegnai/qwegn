// qwattn_paged_sparse.metal — Sparse paged KV decode and prefill kernels for LSA.
// Attends over selected 4-token blocks `blockList[0..<blockCount]` plus the
// unaligned tail `[4 * ((pos + 1) / 4) .. pos]`, supporting f16, q8, and q4 KV pools.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_attn_sparse_decode.metalh"

using namespace metal;

#if !defined(QW_DISABLE_NAX)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#if !defined(__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__) || __ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ < 260200
#error "NAX cooperative_tensor kernels require -mmacosx-version-min=26.2 or newer (Apple MPP operand_index ABI)."
#endif
#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "include/qw_attn_sparse_prefill_i8.metalh"

using namespace mlx::steel;

#define QW_LSA_NAX_HDIM     256u
#endif // !defined(QW_DISABLE_NAX)

// 1. Sparse Split-K Decode & Batched Decode over Paged KV Pool (f16, q8, q4)
#define QW_DECODE_SPARSE_ENTRY(SFX, MODE) \
kernel void qwg_attn_decode_sparse##SFX( \
        const device half* qgate         [[buffer(0)]], \
        device half*       kpool         [[buffer(1)]], \
        device half*       vpool         [[buffer(2)]], \
        const device int*  ktab          [[buffer(3)]], \
        const device int*  vtab          [[buffer(4)]], \
        const device half* knew          [[buffer(5)]], \
        const device half* vnew          [[buffer(6)]], \
        device float*      partials      [[buffer(7)]], \
        constant uint&     C             [[buffer(8)]], \
        constant uint&     splits        [[buffer(9)]], \
        device half*       outp          [[buffer(10)]], \
        const device int*  blockList     [[buffer(11)]], \
        const device uint* blockCountBuf [[buffer(12)]], \
        uint tid [[thread_index_in_threadgroup]], \
        uint3 tg [[threadgroup_position_in_grid]]) { \
    threadgroup half stageK[16u * QW_LSA_HEAD_DIM_MAX], stageV[16u * QW_LSA_HEAD_DIM_MAX]; \
    lsa_decode_sparse_core<MODE, false>(qgate, kpool, vpool, ktab, vtab, knew, vnew, partials, outp, \
                                        blockList, blockCountBuf, C, splits, 0u, true, tid, tg, stageK, stageV); \
} \
kernel void qwg_attn_decode_sparse_batched##SFX( \
        const device half* qgate         [[buffer(0)]], \
        device half*       kpool         [[buffer(1)]], \
        device half*       vpool         [[buffer(2)]], \
        const device int*  ktab          [[buffer(3)]], \
        const device int*  vtab          [[buffer(4)]], \
        const device half* knew          [[buffer(5)]], \
        const device half* vnew          [[buffer(6)]], \
        device float*      partials      [[buffer(7)]], \
        constant uint&     cBase         [[buffer(8)]], \
        constant uint&     splits        [[buffer(9)]], \
        device half*       outp          [[buffer(10)]], \
        const device int*  blockList     [[buffer(11)]], \
        const device uint* blockCountBuf [[buffer(12)]], \
        constant uint&     listStride    [[buffer(13)]], \
        constant uint&     writeKV       [[buffer(14)]], \
        uint tid [[thread_index_in_threadgroup]], \
        uint3 tg [[threadgroup_position_in_grid]]) { \
    threadgroup half stageK[16u * QW_LSA_HEAD_DIM_MAX], stageV[16u * QW_LSA_HEAD_DIM_MAX]; \
    lsa_decode_sparse_core<MODE, true>(qgate, kpool, vpool, ktab, vtab, knew, vnew, partials, outp, \
                                       blockList, blockCountBuf, cBase, splits, listStride, writeKV != 0u, tid, tg, stageK, stageV); \
}
QW_DECODE_SPARSE_ENTRY(, 0u)
QW_DECODE_SPARSE_ENTRY(_q8, 1u)
QW_DECODE_SPARSE_ENTRY(_q4, 2u)
#undef QW_DECODE_SPARSE_ENTRY

// -----------------------------------------------------------------------------
// 2. General Sparse Paged Prefill (scalar simdgroup per (head, query), f16/q8/q4)
// -----------------------------------------------------------------------------
kernel void qwg_attn_prefill_paged_general_sparse(
        const device half* qgate         [[buffer(0)]],
        device half*       kpool         [[buffer(1)]],
        device half*       vpool         [[buffer(2)]],
        const device int*  ktab          [[buffer(3)]],
        const device int*  vtab          [[buffer(4)]],
        device half*       outp          [[buffer(5)]],
        constant uint&     pos           [[buffer(6)]],
        constant uint&     q8mode        [[buffer(7)]],
        const device int*  blockList     [[buffer(8)]],
        const device uint* blockCountBuf [[buffer(9)]],
        constant uint&     listStride    [[buffer(10)]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 tg [[threadgroup_position_in_grid]]) {
    const uint head = tg.x;
    const uint qi   = tg.y;
    const uint lane = tid & 31u;
    const uint d0   = lane * 8u;
    const bool live = d0 < QW_HEAD_DIM;
    const uint kvh  = head / QW_GQA_RATIO;

    const device half* qrow = qgate + (uint64_t)qi * QW_QPROJ_OUT
                            + (uint64_t)head * QW_Q_STRIDE * QW_HEAD_DIM;
    float q8[8];
    for (uint i = 0u; i < 8u; i++) q8[i] = 0.0f;
    if (live) {
        for (uint i = 0u; i < 8u; i++) q8[i] = float(qrow[d0 + i]);
    }
    float gf[8];
    for (uint i = 0u; i < 8u; i++) gf[i] = 1.0f;
    if (QW_HAS_Q_GATE && live) {
        const device half* grow = qrow + QW_HEAD_DIM;
        for (uint i = 0u; i < 8u; i++) {
            gf[i] = 1.0f / (1.0f + metal::exp(-float(grow[d0 + i])));
        }
    }

    const uint kvRow  = QW_FC_KV_ROW;
    const uint kvPgel = QW_FC_KV_PGEL;
    const uint kvEnd  = pos + qi + 1u;
    const uint totalCompleteBlocks = kvEnd >> 2u;
    const uint uCount = min(blockCountBuf[qi], totalCompleteBlocks);
    const uint tailStart = totalCompleteBlocks * 4u;
    const uint nTail = (kvEnd > tailStart) ? (kvEnd - tailStart) : 0u;
    const uint totalSlots = uCount * 4u + nTail;
    const device int* myList = blockList + uint64_t(qi) * listStride;

    float m = -1e30f, lsum = 0.0f;
    float acc[8];
    for (uint i = 0u; i < 8u; i++) acc[i] = 0.0f;

    for (uint slotIdx = 0u; slotIdx < totalSlots; slotIdx++) {
        const uint j = (slotIdx < uCount * 4u)
            ? (uint(myList[slotIdx >> 2u]) * 4u + (slotIdx & 3u))
            : (tailStart + (slotIdx - uCount * 4u));
        if (j >= kvEnd) continue;
        const uint page = j >> 8u, slot = j & 255u;
        const uint pk = (uint)ktab[page];
        const uint pv = (uint)vtab[page];
        float k8[8], v8[8];
        for (uint i = 0u; i < 8u; i++) { k8[i] = 0.0f; v8[i] = 0.0f; }
        if (live) {
            if (q8mode == 0u) {
                const uint64_t rowOff = (uint64_t)slot * kvRow + (uint64_t)kvh * QW_HEAD_DIM;
                device const half* sk = kpool + (uint64_t)pk * kvPgel + rowOff;
                device const half* sv = vpool + (uint64_t)pv * kvPgel + rowOff;
                for (uint i = 0u; i < 8u; i++) { k8[i] = float(sk[d0 + i]); v8[i] = float(sv[d0 + i]); }
            } else {
                half kh[8], vh[8];
                QW_KV_LOAD8(kh, kpool, pk, slot, kvh, d0, q8mode)
                QW_KV_LOAD8(vh, vpool, pv, slot, kvh, d0, q8mode)
                for (uint i = 0u; i < 8u; i++) { k8[i] = float(kh[i]); v8[i] = float(vh[i]); }
            }
        }
        float dot = 0.0f;
        for (uint i = 0u; i < 8u; i++) dot += q8[i] * k8[i];
        const float s = simd_sum(dot) * QW_ATTN_SCALE;
        const float mnew = max(m, s);
        const float corr = (m > -1e29f) ? metal::exp(m - mnew) : 1.0f;
        const float p = metal::exp(s - mnew);
        for (uint i = 0u; i < 8u; i++) acc[i] = acc[i] * corr + p * v8[i];
        lsum = lsum * corr + p;
        m = mnew;
    }

    half hv[8];
    const float linv = (lsum > 0.0f) ? (1.0f / lsum) : 0.0f;
    for (uint i = 0u; i < 8u; i++) {
        hv[i] = live ? half(acc[i] * linv * gf[i]) : half(0.0f);
    }
    fwht32_inplace(hv, lane);
    if (live) {
        device half* o = outp + (uint64_t)qi * (QW_Q_HEADS * QW_HEAD_DIM)
                       + (uint64_t)head * QW_HEAD_DIM + d0;
        for (uint i = 0u; i < 8u; i++) o[i] = hv[i];
    }
}

#if !defined(QW_DISABLE_NAX)
// 3. M5 NAX Cooperative Tensor Sparse Paged Prefill (f16, q8, q4)
// Uses the tile's last query row `qRef = min(tile0 + 31, T - 1)` block list on 16x32 NAX MMA tiles.
template <uint MODE, uint STAGE_HDIM = QW_LSA_NAX_HDIM>
METAL_FUNC void lsa_prefill_paged_nax_sparse_body(
        const device half* qgate, const device half* kpool, const device half* vpool,
        const device int* ktab, const device int* vtab, device half* outp,
        constant uint& pos, constant uint& T,
        const device int* blockList, const device uint* blockCountBuf, constant uint& listStride,
        uint tid, uint sgi, uint3 tg, threadgroup half* kvblk, threadgroup half* vblk) {
    const uint kThreads   = 64u * QW_GQA_RATIO;
    const uint num_groups = QW_KV_HEADS;
    const uint kvh        = tg.x % num_groups;
    const uint tile0      = (tg.x / num_groups) * 32u;
    const uint headBase   = kvh * QW_GQA_RATIO;
    const uint qRef       = min(tile0 + 31u, T - 1u);
    const uint kEnd       = pos + min(tile0 + 32u, T);

    const uint totalCompleteBlocks = kEnd >> 2u;
    const uint uCount     = min(blockCountBuf[qRef], totalCompleteBlocks);
    const uint tailStart  = totalCompleteBlocks * 4u;
    const uint nTail      = (kEnd > tailStart) ? (kEnd - tailStart) : 0u;
    const uint totalSlots = uCount * 4u + nTail;
    const device int* myList = blockList + uint64_t(qRef) * listStride;

    const uint sgi_h     = sgi >> 1u;
    const uint sgi_local = sgi & 1u;
    const uint head      = headBase + sgi_h;
    const uint rowBase   = tile0 + sgi_local * QW_LSA_KU;

    const float scale2 = QW_ATTN_SCALE * 1.4426950408889634f;
    constexpr short kU = 16;
    constexpr short kMaxNq = short(STAGE_HDIM >> 4u);
    constexpr short kMaxPair = short(STAGE_HDIM >> 5u);
    const short nq = short(QW_HEAD_DIM >> 4u);
    const short npair = short(QW_HEAD_DIM >> 5u);

    const device half* qdev = qgate + (uint64_t)rowBase * QW_QPROJ_OUT
                              + (uint64_t)head * QW_Q_STRIDE * QW_HEAD_DIM;

    const short2 sc = BaseNAXFrag::get_coord();
    const short sm = sc.y, sn = sc.x;

    NAXTile<half, 1, 1> Qf[kMaxNq];
    const short validRows = short(clamp(int(T) - int(rowBase), 0, 16));
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < kMaxNq; id++) {
        if (id < nq) {
            if (validRows == 16)
                Qf[id].load(qdev + id * kU, int(QW_QPROJ_OUT));
            else
                Qf[id].load_safe(qdev + id * kU, int(QW_QPROJ_OUT), short2(kU, validRows));
        }
    }

    NAXTile<float, 1, kMaxNq> Otile;
    Otile.clear();

    float mrow[2], lrow[2];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) { mrow[i] = -1e30f; lrow[i] = 0.0f; }

    const uint kLoadVec8 = 4u * QW_HEAD_DIM;
    const uint kStageStride = STAGE_HDIM;

    for (uint s0 = 0u; s0 < totalSlots; s0 += QW_LSA_BK) {
        NAXTile<half, 1, 2> Pt;

        for (uint e = tid; e < kLoadVec8; e += kThreads) {
            const uint VecPerRow = QW_HEAD_DIM >> 3u;
            const uint kr = (QW_HEAD_DIM == 256u) ? (e >> 5u) : (e / VecPerRow);
            const uint dim0 = (QW_HEAD_DIM == 256u) ? ((e & 31u) << 3u) : ((e % VecPerRow) << 3u);
            const uint slotIdx = s0 + kr;
            uint absKey, pk, pv;
            lsa_sparse_resolve_slot_page(slotIdx, uCount, tailStart, totalSlots, kEnd,
                                         myList, ktab, vtab, tid, absKey, pk, pv);
            const uint ob = kr * kStageStride + dim0;
            if (slotIdx < totalSlots && absKey < kEnd) {
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

        NAXTile<float, 1, 2> Stile;
        Stile.clear();
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < kMaxNq; id++) {
            if (id < nq) {
                NAXTile<half, 2, 1> Kt;
                Kt.template load<half, STAGE_HDIM, 1>(kvblk + id * kU);
                BaseNAXFrag::mma(Stile.frag_at(0, 0),
                                 Stile.frag_at(0, 1),
                                 Qf[id].frag_at(0, 0), metal::false_type{},
                                 Kt.frag_at(0, 0), Kt.frag_at(1, 0),
                                 metal::true_type{});
            }
        }

        const uint lastSlotInStep = s0 + QW_LSA_BK - 1u;
        const uint maxAbsKeyInStep = (lastSlotInStep < uCount * 4u)
            ? (uint(myList[lastSlotInStep >> 2u]) * 4u + (lastSlotInStep & 3u))
            : (tailStart + (lastSlotInStep - uCount * 4u));
        const bool isUnmasked = (s0 + QW_LSA_BK <= totalSlots) && (maxAbsKeyInStep <= pos + rowBase);

        if (isUnmasked) {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < 2; ik++) {
                thread auto& fg = Stile.frag_at(0, ik);
                STEEL_PRAGMA_UNROLL
                for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                    fg[loc] *= scale2;
                }
            }
        } else {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < 2; ik++) {
                thread auto& fg = Stile.frag_at(0, ik);
                const uint col0 = s0 + (uint)(ik * kU) + (uint)sn;
                STEEL_PRAGMA_UNROLL
                for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                    const uint i = (uint)(loc >> 2), j = (uint)(loc & 3);
                    const uint slotC = col0 + j;
                    const uint abs_cc = (slotC < uCount * 4u)
                        ? (uint(myList[slotC >> 2u]) * 4u + (slotC & 3u))
                        : (tailStart + (slotC - uCount * 4u));
                    const uint abs_rr = pos + rowBase + (uint)sm + i * 8u;
                    fg[loc] = (slotC < totalSlots && abs_cc <= abs_rr && abs_cc < (pos + T))
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

        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            const bool grew = (mnew[i] > mrow[i]) && (mrow[i] > -1e29f);
            const float corr = grew ? fast::exp2(mrow[i] - mnew[i]) : 1.0f;
            lrow[i] = lrow[i] * corr + psum[i];
            if (grew) {
                STEEL_PRAGMA_UNROLL
                for (short id = 0; id < kMaxNq; id++) {
                    if (id < nq) {
                        thread auto& of = Otile.frag_at(0, id);
                        STEEL_PRAGMA_UNROLL
                        for (short loc = 0; loc < 4; loc++) {
                            of[i * 4 + loc] *= corr;
                        }
                    }
                }
            }
            mrow[i] = mnew[i];
        }

        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < 2; ik++) {
            STEEL_PRAGMA_UNROLL
            for (short idp = 0; idp < kMaxPair; idp++) {
                if (idp < npair) {
                    const short id0 = idp * 2;
                    NAXTile<half, 1, 2> Vt;
                    Vt.template load<half, STAGE_HDIM, 1>(vblk + (ik * kU) * kStageStride + id0 * kU);
                    BaseNAXFrag::mma(Otile.frag_at(0, id0),
                                     Otile.frag_at(0, id0 + 1),
                                     Pt.frag_at(0, ik), metal::false_type{},
                                     Vt.frag_at(0, 0), Vt.frag_at(0, 1),
                                     metal::false_type{});
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    lsa_sparse_nax_epilogue<STAGE_HDIM>(Otile, lrow, qdev, outp, rowBase, head, validRows, uint(sn), nq, npair);
}

#define QW_NAX_SPARSE_PREFILL_ENTRY(SFX, MODE, STAGE_HDIM) \
kernel void qwg_attn_prefill_paged_nax_sparse_pair##SFX( \
        const device half*  qgate         [[buffer(0)]], \
        const device half*  kpool         [[buffer(1)]], \
        const device half*  vpool         [[buffer(2)]], \
        const device int*   ktab          [[buffer(3)]], \
        const device int*   vtab          [[buffer(4)]], \
        device half*        outp          [[buffer(5)]], \
        constant uint&      pos           [[buffer(6)]], \
        constant uint&      T             [[buffer(7)]], \
        const device int*   blockList     [[buffer(8)]], \
        const device uint*  blockCountBuf [[buffer(9)]], \
        constant uint&      listStride    [[buffer(10)]], \
        uint tid [[thread_index_in_threadgroup]], \
        uint sgi [[simdgroup_index_in_threadgroup]], \
        uint3 tg [[threadgroup_position_in_grid]]) { \
    threadgroup half kvblk[QW_LSA_BK * STAGE_HDIM]; \
    threadgroup half vblk[QW_LSA_BK * STAGE_HDIM]; \
    if (MODE != 0u && (QW_HEAD_DIM & (QW_HEAD_DIM - 1u)) == 0u) { \
        threadgroup int8_t* kvblk_i8 = (threadgroup int8_t*)kvblk; \
        threadgroup float* kScale = (threadgroup float*)(kvblk_i8 + QW_LSA_BK * (STAGE_HDIM + 16u)); \
        lsa_prefill_paged_nax_sparse_i8_body<MODE, STAGE_HDIM>( \
            qgate, kpool, vpool, ktab, vtab, outp, pos, T, \
            blockList, blockCountBuf, listStride, tid, sgi, tg, kvblk_i8, kScale, vblk); \
    } else { \
        lsa_prefill_paged_nax_sparse_body<MODE, STAGE_HDIM>( \
            qgate, kpool, vpool, ktab, vtab, outp, pos, T, \
            blockList, blockCountBuf, listStride, tid, sgi, tg, kvblk, vblk); \
    } \
}
QW_NAX_SPARSE_PREFILL_ENTRY(, 0u, 256u)
QW_NAX_SPARSE_PREFILL_ENTRY(_q8, 1u, 256u)
QW_NAX_SPARSE_PREFILL_ENTRY(_q4, 2u, 256u)
QW_NAX_SPARSE_PREFILL_ENTRY(_hd128, 0u, 128u)
QW_NAX_SPARSE_PREFILL_ENTRY(_hd128_q8, 1u, 128u)
QW_NAX_SPARSE_PREFILL_ENTRY(_hd128_q4, 2u, 128u)
#undef QW_NAX_SPARSE_PREFILL_ENTRY
#endif // !defined(QW_DISABLE_NAX)
