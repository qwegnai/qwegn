// qwattn_paged_mma.metal — Apple9+ simdgroup_half8x8 chunked dense and sparse
// FlashAttention prefill over paged KV cache (f16/q8/q4, GQA, Q-gate, FWHT).
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "include/qw_geometry_constants.metalh"
#include "include/qw_kv_paged.metalh"
#include "include/qw_dequant.metalh"

using namespace metal;

#define QW_MMA_BK 32u

static inline uint qw_paged_mma_row(uint l) {
    return ((l >> 4) & 1u) * 4u + ((l >> 2) & 1u) * 2u + ((l >> 1) & 1u);
}

static inline uint qw_paged_mma_col(uint l) {
    return ((l >> 3) & 1u) * 4u + (l & 1u) * 2u;
}

static inline void qw_sparse_resolve_slot_page(
    uint slotIdx, uint uCount, uint tailStart, uint totalSlots, uint kEnd,
    const device int* myList, const device int* ktab, const device int* vtab,
    uint tid, thread uint& outAbsKey, thread uint& outPk, thread uint& outPv
) {
    uint absKey = 0u, pk = 0u, pv = 0u;
    const bool needWarp = (QW_HEAD_DIM == 256u) || (QW_HEAD_DIM == 128u);
    const bool leader = (QW_HEAD_DIM == 256u) ? ((tid & 31u) == 0u)
                      : (QW_HEAD_DIM == 128u) ? ((tid & 15u) == 0u) : true;
    if (leader && slotIdx < totalSlots) {
        absKey = (slotIdx < uCount * 4u)
            ? (uint(myList[slotIdx >> 2u]) * 4u + (slotIdx & 3u))
            : (tailStart + (slotIdx - uCount * 4u));
        if (absKey < kEnd) {
            const uint page = absKey >> 8u;
            pk = (uint)ktab[page];
            pv = (uint)vtab[page];
        }
    }
    if (needWarp) {
        const ushort src = (QW_HEAD_DIM == 256u) ? ushort(0) : ushort(tid & 16u);
        absKey = simd_shuffle(absKey, src);
        pk = simd_shuffle(pk, src);
        pv = simd_shuffle(pv, src);
    }
    outAbsKey = absKey;
    outPk = pk;
    outPv = pv;
}

template <uint STAGE_HDIM>
METAL_FUNC void qw_paged_mma_epilogue(
    thread simdgroup_float8x8 (*oacc)[STAGE_HDIM >> 3u],
    thread float* lrow,
    const device half* qgate, device half* outp,
    uint tile0, uint rowBase, uint headBase, uint head, uint sgi_h, uint sgi_local,
    uint T, uint rl, uint cl, uint tid, uint kThreads, uint nq8,
    threadgroup half* buf0, threadgroup half* buf1
) {
    float rcp[2] = {
        (lrow[0] > 0.0f) ? (1.0f / lrow[0]) : 0.0f,
        (lrow[1] > 0.0f) ? (1.0f / lrow[1]) : 0.0f
    };
    const uint numPasses = (QW_GQA_RATIO + 1u) / 2u;
    for (uint pass = 0u; pass < numPasses; pass++) {
        const uint h0 = headBase + pass * 2u;
        const uint h1 = h0 + 1u;
        const bool hasH1 = (pass * 2u + 1u) < QW_GQA_RATIO;
        if ((sgi_h >> 1u) == pass) {
            threadgroup half* obuf = ((sgi_h & 1u) == 0u) ? buf0 : buf1;
            for (uint bm = 0u; bm < 2u; bm++) {
                const uint rowInTile = sgi_local * 16u + bm * 8u + rl;
                const uint rchunk = tile0 + rowInTile;
                if (rchunk < T) {
                    const device half* grow = qgate
                        + (uint64_t)rchunk * QW_QPROJ_OUT
                        + (uint64_t)head * QW_Q_STRIDE * QW_HEAD_DIM + QW_HEAD_DIM;
                    for (uint d = 0u; d < nq8; d++) {
                        auto e = oacc[bm][d].thread_elements();
                        const uint col0 = d * 8u + cl;
                        float sig0 = 1.0f, sig1 = 1.0f;
                        if (QW_HAS_Q_GATE) {
                            sig0 = 1.0f / (1.0f + metal::exp(-float(grow[col0])));
                            sig1 = 1.0f / (1.0f + metal::exp(-float(grow[col0 + 1u])));
                        }
                        obuf[rowInTile * QW_HEAD_DIM + col0]      = half(float(e[0]) * rcp[bm] * sig0);
                        obuf[rowInTile * QW_HEAD_DIM + col0 + 1u] = half(float(e[1]) * rcp[bm] * sig1);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint fwhtThreads = min(kThreads, 256u);
        if (tid < fwhtThreads) {
            for (uint e = tid; e < 32u * QW_HEAD_DIM; e += fwhtThreads) {
                const uint rowInTile = (QW_HEAD_DIM == 256u) ? (e >> 8u) : (e / QW_HEAD_DIM);
                const uint r_out = tile0 + rowInTile;
                const uint c = (QW_HEAD_DIM == 256u) ? (e & 255u) : (e % QW_HEAD_DIM);
                const half rot0 = simd_fwht32(buf0[e], tid & 31u);
                const half rot1 = hasH1 ? simd_fwht32(buf1[e], tid & 31u) : half(0.0h);
                if (r_out < T) {
                    outp[(uint64_t)r_out * (QW_Q_HEADS * QW_HEAD_DIM) + h0 * QW_HEAD_DIM + c] = rot0;
                    if (hasH1) {
                        outp[(uint64_t)r_out * (QW_Q_HEADS * QW_HEAD_DIM) + h1 * QW_HEAD_DIM + c] = rot1;
                    }
                }
            }
        }
        if (pass + 1u < numPasses) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

template <uint MODE>
METAL_FUNC void qwg_attn_prefill_paged_mma_pair_body(
    const device half* qgate, const device half* kpool, const device half* vpool,
    const device int* ktab, const device int* vtab, device half* outp,
    uint pos, uint T, uint slidingWindow,
    uint tid, uint sgi, uint lane, uint3 tg,
    threadgroup half* kvblk, threadgroup half* vblk
) {
    constexpr uint STAGE_HDIM = 256u;
    constexpr uint NQ8_MAX = STAGE_HDIM >> 3u;
    const uint kThreads = 64u * QW_GQA_RATIO;
    const uint num_groups = QW_KV_HEADS;
    const uint kvh = tg.x % num_groups;
    const uint tile0 = (tg.x / num_groups) * 32u;
    const uint headBase = kvh * QW_GQA_RATIO;
    const uint kEnd = pos + min(tile0 + 32u, T);

    const uint sgi_h = sgi >> 1u;
    const uint sgi_local = sgi & 1u;
    const uint head = headBase + sgi_h;
    const uint rowBase = tile0 + sgi_local * 16u;
    const float scale2 = QW_ATTN_SCALE * 1.4426950408889634f;
    const uint nq8 = QW_HEAD_DIM >> 3u;
    const uint rl = qw_paged_mma_row(lane), cl = qw_paged_mma_col(lane);

    const device half* qdev = qgate + (uint64_t)rowBase * QW_QPROJ_OUT
                              + (uint64_t)head * QW_Q_STRIDE * QW_HEAD_DIM;
    simdgroup_float8x8 oacc[2][NQ8_MAX];
    for (uint bm = 0u; bm < 2u; bm++) {
        for (uint d = 0u; d < nq8; d++) oacc[bm][d] = simdgroup_float8x8(0.0f);
    }

    float mrow[2] = {-1e30f, -1e30f};
    float lrow[2] = {0.0f, 0.0f};
    const uint kLoadVec8 = 4u * QW_HEAD_DIM;
    const uint kStartRaw = (slidingWindow > 0u && (pos + tile0 + 1u) > slidingWindow)
        ? ((pos + tile0 + 1u) - slidingWindow) : 0u;
    const uint nStart = kStartRaw & ~(QW_MMA_BK - 1u);

    for (uint n0 = nStart; n0 < kEnd; n0 += QW_MMA_BK) {
        for (uint e = tid; e < kLoadVec8; e += kThreads) {
            const uint VecPerRow = QW_HEAD_DIM >> 3u;
            const uint kr = (QW_HEAD_DIM == 256u) ? (e >> 5u) : (e / VecPerRow);
            const uint dim0 = (QW_HEAD_DIM == 256u) ? ((e & 31u) << 3u) : ((e % VecPerRow) << 3u);
            const uint absKey = n0 + kr;
            const uint ob = kr * STAGE_HDIM + dim0;
            if (absKey < kEnd) {
                const uint page = n0 >> 8u, slot = absKey & 255u;
                const uint pk = (uint)ktab[page], pv = (uint)vtab[page];
                if constexpr (MODE == 0u) {
                    const uint64_t rowOff = (uint64_t)slot * QW_ROW + (uint64_t)kvh * QW_HEAD_DIM + dim0;
                    device const half* sk = kpool + (uint64_t)pk * QW_PGEL + rowOff;
                    device const half* sv = vpool + (uint64_t)pv * QW_PGEL + rowOff;
                    *(threadgroup half4*)(kvblk + ob)      = *(device const half4*)sk;
                    *(threadgroup half4*)(kvblk + ob + 4u) = *(device const half4*)(sk + 4u);
                    *(threadgroup half4*)(vblk + ob)       = *(device const half4*)sv;
                    *(threadgroup half4*)(vblk + ob + 4u)  = *(device const half4*)(sv + 4u);
                } else if constexpr (MODE == 1u) {
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

        simdgroup_float8x8 stile[2][4];
        for (uint bm = 0u; bm < 2u; bm++) {
            for (uint ik = 0u; ik < 4u; ik++) stile[bm][ik] = simdgroup_float8x8(0.0f);
        }
        for (uint d = 0u; d < nq8; d++) {
            simdgroup_half8x8 qf[2], kf[4];
            for (uint bm = 0u; bm < 2u; bm++) {
                if (rowBase + bm * 8u + 8u <= T) {
                    simdgroup_load(qf[bm], qdev + (uint64_t)(bm * 8u) * QW_QPROJ_OUT + d * 8u, QW_QPROJ_OUT);
                } else {
                    const uint r = rowBase + bm * 8u + rl;
                    auto qe = qf[bm].thread_elements();
                    const half2 q2 = (r < T)
                        ? *(const device half2*)(qdev + (uint64_t)(bm * 8u + rl) * QW_QPROJ_OUT + d * 8u + cl)
                        : half2(0.0h);
                    qe[0] = q2[0]; qe[1] = q2[1];
                    qf[bm].thread_elements() = qe;
                }
            }
            for (uint ik = 0u; ik < 4u; ik++) {
                simdgroup_load(kf[ik], kvblk + ik * 8u * STAGE_HDIM + d * 8u, STAGE_HDIM, ulong2(0, 0), true);
            }
            for (uint bm = 0u; bm < 2u; bm++) {
                for (uint ik = 0u; ik < 4u; ik++) {
                    simdgroup_multiply_accumulate(stile[bm][ik], qf[bm], kf[ik], stile[bm][ik]);
                }
            }
        }

        const uint maxRowPlus1 = pos + rowBase + 16u;
        const bool isUnmasked = (rowBase + 16u <= T)
            && (n0 + QW_MMA_BK <= pos + rowBase)
            && (slidingWindow == 0u || maxRowPlus1 <= n0 + slidingWindow);
        simdgroup_half8x8 pt[2][4];
        for (uint bm = 0u; bm < 2u; bm++) {
            float mnew = mrow[bm];
            const uint abs_rr = pos + rowBase + bm * 8u + rl;
            const bool rowLive = (rowBase + bm * 8u + rl) < T;
            for (uint ik = 0u; ik < 4u; ik++) {
                auto se = stile[bm][ik].thread_elements();
                if (isUnmasked) {
                    se[0] *= scale2; se[1] *= scale2;
                } else {
                    const uint col0 = n0 + ik * 8u + cl;
                    const bool inW0 = (slidingWindow == 0u || (abs_rr - col0) < slidingWindow);
                    const bool inW1 = (slidingWindow == 0u || (abs_rr - (col0 + 1u)) < slidingWindow);
                    se[0] = (rowLive && col0 <= abs_rr && col0 < pos + T && inW0) ? (se[0] * scale2) : -1e30f;
                    se[1] = (rowLive && (col0 + 1u) <= abs_rr && (col0 + 1u) < pos + T && inW1)
                          ? (se[1] * scale2) : -1e30f;
                }
                stile[bm][ik].thread_elements() = se;
                mnew = max(mnew, max(se[0], se[1]));
            }
            mnew = max(mnew, simd_shuffle_xor(mnew, ushort(1)));
            mnew = max(mnew, simd_shuffle_xor(mnew, ushort(8)));

            float psum = 0.0f;
            for (uint ik = 0u; ik < 4u; ik++) {
                const auto se = stile[bm][ik].thread_elements();
                const float x0 = (se[0] > -1e29f) ? fast::exp2(se[0] - mnew) : 0.0f;
                const float x1 = (se[1] > -1e29f) ? fast::exp2(se[1] - mnew) : 0.0f;
                psum += x0 + x1;
                auto pe = pt[bm][ik].thread_elements();
                pe[0] = half(x0); pe[1] = half(x1);
                pt[bm][ik].thread_elements() = pe;
            }
            psum += simd_shuffle_xor(psum, ushort(1));
            psum += simd_shuffle_xor(psum, ushort(8));

            const bool grew = (mnew > mrow[bm]) && (mrow[bm] > -1e29f);
            const float corr = (mrow[bm] <= -1e29f) ? 0.0f : (grew ? fast::exp2(mrow[bm] - mnew) : 1.0f);
            lrow[bm] = lrow[bm] * corr + psum;
            if (grew) {
                for (uint d = 0u; d < nq8; d++) {
                    auto oe = oacc[bm][d].thread_elements();
                    oe[0] *= corr; oe[1] *= corr;
                    oacc[bm][d].thread_elements() = oe;
                }
            }
            mrow[bm] = mnew;
        }

        for (uint ik = 0u; ik < 4u; ik++) {
            for (uint d = 0u; d < nq8; d++) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf, vblk + ik * 8u * STAGE_HDIM + d * 8u, STAGE_HDIM);
                simdgroup_multiply_accumulate(oacc[0][d], pt[0][ik], vf, oacc[0][d]);
                simdgroup_multiply_accumulate(oacc[1][d], pt[1][ik], vf, oacc[1][d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    qw_paged_mma_epilogue<STAGE_HDIM>(
        oacc, lrow, qgate, outp, tile0, rowBase, headBase, head, sgi_h, sgi_local,
        T, rl, cl, tid, kThreads, nq8, kvblk, vblk);
}

#define QW_PAGED_MMA_PREFILL_ENTRY(SFX, MODE) \
kernel void qwg_attn_prefill_paged_mma_pair##SFX( \
    const device half* qgate [[buffer(0)]], const device half* kpool [[buffer(1)]], \
    const device half* vpool [[buffer(2)]], const device int* ktab [[buffer(3)]], \
    const device int* vtab [[buffer(4)]], device half* outp [[buffer(5)]], \
    constant uint& pos [[buffer(6)]], constant uint& T [[buffer(7)]], \
    constant uint& slidingWindow [[buffer(8)]], \
    uint tid [[thread_index_in_threadgroup]], uint sgi [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]] \
) { \
    threadgroup half kvblk[QW_MMA_BK * 256u]; \
    threadgroup half vblk[QW_MMA_BK * 256u]; \
    qwg_attn_prefill_paged_mma_pair_body<MODE>( \
        qgate, kpool, vpool, ktab, vtab, outp, pos, T, slidingWindow, tid, sgi, lane, tg, kvblk, vblk); \
}
QW_PAGED_MMA_PREFILL_ENTRY(, 0u)
QW_PAGED_MMA_PREFILL_ENTRY(_q8, 1u)
QW_PAGED_MMA_PREFILL_ENTRY(_q4, 2u)
#undef QW_PAGED_MMA_PREFILL_ENTRY

template <uint MODE, uint STAGE_HDIM>
METAL_FUNC void qwg_attn_prefill_paged_mma_sparse_body(
    const device half* qgate, const device half* kpool, const device half* vpool,
    const device int* ktab, const device int* vtab, device half* outp,
    uint pos, uint T, const device int* blockList, const device uint* blockCountBuf, uint listStride,
    uint tid, uint sgi, uint lane, uint3 tg,
    threadgroup half* kvblk, threadgroup half* vblk
) {
    constexpr uint NQ8_MAX = STAGE_HDIM >> 3u;
    const uint kThreads = 64u * QW_GQA_RATIO;
    const uint num_groups = QW_KV_HEADS;
    const uint kvh = tg.x % num_groups;
    const uint tile0 = (tg.x / num_groups) * 32u;
    const uint headBase = kvh * QW_GQA_RATIO;
    const uint qRef = min(tile0 + 31u, T - 1u);
    const uint kEnd = pos + min(tile0 + 32u, T);

    const uint totalCompleteBlocks = kEnd >> 2u;
    const uint uCount = min(blockCountBuf[qRef], totalCompleteBlocks);
    const uint tailStart = totalCompleteBlocks * 4u;
    const uint nTail = (kEnd > tailStart) ? (kEnd - tailStart) : 0u;
    const uint totalSlots = uCount * 4u + nTail;
    const device int* myList = blockList + (uint64_t)qRef * listStride;

    const uint sgi_h = sgi >> 1u;
    const uint sgi_local = sgi & 1u;
    const uint head = headBase + sgi_h;
    const uint rowBase = tile0 + sgi_local * 16u;
    const float scale2 = QW_ATTN_SCALE * 1.4426950408889634f;
    const uint nq8 = QW_HEAD_DIM >> 3u;
    const uint rl = qw_paged_mma_row(lane), cl = qw_paged_mma_col(lane);

    const device half* qdev = qgate + (uint64_t)rowBase * QW_QPROJ_OUT
                              + (uint64_t)head * QW_Q_STRIDE * QW_HEAD_DIM;
    simdgroup_float8x8 oacc[2][NQ8_MAX];
    for (uint bm = 0u; bm < 2u; bm++) {
        for (uint d = 0u; d < nq8; d++) oacc[bm][d] = simdgroup_float8x8(0.0f);
    }

    float mrow[2] = {-1e30f, -1e30f};
    float lrow[2] = {0.0f, 0.0f};
    const uint kLoadVec8 = 4u * QW_HEAD_DIM;

    for (uint s0 = 0u; s0 < totalSlots; s0 += QW_MMA_BK) {
        for (uint e = tid; e < kLoadVec8; e += kThreads) {
            const uint VecPerRow = QW_HEAD_DIM >> 3u;
            const uint kr = (QW_HEAD_DIM == 256u) ? (e >> 5u) : (e / VecPerRow);
            const uint dim0 = (QW_HEAD_DIM == 256u) ? ((e & 31u) << 3u) : ((e % VecPerRow) << 3u);
            const uint slotIdx = s0 + kr;
            uint absKey, pk, pv;
            qw_sparse_resolve_slot_page(slotIdx, uCount, tailStart, totalSlots, kEnd,
                                        myList, ktab, vtab, tid, absKey, pk, pv);
            const uint ob = kr * STAGE_HDIM + dim0;
            if (slotIdx < totalSlots && absKey < kEnd) {
                const uint slot = absKey & 255u;
                if constexpr (MODE == 0u) {
                    const uint64_t rowOff = (uint64_t)slot * QW_ROW + (uint64_t)kvh * QW_HEAD_DIM + dim0;
                    device const half* sk = kpool + (uint64_t)pk * QW_PGEL + rowOff;
                    device const half* sv = vpool + (uint64_t)pv * QW_PGEL + rowOff;
                    *(threadgroup half4*)(kvblk + ob)      = *(device const half4*)sk;
                    *(threadgroup half4*)(kvblk + ob + 4u) = *(device const half4*)(sk + 4u);
                    *(threadgroup half4*)(vblk + ob)       = *(device const half4*)sv;
                    *(threadgroup half4*)(vblk + ob + 4u)  = *(device const half4*)(sv + 4u);
                } else if constexpr (MODE == 1u) {
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

        simdgroup_float8x8 stile[2][4];
        for (uint bm = 0u; bm < 2u; bm++) {
            for (uint ik = 0u; ik < 4u; ik++) stile[bm][ik] = simdgroup_float8x8(0.0f);
        }
        for (uint d = 0u; d < nq8; d++) {
            simdgroup_half8x8 qf[2], kf[4];
            for (uint bm = 0u; bm < 2u; bm++) {
                if (rowBase + bm * 8u + 8u <= T) {
                    simdgroup_load(qf[bm], qdev + (uint64_t)(bm * 8u) * QW_QPROJ_OUT + d * 8u, QW_QPROJ_OUT);
                } else {
                    const uint r = rowBase + bm * 8u + rl;
                    auto qe = qf[bm].thread_elements();
                    const half2 q2 = (r < T)
                        ? *(const device half2*)(qdev + (uint64_t)(bm * 8u + rl) * QW_QPROJ_OUT + d * 8u + cl)
                        : half2(0.0h);
                    qe[0] = q2[0]; qe[1] = q2[1];
                    qf[bm].thread_elements() = qe;
                }
            }
            for (uint ik = 0u; ik < 4u; ik++) {
                simdgroup_load(kf[ik], kvblk + ik * 8u * STAGE_HDIM + d * 8u, STAGE_HDIM, ulong2(0, 0), true);
            }
            for (uint bm = 0u; bm < 2u; bm++) {
                for (uint ik = 0u; ik < 4u; ik++) {
                    simdgroup_multiply_accumulate(stile[bm][ik], qf[bm], kf[ik], stile[bm][ik]);
                }
            }
        }

        const uint lastSlotInStep = s0 + QW_MMA_BK - 1u;
        const uint maxAbsKey = (lastSlotInStep < uCount * 4u)
            ? (uint(myList[lastSlotInStep >> 2u]) * 4u + (lastSlotInStep & 3u))
            : (tailStart + (lastSlotInStep - uCount * 4u));
        const bool isUnmasked = (rowBase + 16u <= T)
            && (s0 + QW_MMA_BK <= totalSlots) && (maxAbsKey <= pos + rowBase);

        simdgroup_half8x8 pt[2][4];
        for (uint bm = 0u; bm < 2u; bm++) {
            float mnew = mrow[bm];
            const uint abs_rr = pos + rowBase + bm * 8u + rl;
            const bool rowLive = (rowBase + bm * 8u + rl) < T;
            for (uint ik = 0u; ik < 4u; ik++) {
                auto se = stile[bm][ik].thread_elements();
                if (isUnmasked) {
                    se[0] *= scale2; se[1] *= scale2;
                } else {
                    const uint slotC0 = s0 + ik * 8u + cl;
                    const uint slotC1 = slotC0 + 1u;
                    const uint abs0 = (slotC0 < uCount * 4u)
                        ? (uint(myList[slotC0 >> 2u]) * 4u + (slotC0 & 3u)) : (tailStart + (slotC0 - uCount * 4u));
                    const uint abs1 = (slotC1 < uCount * 4u)
                        ? (uint(myList[slotC1 >> 2u]) * 4u + (slotC1 & 3u)) : (tailStart + (slotC1 - uCount * 4u));
                    se[0] = (rowLive && slotC0 < totalSlots && abs0 <= abs_rr && abs0 < pos + T)
                          ? (se[0] * scale2) : -1e30f;
                    se[1] = (rowLive && slotC1 < totalSlots && abs1 <= abs_rr && abs1 < pos + T)
                          ? (se[1] * scale2) : -1e30f;
                }
                stile[bm][ik].thread_elements() = se;
                mnew = max(mnew, max(se[0], se[1]));
            }
            mnew = max(mnew, simd_shuffle_xor(mnew, ushort(1)));
            mnew = max(mnew, simd_shuffle_xor(mnew, ushort(8)));

            float psum = 0.0f;
            for (uint ik = 0u; ik < 4u; ik++) {
                const auto se = stile[bm][ik].thread_elements();
                const float x0 = (se[0] > -1e29f) ? fast::exp2(se[0] - mnew) : 0.0f;
                const float x1 = (se[1] > -1e29f) ? fast::exp2(se[1] - mnew) : 0.0f;
                psum += x0 + x1;
                auto pe = pt[bm][ik].thread_elements();
                pe[0] = half(x0); pe[1] = half(x1);
                pt[bm][ik].thread_elements() = pe;
            }
            psum += simd_shuffle_xor(psum, ushort(1));
            psum += simd_shuffle_xor(psum, ushort(8));

            const bool grew = (mnew > mrow[bm]) && (mrow[bm] > -1e29f);
            const float corr = (mrow[bm] <= -1e29f) ? 0.0f : (grew ? fast::exp2(mrow[bm] - mnew) : 1.0f);
            lrow[bm] = lrow[bm] * corr + psum;
            if (grew) {
                for (uint d = 0u; d < nq8; d++) {
                    auto oe = oacc[bm][d].thread_elements();
                    oe[0] *= corr; oe[1] *= corr;
                    oacc[bm][d].thread_elements() = oe;
                }
            }
            mrow[bm] = mnew;
        }

        for (uint ik = 0u; ik < 4u; ik++) {
            for (uint d = 0u; d < nq8; d++) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf, vblk + ik * 8u * STAGE_HDIM + d * 8u, STAGE_HDIM);
                simdgroup_multiply_accumulate(oacc[0][d], pt[0][ik], vf, oacc[0][d]);
                simdgroup_multiply_accumulate(oacc[1][d], pt[1][ik], vf, oacc[1][d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    qw_paged_mma_epilogue<STAGE_HDIM>(
        oacc, lrow, qgate, outp, tile0, rowBase, headBase, head, sgi_h, sgi_local,
        T, rl, cl, tid, kThreads, nq8, kvblk, vblk);
}

#define QW_PAGED_MMA_SPARSE_ENTRY(SFX, MODE, STAGE_HDIM) \
kernel void qwg_attn_prefill_paged_mma_sparse_pair##SFX( \
    const device half* qgate [[buffer(0)]], const device half* kpool [[buffer(1)]], \
    const device half* vpool [[buffer(2)]], const device int* ktab [[buffer(3)]], \
    const device int* vtab [[buffer(4)]], device half* outp [[buffer(5)]], \
    constant uint& pos [[buffer(6)]], constant uint& T [[buffer(7)]], \
    const device int* blockList [[buffer(8)]], const device uint* blockCountBuf [[buffer(9)]], \
    constant uint& listStride [[buffer(10)]], \
    uint tid [[thread_index_in_threadgroup]], uint sgi [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]] \
) { \
    threadgroup half kvblk[QW_MMA_BK * STAGE_HDIM]; \
    threadgroup half vblk[QW_MMA_BK * STAGE_HDIM]; \
    qwg_attn_prefill_paged_mma_sparse_body<MODE, STAGE_HDIM>( \
        qgate, kpool, vpool, ktab, vtab, outp, pos, T, blockList, blockCountBuf, listStride, \
        tid, sgi, lane, tg, kvblk, vblk); \
}
QW_PAGED_MMA_SPARSE_ENTRY(, 0u, 256u)
QW_PAGED_MMA_SPARSE_ENTRY(_q8, 1u, 256u)
QW_PAGED_MMA_SPARSE_ENTRY(_q4, 2u, 256u)
QW_PAGED_MMA_SPARSE_ENTRY(_hd128, 0u, 128u)
QW_PAGED_MMA_SPARSE_ENTRY(_hd128_q8, 1u, 128u)
QW_PAGED_MMA_SPARSE_ENTRY(_hd128_q4, 2u, 128u)
#undef QW_PAGED_MMA_SPARSE_ENTRY
