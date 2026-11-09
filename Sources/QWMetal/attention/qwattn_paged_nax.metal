// qwattn_paged_nax.metal — chunked prefill FlashAttention over paged KV cache.
// Causal-masked within the chunk, full attention over the cached prefix, one
// kernel for fresh prefill and continuation. NAX MMA tiles, f16/q8/q4 modes via
// qw_kv_paged.metalh; online INT8 Q·K^T for long contexts (pos + T >= 192).
// Law: see docs/KERNEL_NOTES.md#qwattn_paged_nax_metal.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "include/qw_geometry_constants.metalh"
#include "include/qw_kv_paged.metalh"
#include "include/qw_dequant.metalh"

using namespace metal;
using namespace mlx::steel;

#define QW_KU       16u                          // NAX frag edge
#define QW_BK       32u                          // keys per online-softmax block
#define QW_NAX_HDIM 256u                         // NAX tile head_dim = 16 frags × 16

template <uint MODE>
METAL_FUNC void qwg_attn_prefill_paged_nax_pair_f16_body(
    const device half*  qgate,
    const device half*  kpool,
    const device half*  vpool,
    const device int*   ktab,
    const device int*   vtab,
    device half*        outp,
    constant uint&      pos,
    constant uint&      T,
    uint tid,
    uint sgi,
    uint3 tg,
    threadgroup half*   kvblk,
    threadgroup half*   vblk) {

    const uint kThreads   = 64u * QW_GQA_RATIO;
    const uint num_groups = QW_KV_HEADS;
    const uint kvh        = tg.x % num_groups;
    const uint tile0      = (tg.x / num_groups) * 32u;
    const uint headBase   = kvh * QW_GQA_RATIO;
    const uint kEnd       = pos + min(tile0 + 32u, T);

    const uint sgi_h      = sgi >> 1;
    const uint sgi_local  = sgi & 1u;
    const uint head       = headBase + sgi_h;
    const uint rowBase    = tile0 + sgi_local * QW_KU;

    const float scale2 = QW_ATTN_SCALE * 1.4426950408889634f;
    constexpr short kU = 16;
    const short nq = short(QW_HEAD_DIM >> 4);
    const short npair = short(QW_HEAD_DIM >> 5);

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
    const uint kStageStride = QW_NAX_HDIM;

    for (uint n0 = 0u; n0 < kEnd; n0 += QW_BK) {
        NAXTile<half, 1, 2> Pt;

        for (uint e = tid; e < kLoadVec8; e += kThreads) {
            const uint VecPerRow = QW_HEAD_DIM >> 3;
            const uint kr = (QW_HEAD_DIM == 256u) ? (e >> 5) : (e / VecPerRow);
            const uint dim0 = (QW_HEAD_DIM == 256u) ? ((e & 31u) << 3) : ((e % VecPerRow) << 3);
            const uint absKey = n0 + kr;
            const uint ob = kr * kStageStride + dim0;
            if (absKey < kEnd) {
                const uint page = n0 >> 8;
                const uint slot = absKey & 255u;
                const uint pk = (uint)ktab[page];
                const uint pv = (uint)vtab[page];
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
        for (short id = 0; id < 16; id++) {
            if (id < nq) {
                NAXTile<half, 2, 1> Kt;
                Kt.template load<half, QW_NAX_HDIM, 1>(kvblk + id * kU);
                BaseNAXFrag::mma(Stile.frag_at(0, 0),
                                 Stile.frag_at(0, 1),
                                 Qf[id].frag_at(0, 0), metal::false_type{},
                                 Kt.frag_at(0, 0), Kt.frag_at(1, 0),
                                 metal::true_type{});
            }
        }

        const bool isUnmasked = (n0 + QW_BK <= pos + rowBase);
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
                const uint col0 = n0 + (uint)(ik * kU) + (uint)sn;
                STEEL_PRAGMA_UNROLL
                for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                    const uint i = (uint)(loc >> 2), j = (uint)(loc & 3);
                    const uint abs_cc = col0 + j;
                    const uint abs_rr = pos + rowBase + (uint)sm + i * 8u;
                    fg[loc] = (abs_cc <= abs_rr && abs_cc < (pos + T)) ? (fg[loc] * scale2) : (-1e30f);
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
                for (short d = 0; d < 16; d++) {
                    if (d < nq) {
                        thread auto& ofg = Otile.frag_at(0, d);
                        STEEL_PRAGMA_UNROLL
                        for (short j = 0; j < 4; j++)
                            ofg[i * 4 + j] *= corr;
                    }
                }
            }
            mrow[i] = mnew[i];
        }

        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < short(QW_BK / kU); ik++) {
            STEEL_PRAGMA_UNROLL
            for (short pair = 0; pair < 8; pair++) {
                if (pair < npair) {
                    NAXTile<half, 1, 2> Vt;
                    Vt.template load<half, QW_NAX_HDIM, 1>(vblk + (uint(ik * kU)) * kStageStride + pair * (2 * kU));
                    BaseNAXFrag::mma(Otile.frag_at(0, 2 * pair),
                                     Otile.frag_at(0, 2 * pair + 1),
                                     Pt.frag_at(0, ik), metal::false_type{},
                                     Vt.frag_at(0, 0), Vt.frag_at(0, 1),
                                     metal::false_type{});
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rcp[2];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) rcp[i] = (lrow[i] > 0.0f) ? (1.0f / lrow[i]) : 0.0f;

    const uint numPasses = (QW_GQA_RATIO + 1u) / 2u;
    for (uint pass = 0u; pass < numPasses; pass++) {
        const uint h0 = headBase + pass * 2u;
        const uint h1 = h0 + 1u;
        const bool hasH1 = (pass * 2u + 1u) < QW_GQA_RATIO;
        if ((sgi_h >> 1u) == pass) {
            threadgroup half* obuf = ((sgi_h & 1u) == 0u) ? kvblk : vblk;
            STEEL_PRAGMA_UNROLL
            for (short d = 0; d < 16; d++) {
                if (d < nq) {
                    const uint col0 = (uint)(d * kU) + (uint)sn;
                    STEEL_PRAGMA_UNROLL
                    for (short i = 0; i < 2; i++) {
                        const uint rowInTile = sgi_local * 16u + (uint)sm + (uint)i * 8u;
                        const uint rchunk = tile0 + rowInTile;
                        if (rchunk < T) {
                            const device half* grow = qgate
                                + (uint64_t)rchunk * QW_QPROJ_OUT
                                + (uint64_t)head * QW_Q_STRIDE * QW_HEAD_DIM + QW_HEAD_DIM;
                            thread auto& fg = Otile.frag_at(0, d);
                            STEEL_PRAGMA_UNROLL
                            for (short j = 0; j < 4; j++) {
                                float sig = 1.0f;
                                if (QW_HAS_Q_GATE) {
                                    const half gval = grow[col0 + j];
                                    sig = 1.0f / (1.0f + metal::exp(-float(gval)));
                                }
                                obuf[rowInTile * QW_HEAD_DIM + col0 + j] = half(fg[i * 4 + j] * rcp[i] * sig);
                            }
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint fwhtThreads = min(kThreads, 256u);
        if (tid < fwhtThreads) {
            for (uint e = tid; e < 32u * QW_HEAD_DIM; e += fwhtThreads) {
                const uint rowInTile = (QW_HEAD_DIM == 256u) ? (e >> 8) : (e / QW_HEAD_DIM);
                const uint r_out = tile0 + rowInTile;
                const uint c = (QW_HEAD_DIM == 256u) ? (e & 255u) : (e % QW_HEAD_DIM);
                const half rot0 = simd_fwht32(kvblk[e], tid & 31u);
                const half rot1 = hasH1 ? simd_fwht32(vblk[e], tid & 31u) : half(0.0h);
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
METAL_FUNC void qwg_attn_prefill_paged_nax_pair_i8_body(
    const device half*  qgate,
    const device half*  kpool,
    const device half*  vpool,
    const device int*   ktab,
    const device int*   vtab,
    device half*        outp,
    constant uint&      pos,
    constant uint&      T,
    uint tid,
    uint sgi,
    uint3 tg,
    threadgroup int8_t* kvblk_i8,
    threadgroup float*  kScale,
    threadgroup half*   vblk,
    threadgroup half*   obuf1) {

    const uint kThreads   = 64u * QW_GQA_RATIO;
    const uint num_groups = QW_KV_HEADS;
    const uint kvh        = tg.x % num_groups;
    const uint tile0      = (tg.x / num_groups) * 32u;
    const uint headBase   = kvh * QW_GQA_RATIO;
    const uint kEnd       = pos + min(tile0 + 32u, T);

    const uint sgi_h      = sgi >> 1;
    const uint sgi_local  = sgi & 1u;
    const uint head       = headBase + sgi_h;
    const uint rowBase    = tile0 + sgi_local * QW_KU;

    const float scale2 = QW_ATTN_SCALE * 1.4426950408889634f;
    constexpr short kU = 16;
    const short nq = short(QW_HEAD_DIM >> 4);
    const short npair = short(QW_HEAD_DIM >> 5);

    const device half* qdev = qgate + (uint64_t)rowBase * QW_QPROJ_OUT
                              + (uint64_t)head * QW_Q_STRIDE * QW_HEAD_DIM;

    const short2 sc = BaseNAXFrag::get_coord();
    const short sm = sc.y, sn = sc.x;

    NAXTile<int8_t, 1, 1> Qfi8[16];
    float qScale2[2];
    {
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
        float qmax[2] = {0.0f, 0.0f};
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < 16; id++) {
            if (id < nq) {
                thread auto& fg = Qf[id].frag_at(0, 0);
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; i++) {
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; j++) {
                        qmax[i] = fmax(qmax[i], fabs(float(fg[i * 4 + j])));
                    }
                }
            }
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            qmax[i] = max(qmax[i], simd_shuffle_xor(qmax[i], ushort(1)));
            qmax[i] = max(qmax[i], simd_shuffle_xor(qmax[i], ushort(8)));
        }
        float qInv[2];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            qScale2[i] = (qmax[i] > 0.0f) ? ((qmax[i] / 127.0f) * scale2) : 0.0f;
            qInv[i]    = (qmax[i] > 0.0f) ? (127.0f / qmax[i]) : 0.0f;
        }
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < 16; id++) {
            if (id < nq) {
                thread auto& src = Qf[id].frag_at(0, 0);
                thread auto& dst = Qfi8[id].frag_at(0, 0);
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; i++) {
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; j++) {
                        dst[i * 4 + j] = int8_t(rint(float(src[i * 4 + j]) * qInv[i]));
                    }
                }
            }
        }
    }

    NAXTile<float, 1, 16> Otile;
    Otile.clear();

    float mrow[2], lrow[2];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) { mrow[i] = -1e30f; lrow[i] = 0.0f; }

    const uint kLoadVec8 = 4u * QW_HEAD_DIM;
    constexpr uint kStageStrideI8 = QW_NAX_HDIM + 16u;
    constexpr uint kStageStrideV  = QW_NAX_HDIM;

    for (uint n0 = 0u; n0 < kEnd; n0 += QW_BK) {
        NAXTile<half, 1, 2> Pt;

        for (uint e = tid; e < kLoadVec8; e += kThreads) {
            const uint VecPerRow = QW_HEAD_DIM >> 3;
            const uint kr = (QW_HEAD_DIM == 256u) ? (e >> 5) : (e / VecPerRow);
            const uint laneInRow = (QW_HEAD_DIM == 256u) ? (e & 31u) : (e % VecPerRow);
            const uint dim0 = laneInRow << 3;
            const uint absKey = n0 + kr;
            const uint obI8 = kr * kStageStrideI8 + dim0;
            const uint obV  = kr * kStageStrideV  + dim0;
            half kh[8];
            if (absKey < kEnd) {
                const uint page = n0 >> 8;
                const uint slot = absKey & 255u;
                const uint pk = (uint)ktab[page];
                const uint pv = (uint)vtab[page];
                if (MODE == 0u) {
                    const uint64_t rowOff = (uint64_t)slot * QW_ROW + (uint64_t)kvh * QW_HEAD_DIM + dim0;
                    device const half* sk = kpool + (uint64_t)pk * QW_PGEL + rowOff;
                    device const half* sv = vpool + (uint64_t)pv * QW_PGEL + rowOff;
                    *(thread half4*)(kh)                   = *(device const half4*)sk;
                    *(thread half4*)(kh + 4u)              = *(device const half4*)(sk + 4u);
                    *(threadgroup half4*)(vblk + obV)      = *(device const half4*)sv;
                    *(threadgroup half4*)(vblk + obV + 4u) = *(device const half4*)(sv + 4u);
                } else if (MODE == 1u) {
                    const uint64_t pageOff = (uint64_t)pk * QW_Q8_PGE;
                    const device char* sk8 = (const device char*)kpool + pageOff
                                             + (uint64_t)slot * QW_ROW8 + (uint64_t)kvh * QW_HEAD_DIM + dim0;
                    *(threadgroup char4*)(kvblk_i8 + obI8)      = *(const device char4*)sk8;
                    *(threadgroup char4*)(kvblk_i8 + obI8 + 4u) = *(const device char4*)(sk8 + 4u);
                    if (laneInRow == 0u) {
                        kScale[kr] = float(*(const device half*)((const device char*)kpool + pageOff + QW_Q8_HDR
                                           + (uint64_t)slot * QW_Q8_HROW + (uint64_t)kvh * 2u));
                    }
                    qwg_q8_load8((const device char*)vpool, (uint64_t)pv * QW_Q8_PGE, slot, kvh, dim0, vblk, obV);
                } else {
                    qwg_q4_load8((const device uchar*)kpool, (uint64_t)pk * QW_Q4_PGE, slot, kvh, dim0, kh, 0u);
                    qwg_q4_load8((const device uchar*)vpool, (uint64_t)pv * QW_Q4_PGE, slot, kvh, dim0, vblk, obV);
                }
            } else {
                STEEL_PRAGMA_UNROLL
                for (uint i = 0; i < 8u; i++) kh[i] = half(0.0h);
                if (MODE == 1u) {
                    *(threadgroup char4*)(kvblk_i8 + obI8)      = char4(0);
                    *(threadgroup char4*)(kvblk_i8 + obI8 + 4u) = char4(0);
                    if (laneInRow == 0u) kScale[kr] = 0.0f;
                }
                *(threadgroup half4*)(vblk + obV)      = half4(0.0h);
                *(threadgroup half4*)(vblk + obV + 4u) = half4(0.0h);
            }
            if (MODE != 1u) {
                float amax = 0.0f;
                STEEL_PRAGMA_UNROLL
                for (uint i = 0; i < 8u; i++) amax = fmax(amax, fabs(float(kh[i])));
                if (QW_HEAD_DIM == 256u) {
                    amax = simd_max(amax);
                } else {
                    for (uint off = VecPerRow >> 1; off > 0u; off >>= 1)
                        amax = fmax(amax, simd_shuffle_xor(amax, ushort(off)));
                }
                const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                char4 q0, q1;
                STEEL_PRAGMA_UNROLL
                for (uint i = 0; i < 4u; i++) {
                    q0[i] = char(int(rint(float(kh[i]) * inv)));
                    q1[i] = char(int(rint(float(kh[i + 4u]) * inv)));
                }
                *(threadgroup char4*)(kvblk_i8 + obI8)      = q0;
                *(threadgroup char4*)(kvblk_i8 + obI8 + 4u) = q1;
                if (laneInRow == 0u) kScale[kr] = amax / 127.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        NAXTile<int32_t, 1, 2> Stile_i32;
        Stile_i32.clear();
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < 16; id++) {
            if (id < nq) {
                NAXTile<int8_t, 2, 1> Kt;
                Kt.template load<int8_t, kStageStrideI8, 1>(kvblk_i8 + id * kU);
                BaseNAXFrag::mma(Stile_i32.frag_at(0, 0),
                                 Stile_i32.frag_at(0, 1),
                                 Qfi8[id].frag_at(0, 0), metal::false_type{},
                                 Kt.frag_at(0, 0), Kt.frag_at(1, 0),
                                 metal::true_type{});
            }
        }

        NAXTile<float, 1, 2> Stile;
        const bool isUnmasked = (n0 + QW_BK <= pos + rowBase);
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < 2; ik++) {
            const uint kColBase = uint(ik * kU) + uint(sn);
            const float4 ks = *(const threadgroup float4*)(kScale + kColBase);
            thread auto& ifg = Stile_i32.frag_at(0, ik);
            thread auto& fg  = Stile.frag_at(0, ik);
            if (isUnmasked) {
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; i++) {
                    const float qs = qScale2[i];
                    fg[i * 4 + 0] = float(ifg[i * 4 + 0]) * (qs * ks.x);
                    fg[i * 4 + 1] = float(ifg[i * 4 + 1]) * (qs * ks.y);
                    fg[i * 4 + 2] = float(ifg[i * 4 + 2]) * (qs * ks.z);
                    fg[i * 4 + 3] = float(ifg[i * 4 + 3]) * (qs * ks.w);
                }
            } else {
                const uint col0 = n0 + kColBase;
                const float ksv[4] = {ks.x, ks.y, ks.z, ks.w};
                STEEL_PRAGMA_UNROLL
                for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                    const uint i = (uint)(loc >> 2), j = (uint)(loc & 3);
                    const uint abs_cc = col0 + j;
                    const uint abs_rr = pos + rowBase + (uint)sm + i * 8u;
                    fg[loc] = (abs_cc <= abs_rr && abs_cc < (pos + T))
                        ? (float(ifg[loc]) * (qScale2[i] * ksv[j]))
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
                for (short d = 0; d < 16; d++) {
                    if (d < nq) {
                        thread auto& ofg = Otile.frag_at(0, d);
                        STEEL_PRAGMA_UNROLL
                        for (short j = 0; j < 4; j++)
                            ofg[i * 4 + j] *= corr;
                    }
                }
            }
            mrow[i] = mnew[i];
        }

        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < short(QW_BK / kU); ik++) {
            STEEL_PRAGMA_UNROLL
            for (short pair = 0; pair < 8; pair++) {
                if (pair < npair) {
                    NAXTile<half, 1, 2> Vt;
                    Vt.template load<half, kStageStrideV, 1>(vblk + (uint(ik * kU)) * kStageStrideV + pair * (2 * kU));
                    BaseNAXFrag::mma(Otile.frag_at(0, 2 * pair),
                                     Otile.frag_at(0, 2 * pair + 1),
                                     Pt.frag_at(0, ik), metal::false_type{},
                                     Vt.frag_at(0, 0), Vt.frag_at(0, 1),
                                     metal::false_type{});
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rcp[2];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) rcp[i] = (lrow[i] > 0.0f) ? (1.0f / lrow[i]) : 0.0f;

    const uint numPasses = (QW_GQA_RATIO + 1u) / 2u;
    for (uint pass = 0u; pass < numPasses; pass++) {
        const uint h0 = headBase + pass * 2u;
        const uint h1 = h0 + 1u;
        const bool hasH1 = (pass * 2u + 1u) < QW_GQA_RATIO;
        if ((sgi_h >> 1u) == pass) {
            threadgroup half* obuf = ((sgi_h & 1u) == 0u) ? obuf1 : vblk;
            STEEL_PRAGMA_UNROLL
            for (short d = 0; d < 16; d++) {
                if (d < nq) {
                    const uint col0 = (uint)(d * kU) + (uint)sn;
                    STEEL_PRAGMA_UNROLL
                    for (short i = 0; i < 2; i++) {
                        const uint rowInTile = sgi_local * 16u + (uint)sm + (uint)i * 8u;
                        const uint rchunk = tile0 + rowInTile;
                        if (rchunk < T) {
                            const device half* grow = qgate
                                + (uint64_t)rchunk * QW_QPROJ_OUT
                                + (uint64_t)head * QW_Q_STRIDE * QW_HEAD_DIM + QW_HEAD_DIM;
                            thread auto& fg = Otile.frag_at(0, d);
                            STEEL_PRAGMA_UNROLL
                            for (short j = 0; j < 4; j++) {
                                float sig = 1.0f;
                                if (QW_HAS_Q_GATE) {
                                    const half gval = grow[col0 + j];
                                    sig = 1.0f / (1.0f + metal::exp(-float(gval)));
                                }
                                obuf[rowInTile * QW_HEAD_DIM + col0 + j] = half(fg[i * 4 + j] * rcp[i] * sig);
                            }
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint fwhtThreads = min(kThreads, 256u);
        if (tid < fwhtThreads) {
            for (uint e = tid; e < 32u * QW_HEAD_DIM; e += fwhtThreads) {
                const uint rowInTile = (QW_HEAD_DIM == 256u) ? (e >> 8) : (e / QW_HEAD_DIM);
                const uint r_out = tile0 + rowInTile;
                const uint c = (QW_HEAD_DIM == 256u) ? (e & 255u) : (e % QW_HEAD_DIM);
                const half rot0 = simd_fwht32(obuf1[e], tid & 31u);
                const half rot1 = hasH1 ? simd_fwht32(vblk[e], tid & 31u) : half(0.0h);
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
METAL_FUNC void qwg_attn_prefill_paged_nax_pair_body(
    const device half*  qgate,
    const device half*  kpool,
    const device half*  vpool,
    const device int*   ktab,
    const device int*   vtab,
    device half*        outp,
    constant uint&      pos,
    constant uint&      T,
    uint tid,
    uint sgi,
    uint3 tg,
    threadgroup half*   kvblk,
    threadgroup half*   vblk) {
    if ((QW_HEAD_DIM & (QW_HEAD_DIM - 1u)) == 0u) {
        threadgroup int8_t* kvblk_i8 = (threadgroup int8_t*)kvblk;
        threadgroup float* kScale = (threadgroup float*)(kvblk_i8 + QW_BK * (QW_NAX_HDIM + 16u));
        qwg_attn_prefill_paged_nax_pair_i8_body<MODE>(qgate, kpool, vpool, ktab, vtab, outp, pos, T, tid, sgi, tg, kvblk_i8, kScale, vblk, kvblk);
    } else {
        qwg_attn_prefill_paged_nax_pair_f16_body<MODE>(qgate, kpool, vpool, ktab, vtab, outp, pos, T, tid, sgi, tg, kvblk, vblk);
    }
}

#define QW_PAGED_PREFILL_ENTRY(SFX, MODE) \
kernel void qwg_attn_prefill_paged_nax_pair##SFX( \
    const device half*  qgate   [[buffer(0)]], \
    const device half*  kpool   [[buffer(1)]], \
    const device half*  vpool   [[buffer(2)]], \
    const device int*   ktab    [[buffer(3)]], \
    const device int*   vtab    [[buffer(4)]], \
    device half*        outp    [[buffer(5)]], \
    constant uint&      pos     [[buffer(6)]], \
    constant uint&      T       [[buffer(7)]], \
    uint tid   [[thread_index_in_threadgroup]], \
    uint sgi   [[simdgroup_index_in_threadgroup]], \
    uint3 tg   [[threadgroup_position_in_grid]]) { \
    threadgroup half kvblk[QW_BK * QW_NAX_HDIM]; \
    threadgroup half vblk[QW_BK * QW_NAX_HDIM]; \
    qwg_attn_prefill_paged_nax_pair_body<MODE>(qgate, kpool, vpool, ktab, vtab, outp, pos, T, tid, sgi, tg, kvblk, vblk); \
}
QW_PAGED_PREFILL_ENTRY(, 0)
QW_PAGED_PREFILL_ENTRY(_q8, 1)
QW_PAGED_PREFILL_ENTRY(_q4, 2)
#undef QW_PAGED_PREFILL_ENTRY
