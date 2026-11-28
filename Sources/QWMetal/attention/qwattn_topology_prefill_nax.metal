// qwattn_topology_prefill_nax.metal — Metal 4 NAX FlashAttention for generalized decision topologies.
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

using namespace metal;
using namespace mlx::steel;

struct QWTopoAttnNAXParams {
    uint seqLen;
    uint qHeads;
    uint kvHeads;
    uint headDim;
    uint topologyMode;
    uint stateEndPos;
    uint mediaPrefixLen;
    uint slidingWindow;
    float scale;
};

METAL_FUNC bool qwg_topo_allow_elem(
    uint qPos, uint kPos, int qGroup, int kGroup, constant QWTopoAttnNAXParams& p) {
    if (qGroup < 0 || kGroup < 0) return false;
    if (p.slidingWindow > 0u) {
        return (kPos <= qPos) && ((qPos - kPos) < p.slidingWindow);
    }
    if (p.topologyMode == 0u) {
        return (kPos <= qPos);
    }
    if (p.topologyMode == 1u) {
        return (qPos < p.stateEndPos && kPos < p.stateEndPos) || (kPos <= qPos);
    }
    if (p.topologyMode == 3u) {
        return (qGroup == 0) || (kGroup == 0) || (kGroup == qGroup);
    }
    if (p.topologyMode == 4u) {
        return (p.mediaPrefixLen == 0u) || ((qPos < p.mediaPrefixLen) == (kPos < p.mediaPrefixLen));
    }
    return true;
}

template <uint HDIM, uint HD_LOG2>
METAL_FUNC void qwg_attn_topology_prefill_nax_body(
    device const half* q, device const half* k, device const half* v,
    device const int* groupIds, device half* outp,
    constant QWTopoAttnNAXParams& p,
    threadgroup half* kvblk, threadgroup int* qGrpSh, threadgroup int* kGrpSh,
    threadgroup bool& tgStop, threadgroup bool& tgSkipBlock, threadgroup bool& tgAllAllow,
    uint tid, uint sgi, uint2 tg) {
    const uint h = tg.x;
    const uint tile0 = tg.y * 32u;
    if (h >= p.qHeads || tile0 >= p.seqLen) return;

    const uint lane = tid & 31u;
    const uint gqaRatio = max(1u, p.qHeads / max(1u, p.kvHeads));
    const uint kvHead = min(p.kvHeads - 1u, h / gqaRatio);
    const size_t qRowStride = (size_t)p.qHeads * HDIM;
    const size_t kvRowStride = (size_t)p.kvHeads * HDIM;
    threadgroup half* const obuf = kvblk;

    if (tid < 32u) {
        const uint qp = tile0 + lane;
        const int qg = (qp < p.seqLen) ? groupIds[qp] : -1;
        qGrpSh[lane] = qg;
        const bool anyVal = simd_any(qg >= 0);
        if (lane == 0u) tgStop = !anyVal;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tgStop) {
        for (uint e = tid; e < 32u * HDIM; e += 128u) {
            const uint qr = e >> HD_LOG2, qc = e & (HDIM - 1u);
            if (tile0 + qr < p.seqLen) {
                outp[((size_t)(tile0 + qr) * p.qHeads + h) * HDIM + qc] = 0.0h;
            }
        }
        return;
    }

    const uint qMaxIdx = min(p.seqLen - 1u, tile0 + 31u);
    uint kLoopStart = 0u;
    uint kLoopEnd = p.seqLen;
    if (p.slidingWindow > 0u) {
        kLoopEnd = min(p.seqLen, qMaxIdx + 1u);
        kLoopStart = ((tile0 + 1u) > p.slidingWindow) ? (((tile0 + 1u - p.slidingWindow) >> 5u) << 5u) : 0u;
    } else if (p.topologyMode == 0u) {
        kLoopEnd = min(p.seqLen, qMaxIdx + 1u);
    } else if (p.topologyMode == 1u) {
        kLoopEnd = (tile0 >= p.stateEndPos) ? min(p.seqLen, qMaxIdx + 1u)
            : max(min(p.seqLen, p.stateEndPos), min(p.seqLen, qMaxIdx + 1u));
    } else if (p.topologyMode == 4u && p.mediaPrefixLen > 0u) {
        if (tile0 >= p.mediaPrefixLen) kLoopStart = (p.mediaPrefixLen >> 5u) << 5u;
        else if (qMaxIdx < p.mediaPrefixLen) kLoopEnd = min(p.seqLen, p.mediaPrefixLen);
    }

    const uint rowBase = tile0 + sgi * 16u;
    const bool sgCompute = (sgi < 2u) && (rowBase < p.seqLen);
    constexpr short kU = 16;
    constexpr short NQ = short(HDIM) / kU;
    const short2 sc = BaseNAXFrag::get_coord();
    const short sm = sc.y, sn = sc.x;
    const int r = int(rowBase) + int(sm);
    const float scale2 = p.scale * 1.4426950408889634f;

    NAXTile<half, 1, 1> Qf[NQ];
    NAXTile<float, 1, NQ> Otile;
    float mrow[2] = {-1e30f, -1e30f};
    float lrow[2] = {0.0f, 0.0f};
    if (sgCompute) {
        Otile.clear();
        const device half* qdev = q + (size_t)rowBase * qRowStride + (size_t)h * HDIM;
        const bool fullQ = (rowBase + 16u <= p.seqLen);
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < NQ; id++) {
            if (fullQ) Qf[id].load(qdev + id * kU, int(qRowStride));
            else Qf[id].load_safe(qdev + id * kU, int(qRowStride), short2(int(HDIM) - int(id) * kU, int(p.seqLen - rowBase)));
        }
    }

    const device half* kdev = k + (size_t)kvHead * HDIM;
    const device half* vdev = v + (size_t)kvHead * HDIM;

    for (uint n0 = kLoopStart; n0 < kLoopEnd; n0 += 32u) {
        if (tid < 32u) {
            const uint kp = n0 + lane;
            const int kg = (kp < p.seqLen) ? groupIds[kp] : -1;
            kGrpSh[lane] = kg;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const uint qr = tile0 + lane;
            const int qg = qGrpSh[lane];
            bool rowAny = false;
            bool rowAll = (qr < p.seqLen) && (qg >= 0);
            if (qr < p.seqLen && qg >= 0) {
                for (uint c = 0u; c < 32u; c++) {
                    const uint kc = n0 + c;
                    const bool ok = (kc < p.seqLen) && qwg_topo_allow_elem(qr, kc, qg, kGrpSh[c], p);
                    rowAny = rowAny || ok;
                    rowAll = rowAll && ok;
                }
            }
            const bool blockAny = simd_any(rowAny);
            const bool blockAll = simd_all((qr >= p.seqLen) || rowAll);
            if (lane == 0u) {
                tgSkipBlock = !blockAny;
                tgAllAllow = blockAll;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tgSkipBlock) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        const bool fullKBlk = (n0 + 32u <= p.seqLen);
        for (uint e = tid; e < 32u * HDIM; e += 128u) {
            const uint kr = e >> HD_LOG2, kc = e & (HDIM - 1u);
            const uint kPos = n0 + kr;
            kvblk[e] = (fullKBlk || kPos < p.seqLen) ? kdev[(size_t)kPos * kvRowStride + kc] : 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        NAXTile<half, 1, 2> Pt;
        if (sgCompute) {
            NAXTile<float, 1, 2> Stile;
            Stile.clear();
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < NQ; id++) {
                NAXTile<half, 2, 1> Kt;
                Kt.template load<half, int(HDIM), 1>(kvblk + id * kU);
                BaseNAXFrag::mma(Stile.frag_at(0, 0), Stile.frag_at(0, 1),
                                 Qf[id].frag_at(0, 0), metal::false_type{},
                                 Kt.frag_at(0, 0), Kt.frag_at(1, 0), metal::true_type{});
            }

            if (tgAllAllow && fullKBlk) {
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
                    const int col0 = int(n0) + int(ik * kU) + int(sn);
                    STEEL_PRAGMA_UNROLL
                    for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                        const short i = loc >> 2, j = loc & 3;
                        const int rr = r + int(i) * 8;
                        const int cc = col0 + int(j);
                        bool ok = (rr >= 0 && uint(rr) < p.seqLen && cc >= 0 && uint(cc) < p.seqLen);
                        if (ok) {
                            const int qg = qGrpSh[uint(rr) - tile0];
                            const int kg = kGrpSh[uint(cc) - n0];
                            ok = qwg_topo_allow_elem(uint(rr), uint(cc), qg, kg, p);
                        }
                        fg[loc] = ok ? (fg[loc] * scale2) : -1e30f;
                    }
                }
            }

            float mnew[2] = {mrow[0], mrow[1]};
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < 2; ik++) {
                thread auto& fg = Stile.frag_at(0, ik);
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < BaseNAXFrag::kElemRows; i++) {
                    float mv = max(max(fg[i * 4 + 0], fg[i * 4 + 1]), max(fg[i * 4 + 2], fg[i * 4 + 3]));
                    mnew[i] = max(mnew[i], mv);
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
                thread auto& sf = Stile.frag_at(0, ik);
                thread auto& pf = Pt.frag_at(0, ik);
                STEEL_PRAGMA_UNROLL
                for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                    const short i = loc >> 2;
                    const float x = (mnew[i] > -1e29f) ? fast::exp2(sf[loc] - mnew[i]) : 0.0f;
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
                const bool grew = (mnew[i] > mrow[i]) && (mrow[i] > -1e29f);
                const float corr = grew ? fast::exp2(mrow[i] - mnew[i]) : 1.0f;
                lrow[i] = lrow[i] * corr + psum[i];
                if (grew) {
                    STEEL_PRAGMA_UNROLL
                    for (short d = 0; d < NQ; d++) {
                        thread auto& ofg = Otile.frag_at(0, d);
                        STEEL_PRAGMA_UNROLL
                        for (short j = 0; j < 4; j++) ofg[i * 4 + j] *= corr;
                    }
                }
                mrow[i] = mnew[i];
            }
        }

        if (fullKBlk) {
            if (sgCompute) {
                constexpr short NPAIRS = NQ / 2;
                STEEL_PRAGMA_UNROLL
                for (short ik = 0; ik < 2; ik++) {
                    STEEL_PRAGMA_UNROLL
                    for (short pair = 0; pair < NPAIRS; pair++) {
                        NAXTile<half, 1, 2> Vt;
                        Vt.load(vdev + (size_t)(n0 + uint(ik * kU)) * kvRowStride + pair * (2 * kU), int(kvRowStride));
                        BaseNAXFrag::mma(Otile.frag_at(0, 2 * pair), Otile.frag_at(0, 2 * pair + 1),
                                         Pt.frag_at(0, ik), metal::false_type{},
                                         Vt.frag_at(0, 0), Vt.frag_at(0, 1), metal::false_type{});
                    }
                }
            }
        } else {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = tid; e < 32u * HDIM; e += 128u) {
                const uint vr = e >> HD_LOG2, vc = e & (HDIM - 1u);
                const uint vPos = n0 + vr;
                kvblk[e] = (vPos < p.seqLen) ? vdev[(size_t)vPos * kvRowStride + vc] : 0.0h;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgCompute) {
                constexpr short NPAIRS = NQ / 2;
                STEEL_PRAGMA_UNROLL
                for (short ik = 0; ik < 2; ik++) {
                    STEEL_PRAGMA_UNROLL
                    for (short pair = 0; pair < NPAIRS; pair++) {
                        NAXTile<half, 1, 2> Vt;
                        Vt.template load<half, int(HDIM), 1>(kvblk + uint(ik * kU) * HDIM + pair * (2 * kU));
                        BaseNAXFrag::mma(Otile.frag_at(0, 2 * pair), Otile.frag_at(0, 2 * pair + 1),
                                         Pt.frag_at(0, ik), metal::false_type{},
                                         Vt.frag_at(0, 0), Vt.frag_at(0, 1), metal::false_type{});
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgCompute) {
        float rcp[2];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) rcp[i] = (lrow[i] > 0.0f) ? (1.0f / lrow[i]) : 0.0f;

        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < NQ; d++) {
            const uint col0 = uint(d * kU) + uint(sn);
            thread auto& fg = Otile.frag_at(0, d);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; j++) {
                    obuf[uint(sgi * 16u + uint(sm) + uint(i) * 8u) * HDIM + col0 + uint(j)] =
                        half(fg[i * 4 + j] * rcp[i]);
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < 16u * HDIM; i += 32u) {
            const uint row = i >> HD_LOG2, r2 = rowBase + row;
            if (r2 < p.seqLen) {
                const bool validQ = qGrpSh[sgi * 16u + row] >= 0;
                outp[((size_t)r2 * p.qHeads + h) * HDIM + (i & (HDIM - 1u))] =
                    validQ ? obuf[(sgi * 16u + row) * HDIM + (i & (HDIM - 1u))] : 0.0h;
            }
        }
    }
}

kernel void qwg_attn_topology_prefill_nax_d64(
    device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]], device const int* groupIds [[buffer(3)]],
    device half* outp [[buffer(4)]], constant QWTopoAttnNAXParams& p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]], uint sgi [[simdgroup_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
    threadgroup half kvblk[32u * 64u];
    threadgroup int qGrpSh[32];
    threadgroup int kGrpSh[32];
    threadgroup bool tgStop, tgSkipBlock, tgAllAllow;
    qwg_attn_topology_prefill_nax_body<64u, 6u>(
        q, k, v, groupIds, outp, p, kvblk, qGrpSh, kGrpSh, tgStop, tgSkipBlock, tgAllAllow, tid, sgi, tg);
}

kernel void qwg_attn_topology_prefill_nax_d128(
    device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]], device const int* groupIds [[buffer(3)]],
    device half* outp [[buffer(4)]], constant QWTopoAttnNAXParams& p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]], uint sgi [[simdgroup_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
    threadgroup half kvblk[32u * 128u];
    threadgroup int qGrpSh[32];
    threadgroup int kGrpSh[32];
    threadgroup bool tgStop, tgSkipBlock, tgAllAllow;
    qwg_attn_topology_prefill_nax_body<128u, 7u>(
        q, k, v, groupIds, outp, p, kvblk, qGrpSh, kGrpSh, tgStop, tgSkipBlock, tgAllAllow, tid, sgi, tg);
}
#endif // !defined(QW_DISABLE_NAX)
