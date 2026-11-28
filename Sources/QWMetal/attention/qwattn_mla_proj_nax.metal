// qwattn_mla_proj_nax.metal — Apple10+ NAX absorb_q & proj_uv (BM=64, BN=64).
// Isolated _sub2 / non-_sub2 PSOs keep constant LUTs <= 20 KiB per kernel.
#if !defined(QW_DISABLE_NAX)
#include <metal_stdlib>
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "../include/qw_mla_dequant.metalh"
using namespace metal;
using namespace mlx::steel;

#if !defined(__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__) || (__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ < 260200)
#error "Metal 4 NAX kernels require -mmacosx-version-min=26.2 (SDK MPPTensorOpsAvailability)"
#endif

#define MLA_PROJ_NAX_OS 32u

template <int MODE>
METAL_FUNC size_t qw_mla_nax_row_bytes(uint kLen, constant MLAAbsorbParams& p,
                                       thread uint& groupBytes, thread uint& sliceBytes) {
    if constexpr (MODE == 3) {
        groupBytes = 0u;
        sliceBytes = 0u;
        return (size_t)kLen * sizeof(half);
    } else if constexpr (MODE == 1) {
        groupBytes = 0u;
        sliceBytes = 0u;
        return (size_t)(kLen >> 5u) * sizeof(Block_HQ4_G32);
    } else if constexpr (MODE == 2) {
        groupBytes = qwg_shvq_group_bytes(int(p.bits), int(p.group));
        sliceBytes = qwg_shvq_slice_bytes(int(p.bits));
        return (size_t)(kLen / p.group) * groupBytes;
    } else {
        groupBytes = 4u + ((p.group * p.bits) >> 3u);
        sliceBytes = 0u;
        return (size_t)(kLen / p.group) * groupBytes;
    }
}

template <int MODE, bool SUB2>
METAL_FUNC void qw_mla_dequant_col32_pair(
    device const char* rPtr, uint colBase, uint bits, uint gShift, uint gMask,
    uint groupBytes, uint sliceBytes, threadgroup const ShvqMlaTgLut& tgLut,
    short2 sc, ushort wSub, ushort wShift, thread half4& vLo, thread half4& vHi
) {
    if constexpr (MODE == 3) {
        device const half* hPtr = (device const half*)rPtr + colBase + uint(sc.x);
        vLo = *(device const half4*)hPtr;
        vHi = *(device const half4*)(hPtr + 16u);
    } else if constexpr (MODE == 1) {
        const device Block_HQ4_G32& blk = ((const device Block_HQ4_G32*)rPtr)[colBase >> 5u];
        const half2 dm = *(const device half2*)(&blk);
        const half2 s2 = half2(dm.x), b2 = half2(dm.y);
        const ushort2 raw0 = as_type<ushort2>(blk.qs[wSub]) >> wShift;
        const ushort2 raw1 = as_type<ushort2>(blk.qs[2u + wSub]) >> wShift;
        const half2 w01 = fma(as_type<half2>((raw0 & ushort2(0x000Fu)) | ushort2(0x6400u)), s2, b2);
        const half2 w23 = fma(as_type<half2>(((raw0 >> 4u) & ushort2(0x000Fu)) | ushort2(0x6400u)), s2, b2);
        const half2 w45 = fma(as_type<half2>((raw1 & ushort2(0x000Fu)) | ushort2(0x6400u)), s2, b2);
        const half2 w67 = fma(as_type<half2>(((raw1 >> 4u) & ushort2(0x000Fu)) | ushort2(0x6400u)), s2, b2);
        vLo = half4(w01.x, w01.y, w23.x, w23.y);
        vHi = half4(w45.x, w45.y, w67.x, w67.y);
    } else {
        const uint gIdx = colBase >> gShift;
        const uint inGroup = colBase & gMask;
        device const char* gp = rPtr + (size_t)gIdx * groupBytes;
        if constexpr (MODE == 2) {
            const half scS = *(device const half*)gp;
            device const uchar* pay = (device const uchar*)(gp + 2u + ((inGroup >> 5u) * sliceBytes));
            vLo = qw_mla_shvq_quad4<SUB2>(pay, uint(sc.x) >> 2u, bits, tgLut) * scS;
            vHi = qw_mla_shvq_quad4<SUB2>(pay, (uint(sc.x) + 16u) >> 2u, bits, tgLut) * scS;
        } else {
            const half2 sb = *(device const half2*)gp;
            const half2 s2 = half2(sb.x), b2 = half2(sb.y), kBias = half2(-1024.0h);
            device const uchar* pay = (device const uchar*)(gp + 4u);
            if constexpr (MODE == 8) {
                const uint bo = inGroup + uint(sc.x);
                const ushort2 rLo = as_type<ushort2>(*(device const uint*)(pay + bo));
                const ushort2 rHi = as_type<ushort2>(*(device const uint*)(pay + bo + 16u));
                const half2 l0 = fma(as_type<half2>((rLo & ushort2(0x00FFu)) | ushort2(0x6400u)) + kBias, s2, b2);
                const half2 l1 = fma(as_type<half2>((rLo >> 8u) | ushort2(0x6400u)) + kBias, s2, b2);
                const half2 h0 = fma(as_type<half2>((rHi & ushort2(0x00FFu)) | ushort2(0x6400u)) + kBias, s2, b2);
                const half2 h1 = fma(as_type<half2>((rHi >> 8u) | ushort2(0x6400u)) + kBias, s2, b2);
                vLo = half4(l0.x, l1.x, l0.y, l1.y);
                vHi = half4(h0.x, h1.x, h0.y, h1.y);
            } else if constexpr (MODE == 4) {
                const uint bo = (inGroup + uint(sc.x)) >> 1u;
                const ushort2 raw = ushort2(*(device const ushort*)(pay + bo),
                                            *(device const ushort*)(pay + bo + 8u));
                const half2 w0 = fma(as_type<half2>((raw & ushort2(0x000Fu)) | ushort2(0x6400u)) + kBias, s2, b2);
                const half2 w1 = fma(as_type<half2>(((raw >> 4u) & ushort2(0x000Fu)) | ushort2(0x6400u)) + kBias, s2, b2);
                const half2 w2 = fma(as_type<half2>(((raw >> 8u) & ushort2(0x000Fu)) | ushort2(0x6400u)) + kBias, s2, b2);
                const half2 w3 = fma(as_type<half2>((raw >> 12u) | ushort2(0x6400u)) + kBias, s2, b2);
                vLo = half4(w0.x, w1.x, w2.x, w3.x);
                vHi = half4(w0.y, w1.y, w2.y, w3.y);
            } else {
                const float scAQ = float(sb.x), biAQ = float(sb.y);
                const int bitsI = int(bits), b0 = int(inGroup + uint(sc.x));
                vLo = half4(half(fma(float(qwg_aq_read_bits(pay, bitsI, b0 + 0)), scAQ, biAQ)),
                            half(fma(float(qwg_aq_read_bits(pay, bitsI, b0 + 1)), scAQ, biAQ)),
                            half(fma(float(qwg_aq_read_bits(pay, bitsI, b0 + 2)), scAQ, biAQ)),
                            half(fma(float(qwg_aq_read_bits(pay, bitsI, b0 + 3)), scAQ, biAQ)));
                vHi = half4(half(fma(float(qwg_aq_read_bits(pay, bitsI, b0 + 16)), scAQ, biAQ)),
                            half(fma(float(qwg_aq_read_bits(pay, bitsI, b0 + 17)), scAQ, biAQ)),
                            half(fma(float(qwg_aq_read_bits(pay, bitsI, b0 + 18)), scAQ, biAQ)),
                            half(fma(float(qwg_aq_read_bits(pay, bitsI, b0 + 19)), scAQ, biAQ)));
            }
        }
    }
}

// NAX absorb_q: qAbs[T, qHeads, kvLoraRank] = q_nope[T, h] @ W_UK[nopeDim, kvLoraRank].
template <int MODE, bool SUB2 = false>
METAL_FUNC void qw_mla_absorb_q_nax_run(
    device const half* mlaQFull, device const char* kvBBlocks, device half* qAbsOut,
    constant MLAAbsorbParams& p, threadgroup const ShvqMlaTgLut& tgLut,
    uint h, uint warpT0, uint warpC0, int sgpSm
) {
    NAXTile<float, 2, 2> Dtile;
    Dtile.clear();

    const short2 sc = BaseNAXFrag::get_coord();
    const uint gShift = ctz(p.group), gMask = p.group - 1u;
    uint groupBytes = 0u, sliceBytes = 0u;
    const size_t rowBytes = qw_mla_nax_row_bytes<MODE>(p.kvLoraRank, p, groupBytes, sliceBytes);

    const uint headRowBase = h * (p.nopeDim + p.vHeadDim);
    const int ldA = int(p.qFullStride);
    device const half* aBase = mlaQFull + (size_t)warpT0 * p.qFullStride + h * p.nopeDim;
    const ushort wSub = ushort(sc.x >> 3);
    const ushort wShift = ushort((sc.x & 4) << 1);
    const bool alignedM = (sgpSm == 32);

    STEEL_PRAGMA_NO_UNROLL
    for (uint k0 = 0u; k0 < p.nopeDim; k0 += 32u) {
        NAXTile<half, 2, 2> Atile;
        NAXTile<half, 2, 2> Btile;
        const bool alignedK = (k0 + 32u <= p.nopeDim);
        if (alignedM && alignedK) {
            Atile.load(aBase + k0, ldA);
        } else {
            Atile.load_safe(aBase + k0, ldA, short2(short(min(32u, p.nopeDim - k0)), short(sgpSm)));
        }

        STEEL_PRAGMA_UNROLL
        for (short r = 0; r < 2; ++r) {
            thread auto& f0 = Btile.frag_at(r, 0);
            thread auto& f1 = Btile.frag_at(r, 1);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; ++i) {
                const uint kRow = k0 + uint(r * 16 + int(sc.y) + i * 8);
                half4 vLo = half4(0.0h), vHi = half4(0.0h);
                if (kRow < p.nopeDim) {
                    device const char* rPtr = kvBBlocks + (size_t)(headRowBase + kRow) * rowBytes;
                    qw_mla_dequant_col32_pair<MODE, SUB2>(
                        rPtr, warpC0, p.bits, gShift, gMask, groupBytes, sliceBytes,
                        tgLut, sc, wSub, wShift, vLo, vHi);
                }
                f0[i * 4 + 0] = vLo.x; f0[i * 4 + 1] = vLo.y; f0[i * 4 + 2] = vLo.z; f0[i * 4 + 3] = vLo.w;
                f1[i * 4 + 0] = vHi.x; f1[i * 4 + 1] = vHi.y; f1[i * 4 + 2] = vHi.z; f1[i * 4 + 3] = vHi.w;
            }
        }
        tile_matmad_nax(Dtile, Atile, metal::false_type{}, Btile, metal::false_type{});
    }

    device half* dst = qAbsOut + ((size_t)warpT0 * p.qHeads + h) * p.kvLoraRank + warpC0;
    const int ldOut = int(p.qHeads * p.kvLoraRank);
    if (alignedM && warpC0 + 32u <= p.kvLoraRank) {
        Dtile.store(dst, ldOut);
    } else {
        Dtile.store_safe(dst, ldOut, short2(short(min(32u, p.kvLoraRank - warpC0)), short(sgpSm)));
    }
}

template <bool SUB2>
METAL_FUNC void qw_mla_absorb_q_nax_body(
    device const half* mlaQFull, device const char* kvBBlocks, device half* qAbsOut,
    constant MLAAbsorbParams& p, threadgroup ShvqMlaTgLut& tgLut,
    uint3 tg, uint tid, uint sg
) {
    const uint cBase = tg.x * 64u, tBase = tg.y * 64u, h = tg.z;
    if (h >= p.qHeads || tBase >= p.T || cBase >= p.kvLoraRank) return;

    const uint inc = p.isIncumbent & 0xFFu;
    if (inc == 2u && qw_mla_load_shvq_lut<SUB2>(tgLut, p.bits, tid, 128u)) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint wm = sg >> 1u, wn = sg & 1u;
    const uint warpT0 = tBase + wm * 32u, warpC0 = cBase + wn * 32u;
    const int sgpSm = min(32, int(p.T) - int(warpT0));
    if (sgpSm <= 0 || warpC0 >= p.kvLoraRank) return;

    if (inc == 2u) {
        qw_mla_absorb_q_nax_run<2, SUB2>(mlaQFull, kvBBlocks, qAbsOut, p, tgLut, h, warpT0, warpC0, sgpSm);
    } else if (inc == 3u || p.bits == 16u) {
        qw_mla_absorb_q_nax_run<3>(mlaQFull, kvBBlocks, qAbsOut, p, tgLut, h, warpT0, warpC0, sgpSm);
    } else if (inc == 1u) {
        qw_mla_absorb_q_nax_run<1>(mlaQFull, kvBBlocks, qAbsOut, p, tgLut, h, warpT0, warpC0, sgpSm);
    } else if (p.bits == 8u) {
        qw_mla_absorb_q_nax_run<8>(mlaQFull, kvBBlocks, qAbsOut, p, tgLut, h, warpT0, warpC0, sgpSm);
    } else if (p.bits == 4u) {
        qw_mla_absorb_q_nax_run<4>(mlaQFull, kvBBlocks, qAbsOut, p, tgLut, h, warpT0, warpC0, sgpSm);
    } else {
        qw_mla_absorb_q_nax_run<0>(mlaQFull, kvBBlocks, qAbsOut, p, tgLut, h, warpT0, warpC0, sgpSm);
    }
}

// Grid: (ceil(kvLoraRank / 64), ceil(T / 64), qHeads), Threadgroup: (128, 1, 1).
kernel void qwg_mla_absorb_q_nax(
    device const half* mlaQFull [[buffer(0)]], device const char* kvBBlocks [[buffer(1)]],
    device half* qAbsOut [[buffer(2)]], constant MLAAbsorbParams& p [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup ShvqMlaTgLut tgLut;
    qw_mla_absorb_q_nax_body<false>(mlaQFull, kvBBlocks, qAbsOut, p, tgLut, tg, tid, sg);
}

kernel void qwg_mla_absorb_q_nax_sub2(
    device const half* mlaQFull [[buffer(0)]], device const char* kvBBlocks [[buffer(1)]],
    device half* qAbsOut [[buffer(2)]], constant MLAAbsorbParams& p [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup ShvqMlaTgLut tgLut;
    qw_mla_absorb_q_nax_body<true>(mlaQFull, kvBBlocks, qAbsOut, p, tgLut, tg, tid, sg);
}

// NAX proj_uv: attOut[T, qHeads, vHeadDim] = uLat[T, h] @ W_UV[vHeadDim, kvLoraRank]^T.
union MLAProjNaxTg {
    ShvqMlaTgLut tgLut;
    float os[4][32u * MLA_PROJ_NAX_OS];
};

template <int MODE, bool SUB2 = false>
METAL_FUNC void qw_mla_proj_uv_nax_run(
    device const half* uLat, device const char* kvBBlocks, device half* attOut,
    constant MLAAbsorbParams& p, threadgroup float* osBase, threadgroup const ShvqMlaTgLut& tgLut,
    uint h, uint warpT0, uint warpC0, int sgpSm, uint sg, uint lane
) {
    NAXTile<float, 2, 2> Dtile;
    Dtile.clear();

    const short2 sc = BaseNAXFrag::get_coord();
    const uint gShift = ctz(p.group), gMask = p.group - 1u;
    uint groupBytes = 0u, sliceBytes = 0u;
    const size_t rowBytes = qw_mla_nax_row_bytes<MODE>(p.kvLoraRank, p, groupBytes, sliceBytes);

    const uint headVBase = h * (p.nopeDim + p.vHeadDim) + p.nopeDim;
    device const char* wRowPtr[4];
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < 2; ++r) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; ++i) {
            const uint nCol = warpC0 + uint(r * 16 + int(sc.y) + i * 8);
            wRowPtr[r * 2 + i] = (nCol < p.vHeadDim)
                ? (kvBBlocks + (size_t)(headVBase + nCol) * rowBytes)
                : nullptr;
        }
    }

    const int ldA = int(p.qHeads * p.kvLoraRank);
    device const half* aBase = uLat + ((size_t)warpT0 * p.qHeads + h) * p.kvLoraRank;
    const ushort wSub = ushort(sc.x >> 3);
    const ushort wShift = ushort((sc.x & 4) << 1);
    const bool alignedM = (sgpSm == 32);

    STEEL_PRAGMA_NO_UNROLL
    for (uint k0 = 0u; k0 < p.kvLoraRank; k0 += 32u) {
        NAXTile<half, 2, 2> Atile;
        NAXTile<half, 2, 2> Btile;
        const bool alignedK = (k0 + 32u <= p.kvLoraRank);
        if (alignedM && alignedK) {
            Atile.load(aBase + k0, ldA);
        } else {
            Atile.load_safe(aBase + k0, ldA, short2(short(min(32u, p.kvLoraRank - k0)), short(sgpSm)));
        }

        STEEL_PRAGMA_UNROLL
        for (short r = 0; r < 2; ++r) {
            thread auto& f0 = Btile.frag_at(r, 0);
            thread auto& f1 = Btile.frag_at(r, 1);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; ++i) {
                device const char* rPtr = wRowPtr[r * 2 + i];
                half4 vLo = half4(0.0h), vHi = half4(0.0h);
                if (rPtr != nullptr) {
                    qw_mla_dequant_col32_pair<MODE, SUB2>(
                        rPtr, k0, p.bits, gShift, gMask, groupBytes, sliceBytes,
                        tgLut, sc, wSub, wShift, vLo, vHi);
                }
                f0[i * 4 + 0] = vLo.x; f0[i * 4 + 1] = vLo.y; f0[i * 4 + 2] = vLo.z; f0[i * 4 + 3] = vLo.w;
                f1[i * 4 + 0] = vHi.x; f1[i * 4 + 1] = vHi.y; f1[i * 4 + 2] = vHi.z; f1[i * 4 + 3] = vHi.w;
            }
        }
        tile_matmad_nax(Dtile, Atile, metal::false_type{}, Btile, metal::true_type{});
    }

    const bool skipFwht = (p.isIncumbent & 0x10000u) != 0u;
    if constexpr (MODE == 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (skipFwht) {
        device half* dst = attOut + ((size_t)warpT0 * p.qHeads + h) * p.vHeadDim + warpC0;
        const int ldOut = int(p.qHeads * p.vHeadDim);
        if (alignedM && warpC0 + 32u <= p.vHeadDim) {
            Dtile.store(dst, ldOut);
        } else {
            Dtile.store_safe(dst, ldOut, short2(short(min(32u, p.vHeadDim - warpC0)), short(sgpSm)));
        }
        return;
    }

    threadgroup float* osWarp = osBase + sg * (32u * MLA_PROJ_NAX_OS);
    Dtile.store<float, MLA_PROJ_NAX_OS, 1>(osWarp);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint outStride = p.qHeads * p.vHeadDim;
    const uint globalCol = warpC0 + lane;
    for (uint m = 0u; m < uint(sgpSm); ++m) {
        const float rot = qwg_fwht32_lane_f32(osWarp[m * MLA_PROJ_NAX_OS + lane], lane);
        if (globalCol < p.vHeadDim) {
            attOut[(size_t)(warpT0 + m) * outStride + h * p.vHeadDim + globalCol] = half(rot);
        }
    }
}

template <bool SUB2>
METAL_FUNC void qw_mla_proj_uv_nax_body(
    device const half* uLat, device const char* kvBBlocks, device half* attOut,
    constant MLAAbsorbParams& p, threadgroup MLAProjNaxTg& st,
    uint3 tg, uint tid, uint sg, uint lane
) {
    const uint cBase = tg.x * 64u, tBase = tg.y * 64u, h = tg.z;
    if (h >= p.qHeads || tBase >= p.T || cBase >= p.vHeadDim) return;

    const uint inc = p.isIncumbent & 0xFFu;
    if (inc == 2u && qw_mla_load_shvq_lut<SUB2>(st.tgLut, p.bits, tid, 128u)) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint wm = sg >> 1u, wn = sg & 1u;
    const uint warpT0 = tBase + wm * 32u, warpC0 = cBase + wn * 32u;
    const int sgpSm = min(32, int(p.T) - int(warpT0));

    if (inc == 2u) {
        if (sgpSm <= 0 || warpC0 >= p.vHeadDim) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            return;
        }
        qw_mla_proj_uv_nax_run<2, SUB2>(uLat, kvBBlocks, attOut, p, &st.os[0][0], st.tgLut,
                                        h, warpT0, warpC0, sgpSm, sg, lane);
        return;
    }

    if (sgpSm <= 0 || warpC0 >= p.vHeadDim) return;

    if (inc == 3u || p.bits == 16u) {
        qw_mla_proj_uv_nax_run<3>(uLat, kvBBlocks, attOut, p, &st.os[0][0], st.tgLut,
                                  h, warpT0, warpC0, sgpSm, sg, lane);
    } else if (inc == 1u) {
        qw_mla_proj_uv_nax_run<1>(uLat, kvBBlocks, attOut, p, &st.os[0][0], st.tgLut,
                                  h, warpT0, warpC0, sgpSm, sg, lane);
    } else if (p.bits == 8u) {
        qw_mla_proj_uv_nax_run<8>(uLat, kvBBlocks, attOut, p, &st.os[0][0], st.tgLut,
                                  h, warpT0, warpC0, sgpSm, sg, lane);
    } else if (p.bits == 4u) {
        qw_mla_proj_uv_nax_run<4>(uLat, kvBBlocks, attOut, p, &st.os[0][0], st.tgLut,
                                  h, warpT0, warpC0, sgpSm, sg, lane);
    } else {
        qw_mla_proj_uv_nax_run<0>(uLat, kvBBlocks, attOut, p, &st.os[0][0], st.tgLut,
                                  h, warpT0, warpC0, sgpSm, sg, lane);
    }
}

// Grid: (ceil(vHeadDim / 64), ceil(T / 64), qHeads), Threadgroup: (128, 1, 1).
kernel void qwg_mla_proj_uv_nax(
    device const half* uLat [[buffer(0)]], device const char* kvBBlocks [[buffer(1)]],
    device half* attOut [[buffer(2)]], constant MLAAbsorbParams& p [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup MLAProjNaxTg st;
    qw_mla_proj_uv_nax_body<false>(uLat, kvBBlocks, attOut, p, st, tg, tid, sg, lane);
}

kernel void qwg_mla_proj_uv_nax_sub2(
    device const half* uLat [[buffer(0)]], device const char* kvBBlocks [[buffer(1)]],
    device half* attOut [[buffer(2)]], constant MLAAbsorbParams& p [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup MLAProjNaxTg st;
    qw_mla_proj_uv_nax_body<true>(uLat, kvBBlocks, attOut, p, st, tg, tid, sg, lane);
}
#endif // !defined(QW_DISABLE_NAX)
