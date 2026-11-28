// qwgemm_fp_moe_nax_swiglu.metal — Fused BN=32 grouped-NAX MoE gate+up+softcap SwiGLU and sorted gather.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_dequant_fp.metalh"
#include "qw_moe_tile.metalh"

using namespace metal;

#if !defined(QW_DISABLE_NAX)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#if !defined(__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__) || __ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ < 260200
#error "NAX cooperative_tensor kernels require -mmacosx-version-min=26.2 or newer (Apple MPP operand_index ABI)."
#endif
#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "qw_moe_fp_nax.metalh"

using namespace mlx::steel;
#endif // !defined(QW_DISABLE_NAX)

struct QWGemmFPMoENaxSwigluParams {
    uint experts, n, k, wRows, inDtype, outDtype, flags;
    float gateCap, upCap;
};

struct QWGemmFPMoEGatherParams {
    uint totalPairs, k, inDtype, outDtype;
};

kernel void qwg_moe_gather_x(
    device const char* x [[buffer(0)]],
    device const int32_t* sortedTok [[buffer(1)]],
    device char* xSorted [[buffer(2)]],
    constant QWGemmFPMoEGatherParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint s = gid.y;
    const uint c8 = gid.x * 8u;
    if (s >= p.totalPairs || c8 >= p.k) return;
    const int tok = sortedTok[s];
    const size_t dstBase = size_t(s) * p.k + c8;
    if (tok < 0) {
        for (uint j = 0u; j < 8u && c8 + j < p.k; ++j) qwg_fp_write(xSorted, dstBase + j, 0.0f, p.outDtype);
        return;
    }
    const size_t srcBase = size_t(tok) * p.k + c8;
    const bool vec8 = (c8 + 8u <= p.k) && ((p.k & 7u) == 0u) && (p.inDtype <= 1u) && (p.outDtype <= 1u)
        && ((size_t(x) & 15u) == 0u) && ((size_t(xSorted) & 15u) == 0u);
    if (vec8 && p.inDtype == p.outDtype) {
        *(device uint4*)(xSorted + dstBase * 2u) = *(device const uint4*)(x + srcBase * 2u);
        return;
    }
    if (vec8 && p.inDtype == 1u && p.outDtype == 0u) {
        const ushort4 u0 = *(device const ushort4*)((device const bfloat*)x + srcBase);
        const ushort4 u1 = *(device const ushort4*)((device const bfloat*)x + srcBase + 4u);
        const float4 f0 = clamp(float4(as_type<bfloat4>(u0)), -65504.0f, 65504.0f);
        const float4 f1 = clamp(float4(as_type<bfloat4>(u1)), -65504.0f, 65504.0f);
        const ushort4 h0 = as_type<ushort4>(half4(f0));
        const ushort4 h1 = as_type<ushort4>(half4(f1));
        *(device uint4*)((device half*)xSorted + dstBase) = uint4(
            uint(h0.x) | (uint(h0.y) << 16u), uint(h0.z) | (uint(h0.w) << 16u),
            uint(h1.x) | (uint(h1.y) << 16u), uint(h1.z) | (uint(h1.w) << 16u));
        return;
    }
    for (uint j = 0u; j < 8u && c8 + j < p.k; ++j) {
        float v = qwg_fp_read(x, srcBase + j, p.inDtype);
        if (p.outDtype == 0u) v = clamp(v, -65504.0f, 65504.0f);
        qwg_fp_write(xSorted, dstBase + j, v, p.outDtype);
    }
}

#if !defined(QW_DISABLE_NAX)
template <bool IS_MXFP8, typename ActT, typename WT, int BM, int BN, int WM, int WN>
METAL_FUNC void qwg_moe_nax_fp_swiglu_tile_impl(
    device const uint32_t* wExp, device const uchar* scExp, float s2Gate, float s2Up,
    device const ActT* xBase, device char* yBase, int rowCount, uint yCol,
    constant QWGemmFPMoENaxSwigluParams& p, device const int32_t* sortedTok,
    threadgroup const QWGFPTgLut& tgLut, threadgroup ushort4* bShU4,
    uint tidY, uint lid, uint simd_gid) {
    constexpr int SK = 32;
    constexpr short SM = BM / WM, SN = BN / WN;
    constexpr short TM = SM / 16, TN = SN / 16, TK = SK / 16;
    const short tm = SM * short(simd_gid / WN), tn = SN * short(simd_gid % WN);
    const int y_row = int(tidY) * BM;
    const short sgp_sm = short(min(int(SM), rowCount - (y_row + int(tm))));
    const short sgp_sn = short(min(int(SN), int(p.n) - (int(yCol) + int(tn))));
    const int tgp_bn = min(BN, int(p.n) - int(yCol));
    if (rowCount <= y_row || tgp_bn <= 0) return;
    const bool sgActive = (sgp_sm > 0 && sgp_sn > 0 && (BM != 16 || simd_gid < 2u));

    const short2 sc = BaseNAXFrag::get_coord();
    const bool hasGather = (p.flags & 2u) != 0u;
    const bool alignedM = (sgp_sm == SM);
    const uint kGroups = IS_MXFP8 ? (p.k >> 5u) : (p.k >> 4u);
    const uint wWordsPerRow = IS_MXFP8 ? (kGroups * 8u) : (kGroups * 2u);
    const size_t gateRowBase = size_t(yCol), upRowBase = size_t(p.n) + size_t(yCol);
    const bool alignedW = IS_MXFP8 ? ((size_t(wExp) & 15u) == 0u) : ((size_t(wExp) & 7u) == 0u);

    NAXTile<float, TM, TN> Dgate, Dup;
    Dgate.clear();
    Dup.clear();

    if (hasGather) {
        const device ActT* x_rows[TM * 2];
        if (sgActive) {
            STEEL_PRAGMA_UNROLL
            for (short m = 0; m < TM; ++m) {
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; ++i) {
                    const int r = y_row + int(tm) + int(m) * 16 + int(sc.y) + int(i) * 8;
                    const int tok = (r < rowCount) ? sortedTok[r] : -1;
                    x_rows[m * 2 + i] = (tok >= 0) ? (xBase + size_t(tok) * p.k + int(sc.x)) : nullptr;
                }
            }
        }
        STEEL_PRAGMA_NO_UNROLL
        for (uint k0 = 0u; k0 < p.k; k0 += uint(SK)) {
            qwg_moe_stage_b_k32<IS_MXFP8, WT, true>(
                bShU4, tgLut, wExp, scExp, gateRowBase, upRowBase,
                k0, p.k, kGroups, wWordsPerRow, tgp_bn, alignedW, lid);
            NAXTile<ActT, TM, TK> Atile;
            if (sgActive) qwg_moe_nax_load_a_tile<ActT, TM, TK, true>(Atile, x_rows, xBase, k0, p.k, sgp_sm, alignedM);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgActive) {
                NAXTile<WT, TN, TK> Bgate, Bup;
                Bgate.template load<WT, 36, 1>((threadgroup const WT*)(bShU4 + size_t(tn) * 9u));
                tile_matmad_nax(Dgate, Atile, metal::bool_constant<false>{}, Bgate, metal::bool_constant<true>{});
                Bup.template load<WT, 36, 1>((threadgroup const WT*)(bShU4 + size_t(32 + tn) * 9u));
                tile_matmad_nax(Dup, Atile, metal::bool_constant<false>{}, Bup, metal::bool_constant<true>{});
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        const device ActT* xPtr = xBase + size_t(min(rowCount - 1, y_row + int(tm))) * p.k;
        const device ActT* dummyRows[TM * 2] = { nullptr };
        const bool fastA = alignedM && ((p.k & 31u) == 0u);
        if (fastA) {
            STEEL_PRAGMA_NO_UNROLL
            for (uint k0 = 0u; k0 < p.k; k0 += uint(SK)) {
                qwg_moe_stage_b_k32<IS_MXFP8, WT, true>(
                    bShU4, tgLut, wExp, scExp, gateRowBase, upRowBase,
                    k0, p.k, kGroups, wWordsPerRow, tgp_bn, alignedW, lid);
                NAXTile<ActT, TM, TK> Atile;
                if (sgActive) Atile.load(xPtr + k0, int(p.k));
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sgActive) {
                    NAXTile<WT, TN, TK> Bgate, Bup;
                    Bgate.template load<WT, 36, 1>((threadgroup const WT*)(bShU4 + size_t(tn) * 9u));
                    tile_matmad_nax(Dgate, Atile, metal::bool_constant<false>{}, Bgate, metal::bool_constant<true>{});
                    Bup.template load<WT, 36, 1>((threadgroup const WT*)(bShU4 + size_t(32 + tn) * 9u));
                    tile_matmad_nax(Dup, Atile, metal::bool_constant<false>{}, Bup, metal::bool_constant<true>{});
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        } else {
            STEEL_PRAGMA_NO_UNROLL
            for (uint k0 = 0u; k0 < p.k; k0 += uint(SK)) {
                qwg_moe_stage_b_k32<IS_MXFP8, WT, true>(
                    bShU4, tgLut, wExp, scExp, gateRowBase, upRowBase,
                    k0, p.k, kGroups, wWordsPerRow, tgp_bn, alignedW, lid);
                NAXTile<ActT, TM, TK> Atile;
                if (sgActive) qwg_moe_nax_load_a_tile<ActT, TM, TK, false>(Atile, dummyRows, xPtr, k0, p.k, sgp_sm, alignedM);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sgActive) {
                    NAXTile<WT, TN, TK> Bgate, Bup;
                    Bgate.template load<WT, 36, 1>((threadgroup const WT*)(bShU4 + size_t(tn) * 9u));
                    tile_matmad_nax(Dgate, Atile, metal::bool_constant<false>{}, Bgate, metal::bool_constant<true>{});
                    Bup.template load<WT, 36, 1>((threadgroup const WT*)(bShU4 + size_t(32 + tn) * 9u));
                    tile_matmad_nax(Dup, Atile, metal::bool_constant<false>{}, Bup, metal::bool_constant<true>{});
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    if (!sgActive) return;
    const size_t yAlign = (p.outDtype == 2u) ? 15u : 7u;
    const_for_loop<0, TM, 1>([&](auto mm) {
        const_for_loop<0, TN, 1>([&](auto nn) {
            thread const auto& gf = Dgate.template frag_at<mm.value, nn.value>();
            thread const auto& uf = Dup.template frag_at<mm.value, nn.value>();
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; ++i) {
                const int gr = y_row + int(tm) + mm.value * 16 + int(sc.y) + int(i) * 8;
                if (gr >= rowCount) continue;
                const uint gc0 = yCol + uint(tn) + uint(nn.value * 16 + int(sc.x));
                const size_t baseIdx = size_t(gr) * size_t(p.n) + size_t(gc0);
                float4 ov;
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; ++j) {
                    ov[j] = qwg_fp_softcap_swiglu(gf[i * 4 + j] * s2Gate, uf[i * 4 + j] * s2Up,
                                                  p.gateCap, p.upCap);
                }
                if (gc0 + 4u <= p.n && (baseIdx & 3u) == 0u && ((size_t(yBase) & yAlign) == 0u)) {
                    qwg_fp_write4(yBase, baseIdx, ov, p.outDtype);
                } else {
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; ++j) {
                        if (gc0 + uint(j) < p.n) qwg_fp_write(yBase, baseIdx + uint(j), ov[j], p.outDtype);
                    }
                }
            }
        });
    });
}

template <bool IS_MXFP8, typename ActT, typename WT>
METAL_FUNC void qwg_moe_nax_fp_swiglu_launch(
    int bm, device const uint32_t* wExp, device const uchar* scExp,
    float s2Gate, float s2Up, device const ActT* xPtr, device char* yExp,
    int rowCount, uint yCol, constant QWGemmFPMoENaxSwigluParams& p,
    device const int32_t* st, threadgroup const QWGFPTgLut& tgLut,
    threadgroup ushort4* bShU4, uint tidY, uint lid, uint simd_gid) {
    if (bm == 16) {
        qwg_moe_nax_fp_swiglu_tile_impl<IS_MXFP8, ActT, WT, 16, 32, 1, 2>(
            wExp, scExp, s2Gate, s2Up, xPtr, yExp, rowCount, yCol, p, st, tgLut, bShU4, tidY, lid, simd_gid);
    } else if (bm == 32) {
        qwg_moe_nax_fp_swiglu_tile_impl<IS_MXFP8, ActT, WT, 32, 32, 2, 2>(
            wExp, scExp, s2Gate, s2Up, xPtr, yExp, rowCount, yCol, p, st, tgLut, bShU4, tidY, lid, simd_gid);
    } else {
        qwg_moe_nax_fp_swiglu_tile_impl<IS_MXFP8, ActT, WT, 64, 32, 4, 1>(
            wExp, scExp, s2Gate, s2Up, xPtr, yExp, rowCount, yCol, p, st, tgLut, bShU4, tidY, lid, simd_gid);
    }
}

template <bool IS_MXFP8>
METAL_FUNC void qwg_moe_nax_fp_swiglu_dispatch(
    device const uint32_t* w, device const uchar* sc, device const float* s2,
    device const char* x, device char* y,
    device const int32_t* expertOff, device const int32_t* tileE,
    device const int32_t* tileR, device const int32_t* tileCount,
    constant QWGemmFPMoENaxSwigluParams& p, uint tileBlock,
    device const int32_t* sortedTok, threadgroup QWGFPTgLut& tgLut,
    threadgroup ushort4* bShU4, uint3 tg, uint lid, uint simd_gid) {
    const uint t = tg.y + ((tileBlock >> 8) & 0x7FFFFFu);
    if (int(t) >= tileCount[0]) return;
    const uint e = uint(tileE[t]);
    if (e >= p.experts) return;
    const int rowBase = int(expertOff[e]), rowCount = int(expertOff[e + 1]) - rowBase;
    if (rowCount <= 0) return;

    tgLut.load(lid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint kGroups = IS_MXFP8 ? (p.k >> 5u) : (p.k >> 4u);
    const uint wWordsPerRow = IS_MXFP8 ? (kGroups * 8u) : (kGroups * 2u);
    const size_t expRows = size_t(e) * size_t(p.wRows);
    device const uint32_t* wExp = w + expRows * wWordsPerRow;
    device const uchar* scExp = sc + expRows * kGroups;
    const bool useSplitS2 = !IS_MXFP8 && ((p.flags & 9u) == 9u);
    const float s2Gate = (!IS_MXFP8 && (p.flags & 1u) != 0u) ? s2[e] : 1.0f;
    const float s2Up = useSplitS2 ? s2[p.experts + e] : s2Gate;
    const bool hasGather = (p.flags & 2u) != 0u;
    const size_t outElemSz = (p.outDtype == 2u) ? 4u : 2u;
    device char* yExp = y + size_t(rowBase) * size_t(p.n) * outElemSz;
    device const int32_t* st = sortedTok + rowBase;

    int bm; uint tidY;
    qw_moe_decode_tile(tileR[t], rowCount, int(tileBlock & 0x3Fu), bm, tidY);
    const uint yCol = tg.x * 32u;

    if constexpr (IS_MXFP8) {
        if (p.inDtype == 1u) {
            device const bfloat* xb = (device const bfloat*)x + (hasGather ? 0 : size_t(rowBase) * p.k);
            qwg_moe_nax_fp_swiglu_launch<true, bfloat, bfloat>(
                bm, wExp, scExp, 1.0f, 1.0f, xb, yExp, rowCount, yCol, p, st, tgLut, bShU4, tidY, lid, simd_gid);
        } else {
            device const half* xh = (device const half*)x + (hasGather ? 0 : size_t(rowBase) * p.k);
            qwg_moe_nax_fp_swiglu_launch<true, half, bfloat>(
                bm, wExp, scExp, 1.0f, 1.0f, xh, yExp, rowCount, yCol, p, st, tgLut, bShU4, tidY, lid, simd_gid);
        }
    } else if (p.inDtype == 1u) {
        device const bfloat* xb = (device const bfloat*)x + (hasGather ? 0 : size_t(rowBase) * p.k);
        qwg_moe_nax_fp_swiglu_launch<false, bfloat, bfloat>(
            bm, wExp, scExp, s2Gate, s2Up, xb, yExp, rowCount, yCol, p, st, tgLut, bShU4, tidY, lid, simd_gid);
    } else {
        device const half* xh = (device const half*)x + (hasGather ? 0 : size_t(rowBase) * p.k);
        qwg_moe_nax_fp_swiglu_launch<false, half, half>(
            bm, wExp, scExp, s2Gate, s2Up, xh, yExp, rowCount, yCol, p, st, tgLut, bShU4, tidY, lid, simd_gid);
    }
}

kernel void qwg_moe_nax_gemm_nvfp4_g16_swiglu(
    device const uint32_t* w [[buffer(0)]], device const uchar* sc [[buffer(1)]],
    device const float* s2 [[buffer(2)]], device const char* x [[buffer(3)]],
    device char* y [[buffer(4)]], device const int32_t* expertOff [[buffer(5)]],
    device const int32_t* tileE [[buffer(6)]], device const int32_t* tileR [[buffer(7)]],
    device const int32_t* tileCount [[buffer(8)]], constant QWGemmFPMoENaxSwigluParams& p [[buffer(9)]],
    constant uint& tileBlock [[buffer(10)]], device const int32_t* sortedTok [[buffer(11)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
    threadgroup QWGFPTgLut tgLut;
    threadgroup ushort4 bShU4[64 * 9];
    qwg_moe_nax_fp_swiglu_dispatch<false>(
        w, sc, s2, x, y, expertOff, tileE, tileR, tileCount, p, tileBlock, sortedTok, tgLut, bShU4, tg, lid, simd_gid);
}

kernel void qwg_moe_nax_gemm_mxfp8_g32_swiglu(
    device const uint32_t* w [[buffer(0)]], device const uchar* sc [[buffer(1)]],
    device const char* x [[buffer(2)]], device char* y [[buffer(3)]],
    device const int32_t* expertOff [[buffer(4)]], device const int32_t* tileE [[buffer(5)]],
    device const int32_t* tileR [[buffer(6)]], device const int32_t* tileCount [[buffer(7)]],
    constant QWGemmFPMoENaxSwigluParams& p [[buffer(8)]], constant uint& tileBlock [[buffer(9)]],
    device const int32_t* sortedTok [[buffer(10)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
    threadgroup QWGFPTgLut tgLut;
    threadgroup ushort4 bShU4[64 * 9];
    qwg_moe_nax_fp_swiglu_dispatch<true>(
        w, sc, nullptr, x, y, expertOff, tileE, tileR, tileCount, p, tileBlock, sortedTok, tgLut, bShU4, tg, lid, simd_gid);
}
#endif // !defined(QW_DISABLE_NAX)
