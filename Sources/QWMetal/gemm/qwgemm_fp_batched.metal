// qwgemm_fp_batched.metal — Generalized BF16 NAX/GEMV, NVFP4_G16 & MXFP8_G32 MoE NAX prefill,
// active-bucket compact decode/verify, softcap SwiGLU, and BF16 combine kernels.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_dequant_fp.metalh"
#include "qw_moe_compact_fp.metalh"
#include "qw_moe_tile.metalh"

using namespace metal;

#if !defined(QW_DISABLE_NAX)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#if !defined(__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__) || __ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ < 260200
#error "NAX cooperative_tensor kernels require -mmacosx-version-min=26.2 or newer (Apple MPP operand_index ABI)."
#endif
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "qw_moe_fp_nax.metalh"

using namespace mlx::steel;
#endif // !defined(QW_DISABLE_NAX)

struct QWGemmFPMoENaxParams {
    uint experts, n, k, yStride, yColOff, inDtype, outDtype, flags;
};

struct QWGemmFPSwiGLUParams {
    uint M, I;
    float gateCap, upCap;
    uint flags;
};

struct QWGemmFPCombineParams {
    uint hidden, topK, tokens, inDtype, outDtype, sharedDtype, flags;
};

#if !defined(QW_DISABLE_NAX)
template <int BM, int BN, int BK, int WM, int WN>
METAL_FUNC void qwg_gemm_bf16_nax_body(
    device const bfloat* W, device const bfloat* xBF16, device const float* xF32,
    device bfloat* yBF16, device float* yF32, constant QWGemmFPBF16Params& p,
    uint2 tg, uint simd_gid) {
    constexpr short SM = BM / WM, SN = BN / WN, SK = BK;
    constexpr short TM = SM / 16, TN = SN / 16, TK = SK / 16;
    const short sgM = short(simd_gid / WN), sgN = short(simd_gid % WN);
    const uint m0 = tg.y * uint(BM) + uint(sgM * SM);
    const uint n0 = tg.x * uint(BN) + uint(sgN * SN);
    const short sgp_sm = short(min(int(SM), max(0, int(p.M) - int(m0))));
    const short sgp_sn = short(min(int(SN), max(0, int(p.N) - int(n0))));
    if (sgp_sm <= 0 || sgp_sn <= 0) return;

    NAXTile<float, TM, TN> Dtile;
    Dtile.clear();
    const short2 sc = BaseNAXFrag::get_coord();
    const bool alignedM = (sgp_sm == SM), alignedN = (sgp_sn == SN);
    const bool vecW = ((p.K & 3u) == 0u) && ((size_t(W) & 7u) == 0u);
    const bool vecX = (p.inIsF32 == 1u)
        ? (((p.strideX & 3u) == 0u) && ((size_t(xF32) & 15u) == 0u))
        : (((p.strideX & 3u) == 0u) && ((size_t(xBF16) & 7u) == 0u));

    for (uint k0 = 0u; k0 < p.K; k0 += uint(BK)) {
        NAXTile<bfloat, TM, TK> Atile;
        NAXTile<bfloat, TN, TK> Btile;
        const bool fullK = (k0 + uint(BK) <= p.K);

        const_for_loop<0, TM, 1>([&](auto mr) {
            const_for_loop<0, TK, 1>([&](auto kc) {
                thread auto& af = Atile.template frag_at<mr.value, kc.value>();
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; ++i) {
                    const uint r = m0 + uint(mr.value * 16 + int(sc.y) + i * 8);
                    const uint c = k0 + uint(kc.value * 16 + int(sc.x));
                    if (p.inIsF32 == 0u && ((alignedM && fullK) || (r < p.M && c + 4u <= p.K)) && vecX) {
                        const ushort4 v = *(device const ushort4*)(xBF16 + size_t(r) * p.strideX + c);
                        af[i * 4 + 0] = as_type<bfloat>(v.x); af[i * 4 + 1] = as_type<bfloat>(v.y);
                        af[i * 4 + 2] = as_type<bfloat>(v.z); af[i * 4 + 3] = as_type<bfloat>(v.w);
                    } else if (r < p.M && c + 4u <= p.K && vecX) {
                        if (p.inIsF32 == 1u) {
                            const float4 vf = *(device const float4*)(xF32 + size_t(r) * p.strideX + c);
                            af[i * 4 + 0] = qwg_fp_bf16(vf.x); af[i * 4 + 1] = qwg_fp_bf16(vf.y);
                            af[i * 4 + 2] = qwg_fp_bf16(vf.z); af[i * 4 + 3] = qwg_fp_bf16(vf.w);
                        } else {
                            const ushort4 v = *(device const ushort4*)((device const ushort*)xBF16 + size_t(r) * p.strideX + c);
                            af[i * 4 + 0] = qwg_fp_bf16(float(as_type<half>(v.x)));
                            af[i * 4 + 1] = qwg_fp_bf16(float(as_type<half>(v.y)));
                            af[i * 4 + 2] = qwg_fp_bf16(float(as_type<half>(v.z)));
                            af[i * 4 + 3] = qwg_fp_bf16(float(as_type<half>(v.w)));
                        }
                    } else {
                        STEEL_PRAGMA_UNROLL
                        for (short j = 0; j < 4; ++j) {
                            if (r < p.M && c + uint(j) < p.K) {
                                const size_t idx = size_t(r) * p.strideX + c + uint(j);
                                if (p.inIsF32 == 1u) af[i * 4 + j] = qwg_fp_bf16(xF32[idx]);
                                else if (p.inIsF32 == 2u) af[i * 4 + j] = qwg_fp_bf16(float(((device const half*)xBF16)[idx]));
                                else af[i * 4 + j] = xBF16[idx];
                            } else {
                                af[i * 4 + j] = bfloat(0.0f);
                            }
                        }
                    }
                }
            });
        });

        const_for_loop<0, TN, 1>([&](auto nr) {
            const_for_loop<0, TK, 1>([&](auto kc) {
                thread auto& bf = Btile.template frag_at<nr.value, kc.value>();
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; ++i) {
                    const uint r = n0 + uint(nr.value * 16 + int(sc.y) + i * 8);
                    const uint c = k0 + uint(kc.value * 16 + int(sc.x));
                    if (((alignedN && fullK) || (r < p.N && c + 4u <= p.K)) && vecW) {
                        const ushort4 v = *(device const ushort4*)(W + size_t(r) * p.K + c);
                        bf[i * 4 + 0] = as_type<bfloat>(v.x); bf[i * 4 + 1] = as_type<bfloat>(v.y);
                        bf[i * 4 + 2] = as_type<bfloat>(v.z); bf[i * 4 + 3] = as_type<bfloat>(v.w);
                    } else {
                        STEEL_PRAGMA_UNROLL
                        for (short j = 0; j < 4; ++j) {
                            bf[i * 4 + j] = (r < p.N && c + uint(j) < p.K)
                                ? W[size_t(r) * p.K + c + uint(j)] : bfloat(0.0f);
                        }
                    }
                }
            });
        });

        tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
    }

    const bool fastStoreBF = (p.addResidual == 0u) && (p.softcap == 0.0f) && (p.outIsF32 == 0u)
        && ((p.strideY & 3u) == 0u) && ((size_t(yBF16) & 7u) == 0u);
    const_for_loop<0, TM, 1>([&](auto mm) {
        const_for_loop<0, TN, 1>([&](auto nn) {
            thread const auto& df = Dtile.template frag_at<mm.value, nn.value>();
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; ++i) {
                const uint gr = m0 + uint(mm.value * 16 + int(sc.y) + i * 8);
                if (gr >= p.M) continue;
                const uint gc0 = n0 + uint(nn.value * 16 + int(sc.x));
                if (fastStoreBF && gc0 + 4u <= p.N) {
                    *(device ushort4*)(yBF16 + size_t(gr) * p.strideY + gc0) = ushort4(
                        as_type<ushort>(qwg_fp_bf16(df[i * 4 + 0])),
                        as_type<ushort>(qwg_fp_bf16(df[i * 4 + 1])),
                        as_type<ushort>(qwg_fp_bf16(df[i * 4 + 2])),
                        as_type<ushort>(qwg_fp_bf16(df[i * 4 + 3])));
                } else {
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; ++j) {
                        const uint gc = gc0 + uint(j);
                        if (gc >= p.N) continue;
                        float val = df[i * 4 + j];
                        const size_t yIdx = size_t(gr) * p.strideY + gc;
                        if (p.addResidual != 0u) {
                            if (p.outIsF32 == 1u) val += yF32[yIdx];
                            else if (p.outIsF32 == 2u) val += float(((device const half*)yBF16)[yIdx]);
                            else val += float(yBF16[yIdx]);
                        }
                        if (p.softcap > 0.0f) val = precise::tanh(clamp(val / p.softcap, -15.0f, 15.0f)) * p.softcap;
                        if (p.outIsF32 == 1u) yF32[yIdx] = val;
                        else if (p.outIsF32 == 2u) ((device half*)yBF16)[yIdx] = half(val);
                        else yBF16[yIdx] = qwg_fp_bf16(val);
                    }
                }
            }
        });
    });
}

kernel void qwg_gemm_bf16_nax_bm32(device const bfloat* W [[buffer(0)]],
                                   device const bfloat* xBF16 [[buffer(1)]],
                                   device const float* xF32 [[buffer(2)]],
                                   device bfloat* yBF16 [[buffer(3)]],
                                   device float* yF32 [[buffer(4)]],
                                   constant QWGemmFPBF16Params& p [[buffer(5)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint lid [[thread_index_in_threadgroup]],
                                   uint simd_gid [[simdgroup_index_in_threadgroup]]) {
    threadgroup ushort4 aShU4[32 * 9];
    threadgroup ushort4 bShU4[64 * 9];
    const bool coopOk = (p.K & 31u) == 0u && ((size_t(W) & 7u) == 0u)
        && ((size_t(xBF16) & 7u) == 0u) && (p.strideX & 3u) == 0u;
    if (coopOk && p.inIsF32 == 0u) {
        qwg_gemm_bf16_nax_coop_fast<32, false>(W, xBF16, yBF16, yF32, p, aShU4, bShU4, tg, lid, simd_gid);
        return;
    }
    if (coopOk && p.inIsF32 == 2u) {
        qwg_gemm_bf16_nax_coop_fast<32, true>(W, xBF16, yBF16, yF32, p, aShU4, bShU4, tg, lid, simd_gid);
        return;
    }
    qwg_gemm_bf16_nax_body<32, 64, 64, 2, 2>(W, xBF16, xF32, yBF16, yF32, p, tg, simd_gid);
}

kernel void qwg_gemm_bf16_nax_bm64(device const bfloat* W [[buffer(0)]],
                                   device const bfloat* xBF16 [[buffer(1)]],
                                   device const float* xF32 [[buffer(2)]],
                                   device bfloat* yBF16 [[buffer(3)]],
                                   device float* yF32 [[buffer(4)]],
                                   constant QWGemmFPBF16Params& p [[buffer(5)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint lid [[thread_index_in_threadgroup]],
                                   uint simd_gid [[simdgroup_index_in_threadgroup]]) {
    threadgroup ushort4 aShU4[64 * 9];
    threadgroup ushort4 bShU4[64 * 9];
    const bool coopOk = (p.K & 31u) == 0u && ((size_t(W) & 7u) == 0u)
        && ((size_t(xBF16) & 7u) == 0u) && (p.strideX & 3u) == 0u;
    if (coopOk && p.inIsF32 == 0u) {
        qwg_gemm_bf16_nax_coop_fast<64, false>(W, xBF16, yBF16, yF32, p, aShU4, bShU4, tg, lid, simd_gid);
        return;
    }
    if (coopOk && p.inIsF32 == 2u) {
        qwg_gemm_bf16_nax_coop_fast<64, true>(W, xBF16, yBF16, yF32, p, aShU4, bShU4, tg, lid, simd_gid);
        return;
    }
    qwg_gemm_bf16_nax_body<64, 64, 32, 2, 2>(W, xBF16, xF32, yBF16, yF32, p, tg, simd_gid);
}
#endif // !defined(QW_DISABLE_NAX)

#define QW_GEMV_BF16_INST(R, MVAL)                                                                \
    kernel void qwg_gemv_bf16_rows##R##_m##MVAL(                                                  \
        device const bfloat* W [[buffer(0)]], device const bfloat* xBF16 [[buffer(1)]],           \
        device const float* xF32 [[buffer(2)]], device bfloat* yBF16 [[buffer(3)]],               \
        device float* yF32 [[buffer(4)]], constant QWGemmFPBF16Params& p [[buffer(5)]],           \
        uint2 tg [[threadgroup_position_in_grid]],                                                \
        uint simd_id [[simdgroup_index_in_threadgroup]],                                          \
        uint lane [[thread_index_in_simdgroup]]) {                                                \
        qwg_gemv_bf16_rows_r_m_body<MVAL##u, R##u>(W, xBF16, xF32, yBF16, yF32, p, tg, simd_id, lane); \
    }

QW_GEMV_BF16_INST(2, 1) QW_GEMV_BF16_INST(2, 2) QW_GEMV_BF16_INST(2, 3) QW_GEMV_BF16_INST(2, 4)
QW_GEMV_BF16_INST(2, 5) QW_GEMV_BF16_INST(2, 6) QW_GEMV_BF16_INST(2, 7) QW_GEMV_BF16_INST(2, 8)
QW_GEMV_BF16_INST(4, 1) QW_GEMV_BF16_INST(4, 2) QW_GEMV_BF16_INST(4, 3) QW_GEMV_BF16_INST(4, 4)
QW_GEMV_BF16_INST(4, 5) QW_GEMV_BF16_INST(4, 6) QW_GEMV_BF16_INST(4, 7) QW_GEMV_BF16_INST(4, 8)

#if !defined(QW_DISABLE_NAX)
template <bool IS_MXFP8, typename ActT, typename WT, int BM, int WM, int WN>
METAL_FUNC void qwg_moe_nax_fp_tile_impl(
    device const uint32_t* wExp, device const uchar* scExp, float s2Gate, float s2Up,
    bool splitS2, device const ActT* xBase, device char* yBase, int rowCount, uint yCol,
    constant QWGemmFPMoENaxParams& p, device const int32_t* sortedTok,
    threadgroup const QWGFPTgLut& tgLut, threadgroup ushort4* bShU4,
    uint tidY, uint lid, uint simd_gid) {
    constexpr int BN = 64, SK = 32;
    constexpr short SM = BM / WM, SN = BN / WN;
    constexpr short TM = SM / 16, TN = SN / 16, TK = SK / 16;
    const short tm = SM * short(simd_gid / WN), tn = SN * short(simd_gid % WN);
    const int y_row = int(tidY) * BM;
    const short sgp_sm = short(min(int(SM), rowCount - (y_row + int(tm))));
    const short sgp_sn = short(min(int(SN), int(p.n) - (int(yCol) + int(tn))));
    const int tgp_bn = min(BN, int(p.n) - int(yCol));
    if (rowCount <= y_row || tgp_bn <= 0) return;
    const bool sgActive = (sgp_sm > 0 && sgp_sn > 0);

    const short2 sc = BaseNAXFrag::get_coord();
    const bool hasGather = (p.flags & 2u) != 0u;
    NAXTile<float, TM, TN> Dtile;
    Dtile.clear();
    const bool alignedM = (sgp_sm == SM);
    const uint kGroups = IS_MXFP8 ? (p.k >> 5u) : (p.k >> 4u);
    const uint wWordsPerRow = IS_MXFP8 ? (kGroups * 8u) : (kGroups * 2u);
    const bool alignedW = IS_MXFP8 ? ((size_t(wExp) & 15u) == 0u) : ((size_t(wExp) & 7u) == 0u);

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
            qwg_moe_stage_b_k32<IS_MXFP8, WT, false>(
                bShU4, tgLut, wExp, scExp, size_t(yCol), 0u,
                k0, p.k, kGroups, wWordsPerRow, tgp_bn, alignedW, lid);
            NAXTile<ActT, TM, TK> Atile;
            if (sgActive) qwg_moe_nax_load_a_tile<ActT, TM, TK, true>(Atile, x_rows, xBase, k0, p.k, sgp_sm, alignedM);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgActive) {
                NAXTile<WT, TN, TK> Btile;
                Btile.template load<WT, 36, 1>((threadgroup const WT*)(bShU4 + size_t(tn) * 9u));
                tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
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
                qwg_moe_stage_b_k32<IS_MXFP8, WT, false>(
                    bShU4, tgLut, wExp, scExp, size_t(yCol), 0u,
                    k0, p.k, kGroups, wWordsPerRow, tgp_bn, alignedW, lid);
                NAXTile<ActT, TM, TK> Atile;
                if (sgActive) Atile.load(xPtr + k0, int(p.k));
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sgActive) {
                    NAXTile<WT, TN, TK> Btile;
                    Btile.template load<WT, 36, 1>((threadgroup const WT*)(bShU4 + size_t(tn) * 9u));
                    tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        } else {
            STEEL_PRAGMA_NO_UNROLL
            for (uint k0 = 0u; k0 < p.k; k0 += uint(SK)) {
                qwg_moe_stage_b_k32<IS_MXFP8, WT, false>(
                    bShU4, tgLut, wExp, scExp, size_t(yCol), 0u,
                    k0, p.k, kGroups, wWordsPerRow, tgp_bn, alignedW, lid);
                NAXTile<ActT, TM, TK> Atile;
                if (sgActive) qwg_moe_nax_load_a_tile<ActT, TM, TK, false>(Atile, dummyRows, xPtr, k0, p.k, sgp_sm, alignedM);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sgActive) {
                    NAXTile<WT, TN, TK> Btile;
                    Btile.template load<WT, 36, 1>((threadgroup const WT*)(bShU4 + size_t(tn) * 9u));
                    tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    if (!sgActive) return;
    const uint yStr = (p.yStride > 0u) ? p.yStride : p.n;
    const uint halfN = p.n >> 1u;
    const bool uniformS2 = !splitS2 || (yCol + 64u <= halfN) || (yCol >= halfN);
    const float tileS2 = (splitS2 && yCol >= halfN) ? s2Up : s2Gate;
    const size_t yAlign = (p.outDtype == 2u) ? 15u : 7u;
    const_for_loop<0, TM, 1>([&](auto mm) {
        const_for_loop<0, TN, 1>([&](auto nn) {
            thread const auto& df = Dtile.template frag_at<mm.value, nn.value>();
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; ++i) {
                const int gr = y_row + int(tm) + mm.value * 16 + int(sc.y) + int(i) * 8;
                if (gr >= rowCount) continue;
                const uint gc0 = yCol + uint(tn) + uint(nn.value * 16 + int(sc.x));
                const size_t baseIdx = size_t(gr) * size_t(yStr) + size_t(p.yColOff + gc0);
                if (uniformS2 && gc0 + 4u <= p.n && (baseIdx & 3u) == 0u && ((size_t(yBase) & yAlign) == 0u)) {
                    const float4 v4 = float4(df[i * 4 + 0], df[i * 4 + 1], df[i * 4 + 2], df[i * 4 + 3]) * tileS2;
                    qwg_fp_write4(yBase, baseIdx, v4, p.outDtype);
                } else {
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; ++j) {
                        const uint gc = gc0 + uint(j);
                        if (gc < p.n) {
                            const float rowS2 = uniformS2 ? tileS2 : ((gc >= halfN) ? s2Up : s2Gate);
                            qwg_fp_write(yBase, baseIdx + uint(j), df[i * 4 + j] * rowS2, p.outDtype);
                        }
                    }
                }
            }
        });
    });
}

template <bool IS_MXFP8>
METAL_FUNC void qwg_moe_nax_fp_dispatch(
    device const uint32_t* w, device const uchar* sc, device const float* s2,
    device const char* x, device char* y,
    device const int32_t* expertOff, device const int32_t* tileE,
    device const int32_t* tileR, device const int32_t* tileCount,
    constant QWGemmFPMoENaxParams& p, uint tileBlock,
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
    const size_t expRows = size_t(e) * size_t(p.n);
    device const uint32_t* wExp = w + expRows * wWordsPerRow;
    device const uchar* scExp = sc + expRows * kGroups;
    const bool useSplitS2 = !IS_MXFP8 && ((p.flags & 9u) == 9u);
    const float s2Gate = (!IS_MXFP8 && (p.flags & 1u) != 0u) ? s2[e] : 1.0f;
    const float s2Up = useSplitS2 ? s2[p.experts + e] : s2Gate;
    const bool hasGather = (p.flags & 2u) != 0u;
    const uint yStr = (p.yStride > 0u) ? p.yStride : p.n;
    const size_t outElemSz = (p.outDtype == 2u) ? 4u : 2u;
    device char* yExp = y + size_t(rowBase) * size_t(yStr) * outElemSz;
    device const int32_t* st = sortedTok + rowBase;

    int bm; uint tidY;
    qw_moe_decode_tile(tileR[t], rowCount, int(tileBlock & 0x3Fu), bm, tidY);
    const uint yCol = tg.x * 64u;

    if constexpr (IS_MXFP8) {
        if (p.inDtype == 1u) {
            device const bfloat* xb = (device const bfloat*)x + (hasGather ? 0 : size_t(rowBase) * p.k);
            if (bm == 16) qwg_moe_nax_fp_tile_impl<true, bfloat, bfloat, 16, 1, 4>(wExp, scExp, 1.0f, 1.0f, false, xb, yExp, rowCount, yCol, p, st, tgLut, bShU4, tidY, lid, simd_gid);
            else if (bm == 32) qwg_moe_nax_fp_tile_impl<true, bfloat, bfloat, 32, 2, 2>(wExp, scExp, 1.0f, 1.0f, false, xb, yExp, rowCount, yCol, p, st, tgLut, bShU4, tidY, lid, simd_gid);
            else qwg_moe_nax_fp_tile_impl<true, bfloat, bfloat, 64, 2, 2>(wExp, scExp, 1.0f, 1.0f, false, xb, yExp, rowCount, yCol, p, st, tgLut, bShU4, tidY, lid, simd_gid);
        } else {
            device const half* xh = (device const half*)x + (hasGather ? 0 : size_t(rowBase) * p.k);
            if (bm == 16) qwg_moe_nax_fp_tile_impl<true, half, bfloat, 16, 1, 4>(wExp, scExp, 1.0f, 1.0f, false, xh, yExp, rowCount, yCol, p, st, tgLut, bShU4, tidY, lid, simd_gid);
            else if (bm == 32) qwg_moe_nax_fp_tile_impl<true, half, bfloat, 32, 2, 2>(wExp, scExp, 1.0f, 1.0f, false, xh, yExp, rowCount, yCol, p, st, tgLut, bShU4, tidY, lid, simd_gid);
            else qwg_moe_nax_fp_tile_impl<true, half, bfloat, 64, 2, 2>(wExp, scExp, 1.0f, 1.0f, false, xh, yExp, rowCount, yCol, p, st, tgLut, bShU4, tidY, lid, simd_gid);
        }
    } else {
        if (p.inDtype == 1u) {
            device const bfloat* xb = (device const bfloat*)x + (hasGather ? 0 : size_t(rowBase) * p.k);
            if (bm == 16) qwg_moe_nax_fp_tile_impl<false, bfloat, bfloat, 16, 1, 4>(wExp, scExp, s2Gate, s2Up, useSplitS2, xb, yExp, rowCount, yCol, p, st, tgLut, bShU4, tidY, lid, simd_gid);
            else if (bm == 32) qwg_moe_nax_fp_tile_impl<false, bfloat, bfloat, 32, 2, 2>(wExp, scExp, s2Gate, s2Up, useSplitS2, xb, yExp, rowCount, yCol, p, st, tgLut, bShU4, tidY, lid, simd_gid);
            else qwg_moe_nax_fp_tile_impl<false, bfloat, bfloat, 64, 2, 2>(wExp, scExp, s2Gate, s2Up, useSplitS2, xb, yExp, rowCount, yCol, p, st, tgLut, bShU4, tidY, lid, simd_gid);
        } else {
            device const half* xh = (device const half*)x + (hasGather ? 0 : size_t(rowBase) * p.k);
            if (bm == 16) qwg_moe_nax_fp_tile_impl<false, half, half, 16, 1, 4>(wExp, scExp, s2Gate, s2Up, useSplitS2, xh, yExp, rowCount, yCol, p, st, tgLut, bShU4, tidY, lid, simd_gid);
            else if (bm == 32) qwg_moe_nax_fp_tile_impl<false, half, half, 32, 2, 2>(wExp, scExp, s2Gate, s2Up, useSplitS2, xh, yExp, rowCount, yCol, p, st, tgLut, bShU4, tidY, lid, simd_gid);
            else qwg_moe_nax_fp_tile_impl<false, half, half, 64, 2, 2>(wExp, scExp, s2Gate, s2Up, useSplitS2, xh, yExp, rowCount, yCol, p, st, tgLut, bShU4, tidY, lid, simd_gid);
        }
    }
}

kernel void qwg_moe_nax_gemm_nvfp4_g16(
    device const uint32_t* w [[buffer(0)]], device const uchar* sc [[buffer(1)]],
    device const float* s2 [[buffer(2)]], device const char* x [[buffer(3)]],
    device char* y [[buffer(4)]], device const int32_t* expertOff [[buffer(5)]],
    device const int32_t* tileE [[buffer(6)]], device const int32_t* tileR [[buffer(7)]],
    device const int32_t* tileCount [[buffer(8)]], constant QWGemmFPMoENaxParams& p [[buffer(9)]],
    constant uint& tileBlock [[buffer(10)]], device const int32_t* sortedTok [[buffer(11)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
    threadgroup QWGFPTgLut tgLut;
    threadgroup ushort4 bShU4[64 * 9];
    qwg_moe_nax_fp_dispatch<false>(w, sc, s2, x, y, expertOff, tileE, tileR, tileCount, p, tileBlock, sortedTok, tgLut, bShU4, tg, lid, simd_gid);
}

kernel void qwg_moe_nax_gemm_mxfp8_g32(
    device const uint32_t* w [[buffer(0)]], device const uchar* sc [[buffer(1)]],
    device const char* x [[buffer(2)]], device char* y [[buffer(3)]],
    device const int32_t* expertOff [[buffer(4)]], device const int32_t* tileE [[buffer(5)]],
    device const int32_t* tileR [[buffer(6)]], device const int32_t* tileCount [[buffer(7)]],
    constant QWGemmFPMoENaxParams& p [[buffer(8)]], constant uint& tileBlock [[buffer(9)]],
    device const int32_t* sortedTok [[buffer(10)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
    threadgroup QWGFPTgLut tgLut;
    threadgroup ushort4 bShU4[64 * 9];
    qwg_moe_nax_fp_dispatch<true>(w, sc, nullptr, x, y, expertOff, tileE, tileR, tileCount, p, tileBlock, sortedTok, tgLut, bShU4, tg, lid, simd_gid);
}
#endif // !defined(QW_DISABLE_NAX)

kernel void qwg_moe_build_active_buckets(
    device const int32_t* expertOff [[buffer(0)]], device int32_t* activeExperts [[buffer(1)]],
    device uint* indirectArgs [[buffer(2)]], constant uint& experts [[buffer(3)]],
    constant uint& guGridX [[buffer(4)]], constant uint& dnGridX [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]], uint sgi [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sgSums[8], sgOffsets[8], chunkTotal;
    uint runningBase = 0u;
    for (uint base = 0u; base < experts; base += 256u) {
        const uint e = base + tid;
        const uint isAct = (e < experts && expertOff[e + 1] > expertOff[e]) ? 1u : 0u;
        const uint laneOff = simd_prefix_exclusive_sum(isAct);
        if (lane == 31u) sgSums[sgi] = laneOff + isAct;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgi == 0u && lane < 8u) {
            const uint v = sgSums[lane];
            const uint so = simd_prefix_exclusive_sum(v);
            sgOffsets[lane] = so;
            if (lane == 7u) chunkTotal = so + v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (isAct != 0u) activeExperts[runningBase + sgOffsets[sgi] + laneOff] = int32_t(e);
        runningBase += chunkTotal;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        indirectArgs[0] = runningBase;
        indirectArgs[4] = (runningBase > 0u) ? guGridX : 0u;
        indirectArgs[5] = runningBase;
        indirectArgs[6] = (runningBase > 0u) ? 1u : 0u;
        indirectArgs[8] = (runningBase > 0u) ? dnGridX : 0u;
        indirectArgs[9] = runningBase;
        indirectArgs[10] = (runningBase > 0u) ? 1u : 0u;
    }
}

kernel void qwg_moe_compact_nvfp4_gate_up_swiglu(
    device const char* x [[buffer(0)]], device const uint32_t* gateW [[buffer(1)]],
    device const uchar* gateSc [[buffer(2)]], device const float* gateS2 [[buffer(3)]],
    device const uint32_t* upW [[buffer(4)]], device const uchar* upSc [[buffer(5)]],
    device const float* upS2 [[buffer(6)]], device const int32_t* expertOff [[buffer(7)]],
    device const int32_t* activeExperts [[buffer(8)]], device const uint* activeArgs [[buffer(9)]],
    device const int32_t* sortedTok [[buffer(10)]], device const int32_t* sortedPairIdx [[buffer(11)]],
    device char* actOut [[buffer(12)]], constant QWGemmFPCompactMoEParams& p [[buffer(13)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half2 tgFP4[256];
    threadgroup float tgFP8[256];
    qwg_moe_compact_gate_up_body<false>(x, gateW, gateSc, gateS2, upW, upSc, upS2,
        expertOff, activeExperts, activeArgs, sortedTok, sortedPairIdx, actOut, p, tgFP4, tgFP8, tg, tid, simd_id, lane);
}

kernel void qwg_moe_compact_nvfp4_down(
    device const char* actIn [[buffer(0)]], device const uint32_t* downW [[buffer(1)]],
    device const uchar* downSc [[buffer(2)]], device const float* downS2 [[buffer(3)]],
    device const int32_t* expertOff [[buffer(4)]], device const int32_t* activeExperts [[buffer(5)]],
    device const uint* activeArgs [[buffer(6)]], device const int32_t* sortedPairIdx [[buffer(7)]],
    device char* slotOut [[buffer(8)]], constant QWGemmFPCompactMoEParams& p [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half2 tgFP4[256];
    threadgroup float tgFP8[256];
    qwg_moe_compact_down_body<false>(actIn, downW, downSc, downS2,
        expertOff, activeExperts, activeArgs, sortedPairIdx, slotOut, p, tgFP4, tgFP8, tg, tid, simd_id, lane);
}

kernel void qwg_moe_compact_mxfp8_gate_up_swiglu(
    device const char* x [[buffer(0)]], device const uint32_t* gateW [[buffer(1)]],
    device const uchar* gateSc [[buffer(2)]], device const uint32_t* upW [[buffer(3)]],
    device const uchar* upSc [[buffer(4)]], device const int32_t* expertOff [[buffer(5)]],
    device const int32_t* activeExperts [[buffer(6)]], device const uint* activeArgs [[buffer(7)]],
    device const int32_t* sortedTok [[buffer(8)]], device const int32_t* sortedPairIdx [[buffer(9)]],
    device char* actOut [[buffer(10)]], constant QWGemmFPCompactMoEParams& p [[buffer(11)]],
    uint2 tg [[threadgroup_position_in_grid]], uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    qwg_moe_compact_gate_up_body<true>(x, gateW, gateSc, nullptr, upW, upSc, nullptr,
        expertOff, activeExperts, activeArgs, sortedTok, sortedPairIdx, actOut, p, nullptr, nullptr, tg, 0u, simd_id, lane);
}

kernel void qwg_moe_compact_mxfp8_down(
    device const char* actIn [[buffer(0)]], device const uint32_t* downW [[buffer(1)]],
    device const uchar* downSc [[buffer(2)]], device const int32_t* expertOff [[buffer(3)]],
    device const int32_t* activeExperts [[buffer(4)]], device const uint* activeArgs [[buffer(5)]],
    device const int32_t* sortedPairIdx [[buffer(6)]], device char* slotOut [[buffer(7)]],
    constant QWGemmFPCompactMoEParams& p [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    qwg_moe_compact_down_body<true>(actIn, downW, downSc, nullptr,
        expertOff, activeExperts, activeArgs, sortedPairIdx, slotOut, p, nullptr, nullptr, tg, 0u, simd_id, lane);
}

kernel void qwg_moe_compact_nvfp4_gate_up_swiglu_t1(
    device const char* x [[buffer(0)]], device const uint32_t* gateW [[buffer(1)]],
    device const uchar* gateSc [[buffer(2)]], device const float* gateS2 [[buffer(3)]],
    device const uint32_t* upW [[buffer(4)]], device const uchar* upSc [[buffer(5)]],
    device const float* upS2 [[buffer(6)]], device const int32_t* activeExperts [[buffer(8)]],
    device char* actOut [[buffer(12)]], constant QWGemmFPCompactMoEParams& p [[buffer(13)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half2 tgFP4[256];
    threadgroup float tgFP8[256];
    qwg_moe_compact_gate_up_t1_body<false>(x, gateW, gateSc, gateS2, upW, upSc, upS2,
        activeExperts, actOut, p, tgFP4, tgFP8, tg, tid, simd_id, lane);
}

kernel void qwg_moe_compact_nvfp4_down_t1(
    device const char* actIn [[buffer(0)]], device const uint32_t* downW [[buffer(1)]],
    device const uchar* downSc [[buffer(2)]], device const float* downS2 [[buffer(3)]],
    device const int32_t* activeExperts [[buffer(5)]], device char* slotOut [[buffer(8)]],
    constant QWGemmFPCompactMoEParams& p [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half2 tgFP4[256];
    threadgroup float tgFP8[256];
    qwg_moe_compact_down_t1_body<false>(actIn, downW, downSc, downS2,
        activeExperts, slotOut, p, tgFP4, tgFP8, tg, tid, simd_id, lane);
}

kernel void qwg_moe_compact_mxfp8_gate_up_swiglu_t1(
    device const char* x [[buffer(0)]], device const uint32_t* gateW [[buffer(1)]],
    device const uchar* gateSc [[buffer(2)]], device const uint32_t* upW [[buffer(3)]],
    device const uchar* upSc [[buffer(4)]], device const int32_t* activeExperts [[buffer(6)]],
    device char* actOut [[buffer(10)]], constant QWGemmFPCompactMoEParams& p [[buffer(11)]],
    uint2 tg [[threadgroup_position_in_grid]], uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    qwg_moe_compact_gate_up_t1_body<true>(x, gateW, gateSc, nullptr, upW, upSc, nullptr,
        activeExperts, actOut, p, nullptr, nullptr, tg, 0u, simd_id, lane);
}

kernel void qwg_moe_compact_mxfp8_down_t1(
    device const char* actIn [[buffer(0)]], device const uint32_t* downW [[buffer(1)]],
    device const uchar* downSc [[buffer(2)]], device const int32_t* activeExperts [[buffer(4)]],
    device char* slotOut [[buffer(7)]], constant QWGemmFPCompactMoEParams& p [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    qwg_moe_compact_down_t1_body<true>(actIn, downW, downSc, nullptr,
        activeExperts, slotOut, p, nullptr, nullptr, tg, 0u, simd_id, lane);
}

kernel void qwg_swiglu_softcap_f16(
    device const char* gu [[buffer(0)]], device char* act [[buffer(1)]],
    constant QWGemmFPSwiGLUParams& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    const uint mi = gid.x * 8u, m = gid.y;
    if (m >= p.M || mi >= p.I) return;
    const uint inDt = (p.flags & 4u) != 0u ? 2u : ((p.flags & 1u) != 0u ? 1u : 0u);
    const uint outDt = (p.flags & 8u) != 0u ? 2u : ((p.flags & 2u) != 0u ? 1u : 0u);
    const bool rotate = (p.flags & 16u) != 0u;
    const size_t inRow = size_t(m) * (2u * p.I), outRow = size_t(m) * p.I;
    const size_t inAlign = (inDt == 2u) ? 15u : 7u;
    const size_t outAlign = (outDt == 2u) ? 15u : 7u;
    float ov[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (mi + 8u <= p.I && (p.I & 3u) == 0u && ((size_t(gu) & inAlign) == 0u)) {
        const float4 g0 = qwg_fp_read4(gu, inRow + mi, inDt), g1 = qwg_fp_read4(gu, inRow + mi + 4u, inDt);
        const float4 u0 = qwg_fp_read4(gu, inRow + p.I + mi, inDt), u1 = qwg_fp_read4(gu, inRow + p.I + mi + 4u, inDt);
        ov[0] = qwg_fp_softcap_swiglu(g0.x, u0.x, p.gateCap, p.upCap);
        ov[1] = qwg_fp_softcap_swiglu(g0.y, u0.y, p.gateCap, p.upCap);
        ov[2] = qwg_fp_softcap_swiglu(g0.z, u0.z, p.gateCap, p.upCap);
        ov[3] = qwg_fp_softcap_swiglu(g0.w, u0.w, p.gateCap, p.upCap);
        ov[4] = qwg_fp_softcap_swiglu(g1.x, u1.x, p.gateCap, p.upCap);
        ov[5] = qwg_fp_softcap_swiglu(g1.y, u1.y, p.gateCap, p.upCap);
        ov[6] = qwg_fp_softcap_swiglu(g1.z, u1.z, p.gateCap, p.upCap);
        ov[7] = qwg_fp_softcap_swiglu(g1.w, u1.w, p.gateCap, p.upCap);
    } else {
        for (uint k = 0u; k < 8u && mi + k < p.I; ++k) {
            const float g = qwg_fp_read(gu, inRow + mi + k, inDt);
            const float u = qwg_fp_read(gu, inRow + p.I + mi + k, inDt);
            ov[k] = qwg_fp_softcap_swiglu(g, u, p.gateCap, p.upCap);
        }
    }
#if !defined(QW_BASIS_STANDARD)
    if (rotate && (p.I & 31u) == 0u) {
        const uint lane4 = gid.x & 3u;
        for (int i = 0; i < 8; i += 2) {
            const float a = ov[i], b = ov[i + 1];
            ov[i] = a + b; ov[i + 1] = a - b;
        }
        for (int i = 0; i < 8; i += 4) {
            const float a0 = ov[i], a1 = ov[i + 1], b0 = ov[i + 2], b1 = ov[i + 3];
            ov[i] = a0 + b0; ov[i + 1] = a1 + b1;
            ov[i + 2] = a0 - b0; ov[i + 3] = a1 - b1;
        }
        for (int i = 0; i < 4; ++i) {
            const float a = ov[i], b = ov[i + 4];
            ov[i] = a + b; ov[i + 4] = a - b;
        }
        const bool neg1 = (lane4 & 1u) != 0u;
        for (int i = 0; i < 8; ++i) {
            const float pr = simd_shuffle_xor(ov[i], ushort(1));
            ov[i] = neg1 ? (pr - ov[i]) : (ov[i] + pr);
        }
        const bool neg2 = (lane4 & 2u) != 0u;
        constexpr float invSqrt32 = 0.17677669529663689f;
        for (int i = 0; i < 8; ++i) {
            const float pr = simd_shuffle_xor(ov[i], ushort(2));
            ov[i] = (neg2 ? (pr - ov[i]) : (ov[i] + pr)) * invSqrt32;
        }
    }
#else
    (void)rotate;
#endif
    if (mi + 8u <= p.I && (p.I & 3u) == 0u && ((size_t(act) & outAlign) == 0u)) {
        qwg_fp_write4(act, outRow + mi, float4(ov[0], ov[1], ov[2], ov[3]), outDt);
        qwg_fp_write4(act, outRow + mi + 4u, float4(ov[4], ov[5], ov[6], ov[7]), outDt);
    } else {
        for (uint k = 0u; k < 8u && mi + k < p.I; ++k) {
            qwg_fp_write(act, outRow + mi + k, ov[k], outDt);
        }
    }
}

kernel void qwg_moe_down_reduce_combine_bf16(
    device const char* dnAcc [[buffer(0)]], device const int32_t* invPerm [[buffer(1)]],
    device const float* weights [[buffer(2)]], device const char* shared [[buffer(3)]],
    device const float* gate [[buffer(4)]], device char* out [[buffer(5)]],
    constant QWGemmFPCombineParams& p [[buffer(6)]], device const int32_t* ids [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint h = (tg.x * 256u + tid) * 4u, t = tg.y;
    if (t >= p.tokens || h >= p.hidden) return;
    const bool idPerm = (p.flags & 1u) != 0u, skipSh = (p.flags & 2u) != 0u;
    const float g = (skipSh || (p.flags & 8u) == 0u) ? 1.0f : gate[t];
    const bool hasIds = (p.flags & 16u) != 0u;
    const uint basePair = t * p.topK;
    const bool vec4 = (h + 4u <= p.hidden) && ((p.hidden & 3u) == 0u)
        && ((size_t(dnAcc) & 15u) == 0u) && ((size_t(out) & 15u) == 0u)
        && (skipSh || ((size_t(shared) & 15u) == 0u));

    if (vec4) {
        float4 acc = float4(0.0f);
        for (uint k = 0u; k < p.topK; ++k) {
            if (hasIds && ids[basePair + k] < 0) continue;
            const float wk = weights[basePair + k];
            if (wk == 0.0f) continue;
            const int slot = idPerm ? int(basePair + k) : invPerm[basePair + k];
            if (slot < 0) continue;
            acc += wk * qwg_fp_read4(dnAcc, size_t(slot) * p.hidden + h, p.inDtype);
        }
        const size_t outIdx = size_t(t) * p.hidden + h;
        if (!skipSh) acc += g * qwg_fp_read4(shared, outIdx, p.sharedDtype);
        qwg_fp_write4(out, outIdx, acc, p.outDtype);
    } else {
        for (uint i = 0u; h + i < p.hidden; ++i) {
            float acc = 0.0f;
            for (uint k = 0u; k < p.topK; ++k) {
                if (hasIds && ids[basePair + k] < 0) continue;
                const float wk = weights[basePair + k];
                if (wk == 0.0f) continue;
                const int slot = idPerm ? int(basePair + k) : invPerm[basePair + k];
                if (slot < 0) continue;
                acc += wk * qwg_fp_read(dnAcc, size_t(slot) * p.hidden + h + i, p.inDtype);
            }
            const size_t outIdx = size_t(t) * p.hidden + h + i;
            if (!skipSh) acc += g * qwg_fp_read(shared, outIdx, p.sharedDtype);
            qwg_fp_write(out, outIdx, acc, p.outDtype);
        }
    }
}
