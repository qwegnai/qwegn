// qwattn_hca_wo_a_nax.metal — Apple10+ NAX grouped wo_a kernels.
// Isolates BM=64 (T>16, BK=64) from BM=16 (T<=16) K-split PSOs.

#if !defined(QW_DISABLE_NAX)
#if !defined(__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__) || (__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ < 260200)
#error "Metal 4 NAX kernels require -mmacosx-version-min=26.2"
#endif

#include <metal_stdlib>
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "../include/qw_mla_dequant.metalh"
using namespace metal;
using namespace mlx::steel;

struct HCAGroupedWoAParams {
    uint T;
    uint oGroups;
    uint oLoraRank;
    uint groupIn;
    uint qUpStride;
    uint oGateOffset;
    uint bits;
    uint group;
    uint isIncumbent;
};

#define HCA_WOA_NAX_OS 32u

template <int MODE, bool SUB2, short TM, bool ALIGNED_M>
METAL_FUNC void qw_hca_wo_a_nax_step32(
    thread NAXTile<float, TM, 2>& Dtile,
    device const half* aBase,
    thread const device char* const (&wRowPtr)[4],
    constant HCAGroupedWoAParams& p,
    threadgroup const ShvqMlaTgLut* tgLut,
    uint k0,
    int ldA,
    int sgpSm,
    uint gShift,
    uint gMask,
    uint groupBytes,
    uint sliceBytes,
    short2 sc,
    ushort wSub,
    ushort wShift
) {
    NAXTile<half, TM, 2> Atile;
    NAXTile<half, 2, 2> Btile;
    if constexpr (ALIGNED_M) {
        Atile.load(aBase + k0, ldA);
    } else {
        Atile.load_safe(aBase + k0, ldA, short2(32, short(sgpSm)));
    }
    const uint gIdx = k0 >> gShift, inGroup = k0 & gMask;
    const uint payOff = (MODE == 2) ? ((inGroup >> 5u) * sliceBytes) : 0u;
    const uint q0 = uint(sc.x) >> 2u, q1 = (uint(sc.x) + 16u) >> 2u;

    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < 2; ++r) {
        thread auto& f0 = Btile.frag_at(r, 0);
        thread auto& f1 = Btile.frag_at(r, 1);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; ++i) {
            device const char* rPtr = wRowPtr[r * 2 + i];
            half4 vLo, vHi;
            if constexpr (MODE == 3) {
                device const half* hPtr = (device const half*)rPtr + k0 + uint(sc.x);
                vLo = *(device const half4*)hPtr;
                vHi = *(device const half4*)(hPtr + 16u);
            } else if constexpr (MODE == 1) {
                const device Block_HQ4_G32& blk = ((const device Block_HQ4_G32*)rPtr)[k0 >> 5u];
                const half2 dm = *(const device half2*)(&blk), s2 = half2(dm.x), b2 = half2(dm.y);
                const ushort2 raw0 = as_type<ushort2>(blk.qs[wSub]) >> wShift;
                const ushort2 raw1 = as_type<ushort2>(blk.qs[2u + wSub]) >> wShift;
                const half2 w01 = fma(as_type<half2>((raw0 & ushort2(0x000Fu)) | ushort2(0x6400u)), s2, b2);
                const half2 w23 = fma(as_type<half2>(((raw0 >> 4u) & ushort2(0x000Fu)) | ushort2(0x6400u)), s2, b2);
                const half2 w45 = fma(as_type<half2>((raw1 & ushort2(0x000Fu)) | ushort2(0x6400u)), s2, b2);
                const half2 w67 = fma(as_type<half2>(((raw1 >> 4u) & ushort2(0x000Fu)) | ushort2(0x6400u)), s2, b2);
                vLo = half4(w01.x, w01.y, w23.x, w23.y);
                vHi = half4(w45.x, w45.y, w67.x, w67.y);
            } else if constexpr (MODE == 2) {
                device const char* gp = rPtr + (size_t)gIdx * groupBytes;
                const half shvqScale = *(device const half*)gp;
                device const uchar* pay = (device const uchar*)(gp + 2u + payOff);
                vLo = qw_mla_shvq_quad4<SUB2>(pay, q0, p.bits, *tgLut) * shvqScale;
                vHi = qw_mla_shvq_quad4<SUB2>(pay, q1, p.bits, *tgLut) * shvqScale;
            } else if constexpr (MODE == 8) {
                device const char* gp = rPtr + (size_t)gIdx * groupBytes;
                const half2 sb = *(device const half2*)gp;
                device const uchar* pay = (device const uchar*)(gp + 4u) + inGroup + uint(sc.x);
                const uchar4 uLo = as_type<uchar4>(*(device const uint*)pay);
                const uchar4 uHi = as_type<uchar4>(*(device const uint*)(pay + 16u));
                const half4 s4 = half4(sb.x), b4 = half4(sb.y);
                vLo = fma(half4(uLo), s4, b4);
                vHi = fma(half4(uHi), s4, b4);
            } else if constexpr (MODE == 4) {
                device const char* gp = rPtr + (size_t)gIdx * groupBytes;
                const half2 sb = *(device const half2*)gp, s2 = half2(sb.x), b2 = half2(sb.y), kBias = half2(-1024.0h);
                device const uchar* pay = (device const uchar*)(gp + 4u);
                const uint byteOff = (inGroup + uint(sc.x)) >> 1u;
                const ushort2 raw = ushort2(*(device const ushort*)(pay + byteOff), *(device const ushort*)(pay + byteOff + 8u));
                const half2 w0 = fma(as_type<half2>((raw & ushort2(0x000Fu)) | ushort2(0x6400u)) + kBias, s2, b2);
                const half2 w1 = fma(as_type<half2>(((raw >> 4u) & ushort2(0x000Fu)) | ushort2(0x6400u)) + kBias, s2, b2);
                const half2 w2 = fma(as_type<half2>(((raw >> 8u) & ushort2(0x000Fu)) | ushort2(0x6400u)) + kBias, s2, b2);
                const half2 w3 = fma(as_type<half2>((raw >> 12u) | ushort2(0x6400u)) + kBias, s2, b2);
                vLo = half4(w0.x, w1.x, w2.x, w3.x);
                vHi = half4(w0.y, w1.y, w2.y, w3.y);
            } else {
                device const char* gp = rPtr + (size_t)gIdx * groupBytes;
                const float scAQ = float(*(device const half*)gp), biAQ = float(*(device const half*)(gp + 2u));
                device const uchar* pay = (device const uchar*)(gp + 4u);
                const int bitsI = int(p.bits), b0 = int(inGroup + uint(sc.x));
                for (short k = 0; k < 4; ++k) {
                    vLo[k] = half(fma(float(qwg_aq_read_bits(pay, bitsI, b0 + k)), scAQ, biAQ));
                    vHi[k] = half(fma(float(qwg_aq_read_bits(pay, bitsI, b0 + 16 + k)), scAQ, biAQ));
                }
            }
            for (short k = 0; k < 4; ++k) { f0[i * 4 + k] = vLo[k]; f1[i * 4 + k] = vHi[k]; }
        }
    }
    tile_matmad_nax(Dtile, Atile, metal::false_type{}, Btile, metal::true_type{});
}

template <int MODE, bool SUB2, int BM, bool ALIGNED_M>
METAL_FUNC void qw_hca_wo_a_nax_kloop(
    thread NAXTile<float, (BM == 16) ? 1 : 2, 2>& Dtile,
    device const half* aBase,
    thread const device char* const (&wRowPtr)[4],
    constant HCAGroupedWoAParams& p,
    threadgroup const ShvqMlaTgLut* tgLut,
    int ldA,
    int sgpSm,
    uint wk,
    uint gShift,
    uint gMask,
    uint groupBytes,
    uint sliceBytes,
    short2 sc,
    ushort wSub,
    ushort wShift
) {
    constexpr short TM = (BM == 16) ? 1 : 2;
    if constexpr (BM == 16) {
        STEEL_PRAGMA_NO_UNROLL
        for (uint k0 = wk * 32u; k0 < p.groupIn; k0 += 64u) {
            qw_hca_wo_a_nax_step32<MODE, SUB2, TM, ALIGNED_M>(
                Dtile, aBase, wRowPtr, p, tgLut, k0, ldA, sgpSm,
                gShift, gMask, groupBytes, sliceBytes, sc, wSub, wShift);
        }
    } else {
        STEEL_PRAGMA_NO_UNROLL
        for (uint k0 = 0u; k0 < p.groupIn; k0 += 32u) {
            qw_hca_wo_a_nax_step32<MODE, SUB2, TM, ALIGNED_M>(
                Dtile, aBase, wRowPtr, p, tgLut, k0, ldA, sgpSm,
                gShift, gMask, groupBytes, sliceBytes, sc, wSub, wShift);
        }
    }
}

template <int MODE, bool SUB2 = false, int BM = 64, bool USE_SMALL_OS = false>
METAL_FUNC void qw_hca_wo_a_nax_run(
    device const half* zIn, device const char* woABlocks,
    device const half* qUpGate, device half* zGatedOut,
    constant HCAGroupedWoAParams& p, threadgroup float* osBase,
    threadgroup const ShvqMlaTgLut* tgLut, uint g,
    uint warpT0, uint warpC0, int sgpSm, uint wk, uint sg, uint lane
) {
    constexpr short TM = (BM == 16) ? 1 : 2;
    NAXTile<float, TM, 2> Dtile;
    Dtile.clear();

    const short2 sc = BaseNAXFrag::get_coord();
    const uint gShift = ctz(p.group), gMask = p.group - 1u, nGroups = p.groupIn >> gShift;
    size_t rowBytes = 0;
    uint groupBytes = 0, sliceBytes = 0;
    if constexpr (MODE == 3) {
        rowBytes = (size_t)p.groupIn * sizeof(half);
    } else if constexpr (MODE == 1) {
        rowBytes = (size_t)(p.groupIn >> 5u) * sizeof(Block_HQ4_G32);
    } else if constexpr (MODE == 2) {
        groupBytes = qwg_shvq_group_bytes(int(p.bits), int(p.group));
        sliceBytes = qwg_shvq_slice_bytes(int(p.bits));
        rowBytes = (size_t)nGroups * groupBytes;
    } else {
        groupBytes = 4u + ((p.group * p.bits) >> 3u);
        rowBytes = (size_t)nGroups * groupBytes;
    }

    device const char* wRowPtr[4];
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < 2; ++r) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; ++i) {
            const uint wRow = g * p.oLoraRank + warpC0 + uint(r * 16 + int(sc.y) + i * 8);
            wRowPtr[r * 2 + i] = woABlocks + (size_t)wRow * rowBytes;
        }
    }

    const int ldA = int(p.oGroups * p.groupIn);
    device const half* aBase = zIn + ((size_t)warpT0 * p.oGroups + g) * p.groupIn;
    const ushort wSub = ushort(sc.x >> 3), wShift = ushort((sc.x & 4) << 1);
    const bool alignedM = (sgpSm == ((BM == 16) ? 16 : 32));
    if (alignedM) {
        qw_hca_wo_a_nax_kloop<MODE, SUB2, BM, true>(
            Dtile, aBase, wRowPtr, p, tgLut, ldA, sgpSm, wk,
            gShift, gMask, groupBytes, sliceBytes, sc, wSub, wShift);
    } else {
        qw_hca_wo_a_nax_kloop<MODE, SUB2, BM, false>(
            Dtile, aBase, wRowPtr, p, tgLut, ldA, sgpSm, wk,
            gShift, gMask, groupBytes, sliceBytes, sc, wSub, wShift);
    }

    if constexpr (MODE == 2) threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool skipFwht = (p.isIncumbent & 0x10000u) != 0u;
    const uint outStride = p.oGroups * p.oLoraRank;
    const uint globalCol = g * p.oLoraRank + warpC0 + lane;
    if constexpr (BM == 64 && USE_SMALL_OS) {
        threadgroup float* osWarp = osBase + sg * (16u * HCA_WOA_NAX_OS);
        const_for_loop<0, 2, 1>([&](auto rr) {
            constexpr short r = rr.value;
            BaseNAXFrag::store<float>(
                Dtile.template frag_at<r, 0>(), osWarp, Int<HCA_WOA_NAX_OS>{}, Int<1>{}, Int<0>{}, Int<0>{});
            BaseNAXFrag::store<float>(
                Dtile.template frag_at<r, 1>(), osWarp, Int<HCA_WOA_NAX_OS>{}, Int<1>{}, Int<0>{}, Int<16>{});
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const uint rowStart = uint(r) * 16u;
            const uint rowEnd = (uint(sgpSm) > rowStart) ? min(16u, uint(sgpSm) - rowStart) : 0u;
            for (uint m = 0u; m < rowEnd; ++m) {
                const float raw = osWarp[m * HCA_WOA_NAX_OS + lane];
                const float gate = float(qUpGate[(size_t)(warpT0 + rowStart + m) * p.qUpStride + p.oGateOffset + globalCol]);
                const float gated = raw / (1.0f + exp(-gate));
                zGatedOut[(size_t)(warpT0 + rowStart + m) * outStride + globalCol] =
                    half(skipFwht ? gated : qwg_fwht32_lane_f32(gated, lane));
            }
            if (r == 0) simdgroup_barrier(mem_flags::mem_threadgroup);
        });
    } else {
        threadgroup float* osWarp = osBase + sg * (32u * HCA_WOA_NAX_OS);
        Dtile.template store<float, HCA_WOA_NAX_OS, 1>(osWarp);
        if constexpr (BM == 16) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (wk != 0u) return;
        } else {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup const float* osPeer = (BM == 16) ? (osBase + (sg + 2u) * (32u * HCA_WOA_NAX_OS)) : nullptr;
        const uint validRows = uint(sgpSm);
        for (uint m = 0u; m < validRows; ++m) {
            const float raw = (BM == 16) ? (osWarp[m * HCA_WOA_NAX_OS + lane] + osPeer[m * HCA_WOA_NAX_OS + lane])
                                         : osWarp[m * HCA_WOA_NAX_OS + lane];
            const float gate = float(qUpGate[(size_t)(warpT0 + m) * p.qUpStride + p.oGateOffset + globalCol]);
            const float gated = raw / (1.0f + exp(-gate));
            zGatedOut[(size_t)(warpT0 + m) * outStride + globalCol] =
                half(skipFwht ? gated : qwg_fwht32_lane_f32(gated, lane));
        }
    }
}

union HCAWoANaxTg {
    ShvqMlaTgLut tgLut;
    float os[4][32u * HCA_WOA_NAX_OS];
};

template <int BM, bool SUB2>
METAL_FUNC void qw_hca_grouped_wo_a_nax_body(
    device const half* zIn, device const char* woABlocks,
    device const half* qUpGate, device half* zGatedOut,
    constant HCAGroupedWoAParams& p, threadgroup HCAWoANaxTg& st,
    uint3 tg, uint tid, uint sg, uint lane
) {
    const uint cBase = tg.x * 64u, tBase = tg.y * uint(BM), g = tg.z;
    if (g >= p.oGroups || tBase >= p.T || cBase >= p.oLoraRank) return;

    const uint inc = p.isIncumbent & 0xFFu;
    if (inc == 2u && qw_mla_load_shvq_lut<SUB2>(st.tgLut, p.bits, tid, 128u)) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint wn = sg & 1u, warpC0 = cBase + wn * 32u;
    const uint wk = (BM == 16) ? (sg >> 1u) : 0u;
    const uint warpT0 = (BM == 16) ? tBase : (tBase + (sg >> 1u) * 32u);
    const int sgpSm = min((BM == 16) ? 16 : 32, int(p.T) - int(warpT0));
    if (inc == 2u) {
        if (sgpSm <= 0 || warpC0 >= p.oLoraRank) {
            if constexpr (BM == 16) threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            return;
        }
        qw_hca_wo_a_nax_run<2, SUB2, BM>(zIn, woABlocks, qUpGate, zGatedOut, p, &st.os[0][0], &st.tgLut, g, warpT0, warpC0, sgpSm, wk, sg, lane);
        return;
    }
    if (sgpSm <= 0 || warpC0 >= p.oLoraRank) {
        if constexpr (BM == 16) threadgroup_barrier(mem_flags::mem_threadgroup);
        return;
    }
    if (inc == 3u) qw_hca_wo_a_nax_run<3, SUB2, BM>(zIn, woABlocks, qUpGate, zGatedOut, p, &st.os[0][0], &st.tgLut, g, warpT0, warpC0, sgpSm, wk, sg, lane);
    else if (inc == 1u) qw_hca_wo_a_nax_run<1, SUB2, BM>(zIn, woABlocks, qUpGate, zGatedOut, p, &st.os[0][0], &st.tgLut, g, warpT0, warpC0, sgpSm, wk, sg, lane);
    else if (p.bits == 8u) qw_hca_wo_a_nax_run<8, SUB2, BM>(zIn, woABlocks, qUpGate, zGatedOut, p, &st.os[0][0], &st.tgLut, g, warpT0, warpC0, sgpSm, wk, sg, lane);
    else if (p.bits == 4u) qw_hca_wo_a_nax_run<4, SUB2, BM>(zIn, woABlocks, qUpGate, zGatedOut, p, &st.os[0][0], &st.tgLut, g, warpT0, warpC0, sgpSm, wk, sg, lane);
    else qw_hca_wo_a_nax_run<0, SUB2, BM>(zIn, woABlocks, qUpGate, zGatedOut, p, &st.os[0][0], &st.tgLut, g, warpT0, warpC0, sgpSm, wk, sg, lane);
}

#define QW_HCA_WOA_NAX_KERNEL(NAME, BM, SUB2) \
kernel void NAME( \
    device const half* zIn [[buffer(0)]], device const char* woABlocks [[buffer(1)]], \
    device const half* qUpGate [[buffer(2)]], device half* zGatedOut [[buffer(3)]], \
    constant HCAGroupedWoAParams& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]] \
) { \
    threadgroup HCAWoANaxTg st; \
    qw_hca_grouped_wo_a_nax_body<BM, SUB2>(zIn, woABlocks, qUpGate, zGatedOut, p, st, tg, tid, sg, lane); \
}

// Specialized zero-LUT, 8 KiB os BM=64 PSOs (4 TGs/core on AGX) for non-SHVQ containers.
#define QW_HCA_WOA_NAX_MODE_KERNEL(NAME, MODE) \
kernel void NAME( \
    device const half* zIn [[buffer(0)]], device const char* woABlocks [[buffer(1)]], \
    device const half* qUpGate [[buffer(2)]], device half* zGatedOut [[buffer(3)]], \
    constant HCAGroupedWoAParams& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]] \
) { \
    const uint cBase = tg.x * 64u, tBase = tg.y * 64u, g = tg.z; \
    const uint warpT0 = tBase + (sg >> 1u) * 32u, warpC0 = cBase + (sg & 1u) * 32u; \
    const int sgpSm = min(32, int(p.T) - int(warpT0)); \
    if (g >= p.oGroups || sgpSm <= 0 || warpC0 >= p.oLoraRank) return; \
    threadgroup float os[4][16u * HCA_WOA_NAX_OS]; \
    qw_hca_wo_a_nax_run<MODE, false, 64, true>(zIn, woABlocks, qUpGate, zGatedOut, p, &os[0][0], nullptr, g, warpT0, warpC0, sgpSm, 0u, sg, lane); \
}

// Prefill BM=64 (grid height (T + 63) / 64) and T<=16 BM=16 (grid height (T + 15) / 16).
QW_HCA_WOA_NAX_KERNEL(qwg_hca_grouped_wo_a_nax, 64, false)
QW_HCA_WOA_NAX_KERNEL(qwg_hca_grouped_wo_a_nax_sub2, 64, true)
QW_HCA_WOA_NAX_KERNEL(qwg_hca_grouped_wo_a_nax_t16, 16, false)
QW_HCA_WOA_NAX_KERNEL(qwg_hca_grouped_wo_a_nax_t16_sub2, 16, true)
QW_HCA_WOA_NAX_MODE_KERNEL(qwg_hca_grouped_wo_a_nax_hq4, 1)
QW_HCA_WOA_NAX_MODE_KERNEL(qwg_hca_grouped_wo_a_nax_aq8, 8)
QW_HCA_WOA_NAX_MODE_KERNEL(qwg_hca_grouped_wo_a_nax_aq4, 4)
#endif // !defined(QW_DISABLE_NAX)
