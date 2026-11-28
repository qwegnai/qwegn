// qwattn_hca_out_shvq4.metal — Isolated SHVQ4 DirectLUT HCA grouped wo_a NAX kernel
// (BM=16 K-split for T<=16 verify, BM=64 for T>16 prefill; 32 KiB LUT per Law 27).

#if !defined(QW_DISABLE_NAX)
#if !defined(__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__) || (__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ < 260200)
#error "Metal 4 NAX kernels require -mmacosx-version-min=26.2"
#endif

#define QW_SHVQ_ROUTED_DIRECT_LUT 1
#include <metal_stdlib>
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "../include/qw_dequant_shvq.metalh"
#include "../include/qw_weight_basis.metalh"
using namespace metal;
using namespace mlx::steel;

struct HCAGroupedWoAShvq4Params {
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

#define HCA_WOA_SHVQ4_OS 32u

METAL_FUNC void qw_hca_wo_a_shvq4_epilogue(
    float raw,
    device const half* qUpGate,
    device half* zGatedOut,
    constant HCAGroupedWoAShvq4Params& p,
    uint t,
    uint globalCol,
    bool skipFwht,
    uint lane
) {
    const float gate = float(qUpGate[(size_t)t * p.qUpStride + p.oGateOffset + globalCol]);
    const float gated = raw / (1.0f + exp(-gate));
    zGatedOut[(size_t)t * (p.oGroups * p.oLoraRank) + globalCol] =
        half(skipFwht ? gated : qwg_fwht32_lane_f32(gated, lane));
}

template <int BM>
METAL_FUNC void qw_hca_wo_a_nax_shvq4_step32(
    thread NAXTile<float, (BM == 16) ? 1 : 2, 2>& Dtile,
    device const half* aBase,
    thread const device char* const (&wRowPtr)[4],
    uint k0,
    int ldA,
    bool alignedM,
    int sgpSm,
    uint gShift,
    uint gMask,
    uint groupBytes,
    uint sliceBytes,
    uint q0,
    uint q1
) {
    constexpr short TM = (BM == 16) ? 1 : 2;
    NAXTile<half, TM, 2> Atile;
    NAXTile<half, 2, 2> Btile;
    if (alignedM) {
        Atile.load(aBase + k0, ldA);
    } else {
        Atile.load_safe(aBase + k0, ldA, short2(32, short(sgpSm)));
    }
    const uint gIdx = k0 >> gShift, inGroup = k0 & gMask;
    const uint payOff = (inGroup >> 5u) * sliceBytes;

    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < 2; ++r) {
        thread auto& f0 = Btile.frag_at(r, 0);
        thread auto& f1 = Btile.frag_at(r, 1);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; ++i) {
            device const char* gp = wRowPtr[r * 2 + i] + (size_t)gIdx * groupBytes;
            const half shvqScale = *(device const half*)gp;
            device const uchar* pay = (device const uchar*)(gp + 2u + payOff);
            const half4 vLo = qwg_shvq_unpack_quad4<4, false>(pay, q0) * shvqScale;
            const half4 vHi = qwg_shvq_unpack_quad4<4, false>(pay, q1) * shvqScale;
            for (short k = 0; k < 4; ++k) {
                f0[i * 4 + k] = vLo[k];
                f1[i * 4 + k] = vHi[k];
            }
        }
    }
    tile_matmad_nax(Dtile, Atile, metal::false_type{}, Btile, metal::true_type{});
}

template <int BM>
METAL_FUNC void qw_hca_wo_a_nax_shvq4_impl(
    device const half* zIn,
    device const char* woABlocks,
    device const half* qUpGate,
    device half* zGatedOut,
    constant HCAGroupedWoAShvq4Params& p,
    threadgroup float (&os)[4][32u * HCA_WOA_SHVQ4_OS],
    uint3 tg,
    uint sg,
    uint lane
) {
    const uint cBase = tg.x * 64u, tBase = tg.y * uint(BM), g = tg.z;
    if (g >= p.oGroups || tBase >= p.T || cBase >= p.oLoraRank) return;
    const uint wk = (BM == 16) ? (sg >> 1u) : 0u;
    const uint wm = (BM == 16) ? 0u : (sg >> 1u);
    const uint wn = sg & 1u;
    const uint warpT0 = tBase + wm * 32u, warpC0 = cBase + wn * 32u;
    const int sgpSm = min((BM == 16) ? 16 : 32, int(p.T) - int(warpT0));
    if (sgpSm <= 0 || warpC0 >= p.oLoraRank) {
        if constexpr (BM == 16) threadgroup_barrier(mem_flags::mem_threadgroup);
        return;
    }

    constexpr short TM = (BM == 16) ? 1 : 2;
    NAXTile<float, TM, 2> Dtile;
    Dtile.clear();

    const short2 sc = BaseNAXFrag::get_coord();
    const uint gShift = ctz(p.group), gMask = p.group - 1u;
    const uint nGroups = p.groupIn >> gShift;
    const uint groupBytes = qwg_shvq_group_bytes(4, int(p.group));
    constexpr uint sliceBytes = qwg_shvq_slice_bytes(4);
    const size_t rowBytes = (size_t)nGroups * groupBytes;

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
    const bool alignedM = (sgpSm == ((BM == 16) ? 16 : 32));
    const uint q0 = uint(sc.x) >> 2u, q1 = (uint(sc.x) + 16u) >> 2u;

    if constexpr (BM == 16) {
        STEEL_PRAGMA_NO_UNROLL
        for (uint k0 = wk * 32u; k0 < p.groupIn; k0 += 64u) {
            qw_hca_wo_a_nax_shvq4_step32<BM>(
                Dtile, aBase, wRowPtr, k0, ldA, alignedM, sgpSm,
                gShift, gMask, groupBytes, sliceBytes, q0, q1);
        }
    } else {
        STEEL_PRAGMA_NO_UNROLL
        for (uint k0 = 0u; k0 + 64u <= p.groupIn; k0 += 64u) {
            STEEL_PRAGMA_UNROLL
            for (uint dk = 0u; dk < 64u; dk += 32u) {
                qw_hca_wo_a_nax_shvq4_step32<BM>(
                    Dtile, aBase, wRowPtr, k0 + dk, ldA, alignedM, sgpSm,
                    gShift, gMask, groupBytes, sliceBytes, q0, q1);
            }
        }
        if ((p.groupIn & 32u) != 0u) {
            qw_hca_wo_a_nax_shvq4_step32<BM>(
                Dtile, aBase, wRowPtr, p.groupIn - 32u, ldA, alignedM, sgpSm,
                gShift, gMask, groupBytes, sliceBytes, q0, q1);
        }
    }

    threadgroup float* osWarp = &os[sg][0];
    Dtile.template store<float, HCA_WOA_SHVQ4_OS, 1>(osWarp);
    if constexpr (BM == 16) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (wk != 0u) return;
    } else {
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup const float* osPeer = (BM == 16) ? &os[sg + 2u][0] : nullptr;
    const bool skipFwht = (p.isIncumbent & 0x10000u) != 0u;
    const uint validRows = uint(sgpSm);
    const uint globalCol = g * p.oLoraRank + warpC0 + lane;
    for (uint m = 0u; m < validRows; ++m) {
        const float v = (BM == 16)
            ? (osWarp[m * HCA_WOA_SHVQ4_OS + lane] + osPeer[m * HCA_WOA_SHVQ4_OS + lane])
            : osWarp[m * HCA_WOA_SHVQ4_OS + lane];
        qw_hca_wo_a_shvq4_epilogue(v, qUpGate, zGatedOut, p, warpT0 + m, globalCol, skipFwht, lane);
    }
}

// 1. NAX SHVQ4 grouped wo_a BM=64 (T>16) and BM=16 (T<=16) isolated PSOs.
kernel void qwg_hca_grouped_wo_a_nax_shvq4(
    device const half* zIn [[buffer(0)]], device const char* woABlocks [[buffer(1)]],
    device const half* qUpGate [[buffer(2)]], device half* zGatedOut [[buffer(3)]],
    constant HCAGroupedWoAShvq4Params& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float os[4][32u * HCA_WOA_SHVQ4_OS];
    qw_hca_wo_a_nax_shvq4_impl<64>(zIn, woABlocks, qUpGate, zGatedOut, p, os, tg, sg, lane);
}

kernel void qwg_hca_grouped_wo_a_nax_t16_shvq4(
    device const half* zIn [[buffer(0)]], device const char* woABlocks [[buffer(1)]],
    device const half* qUpGate [[buffer(2)]], device half* zGatedOut [[buffer(3)]],
    constant HCAGroupedWoAShvq4Params& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float os[4][32u * HCA_WOA_SHVQ4_OS];
    qw_hca_wo_a_nax_shvq4_impl<16>(zIn, woABlocks, qUpGate, zGatedOut, p, os, tg, sg, lane);
}
#endif // !defined(QW_DISABLE_NAX)
