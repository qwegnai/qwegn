// qwattn_hca_out.metal — HCA output epilogue GPU kernels:
// fused 8x8 MMA VHA head mixing + GPT-J inverse YaRN RoPE, and 3D-batched
// grouped low-rank wo_a projection (GEMV decode/verify + 8x8 MMA prefill)
// with fused sigmoid(o_gate) and H_32 exit rotation (<= 6 KiB constant LUTs).

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "../include/qw_mla_dequant.metalh"
using namespace metal;

#if !defined(QW_DISABLE_NAX)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
using namespace mlx::steel;

#if !defined(__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__) || (__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ < 260200)
#error "Metal 4 NAX kernels require -mmacosx-version-min=26.2 (SDK MPPTensorOpsAvailability)"
#endif
#endif // !defined(QW_DISABLE_NAX)

struct HCAVhaInvRopeParams {
    uint T;
    uint pos0;
    uint qHeads;
    uint headDim;
    uint nopeDim;
    uint ropeDim;
    float invMscale;
    uint hasVHA;
};

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

// 5. Fused 8x8 MMA VHA head mixing + GPT-J Inverse YaRN RoPE + optional H_32 -> zVhaInv.
// Grid: (headDim / 32, T, 1), Threadgroup: (256, 1, 1).
kernel void qwg_hca_vha_invrope(
    device const half*         attIn   [[buffer(0)]],
    device const half*         vhaMix  [[buffer(1)]],
    device const float*        invFreq [[buffer(2)]],
    device half*               zOut    [[buffer(3)]],
    constant HCAVhaInvRopeParams& p    [[buffer(4)]],
    uint2                      tg      [[threadgroup_position_in_grid]],
    uint                       tid     [[thread_index_in_threadgroup]],
    uint                       sg      [[simdgroup_index_in_threadgroup]],
    uint                       lane    [[thread_index_in_simdgroup]]
) {
    const uint dBase = tg.x * 32u;
    const uint t = tg.y;
    if (t >= p.T || dBase >= p.headDim) return;

    constexpr uint YS = 40u;
    constexpr uint OS = 36u;
    threadgroup half  shY[64u * YS];
    threadgroup float shOut[64u * OS];

    const uint pos = p.pos0 + t;
    const uint rowStride = p.qHeads * p.headDim;
    device const half* yRow = attIn + (size_t)t * rowStride;
    device half* dstRow = zOut + (size_t)t * rowStride;
    const bool isRopeSlice = (dBase >= p.nopeDim);
    const bool hasVHA = (p.hasVHA & 1u) != 0u;
    const bool skipFwht = (p.hasVHA & 0x10000u) != 0u;

    if (hasVHA && (p.qHeads & 7u) == 0u && p.qHeads <= 64u) {
        for (uint i = tid; i < p.qHeads * 8u; i += 256u) {
            const uint g = i >> 3u, c4 = (i & 7u) * 4u;
            *(threadgroup half4*)(shY + g * YS + c4) = *(device const half4*)(yRow + g * p.headDim + dBase + c4);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint hBase = sg * 8u;
        if (hBase < p.qHeads) {
            simdgroup_float8x8 acc[4] = {
                simdgroup_float8x8(0.0f), simdgroup_float8x8(0.0f),
                simdgroup_float8x8(0.0f), simdgroup_float8x8(0.0f)
            };
            for (uint g0 = 0u; g0 < p.qHeads; g0 += 8u) {
                simdgroup_half8x8 mf;
                simdgroup_load(mf, vhaMix + hBase * p.qHeads + g0, p.qHeads);
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j) {
                    simdgroup_half8x8 yf;
                    simdgroup_load(yf, shY + g0 * YS + j * 8u, YS);
                    simdgroup_multiply_accumulate(acc[j], mf, yf, acc[j]);
                }
            }
            threadgroup float* sDst = shOut + hBase * OS;
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) simdgroup_store(acc[j], sDst + j * 8u, OS);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint hr = 0u; hr < 8u; ++hr) {
                const uint h = hBase + hr;
                float val = sDst[hr * OS + lane];
                if (isRopeSlice) {
                    const uint pairIdx = ((dBase - p.nopeDim) + lane) >> 1u;
                    const float ang = float(pos) * invFreq[pairIdx];
                    const float c = cos(ang) * p.invMscale, s = sin(ang) * p.invMscale;
                    const float mate = simd_shuffle_xor(val, ushort(1));
                    val = ((lane & 1u) == 0u) ? (val * c + mate * s) : (val * c - mate * s);
                }
                dstRow[h * p.headDim + dBase + lane] = half(skipFwht ? val : qwg_fwht32_lane_f32(val, lane));
            }
        }
        return;
    }

    for (uint h = sg; h < p.qHeads; h += 8u) {
        float val = 0.0f;
        if (hasVHA) {
            for (uint g = 0u; g < p.qHeads; ++g) {
                val = fma(float(vhaMix[h * p.qHeads + g]), float(yRow[g * p.headDim + dBase + lane]), val);
            }
        } else {
            val = float(yRow[h * p.headDim + dBase + lane]);
        }
        if (isRopeSlice) {
            const uint pairIdx = ((dBase - p.nopeDim) + lane) >> 1u;
            const float ang = float(pos) * invFreq[pairIdx];
            const float c = cos(ang) * p.invMscale, s = sin(ang) * p.invMscale;
            const float mate = simd_shuffle_xor(val, ushort(1));
            val = ((lane & 1u) == 0u) ? (val * c + mate * s) : (val * c - mate * s);
        }
        dstRow[h * p.headDim + dBase + lane] = half(skipFwht ? val : qwg_fwht32_lane_f32(val, lane));
    }
}

METAL_FUNC void qw_hca_nax_fwht32_pair(thread float (&e)[4], thread float (&f)[4], uint lane) {
    float t0 = e[0] + e[1], t1 = e[0] - e[1], t2 = e[2] + e[3], t3 = e[2] - e[3];
    e[0] = t0 + t2; e[1] = t1 + t3; e[2] = t0 - t2; e[3] = t1 - t3;
    float u0 = f[0] + f[1], u1 = f[0] - f[1], u2 = f[2] + f[3], u3 = f[2] - f[3];
    f[0] = u0 + u2; f[1] = u1 + u3; f[2] = u0 - u2; f[3] = u1 - u3;
    for (short k = 0; k < 4; ++k) { const float s = e[k] + f[k]; f[k] = e[k] - f[k]; e[k] = s; }
    const float s4 = ((lane & 1u) != 0u) ? -1.0f : 1.0f;
    const float s8 = ((lane & 8u) != 0u) ? -1.0f : 1.0f;
    for (short k = 0; k < 4; ++k) {
        e[k] = s4 * e[k] + simd_shuffle_xor(e[k], ushort(1));
        f[k] = s4 * f[k] + simd_shuffle_xor(f[k], ushort(1));
    }
    for (short k = 0; k < 4; ++k) {
        e[k] = s8 * e[k] + simd_shuffle_xor(e[k], ushort(8));
        f[k] = s8 * f[k] + simd_shuffle_xor(f[k], ushort(8));
    }
    constexpr float invSqrt32 = 0.1767766952966369f;
    for (short k = 0; k < 4; ++k) { e[k] *= invSqrt32; f[k] *= invSqrt32; }
}

#if !defined(QW_DISABLE_NAX)
// 5b. Apple10+ NAX VHA head mixing + GPT-J Inverse YaRN RoPE + optional H_32 (BLOCK=128).
// Grid: ((headDim + 127) / 128, T, 1), Threadgroup: (256, 1, 1).
kernel void qwg_hca_vha_invrope_nax(
    device const half*         attIn   [[buffer(0)]],
    device const half*         vhaMix  [[buffer(1)]],
    device const float*        invFreq [[buffer(2)]],
    device half*               zOut    [[buffer(3)]],
    constant HCAVhaInvRopeParams& p    [[buffer(4)]],
    uint2                      tg      [[threadgroup_position_in_grid]],
    uint                       sg      [[simdgroup_index_in_threadgroup]],
    uint                       lane    [[thread_index_in_simdgroup]]
) {
    const uint dBase = tg.x * 128u, t = tg.y;
    if (t >= p.T || dBase >= p.headDim) return;
    const uint warpH0 = (sg >> 2u) * 32u, warpD0 = dBase + (sg & 3u) * 32u;
    if (warpH0 >= p.qHeads || warpD0 >= p.headDim) return;

    const short2 sc = BaseNAXFrag::get_coord();
    const uint d0 = warpD0 + uint(sc.x), d1 = d0 + 16u, rowStride = p.qHeads * p.headDim;
    device const half* yRow = attIn + (size_t)t * rowStride;
    device half* zRow = zOut + (size_t)t * rowStride;
    const bool hasVHA = (p.hasVHA & 1u) != 0u, skipFwht = (p.hasVHA & 0x10000u) != 0u;

    NAXTile<float, 2, 2> Dtile;
    Dtile.clear();
    if (hasVHA) {
        const int sgpH = min(32, int(p.qHeads) - int(warpH0));
        const int sgpD = min(32, int(p.headDim) - int(warpD0));
        const bool full = (sgpH == 32 && sgpD == 32);
        for (uint k0 = 0u; k0 < p.qHeads; k0 += 32u) {
            NAXTile<half, 2, 2> Atile, Btile;
            const uint kValid = min(32u, p.qHeads - k0);
            if (full && kValid == 32u) {
                Atile.load(vhaMix + warpH0 * p.qHeads + k0, int(p.qHeads));
                Btile.load(yRow + (size_t)k0 * p.headDim + warpD0, int(p.headDim));
            } else {
                Atile.load_safe(vhaMix + warpH0 * p.qHeads + k0, int(p.qHeads), short2(short(kValid), short(sgpH)));
                Btile.load_safe(yRow + (size_t)k0 * p.headDim + warpD0, int(p.headDim), short2(short(sgpD), short(kValid)));
            }
            tile_matmad_nax(Dtile, Atile, metal::false_type{}, Btile, metal::false_type{});
        }
    }

    const float posF = float(p.pos0 + t);
    const uint ropeEnd = p.nopeDim + p.ropeDim;
    const bool r0a = (d0 >= p.nopeDim && d0 + 1u < ropeEnd), r0b = (d0 + 2u >= p.nopeDim && d0 + 3u < ropeEnd);
    const bool r1a = (d1 >= p.nopeDim && d1 + 1u < ropeEnd), r1b = (d1 + 2u >= p.nopeDim && d1 + 3u < ropeEnd);
    float c0a = 1.0f, s0a = 0.0f, c0b = 1.0f, s0b = 0.0f, c1a = 1.0f, s1a = 0.0f, c1b = 1.0f, s1b = 0.0f;
    if (r0a) { const float a = posF * invFreq[(d0 - p.nopeDim) >> 1u]; c0a = cos(a) * p.invMscale; s0a = sin(a) * p.invMscale; }
    if (r0b) { const float a = posF * invFreq[(d0 + 2u - p.nopeDim) >> 1u]; c0b = cos(a) * p.invMscale; s0b = sin(a) * p.invMscale; }
    if (r1a) { const float a = posF * invFreq[(d1 - p.nopeDim) >> 1u]; c1a = cos(a) * p.invMscale; s1a = sin(a) * p.invMscale; }
    if (r1b) { const float a = posF * invFreq[(d1 + 2u - p.nopeDim) >> 1u]; c1b = cos(a) * p.invMscale; s1b = sin(a) * p.invMscale; }

    STEEL_PRAGMA_UNROLL
    for (short m = 0; m < 2; ++m) {
        const thread auto& df0 = Dtile.frag_at(m, 0);
        const thread auto& df1 = Dtile.frag_at(m, 1);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; ++i) {
            const uint h = warpH0 + uint(m * 16 + int(sc.y) + i * 8);
            const bool validH = (h < p.qHeads);
            float e[4], f[4];
            if (hasVHA) {
                for (short k = 0; k < 4; ++k) { e[k] = df0[i * 4 + k]; f[k] = df1[i * 4 + k]; }
            } else {
                const half4 h0 = (validH && d0 + 3u < p.headDim) ? *(device const half4*)(yRow + h * p.headDim + d0) : half4(0.0h);
                const half4 h1 = (validH && d1 + 3u < p.headDim) ? *(device const half4*)(yRow + h * p.headDim + d1) : half4(0.0h);
                for (short k = 0; k < 4; ++k) { e[k] = float(h0[k]); f[k] = float(h1[k]); }
            }
            if (r0a) { const float a = e[0], b = e[1]; e[0] = a * c0a + b * s0a; e[1] = b * c0a - a * s0a; }
            if (r0b) { const float a = e[2], b = e[3]; e[2] = a * c0b + b * s0b; e[3] = b * c0b - a * s0b; }
            if (r1a) { const float a = f[0], b = f[1]; f[0] = a * c1a + b * s1a; f[1] = b * c1a - a * s1a; }
            if (r1b) { const float a = f[2], b = f[3]; f[2] = a * c1b + b * s1b; f[3] = b * c1b - a * s1b; }
            if (!skipFwht) qw_hca_nax_fwht32_pair(e, f, lane);
            if (validH && d0 + 3u < p.headDim) *(device half4*)(zRow + h * p.headDim + d0) = half4(half(e[0]), half(e[1]), half(e[2]), half(e[3]));
            if (validH && d1 + 3u < p.headDim) *(device half4*)(zRow + h * p.headDim + d1) = half4(half(f[0]), half(f[1]), half(f[2]), half(f[3]));
        }
    }
}
#endif // !defined(QW_DISABLE_NAX)

template <bool SUB2 = false>
METAL_FUNC void qw_hca_unpack_wo_a_32(device const char* woABlocks,
                                      uint row, uint col0,
                                      constant HCAGroupedWoAParams& p,
                                      threadgroup const ShvqMlaTgLut& tgLut,
                                      threadgroup half* dst, uint dstStride) {
    const uint inc = p.isIncumbent & 0xFFu;
    if (inc == 3u) {
        device const half* hSrc = (device const half*)woABlocks + (size_t)row * p.groupIn + col0;
        #pragma unroll
        for (uint i = 0u; i < 32u; ++i) dst[i * dstStride] = hSrc[i];
        return;
    }
    MLAAbsorbParams mp = {p.T, p.oGroups, 0u, p.oLoraRank, p.groupIn, 0u, p.bits, p.group, inc};
    qw_mla_unpack_slice32<SUB2>(woABlocks, row, col0, mp, tgLut, dst, dstStride);
}

template <bool SUB2 = false>
METAL_FUNC void qw_hca_unpack_wo_a_reg32(device const char* woABlocks,
                                         uint row, uint col0,
                                         constant HCAGroupedWoAParams& p,
                                         threadgroup const ShvqMlaTgLut& tgLut,
                                         thread float4 (&wLo)[4],
                                         thread float4 (&wHi)[4]) {
    const uint inc = p.isIncumbent & 0xFFu;
    if (inc == 3u) {
        device const half* hSrc = (device const half*)woABlocks + (size_t)row * p.groupIn + col0;
        #pragma unroll
        for (uint w = 0u; w < 4u; ++w) {
            wLo[w] = float4(*(device const half4*)(hSrc + w * 8u));
            wHi[w] = float4(*(device const half4*)(hSrc + w * 8u + 4u));
        }
        return;
    }
    MLAAbsorbParams mp = {p.T, p.oGroups, 0u, p.oLoraRank, p.groupIn, 0u, p.bits, p.group, inc};
    qw_mla_unpack_slice32_reg<SUB2>(woABlocks, row, col0, mp, tgLut, wLo, wHi);
}

METAL_FUNC void qw_hca_accum32_row(thread float& acc, thread const float4 (&wLo)[4], thread const float4 (&wHi)[4], threadgroup const half* xPtr) {
    #pragma unroll
    for (uint w = 0u; w < 4u; ++w) {
        const float4 x0 = float4(*(threadgroup const half4*)(xPtr + w * 8u));
        const float4 x1 = float4(*(threadgroup const half4*)(xPtr + w * 8u + 4u));
        for (short c = 0; c < 4; ++c) { acc = precise::fma(wLo[w][c], x0[c], acc); acc = precise::fma(wHi[w][c], x1[c], acc); }
    }
}

METAL_FUNC void qw_hca_reduce_t(const thread float (&acc)[4], threadgroup float* shOutRow, uint sg, uint lane) {
    #pragma unroll
    for (uint r = 0u; r < 4u; ++r) { const float sum = simd_sum(acc[r]); if (lane == 0u) shOutRow[sg * 4u + r] = sum; }
}

// 6a. 3D-batched grouped wo_a GEMV + sigmoid(o_gate) + H_32 exit (decode/verify T <= 15).
// Grid: (oLoraRank / 32, oGroups, T), Threadgroup: (256, 1, 1).
struct HCAWoAGemvTg {
    half shX[1024];
    float shOut[32];
    ShvqMlaTgLut lut;
};

template <bool SUB2>
METAL_FUNC void qw_hca_grouped_wo_a_gemv_body(
    device const half*          zIn,
    device const char*          woABlocks,
    device const half*          qUpGate,
    device half*                zGatedOut,
    constant HCAGroupedWoAParams& p,
    threadgroup HCAWoAGemvTg&   st,
    uint3                       tg,
    uint                        tid,
    uint                        sg,
    uint                        lane
) {
    const uint cBase = tg.x * 32u;
    const uint g = tg.y;
    const uint t = tg.z;
    if (t >= p.T || g >= p.oGroups || cBase >= p.oLoraRank) return;

    const uint inc = p.isIncumbent & 0xFFu;
    const bool skipFwht = (p.isIncumbent & 0x10000u) != 0u;
    if (inc == 2u) {
        qw_mla_load_shvq_lut<SUB2>(st.lut, p.bits, tid, 256u);
    }

    const uint baseRow = g * p.oLoraRank + cBase + sg * 4u;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint kBase = 0u; kBase < p.groupIn; kBase += 1024u) {
        const uint chunkCols = min(1024u, p.groupIn - kBase);
        const uint vecCols = chunkCols >> 2u;
        device const half4* src4 = (device const half4*)(
            zIn + ((size_t)t * p.oGroups + g) * p.groupIn + kBase);
        for (uint i = tid; i < vecCols; i += 256u) {
            *(threadgroup half4*)(st.shX + i * 4u) = src4[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint colOff = lane * 32u;
        if (colOff < chunkCols) {
            const uint col0 = kBase + colOff;
            #pragma unroll
            for (uint r = 0u; r < 4u; ++r) {
                float4 wLo[4], wHi[4];
                qw_hca_unpack_wo_a_reg32<SUB2>(woABlocks, baseRow + r, col0, p, st.lut, wLo, wHi);
                qw_hca_accum32_row(acc[r], wLo, wHi, st.shX + colOff);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for (uint r = 0u; r < 4u; ++r) {
        const float sum = simd_sum(acc[r]);
        if (lane == 0u) st.shOut[sg * 4u + r] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        const uint globalCol = g * p.oLoraRank + cBase + lane;
        const float gate = float(qUpGate[(size_t)t * p.qUpStride + p.oGateOffset + globalCol]);
        const float gated = st.shOut[lane] / (1.0f + exp(-gate));
        zGatedOut[(size_t)t * (p.oGroups * p.oLoraRank) + globalCol] = half(skipFwht ? gated : qwg_fwht32_lane_f32(gated, lane));
    }
}

kernel void qwg_hca_grouped_wo_a_gemv(
    device const half* zIn [[buffer(0)]], device const char* woABlocks [[buffer(1)]],
    device const half* qUpGate [[buffer(2)]], device half* zGatedOut [[buffer(3)]],
    constant HCAGroupedWoAParams& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) { threadgroup HCAWoAGemvTg st; qw_hca_grouped_wo_a_gemv_body<false>(zIn, woABlocks, qUpGate, zGatedOut, p, st, tg, tid, sg, lane); }

kernel void qwg_hca_grouped_wo_a_gemv_sub2(
    device const half* zIn [[buffer(0)]], device const char* woABlocks [[buffer(1)]],
    device const half* qUpGate [[buffer(2)]], device half* zGatedOut [[buffer(3)]],
    constant HCAGroupedWoAParams& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) { threadgroup HCAWoAGemvTg st; qw_hca_grouped_wo_a_gemv_body<true>(zIn, woABlocks, qUpGate, zGatedOut, p, st, tg, tid, sg, lane); }

// 6b. 3D-batched grouped wo_a 8x8 MMA + sigmoid(o_gate) + H_32 exit (prefill T >= 16).
// Grid: (oGroups, (T + 31) / 32, oLoraRank / 128), Threadgroup: (128, 1, 1).
#define HCA_WOA_TM 32u
#define HCA_WOA_TN 128u
#define HCA_WOA_TK 32u

union HCAWoATgStage {
    struct {
        half us[HCA_WOA_TM * (HCA_WOA_TK + 8u)];
        half ws[HCA_WOA_TK * (HCA_WOA_TN + 8u)];
    } tile;
    float os[HCA_WOA_TM * (HCA_WOA_TN + 4u)];
};

template <bool SUB2>
METAL_FUNC void qw_hca_grouped_wo_a_mma_body(
    device const half*          zIn,
    device const char*          woABlocks,
    device const half*          qUpGate,
    device half*                zGatedOut,
    constant HCAGroupedWoAParams& p,
    threadgroup HCAWoATgStage&  st,
    threadgroup ShvqMlaTgLut&   tgLut,
    uint3                       tg,
    uint                        tid,
    uint                        sg,
    uint                        lane
) {
    const uint g = tg.x;
    const uint tBase = tg.y * HCA_WOA_TM;
    const uint cBase = tg.z * HCA_WOA_TN;
    if (g >= p.oGroups || tBase >= p.T || cBase >= p.oLoraRank) return;

    const uint inc = p.isIncumbent & 0xFFu;
    const bool skipFwht = (p.isIncumbent & 0x10000u) != 0u;
    if (inc == 2u && qw_mla_load_shvq_lut<SUB2>(tgLut, p.bits, tid, 128u)) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint usStride = HCA_WOA_TK + 8u, wsStride = HCA_WOA_TN + 8u, osStride = HCA_WOA_TN + 4u;

    simdgroup_float8x8 acc[4][4];
    #pragma unroll
    for (uint i = 0u; i < 4u; ++i) {
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
    }

    const uint wRow = g * p.oLoraRank + cBase + tid;
    const uint uRow = tid >> 2u, uCol = (tid & 3u) * 8u;
    const bool uValid = tBase + uRow < p.T;
    device const half* uSrc = zIn + ((size_t)(tBase + uRow) * p.oGroups + g) * p.groupIn + uCol;

    for (uint k0 = 0u; k0 < p.groupIn; k0 += HCA_WOA_TK) {
        half4 a = half4(0.0h), b = half4(0.0h);
        if (uValid) {
            a = *(device const half4*)(uSrc + k0);
            b = *(device const half4*)(uSrc + k0 + 4u);
        }
        *(threadgroup half4*)(st.tile.us + uRow * usStride + uCol) = a;
        *(threadgroup half4*)(st.tile.us + uRow * usStride + uCol + 4u) = b;
        qw_hca_unpack_wo_a_32<SUB2>(woABlocks, wRow, k0, p, tgLut, st.tile.ws + tid, wsStride);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0u; kk < HCA_WOA_TK; kk += 8u) {
            simdgroup_half8x8 af[4], bf[4];
            #pragma unroll
            for (uint i = 0u; i < 4u; ++i) simdgroup_load(af[i], st.tile.us + i * 8u * usStride + kk, usStride);
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) simdgroup_load(bf[j], st.tile.ws + kk * wsStride + sg * 32u + j * 8u, wsStride);
            #pragma unroll
            for (uint i = 0u; i < 4u; ++i) {
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j) simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint i = 0u; i < 4u; ++i) {
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) simdgroup_store(acc[i][j], st.os + i * 8u * osStride + sg * 32u + j * 8u, osStride);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint outStride = p.oGroups * p.oLoraRank;
    const uint rows = min(HCA_WOA_TM, p.T - tBase);
    const uint globalCol = g * p.oLoraRank + cBase + sg * 32u + lane;
    for (uint m = 0u; m < rows; ++m) {
        const float raw = st.os[m * osStride + sg * 32u + lane];
        const float gate = float(qUpGate[(size_t)(tBase + m) * p.qUpStride + p.oGateOffset + globalCol]);
        const float gated = raw / (1.0f + exp(-gate));
        zGatedOut[(size_t)(tBase + m) * outStride + globalCol] = half(skipFwht ? gated : qwg_fwht32_lane_f32(gated, lane));
    }
}

kernel void qwg_hca_grouped_wo_a_mma(
    device const half* zIn [[buffer(0)]], device const char* woABlocks [[buffer(1)]],
    device const half* qUpGate [[buffer(2)]], device half* zGatedOut [[buffer(3)]],
    constant HCAGroupedWoAParams& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) { threadgroup HCAWoATgStage st; threadgroup ShvqMlaTgLut tgLut; qw_hca_grouped_wo_a_mma_body<false>(zIn, woABlocks, qUpGate, zGatedOut, p, st, tgLut, tg, tid, sg, lane); }

kernel void qwg_hca_grouped_wo_a_mma_sub2(
    device const half* zIn [[buffer(0)]], device const char* woABlocks [[buffer(1)]],
    device const half* qUpGate [[buffer(2)]], device half* zGatedOut [[buffer(3)]],
    constant HCAGroupedWoAParams& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) { threadgroup HCAWoATgStage st; threadgroup ShvqMlaTgLut tgLut; qw_hca_grouped_wo_a_mma_body<true>(zIn, woABlocks, qUpGate, zGatedOut, p, st, tgLut, tg, tid, sg, lane); }

// 6c. Multi-row grouped wo_a GEMV (2 <= T <= 8 verify) reusing woABlocks across all T tokens.
// Grid: (oLoraRank / 32, oGroups, 1), Threadgroup: (256, 1, 1).
struct HCAWoAGemvMRTg {
    half shX[8][1024];
    float shOut[8][32];
    ShvqMlaTgLut lut;
};

template <bool SUB2>
METAL_FUNC void qw_hca_grouped_wo_a_gemv_mr_body(
    device const half*          zIn,
    device const char*          woABlocks,
    device const half*          qUpGate,
    device half*                zGatedOut,
    constant HCAGroupedWoAParams& p,
    threadgroup HCAWoAGemvMRTg& st,
    uint3                       tg,
    uint                        tid,
    uint                        sg,
    uint                        lane
) {
    const uint cBase = tg.x * 32u;
    const uint g = tg.y;
    const uint rowsT = min(p.T, 8u);
    if (rowsT == 0u || g >= p.oGroups || cBase >= p.oLoraRank) return;

    const uint inc = p.isIncumbent & 0xFFu;
    const bool skipFwht = (p.isIncumbent & 0x10000u) != 0u;
    if (inc == 2u) {
        qw_mla_load_shvq_lut<SUB2>(st.lut, p.bits, tid, 256u);
    }

    const uint baseRow = g * p.oLoraRank + cBase + sg * 4u;
    float acc[8][4];
    for (uint t = 0u; t < 8u; ++t) for (uint r = 0u; r < 4u; ++r) acc[t][r] = 0.0f;

    for (uint kBase = 0u; kBase < p.groupIn; kBase += 1024u) {
        const uint chunkCols = min(1024u, p.groupIn - kBase);
        const uint vecCols = chunkCols >> 2u;
        for (uint t = 0u; t < rowsT; ++t) {
            device const half4* src4 = (device const half4*)(
                zIn + ((size_t)t * p.oGroups + g) * p.groupIn + kBase);
            for (uint i = tid; i < vecCols; i += 256u) {
                *(threadgroup half4*)(st.shX[t] + i * 4u) = src4[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint colOff = lane * 32u;
        if (colOff < chunkCols) {
            const uint col0 = kBase + colOff;
            #pragma unroll
            for (uint r = 0u; r < 4u; ++r) {
                float4 wLo[4], wHi[4];
                qw_hca_unpack_wo_a_reg32<SUB2>(woABlocks, baseRow + r, col0, p, st.lut, wLo, wHi);
                for (uint t = 0u; t < rowsT; ++t) qw_hca_accum32_row(acc[t][r], wLo, wHi, st.shX[t] + colOff);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint t = 0u; t < rowsT; ++t) qw_hca_reduce_t(acc[t], st.shOut[t], sg, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg < rowsT) {
        const uint outStride = p.oGroups * p.oLoraRank;
        const uint globalCol = g * p.oLoraRank + cBase + lane;
        const float gate = float(qUpGate[(size_t)sg * p.qUpStride + p.oGateOffset + globalCol]);
        const float gated = st.shOut[sg][lane] / (1.0f + exp(-gate));
        zGatedOut[(size_t)sg * outStride + globalCol] =
            half(skipFwht ? gated : qwg_fwht32_lane_f32(gated, lane));
    }
}

kernel void qwg_hca_grouped_wo_a_gemv_mr(
    device const half* zIn [[buffer(0)]], device const char* woABlocks [[buffer(1)]],
    device const half* qUpGate [[buffer(2)]], device half* zGatedOut [[buffer(3)]],
    constant HCAGroupedWoAParams& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup HCAWoAGemvMRTg st;
    qw_hca_grouped_wo_a_gemv_mr_body<false>(zIn, woABlocks, qUpGate, zGatedOut, p, st, tg, tid, sg, lane);
}

kernel void qwg_hca_grouped_wo_a_gemv_mr_sub2(
    device const half* zIn [[buffer(0)]], device const char* woABlocks [[buffer(1)]],
    device const half* qUpGate [[buffer(2)]], device half* zGatedOut [[buffer(3)]],
    constant HCAGroupedWoAParams& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup HCAWoAGemvMRTg st;
    qw_hca_grouped_wo_a_gemv_mr_body<true>(zIn, woABlocks, qUpGate, zGatedOut, p, st, tg, tid, sg, lane);
}
