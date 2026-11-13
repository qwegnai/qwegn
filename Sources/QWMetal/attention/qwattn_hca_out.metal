// qwattn_hca_out.metal — HCA output epilogue GPU kernels:
// fused 8x8 MMA VHA head mixing + GPT-J inverse YaRN RoPE, and 3D-batched
// grouped low-rank wo_a projection (GEMV decode/verify + 8x8 MMA prefill)
// with fused sigmoid(o_gate) and H_32 exit rotation (<= 6 KiB constant LUTs).

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "../include/qw_mla_dequant.metalh"
using namespace metal;

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
    qw_mla_unpack_slice32<false>(woABlocks, row, col0, mp, tgLut, dst, dstStride);
}

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
    qw_mla_unpack_slice32_reg<false>(woABlocks, row, col0, mp, tgLut, wLo, wHi);
}

// 6a. 3D-batched grouped wo_a GEMV + sigmoid(o_gate) + H_32 exit (decode/verify T <= 15).
// Grid: (oLoraRank / 32, oGroups, T), Threadgroup: (256, 1, 1).
kernel void qwg_hca_grouped_wo_a_gemv(
    device const half*          zIn        [[buffer(0)]],
    device const char*          woABlocks  [[buffer(1)]],
    device const half*          qUpGate    [[buffer(2)]],
    device half*                zGatedOut  [[buffer(3)]],
    constant HCAGroupedWoAParams& p        [[buffer(4)]],
    uint3                       tg         [[threadgroup_position_in_grid]],
    uint                        tid        [[thread_index_in_threadgroup]],
    uint                        sg         [[simdgroup_index_in_threadgroup]],
    uint                        lane       [[thread_index_in_simdgroup]]
) {
    const uint cBase = tg.x * 32u;
    const uint g = tg.y;
    const uint t = tg.z;
    if (t >= p.T || g >= p.oGroups || cBase >= p.oLoraRank) return;

    threadgroup half shX[4096];
    threadgroup float shOut[32];
    threadgroup ShvqMlaTgLut tgLut;

    const uint inc = p.isIncumbent & 0xFFu;
    const bool skipFwht = (p.isIncumbent & 0x10000u) != 0u;
    if (inc == 2u) {
        qw_mla_load_shvq_lut<false>(tgLut, p.bits, tid, 256u);
    }
    device const half* xSrc = zIn + ((size_t)t * p.oGroups + g) * p.groupIn;
    for (uint k = tid; k < p.groupIn; k += 256u) shX[k] = xSrc[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint baseRow = g * p.oLoraRank + cBase + sg * 4u;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint col0 = lane * 32u; col0 < p.groupIn; col0 += 1024u) {
        float4 xLo[4], xHi[4];
        #pragma unroll
        for (uint w = 0u; w < 4u; ++w) {
            xLo[w] = float4(*(threadgroup const half4*)(shX + col0 + w * 8u));
            xHi[w] = float4(*(threadgroup const half4*)(shX + col0 + w * 8u + 4u));
        }
        #pragma unroll
        for (uint r = 0u; r < 4u; ++r) {
            float4 wLo[4], wHi[4];
            qw_hca_unpack_wo_a_reg32(woABlocks, baseRow + r, col0, p, tgLut, wLo, wHi);
            #pragma unroll
            for (uint w = 0u; w < 4u; ++w) {
                acc[r] += dot(wLo[w], xLo[w]) + dot(wHi[w], xHi[w]);
            }
        }
    }
    #pragma unroll
    for (uint r = 0u; r < 4u; ++r) {
        const float sum = simd_sum(acc[r]);
        if (lane == 0u) shOut[sg * 4u + r] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        const uint globalCol = g * p.oLoraRank + cBase + lane;
        const float gate = float(qUpGate[(size_t)t * p.qUpStride + p.oGateOffset + globalCol]);
        const float gated = shOut[lane] / (1.0f + exp(-gate));
        zGatedOut[(size_t)t * (p.oGroups * p.oLoraRank) + globalCol] = half(skipFwht ? gated : qwg_fwht32_lane_f32(gated, lane));
    }
}

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

kernel void qwg_hca_grouped_wo_a_mma(
    device const half*          zIn        [[buffer(0)]],
    device const char*          woABlocks  [[buffer(1)]],
    device const half*          qUpGate    [[buffer(2)]],
    device half*                zGatedOut  [[buffer(3)]],
    constant HCAGroupedWoAParams& p        [[buffer(4)]],
    uint3                       tg         [[threadgroup_position_in_grid]],
    uint                        tid        [[thread_index_in_threadgroup]],
    uint                        sg         [[simdgroup_index_in_threadgroup]],
    uint                        lane       [[thread_index_in_simdgroup]]
) {
    const uint g = tg.x;
    const uint tBase = tg.y * HCA_WOA_TM;
    const uint cBase = tg.z * HCA_WOA_TN;
    if (g >= p.oGroups || tBase >= p.T || cBase >= p.oLoraRank) return;

    threadgroup HCAWoATgStage st;
    threadgroup ShvqMlaTgLut tgLut;

    const uint inc = p.isIncumbent & 0xFFu;
    const bool skipFwht = (p.isIncumbent & 0x10000u) != 0u;
    if (inc == 2u && qw_mla_load_shvq_lut<false>(tgLut, p.bits, tid, 128u)) {
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
        qw_hca_unpack_wo_a_32(woABlocks, wRow, k0, p, tgLut, st.tile.ws + tid, wsStride);
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
