// qwnorm_gated.metal — Fused low-rank GatedNorm and sandwich-residual GatedNorm kernels.
// Exact BF16-rounded unrotated math (`modeling_gigachat4.py:161-173`) with optional in-register
// 32-point FWHT rotation and optional MoE `(xRot32, xPlain32)` F32 bridge.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_dequant.metalh"
#include "include/qw_bf16_math.metalh"
using namespace metal;

#pragma clang fp contract(off) reassociate(off)

struct QWGatedNormParams {
    uint  rows;
    uint  hidden;
    uint  rank;
    uint  flags;
    float eps;
    float gatingWeight;
    uint  rowTile;
    uint  pad0;
};

struct QWGatedSandwichParams {
    uint  rows;
    uint  hidden;
    uint  postRank;
    uint  preRank;
    uint  flags;
    uint  rowTile;
    float postEps;
    float postGatingWeight;
    float preEps;
    float preGatingWeight;
};

// Flag bit layout shared across QWGatedNormParams and QWGatedSandwichParams:
// bit 0: post/main weight is BF16   | bit 1: post/main gateUp is BF16 | bit 2: post/main gateDown is BF16
// bit 3: post/main zeroCentered     | bit 4: applyFwht32 on output    | bit 5: emit MoE F32 bridge
// bit 6: skipRot32 in MoE bridge    | bit 7: hasNextPreNorm
// bit 8: pre weight is BF16         | bit 9: pre gateUp is BF16       | bit 10: pre gateDown is BF16
// bit 11: pre zeroCentered

METAL_FUNC float qw_gated_bf16_f32(float v) {
    if (!qw_is_finite_bits(v)) return v;
    uint u = as_type<uint>(v);
    u = u + 0x7FFFu + ((u >> 16u) & 1u);
    return as_type<float>(u & 0xFFFF0000u);
}

METAL_FUNC ushort qw_gated_f32_to_bf16_bits(float v) {
    if (!qw_is_finite_bits(v)) return as_type<ushort>(bfloat(v));
    uint u = as_type<uint>(v);
    u = u + 0x7FFFu + ((u >> 16u) & 1u);
    return ushort(u >> 16u);
}

METAL_FUNC float qw_gated_load_w(device const ushort* p, size_t idx, bool isBF16) {
    const ushort bits = p[idx];
    return isBF16 ? as_type<float>(uint(bits) << 16u) : float(as_type<half>(bits));
}

METAL_FUNC float qw_gated_dot16_u4(uint4 d0, uint4 d1, threadgroup const float* g1Row, bool isBF16) {
    float dotD = 0.0f;
    if (isBF16) {
        dotD += as_type<float>(d0.x << 16u) * g1Row[0];
        dotD += as_type<float>(d0.x & 0xFFFF0000u) * g1Row[1];
        dotD += as_type<float>(d0.y << 16u) * g1Row[2];
        dotD += as_type<float>(d0.y & 0xFFFF0000u) * g1Row[3];
        dotD += as_type<float>(d0.z << 16u) * g1Row[4];
        dotD += as_type<float>(d0.z & 0xFFFF0000u) * g1Row[5];
        dotD += as_type<float>(d0.w << 16u) * g1Row[6];
        dotD += as_type<float>(d0.w & 0xFFFF0000u) * g1Row[7];
        dotD += as_type<float>(d1.x << 16u) * g1Row[8];
        dotD += as_type<float>(d1.x & 0xFFFF0000u) * g1Row[9];
        dotD += as_type<float>(d1.y << 16u) * g1Row[10];
        dotD += as_type<float>(d1.y & 0xFFFF0000u) * g1Row[11];
        dotD += as_type<float>(d1.z << 16u) * g1Row[12];
        dotD += as_type<float>(d1.z & 0xFFFF0000u) * g1Row[13];
        dotD += as_type<float>(d1.w << 16u) * g1Row[14];
        dotD += as_type<float>(d1.w & 0xFFFF0000u) * g1Row[15];
    } else {
        const half2 h0 = as_type<half2>(d0.x), h1 = as_type<half2>(d0.y);
        const half2 h2 = as_type<half2>(d0.z), h3 = as_type<half2>(d0.w);
        const half2 h4 = as_type<half2>(d1.x), h5 = as_type<half2>(d1.y);
        const half2 h6 = as_type<half2>(d1.z), h7 = as_type<half2>(d1.w);
        dotD += float(h0.x) * g1Row[0]; dotD += float(h0.y) * g1Row[1];
        dotD += float(h1.x) * g1Row[2]; dotD += float(h1.y) * g1Row[3];
        dotD += float(h2.x) * g1Row[4]; dotD += float(h2.y) * g1Row[5];
        dotD += float(h3.x) * g1Row[6]; dotD += float(h3.y) * g1Row[7];
        dotD += float(h4.x) * g1Row[8]; dotD += float(h4.y) * g1Row[9];
        dotD += float(h5.x) * g1Row[10]; dotD += float(h5.y) * g1Row[11];
        dotD += float(h6.x) * g1Row[12]; dotD += float(h6.y) * g1Row[13];
        dotD += float(h7.x) * g1Row[14]; dotD += float(h7.y) * g1Row[15];
    }
    return dotD;
}

METAL_FUNC void qw_gated_emit_output(
        float yUnrot,
        uint rowIdx,
        uint colIdx,
        uint hidden,
        uint lane,
        bool applyFwht32,
        bool emitMoE,
        bool skipRot32,
        device half* outBase,
        device float* xRot32Base,
        device float* xPlain32Base) {
    float outVal = yUnrot;
    if (QW_FWHT_ROTATED && applyFwht32) {
        outVal = qwg_fwht32_lane_f32(yUnrot, lane);
    }
    const half outH = half(outVal);
    const size_t off = (size_t)rowIdx * hidden + colIdx;
    outBase[off] = outH;
    if (!emitMoE) return;
    const float outF = float(outH);
    if (!skipRot32) {
        xRot32Base[off] = outF;
    }
    if (QW_FWHT_ROTATED && applyFwht32) {
        const float unrotF = qwg_fwht32_lane_f32(outF, lane);
        const half hi = half(unrotF);
        const half lo = half(unrotF - float(hi));
        xPlain32Base[off] = float(hi) + float(lo);
    } else {
        xPlain32Base[off] = outF;
    }
}

METAL_FUNC void qw_gated_stage_h_and_g1(
        device const half* xBase,
        device const ushort* w,
        device const ushort* gateUp,
        uint row0,
        uint mRows,
        uint hidden,
        uint rank,
        float eps,
        bool wBF16,
        bool upBF16,
        bool zeroCentered,
        uint tid,
        uint sg,
        uint lane,
        uint tptg,
        threadgroup ushort* shH,
        threadgroup float* shG1,
        threadgroup float* sgRed) {
    const uint nSg = max(1u, tptg >> 5u);
    if (tid < 256u) {
        for (uint m = 0u; m < mRows; ++m) {
            device const half* xRow = xBase + (size_t)(row0 + m) * hidden;
            float sq = 0.0f;
            for (uint c = tid; c < hidden; c += 256u) {
                const float xv = float(xRow[c]);
                sq += xv * xv;
            }
            sq = qwg_simd_reduce_down<32u>(sq);
            if (lane == 0u) {
                sgRed[m * 8u + sg] = sq;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint m = 0u; m < mRows; ++m) {
        threadgroup const float* rs = sgRed + m * 8u;
        const float sumSq = ((rs[0] + rs[1]) + (rs[2] + rs[3])) + ((rs[4] + rs[5]) + (rs[6] + rs[7]));
        const float rstd = precise::divide(1.0f, precise::sqrt(precise::divide(sumSq, float(hidden)) + eps));
        device const half* xRow = xBase + (size_t)(row0 + m) * hidden;
        threadgroup ushort* hRow = shH + m * hidden;
        for (uint c = tid; c < hidden; c += tptg) {
            const float wv = qw_gated_load_w(w, c, wBF16);
            const float scale = zeroCentered ? (1.0f + wv) : wv;
            const float hVal = (float(xRow[c]) * rstd) * scale;
            hRow[c] = qw_gated_f32_to_bf16_bits(hVal);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = sg; r < rank; r += nSg) {
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        const size_t upRowOff = (size_t)r * hidden;
        for (uint c = lane; c < hidden; c += 32u) {
            const float wu = qw_gated_load_w(gateUp, upRowOff + c, upBF16);
            acc0 += wu * as_type<float>(uint(shH[c]) << 16u);
            if (mRows >= 2u) acc1 += wu * as_type<float>(uint(shH[hidden + c]) << 16u);
            if (mRows >= 3u) acc2 += wu * as_type<float>(uint(shH[2u * hidden + c]) << 16u);
            if (mRows >= 4u) acc3 += wu * as_type<float>(uint(shH[3u * hidden + c]) << 16u);
        }
        acc0 = qwg_simd_reduce_down<32u>(acc0);
        if (mRows >= 2u) acc1 = qwg_simd_reduce_down<32u>(acc1);
        if (mRows >= 3u) acc2 = qwg_simd_reduce_down<32u>(acc2);
        if (mRows >= 4u) acc3 = qwg_simd_reduce_down<32u>(acc3);
        if (lane == 0u) {
            shG1[r] = qw_gated_bf16_f32(precise::divide(acc0, 1.0f + precise::exp(-acc0)));
            if (mRows >= 2u) shG1[32u + r] = qw_gated_bf16_f32(precise::divide(acc1, 1.0f + precise::exp(-acc1)));
            if (mRows >= 3u) shG1[64u + r] = qw_gated_bf16_f32(precise::divide(acc2, 1.0f + precise::exp(-acc2)));
            if (mRows >= 4u) shG1[96u + r] = qw_gated_bf16_f32(precise::divide(acc3, 1.0f + precise::exp(-acc3)));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Standalone single-TG / multi-row cooperative GatedNorm (`rows >= 1`, `D <= 8192`).
kernel void qwg_gated_rmsnorm_f16(
        device const half*          x         [[buffer(0)]],
        device const ushort*        w         [[buffer(1)]],
        device const ushort*        gateUp    [[buffer(2)]],
        device const ushort*        gateDown  [[buffer(3)]],
        device half*                out       [[buffer(4)]],
        constant QWGatedNormParams& p         [[buffer(5)]],
        device float*               xRot32    [[buffer(6)]],
        device float*               xPlain32  [[buffer(7)]],
        threadgroup ushort*         shH       [[threadgroup(0)]],
        uint tg   [[threadgroup_position_in_grid]],
        uint tid  [[thread_index_in_threadgroup]],
        uint sg   [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint tptg [[threads_per_threadgroup]]) {
    threadgroup float shG1[128];
    threadgroup float sgRed[32];

    const uint row0 = tg * p.rowTile;
    if (row0 >= p.rows) return;
    const uint mRows = min(p.rowTile, p.rows - row0);
    const uint hidden = p.hidden;
    const uint rank = p.rank;
    const uint nSg = max(1u, tptg >> 5u);
    const bool wBF16 = (p.flags & 1u) != 0u;
    const bool upBF16 = (p.flags & 2u) != 0u;
    const bool downBF16 = (p.flags & 4u) != 0u;
    const bool zeroCentered = (p.flags & 8u) != 0u;
    const bool applyFwht32 = (p.flags & 16u) != 0u;
    const bool emitMoE = (p.flags & 32u) != 0u;
    const bool skipRot32 = (p.flags & 64u) != 0u;

    qw_gated_stage_h_and_g1(
        x, w, gateUp, row0, mRows, hidden, rank, p.eps,
        wBF16, upBF16, zeroCentered, tid, sg, lane, tptg, shH, shG1, sgRed);

    const uint nBlocks = (hidden + 31u) >> 5u;
    const bool vecDown16 = (rank == 16u) && (((size_t)gateDown & 15u) == 0u);
    for (uint blk = sg; blk < nBlocks; blk += nSg) {
        const uint col = blk * 32u + lane;
        uint4 d0 = uint4(0u), d1 = uint4(0u);
        if (vecDown16 && col < hidden) {
            device const uint4* dp4 = (device const uint4*)(gateDown + (size_t)col * 16u);
            d0 = dp4[0];
            d1 = dp4[1];
        }
        for (uint m = 0u; m < mRows; ++m) {
            float yUnrot = 0.0f;
            if (col < hidden) {
                threadgroup const float* g1Row = shG1 + m * 32u;
                float dotD = 0.0f;
                if (vecDown16) {
                    dotD = qw_gated_dot16_u4(d0, d1, g1Row, downBF16);
                } else {
                    const size_t downOff = (size_t)col * rank;
                    for (uint r = 0u; r < rank; ++r) {
                        dotD += qw_gated_load_w(gateDown, downOff + r, downBF16) * g1Row[r];
                    }
                }
                const float g2 = qw_gated_bf16_f32(precise::divide(1.0f, 1.0f + precise::exp(-dotD)));
                const float hVal = as_type<float>(uint(shH[m * hidden + col]) << 16u);
                yUnrot = qw_gated_bf16_f32((hVal * p.gatingWeight) * g2);
            }
            if (applyFwht32) {
                qw_gated_emit_output(
                    yUnrot, row0 + m, col, hidden, lane, true,
                    emitMoE, skipRot32, out, xRot32, xPlain32);
            } else if (col < hidden) {
                qw_gated_emit_output(
                    yUnrot, row0 + m, col, hidden, lane, false,
                    emitMoE, skipRot32, out, xRot32, xPlain32);
            }
        }
    }
}

// Fused sandwich residual: `x_new = half(float(residual) + GatedNorm_post(branchOut))` -> `residual`,
// plus optional immediate `out = optional_fwht32(GatedNorm_pre(x_new))` + MoE `(xRot32, xPlain32)`.
kernel void qwg_gated_rmsnorm_sandwich_residual_f16(
        device const half*              branchOut    [[buffer(0)]],
        device half*                    residual     [[buffer(1)]],
        device const ushort*            postW        [[buffer(2)]],
        device const ushort*            postGateUp   [[buffer(3)]],
        device const ushort*            postGateDown [[buffer(4)]],
        device const ushort*            preW         [[buffer(5)]],
        device const ushort*            preGateUp    [[buffer(6)]],
        device const ushort*            preGateDown  [[buffer(7)]],
        device half*                    out          [[buffer(8)]],
        constant QWGatedSandwichParams& p            [[buffer(9)]],
        device float*                   xRot32       [[buffer(10)]],
        device float*                   xPlain32     [[buffer(11)]],
        threadgroup ushort*             shH          [[threadgroup(0)]],
        uint tg   [[threadgroup_position_in_grid]],
        uint tid  [[thread_index_in_threadgroup]],
        uint sg   [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint tptg [[threads_per_threadgroup]]) {
    threadgroup float shG1[128];
    threadgroup float sgRed[32];

    const uint row0 = tg * p.rowTile;
    if (row0 >= p.rows) return;
    const uint mRows = min(p.rowTile, p.rows - row0);
    const uint hidden = p.hidden;
    const uint postRank = p.postRank;
    const uint nSg = max(1u, tptg >> 5u);
    const bool postWBF16 = (p.flags & 1u) != 0u;
    const bool postUpBF16 = (p.flags & 2u) != 0u;
    const bool postDownBF16 = (p.flags & 4u) != 0u;
    const bool postZeroCentered = (p.flags & 8u) != 0u;
    const bool applyFwht32 = (p.flags & 16u) != 0u;
    const bool emitMoE = (p.flags & 32u) != 0u;
    const bool skipRot32 = (p.flags & 64u) != 0u;
    const bool hasNextPre = (p.flags & 128u) != 0u;

    qw_gated_stage_h_and_g1(
        branchOut, postW, postGateUp, row0, mRows, hidden, postRank, p.postEps,
        postWBF16, postUpBF16, postZeroCentered, tid, sg, lane, tptg, shH, shG1, sgRed);

    const bool vecPostDown16 = (postRank == 16u) && (((size_t)postGateDown & 15u) == 0u);
    for (uint c = tid; c < hidden; c += tptg) {
        uint4 pd0 = uint4(0u), pd1 = uint4(0u);
        if (vecPostDown16) {
            device const uint4* dp4 = (device const uint4*)(postGateDown + (size_t)c * 16u);
            pd0 = dp4[0];
            pd1 = dp4[1];
        }
        for (uint m = 0u; m < mRows; ++m) {
            device half* resRow = residual + (size_t)(row0 + m) * hidden;
            threadgroup const float* g1Row = shG1 + m * 32u;
            float dotD = 0.0f;
            if (vecPostDown16) {
                dotD = qw_gated_dot16_u4(pd0, pd1, g1Row, postDownBF16);
            } else {
                const size_t downOff = (size_t)c * postRank;
                for (uint r = 0u; r < postRank; ++r) {
                    dotD += qw_gated_load_w(postGateDown, downOff + r, postDownBF16) * g1Row[r];
                }
            }
            const float g2 = qw_gated_bf16_f32(precise::divide(1.0f, 1.0f + precise::exp(-dotD)));
            const float hVal = as_type<float>(uint(shH[m * hidden + c]) << 16u);
            const float yPost = qw_gated_bf16_f32((hVal * p.postGatingWeight) * g2);
            const half xNew = half(float(resRow[c]) + yPost);
            resRow[c] = xNew;
            if (hasNextPre) {
                shH[m * hidden + c] = as_type<ushort>(xNew);
            }
        }
    }
    if (!hasNextPre) return;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 256u) {
        for (uint m = 0u; m < mRows; ++m) {
            threadgroup const ushort* resBits = shH + m * hidden;
            float sq = 0.0f;
            for (uint c = tid; c < hidden; c += 256u) {
                const float xf = float(as_type<half>(resBits[c]));
                sq += xf * xf;
            }
            sq = qwg_simd_reduce_down<32u>(sq);
            if (lane == 0u) {
                sgRed[m * 8u + sg] = sq;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint preRank = p.preRank;
    const bool preWBF16 = (p.flags & 256u) != 0u;
    const bool preUpBF16 = (p.flags & 512u) != 0u;
    const bool preDownBF16 = (p.flags & 1024u) != 0u;
    const bool preZeroCentered = (p.flags & 2048u) != 0u;

    for (uint m = 0u; m < mRows; ++m) {
        threadgroup const float* rs = sgRed + m * 8u;
        const float sumSq = ((rs[0] + rs[1]) + (rs[2] + rs[3])) + ((rs[4] + rs[5]) + (rs[6] + rs[7]));
        const float rstd = precise::divide(1.0f, precise::sqrt(precise::divide(sumSq, float(hidden)) + p.preEps));
        threadgroup ushort* hRow = shH + m * hidden;
        for (uint c = tid; c < hidden; c += tptg) {
            const float wv = qw_gated_load_w(preW, c, preWBF16);
            const float scale = preZeroCentered ? (1.0f + wv) : wv;
            const float hVal = (float(as_type<half>(hRow[c])) * rstd) * scale;
            hRow[c] = qw_gated_f32_to_bf16_bits(hVal);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = sg; r < preRank; r += nSg) {
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        const size_t upRowOff = (size_t)r * hidden;
        for (uint c = lane; c < hidden; c += 32u) {
            const float wu = qw_gated_load_w(preGateUp, upRowOff + c, preUpBF16);
            acc0 += wu * as_type<float>(uint(shH[c]) << 16u);
            if (mRows >= 2u) acc1 += wu * as_type<float>(uint(shH[hidden + c]) << 16u);
            if (mRows >= 3u) acc2 += wu * as_type<float>(uint(shH[2u * hidden + c]) << 16u);
            if (mRows >= 4u) acc3 += wu * as_type<float>(uint(shH[3u * hidden + c]) << 16u);
        }
        acc0 = qwg_simd_reduce_down<32u>(acc0);
        if (mRows >= 2u) acc1 = qwg_simd_reduce_down<32u>(acc1);
        if (mRows >= 3u) acc2 = qwg_simd_reduce_down<32u>(acc2);
        if (mRows >= 4u) acc3 = qwg_simd_reduce_down<32u>(acc3);
        if (lane == 0u) {
            shG1[r] = qw_gated_bf16_f32(precise::divide(acc0, 1.0f + precise::exp(-acc0)));
            if (mRows >= 2u) shG1[32u + r] = qw_gated_bf16_f32(precise::divide(acc1, 1.0f + precise::exp(-acc1)));
            if (mRows >= 3u) shG1[64u + r] = qw_gated_bf16_f32(precise::divide(acc2, 1.0f + precise::exp(-acc2)));
            if (mRows >= 4u) shG1[96u + r] = qw_gated_bf16_f32(precise::divide(acc3, 1.0f + precise::exp(-acc3)));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint nBlocks = (hidden + 31u) >> 5u;
    const bool vecPreDown16 = (preRank == 16u) && (((size_t)preGateDown & 15u) == 0u);
    for (uint blk = sg; blk < nBlocks; blk += nSg) {
        const uint col = blk * 32u + lane;
        uint4 pd0 = uint4(0u), pd1 = uint4(0u);
        if (vecPreDown16 && col < hidden) {
            device const uint4* dp4 = (device const uint4*)(preGateDown + (size_t)col * 16u);
            pd0 = dp4[0];
            pd1 = dp4[1];
        }
        for (uint m = 0u; m < mRows; ++m) {
            float yUnrot = 0.0f;
            if (col < hidden) {
                threadgroup const float* g1Row = shG1 + m * 32u;
                float dotD = 0.0f;
                if (vecPreDown16) {
                    dotD = qw_gated_dot16_u4(pd0, pd1, g1Row, preDownBF16);
                } else {
                    const size_t downOff = (size_t)col * preRank;
                    for (uint r = 0u; r < preRank; ++r) {
                        dotD += qw_gated_load_w(preGateDown, downOff + r, preDownBF16) * g1Row[r];
                    }
                }
                const float g2 = qw_gated_bf16_f32(precise::divide(1.0f, 1.0f + precise::exp(-dotD)));
                const float hVal = as_type<float>(uint(shH[m * hidden + col]) << 16u);
                yUnrot = qw_gated_bf16_f32((hVal * p.preGatingWeight) * g2);
            }
            if (applyFwht32) {
                qw_gated_emit_output(
                    yUnrot, row0 + m, col, hidden, lane, true,
                    emitMoE, skipRot32, out, xRot32, xPlain32);
            } else if (col < hidden) {
                qw_gated_emit_output(
                    yUnrot, row0 + m, col, hidden, lane, false,
                    emitMoE, skipRot32, out, xRot32, xPlain32);
            }
        }
    }
}

// Fused Multi-TG Stage 1+2 (`hidden <= 4096`): `grid = (rank, rows, 1)`, `256` threads/TG.
kernel void qwg_gated_stage_h_g1_f16(
        device const half*          x         [[buffer(0)]],
        device const ushort*        w         [[buffer(1)]],
        device const ushort*        gateUp    [[buffer(2)]],
        device ushort*              hScratch  [[buffer(3)]],
        device float*               g1Scratch [[buffer(4)]],
        constant QWGatedNormParams& p         [[buffer(5)]],
        uint2 tg  [[threadgroup_position_in_grid]],
        uint  tid [[thread_index_in_threadgroup]],
        uint  sg  [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sgRed[8];
    threadgroup float shProd[4096];
    const uint r = tg.x;
    const uint row = tg.y;
    if (r >= p.rank || row >= p.rows) return;
    const uint hidden = p.hidden;
    const bool wBF16 = (p.flags & 1u) != 0u;
    const bool upBF16 = (p.flags & 2u) != 0u;
    const bool zeroCentered = (p.flags & 8u) != 0u;
    device const half* xRow = x + (size_t)row * hidden;
    float sq = 0.0f;
    for (uint c = tid; c < hidden; c += 256u) {
        const float xv = float(xRow[c]);
        sq += xv * xv;
    }
    sq = qwg_simd_reduce_down<32u>(sq);
    if (lane == 0u) sgRed[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sumSq = ((sgRed[0] + sgRed[1]) + (sgRed[2] + sgRed[3]))
                      + ((sgRed[4] + sgRed[5]) + (sgRed[6] + sgRed[7]));
    const float rstd = precise::divide(1.0f, precise::sqrt(precise::divide(sumSq, float(hidden)) + p.eps));
    device ushort* hRow = hScratch + (size_t)row * hidden;
    const size_t upRowOff = (size_t)r * hidden;
    for (uint c = tid; c < hidden; c += 256u) {
        const float wv = qw_gated_load_w(w, c, wBF16);
        const float scale = zeroCentered ? (1.0f + wv) : wv;
        const ushort hb = qw_gated_f32_to_bf16_bits((float(xRow[c]) * rstd) * scale);
        if (r == 0u) hRow[c] = hb;
        shProd[c] = qw_gated_load_w(gateUp, upRowOff + c, upBF16) * as_type<float>(uint(hb) << 16u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float acc = 0.0f;
        for (uint c = lane; c < hidden; c += 32u) {
            acc += shProd[c];
        }
        acc = qwg_simd_reduce_down<32u>(acc);
        if (lane == 0u) {
            g1Scratch[(size_t)row * 32u + r] =
                qw_gated_bf16_f32(precise::divide(acc, 1.0f + precise::exp(-acc)));
        }
    }
}

// Multi-TG Stage 1 (`rows <= 16`): exact 256-thread 8-SIMDgroup RMSNorm -> BF16 `hScratch[rows, hidden]`.
kernel void qwg_gated_stage_h_f16(
        device const half*          x         [[buffer(0)]],
        device const ushort*        w         [[buffer(1)]],
        device ushort*              hScratch  [[buffer(2)]],
        constant QWGatedNormParams& p         [[buffer(3)]],
        uint row  [[threadgroup_position_in_grid]],
        uint tid  [[thread_index_in_threadgroup]],
        uint sg   [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sgRed[8];
    if (row >= p.rows) return;
    const uint hidden = p.hidden;
    const bool wBF16 = (p.flags & 1u) != 0u;
    const bool zeroCentered = (p.flags & 8u) != 0u;
    device const half* xRow = x + (size_t)row * hidden;
    float sq = 0.0f;
    for (uint c = tid; c < hidden; c += 256u) {
        const float xv = float(xRow[c]);
        sq += xv * xv;
    }
    sq = qwg_simd_reduce_down<32u>(sq);
    if (lane == 0u) sgRed[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sumSq = ((sgRed[0] + sgRed[1]) + (sgRed[2] + sgRed[3]))
                      + ((sgRed[4] + sgRed[5]) + (sgRed[6] + sgRed[7]));
    const float rstd = precise::divide(1.0f, precise::sqrt(precise::divide(sumSq, float(hidden)) + p.eps));
    device ushort* hRow = hScratch + (size_t)row * hidden;
    for (uint c = tid; c < hidden; c += 256u) {
        const float wv = qw_gated_load_w(w, c, wBF16);
        const float scale = zeroCentered ? (1.0f + wv) : wv;
        hRow[c] = qw_gated_f32_to_bf16_bits((float(xRow[c]) * rstd) * scale);
    }
}

// Multi-TG Stage 2: 256-thread (`8`-SG) cooperative load + exact 32-lane reduction when `(flags & 512) != 0`.
kernel void qwg_gated_stage_g1_f16(
        device const ushort*        hScratch  [[buffer(0)]],
        device const ushort*        gateUp    [[buffer(1)]],
        device float*               g1Scratch [[buffer(2)]],
        constant QWGatedNormParams& p         [[buffer(3)]],
        uint2 tg  [[threadgroup_position_in_grid]],
        uint  tid [[thread_index_in_threadgroup]],
        uint  sg  [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float shProd[4096];
    const uint r = tg.x;
    const uint row0 = tg.y * 4u;
    if (r >= p.rank || row0 >= p.rows) return;
    const uint mRows = min(4u, p.rows - row0);
    const uint hidden = p.hidden;
    const bool upBF16 = (p.flags & 2u) != 0u;
    const size_t upRowOff = (size_t)r * hidden;
    device const ushort* h0 = hScratch + (size_t)row0 * hidden;
    if ((p.flags & 512u) != 0u && hidden <= 4096u) {
        for (uint m = 0u; m < mRows; ++m) {
            device const ushort* hm = h0 + (size_t)m * hidden;
            for (uint c = tid; c < hidden; c += 256u) {
                shProd[c] = qw_gated_load_w(gateUp, upRowOff + c, upBF16)
                          * as_type<float>(uint(hm[c]) << 16u);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0u) {
                float acc = 0.0f;
                for (uint c = lane; c < hidden; c += 32u) {
                    acc += shProd[c];
                }
                acc = qwg_simd_reduce_down<32u>(acc);
                if (lane == 0u) {
                    g1Scratch[(row0 + m) * 32u + r] =
                        qw_gated_bf16_f32(precise::divide(acc, 1.0f + precise::exp(-acc)));
                }
            }
            if (m + 1u < mRows) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        return;
    }
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    uint c = lane;
    for (; c + 96u < hidden; c += 128u) {
        const float wu0 = qw_gated_load_w(gateUp, upRowOff + c, upBF16);
        const float wu1 = qw_gated_load_w(gateUp, upRowOff + c + 32u, upBF16);
        const float wu2 = qw_gated_load_w(gateUp, upRowOff + c + 64u, upBF16);
        const float wu3 = qw_gated_load_w(gateUp, upRowOff + c + 96u, upBF16);
        acc0 = (((acc0 + wu0 * as_type<float>(uint(h0[c]) << 16u))
               + wu1 * as_type<float>(uint(h0[c + 32u]) << 16u))
               + wu2 * as_type<float>(uint(h0[c + 64u]) << 16u))
               + wu3 * as_type<float>(uint(h0[c + 96u]) << 16u);
        if (mRows >= 2u) {
            device const ushort* h1 = h0 + hidden;
            acc1 = (((acc1 + wu0 * as_type<float>(uint(h1[c]) << 16u))
                   + wu1 * as_type<float>(uint(h1[c + 32u]) << 16u))
                   + wu2 * as_type<float>(uint(h1[c + 64u]) << 16u))
                   + wu3 * as_type<float>(uint(h1[c + 96u]) << 16u);
        }
        if (mRows >= 3u) {
            device const ushort* h2 = h0 + 2u * hidden;
            acc2 = (((acc2 + wu0 * as_type<float>(uint(h2[c]) << 16u))
                   + wu1 * as_type<float>(uint(h2[c + 32u]) << 16u))
                   + wu2 * as_type<float>(uint(h2[c + 64u]) << 16u))
                   + wu3 * as_type<float>(uint(h2[c + 96u]) << 16u);
        }
        if (mRows >= 4u) {
            device const ushort* h3 = h0 + 3u * hidden;
            acc3 = (((acc3 + wu0 * as_type<float>(uint(h3[c]) << 16u))
                   + wu1 * as_type<float>(uint(h3[c + 32u]) << 16u))
                   + wu2 * as_type<float>(uint(h3[c + 64u]) << 16u))
                   + wu3 * as_type<float>(uint(h3[c + 96u]) << 16u);
        }
    }
    for (; c < hidden; c += 32u) {
        const float wu = qw_gated_load_w(gateUp, upRowOff + c, upBF16);
        acc0 += wu * as_type<float>(uint(h0[c]) << 16u);
        if (mRows >= 2u) acc1 += wu * as_type<float>(uint(h0[hidden + c]) << 16u);
        if (mRows >= 3u) acc2 += wu * as_type<float>(uint(h0[2u * hidden + c]) << 16u);
        if (mRows >= 4u) acc3 += wu * as_type<float>(uint(h0[3u * hidden + c]) << 16u);
    }
    acc0 = qwg_simd_reduce_down<32u>(acc0);
    if (mRows >= 2u) acc1 = qwg_simd_reduce_down<32u>(acc1);
    if (mRows >= 3u) acc2 = qwg_simd_reduce_down<32u>(acc2);
    if (mRows >= 4u) acc3 = qwg_simd_reduce_down<32u>(acc3);
    if (lane == 0u) {
        g1Scratch[row0 * 32u + r] = qw_gated_bf16_f32(precise::divide(acc0, 1.0f + precise::exp(-acc0)));
        if (mRows >= 2u) g1Scratch[(row0 + 1u) * 32u + r] = qw_gated_bf16_f32(precise::divide(acc1, 1.0f + precise::exp(-acc1)));
        if (mRows >= 3u) g1Scratch[(row0 + 2u) * 32u + r] = qw_gated_bf16_f32(precise::divide(acc2, 1.0f + precise::exp(-acc2)));
        if (mRows >= 4u) g1Scratch[(row0 + 3u) * 32u + r] = qw_gated_bf16_f32(precise::divide(acc3, 1.0f + precise::exp(-acc3)));
    }
}

// Multi-TG Stage 3: 256-thread (`8` SIMDgroup) TGs across all 40 cores (`256` cols x `4` rows per TG).
kernel void qwg_gated_apply_down_f16(
        device const ushort*        hScratch  [[buffer(0)]],
        device const float*         g1Scratch [[buffer(1)]],
        device const ushort*        gateDown  [[buffer(2)]],
        device half*                outOrRes  [[buffer(3)]],
        constant QWGatedNormParams& p         [[buffer(4)]],
        device float*               xRot32    [[buffer(5)]],
        device float*               xPlain32  [[buffer(6)]],
        uint2 tg  [[threadgroup_position_in_grid]],
        uint  sg  [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint col = (tg.x * 8u + sg) * 32u + lane;
    const uint row0 = tg.y * 4u;
    if (row0 >= p.rows) return;
    const uint mRows = min(4u, p.rows - row0);
    const uint hidden = p.hidden;
    const uint rank = p.rank;
    const bool downBF16 = (p.flags & 4u) != 0u;
    const bool applyFwht32 = (p.flags & 16u) != 0u;
    const bool emitMoE = (p.flags & 32u) != 0u;
    const bool skipRot32 = (p.flags & 64u) != 0u;
    const bool addResidual = (p.flags & 128u) != 0u;
    const bool vecDown16 = (rank == 16u) && (((size_t)gateDown & 15u) == 0u);
    uint4 d0 = uint4(0u), d1 = uint4(0u);
    if (vecDown16 && col < hidden) {
        device const uint4* dp4 = (device const uint4*)(gateDown + (size_t)col * 16u);
        d0 = dp4[0];
        d1 = dp4[1];
    }
    for (uint m = 0u; m < mRows; ++m) {
        const uint rowIdx = row0 + m;
        float yUnrot = 0.0f;
        if (col < hidden) {
            device const float* g1 = g1Scratch + (size_t)rowIdx * 32u;
            float dotD = 0.0f;
            if (vecDown16) {
                if (downBF16) {
                    dotD += as_type<float>(d0.x << 16u) * g1[0];
                    dotD += as_type<float>(d0.x & 0xFFFF0000u) * g1[1];
                    dotD += as_type<float>(d0.y << 16u) * g1[2];
                    dotD += as_type<float>(d0.y & 0xFFFF0000u) * g1[3];
                    dotD += as_type<float>(d0.z << 16u) * g1[4];
                    dotD += as_type<float>(d0.z & 0xFFFF0000u) * g1[5];
                    dotD += as_type<float>(d0.w << 16u) * g1[6];
                    dotD += as_type<float>(d0.w & 0xFFFF0000u) * g1[7];
                    dotD += as_type<float>(d1.x << 16u) * g1[8];
                    dotD += as_type<float>(d1.x & 0xFFFF0000u) * g1[9];
                    dotD += as_type<float>(d1.y << 16u) * g1[10];
                    dotD += as_type<float>(d1.y & 0xFFFF0000u) * g1[11];
                    dotD += as_type<float>(d1.z << 16u) * g1[12];
                    dotD += as_type<float>(d1.z & 0xFFFF0000u) * g1[13];
                    dotD += as_type<float>(d1.w << 16u) * g1[14];
                    dotD += as_type<float>(d1.w & 0xFFFF0000u) * g1[15];
                } else {
                    const half2 h0 = as_type<half2>(d0.x), h1 = as_type<half2>(d0.y);
                    const half2 h2 = as_type<half2>(d0.z), h3 = as_type<half2>(d0.w);
                    const half2 h4 = as_type<half2>(d1.x), h5 = as_type<half2>(d1.y);
                    const half2 h6 = as_type<half2>(d1.z), h7 = as_type<half2>(d1.w);
                    dotD += float(h0.x) * g1[0]; dotD += float(h0.y) * g1[1];
                    dotD += float(h1.x) * g1[2]; dotD += float(h1.y) * g1[3];
                    dotD += float(h2.x) * g1[4]; dotD += float(h2.y) * g1[5];
                    dotD += float(h3.x) * g1[6]; dotD += float(h3.y) * g1[7];
                    dotD += float(h4.x) * g1[8]; dotD += float(h4.y) * g1[9];
                    dotD += float(h5.x) * g1[10]; dotD += float(h5.y) * g1[11];
                    dotD += float(h6.x) * g1[12]; dotD += float(h6.y) * g1[13];
                    dotD += float(h7.x) * g1[14]; dotD += float(h7.y) * g1[15];
                }
            } else {
                const size_t downOff = (size_t)col * rank;
                for (uint r = 0u; r < rank; ++r) {
                    dotD += qw_gated_load_w(gateDown, downOff + r, downBF16) * g1[r];
                }
            }
            const float g2 = qw_gated_bf16_f32(precise::divide(1.0f, 1.0f + precise::exp(-dotD)));
            const float hVal = as_type<float>(uint(hScratch[(size_t)rowIdx * hidden + col]) << 16u);
            yUnrot = qw_gated_bf16_f32((hVal * p.gatingWeight) * g2);
        }
        if (addResidual) {
            if (col < hidden) {
                const size_t off = (size_t)rowIdx * hidden + col;
                outOrRes[off] = half(float(outOrRes[off]) + yUnrot);
            }
        } else if (applyFwht32) {
            qw_gated_emit_output(
                yUnrot, rowIdx, col, hidden, lane, true,
                emitMoE, skipRot32, outOrRes, xRot32, xPlain32);
        } else if (col < hidden) {
            qw_gated_emit_output(
                yUnrot, rowIdx, col, hidden, lane, false,
                emitMoE, skipRot32, outOrRes, xRot32, xPlain32);
        }
    }
}
