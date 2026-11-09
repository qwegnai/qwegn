// qwgemm_moe_f32.metal — Routed MoE f32 GEMM kernels for quantized weights.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
#include "qw_dequant.metalh"
#include "qw_dequant_f32.metalh"
#include "qw_dequant_multi.metalh"

// Four output rows of one expert share one x load: the routed MoE is bound by the activation re-reads, not the weights.
METAL_FUNC float4 qwg_moe_rows4_f32(device const Block_HQ4_G32* blocks, device const float* x,
                                    uint rowBase, uint rowsLeft, uint K, uint lane) {
    const uint nBlocks = K >> 5u;
    device const Block_HQ4_G32* r0 = blocks + (size_t)rowBase * nBlocks;
    device const Block_HQ4_G32* r1 = (rowsLeft > 1u) ? (r0 + nBlocks) : r0;
    device const Block_HQ4_G32* r2 = (rowsLeft > 2u) ? (r0 + 2u * nBlocks) : r0;
    device const Block_HQ4_G32* r3 = (rowsLeft > 3u) ? (r0 + 3u * nBlocks) : r0;
    float4 acc = float4(0.0f);
    for (uint g = lane; g < nBlocks; g += 32u) {
        device const float* xg = x + (g << 5u);
        float4 lo[4], hi[4];
        #pragma unroll
        for (uint w = 0; w < 4; ++w) {
            device const float4* p4 = (device const float4*)(xg + w * 8u);
            lo[w] = p4[0];
            hi[w] = p4[1];
        }
        acc.x += qwg_dot_hq4_block_f32_reg(r0[g], lo, hi);
        acc.y += qwg_dot_hq4_block_f32_reg(r1[g], lo, hi);
        acc.z += qwg_dot_hq4_block_f32_reg(r2[g], lo, hi);
        acc.w += qwg_dot_hq4_block_f32_reg(r3[g], lo, hi);
    }
    acc += simd_shuffle_down(acc, 16);
    acc += simd_shuffle_down(acc, 8);
    acc += simd_shuffle_down(acc, 4);
    acc += simd_shuffle_down(acc, 2);
    acc += simd_shuffle_down(acc, 1);
    return acc;
}

METAL_FUNC void qwg_moe_rows4_f32_2(device const Block_HQ4_G32* blocks,
                                    device const float* x0,
                                    device const float* x1,
                                    uint rowBase, uint rowsLeft, uint K, uint lane,
                                    thread float4 &acc0, thread float4 &acc1) {
    const uint nBlocks = K >> 5u;
    device const Block_HQ4_G32* r0 = blocks + (size_t)rowBase * nBlocks;
    device const Block_HQ4_G32* r1 = (rowsLeft > 1u) ? (r0 + nBlocks) : r0;
    device const Block_HQ4_G32* r2 = (rowsLeft > 2u) ? (r0 + 2u * nBlocks) : r0;
    device const Block_HQ4_G32* r3 = (rowsLeft > 3u) ? (r0 + 3u * nBlocks) : r0;
    acc0 = float4(0.0f);
    acc1 = float4(0.0f);
    for (uint g = lane; g < nBlocks; g += 32u) {
        const uint col = g << 5u;
        float4 lo0[4], hi0[4], lo1[4], hi1[4];
        #pragma unroll
        for (uint w = 0; w < 4; ++w) {
            device const float4* p0 = (device const float4*)(x0 + col + w * 8u);
            device const float4* p1 = (device const float4*)(x1 + col + w * 8u);
            lo0[w] = p0[0]; hi0[w] = p0[1];
            lo1[w] = p1[0]; hi1[w] = p1[1];
        }
        const float2 d0 = qwg_dot_hq4_block_f32_reg2(r0[g], lo0, hi0, lo1, hi1);
        const float2 d1 = qwg_dot_hq4_block_f32_reg2(r1[g], lo0, hi0, lo1, hi1);
        const float2 d2 = qwg_dot_hq4_block_f32_reg2(r2[g], lo0, hi0, lo1, hi1);
        const float2 d3 = qwg_dot_hq4_block_f32_reg2(r3[g], lo0, hi0, lo1, hi1);
        acc0 += float4(d0.x, d1.x, d2.x, d3.x);
        acc1 += float4(d0.y, d1.y, d2.y, d3.y);
    }
    acc0 += simd_shuffle_down(acc0, 16); acc1 += simd_shuffle_down(acc1, 16);
    acc0 += simd_shuffle_down(acc0, 8);  acc1 += simd_shuffle_down(acc1, 8);
    acc0 += simd_shuffle_down(acc0, 4);  acc1 += simd_shuffle_down(acc1, 4);
    acc0 += simd_shuffle_down(acc0, 2);  acc1 += simd_shuffle_down(acc1, 2);
    acc0 += simd_shuffle_down(acc0, 1);  acc1 += simd_shuffle_down(acc1, 1);
}

// Routed gate_up with one output row per simdgroup (4 simdgroups = 128 threads = 4 rows per TG, zero barrier),
// or fused 32-channel paired gate+up + SwiGLU + H_32 when top bit of E is set (8 simdgroups = 256 threads = 32 intermediate channels per TG).
kernel void qwg_moe_gateup_q4g_f32_r1(device const Block_HQ4_G32* blocks [[buffer(0)]],
                                       device const float*         x      [[buffer(1)]],
                                       device const int*           ids    [[buffer(2)]],
                                       device float*               out    [[buffer(3)]],
                                       constant uint&              N      [[buffer(4)]],
                                       constant uint&              K      [[buffer(5)]],
                                       constant uint&              E      [[buffer(6)]],
                                       device const Block_HQ4_G32* coldBlocks [[buffer(7)]],
                                       uint3 tg   [[threadgroup_position_in_grid]],
                                       uint  sgi  [[simdgroup_index_in_threadgroup]],
                                       uint  lane [[thread_index_in_simdgroup]]) {
    const bool fuseSiluFwht = (E & 0x80000000u) != 0u;
    const uint residentE = (E >> 16u) & 0x7FFFu;
    const uint realE = E & 0x3FFu;
    const uint numToks = (E >> 10u) & 0x7u;
    const bool skipCold = (E & 0x4000u) != 0u;
    const bool onlyCold = (E & 0x8000u) != 0u;
    const uint k = tg.y, t = tg.z;
    const int rawExpSigned = ids[t * realE + k];
    if (rawExpSigned < 0) return;
    const uint rawExpert = uint(rawExpSigned);
    const bool isCold = (residentE > 0u) && (rawExpert >= residentE);
    if ((skipCold && isCold) || (onlyCold && !isCold)) return;
    if (fuseSiluFwht) {
        uint mCount = 1u;
        uint4 tokIds = uint4(t, 0u, 0u, 0u);
        uint4 pairSlots = uint4(t * realE + k, 0u, 0u, 0u);
        if (numToks >= 2u) {
            for (uint tPrev = 0u; tPrev <= t; ++tPrev) {
                const uint kLim = (tPrev == t) ? k : realE;
                for (uint kPrev = 0u; kPrev < kLim; ++kPrev) {
                    if (ids[tPrev * realE + kPrev] == rawExpSigned) return;
                }
            }
            for (uint tNext = t + 1u; tNext < numToks && mCount < 4u; ++tNext) {
                for (uint kNext = 0u; kNext < realE; ++kNext) {
                    if (ids[tNext * realE + kNext] == rawExpSigned) {
                        tokIds[mCount] = tNext;
                        pairSlots[mCount] = tNext * realE + kNext;
                        ++mCount;
                        break;
                    }
                }
            }
        }
        threadgroup float tgAct[128];
        const uint I = N >> 1u;
        const uint rBase = tg.x * 32u;
        const uint r0 = rBase + sgi * 4u;
        const uint expert = isCold ? (rawExpert - residentE) : rawExpert;
        device const Block_HQ4_G32* wSrc = isCold ? coldBlocks : blocks;
        const uint nBlocks = K >> 5u;
        const uint upHalf = lane >> 4u, hl = lane & 15u;
        const size_t eBase = (size_t)expert * N + (upHalf != 0u ? I : 0u);
        const uint rows = (r0 < I) ? min(4u, I - r0) : 0u;
        device const Block_HQ4_G32* a0 = wSrc + (eBase + (rows != 0u ? r0 : 0u)) * nBlocks;
        const uint st1 = rows > 1u ? nBlocks : 0u;
        const uint st2 = rows > 2u ? 2u * nBlocks : 0u;
        const uint st3 = rows > 3u ? 3u * nBlocks : 0u;
        if (mCount == 1u) {
            device const float* xt = x + (size_t)t * K;
            float4 acc = float4(0.0f);
            for (uint g = hl; rows != 0u && g < nBlocks; g += 16u) {
                device const float* xg = xt + (g << 5u);
                float4 lo[4], hi[4];
                #pragma unroll
                for (uint w0 = 0; w0 < 4; ++w0) {
                    device const float4* p4 = (device const float4*)(xg + w0 * 8u);
                    lo[w0] = p4[0];
                    hi[w0] = p4[1];
                }
                device const Block_HQ4_G32* p = a0 + g;
                acc.x += qwg_dot_hq4_block_f32_reg(p[0], lo, hi);
                acc.y += qwg_dot_hq4_block_f32_reg(p[st1], lo, hi);
                acc.z += qwg_dot_hq4_block_f32_reg(p[st2], lo, hi);
                acc.w += qwg_dot_hq4_block_f32_reg(p[st3], lo, hi);
            }
            acc += simd_shuffle_down(acc, 8);
            acc += simd_shuffle_down(acc, 4);
            acc += simd_shuffle_down(acc, 2);
            acc += simd_shuffle_down(acc, 1);
            const float4 uAcc = simd_shuffle_down(acc, 16);
            if (lane == 0u) {
                const uint c = sgi * 4u;
                tgAct[c + 0u] = rows > 0u ? qwg_swiglu_mul(acc.x, uAcc.x) : 0.0f;
                tgAct[c + 1u] = rows > 1u ? qwg_swiglu_mul(acc.y, uAcc.y) : 0.0f;
                tgAct[c + 2u] = rows > 2u ? qwg_swiglu_mul(acc.z, uAcc.z) : 0.0f;
                tgAct[c + 3u] = rows > 3u ? qwg_swiglu_mul(acc.w, uAcc.w) : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgi == 0u && rBase + lane < I) {
                const float v = qwg_fwht32_lane_f32(tgAct[lane], lane);
                out[(size_t)(t * realE + k) * I + rBase + lane] = v;
            }
            return;
        }
        float4 accM[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
        for (uint g = hl; rows != 0u && g < nBlocks; g += 16u) {
            const uint col = g << 5u;
            device const Block_HQ4_G32* p = a0 + g;
            const Block_HQ4_G32 b0 = p[0], b1 = p[st1], b2 = p[st2], b3 = p[st3];
            float4 lo0[4], hi0[4], lo1[4], hi1[4];
            device const float* xg0 = x + (size_t)tokIds[0] * K + col;
            device const float* xg1 = x + (size_t)tokIds[1] * K + col;
            #pragma unroll
            for (uint w0 = 0; w0 < 4; ++w0) {
                device const float4* p0 = (device const float4*)(xg0 + w0 * 8u);
                device const float4* p1 = (device const float4*)(xg1 + w0 * 8u);
                lo0[w0] = p0[0]; hi0[w0] = p0[1];
                lo1[w0] = p1[0]; hi1[w0] = p1[1];
            }
            const float2 d0 = qwg_dot_hq4_block_f32_reg2(b0, lo0, hi0, lo1, hi1);
            const float2 d1 = qwg_dot_hq4_block_f32_reg2(b1, lo0, hi0, lo1, hi1);
            const float2 d2 = qwg_dot_hq4_block_f32_reg2(b2, lo0, hi0, lo1, hi1);
            const float2 d3 = qwg_dot_hq4_block_f32_reg2(b3, lo0, hi0, lo1, hi1);
            accM[0] += float4(d0.x, d1.x, d2.x, d3.x);
            accM[1] += float4(d0.y, d1.y, d2.y, d3.y);
            if (mCount == 3u) {
                device const float* xg2 = x + (size_t)tokIds[2] * K + col;
                #pragma unroll
                for (uint w0 = 0; w0 < 4; ++w0) {
                    device const float4* p2 = (device const float4*)(xg2 + w0 * 8u);
                    lo0[w0] = p2[0]; hi0[w0] = p2[1];
                }
                accM[2].x += qwg_dot_hq4_block_f32_reg(b0, lo0, hi0);
                accM[2].y += qwg_dot_hq4_block_f32_reg(b1, lo0, hi0);
                accM[2].z += qwg_dot_hq4_block_f32_reg(b2, lo0, hi0);
                accM[2].w += qwg_dot_hq4_block_f32_reg(b3, lo0, hi0);
            } else if (mCount == 4u) {
                device const float* xg2 = x + (size_t)tokIds[2] * K + col;
                device const float* xg3 = x + (size_t)tokIds[3] * K + col;
                #pragma unroll
                for (uint w0 = 0; w0 < 4; ++w0) {
                    device const float4* p2 = (device const float4*)(xg2 + w0 * 8u);
                    device const float4* p3 = (device const float4*)(xg3 + w0 * 8u);
                    lo0[w0] = p2[0]; hi0[w0] = p2[1];
                    lo1[w0] = p3[0]; hi1[w0] = p3[1];
                }
                const float2 e0 = qwg_dot_hq4_block_f32_reg2(b0, lo0, hi0, lo1, hi1);
                const float2 e1 = qwg_dot_hq4_block_f32_reg2(b1, lo0, hi0, lo1, hi1);
                const float2 e2 = qwg_dot_hq4_block_f32_reg2(b2, lo0, hi0, lo1, hi1);
                const float2 e3 = qwg_dot_hq4_block_f32_reg2(b3, lo0, hi0, lo1, hi1);
                accM[2] += float4(e0.x, e1.x, e2.x, e3.x);
                accM[3] += float4(e0.y, e1.y, e2.y, e3.y);
            }
        }
        for (uint m = 0u; m < mCount; ++m) {
            accM[m] += simd_shuffle_down(accM[m], 8);
            accM[m] += simd_shuffle_down(accM[m], 4);
            accM[m] += simd_shuffle_down(accM[m], 2);
            accM[m] += simd_shuffle_down(accM[m], 1);
            const float4 uAcc = simd_shuffle_down(accM[m], 16);
            if (lane == 0u) {
                const uint c = m * 32u + sgi * 4u;
                tgAct[c + 0u] = rows > 0u ? qwg_swiglu_mul(accM[m].x, uAcc.x) : 0.0f;
                tgAct[c + 1u] = rows > 1u ? qwg_swiglu_mul(accM[m].y, uAcc.y) : 0.0f;
                tgAct[c + 2u] = rows > 2u ? qwg_swiglu_mul(accM[m].z, uAcc.z) : 0.0f;
                tgAct[c + 3u] = rows > 3u ? qwg_swiglu_mul(accM[m].w, uAcc.w) : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgi < mCount && rBase + lane < I) {
            const float v = qwg_fwht32_lane_f32(tgAct[sgi * 32u + lane], lane);
            out[(size_t)pairSlots[sgi] * I + rBase + lane] = v;
        }
        return;
    }
    const uint r = tg.x * 4u + sgi;
    if (r >= N) return;
    const uint expert = isCold ? (rawExpert - residentE) : rawExpert;
    device const Block_HQ4_G32* wSrc = isCold ? coldBlocks : blocks;
    const float v = qwg_gemv_hq4_row_f32(wSrc, x + (size_t)t * K, expert * N + r, K, lane);
    if (lane == 0u) out[(t * realE + k) * N + r] = v;
}

// Routed down + shared-expert combine, f32 activations (4 simdgroups = 128 threads = 16 rows per TG, zero barrier).
static inline void qwg_moe_write_route4(device float* out, uint slot, uint N, uint r0,
                                        uint rowsLeft, float4 value) {
    const size_t base = size_t(slot) * N + r0;
    out[base] = value.x;
    if (rowsLeft > 1u) out[base + 1u] = value.y;
    if (rowsLeft > 2u) out[base + 2u] = value.z;
    if (rowsLeft > 3u) out[base + 3u] = value.w;
}

kernel void qwg_moe_down_shared_combine_q4g_f32(device const Block_HQ4_G32* blocks  [[buffer(0)]],
                                                device const float*         x       [[buffer(1)]],
                                                device const float*         w       [[buffer(2)]],
                                                device const float*         shared  [[buffer(3)]],
                                                device const float*         gate    [[buffer(4)]],
                                                device float*               out     [[buffer(5)]],
                                                constant uint&              N       [[buffer(6)]],
                                                constant uint&              K       [[buffer(7)]],
                                                constant uint&              E       [[buffer(8)]],
                                                device const int*           ids     [[buffer(9)]],
                                                device half*                out16   [[buffer(10)]],
                                                constant uint&              hasOut16 [[buffer(11)]],
                                                device const Block_HQ4_G32* coldBlocks [[buffer(12)]],
                                                uint2 tg   [[threadgroup_position_in_grid]],
                                                uint  sgi  [[simdgroup_index_in_threadgroup]],
                                                uint  lane [[thread_index_in_simdgroup]]) {
    const uint r0 = (tg.x * 4u + sgi) * 4u, t = tg.y;
    const uint rowsLeft = (r0 < N) ? (N - r0) : 0u;
    if (rowsLeft == 0u) return;
    const uint filterMode = (E >> 30u) & 3u;
    const uint residentE = (E >> 16u) & 0x3FFFu;
    const uint numToks = (E >> 8u) & 0x7u;
    const uint realE = E & 0xFFu;
    const bool routeOutput = (hasOut16 & 2u) != 0u;
    const uint nBlocks = K >> 5u;
    if (numToks >= 2u) {
        float4 accM[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
        for (uint t0 = 0u; t0 < numToks; ++t0) {
            for (uint k0 = 0u; k0 < realE; ++k0) {
                const uint slot0 = t0 * realE + k0;
                const int rawExpSigned = ids[slot0];
                if (rawExpSigned < 0) continue;
                const uint rawExp = uint(rawExpSigned);
                const bool isCold = (residentE > 0u) && (rawExp >= residentE);
                if ((filterMode == 1u && isCold) || (filterMode == 2u && !isCold)) continue;
                bool seen = false;
                for (uint tp = 0u; tp <= t0 && !seen; ++tp) {
                    const uint kLim = (tp == t0) ? k0 : realE;
                    for (uint kp = 0u; kp < kLim; ++kp) {
                        if (ids[tp * realE + kp] == rawExpSigned) { seen = true; break; }
                    }
                }
                if (seen) continue;
                uint mCount = 1u;
                uint4 mTok = uint4(t0, 0u, 0u, 0u);
                uint4 mSlot = uint4(slot0, 0u, 0u, 0u);
                for (uint tn = t0 + 1u; tn < numToks && mCount < 4u; ++tn) {
                    for (uint kn = 0u; kn < realE; ++kn) {
                        if (ids[tn * realE + kn] == rawExpSigned) {
                            mTok[mCount] = tn;
                            mSlot[mCount] = tn * realE + kn;
                            ++mCount;
                            break;
                        }
                    }
                }
                const uint wExp = isCold ? (rawExp - residentE) : rawExp;
                device const Block_HQ4_G32* wSrc = isCold ? coldBlocks : blocks;
                for (uint pair = 0u; pair < mCount; pair += 2u) {
                    const uint sA = mSlot[pair], tA = mTok[pair];
                    if (pair + 1u < mCount) {
                        const uint sB = mSlot[pair + 1u], tB = mTok[pair + 1u];
                        float4 vx0 = float4(0.0f), vx1 = float4(0.0f);
                        if (nBlocks <= 32u && rowsLeft >= 4u) {
                            if (lane < nBlocks) {
                                device const float* xg0 = x + (size_t)sA * K + (lane << 5u);
                                device const float* xg1 = x + (size_t)sB * K + (lane << 5u);
                                float4 lo0[4], hi0[4], lo1[4], hi1[4];
                                #pragma unroll
                                for (uint w0 = 0; w0 < 4; ++w0) {
                                    device const float4* p0 = (device const float4*)(xg0 + w0 * 8u);
                                    device const float4* p1 = (device const float4*)(xg1 + w0 * 8u);
                                    lo0[w0] = p0[0]; hi0[w0] = p0[1];
                                    lo1[w0] = p1[0]; hi1[w0] = p1[1];
                                }
                                device const Block_HQ4_G32* rPtr = wSrc + (size_t)(wExp * N + r0) * nBlocks + lane;
                                const float2 d0 = qwg_dot_hq4_block_f32_reg2(rPtr[0], lo0, hi0, lo1, hi1);
                                const float2 d1 = qwg_dot_hq4_block_f32_reg2(rPtr[nBlocks], lo0, hi0, lo1, hi1);
                                const float2 d2 = qwg_dot_hq4_block_f32_reg2(rPtr[2u * nBlocks], lo0, hi0, lo1, hi1);
                                const float2 d3 = qwg_dot_hq4_block_f32_reg2(rPtr[3u * nBlocks], lo0, hi0, lo1, hi1);
                                vx0 = float4(d0.x, d1.x, d2.x, d3.x);
                                vx1 = float4(d0.y, d1.y, d2.y, d3.y);
                            }
                            vx0 += simd_shuffle_down(vx0, 16); vx1 += simd_shuffle_down(vx1, 16);
                            vx0 += simd_shuffle_down(vx0, 8);  vx1 += simd_shuffle_down(vx1, 8);
                            vx0 += simd_shuffle_down(vx0, 4);  vx1 += simd_shuffle_down(vx1, 4);
                            vx0 += simd_shuffle_down(vx0, 2);  vx1 += simd_shuffle_down(vx1, 2);
                            vx0 += simd_shuffle_down(vx0, 1);  vx1 += simd_shuffle_down(vx1, 1);
                        } else if ((nBlocks & 15u) == 0u && rowsLeft >= 4u) {
                            const uint upHalf = lane >> 4u, hl = lane & 15u;
                            const uint rowBase = wExp * N + r0 + upHalf * 2u;
                            device const Block_HQ4_G32* rPtr0 = wSrc + (size_t)rowBase * nBlocks;
                            device const Block_HQ4_G32* rPtr1 = rPtr0 + nBlocks;
                            device const float* xs0 = x + (size_t)sA * K;
                            device const float* xs1 = x + (size_t)sB * K;
                            float2 v0 = float2(0.0f), v1 = float2(0.0f);
                            for (uint g = hl; g < nBlocks; g += 16u) {
                                device const float* xg0 = xs0 + (g << 5u);
                                device const float* xg1 = xs1 + (g << 5u);
                                float4 lo0[4], hi0[4], lo1[4], hi1[4];
                                #pragma unroll
                                for (uint w0 = 0; w0 < 4; ++w0) {
                                    device const float4* p0 = (device const float4*)(xg0 + w0 * 8u);
                                    device const float4* p1 = (device const float4*)(xg1 + w0 * 8u);
                                    lo0[w0] = p0[0]; hi0[w0] = p0[1];
                                    lo1[w0] = p1[0]; hi1[w0] = p1[1];
                                }
                                const float2 d0 = qwg_dot_hq4_block_f32_reg2(rPtr0[g], lo0, hi0, lo1, hi1);
                                const float2 d1 = qwg_dot_hq4_block_f32_reg2(rPtr1[g], lo0, hi0, lo1, hi1);
                                v0 += float2(d0.x, d1.x);
                                v1 += float2(d0.y, d1.y);
                            }
                            v0 += simd_shuffle_down(v0, 8); v1 += simd_shuffle_down(v1, 8);
                            v0 += simd_shuffle_down(v0, 4); v1 += simd_shuffle_down(v1, 4);
                            v0 += simd_shuffle_down(v0, 2); v1 += simd_shuffle_down(v1, 2);
                            v0 += simd_shuffle_down(v0, 1); v1 += simd_shuffle_down(v1, 1);
                            const float2 v0Hi = simd_shuffle_down(v0, 16);
                            const float2 v1Hi = simd_shuffle_down(v1, 16);
                            vx0 = float4(v0.x, v0.y, v0Hi.x, v0Hi.y);
                            vx1 = float4(v1.x, v1.y, v1Hi.x, v1Hi.y);
                        } else {
                            qwg_moe_rows4_f32_2(wSrc, x + (size_t)sA * K, x + (size_t)sB * K,
                                                wExp * N + r0, rowsLeft, K, lane, vx0, vx1);
                        }
                        if (routeOutput) {
                            if (lane == 0u) {
                                qwg_moe_write_route4(out, sA, N, r0, rowsLeft, vx0);
                                qwg_moe_write_route4(out, sB, N, r0, rowsLeft, vx1);
                            }
                        } else {
                            accM[tA] += w[sA] * vx0;
                            accM[tB] += w[sB] * vx1;
                        }
                    } else {
                        float4 vx0 = float4(0.0f);
                        if (nBlocks <= 32u && rowsLeft >= 4u) {
                            if (lane < nBlocks) {
                                device const float* xg0 = x + (size_t)sA * K + (lane << 5u);
                                float4 lo0[4], hi0[4];
                                #pragma unroll
                                for (uint w0 = 0; w0 < 4; ++w0) {
                                    device const float4* p0 = (device const float4*)(xg0 + w0 * 8u);
                                    lo0[w0] = p0[0]; hi0[w0] = p0[1];
                                }
                                device const Block_HQ4_G32* rPtr = wSrc + (size_t)(wExp * N + r0) * nBlocks + lane;
                                vx0.x = qwg_dot_hq4_block_f32_reg(rPtr[0], lo0, hi0);
                                vx0.y = qwg_dot_hq4_block_f32_reg(rPtr[nBlocks], lo0, hi0);
                                vx0.z = qwg_dot_hq4_block_f32_reg(rPtr[2u * nBlocks], lo0, hi0);
                                vx0.w = qwg_dot_hq4_block_f32_reg(rPtr[3u * nBlocks], lo0, hi0);
                            }
                            vx0 += simd_shuffle_down(vx0, 16);
                            vx0 += simd_shuffle_down(vx0, 8);
                            vx0 += simd_shuffle_down(vx0, 4);
                            vx0 += simd_shuffle_down(vx0, 2);
                            vx0 += simd_shuffle_down(vx0, 1);
                        } else if ((nBlocks & 15u) == 0u && rowsLeft >= 4u) {
                            const uint upHalf = lane >> 4u, hl = lane & 15u;
                            const uint rowBase = wExp * N + r0 + upHalf * 2u;
                            device const Block_HQ4_G32* rPtr0 = wSrc + (size_t)rowBase * nBlocks;
                            device const Block_HQ4_G32* rPtr1 = rPtr0 + nBlocks;
                            device const float* xs0 = x + (size_t)sA * K;
                            float2 v0 = float2(0.0f);
                            for (uint g = hl; g < nBlocks; g += 16u) {
                                device const float* xg0 = xs0 + (g << 5u);
                                float4 lo0[4], hi0[4];
                                #pragma unroll
                                for (uint w0 = 0; w0 < 4; ++w0) {
                                    device const float4* p0 = (device const float4*)(xg0 + w0 * 8u);
                                    lo0[w0] = p0[0]; hi0[w0] = p0[1];
                                }
                                v0.x += qwg_dot_hq4_block_f32_reg(rPtr0[g], lo0, hi0);
                                v0.y += qwg_dot_hq4_block_f32_reg(rPtr1[g], lo0, hi0);
                            }
                            v0 += simd_shuffle_down(v0, 8);
                            v0 += simd_shuffle_down(v0, 4);
                            v0 += simd_shuffle_down(v0, 2);
                            v0 += simd_shuffle_down(v0, 1);
                            const float2 v0Hi = simd_shuffle_down(v0, 16);
                            vx0 = float4(v0.x, v0.y, v0Hi.x, v0Hi.y);
                        } else {
                            vx0 = qwg_moe_rows4_f32(wSrc, x + (size_t)sA * K, wExp * N + r0, rowsLeft, K, lane);
                        }
                        if (routeOutput) {
                            if (lane == 0u) qwg_moe_write_route4(out, sA, N, r0, rowsLeft, vx0);
                        } else {
                            accM[tA] += w[sA] * vx0;
                        }
                    }
                }
            }
        }
        if (routeOutput) return;
        if (lane == 0u) {
            for (uint tm = 0u; tm < numToks; ++tm) {
                const size_t base = (size_t)tm * N + r0;
                const float g = gate[tm];
                const float y0 = accM[tm].x + ((filterMode == 2u) ? out[base] : (g * shared[base]));
                out[base] = y0;
                if (hasOut16 != 0u) out16[base] = half(y0);
                if (rowsLeft > 1u) {
                    const float y1 = accM[tm].y + ((filterMode == 2u) ? out[base + 1u] : (g * shared[base + 1u]));
                    out[base + 1u] = y1;
                    if (hasOut16 != 0u) out16[base + 1u] = half(y1);
                }
                if (rowsLeft > 2u) {
                    const float y2 = accM[tm].z + ((filterMode == 2u) ? out[base + 2u] : (g * shared[base + 2u]));
                    out[base + 2u] = y2;
                    if (hasOut16 != 0u) out16[base + 2u] = half(y2);
                }
                if (rowsLeft > 3u) {
                    const float y3 = accM[tm].w + ((filterMode == 2u) ? out[base + 3u] : (g * shared[base + 3u]));
                    out[base + 3u] = y3;
                    if (hasOut16 != 0u) out16[base + 3u] = half(y3);
                }
            }
        }
        return;
    }
    float4 acc = float4(0.0f);
    if (nBlocks <= 32u && rowsLeft >= 4u) {
        const bool active = (lane < nBlocks);
        for (uint k = 0; k < realE; k++) {
            const uint slot = t * realE + k;
            const int rawExpSigned = ids[slot];
            if (rawExpSigned < 0) continue;
            const uint rawExp = uint(rawExpSigned);
            const bool isCold = (residentE > 0u) && (rawExp >= residentE);
            if ((filterMode == 1u && isCold) || (filterMode == 2u && !isCold)) continue;
            const uint wExp = isCold ? (rawExp - residentE) : rawExp;
            device const Block_HQ4_G32* wSrc = isCold ? coldBlocks : blocks;
            const uint rowBase = wExp * N + r0;
            float4 vx = float4(0.0f);
            if (active) {
                device const float* xg = x + (size_t)slot * K + (lane << 5u);
                float4 lo[4], hi[4];
                #pragma unroll
                for (uint w0 = 0; w0 < 4; ++w0) {
                    device const float4* p4 = (device const float4*)(xg + w0 * 8u);
                    lo[w0] = p4[0];
                    hi[w0] = p4[1];
                }
                device const Block_HQ4_G32* rPtr = wSrc + (size_t)rowBase * nBlocks + lane;
                vx.x = qwg_dot_hq4_block_f32_reg(rPtr[0], lo, hi);
                vx.y = qwg_dot_hq4_block_f32_reg(rPtr[nBlocks], lo, hi);
                vx.z = qwg_dot_hq4_block_f32_reg(rPtr[2u * nBlocks], lo, hi);
                vx.w = qwg_dot_hq4_block_f32_reg(rPtr[3u * nBlocks], lo, hi);
            }
            vx += simd_shuffle_down(vx, 16);
            vx += simd_shuffle_down(vx, 8);
            vx += simd_shuffle_down(vx, 4);
            vx += simd_shuffle_down(vx, 2);
            vx += simd_shuffle_down(vx, 1);
            if (routeOutput) {
                if (lane == 0u) qwg_moe_write_route4(out, slot, N, r0, rowsLeft, vx);
            } else {
                acc += w[slot] * vx;
            }
        }
    } else if ((nBlocks & 15u) == 0u && rowsLeft >= 4u) {
        const uint upHalf = lane >> 4u, hl = lane & 15u;
        for (uint k = 0; k < realE; k++) {
            const uint slot = t * realE + k;
            const int rawExpSigned = ids[slot];
            if (rawExpSigned < 0) continue;
            const uint rawExp = uint(rawExpSigned);
            const bool isCold = (residentE > 0u) && (rawExp >= residentE);
            if ((filterMode == 1u && isCold) || (filterMode == 2u && !isCold)) continue;
            const uint wExp = isCold ? (rawExp - residentE) : rawExp;
            device const Block_HQ4_G32* wSrc = isCold ? coldBlocks : blocks;
            const uint rowBase = wExp * N + r0 + upHalf * 2u;
            device const Block_HQ4_G32* rPtr0 = wSrc + (size_t)rowBase * nBlocks;
            device const Block_HQ4_G32* rPtr1 = rPtr0 + nBlocks;
            device const float* xs = x + (size_t)slot * K;
            float2 vx = float2(0.0f);
            for (uint g = hl; g < nBlocks; g += 16u) {
                device const float* xg = xs + (g << 5u);
                float4 lo[4], hi[4];
                #pragma unroll
                for (uint w0 = 0; w0 < 4; ++w0) {
                    device const float4* p4 = (device const float4*)(xg + w0 * 8u);
                    lo[w0] = p4[0];
                    hi[w0] = p4[1];
                }
                vx.x += qwg_dot_hq4_block_f32_reg(rPtr0[g], lo, hi);
                vx.y += qwg_dot_hq4_block_f32_reg(rPtr1[g], lo, hi);
            }
            vx += simd_shuffle_down(vx, 8);
            vx += simd_shuffle_down(vx, 4);
            vx += simd_shuffle_down(vx, 2);
            vx += simd_shuffle_down(vx, 1);
            const float2 vxHi = simd_shuffle_down(vx, 16);
            const float4 value = float4(vx.x, vx.y, vxHi.x, vxHi.y);
            if (routeOutput) {
                if (lane == 0u) qwg_moe_write_route4(out, slot, N, r0, rowsLeft, value);
            } else {
                acc += w[slot] * value;
            }
        }
    } else {
        for (uint k = 0; k < realE; k++) {
            const uint slot = t * realE + k;
            const int rawExpSigned = ids[slot];
            if (rawExpSigned < 0) continue;
            const uint rawExp = uint(rawExpSigned);
            const bool isCold = (residentE > 0u) && (rawExp >= residentE);
            if ((filterMode == 1u && isCold) || (filterMode == 2u && !isCold)) continue;
            const uint wExp = isCold ? (rawExp - residentE) : rawExp;
            device const Block_HQ4_G32* wSrc = isCold ? coldBlocks : blocks;
            const float4 vx = qwg_moe_rows4_f32(wSrc, x + (size_t)slot * K, wExp * N + r0, rowsLeft, K, lane);
            if (routeOutput) {
                if (lane == 0u) qwg_moe_write_route4(out, slot, N, r0, rowsLeft, vx);
            } else {
                acc += w[slot] * vx;
            }
        }
    }
    if (routeOutput) return;
    if (lane == 0u) {
        const size_t base = (size_t)t * N + r0;
        const float g = gate[t];
        const float y0 = acc.x + ((filterMode == 2u) ? out[base] : (g * shared[base]));
        out[base] = y0;
        if ((hasOut16 & 1u) != 0u) out16[base] = half(y0);
        if (rowsLeft > 1u) {
            const float y1 = acc.y + ((filterMode == 2u) ? out[base + 1u] : (g * shared[base + 1u]));
            out[base + 1u] = y1;
            if ((hasOut16 & 1u) != 0u) out16[base + 1u] = half(y1);
        }
        if (rowsLeft > 2u) {
            const float y2 = acc.z + ((filterMode == 2u) ? out[base + 2u] : (g * shared[base + 2u]));
            out[base + 2u] = y2;
            if ((hasOut16 & 1u) != 0u) out16[base + 2u] = half(y2);
        }
        if (rowsLeft > 3u) {
            const float y3 = acc.w + ((filterMode == 2u) ? out[base + 3u] : (g * shared[base + 3u]));
            out[base + 3u] = y3;
            if ((hasOut16 & 1u) != 0u) out16[base + 3u] = half(y3);
        }
    }
}
