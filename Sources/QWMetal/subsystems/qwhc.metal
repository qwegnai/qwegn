// qwhc.metal — Hyper-connection mixer (combine=True), f32 end to end.
//   xn = groupRMSNorm(x, groups of hidden, (1+w))            (hc groups)
//   low = silu(down . xn / hc);  gate = sigmoid(up . low);   mixed[h] = sum_c gate[c,h] * xn[c,h] / hc
//   injection[c] = 2 * sigmoid(inject . xn / hc)              (attention and MLP sublayers only)
#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_dequant.metalh"
#include "qw_dequant_f32.metalh"
#include "qw_dequant_multi.metalh"
#include "qwhc_common.metalh"
using namespace metal;

inline void qwg_hc_norm_impl(device const float* x, device const float* w, device float* xn,
                             uint hidden, float eps, uint width, uint xStride,
                             uint tg, uint tid, threadgroup float* partial) {
    const uint groups = width / hidden;
    const uint t = tg / groups, g = tg - t * groups;
    const uint inBase = t * xStride + g * hidden;
    const uint outBase = t * width + g * hidden;
    float ss = 0.0f;
    for (uint k = tid; k < hidden; k += 256u) { const float v = x[inBase + k]; ss += v * v; }
    ss = simd_sum(ss);
    if ((tid & 31u) == 0u) partial[tid >> 5u] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0u; i < 8u; i++) total += partial[i];
    const float inv = 1.0f / sqrt(total / float(hidden) + eps);
    for (uint k = tid; k < hidden; k += 256u) xn[outBase + k] = x[inBase + k] * inv * (1.0f + w[g * hidden + k]);
}

kernel void qwg_hc_norm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                        device float* xn [[buffer(2)]], constant uint& hidden [[buffer(3)]],
                        constant float& eps [[buffer(4)]], constant uint& width [[buffer(5)]],
                        uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[8];
    qwg_hc_norm_impl(x, w, xn, hidden, eps, width, width, tg, tid, partial);
}

kernel void qwg_hc_norm_strided(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                                device float* xn [[buffer(2)]], constant uint& hidden [[buffer(3)]],
                                constant float& eps [[buffer(4)]], constant uint& width [[buffer(5)]],
                                constant uint& xStride [[buffer(6)]],
                                uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[8];
    qwg_hc_norm_impl(x, w, xn, hidden, eps, width, xStride, tg, tid, partial);
}

// Fused hc_mix + 32-point WHT (mixed -> rotMixed) + optional hc.inject BF16 dot across T rows.
kernel void qwg_hc_mix_h32(device const float*  gate       [[buffer(0)]],
                           device const float*  xn         [[buffer(1)]],
                           device float*        mixed      [[buffer(2)]],
                           device float*        rotMixed   [[buffer(3)]],
                           constant uint&       hidden     [[buffer(4)]],
                           constant uint&       hc         [[buffer(5)]],
                           device half*         rotMixed16 [[buffer(6)]],
                           constant uint&       hasOut16   [[buffer(7)]],
                           device const ushort* injW       [[buffer(8)]],
                           device float*        yInj       [[buffer(9)]],
                           device const float*  xs         [[buffer(10)]],
                           device const float*  normW      [[buffer(11)]],
                           uint2 tg   [[threadgroup_position_in_grid]],
                           uint  sg   [[simdgroup_index_in_threadgroup]],
                           uint  lane [[thread_index_in_simdgroup]]) {
    const uint tid = sg * 32u + lane, t = tg.y, width = hc * hidden;
    const bool writeOut16 = (hasOut16 & 1u) != 0u;
    const bool applySig   = (hasOut16 & 2u) != 0u;
    const bool gateIsF16  = (hasOut16 & 4u) != 0u;
    const bool skipRot32  = (hasOut16 & 8u) != 0u;
    const bool fuseInj    = (hasOut16 & 16u) != 0u;
    // xn holds per-(row, stream) rinv; recompute xn = x * rinv * (1 + w) exactly as qwg_hc_norm_h32 does.
    const bool recompute  = (hasOut16 & 32u) != 0u;
    device const half* gate16 = (device const half*)gate;
    float rinv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (recompute) {
        for (uint c = 0u; c < hc && c < 4u; c++) rinv[c] = xn[(size_t)t * hc + c];
    }
    if (fuseInj) {
        threadgroup float4 partial[8];
        float4 injAcc = float4(0.0f);
        const float invHC = 1.0f / float(hc);
        for (uint h = tid; h < hidden; h += 256u) {
            const size_t rowOff = (size_t)t * width;
            float acc = 0.0f;
            for (uint c = 0u; c < hc; c++) {
                const size_t col = c * hidden + h;
                const float xnv = recompute ? (xs[rowOff + col] * rinv[c] * (1.0f + normW[col])) : xn[rowOff + col];
                float gVal = gateIsF16 ? float(gate16[rowOff + col]) : gate[rowOff + col];
                if (applySig) gVal = 1.0f / (1.0f + exp(-gVal));
                acc = fma(gVal, xnv, acc);
                const float w0 = as_type<float>(uint(injW[0u * width + col]) << 16u);
                const float w1 = (hc > 1u) ? as_type<float>(uint(injW[1u * width + col]) << 16u) : 0.0f;
                const float w2 = (hc > 2u) ? as_type<float>(uint(injW[2u * width + col]) << 16u) : 0.0f;
                const float w3 = (hc > 3u) ? as_type<float>(uint(injW[3u * width + col]) << 16u) : 0.0f;
                injAcc = fma(float4(w0, w1, w2, w3), float4(xnv), injAcc);
            }
            float v = acc * invHC;
            const size_t outIdx = (size_t)t * hidden + h;
            mixed[outIdx] = v;
            v = qwg_fwht32_lane_f32(v, lane);
            if (!skipRot32) rotMixed[outIdx] = v;
            if (writeOut16) rotMixed16[outIdx] = half(v);
        }
        injAcc = float4(simd_sum(injAcc.x), simd_sum(injAcc.y), simd_sum(injAcc.z), simd_sum(injAcc.w));
        if (lane == 0u) partial[sg] = injAcc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < hc && tid < 4u) {
            float4 tot = float4(0.0f);
            for (uint i = 0u; i < 8u; i++) tot += partial[i];
            const float vals[4] = { tot.x, tot.y, tot.z, tot.w };
            yInj[(size_t)t * hc + tid] = 2.0f / (1.0f + exp(-(vals[tid] * invHC)));
        }
        return;
    }
    const uint g = tg.x * 8u + sg;
    const uint h = g * 32u + lane;
    if (h >= hidden) return;
    const size_t base = (size_t)t * width + h;
    float acc = 0.0f;
    for (uint c = 0u; c < hc; c++) {
        float gVal = gateIsF16 ? float(gate16[base + c * hidden]) : gate[base + c * hidden];
        if (applySig) gVal = 1.0f / (1.0f + exp(-gVal));
        const float xnv = recompute ? (xs[base + c * hidden] * rinv[c] * (1.0f + normW[c * hidden + h])) : xn[base + c * hidden];
        acc += gVal * xnv;
    }
    float v = acc / float(hc);
    const size_t outIdx = (size_t)t * hidden + h;
    mixed[outIdx] = v;
    v = qwg_fwht32_lane_f32(v, lane);
    const float rotVal = v;
    if (!skipRot32) rotMixed[outIdx] = rotVal;
    if (writeOut16) rotMixed16[outIdx] = half(rotVal);
}

// Fused H_32(hcLow) + hc.up HQ4 [hc*hidden, R] + sigmoid + 4-stream hc_mix + H_32(mixed -> rotMixed).
// Dispatch: (hidden / 32, T, 1) threadgroups of 256 threads; R <= 512 and R % 32 == 0.
// Each 32-lane simdgroup splits into two 16-lane half-simdgroups to process stream pairs (0,1) and (2,3) concurrently.
kernel void qwg_hc_up_mix_h32_q4g_f32(device const Block_HQ4_G32* upBlocks [[buffer(0)]],
                                      device const float*         hcLow    [[buffer(1)]],
                                      device const float*         xn       [[buffer(2)]],
                                      device float*               mixed    [[buffer(3)]],
                                      device float*               rotMixed [[buffer(4)]],
                                      constant uint&              hidden   [[buffer(5)]],
                                      constant uint&              hc       [[buffer(6)]],
                                      constant uint&              R        [[buffer(7)]],
                                      uint2 tg   [[threadgroup_position_in_grid]],
                                      uint  sg   [[simdgroup_index_in_threadgroup]],
                                      uint  lane [[thread_index_in_simdgroup]]) {
    threadgroup float sRot[512];
    threadgroup float sMix[32];
    const uint b = tg.x;
    const uint realHC = hc & 0xFFu;
    const uint nBlocks = R >> 5u;
    const uint halfId = lane >> 4u;
    const uint g = lane & 15u;
    const uint h0 = b * 32u + sg * 4u;
    const uint t = tg.y;
    for (uint jb = sg; jb < nBlocks; jb += 8u) {
        float v = hcLow[(size_t)t * R + jb * 32u + lane];
        v = qwg_fwht32_lane_f32(v, lane);
        sRot[jb * 32u + lane] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 lo[4], hi[4];
    if (g < nBlocks) {
        threadgroup const float* xg = sRot + (g << 5u);
        #pragma unroll
        for (uint w = 0u; w < 4u; ++w) {
            threadgroup const float4* p4 = (threadgroup const float4*)(xg + w * 8u);
            lo[w] = p4[0];
            hi[w] = p4[1];
        }
    }
    float4 prodEven = float4(0.0f);
    float4 prodOdd = float4(0.0f);
    #pragma unroll
    for (uint p = 0u; p < 2u; ++p) {
        const uint c = halfId * 2u + p;
        float4 u = float4(0.0f);
        if (g < nBlocks && c < realHC) {
            const size_t rowBase = (size_t)(c * hidden + h0) * nBlocks + g;
            u.x = qwg_dot_hq4_block_f32_reg(upBlocks[rowBase], lo, hi);
            u.y = qwg_dot_hq4_block_f32_reg(upBlocks[rowBase + nBlocks], lo, hi);
            u.z = qwg_dot_hq4_block_f32_reg(upBlocks[rowBase + 2u * nBlocks], lo, hi);
            u.w = qwg_dot_hq4_block_f32_reg(upBlocks[rowBase + 3u * nBlocks], lo, hi);
        }
        u += simd_shuffle_down(u, 8);
        u += simd_shuffle_down(u, 4);
        u += simd_shuffle_down(u, 2);
        u += simd_shuffle_down(u, 1);
        if (g == 0u && c < realHC) {
            const float4 gate = 1.0f / (1.0f + exp(-u));
            const float4 xnc = *(device const float4*)(xn + ((size_t)t * realHC + c) * hidden + h0);
            if (p == 0u) prodEven = gate * xnc;
            else prodOdd = gate * xnc;
        }
    }
    const float4 p2 = simd_shuffle_down(prodEven, 16);
    const float4 p3 = simd_shuffle_down(prodOdd, 16);
    if (lane == 0u) {
        const float4 acc = ((prodEven + prodOdd) + p2) + p3;
        const float4 mv = acc / float(realHC);
        *(device float4*)(mixed + (size_t)t * hidden + h0) = mv;
        sMix[sg * 4u + 0u] = mv.x;
        sMix[sg * 4u + 1u] = mv.y;
        sMix[sg * 4u + 2u] = mv.z;
        sMix[sg * 4u + 3u] = mv.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float v = sMix[lane];
        v = qwg_fwht32_lane_f32(v, lane);
        rotMixed[(size_t)t * hidden + b * 32u + lane] = v;
    }
}

// hc_norm followed by the 32-point WHT of the normalized row, in one dispatch. One threadgroup per (token, group):
// the norm writes xn and rotates v directly in registers before writing rot (zero device barrier or xn re-read).
kernel void qwg_hc_norm_h32(device const float* x        [[buffer(0)]],
                            device const float* w        [[buffer(1)]],
                            device float*       xn       [[buffer(2)]],
                            device float*       rot      [[buffer(3)]],
                            constant uint&      hidden   [[buffer(4)]],
                            constant float&     eps      [[buffer(5)]],
                            constant uint&      width    [[buffer(6)]],
                            device half*        rot16    [[buffer(7)]],
                            constant uint&      hasRot16 [[buffer(8)]],
                            uint tg  [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[8];
    const uint groups = width / hidden;
    const uint t = tg / groups, g = tg - t * groups;
    const uint base = t * width + g * hidden;
    const uint lane = tid & 31u;
    const bool rinvOnly = (hasRot16 & 2u) != 0u;
    if (hidden == 2560u) {
        float regV[10];
        float ss = 0.0f;
        #pragma unroll
        for (uint i = 0u; i < 10u; i++) {
            const float v = x[base + tid + i * 256u];
            regV[i] = v;
            ss += v * v;
        }
        ss = simd_sum(ss);
        if (lane == 0u) partial[tid >> 5u] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint i = 0u; i < 8u; i++) total += partial[i];
        const float inv = 1.0f / sqrt(total / 2560.0f + eps);
        if (rinvOnly && tid == 0u) xn[t * groups + g] = inv;
        #pragma unroll
        for (uint i = 0u; i < 10u; i++) {
            const uint k = tid + i * 256u;
            float v = regV[i] * inv * (1.0f + w[g * 2560u + k]);
            if (!rinvOnly) xn[base + k] = v;
            v = qwg_fwht32_lane_f32(v, lane);
            if ((hasRot16 & 1u) != 0u) rot16[base + k] = half(v);
            else rot[base + k] = v;
        }
        return;
    } else if (hidden <= 4096u) {
        float regV[16];
        float ss = 0.0f;
        #pragma unroll
        for (uint i = 0u; i < 16u; i++) {
            const uint k = tid + i * 256u;
            if (k >= hidden) break;
            const float v = x[base + k];
            regV[i] = v;
            ss += v * v;
        }
        ss = simd_sum(ss);
        if (lane == 0u) partial[tid >> 5u] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint i = 0u; i < 8u; i++) total += partial[i];
        const float inv = 1.0f / sqrt(total / float(hidden) + eps);
        if (rinvOnly && tid == 0u) xn[t * groups + g] = inv;
        #pragma unroll
        for (uint i = 0u; i < 16u; i++) {
            const uint k = tid + i * 256u;
            if (k >= hidden) break;
            float v = regV[i] * inv * (1.0f + w[g * hidden + k]);
            if (!rinvOnly) xn[base + k] = v;
            v = qwg_fwht32_lane_f32(v, lane);
            if ((hasRot16 & 1u) != 0u) rot16[base + k] = half(v);
            else rot[base + k] = v;
        }
        return;
    }
    float ss = 0.0f;
    for (uint k = tid; k < hidden; k += 256u) { const float v = x[base + k]; ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0u) partial[tid >> 5u] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0u; i < 8u; i++) total += partial[i];
    const float inv = 1.0f / sqrt(total / float(hidden) + eps);
        if (rinvOnly && tid == 0u) xn[t * groups + g] = inv;
    for (uint k = tid; k < hidden; k += 256u) {
        float v = x[base + k] * inv * (1.0f + w[g * hidden + k]);
        if (!rinvOnly) xn[base + k] = v;
        v = qwg_fwht32_lane_f32(v, lane);
        if ((hasRot16 & 1u) != 0u) rot16[base + k] = half(v);
        else rot[base + k] = v;
    }
}

// Fused residual injection (xOut = base + src * inj) + hc_norm + 32-point WHT (xn, rot) in one dispatch.
static inline void qwg_hc_inject_norm_h32_impl(device const float* base,
                                               device const float* src,
                                               device const float* inj,
                                               device float*       xOut,
                                               device const float* w,
                                               device float*       xn,
                                               device float*       rot,
                                               uint hidden, float eps, uint width,
                                               uint tg, uint tid,
                                               threadgroup float* partial,
                                               device half* rot16,
                                               uint hasRot16) {
    const uint groups = width / hidden;
    const uint t = tg / groups, g = tg - t * groups;
    const uint off = t * width + g * hidden;
    const uint lane = tid & 31u;
    const bool rinvOnly = (hasRot16 & 2u) != 0u;
    const float scale = inj[t * groups + g];
    device const float* srcT = src + t * hidden;
    if (hidden == 2560u) {
        float regV[10];
        float ss = 0.0f;
        #pragma unroll
        for (uint i = 0u; i < 10u; i++) {
            const uint k = tid + i * 256u;
            const float v = base[off + k] + srcT[k] * scale;
            xOut[off + k] = v;
            regV[i] = v;
            ss += v * v;
        }
        ss = simd_sum(ss);
        if (lane == 0u) partial[tid >> 5u] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint i = 0u; i < 8u; i++) total += partial[i];
        const float inv = 1.0f / sqrt(total / 2560.0f + eps);
        if (rinvOnly && tid == 0u) xn[t * groups + g] = inv;
        #pragma unroll
        for (uint i = 0u; i < 10u; i++) {
            const uint k = tid + i * 256u;
            float v = regV[i] * inv * (1.0f + w[g * 2560u + k]);
            if (!rinvOnly) xn[off + k] = v;
            v = qwg_fwht32_lane_f32(v, lane);
            if ((hasRot16 & 1u) != 0u) rot16[off + k] = half(v);
            else rot[off + k] = v;
        }
        return;
    } else if (hidden <= 4096u) {
        float regV[16];
        float ss = 0.0f;
        #pragma unroll
        for (uint i = 0u; i < 16u; i++) {
            const uint k = tid + i * 256u;
            if (k >= hidden) break;
            const float v = base[off + k] + srcT[k] * scale;
            xOut[off + k] = v;
            regV[i] = v;
            ss += v * v;
        }
        ss = simd_sum(ss);
        if (lane == 0u) partial[tid >> 5u] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint i = 0u; i < 8u; i++) total += partial[i];
        const float inv = 1.0f / sqrt(total / float(hidden) + eps);
        if (rinvOnly && tid == 0u) xn[t * groups + g] = inv;
        #pragma unroll
        for (uint i = 0u; i < 16u; i++) {
            const uint k = tid + i * 256u;
            if (k >= hidden) break;
            float v = regV[i] * inv * (1.0f + w[g * hidden + k]);
            if (!rinvOnly) xn[off + k] = v;
            v = qwg_fwht32_lane_f32(v, lane);
            if ((hasRot16 & 1u) != 0u) rot16[off + k] = half(v);
            else rot[off + k] = v;
        }
        return;
    }
    float ss = 0.0f;
    for (uint k = tid; k < hidden; k += 256u) {
        const float v = base[off + k] + srcT[k] * scale;
        xOut[off + k] = v;
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (lane == 0u) partial[tid >> 5u] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0u; i < 8u; i++) total += partial[i];
    const float inv = 1.0f / sqrt(total / float(hidden) + eps);
        if (rinvOnly && tid == 0u) xn[t * groups + g] = inv;
    for (uint k = tid; k < hidden; k += 256u) {
        float v = xOut[off + k] * inv * (1.0f + w[g * hidden + k]);
        if (!rinvOnly) xn[off + k] = v;
        v = qwg_fwht32_lane_f32(v, lane);
        if ((hasRot16 & 1u) != 0u) rot16[off + k] = half(v);
        else rot[off + k] = v;
    }
}

kernel void qwg_hc_inject_norm_h32(device const float* base     [[buffer(0)]],
                                   device const float* src      [[buffer(1)]],
                                   device const float* inj      [[buffer(2)]],
                                   device float*       xOut     [[buffer(3)]],
                                   device const float* w        [[buffer(4)]],
                                   device float*       xn       [[buffer(5)]],
                                   device float*       rot      [[buffer(6)]],
                                   constant uint&      hidden   [[buffer(7)]],
                                   constant float&     eps      [[buffer(8)]],
                                   constant uint&      width    [[buffer(9)]],
                                   device half*        rot16    [[buffer(10)]],
                                   constant uint&      hasRot16 [[buffer(11)]],
                                   uint tg  [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[8];
    qwg_hc_inject_norm_h32_impl(base, src, inj, xOut, w, xn, rot, hidden, eps, width, tg, tid, partial, rot16, hasRot16);
}

// Same as qwg_hc_inject_norm_h32 for 1 output row, selecting the last accepted row in 0..<totalRows.
kernel void qwg_hc_inject_norm_h32_sel(device const float* base        [[buffer(0)]],
                                       device const float* src         [[buffer(1)]],
                                       device const float* inj         [[buffer(2)]],
                                       device float*       xOut        [[buffer(3)]],
                                       device const float* w           [[buffer(4)]],
                                       device float*       xn          [[buffer(5)]],
                                       device float*       rot         [[buffer(6)]],
                                       constant uint&      hidden      [[buffer(7)]],
                                       constant float&     eps         [[buffer(8)]],
                                       constant uint&      width       [[buffer(9)]],
                                       device const int*   verifiedTok [[buffer(10)]],
                                       device const int*   draftTok    [[buffer(11)]],
                                       constant uint&      totalRows   [[buffer(12)]],
                                       uint tg  [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[8];
    uint srcRow = 0u;
    for (uint r = 0u; r + 1u < totalRows; ++r) {
        if (verifiedTok[r] == draftTok[r + 1u]) {
            srcRow = r + 1u;
        } else {
            break;
        }
    }
    const uint groups = width / hidden;
    qwg_hc_inject_norm_h32_impl(base + srcRow * width, src + srcRow * hidden, inj + srcRow * groups,
                                xOut, w, xn, rot, hidden, eps, width, tg, tid, partial, nullptr, 0u);
}

// Fused silu(y / scale) + 32-point WHT over [T, R] low-rank HC activations, writing half to out16 or float to yLow.
kernel void qwg_hc_low_epilogue_f32(device float*       yLow     [[buffer(0)]],
                                    device half*        out16    [[buffer(1)]],
                                    constant uint&      T        [[buffer(2)]],
                                    constant uint&      R        [[buffer(3)]],
                                    constant float&     scale    [[buffer(4)]],
                                    constant uint&      hasOut16 [[buffer(5)]],
                                    uint2 tg   [[threadgroup_position_in_grid]],
                                    uint  sg   [[simdgroup_index_in_threadgroup]],
                                    uint  lane [[thread_index_in_simdgroup]]) {
    const uint t = tg.y * 8u + sg;
    const uint col = tg.x * 32u + lane;
    if (t >= T || col >= R) return;
    const size_t idx = (size_t)t * R + col;
    const float z = yLow[idx] / scale;
    float v = z / (1.0f + exp(-z));
    v = qwg_fwht32_lane_f32(v, lane);
    if (hasOut16 != 0u) {
        out16[idx] = half(v);
    } else {
        yLow[idx] = v;
    }
}

// Fused decode (T=1) hc.down HQ4 [NDown, K] (silu(y / scale)) + hc.inject BF16 [NInj, K] (2 * sigmoid(y / scale)).
// Dispatch: (nDownTG + NInj, 1, 1) threadgroups of 128 threads (4 simdgroups), where nDownTG = (NDown + 15) / 16.
// TGs 0..<nDownTG compute 16 HQ4 rows each (zero barrier); TGs nDownTG..<nDownTG+NInj compute 1 BF16 row each.
kernel void qwg_hc_down_inject_f32(device const Block_HQ4_G32* downBlocks [[buffer(0)]],
                                   device const float*         xRot       [[buffer(1)]],
                                   device float*               yLow       [[buffer(2)]],
                                   device const ushort*        injW       [[buffer(3)]],
                                   device const float*         xXn        [[buffer(4)]],
                                   device float*               yInj       [[buffer(5)]],
                                   constant uint&              NDown      [[buffer(6)]],
                                   constant uint&              K          [[buffer(7)]],
                                   constant float&             scale      [[buffer(8)]],
                                   constant uint&              CH         [[buffer(9)]],
                                   uint2 tgp [[threadgroup_position_in_grid]],
                                   uint sg   [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    const uint tg = tgp.x, rw = tgp.y, realCH = CH & 0xFFu, nDownTG = (NDown + 3u) >> 2u;
    threadgroup float4 part4[4];
    if (tg < nDownTG) {
        const uint row0 = tg * 4u;
        const float4 acc = qwg_gemv_hq4_rows4_part_f32(downBlocks, xRot + rw * K, row0, NDown, K, sg * 32u + lane, 128u);
        qwg_hc_down_rows4_store(acc, yLow, NDown, rw, row0, scale, sg, lane, part4);
        return;
    }
    qwg_hc_inject_bf16_f32_impl(injW, xXn, yInj, K, scale, realCH, rw, tg - nDownTG, sg, lane, (threadgroup float*)part4);
}
