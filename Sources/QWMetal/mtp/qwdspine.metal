// ── QWDSpine — DSpine adjacent causal code bridge & refinement tail kernels ──
// Custom fused Metal kernels for DSpine speculative decoding: code L2-norm +
// anchor codebook gather, adjacent causal gate message, multi-row bridge write
// projection + scaled residual update, and 16-state exact-sender refinement.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_dequant.metalh"
#include "include/qw_dequant_shvq.metalh"
using namespace metal;

struct DSpineH8 { half v[8]; } __attribute__((__packed__, __aligned__(16)));
static inline DSpineH8 asDSpineH8(uint4 w) { return *(thread DSpineH8*)&w; }
static inline float dspine_bf16_to_f32(ushort b) { return as_type<float>(uint(b) << 16u); }

// Kernel 1: Code bridge state transform (32 threads/TG).
// mode 0: anchor codebook gather (tg=0) + L2 code normalization (tg>0) -> bridgeCodes [M, C].
// mode 1: adjacent causal sigmoid gate message (t-1 -> t) -> bridgeMsg [D, R].
kernel void qwg_dspine_bridge_state(
        device const float* projOrPacked   [[buffer(0)]],
        device const ushort* codebookOrBias [[buffer(1)]],
        device half* outBuf                [[buffer(2)]],
        constant int& anchorToken          [[buffer(3)]],
        constant uint& dim                 [[buffer(4)]],
        constant float& eps                [[buffer(5)]],
        constant uint& mode                [[buffer(6)]],
        uint tg [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    if (mode == 0u) {
        const float scale = sqrt(float(dim));
        device half* dst = outBuf + (size_t)tg * dim;
        if (tg == 0u) {
            device const ushort* cb = codebookOrBias + (size_t)uint(anchorToken) * dim;
            for (uint i = lane; i < dim; i += 32u) {
                dst[i] = half(scale * dspine_bf16_to_f32(cb[i]));
            }
        } else {
            device const float* src = projOrPacked + (size_t)(tg - 1u) * dim;
            float sq = 0.0f;
            for (uint i = lane; i < dim; i += 32u) {
                float v = src[i];
                sq += v * v;
            }
            const float invNorm = scale / (sqrt(simd_sum(sq)) + eps);
            for (uint i = lane; i < dim; i += 32u) {
                dst[i] = half(src[i] * invNorm);
            }
        }
    } else {
        const uint d = tg;
        const uint stride = 3u * dim;
        device const float* qRow = projOrPacked + (size_t)(d + 1u) * stride;
        device const float* sPrev = projOrPacked + (size_t)d * stride + dim;
        device const float* vPrev = sPrev + dim;
        device const half* bias = (device const half*)codebookOrBias;
        device half* dst = outBuf + (size_t)d * dim;
        for (uint r = lane; r < dim; r += 32u) {
            float g = 1.0f / (1.0f + exp(-(qRow[r] + sPrev[r] + float(bias[r]))));
            dst[r] = half(g * vPrev[r]);
        }
    }
}

// Kernel 2: Fused multi-row bridge write projection [H, R] + scaled residual update.
// Streams W_write [H, R] once across D=1..15 rows and writes rawOut = rawIn + rms * delta.
// Grid = (H / 4, 1, 1), threadsPerThreadgroup = (128, 1, 1).
kernel void qwg_dspine_bridge_write_add(
        device const uint4* W        [[buffer(0)]],
        device const uint4* msg      [[buffer(1)]],
        device const float* rawIn    [[buffer(2)]],
        device const float* rmsIn    [[buffer(3)]],
        device float* rawOut         [[buffer(4)]],
        constant uint& H             [[buffer(5)]],
        constant uint& R             [[buffer(6)]],
        constant uint& D             [[buffer(7)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    const uint row = tg * 4u + simd_id;
    if (row >= H) return;
    const uint chunks = R >> 3u;
    device const uint4* wr = W + (size_t)row * chunks;
    float acc[16] = {0.0f};
    for (uint c = lane; c < chunks; c += 32u) {
        DSpineH8 wv = asDSpineH8(wr[c]);
        for (uint d = 0; d < D; ++d) {
            DSpineH8 mv = asDSpineH8(msg[(size_t)d * chunks + c]);
            float s = 0.0f;
            for (int i = 0; i < 8; ++i) s += float(wv.v[i]) * float(mv.v[i]);
            acc[d] += s;
        }
    }
    for (uint d = 0; d < D; ++d) {
        float val = simd_sum(acc[d]);
        if (lane == 0u) {
            size_t idx = (size_t)d * H + row;
            rawOut[idx] = rawIn[idx] + rmsIn[d] * val;
        }
    }
}

// Kernel 3: Fused candidate codebook gather + [G_p; A] projection + sigmoid gate message.
// Grid = (R / 4, D, 1), threadsPerThreadgroup = (128, 1, 1).
kernel void qwg_dspine_refine_code_to_msg(
        device const uint4* packedWeight [[buffer(0)]],
        device const ushort* codebook    [[buffer(1)]],
        device const int* candidates     [[buffer(2)]],
        device const float* bridgePacked [[buffer(3)]],
        device const half* gateBias      [[buffer(4)]],
        device half* refineMsg           [[buffer(5)]],
        constant uint& R                 [[buffer(6)]],
        constant uint& C                 [[buffer(7)]],
        constant uint& topK              [[buffer(8)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    const uint r = tg.x * 4u + simd_id;
    const uint d = tg.y;
    if (r >= R) return;

    const float qVal = bridgePacked[(size_t)(d + 1u) * (3u * R) + r];
    const float bVal = float(gateBias[r]);
    if (d == 0u) {
        if (lane == 0u) {
            float s0 = bridgePacked[R + r];
            float v0 = bridgePacked[2u * R + r];
            float g0 = 1.0f / (1.0f + exp(-(qVal + s0 + bVal)));
            refineMsg[r] = half(g0 * v0);
            for (uint k = 1u; k < topK; ++k) refineMsg[(size_t)k * R + r] = half(0.0f);
        }
        return;
    }

    const uint chunks = C >> 3u;
    const float sqrtC = sqrt(float(C));
    device const uint4* wGp = packedWeight + (size_t)(R + r) * chunks;
    device const uint4* wPrev = packedWeight + (size_t)(2u * R + r) * chunks;
    device const int* predIds = candidates + (size_t)(d - 1u) * 16u;

    float accGp[16] = {0.0f};
    float accPrev[16] = {0.0f};
    for (uint c = lane; c < chunks; c += 32u) {
        DSpineH8 gp8 = asDSpineH8(wGp[c]);
        DSpineH8 pv8 = asDSpineH8(wPrev[c]);
        const uint colOff = c << 3u;
        for (uint k = 0; k < topK; ++k) {
            uint sid = uint(predIds[k]);
            device const ushort* cb = codebook + (size_t)sid * C + colOff;
            float sGp = 0.0f, sPv = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float e = dspine_bf16_to_f32(cb[i]);
                sGp += float(gp8.v[i]) * e;
                sPv += float(pv8.v[i]) * e;
            }
            accGp[k] += sGp;
            accPrev[k] += sPv;
        }
    }
    for (uint k = 0; k < topK; ++k) {
        float s = simd_sum(accGp[k]) * sqrtC;
        float v = simd_sum(accPrev[k]) * sqrtC;
        if (lane == 0u) {
            float g = 1.0f / (1.0f + exp(-(qVal + s + bVal)));
            refineMsg[((size_t)d * 16u + k) * R + r] = half(g * v);
        }
    }
}

// Kernel 4: Fused 16-state layer-4 bridge write projection + scaled residual update.
// Grid = (H / 4, D, 1), threadsPerThreadgroup = (128, 1, 1).
kernel void qwg_dspine_refine_write(
        device const uint4* W          [[buffer(0)]],
        device const uint4* refineMsg  [[buffer(1)]],
        device const float* rawIn      [[buffer(2)]],
        device const float* rmsIn      [[buffer(3)]],
        device float* refineStates     [[buffer(4)]],
        constant uint& H               [[buffer(5)]],
        constant uint& R               [[buffer(6)]],
        constant uint& topK            [[buffer(7)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    const uint row = tg.x * 4u + simd_id;
    const uint d = tg.y;
    if (row >= H) return;

    const uint chunks = R >> 3u;
    const uint kLimit = (d == 0u) ? 1u : topK;
    device const uint4* wr = W + (size_t)row * chunks;
    device const uint4* msgBase = refineMsg + (size_t)d * 16u * chunks;
    float acc[16] = {0.0f};
    for (uint c = lane; c < chunks; c += 32u) {
        DSpineH8 wv = asDSpineH8(wr[c]);
        for (uint k = 0; k < kLimit; ++k) {
            DSpineH8 mv = asDSpineH8(msgBase[(size_t)k * chunks + c]);
            float s = 0.0f;
            for (int i = 0; i < 8; ++i) s += float(wv.v[i]) * float(mv.v[i]);
            acc[k] += s;
        }
    }
    const float baseRaw = rawIn[(size_t)d * H + row];
    const float rmsVal = rmsIn[d];
    for (uint k = 0; k < kLimit; ++k) {
        float val = simd_sum(acc[k]);
        if (lane == 0u) {
            refineStates[((size_t)d * 16u + k) * H + row] = baseRaw + rmsVal * val;
        }
    }
}

// Kernel 5: Fused 16-state indexed target LM head scorer (HQ4 / AQ4 / SHVQ / F16).
// Grid = (16, D, 1), threadsPerThreadgroup = (32, 1, 1).
template <bool SUB2>
static inline void qwg_dspine_refine_scores_impl(
        device const void* headWeights,
        device const half* refineAct,
        device const int* candidates,
        device float* refineTable,
        uint H,
        uint topK,
        uint headMode,
        uint2 tg,
        uint lane) {
    const uint kCurr = tg.x;
    const uint d = tg.y;
    if (kCurr >= topK) return;

    const uint cand = uint(candidates[d * 16u + kCurr]);
    device const half* xBase = refineAct + (size_t)d * 16u * H;

    if (headMode == 0u) {
        const uint chunks = H >> 3u;
        const uint kLimit = (d == 0u) ? 1u : topK;
        device const uint4* wr = (device const uint4*)headWeights + (size_t)cand * chunks;
        device const uint4* x4 = (device const uint4*)xBase;
        float acc[16] = {0.0f};
        for (uint c = lane; c < chunks; c += 32u) {
            DSpineH8 wv = asDSpineH8(wr[c]);
            for (uint kp = 0; kp < kLimit; ++kp) {
                DSpineH8 xv = asDSpineH8(x4[(size_t)kp * chunks + c]);
                float s = 0.0f;
                for (int i = 0; i < 8; ++i) s += float(wv.v[i]) * float(xv.v[i]);
                acc[kp] += s;
            }
        }
        for (uint kp = 0; kp < kLimit; ++kp) {
            float val = simd_sum(acc[kp]);
            if (lane == 0u) refineTable[(d * 16u + kp) * 16u + kCurr] = val;
        }
    } else if ((headMode & 0x80000000u) != 0u) {
        const uint bits = headMode & 0xFFu;
        const uint group = (headMode >> 8u) & 0xFFFFu;
        const uint groupBytes = qwg_shvq_group_bytes(int(bits), int(group));
        const uint sliceBytes = qwg_shvq_slice_bytes(int(bits));
        const uint rowBytes = (H / group) * groupBytes;
        device const char* wRow = (device const char*)headWeights + (size_t)cand * rowBytes;
        const uint kLimit = (d == 0u) ? 1u : topK;
        const uint nSlices = H >> 5u;
        float acc[16] = {0.0f};
        for (uint g = lane; g < nSlices; g += 32u) {
            const uint col = g << 5u;
            device const char* gp = wRow + size_t(col / group) * groupBytes;
            const float sc = float(*(device const half*)gp);
            device const uchar* p = (device const uchar*)(gp + 2 + ((col % group) >> 5u) * sliceBytes);
            float4 wLo[4], wHi[4];
            #pragma unroll
            for (uint w = 0u; w < 4u; ++w) {
                half4 uLo, uHi;
                if constexpr (SUB2) {
                    if (bits == 1u) qwg_shvq_unpack_oct8<1>(p, w, uLo, uHi);
                    else qwg_shvq_unpack_oct8<15>(p, w, uLo, uHi);
                } else {
                    if (bits == 2u) qwg_shvq_unpack_oct8<2>(p, w, uLo, uHi);
                    else if (bits == 3u) qwg_shvq_unpack_oct8<3>(p, w, uLo, uHi);
                    else qwg_shvq_unpack_oct8<4>(p, w, uLo, uHi);
                }
                wLo[w] = float4(uLo) * sc;
                wHi[w] = float4(uHi) * sc;
            }
            for (uint kp = 0u; kp < kLimit; ++kp) {
                half4 xLo[4], xHi[4];
                qwg_shvq_load_half32(xBase + (size_t)kp * H + col, xLo, xHi);
                float s = 0.0f;
                #pragma unroll
                for (uint w = 0u; w < 4u; ++w) {
                    s += dot(wLo[w], float4(xLo[w])) + dot(wHi[w], float4(xHi[w]));
                }
                acc[kp] += s;
            }
        }
        for (uint kp = 0u; kp < kLimit; ++kp) {
            const float val = simd_sum(acc[kp]);
            if (lane == 0u) refineTable[(d * 16u + kp) * 16u + kCurr] = val;
        }
    } else if (d == 0u) {
        float out1[1];
        qwg_gemv_hq4_row_m<1>((device const Block_HQ4_G32*)headWeights, xBase, cand, H, lane, out1, headMode == 2u);
        if (lane == 0u) refineTable[kCurr] = out1[0];
    } else {
        float out16[16];
        qwg_gemv_hq4_row_m<16>((device const Block_HQ4_G32*)headWeights, xBase, cand, H, lane, out16, headMode == 2u);
        if (lane == 0u) {
            for (uint kp = 0; kp < topK; ++kp) {
                refineTable[(d * 16u + kp) * 16u + kCurr] = out16[kp];
            }
        }
    }
}

kernel void qwg_dspine_refine_scores(
        device const void* headWeights [[buffer(0)]],
        device const half* refineAct   [[buffer(1)]],
        device const int* candidates   [[buffer(2)]],
        device float* refineTable      [[buffer(3)]],
        constant uint& H               [[buffer(4)]],
        constant uint& topK            [[buffer(5)]],
        constant uint& headMode        [[buffer(6)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    qwg_dspine_refine_scores_impl<false>(headWeights, refineAct, candidates, refineTable, H, topK, headMode, tg, lane);
}

kernel void qwg_dspine_refine_scores_sub2(
        device const void* headWeights [[buffer(0)]],
        device const half* refineAct   [[buffer(1)]],
        device const int* candidates   [[buffer(2)]],
        device float* refineTable      [[buffer(3)]],
        constant uint& H               [[buffer(4)]],
        constant uint& topK            [[buffer(5)]],
        constant uint& headMode        [[buffer(6)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    qwg_dspine_refine_scores_impl<true>(headWeights, refineAct, candidates, refineTable, H, topK, headMode, tg, lane);
}

// Kernel 6: Sequential D-step transition chain walker (32 threads, single SIMD group).
// Grid = (1, 1, 1), threadsPerThreadgroup = (32, 1, 1).
kernel void qwg_dspine_refine_chain(
        device const float* refineTable [[buffer(0)]],
        device const int* candidates    [[buffer(1)]],
        device int* draftTokensOut      [[buffer(2)]],
        constant uint& draftCount       [[buffer(3)]],
        constant uint& topK             [[buffer(4)]],
        uint lane [[thread_index_in_threadgroup]]) {
    uint sender = 0u;
    for (uint d = 0u; d < draftCount; ++d) {
        float val = (lane < topK) ? refineTable[(d * 16u + sender) * 16u + lane] : -INFINITY;
        uint bestK = lane;
        for (ushort offset = 16; offset > 0; offset >>= 1) {
            float otherVal = simd_shuffle_down(val, offset);
            uint otherK = simd_shuffle_down(bestK, offset);
            if (otherVal > val || (otherVal == val && otherK < bestK)) {
                val = otherVal;
                bestK = otherK;
            }
        }
        sender = simd_broadcast_first(bestK);
        if (lane == 0u) {
            draftTokensOut[d] = candidates[d * 16u + sender];
        }
    }
}
