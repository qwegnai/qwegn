// ── QWDFlash — DFlash 2 block-diffusion drafter kernels ─────────────────────
// Implements target hidden capture, standard RMSNorm, grouped dynamic causal
// conv, 128-dim RoPE + rotating KV cache, non-causal GQA attention, top-16
// candidate extraction, and sequential pairwise CandidateSelector.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_dequant.metalh"
using namespace metal;

constant bool FC_DRAFTER_ROTATED [[function_constant(236)]];
constant bool FC_HEAD_ROTATED    [[function_constant(237)]];

struct DFlashH8 { half v[8]; } __attribute__((__packed__, __aligned__(16)));
static inline DFlashH8 asDFlashH8(uint4 w) { return *(thread DFlashH8*)&w; }

// Unconditional 32-point normalized WHT in FP32 matching fwht32_f32_to_h8 butterflies and scaling.
static inline void dflash_fwht32_always(thread float (&f)[8], uint lane_id) {
    #pragma unroll
    for (int i = 0; i < 8; i += 2) { float a = f[i], b = f[i + 1]; f[i] = a + b; f[i + 1] = a - b; }
    #pragma unroll
    for (int i = 0; i < 8; i += 4) {
        float a0 = f[i], a1 = f[i + 1], b0 = f[i + 2], b1 = f[i + 3];
        f[i] = a0 + b0; f[i + 1] = a1 + b1; f[i + 2] = a0 - b0; f[i + 3] = a1 - b1;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) { float a = f[i], b = f[i + 4]; f[i] = a + b; f[i + 4] = a - b; }
    bool neg1 = (lane_id & 1u) != 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) { float p = simd_shuffle_xor(f[i], ushort(1)); f[i] = neg1 ? (p - f[i]) : (f[i] + p); }
    bool neg2 = (lane_id & 2u) != 0;
    constexpr float inv_sqrt32 = 0.17677669529663689f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float p = simd_shuffle_xor(f[i], ushort(2));
        f[i] = (neg2 ? (p - f[i]) : (f[i] + p)) * inv_sqrt32;
    }
}

static inline half dflash_simd_fwht32_always(half val, uint lane_id) {
    float x = float(val);
    #pragma unroll
    for (ushort s = 1; s < 32; s <<= 1) {
        float partner = simd_shuffle_xor(x, s);
        x = ((lane_id & s) == 0) ? (x + partner) : (partner - x);
    }
    return half(x * 0.17677669529663689f);
}

static inline void dflash_cond_fwht32_to_h8(thread float (&f)[8], thread half (&v)[8], uint lane_id, bool rotated) {
    if (rotated) dflash_fwht32_always(f, lane_id);
    #pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = half(f[i]);
}

// Standard Qwen3RMSNorm (multiplies by gamma directly, NO +1).
// mode 0/1/2: RMSNorm; mode 3: x+=y -> yRot; mode 4: bridge; mode 5: y + yRot + yRot32.
kernel void qwg_dflash_rmsnorm(
        device void* x               [[buffer(0)]],
        device const uint4* gamma    [[buffer(1)]],
        device uint4* y              [[buffer(2)]],
        device uint4* yRot           [[buffer(3)]],
        constant float& eps          [[buffer(4)]],
        constant uint& mode          [[buffer(5)]],
        constant uint& hidden        [[buffer(6)]],
        device float* yRot32         [[buffer(7)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    const uint hiddenU4 = hidden >> 3u;
    const uint hiddenF4 = hidden >> 2u;
    float acc = 0.0f;
    if (mode == 1u) {
        device const uint4* xr = (device const uint4*)x + (size_t)tg * hiddenU4;
        for (uint u = ti; u < hiddenU4; u += 32u) {
            DFlashH8 hv = asDFlashH8(xr[u]);
            for (int i = 0; i < 8; i++) {
                float v = float(hv.v[i]);
                acc += v * v;
            }
        }
    } else if (mode == 3u || mode == 4u) {
        device float4* xr = (device float4*)x + (size_t)tg * hiddenF4;
        device const uint4* dr = ((mode == 4u) ? gamma : (device const uint4*)y) + (size_t)tg * hiddenU4;
        for (uint u = ti; u < hiddenU4; u += 32u) {
            DFlashH8 dv = asDFlashH8(dr[u]);
            float4 v0 = xr[u * 2u + 0u] + float4(float(dv.v[0]), float(dv.v[1]), float(dv.v[2]), float(dv.v[3]));
            float4 v1 = xr[u * 2u + 1u] + float4(float(dv.v[4]), float(dv.v[5]), float(dv.v[6]), float(dv.v[7]));
            xr[u * 2u + 0u] = v0;
            xr[u * 2u + 1u] = v1;
            acc += dot(v0, v0) + dot(v1, v1);
        }
    } else {
        device const float4* xr = (device const float4*)x + (size_t)tg * hiddenF4;
        for (uint u = ti; u < hiddenU4; u += 32u) {
            float4 v0 = xr[u * 2u + 0u];
            float4 v1 = xr[u * 2u + 1u];
            acc += dot(v0, v0) + dot(v1, v1);
        }
    }
    const float meanSq = simd_sum(acc) * (1.0f / float(hidden));
    device uint4* yr = y + (size_t)tg * hiddenU4;
    device uint4* yrr = yRot + (size_t)tg * hiddenU4;
    if (mode == 4u) {
        const float rms = sqrt(max(meanSq, 1.17549435e-38f));
        if (ti == 0u) ((device float*)yRot)[tg] = rms;
        const float invDenom = 1.0f / (rms + eps);
        device const float4* xr = (device const float4*)x + (size_t)tg * hiddenF4;
        for (uint u = ti; u < hiddenU4; u += 32u) {
            float4 v0 = xr[u * 2u + 0u] * invDenom;
            float4 v1 = xr[u * 2u + 1u] * invDenom;
            DFlashH8 ov = {{half(v0.x), half(v0.y), half(v0.z), half(v0.w),
                            half(v1.x), half(v1.y), half(v1.z), half(v1.w)}};
            yr[u] = *(thread uint4*)&ov;
        }
        return;
    }
    float rstd = rsqrt(meanSq + eps);
    if (mode == 1u) {
        device const uint4* xr = (device const uint4*)x + (size_t)tg * hiddenU4;
        for (uint u = ti; u < hiddenU4; u += 32u) {
            DFlashH8 hv = asDFlashH8(xr[u]), gv = asDFlashH8(gamma[u]), rv;
            float f[8];
            for (int i = 0; i < 8; i++) {
                f[i] = float(hv.v[i]) * rstd * float(gv.v[i]);
            }
            dflash_cond_fwht32_to_h8(f, rv.v, ti, FC_DRAFTER_ROTATED);
            yr[u] = *(thread uint4*)&rv;
        }
    } else {
        device const float4* xr = (device const float4*)x + (size_t)tg * hiddenF4;
        device float4* y32r = (mode == 5u) ? ((device float4*)yRot32 + (size_t)tg * hiddenF4) : nullptr;
        for (uint u = ti; u < hiddenU4; u += 32u) {
            float4 v0 = xr[u * 2u + 0u];
            float4 v1 = xr[u * 2u + 1u];
            DFlashH8 gv = asDFlashH8(gamma[u]), ov, rv;
            float xv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
            float f[8];
            for (int i = 0; i < 8; i++) {
                f[i] = xv[i] * rstd * float(gv.v[i]);
                ov.v[i] = half(f[i]);
            }
            if (mode == 0u) {
                yr[u] = *(thread uint4*)&ov;
            } else if (mode == 3u) {
                dflash_cond_fwht32_to_h8(f, rv.v, ti, FC_DRAFTER_ROTATED);
                yrr[u] = *(thread uint4*)&rv;
            } else if (mode == 5u) {
                yr[u] = *(thread uint4*)&ov;
                dflash_cond_fwht32_to_h8(f, rv.v, ti, FC_HEAD_ROTATED);
                yrr[u] = *(thread uint4*)&rv;
                y32r[u * 2u + 0u] = float4(f[0], f[1], f[2], f[3]);
                y32r[u * 2u + 1u] = float4(f[4], f[5], f[6], f[7]);
            } else {
                yr[u] = *(thread uint4*)&ov;
                dflash_cond_fwht32_to_h8(f, rv.v, ti, FC_DRAFTER_ROTATED);
                yrr[u] = *(thread uint4*)&rv;
            }
        }
    }
}

// Batched FP16 GEMV for small N (conv kernel_projection, selector, DSpine bridge), M=1..16. Emits FP32 [M, N].
kernel void qwg_dflash_f16_gemv_batched(
        device const uint4* W        [[buffer(0)]],
        device const uint4* x        [[buffer(1)]],
        device float* y              [[buffer(2)]],
        constant uint& N             [[buffer(3)]],
        constant uint& K             [[buffer(4)]],
        constant uint& M             [[buffer(5)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    const uint row = tg * 4u + simd_id;
    if (row >= N) return;
    const uint chunks = K >> 3u;
    device const uint4* wr = W + (size_t)row * chunks;
    const uint safeM = min(M, 16u);
    float acc[16] = {0.0f};
    for (uint c = lane; c < chunks; c += 32u) {
        DFlashH8 wv = asDFlashH8(wr[c]);
        for (uint m = 0; m < safeM; ++m) {
            DFlashH8 xv = asDFlashH8(x[(size_t)m * chunks + c]);
            float s = 0.0f;
            for (int i = 0; i < 8; ++i) s += float(wv.v[i]) * float(xv.v[i]);
            acc[m] += s;
        }
    }
    for (uint m = 0; m < safeM; ++m) {
        float val = simd_sum(acc[m]);
        if (lane == 0u) y[(size_t)m * N + row] = val;
    }
}

// GroupedDynamicCausalConv (k=2): mode 0 = fwht32 half; mode 1 = f32 += conv; mode 2 = unrot half.
kernel void qwg_dflash_conv_apply(
        device const uint4* x          [[buffer(0)]],
        device const float* dyn        [[buffer(1)]],
        device const uint4* baseKernel [[buffer(2)]],
        device void* outOrResidual     [[buffer(3)]],
        constant uint& mode            [[buffer(4)]],
        constant uint& hiddenU4        [[buffer(5)]],
        constant uint& convGroups      [[buffer(6)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint2 numTGs [[threadgroups_per_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    if (convGroups == 0u) return;
    const uint t = tg.x;
    const uint side = (mode >= 1u) ? 1u : 0u;
    const uint hiddenF4 = hiddenU4 << 1u;
    const uint dynStride = convGroups * 4u;
    const uint uPerGroup = max(1u, (hiddenU4 + convGroups - 1u) / convGroups);
    device const uint4* xCur = x + (size_t)t * hiddenU4;
    device const uint4* xPrev = (t > 0u) ? (x + (size_t)(t - 1u) * hiddenU4) : xCur;
    device const uint4* bTap0 = baseKernel + (size_t)(side * 2u + 0u) * hiddenU4;
    device const uint4* bTap1 = baseKernel + (size_t)(side * 2u + 1u) * hiddenU4;
    device const float* dTap0 = dyn + (size_t)t * dynStride + (size_t)side * (convGroups * 2u);
    device const float* dTap1 = dTap0 + convGroups;

    const uint uStart = tg.y * 32u + ti;
    const uint uStep = max(1u, numTGs.y) * 32u;
    for (uint u = uStart; u < hiddenU4; u += uStep) {
        const uint g = min(u / uPerGroup, convGroups - 1u);
        const float dw0 = dTap0[g];
        const float dw1 = dTap1[g];
        DFlashH8 xc = asDFlashH8(xCur[u]);
        DFlashH8 xp = (t > 0u) ? asDFlashH8(xPrev[u]) : DFlashH8{{0, 0, 0, 0, 0, 0, 0, 0}};
        DFlashH8 b0 = asDFlashH8(bTap0[u]);
        DFlashH8 b1 = asDFlashH8(bTap1[u]);
        float f[8];
        for (int i = 0; i < 8; ++i) {
            float w0 = float(b0.v[i]) + dw0;
            float w1 = float(b1.v[i]) + dw1;
            float v = float(xc.v[i]) * w0 + ((t > 0u) ? (float(xp.v[i]) * w1) : 0.0f);
            f[i] = v;
        }
        if (mode == 0u || mode == 2u) {
            device uint4* dst16 = (device uint4*)outOrResidual + (size_t)t * hiddenU4;
            DFlashH8 rot;
            dflash_cond_fwht32_to_h8(f, rot.v, ti, (mode == 0u) && FC_DRAFTER_ROTATED);
            dst16[u] = *(thread uint4*)&rot;
        } else {
            device float4* dst32 = (device float4*)outOrResidual + (size_t)t * hiddenF4;
            dst32[u * 2u + 0u] += float4(f[0], f[1], f[2], f[3]);
            dst32[u * 2u + 1u] += float4(f[4], f[5], f[6], f[7]);
        }
    }
}

// Context K norm + 128-dim RoPE + rotating KV cache write.
// Grid = (S, kvHeads, 1), threadsPerThreadgroup = (32, 1, 1).
kernel void qwg_dflash_rope_kv_ctx(
        device const half* kvCtxIn   [[buffer(0)]],
        device const half* wk        [[buffer(1)]],
        device const float* invFreq  [[buffer(2)]],
        device half* kCache          [[buffer(3)]],
        device half* vCache          [[buffer(4)]],
        constant uint& cacheOffset   [[buffer(5)]],
        constant float& eps          [[buffer(6)]],
        constant uint& maxCacheSize  [[buffer(7)]],
        constant uint& kvHeads       [[buffer(8)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    const uint s = tg.x;
    const uint h = tg.y;
    const uint pos = cacheOffset + s;
    const uint slot = pos % maxCacheSize;
    const uint kvDim = kvHeads * 128u;
    const uint kvInStride = kvDim * 2u;
    device const half* kh = kvCtxIn + (size_t)s * kvInStride + h * 128u;
    device const half* vh = kvCtxIn + (size_t)s * kvInStride + kvDim + h * 128u;
    device half* kDst = kCache + (size_t)slot * kvDim + h * 128u;
    device half* vDst = vCache + (size_t)slot * kvDim + h * 128u;

    float x0 = float(kh[lane]);
    float x1 = float(kh[lane + 32u]);
    float x2 = float(kh[lane + 64u]);
    float x3 = float(kh[lane + 96u]);
    float acc = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    float rstd = rsqrt(simd_sum(acc) * (1.0f / 128.0f) + eps);

    float n0 = x0 * rstd * float(wk[lane]);
    float n1 = x1 * rstd * float(wk[lane + 32u]);
    float n2 = x2 * rstd * float(wk[lane + 64u]);
    float n3 = x3 * rstd * float(wk[lane + 96u]);

    float a0 = float(pos) * invFreq[lane];
    float a1 = float(pos) * invFreq[lane + 32u];
    float c0 = cos(a0), s0 = sin(a0);
    float c1 = cos(a1), s1 = sin(a1);

    kDst[lane]       = half(n0 * c0 - n2 * s0);
    kDst[lane + 32u] = half(n1 * c1 - n3 * s1);
    kDst[lane + 64u] = half(n2 * c0 + n0 * s0);
    kDst[lane + 96u] = half(n3 * c1 + n1 * s1);

    vDst[lane]       = vh[lane];
    vDst[lane + 32u] = vh[lane + 32u];
    vDst[lane + 64u] = vh[lane + 64u];
    vDst[lane + 96u] = vh[lane + 96u];
}

// Proposal Q (qHeads) & K (kvHeads) norm + 128-dim RoPE in-place on qkvProp.
// Grid = (M, qHeads + kvHeads, 1), threadsPerThreadgroup = (32, 1, 1).
kernel void qwg_dflash_rope_qk_prop(
        device half* qkvProp         [[buffer(0)]],
        device const half* wq        [[buffer(1)]],
        device const half* wk        [[buffer(2)]],
        device const float* invFreq  [[buffer(3)]],
        constant uint& propOffset    [[buffer(4)]],
        constant float& eps          [[buffer(5)]],
        constant uint& qHeads        [[buffer(6)]],
        constant uint& kvHeads       [[buffer(7)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    const uint m = tg.x;
    const bool isQ = tg.y < qHeads;
    const uint h = isQ ? tg.y : (tg.y - qHeads);
    const uint pos = propOffset + m;
    const uint qDim = qHeads * 128u;
    const uint kvDim = kvHeads * 128u;
    const uint qkvStride = qDim + 2u * kvDim;
    device half* vec = isQ ? (qkvProp + (size_t)m * qkvStride + h * 128u)
                           : (qkvProp + (size_t)m * qkvStride + qDim + h * 128u);
    device const half* w = isQ ? wq : wk;

    float x0 = float(vec[lane]);
    float x1 = float(vec[lane + 32u]);
    float x2 = float(vec[lane + 64u]);
    float x3 = float(vec[lane + 96u]);
    float acc = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    float rstd = rsqrt(simd_sum(acc) * (1.0f / 128.0f) + eps);

    float n0 = x0 * rstd * float(w[lane]);
    float n1 = x1 * rstd * float(w[lane + 32u]);
    float n2 = x2 * rstd * float(w[lane + 64u]);
    float n3 = x3 * rstd * float(w[lane + 96u]);

    float a0 = float(pos) * invFreq[lane];
    float a1 = float(pos) * invFreq[lane + 32u];
    float c0 = cos(a0), s0 = sin(a0);
    float c1 = cos(a1), s1 = sin(a1);

    vec[lane]       = half(n0 * c0 - n2 * s0);
    vec[lane + 32u] = half(n1 * c1 - n3 * s1);
    vec[lane + 64u] = half(n2 * c0 + n0 * s0);
    vec[lane + 96u] = half(n3 * c1 + n1 * s1);
}

// Non-causal (DFlash2) or causal (DSpine) within-block + sliding-window context GQA attention.
// Grid = (qHeads, M, 1), threadsPerThreadgroup = (32, 1, 1). Rotates output via simd_fwht32.
kernel void qwg_dflash_attn(
        device const half* qkvProp   [[buffer(0)]],
        device const half* kCache    [[buffer(1)]],
        device const half* vCache    [[buffer(2)]],
        device half* out             [[buffer(3)]],
        constant uint& newOffset     [[buffer(4)]],
        constant uint& M             [[buffer(5)]],
        constant uint& maxCacheSize  [[buffer(6)]],
        constant uint& isCausal      [[buffer(7)]],
        constant uint& qHeads        [[buffer(8)]],
        constant uint& kvHeads       [[buffer(9)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    const uint qh = tg.x;
    const uint m = tg.y;
    const uint gqaRatio = max(1u, qHeads / kvHeads);
    const uint kvh = qh / gqaRatio;
    const uint qDim = qHeads * 128u;
    const uint kvDim = kvHeads * 128u;
    const uint qkvStride = qDim + 2u * kvDim;
    const float scale = 0.08838834764831845f; // 1 / sqrt(128)

    device const half* qv = qkvProp + (size_t)m * qkvStride + qh * 128u;
    const float q0 = float(qv[lane]) * scale;
    const float q1 = float(qv[lane + 32u]) * scale;
    const float q2 = float(qv[lane + 64u]) * scale;
    const float q3 = float(qv[lane + 96u]) * scale;

    float maxLogit = -3.4e38f;
    bool hasLogit = false;
    float lsum = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    const uint absQ = newOffset + m;
    const uint startK = (absQ > maxCacheSize) ? (absQ - maxCacheSize) : 0u;
    for (uint absK = startK; absK < newOffset; ++absK) {
        const uint slot = absK % maxCacheSize;
        device const half* kv = kCache + (size_t)slot * kvDim + kvh * 128u;
        device const half* vv = vCache + (size_t)slot * kvDim + kvh * 128u;
        float dot = q0 * float(kv[lane]) + q1 * float(kv[lane + 32u])
                  + q2 * float(kv[lane + 64u]) + q3 * float(kv[lane + 96u]);
        float s = simd_sum(dot);
        float newMax = hasLogit ? max(maxLogit, s) : s;
        float corr = hasLogit ? exp(maxLogit - newMax) : 0.0f;
        float p = exp(s - newMax);
        lsum = lsum * corr + p;
        o0 = o0 * corr + p * float(vv[lane]);
        o1 = o1 * corr + p * float(vv[lane + 32u]);
        o2 = o2 * corr + p * float(vv[lane + 64u]);
        o3 = o3 * corr + p * float(vv[lane + 96u]);
        maxLogit = newMax;
        hasLogit = true;
    }

    const uint pLimit = (isCausal != 0u) ? (m + 1u) : M;
    for (uint pIdx = 0u; pIdx < pLimit; ++pIdx) {
        device const half* kv = qkvProp + (size_t)pIdx * qkvStride + qDim + kvh * 128u;
        device const half* vv = qkvProp + (size_t)pIdx * qkvStride + qDim + kvDim + kvh * 128u;
        float dot = q0 * float(kv[lane]) + q1 * float(kv[lane + 32u])
                  + q2 * float(kv[lane + 64u]) + q3 * float(kv[lane + 96u]);
        float s = simd_sum(dot);
        float newMax = hasLogit ? max(maxLogit, s) : s;
        float corr = hasLogit ? exp(maxLogit - newMax) : 0.0f;
        float p = exp(s - newMax);
        lsum = lsum * corr + p;
        o0 = o0 * corr + p * float(vv[lane]);
        o1 = o1 * corr + p * float(vv[lane + 32u]);
        o2 = o2 * corr + p * float(vv[lane + 64u]);
        o3 = o3 * corr + p * float(vv[lane + 96u]);
        maxLogit = newMax;
        hasLogit = true;
    }

    const float invL = (lsum > 0.0f) ? (1.0f / lsum) : 0.0f;
    device half* dst = out + (size_t)m * qDim + qh * 128u;
    half h0 = half(o0 * invL), h1 = half(o1 * invL), h2 = half(o2 * invL), h3 = half(o3 * invL);
    if (FC_DRAFTER_ROTATED) {
        h0 = dflash_simd_fwht32_always(h0, lane);
        h1 = dflash_simd_fwht32_always(h1, lane);
        h2 = dflash_simd_fwht32_always(h2, lane);
        h3 = dflash_simd_fwht32_always(h3, lane);
    }
    dst[lane]       = h0;
    dst[lane + 32u] = h1;
    dst[lane + 64u] = h2;
    dst[lane + 96u] = h3;
}

struct DFlashTapPackP {
    uint srcPlaneStrideU4;
    uint srcRowOffset;
    uint dstRowOffset;
    uint hiddenU4;
    uint totalSlots;
    uint useRowMap;
    uint rowMap[16];
};

// Pack target BF16 taps src[slot, srcRow, H] -> targetHidden[dstRow, slot * H] FP16.
// Grid = (rows, totalSlots, 1), threadsPerThreadgroup = (32, 1, 1).
kernel void qwg_dflash_pack_bf16_taps(
        device const uint4* src        [[buffer(0)]],
        device uint4* dst              [[buffer(1)]],
        constant DFlashTapPackP& p     [[buffer(2)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint ti  [[thread_index_in_threadgroup]]) {
    const uint r = tg.x;
    const uint slot = tg.y;
    const uint srcRel = p.useRowMap != 0u ? p.rowMap[r] : r;
    const uint srcRow = p.srcRowOffset + srcRel;
    const uint dstRow = p.dstRowOffset + r;
    device const uint4* xr = src + (size_t)slot * p.srcPlaneStrideU4 + (size_t)srcRow * p.hiddenU4;
    device uint4* yr = dst + (size_t)dstRow * p.totalSlots * p.hiddenU4 + (size_t)slot * p.hiddenU4;
    for (uint u = ti; u < p.hiddenU4; u += 32u) {
        uint4 raw = xr[u];
        ushort b[8] = {
            ushort(raw.x & 0xffffu), ushort(raw.x >> 16u),
            ushort(raw.y & 0xffffu), ushort(raw.y >> 16u),
            ushort(raw.z & 0xffffu), ushort(raw.z >> 16u),
            ushort(raw.w & 0xffffu), ushort(raw.w >> 16u)
        };
        float f[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) f[i] = as_type<float>(uint(b[i]) << 16u);
        DFlashH8 hv;
        dflash_cond_fwht32_to_h8(f, hv.v, ti, FC_DRAFTER_ROTATED);
        yr[u] = *(thread uint4*)&hv;
    }
}

// Batched MLP SiLU * Up epilogue (upper 16 bits of M encode optional f16 swigluLimit).
kernel void qwg_dflash_silu_mul_f16(
        device const half* gu [[buffer(0)]],
        device half* out      [[buffer(1)]],
        constant uint& mPacked [[buffer(2)]],
        constant uint& I      [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    constexpr uint kU = 8u;
    const uint M = mPacked & 0xFFFFu;
    const float swigluLimit = float(as_type<half>(ushort(mPacked >> 16u)));
    const uint mi = gid.x * kU;
    const uint m = gid.y;
    if (m >= M || mi >= I) return;
    const size_t inRow = (size_t)m * (2u * I);
    const size_t outRow = (size_t)m * I;
    const bool hasLim = (swigluLimit > 0.0f);
    if (mi + kU <= I) {
        device const half4* gp = (device const half4*)(gu + inRow + mi);
        device const half4* up = (device const half4*)(gu + inRow + I + mi);
        const half4 g0 = gp[0], g1 = gp[1];
        const half4 u0 = up[0], u1 = up[1];
        float gv[8] = {float(g0[0]), float(g0[1]), float(g0[2]), float(g0[3]),
                       float(g1[0]), float(g1[1]), float(g1[2]), float(g1[3])};
        float uv[8] = {float(u0[0]), float(u0[1]), float(u0[2]), float(u0[3]),
                       float(u1[0]), float(u1[1]), float(u1[2]), float(u1[3])};
        float f[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float gc = hasLim ? min(gv[k], swigluLimit) : gv[k];
            const float uc = hasLim ? clamp(uv[k], -swigluLimit, swigluLimit) : uv[k];
            f[k] = (gc / (1.0f + exp(-gc))) * uc;
        }
        half ov[8];
        dflash_cond_fwht32_to_h8(f, ov, gid.x & 31u, FC_DRAFTER_ROTATED && ((I & 31u) == 0u));
        device half4* op = (device half4*)(out + outRow + mi);
        op[0] = half4(ov[0], ov[1], ov[2], ov[3]);
        op[1] = half4(ov[4], ov[5], ov[6], ov[7]);
    } else {
        for (uint k = 0; k < kU && mi + k < I; ++k) {
            float g = float(gu[inRow + mi + k]);
            float u = float(gu[inRow + I + mi + k]);
            if (hasLim) {
                g = min(g, swigluLimit);
                u = clamp(u, -swigluLimit, swigluLimit);
            }
            out[outRow + mi + k] = half((g / (1.0f + exp(-g))) * u);
        }
    }
}
