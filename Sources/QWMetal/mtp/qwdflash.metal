// ── QWDFlash — DFlash 2 block-diffusion drafter kernels ─────────────────────
// Implements target hidden capture, standard RMSNorm, grouped dynamic causal
// conv, 128-dim RoPE + rotating KV cache, non-causal GQA attention, top-16
// candidate extraction, and sequential pairwise CandidateSelector.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_dequant.metalh"
using namespace metal;

struct DFlashH8 { half v[8]; } __attribute__((__packed__, __aligned__(16)));
static inline DFlashH8 asDFlashH8(uint4 w) { return *(thread DFlashH8*)&w; }
static inline float dflash_bf16_to_f32(ushort b) { return as_type<float>(uint(b) << 16u); }

// Kernel 1: Capture unrotated target residual [rows, 5120], rotate via fwht32,
// and scatter into [maxRows, 25600] at slot * 5120 (slot in 0..4).
kernel void qwg_dflash_capture_target_hidden(
        device const uint4* src      [[buffer(0)]],
        device uint4* dst            [[buffer(1)]],
        constant uint& srcRowOffset  [[buffer(2)]],
        constant uint& dstRowOffset  [[buffer(3)]],
        constant uint& slot          [[buffer(4)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    device const uint4* xr = src + (size_t)(srcRowOffset + tg) * 640u;
    device uint4* yr = dst + (size_t)(dstRowOffset + tg) * 3200u + (size_t)slot * 640u;
    for (uint u = ti; u < 640u; u += 32u) {
        DFlashH8 hv = asDFlashH8(xr[u]);
        fwht32_inplace(hv.v, ti);
        yr[u] = *(thread uint4*)&hv;
    }
}

// Kernel 2: Standard Qwen3RMSNorm (multiplies by gamma directly, NO +1).
// mode 0/1/2: standard RMSNorm; mode 3: x(f32)+=y(f16) -> fwht32 yRot; mode 4: bridge prep.
kernel void qwg_dflash_rmsnorm(
        device void* x               [[buffer(0)]],
        device const uint4* gamma    [[buffer(1)]],
        device uint4* y              [[buffer(2)]],
        device uint4* yRot           [[buffer(3)]],
        constant float& eps          [[buffer(4)]],
        constant uint& mode          [[buffer(5)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    float acc = 0.0f;
    if (mode == 1u) {
        device const uint4* xr = (device const uint4*)x + (size_t)tg * 640u;
        for (uint u = ti; u < 640u; u += 32u) {
            DFlashH8 hv = asDFlashH8(xr[u]);
            for (int i = 0; i < 8; i++) {
                float v = float(hv.v[i]);
                acc += v * v;
            }
        }
    } else if (mode == 3u || mode == 4u) {
        device float4* xr = (device float4*)x + (size_t)tg * 1280u;
        device const uint4* dr = ((mode == 4u) ? gamma : (device const uint4*)y) + (size_t)tg * 640u;
        for (uint u = ti; u < 640u; u += 32u) {
            DFlashH8 dv = asDFlashH8(dr[u]);
            float4 v0 = xr[u * 2u + 0u] + float4(float(dv.v[0]), float(dv.v[1]), float(dv.v[2]), float(dv.v[3]));
            float4 v1 = xr[u * 2u + 1u] + float4(float(dv.v[4]), float(dv.v[5]), float(dv.v[6]), float(dv.v[7]));
            xr[u * 2u + 0u] = v0;
            xr[u * 2u + 1u] = v1;
            acc += dot(v0, v0) + dot(v1, v1);
        }
    } else {
        device const float4* xr = (device const float4*)x + (size_t)tg * 1280u;
        for (uint u = ti; u < 640u; u += 32u) {
            float4 v0 = xr[u * 2u + 0u];
            float4 v1 = xr[u * 2u + 1u];
            acc += dot(v0, v0) + dot(v1, v1);
        }
    }
    const float meanSq = simd_sum(acc) * (1.0f / 5120.0f);
    device uint4* yr = y + (size_t)tg * 640u;
    device uint4* yrr = yRot + (size_t)tg * 640u;
    if (mode == 4u) {
        const float rms = sqrt(max(meanSq, 1.17549435e-38f));
        if (ti == 0u) ((device float*)yRot)[tg] = rms;
        const float invDenom = 1.0f / (rms + eps);
        device const float4* xr = (device const float4*)x + (size_t)tg * 1280u;
        for (uint u = ti; u < 640u; u += 32u) {
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
        device const uint4* xr = (device const uint4*)x + (size_t)tg * 640u;
        for (uint u = ti; u < 640u; u += 32u) {
            DFlashH8 hv = asDFlashH8(xr[u]), gv = asDFlashH8(gamma[u]), rv;
            float f[8];
            for (int i = 0; i < 8; i++) {
                f[i] = float(hv.v[i]) * rstd * float(gv.v[i]);
            }
            fwht32_f32_to_h8(f, rv.v, ti);
            yr[u] = *(thread uint4*)&rv;
        }
    } else {
        device const float4* xr = (device const float4*)x + (size_t)tg * 1280u;
        for (uint u = ti; u < 640u; u += 32u) {
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
                fwht32_f32_to_h8(f, rv.v, ti);
                yrr[u] = *(thread uint4*)&rv;
            } else {
                yr[u] = *(thread uint4*)&ov;
                fwht32_f32_to_h8(f, rv.v, ti);
                yrr[u] = *(thread uint4*)&rv;
            }
        }
    }
}

// Kernel 3: Batched FP16 GEMV for small N (kernel_projection N=1280,
// hidden_projection N=256, DSpine bridge read/gate), K=1024..5120, M=1..16. Emits FP32 [M, N].
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
    float acc[16] = {0.0f};
    for (uint c = lane; c < chunks; c += 32u) {
        DFlashH8 wv = asDFlashH8(wr[c]);
        for (uint m = 0; m < M; ++m) {
            DFlashH8 xv = asDFlashH8(x[(size_t)m * chunks + c]);
            float s = 0.0f;
            for (int i = 0; i < 8; ++i) s += float(wv.v[i]) * float(xv.v[i]);
            acc[m] += s;
        }
    }
    for (uint m = 0; m < M; ++m) {
        float val = simd_sum(acc[m]);
        if (lane == 0u) y[(size_t)m * N + row] = val;
    }
}

// Kernel 4: GroupedDynamicCausalConv (kernel_size=2, group_size=16, groups=320).
// mode 0 (prepare): out (f16) = fwht32(conv(x, dyn[0], base[0])) [M, 5120]
// mode 1 (finish):  residual (f32) += conv(x, dyn[1], base[1]) [M, 5120]
kernel void qwg_dflash_conv_apply(
        device const uint4* x          [[buffer(0)]],
        device const float* dyn        [[buffer(1)]],
        device const uint4* baseKernel [[buffer(2)]],
        device void* outOrResidual     [[buffer(3)]],
        constant uint& mode            [[buffer(4)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    const uint t = tg;
    device const uint4* xCur = x + (size_t)t * 640u;
    device const uint4* xPrev = (t > 0u) ? (x + (size_t)(t - 1u) * 640u) : xCur;
    device const uint4* bTap0 = baseKernel + (size_t)(mode * 2u + 0u) * 640u;
    device const uint4* bTap1 = baseKernel + (size_t)(mode * 2u + 1u) * 640u;
    device const float* dTap0 = dyn + (size_t)t * 1280u + (size_t)mode * 640u;
    device const float* dTap1 = dTap0 + 320u;

    for (uint u = ti; u < 640u; u += 32u) {
        const uint g = u >> 1u;
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
        if (mode == 0u) {
            device uint4* dst16 = (device uint4*)outOrResidual + (size_t)t * 640u;
            DFlashH8 rot;
            fwht32_f32_to_h8(f, rot.v, ti);
            dst16[u] = *(thread uint4*)&rot;
        } else {
            device float4* dst32 = (device float4*)outOrResidual + (size_t)t * 1280u;
            dst32[u * 2u + 0u] += float4(f[0], f[1], f[2], f[3]);
            dst32[u * 2u + 1u] += float4(f[4], f[5], f[6], f[7]);
        }
    }
}

// Kernel 5: Context K norm + 128-dim RoPE + rotating KV cache write.
// Reads packed kvCtxIn [S, 2048] (k [0..1023], v [1024..2047]).
// Grid = (S, 8, 1), threadsPerThreadgroup = (32, 1, 1).
kernel void qwg_dflash_rope_kv_ctx(
        device const half* kvCtxIn   [[buffer(0)]],
        device const half* wk        [[buffer(1)]],
        device const float* invFreq  [[buffer(2)]],
        device half* kCache          [[buffer(3)]],
        device half* vCache          [[buffer(4)]],
        constant uint& cacheOffset   [[buffer(5)]],
        constant float& eps          [[buffer(6)]],
        constant uint& maxCacheSize  [[buffer(7)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    const uint s = tg.x;
    const uint h = tg.y;
    const uint pos = cacheOffset + s;
    const uint slot = pos % maxCacheSize;
    device const half* kh = kvCtxIn + (size_t)s * 2048u + h * 128u;
    device const half* vh = kvCtxIn + (size_t)s * 2048u + 1024u + h * 128u;
    device half* kDst = kCache + (size_t)slot * 1024u + h * 128u;
    device half* vDst = vCache + (size_t)slot * 1024u + h * 128u;

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

// Kernel 6: Proposal Q (32 heads) & K (8 heads) norm + 128-dim RoPE in-place on qkvProp [M, 6144].
// Grid = (M, 40, 1), threadsPerThreadgroup = (32, 1, 1).
kernel void qwg_dflash_rope_qk_prop(
        device half* qkvProp         [[buffer(0)]],
        device const half* wq        [[buffer(1)]],
        device const half* wk        [[buffer(2)]],
        device const float* invFreq  [[buffer(3)]],
        constant uint& propOffset    [[buffer(4)]],
        constant float& eps          [[buffer(5)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    const uint m = tg.x;
    const bool isQ = tg.y < 32u;
    const uint h = isQ ? tg.y : (tg.y - 32u);
    const uint pos = propOffset + m;
    device half* vec = isQ ? (qkvProp + (size_t)m * 6144u + h * 128u)
                           : (qkvProp + (size_t)m * 6144u + 4096u + h * 128u);
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

// Kernel 7: Non-causal (DFlash2) or causal (DSpine) within-block + sliding-window context GQA attention.
// Reads q/k/v from qkvProp [M, 6144] and context k/v from kCache/vCache [maxCacheSize, 1024].
// Grid = (32, M, 1), threadsPerThreadgroup = (32, 1, 1). Rotates output via simd_fwht32.
kernel void qwg_dflash_attn(
        device const half* qkvProp   [[buffer(0)]],
        device const half* kCache    [[buffer(1)]],
        device const half* vCache    [[buffer(2)]],
        device half* out             [[buffer(3)]],
        constant uint& newOffset     [[buffer(4)]],
        constant uint& M             [[buffer(5)]],
        constant uint& maxCacheSize  [[buffer(6)]],
        constant uint& isCausal      [[buffer(7)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    const uint qh = tg.x;
    const uint m = tg.y;
    const uint kvh = qh >> 2u;
    const float scale = 0.08838834764831845f; // 1 / sqrt(128)

    device const half* qv = qkvProp + (size_t)m * 6144u + qh * 128u;
    const float q0 = float(qv[lane]) * scale;
    const float q1 = float(qv[lane + 32u]) * scale;
    const float q2 = float(qv[lane + 64u]) * scale;
    const float q3 = float(qv[lane + 96u]) * scale;

    float maxLogit = -INFINITY;
    float lsum = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    const uint absQ = newOffset + m;
    const uint startK = (absQ > maxCacheSize) ? (absQ - maxCacheSize) : 0u;
    for (uint absK = startK; absK < newOffset; ++absK) {
        const uint slot = absK % maxCacheSize;
        device const half* kv = kCache + (size_t)slot * 1024u + kvh * 128u;
        device const half* vv = vCache + (size_t)slot * 1024u + kvh * 128u;
        float dot = q0 * float(kv[lane]) + q1 * float(kv[lane + 32u])
                  + q2 * float(kv[lane + 64u]) + q3 * float(kv[lane + 96u]);
        float s = simd_sum(dot);
        float newMax = max(maxLogit, s);
        float corr = (maxLogit == -INFINITY) ? 0.0f : exp(maxLogit - newMax);
        float p = exp(s - newMax);
        lsum = lsum * corr + p;
        o0 = o0 * corr + p * float(vv[lane]);
        o1 = o1 * corr + p * float(vv[lane + 32u]);
        o2 = o2 * corr + p * float(vv[lane + 64u]);
        o3 = o3 * corr + p * float(vv[lane + 96u]);
        maxLogit = newMax;
    }

    const uint pLimit = (isCausal != 0u) ? (m + 1u) : M;
    for (uint pIdx = 0u; pIdx < pLimit; ++pIdx) {
        device const half* kv = qkvProp + (size_t)pIdx * 6144u + 4096u + kvh * 128u;
        device const half* vv = qkvProp + (size_t)pIdx * 6144u + 5120u + kvh * 128u;
        float dot = q0 * float(kv[lane]) + q1 * float(kv[lane + 32u])
                  + q2 * float(kv[lane + 64u]) + q3 * float(kv[lane + 96u]);
        float s = simd_sum(dot);
        float newMax = max(maxLogit, s);
        float corr = (maxLogit == -INFINITY) ? 0.0f : exp(maxLogit - newMax);
        float p = exp(s - newMax);
        lsum = lsum * corr + p;
        o0 = o0 * corr + p * float(vv[lane]);
        o1 = o1 * corr + p * float(vv[lane + 32u]);
        o2 = o2 * corr + p * float(vv[lane + 64u]);
        o3 = o3 * corr + p * float(vv[lane + 96u]);
        maxLogit = newMax;
    }

    const float invL = (lsum > 0.0f) ? (1.0f / lsum) : 0.0f;
    device half* dst = out + (size_t)m * 4096u + qh * 128u;
    dst[lane]       = simd_fwht32(half(o0 * invL), lane);
    dst[lane + 32u] = simd_fwht32(half(o1 * invL), lane);
    dst[lane + 64u] = simd_fwht32(half(o2 * invL), lane);
    dst[lane + 96u] = simd_fwht32(half(o3 * invL), lane);
}

static inline void dflash_tg_argmax(
        float localVal, int localId, uint ti,
        threadgroup float* sVal, threadgroup int* sId,
        thread float& outVal, thread int& outId) {
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        float otherVal = simd_shuffle_down(localVal, offset);
        int otherId = simd_shuffle_down(localId, offset);
        if (otherVal > localVal || (otherVal == localVal && otherId < localId)) {
            localVal = otherVal;
            localId = otherId;
        }
    }
    if (lane == 0u) {
        sVal[simd_id] = localVal;
        sId[simd_id] = localId;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        float v = (lane < 8u) ? sVal[lane] : -INFINITY;
        int id = (lane < 8u) ? sId[lane] : 0x7fffffff;
        for (ushort offset = 4; offset > 0; offset >>= 1) {
            float ov = simd_shuffle_down(v, offset);
            int oid = simd_shuffle_down(id, offset);
            if (ov > v || (ov == v && oid < id)) {
                v = ov;
                id = oid;
            }
        }
        if (lane == 0u) {
            sVal[0] = v;
            sId[0] = id;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    outVal = sVal[0];
    outId = sId[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Kernel 8: Top-16 Stage 1 over 248320 vocab logits per draft row.
// Grid = (256, D, 1), threadsPerThreadgroup = (256, 1, 1).
kernel void qwg_dflash_top16_stage1(
        device const float* logits   [[buffer(0)]],
        device float* partialVals    [[buffer(1)]],
        device int* partialIds       [[buffer(2)]],
        constant uint& vocab         [[buffer(3)]],
        constant uint& topK          [[buffer(4)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    const uint chunk = tg.x;
    const uint row = tg.y;
    const uint base = chunk * 970u;
    device const float* rowLogits = logits + (size_t)row * vocab;

    float regVal[4];
    int regId[4];
    #pragma unroll
    for (uint i = 0; i < 4u; ++i) {
        uint idx = base + i * 256u + ti;
        if (idx < min(vocab, base + 970u)) {
            regVal[i] = rowLogits[idx];
            regId[i] = int(idx);
        } else {
            regVal[i] = -INFINITY;
            regId[i] = 0x7fffffff;
        }
    }

    threadgroup float sVal[8];
    threadgroup int sId[8];
    const uint outBase = (row * 256u + chunk) * 16u;

    for (uint k = 0; k < topK; ++k) {
        float bestLocalVal = -INFINITY;
        int bestLocalId = 0x7fffffff;
        #pragma unroll
        for (uint i = 0; i < 4u; ++i) {
            if (regVal[i] > bestLocalVal || (regVal[i] == bestLocalVal && regId[i] < bestLocalId)) {
                bestLocalVal = regVal[i];
                bestLocalId = regId[i];
            }
        }
        float winVal;
        int winId;
        dflash_tg_argmax(bestLocalVal, bestLocalId, ti, sVal, sId, winVal, winId);
        if (ti == 0u) {
            partialVals[outBase + k] = winVal;
            partialIds[outBase + k] = winId;
        }
        #pragma unroll
        for (uint i = 0; i < 4u; ++i) {
            if (regId[i] == winId) {
                regVal[i] = -INFINITY;
                regId[i] = 0x7fffffff;
            }
        }
    }
}

// Kernel 9: Top-16 Stage 2 reducing 256*16 partial candidates per draft row to topK.
// Grid = (D, 1, 1), threadsPerThreadgroup = (256, 1, 1).
kernel void qwg_dflash_top16_stage2(
        device const float* partialVals [[buffer(0)]],
        device const int* partialIds    [[buffer(1)]],
        device float* unaryOut          [[buffer(2)]],
        device int* candidatesOut       [[buffer(3)]],
        constant uint& topK             [[buffer(4)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    const uint row = tg;
    const uint inBase = row * 256u * 16u;

    float regVal[16];
    int regId[16];
    for (uint i = 0; i < 16u; ++i) {
        if (i < topK) {
            regVal[i] = partialVals[inBase + ti * 16u + i];
            regId[i] = partialIds[inBase + ti * 16u + i];
        } else {
            regVal[i] = -INFINITY;
            regId[i] = 0x7fffffff;
        }
    }

    threadgroup float sVal[8];
    threadgroup int sId[8];
    const uint outBase = row * 16u;

    for (uint k = 0; k < topK; ++k) {
        float bestLocalVal = -INFINITY;
        int bestLocalId = 0x7fffffff;
        for (uint i = 0; i < 16u; ++i) {
            if (regVal[i] > bestLocalVal || (regVal[i] == bestLocalVal && regId[i] < bestLocalId)) {
                bestLocalVal = regVal[i];
                bestLocalId = regId[i];
            }
        }
        float winVal;
        int winId;
        dflash_tg_argmax(bestLocalVal, bestLocalId, ti, sVal, sId, winVal, winId);
        if (ti == 0u) {
            unaryOut[outBase + k] = winVal;
            candidatesOut[outBase + k] = winId;
        }
        for (uint i = 0; i < 16u; ++i) {
            if (regId[i] == winId) {
                regVal[i] = -INFINITY;
                regId[i] = 0x7fffffff;
            }
        }
    }
}

// Kernel 10: Greedy sequential pairwise CandidateSelector over BF16 codebooks [248320, 256].
// Grid = (1, 1, 1), threadsPerThreadgroup = (256, 1, 1).
kernel void qwg_dflash_selector_greedy(
        device const float* hiddenProj   [[buffer(0)]],
        device const float* unary        [[buffer(1)]],
        device const int* candidates     [[buffer(2)]],
        device const ushort* predBook    [[buffer(3)]],
        device const ushort* succBook    [[buffer(4)]],
        device int* draftTokensOut       [[buffer(5)]],
        constant int& anchorToken        [[buffer(6)]],
        constant uint& draftCount        [[buffer(7)]],
        constant uint& topK              [[buffer(8)]],
        uint ti [[thread_index_in_threadgroup]]) {
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    threadgroup float sSums[8];
    threadgroup int sWinner;

    int pred = anchorToken;
    for (uint pos = 0; pos < draftCount; ++pos) {
        float hVal = hiddenProj[pos * 256u + ti];
        float pVal = dflash_bf16_to_f32(predBook[(size_t)pred * 256u + ti]) * hVal;
        float bestScore = -INFINITY;
        int bestToken = 0;

        for (uint k = 0; k < topK; ++k) {
            int cand = candidates[pos * 16u + k];
            float sVal = dflash_bf16_to_f32(succBook[(size_t)cand * 256u + ti]);
            float prod = simd_sum(pVal * sVal);
            if (lane == 0u) sSums[simd_id] = prod;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ti == 0u) {
                float edge = 0.0f;
                for (uint s = 0; s < 8u; ++s) edge += sSums[s];
                float score = unary[pos * 16u + k] + edge;
                if (score > bestScore) {
                    bestScore = score;
                    bestToken = cand;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (ti == 0u) {
            sWinner = bestToken;
            draftTokensOut[pos] = bestToken;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        pred = sWinner;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
