#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

struct QWHeadH8 { half v[8]; } __attribute__((__packed__, __aligned__(16)));

static inline float4 qwhead_simd_sum4(float4 v) {
    v += simd_shuffle_down(v, 16);
    v += simd_shuffle_down(v, 8);
    v += simd_shuffle_down(v, 4);
    v += simd_shuffle_down(v, 2);
    v += simd_shuffle_down(v, 1);
    return v;
}

template<uint M>
static inline void qwhead_f16_batched(
        device const QWHeadH8* W, device const QWHeadH8* x,
        device float* y, uint N, uint K, uint tg, uint ti) {
    const uint row0 = (tg * 4u + (ti >> 5u)) * 2u;
    const uint lane = ti & 31u;
    if (row0 >= N) return;
    const bool hasRow1 = (row0 + 1u < N);
    const uint chunks = K / 8;
    device const QWHeadH8* wr0 = W + (size_t)row0 * chunks;
    device const QWHeadH8* wr1 = hasRow1 ? (wr0 + chunks) : wr0;
    float a0[8] = {};
    float a1[8] = {};
    for (uint c = lane; c < chunks; c += 32) {
        const QWHeadH8 w0 = wr0[c];
        const QWHeadH8 w1 = hasRow1 ? wr1[c] : w0;
        #pragma unroll
        for (uint m = 0; m < M; m++) {
            const QWHeadH8 xm = x[(size_t)m * chunks + c];
            for (int i = 0; i < 8; i++) {
                const float xv = float(xm.v[i]);
                a0[m] += float(w0.v[i]) * xv;
                if (hasRow1) a1[m] += float(w1.v[i]) * xv;
            }
        }
    }
    // Match qwg_gemv_f16_v4's shuffle-down reduction order per (vocabRow, batchRow).
    float4 lo0 = qwhead_simd_sum4(float4(a0[0], a0[1], M >= 3 ? a0[2] : 0.0f, M >= 4 ? a0[3] : 0.0f));
    float4 hi0 = float4(0.0f);
    if (M > 4) {
        hi0 = qwhead_simd_sum4(float4(a0[4], M >= 6 ? a0[5] : 0.0f, M >= 7 ? a0[6] : 0.0f, M >= 8 ? a0[7] : 0.0f));
    }
    float4 lo1 = float4(0.0f);
    float4 hi1 = float4(0.0f);
    if (hasRow1) {
        lo1 = qwhead_simd_sum4(float4(a1[0], a1[1], M >= 3 ? a1[2] : 0.0f, M >= 4 ? a1[3] : 0.0f));
        if (M > 4) {
            hi1 = qwhead_simd_sum4(float4(a1[4], M >= 6 ? a1[5] : 0.0f, M >= 7 ? a1[6] : 0.0f, M >= 8 ? a1[7] : 0.0f));
        }
    }
    if (lane == 0u) {
        y[row0] = lo0.x;
        y[(size_t)N + row0] = lo0.y;
        if (M >= 3) y[(size_t)2 * N + row0] = lo0.z;
        if (M >= 4) y[(size_t)3 * N + row0] = lo0.w;
        if (M >= 5) y[(size_t)4 * N + row0] = hi0.x;
        if (M >= 6) y[(size_t)5 * N + row0] = hi0.y;
        if (M >= 7) y[(size_t)6 * N + row0] = hi0.z;
        if (M >= 8) y[(size_t)7 * N + row0] = hi0.w;
        if (hasRow1) {
            const uint row1 = row0 + 1u;
            y[row1] = lo1.x;
            y[(size_t)N + row1] = lo1.y;
            if (M >= 3) y[(size_t)2 * N + row1] = lo1.z;
            if (M >= 4) y[(size_t)3 * N + row1] = lo1.w;
            if (M >= 5) y[(size_t)4 * N + row1] = hi1.x;
            if (M >= 6) y[(size_t)5 * N + row1] = hi1.y;
            if (M >= 7) y[(size_t)6 * N + row1] = hi1.z;
            if (M >= 8) y[(size_t)7 * N + row1] = hi1.w;
        }
    }
}

#define QW_HEAD_F16_ENTRY(M) \
kernel void qwg_head_f16_m##M( \
        device const QWHeadH8* W [[buffer(0)]], \
        device const QWHeadH8* x [[buffer(1)]], \
        device float* y [[buffer(3)]], \
        constant uint& N [[buffer(4)]], \
        constant uint& K [[buffer(5)]], \
        uint tg [[threadgroup_position_in_grid]], \
        uint ti [[thread_index_in_threadgroup]]) { \
    qwhead_f16_batched<M>(W, x, y, N, K, tg, ti); \
}
QW_HEAD_F16_ENTRY(2)
QW_HEAD_F16_ENTRY(3)
QW_HEAD_F16_ENTRY(4)
QW_HEAD_F16_ENTRY(5)
QW_HEAD_F16_ENTRY(6)
QW_HEAD_F16_ENTRY(7)
QW_HEAD_F16_ENTRY(8)

// Isolated F32 logit softcap epilogue (`cap * precise::tanh(clamp(v / cap, -15, 15))`, 0 B LUTs).
kernel void qwg_head_softcap_f32(
        device float*   logits  [[buffer(0)]],
        constant uint&  count   [[buffer(1)]],
        constant float& softcap [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= count || !(softcap > 0.0f)) return;
    const float v = logits[gid];
    logits[gid] = precise::tanh(clamp(v / softcap, -15.0f, 15.0f)) * softcap;
}
