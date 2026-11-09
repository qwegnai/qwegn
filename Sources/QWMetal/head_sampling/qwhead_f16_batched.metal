#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

struct QWHeadH8 { half v[8]; } __attribute__((__packed__, __aligned__(16)));

template<uint M>
static inline void qwhead_f16_batched(
        device const QWHeadH8* W, constant const QWHeadH8* x,
        device float* y, uint N, uint K, uint row, uint lane) {
    if (row >= N) return;
    const uint chunks = K / 8;
    device const QWHeadH8* wr = W + (size_t)row * chunks;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    for (uint c = lane; c < chunks; c += 32) {
        const QWHeadH8 weight = wr[c];
        for (int i = 0; i < 8; i++) {
            const float w = float(weight.v[i]);
            a0 += w * float(x[c].v[i]);
            a1 += w * float(x[chunks + c].v[i]);
            if (M >= 3) a2 += w * float(x[2 * chunks + c].v[i]);
            if (M >= 4) a3 += w * float(x[3 * chunks + c].v[i]);
            if (M >= 5) a4 += w * float(x[4 * chunks + c].v[i]);
            if (M >= 6) a5 += w * float(x[5 * chunks + c].v[i]);
            if (M >= 7) a6 += w * float(x[6 * chunks + c].v[i]);
            if (M >= 8) a7 += w * float(x[7 * chunks + c].v[i]);
        }
    }
    // Match the shipping GEMV's reduction order independently for each row.
    float4 v = float4(a0, a1, a2, a3);
    v += simd_shuffle_down(v, 16);
    v += simd_shuffle_down(v, 8);
    v += simd_shuffle_down(v, 4);
    v += simd_shuffle_down(v, 2);
    v += simd_shuffle_down(v, 1);
    float4 tail = float4(a4, a5, a6, a7);
    tail += simd_shuffle_down(tail, 16);
    tail += simd_shuffle_down(tail, 8);
    tail += simd_shuffle_down(tail, 4);
    tail += simd_shuffle_down(tail, 2);
    tail += simd_shuffle_down(tail, 1);
    if (lane == 0) {
        y[row] = v.x;
        y[(size_t)N + row] = v.y;
        if (M >= 3) y[(size_t)2 * N + row] = v.z;
        if (M >= 4) y[(size_t)3 * N + row] = v.w;
        if (M >= 5) y[(size_t)4 * N + row] = tail.x;
        if (M >= 6) y[(size_t)5 * N + row] = tail.y;
        if (M >= 7) y[(size_t)6 * N + row] = tail.z;
        if (M >= 8) y[(size_t)7 * N + row] = tail.w;
    }
}

kernel void qwg_head_f16_m2(
        device const QWHeadH8* W [[buffer(0)]],
        constant const QWHeadH8* x [[buffer(1)]],
        device float* y [[buffer(3)]],
        constant uint& N [[buffer(4)]],
        constant uint& K [[buffer(5)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    qwhead_f16_batched<2>(W, x, y, N, K, row, lane);
}

kernel void qwg_head_f16_m3(
        device const QWHeadH8* W [[buffer(0)]],
        constant const QWHeadH8* x [[buffer(1)]],
        device float* y [[buffer(3)]],
        constant uint& N [[buffer(4)]],
        constant uint& K [[buffer(5)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    qwhead_f16_batched<3>(W, x, y, N, K, row, lane);
}

kernel void qwg_head_f16_m4(
        device const QWHeadH8* W [[buffer(0)]],
        constant const QWHeadH8* x [[buffer(1)]],
        device float* y [[buffer(3)]],
        constant uint& N [[buffer(4)]],
        constant uint& K [[buffer(5)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    qwhead_f16_batched<4>(W, x, y, N, K, row, lane);
}

kernel void qwg_head_f16_m5(
        device const QWHeadH8* W [[buffer(0)]],
        constant const QWHeadH8* x [[buffer(1)]],
        device float* y [[buffer(3)]],
        constant uint& N [[buffer(4)]],
        constant uint& K [[buffer(5)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    qwhead_f16_batched<5>(W, x, y, N, K, row, lane);
}

kernel void qwg_head_f16_m6(
        device const QWHeadH8* W [[buffer(0)]],
        constant const QWHeadH8* x [[buffer(1)]],
        device float* y [[buffer(3)]],
        constant uint& N [[buffer(4)]],
        constant uint& K [[buffer(5)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    qwhead_f16_batched<6>(W, x, y, N, K, row, lane);
}

kernel void qwg_head_f16_m7(
        device const QWHeadH8* W [[buffer(0)]],
        constant const QWHeadH8* x [[buffer(1)]],
        device float* y [[buffer(3)]],
        constant uint& N [[buffer(4)]],
        constant uint& K [[buffer(5)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    qwhead_f16_batched<7>(W, x, y, N, K, row, lane);
}

kernel void qwg_head_f16_m8(
        device const QWHeadH8* W [[buffer(0)]],
        constant const QWHeadH8* x [[buffer(1)]],
        device float* y [[buffer(3)]],
        constant uint& N [[buffer(4)]],
        constant uint& K [[buffer(5)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    qwhead_f16_batched<8>(W, x, y, N, K, row, lane);
}
