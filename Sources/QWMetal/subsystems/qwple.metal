// qwple.metal — Per-layer embedding math after key/value projections.
// gate: per (token, stream), g = <keyNorm, queryNorm> / sqrt(H); gate = sign(g) * sqrt(max(|g|, 1e-6));
//       gv[t, c, :] = sigmoid(gate) * vals[t, :].
// conv: out[t, c] = gv[t, c] + SiLU(sum_j w[c, j] * gvNorm[t - (K - 1 - j) * dilation, c]), causal, zero history.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

inline void qwg_ple_gate_impl(device const float* keyNorm, device const float* queryNorm,
                               device const float* vals, device float* gv,
                               uint H, uint hc, uint valStride, uint tg, uint tid,
                               threadgroup float* partial, threadgroup float* sig) {
    const uint W = H * hc;
    const uint t = tg / hc, c = tg - t * hc;
    float g = 0.0f;
    for (uint d = tid; d < H; d += 128u) g += keyNorm[t * W + c * H + d] * queryNorm[t * W + c * H + d];
    g = simd_sum(g);
    if ((tid & 31u) == 0u) partial[tid >> 5u] = g;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        const float total = (partial[0] + partial[1] + partial[2] + partial[3]) * rsqrt(float(H));
        const float mag = sqrt(max(fabs(total), 1e-6f));
        const float gate = (total > 0.0f) ? mag : ((total < 0.0f) ? -mag : 0.0f);
        sig[0] = 1.0f / (1.0f + exp(-gate));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < H; d += 128u) gv[t * W + c * H + d] = sig[0] * vals[t * valStride + d];
}

kernel void qwg_ple_gate(device const float* keyNorm [[buffer(0)]], device const float* queryNorm [[buffer(1)]],
                         device const float* vals [[buffer(2)]], device float* gv [[buffer(3)]],
                         constant uint& H [[buffer(4)]], constant uint& hc [[buffer(5)]],
                         uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[4], sig[1];
    qwg_ple_gate_impl(keyNorm, queryNorm, vals, gv, H, hc, H, tg, tid, partial, sig);
}

kernel void qwg_ple_gate_strided(device const float* keyNorm [[buffer(0)]], device const float* queryNorm [[buffer(1)]],
                                 device const float* vals [[buffer(2)]], device float* gv [[buffer(3)]],
                                 constant uint& H [[buffer(4)]], constant uint& hc [[buffer(5)]],
                                 constant uint& valStride [[buffer(6)]],
                                 uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[4], sig[1];
    qwg_ple_gate_impl(keyNorm, queryNorm, vals, gv, H, hc, valStride, tg, tid, partial, sig);
}

// conv buffers: 0 gv f32 [T, W] | 1 gvNorm f32 [T, W] | 2 convWeight f32 [W, K] | 3 out f32 [T, W] | 4 W | 5 K | 6 dilation | 7 T
// dispatch: (W, T) threads.
// Causal dilated depthwise conv over normed values; rows before the chunk come from `hist` ([HL, W]), HL = 0 = none.
kernel void qwg_ple_conv(device const float* gv         [[buffer(0)]],
                         device const float* gvNorm     [[buffer(1)]],
                         device const float* convWeight [[buffer(2)]],
                         device float*       out        [[buffer(3)]],
                         constant uint&      W          [[buffer(4)]],
                         constant uint&      K          [[buffer(5)]],
                         constant uint&      dilation   [[buffer(6)]],
                         constant uint&      T          [[buffer(7)]],
                         device const float* hist       [[buffer(8)]],
                         constant uint&      HL         [[buffer(9)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, t = gid.y;
    if (c >= W || t >= T) return;
    float acc = 0.0f;
    for (uint j = 0u; j < K; j++) {
        const int src = int(t) - int((K - 1u - j) * dilation);
        if (src >= 0) acc += convWeight[c * K + j] * gvNorm[uint(src) * W + c];
        else if (HL > 0u) acc += convWeight[c * K + j] * hist[(int(HL) + src) * int(W) + int(c)];
    }
    out[t * W + c] = gv[t * W + c] + acc / (1.0f + exp(-acc));
}

// History roll: newHist[i] = the last HL rows of [oldHist (HL rows) | gvNorm (T rows)]. dispatch: (W, HL) threads.
kernel void qwg_ple_hist(device const float* oldHist [[buffer(0)]],
                         device const float* gvNorm  [[buffer(1)]],
                         device float*       newHist [[buffer(2)]],
                         constant uint&      W       [[buffer(3)]],
                         constant uint&      HL      [[buffer(4)]],
                         constant uint&      T       [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, i = gid.y;
    if (c >= W || i >= HL) return;
    const uint k = i + T;
    newHist[i * W + c] = k < HL ? oldHist[k * W + c] : gvNorm[(k - HL) * W + c];
}

// inject buffers: 0 base f32 [T, C, H] | 1 src f32 [T, H] | 2 inj f32 [T, C] | 3 out f32 [T, C, H] | 4 H | 5 C
// Residual injection of a sub-layer output into the HC streams: out[t, c, :] = base[t, c, :] + src[t, :] * inj[t, c].
// dispatch: (C * H, T) threads.
kernel void qwg_inject(device const float* base [[buffer(0)]],
                       device const float* src  [[buffer(1)]],
                       device const float* inj  [[buffer(2)]],
                       device float*       out  [[buffer(3)]],
                       constant uint&      H    [[buffer(4)]],
                       constant uint&      C    [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x / H, d = gid.x - c * H, t = gid.y;
    if (c >= C) return;
    const uint o = (t * C + c) * H + d;
    out[o] = base[o] + src[t * H + d] * inj[t * C + c];
}
