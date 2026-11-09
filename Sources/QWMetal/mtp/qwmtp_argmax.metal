#include <metal_stdlib>
using namespace metal;

// UINT_MAX outranks every finite sortable score. Its low key is zero, so
// the shared final reduction returns ~0 (invalid token) if any row is bad.
static inline void qwg_mtp_argmax_checked_body(
        device const float* logits, device uint* keysHi, device uint* keysLo,
        uint N, bool canonicalZero, uint tg, uint tid,
        threadgroup uint* sh, threadgroup uint* sl) {
    const uint i = tg * 256u + tid;
    const float v = i < N ? logits[i] : -INFINITY;
    const uint raw = as_type<uint>(v);
    const bool invalid = i < N && (raw & 0x7f800000u) == 0x7f800000u;
    const uint bits = canonicalZero && (raw & 0x7fffffffu) == 0u ? 0u : raw;
    const uint hi = invalid ? 0xffffffffu : ((bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u));
    const uint lo = invalid || i >= N ? 0u : ~i;
    const uint maxHi = simd_max(hi);
    const uint maxLo = simd_max(hi == maxHi ? lo : 0u);
    if ((tid & 31u) == 0u) { sh[tid >> 5u] = maxHi; sl[tid >> 5u] = maxLo; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint bh = sh[0], bl = sl[0];
        for (uint q = 1u; q < 8u; q++) {
            if (sh[q] > bh || (sh[q] == bh && sl[q] > bl)) { bh = sh[q]; bl = sl[q]; }
        }
        keysHi[tg] = bh;
        keysLo[tg] = bl;
    }
}

kernel void qwg_mtp_argmax_checked(
        device const float* logits [[buffer(0)]],
        device uint* keysHi [[buffer(1)]],
        device uint* keysLo [[buffer(2)]],
        constant uint& N [[buffer(3)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint sh[8], sl[8];
    qwg_mtp_argmax_checked_body(logits, keysHi, keysLo, N, true, tg.x, tid, sh, sl);
}

kernel void qwg_mtp_argmax_target_checked(
        device const float* logits [[buffer(0)]],
        device uint* keysHi [[buffer(1)]],
        device uint* keysLo [[buffer(2)]],
        constant uint& N [[buffer(3)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint sh[8], sl[8];
    qwg_mtp_argmax_checked_body(logits, keysHi, keysLo, N, false, tg.x, tid, sh, sl);
}
