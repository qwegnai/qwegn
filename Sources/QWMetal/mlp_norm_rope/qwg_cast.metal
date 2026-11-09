// qwg_cast.metal — f32 → f16 seam cast (C16 chain leg; C21 spine seam too).
// The GEMV doors (qwg_gemv_q4g_v4 / head gemv) emit f32 (the accumulate
// law), while every downstream door (norm, rope, attn decode, MLP) takes
// Design note: docs/KERNEL_NOTES.md#qwg_cast.
#include <metal_stdlib>
using namespace metal;

// uint4 == 8 halves; H8 reinterprets it for element access in registers
// (same 16-byte pattern as qwnorm.metal — files stay self-contained).
struct H8 { half v[8]; } __attribute__((__packed__, __aligned__(16)));

kernel void qwg_cast_f32_to_f16(device const float* x [[buffer(0)]],
                                device uint4* y       [[buffer(1)]],
                                constant uint& nF8    [[buffer(2)]],   // groups of 8
                                uint tg [[threadgroup_position_in_grid]],
                                uint ti [[thread_index_in_threadgroup]]) {
    uint u = tg * 256u + ti;
    if (u >= nF8) return;
    device const float* xr = x + (size_t)u * 8u;
    H8 o;
    for (uint i = 0; i < 8; i++) o.v[i] = half(xr[i]);
    y[u] = *(thread uint4*)&o;
}
