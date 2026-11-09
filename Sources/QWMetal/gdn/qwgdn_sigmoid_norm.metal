// qwgdn_sigmoid_norm.metal — GDN output epilogue: gated RMSNorm with a sigmoid gate + H_32 rotation.
// One threadgroup per value head (48) of 128 lanes; lane c owns column c.
// out[h,c] = H_32(o[h,c] * rsqrt(mean_c o^2 + 1e-6) * w[c] * sigmoid(z[h,c])).
// buffers: 0 o_raw f32 [48,128] | 1 z f16 [48,128] | 2 w f32 [128] | 3 out f16 [48,128]
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_geometry_constants.metalh"
#include "include/qw_weight_basis.metalh"
using namespace metal;

kernel void qwg_gdn_norm_sigmoid(device const float* o_raw [[buffer(0)]],
                                 device const half*  z     [[buffer(1)]],
                                 device const float* w     [[buffer(2)]],
                                 device half*        out   [[buffer(3)]],
                                 uint h [[threadgroup_position_in_grid]],
                                 uint c [[thread_position_in_threadgroup]]) {
    threadgroup float partial[4];
    const uint idx = h * 128u + c;
    const float o = o_raw[idx];
    const float sq = simd_sum(o * o);
    if ((c & 31u) == 0u) partial[c >> 5u] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = partial[0] + partial[1] + partial[2] + partial[3];
    const float rms = rsqrt(total * (1.0f / 128.0f) + QW_RMS_EPS);
    const float gate = 1.0f / (1.0f + exp(-float(z[idx])));
    float x = o * rms * w[c] * gate;
    // H_32 over consecutive dims (one simdgroup = 32 consecutive dims): the input basis of the HQ4 out_proj.
    const uint lane = c & 31u;
    x = qwg_fwht32_lane_f32(x, lane);
    out[idx] = half(x);
}
