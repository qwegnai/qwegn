// qwengine.metal — embed/lm_head bf16 planes → f16 engine storage.
// A bf16 value is the top 16 bits of an f32: widening is exact, the
// f32→f16 narrowing is RN-even (same rounding as torch .half()).
#include <metal_stdlib>
using namespace metal;

kernel void qwg_bf16_to_f16(device const ushort* src [[buffer(0)]],
                            device half* dst           [[buffer(1)]],
                            constant uint& start       [[buffer(2)]],
                            constant uint& n           [[buffer(3)]],
                            uint2 tg                   [[threadgroup_position_in_grid]],
                            uint ti                    [[thread_index_in_threadgroup]]) {
    const uint i = tg.x * 256u + ti;
    if (i >= n) return;
    dst[i] = half(as_type<float>(uint(src[start + i]) << 16));
}
