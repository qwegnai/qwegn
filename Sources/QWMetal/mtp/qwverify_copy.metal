#include <metal_stdlib>
using namespace metal;

kernel void qwg_verify_copy(
        device const uint* source [[buffer(0)]],
        device uint* destination [[buffer(1)]],
        constant uint& count [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
    if (i < count) destination[i] = source[i];
}
