// qw_basis_marker.metal — which basis was this metallib compiled for?
//
// The basis is a build-time property (specs/02-qwhal.md), so there are two
// libraries, one per basis. The marker lets the HAL find out which one it
// got and refuse a mismatch, rather than run a model in the wrong basis.
// Only the NAME is read; the kernel is never dispatched.
#include <metal_stdlib>
using namespace metal;

#if defined(QW_BASIS_STANDARD)
/// Present only in the STANDARD-basis build.
kernel void qwg_basis_marker_standard(device uint* out [[buffer(0)]]) {
    if (out != nullptr) out[0] = 0u;
}
#else
/// Present only in the ROTATED build — the incumbent's own format.
kernel void qwg_basis_marker(device uint* out [[buffer(0)]]) {
    if (out != nullptr) out[0] = 1u;
}
#endif
