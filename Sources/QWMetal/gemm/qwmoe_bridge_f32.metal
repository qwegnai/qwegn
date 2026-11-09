// qwmoe_bridge_f32.metal — the standard MoE path's f32 bridge.
//
// The dense engine's residual stream is f16; every MoE operator reads f32.
// One pass over the f16 normed hidden produces the two f32 copies the chain
// needs: the rotated one (the expert GEMMs' input) and the un-rotated one
// (the router's input). The un-rotation is the 32-point WHT — the butterfly
// the norm applies, which is its own inverse.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
#include "qw_dequant.metalh"

// uint4 == 8 halves; H8 reinterprets it for element access in registers.
struct H8 { half v[8]; } __attribute__((__packed__, __aligned__(16)));

kernel void qwg_moe_bridge_f16_to_f32(device const half* xRot    [[buffer(0)]],
                                       device float*      xRot32  [[buffer(1)]],
                                       device float*      xPlain32 [[buffer(2)]],
                                       constant uint&     H       [[buffer(3)]],
                                       uint2 tg2 [[threadgroup_position_in_grid]],
                                       uint2 ng2 [[threadgroups_per_grid]],
                                       uint ti [[thread_index_in_threadgroup]]) {
    // Row in y; a single-row (decode) dispatch spreads the row's uint4 blocks over x threadgroups of 32 lanes.
    const uint tg = tg2.y;
    // Every MoE operator reads f32: the gate/up kernel reads
    // `device const float* x`, the router reads `device const float* X`, and
    // the down kernel reads f32 too. The dense engine's residual stream is
    // f16 everywhere — every norm kernel ends in `fwht32_f32_to_h8` — so the
    // chain needs a widening, and the engine has no kernel that does one.
    //
    // The two copies are:
    //   xRot32   — the ROTATED hidden, the one the expert GEMMs read. The
    //              packer rotates every 2-D weight's input dimension, so the
    //              activation entering a rotated weight is rotated too.
    //   xPlain32 — the UN-ROTATED hidden, the one the router reads. The
    //              router's weight is the packer's RAW, un-rotated plane, so
    //              the router must read the hidden in the standard basis.
    //
    // The un-rotation is the 32-point WHT — the same butterfly the norm
    // kernel applies, and the WHT is its own inverse. It is done in f32 by
    // reusing the shared header's `fwht32_f32_to_split_h8` (the f32 butterfly
    // that emits an f16 hi/lo pair) and recombining the pair into f32 — no
    // third copy of the butterfly in the tree.
    const bool skipRot32 = (H & 0x80000000u) != 0u;
    const uint realH = H & 0x7FFFFFFFu;
    const uint hiddenU4 = realH >> 3;             // uint4 groups per row (8 halves each)
    device const uint4* xr = (device const uint4*)xRot + (size_t)tg * hiddenU4;
    device float* rot32 = xRot32 + (size_t)tg * realH;
    device float* plain32 = xPlain32 + (size_t)tg * realH;
    // One threadgroup per row, 32 threads — the same lane geometry the
    // norm kernel's butterfly uses, so the 4-lane groups line up.
    for (uint u = tg2.x * 32u + ti; u < hiddenU4; u += 32u * ng2.x) {
        uint4 w = xr[u];
        H8 v = *(thread H8*)&w;
        float f[8];
        for (int i = 0; i < 8; i++) {
            f[i] = float(v.v[i]);
            if (!skipRot32) rot32[u * 8u + i] = f[i]; // the rotated copy: a plain widening
        }
        half hi[8], lo[8];
        if (QW_FWHT_ROTATED) {
            fwht32_f32_to_split_h8(f, hi, lo, ti);    // the WHT is its own inverse
            for (int i = 0; i < 8; i++) {
                plain32[u * 8u + i] = float(hi[i]) + float(lo[i]);
            }
        } else {
            // The STANDARD basis: there is nothing to un-rotate, because the
            // hidden that reaches this bridge is already the plain one. The
            // helper's own gate already makes that true — this branch is a
            // SHORT-CIRCUIT, not what makes the basis right. It is here to
            // keep the f32 value: the helper's standard path would split it
            // into an f16 hi/lo pair and back, and the router's copy must be
            // the number it started from.
            for (int i = 0; i < 8; i++) {
                plain32[u * 8u + i] = f[i];
            }
        }
    }
}
