// qw_kv_write_general.metal — the paged KV writer for a head the frozen
// writer cannot serve. The frozen writer's row stride and its per-head
// scale/group map are the 27B's 4 heads × 256 halves; this one takes the
// strides from the geometry and maps the same 32 lanes × 8 halves onto the
// head's OWN width — the head window: a lane past the head's data writes
// nothing. `q8mode` selects f16 (0), q8 (1) or q4 (2) pages; the compressed
// segments keep the head-window mapping, so every head width ≤ 256 is served.
#include <metal_stdlib>
#include "include/qw_geometry_constants.metalh"
#include "include/qw_kv_paged.metalh"
using namespace metal;

kernel void qwg_kv_paged_write_general(device half*       kpool    [[buffer(0)]],
                                      device half*       vpool    [[buffer(1)]],
                                      const device int*  ktab     [[buffer(2)]],
                                      const device int*  vtab     [[buffer(3)]],
                                      device const half* k        [[buffer(4)]],
                                      device const half* v        [[buffer(5)]],
                                      constant uint&     startPos [[buffer(6)]],
                                      constant uint&     T        [[buffer(7)]],
                                      constant uint&     q8mode   [[buffer(8)]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint3 tg [[threadgroup_position_in_grid]]) {
    const uint kvh = tg.x;   // the kv head
    const uint t = tg.y;     // the token of this chunk
    if (t >= T) return;
    const uint lane = tid & 31u;
    const uint d0 = lane * 8u;
    const bool live = (d0 + 8u <= QW_HEAD_DIM);

    const uint tok = startPos + t;
    const uint page = tok >> 8;
    const uint slot = tok & 255u;

    const uint kvRow = QW_FC_KV_ROW;
    const uint kvPgel = QW_FC_KV_PGEL;

    if (q8mode == 0u) {
        if (!live) return;
        device const half* kn = k + (uint64_t)t * kvRow + kvh * QW_HEAD_DIM + d0;
        device const half* vn = v + (uint64_t)t * kvRow + kvh * QW_HEAD_DIM + d0;
        device half* kw = kpool + (uint64_t)ktab[page] * kvPgel
                          + (uint64_t)slot * kvRow + kvh * QW_HEAD_DIM + d0;
        device half* vw = vpool + (uint64_t)vtab[page] * kvPgel
                          + (uint64_t)slot * kvRow + kvh * QW_HEAD_DIM + d0;
        *(device uint4*)kw = *(device const uint4*)kn;
        *(device uint4*)vw = *(device const uint4*)vn;
        return;
    }

    half kv8[8], vv8[8];
    float kMax = 0.0f, vMax = 0.0f;
    if (live) {
        device const half* kn = k + (uint64_t)t * kvRow + kvh * QW_HEAD_DIM + d0;
        device const half* vn = v + (uint64_t)t * kvRow + kvh * QW_HEAD_DIM + d0;
        for (uint i = 0; i < 8u; i++) {
            kv8[i] = kn[i];
            vv8[i] = vn[i];
            kMax = fmax(kMax, fabs(float(kv8[i])));
            vMax = fmax(vMax, fabs(float(vv8[i])));
        }
    }
    if (q8mode == 1u) {
        const float kSegMax = simd_max(kMax);
        const float vSegMax = simd_max(vMax);
        if (!live) return;
        qwg_q8_store_seg((device char*)kpool, uint64_t(ktab[page]) * QW_Q8_PGE, slot, kvh, d0, kv8, kSegMax);
        qwg_q8_store_seg((device char*)vpool, uint64_t(vtab[page]) * QW_Q8_PGE, slot, kvh, d0, vv8, vSegMax);
        return;
    }
    float mnk = 1e30f, mxk = -1e30f, mnv4 = 1e30f, mxv4 = -1e30f;
    if (live) {
        for (uint i = 0; i < 8u; i++) {
            const float a = float(kv8[i]), b = float(vv8[i]);
            mnk = fmin(mnk, a); mxk = fmax(mxk, a);
            mnv4 = fmin(mnv4, b); mxv4 = fmax(mxv4, b);
        }
    }
    #pragma unroll
    for (uint sh = 1u; sh <= 8u; sh <<= 1) {
        mnk = fmin(mnk, simd_shuffle_xor(mnk, sh)); mxk = fmax(mxk, simd_shuffle_xor(mxk, sh));
        mnv4 = fmin(mnv4, simd_shuffle_xor(mnv4, sh)); mxv4 = fmax(mxv4, simd_shuffle_xor(mxv4, sh));
    }
    if (!live) return;
    qwg_q4_store_seg((device uchar*)kpool, uint64_t(ktab[page]) * QW_Q4_PGE, slot, kvh, d0, kv8, mnk, mxk);
    qwg_q4_store_seg((device uchar*)vpool, uint64_t(vtab[page]) * QW_Q4_PGE, slot, kvh, d0, vv8, mnv4, mxv4);
}
