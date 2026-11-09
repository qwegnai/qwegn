// qwattn_paged_general.metal — the standard path's PAGED prefill: one
// scalar kernel that attends a chunk of T queries at `pos` against all the
// keys in [0, pos+T), reading the K/V out of the paged KV pool through the
// page table. One simdgroup per (query, head): the 32 lanes × 8 halves map
// onto the head's OWN width — the head window: the lanes past the head's
// data carry zeros, so the fold sees the head's elements only. The law is
// the plain prefill's: causal, the softmax, the gate, the rotation (C109).
#include <metal_stdlib>
#include "include/qw_geometry_constants.metalh"
#include "include/qw_dequant.metalh"
#include "include/qw_kv_paged.metalh"
using namespace metal;

kernel void qwg_attn_prefill_paged_general(const device half* qgate [[buffer(0)]],
                                           device half*       kpool [[buffer(1)]],
                                           device half*       vpool [[buffer(2)]],
                                           const device int*  ktab  [[buffer(3)]],
                                           const device int*  vtab  [[buffer(4)]],
                                           device half*       outp  [[buffer(5)]],
                                           constant uint&     pos   [[buffer(6)]],
                                           constant uint&     q8mode [[buffer(7)]],
                                           uint tid [[thread_index_in_threadgroup]],
                                           uint3 tg [[threadgroup_position_in_grid]]) {
    const uint head = tg.x;                       // the q head
    const uint qi = tg.y;                         // the query of this chunk
    const uint lane = tid & 31u;                  // 32 lanes onto the head
    const uint d0 = lane * 8u;
    const bool live = d0 < QW_HEAD_DIM;           // the head window
    const uint kvh = head / QW_GQA_RATIO;

    // The query's row: the q, and (when the family gates) the gate that
    // follows it in the same row.
    const device half* qrow = qgate + (uint64_t)qi * QW_QPROJ_OUT
                            + (uint64_t)head * QW_Q_STRIDE * QW_HEAD_DIM;
    float q8[8];
    for (uint i = 0; i < 8u; i++) q8[i] = 0.0f;
    if (live) {
        for (uint i = 0; i < 8u; i++) q8[i] = float(qrow[d0 + i]);
    }
    float gf[8];
    for (uint i = 0; i < 8u; i++) gf[i] = 1.0f;
    if (QW_HAS_Q_GATE && live) {
        const device half* grow = qrow + QW_HEAD_DIM;
        for (uint i = 0; i < 8u; i++) {
            gf[i] = 1.0f / (1.0f + metal::exp(-float(grow[d0 + i])));
        }
    }

    const uint kvRow = QW_FC_KV_ROW;
    const uint kvPgel = QW_FC_KV_PGEL;
    const uint kvEnd = pos + qi + 1u;              // causal: [0, pos+qi]

    float m = -1e30f, lsum = 0.0f;
    float acc[8];
    for (uint i = 0; i < 8u; i++) acc[i] = 0.0f;

    for (uint j = 0; j < kvEnd; j++) {
        // The page table's indirection: a physical page id, then the row.
        const uint page = j >> 8, slot = j & 255u;
        const uint pk = (uint)ktab[page];
        const uint pv = (uint)vtab[page];
        float k8[8], v8[8];
        for (uint i = 0; i < 8u; i++) { k8[i] = 0.0f; v8[i] = 0.0f; }
        if (live) {
            if (q8mode == 0u) {
                const uint64_t rowOff = (uint64_t)slot * kvRow + (uint64_t)kvh * QW_HEAD_DIM;
                device const half* sk = kpool + (uint64_t)pk * kvPgel + rowOff;
                device const half* sv = vpool + (uint64_t)pv * kvPgel + rowOff;
                for (uint i = 0; i < 8u; i++) { k8[i] = float(sk[d0 + i]); v8[i] = float(sv[d0 + i]); }
            } else {
                half kh[8], vh[8];
                QW_KV_LOAD8(kh, kpool, pk, slot, kvh, d0, q8mode)
                QW_KV_LOAD8(vh, vpool, pv, slot, kvh, d0, q8mode)
                for (uint i = 0; i < 8u; i++) { k8[i] = float(kh[i]); v8[i] = float(vh[i]); }
            }
        }
        float dot = 0.0f;
        for (uint i = 0; i < 8u; i++) dot += q8[i] * k8[i];
        const float s = simd_sum(dot) * QW_ATTN_SCALE;
        // The online softmax: the running max, the rescale of what is
        // accumulated, and this key's share.
        const float mnew = max(m, s);
        const float corr = (m > -1e29f) ? metal::exp(m - mnew) : 1.0f;
        const float p = metal::exp(s - mnew);
        for (uint i = 0; i < 8u; i++) acc[i] = acc[i] * corr + p * v8[i];
        lsum = lsum * corr + p;
        m = mnew;
    }

    // o = acc / lsum, the gate, and the 32-point rotation. The lanes past
    // the head's data hold zeros and write nothing.
    half hv[8];
    for (uint i = 0; i < 8u; i++) {
        hv[i] = live ? half(acc[i] / lsum * gf[i]) : half(0.0f);
    }
    fwht32_inplace(hv, lane);
    if (live) {
        device half* o = outp + (uint64_t)qi * (QW_Q_HEADS * QW_HEAD_DIM)
                       + (uint64_t)head * QW_HEAD_DIM + d0;
        for (uint i = 0; i < 8u; i++) o[i] = hv[i];
    }
}
