// qwattn_decode_general.metal — the standard path's decode attention: a
// head narrower than 256 halves. The frozen decode maps a simdgroup's 32
// lanes × 8 halves onto ONE head; this one keeps the mapping and MASKS the
// lanes past the head's own width (the head window), so they carry zeros
// into the simdgroup collectives and write nothing. The staging stays
// 256-wide per key. The f16, q8 and q4 page formats are all read and
// written through the same head window.
#include <metal_stdlib>
#include "include/qw_geometry_constants.metalh"
#include "include/qw_dequant.metalh"
#include "include/qw_kv_paged.metalh"
using namespace metal;

#define QW_MAXSPLITS 64u
#define QW_MAXKEYS   16u   // keys staged per phase: 2 × the GQA ratio, capped
#define QW_HEAD_DIM_MAX 256u

/// The head's logit: the fold is the FULL 32-lane simd_sum (the dead lanes
/// carry zeros, so the sum is over the head's own elements), then the
/// model's scale 1/sqrt(head_dim).
static inline float qwg_dec_logit_general(float v) {
    return simd_sum(v) * QW_ATTN_SCALE;
}

template <uint MODE>
static inline void qwg_decode_general_body(
        const device half* qgate, device half* kpool, device half* vpool,
        const device int* ktab, const device int* vtab,
        const device half* knew, const device half* vnew, device float* partials,
        device half* outp,
        uint C, uint splits, uint tid, uint3 tg,
        threadgroup half* stageK, threadgroup half* stageV) {
    const uint sg   = tid >> 5;
    const uint lane = tid & 31u;
    const uint head = tg.x * QW_GQA_RATIO + sg;
    const uint kvh  = head / QW_GQA_RATIO;
    const uint d0   = lane * 8u;
    const bool live = d0 < QW_HEAD_DIM;      // the head window

    const uint nKeys = C + 1u;
    const uint per = (nKeys + splits - 1u) / splits;
    const uint k0 = min(tg.y * per, nKeys);
    const uint k1 = min(k0 + per, nKeys);
    const uint cEnd = min(k1, C);
    const uint keysPerPhase = 2u * QW_GQA_RATIO;

    float q8[8];
    if (live) {
        device const half* q4 = qgate + head * QW_Q_STRIDE * QW_HEAD_DIM + d0;
        for (uint i = 0; i < 8u; i++) q8[i] = float(q4[i]);
    } else {
        for (uint i = 0; i < 8u; i++) q8[i] = 0.0f;
    }

    float m = -1e30f, lsum = 0.0f;
    float oacc[8]; for (uint i = 0; i < 8u; i++) oacc[i] = 0.0f;

    #define QW_MERGE(s, vv) {                                            \
        const float _mn = max(m, s);                                     \
        const float _c  = (s > m) ? metal::exp(m - s) : 1.0f;             \
        const float _p  = metal::exp(s - _mn);                           \
        lsum = lsum * _c + _p;                                           \
        for (uint _i = 0; _i < 8u; _i++) oacc[_i] = oacc[_i] * _c + _p * float(vv[_i]); \
        m = _mn;                                                         \
    }
    #define QW_DOT(kk) ({                                                \
        float _pd = 0.0f;                                                \
        for (uint _i = 0; _i < 8u; _i++) _pd += q8[_i] * float(kk[_i]);  \
        qwg_dec_logit_general(_pd); })

    uint r = k0;
    for (; r + keysPerPhase <= cEnd; r += keysPerPhase) {
        if (QW_HEAD_DIM <= 128u) {
            const uint keyIdx = tid >> 4u;
            const uint col = (tid & 15u) << 3u;
            const uint row = r + keyIdx;
            const uint pk = (uint)ktab[row >> 8], pv = (uint)vtab[row >> 8];
            const uint e = keyIdx * QW_HEAD_DIM_MAX + col;
            threadgroup half* dk = &stageK[e]; threadgroup half* dv = &stageV[e];
            if (col < QW_HEAD_DIM) {
                if (MODE == 0u) {
                    const uint off = (row & 255u) * QW_ROW + kvh * QW_HEAD_DIM + col;
                    device const half* sk = kpool + (uint64_t)pk * QW_PGEL + off;
                    device const half* sv = vpool + (uint64_t)pv * QW_PGEL + off;
                    *(threadgroup uint4*)dk = *(device const uint4*)sk;
                    *(threadgroup uint4*)dv = *(device const uint4*)sv;
                } else {
                    half ka[8], va[8];
                    QW_KV_LOAD8(ka, kpool, pk, row & 255u, kvh, col, MODE)
                    QW_KV_LOAD8(va, vpool, pv, row & 255u, kvh, col, MODE)
                    #pragma unroll
                    for (uint i = 0; i < 8u; i++) { dk[i] = ka[i]; dv[i] = va[i]; }
                }
            }
        } else {
            #pragma unroll
            for (uint s = 0; s < 2u; ++s) {
                const uint e = s * (QW_GQA_RATIO * 32u * 8u) + tid * 8u;
                const uint row = r + (e >> 8);
                const uint pk = (uint)ktab[row >> 8], pv = (uint)vtab[row >> 8];
                threadgroup half* dk = &stageK[e]; threadgroup half* dv = &stageV[e];
                if (live) {
                    if (MODE == 0u) {
                        const uint off = (row & 255u) * QW_ROW + kvh * QW_HEAD_DIM + (e & 255u);
                        device const half* sk = kpool + (uint64_t)pk * QW_PGEL + off;
                        device const half* sv = vpool + (uint64_t)pv * QW_PGEL + off;
                        *(threadgroup uint4*)dk = *(device const uint4*)sk;
                        *(threadgroup uint4*)dv = *(device const uint4*)sv;
                    } else {
                        half ka[8], va[8];
                        QW_KV_LOAD8(ka, kpool, pk, row & 255u, kvh, e & 255u, MODE)
                        QW_KV_LOAD8(va, vpool, pv, row & 255u, kvh, e & 255u, MODE)
                        #pragma unroll
                        for (uint i = 0; i < 8u; i++) { dk[i] = ka[i]; dv[i] = va[i]; }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s12[QW_MAXKEYS];
        for (uint j = 0; j < keysPerPhase; j++) {
            threadgroup const half* ks = &stageK[j * QW_HEAD_DIM_MAX + d0];
            half kk[8];
            if (live) {
                *(thread uint4*)kk = *(threadgroup const uint4*)ks;
            } else {
                #pragma unroll
                for (uint i = 0; i < 8u; i++) kk[i] = half(0.0h);
            }
            s12[j] = QW_DOT(kk);
        }
        float bm = s12[0];
        for (uint j = 1; j < keysPerPhase; j++) bm = max(bm, s12[j]);
        const float Mb = max(m, bm);
        float lp = 0.0f; float op[8];
        #pragma unroll
        for (uint i = 0; i < 8u; i++) op[i] = 0.0f;
        for (uint j = 0; j < keysPerPhase; j++) {
            const float pj = metal::exp(s12[j] - Mb);
            lp += pj;
            threadgroup const half* vs = &stageV[j * QW_HEAD_DIM_MAX + d0];
            half vv[8];
            if (live) {
                *(thread uint4*)vv = *(threadgroup const uint4*)vs;
            } else {
                #pragma unroll
                for (uint i = 0; i < 8u; i++) vv[i] = half(0.0h);
            }
            #pragma unroll
            for (uint i = 0; i < 8u; i++) op[i] += pj * float(vv[i]);
        }
        const float cb = metal::exp(m - Mb);
        lsum = lsum * cb + lp;
        #pragma unroll
        for (uint i = 0; i < 8u; i++) oacc[i] = oacc[i] * cb + op[i];
        m = Mb;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (; r < cEnd; r++) {
        half kk[8], vv[8];
        if (live) {
            if (MODE == 0u) {
                device const half* kb = kpool + (uint64_t)ktab[r >> 8] * QW_PGEL
                                      + (uint64_t)(r & 255u) * QW_ROW + kvh * QW_HEAD_DIM + d0;
                device const half* vb = vpool + (uint64_t)vtab[r >> 8] * QW_PGEL
                                      + (uint64_t)(r & 255u) * QW_ROW + kvh * QW_HEAD_DIM + d0;
                *(thread uint4*)kk = *(device const uint4*)kb;
                *(thread uint4*)vv = *(device const uint4*)vb;
            } else {
                QW_KV_LOAD8(kk, kpool, ktab[r >> 8], r & 255u, kvh, d0, MODE)
                QW_KV_LOAD8(vv, vpool, vtab[r >> 8], r & 255u, kvh, d0, MODE)
            }
        } else {
            for (uint i = 0; i < 8u; i++) { kk[i] = half(0.0h); vv[i] = half(0.0h); }
        }
        QW_MERGE(QW_DOT(kk), vv)
    }
    if (C >= k0 && C < k1) {
        half kk[8], vv[8];
        if (live) {
            device const half* kb = knew + kvh * QW_HEAD_DIM + d0;
            device const half* vb = vnew + kvh * QW_HEAD_DIM + d0;
            *(thread uint4*)kk = *(device const uint4*)kb;
            *(thread uint4*)vv = *(device const uint4*)vb;
        } else {
            for (uint i = 0; i < 8u; i++) { kk[i] = half(0.0h); vv[i] = half(0.0h); }
        }
        QW_MERGE(QW_DOT(kk), vv)
    }

    // The partials row per (head, split): [O head_dim f32][m][l].
    device float* pb = partials + (head * QW_MAXSPLITS + tg.y) * (QW_HEAD_DIM + 2u);
    if (live) {
        for (uint i = 0; i < 8u; i++) pb[d0 + i] = oacc[i];
    }
    if (lane == 0u) { pb[QW_HEAD_DIM] = m; pb[QW_HEAD_DIM + 1u] = lsum; }

    if (splits == 1u && live) {
        float gf[8];
        #pragma unroll
        for (uint i = 0; i < 8u; i++) gf[i] = 1.0f;
        if (QW_HAS_Q_GATE) {
            device const half* g4 = qgate + head * QW_Q_STRIDE * QW_HEAD_DIM + QW_HEAD_DIM + d0;
            #pragma unroll
            for (uint i = 0; i < 8u; i++) gf[i] = 1.0f / (1.0f + metal::exp(-float(g4[i])));
        }
        const float linv = (lsum > 0.0f) ? 1.0f / lsum : 0.0f;
        half hv[8];
        #pragma unroll
        for (uint i = 0; i < 8u; i++) hv[i] = half(oacc[i] * linv * gf[i]);
        fwht32_inplace(hv, lane);
        device half* ov = outp + head * QW_HEAD_DIM + d0;
        *(device uint4*)ov = *(thread uint4*)hv;
    }

    // The new token's K/V go into the page slot: split 0, one simdgroup.
    if (tg.y == 0u && sg == 0u) {
        if (MODE == 0u) {
            if (live) {
                device half* kw = kpool + (uint64_t)ktab[C >> 8] * QW_PGEL
                                  + (uint64_t)(C & 255u) * QW_ROW + kvh * QW_HEAD_DIM + d0;
                device half* vw = vpool + (uint64_t)vtab[C >> 8] * QW_PGEL
                                  + (uint64_t)(C & 255u) * QW_ROW + kvh * QW_HEAD_DIM + d0;
                *(device uint4*)kw = *(device const uint4*)(knew + kvh * QW_HEAD_DIM + d0);
                *(device uint4*)vw = *(device const uint4*)(vnew + kvh * QW_HEAD_DIM + d0);
            }
        } else if (MODE == 1u) {
            half kn8[8], vn8[8];
            float kMax = 0.0f, vMax = 0.0f;
            if (live) {
                for (uint i = 0; i < 8u; i++) {
                    kn8[i] = knew[kvh * QW_HEAD_DIM + d0 + i];
                    vn8[i] = vnew[kvh * QW_HEAD_DIM + d0 + i];
                    kMax = fmax(kMax, fabs(float(kn8[i])));
                    vMax = fmax(vMax, fabs(float(vn8[i])));
                }
            }
            const float kSegMax = simd_max(kMax);
            const float vSegMax = simd_max(vMax);
            if (live) {
                qwg_q8_store_seg((device char*)kpool, uint64_t(ktab[C >> 8]) * QW_Q8_PGE, C & 255u, kvh, d0, kn8, kSegMax);
                qwg_q8_store_seg((device char*)vpool, uint64_t(vtab[C >> 8]) * QW_Q8_PGE, C & 255u, kvh, d0, vn8, vSegMax);
            }
        } else {
            half kn8[8], vn8[8];
            float mnk = 1e30f, mxk = -1e30f, mnv4 = 1e30f, mxv4 = -1e30f;
            if (live) {
                for (uint i = 0; i < 8u; i++) {
                    kn8[i] = knew[kvh * QW_HEAD_DIM + d0 + i];
                    vn8[i] = vnew[kvh * QW_HEAD_DIM + d0 + i];
                    const float a = float(kn8[i]), b = float(vn8[i]);
                    mnk = fmin(mnk, a); mxk = fmax(mxk, a);
                    mnv4 = fmin(mnv4, b); mxv4 = fmax(mxv4, b);
                }
            }
            #pragma unroll
            for (uint sh = 1u; sh <= 8u; sh <<= 1) {
                mnk = fmin(mnk, simd_shuffle_xor(mnk, sh)); mxk = fmax(mxk, simd_shuffle_xor(mxk, sh));
                mnv4 = fmin(mnv4, simd_shuffle_xor(mnv4, sh)); mxv4 = fmax(mxv4, simd_shuffle_xor(mxv4, sh));
            }
            if (live) {
                qwg_q4_store_seg((device uchar*)kpool, uint64_t(ktab[C >> 8]) * QW_Q4_PGE, C & 255u, kvh, d0, kn8, mnk, mxk);
                qwg_q4_store_seg((device uchar*)vpool, uint64_t(vtab[C >> 8]) * QW_Q4_PGE, C & 255u, kvh, d0, vn8, mnv4, mxv4);
            }
        }
    }
}

kernel void qwg_attn_decode_general(
        const device half* qgate [[buffer(0)]],
        device half* kpool [[buffer(1)]],
        device half* vpool [[buffer(2)]],
        const device int* ktab [[buffer(3)]],
        const device int* vtab [[buffer(4)]],
        const device half* knew [[buffer(5)]],
        const device half* vnew [[buffer(6)]],
        device float* partials [[buffer(7)]],
        constant uint& C [[buffer(8)]],
        constant uint& splits [[buffer(9)]],
        device half* outp [[buffer(10)]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 tg [[threadgroup_position_in_grid]]) {
    threadgroup half stageK[QW_MAXKEYS * QW_HEAD_DIM_MAX], stageV[QW_MAXKEYS * QW_HEAD_DIM_MAX];
    qwg_decode_general_body<0u>(qgate, kpool, vpool, ktab, vtab, knew, vnew, partials, outp,
                                 C, splits, tid, tg, &stageK[0], &stageV[0]);
}

kernel void qwg_attn_decode_general_q8(
        const device half* qgate [[buffer(0)]],
        device half* kpool [[buffer(1)]],
        device half* vpool [[buffer(2)]],
        const device int* ktab [[buffer(3)]],
        const device int* vtab [[buffer(4)]],
        const device half* knew [[buffer(5)]],
        const device half* vnew [[buffer(6)]],
        device float* partials [[buffer(7)]],
        constant uint& C [[buffer(8)]],
        constant uint& splits [[buffer(9)]],
        device half* outp [[buffer(10)]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 tg [[threadgroup_position_in_grid]]) {
    threadgroup half stageK[QW_MAXKEYS * QW_HEAD_DIM_MAX], stageV[QW_MAXKEYS * QW_HEAD_DIM_MAX];
    qwg_decode_general_body<1u>(qgate, kpool, vpool, ktab, vtab, knew, vnew, partials, outp,
                                 C, splits, tid, tg, &stageK[0], &stageV[0]);
}

kernel void qwg_attn_decode_general_q4(
        const device half* qgate [[buffer(0)]],
        device half* kpool [[buffer(1)]],
        device half* vpool [[buffer(2)]],
        const device int* ktab [[buffer(3)]],
        const device int* vtab [[buffer(4)]],
        const device half* knew [[buffer(5)]],
        const device half* vnew [[buffer(6)]],
        device float* partials [[buffer(7)]],
        constant uint& C [[buffer(8)]],
        constant uint& splits [[buffer(9)]],
        device half* outp [[buffer(10)]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 tg [[threadgroup_position_in_grid]]) {
    threadgroup half stageK[QW_MAXKEYS * QW_HEAD_DIM_MAX], stageV[QW_MAXKEYS * QW_HEAD_DIM_MAX];
    qwg_decode_general_body<2u>(qgate, kpool, vpool, ktab, vtab, knew, vnew, partials, outp,
                                 C, splits, tid, tg, &stageK[0], &stageV[0]);
}

kernel void qwg_attn_decode_combine_general(
        const device half* qgate [[buffer(0)]],
        device half* outp [[buffer(1)]],
        const device float* partials [[buffer(2)]],
        constant uint& splits [[buffer(3)]],
        uint lane [[thread_index_in_threadgroup]],
        uint3 tg [[threadgroup_position_in_grid]]) {
    const uint head = tg.x;
    const uint d0 = lane * 8u;
    if (d0 >= QW_HEAD_DIM) return;          // the head window
    const device float* base = partials + head * (QW_MAXSPLITS * (QW_HEAD_DIM + 2u));

    float M = -1e30f;
    for (uint i = 0; i < splits; i++) {
        const float mi = base[i * (QW_HEAD_DIM + 2u) + QW_HEAD_DIM];
        if (mi > M) M = mi;
    }
    float L = 0.0f;
    for (uint i = 0; i < splits; i++) {
        const float mi = base[i * (QW_HEAD_DIM + 2u) + QW_HEAD_DIM];
        if (mi <= -1e29f) continue;
        L += base[i * (QW_HEAD_DIM + 2u) + QW_HEAD_DIM + 1u] * metal::exp(mi - M);
    }
    float o[8]; for (uint i = 0; i < 8u; i++) o[i] = 0.0f;
    for (uint i = 0; i < splits; i++) {
        const float mi = base[i * (QW_HEAD_DIM + 2u) + QW_HEAD_DIM];
        if (mi <= -1e29f) continue;
        const float w = metal::exp(mi - M);
        device const float* ob = base + i * (QW_HEAD_DIM + 2u) + d0;
        for (uint j = 0; j < 8u; j++) o[j] += w * ob[j];
    }
    float gf[8];
    for (uint i = 0; i < 8u; i++) gf[i] = 1.0f;
    if (QW_HAS_Q_GATE) {
        device const half* g4 = qgate + head * QW_Q_STRIDE * QW_HEAD_DIM + QW_HEAD_DIM + d0;
        half gg[8];
        for (uint i = 0; i < 8u; i++) gg[i] = g4[i];
        for (uint i = 0; i < 8u; i++) gf[i] = 1.0f / (1.0f + metal::exp(-float(gg[i])));
    }
    const float linv = (L > 0.0f) ? 1.0f / L : 0.0f;
    half hv[8];
    for (uint i = 0; i < 8u; i++) hv[i] = half(o[i] * linv * gf[i]);
    fwht32_inplace(hv, lane);
    device half* ov = outp + head * QW_HEAD_DIM + d0;
    *(device uint4*)ov = *(thread uint4*)hv;
}
