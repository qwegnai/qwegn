// qwattn_decode_batched.metal — EXP-9 opt-in batched prefix-aware decode
// attention: qwg_attn_decode_batched{,_q8,_q4} + qwg_attn_decode_combine_batched.
// Split from qwattn_decode.metal; shared helpers: include/qw_attn_decode.metalh.
#include <metal_stdlib>
#include "include/qw_geometry_constants.metalh"
#include "include/qw_dequant.metalh"
using namespace metal;

#define QW_MAXSPLITS 64u
#define QW_ST   12u                        // keys staged per phase (12 KiB)
#define QW_TGV  192u                       // 6 simdgroups = the 6 heads of one kv-head
#define QW_HEAD_DIM_MAX 256u
#include "include/qw_kv_paged.metalh"
#include "include/qw_attn_decode.metalh"

// EXP-9 sub-item 3 (OPT-IN, NOT the shipping default) — batched
// prefix-aware decode attention for pos>0 continuations. "Local
// MLX-derived" scope note (binding): this mechanism is NOT imported from
template <uint MODE>
static inline void qwg_decode_batched_body(const device half* qgate [[buffer(0)]],  // [T][12288]
                                     device half* kpool [[buffer(1)]],
                                     device half* vpool [[buffer(2)]],
                                     const device int*  ktab  [[buffer(3)]],
                                     const device int*  vtab  [[buffer(4)]],
                                     const device half* knew  [[buffer(5)]],  // [T][1024]
                                     const device half* vnew  [[buffer(6)]],  // [T][1024]
                                     device float* partials   [[buffer(7)]],  // [T][24][MAXSPLITS][258]
                                     constant uint& cBase  [[buffer(8)]],     // pos (past keys before this chunk)
                                     constant uint& splits [[buffer(9)]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint3 tg [[threadgroup_position_in_grid]],
                                     threadgroup half* stageK, threadgroup half* stageV) { // (kv-head, split, qi)
    const uint qi  = tg.z;
    const uint C   = cBase + qi;             // THIS query's own past-keys count
    const uint sg   = tid >> 5;
    const uint lane = tid & 31u;
    const uint head = tg.x * QW_GQA_RATIO + sg;
    const uint kvh  = head / QW_GQA_RATIO;
    const uint d0   = lane * 8u;

    const uint nKeys = C + 1u;
    const uint per = (nKeys + splits - 1u) / splits;
    const uint k0 = min(tg.y * per, nKeys);
    const uint k1 = min(k0 + per, nKeys);
    const uint cEnd = min(k1, C);

    const device half* qgateQ = qgate + (uint64_t)qi * QW_QPROJ_OUT;
    const device half* knewQ  = knew + (uint64_t)qi * (QW_KV_HEADS * QW_HEAD_DIM);
    const device half* vnewQ  = vnew + (uint64_t)qi * (QW_KV_HEADS * QW_HEAD_DIM);

    float q8[8];
    {
        device const half* q4 = qgateQ + head * (2u * QW_HEAD_DIM) + d0;
        half qq[8];
        for (uint i = 0; i < 8u; i++) qq[i] = q4[i];
        for (uint i = 0; i < 8u; i++) q8[i] = float(qq[i]);
    }

    float m = -1e30f, lsum = 0.0f;
    float oacc[8]; for (uint i = 0; i < 8u; i++) oacc[i] = 0.0f;

    #define QWB_MERGE(s, vv) {                                           \
        const float _mn = max(m, s);                                     \
        const float _c  = (s > m) ? metal::exp(m - s) : 1.0f;             \
        const float _p  = metal::exp(s - _mn);                           \
        lsum = lsum * _c + _p;                                           \
        for (uint _i = 0; _i < 8u; _i++) oacc[_i] = oacc[_i] * _c + _p * float(vv[_i]); \
        m = _mn;                                                         \
    }
    #define QWB_DOT(kk) ({                                               \
        float _pd = 0.0f;                                                \
        for (uint _i = 0; _i < 8u; _i++) _pd += q8[_i] * float(kk[_i]);  \
        qwg_dec_logit(_pd); })

    uint r = k0;
    for (; r + QW_ST <= cEnd; r += QW_ST) {
        #pragma unroll
        for (uint s = 0; s < 2u; ++s) {
            const uint e = s * (192u * 8u) + tid * 8u;
            const uint row = r + (e >> 8);
            const uint pk = (uint)ktab[row >> 8], pv = (uint)vtab[row >> 8];
            const uint off = (row & 255u) * QW_ROW + kvh * QW_HEAD_DIM + (e & 255u);
            device const half* sk = kpool + (uint64_t)pk * QW_PGEL + off;
            device const half* sv = vpool + (uint64_t)pv * QW_PGEL + off;
            threadgroup half* dk = &stageK[e]; threadgroup half* dv = &stageV[e];
            half ka[8], va[8];
            if ((MODE == 0u)) {
                for (uint i = 0; i < 8u; i++) { ka[i] = sk[i]; va[i] = sv[i]; }
            } else {
                QW_KV_LOAD8(ka, kpool, pk, row & 255u, kvh, (e & 255u), MODE)
                QW_KV_LOAD8(va, vpool, pv, row & 255u, kvh, (e & 255u), MODE)
            }
            for (uint i = 0; i < 8u; i++) { dk[i] = ka[i]; dv[i] = va[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s12[QW_ST];
        #pragma unroll
        for (uint j = 0; j < QW_ST; j++) {
            threadgroup const half* ks = &stageK[j * QW_HEAD_DIM + d0];
            half kk[8];
            for (uint i = 0; i < 8u; i++) kk[i] = ks[i];
            s12[j] = QWB_DOT(kk);
        }
        float bm = s12[0];
        #pragma unroll
        for (uint j = 1; j < QW_ST; j++) bm = max(bm, s12[j]);
        const float Mb = max(m, bm);
        float lp = 0.0f; float op[8];
        #pragma unroll
        for (uint i = 0; i < 8u; i++) op[i] = 0.0f;
        #pragma unroll
        for (uint j = 0; j < QW_ST; j++) {
            const float pj = metal::exp(s12[j] - Mb);
            lp += pj;
            threadgroup const half* vs = &stageV[j * QW_HEAD_DIM + d0];
            half vv[8];
            for (uint i = 0; i < 8u; i++) vv[i] = vs[i];
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
    for (; r + 4u <= cEnd; r += 4u) {
        device const half* k0p = kpool + (uint64_t)ktab[(r + 0u) >> 8] * QW_PGEL + (uint64_t)((r + 0u) & 255u) * QW_ROW + kvh * QW_HEAD_DIM + d0;
        device const half* k1p = kpool + (uint64_t)ktab[(r + 1u) >> 8] * QW_PGEL + (uint64_t)((r + 1u) & 255u) * QW_ROW + kvh * QW_HEAD_DIM + d0;
        device const half* k2p = kpool + (uint64_t)ktab[(r + 2u) >> 8] * QW_PGEL + (uint64_t)((r + 2u) & 255u) * QW_ROW + kvh * QW_HEAD_DIM + d0;
        device const half* k3p = kpool + (uint64_t)ktab[(r + 3u) >> 8] * QW_PGEL + (uint64_t)((r + 3u) & 255u) * QW_ROW + kvh * QW_HEAD_DIM + d0;
        device const half* v0p = vpool + (uint64_t)vtab[(r + 0u) >> 8] * QW_PGEL + (uint64_t)((r + 0u) & 255u) * QW_ROW + kvh * QW_HEAD_DIM + d0;
        device const half* v1p = vpool + (uint64_t)vtab[(r + 1u) >> 8] * QW_PGEL + (uint64_t)((r + 1u) & 255u) * QW_ROW + kvh * QW_HEAD_DIM + d0;
        device const half* v2p = vpool + (uint64_t)vtab[(r + 2u) >> 8] * QW_PGEL + (uint64_t)((r + 2u) & 255u) * QW_ROW + kvh * QW_HEAD_DIM + d0;
        device const half* v3p = vpool + (uint64_t)vtab[(r + 3u) >> 8] * QW_PGEL + (uint64_t)((r + 3u) & 255u) * QW_ROW + kvh * QW_HEAD_DIM + d0;
        half kA[8], kB[8], kCc[8], kD[8], vA[8], vB[8], vCc[8], vD[8];
        if ((MODE == 0u)) {
        for (uint i = 0; i < 8u; i++) { kA[i] = k0p[i]; kB[i] = k1p[i]; kCc[i] = k2p[i]; kD[i] = k3p[i];
                                        vA[i] = v0p[i]; vB[i] = v1p[i]; vCc[i] = v2p[i]; vD[i] = v3p[i]; }
        } else {
            QW_KV_LOAD8(kA, kpool, ktab[(r + 0u) >> 8], (r + 0u) & 255u, kvh, d0, MODE);
            QW_KV_LOAD8(kB, kpool, ktab[(r + 1u) >> 8], (r + 1u) & 255u, kvh, d0, MODE);
            QW_KV_LOAD8(kCc, kpool, ktab[(r + 2u) >> 8], (r + 2u) & 255u, kvh, d0, MODE);
            QW_KV_LOAD8(kD, kpool, ktab[(r + 3u) >> 8], (r + 3u) & 255u, kvh, d0, MODE);
            QW_KV_LOAD8(vA, vpool, vtab[(r + 0u) >> 8], (r + 0u) & 255u, kvh, d0, MODE);
            QW_KV_LOAD8(vB, vpool, vtab[(r + 1u) >> 8], (r + 1u) & 255u, kvh, d0, MODE);
            QW_KV_LOAD8(vCc, vpool, vtab[(r + 2u) >> 8], (r + 2u) & 255u, kvh, d0, MODE);
            QW_KV_LOAD8(vD, vpool, vtab[(r + 3u) >> 8], (r + 3u) & 255u, kvh, d0, MODE);
        }
        const float sA = QWB_DOT(kA), sB = QWB_DOT(kB), sC = QWB_DOT(kCc), sD = QWB_DOT(kD);
        QWB_MERGE(sA, vA)  QWB_MERGE(sB, vB)  QWB_MERGE(sC, vCc)  QWB_MERGE(sD, vD)
    }
    for (; r < cEnd; r++) {
        device const half* kb = kpool + (uint64_t)ktab[r >> 8] * QW_PGEL + (uint64_t)(r & 255u) * QW_ROW + kvh * QW_HEAD_DIM + d0;
        device const half* vb = vpool + (uint64_t)vtab[r >> 8] * QW_PGEL + (uint64_t)(r & 255u) * QW_ROW + kvh * QW_HEAD_DIM + d0;
        half kk[8], vv[8];
        if ((MODE == 0u)) {
            for (uint i = 0; i < 8u; i++) { kk[i] = kb[i]; vv[i] = vb[i]; }
        } else {
            QW_KV_LOAD8(kk, kpool, ktab[r >> 8], r & 255u, kvh, d0, MODE);
            QW_KV_LOAD8(vv, vpool, vtab[r >> 8], r & 255u, kvh, d0, MODE);
        }
        QWB_MERGE(QWB_DOT(kk), vv)
    }
    if (C >= k0 && C < k1) {
        device const half* kb = knewQ + kvh * QW_HEAD_DIM + d0;
        device const half* vb = vnewQ + kvh * QW_HEAD_DIM + d0;
        half kk[8], vv[8];
        if ((MODE == 0u)) {
            for (uint i = 0; i < 8u; i++) { kk[i] = kb[i]; vv[i] = vb[i]; }
        } else if ((MODE == 1u)) {
            // In-register quantise of the new row (same maths as the f16 twin
            // in qwg_attn_decode and the batched epilogue's store_seg).
            float km = 0.0f, vm = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 8u; i++) { km = fmax(km, fabs(float(kb[i])));
                                           vm = fmax(vm, fabs(float(vb[i]))); }
            km = simd_max(km); vm = simd_max(vm);
            const float ks = float(half(fmax(km / 127.0f, 1e-8f)));
            const float vs = float(half(fmax(vm / 127.0f, 1e-8f)));
            const float kinv = 1.0f / ks, vinv = 1.0f / vs;
            #pragma unroll
            for (uint i = 0; i < 8u; i++) {
                kk[i] = half(clamp(round(float(kb[i]) * kinv), -127.0f, 127.0f) * ks);
                vv[i] = half(clamp(round(float(vb[i]) * vinv), -127.0f, 127.0f) * vs);
            }
        } else {
            uint knib[8], vnib[8]; float ksc, kmn, vsc, vmn;
            qwg_q4_quant8(kb, &kk[0], knib, &ksc, &kmn);
            qwg_q4_quant8(vb, &vv[0], vnib, &vsc, &vmn);
        }
        QWB_MERGE(QWB_DOT(kk), vv)
    }

    const uint splitStride = max(QW_MAXSPLITS, splits);
    device float* pb = partials
        + (uint64_t)qi * (QW_Q_HEADS * splitStride * (QW_HEAD_DIM + 2u))
        + (head * splitStride + tg.y) * (QW_HEAD_DIM + 2u);
    for (uint i = 0; i < 8u; i++) pb[d0 + i] = oacc[i];
    if (lane == 0u) { pb[QW_HEAD_DIM] = m; pb[QW_HEAD_DIM + 1u] = lsum; }

    if (tg.y == 0u && sg == 0u) {
        if ((MODE == 0u)) {
            device half* kw = kpool + (uint64_t)ktab[C >> 8] * QW_PGEL
                              + (uint64_t)(C & 255u) * QW_ROW + kvh * QW_HEAD_DIM + lane * 8u;
            device half* vw = vpool + (uint64_t)vtab[C >> 8] * QW_PGEL
                              + (uint64_t)(C & 255u) * QW_ROW + kvh * QW_HEAD_DIM + lane * 8u;
            device const half* kn = knewQ + kvh * QW_HEAD_DIM + lane * 8u;
            device const half* vn = vnewQ + kvh * QW_HEAD_DIM + lane * 8u;
            half ks[8], vs[8];
            for (uint i = 0; i < 8u; i++) { ks[i] = kn[i]; vs[i] = vn[i]; }
            for (uint i = 0; i < 8u; i++) { kw[i] = ks[i]; vw[i] = vs[i]; }
        } else if ((MODE == 1u)) {
            device const half* kn = knewQ + kvh * QW_HEAD_DIM + lane * 8u;
            device const half* vn = vnewQ + kvh * QW_HEAD_DIM + lane * 8u;
            half ks[8], vs[8];
            for (uint i = 0; i < 8u; i++) { ks[i] = kn[i]; vs[i] = vn[i]; }
            float kMax = 0.0f, vMax = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 8u; i++) {
                kMax = fmax(kMax, fabs(float(ks[i])));
                vMax = fmax(vMax, fabs(float(vs[i])));
            }
            kMax = simd_max(kMax); vMax = simd_max(vMax);
            qwg_q8_store_seg((device char*)kpool,
                             uint64_t(ktab[C >> 8]) * QW_Q8_PGE, C & 255u, kvh, d0, ks, kMax);
            qwg_q8_store_seg((device char*)vpool,
                             uint64_t(vtab[C >> 8]) * QW_Q8_PGE, C & 255u, kvh, d0, vs, vMax);
        } else {
            device const half* kn = knewQ + kvh * QW_HEAD_DIM + lane * 8u;
            device const half* vn = vnewQ + kvh * QW_HEAD_DIM + lane * 8u;
            half ks[8], vs[8], kd[8], vd[8];
            for (uint i = 0; i < 8u; i++) { ks[i] = kn[i]; vs[i] = vn[i]; }
            uint knib[8], vnib[8]; float ksc, kmn, vsc, vmn;
            qwg_q4_quant8(ks, kd, knib, &ksc, &kmn);
            qwg_q4_quant8(vs, vd, vnib, &vsc, &vmn);
            qwg_q4_store8((device char*)kpool, uint64_t(ktab[C >> 8]),
                          C & 255u, kvh, d0, knib, ksc, kmn);
            qwg_q4_store8((device char*)vpool, uint64_t(vtab[C >> 8]),
                          C & 255u, kvh, d0, vnib, vsc, vmn);
        }
    }
}

// Batched combine: grid (24 heads, T queries), 32 threads — same
// log-sum-exp merge + gate epilogue as `qwg_attn_decode_combine`, indexed
// by qi = tg.y for qgate/partials/output row selection.

#define QW_DECODEB_ENTRY(SFX, MODE) \
kernel void qwg_attn_decode_batched##SFX( \
        const device half* qgate [[buffer(0)]], device half* kpool [[buffer(1)]], \
        device half* vpool [[buffer(2)]], const device int* ktab [[buffer(3)]], \
        const device int* vtab [[buffer(4)]], const device half* knewQ [[buffer(5)]], \
        const device half* vnewQ [[buffer(6)]], device float* partials [[buffer(7)]], \
        constant uint& cBase [[buffer(8)]], constant uint& splits [[buffer(9)]], \
        uint tid [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) { \
    threadgroup half stageK[QW_ST * QW_HEAD_DIM_MAX], stageV[QW_ST * QW_HEAD_DIM_MAX]; \
    qwg_decode_batched_body<MODE>(qgate, kpool, vpool, ktab, vtab, knewQ, vnewQ, partials, \
                                  cBase, splits, tid, tg, stageK, stageV); \
}
QW_DECODEB_ENTRY(, 0)
QW_DECODEB_ENTRY(_q8, 1)
QW_DECODEB_ENTRY(_q4, 2)
#undef QW_DECODEB_ENTRY

kernel void qwg_attn_decode_combine_batched(const device half* qgate     [[buffer(0)]], // [T][12288]
                                             device half* outp            [[buffer(1)]], // [T][6144]
                                             const device float* partials [[buffer(2)]], // [T][24][MAXSPLITS][258]
                                             constant uint& splits   [[buffer(3)]],
                                             uint lane [[thread_index_in_threadgroup]],
                                             uint3 tg  [[threadgroup_position_in_grid]]) { // (head, qi)
    const uint head = tg.x;
    const uint qi = tg.y;
    const uint d0 = lane * 8u;
    const uint splitStride = max(QW_MAXSPLITS, splits);
    const device float* base = partials
        + (uint64_t)qi * (QW_Q_HEADS * splitStride * (QW_HEAD_DIM + 2u))
        + head * (splitStride * (QW_HEAD_DIM + 2u));

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
    const device half* g4 = qgate + (uint64_t)qi * QW_QPROJ_OUT + head * (2u * QW_HEAD_DIM) + QW_HEAD_DIM + d0;
    half gg[8];
    for (uint i = 0; i < 8u; i++) gg[i] = g4[i];
    const float linv = (L > 0.0f) ? 1.0f / L : 0.0f;
    half hv[8];
    for (uint i = 0; i < 8u; i++)
        hv[i] = half(o[i] * linv * (1.0f / (1.0f + metal::exp(-float(gg[i])))));
    fwht32_inplace(hv, lane);
    device half* ov = outp + (uint64_t)qi * (QW_Q_HEADS * QW_HEAD_DIM) + head * QW_HEAD_DIM + d0;
    for (uint i = 0; i < 8u; i++)
        ov[i] = hv[i];
}
