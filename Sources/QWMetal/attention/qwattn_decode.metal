// qwattn_decode.metal — C09: split-K flash-DECODE attention over paged KV.
//
// Work (spec 09): ONE query token (row T-1) attends a paged KV cache of
// Design note: docs/KERNEL_NOTES.md#qwattn_decode.
#include <metal_stdlib>
#include "include/qw_geometry_constants.metalh"
#include "include/qw_dequant.metalh"
using namespace metal;

#define QW_MAXSPLITS 64u
#define QW_ST   12u                        // keys staged per phase (12 KiB)
#define QW_TGV  (QW_GQA_RATIO * 32u)      // one simdgroup per q head of a kv head
#define QW_HEAD_DIM_MAX 256u
#include "include/qw_kv_paged.metalh"
#include "include/qw_attn_decode.metalh"


// All window K/V rows must be materialized and fenced before this read-only
// dispatch. The host falls back when query split strides differ.
template <uint B, uint KV>   // KV: 0 f16, 1 q8, 2 q4 — COMPILE-TIME pool
                              // addressing (the f16 instantiation must stay
                              // bit-identical to the pre-4c kernel: runtime
                              // branches around the load nests perturbed the
static inline void qwg_decode_shared(
        const device half* qgate, const device half* kpool, const device half* vpool,
        const device int* ktab, const device int* vtab,
        const device half* knew, const device half* vnew, device float* partials,
        uint C, uint splits, uint tid, uint3 tg,
        threadgroup half* stageK, threadgroup half* stageV) {
    const uint lane = tid & 31u, head = tg.x * QW_GQA_RATIO + (tid >> 5);
    const uint d0 = lane * 8u;
    const uint per = (C + 1u + splits - 1u) / splits;
    const uint start = tg.y * per;
    float q[B][8], m[B], l[B], acc[B][8];
    uint end[B], tiledEnd[B];
    uint sharedEnd = start;
    #pragma unroll
    for (uint b = 0; b < B; ++b) {
        end[b] = min(min(start + per, C + b + 1u), C + b);
        tiledEnd[b] = start + (end[b] > start ? (end[b] - start) / QW_ST * QW_ST : 0u);
        sharedEnd = max(sharedEnd, tiledEnd[b]);
        m[b] = -1e30f; l[b] = 0.0f;
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            q[b][i] = float(qgate[b * QW_QPROJ_OUT + head * QW_Q_STRIDE * QW_HEAD_DIM + d0 + i]);
            acc[b][i] = 0.0f;
        }
    }
    for (uint r = start; r < sharedEnd; r += QW_ST) {
        #pragma unroll
        for (uint s = 0; s < 2u; ++s) {
            const uint e = s * (QW_TGV * 8u) + tid * 8u;
            const uint row = r + (e >> 8);
            const uint off = (row & 255u) * QW_ROW + tg.x * QW_HEAD_DIM + (e & 255u);
            const device half* sk = kpool + uint64_t(ktab[row >> 8]) * QW_PGEL + off;
            const device half* sv = vpool + uint64_t(vtab[row >> 8]) * QW_PGEL + off;
            if (KV == 0u) {
                #pragma unroll
                for (uint i = 0; i < 8u; ++i) {
                    stageK[e + i] = sk[i]; stageV[e + i] = sv[i];
                }
            } else {
                QW_KV_LOAD8S(stageK, kpool, ktab[row >> 8], row & 255u, tg.x, (e & 255u), e, KV);
                QW_KV_LOAD8S(stageV, vpool, vtab[row >> 8], row & 255u, tg.x, (e & 255u), e, KV);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint b = 0; b < B; ++b) {
            if (r >= tiledEnd[b]) continue;
            float scores[QW_ST];
            for (uint j = 0; j < QW_ST; ++j) {
                float dot = 0.0f;
                for (uint i = 0; i < 8u; ++i)
                    dot += q[b][i] * float(stageK[j * QW_HEAD_DIM + d0 + i]);
                scores[j] = simd_sum(dot) * QW_ATTN_SCALE;
            }
            float bm = scores[0];
            for (uint j = 1; j < QW_ST; ++j) bm = max(bm, scores[j]);
            const float mb = max(m[b], bm);
            float lp = 0.0f, op[8];
            for (uint i = 0; i < 8u; ++i) op[i] = 0.0f;
            for (uint j = 0; j < QW_ST; ++j) {
                const float p = metal::exp(scores[j] - mb);
                lp += p;
                for (uint i = 0; i < 8u; ++i)
                    op[i] += p * float(stageV[j * QW_HEAD_DIM + d0 + i]);
            }
            const float correction = metal::exp(m[b] - mb);
            l[b] = l[b] * correction + lp;
            for (uint i = 0; i < 8u; ++i) acc[b][i] = acc[b][i] * correction + op[i];
            m[b] = mb;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint b = 0; b < B; ++b) {
        const uint last = min(start + per, C + b + 1u);
        for (uint r = tiledEnd[b]; r < last; ++r) {
            const bool own = r == C + b;
            const uint off = (r & 255u) * QW_ROW + tg.x * QW_HEAD_DIM + d0;
            const device half* kp = own ? knew + b * QW_ROW + tg.x * QW_HEAD_DIM + d0 :
                kpool + uint64_t(ktab[r >> 8]) * QW_PGEL + off;
            const device half* vp = own ? vnew + b * QW_ROW + tg.x * QW_HEAD_DIM + d0 :
                vpool + uint64_t(vtab[r >> 8]) * QW_PGEL + off;
            // Match scalar tail loads; direct device operands changed FP16 output bits.
            half kk[8], vv[8];
            if (KV == 0u) {
                for (uint i = 0; i < 8u; ++i) { kk[i] = kp[i]; vv[i] = vp[i]; }
            } else if (own) {
                // The new row quantises in-register NOW with the mode's own
                // contract, so the score uses EXACTLY the values later steps
                // dequantise from the pool (see qwg_q8/q4 own-row comments in
                // the single/batched kernels — identical maths, kp from new).
                if (KV == 1u) {
                    float km = 0.0f, vm = 0.0f;
                    #pragma unroll
                    for (uint i = 0; i < 8u; ++i) { km = fmax(km, fabs(float(kp[i])));
                                                   vm = fmax(vm, fabs(float(vp[i]))); }
                    km = simd_max(km); vm = simd_max(vm);
                    const float ks = float(half(fmax(km / 127.0f, 1e-8f)));
                    const float vs = float(half(fmax(vm / 127.0f, 1e-8f)));
                    const float kinv = 1.0f / ks, vinv = 1.0f / vs;
                    #pragma unroll
                    for (uint i = 0; i < 8u; ++i) {
                        kk[i] = half(clamp(round(float(kp[i]) * kinv), -127.0f, 127.0f) * ks);
                        vv[i] = half(clamp(round(float(vp[i]) * vinv), -127.0f, 127.0f) * vs);
                    }
                } else {
                    // q4 read site: compute ONLY (encodePagedKVWrite already
                    // materialised this row's bytes; identical maths means
                    // our kk equals what a later step loads from the pool).
                    uint knib[8], vnib[8]; float ksc, kmn, vsc, vmn;
                    qwg_q4_quant8(kp, &kk[0], knib, &ksc, &kmn);
                    qwg_q4_quant8(vp, &vv[0], vnib, &vsc, &vmn);
                }
            } else {
                QW_KV_LOAD8(kk, kpool, ktab[r >> 8], r & 255u, tg.x, d0, KV);
                QW_KV_LOAD8(vv, vpool, vtab[r >> 8], r & 255u, tg.x, d0, KV);
            }
            float dot = 0.0f;
            for (uint i = 0; i < 8u; ++i) dot += q[b][i] * float(kk[i]);
            const float score = simd_sum(dot) * QW_ATTN_SCALE;
            const float mn = max(m[b], score);
            const float correction = score > m[b] ? metal::exp(m[b] - score) : 1.0f;
            const float p = metal::exp(score - mn);
            l[b] = l[b] * correction + p;
            for (uint i = 0; i < 8u; ++i)
                acc[b][i] = acc[b][i] * correction + p * float(vv[i]);
            m[b] = mn;
        }
        const uint splitStride = max(QW_MAXSPLITS, splits);
        device float* dst = partials + (b * QW_Q_HEADS * splitStride + head * splitStride + tg.y) * (QW_HEAD_DIM + 2u);
        for (uint i = 0; i < 8u; ++i) dst[d0 + i] = acc[b][i];
        if (lane == 0u) { dst[QW_HEAD_DIM] = m[b]; dst[QW_HEAD_DIM + 1u] = l[b]; }
    }
}

#define QW_SHARED_ENTRY(B, KV, SFX) \
kernel void qwg_attn_decode_shared_m##B##SFX( \
        const device half* qgate [[buffer(0)]], const device half* kpool [[buffer(1)]], \
        const device half* vpool [[buffer(2)]], const device int* ktab [[buffer(3)]], \
        const device int* vtab [[buffer(4)]], const device half* knew [[buffer(5)]], \
        const device half* vnew [[buffer(6)]], device float* partials [[buffer(7)]], \
        constant uint& C [[buffer(8)]], constant uint& splits [[buffer(9)]], \
        uint tid [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) { \
    threadgroup half stageK[QW_ST * QW_HEAD_DIM_MAX], stageV[QW_ST * QW_HEAD_DIM_MAX]; \
    qwg_decode_shared<B, KV>(qgate, kpool, vpool, ktab, vtab, knew, vnew, partials, \
                        C, splits, tid, tg, stageK, stageV); \
}
QW_SHARED_ENTRY(2, 0,)
QW_SHARED_ENTRY(3, 0,)
QW_SHARED_ENTRY(4, 0,)
QW_SHARED_ENTRY(5, 0,)
QW_SHARED_ENTRY(6, 0,)
QW_SHARED_ENTRY(7, 0,)
QW_SHARED_ENTRY(8, 0,)
QW_SHARED_ENTRY(2, 1, _q8)
QW_SHARED_ENTRY(3, 1, _q8)
QW_SHARED_ENTRY(4, 1, _q8)
QW_SHARED_ENTRY(5, 1, _q8)
QW_SHARED_ENTRY(6, 1, _q8)
QW_SHARED_ENTRY(7, 1, _q8)
QW_SHARED_ENTRY(8, 1, _q8)
QW_SHARED_ENTRY(2, 2, _q4)
QW_SHARED_ENTRY(3, 2, _q4)
QW_SHARED_ENTRY(4, 2, _q4)
QW_SHARED_ENTRY(5, 2, _q4)
QW_SHARED_ENTRY(6, 2, _q4)
QW_SHARED_ENTRY(7, 2, _q4)
QW_SHARED_ENTRY(8, 2, _q4)
#undef QW_SHARED_ENTRY

// Pool-mode becomes a COMPILE-TIME constant: the f16 instantiation must
// compile bit-identical to the pre-4c kernel (a runtime branch around the
// load nests reorders the compiler's butterfly and flips last-bit f16 sums —
template <uint MODE>
static inline void qwg_decode_body(const device half* qgate [[buffer(0)]],   // [1][12288] q̂‖gate
                            device half* kpool [[buffer(1)]],         // page pool (K; epilogue writes)
                            device half* vpool [[buffer(2)]],         // page pool (V; epilogue writes)
                            const device int*  ktab  [[buffer(3)]],   // int32 page ids (pool-local)
                            const device int*  vtab  [[buffer(4)]],
                            const device half* knew  [[buffer(5)]],   // [1024] new k̂ (normed+roped)
                            const device half* vnew  [[buffer(6)]],   // [1024]
                            device float* partials   [[buffer(7)]],   // [24][SPLITS][258]
                            constant uint& C      [[buffer(8)]],      // past keys (cache rows; new row = C)
                            constant uint& splits [[buffer(9)]],      // active splits (≤64)
                            uint tid [[thread_index_in_threadgroup]],   // 0..191
                            uint3 tg [[threadgroup_position_in_grid]],  // (kv-head 0..3, split, 0)
                            threadgroup half* stageK, threadgroup half* stageV) {
    const uint sg   = tid >> 5;            // 0..5 = q-head within the group
    const uint lane = tid & 31u;           // lane in simdgroup
    const uint head = tg.x * QW_GQA_RATIO + sg;      // h = g·6+sg (0..23)
    const uint kvh  = head / QW_GQA_RATIO;           // == tg.x (fusion unit)
    const uint d0   = lane * 8u;           // dims [8l..8l+7] of MY head

    // key range: contiguous split of rows 0..C (C = the new row = nKeys−1)
    const uint nKeys = C + 1u;
    const uint per = (nKeys + splits - 1u) / splits;
    const uint k0 = min(tg.y * per, nKeys);
    const uint k1 = min(k0 + per, nKeys);
    const uint cEnd = min(k1, C);                   // pool keys are rows < C

    // q̂ for my head: qgate head row = [q̂(256) ‖ gate(256)] → head·512 + d0
    float q8[8];
    {
        device const half* q4 = qgate + head * QW_Q_STRIDE * QW_HEAD_DIM + d0;
        half qq[8];
        for (uint i = 0; i < 8u; i++) qq[i] = q4[i];
        for (uint i = 0; i < 8u; i++) q8[i] = float(qq[i]);
    }

    float m = -1e30f, lsum = 0.0f;
    float oacc[8]; for (uint i = 0; i < 8u; i++) oacc[i] = 0.0f;

    #define QW_MERGE(s, vv) {                                            \
        const float _mn = max(m, s);                                     \
        /* corr = exp(m−mn) is 1.0 except when the max GROWS — and the     \
           grow-test is lane-UNIFORM (s uniform per head after the fold):  \
           warp-uniform branch, kills ~95% of the merge exps (pattern      \
           data: monotone max grows O(log n) times). exp(0)=1 exactly ⇒   \
           select is bit-identical to always-exp. */                       \
        const float _c  = (s > m) ? metal::exp(m - s) : 1.0f;             \
        const float _p  = metal::exp(s - _mn);                           \
        lsum = lsum * _c + _p;                                           \
        for (uint _i = 0; _i < 8u; _i++) oacc[_i] = oacc[_i] * _c + _p * float(vv[_i]); \
        m = _mn;                                                         \
    }
    #define QW_DOT(kk) ({                                                \
        float _pd = 0.0f;                                                \
        for (uint _i = 0; _i < 8u; _i++) _pd += q8[_i] * float(kk[_i]);  \
        qwg_dec_logit(_pd); })

    // ── v4 STAGED PHASE (the spec's threadgroup design; race-won by bench
    // measurement: staged probe 311 GB/s vs gather-only 267 / direct-load
    // math 274 @ splits=32 — six heads broadcast-LOADING the same device
    uint r = k0;
    for (; r + QW_ST <= cEnd; r += QW_ST) {
        #pragma unroll
        for (uint s = 0; s < 2u; ++s) {
            const uint e = s * (QW_GQA_RATIO * 32u * 8u) + tid * 8u;
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
            threadgroup const half* ks = &stageK[j * QW_HEAD_DIM_MAX + d0];
            half kk[8];
            for (uint i = 0; i < 8u; i++) kk[i] = ks[i];
            s12[j] = QW_DOT(kk);
        }
        float bm = s12[0];
        #pragma unroll
        for (uint j = 1; j < QW_ST; j++) bm = max(bm, s12[j]);
        const float Mb = max(m, bm);                            // overflow law
        float lp = 0.0f; float op[8];
        #pragma unroll
        for (uint i = 0; i < 8u; i++) op[i] = 0.0f;
        #pragma unroll
        for (uint j = 0; j < QW_ST; j++) {
            const float pj = metal::exp(s12[j] - Mb);
            lp += pj;
            threadgroup const half* vs = &stageV[j * QW_HEAD_DIM_MAX + d0];
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
        threadgroup_barrier(mem_flags::mem_threadgroup);         // re-stage
    }
    for (; r + 4u <= cEnd; r += 4u) {        // MLP ×4 tail (≤7 keys)
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
        const float sA = QW_DOT(kA), sB = QW_DOT(kB), sC = QW_DOT(kCc), sD = QW_DOT(kD);
        QW_MERGE(sA, vA)  QW_MERGE(sB, vB)  QW_MERGE(sC, vCc)  QW_MERGE(sD, vD)
    }
    for (; r < cEnd; r++) {                  // <4 past keys (≤3)
        device const half* kb = kpool + (uint64_t)ktab[r >> 8] * QW_PGEL + (uint64_t)(r & 255u) * QW_ROW + kvh * QW_HEAD_DIM + d0;
        device const half* vb = vpool + (uint64_t)vtab[r >> 8] * QW_PGEL + (uint64_t)(r & 255u) * QW_ROW + kvh * QW_HEAD_DIM + d0;
        half kk[8], vv[8];
        if ((MODE == 0u)) {
            for (uint i = 0; i < 8u; i++) { kk[i] = kb[i]; vv[i] = vb[i]; }
        } else {
            QW_KV_LOAD8(kk, kpool, ktab[r >> 8], r & 255u, kvh, d0, MODE);
            QW_KV_LOAD8(vv, vpool, vtab[r >> 8], r & 255u, kvh, d0, MODE);
        }
        QW_MERGE(QW_DOT(kk), vv)
    }
    if (C >= k0 && C < k1) {                 // the new-token row, LAST key
        device const half* kb = knew + kvh * QW_HEAD_DIM + d0;
        device const half* vb = vnew + kvh * QW_HEAD_DIM + d0;
        half kk[8], vv[8];
        if ((MODE == 0u)) {
            for (uint i = 0; i < 8u; i++) { kk[i] = kb[i]; vv[i] = vb[i]; }
        } else if ((MODE == 1u)) {
            // The new row quantises in-register NOW (identical maths to the
            // epilogue's qwg_q8_store_seg — same f16 scale, same round, same
            // clamp): the score uses EXACTLY the values every later step will
            float km = 0.0f, vm = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 8u; i++) { km = fmax(km, fabs(float(kb[i])));
                                           vm = fmax(vm, fabs(float(vb[i]))); }
            km = simd_max(km); vm = simd_max(vm);
            const float ks = float(half(fmax(km / 127.0f, 1e-8f)));
            const float vs = float(half(fmax(vm / 127.0f, 1e-8f)));
            // Deterministically byte-identical to qwg_q8_store_seg's and the
            // Swift quantizeQ8/dequantizeQ8 reference (all three: scale=f16,
            // round(x * (1/s16)), clamp ±127, dequant q*scale) — only then is
            // the new-row score equal to what every later step reads back.
            const float kinv = 1.0f / ks, vinv = 1.0f / vs;
            #pragma unroll
            for (uint i = 0; i < 8u; i++) {
                kk[i] = half(clamp(round(float(kb[i]) * kinv), -127.0f, 127.0f) * ks);
                vv[i] = half(clamp(round(float(vb[i]) * vinv), -127.0f, 127.0f) * vs);
            }
        } else {
            // q4: compute-only (encodePagedKVWrite owns the stored bytes;
            // qwg_q4_quant8 reproduces the stored value bit-for-bit).
            uint knib[8], vnib[8]; float ksc, kmn, vsc, vmn;
            qwg_q4_quant8(kb, kk, knib, &ksc, &kmn);
            qwg_q4_quant8(vb, vv, vnib, &vsc, &vmn);
        }
        QW_MERGE(QW_DOT(kk), vv)
    }

    // partials row per (head, split): [O 256 f32][m][l]; m/l lane-uniform
    // per head (fold is within-lane-group); each simdgroup's lane 0 writes
    // its own head's m/l — six distinct rows per workgroup.
    const uint splitStride = max(QW_MAXSPLITS, splits);
    device float* pb = partials + (head * splitStride + tg.y) * (QW_HEAD_DIM + 2u);
    for (uint i = 0; i < 8u; i++) pb[d0 + i] = oacc[i];
    if (lane == 0u) { pb[QW_HEAD_DIM] = m; pb[QW_HEAD_DIM + 1u] = lsum; }

    // new-token K/V materialisation into the page slot: split-0 only; this
    // workgroup owns kv-head tg.x's 2048-B slice (four 512 B head-blocks).
    // 192 lanes × 11 halves… exact: 32-halves per lane per iter? Use the
    // scalar stripe: halves h = lane·1 + i·32… CLEAN: each lane copies the
    // 8-halve block at offset lane·8 within EACH of the kv-head's four 512 B
    // head blocks? NO — the slice is [4 kv-head blocks × 512 B] and this WG
    // owns ONE kv-head = 256 halves = 32 lanes × 8 halves… TWO iters of 32
    // cover the 256-half slice? 32·8 = 256 ✓ ONE iter of the 32-lane stripe
    // covers the whole 256-half kv-head slice — simdgroup 0 lanes do it.
    if (tg.y == 0u && sg == 0u) {            // one simdgroup writes the row
        if ((MODE == 0u)) {
        device half* kw = kpool + (uint64_t)ktab[C >> 8] * QW_PGEL
                          + (uint64_t)(C & 255u) * QW_ROW + kvh * QW_HEAD_DIM + lane * 8u;
        device half* vw = vpool + (uint64_t)vtab[C >> 8] * QW_PGEL
                          + (uint64_t)(C & 255u) * QW_ROW + kvh * QW_HEAD_DIM + lane * 8u;
        device const half* kn = knew + kvh * QW_HEAD_DIM + lane * 8u;
        device const half* vn = vnew + kvh * QW_HEAD_DIM + lane * 8u;
        half ks[8], vs[8];
        for (uint i = 0; i < 8u; i++) { ks[i] = kn[i]; vs[i] = vn[i]; }
        for (uint i = 0; i < 8u; i++) { kw[i] = ks[i]; vw[i] = vs[i]; }
        } else if ((MODE == 1u)) {
            // q8: the simdgroup owns exactly the 256-dim (row=C, kv-head)
            // segment — one simd_max per segment, then quantise-store. This
            // is the quantizeQ8 contract with C-round() on both sides
            device const half* kn = knew + kvh * QW_HEAD_DIM + lane * 8u;
            device const half* vn = vnew + kvh * QW_HEAD_DIM + lane * 8u;
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
            // q4: one 128-elem group = an aligned simd_group<16>; quantise
            // then pack nibbles + header (the ONLY writer of this row's
            // bytes — split-0 simdgroup, same as the f16 path).
            device const half* kn = knew + kvh * QW_HEAD_DIM + lane * 8u;
            device const half* vn = vnew + kvh * QW_HEAD_DIM + lane * 8u;
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

// combine: one workgroup per q-head (24 × 32 lanes). Lane owns dims
// [8·lane..8·lane+7]; merges ≤64 partial rows by log-sum-exp (lane-uniform
// serial over ≤64 — negligible vs the decode kernel), finalises
// O/L ⊙ σ(gate) → outp[6144]. Neutral rows (m ≤ −1e29: empty ranges from
// ragged plans) are skipped, not merged.

#define QW_DECODE_ENTRY(SFX, MODE) \
kernel void qwg_attn_decode##SFX( \
        const device half* qgate [[buffer(0)]], device half* kpool [[buffer(1)]], \
        device half* vpool [[buffer(2)]], const device int* ktab [[buffer(3)]], \
        const device int* vtab [[buffer(4)]], const device half* knew [[buffer(5)]], \
        const device half* vnew [[buffer(6)]], device float* partials [[buffer(7)]], \
        constant uint& C [[buffer(8)]], constant uint& splits [[buffer(9)]], \
        uint tid [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) { \
    threadgroup half stageK[QW_ST * QW_HEAD_DIM_MAX], stageV[QW_ST * QW_HEAD_DIM_MAX]; \
    qwg_decode_body<MODE>(qgate, kpool, vpool, ktab, vtab, knew, vnew, partials, \
                          C, splits, tid, tg, stageK, stageV); \
}
QW_DECODE_ENTRY(, 0)
QW_DECODE_ENTRY(_q8, 1)
QW_DECODE_ENTRY(_q4, 2)
#undef QW_DECODE_ENTRY

kernel void qwg_attn_decode_combine(const device half* qgate     [[buffer(0)]],
                                    device half* outp            [[buffer(1)]],   // [6144]
                                    const device float* partials [[buffer(2)]],   // [24][SPLITS][258]
                                    constant uint& splits   [[buffer(3)]],
                                    uint lane [[thread_index_in_threadgroup]],
                                    uint3 tg  [[threadgroup_position_in_grid]]) {
    const uint head = tg.x;
    const uint d0 = lane * 8u;
    const uint splitStride = max(QW_MAXSPLITS, splits);
    const device float* base = partials + head * (splitStride * (QW_HEAD_DIM + 2u));

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
    // gate half of the head's qgate row; epilogue o·σ(g) (HF :818).
    // A model without an attention output gate multiplies by one.
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
    for (uint i = 0; i < 8u; i++)
        hv[i] = half(o[i] * linv * gf[i]);
    fwht32_inplace(hv, lane);
    device half* ov = outp + head * QW_HEAD_DIM + d0;
    for (uint i = 0; i < 8u; i++)
        ov[i] = hv[i];
}
