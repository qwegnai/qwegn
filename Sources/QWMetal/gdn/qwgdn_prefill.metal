#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "qwgdn_prefill_tables.h"
#include "include/qw_dequant.metalh"
#include "include/qw_geometry_constants.metalh"
using namespace metal;

// qwgdn_prefill.metal — C13: chunked (UT-transform) GDN prefill.
//
// THE SPLIT (derived + fp64-validated vs the naive walk in
// gen_gdn_prefill_fixtures.py — δ there, Ũ/U/W here). Per chunk t0..t0+Lc−1
// (lane-local token index r), Γ_r = Σ_{j≤r} g_j, unit-diag lower-tri
// M[r,i] = β_r·(k̂_r·k̂_i)·e^{Γ_r−Γ_i}:
//   δ_r = U_r − Σ_i W[r,i]·S0[i,:]   with  M·U = β∘V ,
//                                             M·W = diag(β·e^Γ)·K̂
//   o_r = A_r + Σ_i B[r,i]·S0[i,:]   with  A = tril(QKg,inc)·U ,
//         B[r,:] = e^{Γ_r}·q̂_r − (tril(QKg,inc)·W)[r,:]
//         QKg[r,i] = e^{Γ_r−Γ_i}·q̂_r·k̂_i
//   carry S_L = e^{Γ_{L−1}}·(S0 + K̂ᵀ·(e^{−Γ}∘U) − (K̂ᵀ·W)·e^{−Γ}·S0)  —
//         K2 evaluates this as S = e^{ΓL}·S0 + kge·WS − kge·U with
//         kge = e^{ΓL−Γ_t}·k̂_t, WS[t,:] = W[t,:]·S0 (validated 8e-8 fp64).
// U,W,A,B are S0-INDEPENDENT ⇒ K1 (grid = chunks × 16 key-heads × 384
// lanes) runs FULLY PARALLEL; K2 (48 heads × 8 v-bands of 16) is the ONLY
// sequential loop (chunk carries, state slab resident in threadgroup
// memory); K3 is the per-(token,head) GatedRMSNorm epilogue (identical math
// to C12's qwg_gdn_norm_sigmoid, batched — o_scr row layout t*48+h agrees with K2's
// writes).
//
// MUFU LAW (measured THREE times on this kernel family — K2 v1: 4096
// exps/lane-chunk = 9.66 µs/tok; K1 v3: 8192 = 3.65 µs/tok; carry probe:
// removing the MUFU table −0.3 µs/4K). exp() INSIDE a dot-loop is the wall:
// quarter-rate SFU sharing the serialized issue port with LSU. DECAY
// FACTORIZATION kills it exactly: g ≤ 0 ⇒ Γ decreasing ⇒ anchor at
// Γref = Γ_{Lc−1} (the MIN) and s_r = e^{Γref−Γr} ≤ 1 — the SAFE direction
// (the flipped anchor overflows: e^{Σ|g|}). With s-scaled unknowns
// Z_r = s_r·U_r carried through the recurrence (e^{Γr−Γc}U_c = Z_c/s_r),
// every exp inside the dot-loops becomes two register multiplies; one
// divide only at stores:
//   U-sub: Z_r = β_r·(s_r·rhs_r − Σ_{c<r} raw_rc·Z_c); store U_r = Z_r/s_r
//   fold A: A_t = (Σ_{c≤t} qraw_tc·Z_c)/s_t   (diag qraw_tt = q̂·k̂ dot)
//   W-sub:  Ŵ_r = β_r·e^{Γref}·k̂_r[d] − β_r·Σ raw·Ŵ_c  (rhs exact: the
//           anchor cancels — s_r·e^{Γr} = e^{Γref}); store W_r = Ŵ_r/s_r
//   fold B: B_t = (e^{Γref}·q̂_t[d] − Σ_{c≤t} qraw_tc·Ŵ_c)/s_t
// exps/lane-chunk: 64 (s-table) + 1 vs 8192. Values agree with the
// single-exp form to ~1e-7 (products of correctly-rounded exps; the rungs
// bind 1e-4). MEASURED: K1 3.65 → 1.94 µs/tok @4K; total 1.26 → 1.83× vs
// the naive C12 path.
//
// DOT LAW (v2; v1 was dot-redundancy ACROSS lanes — every lane recomputed
// every Gram entry: 38.8 µs/tok): each raw dot runs ONCE per (r,c) pair,
// pairs striped over all 384 lanes, staged in ONE packed-triangle f32 array
// (nM+nQ ≤ 2080 × 4 B = 8.1 KiB → 2+ threadgroups/SM co-reside). The v-dim
// split (Z columns) is REGISTER-PRIVATE (lane-owned ⇒ zero barriers inside
// the sub loops); barriers only between phases that hand data across lanes.
// v3 shares each raw dot across the THREE v-heads of one key-head (dots ÷3
// vs v2); the decay β_h·e^{ΔΓ} folds at use time (M̃'s unit diagonal is
// never stored).
//
// f16 INTERMEDIATES (U/W/A/B + o_scr) were RACED and REJECTED: 4K speedup
// 1.83 → 1.87 (traffic is NOT the wall — the wall is per-lane serial LDS
// chains) while f16 quantization eats the 1e-4 rung tolerance (rung worst
// 1.85e-3 = f16 rounding, not a bug). Kept f32.
//
// Numerics: fp32 solve mandatory (spec risk: β-saturated heads →
// ill-conditioned M; fixtures force β∈(0.96,1) heads through T=1024).
// v-head h reads key-head i = h/3 from packed y [q̂16|k̂16|v48]×128 (C12 law).

#define NH128 128u
#define NVH QW_GDN_V_HEADS
#define NCOL QW_GDN_Z_CH      // NVH*128 per token row
#define QW_LOG2E 1.4426950408889634f
#define QW_EXP(x) exp2((x) * QW_LOG2E)   // guaranteed MUFU path
// f16 unpack from a u32 (uint-bit — Metal bans casting a private var's
// address; each u32 = two consecutive f16). fp32 result ⇒ f16×f32 products
// in the dots are EXACT (C08 P·V class).
#define HALF_AT(u, k) (((k) == 0) ? float(as_type<half>((ushort)((u) & 0xffffu))) : float(as_type<half>((ushort)((u) >> 16))))

static inline float qw_dot128(device const half* a, device const half* b) {
    // uint4 = 16 B = 8 f16: 16 load-pairs per 128-dim dot (scalar half loads
    // were 128; half4/half8 NAMES are ambiguous in this SDK). Products
    // convert to fp32 BEFORE multiply (f16×f16 in half precision rounds at
    device const uint4* a4 = (device const uint4*)a;
    device const uint4* b4 = (device const uint4*)b;
    float acc = 0.0f;
    for (uint p = 0u; p < 16u; p++) {
        uint4 av = a4[p], bv = b4[p];
        float2 p0 = float2(HALF_AT(av.x,0) * HALF_AT(bv.x,0), HALF_AT(av.x,1) * HALF_AT(bv.x,1));
        float2 p1 = float2(HALF_AT(av.y,0) * HALF_AT(bv.y,0), HALF_AT(av.y,1) * HALF_AT(bv.y,1));
        float2 p2 = float2(HALF_AT(av.z,0) * HALF_AT(bv.z,0), HALF_AT(av.z,1) * HALF_AT(bv.z,1));
        float2 p3 = float2(HALF_AT(av.w,0) * HALF_AT(bv.w,0), HALF_AT(av.w,1) * HALF_AT(bv.w,1));
        acc += (p0.x + p0.y) + (p1.x + p1.y) + (p2.x + p2.y) + (p3.x + p3.y);
    }
    return acc;
}
// strict-lower-tri pair decode p = r(r-1)/2 + c (r>c) / incl-diag q =
// r(r+1)/2 + c: FLOAT sqrt + integer bracket fixups (Metal bans double; the
// fp32 root of 8p+1 is 24-bit exact for p<2080). Returns (r<<8)|c — pure
// VALUE return: Metal's address-space rules ban ref/ptr params wrapping
// threadgroup temporaries, and lambdas cannot declare them.
[[maybe_unused]] static inline uint qwp_decode(uint p) {
    uint r = (uint)(0.5f * (sqrt(8.0f * (float)p + 1.0f) + 1.0f));
    while (r * (r - 1u) / 2u > p) r--;
    while ((r + 1u) * r / 2u <= p) r++;
    return (r << 8u) | (p - r * (r - 1u) / 2u);
}
[[maybe_unused]] static inline uint qwpq_decode(uint q) {
    uint r = (uint)(0.5f * (sqrt(8.0f * (float)q + 1.0f) - 1.0f));
    while ((r + 1u) * (r + 2u) / 2u <= q) r++;
    while (r * (r + 1u) / 2u > q) r--;
    return (r << 8u) | (q - r * (r + 1u) / 2u);
}

// K1 — the S0-independent stage (grid: chunks × 16 KEY-heads; 384 lanes =
// 3 v-heads × 128 v-dims). Raw dots shared per key-head; Z columns are
// register-private per lane ⇒ zero barriers inside the sub chains.
kernel void qwg_gdn_prefill_solve(
        device const half* y      [[buffer(0)]],   // [T][10240] q̂|k̂|v
        device const float* betas [[buffer(1)]],   // [T][48]
        device const float* glog  [[buffer(2)]],   // [T][48]
        device float* Ubuf        [[buffer(3)]],   // [T][48][128] f32
        device float* Wbuf        [[buffer(4)]],
        device float* Abuf        [[buffer(5)]],
        device float* Bbuf        [[buffer(6)]],
        constant uint& T          [[buffer(7)]],
        constant uint& L          [[buffer(8)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint t0 = tg.x * L;
    const uint Lc = min(L, T - t0);
    const uint h  = tg.y;                        // one v-head per threadgroup
    const uint i  = h / QW_GDN_V_PER_QK;         // key-head it shares (dot src)
    const uint d  = tid;                         // v-dim (128 lanes)
    const uint i0 = h * NH128 + d;
    const uint qCh = QW_GDN_QK_HEADS * NH128;
    const uint qkCh = 2u * qCh;

    threadgroup float raw[2080u];              // packed triangle (nM strict, then diag)
    // MTILDE (folded-decay, per-V-head): the fold lives in DEDICATED
    // storage — writing it back into `raw` is ILLEGAL: glog/beta are per
    // v-head (h = i·3 + hh), so lanes of the other two v-heads would read
    float gsum[64];
    float eg[64], eGam[64], bg[64], ucol[64], wcol[64];
    for (uint r = 0u; r < Lc; r++) {
        const float gl = glog[(t0 + r) * NVH + h];
        gsum[r] = (r == 0u) ? gl : (gsum[r - 1u] + gl);
        eg[r] = QW_EXP(gl);
        bg[r] = betas[(t0 + r) * NVH + h];
        eGam[r] = (r == 0u) ? eg[0] : (eGam[r - 1u] * eg[r]);
    }
    const uint nM = Lc * (Lc - 1u) / 2u, nQ = nM + Lc;
    const uint diagBase = nM;                  // diagonal lives at raw[nM + r]
    device const half* kbase = y + qCh + i * NH128;   // k̂ block row 0
    device const half* qbase = y + i * NH128;         // q̂ block row 0

    // Raw M̃ dots (k̂_r·k̂_c): per-v-head triangle (128 lanes). The fold
    // below is per-v-head legal (glog/beta are per v-head); sharing the
    // triangle across the 3 v-heads of a key-head is ILLEGAL (measured:
    #define STAGE_M() do { for (uint p = tid; p < nM; p += 128u) { \
            constant auto& entry = c_qwp_m[p]; \
            raw[p] = qw_dot128(kbase + (uint64_t)(t0 + entry.r) * QW_GDN_QKV_CH, \
                               kbase + (uint64_t)(t0 + entry.c) * QW_GDN_QKV_CH); } } while (0)
    // Raw QKg dots (q̂_r·k̂_c) incl diagonal: zero-cost table decode and direct dest store.
    #define STAGE_Q() do { for (uint p = tid; p < nQ; p += 128u) { \
            constant auto& entry = c_qwp_q[p]; \
            raw[(entry.r > entry.c) ? (entry.r * (entry.r - 1u) / 2u + entry.c) : (diagBase + entry.r)] = qw_dot128(qbase + (uint64_t)(t0 + entry.r) * QW_GDN_QKV_CH, \
                                        kbase + (uint64_t)(t0 + entry.c) * QW_GDN_QKV_CH); } } while (0)

    STAGE_M();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // MTILDE fold: decay folded into the matrix IN PARALLEL (one exp per
    // packed entry, same (r,c) table), then the chains are pure fma.
    for (uint p = tid; p < nM; p += 128u) {
        constant auto& entry = c_qwp_m[p];
        raw[p] = raw[p] * QW_EXP(gsum[entry.r] - gsum[entry.c]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Fused U-sub & W-sub over M̃ triangle — pure-fma chains
    for (uint r = 0u; r < Lc; r++) {
        float su = 0.0f;
        float sw = 0.0f;
        for (uint c = r; c-- > 0u;) {
            float m_rc = raw[r * (r - 1u) / 2u + c];
            su = fma(m_rc, ucol[c], su);
            sw = fma(m_rc, wcol[c], sw);
        }
        float rhs = float(y[(uint64_t)(t0 + r) * QW_GDN_QKV_CH + qkCh + i0]);
        float U_r = bg[r] * (rhs - su);
        ucol[r] = U_r;
        Ubuf[(uint64_t)(t0 + r) * NCOL + i0] = U_r;

        float wh = fma(-bg[r], sw, bg[r] * eGam[r] * float(kbase[(uint64_t)(t0 + r) * QW_GDN_QKV_CH + d]));
        wcol[r] = wh;
        Wbuf[(uint64_t)(t0 + r) * NCOL + i0] = wh;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);        // M-raw consumed

    STAGE_Q();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < nM; p += 128u) {      // diagonal decay = 1 (untouched)
        constant auto& entry = c_qwp_m[p];
        raw[p] = raw[p] * QW_EXP(gsum[entry.r] - gsum[entry.c]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Fused Fold A & Fold B over QKg triangle — pure-fma chains
    for (uint t = 0u; t < Lc; t++) {
        float acc_a = 0.0f;
        float acc_b = 0.0f;
        for (uint c = t; c-- > 0u;) {
            float qk_tc = raw[t * (t - 1u) / 2u + c];
            acc_a = fma(qk_tc, ucol[c], acc_a);
            acc_b = fma(-qk_tc, wcol[c], acc_b);
        }
        float diag = raw[diagBase + t];
        acc_a = fma(diag, ucol[t], acc_a);
        acc_b = fma(-diag, wcol[t], acc_b);
        acc_b += eGam[t] * float(qbase[(uint64_t)(t0 + t) * QW_GDN_QKV_CH + d]);

        Abuf[(uint64_t)(t0 + t) * NCOL + i0] = acc_a;
        Bbuf[(uint64_t)(t0 + t) * NCOL + i0] = acc_b;
    }
    #undef STAGE_M
    #undef STAGE_Q
}

// qwg_gdn_prefill_solve_nilpotent — Phase 1 Nilpotent Triangular Inversion.
// Solves (I + L) U = B·V and (I + L) W = B·e^Γ·K by evaluating M_inv = (I + L)^(-1)
// in threadgroup SRAM across subdiagonals in parallel, eliminating the scalar
kernel void qwg_gdn_prefill_solve_nilpotent(
        device const half* y      [[buffer(0)]],   // [T][10240] q̂|k̂|v
        device const float* betas [[buffer(1)]],   // [T][48]
        device const float* glog  [[buffer(2)]],   // [T][48]
        device float* Ubuf        [[buffer(3)]],   // [T][48][128] f32
        device float* Wbuf        [[buffer(4)]],
        device float* Abuf        [[buffer(5)]],
        device float* Bbuf        [[buffer(6)]],
        constant uint& T          [[buffer(7)]],
        constant uint& L          [[buffer(8)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint t0 = tg.x * L;
    const uint Lc = min(L, T - t0);
    const uint h  = tg.y;
    const uint i  = h / QW_GDN_V_PER_QK;
    const uint d  = tid;
    const uint i0 = h * NH128 + d;
    const uint qCh = QW_GDN_QK_HEADS * NH128;
    const uint qkCh = 2u * qCh;

    threadgroup float raw[2080u];              // packed triangle (nM strict, then diag)
    threadgroup float Minv[2080u];             // packed strictly lower triangle of (I + L)^(-1)

    float gsum[64];
    float eg[64], eGam[64], bg[64], ucol[64], wcol[64];
    for (uint r = 0u; r < Lc; r++) {
        const float gl = glog[(t0 + r) * NVH + h];
        gsum[r] = (r == 0u) ? gl : (gsum[r - 1u] + gl);
        eg[r] = QW_EXP(gl);
        bg[r] = betas[(t0 + r) * NVH + h];
        eGam[r] = (r == 0u) ? eg[0] : (eGam[r - 1u] * eg[r]);
    }
    const uint nM = Lc * (Lc - 1u) / 2u, nQ = nM + Lc;
    const uint diagBase = nM;
    device const half* kbase = y + qCh + i * NH128;
    device const half* qbase = y + i * NH128;

    #define STAGE_M() do { for (uint p = tid; p < nM; p += 128u) { \
            constant auto& entry = c_qwp_m[p]; \
            raw[p] = qw_dot128(kbase + (uint64_t)(t0 + entry.r) * QW_GDN_QKV_CH, \
                               kbase + (uint64_t)(t0 + entry.c) * QW_GDN_QKV_CH); } } while (0)
    #define STAGE_Q() do { for (uint p = tid; p < nQ; p += 128u) { \
            constant auto& entry = c_qwp_q[p]; \
            raw[(entry.r > entry.c) ? (entry.r * (entry.r - 1u) / 2u + entry.c) : (diagBase + entry.r)] = qw_dot128(qbase + (uint64_t)(t0 + entry.r) * QW_GDN_QKV_CH, \
                                        kbase + (uint64_t)(t0 + entry.c) * QW_GDN_QKV_CH); } } while (0)

    STAGE_M();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < nM; p += 128u) {
        constant auto& entry = c_qwp_m[p];
        raw[p] = raw[p] * QW_EXP(gsum[entry.r] - gsum[entry.c]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel Subdiagonal Inversion of strictly lower triangular matrix (I + L)
    // L[r, c] = bg[r] * raw[r*(r-1)/2 + c] for r > c.
    for (uint s = 1u; s < Lc; s++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < Lc - s) {
            const uint c = tid;
            const uint r = c + s;
            const uint prc = r * (r - 1u) / 2u + c;
            float val = -bg[r] * raw[prc];
            for (uint k = c + 1u; k < r; k++) {
                const uint prk = r * (r - 1u) / 2u + k;
                const uint pkc = k * (k - 1u) / 2u + c;
                val = fma(-Minv[prk], bg[k] * raw[pkc], val);
            }
            Minv[prc] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Dense parallel matrix-vector products (zero loop-carried dependency)
    float RU[64], RW[64];
    for (uint r = 0u; r < Lc; r++) {
        float rhs = float(y[(uint64_t)(t0 + r) * QW_GDN_QKV_CH + qkCh + i0]);
        RU[r] = bg[r] * rhs;
        RW[r] = bg[r] * eGam[r] * float(kbase[(uint64_t)(t0 + r) * QW_GDN_QKV_CH + d]);
    }
    for (uint r = 0u; r < Lc; r++) {
        float u = RU[r];
        float w = RW[r];
        for (uint c = 0u; c < r; c++) {
            float minv_rc = Minv[r * (r - 1u) / 2u + c];
            u = fma(minv_rc, RU[c], u);
            w = fma(minv_rc, RW[c], w);
        }
        ucol[r] = u;
        wcol[r] = w;
        Ubuf[(uint64_t)(t0 + r) * NCOL + i0] = u;
        Wbuf[(uint64_t)(t0 + r) * NCOL + i0] = w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STAGE_Q();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < nM; p += 128u) {
        constant auto& entry = c_qwp_m[p];
        raw[p] = raw[p] * QW_EXP(gsum[entry.r] - gsum[entry.c]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0u; t < Lc; t++) {
        float acc_a = 0.0f;
        float acc_b = 0.0f;
        for (uint c = t; c-- > 0u;) {
            float qk_tc = raw[t * (t - 1u) / 2u + c];
            acc_a = fma(qk_tc, ucol[c], acc_a);
            acc_b = fma(-qk_tc, wcol[c], acc_b);
        }
        float diag = raw[diagBase + t];
        acc_a = fma(diag, ucol[t], acc_a);
        acc_b = fma(-diag, wcol[t], acc_b);
        acc_b += eGam[t] * float(qbase[(uint64_t)(t0 + t) * QW_GDN_QKV_CH + d]);

        Abuf[(uint64_t)(t0 + t) * NCOL + i0] = acc_a;
        Bbuf[(uint64_t)(t0 + t) * NCOL + i0] = acc_b;
    }
    #undef STAGE_M
    #undef STAGE_Q
}

// K2 — the sequential chunk-carry chain (grid: 48 heads × 8 v-bands of 16;
// 128 lanes). One threadgroup per (head, band) keeps its state slab
// resident across the whole walk — the carry is the only serial dependency.
kernel void qwg_gdn_prefill_scan(
        device const half* y   [[buffer(0)]],
        device const float* Ubuf [[buffer(1)]], device const float* Wbuf [[buffer(2)]],
        device const float* Abuf [[buffer(3)]], device const float* Bbuf [[buffer(4)]],
        device const float* glog [[buffer(5)]],
        device float* state    [[buffer(6)]],    // [48][128][128] in place
        device float* o_scr    [[buffer(7)]],    // [T][48][128] f32 pre-norm
        constant uint& T       [[buffer(8)]],
        constant uint& L       [[buffer(9)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint h = tg.x, vb = tg.y, lane = tid;
    const uint i = h / QW_GDN_V_PER_QK;
    const uint qCh = QW_GDN_QK_HEADS * NH128;
    const uint nb = 16u, vx = vb * nb;
    threadgroup float S[128u * 16u];              // COLUMN-major: S[x*128+j]
    threadgroup float WS[64u * 16u];              // W·Sᵀ [t*16+x], then UD
    device float* stg = state + (uint64_t)h * 16384u;

    for (uint e = lane; e < 128u * nb; e += NH128) {   // load+transpose band
        const uint j = e / nb, x = e % nb;
        S[x * NH128 + j] = stg[j * NH128 + vx + x];
    }

    const uint nchunks = (T + L - 1u) / L;
    for (uint c = 0u; c < nchunks; c++) {
        const uint t0 = c * L;
        const uint Lc = min(L, T - t0);
        float gsum = 0.0f;
        for (uint r = 0u; r < Lc; r++) gsum += glog[(t0 + r) * NVH + h];
        const float egL = QW_EXP(gsum);
        float kgeTab[64];
        {   // kge[t] = e^{ΓL−Γt}: ONE backward recurrence — 64 MUFU total.
            float kg = 1.0f;                       // e^{ΓL−Γ(Lc−1)} = 1
            for (uint t = Lc; t-- > 0u;) {
                if (t != Lc - 1u) kg *= QW_EXP(glog[(t0 + t + 1u) * NVH + h]);
                kgeTab[t] = kg;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // S band resident

        // WS + o_scr dots: lane owns (t,x); j-loops float4 BOTH sides.
        // oacc STARTS at A[t] — adding A again at the store double-counts
        // it (off-by-A law: the ~1-magnitude fingerprint the bisect caught
        // at t=1 while K1 rungs were green). The three A/WS/o sites use the
        // GLOBAL v-index vx+x (band-local x aliases bands 1-3 — caught by
        // the K1-green/K2-red bisect; K1 is v-flat by layout).
        const uint n2 = Lc * nb;
        for (uint e = lane; e < n2; e += NH128) {
            const uint t = e / nb, x = e % nb;
            device const float4* Wr = (device const float4*)(Wbuf + (uint64_t)(t0 + t) * NCOL + h * NH128);
            device const float4* Br = (device const float4*)(Bbuf + (uint64_t)(t0 + t) * NCOL + h * NH128);
            threadgroup const float4* Sr = (threadgroup const float4*)&S[x * NH128];
            float ws = 0.0f;
            float oacc = Abuf[(uint64_t)(t0 + t) * NCOL + h * NH128 + vx + x];
            for (uint q = 0u; q < 32u; q++) {
                float4 sv = Sr[q];
                float4 wv = Wr[q];
                float4 bv = Br[q];
                ws = fma(wv.x, sv.x, ws); ws = fma(wv.y, sv.y, ws);
                ws = fma(wv.z, sv.z, ws); ws = fma(wv.w, sv.w, ws);
                oacc = fma(bv.x, sv.x, oacc); oacc = fma(bv.y, sv.y, oacc);
                oacc = fma(bv.z, sv.z, oacc); oacc = fma(bv.w, sv.w, oacc);
            }
            WS[t * nb + x] = ws;
            o_scr[(uint64_t)(t0 + t) * NCOL + h * NH128 + vx + x] = oacc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // WS staged

        // UD = U − WS in place over the WS region.
        for (uint e = lane; e < n2; e += NH128) {
            const uint t = e / nb, x = e % nb;
            WS[e] = Ubuf[(uint64_t)(t0 + t) * NCOL + h * NH128 + vx + x] - WS[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // UD staged

        // carry: S[x, j-quad] = egL·S + Σ_t (k̂_t[j]·kge[t])·UD[t, x].
        // 8 groups × 16 j = 128 per pass; 4 INDEPENDENT float4 chains per
        // lane (serial-chain-per-lane was the K2 v1 class); ZERO MUFU and
        // ZERO device scalar loads in this loop.
        const uint x0 = lane & 15u, g0 = (lane >> 4u) * 16u;
        threadgroup float4* Sp = (threadgroup float4*)&S[x0 * NH128 + g0];
        float4 acc0 = Sp[0] * egL, acc1 = Sp[1] * egL, acc2 = Sp[2] * egL, acc3 = Sp[3] * egL;
        device const half4* kdev4 = (device const half4*)
            (y + (uint64_t)t0 * QW_GDN_QKV_CH + qCh + i * NH128 + g0);
        for (uint t = 0u; t < Lc; t++) {
            const float scale = kgeTab[t] * WS[t * nb + x0];
            const size_t t_stride = (size_t)t * (QW_GDN_QKV_CH / 4u);
            acc0 = fma(float4(kdev4[t_stride + 0u]), scale, acc0);
            acc1 = fma(float4(kdev4[t_stride + 1u]), scale, acc1);
            acc2 = fma(float4(kdev4[t_stride + 2u]), scale, acc2);
            acc3 = fma(float4(kdev4[t_stride + 3u]), scale, acc3);
        }
        Sp[0] = acc0; Sp[1] = acc1; Sp[2] = acc2; Sp[3] = acc3;
        threadgroup_barrier(mem_flags::mem_threadgroup);   // next chunk reads S
    }
    for (uint e = lane; e < 128u * nb; e += NH128) {      // write back j-major
        const uint j = e / nb, x = e % nb;
        stg[j * NH128 + vx + x] = S[x * NH128 + j];
    }
}

// ── qwg_gdn_prefill_scan_hybrid — Hybrid Tensor Carry Scan ─────────────────
// Combines proven scalar orchestration for K1/K3 with simdgroup_matrix
// tensor-core micro-GEMM carry acceleration.
kernel void qwg_gdn_prefill_scan_hybrid(
        device const half* y   [[buffer(0)]],
        device const float* Ubuf [[buffer(1)]], device const float* Wbuf [[buffer(2)]],
        device const float* Abuf [[buffer(3)]], device const float* Bbuf [[buffer(4)]],
        device const float* glog [[buffer(5)]],
        device float* state    [[buffer(6)]],    // [48][128][128] in place
        device float* o_scr    [[buffer(7)]],    // [T][48][128] f32 pre-norm
        constant uint& T       [[buffer(8)]],
        constant uint& L       [[buffer(9)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint h = tg.x, vb = tg.y, lane = tid;
    const uint i = h / QW_GDN_V_PER_QK;
    const uint qCh = QW_GDN_QK_HEADS * NH128;
    const uint nb = 16u, vx = vb * nb;
    threadgroup float S[128u * 16u];              // COLUMN-major: S[x*128+j]
    threadgroup float WS[64u * 16u];              // kge·UD [x*64+t] (4 KiB)
    threadgroup float K_shared[8][128];          // [t8][j128] cooperative staging (4 KiB) -> Total TG = 16 KiB <= 16 KiB (2 TG/SM)
    device float* stg = state + (uint64_t)h * 16384u;

    for (uint e = lane; e < 128u * nb; e += NH128) {   // load+transpose band
        const uint j = e / nb, x = e % nb;
        S[x * NH128 + j] = stg[j * NH128 + vx + x];
    }

    const uint nchunks = (T + L - 1u) / L;
    for (uint c = 0u; c < nchunks; c++) {
        const uint t0 = c * L;
        const uint Lc = min(L, T - t0);
        float gsum = 0.0f;
        for (uint r = 0u; r < Lc; r++) gsum += glog[(t0 + r) * NVH + h];
        const float egL = QW_EXP(gsum);
        float kgeTab[64];
        {   // kge[t] = e^{ΓL−Γt}: ONE backward recurrence — 64 MUFU total.
            float kg = 1.0f;                       // e^{ΓL−Γ(Lc−1)} = 1
            for (uint t = Lc; t-- > 0u;) {
                if (t != Lc - 1u) kg *= QW_EXP(glog[(t0 + t + 1u) * NVH + h]);
                kgeTab[t] = kg;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // S band resident

        // ── Phase 1: UD + o_scr dots (Tensor accelerated for Lc==64) ──
        if (Lc == 64u) {
            // Full-chunk Tensor Phase 1: 4 SIMDgroups compute S x W^T and S x B^T
            const uint sg = tid >> 5u;
            const uint t0_sg = sg * 16u;
            device const float* W_base = Wbuf + (uint64_t)(t0 + t0_sg) * NCOL + h * NH128;
            device const float* B_base = Bbuf + (uint64_t)(t0 + t0_sg) * NCOL + h * NH128;
            device const float* A_base = Abuf + (uint64_t)(t0 + t0_sg) * NCOL + h * NH128 + vx;
            device float* o_base = o_scr + (uint64_t)(t0 + t0_sg) * NCOL + h * NH128 + vx;

            simdgroup_float8x8 ws_acc[2][2];
            simdgroup_float8x8 os_acc[2][2];

            // Initialize os_acc with Abuf transposed (16x16: x in 0..15, t in t0_sg..t0_sg+15)
            simdgroup_load(os_acc[0][0], A_base + 0u * NCOL + 0u, NCOL, ulong2(0, 0), true);
            simdgroup_load(os_acc[0][1], A_base + 8u * NCOL + 0u, NCOL, ulong2(0, 0), true);
            simdgroup_load(os_acc[1][0], A_base + 0u * NCOL + 8u, NCOL, ulong2(0, 0), true);
            simdgroup_load(os_acc[1][1], A_base + 8u * NCOL + 8u, NCOL, ulong2(0, 0), true);

            #pragma unroll
            for (uint i = 0u; i < 2u; i++) {
                #pragma unroll
                for (uint j = 0u; j < 2u; j++) {
                    ws_acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }
            }

            #pragma unroll
            for (uint j_blk = 0u; j_blk < 16u; j_blk++) {
                const uint j_base = j_blk * 8u;

                simdgroup_float8x8 sf[2];
                simdgroup_load(sf[0], S + 0u * 128u + j_base, 128, ulong2(0, 0), false);
                simdgroup_load(sf[1], S + 8u * 128u + j_base, 128, ulong2(0, 0), false);

                simdgroup_float8x8 wf[2];
                simdgroup_load(wf[0], W_base + 0u * NCOL + j_base, NCOL, ulong2(0, 0), true);
                simdgroup_load(wf[1], W_base + 8u * NCOL + j_base, NCOL, ulong2(0, 0), true);

                simdgroup_float8x8 bf[2];
                simdgroup_load(bf[0], B_base + 0u * NCOL + j_base, NCOL, ulong2(0, 0), true);
                simdgroup_load(bf[1], B_base + 8u * NCOL + j_base, NCOL, ulong2(0, 0), true);

                simdgroup_multiply_accumulate(ws_acc[0][0], sf[0], wf[0], ws_acc[0][0]);
                simdgroup_multiply_accumulate(ws_acc[0][1], sf[0], wf[1], ws_acc[0][1]);
                simdgroup_multiply_accumulate(ws_acc[1][0], sf[1], wf[0], ws_acc[1][0]);
                simdgroup_multiply_accumulate(ws_acc[1][1], sf[1], wf[1], ws_acc[1][1]);

                simdgroup_multiply_accumulate(os_acc[0][0], sf[0], bf[0], os_acc[0][0]);
                simdgroup_multiply_accumulate(os_acc[0][1], sf[0], bf[1], os_acc[0][1]);
                simdgroup_multiply_accumulate(os_acc[1][0], sf[1], bf[0], os_acc[1][0]);
                simdgroup_multiply_accumulate(os_acc[1][1], sf[1], bf[1], os_acc[1][1]);
            }

            // Store os_acc transposed directly to o_scr in device memory
            simdgroup_store(os_acc[0][0], o_base + 0u * NCOL + 0u, NCOL, ulong2(0, 0), true);
            simdgroup_store(os_acc[0][1], o_base + 8u * NCOL + 0u, NCOL, ulong2(0, 0), true);
            simdgroup_store(os_acc[1][0], o_base + 0u * NCOL + 8u, NCOL, ulong2(0, 0), true);
            simdgroup_store(os_acc[1][1], o_base + 8u * NCOL + 8u, NCOL, ulong2(0, 0), true);

            // Store ws_acc to WS in threadgroup memory
            simdgroup_store(ws_acc[0][0], WS + 0u * 64u + t0_sg, 64, ulong2(0, 0), false);
            simdgroup_store(ws_acc[0][1], WS + 0u * 64u + t0_sg + 8u, 64, ulong2(0, 0), false);
            simdgroup_store(ws_acc[1][0], WS + 8u * 64u + t0_sg, 64, ulong2(0, 0), false);
            simdgroup_store(ws_acc[1][1], WS + 8u * 64u + t0_sg + 8u, 64, ulong2(0, 0), false);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Cooperative (U - ws) * kgeTab update
            for (uint e = lane; e < 64u * nb; e += NH128) {
                const uint t = e / nb, x = e % nb;
                WS[x * 64u + t] = (Ubuf[(uint64_t)(t0 + t) * NCOL + h * NH128 + vx + x] - WS[x * 64u + t]) * kgeTab[t];
            }
        } else {
            // Ragged chunk fallback
            const uint n2 = Lc * nb;
            for (uint e = lane; e < n2; e += NH128) {
                const uint t = e / nb, x = e % nb;
                device const float4* Wr = (device const float4*)(Wbuf + (uint64_t)(t0 + t) * NCOL + h * NH128);
                device const float4* Br = (device const float4*)(Bbuf + (uint64_t)(t0 + t) * NCOL + h * NH128);
                threadgroup const float4* Sr = (threadgroup const float4*)&S[x * NH128];
                float ws = 0.0f;
                float oacc = Abuf[(uint64_t)(t0 + t) * NCOL + h * NH128 + vx + x];
                for (uint q = 0u; q < 32u; q++) {
                    float4 sv = Sr[q];
                    float4 wv = Wr[q];
                    float4 bv = Br[q];
                    ws = fma(wv.x, sv.x, ws); ws = fma(wv.y, sv.y, ws);
                    ws = fma(wv.z, sv.z, ws); ws = fma(wv.w, sv.w, ws);
                    oacc = fma(bv.x, sv.x, oacc); oacc = fma(bv.y, sv.y, oacc);
                    oacc = fma(bv.z, sv.z, oacc); oacc = fma(bv.w, sv.w, oacc);
                }
                WS[x * 64u + t] = (Ubuf[(uint64_t)(t0 + t) * NCOL + h * NH128 + vx + x] - ws) * kgeTab[t];
                o_scr[(uint64_t)(t0 + t) * NCOL + h * NH128 + vx + x] = oacc;
            }
            for (uint e = lane + n2; e < 64u * nb; e += NH128) {
                const uint t = e / nb, x = e % nb;
                WS[x * 64u + t] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // kge·UD staged in WS

        // ── Phase 2: Single-Pass Cooperative Hybrid Tensor Carry (16 KiB TG Mem) ──
        const uint sg = tid >> 5u;                 // SIMDgroup index (0..3)
        const uint g0 = sg * 16u;                  // Tile 0: cols 0..15, 16..31, 32..47, 48..63
        const uint g1 = (sg + 4u) * 16u;           // Tile 1: cols 64..79, 80..95, 96..111, 112..127
        device const half* kdev_base = y + qCh + i * NH128;

        simdgroup_float8x8 accum0[2][2], accum1[2][2];
        #pragma unroll
        for (uint xh = 0u; xh < 2u; xh++) {
            #pragma unroll
            for (uint nh = 0u; nh < 2u; nh++) {
                simdgroup_load(accum0[xh][nh], S + (8u * xh) * 128u + g0 + 8u * nh, 128, ulong2(0, 0), false);
                auto th0 = accum0[xh][nh].thread_elements();
                th0[0] *= egL; th0[1] *= egL;
                accum0[xh][nh].thread_elements() = th0;

                simdgroup_load(accum1[xh][nh], S + (8u * xh) * 128u + g1 + 8u * nh, 128, ulong2(0, 0), false);
                auto th1 = accum1[xh][nh].thread_elements();
                th1[0] *= egL; th1[1] *= egL;
                accum1[xh][nh].thread_elements() = th1;
            }
        }

        const uint t_r0 = lane >> 5u;              // 0..3
        const uint col0 = (lane & 31u) << 2u;      // 0, 4, 8, ... 124
        const uint t_r1 = t_r0 + 4u;               // 4..7

        for (uint b_sub = 0u; b_sub < 8u; b_sub++) {
            const uint t_base = b_sub * 8u;
            if (t_base >= Lc) break;

            // Cooperative 128-thread load: 128 lanes load all 128 cols x 8 timesteps (2 float4s per lane)
            const uint t_curr0 = t_base + t_r0;
            if (t_curr0 < Lc) {
                device const half4* k_ptr0 = (device const half4*)(kdev_base + (uint64_t)(t0 + t_curr0) * QW_GDN_QKV_CH + col0);
                *(threadgroup float4*)&K_shared[t_r0][col0] = float4(*k_ptr0);
            } else {
                *(threadgroup float4*)&K_shared[t_r0][col0] = float4(0.0f);
            }

            const uint t_curr1 = t_base + t_r1;
            if (t_curr1 < Lc) {
                device const half4* k_ptr1 = (device const half4*)(kdev_base + (uint64_t)(t0 + t_curr1) * QW_GDN_QKV_CH + col0);
                *(threadgroup float4*)&K_shared[t_r1][col0] = float4(*k_ptr1);
            } else {
                *(threadgroup float4*)&K_shared[t_r1][col0] = float4(0.0f);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 af[2];
            simdgroup_load(af[0], WS + 0u * 64u + t_base, 64, ulong2(0, 0), false);
            simdgroup_load(af[1], WS + 8u * 64u + t_base, 64, ulong2(0, 0), false);

            simdgroup_float8x8 bf0[2], bf1[2];
            simdgroup_load(bf0[0], &K_shared[0][g0], 128, ulong2(0, 0), false);
            simdgroup_load(bf0[1], &K_shared[0][g0 + 8u], 128, ulong2(0, 0), false);
            simdgroup_load(bf1[0], &K_shared[0][g1], 128, ulong2(0, 0), false);
            simdgroup_load(bf1[1], &K_shared[0][g1 + 8u], 128, ulong2(0, 0), false);

            simdgroup_multiply_accumulate(accum0[0][0], af[0], bf0[0], accum0[0][0]);
            simdgroup_multiply_accumulate(accum0[0][1], af[0], bf0[1], accum0[0][1]);
            simdgroup_multiply_accumulate(accum0[1][0], af[1], bf0[0], accum0[1][0]);
            simdgroup_multiply_accumulate(accum0[1][1], af[1], bf0[1], accum0[1][1]);

            simdgroup_multiply_accumulate(accum1[0][0], af[0], bf1[0], accum1[0][0]);
            simdgroup_multiply_accumulate(accum1[0][1], af[0], bf1[1], accum1[0][1]);
            simdgroup_multiply_accumulate(accum1[1][0], af[1], bf1[0], accum1[1][0]);
            simdgroup_multiply_accumulate(accum1[1][1], af[1], bf1[1], accum1[1][1]);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        #pragma unroll
        for (uint xh = 0u; xh < 2u; xh++) {
            #pragma unroll
            for (uint nh = 0u; nh < 2u; nh++) {
                simdgroup_store(accum0[xh][nh], S + (8u * xh) * 128u + g0 + 8u * nh, 128, ulong2(0, 0), false);
                simdgroup_store(accum1[xh][nh], S + (8u * xh) * 128u + g1 + 8u * nh, 128, ulong2(0, 0), false);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // next chunk reads S
    }
    for (uint e = lane; e < 128u * nb; e += NH128) {      // write back j-major
        const uint j = e / nb, x = e % nb;
        stg[j * NH128 + vx + x] = S[x * NH128 + j];
    }
}

// K3 — per-row GatedRMSNorm over [T][48] rows, row = t*48+h (agrees with
// K2's o_scr layout; math identical to C12 qwg_gdn_norm_sigmoid: o·rsqrt(mean(o²)
// +1e-6)·w ⊙ silu(z_f32) → f16 store).
kernel void qwg_gdn_prefill_norm(
        device const float* o_scr [[buffer(0)]],
        device const half*  zrow  [[buffer(1)]],   // [T][6144]
        device const half*  w     [[buffer(2)]],   // [128]
        device half*        o_out [[buffer(3)]],   // [T][6144] f16
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint simd [[thread_index_in_simdgroup]],
        uint sg  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[4];
    const uint row = tg.x;                    // t*48 + h
    const uint64_t row_off = (uint64_t)row * NH128;
    const float o = o_scr[row_off + tid];
    const float sq = simd_sum(o * o);
    if (simd == 0u) red[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rms = rsqrt((red[0] + red[1] + red[2] + red[3]) * (1.0f / 128.0f) + QW_RMS_EPS);
    // z: token t = row/48; z[h] at t·6144 + h·128 + tid = row·128 + tid (exact algebraic identity)
    const uint64_t off = row_off + tid;
    const float zn = float(zrow[off]);
    const float gate = QW_GDN_VECTOR_DECAY ? (1.0f / (1.0f + exp(-zn))) : (zn / (1.0f + exp(-zn)));
    const half hg = half((o * rms * float(w[tid])) * gate);
    o_out[off] = simd_fwht32(hg, simd);
}

// qwg_gdn_prefill_regres — EXP-9 sub-item 1, OPT-IN ALTERNATIVE prefill
// recurrence (NOT the shipping default; MTILDE K1/K2/K3 above stays the
// default path). Described as "local MLX-derived": the geometry (four
kernel void qwg_gdn_prefill_regres(
        device const half*  y        [[buffer(0)]],   // [T][10240] q̂|k̂|v
        device const float* betas    [[buffer(1)]],   // [T][48]
        device const float* glog     [[buffer(2)]],   // [T][48] or [T][Z_CH] (vector decay: exp(g))
        device float*       state    [[buffer(3)]],   // [48][128][128] f32, in place
        device float*       oScratch [[buffer(4)]],   // [T][48][128] f32 pre-norm
        constant uint&      T        [[buffer(5)]],
        uint3 tg   [[threadgroup_position_in_grid]],
        uint  lane [[thread_index_in_simdgroup]],
        uint  sg   [[simdgroup_index_in_threadgroup]]) {
    constexpr uint lanesPerRow = 4u;
    constexpr uint rowsPerSG = 8u;          // 32 / lanesPerRow
    constexpr uint valuesPerLane = 32u;     // Dk(128) / lanesPerRow

    const uint h = tg.x;                              // v-head 0..V_HEADS-1
    const uint rowGroup = tg.y * 4u + sg;             // 0..15 (4 tg.y blocks x 4 simdgroups/TG)
    const uint i = h / QW_GDN_V_PER_QK;               // key-head this v-head shares
    const uint qCh = QW_GDN_QK_HEADS * NH128;
    const uint qkCh = 2u * qCh;
    const uint laneInRow = lane & (lanesPerRow - 1u);
    const uint rowInSG = lane >> 2u;                  // 0..7
    const uint v = rowGroup * rowsPerSG + rowInSG;    // this lane's value row 0..127
    const uint kBase = laneInRow * valuesPerLane;     // first Dk index this lane owns

    device float* stg = state + (uint64_t)h * 16384u; // [128 k][128 v]
    float s[valuesPerLane];
    #pragma unroll
    for (uint e = 0u; e < valuesPerLane; e++) s[e] = stg[(kBase + e) * NH128 + v];

    device const half* qbase = y + (uint64_t)i * NH128 + kBase;
    device const half* kbase = y + qCh + (uint64_t)i * NH128 + kBase;

    for (uint t = 0u; t < T; t++) {
        const float gt_scalar = QW_GDN_VECTOR_DECAY ? 1.0f : exp(glog[t * NVH + h]);
        const float beta = betas[t * NVH + h];
        device const half* qrow = qbase + (uint64_t)t * QW_GDN_QKV_CH;
        device const half* krow = kbase + (uint64_t)t * QW_GDN_QKV_CH;
        device const float* grow = QW_GDN_VECTOR_DECAY ? (glog + ((uint64_t)t * NVH + h) * NH128 + kBase) : nullptr;

        float rhoP = 0.0f, kvP = 0.0f, upP = 0.0f;
        #pragma unroll
        for (uint e = 0u; e < valuesPerLane; e++) {
            const float kv_ = float(krow[e]);
            const float qv_ = float(qrow[e]);
            rhoP = fma(kv_, qv_, rhoP);
            const float gt = QW_GDN_VECTOR_DECAY ? grow[e] : gt_scalar;
            const float sd = s[e] * gt;
            s[e] = sd;
            kvP = fma(sd, kv_, kvP);
            upP = fma(sd, qv_, upP);
        }
        float3 red = float3(kvP, upP, rhoP);
        red += simd_shuffle_xor(red, 1u);
        red += simd_shuffle_xor(red, 2u);

        const float vin = float(y[(uint64_t)t * QW_GDN_QKV_CH + qkCh + h * NH128 + v]);
        const float delta = (vin - red.x) * beta;
        const float o = fma(red.z, delta, red.y);

        #pragma unroll
        for (uint e = 0u; e < valuesPerLane; e++) s[e] = fma(float(krow[e]), delta, s[e]);

        if (laneInRow == 0u) {
            oScratch[(uint64_t)t * NCOL + h * NH128 + v] = o;
        }
    }
    #pragma unroll
    for (uint e = 0u; e < valuesPerLane; e++) stg[(kBase + e) * NH128 + v] = s[e];
}
