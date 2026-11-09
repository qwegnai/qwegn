// ── QWGatedResidual — Fused 4-Branch Gated Residual & Epilogue (spec-23) ───────
#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_geometry_constants.metalh"

using namespace metal;

// Register-tile size for 5120 hidden size: 640 uint4 ÷ 32 lanes = 20 uint4/lane.
#define QW_RES2_U4_PER_LANE 20u
#define QW_RES2_U4_TOTAL (QW_RES2_U4_PER_LANE * 32u)

// uint4 == 8 halves; H8 reinterprets it for element access in registers.
struct H8 { half v[8]; } __attribute__((__packed__, __aligned__(16)));
static inline H8 asH8(uint4 w) { return *(thread H8*)&w; }

// Plus-one helper for standard Qwen RMSNorm gamma weights: (1 + gamma)
static inline H8 plusOneH8(uint4 w) {
    H8 g = asH8(w), o;
    for (int i = 0; i < 8; i++) o.v[i] = half(float(g.v[i]) + 1.0f);
    return o;
}

// Flags for 4-branch gated residual execution
constant uint QW_RES_HAS_GDN          = 1u << 0;
constant uint QW_RES_HAS_ATTN         = 1u << 1;
constant uint QW_RES_HAS_MLP          = 1u << 2;
constant uint QW_RES_HAS_PLE          = 1u << 3;
constant uint QW_RES_HAS_GATE         = 1u << 4;
constant uint QW_RES_PLE_SIGMOID      = 1u << 5;
constant uint QW_RES_GAMMA_PLUS_ONE   = 1u << 6;

static inline half computeFusedResidualElement(
    half cur,
    half gdn_val,
    half attn_val,
    half mlp_val,
    half ple_val,
    half gate_val,
    uint flags
) {
    // 1. Recurrent GDN state stream
    if (flags & QW_RES_HAS_GDN) {
        cur += gdn_val;
    }
    // 2. Full Attention output stream
    if (flags & QW_RES_HAS_ATTN) {
        cur += attn_val;
    }
    // 3. SwiGLU MLP output stream
    if (flags & QW_RES_HAS_MLP) {
        cur += mlp_val;
    }
    // 4. Discrete PLE / N-Gram injection stream (gated via sigma(gate) * ple)
    if (flags & QW_RES_HAS_PLE) {
        half g = 1.0h;
        if (flags & QW_RES_HAS_GATE) {
            if (flags & QW_RES_PLE_SIGMOID) {
                float gf = 1.0f / (1.0f + metal::precise::exp(-float(gate_val)));
                g = half(gf);
            } else {
                g = gate_val;
            }
        }
        cur = metal::fma(ple_val, g, cur);
    }
    return cur;
}

static inline H8 computeFusedResidualH8(
    H8 cur,
    H8 gdn,
    H8 attn,
    H8 mlp,
    H8 ple,
    H8 gate,
    uint flags
) {
    H8 out;
    for (int i = 0; i < 8; i++) {
        out.v[i] = computeFusedResidualElement(
            cur.v[i],
            gdn.v[i],
            attn.v[i],
            mlp.v[i],
            ple.v[i],
            gate.v[i],
            flags
        );
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: Fused 4-Branch Gated Residual Addition (In-place residual update)
// ─────────────────────────────────────────────────────────────────────────────
kernel void qwg_gated_residual_4branch(
    device uint4* residual            [[buffer(0)]], // in/out: existing residual
    device const uint4* gdn_out       [[buffer(1)]], // optional: GDN recurrent output
    device const uint4* attn_out      [[buffer(2)]], // optional: Attention output
    device const uint4* mlp_out       [[buffer(3)]], // optional: SwiGLU MLP output
    device const uint4* ple_embed     [[buffer(4)]], // optional: Discrete PLE / ngram embedding
    device const uint4* ple_gate      [[buffer(5)]], // optional: PLE gating activations
    constant uint& hiddenU4           [[buffer(6)]], // hidden / 8
    constant uint& flags              [[buffer(7)]],
    uint tg [[threadgroup_position_in_grid]],
    uint ti [[thread_index_in_threadgroup]]
) {
    device uint4* rr = residual + (size_t)tg * hiddenU4;
    device const uint4* gr = gdn_out ? gdn_out + (size_t)tg * hiddenU4 : nullptr;
    device const uint4* ar = attn_out ? attn_out + (size_t)tg * hiddenU4 : nullptr;
    device const uint4* mr = mlp_out ? mlp_out + (size_t)tg * hiddenU4 : nullptr;
    device const uint4* pr = ple_embed ? ple_embed + (size_t)tg * hiddenU4 : nullptr;
    device const uint4* tr = ple_gate ? ple_gate + (size_t)tg * hiddenU4 : nullptr;

    for (uint u = ti; u < hiddenU4; u += 32) {
        H8 cur = asH8(rr[u]);
        H8 gdn = ((flags & QW_RES_HAS_GDN) && gr) ? asH8(gr[u]) : H8{};
        H8 attn = ((flags & QW_RES_HAS_ATTN) && ar) ? asH8(ar[u]) : H8{};
        H8 mlp = ((flags & QW_RES_HAS_MLP) && mr) ? asH8(mr[u]) : H8{};
        H8 ple = ((flags & QW_RES_HAS_PLE) && pr) ? asH8(pr[u]) : H8{};
        H8 gate = ((flags & QW_RES_HAS_GATE) && tr) ? asH8(tr[u]) : H8{};

        H8 updated = computeFusedResidualH8(cur, gdn, attn, mlp, ple, gate, flags);
        rr[u] = *(thread uint4*)&updated;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: Fused 4-Branch Gated Residual + RMSNorm Epilogue
// Updates residual in place and produces y = rmsnorm(residual) * (1 + gamma)
kernel void qwg_rmsnorm_gated_residual_4branch(
    device uint4* residual            [[buffer(0)]], // in/out: existing residual
    device const uint4* gdn_out       [[buffer(1)]], // optional: GDN recurrent output
    device const uint4* attn_out      [[buffer(2)]], // optional: Attention output
    device const uint4* mlp_out       [[buffer(3)]], // optional: SwiGLU MLP output
    device const uint4* ple_embed     [[buffer(4)]], // optional: Discrete PLE / ngram embedding
    device const uint4* ple_gate      [[buffer(5)]], // optional: PLE gating activations
    device const uint4* gamma         [[buffer(6)]], // RMSNorm weight [hidden/8]
    device uint4* y                   [[buffer(7)]], // output normed row [rows * hidden/8]
    constant uint& hiddenU4           [[buffer(8)]], // hidden / 8
    constant float& eps               [[buffer(9)]], // RMSNorm eps (1e-6)
    constant uint& flags              [[buffer(10)]],
    uint tg [[threadgroup_position_in_grid]],
    uint ti [[thread_index_in_threadgroup]]
) {
    device uint4* rr = residual + (size_t)tg * hiddenU4;
    device const uint4* gr = gdn_out ? gdn_out + (size_t)tg * hiddenU4 : nullptr;
    device const uint4* ar = attn_out ? attn_out + (size_t)tg * hiddenU4 : nullptr;
    device const uint4* mr = mlp_out ? mlp_out + (size_t)tg * hiddenU4 : nullptr;
    device const uint4* pr = ple_embed ? ple_embed + (size_t)tg * hiddenU4 : nullptr;
    device const uint4* tr = ple_gate ? ple_gate + (size_t)tg * hiddenU4 : nullptr;
    device uint4* yr = y + (size_t)tg * hiddenU4;

    float acc = 0.0f;

    // Fast-path: register-resident when hidden matches standard 5120 geometry
    if (hiddenU4 == QW_RES2_U4_TOTAL) {
        uint4 reg[QW_RES2_U4_PER_LANE];

        #pragma unroll
        for (uint k = 0; k < QW_RES2_U4_PER_LANE; k++) {
            uint u = ti + 32 * k;
            H8 cur = asH8(rr[u]);
            H8 gdn = ((flags & QW_RES_HAS_GDN) && gr) ? asH8(gr[u]) : H8{};
            H8 attn = ((flags & QW_RES_HAS_ATTN) && ar) ? asH8(ar[u]) : H8{};
            H8 mlp = ((flags & QW_RES_HAS_MLP) && mr) ? asH8(mr[u]) : H8{};
            H8 ple = ((flags & QW_RES_HAS_PLE) && pr) ? asH8(pr[u]) : H8{};
            H8 gate = ((flags & QW_RES_HAS_GATE) && tr) ? asH8(tr[u]) : H8{};

            H8 updated = computeFusedResidualH8(cur, gdn, attn, mlp, ple, gate, flags);
            for (int i = 0; i < 8; i++) {
                float t = float(updated.v[i]);
                acc += t * t;
            }
            uint4 packed = *(thread uint4*)&updated;
            rr[u] = packed;
            reg[k] = packed;
        }

        float total = simd_sum(acc);
        float rstd = rsqrt(total / float(hiddenU4 * 8) + eps);

        #pragma unroll
        for (uint k = 0; k < QW_RES2_U4_PER_LANE; k++) {
            uint u = ti + 32 * k;
            H8 bv = asH8(reg[k]);
            H8 gv = (flags & QW_RES_GAMMA_PLUS_ONE) ? plusOneH8(gamma[u]) : asH8(gamma[u]);
            H8 ov;
            for (int i = 0; i < 8; i++) {
                ov.v[i] = half(float(bv.v[i]) * rstd * float(gv.v[i]));
            }
            yr[u] = *(thread uint4*)&ov;
        }
    } else {
        // Dynamic path for arbitrary hidden size (zero shared memory allocation)
        for (uint u = ti; u < hiddenU4; u += 32) {
            H8 cur = asH8(rr[u]);
            H8 gdn = ((flags & QW_RES_HAS_GDN) && gr) ? asH8(gr[u]) : H8{};
            H8 attn = ((flags & QW_RES_HAS_ATTN) && ar) ? asH8(ar[u]) : H8{};
            H8 mlp = ((flags & QW_RES_HAS_MLP) && mr) ? asH8(mr[u]) : H8{};
            H8 ple = ((flags & QW_RES_HAS_PLE) && pr) ? asH8(pr[u]) : H8{};
            H8 gate = ((flags & QW_RES_HAS_GATE) && tr) ? asH8(tr[u]) : H8{};

            H8 updated = computeFusedResidualH8(cur, gdn, attn, mlp, ple, gate, flags);
            for (int i = 0; i < 8; i++) {
                float t = float(updated.v[i]);
                acc += t * t;
            }
            rr[u] = *(thread uint4*)&updated;
        }

        float total = simd_sum(acc);
        float rstd = rsqrt(total / float(hiddenU4 * 8) + eps);

        for (uint u = ti; u < hiddenU4; u += 32) {
            H8 bv = asH8(rr[u]);
            H8 gv = (flags & QW_RES_GAMMA_PLUS_ONE) ? plusOneH8(gamma[u]) : asH8(gamma[u]);
            H8 ov;
            for (int i = 0; i < 8; i++) {
                ov.v[i] = half(float(bv.v[i]) * rstd * float(gv.v[i]));
            }
            yr[u] = *(thread uint4*)&ov;
        }
    }
}

