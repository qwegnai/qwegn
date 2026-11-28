#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "include/qw_dequant.metalh"
#include "include/qw_geometry_constants.metalh"
#include "include/qw_gdn_norm.metalh"
using namespace metal;

// qwgdn_decode.metal — C12: GDN recurrent decode step (post-conv q̂k̂vz,
// state S[48][128k][128v] fp32, in place) + gated-norm epilogue.
//
// Recurrence (golden T0.3, oracle GatedDeltaNet == HF :159-173), per v-head
// h sharing key-head i = h/3 (oracle repeat_interleave(3)):
//   S ← S·exp(g);  kv[v]=Σ_k S[k,v]·k̂[k];  δ=(v−kv)·β;  S[k,v]+=k̂[k]·δ;
//   o[v]=Σ_k S[k,v]·q̂[k] (AFTER update).
// Then per-head GatedRMSNorm (kernel 2): o·rsqrt(mean(o²)+1e-6)·w ⊙ silu(z).
//
// v2 DESIGN (spec-12 "tile split … raced"): v1's column-ownership form
// (lane owns a v-column, 48 TG × 128 lanes) was MEASURED thread-starved —
// 6144 threads issue only 128 strided 4 B accesses each, and the 512 B
// inter-lane stride fragmented every k-row into 16 B sectors (p50 34.4 µs,
// ~2× slower than the state could physically stream). v2 splits the k-axis
// 8-way: 48 TG × 1024 lanes = 49,152 threads; lane = (v-column c = tid&127,
// k-chunk = tid>>7). Within a simdgroup the 32 lanes span 32 CONSECUTIVE
// columns ⇒ every k-row read is one fully coalesced 128 B line. Chunk
// folds cross simdgroups ⇒ two threadgroup-staged reductions (4 KiB red
// buffer, one barrier per phase) — the C09 staged-broadcast doctrine.
//
// ORDER LAW: the staged fold re-associates the fp32 k-sums vs the oracle's
// torch .sum(dim=0) tree. Deltas are fp32-rounding (~1e-7 relative); the
// binding question is whether 1000 compounding steps amplify them past the
// 1e-4 state gate — the parity test is the empirical verdict, and it was
// green with this exact fold order.
//
// Numerics: fp32 from load; q̂ already ×128^-0.5, k̂ already l2-normed
// (norming contract — the producer is the upstream prologue).

#define QW_CHUNKS 8u
#define QW_KC 16u            // k rows per chunk (128 / 8)

// Rejection replay only updates state; it omits readout, norm, and projections.
kernel void qwg_gdn_replay_state(
        device const half* y [[buffer(0)]],
        device const float* betas [[buffer(1)]],
        device const float* glog [[buffer(2)]],
        device float* state [[buffer(3)]],
        uint h [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint c = tid & 127u;
    const uint k0 = (tid >> 7u) * QW_KC;
    const uint qCh = QW_GDN_QK_HEADS * 128u;
    const uint qkCh = 2u * qCh;
    device float* S = state + (uint64_t)h * 16384u;
    device const half* krow = y + qCh + (uint64_t)(h / QW_GDN_V_PER_QK) * 128u;
    const float eg_scalar = QW_GDN_VECTOR_DECAY ? 1.0f : exp(glog[h]);
    threadgroup float red[1024];
    float sreg[QW_KC];
    float kvp = 0.0f;
    #pragma unroll
    for (uint j = 0u; j < QW_KC; ++j) {
        const uint k = k0 + j;
        const float eg = QW_GDN_VECTOR_DECAY ? glog[h * 128u + k] : eg_scalar;
        const float s = S[k * 128u + c] * eg;
        sreg[j] = s;
        kvp = fma(s, float(krow[k]), kvp);
    }
    red[tid] = kvp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float kv = 0.0f;
    for (uint q = 0u; q < QW_CHUNKS; q++) kv += red[q * 128u + c];
    const float delta = (float(y[qkCh + (uint64_t)h * 128u + c]) - kv) * betas[h];
    #pragma unroll
    for (uint j = 0u; j < QW_KC; ++j) {
        const uint k = k0 + j;
        S[k * 128u + c] = sreg[j] + float(krow[k]) * delta;
    }
}

struct QWGDNBatchPtrTable8 {
    ulong addrs[8];
};

struct QWGDNBatchStrides {
    uint yStride;
    uint betasStride;
    uint glogStride;
    uint zStride;
    uint oStride;
    uint oRawStride;
};

static inline void qwg_gdn_recurrent_row(
        device const half* y,
        device const float* betas,
        device const float* glog,
        device float* state,
        device float* o_raw,
        device const half* zrow,
        device const half* w,
        device half* o_out,
        device float* out_gs,
        device float* innovation,
        threadgroup float2* red,
        threadgroup float& rho_shared,
        threadgroup float* redNorm,
        uint h,
        uint tid) {
    const uint c = tid & 127u;           // v-column (32 consecutive per SG)
    const uint kch = tid >> 7u;          // k-chunk 0..7
    const uint i = h / QW_GDN_V_PER_QK;  // KEY head it shares
    const uint qCh = QW_GDN_QK_HEADS * 128u;
    const uint qkCh = 2u * qCh;
    const float beta = betas[h];
    const float eg_scalar = QW_GDN_VECTOR_DECAY ? 1.0f : exp(glog[h]);
    device float* S = state + (uint64_t)h * 16384u;          // S[k][c] at k*128+c
    device const half* qrow = y + (uint64_t)i * 128u;
    device const half* krow = y + qCh + (uint64_t)i * 128u;
    const uint k0 = kch * QW_KC;

    // SIMD group 0 computes scalar rho = dot(k̂, q̂) across the 128 dims in parallel (4 halves per lane)
    if (tid < 32u) {
        device const half4* k4 = (device const half4*)krow;
        device const half4* q4 = (device const half4*)qrow;
        half4 kh = k4[tid];
        half4 qh = q4[tid];
        float partial = float(kh.x) * float(qh.x) + float(kh.y) * float(qh.y)
                      + float(kh.z) * float(qh.z) + float(kh.w) * float(qh.w);
        float rho_sum = simd_sum(partial);
        if (tid == 0u) {
            rho_shared = rho_sum;
        }
    }

    // phase 1: decay + kv partial + u partial (lane owns rows k0..k0+15 of column c)
    float sreg[QW_KC];
    float kvp = 0.0f;
    float up  = 0.0f;
    #pragma unroll
    for (uint j = 0u; j < QW_KC; ++j) {
        const uint k = k0 + j;
        const float eg = QW_GDN_VECTOR_DECAY ? glog[h * 128u + k] : eg_scalar;
        const float s = S[k * 128u + c] * eg;
        sreg[j] = s;
        kvp = fma(s, float(krow[k]), kvp);
        up  = fma(s, float(qrow[k]), up);
    }
    red[tid] = float2(kvp, up);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // fold the 8 chunk-partials of MY column (strided read, 8 loads)
    float2 kv_u = float2(0.0f, 0.0f);
    for (uint q = 0u; q < QW_CHUNKS; q++) {
        kv_u += red[q * 128u + c];
    }
    const float kv = kv_u.x;
    const float u  = kv_u.y;
    const float delta = (float(y[qkCh + (uint64_t)h * 128u + c]) - kv) * beta;

    // Rank-1 readout: o = u + rho * delta (eliminates phase 2 S·q matrix mult + 2 barriers)
    const float o = u + rho_shared * delta;
    if (kch == 0u) {
        if (innovation != nullptr) {
            innovation[(uint64_t)h * 128u + c] = delta;
        }
        if (o_raw != nullptr) {
            o_raw[(uint64_t)h * 128u + c] = o;
        }
        // In-kernel GatedRMSNorm partials: 4 simdgroups in chunk 0
        const float sq = simd_sum(o * o);
        if ((tid & 31u) == 0u) {
            redNorm[tid >> 5u] = sq;
        }
    }

    // phase 2: pure rank-1 state update in place (no reduction)
    #pragma unroll
    for (uint j = 0u; j < QW_KC; ++j) {
        const uint k = k0 + j;
        S[k * 128u + c] = sreg[j] + float(krow[k]) * delta;
    }

    // GatedRMSNorm finish: synchronize redNorm across chunk 0 simdgroups
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup half* hg_buf = (threadgroup half*)red;
    if (kch == 0u && o_out != nullptr) {
        const float total = redNorm[0] + redNorm[1] + redNorm[2] + redNorm[3];
        const float rms = rsqrt(total * (1.0f / 128.0f) + QW_RMS_EPS);
        const half hg = qw_gdn_gated_norm_elem(o, rms, w[c], zrow[(uint64_t)h * 128u + c], c & 31u);
        o_out[(uint64_t)h * 128u + c] = hg;
        if (out_gs != nullptr) {
            hg_buf[c] = hg;
        }
    }

    if (out_gs != nullptr) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32u) {
            const uint ti = tid;
            float s = 0.0f;
            s += float(hg_buf[ti]);
            s += float(hg_buf[ti + 32u]);
            s += float(hg_buf[ti + 64u]);
            s += float(hg_buf[ti + 96u]);
            float4 v = float4(s, 0.0f, 0.0f, 0.0f);
            v += simd_shuffle_down(v, 16);
            v += simd_shuffle_down(v, 8);
            v += simd_shuffle_down(v, 4);
            v += simd_shuffle_down(v, 2);
            v += simd_shuffle_down(v, 1);
            if (ti == 0u) {
                out_gs[h] = v.x;
            }
        }
    }
}

kernel void qwg_gdn_recurrent(
        device const half* y        [[buffer(0)]],   // this step: q̂‖k̂‖v (10240)
        device const float* betas   [[buffer(1)]],   // [48]
        device const float* glog    [[buffer(2)]],   // [48]
        device float* state         [[buffer(3)]],   // [48][128][128] in place
        device float* o_raw         [[buffer(4)]],   // [48][128] fp32 (pre-norm) optional
        device const half*  zrow    [[buffer(5)]],   // this step's z (6144)
        device const half*  w       [[buffer(6)]],   // norm weight (128)
        device half*        o_out   [[buffer(7)]],   // [48][128] f16 (gated norm out)
        device float*       out_gs  [[buffer(8)]],   // [48] f32 optional OutProj group-sums
        device float*       innovation [[buffer(9)]], // [48][128] optional rank-1 update vector
        uint3 tg                    [[threadgroup_position_in_grid]],
        uint tid                    [[thread_index_in_threadgroup]]) {
    threadgroup float2 red[1024];
    threadgroup float rho_shared;
    threadgroup float redNorm[4];
    qwg_gdn_recurrent_row(y, betas, glog, state, o_raw, zrow, w, o_out, out_gs,
                          innovation, red, rho_shared, redNorm, tg.x, tid);
}

kernel void qwg_gdn_recurrent_batch(
        device const half* y                    [[buffer(0)]],
        device const float* betas               [[buffer(1)]],
        device const float* glog                [[buffer(2)]],
        device const half* zBase                [[buffer(3)]],
        device const half* w                    [[buffer(4)]],
        device half* o_out                      [[buffer(5)]],
        device float* o_raw                     [[buffer(6)]],
        constant QWGDNBatchPtrTable8& states    [[buffer(7)]],
        constant QWGDNBatchStrides& strides     [[buffer(8)]],
        uint3 tg                                [[threadgroup_position_in_grid]],
        uint tid                                [[thread_index_in_threadgroup]]) {
    threadgroup float2 red[1024];
    threadgroup float rho_shared;
    threadgroup float redNorm[4];
    const uint b = tg.z;
    device float* state = (device float*)states.addrs[b];
    device float* oRawRow = (o_raw != nullptr) ? (o_raw + (uint64_t)b * strides.oRawStride) : nullptr;
    qwg_gdn_recurrent_row(
        y + (uint64_t)b * strides.yStride,
        betas + (uint64_t)b * strides.betasStride,
        glog + (uint64_t)b * strides.glogStride,
        state, oRawRow,
        zBase + (uint64_t)b * strides.zStride,
        w,
        o_out + (uint64_t)b * strides.oStride,
        nullptr, nullptr, red, rho_shared, redNorm, tg.x, tid);
}

// Verify rows: R tokens through the T = 1 recurrence (qwg_gdn_recurrent) in ONE dispatch.
// Carries S in registers across rows, optionally writing to separate `stateOut` and computing suffix stability.
kernel void qwg_gdn_recurrent_rows(
        device const half*        y                  [[buffer(0)]],   // [R][10240] post-conv q̂‖k̂‖v
        device const float*       betas              [[buffer(1)]],   // [R][48]
        device const float*       glog               [[buffer(2)]],   // [R][48] or [R][Z_CH]
        device const float*       state              [[buffer(3)]],   // [48][128][128] input
        device const half*        zBase              [[buffer(4)]],   // row r at zBase + r * zStride
        device const half*        w                  [[buffer(5)]],   // norm weight [128]
        device half*              o_out              [[buffer(6)]],   // [R][48][128] f16 (gated norm + fwht32 out)
        device float*             innovation         [[buffer(7)]],   // optional [R][48][128] rank-1 update vector
        constant uint&            R                  [[buffer(8)]],
        constant uint&            zStride            [[buffer(9)]],
        device float*             stateOut           [[buffer(10)]],  // [48][128][128] output (may alias state)
        device atomic_uint*       unstableSuffixMask [[buffer(11)]],  // optional bit s => suffix s..<R unstable
        uint3 tg                                     [[threadgroup_position_in_grid]],
        uint tid                                     [[thread_index_in_threadgroup]]) {
    const uint h = tg.x;                 // v-head 0..V_HEADS-1
    const uint c = tid & 127u;           // v-column (32 consecutive per SG)
    const uint kch = tid >> 7u;          // k-chunk 0..7
    const uint i = h / QW_GDN_V_PER_QK;  // KEY head it shares
    const uint qCh = QW_GDN_QK_HEADS * 128u;
    const uint qkCh = 2u * qCh;
    const uint k0 = kch * QW_KC;
    device const float* S_in = state + (uint64_t)h * 16384u;
    device float* S_out = ((stateOut != nullptr) ? stateOut : (device float*)state) + (uint64_t)h * 16384u;

    float sreg[QW_KC];
    #pragma unroll
    for (uint j = 0u; j < QW_KC; ++j) {
        sreg[j] = S_in[(k0 + j) * 128u + c];
    }

    threadgroup float2 red[1024];
    threadgroup float rho_shared;
    threadgroup float redNorm[4];
    uint badMask = 0u;

    for (uint r = 0u; r < R; ++r) {
        device const half* yRow = y + (uint64_t)r * QW_GDN_QKV_CH;
        device const half* qrow = yRow + (uint64_t)i * 128u;
        device const half* krow = yRow + qCh + (uint64_t)i * 128u;
        const float beta = betas[(uint64_t)r * QW_GDN_V_HEADS + h];
        const float eg_scalar = QW_GDN_VECTOR_DECAY ? 1.0f : exp(glog[(uint64_t)r * QW_GDN_V_HEADS + h]);
        device const float* grow = QW_GDN_VECTOR_DECAY ? (glog + ((uint64_t)r * QW_GDN_V_HEADS + h) * 128u) : nullptr;

        if (tid < 32u) {
            device const half4* k4 = (device const half4*)krow;
            device const half4* q4 = (device const half4*)qrow;
            half4 kh = k4[tid];
            half4 qh = q4[tid];
            float partial = float(kh.x) * float(qh.x) + float(kh.y) * float(qh.y)
                          + float(kh.z) * float(qh.z) + float(kh.w) * float(qh.w);
            float rho_sum = simd_sum(partial);
            if (tid == 0u) {
                rho_shared = rho_sum;
            }
        }

        float kvp = 0.0f;
        float up  = 0.0f;
        #pragma unroll
        for (uint j = 0u; j < QW_KC; ++j) {
            const uint k = k0 + j;
            const float eg = QW_GDN_VECTOR_DECAY ? grow[k] : eg_scalar;
            const float s = sreg[j] * eg;
            sreg[j] = s;
            kvp = fma(s, float(krow[k]), kvp);
            up  = fma(s, float(qrow[k]), up);
        }
        red[tid] = float2(kvp, up);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float2 kv_u = float2(0.0f, 0.0f);
        for (uint q = 0u; q < QW_CHUNKS; q++) {
            kv_u += red[q * 128u + c];
        }
        const float kv = kv_u.x;
        const float u  = kv_u.y;
        const float delta = (float(yRow[qkCh + (uint64_t)h * 128u + c]) - kv) * beta;

        const float o = u + rho_shared * delta;
        if (kch == 0u) {
            if (innovation != nullptr) {
                innovation[(uint64_t)r * QW_GDN_Z_CH + (uint64_t)h * 128u + c] = delta;
            }
            if (unstableSuffixMask != nullptr && R <= 16u) {
                const bool badDelta = (as_type<uint>(delta) & 0x7F800000u) == 0x7F800000u;
                const bool badKey = (as_type<ushort>(krow[c]) & 0x7C00u) == 0x7C00u;
                if (badDelta || badKey) {
                    badMask |= (1u << (r + 1u)) - 1u;
                }
            }
            const float sq = simd_sum(o * o);
            if ((tid & 31u) == 0u) {
                redNorm[tid >> 5u] = sq;
            }
        }

        #pragma unroll
        for (uint j = 0u; j < QW_KC; ++j) {
            const uint k = k0 + j;
            sreg[j] = sreg[j] + float(krow[k]) * delta;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kch == 0u && o_out != nullptr) {
            const uint idx = h * 128u + c;
            const float total = redNorm[0] + redNorm[1] + redNorm[2] + redNorm[3];
            const float rms = rsqrt(total * (1.0f / 128.0f) + QW_RMS_EPS);
            const half hg = qw_gdn_gated_norm_elem(o, rms, w[c], zBase[(uint64_t)r * zStride + idx], c & 31u);
            o_out[(uint64_t)r * QW_GDN_Z_CH + idx] = hg;
        }
        if (r + 1u < R) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    #pragma unroll
    for (uint j = 0u; j < QW_KC; ++j) {
        S_out[(k0 + j) * 128u + c] = sreg[j];
    }

    if (unstableSuffixMask != nullptr && R <= 16u) {
        if (QW_GDN_VECTOR_DECAY) {
            if (kch == 0u) {
                for (uint s = 0u; s < R; ++s) {
                    float cum = 1.0f;
                    for (uint rr = s; rr < R; ++rr) {
                        const float d = glog[((uint64_t)rr * QW_GDN_V_HEADS + h) * 128u + c];
                        if ((as_type<uint>(d) & 0x7F800000u) == 0x7F800000u || d < 1.0e-3f || d > 1.0e38f) {
                            badMask |= (1u << s);
                            break;
                        }
                        cum *= d;
                        if ((as_type<uint>(cum) & 0x7F800000u) == 0x7F800000u || cum < 1.0e-3f || cum > 1.0e38f) {
                            badMask |= (1u << s);
                            break;
                        }
                    }
                }
            }
        } else if (tid == 0u) {
            for (uint s = 0u; s < R; ++s) {
                float cum = 0.0f;
                for (uint rr = s; rr < R; ++rr) {
                    const float d = glog[(uint64_t)rr * QW_GDN_V_HEADS + h];
                    if ((as_type<uint>(d) & 0x7F800000u) == 0x7F800000u || d < -6.907755f || d > 88.0f) {
                        badMask |= (1u << s);
                        break;
                    }
                    cum += d;
                    if ((as_type<uint>(cum) & 0x7F800000u) == 0x7F800000u || cum < -6.907755f) {
                        badMask |= (1u << s);
                        break;
                    }
                }
            }
        }
        badMask = simd_or(badMask);
        if ((tid & 31u) == 0u && badMask != 0u) {
            atomic_fetch_or_explicit(unstableSuffixMask, badMask, memory_order_relaxed);
        }
    }
}

// Undo R normalized GDN rank-1 updates in reverse order (r = R - 1 down to 0) in registers.
kernel void qwg_gdn_rollback_rows(
        device const half*  y          [[buffer(0)]],   // [R][QKV_CH]
        device const float* innovation [[buffer(1)]],   // [R][Z_CH]
        device const float* glog       [[buffer(2)]],   // [R][V_HEADS] or [R][Z_CH]
        device const float* state      [[buffer(3)]],   // [V_HEADS][128][128] input
        constant uint&      R          [[buffer(4)]],
        device float*       stateOut   [[buffer(5)]],   // [V_HEADS][128][128] output (may alias state)
        uint h                         [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]]) {
    const uint c = tid & 127u;
    const uint kChunk = tid >> 7u;
    const uint k0 = kChunk * 16u;
    const uint qCh = QW_GDN_QK_HEADS * 128u;
    device const float* S_in = state + (uint64_t)h * 16384u;
    device float* S_out = ((stateOut != nullptr) ? stateOut : (device float*)state) + (uint64_t)h * 16384u;
    float sreg[16];
    #pragma unroll
    for (uint j = 0u; j < 16u; ++j) {
        sreg[j] = S_in[(k0 + j) * 128u + c];
    }
    for (int r = int(R) - 1; r >= 0; --r) {
        device const half* key = y + (uint64_t)r * QW_GDN_QKV_CH + qCh + (uint64_t)(h / QW_GDN_V_PER_QK) * 128u;
        const float alpha_scalar = QW_GDN_VECTOR_DECAY ? 1.0f : exp(glog[(uint64_t)r * QW_GDN_V_HEADS + h]);
        device const float* grow = QW_GDN_VECTOR_DECAY ? (glog + ((uint64_t)r * QW_GDN_V_HEADS + h) * 128u) : nullptr;
        const float delta = innovation[(uint64_t)r * QW_GDN_Z_CH + (uint64_t)h * 128u + c];
        #pragma unroll
        for (uint j = 0u; j < 16u; ++j) {
            const float alpha = QW_GDN_VECTOR_DECAY ? grow[k0 + j] : alpha_scalar;
            sreg[j] = (sreg[j] - float(key[k0 + j]) * delta) / alpha;
        }
    }
    #pragma unroll
    for (uint j = 0u; j < 16u; ++j) {
        S_out[(k0 + j) * 128u + c] = sreg[j];
    }
}

// Fused low-rank KDA gates: f_b_proj(f_low) + dt_bias -> glog (exp(g)), g_b_proj(g_low) -> zOut.
// Grid: (QW_GDN_V_HEADS, T, 1) x QW_GDN_HEAD_DIM threads/TG.
kernel void qwg_kda_gates(
        device const half*  fgb            [[buffer(0)]],   // [T][2*lowRank + QW_GDN_V_HEADS]
        device const half*  fB             [[buffer(1)]],   // [QW_GDN_Z_CH][lowRank]
        device const half*  gB             [[buffer(2)]],   // [QW_GDN_Z_CH][lowRank]
        device const float* aLog           [[buffer(3)]],   // [QW_GDN_V_HEADS]
        device const float* dtBias         [[buffer(4)]],   // [QW_GDN_Z_CH]
        device float*       betas          [[buffer(5)]],   // [T][QW_GDN_V_HEADS]
        device float*       glog           [[buffer(6)]],   // [T][QW_GDN_Z_CH]
        device half*        zOut           [[buffer(7)]],   // [T][QW_GDN_Z_CH]
        constant float&     gateLowerBound [[buffer(8)]],
        constant uint&      T              [[buffer(9)]],
        constant uint&      lowRank        [[buffer(10)]],
        uint2 tg  [[threadgroup_position_in_grid]],
        uint  tid [[thread_index_in_threadgroup]]) {
    const uint h = tg.x;
    const uint t = tg.y;
    if (h >= QW_GDN_V_HEADS || t >= T || tid >= QW_GDN_HEAD_DIM) return;

    const uint fgbStride = 2u * lowRank + QW_GDN_V_HEADS;
    device const half* row = fgb + (size_t)t * fgbStride;

    threadgroup float sF[256];
    threadgroup float sG[256];
    threadgroup float sDecayRate;

    for (uint i = tid; i < lowRank; i += QW_GDN_HEAD_DIM) {
        sF[i] = float(row[i]);
        sG[i] = float(row[lowRank + i]);
    }
    if (tid == 0u) {
        const float bVal = float(row[2u * lowRank + h]);
        betas[(size_t)t * QW_GDN_V_HEADS + h] = 1.0f / (1.0f + exp(-bVal));
        sDecayRate = exp(aLog[h]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint c = h * QW_GDN_HEAD_DIM + tid;
    device const half4* fRow4 = (device const half4*)(fB + (size_t)c * lowRank);
    device const half4* gRow4 = (device const half4*)(gB + (size_t)c * lowRank);
    threadgroup const float4* sF4 = (threadgroup const float4*)sF;
    threadgroup const float4* sG4 = (threadgroup const float4*)sG;

    float fAcc = 0.0f;
    float gAcc = 0.0f;
    const uint qCount = lowRank >> 2u;
    for (uint q = 0u; q < qCount; ++q) {
        const float4 fv = sF4[q];
        const float4 gv = sG4[q];
        const float4 fw = float4(fRow4[q]);
        const float4 gw = float4(gRow4[q]);
        fAcc = fma(fv.x, fw.x, fma(fv.y, fw.y, fma(fv.z, fw.z, fma(fv.w, fw.w, fAcc))));
        gAcc = fma(gv.x, gw.x, fma(gv.y, gw.y, fma(gv.z, gw.z, fma(gv.w, gw.w, gAcc))));
    }

    const float gVal = fAcc + dtBias[c];
    const size_t outIdx = (size_t)t * QW_GDN_Z_CH + c;
    if (gateLowerBound > -1.0e30f) {
        glog[outIdx] = exp(gateLowerBound / (1.0f + exp(-(sDecayRate * gVal))));
    } else {
        const float sp = (gVal > 20.0f) ? gVal : log(1.0f + exp(gVal));
        glog[outIdx] = exp(-sDecayRate * sp);
    }
    zOut[outIdx] = half(gAcc);
}

// Prefill variant of qwg_kda_gates on simdgroup MMA (32 tokens x one head, sequential f/g stage <= 17 KiB).
// Grid: (QW_GDN_V_HEADS, (T + 31) / 32, 1) x 128 threads/TG.
#define KDA_MMA_TM 32u
#define KDA_MMA_AS 264u

static inline uint qw_kda_row(uint l) { return ((l >> 4) & 1u) * 4u + ((l >> 2) & 1u) * 2u + ((l >> 1) & 1u); }
static inline uint qw_kda_col(uint l) { return ((l >> 3) & 1u) * 4u + (l & 1u) * 2u; }

kernel void qwg_kda_gates_mma(
        device const half*  fgb            [[buffer(0)]],
        device const half*  fB             [[buffer(1)]],
        device const half*  gB             [[buffer(2)]],
        device const float* aLog           [[buffer(3)]],
        device const float* dtBias         [[buffer(4)]],
        device float*       betas          [[buffer(5)]],
        device float*       glog           [[buffer(6)]],
        device half*        zOut           [[buffer(7)]],
        constant float&     gateLowerBound [[buffer(8)]],
        constant uint&      T              [[buffer(9)]],
        constant uint&      lowRank        [[buffer(10)]],
        uint2 tg   [[threadgroup_position_in_grid]],
        uint  tid  [[thread_index_in_threadgroup]],
        uint  sg   [[simdgroup_index_in_threadgroup]],
        uint  lane [[thread_index_in_simdgroup]]) {
    const uint h = tg.x;
    const uint tBase = tg.y * KDA_MMA_TM;
    if (h >= QW_GDN_V_HEADS || tBase >= T) return;
    const uint fgbStride = 2u * lowRank + QW_GDN_V_HEADS;
    const uint aStride = lowRank + 8u;
    const uint jTiles = QW_GDN_HEAD_DIM >> 3u;

    threadgroup half shA[KDA_MMA_TM * KDA_MMA_AS];
    if (tid < KDA_MMA_TM && tBase + tid < T) {
        const float bVal = float(fgb[(size_t)(tBase + tid) * fgbStride + 2u * lowRank + h]);
        betas[(size_t)(tBase + tid) * QW_GDN_V_HEADS + h] = 1.0f / (1.0f + exp(-bVal));
    }
    const float decayRate = exp(aLog[h]);
    const uint t = tBase + sg * 8u + qw_kda_row(lane);
    const bool tValid = t < T;
    const uint total = KDA_MMA_TM * lowRank;
    for (uint m = 0u; m < 2u; ++m) {
        const uint srcOff = m * lowRank;
        for (uint i = tid; i < total; i += 128u) {
            const uint r = i / lowRank;
            const uint k = i - r * lowRank;
            shA[r * aStride + k] = (tBase + r < T) ? fgb[(size_t)(tBase + r) * fgbStride + srcOff + k] : half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* a = shA + sg * 8u * aStride;
        device const half* W = (m == 0u ? fB : gB) + (size_t)h * QW_GDN_HEAD_DIM * lowRank;
        simdgroup_float8x8 acc[16];
        #pragma unroll
        for (uint j = 0u; j < 16u; ++j) acc[j] = simdgroup_float8x8(0.0f);
        for (uint kk = 0u; kk < lowRank; kk += 8u) {
            simdgroup_half8x8 af;
            simdgroup_load(af, a + kk, aStride);
            for (uint j = 0u; j < jTiles; ++j) {
                simdgroup_half8x8 bf;
                simdgroup_load(bf, W + (size_t)j * 8u * lowRank + kk, lowRank, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc[j], af, bf, acc[j]);
            }
        }
        if (tValid) {
            for (uint j = 0u; j < jTiles; ++j) {
                auto e = acc[j].thread_elements();
                const uint c0 = h * QW_GDN_HEAD_DIM + j * 8u + qw_kda_col(lane);
                const size_t o = (size_t)t * QW_GDN_Z_CH + c0;
                #pragma unroll
                for (uint q = 0u; q < 2u; ++q) {
                    const float v = e[q];
                    if (m == 0u) {
                        const float gVal = v + dtBias[c0 + q];
                        if (gateLowerBound > -1.0e30f) {
                            glog[o + q] = exp(gateLowerBound / (1.0f + exp(-(decayRate * gVal))));
                        } else {
                            const float sp = (gVal > 20.0f) ? gVal : log(1.0f + exp(gVal));
                            glog[o + q] = exp(-decayRate * sp);
                        }
                    } else {
                        zOut[o + q] = half(v);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Full-rank KDA gate transformation (`no_kda_lora = true`) with optional raw F16 b_proj dot product.
// Grid: (QW_GDN_V_HEADS, T, 1) x QW_GDN_HEAD_DIM threads/TG.
kernel void qwg_kda_gates_direct(
        device const half*  fgb            [[buffer(0)]],
        device const half*  ln1            [[buffer(1)]],
        device const half*  bProjF16       [[buffer(2)]],
        device const float* aLog           [[buffer(3)]],
        device const float* dtBias         [[buffer(4)]],
        device float*       betas          [[buffer(5)]],
        device float*       glog           [[buffer(6)]],
        device half*        zOut           [[buffer(7)]],
        constant float&     gateLowerBound [[buffer(8)]],
        constant uint&      T              [[buffer(9)]],
        constant uint&      hasRawBProj    [[buffer(10)]],
        uint2 tg  [[threadgroup_position_in_grid]],
        uint  tid [[thread_index_in_threadgroup]]) {
    const uint h = tg.x;
    const uint t = tg.y;
    if (h >= QW_GDN_V_HEADS || t >= T || tid >= QW_GDN_HEAD_DIM) return;

    const uint fgbStride = (hasRawBProj != 0u) ? (2u * QW_GDN_Z_CH) : (2u * QW_GDN_Z_CH + QW_GDN_V_HEADS);
    device const half* fgbRow = fgb + (size_t)t * fgbStride;

    threadgroup float sgSum[8];
    threadgroup float sDecayRate;
    if (hasRawBProj != 0u) {
        device const half* xRow = ln1 + (size_t)t * QW_HIDDEN;
        device const half* bRow = bProjF16 + (size_t)h * QW_HIDDEN;
        float bPart = 0.0f;
        for (uint k = tid; k < QW_HIDDEN; k += QW_GDN_HEAD_DIM) {
            bPart = fma(float(xRow[k]), float(bRow[k]), bPart);
        }
        bPart = simd_sum(bPart);
        if ((tid & 31u) == 0u) sgSum[tid >> 5u] = bPart;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            const uint nSG = (QW_GDN_HEAD_DIM + 31u) >> 5u;
            float bVal = 0.0f;
            for (uint s = 0u; s < nSG; ++s) bVal += sgSum[s];
            betas[(size_t)t * QW_GDN_V_HEADS + h] = 1.0f / (1.0f + exp(-bVal));
            sDecayRate = exp(aLog[h]);
        }
    } else if (tid == 0u) {
        const float bVal = float(fgbRow[2u * QW_GDN_Z_CH + h]);
        betas[(size_t)t * QW_GDN_V_HEADS + h] = 1.0f / (1.0f + exp(-bVal));
        sDecayRate = exp(aLog[h]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint c = h * QW_GDN_HEAD_DIM + tid;
    const float gVal = float(fgbRow[c]) + dtBias[c];
    const size_t outIdx = (size_t)t * QW_GDN_Z_CH + c;
    if (gateLowerBound > -1.0e30f) {
        glog[outIdx] = exp(gateLowerBound / (1.0f + exp(-(sDecayRate * gVal))));
    } else {
        const float sp = (gVal > 20.0f) ? gVal : log(1.0f + exp(gVal));
        glog[outIdx] = exp(-sDecayRate * sp);
    }
    zOut[outIdx] = fgbRow[QW_GDN_Z_CH + c];
}
