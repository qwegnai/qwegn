#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "include/qw_dequant.metalh"
#include "include/qw_geometry_constants.metalh"
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
    const uint h = tg.x;                 // v-head 0..V_HEADS-1
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
    threadgroup float2 red[1024];
    threadgroup float rho_shared;
    threadgroup float redNorm[4];
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
        const float zn = float(zrow[(uint64_t)h * 128u + c]);
        const float gate = QW_GDN_VECTOR_DECAY ? (1.0f / (1.0f + exp(-zn))) : (zn / (1.0f + exp(-zn)));
        const float gated = (o * rms * float(w[c])) * gate;
        const half hg = simd_fwht32(half(gated), c & 31u);
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

// Verify rows: R tokens through the T = 1 recurrence (qwg_gdn_recurrent) in ONE dispatch.
// Carries S in registers across rows so the 151 MB GDN state is read once and written once per verify round.
kernel void qwg_gdn_recurrent_rows(
        device const half*  y          [[buffer(0)]],   // [R][10240] post-conv q̂‖k̂‖v
        device const float* betas      [[buffer(1)]],   // [R][48]
        device const float* glog       [[buffer(2)]],   // [R][48] or [R][Z_CH]
        device float*       state      [[buffer(3)]],   // [48][128][128] in place
        device const half*  zBase      [[buffer(4)]],   // row r at zBase + r * zStride
        device const half*  w          [[buffer(5)]],   // norm weight [128]
        device half*        o_out      [[buffer(6)]],   // [R][48][128] f16 (gated norm + fwht32 out)
        device float*       innovation [[buffer(7)]],   // optional [R][48][128] rank-1 update vector
        constant uint&      R          [[buffer(8)]],
        constant uint&      zStride    [[buffer(9)]],
        uint3 tg                       [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]]) {
    const uint h = tg.x;                 // v-head 0..V_HEADS-1
    const uint c = tid & 127u;           // v-column (32 consecutive per SG)
    const uint kch = tid >> 7u;          // k-chunk 0..7
    const uint i = h / QW_GDN_V_PER_QK;  // KEY head it shares
    const uint qCh = QW_GDN_QK_HEADS * 128u;
    const uint qkCh = 2u * qCh;
    const uint k0 = kch * QW_KC;
    device float* S = state + (uint64_t)h * 16384u;

    float sreg[QW_KC];
    #pragma unroll
    for (uint j = 0u; j < QW_KC; ++j) {
        sreg[j] = S[(k0 + j) * 128u + c];
    }

    threadgroup float2 red[1024];
    threadgroup float rho_shared;
    threadgroup float redNorm[4];

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
            const float zn = float(zBase[(uint64_t)r * zStride + idx]);
            const float gate = QW_GDN_VECTOR_DECAY ? (1.0f / (1.0f + exp(-zn))) : (zn / (1.0f + exp(-zn)));
            const float gated = (o * rms * float(w[c])) * gate;
            const half hg = simd_fwht32(half(gated), c & 31u);
            o_out[(uint64_t)r * QW_GDN_Z_CH + idx] = hg;
        }
        if (r + 1u < R) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    #pragma unroll
    for (uint j = 0u; j < QW_KC; ++j) {
        S[(k0 + j) * 128u + c] = sreg[j];
    }
}

// Undo R normalized GDN rank-1 updates in reverse order (r = R - 1 down to 0) in registers.
kernel void qwg_gdn_rollback_rows(
        device const half*  y          [[buffer(0)]],   // [R][QKV_CH]
        device const float* innovation [[buffer(1)]],   // [R][Z_CH]
        device const float* glog       [[buffer(2)]],   // [R][V_HEADS] or [R][Z_CH]
        device float*       state      [[buffer(3)]],   // [V_HEADS][128][128]
        constant uint&      R          [[buffer(4)]],
        uint h                         [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]]) {
    const uint c = tid & 127u;
    const uint kChunk = tid >> 7u;
    const uint k0 = kChunk * 16u;
    const uint qCh = QW_GDN_QK_HEADS * 128u;
    device float* S = state + (uint64_t)h * 16384u;
    float sreg[16];
    #pragma unroll
    for (uint j = 0u; j < 16u; ++j) {
        sreg[j] = S[(k0 + j) * 128u + c];
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
        S[(k0 + j) * 128u + c] = sreg[j];
    }
}

// Fused low-rank KDA gates: f_b_proj(f_low) + dt_bias -> glog (stored as exp(g), the per-channel
// decay, so recurrences skip the exp), g_b_proj(g_low) -> zOut, and sigmoid(b_raw) -> betas.
// Grid: (QW_GDN_V_HEADS, T, 1) x 128 threads/TG.
kernel void qwg_kda_gates(
        device const half*  fgb            [[buffer(0)]],   // [T][256 + QW_GDN_V_HEADS]
        device const half*  fB             [[buffer(1)]],   // [QW_GDN_Z_CH][128]
        device const half*  gB             [[buffer(2)]],   // [QW_GDN_Z_CH][128]
        device const float* aLog           [[buffer(3)]],   // [QW_GDN_V_HEADS]
        device const float* dtBias         [[buffer(4)]],   // [QW_GDN_Z_CH]
        device float*       betas          [[buffer(5)]],   // [T][QW_GDN_V_HEADS]
        device float*       glog           [[buffer(6)]],   // [T][QW_GDN_Z_CH]
        device half*        zOut           [[buffer(7)]],   // [T][QW_GDN_Z_CH]
        constant float&     gateLowerBound [[buffer(8)]],
        constant uint&      T              [[buffer(9)]],
        uint2 tg  [[threadgroup_position_in_grid]],
        uint  tid [[thread_index_in_threadgroup]]) {
    const uint h = tg.x;
    const uint t = tg.y;
    if (h >= QW_GDN_V_HEADS || t >= T || tid >= 128u) return;

    const uint fgbStride = 256u + QW_GDN_V_HEADS;
    device const half* row = fgb + (size_t)t * fgbStride;

    threadgroup float sF[128];
    threadgroup float sG[128];
    threadgroup float sDecayRate;

    sF[tid] = float(row[tid]);
    sG[tid] = float(row[128u + tid]);
    if (tid == 0u) {
        const float bVal = float(row[256u + h]);
        betas[(size_t)t * QW_GDN_V_HEADS + h] = 1.0f / (1.0f + exp(-bVal));
        sDecayRate = exp(aLog[h]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint c = h * 128u + tid;
    device const half4* fRow4 = (device const half4*)(fB + (size_t)c * 128u);
    device const half4* gRow4 = (device const half4*)(gB + (size_t)c * 128u);
    threadgroup const float4* sF4 = (threadgroup const float4*)sF;
    threadgroup const float4* sG4 = (threadgroup const float4*)sG;

    float fAcc = 0.0f;
    float gAcc = 0.0f;
    #pragma unroll
    for (uint q = 0u; q < 32u; ++q) {
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


// Prefill variant of qwg_kda_gates on simdgroup MMA: one threadgroup computes 32 tokens x one
// head (128 channels) for both expansions, so each f_b/g_b head slice is read once per 32 tokens
// instead of once per token. Same epilogue as the scalar kernel.
// Grid: (QW_GDN_V_HEADS, (T + 31) / 32, 1) x 128 threads/TG.
#define KDA_MMA_TM 32u
#define KDA_MMA_AS 136u

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
        uint2 tg   [[threadgroup_position_in_grid]],
        uint  tid  [[thread_index_in_threadgroup]],
        uint  sg   [[simdgroup_index_in_threadgroup]],
        uint  lane [[thread_index_in_simdgroup]]) {
    const uint h = tg.x;
    const uint tBase = tg.y * KDA_MMA_TM;
    if (h >= QW_GDN_V_HEADS || tBase >= T) return;
    const uint fgbStride = 256u + QW_GDN_V_HEADS;

    threadgroup half aF[KDA_MMA_TM * KDA_MMA_AS];
    threadgroup half aG[KDA_MMA_TM * KDA_MMA_AS];
    for (uint i = tid; i < KDA_MMA_TM * 256u; i += 128u) {
        const uint r = i >> 8u, k = i & 255u;
        const half v = (tBase + r < T) ? fgb[(size_t)(tBase + r) * fgbStride + k] : half(0.0h);
        if (k < 128u) aF[r * KDA_MMA_AS + k] = v; else aG[r * KDA_MMA_AS + (k - 128u)] = v;
    }
    if (tid < KDA_MMA_TM && tBase + tid < T) {
        const float bVal = float(fgb[(size_t)(tBase + tid) * fgbStride + 256u + h]);
        betas[(size_t)(tBase + tid) * QW_GDN_V_HEADS + h] = 1.0f / (1.0f + exp(-bVal));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float decayRate = exp(aLog[h]);
    const uint t = tBase + sg * 8u + qw_kda_row(lane);
    const bool tValid = t < T;
    for (uint m = 0u; m < 2u; ++m) {
        threadgroup const half* a = (m == 0u ? aF : aG) + sg * 8u * KDA_MMA_AS;
        device const half* W = (m == 0u ? fB : gB) + (size_t)h * 128u * 128u;
        simdgroup_float8x8 acc[16];
        #pragma unroll
        for (uint j = 0u; j < 16u; ++j) acc[j] = simdgroup_float8x8(0.0f);
        for (uint kk = 0u; kk < 128u; kk += 8u) {
            simdgroup_half8x8 af;
            simdgroup_load(af, a + kk, KDA_MMA_AS);
            #pragma unroll
            for (uint j = 0u; j < 16u; ++j) {
                simdgroup_half8x8 bf;
                simdgroup_load(bf, W + (size_t)j * 8u * 128u + kk, 128u, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc[j], af, bf, acc[j]);
            }
        }
        if (!tValid) continue;
        #pragma unroll
        for (uint j = 0u; j < 16u; ++j) {
            auto e = acc[j].thread_elements();
            const uint c0 = h * 128u + j * 8u + qw_kda_col(lane);
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
}
