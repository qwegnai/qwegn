#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
#include "include/qw_dequant.metalh"   // the fold law (C05 winner body, verbatim)
#include "qw_dequant_multi.metalh"     // the size-preserving family's blocks

// qwhead.metal — C15: LM-head GEMV (split-K raced vs C05 plain) + GPU
// argmax-key kernel for the greedy path. lm_head [V, 5120] = 2.54 GB BF16 /
// ~710 MB Q4 → 1.16 ms BW floor at 614 GB/s.
//
// WHY split-K exists but may lose (the spec race, measured not assumed):
// N=248320 rows is already 248320/48SM ≈ 5170 threadgroups deep — the GPU
// is full without splitting. Split-K earns its partials traffic ONLY at
// small N (draft/vocab-chunk decode); bench races S ∈ {1,4} at the real N.
//   qwg_head_gemv_split  grid = nBand × S; group range [G·s/S, G·(s+1)/S)
//     per split; S=1 writes y directly (zero partial traffic); fold body =
//     qwg_fold_q4g_group EXACTLY (constant x/gs — C05 ADR-001 winner
//     reused, never silently re-raced); 8 rows × 32-lane simdgroups = 256
//     lanes/TG.
//   qwg_head_combine     S>1 only: y[row] = Σ_s partials[s][row].
//
// GREEDY PATH (≤80 µs gate): qwg_head_argmax — ONE coalesced pass, key
// fold = u64 (sortable(v) << 32 | ~id) under atomic_max ⇒ value DESC with
// ties → LOWEST id = the numpy/torch argmax golden law exactly (the 10k-
// position parity gate). KEY LAW: sortable(v) = v ≥ 0 ? bits(v) : ~bits(v)
// (monotone u32); decode value = s ≥ 0x80000000 ? bits(s) : bits(~s); id
// = ~low.
//
// FULL SAMPLER (top-k/top-p/Gumbel-free inverse-cdf) runs CPU-SIDE (spec
// line 9: "CPU or GPU — RACED"; Apple Silicon UMA = storageModeShared is
// CPU-mapped, the logits read is zero-copy, no PCIe readback tax). Swift
// Philox-4x32-10 (golden-pinned, deterministic per (seed, call#) — the
// C16 speculative-verify contract) + the torch-verified nucleus tie law.
// A GPU sampler kernel with the band-top-64 sort needs 248320/256 × 64 ×
// 8 B = 495 KB of device sort (×4 over the 32 KiB threadgroup cap per
// band-group) — multi-pass device bitonic is a bench-time optimization
// after the CPU/GPU race prints its verdict, NOT a build-time guess.
//
// Buffers keep the C05 GEMV contract EXACTLY: 0 nibbles | 1 x const f16 |
// 2 gs const f32 | 3 y f32 | 4 N | 5 K | 6 scales | 7 mins; split adds
// 8 S | 9 nBand | 10 partials.

#define QW_HEAD_C 8u      // rows per GEMV threadgroup

// Monotone float→u32 (IEEE order-preserving): flip everything for negative
// (sign set), set the high bit for non-negative. A naive `v>=0?b:~b` puts
// negatives INSIDE the positive range inverted — measured, it ranked
static inline uint qw_sortable(float v) {
    const uint b = as_type<uint>(v);
    return (b & 0x80000000u) ? ~b : (b ^ 0x80000000u);
}

kernel void qwg_head_gemv_split(
        device const Block_HQ4_G32* blocks   [[buffer(0)]],
        constant const half*        x        [[buffer(1)]],
        device float*               y        [[buffer(3)]],
        constant uint&              N        [[buffer(4)]],
        constant uint&              K        [[buffer(5)]],
        constant uint&              S        [[buffer(8)]],   // splits (1 = direct write)
        constant uint&              nBand    [[buffer(9)]],
        device float*               partials [[buffer(10)]],  // [S][N] (S>1 only)
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    (void)nBand;
    const uint s = tg % S, band = tg / S;
    const uint row = band * QW_HEAD_C + (ti >> 5u);
    if (row >= N) return;
    const uint G = K >> 5u;
    const uint gA = (G * s) / S, gB = (G * (s + 1u)) / S;   // group range
    device const Block_HQ4_G32* rBlocks = blocks + (size_t)row * G;
    const uint lane = ti & 31u;
    float acc = 0.0f;
    for (uint g = lane; g < gB; g += 32u) {
        if (g >= gA) {
            acc += qwg_dot_hq4_block(rBlocks[g], x + (g << 5u));
        }
    }
    float4 v = float4(acc, 0.0f, 0.0f, 0.0f);
    v += simd_shuffle_down(v, 16u); v += simd_shuffle_down(v, 8u);
    v += simd_shuffle_down(v, 4u);  v += simd_shuffle_down(v, 2u);
    v += simd_shuffle_down(v, 1u);
    if (lane == 0u) {
        if (S == 1u) y[row] = v.x;
        else partials[(size_t)s * N + row] = v.x;
    }
}

kernel void qwg_head_gemv_q4g_argmax(
        device const Block_HQ4_G32* blocks [[buffer(0)]],
        device const half*          x      [[buffer(1)]],
        device float*               y      [[buffer(3)]],
        constant uint&              N      [[buffer(4)]],
        constant uint&              K      [[buffer(5)]],
        device uint*                keysHi [[buffer(6)]],
        device uint*                keysLo [[buffer(7)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    const uint sg = ti >> 5u;
    const uint lane = ti & 31u;
    threadgroup uint sh[QW_HEAD_C], sl[QW_HEAD_C];
    const bool skipY = (N & 0x40000000u) != 0u;
    if ((N & 0x80000000u) != 0u) {
        const uint realN = N & 0x3FFFFFFFu;
        const uint row0 = (tg * QW_HEAD_C + sg) * 4u;
        float4 vx = (row0 < realN)
            ? qwg_gemv_hq4_rows4(blocks, x, row0, realN, K, lane)
            : float4(-INFINITY);
        if (lane == 0u) {
            float vals[4] = { vx.x, vx.y, vx.z, vx.w };
            uint bh = qw_sortable(-INFINITY), bl = 0u;
            #pragma unroll
            for (uint r = 0u; r < 4u; r++) {
                const uint row = row0 + r;
                if (row < realN) {
                    const float v = vals[r];
                    if (!skipY) y[row] = v;
                    const uint kh = qw_sortable(v);
                    const uint kl = ~row;
                    if (kh > bh || (kh == bh && kl > bl)) {
                        bh = kh;
                        bl = kl;
                    }
                }
            }
            sh[sg] = bh;
            sl[sg] = bl;
        }
    } else {
        const uint realN = N & 0x3FFFFFFFu;
        const uint row = tg * QW_HEAD_C + sg;
        float vx = -INFINITY;
        if (row < realN) {
            vx = qwg_gemv_hq4_row(blocks, x, row, K, lane);
        }
        if (lane == 0u) {
            if (row < realN && !skipY) {
                y[row] = vx;
            }
            sh[sg] = qw_sortable(vx);
            sl[sg] = (row < realN) ? ~row : 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ti == 0u) {
        uint bh = sh[0], bl = sl[0];
        for (uint q = 1u; q < QW_HEAD_C; q++) {
            if (sh[q] > bh || (sh[q] == bh && sl[q] > bl)) {
                bh = sh[q];
                bl = sl[q];
            }
        }
        keysHi[tg] = bh;
        keysLo[tg] = bl;
    }
}

kernel void qwg_mtp_head_gemv_q4g_argmax(
        device const Block_HQ4_G32* blocks      [[buffer(0)]],
        device const half*          x           [[buffer(1)]],
        device float*               y           [[buffer(3)]],
        constant uint&              N           [[buffer(4)]],
        constant uint&              K           [[buffer(5)]],
        device uint*                keysHi      [[buffer(6)]],
        device uint*                keysLo      [[buffer(7)]],
        constant uint&              writeLogits [[buffer(8)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    const uint sg = ti >> 5u;
    const uint lane = ti & 31u;
    const uint row = tg * QW_HEAD_C + sg;
    float vx = -INFINITY;
    if (row < N) {
        vx = qwg_gemv_hq4_row(blocks, x, row, K, lane);
    }
    threadgroup uint sh[QW_HEAD_C], sl[QW_HEAD_C];
    if (lane == 0u) {
        if (writeLogits != 0u && row < N) {
            y[row] = vx;
        }
        const uint raw = as_type<uint>(vx);
        const bool invalid = row < N && (raw & 0x7f800000u) == 0x7f800000u;
        const uint bits = (raw & 0x7fffffffu) == 0u ? 0u : raw;
        sh[sg] = invalid ? 0xffffffffu : ((bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u));
        sl[sg] = (invalid || row >= N) ? 0u : ~row;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ti == 0u) {
        uint bh = sh[0], bl = sl[0];
        for (uint q = 1u; q < QW_HEAD_C; q++) {
            if (sh[q] > bh || (sh[q] == bh && sl[q] > bl)) {
                bh = sh[q];
                bl = sl[q];
            }
        }
        keysHi[tg] = bh;
        keysLo[tg] = bl;
    }
}

kernel void qwg_head_combine(
        device const float* partials [[buffer(0)]],   // [S][N]
        device float* y               [[buffer(1)]],
        constant uint& N              [[buffer(2)]],
        constant uint& S              [[buffer(3)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint row = tg.x * 256u + tid;
    if (row >= N) return;
    float acc = 0.0f;
    for (uint s = 0u; s < S; s++) acc += partials[(size_t)s * N + row];
    y[row] = acc;
}

// ── greedy argmax (the T=0 path: ties → lowest id, the golden law) ───────
// NO 64-BIT ATOMICS: this toolchain supports none (measured: atomic_ulong
// load/max/CAS all rejected by metal 4.0 compiler — recorded law).
kernel void qwg_head_argmax(
        device const float* logits [[buffer(0)]],
        device uint*  keysHi       [[buffer(1)]],    // [nTG]
        device uint*  keysLo       [[buffer(2)]],    // [nTG]
        constant uint& N           [[buffer(3)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint i = tg.x * 256u + tid;
    const float v = (i < N) ? logits[i] : -INFINITY;
    const uint hi = qw_sortable(v);
    const uint lo = ~i;
    const uint maxHi = simd_max(hi);
    const uint maxLo = simd_max((hi == maxHi) ? lo : 0u);
    threadgroup uint sh[8], sl[8];
    if ((tid & 31u) == 0u) { sh[tid >> 5u] = maxHi; sl[tid >> 5u] = maxLo; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint bh = sh[0], bl = sl[0];               // ties across simdgroups:
        for (uint q = 1u; q < 8u; q++) {          // compare lo too
            if (sh[q] > bh || (sh[q] == bh && sl[q] > bl)) { bh = sh[q]; bl = sl[q]; }
        }
        keysHi[tg.x] = bh; keysLo[tg.x] = bl;
    }
}

kernel void qwg_head_argmax_fin(
        device const uint* keysHi [[buffer(0)]],
        device const uint* keysLo [[buffer(1)]],
        device uint* token        [[buffer(2)]],
        constant uint& nTG        [[buffer(3)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint row = tg.y;
    device const uint* rHi = keysHi + (size_t)row * nTG;
    device const uint* rLo = keysLo + (size_t)row * nTG;
    uint bh = 0u, bl = 0u;
    for (uint q = tid; q < nTG; q += 1024u) {
        const uint h = rHi[q], l = rLo[q];
        if (h > bh || (h == bh && l > bl)) { bh = h; bl = l; }
    }
    const uint maxHi = simd_max(bh);
    const uint maxLo = simd_max((bh == maxHi) ? bl : 0u);
    threadgroup uint sh[32], sl[32];
    if ((tid & 31u) == 0u) { sh[tid >> 5u] = maxHi; sl[tid >> 5u] = maxLo; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint bh2 = sh[0], bl2 = sl[0];
        // ALL 32 simdgroups own stride slots (the stride is per-lane over
        // the whole 1024, not per round) — fold every leader pair.
        for (uint q = 1u; q < 32u; q++) {
            if (sh[q] > bh2 || (sh[q] == bh2 && sl[q] > bl2)) { bh2 = sh[q]; bl2 = sl[q]; }
        }
        token[row] = ~bl2;
    }
}

// ── variational relaxed acceptance argmax-fin (arXiv:2610.26693 Eq. 7) ──────
// Draft rows accept draftTok[row+1] when its logit >= max - margin and, for a finite
// minLogProb, log softmax(draft) >= minLogProb (LSE pass only for margin-passing non-argmax drafts).
kernel void qwg_head_argmax_fin_variational(
        device const uint*  keysHi      [[buffer(0)]],
        device const uint*  keysLo      [[buffer(1)]],
        device uint*        token       [[buffer(2)]],
        constant uint&      nTG         [[buffer(3)]],
        device const float* logits      [[buffer(4)]],
        device const uint*  draftTok    [[buffer(5)]],
        constant uint&      vocab       [[buffer(6)]],
        constant uint&      totalRows   [[buffer(7)]],
        constant float&     margin      [[buffer(8)]],
        constant float&     minLogProb  [[buffer(9)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint row = tg.y;
    device const uint* rHi = keysHi + (size_t)row * nTG;
    device const uint* rLo = keysLo + (size_t)row * nTG;
    uint bh = 0u, bl = 0u;
    for (uint q = tid; q < nTG; q += 1024u) {
        const uint h = rHi[q], l = rLo[q];
        if (h > bh || (h == bh && l > bl)) { bh = h; bl = l; }
    }
    const uint maxHi = simd_max(bh);
    const uint maxLo = simd_max((bh == maxHi) ? bl : 0u);
    threadgroup uint sh[32], sl[32];
    threadgroup uint sTok, sDraft;
    // Lane 0 publishes the decision to the group at the barrier.
    threadgroup uint sGate[1];
    threadgroup float sMax, sDraftLogit;
    threadgroup float sSum[32];
    if ((tid & 31u) == 0u) { sh[tid >> 5u] = maxHi; sl[tid >> 5u] = maxLo; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint bh2 = sh[0], bl2 = sl[0];
        for (uint q = 1u; q < 32u; q++) {
            if (sh[q] > bh2 || (sh[q] == bh2 && sl[q] > bl2)) { bh2 = sh[q]; bl2 = sl[q]; }
        }
        uint outTok = ~bl2;
        uint gate = 0u;
        if (margin > 0.0f && (row + 1u) < totalRows) {
            const uint dTok = draftTok[row + 1u];
            if (dTok < vocab && dTok != outTok) {
                const uint rawMax = (bh2 & 0x80000000u) ? (bh2 ^ 0x80000000u) : ~bh2;
                const float maxLogit = as_type<float>(rawMax);
                const float draftLogit = logits[(size_t)row * vocab + dTok];
                if (draftLogit >= maxLogit - margin) {
                    if (isfinite(minLogProb)) {
                        gate = 1u;
                        sDraft = dTok;
                        sMax = maxLogit;
                        sDraftLogit = draftLogit;
                    } else {
                        outTok = dTok;
                    }
                }
            }
        }
        sTok = outTok;
        sGate[0] = gate;
        if (gate == 0u) token[row] = outTok;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sGate[0] == 0u) return;
    device const float* rowLogits = logits + (size_t)row * vocab;
    const float mx = sMax;
    float part = 0.0f;
    for (uint v = tid; v < vocab; v += 1024u) part += exp(rowLogits[v] - mx);
    part = simd_sum(part);
    if ((tid & 31u) == 0u) sSum[tid >> 5u] = part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        for (uint q = 0u; q < 32u; q++) total += sSum[q];
        const float logProb = (sDraftLogit - mx) - log(total);
        token[row] = (logProb >= minLogProb) ? sDraft : sTok;
    }
}

// ── the size-preserving family's argmax head ────────────────────────────────
//
// The same epilogue as `qwg_head_gemv_q4g_argmax` — the sortable key, the
// tiebreak, the threadgroup fold — over the family's blocks: the payload is
// a continuous little-endian bit stream, the header is the group's own
// scale and bias, and there is no fold and no magic. The arithmetic is the
// per-element affine form, the same one the decode GEMV uses, so the head
// and the layers it follows read a bank in ONE law.
//
// buffers: 0 blocks | 1 x f16 | 3 y f32 | 4 N | 5 K | 6 keysHi | 7 keysLo
#define QW_HEAD_AQ_ARGMAX(BITS, GROUP, NAME)                              \
    kernel void NAME(device const char* blocks [[buffer(0)]],               \
                      device const half*   x      [[buffer(1)]],            \
                      device float*        y      [[buffer(3)]],            \
                      constant uint&       N      [[buffer(4)]],            \
                      constant uint&       K      [[buffer(5)]],            \
                      device uint*         keysHi [[buffer(6)]],            \
                      device uint*         keysLo [[buffer(7)]],            \
                      uint tg [[threadgroup_position_in_grid]],              \
                      uint ti [[thread_index_in_threadgroup]]) {             \
        const uint sg = ti >> 5u;                                           \
        const uint lane = ti & 31u;                                         \
        threadgroup uint sh[QW_HEAD_C], sl[QW_HEAD_C];                      \
        const bool skipY = (N & 0x40000000u) != 0u;                         \
        if ((N & 0x80000000u) != 0u) {                                      \
            const uint realN = N & 0x3FFFFFFFu;                             \
            const uint row0 = (tg * QW_HEAD_C + sg) * 4u;                   \
            float4 vx = (row0 < realN)                                      \
                ? qwg_gemv_aq_rowsN<BITS, GROUP, 4u>(blocks, x, row0, realN, \
                                                      K, lane)               \
                : float4(-INFINITY);                                        \
            if (lane == 0u) {                                               \
                float vals[4] = { vx.x, vx.y, vx.z, vx.w };                 \
                uint bh = qw_sortable(-INFINITY), bl = 0u;                  \
                _Pragma("unroll")                                          \
                for (uint r = 0u; r < 4u; r++) {                            \
                    const uint row = row0 + r;                              \
                    if (row < realN) {                                      \
                        const float v = vals[r];                            \
                        if (!skipY) y[row] = v;                             \
                        const uint kh = qw_sortable(v);                     \
                        const uint kl = ~row;                               \
                        if (kh > bh || (kh == bh && kl > bl)) {             \
                            bh = kh;                                        \
                            bl = kl;                                        \
                        }                                                   \
                    }                                                       \
                }                                                           \
                sh[sg] = bh;                                                \
                sl[sg] = bl;                                                \
            }                                                               \
        } else {                                                            \
            const uint realN = N & 0x3FFFFFFFu;                             \
            const uint row = tg * QW_HEAD_C + sg;                           \
            float vx = -INFINITY;                                           \
            if (row < realN) {                                              \
                vx = qwg_gemv_aq_rowsN<BITS, GROUP, 1u>(blocks, x, row,     \
                                                        realN, K, lane).x;   \
            }                                                               \
            if (lane == 0u) {                                               \
                if (row < realN && !skipY) {                                \
                    y[row] = vx;                                            \
                }                                                           \
                sh[sg] = qw_sortable(vx);                                   \
                sl[sg] = (row < realN) ? ~row : 0u;                         \
            }                                                               \
        }                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
        if (ti == 0u) {                                                     \
            uint bh = sh[0], bl = sl[0];                                    \
            for (uint q = 1u; q < QW_HEAD_C; q++) {                         \
                if (sh[q] > bh || (sh[q] == bh && sl[q] > bl)) {            \
                    bh = sh[q];                                             \
                    bl = sl[q];                                             \
                }                                                           \
            }                                                               \
            keysHi[tg] = bh;                                                \
            keysLo[tg] = bl;                                                \
        }                                                                   \
    }

QW_HEAD_AQ_ARGMAX(2, 32, qwg_head_gemv_aq2_g32_argmax)
QW_HEAD_AQ_ARGMAX(2, 64, qwg_head_gemv_aq2_g64_argmax)
QW_HEAD_AQ_ARGMAX(2, 128, qwg_head_gemv_aq2_g128_argmax)
QW_HEAD_AQ_ARGMAX(2, 256, qwg_head_gemv_aq2_g256_argmax)
QW_HEAD_AQ_ARGMAX(3, 32, qwg_head_gemv_aq3_g32_argmax)
QW_HEAD_AQ_ARGMAX(3, 64, qwg_head_gemv_aq3_g64_argmax)
QW_HEAD_AQ_ARGMAX(3, 128, qwg_head_gemv_aq3_g128_argmax)
QW_HEAD_AQ_ARGMAX(3, 256, qwg_head_gemv_aq3_g256_argmax)
QW_HEAD_AQ_ARGMAX(4, 32, qwg_head_gemv_aq4_g32_argmax)
QW_HEAD_AQ_ARGMAX(4, 64, qwg_head_gemv_aq4_g64_argmax)
QW_HEAD_AQ_ARGMAX(4, 128, qwg_head_gemv_aq4_g128_argmax)
QW_HEAD_AQ_ARGMAX(4, 256, qwg_head_gemv_aq4_g256_argmax)
QW_HEAD_AQ_ARGMAX(6, 32, qwg_head_gemv_aq6_g32_argmax)
QW_HEAD_AQ_ARGMAX(6, 64, qwg_head_gemv_aq6_g64_argmax)
QW_HEAD_AQ_ARGMAX(6, 128, qwg_head_gemv_aq6_g128_argmax)
QW_HEAD_AQ_ARGMAX(6, 256, qwg_head_gemv_aq6_g256_argmax)
QW_HEAD_AQ_ARGMAX(8, 32, qwg_head_gemv_aq8_g32_argmax)
QW_HEAD_AQ_ARGMAX(8, 64, qwg_head_gemv_aq8_g64_argmax)
QW_HEAD_AQ_ARGMAX(8, 128, qwg_head_gemv_aq8_g128_argmax)
QW_HEAD_AQ_ARGMAX(8, 256, qwg_head_gemv_aq8_g256_argmax)
