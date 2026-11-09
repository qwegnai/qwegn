// qwg_attn_prefill.metal — C08: causal GQA flash attention, f16 MMA, fp32 softmax.
//
// Work (spec 08): per (query-tile, head): S = (q̂·k̂ᵀ)/√256 → causal-masked
// Design note: docs/KERNEL_NOTES.md#qwattn.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "include/qw_geometry_constants.metalh"
#include "include/qw_dequant.metalh"
using namespace metal;

// measured M5 Max fragment map (probe; qw_attn_probe_tests re-proves it every
// CI run — these two lines must match the probe's qw_row/qw_col).
static inline uint qw_row(uint l) { return ((l >> 4) & 1u) * 4u + ((l >> 2) & 1u) * 2u + ((l >> 1) & 1u); }
static inline uint qw_col(uint l) { return ((l >> 3) & 1u) * 4u + (l & 1u) * 2u; }

#define QW_DT   32                          // dim-tiles: head_dim 256 / 8
#define QW_TN   16                          // keys per softmax/online block
#define QW_NT   2                           // key-tiles (8 keys) per block
#define QW_SB   64                          // keys per superblock
#define QW_NBB  (QW_SB / QW_TN)             // blocks per superblock = 4
#define QW_DTP  (256u / 2u / 8u)            // PV dim-frags per pass = 16
#define QW_SBC  (QW_HEAD_DIM / 2u)          // PV columns per pass = 128
#define QW_KVROW (QW_KV_HEADS * QW_HEAD_DIM)   // k/v row stride: 1024

kernel void qwg_attn_prefill(const device half* qgate [[buffer(0)]],  // [S][12288]
                             const device half* kbuf  [[buffer(1)]],  // [S][1024]
                             const device half* vbuf  [[buffer(2)]],  // [S][1024]
                             device half* outp        [[buffer(3)]],   // [S][6144]
                             constant uint& S      [[buffer(4)]],      // tokens
                             uint tid   [[thread_index_in_threadgroup]],
                             uint sgi   [[simdgroup_index_in_threadgroup]],
                             uint3 tg   [[threadgroup_position_in_grid]]) {  // (tile·24+h)
    const uint lane = tid & 31u;
    const uint rl = qw_row(lane), cl = qw_col(lane);
    const uint head = tg.x % QW_Q_HEADS;
    const uint tile0 = (tg.x / QW_Q_HEADS) * 32u;   // grid.x = tiles×heads
    const uint kvh = head / QW_GQA_RATIO;            // GQA: h/6
    const uint rowBase = tile0 + sgi * 8u;           // this strip's first row
    const uint r = rowBase + rl;                     // this lane's row

    const uint kEnd = min(tile0 + 32u, S);           // causal bound
    const float scale = 0.0625f;                     // 1/√256 exact

    const device half* kdev = kbuf + (uint64_t)kvh * QW_HEAD_DIM;
    const device half* vdev = vbuf + (uint64_t)kvh * QW_HEAD_DIM;

    threadgroup half  pbuf[32u * QW_SB];             // 4 KiB  staged P̂ [row][key]
    threadgroup float corrFrag[QW_NBB][4u][32u];     // 2 KiB  per-block rescale
    threadgroup half  obuf[32u * 256u];              // 16 KiB epilogue reorder
    const uint rr = sgi * 8u + rl;                   // strip row within [0,32)

    // ── dim-half passes (v5.1): oacc[16] accumulates over the WHOLE key
    // axis per pass; m/lsum restart identically each pass so P̂/corr replay
    // value-identically. Both passes are full loops ⇒ every barrier below
    // hits every thread the same number of times.
    for (uint pass = 0; pass < 2u; pass++) {
        simdgroup_float8x8 oacc[QW_DTP];
        for (uint d = 0; d < QW_DTP; d++) oacc[d].thread_elements() = {};
        float m = -1e30f, lsum = 0.0f;

        for (uint a0 = 0; a0 < kEnd; a0 += QW_SB) {  // ── superblock
            const uint a1 = min(a0 + QW_SB, kEnd);

            // ── phase A: S̃ GEMM + online softmax; qf lives only here (dead
            // at the barrier entering B; reload = L2 hit). Collective loads
            // cannot clamp per-row → tail rows past S read the caller's ZERO
            // PADDING (buffers carry +pad rows; padded rows give S̃=0, are
            // masked by causality; their O is masked at store) — proven by
            // the tile-tail fixtures S=7/S=512.
            simdgroup_half8x8 qf[QW_DT], pf[QW_NT];
            simdgroup_float8x8 sacc[QW_NT];
            const device half* qbase = qgate + (uint64_t)rowBase * QW_QPROJ_OUT
                                       + head * (2u * QW_HEAD_DIM);
            for (uint d = 0; d < QW_DT; d++)
                simdgroup_load(qf[d], qbase + d * 8u, QW_QPROJ_OUT, ulong2(0, 0), false);

            for (uint nb = a0; nb < a1; nb += QW_TN) {
                const uint n0 = nb;                      // first key
                const uint bi = (n0 - a0) / QW_TN;       // block in superblock
                // S̃ block GEMM (fp32 accum; key rows past S read zero-pad)
                for (uint ct = 0; ct < QW_NT; ct++) {
                    sacc[ct].thread_elements() = {};
                    const device half* kbase = kdev + (uint64_t)(n0 + ct * 8u) * QW_KVROW;
                    for (uint d = 0; d < QW_DT; d++) {
                        simdgroup_half8x8 kf;
                        simdgroup_load(kf, kbase + d * 8u, QW_KVROW, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(sacc[ct], qf[d], kf, sacc[ct]);
                    }
                }
                // causal-masked block max (masked cols EXCLUDED — probe law);
                // fold = xor 1 + xor 8 (row-mates {l,l^1,l^8,l^9}).
                float mnew = m;
                for (uint ct = 0; ct < QW_NT; ct++) {
                    auto e = sacc[ct].thread_elements();
                    const uint col0 = n0 + ct * 8u + cl;
                    for (uint i = 0; i < 2; i++)
                        if (col0 + i <= r) {
                            const float v = e[i] * scale;
                            if (v > mnew) mnew = v;
                        }
                }
                mnew = max(mnew, simd_shuffle_xor(mnew, 1u));
                mnew = max(mnew, simd_shuffle_xor(mnew, 8u));
                const bool grew = (mnew > m) && (m > -1e29f);
                const float corr = grew ? metal::exp(m - mnew) : 1.0f;
                if (grew) lsum *= corr;
                corrFrag[bi][sgi][lane] = corr;          // staged for phase B
                m = mnew;

                // P = exp(s·scale − m) masked; the l fold uses the FP32 exp
                // values; P̂ staged f16 (every (row,key) of the tile written
                // by exactly one convergent lane; masked cols stage 0.0).
                float blkSum = 0.0f;
                for (uint ct = 0; ct < QW_NT; ct++) {
                    auto e = sacc[ct].thread_elements();
                    const uint col0 = n0 + ct * 8u + cl;
                    float t0 = 0.0f;
                    for (uint i = 0; i < 2; i++) {
                        const bool on = (col0 + i <= r);
                        const float x = on ? metal::exp(e[i] * scale - m) : 0.0f;
                        pbuf[rr * QW_SB + (n0 - a0) + ct * 8u + cl + i] = half(x);
                        t0 += x;
                    }
                    t0 += simd_shuffle_xor(t0, 1u);      // row-mate fold
                    t0 += simd_shuffle_xor(t0, 8u);
                    blkSum += t0;
                }
                lsum += blkSum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);   // P̂+corr visible

            // ── phase B: PV replay over this pass's dim half, (corr→MMA)
            // per block ascending — bit-equal per-dim chain to v1's single
            // pass. oacc[16] is the ONLY fp32 live accumulator set here.
            for (uint nb = a0; nb < a1; nb += QW_TN) {
                const uint n0 = nb;
                const uint bi = (n0 - a0) / QW_TN;
                const float corr = corrFrag[bi][sgi][lane];
                if (corr != 1.0f) {                    // grew-gate replay
                    for (uint d = 0; d < QW_DTP; d++) {
                        auto oe = oacc[d].thread_elements();
                        for (uint i = 0; i < 2; i++) oe[i] *= corr;
                        oacc[d].thread_elements() = oe;
                    }
                }
                for (uint ct = 0; ct < QW_NT; ct++) {
                    simdgroup_load(pf[ct], pbuf + (sgi * 8u) * QW_SB + (n0 - a0) + ct * 8u,
                                   QW_SB, ulong2(0, 0), false);
                    const device half* vbase = vdev + (uint64_t)(n0 + ct * 8u) * QW_KVROW
                                               + pass * QW_SBC;
                    for (uint d = 0; d < QW_DTP; d++) {
                        simdgroup_half8x8 vf;
                        simdgroup_load(vf, vbase + d * 8u, QW_KVROW);
                        simdgroup_multiply_accumulate(oacc[d], pf[ct], vf, oacc[d]);
                    }
                }
            }
        }

        // ── epilogue (ONCE per pass): O = (O/l) ⊙ σ(gate) (HF :818), manual
        // measured-map store into obuf, single legal barrier, coalesced copy
        // to device. l ≥ 1 (diagonal always unmasked) ⇒ no zero-div. Tail
        // rows hold clamp-read garbage but are never copied out.
        for (uint d = 0; d < QW_DTP; d++) {
            auto oe = oacc[d].thread_elements();
            const uint col0 = pass * QW_SBC + d * 8u + cl;
            const uint rg = min(r, S - 1u);                      // pad rows clamp
            for (uint i = 0; i < 2; i++) {
                const uint c = col0 + i;
                const float o = oe[i] / lsum;
                const float g = float(qgate[(uint64_t)rg * QW_QPROJ_OUT
                                            + head * (2u * QW_HEAD_DIM) + QW_HEAD_DIM + c]);
                obuf[rr * 256u + c] = half(o * (1.0f / (1.0f + metal::exp(-g))));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 32u * QW_SBC; i += 128u) {
            const uint row = i >> 7, r2 = tile0 + row;
            const uint cc = (i & 127u) + pass * QW_SBC;
            const half rot = simd_fwht32(obuf[row * 256u + cc], lane);
            if (r2 < S)
                outp[(uint64_t)r2 * (QW_Q_HEADS * QW_HEAD_DIM)
                     + head * QW_HEAD_DIM + cc] = rot;
        }
        // terminal barrier: keep the workgroup lockstep so the next pass's
        // phase A (pbuf writes) cannot outrun this pass's copy-out reads.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
