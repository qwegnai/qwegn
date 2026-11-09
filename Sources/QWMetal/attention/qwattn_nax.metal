// qwg_attn_prefill v5.1 (Sources/QWMetal/qwattn.metal):
//   qgate [S][12288] f16 = [q̂(256)‖gate(256)]×24 (norm+rope applied upstream;
//     gate read ONLY in the epilogue: out = (O/l)·σ(gate), NOT silu)
// Design note: docs/KERNEL_NOTES.md#qwattn_nax.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "qw_geometry_constants.metalh"
#include "qw_dequant.metalh"
using namespace metal;
using namespace mlx::steel;

#define QW_KU       16u                          // NAX frag edge
#define QW_BK       32u                          // keys per online-softmax block
#define QW_KVROW    (QW_KV_HEADS * QW_HEAD_DIM)  // 1024
#define QW_NAX_HDIM 256u                         // NAX tile head_dim = 16 frags × 16

kernel void qwg_attn_prefill_nax(const device half* qgate [[buffer(0)]],
                                 const device half* kbuf  [[buffer(1)]],
                                 const device half* vbuf  [[buffer(2)]],
                                 device half* outp        [[buffer(3)]],
                                 constant uint& S      [[buffer(4)]],
                                 uint tid   [[thread_index_in_threadgroup]],
                                 uint sgi   [[simdgroup_index_in_threadgroup]],
                                 uint3 tg   [[threadgroup_position_in_grid]]) {
    threadgroup half kvblk[QW_BK * QW_NAX_HDIM]; // 16 KiB staged K block
    // OPT-02: epilogue obuf ALIASES kvblk (dead after the loop-tail barrier,
    // same 8192-half size; pattern proven in qwattn_paged_nax.metal). Halves
    // TG SRAM 32->16 KiB: 1 -> 2 resident TGs/core on 32 KiB-SRAM parts.
    threadgroup half* const obuf = kvblk;
    const uint head  = tg.x % QW_Q_HEADS;
    const uint tile0 = (tg.x / QW_Q_HEADS) * 32u;
    const uint lane = tid & 31u;
    const uint kvh  = head / QW_GQA_RATIO;
    const uint rowBase = tile0 + sgi * QW_KU;      // this strip's first row
    // Threadgroup-shared walk bound so all 4 simdgroups execute identical
    // barrier counts while only sgi 0/1 perform compute.
    const uint kEnd = min(tile0 + 32u, S);

    // MLX formulation: log2e folded into the scale, exp2 below (parity
    // delta vs v5.1's exact 0.0625 + exp — see header).
    const float scale2 = 0.0625f * 1.4426950408889634f;
    constexpr short kU = 16;
    constexpr short NQ = QW_NAX_HDIM / kU;         // 16 head-dim frags

    const device half* qdev = qgate + (uint64_t)rowBase * QW_QPROJ_OUT
                              + head * (2u * QW_HEAD_DIM);
    const device half* kdev = kbuf + (uint64_t)kvh * QW_HEAD_DIM;
    const device half* vdev = vbuf + (uint64_t)kvh * QW_HEAD_DIM;

    const short2 sc = BaseNAXFrag::get_coord();    // (fn, fm)
    const short sm = sc.y, sn = sc.x;
    const short r = short(rowBase) + sm;           // this lane's frag row

    // Q frags RESIDENT across the whole k-loop (16 × 8 half = 128
    // half-regs/lane; dsplit v2 lives in the bench note).
    NAXTile<half, 1, 1> Qf[NQ];
    if (sgi < 2u) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < NQ; id++)
            Qf[id].load(qdev + id * kU, int(QW_QPROJ_OUT));
    }

    NAXTile<float, 1, 16> Otile;               // O[16 rows × 256 dim cols]
    if (sgi < 2u) {
        Otile.clear();
    }

    float mrow[2], lrow[2];
    if (sgi < 2u) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) { mrow[i] = -1e30f; lrow[i] = 0.0f; }
    }

    for (uint n0 = 0u; n0 < kEnd; n0 += QW_BK) {
        NAXTile<half, 1, 2> Pt;
        if (n0 != 0u) {
            // x64 spike #2: carry the overwrite fence at loop top so the
            // last iteration avoids one full-TG barrier.
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // Stage K block once per threadgroup.
        for (uint e = tid; e < QW_BK * QW_HEAD_DIM; e += 128u) {
            const uint kr = e / QW_HEAD_DIM, kc = e % QW_HEAD_DIM;
            kvblk[e] = kdev[(uint64_t)(n0 + kr) * QW_KVROW + kc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgi < 2u) {
            // ── S̃ = Q·Kᵀ [16×32], 16 head-dim frags, fp32 tensor acc.
            // MLX attention_nax loop shape: per head-dim frag id, load K's
            // 32 key-rows × 16 dim-cols (Kt [2,1]; frag_at(1,*) = +16 rows);
            // ONE mma (16,32,16 desc) accumulates BOTH C key-frags (N=32).
            NAXTile<float, 1, 2> Stile;
            Stile.clear();
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < NQ; id++) {
                NAXTile<half, 2, 1> Kt;
                Kt.template load<half, QW_NAX_HDIM, 1>(kvblk + id * kU);
                BaseNAXFrag::mma(Stile.frag_at(0, 0),
                                 Stile.frag_at(0, 1),
                                 Qf[id].frag_at(0, 0), metal::false_type{},
                                 Kt.frag_at(0, 0), Kt.frag_at(1, 0),
                                 metal::true_type{});
            }

            // scale (log2e folded) + causal MASK BEFORE softmax (masked cols
            // := finite_min; they fold into the row max only BELOW any real
            // score — never corrupting it once the diagonal block ran).
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < 2; ik++) {
                thread auto& fg = Stile.frag_at(0, ik);
                const short col0 = short(n0) + short(ik * kU) + sn;  // 4 cols
                STEEL_PRAGMA_UNROLL
                for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                    const short i = loc >> 2, j = loc & 3;           // row +i·8
                    const short cc = col0 + j, rr = r + i * 8;
                    fg[loc] = (cc <= rr) ? (fg[loc] * scale2) : (-1e30f);
                }
            }

            // masked row max (partial per lane, fold xor1 then xor8 = NAX
            // row-mate law, identical to the shipping kernel).
            float mnew[2];
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) mnew[i] = mrow[i];
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < 2; ik++) {
                thread auto& fg = Stile.frag_at(0, ik);
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < BaseNAXFrag::kElemRows; i++) {
                    float v = fg[i * 4 + 0] > fg[i * 4 + 1] ? fg[i * 4 + 0] : fg[i * 4 + 1];
                    v = v > fg[i * 4 + 2] ? v : fg[i * 4 + 2];
                    v = v > fg[i * 4 + 3] ? v : fg[i * 4 + 3];
                    mnew[i] = mnew[i] > v ? mnew[i] : v;
                }
            }
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                mnew[i] = max(mnew[i], simd_shuffle_xor(mnew[i], ushort(1)));
                mnew[i] = max(mnew[i], simd_shuffle_xor(mnew[i], ushort(8)));
            }

            // P̂ = exp2(s − m) staged f16 (engine storage law); l folds the
            // FP32 exps (v5.1/MLX law). Fully-masked cols give exp2(min−real)
            // → exact 0 contribution.
            float psum[2] = {0.0f, 0.0f};
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < 2; ik++) {
                thread auto& sf = Stile.frag_at(0, ik);
                thread auto& pf = Pt.frag_at(0, ik);
                STEEL_PRAGMA_UNROLL
                for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                    const short i = loc >> 2;
                    const float x = fast::exp2(sf[loc] - mnew[i]);
                    pf[loc] = half(x);
                    psum[i] += x;
                }
            }
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                psum[i] += simd_shuffle_xor(psum[i], ushort(1));
                psum[i] += simd_shuffle_xor(psum[i], ushort(8));
            }

            // grew-gate (FA2): max unchanged ⇒ corr = 1 exactly ⇒ rescale
            // skipped. Rescale happens HERE, before this block's P̂·V MMA —
            // same ordering as MLX attention_nax and v5.1's phase B replay.
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                const bool grew = (mnew[i] > mrow[i]) && (mrow[i] > -1e29f);
                const float corr = grew ? fast::exp2(mrow[i] - mnew[i]) : 1.0f;
                lrow[i] = lrow[i] * corr + psum[i];
                if (grew) {
                    STEEL_PRAGMA_UNROLL
                    for (short d = 0; d < 16; d++) {      // 16 O frags [16×16]
                        thread auto& ofg = Otile.frag_at(0, d);
                        STEEL_PRAGMA_UNROLL
                        for (short j = 0; j < 4; j++)
                            ofg[i * 4 + j] *= corr;
                    }
                }
                mrow[i] = mnew[i];
            }
        }
        // Ensure both compute simdgroups finished K use before V overwrite.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgi < 2u) {
            // ── PV: O[16×256] += P̂[16×32 keys]·V[32×256] — MLX attention_nax
            // exact shape: reduction = key-frags ik (BK/16 = 2), N = dim-pairs
            // (2 frags of 16 cols per mma). Otile frags: 0..15 → dim cols
            // frag*16. Per-mma live C pair = 16 fp32 (MLX's live-set choice).
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < short(QW_BK / kU); ik++) {         // 2 key-frags
                STEEL_PRAGMA_UNROLL
                for (short pair = 0; pair < 8; pair++) {              // 8×(2×16) dim cols
                    NAXTile<half, 1, 2> Vt;
                    Vt.load(vdev + (n0 + uint(ik * kU)) * QW_KVROW + pair * (2 * kU),
                            int(QW_KVROW));
                    BaseNAXFrag::mma(Otile.frag_at(0, 2 * pair),
                                     Otile.frag_at(0, 2 * pair + 1),
                                     Pt.frag_at(0, ik), metal::false_type{},
                                     Vt.frag_at(0, 0), Vt.frag_at(0, 1),
                                     metal::false_type{});
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgi < 2u) {
        // ── epilogue: O = (O/l)·σ(gate) (HF :818 sigmoid, NOT silu) → obuf,
        // manual measured-map store; rows past S clamp-read but never copy.
        float rcp[2];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) rcp[i] = 1.0f / lrow[i];   // l ≥ 1

        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < 16; d++) {
            const uint col0 = uint(d * kU) + uint(sn);          // dim cols 16/lane-frag
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                const short rr = r + i * 8;
                // ABSOLUTE row clamp (v5.1 min(r, S−1) law). v1's
                // min(tile0 + rr, S−1) double-offset the row → every tile
                // ≥1 applied the wrong row's σ(gate). See header for the
                // probe witness (nax_probe_dbg2: carry state was clean).
                const uint rgate = min(uint(rr), S - 1u);
                const device half* grow = qgate
                    + (uint64_t)rgate * QW_QPROJ_OUT
                    + head * (2u * QW_HEAD_DIM) + QW_HEAD_DIM;
                thread auto& fg = Otile.frag_at(0, d);
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; j++) {
                    const float o = fg[i * 4 + j] * rcp[i];
                    const float g = float(grow[col0 + uint(j)]);
                    obuf[uint(sgi * QW_KU + sm + i * 8) * QW_HEAD_DIM + col0 + uint(j)] =
                        half(o * (1.0f / (1.0f + fast::exp(-g))));
                }
            }
        }
        // ── copy-out: THIS simdgroup's 16 rows only (rows [0,16) written by
        // sgi0, [16,32) by sgi1 at obuf row sgi*16+…). Lane l's epilogue
        // stores and a different lane's copy-out loads share obuf rows ⇒
        // ONE intra-simdgroup barrier first (MLX law: tensor/simd traffic
        // synchronizes INTRA-simdgroup; sgi 2,3 never arrive anywhere, so
        // there is NO cross-threadgroup traffic and no TG barrier needed).
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < 16u * QW_HEAD_DIM; i += 32u) {
            const uint row = i >> 8, r2 = rowBase + row;
            const half rot = simd_fwht32(obuf[(sgi * QW_KU + row) * QW_HEAD_DIM + (i & 255u)], lane);
            if (r2 < S) {
                outp[(uint64_t)r2 * (QW_Q_HEADS * QW_HEAD_DIM)
                     + head * QW_HEAD_DIM + (i & 255u)] = rot;
            }
        }
    }
}


kernel void qwg_attn_prefill_nax_x64(const device half* qgate [[buffer(0)]],
                                 const device half* kbuf  [[buffer(1)]],
                                 const device half* vbuf  [[buffer(2)]],
                                 device half* outp        [[buffer(3)]],
                                 constant uint& S      [[buffer(4)]],
                                 uint tid   [[thread_index_in_threadgroup]],
                                 uint sgi   [[simdgroup_index_in_threadgroup]],
                                 uint3 tg   [[threadgroup_position_in_grid]]) {
    threadgroup half kvblk[QW_BK * QW_NAX_HDIM]; // 16 KiB staged K block
    // OPT-02: epilogue obuf ALIASES kvblk (dead after the loop-tail barrier,
    // same 8192-half size; pattern proven in qwattn_paged_nax.metal). Halves
    // TG SRAM 32->16 KiB: 1 -> 2 resident TGs/core on 32 KiB-SRAM parts.
    threadgroup half* const obuf = kvblk;

    const uint head  = tg.x % QW_Q_HEADS;
    const uint tile0 = (tg.x / QW_Q_HEADS) * 32u;
    const uint lane = tid & 31u;
    const uint kvh  = head / QW_GQA_RATIO;
    const uint rowBase = tile0 + sgi * QW_KU;      // this strip's first row
    // Threadgroup-shared walk bound so all 4 simdgroups execute identical
    // barrier counts while only sgi 0/1 perform compute.
    const uint kEnd = min(tile0 + 32u, S);

    // MLX formulation: log2e folded into the scale, exp2 below (parity
    // delta vs v5.1's exact 0.0625 + exp — see header).
    const float scale2 = 0.0625f * 1.4426950408889634f;
    constexpr short kU = 16;
    constexpr short NQ = QW_NAX_HDIM / kU;         // 16 head-dim frags

    const device half* qdev = qgate + (uint64_t)rowBase * QW_QPROJ_OUT
                              + head * (2u * QW_HEAD_DIM);
    const device half* kdev = kbuf + (uint64_t)kvh * QW_HEAD_DIM;
    const device half* vdev = vbuf + (uint64_t)kvh * QW_HEAD_DIM;

    const short2 sc = BaseNAXFrag::get_coord();    // (fn, fm)
    const short sm = sc.y, sn = sc.x;
    const short r = short(rowBase) + sm;           // this lane's frag row

    // Q frags RESIDENT across the whole k-loop (16 × 8 half = 128
    // half-regs/lane; dsplit v2 lives in the bench note).
    NAXTile<half, 1, 1> Qf[NQ];
    if (sgi < 2u) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < NQ; id++)
            Qf[id].load(qdev + id * kU, int(QW_QPROJ_OUT));
    }

    NAXTile<float, 1, 16> Otile;               // O[16 rows × 256 dim cols]
    if (sgi < 2u) {
        Otile.clear();
    }

    float mrow[2], lrow[2];
    if (sgi < 2u) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) { mrow[i] = -1e30f; lrow[i] = 0.0f; }
    }

    for (uint n0 = 0u; n0 < kEnd; n0 += QW_BK) {
        NAXTile<half, 1, 2> Pt;
        // Stage K block once per threadgroup.
        for (uint e = tid; e < QW_BK * QW_HEAD_DIM; e += 128u) {
            const uint kr = e / QW_HEAD_DIM, kc = e % QW_HEAD_DIM;
            kvblk[e] = kdev[(uint64_t)(n0 + kr) * QW_KVROW + kc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgi < 2u) {
            // ── S̃ = Q·Kᵀ [16×32], 16 head-dim frags, fp32 tensor acc.
            // MLX attention_nax loop shape: per head-dim frag id, load K's
            // 32 key-rows × 16 dim-cols (Kt [2,1]; frag_at(1,*) = +16 rows);
            // ONE mma (16,32,16 desc) accumulates BOTH C key-frags (N=32).
            NAXTile<float, 1, 2> Stile;
            Stile.clear();
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < NQ; id++) {
                NAXTile<half, 2, 1> Kt;
                Kt.template load<half, QW_NAX_HDIM, 1>(kvblk + id * kU);
                BaseNAXFrag::mma(Stile.frag_at(0, 0),
                                 Stile.frag_at(0, 1),
                                 Qf[id].frag_at(0, 0), metal::false_type{},
                                 Kt.frag_at(0, 0), Kt.frag_at(1, 0),
                                 metal::true_type{});
            }

            // scale (log2e folded) + causal MASK BEFORE softmax (masked cols
            // := finite_min; they fold into the row max only BELOW any real
            // score — never corrupting it once the diagonal block ran).
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < 2; ik++) {
                thread auto& fg = Stile.frag_at(0, ik);
                const short col0 = short(n0) + short(ik * kU) + sn;  // 4 cols
                STEEL_PRAGMA_UNROLL
                for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                    const short i = loc >> 2, j = loc & 3;           // row +i·8
                    const short cc = col0 + j, rr = r + i * 8;
                    fg[loc] = (cc <= rr) ? (fg[loc] * scale2) : (-1e30f);
                }
            }

            // masked row max (partial per lane, fold xor1 then xor8 = NAX
            // row-mate law, identical to the shipping kernel).
            float mnew[2];
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) mnew[i] = mrow[i];
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < 2; ik++) {
                thread auto& fg = Stile.frag_at(0, ik);
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < BaseNAXFrag::kElemRows; i++) {
                    float v = fg[i * 4 + 0] > fg[i * 4 + 1] ? fg[i * 4 + 0] : fg[i * 4 + 1];
                    v = v > fg[i * 4 + 2] ? v : fg[i * 4 + 2];
                    v = v > fg[i * 4 + 3] ? v : fg[i * 4 + 3];
                    mnew[i] = mnew[i] > v ? mnew[i] : v;
                }
            }
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                mnew[i] = max(mnew[i], simd_shuffle_xor(mnew[i], ushort(1)));
                mnew[i] = max(mnew[i], simd_shuffle_xor(mnew[i], ushort(8)));
            }

            // P̂ = exp2(s − m) staged f16 (engine storage law); l folds the
            // FP32 exps (v5.1/MLX law). Fully-masked cols give exp2(min−real)
            // → exact 0 contribution.
            float psum[2] = {0.0f, 0.0f};
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < 2; ik++) {
                thread auto& sf = Stile.frag_at(0, ik);
                thread auto& pf = Pt.frag_at(0, ik);
                STEEL_PRAGMA_UNROLL
                for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                    const short i = loc >> 2;
                    const float x = fast::exp2(sf[loc] - mnew[i]);
                    pf[loc] = half(x);
                    psum[i] += x;
                }
            }
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                psum[i] += simd_shuffle_xor(psum[i], ushort(1));
                psum[i] += simd_shuffle_xor(psum[i], ushort(8));
            }

            // grew-gate (FA2): max unchanged ⇒ corr = 1 exactly ⇒ rescale
            // skipped. Rescale happens HERE, before this block's P̂·V MMA —
            // same ordering as MLX attention_nax and v5.1's phase B replay.
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                const bool grew = (mnew[i] > mrow[i]) && (mrow[i] > -1e29f);
                const float corr = grew ? fast::exp2(mrow[i] - mnew[i]) : 1.0f;
                lrow[i] = lrow[i] * corr + psum[i];
                if (grew) {
                    STEEL_PRAGMA_UNROLL
                    for (short d = 0; d < 16; d++) {      // 16 O frags [16×16]
                        thread auto& ofg = Otile.frag_at(0, d);
                        STEEL_PRAGMA_UNROLL
                        for (short j = 0; j < 4; j++)
                            ofg[i * 4 + j] *= corr;
                    }
                }
                mrow[i] = mnew[i];
            }
        }
        if (sgi < 2u) {
            // ── PV: O[16×256] += P̂[16×32 keys]·V[32×256] — MLX attention_nax
            // exact shape: reduction = key-frags ik (BK/16 = 2), N = dim-pairs
            // (2 frags of 16 cols per mma). Otile frags: 0..15 → dim cols
            // frag*16. Per-mma live C pair = 16 fp32 (MLX's live-set choice).
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < short(QW_BK / kU); ik++) {         // 2 key-frags
                STEEL_PRAGMA_UNROLL
                for (short pair = 0; pair < 8; pair++) {              // 8×(2×16) dim cols
                    NAXTile<half, 1, 2> Vt;
                    Vt.load(vdev + (n0 + uint(ik * kU)) * QW_KVROW + pair * (2 * kU),
                            int(QW_KVROW));
                    BaseNAXFrag::mma(Otile.frag_at(0, 2 * pair),
                                     Otile.frag_at(0, 2 * pair + 1),
                                     Pt.frag_at(0, ik), metal::false_type{},
                                     Vt.frag_at(0, 0), Vt.frag_at(0, 1),
                                     metal::false_type{});
                }
            }
        }
        // Every thread must reach the fence before the next K overwrite.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgi < 2u) {
        // ── epilogue: O = (O/l)·σ(gate) (HF :818 sigmoid, NOT silu) → obuf,
        // manual measured-map store; rows past S clamp-read but never copy.
        float rcp[2];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) rcp[i] = 1.0f / lrow[i];   // l ≥ 1

        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < 16; d++) {
            const uint col0 = uint(d * kU) + uint(sn);          // dim cols 16/lane-frag
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                const short rr = r + i * 8;
                // ABSOLUTE row clamp (v5.1 min(r, S−1) law). v1's
                // min(tile0 + rr, S−1) double-offset the row → every tile
                // ≥1 applied the wrong row's σ(gate). See header for the
                // probe witness (nax_probe_dbg2: carry state was clean).
                const uint rgate = min(uint(rr), S - 1u);
                const device half* grow = qgate
                    + (uint64_t)rgate * QW_QPROJ_OUT
                    + head * (2u * QW_HEAD_DIM) + QW_HEAD_DIM;
                thread auto& fg = Otile.frag_at(0, d);
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; j++) {
                    const float o = fg[i * 4 + j] * rcp[i];
                    const float g = float(grow[col0 + uint(j)]);
                    obuf[uint(sgi * QW_KU + sm + i * 8) * QW_HEAD_DIM + col0 + uint(j)] =
                        half(o * (1.0f / (1.0f + fast::exp(-g))));
                }
            }
        }
        // ── copy-out: THIS simdgroup's 16 rows only (rows [0,16) written by
        // sgi0, [16,32) by sgi1 at obuf row sgi*16+…). Lane l's epilogue
        // stores and a different lane's copy-out loads share obuf rows ⇒
        // ONE intra-simdgroup barrier first (MLX law: tensor/simd traffic
        // synchronizes INTRA-simdgroup; sgi 2,3 never arrive anywhere, so
        // there is NO cross-threadgroup traffic and no TG barrier needed).
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < 16u * QW_HEAD_DIM; i += 32u) {
            const uint row = i >> 8, r2 = rowBase + row;
            const half rot = simd_fwht32(obuf[(sgi * QW_KU + row) * QW_HEAD_DIM + (i & 255u)], lane);
            if (r2 < S) {
                outp[(uint64_t)r2 * (QW_Q_HEADS * QW_HEAD_DIM)
                     + head * QW_HEAD_DIM + (i & 255u)] = rot;
            }
        }
    }
}


// EXP-9 sub-item 2 (OPT-IN, NOT the shipping default) — fully-participating
// work distribution: ALL FOUR simdgroups own a disjoint 16-row query strip
// (local MLX-derived; inspired by the GENERIC `attention_nax` kernel in
kernel void qwg_attn_prefill_nax_quad(const device half* qgate [[buffer(0)]],
                                 const device half* kbuf  [[buffer(1)]],
                                 const device half* vbuf  [[buffer(2)]],
                                 device half* outp        [[buffer(3)]],
                                 constant uint& S      [[buffer(4)]],
                                 uint tid   [[thread_index_in_threadgroup]],
                                 uint sgi   [[simdgroup_index_in_threadgroup]],
                                 uint3 tg   [[threadgroup_position_in_grid]]) {
    threadgroup half kvblk[QW_BK * QW_NAX_HDIM]; // 16 KiB staged K block (ONLY tg memory)

    const uint head  = tg.x % QW_Q_HEADS;
    const uint tile0 = (tg.x / QW_Q_HEADS) * 64u;  // 4 strips of 16 rows = 64
    const uint kvh  = head / QW_GQA_RATIO;
    const uint rowBase = tile0 + sgi * QW_KU;      // this strip's first row (sgi 0..3, ALL active)
    const uint kEnd = min(tile0 + 64u, S);

    const float scale2 = 0.0625f * 1.4426950408889634f;
    constexpr short kU = 16;
    constexpr short NQ = QW_NAX_HDIM / kU;

    const device half* qdev = qgate + (uint64_t)rowBase * QW_QPROJ_OUT
                              + head * (2u * QW_HEAD_DIM);
    const device half* kdev = kbuf + (uint64_t)kvh * QW_HEAD_DIM;
    const device half* vdev = vbuf + (uint64_t)kvh * QW_HEAD_DIM;

    const short2 sc = BaseNAXFrag::get_coord();
    const short sm = sc.y, sn = sc.x;
    const short r = short(rowBase) + sm;

    NAXTile<half, 1, 1> Qf[NQ];
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < NQ; id++)
        Qf[id].load(qdev + id * kU, int(QW_QPROJ_OUT));

    NAXTile<float, 1, 16> Otile;
    Otile.clear();

    float mrow[2], lrow[2];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) { mrow[i] = -1e30f; lrow[i] = 0.0f; }

    for (uint n0 = 0u; n0 < kEnd; n0 += QW_BK) {
        NAXTile<half, 1, 2> Pt;
        for (uint e = tid; e < QW_BK * QW_HEAD_DIM; e += 128u) {
            const uint kr = e / QW_HEAD_DIM, kc = e % QW_HEAD_DIM;
            kvblk[e] = kdev[(uint64_t)(n0 + kr) * QW_KVROW + kc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        NAXTile<float, 1, 2> Stile;
        Stile.clear();
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < NQ; id++) {
            NAXTile<half, 2, 1> Kt;
            Kt.template load<half, QW_NAX_HDIM, 1>(kvblk + id * kU);
            BaseNAXFrag::mma(Stile.frag_at(0, 0),
                             Stile.frag_at(0, 1),
                             Qf[id].frag_at(0, 0), metal::false_type{},
                             Kt.frag_at(0, 0), Kt.frag_at(1, 0),
                             metal::true_type{});
        }

        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < 2; ik++) {
            thread auto& fg = Stile.frag_at(0, ik);
            const short col0 = short(n0) + short(ik * kU) + sn;
            STEEL_PRAGMA_UNROLL
            for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                const short i = loc >> 2, j = loc & 3;
                const short cc = col0 + j, rr = r + i * 8;
                fg[loc] = (cc <= rr) ? (fg[loc] * scale2) : (-1e30f);
            }
        }

        float mnew[2];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) mnew[i] = mrow[i];
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < 2; ik++) {
            thread auto& fg = Stile.frag_at(0, ik);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < BaseNAXFrag::kElemRows; i++) {
                float v = fg[i * 4 + 0] > fg[i * 4 + 1] ? fg[i * 4 + 0] : fg[i * 4 + 1];
                v = v > fg[i * 4 + 2] ? v : fg[i * 4 + 2];
                v = v > fg[i * 4 + 3] ? v : fg[i * 4 + 3];
                mnew[i] = mnew[i] > v ? mnew[i] : v;
            }
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            mnew[i] = max(mnew[i], simd_shuffle_xor(mnew[i], ushort(1)));
            mnew[i] = max(mnew[i], simd_shuffle_xor(mnew[i], ushort(8)));
        }

        float psum[2] = {0.0f, 0.0f};
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < 2; ik++) {
            thread auto& sf = Stile.frag_at(0, ik);
            thread auto& pf = Pt.frag_at(0, ik);
            STEEL_PRAGMA_UNROLL
            for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                const short i = loc >> 2;
                const float x = fast::exp2(sf[loc] - mnew[i]);
                pf[loc] = half(x);
                psum[i] += x;
            }
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            psum[i] += simd_shuffle_xor(psum[i], ushort(1));
            psum[i] += simd_shuffle_xor(psum[i], ushort(8));
        }

        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            const bool grew = (mnew[i] > mrow[i]) && (mrow[i] > -1e29f);
            const float corr = grew ? fast::exp2(mrow[i] - mnew[i]) : 1.0f;
            lrow[i] = lrow[i] * corr + psum[i];
            if (grew) {
                STEEL_PRAGMA_UNROLL
                for (short d = 0; d < 16; d++) {
                    thread auto& ofg = Otile.frag_at(0, d);
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; j++)
                        ofg[i * 4 + j] *= corr;
                }
            }
            mrow[i] = mnew[i];
        }
        // ALL 4 simdgroups touch kvblk this iteration (vs 2 in the shipping
        // kernel) — the barrier below is still TRUE threadgroup-wide,
        // uniformly reached by every thread (no `if (sgi<2u)` divergence).
        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < short(QW_BK / kU); ik++) {
            STEEL_PRAGMA_UNROLL
            for (short pair = 0; pair < 8; pair++) {
                NAXTile<half, 1, 2> Vt;
                Vt.load(vdev + (n0 + uint(ik * kU)) * QW_KVROW + pair * (2 * kU),
                        int(QW_KVROW));
                BaseNAXFrag::mma(Otile.frag_at(0, 2 * pair),
                                 Otile.frag_at(0, 2 * pair + 1),
                                 Pt.frag_at(0, ik), metal::false_type{},
                                 Vt.frag_at(0, 0), Vt.frag_at(0, 1),
                                 metal::false_type{});
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint lane = tid & 31u;
    float rcp[2];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) rcp[i] = 1.0f / lrow[i];

    for (uint grp = 0; grp < 2u; grp++) {
        if ((sgi >> 1) == grp) {
            const uint localSgi = sgi & 1u;
            STEEL_PRAGMA_UNROLL
            for (short d = 0; d < 16; d++) {
                const uint col0 = uint(d * kU) + uint(sn);
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; i++) {
                    const uint rg = min(uint(r + i * 8), S - 1u);
                    const device half* grow = qgate
                        + (uint64_t)rg * QW_QPROJ_OUT
                        + head * (2u * QW_HEAD_DIM) + QW_HEAD_DIM;
                    thread auto& fg = Otile.frag_at(0, d);
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; j++) {
                        const float o = fg[i * 4 + j] * rcp[i];
                        const float g = float(grow[col0 + uint(j)]);
                        kvblk[uint(localSgi * QW_KU + sm + i * 8) * QW_HEAD_DIM + col0 + uint(j)] =
                            half(o * (1.0f / (1.0f + fast::exp(-g))));
                    }
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lane; i < 16u * QW_HEAD_DIM; i += 32u) {
                const uint row = i >> 8, r2 = rowBase + row;
                const half rot = simd_fwht32(kvblk[(localSgi * QW_KU + row) * QW_HEAD_DIM + (i & 255u)], lane);
                if (r2 < S) {
                    outp[(uint64_t)r2 * (QW_Q_HEADS * QW_HEAD_DIM)
                         + head * QW_HEAD_DIM + (i & 255u)] = rot;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
