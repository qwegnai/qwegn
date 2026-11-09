#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_dequant.metalh"
using namespace metal;

// qwg_mlp.metal — C14: SwiGLU FFN, batched decode/MTP path (M ∈ 1..8) under HQ4_G32_M5.
//
// FFN = down( fwht32(silu(gate·x) · (up·x)) ), H=5120, I=17408.
// Weight layout: contiguous 20-byte Block_HQ4_G32 blocks (group_size = 32).

kernel void qwg_mlp_gu(
        device const Block_HQ4_G32* blocks [[buffer(0)]],
        constant const half*        xrow   [[buffer(3)]],   // [M][K] f16 — CONSTANT (rotated H_32)
        device half*                hrow   [[buffer(5)]],   // [M][I] f16 out
        constant uint&              I      [[buffer(6)]],   // 17408
        constant uint&              K      [[buffer(7)]],   // 5120
        constant uint&              M      [[buffer(8)]],   // ≤ 8
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint i = tg.x;                              // packed row pair id
    const uint b = tid >> 6u, side = (tid >> 5u) & 1u, f = tid & 31u;
    const uint row = side == 0u ? i : i + I;
    float vx = 0.0f;
    if (b < M) {
        constant const half* xr = xrow + (size_t)b * K;
        vx = qwg_gemv_hq4_row(blocks, xr, row, K, f);
    }
    threadgroup float part[16];                       // 2 slots per batch row
    if (f == 0u && b < M) part[b * 2u + side] = vx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (f == 0u && side == 0u && b < M) {
        const float gate = part[b * 2u], upv = part[b * 2u + 1u];
        hrow[(size_t)b * I + i] = half(qwg_swiglu_mul(gate, upv));
    }
}

kernel void qwg_mlp_down(
        device const Block_HQ4_G32* blocks [[buffer(0)]],
        device const half*          hrow   [[buffer(3)]],   // [M][I] f16 — DEVICE (rotated H_32)
        device half*                yrow   [[buffer(5)]],   // [M][N] f16
        constant uint&              N      [[buffer(6)]],   // 5120 (rows)
        constant uint&              I      [[buffer(7)]],   // 17408 (inner)
        constant uint&              M      [[buffer(8)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint row = tg.x;                            // down_proj row
    const uint lane = tid;                            // blockDim == 32
    const uint G = I >> 5u;                           // 544 inner blocks of 32
    device const Block_HQ4_G32* rBlocks = blocks + (size_t)row * G;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    const size_t Ib4 = (size_t)(I >> 2);              // row stride in half4
    for (uint g = lane; g < G; g += 32u) {
        const Block_HQ4_G32 blk = rBlocks[g];
        const half2 s2 = half2(blk.d);
        const half2 b2 = half2(blk.m);
        const size_t off = (size_t)g * 32u;
        device const half4* hb = (device const half4*)(hrow + off);
        #pragma unroll
        for (uint wi = 0u; wi < 4u; wi++) {
            half2 w01, w23, w45, w67;
            unpack_hq4_word8(blk.qs[wi], s2, b2, w01, w23, w45, w67);
            const size_t v = (size_t)(wi * 2u);
            const float n0 = float(w01.x), n1 = float(w01.y),
                        n2 = float(w23.x), n3 = float(w23.y),
                        n4 = float(w45.x), n5 = float(w45.y),
                        n6 = float(w67.x), n7 = float(w67.y);
            #define QW_DOTB(ACC, BOFF) {                       \
                float4 l = float4(hb[(BOFF) + v]);             \
                float4 u2 = float4(hb[(BOFF) + v + 1u]);       \
                ACC += n0 * l.x + n1 * l.y + n2 * l.z + n3 * l.w \
                     + n4 * u2.x + n5 * u2.y + n6 * u2.z + n7 * u2.w; \
            }
            if (M > 0u) QW_DOTB(acc0, 0u * Ib4)
            if (M > 1u) QW_DOTB(acc1, 1u * Ib4)
            if (M > 2u) QW_DOTB(acc2, 2u * Ib4)
            if (M > 3u) QW_DOTB(acc3, 3u * Ib4)
            if (M > 4u) QW_DOTB(acc4, 4u * Ib4)
            if (M > 5u) QW_DOTB(acc5, 5u * Ib4)
            if (M > 6u) QW_DOTB(acc6, 6u * Ib4)
            if (M > 7u) QW_DOTB(acc7, 7u * Ib4)
            #undef QW_DOTB
        }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1); acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3); acc4 = simd_sum(acc4); acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6); acc7 = simd_sum(acc7);
    if (lane == 0u) {
        if (M > 0u) yrow[(size_t)0u * N + row] = half(acc0);
        if (M > 1u) yrow[(size_t)1u * N + row] = half(acc1);
        if (M > 2u) yrow[(size_t)2u * N + row] = half(acc2);
        if (M > 3u) yrow[(size_t)3u * N + row] = half(acc3);
        if (M > 4u) yrow[(size_t)4u * N + row] = half(acc4);
        if (M > 5u) yrow[(size_t)5u * N + row] = half(acc5);
        if (M > 6u) yrow[(size_t)6u * N + row] = half(acc6);
        if (M > 7u) yrow[(size_t)7u * N + row] = half(acc7);
    }
}

// ── qwg_mlp_down_m1_rows4 — C14 addendum: 16 rows/TG, 4 per simdgroup.
// One h-block load serves all four rows (same QW_DOTB tree as qwg_mlp_down).
// Host routes here only for M == 1 with QW_MLP_DOWN_ROWS4=1.
kernel void qwg_mlp_down_m1_rows4(
        device const Block_HQ4_G32* blocks [[buffer(0)]],
        device const half*          hrow   [[buffer(3)]],
        device half*                yrow   [[buffer(5)]],
        constant uint&              N      [[buffer(6)]],
        constant uint&              I      [[buffer(7)]],
        constant uint&              M      [[buffer(8)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    if (M != 1u || N == 0u) return;
    const uint simd = tid >> 5u, lane = tid & 31u;
    const uint row0 = tg.x * 16u + simd * 4u;
    const uint G = I >> 5u;
    device const Block_HQ4_G32* rA = blocks + (size_t)min(row0 + 0u, N - 1u) * G;
    device const Block_HQ4_G32* rB = blocks + (size_t)min(row0 + 1u, N - 1u) * G;
    device const Block_HQ4_G32* rC = blocks + (size_t)min(row0 + 2u, N - 1u) * G;
    device const Block_HQ4_G32* rD = blocks + (size_t)min(row0 + 3u, N - 1u) * G;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint g = lane; g < G; g += 32u) {
        device const half4* hb = (device const half4*)(hrow + (size_t)g * 32u);
        const Block_HQ4_G32 b0 = rA[g], b1 = rB[g], b2 = rC[g], b3 = rD[g];
        #pragma unroll
        for (uint wi = 0u; wi < 4u; wi++) {
            const size_t v = (size_t)(wi * 2u);
            const float4 l = float4(hb[v]);
            const float4 u2 = float4(hb[v + 1u]);
            #define QW_ROWACC(ACC, BLK) {                                      \
                half2 w01, w23, w45, w67;                                       \
                unpack_hq4_word8((BLK).qs[wi], half2((BLK).d), half2((BLK).m),  \
                                 w01, w23, w45, w67);                           \
                const float n0 = float(w01.x), n1 = float(w01.y),               \
                            n2 = float(w23.x), n3 = float(w23.y),               \
                            n4 = float(w45.x), n5 = float(w45.y),               \
                            n6 = float(w67.x), n7 = float(w67.y);               \
                ACC += n0 * l.x + n1 * l.y + n2 * l.z + n3 * l.w                \
                     + n4 * u2.x + n5 * u2.y + n6 * u2.z + n7 * u2.w;           \
            }
            QW_ROWACC(acc0, b0)
            QW_ROWACC(acc1, b1)
            QW_ROWACC(acc2, b2)
            QW_ROWACC(acc3, b3)
            #undef QW_ROWACC
        }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    if (lane == 0u) {
        if (row0 + 0u < N) yrow[(size_t)row0 + 0u] = half(acc0);
        if (row0 + 1u < N) yrow[(size_t)row0 + 1u] = half(acc1);
        if (row0 + 2u < N) yrow[(size_t)row0 + 2u] = half(acc2);
        if (row0 + 3u < N) yrow[(size_t)row0 + 3u] = half(acc3);
    }
}

// ── Phase 1: SwiGLU Saliency Truncation Kernels ─────────────────────────────
kernel void qwg_mlp_gu_pruned(
        device const Block_HQ4_G32* blocks [[buffer(0)]],
        constant const half*        xrow   [[buffer(3)]],   // [M][K] f16 — CONSTANT
        device half*                hrow   [[buffer(5)]],   // [M][I] f16 out
        constant uint&              I      [[buffer(6)]],   // 17408
        constant uint&              K      [[buffer(7)]],   // 5120
        constant uint&              M      [[buffer(8)]],   // ≤ 8
        device const uint* active_channels [[buffer(9)]],   // [I_active]
        constant uint&              I_active [[buffer(10)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    if (tg.x >= I_active) return;
    const uint i = active_channels[tg.x];
    const uint b = tid >> 6u, side = (tid >> 5u) & 1u, f = tid & 31u;
    const uint G = K >> 5u;                           // 160
    const uint row = side == 0u ? i : i + I;
    float acc = 0.0f;
    if (b < M) {
        device const Block_HQ4_G32* rBlocks = blocks + (size_t)row * G;
        constant const half* xr = xrow + (size_t)b * K;
        for (uint g = f; g < G; g += 32u) {
            acc += qwg_dot_hq4_block(rBlocks[g], xr + g * 32u);
        }
    }
    float4 v = float4(acc, 0.0f, 0.0f, 0.0f);
    v += simd_shuffle_down(v, 16); v += simd_shuffle_down(v, 8);
    v += simd_shuffle_down(v, 4);  v += simd_shuffle_down(v, 2);
    v += simd_shuffle_down(v, 1);
    threadgroup float part[16];
    if (f == 0u && b < M) part[b * 2u + side] = v.x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (f == 0u && side == 0u && b < M) {
        const float gate = part[b * 2u], upv = part[b * 2u + 1u];
        hrow[(size_t)b * I + i] = half(qwg_swiglu_mul(gate, upv));
    }
}

kernel void qwg_mlp_down_pruned(
        device const Block_HQ4_G32* blocks [[buffer(0)]],
        device const half*          hrow   [[buffer(3)]],   // [M][I] f16 — DEVICE
        device half*                yrow   [[buffer(5)]],   // [M][N] f16
        constant uint&              N      [[buffer(6)]],   // 5120 (rows)
        constant uint&              I      [[buffer(7)]],   // 17408 (inner)
        constant uint&              M      [[buffer(8)]],
        device const uint* active_groups   [[buffer(9)]],   // [G_active] (128-channel super-groups)
        constant uint&              G_active [[buffer(10)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint row = tg.x;
    const uint lane = tid;
    const uint G = I >> 5u;                           // 544 inner blocks of 32
    device const Block_HQ4_G32* rBlocks = blocks + (size_t)row * G;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    const size_t Ib4 = (size_t)(I >> 2);
    const uint totalActiveBlocks = G_active * 4u;
    for (uint bi = lane; bi < totalActiveBlocks; bi += 32u) {
        const uint g = active_groups[bi >> 2u] * 4u + (bi & 3u);
        const Block_HQ4_G32 blk = rBlocks[g];
        const half d = blk.d;
        const half m = blk.m;
        const size_t off = (size_t)g * 32u;
        device const half4* hb = (device const half4*)(hrow + off);
        #pragma unroll
        for (uint wi = 0u; wi < 4u; wi++) {
            half w8[8];
            unpack_hq4_word8(blk.qs[wi], d, m, w8);
            const size_t v = (size_t)(wi * 2u);
            const float n0 = float(w8[0]), n1 = float(w8[1]),
                        n2 = float(w8[2]), n3 = float(w8[3]),
                        n4 = float(w8[4]), n5 = float(w8[5]),
                        n6 = float(w8[6]), n7 = float(w8[7]);
            #define QW_DOTB_PRUNED(ACC, BOFF) {                \
                float4 l = float4(hb[(BOFF) + v]);             \
                float4 u2 = float4(hb[(BOFF) + v + 1u]);       \
                ACC += n0 * l.x + n1 * l.y + n2 * l.z + n3 * l.w \
                     + n4 * u2.x + n5 * u2.y + n6 * u2.z + n7 * u2.w; \
            }
            if (M > 0u) QW_DOTB_PRUNED(acc0, 0u * Ib4)
            if (M > 1u) QW_DOTB_PRUNED(acc1, 1u * Ib4)
            if (M > 2u) QW_DOTB_PRUNED(acc2, 2u * Ib4)
            if (M > 3u) QW_DOTB_PRUNED(acc3, 3u * Ib4)
            if (M > 4u) QW_DOTB_PRUNED(acc4, 4u * Ib4)
            if (M > 5u) QW_DOTB_PRUNED(acc5, 5u * Ib4)
            if (M > 6u) QW_DOTB_PRUNED(acc6, 6u * Ib4)
            if (M > 7u) QW_DOTB_PRUNED(acc7, 7u * Ib4)
            #undef QW_DOTB_PRUNED
        }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1); acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3); acc4 = simd_sum(acc4); acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6); acc7 = simd_sum(acc7);
    if (lane == 0u) {
        if (M > 0u) yrow[(size_t)0u * N + row] = half(acc0);
        if (M > 1u) yrow[(size_t)1u * N + row] = half(acc1);
        if (M > 2u) yrow[(size_t)2u * N + row] = half(acc2);
        if (M > 3u) yrow[(size_t)3u * N + row] = half(acc3);
        if (M > 4u) yrow[(size_t)4u * N + row] = half(acc4);
        if (M > 5u) yrow[(size_t)5u * N + row] = half(acc5);
        if (M > 6u) yrow[(size_t)6u * N + row] = half(acc6);
        if (M > 7u) yrow[(size_t)7u * N + row] = half(acc7);
    }
}
