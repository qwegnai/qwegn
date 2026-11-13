// qwgemm_f32.metal — Shared f32 GEMM/GEMV, Walsh-Hadamard rotation, SwiGLU, and elementwise kernels.
// Includes weight-stationary batched HQ4_G32 GEMM (64 tokens x 32 rows), 4-row/1-row HQ4 GEMV,
// dense f32 GEMV (single and 4-row batched), H_32 rotations (f32 and f16), silu_mul_f32, and add_f32.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "qw_dequant.metalh"
#include "qw_dequant_f32.metalh"
#include "qw_dequant_multi.metalh"
using namespace metal;

#define QWG_F32_BM_TOKENS 64
#define QWG_F32_BM_ROWS   32
#define QWG_F32_BM_LD     36

kernel void qwg_gemm_q4g_f32_bm(device const Block_HQ4_G32* blocks [[buffer(0)]],
                                device const float*         x      [[buffer(1)]],
                                device float*               y      [[buffer(2)]],
                                constant uint&              M      [[buffer(3)]],
                                constant uint&              N      [[buffer(4)]],
                                constant uint&              K      [[buffer(5)]],
                                constant uint&              epi    [[buffer(6)]],
                                constant float&             scale  [[buffer(7)]],
                                uint2 tg   [[threadgroup_position_in_grid]],
                                uint  tid  [[thread_index_in_threadgroup]]) {
    threadgroup float As[QWG_F32_BM_TOKENS * QWG_F32_BM_LD];  // [token][k]
    threadgroup float Bs[QWG_F32_BM_ROWS * QWG_F32_BM_LD];    // [output row][k]
    threadgroup float* const Cs = As;  // epilogue aliases As: As is dead after the K loop

    const int t0 = int(tg.y) * QWG_F32_BM_TOKENS;
    if (t0 >= int(M)) return;
    const int nBase = int(tg.x) * QWG_F32_BM_ROWS;
    const uint sg = tid >> 5u;   // simdgroup 0..7 owns tokens sg*8..+8
    const uint gi = tid >> 3u;   // 8-lane group 0..31 owns output row nBase+gi
    const uint dl = tid & 7u;    // lane within the group
    const int nBlocks = int(K >> 5u);
    const int nRow = min(nBase + int(gi), int(N) - 1);
    device const Block_HQ4_G32* wrow = blocks + long(nRow) * long(nBlocks);

    simdgroup_float8x8 acc[4];
    for (int nb = 0; nb < 4; ++nb) acc[nb] = simdgroup_float8x8(0.0f);

    for (int g = 0; g < nBlocks; ++g) {
        // stage A: 64 tokens x 32 k of the already-rotated activation
        #pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int tk = int(gi) + pass * 32;
            const int tokAbs = t0 + tk;
            float4 v = float4(0.0f);
            if (tokAbs < int(M)) {
                device const float* xm = x + long(tokAbs) * long(K) + (g << 5) + int(dl << 2u);
                v = *(device const float4*)xm;
            }
            As[tk * QWG_F32_BM_LD + int(dl) * 4 + 0] = v.x;
            As[tk * QWG_F32_BM_LD + int(dl) * 4 + 1] = v.y;
            As[tk * QWG_F32_BM_LD + int(dl) * 4 + 2] = v.z;
            As[tk * QWG_F32_BM_LD + int(dl) * 4 + 3] = v.w;
        }
        // stage B: one HQ4 block per output row, 4 values per lane
        {
            float2 w01, w23;
            qwg_unpack_hq4_lane4_f32(wrow[g], dl, w01, w23);
            const int base = int(gi) * QWG_F32_BM_LD + int(dl) * 4;
            Bs[base + 0] = w01.x;
            Bs[base + 1] = w01.y;
            Bs[base + 2] = w23.x;
            Bs[base + 3] = w23.y;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int k8 = 0; k8 < 4; ++k8) {
            simdgroup_float8x8 a;
            simdgroup_load(a, As + int(sg) * 8 * QWG_F32_BM_LD + k8 * 8, QWG_F32_BM_LD, ulong2(0, 0), false);
            #pragma unroll
            for (int nb = 0; nb < 4; ++nb) {
                simdgroup_float8x8 b;
                simdgroup_load(b, Bs + nb * 8 * QWG_F32_BM_LD + k8 * 8, QWG_F32_BM_LD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc[nb], a, b, acc[nb]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int nb = 0; nb < 4; ++nb)
        simdgroup_store(acc[nb], Cs + int(sg) * 8 * QWG_F32_BM_LD + nb * 8, QWG_F32_BM_LD, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = int(tid); idx < QWG_F32_BM_TOKENS * QWG_F32_BM_ROWS; idx += 256) {
        const int m = idx / QWG_F32_BM_ROWS;
        const int n = idx % QWG_F32_BM_ROWS;
        const int tokAbs = t0 + m, col = nBase + n;
        float v = (tokAbs < int(M) && col < int(N)) ? Cs[m * QWG_F32_BM_LD + n] : 0.0f;
        if (epi == 1u || epi == 3u) { const float z = v / scale; v = z / (1.0f + exp(-z)); }
        else if (epi == 2u) { v = 1.0f / (1.0f + exp(-v)); }
        if (epi == 3u) {
            v = qwg_fwht32_lane_f32(v, uint(n));
        }
        if (tokAbs < int(M) && col < int(N)) y[long(tokAbs) * long(N) + col] = v;
    }
}

// GEMV, f32 activations, HQ4_G32 rows in the rot domain (x pre-rotated by qwg_h32_f32).
// 4 weight rows per simdgroup share the x load; 4 simdgroups per 128-thread threadgroup (16 rows/TG).
// epi: 0 none, 1 silu(y / scale), 2 sigmoid(y), 3 silu(y / scale) + H_32 across 2x16 row pairs.
kernel void qwg_gemv_q4g_f32_v4(device const Block_HQ4_G32* blocks [[buffer(0)]],
                                 device const float*         x      [[buffer(1)]],
                                 device float*               y      [[buffer(2)]],
                                 constant uint&              N      [[buffer(3)]],
                                 constant uint&              K      [[buffer(4)]],
                                 constant uint&              epi    [[buffer(5)]],
                                 constant float&             scale  [[buffer(6)]],
                                 uint3 tg   [[threadgroup_position_in_grid]],
                                 uint sg   [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tgC[32];
    const uint ep = epi & 0xFFu;
    const uint M = max(1u, (epi >> 8u) & 0xFFu);
    if (ep == 3u) {
        const uint rowBase = tg.x * 32u;
        if (rowBase >= N) return;
        for (uint pass = 0u; pass < 2u; ++pass) {
            const uint row0 = rowBase + pass * 16u + sg * 4u;
            float4 acc = float4(0.0f);
            if (row0 < N) {
                acc = qwg_gemv_hq4_rows4_f32(blocks, x + (size_t)tg.y * K, row0, N, K, lane);
            }
            if (lane == 0u) {
                float o[4] = {acc.x, acc.y, acc.z, acc.w};
                for (uint r = 0u; r < 4u; ++r) {
                    const uint col = pass * 16u + sg * 4u + r;
                    float v = (rowBase + col < N) ? o[r] : 0.0f;
                    const float z = v / scale;
                    v = z / (1.0f + exp(-z));
                    tgC[col] = v;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float v = qwg_fwht32_lane_f32(tgC[lane], lane);
            const uint row = rowBase + lane;
            if (row < N) y[(size_t)tg.y * N + row] = v;
        }
        return;
    }
    const bool halves = ((K >> 5u) & 31u) == 16u;
    if (halves) {
        const uint hl = lane & 15u;
        const uint row0 = (tg.x * 4u + sg) * 8u + (lane >> 4u) * 4u;
        if (row0 >= N) return;
        float4 acc[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
        if (M == 2u) {
            qwg_gemv_hq4_rows4_m2_f32<16u>(blocks, x, x + K, row0, N, K, hl, acc[0], acc[1]);
        } else if (M > 2u && M <= 4u) {
            qwg_gemv_hq4_rows4_m4_f32<16u>(blocks, x, M, row0, N, K, hl, acc);
        } else {
            acc[0] = qwg_gemv_hq4_rows4_f32<16u>(blocks, x + (size_t)tg.y * K, row0, N, K, hl);
        }
        if (hl != 0u) return;
        const bool multi = M > 1u && M <= 4u;
        const uint mN = multi ? M : 1u;
        for (uint m = 0u; m < mN; ++m) {
            float o[4] = {acc[m].x, acc[m].y, acc[m].z, acc[m].w};
            const size_t yb = (size_t)(multi ? m : tg.y) * N;
            for (uint r = 0; r < 4u; ++r) {
                const uint row = row0 + r;
                if (row >= N) break;
                float v = o[r];
                if (ep == 1u) { const float z = v / scale; v = z / (1.0f + exp(-z)); }
                else if (ep == 2u) { v = 1.0f / (1.0f + exp(-v)); }
                y[yb + row] = v;
            }
        }
        return;
    }
    const uint row0 = (tg.x * 4u + sg) * 4u;
    if (row0 >= N) return;
    if (M > 1u && M <= 4u) {
        float4 acc[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
        if (M == 2u) {
            qwg_gemv_hq4_rows4_m2_f32(blocks, x, x + K, row0, N, K, lane, acc[0], acc[1]);
        } else {
            qwg_gemv_hq4_rows4_m4_f32(blocks, x, M, row0, N, K, lane, acc);
        }
        if (lane == 0u) {
            #pragma unroll
            for (uint m = 0u; m < 4u; ++m) {
                if (m < M) {
                    float o[4] = {acc[m].x, acc[m].y, acc[m].z, acc[m].w};
                    for (uint r = 0; r < 4u; ++r) {
                        const uint row = row0 + r;
                        if (row >= N) break;
                        float v = o[r];
                        if (ep == 1u) { const float z = v / scale; v = z / (1.0f + exp(-z)); }
                        else if (ep == 2u) { v = 1.0f / (1.0f + exp(-v)); }
                        y[(size_t)m * N + row] = v;
                    }
                }
            }
        }
        return;
    }
    const float4 acc = qwg_gemv_hq4_rows4_f32(blocks, x + (size_t)tg.y * K, row0, N, K, lane);
    if (lane != 0u) return;
    float o[4] = {acc.x, acc.y, acc.z, acc.w};
    for (uint r = 0; r < 4u; ++r) {
        const uint row = row0 + r;
        if (row >= N) break;
        float v = o[r];
        if (ep == 1u) { const float z = v / scale; v = z / (1.0f + exp(-z)); }
        else if (ep == 2u) { v = 1.0f / (1.0f + exp(-v)); }
        y[(size_t)tg.y * N + row] = v;
    }
}

// GEMV + in-kernel block argmax epilogue (the LM-head T=1 greedy path): same rows as
// qwg_gemv_q4g_f32_v4, but each threadgroup folds its 16 (logit, row) pairs into one
// sortable key pair in keysHi/keysLo[tg.y * slots + tg.x] (ties → lowest row; encoding = qwg_head_argmax).
kernel void qwg_gemv_q4g_f32_v4_am(device const Block_HQ4_G32* blocks [[buffer(0)]],
                                   device const float*         x      [[buffer(1)]],
                                   device float*               y      [[buffer(2)]],
                                   device uint*                keysHi [[buffer(7)]],
                                   device uint*                keysLo [[buffer(8)]],
                                   constant uint&              N      [[buffer(3)]],
                                   constant uint&              K      [[buffer(4)]],
                                   constant uint&              epi    [[buffer(5)]],
                                   constant float&             scale  [[buffer(6)]],
                                   uint3 tg   [[threadgroup_position_in_grid]],
                                   uint  tid  [[thread_index_in_threadgroup]],
                                   uint  sg   [[simdgroup_index_in_threadgroup]],
                                   uint  lane [[thread_index_in_simdgroup]]) {
    const uint ep = epi & 0xFFu;
    const uint M = epi >> 8u;
    const uint row0 = (tg.x * 4u + sg) * 4u;
    threadgroup uint bHi[16], bLo[16];
    if (M > 1u && M <= 4u) {
        float4 acc[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
        if (row0 < N) {
            if (M == 2u) {
                qwg_gemv_hq4_rows4_m2_f32(blocks, x, x + K, row0, N, K, lane, acc[0], acc[1]);
            } else {
                qwg_gemv_hq4_rows4_m4_f32(blocks, x, M, row0, N, K, lane, acc);
            }
        }
        if (lane == 0u) {
            #pragma unroll
            for (uint m = 0u; m < 4u; ++m) {
                if (m < M) {
                    uint bh = 0u, bl = 0u;
                    if (row0 < N) {
                        float o[4] = {acc[m].x, acc[m].y, acc[m].z, acc[m].w};
                        for (uint r = 0; r < 4u; ++r) {
                            const uint row = row0 + r;
                            if (row >= N) break;
                            float v = o[r];
                            if (ep == 1u) { const float z = v / scale; v = z / (1.0f + exp(-z)); }
                            else if (ep == 2u) { v = 1.0f / (1.0f + exp(-v)); }
                            y[(size_t)m * N + row] = v;
                            const uint u = as_type<uint>(v);
                            const uint hi = (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
                            const uint lo = ~row;
                            if (r == 0u || hi > bh || (hi == bh && lo > bl)) { bh = hi; bl = lo; }
                        }
                    }
                    bHi[m * 4u + sg] = bh; bLo[m * 4u + sg] = bl;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < M) {
            const uint m = tid;
            uint bh = bHi[m * 4u], bl = bLo[m * 4u];
            for (uint q = 1u; q < 4u; q++) {
                const uint idx = m * 4u + q;
                if (bHi[idx] > bh || (bHi[idx] == bh && bLo[idx] > bl)) { bh = bHi[idx]; bl = bLo[idx]; }
            }
            const uint slots = (N + 15u) >> 4u;
            keysHi[m * slots + tg.x] = bh; keysLo[m * slots + tg.x] = bl;
        }
        return;
    }
    const float4 acc = qwg_gemv_hq4_rows4_f32(blocks, x + (size_t)tg.y * K, row0, N, K, lane);
    if (lane == 0u) {
        uint bh = 0u, bl = 0u;
        if (row0 < N) {
            float o[4] = {acc.x, acc.y, acc.z, acc.w};
            for (uint r = 0; r < 4u; ++r) {
                const uint row = row0 + r;
                if (row >= N) break;
                float v = o[r];
                if (ep == 1u) { const float z = v / scale; v = z / (1.0f + exp(-z)); }
                else if (ep == 2u) { v = 1.0f / (1.0f + exp(-v)); }
                y[(size_t)tg.y * N + row] = v;
                const uint hi = (as_type<uint>(v) & 0x80000000u) ? ~as_type<uint>(v)
                                                                 : (as_type<uint>(v) ^ 0x80000000u);
                const uint lo = ~row;
                if (r == 0u || hi > bh || (hi == bh && lo > bl)) { bh = hi; bl = lo; }
            }
        }
        bHi[sg] = bh; bLo[sg] = bl;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint bh = bHi[0], bl = bLo[0];
        for (uint q = 1u; q < 4u; q++) {
            if (bHi[q] > bh || (bHi[q] == bh && bLo[q] > bl)) { bh = bHi[q]; bl = bLo[q]; }
        }
        const uint slots = (N + 15u) >> 4u;
        keysHi[tg.y * slots + tg.x] = bh; keysLo[tg.y * slots + tg.x] = bl;
    }
}

// 2-row LM-head GEMV + in-kernel block argmax epilogue: reads HQ4 blocks once across both token rows (m = 0, 1).
kernel void qwg_gemv_q4g_f32_v4_am2(device const Block_HQ4_G32* blocks [[buffer(0)]],
                                    device const float*         x      [[buffer(1)]],
                                    device uint*                keysHi [[buffer(2)]],
                                    device uint*                keysLo [[buffer(3)]],
                                    constant uint&              N      [[buffer(4)]],
                                    constant uint&              K      [[buffer(5)]],
                                    device float*               y      [[buffer(6)]],
                                    uint3 tg   [[threadgroup_position_in_grid]],
                                    uint  tid  [[thread_index_in_threadgroup]],
                                    uint  sg   [[simdgroup_index_in_threadgroup]],
                                    uint  lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg.x * 4u + sg) * 4u;
    float4 a0 = float4(0.0f), a1 = float4(0.0f);
    if (row0 < N) {
        qwg_gemv_hq4_rows4_m2_f32(blocks, x, x + K, row0, N, K, lane, a0, a1);
    }
    threadgroup uint bHi0[4], bLo0[4], bHi1[4], bLo1[4];
    if (lane == 0u) {
        uint bh0 = 0u, bl0 = 0u, bh1 = 0u, bl1 = 0u;
        if (row0 < N) {
            float o0[4] = {a0.x, a0.y, a0.z, a0.w};
            float o1[4] = {a1.x, a1.y, a1.z, a1.w};
            for (uint r = 0; r < 4u; ++r) {
                const uint row = row0 + r;
                if (row >= N) break;
                y[row] = o0[r];
                y[N + row] = o1[r];
                const uint u0 = as_type<uint>(o0[r]), u1 = as_type<uint>(o1[r]);
                const uint hi0 = (u0 & 0x80000000u) ? ~u0 : (u0 ^ 0x80000000u);
                const uint hi1 = (u1 & 0x80000000u) ? ~u1 : (u1 ^ 0x80000000u);
                const uint lo = ~row;
                if (r == 0u || hi0 > bh0 || (hi0 == bh0 && lo > bl0)) { bh0 = hi0; bl0 = lo; }
                if (r == 0u || hi1 > bh1 || (hi1 == bh1 && lo > bl1)) { bh1 = hi1; bl1 = lo; }
            }
        }
        bHi0[sg] = bh0; bLo0[sg] = bl0;
        bHi1[sg] = bh1; bLo1[sg] = bl1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint bh0 = bHi0[0], bl0 = bLo0[0], bh1 = bHi1[0], bl1 = bLo1[0];
        for (uint q = 1u; q < 4u; q++) {
            if (bHi0[q] > bh0 || (bHi0[q] == bh0 && bLo0[q] > bl0)) { bh0 = bHi0[q]; bl0 = bLo0[q]; }
            if (bHi1[q] > bh1 || (bHi1[q] == bh1 && bLo1[q] > bl1)) { bh1 = bHi1[q]; bl1 = bLo1[q]; }
        }
        const uint slots = (N + 15u) >> 4u;
        keysHi[tg.x] = bh0; keysLo[tg.x] = bl0;
        keysHi[slots + tg.x] = bh1; keysLo[slots + tg.x] = bl1;
    }
}

// Embedding gather: out[c * hidden + d] = bf16tof32(table[token * hidden + d]) for
// c in 0..<hc. The token id is read on-device (the head's token tensor), so the
// decode step needs no host round-trip to stage the input stream.
kernel void qwg_embed_bf16_f32(device const ushort* table [[buffer(0)]],   // [vocab, hidden] raw bf16
                               device const int*   token  [[buffer(1)]],   // [rows]
                               device float*       out    [[buffer(2)]],   // [rows, hc * hidden]
                               constant uint&      hidden [[buffer(3)]],
                               constant uint&      hc     [[buffer(4)]],
                               uint3 tg  [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    const uint W = hidden * hc;
    const uint r = tg.y;
    const int tok = max(token[r], 0);   // caller guarantees tok < vocab
    device float* outRow = out + (size_t)r * W;
    for (uint i = tg.x * 256u + tid; i < W; i += 256u * 4096u) {
        outRow[i] = as_type<float>(uint(table[(uint)tok * hidden + i % hidden]) << 16u);
    }
}

// gemv buffers: 0 W f32 [N, K] | 1 X f32 [T, K] | 2 Y f32 [T, N] | 3 N | 4 K | 5 epilogue | 6 scale
// epilogue 0 none, 1 silu(y/scale), 2 sigmoid(y), 3 2*sigmoid(y/scale). TG per (row n, token t), 256 threads.
kernel void qwg_f32_gemv(device const float* W     [[buffer(0)]],
                         device const float* X     [[buffer(1)]],
                         device float*       Y     [[buffer(2)]],
                         constant uint&      N     [[buffer(3)]],
                         constant uint&      K     [[buffer(4)]],
                         constant uint&      epi   [[buffer(5)]],
                         constant float&     scale [[buffer(6)]],
                         uint2 tg  [[threadgroup_position_in_grid]],
                         uint  tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[8];
    const uint n = tg.x, t = tg.y;
    float acc = 0.0f;
    for (uint k = tid; k < K; k += 256u) acc += W[n * K + k] * X[t * K + k];
    acc = simd_sum(acc);
    if ((tid & 31u) == 0u) partial[tid >> 5u] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0u) return;
    float y = 0.0f;
    for (uint i = 0u; i < 8u; i++) y += partial[i];
    if (epi == 1u) { const float z = y / scale; y = z / (1.0f + exp(-z)); }
    else if (epi == 2u) { y = 1.0f / (1.0f + exp(-y)); }
    else if (epi == 3u) { y = 2.0f / (1.0f + exp(-(y / scale))); }
    Y[t * N + n] = y;
}

// Batched f32 GEMV (M >= 4): 32x32 simdgroup_matrix or 8-SG vectorized when M >= 16; 4-row shared-W GEMV for M < 16.
kernel void qwg_f32_gemv_bm(device const float* W     [[buffer(0)]],
                            device const float* X     [[buffer(1)]],
                            device float*       Y     [[buffer(2)]],
                            constant uint&      N     [[buffer(3)]],
                            constant uint&      K     [[buffer(4)]],
                            constant uint&      epi   [[buffer(5)]],
                            constant float&     scale [[buffer(6)]],
                            constant uint&      M     [[buffer(7)]],
                            uint2 tg  [[threadgroup_position_in_grid]],
                            uint  tid [[thread_index_in_threadgroup]]) {
    threadgroup float As[32][36];
    threadgroup float Bs[32][36];
    if (M >= 16u && ((N >= 16u && (K & 31u) == 0u) || (N < 16u && (K & 3u) == 0u))) {
        qwg_dense_gemv_bm_fast<false>(W, X, Y, N, K, epi, scale, M, tg, tid, As, Bs);
        return;
    }
    const uint t0 = tg.x * 4u, n = tg.y;
    if (t0 >= M || n >= N) return;
    const uint t1 = min(t0 + 1u, M - 1u), t2 = min(t0 + 2u, M - 1u), t3 = min(t0 + 3u, M - 1u);
    device const float* Wn = W + (size_t)n * K;
    device const float* X0 = X + (size_t)t0 * K;
    device const float* X1 = X + (size_t)t1 * K;
    device const float* X2 = X + (size_t)t2 * K;
    device const float* X3 = X + (size_t)t3 * K;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint k = tid; k < K; k += 256u) {
        const float wk = Wn[k];
        a0 += wk * X0[k];
        a1 += wk * X1[k];
        a2 += wk * X2[k];
        a3 += wk * X3[k];
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
    if ((tid & 31u) == 0u) {
        const uint sg = tid >> 5u;
        As[0][sg] = a0; As[1][sg] = a1; As[2][sg] = a2; As[3][sg] = a3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4u && t0 + tid < M) {
        float y = 0.0f;
        for (uint i = 0u; i < 8u; i++) y += As[tid][i];
        Y[(size_t)(t0 + tid) * N + n] = qwg_apply_epi_f32(y, epi, scale);
    }
}

// Rotation of the normalized Walsh-Hadamard domain (f32 output).
kernel void qwg_h32_f32(device const float* x   [[buffer(0)]],
                        device float*       out [[buffer(1)]],
                        constant uint&      H   [[buffer(2)]],
                        uint2 tg   [[threadgroup_position_in_grid]],
                        uint  lane [[thread_index_in_simdgroup]]) {
    const uint g = tg.x, t = tg.y;
    const float v = qwg_fwht32_lane_f32(x[t * H + g * 32u + lane], lane);
    out[t * H + g * 32u + lane] = v;
}

// Optional RMSNorm (1 + gamma) + H_32 rotation on f32 [rows, L], emitting f32 and/or f16.
// flags: bit 0 = hasNorm, bit 1 = writeOut16, bit 2 = skipOut32. Requires L % 32 == 0.
kernel void qwg_latent_norm_h32_f32(device const float* x     [[buffer(0)]],
                                    device const half*  gamma [[buffer(1)]],
                                    device float*       out32 [[buffer(2)]],
                                    device half*        out16 [[buffer(3)]],
                                    constant uint&      L     [[buffer(4)]],
                                    constant float&     eps   [[buffer(5)]],
                                    constant uint&      flags [[buffer(6)]],
                                    uint2 tg   [[threadgroup_position_in_grid]],
                                    uint  sg   [[simdgroup_index_in_threadgroup]],
                                    uint  lane [[thread_index_in_simdgroup]]) {
    const bool hasNorm = (flags & 1u) != 0u;
    const bool write16 = (flags & 2u) != 0u;
    const bool skip32 = (flags & 4u) != 0u;
    if (!hasNorm) {
        const uint col = tg.x * 32u + lane;
        if (col >= L) return;
        const size_t idx = (size_t)tg.y * L + col;
        const float v = qwg_fwht32_lane_f32(x[idx], lane);
        if (!skip32) out32[idx] = v;
        if (write16) out16[idx] = half(v);
        return;
    }
    threadgroup float partSS[8];
    const uint tid = sg * 32u + lane;
    device const float* xr = x + (size_t)tg.y * L;
    if (L <= 2048u) {
        float regX[8];
        float ss = 0.0f;
        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
            const uint c = i * 256u + tid;
            const float v = (c < L) ? xr[c] : 0.0f;
            regX[i] = v;
            ss = fma(v, v, ss);
        }
        ss = simd_sum(ss);
        if (lane == 0u) partSS[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float totSS = 0.0f;
        for (uint i = 0u; i < 8u; ++i) totSS += partSS[i];
        const float rstd = rsqrt(totSS / float(L) + eps);
        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
            const uint b0 = i * 256u;
            if (b0 >= L) break;
            const uint c = b0 + tid;
            float v = (c < L) ? (regX[i] * rstd * (1.0f + float(gamma[c]))) : 0.0f;
            v = qwg_fwht32_lane_f32(v, lane);
            if (c < L) {
                const size_t idx = (size_t)tg.y * L + c;
                if (!skip32) out32[idx] = v;
                if (write16) out16[idx] = half(v);
            }
        }
        return;
    }
    float ss = 0.0f;
    for (uint c = tid; c < L; c += 256u) {
        const float v = xr[c];
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    if (lane == 0u) partSS[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float totSS = 0.0f;
    for (uint i = 0u; i < 8u; ++i) totSS += partSS[i];
    const float rstd = rsqrt(totSS / float(L) + eps);
    const uint lim = (L + 255u) & ~255u;
    for (uint b0 = 0u; b0 < lim; b0 += 256u) {
        const uint c = b0 + tid;
        float v = (c < L) ? (xr[c] * rstd * (1.0f + float(gamma[c]))) : 0.0f;
        v = qwg_fwht32_lane_f32(v, lane);
        if (c < L) {
            const size_t idx = (size_t)tg.y * L + c;
            if (!skip32) out32[idx] = v;
            if (write16) out16[idx] = half(v);
        }
    }
}

// 32-point normalized WHT in f32, same butterflies and scale as fwht32_f32_to_h8 (lane = simdgroup lane).
// In the STANDARD basis the transform is skipped: the activation stays as it
// is, matching the weights it is about to meet.
inline void qwg_fwht32_f32(thread float (&f)[8], uint lane_id) {
    if (!QW_FWHT_ROTATED) return;
    for (int i = 0; i < 8; i += 2) { float a = f[i], b = f[i + 1]; f[i] = a + b; f[i + 1] = a - b; }
    for (int i = 0; i < 8; i += 4) {
        float a0 = f[i], a1 = f[i + 1], b0 = f[i + 2], b1 = f[i + 3];
        f[i] = a0 + b0; f[i + 1] = a1 + b1; f[i + 2] = a0 - b0; f[i + 3] = a1 - b1;
    }
    for (int i = 0; i < 4; ++i) { float a = f[i], b = f[i + 4]; f[i] = a + b; f[i + 4] = a - b; }
    bool neg1 = (lane_id & 1u) != 0;
    for (int i = 0; i < 8; ++i) { float p = simd_shuffle_xor(f[i], ushort(1)); f[i] = neg1 ? (p - f[i]) : (f[i] + p); }
    bool neg2 = (lane_id & 2u) != 0;
    constexpr float inv_sqrt32 = 0.17677669529663689f;
    for (int i = 0; i < 8; ++i) {
        float p = simd_shuffle_xor(f[i], ushort(2));
        f[i] = (neg2 ? (p - f[i]) : (f[i] + p)) * inv_sqrt32;
    }
}

// silu(gate) * up in f32, then the 32-point WHT (I % 32 == 0), as qwg_silu_mul_f16. Requires I % 8 == 0.
// buffers: 0 gu f32 [M, 2I] | 1 out f32 [M, I] | 2 M | 3 I | 4 out16 f16 [M, I] | 5 hasOut16; grid (I/8, M), 32-lane simdgroups.
kernel void qwg_silu_mul_f32(device const float* gu       [[buffer(0)]],
                             device float*       out      [[buffer(1)]],
                             constant uint&      M        [[buffer(2)]],
                             constant uint&      I        [[buffer(3)]],
                             device half*        out16    [[buffer(4)]],
                             constant uint&      hasOut16 [[buffer(5)]],
                             uint2 gid  [[thread_position_in_grid]],
                             uint  lane [[thread_index_in_simdgroup]]) {
    const uint mi = gid.x * 8u;
    const uint m = gid.y;
    if (m >= M || mi >= I) return;
    device const float* g = gu + (size_t)m * 2u * I + mi;
    device const float* u = g + I;
    float ov[8];
    for (uint k = 0; k < 8; ++k) {
        ov[k] = qwg_swiglu_mul(g[k], u[k]);
    }
    if ((I & 31u) == 0u) qwg_fwht32_f32(ov, gid.x & 31u);
    const size_t base = (size_t)m * I + mi;
    if (hasOut16 != 0u) {
        device half* o16 = out16 + base;
        for (uint k = 0; k < 8; ++k) o16[k] = half(ov[k]);
    } else {
        device float* o = out + base;
        for (uint k = 0; k < 8; ++k) o[k] = ov[k];
    }
}

// add buffers: 0 a f32 [N] | 1 b f32 [N] | 2 out f32 [N] | 3 N ; dispatch: N threads.
kernel void qwg_add_f32(device const float* a   [[buffer(0)]],
                        device const float* b   [[buffer(1)]],
                        device float*       out [[buffer(2)]],
                        constant uint&      N   [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < N) out[i] = a[i] + b[i];
}

// Batched MLP SiLU × Up epilogue + fused FWHT-32 rotation in FP32, emitting hi/lo FP16 split halves.
kernel void qwg_silu_mul_split_f16(device const half* gu    [[buffer(0)]],
                                   device half*       outHi [[buffer(1)]],
                                   device half*       outLo [[buffer(2)]],
                                   constant uint&     M     [[buffer(3)]],
                                   constant uint&     I     [[buffer(4)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    constexpr uint kU = 8u;
    const uint mi = gid.x * kU;
    const uint m = gid.y;
    if (m >= M || mi >= I) return;
    const size_t in_row = (size_t)m * (2u * I);
    const size_t out_row = (size_t)m * I;
    if (mi + kU <= I) {
        device const half4* gp = (device const half4*)(gu + in_row + mi);
        device const half4* up = (device const half4*)(gu + in_row + I + mi);
        const half4 g0 = gp[0], g1 = gp[1];
        const half4 u0 = up[0], u1 = up[1];
        float fv[8] = {
            qwg_swiglu_mul(float(g0[0]), float(u0[0])),
            qwg_swiglu_mul(float(g0[1]), float(u0[1])),
            qwg_swiglu_mul(float(g0[2]), float(u0[2])),
            qwg_swiglu_mul(float(g0[3]), float(u0[3])),
            qwg_swiglu_mul(float(g1[0]), float(u1[0])),
            qwg_swiglu_mul(float(g1[1]), float(u1[1])),
            qwg_swiglu_mul(float(g1[2]), float(u1[2])),
            qwg_swiglu_mul(float(g1[3]), float(u1[3]))
        };
        half hi[8];
        if ((I & 31u) == 0u) {
            fwht32_f32_to_h8(fv, hi, gid.x & 31u);
        } else {
            for (uint k = 0; k < 8u; k++) hi[k] = half(fv[k]);
        }
        device half4* opHi = (device half4*)(outHi + out_row + mi);
        opHi[0] = half4(hi[0], hi[1], hi[2], hi[3]);
        opHi[1] = half4(hi[4], hi[5], hi[6], hi[7]);
    } else {
        for (uint k = 0; k < kU && mi + k < I; k++) {
            const float g = float(gu[in_row + mi + k]);
            const float u = float(gu[in_row + I + mi + k]);
            outHi[out_row + mi + k] = half(qwg_swiglu_mul(g, u));
        }
    }
}
