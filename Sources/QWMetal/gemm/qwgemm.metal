// QWGEMM engine kernels (C05) — HQ4_G32_M5 architecture.
// All fp32 accumulate (mandatory).
#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_dequant.metalh"
using namespace metal;

constant uint QW_FC_GDN_QK_HEADS [[function_constant(12)]];
constant uint QW_FC_GDN_V_HEADS  [[function_constant(13)]];
constant uint QW_FC_GDN_HEAD_DIM [[function_constant(14)]];
constant bool QW_HAS_GDN_QK_HEADS = is_function_constant_defined(QW_FC_GDN_QK_HEADS);
constant bool QW_HAS_GDN_V_HEADS  = is_function_constant_defined(QW_FC_GDN_V_HEADS);
constant bool QW_HAS_GDN_HEAD_DIM = is_function_constant_defined(QW_FC_GDN_HEAD_DIM);
constant uint QW_GDN_QK_HEADS = QW_HAS_GDN_QK_HEADS ? QW_FC_GDN_QK_HEADS : 16u;
constant uint QW_GDN_V_HEADS  = QW_HAS_GDN_V_HEADS  ? QW_FC_GDN_V_HEADS  : 48u;
constant uint QW_GDN_HEAD_DIM = QW_HAS_GDN_HEAD_DIM ? QW_FC_GDN_HEAD_DIM : 128u;
constant uint QW_GDN_CONV_CH  = (2u * QW_GDN_QK_HEADS + QW_GDN_V_HEADS) * QW_GDN_HEAD_DIM;
constant uint QW_GDN_QKV_CH   = QW_GDN_CONV_CH;
constant uint QW_GDN_Z_CH     = QW_GDN_V_HEADS * QW_GDN_HEAD_DIM;

// 16 B = 8 halves load type (own type: SDK exports packed_half8 in two
// namespaces -> ambiguous once mpp is imported; plain row-major halves).
struct H8 { half v[8]; } __attribute__((__packed__, __aligned__(16)));

// ── qwg_fwht32_f16 — in-place 32-point Fast Walsh-Hadamard Transform ─────────
// Grid: N elements (must be a multiple of 32), threadsPerThreadgroup = 32 (or multiple of 32).
kernel void qwg_fwht32_f16(device half* x [[buffer(0)]],
                           uint gid [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    half val = x[gid];
    x[gid] = simd_fwht32(val, lane);
}

// ── qwg_gemv_f16_v4 — f16 escape-hatch GEMV (escape hatch / debug path) ────
//   buffers: 0 W f16 [N,K] | 1 x f16 | 2 (unused) | 3 y f32 | 4 N | 5 K
kernel void qwg_gemv_f16_v4(device const H8* W             [[buffer(0)]],
                            constant const H8* x           [[buffer(1)]],
                            device float* y                [[buffer(3)]],
                            constant uint& N               [[buffer(4)]],
                            constant uint& K               [[buffer(5)]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint ti [[thread_index_in_threadgroup]]) {
    uint row = tg;
    if (row >= N) return;
    uint kChunks = K / 8;
    device const H8* wr = W + (size_t)row * kChunks;
    float acc = 0.0f;
    for (uint c = ti; c < kChunks; c += 32) {
        device const H8* wv = wr + c;
        constant const H8* xv = x + c;
        for (int i = 0; i < 8; i++) acc += float(wv->v[i]) * float(xv->v[i]);
    }
    float4 v = float4(acc, 0.0f, 0.0f, 0.0f);
    v += simd_shuffle_down(v, 16);
    v += simd_shuffle_down(v, 8);
    v += simd_shuffle_down(v, 4);
    v += simd_shuffle_down(v, 2);
    v += simd_shuffle_down(v, 1);
    if (ti == 0) y[row] = v.x;
}

// ── qwg_gemv_q4g_v4 — HQ4_G32_M5 F1/F2 GEMV ───────────────────────────────
//   buffers: 0 blocks [N, K/32] | 1 x_rot f16 | 3 y f32 | 4 N u32 | 5 K u32
kernel void qwg_gemv_q4g_v4(device const Block_HQ4_G32* blocks [[buffer(0)]],
                            device const half* x               [[buffer(1)]],
                            device float* y                    [[buffer(3)]],
                            constant uint& N                   [[buffer(4)]],
                            constant uint& K                   [[buffer(5)]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint ti [[thread_index_in_threadgroup]],
                            uint tptg [[threads_per_threadgroup]]) {
    uint simd_id = ti >> 5u;
    uint lane = ti & 31u;
    uint row = tg * max(1u, tptg >> 5u) + simd_id;
    if (row >= N) return;
    float vx = qwg_gemv_hq4_row(blocks, x, row, K, lane);
    if (lane == 0u) y[row] = vx;
}

// ── qwg_gemv_q4g_v4_h16 / _dual — f16 store (+ optional f32 store) GEMV ───
constant bool store32 [[function_constant(0)]];

template <bool store32Val>
static inline void qwg_gemv_q4g_v4_h16_body(device const Block_HQ4_G32* blocks,
                                            device const half* x,
                                            device float* y32,
                                            constant uint& N,
                                            constant uint& K,
                                            device half* y16,
                                            uint tg,
                                            uint ti,
                                            uint tptg = 32u) {
    uint rows_per_tg = max(1u, tptg >> 5u);
    uint simd_id = ti >> 5u;
    uint lane = ti & 31u;
    uint row = tg * rows_per_tg + simd_id;
    if (row >= N) return;
    float vx = qwg_gemv_hq4_row(blocks, x, row, K, lane);
    if (lane == 0u) {
        if (store32Val) y32[row] = vx;
        y16[row] = half(vx);
    }
}

template <bool store32Val>
static inline void qwg_gemv_q4g_v4_h16_rows2_body(device const Block_HQ4_G32* blocks,
                                                  device const half* x,
                                                  device float* y32,
                                                  uint realN,
                                                  constant uint& K,
                                                  device half* y16,
                                                  uint tg,
                                                  uint ti,
                                                  uint tptg = 128u) {
    uint simd_per_tg = max(1u, tptg >> 5u);
    uint simd_id = ti >> 5u;
    uint lane = ti & 31u;
    uint row0 = (tg * simd_per_tg + simd_id) * 2u;
    if (row0 >= realN) return;
    const uint nBlocks = K >> 5u;
    device const Block_HQ4_G32* r0 = blocks + (size_t)row0 * nBlocks;
    device const Block_HQ4_G32* r1 = (row0 + 1u < realN) ? (r0 + nBlocks) : r0;
    float2 acc = float2(0.0f);
    for (uint g = lane; g < nBlocks; g += 32u) {
        device const half* xg = x + (g << 5u);
        half4 lo[4], hi[4];
        #pragma unroll
        for (uint w = 0; w < 4; ++w) {
            qwg_load_half8(xg + w * 8u, lo[w], hi[w]);
        }
        acc.x += qwg_dot_hq4_block_reg(r0[g], lo, hi);
        acc.y += qwg_dot_hq4_block_reg(r1[g], lo, hi);
    }
    acc += simd_shuffle_down(acc, 16);
    acc += simd_shuffle_down(acc, 8);
    acc += simd_shuffle_down(acc, 4);
    acc += simd_shuffle_down(acc, 2);
    acc += simd_shuffle_down(acc, 1);
    if (lane == 0u) {
        if (store32Val) {
            y32[row0] = acc.x;
            if (row0 + 1u < realN) y32[row0 + 1u] = acc.y;
        }
        y16[row0] = half(acc.x);
        if (row0 + 1u < realN) y16[row0 + 1u] = half(acc.y);
    }
}

template <bool store32Val>
static inline void qwg_gemv_q4g_v4_h16_rows4_body(device const Block_HQ4_G32* blocks,
                                                  device const half* x,
                                                  device float* y32,
                                                  uint realN,
                                                  constant uint& K,
                                                  device half* y16,
                                                  uint tg,
                                                  uint ti,
                                                  uint tptg = 128u) {
    uint simd_per_tg = max(1u, tptg >> 5u);
    uint simd_id = ti >> 5u;
    uint lane = ti & 31u;
    uint row0 = (tg * simd_per_tg + simd_id) * 4u;
    if (row0 >= realN) return;
    float4 acc = qwg_gemv_hq4_rows4(blocks, x, row0, realN, K, lane);
    if (lane == 0u) {
        if (store32Val) {
            y32[row0] = acc.x;
            if (row0 + 1u < realN) y32[row0 + 1u] = acc.y;
            if (row0 + 2u < realN) y32[row0 + 2u] = acc.z;
            if (row0 + 3u < realN) y32[row0 + 3u] = acc.w;
        }
        y16[row0] = half(acc.x);
        if (row0 + 1u < realN) y16[row0 + 1u] = half(acc.y);
        if (row0 + 2u < realN) y16[row0 + 2u] = half(acc.z);
        if (row0 + 3u < realN) y16[row0 + 3u] = half(acc.w);
    }
}

kernel void qwg_gemv_q4g_v4_h16(device const Block_HQ4_G32* blocks [[buffer(0)]],
                                device const half* x               [[buffer(1)]],
                                device float* y32                  [[buffer(3)]],
                                constant uint& N                   [[buffer(4)]],
                                constant uint& K                   [[buffer(5)]],
                                device half* y16                   [[buffer(8)]],
                                uint tg [[threadgroup_position_in_grid]],
                                uint ti [[thread_index_in_threadgroup]],
                                uint tptg [[threads_per_threadgroup]]) {
    qwg_gemv_q4g_v4_h16_body<false>(blocks, x, y32, N, K, y16, tg, ti, tptg);
}

kernel void qwg_gemv_q4g_v4_h16_dual(device const Block_HQ4_G32* blocks [[buffer(0)]],
                                     device const half* x               [[buffer(1)]],
                                     device float* y32                  [[buffer(3)]],
                                     constant uint& N                   [[buffer(4)]],
                                     constant uint& K                   [[buffer(5)]],
                                     device half* y16                   [[buffer(8)]],
                                     uint tg [[threadgroup_position_in_grid]],
                                     uint ti [[thread_index_in_threadgroup]],
                                     uint tptg [[threads_per_threadgroup]]) {
    qwg_gemv_q4g_v4_h16_body<true>(blocks, x, y32, N, K, y16, tg, ti, tptg);
}

// ── qwg_gemv_q4g_silu_pair — epilogue .siluMulPair (packed gate‖up FFN) ────
kernel void qwg_gemv_q4g_silu_pair(device const Block_HQ4_G32* blocks [[buffer(0)]],
                                   device const half* x               [[buffer(1)]],
                                   device float* y                    [[buffer(3)]],
                                   constant uint& N                   [[buffer(4)]],
                                   constant uint& K                   [[buffer(5)]],
                                   uint tg [[threadgroup_position_in_grid]],
                                   uint ti [[thread_index_in_threadgroup]]) {
    uint I = N / 2;
    uint row = tg;
    if (row >= I) return;
    uint sub = ti >> 5;                       // 0 = gate row, 1 = up row
    uint lane = ti & 31;
    uint outRow = sub == 0 ? row : row + I;   // gate row r, up row r+I
    float vx = qwg_gemv_hq4_row(blocks, x, outRow, K, lane);
    threadgroup float part[2];
    if (lane == 0) part[sub] = vx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ti == 0) {
        float gate = part[0], up = part[1];
        y[row] = qwg_swiglu_mul(gate, up);   // silu(gate)·up, fp32
    }
}

// ── qwg_gemv_q4g_silu_pair_h16 — epilogue .siluMulPair with f16 store ──────
static inline void qwg_gemv_q4g_silu_pair_h16_body(device const Block_HQ4_G32* blocks,
                                                   device const half* x,
                                                   constant uint& N,
                                                   constant uint& K,
                                                   device half* h16,
                                                   uint tg,
                                                   uint ti,
                                                   threadgroup float* part,
                                                   uint tptg = 64u) {
    uint I = N / 2;
    uint pairs_per_tg = max(1u, tptg >> 6u);
    uint pair_id = ti >> 6u;
    uint sub = (ti >> 5u) & 1u;               // 0 = gate row, 1 = up row
    uint lane = ti & 31u;
    uint row = tg * pairs_per_tg + pair_id;
    if (row < I) {
        uint outRow = sub == 0u ? row : row + I;   // gate row r, up row r+I
        float vx = qwg_gemv_hq4_row(blocks, x, outRow, K, lane);
        if (lane == 0u) part[pair_id * 2u + sub] = vx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sub == 0u && lane == 0u && row < I) {
        float gate = part[pair_id * 2u], up = part[pair_id * 2u + 1u];
        float prod = qwg_swiglu_mul(gate, up);
        h16[row] = half(prod);
    }
}

kernel void qwg_gemv_q4g_silu_pair_h16(device const Block_HQ4_G32* blocks [[buffer(0)]],
                                       device const half* x               [[buffer(1)]],
                                       constant uint& N                   [[buffer(4)]],
                                       constant uint& K                   [[buffer(5)]],
                                       device half* h16                   [[buffer(8)]],
                                       uint tg [[threadgroup_position_in_grid]],
                                       uint ti [[thread_index_in_threadgroup]],
                                       uint tptg [[threads_per_threadgroup]]) {
    threadgroup float part[16];
    qwg_gemv_q4g_silu_pair_h16_body(blocks, x, N, K, h16, tg, ti, part, tptg);
}

kernel void qwg_gemv_q4g_silu_pair_dx_h16(device const Block_HQ4_G32* blocks [[buffer(0)]],
                                          device const half* x               [[buffer(1)]],
                                          constant uint& N                   [[buffer(4)]],
                                          constant uint& K                   [[buffer(5)]],
                                          device half* h16                   [[buffer(8)]],
                                          uint tg [[threadgroup_position_in_grid]],
                                          uint ti [[thread_index_in_threadgroup]],
                                          uint tptg [[threads_per_threadgroup]]) {
    if ((N & 0x40000000u) != 0u) {
        threadgroup half tgAct[32];
        const uint realN = N & 0x3FFFFFFFu;
        const uint I = realN >> 1u;
        const uint rBase = tg * 32u;
        const uint sgi = ti >> 5u;
        const uint lane = ti & 31u;
        const uint r0 = rBase + sgi * 4u;
        if (r0 < I) {
            const uint nBlocks = K >> 5u;
            device const Block_HQ4_G32* g0 = blocks + (size_t)r0 * nBlocks;
            device const Block_HQ4_G32* g1 = (r0 + 1u < I) ? (g0 + nBlocks) : g0;
            device const Block_HQ4_G32* g2 = (r0 + 2u < I) ? (g0 + 2u * nBlocks) : g0;
            device const Block_HQ4_G32* g3 = (r0 + 3u < I) ? (g0 + 3u * nBlocks) : g0;
            device const Block_HQ4_G32* u0 = blocks + (size_t)(I + r0) * nBlocks;
            device const Block_HQ4_G32* u1 = (r0 + 1u < I) ? (u0 + nBlocks) : u0;
            device const Block_HQ4_G32* u2 = (r0 + 2u < I) ? (u0 + 2u * nBlocks) : u0;
            device const Block_HQ4_G32* u3 = (r0 + 3u < I) ? (u0 + 3u * nBlocks) : u0;
            float4 gAcc = float4(0.0f);
            float4 uAcc = float4(0.0f);
            for (uint g = lane; g < nBlocks; g += 32u) {
                device const half* xg = x + (g << 5u);
                half4 lo[4], hi[4];
                #pragma unroll
                for (uint w = 0; w < 4; ++w) {
                    qwg_load_half8(xg + w * 8u, lo[w], hi[w]);
                }
                gAcc.x += qwg_dot_hq4_block_reg(g0[g], lo, hi);
                gAcc.y += qwg_dot_hq4_block_reg(g1[g], lo, hi);
                gAcc.z += qwg_dot_hq4_block_reg(g2[g], lo, hi);
                gAcc.w += qwg_dot_hq4_block_reg(g3[g], lo, hi);
                uAcc.x += qwg_dot_hq4_block_reg(u0[g], lo, hi);
                uAcc.y += qwg_dot_hq4_block_reg(u1[g], lo, hi);
                uAcc.z += qwg_dot_hq4_block_reg(u2[g], lo, hi);
                uAcc.w += qwg_dot_hq4_block_reg(u3[g], lo, hi);
            }
            gAcc += simd_shuffle_down(gAcc, 16);
            gAcc += simd_shuffle_down(gAcc, 8);
            gAcc += simd_shuffle_down(gAcc, 4);
            gAcc += simd_shuffle_down(gAcc, 2);
            gAcc += simd_shuffle_down(gAcc, 1);
            uAcc += simd_shuffle_down(uAcc, 16);
            uAcc += simd_shuffle_down(uAcc, 8);
            uAcc += simd_shuffle_down(uAcc, 4);
            uAcc += simd_shuffle_down(uAcc, 2);
            uAcc += simd_shuffle_down(uAcc, 1);
            if (lane == 0u) {
                const uint c = sgi * 4u;
                tgAct[c + 0u] = half(qwg_swiglu_mul(gAcc.x, uAcc.x));
                tgAct[c + 1u] = half(qwg_swiglu_mul(gAcc.y, uAcc.y));
                tgAct[c + 2u] = half(qwg_swiglu_mul(gAcc.z, uAcc.z));
                tgAct[c + 3u] = half(qwg_swiglu_mul(gAcc.w, uAcc.w));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgi == 0u && rBase + lane < I) {
            h16[rBase + lane] = simd_fwht32(tgAct[lane], lane);
        }
        return;
    }
    if ((N & 0x80000000u) != 0u) {
        const uint realN = N & 0x7FFFFFFFu;
        const uint I = realN >> 1u;
        const uint pairs_per_tg = max(1u, tptg >> 5u);
        const uint sgi = ti >> 5u;
        const uint lane = ti & 31u;
        const uint row = tg * pairs_per_tg + sgi;
        if (row < I) {
            const uint nBlocks = K >> 5u;
            device const Block_HQ4_G32* gRow = blocks + (size_t)row * nBlocks;
            device const Block_HQ4_G32* uRow = blocks + (size_t)(I + row) * nBlocks;
            float gAcc = 0.0f;
            float uAcc = 0.0f;
            for (uint g = lane; g < nBlocks; g += 32u) {
                device const half* xg = x + (g << 5u);
                half4 lo[4], hi[4];
                #pragma unroll
                for (uint w = 0; w < 4; ++w) {
                    qwg_load_half8(xg + w * 8u, lo[w], hi[w]);
                }
                gAcc += qwg_dot_hq4_block_reg(gRow[g], lo, hi);
                uAcc += qwg_dot_hq4_block_reg(uRow[g], lo, hi);
            }
            gAcc += simd_shuffle_down(gAcc, 16);
            gAcc += simd_shuffle_down(gAcc, 8);
            gAcc += simd_shuffle_down(gAcc, 4);
            gAcc += simd_shuffle_down(gAcc, 2);
            gAcc += simd_shuffle_down(gAcc, 1);
            uAcc += simd_shuffle_down(uAcc, 16);
            uAcc += simd_shuffle_down(uAcc, 8);
            uAcc += simd_shuffle_down(uAcc, 4);
            uAcc += simd_shuffle_down(uAcc, 2);
            uAcc += simd_shuffle_down(uAcc, 1);
            if (lane == 0u) {
                h16[row] = half(qwg_swiglu_mul(gAcc, uAcc));
            }
        }
        return;
    }
    threadgroup float part[16];
    qwg_gemv_q4g_silu_pair_h16_body(blocks, x, N, K, h16, tg, ti, part, tptg);
}

// ── qwg_gemv_q4g_dx_h16_t4 / _dual_t4 ────────────────────────────────────
// Under HQ4_G32_M5, nBlocks = K / 32 is an exact multiple of 32 for all engine
// shapes (5120/32=160, 6144/32=192, 17408/32=544), so 1 block per lane per step
kernel void qwg_gemv_q4g_dx_h16_t4(device const Block_HQ4_G32* blocks [[buffer(0)]],
                                   device const half* x               [[buffer(1)]],
                                   device float* y32                  [[buffer(3)]],
                                   constant uint& N                   [[buffer(4)]],
                                   constant uint& K                   [[buffer(5)]],
                                   device half* y16                   [[buffer(8)]],
                                   uint tg [[threadgroup_position_in_grid]],
                                   uint ti [[thread_index_in_threadgroup]],
                                   uint tptg [[threads_per_threadgroup]]) {
    if ((N & 0x40000000u) != 0u) {
        qwg_gemv_q4g_v4_h16_rows4_body<false>(blocks, x, y32, N & 0x3FFFFFFFu, K, y16, tg, ti, tptg);
    } else if ((N & 0x80000000u) != 0u) {
        qwg_gemv_q4g_v4_h16_rows2_body<false>(blocks, x, y32, N & 0x7FFFFFFFu, K, y16, tg, ti, tptg);
    } else {
        qwg_gemv_q4g_v4_h16_body<false>(blocks, x, y32, N, K, y16, tg, ti, tptg);
    }
}

kernel void qwg_gemv_q4g_dx_h16_dual_t4(device const Block_HQ4_G32* blocks [[buffer(0)]],
                                        device const half* x               [[buffer(1)]],
                                        device float* y32                  [[buffer(3)]],
                                        constant uint& N                   [[buffer(4)]],
                                        constant uint& K                   [[buffer(5)]],
                                        device half* y16                   [[buffer(8)]],
                                        uint tg [[threadgroup_position_in_grid]],
                                        uint ti [[thread_index_in_threadgroup]],
                                        uint tptg [[threads_per_threadgroup]]) {
    if ((N & 0x40000000u) != 0u) {
        qwg_gemv_q4g_v4_h16_rows4_body<true>(blocks, x, y32, N & 0x3FFFFFFFu, K, y16, tg, ti, tptg);
    } else if ((N & 0x80000000u) != 0u) {
        qwg_gemv_q4g_v4_h16_rows2_body<true>(blocks, x, y32, N & 0x7FFFFFFFu, K, y16, tg, ti, tptg);
    } else {
        qwg_gemv_q4g_v4_h16_body<true>(blocks, x, y32, N, K, y16, tg, ti, tptg);
    }
}

static inline void qwg_gdn_inproj_store_row(uint row, float vx, device half* y16, device float* y32,
                                            device const float* aLog, device const float* dtBias,
                                            device float* betas, device float* glog) {
    const uint aRow0 = QW_GDN_QKV_CH + QW_GDN_Z_CH;
    const uint bRow0 = aRow0 + QW_GDN_V_HEADS;
    const uint totalN = bRow0 + QW_GDN_V_HEADS;
    if (row < aRow0) {
        y16[row] = half(vx);
    } else if (row < bRow0) {
        const uint h = row - aRow0;
        const float sp = log(1.0f + exp(vx + dtBias[h]));
        glog[h] = -exp(aLog[h]) * sp;
        if (y32) y32[row] = vx;
        y16[row] = half(vx);
    } else if (row < totalN) {
        const uint h = row - bRow0;
        betas[h] = 1.0f / (1.0f + exp(-vx));
        if (y32) y32[row] = vx;
        y16[row] = half(vx);
    }
}

static inline void qwg_gemv_q4g_gdn_inproj_rows4_body(device const Block_HQ4_G32* blocks,
                                                      device const half* x,
                                                      device float* y32,
                                                      uint realN,
                                                      constant uint& K,
                                                      device half* y16,
                                                      device const float* aLog,
                                                      device const float* dtBias,
                                                      device float* betas,
                                                      device float* glog,
                                                      uint tg,
                                                      uint ti,
                                                      uint tptg = 128u) {
    uint simd_per_tg = max(1u, tptg >> 5u);
    uint simd_id = ti >> 5u;
    uint lane = ti & 31u;
    uint row0 = (tg * simd_per_tg + simd_id) * 4u;
    if (row0 >= realN) return;
    float4 acc = qwg_gemv_hq4_rows4(blocks, x, row0, realN, K, lane);
    if (lane == 0u) {
        qwg_gdn_inproj_store_row(row0, acc.x, y16, y32, aLog, dtBias, betas, glog);
        if (row0 + 1u < realN) qwg_gdn_inproj_store_row(row0 + 1u, acc.y, y16, y32, aLog, dtBias, betas, glog);
        if (row0 + 2u < realN) qwg_gdn_inproj_store_row(row0 + 2u, acc.z, y16, y32, aLog, dtBias, betas, glog);
        if (row0 + 3u < realN) qwg_gdn_inproj_store_row(row0 + 3u, acc.w, y16, y32, aLog, dtBias, betas, glog);
    }
}

static inline void qwg_gemv_q4g_gdn_inproj_body(device const Block_HQ4_G32* blocks,
                                                device const half* x,
                                                device float* y32,
                                                constant uint& N,
                                                constant uint& K,
                                                device half* y16,
                                                device const float* aLog,
                                                device const float* dtBias,
                                                device float* betas,
                                                device float* glog,
                                                uint tg,
                                                uint ti,
                                                uint tptg = 32u) {
    uint rows_per_tg = max(1u, tptg >> 5u);
    uint simd_id = ti >> 5u;
    uint lane = ti & 31u;
    uint row = tg * rows_per_tg + simd_id;
    if (row >= N) return;
    float vx = qwg_gemv_hq4_row(blocks, x, row, K, lane);
    if (lane == 0u) {
        qwg_gdn_inproj_store_row(row, vx, y16, y32, aLog, dtBias, betas, glog);
    }
}

kernel void qwg_gemv_q4g_dx_h16_gdn_inproj_t4(device const Block_HQ4_G32* blocks [[buffer(0)]],
                                              device const half* x               [[buffer(1)]],
                                              device float* y32                  [[buffer(3)]],
                                              constant uint& N                   [[buffer(4)]],
                                              constant uint& K                   [[buffer(5)]],
                                              device half* y16                   [[buffer(8)]],
                                              device const float* aLog           [[buffer(9)]],
                                              device const float* dtBias         [[buffer(10)]],
                                              device float* betas                [[buffer(11)]],
                                              device float* glog                 [[buffer(12)]],
                                              uint tg [[threadgroup_position_in_grid]],
                                              uint ti [[thread_index_in_threadgroup]],
                                              uint tptg [[threads_per_threadgroup]]) {
    if ((N & 0x40000000u) != 0u) {
        qwg_gemv_q4g_gdn_inproj_rows4_body(blocks, x, y32, N & 0x3FFFFFFFu, K, y16, aLog, dtBias, betas, glog, tg, ti, tptg);
    } else {
        qwg_gemv_q4g_gdn_inproj_body(blocks, x, y32, N, K, y16, aLog, dtBias, betas, glog, tg, ti, tptg);
    }
}

