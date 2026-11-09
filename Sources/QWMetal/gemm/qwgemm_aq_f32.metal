// qwgemm_aq_f32.metal — Affine-quantized f32 GEMM/GEMV family over QWAQBlock.supported.
// Instantiates compile-time <BITS, GROUP> twins for bm, v4, v4_am, and v4_am2.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "qw_dequant.metalh"
#include "qw_dequant_f32.metalh"
#include "qw_dequant_multi.metalh"
using namespace metal;

#define QWG_AQ_F32_BM_TOKENS 64
#define QWG_AQ_F32_BM_ROWS   32
#define QWG_AQ_F32_BM_LD     36

template <int BITS, int GROUP>
static inline void qwg_gemm_aq_f32_bm_body(device const char*       w,
                                           device const float*      x,
                                           device float*            y,
                                           uint                     M,
                                           uint                     N,
                                           uint                     K,
                                           uint                     epi,
                                           float                    scale,
                                           uint2                    tg,
                                           uint                     tid,
                                           threadgroup float*       As,
                                           threadgroup float*       Bs) {
    threadgroup float* const Cs = As;
    const int t0 = int(tg.y) * QWG_AQ_F32_BM_TOKENS;
    if (t0 >= int(M)) return;
    const int nBase = int(tg.x) * QWG_AQ_F32_BM_ROWS;
    const uint sg = tid >> 5u;
    const uint gi = tid >> 3u;
    const uint dl = tid & 7u;
    const int nBlocks = int(K >> 5u);
    const int nRow = min(nBase + int(gi), int(N) - 1);
    constexpr uint groupBytes = 4u + ((GROUP * BITS) >> 3);
    const uint rowBytes = (K / GROUP) * groupBytes;
    device const char* wrow = w + size_t(nRow) * rowBytes;

    simdgroup_float8x8 acc[4];
    for (int nb = 0; nb < 4; ++nb) acc[nb] = simdgroup_float8x8(0.0f);

    for (int g = 0; g < nBlocks; ++g) {
        #pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int tk = int(gi) + pass * 32;
            const int tokAbs = t0 + tk;
            float4 v = float4(0.0f);
            if (tokAbs < int(M)) {
                device const float* xm = x + long(tokAbs) * long(K) + (g << 5) + int(dl << 2u);
                v = *(device const float4*)xm;
            }
            As[tk * QWG_AQ_F32_BM_LD + int(dl) * 4 + 0] = v.x;
            As[tk * QWG_AQ_F32_BM_LD + int(dl) * 4 + 1] = v.y;
            As[tk * QWG_AQ_F32_BM_LD + int(dl) * 4 + 2] = v.z;
            As[tk * QWG_AQ_F32_BM_LD + int(dl) * 4 + 3] = v.w;
        }
        {
            float2 w01, w23;
            qwg_unpack_aq_lane4_f32<BITS, GROUP>(wrow, uint(g), dl, w01, w23);
            const int base = int(gi) * QWG_AQ_F32_BM_LD + int(dl) * 4;
            Bs[base + 0] = w01.x;
            Bs[base + 1] = w01.y;
            Bs[base + 2] = w23.x;
            Bs[base + 3] = w23.y;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int k8 = 0; k8 < 4; ++k8) {
            simdgroup_float8x8 a;
            simdgroup_load(a, As + int(sg) * 8 * QWG_AQ_F32_BM_LD + k8 * 8, QWG_AQ_F32_BM_LD, ulong2(0, 0), false);
            #pragma unroll
            for (int nb = 0; nb < 4; ++nb) {
                simdgroup_float8x8 b;
                simdgroup_load(b, Bs + nb * 8 * QWG_AQ_F32_BM_LD + k8 * 8, QWG_AQ_F32_BM_LD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc[nb], a, b, acc[nb]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int nb = 0; nb < 4; ++nb)
        simdgroup_store(acc[nb], Cs + int(sg) * 8 * QWG_AQ_F32_BM_LD + nb * 8, QWG_AQ_F32_BM_LD, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = int(tid); idx < QWG_AQ_F32_BM_TOKENS * QWG_AQ_F32_BM_ROWS; idx += 256) {
        const int m = idx / QWG_AQ_F32_BM_ROWS;
        const int n = idx % QWG_AQ_F32_BM_ROWS;
        const int tokAbs = t0 + m, col = nBase + n;
        float v = (tokAbs < int(M) && col < int(N)) ? Cs[m * QWG_AQ_F32_BM_LD + n] : 0.0f;
        if (epi == 1u || epi == 3u) { const float z = v / scale; v = z / (1.0f + exp(-z)); }
        else if (epi == 2u) { v = 1.0f / (1.0f + exp(-v)); }
        if (epi == 3u) {
            v = qwg_fwht32_lane_f32(v, uint(n));
        }
        if (tokAbs < int(M) && col < int(N)) y[long(tokAbs) * long(N) + col] = v;
    }
}

template <int BITS, int GROUP>
static inline void qwg_gemv_aq_f32_v4_body(device const char*  w,
                                           device const float* x,
                                           device float*       y,
                                           uint                N,
                                           uint                K,
                                           uint                epi,
                                           float               scale,
                                           uint3               tg,
                                           uint                sg,
                                           uint                lane,
                                           threadgroup float*  tgC) {
    const uint ep = epi & 0xFFu;
    const uint M = max(1u, (epi >> 8u) & 0xFFu);
    if (ep == 3u) {
        const uint rowBase = tg.x * 32u;
        if (rowBase >= N) return;
        for (uint pass = 0u; pass < 2u; ++pass) {
            const uint row0 = rowBase + pass * 16u + sg * 4u;
            float4 acc = float4(0.0f);
            if (row0 < N) {
                acc = qwg_gemv_aq_rows4_f32<BITS, GROUP>(w, x + (size_t)tg.y * K, row0, N, K, lane);
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
    // K with an odd multiple of 16 slices (K = 2560 is 80) splits 3/2 over a 32-lane sweep; two 16-lane
    // halves, each on its own 4 rows, split it evenly (the host doubles rows per threadgroup to 32).
    const bool halves = ((K >> 5u) & 31u) == 16u;
    if (halves) {
        const uint hl = lane & 15u;
        const uint row0 = (tg.x * 4u + sg) * 8u + (lane >> 4u) * 4u;
        if (row0 >= N) return;
        float4 acc[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
        if (M == 2u) {
            qwg_gemv_aq_rows4_m2_f32<BITS, GROUP, 16u>(w, x, x + K, row0, N, K, hl, acc[0], acc[1]);
        } else if (M > 2u && M <= 4u) {
            qwg_gemv_aq_rows4_m4_f32<BITS, GROUP, 16u>(w, x, M, row0, N, K, hl, acc);
        } else {
            acc[0] = qwg_gemv_aq_rows4_f32<BITS, GROUP, 16u>(w, x + (size_t)tg.y * K, row0, N, K, hl);
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
            qwg_gemv_aq_rows4_m2_f32<BITS, GROUP>(w, x, x + K, row0, N, K, lane, acc[0], acc[1]);
        } else {
            qwg_gemv_aq_rows4_m4_f32<BITS, GROUP>(w, x, M, row0, N, K, lane, acc);
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
    const float4 acc = qwg_gemv_aq_rows4_f32<BITS, GROUP>(w, x + (size_t)tg.y * K, row0, N, K, lane);
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

template <int BITS, int GROUP>
static inline void qwg_gemv_aq_f32_v4_am_body(device const char*  w,
                                              device const float* x,
                                              device float*       y,
                                              device uint*        keysHi,
                                              device uint*        keysLo,
                                              uint                N,
                                              uint                K,
                                              uint                epi,
                                              float               scale,
                                              uint3               tg,
                                              uint                tid,
                                              uint                sg,
                                              uint                lane,
                                              threadgroup uint*   bHi,
                                              threadgroup uint*   bLo) {
    const uint ep = epi & 0xFFu;
    const uint M = epi >> 8u;
    const uint row0 = (tg.x * 4u + sg) * 4u;
    if (M > 1u && M <= 4u) {
        float4 acc[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
        if (row0 < N) {
            if (M == 2u) {
                qwg_gemv_aq_rows4_m2_f32<BITS, GROUP>(w, x, x + K, row0, N, K, lane, acc[0], acc[1]);
            } else {
                qwg_gemv_aq_rows4_m4_f32<BITS, GROUP>(w, x, M, row0, N, K, lane, acc);
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
    const float4 acc = qwg_gemv_aq_rows4_f32<BITS, GROUP>(w, x + (size_t)tg.y * K, row0, N, K, lane);
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

template <int BITS, int GROUP>
static inline void qwg_gemv_aq_f32_v4_am2_body(device const char*  w,
                                               device const float* x,
                                               device uint*        keysHi,
                                               device uint*        keysLo,
                                               uint                N,
                                               uint                K,
                                               device float*       y,
                                               uint3               tg,
                                               uint                tid,
                                               uint                sg,
                                               uint                lane,
                                               threadgroup uint*   bHi0,
                                               threadgroup uint*   bLo0,
                                               threadgroup uint*   bHi1,
                                               threadgroup uint*   bLo1) {
    const uint row0 = (tg.x * 4u + sg) * 4u;
    float4 a0 = float4(0.0f), a1 = float4(0.0f);
    if (row0 < N) {
        qwg_gemv_aq_rows4_m2_f32<BITS, GROUP>(w, x, x + K, row0, N, K, lane, a0, a1);
    }
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

#define QW_AQ_F32_KERNELS(BITS, GROUP) \
kernel void qwg_gemm_aq##BITS##_g##GROUP##_f32_bm(device const char*  w     [[buffer(0)]], \
                                                  device const float* x     [[buffer(1)]], \
                                                  device float*       y     [[buffer(2)]], \
                                                  constant uint&      M     [[buffer(3)]], \
                                                  constant uint&      N     [[buffer(4)]], \
                                                  constant uint&      K     [[buffer(5)]], \
                                                  constant uint&      epi   [[buffer(6)]], \
                                                  constant float&     scale [[buffer(7)]], \
                                                  uint2 tg   [[threadgroup_position_in_grid]], \
                                                  uint  tid  [[thread_index_in_threadgroup]]) { \
    threadgroup float As[QWG_AQ_F32_BM_TOKENS * QWG_AQ_F32_BM_LD]; \
    threadgroup float Bs[QWG_AQ_F32_BM_ROWS * QWG_AQ_F32_BM_LD]; \
    qwg_gemm_aq_f32_bm_body<BITS, GROUP>(w, x, y, M, N, K, epi, scale, tg, tid, As, Bs); \
} \
kernel void qwg_gemv_aq##BITS##_g##GROUP##_f32_v4(device const char*  w     [[buffer(0)]], \
                                                  device const float* x     [[buffer(1)]], \
                                                  device float*       y     [[buffer(2)]], \
                                                  constant uint&      N     [[buffer(3)]], \
                                                  constant uint&      K     [[buffer(4)]], \
                                                  constant uint&      epi   [[buffer(5)]], \
                                                  constant float&     scale [[buffer(6)]], \
                                                  uint3 tg   [[threadgroup_position_in_grid]], \
                                                  uint  sg   [[simdgroup_index_in_threadgroup]], \
                                                  uint  lane [[thread_index_in_simdgroup]]) { \
    threadgroup float tgC[32]; \
    qwg_gemv_aq_f32_v4_body<BITS, GROUP>(w, x, y, N, K, epi, scale, tg, sg, lane, tgC); \
} \
kernel void qwg_gemv_aq##BITS##_g##GROUP##_f32_v4_am(device const char*  w      [[buffer(0)]], \
                                                     device const float* x      [[buffer(1)]], \
                                                     device float*       y      [[buffer(2)]], \
                                                     device uint*        keysHi [[buffer(7)]], \
                                                     device uint*        keysLo [[buffer(8)]], \
                                                     constant uint&      N      [[buffer(3)]], \
                                                     constant uint&      K      [[buffer(4)]], \
                                                     constant uint&      epi    [[buffer(5)]], \
                                                     constant float&     scale  [[buffer(6)]], \
                                                     uint3 tg   [[threadgroup_position_in_grid]], \
                                                     uint  tid  [[thread_index_in_threadgroup]], \
                                                     uint  sg   [[simdgroup_index_in_threadgroup]], \
                                                     uint  lane [[thread_index_in_simdgroup]]) { \
    threadgroup uint bHi[16], bLo[16]; \
    qwg_gemv_aq_f32_v4_am_body<BITS, GROUP>(w, x, y, keysHi, keysLo, N, K, epi, scale, tg, tid, sg, lane, bHi, bLo); \
} \
kernel void qwg_gemv_aq##BITS##_g##GROUP##_f32_v4_am2(device const char*  w      [[buffer(0)]], \
                                                      device const float* x      [[buffer(1)]], \
                                                      device uint*        keysHi [[buffer(2)]], \
                                                      device uint*        keysLo [[buffer(3)]], \
                                                      constant uint&      N      [[buffer(4)]], \
                                                      constant uint&      K      [[buffer(5)]], \
                                                      device float*       y      [[buffer(6)]], \
                                                      uint3 tg   [[threadgroup_position_in_grid]], \
                                                      uint  tid  [[thread_index_in_threadgroup]], \
                                                      uint  sg   [[simdgroup_index_in_threadgroup]], \
                                                      uint  lane [[thread_index_in_simdgroup]]) { \
    threadgroup uint bHi0[4], bLo0[4], bHi1[4], bLo1[4]; \
    qwg_gemv_aq_f32_v4_am2_body<BITS, GROUP>(w, x, keysHi, keysLo, N, K, y, tg, tid, sg, lane, bHi0, bLo0, bHi1, bLo1); \
}

QW_AQ_F32_KERNELS(2, 32)
QW_AQ_F32_KERNELS(2, 64)
QW_AQ_F32_KERNELS(2, 128)
QW_AQ_F32_KERNELS(2, 256)
QW_AQ_F32_KERNELS(3, 32)
QW_AQ_F32_KERNELS(3, 64)
QW_AQ_F32_KERNELS(3, 128)
QW_AQ_F32_KERNELS(3, 256)
QW_AQ_F32_KERNELS(4, 32)
QW_AQ_F32_KERNELS(4, 64)
QW_AQ_F32_KERNELS(4, 128)
QW_AQ_F32_KERNELS(4, 256)
QW_AQ_F32_KERNELS(6, 32)
QW_AQ_F32_KERNELS(6, 64)
QW_AQ_F32_KERNELS(6, 128)
QW_AQ_F32_KERNELS(6, 256)
QW_AQ_F32_KERNELS(8, 32)
QW_AQ_F32_KERNELS(8, 64)
QW_AQ_F32_KERNELS(8, 128)
QW_AQ_F32_KERNELS(8, 256)
