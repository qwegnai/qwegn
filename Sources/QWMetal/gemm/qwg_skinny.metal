// QWGEMM F5 skinny GEMM v7 (C05 family, C16 unlock) under HQ4_G32_M5: M ∈ {2,3}, K ≤ 17408.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_dequant.metalh"
using namespace metal;

template<uint M>
static inline void qwg_skinny_body(uint row, uint lane,
                                   device const Block_HQ4_G32* blocks,
                                   constant const half*        x,
                                   device float*               y,
                                   uint N, uint K) {
    float acc[M];
    qwg_gemv_hq4_row_m<M>(blocks, x, row, K, lane, acc);
    if (lane == 0u) {
        for (uint m = 0; m < M; ++m) {
            y[(size_t)m * N + row] = acc[m];
        }
    }
}

kernel void qwg_skinny_m2(device const Block_HQ4_G32* blocks [[buffer(0)]],
                          constant const half*        x      [[buffer(1)]],
                          device float*               y      [[buffer(3)]],
                          constant uint&              N      [[buffer(4)]],
                          constant uint&              K      [[buffer(5)]],
                          uint3 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    if (tg.x < N) qwg_skinny_body<2>(tg.x, tid, blocks, x, y, N, K);
}
kernel void qwg_skinny_m3(device const Block_HQ4_G32* blocks [[buffer(0)]],
                          constant const half*        x      [[buffer(1)]],
                          device float*               y      [[buffer(3)]],
                          constant uint&              N      [[buffer(4)]],
                          constant uint&              K      [[buffer(5)]],
                          uint3 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    if (tg.x < N) qwg_skinny_body<3>(tg.x, tid, blocks, x, y, N, K);
}
