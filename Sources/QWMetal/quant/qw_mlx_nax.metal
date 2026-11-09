// qw_mlx_nax.metal — the MLX affine candidate, as a kernel of ours.
//
// The vendored MLX header is a toolbox of inline templates with no entry
// point of its own, so this file owns the kernel. The include chain is
// MLX's own: utils.h, steel/utils.h, steel/gemm/nax.h, then
// quantized_nax.h — the last uses all three and includes none of them. The
// extents are `constant int&` because qmm_t_nax_tgp_impl binds them so.

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"
using namespace metal;

/// The MLX NAX tile GEMM over a 4-bit, group-64 affine checkpoint.
kernel void qwg_gemm_mlx_affine_nax(device const uint* w [[buffer(0)]],
                                    device const half* scales [[buffer(1)]],
                                    device const half* biases [[buffer(2)]],
                                    device const half* x [[buffer(3)]],
                                    device half* y [[buffer(4)]],
                                    constant int& K [[buffer(5)]],
                                    constant int& N [[buffer(6)]],
                                    constant int& M [[buffer(7)]],
                                    uint lid [[thread_index_in_threadgroup]],
                                    uint simd_gid [[simdgroup_index_in_threadgroup]],
                                    uint simd_lid [[thread_index_in_simdgroup]],
                                    uint3 tid [[threadgroup_position_in_grid]]) {
    // The weight tile: BN rows of BK columns, padded to the loader's stride.
    threadgroup half Ws[64 * 72];
    qmm_t_nax_tgp_impl<half, 64, 4, false>(
        w, scales, biases, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}
