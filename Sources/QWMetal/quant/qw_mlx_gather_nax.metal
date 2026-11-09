// qw_mlx_gather_nax.metal — MLX's gather kernel, instantiated for us.
//
// MLX ships the gather as a `[[kernel]]` **template**, so nothing named it in
// the metallib. Upstream instantiates it with BM 32 and BM 64, BN 64, BK 64,
// WM 2, WN 2; both row blocks are here, in upstream's `transpose = true`
// orientation, which is the layout our own bank already has. The design
// note is in `specs/04-qwquant.md`.

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"
#include "mlx/backend/metal/kernels/defines.h"

using namespace metal;

instantiate_kernel("qwg_moe_gather_mlx_nax_bm32", affine_gather_qmm_rhs_nax,
                    half, 64, 4, 32, 64, 64, 2, 2, true);
instantiate_kernel("qwg_moe_gather_mlx_nax_bm64", affine_gather_qmm_rhs_nax,
                    half, 64, 4, 64, 64, 64, 2, 2, true);
