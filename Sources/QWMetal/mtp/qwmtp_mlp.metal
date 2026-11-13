#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_geometry_constants.metalh"
#include "include/qw_dequant.metalh"
using namespace metal;

template <uint M>
static inline void qwm_mlp_gate_up(device const Block_HQ4_G32* blocks, device const half* x,
                                   device half* output, uint N, uint K,
                                   uint tg, uint ti, threadgroup float* paired, uint tptg = 64u) {
    const bool isAQ4 = (N & 0x80000000u) != 0u;
    const uint realN = N & 0x7FFFFFFFu;
    const uint I = realN / 2;
    const uint pairs_per_tg = max(1u, tptg >> 6u);
    const uint pair_id = ti >> 6u;
    const uint sub = (ti >> 5u) & 1u;
    const uint lane = ti & 31u;
    const uint row = tg * pairs_per_tg + pair_id;
    if (row < I) {
        const uint weightRow = row + sub * I;
        float value[M];
        qwg_gemv_hq4_row_m<M>(blocks, x, weightRow, K, lane, value, isAQ4);
        if (lane == 0u) {
            #pragma unroll
            for (uint m = 0; m < M; ++m) {
                paired[(pair_id * M + m) * 2u + sub] = value[m];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sub == 0u && lane == 0u && row < I) {
        #pragma unroll
        for (uint m = 0; m < M; ++m) {
            float gate = paired[(pair_id * M + m) * 2u];
            float up   = paired[(pair_id * M + m) * 2u + 1u];
            output[(size_t)m * I + row] = half(qwg_swiglu_mul(gate, up));
        }
    }
}

template <uint M>
static inline void qwm_mlp_gate_up_fwht32(device const Block_HQ4_G32* blocks, device const half* x,
                                          device half* output, uint N, uint K,
                                          uint tg, uint ti, threadgroup half* s_h16) {
    const bool isAQ4 = (N & 0x80000000u) != 0u;
    const uint realN = N & 0x7FFFFFFFu;
    const uint I = realN >> 1u;
    const uint rowBase = tg << 5u;
    if (rowBase >= I) return;
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    const uint pBase = simd_id << 2u;
    #pragma unroll
    for (uint s = 0; s < 2; ++s) {
        const uint p0 = pBase + (s << 1u);
        const uint r0 = rowBase + p0;
        float g0[M], g1[M], u0[M], u1[M];
        qwg_gemv_hq4_rows2_m<M>(blocks, x, r0, realN, K, lane, g0, g1, isAQ4);
        qwg_gemv_hq4_rows2_m<M>(blocks, x, I + r0, realN, K, lane, u0, u1, isAQ4);
        if (lane == 0u) {
            #pragma unroll
            for (uint m = 0; m < M; ++m) {
                s_h16[m * 32u + p0]      = half(qwg_swiglu_mul(g0[m], u0[m]));
                s_h16[m * 32u + p0 + 1u] = half(qwg_swiglu_mul(g1[m], u1[m]));
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id < M && rowBase + lane < I) {
        half val = s_h16[simd_id * 32u + lane];
        output[(size_t)simd_id * I + rowBase + lane] = simd_fwht32(val, lane);
    }
}

template <uint M>
static inline void qwm_t4_dot(device const Block_HQ4_G32* blocks, device const half* x,
                              thread float (&result)[M], uint N, uint K,
                              uint row, uint lane, bool isAQ4 = false) {
    if (row >= N) return;
    qwg_gemv_hq4_row_m<M>(blocks, x, row, K, lane, result, isAQ4);
}

template <uint M>
static inline void qwm_mlp_down(device const Block_HQ4_G32* blocks, device const half* x,
                                device half* output, uint N, uint K,
                                uint tg, uint ti, uint tptg = 32u) {
    const bool isAQ4 = (N & 0x80000000u) != 0u;
    const uint realN = N & 0x7FFFFFFFu;
    const uint rows_per_tg = max(1u, tptg >> 5u);
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    const uint row = tg * rows_per_tg + simd_id;
    if (row >= realN) return;
    float result[M];
    qwm_t4_dot<M>(blocks, x, result, realN, K, row, lane, isAQ4);
    if (lane == 0u) {
        #pragma unroll
        for (uint m = 0; m < M; ++m) output[(size_t)m * realN + row] = half(result[m]);
    }
}

template <uint M>
static inline void qwm_gdn_store_row(uint row, uint N, float value, uint m,
                                     device half* output, device const float* aLog,
                                     device const float* bias, device float* controls) {
    output[(size_t)m * N + row] = half(value);
    const uint aRow0 = QW_GDN_QKV_CH + QW_GDN_Z_CH;
    const uint bRow0 = aRow0 + QW_GDN_V_HEADS;
    const uint bRowEnd = bRow0 + QW_GDN_V_HEADS;
    const uint ctrlStride = 2u * QW_GDN_V_HEADS;
    if (row >= aRow0 && row < bRow0) {
        uint h = row - aRow0;
        float sp = log(1.0f + exp(value + bias[h]));
        controls[m * ctrlStride + QW_GDN_V_HEADS + h] = -exp(aLog[h]) * sp;
    } else if (row >= bRow0 && row < bRowEnd) {
        controls[m * ctrlStride + row - bRow0] = 1.0f / (1.0f + exp(-value));
    }
}

template <uint M>
static inline void qwm_gdn_projection(device const Block_HQ4_G32* blocks, device const half* x,
                                      device half* output, uint N, uint K,
                                      uint tg, uint ti,
                                      device const float* aLog, device const float* bias, device float* controls,
                                      uint tptg = 32u) {
    const bool isAQ4 = (N & 0x80000000u) != 0u;
    const uint realN = N & 0x7FFFFFFFu;
    const uint rows_per_tg = max(1u, tptg >> 5u);
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    const uint row = tg * rows_per_tg + simd_id;
    if (row >= realN) return;
    float result[M];
    qwm_t4_dot<M>(blocks, x, result, realN, K, row, lane, isAQ4);
    if (lane == 0u) {
        #pragma unroll
        for (uint m = 0; m < M; ++m) {
            qwm_gdn_store_row<M>(row, realN, result[m], m, output, aLog, bias, controls);
        }
    }
}

// Row-paired variants (HQ4/AQ4; selected for M >= 4, where one x load per two weight
// rows wins). Per-row arithmetic matches the one-row kernels above bit for bit.
template <uint M>
static inline void qwm_mlp_gate_up_rp(device const Block_HQ4_G32* blocks, device const half* x,
                                      device half* output, uint N, uint K, uint tg, uint ti, uint tptg) {
    const bool isAQ4 = (N & 0x80000000u) != 0u;
    const uint realN = N & 0x7FFFFFFFu;
    const uint I = realN >> 1u;
    const uint nBlocks = K >> 5u;
    const uint row = tg * (tptg >> 5u) + (ti >> 5u);
    const uint lane = ti & 31u;
    if (row >= I) return;
    device const Block_HQ4_G32* rows[2] = {blocks + (size_t)row * nBlocks, blocks + (size_t)(I + row) * nBlocks};
    float v[2][M];
    qwg_gemv_hq4_rows_r_m<M, 2>(rows, x, K, lane, v, isAQ4);
    if (lane == 0u) {
        #pragma unroll
        for (uint m = 0; m < M; ++m) {
            output[(size_t)m * I + row] = half(qwg_swiglu_mul(v[0][m], v[1][m]));
        }
    }
}

template <uint M>
static inline void qwm_rows_rp(device const Block_HQ4_G32* blocks, device const half* x,
                               uint N, uint K, uint tg, uint ti, uint tptg,
                               thread uint& row0, thread uint& realN, thread float (&v)[2][M]) {
    const bool isAQ4 = (N & 0x80000000u) != 0u;
    realN = N & 0x7FFFFFFFu;
    const uint nBlocks = K >> 5u;
    row0 = (tg * (tptg >> 5u) + (ti >> 5u)) * 2u;
    if (row0 >= realN) return;
    device const Block_HQ4_G32* rows[2] = {blocks + (size_t)row0 * nBlocks,
                                           blocks + (size_t)min(row0 + 1u, realN - 1u) * nBlocks};
    qwg_gemv_hq4_rows_r_m<M, 2>(rows, x, K, ti & 31u, v, isAQ4);
}

template <uint M>
static inline void qwm_mlp_down_rp(device const Block_HQ4_G32* blocks, device const half* x,
                                   device half* output, uint N, uint K, uint tg, uint ti, uint tptg) {
    uint row0, realN;
    float v[2][M];
    qwm_rows_rp<M>(blocks, x, N, K, tg, ti, tptg, row0, realN, v);
    if (row0 >= realN || (ti & 31u) != 0u) return;
    #pragma unroll
    for (uint m = 0; m < M; ++m) {
        output[(size_t)m * realN + row0] = half(v[0][m]);
        if (row0 + 1u < realN) output[(size_t)m * realN + row0 + 1u] = half(v[1][m]);
    }
}

template <uint M>
static inline void qwm_gdn_projection_rp(device const Block_HQ4_G32* blocks, device const half* x,
                                         device half* output, uint N, uint K, uint tg, uint ti, uint tptg,
                                         device const float* aLog, device const float* bias, device float* controls) {
    uint row0, realN;
    float v[2][M];
    qwm_rows_rp<M>(blocks, x, N, K, tg, ti, tptg, row0, realN, v);
    if (row0 >= realN || (ti & 31u) != 0u) return;
    #pragma unroll
    for (uint m = 0; m < M; ++m) {
        qwm_gdn_store_row<M>(row0, realN, v[0][m], m, output, aLog, bias, controls);
        if (row0 + 1u < realN) qwm_gdn_store_row<M>(row0 + 1u, realN, v[1][m], m, output, aLog, bias, controls);
    }
}

#define QWM_MLP_ARGUMENTS \
    device const Block_HQ4_G32* blocks [[buffer(0)]], device const half* x [[buffer(1)]], \
    device half* output [[buffer(3)]], \
    constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]], \
    uint row [[threadgroup_position_in_grid]], uint ti [[thread_index_in_threadgroup]], \
    uint tptg [[threads_per_threadgroup]]

kernel void qwm_mlp_gate_up_m2(QWM_MLP_ARGUMENTS) {
    threadgroup float paired[32];
    qwm_mlp_gate_up<2>(blocks, x, output, N, K, row, ti, paired, tptg);
}
kernel void qwm_mlp_gate_up_m3(QWM_MLP_ARGUMENTS) {
    threadgroup float paired[48];
    qwm_mlp_gate_up<3>(blocks, x, output, N, K, row, ti, paired, tptg);
}
kernel void qwm_mlp_gate_up_m4(QWM_MLP_ARGUMENTS) {
    threadgroup float paired[64];
    qwm_mlp_gate_up<4>(blocks, x, output, N, K, row, ti, paired, tptg);
}
kernel void qwm_mlp_gate_up_m5(QWM_MLP_ARGUMENTS) {
    threadgroup float paired[80];
    qwm_mlp_gate_up<5>(blocks, x, output, N, K, row, ti, paired, tptg);
}
kernel void qwm_mlp_gate_up_m6(QWM_MLP_ARGUMENTS) {
    threadgroup float paired[96];
    qwm_mlp_gate_up<6>(blocks, x, output, N, K, row, ti, paired, tptg);
}
kernel void qwm_mlp_gate_up_m7(QWM_MLP_ARGUMENTS) {
    threadgroup float paired[112];
    qwm_mlp_gate_up<7>(blocks, x, output, N, K, row, ti, paired, tptg);
}
kernel void qwm_mlp_gate_up_m8(QWM_MLP_ARGUMENTS) {
    threadgroup float paired[128];
    qwm_mlp_gate_up<8>(blocks, x, output, N, K, row, ti, paired, tptg);
}
kernel void qwm_mlp_gate_up_fwht32_m2(QWM_MLP_ARGUMENTS) {
    (void)tptg;
    threadgroup half s_h16[2 * 32];
    qwm_mlp_gate_up_fwht32<2>(blocks, x, output, N, K, row, ti, s_h16);
}
kernel void qwm_mlp_gate_up_fwht32_m3(QWM_MLP_ARGUMENTS) {
    (void)tptg;
    threadgroup half s_h16[3 * 32];
    qwm_mlp_gate_up_fwht32<3>(blocks, x, output, N, K, row, ti, s_h16);
}
kernel void qwm_mlp_gate_up_fwht32_m4(QWM_MLP_ARGUMENTS) {
    (void)tptg;
    threadgroup half s_h16[4 * 32];
    qwm_mlp_gate_up_fwht32<4>(blocks, x, output, N, K, row, ti, s_h16);
}
kernel void qwm_mlp_gate_up_fwht32_m5(QWM_MLP_ARGUMENTS) {
    (void)tptg;
    threadgroup half s_h16[5 * 32];
    qwm_mlp_gate_up_fwht32<5>(blocks, x, output, N, K, row, ti, s_h16);
}
kernel void qwm_mlp_gate_up_fwht32_m6(QWM_MLP_ARGUMENTS) {
    (void)tptg;
    threadgroup half s_h16[6 * 32];
    qwm_mlp_gate_up_fwht32<6>(blocks, x, output, N, K, row, ti, s_h16);
}
kernel void qwm_mlp_gate_up_fwht32_m7(QWM_MLP_ARGUMENTS) {
    (void)tptg;
    threadgroup half s_h16[7 * 32];
    qwm_mlp_gate_up_fwht32<7>(blocks, x, output, N, K, row, ti, s_h16);
}
kernel void qwm_mlp_gate_up_fwht32_m8(QWM_MLP_ARGUMENTS) {
    (void)tptg;
    threadgroup half s_h16[8 * 32];
    qwm_mlp_gate_up_fwht32<8>(blocks, x, output, N, K, row, ti, s_h16);
}
kernel void qwm_mlp_down_m2(QWM_MLP_ARGUMENTS) {
    qwm_mlp_down<2>(blocks, x, output, N, K, row, ti, tptg);
}
kernel void qwm_mlp_down_m3(QWM_MLP_ARGUMENTS) {
    qwm_mlp_down<3>(blocks, x, output, N, K, row, ti, tptg);
}
kernel void qwm_mlp_down_m4(QWM_MLP_ARGUMENTS) {
    qwm_mlp_down<4>(blocks, x, output, N, K, row, ti, tptg);
}
kernel void qwm_mlp_down_m5(QWM_MLP_ARGUMENTS) {
    qwm_mlp_down<5>(blocks, x, output, N, K, row, ti, tptg);
}
kernel void qwm_mlp_down_m6(QWM_MLP_ARGUMENTS) {
    qwm_mlp_down<6>(blocks, x, output, N, K, row, ti, tptg);
}
kernel void qwm_mlp_down_m7(QWM_MLP_ARGUMENTS) {
    qwm_mlp_down<7>(blocks, x, output, N, K, row, ti, tptg);
}
kernel void qwm_mlp_down_m8(QWM_MLP_ARGUMENTS) {
    qwm_mlp_down<8>(blocks, x, output, N, K, row, ti, tptg);
}
kernel void qwm_gdn_projection_m2(QWM_MLP_ARGUMENTS,
    device const float* aLog [[buffer(9)]], device const float* bias [[buffer(10)]],
    device float* controls [[buffer(11)]]) {
    qwm_gdn_projection<2>(blocks, x, output, N, K, row, ti, aLog, bias, controls, tptg);
}
kernel void qwm_gdn_projection_m3(QWM_MLP_ARGUMENTS,
    device const float* aLog [[buffer(9)]], device const float* bias [[buffer(10)]],
    device float* controls [[buffer(11)]]) {
    qwm_gdn_projection<3>(blocks, x, output, N, K, row, ti, aLog, bias, controls, tptg);
}
kernel void qwm_gdn_projection_m4(QWM_MLP_ARGUMENTS,
    device const float* aLog [[buffer(9)]], device const float* bias [[buffer(10)]],
    device float* controls [[buffer(11)]]) {
    qwm_gdn_projection<4>(blocks, x, output, N, K, row, ti, aLog, bias, controls, tptg);
}
kernel void qwm_gdn_projection_m5(QWM_MLP_ARGUMENTS,
    device const float* aLog [[buffer(9)]], device const float* bias [[buffer(10)]],
    device float* controls [[buffer(11)]]) {
    qwm_gdn_projection<5>(blocks, x, output, N, K, row, ti, aLog, bias, controls, tptg);
}
kernel void qwm_gdn_projection_m6(QWM_MLP_ARGUMENTS,
    device const float* aLog [[buffer(9)]], device const float* bias [[buffer(10)]],
    device float* controls [[buffer(11)]]) {
    qwm_gdn_projection<6>(blocks, x, output, N, K, row, ti, aLog, bias, controls, tptg);
}
kernel void qwm_gdn_projection_m7(QWM_MLP_ARGUMENTS,
    device const float* aLog [[buffer(9)]], device const float* bias [[buffer(10)]],
    device float* controls [[buffer(11)]]) {
    qwm_gdn_projection<7>(blocks, x, output, N, K, row, ti, aLog, bias, controls, tptg);
}
kernel void qwm_gdn_projection_m8(QWM_MLP_ARGUMENTS,
    device const float* aLog [[buffer(9)]], device const float* bias [[buffer(10)]],
    device float* controls [[buffer(11)]]) {
    qwm_gdn_projection<8>(blocks, x, output, N, K, row, ti, aLog, bias, controls, tptg);
}

#define QWM_RP_KERNELS(M) \
kernel void qwm_mlp_gate_up_rp_m##M(QWM_MLP_ARGUMENTS) { \
    qwm_mlp_gate_up_rp<M>(blocks, x, output, N, K, row, ti, tptg); \
} \
kernel void qwm_mlp_down_rp_m##M(QWM_MLP_ARGUMENTS) { \
    qwm_mlp_down_rp<M>(blocks, x, output, N, K, row, ti, tptg); \
} \
kernel void qwm_gdn_projection_rp_m##M(QWM_MLP_ARGUMENTS, \
    device const float* aLog [[buffer(9)]], device const float* bias [[buffer(10)]], \
    device float* controls [[buffer(11)]]) { \
    qwm_gdn_projection_rp<M>(blocks, x, output, N, K, row, ti, tptg, aLog, bias, controls); \
}
QWM_RP_KERNELS(4)
QWM_RP_KERNELS(5)
QWM_RP_KERNELS(6)
QWM_RP_KERNELS(7)
QWM_RP_KERNELS(8)
