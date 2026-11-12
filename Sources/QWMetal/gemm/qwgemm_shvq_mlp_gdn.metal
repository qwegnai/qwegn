// qwgemm_shvq_mlp_gdn.metal — SwiGLU gate/up GEMV and GDN in-projection
// decode twins for SHVQ{1,15,2,3,4}_G{32,64,128,256}.

#include <metal_stdlib>
#include <metal_simdgroup>
#define QW_SHVQ_ROUTED_DIRECT_LUT 1
#include "qw_dequant_shvq.metalh"
#include "include/qw_geometry_constants.metalh"
using namespace metal;

struct QWSHVQSiluPair {
    float4 gate;
    float4 up;
};

template <int BITS, int GROUP>
static inline QWSHVQSiluPair qwg_shvq_silu_pair_rows4(device const char* blocks,
                                                      device const half* x,
                                                      uint r0, uint I, uint K, uint lane,
                                                      threadgroup const ShvqTgLut<BITS> &tgLut) {
    const uint nBlocks = K >> 5u;
    constexpr uint groupBytes = qwg_shvq_group_bytes(BITS, GROUP);
    const uint rowBytes = (K / uint(GROUP)) * groupBytes;
    device const char* g0 = blocks + (size_t)r0 * rowBytes;
    device const char* g1 = (r0 + 1u < I) ? (g0 + rowBytes) : g0;
    device const char* g2 = (r0 + 2u < I) ? (g0 + 2u * rowBytes) : g0;
    device const char* g3 = (r0 + 3u < I) ? (g0 + 3u * rowBytes) : g0;
    device const char* u0 = blocks + (size_t)(I + r0) * rowBytes;
    device const char* u1 = (r0 + 1u < I) ? (u0 + rowBytes) : u0;
    device const char* u2 = (r0 + 2u < I) ? (u0 + 2u * rowBytes) : u0;
    device const char* u3 = (r0 + 3u < I) ? (u0 + 3u * rowBytes) : u0;
    QWSHVQSiluPair acc = { float4(0.0f), float4(0.0f) };
    for (uint g = lane; g < nBlocks; g += 32u) {
        const uint col = g << 5u;
        half4 xlo[4], xhi[4];
        qwg_shvq_load_half32(x + col, xlo, xhi);
        const size_t groupOff = size_t(col / uint(GROUP)) * groupBytes;
        const uint byteInGroup = qwg_shvq_byte_in_group<BITS, GROUP>(col);
        acc.gate.x += qwg_shvq_slice_dot_tg<BITS>(g0 + groupOff, byteInGroup, xlo, xhi, tgLut);
        acc.gate.y += qwg_shvq_slice_dot_tg<BITS>(g1 + groupOff, byteInGroup, xlo, xhi, tgLut);
        acc.gate.z += qwg_shvq_slice_dot_tg<BITS>(g2 + groupOff, byteInGroup, xlo, xhi, tgLut);
        acc.gate.w += qwg_shvq_slice_dot_tg<BITS>(g3 + groupOff, byteInGroup, xlo, xhi, tgLut);
        acc.up.x += qwg_shvq_slice_dot_tg<BITS>(u0 + groupOff, byteInGroup, xlo, xhi, tgLut);
        acc.up.y += qwg_shvq_slice_dot_tg<BITS>(u1 + groupOff, byteInGroup, xlo, xhi, tgLut);
        acc.up.z += qwg_shvq_slice_dot_tg<BITS>(u2 + groupOff, byteInGroup, xlo, xhi, tgLut);
        acc.up.w += qwg_shvq_slice_dot_tg<BITS>(u3 + groupOff, byteInGroup, xlo, xhi, tgLut);
    }
    acc.gate += simd_shuffle_down(acc.gate, 16);
    acc.gate += simd_shuffle_down(acc.gate, 8);
    acc.gate += simd_shuffle_down(acc.gate, 4);
    acc.gate += simd_shuffle_down(acc.gate, 2);
    acc.gate += simd_shuffle_down(acc.gate, 1);
    acc.up += simd_shuffle_down(acc.up, 16);
    acc.up += simd_shuffle_down(acc.up, 8);
    acc.up += simd_shuffle_down(acc.up, 4);
    acc.up += simd_shuffle_down(acc.up, 2);
    acc.up += simd_shuffle_down(acc.up, 1);
    return acc;
}

template <int BITS, int GROUP>
static inline void qwg_shvq_silu_pair_rows4_m(device const char* blocks,
                                              device const half* x,
                                              device half* h16,
                                              threadgroup half* tgAct,
                                              uint I, uint K, uint mRows,
                                              bool fuseFwht,
                                              uint tg, uint sgi, uint lane,
                                              threadgroup const ShvqTgLut<BITS> &tgLut) {
    const uint rBase = tg * 32u;
    const uint r0 = rBase + sgi * 4u;
    if (r0 < I) {
        const uint nBlocks = K >> 5u;
        constexpr uint groupBytes = qwg_shvq_group_bytes(BITS, GROUP);
        const uint rowBytes = (K / uint(GROUP)) * groupBytes;
        device const char* g0 = blocks + (size_t)r0 * rowBytes;
        device const char* g1 = (r0 + 1u < I) ? (g0 + rowBytes) : g0;
        device const char* g2 = (r0 + 2u < I) ? (g0 + 2u * rowBytes) : g0;
        device const char* g3 = (r0 + 3u < I) ? (g0 + 3u * rowBytes) : g0;
        device const char* u0 = blocks + (size_t)(I + r0) * rowBytes;
        device const char* u1 = (r0 + 1u < I) ? (u0 + rowBytes) : u0;
        device const char* u2 = (r0 + 2u < I) ? (u0 + 2u * rowBytes) : u0;
        device const char* u3 = (r0 + 3u < I) ? (u0 + 3u * rowBytes) : u0;
        float4 gate0 = float4(0.0f), gate1 = float4(0.0f), gate2 = float4(0.0f), gate3 = float4(0.0f);
        float4 up0 = float4(0.0f), up1 = float4(0.0f), up2 = float4(0.0f), up3 = float4(0.0f);
        for (uint g = lane; g < nBlocks; g += 32u) {
            const uint col = g << 5u;
            const size_t groupOff = size_t(col / uint(GROUP)) * groupBytes;
            const uint byteInGroup = qwg_shvq_byte_in_group<BITS, GROUP>(col);
            device const char* gp0 = g0 + groupOff;
            device const char* gp1 = g1 + groupOff;
            device const char* gp2 = g2 + groupOff;
            device const char* gp3 = g3 + groupOff;
            device const char* up_p0 = u0 + groupOff;
            device const char* up_p1 = u1 + groupOff;
            device const char* up_p2 = u2 + groupOff;
            device const char* up_p3 = u3 + groupOff;
            {
                half4 xlo0[4], xhi0[4], xlo1[4], xhi1[4];
                qwg_shvq_load_half32(x + col, xlo0, xhi0);
                qwg_shvq_load_half32(x + (size_t)K + col, xlo1, xhi1);
                const float2 dg0 = qwg_shvq_slice_dot2_tg<BITS>(gp0, byteInGroup, xlo0, xhi0, xlo1, xhi1, tgLut);
                const float2 dg1 = qwg_shvq_slice_dot2_tg<BITS>(gp1, byteInGroup, xlo0, xhi0, xlo1, xhi1, tgLut);
                const float2 dg2 = qwg_shvq_slice_dot2_tg<BITS>(gp2, byteInGroup, xlo0, xhi0, xlo1, xhi1, tgLut);
                const float2 dg3 = qwg_shvq_slice_dot2_tg<BITS>(gp3, byteInGroup, xlo0, xhi0, xlo1, xhi1, tgLut);
                const float2 du0 = qwg_shvq_slice_dot2_tg<BITS>(up_p0, byteInGroup, xlo0, xhi0, xlo1, xhi1, tgLut);
                const float2 du1 = qwg_shvq_slice_dot2_tg<BITS>(up_p1, byteInGroup, xlo0, xhi0, xlo1, xhi1, tgLut);
                const float2 du2 = qwg_shvq_slice_dot2_tg<BITS>(up_p2, byteInGroup, xlo0, xhi0, xlo1, xhi1, tgLut);
                const float2 du3 = qwg_shvq_slice_dot2_tg<BITS>(up_p3, byteInGroup, xlo0, xhi0, xlo1, xhi1, tgLut);
                gate0 += float4(dg0.x, dg1.x, dg2.x, dg3.x);
                gate1 += float4(dg0.y, dg1.y, dg2.y, dg3.y);
                up0 += float4(du0.x, du1.x, du2.x, du3.x);
                up1 += float4(du0.y, du1.y, du2.y, du3.y);
            }
            if (mRows == 3u) {
                half4 xlo2[4], xhi2[4];
                qwg_shvq_load_half32(x + (size_t)2u * K + col, xlo2, xhi2);
                gate2.x += qwg_shvq_slice_dot_tg<BITS>(gp0, byteInGroup, xlo2, xhi2, tgLut);
                gate2.y += qwg_shvq_slice_dot_tg<BITS>(gp1, byteInGroup, xlo2, xhi2, tgLut);
                gate2.z += qwg_shvq_slice_dot_tg<BITS>(gp2, byteInGroup, xlo2, xhi2, tgLut);
                gate2.w += qwg_shvq_slice_dot_tg<BITS>(gp3, byteInGroup, xlo2, xhi2, tgLut);
                up2.x += qwg_shvq_slice_dot_tg<BITS>(up_p0, byteInGroup, xlo2, xhi2, tgLut);
                up2.y += qwg_shvq_slice_dot_tg<BITS>(up_p1, byteInGroup, xlo2, xhi2, tgLut);
                up2.z += qwg_shvq_slice_dot_tg<BITS>(up_p2, byteInGroup, xlo2, xhi2, tgLut);
                up2.w += qwg_shvq_slice_dot_tg<BITS>(up_p3, byteInGroup, xlo2, xhi2, tgLut);
            } else if (mRows >= 4u) {
                half4 xlo2[4], xhi2[4], xlo3[4], xhi3[4];
                qwg_shvq_load_half32(x + (size_t)2u * K + col, xlo2, xhi2);
                qwg_shvq_load_half32(x + (size_t)3u * K + col, xlo3, xhi3);
                const float2 dg0 = qwg_shvq_slice_dot2_tg<BITS>(gp0, byteInGroup, xlo2, xhi2, xlo3, xhi3, tgLut);
                const float2 dg1 = qwg_shvq_slice_dot2_tg<BITS>(gp1, byteInGroup, xlo2, xhi2, xlo3, xhi3, tgLut);
                const float2 dg2 = qwg_shvq_slice_dot2_tg<BITS>(gp2, byteInGroup, xlo2, xhi2, xlo3, xhi3, tgLut);
                const float2 dg3 = qwg_shvq_slice_dot2_tg<BITS>(gp3, byteInGroup, xlo2, xhi2, xlo3, xhi3, tgLut);
                const float2 du0 = qwg_shvq_slice_dot2_tg<BITS>(up_p0, byteInGroup, xlo2, xhi2, xlo3, xhi3, tgLut);
                const float2 du1 = qwg_shvq_slice_dot2_tg<BITS>(up_p1, byteInGroup, xlo2, xhi2, xlo3, xhi3, tgLut);
                const float2 du2 = qwg_shvq_slice_dot2_tg<BITS>(up_p2, byteInGroup, xlo2, xhi2, xlo3, xhi3, tgLut);
                const float2 du3 = qwg_shvq_slice_dot2_tg<BITS>(up_p3, byteInGroup, xlo2, xhi2, xlo3, xhi3, tgLut);
                gate2 += float4(dg0.x, dg1.x, dg2.x, dg3.x);
                gate3 += float4(dg0.y, dg1.y, dg2.y, dg3.y);
                up2 += float4(du0.x, du1.x, du2.x, du3.x);
                up3 += float4(du0.y, du1.y, du2.y, du3.y);
            }
        }
        gate0 += simd_shuffle_down(gate0, 16); gate1 += simd_shuffle_down(gate1, 16);
        gate0 += simd_shuffle_down(gate0, 8);  gate1 += simd_shuffle_down(gate1, 8);
        gate0 += simd_shuffle_down(gate0, 4);  gate1 += simd_shuffle_down(gate1, 4);
        gate0 += simd_shuffle_down(gate0, 2);  gate1 += simd_shuffle_down(gate1, 2);
        gate0 += simd_shuffle_down(gate0, 1);  gate1 += simd_shuffle_down(gate1, 1);
        up0 += simd_shuffle_down(up0, 16); up1 += simd_shuffle_down(up1, 16);
        up0 += simd_shuffle_down(up0, 8);  up1 += simd_shuffle_down(up1, 8);
        up0 += simd_shuffle_down(up0, 4);  up1 += simd_shuffle_down(up1, 4);
        up0 += simd_shuffle_down(up0, 2);  up1 += simd_shuffle_down(up1, 2);
        up0 += simd_shuffle_down(up0, 1);  up1 += simd_shuffle_down(up1, 1);
        if (mRows >= 3u) {
            gate2 += simd_shuffle_down(gate2, 16); up2 += simd_shuffle_down(up2, 16);
            gate2 += simd_shuffle_down(gate2, 8);  up2 += simd_shuffle_down(up2, 8);
            gate2 += simd_shuffle_down(gate2, 4);  up2 += simd_shuffle_down(up2, 4);
            gate2 += simd_shuffle_down(gate2, 2);  up2 += simd_shuffle_down(up2, 2);
            gate2 += simd_shuffle_down(gate2, 1);  up2 += simd_shuffle_down(up2, 1);
        }
        if (mRows >= 4u) {
            gate3 += simd_shuffle_down(gate3, 16); up3 += simd_shuffle_down(up3, 16);
            gate3 += simd_shuffle_down(gate3, 8);  up3 += simd_shuffle_down(up3, 8);
            gate3 += simd_shuffle_down(gate3, 4);  up3 += simd_shuffle_down(up3, 4);
            gate3 += simd_shuffle_down(gate3, 2);  up3 += simd_shuffle_down(up3, 2);
            gate3 += simd_shuffle_down(gate3, 1);  up3 += simd_shuffle_down(up3, 1);
        }
        if (lane == 0u) {
            const uint c = sgi * 4u;
            #pragma unroll
            for (uint r = 0u; r < 4u; ++r) {
                tgAct[0u * 32u + c + r] = half(qwg_swiglu_mul(gate0[r], up0[r]));
                tgAct[1u * 32u + c + r] = half(qwg_swiglu_mul(gate1[r], up1[r]));
                if (mRows >= 3u) tgAct[2u * 32u + c + r] = half(qwg_swiglu_mul(gate2[r], up2[r]));
                if (mRows >= 4u) tgAct[3u * 32u + c + r] = half(qwg_swiglu_mul(gate3[r], up3[r]));
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi < mRows && rBase + lane < I) {
        const half val = tgAct[sgi * 32u + lane];
        h16[(size_t)sgi * I + rBase + lane] = fuseFwht ? simd_fwht32(val, lane) : val;
    }
}

template <int BITS, int GROUP>
static inline float2 qwg_shvq_silu_pair_row1(device const char* blocks,
                                             device const half* x,
                                             uint row, uint I, uint K, uint lane,
                                             threadgroup const ShvqTgLut<BITS> &tgLut) {
    const uint nBlocks = K >> 5u;
    constexpr uint groupBytes = qwg_shvq_group_bytes(BITS, GROUP);
    const uint rowBytes = (K / uint(GROUP)) * groupBytes;
    device const char* gRow = blocks + (size_t)row * rowBytes;
    device const char* uRow = blocks + (size_t)(I + row) * rowBytes;
    float gAcc = 0.0f, uAcc = 0.0f;
    for (uint g = lane; g < nBlocks; g += 32u) {
        const uint col = g << 5u;
        half4 xlo[4], xhi[4];
        qwg_shvq_load_half32(x + col, xlo, xhi);
        const size_t groupOff = size_t(col / uint(GROUP)) * groupBytes;
        const uint byteInGroup = qwg_shvq_byte_in_group<BITS, GROUP>(col);
        gAcc += qwg_shvq_slice_dot_tg<BITS>(gRow + groupOff, byteInGroup, xlo, xhi, tgLut);
        uAcc += qwg_shvq_slice_dot_tg<BITS>(uRow + groupOff, byteInGroup, xlo, xhi, tgLut);
    }
    gAcc += simd_shuffle_down(gAcc, 16); uAcc += simd_shuffle_down(uAcc, 16);
    gAcc += simd_shuffle_down(gAcc, 8);  uAcc += simd_shuffle_down(uAcc, 8);
    gAcc += simd_shuffle_down(gAcc, 4);  uAcc += simd_shuffle_down(uAcc, 4);
    gAcc += simd_shuffle_down(gAcc, 2);  uAcc += simd_shuffle_down(gAcc, 2);
    gAcc += simd_shuffle_down(gAcc, 1);  uAcc += simd_shuffle_down(gAcc, 1);
    return float2(gAcc, uAcc);
}

static inline void qwg_shvq_gdn_store_row(uint row, float vx, device half* y16,
                                          device float* y32, device const float* aLog,
                                          device const float* dtBias, device float* betas,
                                          device float* glog) {
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

template <int BITS, int GROUP>
static inline void qwg_gemv_shvq_gdn_rows4_m_body(device const char* blocks,
                                                  device const half* x,
                                                  uint realN,
                                                  constant uint& K,
                                                  device half* y16,
                                                  device const float* aLog,
                                                  device const float* dtBias,
                                                  device float* controls,
                                                  uint mRows,
                                                  uint tg,
                                                  uint ti,
                                                  uint tptg,
                                                  threadgroup const ShvqTgLut<BITS> &tgLut) {
    const uint simdPerTG = max(1u, tptg >> 5u);
    const uint simdId = ti >> 5u;
    const uint lane = ti & 31u;
    const uint row0 = (tg * simdPerTG + simdId) * 4u;
    if (row0 >= realN) return;
    const uint nBlocks = K >> 5u;
    constexpr uint groupBytes = qwg_shvq_group_bytes(BITS, GROUP);
    const uint rowBytes = (K / uint(GROUP)) * groupBytes;
    device const char* r0 = blocks + (size_t)row0 * rowBytes;
    device const char* r1 = (row0 + 1u < realN) ? (r0 + rowBytes) : r0;
    device const char* r2 = (row0 + 2u < realN) ? (r0 + 2u * rowBytes) : r0;
    device const char* r3 = (row0 + 3u < realN) ? (r0 + 3u * rowBytes) : r0;
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f), acc3 = float4(0.0f);
    for (uint g = lane; g < nBlocks; g += 32u) {
        const uint col = g << 5u;
        const size_t groupOff = size_t(col / uint(GROUP)) * groupBytes;
        const uint byteInGroup = qwg_shvq_byte_in_group<BITS, GROUP>(col);
        device const char* gp0 = r0 + groupOff;
        device const char* gp1 = r1 + groupOff;
        device const char* gp2 = r2 + groupOff;
        device const char* gp3 = r3 + groupOff;
        {
            half4 xlo0[4], xhi0[4], xlo1[4], xhi1[4];
            qwg_shvq_load_half32(x + col, xlo0, xhi0);
            qwg_shvq_load_half32(x + (size_t)K + col, xlo1, xhi1);
            const float2 d0 = qwg_shvq_slice_dot2_tg<BITS>(gp0, byteInGroup, xlo0, xhi0, xlo1, xhi1, tgLut);
            const float2 d1 = qwg_shvq_slice_dot2_tg<BITS>(gp1, byteInGroup, xlo0, xhi0, xlo1, xhi1, tgLut);
            const float2 d2 = qwg_shvq_slice_dot2_tg<BITS>(gp2, byteInGroup, xlo0, xhi0, xlo1, xhi1, tgLut);
            const float2 d3 = qwg_shvq_slice_dot2_tg<BITS>(gp3, byteInGroup, xlo0, xhi0, xlo1, xhi1, tgLut);
            acc0 += float4(d0.x, d1.x, d2.x, d3.x);
            acc1 += float4(d0.y, d1.y, d2.y, d3.y);
        }
        if (mRows == 3u) {
            half4 xlo2[4], xhi2[4];
            qwg_shvq_load_half32(x + (size_t)2u * K + col, xlo2, xhi2);
            acc2.x += qwg_shvq_slice_dot_tg<BITS>(gp0, byteInGroup, xlo2, xhi2, tgLut);
            acc2.y += qwg_shvq_slice_dot_tg<BITS>(gp1, byteInGroup, xlo2, xhi2, tgLut);
            acc2.z += qwg_shvq_slice_dot_tg<BITS>(gp2, byteInGroup, xlo2, xhi2, tgLut);
            acc2.w += qwg_shvq_slice_dot_tg<BITS>(gp3, byteInGroup, xlo2, xhi2, tgLut);
        } else if (mRows >= 4u) {
            half4 xlo2[4], xhi2[4], xlo3[4], xhi3[4];
            qwg_shvq_load_half32(x + (size_t)2u * K + col, xlo2, xhi2);
            qwg_shvq_load_half32(x + (size_t)3u * K + col, xlo3, xhi3);
            const float2 d0 = qwg_shvq_slice_dot2_tg<BITS>(gp0, byteInGroup, xlo2, xhi2, xlo3, xhi3, tgLut);
            const float2 d1 = qwg_shvq_slice_dot2_tg<BITS>(gp1, byteInGroup, xlo2, xhi2, xlo3, xhi3, tgLut);
            const float2 d2 = qwg_shvq_slice_dot2_tg<BITS>(gp2, byteInGroup, xlo2, xhi2, xlo3, xhi3, tgLut);
            const float2 d3 = qwg_shvq_slice_dot2_tg<BITS>(gp3, byteInGroup, xlo2, xhi2, xlo3, xhi3, tgLut);
            acc2 += float4(d0.x, d1.x, d2.x, d3.x);
            acc3 += float4(d0.y, d1.y, d2.y, d3.y);
        }
    }
    acc0 += simd_shuffle_down(acc0, 16); acc1 += simd_shuffle_down(acc1, 16);
    acc0 += simd_shuffle_down(acc0, 8);  acc1 += simd_shuffle_down(acc1, 8);
    acc0 += simd_shuffle_down(acc0, 4);  acc1 += simd_shuffle_down(acc1, 4);
    acc0 += simd_shuffle_down(acc0, 2);  acc1 += simd_shuffle_down(acc1, 2);
    acc0 += simd_shuffle_down(acc0, 1);  acc1 += simd_shuffle_down(acc1, 1);
    if (mRows >= 3u) {
        acc2 += simd_shuffle_down(acc2, 16);
        acc2 += simd_shuffle_down(acc2, 8);
        acc2 += simd_shuffle_down(acc2, 4);
        acc2 += simd_shuffle_down(acc2, 2);
        acc2 += simd_shuffle_down(acc2, 1);
    }
    if (mRows >= 4u) {
        acc3 += simd_shuffle_down(acc3, 16);
        acc3 += simd_shuffle_down(acc3, 8);
        acc3 += simd_shuffle_down(acc3, 4);
        acc3 += simd_shuffle_down(acc3, 2);
        acc3 += simd_shuffle_down(acc3, 1);
    }
    if (lane == 0u) {
        const float4 accs[4] = { acc0, acc1, acc2, acc3 };
        const uint ctrlStride = 2u * QW_GDN_V_HEADS;
        for (uint m = 0u; m < mRows; ++m) {
            device half* ym = y16 + (size_t)m * realN;
            device float* bm = controls + (size_t)m * ctrlStride;
            device float* gm = bm + QW_GDN_V_HEADS;
            #pragma unroll
            for (uint r = 0u; r < 4u; ++r) {
                if (row0 + r < realN) {
                    qwg_shvq_gdn_store_row(row0 + r, accs[m][r], ym, nullptr, aLog, dtBias, bm, gm);
                }
            }
        }
    }
}

#define QW_SHVQ_MLP_GDN_KERNELS(BITS, GROUP)                                           \
    kernel void qwg_gemv_shvq##BITS##_g##GROUP##_silu_pair_dx_h16(                     \
        device const char* blocks [[buffer(0)]],                                       \
        device const half* x      [[buffer(1)]],                                       \
        constant uint&     N      [[buffer(4)]],                                       \
        constant uint&     K      [[buffer(5)]],                                       \
        device half*       h16    [[buffer(8)]],                                       \
        uint tg   [[threadgroup_position_in_grid]],                                    \
        uint ti   [[thread_index_in_threadgroup]],                                     \
        uint tptg [[threads_per_threadgroup]]) {                                       \
        threadgroup ShvqTgLut<BITS> tgLut;                                             \
        if constexpr (BITS <= 3) {                                                     \
            tgLut.load(ti, tptg);                                                      \
            threadgroup_barrier(mem_flags::mem_threadgroup);                           \
        }                                                                              \
        if ((N & 0x40000000u) != 0u) {                                                 \
            const bool fuseFwht = (N & 0x80000000u) != 0u;                             \
            const uint mRows = (N >> 24u) & 0x3Fu;                                     \
            if (mRows >= 2u) {                                                         \
                threadgroup half tgActM[4 * 32];                                       \
                const uint realN = N & 0x00FFFFFFu;                                    \
                qwg_shvq_silu_pair_rows4_m<BITS, GROUP>(                               \
                    blocks, x, h16, tgActM, realN >> 1u, K, mRows, fuseFwht,           \
                    tg, ti >> 5u, ti & 31u, tgLut);                                    \
                return;                                                                \
            }                                                                          \
            threadgroup half tgAct[32];                                                \
            const uint I = (N & 0x3FFFFFFFu) >> 1u;                                    \
            const uint rBase = tg * 32u;                                               \
            const uint sgi = ti >> 5u;                                                 \
            const uint lane = ti & 31u;                                                \
            const uint r0 = rBase + sgi * 4u;                                          \
            if (r0 < I) {                                                              \
                const QWSHVQSiluPair acc = qwg_shvq_silu_pair_rows4<BITS, GROUP>(      \
                    blocks, x, r0, I, K, lane, tgLut);                                 \
                if (lane == 0u) {                                                      \
                    const uint c = sgi * 4u;                                           \
                    tgAct[c + 0u] = half(qwg_swiglu_mul(acc.gate.x, acc.up.x));        \
                    tgAct[c + 1u] = half(qwg_swiglu_mul(acc.gate.y, acc.up.y));        \
                    tgAct[c + 2u] = half(qwg_swiglu_mul(acc.gate.z, acc.up.z));        \
                    tgAct[c + 3u] = half(qwg_swiglu_mul(acc.gate.w, acc.up.w));        \
                }                                                                      \
            }                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                           \
            if (sgi == 0u && rBase + lane < I) {                                       \
                const half val = tgAct[lane];                                          \
                h16[rBase + lane] = fuseFwht ? simd_fwht32(val, lane) : val;           \
            }                                                                          \
            return;                                                                    \
        }                                                                              \
        if ((N & 0x80000000u) != 0u) {                                                 \
            const uint I = (N & 0x7FFFFFFFu) >> 1u;                                    \
            const uint row = tg * max(1u, tptg >> 5u) + (ti >> 5u);                    \
            const uint lane = ti & 31u;                                                \
            if (row < I) {                                                             \
                const float2 acc = qwg_shvq_silu_pair_row1<BITS, GROUP>(               \
                    blocks, x, row, I, K, lane, tgLut);                                \
                if (lane == 0u) h16[row] = half(qwg_swiglu_mul(acc.x, acc.y));         \
            }                                                                          \
            return;                                                                    \
        }                                                                              \
        threadgroup float part[16];                                                    \
        const uint I = (N & 0x3FFFFFFFu) >> 1u;                                        \
        const uint pairId = ti >> 6u;                                                  \
        const uint sub = (ti >> 5u) & 1u;                                              \
        const uint lane = ti & 31u;                                                    \
        const uint row = tg * max(1u, tptg >> 6u) + pairId;                            \
        if (row < I) {                                                                 \
            const uint outRow = (sub == 0u) ? row : (row + I);                         \
            const float v = qwg_gemv_shvq_rowsN_tg<BITS, GROUP, 1u>(                   \
                blocks, x, outRow, outRow + 1u, K, lane, tgLut).x;                     \
            if (lane == 0u) part[pairId * 2u + sub] = v;                               \
        }                                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                               \
        if (sub == 0u && lane == 0u && row < I) {                                      \
            h16[row] = half(qwg_swiglu_mul(part[pairId * 2u], part[pairId * 2u + 1u])); \
        }                                                                              \
    }                                                                                  \
    kernel void qwg_gemv_shvq##BITS##_g##GROUP##_dx_h16_gdn_inproj_t4(                 \
        device const char*  blocks [[buffer(0)]],                                      \
        device const half*  x      [[buffer(1)]],                                      \
        device float*       y32    [[buffer(3)]],                                      \
        constant uint&      N      [[buffer(4)]],                                      \
        constant uint&      K      [[buffer(5)]],                                      \
        device half*        y16    [[buffer(8)]],                                      \
        device const float* aLog   [[buffer(9)]],                                      \
        device const float* dtBias [[buffer(10)]],                                     \
        device float*       betas  [[buffer(11)]],                                     \
        device float*       glog   [[buffer(12)]],                                     \
        uint tg   [[threadgroup_position_in_grid]],                                    \
        uint ti   [[thread_index_in_threadgroup]],                                     \
        uint tptg [[threads_per_threadgroup]]) {                                       \
        threadgroup ShvqTgLut<BITS> tgLut;                                             \
        if constexpr (BITS <= 3) {                                                     \
            tgLut.load(ti, tptg);                                                      \
            threadgroup_barrier(mem_flags::mem_threadgroup);                           \
        }                                                                              \
        if ((N & 0x40000000u) != 0u) {                                                 \
            const uint mRows = (N >> 24u) & 0x3Fu;                                     \
            if (mRows >= 2u) {                                                         \
                qwg_gemv_shvq_gdn_rows4_m_body<BITS, GROUP>(                           \
                    blocks, x, N & 0x00FFFFFFu, K, y16, aLog, dtBias, betas, mRows,    \
                    tg, ti, tptg, tgLut);                                              \
                return;                                                                \
            }                                                                          \
            const uint realN = N & 0x3FFFFFFFu;                                        \
            const uint row0 = (tg * max(1u, tptg >> 5u) + (ti >> 5u)) * 4u;            \
            const uint lane = ti & 31u;                                                \
            if (row0 >= realN) return;                                                 \
            const float4 acc = qwg_gemv_shvq_rowsN_tg<BITS, GROUP, 4u>(                \
                blocks, x, row0, realN, K, lane, tgLut);                               \
            if (lane == 0u) {                                                          \
                qwg_shvq_gdn_store_row(row0, acc.x, y16, y32, aLog, dtBias, betas, glog); \
                if (row0 + 1u < realN) qwg_shvq_gdn_store_row(row0 + 1u, acc.y, y16, y32, aLog, dtBias, betas, glog); \
                if (row0 + 2u < realN) qwg_shvq_gdn_store_row(row0 + 2u, acc.z, y16, y32, aLog, dtBias, betas, glog); \
                if (row0 + 3u < realN) qwg_shvq_gdn_store_row(row0 + 3u, acc.w, y16, y32, aLog, dtBias, betas, glog); \
            }                                                                          \
            return;                                                                    \
        }                                                                              \
        const uint row = tg * max(1u, tptg >> 5u) + (ti >> 5u);                        \
        const uint lane = ti & 31u;                                                    \
        if (row >= N) return;                                                          \
        const float vx = qwg_gemv_shvq_rowsN_tg<BITS, GROUP, 1u>(                      \
            blocks, x, row, N, K, lane, tgLut).x;                                      \
        if (lane == 0u) {                                                              \
            qwg_shvq_gdn_store_row(row, vx, y16, y32, aLog, dtBias, betas, glog);      \
        }                                                                              \
    }

QW_SHVQ_MLP_GDN_KERNELS(1, 32)
QW_SHVQ_MLP_GDN_KERNELS(1, 64)
QW_SHVQ_MLP_GDN_KERNELS(1, 128)
QW_SHVQ_MLP_GDN_KERNELS(1, 256)
QW_SHVQ_MLP_GDN_KERNELS(15, 32)
QW_SHVQ_MLP_GDN_KERNELS(15, 64)
QW_SHVQ_MLP_GDN_KERNELS(15, 128)
QW_SHVQ_MLP_GDN_KERNELS(15, 256)
QW_SHVQ_MLP_GDN_KERNELS(2, 32)
QW_SHVQ_MLP_GDN_KERNELS(2, 64)
QW_SHVQ_MLP_GDN_KERNELS(2, 128)
QW_SHVQ_MLP_GDN_KERNELS(2, 256)
QW_SHVQ_MLP_GDN_KERNELS(3, 32)
QW_SHVQ_MLP_GDN_KERNELS(3, 64)
QW_SHVQ_MLP_GDN_KERNELS(3, 128)
QW_SHVQ_MLP_GDN_KERNELS(3, 256)
QW_SHVQ_MLP_GDN_KERNELS(4, 32)
QW_SHVQ_MLP_GDN_KERNELS(4, 64)
QW_SHVQ_MLP_GDN_KERNELS(4, 128)
QW_SHVQ_MLP_GDN_KERNELS(4, 256)
