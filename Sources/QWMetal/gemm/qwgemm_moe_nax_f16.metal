// qwgemm_moe_nax_f16.metal: fused expert-sorted down reduction + shared expert combine for the NAX MoE path.
#include <metal_stdlib>
using namespace metal;

// Fused expert-sorted down reduction + shared expert combine:
// out[t, h] = sum_{k=0..<TK} w[t*TK + k] * dnAcc[invPerm[t*TK + k], h] + gate[t] * shared[t, h].
kernel void qwg_moe_down_reduce_combine_f32(device const float* dnAcc [[buffer(0)]],
                                            device const int32_t* invPerm [[buffer(1)]],
                                            device const float* w [[buffer(2)]],
                                            device const float* shared [[buffer(3)]],
                                            device const float* gate [[buffer(4)]],
                                            device float* out [[buffer(5)]],
                                            constant uint& N [[buffer(6)]],
                                            constant uint& TK [[buffer(7)]],
                                            constant uint& tokens [[buffer(8)]],
                                            device half* out16 [[buffer(9)]],
                                            constant uint& hasOut16 [[buffer(10)]],
                                            uint2 tg [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]]) {
    const uint t = tg.y;
    const uint h = (tg.x * 256u + tid) * 4u;
    if (t >= tokens || h >= N) return;
    const uint basePair = t * TK;
    const float g = gate[t];
    const bool dnIsF16 = (hasOut16 & 2u) != 0u;
    const bool writeOut16 = (hasOut16 & 1u) != 0u;
    const bool identityPerm = (hasOut16 & 4u) != 0u;
    device const half* dnAcc16 = (device const half*)dnAcc;
    if (h + 4u <= N) {
        float4 acc = float4(0.0f);
        if (dnIsF16) {
            for (uint k = 0; k < TK; ++k) {
                const int p = identityPerm ? int(basePair + k) : invPerm[basePair + k];
                const float wk = w[basePair + k];
                if (wk != 0.0f) acc += wk * float4(*(device const half4*)(dnAcc16 + size_t(p) * N + h));
            }
        } else {
            for (uint k = 0; k < TK; ++k) {
                const int p = identityPerm ? int(basePair + k) : invPerm[basePair + k];
                const float wk = w[basePair + k];
                if (wk != 0.0f) acc += wk * (*(device const float4*)(dnAcc + size_t(p) * N + h));
            }
        }
        const size_t idx = size_t(t) * N + h;
        const float4 y4 = acc + g * (*(device const float4*)(shared + idx));
        *(device float4*)(out + idx) = y4;
        if (writeOut16) *(device half4*)(out16 + idx) = half4(y4);
    } else {
        for (uint i = 0u; h + i < N; ++i) {
            float acc = 0.0f;
            for (uint k = 0; k < TK; ++k) {
                const int p = identityPerm ? int(basePair + k) : invPerm[basePair + k];
                const float dv = dnIsF16 ? float(dnAcc16[size_t(p) * N + h + i]) : dnAcc[size_t(p) * N + h + i];
                const float wk = w[basePair + k];
                if (wk != 0.0f) acc += wk * dv;
            }
            const size_t idx = size_t(t) * N + h + i;
            const float y = acc + g * shared[idx];
            out[idx] = y;
            if (writeOut16) out16[idx] = half(y);
        }
    }
}
