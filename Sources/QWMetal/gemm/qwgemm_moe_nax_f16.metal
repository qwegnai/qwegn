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
                                            device const int32_t* ids [[buffer(11)]],
                                            uint2 tg [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]]) {
    const uint h = (tg.x * 256u + tid) * 4u;
    if (h >= N) return;
    const bool scatterPerRoute = (hasOut16 & 64u) != 0u;
    if (scatterPerRoute) {
        if (tg.y >= tokens * TK) return;
        const bool dnIsF16 = (hasOut16 & 2u) != 0u;
        const bool writeOut16 = (hasOut16 & 1u) != 0u;
        const bool identityPerm = (hasOut16 & 4u) != 0u;
        const bool skipOut32 = (hasOut16 & 128u) != 0u;
        if (ids[tg.y] < 0) return;
        const int p = identityPerm ? int(tg.y) : invPerm[tg.y];
        device const half* dnAcc16 = (device const half*)dnAcc;
        if (h + 4u <= N) {
            const float4 y4 = dnIsF16
                ? float4(*(device const half4*)(dnAcc16 + size_t(p) * N + h))
                : (*(device const float4*)(dnAcc + size_t(p) * N + h));
            const size_t outIdx = size_t(tg.y) * N + h;
            if (!skipOut32) *(device float4*)(out + outIdx) = y4;
            if (writeOut16) *(device half4*)(out16 + outIdx) = half4(y4);
        } else {
            for (uint i = 0u; h + i < N; ++i) {
                const float y = dnIsF16 ? float(dnAcc16[size_t(p) * N + h + i])
                                        : dnAcc[size_t(p) * N + h + i];
                const size_t outIdx = size_t(tg.y) * N + h + i;
                if (!skipOut32) out[outIdx] = y;
                if (writeOut16) out16[outIdx] = half(y);
            }
        }
        return;
    }
    const uint numGroups = (hasOut16 >> 8u) & 0xFFu;
    const uint t = (numGroups > 0u) ? (tg.y / numGroups) : tg.y;
    const uint targetGroup = (numGroups > 0u) ? (tg.y % numGroups) : 0u;
    if (t >= tokens) return;
    const uint basePair = t * TK;
    const bool dnIsF16 = (hasOut16 & 2u) != 0u;
    const bool writeOut16 = (hasOut16 & 1u) != 0u;
    const bool identityPerm = (hasOut16 & 4u) != 0u;
    const bool skipShared = numGroups > 0u || (hasOut16 & 8u) != 0u;
    const bool unweightedSingle = (hasOut16 & 16u) != 0u;
    const bool hasIdsFilter = numGroups > 0u || (hasOut16 & 32u) != 0u;
    const float g = skipShared ? 0.0f : gate[t];
    device const half* dnAcc16 = (device const half*)dnAcc;
    if (h + 4u <= N) {
        const size_t inIdx = size_t(t) * N + h;
        float4 acc = float4(0.0f);
        if (unweightedSingle) {
            acc = dnIsF16 ? float4(*(device const half4*)(dnAcc16 + inIdx))
                          : (*(device const float4*)(dnAcc + inIdx));
        } else if (dnIsF16) {
            for (uint k = 0; k < TK; ++k) {
                if (hasIdsFilter) {
                    const int rawS = ids[basePair + k];
                    if (rawS < 0) continue;
                    if (numGroups > 0u && ((uint(rawS) >> 16u) & 0x7Fu) != targetGroup) continue;
                }
                const int p = identityPerm ? int(basePair + k) : invPerm[basePair + k];
                const float wk = w[basePair + k];
                if (wk != 0.0f) acc += wk * float4(*(device const half4*)(dnAcc16 + size_t(p) * N + h));
            }
        } else {
            for (uint k = 0; k < TK; ++k) {
                if (hasIdsFilter) {
                    const int rawS = ids[basePair + k];
                    if (rawS < 0) continue;
                    if (numGroups > 0u && ((uint(rawS) >> 16u) & 0x7Fu) != targetGroup) continue;
                }
                const int p = identityPerm ? int(basePair + k) : invPerm[basePair + k];
                const float wk = w[basePair + k];
                if (wk != 0.0f) acc += wk * (*(device const float4*)(dnAcc + size_t(p) * N + h));
            }
        }
        const uint outRow = (numGroups > 0u) ? tg.y : t;
        const size_t idx = size_t(outRow) * N + h;
        const float4 y4 = skipShared ? acc : (acc + g * (*(device const float4*)(shared + idx)));
        *(device float4*)(out + idx) = y4;
        if (writeOut16) *(device half4*)(out16 + idx) = half4(y4);
    } else {
        for (uint i = 0u; h + i < N; ++i) {
            const uint outRow = (numGroups > 0u) ? tg.y : t;
            const size_t idx = size_t(outRow) * N + h + i;
            float acc = 0.0f;
            if (unweightedSingle) {
                acc = dnIsF16 ? float(dnAcc16[idx]) : dnAcc[idx];
            } else {
                for (uint k = 0; k < TK; ++k) {
                    if (hasIdsFilter) {
                        const int rawS = ids[basePair + k];
                        if (rawS < 0) continue;
                        if (numGroups > 0u && ((uint(rawS) >> 16u) & 0x7Fu) != targetGroup) continue;
                    }
                    const int p = identityPerm ? int(basePair + k) : invPerm[basePair + k];
                    const float dv = dnIsF16 ? float(dnAcc16[size_t(p) * N + h + i]) : dnAcc[size_t(p) * N + h + i];
                    const float wk = w[basePair + k];
                    if (wk != 0.0f) acc += wk * dv;
                }
            }
            const float y = skipShared ? acc : (acc + g * shared[idx]);
            out[idx] = y;
            if (writeOut16) out16[idx] = half(y);
        }
    }
}
