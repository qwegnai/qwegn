#include <metal_stdlib>
using namespace metal;

// Versioned word conversion shared by future device sampling callers.
inline float qw_philox_uniform_v2(uint word) {
    return float(word >> 8) * (1.0f / 16777216.0f);
}

inline uint4 qw_philox4x32_10(ulong counter, ulong seed) {
    uint k0 = uint(seed >> 32), k1 = uint(seed);
    uint4 c = uint4(uint(counter), uint(counter >> 32), 0u, 0u);
    for (uint round = 0; round < 10; ++round) {
        const uint h0 = mulhi(0xD2511F53u, c.x);
        const uint l0 = 0xD2511F53u * c.x;
        const uint h1 = mulhi(0xCD9E8D57u, c.z);
        const uint l1 = 0xCD9E8D57u * c.z;
        c = uint4(h1 ^ c.y ^ k0, l1, h0 ^ c.w ^ k1, l0);
        k0 += 0xCD9E8D57u;
        k1 += 0xD2511F53u;
    }
    return c;
}

kernel void qwg_gumbel_sample_argmax(
    device const float* draftLogits [[buffer(0)]],
    device const float* targetLogits [[buffer(1)]],
    device int* winners [[buffer(2)]],
    constant uint& vocabulary [[buffer(3)]],
    constant ulong& seed [[buffer(4)]],
    constant ulong& position [[buffer(5)]],
    constant float& temperature [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float draftScores[256];
    threadgroup float targetScores[256];
    threadgroup uint draftIds[256];
    threadgroup uint targetIds[256];

    float draftBest = -INFINITY, targetBest = -INFINITY;
    uint draftId = 0u, targetId = 0u;
    for (uint token = tid; token < vocabulary; token += 256u) {
        const ulong counter = position * ulong(vocabulary) + ulong(token);
        const float uniform = max(qw_philox_uniform_v2(qw_philox4x32_10(counter, seed).x),
                                  1.0f / 16777216.0f);
        const float noise = -log(-log(uniform));
        const float draftScore = draftLogits[token] / temperature + noise;
        const float targetScore = targetLogits[token] / temperature + noise;
        if (draftScore > draftBest || (draftScore == draftBest && token < draftId)) {
            draftBest = draftScore;
            draftId = token;
        }
        if (targetScore > targetBest || (targetScore == targetBest && token < targetId)) {
            targetBest = targetScore;
            targetId = token;
        }
    }
    draftScores[tid] = draftBest;
    targetScores[tid] = targetBest;
    draftIds[tid] = draftId;
    targetIds[tid] = targetId;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const uint d = tid + stride;
            if (draftScores[d] > draftScores[tid] ||
                (draftScores[d] == draftScores[tid] && draftIds[d] < draftIds[tid])) {
                draftScores[tid] = draftScores[d];
                draftIds[tid] = draftIds[d];
            }
            if (targetScores[d] > targetScores[tid] ||
                (targetScores[d] == targetScores[tid] && targetIds[d] < targetIds[tid])) {
                targetScores[tid] = targetScores[d];
                targetIds[tid] = targetIds[d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        winners[0] = int(draftIds[0]);
        winners[1] = int(targetIds[0]);
    }
}

