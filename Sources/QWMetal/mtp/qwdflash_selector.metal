// ── QWDFlash Selector — Top-16 candidate extraction & greedy selector ────────
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_fastmath_safe.metalh"
using namespace metal;

static inline float dflash_bf16_to_f32(ushort b) { return as_type<float>(uint(b) << 16u); }

static inline void dflash_tg_argmax(
        float localVal, int localId, uint ti,
        threadgroup float* sVal, threadgroup int* sId,
        thread float& outVal, thread int& outId) {
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        float otherVal = simd_shuffle_down(localVal, offset);
        int otherId = simd_shuffle_down(localId, offset);
        if (otherVal > localVal || (otherVal == localVal && otherId < localId)) {
            localVal = otherVal;
            localId = otherId;
        }
    }
    if (lane == 0u) {
        sVal[simd_id] = localVal;
        sId[simd_id] = localId;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        float v = (lane < 8u) ? sVal[lane] : -3.4e38f;
        int id = (lane < 8u) ? sId[lane] : 0x7fffffff;
        for (ushort offset = 4; offset > 0; offset >>= 1) {
            float ov = simd_shuffle_down(v, offset);
            int oid = simd_shuffle_down(id, offset);
            if (ov > v || (ov == v && oid < id)) {
                v = ov;
                id = oid;
            }
        }
        if (lane == 0u) {
            sVal[0] = v;
            sId[0] = id;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    outVal = sVal[0];
    outId = sId[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Top-16 Stage 1 over vocab logits per draft row (fast register path <= 393216, general loop above).
// Grid = (256, D, 1), threadsPerThreadgroup = (256, 1, 1).
kernel void qwg_dflash_top16_stage1(
        device const float* logits   [[buffer(0)]],
        device float* partialVals    [[buffer(1)]],
        device int* partialIds       [[buffer(2)]],
        constant uint& vocab         [[buffer(3)]],
        constant uint& topK          [[buffer(4)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    const uint chunk = tg.x;
    const uint row = tg.y;
    const uint safeTopK = min(topK, 16u);
    const uint outBase = (row * 256u + chunk) * 16u;
    if (vocab == 0u) {
        if (ti < safeTopK) {
            partialVals[outBase + ti] = -3.4e38f;
            partialIds[outBase + ti] = 0x7fffffff;
        }
        return;
    }
    const uint chunkStride = (vocab + 255u) >> 8u;
    const uint base = chunk * chunkStride;
    const uint chunkEnd = min(vocab, base + chunkStride);
    device const float* rowLogits = logits + (size_t)row * vocab;

    threadgroup float sVal[8];
    threadgroup int sId[8];

    if (chunkStride <= 1536u) {
        float regVal[6];
        int regId[6];
        #pragma unroll
        for (uint i = 0; i < 6u; ++i) {
            uint idx = base + i * 256u + ti;
            if (idx < chunkEnd) {
                float v = rowLogits[idx];
                regVal[i] = qw_is_finite_bits(v) ? max(v, -1e30f) : -1e30f;
                regId[i] = int(idx);
            } else {
                regVal[i] = -3.4e38f;
                regId[i] = 0x7fffffff;
            }
        }

        for (uint k = 0; k < safeTopK; ++k) {
            float bestLocalVal = -3.4e38f;
            int bestLocalId = 0x7fffffff;
            #pragma unroll
            for (uint i = 0; i < 6u; ++i) {
                if (regVal[i] > bestLocalVal || (regVal[i] == bestLocalVal && regId[i] < bestLocalId)) {
                    bestLocalVal = regVal[i];
                    bestLocalId = regId[i];
                }
            }
            float winVal;
            int winId;
            dflash_tg_argmax(bestLocalVal, bestLocalId, ti, sVal, sId, winVal, winId);
            if (ti == 0u) {
                partialVals[outBase + k] = winVal;
                partialIds[outBase + k] = winId;
            }
            #pragma unroll
            for (uint i = 0; i < 6u; ++i) {
                if (regId[i] == winId) {
                    regVal[i] = -3.4e38f;
                    regId[i] = 0x7fffffff;
                }
            }
        }
        return;
    }

    int chosenIds[16];
    for (uint k = 0; k < safeTopK; ++k) {
        float bestLocalVal = -3.4e38f;
        int bestLocalId = 0x7fffffff;
        for (uint idx = base + ti; idx < chunkEnd; idx += 256u) {
            bool already = false;
            for (uint p = 0; p < k; ++p) {
                if (chosenIds[p] == int(idx)) { already = true; break; }
            }
            if (already) continue;
            float raw = rowLogits[idx];
            float v = qw_is_finite_bits(raw) ? max(raw, -1e30f) : -1e30f;
            if (v > bestLocalVal || (v == bestLocalVal && int(idx) < bestLocalId)) {
                bestLocalVal = v;
                bestLocalId = int(idx);
            }
        }
        float winVal;
        int winId;
        dflash_tg_argmax(bestLocalVal, bestLocalId, ti, sVal, sId, winVal, winId);
        chosenIds[k] = winId;
        if (ti == 0u) {
            partialVals[outBase + k] = winVal;
            partialIds[outBase + k] = winId;
        }
    }
}

// Kernel 9: Top-16 Stage 2 reducing 256*16 partial candidates per draft row to topK.
// Grid = (D, 1, 1), threadsPerThreadgroup = (256, 1, 1).
kernel void qwg_dflash_top16_stage2(
        device const float* partialVals [[buffer(0)]],
        device const int* partialIds    [[buffer(1)]],
        device float* unaryOut          [[buffer(2)]],
        device int* candidatesOut       [[buffer(3)]],
        constant uint& topK             [[buffer(4)]],
        constant uint& vocab            [[buffer(5)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    const uint row = tg;
    const uint inBase = row * 256u * 16u;
    const uint safeTopK = min(topK, 16u);

    float regVal[16];
    int regId[16];
    for (uint i = 0; i < 16u; ++i) {
        if (i < safeTopK) {
            float pv = partialVals[inBase + ti * 16u + i];
            int pid = partialIds[inBase + ti * 16u + i];
            if (pid >= 0 && pid < int(vocab) && pid != 0x7fffffff) {
                regVal[i] = qw_is_finite_bits(pv) ? max(pv, -1e30f) : -1e30f;
                regId[i] = pid;
            } else {
                regVal[i] = -3.4e38f;
                regId[i] = 0x7fffffff;
            }
        } else {
            regVal[i] = -3.4e38f;
            regId[i] = 0x7fffffff;
        }
    }

    threadgroup float sVal[8];
    threadgroup int sId[8];
    const uint outBase = row * 16u;

    for (uint k = 0; k < safeTopK; ++k) {
        float bestLocalVal = -3.4e38f;
        int bestLocalId = 0x7fffffff;
        for (uint i = 0; i < 16u; ++i) {
            if (regVal[i] > bestLocalVal || (regVal[i] == bestLocalVal && regId[i] < bestLocalId)) {
                bestLocalVal = regVal[i];
                bestLocalId = regId[i];
            }
        }
        float winVal;
        int winId;
        dflash_tg_argmax(bestLocalVal, bestLocalId, ti, sVal, sId, winVal, winId);
        if (ti == 0u) {
            bool valid = (winId >= 0 && winId < int(vocab) && winId != 0x7fffffff);
            unaryOut[outBase + k] = valid ? (qw_is_finite_bits(winVal) ? max(winVal, -1e30f) : -1e30f) : -1e30f;
            candidatesOut[outBase + k] = valid ? winId : 0;
        }
        for (uint i = 0; i < 16u; ++i) {
            if (regId[i] == winId) {
                regVal[i] = -3.4e38f;
                regId[i] = 0x7fffffff;
            }
        }
    }
}

static inline float dflash_apply_softcap(float x, float cap) {
    if (cap <= 0.0f) return x;
    return cap * tanh(clamp(x / cap, -15.0f, 15.0f));
}

// Greedy sequential pairwise CandidateSelector over BF16 codebooks [vocab, rank].
// Grid = (1, 1, 1), threadsPerThreadgroup = (256, 1, 1).
kernel void qwg_dflash_selector_greedy(
        device const float* hiddenProj   [[buffer(0)]],
        device const float* unary        [[buffer(1)]],
        device const int* candidates     [[buffer(2)]],
        device const ushort* predBook    [[buffer(3)]],
        device const ushort* succBook    [[buffer(4)]],
        device int* draftTokensOut       [[buffer(5)]],
        constant int& anchorToken        [[buffer(6)]],
        constant uint& draftCount        [[buffer(7)]],
        constant uint& topK              [[buffer(8)]],
        constant uint& vocab             [[buffer(9)]],
        constant float& softcap          [[buffer(10)]],
        constant uint& softcapMode       [[buffer(11)]],
        constant uint& rank              [[buffer(12)]],
        uint ti [[thread_index_in_threadgroup]]) {
    if (vocab == 0u) {
        if (ti == 0u) {
            for (uint pos = 0; pos < draftCount; ++pos) draftTokensOut[pos] = 0;
        }
        return;
    }
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    const uint safeTopK = min(topK, 16u);
    threadgroup float sScores[16];
    threadgroup int   sCands[16];
    threadgroup float sSums[8];
    threadgroup int   sWinner;

    int pred = (anchorToken >= 0 && anchorToken < int(vocab) && anchorToken != 0x7fffffff) ? anchorToken : 0;
    const bool fastRank256 = (rank == 256u)
        && (((size_t)predBook & 15u) == 0u)
        && (((size_t)succBook & 15u) == 0u)
        && (((size_t)hiddenProj & 15u) == 0u);

    if (fastRank256) {
        const uint r8 = lane * 8u;
        for (uint pos = 0; pos < draftCount; ++pos) {
            device const float4* hp4 = (device const float4*)(hiddenProj + (size_t)pos * 256u + r8);
            const float4 h0 = hp4[0], h1 = hp4[1];
            const uint4 pb = *((device const uint4*)(predBook + (size_t)pred * 256u + r8));
            const float4 p0 = float4(
                dflash_bf16_to_f32(ushort(pb.x & 0xFFFFu)),
                dflash_bf16_to_f32(ushort(pb.x >> 16u)),
                dflash_bf16_to_f32(ushort(pb.y & 0xFFFFu)),
                dflash_bf16_to_f32(ushort(pb.y >> 16u))) * h0;
            const float4 p1 = float4(
                dflash_bf16_to_f32(ushort(pb.z & 0xFFFFu)),
                dflash_bf16_to_f32(ushort(pb.z >> 16u)),
                dflash_bf16_to_f32(ushort(pb.w & 0xFFFFu)),
                dflash_bf16_to_f32(ushort(pb.w >> 16u))) * h1;

            for (uint pass = 0u; pass < 2u; ++pass) {
                const uint k = simd_id + pass * 8u;
                if (k < safeTopK) {
                    const int rawCand = candidates[pos * 16u + k];
                    const bool validCand = (rawCand >= 0 && rawCand < int(vocab) && rawCand != 0x7fffffff);
                    const int cand = validCand ? rawCand : 0;
                    const uint4 sb = *((device const uint4*)(succBook + (size_t)cand * 256u + r8));
                    const float4 s0 = float4(
                        dflash_bf16_to_f32(ushort(sb.x & 0xFFFFu)),
                        dflash_bf16_to_f32(ushort(sb.x >> 16u)),
                        dflash_bf16_to_f32(ushort(sb.y & 0xFFFFu)),
                        dflash_bf16_to_f32(ushort(sb.y >> 16u)));
                    const float4 s1 = float4(
                        dflash_bf16_to_f32(ushort(sb.z & 0xFFFFu)),
                        dflash_bf16_to_f32(ushort(sb.z >> 16u)),
                        dflash_bf16_to_f32(ushort(sb.w & 0xFFFFu)),
                        dflash_bf16_to_f32(ushort(sb.w >> 16u)));
                    const float edge = simd_sum(dot(p0, s0) + dot(p1, s1));
                    if (lane == 0u) {
                        if (validCand) {
                            float u = unary[pos * 16u + k];
                            const bool ok = qw_is_finite_bits(u) && qw_is_finite_bits(edge);
                            if (softcapMode == 1u && softcap > 0.0f) {
                                u = dflash_apply_softcap(u, softcap);
                            }
                            float rawScore = u + edge;
                            if (softcapMode == 2u && softcap > 0.0f) {
                                rawScore = dflash_apply_softcap(rawScore, softcap);
                            }
                            sScores[k] = (ok && qw_is_finite_bits(rawScore)) ? max(rawScore, -1e30f) : -1e30f;
                            sCands[k] = cand;
                        } else {
                            sScores[k] = -3.4e38f;
                            sCands[k] = 0;
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ti == 0u) {
                float bestScore = -3.4e38f;
                int bestToken = 0;
                for (uint k = 0; k < safeTopK; ++k) {
                    if (sScores[k] > bestScore) {
                        bestScore = sScores[k];
                        bestToken = sCands[k];
                    }
                }
                sWinner = bestToken;
                draftTokensOut[pos] = bestToken;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            pred = sWinner;
        }
        return;
    }
    for (uint pos = 0; pos < draftCount; ++pos) {
        float bestScore = -3.4e38f;
        int bestToken = 0;

        for (uint k = 0; k < safeTopK; ++k) {
            int rawCand = candidates[pos * 16u + k];
            bool validCand = (rawCand >= 0 && rawCand < int(vocab) && rawCand != 0x7fffffff);
            int cand = validCand ? rawCand : 0;
            float localSum = 0.0f;
            for (uint r = ti; r < rank; r += 256u) {
                float hVal = hiddenProj[(size_t)pos * rank + r];
                float pVal = dflash_bf16_to_f32(predBook[(size_t)pred * rank + r]) * hVal;
                float sVal = dflash_bf16_to_f32(succBook[(size_t)cand * rank + r]);
                localSum += pVal * sVal;
            }
            float prod = simd_sum(localSum);
            if (lane == 0u) sSums[simd_id] = prod;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ti == 0u && validCand) {
                float edge = 0.0f;
                for (uint s = 0; s < 8u; ++s) edge += sSums[s];
                float u = unary[pos * 16u + k];
                bool ok = qw_is_finite_bits(u) && qw_is_finite_bits(edge);
                if (softcapMode == 1u && softcap > 0.0f) {
                    u = dflash_apply_softcap(u, softcap);
                }
                float rawScore = u + edge;
                if (softcapMode == 2u && softcap > 0.0f) {
                    rawScore = dflash_apply_softcap(rawScore, softcap);
                }
                float score = (ok && qw_is_finite_bits(rawScore)) ? max(rawScore, -1e30f) : -1e30f;
                if (score > bestScore) {
                    bestScore = score;
                    bestToken = cand;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (ti == 0u) {
            sWinner = bestToken;
            draftTokensOut[pos] = bestToken;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        pred = sWinner;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
