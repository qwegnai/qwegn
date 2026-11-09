// qwqsa.metal — QSA indexer on GPU (pooled-index design, exact unbounded 2-stage top-k).
// index_prep: index q heads RMSNorm((1+w)) + partial RoPE; raw index k copied. One TG per (token, head) over 4 q heads
// then one per token for the raw key. pool: mean of 4 raw keys per completed block, RMSNorm, RoPE at block start.
// Stage 1 (qwg_qsa_scores): 2D-grid exact block scoring across all GPU cores into qsaScores [T, scoreStride].
// Stage 2 (qwg_qsa_topk_radix): unbounded O(1)-register 4-pass radix top-512 by (score desc, index desc), ascending list.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_geometry_constants.metalh"
using namespace metal;

// index_prep buffers: 0 src f32 [T,rowStride] | 1 iqNorm f32 [128] | 2 cos f32 [T,64] | 3 sin f32 [T,64]
//                     | 4 idxQ f32 [T,4,128] | 5 idxK f16 [T,128] | 6 T | 7 rowStride | 8 iqOff | 9 ikOff
kernel void qwg_index_prep(device const float* src      [[buffer(0)]],
                           device const float* iqNorm   [[buffer(1)]],
                           device const float* cosT     [[buffer(2)]],
                           device const float* sinT     [[buffer(3)]],
                           device float*       idxQ     [[buffer(4)]],
                           device half*        idxK     [[buffer(5)]],
                           constant uint&      T        [[buffer(6)]],
                           constant uint&      rowStride [[buffer(7)]],
                           constant uint&      iqOff    [[buffer(8)]],
                           constant uint&      ikOff    [[buffer(9)]],
                           uint tg [[threadgroup_position_in_grid]],
                           uint d  [[thread_position_in_threadgroup]]) {
    constexpr uint DI = 128u, ROT = 64u, HALF = 32u, NI = 4u;
    threadgroup float partial[4];
    threadgroup float staged[DI];
    const uint t = tg / (NI + 1u);
    const uint r = tg - t * (NI + 1u);
    const uint rowBase = t * rowStride;
    if (r == NI) {
        idxK[t * DI + d] = half(src[rowBase + ikOff + d]);
        return;
    }
    const float x = src[rowBase + iqOff + r * DI + d];
    const float ss = simd_sum(x * x);
    if ((d & 31u) == 0u) partial[d >> 5u] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = partial[0] + partial[1] + partial[2] + partial[3];
    const float inv = rsqrt(total / float(DI) + QW_RMS_EPS);
    staged[d] = x * inv * (1.0f + iqNorm[d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = staged[d];
    if (d < ROT) {
        const float rot = (d < HALF) ? -staged[d + HALF] : staged[d - HALF];
        o = staged[d] * cosT[t * ROT + d] + rot * sinT[t * ROT + d];
    }
    idxQ[(t * NI + r) * DI + d] = o;
}

// pool buffers: 0 idxK f16 [T,128] | 1 ikNorm f32 [128] | 2 bcos f32 [B,64] | 3 bsin f32 [B,64] | 4 pooled f16 [B,128]
//               | 5 compress (4)
kernel void qwg_index_pool(device const half*  idxK   [[buffer(0)]],
                           device const float* ikNorm [[buffer(1)]],
                           device const float* bcos   [[buffer(2)]],
                           device const float* bsin   [[buffer(3)]],
                           device half*        pooled [[buffer(4)]],
                           constant uint&      compress [[buffer(5)]],
                           uint b [[threadgroup_position_in_grid]],
                           uint d [[thread_position_in_threadgroup]]) {
    constexpr uint DI = 128u, ROT = 64u, HALF = 32u;
    threadgroup float partial[4];
    threadgroup float staged[DI];
    float acc = 0.0f;
    for (uint k = 0u; k < compress; k++) acc += float(idxK[(b * compress + k) * DI + d]);
    const float x = acc / float(compress);
    const float ss = simd_sum(x * x);
    if ((d & 31u) == 0u) partial[d >> 5u] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = partial[0] + partial[1] + partial[2] + partial[3];
    const float inv = rsqrt(total / float(DI) + QW_RMS_EPS);
    staged[d] = x * inv * (1.0f + ikNorm[d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = staged[d];
    if (d < ROT) {
        const float rot = (d < HALF) ? -staged[d + HALF] : staged[d - HALF];
        o = staged[d] * bcos[b * ROT + d] + rot * bsin[b * ROT + d];
    }
    pooled[b * DI + d] = half(o);
}

// Stage 1A — qwg_qsa_scores: 2D grid (blockChunks, T). Each TG of 256 threads stages idxQ[t] in threadgroup memory
// and computes 256 block scores in exact CPU summation order, writing bit-cast uint32 scores to scores[t, b].
// buffers: 0 idxQ [T,4,128] | 1 pooled f16 [B,128] | 2 scores [T,scoreStride] | 3 T | 4 topK | 5 pos0 | 6 scoreStride
kernel void qwg_qsa_scores(device const float* idxQ        [[buffer(0)]],
                           device const half*  pooled      [[buffer(1)]],
                           device uint*        scores      [[buffer(2)]],
                           constant uint&      T           [[buffer(3)]],
                           constant uint&      topK        [[buffer(4)]],
                           constant uint&      pos0        [[buffer(5)]],
                           constant uint&      scoreStride [[buffer(6)]],
                           uint2 tg  [[threadgroup_position_in_grid]],
                           uint  tid [[thread_index_in_threadgroup]]) {
    constexpr uint DI = 128u, NI = 4u;
    threadgroup float qs[NI * DI];
    const uint s = tg.x;
    const uint t = tg.y;
    if (t >= T) return;
    const uint n = (t + pos0 + 1u) / 4u;
    if (n <= topK) return;
    const uint b = (s << 8u) + tid;
    for (uint i = tid; i < NI * DI; i += 256u) qs[i] = idxQ[t * NI * DI + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (b >= n) return;
    float score = 0.0f;
    device const half4* pb = (device const half4*)(pooled + b * DI);
    for (uint h = 0u; h < NI; h++) {
        threadgroup const float4* qh = (threadgroup const float4*)(qs + h * DI);
        float dot = 0.0f;
        #pragma unroll
        for (uint d4 = 0u; d4 < DI / 4u; d4++) {
            const float4 qv = qh[d4], pv = float4(pb[d4]);
            dot += qv.x * pv.x;
            dot += qv.y * pv.y;
            dot += qv.z * pv.z;
            dot += qv.w * pv.w;
        }
        score += max(dot, 0.0f);
    }
    scores[t * scoreStride + b] = as_type<uint>((score / sqrt(float(DI))) + 0.0f);
}

// Stage 1B — qwg_qsa_scores_tiled: 2D grid (blockChunks, ceil(T / 8)).
// Each TG of 256 threads stages pooled[b_0..b_255, 128] (64 KiB f16 -> loaded into registers per thread)
// and computes exact scores for 8 consecutive query rows t_0..t_7, reducing pooled DRAM traffic by 8x in prefill.
kernel void qwg_qsa_scores_tiled(device const float* idxQ        [[buffer(0)]],
                                 device const half*  pooled      [[buffer(1)]],
                                 device uint*        scores      [[buffer(2)]],
                                 constant uint&      T           [[buffer(3)]],
                                 constant uint&      topK        [[buffer(4)]],
                                 constant uint&      pos0        [[buffer(5)]],
                                 constant uint&      scoreStride [[buffer(6)]],
                                 uint2 tg  [[threadgroup_position_in_grid]],
                                 uint  tid [[thread_index_in_threadgroup]]) {
    constexpr uint DI = 128u, NI = 4u, BM = 8u;
    threadgroup float qs[BM * NI * DI];
    const uint s = tg.x;
    const uint tBase = tg.y * BM;
    if (tBase >= T) return;
    const uint b = (s << 8u) + tid;
    const uint maxT = min(tBase + BM, T);
    const uint maxN = (maxT - 1u + pos0 + 1u) / 4u;
    if ((s << 8u) >= maxN) return;
    const uint numRows = maxT - tBase;
    for (uint i = tid; i < numRows * NI * DI; i += 256u) {
        qs[i] = idxQ[tBase * NI * DI + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (b >= maxN) return;
    device const half4* pb = (device const half4*)(pooled + b * DI);
    half4 pv[DI / 4u];
    #pragma unroll
    for (uint d4 = 0u; d4 < DI / 4u; d4++) {
        pv[d4] = pb[d4];
    }
    const float invSqrtD = 1.0f / sqrt(float(DI));
    for (uint r = 0u; r < numRows; r++) {
        const uint t = tBase + r;
        const uint n = (t + pos0 + 1u) / 4u;
        if (n <= topK || b >= n) continue;
        float score = 0.0f;
        for (uint h = 0u; h < NI; h++) {
            threadgroup const float4* qh = (threadgroup const float4*)(qs + (r * NI + h) * DI);
            float dot = 0.0f;
            #pragma unroll
            for (uint d4 = 0u; d4 < DI / 4u; d4++) {
                const float4 qv = qh[d4];
                const float4 pvf = float4(pv[d4]);
                dot += qv.x * pvf.x;
                dot += qv.y * pvf.y;
                dot += qv.z * pvf.z;
                dot += qv.w * pvf.w;
            }
            score += max(dot, 0.0f);
        }
        scores[t * scoreStride + b] = as_type<uint>((score * invSqrtD) + 0.0f);
    }
}

// Parallel replacement for a serial descending scan of hist[255..0]: finds the first bin (from 255 down) where
// greaterAccum + (count of bins above) + hist[bin] >= kEff. Same winner as the serial scan.
static inline void qwg_qsa_pick_bin(threadgroup uint* hist, threadgroup uint* scanSum, threadgroup uint& greaterAccum,
                                    threadgroup uint& chosenBin, threadgroup uint& sgTotal, uint kEff, bool lastPass,
                                    uint tid, uint sg, uint lane) {
    const uint bin = 255u - tid;
    const uint c = hist[bin];
    const uint p = simd_prefix_inclusive_sum(c);
    if (lane == 31u) scanSum[sg] = p;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = greaterAccum;
    for (uint j = 0u; j < sg; j++) base += scanSum[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint incl = base + p;
    const uint excl = incl - c;
    if (excl < kEff && incl >= kEff) {
        greaterAccum = excl;
        chosenBin = bin;
        if (lastPass) sgTotal = incl;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Exclusive scan of cnt[0..<128] into base (all 256 threads must call; ends with a barrier).
static inline void qwg_qsa_scan128(threadgroup uint* cnt, threadgroup uint* base, threadgroup uint* scanSum,
                                   uint tid, uint sg) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint v = (tid < 128u) ? cnt[tid] : 0u;
    const uint p = simd_prefix_exclusive_sum(v);
    const uint tot = simd_sum(v);
    if ((tid & 31u) == 0u) scanSum[sg] = tot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 128u) {
        uint off = 0u;
        for (uint j = 0u; j < sg; j++) off += scanSum[j];
        base[tid] = off + p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Stage 2 — qwg_qsa_topk_radix: One TG (256 threads) per query t. Unbounded O(1)-register 4-pass byte radix
// histogram + ascending compact pass over scores[t, 0..<n] in device memory (L2-resident). Zero MAX_CHUNKS limit.
// buffers: 0 scores u32 [T,scoreStride] | 1 list i32 [T,512] | 2 T | 3 topK (512) | 4 pos0 | 5 scoreStride
kernel void qwg_qsa_topk_radix(device const uint* scores      [[buffer(0)]],
                               device int*        list        [[buffer(1)]],
                               constant uint&     T           [[buffer(2)]],
                               constant uint&     topK        [[buffer(3)]],
                               constant uint&     pos0        [[buffer(4)]],
                               constant uint&     scoreStride [[buffer(5)]],
                               uint t   [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint hist[256];
    threadgroup uint chunkMax[260];
    threadgroup uint sgSum[8];
    threadgroup uint scanSum[8];
    threadgroup uint rankCnt[128];
    threadgroup uint rankBase[128];
    threadgroup uint sgTotal;
    threadgroup uint chosenBin;
    threadgroup uint greaterAccum;
    if (t >= T) return;
    const uint n = (t + pos0 + 1u) / 4u;
    const uint kEff = min(topK, n);
    if (n <= topK) {
        for (uint b = tid; b < n; b += 256u) list[t * topK + b] = int(b);
        return;
    }
    const uint sg = tid >> 5u, lane = tid & 31u;
    const uint numChunks = (n + 255u) >> 8u;
    const bool trackChunkMax = (T <= 8u) && (numChunks > 16u);
    device const uint* rowScores = scores + t * scoreStride;
    if (trackChunkMax) {
        for (uint i = tid; i < 260u; i += 256u) chunkMax[i] = 0u;
    }
    if (tid == 0u) greaterAccum = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint threshU = 0u;
    uint mask = 0u;
    if (numChunks <= 16u) {
        uint myS[16];
        #pragma unroll
        for (uint s = 0u; s < 16u; s++) {
            if (s >= numChunks) break;
            const uint b = (s << 8u) + tid;
            myS[s] = (b < n) ? rowScores[b] : 0u;
        }
        for (uint pass = 0u; pass < 4u; pass++) {
            const uint shift = 24u - (pass << 3u);
            hist[tid] = 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (uint s = 0u; s < 16u; s++) {
                if (s >= numChunks) break;
                const uint b = (s << 8u) + tid;
                const uint u = myS[s];
                const bool active = (b < n) && ((u & mask) == threshU);
                const uint bin = (u >> shift) & 0xFFu;
                const uint refBin = simd_max(active ? bin : 0u);
                const uint isRef = (active && bin == refBin) ? 1u : 0u;
                const uint refCnt = simd_sum(isRef);
                if (lane == 0u && refCnt != 0u) {
                    atomic_fetch_add_explicit((threadgroup atomic_uint*)&hist[refBin], refCnt, memory_order_relaxed);
                }
                if (active && bin != refBin) {
                    atomic_fetch_add_explicit((threadgroup atomic_uint*)&hist[bin], 1u, memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            qwg_qsa_pick_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, kEff, pass == 3u, tid, sg, lane);
            threshU |= (chosenBin << shift);
            mask |= (0xFFu << shift);
        }
        // Register-resident compaction: rank flagged elements in chunk-major (ascending index) order with
        // one scan over the [chunk][simdgroup] counts instead of two barriers per chunk.
        const uint skipTiesR = sgTotal - kEff;
        uint tieMask = 0u;
        if (skipTiesR != 0u) {
            uint isTieMask = 0u;
            #pragma unroll
            for (uint s = 0u; s < 16u; s++) {
                if (s >= numChunks) break;
                const uint b = (s << 8u) + tid;
                const uint f = (b < n && myS[s] == threshU) ? 1u : 0u;
                isTieMask |= f << s;
                const uint c = simd_sum(f);
                if (lane == 0u) rankCnt[s * 8u + sg] = c;
            }
            qwg_qsa_scan128(rankCnt, rankBase, scanSum, tid, sg);
            #pragma unroll
            for (uint s = 0u; s < 16u; s++) {
                if (s >= numChunks) break;
                const uint f = (isTieMask >> s) & 1u;
                const uint r = rankBase[s * 8u + sg] + simd_prefix_exclusive_sum(f);
                if (f != 0u && r >= skipTiesR) tieMask |= 1u << s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint selMask = 0u;
        #pragma unroll
        for (uint s = 0u; s < 16u; s++) {
            if (s >= numChunks) break;
            const uint b = (s << 8u) + tid;
            const uint u = myS[s];
            const uint f = (b < n && (u > threshU || (skipTiesR == 0u ? u == threshU : ((tieMask >> s) & 1u) != 0u))) ? 1u : 0u;
            selMask |= f << s;
            const uint c = simd_sum(f);
            if (lane == 0u) rankCnt[s * 8u + sg] = c;
        }
        qwg_qsa_scan128(rankCnt, rankBase, scanSum, tid, sg);
        #pragma unroll
        for (uint s = 0u; s < 16u; s++) {
            if (s >= numChunks) break;
            const uint f = (selMask >> s) & 1u;
            const uint r = rankBase[s * 8u + sg] + simd_prefix_exclusive_sum(f);
            if (f != 0u) list[t * topK + r] = int((s << 8u) + tid);
        }
        return;
    } else {
        for (uint pass = 0u; pass < 4u; pass++) {
            const uint shift = 24u - (pass << 3u);
            hist[tid] = 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = 0u; s < numChunks; s++) {
                if (trackChunkMax && pass > 0u && s < 260u && chunkMax[s] < threshU) continue;
                const uint b = (s << 8u) + tid;
                const uint u = (b < n) ? rowScores[b] : 0u;
                if (trackChunkMax && pass == 0u && s < 260u) {
                    const uint sgMax = simd_max(u);
                    if (lane == 0u && sgMax != 0u) {
                        atomic_fetch_max_explicit((threadgroup atomic_uint*)&chunkMax[s], sgMax, memory_order_relaxed);
                    }
                }
                const bool active = (b < n) && ((u & mask) == threshU);
                const uint bin = (u >> shift) & 0xFFu;
                const uint refBin = simd_max(active ? bin : 0u);
                const uint isRef = (active && bin == refBin) ? 1u : 0u;
                const uint refCnt = simd_sum(isRef);
                if (lane == 0u && refCnt != 0u) {
                    atomic_fetch_add_explicit((threadgroup atomic_uint*)&hist[refBin], refCnt, memory_order_relaxed);
                }
                if (active && bin != refBin) {
                    atomic_fetch_add_explicit((threadgroup atomic_uint*)&hist[bin], 1u, memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            qwg_qsa_pick_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, kEff, pass == 3u, tid, sg, lane);
            threshU |= (chosenBin << shift);
            mask |= (0xFFu << shift);
        }
    }
    const uint skipTies = sgTotal - kEff;
    uint tiesSeen = 0u;
    uint outCount = 0u;
    if (skipTies == 0u) {
        for (uint s = 0u; s < numChunks; s++) {
            if (trackChunkMax && s < 260u && chunkMax[s] < threshU) continue;
            const uint b = (s << 8u) + tid;
            const uint u = (b < n) ? rowScores[b] : 0u;
            const uint sel = (b < n && u >= threshU) ? 1u : 0u;
            const uint sgSelPref = simd_prefix_exclusive_sum(sel);
            const uint sgSelTot = simd_sum(sel);
            if (lane == 0u) sgSum[sg] = sgSelTot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint chunkSel = sgSum[0] + sgSum[1] + sgSum[2] + sgSum[3]
                                + sgSum[4] + sgSum[5] + sgSum[6] + sgSum[7];
            if (chunkSel != 0u) {
                uint outBase = outCount;
                for (uint j = 0u; j < sg; j++) outBase += sgSum[j];
                if (sel != 0u) list[t * topK + outBase + sgSelPref] = int(b);
                outCount += chunkSel;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (outCount >= kEff) break;
        }
    } else {
        for (uint s = 0u; s < numChunks; s++) {
            if (trackChunkMax && s < 260u && chunkMax[s] < threshU) continue;
            const uint b = (s << 8u) + tid;
            const uint u = (b < n) ? rowScores[b] : 0u;
            const uint isTie = (b < n && u == threshU) ? 1u : 0u;
            const uint sgTiePref = simd_prefix_exclusive_sum(isTie);
            const uint sgTieTot = simd_sum(isTie);
            if (lane == 0u) sgSum[sg] = sgTieTot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint tieBase = tiesSeen;
            for (uint j = 0u; j < sg; j++) tieBase += sgSum[j];
            const uint tieIdx = tieBase + sgTiePref + isTie;
            const uint chunkTies = sgSum[0] + sgSum[1] + sgSum[2] + sgSum[3]
                                 + sgSum[4] + sgSum[5] + sgSum[6] + sgSum[7];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            tiesSeen += chunkTies;
            const uint sel = (b < n && (u > threshU || (u == threshU && tieIdx > skipTies))) ? 1u : 0u;
            const uint sgSelPref = simd_prefix_exclusive_sum(sel);
            const uint sgSelTot = simd_sum(sel);
            if (lane == 0u) sgSum[sg] = sgSelTot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint outBase = outCount;
            for (uint j = 0u; j < sg; j++) outBase += sgSum[j];
            if (sel != 0u) list[t * topK + outBase + sgSelPref] = int(b);
            const uint chunkSel = sgSum[0] + sgSum[1] + sgSum[2] + sgSum[3]
                                + sgSum[4] + sgSum[5] + sgSum[6] + sgSum[7];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            outCount += chunkSel;
            if (outCount >= kEff) break;
        }
    }
}
