// qwlsa_select.metal — Hierarchical and parallel 4-stage LSA block selection kernels.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

static inline uint lsa_float_to_sortable_u32(float x) {
    if (isnan(x)) return 0u;
    const uint bits = as_type<uint>(x);
    return (bits & 0x80000000u) ? (~bits) : (bits ^ 0x80000000u);
}

static inline void lsa_pick_radix_bin(threadgroup uint* hist,
                                      threadgroup uint* scanSum,
                                      threadgroup uint& greaterAccum,
                                      threadgroup uint& chosenBin,
                                      threadgroup uint& sgTotal,
                                      uint kEff,
                                      bool lastPass,
                                      uint tid,
                                      uint sg,
                                      uint lane) {
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

// Single-TG hierarchical LSA block selector (sink block 0 + top-k middle + local window).
// Shared with `qwg_lsa_select_finalize` fallback for bit-exact parity.
static inline void lsa_select_blocks_body(device const half*   qIdx,
                                          device const half*   pooledKeys,
                                          device const half*   centroids,
                                          device const uchar4* bucketMembers,
                                          device const uchar*  bucketSizes,
                                          device uint*         scoreScratch,
                                          device int*          blockList,
                                          device uint*         blockCount,
                                          uint pos0, uint indexDim, uint budgetBlocks,
                                          uint windowBlocks, uint listStride, uint scoreStride,
                                          uint rowStep, uint maxRow, uint t, uint tid,
                                          threadgroup float* qShared,
                                          threadgroup uint*  hist,
                                          threadgroup uint*  scanSum,
                                          threadgroup uint&  sgTotal,
                                          threadgroup uint&  chosenBin,
                                          threadgroup uint&  greaterAccum) {
    const uint qRow = (rowStep > 1u) ? min(t * rowStep + (rowStep - 1u), maxRow - 1u) : t;
    const uint pq = pos0 + qRow;
    const uint totalBlocks = (pq + 1u) >> 2u;
    const uint maxDenseBlocks = 1u + budgetBlocks + windowBlocks;

    if (totalBlocks <= maxDenseBlocks) {
        for (uint b = tid; b < totalBlocks; b += 256u) {
            blockList[uint64_t(qRow) * listStride + b] = int(b);
        }
        if (tid == 0u) blockCount[qRow] = totalBlocks;
        return;
    }

    const uint sg = tid >> 5u;
    const uint lane = tid & 31u;
    const uint dLim = min(indexDim, 128u);
    if (tid < dLim) {
        qShared[tid] = float(qIdx[uint64_t(qRow) * indexDim + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint middleEnd = totalBlocks - windowBlocks;
    const uint numMiddle = middleEnd - 1u;
    const uint kEff = min(budgetBlocks, numMiddle);
    device uint* rowScores = scoreScratch + uint64_t(t) * scoreStride;

    const uint numClusters = middleEnd >> 7u;
    const uint numBuckets = numClusters * 64u;

    float centroidThresh = -1e30f;
    if (numBuckets > max(256u, kEff * 2u)) {
        hist[tid] = 0u;
        if (tid == 0u) greaterAccum = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint b = tid; b < numBuckets; b += 256u) {
            device const half* cPtr = centroids + uint64_t(b) * indexDim;
            float dot = 0.0f;
            for (uint d = 0u; d < dLim; d++) dot = fma(qShared[d], float(cPtr[d]), dot);
            const uint u = lsa_float_to_sortable_u32(dot);
            atomic_fetch_add_explicit((threadgroup atomic_uint*)&hist[(u >> 24u) & 0xFFu], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint targetBuckets = min(numBuckets, max(128u, kEff));
        lsa_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, targetBuckets, true, tid, sg, lane);

        const uint topByte = chosenBin << 24u;
        hist[tid] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint b = tid; b < numBuckets; b += 256u) {
            device const half* cPtr = centroids + uint64_t(b) * indexDim;
            float dot = 0.0f;
            for (uint d = 0u; d < dLim; d++) dot = fma(qShared[d], float(cPtr[d]), dot);
            const uint u = lsa_float_to_sortable_u32(dot);
            if ((u & 0xFF000000u) == topByte) {
                atomic_fetch_add_explicit((threadgroup atomic_uint*)&hist[(u >> 16u) & 0xFFu], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        lsa_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, targetBuckets, true, tid, sg, lane);
        const uint threshSortable = topByte | (chosenBin << 16u);
        const uint rawBits = (threshSortable & 0x80000000u) ? (threshSortable ^ 0x80000000u) : (~threshSortable);
        centroidThresh = as_type<float>(rawBits);
    }

    if (centroidThresh > -1e20f) {
        for (uint m = 1u + tid; m < numClusters * 128u; m += 256u) rowScores[m - 1u] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint b = tid; b < numBuckets; b += 256u) {
            device const half* cPtr = centroids + uint64_t(b) * indexDim;
            float cDot = 0.0f;
            for (uint d = 0u; d < dLim; d++) cDot = fma(qShared[d], float(cPtr[d]), cDot);
            if (cDot >= centroidThresh) {
                const uint baseBlk = (b >> 6u) * 128u;
                const uchar4 mem = bucketMembers[b];
                const uchar mArr[4] = { mem.x, mem.y, mem.z, mem.w };
                const uint nMem = min(4u, uint(bucketSizes[b]) >> 2u);
                for (uint k = 0u; k < nMem; k++) {
                    const uint m = baseBlk + uint(mArr[k]);
                    if (m >= 1u && m < middleEnd) {
                        device const half* kPtr = pooledKeys + uint64_t(m) * indexDim;
                        float kDot = 0.0f;
                        for (uint d = 0u; d < dLim; d++) kDot = fma(qShared[d], float(kPtr[d]), kDot);
                        rowScores[m - 1u] = lsa_float_to_sortable_u32(kDot);
                    }
                }
            }
        }
        for (uint m = max(1u, numClusters * 128u) + tid; m < middleEnd; m += 256u) {
            device const half* kPtr = pooledKeys + uint64_t(m) * indexDim;
            float kDot = 0.0f;
            for (uint d = 0u; d < dLim; d++) kDot = fma(qShared[d], float(kPtr[d]), kDot);
            rowScores[m - 1u] = lsa_float_to_sortable_u32(kDot);
        }
    } else {
        for (uint m = 1u + tid; m < middleEnd; m += 256u) {
            device const half* kPtr = pooledKeys + uint64_t(m) * indexDim;
            float kDot = 0.0f;
            for (uint d = 0u; d < dLim; d++) kDot = fma(qShared[d], float(kPtr[d]), kDot);
            rowScores[m - 1u] = lsa_float_to_sortable_u32(kDot);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) greaterAccum = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint threshU = 0u;
    uint mask = 0u;
    for (uint pass = 0u; pass < 4u; pass++) {
        const uint shift = 24u - (pass << 3u);
        hist[tid] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < numMiddle; i += 256u) {
            const uint u = rowScores[i];
            if ((u & mask) == threshU) {
                atomic_fetch_add_explicit((threadgroup atomic_uint*)&hist[(u >> shift) & 0xFFu], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        lsa_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, kEff, pass == 3u, tid, sg, lane);
        threshU |= (chosenBin << shift);
        mask |= (0xFFu << shift);
    }

    const uint skipTies = sgTotal - kEff;
    device int* outRow = blockList + uint64_t(qRow) * listStride;
    if (tid == 0u) {
        outRow[0] = 0;
        chosenBin = 1u;
        greaterAccum = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint numChunks = (numMiddle + 255u) >> 8u;
    for (uint c = 0u; c < numChunks; c++) {
        const uint idx = (c << 8u) + tid;
        const uint u = (idx < numMiddle) ? rowScores[idx] : 0u;
        const bool isGt = (idx < numMiddle) && (u > threshU);
        const bool isTie = (idx < numMiddle) && (u == threshU);

        const uint tieRank = simd_prefix_exclusive_sum(isTie ? 1u : 0u);
        const uint tieSum = simd_sum(isTie ? 1u : 0u);
        if (lane == 0u) scanSum[sg] = tieSum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tieBase = greaterAccum;
        for (uint s = 0u; s < sg; s++) tieBase += scanSum[s];
        const bool keepTie = isTie && ((tieBase + tieRank) >= skipTies);
        if (tid == 0u) {
            uint chunkTies = 0u;
            for (uint s = 0u; s < 8u; s++) chunkTies += scanSum[s];
            greaterAccum += chunkTies;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint keep = (isGt || keepTie) ? 1u : 0u;
        const uint wRank = simd_prefix_exclusive_sum(keep);
        const uint wSum = simd_sum(keep);
        if (lane == 0u) scanSum[sg] = wSum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint wBase = chosenBin;
        for (uint s = 0u; s < sg; s++) wBase += scanSum[s];
        if (keep != 0u && (wBase + wRank) <= kEff) {
            outRow[wBase + wRank] = int(idx + 1u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint chunkKept = 0u;
            for (uint s = 0u; s < 8u; s++) chunkKept += scanSum[s];
            chosenBin += chunkKept;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint windowOutBase = 1u + kEff;
    for (uint w = tid; w < windowBlocks; w += 256u) {
        outRow[windowOutBase + w] = int(middleEnd + w);
    }
    if (tid == 0u) {
        blockCount[qRow] = windowOutBase + windowBlocks;
    }
}

kernel void qwg_lsa_select_blocks(device const half*   qIdx          [[buffer(0)]],
                                  device const half*   pooledKeys    [[buffer(1)]],
                                  device const half*   centroids     [[buffer(2)]],
                                  device const uchar4* bucketMembers [[buffer(3)]],
                                  device const uchar*  bucketSizes   [[buffer(4)]],
                                  device uint*         scoreScratch  [[buffer(5)]],
                                  device int*          blockList     [[buffer(6)]],
                                  device uint*         blockCount    [[buffer(7)]],
                                  constant uint&       T             [[buffer(8)]],
                                  constant uint&       pos0          [[buffer(9)]],
                                  constant uint&       indexDim      [[buffer(10)]],
                                  constant uint&       budgetBlocks  [[buffer(11)]],
                                  constant uint&       windowBlocks  [[buffer(12)]],
                                  constant uint&       listStride    [[buffer(13)]],
                                  constant uint&       scoreStride   [[buffer(14)]],
                                  constant uint&       rowStep       [[buffer(15)]],
                                  constant uint&       maxRow        [[buffer(16)]],
                                  uint t   [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
    if (t >= T) return;
    threadgroup float qShared[128];
    threadgroup uint  hist[256];
    threadgroup uint  scanSum[8];
    threadgroup uint  sgTotal;
    threadgroup uint  chosenBin;
    threadgroup uint  greaterAccum;
    lsa_select_blocks_body(qIdx, pooledKeys, centroids, bucketMembers, bucketSizes, scoreScratch,
                           blockList, blockCount, pos0, indexDim, budgetBlocks, windowBlocks,
                           listStride, scoreStride, rowStep, maxRow, t, tid,
                           qShared, hist, scanSum, sgTotal, chosenBin, greaterAccum);
}

// Parallel 4-stage selector (`_score` -> `_hist2` -> `_expand` -> `_finalize`),
// bit-identical to `qwg_lsa_select_blocks` with automatic fallback inside `_finalize`.
struct LSASelParams {
    uint T;
    uint pos0;
    uint indexDim;
    uint budgetBlocks;
    uint windowBlocks;
    uint listStride;
    uint scoreStride;
    uint rowStep;
    uint maxRow;
    uint bucketStride;
    uint candCap;
    uint markStride;
};

struct LSASelGeom {
    uint qRow;
    uint totalBlocks;
    uint middleEnd;
    uint numMiddle;
    uint kEff;
    uint numClusters;
    uint numBuckets;
    uint targetBuckets;
    bool filter;
};

static inline LSASelGeom lsa_sel_geom(constant LSASelParams& p, uint unit) {
    LSASelGeom g;
    g.qRow = (p.rowStep > 1u) ? min(unit * p.rowStep + (p.rowStep - 1u), p.maxRow - 1u) : unit;
    g.totalBlocks = (p.pos0 + g.qRow + 1u) >> 2u;
    g.filter = false;
    g.middleEnd = 0u; g.numMiddle = 0u; g.kEff = 0u;
    g.numClusters = 0u; g.numBuckets = 0u; g.targetBuckets = 0u;
    if (g.totalBlocks <= 1u + p.budgetBlocks + p.windowBlocks) return g;
    g.middleEnd = g.totalBlocks - p.windowBlocks;
    g.numMiddle = g.middleEnd - 1u;
    g.kEff = min(p.budgetBlocks, g.numMiddle);
    g.numClusters = g.middleEnd >> 7u;
    g.numBuckets = g.numClusters * 64u;
    g.targetBuckets = min(g.numBuckets, max(128u, g.kEff));
    g.filter = g.numBuckets > max(256u, g.kEff * 2u);
    return g;
}

static inline float lsa_sortable_to_float(uint u) {
    return as_type<float>((u & 0x80000000u) ? (u ^ 0x80000000u) : (~u));
}

static inline void lsa_load_q(device const half* qIdx, uint qRow, uint indexDim, uint tid,
                              threadgroup float* qShared) {
    const uint dLim = min(indexDim, 128u);
    if (tid < dLim) qShared[tid] = float(qIdx[uint64_t(qRow) * indexDim + tid]);
}

// Replays the two-byte centroid threshold from device histograms.
static inline uint lsa_pick_centroid_bins(device const uint* uHist, bool second, uint target,
                                          uint tid, uint sg, uint lane,
                                          threadgroup uint* hist, threadgroup uint* scanSum,
                                          threadgroup uint& greaterAccum, threadgroup uint& chosenBin,
                                          threadgroup uint& sgTotal) {
    hist[tid] = uHist[tid];
    if (tid == 0u) greaterAccum = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lsa_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, target, true, tid, sg, lane);
    const uint topByte = chosenBin << 24u;
    if (!second) return topByte;
    hist[tid] = uHist[256u + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lsa_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, target, true, tid, sg, lane);
    return topByte | (chosenBin << 16u);
}

kernel void qwg_lsa_select_score(device const half* qIdx      [[buffer(0)]],
                                 device const half* centroids [[buffer(1)]],
                                 device uint*       cScore    [[buffer(2)]],
                                 device atomic_uint* uHist    [[buffer(3)]],
                                 constant LSASelParams& p     [[buffer(4)]],
                                 uint2 tg  [[threadgroup_position_in_grid]],
                                 uint  tid [[thread_index_in_threadgroup]]) {
    const uint unit = tg.y;
    if (unit >= p.T) return;
    const LSASelGeom g = lsa_sel_geom(p, unit);
    if (!g.filter || tg.x * 256u >= g.numBuckets) return;
    threadgroup float qShared[128];
    threadgroup atomic_uint hist[256];
    lsa_load_q(qIdx, g.qRow, p.indexDim, tid, qShared);
    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint b = tg.x * 256u + tid;
    if (b < g.numBuckets) {
        const uint dLim = min(p.indexDim, 128u);
        device const half* cPtr = centroids + uint64_t(b) * p.indexDim;
        float dot = 0.0f;
        for (uint d = 0u; d < dLim; d++) dot = fma(qShared[d], float(cPtr[d]), dot);
        const uint u = lsa_float_to_sortable_u32(dot);
        cScore[uint64_t(unit) * p.bucketStride + b] = u;
        atomic_fetch_add_explicit(&hist[(u >> 24u) & 0xFFu], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint c = atomic_load_explicit(&hist[tid], memory_order_relaxed);
    if (c != 0u) atomic_fetch_add_explicit(&uHist[unit * 512u + tid], c, memory_order_relaxed);
}

kernel void qwg_lsa_select_hist2(device const uint*  cScore [[buffer(0)]],
                                 device atomic_uint* uHist  [[buffer(1)]],
                                 constant LSASelParams& p   [[buffer(2)]],
                                 uint2 tg  [[threadgroup_position_in_grid]],
                                 uint  tid [[thread_index_in_threadgroup]]) {
    const uint unit = tg.y;
    if (unit >= p.T) return;
    const LSASelGeom g = lsa_sel_geom(p, unit);
    if (!g.filter || tg.x * 256u >= g.numBuckets) return;
    threadgroup uint hist[256];
    threadgroup uint scanSum[8];
    threadgroup uint sgTotal, chosenBin, greaterAccum;
    const uint sg = tid >> 5u, lane = tid & 31u;
    device const uint* uh = (device const uint*)(uHist) + unit * 512u;
    const uint topByte = lsa_pick_centroid_bins(uh, false, g.targetBuckets, tid, sg, lane,
                                                hist, scanSum, greaterAccum, chosenBin, sgTotal);
    hist[tid] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint b = tg.x * 256u + tid;
    if (b < g.numBuckets) {
        const uint u = cScore[uint64_t(unit) * p.bucketStride + b];
        if ((u & 0xFF000000u) == topByte) {
            atomic_fetch_add_explicit((threadgroup atomic_uint*)&hist[(u >> 16u) & 0xFFu], 1u,
                                      memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (hist[tid] != 0u) atomic_fetch_add_explicit(&uHist[unit * 512u + 256u + tid], hist[tid],
                                                   memory_order_relaxed);
}

static inline void lsa_add_candidate(uint unit, uint m, float kDot, constant LSASelParams& p,
                                     device atomic_uint* marks, device atomic_uint* candCount,
                                     device uint* candM, device uint* candU) {
    const uint bit = 1u << (m & 31u);
    if ((atomic_fetch_or_explicit(&marks[uint64_t(unit) * p.markStride + (m >> 5u)], bit,
                                  memory_order_relaxed) & bit) != 0u) {
        return;
    }
    const uint slot = atomic_fetch_add_explicit(&candCount[unit], 1u, memory_order_relaxed);
    if (slot < p.candCap) {
        candM[uint64_t(unit) * p.candCap + slot] = m;
        candU[uint64_t(unit) * p.candCap + slot] = lsa_float_to_sortable_u32(kDot);
    }
}

kernel void qwg_lsa_select_expand(device const half*   qIdx          [[buffer(0)]],
                                  device const half*   pooledKeys    [[buffer(1)]],
                                  device const uchar4* bucketMembers [[buffer(2)]],
                                  device const uchar*  bucketSizes   [[buffer(3)]],
                                  device const uint*   cScore        [[buffer(4)]],
                                  device const uint*   uHist         [[buffer(5)]],
                                  device atomic_uint*  marks         [[buffer(6)]],
                                  device atomic_uint*  candCount     [[buffer(7)]],
                                  device uint*         candM         [[buffer(8)]],
                                  device uint*         candU         [[buffer(9)]],
                                  constant LSASelParams& p           [[buffer(10)]],
                                  uint2 tg  [[threadgroup_position_in_grid]],
                                  uint  tid [[thread_index_in_threadgroup]]) {
    const uint unit = tg.y;
    if (unit >= p.T) return;
    const LSASelGeom g = lsa_sel_geom(p, unit);
    if (!g.filter || tg.x * 256u >= g.numBuckets) return;
    threadgroup float qShared[128];
    threadgroup uint hist[256];
    threadgroup uint scanSum[8];
    threadgroup uint sgTotal, chosenBin, greaterAccum;
    const uint sg = tid >> 5u, lane = tid & 31u;
    lsa_load_q(qIdx, g.qRow, p.indexDim, tid, qShared);
    const uint threshSortable = lsa_pick_centroid_bins(uHist + unit * 512u, true, g.targetBuckets,
                                                       tid, sg, lane, hist, scanSum, greaterAccum,
                                                       chosenBin, sgTotal);
    const float centroidThresh = lsa_sortable_to_float(threshSortable);
    if (!(centroidThresh > -1e20f)) return;

    const uint dLim = min(p.indexDim, 128u);
    const uint b = tg.x * 256u + tid;
    if (b < g.numBuckets) {
        const float cDot = lsa_sortable_to_float(cScore[uint64_t(unit) * p.bucketStride + b]);
        if (cDot >= centroidThresh) {
            const uint baseBlk = (b >> 6u) * 128u;
            const uchar4 mem = bucketMembers[b];
            const uchar mArr[4] = { mem.x, mem.y, mem.z, mem.w };
            const uint nMem = min(4u, uint(bucketSizes[b]) >> 2u);
            for (uint k = 0u; k < nMem; k++) {
                const uint m = baseBlk + uint(mArr[k]);
                if (m >= 1u && m < g.middleEnd) {
                    device const half* kPtr = pooledKeys + uint64_t(m) * p.indexDim;
                    float kDot = 0.0f;
                    for (uint d = 0u; d < dLim; d++) kDot = fma(qShared[d], float(kPtr[d]), kDot);
                    lsa_add_candidate(unit, m, kDot, p, marks, candCount, candM, candU);
                }
            }
        }
    }
    if (tg.x == 0u) {
        for (uint m = max(1u, g.numClusters * 128u) + tid; m < g.middleEnd; m += 256u) {
            device const half* kPtr = pooledKeys + uint64_t(m) * p.indexDim;
            float kDot = 0.0f;
            for (uint d = 0u; d < dLim; d++) kDot = fma(qShared[d], float(kPtr[d]), kDot);
            lsa_add_candidate(unit, m, kDot, p, marks, candCount, candM, candU);
        }
    }
}

kernel void qwg_lsa_select_finalize(device const half*   qIdx          [[buffer(0)]],
                                    device const half*   pooledKeys    [[buffer(1)]],
                                    device const half*   centroids     [[buffer(2)]],
                                    device const uchar4* bucketMembers [[buffer(3)]],
                                    device const uchar*  bucketSizes   [[buffer(4)]],
                                    device uint*         scoreScratch  [[buffer(5)]],
                                    device int*          blockList     [[buffer(6)]],
                                    device uint*         blockCount    [[buffer(7)]],
                                    device uint*         uHist         [[buffer(8)]],
                                    device uint*         marks         [[buffer(9)]],
                                    device uint*         candCount     [[buffer(10)]],
                                    device const uint*   candM         [[buffer(11)]],
                                    device const uint*   candU         [[buffer(12)]],
                                    constant LSASelParams& p           [[buffer(13)]],
                                    uint unit [[threadgroup_position_in_grid]],
                                    uint tid  [[thread_index_in_threadgroup]]) {
    if (unit >= p.T) return;
    constexpr uint SORT_N = 4096u;
    threadgroup float qShared[128];
    threadgroup uint  hist[256];
    threadgroup uint  scanSum[8];
    threadgroup uint  sgTotal;
    threadgroup uint  chosenBin;
    threadgroup uint  greaterAccum;
    threadgroup uint  keep[SORT_N];
    threadgroup uint  keepCount;

    const uint sg = tid >> 5u, lane = tid & 31u;
    const LSASelGeom g = lsa_sel_geom(p, unit);
    const uint rawCount = g.filter ? candCount[unit] : 0u;
    const uint nCand = min(rawCount, p.candCap);
    device const uint* cm = candM + uint64_t(unit) * p.candCap;
    device const uint* cu = candU + uint64_t(unit) * p.candCap;

    bool fast = g.filter && rawCount <= p.candCap && nCand >= g.kEff;
    uint threshU = 0u;
    if (fast) {
        if (tid == 0u) greaterAccum = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint mask = 0u;
        for (uint pass = 0u; pass < 4u; pass++) {
            const uint shift = 24u - (pass << 3u);
            hist[tid] = 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < nCand; i += 256u) {
                const uint u = cu[i];
                if ((u & mask) == threshU) {
                    atomic_fetch_add_explicit((threadgroup atomic_uint*)&hist[(u >> shift) & 0xFFu], 1u,
                                              memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            lsa_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, g.kEff, pass == 3u,
                               tid, sg, lane);
            threshU |= (chosenBin << shift);
            mask |= (0xFFu << shift);
        }
        fast = (threshU != 0u) && (sgTotal <= SORT_N);
    }

    if (fast) {
        constexpr uint TIE_FLAG = 0x80000000u;
        const uint greater = greaterAccum;
        const uint total = sgTotal;
        const uint needTies = g.kEff - greater;
        uint n = 256u;
        while (n < total) n <<= 1u;
        if (tid == 0u) keepCount = 0u;
        for (uint i = tid; i < n; i += 256u) keep[i] = 0xFFFFFFFFu;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < nCand; i += 256u) {
            const uint u = cu[i];
            if (u >= threshU) {
                const uint slot = atomic_fetch_add_explicit((threadgroup atomic_uint*)&keepCount, 1u,
                                                            memory_order_relaxed);
                keep[slot] = cm[i] | (u == threshU ? TIE_FLAG : 0u);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 2u; k <= n; k <<= 1u) {
            for (uint j = k >> 1u; j > 0u; j >>= 1u) {
                for (uint i = tid; i < n; i += 256u) {
                    const uint ixj = i ^ j;
                    if (ixj > i) {
                        const uint a = keep[i], c = keep[ixj];
                        const bool up = (i & k) == 0u;
                        if ((a > c) == up) { keep[i] = c; keep[ixj] = a; }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        device int* outRow = blockList + uint64_t(g.qRow) * p.listStride;
        const uint bBase = total - needTies;
        for (uint i = tid; i < greater; i += 256u) {
            const uint v = keep[i];
            uint lo = 0u, hi = needTies;
            while (lo < hi) {
                const uint mid = (lo + hi) >> 1u;
                if ((keep[bBase + mid] & ~TIE_FLAG) < v) lo = mid + 1u; else hi = mid;
            }
            outRow[1u + i + lo] = int(v);
        }
        for (uint j = tid; j < needTies; j += 256u) {
            const uint w = keep[bBase + j] & ~TIE_FLAG;
            uint lo = 0u, hi = greater;
            while (lo < hi) {
                const uint mid = (lo + hi) >> 1u;
                if (keep[mid] < w) lo = mid + 1u; else hi = mid;
            }
            outRow[1u + j + lo] = int(w);
        }
        if (tid == 0u) outRow[0] = 0;
        const uint windowOutBase = 1u + g.kEff;
        for (uint w = tid; w < p.windowBlocks; w += 256u) outRow[windowOutBase + w] = int(g.middleEnd + w);
        if (tid == 0u) blockCount[g.qRow] = windowOutBase + p.windowBlocks;
    } else {
        lsa_select_blocks_body(qIdx, pooledKeys, centroids, bucketMembers, bucketSizes, scoreScratch,
                               blockList, blockCount, p.pos0, p.indexDim, p.budgetBlocks, p.windowBlocks,
                               p.listStride, p.scoreStride, p.rowStep, p.maxRow, unit, tid,
                               qShared, hist, scanSum, sgTotal, chosenBin, greaterAccum);
    }

    if (g.filter) {
        device uint* mk = marks + uint64_t(unit) * p.markStride;
        if (rawCount > p.candCap) {
            for (uint w = tid; w < ((g.middleEnd + 31u) >> 5u); w += 256u) mk[w] = 0u;
        } else {
            for (uint i = tid; i < nCand; i += 256u) mk[cm[i] >> 5u] = 0u;
        }
        uHist[unit * 512u + tid] = 0u;
        uHist[unit * 512u + 256u + tid] = 0u;
        if (tid == 0u) candCount[unit] = 0u;
    }
}
