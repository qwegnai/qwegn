// qwemeans_coarse_scorers.metal — Subsystem-specific coarse centroid, candidate pool,
// and exact flat overflow-fallback kernels for LSA, Flash-Next QSA, and NoPE MLA.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "../include/qw_emeans_coarse.metalh"
using namespace metal;

struct EmeansFallbackTG {
    uint        hist[256];
    uint        scanSum[8];
    uint        greaterAccum;
    uint        chosenBin;
    uint        sgTotal;
    uint        keep[2048];
    atomic_uint gtSlot;
};

template <typename ScoreFn>
static inline void emeans_run_flat_fallback(device int*                  blockList,
                                            device uint*                 blockCount,
                                            device uint*                 candState,
                                            constant EmeansSelParams&    p,
                                            const EmeansSelGeom          g,
                                            threadgroup EmeansFallbackTG& s,
                                            uint                         unit,
                                            uint                         tid,
                                            ScoreFn                      scoreFn) {
    constexpr uint MAX_KEEP = 2048u;
    const uint sg = tid >> 5u, lane = tid & 31u;
    const uint nCand = g.numMiddle;
    const uint kTarget = min(min(g.kEff, nCand), MAX_KEEP);
    device int* outRow = blockList + uint64_t(g.qRow) * p.listStride;

    uint nPow2 = 256u;
    while (nPow2 < kTarget) nPow2 <<= 1u;
    for (uint i = tid; i < nPow2; i += 256u) s.keep[i] = 0xFFFFFFFFu;
    if (tid == 0u) {
        s.greaterAccum = 0u; s.chosenBin = 0u; s.sgTotal = 0u;
        atomic_store_explicit(&s.gtSlot, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (kTarget > 0u) {
        uint threshU = 0u, mask = 0u;
        for (uint pass = 0u; pass < 4u; pass++) {
            const uint shift = 24u - (pass << 3u);
            s.hist[tid] = 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < nCand; i += 256u) {
                const uint u = scoreFn(g.minPool + i);
                if ((u & mask) == threshU) {
                    atomic_fetch_add_explicit((threadgroup atomic_uint*)&s.hist[(u >> shift) & 0xFFu], 1u,
                                              memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            emeans_pick_radix_bin(s.hist, s.scanSum, s.greaterAccum, s.chosenBin, s.sgTotal, kTarget,
                                  pass == 3u, tid, sg, lane);
            threshU |= (s.chosenBin << shift);
            mask |= (0xFFu << shift);
        }

        const uint needTies = kTarget - s.greaterAccum;
        uint minTiePool = 0u;
        if (s.sgTotal > kTarget && needTies > 0u) {
            if (tid == 0u) s.greaterAccum = 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint tieMask = 0u;
            for (uint p2 = 0u; p2 < 3u; p2++) {
                const uint sh2 = 16u - (p2 << 3u);
                s.hist[tid] = 0u;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = tid; i < nCand; i += 256u) {
                    const uint m = g.minPool + i;
                    if (scoreFn(m) == threshU && (m & tieMask) == minTiePool) {
                        atomic_fetch_add_explicit((threadgroup atomic_uint*)&s.hist[(m >> sh2) & 0xFFu], 1u,
                                                  memory_order_relaxed);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                emeans_pick_radix_bin(s.hist, s.scanSum, s.greaterAccum, s.chosenBin, s.sgTotal, needTies,
                                      p2 == 2u, tid, sg, lane);
                minTiePool |= (s.chosenBin << sh2);
                tieMask |= (0xFFu << sh2);
            }
        }

        for (uint i = tid; i < nCand; i += 256u) {
            const uint m = g.minPool + i;
            const uint u = scoreFn(m);
            if (u > threshU || (u == threshU && needTies > 0u && m >= minTiePool)) {
                const uint sl = atomic_fetch_add_explicit(&s.gtSlot, 1u, memory_order_relaxed);
                if (sl < kTarget) s.keep[sl] = m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    emeans_sort_and_emit(outRow, blockCount, p, g, s.keep, kTarget, nPow2, tid);
    if (tid < 2u) candState[unit * 2u + tid] = 0u;
}

// 1a. Coarse centroid scoring for LSA (cosine similarity).
kernel void qwg_emeans_lsa_score_coarse(device const half*        qIdx         [[buffer(0)]],
                                        device const half*        centroids    [[buffer(1)]],
                                        device const ushort2*     clusterSpans [[buffer(2)]],
                                        device uint*              cScore       [[buffer(3)]],
                                        device atomic_uint*       uHist        [[buffer(4)]],
                                        constant EmeansSelParams& p            [[buffer(5)]],
                                        uint2 tg  [[threadgroup_position_in_grid]],
                                        uint  tid [[thread_index_in_threadgroup]]) {
    const uint unit = tg.y;
    if (unit >= p.T) return;
    const EmeansSelGeom g = emeans_sel_geom(p, unit);
    if (g.dense || tg.x * 256u >= g.numClusters) return;
    threadgroup float qShared[128];
    threadgroup atomic_uint hist[256];
    const uint dLim = min(p.indexDim, 128u);
    if (tid < dLim) qShared[tid] = float(qIdx[uint64_t(g.qRow) * p.indexDim + tid]);
    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint b = tg.x * 256u + tid;
    if (b < g.numClusters) {
        uint u = 0u;
        if (clusterSpans[b].y > 0u) {
            device const half* cPtr = centroids + uint64_t(b) * p.indexDim;
            float dot = 0.0f;
            for (uint d = 0u; d < dLim; d++) dot = fma(qShared[d], float(cPtr[d]), dot);
            u = emeans_nonzero_score_u32(dot);
        }
        cScore[uint64_t(unit) * p.clusterStride + b] = u;
        atomic_fetch_add_explicit(&hist[(u >> 24u) & 0xFFu], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint c = atomic_load_explicit(&hist[tid], memory_order_relaxed);
    if (c != 0u) atomic_fetch_add_explicit(&uHist[unit * 512u + tid], c, memory_order_relaxed);
}

// 1b. Dense coalesced candidate scoring for LSA (cosine similarity).
kernel void qwg_emeans_lsa_score_cands(device const half*        qIdx      [[buffer(0)]],
                                       device const half*        pooled    [[buffer(1)]],
                                       device const uint*        candState [[buffer(2)]],
                                       device const uint*        candPools [[buffer(3)]],
                                       device uint*              candU     [[buffer(4)]],
                                       constant EmeansSelParams& p         [[buffer(5)]],
                                       uint2 tg  [[threadgroup_position_in_grid]],
                                       uint  tid [[thread_index_in_threadgroup]]) {
    const uint unit = tg.y;
    if (unit >= p.T) return;
    const EmeansSelGeom g = emeans_sel_geom(p, unit);
    if (g.dense) return;
    const bool flatContig = (p.probeClusters == 0u && g.numMiddle <= p.candCap);
    if (!flatContig && candState[unit * 2u + 1u] != 0u) return;
    const uint nCand = flatContig ? g.numMiddle : min(candState[unit * 2u], p.candCap);
    const uint idx = tg.x * 256u + tid;
    if (tg.x * 256u >= nCand) return;
    threadgroup float qShared[128];
    const uint dLim = min(p.indexDim, 128u);
    if (tid < dLim) qShared[tid] = float(qIdx[uint64_t(g.qRow) * p.indexDim + tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (idx < nCand) {
        const uint m = flatContig ? (g.minPool + idx) : candPools[uint64_t(unit) * p.candCap + idx];
        device const half* kPtr = pooled + uint64_t(m) * p.indexDim;
        float dot = 0.0f;
        for (uint d = 0u; d < dLim; d++) dot = fma(qShared[d], float(kPtr[d]), dot);
        candU[uint64_t(unit) * p.candCap + idx] = emeans_float_to_u32(dot);
    }
}

// 1c. Exact flat overflow fallback for LSA.
kernel void qwg_emeans_lsa_flat_fallback(device const half*        qIdx       [[buffer(0)]],
                                         device const half*        pooled     [[buffer(1)]],
                                         device uint*              candState  [[buffer(2)]],
                                         device int*               blockList  [[buffer(3)]],
                                         device uint*              blockCount [[buffer(4)]],
                                         constant EmeansSelParams& p          [[buffer(5)]],
                                         uint unit [[threadgroup_position_in_grid]],
                                         uint tid  [[thread_index_in_threadgroup]]) {
    if (unit >= p.T) return;
    const EmeansSelGeom g = emeans_sel_geom(p, unit);
    if (g.dense) return;
    const bool forceFlat = (p.probeClusters == 0u && g.numMiddle > p.candCap);
    if (!forceFlat && candState[unit * 2u + 1u] == 0u) return;
    threadgroup float qShared[128];
    threadgroup EmeansFallbackTG fb;
    const uint dLim = min(p.indexDim, 128u);
    if (tid < dLim) qShared[tid] = float(qIdx[uint64_t(g.qRow) * p.indexDim + tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    emeans_run_flat_fallback(blockList, blockCount, candState, p, g, fb, unit, tid, [&](uint m) -> uint {
        device const half* kPtr = pooled + uint64_t(m) * p.indexDim;
        float dot = 0.0f;
        for (uint d = 0u; d < dLim; d++) dot = fma(qShared[d], float(kPtr[d]), dot);
        return emeans_float_to_u32(dot);
    });
}

// 2a. Coarse centroid scoring for Flash-Next QSA (4-head ReLU, d=128).
kernel void qwg_emeans_qsa_score_coarse(device const float*       idxQ         [[buffer(0)]],
                                        device const half*        centroids    [[buffer(1)]],
                                        device const ushort2*     clusterSpans [[buffer(2)]],
                                        device uint*              cScore       [[buffer(3)]],
                                        device atomic_uint*       uHist        [[buffer(4)]],
                                        constant EmeansSelParams& p            [[buffer(5)]],
                                        uint2 tg  [[threadgroup_position_in_grid]],
                                        uint  tid [[thread_index_in_threadgroup]]) {
    const uint unit = tg.y;
    if (unit >= p.T) return;
    const EmeansSelGeom g = emeans_sel_geom(p, unit);
    if (g.dense || tg.x * 256u >= g.numClusters) return;
    constexpr uint DI = 128u, NI = 4u;
    threadgroup float qs[NI * DI];
    threadgroup atomic_uint hist[256];
    for (uint i = tid; i < NI * DI; i += 256u) qs[i] = idxQ[uint64_t(g.qRow) * NI * DI + i];
    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint b = tg.x * 256u + tid;
    if (b < g.numClusters) {
        uint u = 0u;
        if (clusterSpans[b].y > 0u) {
            device const half4* cb = (device const half4*)(centroids + uint64_t(b) * DI);
            float sumReLU = 0.0f;
            for (uint h = 0u; h < NI; h++) {
                threadgroup const float4* qh = (threadgroup const float4*)(qs + h * DI);
                float dot = 0.0f;
                for (uint d4 = 0u; d4 < DI / 4u; d4++) {
                    const float4 qv = qh[d4], cv = float4(cb[d4]);
                    dot += qv.x * cv.x + qv.y * cv.y + qv.z * cv.z + qv.w * cv.w;
                }
                sumReLU += max(dot, 0.0f);
            }
            u = emeans_nonzero_score_u32(sumReLU * rsqrt(float(DI)));
        }
        cScore[uint64_t(unit) * p.clusterStride + b] = u;
        atomic_fetch_add_explicit(&hist[(u >> 24u) & 0xFFu], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint c = atomic_load_explicit(&hist[tid], memory_order_relaxed);
    if (c != 0u) atomic_fetch_add_explicit(&uHist[unit * 512u + tid], c, memory_order_relaxed);
}

// 2b. Dense coalesced candidate scoring for Flash-Next QSA (4-head ReLU).
kernel void qwg_emeans_qsa_score_cands(device const float*       idxQ      [[buffer(0)]],
                                       device const half*        pooled    [[buffer(1)]],
                                       device const uint*        candState [[buffer(2)]],
                                       device const uint*        candPools [[buffer(3)]],
                                       device uint*              candU     [[buffer(4)]],
                                       constant EmeansSelParams& p         [[buffer(5)]],
                                       uint2 tg  [[threadgroup_position_in_grid]],
                                       uint  tid [[thread_index_in_threadgroup]]) {
    const uint unit = tg.y;
    if (unit >= p.T) return;
    const EmeansSelGeom g = emeans_sel_geom(p, unit);
    if (g.dense) return;
    const bool flatContig = (p.probeClusters == 0u && g.numMiddle <= p.candCap);
    if (!flatContig && candState[unit * 2u + 1u] != 0u) return;
    const uint nCand = flatContig ? g.numMiddle : min(candState[unit * 2u], p.candCap);
    if (tg.x * 256u >= nCand) return;
    constexpr uint DI = 128u, NI = 4u;
    threadgroup float qs[NI * DI];
    for (uint i = tid; i < NI * DI; i += 256u) qs[i] = idxQ[uint64_t(g.qRow) * NI * DI + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint idx = tg.x * 256u + tid;
    if (idx < nCand) {
        const uint m = flatContig ? (g.minPool + idx) : candPools[uint64_t(unit) * p.candCap + idx];
        device const half4* pb = (device const half4*)(pooled + uint64_t(m) * DI);
        float sumReLU = 0.0f;
        for (uint h = 0u; h < NI; h++) {
            threadgroup const float4* qh = (threadgroup const float4*)(qs + h * DI);
            float dot = 0.0f;
            for (uint d4 = 0u; d4 < DI / 4u; d4++) {
                const float4 qv = qh[d4], pv = float4(pb[d4]);
                dot += qv.x * pv.x + qv.y * pv.y + qv.z * pv.z + qv.w * pv.w;
            }
            sumReLU += max(dot, 0.0f);
        }
        candU[uint64_t(unit) * p.candCap + idx] = emeans_float_to_u32(sumReLU * rsqrt(float(DI)));
    }
}

// 2c. Exact flat overflow fallback for Flash-Next QSA.
kernel void qwg_emeans_qsa_flat_fallback(device const float*       idxQ       [[buffer(0)]],
                                         device const half*        pooled     [[buffer(1)]],
                                         device uint*              candState  [[buffer(2)]],
                                         device int*               blockList  [[buffer(3)]],
                                         device uint*              blockCount [[buffer(4)]],
                                         constant EmeansSelParams& p          [[buffer(5)]],
                                         uint unit [[threadgroup_position_in_grid]],
                                         uint tid  [[thread_index_in_threadgroup]]) {
    if (unit >= p.T) return;
    const EmeansSelGeom g = emeans_sel_geom(p, unit);
    if (g.dense) return;
    const bool forceFlat = (p.probeClusters == 0u && g.numMiddle > p.candCap);
    if (!forceFlat && candState[unit * 2u + 1u] == 0u) return;
    constexpr uint DI = 128u, NI = 4u;
    threadgroup float qs[NI * DI];
    threadgroup EmeansFallbackTG fb;
    for (uint i = tid; i < NI * DI; i += 256u) qs[i] = idxQ[uint64_t(g.qRow) * NI * DI + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = rsqrt(float(DI));

    emeans_run_flat_fallback(blockList, blockCount, candState, p, g, fb, unit, tid, [&](uint m) -> uint {
        device const half4* pb = (device const half4*)(pooled + uint64_t(m) * DI);
        float sumReLU = 0.0f;
        for (uint h = 0u; h < NI; h++) {
            threadgroup const float4* qh = (threadgroup const float4*)(qs + h * DI);
            float dot = 0.0f;
            for (uint d4 = 0u; d4 < DI / 4u; d4++) {
                const float4 qv = qh[d4], pv = float4(pb[d4]);
                dot += qv.x * pv.x + qv.y * pv.y + qv.z * pv.z + qv.w * pv.w;
            }
            sumReLU += max(dot, 0.0f);
        }
        return emeans_float_to_u32(sumReLU * scale);
    });
}

// 3a. Coarse centroid scoring for NoPE MLA (32-head weighted ReLU via simdgroup 8x8 MMA).
kernel void qwg_emeans_mla_score_coarse(device const half*        mlaQFull     [[buffer(0)]],
                                        device const float*       idxWeights   [[buffer(1)]],
                                        device const half*        centroids    [[buffer(2)]],
                                        device const ushort2*     clusterSpans [[buffer(3)]],
                                        device uint*              cScore       [[buffer(4)]],
                                        device atomic_uint*       uHist        [[buffer(5)]],
                                        constant EmeansSelParams& p            [[buffer(6)]],
                                        uint2 tg   [[threadgroup_position_in_grid]],
                                        uint  tid  [[thread_index_in_threadgroup]],
                                        uint  sg   [[simdgroup_index_in_threadgroup]],
                                        uint  lane [[thread_index_in_simdgroup]]) {
    const uint unit = tg.y;
    if (unit >= p.T) return;
    const EmeansSelGeom g = emeans_sel_geom(p, unit);
    const uint c0 = tg.x * 64u;
    if (g.dense || c0 >= g.numClusters) return;
    constexpr uint HEADS = 32u, DIM = 128u, QS = DIM + 8u;
    threadgroup half  shQ[HEADS * QS];
    threadgroup float shW[HEADS];
    threadgroup float shS[8u * HEADS * 8u];
    threadgroup atomic_uint hist[256];

    device const half* qRow = mlaQFull + uint64_t(g.qRow) * (p.qChannels + p.idxHeads * p.indexDim) + p.qChannels;
    for (uint i = tid; i < HEADS * DIM / 4u; i += 256u) {
        const uint h = i / (DIM / 4u), c = (i % (DIM / 4u)) * 4u;
        *(threadgroup half4*)(shQ + h * QS + c) = *(device const half4*)(qRow + h * DIM + c);
    }
    if (tid < HEADS) shW[tid] = idxWeights[uint64_t(g.qRow) * p.idxHeads + tid];
    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint cBase = c0 + sg * 8u;
    if (cBase < g.numClusters) {
        simdgroup_float8x8 acc[4];
        for (uint a = 0u; a < 4u; ++a) acc[a] = simdgroup_float8x8(0.0f);
        device const half* kBase = centroids + uint64_t(cBase) * DIM;
        for (uint d = 0u; d < DIM / 8u; ++d) {
            simdgroup_half8x8 kf;
            simdgroup_load(kf, kBase + d * 8u, DIM, ulong2(0, 0), true);
            for (uint a = 0u; a < 4u; ++a) {
                simdgroup_half8x8 qf;
                simdgroup_load(qf, shQ + a * 8u * QS + d * 8u, QS);
                simdgroup_multiply_accumulate(acc[a], qf, kf, acc[a]);
            }
        }
        threadgroup float* s = shS + sg * (HEADS * 8u);
        for (uint a = 0u; a < 4u; ++a) simdgroup_store(acc[a], s + a * 64u, 8u);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const float scale = rsqrt(float(DIM));
        const uint pc = lane & 7u, h0 = (lane >> 3u) * 8u;
        float v = 0.0f;
        for (uint h = 0u; h < 8u; ++h) v = fma(shW[h0 + h], max(0.0f, s[(h0 + h) * 8u + pc] * scale), v);
        v += simd_shuffle_xor(v, 8u);
        v += simd_shuffle_xor(v, 16u);
        if (lane < 8u && cBase + lane < g.numClusters) {
            const uint b = cBase + lane;
            const uint u = (clusterSpans[b].y > 0u) ? emeans_nonzero_score_u32(v) : 0u;
            cScore[uint64_t(unit) * p.clusterStride + b] = u;
            atomic_fetch_add_explicit(&hist[(u >> 24u) & 0xFFu], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint cnt = atomic_load_explicit(&hist[tid], memory_order_relaxed);
    if (cnt != 0u) atomic_fetch_add_explicit(&uHist[unit * 512u + tid], cnt, memory_order_relaxed);
}

// 3b. Dense coalesced candidate scoring for NoPE MLA (32-head weighted ReLU via 8x8 MMA).
kernel void qwg_emeans_mla_score_cands(device const half*        mlaQFull   [[buffer(0)]],
                                       device const float*       idxWeights [[buffer(1)]],
                                       device const half*        pooled     [[buffer(2)]],
                                       device const uint*        candState  [[buffer(3)]],
                                       device const uint*        candPools  [[buffer(4)]],
                                       device uint*              candU      [[buffer(5)]],
                                       constant EmeansSelParams& p          [[buffer(6)]],
                                       uint2 tg   [[threadgroup_position_in_grid]],
                                       uint  tid  [[thread_index_in_threadgroup]],
                                       uint  sg   [[simdgroup_index_in_threadgroup]],
                                       uint  lane [[thread_index_in_simdgroup]]) {
    const uint unit = tg.y;
    if (unit >= p.T) return;
    const EmeansSelGeom g = emeans_sel_geom(p, unit);
    if (g.dense) return;
    const bool flatContig = (p.probeClusters == 0u && g.numMiddle <= p.candCap);
    if (!flatContig && candState[unit * 2u + 1u] != 0u) return;
    const uint nCand = flatContig ? g.numMiddle : min(candState[unit * 2u], p.candCap);
    const uint i0 = tg.x * 64u;
    if (i0 >= nCand) return;
    constexpr uint HEADS = 32u, DIM = 128u, QS = DIM + 8u;
    threadgroup half  shQ[HEADS * QS];
    threadgroup float shW[HEADS];
    threadgroup half  shK[64u * DIM];

    device const half* qRow = mlaQFull + uint64_t(g.qRow) * (p.qChannels + p.idxHeads * p.indexDim) + p.qChannels;
    for (uint i = tid; i < HEADS * DIM / 4u; i += 256u) {
        const uint h = i / (DIM / 4u), c = (i % (DIM / 4u)) * 4u;
        *(threadgroup half4*)(shQ + h * QS + c) = *(device const half4*)(qRow + h * DIM + c);
    }
    if (tid < HEADS) shW[tid] = idxWeights[uint64_t(g.qRow) * p.idxHeads + tid];
    const uint tileCnt = min(64u, nCand - i0);
    for (uint v = tid; v < 64u * (DIM / 4u); v += 256u) {
        const uint kSlot = v / (DIM / 4u), d4 = (v % (DIM / 4u)) * 4u;
        half4 val = half4(0.0h);
        if (kSlot < tileCnt) {
            const uint m = flatContig ? (g.minPool + i0 + kSlot) : candPools[uint64_t(unit) * p.candCap + i0 + kSlot];
            val = *(device const half4*)(pooled + uint64_t(m) * DIM + d4);
        }
        *(threadgroup half4*)(shK + kSlot * DIM + d4) = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint kBase = sg * 8u;
    if (kBase < tileCnt) {
        simdgroup_float8x8 acc[4];
        for (uint a = 0u; a < 4u; ++a) acc[a] = simdgroup_float8x8(0.0f);
        threadgroup const half* kp = shK + kBase * DIM;
        for (uint d = 0u; d < DIM / 8u; ++d) {
            simdgroup_half8x8 kf;
            simdgroup_load(kf, kp + d * 8u, DIM, ulong2(0, 0), true);
            for (uint a = 0u; a < 4u; ++a) {
                simdgroup_half8x8 qf;
                simdgroup_load(qf, shQ + a * 8u * QS + d * 8u, QS);
                simdgroup_multiply_accumulate(acc[a], qf, kf, acc[a]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* s = (threadgroup float*)(shK + kBase * DIM);
        for (uint a = 0u; a < 4u; ++a) simdgroup_store(acc[a], s + a * 64u, 8u);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const float scale = rsqrt(float(DIM));
        const uint pc = lane & 7u, h0 = (lane >> 3u) * 8u;
        float v = 0.0f;
        for (uint h = 0u; h < 8u; ++h) v = fma(shW[h0 + h], max(0.0f, s[(h0 + h) * 8u + pc] * scale), v);
        v += simd_shuffle_xor(v, 8u);
        v += simd_shuffle_xor(v, 16u);
        if (lane < 8u && i0 + kBase + lane < nCand) {
            candU[uint64_t(unit) * p.candCap + i0 + kBase + lane] = emeans_float_to_u32(v);
        }
    }
}

// 3c. Exact flat overflow fallback for NoPE MLA.
kernel void qwg_emeans_mla_flat_fallback(device const half*        mlaQFull   [[buffer(0)]],
                                         device const float*       idxWeights [[buffer(1)]],
                                         device const half*        pooled     [[buffer(2)]],
                                         device uint*              candState  [[buffer(3)]],
                                         device int*               blockList  [[buffer(4)]],
                                         device uint*              blockCount [[buffer(5)]],
                                         constant EmeansSelParams& p          [[buffer(6)]],
                                         uint unit [[threadgroup_position_in_grid]],
                                         uint tid  [[thread_index_in_threadgroup]]) {
    if (unit >= p.T) return;
    const EmeansSelGeom g = emeans_sel_geom(p, unit);
    if (g.dense) return;
    const bool forceFlat = (p.probeClusters == 0u && g.numMiddle > p.candCap);
    if (!forceFlat && candState[unit * 2u + 1u] == 0u) return;
    constexpr uint HEADS = 32u, DIM = 128u;
    threadgroup half  shQ[HEADS * DIM];
    threadgroup float shW[HEADS];
    threadgroup EmeansFallbackTG fb;
    device const half* qRow = mlaQFull + uint64_t(g.qRow) * (p.qChannels + p.idxHeads * p.indexDim) + p.qChannels;
    for (uint i = tid; i < HEADS * DIM / 4u; i += 256u) {
        *(threadgroup half4*)(shQ + i * 4u) = *(device const half4*)(qRow + i * 4u);
    }
    if (tid < HEADS) shW[tid] = idxWeights[uint64_t(g.qRow) * p.idxHeads + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = rsqrt(float(DIM));

    emeans_run_flat_fallback(blockList, blockCount, candState, p, g, fb, unit, tid, [&](uint m) -> uint {
        device const half4* pb = (device const half4*)(pooled + uint64_t(m) * DIM);
        float sumVal = 0.0f;
        for (uint h = 0u; h < HEADS; h++) {
            threadgroup const half4* qh = (threadgroup const half4*)(shQ + h * DIM);
            float dot = 0.0f;
            for (uint d4 = 0u; d4 < DIM / 4u; d4++) {
                const float4 qv = float4(qh[d4]), pv = float4(pb[d4]);
                dot += qv.x * pv.x + qv.y * pv.y + qv.z * pv.z + qv.w * pv.w;
            }
            sumVal = fma(shW[h], max(0.0f, dot * scale), sumVal);
        }
        return emeans_float_to_u32(sumVal);
    });
}
