// qwemeans_coarse.metal — Capacity-balanced e-means (arXiv:2611.09501) macro-chunk
// clustering (1024 pools -> 64 CSR clusters) and shared radix/CSR prune+finalize stages.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "../include/qw_emeans_coarse.metalh"
using namespace metal;

// 1. qwg_emeans_cluster_1024: partitions each newly completed 1024-pool (4096-token)
// macro-chunk into 64 CSR clusters (56 balanced e-means + 8 max-residual outliers).
kernel void qwg_emeans_cluster_1024(device const half* pooledKeys   [[buffer(0)]],
                                    device half*       centroids    [[buffer(1)]],
                                    device ushort*     memberPools  [[buffer(2)]],
                                    device ushort2*    clusterSpans [[buffer(3)]],
                                    constant uint&     startMacro   [[buffer(4)]],
                                    constant uint&     numMacros    [[buffer(5)]],
                                    constant uint&     indexDim     [[buffer(6)]],
                                    uint tg  [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
    if (tg >= numMacros) return;
    constexpr uint K_ALL = 64u, K_BASE = 56u, B = 128u, STEPS = 16u;
    const uint macroIdx = startMacro + tg;
    const uint basePool = macroIdx * 1024u;
    const uint dLim = min(indexDim, 128u);
    const uint sg = tid >> 5u, lane = tid & 31u;
    const uint cId = tid >> 2u, sub = tid & 3u;
    const uint subDim = dLim >> 2u, dBase = sub * subDim;
    const uint bId = tid >> 1u, halfSide = tid & 1u;
    const uint halfDim = dLim >> 1u, hBase = halfSide * halfDim;

    threadgroup half        tgCent[64 * 128];
    threadgroup float       tgInvNorm[1024];
    threadgroup float       tgCountEma[64];
    threadgroup float       tgLastCount[64];
    threadgroup float       tgBias[64];
    threadgroup float       tgNorm2[64];
    threadgroup ushort      tgAssign[1024];
    threadgroup float       tgMaxSim[1024];
    threadgroup atomic_uint tgCnt[64];
    threadgroup ushort      tgOff[65];
    threadgroup float       sgVal[8];
    threadgroup uint        sgIdx[8];
    threadgroup uint        tgChosen;

    device const half* chunkKeys = pooledKeys + uint64_t(basePool) * indexDim;
    for (uint i = tid; i < 1024u; i += 256u) {
        device const half* xi = chunkKeys + i * indexDim;
        float n2 = 0.0f;
        for (uint d = 0u; d < dLim; d++) {
            const float v = float(xi[d]);
            n2 = fma(v, v, n2);
        }
        tgInvNorm[i] = (n2 > 1e-12f) ? rsqrt(n2) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float regSumEma[32];
    float regLastSum[32];
    const uint initPool = (cId * 197u + 13u) & 1023u;
    const float invInit = tgInvNorm[initPool];
    float initNorm2 = 0.0f;
    for (uint d = 0u; d < subDim; d++) {
        const float v = (cId < K_BASE) ? (float(chunkKeys[initPool * indexDim + dBase + d]) * invInit) : 0.0f;
        regSumEma[d] = v * 1e-5f;
        regLastSum[d] = 0.0f;
        const half h = half(v);
        tgCent[cId * 128u + dBase + d] = h;
        const float hf = float(h);
        initNorm2 = fma(hf, hf, initNorm2);
    }
    initNorm2 += simd_shuffle_xor(initNorm2, 1u);
    initNorm2 += simd_shuffle_xor(initNorm2, 2u);
    if (sub == 0u) {
        tgCountEma[cId] = 1e-5f;
        tgLastCount[cId] = 0.0f;
        tgBias[cId] = 0.0f;
        tgNorm2[cId] = initNorm2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = 0u; step < STEPS; step++) {
        if (tid < K_ALL) atomic_store_explicit(&tgCnt[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint poolIdx = ((bId + (step & 7u) * B) * 197u + (step >> 3u) * 37u) & 1023u;
        const float invB = tgInvNorm[poolIdx];
        float xReg[64];
        for (uint d = 0u; d < halfDim; d++) {
            xReg[d] = float(chunkKeys[poolIdx * indexDim + hBase + d]) * invB;
        }
        float bestDist = 1e30f;
        uint bestJ = 0u;
        for (uint j = 0u; j < K_BASE; j++) {
            float dot = 0.0f;
            threadgroup const half* cj = tgCent + j * 128u + hBase;
            for (uint d = 0u; d < halfDim; d++) dot = fma(xReg[d], float(cj[d]), dot);
            dot += simd_shuffle_xor(dot, 1u);
            const float dist = tgNorm2[j] - 2.0f * dot;
            if (dist < bestDist) { bestDist = dist; bestJ = j; }
        }
        if (halfSide == 0u) {
            tgAssign[bId] = ushort(bestJ);
            atomic_fetch_add_explicit(&tgCnt[bestJ], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float cosD = 0.5f * (1.0f + cos(3.14159265f * float(step) / float(STEPS)));
        const float alphaT = 0.45f * (0.1f + 0.9f * cosD);
        const float gammaT = 1.0f - alphaT;

        if (cId < K_BASE) {
            const float bCnt = float(atomic_load_explicit(&tgCnt[cId], memory_order_relaxed));
            float bSum[32];
            for (uint d = 0u; d < subDim; d++) bSum[d] = 0.0f;
            if (bCnt > 0.0f) {
                for (uint b = 0u; b < B; b++) {
                    if (tgAssign[b] == cId) {
                        const uint pIdx = ((b + (step & 7u) * B) * 197u + (step >> 3u) * 37u) & 1023u;
                        const float invP = tgInvNorm[pIdx];
                        device const half* xp = chunkKeys + pIdx * indexDim + dBase;
                        for (uint d = 0u; d < subDim; d++) bSum[d] += float(xp[d]) * invP;
                    }
                }
            }
            const float newCnt = gammaT * tgCountEma[cId] + alphaT * bCnt;
            const float targetB = float(B) / float(K_BASE);
            if (sub == 0u) {
                tgCountEma[cId] = newCnt;
                tgBias[cId] += 0.025f * clamp((targetB - bCnt) / targetB, -1.5f, 1.5f);
                if (step >= (STEPS >> 1u)) tgLastCount[cId] += bCnt;
            }
            const float invCnt = 1.0f / max(newCnt, 1e-12f);
            float n2 = 0.0f;
            for (uint d = 0u; d < subDim; d++) {
                regSumEma[d] = gammaT * regSumEma[d] + alphaT * bSum[d];
                if (step >= (STEPS >> 1u)) regLastSum[d] += bSum[d];
                const half ch = half(regSumEma[d] * invCnt);
                tgCent[cId * 128u + dBase + d] = ch;
                const float cf = float(ch);
                n2 = fma(cf, cf, n2);
            }
            n2 += simd_shuffle_xor(n2, 1u);
            n2 += simd_shuffle_xor(n2, 2u);
            if (sub == 0u) tgNorm2[cId] = n2 - tgBias[cId];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (((step + 1u) % 5u) == 0u && step < 12u) {
            if (cId < K_BASE && tgCountEma[cId] < 0.05f) {
                const uint rPool = (cId * 397u + step * 131u + 59u) & 1023u;
                const float invR = tgInvNorm[rPool];
                device const half* xr = chunkKeys + rPool * indexDim + dBase;
                float n2 = 0.0f;
                for (uint d = 0u; d < subDim; d++) {
                    const float rv = float(xr[d]) * invR;
                    regSumEma[d] = 0.5f * rv;
                    const half ch = half(rv);
                    tgCent[cId * 128u + dBase + d] = ch;
                    n2 = fma(float(ch), float(ch), n2);
                }
                n2 += simd_shuffle_xor(n2, 1u);
                n2 += simd_shuffle_xor(n2, 2u);
                if (sub == 0u) {
                    tgCountEma[cId] = 0.5f;
                    tgBias[cId] = 0.02f;
                    tgNorm2[cId] = n2 - 0.02f;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (cId < K_BASE) {
        const bool useLast = (tgLastCount[cId] > 0.0f);
        const float denom = 1.0f / max(useLast ? tgLastCount[cId] : tgCountEma[cId], 1e-12f);
        float n2 = 0.0f;
        float vals[32];
        for (uint d = 0u; d < subDim; d++) {
            vals[d] = (useLast ? regLastSum[d] : regSumEma[d]) * denom;
            n2 = fma(vals[d], vals[d], n2);
        }
        n2 += simd_shuffle_xor(n2, 1u);
        n2 += simd_shuffle_xor(n2, 2u);
        const float invN = rsqrt(max(n2, 1e-12f));
        for (uint d = 0u; d < subDim; d++) {
            tgCent[cId * 128u + dBase + d] = half(vals[d] * invN);
        }
        if (sub == 0u) tgBias[cId] = 0.0f;
    } else if (sub == 0u) {
        tgBias[cId] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < 1024u; i += 256u) {
        const float invI = tgInvNorm[i];
        if (invI == 0.0f) {
            tgMaxSim[i] = 2.0f;
        } else {
            device const half* xi = chunkKeys + i * indexDim;
            float maxS = -1e30f;
            for (uint j = 0u; j < K_BASE; j++) {
                threadgroup const half* cj = tgCent + j * 128u;
                float dot = 0.0f;
                for (uint d = 0u; d < dLim; d++) dot = fma(float(xi[d]), float(cj[d]), dot);
                maxS = max(maxS, dot);
            }
            tgMaxSim[i] = maxS * invI;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint o = 0u; o < (K_ALL - K_BASE); o++) {
        float minS = 1e30f;
        uint minI = tid;
        for (uint i = tid; i < 1024u; i += 256u) {
            const float s = tgMaxSim[i];
            if (s < minS || (s == minS && i < minI)) { minS = s; minI = i; }
        }
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float os = simd_shuffle_down(minS, off);
            const uint  oi = simd_shuffle_down(minI, off);
            if (os < minS || (os == minS && oi < minI)) { minS = os; minI = oi; }
        }
        if (lane == 0u) { sgVal[sg] = minS; sgIdx[sg] = minI; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float wS = sgVal[0]; uint wI = sgIdx[0];
            for (uint s = 1u; s < 8u; s++) {
                if (sgVal[s] < wS || (sgVal[s] == wS && sgIdx[s] < wI)) { wS = sgVal[s]; wI = sgIdx[s]; }
            }
            tgChosen = wI;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint outSlot = K_BASE + o;
        const uint winPool = tgChosen;
        const float invW = tgInvNorm[winPool];
        if (tid < dLim) {
            tgCent[outSlot * 128u + tid] = half(float(chunkKeys[winPool * indexDim + tid]) * invW);
        }
        if (tid == 0u) tgMaxSim[winPool] = 2.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < 1024u; i += 256u) {
            if (tgMaxSim[i] <= 1.5f) {
                device const half* xi = chunkKeys + i * indexDim;
                threadgroup const half* co = tgCent + outSlot * 128u;
                float dot = 0.0f;
                for (uint d = 0u; d < dLim; d++) dot = fma(float(xi[d]), float(co[d]), dot);
                tgMaxSim[i] = max(tgMaxSim[i], dot * tgInvNorm[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint pol = 0u; pol < 4u; pol++) {
        if (tid < K_ALL) atomic_store_explicit(&tgCnt[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < 1024u; i += 256u) {
            const float invI = tgInvNorm[i];
            device const half* xi = chunkKeys + i * indexDim;
            float bestS = -1e30f;
            uint bestJ = 0u;
            for (uint j = 0u; j < K_ALL; j++) {
                threadgroup const half* cj = tgCent + j * 128u;
                float dot = 0.0f;
                for (uint d = 0u; d < dLim; d++) dot = fma(float(xi[d]), float(cj[d]), dot);
                const float sc = dot * invI + tgBias[j];
                if (sc > bestS) { bestS = sc; bestJ = j; }
            }
            tgAssign[i] = ushort(bestJ);
            atomic_fetch_add_explicit(&tgCnt[bestJ], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (pol < 3u && tid < K_ALL) {
            const float eta = (pol == 0u) ? 0.04f : ((pol == 1u) ? 0.03f : 0.02f);
            const float c = float(atomic_load_explicit(&tgCnt[tid], memory_order_relaxed));
            if (tid < K_BASE) {
                tgBias[tid] += eta * clamp((17.5f - c) / 17.5f, -1.5f, 1.5f);
            } else if (c > 16.0f) {
                tgBias[tid] += eta * clamp((16.0f - c) / 16.0f, -1.5f, 0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        ushort run = 0u;
        for (uint j = 0u; j < K_ALL; j++) {
            tgOff[j] = run;
            const ushort cnt = ushort(atomic_load_explicit(&tgCnt[j], memory_order_relaxed));
            clusterSpans[macroIdx * K_ALL + j] = ushort2(run, cnt);
            run += cnt;
        }
        tgOff[K_ALL] = run;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < K_ALL) {
        ushort pos = tgOff[tid];
        device ushort* dstMem = memberPools + uint64_t(macroIdx) * 1024u;
        for (uint i = 0u; i < 1024u; i++) {
            if (tgAssign[i] == tid) dstMem[pos++] = ushort(i);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ushort startM = tgOff[cId];
    const ushort countM = tgOff[cId + 1u] - startM;
    device const ushort* srcMem = memberPools + uint64_t(macroIdx) * 1024u;
    float meanSub[32];
    for (uint d = 0u; d < subDim; d++) meanSub[d] = 0.0f;
    for (ushort k = 0u; k < countM; k++) {
        const uint pLocal = uint(srcMem[startM + k]);
        const float invP = tgInvNorm[pLocal];
        device const half* xp = chunkKeys + pLocal * indexDim + dBase;
        for (uint d = 0u; d < subDim; d++) meanSub[d] += float(xp[d]) * invP;
    }
    float mNorm2 = 0.0f;
    for (uint d = 0u; d < subDim; d++) {
        if (countM == 0u) meanSub[d] = float(tgCent[cId * 128u + dBase + d]);
        mNorm2 = fma(meanSub[d], meanSub[d], mNorm2);
    }
    mNorm2 += simd_shuffle_xor(mNorm2, 1u);
    mNorm2 += simd_shuffle_xor(mNorm2, 2u);
    const float invM = rsqrt(max(mNorm2, 1e-12f));
    device half* dstC = centroids + uint64_t(macroIdx * K_ALL + cId) * indexDim + dBase;
    for (uint d = 0u; d < subDim; d++) {
        dstC[d] = half(meanSub[d] * invM);
    }
}

// 2. Single-TG deterministic CSR candidate collection (fuses 2nd-byte radix + ascending tie-break + overflow flag).
kernel void qwg_emeans_collect(device const ushort*      memberPools  [[buffer(0)]],
                               device const ushort2*     clusterSpans [[buffer(1)]],
                               device const uint*        cScore       [[buffer(2)]],
                               device const uint*        uHist        [[buffer(3)]],
                               device uint*              candState    [[buffer(4)]],
                               device uint*              candPools    [[buffer(5)]],
                               constant EmeansSelParams& p            [[buffer(6)]],
                               uint2 tg  [[threadgroup_position_in_grid]],
                               uint  tid [[thread_index_in_threadgroup]]) {
    const uint unit = tg.y;
    if (unit >= p.T || tg.x > 0u) return;
    const EmeansSelGeom g = emeans_sel_geom(p, unit);
    if (g.dense) return;
    if (p.probeClusters == 0u) {
        if (tid == 0u) {
            candState[unit * 2u] = min(g.numMiddle, p.candCap);
            candState[unit * 2u + 1u] = (g.numMiddle > p.candCap) ? 1u : 0u;
        }
        return;
    }

    threadgroup uint hist[256];
    threadgroup uint scanSum[8];
    threadgroup uint greaterAccum, chosenBin, sgTotal;
    threadgroup uint runningTies, runningCandCount;
    const uint sg = tid >> 5u, lane = tid & 31u;

    if (tid == 0u) {
        runningTies = 0u;
        runningCandCount = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (g.numClusters > 0u) {
        hist[tid] = uHist[unit * 512u + tid];
        if (tid == 0u) { greaterAccum = 0u; chosenBin = 0u; sgTotal = 0u; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        emeans_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, g.probeEff, false, tid, sg, lane);
        const uint topByte = chosenBin << 24u;

        hist[tid] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint b = tid; b < g.numClusters; b += 256u) {
            const uint u = cScore[uint64_t(unit) * p.clusterStride + b];
            if ((u & 0xFF000000u) == topByte) {
                atomic_fetch_add_explicit((threadgroup atomic_uint*)&hist[(u >> 16u) & 0xFFu], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        emeans_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, g.probeEff, true, tid, sg, lane);
        const uint thresh16 = topByte | (chosenBin << 16u);
        const bool takeAllTies = (sgTotal <= (p.candCap / 20u));
        const uint maxTies = (g.probeEff > greaterAccum) ? (g.probeEff - greaterAccum) : 0u;

        const uint numChunks = (g.numClusters + 255u) >> 8u;
        for (uint c = 0u; c < numChunks; c++) {
            const uint b = (c << 8u) + tid;
            const ushort2 span = (b < g.numClusters) ? clusterSpans[b] : ushort2(0u, 0u);
            const bool nonEmpty = (span.y > 0u);
            const uint u16 = nonEmpty ? (cScore[uint64_t(unit) * p.clusterStride + b] & 0xFFFF0000u) : 0u;
            const bool isGt = nonEmpty && (takeAllTies ? (u16 >= thresh16) : (u16 > thresh16));
            const bool isTie = nonEmpty && !takeAllTies && (u16 == thresh16) && (maxTies > 0u);

            const uint tieVal = isTie ? 1u : 0u;
            const uint tieRank = simd_prefix_exclusive_sum(tieVal);
            const uint tieSum = simd_sum(tieVal);
            if (lane == 0u) scanSum[sg] = tieSum;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint tieBase = runningTies;
            for (uint s = 0u; s < sg; s++) tieBase += scanSum[s];
            const bool keepTie = isTie && ((tieBase + tieRank) < maxTies);
            if (tid == 0u) {
                uint chunkTies = 0u;
                for (uint s = 0u; s < 8u; s++) chunkTies += scanSum[s];
                runningTies += chunkTies;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint validCnt = 0u;
            const uint macroIdx = b >> 6u;
            const uint basePool = macroIdx * 1024u;
            device const ushort* mem = nonEmpty ? (memberPools + uint64_t(macroIdx) * 1024u + uint(span.x)) : nullptr;
            if (isGt || keepTie) {
                for (ushort k = 0u; k < span.y; k++) {
                    const uint m = basePool + uint(mem[k]);
                    if (m >= g.minPool && m < g.middleEnd) validCnt++;
                }
            }

            const uint candPrefix = simd_prefix_exclusive_sum(validCnt);
            const uint candSum = simd_sum(validCnt);
            if (lane == 0u) scanSum[sg] = candSum;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint slot0 = runningCandCount + candPrefix;
            for (uint s = 0u; s < sg; s++) slot0 += scanSum[s];
            if (validCnt > 0u && slot0 < p.candCap) {
                uint w = 0u;
                for (ushort k = 0u; k < span.y; k++) {
                    const uint m = basePool + uint(mem[k]);
                    if (m >= g.minPool && m < g.middleEnd) {
                        if (slot0 + w < p.candCap) candPools[uint64_t(unit) * p.candCap + slot0 + w] = m;
                        w++;
                    }
                }
            }
            if (tid == 0u) {
                uint chunkValid = 0u;
                for (uint s = 0u; s < 8u; s++) chunkValid += scanSum[s];
                runningCandCount += chunkValid;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint tailStart = max(g.minPool, g.numMacros * 1024u);
    const uint tailLen = (g.middleEnd > tailStart) ? (g.middleEnd - tailStart) : 0u;
    const uint baseTail = runningCandCount;
    for (uint i = tid; i < tailLen; i += 256u) {
        if (baseTail + i < p.candCap) candPools[uint64_t(unit) * p.candCap + baseTail + i] = tailStart + i;
    }
    if (tid == 0u) {
        const uint totalValid = baseTail + tailLen;
        candState[unit * 2u] = min(totalValid, p.candCap);
        candState[unit * 2u + 1u] = (totalValid > p.candCap) ? 1u : 0u;
    }
}

// 3. Final top-k radix selection + bitonic ascending pool writeout + zero-at-rest cleanup.
kernel void qwg_emeans_finalize(device int*               blockList  [[buffer(0)]],
                                device uint*              blockCount [[buffer(1)]],
                                device uint*              uHist      [[buffer(2)]],
                                device uint*              candState  [[buffer(3)]],
                                device const uint*        candPools  [[buffer(4)]],
                                device const uint*        candU      [[buffer(5)]],
                                constant EmeansSelParams& p          [[buffer(6)]],
                                uint unit [[threadgroup_position_in_grid]],
                                uint tid  [[thread_index_in_threadgroup]]) {
    if (unit >= p.T) return;
    const EmeansSelGeom g = emeans_sel_geom(p, unit);
    device int* outRow = blockList + uint64_t(g.qRow) * p.listStride;
    if (g.dense) {
        for (uint b = tid; b < g.totalBlocks; b += 256u) outRow[b] = int(b);
        if (p.writeCount != 0u && tid == 0u) blockCount[g.qRow] = g.totalBlocks;
        return;
    }

    const bool flatContig = (p.probeClusters == 0u && g.numMiddle <= p.candCap);
    const bool overflow = (candState[unit * 2u + 1u] != 0u) || (p.probeClusters == 0u && g.numMiddle > p.candCap);
    if (overflow) {
        uHist[unit * 512u + tid] = 0u;
        uHist[unit * 512u + 256u + tid] = 0u;
        if (tid == 0u) candState[unit * 2u + 1u] = 1u;
        return;
    }

    constexpr uint MAX_KEEP = 2048u;
    threadgroup uint hist[256];
    threadgroup uint scanSum[8];
    threadgroup uint greaterAccum, chosenBin, sgTotal;
    threadgroup uint keep[MAX_KEEP];
    threadgroup atomic_uint gtSlot;

    const uint sg = tid >> 5u, lane = tid & 31u;
    const uint nCand = flatContig ? g.numMiddle : min(candState[unit * 2u], p.candCap);
    const uint kTarget = min(min(g.kEff, nCand), MAX_KEEP);
    device const uint* cm = candPools + uint64_t(unit) * p.candCap;
    device const uint* cu = candU + uint64_t(unit) * p.candCap;

    uint nPow2 = 256u;
    while (nPow2 < kTarget) nPow2 <<= 1u;
    for (uint i = tid; i < nPow2; i += 256u) keep[i] = 0xFFFFFFFFu;
    if (tid == 0u) {
        greaterAccum = 0u; chosenBin = 0u; sgTotal = 0u;
        atomic_store_explicit(&gtSlot, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (kTarget > 0u) {
        uint threshU = 0u, mask = 0u;
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
            emeans_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, kTarget,
                                  pass == 3u, tid, sg, lane);
            threshU |= (chosenBin << shift);
            mask |= (0xFFu << shift);
        }

        const uint needTies = kTarget - greaterAccum;
        uint minTiePool = 0u;
        if (sgTotal > kTarget && needTies > 0u) {
            if (tid == 0u) greaterAccum = 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint tieMask = 0u;
            for (uint p2 = 0u; p2 < 3u; p2++) {
                const uint sh2 = 16u - (p2 << 3u);
                hist[tid] = 0u;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = tid; i < nCand; i += 256u) {
                    if (cu[i] == threshU) {
                        const uint m = flatContig ? (g.minPool + i) : cm[i];
                        if ((m & tieMask) == minTiePool) {
                            atomic_fetch_add_explicit((threadgroup atomic_uint*)&hist[(m >> sh2) & 0xFFu], 1u,
                                                      memory_order_relaxed);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                emeans_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, needTies,
                                      p2 == 2u, tid, sg, lane);
                minTiePool |= (chosenBin << sh2);
                tieMask |= (0xFFu << sh2);
            }
        }

        for (uint i = tid; i < nCand; i += 256u) {
            const uint u = cu[i];
            const uint m = flatContig ? (g.minPool + i) : cm[i];
            if (u > threshU || (u == threshU && needTies > 0u && m >= minTiePool)) {
                const uint s = atomic_fetch_add_explicit(&gtSlot, 1u, memory_order_relaxed);
                if (s < kTarget) keep[s] = m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    emeans_sort_and_emit(outRow, blockCount, p, g, keep, kTarget, nPow2, tid);

    uHist[unit * 512u + tid] = 0u;
    uHist[unit * 512u + 256u + tid] = 0u;
    if (tid < 2u) candState[unit * 2u + tid] = 0u;
}
