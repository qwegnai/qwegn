// qwlsa.metal — Latent Sparse Attention (LSA) indexing, 4-token key pooling,
// dynamic threadgroup Ward clustering (512 tokens -> 64 buckets, cap 16 keys),
// and QSA hierarchical block selection (`qwg_lsa_qsa_select`).
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

static inline void lsa_pick_radix_bin_qsa(threadgroup uint* hist,
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

// 1. qwg_lsa_index_prep: projects unrotated post-RMSNorm Q/K slices into
// L2-normalized qIdxOut [T, indexDim] and raw kPending [ringCap, indexDim].
kernel void qwg_lsa_index_prep(device const half* qgate      [[buffer(0)]],
                               device const half* kIn        [[buffer(1)]],
                               device const half* wqIdx      [[buffer(2)]],
                               device const half* wkIdx      [[buffer(3)]],
                               device half*       qIdxOut    [[buffer(4)]],
                               device half*       kPending   [[buffer(5)]],
                               constant uint&     T          [[buffer(6)]],
                               constant uint&     pos0       [[buffer(7)]],
                               constant uint&     qHeads     [[buffer(8)]],
                               constant uint&     kvHeads    [[buffer(9)]],
                               constant uint&     headDim    [[buffer(10)]],
                               constant uint&     qProjOut   [[buffer(11)]],
                               constant uint&     qStride    [[buffer(12)]],
                               constant uint&     sliceStart [[buffer(13)]],
                               constant uint&     sliceDim   [[buffer(14)]],
                               constant uint&     indexDim   [[buffer(15)]],
                               constant uint&     ringMask   [[buffer(16)]],
                               uint t   [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    if (t >= T) return;
    threadgroup float qSlice[2048];
    threadgroup float kSlice[2048];
    threadgroup float qProj[128];
    threadgroup float partial[4];

    const uint inputDim = min(kvHeads * sliceDim, 2048u);
    const uint gqaRatio = max(1u, qHeads / max(1u, kvHeads));
    const float invGqa = 1.0f / float(gqaRatio);
    device const half* qRow = qgate + uint64_t(t) * qProjOut;
    device const half* kRow = kIn + uint64_t(t) * (kvHeads * headDim);

    for (uint idx = tid; idx < inputDim; idx += 128u) {
        const uint kvh = idx / sliceDim;
        const uint c = idx - kvh * sliceDim;
        const uint d = sliceStart + c;
        kSlice[idx] = float(kRow[kvh * headDim + d]);
        float qSum = 0.0f;
        const uint hBase = kvh * gqaRatio;
        for (uint r = 0u; r < gqaRatio; r++) {
            qSum += float(qRow[(hBase + r) * qStride * headDim + d]);
        }
        qSlice[idx] = qSum * invGqa;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float qVal = 0.0f;
    float kVal = 0.0f;
    if (tid < indexDim) {
        device const half* wqRow = wqIdx + uint64_t(tid) * inputDim;
        device const half* wkRow = wkIdx + uint64_t(tid) * inputDim;
        float accQ = 0.0f;
        float accK = 0.0f;
        for (uint j = 0u; j < inputDim; j++) {
            accQ = fma(qSlice[j], float(wqRow[j]), accQ);
            accK = fma(kSlice[j], float(wkRow[j]), accK);
        }
        qVal = accQ;
        kVal = accK;
        qProj[tid] = qVal;
    }

    const float ss = simd_sum(qVal * qVal);
    if ((tid & 31u) == 0u) partial[tid >> 5u] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float qNormSq = partial[0] + partial[1] + partial[2] + partial[3];
    const float qInv = rsqrt(max(qNormSq, 1e-12f));
    const uint ringSlot = (pos0 + t) & ringMask;

    if (tid < indexDim) {
        qIdxOut[t * indexDim + tid] = half(qProj[tid] * qInv);
        kPending[ringSlot * indexDim + tid] = half(kVal);
    }
}

// 2. qwg_lsa_pool_keys: pools 4 consecutive raw index keys from kPending into
// a unit L2-normalized pooled key in lsaPooledKeys [maxBlocks, indexDim].
kernel void qwg_lsa_pool_keys(device const half* kPending   [[buffer(0)]],
                              device half*       pooledKeys [[buffer(1)]],
                              constant uint&     startBlock [[buffer(2)]],
                              constant uint&     numBlocks  [[buffer(3)]],
                              constant uint&     indexDim   [[buffer(4)]],
                              constant uint&     ringMask   [[buffer(5)]],
                              uint tg  [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
    if (tg >= numBlocks) return;
    threadgroup float partial[4];
    const uint b = startBlock + tg;
    const uint baseTok = b * 4u;

    float meanVal = 0.0f;
    if (tid < indexDim) {
        float sum4 = 0.0f;
        for (uint k = 0u; k < 4u; k++) {
            const uint slot = (baseTok + k) & ringMask;
            sum4 += float(kPending[slot * indexDim + tid]);
        }
        meanVal = sum4 * 0.25f;
    }
    const float ss = simd_sum(meanVal * meanVal);
    if ((tid & 31u) == 0u) partial[tid >> 5u] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float normSq = partial[0] + partial[1] + partial[2] + partial[3];
    const float invNorm = rsqrt(max(normSq, 1e-12f));
    if (tid < indexDim) {
        pooledKeys[b * indexDim + tid] = half(meanVal * invNorm);
    }
}

// 3. qwg_lsa_ward_cluster_512: dynamic greedy Ward clustering over 128 pooled
// 4-token blocks (512 tokens) into 64 buckets with hard cap <= 4 blocks (16 tokens).
kernel void qwg_lsa_ward_cluster_512(device const half* pooledKeys    [[buffer(0)]],
                                     device half*       centroids     [[buffer(1)]],
                                     device uchar4*     bucketMembers [[buffer(2)]],
                                     device uchar*      bucketSizes   [[buffer(3)]],
                                     constant uint&     startCluster  [[buffer(4)]],
                                     constant uint&     numClusters   [[buffer(5)]],
                                     constant uint&     indexDim      [[buffer(6)]],
                                     uint tg  [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]]) {
    if (tg >= numClusters) return;
    const uint cIdx = startCluster + tg;
    const uint baseBlock = cIdx * 128u;
    const uint sg = tid >> 5u;
    const uint lane = tid & 31u;

    threadgroup half   tgSum[128 * 64];
    threadgroup float  tgNormSq[128];
    threadgroup uchar4 tgMembers[128];
    threadgroup uchar  tgCount[128];
    threadgroup float  tgBestCost[128];
    threadgroup uchar  tgBestPartner[128];
    threadgroup float  sgMinCost[4];
    threadgroup uint   sgMinPair[4];
    threadgroup uint   tgWinnerA;
    threadgroup uint   tgWinnerB;

    const uint stride = max(1u, indexDim / 64u);
    device const half* myPool = pooledKeys + uint64_t(baseBlock + tid) * indexDim;
    float normSq = 0.0f;
    float vLocal[64];
    for (uint d = 0u; d < 64u; d++) {
        const uint srcD = min(d * stride, indexDim - 1u);
        const float val = float(myPool[srcD]);
        vLocal[d] = val;
        normSq = fma(val, val, normSq);
    }
    const float invN = rsqrt(max(normSq, 1e-12f));
    for (uint d = 0u; d < 64u; d++) {
        tgSum[tid * 64u + d] = half(vLocal[d] * invN);
    }
    tgNormSq[tid] = 1.0f;
    tgMembers[tid] = uchar4(uchar(tid), 255u, 255u, 255u);
    tgCount[tid] = 1u;
    tgBestCost[tid] = 1e30f;
    tgBestPartner[tid] = 255u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 127u) {
        float bestC = 1e30f;
        uchar bestJ = 255u;
        const float nSqI = tgNormSq[tid];
        for (uint j = tid + 1u; j < 128u; j++) {
            float dot = 0.0f;
            for (uint d = 0u; d < 64u; d += 4u) {
                const float4 a = float4(*(threadgroup const half4*)(&tgSum[tid * 64u + d]));
                const float4 b = float4(*(threadgroup const half4*)(&tgSum[j * 64u + d]));
                dot += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
            const float distSq = max(0.0f, nSqI + tgNormSq[j] - 2.0f * dot);
            const float wardCost = 0.5f * distSq;
            if (wardCost < bestC) {
                bestC = wardCost;
                bestJ = uchar(j);
            }
        }
        tgBestCost[tid] = bestC;
        tgBestPartner[tid] = bestJ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = 0u; step < 64u; step++) {
        if (step > 0u) {
            const uint prevA = tgWinnerA;
            const uint prevB = tgWinnerB;
            const uchar myCnt = tgCount[tid];
            if (myCnt > 0u && myCnt < 4u && tid < 127u) {
                const uchar partner = tgBestPartner[tid];
                const bool needFullRescan = (tid == prevA) || (partner == prevA) || (partner == prevB) ||
                    (partner < 128u && (tgCount[partner] == 0u || (myCnt + tgCount[partner]) > 4u));
                if (needFullRescan) {
                    float bestC = 1e30f;
                    uchar bestJ = 255u;
                    const float invSi = 1.0f / float(myCnt);
                    const float nSqI = tgNormSq[tid] * (invSi * invSi);
                    for (uint j = tid + 1u; j < 128u; j++) {
                        const uchar cntJ = tgCount[j];
                        if (cntJ == 0u || (myCnt + cntJ) > 4u) continue;
                        const float invSj = 1.0f / float(cntJ);
                        float dot = 0.0f;
                        for (uint d = 0u; d < 64u; d += 4u) {
                            const float4 a = float4(*(threadgroup const half4*)(&tgSum[tid * 64u + d]));
                            const float4 b = float4(*(threadgroup const half4*)(&tgSum[j * 64u + d]));
                            dot += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                        }
                        const float distSq = max(0.0f, nSqI + tgNormSq[j] * (invSj * invSj) - 2.0f * dot * (invSi * invSj));
                        const float wardCost = (float(myCnt * cntJ) / float(myCnt + cntJ)) * distSq;
                        if (wardCost < bestC) {
                            bestC = wardCost;
                            bestJ = uchar(j);
                        }
                    }
                    tgBestCost[tid] = bestC;
                    tgBestPartner[tid] = bestJ;
                } else if (tid < prevA && tgCount[prevA] > 0u && (myCnt + tgCount[prevA]) <= 4u) {
                    const uchar cntA = tgCount[prevA];
                    float dot = 0.0f;
                    for (uint d = 0u; d < 64u; d += 4u) {
                        const float4 a = float4(*(threadgroup const half4*)(&tgSum[tid * 64u + d]));
                        const float4 b = float4(*(threadgroup const half4*)(&tgSum[prevA * 64u + d]));
                        dot += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                    }
                    const float invSi = 1.0f / float(myCnt);
                    const float invSa = 1.0f / float(cntA);
                    const float distSq = max(0.0f,
                        tgNormSq[tid] * (invSi * invSi) + tgNormSq[prevA] * (invSa * invSa) - 2.0f * dot * (invSi * invSa));
                    const float wardCost = (float(myCnt * cntA) / float(myCnt + cntA)) * distSq;
                    if (wardCost < tgBestCost[tid]) {
                        tgBestCost[tid] = wardCost;
                        tgBestPartner[tid] = uchar(prevA);
                    }
                }
            } else {
                tgBestCost[tid] = 1e30f;
                tgBestPartner[tid] = 255u;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float cVal = (tgCount[tid] > 0u && tgBestPartner[tid] < 128u) ? tgBestCost[tid] : 1e30f;
        uint  pVal = (tid << 8u) | uint(tgBestPartner[tid]);
        for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
            const float otherC = simd_shuffle_down(cVal, offset);
            const uint  otherP = simd_shuffle_down(pVal, offset);
            if (otherC < cVal || (otherC == cVal && otherP < pVal)) {
                cVal = otherC;
                pVal = otherP;
            }
        }
        if (lane == 0u) {
            sgMinCost[sg] = cVal;
            sgMinPair[sg] = pVal;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float winC = sgMinCost[0];
            uint  winP = sgMinPair[0];
            for (uint s = 1u; s < 4u; s++) {
                if (sgMinCost[s] < winC || (sgMinCost[s] == winC && sgMinPair[s] < winP)) {
                    winC = sgMinCost[s];
                    winP = sgMinPair[s];
                }
            }
            uint wa = (winP >> 8u) & 0xFFu;
            uint wb = winP & 0xFFu;
            if (wa >= 128u || wb >= 128u || wa == wb || tgCount[wa] == 0u || tgCount[wb] == 0u) {
                wa = 255u;
                wb = 255u;
                for (uint i = 0u; i < 128u && wa == 255u; i++) {
                    if (tgCount[i] == 0u || tgCount[i] >= 4u) continue;
                    for (uint j = i + 1u; j < 128u; j++) {
                        if (tgCount[j] > 0u && (tgCount[i] + tgCount[j]) <= 4u) {
                            wa = i;
                            wb = j;
                            break;
                        }
                    }
                }
            }
            tgWinnerA = wa;
            tgWinnerB = wb;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint wa = tgWinnerA;
        const uint wb = tgWinnerB;
        if (wa < 128u && wb < 128u && tid < 64u) {
            const float sumD = float(tgSum[wa * 64u + tid]) + float(tgSum[wb * 64u + tid]);
            tgSum[wa * 64u + tid] = half(sumD);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (wa < 128u && wb < 128u) {
            const float v = (tid < 64u) ? float(tgSum[wa * 64u + tid]) : 0.0f;
            const float ss = simd_sum(v * v);
            if (lane == 0u) sgMinCost[sg] = ss;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0u) {
                tgNormSq[wa] = sgMinCost[0] + sgMinCost[1];
                const uchar ca = tgCount[wa];
                const uchar cb = tgCount[wb];
                uchar mA[4] = { tgMembers[wa].x, tgMembers[wa].y, tgMembers[wa].z, tgMembers[wa].w };
                const uchar mB[4] = { tgMembers[wb].x, tgMembers[wb].y, tgMembers[wb].z, tgMembers[wb].w };
                for (uchar k = 0u; k < cb && (ca + k) < 4u; k++) {
                    mA[ca + k] = mB[k];
                }
                tgMembers[wa] = uchar4(mA[0], mA[1], mA[2], mA[3]);
                tgCount[wa] = ca + cb;
                tgCount[wb] = 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint isAlive = (tgCount[tid] > 0u) ? 1u : 0u;
    const uint prefix = simd_prefix_exclusive_sum(isAlive);
    const uint sgTot = simd_sum(isAlive);
    if (lane == 0u) sgMinPair[sg] = sgTot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint bucketIdx = prefix;
    for (uint s = 0u; s < sg; s++) bucketIdx += sgMinPair[s];

    if (isAlive != 0u && bucketIdx < 64u) {
        const uchar4 mem = tgMembers[tid];
        const uchar cnt = tgCount[tid];
        const uint outBucket = cIdx * 64u + bucketIdx;
        bucketMembers[outBucket] = mem;
        bucketSizes[outBucket] = uchar(cnt * 4u);

        const uchar mArr[4] = { mem.x, mem.y, mem.z, mem.w };
        float normAcc = 0.0f;
        float centroidVec[128];
        const uint dLim = min(indexDim, 128u);
        for (uint d = 0u; d < dLim; d++) {
            float s = 0.0f;
            for (uchar k = 0u; k < cnt; k++) {
                s += float(pooledKeys[uint64_t(baseBlock + uint(mArr[k])) * indexDim + d]);
            }
            centroidVec[d] = s;
            normAcc = fma(s, s, normAcc);
        }
        const float invNorm = rsqrt(max(normAcc, 1e-12f));
        device half* dstC = centroids + uint64_t(outBucket) * indexDim;
        for (uint d = 0u; d < dLim; d++) {
            dstC[d] = half(centroidVec[d] * invNorm);
        }
    }
}

// 5. qwg_lsa_qsa_select: hierarchical Ward-clustered selector for Qwen3.8-Flash-Next
// (d_idx = 128, 4 query index heads). Emits `blockTopK` ascending 4-token blocks.
kernel void qwg_lsa_qsa_select(device const float*  idxQ          [[buffer(0)]],
                               device const half*   pooled        [[buffer(1)]],
                               device const half*   centroids     [[buffer(2)]],
                               device const uchar4* bucketMembers [[buffer(3)]],
                               device const uchar*  bucketSizes   [[buffer(4)]],
                               device uint*         scores        [[buffer(5)]],
                               device int*          list          [[buffer(6)]],
                               constant uint&       T             [[buffer(7)]],
                               constant uint&       blockTopK     [[buffer(8)]],
                               constant uint&       pos0          [[buffer(9)]],
                               constant uint&       scoreStride   [[buffer(10)]],
                               uint t   [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    if (t >= T) return;
    constexpr uint DI = 128u, NI = 4u;
    threadgroup float qs[NI * DI];
    threadgroup uint  hist[256];
    threadgroup uint  scanSum[8];
    threadgroup uint  sgTotal;
    threadgroup uint  chosenBin;
    threadgroup uint  greaterAccum;

    const uint pq = pos0 + t;
    const uint n = (pq + 1u) >> 2u;
    if (n <= blockTopK) {
        for (uint b = tid; b < n; b += 256u) {
            list[t * blockTopK + b] = int(b);
        }
        return;
    }

    const uint sg = tid >> 5u;
    const uint lane = tid & 31u;
    for (uint i = tid; i < NI * DI; i += 256u) {
        qs[i] = idxQ[t * NI * DI + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device uint* rowScores = scores + uint64_t(t) * scoreStride;
    const float invSqrtD = 1.0f / sqrt(float(DI));
    const uint numClusters = n >> 7u;
    const uint numBuckets = numClusters * 64u;

    float centroidThresh = -1e30f;
    if (numBuckets > blockTopK * 2u) {
        hist[tid] = 0u;
        if (tid == 0u) greaterAccum = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint b = tid; b < numBuckets; b += 256u) {
            device const half4* cb = (device const half4*)(centroids + uint64_t(b) * DI);
            float cScore = 0.0f;
            for (uint h = 0u; h < NI; h++) {
                threadgroup const float4* qh = (threadgroup const float4*)(qs + h * DI);
                float dot = 0.0f;
                for (uint d4 = 0u; d4 < DI / 4u; d4++) {
                    const float4 qv = qh[d4], cv = float4(cb[d4]);
                    dot += qv.x * cv.x + qv.y * cv.y + qv.z * cv.z + qv.w * cv.w;
                }
                cScore += max(dot, 0.0f);
            }
            const uint u = as_type<uint>((cScore * invSqrtD) + 0.0f);
            atomic_fetch_add_explicit((threadgroup atomic_uint*)&hist[(u >> 24u) & 0xFFu], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint targetBuckets = min(numBuckets, blockTopK);
        lsa_pick_radix_bin_qsa(hist, scanSum, greaterAccum, chosenBin, sgTotal, targetBuckets, true, tid, sg, lane);
        centroidThresh = as_type<float>(chosenBin << 24u);
    }

    if (centroidThresh > 0.0f) {
        for (uint b = tid; b < numClusters * 128u; b += 256u) {
            rowScores[b] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint b = tid; b < numBuckets; b += 256u) {
            device const half4* cb = (device const half4*)(centroids + uint64_t(b) * DI);
            float cScore = 0.0f;
            for (uint h = 0u; h < NI; h++) {
                threadgroup const float4* qh = (threadgroup const float4*)(qs + h * DI);
                float dot = 0.0f;
                for (uint d4 = 0u; d4 < DI / 4u; d4++) {
                    const float4 qv = qh[d4], cv = float4(cb[d4]);
                    dot += qv.x * cv.x + qv.y * cv.y + qv.z * cv.z + qv.w * cv.w;
                }
                cScore += max(dot, 0.0f);
            }
            if ((cScore * invSqrtD) >= centroidThresh || b == 0u || (b >> 6u) + 1u == numClusters) {
                const uint baseBlk = (b >> 6u) * 128u;
                const uchar4 mem = bucketMembers[b];
                const uchar mArr[4] = { mem.x, mem.y, mem.z, mem.w };
                const uint nMem = min(4u, uint(bucketSizes[b]) >> 2u);
                for (uint k = 0u; k < nMem; k++) {
                    const uint m = baseBlk + uint(mArr[k]);
                    if (m < n) {
                        device const half4* pb = (device const half4*)(pooled + uint64_t(m) * DI);
                        float score = 0.0f;
                        for (uint h = 0u; h < NI; h++) {
                            threadgroup const float4* qh = (threadgroup const float4*)(qs + h * DI);
                            float dot = 0.0f;
                            for (uint d4 = 0u; d4 < DI / 4u; d4++) {
                                const float4 qv = qh[d4], pv = float4(pb[d4]);
                                dot += qv.x * pv.x + qv.y * pv.y + qv.z * pv.z + qv.w * pv.w;
                            }
                            score += max(dot, 0.0f);
                        }
                        rowScores[m] = as_type<uint>((score * invSqrtD) + 0.0f);
                    }
                }
            }
        }
        for (uint m = numClusters * 128u + tid; m < n; m += 256u) {
            device const half4* pb = (device const half4*)(pooled + uint64_t(m) * DI);
            float score = 0.0f;
            for (uint h = 0u; h < NI; h++) {
                threadgroup const float4* qh = (threadgroup const float4*)(qs + h * DI);
                float dot = 0.0f;
                for (uint d4 = 0u; d4 < DI / 4u; d4++) {
                    const float4 qv = qh[d4], pv = float4(pb[d4]);
                    dot += qv.x * pv.x + qv.y * pv.y + qv.z * pv.z + qv.w * pv.w;
                }
                score += max(dot, 0.0f);
            }
            rowScores[m] = as_type<uint>((score * invSqrtD) + 0.0f);
        }
    } else {
        for (uint m = tid; m < n; m += 256u) {
            device const half4* pb = (device const half4*)(pooled + uint64_t(m) * DI);
            float score = 0.0f;
            for (uint h = 0u; h < NI; h++) {
                threadgroup const float4* qh = (threadgroup const float4*)(qs + h * DI);
                float dot = 0.0f;
                for (uint d4 = 0u; d4 < DI / 4u; d4++) {
                    const float4 qv = qh[d4], pv = float4(pb[d4]);
                    dot += qv.x * pv.x + qv.y * pv.y + qv.z * pv.z + qv.w * pv.w;
                }
                score += max(dot, 0.0f);
            }
            rowScores[m] = as_type<uint>((score * invSqrtD) + 0.0f);
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

        for (uint i = tid; i < n; i += 256u) {
            const uint u = rowScores[i];
            if ((u & mask) == threshU) {
                const uint bin = (u >> shift) & 0xFFu;
                atomic_fetch_add_explicit((threadgroup atomic_uint*)&hist[bin], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        lsa_pick_radix_bin_qsa(hist, scanSum, greaterAccum, chosenBin, sgTotal, blockTopK, pass == 3u, tid, sg, lane);
        threshU |= (chosenBin << shift);
        mask |= (0xFFu << shift);
    }

    const uint skipTies = sgTotal - blockTopK;
    device int* outRow = list + uint64_t(t) * blockTopK;
    if (tid == 0u) {
        chosenBin = 0u;
        greaterAccum = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint numChunks = (n + 255u) >> 8u;
    for (uint c = 0u; c < numChunks; c++) {
        const uint idx = (c << 8u) + tid;
        const uint u = (idx < n) ? rowScores[idx] : 0u;
        const bool isGt = (idx < n) && (u > threshU);
        const bool isTie = (idx < n) && (u == threshU);

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
        if (keep != 0u && (wBase + wRank) < blockTopK) {
            outRow[wBase + wRank] = int(idx);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint chunkKept = 0u;
            for (uint s = 0u; s < 8u; s++) chunkKept += scanSum[s];
            chosenBin += chunkKept;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
