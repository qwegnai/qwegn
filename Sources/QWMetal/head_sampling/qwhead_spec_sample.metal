// qwhead_spec_sample.metal — Metal 3.2+ stochastic draft sampler & batched LC verifier.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_fastmath_safe.metalh"
using namespace metal;

#define QW_SPEC_MAX_CAND 512u
#define QW_SPEC_MAX_REP  256u
#define QW_SPEC_EMPTY_ID 0xFFFFFFFFu

struct QWSpecHeader {
    int   token;
    uint  mode;
    uint  count;
    uint  overflow;
    uint  finite;
    float maxLogit;
    float logSumExp;
    float logZ;
    float totalSum;
    float cutoffLogit;
    float temperature;
    uint  topK;
    float topP;
    int   gumbelToken;
    float gumbelLogProb;
    uint  reserved;
};

static inline float qw_safe_exp(float x) {
    return (x < -88.0f) ? 0.0f : exp(x);
}

static inline uint4 qw_philox4(uint4 ctr, uint k0, uint k1) {
    const uint d1 = 0xD2511F53u, d2 = 0xCD9E8D57u;
    for (uint r = 0u; r < 10u; ++r) {
        uint h0_hi = mulhi(d1, ctr.x), h0_lo = d1 * ctr.x;
        uint h1_hi = mulhi(d2, ctr.z), h1_lo = d2 * ctr.z;
        ctr = uint4(h1_hi ^ ctr.y ^ k0, h1_lo, h0_hi ^ ctr.w ^ k1, h0_lo);
        k0 += d2;
        k1 += d1;
    }
    return ctr;
}

static inline float qw_philox_u24(uint word) {
    return float(word >> 8u) * (1.0f / 16777216.0f);
}

static inline float qw_philox_u24_offset(uint lo, uint hi, ulong delta, uint k0, uint k1) {
    ulong n = ((ulong(hi) << 32u) | ulong(lo)) + delta;
    uint4 r = qw_philox4(uint4(uint(n & 0xFFFFFFFFull), uint(n >> 32u), 0u, 0u), k0, k1);
    return qw_philox_u24(r.x);
}

static inline void qw_spec_init_rep_mask(
        device const uint* repTokens, uint repSlot, uint repCount, uint tid,
        threadgroup uint* sRep, threadgroup atomic_uint* sBloom) {
    if (repCount == 0u) return;
    if (tid < 16u) atomic_store_explicit(&sBloom[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < repCount) {
        uint rt = repTokens[(size_t)repSlot * QW_SPEC_MAX_REP + tid];
        sRep[tid] = rt;
        uint h = (rt * 2654435761u) >> 23u;
        atomic_fetch_or_explicit(&sBloom[h >> 5u], 1u << (h & 31u), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline float qw_spec_apply_rep(
        float raw, uint v, uint repCount, float repPenalty,
        threadgroup const uint* sRep, threadgroup atomic_uint* sBloom) {
    if (repCount == 0u || repPenalty == 1.0f) return raw;
    uint h = (v * 2654435761u) >> 23u;
    uint bits = atomic_load_explicit(&sBloom[h >> 5u], memory_order_relaxed);
    if ((bits & (1u << (h & 31u))) == 0u) return raw;
    for (uint j = 0u; j < repCount; ++j) {
        if (sRep[j] == v) return (raw < 0.0f) ? (raw * repPenalty) : (raw / repPenalty);
    }
    return raw;
}

static inline float qw_tg_max_256(float m, bool hasVal, uint tid, threadgroup float* sMax, threadgroup uint* sHas) {
    const uint sg = tid >> 5u, lane = tid & 31u;
    float sgMax = simd_max(hasVal ? m : -3.4028234e38f);
    uint sgHas = simd_max(hasVal ? 1u : 0u);
    if (lane == 0u) { sMax[sg] = sgMax; sHas[sg] = sgHas; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float gm = -3.4028234e38f;
        uint gHas = 0u;
        for (uint q = 0u; q < 8u; ++q) {
            if (sHas[q] != 0u) { gm = (gHas == 0u) ? sMax[q] : max(gm, sMax[q]); gHas = 1u; }
        }
        sMax[0] = gm; sHas[0] = gHas;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sMax[0];
}

static inline bool qw_cand_desc_before(float aVal, uint aId, float bVal, uint bId) {
    if (aId == QW_SPEC_EMPTY_ID) return false;
    if (bId == QW_SPEC_EMPTY_ID) return true;
    return (aVal != bVal) ? (aVal > bVal) : (aId < bId);
}

static inline void qw_bitonic_sort_512(threadgroup float* sVal, threadgroup uint* sId, uint tid, bool byIdAsc) {
    for (uint k = 2u; k <= QW_SPEC_MAX_CAND; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            uint low = tid & (j - 1u);
            uint i = ((tid - low) << 1u) + low, ixj = i | j;
            bool dir = ((i & k) == 0u);
            float va = sVal[i], vb = sVal[ixj];
            uint ia = sId[i], ib = sId[ixj];
            bool firstBefore = byIdAsc ? (ia < ib) : qw_cand_desc_before(va, ia, vb, ib);
            if (firstBefore != dir) {
                sVal[i] = vb; sVal[ixj] = va;
                sId[i] = ib; sId[ixj] = ia;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

static inline float qw_lookup_sorted_prob(
        device const uint* ids, device const float* probs, uint count, uint target) {
    uint lo = 0u, hi = count;
    while (lo < hi) {
        uint mid = lo + ((hi - lo) >> 1u), val = ids[mid];
        if (val == target) return probs[mid];
        if (val < target) lo = mid + 1u; else hi = mid;
    }
    return 0.0f;
}

static inline float qw_lookup_tg_prob(
        threadgroup const uint* ids, threadgroup const float* probs, uint count, uint target) {
    uint lo = 0u, hi = count;
    while (lo < hi) {
        uint mid = lo + ((hi - lo) >> 1u), val = ids[mid];
        if (val == target) return probs[mid];
        if (val < target) lo = mid + 1u; else hi = mid;
    }
    return 0.0f;
}

static inline uint qw_sample_cdf_tg(
        threadgroup const uint* ids, threadgroup const float* weights, uint count, float mass, float u) {
    float cum = 0.0f;
    uint chosen = 0u, lastPos = 0u;
    bool found = false;
    for (uint i = 0u; i < count; ++i) {
        float w = weights[i];
        if (w > 0.0f) lastPos = i;
        cum += w / mass;
        if (!found && cum > u) { chosen = i; found = true; }
    }
    return ids[found ? chosen : lastPos];
}

// Builds normalized top-k/top-p sparse distribution in TG SRAM (sorted ascending by token ID).
static inline bool qw_build_sparse_dist(
        device const uint* inIds, device const float* inLogits, uint rawCount,
        QWSpecHeader hdr, uint vocab, float invT, uint effK, float topP, uint tid,
        threadgroup float* sVal, threadgroup uint* sId,
        threadgroup float* sTmp, threadgroup uint* sCtrl, thread uint& outKeepN) {
    for (uint i = tid; i < QW_SPEC_MAX_CAND; i += 256u) {
        sVal[i] = (i < rawCount) ? inLogits[i] : -3.4028234e38f;
        sId[i] = (i < rawCount) ? inIds[i] : QW_SPEC_EMPTY_ID;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qw_bitonic_sort_512(sVal, sId, tid, false);

    float localSum = 0.0f;
    for (uint i = tid; i < QW_SPEC_MAX_CAND; i += 256u) {
        float w = (i < rawCount) ? qw_safe_exp((sVal[i] - hdr.maxLogit) * invT) : 0.0f;
        sVal[i] = w;
        localSum += w;
    }
    float sgSum = simd_sum(localSum);
    if ((tid & 31u) == 0u) sTmp[tid >> 5u] = sgSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float candW = 0.0f;
        for (uint q = 0u; q < 8u; ++q) candW += sTmp[q];
        float candProbSum = candW / hdr.totalSum;
        bool gateOK = (rawCount >= effK) || (candProbSum >= topP);
        uint need = 1u;
        float keptMass = 0.0f;
        if (gateOK) {
            float cum = 0.0f;
            uint limit = min(effK, rawCount);
            for (uint r = 0u; r < limit; ++r) {
                float before = cum;
                cum += sVal[r] / hdr.totalSum;
                if (before < topP) need = r + 1u; else break;
            }
            for (uint r = 0u; r < need; ++r) keptMass += sVal[r];
            if (!qw_is_finite_bits(keptMass) || keptMass <= 0.0f) gateOK = false;
        }
        sCtrl[0] = gateOK ? 1u : 0u;
        sCtrl[1] = need;
        sTmp[0] = keptMass;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sCtrl[0] == 0u) return false;

    const uint keepN = sCtrl[1];
    const float keptMass = sTmp[0];
    for (uint i = tid; i < QW_SPEC_MAX_CAND; i += 256u) {
        if (i < keepN) { sVal[i] = sVal[i] / keptMass; }
        else { sVal[i] = 0.0f; sId[i] = QW_SPEC_EMPTY_ID; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qw_bitonic_sort_512(sVal, sId, tid, true);
    outKeepN = keepN;
    return true;
}

// Stage 1 stats: 256 TGs x 256 threads per row, 1 coalesced read of logits[V].
kernel void qwg_spec_stats(
        device const float* logits        [[buffer(0)]],
        device const uint*  repTokens     [[buffer(1)]],
        device const uint*  repCounts     [[buffer(2)]],
        device float*       tgMax         [[buffer(3)]],
        device float*       tgSum         [[buffer(4)]],
        device uint*        tgFinite      [[buffer(5)]],
        device uint*        tgGHi         [[buffer(6)]],
        device uint*        tgGLo         [[buffer(7)]],
        constant uint&      vocab         [[buffer(8)]],
        constant float&     invT          [[buffer(9)]],
        constant float&     repPenalty    [[buffer(10)]],
        constant uint&      hasRepMask    [[buffer(11)]],
        constant uint&      repSlotBase   [[buffer(12)]],
        constant uint&      computeGumbel [[buffer(13)]],
        constant uint&      seedHi        [[buffer(14)]],
        constant uint&      seedLo        [[buffer(15)]],
        constant uint&      philoxBaseLo  [[buffer(16)]],
        constant uint&      philoxBaseHi  [[buffer(17)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint row = tg.y, repSlot = repSlotBase + row;
    const uint repCount = (hasRepMask != 0u) ? min(repCounts[repSlot], QW_SPEC_MAX_REP) : 0u;
    threadgroup uint sRep[QW_SPEC_MAX_REP];
    threadgroup atomic_uint sBloom[16];
    qw_spec_init_rep_mask(repTokens, repSlot, repCount, tid, sRep, sBloom);

    device const float* rowLogits = logits + (size_t)row * vocab;
    const ulong ctr = ((ulong(philoxBaseHi) << 32u) | ulong(philoxBaseLo)) + ulong(row);
    const uint ctrLo = uint(ctr & 0xFFFFFFFFull), ctrHi = uint(ctr >> 32u);

    bool hasVal = false;
    float m = -3.4028234e38f, s = 0.0f;
    uint fin = 1u, bestGHi = 0u, bestGLo = 0u;
    for (uint v = tg.x * 256u + tid; v < vocab; v += 65536u) {
        float raw = rowLogits[v];
        if (!qw_is_finite_bits(raw)) { fin = 0u; continue; }
        float z = qw_spec_apply_rep(raw, v, repCount, repPenalty, sRep, sBloom);
        if (!qw_is_finite_bits(z)) { fin = 0u; continue; }
        if (!hasVal) { m = z; s = 1.0f; hasVal = true; }
        else if (z > m) { s = s * qw_safe_exp((m - z) * invT) + 1.0f; m = z; }
        else { s += qw_safe_exp((z - m) * invT); }
        if (computeGumbel != 0u) {
            uint4 r = qw_philox4(uint4(ctrLo, ctrHi, v, 0x474D424Cu), seedHi, seedLo);
            float u = max(qw_philox_u24(r.x), 1.0f / 16777216.0f);
            uint gh = qw_sortable_f32_u32(z * invT - log(-log(u))), gl = ~v;
            if (gh > bestGHi || (gh == bestGHi && gl > bestGLo)) { bestGHi = gh; bestGLo = gl; }
        }
    }

    const uint sg = tid >> 5u, lane = tid & 31u;
    threadgroup float sMax[8], sSum[8];
    threadgroup uint sHas[8], sFin[8], sGHi[8], sGLo[8];
    float tgM = qw_tg_max_256(m, hasVal, tid, sMax, sHas);
    float rescaled = hasVal ? (s * qw_safe_exp((m - tgM) * invT)) : 0.0f;
    float sgSum = simd_sum(rescaled);
    uint sgFin = simd_min(fin), sgGHi = simd_max(bestGHi);
    uint sgGLo = simd_max((bestGHi == sgGHi) ? bestGLo : 0u);
    if (lane == 0u) { sSum[sg] = sgSum; sFin[sg] = sgFin; sGHi[sg] = sgGHi; sGLo[sg] = sgGLo; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float totalS = 0.0f;
        uint allFin = 1u, gh = 0u, gl = 0u;
        for (uint q = 0u; q < 8u; ++q) {
            totalS += sSum[q];
            allFin = min(allFin, sFin[q]);
            if (sGHi[q] > gh || (sGHi[q] == gh && sGLo[q] > gl)) { gh = sGHi[q]; gl = sGLo[q]; }
        }
        size_t outIdx = (size_t)row * 256u + tg.x;
        tgMax[outIdx] = tgM;
        tgSum[outIdx] = (sHas[0] != 0u) ? totalS : 0.0f;
        tgFinite[outIdx] = allFin;
        tgGHi[outIdx] = gh;
        tgGLo[outIdx] = gl;
    }
}

// Stage 2 stats: 1 TG x 256 threads per row -> global (maxZ, logZ, gumbel, cutoff).
kernel void qwg_spec_stats_fin(
        device const float*  logits           [[buffer(0)]],
        device const uint*   repTokens        [[buffer(1)]],
        device const uint*   repCounts        [[buffer(2)]],
        device const float*  tgMax            [[buffer(3)]],
        device const float*  tgSum            [[buffer(4)]],
        device const uint*   tgFinite         [[buffer(5)]],
        device const uint*   tgGHi            [[buffer(6)]],
        device const uint*   tgGLo            [[buffer(7)]],
        device QWSpecHeader* headers          [[buffer(8)]],
        device uint*         candCounts       [[buffer(9)]],
        device uint*         candOverflows    [[buffer(10)]],
        device int*          draftTokenOut    [[buffer(11)]],
        device int*          journalOut       [[buffer(12)]],
        constant uint&       vocab            [[buffer(13)]],
        constant float&      temperature      [[buffer(14)]],
        constant uint&       topK             [[buffer(15)]],
        constant float&      topP             [[buffer(16)]],
        constant float&      repPenalty       [[buffer(17)]],
        constant uint&       hasRepMask       [[buffer(18)]],
        constant uint&       repSlotBase      [[buffer(19)]],
        constant uint&       computeGumbel    [[buffer(20)]],
        constant uint&       isDraft          [[buffer(21)]],
        constant uint&       proposalSlotBase [[buffer(22)]],
        constant uint&       hasJournal       [[buffer(23)]],
        constant uint&       journalOffset    [[buffer(24)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint row = tg.y, sg = tid >> 5u, lane = tid & 31u;
    const size_t idx = (size_t)row * 256u + tid;
    const float m = tgMax[idx], s = tgSum[idx], invT = 1.0f / temperature;
    const uint fin = tgFinite[idx];
    const bool has = (s > 0.0f);

    threadgroup float sMax[8], sSum[8];
    threadgroup uint sHas[8], sFin[8], sGHi[8], sGLo[8];
    const float maxZ = qw_tg_max_256(m, has, tid, sMax, sHas);
    const float sgSum = simd_sum(has ? (s * qw_safe_exp((m - maxZ) * invT)) : 0.0f);
    const uint sgFin = simd_min(fin);
    const uint ghIn = (computeGumbel != 0u) ? tgGHi[idx] : 0u;
    const uint glIn = (computeGumbel != 0u) ? tgGLo[idx] : 0u;
    const uint sgGHi = simd_max(ghIn), sgGLo = simd_max((ghIn == sgGHi) ? glIn : 0u);
    if (lane == 0u) { sSum[sg] = sgSum; sFin[sg] = sgFin; sGHi[sg] = sgGHi; sGLo[sg] = sgGLo; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float totalSum = 0.0f;
        uint allFin = sHas[0], bestGH = 0u, bestGL = 0u;
        for (uint q = 0u; q < 8u; ++q) {
            totalSum += sSum[q];
            allFin = min(allFin, sFin[q]);
            if (sGHi[q] > bestGH || (sGHi[q] == bestGH && sGLo[q] > bestGL)) {
                bestGH = sGHi[q]; bestGL = sGLo[q];
            }
        }
        const bool ok = (allFin != 0u) && qw_is_finite_bits(totalSum) && (totalSum > 0.0f);
        const float logSumExp = ok ? log(max(totalSum, 1.17549435e-38f)) : 0.0f;
        const float logZ = ok ? (maxZ * invT + logSumExp) : 0.0f;
        int gTok = 0;
        float gLogP = 0.0f;
        if (ok && computeGumbel != 0u && bestGH != 0u) {
            uint cand = ~bestGL;
            if (cand < vocab) {
                gTok = int(cand);
                float raw = logits[(size_t)row * vocab + cand];
                if (hasRepMask != 0u && repPenalty != 1.0f) {
                    const uint repSlot = repSlotBase + row;
                    const uint repCount = min(repCounts[repSlot], QW_SPEC_MAX_REP);
                    for (uint j = 0u; j < repCount; ++j) {
                        if (repTokens[(size_t)repSlot * QW_SPEC_MAX_REP + j] == cand) {
                            raw = (raw < 0.0f) ? (raw * repPenalty) : (raw / repPenalty);
                            break;
                        }
                    }
                }
                gLogP = (raw - maxZ) * invT - logSumExp;
            }
        }
        const uint effK = min(topK, vocab);
        const float pMax = 1.0f / totalSum;
        float cutoffProb = (topP < 1.0f) ? clamp((1.0f - topP) / 512.0f, 1e-4f, 1e-3f) : 1e-4f;
        if (effK <= QW_SPEC_MAX_CAND && (float(effK) * pMax) < topP) {
            cutoffProb = max(1e-4f, min(pMax * 0.5f, 1.0f / float(QW_SPEC_MAX_CAND)));
        }
        float cutoffLogit = (vocab <= QW_SPEC_MAX_CAND)
            ? -3.4028234e38f
            : (maxZ + temperature * (log(cutoffProb) + logSumExp) - 1e-4f * temperature);

        const uint outRow = proposalSlotBase + row;
        QWSpecHeader hdr = {
            ok ? gTok : -1, 1u, 0u, 0u, ok ? 1u : 0u,
            maxZ, logSumExp, logZ, totalSum, cutoffLogit,
            temperature, topK, topP, ok ? gTok : -1, gLogP, 0u
        };
        headers[outRow] = hdr;
        candCounts[row] = 0u;
        candOverflows[row] = 0u;
        if (isDraft != 0u && (!ok || (topP >= 1.0f && topK >= vocab))) {
            draftTokenOut[row] = hdr.token;
            if (hasJournal != 0u) journalOut[journalOffset + row] = hdr.token;
        }
    }
}

// Collects candidate tokens >= cutoffLogit into 512-slot buffer via 32-bit atomics.
kernel void qwg_spec_collect_trunc(
        device const float*        logits           [[buffer(0)]],
        device const uint*         repTokens        [[buffer(1)]],
        device const uint*         repCounts        [[buffer(2)]],
        device const QWSpecHeader* headers          [[buffer(3)]],
        device uint*               candIds          [[buffer(4)]],
        device float*              candLogits       [[buffer(5)]],
        device atomic_uint*        candCounts       [[buffer(6)]],
        device atomic_uint*        candOverflows    [[buffer(7)]],
        constant uint&             vocab            [[buffer(8)]],
        constant float&            repPenalty       [[buffer(9)]],
        constant uint&             hasRepMask       [[buffer(10)]],
        constant uint&             repSlotBase      [[buffer(11)]],
        constant uint&             proposalSlotBase [[buffer(12)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint row = tg.y;
    const QWSpecHeader hdr = headers[proposalSlotBase + row];
    if (hdr.finite == 0u) return;

    const uint repSlot = repSlotBase + row;
    const uint repCount = (hasRepMask != 0u) ? min(repCounts[repSlot], QW_SPEC_MAX_REP) : 0u;
    threadgroup uint sRep[QW_SPEC_MAX_REP];
    threadgroup atomic_uint sBloom[16];
    qw_spec_init_rep_mask(repTokens, repSlot, repCount, tid, sRep, sBloom);

    device const float* rowLogits = logits + (size_t)row * vocab;
    device uint* outIds = candIds + (size_t)row * QW_SPEC_MAX_CAND;
    device float* outLogits = candLogits + (size_t)row * QW_SPEC_MAX_CAND;
    const float cutoff = hdr.cutoffLogit;

    for (uint v = tg.x * 256u + tid; v < vocab; v += 65536u) {
        float z = qw_spec_apply_rep(rowLogits[v], v, repCount, repPenalty, sRep, sBloom);
        if (z >= cutoff) {
            uint slot = atomic_fetch_add_explicit(&candCounts[row], 1u, memory_order_relaxed);
            if (slot < QW_SPEC_MAX_CAND) { outIds[slot] = v; outLogits[slot] = z; }
            else { atomic_store_explicit(&candOverflows[row], 1u, memory_order_relaxed); }
        }
    }
}

// 1 TG x 256 threads (4 KiB TG SRAM): exact gate, bitonic sort, top-k/top-p draw.
kernel void qwg_spec_sample_trunc(
        device QWSpecHeader* headers          [[buffer(0)]],
        device const uint*   candIds          [[buffer(1)]],
        device const float*  candLogits       [[buffer(2)]],
        device const uint*   candCounts       [[buffer(3)]],
        device const uint*   candOverflows    [[buffer(4)]],
        device uint*         outIds           [[buffer(5)]],
        device float*        outProbs         [[buffer(6)]],
        device int*          draftTokenOut    [[buffer(7)]],
        device int*          journalOut       [[buffer(8)]],
        constant uint&       vocab            [[buffer(9)]],
        constant float&      temperature      [[buffer(10)]],
        constant uint&       topK             [[buffer(11)]],
        constant float&      topP             [[buffer(12)]],
        constant uint&       seedHi           [[buffer(13)]],
        constant uint&       seedLo           [[buffer(14)]],
        constant uint&       philoxBaseLo     [[buffer(15)]],
        constant uint&       philoxBaseHi     [[buffer(16)]],
        constant uint&       proposalSlotBase [[buffer(17)]],
        constant uint&       hasJournal       [[buffer(18)]],
        constant uint&       journalOffset    [[buffer(19)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint row = tg.y, outRow = proposalSlotBase + row;
    const QWSpecHeader hdr = headers[outRow];
    if (hdr.finite == 0u) {
        if (tid == 0u) {
            headers[outRow].token = -1; headers[outRow].mode = 1u; headers[outRow].overflow = 1u;
            draftTokenOut[row] = -1;
            if (hasJournal != 0u) journalOut[journalOffset + row] = -1;
        }
        return;
    }

    const uint rawCount = candCounts[row], ov = candOverflows[row];
    threadgroup float sVal[QW_SPEC_MAX_CAND], sTmp[8];
    threadgroup uint  sId[QW_SPEC_MAX_CAND], sCtrl[4];
    uint keepN = 0u;
    bool built = false;
    if (ov == 0u && rawCount > 0u && rawCount <= QW_SPEC_MAX_CAND) {
        built = qw_build_sparse_dist(
            candIds + (size_t)row * QW_SPEC_MAX_CAND,
            candLogits + (size_t)row * QW_SPEC_MAX_CAND,
            rawCount, hdr, vocab, 1.0f / temperature, min(topK, vocab), topP,
            tid, sVal, sId, sTmp, sCtrl, keepN);
    }
    if (!built) {
        if (tid == 0u) {
            headers[outRow].token = hdr.gumbelToken;
            headers[outRow].mode = 1u;
            headers[outRow].count = 0u;
            headers[outRow].overflow = 1u;
            draftTokenOut[row] = hdr.gumbelToken;
            if (hasJournal != 0u) journalOut[journalOffset + row] = hdr.gumbelToken;
        }
        return;
    }

    device uint* dstIds = outIds + (size_t)outRow * QW_SPEC_MAX_CAND;
    device float* dstProbs = outProbs + (size_t)outRow * QW_SPEC_MAX_CAND;
    for (uint i = tid; i < keepN; i += 256u) { dstIds[i] = sId[i]; dstProbs[i] = sVal[i]; }
    if (tid == 0u) {
        float u = qw_philox_u24_offset(philoxBaseLo, philoxBaseHi, ulong(row), seedHi, seedLo);
        uint chosen = qw_sample_cdf_tg(sId, sVal, keepN, 1.0f, u);
        headers[outRow].token = int(chosen);
        headers[outRow].mode = 0u;
        headers[outRow].count = keepN;
        headers[outRow].overflow = 0u;
        draftTokenOut[row] = int(chosen);
        if (hasJournal != 0u) journalOut[journalOffset + row] = int(chosen);
    }
}

// Phase 2 GPU Leviathan-Chen verifier: 1 TG resolves B = D + 1 rows sequentially.
kernel void qwg_spec_verify_lc_resolve(
        device const QWSpecHeader* targetHeaders  [[buffer(0)]],
        device const uint*         candIds        [[buffer(1)]],
        device const float*        candLogits     [[buffer(2)]],
        device const uint*         candCounts     [[buffer(3)]],
        device const uint*         candOverflows  [[buffer(4)]],
        device const QWSpecHeader* draftHeaders   [[buffer(5)]],
        device const uint*         draftIds       [[buffer(6)]],
        device const float*        draftProbs     [[buffer(7)]],
        device int*                resolvedTokens [[buffer(8)]],
        device uint*               verifyMeta     [[buffer(9)]],
        constant uint&             draftCount     [[buffer(10)]],
        constant uint&             targetVocab    [[buffer(11)]],
        constant float&            temperature    [[buffer(12)]],
        constant uint&             topK           [[buffer(13)]],
        constant float&            topP           [[buffer(14)]],
        constant uint&             seedHi         [[buffer(15)]],
        constant uint&             seedLo         [[buffer(16)]],
        constant uint&             philoxBaseLo   [[buffer(17)]],
        constant uint&             philoxBaseHi   [[buffer(18)]],
        constant uint&             draftSlotBase  [[buffer(19)]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint totalRows = draftCount + 1u;
    threadgroup float sVal[QW_SPEC_MAX_CAND], sRes[QW_SPEC_MAX_CAND], sTmp[8];
    threadgroup uint  sId[QW_SPEC_MAX_CAND], sCtrl[4];

    if (tid == 0u) {
        uint status = 1u;
        for (uint r = 0u; r < totalRows; ++r) {
            if (targetHeaders[r].finite == 0u) { status = 2u; break; }
        }
        if (status == 1u) {
            for (uint r = 0u; r < draftCount; ++r) {
                QWSpecHeader dh = draftHeaders[draftSlotBase + r];
                if (dh.finite == 0u) { status = 2u; break; }
                if (dh.mode != 0u || dh.count == 0u || dh.count > QW_SPEC_MAX_CAND || dh.token < 0) status = 0u;
            }
        }
        sCtrl[0] = status;
        if (status != 1u) { verifyMeta[0] = status; verifyMeta[1] = 0u; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sCtrl[0] != 1u) return;

    const float invT = 1.0f / temperature;
    const uint effK = min(topK, targetVocab);

    for (uint row = 0u; row < totalRows; ++row) {
        const QWSpecHeader th = targetHeaders[row];
        const uint rawCount = candCounts[row], ov = candOverflows[row];
        uint keepN = 0u;
        bool built = false;
        if (ov == 0u && rawCount > 0u && rawCount <= QW_SPEC_MAX_CAND) {
            built = qw_build_sparse_dist(
                candIds + (size_t)row * QW_SPEC_MAX_CAND,
                candLogits + (size_t)row * QW_SPEC_MAX_CAND,
                rawCount, th, targetVocab, invT, effK, topP, tid, sVal, sId, sTmp, sCtrl, keepN);
        }
        if (!built) {
            if (tid == 0u) { verifyMeta[0] = 0u; verifyMeta[1] = 0u; }
            return;
        }

        if (row == draftCount) {
            if (tid == 0u) {
                float uB = qw_philox_u24_offset(philoxBaseLo, philoxBaseHi, ulong(2u * draftCount), seedHi, seedLo);
                resolvedTokens[draftCount] = int(qw_sample_cdf_tg(sId, sVal, keepN, 1.0f, uB));
                verifyMeta[0] = 1u;
                verifyMeta[1] = draftCount;
            }
            return;
        }

        const uint dSlot = draftSlotBase + row;
        const QWSpecHeader dh = draftHeaders[dSlot];
        const uint dTok = uint(dh.token);
        device const uint* dIds = draftIds + (size_t)dSlot * QW_SPEC_MAX_CAND;
        device const float* dProbs = draftProbs + (size_t)dSlot * QW_SPEC_MAX_CAND;

        if (tid == 0u) {
            float pD = (dTok < targetVocab) ? qw_lookup_tg_prob(sId, sVal, keepN, dTok) : 0.0f;
            float qD = qw_lookup_sorted_prob(dIds, dProbs, dh.count, dTok);
            float uAcc = qw_philox_u24_offset(philoxBaseLo, philoxBaseHi, ulong(2u * row), seedHi, seedLo);
            bool accepted = (pD > 0.0f) && (qD <= 0.0f || pD >= qD || uAcc < (pD / qD));
            sCtrl[2] = accepted ? 1u : 0u;
            if (accepted) resolvedTokens[row] = int(dTok);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sCtrl[2] != 0u) continue;

        float localRes = 0.0f;
        for (uint i = tid; i < keepN; i += 256u) {
            float rv = max(0.0f, sVal[i] - qw_lookup_sorted_prob(dIds, dProbs, dh.count, sId[i]));
            sRes[i] = rv;
            localRes += rv;
        }
        float sgRes = simd_sum(localRes);
        if ((tid & 31u) == 0u) sTmp[tid >> 5u] = sgRes;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float resMass = 0.0f;
            for (uint q = 0u; q < 8u; ++q) resMass += sTmp[q];
            float uRes = qw_philox_u24_offset(philoxBaseLo, philoxBaseHi, ulong(2u * row + 1u), seedHi, seedLo);
            uint tok = (resMass > 0.0f && qw_is_finite_bits(resMass))
                ? qw_sample_cdf_tg(sId, sRes, keepN, resMass, uRes)
                : qw_sample_cdf_tg(sId, sVal, keepN, 1.0f, uRes);
            resolvedTokens[row] = int(tok);
            verifyMeta[0] = 1u;
            verifyMeta[1] = row;
        }
        return;
    }
}
