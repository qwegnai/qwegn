// qwattn_mla_dsa_prep.metal — Decoupled-RoPE + DSA latent/query prep kernels.
// Supports standard RMSNorm or exact BF16-rounded low-rank GatedNorm on q_a/kv_a,
// alphaQ/alphaKV scaling, LayerNorm or RMSNorm k_idx, fused oGateInQKV stride, and H_128.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "../include/qw_mla_dequant.metalh"
#include "../include/qw_bf16_math.metalh"
#include "../include/qw_geometry_constants.metalh"
using namespace metal;

struct MLADSALatentPrepParams {
    uint  T;
    uint  pos0;
    uint  qLoraRank;
    uint  kvLoraRank;
    uint  ropeDim;
    uint  idxDim;
    uint  latentStride;
    uint  kvMode;
    float rmsEps;
    float kNormEps;
    float mscale;
    uint  useHadamard;
    uint  ropeFlags;
    uint  gatedFlags;
    uint  qAGateRank;
    uint  kvAGateRank;
    float qAGatingWeight;
    float kvAGatingWeight;
    float alphaQ;
    float alphaKV;
};

struct MLAQRopeIdxPrepParams {
    uint  T;
    uint  pos0;
    uint  qHeads;
    uint  idxHeads;
    uint  ropeDim;
    uint  idxDim;
    uint  qRopeOffset;
    uint  qChannels;
    uint  qFullStride;
    float mscale;
    uint  useHadamard;
    uint  ropeFlags;
};

// Architectural orthonormal 128-point Sylvester Walsh-Hadamard transform (H_128 / sqrt(128))
// across 32 lanes x float4; active regardless of QW_FWHT_ROTATED weight-basis compilation.
METAL_FUNC float4 qw_mla_fwht128_ortho(float4 v, uint lane) {
    const float a0 = v.x + v.y, a1 = v.x - v.y, a2 = v.z + v.w, a3 = v.z - v.w;
    v = float4(a0 + a2, a1 + a3, a0 - a2, a1 - a3);
    #pragma unroll
    for (ushort step = 1u; step < 32u; step <<= 1u) {
        const float4 mate = simd_shuffle_xor(v, step);
        v = ((lane & uint(step)) == 0u) ? (v + mate) : (mate - v);
    }
    return v * 0.08838834764831844f;
}

METAL_FUNC float qw_mla_prep_bf16_f32(float v) {
    #pragma clang fp contract(off) reassociate(off)
    if (!qw_is_finite_bits(v)) return v;
    uint u = as_type<uint>(v);
    u = u + 0x7FFFu + ((u >> 16u) & 1u);
    return as_type<float>(u & 0xFFFF0000u);
}

METAL_FUNC ushort qw_mla_prep_f32_to_bf16_bits(float v) {
    #pragma clang fp contract(off) reassociate(off)
    if (!qw_is_finite_bits(v)) return as_type<ushort>(bfloat(v));
    uint u = as_type<uint>(v);
    u = u + 0x7FFFu + ((u >> 16u) & 1u);
    return ushort(u >> 16u);
}

METAL_FUNC float qw_mla_prep_load_w(device const ushort* p, size_t idx, bool isBF16) {
    const ushort bits = p[idx];
    return isBF16 ? as_type<float>(uint(bits) << 16u) : float(as_type<half>(bits));
}

METAL_FUNC void qw_mla_prep_stage_gated(
        device const half* xRow,
        device const ushort* normW,
        device const ushort* gateUp,
        uint dim,
        uint rank,
        float eps,
        bool wBF16,
        bool upBF16,
        bool zeroCentered,
        uint tid,
        uint sg,
        uint lane,
        threadgroup ushort* shH,
        threadgroup float* shG1,
        threadgroup float* sgScratch) {
    #pragma clang fp contract(off) reassociate(off)
    float sq = 0.0f;
    for (uint c = tid; c < dim; c += 256u) {
        const float xv = float(xRow[c]);
        sq += xv * xv;
    }
    sq = qwg_simd_reduce_down<32u>(sq);
    if (lane == 0u) sgScratch[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float sumSq = ((sgScratch[0] + sgScratch[1]) + (sgScratch[2] + sgScratch[3]))
                      + ((sgScratch[4] + sgScratch[5]) + (sgScratch[6] + sgScratch[7]));
    const float rstd = precise::divide(1.0f, precise::sqrt(precise::divide(sumSq, float(dim)) + eps));
    for (uint c = tid; c < dim; c += 256u) {
        const float wv = qw_mla_prep_load_w(normW, c, wBF16);
        const float scale = zeroCentered ? (1.0f + wv) : wv;
        const float hVal = (float(xRow[c]) * rstd) * scale;
        shH[c] = qw_mla_prep_f32_to_bf16_bits(hVal);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = sg; r < rank; r += 8u) {
        float acc = 0.0f;
        const size_t upOff = (size_t)r * dim;
        for (uint c = lane; c < dim; c += 32u) {
            acc += qw_mla_prep_load_w(gateUp, upOff + c, upBF16) * as_type<float>(uint(shH[c]) << 16u);
        }
        acc = qwg_simd_reduce_down<32u>(acc);
        if (lane == 0u) {
            shG1[r] = qw_mla_prep_bf16_f32(precise::divide(acc, 1.0f + precise::exp(-acc)));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

METAL_FUNC float qw_mla_prep_eval_gated(
        uint idx,
        uint rank,
        float gatingWeight,
        bool downBF16,
        device const ushort* gateDown,
        threadgroup const ushort* shH,
        threadgroup const float* shG1) {
    #pragma clang fp contract(off) reassociate(off)
    const size_t downOff = (size_t)idx * rank;
    float dotD = 0.0f;
    if (rank == 16u && (((size_t)gateDown & 15u) == 0u)) {
        device const uint4* dp4 = (device const uint4*)(gateDown + downOff);
        const uint4 d0 = dp4[0], d1 = dp4[1];
        if (downBF16) {
            dotD += as_type<float>(d0.x << 16u) * shG1[0];
            dotD += as_type<float>(d0.x & 0xFFFF0000u) * shG1[1];
            dotD += as_type<float>(d0.y << 16u) * shG1[2];
            dotD += as_type<float>(d0.y & 0xFFFF0000u) * shG1[3];
            dotD += as_type<float>(d0.z << 16u) * shG1[4];
            dotD += as_type<float>(d0.z & 0xFFFF0000u) * shG1[5];
            dotD += as_type<float>(d0.w << 16u) * shG1[6];
            dotD += as_type<float>(d0.w & 0xFFFF0000u) * shG1[7];
            dotD += as_type<float>(d1.x << 16u) * shG1[8];
            dotD += as_type<float>(d1.x & 0xFFFF0000u) * shG1[9];
            dotD += as_type<float>(d1.y << 16u) * shG1[10];
            dotD += as_type<float>(d1.y & 0xFFFF0000u) * shG1[11];
            dotD += as_type<float>(d1.z << 16u) * shG1[12];
            dotD += as_type<float>(d1.z & 0xFFFF0000u) * shG1[13];
            dotD += as_type<float>(d1.w << 16u) * shG1[14];
            dotD += as_type<float>(d1.w & 0xFFFF0000u) * shG1[15];
        } else {
            const half2 h0 = as_type<half2>(d0.x), h1 = as_type<half2>(d0.y);
            const half2 h2 = as_type<half2>(d0.z), h3 = as_type<half2>(d0.w);
            const half2 h4 = as_type<half2>(d1.x), h5 = as_type<half2>(d1.y);
            const half2 h6 = as_type<half2>(d1.z), h7 = as_type<half2>(d1.w);
            dotD += float(h0.x) * shG1[0]; dotD += float(h0.y) * shG1[1];
            dotD += float(h1.x) * shG1[2]; dotD += float(h1.y) * shG1[3];
            dotD += float(h2.x) * shG1[4]; dotD += float(h2.y) * shG1[5];
            dotD += float(h3.x) * shG1[6]; dotD += float(h3.y) * shG1[7];
            dotD += float(h4.x) * shG1[8]; dotD += float(h4.y) * shG1[9];
            dotD += float(h5.x) * shG1[10]; dotD += float(h5.y) * shG1[11];
            dotD += float(h6.x) * shG1[12]; dotD += float(h6.y) * shG1[13];
            dotD += float(h7.x) * shG1[14]; dotD += float(h7.y) * shG1[15];
        }
    } else {
        for (uint r = 0u; r < rank; ++r) {
            dotD += qw_mla_prep_load_w(gateDown, downOff + r, downBF16) * shG1[r];
        }
    }
    const float g2 = qw_mla_prep_bf16_f32(precise::divide(1.0f, 1.0f + precise::exp(-dotD)));
    const float hVal = as_type<float>(uint(shH[idx]) << 16u);
    return qw_mla_prep_bf16_f32((hVal * gatingWeight) * g2);
}

// 1. Fused q_a/kv_a norm (+ optional GatedNorm + alphaQ/alphaKV) + paged KV write + k_pe/k_idx RoPE + H_128.
// Grid: (T, 1, 1), Threadgroup: (256, 1, 1).
kernel void qwg_mla_dsa_latent_prep(
    device const half*              mlaLatent   [[buffer(0)]],
    device const ushort*            qANormW     [[buffer(1)]],
    device const ushort*            kvANormW    [[buffer(2)]],
    device const ushort*            kNormW      [[buffer(3)]],
    device const half*              kNormB      [[buffer(4)]],
    device const float*             invFreq     [[buffer(5)]],
    device half*                    qResidOut   [[buffer(6)]],
    device half*                    kPool       [[buffer(7)]],
    device half*                    vPool       [[buffer(8)]],
    device const int*               kTable      [[buffer(9)]],
    device const int*               vTable      [[buffer(10)]],
    device half*                    kPeCache    [[buffer(11)]],
    device half*                    pooledKeys  [[buffer(12)]],
    constant MLADSALatentPrepParams& p          [[buffer(13)]],
    device const int*               mropePos    [[buffer(14)]],
    device const uint*              ropeAxisMap [[buffer(15)]],
    device const ushort*            qAGateUp    [[buffer(16)]],
    device const ushort*            qAGateDown  [[buffer(17)]],
    device const ushort*            kvAGateUp   [[buffer(18)]],
    device const ushort*            kvAGateDown [[buffer(19)]],
    threadgroup ushort*             shH         [[threadgroup(0)]],
    uint                            t           [[threadgroup_position_in_grid]],
    uint                            tid         [[thread_index_in_threadgroup]],
    uint                            sg          [[simdgroup_index_in_threadgroup]],
    uint                            lane        [[thread_index_in_simdgroup]]
) {
    if (t >= p.T) return;
    threadgroup float sgScratch[8];
    threadgroup float sgMinScratch[8];
    threadgroup half2 q4Sm[2];
    threadgroup float shG1[32];
    threadgroup half shX[512];

    const uint pos = p.pos0 + t;
    const uint defaultStride = p.qLoraRank + p.kvLoraRank + p.ropeDim + p.idxDim;
    const uint latentStride = (p.latentStride > 0u) ? p.latentStride : defaultStride;
    device const half* rowLat = mlaLatent + (size_t)t * latentStride;

    const bool hasQAGated = (p.gatedFlags & 1u) != 0u;
    const bool qAWBF16 = (p.gatedFlags & 2u) != 0u;
    const bool qAUpBF16 = (p.gatedFlags & 4u) != 0u;
    const bool qADownBF16 = (p.gatedFlags & 8u) != 0u;
    const bool qAZeroCentered = (p.gatedFlags & 16u) != 0u;
    const bool hasKVAGated = (p.gatedFlags & 32u) != 0u;
    const bool kvAWBF16 = (p.gatedFlags & 64u) != 0u;
    const bool kvAUpBF16 = (p.gatedFlags & 128u) != 0u;
    const bool kvADownBF16 = (p.gatedFlags & 256u) != 0u;
    const bool kvAZeroCentered = (p.gatedFlags & 512u) != 0u;
    const bool kNormWBF16 = (p.gatedFlags & 1024u) != 0u;
    const float alphaQ = (p.alphaQ != 0.0f) ? p.alphaQ : 1.0f;
    const float alphaKV = (p.alphaKV != 0.0f) ? p.alphaKV : 1.0f;

    if (p.qLoraRank > 0u) {
        device half* dstQResid = qResidOut + (size_t)t * p.qLoraRank;
        if (hasQAGated) {
            qw_mla_prep_stage_gated(
                rowLat, qANormW, qAGateUp, p.qLoraRank, p.qAGateRank, p.rmsEps,
                qAWBF16, qAUpBF16, qAZeroCentered, tid, sg, lane, shH, shG1, sgScratch);
            for (uint base = sg * 32u; base < p.qLoraRank; base += 256u) {
                const uint idx = base + lane;
                float v = qw_mla_prep_eval_gated(
                    idx, p.qAGateRank, p.qAGatingWeight, qADownBF16, qAGateDown, shH, shG1);
                v *= alphaQ;
                dstQResid[idx] = half(qwg_fwht32_lane_f32(v, lane));
            }
        } else {
            float sqQ = 0.0f;
            for (uint i = tid; i < p.qLoraRank; i += 256u) {
                const float v = float(rowLat[i]);
                sqQ = fma(v, v, sqQ);
            }
            sqQ = simd_sum(sqQ);
            if (lane == 0u) sgScratch[sg] = sqQ;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0u) {
                float s = (lane < 8u) ? sgScratch[lane] : 0.0f;
                s = simd_sum(s);
                if (lane == 0u) sgScratch[0] = rsqrt(s / float(p.qLoraRank) + p.rmsEps);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float invRmsQ = sgScratch[0];

            for (uint base = sg * 32u; base < p.qLoraRank; base += 256u) {
                const uint idx = base + lane;
                const float wQ = qw_mla_prep_load_w(qANormW, idx, qAWBF16);
                const float scQ = qAZeroCentered ? (1.0f + wQ) : wQ;
                float v = float(rowLat[idx]) * invRmsQ * scQ * alphaQ;
                dstQResid[idx] = half(qwg_fwht32_lane_f32(v, lane));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device const half* kvLat = rowLat + p.qLoraRank;
    float invRmsKV = 0.0f;
    if (hasKVAGated) {
        qw_mla_prep_stage_gated(
            kvLat, kvANormW, kvAGateUp, p.kvLoraRank, p.kvAGateRank, p.rmsEps,
            kvAWBF16, kvAUpBF16, kvAZeroCentered, tid, sg, lane, shH, shG1, sgScratch);
    } else {
        float sqKV = 0.0f;
        for (uint i = tid; i < p.kvLoraRank; i += 256u) {
            const float v = float(kvLat[i]);
            sqKV = fma(v, v, sqKV);
        }
        sqKV = simd_sum(sqKV);
        if (lane == 0u) sgScratch[sg] = sqKV;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float s = (lane < 8u) ? sgScratch[lane] : 0.0f;
            s = simd_sum(s);
            if (lane == 0u) sgScratch[0] = rsqrt(s / float(p.kvLoraRank) + p.rmsEps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        invRmsKV = sgScratch[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint pageSlot = pos >> 8u;
    const uint pageOff = pos & 255u;
    const uint pageK = uint(kTable[pageSlot]);
    const uint pageV = uint(vTable[pageSlot]);
    const uint halfRank = p.kvLoraRank >> 1u;

    for (uint base = 0u; base < p.kvLoraRank; base += 256u) {
        const uint idx = base + (sg * 32u + lane);
        float v = 0.0f;
        if (hasKVAGated) {
            v = qw_mla_prep_eval_gated(
                idx, p.kvAGateRank, p.kvAGatingWeight, kvADownBF16, kvAGateDown, shH, shG1);
        } else {
            const float wKV = qw_mla_prep_load_w(kvANormW, idx, kvAWBF16);
            const float scKV = kvAZeroCentered ? (1.0f + wKV) : wKV;
            v = float(kvLat[idx]) * invRmsKV * scKV;
        }
        v *= alphaKV;
        v = qwg_fwht32_lane_f32(v, lane);
        const uint col = idx & (halfRank - 1u);
        device uchar* poolBytes = (base < halfRank) ? (device uchar*)kPool : (device uchar*)vPool;
        const uint pageID = (base < halfRank) ? pageK : pageV;

        if (p.kvMode == 0u) {
            device half* dst = ((device half*)poolBytes) + ((size_t)pageID * 256u + pageOff) * halfRank;
            dst[col] = half(v);
        } else if (p.kvMode == 1u) {
            float maxAbs = simd_max(fabs(v));
            if (lane == 0u) sgScratch[sg] = maxAbs;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0u) {
                float mVal = (lane < 8u) ? sgScratch[lane] : 0.0f;
                mVal = simd_max(mVal);
                if (lane == 0u) sgScratch[0] = float(half(max(mVal / 127.0f, 6.103515625e-5f)));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const half scH = half(sgScratch[0]);
            const float invSc = (float(scH) > 0.0f) ? (1.0f / float(scH)) : 0.0f;
            const size_t dataBytes = 256ul * halfRank;
            const size_t pageBytes = dataBytes + 512ul;
            device uchar* pageBase = poolBytes + (size_t)pageID * pageBytes;
            ((device char*)(pageBase + (size_t)pageOff * halfRank))[col] = char(clamp(round(v * invSc), -127.0f, 127.0f));
            if (col == 0u) *((device half*)(pageBase + dataBytes + (size_t)pageOff * 2ul)) = scH;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            const uint grp = sg >> 2u;
            float minV = simd_min(v), maxV = simd_max(v);
            if (lane == 0u) { sgMinScratch[sg] = minV; sgScratch[sg] = maxV; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0u && lane < 8u) {
                float gMin = sgMinScratch[lane], gMax = sgScratch[lane];
                for (uint off = 2u; off >= 1u; off >>= 1u) {
                    gMin = min(gMin, simd_shuffle_xor(gMin, ushort(off)));
                    gMax = max(gMax, simd_shuffle_xor(gMax, ushort(off)));
                }
                if ((lane & 3u) == 0u) {
                    q4Sm[lane >> 2u] = half2(half(max((gMax - gMin) / 15.0f, 6.103515625e-5f)), half(gMin));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const half2 smH = q4Sm[grp];
            const half scH = smH.x, mnH = smH.y;
            const float invSc4 = (float(scH) > 0.0f) ? (1.0f / float(scH)) : 0.0f;
            shX[col] = half(uint(clamp(round((v - float(mnH)) * invSc4), 0.0f, 15.0f)));
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint groupsPerHalf = max(1u, halfRank >> 7u);
            const size_t dataBytes = 128ul * halfRank;
            const size_t pageBytes = dataBytes + 256ul * groupsPerHalf * 4ul;
            device uchar* pageBase = poolBytes + (size_t)pageID * pageBytes;
            device uchar* qRow = pageBase + (size_t)pageOff * (halfRank >> 1u);
            if (tid < (halfRank >> 1u)) {
                const uint q0 = uint(shX[2u * tid]), q1 = uint(shX[2u * tid + 1u]);
                qRow[tid] = uchar((q0 & 0x0Fu) | ((q1 & 0x0Fu) << 4u));
            }
            if (tid < groupsPerHalf) {
                *((device half2*)(pageBase + dataBytes) + (size_t)pageOff * groupsPerHalf + tid) = q4Sm[tid];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const half* kPeRaw = rowLat + p.qLoraRank + p.kvLoraRank;
    const uint halfRope = p.ropeDim >> 1u;
    const bool interleaveRope = (p.ropeFlags & 1u) != 0u;
    const bool hasMRoPE = (p.ropeFlags & 2u) != 0u;
    for (uint pairIdx = tid; pairIdx < halfRope; pairIdx += 256u) {
        const uint ax = hasMRoPE ? ropeAxisMap[pairIdx] : 0u;
        const float rotPos = hasMRoPE ? float(mropePos[(size_t)ax * p.T + t]) : float(pos);
        const float ang = rotPos * invFreq[pairIdx];
        const float c = cos(ang) * p.mscale, s = sin(ang) * p.mscale;
        device half* dstPe = kPeCache + (size_t)pos * p.ropeDim;
        if (interleaveRope) {
            const float even = float(kPeRaw[2u * pairIdx]), odd = float(kPeRaw[2u * pairIdx + 1u]);
            dstPe[2u * pairIdx] = half(even * c - odd * s);
            dstPe[2u * pairIdx + 1u] = half(odd * c + even * s);
        } else {
            const float x0 = float(kPeRaw[pairIdx]), x1 = float(kPeRaw[pairIdx + halfRope]);
            dstPe[pairIdx] = half(x0 * c - x1 * s);
            dstPe[pairIdx + halfRope] = half(x1 * c + x0 * s);
        }
    }

    if (p.idxDim > 0u) {
        device const half* kRaw = kPeRaw + p.ropeDim;
        const float xK = (tid < p.idxDim) ? float(kRaw[tid]) : 0.0f;
        const bool kNormIsRMS = (p.ropeFlags & 4u) != 0u;
        float meanK = 0.0f;
        if (!kNormIsRMS) {
            float sumK = simd_sum(xK);
            if (lane == 0u) sgScratch[sg] = sumK;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0u) {
                float s = (lane < 8u) ? sgScratch[lane] : 0.0f;
                s = simd_sum(s);
                if (lane == 0u) sgScratch[0] = s / float(p.idxDim);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            meanK = sgScratch[0];
        }

        const float diffK = (tid < p.idxDim) ? (xK - meanK) : 0.0f;
        float varK = simd_sum(diffK * diffK);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) sgScratch[sg] = varK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float s = (lane < 8u) ? sgScratch[lane] : 0.0f;
            s = simd_sum(s);
            if (lane == 0u) sgScratch[0] = rsqrt(s / float(p.idxDim) + p.kNormEps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float invStdK = sgScratch[0];

        if (tid < p.idxDim) {
            const float wK = qw_mla_prep_load_w(kNormW, tid, kNormWBF16);
            const float bK = kNormIsRMS ? 0.0f : float(kNormB[tid]);
            shX[tid] = half(diffK * invStdK * wK + bK);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < halfRope) {
            const float x0 = float(shX[tid]), x1 = float(shX[tid + halfRope]);
            const float ang = float(pos) * invFreq[tid];
            const float c = cos(ang) * p.mscale, s = sin(ang) * p.mscale;
            shX[tid] = half(x0 * c - x1 * s);
            shX[tid + halfRope] = half(x1 * c + x0 * s);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0u && p.idxDim == 128u) {
            float4 kv4 = float4(*(threadgroup const half4*)(shX + lane * 4u));
            if (p.useHadamard != 0u) kv4 = qw_mla_fwht128_ortho(kv4, lane);
            *(device half4*)(pooledKeys + (size_t)pos * 128u + lane * 4u) = half4(kv4);
        }
    }
}

// 2. In-place GPT-J/NeoX RoPE on q_rope + optional NeoX RoPE and H_128 on q_idx inside mlaQFull.
// Grid: (qHeads + idxHeads, T, 1), Threadgroup: (32, 1, 1).
kernel void qwg_mla_q_rope_idx_prep(
    device half*                  mlaQFull    [[buffer(0)]],
    device const float*           invFreq     [[buffer(1)]],
    constant MLAQRopeIdxPrepParams& p         [[buffer(2)]],
    device const int*             mropePos    [[buffer(3)]],
    device const uint*            ropeAxisMap [[buffer(4)]],
    uint2                         tg          [[threadgroup_position_in_grid]],
    uint                          lane        [[thread_index_in_simdgroup]]
) {
    const uint t = tg.y;
    if (t >= p.T) return;
    const uint pos = p.pos0 + t;
    const uint halfRope = p.ropeDim >> 1u;
    const bool interleaveRope = (p.ropeFlags & 1u) != 0u;
    const bool hasMRoPE = (p.ropeFlags & 2u) != 0u;
    device half* row = mlaQFull + (size_t)t * p.qFullStride;

    if (tg.x < p.qHeads) {
        device half* qr = row + p.qRopeOffset + tg.x * p.ropeDim;
        for (uint pairIdx = lane; pairIdx < halfRope; pairIdx += 32u) {
            const uint ax = hasMRoPE ? ropeAxisMap[pairIdx] : 0u;
            const float rotPos = hasMRoPE ? float(mropePos[(size_t)ax * p.T + t]) : float(pos);
            const float ang = rotPos * invFreq[pairIdx];
            const float c = cos(ang) * p.mscale, s = sin(ang) * p.mscale;
            if (interleaveRope) {
                const float even = float(qr[2u * pairIdx]), odd = float(qr[2u * pairIdx + 1u]);
                qr[2u * pairIdx] = half(even * c - odd * s);
                qr[2u * pairIdx + 1u] = half(odd * c + even * s);
            } else {
                const float x0 = float(qr[pairIdx]), x1 = float(qr[pairIdx + halfRope]);
                qr[pairIdx] = half(x0 * c - x1 * s);
                qr[pairIdx + halfRope] = half(x1 * c + x0 * s);
            }
        }
        return;
    }

    const uint idxH = tg.x - p.qHeads;
    if (idxH >= p.idxHeads || p.idxDim != 128u) return;
    device half* qi = row + p.qChannels + idxH * 128u;
    float4 v = float4(*(device const half4*)(qi + lane * 4u));
    if (p.ropeDim == 64u) {
        const float4 mate = simd_shuffle_xor(v, ushort(8));
        if (lane < 16u) {
            const uint baseFreq = (lane & 7u) * 4u;
            const float4 ang4 = float(pos) * float4(invFreq[baseFreq + 0u], invFreq[baseFreq + 1u],
                                                    invFreq[baseFreq + 2u], invFreq[baseFreq + 3u]);
            const float4 c4 = cos(ang4) * p.mscale, s4 = sin(ang4) * p.mscale;
            v = (lane < 8u) ? (v * c4 - mate * s4) : (v * c4 + mate * s4);
        }
    }
    if (p.useHadamard != 0u) v = qw_mla_fwht128_ortho(v, lane);
    *(device half4*)(qi + lane * 4u) = half4(v);
}
