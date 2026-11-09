// qwgdn_f32.metal — GatedDeltaNet prologue, causal depthwise conv, and single-token / sequence recurrence in f32.
// Shared across GDN-equipped models using f32 conv/recurrence buffers and sigmoid-gated output RMSNorm + H_32.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_geometry_constants.metalh"
#include "qw_dequant.metalh"
using namespace metal;

#define QW_GDN_F32_CHUNKS 8u
#define QW_GDN_F32_KC 16u

// Fused T <= 8 depthwise causal conv (K = 4) + SiLU + per-head q/k L2 norm + glog/beta in f32.
// Grid: 64 threadgroups x 128 threads. TGs 0..15 process key heads 0..15 (q and k); TGs 16..63 process value heads 0..47.
kernel void qwg_gdn_conv_prep_f32(device const float* x        [[buffer(0)]],   // [T][xStride]
                                  device const float* w        [[buffer(1)]],   // [10240][4]
                                  device float*       stA      [[buffer(2)]],   // [3][10240]
                                  device float*       stB      [[buffer(3)]],   // [3][10240]
                                  device const float* bIn      [[buffer(4)]],   // [T][baStride]
                                  device const float* aIn      [[buffer(5)]],   // [T][baStride]
                                  device const float* aLog     [[buffer(6)]],   // [48]
                                  device const float* dtBias   [[buffer(7)]],   // [48]
                                  device float*       y        [[buffer(8)]],   // [T][10240]
                                  device float*       glog     [[buffer(9)]],   // [T][48]
                                  device float*       beta     [[buffer(10)]],  // [T][48]
                                  constant uint&      tPacked  [[buffer(11)]],
                                  constant uint&      xStride  [[buffer(12)]],
                                  constant uint&      baStride [[buffer(13)]],
                                  device float*       stC      [[buffer(14)]],  // [3][10240] (or 8-slot ring base)
                                  device float*       stD      [[buffer(15)]],  // [3][10240]
                                  uint tg [[threadgroup_position_in_grid]],
                                  uint d  [[thread_position_in_threadgroup]]) {
    constexpr uint DK = 128u, KH = 16u, VH = 48u, CONV = 10240u, KEYDIM = 2048u;
    const uint T = tPacked & 0xFFu;
    const uint p0 = (tPacked >> 8u) & 7u;
    const bool hasRing = (tPacked & 0x8000u) != 0u;
    threadgroup float qPart[4], kPart[4];
    device float* stOuts[4] = { stB, stC, stD, stA };
    if (tg < KH) {
        const uint cq = tg * DK + d;
        const float wq0 = w[cq * 4u], wq1 = w[cq * 4u + 1u], wq2 = w[cq * 4u + 2u], wq3 = w[cq * 4u + 3u];
        float pq0 = stA[cq], pq1 = stA[CONV + cq], pq2 = stA[2u * CONV + cq];

        const uint ck = KEYDIM + cq;
        const float wk0 = w[ck * 4u], wk1 = w[ck * 4u + 1u], wk2 = w[ck * 4u + 2u], wk3 = w[ck * 4u + 3u];
        float pk0 = stA[ck], pk1 = stA[CONV + ck], pk2 = stA[2u * CONV + ck];
        const float qScale = 1.0f / sqrt(float(DK));

        for (uint t = 0u; t < T; ++t) {
            device float* stOut = hasRing ? (stC + (uint64_t)((p0 + t + 1u) & 7u) * (3u * CONV)) : stOuts[t & 3u];
            const float pq3 = x[(uint64_t)t * xStride + cq];
            const float accQ = wq0 * pq0 + wq1 * pq1 + wq2 * pq2 + wq3 * pq3;
            const float q = accQ / (1.0f + exp(-accQ));
            stOut[cq] = pq1; stOut[CONV + cq] = pq2; stOut[2u * CONV + cq] = pq3;
            pq0 = pq1; pq1 = pq2; pq2 = pq3;

            const float pk3 = x[(uint64_t)t * xStride + ck];
            const float accK = wk0 * pk0 + wk1 * pk1 + wk2 * pk2 + wk3 * pk3;
            const float k = accK / (1.0f + exp(-accK));
            stOut[ck] = pk1; stOut[CONV + ck] = pk2; stOut[2u * CONV + ck] = pk3;
            pk0 = pk1; pk1 = pk2; pk2 = pk3;

            const float qn = simd_sum(q * q), kn = simd_sum(k * k);
            if ((d & 31u) == 0u) { qPart[d >> 5u] = qn; kPart[d >> 5u] = kn; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float qTot = qPart[0] + qPart[1] + qPart[2] + qPart[3];
            const float kTot = kPart[0] + kPart[1] + kPart[2] + kPart[3];
            const float qInv = 1.0f / sqrt(qTot + 1e-6f), kInv = 1.0f / sqrt(kTot + 1e-6f);
            y[(uint64_t)t * CONV + cq] = q * qInv * qScale;
            y[(uint64_t)t * CONV + ck] = k * kInv;
            if (t + 1u < T) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        const uint hv = tg - KH;
        const uint cv = 2u * KEYDIM + hv * DK + d;
        const float wv0 = w[cv * 4u], wv1 = w[cv * 4u + 1u], wv2 = w[cv * 4u + 2u], wv3 = w[cv * 4u + 3u];
        float pv0 = stA[cv], pv1 = stA[CONV + cv], pv2 = stA[2u * CONV + cv];
        const float al = -exp(aLog[hv]), dt = dtBias[hv];
        for (uint t = 0u; t < T; ++t) {
            device float* stOut = hasRing ? (stC + (uint64_t)((p0 + t + 1u) & 7u) * (3u * CONV)) : stOuts[t & 3u];
            const float pv3 = x[(uint64_t)t * xStride + cv];
            const float accV = wv0 * pv0 + wv1 * pv1 + wv2 * pv2 + wv3 * pv3;
            y[(uint64_t)t * CONV + cv] = accV / (1.0f + exp(-accV));
            stOut[cv] = pv1; stOut[CONV + cv] = pv2; stOut[2u * CONV + cv] = pv3;
            pv0 = pv1; pv1 = pv2; pv2 = pv3;
            if (d == 0u) {
                const float xv = aIn[(uint64_t)t * baStride + hv] + dt;
                const float sp = (xv > 20.0f) ? xv : log(1.0f + exp(xv));
                glog[(uint64_t)t * VH + hv] = al * sp;
                beta[(uint64_t)t * VH + hv] = 1.0f / (1.0f + exp(-bIn[(uint64_t)t * baStride + hv]));
            }
        }
    }
}

kernel void qwg_gdn_recurrent_f32(
        device const float* y      [[buffer(0)]],   // this step: q̂‖k̂‖v (10240)
        device const float* betas  [[buffer(1)]],   // [48]
        device const float* glog   [[buffer(2)]],   // [48]
        device float*       state  [[buffer(3)]],   // [48][128][128] in place
        device const float* z      [[buffer(4)]],   // [48 * 128] f32 gate
        device const float* nw     [[buffer(5)]],   // [128] f32 norm weight
        device float*       out    [[buffer(6)]],   // [48 * 128] f32 gated norm + H_32 out
        uint3 tg                   [[threadgroup_position_in_grid]],
        uint  tid                  [[thread_index_in_threadgroup]]) {
    const uint h = tg.x;                 // v-head 0..47
    const uint c = tid & 127u;           // v-column (32 consecutive per SG)
    const uint kch = tid >> 7u;          // k-chunk 0..7
    const uint i = h / 3u;               // KEY head it shares
    const float beta = betas[h];
    const float eg = exp(glog[h]);
    device float* S = state + (uint64_t)h * 16384u;          // S[k][c] at k*128+c
    device const float* qrow = y + (uint64_t)i * 128u;
    device const float* krow = y + 2048u + (uint64_t)i * 128u;
    threadgroup float2 red[1024];
    threadgroup float rho_shared;
    threadgroup float redNorm[4];
    const uint k0 = kch * QW_GDN_F32_KC;

    if (tid < 32u) {
        device const float4* k4 = (device const float4*)krow;
        device const float4* q4 = (device const float4*)qrow;
        float4 kh = k4[tid];
        float4 qh = q4[tid];
        float partial = float(kh.x) * float(qh.x) + float(kh.y) * float(qh.y)
                      + float(kh.z) * float(qh.z) + float(kh.w) * float(qh.w);
        float rho_sum = simd_sum(partial);
        if (tid == 0u) {
            rho_shared = rho_sum;
        }
    }

    float sreg[QW_GDN_F32_KC];
    float kvp = 0.0f;
    float up  = 0.0f;
    #pragma unroll
    for (uint j = 0u; j < QW_GDN_F32_KC; ++j) {
        const uint k = k0 + j;
        const float s = S[k * 128u + c] * eg;
        sreg[j] = s;
        kvp = fma(s, float(krow[k]), kvp);
        up  = fma(s, float(qrow[k]), up);
    }
    red[tid] = float2(kvp, up);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float2 kv_u = float2(0.0f, 0.0f);
    for (uint q = 0u; q < QW_GDN_F32_CHUNKS; q++) {
        kv_u += red[q * 128u + c];
    }
    const float kv = kv_u.x;
    const float u  = kv_u.y;
    const float delta = (float(y[4096u + (uint64_t)h * 128u + c]) - kv) * beta;

    const float o = u + rho_shared * delta;
    if (kch == 0u) {
        const float sq = simd_sum(o * o);
        if ((tid & 31u) == 0u) {
            redNorm[tid >> 5u] = sq;
        }
    }

    #pragma unroll
    for (uint j = 0u; j < QW_GDN_F32_KC; ++j) {
        const uint k = k0 + j;
        S[k * 128u + c] = sreg[j] + float(krow[k]) * delta;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kch == 0u) {
        const uint idx = h * 128u + c;
        const float total = redNorm[0] + redNorm[1] + redNorm[2] + redNorm[3];
        const float rms = rsqrt(total * (1.0f / 128.0f) + QW_RMS_EPS);
        const float gate = 1.0f / (1.0f + exp(-z[idx]));
        float x = o * rms * nw[c] * gate;
        const uint lane = c & 31u;
        x = qwg_fwht32_lane_f32(x, lane);
        out[idx] = x;
    }
}

// Fused batched depthwise causal conv (K = 4) + SiLU + per-head q/k L2 norm + glog/beta in f32.
// Grid: (64, T, 1) threadgroups x 128 threads. TGs 0..15 process key heads 0..15; TGs 16..63 process value heads 0..47.
kernel void qwg_gdn_conv_prep_seq_f32(device const float* x       [[buffer(0)]],   // [T][xStride]
                                      device const float* w       [[buffer(1)]],   // [10240][4]
                                      device const float* stIn    [[buffer(2)]],   // [3][10240]
                                      device float*       stOut   [[buffer(3)]],   // [3][10240]
                                      device const float* bIn     [[buffer(4)]],   // [T][48]
                                      device const float* aIn     [[buffer(5)]],   // [T][48]
                                      device const float* aLog    [[buffer(6)]],   // [48]
                                      device const float* dtBias  [[buffer(7)]],   // [48]
                                      device float*       y       [[buffer(8)]],   // [T][10240]
                                      device float*       glog    [[buffer(9)]],   // [T][48]
                                      device float*       beta    [[buffer(10)]],  // [T][48]
                                      constant uint&      T       [[buffer(11)]],
                                      constant uint&      xStride [[buffer(12)]],
                                      uint2 tg [[threadgroup_position_in_grid]],
                                      uint  d  [[thread_index_in_threadgroup]]) {
    constexpr uint DK = 128u, KH = 16u, VH = 48u, CONV = 10240u, KEYDIM = 2048u;
    threadgroup float qPart[4], kPart[4];
    const uint hIdx = tg.x, t = tg.y;
    device float* yT = y + (uint64_t)t * CONV;
    if (hIdx < KH) {
        const uint cq = hIdx * DK + d;
        const float wq0 = w[cq * 4u], wq1 = w[cq * 4u + 1u], wq2 = w[cq * 4u + 2u], wq3 = w[cq * 4u + 3u];
        const float pq0 = (t >= 3u) ? x[(uint64_t)(t - 3u) * xStride + cq] : stIn[t * CONV + cq];
        const float pq1 = (t >= 2u) ? x[(uint64_t)(t - 2u) * xStride + cq] : stIn[(t + 1u) * CONV + cq];
        const float pq2 = (t >= 1u) ? x[(uint64_t)(t - 1u) * xStride + cq] : stIn[(t + 2u) * CONV + cq];
        const float pq3 = x[(uint64_t)t * xStride + cq];
        const float accQ = wq0 * pq0 + wq1 * pq1 + wq2 * pq2 + wq3 * pq3;
        const float q = accQ / (1.0f + exp(-accQ));

        const uint ck = KEYDIM + cq;
        const float wk0 = w[ck * 4u], wk1 = w[ck * 4u + 1u], wk2 = w[ck * 4u + 2u], wk3 = w[ck * 4u + 3u];
        const float pk0 = (t >= 3u) ? x[(uint64_t)(t - 3u) * xStride + ck] : stIn[t * CONV + ck];
        const float pk1 = (t >= 2u) ? x[(uint64_t)(t - 2u) * xStride + ck] : stIn[(t + 1u) * CONV + ck];
        const float pk2 = (t >= 1u) ? x[(uint64_t)(t - 1u) * xStride + ck] : stIn[(t + 2u) * CONV + ck];
        const float pk3 = x[(uint64_t)t * xStride + ck];
        const float accK = wk0 * pk0 + wk1 * pk1 + wk2 * pk2 + wk3 * pk3;
        const float k = accK / (1.0f + exp(-accK));

        if (t + 1u == T) {
            stOut[cq] = pq1; stOut[CONV + cq] = pq2; stOut[2u * CONV + cq] = pq3;
            stOut[ck] = pk1; stOut[CONV + ck] = pk2; stOut[2u * CONV + ck] = pk3;
        }
        const float qn = simd_sum(q * q), kn = simd_sum(k * k);
        if ((d & 31u) == 0u) { qPart[d >> 5u] = qn; kPart[d >> 5u] = kn; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float qTot = qPart[0] + qPart[1] + qPart[2] + qPart[3];
        const float kTot = kPart[0] + kPart[1] + kPart[2] + kPart[3];
        const float qInv = 1.0f / sqrt(qTot + 1e-6f), kInv = 1.0f / sqrt(kTot + 1e-6f);
        const float qScale = 1.0f / sqrt(float(DK));
        yT[cq] = q * qInv * qScale;
        yT[ck] = k * kInv;
    } else {
        const uint hv = hIdx - KH;
        const uint cv = 2u * KEYDIM + hv * DK + d;
        const float wv0 = w[cv * 4u], wv1 = w[cv * 4u + 1u], wv2 = w[cv * 4u + 2u], wv3 = w[cv * 4u + 3u];
        const float pv0 = (t >= 3u) ? x[(uint64_t)(t - 3u) * xStride + cv] : stIn[t * CONV + cv];
        const float pv1 = (t >= 2u) ? x[(uint64_t)(t - 2u) * xStride + cv] : stIn[(t + 1u) * CONV + cv];
        const float pv2 = (t >= 1u) ? x[(uint64_t)(t - 1u) * xStride + cv] : stIn[(t + 2u) * CONV + cv];
        const float pv3 = x[(uint64_t)t * xStride + cv];
        const float accV = wv0 * pv0 + wv1 * pv1 + wv2 * pv2 + wv3 * pv3;
        yT[cv] = accV / (1.0f + exp(-accV));
        if (t + 1u == T) {
            stOut[cv] = pv1; stOut[CONV + cv] = pv2; stOut[2u * CONV + cv] = pv3;
        }
        if (d == 0u) {
            const float xv = aIn[t * VH + hv] + dtBias[hv];
            const float sp = (xv > 20.0f) ? xv : log(1.0f + exp(xv));
            glog[t * VH + hv] = -exp(aLog[hv]) * sp;
            beta[t * VH + hv] = 1.0f / (1.0f + exp(-bIn[t * VH + hv]));
        }
    }
}

// Register-resident multi-token GDN recurrence in f32: 4 column-blocks per v-head (192 TGs x 128 threads), 4-lane
// simd_shuffle reduction. BLK tokens of k/q/v/beta/eg are staged in threadgroup memory per barrier pair so the
// serial token chain reads threadgroup memory instead of device memory. Writes raw o to out.
template <uint BLK>
static inline void qwg_gdn_recurrent_seq_tg_impl(device const float* y, device const float* betas,
                                                 device const float* glog, device float* state, device float* out,
                                                 uint T, uint CONV, uint VHEADS, uint2 tg, uint sgi, uint lane,
                                                 threadgroup float* kS, threadgroup float* qS,
                                                 threadgroup float* vS, threadgroup float* bS, threadgroup float* eS) {
    const uint cBlock = tg.x;
    const uint h = tg.y;
    const uint kch = lane & 3u;
    const uint cl = (sgi << 3u) + (lane >> 2u);
    const uint c = cBlock * 32u + cl;
    const uint i = h / 3u;
    const uint k0 = kch * 32u;
    const uint VD = 128u;
    const uint VSTRIDE = VHEADS * VD;
    const uint tid = sgi * 32u + lane;
    device float* S = state + (uint64_t)h * 16384u;

    float sreg[32];
    #pragma unroll
    for (uint j = 0u; j < 32u; ++j) sreg[j] = S[(k0 + j) * VD + c];

    for (uint tb = 0u; tb < T; tb += BLK) {
        const uint nb = min(BLK, T - tb);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint b = 0u; b < nb; ++b) {
            device const float* yT = y + (uint64_t)(tb + b) * CONV;
            kS[b * 128u + tid] = yT[2048u + i * VD + tid];
            qS[b * 128u + tid] = yT[i * VD + tid];
            if (tid < 32u) vS[b * 32u + tid] = yT[4096u + (uint64_t)h * VD + cBlock * 32u + tid];
        }
        if (tid < nb) {
            bS[tid] = betas[(uint64_t)(tb + tid) * VHEADS + h];
            eS[tid] = exp(glog[(uint64_t)(tb + tid) * VHEADS + h]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint b = 0u; b < nb; ++b) {
            threadgroup const float* krow = kS + b * 128u;
            threadgroup const float* qrow = qS + b * 128u;
            const float beta = bS[b];
            const float eg = eS[b];
            const float4 kh = ((threadgroup const float4*)krow)[lane];
            const float4 qh = ((threadgroup const float4*)qrow)[lane];
            const float partial = float(kh.x) * float(qh.x) + float(kh.y) * float(qh.y)
                                + float(kh.z) * float(qh.z) + float(kh.w) * float(qh.w);
            const float rho = simd_sum(partial);

            float kvp = 0.0f;
            float up = 0.0f;
            #pragma unroll
            for (uint j = 0u; j < 32u; ++j) {
                const uint k = k0 + j;
                const float s = sreg[j] * eg;
                sreg[j] = s;
                kvp = fma(s, krow[k], kvp);
                up = fma(s, qrow[k], up);
            }
            float2 kv_u = float2(kvp, up);
            kv_u += simd_shuffle_xor(kv_u, 1u);
            kv_u += simd_shuffle_xor(kv_u, 2u);
            const float delta = (vS[b * 32u + cl] - kv_u.x) * beta;

            #pragma unroll
            for (uint j = 0u; j < 32u; ++j) {
                const uint k = k0 + j;
                sreg[j] += krow[k] * delta;
            }

            if (kch == 0u) {
                out[(uint64_t)(tb + b) * VSTRIDE + (uint64_t)h * VD + c] = kv_u.y + rho * delta;
            }
        }
    }
    #pragma unroll
    for (uint j = 0u; j < 32u; ++j) S[(k0 + j) * VD + c] = sreg[j];
}

kernel void qwg_gdn_recurrent_seq_f32(
        device const float* y      [[buffer(0)]],   // [T][CONV] conv output
        device const float* betas  [[buffer(1)]],   // [T][vHeads]
        device const float* glog   [[buffer(2)]],   // [T][vHeads]
        device float*       state  [[buffer(3)]],   // [vHeads][128][128], in place
        device float*       out    [[buffer(4)]],   // [T][vHeads*128] raw o
        constant uint&      T      [[buffer(5)]],
        constant uint&      CONV   [[buffer(6)]],
        constant uint&      VHEADS [[buffer(7)]],
        uint2 tg   [[threadgroup_position_in_grid]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  lane [[thread_index_in_simdgroup]]) {
    constexpr uint BLK = 8u;
    threadgroup float kS[BLK * 128u], qS[BLK * 128u], vS[BLK * 32u], bS[BLK], eS[BLK];
    qwg_gdn_recurrent_seq_tg_impl<BLK>(y, betas, glog, state, out, T, CONV, VHEADS, tg, sgi, lane, kS, qS, vS, bS, eS);
}

// Per-head in-place RMSNorm + sigmoid(z) gate + 32-point WHT (H_32) over [T, VHEADS * 128] f32 output.
kernel void qwg_gdn_norm_sigmoid_f32(
        device float*       out      [[buffer(0)]],   // [T][vHeads*128], in-place
        device const float* z        [[buffer(1)]],   // [T][ZSTRIDE]
        device const float* nw       [[buffer(2)]],   // [128]
        constant uint&      VHEADS   [[buffer(3)]],
        constant uint&      ZSTRIDE  [[buffer(4)]],
        device half*        out16    [[buffer(5)]],
        constant uint&      hasOut16 [[buffer(6)]],
        uint2 tg  [[threadgroup_position_in_grid]],
        uint  tid [[thread_index_in_threadgroup]]) {
    const uint h = tg.x, t = tg.y, c = tid;
    const uint VD = 128u;
    const uint VSTRIDE = VHEADS * VD;
    const uint64_t idx = (uint64_t)t * VSTRIDE + (uint64_t)h * VD + c;
    const float o = out[idx];
    threadgroup float redNorm[4];
    const float sq = simd_sum(o * o);
    if ((tid & 31u) == 0u) redNorm[tid >> 5u] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = redNorm[0] + redNorm[1] + redNorm[2] + redNorm[3];
    const float rms = rsqrt(total * (1.0f / 128.0f) + QW_RMS_EPS);
    const float zn = z[(uint64_t)t * ZSTRIDE + (uint64_t)h * VD + c];
    float x = o * rms * float(nw[c]) * (1.0f / (1.0f + exp(-zn)));
    const uint lane = tid & 31u;
    x = qwg_fwht32_lane_f32(x, lane);
    const float rotVal = x;
    if (hasOut16 != 0u) {
        out16[idx] = half(rotVal);
    } else {
        out[idx] = rotVal;
    }
}

// Verify rows: R tokens through the T = 1 recurrence (qwg_gdn_recurrent_f32) in ONE dispatch. The per-row
// arithmetic and reduction order are the T = 1 kernel's; the state is carried in registers between rows, so
// each row reads exactly the value the T = 1 step would reload from memory (a float store/reload is exact).
kernel void qwg_gdn_recurrent_rows_f32(
        device const float* y         [[buffer(0)]],   // [R][10240] conv output per row
        device const float* betas     [[buffer(1)]],   // [R][48]
        device const float* glog      [[buffer(2)]],   // [R][48]
        device float*       state     [[buffer(3)]],   // [48][128][128] initial state (and post-row-3 slot)
        device const float* z         [[buffer(4)]],   // row t at z + t * zStride
        device const float* nw        [[buffer(5)]],   // [128]
        device float*       out       [[buffer(6)]],   // [R][48 * 128]
        constant uint&      rPacked   [[buffer(7)]],
        constant uint&      zStride   [[buffer(8)]],
        device float*       stateRow0 [[buffer(9)]],   // [48][128][128] post-row-0 slot
        device float*       stateRow1 [[buffer(10)]],  // [48][128][128] post-row-1 slot (or 8-slot ring base)
        device float*       stateRow2 [[buffer(11)]],  // [48][128][128] post-row-2 slot
        uint3 tg                      [[threadgroup_position_in_grid]],
        uint  tid                     [[thread_index_in_threadgroup]]) {
    const uint R = rPacked & 0xFFu;
    const uint sp0 = (rPacked >> 8u) & 7u;
    const bool hasRing = (rPacked & 0x8000u) != 0u;
    const uint h = tg.x;                 // v-head 0..47
    const uint c = tid & 127u;           // v-column
    const uint kch = tid >> 7u;          // k-chunk 0..7
    const uint i = h / 3u;               // KEY head it shares
    const uint k0 = kch * QW_GDN_F32_KC;
    const uint64_t hOff = (uint64_t)h * 16384u;
    device float* S0 = state + hOff;
    device float* S1 = stateRow0 + hOff;
    device float* S2 = stateRow1 + hOff;
    device float* S3 = stateRow2 + hOff;
    device float* SOuts[4] = { S1, S2, S3, S0 };

    float sreg[QW_GDN_F32_KC];
    #pragma unroll
    for (uint j = 0u; j < QW_GDN_F32_KC; ++j) sreg[j] = S0[(k0 + j) * 128u + c];

    threadgroup float2 red[1024];
    threadgroup float rho_shared;
    threadgroup float redNorm[4];

    for (uint t = 0u; t < R; ++t) {
        device const float* yT = y + (uint64_t)t * 10240u;
        device const float* qrow = yT + (uint64_t)i * 128u;
        device const float* krow = yT + 2048u + (uint64_t)i * 128u;
        const float beta = betas[t * 48u + h];
        const float eg = exp(glog[t * 48u + h]);

        if (tid < 32u) {
            device const float4* k4 = (device const float4*)krow;
            device const float4* q4 = (device const float4*)qrow;
            float4 kh = k4[tid];
            float4 qh = q4[tid];
            float partial = float(kh.x) * float(qh.x) + float(kh.y) * float(qh.y)
                          + float(kh.z) * float(qh.z) + float(kh.w) * float(qh.w);
            float rho_sum = simd_sum(partial);
            if (tid == 0u) {
                rho_shared = rho_sum;
            }
        }

        float kvp = 0.0f;
        float up  = 0.0f;
        #pragma unroll
        for (uint j = 0u; j < QW_GDN_F32_KC; ++j) {
            const uint k = k0 + j;
            const float s = sreg[j] * eg;
            sreg[j] = s;
            kvp = fma(s, float(krow[k]), kvp);
            up  = fma(s, float(qrow[k]), up);
        }
        red[tid] = float2(kvp, up);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float2 kv_u = float2(0.0f, 0.0f);
        for (uint q = 0u; q < QW_GDN_F32_CHUNKS; q++) {
            kv_u += red[q * 128u + c];
        }
        const float kv = kv_u.x;
        const float u  = kv_u.y;
        const float delta = (float(yT[4096u + (uint64_t)h * 128u + c]) - kv) * beta;

        const float o = u + rho_shared * delta;
        if (kch == 0u) {
            const float sq = simd_sum(o * o);
            if ((tid & 31u) == 0u) {
                redNorm[tid >> 5u] = sq;
            }
        }

        #pragma unroll
        for (uint j = 0u; j < QW_GDN_F32_KC; ++j) {
            const uint k = k0 + j;
            sreg[j] = sreg[j] + float(krow[k]) * delta;
        }
        device float* SOut = hasRing
            ? (stateRow1 + (uint64_t)((sp0 + t + 1u) & 7u) * (48u * 16384u) + hOff)
            : SOuts[t & 3u];
        #pragma unroll
        for (uint j = 0u; j < QW_GDN_F32_KC; ++j) SOut[(k0 + j) * 128u + c] = sreg[j];

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kch == 0u) {
            const uint idx = h * 128u + c;
            const float total = redNorm[0] + redNorm[1] + redNorm[2] + redNorm[3];
            const float rms = rsqrt(total * (1.0f / 128.0f) + QW_RMS_EPS);
            const float gate = 1.0f / (1.0f + exp(-z[t * zStride + idx]));
            float x = o * rms * nw[c] * gate;
            const uint lane = c & 31u;
            x = qwg_fwht32_lane_f32(x, lane);
            out[t * 6144u + idx] = x;
        }
        if (t + 1u < R) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
