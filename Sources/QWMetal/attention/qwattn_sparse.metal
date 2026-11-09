// qwattn_sparse.metal — Sparse gated attention core with online softmax over QSA selection and causal tail.
// Supports KV cache modes: f32 (0), f16 (1), q8 (2), q4 (3); Q, gate, and out remain f32.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_weight_basis.metalh"
using namespace metal;

template <uint KV_MODE>
inline float qwg_fn_kv_read(device const uchar* buf, device const half* meta, uint seg, uint d) {
    constexpr uint D = 256u;
    if constexpr (KV_MODE == 0u) {
        return ((device const float*)buf)[(uint64_t)seg * D + d];
    } else if constexpr (KV_MODE == 1u) {
        return float(((device const half*)buf)[(uint64_t)seg * D + d]);
    } else if constexpr (KV_MODE == 2u) {
        const float sc = float(meta[seg]);
        const int8_t q = ((device const char*)buf)[(uint64_t)seg * D + d];
        return float(q) * sc;
    } else {
        const uint mIdx = (seg * 2u + (d >> 7u)) * 2u;
        const float sc = float(meta[mIdx]);
        const float mn = float(meta[mIdx + 1u]);
        const uchar byteVal = buf[(uint64_t)seg * (D / 2u) + (d >> 1u)];
        const uint nib = ((d & 1u) == 0u) ? (byteVal & 0x0Fu) : (byteVal >> 4u);
        return float(nib) * sc + mn;
    }
}

template <uint KV_MODE>
inline void qwg_fn_kv_read_row8(device const uchar* buf, device const half* meta, uint seg, uint lane, thread float (&kvOut)[8]) {
    constexpr uint D = 256u;
    if constexpr (KV_MODE == 0u) {
        device const float* row = ((device const float*)buf) + (uint64_t)seg * D + lane;
        #pragma unroll
        for (uint c = 0u; c < 8u; c++) kvOut[c] = row[c * 32u];
    } else if constexpr (KV_MODE == 1u) {
        device const half* row = ((device const half*)buf) + (uint64_t)seg * D + lane;
        #pragma unroll
        for (uint c = 0u; c < 8u; c++) kvOut[c] = float(row[c * 32u]);
    } else if constexpr (KV_MODE == 2u) {
        const float sc = float(meta[seg]);
        device const char* row = ((device const char*)buf) + (uint64_t)seg * D + lane;
        #pragma unroll
        for (uint c = 0u; c < 8u; c++) kvOut[c] = float(row[c * 32u]) * sc;
    } else {
        const half4 m = ((device const half4*)meta)[seg];
        const float sc0 = float(m.x), mn0 = float(m.y);
        const float sc1 = float(m.z), mn1 = float(m.w);
        device const uchar* row = buf + (uint64_t)seg * (D / 2u) + (lane >> 1u);
        const uint shift = (lane & 1u) << 2u;
        #pragma unroll
        for (uint c = 0u; c < 4u; c++) {
            const uint nib = (uint(row[c * 16u]) >> shift) & 0x0Fu;
            kvOut[c] = float(nib) * sc0 + mn0;
        }
        #pragma unroll
        for (uint c = 4u; c < 8u; c++) {
            const uint nib = (uint(row[c * 16u]) >> shift) & 0x0Fu;
            kvOut[c] = float(nib) * sc1 + mn1;
        }
    }
}

template <uint KV_MODE>
inline float4 qwg_fn_kv_read_vec4(device const uchar* buf, device const half* meta, uint seg, uint vHalf, uint lane) {
    constexpr uint D = 256u;
    const uint dBase = vHalf * 128u + lane * 4u;
    if constexpr (KV_MODE == 0u) {
        return *((device const float4*)(((device const float*)buf) + (uint64_t)seg * D + dBase));
    } else if constexpr (KV_MODE == 1u) {
        const half4 h = *((device const half4*)(((device const half*)buf) + (uint64_t)seg * D + dBase));
        return float4(h);
    } else if constexpr (KV_MODE == 2u) {
        const char4 c = *((device const char4*)(((device const char*)buf) + (uint64_t)seg * D + dBase));
        const float sc = float(meta[seg]);
        return float4(c) * sc;
    } else {
        const half2 m = ((device const half2*)meta)[seg * 2u + vHalf];
        const float sc = float(m.x), mn = float(m.y);
        const uchar2 b2 = *((device const uchar2*)(buf + (uint64_t)seg * (D / 2u) + (dBase >> 1u)));
        return float4(float(b2.x & 0x0Fu), float(b2.x >> 4u),
                      float(b2.y & 0x0Fu), float(b2.y >> 4u)) * sc + mn;
    }
}

template <uint KV_MODE>
inline void qwg_attn_sparse_gated_impl(device const float* q,
                                       device const uchar* k,
                                       device const uchar* v,
                                       device const float* gate,
                                       device const int*   list,
                                       device float*       out,
                                       uint T, uint nh, uint nkv, uint pos0,
                                       device const half*  kMeta,
                                       device const half*  vMeta,
                                       device half*        out16,
                                       uint                hasOut16,
                                       threadgroup float*  score,
                                       threadgroup uint*   blkIdx,
                                       threadgroup float*  sgScratch,
                                       uint tg, uint d) {
    constexpr uint D = 256u;
    constexpr uint BLOCK = 4u;
    constexpr uint TOPK = 512u;
    const uint t = tg / nh;
    const uint h = tg - t * nh;
    const uint g = h / (nh / nkv);
    const float scale = rsqrt((float)D);
    const uint pq = pos0 + t;
    const uint n = (pq + 1u) / BLOCK;
    const uint cnt = min(TOPK, n);
    const uint tailStart = n * BLOCK;
    const uint slots = cnt * BLOCK + (pq + 1u - tailStart);
    const uint sg = d >> 5u, lane = d & 31u;
    if (n <= TOPK) {
        if (d < cnt) blkIdx[d] = d * BLOCK;
        if (d + 256u < cnt) blkIdx[d + 256u] = (d + 256u) * BLOCK;
    } else {
        if (d < cnt) blkIdx[d] = uint(list[t * TOPK + d]) * BLOCK;
        if (d + 256u < cnt) blkIdx[d + 256u] = uint(list[t * TOPK + d + 256u]) * BLOCK;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float* qRow = q + (t * nh + h) * D;
    float qReg[8];
    #pragma unroll
    for (uint c = 0u; c < 8u; c++) qReg[c] = qRow[c * 32u + lane];
    for (uint slot = sg; slot < slots; slot += 8u) {
        const uint j = (slot < cnt * BLOCK)
            ? blkIdx[slot >> 2u] + (slot & 3u)
            : tailStart + (slot - cnt * BLOCK);
        const uint seg = j * nkv + g;
        float kv[8];
        qwg_fn_kv_read_row8<KV_MODE>(k, kMeta, seg, lane, kv);
        float s0 = 0.0f;
        #pragma unroll
        for (uint c = 0u; c < 8u; c++) {
            s0 = fma(qReg[c], kv[c], s0);
        }
        const float dot = simd_sum(s0) * scale;
        if (lane == 0u) {
            score[slot] = dot;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localMax = -INFINITY;
    for (uint s = d; s < slots; s += 256u) {
        localMax = max(localMax, score[s]);
    }
    localMax = simd_max(localMax);
    if (lane == 0u) sgScratch[sg] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float vMax = (lane < 8u) ? sgScratch[lane] : -INFINITY;
        vMax = simd_max(vMax);
        if (lane == 0u) sgScratch[0] = vMax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float m = sgScratch[0];

    float localSum = 0.0f;
    for (uint s = d; s < slots; s += 256u) {
        const float w = exp(score[s] - m);
        score[s] = w;
        localSum += w;
    }
    localSum = simd_sum(localSum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) sgScratch[sg] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float vSum = (lane < 8u) ? sgScratch[lane] : 0.0f;
        vSum = simd_sum(vSum);
        if (lane == 0u) sgScratch[0] = (vSum > 0.0f) ? (1.0f / vSum) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invL = sgScratch[0];

    float acc = 0.0f;
    for (uint b = 0u; b < cnt; b++) {
        const uint baseSeg = blkIdx[b] * nkv + g;
        const uint s0 = b << 2u;
        const float v0 = qwg_fn_kv_read<KV_MODE>(v, vMeta, baseSeg, d);
        const float v1 = qwg_fn_kv_read<KV_MODE>(v, vMeta, baseSeg + nkv, d);
        const float v2 = qwg_fn_kv_read<KV_MODE>(v, vMeta, baseSeg + 2u * nkv, d);
        const float v3 = qwg_fn_kv_read<KV_MODE>(v, vMeta, baseSeg + 3u * nkv, d);
        acc = fma(score[s0],      v0, acc);
        acc = fma(score[s0 + 1u], v1, acc);
        acc = fma(score[s0 + 2u], v2, acc);
        acc = fma(score[s0 + 3u], v3, acc);
    }
    for (uint slot = cnt * BLOCK; slot < slots; slot++) {
        const uint j = tailStart + (slot - cnt * BLOCK);
        acc = fma(score[slot], qwg_fn_kv_read<KV_MODE>(v, vMeta, j * nkv + g, d), acc);
    }
    const float o = acc * invL;
    const uint outIdx = (t * nh + h) * D + d;
    float x = o / (1.0f + exp(-gate[outIdx]));
    const uint lane2 = d & 31u;
    x = qwg_fwht32_lane_f32(x, lane2);
    const float rotVal = x;
    if (hasOut16 != 0u) {
        out16[outIdx] = half(rotVal);
    } else {
        out[outIdx] = rotVal;
    }
}

template <uint KV_MODE>
inline void qwg_attn_sparse_gated_gqa3_impl(device const float* q,
                                            device const uchar* k,
                                            device const uchar* v,
                                            device const float* gate,
                                            device const int*   list,
                                            device float*       out,
                                            uint T, uint nh, uint nkv, uint pos0,
                                            device const half*  kMeta,
                                            device const half*  vMeta,
                                            device half*        out16,
                                            uint                hasOut16,
                                            threadgroup float*  score,
                                            threadgroup uint*   blkIdx,
                                            threadgroup float*  sgScratch,
                                            uint tg, uint d) {
    constexpr uint D = 256u;
    constexpr uint BLOCK = 4u;
    constexpr uint TOPK = 512u;
    constexpr uint STRIDE = 2052u;
    const uint nGroups = nh / 3u;
    const uint t = tg / nGroups;
    const uint grp = tg - t * nGroups;
    const uint h0 = grp * 3u;
    const uint g = h0 / (nh / nkv);
    const float scale = rsqrt((float)D);
    const uint pq = pos0 + t;
    const uint n = (pq + 1u) / BLOCK;
    const uint cnt = min(TOPK, n);
    const uint tailStart = n * BLOCK;
    const uint slots = cnt * BLOCK + (pq + 1u - tailStart);
    const uint sg = d >> 5u, lane = d & 31u;
    if (n <= TOPK) {
        if (d < cnt) blkIdx[d] = d * BLOCK;
        if (d + 256u < cnt) blkIdx[d + 256u] = (d + 256u) * BLOCK;
    } else {
        if (d < cnt) blkIdx[d] = uint(list[t * TOPK + d]) * BLOCK;
        if (d + 256u < cnt) blkIdx[d + 256u] = uint(list[t * TOPK + d + 256u]) * BLOCK;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float* qRow0 = q + (t * nh + h0) * D;
    device const float* qRow1 = qRow0 + D;
    device const float* qRow2 = qRow0 + 2u * D;
    float qReg0[8], qReg1[8], qReg2[8];
    #pragma unroll
    for (uint c = 0u; c < 8u; c++) {
        const uint col = c * 32u + lane;
        qReg0[c] = qRow0[col];
        qReg1[c] = qRow1[col];
        qReg2[c] = qRow2[col];
    }
    for (uint slot = sg; slot < slots; slot += 8u) {
        const uint j = (slot < cnt * BLOCK)
            ? blkIdx[slot >> 2u] + (slot & 3u)
            : tailStart + (slot - cnt * BLOCK);
        const uint seg = j * nkv + g;
        float kv[8];
        qwg_fn_kv_read_row8<KV_MODE>(k, kMeta, seg, lane, kv);
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
        #pragma unroll
        for (uint c = 0u; c < 8u; c++) {
            s0 = fma(qReg0[c], kv[c], s0);
            s1 = fma(qReg1[c], kv[c], s1);
            s2 = fma(qReg2[c], kv[c], s2);
        }
        const float dot0 = simd_sum(s0) * scale;
        const float dot1 = simd_sum(s1) * scale;
        const float dot2 = simd_sum(s2) * scale;
        if (lane == 0u) {
            score[slot]              = dot0;
            score[STRIDE + slot]     = dot1;
            score[2u * STRIDE + slot] = dot2;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float max0 = -INFINITY, max1 = -INFINITY, max2 = -INFINITY;
    for (uint s = d; s < slots; s += 256u) {
        max0 = max(max0, score[s]);
        max1 = max(max1, score[STRIDE + s]);
        max2 = max(max2, score[2u * STRIDE + s]);
    }
    max0 = simd_max(max0);
    max1 = simd_max(max1);
    max2 = simd_max(max2);
    if (lane == 0u) {
        sgScratch[sg]       = max0;
        sgScratch[8u + sg]  = max1;
        sgScratch[16u + sg] = max2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float v0 = (lane < 8u) ? sgScratch[lane]       : -INFINITY;
        float v1 = (lane < 8u) ? sgScratch[8u + lane]  : -INFINITY;
        float v2 = (lane < 8u) ? sgScratch[16u + lane] : -INFINITY;
        v0 = simd_max(v0);
        v1 = simd_max(v1);
        v2 = simd_max(v2);
        if (lane == 0u) {
            sgScratch[0] = v0;
            sgScratch[1] = v1;
            sgScratch[2] = v2;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float m0 = sgScratch[0], m1 = sgScratch[1], m2 = sgScratch[2];

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
    for (uint s = d; s < slots; s += 256u) {
        const float w0 = exp(score[s] - m0);
        const float w1 = exp(score[STRIDE + s] - m1);
        const float w2 = exp(score[2u * STRIDE + s] - m2);
        score[s]               = w0;
        score[STRIDE + s]      = w1;
        score[2u * STRIDE + s] = w2;
        sum0 += w0;
        sum1 += w1;
        sum2 += w2;
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        sgScratch[sg]       = sum0;
        sgScratch[8u + sg]  = sum1;
        sgScratch[16u + sg] = sum2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float v0 = (lane < 8u) ? sgScratch[lane]       : 0.0f;
        float v1 = (lane < 8u) ? sgScratch[8u + lane]  : 0.0f;
        float v2 = (lane < 8u) ? sgScratch[16u + lane] : 0.0f;
        v0 = simd_sum(v0);
        v1 = simd_sum(v1);
        v2 = simd_sum(v2);
        if (lane == 0u) {
            sgScratch[0] = (v0 > 0.0f) ? (1.0f / v0) : 0.0f;
            sgScratch[1] = (v1 > 0.0f) ? (1.0f / v1) : 0.0f;
            sgScratch[2] = (v2 > 0.0f) ? (1.0f / v2) : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invL0 = sgScratch[0], invL1 = sgScratch[1], invL2 = sgScratch[2];

    const uint vHalf = sg & 1u;
    const uint vGrp  = sg >> 1u;
    const uint dBase = vHalf * 128u + lane * 4u;
    float4 vAcc0 = 0.0f, vAcc1 = 0.0f, vAcc2 = 0.0f;
    for (uint b = vGrp; b < cnt; b += 4u) {
        const uint baseSeg = blkIdx[b] * nkv + g;
        const uint s0 = b << 2u;
        const float4 sc0 = *((threadgroup const float4*)(score + s0));
        const float4 sc1 = *((threadgroup const float4*)(score + STRIDE + s0));
        const float4 sc2 = *((threadgroup const float4*)(score + 2u * STRIDE + s0));
        const float4 v0 = qwg_fn_kv_read_vec4<KV_MODE>(v, vMeta, baseSeg,            vHalf, lane);
        const float4 v1 = qwg_fn_kv_read_vec4<KV_MODE>(v, vMeta, baseSeg + nkv,      vHalf, lane);
        const float4 v2 = qwg_fn_kv_read_vec4<KV_MODE>(v, vMeta, baseSeg + 2u * nkv, vHalf, lane);
        const float4 v3 = qwg_fn_kv_read_vec4<KV_MODE>(v, vMeta, baseSeg + 3u * nkv, vHalf, lane);
        vAcc0 = fma(sc0.x, v0, vAcc0);
        vAcc0 = fma(sc0.y, v1, vAcc0);
        vAcc0 = fma(sc0.z, v2, vAcc0);
        vAcc0 = fma(sc0.w, v3, vAcc0);
        vAcc1 = fma(sc1.x, v0, vAcc1);
        vAcc1 = fma(sc1.y, v1, vAcc1);
        vAcc1 = fma(sc1.z, v2, vAcc1);
        vAcc1 = fma(sc1.w, v3, vAcc1);
        vAcc2 = fma(sc2.x, v0, vAcc2);
        vAcc2 = fma(sc2.y, v1, vAcc2);
        vAcc2 = fma(sc2.z, v2, vAcc2);
        vAcc2 = fma(sc2.w, v3, vAcc2);
    }
    if (vGrp == 0u) {
        for (uint slot = cnt * BLOCK; slot < slots; slot++) {
            const uint j = tailStart + (slot - cnt * BLOCK);
            const float4 vv = qwg_fn_kv_read_vec4<KV_MODE>(v, vMeta, j * nkv + g, vHalf, lane);
            vAcc0 = fma(score[slot],               vv, vAcc0);
            vAcc1 = fma(score[STRIDE + slot],      vv, vAcc1);
            vAcc2 = fma(score[2u * STRIDE + slot], vv, vAcc2);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *((threadgroup float4*)(score + (0u * 4u + vGrp) * D + dBase)) = vAcc0;
    *((threadgroup float4*)(score + (1u * 4u + vGrp) * D + dBase)) = vAcc1;
    *((threadgroup float4*)(score + (2u * 4u + vGrp) * D + dBase)) = vAcc2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float acc0 = score[0u * D + d] + score[1u * D + d] + score[2u * D + d] + score[3u * D + d];
    const float acc1 = score[4u * D + d] + score[5u * D + d] + score[6u * D + d] + score[7u * D + d];
    const float acc2 = score[8u * D + d] + score[9u * D + d] + score[10u * D + d] + score[11u * D + d];
    const float oArr[3] = { acc0 * invL0, acc1 * invL1, acc2 * invL2 };
    const uint lane2 = d & 31u;
    #pragma unroll
    for (uint u = 0u; u < 3u; u++) {
        const uint idx = (t * nh + h0 + u) * D + d;
        float x = oArr[u] / (1.0f + exp(-gate[idx]));
        x = qwg_fwht32_lane_f32(x, lane2);
        const float rotVal = x;
        if (hasOut16 != 0u) {
            out16[idx] = half(rotVal);
        } else {
            out[idx] = rotVal;
        }
    }
}

#define DEFINE_ATTN_SPARSE_GATED_KERNEL(NAME, MODE) \
kernel void NAME(device const float* q        [[buffer(0)]], \
                 device const uchar* k        [[buffer(1)]], \
                 device const uchar* v        [[buffer(2)]], \
                 device const float* gate     [[buffer(3)]], \
                 device const int*   list     [[buffer(4)]], \
                 device float*       out      [[buffer(5)]], \
                 constant uint&      T        [[buffer(6)]], \
                 constant uint&      nh       [[buffer(7)]], \
                 constant uint&      nkv      [[buffer(8)]], \
                 constant uint&      pos0     [[buffer(9)]], \
                 device const half*  kMeta    [[buffer(10)]], \
                 device const half*  vMeta    [[buffer(11)]], \
                 device half*        out16    [[buffer(12)]], \
                 constant uint&      hasOut16 [[buffer(13)]], \
                 uint tg [[threadgroup_position_in_grid]], \
                 uint d  [[thread_position_in_threadgroup]]) { \
    threadgroup float score[2052]; \
    threadgroup uint blkIdx[512]; \
    threadgroup float sgScratch[8]; \
    qwg_attn_sparse_gated_impl<MODE>(q, k, v, gate, list, out, T, nh, nkv, pos0, kMeta, vMeta, out16, hasOut16, score, blkIdx, sgScratch, tg, d); \
}

#define DEFINE_ATTN_SPARSE_GATED_GQA3_KERNEL(NAME, MODE) \
kernel void NAME(device const float* q        [[buffer(0)]], \
                 device const uchar* k        [[buffer(1)]], \
                 device const uchar* v        [[buffer(2)]], \
                 device const float* gate     [[buffer(3)]], \
                 device const int*   list     [[buffer(4)]], \
                 device float*       out      [[buffer(5)]], \
                 constant uint&      T        [[buffer(6)]], \
                 constant uint&      nh       [[buffer(7)]], \
                 constant uint&      nkv      [[buffer(8)]], \
                 constant uint&      pos0     [[buffer(9)]], \
                 device const half*  kMeta    [[buffer(10)]], \
                 device const half*  vMeta    [[buffer(11)]], \
                 device half*        out16    [[buffer(12)]], \
                 constant uint&      hasOut16 [[buffer(13)]], \
                 uint tg [[threadgroup_position_in_grid]], \
                 uint d  [[thread_position_in_threadgroup]]) { \
    threadgroup float score[6156]; \
    threadgroup uint blkIdx[512]; \
    threadgroup float sgScratch[24]; \
    qwg_attn_sparse_gated_gqa3_impl<MODE>(q, k, v, gate, list, out, T, nh, nkv, pos0, kMeta, vMeta, out16, hasOut16, score, blkIdx, sgScratch, tg, d); \
}

DEFINE_ATTN_SPARSE_GATED_KERNEL(qwg_attn_sparse_gated_f32, 0u)
DEFINE_ATTN_SPARSE_GATED_KERNEL(qwg_attn_sparse_gated_f16, 1u)
DEFINE_ATTN_SPARSE_GATED_KERNEL(qwg_attn_sparse_gated_q8,  2u)
DEFINE_ATTN_SPARSE_GATED_KERNEL(qwg_attn_sparse_gated_q4,  3u)

DEFINE_ATTN_SPARSE_GATED_GQA3_KERNEL(qwg_attn_sparse_gated_gqa3_f32, 0u)
DEFINE_ATTN_SPARSE_GATED_GQA3_KERNEL(qwg_attn_sparse_gated_gqa3_f16, 1u)
DEFINE_ATTN_SPARSE_GATED_GQA3_KERNEL(qwg_attn_sparse_gated_gqa3_q8,  2u)
DEFINE_ATTN_SPARSE_GATED_GQA3_KERNEL(qwg_attn_sparse_gated_gqa3_q4,  3u)
