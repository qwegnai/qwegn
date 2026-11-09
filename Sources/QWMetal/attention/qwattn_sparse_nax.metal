// qwattn_sparse_nax.metal — M5 NAX (tensor-op) sparse gated attention for prefill rows (T >= 4).
// One threadgroup per (row, kv head) holds all GQA query heads; Q·Kᵀ and P·V run on matmul2d tiles.
// KV cache modes: f32 (0), f16 (1), q8 (2), q4 (3); dequantised on the fly. Q, gate, out stay f32.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "include/qw_weight_basis.metalh"
using namespace metal;

template <uint KV_MODE>
static inline half2 qwg_fn_kv_meta_half(device const half* meta, uint seg, ushort dh) {
    if constexpr (KV_MODE == 2u) {
        return half2(meta[seg], 0.0h);
    } else if constexpr (KV_MODE == 3u) {
        return ((device const half2*)meta)[seg * 2u + uint(dh)];
    } else {
        return half2(1.0h, 0.0h);
    }
}

template <uint KV_MODE>
static inline vec<half, 8> qwg_fn_kv_load_vec8(device const uchar* buf, uint seg, uint d0, half2 m) {
    constexpr uint D = 256u;
    if constexpr (KV_MODE == 0u) {
        device const float* row = ((device const float*)buf) + (uint64_t)seg * D + d0;
        const float4 lo = *(device const float4*)row;
        const float4 hi = *(device const float4*)(row + 4u);
        return vec<half, 8>(half(lo.x), half(lo.y), half(lo.z), half(lo.w),
                            half(hi.x), half(hi.y), half(hi.z), half(hi.w));
    } else if constexpr (KV_MODE == 1u) {
        return *(device const vec<half, 8>*)(((device const half*)buf) + (uint64_t)seg * D + d0);
    } else if constexpr (KV_MODE == 2u) {
        device const char* row = ((device const char*)buf) + (uint64_t)seg * D + d0;
        const char4 c0 = *(device const char4*)row;
        const char4 c1 = *(device const char4*)(row + 4u);
        const half sc = m.x;
        return vec<half, 8>(half(c0.x) * sc, half(c0.y) * sc, half(c0.z) * sc, half(c0.w) * sc,
                            half(c1.x) * sc, half(c1.y) * sc, half(c1.z) * sc, half(c1.w) * sc);
    } else {
        const uchar4 b4 = *(device const uchar4*)(buf + (uint64_t)seg * (D / 2u) + (d0 >> 1u));
        const half sc = m.x, mn = m.y;
        return vec<half, 8>(fma(half(b4.x & 0x0Fu), sc, mn),
                            fma(half(b4.x >> 4u),   sc, mn),
                            fma(half(b4.y & 0x0Fu), sc, mn),
                            fma(half(b4.y >> 4u),   sc, mn),
                            fma(half(b4.z & 0x0Fu), sc, mn),
                            fma(half(b4.z >> 4u),   sc, mn),
                            fma(half(b4.w & 0x0Fu), sc, mn),
                            fma(half(b4.w >> 4u),   sc, mn));
    }
}

METAL_FUNC void qwg_fwht32_vec8_f32(thread float (&w)[8], ushort lane) {
    if (!QW_FWHT_ROTATED) return;
    #pragma unroll
    for (short k = 0; k < 8; k += 2) {
        const float a = w[k], b = w[k + 1];
        w[k] = a + b;
        w[k + 1] = a - b;
    }
    #pragma unroll
    for (short k = 0; k < 8; k += 4) {
        const float a0 = w[k], b0 = w[k + 2];
        const float a1 = w[k + 1], b1 = w[k + 3];
        w[k] = a0 + b0;     w[k + 2] = a0 - b0;
        w[k + 1] = a1 + b1; w[k + 3] = a1 - b1;
    }
    #pragma unroll
    for (short k = 0; k < 4; ++k) {
        const float a = w[k], b = w[k + 4];
        w[k] = a + b;
        w[k + 4] = a - b;
    }
    const bool hi8 = (lane & 1u) != 0u;
    #pragma unroll
    for (short k = 0; k < 8; ++k) {
        const float p = simd_shuffle_xor(w[k], ushort(1));
        w[k] = hi8 ? (p - w[k]) : (w[k] + p);
    }
    const bool hi16 = (lane & 8u) != 0u;
    #pragma unroll
    for (short k = 0; k < 8; ++k) {
        const float p = simd_shuffle_xor(w[k], ushort(8));
        w[k] = (hi16 ? (p - w[k]) : (w[k] + p)) * 0.17677669529663689f;
    }
}

template <uint KV_MODE>
static inline void qwg_attn_sparse_gated_nax_impl(device const float* q,
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
                                                  threadgroup float (&xchg)[2][256],
                                                  uint2 tg, ushort dh, ushort lane) {
    constexpr uint D = 256u;
    constexpr uint TOPK = 512u;
    constexpr int SB = 8;
    const uint tq = tg.x;
    const uint kvh = tg.y;
    if (tq >= T || kvh >= nkv) return;
    const uint gqa = nh / nkv;
    const short qid = short(lane >> 2u);
    const short fm = (qid & 4) | (short(lane >> 1u) & 3);
    const short fn = ((qid & 2) | short(lane & 1u)) * 4;
    const uint fcol = uint(2 * fn);
    const uint dBase = uint(dh) * 128u + fcol;
    const bool r0_ok = uint(fm) < gqa;
    const bool r1_ok = uint(fm + 8) < gqa;
    const uint h0 = kvh * gqa + (r0_ok ? uint(fm) : 0u);
    const uint h1 = kvh * gqa + (r1_ok ? uint(fm + 8) : 0u);

    device const float* qp0 = q + (tq * nh + h0) * D + dBase;
    device const float* qp1 = q + (tq * nh + h1) * D + dBase;
    vec<half, 8> q0_reg[4];
    vec<half, 8> q1_reg[4];
    #pragma unroll
    for (short jj = 0; jj < 4; ++jj) {
        if (r0_ok) {
            const float4 lo = *(device const float4*)(qp0 + 32 * jj);
            const float4 hi = *(device const float4*)(qp0 + 32 * jj + 4);
            q0_reg[jj] = vec<half, 8>(half(lo.x), half(lo.y), half(lo.z), half(lo.w),
                                      half(hi.x), half(hi.y), half(hi.z), half(hi.w));
        } else {
            q0_reg[jj] = vec<half, 8>(0.0h);
        }
        if (r1_ok) {
            const float4 lo = *(device const float4*)(qp1 + 32 * jj);
            const float4 hi = *(device const float4*)(qp1 + 32 * jj + 4);
            q1_reg[jj] = vec<half, 8>(half(lo.x), half(lo.y), half(lo.z), half(lo.w),
                                      half(hi.x), half(hi.y), half(hi.z), half(hi.w));
        } else {
            q1_reg[jj] = vec<half, 8>(0.0h);
        }
    }

    constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 32, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
    constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 32, false, false, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<pv_desc, metal::execution_simdgroup> pv_op;

    auto qa_ct = qk_op.template get_left_input_cooperative_tensor<half, half, float>();
    auto kb_ct = qk_op.template get_right_input_cooperative_tensor<half, half, float>();
    using qa_t = metal::remove_addrspace_t<decltype(qa_ct)>;
    using kb_t = metal::remove_addrspace_t<decltype(kb_ct)>;
    auto pa_ct = pv_op.template get_left_input_cooperative_tensor<half, half, float>();
    auto vb_ct = pv_op.template get_right_input_cooperative_tensor<half, half, float>();
    using pa_t = metal::remove_addrspace_t<decltype(pa_ct)>;
    using vb_t = metal::remove_addrspace_t<decltype(vb_ct)>;

    auto of0 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
    auto of1 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
    auto of2 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
    auto of3 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
    #pragma unroll
    for (short i = 0; i < 16; ++i) {
        of0[i] = 0.0f;
        of1[i] = 0.0f;
        of2[i] = 0.0f;
        of3[i] = 0.0f;
    }

    const uint pq = pos0 + tq;
    const int complete = int((pq + 1u) >> 2u);
    const int nsel = min(int(TOPK), complete);
    const int ntail = int(pq + 1u) - (complete << 2);
    const int U = nsel + (ntail > 0 ? 1 : 0);
    const int nsteps = (U + SB - 1) / SB;
    const float scale2 = rsqrt(float(D)) * 1.4426950408889634f;
    device const int* blk = list + tq * TOPK;

    float max_s[2] = {-FLT_MAX, -FLT_MAX};
    float sum_s[2] = {0.0f, 0.0f};

    auto blk_at = [&](int u) -> int {
        if (u < nsel) {
            return (complete <= int(TOPK)) ? u : blk[u];
        }
        return complete;
    };
    int nrow_b[4];
    int ncol_n[2];
    auto fetch = [&](int u0) {
        #pragma unroll
        for (short r = 0; r < 4; ++r) {
            const int u = u0 + ((16 * (r >> 1) + int(fm) + 8 * (r & 1)) >> 2);
            nrow_b[r] = (u < U) ? blk_at(u) : 0;
        }
        #pragma unroll
        for (short f = 0; f < 2; ++f) {
            const int u = u0 + 4 * f + (int(fn) >> 2);
            ncol_n[f] = (u < nsel) ? 4 : ((u < U) ? ntail : 0);
        }
    };
    fetch(0);

    for (int step = 0; step < nsteps; ++step) {
        uint kseg[4];
        half2 km[4];
        #pragma unroll
        for (short r = 0; r < 4; ++r) {
            const uint krow = min(uint(nrow_b[r]) * 4u + uint(fm & 3), pq);
            kseg[r] = krow * nkv + kvh;
            km[r] = qwg_fn_kv_meta_half<KV_MODE>(kMeta, kseg[r], dh);
        }
        int nvis[2];
        #pragma unroll
        for (short f = 0; f < 2; ++f) {
            nvis[f] = ncol_n[f];
        }
        if (step + 1 < nsteps) {
            fetch((step + 1) * SB);
        }

        auto s_ct = qk_op.template get_destination_cooperative_tensor<qa_t, kb_t, float>();
        #pragma unroll
        for (short i = 0; i < 16; ++i) {
            s_ct[i] = 0.0f;
        }
        #pragma unroll
        for (short jj = 0; jj < 4; ++jj) {
            const uint d0 = dBase + uint(32 * jj);
            const vec<half, 8> qa = q0_reg[jj];
            const vec<half, 8> qb = q1_reg[jj];
            const vec<half, 8> a0 = qwg_fn_kv_load_vec8<KV_MODE>(k, kseg[0], d0, km[0]);
            const vec<half, 8> a1 = qwg_fn_kv_load_vec8<KV_MODE>(k, kseg[1], d0, km[1]);
            const vec<half, 8> a2 = qwg_fn_kv_load_vec8<KV_MODE>(k, kseg[2], d0, km[2]);
            const vec<half, 8> a3 = qwg_fn_kv_load_vec8<KV_MODE>(k, kseg[3], d0, km[3]);
            #pragma unroll
            for (short j = 0; j < 4; ++j) {
                qa_ct[j]      = qa[j];
                qa_ct[4 + j]  = qb[j];
                qa_ct[8 + j]  = qa[4 + j];
                qa_ct[12 + j] = qb[4 + j];
                kb_ct[j]      = a0[j];
                kb_ct[4 + j]  = a1[j];
                kb_ct[8 + j]  = a2[j];
                kb_ct[12 + j] = a3[j];
                kb_ct[16 + j] = a0[4 + j];
                kb_ct[20 + j] = a1[4 + j];
                kb_ct[24 + j] = a2[4 + j];
                kb_ct[28 + j] = a3[4 + j];
            }
            qk_op.run(qa_ct, kb_ct, s_ct);
        }

        vec<float, 8> s[2];
        #pragma unroll
        for (short i = 0; i < 8; ++i) {
            s[0][i] = s_ct[i];
            s[1][i] = s_ct[8 + i];
        }
        #pragma unroll
        for (short f = 0; f < 2; ++f) {
            threadgroup float* mine = xchg[dh];
            const threadgroup float* peer = xchg[1u - dh];
            #pragma unroll
            for (short i = 0; i < 8; ++i) {
                mine[lane * 8u + ushort(i)] = s[f][i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (short i = 0; i < 8; ++i) {
                s[f][i] += peer[lane * 8u + ushort(i)];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        #pragma unroll
        for (short f = 0; f < 2; ++f) {
            #pragma unroll
            for (short j = 0; j < 4; ++j) {
                const bool ok = j < nvis[f];
                s[f][j]     = ok ? (s[f][j] * scale2) : -INFINITY;
                s[f][4 + j] = ok ? (s[f][4 + j] * scale2) : -INFINITY;
            }
        }

        float factor[2];
        #pragma unroll
        for (short i = 0; i < 2; ++i) {
            float m = -INFINITY;
            #pragma unroll
            for (short f = 0; f < 2; ++f) {
                m = max(m, max(max(s[f][4 * i], s[f][4 * i + 1]), max(s[f][4 * i + 2], s[f][4 * i + 3])));
            }
            m = max(m, simd_shuffle_xor(m, ushort(1)));
            m = max(m, simd_shuffle_xor(m, ushort(8)));
            const float new_max = max(max_s[i], m);
            float rs = 0.0f;
            #pragma unroll
            for (short f = 0; f < 2; ++f) {
                #pragma unroll
                for (short j = 0; j < 4; ++j) {
                    s[f][4 * i + j] = fast::exp2(s[f][4 * i + j] - new_max);
                    rs += s[f][4 * i + j];
                }
            }
            rs += simd_shuffle_xor(rs, ushort(1));
            rs += simd_shuffle_xor(rs, ushort(8));
            factor[i] = fast::exp2(max_s[i] - new_max);
            max_s[i] = new_max;
            sum_s[i] = sum_s[i] * factor[i] + rs;
        }

        if (simd_any(factor[0] != 1.0f || factor[1] != 1.0f)) {
            #pragma unroll
            for (short h = 0; h < 2; ++h) {
                #pragma unroll
                for (short j = 0; j < 4; ++j) {
                    of0[8 * h + j]     *= factor[0];
                    of0[8 * h + 4 + j] *= factor[1];
                    of1[8 * h + j]     *= factor[0];
                    of1[8 * h + 4 + j] *= factor[1];
                    of2[8 * h + j]     *= factor[0];
                    of2[8 * h + 4 + j] *= factor[1];
                    of3[8 * h + j]     *= factor[0];
                    of3[8 * h + 4 + j] *= factor[1];
                }
            }
        }

        // P·V: one K=32 MMA per 32-key step; f=0 keys fill the first K half, f=1 keys the second.
        #pragma unroll
        for (short i = 0; i < 8; ++i) {
            pa_ct[i]     = half(s[0][i]);
            pa_ct[8 + i] = half(s[1][i]);
        }
        const half2 vm0 = qwg_fn_kv_meta_half<KV_MODE>(vMeta, kseg[0], dh);
        const half2 vm1 = qwg_fn_kv_meta_half<KV_MODE>(vMeta, kseg[1], dh);
        const half2 vm2 = qwg_fn_kv_meta_half<KV_MODE>(vMeta, kseg[2], dh);
        const half2 vm3 = qwg_fn_kv_meta_half<KV_MODE>(vMeta, kseg[3], dh);
        #pragma unroll
        for (short jj = 0; jj < 4; ++jj) {
            const uint d0 = dBase + uint(32 * jj);
            const vec<half, 8> ra = qwg_fn_kv_load_vec8<KV_MODE>(v, kseg[0], d0, vm0);
            const vec<half, 8> rb = qwg_fn_kv_load_vec8<KV_MODE>(v, kseg[1], d0, vm1);
            const vec<half, 8> rc = qwg_fn_kv_load_vec8<KV_MODE>(v, kseg[2], d0, vm2);
            const vec<half, 8> rd = qwg_fn_kv_load_vec8<KV_MODE>(v, kseg[3], d0, vm3);
            #pragma unroll
            for (short j = 0; j < 4; ++j) {
                vb_ct[j]      = ra[j];
                vb_ct[4 + j]  = rb[j];
                vb_ct[8 + j]  = ra[4 + j];
                vb_ct[12 + j] = rb[4 + j];
                vb_ct[16 + j] = rc[j];
                vb_ct[20 + j] = rd[j];
                vb_ct[24 + j] = rc[4 + j];
                vb_ct[28 + j] = rd[4 + j];
            }
            if (jj == 0) {
                pv_op.run(pa_ct, vb_ct, of0);
            } else if (jj == 1) {
                pv_op.run(pa_ct, vb_ct, of1);
            } else if (jj == 2) {
                pv_op.run(pa_ct, vb_ct, of2);
            } else {
                pv_op.run(pa_ct, vb_ct, of3);
            }
        }
    }

    #pragma unroll
    for (short i = 0; i < 2; ++i) {
        const bool row_ok = (i == 0) ? r0_ok : r1_ok;
        const uint h = (i == 0) ? h0 : h1;
        const float rr = (sum_s[i] > 0.0f) ? (1.0f / sum_s[i]) : 0.0f;
        #pragma unroll
        for (short jj = 0; jj < 4; ++jj) {
            const uint baseIdx = (tq * nh + h) * D + dBase + uint(32 * jj);
            float w[8];
            #pragma unroll
            for (short j = 0; j < 4; ++j) {
                float e0, e1;
                if (jj == 0) {
                    e0 = of0[4 * i + j];
                    e1 = of0[8 + 4 * i + j];
                } else if (jj == 1) {
                    e0 = of1[4 * i + j];
                    e1 = of1[8 + 4 * i + j];
                } else if (jj == 2) {
                    e0 = of2[4 * i + j];
                    e1 = of2[8 + 4 * i + j];
                } else {
                    e0 = of3[4 * i + j];
                    e1 = of3[8 + 4 * i + j];
                }
                w[j]     = e0 * rr;
                w[4 + j] = e1 * rr;
            }
            if (row_ok) {
                const float4 g0 = *(device const float4*)(gate + baseIdx);
                const float4 g1 = *(device const float4*)(gate + baseIdx + 4u);
                w[0] /= (1.0f + exp(-g0.x));
                w[1] /= (1.0f + exp(-g0.y));
                w[2] /= (1.0f + exp(-g0.z));
                w[3] /= (1.0f + exp(-g0.w));
                w[4] /= (1.0f + exp(-g1.x));
                w[5] /= (1.0f + exp(-g1.y));
                w[6] /= (1.0f + exp(-g1.z));
                w[7] /= (1.0f + exp(-g1.w));
            } else {
                #pragma unroll
                for (short k = 0; k < 8; ++k) w[k] = 0.0f;
            }
            qwg_fwht32_vec8_f32(w, lane);
            if (row_ok) {
                if (hasOut16 != 0u) {
                    *(device half4*)(out16 + baseIdx)      = half4(half(w[0]), half(w[1]), half(w[2]), half(w[3]));
                    *(device half4*)(out16 + baseIdx + 4u) = half4(half(w[4]), half(w[5]), half(w[6]), half(w[7]));
                } else {
                    *(device float4*)(out + baseIdx)      = float4(w[0], w[1], w[2], w[3]);
                    *(device float4*)(out + baseIdx + 4u) = float4(w[4], w[5], w[6], w[7]);
                }
            }
        }
    }
}

#define DEFINE_ATTN_SPARSE_GATED_NAX_KERNEL(NAME, MODE) \
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
                 uint2  tg   [[threadgroup_position_in_grid]], \
                 ushort dh   [[simdgroup_index_in_threadgroup]], \
                 ushort lane [[thread_index_in_simdgroup]]) { \
    threadgroup float xchg[2][256]; \
    qwg_attn_sparse_gated_nax_impl<MODE>(q, k, v, gate, list, out, T, nh, nkv, pos0, kMeta, vMeta, out16, hasOut16, xchg, tg, dh, lane); \
}

DEFINE_ATTN_SPARSE_GATED_NAX_KERNEL(qwg_attn_sparse_gated_nax_f32, 0u)
DEFINE_ATTN_SPARSE_GATED_NAX_KERNEL(qwg_attn_sparse_gated_nax_f16, 1u)
DEFINE_ATTN_SPARSE_GATED_NAX_KERNEL(qwg_attn_sparse_gated_nax_q8,  2u)
DEFINE_ATTN_SPARSE_GATED_NAX_KERNEL(qwg_attn_sparse_gated_nax_q4,  3u)
