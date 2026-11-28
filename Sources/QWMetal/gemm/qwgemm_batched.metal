// qwgemm_batched.metal — High-performance batched HQ4_G32_M5 GEMM for prefill on Apple Silicon (M5 NAX).
// Computes Y = X · Wᵀ where:
//   X: [M, K] half (device memory, rotated in H_32 basis)
//   W: [N, K/32] Block_HQ4_G32 quantized weights (device memory, 20B/block)
//   Y: [M, N] half (+ optional f32 store)
//
// Powered by Apple Silicon Neural Accelerator Extensions (NAX) / MetalPerformancePrimitives.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_dequant.metalh"
#include "include/qw_moe_tile.metalh"
#include "include/qw_geometry_constants.metalh"

using namespace metal;

#if !defined(QW_DISABLE_NAX)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#if !defined(__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__) || __ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ < 260200
#error "NAX cooperative_tensor kernels require -mmacosx-version-min=26.2 or newer (Apple MPP operand_index ABI)."
#endif
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"

using namespace mlx::steel;

#include "include/qw_gemm_nax.metalh"

kernel void qwg_gemm_q4g_h16(device const Block_HQ4_G32* blocks [[buffer(0)]],
                             device const half*          x      [[buffer(1)]],
                             device half*                y16    [[buffer(2)]],
                             device float*               y32    [[buffer(3)]],
                             constant uint&              M      [[buffer(4)]],
                             constant uint&              N      [[buffer(5)]],
                             constant uint&              K      [[buffer(6)]],
                             uint3 tg [[threadgroup_position_in_grid]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint simd_gid [[simdgroup_index_in_threadgroup]],
                             uint simd_lid [[thread_index_in_simdgroup]]) {
    qwg_gemm_h16_mode_dispatch((const device char*)blocks, x, y16, y32, M, N, K, tg, lid, simd_gid, simd_lid);
}

kernel void qwg_gemm_q4g_h16_gdn_inproj(device const Block_HQ4_G32* blocks [[buffer(0)]],
                                        device const half*          x      [[buffer(1)]],
                                        device half*                qkv16  [[buffer(2)]],
                                        device float*               y32    [[buffer(3)]],
                                        constant uint&              M      [[buffer(4)]],
                                        constant uint&              N      [[buffer(5)]],
                                        constant uint&              K      [[buffer(6)]],
                                        device half*                z16    [[buffer(7)]],
                                        uint3 tg [[threadgroup_position_in_grid]],
                                        uint lid [[thread_index_in_threadgroup]],
                                        uint simd_gid [[simdgroup_index_in_threadgroup]],
                                        uint simd_lid [[thread_index_in_simdgroup]]) {
    qmm_t_nax_tgp_dual_impl<half, half, false, 64, 64, 64, 2, 2, true, true>(
        (const device char*)blocks, x, qkv16, y32, nullptr, int(K), int(N), int(M), tg, lid, simd_gid, simd_lid, z16);
}
#endif // !defined(QW_DISABLE_NAX)


// ── Batched MLP SiLU × Up epilogue + fused FWHT-32 rotation ──
// Takes gu: [M, 2 * I] half, writes out: [M, I] half in rotated H_32 basis.
// gu row m has gate at [0..I-1], up at [I..2I-1].
kernel void qwg_silu_mul_f16(device const half* gu [[buffer(0)]],
                             device half*       out [[buffer(1)]],
                             constant uint&     M  [[buffer(2)]],
                             constant uint&     I  [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]]) {
    constexpr uint kU = 8u;                      // elements per thread (128-bit)
    const uint mi = gid.x * kU;                  // column of this thread's first elem
    const uint m = gid.y;
    if (m >= M || mi >= I) return;
    const size_t in_row = (size_t)m * (2u * I);
    const size_t out_row = (size_t)m * I;
    if (mi + kU <= I) {                          // fast path: all eight in range
        device const half4* gp = (device const half4*)(gu + in_row + mi);
        device const half4* up = (device const half4*)(gu + in_row + I + mi);
        const half4 g0 = gp[0], g1 = gp[1];
        const half4 u0 = up[0], u1 = up[1];
        half ov[8] = {
            half(qwg_swiglu_mul(float(g0[0]), float(u0[0]))),
            half(qwg_swiglu_mul(float(g0[1]), float(u0[1]))),
            half(qwg_swiglu_mul(float(g0[2]), float(u0[2]))),
            half(qwg_swiglu_mul(float(g0[3]), float(u0[3]))),
            half(qwg_swiglu_mul(float(g1[0]), float(u1[0]))),
            half(qwg_swiglu_mul(float(g1[1]), float(u1[1]))),
            half(qwg_swiglu_mul(float(g1[2]), float(u1[2]))),
            half(qwg_swiglu_mul(float(g1[3]), float(u1[3])))
        };
        if ((I & 31u) == 0u) {
            fwht32_inplace(ov, gid.x & 31u);
        }
        device half4* op = (device half4*)(out + out_row + mi);
        op[0] = half4(ov[0], ov[1], ov[2], ov[3]);
        op[1] = half4(ov[4], ov[5], ov[6], ov[7]);
    } else {                                     // tail lanes (I % 8 != 0)
        for (uint k = 0; k < kU && mi + k < I; k++) {
            const float g = float(gu[in_row + mi + k]);
            const float u = float(gu[in_row + I + mi + k]);
            out[out_row + mi + k] = half(qwg_swiglu_mul(g, u));
        }
    }
}

kernel void qwg_kv_paged_write(device half*       kpool    [[buffer(0)]],
                               device half*       vpool    [[buffer(1)]],
                               const device int*  ktab     [[buffer(2)]],
                               const device int*  vtab     [[buffer(3)]],
                               device const half* k        [[buffer(4)]], // [T, 1024]
                               device const half* v        [[buffer(5)]], // [T, 1024]
                               constant uint&     startPos [[buffer(6)]],
                               constant uint&     T        [[buffer(7)]],
                               constant uint&     q8mode   [[buffer(8)]], // 0 f16, 1 q8
                               uint tid [[thread_index_in_threadgroup]],
                               uint3 tg [[threadgroup_position_in_grid]]) {
    const uint kvh = tg.x; // 0..3 (4 kv-heads)
    const uint t = tg.y;   // 0..<T
    if (t >= T) return;
    const uint lane = tid & 31u;
    const uint tok = startPos + t;
    const uint page = tok >> 8;
    const uint slot = tok & 255u;

    constexpr uint QW_KV_HEADS = 4u;
    constexpr uint QW_HEAD_DIM = 256u;
    constexpr uint QW_KV_ROW = QW_KV_HEADS * QW_HEAD_DIM;      // 1024
    constexpr uint QW_KV_PGEL = 256u * QW_KV_ROW;               // 262144
    // q8 page = 262144 B payload + 2048 B f16 scale array (QWKVCompressedLayout).
    constexpr uint64_t QW_KV_PGE8 = 262144ull + 2048ull;
    const uint d0 = lane * 8u;

    device const half* kn = k + (uint64_t)t * QW_KV_ROW + kvh * QW_HEAD_DIM + d0;
    device const half* vn = v + (uint64_t)t * QW_KV_ROW + kvh * QW_HEAD_DIM + d0;

    if (q8mode == 0u) {
        device half* kw = kpool + (uint64_t)ktab[page] * QW_KV_PGEL
                          + (uint64_t)slot * QW_KV_ROW + kvh * QW_HEAD_DIM + d0;
        device half* vw = vpool + (uint64_t)vtab[page] * QW_KV_PGEL
                          + (uint64_t)slot * QW_KV_ROW + kvh * QW_HEAD_DIM + d0;
        *(device uint4*)kw = *(device const uint4*)kn;
        *(device uint4*)vw = *(device const uint4*)vn;
        return;
    }
    half ks[8], vs[8];
    for (uint i = 0; i < 8u; i++) { ks[i] = kn[i]; vs[i] = vn[i]; }
    if (q8mode == 1u) {
        // q8: the workgroup's 32 lanes own EXACTLY the (row=tok, head=kvh)
        // 256-dim segment → one simd_max, then quantise. Identical maths to
        // qwg_q8_store_seg in qwattn_decode.metal (same f16 scale value, same
        float km = 0.0f, vm = 0.0f;
        for (uint i = 0; i < 8u; i++) { km = fmax(km, fabs(float(ks[i])));
                                       vm = fmax(vm, fabs(float(vs[i]))); }
        km = simd_max(km); vm = simd_max(vm);
        const half k16 = half(fmax(km / 127.0f, 1e-8f));
        const half v16 = half(fmax(vm / 127.0f, 1e-8f));
        const float kinv = 1.0f / float(k16), vinv = 1.0f / float(v16);
        device char* kw = (device char*)kpool + (uint64_t)ktab[page] * QW_KV_PGE8
                          + (uint64_t)slot * QW_KV_ROW + kvh * QW_HEAD_DIM + d0;
        device char* vw = (device char*)vpool + (uint64_t)vtab[page] * QW_KV_PGE8
                          + (uint64_t)slot * QW_KV_ROW + kvh * QW_HEAD_DIM + d0;
        for (uint i = 0; i < 8u; i++) {
            kw[i] = (char)(int)clamp(round(float(ks[i]) * kinv), -127.0f, 127.0f);
            vw[i] = (char)(int)clamp(round(float(vs[i]) * vinv), -127.0f, 127.0f);
        }
        if (d0 == 0u) {   // scale array inside the page, after the payload
            *(device half*)((device char*)kpool + (uint64_t)ktab[page] * QW_KV_PGE8
                            + QW_KV_PGEL + (uint64_t)slot * (QW_KV_HEADS * 2u)
                            + kvh * 2u) = k16;
            *(device half*)((device char*)vpool + (uint64_t)vtab[page] * QW_KV_PGE8
                            + QW_KV_PGEL + (uint64_t)slot * (QW_KV_HEADS * 2u)
                            + kvh * 2u) = v16;
        }
        return;
    }

    // q4 (asymmetric group-of-128): the 128-element quantisation group is
    // covered by an aligned 16-lane set (lane owns head-dims [d0, d0+8);
    // group g = head-dims [128g,128g+128) = lanes 16g..16g+15). Reduce with
    // xor butterflies 1,2,4,8 — bit 4 of the lane index never flips, so each
    // aligned 16 set folds independently (32-lane simd_max would span FOUR
    // groups and is WRONG here). Math bit-for-bit quantizeQ4:
    //   sc = f16(max((mx-mn)/15, 1e-8)); nib = clamp(round((x-f16(mn))/sc),0,15)
    // payload: LOW nibble = even element (quantizeQ4 packs q0 | q1<<4);
    // header per group: {sc f16, mn f16} at page offset 131072 + grp*4.
    constexpr uint64_t QW_KV_PGE4 = 131072ull + 8192ull;
    float mnk = 1e30f, mxk = -1e30f, mnv4 = 1e30f, mxv4 = -1e30f;
    for (uint i = 0; i < 8u; i++) {
        const float a = float(ks[i]), b = float(vs[i]);
        mnk = fmin(mnk, a); mxk = fmax(mxk, a);
        mnv4 = fmin(mnv4, b); mxv4 = fmax(mxv4, b);
    }
    #pragma unroll
    for (uint k = 1u; k <= 8u; k <<= 1) {
        mnk = fmin(mnk, simd_shuffle_xor(mnk, k)); mxk = fmax(mxk, simd_shuffle_xor(mxk, k));
        mnv4 = fmin(mnv4, simd_shuffle_xor(mnv4, k)); mxv4 = fmax(mxv4, simd_shuffle_xor(mxv4, k));
    }
    const half ksc16 = half(fmax((mxk - mnk) / 15.0f, 1e-8f));
    const half kmn16 = half(mnk);
    const half vsc16 = half(fmax((mxv4 - mnv4) / 15.0f, 1e-8f));
    const half vmn16 = half(mnv4);
    const float ksc = float(ksc16), kmn = float(kmn16);
    const float vsc = float(vsc16), vmn = float(vmn16);
    const float kinv = 1.0f / ksc, vinv = 1.0f / vsc;
    // 8 dims = 4 payload bytes at byte (head*128 + d0/2) of the row.
    device uchar* kd = (device uchar*)kpool + (uint64_t)ktab[page] * QW_KV_PGE4
                       + (uint64_t)slot * (QW_KV_ROW / 2u)
                       + (uint64_t)kvh * (QW_HEAD_DIM / 2u) + (d0 >> 1);
    device uchar* vd = (device uchar*)vpool + (uint64_t)vtab[page] * QW_KV_PGE4
                       + (uint64_t)slot * (QW_KV_ROW / 2u)
                       + (uint64_t)kvh * (QW_HEAD_DIM / 2u) + (d0 >> 1);
    #pragma unroll
    for (uint i = 0; i < 4u; i++) {
        const uint kq0 = (uint)clamp(round((float(ks[2u*i])   - kmn) * kinv), 0.0f, 15.0f);
        const uint kq1 = (uint)clamp(round((float(ks[2u*i+1]) - kmn) * kinv), 0.0f, 15.0f);
        const uint vq0 = (uint)clamp(round((float(vs[2u*i])   - vmn) * vinv), 0.0f, 15.0f);
        const uint vq1 = (uint)clamp(round((float(vs[2u*i+1]) - vmn) * vinv), 0.0f, 15.0f);
        kd[i] = (uchar)(kq0 | (kq1 << 4));
        vd[i] = (uchar)(vq0 | (vq1 << 4));
    }
    if ((d0 & 127u) == 0u) {   // group's dim-0 lane writes {sc, mn}
        const uint grp = kvh * 2u + (d0 >> 7);
        device half* kh = (device half*)((device char*)kpool + (uint64_t)ktab[page] * QW_KV_PGE4
                          + 131072ull + (uint64_t)slot * 32u + (uint64_t)grp * 4u);
        device half* vh = (device half*)((device char*)vpool + (uint64_t)vtab[page] * QW_KV_PGE4
                          + 131072ull + (uint64_t)slot * 32u + (uint64_t)grp * 4u);
        kh[0] = ksc16; kh[1] = kmn16;
        vh[0] = vsc16; vh[1] = vmn16;
    }
}

kernel void qwg_zero_pad_attn_f16(device half*   q  [[buffer(0)]],
                                  device half*   k  [[buffer(1)]],
                                  device half*   v  [[buffer(2)]],
                                  constant uint& t  [[buffer(3)]],
                                  constant uint& sp [[buffer(4)]],
                                  constant uint& qProjOut   [[buffer(5)]],
                                  constant uint& kvChannels [[buffer(6)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint row = t + gid.y;
    if (row >= sp) return;
    if (col < qProjOut) {
        q[(size_t)row * qProjOut + col] = 0;
    }
    if (col < kvChannels) {
        k[(size_t)row * kvChannels + col] = 0;
        v[(size_t)row * kvChannels + col] = 0;
    }
}

#if !defined(QW_DISABLE_NAX)
kernel void qwg_moe_nax_gemm(device const Block_HQ4_G32* w [[buffer(0)]],
                             device const half* x [[buffer(1)]],
                             device half* y [[buffer(2)]],
                             device const int32_t* expertOff [[buffer(3)]],
                             device const int32_t* tileE [[buffer(4)]],
                             device const int32_t* tileR [[buffer(5)]],
                             device const int32_t* tileCount [[buffer(6)]],
                             constant QWGemmGroupedParams& p [[buffer(7)]],
                             constant uint& tileBlock [[buffer(8)]],
                             device const int32_t* sortedTok [[buffer(9)]],
                             constant uint& hasGather [[buffer(10)]],
                             device const Block_HQ4_G32* coldW [[buffer(11)]],
                             uint3 tg [[threadgroup_position_in_grid]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint simd_gid [[simdgroup_index_in_threadgroup]],
                             uint simd_lid [[thread_index_in_simdgroup]]) {
    qwg_moe_nax_gemm_body((device const char*)w, x, y, expertOff, tileE, tileR, tileCount, p, tileBlock,
                          sortedTok, hasGather, (device const char*)coldW, tg, lid, simd_gid, simd_lid);
}

// Dense NAX/MPP GEMM with single-pass f16 activations (xh) and fp32 accumulate/output (or f16 when M bit 31 is set).
kernel void qwg_gemm_q4g_nax_f32(device const Block_HQ4_G32* blocks [[buffer(0)]],
                                 device const half* xh [[buffer(1)]],
                                 device const half* xl [[buffer(2)]],
                                 device float* y32 [[buffer(3)]],
                                 constant uint& M [[buffer(4)]],
                                 constant uint& N [[buffer(5)]],
                                 constant uint& K [[buffer(6)]],
                                 uint3 tg [[threadgroup_position_in_grid]],
                                 uint simd_gid [[simdgroup_index_in_threadgroup]]) {
    (void)xl;
    qwg_gemm_nax_f32_dispatch((const device char*)blocks, xh, xl, y32, M, N, K, tg, simd_gid);
}

kernel void qwg_moe_nax_split_f32(device const Block_HQ4_G32* w [[buffer(0)]],
                                  device const half* xh [[buffer(1)]],
                                  device const half* xl [[buffer(2)]],
                                  device float* y32 [[buffer(3)]],
                                  device const int32_t* expertOff [[buffer(4)]],
                                  device const int32_t* tileE [[buffer(5)]],
                                  device const int32_t* tileR [[buffer(6)]],
                                  device const int32_t* tileCount [[buffer(7)]],
                                  constant QWGemmGroupedParams& p [[buffer(8)]],
                                  constant uint& tileBlock [[buffer(9)]],
                                  device const Block_HQ4_G32* coldW [[buffer(10)]],
                                  uint3 tg [[threadgroup_position_in_grid]],
                                  uint simd_gid [[simdgroup_index_in_threadgroup]]) {
    (void)xl;
    qwg_moe_nax_split_f32_body<0, 0, false, false>(
        (device const char*)w, xh, xl, y32, expertOff, tileE, tileR, tileCount,
        p, tileBlock, (device const char*)coldW, tg, simd_gid);
}
#endif // !defined(QW_DISABLE_NAX)
