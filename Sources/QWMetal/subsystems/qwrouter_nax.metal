// qwrouter_nax.metal — prefill MoE router logits on the NAX/MPP units at f32-class precision.
// Y[t, e] = sum_k X[t, k] * W[e, k] with bf16 W (exact) and f32 X split into three bf16 terms
// (hi + mid + lo carry 24 mantissa bits), all accumulated in f32. One extra column of
// threadgroups computes the shared-expert gate sigmoid(X[t] . shGateW).
#include <metal_stdlib>
#include "include/qw_gemm_nax.metalh"
using namespace metal;

kernel void qwg_router_nax(device const ushort* W [[buffer(0)]],
                           device const float*  X [[buffer(1)]],
                           device float*        Y [[buffer(2)]],
                           constant uint&       M [[buffer(3)]],
                           constant uint&       N [[buffer(4)]],
                           constant uint&       K [[buffer(5)]],
                           device const float*  shGateW [[buffer(7)]],
                           device float*        shGateOut [[buffer(8)]],
                           constant uint&       hasShGate [[buffer(9)]],
                           uint3 tid [[threadgroup_position_in_grid]],
                           uint simd_gid [[simdgroup_index_in_threadgroup]],
                           uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr short BM = 64, BN = 64, WN = 2, SM = 32, SN = 32, TM = 2, TN = 2;
    const int Mi = int(M), Ni = int(N), Ki = int(K);
    const int y_row = int(tid.y) * BM;
    if (int(tid.x) * BN >= Ni) {
        if (hasShGate == 0u) return;
        for (int r = y_row + int(simd_gid) * 16; r < min(Mi, y_row + int(simd_gid) * 16 + 16); ++r) {
            const device float4* x4 = (const device float4*)(X + size_t(r) * Ki);
            const device float4* g4 = (const device float4*)shGateW;
            float acc = 0.0f;
            for (int k4 = int(simd_lid); k4 < Ki / 4; k4 += 32) acc += dot(x4[k4], g4[k4]);
            acc = simd_sum(acc);
            if (simd_lid == 0u) shGateOut[r] = 1.0f / (1.0f + exp(-acc));
        }
        return;
    }
    const int y_col = int(tid.x) * BN;
    const short tm = SM * short(simd_gid / WN);
    const short tn = SN * short(simd_gid % WN);
    const int sgp_sm = min(int(SM), Mi - (y_row + int(tm)));
    if (sgp_sm <= 0) return;

    const short2 sc = BaseNAXFrag::get_coord();
    const device float* xr[TM * 2];
    bool xok[TM * 2];
    STEEL_PRAGMA_UNROLL
    for (short m = 0; m < TM; ++m) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; ++i) {
            const int r = y_row + int(tm) + int(m) * 16 + int(sc.y) + int(i) * 8;
            xok[m * 2 + i] = r < Mi;
            xr[m * 2 + i] = X + size_t(min(r, Mi - 1)) * Ki + int(sc.x);
        }
    }
    const device ushort* wr[TN * 2];
    STEEL_PRAGMA_UNROLL
    for (short n = 0; n < TN; ++n) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; ++i) {
            const int c = y_col + int(tn) + int(n) * 16 + int(sc.y) + int(i) * 8;
            wr[n * 2 + i] = W + size_t(c) * Ki + int(sc.x);
        }
    }

    NAXTile<float, TM, TN> D;
    D.clear();
    STEEL_PRAGMA_NO_UNROLL
    for (int k = 0; k < Ki; k += 32) {
        NAXTile<bfloat, TM, 2> Ah, Am, Al;
        NAXTile<bfloat, TN, 2> B;
        const_for_loop<0, TM, 1>([&](auto mi) {
            constexpr short m = mi.value;
            STEEL_PRAGMA_UNROLL
            for (short kk = 0; kk < 2; ++kk) {
                thread auto& fh = Ah.template frag_at<m, 0>();
                thread auto& fm = Am.template frag_at<m, 0>();
                thread auto& fl = Al.template frag_at<m, 0>();
                thread auto& gh = Ah.template frag_at<m, 1>();
                thread auto& gm = Am.template frag_at<m, 1>();
                thread auto& gl = Al.template frag_at<m, 1>();
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; ++i) {
                    const float4 v = xok[m * 2 + i] ? *(const device float4*)(xr[m * 2 + i] + k + kk * 16) : float4(0.0f);
                    const bfloat4 h = bfloat4(v);
                    const float4 r1 = v - float4(h);
                    const bfloat4 md = bfloat4(r1);
                    const bfloat4 lo = bfloat4(r1 - float4(md));
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; ++j) {
                        if (kk == 0) { fh[i * 4 + j] = h[j]; fm[i * 4 + j] = md[j]; fl[i * 4 + j] = lo[j]; }
                        else         { gh[i * 4 + j] = h[j]; gm[i * 4 + j] = md[j]; gl[i * 4 + j] = lo[j]; }
                    }
                }
            }
        });
        const_for_loop<0, TN, 1>([&](auto ni) {
            constexpr short n = ni.value;
            thread auto& f0 = B.template frag_at<n, 0>();
            thread auto& f1 = B.template frag_at<n, 1>();
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; ++i) {
                const bfloat4 w0 = as_type<bfloat4>(*(const device ushort4*)(wr[n * 2 + i] + k));
                const bfloat4 w1 = as_type<bfloat4>(*(const device ushort4*)(wr[n * 2 + i] + k + 16));
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; ++j) { f0[i * 4 + j] = w0[j]; f1[i * 4 + j] = w1[j]; }
            }
        });
        tile_matmad_nax(D, Al, metal::bool_constant<false>{}, B, metal::bool_constant<true>{});
        tile_matmad_nax(D, Am, metal::bool_constant<false>{}, B, metal::bool_constant<true>{});
        tile_matmad_nax(D, Ah, metal::bool_constant<false>{}, B, metal::bool_constant<true>{});
    }
    device float* yp = Y + size_t(y_row + int(tm)) * Ni + y_col + int(tn);
    if (sgp_sm == SM) D.store(yp, Ni);
    else D.store_safe(yp, Ni, short2(SN, short(sgp_sm)));
}
