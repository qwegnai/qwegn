// qwrouter_nax.metal — prefill MoE router logits on NAX/MPP units at f32 precision.
// Y[t, e] = sum_k X[t, k] * W[e, k]: native FP16 NAX for f16 X (1x BF16 / 2x F32 W);
// sequential BF16 splits for f32 X. Extra TG column computes sigmoid(X[t] . shGateW).
#if !defined(QW_DISABLE_NAX)
#if !defined(__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__) || __ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ < 260200
#error "qwrouter_nax.metal requires -mmacosx-version-min=26.2 or newer (Apple MPP NAX ABI)."
#endif
#include <metal_stdlib>
#include "include/qw_gemm_nax.metalh"
using namespace metal;

template <bool W_F32, bool ALIGNED>
inline void qw_router_nax_tile_f16(const device void* W,
                                   const device half* X,
                                   device float*      Y,
                                   int                Ni,
                                   int                Ki,
                                   int                tileRow,
                                   int                tileCol,
                                   int                sgp_sm,
                                   int                sgp_sn,
                                   short2             sc) {
    constexpr short TM = 2, TN = 2;
    using WT = typename metal::conditional<W_F32, float, ushort>::type;
    const device WT* wr[TN * 2];
    bool wok[TN * 2];
    (void)wok;
    STEEL_PRAGMA_UNROLL
    for (short n = 0; n < TN; ++n) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; ++i) {
            const int c = tileCol + int(n) * 16 + int(sc.y) + int(i) * 8;
            if constexpr (ALIGNED) {
                wok[n * 2 + i] = true;
                wr[n * 2 + i] = (const device WT*)W + size_t(c) * Ki + int(sc.x);
            } else {
                wok[n * 2 + i] = c < Ni;
                wr[n * 2 + i] = (const device WT*)W + size_t(min(c, Ni - 1)) * Ki + int(sc.x);
            }
        }
    }

    NAXTile<float, TM, TN> D;
    D.clear();
    const device half* xTile = X + size_t(tileRow) * Ki;
    STEEL_PRAGMA_NO_UNROLL
    for (int k = 0; k < Ki; k += 32) {
        NAXTile<half, TM, 2> A;
        if constexpr (ALIGNED) {
            A.load(xTile + k, Ki);
        } else {
            A.load_rows(xTile + k, Ki, short(sgp_sm));
        }
        NAXTile<half, TN, 2> Bh;
        if constexpr (W_F32) {
            NAXTile<half, TN, 2> Bl;
            const_for_loop<0, TN, 1>([&](auto ni) {
                constexpr short n = ni.value;
                thread auto& fh0 = Bh.template frag_at<n, 0>();
                thread auto& fh1 = Bh.template frag_at<n, 1>();
                thread auto& fl0 = Bl.template frag_at<n, 0>();
                thread auto& fl1 = Bl.template frag_at<n, 1>();
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; ++i) {
                    bool wValid = true;
                    if constexpr (!ALIGNED) wValid = wok[n * 2 + i];
                    const float4 w0 = wValid ? *(const device float4*)(wr[n * 2 + i] + k) : float4(0.0f);
                    const float4 w1 = wValid ? *(const device float4*)(wr[n * 2 + i] + k + 16) : float4(0.0f);
                    const half4 wh0 = half4(w0);
                    const half4 wl0 = half4(w0 - float4(wh0));
                    const half4 wh1 = half4(w1);
                    const half4 wl1 = half4(w1 - float4(wh1));
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; ++j) {
                        fh0[i * 4 + j] = wh0[j]; fl0[i * 4 + j] = wl0[j];
                        fh1[i * 4 + j] = wh1[j]; fl1[i * 4 + j] = wl1[j];
                    }
                }
            });
            tile_matmad_nax(D, A, metal::bool_constant<false>{}, Bl, metal::bool_constant<true>{});
            tile_matmad_nax(D, A, metal::bool_constant<false>{}, Bh, metal::bool_constant<true>{});
        } else {
            const_for_loop<0, TN, 1>([&](auto ni) {
                constexpr short n = ni.value;
                thread auto& fh0 = Bh.template frag_at<n, 0>();
                thread auto& fh1 = Bh.template frag_at<n, 1>();
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; ++i) {
                    bool wValid = true;
                    if constexpr (!ALIGNED) wValid = wok[n * 2 + i];
                    const ushort4 r0 = wValid ? *(const device ushort4*)(wr[n * 2 + i] + k) : ushort4(0u);
                    const ushort4 r1 = wValid ? *(const device ushort4*)(wr[n * 2 + i] + k + 16) : ushort4(0u);
                    const half4 wh0 = half4(clamp(float4(as_type<bfloat4>(r0)), -65504.0f, 65504.0f));
                    const half4 wh1 = half4(clamp(float4(as_type<bfloat4>(r1)), -65504.0f, 65504.0f));
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; ++j) {
                        fh0[i * 4 + j] = wh0[j];
                        fh1[i * 4 + j] = wh1[j];
                    }
                }
            });
            tile_matmad_nax(D, A, metal::bool_constant<false>{}, Bh, metal::bool_constant<true>{});
        }
    }
    device float* yp = Y + size_t(tileRow) * Ni + tileCol;
    if constexpr (ALIGNED) {
        (void)sgp_sm; (void)sgp_sn;
        D.store(yp, Ni);
    } else {
        D.store_safe(yp, Ni, short2(short(sgp_sn), short(sgp_sm)));
    }
}

template <bool W_F32, bool ALIGNED>
inline void qw_router_nax_tile_f32(const device void*  W,
                                   const device float* X,
                                   device float*       Y,
                                   int                 Mi,
                                   int                 Ni,
                                   int                 Ki,
                                   int                 tileRow,
                                   int                 tileCol,
                                   int                 sgp_sm,
                                   int                 sgp_sn,
                                   short2              sc) {
    constexpr short TM = 2, TN = 2;
    using WT = typename metal::conditional<W_F32, float, ushort>::type;
    const device float* xr[TM * 2];
    bool xok[TM * 2];
    (void)xok;
    STEEL_PRAGMA_UNROLL
    for (short m = 0; m < TM; ++m) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; ++i) {
            const int r = tileRow + int(m) * 16 + int(sc.y) + int(i) * 8;
            if constexpr (ALIGNED) {
                xok[m * 2 + i] = true;
                xr[m * 2 + i] = X + size_t(r) * Ki + int(sc.x);
            } else {
                xok[m * 2 + i] = r < Mi;
                xr[m * 2 + i] = X + size_t(min(r, Mi - 1)) * Ki + int(sc.x);
            }
        }
    }
    const device WT* wr[TN * 2];
    bool wok[TN * 2];
    (void)wok;
    STEEL_PRAGMA_UNROLL
    for (short n = 0; n < TN; ++n) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; ++i) {
            const int c = tileCol + int(n) * 16 + int(sc.y) + int(i) * 8;
            if constexpr (ALIGNED) {
                wok[n * 2 + i] = true;
                wr[n * 2 + i] = (const device WT*)W + size_t(c) * Ki + int(sc.x);
            } else {
                wok[n * 2 + i] = c < Ni;
                wr[n * 2 + i] = (const device WT*)W + size_t(min(c, Ni - 1)) * Ki + int(sc.x);
            }
        }
    }

    NAXTile<float, TM, TN> D;
    D.clear();
    STEEL_PRAGMA_NO_UNROLL
    for (int k = 0; k < Ki; k += 32) {
        NAXTile<bfloat, TM, 2> Ah, Am, Al;
        NAXTile<bfloat, TN, 2> Bh, Bl;
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
                    bool xValid = true;
                    if constexpr (!ALIGNED) xValid = xok[m * 2 + i];
                    const float4 v = xValid ? *(const device float4*)(xr[m * 2 + i] + k + kk * 16) : float4(0.0f);
                    const bfloat4 h = bfloat4(v);
                    const float4 r1 = v - float4(h);
                    const bfloat4 md = bfloat4(r1);
                    const bfloat4 lo = bfloat4(r1 - float4(md));
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; ++j) {
                        if (kk == 0) {
                            fh[i * 4 + j] = h[j]; fm[i * 4 + j] = md[j]; fl[i * 4 + j] = lo[j];
                        } else {
                            gh[i * 4 + j] = h[j]; gm[i * 4 + j] = md[j]; gl[i * 4 + j] = lo[j];
                        }
                    }
                }
            }
        });
        const_for_loop<0, TN, 1>([&](auto ni) {
            constexpr short n = ni.value;
            thread auto& fh0 = Bh.template frag_at<n, 0>();
            thread auto& fh1 = Bh.template frag_at<n, 1>();
            thread auto& fl0 = Bl.template frag_at<n, 0>();
            thread auto& fl1 = Bl.template frag_at<n, 1>();
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; ++i) {
                bool wValid = true;
                if constexpr (!ALIGNED) wValid = wok[n * 2 + i];
                if constexpr (W_F32) {
                    const float4 w0 = wValid ? *(const device float4*)(wr[n * 2 + i] + k) : float4(0.0f);
                    const float4 w1 = wValid ? *(const device float4*)(wr[n * 2 + i] + k + 16) : float4(0.0f);
                    const bfloat4 wh0 = bfloat4(w0);
                    const bfloat4 wl0 = bfloat4(w0 - float4(wh0));
                    const bfloat4 wh1 = bfloat4(w1);
                    const bfloat4 wl1 = bfloat4(w1 - float4(wh1));
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; ++j) {
                        fh0[i * 4 + j] = wh0[j]; fl0[i * 4 + j] = wl0[j];
                        fh1[i * 4 + j] = wh1[j]; fl1[i * 4 + j] = wl1[j];
                    }
                } else {
                    const bfloat4 w0 = wValid ? as_type<bfloat4>(*(const device ushort4*)(wr[n * 2 + i] + k)) : bfloat4(0.0bf);
                    const bfloat4 w1 = wValid ? as_type<bfloat4>(*(const device ushort4*)(wr[n * 2 + i] + k + 16)) : bfloat4(0.0bf);
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; ++j) { fh0[i * 4 + j] = w0[j]; fh1[i * 4 + j] = w1[j]; }
                }
            }
        });
        tile_matmad_nax(D, Al, metal::bool_constant<false>{}, Bh, metal::bool_constant<true>{});
        tile_matmad_nax(D, Am, metal::bool_constant<false>{}, Bh, metal::bool_constant<true>{});
        if constexpr (W_F32) {
            tile_matmad_nax(D, Ah, metal::bool_constant<false>{}, Bl, metal::bool_constant<true>{});
        }
        tile_matmad_nax(D, Ah, metal::bool_constant<false>{}, Bh, metal::bool_constant<true>{});
    }
    device float* yp = Y + size_t(tileRow) * Ni + tileCol;
    if constexpr (ALIGNED) {
        (void)sgp_sm; (void)sgp_sn;
        D.store(yp, Ni);
    } else {
        D.store_safe(yp, Ni, short2(short(sgp_sn), short(sgp_sm)));
    }
}

template <bool W_F32, bool X_F16>
inline void qw_router_nax_impl(const device void*  W,
                               const device void*  X,
                               device float*       Y,
                               uint                M,
                               uint                N,
                               uint                K,
                               device const float* shGateW,
                               device float*       shGateOut,
                               bool                hasShGate,
                               uint3               tid,
                               uint                simd_gid,
                               uint                simd_lid) {
    constexpr short BM = 64, BN = 64, WN = 2, SM = 32, SN = 32;
    const int Mi = int(M), Ni = int(N), Ki = int(K);
    const int y_row = int(tid.y) * BM;
    if (int(tid.x) * BN >= Ni) {
        if (!hasShGate) return;
        for (int r = y_row + int(simd_gid) * 16; r < min(Mi, y_row + int(simd_gid) * 16 + 16); ++r) {
            const device float4* g4 = (const device float4*)shGateW;
            float acc = 0.0f;
            if constexpr (X_F16) {
                const device half4* x4 = (const device half4*)((const device half*)X + size_t(r) * Ki);
                for (int k4 = int(simd_lid); k4 < Ki / 4; k4 += 32) acc += dot(float4(x4[k4]), g4[k4]);
            } else {
                const device float4* x4 = (const device float4*)((const device float*)X + size_t(r) * Ki);
                for (int k4 = int(simd_lid); k4 < Ki / 4; k4 += 32) acc += dot(x4[k4], g4[k4]);
            }
            acc = simd_sum(acc);
            if (simd_lid == 0u) shGateOut[r] = 1.0f / (1.0f + exp(-acc));
        }
        return;
    }
    const int y_col = int(tid.x) * BN;
    const short tm = SM * short(simd_gid / WN);
    const short tn = SN * short(simd_gid % WN);
    const int sgp_sm = min(int(SM), Mi - (y_row + int(tm)));
    const int sgp_sn = min(int(SN), Ni - (y_col + int(tn)));
    if (sgp_sm <= 0 || sgp_sn <= 0) return;

    const short2 sc = BaseNAXFrag::get_coord();
    dispatch_bool(sgp_sm == SM && sgp_sn == SN, [&](auto kAlignedTile) {
        constexpr bool ALIGNED = decltype(kAlignedTile)::value;
        if constexpr (X_F16) {
            qw_router_nax_tile_f16<W_F32, ALIGNED>(
                W, (const device half*)X, Y, Ni, Ki, y_row + int(tm), y_col + int(tn), sgp_sm, sgp_sn, sc);
        } else {
            qw_router_nax_tile_f32<W_F32, ALIGNED>(
                W, (const device float*)X, Y, Mi, Ni, Ki, y_row + int(tm), y_col + int(tn), sgp_sm, sgp_sn, sc);
        }
    });
}

kernel void qwg_router_nax(device const ushort* W [[buffer(0)]],
                           device const float*  X [[buffer(1)]],
                           device float*        Y [[buffer(2)]],
                           constant uint&       M [[buffer(3)]],
                           constant uint&       N [[buffer(4)]],
                           constant uint&       K [[buffer(5)]],
                           device const float*  shGateW [[buffer(7)]],
                           device float*        shGateOut [[buffer(8)]],
                           constant uint&       flags [[buffer(9)]],
                           uint3 tid [[threadgroup_position_in_grid]],
                           uint simd_gid [[simdgroup_index_in_threadgroup]],
                           uint simd_lid [[thread_index_in_simdgroup]]) {
    const bool hasShGate = (flags & 1u) != 0u;
    if ((flags & 2u) != 0u) {
        qw_router_nax_impl<false, true>(W, X, Y, M, N, K, shGateW, shGateOut, hasShGate, tid, simd_gid, simd_lid);
    } else {
        qw_router_nax_impl<false, false>(W, X, Y, M, N, K, shGateW, shGateOut, hasShGate, tid, simd_gid, simd_lid);
    }
}

kernel void qwg_router_nax_f32(device const float* W [[buffer(0)]],
                               device const float* X [[buffer(1)]],
                               device float*       Y [[buffer(2)]],
                               constant uint&      M [[buffer(3)]],
                               constant uint&      N [[buffer(4)]],
                               constant uint&      K [[buffer(5)]],
                               device const float* shGateW [[buffer(7)]],
                               device float*       shGateOut [[buffer(8)]],
                               constant uint&      flags [[buffer(9)]],
                               uint3 tid [[threadgroup_position_in_grid]],
                               uint simd_gid [[simdgroup_index_in_threadgroup]],
                               uint simd_lid [[thread_index_in_simdgroup]]) {
    const bool hasShGate = (flags & 1u) != 0u;
    if ((flags & 2u) != 0u) {
        qw_router_nax_impl<true, true>(W, X, Y, M, N, K, shGateW, shGateOut, hasShGate, tid, simd_gid, simd_lid);
    } else {
        qw_router_nax_impl<true, false>(W, X, Y, M, N, K, shGateW, shGateOut, hasShGate, tid, simd_gid, simd_lid);
    }
}
#endif // !defined(QW_DISABLE_NAX)
