// qwgemm_shvq_batched.metal — NAX/MPP prefill GEMM, GDN in-projection GEMM,
// and grouped MoE NAX twins for SHVQ{2,3,4}_G{32,64,128,256}.

#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_moe_tile.metalh"
#include "include/qw_gemm_nax.metalh"

struct QWGemmSHVQGroupedParams {
    uint experts;
    uint n;
    uint k;
    uint blocksPerRow;
    uint rowTiles;
    uint colTiles;
};

template <const int BM, const int WM, const int WN, const int BITS, const int GROUP,
          const bool splitLo = true, typename OutT = float>
static inline void qmm_t_nax_split_shvq_impl(
    const device char* w,
    const device half* xh,
    const device half* xl,
    device OutT* y,
    int K, int N, int M,
    uint3 tid,
    uint simd_gid) {
    constexpr int BN = 64;
    constexpr short SM = BM / WM;
    constexpr short SN = BN / WN;
    constexpr short TM = SM / 16;
    constexpr short TN = SN / 16;
    constexpr short TK = 2;
    const int K_g = K / 32;
    const int y_row = int(tid.y) * BM;
    const int y_col = int(tid.x) * BN;
    const short tm = SM * short(simd_gid / WN);
    const short tn = SN * short(simd_gid % WN);
    const short sgp_sm = short(min(int(SM), M - (y_row + int(tm))));
    const short sgp_sn = short(min(int(SN), N - (y_col + int(tn))));
    const short tgp_bn = short(min(BN, N - y_col));
    if (sgp_sm <= 0 || sgp_sn <= 0) return;

    xh += y_row * static_cast<int64_t>(K);
    if constexpr (splitLo) xl += y_row * static_cast<int64_t>(K);
    y += y_row * static_cast<int64_t>(N) + y_col;

    const short2 sc = BaseNAXFrag::get_coord();
    constexpr uint groupBytes = 2u + (uint(GROUP * BITS) >> 3u);
    constexpr uint slicesPerGroup = uint(GROUP) >> 5u;
    const uint rowBytes = uint(K / GROUP) * groupBytes;
    const device char* w_tile = w + static_cast<int64_t>(y_col) * rowBytes;

    NAXTile<float, TM, TN> Dtile;
    Dtile.clear();
    xh += tm * K;
    if constexpr (splitLo) xl += tm * K;

    dispatch_bool(sgp_sm == SM, [&](auto kAlignedM) {
        dispatch_bool(tgp_bn == BN, [&](auto kAlignedN) {
            STEEL_PRAGMA_NO_UNROLL
            for (int g = 0; g < K_g; ++g) {
                NAXTile<half, TM, TK> Ah;
                NAXTile<half, TM, TK> Al;
                NAXTile<half, TN, TK> Btile;
                if constexpr (kAlignedM.value) {
                    Ah.load(xh, K);
                    if constexpr (splitLo) Al.load(xl, K);
                } else {
                    Ah.load_safe(xh, K, short2(32, sgp_sm));
                    if constexpr (splitLo) Al.load_safe(xl, K, short2(32, sgp_sm));
                }
                const uint groupIdx = uint(g) / slicesPerGroup;
                const uint colInGroup = ((uint(g) & (slicesPerGroup - 1u)) << 5u) + uint(sc.x);
                const device char* groupBase =
                    w_tile + static_cast<int64_t>(groupIdx) * groupBytes;

                const_for_loop<0, TN, 1>([&](auto idx_row) {
                    thread auto& f0 = Btile.template frag_at<idx_row.value, 0>();
                    thread auto& f1 = Btile.template frag_at<idx_row.value, 1>();
                    STEEL_PRAGMA_UNROLL
                    for (short i = 0; i < 2; ++i) {
                        const int r_tile = int(tn) + idx_row.value * 16 + int(sc.y) + int(i) * 8;
                        if (kAlignedN.value || r_tile < int(tgp_bn)) {
                            const device char* gp = groupBase + static_cast<int64_t>(r_tile) * rowBytes;
                            half4 lo, hi;
                            qwg_shvq_frag8<BITS, BITS == 4>(gp, colInGroup, lo, hi);
                            f0[i * 4 + 0] = lo.x; f0[i * 4 + 1] = lo.y;
                            f0[i * 4 + 2] = lo.z; f0[i * 4 + 3] = lo.w;
                            f1[i * 4 + 0] = hi.x; f1[i * 4 + 1] = hi.y;
                            f1[i * 4 + 2] = hi.z; f1[i * 4 + 3] = hi.w;
                        } else {
                            f0[i * 4 + 0] = half(0); f0[i * 4 + 1] = half(0);
                            f0[i * 4 + 2] = half(0); f0[i * 4 + 3] = half(0);
                            f1[i * 4 + 0] = half(0); f1[i * 4 + 1] = half(0);
                            f1[i * 4 + 2] = half(0); f1[i * 4 + 3] = half(0);
                        }
                    }
                });
                tile_matmad_nax(Dtile, Ah, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
                if constexpr (splitLo) {
                    tile_matmad_nax(Dtile, Al, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
                }
                xh += 32;
                if constexpr (splitLo) xl += 32;
            }
            if constexpr (kAlignedM.value && kAlignedN.value) {
                Dtile.store(y + tm * N + tn, N);
            } else {
                Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
            }
        });
    });
}

template <int BITS, int GROUP>
static inline void qwg_moe_nax_gemm_shvq_body(device const char* w,
                                              device const half* x,
                                              device half* y,
                                              device const int32_t* expertOff,
                                              device const int32_t* tileE,
                                              device const int32_t* tileR,
                                              device const int32_t* tileCount,
                                              constant QWGemmSHVQGroupedParams& p,
                                              uint tileBlock,
                                              device const int32_t* sortedTok,
                                              uint hasGather,
                                              device const char* coldW,
                                              uint3 tg, uint lid, uint simd_gid, uint simd_lid) {
    const uint t = tg.y + ((tileBlock >> 8) & 0x7FFFFFu);
    if (int(t) >= tileCount[0]) return;
    const uint e = uint(tileE[t]);
    const int rowBase = int(expertOff[e]);
    const int rowCount = int(expertOff[e + 1]) - rowBase;
    constexpr size_t groupBytes = 2u + (uint(GROUP * BITS) >> 3u);
    const size_t rowBytes = (size_t(p.k) / GROUP) * groupBytes;
    const uint residentE = p.experts >> 16u;
    const bool isCold = (residentE > 0u) && (e >= residentE);
    if (((tileBlock & 0x40u) != 0u && isCold) || ((tileBlock & 0x80u) != 0u && !isCold)) return;
    const uint wExp = isCold ? (e - residentE) : e;
    device const char* we = (isCold ? coldW : w) + size_t(wExp) * size_t(p.n) * rowBytes;
    device half* ye = y + size_t(rowBase) * size_t(p.n);
    const uint3 tid = uint3(tg.x, uint(tileR[t]), 0);
    const int bm = qw_moe_tile_block(rowCount, int(tileBlock & 0x3Fu));
    if (hasGather != 0u) {
        device const int32_t* st = sortedTok + rowBase;
        if (bm == 16) {
            qmm_t_nax_tgp_dual_impl<half, half, true, 16, 64, 64, 1, 4, false, false, true, BITS, GROUP, true>(
                we, x, ye, nullptr, nullptr, int(p.k), int(p.n), rowCount, tid, lid, simd_gid, simd_lid, nullptr, st);
        } else if (bm == 32) {
            qmm_t_nax_tgp_dual_impl<half, half, true, 32, 64, 64, 2, 2, false, false, true, BITS, GROUP, true>(
                we, x, ye, nullptr, nullptr, int(p.k), int(p.n), rowCount, tid, lid, simd_gid, simd_lid, nullptr, st);
        } else {
            qmm_t_nax_tgp_dual_impl<half, half, true, 64, 64, 64, 2, 2, false, false, true, BITS, GROUP, true>(
                we, x, ye, nullptr, nullptr, int(p.k), int(p.n), rowCount, tid, lid, simd_gid, simd_lid, nullptr, st);
        }
    } else {
        device const half* xe = x + size_t(rowBase) * size_t(p.k);
        if (bm == 16) {
            qmm_t_nax_tgp_dual_impl<half, half, true, 16, 64, 64, 1, 4, false, false, false, BITS, GROUP, true>(
                we, xe, ye, nullptr, nullptr, int(p.k), int(p.n), rowCount, tid, lid, simd_gid, simd_lid);
        } else if (bm == 32) {
            qmm_t_nax_tgp_dual_impl<half, half, true, 32, 64, 64, 2, 2, false, false, false, BITS, GROUP, true>(
                we, xe, ye, nullptr, nullptr, int(p.k), int(p.n), rowCount, tid, lid, simd_gid, simd_lid);
        } else {
            qmm_t_nax_tgp_dual_impl<half, half, true, 64, 64, 64, 2, 2, false, false, false, BITS, GROUP, true>(
                we, xe, ye, nullptr, nullptr, int(p.k), int(p.n), rowCount, tid, lid, simd_gid, simd_lid);
        }
    }
}

template <int BITS, int GROUP>
static inline void qwg_moe_nax_split_shvq_f32_body(device const char* w,
                                                   device const half* xh,
                                                   device const half* xl,
                                                   device float* y32,
                                                   device const int32_t* expertOff,
                                                   device const int32_t* tileE,
                                                   device const int32_t* tileR,
                                                   device const int32_t* tileCount,
                                                   constant QWGemmSHVQGroupedParams& p,
                                                   uint tileBlock,
                                                   device const char* coldW,
                                                   uint3 tg, uint simd_gid) {
    const uint t = tg.y + ((tileBlock >> 8) & 0x7FFFFFu);
    if (int(t) >= tileCount[0]) return;
    const uint e = uint(tileE[t]);
    const int rowBase = int(expertOff[e]);
    const int rowCount = int(expertOff[e + 1]) - rowBase;
    constexpr size_t groupBytes = 2u + (uint(GROUP * BITS) >> 3u);
    const size_t rowBytes = (size_t(p.k) / GROUP) * groupBytes;
    const uint residentE = p.experts >> 16u;
    const bool isCold = (residentE > 0u) && (e >= residentE);
    if (((tileBlock & 0x40u) != 0u && isCold) || ((tileBlock & 0x80u) != 0u && !isCold)) return;
    const uint wExp = isCold ? (e - residentE) : e;
    device const char* we = (isCold ? coldW : w) + size_t(wExp) * size_t(p.n) * rowBytes;
    device const half* xhe = xh + size_t(rowBase) * size_t(p.k);
    device const half* xle = xl + size_t(rowBase) * size_t(p.k);
    const uint3 tid = uint3(tg.x, uint(tileR[t]), 0);
    const int bm = qw_moe_tile_block(rowCount, int(tileBlock & 0x3Fu));
    if ((tileBlock & 0x80000000u) != 0u) {
        device half* ye16 = ((device half*)y32) + size_t(rowBase) * size_t(p.n);
        if (bm == 16) qmm_t_nax_split_shvq_impl<16, 1, 4, BITS, GROUP, false, half>(we, xhe, xle, ye16, int(p.k), int(p.n), rowCount, tid, simd_gid);
        else if (bm == 32) qmm_t_nax_split_shvq_impl<32, 2, 2, BITS, GROUP, false, half>(we, xhe, xle, ye16, int(p.k), int(p.n), rowCount, tid, simd_gid);
        else qmm_t_nax_split_shvq_impl<64, 2, 2, BITS, GROUP, false, half>(we, xhe, xle, ye16, int(p.k), int(p.n), rowCount, tid, simd_gid);
    } else {
        device float* ye = y32 + size_t(rowBase) * size_t(p.n);
        if (bm == 16) qmm_t_nax_split_shvq_impl<16, 1, 4, BITS, GROUP, false, float>(we, xhe, xle, ye, int(p.k), int(p.n), rowCount, tid, simd_gid);
        else if (bm == 32) qmm_t_nax_split_shvq_impl<32, 2, 2, BITS, GROUP, false, float>(we, xhe, xle, ye, int(p.k), int(p.n), rowCount, tid, simd_gid);
        else qmm_t_nax_split_shvq_impl<64, 2, 2, BITS, GROUP, false, float>(we, xhe, xle, ye, int(p.k), int(p.n), rowCount, tid, simd_gid);
    }
}

#define QW_SHVQ_BATCHED_KERNELS(BITS, GROUP)                                           \
    kernel void qwg_gemm_shvq##BITS##_g##GROUP##_h16(                                   \
        device const char* blocks [[buffer(0)]],                                       \
        device const half* x      [[buffer(1)]],                                       \
        device half*       y16    [[buffer(2)]],                                       \
        device float*      y32    [[buffer(3)]],                                       \
        constant uint& M [[buffer(4)]], constant uint& N [[buffer(5)]],                \
        constant uint& K [[buffer(6)]],                                                \
        uint3 tg       [[threadgroup_position_in_grid]],                               \
        uint  lid      [[thread_index_in_threadgroup]],                                \
        uint  simd_gid [[simdgroup_index_in_threadgroup]],                             \
        uint  simd_lid [[thread_index_in_simdgroup]]) {                                \
        qmm_t_nax_tgp_dual_impl<half, half, false, 64, 64, 64, 2, 2, false, false,     \
                                false, BITS, GROUP, true>(                             \
            blocks, x, y16, y32, nullptr, int(K), int(N), int(M), tg, lid,             \
            simd_gid, simd_lid);                                                       \
    }                                                                                  \
    kernel void qwg_gemm_shvq##BITS##_g##GROUP##_h16_bm128(                             \
        device const char* blocks [[buffer(0)]],                                       \
        device const half* x      [[buffer(1)]],                                       \
        device half*       y16    [[buffer(2)]],                                       \
        device float*      y32    [[buffer(3)]],                                       \
        constant uint& M [[buffer(4)]], constant uint& N [[buffer(5)]],                \
        constant uint& K [[buffer(6)]],                                                \
        uint3 tg       [[threadgroup_position_in_grid]],                               \
        uint  lid      [[thread_index_in_threadgroup]],                                \
        uint  simd_gid [[simdgroup_index_in_threadgroup]],                             \
        uint  simd_lid [[thread_index_in_simdgroup]]) {                                \
        qmm_t_nax_tgp_dual_impl<half, half, false, 128, 64, 64, 2, 2, false, false,    \
                                false, BITS, GROUP, true>(                             \
            blocks, x, y16, y32, nullptr, int(K), int(N), int(M), tg, lid,             \
            simd_gid, simd_lid);                                                       \
    }                                                                                  \
    kernel void qwg_gemm_shvq##BITS##_g##GROUP##_h16_gdn_inproj(                        \
        device const char* blocks [[buffer(0)]],                                       \
        device const half* x      [[buffer(1)]],                                       \
        device half*       qkv16  [[buffer(2)]],                                       \
        device float*      y32    [[buffer(3)]],                                       \
        constant uint& M [[buffer(4)]], constant uint& N [[buffer(5)]],                \
        constant uint& K [[buffer(6)]],                                                \
        device half*       z16    [[buffer(7)]],                                       \
        uint3 tg       [[threadgroup_position_in_grid]],                               \
        uint  lid      [[thread_index_in_threadgroup]],                                \
        uint  simd_gid [[simdgroup_index_in_threadgroup]],                             \
        uint  simd_lid [[thread_index_in_simdgroup]]) {                                \
        qmm_t_nax_tgp_dual_impl<half, half, false, 64, 64, 64, 2, 2, true, true,       \
                                false, BITS, GROUP, true>(                             \
            blocks, x, qkv16, y32, nullptr, int(K), int(N), int(M), tg, lid,           \
            simd_gid, simd_lid, z16);                                                  \
    }                                                                                  \
    kernel void qwg_gemm_shvq##BITS##_g##GROUP##_nax_f32(                               \
        device const char* w   [[buffer(0)]],                                          \
        device const half* xh  [[buffer(1)]],                                          \
        device const half* xl  [[buffer(2)]],                                          \
        device float*      y32 [[buffer(3)]],                                          \
        constant uint& M [[buffer(4)]], constant uint& N [[buffer(5)]],                \
        constant uint& K [[buffer(6)]],                                                \
        uint3 tg       [[threadgroup_position_in_grid]],                               \
        uint  simd_gid [[simdgroup_index_in_threadgroup]]) {                           \
        const int realM = int(M & 0x7FFFFFFFu);                                        \
        if ((M & 0x80000000u) != 0u) {                                                 \
            qmm_t_nax_split_shvq_impl<64, 2, 2, BITS, GROUP, false, half>(             \
                w, xh, xl, (device half*)y32, int(K), int(N), realM, tg, simd_gid);    \
        } else {                                                                       \
            qmm_t_nax_split_shvq_impl<64, 2, 2, BITS, GROUP, false, float>(            \
                w, xh, xl, y32, int(K), int(N), realM, tg, simd_gid);                  \
        }                                                                              \
    }                                                                                  \
    kernel void qwg_moe_nax_gemm_shvq##BITS##_g##GROUP(                                 \
        device const char*    w         [[buffer(0)]],                                 \
        device const half*    x         [[buffer(1)]],                                 \
        device half*          y         [[buffer(2)]],                                 \
        device const int32_t* expertOff [[buffer(3)]],                                 \
        device const int32_t* tileE     [[buffer(4)]],                                 \
        device const int32_t* tileR     [[buffer(5)]],                                 \
        device const int32_t* tileCount [[buffer(6)]],                                 \
        constant QWGemmSHVQGroupedParams& p [[buffer(7)]],                             \
        constant uint&        tileBlock [[buffer(8)]],                                 \
        device const int32_t* sortedTok [[buffer(9)]],                                 \
        constant uint&        hasGather [[buffer(10)]],                                \
        device const char*    coldW     [[buffer(11)]],                                \
        uint3 tg       [[threadgroup_position_in_grid]],                               \
        uint  lid      [[thread_index_in_threadgroup]],                                \
        uint  simd_gid [[simdgroup_index_in_threadgroup]],                             \
        uint  simd_lid [[thread_index_in_simdgroup]]) {                                \
        qwg_moe_nax_gemm_shvq_body<BITS, GROUP>(w, x, y, expertOff, tileE, tileR,      \
            tileCount, p, tileBlock, sortedTok, hasGather, coldW, tg, lid, simd_gid, simd_lid); \
    }                                                                                  \
    kernel void qwg_moe_nax_split_shvq##BITS##_g##GROUP##_f32(                          \
        device const char*    w         [[buffer(0)]],                                 \
        device const half*    xh        [[buffer(1)]],                                 \
        device const half*    xl        [[buffer(2)]],                                 \
        device float*         y32       [[buffer(3)]],                                 \
        device const int32_t* expertOff [[buffer(4)]],                                 \
        device const int32_t* tileE     [[buffer(5)]],                                 \
        device const int32_t* tileR     [[buffer(6)]],                                 \
        device const int32_t* tileCount [[buffer(7)]],                                 \
        constant QWGemmSHVQGroupedParams& p [[buffer(8)]],                             \
        constant uint&        tileBlock [[buffer(9)]],                                 \
        device const char*    coldW     [[buffer(10)]],                                \
        uint3 tg       [[threadgroup_position_in_grid]],                               \
        uint  simd_gid [[simdgroup_index_in_threadgroup]]) {                           \
        qwg_moe_nax_split_shvq_f32_body<BITS, GROUP>(w, xh, xl, y32, expertOff, tileE, \
            tileR, tileCount, p, tileBlock, coldW, tg, simd_gid);                      \
    }

QW_SHVQ_BATCHED_KERNELS(2, 32)
QW_SHVQ_BATCHED_KERNELS(2, 64)
QW_SHVQ_BATCHED_KERNELS(2, 128)
QW_SHVQ_BATCHED_KERNELS(2, 256)
QW_SHVQ_BATCHED_KERNELS(3, 32)
QW_SHVQ_BATCHED_KERNELS(3, 64)
QW_SHVQ_BATCHED_KERNELS(3, 128)
QW_SHVQ_BATCHED_KERNELS(3, 256)
QW_SHVQ_BATCHED_KERNELS(4, 32)
QW_SHVQ_BATCHED_KERNELS(4, 64)
QW_SHVQ_BATCHED_KERNELS(4, 128)
QW_SHVQ_BATCHED_KERNELS(4, 256)
