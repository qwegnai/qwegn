// qwgemm_aq_batched.metal — the size-preserving affine family's prefill
// twins: the incumbent's `qwg_gemm_q4g_h16` geometry over the family's own
// blocks. The same NAX template, the family's arithmetic, and NO rotation —
// the family's weights carry none.

// buffers: 0 blocks | 1 x f16 | 2 y16 f16 | 3 y32 f32 | 4 M | 5 N | 6 K
// The GDN in-projection twin: 0 blocks | 1 x | 2 qkv16 | 3 y32 | 4 M | 5 N
// | 6 K | 7 z16.

#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_moe_tile.metalh"
#include "include/qw_gemm_nax.metalh"

// ── The size-preserving affine family's prefill twins ──
// The incumbent's prefill GEMMs over the family's own container: the same
// geometry, the family's arithmetic, and NO rotation — its weights carry none.
#define QW_AQ_GEMM_KERNEL(BITS, GROUP, NAME, BM)                                 \
    kernel void NAME(device const char* blocks [[buffer(0)]],                      \
                     device const half* x [[buffer(1)]],                           \
                     device half* y16 [[buffer(2)]],                               \
                     device float* y32 [[buffer(3)]],                              \
                     constant uint& M [[buffer(4)]], constant uint& N [[buffer(5)]], \
                     constant uint& K [[buffer(6)]],                               \
                     uint3 tg [[threadgroup_position_in_grid]],                    \
                     uint lid [[thread_index_in_threadgroup]],                    \
                     uint simd_gid [[simdgroup_index_in_threadgroup]],           \
                     uint simd_lid [[thread_index_in_simdgroup]]) {              \
        qmm_t_nax_tgp_dual_impl<half, half, false, BM, 64, 64, 2, 2, false,       \
                                false, false, BITS, GROUP>(                       \
            blocks, x, y16, y32, nullptr, int(K), int(N), int(M), tg, lid,         \
            simd_gid, simd_lid);                                                  \
    }

QW_AQ_GEMM_KERNEL(2, 32, qwg_gemm_aq2_g32_h16, 64)
QW_AQ_GEMM_KERNEL(2, 64, qwg_gemm_aq2_g64_h16, 64)
QW_AQ_GEMM_KERNEL(2, 128, qwg_gemm_aq2_g128_h16, 64)
QW_AQ_GEMM_KERNEL(2, 256, qwg_gemm_aq2_g256_h16, 64)
QW_AQ_GEMM_KERNEL(3, 32, qwg_gemm_aq3_g32_h16, 64)
QW_AQ_GEMM_KERNEL(3, 64, qwg_gemm_aq3_g64_h16, 64)
QW_AQ_GEMM_KERNEL(3, 128, qwg_gemm_aq3_g128_h16, 64)
QW_AQ_GEMM_KERNEL(3, 256, qwg_gemm_aq3_g256_h16, 64)
QW_AQ_GEMM_KERNEL(4, 32, qwg_gemm_aq4_g32_h16, 64)
QW_AQ_GEMM_KERNEL(4, 64, qwg_gemm_aq4_g64_h16, 64)
QW_AQ_GEMM_KERNEL(4, 128, qwg_gemm_aq4_g128_h16, 64)
QW_AQ_GEMM_KERNEL(4, 256, qwg_gemm_aq4_g256_h16, 64)
QW_AQ_GEMM_KERNEL(6, 32, qwg_gemm_aq6_g32_h16, 64)
QW_AQ_GEMM_KERNEL(6, 64, qwg_gemm_aq6_g64_h16, 64)
QW_AQ_GEMM_KERNEL(6, 128, qwg_gemm_aq6_g128_h16, 64)
QW_AQ_GEMM_KERNEL(6, 256, qwg_gemm_aq6_g256_h16, 64)
QW_AQ_GEMM_KERNEL(8, 32, qwg_gemm_aq8_g32_h16, 64)
QW_AQ_GEMM_KERNEL(8, 64, qwg_gemm_aq8_g64_h16, 64)
QW_AQ_GEMM_KERNEL(8, 128, qwg_gemm_aq8_g128_h16, 64)
QW_AQ_GEMM_KERNEL(8, 256, qwg_gemm_aq8_g256_h16, 64)
QW_AQ_GEMM_KERNEL(2, 32, qwg_gemm_aq2_g32_h16_bm128, 128)
QW_AQ_GEMM_KERNEL(2, 64, qwg_gemm_aq2_g64_h16_bm128, 128)
QW_AQ_GEMM_KERNEL(2, 128, qwg_gemm_aq2_g128_h16_bm128, 128)
QW_AQ_GEMM_KERNEL(2, 256, qwg_gemm_aq2_g256_h16_bm128, 128)
QW_AQ_GEMM_KERNEL(3, 32, qwg_gemm_aq3_g32_h16_bm128, 128)
QW_AQ_GEMM_KERNEL(3, 64, qwg_gemm_aq3_g64_h16_bm128, 128)
QW_AQ_GEMM_KERNEL(3, 128, qwg_gemm_aq3_g128_h16_bm128, 128)
QW_AQ_GEMM_KERNEL(3, 256, qwg_gemm_aq3_g256_h16_bm128, 128)
QW_AQ_GEMM_KERNEL(4, 32, qwg_gemm_aq4_g32_h16_bm128, 128)
QW_AQ_GEMM_KERNEL(4, 64, qwg_gemm_aq4_g64_h16_bm128, 128)
QW_AQ_GEMM_KERNEL(4, 128, qwg_gemm_aq4_g128_h16_bm128, 128)
QW_AQ_GEMM_KERNEL(4, 256, qwg_gemm_aq4_g256_h16_bm128, 128)
QW_AQ_GEMM_KERNEL(6, 32, qwg_gemm_aq6_g32_h16_bm128, 128)
QW_AQ_GEMM_KERNEL(6, 64, qwg_gemm_aq6_g64_h16_bm128, 128)
QW_AQ_GEMM_KERNEL(6, 128, qwg_gemm_aq6_g128_h16_bm128, 128)
QW_AQ_GEMM_KERNEL(6, 256, qwg_gemm_aq6_g256_h16_bm128, 128)
QW_AQ_GEMM_KERNEL(8, 32, qwg_gemm_aq8_g32_h16_bm128, 128)
QW_AQ_GEMM_KERNEL(8, 64, qwg_gemm_aq8_g64_h16_bm128, 128)
QW_AQ_GEMM_KERNEL(8, 128, qwg_gemm_aq8_g128_h16_bm128, 128)
QW_AQ_GEMM_KERNEL(8, 256, qwg_gemm_aq8_g256_h16_bm128, 128)

// The GDN's in-projection prefill twin: the incumbent's
// `qwg_gemm_q4g_h16_gdn_inproj` over the family's blocks — the same NAX
// geometry, the same split-GDN epilogue, the family's arithmetic.
#define QW_AQ_GEMM_GDN_KERNEL(BITS, GROUP, NAME)                                     \
    kernel void NAME(device const char* blocks [[buffer(0)]],                        \
                     device const half* x [[buffer(1)]],                             \
                     device half* qkv16 [[buffer(2)]],                               \
                     device float* y32 [[buffer(3)]],                                \
                     constant uint& M [[buffer(4)]], constant uint& N [[buffer(5)]],   \
                     constant uint& K [[buffer(6)]],                                  \
                     device half* z16 [[buffer(7)]],                                  \
                     uint3 tg [[threadgroup_position_in_grid]],                        \
                     uint lid [[thread_index_in_threadgroup]],                        \
                     uint simd_gid [[simdgroup_index_in_threadgroup]],                \
                     uint simd_lid [[thread_index_in_simdgroup]]) {                   \
        qmm_t_nax_tgp_dual_impl<half, half, false, 64, 64, 64, 2, 2, true, true,     \
                                false, BITS, GROUP>(                                  \
            blocks, x, qkv16, y32, nullptr, int(K), int(N), int(M), tg, lid,         \
            simd_gid, simd_lid, z16);                                                 \
    }

QW_AQ_GEMM_GDN_KERNEL(2, 32, qwg_gemm_aq2_g32_h16_gdn_inproj)
QW_AQ_GEMM_GDN_KERNEL(2, 64, qwg_gemm_aq2_g64_h16_gdn_inproj)
QW_AQ_GEMM_GDN_KERNEL(2, 128, qwg_gemm_aq2_g128_h16_gdn_inproj)
QW_AQ_GEMM_GDN_KERNEL(2, 256, qwg_gemm_aq2_g256_h16_gdn_inproj)
QW_AQ_GEMM_GDN_KERNEL(3, 32, qwg_gemm_aq3_g32_h16_gdn_inproj)
QW_AQ_GEMM_GDN_KERNEL(3, 64, qwg_gemm_aq3_g64_h16_gdn_inproj)
QW_AQ_GEMM_GDN_KERNEL(3, 128, qwg_gemm_aq3_g128_h16_gdn_inproj)
QW_AQ_GEMM_GDN_KERNEL(3, 256, qwg_gemm_aq3_g256_h16_gdn_inproj)
QW_AQ_GEMM_GDN_KERNEL(4, 32, qwg_gemm_aq4_g32_h16_gdn_inproj)
QW_AQ_GEMM_GDN_KERNEL(4, 64, qwg_gemm_aq4_g64_h16_gdn_inproj)
QW_AQ_GEMM_GDN_KERNEL(4, 128, qwg_gemm_aq4_g128_h16_gdn_inproj)
QW_AQ_GEMM_GDN_KERNEL(4, 256, qwg_gemm_aq4_g256_h16_gdn_inproj)
QW_AQ_GEMM_GDN_KERNEL(6, 32, qwg_gemm_aq6_g32_h16_gdn_inproj)
QW_AQ_GEMM_GDN_KERNEL(6, 64, qwg_gemm_aq6_g64_h16_gdn_inproj)
QW_AQ_GEMM_GDN_KERNEL(6, 128, qwg_gemm_aq6_g128_h16_gdn_inproj)
QW_AQ_GEMM_GDN_KERNEL(6, 256, qwg_gemm_aq6_g256_h16_gdn_inproj)
QW_AQ_GEMM_GDN_KERNEL(8, 32, qwg_gemm_aq8_g32_h16_gdn_inproj)
QW_AQ_GEMM_GDN_KERNEL(8, 64, qwg_gemm_aq8_g64_h16_gdn_inproj)
QW_AQ_GEMM_GDN_KERNEL(8, 128, qwg_gemm_aq8_g128_h16_gdn_inproj)
QW_AQ_GEMM_GDN_KERNEL(8, 256, qwg_gemm_aq8_g256_h16_gdn_inproj)

struct QWGemmAQGroupedParams { uint experts; uint n; uint k; uint blocksPerRow; uint rowTiles; uint colTiles; };

template <const int BM, const int WM, const int WN, const int BITS, const int GROUP, const bool splitLo = true, typename OutT = float>
METAL_FUNC void qmm_t_nax_split_aq_impl(
    const device char* w,
    const device half* xh,
    const device half* xl,
    device OutT* y,
    int K,
    int N,
    int M,
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
  constexpr uint aqGroupBytes = (GROUP * BITS) / 8 + 4u;
  constexpr uint slicesPerGroup = uint(GROUP) >> 5u;
  const uint aqRowBytes = uint(K / GROUP) * aqGroupBytes;
  const device char* w_aq_tile = w + static_cast<int64_t>(y_col) * aqRowBytes;

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
        const uint aqGroupIndex = uint(g) / slicesPerGroup;
        const uint aqBitInGroup = (((uint(g) & (slicesPerGroup - 1u)) << 5u) + uint(sc.x)) * uint(BITS);
        const device char* aqGroupBase =
            w_aq_tile + static_cast<int64_t>(aqGroupIndex) * aqGroupBytes;

        const_for_loop<0, TN, 1>([&](auto idx_row) {
          thread auto& f0 = Btile.template frag_at<idx_row.value, 0>();
          thread auto& f1 = Btile.template frag_at<idx_row.value, 1>();
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < 2; ++i) {
            const int r_tile = int(tn) + idx_row.value * 16 + int(sc.y) + int(i) * 8;
            if (kAlignedN.value || r_tile < int(tgp_bn)) {
              const device char* gp = aqGroupBase + static_cast<int64_t>(r_tile) * aqRowBytes;
              half4 lo, hi;
              qwg_aq_frag8<BITS>(gp, aqBitInGroup, lo, hi);
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
static inline void qwg_moe_nax_gemm_aq_body(device const char* w,
                                            device const half* x,
                                            device half* y,
                                            device const int32_t* expertOff,
                                            device const int32_t* tileE,
                                            device const int32_t* tileR,
                                            device const int32_t* tileCount,
                                            constant QWGemmAQGroupedParams& p,
                                            uint tileBlock,
                                            device const int32_t* sortedTok,
                                            uint hasGather,
                                            device const char* coldW,
                                            uint3 tg,
                                            uint lid,
                                            uint simd_gid,
                                            uint simd_lid) {
    uint t = tg.y + ((tileBlock >> 8) & 0x7FFFFFu);
    if (int(t) >= tileCount[0]) return;
    uint e = uint(tileE[t]);
    int rowBase = int(expertOff[e]);
    int rowCount = int(expertOff[e + 1]) - rowBase;
    constexpr size_t groupBytes = (GROUP * BITS) / 8 + 4u;
    const size_t rowBytes = (size_t(p.k) / GROUP) * groupBytes;
    const uint residentE = p.experts >> 16u;
    const bool isCold = (residentE > 0u) && (e >= residentE);
    if (((tileBlock & 0x40u) != 0u && isCold) || ((tileBlock & 0x80u) != 0u && !isCold)) return;
    const uint wExp = isCold ? (e - residentE) : e;
    device const char* we = (isCold ? coldW : w) + size_t(wExp) * size_t(p.n) * rowBytes;
    device half* ye = y + size_t(rowBase) * size_t(p.n);
    uint3 tid = uint3(tg.x, uint(tileR[t]), 0);
    const int bm = qw_moe_tile_block(rowCount, int(tileBlock & 0x3Fu));
    if (hasGather != 0u) {
        device const int32_t* st = sortedTok + rowBase;
        if (bm == 16) {
            qmm_t_nax_tgp_dual_impl<half, half, true, 16, 64, 64, 1, 4, false, false, true, BITS, GROUP>(
                we, x, ye, nullptr, nullptr, int(p.k), int(p.n), rowCount, tid, lid, simd_gid, simd_lid, nullptr, st);
        } else if (bm == 32) {
            qmm_t_nax_tgp_dual_impl<half, half, true, 32, 64, 64, 2, 2, false, false, true, BITS, GROUP>(
                we, x, ye, nullptr, nullptr, int(p.k), int(p.n), rowCount, tid, lid, simd_gid, simd_lid, nullptr, st);
        } else {
            qmm_t_nax_tgp_dual_impl<half, half, true, 64, 64, 64, 2, 2, false, false, true, BITS, GROUP>(
                we, x, ye, nullptr, nullptr, int(p.k), int(p.n), rowCount, tid, lid, simd_gid, simd_lid, nullptr, st);
        }
    } else {
        device const half* xe = x + size_t(rowBase) * size_t(p.k);
        if (bm == 16) {
            qmm_t_nax_tgp_dual_impl<half, half, true, 16, 64, 64, 1, 4, false, false, false, BITS, GROUP>(
                we, xe, ye, nullptr, nullptr, int(p.k), int(p.n), rowCount, tid, lid, simd_gid, simd_lid);
        } else if (bm == 32) {
            qmm_t_nax_tgp_dual_impl<half, half, true, 32, 64, 64, 2, 2, false, false, false, BITS, GROUP>(
                we, xe, ye, nullptr, nullptr, int(p.k), int(p.n), rowCount, tid, lid, simd_gid, simd_lid);
        } else {
            qmm_t_nax_tgp_dual_impl<half, half, true, 64, 64, 64, 2, 2, false, false, false, BITS, GROUP>(
                we, xe, ye, nullptr, nullptr, int(p.k), int(p.n), rowCount, tid, lid, simd_gid, simd_lid);
        }
    }
}

template <int BITS, int GROUP>
static inline void qwg_moe_nax_split_aq_f32_body(device const char* w,
                                                 device const half* xh,
                                                 device const half* xl,
                                                 device float* y32,
                                                 device const int32_t* expertOff,
                                                 device const int32_t* tileE,
                                                 device const int32_t* tileR,
                                                 device const int32_t* tileCount,
                                                 constant QWGemmAQGroupedParams& p,
                                                 uint tileBlock,
                                                 device const char* coldW,
                                                 uint3 tg,
                                                 uint simd_gid) {
    uint t = tg.y + ((tileBlock >> 8) & 0x7FFFFFu);
    if (int(t) >= tileCount[0]) return;
    uint e = uint(tileE[t]);
    int rowBase = int(expertOff[e]);
    int rowCount = int(expertOff[e + 1]) - rowBase;
    constexpr size_t groupBytes = (GROUP * BITS) / 8 + 4u;
    const size_t rowBytes = (size_t(p.k) / GROUP) * groupBytes;
    const uint residentE = p.experts >> 16u;
    const bool isCold = (residentE > 0u) && (e >= residentE);
    if (((tileBlock & 0x40u) != 0u && isCold) || ((tileBlock & 0x80u) != 0u && !isCold)) return;
    const uint wExp = isCold ? (e - residentE) : e;
    device const char* we = (isCold ? coldW : w) + size_t(wExp) * size_t(p.n) * rowBytes;
    device const half* xhe = xh + size_t(rowBase) * size_t(p.k);
    device const half* xle = xl + size_t(rowBase) * size_t(p.k);
    uint3 tid = uint3(tg.x, uint(tileR[t]), 0);
    const int bm = qw_moe_tile_block(rowCount, int(tileBlock & 0x3Fu));
    if ((tileBlock & 0x80000000u) != 0u) {
        device half* ye16 = ((device half*)y32) + size_t(rowBase) * size_t(p.n);
        if (bm == 16) {
            qmm_t_nax_split_aq_impl<16, 1, 4, BITS, GROUP, false, half>(we, xhe, xle, ye16, int(p.k), int(p.n), rowCount, tid, simd_gid);
        } else if (bm == 32) {
            qmm_t_nax_split_aq_impl<32, 2, 2, BITS, GROUP, false, half>(we, xhe, xle, ye16, int(p.k), int(p.n), rowCount, tid, simd_gid);
        } else {
            qmm_t_nax_split_aq_impl<64, 2, 2, BITS, GROUP, false, half>(we, xhe, xle, ye16, int(p.k), int(p.n), rowCount, tid, simd_gid);
        }
    } else {
        device float* ye = y32 + size_t(rowBase) * size_t(p.n);
        if (bm == 16) {
            qmm_t_nax_split_aq_impl<16, 1, 4, BITS, GROUP, false, float>(we, xhe, xle, ye, int(p.k), int(p.n), rowCount, tid, simd_gid);
        } else if (bm == 32) {
            qmm_t_nax_split_aq_impl<32, 2, 2, BITS, GROUP, false, float>(we, xhe, xle, ye, int(p.k), int(p.n), rowCount, tid, simd_gid);
        } else {
            qmm_t_nax_split_aq_impl<64, 2, 2, BITS, GROUP, false, float>(we, xhe, xle, ye, int(p.k), int(p.n), rowCount, tid, simd_gid);
        }
    }
}

#define QW_AQ_NAX_F32_KERNELS(BITS, GROUP) \
kernel void qwg_gemm_aq##BITS##_g##GROUP##_nax_f32(device const char* w   [[buffer(0)]], \
                                                   device const half* xh  [[buffer(1)]], \
                                                   device const half* xl  [[buffer(2)]], \
                                                   device float*      y32 [[buffer(3)]], \
                                                   constant uint&     M   [[buffer(4)]], \
                                                   constant uint&     N   [[buffer(5)]], \
                                                   constant uint&     K   [[buffer(6)]], \
                                                   uint3 tg [[threadgroup_position_in_grid]], \
                                                   uint simd_gid [[simdgroup_index_in_threadgroup]]) { \
    const int realM = int(M & 0x7FFFFFFFu); \
    if ((M & 0x80000000u) != 0u) { \
        qmm_t_nax_split_aq_impl<64, 2, 2, BITS, GROUP, false, half>(w, xh, xl, (device half*)y32, int(K), int(N), realM, tg, simd_gid); \
    } else { \
        qmm_t_nax_split_aq_impl<64, 2, 2, BITS, GROUP, false, float>(w, xh, xl, y32, int(K), int(N), realM, tg, simd_gid); \
    } \
} \
kernel void qwg_moe_nax_gemm_aq##BITS##_g##GROUP(device const char* w                  [[buffer(0)]], \
                                                 device const half* x                  [[buffer(1)]], \
                                                 device half* y                        [[buffer(2)]], \
                                                 device const int32_t* expertOff       [[buffer(3)]], \
                                                 device const int32_t* tileE           [[buffer(4)]], \
                                                 device const int32_t* tileR           [[buffer(5)]], \
                                                 device const int32_t* tileCount       [[buffer(6)]], \
                                                 constant QWGemmAQGroupedParams& p     [[buffer(7)]], \
                                                 constant uint& tileBlock              [[buffer(8)]], \
                                                 device const int32_t* sortedTok       [[buffer(9)]], \
                                                 constant uint& hasGather              [[buffer(10)]], \
                                                 device const char* coldW              [[buffer(11)]], \
                                                 uint3 tg [[threadgroup_position_in_grid]], \
                                                 uint lid [[thread_index_in_threadgroup]], \
                                                 uint simd_gid [[simdgroup_index_in_threadgroup]], \
                                                 uint simd_lid [[thread_index_in_simdgroup]]) { \
    qwg_moe_nax_gemm_aq_body<BITS, GROUP>(w, x, y, expertOff, tileE, tileR, tileCount, p, tileBlock, sortedTok, hasGather, coldW, tg, lid, simd_gid, simd_lid); \
} \
kernel void qwg_moe_nax_split_aq##BITS##_g##GROUP##_f32(device const char* w                  [[buffer(0)]], \
                                                        device const half* xh                 [[buffer(1)]], \
                                                        device const half* xl                 [[buffer(2)]], \
                                                        device float* y32                     [[buffer(3)]], \
                                                        device const int32_t* expertOff       [[buffer(4)]], \
                                                        device const int32_t* tileE           [[buffer(5)]], \
                                                        device const int32_t* tileR           [[buffer(6)]], \
                                                        device const int32_t* tileCount       [[buffer(7)]], \
                                                        constant QWGemmAQGroupedParams& p     [[buffer(8)]], \
                                                        constant uint& tileBlock              [[buffer(9)]], \
                                                        device const char* coldW              [[buffer(10)]], \
                                                        uint3 tg [[threadgroup_position_in_grid]], \
                                                        uint simd_gid [[simdgroup_index_in_threadgroup]]) { \
    qwg_moe_nax_split_aq_f32_body<BITS, GROUP>(w, xh, xl, y32, expertOff, tileE, tileR, tileCount, p, tileBlock, coldW, tg, simd_gid); \
}

QW_AQ_NAX_F32_KERNELS(2, 32)
QW_AQ_NAX_F32_KERNELS(2, 64)
QW_AQ_NAX_F32_KERNELS(2, 128)
QW_AQ_NAX_F32_KERNELS(2, 256)
QW_AQ_NAX_F32_KERNELS(3, 32)
QW_AQ_NAX_F32_KERNELS(3, 64)
QW_AQ_NAX_F32_KERNELS(3, 128)
QW_AQ_NAX_F32_KERNELS(3, 256)
QW_AQ_NAX_F32_KERNELS(4, 32)
QW_AQ_NAX_F32_KERNELS(4, 64)
QW_AQ_NAX_F32_KERNELS(4, 128)
QW_AQ_NAX_F32_KERNELS(4, 256)
QW_AQ_NAX_F32_KERNELS(6, 32)
QW_AQ_NAX_F32_KERNELS(6, 64)
QW_AQ_NAX_F32_KERNELS(6, 128)
QW_AQ_NAX_F32_KERNELS(6, 256)
QW_AQ_NAX_F32_KERNELS(8, 32)
QW_AQ_NAX_F32_KERNELS(8, 64)
QW_AQ_NAX_F32_KERNELS(8, 128)
QW_AQ_NAX_F32_KERNELS(8, 256)
