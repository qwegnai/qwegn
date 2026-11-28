// qwgemm_shvq_head_nax.metal — 16 KiB SRAM-LUT NAX small-M (M <= 16) GEMM
// for wide-vocabulary SHVQ4_G{32,64,128,256} heads (Law 27 <= 16 KiB constant).
#if !defined(QW_DISABLE_NAX)
#if !defined(__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__) || __ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ < 260200
#error "Metal 4 NAX kernels require -mmacosx-version-min=26.2"
#endif
#include <metal_stdlib>
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "include/qw_shvq_tables.metalh"
#include "include/qw_gemm_nax.metalh"
using namespace metal;

METAL_FUNC void qwg_shvq4_frag8_tg16(const device char* gp, uint colInGroup,
                                     threadgroup const uint* lut4,
                                     thread half4 &lo, thread half4 &hi) {
    const half sc = *(const device half*)gp;
    const device uchar* p = (const device uchar*)(gp + 2);
    const uint q0 = colInGroup >> 2u;
    const ushort u0 = *(const device ushort*)(p + 2u * q0);
    const ushort u1 = *(const device ushort*)(p + 2u * (q0 + 4u));
    const ushort4 p0 = as_type<ushort4>(half4(float4(as_type<uchar4>(lut4[u0 & 0x0FFFu])) * kShvq4StepScaleF32));
    const ushort4 p1 = as_type<ushort4>(half4(float4(as_type<uchar4>(lut4[u1 & 0x0FFFu])) * kShvq4StepScaleF32));
    const ushort4 s0 = (ushort4(ushort(u0 >> 12u)) << ushort4(15u, 14u, 13u, 12u)) & ushort4(0x8000u);
    const ushort4 s1 = (ushort4(ushort(u1 >> 12u)) << ushort4(15u, 14u, 13u, 12u)) & ushort4(0x8000u);
    lo = as_type<half4>(p0 | s0) * sc;
    hi = as_type<half4>(p1 | s1) * sc;
}

template <int GROUP, typename OutT>
METAL_FUNC void qwg_shvq4_nax_tg16_m16_impl(const device char* w, const device half* xh, device OutT* y,
                                            int K, int N, int M, uint3 tid, uint simd_gid,
                                            threadgroup const uint* lut4) {
    constexpr int BM = 16, BN = 64, WM = 1, WN = 4;
    constexpr short SM = BM / WM, SN = BN / WN, TM = SM / 16, TN = SN / 16, TK = 2;
    constexpr uint aqGroupBytes = qwg_shvq_group_bytes(4, GROUP);
    constexpr uint slicesPerGroup = uint(GROUP) >> 5u;
    const int K_g = K / 32, y_row = int(tid.y) * BM, y_col = int(tid.x) * BN;
    const short tm = SM * short(simd_gid / WN), tn = SN * short(simd_gid % WN);
    const short sgp_sm = short(min(int(SM), M - (y_row + int(tm))));
    const short sgp_sn = short(min(int(SN), N - (y_col + int(tn))));
    const short tgp_bn = short(min(BN, N - y_col));
    if (sgp_sm <= 0 || sgp_sn <= 0) return;

    xh += (y_row + tm) * static_cast<int64_t>(K);
    y += y_row * static_cast<int64_t>(N) + y_col;
    const uint aqRowBytes = uint(K / GROUP) * aqGroupBytes;
    const device char* w_aq_tile = w + static_cast<int64_t>(y_col) * aqRowBytes;
    const short2 sc = mlx::steel::BaseNAXFrag::get_coord();
    mlx::steel::NAXTile<float, TM, TN> Dtile;
    Dtile.clear();

    mlx::steel::dispatch_bool(sgp_sm == SM, [&](auto kAlignedM) {
        mlx::steel::dispatch_bool(tgp_bn == BN, [&](auto kAlignedN) {
            for (int g = 0; g < K_g; ++g) {
                mlx::steel::NAXTile<half, TM, TK> Ah;
                mlx::steel::NAXTile<half, TN, TK> Btile;
                if constexpr (kAlignedM.value) Ah.load(xh, K);
                else Ah.load_safe(xh, K, short2(32, sgp_sm));
                const uint groupIdx = uint(g) / slicesPerGroup;
                const uint colInGroup = ((uint(g) & (slicesPerGroup - 1u)) << 5u) + uint(sc.x);
                const device char* aqGroupBase = w_aq_tile + static_cast<int64_t>(groupIdx) * aqGroupBytes;
                mlx::steel::const_for_loop<0, TN, 1>([&](auto idx_row) {
                    thread auto& f0 = Btile.template frag_at<idx_row.value, 0>();
                    thread auto& f1 = Btile.template frag_at<idx_row.value, 1>();
                    STEEL_PRAGMA_UNROLL
                    for (short i = 0; i < 2; ++i) {
                        const int r_tile = tn + idx_row.value * 16 + int(sc.y) + int(i) * 8;
                        if (kAlignedN.value || r_tile < tgp_bn) {
                            half4 lo, hi;
                            qwg_shvq4_frag8_tg16(aqGroupBase + static_cast<int64_t>(r_tile) * aqRowBytes, colInGroup, lut4, lo, hi);
                            f0[i * 4 + 0] = lo.x; f0[i * 4 + 1] = lo.y; f0[i * 4 + 2] = lo.z; f0[i * 4 + 3] = lo.w;
                            f1[i * 4 + 0] = hi.x; f1[i * 4 + 1] = hi.y; f1[i * 4 + 2] = hi.z; f1[i * 4 + 3] = hi.w;
                        } else {
                            for (short z = 0; z < 4; ++z) { f0[i * 4 + z] = half(0); f1[i * 4 + z] = half(0); }
                        }
                    }
                });
                mlx::steel::tile_matmad_nax(Dtile, Ah, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
                xh += 32;
            }
            if constexpr (kAlignedM.value && kAlignedN.value) Dtile.store(y + tm * N + tn, N);
            else Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
        });
    });
}

#define QW_SHVQ4_HEAD_NAX_TG16(GROUP)                                                  \
    kernel void qwg_gemm_shvq4_g##GROUP##_nax_tg16_f32(                                \
        device const char* w   [[buffer(0)]],                                          \
        device const half* xh  [[buffer(1)]],                                          \
        device const half* xl  [[buffer(2)]],                                          \
        device float*      y32 [[buffer(3)]],                                          \
        constant uint& M [[buffer(4)]], constant uint& N [[buffer(5)]],                \
        constant uint& K [[buffer(6)]],                                                \
        uint3 tg       [[threadgroup_position_in_grid]],                               \
        uint  lid      [[thread_index_in_threadgroup]],                                \
        uint  simd_gid [[simdgroup_index_in_threadgroup]]) {                           \
        (void)xl;                                                                      \
        threadgroup uint lut4[4096];                                                   \
        for (uint i = lid; i < 4096u; i += 128u) lut4[i] = kShvqLut4PackedU8[i];       \
        threadgroup_barrier(mem_flags::mem_threadgroup);                               \
        const int realM = int(M & 0x0FFFFFFFu);                                        \
        if ((M & 0x80000000u) != 0u) {                                                 \
            qwg_shvq4_nax_tg16_m16_impl<GROUP, half>(w, xh, (device half*)y32, int(K), int(N), realM, tg, simd_gid, lut4); \
        } else if ((M & 0x10000000u) != 0u) {                                          \
            qwg_shvq4_nax_tg16_m16_impl<GROUP, bfloat>(w, xh, (device bfloat*)y32, int(K), int(N), realM, tg, simd_gid, lut4); \
        } else {                                                                       \
            qwg_shvq4_nax_tg16_m16_impl<GROUP, float>(w, xh, y32, int(K), int(N), realM, tg, simd_gid, lut4); \
        }                                                                              \
    }

QW_SHVQ4_HEAD_NAX_TG16(32)
QW_SHVQ4_HEAD_NAX_TG16(64)
QW_SHVQ4_HEAD_NAX_TG16(128)
QW_SHVQ4_HEAD_NAX_TG16(256)
#endif
