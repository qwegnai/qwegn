// qwgdn_prefill_nax.metal — K2 (GDN chunk-carry scan, C13) on simdgroup_matrix.
// Drop-in for qwg_gdn_prefill_scan: SAME 10-buffer contract, grid (48,8)x128,
// kernel name qwg_gdn_prefill_scan_nax. ONE simdgroup (32 threads) per
// Design note: docs/KERNEL_NOTES.md#qwgdn_prefill_nax.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "include/qw_geometry_constants.metalh"
using namespace metal;

#define NH128 128u
#define NVH (QW_GDN_V_HEADS)
#define NCOL (QW_GDN_Z_CH)
#define QW_LOG2E 1.4426950408889634f
#define QW_EXP(x) exp2((x) * QW_LOG2E)
#define HALF_AT(u, k) (((k) == 0) ? float(as_type<half>((ushort)((u) & 0xffffu))) : float(as_type<half>((ushort)((u) >> 16))))

kernel void qwg_gdn_prefill_scan_nax(
        device const half* y   [[buffer(0)]],
        device const float* Ubuf [[buffer(1)]], device const float* Wbuf [[buffer(2)]],
        device const float* Abuf [[buffer(3)]], device const float* Bbuf [[buffer(4)]],
        device const float* glog [[buffer(5)]],
        device float* state    [[buffer(6)]],    // [V_HEADS][128][128] in place
        device float* o_scr    [[buffer(7)]],    // [T][V_HEADS][128] f32 pre-norm
        constant uint& T       [[buffer(8)]],
        constant uint& L       [[buffer(9)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint h = tg.x, vb = tg.y;
    const uint nchunks = (T + L - 1u) / L;
    if (tid >= 32u) return;  // defensive: nax path dispatches 32 threads/TG
    const uint lane = tid;
    const uint i = h / QW_GDN_V_PER_QK;
    const uint vx = vb * 16u;

    threadgroup float S[128u * 16u];             // x-major: S[x*128 + jl]
    threadgroup float WSu[64u * 16u];            // t-major: UD[t*16 + x]
    threadgroup float KG[64u * 16u];             // k̂·kge [k*16 + jloc] (per w)
    threadgroup float kgeTab[64];                // e^{ΓL−Γt} — TG-shared, no per-thread spill

    device float* stg = state + (uint64_t)h * 16384u;
    for (uint e = lane; e < 128u * 16u; e += 32u) {   // load+transpose band
        const uint j = e >> 4u, x = e & 15u;
        S[x * NH128 + j] = stg[j * NH128 + vx + x];
    }

    device const half* krow = y + (QW_GDN_QK_HEADS * NH128) + i * NH128;      // k̂ block row 0

    for (uint c = 0u; c < nchunks; c++) {
        const uint t0 = c * L;
        const uint Lc = min(L, T - t0);
        float gsum = 0.0f;
        for (uint r = 0u; r < Lc; r++) gsum += glog[(t0 + r) * NVH + h];
        const float egL = QW_EXP(gsum);
        if (lane == 0u) {   // kge[t] = e^{ΓL−Γt}: ONE backward recurrence — 64 MUFU
            float kg = 1.0f;
            for (uint t = Lc; t-- > 0u;) {
                if (t != Lc - 1u) kg *= QW_EXP(glog[(t0 + t + 1u) * NVH + h]);
                kgeTab[t] = kg;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── pass1 (lane cells (t = lane>>4 + 2j, x = lane&15)): WS dot +
        // o fold, float4 BOTH sides (shipping dot law), oacc starts at A[t].
        const uint x1 = lane & 15u;
        for (uint t = (lane >> 4u); t < Lc; t += 2u) {
            device const float4* Wr = (device const float4*)(Wbuf + (uint64_t)(t0 + t) * NCOL + h * NH128);
            device const float4* Br = (device const float4*)(Bbuf + (uint64_t)(t0 + t) * NCOL + h * NH128);
            threadgroup const float4* Sr = (threadgroup const float4*)&S[x1 * NH128];
            float ws = 0.0f;
            float oacc = Abuf[(uint64_t)(t0 + t) * NCOL + h * NH128 + vx + x1];
            for (uint q = 0u; q < 32u; q++) {
                float4 sv = Sr[q];
                float4 wv = Wr[q];
                float4 bv = Br[q];
                ws = fma(wv.x, sv.x, ws); ws = fma(wv.y, sv.y, ws);
                ws = fma(wv.z, sv.z, ws); ws = fma(wv.w, sv.w, ws);
                oacc = fma(bv.x, sv.x, oacc); oacc = fma(bv.y, sv.y, oacc);
                oacc = fma(bv.z, sv.z, oacc); oacc = fma(bv.w, sv.w, oacc);
            }
            WSu[t * 16u + x1] = ws;
            o_scr[(uint64_t)(t0 + t) * NCOL + h * NH128 + vx + x1] = oacc;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);   // WS staged

        // ── UD fold in place (ragged: t ≥ Lc cells ZERO).
        for (uint t = (lane >> 4u); t < 64u; t += 2u) {
            WSu[t * 16u + x1] = (t < Lc)
                ? (Ubuf[(uint64_t)(t0 + t) * NCOL + h * NH128 + vx + x1] - WSu[t * 16u + x1])
                : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);   // UD staged

        // ── carry: explicit 8×8 simdgroup_matrix path. This is the proven
        // fragment law used by the shipped attention kernels, and it keeps the
        // K2 recurrence algebra but removes redundant scalar indexing in the
        // carry window.
        for (uint w = 0u; w < 8u; w++) {
            // stage the K̂g window in the exact same local order, but without
            // re-reading the source once per scalar element.
            for (uint e = lane; e < 64u * 16u; e += 32u) {
                const uint t = e >> 4u, jl = e & 15u;
                const float kg = (t < Lc) ? kgeTab[t] : 0.0f;
                const uint tt = min(t0 + t, T - 1u);
                KG[e] = kg * float(krow[(uint64_t)tt * QW_GDN_QKV_CH + (uint)w * 16u + jl]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 accum[2][2];
            for (uint xh = 0u; xh < 2u; xh++)
                for (uint nh = 0u; nh < 2u; nh++) {
                    simdgroup_load(accum[xh][nh], S + 1024u * xh + 16u * w + 8u * nh,
                                   128, ulong2(0, 0), false);
                    auto th = accum[xh][nh].thread_elements();
                    th[0] *= egL; th[1] *= egL;
                    accum[xh][nh].thread_elements() = th;
                }

            for (uint b = 0u; b < 8u; b++) {
                simdgroup_float8x8 af[2], bf[2];
                for (uint xh = 0u; xh < 2u; xh++)
                    simdgroup_load(af[xh], WSu + 128u * b + 8u * xh, 16, ulong2(0, 0), true);
                for (uint nh = 0u; nh < 2u; nh++)
                    simdgroup_load(bf[nh], KG + (128u * b) + 8u * nh, 16, ulong2(0, 0), false);
                for (uint xh = 0u; xh < 2u; xh++)
                    for (uint nh = 0u; nh < 2u; nh++)
                        simdgroup_multiply_accumulate(accum[xh][nh], af[xh], bf[nh], accum[xh][nh]);
            }

            for (uint xh = 0u; xh < 2u; xh++)
                for (uint nh = 0u; nh < 2u; nh++)
                    simdgroup_store(accum[xh][nh], S + 1024u * xh + 16u * w + 8u * nh,
                                    128, ulong2(0, 0), false);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint e = lane; e < 128u * 16u; e += 32u) {             // j-major tail
        const uint j = e >> 4u, x = e & 15u;
        stg[j * NH128 + vx + x] = S[x * NH128 + j];
    }
}
