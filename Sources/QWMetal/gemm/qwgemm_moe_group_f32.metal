// qwgemm_moe_group_f32.metal — expert-sorted grouped MoE, f32 (MLX SwitchGLU shape).
// The (token, route) pairs are sorted by expert id, then one grouped GEMM per expert applies that expert's
// slab to every pair routed to it: the slab is read once per expert, not once per pair.
// Tile: 64 pairs x 32 columns per threadgroup; the weight tile is staged once per K block.
// buffers (group pairs): 0 ids | 1 expertOff | 2 sortedPairIdx | 3 sortedTok | 4 experts | 5 totalPairs | 6 topK
// buffers (gate/up, down): see the Swift wrappers in QWGemmMoE.swift
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "qw_dequant.metalh"
#include "qw_dequant_f32.metalh"
#include "qw_dequant_multi.metalh"
#include "../include/qw_moe_tile.metalh"
using namespace metal;

#define QWG_MG_BM 64   // routed pairs per row block
#define QWG_MG_BN 32   // output columns per threadgroup
#define QWG_MG_LD 36   // threadgroup row stride (padded to dodge bank conflicts)

/// Expert-major pair order (MLX's _gather_sort): counting sort of pairs by expert id. expertOff[e]..expertOff[e+1]
/// is expert e's sorted range. Slot order within an expert is arbitrary: each pair's dot product is independent.
kernel void qwg_moe_group_pairs_f32(device const int* selected [[buffer(0)]],
                                     device int*       expertOff [[buffer(1)]],
                                     device int*       sortedPairIdx [[buffer(2)]],
                                     device int*       sortedTok [[buffer(3)]],
                                     constant uint&    experts [[buffer(4)]],
                                     constant uint&    totalPairs [[buffer(5)]],
                                     constant uint&    topK [[buffer(6)]],
                                     device int*       invPerm [[buffer(7)]],
                                     constant uint&    writeInv [[buffer(8)]],
                                     device int*       tileE [[buffer(9)]],
                                     device int*       tileR [[buffer(10)]],
                                     device int*       tileCount [[buffer(11)]],
                                     constant uint&    writeTiles [[buffer(12)]],
                                     uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_int tgCount[512];
    threadgroup int tgOff[513];
    for (int e = int(tid); e < int(experts); e += 256)
        atomic_store_explicit(&tgCount[e], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(tid); i < int(totalPairs); i += 256)
        atomic_fetch_add_explicit(&tgCount[selected[i]], 1, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32u) {
        const uint lane = tid;
        const uint chunk = (experts + 31u) / 32u;
        const uint e0 = min(lane * chunk, experts);
        const uint e1 = min(e0 + chunk, experts);
        int mySum = 0;
        for (uint e = e0; e < e1; ++e) {
            mySum += atomic_load_explicit(&tgCount[e], memory_order_relaxed);
        }
        int cursor = simd_prefix_exclusive_sum(mySum);
        for (uint e = e0; e < e1; ++e) {
            const int c = atomic_load_explicit(&tgCount[e], memory_order_relaxed);
            expertOff[e] = cursor;
            tgOff[e] = cursor;
            atomic_store_explicit(&tgCount[e], cursor, memory_order_relaxed);
            cursor += c;
        }
        if (lane == 31u) {
            expertOff[experts] = cursor;
            tgOff[experts] = cursor;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(tid); i < int(totalPairs); i += 256) {
        const int e = selected[i];
        const int at = atomic_fetch_add_explicit(&tgCount[e], 1, memory_order_relaxed);
        sortedPairIdx[at] = i;
        sortedTok[at] = i / int(topK);
        if (writeInv != 0u) invPerm[i] = at;
    }
    if (writeTiles != 0u && tid < 32u) {
        const uint lane = tid;
        const uint chunk = (experts + 31u) / 32u;
        const uint e0 = min(lane * chunk, experts);
        const uint e1 = min(e0 + chunk, experts);
        int myTiles = 0;
        for (uint e = e0; e < e1; ++e) {
            const int c = tgOff[e + 1] - tgOff[e];
            if (c > 0) {
                const int bm = qw_moe_tile_block(c, 0);
                myTiles += (c + bm - 1) / bm;
            }
        }
        int t = simd_prefix_exclusive_sum(myTiles);
        for (uint e = e0; e < e1; ++e) {
            const int c = tgOff[e + 1] - tgOff[e];
            if (c > 0) {
                const int bm = qw_moe_tile_block(c, 0);
                for (int r = 0; r * bm < c; ++r) {
                    tileE[t] = int(e);
                    tileR[t] = r;
                    ++t;
                }
            }
        }
        if (lane == 31u) tileCount[0] = t;
    }
}

/// One expert x one 32-column tile: the expert's slab is staged once per K block and applied to every pair
/// routed to that expert. grid.x = expert, grid.y = column tile.
kernel void qwg_moe_gateup_group_f32(device const Block_HQ4_G32* blocks [[buffer(0)]],
                                     device const float*         x      [[buffer(1)]],
                                     device float*               out    [[buffer(2)]],
                                     device const int*           expertOff [[buffer(3)]],
                                     device const int*           sortedPairIdx [[buffer(4)]],
                                     device const int*           sortedTok [[buffer(5)]],
                                     constant uint&              N      [[buffer(6)]],
                                     constant uint&              K      [[buffer(7)]],
                                     uint2 tg [[threadgroup_position_in_grid]],
                                     uint  tid [[thread_index_in_threadgroup]]) {
    threadgroup float As[QWG_MG_BM * QWG_MG_LD];  // [pair][k]
    threadgroup float Bs[QWG_MG_BN * QWG_MG_LD];  // [output col][k]
    threadgroup float* const Cs = As;             // epilogue aliases As: As is dead after the K loop

    const int e = int(tg.x);
    const int lo = expertOff[e], hi = expertOff[e + 1];
    if (lo >= hi) return;
    const int nBase = int(tg.y) * QWG_MG_BN;
    if (nBase >= int(N)) return;

    const uint sg = tid >> 5u;   // simdgroup 0..7 owns pairs sg*8..+8
    const uint gi = tid >> 3u;   // 8-lane group 0..31 owns output row nBase+gi
    const uint dl = tid & 7u;    // lane inside the group
    const int nBlocks = int(K >> 5u);
    const int nRow = min(nBase + int(gi), int(N) - 1);
    device const Block_HQ4_G32* const wrowBase = blocks + long(e * int(N) + nRow) * long(nBlocks);

    for (int p0 = lo; p0 < hi; p0 += QWG_MG_BM) {
        const int cnt = min(QWG_MG_BM, hi - p0);
        // The two rows this thread stages are fixed for the whole K loop, so the gather is hoisted out.
        const int row0 = int(gi), row1 = int(gi) + 32;
        const int pair0 = (row0 < cnt) ? sortedPairIdx[p0 + row0] : -1;
        const int pair1 = (row1 < cnt) ? sortedPairIdx[p0 + row1] : -1;
        const int tok0 = (pair0 >= 0) ? sortedTok[p0 + row0] : -1;
        const int tok1 = (pair1 >= 0) ? sortedTok[p0 + row1] : -1;
        simdgroup_float8x8 acc[4];
        for (int nb = 0; nb < 4; ++nb) acc[nb] = simdgroup_float8x8(0.0f);

        for (int g = 0; g < nBlocks; ++g) {
            // stage A: the routed pairs' rotated activations, 64 pairs x 32 k
            #pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                const int tk = int(gi) + pass * 32;
                const int tok = (pass == 0) ? tok0 : tok1;
                float4 v = float4(0.0f);
                if (tok >= 0) {
                    device const float* xm = x + long(tok) * long(K) + (g << 5) + int(dl << 2u);
                    v = *(device const float4*)xm;
                }
                As[tk * QWG_MG_LD + int(dl) * 4 + 0] = v.x;
                As[tk * QWG_MG_LD + int(dl) * 4 + 1] = v.y;
                As[tk * QWG_MG_LD + int(dl) * 4 + 2] = v.z;
                As[tk * QWG_MG_LD + int(dl) * 4 + 3] = v.w;
            }
            // stage B: 32 output rows x 32 k of this expert's slab, dequantized once for the whole block
            {
                float2 w01, w23;
                qwg_unpack_hq4_lane4_f32(wrowBase[g], dl, w01, w23);
                const int b = int(gi) * QWG_MG_LD + int(dl) * 4;
                Bs[b + 0] = w01.x;
                Bs[b + 1] = w01.y;
                Bs[b + 2] = w23.x;
                Bs[b + 3] = w23.y;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // A simdgroup owns rows sg*8..+8 of the tile, so the ones past the pair count have nothing to
            // compute: the routing is skewed and an expert holds far fewer pairs than a 64-row tile.
            if (int(sg) * 8 < cnt) {
                #pragma unroll
                for (int k8 = 0; k8 < 4; ++k8) {
                    simdgroup_float8x8 a;
                    simdgroup_load(a, As + int(sg) * 8 * QWG_MG_LD + k8 * 8, QWG_MG_LD, ulong2(0, 0), false);
                    #pragma unroll
                    for (int nb = 0; nb < 4; ++nb) {
                        simdgroup_float8x8 b;
                        simdgroup_load(b, Bs + nb * 8 * QWG_MG_LD + k8 * 8, QWG_MG_LD, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(acc[nb], a, b, acc[nb]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (int nb = 0; nb < 4; ++nb)
            simdgroup_store(acc[nb], Cs + int(sg) * 8 * QWG_MG_LD + nb * 8, QWG_MG_LD, ulong2(0, 0), false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = int(tid); idx < QWG_MG_BM * QWG_MG_BN; idx += 256) {
            const int m = idx / QWG_MG_BN;
            if (m >= cnt) continue;
            const int col = nBase + idx % QWG_MG_BN;
            if (col >= int(N)) continue;
            out[long(sortedPairIdx[p0 + m]) * long(N) + col] = Cs[m * QWG_MG_LD + idx % QWG_MG_BN];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

