// 32-head (256T) head-major + heavy-first prefill NAX kernel for MLA;
// shares blkIdx across two 16-head groups so SLC waves reuse sparse KV pages.
#if !defined(QW_DISABLE_NAX)
#include "qw_mla_nax_common.metalh"

// Grid: (qHeads / 32, T, 1), Threadgroup: (256, 1, 1); 20 KiB TG SRAM (16+4).
kernel void qwg_mla_sparse_attn_nax_h32(
    device const half*           qAbs      [[buffer(0)]],
    device const half*           kPool     [[buffer(1)]],
    device const half*           vPool     [[buffer(2)]],
    device const int*            kTable    [[buffer(3)]],
    device const int*            vTable    [[buffer(4)]],
    device const int*            blockList [[buffer(5)]],
    device half*                 uLatOut   [[buffer(6)]],
    constant MLASparseAttnParams& p        [[buffer(7)]],
    uint2                        tg        [[threadgroup_position_in_grid]],
    uint                         tid       [[thread_index_in_threadgroup]],
    uint                         sg        [[simdgroup_index_in_threadgroup]],
    uint                         lane      [[thread_index_in_simdgroup]]
) {
    if (tg.y >= p.T || tg.x * 32u >= p.qHeads) return;
    threadgroup float sPart[2 * 4 * 16 * 32];
    threadgroup uint  blkIdx[1024];
    const uint t = p.T - 1u - tg.y;
    const uint sg_h = sg >> 2u;
    const uint sg_d = sg & 3u;
    const uint h0 = tg.x * 32u + sg_h * MLA_NAX_HG;
    if (p.kvMode == 0u) {
        qwg_mla_nax_attend<false, 0u, 2u>(qAbs, kPool, vPool, kTable, vTable, blockList, uLatOut, nullptr, p,
                                          t, h0, 0u, 1u, tid, sg_h, sg_d, lane, sPart, blkIdx);
    } else if (p.kvMode == 1u) {
        qwg_mla_nax_attend<false, 1u, 2u>(qAbs, kPool, vPool, kTable, vTable, blockList, uLatOut, nullptr, p,
                                          t, h0, 0u, 1u, tid, sg_h, sg_d, lane, sPart, blkIdx);
    } else {
        qwg_mla_nax_attend<false, 2u, 2u>(qAbs, kPool, vPool, kTable, vTable, blockList, uLatOut, nullptr, p,
                                          t, h0, 0u, 1u, tid, sg_h, sg_d, lane, sPart, blkIdx);
    }
}
#endif // !defined(QW_DISABLE_NAX)
