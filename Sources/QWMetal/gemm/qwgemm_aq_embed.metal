// qwgemm_aq_embed.metal — the size-preserving affine family's embedding
// gather: the incumbent's `qwg_embed_bf16_f32` over the family's own blocks.
// The row is decoded on the device, so the table keeps its native bytes.

// buffers: 0 blocks | 1 token | 2 out f32 | 3 hidden | 4 hc | 5 bits | 6 group

#include <metal_stdlib>
#include "qw_dequant_multi.metalh"
using namespace metal;

/// The family's embedding gather: the token's row, decoded on the device.
/// `out[r * hidden * hc + c * hidden + d]` is the value of column d — the
/// code's scale and the group's own bias, in the family's one arithmetic.
kernel void qwg_embed_aq_f32(device const char* blocks [[buffer(0)]],
                             device const int* token [[buffer(1)]],
                             device float* out [[buffer(2)]],
                             constant uint& hidden [[buffer(3)]],
                             constant uint& hc [[buffer(4)]],
                             constant uint& bits [[buffer(5)]],
                             constant uint& group [[buffer(6)]],
                             uint3 tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    const uint r = tg.y;
    const int tok = max(token[r], 0);   // caller guarantees tok < vocab
    const uint groupBytes = group * bits / 8 + 4u;
    const uint rowBytes = (hidden / group) * groupBytes;
    device const char* rowBlocks = blocks + (size_t)tok * rowBytes;
    device float* outRow = out + (size_t)r * hidden * hc;

    // The stride is a multiple of the group, so the group index advances by a
    // constant and the column inside the group never moves: one division per
    // thread, not one per element.
    const uint stride = 256u * 4096u;
    const uint groupStride = stride / group;
    uint d = tg.x * 256u + tid;
    if (d >= hidden) return;
    uint g = d / group;
    const uint j = d - g * group;

    for (; d < hidden; d += stride) {
        device const char* gp = rowBlocks + (size_t)g * groupBytes;
        const float scale = float(*(device const half*)gp);
        const float bias = float(*(device const half*)(gp + 2));
        device const uchar* p = (device const uchar*)(gp + 4);
        const float code = float(qwg_aq_read_bits(p, int(bits), int(j)));
        const float v = code * scale + bias;
        for (uint c = 0; c < hc; ++c) {
            outRow[(size_t)c * hidden + d] = v;
        }
        g += groupStride;
    }
}
