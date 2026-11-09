#include <metal_stdlib>
using namespace metal;

kernel void qwg_mtp_embed_bf16(
    device const ushort* table [[buffer(0)]],
    device const int* token [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& vocabulary [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= hidden) return;
    const int id = token[0];
    // The host checks every retained token after completion. Invalid feedback
    // must not index the table while the rest of the command buffer drains.
    output[i] = id >= 0 && uint(id) < vocabulary
        ? half(as_type<float>(uint(table[ulong(id) * hidden + i]) << 16))
        : half(0);
}
