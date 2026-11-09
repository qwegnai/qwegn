// qwmoe_promote.metal — indirect staging→bank expert copies and router-id → bank-row maps for dynamic residency.
//
// The host writes the promotion list ([count, staging0, bank0, ...]) after the
// route readback, while this dispatch is already committed behind an event wait.
#include <metal_stdlib>
using namespace metal;

kernel void qwg_moe_promote_copy(
        device const uint4* staging [[buffer(0)]],
        device uint4* bank [[buffer(1)]],
        device const uint* list [[buffer(2)]],
        constant uint& vecPerExpert [[buffer(3)]],
        uint tid [[thread_position_in_grid]],
        uint threads [[threads_per_grid]]) {
    const uint count = list[0];
    for (uint p = 0; p < count; ++p) {
        const ulong src = ulong(list[1 + 2 * p]) * vecPerExpert;
        const ulong dst = ulong(list[2 + 2 * p]) * vecPerExpert;
        for (uint i = tid; i < vecPerExpert; i += threads) bank[dst + i] = staging[src + i];
    }
}

// Dependent-FMA delay encoded after a cold route CB's ready signal: an event wait that blocks
// costs ~115 us to resume, one already signalled by the host is nearly free, so the GPU idles
// on its own until the host's remap lands. `iters` is tuned per step (ColdEventPipeline).
kernel void qwg_cold_event_bridge(device float* tick [[buffer(0)]], constant uint& iters [[buffer(1)]]) {
    float x = tick[0];
    for (uint i = 0u; i < iters; ++i) x = fma(x, 1.0001f, 0.5f);
    tick[0] = x;
}

// Maps the router's physical expert ids to resident bank slots through the host-mirrored
// residency table (slot, or -1 when not resident). Misses and dropped routes become -1, so a
// resident-only pass can run while the host is still staging the misses.
kernel void qwg_moe_resident_ids(
        device const int* ids [[buffer(0)]],
        device const int* slotMap [[buffer(1)]],
        device int* out [[buffer(2)]],
        constant uint& count [[buffer(3)]],
        constant uint& experts [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= count) return;
    const int e = ids[tid];
    out[tid] = (e >= 0 && uint(e) < experts) ? slotMap[e] : -1;
}

// Span router ids → bank rows for dynamic residency: resident → its slot, each distinct miss → residentE + its
// rank among the span's misses in expert-id order, invalid → -1. The host stages misses from this same output,
// so both sides agree by construction and the resident pass can precede the route readback. 1 TG; experts ≤ 512.
kernel void qwg_moe_span_ids(
        device const int* ids [[buffer(0)]],
        device const int* slotMap [[buffer(1)]],
        device int* out [[buffer(2)]],
        constant uint& count [[buffer(3)]],
        constant uint& experts [[buffer(4)]],
        constant uint& residentE [[buffer(5)]],
        uint tid [[thread_index_in_threadgroup]],
        uint tgSize [[threads_per_threadgroup]]) {
    threadgroup atomic_uint miss[512];
    threadgroup uint rank[512];
    for (uint e = tid; e < experts; e += tgSize) atomic_store_explicit(&miss[e], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < count; i += tgSize) {
        const int e = ids[i];
        if (e >= 0 && uint(e) < experts && slotMap[e] < 0) atomic_store_explicit(&miss[e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32u) {
        const uint chunk = (experts + 31u) / 32u;
        const uint e0 = min(tid * chunk, experts);
        const uint e1 = min(e0 + chunk, experts);
        uint mine = 0u;
        for (uint e = e0; e < e1; ++e) mine += atomic_load_explicit(&miss[e], memory_order_relaxed);
        uint cursor = simd_prefix_exclusive_sum(mine);
        for (uint e = e0; e < e1; ++e) {
            rank[e] = cursor;
            cursor += atomic_load_explicit(&miss[e], memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < count; i += tgSize) {
        const int e = ids[i];
        int o = -1;
        if (e >= 0 && uint(e) < experts) {
            const int s = slotMap[e];
            o = s >= 0 ? s : int(residentE + rank[e]);
        }
        out[i] = o;
    }
}
