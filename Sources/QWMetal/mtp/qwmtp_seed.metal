#include <metal_stdlib>
using namespace metal;

// The pair-cache layout follows the engine KV mode (4c): the MTP draft cache
// is built with the engine's kvMode and read back by qwattn_decode/_batched/
// _shared through the same runtime q8mode addressing, so this writer must
kernel void qwg_mtp_seed_kv(
        device const half* k [[buffer(0)]],
        device const half* v [[buffer(1)]],
        device half* kPool [[buffer(2)]],
        device half* vPool [[buffer(3)]],
        device const int* kTable [[buffer(4)]],
        device const int* vTable [[buffer(5)]],
        constant uint& row [[buffer(6)]],
        constant uint& q8mode [[buffer(7)]],   // 0 f16, 1 q8, 2 q4
        uint e [[thread_position_in_grid]]) {
    constexpr uint64_t Q8_PAGE = 262144ull + 2048ull;
    constexpr uint64_t Q4_PAGE = 131072ull + 8192ull;
    if (e >= 1024u) return;
    const uint page = row / 256u;
    const uint slot = row % 256u;

    if (q8mode == 0u) {
        const uint offset = slot * 1024u + e;
        kPool[(ulong)kTable[page] * 256u * 1024u + offset] = k[e];
        vPool[(ulong)vTable[page] * 256u * 1024u + offset] = v[e];
        return;
    }

    const half kx = k[e], vx = v[e];
    const uint head = e >> 8;         // 0..3 = threadgroup index
    const uint dim  = e & 255u;       // dim within the head

    if (q8mode == 1u) {
        threadgroup float sK[4][8], sV[4][8];
        float km = simd_max(fabs(float(kx)));
        float vm = simd_max(fabs(float(vx)));
        const uint sub = dim >> 5;
        if ((e & 31u) == 0u) { sK[head][sub] = km; sV[head][sub] = vm; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        km = sK[head][0]; vm = sV[head][0];
        #pragma unroll
        for (uint i = 1u; i < 8u; i++) { km = fmax(km, sK[head][i]); vm = fmax(vm, sV[head][i]); }
        const float ks = float(half(fmax(km / 127.0f, 1e-8f)));
        const float vs = float(half(fmax(vm / 127.0f, 1e-8f)));
        device char* kd = (device char*)kPool + (uint64_t)kTable[page] * Q8_PAGE
                          + (uint64_t)slot * 1024u + (uint64_t)head * 256u + dim;
        device char* vd = (device char*)vPool + (uint64_t)vTable[page] * Q8_PAGE
                          + (uint64_t)slot * 1024u + (uint64_t)head * 256u + dim;
        *kd = (char)(int)clamp(round(float(kx) * (1.0f / ks)), -127.0f, 127.0f);
        *vd = (char)(int)clamp(round(float(vx) * (1.0f / vs)), -127.0f, 127.0f);
        if (dim == 0u) {   // one scale word per (slot, head)
            *(device half*)((device char*)kPool + (uint64_t)kTable[page] * Q8_PAGE
                            + 262144ull + (uint64_t)slot * 8u + (uint64_t)head * 2u) = half(ks);
            *(device half*)((device char*)vPool + (uint64_t)vTable[page] * Q8_PAGE
                            + 262144ull + (uint64_t)slot * 8u + (uint64_t)head * 2u) = half(vs);
        }
        return;
    }

    // q4: group-of-128 = head's dims [128·g, 128·g+128) = lanes (aligned)
    // 4 simdgroups; per-subgroup min/max to scratch, one barrier, fold 4.
    threadgroup float mnK[8][4], mxK[8][4], mnV[8][4], mxV[8][4];
    const uint grp = head * 2u + (dim >> 7);     // 0..7 global group
    const uint sub = (dim >> 5) & 3u;            // subgroup within the group
    float kmin = simd_min(float(kx)), kmax = simd_max(float(kx));
    float vmin = simd_min(float(vx)), vmax = simd_max(float(vx));
    if ((e & 31u) == 0u) {
        mnK[grp][sub] = kmin; mxK[grp][sub] = kmax;
        mnV[grp][sub] = vmin; mxV[grp][sub] = vmax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kmin = mnK[grp][0]; kmax = mxK[grp][0]; vmin = mnV[grp][0]; vmax = mxV[grp][0];
    #pragma unroll
    for (uint i = 1u; i < 4u; i++) {
        kmin = fmin(kmin, mnK[grp][i]); kmax = fmax(kmax, mxK[grp][i]);
        vmin = fmin(vmin, mnV[grp][i]); vmax = fmax(vmax, mxV[grp][i]);
    }
    const float ksc = float(half(fmax((kmax - kmin) / 15.0f, 1e-8f)));
    const float kmn = float(half(kmin));
    const float vsc = float(half(fmax((vmax - vmin) / 15.0f, 1e-8f)));
    const float vmn = float(half(vmin));
    const float knib = clamp(round((float(kx) - kmn) * (1.0f / ksc)), 0.0f, 15.0f);
    const float vnib = clamp(round((float(vx) - vmn) * (1.0f / vsc)), 0.0f, 15.0f);
    // Odd dim borrows the EVEN neighbour's nibble (same aligned simdgroup:
    // dim parity == lane parity within the head). Even lanes write the byte.
    const uint dInGrp = dim & 127u;
    const float kNbHi = simd_shuffle_xor(knib, 1);
    const float vNbHi = simd_shuffle_xor(vnib, 1);
    device uchar* kd = (device uchar*)kPool + (uint64_t)kTable[page] * Q4_PAGE
                       + (uint64_t)slot * 512u + (uint64_t)grp * 64u + (dInGrp >> 1);
    device uchar* vd = (device uchar*)vPool + (uint64_t)vTable[page] * Q4_PAGE
                       + (uint64_t)slot * 512u + (uint64_t)grp * 64u + (dInGrp >> 1);
    if ((dInGrp & 1u) == 0u) {
        *kd = (uchar)((uint)knib | ((uint)kNbHi << 4));
        *vd = (uchar)((uint)vnib | ((uint)vNbHi << 4));
    }
    if (dInGrp == 0u) {   // group's dim-0 lane writes {sc, mn}
        device half* kh = (device half*)((device char*)kPool + (uint64_t)kTable[page] * Q4_PAGE
                          + 131072ull + (uint64_t)slot * 32u + (uint64_t)grp * 4u);
        device half* vh = (device half*)((device char*)vPool + (uint64_t)vTable[page] * Q4_PAGE
                          + 131072ull + (uint64_t)slot * 32u + (uint64_t)grp * 4u);
        kh[0] = half(ksc); kh[1] = half(kmn);
        vh[0] = half(vsc); vh[1] = half(vmn);
    }
}
