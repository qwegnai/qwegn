// qwgdn_conv.metal — C11: causal depthwise conv1d (k=4) over the packed
// GDN projections [T][10240] + SiLU + conv-state rotation.
//
// Design note: docs/KERNEL_NOTES.md#qwgdn_conv.
#include <metal_stdlib>
#include "include/qw_geometry_constants.metalh"
using namespace metal;

#define QW_GDN_C  QW_GDN_QKV_CH     // QKV_CH = (2·QK_HEADS + V_HEADS)·128
#define QW_TOKB   32u               // tokens per band
#define QW_BANDCH 256u              // channels per workgroup (32 lanes × 8 = 256 channels/band)

kernel void qwg_gdn_conv(const device half* x     [[buffer(0)]],   // [T][10240]
                         const device half* w     [[buffer(1)]],   // [10240][4]
                         const device half* bias  [[buffer(2)]],   // [10240]
                         const device half* stIn  [[buffer(3)]],   // [3][10240] age
                         device half* y           [[buffer(4)]],   // [T][10240]
                         device half* stOut       [[buffer(5)]],   // [3][10240]
                         constant uint& T         [[buffer(6)]],
                         uint tid  [[thread_index_in_threadgroup]],
                         uint3 tg  [[threadgroup_position_in_grid]]) {   // (chBand, tokBand, 0)
    const uint c0 = tg.x * 256u + tid * 8u;         // lane's 8 channels (256-channel band = 32 lanes)
    const uint t0 = tg.y * QW_TOKB;
    if (t0 >= T || c0 >= QW_GDN_C) { return; }
    const uint t1 = min(t0 + QW_TOKB, T);

    // src(u) → the padded row for padded-index u (u ≥ 0 by contract):
    //   u < 3 → state row u;  u ≥ 3 → x row u−3.  8 halves → dst[8].
    // PRECEDENCE LAW (caught by the _pfx tests): the ternary binds LOOSER
    #define QW_SRC(u, dst) {                                              \
        if ((u) < 3u) {                                                   \
            const device half* _p = (((u) == 0u) ? stIn                                                                   \
                                      : (((u) == 1u) ? (stIn + QW_GDN_C) : (stIn + 2u * QW_GDN_C))) + c0;                 \
            for (uint _j = 0; _j < 8u; _j++) dst[_j] = _p[_j];            \
        } else {                                                          \
            const device half* _p = x + ((u) - 3u) * QW_GDN_C + c0;       \
            for (uint _j = 0; _j < 8u; _j++) dst[_j] = _p[_j];            \
        }                                                                 \
    }

    // lane's 8-channel tap block + bias (loaded once per workgroup band)
    half wv[32];                                     // row-major [c][4]
    { const device half* wp = w + c0 * 4u; for (uint i = 0; i < 32u; i++) wv[i] = wp[i]; }
    half bv[8];
    { const device half* bp = bias + c0; for (uint i = 0; i < 8u; i++) bv[i] = bp[i]; }

    // sliding window: p[k] = padded[t + k], seeded at band start
    half p0[8], p1[8], p2[8], p3[8];
    QW_SRC(t0 + 0u, p0)
    QW_SRC(t0 + 1u, p1)
    QW_SRC(t0 + 2u, p2)
    QW_SRC(t0 + 3u, p3)

    for (uint t = t0; t < t1; t++) {
        device half* yp = y + t * QW_GDN_C + c0;
        #pragma unroll
        for (uint j = 0; j < 8u; j++) {
            float acc = float(bv[j]);                // fp32 accumulation,
            acc += float(wv[j * 4u + 0u]) * float(p0[j]);   // taps in k-ascending
            acc += float(wv[j * 4u + 1u]) * float(p1[j]);   // order (oracle
            acc += float(wv[j * 4u + 2u]) * float(p2[j]);   // F.conv1d parity)
            acc += float(wv[j * 4u + 3u]) * float(p3[j]);
            yp[j] = half(acc / (1.0f + metal::exp(-acc)));  // silu fp32 (golden)
        }
        // slide registers (zero DRAM move — the state rotate is by registers)
        for (uint j = 0; j < 8u; j++) {
            p0[j] = p1[j]; p1[j] = p2[j]; p2[j] = p3[j];
        }
        if (t + 1u < t1) { QW_SRC(t + 4u, p3) }      // refetch only if the
        //  band continues: the final iteration's src(t+4) would read x[T]   \
        //  (one row OOB) — a value never used. Guarded here, not by padding.
    }

    // rotated state: the band holding token T−1 writes the last three padded
    // rows (byte-copy — raw, pre-SiLU). Three src() row loads per lane; this
    // is also correct for T ≤ 3 (older state rows resurface, per the oracle:
    // state2 = padded[T..T+2] regardless of what was in stIn).
    if (t1 == T && (T <= QW_TOKB || stIn != stOut)) {
        half s0[8], s1[8], s2r[8];
        QW_SRC(T + 0u, s0)
        QW_SRC(T + 1u, s1)
        QW_SRC(T + 2u, s2r)
        device half* so = stOut + c0;
        for (uint j = 0; j < 8u; j++) so[j] = s0[j];
        so = stOut + QW_GDN_C + c0;
        for (uint j = 0; j < 8u; j++) so[j] = s1[j];
        so = stOut + 2u * QW_GDN_C + c0;
        for (uint j = 0; j < 8u; j++) so[j] = s2r[j];
    }
}

// ── qwg_gdn_conv_headnorm_t1 — Fused Conv1D (k=4) + HeadNorm for T=1 decode ─
// Fuses C11 depthwise conv (10,240 channels) with C23 headnorm (16 Q heads +
// 16 K heads = 32 heads of 128 channels).
kernel void qwg_gdn_conv_headnorm_t1(const device half* x    [[buffer(0)]],   // [10240]
                                    const device half* w    [[buffer(1)]],   // [10240][4]
                                    const device half* bias [[buffer(2)]],   // [10240]
                                    const device half* stIn [[buffer(3)]],   // [3][10240]
                                    device half* y          [[buffer(4)]],   // [10240]
                                    device half* stOut      [[buffer(5)]],   // [3][10240]
                                    uint tg                 [[threadgroup_position_in_grid]],
                                    uint tid                [[thread_index_in_threadgroup]]) {
    const uint h = tg;
    const uint totalHeads = 2u * QW_GDN_QK_HEADS + QW_GDN_V_HEADS;
    if (h >= totalHeads) return;
    const uint c = h * 128u + tid;

    // Load state taps + input
    const half p0 = stIn[c];
    const half p1 = stIn[QW_GDN_C + c];
    const half p2 = stIn[2u * QW_GDN_C + c];
    const half p3 = x[c];

    // Evaluate 4-tap FIR + SiLU
    const device half* wp = w + (size_t)c * 4u;
    float acc = float(bias[c]);
    acc += float(wp[0]) * float(p0);
    acc += float(wp[1]) * float(p1);
    acc += float(wp[2]) * float(p2);
    acc += float(wp[3]) * float(p3);
    const float y_silu = acc / (1.0f + metal::exp(-acc));

    // Update state ring buffer (safe in-place because each thread owns channel c)
    stOut[c] = p1;
    stOut[QW_GDN_C + c] = p2;
    stOut[2u * QW_GDN_C + c] = p3;

    if (h < 2u * QW_GDN_QK_HEADS) {
        // HeadNorm: match oracle/pipeline rounding law (cast to half then float)
        const float v = float(half(y_silu));
        const float ss = simd_sum(v * v);
        threadgroup float part[4];
        if ((tid & 31u) == 0u) {
            part[tid >> 5u] = ss;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float total = part[0] + part[1] + part[2] + part[3];
        const float scale = (h < QW_GDN_QK_HEADS) ? 0.08838834764831845f : 1.0f; // 128^-0.5 for Q
        y[c] = half(v * rsqrt(total + 1e-6f) * scale);
    } else {
        y[c] = half(y_silu);
    }
}

// ── qwg_gdn_conv_headnorm_rows — Multi-row Conv1D (k=4) + HeadNorm + Capture ─
// Steps R verification rows sequentially in registers (grid = totalHeads × 128 lanes)
// and writes the final 3-tap ring state once at the end.
kernel void qwg_gdn_conv_headnorm_rows(
    const device half*  x            [[buffer(0)]],   // [R][xStride]
    const device half*  w            [[buffer(1)]],   // [QKV_CH][4]
    const device half*  bias         [[buffer(2)]],   // [QKV_CH]
    const device half*  stIn         [[buffer(3)]],   // [3][QKV_CH]
    device half*        y            [[buffer(4)]],   // [R][QKV_CH]
    device half*        stOut        [[buffer(5)]],   // [3][QKV_CH]
    constant uint&      R            [[buffer(6)]],
    constant uint&      xStride      [[buffer(7)]],
    device half*        rawCapture   [[buffer(8)]],   // optional [R][QKV_CH]
    const device float* controls     [[buffer(9)]],   // optional [R][2*V_HEADS]
    device float*       betasCapture [[buffer(10)]],  // optional [R][V_HEADS]
    device float*       glogCapture  [[buffer(11)]],  // optional [R][V_HEADS]
    uint tg                          [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]) {
    const uint h = tg;
    const uint totalHeads = 2u * QW_GDN_QK_HEADS + QW_GDN_V_HEADS;
    if (h >= totalHeads) return;
    const uint c = h * 128u + tid;

    half p0 = stIn[c];
    half p1 = stIn[QW_GDN_C + c];
    half p2 = stIn[2u * QW_GDN_C + c];

    const device half* wp = w + (size_t)c * 4u;
    const float b  = float(bias[c]);
    const float w0 = float(wp[0]);
    const float w1 = float(wp[1]);
    const float w2 = float(wp[2]);
    const float w3 = float(wp[3]);

    threadgroup float part[4];
    const float scale = (h < QW_GDN_QK_HEADS) ? 0.08838834764831845f : 1.0f;

    for (uint r = 0u; r < R; ++r) {
        const half p3 = x[(size_t)r * xStride + c];
        if (rawCapture != nullptr) {
            rawCapture[(size_t)r * QW_GDN_C + c] = p3;
        }
        float acc = b;
        acc += w0 * float(p0);
        acc += w1 * float(p1);
        acc += w2 * float(p2);
        acc += w3 * float(p3);
        const float y_silu = acc / (1.0f + metal::exp(-acc));
        p0 = p1;
        p1 = p2;
        p2 = p3;

        if (h < 2u * QW_GDN_QK_HEADS) {
            const float v = float(half(y_silu));
            const float ss = simd_sum(v * v);
            if ((tid & 31u) == 0u) {
                part[tid >> 5u] = ss;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float total = part[0] + part[1] + part[2] + part[3];
            y[(size_t)r * QW_GDN_C + c] = half(v * rsqrt(total + 1e-6f) * scale);
            if (r + 1u < R) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        } else {
            y[(size_t)r * QW_GDN_C + c] = half(y_silu);
        }
    }

    stOut[c] = p0;
    stOut[QW_GDN_C + c] = p1;
    stOut[2u * QW_GDN_C + c] = p2;

    if (h == 0u && tid < QW_GDN_V_HEADS && controls != nullptr) {
        for (uint r = 0u; r < R; ++r) {
            betasCapture[(size_t)r * QW_GDN_V_HEADS + tid] = controls[(size_t)r * (2u * QW_GDN_V_HEADS) + tid];
            glogCapture[(size_t)r * QW_GDN_V_HEADS + tid]  = controls[(size_t)r * (2u * QW_GDN_V_HEADS) + QW_GDN_V_HEADS + tid];
        }
    }
}

// ── qwg_gdn_conv_headnorm — Fused Conv1D (k=4) + HeadNorm for T >= 1 prefill ───
// Fuses C11 depthwise conv (10,240 channels) with C23 headnorm (16 Q heads +
// 16 K heads = 32 heads of 128 channels).
kernel void qwg_gdn_conv_headnorm(const device half* x       [[buffer(0)]],   // [T][10240]
                                  const device half* w       [[buffer(1)]],   // [10240][4]
                                  const device half* bias    [[buffer(2)]],   // [10240]
                                  const device half* stIn    [[buffer(3)]],   // [3][10240] age
                                  device half* y             [[buffer(4)]],   // [T][10240]
                                  device half* stOut         [[buffer(5)]],   // [3][10240]
                                  constant uint& T           [[buffer(6)]],
                                  const device float* gdnY32 [[buffer(7)]],
                                  const device float* aLog   [[buffer(8)]],
                                  const device float* dtBias [[buffer(9)]],
                                  device float* betas        [[buffer(10)]],
                                  device float* glog         [[buffer(11)]],
                                  uint tid  [[thread_index_in_threadgroup]],
                                  uint3 tg  [[threadgroup_position_in_grid]]) {   // (chBand, tokBand, 0)
    const uint c0 = tg.x * QW_BANDCH + tid * 8u;    // lane's 8 channels
    const uint t0 = tg.y * QW_TOKB;
    if (t0 >= T || c0 >= QW_GDN_C) { return; }
    const uint t1 = min(t0 + QW_TOKB, T);

    #define QW_SRC_FUSED(u, dst) {                                        \
        if ((u) < 3u) {                                                   \
            const device half* _p = (((u) == 0u) ? stIn                   \
                                      : (((u) == 1u) ? (stIn + QW_GDN_C) : (stIn + 2u * QW_GDN_C))) + c0; \
            for (uint _j = 0; _j < 8u; _j++) dst[_j] = _p[_j];            \
        } else {                                                          \
            const device half* _p = x + ((u) - 3u) * QW_GDN_C + c0;       \
            for (uint _j = 0; _j < 8u; _j++) dst[_j] = _p[_j];            \
        }                                                                 \
    }

    half wv[32];
    { const device half* wp = w + c0 * 4u; for (uint i = 0; i < 32u; i++) wv[i] = wp[i]; }
    half bv[8];
    { const device half* bp = bias + c0; for (uint i = 0; i < 8u; i++) bv[i] = bp[i]; }

    half p0[8], p1[8], p2[8], p3[8];
    QW_SRC_FUSED(t0 + 0u, p0)
    QW_SRC_FUSED(t0 + 1u, p1)
    QW_SRC_FUSED(t0 + 2u, p2)
    QW_SRC_FUSED(t0 + 3u, p3)

    if (tg.x >= QW_GDN_QK_HEADS) {
        // V heads: pass-through SiLU
        for (uint t = t0; t < t1; t++) {
            device half* yp = y + t * QW_GDN_C + c0;
            #pragma unroll
            for (uint j = 0; j < 8u; j++) {
                float acc = float(bv[j]);
                acc += float(wv[j * 4u + 0u]) * float(p0[j]);
                acc += float(wv[j * 4u + 1u]) * float(p1[j]);
                acc += float(wv[j * 4u + 2u]) * float(p2[j]);
                acc += float(wv[j * 4u + 3u]) * float(p3[j]);
                yp[j] = half(acc / (1.0f + metal::exp(-acc)));
            }
            for (uint j = 0; j < 8u; j++) {
                p0[j] = p1[j]; p1[j] = p2[j]; p2[j] = p3[j];
            }
            if (t + 1u < t1) { QW_SRC_FUSED(t + 4u, p3) }
        }
    } else {
        // Q heads and K heads: 2 heads of 128 channels per 256-channel band.
        // Each head is 16 lanes x 8 channels = 128 channels.
        // 16 lanes reside entirely within a single SIMD group half (bits 0..3 of lane ID).
        const float scale = (tg.x < (QW_GDN_QK_HEADS / 2u)) ? 0.08838834764831845f : 1.0f; // 128^-0.5 for Q
        for (uint t = t0; t < t1; t++) {
            device half* yp = y + t * QW_GDN_C + c0;
            float lane_ss = 0.0f;
            half y_silu[8];
            #pragma unroll
            for (uint j = 0; j < 8u; j++) {
                float acc = float(bv[j]);
                acc += float(wv[j * 4u + 0u]) * float(p0[j]);
                acc += float(wv[j * 4u + 1u]) * float(p1[j]);
                acc += float(wv[j * 4u + 2u]) * float(p2[j]);
                acc += float(wv[j * 4u + 3u]) * float(p3[j]);
                const half yh = half(acc / (1.0f + metal::exp(-acc)));
                y_silu[j] = yh;
                const float v = float(yh);
                lane_ss += v * v;
            }
            // 16-lane reduction across the head's 128 channels
            float total = lane_ss;
            total += simd_shuffle_xor(total, ushort(1));
            total += simd_shuffle_xor(total, ushort(2));
            total += simd_shuffle_xor(total, ushort(4));
            total += simd_shuffle_xor(total, ushort(8));
            const float inv_norm = rsqrt(total + 1e-6f) * scale;
            #pragma unroll
            for (uint j = 0; j < 8u; j++) {
                yp[j] = half(float(y_silu[j]) * inv_norm);
            }
            for (uint j = 0; j < 8u; j++) {
                p0[j] = p1[j]; p1[j] = p2[j]; p2[j] = p3[j];
            }
            if (t + 1u < t1) { QW_SRC_FUSED(t + 4u, p3) }
        }
    }

    if (t1 == T && (T <= QW_TOKB || stIn != stOut)) {
        half s0[8], s1[8], s2r[8];
        QW_SRC_FUSED(T + 0u, s0)
        QW_SRC_FUSED(T + 1u, s1)
        QW_SRC_FUSED(T + 2u, s2r)
        device half* so = stOut + c0;
        for (uint j = 0; j < 8u; j++) so[j] = s0[j];
        so = stOut + QW_GDN_C + c0;
        for (uint j = 0; j < 8u; j++) so[j] = s1[j];
        so = stOut + 2u * QW_GDN_C + c0;
        for (uint j = 0; j < 8u; j++) so[j] = s2r[j];
    }

    if (gdnY32 != nullptr && tg.x < 2u) {
        const uint h = tg.x * 32u + tid;
        if (h < QW_GDN_V_HEADS) {
            const float al = aLog[h];
            const float db = dtBias[h];
            const uint aRow0 = QW_GDN_QKV_CH + QW_GDN_Z_CH;
            const uint bRow0 = aRow0 + QW_GDN_V_HEADS;
            const uint totalN = bRow0 + QW_GDN_V_HEADS;
            for (uint t = t0; t < t1; t++) {
                const size_t in_row = (size_t)t * totalN;
                const float av = gdnY32[in_row + aRow0 + h];
                const float bv = gdnY32[in_row + bRow0 + h];
                const size_t outIdx = (size_t)t * QW_GDN_V_HEADS + h;
                betas[outIdx] = 1.0f / (1.0f + metal::exp(-bv));
                const float sp = metal::log(1.0f + metal::exp(av + db));
                glog[outIdx] = -metal::exp(al) * sp;
            }
        }
    }

    #undef QW_SRC_FUSED
}

// Separate dispatch prevents the last token band from overwriting state
// before the first band has loaded it on an in-place multi-band call.
kernel void qwg_gdn_conv_rotate(
    const device half* x [[buffer(0)]],
    const device half* stIn [[buffer(1)]],
    device half* stOut [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    uint c [[thread_position_in_grid]]) {
    if (c >= QW_GDN_C) return;
    const half s0 = T < 3u ? stIn[T * QW_GDN_C + c] : x[(T - 3u) * QW_GDN_C + c];
    const half s1 = T < 2u ? stIn[(T + 1u) * QW_GDN_C + c] : x[(T - 2u) * QW_GDN_C + c];
    const half s2 = x[(T - 1u) * QW_GDN_C + c];
    stOut[c] = s0;
    stOut[QW_GDN_C + c] = s1;
    stOut[2u * QW_GDN_C + c] = s2;
}
