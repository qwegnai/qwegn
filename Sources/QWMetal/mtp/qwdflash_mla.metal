// qwdflash_mla.metal — Matrix-Absorbed MLA and GatedNorm kernels for DFlash 2.
// Implements BF16-exact GatedNorm, fused sandwich norms, compressed c_KV/k_pe
// context & latent prep, W_UK query absorption, MLA block attention, and W_UV+gate.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_mla_dequant.metalh"
using namespace metal;

constant bool FC_DRAFTER_ROTATED [[function_constant(236)]];
constant bool FC_HEAD_ROTATED    [[function_constant(237)]];

static inline float dflash_bf16_to_f32(ushort b) {
    return as_type<float>(uint(b) << 16u);
}

static inline float dflash_bf16_round_f32(float v) {
    uint u = as_type<uint>(v);
    if ((u & 0x7F800000u) == 0x7F800000u) return v;
    u = u + 0x7FFFu + ((u >> 16u) & 1u);
    return as_type<float>(u & 0xFFFF0000u);
}

static inline float dflash_mla_simd_fwht32_f32(float x, uint lane) {
    #pragma unroll
    for (ushort s = 1; s < 32; s <<= 1) {
        float partner = simd_shuffle_xor(x, s);
        x = ((lane & uint(s)) == 0u) ? (x + partner) : (partner - x);
    }
    return x * 0.17677669529663689f;
}

static inline void dflash_mla_rope_pair(
        device const half* src, device half* dst, device const float* invFreq,
        uint pos, uint pairIdx, uint ropeDim, uint interleave, float mscale) {
    const float ang = float(pos) * invFreq[pairIdx];
    const float c = cos(ang) * mscale, s = sin(ang) * mscale;
    if (interleave != 0u) {
        const float x0 = float(src[2u * pairIdx]), x1 = float(src[2u * pairIdx + 1u]);
        dst[2u * pairIdx]      = half(x0 * c - x1 * s);
        dst[2u * pairIdx + 1u] = half(x1 * c + x0 * s);
    } else {
        const uint halfDim = ropeDim >> 1u;
        const float x0 = float(src[pairIdx]), x1 = float(src[pairIdx + halfDim]);
        dst[pairIdx]           = half(x0 * c - x1 * s);
        dst[pairIdx + halfDim] = half(x1 * c + x0 * s);
    }
}

struct DFlashGatedNormParams {
    float eps;
    float gatingWeight;
    float scale;
    uint hidden;
    uint rank;
    uint zeroCentered;
    uint mode;
    uint hasNextPreNorm;
};

static inline void dflash_eval_gated_norm_stage12(
        threadgroup float* shPart, threadgroup float* shH, threadgroup float* shG1,
        device const ushort* normW, device const ushort* gateUp,
        float eps, uint hidden, uint rank, uint zeroCentered,
        float localSq, uint tid, uint tptg) {
    const uint sg = tid >> 5u, lane = tid & 31u, nSg = tptg >> 5u;
    const float sgSum = simd_sum(localSq);
    if (lane == 0u) shPart[sg] = sgSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float tot = simd_sum((lane < nSg) ? shPart[lane] : 0.0f);
        if (lane == 0u) shPart[0] = rsqrt(tot * (1.0f / float(hidden)) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rstd = shPart[0];
    for (uint i = tid; i < hidden; i += tptg) {
        float w = dflash_bf16_to_f32(normW[i]) + ((zeroCentered != 0u) ? 1.0f : 0.0f);
        shH[i] = dflash_bf16_round_f32(shH[i] * rstd * w);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = sg; r < rank; r += nSg) {
        device const ushort* guRow = gateUp + (size_t)r * hidden;
        float acc = 0.0f;
        for (uint c = lane; c < hidden; c += 32u) {
            acc += dflash_bf16_to_f32(guRow[c]) * shH[c];
        }
        const float s = simd_sum(acc);
        if (lane == 0u) shG1[r] = dflash_bf16_round_f32(s / (1.0f + exp(-s)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline float dflash_eval_gated_norm_elem(
        uint i, threadgroup const float* shH, threadgroup const float* shG1,
        device const ushort* gateDown, uint rank, float gatingWeight, float scale) {
    device const ushort* gdRow = gateDown + (size_t)i * rank;
    float gdAcc = 0.0f;
    if (rank == 16u && (((size_t)gdRow & 15u) == 0u)) {
        device const uint4* dp4 = (device const uint4*)gdRow;
        const uint4 d0 = dp4[0], d1 = dp4[1];
        gdAcc = fma(as_type<float>(d0.x << 16u), shG1[0], gdAcc);
        gdAcc = fma(as_type<float>(d0.x & 0xFFFF0000u), shG1[1], gdAcc);
        gdAcc = fma(as_type<float>(d0.y << 16u), shG1[2], gdAcc);
        gdAcc = fma(as_type<float>(d0.y & 0xFFFF0000u), shG1[3], gdAcc);
        gdAcc = fma(as_type<float>(d0.z << 16u), shG1[4], gdAcc);
        gdAcc = fma(as_type<float>(d0.z & 0xFFFF0000u), shG1[5], gdAcc);
        gdAcc = fma(as_type<float>(d0.w << 16u), shG1[6], gdAcc);
        gdAcc = fma(as_type<float>(d0.w & 0xFFFF0000u), shG1[7], gdAcc);
        gdAcc = fma(as_type<float>(d1.x << 16u), shG1[8], gdAcc);
        gdAcc = fma(as_type<float>(d1.x & 0xFFFF0000u), shG1[9], gdAcc);
        gdAcc = fma(as_type<float>(d1.y << 16u), shG1[10], gdAcc);
        gdAcc = fma(as_type<float>(d1.y & 0xFFFF0000u), shG1[11], gdAcc);
        gdAcc = fma(as_type<float>(d1.z << 16u), shG1[12], gdAcc);
        gdAcc = fma(as_type<float>(d1.z & 0xFFFF0000u), shG1[13], gdAcc);
        gdAcc = fma(as_type<float>(d1.w << 16u), shG1[14], gdAcc);
        gdAcc = fma(as_type<float>(d1.w & 0xFFFF0000u), shG1[15], gdAcc);
    } else {
        for (uint r = 0u; r < rank; ++r) gdAcc += dflash_bf16_to_f32(gdRow[r]) * shG1[r];
    }
    const float g2 = dflash_bf16_round_f32(1.0f / (1.0f + exp(-gdAcc)));
    float yi = dflash_bf16_round_f32(shH[i] * gatingWeight * g2);
    if (scale != 1.0f) yi = dflash_bf16_round_f32(yi * scale);
    return yi;
}

// Standalone GatedNorm: mode 0=f32->unrot f16, 1=f16->rot f16, 2=f32->unrot+rot f16, 5=finalNorm.
kernel void qwg_dflash_gated_rmsnorm(
        device const void* x              [[buffer(0)]],
        device const ushort* normW        [[buffer(1)]],
        device const ushort* gateUp       [[buffer(2)]],
        device const ushort* gateDown     [[buffer(3)]],
        device half* y                    [[buffer(4)]],
        device half* yRot                 [[buffer(5)]],
        device float* yRot32              [[buffer(6)]],
        constant DFlashGatedNormParams& p [[buffer(7)]],
        uint row  [[threadgroup_position_in_grid]],
        uint tid  [[thread_index_in_threadgroup]],
        uint tptg [[threads_per_threadgroup]]) {
    threadgroup float shPart[16];
    threadgroup float shH[5120];
    threadgroup float shG1[32];

    const uint sg = tid >> 5u, lane = tid & 31u, nSg = max(1u, tptg >> 5u);
    float sq = 0.0f;
    if (p.mode == 1u) {
        device const half* xr = (device const half*)x + (size_t)row * p.hidden;
        for (uint i = tid; i < p.hidden; i += tptg) { float v = float(xr[i]); shH[i] = v; sq += v * v; }
    } else {
        device const float* xr = (device const float*)x + (size_t)row * p.hidden;
        for (uint i = tid; i < p.hidden; i += tptg) { float v = xr[i]; shH[i] = v; sq += v * v; }
    }
    dflash_eval_gated_norm_stage12(
        shPart, shH, shG1, normW, gateUp, p.eps, p.hidden, p.rank, p.zeroCentered, sq, tid, tptg);

    const uint nBlk = p.hidden >> 5u;
    device half* yRow = y + (size_t)row * p.hidden;
    device half* yRotRow = yRot + (size_t)row * p.hidden;
    device float* y32Row = (p.mode == 5u) ? (yRot32 + (size_t)row * p.hidden) : nullptr;
    for (uint blk = sg; blk < nBlk; blk += nSg) {
        const uint i = (blk << 5u) + lane;
        const float yi = dflash_eval_gated_norm_elem(i, shH, shG1, gateDown, p.rank, p.gatingWeight, p.scale);
        if (p.mode == 0u) {
            yRow[i] = half(yi);
        } else if (p.mode == 1u) {
            yRow[i] = half(FC_DRAFTER_ROTATED ? dflash_mla_simd_fwht32_f32(yi, lane) : yi);
        } else if (p.mode == 5u) {
            yRow[i] = half(yi);
            const float rv = FC_HEAD_ROTATED ? dflash_mla_simd_fwht32_f32(yi, lane) : yi;
            yRotRow[i] = half(rv);
            y32Row[i] = rv;
        } else {
            yRow[i] = half(yi);
            yRotRow[i] = half(FC_DRAFTER_ROTATED ? dflash_mla_simd_fwht32_f32(yi, lane) : yi);
        }
    }
}

// Fused sandwich residual: propX += GatedNorm_post(branch16), plus optional out16 = GatedNorm_pre(propX).
kernel void qwg_dflash_gated_rmsnorm_sandwich(
        device float* propX               [[buffer(0)]],
        device const half* branch16       [[buffer(1)]],
        device const ushort* postW        [[buffer(2)]],
        device const ushort* postGU       [[buffer(3)]],
        device const ushort* postGD       [[buffer(4)]],
        device const ushort* preW         [[buffer(5)]],
        device const ushort* preGU        [[buffer(6)]],
        device const ushort* preGD        [[buffer(7)]],
        device half* outPre16             [[buffer(8)]],
        constant DFlashGatedNormParams& p [[buffer(9)]],
        uint row  [[threadgroup_position_in_grid]],
        uint tid  [[thread_index_in_threadgroup]],
        uint tptg [[threads_per_threadgroup]]) {
    threadgroup float shPart[16];
    threadgroup float shH[5120];
    threadgroup float shG1[32];

    device float* xRow = propX + (size_t)row * p.hidden;
    device const half* bRow = branch16 + (size_t)row * p.hidden;
    float sqPost = 0.0f;
    for (uint i = tid; i < p.hidden; i += tptg) { float v = float(bRow[i]); shH[i] = v; sqPost += v * v; }
    dflash_eval_gated_norm_stage12(
        shPart, shH, shG1, postW, postGU, p.eps, p.hidden, p.rank, p.zeroCentered, sqPost, tid, tptg);

    float sqPre = 0.0f;
    for (uint i = tid; i < p.hidden; i += tptg) {
        const float postY = dflash_eval_gated_norm_elem(i, shH, shG1, postGD, p.rank, p.gatingWeight, 1.0f);
        const float nx = xRow[i] + postY;
        xRow[i] = nx;
        if (p.hasNextPreNorm != 0u) { shH[i] = nx; sqPre += nx * nx; }
    }
    if (p.hasNextPreNorm == 0u) return;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    dflash_eval_gated_norm_stage12(
        shPart, shH, shG1, preW, preGU, p.eps, p.hidden, p.rank, p.zeroCentered, sqPre, tid, tptg);
    device half* outRow = outPre16 + (size_t)row * p.hidden;
    const uint sg = tid >> 5u, lane = tid & 31u, nSg = max(1u, tptg >> 5u), nBlk = p.hidden >> 5u;
    for (uint blk = sg; blk < nBlk; blk += nSg) {
        const uint i = (blk << 5u) + lane;
        float yi = dflash_eval_gated_norm_elem(i, shH, shG1, preGD, p.rank, p.gatingWeight, 1.0f);
        if (p.mode == 1u && FC_DRAFTER_ROTATED) yi = dflash_mla_simd_fwht32_f32(yi, lane);
        outRow[i] = half(yi);
    }
}

struct DFlashMLALatentParams {
    float eps, gatingWeight, alphaQ, alphaKV, ropeMscale;
    uint qLoraRank, kvLoraRank, ropeDim, gateDim;
    uint normRank, zeroCentered, ropeInterleave, posOffset, maxCacheSize;
};

// Context c_KV (kv_a_layernorm * alphaKV + fwht32) + k_pe (YaRN RoPE) -> ring slot.
kernel void qwg_dflash_mla_kv_ctx_prep(
        device const half* kvCtxIn        [[buffer(0)]],
        device const ushort* kvNormW      [[buffer(1)]],
        device const ushort* kvGateUp     [[buffer(2)]],
        device const ushort* kvGateDown   [[buffer(3)]],
        device const float* invFreq       [[buffer(4)]],
        device half* kCache               [[buffer(5)]],
        device half* vCache               [[buffer(6)]],
        constant DFlashMLALatentParams& p [[buffer(7)]],
        uint s   [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shPart[4];
    threadgroup float shH[1024];
    threadgroup float shG1[32];

    const uint pos = p.posOffset + s, slot = pos % p.maxCacheSize;
    device const half* cIn = kvCtxIn + (size_t)s * (p.kvLoraRank + p.ropeDim);
    device const half* peIn = cIn + p.kvLoraRank;
    device half* cDst = kCache + (size_t)slot * p.kvLoraRank;
    device half* peDst = vCache + (size_t)slot * p.ropeDim;

    float sq = 0.0f;
    for (uint i = tid; i < p.kvLoraRank; i += 128u) { float v = float(cIn[i]); shH[i] = v; sq += v * v; }
    dflash_eval_gated_norm_stage12(
        shPart, shH, shG1, kvNormW, kvGateUp, p.eps, p.kvLoraRank, p.normRank, p.zeroCentered, sq, tid, 128u);

    const uint sg = tid >> 5u, lane = tid & 31u, nBlk = p.kvLoraRank >> 5u;
    for (uint blk = sg; blk < nBlk; blk += 4u) {
        const uint i = (blk << 5u) + lane;
        float yi = dflash_eval_gated_norm_elem(i, shH, shG1, kvGateDown, p.normRank, p.gatingWeight, p.alphaKV);
        if (FC_DRAFTER_ROTATED) yi = dflash_mla_simd_fwht32_f32(yi, lane);
        cDst[i] = half(yi);
    }
    for (uint pairIdx = tid; pairIdx < (p.ropeDim >> 1u); pairIdx += 128u) {
        dflash_mla_rope_pair(peIn, peDst, invFreq, pos, pairIdx, p.ropeDim, p.ropeInterleave, p.ropeMscale);
    }
}

// Proposal latent prep: tg.y==0 -> q_a_layernorm*alphaQ; tg.y==1 -> kv_a_layernorm*alphaKV + k_pe RoPE.
kernel void qwg_dflash_mla_latent_prep(
        device const half* qkvProp        [[buffer(0)]],
        device const ushort* qNormW       [[buffer(1)]],
        device const ushort* qGateUp      [[buffer(2)]],
        device const ushort* qGateDown    [[buffer(3)]],
        device const ushort* kvNormW      [[buffer(4)]],
        device const ushort* kvGateUp     [[buffer(5)]],
        device const ushort* kvGateDown   [[buffer(6)]],
        device const float* invFreq       [[buffer(7)]],
        device half* mlaQLat              [[buffer(8)]],
        device half* propCKV              [[buffer(9)]],
        device half* propKPe              [[buffer(10)]],
        constant DFlashMLALatentParams& p [[buffer(11)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shPart[8];
    threadgroup float shH[2048];
    threadgroup float shG1[32];

    const uint m = tg.x, sg = tid >> 5u, lane = tid & 31u;
    const uint qkvStride = p.qLoraRank + p.kvLoraRank + p.ropeDim + p.gateDim;
    device const half* rowIn = qkvProp + (size_t)m * qkvStride;

    if (tg.y == 0u) {
        float sq = 0.0f;
        for (uint i = tid; i < p.qLoraRank; i += 256u) { float v = float(rowIn[i]); shH[i] = v; sq += v * v; }
        dflash_eval_gated_norm_stage12(
            shPart, shH, shG1, qNormW, qGateUp, p.eps, p.qLoraRank, p.normRank, p.zeroCentered, sq, tid, 256u);
        device half* qOut = mlaQLat + (size_t)m * p.qLoraRank;
        for (uint blk = sg; blk < (p.qLoraRank >> 5u); blk += 8u) {
            const uint i = (blk << 5u) + lane;
            float yi = dflash_eval_gated_norm_elem(i, shH, shG1, qGateDown, p.normRank, p.gatingWeight, p.alphaQ);
            if (FC_DRAFTER_ROTATED) yi = dflash_mla_simd_fwht32_f32(yi, lane);
            qOut[i] = half(yi);
        }
    } else {
        device const half* kvIn = rowIn + p.qLoraRank;
        device const half* peIn = kvIn + p.kvLoraRank;
        float sq = 0.0f;
        for (uint i = tid; i < p.kvLoraRank; i += 256u) { float v = float(kvIn[i]); shH[i] = v; sq += v * v; }
        dflash_eval_gated_norm_stage12(
            shPart, shH, shG1, kvNormW, kvGateUp, p.eps, p.kvLoraRank, p.normRank, p.zeroCentered, sq, tid, 256u);
        device half* cOut = propCKV + (size_t)m * p.kvLoraRank;
        for (uint blk = sg; blk < (p.kvLoraRank >> 5u); blk += 8u) {
            const uint i = (blk << 5u) + lane;
            float yi = dflash_eval_gated_norm_elem(i, shH, shG1, kvGateDown, p.normRank, p.gatingWeight, p.alphaKV);
            if (FC_DRAFTER_ROTATED) yi = dflash_mla_simd_fwht32_f32(yi, lane);
            cOut[i] = half(yi);
        }
        device half* peOut = propKPe + (size_t)m * p.ropeDim;
        for (uint pairIdx = tid; pairIdx < (p.ropeDim >> 1u); pairIdx += 256u) {
            dflash_mla_rope_pair(peIn, peOut, invFreq, p.posOffset + m, pairIdx, p.ropeDim, p.ropeInterleave, p.ropeMscale);
        }
    }
}

struct DFlashMLAQBParams {
    uint qHeads, nopeDim, ropeDim, vHeadDim, kvLoraRank, M, posOffset, ropeInterleave;
    float ropeMscale;
    uint quantKind, bits, group, qkvStride, gateOffset, gatedAttention, pad;
};

template <bool SUB2>
static inline void dflash_mla_absorb_impl(
        device half* mlaQFull, device const char* kvBBlocks, device const float* invFreq,
        device half* qAbs, constant DFlashMLAQBParams& p,
        threadgroup half (&shQ)[8][256], threadgroup ShvqMlaTgLut& tgLut,
        uint2 tg, uint tid, uint tptg) {
    const uint nSG = max(1u, tptg >> 5u);
    const uint sg = tid >> 5u, lane = tid & 31u;
    const uint colBlk = tg.x * nSG + sg, qh = tg.y, validM = min(p.M, 8u);
    const uint qFullStride = p.qHeads * (p.nopeDim + p.ropeDim);
    const uint qRopeBase = p.qHeads * p.nopeDim + qh * p.ropeDim;

    if (tg.x == 0u && p.ropeDim > 0u) {
        const uint halfRope = p.ropeDim >> 1u;
        for (uint idx = tid; idx < validM * halfRope; idx += tptg) {
            const uint m = idx / halfRope, pairIdx = idx - m * halfRope;
            device half* rPtr = mlaQFull + (size_t)m * qFullStride + qRopeBase;
            dflash_mla_rope_pair(rPtr, rPtr, invFreq, p.posOffset + m, pairIdx, p.ropeDim, p.ropeInterleave, p.ropeMscale);
        }
    }
    if (p.quantKind == 2u) (void)qw_mla_load_shvq_lut<SUB2>(tgLut, p.bits, tid, tptg);
    for (uint idx = tid; idx < validM * p.nopeDim; idx += tptg) {
        const uint m = idx / p.nopeDim, r = idx - m * p.nopeDim;
        shQ[m][r] = mlaQFull[(size_t)m * qFullStride + qh * p.nopeDim + r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg >= nSG) return;
    const uint col0 = colBlk * 128u + lane * 4u;
    if (col0 >= p.kvLoraRank) return;
    const uint headRowBase = qh * (p.nopeDim + p.vHeadDim);
    float4 acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    if (p.quantKind == 3u) {
        device const ushort* bPtr = (device const ushort*)kvBBlocks + (size_t)headRowBase * p.kvLoraRank + col0;
        for (uint r = 0u; r < p.nopeDim; ++r, bPtr += p.kvLoraRank) {
            const float4 w4 = float4(
                dflash_bf16_to_f32(bPtr[0]), dflash_bf16_to_f32(bPtr[1]),
                dflash_bf16_to_f32(bPtr[2]), dflash_bf16_to_f32(bPtr[3]));
            #pragma unroll
            for (uint m = 0u; m < 8u; ++m) if (m < validM) acc[m] = fma(float4(float(shQ[m][r])), w4, acc[m]);
        }
    } else if (p.quantKind == 1u) {
        const uint blk = col0 >> 5u, idx = col0 & 31u, wIdx = idx >> 3u, blocksPerRow = p.kvLoraRank >> 5u;
        const ushort baseShift = ushort((idx & 4u) << 1u);
        device const Block_HQ4_G32* bPtr = (device const Block_HQ4_G32*)kvBBlocks + (size_t)headRowBase * blocksPerRow + blk;
        for (uint r = 0u; r < p.nopeDim; ++r, bPtr += blocksPerRow) {
            const half2 s2 = half2(bPtr->scale), b2 = half2(bPtr->bias);
            const ushort2 rsh = as_type<ushort2>(bPtr->qs[wIdx]) >> baseShift;
            const half2 w01 = fma(as_type<half2>((rsh & ushort2(0x000Fu)) | ushort2(0x6400u)), s2, b2);
            const half2 w23 = fma(as_type<half2>(((rsh >> 4u) & ushort2(0x000Fu)) | ushort2(0x6400u)), s2, b2);
            const float4 w4 = float4(float(w01.x), float(w01.y), float(w23.x), float(w23.y));
            #pragma unroll
            for (uint m = 0u; m < 8u; ++m) if (m < validM) acc[m] = fma(float4(float(shQ[m][r])), w4, acc[m]);
        }
    } else if (p.quantKind == 2u) {
        const uint nGroups = p.kvLoraRank / p.group;
        const uint bytesPerGroup = qwg_shvq_group_bytes(int(p.bits), int(p.group));
        const size_t rowBytes = (size_t)nGroups * bytesPerGroup;
        const uint gIdx = col0 / p.group, quadInGroup = (col0 - gIdx * p.group) >> 2u;
        device const char* gp = kvBBlocks + ((size_t)headRowBase * nGroups + gIdx) * bytesPerGroup;
        for (uint r = 0u; r < p.nopeDim; ++r, gp += rowBytes) {
            const float sc = float(*(device const half*)gp);
            const float4 w4 = float4(qw_mla_shvq_quad4<SUB2>((device const uchar*)(gp + 2), quadInGroup, p.bits, tgLut)) * sc;
            #pragma unroll
            for (uint m = 0u; m < 8u; ++m) if (m < validM) acc[m] = fma(float4(float(shQ[m][r])), w4, acc[m]);
        }
    } else {
        const uint nGroups = p.kvLoraRank / p.group, bytesPerGroup = 4u + ((p.group * p.bits) >> 3u);
        const size_t rowBytes = (size_t)nGroups * bytesPerGroup;
        const uint gIdx = col0 / p.group;
        const int inGroup = int(col0 - gIdx * p.group), bitsI = int(p.bits);
        device const char* gp = kvBBlocks + ((size_t)headRowBase * nGroups + gIdx) * bytesPerGroup;
        for (uint r = 0u; r < p.nopeDim; ++r, gp += rowBytes) {
            const float sc = float(*(device const half*)gp), bi = float(*(device const half*)(gp + 2));
            device const uchar* pay = (device const uchar*)(gp + 4);
            const float4 q4 = float4(
                float(qwg_aq_read_bits(pay, bitsI, inGroup + 0)), float(qwg_aq_read_bits(pay, bitsI, inGroup + 1)),
                float(qwg_aq_read_bits(pay, bitsI, inGroup + 2)), float(qwg_aq_read_bits(pay, bitsI, inGroup + 3)));
            const float4 w4 = fma(q4, float4(sc), float4(bi));
            #pragma unroll
            for (uint m = 0u; m < 8u; ++m) if (m < validM) acc[m] = fma(float4(float(shQ[m][r])), w4, acc[m]);
        }
    }
    for (uint m = 0u; m < validM; ++m) {
        *(device half4*)(qAbs + ((size_t)m * p.qHeads + qh) * p.kvLoraRank + col0) = half4(acc[m]);
    }
}

// Fused q_rope YaRN RoPE + q_nope @ W_UK absorption (SHVQ{2,3,4}, HQ4_G32, AQ, BF16).
kernel void qwg_dflash_mla_q_absorb(
        device half* mlaQFull         [[buffer(0)]],
        device const char* kvBBlocks  [[buffer(1)]],
        device const float* invFreq   [[buffer(2)]],
        device half* qAbs             [[buffer(3)]],
        constant DFlashMLAQBParams& p [[buffer(4)]],
        uint2 tg    [[threadgroup_position_in_grid]],
        uint  tid   [[thread_index_in_threadgroup]],
        uint2 tptg2 [[threads_per_threadgroup]]) {
    threadgroup half shQ[8][256];
    threadgroup ShvqMlaTgLut tgLut;
    dflash_mla_absorb_impl<false>(mlaQFull, kvBBlocks, invFreq, qAbs, p, shQ, tgLut, tg, tid, max(32u, tptg2.x));
}

// Sub-2-bit SHVQ{1,15} specialization of qwg_dflash_mla_q_absorb.
kernel void qwg_dflash_mla_q_absorb_sub2(
        device half* mlaQFull         [[buffer(0)]],
        device const char* kvBBlocks  [[buffer(1)]],
        device const float* invFreq   [[buffer(2)]],
        device half* qAbs             [[buffer(3)]],
        constant DFlashMLAQBParams& p [[buffer(4)]],
        uint2 tg    [[threadgroup_position_in_grid]],
        uint  tid   [[thread_index_in_threadgroup]],
        uint2 tptg2 [[threads_per_threadgroup]]) {
    threadgroup half shQ[8][256];
    threadgroup ShvqMlaTgLut tgLut;
    dflash_mla_absorb_impl<true>(mlaQFull, kvBBlocks, invFreq, qAbs, p, shQ, tgLut, tg, tid, max(32u, tptg2.x));
}

struct DFlashMLAAttnParams {
    uint qHeads, nopeDim, ropeDim, kvLoraRank, newOffset, M, maxCacheSize, isCausal;
    float softmaxScale;
    uint pad0, pad1, pad2;
};

// Matrix-absorbed MLA context + block attention -> uLat [M, qHeads, kvLoraRank].
kernel void qwg_dflash_mla_attn(
        device const half* qAbs         [[buffer(0)]],
        device const half* mlaQFull     [[buffer(1)]],
        device const half* kCache       [[buffer(2)]],
        device const half* vCache       [[buffer(3)]],
        device const half* propCKV      [[buffer(4)]],
        device const half* propKPe      [[buffer(5)]],
        device half* uLat               [[buffer(6)]],
        constant DFlashMLAAttnParams& p [[buffer(7)]],
        uint2 tg    [[threadgroup_position_in_grid]],
        uint  sg    [[simdgroup_index_in_threadgroup]],
        uint  lane  [[thread_index_in_simdgroup]],
        uint2 tptg2 [[threads_per_threadgroup]]) {
    threadgroup float shMax[8];
    threadgroup float shL[8];
    threadgroup float4 shO[7][128];

    const uint nSG = min(8u, max(1u, tptg2.x >> 5u));
    if (sg >= nSG) return;
    const uint qh = tg.x, m = tg.y;
    const uint kvChunks = min(4u, p.kvLoraRank >> 7u), ropePerLane = min(4u, p.ropeDim >> 5u);
    const uint qFullStride = p.qHeads * (p.nopeDim + p.ropeDim);
    device const half* qAbsRow = qAbs + ((size_t)m * p.qHeads + qh) * p.kvLoraRank;
    device const half* qRopeRow = mlaQFull + (size_t)m * qFullStride + p.qHeads * p.nopeDim + qh * p.ropeDim;

    float4 qAbsReg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float qRopeReg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint c = 0u; c < kvChunks; ++c) {
        qAbsReg[c] = float4(*(device const half4*)(qAbsRow + c * 128u + lane * 4u)) * p.softmaxScale;
    }
    for (uint r = 0u; r < ropePerLane; ++r) {
        qRopeReg[r] = float(qRopeRow[r * 32u + lane]) * p.softmaxScale;
    }

    float maxLogit = -3.4e38f, lsum = 0.0f;
    bool hasLogit = false;
    float4 oAcc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const uint absQ = p.newOffset + m;
    const uint startK = (absQ > p.maxCacheSize) ? (absQ - p.maxCacheSize) : 0u;
    const uint ctxLen = (p.newOffset > startK) ? (p.newOffset - startK) : 0u;

    if (nSG == 1u) {
        for (uint absK = startK; absK < p.newOffset; ++absK) {
            const uint slot = absK % p.maxCacheSize;
            device const half* ckv = kCache + (size_t)slot * p.kvLoraRank;
            device const half* kpe = vCache + (size_t)slot * p.ropeDim;
            float4 ck[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float dotPart = 0.0f;
            for (uint c = 0u; c < kvChunks; ++c) {
                ck[c] = float4(*(device const half4*)(ckv + c * 128u + lane * 4u));
                dotPart += dot(qAbsReg[c], ck[c]);
            }
            for (uint r = 0u; r < ropePerLane; ++r) dotPart += qRopeReg[r] * float(kpe[r * 32u + lane]);
            const float s = simd_sum(dotPart);
            const float newMax = hasLogit ? max(maxLogit, s) : s;
            const float corr = hasLogit ? exp(maxLogit - newMax) : 0.0f, prob = exp(s - newMax);
            lsum = lsum * corr + prob;
            for (uint c = 0u; c < kvChunks; ++c) oAcc[c] = oAcc[c] * corr + ck[c] * prob;
            maxLogit = newMax;
            hasLogit = true;
        }
    } else {
        uint kRel = sg * 4u;
        const uint stride4 = nSG * 4u;
        for (; kRel + 3u < ctxLen; kRel += stride4) {
            const uint s0 = (startK + kRel + 0u) % p.maxCacheSize;
            const uint s1 = (startK + kRel + 1u) % p.maxCacheSize;
            const uint s2 = (startK + kRel + 2u) % p.maxCacheSize;
            const uint s3 = (startK + kRel + 3u) % p.maxCacheSize;
            float4 ck0[4], ck1[4], ck2[4], ck3[4];
            float4 dot4 = float4(0.0f);
            for (uint c = 0u; c < kvChunks; ++c) {
                const uint off = c * 128u + lane * 4u;
                ck0[c] = float4(*(device const half4*)(kCache + (size_t)s0 * p.kvLoraRank + off));
                ck1[c] = float4(*(device const half4*)(kCache + (size_t)s1 * p.kvLoraRank + off));
                ck2[c] = float4(*(device const half4*)(kCache + (size_t)s2 * p.kvLoraRank + off));
                ck3[c] = float4(*(device const half4*)(kCache + (size_t)s3 * p.kvLoraRank + off));
                dot4 += float4(dot(qAbsReg[c], ck0[c]), dot(qAbsReg[c], ck1[c]),
                               dot(qAbsReg[c], ck2[c]), dot(qAbsReg[c], ck3[c]));
            }
            for (uint r = 0u; r < ropePerLane; ++r) {
                const uint roff = r * 32u + lane;
                dot4 += qRopeReg[r] * float4(float(vCache[(size_t)s0 * p.ropeDim + roff]),
                                             float(vCache[(size_t)s1 * p.ropeDim + roff]),
                                             float(vCache[(size_t)s2 * p.ropeDim + roff]),
                                             float(vCache[(size_t)s3 * p.ropeDim + roff]));
            }
            const float4 s4 = simd_sum(dot4);
            const float blkMax = max(max(s4.x, s4.y), max(s4.z, s4.w));
            const float newMax = hasLogit ? max(maxLogit, blkMax) : blkMax;
            const float corr = hasLogit ? exp(maxLogit - newMax) : 0.0f;
            const float4 p4 = exp(s4 - newMax);
            lsum = lsum * corr + (p4.x + p4.y + p4.z + p4.w);
            for (uint c = 0u; c < kvChunks; ++c) {
                oAcc[c] = oAcc[c] * corr + ck0[c] * p4.x + ck1[c] * p4.y + ck2[c] * p4.z + ck3[c] * p4.w;
            }
            maxLogit = newMax;
            hasLogit = true;
        }
        for (uint rem = kRel; rem < ctxLen && rem < kRel + 4u; ++rem) {
            const uint slot = (startK + rem) % p.maxCacheSize;
            device const half* ckv = kCache + (size_t)slot * p.kvLoraRank;
            device const half* kpe = vCache + (size_t)slot * p.ropeDim;
            float4 ck[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float dotPart = 0.0f;
            for (uint c = 0u; c < kvChunks; ++c) {
                ck[c] = float4(*(device const half4*)(ckv + c * 128u + lane * 4u));
                dotPart += dot(qAbsReg[c], ck[c]);
            }
            for (uint r = 0u; r < ropePerLane; ++r) dotPart += qRopeReg[r] * float(kpe[r * 32u + lane]);
            const float s = simd_sum(dotPart);
            const float newMax = hasLogit ? max(maxLogit, s) : s;
            const float corr = hasLogit ? exp(maxLogit - newMax) : 0.0f, prob = exp(s - newMax);
            lsum = lsum * corr + prob;
            for (uint c = 0u; c < kvChunks; ++c) oAcc[c] = oAcc[c] * corr + ck[c] * prob;
            maxLogit = newMax;
            hasLogit = true;
        }
    }

    const uint pLimit = (p.isCausal != 0u) ? (m + 1u) : p.M;
    for (uint pIdx = sg; pIdx < pLimit; pIdx += nSG) {
        device const half* ckv = propCKV + (size_t)pIdx * p.kvLoraRank;
        device const half* kpe = propKPe + (size_t)pIdx * p.ropeDim;
        float4 ck[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float dotPart = 0.0f;
        for (uint c = 0u; c < kvChunks; ++c) {
            ck[c] = float4(*(device const half4*)(ckv + c * 128u + lane * 4u));
            dotPart += dot(qAbsReg[c], ck[c]);
        }
        for (uint r = 0u; r < ropePerLane; ++r) dotPart += qRopeReg[r] * float(kpe[r * 32u + lane]);
        const float s = simd_sum(dotPart);
        const float newMax = hasLogit ? max(maxLogit, s) : s;
        const float corr = hasLogit ? exp(maxLogit - newMax) : 0.0f, prob = exp(s - newMax);
        lsum = lsum * corr + prob;
        for (uint c = 0u; c < kvChunks; ++c) oAcc[c] = oAcc[c] * corr + ck[c] * prob;
        maxLogit = newMax;
        hasLogit = true;
    }

    device half* uOut = uLat + ((size_t)m * p.qHeads + qh) * p.kvLoraRank;
    if (nSG == 1u) {
        const float invL = (lsum > 0.0f) ? (1.0f / lsum) : 0.0f;
        for (uint c = 0u; c < kvChunks; ++c) {
            *(device half4*)(uOut + c * 128u + lane * 4u) = half4(oAcc[c] * invL);
        }
        return;
    }

    if (lane == 0u) {
        shMax[sg] = hasLogit ? maxLogit : -3.4e38f;
        shL[sg] = hasLogit ? lsum : 0.0f;
    }
    if (sg > 0u) {
        for (uint c = 0u; c < kvChunks; ++c) shO[sg - 1u][c * 32u + lane] = oAcc[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        float gMax = -3.4e38f;
        for (uint s = 0u; s < nSG; ++s) if (shL[s] > 0.0f) gMax = max(gMax, shMax[s]);
        float gL = 0.0f;
        float4 gO[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint s = 0u; s < nSG; ++s) {
            const float ls = shL[s];
            if (ls > 0.0f) {
                const float w = exp(shMax[s] - gMax);
                gL += ls * w;
                if (s == 0u) {
                    for (uint c = 0u; c < kvChunks; ++c) gO[c] += oAcc[c] * w;
                } else {
                    for (uint c = 0u; c < kvChunks; ++c) gO[c] += shO[s - 1u][c * 32u + lane] * w;
                }
            }
        }
        const float invL = (gL > 0.0f) ? (1.0f / gL) : 0.0f;
        for (uint c = 0u; c < kvChunks; ++c) {
            *(device half4*)(uOut + c * 128u + lane * 4u) = half4(gO[c] * invL);
        }
    }
}

template <bool SUB2>
static inline void dflash_mla_unpack_wuv_slice32(
        device const char* kvBBlocks, uint row, uint col0, constant DFlashMLAQBParams& p,
        threadgroup const ShvqMlaTgLut& tgLut, thread float4 (&wLo)[4], thread float4 (&wHi)[4]) {
    if (p.quantKind == 3u) {
        device const ushort* rp = (device const ushort*)kvBBlocks + (size_t)row * p.kvLoraRank + col0;
        #pragma unroll
        for (uint w = 0u; w < 4u; ++w) {
            const uint b = w * 8u;
            wLo[w] = float4(dflash_bf16_to_f32(rp[b + 0u]), dflash_bf16_to_f32(rp[b + 1u]),
                            dflash_bf16_to_f32(rp[b + 2u]), dflash_bf16_to_f32(rp[b + 3u]));
            wHi[w] = float4(dflash_bf16_to_f32(rp[b + 4u]), dflash_bf16_to_f32(rp[b + 5u]),
                            dflash_bf16_to_f32(rp[b + 6u]), dflash_bf16_to_f32(rp[b + 7u]));
        }
    } else if (p.quantKind == 1u) {
        const Block_HQ4_G32 blk = *((device const Block_HQ4_G32*)kvBBlocks + (size_t)row * (p.kvLoraRank >> 5u) + (col0 >> 5u));
        const half2 s2 = half2(blk.scale), b2 = half2(blk.bias);
        #pragma unroll
        for (uint w = 0u; w < 4u; ++w) {
            half2 a01, a23, a45, a67;
            unpack_hq4_word8(blk.qs[w], s2, b2, a01, a23, a45, a67);
            wLo[w] = float4(float(a01.x), float(a01.y), float(a23.x), float(a23.y));
            wHi[w] = float4(float(a45.x), float(a45.y), float(a67.x), float(a67.y));
        }
    } else if (p.quantKind == 2u) {
        const uint nGroups = p.kvLoraRank / p.group;
        const uint bytesPerGroup = qwg_shvq_group_bytes(int(p.bits), int(p.group));
        const uint gIdx = col0 / p.group, inGroup = col0 - gIdx * p.group;
        device const char* gp = kvBBlocks + ((size_t)row * nGroups + gIdx) * bytesPerGroup;
        const float sc = float(*(device const half*)gp);
        device const uchar* pay = (device const uchar*)(gp + 2 + (inGroup >> 5u) * qwg_shvq_slice_bytes(int(p.bits)));
        #pragma unroll
        for (uint w = 0u; w < 4u; ++w) {
            half4 uLo, uHi;
            qw_mla_shvq_oct8<SUB2>(pay, w, p.bits, tgLut, uLo, uHi);
            wLo[w] = float4(uLo) * sc;
            wHi[w] = float4(uHi) * sc;
        }
    } else {
        const uint nGroups = p.kvLoraRank / p.group, bytesPerGroup = 4u + ((p.group * p.bits) >> 3u);
        const uint gIdx = col0 / p.group, inGroup = col0 - gIdx * p.group;
        device const char* gp = kvBBlocks + ((size_t)row * nGroups + gIdx) * bytesPerGroup;
        const float sc = float(*(device const half*)gp), bi = float(*(device const half*)(gp + 2));
        device const uchar* pay = (device const uchar*)(gp + 4);
        #pragma unroll
        for (uint w = 0u; w < 4u; ++w) {
            const uint base = inGroup + w * 8u;
            wLo[w] = float4(
                fma(float(qwg_aq_read_bits(pay, int(p.bits), int(base + 0u))), sc, bi),
                fma(float(qwg_aq_read_bits(pay, int(p.bits), int(base + 1u))), sc, bi),
                fma(float(qwg_aq_read_bits(pay, int(p.bits), int(base + 2u))), sc, bi),
                fma(float(qwg_aq_read_bits(pay, int(p.bits), int(base + 3u))), sc, bi));
            wHi[w] = float4(
                fma(float(qwg_aq_read_bits(pay, int(p.bits), int(base + 4u))), sc, bi),
                fma(float(qwg_aq_read_bits(pay, int(p.bits), int(base + 5u))), sc, bi),
                fma(float(qwg_aq_read_bits(pay, int(p.bits), int(base + 6u))), sc, bi),
                fma(float(qwg_aq_read_bits(pay, int(p.bits), int(base + 7u))), sc, bi));
        }
    }
}

template <bool SUB2>
static inline void dflash_mla_proj_uv_gate_impl(
        device const half* uLat, device const char* kvBBlocks, device const half* qkvProp,
        device half* attnOut, constant DFlashMLAQBParams& p,
        threadgroup half (&shU)[8][512], threadgroup float (&shOut)[8][32],
        threadgroup ShvqMlaTgLut& tgLut, uint2 tg, uint tid) {
    const uint vBlk = tg.x, qh = tg.y, validM = min(p.M, 8u), sg = tid >> 5u, lane = tid & 31u;
    if (p.quantKind == 2u) (void)qw_mla_load_shvq_lut<SUB2>(tgLut, p.bits, tid, 128u);
    for (uint idx = tid; idx < validM * p.kvLoraRank; idx += 128u) {
        const uint m = idx / p.kvLoraRank, c = idx - m * p.kvLoraRank;
        shU[m][c] = uLat[((size_t)m * p.qHeads + qh) * p.kvLoraRank + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint subLane = lane & 15u, halfWarp = lane >> 4u, nSlices = p.kvLoraRank >> 5u;
    const uint headRowBase = qh * (p.nopeDim + p.vHeadDim) + p.nopeDim;
    for (uint rPair = 0u; rPair < 4u; ++rPair) {
        const uint localV = sg * 8u + rPair * 2u + halfWarp;
        const uint row = headRowBase + vBlk * 32u + localV;
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint sIdx = subLane; sIdx < nSlices; sIdx += 16u) {
            const uint col0 = sIdx << 5u;
            float4 wLo[4], wHi[4];
            dflash_mla_unpack_wuv_slice32<SUB2>(kvBBlocks, row, col0, p, tgLut, wLo, wHi);
            #pragma unroll
            for (uint m = 0u; m < 8u; ++m) if (m < validM) {
                float d = 0.0f;
                #pragma unroll
                for (uint w = 0u; w < 4u; ++w) {
                    const uint b = col0 + w * 8u;
                    const float4 uLo = float4(*(threadgroup const half4*)(&shU[m][b + 0u]));
                    const float4 uHi = float4(*(threadgroup const half4*)(&shU[m][b + 4u]));
                    d += dot(wLo[w], uLo) + dot(wHi[w], uHi);
                }
                acc[m] += d;
            }
        }
        #pragma unroll
        for (ushort s = 1; s <= 8; s <<= 1) {
            #pragma unroll
            for (uint m = 0u; m < 8u; ++m) if (m < validM) acc[m] += simd_shuffle_xor(acc[m], s);
        }
        if (subLane == 0u) {
            for (uint m = 0u; m < validM; ++m) shOut[m][localV] = acc[m];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint outDim = p.qHeads * p.vHeadDim, outCol = qh * p.vHeadDim + vBlk * 32u + lane;
    for (uint m = sg; m < validM; m += 4u) {
        float val = shOut[m][lane];
        if (p.gatedAttention != 0u) {
            const float g = float(qkvProp[(size_t)m * p.qkvStride + p.gateOffset + outCol]);
            val *= (1.0f / (1.0f + exp(-g)));
        }
        if (FC_DRAFTER_ROTATED) val = dflash_mla_simd_fwht32_f32(val, lane);
        attnOut[(size_t)m * outDim + outCol] = half(val);
    }
}

// Fused uLat @ W_UV^T + FP32 sigmoid(gate) + FC_DRAFTER_ROTATED fwht32 -> attnOut [M, qHeads * vHeadDim].
kernel void qwg_dflash_mla_proj_uv_gate(
        device const half* uLat       [[buffer(0)]],
        device const char* kvBBlocks  [[buffer(1)]],
        device const half* qkvProp    [[buffer(2)]],
        device half* attnOut          [[buffer(3)]],
        constant DFlashMLAQBParams& p [[buffer(4)]],
        uint2 tg  [[threadgroup_position_in_grid]],
        uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup half shU[8][512];
    threadgroup float shOut[8][32];
    threadgroup ShvqMlaTgLut tgLut;
    dflash_mla_proj_uv_gate_impl<false>(uLat, kvBBlocks, qkvProp, attnOut, p, shU, shOut, tgLut, tg, tid);
}

// Sub-2-bit SHVQ{1,15} specialization of qwg_dflash_mla_proj_uv_gate.
kernel void qwg_dflash_mla_proj_uv_gate_sub2(
        device const half* uLat       [[buffer(0)]],
        device const char* kvBBlocks  [[buffer(1)]],
        device const half* qkvProp    [[buffer(2)]],
        device half* attnOut          [[buffer(3)]],
        constant DFlashMLAQBParams& p [[buffer(4)]],
        uint2 tg  [[threadgroup_position_in_grid]],
        uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup half shU[8][512];
    threadgroup float shOut[8][32];
    threadgroup ShvqMlaTgLut tgLut;
    dflash_mla_proj_uv_gate_impl<true>(uLat, kvBBlocks, qkvProp, attnOut, p, shU, shOut, tgLut, tg, tid);
}
