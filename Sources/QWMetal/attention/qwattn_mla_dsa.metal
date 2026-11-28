// qwattn_mla_dsa.metal — Decoupled-RoPE + Dynamic Sparse Attention (DSA) MLA execution kernels:
// scalar DSA sparse attention, head gate projection, 8x8 MMA VHA + sigmoid(o_gate) + H_32 exit,
// and FlashIndexer head selection (latent/RoPE prep lives in qwattn_mla_dsa_prep.metal).
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "../include/qw_mla_dequant.metalh"
#include "../include/qw_geometry_constants.metalh"
using namespace metal;

struct MLAVHAGateParams {
    uint T;
    uint qHeads;
    uint vHeadDim;
    uint gateStride;
    uint oGateOffset;
    uint hasVHA;
    uint hasOGate;
};

struct MLAHeadGateProjParams {
    uint T;
    uint hidden;
    uint nGate;
};

METAL_FUNC float qw_mla_read_kv_scalar(device const uchar* poolBytes,
                                       uint pageID, uint pageOff,
                                       uint halfRank, uint col, uint kvMode) {
    if (kvMode == 0u) {
        return float(((device const half*)poolBytes)[((size_t)pageID * 256u + pageOff) * halfRank + col]);
    } else if (kvMode == 1u) {
        const size_t dataBytes = 256ul * halfRank;
        const size_t pageBytes = dataBytes + 512ul;
        device const uchar* pageBase = poolBytes + (size_t)pageID * pageBytes;
        const half sc = *((device const half*)(pageBase + dataBytes + (size_t)pageOff * 2ul));
        return float(((device const char*)(pageBase + (size_t)pageOff * halfRank))[col]) * float(sc);
    } else {
        const uint groupsPerHalf = max(1u, halfRank >> 7u);
        const size_t dataBytes = 128ul * halfRank;
        const size_t pageBytes = dataBytes + 256ul * groupsPerHalf * 4ul;
        device const uchar* pageBase = poolBytes + (size_t)pageID * pageBytes;
        const half2 sm = *((device const half2*)(pageBase + dataBytes) + (size_t)pageOff * groupsPerHalf + (col >> 7u));
        const uchar b = (pageBase + (size_t)pageOff * (halfRank >> 1u))[col >> 1u];
        const uint nib = ((col & 1u) == 0u) ? uint(b & 0x0Fu) : uint(b >> 4u);
        return fma(float(nib), float(sm.x), float(sm.y));
    }
}

// 3. O(1)-SRAM streaming scalar Decoupled-RoPE MLA attention (supports arbitrary seqLen & kvLoraRank <= 1024).
// Grid: (qHeads * T, 1, 1), Threadgroup: (256, 1, 1).
kernel void qwg_mla_dsa_sparse_attn(
    device const half*              qAbs      [[buffer(0)]],
    device const half*              kPool     [[buffer(1)]],
    device const half*              vPool     [[buffer(2)]],
    device const int*               kTable    [[buffer(3)]],
    device const int*               vTable    [[buffer(4)]],
    device const int*               blockList [[buffer(5)]],
    device half*                    uLatOut   [[buffer(6)]],
    constant MLADSASparseAttnParams& p        [[buffer(7)]],
    device const half*              mlaQFull  [[buffer(8)]],
    device const half*              kPeCache  [[buffer(9)]],
    device const float*             attnSink  [[buffer(10)]],
    uint                            tg        [[threadgroup_position_in_grid]],
    uint                            d         [[thread_index_in_threadgroup]],
    uint                            sg        [[simdgroup_index_in_threadgroup]],
    uint                            lane      [[thread_index_in_simdgroup]]
) {
    const uint t = tg / p.qHeads;
    const uint h = tg - t * p.qHeads;
    if (t >= p.T) return;

    threadgroup float shQAbs[1024];
    threadgroup float shQRope[128];
    threadgroup float shScore[8];
    threadgroup uint  shTokJ[8];

    const uint seqLen = p.pos0 + t + 1u;
    const bool isDenseStream = (p.topK == 0u);
    const uint startTok = isDenseStream
        ? ((p.windowTokens > 0u && seqLen > p.windowTokens) ? (seqLen - p.windowTokens) : 0u)
        : 0u;
    const uint denseLim = p.topK + p.windowTokens;
    const uint slots = isDenseStream ? (seqLen - startTok) : min(min(seqLen, denseLim), 2560u);
    const float scale = (p.softmaxScale > 0.0f) ? p.softmaxScale : rsqrt(float(p.headDim));
    const uint halfRank = p.kvLoraRank >> 1u;
    const uint kChunks = min(4u, (p.kvLoraRank + 255u) >> 8u);

    device const half* qRow = qAbs + ((size_t)t * p.qHeads + h) * p.kvLoraRank;
    for (uint k = d; k < p.kvLoraRank; k += 256u) shQAbs[k] = float(qRow[k]);
    device const half* qrRow = mlaQFull + (size_t)t * p.qFullStride + p.qRopeOffset + h * p.ropeDim;
    for (uint r = d; r < p.ropeDim; r += 256u) shQRope[r] = float(qrRow[r]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -1e30f;
    float lsum = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const int* rowList = blockList + (size_t)t * p.listStride;

    for (uint blk = 0u; blk < slots; blk += 8u) {
        const uint nG = min(8u, slots - blk);
        const uint slot = blk + sg;
        if (slot < slots) {
            const uint j = isDenseStream
                ? (startTok + slot)
                : ((seqLen <= denseLim) ? slot : min(uint(max(rowList[slot], 0)), seqLen - 1u));
            const uint pageSlot = j >> 8u, pageOff = j & 255u;
            const uint pageK = uint(kTable[pageSlot]), pageV = uint(vTable[pageSlot]);
            float s0 = 0.0f;
            for (uint col = lane; col < halfRank; col += 32u) {
                s0 = fma(shQAbs[col], qw_mla_read_kv_scalar((device const uchar*)kPool, pageK, pageOff, halfRank, col, p.kvMode), s0);
                s0 = fma(shQAbs[halfRank + col], qw_mla_read_kv_scalar((device const uchar*)vPool, pageV, pageOff, halfRank, col, p.kvMode), s0);
            }
            if (p.ropeDim > 0u) {
                device const half* kp = kPeCache + (size_t)j * p.ropeDim;
                for (uint r = lane; r < p.ropeDim; r += 32u) {
                    s0 = fma(shQRope[r], float(kp[r]), s0);
                }
            }
            const float dotVal = simd_sum(s0) * scale;
            if (lane == 0u) {
                shScore[sg] = dotVal;
                shTokJ[sg] = j;
            }
        } else if (lane == 0u) {
            shScore[sg] = -1e30f;
            shTokJ[sg] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float mNew = m;
        for (uint g = 0u; g < nG; ++g) mNew = max(mNew, shScore[g]);
        const float corr = (m > -1e29f && mNew > m) ? exp(m - mNew) : ((m > -1e29f) ? 1.0f : 0.0f);
        lsum *= corr;
        for (uint cChunk = 0u; cChunk < kChunks; ++cChunk) acc[cChunk] *= corr;
        for (uint g = 0u; g < nG; ++g) {
            const float w = exp(shScore[g] - mNew);
            lsum += w;
            const uint j = shTokJ[g];
            const uint pageSlot = j >> 8u, pageOff = j & 255u;
            for (uint cChunk = 0u; cChunk < kChunks; ++cChunk) {
                const uint col = cChunk * 256u + d;
                if (col < p.kvLoraRank) {
                    const uint pg = (col < halfRank) ? uint(kTable[pageSlot]) : uint(vTable[pageSlot]);
                    device const uchar* pl = (col < halfRank) ? (device const uchar*)kPool : (device const uchar*)vPool;
                    const uint pc = (col < halfRank) ? col : (col - halfRank);
                    acc[cChunk] = fma(w, qw_mla_read_kv_scalar(pl, pg, pageOff, halfRank, pc, p.kvMode), acc[cChunk]);
                }
            }
        }
        m = mNew;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (p.hasSink != 0u) {
        const float sk = attnSink[h];
        if (sk > -1e29f) {
            const float mFinal = max(m, sk);
            const float corr = (m > -1e29f) ? exp(m - mFinal) : 0.0f;
            for (uint cChunk = 0u; cChunk < kChunks; ++cChunk) acc[cChunk] *= corr;
            lsum = lsum * corr + exp(sk - mFinal);
        }
    }
    const float invL = (lsum > 0.0f) ? (1.0f / lsum) : 0.0f;
    device half* dstU = uLatOut + ((size_t)t * p.qHeads + h) * p.kvLoraRank;
    for (uint cChunk = 0u; cChunk < kChunks; ++cChunk) {
        const uint col = cChunk * 256u + d;
        if (col < p.kvLoraRank) dstU[col] = half(acc[cChunk] * invL);
    }
}

// 4a. Direct raw F16 gate projection for MLA g_proj (`[T, hidden] x [nGate, hidden]^T -> [T, nGate]`).
// Grid: (nGate, T, 1), Threadgroup: (128, 1, 1).
kernel void qwg_mla_head_gate_proj(
    device const half*              x    [[buffer(0)]],
    device const half*              w    [[buffer(1)]],
    device half*                    out  [[buffer(2)]],
    constant MLAHeadGateProjParams& p    [[buffer(3)]],
    uint2                           tg   [[threadgroup_position_in_grid]],
    uint                            tid  [[thread_index_in_threadgroup]],
    uint                            sg   [[simdgroup_index_in_threadgroup]],
    uint                            lane [[thread_index_in_simdgroup]]
) {
    const uint j = tg.x, t = tg.y;
    if (j >= p.nGate || t >= p.T) return;
    device const half* xRow = x + (size_t)t * p.hidden;
    device const half* wRow = w + (size_t)j * p.hidden;
    float acc = 0.0f;
    for (uint k = tid; k < p.hidden; k += 128u) {
        acc = fma(float(xRow[k]), float(wRow[k]), acc);
    }
    acc = simd_sum(acc);
    threadgroup float sgSum[4];
    if (lane == 0u) sgSum[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        out[(size_t)t * p.nGate + j] = half(sgSum[0] + sgSum[1] + sgSum[2] + sgSum[3]);
    }
}

// 4b. Fused 8x8 MMA VHA head mixing + sigmoid(o_gate) (element-wise or head-wise) + H_32 -> attOut.
// Grid: (vHeadDim / 32, T, 1), Threadgroup: (256, 1, 1).
kernel void qwg_mla_vha_gate_rot(
    device const half*       yIn      [[buffer(0)]],
    device const half*       gateSrc  [[buffer(1)]],
    device const half*       vhaMix   [[buffer(2)]],
    device half*             attOut   [[buffer(3)]],
    constant MLAVHAGateParams& p      [[buffer(4)]],
    uint2                    tg       [[threadgroup_position_in_grid]],
    uint                     tid      [[thread_index_in_threadgroup]],
    uint                     sg       [[simdgroup_index_in_threadgroup]],
    uint                     lane     [[thread_index_in_simdgroup]]
) {
    const uint dBase = tg.x * 32u;
    const uint t = tg.y;
    if (t >= p.T || dBase >= p.vHeadDim) return;

    constexpr uint YS = 40u;
    constexpr uint OS = 36u;
    threadgroup half  shY[64u * YS];
    threadgroup float shOut[64u * OS];

    const uint rowStride = p.qHeads * p.vHeadDim;
    device const half* yRow = yIn + (size_t)t * rowStride;
    device const half* gRow = gateSrc + (size_t)t * p.gateStride + p.oGateOffset;
    device half* dstRow = attOut + (size_t)t * rowStride;

    if (p.hasVHA != 0u && (p.qHeads & 7u) == 0u && p.qHeads <= 64u) {
        for (uint i = tid; i < p.qHeads * 8u; i += 256u) {
            const uint g = i >> 3u, c4 = (i & 7u) * 4u;
            *(threadgroup half4*)(shY + g * YS + c4) = *(device const half4*)(yRow + g * p.vHeadDim + dBase + c4);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint hBase = sg * 8u;
        if (hBase < p.qHeads) {
            simdgroup_float8x8 acc[4] = {
                simdgroup_float8x8(0.0f), simdgroup_float8x8(0.0f),
                simdgroup_float8x8(0.0f), simdgroup_float8x8(0.0f)
            };
            for (uint g0 = 0u; g0 < p.qHeads; g0 += 8u) {
                simdgroup_half8x8 mf;
                simdgroup_load(mf, vhaMix + hBase * p.qHeads + g0, p.qHeads);
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j) {
                    simdgroup_half8x8 yf;
                    simdgroup_load(yf, shY + g0 * YS + j * 8u, YS);
                    simdgroup_multiply_accumulate(acc[j], mf, yf, acc[j]);
                }
            }
            threadgroup float* sDst = shOut + hBase * OS;
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) simdgroup_store(acc[j], sDst + j * 8u, OS);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint hr = 0u; hr < 8u; ++hr) {
                const uint h = hBase + hr;
                const uint d = dBase + lane;
                float val = sDst[hr * OS + lane];
                if (p.hasOGate == 1u) {
                    const float g = float(gRow[h * p.vHeadDim + d]);
                    val *= 1.0f / (1.0f + exp(-g));
                } else if (p.hasOGate == 2u) {
                    const float g = float(gRow[h]);
                    val *= 1.0f / (1.0f + exp(-g));
                }
                dstRow[h * p.vHeadDim + d] = half(qwg_fwht32_lane_f32(val, lane));
            }
        }
        return;
    }

    for (uint h = sg; h < p.qHeads; h += 8u) {
        const uint d = dBase + lane;
        float val = 0.0f;
        if (p.hasVHA != 0u) {
            for (uint g = 0u; g < p.qHeads; ++g) {
                val = fma(float(vhaMix[h * p.qHeads + g]), float(yRow[g * p.vHeadDim + d]), val);
            }
        } else {
            val = float(yRow[h * p.vHeadDim + d]);
        }
        if (p.hasOGate == 1u) {
            const float g = float(gRow[h * p.vHeadDim + d]);
            val *= 1.0f / (1.0f + exp(-g));
        } else if (p.hasOGate == 2u) {
            const float g = float(gRow[h]);
            val *= 1.0f / (1.0f + exp(-g));
        }
        dstRow[h * p.vHeadDim + d] = half(qwg_fwht32_lane_f32(val, lane));
    }
}

struct MLAFlashIndexerParams {
    uint T;
    uint hidden;
    uint fullIdxHeads;
    uint selHeads;
    uint idxDim;
    uint qChannels;
    uint qFullStride;
};

// 5. FlashIndexer head selection: gates = ln1 @ gateW^T, scores = gates + bias, top-K gather.
// Grid: (ceil(T / 32), 1, 1), Threadgroup: (256, 1, 1).
kernel void qwg_mla_flashindexer_select_heads(
    device const half*              ln1        [[buffer(0)]],
    device const half*              gateW      [[buffer(1)]],
    device const float*             gateBias   [[buffer(2)]],
    device const float*             idxWeights [[buffer(3)]],
    device const half*              mlaQFull   [[buffer(4)]],
    device half*                    qSelOut    [[buffer(5)]],
    device float*                   wSelOut    [[buffer(6)]],
    constant MLAFlashIndexerParams& p          [[buffer(7)]],
    uint                            tg         [[threadgroup_position_in_grid]],
    uint                            tid        [[thread_index_in_threadgroup]],
    uint                            sg         [[simdgroup_index_in_threadgroup]],
    uint                            lane       [[thread_index_in_simdgroup]]
) {
    constexpr uint BM = 32u, BN = 64u, BK = 32u, KSTR = 40u, SSTR = 68u;
    const uint tBase = tg * BM;
    if (tBase >= p.T || p.fullIdxHeads > BN) return;

    threadgroup half  shX[BM * KSTR];
    threadgroup half  shW[BN * KSTR];
    threadgroup float shS[BM * SSTR];

    simdgroup_float8x8 acc[2][2];
    for (uint a = 0u; a < 2u; ++a) {
        for (uint b = 0u; b < 2u; ++b) acc[a][b] = simdgroup_float8x8(0.0f);
    }

    const uint xRow = tid >> 3u, xC4 = (tid & 7u) * 4u;
    const uint sgRow = sg >> 2u, sgCol = sg & 3u;
    for (uint k0 = 0u; k0 < p.hidden; k0 += BK) {
        half4 xv = half4(0.0h);
        if (tBase + xRow < p.T && k0 + xC4 < p.hidden) {
            device const half* xp = ln1 + (size_t)(tBase + xRow) * p.hidden + k0 + xC4;
            if (k0 + xC4 + 4u <= p.hidden) {
                xv = *(device const half4*)xp;
            } else {
                for (uint c = 0u; c < 4u && (k0 + xC4 + c < p.hidden); ++c) xv[c] = xp[c];
            }
        }
        *(threadgroup half4*)(shX + xRow * KSTR + xC4) = xv;

        for (uint u = 0u; u < 2u; ++u) {
            const uint wIdx = tid + u * 256u;
            const uint h = wIdx >> 3u, c4 = (wIdx & 7u) * 4u;
            half4 wv = half4(0.0h);
            if (h < p.fullIdxHeads && k0 + c4 < p.hidden) {
                device const half* wp = gateW + (size_t)h * p.hidden + k0 + c4;
                if (k0 + c4 + 4u <= p.hidden) {
                    wv = *(device const half4*)wp;
                } else {
                    for (uint c = 0u; c < 4u && (k0 + c4 + c < p.hidden); ++c) wv[c] = wp[c];
                }
            }
            *(threadgroup half4*)(shW + h * KSTR + c4) = wv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_half8x8 af[2], bf[2];
            for (uint a = 0u; a < 2u; ++a) simdgroup_load(af[a], shX + (sgRow * 16u + a * 8u) * KSTR + kk, KSTR);
            for (uint b = 0u; b < 2u; ++b) {
                simdgroup_load(bf[b], shW + (sgCol * 16u + b * 8u) * KSTR + kk, KSTR, ulong2(0, 0), true);
            }
            for (uint a = 0u; a < 2u; ++a) {
                for (uint b = 0u; b < 2u; ++b) simdgroup_multiply_accumulate(acc[a][b], af[a], bf[b], acc[a][b]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint a = 0u; a < 2u; ++a) {
        for (uint b = 0u; b < 2u; ++b) {
            simdgroup_store(acc[a][b], shS + (sgRow * 16u + a * 8u) * SSTR + (sgCol * 16u + b * 8u), SSTR);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint h0 = lane, h1 = lane + 32u;
    const float bias0 = (h0 < p.fullIdxHeads) ? gateBias[h0] : 0.0f;
    const float bias1 = (h1 < p.fullIdxHeads) ? gateBias[h1] : 0.0f;
    const uint selK = min(p.selHeads, p.fullIdxHeads);
    for (uint tLocal = sg; tLocal < BM; tLocal += 8u) {
        const uint t = tBase + tLocal;
        if (t >= p.T) break;
        const float g0 = (h0 < p.fullIdxHeads) ? shS[tLocal * SSTR + h0] : 0.0f;
        const float g1 = (h1 < p.fullIdxHeads) ? shS[tLocal * SSTR + h1] : 0.0f;
        float s0 = (h0 < p.fullIdxHeads) ? (g0 + bias0) : -INFINITY;
        float s1 = (h1 < p.fullIdxHeads) ? (g1 + bias1) : -INFINITY;
        device const half* qSrcBase = mlaQFull + (size_t)t * p.qFullStride + p.qChannels;
        device half* qDstBase = qSelOut + (size_t)t * p.selHeads * p.idxDim;
        for (uint k = 0u; k < selK; ++k) {
            float candS = (s0 >= s1) ? s0 : s1;
            uint candH = (s0 >= s1) ? h0 : h1;
            for (ushort mask = 16; mask > 0; mask >>= 1) {
                const float oS = simd_shuffle_xor(candS, mask);
                const uint oH = simd_shuffle_xor(candH, mask);
                if (oS > candS || (oS == candS && oH < candH)) {
                    candS = oS;
                    candH = oH;
                }
            }
            if (lane == 0u) {
                const float winnerGate = shS[tLocal * SSTR + candH];
                wSelOut[(size_t)t * p.selHeads + k] = idxWeights[(size_t)t * p.fullIdxHeads + candH] * winnerGate;
            }
            device const half* qHeadSrc = qSrcBase + (size_t)candH * p.idxDim;
            device half* qHeadDst = qDstBase + (size_t)k * p.idxDim;
            for (uint d4 = lane * 4u; d4 < p.idxDim; d4 += 128u) {
                *(device half4*)(qHeadDst + d4) = *(device const half4*)(qHeadSrc + d4);
            }
            if (h0 == candH) s0 = -INFINITY;
            if (h1 == candH) s1 = -INFINITY;
        }
        if (lane == 0u) {
            for (uint k = selK; k < p.selHeads; ++k) wSelOut[(size_t)t * p.selHeads + k] = 0.0f;
        }
    }
}
