// shvq_metal_enc.mm — Native Metal SHVQ{1,15,2,3,4}_G{32,64,128,256} encoder.
// Executes FWHT-32 + 2-step ALS e-means codebook search 100% in registers
// with uniform constant-cache broadcast and a single reusable 146 MiB buffer.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>
#include <mutex>

static const char* kShvqMetalSrc = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

struct EncParams {
    uint total_groups;
    uint group_size;
    uint bits;
    uint rotate;
    uint is_bf16;
    uint als_iters;
};

static inline float fwht32_lane(float x, uint lane) {
    for (uint h = 1u; h < 32u; h <<= 1u) {
        const float mate = simd_shuffle_xor(x, ushort(h));
        x = ((lane & h) == 0u) ? (x + mate) : (mate - x);
    }
    return x * 0.17677669529663689f;
}

kernel void shvq_encode_chunk(
    device const uchar* in_raw   [[buffer(0)]],
    device uchar*       out_raw  [[buffer(1)]],
    constant float4*    cb2_vec  [[buffer(2)]],
    constant float4*    cb3_vec  [[buffer(3)]],
    constant float4*    cb4_vec  [[buffer(4)]],
    constant EncParams& p        [[buffer(5)]],
    constant float4*    cb1_vec  [[buffer(6)]],
    constant float4*    cb15_vec [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgi  [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint W_PER_TG = 2048u;
    const uint groups_per_tg = W_PER_TG / p.group_size;
    const uint base_group = tgid * groups_per_tg;
    if (base_group >= p.total_groups) return;
    const uint valid_groups = min(groups_per_tg, p.total_groups - base_group);
    const uint valid_weights = valid_groups * p.group_size;
    const size_t in_elem_base = size_t(base_group) * size_t(p.group_size);
    const bool is_sub2 = (p.bits == 1u || p.bits == 15u);

    threadgroup float tgAbs[W_PER_TG];
    threadgroup uchar tgNeg[W_PER_TG];
    threadgroup float tgScale[64];
    threadgroup ushort tgIdx[512];

    for (uint pass = 0u; pass < 8u; ++pass) {
        const uint local_w = (pass * 8u + sgi) * 32u + lane;
        float v = 0.0f;
        if (local_w < valid_weights) {
            const size_t idx = in_elem_base + local_w;
            if (p.is_bf16 != 0u) {
                const ushort b16 = ((device const ushort*)in_raw)[idx];
                v = as_type<float>(uint(b16) << 16u);
            } else {
                v = ((device const float*)in_raw)[idx];
            }
        }
        if (p.rotate != 0u) {
            v = fwht32_lane(v, lane);
        }
        if (local_w < valid_weights) {
            if (is_sub2) {
                tgNeg[local_w] = 0u;
                tgAbs[local_w] = v;
            } else {
                tgNeg[local_w] = (as_type<uint>(v) >> 31u) != 0u ? 1u : 0u;
                tgAbs[local_w] = fabs(v);
            }
        } else {
            tgNeg[local_w] = 0u;
            tgAbs[local_w] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < valid_groups) {
        const uint gOff = tid * p.group_size;
        float sq = 0.0f;
        for (uint i = 0u; i < p.group_size; ++i) {
            const float a = tgAbs[gOff + i];
            sq += a * a;
        }
        tgScale[tid] = clamp(sqrt(sq / float(p.group_size)), 1e-8f, 65504.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint it = 0u; it <= p.als_iters; ++it) {
        if (is_sub2) {
            const uint wOff = tid * 8u;
            const uint gId = wOff / p.group_size;
            float invSc = 0.0f;
            float4 x0 = float4(0.0f), x1 = float4(0.0f);
            if (gId < valid_groups) {
                const float sc = max(float(half(tgScale[gId])), 1e-8f);
                invSc = 1.0f / sc;
                x0 = float4(tgAbs[wOff + 0u], tgAbs[wOff + 1u], tgAbs[wOff + 2u], tgAbs[wOff + 3u]);
                x1 = float4(tgAbs[wOff + 4u], tgAbs[wOff + 5u], tgAbs[wOff + 6u], tgAbs[wOff + 7u]);
            }
            const float4 u0 = x0 * invSc;
            const float4 u1 = x1 * invSc;
            const uint kMax = (p.bits == 1u) ? 128u : 2048u;
            constant float4* cb = (p.bits == 1u) ? cb1_vec : cb15_vec;
            float bestD = INFINITY;
            uint bestK = 0u;
            uchar bestSign = 0u;
            for (uint k = 0u; k < kMax; ++k) {
                const float4 c0 = cb[2u * k];
                const float4 c1 = cb[2u * k + 1u];
                const float cn2 = dot(c0, c0) + dot(c1, c1);
                const float d = dot(u0, c0) + dot(u1, c1);
                const float dist = cn2 - 2.0f * fabs(d);
                if (dist < bestD) {
                    bestD = dist;
                    bestK = k;
                    bestSign = (d < 0.0f) ? 1u : 0u;
                }
            }
            if (gId < valid_groups) {
                tgIdx[tid] = ushort(bestK);
                tgNeg[tid] = bestSign;
            }
        } else if (p.bits == 2u) {
            const uint wOff = tid * 8u;
            const uint gId = wOff / p.group_size;
            float invSc = 0.0f;
            float4 x0 = float4(0.0f), x1 = float4(0.0f);
            if (gId < valid_groups) {
                const float sc = max(float(half(tgScale[gId])), 1e-8f);
                invSc = 1.0f / sc;
                x0 = float4(tgAbs[wOff + 0u], tgAbs[wOff + 1u], tgAbs[wOff + 2u], tgAbs[wOff + 3u]);
                x1 = float4(tgAbs[wOff + 4u], tgAbs[wOff + 5u], tgAbs[wOff + 6u], tgAbs[wOff + 7u]);
            }
            const float4 u0 = x0 * invSc;
            const float4 u1 = x1 * invSc;
            float bestD = INFINITY;
            uint bestK = 0u;
            for (uint k = 0u; k < 256u; ++k) {
                const float4 c0 = cb2_vec[2u * k];
                const float4 c1 = cb2_vec[2u * k + 1u];
                const float4 d0 = u0 - c0;
                const float4 d1 = u1 - c1;
                const float dist = dot(d0, d0) + dot(d1, d1);
                if (dist < bestD) { bestD = dist; bestK = k; }
            }
            if (gId < valid_groups) {
                tgIdx[tid] = ushort(bestK);
            }
        } else {
            const uint kMax = (p.bits == 3u) ? 256u : 4096u;
            constant float4* cb = (p.bits == 3u) ? cb3_vec : cb4_vec;
            for (uint subPass = 0u; subPass < 2u; ++subPass) {
                const uint subId = subPass * 256u + tid;
                const uint wOff = subId * 4u;
                const uint gId = wOff / p.group_size;
                float invSc = 0.0f;
                float4 x0 = float4(0.0f);
                if (gId < valid_groups) {
                    const float sc = max(float(half(tgScale[gId])), 1e-8f);
                    invSc = 1.0f / sc;
                    x0 = float4(tgAbs[wOff + 0u], tgAbs[wOff + 1u], tgAbs[wOff + 2u], tgAbs[wOff + 3u]);
                }
                const float4 u0 = x0 * invSc;
                float bestD = INFINITY;
                uint bestK = 0u;
                for (uint k = 0u; k < kMax; ++k) {
                    const float4 c0 = cb[k];
                    const float4 d0 = u0 - c0;
                    const float dist = dot(d0, d0);
                    if (dist < bestD) { bestD = dist; bestK = k; }
                }
                if (gId < valid_groups) {
                    tgIdx[subId] = ushort(bestK);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (it < p.als_iters && tid < valid_groups) {
            const uint gOff = tid * p.group_size;
            float num = 0.0f;
            float den = 0.0f;
            if (is_sub2) {
                constant float4* cb = (p.bits == 1u) ? cb1_vec : cb15_vec;
                const uint nOct = p.group_size >> 3u;
                const uint octBase = gOff >> 3u;
                for (uint o = 0u; o < nOct; ++o) {
                    const uint oct = octBase + o;
                    const uint k = uint(tgIdx[oct]);
                    const float sgn = (tgNeg[oct] != 0u) ? -1.0f : 1.0f;
                    const float4 c0 = cb[2u * k] * sgn;
                    const float4 c1 = cb[2u * k + 1u] * sgn;
                    const uint w = gOff + o * 8u;
                    num += tgAbs[w+0]*c0.x + tgAbs[w+1]*c0.y + tgAbs[w+2]*c0.z + tgAbs[w+3]*c0.w
                         + tgAbs[w+4]*c1.x + tgAbs[w+5]*c1.y + tgAbs[w+6]*c1.z + tgAbs[w+7]*c1.w;
                    den += dot(c0, c0) + dot(c1, c1);
                }
            } else if (p.bits == 2u) {
                const uint nOct = p.group_size >> 3u;
                const uint octBase = gOff >> 3u;
                for (uint o = 0u; o < nOct; ++o) {
                    const uint k = uint(tgIdx[octBase + o]);
                    const float4 c0 = cb2_vec[2u * k];
                    const float4 c1 = cb2_vec[2u * k + 1u];
                    const uint w = gOff + o * 8u;
                    num += tgAbs[w+0]*c0.x + tgAbs[w+1]*c0.y + tgAbs[w+2]*c0.z + tgAbs[w+3]*c0.w
                         + tgAbs[w+4]*c1.x + tgAbs[w+5]*c1.y + tgAbs[w+6]*c1.z + tgAbs[w+7]*c1.w;
                    den += dot(c0, c0) + dot(c1, c1);
                }
            } else {
                constant float4* cb = (p.bits == 3u) ? cb3_vec : cb4_vec;
                const uint nQuad = p.group_size >> 2u;
                const uint quadBase = gOff >> 2u;
                for (uint q = 0u; q < nQuad; ++q) {
                    const uint k = uint(tgIdx[quadBase + q]);
                    const float4 c0 = cb[k];
                    const uint w = gOff + q * 4u;
                    num += tgAbs[w+0]*c0.x + tgAbs[w+1]*c0.y + tgAbs[w+2]*c0.z + tgAbs[w+3]*c0.w;
                    den += dot(c0, c0);
                }
            }
            tgScale[tid] = clamp(num / max(den, 1e-12f), 1e-8f, 65504.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < valid_groups) {
        const uint payBytes = (p.bits == 15u) ? ((p.group_size * 3u) >> 4u) : ((p.group_size * p.bits) >> 3u);
        const uint bbytes = 2u + payBytes;
        device uchar* dst = out_raw + size_t(base_group + tid) * size_t(bbytes);
        const ushort scBits = as_type<ushort>(half(tgScale[tid]));
        dst[0] = uchar(scBits & 0xFFu);
        dst[1] = uchar(scBits >> 8u);
        const uint gOff = tid * p.group_size;
        const uint nOct = p.group_size >> 3u;
        if (p.bits == 1u) {
            const uint octBase = gOff >> 3u;
            for (uint o = 0u; o < nOct; ++o) {
                const uint oct = octBase + o;
                dst[2u + o] = uchar((uint(tgIdx[oct]) & 0x7Fu) | (uint(tgNeg[oct]) << 7u));
            }
        } else if (p.bits == 15u) {
            const uint octBase = gOff >> 3u;
            const uint nPairs = nOct >> 1u;
            for (uint pr = 0u; pr < nPairs; ++pr) {
                const uint o0 = octBase + 2u * pr;
                const uint o1 = o0 + 1u;
                const uint u0 = (uint(tgIdx[o0]) & 0x07FFu) | (uint(tgNeg[o0]) << 11u);
                const uint u1 = (uint(tgIdx[o1]) & 0x07FFu) | (uint(tgNeg[o1]) << 11u);
                dst[2u + 3u * pr + 0u] = uchar(u0 & 0xFFu);
                dst[2u + 3u * pr + 1u] = uchar(((u0 >> 8u) & 0x0Fu) | ((u1 & 0x0Fu) << 4u));
                dst[2u + 3u * pr + 2u] = uchar((u1 >> 4u) & 0xFFu);
            }
        } else if (p.bits == 2u) {
            const uint octBase = gOff >> 3u;
            for (uint o = 0u; o < nOct; ++o) {
                uchar signs = 0u;
                for (uint j = 0u; j < 8u; ++j) signs |= uchar(tgNeg[gOff + o * 8u + j] << j);
                dst[2u + 2u * o] = signs;
                dst[2u + 2u * o + 1u] = uchar(tgIdx[octBase + o] & 0xFFu);
            }
        } else if (p.bits == 3u) {
            const uint quadBase = gOff >> 2u;
            for (uint o = 0u; o < nOct; ++o) {
                uchar signs = 0u;
                for (uint j = 0u; j < 8u; ++j) signs |= uchar(tgNeg[gOff + o * 8u + j] << j);
                dst[2u + 3u * o] = signs;
                dst[2u + 3u * o + 1u] = uchar(tgIdx[quadBase + 2u * o] & 0xFFu);
                dst[2u + 3u * o + 2u] = uchar(tgIdx[quadBase + 2u * o + 1u] & 0xFFu);
            }
        } else {
            const uint nQuad = p.group_size >> 2u;
            const uint quadBase = gOff >> 2u;
            for (uint q = 0u; q < nQuad; ++q) {
                ushort signs4 = 0u;
                for (uint j = 0u; j < 4u; ++j) signs4 |= ushort(tgNeg[gOff + q * 4u + j]) << j;
                const ushort u16 = (tgIdx[quadBase + q] & 0x0FFFu) | (signs4 << 12u);
                dst[2u + 2u * q] = uchar(u16 & 0xFFu);
                dst[2u + 2u * q + 1u] = uchar(u16 >> 8u);
            }
        }
    }
}
)MSL";

namespace {
struct MetalEncCtx {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pso = nil;
    id<MTLBuffer> cb1Buf = nil;
    id<MTLBuffer> cb15Buf = nil;
    id<MTLBuffer> cb2Buf = nil;
    id<MTLBuffer> cb3Buf = nil;
    id<MTLBuffer> cb4Buf = nil;
    id<MTLBuffer> inBuf = nil;
    id<MTLBuffer> outBuf = nil;
    size_t maxChunkWeights = 32ULL * 1024ULL * 1024ULL;
    std::mutex mtx;
};

static MetalEncCtx gCtx;
} // namespace

extern "C" int shvq_metal_init(
    const float* cb1_f32,
    const float* cb15_f32,
    const float* cb2_f32,
    const float* cb3_f32,
    const float* cb4_f32
) {
    std::lock_guard<std::mutex> lk(gCtx.mtx);
    if (gCtx.pso != nil) return 0;
    @autoreleasepool {
        gCtx.dev = MTLCreateSystemDefaultDevice();
        if (!gCtx.dev) return -1;
        gCtx.queue = [gCtx.dev newCommandQueue];
        NSError* err = nil;
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> lib = [gCtx.dev newLibraryWithSource:[NSString stringWithUTF8String:kShvqMetalSrc]
                                                    options:opts
                                                      error:&err];
        if (!lib) {
            fprintf(stderr, "shvq_metal compile error: %s\n", [[err localizedDescription] UTF8String]);
            return -2;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"shvq_encode_chunk"];
        gCtx.pso = [gCtx.dev newComputePipelineStateWithFunction:fn error:&err];
        if (!gCtx.pso) return -3;

        gCtx.cb1Buf  = [gCtx.dev newBufferWithBytes:cb1_f32  length:128 * 8 * sizeof(float)  options:MTLResourceStorageModeShared];
        gCtx.cb15Buf = [gCtx.dev newBufferWithBytes:cb15_f32 length:2048 * 8 * sizeof(float) options:MTLResourceStorageModeShared];
        gCtx.cb2Buf  = [gCtx.dev newBufferWithBytes:cb2_f32  length:256 * 8 * sizeof(float)  options:MTLResourceStorageModeShared];
        gCtx.cb3Buf  = [gCtx.dev newBufferWithBytes:cb3_f32  length:256 * 4 * sizeof(float)  options:MTLResourceStorageModeShared];
        gCtx.cb4Buf  = [gCtx.dev newBufferWithBytes:cb4_f32  length:4096 * 4 * sizeof(float) options:MTLResourceStorageModeShared];
        gCtx.inBuf   = [gCtx.dev newBufferWithLength:gCtx.maxChunkWeights * sizeof(float) options:MTLResourceStorageModeShared];
        gCtx.outBuf  = [gCtx.dev newBufferWithLength:(gCtx.maxChunkWeights / 32) * 18 options:MTLResourceStorageModeShared];
        if (!gCtx.inBuf || !gCtx.outBuf || !gCtx.cb1Buf || !gCtx.cb15Buf) return -4;
    }
    return 0;
}

struct EncParamsHost {
    uint32_t total_groups;
    uint32_t group_size;
    uint32_t bits;
    uint32_t rotate;
    uint32_t is_bf16;
    uint32_t als_iters;
};

extern "C" int shvq_metal_encode(
    const void* in_ptr,
    uint8_t* out_ptr,
    uint64_t total_groups,
    uint32_t group_size,
    uint32_t bits,
    uint32_t rotate,
    uint32_t is_bf16,
    uint32_t als_iters
) {
    std::lock_guard<std::mutex> lk(gCtx.mtx);
    if (!gCtx.pso) return -1;
    const uint64_t chunk_groups = gCtx.maxChunkWeights / group_size;
    const size_t elem_bytes = is_bf16 ? 2 : 4;
    const size_t pay_bytes = (bits == 15) ? ((group_size * 3) / 16) : ((group_size * bits) / 8);
    const size_t bbytes = 2 + pay_bytes;

    @autoreleasepool {
        uint64_t done = 0;
        while (done < total_groups) {
            const uint32_t cur_groups = (uint32_t)std::min<uint64_t>(chunk_groups, total_groups - done);
            const size_t in_bytes = size_t(cur_groups) * size_t(group_size) * elem_bytes;
            const size_t out_bytes = size_t(cur_groups) * bbytes;
            memcpy([gCtx.inBuf contents], (const uint8_t*)in_ptr + done * size_t(group_size) * elem_bytes, in_bytes);

            EncParamsHost p = {cur_groups, group_size, bits, rotate, is_bf16, als_iters};
            const uint32_t groups_per_tg = 2048u / group_size;
            const uint32_t n_tgs = (cur_groups + groups_per_tg - 1u) / groups_per_tg;

            id<MTLCommandBuffer> cb = [gCtx.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:gCtx.pso];
            [enc setBuffer:gCtx.inBuf offset:0 atIndex:0];
            [enc setBuffer:gCtx.outBuf offset:0 atIndex:1];
            [enc setBuffer:gCtx.cb2Buf offset:0 atIndex:2];
            [enc setBuffer:gCtx.cb3Buf offset:0 atIndex:3];
            [enc setBuffer:gCtx.cb4Buf offset:0 atIndex:4];
            [enc setBytes:&p length:sizeof(p) atIndex:5];
            [enc setBuffer:gCtx.cb1Buf offset:0 atIndex:6];
            [enc setBuffer:gCtx.cb15Buf offset:0 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(n_tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) return -2;

            memcpy(out_ptr + done * bbytes, [gCtx.outBuf contents], out_bytes);
            done += cur_groups;
        }
    }
    return 0;
}
