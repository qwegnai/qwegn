// EncodeKernels.swift — QWEngineDriver EncodeKernels family (split from QWEngineDriver.swift; same-module extension).
import Foundation
import Darwin
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWMLP
import QWHead
import QWMTP
import QWQuant

extension QWEngineDriver {
    /// A path that reads the INCUMBENT's container must REFUSE the family's,
    /// by name: the two layouts differ, and the incumbent's kernel would read
    /// the family's blocks as if they were its own.
    static func requireIncumbentContainer(_ view: QWWeightView, _ what: String) throws {
        guard view.isIncumbentContainer else {
            throw QWEngineDriverError.state(
                "\(what) reads the incumbent's container, and this view is in the "
                + "size-preserving family")
        }
    }

    func encodeGemvH16(_ enc: any MTLComputeCommandEncoder, pso: any MTLComputePipelineState,
                               n: Int, k: Int, x: QWTensorValue, wQ4: QWWeightView,
                               y16: QWTensorValue, y32: QWTensorValue?) {
        enc.setComputePipelineState(pso)
        enc.bindTensors(wQ4.blocks, x, startIndex: 0)
        (y32 ?? y16).bind(enc, index: 3)
        enc.setUInt32s(n, k, startIndex: 4)
        y16.bind(enc, index: 8)
    }

    static func resolveWeightPSO(
        for wQ4: QWWeightView,
        weightFormat: QWWeightFormat,
        hal: QWHal,
        defaultPSO: (any MTLComputePipelineState)?,
        opName: String,
        incumbentKernel: String,
        swigluClamp: QWSwiGLUClampSpec? = nil,
        aqKernel: () -> String?
    ) throws -> any MTLComputePipelineState {
        let matchesDefault = swigluClamp == nil
            && wQ4.isIncumbentContainer == weightFormat.denseIsIncumbentContainer
            && wQ4.bits == weightFormat.bits
            && wQ4.groupSize == weightFormat.groupSize
            && wQ4.isSHVQ == weightFormat.denseIsSHVQ
        if matchesDefault, let defaultPSO { return defaultPSO }
        let name = wQ4.isIncumbentContainer ? incumbentKernel : aqKernel()
        guard let kernelName = name else {
            throw QWEngineDriverError.unsupportedModel(
                "no \(opName) kernel for \(wQ4.bits)-bit group \(wQ4.groupSize)")
        }
        return try hal.pipeline(kernelName, swigluClamp: swigluClamp)
    }

    func gemvH16T4(_ enc: any MTLComputeCommandEncoder,
                   n: Int, k: Int, x: QWTensorValue, gs: QWTensorValue,
                   wQ4: QWWeightView, y16: QWTensorValue, y32: QWTensorValue? = nil) throws {
        _ = gs
        if y32 != nil {
            // The dual-store kernel writes f32 and f16 from one launch, and
            // the family has no twin for it — its decode GEMV writes f16 only.
            try Self.requireIncumbentContainer(wQ4, "the dual-store decode GEMV")
        }
        let pso: any MTLComputePipelineState
        if y32 != nil {
            pso = psoGEMVDxH16DualT4
        } else {
            pso = try Self.resolveWeightPSO(
                for: wQ4, weightFormat: weightFormat, hal: hal,
                defaultPSO: psoGEMVDxH16T4, opName: "decode GEMV",
                incumbentKernel: "qwg_gemv_q4g_dx_h16_t4",
                aqKernel: { QWGemmAQ.decodeKernelName(bits: wQ4.bits, group: wQ4.groupSize, isSHVQ: wQ4.isSHVQ) }
            )
        }
        if wQ4.isIncumbentContainer {
            encodeGemvH16(enc, pso: pso, n: n, k: k, x: x, wQ4: wQ4, y16: y16, y32: y32)
            enc.dispatchThreadgroups(MTLSize(width: (n + 3) / 4, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else {
            encodeGemvH16(enc, pso: pso, n: n | 0x4000_0000, k: k, x: x, wQ4: wQ4, y16: y16, y32: y32)
            enc.dispatchThreadgroups(MTLSize(width: (n + 15) / 16, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
    }

    func gemvH16GdnInProjT4(_ enc: any MTLComputeCommandEncoder,
                            n: Int, k: Int, x: QWTensorValue, gs: QWTensorValue,
                            wQ4: QWWeightView, y16: QWTensorValue, y32: QWTensorValue? = nil,
                            aLog: QWTensorValue, dtBias: QWTensorValue,
                            betas: QWTensorValue, glog: QWTensorValue) throws {
        _ = gs
        let pso = try Self.resolveWeightPSO(
            for: wQ4, weightFormat: weightFormat, hal: hal,
            defaultPSO: psoGEMVDxH16GdnInProjT4, opName: "GDN decode",
            incumbentKernel: "qwg_gemv_q4g_dx_h16_gdn_inproj_t4",
            aqKernel: { QWGemmAQ.gdnInProjDecodeKernelName(bits: wQ4.bits, group: wQ4.groupSize, isSHVQ: wQ4.isSHVQ) }
        )
        if wQ4.isIncumbentContainer {
            encodeGemvH16(enc, pso: pso, n: n, k: k, x: x, wQ4: wQ4, y16: y16, y32: y32)
            enc.bindTensors(aLog, dtBias, betas, glog, startIndex: 9)
            enc.dispatchThreadgroups(MTLSize(width: (n + 3) / 4, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else {
            encodeGemvH16(enc, pso: pso, n: n | 0x4000_0000, k: k, x: x, wQ4: wQ4, y16: y16, y32: y32)
            enc.bindTensors(aLog, dtBias, betas, glog, startIndex: 9)
            enc.dispatchThreadgroups(MTLSize(width: (n + 15) / 16, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
    }

    func gemmBatchedH16(_ enc: any MTLComputeCommandEncoder,
                        m: Int, n: Int, k: Int,
                        x: QWTensorValue, wQ4: QWWeightView,
                        y16: QWTensorValue,
                        xStride: Int? = nil, yStride: Int? = nil) throws {
        let useNAX = hal.caps.isMetal4TensorCapable
        guard hal.caps.supportsSimdgroupMMA else {
            throw QWEngineDriverError.unsupportedModel("prefill H16 GEMM requires Apple7+ simdgroup or Apple10+ NAX hardware")
        }
        // BM64 beats BM128 at every M (QWGemm.encodeNaxF32); rows are chunked to keep x in the SLC.
        let pso = try Self.resolveWeightPSO(
            for: wQ4, weightFormat: weightFormat, hal: hal,
            defaultPSO: psoGemmBatchedH16, opName: "prefill GEMM",
            incumbentKernel: useNAX ? "qwg_gemm_q4g_h16" : "qwg_gemm_q4g_sg",
            aqKernel: { QWGemmAQ.gemmKernelName(bits: wQ4.bits, group: wQ4.groupSize, isSHVQ: wQ4.isSHVQ, useNAX: useNAX) }
        )
        enc.setComputePipelineState(pso)
        enc.bindTensors(wQ4.blocks, x, y16, y16, startIndex: 0) // index 3 is dummy slot for unused y32
        enc.setUInt32s(n, k, startIndex: 5)
        let xs = xStride ?? k, ys = yStride ?? n
        enc.setUInt32s(xs, ys, startIndex: 9)
        let useSmallTile = hal.caps.supportsSimdgroupMMA
        let chunk = QWGemm.naxRowChunk(rows: m, k: k, gpuCores: hal.caps.gpuCores)
        var r0 = 0
        while r0 < m {
            let r = min(chunk, m - r0)
            let bm = useSmallTile ? (r <= 16 ? 16 : (r <= 32 ? 32 : 64)) : 64
            let bn = (useNAX && r <= 16) ? 128 : 64
            let modeBits = useSmallTile ? (r <= 16 ? 0x2000_0000 : (r <= 32 ? 0x4000_0000 : 0)) : 0
            if r0 > 0 {
                x.slice(byteOffset: r0 * xs * 2, count: r * xs).bind(enc, index: 1)
                let y = y16.slice(byteOffset: r0 * ys * 2, count: r * ys)
                y.bind(enc, index: 2); y.bind(enc, index: 3)
            }
            enc.setUInt32s(r | modeBits, startIndex: 4)
            enc.dispatchThreadgroups(MTLSize(width: (n + bn - 1) / bn, height: (r + bm - 1) / bm, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            r0 += r
        }
    }

    func gemmBatchedH16SwiGLU(_ enc: any MTLComputeCommandEncoder,
                              m: Int, i: Int, k: Int,
                              x: QWTensorValue, wQ4: QWWeightView,
                              out16: QWTensorValue,
                              xStride: Int? = nil, yStride: Int? = nil,
                              swigluClamp: QWSwiGLUClampSpec? = nil) throws {
        let useNAX = hal.caps.isMetal4TensorCapable
        guard hal.caps.supportsSimdgroupMMA else {
            throw QWEngineDriverError.unsupportedModel("fused SwiGLU GEMM requires Apple7+ simdgroup or Apple10+ NAX hardware")
        }
        let pso = try Self.resolveWeightPSO(
            for: wQ4, weightFormat: weightFormat, hal: hal,
            defaultPSO: psoGemmBatchedH16SwiGLU, opName: "fused SwiGLU GEMM",
            incumbentKernel: useNAX ? "qwg_gemm_q4g_h16_swiglu" : "qwg_gemm_q4g_sg_swiglu",
            swigluClamp: swigluClamp,
            aqKernel: { QWGemmAQ.swigluGemmKernelName(bits: wQ4.bits, group: wQ4.groupSize, isSHVQ: wQ4.isSHVQ, useNAX: useNAX) }
        )
        enc.setComputePipelineState(pso)
        enc.bindTensors(wQ4.blocks, x, out16, startIndex: 0)
        let xs = xStride ?? k, ys = yStride ?? i
        let useSmallTile = hal.caps.supportsSimdgroupMMA
        let chunk = QWGemm.naxRowChunk(rows: m, k: k, gpuCores: hal.caps.gpuCores)
        var r0 = 0
        while r0 < m {
            let r = min(chunk, m - r0)
            let use64 = useNAX && r <= 16 && (i & 63) == 0
            let bm = useSmallTile ? (use64 ? 16 : (r <= 32 ? 32 : 64)) : 64
            let bn = use64 ? 64 : 32
            let modeBits = useSmallTile ? (use64 ? 0x2000_0000 : (r <= 32 ? 0x4000_0000 : 0)) : 0
            if r0 > 0 {
                x.slice(byteOffset: r0 * xs * 2, count: r * xs).bind(enc, index: 1)
                out16.slice(byteOffset: r0 * ys * 2, count: r * ys).bind(enc, index: 2)
            }
            enc.setUInt32s(r | modeBits, i, k, xs, ys, startIndex: 3)
            enc.dispatchThreadgroups(MTLSize(width: i / bn, height: (r + bm - 1) / bm, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            r0 += r
        }
    }

    func gemmBatchedH16GdnInProj(_ enc: any MTLComputeCommandEncoder,
                                 m: Int, n: Int, k: Int,
                                 x: QWTensorValue, wQ4: QWWeightView,
                                 qkv16: QWTensorValue, z16: QWTensorValue, y32: QWTensorValue) {
        guard let pso = psoGemmBatchedH16GdnInProj else {
            preconditionFailure("GDN in-proj batched GEMM requires Apple7+ simdgroup or Apple10+ NAX hardware")
        }
        enc.setComputePipelineState(pso)
        enc.bindTensors(wQ4.blocks, x, qkv16, y32, startIndex: 0)
        enc.setUInt32s(n, k, startIndex: 5)
        z16.bind(enc, index: 7)
        let qkvC = geometry.gdnQKVChannels, zC = geometry.gdnZChannels
        let abCols = max(1, n - qkvC - zC)
        let chunk = QWGemm.naxRowChunk(rows: m, k: k, gpuCores: hal.caps.gpuCores)
        var r0 = 0
        while r0 < m {
            let r = min(chunk, m - r0)
            if r0 > 0 {
                x.slice(byteOffset: r0 * k * 2, count: r * k).bind(enc, index: 1)
                qkv16.slice(byteOffset: r0 * qkvC * 2, count: r * qkvC).bind(enc, index: 2)
                y32.slice(byteOffset: r0 * abCols * 4, count: r * abCols).bind(enc, index: 3)
                z16.slice(byteOffset: r0 * zC * 2, count: r * zC).bind(enc, index: 7)
            }
            enc.setUInt32s(r, startIndex: 4)
            enc.dispatchThreadgroups(MTLSize(width: (n + 63) / 64, height: (r + 63) / 64, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            r0 += r
        }
    }

    func encodeSiluMulF16(_ enc: any MTLComputeCommandEncoder,
                          gu: QWTensorValue, out: QWTensorValue,
                          m: Int, i: Int,
                          swigluClamp: QWSwiGLUClampSpec? = nil) throws {
        let pso = try swigluClamp.map { try hal.pipeline("qwg_silu_mul_f16", swigluClamp: $0) } ?? psoSiluMulF16
        enc.setComputePipelineState(pso)
        enc.bindTensors(gu, out, startIndex: 0)
        enc.setUInt32s(m, i, startIndex: 2)
        // grid.x covers ceil(I/8) = 2176 = 17 * 128 — 128 threads/TG has zero tail waste.
        // The grid is rounded UP to the threadgroup, so an intermediate that
        // is not a multiple of 1024 still dispatches; the tail is absorbed.
        enc.dispatchThreadgroups(MTLSize(width: Self.siluThreadgroups(i: i), height: m, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    func encodePagedKVWrite(_ enc: any MTLComputeCommandEncoder,
                            kv: (k: QWKVLayerView, v: QWKVLayerView),
                            k: QWTensorValue, v: QWTensorValue,
                            startPos: Int, t: Int) {
        // The specialized writer's row stride and per-head scale/group map
        // require 4 KV heads × 256 halves (1024-half row stride); others use general.
        let general = !geometry.matchesKVStride1024Kernel
        enc.setComputePipelineState(general ? psoKVPagedWriteGeneral : psoKVPagedWrite)
        enc.setBuffer(kv.k.pool.buffer, offset: kv.k.pool.base, index: 0)
        enc.setBuffer(kv.v.pool.buffer, offset: kv.v.pool.base, index: 1)
        enc.bindTensors(kv.k.table, kv.v.table, k, v, startIndex: 2)
        let q8U: Int
        switch kv.k.mode {
        case .f16: q8U = 0
        case .q8:  q8U = 1
        case .q4:  q8U = 2
        @unknown default: fatalError("encodePagedKVWrite: unknown KV mode")
        }
        enc.setUInt32s(startPos, t, q8U, startIndex: 6)
        enc.dispatchThreadgroups(MTLSize(width: geometry.kvHeads, height: t, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    /// Whether the attention zero-pad serves the specialized non-paged prefill kernel.
    public static func zeroPadServes(geometry: QWModelGeometry) -> Bool {
        geometry.matchesHead256GQA6Kernel && geometry.attnOutputGate
            && (geometry.descriptor.slidingWindow ?? 0) == 0
    }

    /// The threadgroup count for the SwiGLU kernel's grid. Rounded UP to
    /// the threadgroup so an intermediate that is not a multiple of 1024
    /// still dispatches; the kernel's `mi >= I` bound absorbs the tail.
    public static func siluThreadgroups(i: Int) -> Int {
        let cols = (i + 7) / 8
        return (cols + 127) / 128
    }

    /// The threadgroup count for the zero-pad's grid, rounded up to the
    /// threadgroup for the same reason.
    public static func zeroPadThreadgroups(qProjOut: Int) -> Int {
        (qProjOut + 255) / 256
    }

    /// The 27B's prefill pad: the NAX prefill reads whole 64-row tiles,
    /// so the chunk's last tile is zero-filled. The kernel writes with the
    /// 27B's own row strides, so it serves that geometry only.
    func encodeZeroPadAttn(_ enc: any MTLComputeCommandEncoder,
                           q: QWTensorValue, k: QWTensorValue, v: QWTensorValue,
                           t: Int, sp: Int) throws {
        // The standard path does not pad at all: its general prefill
        // dispatches one threadgroup per real query and reads only the
        // chunk's T rows, and its K/V live in the pool (C105).
        guard Self.zeroPadServes(geometry: geometry) else {
            throw QWEngineDriverError.state(
                "geometry does not support NAX zero-pad: q \(geometry.qProjOut) "
                + "kv \(geometry.kvChannels)")
        }
        enc.setComputePipelineState(psoZeroPadAttn)
        enc.bindTensors(q, k, v, startIndex: 0)
        enc.setUInt32s(t, sp, geometry.qProjOut, geometry.kvChannels, startIndex: 3)
        // The grid is rounded UP to the threadgroup so a q width that is not
        // a multiple of 256 still dispatches; the kernel's `col < 12288`
        // bound absorbs the tail, and the mapping is unchanged.
        enc.dispatchThreadgroups(MTLSize(width: Self.zeroPadThreadgroups(qProjOut: geometry.qProjOut),
                                         height: sp - t, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func mlpSiluPairH16(_ enc: any MTLComputeCommandEncoder,
                        x: QWTensorValue, gs: QWTensorValue, wQ4: QWWeightView,
                        h16: QWTensorValue, fuseFwht32: Bool = false,
                        swigluClamp: QWSwiGLUClampSpec? = nil) throws {
        _ = gs
        let pso = try Self.resolveWeightPSO(
            for: wQ4, weightFormat: weightFormat, hal: hal,
            defaultPSO: psoGEMVSiluPairH16, opName: "MLP SiLU pair GEMV",
            incumbentKernel: "qwg_gemv_q4g_silu_pair_dx_h16",
            swigluClamp: swigluClamp,
            aqKernel: { QWGemmAQ.siluPairKernelName(bits: wQ4.bits, group: wQ4.groupSize, isSHVQ: wQ4.isSHVQ) }
        )
        enc.setComputePipelineState(pso)
        enc.bindTensors(wQ4.blocks, x, startIndex: 0)
        h16.bind(enc, index: 3)
        let intermediate = wQ4.n / 2
        if wQ4.isIncumbentContainer && !fuseFwht32 {
            enc.setUInt32s(wQ4.n, wQ4.k, startIndex: 4)
            h16.bind(enc, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: (intermediate + 3) / 4, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        } else {
            let useFusedFwht = fuseFwht32 && intermediate > 0 && (intermediate % 32 == 0)
            let nPacked = useFusedFwht ? (wQ4.n | 0xC000_0000) : (wQ4.n | 0x8000_0000)
            enc.setUInt32s(nPacked, wQ4.k, startIndex: 4)
            h16.bind(enc, index: 8)
            let tgWidth = useFusedFwht ? (intermediate / 32) : ((intermediate + 7) / 8)
            enc.dispatchThreadgroups(MTLSize(width: tgWidth, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    func encodeFwht32Fast(_ enc: any MTLComputeCommandEncoder,
                          x: QWTensorValue, count: Int? = nil) {
        let n = count ?? x.shape.elementCount
        enc.setComputePipelineState(psoFwht32)
        x.bind(enc, index: 0)
        enc.setUInt32s(n, startIndex: 1)
        let nBlocks = n / 32
        enc.dispatchThreadgroups(MTLSize(width: nBlocks, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    func encodeRMSNormFast(_ enc: any MTLComputeCommandEncoder,
                                   x: QWTensorValue, gamma: QWTensorValue,
                                   out: QWTensorValue, rows: Int = 1) throws {
        try checkNormGeometry()
        enc.setComputePipelineState(psoRMSNorm)
        enc.bindTensors(x, gamma, out, startIndex: 0)
        enc.setUInt32s(geometry.hidden / 8, startIndex: 3)
        enc.setPod(geometry.rmsEps, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    /// The norm family's contract: uint4 rows that rotate into the H_32
    /// basis in groups of 32 halves, so the hidden must be a multiple of
    /// 256. The kernels do not refuse it, so the driver does.
    func checkNormGeometry() throws {
        guard geometry.hidden % 256 == 0 else {
            throw QWEngineDriverError.state(
                "the norm family needs a hidden that is a multiple of 256, got \(geometry.hidden)")
        }
    }

    func canFusePreMoENorm(weights: LayerW, rows: Int) -> Bool {
        guard weights.moe != nil, weights.moe?.coldStreamer == nil, !QWMoEChain.coldTimingProxy else { return false }
        if weights.mhc != nil {
            return mhcSplit != nil && rows <= QWMHCSplitRouter.maxRows
        }
        return true
    }

    func canSkipPreMoERot32(weights: LayerW, scratch: MoEScratch, rows: Int) -> Bool {
        guard let moe = weights.moe, moe.latent == nil,
              rows >= 64, geometry.supportsGroupedNAXMoEPrefill,
              hal.caps.isMetal4TensorCapable else { return false }
        let hasGroupedNaxScratch = scratch.expOff != nil && scratch.sortedPair != nil
            && scratch.sortedTok != nil && scratch.invPerm != nil
            && scratch.tileE != nil && scratch.tileR != nil
            && scratch.tileCount != nil && scratch.dnHi != nil
            && scratch.naxX16 != nil
            && (scratch.gu16 != nil || QWGemm.supportsMoEGateUpNaxSwiglu(moe.gateUp))
        guard hasGroupedNaxScratch else { return false }
        let H = geometry.hidden
        let ISh = geometry.sharedExpertIntermediate
        return ISh == 0 || (H % 64 == 0 && (2 * ISh) % 64 == 0)
    }

    func encodeRMSNormResidualFast(_ enc: any MTLComputeCommandEncoder,
                                           x: QWTensorValue, gamma: QWTensorValue,
                                           out: QWTensorValue, residual: QWTensorValue,
                                           rows: Int = 1, rowParallel: Bool = true,
                                           moeBridge: (xRot32: QWTensorValue, xPlain32: QWTensorValue, skipRot32: Bool)? = nil) throws {
        try checkNormGeometry()
        let hiddenU4 = geometry.hidden / 8
        // Decode/verify use a threadgroup per row (not latency-bound, and verify matches serial
        // decode numerics); prefill fills the GPU anyway and keeps the simdgroup kernel
        // (`rowParallel: false`) so its output is unchanged.
        let wide = rowParallel && hiddenU4 % 32 == 0 && hiddenU4 <= 640
        let pso: any MTLComputePipelineState
        if moeBridge != nil {
            pso = wide ? psoRMSNormResidualMoEWide : psoRMSNormResidualMoE
        } else {
            pso = wide ? psoRMSNormResidualWide : psoRMSNormResidual
        }
        enc.setComputePipelineState(pso)
        enc.bindTensors(x, gamma, out, startIndex: 0)
        var packedU4 = UInt32(hiddenU4)
        if let moeBridge, moeBridge.skipRot32 { packedU4 |= 0x8000_0000 }
        enc.setBytes(&packedU4, length: 4, index: 3)
        enc.setPod(geometry.rmsEps, index: 4)
        residual.bind(enc, index: 5)
        if let moeBridge {
            moeBridge.xRot32.bind(enc, index: 6)
            moeBridge.xPlain32.bind(enc, index: 7)
        }
        if wide {
            enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: hiddenU4, height: 1, depth: 1))
            return
        }
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

}
