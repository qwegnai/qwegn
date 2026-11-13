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
import QWState

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

    private func encodeGemvH16(_ enc: any MTLComputeCommandEncoder, pso: any MTLComputePipelineState,
                               n: Int, k: Int, x: QWTensorValue, wQ4: QWWeightView,
                               y16: QWTensorValue, y32: QWTensorValue?) {
        enc.setComputePipelineState(pso)
        enc.bindTensors(wQ4.blocks, x, startIndex: 0)
        (y32 ?? y16).bind(enc, index: 3)
        enc.setUInt32s(n, k, startIndex: 4)
        y16.bind(enc, index: 8)
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
        } else if !wQ4.isIncumbentContainer,
                  (wQ4.bits != weightFormat.bits || wQ4.groupSize != weightFormat.groupSize
                      || wQ4.isSHVQ != weightFormat.denseIsSHVQ || weightFormat.denseIsIncumbentContainer),
                  let name = QWGemmAQ.decodeKernelName(bits: wQ4.bits, group: wQ4.groupSize, isSHVQ: wQ4.isSHVQ) {
            pso = try hal.pipeline(name)
        } else {
            pso = psoGEMVDxH16T4
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
        if wQ4.isIncumbentContainer {
            encodeGemvH16(enc, pso: psoGEMVDxH16GdnInProjT4, n: n, k: k, x: x, wQ4: wQ4, y16: y16, y32: y32)
            enc.bindTensors(aLog, dtBias, betas, glog, startIndex: 9)
            enc.dispatchThreadgroups(MTLSize(width: (n + 3) / 4, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else {
            let pso: any MTLComputePipelineState
            if wQ4.bits != weightFormat.bits || wQ4.groupSize != weightFormat.groupSize
                || wQ4.isSHVQ != weightFormat.denseIsSHVQ || weightFormat.denseIsIncumbentContainer,
               let name = QWGemmAQ.gdnInProjDecodeKernelName(bits: wQ4.bits, group: wQ4.groupSize, isSHVQ: wQ4.isSHVQ) {
                pso = try hal.pipeline(name)
            } else {
                pso = psoGEMVDxH16GdnInProjT4
            }
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
                        xStride: Int? = nil, yStride: Int? = nil) {
        // BM64 beats BM128 at every M (QWGemm.encodeNaxF32); rows are chunked to keep x in the SLC.
        let pso: any MTLComputePipelineState
        if !wQ4.isIncumbentContainer,
           (wQ4.bits != weightFormat.bits || wQ4.groupSize != weightFormat.groupSize
               || wQ4.isSHVQ != weightFormat.denseIsSHVQ || weightFormat.denseIsIncumbentContainer),
           let name = QWGemmAQ.gemmKernelName(bits: wQ4.bits, group: wQ4.groupSize, bm128: false, isSHVQ: wQ4.isSHVQ),
           let customPSO = try? hal.pipeline(name) {
            pso = customPSO
        } else {
            pso = psoGemmBatchedH16
        }
        enc.setComputePipelineState(pso)
        enc.bindTensors(wQ4.blocks, x, y16, y16, startIndex: 0) // index 3 is dummy slot for unused y32
        enc.setUInt32s(n, k, startIndex: 5)
        let xs = xStride ?? k, ys = yStride ?? n
        enc.setUInt32s(xs, ys, startIndex: 9)
        let useSmallTile = hal.caps.isMetal4TensorCapable
        let chunk = QWGemm.naxRowChunk(rows: m, k: k)
        var r0 = 0
        while r0 < m {
            let r = min(chunk, m - r0)
            let bm = useSmallTile ? (r <= 16 ? 16 : (r <= 32 ? 32 : 64)) : 64
            let bn = (useSmallTile && r <= 16) ? 128 : 64
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

    func gemmBatchedH16GdnInProj(_ enc: any MTLComputeCommandEncoder,
                                 m: Int, n: Int, k: Int,
                                 x: QWTensorValue, wQ4: QWWeightView,
                                 qkv16: QWTensorValue, z16: QWTensorValue, y32: QWTensorValue) {
        enc.setComputePipelineState(psoGemmBatchedH16GdnInProj)
        enc.bindTensors(wQ4.blocks, x, qkv16, y32, startIndex: 0)
        enc.setUInt32s(n, k, startIndex: 5)
        z16.bind(enc, index: 7)
        let qkvC = geometry.gdnQKVChannels, zC = geometry.gdnZChannels
        let chunk = QWGemm.naxRowChunk(rows: m, k: k)
        var r0 = 0
        while r0 < m {
            let r = min(chunk, m - r0)
            if r0 > 0 {
                x.slice(byteOffset: r0 * k * 2, count: r * k).bind(enc, index: 1)
                qkv16.slice(byteOffset: r0 * qkvC * 2, count: r * qkvC).bind(enc, index: 2)
                y32.slice(byteOffset: r0 * n * 4, count: r * n).bind(enc, index: 3)
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
                          m: Int, i: Int) {
        enc.setComputePipelineState(psoSiluMulF16)
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
        // The frozen writer's row stride and its per-head scale/group map
        // are the 27B's 4 heads × 256 halves, so it serves that shape only;
        // every other shape takes the general writer (f16, q8 and q4 pages).
        let general = !geometry.is27BKVShape
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

    /// Whether the attention zero-pad serves this geometry.
    public static func zeroPadServes(geometry: QWModelGeometry) -> Bool {
        geometry.is27BShape
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
                        h16: QWTensorValue, fuseFwht32: Bool = false) {
        _ = gs
        let pso: any MTLComputePipelineState
        if !wQ4.isIncumbentContainer,
           (wQ4.bits != weightFormat.bits || wQ4.groupSize != weightFormat.groupSize
               || wQ4.isSHVQ != weightFormat.denseIsSHVQ || weightFormat.denseIsIncumbentContainer),
           let name = QWGemmAQ.siluPairKernelName(bits: wQ4.bits, group: wQ4.groupSize, isSHVQ: wQ4.isSHVQ),
           let customPSO = try? hal.pipeline(name) {
            pso = customPSO
        } else {
            pso = psoGEMVSiluPairH16
        }
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
    private func checkNormGeometry() throws {
        guard geometry.hidden % 256 == 0 else {
            throw QWEngineDriverError.state(
                "the norm family needs a hidden that is a multiple of 256, got \(geometry.hidden)")
        }
    }

    func encodeRMSNormResidualFast(_ enc: any MTLComputeCommandEncoder,
                                           x: QWTensorValue, gamma: QWTensorValue,
                                           out: QWTensorValue, residual: QWTensorValue,
                                           rows: Int = 1, rowParallel: Bool = true) throws {
        try checkNormGeometry()
        let hiddenU4 = geometry.hidden / 8
        // Decode/verify use a threadgroup per row (not latency-bound, and verify matches serial
        // decode numerics); prefill fills the GPU anyway and keeps the simdgroup kernel
        // (`rowParallel: false`) so its output is unchanged.
        let wide = rowParallel && hiddenU4 % 32 == 0 && hiddenU4 <= 640
        enc.setComputePipelineState(wide ? psoRMSNormResidualWide : psoRMSNormResidual)
        enc.bindTensors(x, gamma, out, startIndex: 0)
        enc.setUInt32s(hiddenU4, startIndex: 3)
        enc.setPod(geometry.rmsEps, index: 4)
        residual.bind(enc, index: 5)
        if wide {
            enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: hiddenU4, height: 1, depth: 1))
            return
        }
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    func encodeMHCEnter(_ enc: any MTLComputeCommandEncoder,
                        x: QWTensorValue, stream: QWTensorValue, rows: Int = 1) throws {
        guard let pso = psoMHCEnter else {
            throw QWEngineDriverError.state("qwg_mhc_enter_f16 pipeline is absent")
        }
        let hidden = geometry.hidden
        enc.setComputePipelineState(pso)
        enc.bindTensors(x, stream, startIndex: 0)
        enc.setUInt32s(hidden, startIndex: 2)
        enc.dispatchThreadgroups(MTLSize(width: (hidden + 255) / 256, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// QW_MHC_ROWS in qwmhc_rows.metal.
    static let mhcRouterRowsPerGroup = 4

    func encodeMHCRouterNorm(_ enc: any MTLComputeCommandEncoder,
                             stream: QWTensorValue, w: QWTensorValue, b: QWTensorValue,
                             gamma: QWTensorValue, controls: QWTensorValue,
                             out: QWTensorValue, rows: Int = 1,
                             mergeBranchOut: QWTensorValue? = nil) throws {
        try checkNormGeometry()
        guard let pso = psoMHCRouterNorm else {
            throw QWEngineDriverError.state("qwg_mhc_router_norm_f16 pipeline is absent")
        }
        let hidden = geometry.hidden
        let preScale = geometry.mhcConfig?.preScale ?? 1.0
        var iterations = UInt32(geometry.mhcConfig?.iterations ?? 20)
        if mergeBranchOut != nil {
            iterations |= 0x8000_0000
        }
        enc.setComputePipelineState(pso)
        if let split = mhcSplit, rows <= QWMHCSplitRouter.maxRows {
            let partialPSO = split.partial, finalizePSO = split.finalize, partials = split.partials
            let parts = (hidden + 63) / 64
            enc.setComputePipelineState(partialPSO)
            enc.bindTensors(stream, w, controls, startIndex: 0)
            (mergeBranchOut ?? out).bind(enc, index: 3)
            enc.setBuffer(partials, offset: 0, index: 4)
            enc.setUInt32s(hidden, startIndex: 5)
            enc.setBytes(&iterations, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: parts, height: rows, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(finalizePSO)
            stream.bind(enc, index: 0)
            enc.setBuffer(partials, offset: 0, index: 1)
            enc.bindTensors(b, gamma, out, controls, startIndex: 2)
            enc.setUInt32s(hidden, startIndex: 6)
            enc.setPod(geometry.rmsEps, index: 7)
            enc.setPod(preScale, index: 8)
            enc.setBytes(&iterations, length: 4, index: 9)
            enc.dispatchThreadgroups(MTLSize(width: (hidden + 255) / 256, height: rows, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            return
        }
        let rowsPSO = rows >= 64 ? psoMHCRouterNormRows : nil
        if let rowsPSO { enc.setComputePipelineState(rowsPSO) }
        enc.bindTensors(stream, w, b, gamma, out, controls, startIndex: 0)
        enc.setUInt32s(hidden, startIndex: 6)
        enc.setPod(geometry.rmsEps, index: 7)
        enc.setPod(preScale, index: 8)
        enc.setBytes(&iterations, length: 4, index: 9)
        (mergeBranchOut ?? out).bind(enc, index: 10)
        if rowsPSO != nil {
            enc.setUInt32s(rows, startIndex: 11)
            enc.dispatchThreadgroups(MTLSize(width: (rows + Self.mhcRouterRowsPerGroup - 1) / Self.mhcRouterRowsPerGroup,
                                             height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            return
        }
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func encodeMHCMerge(_ enc: any MTLComputeCommandEncoder,
                        stream: QWTensorValue, branchOut: QWTensorValue,
                        controls: QWTensorValue, xOut: QWTensorValue,
                        nextGamma: QWTensorValue, nextNormOut: QWTensorValue,
                        rows: Int = 1, isLeave: Bool, sinkhornIters: Int,
                        hasNextNorm: Bool) throws {
        try checkNormGeometry()
        guard let pso = psoMHCMerge else {
            throw QWEngineDriverError.state("qwg_mhc_merge_f16 pipeline is absent")
        }
        let hidden = geometry.hidden
        var flags: UInt32 = UInt32(sinkhornIters & 0xFF) << 8
        if isLeave { flags |= 1 }
        if hasNextNorm { flags |= 2 }
        enc.setComputePipelineState(pso)
        enc.bindTensors(stream, branchOut, controls, xOut, nextGamma, nextNormOut, startIndex: 0)
        enc.setUInt32s(hidden, startIndex: 6)
        enc.setPod(geometry.rmsEps, index: 7)
        enc.setUInt32s(Int(flags), startIndex: 8)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func encodeMHCExitHead(_ enc: any MTLComputeCommandEncoder,
                           stream: QWTensorValue, headW: QWTensorValue, headB: QWTensorValue,
                           normGamma: QWTensorValue, outNorm: QWTensorValue,
                           outRot32: QWTensorValue? = nil,
                           rows: Int = 1, applyNorm: Bool = true) throws {
        try checkNormGeometry()
        guard let pso = psoMHCExitHead else {
            throw QWEngineDriverError.state("qwg_mhc_exit_head_f16 pipeline is absent")
        }
        let preScale = geometry.mhcConfig?.preScale ?? 1.0
        let flags = (applyNorm ? 1 : 0) | (outRot32 != nil ? 2 : 0)
        enc.setComputePipelineState(pso)
        enc.bindTensors(stream, headW, headB, normGamma, outNorm, startIndex: 0)
        (outRot32 ?? outNorm).bind(enc, index: 5)
        enc.setUInt32s(geometry.hidden, startIndex: 6)
        enc.setPod(geometry.rmsEps, index: 7)
        enc.setPod(preScale, index: 8)
        enc.setUInt32s(flags, startIndex: 9)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func encodeTargetHead(_ enc: any MTLComputeCommandEncoder, x: QWTensorValue) throws {
        switch headStorage {
        case .f16(let weight):
            encodeFwht32Fast(enc, x: x)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(psoHeadF16)
            enc.bindTensors(weight, x, startIndex: 0)
            sc.logits.bind(enc, index: 3)
            enc.setUInt32s(geometry.vocab, geometry.hidden, startIndex: 4)
            enc.dispatchThreadgroups(MTLSize(width: geometry.vocab, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            encodeFwht32Fast(enc, x: x)
        case .q4g:
            try head.encode(enc, storage: headStorage, x: x, gs: sc.gs, y: sc.logits)
        }
        if let mm = multimaxParams {
            enc.memoryBarrier(scope: .buffers)
            try head.encodeMultiMax(enc, logits: sc.logits, params: mm, count: geometry.vocab)
        }
    }

    func encodeTargetHeadGreedy(_ enc: any MTLComputeCommandEncoder, x: QWTensorValue,
                                writeLogits: Bool = true) throws {
        if multimaxParams == nil, case .q4g(let weight) = headStorage,
           head.splits == 1 || !weight.view.isIncumbentContainer {
            try head.encodeGreedyQ4g(enc, x: x, w: weight, y: sc.logits,
                                     keysHi: sc.keysHi, keysLo: sc.keysLo, token: sc.token,
                                     writeLogits: writeLogits)
        } else {
            try encodeTargetHead(enc, x: x)
            enc.memoryBarrier(scope: .buffers)
            try head.encodeArgmax(enc, y: sc.logits, keysHi: sc.keysHi, keysLo: sc.keysLo,
                                  token: sc.token, N: geometry.vocab)
        }
    }

    // MARK: - small CPU-side copies (engine-private; steady allocs stay 0) ──
    /// C5 FIX: token ids arrive from the tokenizer and from public API
    /// callers — an id outside 0…vocab−1 would read the embed slab out of
    func copyEmbedRow(_ token: Int, to dst: QWTensorValue) throws {
        guard token >= 0, token < geometry.vocab else { throw QWEngineDriverError.badToken(token) }
        let H = geometry.hidden
        let dstBase = dst.region.buffer.contents().advanced(by: dst.region.base + dst.offset)
            .assumingMemoryBound(to: UInt16.self)
        if let pack {
            let byteOffset = embedRawOff + token * H * 2
            try pack.withRawPointer(at: byteOffset, byteCount: H * 2) { raw in
                let src = raw.assumingMemoryBound(to: UInt16.self)
                QWQuant.bf16ToF16Array(src: src, dst: dstBase, count: H)
            }
            return
        }
        guard let embedReader else {
            throw QWEngineDriverError.weightMissing(embedPlane, "no embedding reader")
        }
        if embedReader.tensorInfo.dtype == .u32,
           let sReader = embedScalesReader,
           let bReader = embedBiasesReader {
            let packedCols = embedReader.tensorInfo.shape.count == 2
                ? embedReader.tensorInfo.shape[1]
                : (embedReader.tensorInfo.elementCount ?? 0) / geometry.vocab
            let groups = sReader.tensorInfo.shape.count == 2
                ? sReader.tensorInfo.shape[1]
                : (sReader.tensorInfo.elementCount ?? 0) / geometry.vocab
            let groupSize = H / max(1, groups)
            let bits = (packedCols * 32) / H
            let sIsF16 = (sReader.tensorInfo.dtype == .f16)
            embedReader.withBytes { wRaw in
                sReader.withBytes { sRaw in
                    bReader.withBytes { bRaw in
                        let w32 = wRaw.bindMemory(to: UInt32.self)
                        guard let wPtr = w32.baseAddress else { return }
                        let s16 = sRaw.bindMemory(to: UInt16.self)
                        let b16 = bRaw.bindMemory(to: UInt16.self)
                        Self.decodeQuantizedEmbedRow(
                            tok: token, H: H, packedCols: packedCols, groups: groups,
                            groupSize: groupSize, bits: bits, sIsF16: sIsF16,
                            wPtr: wPtr, s16: s16, b16: b16, out: dstBase
                        )
                    }
                }
            }
            return
        }
        let isF16 = (embedReader.tensorInfo.dtype == .f16)
        embedReader.withBytes { rawBuf in
            guard let base = rawBuf.baseAddress else { return }
            let src = base.advanced(by: token * H * 2).assumingMemoryBound(to: UInt16.self)
            if isF16 {
                memcpy(dstBase, src, H * 2)
            } else {
                QWQuant.bf16ToF16Array(src: src, dst: dstBase, count: H)
            }
        }
    }

    private static func decodeQuantizedEmbedRow(
        tok: Int, H: Int, packedCols: Int, groups: Int,
        groupSize: Int, bits: Int, sIsF16: Bool,
        wPtr: UnsafePointer<UInt32>,
        s16: UnsafeBufferPointer<UInt16>,
        b16: UnsafeBufferPointer<UInt16>,
        out: UnsafeMutablePointer<UInt16>
    ) {
        let rowWords = wPtr.advanced(by: tok * packedCols)
        let sBase = tok * groups
        if bits == 4 && groupSize % 8 == 0 {
            let wordsPerGroup = groupSize / 8
            for g in 0..<groups {
                let sc: Float = sIsF16
                    ? Float(Float16(bitPattern: s16[sBase + g]))
                    : Float(bitPattern: UInt32(s16[sBase + g]) << 16)
                let bi: Float = sIsF16
                    ? Float(Float16(bitPattern: b16[sBase + g]))
                    : Float(bitPattern: UInt32(b16[sBase + g]) << 16)
                let wGroup = rowWords.advanced(by: g * wordsPerGroup)
                let outGroup = out.advanced(by: g * groupSize)
                for wIdx in 0..<wordsPerGroup {
                    var word = wGroup[wIdx]
                    let cBase = wIdx * 8
                    for k in 0..<8 {
                        let q = Float(word & 0xF)
                        word >>= 4
                        outGroup[cBase + k] = QWQuant.f32tof16Bits(q * sc + bi)
                    }
                }
            }
        } else if bits == 2 && groupSize % 16 == 0 {
            let wordsPerGroup = groupSize / 16
            for g in 0..<groups {
                let sc: Float = sIsF16
                    ? Float(Float16(bitPattern: s16[sBase + g]))
                    : Float(bitPattern: UInt32(s16[sBase + g]) << 16)
                let bi: Float = sIsF16
                    ? Float(Float16(bitPattern: b16[sBase + g]))
                    : Float(bitPattern: UInt32(b16[sBase + g]) << 16)
                let wGroup = rowWords.advanced(by: g * wordsPerGroup)
                let outGroup = out.advanced(by: g * groupSize)
                for wIdx in 0..<wordsPerGroup {
                    var word = wGroup[wIdx]
                    let cBase = wIdx * 16
                    for k in 0..<16 {
                        let q = Float(word & 0x3)
                        word >>= 2
                        outGroup[cBase + k] = QWQuant.f32tof16Bits(q * sc + bi)
                    }
                }
            }
        } else {
            for g in 0..<groups {
                let sc: Float = sIsF16
                    ? Float(Float16(bitPattern: s16[sBase + g]))
                    : Float(bitPattern: UInt32(s16[sBase + g]) << 16)
                let bi: Float = sIsF16
                    ? Float(Float16(bitPattern: b16[sBase + g]))
                    : Float(bitPattern: UInt32(b16[sBase + g]) << 16)
                let cStart = g * groupSize
                let cEnd = min(H, cStart + groupSize)
                for c in cStart..<cEnd {
                    let q = QWBF16.unpackMLXAffineQ(rowWords: rowWords, col: c, bits: bits)
                    out[c] = QWQuant.f32tof16Bits(q * sc + bi)
                }
            }
        }
    }

    func copyEmbedRows(_ tokens: [Int], start: Int, count: Int, to dst: QWTensorValue) throws {
        guard count > 0 else { return }
        if count == 1 {
            try copyEmbedRow(tokens[start], to: dst)
            return
        }
        for t in 0..<count {
            let tok = tokens[start + t]
            guard tok >= 0, tok < geometry.vocab else { throw QWEngineDriverError.badToken(tok) }
        }
        let H = geometry.hidden
        let dstBase = dst.region.buffer.contents().advanced(by: dst.region.base + dst.offset)
            .assumingMemoryBound(to: UInt16.self)
        if let pack {
            for t in 0..<count {
                let tok = tokens[start + t]
                let byteOffset = embedRawOff + tok * H * 2
                try pack.withRawPointer(at: byteOffset, byteCount: H * 2) { raw in
                    let src = raw.assumingMemoryBound(to: UInt16.self)
                    QWQuant.bf16ToF16Array(src: src, dst: dstBase.advanced(by: t * H), count: H)
                }
            }
            return
        }
        guard let embedReader else {
            throw QWEngineDriverError.weightMissing(embedPlane, "no embedding reader")
        }
        if embedReader.tensorInfo.dtype == .u32,
           let sReader = embedScalesReader,
           let bReader = embedBiasesReader {
            let packedCols = embedReader.tensorInfo.shape.count == 2
                ? embedReader.tensorInfo.shape[1]
                : (embedReader.tensorInfo.elementCount ?? 0) / geometry.vocab
            let groups = sReader.tensorInfo.shape.count == 2
                ? sReader.tensorInfo.shape[1]
                : (sReader.tensorInfo.elementCount ?? 0) / geometry.vocab
            let groupSize = H / max(1, groups)
            let bits = (packedCols * 32) / H
            let sIsF16 = (sReader.tensorInfo.dtype == .f16)
            embedReader.withBytes { wRaw in
                sReader.withBytes { sRaw in
                    bReader.withBytes { bRaw in
                        let w32 = wRaw.bindMemory(to: UInt32.self)
                        guard let wPtr = w32.baseAddress else { return }
                        let s16 = sRaw.bindMemory(to: UInt16.self)
                        let b16 = bRaw.bindMemory(to: UInt16.self)
                        let wAddr = Int(bitPattern: wPtr)
                        let sAddr = Int(bitPattern: s16.baseAddress!)
                        let bAddr = Int(bitPattern: b16.baseAddress!)
                        let dstAddr = Int(bitPattern: dstBase)
                        let sCount = s16.count
                        let bCount = b16.count
                        if count >= 16 {
                            DispatchQueue.concurrentPerform(iterations: count) { t in
                                let wP = UnsafePointer<UInt32>(bitPattern: wAddr)!
                                let sBuf = UnsafeBufferPointer(start: UnsafePointer<UInt16>(bitPattern: sAddr), count: sCount)
                                let bBuf = UnsafeBufferPointer(start: UnsafePointer<UInt16>(bitPattern: bAddr), count: bCount)
                                let outP = UnsafeMutablePointer<UInt16>(bitPattern: dstAddr)!.advanced(by: t * H)
                                Self.decodeQuantizedEmbedRow(
                                    tok: tokens[start + t], H: H, packedCols: packedCols, groups: groups,
                                    groupSize: groupSize, bits: bits, sIsF16: sIsF16,
                                    wPtr: wP, s16: sBuf, b16: bBuf, out: outP
                                )
                            }
                        } else {
                            for t in 0..<count {
                                Self.decodeQuantizedEmbedRow(
                                    tok: tokens[start + t], H: H, packedCols: packedCols, groups: groups,
                                    groupSize: groupSize, bits: bits, sIsF16: sIsF16,
                                    wPtr: wPtr, s16: s16, b16: b16, out: dstBase.advanced(by: t * H)
                                )
                            }
                        }
                    }
                }
            }
            return
        }
        let isF16 = (embedReader.tensorInfo.dtype == .f16)
        embedReader.withBytes { rawBuf in
            guard let base = rawBuf.baseAddress else { return }
            let baseAddr = Int(bitPattern: base)
            let dstAddr = Int(bitPattern: dstBase)
            if count >= 64 && !isF16 {
                DispatchQueue.concurrentPerform(iterations: count) { t in
                    let tok = tokens[start + t]
                    let src = UnsafeRawPointer(bitPattern: baseAddr)!.advanced(by: tok * H * 2).assumingMemoryBound(to: UInt16.self)
                    let out = UnsafeMutablePointer<UInt16>(bitPattern: dstAddr)!.advanced(by: t * H)
                    QWQuant.bf16ToF16Array(src: src, dst: out, count: H)
                }
            } else {
                for t in 0..<count {
                    let tok = tokens[start + t]
                    let src = base.advanced(by: tok * H * 2).assumingMemoryBound(to: UInt16.self)
                    let out = dstBase.advanced(by: t * H)
                    if isF16 {
                        memcpy(out, src, H * 2)
                    } else {
                        QWQuant.bf16ToF16Array(src: src, dst: out, count: H)
                    }
                }
            }
        }
    }

    func applyVisualSpans(start: Int, count: Int, to dst: QWTensorValue) {
        guard !activeVisualSpans.isEmpty, count > 0 else { return }
        let H = geometry.hidden
        let dstBase = dst.region.buffer.contents().advanced(by: dst.region.base + dst.offset)
            .assumingMemoryBound(to: UInt16.self)
        for span in activeVisualSpans {
            let lo = max(start, span.tokenOffset)
            let hi = min(start + count, span.tokenOffset + span.tokenCount)
            guard lo < hi else { continue }
            span.embeddings.withUnsafeBufferPointer { embBuf in
                guard let embBase = embBuf.baseAddress else { return }
                for absPos in lo..<hi {
                    let row = absPos - start
                    let visRow = absPos - span.tokenOffset
                    guard (visRow + 1) * H <= embBuf.count else { continue }
                    let src = embBase.advanced(by: visRow * H)
                    let out = dstBase.advanced(by: row * H)
                    for c in 0..<H {
                        out[c] = Float16(src[c]).bitPattern
                    }
                }
            }
        }
    }
}
