// QWGDNDecode.swift — engine-side encoder for the C12 GDN recurrent step.
//
// Mirrors qwgdn_decode.metal: `qwg_gdn_recurrent` (grid = vHeads × 128 lanes)
// performs decay + delta-rule update of the fp32 state IN PLACE plus the fused
// GatedRMSNorm epilogue.
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc

public enum QWGDNDecodeError: Error, CustomStringConvertible {
    case badShape(String)
    public var description: String { switch self { case .badShape(let s): return "QWGDNDecodeError badShape: \(s)" } }
}

public final class QWGDNDecode {
    public static let vHeads = 48
    public static let headDim = 128

    private let hal: QWHal
    private let psoRec: any MTLComputePipelineState
    private let psoRecBatch: any MTLComputePipelineState
    private let psoRecRows: any MTLComputePipelineState
    private let psoRollbackRows: any MTLComputePipelineState
    private var cachedKDAGates: (any MTLComputePipelineState)?
    private var cachedKDAGatesMMA: (any MTLComputePipelineState)?
    private var cachedKDAGatesDirect: (any MTLComputePipelineState)?

    private func kdaGatesPipeline(mma: Bool) throws -> any MTLComputePipelineState {
        if mma {
            if let p = cachedKDAGatesMMA { return p }
            let p = try hal.pipeline("qwg_kda_gates_mma")
            cachedKDAGatesMMA = p
            return p
        }
        if let p = cachedKDAGates { return p }
        let p = try hal.pipeline("qwg_kda_gates")
        cachedKDAGates = p
        return p
    }

    private func kdaGatesDirectPipeline() throws -> any MTLComputePipelineState {
        if let p = cachedKDAGatesDirect { return p }
        let p = try hal.pipeline("qwg_kda_gates_direct")
        cachedKDAGatesDirect = p
        return p
    }
    /// Prefill rows use the 32-token simdgroup-MMA gates kernel (`QW_KDA_GATES_MMA=0` disables).
    public var kdaGatesMMAEnabled = ProcessInfo.processInfo.environment["QW_KDA_GATES_MMA"] != "0"

    private var instVHeads: Int { max(1, hal.geometry.gdnVHeads) }
    private var instHeadDim: Int { max(1, hal.geometry.gdnHeadDim) }
    private var instQKVCh: Int { max(1, hal.geometry.gdnQKVChannels) }
    private var instZCh: Int { max(1, hal.geometry.gdnZChannels) }
    private var instGlogCh: Int { max(1, hal.geometry.gdnGlogChannels) }

    public init(hal: QWHal, normSigmoid: Bool? = nil, gateScale: Float? = nil) throws {
        self.hal = hal
        self.psoRec = try hal.pipeline("qwg_gdn_recurrent", gdnNormSigmoid: normSigmoid, gdnGateScale: gateScale)
        self.psoRecBatch = try hal.pipeline("qwg_gdn_recurrent_batch", gdnNormSigmoid: normSigmoid, gdnGateScale: gateScale)
        self.psoRecRows = try hal.pipeline("qwg_gdn_recurrent_rows", gdnNormSigmoid: normSigmoid, gdnGateScale: gateScale)
        self.psoRollbackRows = try hal.pipeline("qwg_gdn_rollback_rows")
        if hal.geometry.gdnVectorDecay {
            if hal.geometry.kdaConfig?.isFullRank == true {
                _ = try hal.pipeline("qwg_kda_gates_direct")
            } else {
                _ = try hal.pipeline("qwg_kda_gates")
            }
        }
    }

    public func encodeKDAGates(_ enc: any MTLComputeCommandEncoder,
                               fgb: QWTensorValue, fB: QWTensorValue, gB: QWTensorValue,
                               aLog: QWTensorValue, dtBias: QWTensorValue,
                               betas: QWTensorValue, glog: QWTensorValue, zOut: QWTensorValue,
                               rows: Int = 1, lowRankDim: Int? = nil) throws {
        let vh = instVHeads, hd = instHeadDim
        guard rows > 0 else { return }
        let lr = lowRankDim ?? hal.geometry.kdaConfig?.lowRankDim ?? hal.geometry.gdnLowRankDim
        guard lr > 0, lr <= 256, lr % 4 == 0 else {
            throw QWGDNDecodeError.badShape("KDA lowRankDim \(lr) must be multiple of 4 in 4...256")
        }
        let useMMA = kdaGatesMMAEnabled && rows >= 16 && (lr % 8 == 0) && hd <= 128
        enc.setComputePipelineState(try kdaGatesPipeline(mma: useMMA))
        fgb.bind(enc, index: 0)
        fB.bind(enc, index: 1)
        gB.bind(enc, index: 2)
        aLog.bind(enc, index: 3)
        dtBias.bind(enc, index: 4)
        betas.bind(enc, index: 5)
        glog.bind(enc, index: 6)
        zOut.bind(enc, index: 7)
        var glb = hal.geometry.gdnGateLowerBound
        var tU = UInt32(rows)
        var lrU = UInt32(lr)
        enc.setBytes(&glb, length: 4, index: 8)
        enc.setBytes(&tU, length: 4, index: 9)
        enc.setBytes(&lrU, length: 4, index: 10)
        enc.dispatchThreadgroups(MTLSize(width: vh, height: useMMA ? (rows + 31) / 32 : rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: useMMA ? 128 : hd, height: 1, depth: 1))
    }

    public func encodeKDAGatesDirect(_ enc: any MTLComputeCommandEncoder,
                                     fgb: QWTensorValue, ln1: QWTensorValue? = nil,
                                     bProjF16: QWTensorValue? = nil, aLog: QWTensorValue,
                                     dtBias: QWTensorValue, betas: QWTensorValue,
                                     glog: QWTensorValue, zOut: QWTensorValue,
                                     rows: Int = 1) throws {
        let vh = instVHeads, hd = instHeadDim
        guard rows > 0 else { return }
        let hasRawB = (ln1 != nil && bProjF16 != nil)
        enc.setComputePipelineState(try kdaGatesDirectPipeline())
        fgb.bind(enc, index: 0)
        (ln1 ?? fgb).bind(enc, index: 1)
        (bProjF16 ?? fgb).bind(enc, index: 2)
        aLog.bind(enc, index: 3)
        dtBias.bind(enc, index: 4)
        betas.bind(enc, index: 5)
        glog.bind(enc, index: 6)
        zOut.bind(enc, index: 7)
        var glb = hal.geometry.gdnGateLowerBound
        var tU = UInt32(rows)
        var rawBU: UInt32 = hasRawB ? 1 : 0
        enc.setBytes(&glb, length: 4, index: 8)
        enc.setBytes(&tU, length: 4, index: 9)
        enc.setBytes(&rawBU, length: 4, index: 10)
        enc.dispatchThreadgroups(MTLSize(width: vh, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: hd, height: 1, depth: 1))
    }

    /// Encode ONE decode token: `y` (this step's post-conv q̂‖k̂‖v, qkvCh f16),
    /// `betas`/`glog` ([vHeads] or [zCh] f32), `state` (vHeads·128·128 f32, IN PLACE),
    /// `z` (zCh f16), `w` (128 f16 norm weight), `o` (vHeads·128 f16 out),
    public func encode(_ enc: any MTLComputeCommandEncoder,
                       y: QWTensorValue, betas: QWTensorValue, glog: QWTensorValue,
                       state: QWTensorValue, z: QWTensorValue, w: QWTensorValue,
                       o: QWTensorValue, oScratch: QWTensorValue,
                       outGroupSum: QWTensorValue? = nil,
                       innovation: QWTensorValue? = nil) throws {
        let vh = instVHeads, hd = instHeadDim, qkv = instQKVCh, zCh = instZCh, gCh = instGlogCh
        let S = vh * hd * hd
        func ok(_ t: QWTensorValue, _ n: Int, _ dt: QWDType) -> Bool {
            t.shape.elementCount == n && t.shape.dtype == dt
                && t.region.length >= Int((Double(n) * dt.bytesPerParam).rounded())
        }
        guard ok(y, qkv, .f16), ok(betas, vh, .f32), ok(glog, gCh, .f32),
              ok(state, S, .f32), ok(z, zCh, .f16), ok(w, hd, .f16),
              ok(o, zCh, .f16), ok(oScratch, zCh, .f32)
        else { throw QWGDNDecodeError.badShape("y=\(y.shape.elementCount) state=\(state.shape.elementCount)") }
        if let ogs = outGroupSum {
            guard ok(ogs, vh, .f32) else {
                throw QWGDNDecodeError.badShape("outGroupSum=\(ogs.shape.elementCount)")
            }
        }
        if let innovation {
            guard ok(innovation, zCh, .f32) else {
                throw QWGDNDecodeError.badShape("innovation=\(innovation.shape.elementCount)")
            }
            var otherTensors = [y, betas, glog, state, z, w, o, oScratch]
            if let outGroupSum { otherTensors.append(outGroupSum) }
            for input in otherTensors where input.region.buffer === innovation.region.buffer {
                let a = input.region.base + input.offset
                let b = innovation.region.base + innovation.offset
                guard a + input.bytes <= b || b + innovation.bytes <= a else {
                    throw QWGDNDecodeError.badShape("innovation tape aliases a recurrent tensor")
                }
            }
        }

        enc.setComputePipelineState(psoRec)
        y.bind(enc, index: 0)
        betas.bind(enc, index: 1)
        glog.bind(enc, index: 2)
        state.bind(enc, index: 3)          // IN PLACE
        oScratch.bind(enc, index: 4)
        z.bind(enc, index: 5)
        w.bind(enc, index: 6)
        o.bind(enc, index: 7)
        if let ogs = outGroupSum {
            ogs.bind(enc, index: 8)
        } else {
            enc.setBuffer(nil, offset: 0, index: 8)
        }
        if let innovation {
            innovation.bind(enc, index: 9)
        } else {
            enc.setBuffer(nil, offset: 0, index: 9)
        }
        enc.dispatchThreadgroups(MTLSize(width: vh, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: hd * 8, height: 1, depth: 1))
    }

    private struct BatchStrides {
        var yStride: UInt32
        var betasStride: UInt32
        var glogStride: UInt32
        var zStride: UInt32
        var oStride: UInt32
        var oRawStride: UInt32
    }

    /// Encode B independent T=1 recurrent steps in one `grid.z = rows` launch (0-ULP vs serial).
    public func encodeBatch(_ enc: any MTLComputeCommandEncoder,
                            y: QWTensorValue, yStride: Int,
                            betas: QWTensorValue, betasStride: Int,
                            glog: QWTensorValue, glogStride: Int,
                            z: QWTensorValue, zStride: Int,
                            w: QWTensorValue,
                            o: QWTensorValue, oStride: Int,
                            oScratch: QWTensorValue? = nil, oScratchStride: Int = 0,
                            rows: Int,
                            stateAt: (Int) -> QWTensorValue) throws {
        let vh = instVHeads, hd = instHeadDim, qkv = instQKVCh, zCh = instZCh, gCh = instGlogCh
        guard (1...8).contains(rows), hd == 128,
              yStride >= qkv, betasStride >= vh, glogStride >= gCh, zStride >= zCh, oStride >= zCh,
              y.shape.dtype == .f16, y.shape.elementCount >= (rows - 1) * yStride + qkv,
              betas.shape.dtype == .f32, betas.shape.elementCount >= (rows - 1) * betasStride + vh,
              glog.shape.dtype == .f32, glog.shape.elementCount >= (rows - 1) * glogStride + gCh,
              z.shape.dtype == .f16, z.shape.elementCount >= (rows - 1) * zStride + zCh,
              w.shape.dtype == .f16, w.shape.elementCount == hd,
              o.shape.dtype == .f16, o.shape.elementCount >= (rows - 1) * oStride + zCh else {
            throw QWGDNDecodeError.badShape("batch recurrent shape or stride mismatch (rows=\(rows))")
        }
        if let oScratch {
            guard oScratchStride >= zCh, oScratch.shape.dtype == .f32,
                  oScratch.shape.elementCount >= (rows - 1) * oScratchStride + zCh else {
                throw QWGDNDecodeError.badShape("batch oScratch shape or stride mismatch")
            }
        }
        var ptrs = try qwGDNPackBatchStateAddresses(
            enc, rows: rows, expectedElements: vh * hd * hd, dtype: .f32, alignment: 4,
            stateAt: stateAt, badShapeError: { QWGDNDecodeError.badShape($0) })
        var strides = BatchStrides(
            yStride: UInt32(yStride), betasStride: UInt32(betasStride),
            glogStride: UInt32(glogStride), zStride: UInt32(zStride),
            oStride: UInt32(oStride), oRawStride: UInt32(max(0, oScratchStride)))
        enc.setComputePipelineState(psoRecBatch)
        y.bind(enc, index: 0)
        betas.bind(enc, index: 1)
        glog.bind(enc, index: 2)
        z.bind(enc, index: 3)
        w.bind(enc, index: 4)
        o.bind(enc, index: 5)
        if let oScratch {
            oScratch.bind(enc, index: 6)
        } else {
            enc.setBuffer(nil, offset: 0, index: 6)
        }
        enc.setBytes(&ptrs, length: MemoryLayout<QWGDNBatchPtrTable8>.stride, index: 7)
        enc.setBytes(&strides, length: MemoryLayout<BatchStrides>.stride, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: vh, height: 1, depth: rows),
                                 threadsPerThreadgroup: MTLSize(width: hd * 8, height: 1, depth: 1))
    }

    public func encodeBatch(_ enc: any MTLComputeCommandEncoder,
                            y: QWTensorValue, yStride: Int,
                            betas: QWTensorValue, betasStride: Int,
                            glog: QWTensorValue, glogStride: Int,
                            states: [QWTensorValue],
                            z: QWTensorValue, zStride: Int,
                            w: QWTensorValue,
                            o: QWTensorValue, oStride: Int,
                            oScratch: QWTensorValue? = nil, oScratchStride: Int = 0) throws {
        try encodeBatch(enc, y: y, yStride: yStride, betas: betas, betasStride: betasStride,
                        glog: glog, glogStride: glogStride, z: z, zStride: zStride, w: w,
                        o: o, oStride: oStride, oScratch: oScratch, oScratchStride: oScratchStride,
                        rows: states.count, stateAt: { states[$0] })
    }

    /// Replays the shipping recurrence without readout or gated normalization.
    /// Inputs must be the saved post-conv/headnorm rows from verification.
    public func encodeStateReplay(_ enc: any MTLComputeCommandEncoder,
                                  y: QWTensorValue, betas: QWTensorValue,
                                  glog: QWTensorValue, state: QWTensorValue) throws {
        let vh = instVHeads, hd = instHeadDim, qkv = instQKVCh, gCh = instGlogCh
        func valid(_ t: QWTensorValue, count: Int, type: QWDType) -> Bool {
            t.shape.elementCount == count && t.shape.dtype == type
                && t.offset >= 0 && t.region.base >= 0
                && t.offset <= t.region.length && t.bytes <= t.region.length - t.offset
                && t.region.base <= t.region.buffer.length
                && t.region.length <= t.region.buffer.length - t.region.base
                && (t.region.base + t.offset) % 4 == 0
        }
        guard valid(y, count: qkv, type: .f16),
              valid(betas, count: vh, type: .f32),
              valid(glog, count: gCh, type: .f32),
              valid(state, count: vh * hd * hd, type: .f32) else {
            throw QWGDNDecodeError.badShape("state replay input extent, alignment, or geometry")
        }
        func disjoint(_ input: QWTensorValue) -> Bool {
            guard input.region.buffer === state.region.buffer else { return true }
            let start = input.region.base + input.offset
            let stateStart = state.region.base + state.offset
            return start + input.bytes <= stateStart || stateStart + state.bytes <= start
        }
        guard disjoint(y), disjoint(betas), disjoint(glog) else {
            throw QWGDNDecodeError.badShape("state replay input aliases mutable state")
        }
        enc.setComputePipelineState(try hal.pipeline("qwg_gdn_replay_state"))
        y.bind(enc, index: 0)
        betas.bind(enc, index: 1)
        glog.bind(enc, index: 2)
        state.bind(enc, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: vh, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: hd * 8, height: 1, depth: 1))
    }

    /// Encode R verification tokens through the recurrence in a single dispatch,
    /// keeping the GDN state in registers across rows.
    public func encodeRows(_ enc: any MTLComputeCommandEncoder,
                           y: QWTensorValue, betas: QWTensorValue, glog: QWTensorValue,
                           state: QWTensorValue, zBase: QWTensorValue, w: QWTensorValue,
                           o: QWTensorValue, innovation: QWTensorValue? = nil,
                           rows: Int, zStride: Int? = nil,
                           stateOut: QWTensorValue? = nil,
                           unstableSuffixMask: QWTensorValue? = nil) throws {
        let vh = instVHeads, hd = instHeadDim, qkv = instQKVCh, zCh = instZCh, gCh = instGlogCh
        let effectiveZStride = zStride ?? zCh
        let targetState = stateOut ?? state
        let S = vh * hd * hd
        func ok(_ t: QWTensorValue, _ n: Int, _ dt: QWDType) -> Bool {
            t.shape.elementCount >= n && t.shape.dtype == dt
                && t.region.length >= Int((Double(n) * dt.bytesPerParam).rounded())
        }
        guard rows > 0, effectiveZStride >= zCh,
              ok(y, rows * qkv, .f16), ok(betas, rows * vh, .f32), ok(glog, rows * gCh, .f32),
              ok(state, S, .f32), ok(targetState, S, .f32),
              ok(zBase, (rows - 1) * effectiveZStride + zCh, .f16), ok(w, hd, .f16),
              ok(o, rows * zCh, .f16)
        else { throw QWGDNDecodeError.badShape("encodeRows rows=\(rows) y=\(y.shape.elementCount)") }
        if let innovation {
            guard ok(innovation, rows * zCh, .f32) else {
                throw QWGDNDecodeError.badShape("innovation=\(innovation.shape.elementCount)")
            }
        }
        if let unstableSuffixMask {
            guard ok(unstableSuffixMask, 1, .i32) else {
                throw QWGDNDecodeError.badShape("unstableSuffixMask=\(unstableSuffixMask.shape.elementCount)")
            }
        }
        enc.setComputePipelineState(psoRecRows)
        y.bind(enc, index: 0)
        betas.bind(enc, index: 1)
        glog.bind(enc, index: 2)
        state.bind(enc, index: 3)
        zBase.bind(enc, index: 4)
        w.bind(enc, index: 5)
        o.bind(enc, index: 6)
        if let innovation { innovation.bind(enc, index: 7) }
        else { enc.setBuffer(nil, offset: 0, index: 7) }
        var rU = UInt32(rows), strideU = UInt32(effectiveZStride)
        enc.setBytes(&rU, length: 4, index: 8)
        enc.setBytes(&strideU, length: 4, index: 9)
        targetState.bind(enc, index: 10)
        if let unstableSuffixMask { unstableSuffixMask.bind(enc, index: 11) }
        else { enc.setBuffer(nil, offset: 0, index: 11) }
        enc.dispatchThreadgroups(MTLSize(width: vh, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: hd * 8, height: 1, depth: 1))
    }

    /// Reverses `rows` captured normalized recurrence steps in reverse order in a single dispatch.
    public func encodeStateRollbackRows(_ enc: any MTLComputeCommandEncoder,
                                        y: QWTensorValue, innovation: QWTensorValue,
                                        glog: QWTensorValue, state: QWTensorValue,
                                        rows: Int, stateOut: QWTensorValue? = nil) throws {
        let vh = instVHeads, hd = instHeadDim, qkv = instQKVCh, zCh = instZCh, gCh = instGlogCh
        let targetState = stateOut ?? state
        func valid(_ t: QWTensorValue, count: Int, type: QWDType) -> Bool {
            t.shape.elementCount == count && t.shape.dtype == type
                && t.offset >= 0 && t.region.base >= 0
                && t.offset <= t.region.length && t.bytes <= t.region.length - t.offset
                && t.region.base <= t.region.buffer.length
                && t.region.length <= t.region.buffer.length - t.region.base
                && (t.region.base + t.offset) % 4 == 0
        }
        let stateElements = vh * hd * hd
        guard rows > 0,
              valid(y, count: rows * qkv, type: .f16),
              valid(innovation, count: rows * zCh, type: .f32),
              valid(glog, count: rows * gCh, type: .f32),
              valid(state, count: stateElements, type: .f32),
              valid(targetState, count: stateElements, type: .f32) else {
            throw QWGDNDecodeError.badShape("state rollback rows input extent, alignment, or geometry")
        }
        func disjoint(_ input: QWTensorValue, from dst: QWTensorValue) -> Bool {
            guard input.region.buffer === dst.region.buffer else { return true }
            let start = input.region.base + input.offset
            let stateStart = dst.region.base + dst.offset
            return start + input.bytes <= stateStart || stateStart + dst.bytes <= start
        }
        guard disjoint(y, from: state), disjoint(innovation, from: state), disjoint(glog, from: state),
              disjoint(y, from: targetState), disjoint(innovation, from: targetState),
              disjoint(glog, from: targetState) else {
            throw QWGDNDecodeError.badShape("state rollback rows input aliases state")
        }
        enc.setComputePipelineState(psoRollbackRows)
        y.bind(enc, index: 0)
        innovation.bind(enc, index: 1)
        glog.bind(enc, index: 2)
        state.bind(enc, index: 3)
        var rU = UInt32(rows)
        enc.setBytes(&rU, length: 4, index: 4)
        targetState.bind(enc, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: vh, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: hd * 8, height: 1, depth: 1))
    }
}
