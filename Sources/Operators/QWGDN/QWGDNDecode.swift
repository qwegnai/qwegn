// QWGDNDecode.swift — engine-side encoder for the C12 GDN recurrent step.
//
// Mirrors qwgdn_decode.metal: two launches per token —
//   qwg_gdn_recurrent    : grid (48 heads) × 128 lanes (lane owns v-column) —
//                          decay + delta-rule update of the fp32 state IN PLACE
//                          + raw o fold;
//   qwg_gdn_norm_sigmoid : grid (48 heads) × 128 lanes — GatedRMSNorm epilogue
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
    /// fp32 elements of per-step o scratch (48 heads × 128 v).
    public static let oScratchElems = vHeads * headDim

    private let hal: QWHal
    private let psoRec: any MTLComputePipelineState
    private let psoRecRows: any MTLComputePipelineState
    private let psoRollbackRows: any MTLComputePipelineState
    private lazy var psoKDAGates: any MTLComputePipelineState = { try! hal.pipeline("qwg_kda_gates") }()
    private lazy var psoKDAGatesMMA: any MTLComputePipelineState = { try! hal.pipeline("qwg_kda_gates_mma") }()
    /// Prefill rows use the 32-token simdgroup-MMA gates kernel (`QW_KDA_GATES_MMA=0` disables).
    public var kdaGatesMMAEnabled = ProcessInfo.processInfo.environment["QW_KDA_GATES_MMA"] != "0"

    private var instVHeads: Int { max(1, hal.geometry.gdnVHeads) }
    private var instHeadDim: Int { max(1, hal.geometry.gdnHeadDim) }
    private var instQKVCh: Int { max(1, hal.geometry.gdnQKVChannels) }
    private var instZCh: Int { max(1, hal.geometry.gdnZChannels) }
    private var instGlogCh: Int { max(1, hal.geometry.gdnGlogChannels) }

    public init(hal: QWHal) throws {
        self.hal = hal
        self.psoRec = try hal.pipeline("qwg_gdn_recurrent")
        self.psoRecRows = try hal.pipeline("qwg_gdn_recurrent_rows")
        self.psoRollbackRows = try hal.pipeline("qwg_gdn_rollback_rows")
        if hal.geometry.gdnVectorDecay {
            _ = try hal.pipeline("qwg_kda_gates")
        }
    }

    public func encodeKDAGates(_ enc: any MTLComputeCommandEncoder,
                               fgb: QWTensorValue, fB: QWTensorValue, gB: QWTensorValue,
                               aLog: QWTensorValue, dtBias: QWTensorValue,
                               betas: QWTensorValue, glog: QWTensorValue, zOut: QWTensorValue,
                               rows: Int = 1) throws {
        let vh = instVHeads, zCh = instZCh
        guard rows > 0 else { return }
        let useMMA = kdaGatesMMAEnabled && rows >= 16
        enc.setComputePipelineState(useMMA ? psoKDAGatesMMA : psoKDAGates)
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
        enc.setBytes(&glb, length: 4, index: 8)
        enc.setBytes(&tU, length: 4, index: 9)
        enc.dispatchThreadgroups(MTLSize(width: vh, height: useMMA ? (rows + 31) / 32 : rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        _ = zCh
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

    /// One recurrent step without the gated-norm epilogue: writes only the pre-norm readout `oScratch` ([vHeads][128] f32).
    public func encodeRaw(_ enc: any MTLComputeCommandEncoder,
                          y: QWTensorValue, betas: QWTensorValue, glog: QWTensorValue,
                          state: QWTensorValue, oScratch: QWTensorValue) throws {
        let vh = instVHeads, hd = instHeadDim, qkv = instQKVCh, zCh = instZCh, gCh = instGlogCh
        let S = vh * hd * hd
        func ok(_ t: QWTensorValue, _ n: Int, _ dt: QWDType) -> Bool {
            t.shape.elementCount == n && t.shape.dtype == dt
                && t.region.length >= Int((Double(n) * dt.bytesPerParam).rounded())
        }
        guard ok(y, qkv, .f16), ok(betas, vh, .f32), ok(glog, gCh, .f32),
              ok(state, S, .f32), ok(oScratch, zCh, .f32)
        else { throw QWGDNDecodeError.badShape("encodeRaw y=\(y.shape.elementCount) state=\(state.shape.elementCount)") }
        enc.setComputePipelineState(psoRec)
        y.bind(enc, index: 0)
        betas.bind(enc, index: 1)
        glog.bind(enc, index: 2)
        state.bind(enc, index: 3)          // IN PLACE
        oScratch.bind(enc, index: 4)
        for index in 5...9 { enc.setBuffer(nil, offset: 0, index: index) }   // z, w, o_out, out_gs, innovation: absent
        enc.dispatchThreadgroups(MTLSize(width: vh, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: hd * 8, height: 1, depth: 1))
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
                           rows: Int, zStride: Int? = nil) throws {
        let vh = instVHeads, hd = instHeadDim, qkv = instQKVCh, zCh = instZCh, gCh = instGlogCh
        let effectiveZStride = zStride ?? zCh
        let S = vh * hd * hd
        func ok(_ t: QWTensorValue, _ n: Int, _ dt: QWDType) -> Bool {
            t.shape.elementCount >= n && t.shape.dtype == dt
                && t.region.length >= Int((Double(n) * dt.bytesPerParam).rounded())
        }
        guard rows > 0, effectiveZStride >= zCh,
              ok(y, rows * qkv, .f16), ok(betas, rows * vh, .f32), ok(glog, rows * gCh, .f32),
              ok(state, S, .f32), ok(zBase, (rows - 1) * effectiveZStride + zCh, .f16), ok(w, hd, .f16),
              ok(o, rows * zCh, .f16)
        else { throw QWGDNDecodeError.badShape("encodeRows rows=\(rows) y=\(y.shape.elementCount)") }
        if let innovation {
            guard ok(innovation, rows * zCh, .f32) else {
                throw QWGDNDecodeError.badShape("innovation=\(innovation.shape.elementCount)")
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
        enc.dispatchThreadgroups(MTLSize(width: vh, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: hd * 8, height: 1, depth: 1))
    }

    /// Reverses `rows` captured normalized recurrence steps in reverse order in a single dispatch.
    public func encodeStateRollbackRows(_ enc: any MTLComputeCommandEncoder,
                                        y: QWTensorValue, innovation: QWTensorValue,
                                        glog: QWTensorValue, state: QWTensorValue,
                                        rows: Int) throws {
        let vh = instVHeads, hd = instHeadDim, qkv = instQKVCh, zCh = instZCh, gCh = instGlogCh
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
              valid(state, count: stateElements, type: .f32) else {
            throw QWGDNDecodeError.badShape("state rollback rows input extent, alignment, or geometry")
        }
        func disjoint(_ input: QWTensorValue) -> Bool {
            guard input.region.buffer === state.region.buffer else { return true }
            let start = input.region.base + input.offset
            let stateStart = state.region.base + state.offset
            return start + input.bytes <= stateStart || stateStart + state.bytes <= start
        }
        guard disjoint(y), disjoint(innovation), disjoint(glog) else {
            throw QWGDNDecodeError.badShape("state rollback rows input aliases mutable state")
        }
        enc.setComputePipelineState(psoRollbackRows)
        y.bind(enc, index: 0)
        innovation.bind(enc, index: 1)
        glog.bind(enc, index: 2)
        state.bind(enc, index: 3)
        var rU = UInt32(rows)
        enc.setBytes(&rU, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: vh, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: hd * 8, height: 1, depth: 1))
    }
}
