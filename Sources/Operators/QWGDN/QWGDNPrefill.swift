// QWGDNPrefill.swift — engine-side encoder for the C13 chunked GDN prefill.
//
// Three launches per prefill (math validated fp64-vs-naive in
// gen_gdn_prefill_fixtures.py before this file existed):
//   qwg_gdn_prefill_solve  grid (⌈T/L⌉ × 48) × 128 — fully PARALLEL per
//       (chunk, head): forward-substitution of the unit-diag UT system for
//       U (δ without the state term) and W (its state-term kernel), plus
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc

public enum QWGDNPrefillError: Error, CustomStringConvertible {
    case badShape(String)
    public var description: String { switch self { case .badShape(let s): return "QWGDNPrefillError badShape: \(s)" } }
}

public final class QWGDNPrefill {
    public enum ScanBackend: Sendable {
        case scalar
        case nax
        case hybrid
    }

    public enum SolveBackend: String, Sendable {
        case scalar = "qwg_gdn_prefill_solve"
        case nilpotent = "qwg_gdn_prefill_solve_nilpotent"
    }

    public static let vHeads = 48
    public static let headDim = 128
    public static let maxChunk = 64
    /// f32 elements of ONE of U/W/A/B scratch tensors.
    public static func uwabElems(T: Int) -> Int { T * 6144 }
    /// f32 elements of o_scratch.
    public static func oScratchElems(T: Int) -> Int { T * 6144 }
    /// Active scan kernel name (used by parity gates to assert backend routing).
    public let kernelNameScan: String
    public let kernelNameSolve: String
    public let scanThreadsPerThreadgroup: Int
    public var solveBackend: SolveBackend

    private let hal: QWHal
    private let psoSolve: any MTLComputePipelineState
    private let psoScan: any MTLComputePipelineState
    private let psoNorm: any MTLComputePipelineState
    /// EXP-9 sub-item 1 (opt-in, NOT shipping default): register-resident
    /// recurrence, lazily built so callers that never opt in pay nothing.
    private lazy var psoRegRes: any MTLComputePipelineState = { try! hal.pipeline("qwg_gdn_prefill_regres") }()

    public init(hal: QWHal, scanBackend: ScanBackend = .hybrid, solveBackend: SolveBackend = .scalar) throws {
        self.hal = hal
        self.solveBackend = solveBackend
        switch scanBackend {
        case .scalar:
            self.kernelNameScan = "qwg_gdn_prefill_scan"
            self.scanThreadsPerThreadgroup = 128
        case .nax:
            self.kernelNameScan = "qwg_gdn_prefill_scan_nax"
            self.scanThreadsPerThreadgroup = 32
        case .hybrid:
            self.kernelNameScan = "qwg_gdn_prefill_scan_hybrid"
            self.scanThreadsPerThreadgroup = 128
        }
        self.kernelNameSolve = solveBackend.rawValue
        self.psoSolve = try hal.pipeline(self.kernelNameSolve)
        self.psoScan = try hal.pipeline(self.kernelNameScan)
        self.psoNorm = try hal.pipeline("qwg_gdn_prefill_norm")
    }

    private var instVHeads: Int { max(1, hal.geometry.gdnVHeads) }
    private var instHeadDim: Int { max(1, hal.geometry.gdnHeadDim) }
    private var instQKVCh: Int { max(1, hal.geometry.gdnQKVChannels) }
    private var instZCh: Int { max(1, hal.geometry.gdnZChannels) }
    private var instGlogCh: Int { max(1, hal.geometry.gdnGlogChannels) }

    /// Encode a whole prefill of T tokens, chunk length L (≤ 64). `state`
    /// is the persistent fp32 state, updated IN PLACE (start-of-sequence =
    /// zeros; a resumed sequence passes its C12-consistent state).
    public func encode(_ enc: any MTLComputeCommandEncoder,
                       y: QWTensorValue, betas: QWTensorValue, glog: QWTensorValue,
                       z: QWTensorValue, w: QWTensorValue,
                       state: QWTensorValue, o: QWTensorValue,
                       U: QWTensorValue, W: QWTensorValue, A: QWTensorValue, B: QWTensorValue,
                       oScratch: QWTensorValue, T: Int, L: Int,
                       kernelName: String? = nil,
                       solveKernelName: String? = nil,
                       solveBackend: SolveBackend? = nil) throws {
        if hal.geometry.gdnVectorDecay {
            try encodeRegRes(enc, y: y, betas: betas, glog: glog, z: z, w: w,
                             state: state, o: o, oScratch: oScratch, T: T)
            return
        }
        let vh = instVHeads, hd = instHeadDim, qkv = instQKVCh, zCh = instZCh, gCh = instGlogCh
        let uw = T * zCh
        func ok(_ t: QWTensorValue, _ n: Int, _ dt: QWDType) -> Bool {
            t.shape.elementCount == n && t.shape.dtype == dt
                && t.region.length >= Int((Double(n) * dt.bytesPerParam).rounded())
        }
        guard T > 0, L > 0, L <= Self.maxChunk,
              ok(y, T * qkv, .f16), ok(betas, T * vh, .f32), ok(glog, T * gCh, .f32),
              ok(z, T * zCh, .f16), ok(w, hd, .f16),
              ok(state, vh * hd * hd, .f32), ok(o, T * zCh, .f16),
              ok(U, uw, .f32), ok(W, uw, .f32), ok(A, uw, .f32), ok(B, uw, .f32),
              ok(oScratch, uw, .f32)
        else { throw QWGDNPrefillError.badShape("T=\(T) L=\(L) y=\(y.shape.elementCount)") }

        var tU = UInt32(T), lU = UInt32(L)
        let chunks = (T + L - 1) / L

        let effectiveSolveKernel = solveKernelName ?? solveBackend?.rawValue ?? self.solveBackend.rawValue
        if effectiveSolveKernel == self.kernelNameSolve {
            enc.setComputePipelineState(psoSolve)
        } else {
            enc.setComputePipelineState(try hal.pipeline(effectiveSolveKernel))
        }
        y.bind(enc, index: 0); betas.bind(enc, index: 1); glog.bind(enc, index: 2)
        U.bind(enc, index: 3); W.bind(enc, index: 4); A.bind(enc, index: 5); B.bind(enc, index: 6)
        enc.setBytes(&tU, length: 4, index: 7); enc.setBytes(&lU, length: 4, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: chunks, height: vh, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: hd, height: 1, depth: 1))

        if let kn = kernelName { enc.setComputePipelineState(try hal.pipeline(kn)) }
        else { enc.setComputePipelineState(psoScan) }
        y.bind(enc, index: 0)
        U.bind(enc, index: 1); W.bind(enc, index: 2); A.bind(enc, index: 3); B.bind(enc, index: 4)
        glog.bind(enc, index: 5)
        state.bind(enc, index: 6)            // IN PLACE
        oScratch.bind(enc, index: 7)
        enc.setBytes(&tU, length: 4, index: 8); enc.setBytes(&lU, length: 4, index: 9)
        enc.dispatchThreadgroups(MTLSize(width: vh, height: 8, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: scanThreadsPerThreadgroup, height: 1, depth: 1))

        enc.setComputePipelineState(psoNorm)
        oScratch.bind(enc, index: 0)
        z.bind(enc, index: 1)
        w.bind(enc, index: 2)
        o.bind(enc, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: T * vh, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: hd, height: 1, depth: 1))
    }

    /// EXP-9 sub-item 1: 192-TG zero-barrier register-resident recurrence
    /// (`qwg_gdn_prefill_regres`) followed by `qwg_gdn_prefill_norm`.
    public func encodeRegRes(_ enc: any MTLComputeCommandEncoder,
                             y: QWTensorValue, betas: QWTensorValue, glog: QWTensorValue,
                             z: QWTensorValue, w: QWTensorValue,
                             state: QWTensorValue, o: QWTensorValue,
                             oScratch: QWTensorValue, T: Int) throws {
        let vh = instVHeads, hd = instHeadDim, qkv = instQKVCh, zCh = instZCh, gCh = instGlogCh
        func ok(_ t: QWTensorValue, _ n: Int, _ dt: QWDType) -> Bool {
            t.shape.elementCount == n && t.shape.dtype == dt
                && t.region.length >= Int((Double(n) * dt.bytesPerParam).rounded())
        }
        guard T > 0,
              ok(y, T * qkv, .f16), ok(betas, T * vh, .f32), ok(glog, T * gCh, .f32),
              ok(z, T * zCh, .f16), ok(w, hd, .f16),
              ok(state, vh * hd * hd, .f32), ok(o, T * zCh, .f16),
              ok(oScratch, T * zCh, .f32)
        else { throw QWGDNPrefillError.badShape("regres T=\(T) y=\(y.shape.elementCount)") }

        var tU = UInt32(T)
        enc.setComputePipelineState(psoRegRes)
        y.bind(enc, index: 0); betas.bind(enc, index: 1); glog.bind(enc, index: 2)
        state.bind(enc, index: 3)        // IN PLACE
        oScratch.bind(enc, index: 4)
        enc.setBytes(&tU, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: vh, height: 4, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: hd, height: 1, depth: 1))

        enc.setComputePipelineState(psoNorm)
        oScratch.bind(enc, index: 0)
        z.bind(enc, index: 1)
        w.bind(enc, index: 2)
        o.bind(enc, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: T * vh, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: hd, height: 1, depth: 1))
    }
}
