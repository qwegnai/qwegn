// QWAttnApply.swift — engine-side encoder for the C08 prefill attention.
//
// Contract mirrors qwattn.metal exactly: grid = ⌈S/32⌉·24 WORKGROUPS of 128
// threads (dispatchThreadgroups — dispatchThreads would dispatch 24 THREADS
// and leave 3.5 simdgroups inactive, the zero-strip bug of C08). Inputs must
// carry the +32 zero PAD ROWS the collective loads cannot clamp (fragment
// loads run unmasked over tile tails; padded rows produce S̃ = 0, are masked
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc

public enum QWAttnError: Error, CustomStringConvertible {
    case badShape(String)
    public var description: String { switch self { case .badShape(let s): return "QWAttnError badShape: \(s)" } }
}

/// Encoder for `qwg_attn_prefill`. Stateless after init (no host tables).
public final class QWAttnPrefill {
    public enum Backend: Sendable {
        case scalar
        case nax
        case naxX64
        /// EXP-9 sub-item 2 (opt-in, NOT shipping default): fully-participating
        /// work distribution — all 4 simdgroups own a disjoint 16-row query
        /// strip (BQ=64/threadgroup vs 32 for naxX64). See qwattn_nax.metal
        /// `qwg_attn_prefill_nax_quad` header for the design rationale.
        case naxQuad
    }

    /// Extra f16 rows every attention input tensor must carry past S.
    public static let padRows = 32
    public let geometry: QWModelGeometry
    private let hal: QWHal
    private let pso: any MTLComputePipelineState
    public let kernelName: String
    public let threadsPerThreadgroup: Int
    public let tileRows: Int
    public var padRows: Int { tileRows }

    public init(hal: QWHal, backend: Backend = .nax, geometry: QWModelGeometry) throws {
        self.geometry = geometry
        self.hal = hal
        // NAX swap GATE (2026-10-04, all arms green in QWAttnPrefillTests):
        // testPrefillParityNax = 0 violations vs the committed CPU goldens at
        // S ∈ {7,16,32,128,512} + _adv30 + _ties (multi-key-block online-
        // softmax carry proven; v1's epilogue gate-row double-offset fixed —
        // tile 0 masked it because tile0=0 made the offset a no-op). Side-by-
        // side race: BIT-IDENTICAL to v5.1 at S=1024 and S=4096 (0/25M cells
        // differ), ~4× faster; 1-ulp P̂ rounding flips only at S=16384, above
        // PrefillScratch.maxChunk=4096 ⇒ unreachable by the engine. Golden
        // (golden_pre_p3) re-check is the operator's final gate before merge.
        switch backend {
        case .scalar:
            self.kernelName = "qwg_attn_prefill"
            self.threadsPerThreadgroup = 128
            self.tileRows = 32
        case .nax:
            self.kernelName = "qwg_attn_prefill_nax"
            self.threadsPerThreadgroup = 128
            self.tileRows = 32
        case .naxX64:
            self.kernelName = "qwg_attn_prefill_nax_x64"
            self.threadsPerThreadgroup = 128
            self.tileRows = 32
        case .naxQuad:
            self.kernelName = "qwg_attn_prefill_nax_quad"
            self.threadsPerThreadgroup = 128
            self.tileRows = 64
        }
        self.pso = try hal.pipeline(self.kernelName)
    }

    /// Encode one causal GQA prefill. q [S+32][12288] (Q‖gate interleaved),
    /// k/v [S+32][1024] (zero pad rows required — see header), out
    /// [S][6144] = o_proj input. Rows S..<S+32 of inputs are never read for
    public func encode(_ enc: any MTLComputeCommandEncoder,
                       qgate: QWTensorValue, k: QWTensorValue, v: QWTensorValue,
                       out: QWTensorValue, S: Int) throws {
        let sp = (S + tileRows - 1) / tileRows * tileRows
        let wantQGate = sp * geometry.qProjOut
        let wantKV = sp * geometry.kvHeads * geometry.headDim
        let wantOut = S * geometry.qHeads * geometry.headDim
        guard S > 0, qgate.shape.dtype == .f16, k.shape.dtype == .f16, v.shape.dtype == .f16,
              out.shape.dtype == .f16,
              qgate.shape.elementCount == wantQGate,
              k.shape.elementCount == wantKV, v.shape.elementCount == wantKV,
              out.shape.elementCount == wantOut
        else { throw QWAttnError.badShape("S=\(S): inputs need \(wantQGate)/\(wantKV) f16 (+\(tileRows) pad rows), out \(wantOut)") }
        enc.setComputePipelineState(pso)
        qgate.bind(enc, index: 0)
        k.bind(enc, index: 1)
        v.bind(enc, index: 2)
        out.bind(enc, index: 3)
        var sU = UInt32(S)
        enc.setBytes(&sU, length: 4, index: 4)
        let tiles = (S + tileRows - 1) / tileRows   // pad rows guard tail loads; they add no work
        enc.dispatchThreadgroups(MTLSize(width: tiles * geometry.qHeads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1))
    }
}
