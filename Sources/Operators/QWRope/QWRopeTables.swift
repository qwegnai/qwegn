import Foundation
import Dispatch
import QWCore

/// C07 — QWRope: host-side rope table + fused norm+rope apply-kernel
/// (specs/07-qwrope.md). Table math mirrors the HF qwen3_5 recipe
/// (transformers 5.17, pinned): default inv_freq θ=1e7 over the 32 rotary
/// pairs (float32 tensor math, mirrored with Float-only ops), YaRN per
/// _compute_yarn_parameters, interleaved-mrope per
/// Qwen3_5TextRotaryEmbedding.recomposition_frequencies.
///
/// Precision law: everything HF does in float32 tensors we do in Float
/// (single-rounding like torch f32); the scalar ramp bounds (find_correction_
/// dim) are Python floats = Double, mirrored in Double. The parity gate
/// compares tables vs the torch-generated fixture at ≤ 1e-5 maxabs (spec) —
/// pow/log/libm may differ ≤1 ulp between Swift and torch; we assert the
/// measured bound, not bitwise.
///
/// TEXT-ONLY fast path (the engine's decode/prefill reality): position.t =
/// position.h = position.w ⇒ the (11,11,10) interleaved selection is a NO-OP;
/// `cosSinRowMrope` still routes per-pair axes for the vision path.

public struct QWYaRNConfig: Sendable, Hashable {
    public let factor: Float
    public let originalMaxPosition: Int
    public let betaFast: Float
    public let betaSlow: Float
    public let truncate: Bool
    public init(factor: Float, originalMaxPosition: Int = QWModelGeometry.qwen38_27b.maxPositionEmbeddings,
                betaFast: Float = 32, betaSlow: Float = 1, truncate: Bool = true) {
        self.factor = factor; self.originalMaxPosition = originalMaxPosition
        self.betaFast = betaFast; self.betaSlow = betaSlow; self.truncate = truncate
    }

    public static func config1M(
        targetContext: Int = 1_048_576,
        originalMaxPosition: Int = QWModelGeometry.qwen38_27b.maxPositionEmbeddings,
        betaFast: Float = 32,
        betaSlow: Float = 1,
        truncate: Bool = true
    ) -> QWYaRNConfig {
        let factor = Float(targetContext) / Float(originalMaxPosition)
        return QWYaRNConfig(
            factor: factor,
            originalMaxPosition: originalMaxPosition,
            betaFast: betaFast,
            betaSlow: betaSlow,
            truncate: truncate
        )
    }
}

public enum QWRopeError: Error, CustomStringConvertible {
    case badShape(String)
    case posBeyondTable(Int, Int)
    public var description: String {
        switch self {
        case .badShape(let s): return "QWRope: \(s)"
        case .posBeyondTable(let p, let n): return "QWRope: position \(p) beyond table rows \(n)"
        }
    }
}

/// Built once at engine start (or rebuilt on ctx extension), cached per yarn
/// config. Cold budget line item: 262144×64×4 B = 64 MB, ≤ 200 ms.
public final class QWRopeTables: @unchecked Sendable {
    public let rows: Int
    public let yarn: QWYaRNConfig?
    public let geometry: QWModelGeometry
    public let attentionScaling: Float
    /// [rows][width] f32: row = [cos(0..<pairs), sin(0..<pairs)]. Engine
    /// collapse of HF's emb=cat[freqs,freqs]: dim k's cos = entry k&(pairs-1),
    /// sin = entry pairs+(k&(pairs-1)).
    public let cosSin: [Float]
    /// Floats per row: the model's rotary dims (64 for the 27B, 128 for a
    /// standard family whose rotary covers the whole head).
    public let width: Int
    /// Distinct frequencies per row: half the rotary dims.
    public let pairs: Int

    /// HF compute_default_rope_parameters: inv_freq[i] = θ^(−2i/dim), dim=64,
    /// in Float (torch float32) — base ** (arange/dim) is the f32 tensor op.
    static func invFreqDefault(theta: Float = 1e7, rotaryDims: Int = QWModelGeometry.qwen38_27b.rotaryDims) -> [Float] {
        let dim = Float(rotaryDims)
        let pairs = rotaryDims / 2
        return (0..<pairs).map { 1.0 / pow(theta, Float($0) * 2.0 / dim) }   // f32 pow
    }

    /// HF _compute_yarn_parameters — verbatim order of operations. Ramp
    /// bounds in Double (Python floats); ramp/pairs in Float (f32 tensors).
    static func invFreqYarn(_ y: QWYaRNConfig, geometry: QWModelGeometry) -> (invf: [Float], scaling: Float) {
        let dimI = geometry.rotaryDims, dim = Float(dimI)
        let base: Float = Float(geometry.ropeTheta)
        let factor = y.factor
        let pairs = dimI / 2
        let posFreqs = (0..<pairs).map { pow(base, Float($0) * 2.0 / dim) }
        let invEx = posFreqs.map { 1.0 / $0 }
        let invIn = posFreqs.map { 1.0 / (factor * $0) }
        func fcd(_ nr: Double) -> Double {   // find_correction_dim (python float)
            let baseD = geometry.ropeTheta
            return (Double(dimI) * Foundation.log(Double(y.originalMaxPosition) / (nr * 2.0 * Double.pi)))
                 / (2.0 * Foundation.log(baseD))
        }
        var lo = fcd(Double(y.betaFast)), hi = fcd(Double(y.betaSlow))
        if y.truncate { lo = Foundation.floor(lo); hi = Foundation.ceil(hi) }
        let low = max(lo, 0.0)
        var high = min(hi, Double(dimI - 1))
        if low == high { high += 0.001 }      // HF: prevent singularity
        var out = [Float](repeating: 0, count: pairs)
        for i in 0..<pairs {
            let ramp = min(max((Float(i) - Float(low)) / Float(high - low), 0), 1)  // f32 ramp_func
            let eef = 1 - ramp
            out[i] = invIn[i] * (1 - eef) + invEx[i] * eef
        }
        // get_mscale(scale, mscale=1) = 0.1·ln(scale)+1 (attention_factor path)
        let scaling: Float = factor <= 1 ? 1.0 : Float(0.1 * log(Double(factor)) + 1.0)
        return (out, scaling)
    }

    public init(rows: Int, yarn: QWYaRNConfig?, geometry: QWModelGeometry) {
        self.rows = rows
        self.yarn = yarn
        self.geometry = geometry
        self.pairs = geometry.rotaryDims / 2
        self.width = geometry.rotaryDims
        let invf: [Float]
        if let y = yarn {
            let (f, s) = Self.invFreqYarn(y, geometry: geometry); invf = f; attentionScaling = s
        } else {
            invf = Self.invFreqDefault(theta: Float(geometry.ropeTheta), rotaryDims: geometry.rotaryDims)
            attentionScaling = 1.0
        }
        // Table fill is embarrassingly parallel (262 K rows × 64 libm calls);
        // single-threaded debug timing hit 739 ms — the ≤200 ms cold budget is
        // a REAL contract (spec 07), so it gets real concurrency (static fn:
        // concurrentPerform may not capture self pre-init).
        self.cosSin = Self.fill(rows: rows, invf: invf, scaling: attentionScaling)
    }

    /// Row for a scalar text position ([cos, sin] f32, width floats).
    public func cosSinRow(_ p: Int) -> [Float] {
        precondition((0..<rows).contains(p), "position \(p) beyond table rows \(rows)")
        return Array(cosSin[p * width ..< (p + 1) * width])
    }

    /// mrope (vision) row per HF recomposition_frequencies. HF loop:
    /// `for dim, offset in enumerate((1,2), start=1)` (H=1, W=2) replaces
    /// `freqs_thw[dim][idx]` at idx = offset, offset+3, … (residue ≡ dim mod 3,
    public func cosSinRowMrope(_ posThw: (t: Int, h: Int, w: Int)) -> [Float] {
        let invf = yarn.map { Self.invFreqYarn($0, geometry: geometry).invf }
            ?? Self.invFreqDefault(theta: Float(geometry.ropeTheta), rotaryDims: geometry.rotaryDims)
        let (t, h, w) = (Float(posThw.t), Float(posThw.h), Float(posThw.w))
        var row = [Float](repeating: 0, count: width)
        for i in 0..<pairs {
            let pos = i % 3 == 0 ? t : (i % 3 == 1 ? h : w)
            let f = pos * invf[i]
            row[i] = Foundation.cos(f) * attentionScaling
            row[pairs + i] = Foundation.sin(f) * attentionScaling
        }
        return row
    }

    /// Parallel table fill: [rows][2·pairs] f32 rows = [cos, sin].
    static func fill(rows: Int, invf: [Float], scaling: Float) -> [Float] {
        let pairs = invf.count
        let width = pairs * 2
        var cs = [Float](repeating: 0, count: rows * width)
        let chunks = max(1, ProcessInfo.processInfo.activeProcessorCount)
        let per = (rows + chunks - 1) / chunks
        // take the buffer pointer ONCE, outside concurrentPerform. Calling
        // withUnsafeMutableBufferPointer from N threads is a refcount race
        // (each entry uniques the COW storage; concurrent uniques = double
        struct PtrBox: @unchecked Sendable { let p: UnsafeMutableRawPointer }
        let box = cs.withUnsafeMutableBufferPointer { PtrBox(p: UnsafeMutableRawPointer($0.baseAddress!)) }
        DispatchQueue.concurrentPerform(iterations: chunks) { c in
            let lo = c * per, hi = min(rows, lo + per)
            guard lo < hi else { return }
            let buf = box.p.assumingMemoryBound(to: Float.self)
            for p in lo..<hi {
                let pos = Float(p)
                for i in 0..<pairs {
                    let f = pos * invf[i]
                    buf[p * width + i] = Foundation.cos(f) * scaling
                    buf[p * width + pairs + i] = Foundation.sin(f) * scaling
                }
            }
        }
        return cs
    }

}
