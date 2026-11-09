import Foundation

// FlashNext GatedDeltaNet (HF Qwen4ExpTextGatedDeltaNet + torch_recurrent_gated_delta_rule).
// Output norm activation is config.output_gate_type (sigmoid for FlashNext), plain weight (no +1).

public struct FNGatedDeltaNet: Sendable {
    public let hidden: Int
    public let kHeads: Int
    public let vHeads: Int
    public let kDim: Int
    public let vDim: Int
    public let convKernel: Int
    public let eps: Float
    public let inProjQKV: [Float]   // convDim x hidden
    public let inProjZ: [Float]     // valueDim x hidden
    public let inProjB: [Float]     // vHeads x hidden
    public let inProjA: [Float]     // vHeads x hidden
    public let convWeight: [Float]  // convDim x K  (index c*K + j)
    public let dtBias: [Float]      // vHeads
    public let aLog: [Float]        // vHeads
    public let normWeight: [Float]  // vDim
    public let outProj: [Float]     // hidden x valueDim

    public init(hidden: Int, kHeads: Int, vHeads: Int, kDim: Int, vDim: Int, convKernel: Int, eps: Float,
                inProjQKV: [Float], inProjZ: [Float], inProjB: [Float], inProjA: [Float], convWeight: [Float],
                dtBias: [Float], aLog: [Float], normWeight: [Float], outProj: [Float]) {
        self.hidden = hidden; self.kHeads = kHeads; self.vHeads = vHeads; self.kDim = kDim; self.vDim = vDim
        self.convKernel = convKernel; self.eps = eps
        self.inProjQKV = inProjQKV; self.inProjZ = inProjZ; self.inProjB = inProjB; self.inProjA = inProjA
        self.convWeight = convWeight; self.dtBias = dtBias; self.aLog = aLog
        self.normWeight = normWeight; self.outProj = outProj
    }

    var keyDim: Int { kHeads * kDim }
    var valueDim: Int { vHeads * vDim }
    var convDim: Int { 2 * keyDim + valueDim }

    /// Per-token recurrence inputs (post-conv, normalised, gates applied). Layouts: q,k are
    /// T x vHeads x kDim (q pre-scaled by kDim^-0.5); v is T x vHeads x vDim; z is T x valueDim;
    /// glog (log decay) and beta are T x vHeads.
    public struct Inputs: Sendable {
        public let q: [Float], k: [Float], v: [Float], z: [Float], glog: [Float], beta: [Float]
    }

    /// Projections, causal conv + SiLU, q/k normalisation, and the per-head gates.
    public func inputs(_ x: [Float], rows T: Int) -> Inputs {
        let convDim = self.convDim, keyDim = self.keyDim, valueDim = self.valueDim
        var mixed = [Float](repeating: 0, count: T * convDim)
        var z = [Float](repeating: 0, count: T * valueDim)
        var b = [Float](repeating: 0, count: T * vHeads)
        var a = [Float](repeating: 0, count: T * vHeads)
        for t in 0..<T {
            let xt = x[(t * hidden)..<((t + 1) * hidden)]
            let m = FNMath.linear(xt, weight: inProjQKV, out: convDim, inDim: hidden)
            let zz = FNMath.linear(xt, weight: inProjZ, out: valueDim, inDim: hidden)
            let bb = FNMath.linear(xt, weight: inProjB, out: vHeads, inDim: hidden)
            let aa = FNMath.linear(xt, weight: inProjA, out: vHeads, inDim: hidden)
            for c in 0..<convDim { mixed[t * convDim + c] = m[c] }
            for c in 0..<valueDim { z[t * valueDim + c] = zz[c] }
            for h in 0..<vHeads { b[t * vHeads + h] = bb[h]; a[t * vHeads + h] = aa[h] }
        }
        // Causal depthwise conv (left pad K-1) + SiLU.
        var conv = [Float](repeating: 0, count: T * convDim)
        for t in 0..<T {
            for c in 0..<convDim {
                var acc: Float = 0
                for j in 0..<convKernel {
                    let src = t - (convKernel - 1) + j
                    if src >= 0 { acc += convWeight[c * convKernel + j] * mixed[src * convDim + c] }
                }
                conv[t * convDim + c] = FNMath.silu(acc)
            }
        }
        // Split q/k/v; Q,K are l2-normalised per key head, Q is scaled by 1/sqrt(kDim).
        let rep = vHeads / kHeads
        let qScale = 1 / Float(kDim).squareRoot()
        var q = [Float](repeating: 0, count: T * vHeads * kDim)
        var k = q
        var v = [Float](repeating: 0, count: T * vHeads * vDim)
        for t in 0..<T {
            for hk in 0..<kHeads {
                var qn: Float = 0, kn: Float = 0
                for d in 0..<kDim {
                    let qv = conv[t * convDim + hk * kDim + d]
                    let kv = conv[t * convDim + keyDim + hk * kDim + d]
                    qn += qv * qv; kn += kv * kv
                }
                let qInv = 1 / (qn + 1e-6).squareRoot(), kInv = 1 / (kn + 1e-6).squareRoot()
                for r in 0..<rep {
                    let hv = hk * rep + r
                    for d in 0..<kDim {
                        q[(t * vHeads + hv) * kDim + d] = conv[t * convDim + hk * kDim + d] * qInv * qScale
                        k[(t * vHeads + hv) * kDim + d] = conv[t * convDim + keyDim + hk * kDim + d] * kInv
                    }
                }
            }
            for hv in 0..<vHeads {
                for d in 0..<vDim { v[(t * vHeads + hv) * vDim + d] = conv[t * convDim + 2 * keyDim + hv * vDim + d] }
            }
        }
        var glog = [Float](repeating: 0, count: T * vHeads)
        var beta = [Float](repeating: 0, count: T * vHeads)
        for t in 0..<T {
            for hv in 0..<vHeads {
                glog[t * vHeads + hv] = -expf(aLog[hv]) * FNGatedDeltaNet.softplus(a[t * vHeads + hv] + dtBias[hv])
                beta[t * vHeads + hv] = FNMath.sigmoid(b[t * vHeads + hv])
            }
        }
        return Inputs(q: q, k: k, v: v, z: z, glog: glog, beta: beta)
    }

    /// Delta-rule recurrence per value head (state S is kDim x vDim, fp32), then the gated RMSNorm
    /// (plain weight, gate = sigmoid(z)). Returns T x valueDim.
    public func gated(_ p: Inputs, rows T: Int) -> [Float] {
        let vHeads = self.vHeads, kDim = self.kDim, vDim = self.vDim, valueDim = self.valueDim
        var o = [Float](repeating: 0, count: T * valueDim)
        for hv in 0..<vHeads {
            var state = [Float](repeating: 0, count: kDim * vDim)
            for t in 0..<T {
                let decay = expf(p.glog[t * vHeads + hv])
                let beta = p.beta[t * vHeads + hv]
                let base = (t * vHeads + hv)
                for i in 0..<(kDim * vDim) { state[i] *= decay }
                var delta = [Float](repeating: 0, count: vDim)
                for j in 0..<vDim {
                    var kvMem: Float = 0
                    for i in 0..<kDim { kvMem += state[i * vDim + j] * p.k[base * kDim + i] }
                    delta[j] = (p.v[base * vDim + j] - kvMem) * beta
                }
                for i in 0..<kDim {
                    for j in 0..<vDim { state[i * vDim + j] += p.k[base * kDim + i] * delta[j] }
                }
                for j in 0..<vDim {
                    var acc: Float = 0
                    for i in 0..<kDim { acc += state[i * vDim + j] * p.q[base * kDim + i] }
                    o[t * valueDim + hv * vDim + j] = acc
                }
            }
        }
        var gated = [Float](repeating: 0, count: T * valueDim)
        for t in 0..<T {
            for hv in 0..<vHeads {
                var ss: Float = 0
                for j in 0..<vDim { let val = o[t * valueDim + hv * vDim + j]; ss += val * val }
                let inv = 1 / (ss / Float(vDim) + eps).squareRoot()
                for j in 0..<vDim {
                    let idx = t * valueDim + hv * vDim + j
                    gated[idx] = o[idx] * inv * normWeight[j] * FNMath.sigmoid(p.z[idx])
                }
            }
        }
        return gated
    }

    public func forward(_ x: [Float], rows T: Int) -> [Float] {
        let gated = gated(inputs(x, rows: T), rows: T)
        var out = [Float](repeating: 0, count: T * hidden)
        for t in 0..<T {
            let y = FNMath.linear(gated[(t * valueDim)..<((t + 1) * valueDim)], weight: outProj,
                                  out: hidden, inDim: valueDim)
            for h in 0..<hidden { out[t * hidden + h] = y[h] }
        }
        return out
    }

    /// torch.nn.functional.softplus with beta=1, threshold=20.
    static func softplus(_ x: Float) -> Float { x > 20 ? x : log1pf(expf(x)) }
}
