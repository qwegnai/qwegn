import Foundation

// FlashNext full attention (HF Qwen4ExpTextAttention): q/k RMSNorm (1+w), partial RoPE, GQA,
// attention restricted to QSA-selected keys, sigmoid output gate applied before o_proj.

public struct FNAttention: Sendable {
    public let hidden: Int
    public let heads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let eps: Float
    public let qProj: [Float]     // (heads*2*headDim) x hidden ; per head [query | gate]
    public let kProj: [Float]     // (kvHeads*headDim) x hidden
    public let vProj: [Float]     // (kvHeads*headDim) x hidden
    public let oProj: [Float]     // hidden x (heads*headDim)
    public let qNorm: [Float]     // headDim
    public let kNorm: [Float]     // headDim
    public let indexer: FNQSAIndexer
    public let rope: FNRope

    public init(hidden: Int, heads: Int, kvHeads: Int, headDim: Int, eps: Float, qProj: [Float], kProj: [Float],
                vProj: [Float], oProj: [Float], qNorm: [Float], kNorm: [Float], indexer: FNQSAIndexer, rope: FNRope) {
        self.hidden = hidden; self.heads = heads; self.kvHeads = kvHeads; self.headDim = headDim; self.eps = eps
        self.qProj = qProj; self.kProj = kProj; self.vProj = vProj; self.oProj = oProj
        self.qNorm = qNorm; self.kNorm = kNorm; self.indexer = indexer; self.rope = rope
    }

    /// Projections, per-head RMS norms, partial RoPE, and the QSA selection.
    /// q and gate are T x heads x headDim; k and v are T x kvHeads x headDim; selected[t][j] for keys j <= t.
    public struct Prepared: Sendable {
        public let q: [Float], k: [Float], v: [Float], gate: [Float]
        public let selected: [[Bool]]
    }

    public func prepare(_ x: [Float], rows T: Int) -> Prepared {
        let hd = headDim, nh = heads, nkv = kvHeads
        var qRaw = [Float](repeating: 0, count: T * nh * hd)
        var gate = qRaw
        for t in 0..<T {
            let qg = FNMath.linear(x[(t * hidden)..<((t + 1) * hidden)], weight: qProj, out: nh * 2 * hd, inDim: hidden)
            for h in 0..<nh {
                for d in 0..<hd {
                    qRaw[(t * nh + h) * hd + d] = qg[h * 2 * hd + d]
                    gate[(t * nh + h) * hd + d] = qg[h * 2 * hd + hd + d]
                }
            }
        }
        var kRaw = [Float](repeating: 0, count: T * nkv * hd)
        var v = kRaw
        for t in 0..<T {
            let xt = x[(t * hidden)..<((t + 1) * hidden)]
            let k = FNMath.linear(xt, weight: kProj, out: nkv * hd, inDim: hidden)
            let vv = FNMath.linear(xt, weight: vProj, out: nkv * hd, inDim: hidden)
            for c in 0..<(nkv * hd) { kRaw[t * nkv * hd + c] = k[c]; v[t * nkv * hd + c] = vv[c] }
        }
        var q = FNMath.groupRMSNorm(qRaw, rows: T * nh, width: hd, group: hd, weight: qNorm, eps: eps)
        var k = FNMath.groupRMSNorm(kRaw, rows: T * nkv, width: hd, group: hd, weight: kNorm, eps: eps)
        let (cos, sin) = rope.tables(positions: Array(0..<T))
        for t in 0..<T {
            for h in 0..<nh { rope.apply(&q, base: (t * nh + h) * hd, dim: hd, cos: cos, sin: sin, row: t) }
            for g in 0..<nkv { rope.apply(&k, base: (t * nkv + g) * hd, dim: hd, cos: cos, sin: sin, row: t) }
        }
        let selected = indexer.select(x, rows: T, hidden: hidden, rope: rope)
        return Prepared(q: q, k: k, v: v, gate: gate, selected: selected)
    }

    /// Softmax attention over the selected keys (causal, GQA), then the sigmoid output gate. T x heads x headDim.
    public func attend(_ p: Prepared, rows T: Int) -> [Float] {
        let hd = headDim, nh = heads, nkv = kvHeads
        let q = p.q, k = p.k, v = p.v, selected = p.selected
        let scale = 1 / Float(hd).squareRoot()
        let group = nh / nkv
        var o = [Float](repeating: 0, count: T * nh * hd)
        for t in 0..<T {
            for h in 0..<nh {
                let g = h / group
                var scores: [Float] = []
                var keys: [Int] = []
                for j in 0...t where selected[t][j] {
                    var dot: Float = 0
                    for d in 0..<hd { dot += q[(t * nh + h) * hd + d] * k[(j * nkv + g) * hd + d] }
                    scores.append(dot * scale)
                    keys.append(j)
                }
                let m = scores.max() ?? 0
                var denom: Float = 0
                let weights = scores.map { s -> Float in let e = expf(s - m); denom += e; return e }
                for (idx, j) in keys.enumerated() {
                    let pw = weights[idx] / denom
                    for d in 0..<hd { o[(t * nh + h) * hd + d] += pw * v[(j * nkv + g) * hd + d] }
                }
            }
        }
        for i in 0..<o.count { o[i] *= FNMath.sigmoid(p.gate[i]) }
        return o
    }

    public func forward(_ x: [Float], rows T: Int) -> [Float] {
        let nh = heads, hd = headDim
        let o = attend(prepare(x, rows: T), rows: T)
        var out = [Float](repeating: 0, count: T * hidden)
        for t in 0..<T {
            let y = FNMath.linear(o[(t * nh * hd)..<((t + 1) * nh * hd)], weight: oProj, out: hidden, inDim: nh * hd)
            for c in 0..<hidden { out[t * hidden + c] = y[c] }
        }
        return out
    }
}
