import Foundation

// FlashNext QSA indexer (HF Qwen4ExpTextQSAIndexer). Per query token, keeps the top `budget/ratio`
// complete key blocks by relu-scored, mean-pooled block keys, plus the incomplete tail.

public struct FNQSAIndexer: Sendable {
    public let heads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let budget: Int
    public let compress: Int
    public let eps: Float
    public let qkProj: [Float]     // (heads + kvHeads)*headDim x hidden
    public let qNorm: [Float]      // headDim
    public let kNorm: [Float]      // headDim

    public init(heads: Int, kvHeads: Int, headDim: Int, budget: Int, compress: Int, eps: Float,
                qkProj: [Float], qNorm: [Float], kNorm: [Float]) {
        self.heads = heads; self.kvHeads = kvHeads; self.headDim = headDim
        self.budget = budget; self.compress = compress; self.eps = eps
        self.qkProj = qkProj; self.qNorm = qNorm; self.kNorm = kNorm
    }

    /// Returns `selected[t][j]` for a single sequence of `rows` tokens (x: rows x hidden).
    /// j > t is never selected (causality is applied by the caller as well).
    public func select(_ x: [Float], rows T: Int, hidden: Int, rope: FNRope) -> [[Bool]] {
        let qkWidth = (heads + kvHeads) * headDim
        var q = [Float](repeating: 0, count: T * heads * headDim)
        var rawK = [Float](repeating: 0, count: T * headDim)
        for t in 0..<T {
            let qk = FNMath.linear(x[(t * hidden)..<((t + 1) * hidden)], weight: qkProj, out: qkWidth, inDim: hidden)
            for h in 0..<heads { for d in 0..<headDim { q[(t * heads + h) * headDim + d] = qk[h * headDim + d] } }
            for d in 0..<headDim { rawK[t * headDim + d] = qk[heads * headDim + d] }
        }
        q = FNMath.groupRMSNorm(q, rows: T * heads, width: headDim, group: headDim, weight: qNorm, eps: eps)
        let positions = Array(0..<T)
        let (cos, sin) = rope.tables(positions: positions)
        for t in 0..<T {
            for h in 0..<heads { rope.apply(&q, base: (t * heads + h) * headDim, dim: headDim, cos: cos, sin: sin, row: t) }
        }
        // Pooled block keys depend only on the keys: mean over `compress` raw keys, RMSNorm, RoPE at block start.
        let blocks = T / compress
        var pooled = [Float](repeating: 0, count: blocks * headDim)
        for b in 0..<blocks {
            for d in 0..<headDim {
                var acc: Float = 0
                for k in 0..<compress { acc += rawK[(b * compress + k) * headDim + d] }
                pooled[b * headDim + d] = acc / Float(compress)
            }
        }
        pooled = FNMath.groupRMSNorm(pooled, rows: blocks, width: headDim, group: headDim, weight: kNorm, eps: eps)
        let (bc, bs) = rope.tables(positions: (0..<blocks).map { $0 * compress })
        for b in 0..<blocks { rope.apply(&pooled, base: b * headDim, dim: headDim, cos: bc, sin: bs, row: b) }

        let blockTopK = budget / compress
        var selected = [[Bool]](repeating: [Bool](repeating: false, count: T), count: T)
        for t in 0..<T {
            let complete = (t + 1) / compress
            var scored: [(Int, Float)] = []
            for b in 0..<complete {
                var score: Float = 0
                for h in 0..<heads {
                    var dot: Float = 0
                    for d in 0..<headDim { dot += q[(t * heads + h) * headDim + d] * pooled[b * headDim + d] }
                    score += max(dot, 0)
                }
                scored.append((b, score / Float(headDim).squareRoot()))
            }
            // MLX/oMLX contract: last k of a stable ascending sort, so equal scores prefer the higher index.
            scored.sort { $0.1 > $1.1 || ($0.1 == $1.1 && $0.0 > $1.0) }
            for (b, _) in scored.prefix(min(blockTopK, complete)) {
                for k in 0..<compress { selected[t][b * compress + k] = true }
            }
            if complete * compress <= t {
                for j in (complete * compress)...t { selected[t][j] = true }
            }
        }
        return selected
    }
}
