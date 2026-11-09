import Foundation

// FlashNext hyper-connection gated residual (HF Qwen4ExpTextGatedResidual, combine=True).

public struct FNHyperConnection: Sendable {
    public let hc: Int
    public let hidden: Int
    public let lowrank: Int
    public let eps: Float
    public let normWeight: [Float]      // hc*hidden
    public let downWeight: [Float]      // lowrank x (hc*hidden)
    public let upWeight: [Float]        // (hc*hidden) x lowrank
    public let injectWeight: [Float]    // hc x (hc*hidden)

    public init(hc: Int, hidden: Int, lowrank: Int, eps: Float, normWeight: [Float], downWeight: [Float],
                upWeight: [Float], injectWeight: [Float]) {
        self.hc = hc; self.hidden = hidden; self.lowrank = lowrank; self.eps = eps
        self.normWeight = normWeight; self.downWeight = downWeight
        self.upWeight = upWeight; self.injectWeight = injectWeight
    }

    /// x: rows x (hc*hidden). Returns the mixed sub-layer input (rows x hidden) and the injection
    /// weights (rows x hc; empty when `injectWeight` is empty, i.e. the model's final mixer).
    public func combine(_ x: [Float], rows: Int) -> (mixed: [Float], injection: [Float]) {
        let width = hc * hidden
        let xn = FNMath.groupRMSNorm(x, rows: rows, width: width, group: hidden, weight: normWeight, eps: eps)
        var mixed = [Float](repeating: 0, count: rows * hidden)
        var injection = [Float](repeating: 0, count: rows * hc)
        let scale = Float(hc)
        for r in 0..<rows {
            let row = xn[(r * width)..<((r + 1) * width)]
            var low = FNMath.linear(row, weight: downWeight, out: lowrank, inDim: width)
            for i in 0..<lowrank { low[i] = FNMath.silu(low[i] / scale) }
            var gate = FNMath.linear(low[...], weight: upWeight, out: width, inDim: lowrank)
            for i in 0..<width { gate[i] = FNMath.sigmoid(gate[i]) }
            for h in 0..<hidden {
                var acc: Float = 0
                for c in 0..<hc { acc += gate[c * hidden + h] * row[row.startIndex + c * hidden + h] }
                mixed[r * hidden + h] = acc / scale
            }
            if !injectWeight.isEmpty {
                let inj = FNMath.linear(row, weight: injectWeight, out: hc, inDim: width)
                for c in 0..<hc { injection[r * hc + c] = 2 * FNMath.sigmoid(inj[c] / scale) }
            }
        }
        return (mixed, injection)
    }
}
