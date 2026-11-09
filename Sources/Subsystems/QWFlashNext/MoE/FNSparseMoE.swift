import Foundation

// Expert weight source for the sparse MoE. Real layers are too large to hold as fp32 arrays, so the MoE
// asks for one expert at a time (only the routed top-k are ever decoded).

public protocol FNExpertSource: Sendable {
    /// Gate/up projection of expert `e`: (2*intermediate) x hidden; rows [0,I)=gate, [I,2I)=up.
    func gateUp(_ e: Int) -> [Float]
    /// Down projection of expert `e`: hidden x intermediate.
    func down(_ e: Int) -> [Float]
}

/// In-memory experts (tests and tiny goldens): stacks laid out as [E, 2I, H] and [E, H, I].
public struct FNArrayExperts: FNExpertSource {
    let experts: Int
    let gateUpStack: [Float]
    let downStack: [Float]

    public init(experts: Int, gateUp: [Float], down: [Float]) {
        self.experts = experts; self.gateUpStack = gateUp; self.downStack = down
    }

    public func gateUp(_ e: Int) -> [Float] {
        let n = gateUpStack.count / experts
        return Array(gateUpStack[(e * n)..<((e + 1) * n)])
    }

    public func down(_ e: Int) -> [Float] {
        let n = downStack.count / experts
        return Array(downStack[(e * n)..<((e + 1) * n)])
    }
}

// FlashNext sparse MoE block (HF Qwen4ExpTextSparseMoeBlock): top-k router, SwiGLU experts,
// shared expert scaled by a sigmoid gate.
public struct FNSparseMoE: Sendable {
    public let hidden: Int
    public let experts: Int
    public let topK: Int
    public let intermediate: Int
    public let sharedIntermediate: Int
    public let normTopK: Bool
    public let gate: [Float]             // experts x hidden
    public let source: any FNExpertSource
    public let sharedGate: [Float]       // sharedIntermediate x hidden
    public let sharedUp: [Float]         // sharedIntermediate x hidden
    public let sharedDown: [Float]       // hidden x sharedIntermediate
    public let sharedGateScalar: [Float] // 1 x hidden

    public init(hidden: Int, experts: Int, topK: Int, intermediate: Int, sharedIntermediate: Int, normTopK: Bool,
                gate: [Float], source: any FNExpertSource, sharedGate: [Float], sharedUp: [Float],
                sharedDown: [Float], sharedGateScalar: [Float]) {
        self.hidden = hidden; self.experts = experts; self.topK = topK
        self.intermediate = intermediate; self.sharedIntermediate = sharedIntermediate; self.normTopK = normTopK
        self.gate = gate; self.source = source
        self.sharedGate = sharedGate; self.sharedUp = sharedUp; self.sharedDown = sharedDown
        self.sharedGateScalar = sharedGateScalar
    }

    /// Router: float softmax over experts, top-k (ties to the lower index), optional renormalisation.
    /// Returns the chosen experts in rank order (highest probability first) and their weights.
    public func route(_ xt: ArraySlice<Float>) -> (experts: [Int], weights: [Float]) {
        let logits = FNMath.linear(xt, weight: gate, out: experts, inDim: hidden)
        let maxLogit = logits.max() ?? 0
        var probs = logits.map { expf($0 - maxLogit) }
        let total = probs.reduce(0, +)
        probs = probs.map { $0 / total }
        let order = (0..<experts).sorted { probs[$0] > probs[$1] || (probs[$0] == probs[$1] && $0 < $1) }
        let chosen = Array(order.prefix(topK))
        var weights = chosen.map { probs[$0] }
        if normTopK {
            let s = weights.reduce(0, +)
            weights = weights.map { $0 / s }
        }
        return (chosen, weights)
    }

    public func forward(_ x: [Float], rows: Int) -> [Float] {
        var out = [Float](repeating: 0, count: rows * hidden)
        for r in 0..<rows {
            let xt = Array(x[(r * hidden)..<((r + 1) * hidden)])
            let (chosen, weights) = route(xt[...])
            var acc = [Float](repeating: 0, count: hidden)
            for (slot, e) in chosen.enumerated() {
                let gu = FNMath.linear(xt[...], weight: source.gateUp(e), out: 2 * intermediate, inDim: hidden)
                var act = [Float](repeating: 0, count: intermediate)
                for i in 0..<intermediate { act[i] = FNMath.silu(gu[i]) * gu[intermediate + i] }
                let y = FNMath.linear(act[...], weight: source.down(e), out: hidden, inDim: intermediate)
                for h in 0..<hidden { acc[h] += weights[slot] * y[h] }
            }
            let g = FNMath.linear(xt[...], weight: sharedGate, out: sharedIntermediate, inDim: hidden)
            let u = FNMath.linear(xt[...], weight: sharedUp, out: sharedIntermediate, inDim: hidden)
            var sh = [Float](repeating: 0, count: sharedIntermediate)
            for i in 0..<sharedIntermediate { sh[i] = FNMath.silu(g[i]) * u[i] }
            let shared = FNMath.linear(sh[...], weight: sharedDown, out: hidden, inDim: sharedIntermediate)
            let shGate = FNMath.sigmoid(FNMath.linear(xt[...], weight: sharedGateScalar, out: 1, inDim: hidden)[0])
            for h in 0..<hidden { out[r * hidden + h] = acc[h] + shGate * shared[h] }
        }
        return out
    }
}
