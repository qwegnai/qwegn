import Foundation

// FlashNext decoder layer and text model (HF Qwen4ExpTextDecoderLayer / Qwen4ExpTextModel / Qwen4ExpForCausalLM).
// Stream layout: rows x (hc*hidden). Layers: optional PLE residual -> HC -> mixer (GDN | full attention)
// -> inject -> HC -> sparse MoE -> inject.

public enum FNMixer: Sendable {
    case linear(FNGatedDeltaNet)
    case full(FNAttention)
}

public struct FNDecoderLayer: Sendable {
    public let mixer: FNMixer
    public let attnHC: FNHyperConnection
    public let mlpHC: FNHyperConnection
    public let moe: FNSparseMoE
    public let ple: FNPLE?
    public let pleEmbedding: FNNgramEmbedding?

    public init(mixer: FNMixer, attnHC: FNHyperConnection, mlpHC: FNHyperConnection, moe: FNSparseMoE,
                ple: FNPLE?, pleEmbedding: FNNgramEmbedding?) {
        self.mixer = mixer; self.attnHC = attnHC; self.mlpHC = mlpHC; self.moe = moe
        self.ple = ple; self.pleEmbedding = pleEmbedding
    }

    /// stream: rows x (hc*hidden); tokens: the layer's PLE ids (= input ids for a fresh sequence).
    public func forward(_ stream: [Float], tokens: [Int64], rows T: Int) -> [Float] {
        let hc = attnHC.hc, hidden = attnHC.hidden, width = hc * hidden
        var h = stream
        if let ple, let embedding = pleEmbedding {
            let p = ple.forward(embedding: embedding.embed(tokens), hiddenState: h, rows: T)
            for i in 0..<h.count { h[i] += p[i] }
        }
        // Sub-layer 1: HC mix -> mixer -> inject into the residual stream.
        let (mixed, inj1) = attnHC.combine(h, rows: T)
        let o = mixerForward(mixed, rows: T)
        var h2 = h
        for t in 0..<T {
            for c in 0..<hc {
                for d in 0..<hidden { h2[t * width + c * hidden + d] += o[t * hidden + d] * inj1[t * hc + c] }
            }
        }
        // Sub-layer 2: HC mix -> sparse MoE -> inject.
        let (mixed2, inj2) = mlpHC.combine(h2, rows: T)
        let m = moe.forward(mixed2, rows: T)
        var out = h2
        for t in 0..<T {
            for c in 0..<hc {
                for d in 0..<hidden { out[t * width + c * hidden + d] += m[t * hidden + d] * inj2[t * hc + c] }
            }
        }
        return out
    }

    func mixerForward(_ x: [Float], rows T: Int) -> [Float] {
        switch mixer {
        case .linear(let gdn): return gdn.forward(x, rows: T)
        case .full(let attn): return attn.forward(x, rows: T)
        }
    }
}

public struct FNModel: Sendable {
    public let vocab: Int
    public let hidden: Int
    public let hc: Int
    public let embedding: [Float]      // vocab x hidden
    public let layers: [FNDecoderLayer]
    public let finalMixer: FNHyperConnection   // no injection weights
    public let lmHead: [Float]         // vocab x hidden

    public init(vocab: Int, hidden: Int, hc: Int, embedding: [Float], layers: [FNDecoderLayer],
                finalMixer: FNHyperConnection, lmHead: [Float]) {
        self.vocab = vocab; self.hidden = hidden; self.hc = hc; self.embedding = embedding
        self.layers = layers; self.finalMixer = finalMixer; self.lmHead = lmHead
    }

    /// One sequence. Returns logits (rows x vocab) and the stream after each decoder layer (rows x hc*hidden).
    public func forwardTrace(_ tokens: [Int64]) -> (logits: [Float], streams: [[Float]]) {
        let T = tokens.count
        var stream = [Float](repeating: 0, count: T * hc * hidden)
        for t in 0..<T {
            let id = Int(tokens[t])
            for c in 0..<hc {
                for d in 0..<hidden { stream[t * hc * hidden + c * hidden + d] = embedding[id * hidden + d] }
            }
        }
        var streams: [[Float]] = []
        for layer in layers {
            stream = layer.forward(stream, tokens: tokens, rows: T)
            streams.append(stream)
        }
        let (mixed, _) = finalMixer.combine(stream, rows: T)
        var logits = [Float](repeating: 0, count: T * vocab)
        for t in 0..<T {
            let y = FNMath.linear(mixed[(t * hidden)..<((t + 1) * hidden)], weight: lmHead, out: vocab, inDim: hidden)
            for v in 0..<vocab { logits[t * vocab + v] = y[v] }
        }
        return (logits, streams)
    }
}
