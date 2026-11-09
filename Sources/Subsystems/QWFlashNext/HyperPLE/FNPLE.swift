import Foundation

// FlashNext PLE layer (HF Qwen4ExpTextPLELayer): N-gram embedding lookup -> gated value -> causal dilated conv.

public struct FNPLE: Sendable {
    public let hidden: Int
    public let hc: Int
    public let embedDim: Int
    public let convKernel: Int
    public let dilation: Int
    public let eps: Float
    public let keyProj: [Float]     // (hc*hidden) x embedDim
    public let valueProj: [Float]   // hidden x embedDim
    public let normKey: [Float]     // hc*hidden
    public let normQuery: [Float]   // hc*hidden
    public let normConv: [Float]    // hc*hidden
    public let convWeight: [Float]  // (hc*hidden) x K  (index c*K + j)

    public init(hidden: Int, hc: Int, embedDim: Int, convKernel: Int, dilation: Int, eps: Float,
                keyProj: [Float], valueProj: [Float], normKey: [Float], normQuery: [Float], normConv: [Float],
                convWeight: [Float]) {
        self.hidden = hidden; self.hc = hc; self.embedDim = embedDim; self.convKernel = convKernel
        self.dilation = dilation; self.eps = eps; self.keyProj = keyProj; self.valueProj = valueProj
        self.normKey = normKey; self.normQuery = normQuery; self.normConv = normConv; self.convWeight = convWeight
    }

    /// `embedding`: rows x embedDim (N-gram lookup output for this sequence).
    /// `hiddenState`: rows x (hc*hidden). Returns rows x (hc*hidden).
    public func forward(embedding e: [Float], hiddenState h: [Float], rows T: Int) -> [Float] {
        let width = hc * hidden
        var kPre = [Float](repeating: 0, count: T * width)
        var vals = [Float](repeating: 0, count: T * hidden)
        for t in 0..<T {
            let et = e[(t * embedDim)..<((t + 1) * embedDim)]
            let kp = FNMath.linear(et, weight: keyProj, out: width, inDim: embedDim)
            let vp = FNMath.linear(et, weight: valueProj, out: hidden, inDim: embedDim)
            for c in 0..<width { kPre[t * width + c] = kp[c] }
            for c in 0..<hidden { vals[t * hidden + c] = vp[c] }
        }
        let keyNorm = FNMath.groupRMSNorm(kPre, rows: T, width: width, group: hidden, weight: normKey, eps: eps)
        let queryNorm = FNMath.groupRMSNorm(h, rows: T, width: width, group: hidden, weight: normQuery, eps: eps)
        // Gate per (token, stream): sign(g)*sqrt(max(|g|,1e-6)), g = <k,q>/sqrt(H).
        var gv = [Float](repeating: 0, count: T * width)
        let scale = 1 / Float(hidden).squareRoot()
        for t in 0..<T {
            for c in 0..<hc {
                var g: Float = 0
                for d in 0..<hidden { g += keyNorm[t * width + c * hidden + d] * queryNorm[t * width + c * hidden + d] }
                g *= scale
                let gate = (max(abs(g), 1e-6)).squareRoot() * (g > 0 ? 1 : (g < 0 ? -1 : 0))
                let s = FNMath.sigmoid(gate)
                for d in 0..<hidden { gv[t * width + c * hidden + d] = s * vals[t * hidden + d] }
            }
        }
        let gvNorm = FNMath.groupRMSNorm(gv, rows: T, width: width, group: hidden, weight: normConv, eps: eps)
        // Causal depthwise conv with dilation (left pad (K-1)*dilation), then residual + SiLU.
        var out = [Float](repeating: 0, count: T * width)
        for t in 0..<T {
            for c in 0..<width {
                var acc: Float = 0
                for j in 0..<convKernel {
                    let src = t - (convKernel - 1 - j) * dilation
                    if src >= 0 { acc += convWeight[c * convKernel + j] * gvNorm[src * width + c] }
                }
                out[t * width + c] = gv[t * width + c] + FNMath.silu(acc)
            }
        }
        return out
    }
}

/// Row source of the PLE N-gram table (`[totalRows, headDim]`). Rows are read on demand: the real table is
/// ~100 GB and never materialized. Reads are reference-only; an I/O failure is a fatal environment error.
public protocol FNNgramRows: Sendable {
    var headDim: Int { get }
    func row(_ id: Int) -> [Float]
    func copyRow(_ id: Int, to dst: UnsafeMutablePointer<Float>)
}

extension FNNgramRows {
    public func copyRow(_ id: Int, to dst: UnsafeMutablePointer<Float>) {
        let r = row(id)
        r.withUnsafeBufferPointer { src in
            dst.update(from: src.baseAddress!, count: headDim)
        }
    }
}

/// In-memory table (tiny goldens): row-major `[totalRows, headDim]`.
public struct FNArrayRows: FNNgramRows {
    public let headDim: Int
    let table: [Float]

    public init(table: [Float], headDim: Int) {
        self.table = table; self.headDim = headDim
    }

    public func row(_ id: Int) -> [Float] {
        Array(table[(id * headDim)..<((id + 1) * headDim)])
    }
}

/// PLE N-gram embedding: per token, the rows of every hashed head concatenated (HF Qwen4ExpTextNGramEmbedding).
public struct FNNgramEmbedding: Sendable {
    public let vocab: Int
    public let ngram: Int
    public let headsPerNgram: Int
    public let baseVocab: Int
    public let seed: Int
    public let eos: Int64
    public let pleLayerIndex: Int
    public let rows: any FNNgramRows
    public let multipliers: [Int64]
    public let heads: FNNgram.HeadTable

    public init(vocab: Int, ngram: Int, headsPerNgram: Int, baseVocab: Int, seed: Int, eos: Int64,
                pleLayerIndex: Int, rows: any FNNgramRows) {
        self.vocab = vocab; self.ngram = ngram; self.headsPerNgram = headsPerNgram; self.baseVocab = baseVocab
        self.seed = seed; self.eos = eos; self.pleLayerIndex = pleLayerIndex; self.rows = rows
        self.multipliers = FNNgram.layerMultipliers(vocab: vocab, ngram: ngram, pleLayerIndex: pleLayerIndex, seed: seed)
        self.heads = FNNgram.headTable(baseVocab: baseVocab, headCount: (ngram - 1) * headsPerNgram,
                                       pleLayerIndex: pleLayerIndex)
    }

    /// Embedding for one sequence: rows x (headCount*headDim).
    /// The rows of the last token only (decode): the same ids and rows as `embed`, without the earlier positions.
    public func embedLast(into base: UnsafeMutablePointer<Float>, _ tokens: [Int64]) {
        let lastIds = FNNgram.lastIds(tokens: tokens, ngram: ngram, headsPerNgram: headsPerNgram,
                                      eos: eos, multipliers: multipliers, heads: heads)
        let hd = rows.headDim, hc = lastIds.count
        let r = rows
        let addr = Int(bitPattern: base)
        DispatchQueue.concurrentPerform(iterations: hc) { h in
            let dst = UnsafeMutablePointer<Float>(bitPattern: addr + h * hd * 4)!
            r.copyRow(Int(lastIds[h]), to: dst)
        }
    }

    public func embedLast(_ tokens: [Int64]) -> [Float] {
        let hd = rows.headDim, hc = (ngram - 1) * headsPerNgram
        return [Float](unsafeUninitializedCapacity: hc * hd) { buf, count in
            embedLast(into: buf.baseAddress!, tokens)
            count = hc * hd
        }
    }

    @inlinable
    public func embedTokenHead(into base: UnsafeMutablePointer<Float>, tokens: [Int64], startPos: Int, t: Int, head: Int) {
        let hd = rows.headDim, hc = (ngram - 1) * headsPerNgram
        let hpn = headsPerNgram, eosTok = eos
        let mults = multipliers, hds = heads, r = rows
        let addr = Int(bitPattern: base)
        let idx = startPos + t
        let n = 2 + head / hpn
        var mixed = tokens[idx] &* mults[0]
        var hitEos = false
        for s in 1..<n {
            let srcIdx = idx - s
            let v: Int64
            if !hitEos && srcIdx >= 0 {
                let tok = tokens[srcIdx]
                if tok == eosTok { hitEos = true; v = eosTok } else { v = tok }
            } else {
                hitEos = true
                v = eosTok
            }
            mixed ^= v &* mults[s]
        }
        let size = Int64(hds.sizes[head])
        let offset = Int64(hds.offsets[head])
        var residue = mixed % size
        if residue < 0 { residue += size }
        let dst = UnsafeMutablePointer<Float>(bitPattern: addr + (t * hc + head) * hd * 4)!
        r.copyRow(Int(residue + offset), to: dst)
    }

    @inlinable
    public func embedToken(into base: UnsafeMutablePointer<Float>, tokens: [Int64], startPos: Int, t: Int) {
        let hd = rows.headDim, hc = (ngram - 1) * headsPerNgram
        let ng = ngram, hpn = headsPerNgram, eosTok = eos
        let mults = multipliers, hds = heads, r = rows
        let addr = Int(bitPattern: base)
        let idx = startPos + t
        if ng == 3 {
            let s0 = tokens[idx]
            var hitEos = false
            let s1: Int64
            if idx - 1 >= 0 {
                let v = tokens[idx - 1]
                if v == eosTok { hitEos = true; s1 = eosTok } else { s1 = v }
            } else { hitEos = true; s1 = eosTok }
            let s2: Int64
            if !hitEos && idx - 2 >= 0 {
                let v = tokens[idx - 2]
                if v == eosTok { s2 = eosTok } else { s2 = v }
            } else { s2 = eosTok }

            let m0 = s0 &* mults[0]
            let mix2 = m0 ^ (s1 &* mults[1])
            let mix3 = mix2 ^ (s2 &* mults[2])
            let mixes = (mix2, mix3)
            withUnsafeBytes(of: mixes) { raw in
                let mixPtr = raw.bindMemory(to: Int64.self)
                for g in 0..<2 {
                    let mixed = mixPtr[g]
                    let start = g * hpn
                    for head in start..<(start + hpn) {
                        let size = Int64(hds.sizes[head])
                        let offset = Int64(hds.offsets[head])
                        var residue = mixed % size
                        if residue < 0 { residue += size }
                        let dst = UnsafeMutablePointer<Float>(bitPattern: addr + (t * hc + head) * hd * 4)!
                        r.copyRow(Int(residue + offset), to: dst)
                    }
                }
            }
            return
        }
        if ng == 4 {
            let s0 = tokens[idx]
            var hitEos = false
            let s1: Int64
            if idx - 1 >= 0 {
                let v = tokens[idx - 1]
                if v == eosTok { hitEos = true; s1 = eosTok } else { s1 = v }
            } else { hitEos = true; s1 = eosTok }
            let s2: Int64
            if !hitEos && idx - 2 >= 0 {
                let v = tokens[idx - 2]
                if v == eosTok { hitEos = true; s2 = eosTok } else { s2 = v }
            } else { hitEos = true; s2 = eosTok }
            let s3: Int64
            if !hitEos && idx - 3 >= 0 {
                let v = tokens[idx - 3]
                if v == eosTok { s3 = eosTok } else { s3 = v }
            } else { s3 = eosTok }

            let m0 = s0 &* mults[0]
            let mix2 = m0 ^ (s1 &* mults[1])
            let mix3 = mix2 ^ (s2 &* mults[2])
            let mix4 = mix3 ^ (s3 &* mults[3])
            let mixes = (mix2, mix3, mix4)
            withUnsafeBytes(of: mixes) { raw in
                let mixPtr = raw.bindMemory(to: Int64.self)
                for g in 0..<3 {
                    let mixed = mixPtr[g]
                    let start = g * hpn
                    for head in start..<(start + hpn) {
                        let size = Int64(hds.sizes[head])
                        let offset = Int64(hds.offsets[head])
                        var residue = mixed % size
                        if residue < 0 { residue += size }
                        let dst = UnsafeMutablePointer<Float>(bitPattern: addr + (t * hc + head) * hd * 4)!
                        r.copyRow(Int(residue + offset), to: dst)
                    }
                }
            }
            return
        }
        withUnsafeTemporaryAllocation(of: Int64.self, capacity: ng) { shifted in
            shifted[0] = tokens[idx]
            var hitEos = false
            for s in 1..<ng {
                let srcIdx = idx - s
                if !hitEos && srcIdx >= 0 {
                    let v = tokens[srcIdx]
                    if v == eosTok {
                        hitEos = true
                        shifted[s] = eosTok
                    } else {
                        shifted[s] = v
                    }
                } else {
                    shifted[s] = eosTok
                }
            }
            for n in 2...ng {
                var mixed = shifted[0] &* mults[0]
                for pos in 1..<n {
                    mixed ^= shifted[pos] &* mults[pos]
                }
                let start = (n - 2) * hpn
                for head in start..<(start + hpn) {
                    let size = Int64(hds.sizes[head])
                    let offset = Int64(hds.offsets[head])
                    var residue = mixed % size
                    if residue < 0 { residue += size }
                    let dst = UnsafeMutablePointer<Float>(bitPattern: addr + (t * hc + head) * hd * 4)!
                    r.copyRow(Int(residue + offset), to: dst)
                }
            }
        }
    }

    public func embedRange(into base: UnsafeMutablePointer<Float>, tokens: [Int64], startPos: Int, count: Int) {
        let addr = Int(bitPattern: base)
        DispatchQueue.concurrentPerform(iterations: count) { t in
            let ptr = UnsafeMutablePointer<Float>(bitPattern: addr)!
            self.embedToken(into: ptr, tokens: tokens, startPos: startPos, t: t)
        }
    }

    public func embed(_ tokens: [Int64]) -> [Float] {
        let T = tokens.count, hc = (ngram - 1) * headsPerNgram, hd = rows.headDim
        return [Float](unsafeUninitializedCapacity: T * hc * hd) { buf, count in
            if let base = buf.baseAddress {
                embedRange(into: base, tokens: tokens, startPos: 0, count: T)
            }
            count = T * hc * hd
        }
    }
}
