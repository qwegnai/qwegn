import Foundation

// FlashNext PLE N-gram ids (HF Qwen4ExpTextNGramEmbedding). Exact 64-bit integer semantics:
// wrapping multiply/xor, Python-style modulo (result in [0, size)), EOS-segment right shift.

public enum FNNgram {
    static let gamma: UInt64 = 0x9E37_79B9_7F4A_7C15
    static let mix1: UInt64 = 0xBF58_476D_1CE4_E5B9
    static let mix2: UInt64 = 0x94D0_49BB_1331_11EB
    static let prime1: UInt64 = 10007

    public static func splitmix64(_ input: UInt64) -> UInt64 {
        var value = input &+ gamma
        value = (value ^ (value >> 30)) &* mix1
        value = (value ^ (value >> 27)) &* mix2
        return value ^ (value >> 31)
    }

    /// `_build_layer_multipliers`: odd multipliers, one per n-gram position.
    public static func layerMultipliers(vocab: Int, ngram: Int, pleLayerIndex: Int, seed: Int) -> [Int64] {
        let multiplierMax = Int64.max / Int64(max(vocab, 1))
        let half = UInt64(max(1, multiplierMax / 2))
        let baseSeed = UInt64(bitPattern: Int64(seed)) &+ prime1 &* UInt64(pleLayerIndex)
        return (0..<ngram).map { index in
            let value = baseSeed &+ gamma &* UInt64(index + 1)
            return Int64(2 * (splitmix64(value) % half) + 1)
        }
    }

    static func isPrime(_ value: Int) -> Bool {
        guard value >= 2 else { return false }
        if value % 2 == 0 { return value == 2 }
        var divisor = 3
        while divisor * divisor <= value {
            if value % divisor == 0 { return false }
            divisor += 2
        }
        return true
    }

    /// `_find_nth_prime_after`: the `count`-th prime strictly after `start`.
    public static func nthPrimeAfter(_ start: Int, count: Int) -> Int {
        var prime = start
        for _ in 0..<count {
            prime += 1
            while !isPrime(prime) { prime += 1 }
        }
        return prime
    }

    /// Per-head vocab sizes and offsets for one PLE layer.
    public struct HeadTable: Sendable, Equatable {
        public let sizes: [Int]
        public let offsets: [Int]
        public let totalRows: Int
    }

    public static func headTable(baseVocab: Int, headCount: Int, pleLayerIndex: Int) -> HeadTable {
        var sizes: [Int] = []
        var offsets: [Int] = []
        var total = 0
        var prime = baseVocab - 1
        for head in 0..<headCount {
            let step = head == 0 ? (pleLayerIndex * headCount + 1) : 1
            prime = nthPrimeAfter(prime, count: step)
            sizes.append(prime)
            offsets.append(total)
            total += prime
        }
        return HeadTable(sizes: sizes, offsets: offsets, totalRows: total)
    }

    /// `_shift_right_ignore_eos`: shifted-by-`shift` ids inside each EOS-delimited segment.
    public static func shiftRightIgnoreEOS(_ tokens: [Int64], shift: Int, eos: Int64) -> [Int64] {
        if shift == 0 { return tokens }
        let count = tokens.count
        var prevInclusive = [Int](repeating: -1, count: count)
        var run = -1
        for index in 0..<count {
            if tokens[index] == eos { run = index }
            prevInclusive[index] = run
        }
        var out = [Int64](repeating: eos, count: count)
        for index in 0..<count {
            let prevEos = index == 0 ? -1 : prevInclusive[index - 1]
            let positionInSegment = index - (prevEos + 1)
            let source = index - shift
            if positionInSegment >= shift && source >= 0 {
                out[index] = tokens[source]
            }
        }
        return out
    }

    /// Forward ids for a token block with an EOS-initialised context of `ngram - 1` tokens
    /// (fresh cache). Returns one row of `(ngram-1)*headsPerNgram` ids per input token.
    public static func ids(tokens: [Int64], vocab: Int, ngram: Int, headsPerNgram: Int, baseVocab: Int,
                           seed: Int, pleLayerIndex: Int, eos: Int64) -> [[Int64]] {
        let multipliers = layerMultipliers(vocab: vocab, ngram: ngram, pleLayerIndex: pleLayerIndex, seed: seed)
        let heads = headTable(baseVocab: baseVocab, headCount: (ngram - 1) * headsPerNgram,
                              pleLayerIndex: pleLayerIndex)
        return ids(tokens: tokens, ngram: ngram, headsPerNgram: headsPerNgram, eos: eos,
                   multipliers: multipliers, heads: heads)
    }

    public static func ids(tokens: [Int64], ngram: Int, headsPerNgram: Int, eos: Int64,
                           multipliers: [Int64], heads: HeadTable) -> [[Int64]] {
        let context = ngram - 1
        let history = [Int64](repeating: eos, count: context) + tokens
        let shifted = (0..<ngram).map { shiftRightIgnoreEOS(history, shift: $0, eos: eos) }
        var rows = [[Int64]](repeating: [], count: tokens.count)
        let headCount = (ngram - 1) * headsPerNgram
        for position in 0..<tokens.count { rows[position].reserveCapacity(headCount) }
        for n in 2...ngram {
            let start = (n - 2) * headsPerNgram
            for head in start..<(start + headsPerNgram) {
                let size = Int64(heads.sizes[head])
                let offset = Int64(heads.offsets[head])
                for position in 0..<tokens.count {
                    let h = context + position
                    var mixed = shifted[0][h] &* multipliers[0]
                    for pos in 1..<n {
                        mixed ^= shifted[pos][h] &* multipliers[pos]
                    }
                    var residue = mixed % size
                    if residue < 0 { residue += size }
                    rows[position].append(residue + offset)
                }
            }
        }
        return rows
    }

    /// Returns the `(ngram - 1) * headsPerNgram` row ids for the last token of `tokens` without allocating
    /// full sequence shift arrays.
    public static func lastIds(tokens: [Int64], ngram: Int, headsPerNgram: Int, eos: Int64,
                               multipliers: [Int64], heads: HeadTable) -> [Int64] {
        let nTok = tokens.count
        let headCount = (ngram - 1) * headsPerNgram
        return [Int64](unsafeUninitializedCapacity: headCount) { out, count in
            withUnsafeTemporaryAllocation(of: Int64.self, capacity: ngram) { shifted in
                shifted[0] = tokens[nTok - 1]
                var hitEos = false
                for s in 1..<ngram {
                    let srcIdx = nTok - 1 - s
                    if !hitEos && srcIdx >= 0 {
                        let v = tokens[srcIdx]
                        if v == eos {
                            hitEos = true
                            shifted[s] = eos
                        } else {
                            shifted[s] = v
                        }
                    } else {
                        shifted[s] = eos
                    }
                }
                for n in 2...ngram {
                    var mixed = shifted[0] &* multipliers[0]
                    for pos in 1..<n {
                        mixed ^= shifted[pos] &* multipliers[pos]
                    }
                    let start = (n - 2) * headsPerNgram
                    for head in start..<(start + headsPerNgram) {
                        let size = Int64(heads.sizes[head])
                        let offset = Int64(heads.offsets[head])
                        var residue = mixed % size
                        if residue < 0 { residue += size }
                        out[head] = residue + offset
                    }
                }
            }
            count = headCount
        }
    }
}
