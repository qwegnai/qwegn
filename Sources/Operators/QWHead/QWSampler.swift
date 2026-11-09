// QWSampler.swift — C15 sampler: the distribution path (T>0), CPU side.
//
// WHY CPU (the spec's race; DECISIONS): Apple Silicon UMA maps
// storageModeShared CPU-side (QWMRegion.pointer), so the logits hand-off
// is a memory read, not a PCIe trip — but ONLY with the exp pass on
// vecLib vexpf: a scalar fp64 exp over 61440 logits measured 3.0 ms
// (bench_head v1, attribution probe) — 12× the ≤250 µs gate. vexpf is the
import Foundation
import Accelerate
import QWCore

public struct QWSamplerConfig: Sendable, Hashable {
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var seed: UInt64
    public var uniformVersion: QWPhilox.UniformVersion
    public var repetitionPenalty: Float
    public init(temperature: Float = 0.7, topK: Int = 64, topP: Float = 0.9,
                seed: UInt64 = 0, uniformVersion: QWPhilox.UniformVersion = .legacyRounded32,
                repetitionPenalty: Float = 1.0) {
        self.temperature = temperature; self.topK = topK
        self.topP = topP; self.seed = seed
        self.uniformVersion = uniformVersion
        self.repetitionPenalty = repetitionPenalty
    }

    /// True when the configuration asks for deterministic greedy argmax decoding.
    /// The repo's canonical rule, the same one `QWInferenceParameters.isGreedy`
    /// uses: never greedy on one path and stochastic on another.
    public var isGreedy: Bool {
        temperature <= 0 || topK == 1
    }

    /// Deterministic argmax decoding.
    public static let greedy = QWSamplerConfig(temperature: 0, topK: 1, topP: 1,
                                               seed: 0, repetitionPenalty: 1.0)

    /// Builds the sampler's contract from the request's sampling parameters.
    /// It mirrors the dense path's own normalisation exactly, so a request
    /// means the same thing on either family.
    public init(_ parameters: QWSamplingParameters, vocabulary: Int,
                uniformVersion: QWPhilox.UniformVersion = .upper24) {
        let greedyWithPenalty = parameters.isGreedyWithRepetitionPenalty
        self.init(temperature: greedyWithPenalty ? 1 : Float(parameters.temperature),
                  topK: greedyWithPenalty ? 1 : (parameters.topK > 0 ? parameters.topK : vocabulary),
                  topP: greedyWithPenalty ? 1 : Float(parameters.topP),
                  seed: parameters.resolvedSeed(),
                  uniformVersion: uniformVersion,
                  repetitionPenalty: Float(parameters.repetitionPenalty))
    }
}

public struct QWSampleResult: Sendable {
    public let token: Int
    public let logprob: Float
    /// The renormalized nucleus (ids ascending, probabilities sum to 1) —
    /// C16's verify path consumes the distribution directly.
    public let survivorIds: [Int]
    public let survivorProbs: [Float]
}

public enum QWSamplerError: Error, CustomStringConvertible {
    case invalid(String)
    public var description: String {
        switch self { case .invalid(let reason): return "QWSampler: \(reason)" }
    }
}

/// Immutable transformed support, ascending by token ID. Construction owns no
/// draws; speculative acceptance can retain the exact Double probabilities.
public struct QWSamplingDistribution: Sendable {
    public let vocabulary: Int
    public let survivorIds: [Int]
    public let survivorProbs: [Double]
    private let weights: [Double]
    private let mass: Double
    private let fullMass: Double

    fileprivate init(vocabulary: Int, ids: [Int], weights: [Double], mass: Double, fullMass: Double) {
        self.vocabulary = vocabulary
        survivorIds = ids
        survivorProbs = weights.map { $0 / mass }
        self.weights = weights
        self.mass = mass
        self.fullMass = fullMass
    }

    public static func sparse(vocabulary: Int, ids: [Int], weights: [Double]) throws -> QWSamplingDistribution {
        guard ids.count == weights.count, !ids.isEmpty, vocabulary > 0 else {
            throw QWSamplerError.invalid("sparse distribution ids and weights count mismatch")
        }
        var merged: [Int: Double] = [:]
        for (id, w) in zip(ids, weights) {
            guard (0..<vocabulary).contains(id), w.isFinite, w > 0 else {
                throw QWSamplerError.invalid("sparse distribution token ID or weight invalid")
            }
            merged[id, default: 0] += w
        }
        let sorted = merged.sorted { $0.key < $1.key }
        let sortedIds = sorted.map(\.key)
        let sortedWeights = sorted.map(\.value)
        let mass = sortedWeights.reduce(0, +)
        guard mass.isFinite, mass > 0 else {
            throw QWSamplerError.invalid("sparse distribution mass invalid")
        }
        return QWSamplingDistribution(vocabulary: vocabulary, ids: sortedIds,
                                      weights: sortedWeights, mass: mass, fullMass: mass)
    }

    public func sample(uniform: Double,
                       version: QWPhilox.UniformVersion = .upper24) throws -> QWSampleResult {
        let selected = try selectedIndex(uniform: uniform, version: version)
        return QWSampleResult(token: survivorIds[selected],
            logprob: Float(Foundation.log(weights[selected]) - Foundation.log(fullMass)),
            survivorIds: survivorIds, survivorProbs: survivorProbs.map(Float.init))
    }

    public func sampleToken(uniform: Double,
                            version: QWPhilox.UniformVersion = .upper24) throws -> Int {
        survivorIds[try selectedIndex(uniform: uniform, version: version)]
    }

    private func selectedIndex(uniform: Double, version: QWPhilox.UniformVersion) throws -> Int {
        guard uniform.isFinite, uniform >= 0,
              version == .legacyRounded32 ? uniform <= 1 : uniform < 1 else {
            throw QWSamplerError.invalid("uniform outside versioned interval")
        }
        var selected = weights.count - 1
        var cumulative = 0.0
        for index in weights.indices {
            cumulative += version == .legacyRounded32 ? weights[index] : survivorProbs[index]
            let boundary = version == .legacyRounded32 ? uniform * mass : uniform
            if version == .legacyRounded32 ? cumulative >= boundary : cumulative > boundary {
                selected = index
                break
            }
        }
        return selected
    }
}

public final class QWSampler {
    public var config: QWSamplerConfig
    private var callIndex: UInt64 = 0
    // Reused scratch (per-token call; allocator churn is C09 steady-state
    // law territory): e = exp((v − dmax)/T) fp32.
    private var eScratch: [Float] = []
    private var shiftedScratch: [Float] = []
    private var penaltyScratch: [Float] = []
    private var heapScratch: [Int] = []
    private var topKScratch: [Int] = []
    private var seenEpoch: [UInt32] = []
    private var currentEpoch: UInt32 = 0

    public init(config: QWSamplerConfig) { self.config = config }
    public func resetCallSequence() { callIndex = 0 }
    public var drawsConsumed: UInt64 { callIndex }

    @inline(__always)
    private static func allFiniteSIMD(_ buf: UnsafeBufferPointer<Float>) -> Bool {
        let n = buf.count
        guard n > 0, let base = buf.baseAddress else { return true }
        var maxMag: Float = 0
        var rawSum: Float = 0
        vDSP_maxmgv(base, 1, &maxMag, vDSP_Length(n))
        vDSP_sve(base, 1, &rawSum, vDSP_Length(n))
        if maxMag < 1e30 && rawSum.isFinite { return true }
        return buf.allSatisfy(\.isFinite)
    }

    @inline(__always)
    private static func allFiniteNonNegSIMD(_ buf: UnsafeBufferPointer<Float>, sum: Float) -> Bool {
        let n = buf.count
        guard n > 0, let base = buf.baseAddress else { return true }
        var minVal: Float = 0
        var maxMag: Float = 0
        vDSP_minv(base, 1, &minVal, vDSP_Length(n))
        vDSP_maxmgv(base, 1, &maxMag, vDSP_Length(n))
        if minVal >= 0 && maxMag < 1e30 && sum.isFinite { return true }
        return buf.allSatisfy { $0.isFinite && $0 >= 0 }
    }

    public static func adjustedLogits(_ logits: [Float], repetitionHistory: [Int],
                                      repetitionPenalty: Float) throws -> [Float] {
        guard repetitionPenalty.isFinite, repetitionPenalty > 0,
              logits.allSatisfy(\.isFinite) else {
            throw QWSamplerError.invalid("invalid repetition penalty or logit row")
        }
        guard repetitionHistory.allSatisfy({ (0..<logits.count).contains($0) }) else {
            throw QWSamplerError.invalid("repetition history contains token outside the logit vocabulary")
        }
        var adjusted = logits
        for token in Set(repetitionHistory) {
            let value = adjusted[token]
            adjusted[token] = value < 0 ? value * repetitionPenalty : value / repetitionPenalty
        }
        guard adjusted.allSatisfy(\.isFinite) else {
            throw QWSamplerError.invalid("repetition penalty produced non-finite logits")
        }
        return adjusted
    }

    /// Sample one token from one LM-head logit row. O(N) SIMD exp pass +
    /// O(N log K) bounded selection; only the K survivors are sorted.
    public func sample(logits: [Float], repetitionHistory: [Int] = []) throws -> QWSampleResult {
        guard callIndex < UInt64.max else { throw QWSamplerError.invalid("draw counter exhausted") }
        let distribution = try distribution(logits: logits, repetitionHistory: repetitionHistory)
        let uniform = QWPhilox.u(callIndex, seed: config.seed, version: config.uniformVersion)
        let result = try distribution.sample(uniform: Double(uniform), version: config.uniformVersion)
        callIndex += 1
        return result
    }

    /// The shared entry point: one logit row in, one token id out. The draw
    /// counter lives in the sampler, so a family that owns its own sampler
    /// gets the same sequence as any other family with the same seed.
    public func sampleToken(logits: [Float], repetitionHistory: [Int] = []) throws -> Int {
        try sample(logits: logits, repetitionHistory: repetitionHistory).token
    }

    /// The same entry point over a borrowed row: the caller keeps the buffer
    /// and the sampler copies nothing.
    public func sampleToken(logits: UnsafeBufferPointer<Float>,
                             repetitionHistory: [Int] = []) throws -> Int {
        try sample(logits: logits, repetitionHistory: repetitionHistory).token
    }

    /// The greedy token of a borrowed logit row, with the repetition penalty
    /// applied. The penalised row lives in the sampler's own scratch, so the
    /// caller does not allocate one per token.
    public func penalisedGreedyToken(logits: UnsafeBufferPointer<Float>,
                                     repetitionHistory: Set<Int>) throws -> Int {
        let n = logits.count
        guard n > 0 else { throw QWSamplerError.invalid("empty logit row") }
        guard config.repetitionPenalty.isFinite, config.repetitionPenalty > 0 else {
            throw QWSamplerError.invalid("invalid repetition penalty")
        }
        guard logits.allSatisfy(\.isFinite) else {
            throw QWSamplerError.invalid("logit row contains a non-finite value")
        }
        guard repetitionHistory.allSatisfy({ (0..<n).contains($0) }) else {
            throw QWSamplerError.invalid("repetition history contains token outside the logit vocabulary")
        }
        if penaltyScratch.count != n { penaltyScratch = [Float](repeating: 0, count: n) }
        penaltyScratch.withUnsafeMutableBufferPointer { dst in
            dst.baseAddress!.update(from: logits.baseAddress!, count: n)
            for token in repetitionHistory {
                let value = dst[token]
                dst[token] = value < 0 ? value * config.repetitionPenalty
                                       : value / config.repetitionPenalty
            }
        }
        return try QWSampler.greedyToken(logits: penaltyScratch)
    }

    /// The same over an array.
    public func penalisedGreedyToken(logits: [Float],
                                     repetitionHistory: Set<Int>) throws -> Int {
        try logits.withUnsafeBufferPointer {
            try penalisedGreedyToken(logits: $0, repetitionHistory: repetitionHistory)
        }
    }

    /// Samples one token from a borrowed logit row. No copy: the row stays
    /// where the caller put it.
    public func sample(logits: UnsafeBufferPointer<Float>,
                        repetitionHistory: [Int] = []) throws -> QWSampleResult {
        guard callIndex < UInt64.max else { throw QWSamplerError.invalid("draw counter exhausted") }
        let distribution = try distribution(logits: logits, repetitionHistory: repetitionHistory)
        let uniform = QWPhilox.u(callIndex, seed: config.seed, version: config.uniformVersion)
        let result = try distribution.sample(uniform: Double(uniform), version: config.uniformVersion)
        callIndex += 1
        return result
    }

    /// Selects one token from `allowed` only: greedy takes the largest penalised logit
    /// without a draw; sampled configs apply temperature/top-k/top-p over `allowed` and
    /// consume one draw. Disallowed logits are never read.
    public func constrainedToken(logits: UnsafeBufferPointer<Float>, allowed: [Int],
                                 repetitionHistory: Set<Int> = []) throws -> Int {
        let n = logits.count
        guard n > 0 else { throw QWSamplerError.invalid("empty logit row") }
        guard !allowed.isEmpty else { throw QWTokenConstraintError.emptySupport }
        guard config.repetitionPenalty.isFinite, config.repetitionPenalty > 0 else {
            throw QWSamplerError.invalid("invalid repetition penalty")
        }
        var values = [Float](repeating: 0, count: allowed.count)
        var previous = -1
        for (index, token) in allowed.enumerated() {
            guard token > previous, token < n else {
                throw QWSamplerError.invalid("constrained support must be ascending token ids inside the vocabulary")
            }
            previous = token
            var value = logits[token]
            if config.repetitionPenalty != 1, repetitionHistory.contains(token) {
                value = value < 0 ? value * config.repetitionPenalty : value / config.repetitionPenalty
            }
            guard value.isFinite else {
                throw QWSamplerError.invalid("constrained logit is non-finite")
            }
            values[index] = value
        }
        if config.isGreedy {
            var best = 0
            for index in 1..<values.count where values[index] > values[best] { best = index }
            return allowed[best]
        }
        let temperature = config.temperature
        guard temperature.isFinite, temperature > 0, config.topK > 0,
              config.topP.isFinite, config.topP > 0, config.topP <= 1 else {
            throw QWSamplerError.invalid("invalid temperature, top-k or top-p")
        }
        guard callIndex < UInt64.max else { throw QWSamplerError.invalid("draw counter exhausted") }
        let maxValue = values.max()!
        let invT = 1 / Double(temperature)
        var weights = [Double](repeating: 0, count: values.count)
        var fullMass = 0.0
        for index in values.indices {
            let w = Foundation.exp(Double(values[index] - maxValue) * invT)
            weights[index] = w
            fullMass += w
        }
        guard fullMass.isFinite, fullMass > 0 else {
            throw QWSamplerError.invalid("invalid constrained softmax mass")
        }
        let order = Self.topIndices(weights, count: min(config.topK, weights.count))
        let topP = Double(config.topP)
        var keep: [Int] = []
        var cumulative = 0.0
        for index in order {
            if cumulative >= topP { break }
            keep.append(index)
            cumulative += weights[index] / fullMass
        }
        keep.sort()
        let ids = keep.map { allowed[$0] }
        let kept = keep.map { weights[$0] }
        let mass = kept.reduce(0, +)
        guard mass.isFinite, mass > 0 else {
            throw QWSamplerError.invalid("invalid constrained survivor mass")
        }
        let distribution = QWSamplingDistribution(vocabulary: n, ids: ids, weights: kept,
                                                  mass: mass, fullMass: fullMass)
        let uniform = QWPhilox.u(callIndex, seed: config.seed, version: config.uniformVersion)
        let token = try distribution.sampleToken(uniform: Double(uniform), version: config.uniformVersion)
        callIndex += 1
        return token
    }

    /// Indices of the `count` largest weights, descending, ties to the lower index.
    private static func topIndices(_ weights: [Double], count: Int) -> [Int] {
        let before: (Int, Int) -> Bool = { a, b in
            weights[a] != weights[b] ? weights[a] > weights[b] : a < b
        }
        if count >= weights.count {
            return weights.indices.sorted(by: before)
        }
        // Min-heap (by `before`) of the current best `count` indices.
        var heap: [Int] = []
        heap.reserveCapacity(count)
        func siftDown(_ start: Int) {
            var p = start
            while true {
                let l = 2 * p + 1, r = l + 1
                var m = p
                if l < heap.count && before(heap[m], heap[l]) { m = l }
                if r < heap.count && before(heap[m], heap[r]) { m = r }
                if m == p { return }
                heap.swapAt(p, m); p = m
            }
        }
        for index in weights.indices {
            if heap.count < count {
                heap.append(index)
                var c = heap.count - 1
                while c > 0 {
                    let p = (c - 1) >> 1
                    if before(heap[p], heap[c]) { heap.swapAt(c, p); c = p } else { break }
                }
            } else if before(index, heap[0]) {
                heap[0] = index
                siftDown(0)
            }
        }
        return heap.sorted(by: before)
    }

    /// The greedy token of a logit row: the largest logit, ties broken by the
    /// lowest id. A repetition penalty shifts the row first.
    public static func greedyToken(logits: [Float], repetitionHistory: [Int] = [],
                                   repetitionPenalty: Float = 1.0) throws -> Int {
        let row = repetitionPenalty == 1 ? logits
            : try adjustedLogits(logits, repetitionHistory: repetitionHistory,
                                 repetitionPenalty: repetitionPenalty)
        var best = 0
        for index in 1..<row.count where row[index] > row[best] { best = index }
        return best
    }

    public func distribution(logits: [Float],
                             repetitionHistory: [Int] = []) throws -> QWSamplingDistribution {
        try logits.withUnsafeBufferPointer {
            try distribution(logits: $0, repetitionHistory: repetitionHistory)
        }
    }

    /// Reads the row synchronously; the returned distribution owns its support.
    public func distribution(logits: UnsafeBufferPointer<Float>,
                             repetitionHistory: [Int] = []) throws -> QWSamplingDistribution {
        let T = config.temperature, P = Double(config.topP)
        let n = logits.count
        guard T.isFinite, T > 0, (1 / T).isFinite, config.topK > 0,
              config.topP.isFinite, config.topP > 0, config.topP <= 1,
              config.repetitionPenalty.isFinite, config.repetitionPenalty > 0,
              n > 0, n <= Int(Int32.max), Self.allFiniteSIMD(logits) else {
            throw QWSamplerError.invalid("invalid temperature, top-k/top-p or logit row")
        }
        let K = min(config.topK, n)
        if config.repetitionPenalty != 1, !repetitionHistory.isEmpty {
            guard config.repetitionPenalty.isFinite, config.repetitionPenalty > 0 else {
                throw QWSamplerError.invalid("invalid repetition penalty")
            }
            if penaltyScratch.count != n { penaltyScratch = [Float](repeating: 0, count: n) }
            if seenEpoch.count != n {
                seenEpoch = [UInt32](repeating: 0, count: n)
                currentEpoch = 0
            }
            currentEpoch &+= 1
            if currentEpoch == 0 {
                for idx in 0..<n { seenEpoch[idx] = 0 }
                currentEpoch = 1
            }
            let epoch = currentEpoch
            let rep = config.repetitionPenalty
            // The penalty is applied in place: the row is copied once, into
            // the scratch the sampler already owns.
            try penaltyScratch.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: logits.baseAddress!, count: n)
                try seenEpoch.withUnsafeMutableBufferPointer { seen in
                    for token in repetitionHistory {
                        guard token >= 0 && token < n else {
                            throw QWSamplerError.invalid("repetition history contains token outside the logit vocabulary")
                        }
                        if seen[token] != epoch {
                            seen[token] = epoch
                            let value = dst[token]
                            let nextVal = value < 0 ? value * rep : value / rep
                            guard nextVal.isFinite else {
                                throw QWSamplerError.invalid("repetition penalty produced non-finite logits")
                            }
                            dst[token] = nextVal
                        }
                    }
                }
            }
            return try penaltyScratch.withUnsafeBufferPointer {
                try makeDistribution(logits: $0, vocabulary: n, topK: K, topP: P,
                                     temperature: T)
            }
        }
        return try makeDistribution(logits: logits, vocabulary: n, topK: K, topP: P,
                                    temperature: T)
    }

    private func makeDistribution(logits: UnsafeBufferPointer<Float>, vocabulary n: Int,
                                  topK K: Int, topP P: Double,
                                  temperature: Float) throws -> QWSamplingDistribution {
        if eScratch.count != n { eScratch = [Float](repeating: 0, count: n) }
        if shiftedScratch.count != n { shiftedScratch = [Float](repeating: 0, count: n) }
        var dmax: Float = 0
        var sum: Float = 0
        return try eScratch.withUnsafeMutableBufferPointer { ep in
            try shiftedScratch.withUnsafeMutableBufferPointer { sp in
                vDSP_maxv(logits.baseAddress!, 1, &dmax, vDSP_Length(n))
                // shifted = logits − dmax  (vsub: dst = src − scalar)
                var negDmax = -dmax
                vDSP_vsadd(logits.baseAddress!, 1, &negDmax, sp.baseAddress!, 1, vDSP_Length(n))
                // shifted = shifted · (1/T) + 0  (vsmsa: src·scalar + bias)
                var invT = 1 / temperature, bias: Float = 0
                vDSP_vsmsa(sp.baseAddress!, 1, &invT, &bias,
                           sp.baseAddress!, 1, vDSP_Length(n))
                var nu = Int32(n)
                // vForce SIMD exp (NOT vexpf — the Swift overlay shadows
                // it with a scalar variant and the C array form is not
                // imported; measured). Y = e^X.
                vvexpf(ep.baseAddress!, sp.baseAddress!, &nu)   // fp32 SIMD exp
                vDSP_sve(ep.baseAddress!, 1, &sum, vDSP_Length(n))
                return try transformedSupport(e: UnsafeBufferPointer(ep), topK: K, topP: P, sum: sum)
            }
        }
    }

    private func transformedSupport(e: UnsafeBufferPointer<Float>, topK K: Int,
                                    topP P: Double, sum: Float) throws -> QWSamplingDistribution {
        guard sum.isFinite, sum > 0, Self.allFiniteNonNegSIMD(e, sum: sum) else {
            throw QWSamplerError.invalid("invalid softmax mass")
        }
        // ── top-k 64: BOUNDED MIN-HEAP of the k LARGEST (O(N log k),
        //    reused scratch, e never mutated). v1/v2 median-of-medians cost
        //    2977 µs of swaps+pivot allocations (sampler_profile: exp pass
        //    33 µs, select 2977 — the select WAS the sampler). v3 first
        //    cut had the comparison BACKWARDS (e[i] < root keeps the k
        //    SMALLEST → keepK held ~N ids → the desc sort cost 5 ms and
        //    the peaked fixture's tests still passed because the nucleus
        //    cumsum breaks long before the kth boundary matters when the
        //    top-k mass ≥ p). The contract only binds on a FLAT fixture
        //    (top-64 mass < p): torch restricts candidates to the k — a
        //    superset select would draw from the whole tail. Law: test
        //    the boundary the code implements, not just the headline path.
        heapScratch.removeAll(keepingCapacity: true)
        heapScratch.reserveCapacity(K)
        var maxEvicted: Float = -1.0
        let count = e.count
        let initK = min(K, count)
        for i in 0..<initK {
            heapScratch.append(i)
            var c = heapScratch.count - 1
            while c > 0 {
                let p = (c - 1) >> 1
                if e[heapScratch[c]] < e[heapScratch[p]] { heapScratch.swapAt(c, p); c = p } else { break }
            }
        }
        if initK < count, let base = e.baseAddress {
            var i = initK
            var rootVal = e[heapScratch[0]]
            while i + 16 <= count {
                let v = UnsafeRawPointer(base.advanced(by: i)).loadUnaligned(as: SIMD16<Float>.self)
                if v.max() > rootVal {
                    for j in i..<(i + 16) where e[j] > rootVal {
                        if rootVal > maxEvicted { maxEvicted = rootVal }
                        heapScratch[0] = j
                        var p = 0
                        while true {
                            let l = 2 * p + 1, r = l + 1
                            var m = p
                            if l < K && e[heapScratch[l]] < e[heapScratch[m]] { m = l }
                            if r < K && e[heapScratch[r]] < e[heapScratch[m]] { m = r }
                            if m == p { break }
                            heapScratch.swapAt(p, m); p = m
                        }
                        rootVal = e[heapScratch[0]]
                    }
                }
                i += 16
            }
            while i < count {
                if e[i] > rootVal {
                    if rootVal > maxEvicted { maxEvicted = rootVal }
                    heapScratch[0] = i
                    var p = 0
                    while true {
                        let l = 2 * p + 1, r = l + 1
                        var m = p
                        if l < K && e[heapScratch[l]] < e[heapScratch[m]] { m = l }
                        if r < K && e[heapScratch[r]] < e[heapScratch[m]] { m = r }
                        if m == p { break }
                        heapScratch.swapAt(p, m); p = m
                    }
                    rootVal = e[heapScratch[0]]
                }
                i += 1
            }
        }
        // heap now holds the K LARGEST ids (ties included by FIRST-seen
        // ascending-id order: strict < comparisons never evict an equal —
        // exactly torch's tie direction, and it is the stable law the
        let kth = e[heapScratch[0]]
        topKScratch.removeAll(keepingCapacity: true)
        topKScratch.reserveCapacity(K)
        if maxEvicted < kth {
            topKScratch.append(contentsOf: heapScratch)
        } else {
            for i in e.indices {
                if e[i] > kth { topKScratch.append(i) }
            }
            if topKScratch.count < K {
                for i in e.indices where e[i] == kth {
                    topKScratch.append(i)
                    if topKScratch.count == K { break }
                }
            }
        }
        // order keepK DESC by e (ties ascending id) for the nucleus walk
        topKScratch.sort { a, b in let ea = e[a], eb = e[b]; return ea != eb ? ea > eb : a < b }
        // ── top-p nucleus over keepK (≤ K + ties entries); the cumsum
        //    walks NORMALIZED probs — torch thresholds against Σp = 1 ──
        let sumD = Double(sum)
        var cum = 0.0
        var need = 1
        for (r, i) in topKScratch.enumerated() {
            let before = cum
            cum += Double(e[i]) / sumD
            if before < P { need = r + 1 } else { break }
        }
        let qth = e[topKScratch[need - 1]]
        var cntGtQ = 0
        for i in topKScratch where e[i] > qth { cntGtQ += 1 }
        var keep: [Int] = []
        var sl2 = need - cntGtQ
        for i in topKScratch {                        // desc walk, ties asc id
            if e[i] > qth { keep.append(i) }
            else if e[i] == qth && sl2 > 0 { keep.append(i); sl2 -= 1 }
        }
        keep.sort()                                   // ascending id (renorm)
        var massD = 0.0
        for i in keep { massD += Double(e[i]) }
        guard massD.isFinite, massD > 0, !keep.isEmpty else {
            throw QWSamplerError.invalid("invalid survivor mass")
        }
        return QWSamplingDistribution(vocabulary: e.count, ids: keep, weights: keep.map { Double(e[$0]) },
                                      mass: massD, fullMass: sumD)
    }

    /// Computes the full-vocabulary log-probability (`logit - logsumexp`) of `chosenToken`
    /// and optionally the top `topK` (`0..20`) highest-probability candidate tokens.
    public func computeTokenLogprob(logits: UnsafeBufferPointer<Float>,
                                    chosenToken: Int,
                                    topK: Int = 0) -> QWTokenLogprob {
        let n = logits.count
        guard n > 0, chosenToken >= 0, chosenToken < n, let base = logits.baseAddress else {
            return QWTokenLogprob(tokenId: chosenToken, logprob: 0)
        }
        if eScratch.count != n { eScratch = [Float](repeating: 0, count: n) }
        if shiftedScratch.count != n { shiftedScratch = [Float](repeating: 0, count: n) }
        var dmax: Float = 0
        var sum: Float = 0
        eScratch.withUnsafeMutableBufferPointer { ep in
            shiftedScratch.withUnsafeMutableBufferPointer { sp in
                vDSP_maxv(base, 1, &dmax, vDSP_Length(n))
                var negDmax = -dmax
                vDSP_vsadd(base, 1, &negDmax, sp.baseAddress!, 1, vDSP_Length(n))
                var nu = Int32(n)
                vvexpf(ep.baseAddress!, sp.baseAddress!, &nu)
                vDSP_sve(ep.baseAddress!, 1, &sum, vDSP_Length(n))
            }
        }
        let logZ = dmax + Foundation.log(max(sum, Float.leastNormalMagnitude))
        let chosenLP = logits[chosenToken] - logZ
        let k = min(max(0, topK), min(20, n))
        guard k > 0 else {
            return QWTokenLogprob(tokenId: chosenToken, logprob: chosenLP)
        }
        heapScratch.removeAll(keepingCapacity: true)
        heapScratch.reserveCapacity(k)
        for i in 0..<k {
            heapScratch.append(i)
            var c = heapScratch.count - 1
            while c > 0 {
                let p = (c - 1) >> 1
                if logits[heapScratch[c]] < logits[heapScratch[p]] {
                    heapScratch.swapAt(c, p)
                    c = p
                } else {
                    break
                }
            }
        }
        var rootVal = logits[heapScratch[0]]
        for i in k..<n where logits[i] > rootVal {
            heapScratch[0] = i
            var p = 0
            while true {
                let l = 2 * p + 1, r = l + 1
                var m = p
                if l < k && logits[heapScratch[l]] < logits[heapScratch[m]] { m = l }
                if r < k && logits[heapScratch[r]] < logits[heapScratch[m]] { m = r }
                if m == p { break }
                heapScratch.swapAt(p, m)
                p = m
            }
            rootVal = logits[heapScratch[0]]
        }
        let sortedTop = heapScratch.sorted { a, b in
            let la = logits[a], lb = logits[b]
            return la != lb ? la > lb : a < b
        }
        let candidates = sortedTop.map { idx in
            QWTokenLogprob.Candidate(tokenId: idx, logprob: logits[idx] - logZ)
        }
        return QWTokenLogprob(tokenId: chosenToken, logprob: chosenLP, topCandidates: candidates)
    }
}

// (the median-of-medians helpers that lived here are GONE — the bounded
//  min-heap above replaced them; see the select comment for the numbers.)
