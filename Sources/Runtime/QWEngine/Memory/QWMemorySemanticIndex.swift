// QWMemorySemanticIndex.swift — Sub-millisecond e-means semantic centroid index
// for Off-Context QWMemoryVault conceptual recall (arXiv:2611.09501 H4b).
// Preserves 100% Resident Cortex prefix invariance and manifest compatibility.

import Accelerate
import Foundation
import QWCore

public struct QWMemorySemanticIndex: Codable, Sendable, Equatable {
    public static let embeddingDim: Int = 128

    public let dim: Int
    public let clusterCount: Int
    /// Relative paths of indexed files (`count == N`).
    public let filePaths: [String]
    /// Flattened L2-normalized file signatures `[N * dim]` stored as `Float16` bitPatterns for compact JSON/RAM.
    public let fileVectorsF16: [UInt16]
    /// Flattened L2-normalized `e-means` `last-epoch-mean` centroids `[clusterCount * dim]` (`empty` when `N < 16`).
    public let centroidsF16: [UInt16]
    /// CSR inverted lists mapping each `e-means` centroid `0..<clusterCount` to indices in `filePaths`.
    public let invertedLists: [[Int]]
    private let fileVectorsF32: [Float]
    private let centroidsF32: [Float]

    private enum CodingKeys: String, CodingKey {
        case dim, clusterCount, filePaths, fileVectorsF16, centroidsF16, invertedLists
    }

    public static func == (lhs: QWMemorySemanticIndex, rhs: QWMemorySemanticIndex) -> Bool {
        lhs.dim == rhs.dim
            && lhs.clusterCount == rhs.clusterCount
            && lhs.filePaths == rhs.filePaths
            && lhs.fileVectorsF16 == rhs.fileVectorsF16
            && lhs.centroidsF16 == rhs.centroidsF16
            && lhs.invertedLists == rhs.invertedLists
    }

    public init(
        dim: Int = Self.embeddingDim,
        clusterCount: Int,
        filePaths: [String],
        fileVectorsF16: [UInt16],
        centroidsF16: [UInt16],
        invertedLists: [[Int]]
    ) {
        self.dim = dim
        self.clusterCount = clusterCount
        self.filePaths = filePaths
        self.fileVectorsF16 = fileVectorsF16
        self.centroidsF16 = centroidsF16
        self.invertedLists = invertedLists
        self.fileVectorsF32 = fileVectorsF16.map { Float(Float16(bitPattern: $0)) }
        self.centroidsF32 = centroidsF16.map { Float(Float16(bitPattern: $0)) }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let dim = try c.decode(Int.self, forKey: .dim)
        let clusterCount = try c.decode(Int.self, forKey: .clusterCount)
        let filePaths = try c.decode([String].self, forKey: .filePaths)
        let fileVectorsF16 = try c.decode([UInt16].self, forKey: .fileVectorsF16)
        let centroidsF16 = try c.decode([UInt16].self, forKey: .centroidsF16)
        let invertedLists = try c.decode([[Int]].self, forKey: .invertedLists)
        self.init(
            dim: dim,
            clusterCount: clusterCount,
            filePaths: filePaths,
            fileVectorsF16: fileVectorsF16,
            centroidsF16: centroidsF16,
            invertedLists: invertedLists
        )
    }

    /// Builds an IDF-weighted `e-means` semantic centroid index over `entries`.
    /// Enforces `n / k >= 8`: uses `k = min(64, entries.count / 8)` when `entries.count >= 16`, else flat exact scan.
    public static func build(
        entries: [QWMemoryManifest.FileEntry],
        rootPath: String? = nil,
        epochs: Double = 2.0,
        seed: UInt64 = 20261110
    ) -> QWMemorySemanticIndex {
        let d = embeddingDim
        guard !entries.isEmpty else {
            return QWMemorySemanticIndex(
                dim: d, clusterCount: 0, filePaths: [], fileVectorsF16: [], centroidsF16: [], invertedLists: []
            )
        }
        let rootURL = rootPath.map { URL(fileURLWithPath: $0, isDirectory: true) }
        let n = entries.count
        var paths: [String] = []
        paths.reserveCapacity(n)
        var docTermWeights: [[String: Float]] = []
        docTermWeights.reserveCapacity(n)
        var docFreq: [String: Int] = [:]

        for entry in entries {
            paths.append(entry.relativePath)
            let anchorText = entry.relativePath + " " + entry.symbols.joined(separator: " ")
            var bodyCode = ""
            if let rootURL {
                let url = rootURL.appendingPathComponent(entry.relativePath, isDirectory: false)
                if let body = try? String(contentsOf: url, encoding: .utf8), !body.isEmpty {
                    var kept: [Substring] = []
                    for line in body.prefix(6144).split(whereSeparator: \.isNewline) {
                        let t = line.drop(while: { $0 == " " || $0 == "\t" })
                        if t.hasPrefix("//") || t.hasPrefix("/*") || t.hasPrefix("*") || t.hasPrefix("--") {
                            continue
                        }
                        kept.append(t)
                    }
                    bodyCode = kept.joined(separator: " ")
                }
            }
            var weights = extractWeightedTerms(anchorText, scale: 3.0)
            let bodyWeights = extractWeightedTerms(bodyCode, scale: 1.0)
            for (term, w) in bodyWeights {
                weights[term, default: 0] += w
            }
            for term in weights.keys {
                docFreq[term, default: 0] += 1
            }
            docTermWeights.append(weights)
        }

        var vectorsF32 = [Float](repeating: 0, count: n * d)
        let nFloat = Float(n)
        for (idx, weights) in docTermWeights.enumerated() {
            var vec = [Float](repeating: 0, count: d)
            for (term, rawTF) in weights {
                let df = Float(docFreq[term] ?? 1)
                let idf = log((nFloat + 1.0) / (df + 0.5)) + 0.05
                let w = (1.0 + log(max(1.0, rawTF))) * idf
                projectTerm(term, weight: w, into: &vec)
            }
            normalizeUnit(&vec)
            for c in 0..<d {
                vectorsF32[idx * d + c] = vec[c]
            }
        }

        let vecsF16 = vectorsF32.map { Float16($0).bitPattern }
        guard n >= 16 else {
            return QWMemorySemanticIndex(
                dim: d,
                clusterCount: 0,
                filePaths: paths,
                fileVectorsF16: vecsF16,
                centroidsF16: [],
                invertedLists: []
            )
        }

        let k = min(64, max(2, n / 8))
        let (centroidsF32, lists) = fitSphericalEmeansLastEpoch(
            vectors: vectorsF32, count: n, dim: d, k: k, epochs: epochs, seed: seed
        )
        return QWMemorySemanticIndex(
            dim: d,
            clusterCount: k,
            filePaths: paths,
            fileVectorsF16: vecsF16,
            centroidsF16: centroidsF32.map { Float16($0).bitPattern },
            invertedLists: lists
        )
    }

    /// Queries the semantic index in `< 0.5 ms`, probing top `e-means` centroids and returning `(path, cosine)` hits.
    public func query(
        _ queryText: String,
        excludedPaths: Set<String> = [],
        topK: Int = 4,
        minCosine: Float = 0.20
    ) -> [(relativePath: String, cosine: Float)] {
        let n = filePaths.count
        guard n > 0, topK > 0, dim == Self.embeddingDim else { return [] }
        let q = Self.embedText(queryText)
        var qNormSq: Float = 0
        vDSP_dotpr(q, 1, q, 1, &qNormSq, vDSP_Length(dim))
        guard qNormSq > 1e-6 else { return [] }

        let candidateIndices: [Int]
        if clusterCount >= 2 && centroidsF32.count == clusterCount * dim {
            let probeCount = min(clusterCount, max(2, (clusterCount + 1) / 2))
            var cScores: [(idx: Int, sim: Float)] = []
            cScores.reserveCapacity(clusterCount)
            q.withUnsafeBufferPointer { qBuf in
                centroidsF32.withUnsafeBufferPointer { cBuf in
                    let qPtr = qBuf.baseAddress!
                    let cPtr = cBuf.baseAddress!
                    for c in 0..<clusterCount {
                        var dot: Float = 0
                        vDSP_dotpr(qPtr, 1, cPtr.advanced(by: c * dim), 1, &dot, vDSP_Length(dim))
                        cScores.append((c, dot))
                    }
                }
            }
            cScores.sort { $0.sim > $1.sim }
            var gathered: [Int] = []
            for p in 0..<probeCount {
                gathered.append(contentsOf: invertedLists[cScores[p].idx])
            }
            candidateIndices = gathered
        } else {
            candidateIndices = Array(0..<n)
        }

        var hits: [(relativePath: String, cosine: Float)] = []
        q.withUnsafeBufferPointer { qBuf in
            fileVectorsF32.withUnsafeBufferPointer { fBuf in
                let qPtr = qBuf.baseAddress!
                let fPtr = fBuf.baseAddress!
                for idx in candidateIndices {
                    let path = filePaths[idx]
                    if !excludedPaths.isEmpty && excludedPaths.contains(path) { continue }
                    var dot: Float = 0
                    vDSP_dotpr(qPtr, 1, fPtr.advanced(by: idx * dim), 1, &dot, vDSP_Length(dim))
                    if dot >= minCosine {
                        hits.append((path, dot))
                    }
                }
            }
        }
        hits.sort { a, b in
            if abs(a.cosine - b.cosine) > 1e-6 { return a.cosine > b.cosine }
            return a.relativePath < b.relativePath
        }
        return Array(hits.prefix(topK))
    }

    // MARK: - Decompounded Concept Hashing + `last-epoch-mean` `e-means`

    private static let stopwords: Set<String> = [
        "the", "and", "for", "with", "from", "that", "this", "what", "where", "when",
        "which", "how", "does", "are", "was", "were", "have", "has", "had", "into",
        "about", "can", "will", "should", "would", "could", "our", "your", "their",
        "let", "var", "func", "def", "pub", "public", "private", "internal", "static",
        "struct", "class", "enum", "return", "import", "true", "false", "nil", "null",
        "self", "void", "int", "float", "sources", "swift"
    ]

    private static let conceptRing: [String: [String]] = [
        "lru": ["least", "recent", "evict", "cache"],
        "least": ["lru", "evict"],
        "evict": ["lru", "purge", "cache", "slot"],
        "kv": ["key", "value", "cache", "page"],
        "auth": ["authenticate", "credential", "token", "login", "bearer", "jwt"],
        "authenticate": ["auth", "credential", "login", "jwt", "bearer", "verify"],
        "credential": ["auth", "jwt", "bearer", "signature", "verify"],
        "jwt": ["auth", "authenticate", "credential", "login", "token"],
        "bearer": ["auth", "credential", "login", "jwt"],
        "login": ["auth", "jwt", "bearer", "credential"],
        "tx": ["transaction", "ledger", "atomic", "commit"],
        "transaction": ["ledger", "post", "journal", "settle", "balance"],
        "ledger": ["account", "balance", "post", "reconcile", "currency"],
        "reconcile": ["ledger", "balance", "post", "journal", "settle"],
        "account": ["ledger", "post", "journal", "settle", "currency"],
        "balance": ["ledger", "account", "post", "settle", "delta"],
        "post": ["ledger", "balance", "account", "reconcile"],
        "journal": ["ledger", "account", "reconcile", "balance"],
        "db": ["database", "query", "table", "schema", "sql"],
        "sql": ["database", "query", "table", "schema"],
        "cfg": ["config", "setting", "option"],
        "alloc": ["allocate", "memory", "arena", "buffer", "pool", "slab", "carve"],
        "allocate": ["alloc", "arena", "slab", "carve", "buffer", "pool"],
        "memory": ["alloc", "arena", "slab", "buffer", "contiguous"],
        "arena": ["alloc", "allocate", "memory", "buffer", "pool"],
        "slab": ["alloc", "allocate", "memory", "pool"],
        "quant": ["quantize", "codebook", "compress", "bit"],
        "vq": ["vector", "quantize", "codebook", "centroid"],
        "sched": ["schedule", "dispatch", "queue", "worker", "batch", "enqueue"],
        "schedule": ["queue", "enqueue", "worker", "batch", "dispatch"],
        "dispatch": ["enqueue", "queue", "worker", "batch", "background"],
        "task": ["batch", "queue", "enqueue", "job", "worker"],
        "enqueue": ["dispatch", "schedule", "task", "worker"],
        "queue": ["schedule", "dispatch", "worker", "job"],
        "net": ["network", "socket", "http", "endpoint", "rpc"],
        "slot": ["pin", "cache", "store", "entry", "lru"],
    ]

    public static func embedText(_ text: String) -> [Float] {
        let d = embeddingDim
        var vec = [Float](repeating: 0, count: d)
        let weights = extractWeightedTerms(text, scale: 1.0)
        guard !weights.isEmpty else { return vec }
        for (term, tf) in weights {
            let w = 1.0 + log(max(1.0, tf))
            projectTerm(term, weight: w, into: &vec)
        }
        normalizeUnit(&vec)
        return vec
    }

    private static func extractWeightedTerms(_ text: String, scale: Float) -> [String: Float] {
        let rawTokens = splitDecompounded(text)
        guard !rawTokens.isEmpty else { return [:] }
        var counts: [String: Float] = [:]
        for raw in rawTokens {
            let low = raw.lowercased()
            if low.count < 2 || stopwords.contains(low) { continue }
            let stem = normalizeStem(low)
            guard stem.count >= 2, !stopwords.contains(stem) else { continue }
            counts[stem, default: 0] += scale
            if stem.count >= 5 {
                counts["#p_" + String(stem.prefix(5)), default: 0] += 0.40 * scale
            }
            if let expansions = conceptRing[low] ?? conceptRing[stem] ?? conceptRing[stem + "e"] {
                for syn in expansions {
                    counts[normalizeStem(syn), default: 0] += 0.75 * scale
                }
            }
        }
        return counts
    }

    private static func projectTerm(_ term: String, weight: Float, into vec: inout [Float]) {
        let d = embeddingDim
        let h1 = fnv1a64(term, seed: 0xCBF2_9CE4_8422_2325)
        let h2 = fnv1a64(term, seed: 0x9E37_79B9_7F4A_7C15)
        let idx1 = Int(h1 % UInt64(d))
        let sgn1: Float = ((h1 >> 32) & 1) == 0 ? 1.0 : -1.0
        let idx2 = Int(h2 % UInt64(d))
        let sgn2: Float = ((h2 >> 32) & 1) == 0 ? 1.0 : -1.0
        vec[idx1] += sgn1 * weight
        vec[idx2] += sgn2 * 0.5 * weight
    }

    private static func normalizeUnit(_ vec: inout [Float]) {
        var normSq: Float = 0
        vDSP_dotpr(vec, 1, vec, 1, &normSq, vDSP_Length(vec.count))
        if normSq > 1e-12 {
            var inv = 1.0 / sqrt(normSq)
            vDSP_vsmul(vec, 1, &inv, &vec, 1, vDSP_Length(vec.count))
        }
    }

    private static func splitDecompounded(_ text: String) -> [String] {
        var tokens: [String] = []
        var cur: [Character] = []
        let chars = Array(text)
        for i in 0..<chars.count {
            let ch = chars[i]
            if ch.isLetter || ch.isNumber {
                if !cur.isEmpty, ch.isUppercase {
                    let prev = cur[cur.count - 1]
                    let nextIsLower = (i + 1 < chars.count) && chars[i + 1].isLowercase
                    if prev.isLowercase || (prev.isUppercase && nextIsLower && cur.count >= 2) {
                        tokens.append(String(cur))
                        cur.removeAll(keepingCapacity: true)
                    }
                }
                cur.append(ch)
            } else if !cur.isEmpty {
                tokens.append(String(cur))
                cur.removeAll(keepingCapacity: true)
            }
        }
        if !cur.isEmpty { tokens.append(String(cur)) }
        return tokens
    }

    private static func normalizeStem(_ word: String) -> String {
        var s = word
        for suffix in ["ation", "ition", "ments", "ment", "ings", "ing", "edly", "ies", "ers", "ors", "ion", "ed", "es", "er", "or", "ly", "s"] {
            if s.count >= suffix.count + 3 && s.hasSuffix(suffix) {
                if suffix == "ies" {
                    s = String(s.dropLast(3)) + "y"
                } else {
                    s = String(s.dropLast(suffix.count))
                }
                break
            }
        }
        return s
    }

    private static func fnv1a64(_ s: String, seed: UInt64) -> UInt64 {
        var h = seed
        for b in s.utf8 {
            h ^= UInt64(b)
            h = h &* 0x0000_0100_0000_01B3
        }
        h ^= h >> 33
        h = h &* 0xFF51_AFD7_ED55_8CCD
        h ^= h >> 33
        return h
    }

    private static func fitSphericalEmeansLastEpoch(
        vectors: [Float],
        count n: Int,
        dim d: Int,
        k: Int,
        epochs: Double,
        seed: UInt64
    ) -> (centroids: [Float], lists: [[Int]]) {
        let rawB = 0.035 * pow(Double(k), 0.13) * pow(Double(n), 0.88) * pow(max(epochs, 0.5), 0.81)
        let batchSize = min(n, max(k, 1 << Int(max(4.0, round(log2(max(Double(k), rawB)))))))
        let alpha0 = Float(min(1.0, 0.824 * pow(Double(k), -0.46) * pow(Double(batchSize), 0.90)
                               * pow(Double(n), -0.45) * pow(max(epochs, 0.5), -0.11)))
        let totalSteps = max(4, Int(ceil(epochs * Double(n) / Double(batchSize))))

        var rng = seed == 0 ? 0x5157_4D45_4D56_4C54 : seed
        func nextInt(_ bound: Int) -> Int {
            rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27
            return Int((rng &* 0x2545_F491_4F6C_DD1D) % UInt64(bound))
        }

        var pseudoN = [Float](repeating: 1e-10, count: k)
        var pseudoS = [Float](repeating: 0, count: k * d)
        for c in 0..<k {
            let pick = (c * (n / k) + nextInt(max(1, n / k))) % n
            for j in 0..<d {
                pseudoS[c * d + j] = vectors[pick * d + j] * 1e-10
            }
        }

        var centroids = [Float](repeating: 0, count: k * d)
        var batchCounts = [Float](repeating: 0, count: k)
        var batchSums = [Float](repeating: 0, count: k * d)

        func refreshCentroids() {
            for c in 0..<k {
                let invN = 1.0 / max(pseudoN[c], 1e-20)
                var normSq: Float = 0
                for j in 0..<d {
                    let v = pseudoS[c * d + j] * invN
                    centroids[c * d + j] = v
                    normSq += v * v
                }
                if normSq > 1e-12 {
                    let invNorm = 1.0 / sqrt(normSq)
                    for j in 0..<d { centroids[c * d + j] *= invNorm }
                }
            }
        }

        for step in 0..<totalSteps {
            refreshCentroids()
            for c in 0..<k { batchCounts[c] = 0 }
            for i in 0..<(k * d) { batchSums[i] = 0 }

            for _ in 0..<batchSize {
                let idx = nextInt(n)
                var bestC = 0
                var bestDot = -Float.infinity
                for c in 0..<k {
                    var dot: Float = 0
                    for j in 0..<d { dot += vectors[idx * d + j] * centroids[c * d + j] }
                    if dot > bestDot { bestDot = dot; bestC = c }
                }
                batchCounts[bestC] += 1.0
                for j in 0..<d { batchSums[bestC * d + j] += vectors[idx * d + j] }
            }

            let progress = Double(step) / Double(max(1, totalSteps - 1))
            let alpha = alpha0 * Float(0.1 + 0.9 * 0.5 * (1.0 + cos(Double.pi * progress)))
            let gamma = 1.0 - alpha
            for c in 0..<k {
                pseudoN[c] = gamma * pseudoN[c] + alpha * batchCounts[c]
                for j in 0..<d {
                    pseudoS[c * d + j] = gamma * pseudoS[c * d + j] + alpha * batchSums[c * d + j]
                }
            }

            if (step + 1) % 5 == 0 && step + 1 < totalSteps {
                let meanN = pseudoN.reduce(0, +) / Float(k)
                for c in 0..<k where pseudoN[c] < 0.1 * meanN {
                    if let donor = (0..<k).max(by: { pseudoN[$0] < pseudoN[$1] }), donor != c {
                        let halfN = 0.5 * pseudoN[donor]
                        pseudoN[donor] = halfN
                        pseudoN[c] = halfN
                        for j in 0..<d {
                            let baseVal = pseudoS[donor * d + j] * 0.5
                            let jitter: Float = (j & 1) == 0 ? 1e-4 * halfN : -1e-4 * halfN
                            pseudoS[donor * d + j] = baseVal - jitter
                            pseudoS[c * d + j] = baseVal + jitter
                        }
                    }
                }
            }
        }

        refreshCentroids()
        var lists = Array(repeating: [Int](), count: k)
        var finalSums = [Float](repeating: 0, count: k * d)
        for idx in 0..<n {
            var bestC = 0
            var bestDot = -Float.infinity
            for c in 0..<k {
                var dot: Float = 0
                for j in 0..<d { dot += vectors[idx * d + j] * centroids[c * d + j] }
                if dot > bestDot { bestDot = dot; bestC = c }
            }
            lists[bestC].append(idx)
            for j in 0..<d { finalSums[bestC * d + j] += vectors[idx * d + j] }
        }
        for c in 0..<k where !lists[c].isEmpty {
            var normSq: Float = 0
            for j in 0..<d { normSq += finalSums[c * d + j] * finalSums[c * d + j] }
            let inv = 1.0 / sqrt(max(normSq, 1e-12))
            for j in 0..<d { centroids[c * d + j] = finalSums[c * d + j] * inv }
        }
        return (centroids, lists)
    }
}
