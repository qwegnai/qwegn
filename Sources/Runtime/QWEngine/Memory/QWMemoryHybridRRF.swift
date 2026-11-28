// QWMemoryHybridRRF.swift — 16-byte-aligned "QWGQVEC1" Float16 Matryoshka vector
// store (`.qwg/dense_vectors.qvec`) and 3-way Reciprocal Rank Fusion (k = 60).

import Accelerate
import Foundation
import QWCore

public struct QWMemoryDenseVectorStore: Sendable, Equatable {
    public static let magicBytes: [UInt8] = Array("QWGQVEC1".utf8)
    public static let formatVersion: UInt32 = 1
    public static let defaultDimension: Int = 256
    public static let headerByteCount: Int = 32

    private struct FooterMeta: Codable {
        let relativePaths: [String]
        let anchors: [String?]
    }

    public let dimension: Int
    public let relativePaths: [String]
    public let anchors: [String?]
    public let vectorsF16: [UInt16]
    private let vectorsF32: [Float]

    public var count: Int { relativePaths.count }

    public static func == (lhs: QWMemoryDenseVectorStore, rhs: QWMemoryDenseVectorStore) -> Bool {
        lhs.dimension == rhs.dimension
            && lhs.relativePaths == rhs.relativePaths
            && lhs.anchors == rhs.anchors
            && lhs.vectorsF16 == rhs.vectorsF16
    }

    public init(
        dimension: Int = Self.defaultDimension,
        relativePaths: [String],
        anchors: [String?]? = nil,
        vectors: [[Float]]
    ) {
        let d = max(1, dimension)
        let n = min(relativePaths.count, vectors.count)
        self.dimension = d
        self.relativePaths = Array(relativePaths.prefix(n))
        let resolvedAnchors = anchors.map { Array($0.prefix(n)) } ?? Array(repeating: nil, count: n)
        self.anchors = resolvedAnchors.count < n
            ? resolvedAnchors + Array(repeating: nil, count: n - resolvedAnchors.count)
            : resolvedAnchors
        var flatF32 = [Float](repeating: 0, count: n * d)
        var flatF16 = [UInt16](repeating: 0, count: n * d)
        for i in 0..<n {
            let normed = Self.normalizeAndTruncate(vectors[i], dim: d)
            for j in 0..<d {
                let v = normed[j]
                let f16 = Float16(v)
                flatF16[i * d + j] = f16.bitPattern
                flatF32[i * d + j] = Float(f16)
            }
        }
        self.vectorsF16 = flatF16
        self.vectorsF32 = flatF32
    }

    private init(dimension: Int, relativePaths: [String], anchors: [String?], vectorsF16: [UInt16]) {
        self.dimension = dimension
        self.relativePaths = relativePaths
        self.anchors = anchors
        self.vectorsF16 = vectorsF16
        self.vectorsF32 = vectorsF16.map { Float(Float16(bitPattern: $0)) }
    }

    /// Writes `16`-byte-aligned `"QWGQVEC1"` binary header + contiguous `Float16` matrix + metadata footer.
    public func save(to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let footerData = try JSONEncoder().encode(FooterMeta(relativePaths: relativePaths, anchors: anchors))
        let vecBytesCount = vectorsF16.count * MemoryLayout<UInt16>.size
        var out = Data()
        out.reserveCapacity(Self.headerByteCount + vecBytesCount + footerData.count)
        out.append(contentsOf: Self.magicBytes)
        Self.appendUInt32LE(Self.formatVersion, to: &out)
        Self.appendUInt32LE(UInt32(dimension), to: &out)
        Self.appendUInt32LE(UInt32(relativePaths.count), to: &out)
        Self.appendUInt32LE(UInt32(footerData.count), to: &out)
        out.append(contentsOf: [UInt8](repeating: 0, count: 8))
        for bits in vectorsF16 {
            let le = bits.littleEndian
            out.append(UInt8(truncatingIfNeeded: le))
            out.append(UInt8(truncatingIfNeeded: le >> 8))
        }
        out.append(footerData)
        try out.write(to: url, options: .atomic)
    }

    public static func load(from url: URL) -> QWMemoryDenseVectorStore? {
        guard FileManager.default.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url),
              data.count >= headerByteCount else {
            return nil
        }
        let bytes = [UInt8](data)
        guard Array(bytes[0..<8]) == magicBytes else { return nil }
        let ver = readUInt32LE(bytes, offset: 8)
        let dim = Int(readUInt32LE(bytes, offset: 12))
        let count = Int(readUInt32LE(bytes, offset: 16))
        let footerLen = Int(readUInt32LE(bytes, offset: 20))
        guard ver == formatVersion, dim > 0, count >= 0, footerLen >= 2 else { return nil }
        let matrixBytes = count * dim * 2
        let footerStart = headerByteCount + matrixBytes
        guard footerStart + footerLen <= bytes.count else { return nil }

        var f16 = [UInt16](repeating: 0, count: count * dim)
        for i in 0..<(count * dim) {
            let lo = UInt16(bytes[headerByteCount + i * 2])
            let hi = UInt16(bytes[headerByteCount + i * 2 + 1]) << 8
            f16[i] = UInt16(littleEndian: lo | hi)
        }
        let footerData = Data(bytes[footerStart..<(footerStart + footerLen)])
        guard let meta = try? JSONDecoder().decode(FooterMeta.self, from: footerData),
              meta.relativePaths.count == count else {
            return nil
        }
        return QWMemoryDenseVectorStore(
            dimension: dim,
            relativePaths: meta.relativePaths,
            anchors: meta.anchors,
            vectorsF16: f16
        )
    }

    /// Queries the `.qvec` store via `Accelerate` `vDSP_dotpr` (`< 0.35 ms` for `10K` chunks).
    public func query(
        vector: [Float],
        excludedPaths: Set<String> = [],
        topK: Int = 16,
        minCosine: Float = 0.15
    ) -> [(relativePath: String, anchor: String?, cosine: Float)] {
        let n = relativePaths.count
        guard n > 0, topK > 0, !vector.isEmpty else { return [] }
        let q = Self.normalizeAndTruncate(vector, dim: dimension)
        var hits: [(relativePath: String, anchor: String?, cosine: Float)] = []
        q.withUnsafeBufferPointer { qBuf in
            vectorsF32.withUnsafeBufferPointer { mBuf in
                guard let qPtr = qBuf.baseAddress, let mPtr = mBuf.baseAddress else { return }
                for idx in 0..<n {
                    let path = relativePaths[idx]
                    if !excludedPaths.isEmpty && excludedPaths.contains(path) { continue }
                    var dot: Float = 0
                    vDSP_dotpr(qPtr, 1, mPtr.advanced(by: idx * dimension), 1, &dot, vDSP_Length(dimension))
                    if dot >= minCosine {
                        hits.append((path, anchors[idx], dot))
                    }
                }
            }
        }
        hits.sort { a, b in
            if abs(a.cosine - b.cosine) > 1e-6 { return a.cosine > b.cosine }
            return a.relativePath < b.relativePath
        }
        var deduped: [(relativePath: String, anchor: String?, cosine: Float)] = []
        var seen: Set<String> = []
        for h in hits {
            let key = "\(h.relativePath)#\(h.anchor ?? "")"
            guard !seen.contains(key) else { continue }
            seen.insert(key)
            deduped.append(h)
            if deduped.count >= topK { break }
        }
        return deduped
    }

    public static func normalizeAndTruncate(_ raw: [Float], dim: Int) -> [Float] {
        var sliced = Array(raw.prefix(dim))
        if sliced.count < dim {
            sliced.append(contentsOf: [Float](repeating: 0, count: dim - sliced.count))
        }
        var normSq: Float = 0
        vDSP_dotpr(sliced, 1, sliced, 1, &normSq, vDSP_Length(dim))
        if normSq > 1e-12 {
            var inv = 1.0 / sqrt(normSq)
            vDSP_vsmul(sliced, 1, &inv, &sliced, 1, vDSP_Length(dim))
        }
        return sliced
    }

    private static func appendUInt32LE(_ val: UInt32, to data: inout Data) {
        let le = val.littleEndian
        data.append(UInt8(truncatingIfNeeded: le))
        data.append(UInt8(truncatingIfNeeded: le >> 8))
        data.append(UInt8(truncatingIfNeeded: le >> 16))
        data.append(UInt8(truncatingIfNeeded: le >> 24))
    }

    private static func readUInt32LE(_ bytes: [UInt8], offset: Int) -> UInt32 {
        let b0 = UInt32(bytes[offset])
        let b1 = UInt32(bytes[offset + 1]) << 8
        let b2 = UInt32(bytes[offset + 2]) << 16
        let b3 = UInt32(bytes[offset + 3]) << 24
        return UInt32(littleEndian: b0 | b1 | b2 | b3)
    }
}

public enum QWMemoryHybridRRF {
    public static let kConstant: Double = 60.0
    public static let lexicalWeight: Double = 1.15
    public static let emeansWeight: Double = 0.85
    public static let denseWeight: Double = 1.20

    public struct RankedHit: Sendable, Equatable {
        public let id: String
        public let rrfScore: Double
        public let lexicalRank: Int?
        public let emeansRank: Int?
        public let denseRank: Int?
        public let anchor: String?
    }

    /// Computes 3-way weighted Reciprocal Rank Fusion (`k = 60`) across lexical, 128d e-means, and 256d dense ranks.
    public static func fuse(
        lexicalRankedIds: [String],
        emeansRankedIds: [String],
        denseRankedItems: [(id: String, anchor: String?)] = [],
        topK: Int = 12
    ) -> [RankedHit] {
        var scores: [String: Double] = [:]
        var lexRanks: [String: Int] = [:]
        var emRanks: [String: Int] = [:]
        var dnRanks: [String: Int] = [:]
        var anchors: [String: String] = [:]

        for (i, id) in lexicalRankedIds.enumerated() where lexRanks[id] == nil {
            let r = i + 1
            lexRanks[id] = r
            scores[id, default: 0] += lexicalWeight / (kConstant + Double(r))
        }
        for (i, id) in emeansRankedIds.enumerated() where emRanks[id] == nil {
            let r = i + 1
            emRanks[id] = r
            scores[id, default: 0] += emeansWeight / (kConstant + Double(r))
        }
        for (i, item) in denseRankedItems.enumerated() where dnRanks[item.id] == nil {
            let r = i + 1
            dnRanks[item.id] = r
            scores[item.id, default: 0] += denseWeight / (kConstant + Double(r))
            if let a = item.anchor { anchors[item.id] = a }
        }

        let sorted = scores.keys.sorted { lhs, rhs in
            let sL = scores[lhs] ?? 0
            let sR = scores[rhs] ?? 0
            if abs(sL - sR) > 1e-12 { return sL > sR }
            let bestL = min(lexRanks[lhs] ?? 9999, min(emRanks[lhs] ?? 9999, dnRanks[lhs] ?? 9999))
            let bestR = min(lexRanks[rhs] ?? 9999, min(emRanks[rhs] ?? 9999, dnRanks[rhs] ?? 9999))
            if bestL != bestR { return bestL < bestR }
            return lhs < rhs
        }
        return sorted.prefix(max(1, topK)).map { id in
            RankedHit(
                id: id,
                rrfScore: scores[id] ?? 0,
                lexicalRank: lexRanks[id],
                emeansRank: emRanks[id],
                denseRank: dnRanks[id],
                anchor: anchors[id]
            )
        }
    }
}
