// QWLSASidecar.swift — Binary `.qlsa` (`QLSA0001`) sidecar container reader/writer and Weight-SVD synthesizer.
// Holds per-layer low-rank index projections `wq_idx` and `wk_idx` (`f16`, shape `[d_idx, in_dim]`)
// bound to a model's `geometryDigestV2Hex` for Latent Sparse Attention (arXiv:2610.42147).

import Accelerate
import Darwin
import Foundation
import QWCore

public enum QWLSASidecarError: Error, CustomStringConvertible, Equatable {
    case fileMissing(String)
    case badMagic(String)
    case truncated(String)
    case geometryMismatch(sidecar: String, expected: String)
    case modelMismatch(sidecar: String, expected: String)
    case tensorMissing(String)
    case shapeMismatch(tensor: String, got: [Int], expected: [Int])
    case svdFailed(String)

    public var description: String {
        switch self {
        case .fileMissing(let p):
            return "QWLSASidecar: sidecar file not found at '\(p)'"
        case .badMagic(let p):
            return "QWLSASidecar: bad magic in '\(p)' (expected QLSA0001)"
        case .truncated(let p):
            return "QWLSASidecar: truncated sidecar file '\(p)'"
        case .geometryMismatch(let s, let e):
            return "QWLSASidecar: sidecar geometry_digest \(s) != model \(e)"
        case .modelMismatch(let s, let e):
            return "QWLSASidecar: sidecar model_id '\(s)' != model '\(e)'"
        case .tensorMissing(let n):
            return "QWLSASidecar: tensor '\(n)' missing from sidecar"
        case .shapeMismatch(let t, let got, let exp):
            return "QWLSASidecar: tensor '\(t)' shape \(got) != expected \(exp)"
        case .svdFailed(let msg):
            return "QWLSASidecar: SVD synthesis failed: \(msg)"
        }
    }
}

/// Mmap-backed `QLSA0001` sidecar reader and writer.
public final class QWLSASidecar: @unchecked Sendable {
    public static let magic = "QLSA0001"

    public struct TensorEntry: Codable, Sendable, Equatable {
        public let shape: [Int]
        public let dtype: String
        public let rawOff: Int
        public let nBytes: Int

        enum CodingKeys: String, CodingKey {
            case shape
            case dtype
            case rawOff = "raw_off"
            case nBytes = "n_bytes"
        }

        public init(shape: [Int], dtype: String = "f16", rawOff: Int, nBytes: Int) {
            self.shape = shape
            self.dtype = dtype
            self.rawOff = rawOff
            self.nBytes = nBytes
        }
    }

    public struct Header: Codable, Sendable, Equatable {
        public struct Provenance: Codable, Sendable, Equatable {
            public let format: String
            public let modelId: String
            public let geometryDigest: String
            public let method: String
            public let indexDim: Int
            public let poolSize: Int
            public let clusterBlockSize: Int
            public let bucketsPerBlock: Int
            public let maxBucketKeys: Int
            public let recommendedBudget: Int
            public let sinkTokens: Int
            public let windowTokens: Int

            enum CodingKeys: String, CodingKey {
                case format
                case modelId = "model_id"
                case geometryDigest = "geometry_digest"
                case method
                case indexDim = "index_dim"
                case poolSize = "pool_size"
                case clusterBlockSize = "cluster_block_size"
                case bucketsPerBlock = "buckets_per_block"
                case maxBucketKeys = "max_bucket_keys"
                case recommendedBudget = "recommended_budget"
                case sinkTokens = "sink_tokens"
                case windowTokens = "window_tokens"
            }

            public init(
                format: String = QWLSASidecar.magic,
                modelId: String,
                geometryDigest: String,
                method: String,
                indexDim: Int = QWLSAConfig.defaultIndexDim,
                poolSize: Int = QWLSAConfig.poolSize,
                clusterBlockSize: Int = QWLSAConfig.defaultClusterBlock,
                bucketsPerBlock: Int = QWLSAConfig.bucketsPerBlock,
                maxBucketKeys: Int = QWLSAConfig.maxBucketKeys,
                recommendedBudget: Int = QWLSAConfig.defaultBudget,
                sinkTokens: Int = QWLSAConfig.sinkTokens,
                windowTokens: Int = QWLSAConfig.windowTokens
            ) {
                self.format = format
                self.modelId = modelId
                self.geometryDigest = geometryDigest
                self.method = method
                self.indexDim = indexDim
                self.poolSize = poolSize
                self.clusterBlockSize = clusterBlockSize
                self.bucketsPerBlock = bucketsPerBlock
                self.maxBucketKeys = maxBucketKeys
                self.recommendedBudget = recommendedBudget
                self.sinkTokens = sinkTokens
                self.windowTokens = windowTokens
            }
        }

        public let provenance: Provenance
        public let tensors: [String: TensorEntry]

        public init(provenance: Provenance, tensors: [String: TensorEntry]) {
            self.provenance = provenance
            self.tensors = tensors
        }
    }

    public let url: URL
    public let header: Header
    private let rawBase: UnsafeMutableRawPointer
    private let rawCount: Int

    deinit {
        if rawCount > 0 {
            _ = munmap(rawBase, rawCount)
        }
    }

    /// Fast header-only inspection of a `.qlsa` sidecar file without mapping tensor payloads.
    public static func peekInfo(url: URL, descriptor: QWModelDescriptor? = nil) throws -> QWLSASidecarInfo {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw QWLSASidecarError.fileMissing(url.path)
        }
        let fh = try FileHandle(forReadingFrom: url)
        defer { try? fh.close() }
        guard let prefix = try fh.read(upToCount: 16), prefix.count == 16 else {
            throw QWLSASidecarError.truncated(url.path)
        }
        guard String(decoding: prefix[0..<8], as: UTF8.self) == magic else {
            throw QWLSASidecarError.badMagic(url.path)
        }
        var hdrLen: UInt64 = 0
        for i in 0..<8 {
            hdrLen |= UInt64(prefix[8 + i]) << (8 * i)
        }
        let fileSize = fh.seekToEndOfFile()
        guard fileSize >= 16 + hdrLen else {
            throw QWLSASidecarError.truncated(url.path)
        }
        try fh.seek(toOffset: fileSize - hdrLen)
        guard let hdrData = try fh.read(upToCount: Int(hdrLen)), hdrData.count == Int(hdrLen) else {
            throw QWLSASidecarError.truncated(url.path)
        }
        let decoded = try JSONDecoder().decode(Header.self, from: hdrData)
        let prov = decoded.provenance
        let matches: Bool
        if let descriptor {
            matches = (prov.geometryDigest == descriptor.geometryDigestV2Hex)
        } else {
            matches = true
        }
        return QWLSASidecarInfo(
            path: url.path,
            modelId: prov.modelId,
            geometryDigest: prov.geometryDigest,
            method: prov.method,
            indexDim: prov.indexDim,
            poolSize: prov.poolSize,
            clusterBlockSize: prov.clusterBlockSize,
            bucketsPerBlock: prov.bucketsPerBlock,
            maxBucketKeys: prov.maxBucketKeys,
            recommendedBudget: prov.recommendedBudget,
            geometryMatches: matches
        )
    }

    /// Opens a `QLSA0001` sidecar and validates its geometry digest against `descriptor`.
    public init(url: URL, descriptor: QWModelDescriptor? = nil, skipIdentityCheck: Bool = false) throws {
        self.url = url
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw QWLSASidecarError.fileMissing(url.path) }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { throw QWLSASidecarError.fileMissing(url.path) }
        let size = Int(st.st_size)
        guard size > 16 else { throw QWLSASidecarError.truncated(url.path) }
        let mapped = mmap(nil, size, PROT_READ, MAP_SHARED, fd, 0)
        guard let ptr = mapped, ptr != MAP_FAILED else {
            throw QWLSASidecarError.fileMissing(url.path)
        }
        self.rawBase = ptr
        self.rawCount = size

        let bytes = ptr.assumingMemoryBound(to: UInt8.self)
        let magicSlice = UnsafeBufferPointer(start: bytes, count: 8)
        guard String(decoding: magicSlice, as: UTF8.self) == Self.magic else {
            _ = munmap(ptr, size)
            throw QWLSASidecarError.badMagic(url.path)
        }
        var hdrLen: UInt64 = 0
        for i in 0..<8 {
            hdrLen |= UInt64(bytes[8 + i]) << (8 * i)
        }
        let hdrStart = size - Int(hdrLen)
        guard hdrStart >= 16 else {
            _ = munmap(ptr, size)
            throw QWLSASidecarError.truncated(url.path)
        }
        let hdrData = Data(bytes: ptr.advanced(by: hdrStart), count: Int(hdrLen))
        let decoded: Header
        do {
            decoded = try JSONDecoder().decode(Header.self, from: hdrData)
        } catch {
            _ = munmap(ptr, size)
            throw error
        }
        if !skipIdentityCheck, let descriptor {
            let expectedDigest = descriptor.geometryDigestV2Hex
            guard decoded.provenance.geometryDigest == expectedDigest else {
                _ = munmap(ptr, size)
                throw QWLSASidecarError.geometryMismatch(
                    sidecar: decoded.provenance.geometryDigest,
                    expected: expectedDigest
                )
            }
        }
        self.header = decoded
    }

    public var info: QWLSASidecarInfo {
        let p = header.provenance
        return QWLSASidecarInfo(
            path: url.path,
            modelId: p.modelId,
            geometryDigest: p.geometryDigest,
            method: p.method,
            indexDim: p.indexDim,
            poolSize: p.poolSize,
            clusterBlockSize: p.clusterBlockSize,
            bucketsPerBlock: p.bucketsPerBlock,
            maxBucketKeys: p.maxBucketKeys,
            recommendedBudget: p.recommendedBudget,
            geometryMatches: true
        )
    }

    public func entry(_ name: String) throws -> TensorEntry {
        guard let e = header.tensors[name] else {
            throw QWLSASidecarError.tensorMissing(name)
        }
        return e
    }

    /// Copies a sidecar tensor's raw `f16` bytes into `dst`.
    public func copyTensorBytes(_ name: String, to dst: UnsafeMutableRawPointer, expectedShape: [Int]? = nil) throws {
        let e = try entry(name)
        if let expectedShape, e.shape != expectedShape {
            throw QWLSASidecarError.shapeMismatch(tensor: name, got: e.shape, expected: expectedShape)
        }
        let count = e.shape.reduce(1, *) * 2
        guard e.rawOff >= 16, e.rawOff + count <= rawCount else {
            throw QWLSASidecarError.truncated(url.path)
        }
        dst.copyMemory(from: UnsafeRawPointer(rawBase).advanced(by: e.rawOff), byteCount: count)
    }

    /// Reads a sidecar tensor as an array of `UInt16` (`Float16` bitPatterns).
    public func readF16Tensor(_ name: String, expectedShape: [Int]? = nil) throws -> [UInt16] {
        let e = try entry(name)
        if let expectedShape, e.shape != expectedShape {
            throw QWLSASidecarError.shapeMismatch(tensor: name, got: e.shape, expected: expectedShape)
        }
        let elems = e.shape.reduce(1, *)
        let byteCount = elems * 2
        guard e.rawOff >= 16, e.rawOff + byteCount <= rawCount else {
            throw QWLSASidecarError.truncated(url.path)
        }
        var out = [UInt16](repeating: 0, count: elems)
        out.withUnsafeMutableBytes { buf in
            buf.baseAddress!.copyMemory(from: UnsafeRawPointer(rawBase).advanced(by: e.rawOff), byteCount: byteCount)
        }
        return out
    }

    /// Writes a `QLSA0001` sidecar file atomically.
    public static func write(
        to url: URL,
        provenance: Header.Provenance,
        layerTensors: [Int: (wq: [UInt16], wk: [UInt16])],
        wqShape: [Int],
        wkShape: [Int]
    ) throws {
        var payload = Data()
        // Reserve 16-byte prefix (8B magic + 8B trailer length) + pad to 64 bytes
        payload.append(contentsOf: Array(magic.utf8))
        payload.append(contentsOf: [UInt8](repeating: 0, count: 8))
        payload.append(contentsOf: [UInt8](repeating: 0, count: 48)) // offset 64

        var entries: [String: TensorEntry] = [:]
        for layer in layerTensors.keys.sorted() {
            guard let pair = layerTensors[layer] else { continue }
            let wqName = "layers.\(layer).wq_idx"
            let wkName = "layers.\(layer).wk_idx"

            for (name, data, fallbackShape) in [(wqName, pair.wq, wqShape), (wkName, pair.wk, wkShape)] {
                let pad = (64 - (payload.count % 64)) % 64
                if pad > 0 {
                    payload.append(contentsOf: [UInt8](repeating: 0, count: pad))
                }
                let rawOff = payload.count
                let nBytes = data.count * 2
                data.withUnsafeBytes { (rawBuf: UnsafeRawBufferPointer) in
                    payload.append(rawBuf.bindMemory(to: UInt8.self))
                }
                let resolvedShape: [Int]
                if fallbackShape.reduce(1, *) == data.count {
                    resolvedShape = fallbackShape
                } else if provenance.indexDim > 0 && data.count % provenance.indexDim == 0 {
                    resolvedShape = [provenance.indexDim, data.count / provenance.indexDim]
                } else {
                    resolvedShape = fallbackShape
                }
                entries[name] = TensorEntry(shape: resolvedShape, dtype: "f16", rawOff: rawOff, nBytes: nBytes)
            }
        }

        let header = Header(provenance: provenance, tensors: entries)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let hdrData = try encoder.encode(header)
        let hdrLen = UInt64(hdrData.count)
        for i in 0..<8 {
            payload[8 + i] = UInt8((hdrLen >> (8 * i)) & 0xFF)
        }
        payload.append(hdrData)
        try payload.write(to: url, options: .atomic)
    }

    // MARK: - Fast In-Memory Weight-SVD Synthesis (Accelerate BLAS + LAPACK)

    /// Computes the top-`indexDim` principal left singular vectors of a RoPE-free key projection
    /// matrix `wKUnrot` (`[inDim, hidden]`, row-major) using Accelerate `cblas_sgemm` + `ssyev_`.
    /// Returns an orthonormal projection matrix of shape `[indexDim, inDim]` in `Float16` (`UInt16` bitPattern).
    public static func computeTopLeftSingularVectorsF16(
        matrix: [Float],
        rows inDim: Int,
        cols hidden: Int,
        rank indexDim: Int = QWLSAConfig.defaultIndexDim
    ) throws -> [UInt16] {
        guard inDim >= indexDim, hidden > 0, matrix.count == inDim * hidden else {
            throw QWLSASidecarError.svdFailed("invalid dimensions rows=\(inDim) cols=\(hidden) rank=\(indexDim)")
        }
        // Form symmetric Gram matrix G = (1 / hidden) * W * W^T of shape [inDim, inDim]
        var gram = [Float](repeating: 0, count: inDim * inDim)
        let scale: Float = 1.0 / Float(max(1, hidden))
        matrix.withUnsafeBufferPointer { wPtr in
            gram.withUnsafeMutableBufferPointer { gPtr in
                for i in 0..<inDim {
                    let rowI = wPtr.baseAddress!.advanced(by: i * hidden)
                    for j in i..<inDim {
                        let rowJ = wPtr.baseAddress!.advanced(by: j * hidden)
                        var dot: Float = 0
                        vDSP_dotpr(rowI, 1, rowJ, 1, &dot, vDSP_Length(hidden))
                        let val = dot * scale
                        gPtr[i * inDim + j] = val
                        gPtr[j * inDim + i] = val
                    }
                }
            }
        }

        // Subspace Orthogonal Iteration + Rayleigh-Ritz Cyclic Jacobi eigensolver
        // (zero deprecated CLAPACK symbols; uses vDSP_mmul on symmetric Gram matrix G).
        var q = [Float](repeating: 0, count: indexDim * inDim) // row-major [indexDim, inDim]
        var state: UInt64 = 0x5157_4C53_4130_3031
        for d in 0..<indexDim {
            let rowBase = d * inDim
            for c in 0..<inDim {
                state ^= state >> 12
                state ^= state << 25
                state ^= state >> 27
                let u = Float(state &* 0x2545_F491_4F6C_DD1D >> 32) / Float(UInt32.max)
                q[rowBase + c] = u * 2.0 - 1.0
            }
        }
        func orthonormalizeRows(_ mat: inout [Float], rows: Int, cols: Int) {
            mat.withUnsafeMutableBufferPointer { buf in
                guard let base = buf.baseAddress else { return }
                for r in 0..<rows {
                    let rPtr = base.advanced(by: r * cols)
                    for p in 0..<r {
                        let pPtr = base.advanced(by: p * cols)
                        var dot: Float = 0
                        vDSP_dotpr(rPtr, 1, pPtr, 1, &dot, vDSP_Length(cols))
                        var negDot = -dot
                        vDSP_vsma(pPtr, 1, &negDot, rPtr, 1, rPtr, 1, vDSP_Length(cols))
                    }
                    var normSq: Float = 0
                    vDSP_dotpr(rPtr, 1, rPtr, 1, &normSq, vDSP_Length(cols))
                    var invNorm = 1.0 / sqrt(max(1e-12, normSq))
                    vDSP_vsmul(rPtr, 1, &invNorm, rPtr, 1, vDSP_Length(cols))
                }
            }
        }
        orthonormalizeRows(&q, rows: indexDim, cols: inDim)

        var y = [Float](repeating: 0, count: indexDim * inDim)
        for _ in 0..<12 {
            // Since G is symmetric [inDim, inDim], Y = Q * G has shape [indexDim, inDim]
            q.withUnsafeBufferPointer { qPtr in
                gram.withUnsafeBufferPointer { gPtr in
                    y.withUnsafeMutableBufferPointer { yPtr in
                        vDSP_mmul(qPtr.baseAddress!, 1,
                                  gPtr.baseAddress!, 1,
                                  yPtr.baseAddress!, 1,
                                  vDSP_Length(indexDim), vDSP_Length(inDim), vDSP_Length(inDim))
                    }
                }
            }
            q = y
            orthonormalizeRows(&q, rows: indexDim, cols: inDim)
        }

        // Rayleigh-Ritz refinement on [indexDim, indexDim] projected matrix R = Q * G * Q^T
        q.withUnsafeBufferPointer { qPtr in
            gram.withUnsafeBufferPointer { gPtr in
                y.withUnsafeMutableBufferPointer { yPtr in
                    vDSP_mmul(qPtr.baseAddress!, 1,
                              gPtr.baseAddress!, 1,
                              yPtr.baseAddress!, 1,
                              vDSP_Length(indexDim), vDSP_Length(inDim), vDSP_Length(inDim))
                }
            }
        }
        var rMat = [Float](repeating: 0, count: indexDim * indexDim)
        for i in 0..<indexDim {
            for j in i..<indexDim {
                var dot: Float = 0
                vDSP_dotpr(y.withUnsafeBufferPointer { $0.baseAddress! + i * inDim }, 1,
                           q.withUnsafeBufferPointer { $0.baseAddress! + j * inDim }, 1,
                           &dot, vDSP_Length(inDim))
                rMat[i * indexDim + j] = dot
                rMat[j * indexDim + i] = dot
            }
        }
        // Cyclic Jacobi on rMat [indexDim, indexDim]
        var vMat = [Float](repeating: 0, count: indexDim * indexDim)
        for i in 0..<indexDim { vMat[i * indexDim + i] = 1.0 }
        for _ in 0..<10 {
            for p in 0..<(indexDim - 1) {
                for qIdx in (p + 1)..<indexDim {
                    let apq = rMat[p * indexDim + qIdx]
                    if abs(apq) < 1e-9 { continue }
                    let app = rMat[p * indexDim + p]
                    let aqq = rMat[qIdx * indexDim + qIdx]
                    let tau = (aqq - app) / (2.0 * apq)
                    let t = (tau >= 0 ? 1.0 : -1.0) / (abs(tau) + sqrt(1.0 + tau * tau))
                    let c = 1.0 / sqrt(1.0 + t * t)
                    let s = t * c
                    rMat[p * indexDim + p] = app - t * apq
                    rMat[qIdx * indexDim + qIdx] = aqq + t * apq
                    rMat[p * indexDim + qIdx] = 0
                    rMat[qIdx * indexDim + p] = 0
                    for k in 0..<indexDim where k != p && k != qIdx {
                        let akp = rMat[k * indexDim + p]
                        let akq = rMat[k * indexDim + qIdx]
                        rMat[k * indexDim + p] = c * akp - s * akq
                        rMat[p * indexDim + k] = rMat[k * indexDim + p]
                        rMat[k * indexDim + qIdx] = s * akp + c * akq
                        rMat[qIdx * indexDim + k] = rMat[k * indexDim + qIdx]
                    }
                    for k in 0..<indexDim {
                        let vkp = vMat[k * indexDim + p]
                        let vkq = vMat[k * indexDim + qIdx]
                        vMat[k * indexDim + p] = c * vkp - s * vkq
                        vMat[k * indexDim + qIdx] = s * vkp + c * vkq
                    }
                }
            }
        }
        // Sort Ritz values descending and form final eigenvectors U = V^T * Q
        let order = (0..<indexDim).sorted { rMat[$0 * indexDim + $0] > rMat[$1 * indexDim + $1] }
        var outF16 = [UInt16](repeating: 0, count: indexDim * inDim)
        for (d, eigCol) in order.enumerated() {
            var vec = [Float](repeating: 0, count: inDim)
            for k in 0..<indexDim {
                let coeff = vMat[k * indexDim + eigCol]
                let qRow = k * inDim
                for c in 0..<inDim {
                    vec[c] += coeff * q[qRow + c]
                }
            }
            var maxAbs: Float = -1.0
            var sign: Float = 1.0
            for c in 0..<inDim {
                let v = vec[c]
                if abs(v) > maxAbs {
                    maxAbs = abs(v)
                    sign = v < 0 ? -1.0 : 1.0
                }
            }
            let dstBase = d * inDim
            for c in 0..<inDim {
                outF16[dstBase + c] = Float16(vec[c] * sign).bitPattern
            }
        }
        return outF16
    }

    /// Synthesizes Weight-SVD LSA projection matrices (`wq_idx`, `wk_idx`) for all LSA-eligible
    /// decoder layers in parallel using a caller-supplied per-layer weight reader.
    private static func synthesizeLayers(
        descriptor: QWModelDescriptor,
        indexDim: Int,
        normPlusOne: Bool,
        loadLayerWeights: @Sendable (Int, QWModelDescriptor.LSAKeyProjectionSpec) throws -> (kProj: [Float], kNorm: [Float]?)
    ) throws -> [Int: (wq: [UInt16], wk: [UInt16])] {
        let layerSpecs: [(layer: Int, spec: QWModelDescriptor.LSAKeyProjectionSpec)] =
            (0..<descriptor.numLayers).compactMap { l in
                descriptor.lsaKeyProjectionSpec(layer: l).map { (l, $0) }
            }
        guard !layerSpecs.isEmpty else { return [:] }

        final class LayerBox: @unchecked Sendable {
            let lock = NSLock()
            var outputs: [Int: (wq: [UInt16], wk: [UInt16])] = [:]
            var firstError: (any Error)?
        }
        let box = LayerBox()
        DispatchQueue.concurrentPerform(iterations: layerSpecs.count) { idx in
            box.lock.lock()
            let failed = box.firstError != nil
            box.lock.unlock()
            if failed { return }
            let (layer, spec) = layerSpecs[idx]
            let kvHeads = spec.headCount
            let headDim = spec.headDim
            let sliceStart = spec.sliceStart
            let sliceDim = spec.sliceDim
            let inDim = kvHeads * sliceDim
            let cols = spec.inputDim
            do {
                let (kProjFull, kNorm) = try loadLayerWeights(layer, spec)
                guard kProjFull.count >= kvHeads * headDim * cols else {
                    throw QWLSASidecarError.svdFailed(
                        "layer \(layer) key proj count \(kProjFull.count) < \(kvHeads * headDim * cols)"
                    )
                }
                var unrotMatrix = [Float](repeating: 0, count: inDim * cols)
                for h in 0..<kvHeads {
                    for c in 0..<sliceDim {
                        let headChan = sliceStart + c
                        let srcRow = h * headDim + headChan
                        let dstRow = h * sliceDim + c
                        let gamma: Float
                        if let kNorm, !kNorm.isEmpty {
                            let normIdx = kNorm.count == headDim ? headChan : srcRow
                            let rawG = kNorm[normIdx]
                            gamma = normPlusOne ? (1.0 + rawG) : rawG
                        } else {
                            gamma = 1.0
                        }
                        let srcBase = srcRow * cols
                        let dstBase = dstRow * cols
                        for k in 0..<cols {
                            unrotMatrix[dstBase + k] = kProjFull[srcBase + k] * gamma
                        }
                    }
                }
                let basis = try computeTopLeftSingularVectorsF16(
                    matrix: unrotMatrix,
                    rows: inDim,
                    cols: cols,
                    rank: indexDim
                )
                box.lock.lock()
                box.outputs[layer] = (wq: basis, wk: basis)
                box.lock.unlock()
            } catch {
                box.lock.lock()
                if box.firstError == nil { box.firstError = error }
                box.lock.unlock()
            }
        }
        if let err = box.firstError { throw err }
        return box.outputs
    }

    private static func candidateNames(for layer: Int, spec: QWModelDescriptor.LSAKeyProjectionSpec) -> (proj: [String], norm: [String]) {
        switch spec.kind {
        case .gqaKeyProj:
            return (
                proj: [
                    "model.language_model.layers.\(layer).self_attn.k_proj.weight",
                    "model.layers.\(layer).self_attn.k_proj.weight",
                ],
                norm: [
                    "model.language_model.layers.\(layer).self_attn.k_norm.weight",
                    "model.layers.\(layer).self_attn.k_norm.weight",
                ]
            )
        case .hcaCompressWKV:
            return (
                proj: [
                    "model.language_model.layers.\(layer).self_attn.core_attention.compressor.linear_wkv.weight",
                    "model.layers.\(layer).self_attn.core_attention.compressor.linear_wkv.weight",
                    "model.language_model.layers.\(layer).self_attn.compress_kv.wkv.weight",
                    "model.layers.\(layer).self_attn.compress_kv.wkv.weight",
                ],
                norm: [
                    "model.language_model.layers.\(layer).self_attn.comp_norm.weight",
                    "model.layers.\(layer).self_attn.comp_norm.weight",
                    "model.language_model.layers.\(layer).self_attn.core_attention.compressor.norm.weight",
                    "model.layers.\(layer).self_attn.core_attention.compressor.norm.weight",
                    "model.language_model.layers.\(layer).self_attn.compress_kv.norm.weight",
                    "model.layers.\(layer).self_attn.compress_kv.norm.weight",
                ]
            )
        }
    }

    /// Synthesizes Weight-SVD LSA projection matrices (`wq_idx`, `wk_idx`) for all LSA-eligible
    /// decoder layers directly from a `QWQuant.Pack`.
    public static func synthesizeFromPack(
        pack: QWQuant.Pack,
        descriptor: QWModelDescriptor,
        indexDim: Int = QWLSAConfig.defaultIndexDim
    ) throws -> [Int: (wq: [UInt16], wk: [UInt16])] {
        nonisolated(unsafe) let pack = pack
        return try synthesizeLayers(
            descriptor: descriptor,
            indexDim: indexDim,
            normPlusOne: descriptor.usesPlusOneRMSNorm
        ) { layer, spec in
            let names = candidateNames(for: layer, spec: spec)
            let kProjFull: [Float]
            if let kProjName = names.proj.first(where: { (try? pack.entry($0)) != nil }) {
                let entry = try pack.entry(kProjName)
                if entry.rawOff != nil {
                    kProjFull = try pack.rawBf16(kProjName)
                } else if entry.hq4Off != nil {
                    kProjFull = try pack.dequantQ4g(kProjName)
                } else {
                    kProjFull = try pack.dequantHQ8(kProjName)
                }
            } else if spec.kind == .hcaCompressWKV,
                      let hca = descriptor.hcaConfig,
                      let fusedName = [
                          "model.language_model.layers.\(layer).self_attn.fused_in_proj.weight",
                          "model.layers.\(layer).self_attn.fused_in_proj.weight"
                      ].first(where: { (try? pack.entry($0)) != nil }) {
                let full = try pack.dequantQ4g(fusedName)
                let start = (hca.qLoraRank + hca.headDim) * spec.inputDim
                let count = hca.headDim * spec.inputDim
                guard start + count <= full.count else {
                    throw QWLSASidecarError.tensorMissing(spec.tensorName)
                }
                kProjFull = Array(full[start..<(start + count)])
            } else {
                throw QWLSASidecarError.tensorMissing(spec.tensorName)
            }
            let kNorm = names.norm.compactMap { try? pack.rawBf16($0) }.first
            return (kProjFull, kNorm)
        }
    }

    /// Synthesizes Weight-SVD LSA projection matrices (`wq_idx`, `wk_idx`) for all LSA-eligible
    /// decoder layers directly from a `.safetensors` container (including MLX affine `u32`).
    public static func synthesizeFromSafeTensors(
        container: QWSafeTensorsContainer,
        descriptor: QWModelDescriptor,
        indexDim: Int = QWLSAConfig.defaultIndexDim
    ) throws -> [Int: (wq: [UInt16], wk: [UInt16])] {
        try synthesizeLayers(
            descriptor: descriptor,
            indexDim: indexDim,
            normPlusOne: descriptor.usesPlusOneRMSNorm && !container.isMLXCheckpoint
        ) { layer, spec in
            let names = candidateNames(for: layer, spec: spec)
            guard let kProjName = names.proj.first(where: { container.tensor($0) != nil }) else {
                throw QWLSASidecarError.tensorMissing(spec.tensorName)
            }
            let kProjFull = try container.readFloats(kProjName)
            let kNorm = names.norm.compactMap { try? container.readFloats($0) }.first
            return (kProjFull, kNorm)
        }
    }

    /// Synthesizes Weight-SVD LSA projection matrices from either a `.qwp` pack or `.safetensors` checkpoint.
    public static func synthesizeFromModel(
        packPath: String,
        descriptor: QWModelDescriptor,
        indexDim: Int = QWLSAConfig.defaultIndexDim
    ) throws -> [Int: (wq: [UInt16], wk: [UInt16])] {
        let expanded = (packPath as NSString).expandingTildeInPath
        let loc = try QWModelLocation(url: URL(fileURLWithPath: expanded))
        switch loc.kind {
        case .qwp1:
            let pack = try QWQuant.Pack(url: loc.weights, descriptor: descriptor)
            return try synthesizeFromPack(pack: pack, descriptor: descriptor, indexDim: indexDim)
        case .safetensorsSingle, .safetensorsSharded:
            let container = try QWSafeTensorsContainer(url: loc.weights)
            return try synthesizeFromSafeTensors(container: container, descriptor: descriptor, indexDim: indexDim)
        }
    }

    /// Deterministic orthogonal basis fallback (`[indexDim, inDim]`) when weights are streamed from
    /// `.safetensors` without a `.qlsa` sidecar or in unit tests with synthetic descriptors.
    public static func synthesizeDeterministicOrthogonalBasis(
        inDim: Int,
        indexDim: Int = QWLSAConfig.defaultIndexDim,
        seed: UInt64 = 0x5157_4C53_4130_3031
    ) -> [UInt16] {
        var basis = [Float](repeating: 0, count: indexDim * inDim)
        var state = seed
        for d in 0..<indexDim {
            let rowBase = d * inDim
            for c in 0..<inDim {
                state ^= state >> 12
                state ^= state << 25
                state ^= state >> 27
                let u = Float(state &* 0x2545_F491_4F6C_DD1D >> 32) / Float(UInt32.max)
                basis[rowBase + c] = u * 2.0 - 1.0
            }
            // Modified Gram-Schmidt orthogonalization against previous rows
            for prev in 0..<d {
                let prevBase = prev * inDim
                var dot: Float = 0
                for c in 0..<inDim { dot += basis[rowBase + c] * basis[prevBase + c] }
                for c in 0..<inDim { basis[rowBase + c] -= dot * basis[prevBase + c] }
            }
            var normSq: Float = 0
            for c in 0..<inDim { normSq += basis[rowBase + c] * basis[rowBase + c] }
            let invNorm = 1.0 / sqrt(max(1e-12, normSq))
            for c in 0..<inDim { basis[rowBase + c] *= invNorm }
        }
        return basis.map { Float16($0).bitPattern }
    }
}
