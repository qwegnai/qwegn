// QWLSAConfig.swift — Latent Sparse Attention (LSA, arXiv:2610.42147) configuration & geometry math.
// Pure: no Metal, no heavy I/O. Shared across QWCore, QWQuant, QWEngine, QWFlashNext, QWServer, and qwg.

import Foundation

/// User-facing Latent Sparse Attention execution mode (`lsaMode` knob).
public enum QWLSAMode: String, Sendable, Codable, CaseIterable, Equatable, Hashable {
    /// Disable LSA (use native dense attention or native QSA).
    case off = "off"
    /// Automatically select `.qlsa` sidecar if present, `qsa-cluster` on QSA models, or `svd` fallback.
    case auto = "auto"
    /// Require and use a `.qlsa` sidecar (`model.qlsa` or `--lsa-sidecar-path`).
    case sidecar = "sidecar"
    /// Use RoPE-free Weight-SVD index projections (from `.qlsa` sidecar if present, else synthesized at load).
    case svd = "svd"
    /// Cluster QSA 4-token pooled index keys directly via Ward clustering (Flash-Next native indexer).
    case qsaCluster = "qsa-cluster"

    public static func resolve(_ raw: String) -> QWLSAMode? {
        QWLSAMode(rawValue: raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
    }
}

/// Concrete LSA backend selected for a loaded model.
public enum QWLSAResolvedBackend: String, Sendable, Codable, Equatable, Hashable {
    case disabled = "off"
    case sidecar = "sidecar"
    case svd = "svd"
    case qsaCluster = "qsa-cluster"
}

/// Summary metadata of a discovered `.qlsa` sidecar file (`QLSA0001`).
public struct QWLSASidecarInfo: Sendable, Codable, Equatable, Hashable {
    public let path: String
    public let modelId: String
    public let geometryDigest: String
    public let method: String
    public let indexDim: Int
    public let poolSize: Int
    public let clusterBlockSize: Int
    public let bucketsPerBlock: Int
    public let maxBucketKeys: Int
    public let recommendedBudget: Int
    public let geometryMatches: Bool

    public init(
        path: String,
        modelId: String,
        geometryDigest: String,
        method: String,
        indexDim: Int = QWLSAConfig.defaultIndexDim,
        poolSize: Int = QWLSAConfig.poolSize,
        clusterBlockSize: Int = QWLSAConfig.defaultClusterBlock,
        bucketsPerBlock: Int = QWLSAConfig.bucketsPerBlock,
        maxBucketKeys: Int = QWLSAConfig.maxBucketKeys,
        recommendedBudget: Int = QWLSAConfig.defaultBudget,
        geometryMatches: Bool = true
    ) {
        self.path = path
        self.modelId = modelId
        self.geometryDigest = geometryDigest
        self.method = method
        self.indexDim = indexDim
        self.poolSize = poolSize
        self.clusterBlockSize = clusterBlockSize
        self.bucketsPerBlock = bucketsPerBlock
        self.maxBucketKeys = maxBucketKeys
        self.recommendedBudget = recommendedBudget
        self.geometryMatches = geometryMatches
    }
}

/// Complete Latent Sparse Attention runtime configuration.
public struct QWLSAConfig: Sendable, Codable, Equatable, Hashable {
    /// Fixed number of sink tokens always attended ($n_{\mathrm{sink}} = 4$, 1 four-token block).
    public static let sinkTokens: Int = 4
    /// Fixed local sliding window tokens always attended ($n_w = 512$, 128 four-token blocks).
    public static let windowTokens: Int = 512
    /// Initial pooling granularity ($n_{\mathrm{pool}} = 4$ consecutive tokens per pool block).
    public static let poolSize: Int = 4
    /// Default Ward clustering block size ($n_{\mathrm{cluster}} = 512$ tokens = 128 pool blocks).
    public static let defaultClusterBlock: Int = 512
    /// Number of Ward buckets produced per cluster block ($m = 64$, average 8 keys / 2 pool blocks per bucket).
    public static let bucketsPerBlock: Int = 64
    /// Hard cap on keys per Ward bucket ($n_b^{\max} = 16$ keys = 4 four-token pool blocks -> fits in `uchar4`).
    public static let maxBucketKeys: Int = 16
    /// Default low-rank latent index dimension ($d_{\mathrm{idx}} = 64$).
    public static let defaultIndexDim: Int = 64
    /// Latent index dimension when clustering QSA pooled keys directly ($d_{\mathrm{idx}} = 128$).
    public static let qsaClusterIndexDim: Int = 128
    /// Default sparse token budget ($k = 2048$ tokens = 512 four-token blocks).
    public static let defaultBudget: Int = 2048

    public var mode: QWLSAMode
    public var budget: Int
    public var clusterBlock: Int
    public var sidecarPath: String

    public var isEnabled: Bool { mode != .off }

    public init(
        mode: QWLSAMode = .off,
        budget: Int = QWLSAConfig.defaultBudget,
        clusterBlock: Int = QWLSAConfig.defaultClusterBlock,
        sidecarPath: String = ""
    ) {
        self.mode = mode
        self.budget = budget
        self.clusterBlock = clusterBlock
        self.sidecarPath = sidecarPath
    }

    /// Selected token budget $k$, clamped to `512...8192` and aligned to a 4-token block boundary.
    public var alignedBudget: Int {
        let clamped = min(8192, max(512, budget))
        return (clamped / Self.poolSize) * Self.poolSize
    }

    /// Number of 4-token blocks in the selected budget ($k / 4$).
    public var budgetBlocks: Int {
        alignedBudget / Self.poolSize
    }

    /// Cluster block size $n_{\mathrm{cluster}}$ aligned to a supported power-of-two (`256`, `512`, or `1024`).
    public var alignedClusterBlock: Int {
        if clusterBlock <= 256 { return 256 }
        if clusterBlock >= 1024 { return 1024 }
        return 512
    }

    /// Number of 4-token pool blocks per cluster block ($n_{\mathrm{cluster}} / 4$, default `128`).
    public var poolBlocksPerCluster: Int {
        alignedClusterBlock / Self.poolSize
    }

    /// Number of candidate buckets selected before best-first expansion ($p = 2 \cdot (k / \bar{n}_b) = k / 4$).
    public var candidateBucketsToSelect: Int {
        max(16, alignedBudget / 4)
    }

    /// Maximum 4-token blocks emitted in `lsaBlockList` for one query token:
    /// 1 sink block + $k / 4$ selected middle blocks + $(n_w / 4 + 1)$ local window blocks.
    public var maxTotalSelectedBlocks: Int {
        1 + budgetBlocks + (Self.windowTokens / Self.poolSize) + 1
    }

    /// Sequence length below which all tokens are covered by sink + budget + window,
    /// so attention bypasses sparse selection and runs bit-exact dense attention.
    public var denseBypassThreshold: Int {
        Self.sinkTokens + alignedBudget + Self.windowTokens
    }

    /// True when a sequence of `position + 1` tokens should bypass LSA and execute dense attention.
    @inline(__always)
    public func shouldBypassToDense(position: Int) -> Bool {
        position < denseBypassThreshold
    }

    /// Resolves the concrete backend for `descriptor`.
    public func resolveBackend(descriptor: QWModelDescriptor, hasValidSidecar: Bool) -> QWLSAResolvedBackend {
        switch mode {
        case .off:
            return .disabled
        case .sidecar:
            return hasValidSidecar ? .sidecar : .disabled
        case .svd:
            return hasValidSidecar ? .sidecar : .svd
        case .qsaCluster:
            if descriptor.capabilities.contains(.qsaIndexer) {
                return .qsaCluster
            }
            return hasValidSidecar ? .sidecar : .svd
        case .auto:
            if hasValidSidecar {
                return .sidecar
            }
            if descriptor.capabilities.contains(.qsaIndexer) {
                return .disabled
            }
            return .svd
        }
    }

    /// Resolves the `.qlsa` sidecar file URL for a model pack, checking `explicitSidecarPath` first,
    /// then `<modelRoot>/model.qlsa` and `<modelRoot>/assets/model.qlsa`.
    public static func resolveSidecarURL(
        packPath: String,
        explicitSidecarPath: String? = nil,
        fileManager fm: FileManager = .default
    ) -> URL? {
        if let explicit = explicitSidecarPath?.trimmingCharacters(in: .whitespacesAndNewlines),
           !explicit.isEmpty {
            let expanded = (explicit as NSString).expandingTildeInPath
            if fm.fileExists(atPath: expanded) {
                return URL(fileURLWithPath: expanded)
            }
        }
        let trimmedPack = packPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPack.isEmpty else { return nil }
        let expandedPack = (trimmedPack as NSString).expandingTildeInPath
        let root = QWModelRoot(url: URL(fileURLWithPath: expandedPack))
        let candidates = [
            root.directory.appendingPathComponent("model.qlsa"),
            root.assets.appendingPathComponent("model.qlsa"),
        ]
        for candidate in candidates where fm.fileExists(atPath: candidate.path) {
            return candidate
        }
        return nil
    }

    /// Computes the preallocated GPU byte footprint per full-attention layer for `QWLSACacheState`
    /// (`bucketCentroids`: 8 KB/512t, `bucketMembers`: 256 B/512t, `tailPoolKeys`: 16 KB fixed ring).
    public static func cacheBytesPerLayer(
        maxTokens: Int,
        indexDim: Int = defaultIndexDim,
        clusterBlock: Int = defaultClusterBlock,
        bucketsPerBlock: Int = bucketsPerBlock
    ) -> (tailPoolBytes: Int, centroidBytes: Int, memberBytes: Int, totalBytes: Int) {
        func align256(_ n: Int) -> Int { (max(256, n) + 255) & ~255 }
        let maxBlocks = max(1, (maxTokens + clusterBlock - 1) / clusterBlock)
        let poolBlocks = max(128, clusterBlock / poolSize)
        let tailBytes = align256(poolBlocks * indexDim * 2)
        let centBytes = align256(maxBlocks * bucketsPerBlock * indexDim * 2)
        let membBytes = align256(maxBlocks * bucketsPerBlock * 4)
        return (tailBytes, centBytes, membBytes, tailBytes + centBytes + membBytes)
    }
}
