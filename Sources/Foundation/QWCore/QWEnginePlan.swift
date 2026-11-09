// QWEnginePlan.swift — The frozen numeric plan for one model pack.
// Pure: derived from QWModelGeometry alone, no GPU, no I/O. The digest is the
// cache/PSO key, so every field that changes execution takes part in it.

import Foundation
import CryptoKit

/// The numeric shape one module engine executes against.
public struct QWEnginePlan: Sendable, Equatable, Codable {
    /// How a layer attends.
    public enum LayerAttentionKind: String, Sendable, Codable, CaseIterable {
        /// Linear attention (GDN).
        case gdn
        /// Full attention without a sparse indexer.
        case fullAttention
        /// Full attention driven by a QSA sparse indexer.
        case qsaSparse
    }

    /// How a layer's feed-forward block is built.
    public enum LayerMLPKind: String, Sendable, Codable, CaseIterable {
        /// One dense MLP.
        case dense
        /// A mixture of experts.
        case moe
    }

    /// The dispatch key for a layer's kernel: attention mixer, MLP family and
    /// whether the layer carries a PLE injection.
    public struct LayerKernelKey: Hashable, Sendable, Codable {
        public let attention: LayerAttentionKind
        public let mlp: LayerMLPKind
        public let ple: Bool

        public init(attention: LayerAttentionKind, mlp: LayerMLPKind, ple: Bool) {
            self.attention = attention
            self.mlp = mlp
            self.ple = ple
        }
    }

    /// One layer: what it attends with, what it feeds forward with, and
    /// whether it carries a PLE injection. The three are independent — a
    /// MoE model can have linear layers without a PLE injection.
    public struct LayerPlan: Sendable, Equatable, Codable {
        public let index: Int
        public let attention: LayerAttentionKind
        public let mlp: LayerMLPKind
        public let hasPLE: Bool

        public init(index: Int, attention: LayerAttentionKind, mlp: LayerMLPKind, hasPLE: Bool) {
            self.index = index
            self.attention = attention
            self.mlp = mlp
            self.hasPLE = hasPLE
        }

        /// The dispatch key for this layer's kernel.
        public var kernelKey: LayerKernelKey {
            LayerKernelKey(attention: attention, mlp: mlp, ple: hasPLE)
        }
    }

    /// Pack identity this plan was derived from.
    public let modelId: String
    /// The family this plan belongs to, taken from the descriptor.
    public let family: QWModelDescriptor.Family
    /// Digest of the geometry the plan was derived from.
    public let geometryDigest: String
    /// Per-layer shapes, in module order.
    public let layers: [LayerPlan]
    /// Model hidden width.
    public let hidden: Int
    /// Dense MLP intermediate width; 0 for a pure-MoE model.
    public let intermediate: Int
    /// MoE expert intermediate width; 0 for a dense model.
    public let moeIntermediate: Int
    /// Shared-expert intermediate width; 0 when the model has none.
    public let sharedExpertIntermediate: Int
    /// Experts in the pool; 0 for a dense model.
    public let numExperts: Int
    /// Experts routed per token; 0 for a dense model.
    public let expertsPerToken: Int
    /// Query heads.
    public let qHeads: Int
    /// Key/value heads.
    public let kvHeads: Int
    /// Per-head width.
    public let headDim: Int
    /// GDN value heads.
    public let gdnVHeads: Int
    /// GDN per-head width.
    public let gdnHeadDim: Int
    /// GDN QKV channel count.
    public let gdnQKVChannels: Int
    /// q_proj output channels, including the output gate when the model has one.
    public let qProjOut: Int
    /// Key/value width: kvHeads * headDim.
    public let kvWidth: Int
    /// Attention output width: qHeads * headDim (the output gate halves qProjOut).
    public let attnOutWidth: Int
    /// Vocabulary size.
    public let vocab: Int
    /// Context the model was trained for.
    public let nativeWindow: Int
    /// KV allocation granularity, in tokens.
    public let kvStepTokens: Int
    /// Prefill rows per command-buffer submission.
    public let prefillChunkRows: Int
    /// The KV storage mode the model runs in: `f16`, `q8` or `q4`.
    /// It selects the attention kernels, so it is part of the digest.
    public let kvStorageMode: String
    /// The capabilities the model itself advertises, from its config.
    /// A module must implement every one of them or refuse the model.
    public let modelCapabilities: Set<QWModelDescriptor.Capability>
    /// The measured kernel winner map, when one was loaded alongside the
    /// model. The engine asks it which kernel to dispatch, never how.
    public let kernelDispatch: QWKernelDispatchTable?
    /// Digest over every execution-relevant field.
    public let digest: String

    /// The identity of a pack: the directory that holds it. Every pack
    /// ends in `model.qwp`, so the file name says nothing.
    public static func packIdentity(packPath: String) -> String {
        let expanded = (packPath as NSString).expandingTildeInPath
        let directory = (expanded as NSString).deletingLastPathComponent
        let name = (directory as NSString).lastPathComponent
        return name.isEmpty ? (expanded as NSString).lastPathComponent : name
    }

    /// Builds the plan from a loaded geometry.
    /// `prefillChunkRows` is the prefill rows per submission and must be positive.
    /// - Throws: `QWEnginePlanError` when the geometry cannot back a plan.
    public init(geometry: QWModelGeometry, modelId: String,
                geometryDigest: String, prefillChunkRows: Int,
                kvStorageMode: String,
                kernelDispatch: QWKernelDispatchTable? = nil) throws {
        guard prefillChunkRows > 0 else {
            throw QWEnginePlanError.invalidPrefillChunkRows(prefillChunkRows)
        }
        guard !kvStorageMode.isEmpty else {
            throw QWEnginePlanError.invalidKVStorageMode(kvStorageMode)
        }
        if let table = kernelDispatch {
            guard table.schema == QWKernelDispatchTable.currentSchema else {
                throw QWEnginePlanError.unsupportedDispatchSchema(table.schema)
            }
            var seen = Set<String>()
            for entry in table.entries {
                guard seen.insert(entry.shapeKey).inserted else {
                    throw QWEnginePlanError.duplicateDispatchShape(entry.shapeKey)
                }
            }
        }
        let descriptor = geometry.descriptor
        let isMoE = descriptor.capabilities.contains(.moe)
        if isMoE {
            guard let experts = descriptor.numExperts, experts > 0,
                  let perToken = descriptor.expertsPerTok, perToken > 0,
                  let width = descriptor.moeIntermediate, width > 0 else {
                throw QWEnginePlanError.incompleteMoEGeometry
            }
        }
        self.modelId = modelId
        self.family = descriptor.family
        self.geometryDigest = geometryDigest
        self.hidden = geometry.hidden
        self.intermediate = geometry.intermediate
        self.moeIntermediate = descriptor.moeIntermediate ?? 0
        self.sharedExpertIntermediate = descriptor.sharedExpertIntermediate ?? 0
        self.numExperts = descriptor.numExperts ?? 0
        self.expertsPerToken = descriptor.expertsPerTok ?? 0
        self.qHeads = geometry.qHeads
        self.kvHeads = geometry.kvHeads
        self.headDim = geometry.headDim
        self.gdnVHeads = geometry.gdnVHeads
        self.gdnHeadDim = geometry.gdnHeadDim
        self.gdnQKVChannels = geometry.gdnQKVChannels
        self.qProjOut = geometry.qProjOut
        self.kvWidth = geometry.kvChannels
        self.attnOutWidth = geometry.qChannels
        self.vocab = geometry.vocab
        self.nativeWindow = geometry.maxPositionEmbeddings
        self.kvStepTokens = QWCapacityPlan.kvStepTokens
        self.prefillChunkRows = prefillChunkRows
        self.kvStorageMode = kvStorageMode
        self.modelCapabilities = descriptor.capabilities
        self.kernelDispatch = kernelDispatch

        let isQSA = descriptor.capabilities.contains(.qsaIndexer)
        let pleLayers = Set(descriptor.pleModuleIndices)
        var built: [LayerPlan] = []
        built.reserveCapacity(geometry.numLayers)
        for index in 0..<geometry.numLayers {
            let attention: LayerAttentionKind
            if geometry.isFullAttention(index) {
                attention = isQSA ? .qsaSparse : .fullAttention
            } else {
                attention = .gdn
            }
            built.append(LayerPlan(index: index, attention: attention,
                                   mlp: geometry.isMoELayer(index) ? .moe : .dense,
                                   hasPLE: pleLayers.contains(index)))
        }
        self.layers = built

        var canonical = "\(modelId.count):\(modelId)|\(geometryDigest.count):\(geometryDigest)"
        canonical += "|\(descriptor.geometryDigestV2Hex)"
        canonical += "|\(hidden)|\(intermediate)|\(moeIntermediate)|\(sharedExpertIntermediate)"
        canonical += "|\(numExperts)|\(expertsPerToken)|\(qHeads)|\(kvHeads)|\(headDim)"
        canonical += "|\(gdnVHeads)|\(gdnHeadDim)|\(gdnQKVChannels)|\(qProjOut)|\(kvWidth)"
        canonical += "|\(attnOutWidth)|\(vocab)|\(nativeWindow)|\(kvStepTokens)|\(prefillChunkRows)"
        canonical += "|\(family.rawValue)|\(kvStorageMode.count):\(kvStorageMode)"
        let advertised = modelCapabilities.map(\.rawValue).sorted().joined(separator: ",")
        canonical += "|\(advertised.count):\(advertised)"
        for layer in built {
            canonical += "|\(layer.index):\(layer.attention.rawValue):\(layer.mlp.rawValue)"
            canonical += ":\(layer.hasPLE)"
        }
        // The winner map is appended only when it is present, so a plan
        // without one keeps the digest it has always had.
        if let table = kernelDispatch {
            canonical += "|\(table.canonical.count):\(table.canonical)"
        }
        self.digest = SHA256.hash(data: Data(canonical.utf8))
            .map { String(format: "%02x", $0) }.joined()
    }

    /// The kernel the bench measured as fastest for a shape, or `nil` when
    /// the shape was never measured. The caller keeps the incumbent then.
    public func kernelName(nBucket: Int, op: String, bits: Int, group: Int) -> String? {
        kernelDispatch?.winner(nBucket: nBucket, op: op, bits: bits, group: group)
    }
}

/// A plan cannot be built from the given inputs.
public enum QWEnginePlanError: Error, Equatable, CustomStringConvertible {
    case invalidPrefillChunkRows(Int)
    case invalidKVStorageMode(String)
    case incompleteMoEGeometry
    case unsupportedDispatchSchema(String)
    case duplicateDispatchShape(String)

    public var description: String {
        switch self {
        case .invalidPrefillChunkRows(let rows):
            return "QWEnginePlan: prefillChunkRows must be positive (got \(rows))"
        case .invalidKVStorageMode(let mode):
            return "QWEnginePlan: the KV storage mode must be named (got '\(mode)')"
        case .incompleteMoEGeometry:
            return "QWEnginePlan: the model is MoE but its expert geometry is missing"
        case .unsupportedDispatchSchema(let schema):
            return "QWEnginePlan: the winner map's schema '\(schema)' is not this build's"
        case .duplicateDispatchShape(let shape):
            return "QWEnginePlan: the winner map holds two entries for '\(shape)'"
        }
    }
}
