import Foundation

/// The HF `quantization` / `quantization_config` block.
public struct QWQuantizationSpec: Sendable, Hashable, Codable {
    public let method: String?
    public let bits: Int?
    public let groupSize: Int?
    public let mode: String?
    public let trunk: String?
    public let expertGateUp: String?
    public let expertDown: String?

    public init(method: String? = nil, bits: Int? = nil, groupSize: Int? = nil, mode: String? = nil,
                trunk: String? = nil, expertGateUp: String? = nil, expertDown: String? = nil) {
        self.method = method
        self.bits = bits
        self.groupSize = groupSize
        self.mode = mode
        self.trunk = trunk
        self.expertGateUp = expertGateUp
        self.expertDown = expertDown
    }

    private enum CodingKeys: String, CodingKey {
        case method = "quant_method"
        case bits
        case groupSize = "group_size"
        case mode
        case trunk
        case expertGateUp = "expert_gate_up"
        case expertDown = "expert_down"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(method: try c.decodeIfPresent(String.self, forKey: .method),
                  bits: try c.decodeIfPresent(Int.self, forKey: .bits),
                  groupSize: try c.decodeIfPresent(Int.self, forKey: .groupSize),
                  mode: try c.decodeIfPresent(String.self, forKey: .mode),
                  trunk: try c.decodeIfPresent(String.self, forKey: .trunk),
                  expertGateUp: try c.decodeIfPresent(String.self, forKey: .expertGateUp),
                  expertDown: try c.decodeIfPresent(String.self, forKey: .expertDown))
    }
}

/// The flat HuggingFace / MLX config schema: top-level keys, no `text_config`,
/// no `layer_types`, no `linear_*`.
public struct QWStandardConfig: Sendable {
    public let modelType: String
    public let family: QWModelDescriptor.Family
    public let hidden: Int
    public let vocab: Int
    public let numLayers: Int
    public let qHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let intermediate: Int
    public let maxPositionEmbeddings: Int
    public let rmsEps: Float
    public let ropeTheta: Double
    public let partialRotaryFactor: Double
    public let eosTokenId: Int
    /// Every EOS id the checkpoint declares; `eosTokenId` is the first.
    public let eosTokenIds: [Int]
    public let attnOutputGate: Bool
    public let hasQKNorm: Bool
    public let numExperts: Int?
    public let expertsPerTok: Int?
    public let moeIntermediate: Int?
    public let sharedExpertIntermediate: Int?
    public let tieWordEmbeddings: Bool
    public let normTopKProb: Bool
    public let quantization: QWQuantizationSpec?
    public let mtpLayers: Int
    public let mtpUseDedicatedEmbeddings: Bool

    public init(modelType: String, family: QWModelDescriptor.Family, hidden: Int, vocab: Int,
                numLayers: Int, qHeads: Int, kvHeads: Int, headDim: Int, intermediate: Int,
                maxPositionEmbeddings: Int, rmsEps: Float, ropeTheta: Double,
                partialRotaryFactor: Double, eosTokenId: Int, eosTokenIds: [Int],
                attnOutputGate: Bool, hasQKNorm: Bool, numExperts: Int?, expertsPerTok: Int?,
                moeIntermediate: Int?, sharedExpertIntermediate: Int?,
                tieWordEmbeddings: Bool, normTopKProb: Bool,
                quantization: QWQuantizationSpec?,
                mtpLayers: Int = 0, mtpUseDedicatedEmbeddings: Bool = false) {
        self.modelType = modelType
        self.family = family
        self.hidden = hidden
        self.vocab = vocab
        self.numLayers = numLayers
        self.qHeads = qHeads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.intermediate = intermediate
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsEps = rmsEps
        self.ropeTheta = ropeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.eosTokenId = eosTokenId
        self.eosTokenIds = eosTokenIds
        self.attnOutputGate = attnOutputGate
        self.hasQKNorm = hasQKNorm
        self.numExperts = numExperts
        self.expertsPerTok = expertsPerTok
        self.moeIntermediate = moeIntermediate
        self.sharedExpertIntermediate = sharedExpertIntermediate
        self.tieWordEmbeddings = tieWordEmbeddings
        self.normTopKProb = normTopKProb
        self.quantization = quantization
        self.mtpLayers = mtpLayers
        self.mtpUseDedicatedEmbeddings = mtpUseDedicatedEmbeddings
    }
}

/// Compatibility wrapper delegating flat HF/MLX configs to `QWModelDescriptor.decode`.
public enum QWStandardConfigDecoder {
    public enum Error: Swift.Error, Equatable, CustomStringConvertible {
        case notStandardSchema
        case missing(String)
        case badGeometry(String)
        case unsupported(String)

        public var description: String {
            switch self {
            case .notStandardSchema:
                return "not a flat standard config"
            case .missing(let key):
                return "config is missing '\(key)'"
            case .badGeometry(let detail):
                return "config geometry is not derivable: \(detail)"
            case .unsupported(let feature):
                return "config asks for '\(feature)', which the engine does not implement"
            }
        }
    }

    static let hybridOnlyKeys = [
        "text_config", "layer_types", "full_attention_interval",
        "linear_key_head_dim", "linear_num_key_heads", "linear_num_value_heads",
        "linear_value_head_dim", "linear_conv_kernel_dim",
        "mrope_section", "indexer_n_heads", "index_n_heads",
        "compress_ratios", "enable_hyper_connections", "mhc",
        "hc_count", "ple_layer_ids", "ngram_size", "loop_config", "local_layer_ids",
        "full_attention_layers", "linear_attention_type",
    ]

    /// `true` when `data` is a flat standard config (carries no hybrid-only keys).
    public static func isStandardSchema(_ data: Data) -> Bool {
        guard let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return false
        }
        let hybrid = hybridOnlyKeys.filter { $0 != "layer_types" || !isAllFullAttention(obj) }
        return !hybrid.contains { present(obj, $0) }
    }

    static func isAllFullAttention(_ obj: [String: Any]) -> Bool {
        guard let types = obj["layer_types"] as? [String], !types.isEmpty else { return false }
        return types.allSatisfy { $0 == "full_attention" }
    }

    public static let standardFamilies: Set<QWModelDescriptor.Family> = [
        .qwen2Dense, .llamaDense, .mistralDense, .qwen3Dense, .qwen3MoE,
    ]

    static func present(_ obj: [String: Any], _ key: String) -> Bool {
        guard let value = obj[key] else { return false }
        return !(value is NSNull)
    }

    public static func decode(_ data: Data) throws -> QWStandardConfig {
        guard isStandardSchema(data) else { throw Error.notStandardSchema }
        let d = try QWModelDescriptor.decode(configJSON: data)
        return QWStandardConfig(
            modelType: d.modelType,
            family: d.family,
            hidden: d.hidden,
            vocab: d.vocab,
            numLayers: d.numLayers,
            qHeads: d.qHeads,
            kvHeads: d.kvHeads,
            headDim: d.headDim,
            intermediate: d.denseIntermediate ?? 0,
            maxPositionEmbeddings: d.maxPositionEmbeddings,
            rmsEps: d.rmsEps,
            ropeTheta: d.ropeTheta,
            partialRotaryFactor: d.partialRotaryFactor,
            eosTokenId: d.eosTokenId,
            eosTokenIds: d.eosTokenIds,
            attnOutputGate: d.attnOutputGate,
            hasQKNorm: d.hasQKNorm,
            numExperts: d.numExperts,
            expertsPerTok: d.expertsPerTok,
            moeIntermediate: d.moeIntermediate,
            sharedExpertIntermediate: d.sharedExpertIntermediate,
            tieWordEmbeddings: d.tieWordEmbeddings,
            normTopKProb: d.normTopKProb,
            quantization: d.quantization,
            mtpLayers: d.mtpLayers,
            mtpUseDedicatedEmbeddings: d.mtpUseDedicatedEmbeddings
        )
    }
}
