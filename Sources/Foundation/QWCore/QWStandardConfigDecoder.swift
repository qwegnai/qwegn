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

/// Decodes the flat HF/MLX schema into descriptor-ready values.
///
/// A **sibling** of the hybrid decoder: its invariants do not fit a dense model.
public enum QWStandardConfigDecoder {
    public enum Error: Swift.Error, Equatable, CustomStringConvertible {
        case notStandardSchema
        case missing(String)
        case badGeometry(String)
        /// A feature the engine does not implement; the model is refused.
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

    /// Keys only the HYBRID schema carries. A config that has one of them is
    /// a hybrid config, so it must not be read as a standard one — it goes to
    /// the hybrid decoder, which fails it the way it always did.
    static let hybridOnlyKeys = [
        "text_config", "layer_types", "full_attention_interval",
        "linear_key_head_dim", "linear_num_key_heads", "linear_num_value_heads",
        "linear_value_head_dim", "linear_conv_kernel_dim",
        "mrope_section", "indexer_n_heads", "index_n_heads",
        "compress_ratios", "enable_hyper_connections", "mhc",
        "hc_count", "ple_layer_ids", "ngram_size",
    ]

    /// `true` when `data` is a flat standard config — it carries no hybrid
    /// key at all. Unparseable data is left to the hybrid decoder.
    public static func isStandardSchema(_ data: Data) -> Bool {
        guard let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return false
        }
        // A `layer_types` that names only full attention is NOT a hybrid key:
        // the standard path's layers are all full attention anyway, and a
        // recent transformers re-saves every qwen2 config with the list.
        let hybrid = hybridOnlyKeys.filter { $0 != "layer_types" || !isAllFullAttention(obj) }
        return !hybrid.contains { present(obj, $0) }
    }

    /// `true` when the config's `layer_types` names only full attention.
    static func isAllFullAttention(_ obj: [String: Any]) -> Bool {
        guard let types = obj["layer_types"] as? [String], !types.isEmpty else { return false }
        return types.allSatisfy { $0 == "full_attention" }
    }

    /// The feature a standard family's config asks for that the standard
    /// path cannot serve, named so the user sees it instead of the hybrid
    /// decoder asking for a key they never wrote.
    public static func standardFamilyRefusal(_ data: Data) -> String? {
        guard let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let types = obj["layer_types"] as? [String], !types.isEmpty else { return nil }
        let odd = types.filter { $0 != "full_attention" }
        guard !odd.isEmpty else { return nil }
        // A hybrid family legitimately names its linear layers.
        guard let modelType = obj["model_type"] as? String,
              let spec = QWModelFamilyRegistry.shared.spec(forModelType: modelType),
              standardFamilies.contains(spec.family) else { return nil }
        return "layer_types '\(odd[0])'"
    }

    /// The families the standard path serves. A config of one of these that
    /// asks for something else is refused by name.
    public static let standardFamilies: Set<QWModelDescriptor.Family> = [
        .qwen2Dense, .llamaDense, .mistralDense, .qwen3Dense, .qwen3MoE,
    ]

    /// A key that is present but JSON `null` says nothing: it is absent.
    static func present(_ obj: [String: Any], _ key: String) -> Bool {
        guard let value = obj[key] else { return false }
        return !(value is NSNull)
    }

    public static func decode(_ data: Data) throws -> QWStandardConfig {
        guard isStandardSchema(data) else { throw Error.notStandardSchema }
        let raw = try JSONDecoder().decode(Raw.self, from: data)

        func need<T>(_ value: T?, _ name: String) throws -> T {
            guard let value else { throw Error.missing(name) }
            return value
        }

        let modelType = try need(raw.model_type, "model_type")
        let hidden = try need(raw.hidden_size, "hidden_size")
        let numLayers = try need(raw.num_hidden_layers, "num_hidden_layers")
        let qHeads = try need(raw.num_attention_heads, "num_attention_heads")
        let intermediate = try need(raw.intermediate_size, "intermediate_size")
        let vocab = try need(raw.vocab_size, "vocab_size")
        let maxPos = try need(raw.max_position_embeddings, "max_position_embeddings")
        // rms_norm_eps and rope_theta change the model's numerics, so they
        // are required rather than guessed.
        let rmsEps = try need(raw.rms_norm_eps, "rms_norm_eps")
        let ropeTheta = try need(raw.rope_theta ?? raw.rope_parameters?.ropeTheta, "rope_theta")
        let eosIds = try need(raw.eos_token_id?.ids, "eos_token_id")

        // The family carries the model's defaults; an unknown type is refused.
        guard let spec = QWModelFamilyRegistry.shared.spec(forModelType: modelType) else {
            throw QWModelDescriptorError.unsupportedModelType(modelType)
        }

        // The architecture names the model's HEAD. A checkpoint that is not
        // a causal LM — an embedding, a reranker, a guard — has another
        // output, and the engine would decode it as a plain LM.
        if let archs = raw.architectures, !archs.isEmpty,
           !archs.contains(where: { $0.hasSuffix("ForCausalLM") }) {
            throw Error.unsupported("architecture '\(archs[0])'")
        }
        // A checkpoint that carries custom code needs trust_remote_code; the
        // engine runs the kernels it has, never the model's own Python. Only
        // the key's PRESENCE matters, so it is read from the raw object,
        // where no shape of value can fail the decode.
        if let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
           present(obj, "auto_map") {
            throw Error.unsupported("auto_map")
        }

        // A rope scaling curve changes the model's numerics globally, and the
        // engine does not implement one — so it is refused, not run with the
        // wrong positions. A block with no type but a factor IS a curve.
        let ropeType = raw.rope_parameters?.resolvedType ?? raw.rope_scaling?.resolvedType
        if let ropeType, ropeType != "default" {
            throw Error.unsupported("rope_scaling '\(ropeType)'")
        }
        if ropeType == nil, raw.rope_parameters?.factor != nil || raw.rope_scaling?.factor != nil {
            throw Error.unsupported("rope_scaling without a type")
        }
        // A three-part mrope section is the hybrid's; the standard path is 1-D.
        if raw.rope_parameters?.mropeSection != nil {
            throw Error.unsupported("mrope_section")
        }
        // Attention biases change the projection layout; the engine has none.
        if raw.attention_bias == true {
            throw Error.unsupported("attention_bias")
        }
        // The engine's text MLP is SwiGLU and has no bias: another activation
        // or a bias would change every number, and a silently ignored key is
        // a model that runs wrong.
        if let act = raw.hidden_act, act != "silu" {
            throw Error.unsupported("hidden_act '\(act)'")
        }
        if raw.mlp_bias == true {
            throw Error.unsupported("mlp_bias")
        }
        // A tied checkpoint has no lm_head of its own, and the engine has no
        // way to materialise one yet — so the config is refused, not served
        // from a missing weight.
        if raw.tie_word_embeddings == true {
            throw Error.unsupported("tie_word_embeddings")
        }
        // A mixed dense/MoE layer layout cannot be expressed by the
        // descriptor, and a sparse MoE cadence changes which layers are MoE.
        if let only = raw.mlp_only_layers, !only.isEmpty {
            throw Error.unsupported("mlp_only_layers")
        }
        if let step = raw.decoder_sparse_step, step != 1 {
            throw Error.unsupported("decoder_sparse_step \(step)")
        }
        // Sliding-window attention: the engine has none — a token attends the
        // whole context — so a checkpoint with an active window would run
        // with the wrong numbers past the window. The rule is the one
        // transformers uses: `use_sliding_window` decides when the family
        // declares it (Qwen2), and a bare `sliding_window` is the window
        // itself (Mistral).
        if let use = raw.use_sliding_window {
            if use { throw Error.unsupported("sliding_window") }
        } else if raw.sliding_window != nil {
            throw Error.unsupported("sliding_window")
        }

        guard qHeads > 0 else { throw Error.badGeometry("num_attention_heads \(qHeads)") }
        // head_dim may be declared explicitly, in which case it need not be
        // hidden / qHeads — some models use a different split.
        let headDim: Int
        if let declared = raw.head_dim {
            guard declared > 0 else { throw Error.badGeometry("head_dim \(declared)") }
            headDim = declared
        } else {
            guard hidden % qHeads == 0 else {
                throw Error.badGeometry("hidden_size \(hidden) is not divisible by num_attention_heads \(qHeads)")
            }
            headDim = hidden / qHeads
        }
        let kvHeads = raw.num_key_value_heads ?? qHeads
        guard kvHeads > 0, qHeads % kvHeads == 0 else {
            throw Error.badGeometry("num_attention_heads \(qHeads) is not a multiple of num_key_value_heads \(kvHeads)")
        }
        // The lane mapping covers a head in groups of 8 halves, and the
        // attention output's rotation is 32 halves at a time (the packer's
        // own granularity), so a head's width must be a multiple of 32:
        // a narrower group would straddle two heads (C112).
        guard headDim % 32 == 0 else {
            throw Error.badGeometry("head_dim \(headDim) is not a multiple of 32")
        }

        let partialRotaryFactor = raw.partial_rotary_factor
            ?? raw.rope_parameters?.partialRotaryFactor ?? 1.0
        guard partialRotaryFactor > 0, partialRotaryFactor <= 1 else {
            throw Error.badGeometry("partial_rotary_factor \(partialRotaryFactor) out of range")
        }

        // The attention output gate — the SAME rule as the hybrid path: an
        // explicit `attn_output_gate` wins, else a declared `output_gate_type`
        // or the family's default turns the gate on.
        let gateDeclared = raw.output_gate_type != nil
        let attnGate = raw.attn_output_gate ?? (gateDeclared || spec.defaultAttnOutputGate)
        // q/k RMSNorm is present on the Qwen3 class and absent on
        // Qwen2 / Llama / Mistral. The config states no key for it, so the
        // model type is the signal — a heuristic, and it is documented.
        let hasQKNorm = modelType.hasPrefix("qwen3")
        // The q/k norm's presence is the family's own, and the config may
        // say otherwise: a checkpoint that turns the norm OFF would run
        // with a norm the model does not have.
        if hasQKNorm, raw.use_qk_norm == false {
            throw Error.unsupported("use_qk_norm false")
        }
        // A bias on the q/k/v projections changes every attention number,
        // and the engine's projections carry none.
        if raw.qkv_bias == true {
            throw Error.unsupported("qkv_bias")
        }
        // The engine returns no router logits, so a model that asks for
        // them cannot be served.
        if raw.output_router_logits == true {
            throw Error.unsupported("output_router_logits")
        }
        // An expert count on a DENSE family is a mismatch, and the plan would
        // name the layer MoE and refuse it much later. Say so here.
        if raw.num_experts != nil, spec.family != .qwen3MoE {
            throw Error.badGeometry(
                "num_experts \(raw.num_experts!) on the dense family '\(spec.family.rawValue)'")
        }
        // The sparse family without an expert count is a dense model wearing
        // the sparse family's name: every layer would be routed with nothing
        // to route to.
        if spec.family == .qwen3MoE, raw.num_experts == nil {
            throw Error.badGeometry("num_experts is missing on the sparse family")
        }
        // The router's law (C123): the top-k kernel tiles the experts 32 at
        // a time and holds 16 tiles, so E is a multiple of 32, at least one
        // tile, and at most 512. A model outside that cannot be routed, so
        // it is refused here rather than half-served later. Zero experts is
        // outside the law: the tile count would be zero and the route
        // kernel would select nothing.
        if let experts = raw.num_experts {
            guard experts >= 32, experts % 32 == 0, experts <= 512 else {
                throw Error.badGeometry(
                    "num_experts \(experts) must be a multiple of 32, at least 32 and at most 512")
            }
            let topK = try need(raw.num_experts_per_tok, "num_experts_per_tok")
            guard topK > 0, topK <= 16 else {
                throw Error.badGeometry("num_experts_per_tok \(topK) must be in 1...16")
            }
        }
        // The experts' width is the model's own: the bank is carved with it,
        // so a sparse config without it would carve empty banks and fail
        // later on a weight error instead of a config refusal.
        if spec.family == .qwen3MoE {
            _ = try need(raw.moe_intermediate_size, "moe_intermediate_size")
        }
        // An experts' width on a DENSE family is a mismatch: the model has
        // no experts to give the width to, so it is refused by name.
        if let moeWidth = raw.moe_intermediate_size, moeWidth > 0, spec.family != .qwen3MoE {
            throw Error.badGeometry(
                "moe_intermediate_size \(moeWidth) on the dense family '\(spec.family.rawValue)'")
        }
        // A shared expert changes every layer's output and the standard path
        // has none — the shared term is zeroed, so a checkpoint with one
        // would silently lose half its MLP. Refuse it by name. A width of
        // zero IS "no shared expert", so it is honoured, not refused. A
        // DENSE family has no experts at all, so the same key there is a
        // mismatch rather than an unimplemented feature — it is refused
        // either way, and the message says which.
        if let shared = raw.shared_expert_intermediate_size, shared > 0 {
            if spec.family == .qwen3MoE {
                throw Error.unsupported("shared_expert_intermediate_size")
            }
            throw Error.badGeometry(
                "shared_expert_intermediate_size \(shared) on the dense family '\(spec.family.rawValue)'")
        }

        return QWStandardConfig(
            modelType: modelType,
            family: spec.family,
            hidden: hidden,
            vocab: vocab,
            numLayers: numLayers,
            qHeads: qHeads,
            kvHeads: kvHeads,
            headDim: headDim,
            intermediate: intermediate,
            maxPositionEmbeddings: maxPos,
            rmsEps: Float(rmsEps),
            ropeTheta: ropeTheta,
            partialRotaryFactor: partialRotaryFactor,
            eosTokenId: eosIds[0],
            eosTokenIds: eosIds,
            attnOutputGate: attnGate,
            hasQKNorm: hasQKNorm,
            numExperts: raw.num_experts,
            expertsPerTok: raw.num_experts_per_tok,
            moeIntermediate: raw.moe_intermediate_size,
            sharedExpertIntermediate: raw.shared_expert_intermediate_size,
            tieWordEmbeddings: raw.tie_word_embeddings ?? false,
            normTopKProb: raw.norm_topk_prob ?? false,
            quantization: (raw.quantization ?? raw.quantization_config),
            mtpLayers: max(0, raw.mtp_num_hidden_layers ?? raw.num_nextn_predict_layers ?? 0),
            mtpUseDedicatedEmbeddings: raw.mtp_use_dedicated_embeddings ?? false
        )
    }

    /// `eos_token_id` is an int in most configs and a LIST in some
    /// (llama-3.1 ships `[128001, 128008, 128009]`). Keep them all.
    struct EOSTokenIDs: Decodable {
        let ids: [Int]

        init(from decoder: Decoder) throws {
            let single = try decoder.singleValueContainer()
            if let one = try? single.decode(Int.self) {
                self.ids = [one]
                return
            }
            let list = try single.decode([Int].self)
            guard !list.isEmpty else {
                throw DecodingError.dataCorruptedError(in: single,
                                                      debugDescription: "empty eos_token_id")
            }
            self.ids = list
        }
    }

    /// A `rope_scaling` or `rope_parameters` block. Only the type decides
    /// whether the engine can run the model; the curve itself is separate work.
    struct RoPEBlock: Decodable {
        let ropeType: String?
        let legacyType: String?
        let ropeTheta: Double?
        let partialRotaryFactor: Double?
        let mropeSection: [Int]?
        let factor: Double?

        var resolvedType: String? { ropeType ?? legacyType }

        private enum CodingKeys: String, CodingKey {
            case ropeType = "rope_type"
            case legacyType = "type"
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
            case mropeSection = "mrope_section"
            case factor
        }
    }

    private struct Raw: Decodable {
        let model_type: String?
        let vocab_size: Int?
        let hidden_size: Int?
        let num_hidden_layers: Int?
        let num_attention_heads: Int?
        let num_key_value_heads: Int?
        let head_dim: Int?
        let intermediate_size: Int?
        let max_position_embeddings: Int?
        let rms_norm_eps: Double?
        let rope_theta: Double?
        let partial_rotary_factor: Double?
        let rope_parameters: RoPEBlock?
        let rope_scaling: RoPEBlock?
        let eos_token_id: EOSTokenIDs?
        let attn_output_gate: Bool?
        let output_gate_type: String?
        let num_experts: Int?
        let num_experts_per_tok: Int?
        let moe_intermediate_size: Int?
        let shared_expert_intermediate_size: Int?
        let norm_topk_prob: Bool?
        let tie_word_embeddings: Bool?
        let attention_bias: Bool?
        let mlp_bias: Bool?
        let hidden_act: String?
        let mlp_only_layers: [Int]?
        let decoder_sparse_step: Int?
        let sliding_window: Int?
        let use_sliding_window: Bool?
        let architectures: [String]?
        let use_qk_norm: Bool?
        let qkv_bias: Bool?
        let output_router_logits: Bool?
        let mtp_num_hidden_layers: Int?
        let num_nextn_predict_layers: Int?
        let mtp_use_dedicated_embeddings: Bool?
        let quantization: QWQuantizationSpec?
        let quantization_config: QWQuantizationSpec?
    }
}
