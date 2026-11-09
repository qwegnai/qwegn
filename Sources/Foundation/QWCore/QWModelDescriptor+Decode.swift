import Foundation

extension QWModelDescriptor {

    // MARK: - Fail-closed decode from validated config JSON

    internal struct ConfigEnvelope: Decodable {
        struct RopeParameters: Decodable {
            let mrope_section: [Int]?
            let rope_theta: Double?
            let partial_rotary_factor: Double?
        }
        struct VisionConfig: Decodable {
            let depth: Int?
            let hidden_size: Int?
            let num_heads: Int?
            let intermediate_size: Int?
            let patch_size: Int?
            let temporal_patch_size: Int?
            let spatial_merge_size: Int?
            let num_position_embeddings: Int?
            let out_hidden_size: Int?
        }
        struct RawLoopSettings: Decodable {
            let start: Int?
            let end: Int?
            let eval_count: Int?
            let counts: [Int]?
        }
        struct RawMHCConfig: Decodable {
            let start: Int?
            let end: Int?
            let streams: Int?
            let iterations: Int?
            let fixed_identity: Bool?
            let pre_scale: Float?
        }
        enum MHCValue: Decodable {
            case flag(Bool)
            case config(RawMHCConfig)

            init(from decoder: Decoder) throws {
                let c = try decoder.singleValueContainer()
                if let b = try? c.decode(Bool.self) {
                    self = .flag(b)
                    return
                }
                self = .config(try c.decode(RawMHCConfig.self))
            }
        }
        struct RawLinearAttnConfig: Decodable {
            let Linear_type: String?
            let linear_type: String?
            let num_heads: Int?
            let head_dim: Int?
            let short_conv_kernel_size: Int?
            let gate_lower_bound: Float?
            let low_rank_dim: Int?
        }
        struct Text: Decodable {
            let model_type: String
            let hidden_size: Int
            let vocab_size: Int
            let num_hidden_layers: Int
            let layer_types: [String]
            var full_attention_interval: Int? = nil
            let rms_norm_eps: Double
            let max_position_embeddings: Int
            let num_attention_heads: Int
            let num_key_value_heads: Int
            var head_dim: Int? = nil
            var qk_head_dim: Int? = nil
            var qk_nope_head_dim: Int? = nil
            var qk_rope_head_dim: Int? = nil
            var v_head_dim: Int? = nil
            var kv_lora_rank: Int? = nil
            var q_lora_rank: Int? = nil
            var mla_use_nope: Bool? = nil
            var partial_rotary_factor: Double? = nil
            let rope_theta: Double?
            var rope_parameters: RopeParameters? = nil
            let output_gate_type: String?
            var attn_output_gate: Bool? = nil
            let linear_key_head_dim: Int?
            let linear_num_key_heads: Int?
            let linear_num_value_heads: Int?
            let linear_value_head_dim: Int?
            let linear_conv_kernel_dim: Int?
            var linear_attn_config: RawLinearAttnConfig? = nil
            let num_experts: Int?
            var n_routed_experts: Int? = nil
            let num_experts_per_tok: Int?
            let moe_intermediate_size: Int?
            let shared_expert_intermediate_size: Int?
            var n_shared_experts: Int? = nil
            var first_k_dense_replace: Int? = nil
            var moe_layer_freq: Int? = nil
            var routed_scaling_factor: Float? = nil
            var scoring_func: String? = nil
            var topk_method: String? = nil
            var mlp_clamp: Bool? = nil
            var swiglu_limit: Float? = nil
            var swiglu_limits: Float? = nil
            var norm_topk_prob: Bool? = nil
            let indexer_budget: Int?
            var index_topk: Int? = nil
            let indexer_compress_ratio: Int?
            var index_kpool: Int? = nil
            let indexer_head_dim: Int?
            var index_head_dim: Int? = nil
            let indexer_kv_heads: Int?
            let indexer_n_heads: Int?
            var index_n_heads: Int? = nil
            var index_kpool_compress: Bool? = nil
            var index_kpool_always_select_tail: Bool? = nil
            var index_share_for_mtp_iteration: Bool? = nil
            var indexer_rope_interleave: Bool? = nil
            let hc_count: Int?
            let hc_lowrank: Int?
            var hc_mult: Int? = nil
            var hc_sinkhorn_iters: Int? = nil
            let ple_layer_ids: [Int]?
            let ple_embed_dim: Int?
            let ple_conv_kernel_size: Int?
            let ngram_size: Int?
            let heads_per_ngram: Int?
            let ngram_vocab_size_base: Int?
            let make_ngram_vocab_size_divisible_by: Int?
            let split_ngram_parts: Int?
            let mtp_num_hidden_layers: Int?
            var num_nextn_predict_layers: Int? = nil
            let mtp_use_dedicated_embeddings: Bool?
            let eos_token_id: JSONIntOrList
            let intermediate_size: Int?           // dense models / first_k_dense_replace
            var quantization: QWQuantizationSpec? = nil
            var quantization_config: QWQuantizationSpec? = nil
            var loop_settings: RawLoopSettings? = nil
            var mhc: MHCValue? = nil
        }
        let text_config: Text?
        let vision_config: VisionConfig?
        let quantization: QWQuantizationSpec?
        let quantization_config: QWQuantizationSpec?
        let image_token_id: Int?
        let video_token_id: Int?
        let vision_start_token_id: Int?
        let vision_end_token_id: Int?
        let model_type: String?
        let loop_settings: RawLoopSettings?
        let mhc: MHCValue?
        // top-level fallback fields (dense configs have no text_config wrapper)
        let hidden_size: Int?
        let vocab_size: Int?
        let num_hidden_layers: Int?
        let layer_types: [String]?
        let full_attention_interval: Int?
        let rms_norm_eps: Double?
        let max_position_embeddings: Int?
        let num_attention_heads: Int?
        let num_key_value_heads: Int?
        let head_dim: Int?
        let qk_head_dim: Int?
        let qk_nope_head_dim: Int?
        let qk_rope_head_dim: Int?
        let v_head_dim: Int?
        let kv_lora_rank: Int?
        let q_lora_rank: Int?
        let mla_use_nope: Bool?
        let partial_rotary_factor: Double?
        let rope_theta: Double?
        let rope_parameters: RopeParameters?
        let linear_key_head_dim: Int?
        let linear_num_key_heads: Int?
        let linear_num_value_heads: Int?
        let linear_value_head_dim: Int?
        let linear_conv_kernel_dim: Int?
        let linear_attn_config: RawLinearAttnConfig?
        let mtp_num_hidden_layers: Int?
        let num_nextn_predict_layers: Int?
        let mtp_use_dedicated_embeddings: Bool?
        let eos_token_id: JSONIntOrList?
        let intermediate_size: Int?
        let output_router_logits: Bool?
    }

    public enum DescriptorError: Error, CustomStringConvertible, Equatable {
        case missingTextConfig
        case invariant(String)
        case drafterModel(String)
        public var description: String {
            switch self {
            case .missingTextConfig: return "QWCore: config JSON has neither text_config nor top-level geometry"
            case .invariant(let m): return "QWCore: descriptor invariant violated: \(m)"
            case .drafterModel(let m): return m
            }
        }
    }

    /// Decode the pinned HF config.json into a descriptor, re-deriving every
    /// channel count and refusing on a violated identity or cadence.
    public static func decode(configJSON data: Data) throws -> QWModelDescriptor {
        if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if obj["dflash_config"] != nil {
                throw DescriptorError.drafterModel("DFlash2 drafter model (cannot be loaded as a primary model)")
            }
            if let archs = obj["architectures"] as? [String],
               archs.contains(where: { $0.localizedCaseInsensitiveContains("DraftModel") || $0.localizedCaseInsensitiveContains("DFlash") }) {
                throw DescriptorError.drafterModel("DFlash2 drafter model (cannot be loaded as a primary model)")
            }
        }
        if QWStandardConfigDecoder.isStandardSchema(data) {
            return try decodeStandard(data)
        }
        if let feature = QWStandardConfigDecoder.standardFamilyRefusal(data) {
            throw QWStandardConfigDecoder.Error.unsupported(feature)
        }
        let env = try JSONDecoder().decode(ConfigEnvelope.self, from: data)
        let t: ConfigEnvelope.Text
        if let tc = env.text_config {
            t = tc
        } else {
            func need<T>(_ v: T?, _ name: String) throws -> T {
                guard let v else { throw DescriptorError.invariant("flat config missing \(name)") }
                return v
            }
            t = try ConfigEnvelope.Text(
                model_type: need(env.model_type, "model_type"),
                hidden_size: need(env.hidden_size, "hidden_size"),
                vocab_size: need(env.vocab_size, "vocab_size"),
                num_hidden_layers: need(env.num_hidden_layers, "num_hidden_layers"),
                layer_types: need(env.layer_types, "layer_types"),
                full_attention_interval: env.full_attention_interval,
                rms_norm_eps: need(env.rms_norm_eps, "rms_norm_eps"),
                max_position_embeddings: need(env.max_position_embeddings, "max_position_embeddings"),
                num_attention_heads: need(env.num_attention_heads, "num_attention_heads"),
                num_key_value_heads: need(env.num_key_value_heads, "num_key_value_heads"),
                head_dim: env.head_dim,
                qk_head_dim: env.qk_head_dim,
                qk_nope_head_dim: env.qk_nope_head_dim,
                qk_rope_head_dim: env.qk_rope_head_dim,
                v_head_dim: env.v_head_dim,
                kv_lora_rank: env.kv_lora_rank,
                q_lora_rank: env.q_lora_rank,
                mla_use_nope: env.mla_use_nope,
                partial_rotary_factor: env.partial_rotary_factor,
                rope_theta: env.rope_theta, rope_parameters: env.rope_parameters,
                output_gate_type: nil,
                linear_key_head_dim: env.linear_key_head_dim,
                linear_num_key_heads: env.linear_num_key_heads,
                linear_num_value_heads: env.linear_num_value_heads,
                linear_value_head_dim: env.linear_value_head_dim,
                linear_conv_kernel_dim: env.linear_conv_kernel_dim,
                linear_attn_config: env.linear_attn_config,
                num_experts: nil, num_experts_per_tok: nil, moe_intermediate_size: nil,
                shared_expert_intermediate_size: nil,
                indexer_budget: nil, indexer_compress_ratio: nil, indexer_head_dim: nil,
                indexer_kv_heads: nil, indexer_n_heads: nil,
                hc_count: nil, hc_lowrank: nil,
                ple_layer_ids: nil, ple_embed_dim: nil, ple_conv_kernel_size: nil,
                ngram_size: nil, heads_per_ngram: nil, ngram_vocab_size_base: nil,
                make_ngram_vocab_size_divisible_by: nil, split_ngram_parts: nil,
                mtp_num_hidden_layers: env.mtp_num_hidden_layers ?? env.num_nextn_predict_layers,
                num_nextn_predict_layers: env.num_nextn_predict_layers,
                mtp_use_dedicated_embeddings: env.mtp_use_dedicated_embeddings,
                eos_token_id: need(env.eos_token_id, "eos_token_id"),
                intermediate_size: env.intermediate_size,
                loop_settings: env.loop_settings,
                mhc: env.mhc)
        }
        func inv(_ cond: Bool, _ msg: String) throws {
            if !cond { throw DescriptorError.invariant(msg) }
        }
        let normalizedLayerTypes = t.layer_types.map { kind -> String in
            kind == "deepseek_sparse_attention" ? "full_attention" : kind
        }
        try inv(normalizedLayerTypes.count == t.num_hidden_layers,
                "layer_types count \(normalizedLayerTypes.count) != num_hidden_layers \(t.num_hidden_layers)")
        let kindsOK = normalizedLayerTypes.allSatisfy { $0 == "full_attention" || $0 == "linear_attention" }
        try inv(kindsOK, "layer_types contain unknown kind")

        let resolvedFAInterval: Int
        if let explicitInterval = t.full_attention_interval {
            resolvedFAInterval = explicitInterval
        } else if let firstFA = normalizedLayerTypes.firstIndex(of: "full_attention") {
            resolvedFAInterval = firstFA + 1
        } else {
            resolvedFAInterval = 1
        }

        let resolvedHeadDim: Int
        if let hd = t.head_dim, hd > 0 {
            resolvedHeadDim = hd
        } else if let qkHd = t.qk_head_dim, qkHd > 0 {
            resolvedHeadDim = qkHd
        } else if let nope = t.qk_nope_head_dim, let rope = t.qk_rope_head_dim, nope + rope > 0 {
            resolvedHeadDim = nope + rope
        } else {
            throw DescriptorError.invariant("flat config missing head_dim")
        }

        let isNoPEMLA = t.mla_use_nope ?? (t.kv_lora_rank != nil)
        let resolvedPartialRotary: Double
        if let prf = t.partial_rotary_factor ?? t.rope_parameters?.partial_rotary_factor {
            resolvedPartialRotary = prf
        } else if isNoPEMLA {
            resolvedPartialRotary = 1.0
        } else {
            throw DescriptorError.invariant("flat config missing partial_rotary_factor")
        }

        let ropeTheta = t.rope_theta ?? t.rope_parameters?.rope_theta ?? 10000.0
        let mropeSec: (w: Int, h: Int, t: Int)
        let hasMrope: Bool
        if let sec = t.rope_parameters?.mrope_section {
            try inv(sec.count == 3, "mrope_section must have 3 elements")
            mropeSec = (sec[0], sec[1], sec[2])
            hasMrope = true
        } else {
            mropeSec = (11, 11, 10)
            let rotHalf = Int(Double(resolvedHeadDim) * resolvedPartialRotary) / 2
            hasMrope = (rotHalf == 32)
        }

        let resolvedNumExperts = t.num_experts ?? t.n_routed_experts
        if resolvedNumExperts == nil {
            for l in 0..<t.num_hidden_layers {
                let want = (l + 1) % resolvedFAInterval == 0
                try inv((normalizedLayerTypes[l] == "full_attention") == want, "layer \(l) cadence")
            }
        }
        let linCfg = t.linear_attn_config
        let qkHeads = t.linear_num_key_heads ?? linCfg?.num_heads ?? 0
        let vHeads = t.linear_num_value_heads ?? linCfg?.num_heads ?? 0
        let gHeadDim = t.linear_key_head_dim ?? linCfg?.head_dim ?? 0
        let gConv = t.linear_conv_kernel_dim ?? linCfg?.short_conv_kernel_size ?? 0
        try inv(qkHeads > 0 && vHeads > 0 && gHeadDim > 0, "GDN heads/dim missing")
        try inv((t.linear_value_head_dim ?? gHeadDim) == gHeadDim, "GDN v-head dim != qk head dim")
        let qkv = (2 * qkHeads + vHeads) * gHeadDim
        let z = vHeads * gHeadDim

        let linType = (linCfg?.Linear_type ?? linCfg?.linear_type ?? "").lowercased()
        let isKDA = linType == "kda" || linCfg?.gate_lower_bound != nil
        let gdnLowRank = linCfg?.low_rank_dim ?? (isKDA ? gHeadDim : nil)

        if let n = resolvedNumExperts, let k = t.num_experts_per_tok {
            try inv(n > 0 && k > 0 && k <= n, "MoE top-k \(k) > experts \(n)")
        }
        let resolvedIdxHeads = t.indexer_n_heads ?? t.index_n_heads
        let resolvedIdxHeadDim = t.indexer_head_dim ?? t.index_head_dim
        let resolvedIdxBudget = t.indexer_budget ?? t.index_topk
        let resolvedIdxKvHeads = t.indexer_kv_heads ?? (resolvedIdxHeads != nil ? 1 : nil)
        let resolvedIdxCompress = t.indexer_compress_ratio
            ?? t.index_kpool
            ?? (t.index_kpool_compress == true ? 4 : (resolvedIdxHeads != nil ? 1 : nil))
        if let idx = resolvedIdxHeads {
            try inv(resolvedIdxKvHeads != nil && resolvedIdxHeadDim != nil
                    && resolvedIdxBudget != nil && resolvedIdxCompress != nil,
                    "indexer partially specified")
            _ = idx
        }
        if let hc = t.hc_count { try inv(hc >= 2, "hc_count \(hc) < 2") }
        if let pleIds = t.ple_layer_ids, !pleIds.isEmpty {
            try inv((t.ngram_size ?? 0) >= 2, "PLE without ngram_size")
            let heads = (t.ngram_size ?? 0) > 0 ? (t.ngram_size! - 1) * (t.heads_per_ngram ?? 0) : -1
            try inv(heads > 0 && (t.ple_embed_dim ?? 0) % heads == 0,
                    "ple_embed_dim \(t.ple_embed_dim ?? -1) % heads \(heads) != 0")
            for id in pleIds {
                try inv(id >= 1 && id <= t.num_hidden_layers, "ple_layer_ids one-indexed violation: \(id)")
                try inv(normalizedLayerTypes[id - 1] == "linear_attention",
                        "PLE id \(id) must land on linear_attention (got \(normalizedLayerTypes[id-1]))")
            }
        }
        let eos = t.eos_token_id.value
        let eosList = t.eos_token_id.values

        let resolvedSharedIntermediate: Int?
        if let explicitShared = t.shared_expert_intermediate_size {
            resolvedSharedIntermediate = explicitShared
        } else if let nShared = t.n_shared_experts, nShared > 0, let moeInt = t.moe_intermediate_size {
            resolvedSharedIntermediate = nShared * moeInt
        } else {
            resolvedSharedIntermediate = nil
        }

        var caps: Set<Capability> = []
        if env.vision_config != nil { caps.insert(.vision) }
        if resolvedNumExperts != nil { caps.insert(.moe) }
        if (t.ple_layer_ids?.isEmpty == false) { caps.insert(.ple) }
        if resolvedIdxHeads != nil { caps.insert(.qsaIndexer) }
        let rawMHCVal = t.mhc ?? env.mhc
        let parsedMHC: MHCConfig?
        switch rawMHCVal {
        case .config(let rm):
            if let s = rm.start, let e = rm.end, s >= 0, e > s, e <= t.num_hidden_layers {
                parsedMHC = MHCConfig(start: s, end: e, streams: max(2, rm.streams ?? 2),
                                      iterations: max(1, rm.iterations ?? 20),
                                      fixedIdentity: rm.fixed_identity ?? false, preScale: rm.pre_scale ?? 1.0)
            } else {
                parsedMHC = nil
            }
        case .flag(true):
            parsedMHC = MHCConfig(start: 0, end: t.num_hidden_layers,
                                  streams: max(2, t.hc_mult ?? 4),
                                  iterations: max(1, t.hc_sinkhorn_iters ?? 20),
                                  fixedIdentity: false, preScale: 1.0)
        case .flag(false), nil:
            parsedMHC = nil
        }
        if t.hc_count != nil || parsedMHC != nil { caps.insert(.hyperConnections) }
        let resolvedMtpLayers = max(0, t.mtp_num_hidden_layers ?? t.num_nextn_predict_layers ?? 0)
        if resolvedMtpLayers > 0 { caps.insert(.mtp) }

        let rawLoop = t.loop_settings ?? env.loop_settings
        let parsedLoop: LoopSettings? = (rawLoop.flatMap { rl in
            guard let s = rl.start, let e = rl.end, s >= 0, e > s, e <= t.num_hidden_layers else { return nil }
            return LoopSettings(start: s, end: e, evalCount: max(1, rl.eval_count ?? rl.counts?.first ?? 1), counts: rl.counts)
        })

        guard let familySpec = QWModelFamilyRegistry.shared.spec(forModelType: t.model_type) else {
            throw QWModelDescriptorError.unsupportedModelType(t.model_type)
        }
        let family = familySpec.family
        let gateDeclared: Bool = t.output_gate_type != nil
        let attnGate: Bool = t.attn_output_gate ?? (gateDeclared || familySpec.defaultAttnOutputGate)
        let resolvedGateType = t.scoring_func ?? familySpec.defaultExpertGateType
        let vc = env.vision_config
        return QWModelDescriptor(
            family: family, modelType: t.model_type, capabilities: caps,
            hidden: t.hidden_size, vocab: t.vocab_size, denseIntermediate: t.intermediate_size,
            numLayers: t.num_hidden_layers, layerTypes: normalizedLayerTypes, fullAttentionInterval: resolvedFAInterval,
            rmsEps: Float(t.rms_norm_eps), maxPositionEmbeddings: t.max_position_embeddings,
            qHeads: t.num_attention_heads, kvHeads: t.num_key_value_heads, headDim: resolvedHeadDim,
            attnOutputGate: attnGate, partialRotaryFactor: resolvedPartialRotary, ropeTheta: ropeTheta,
            indexerHeads: resolvedIdxHeads, indexerKvHeads: resolvedIdxKvHeads,
            indexerHeadDim: resolvedIdxHeadDim, indexerBudget: resolvedIdxBudget,
            indexerCompressRatio: resolvedIdxCompress,
            numExperts: resolvedNumExperts, expertsPerTok: t.num_experts_per_tok, moeIntermediate: t.moe_intermediate_size,
            sharedExpertIntermediate: resolvedSharedIntermediate, expertGateType: resolvedGateType,
            hcCount: t.hc_count, hcLowrank: t.hc_lowrank,
            gdnQKHeads: qkHeads, gdnVHeads: vHeads, gdnHeadDim: gHeadDim,
            gdnQKVChannels: qkv, gdnZChannels: z, gdnConvKernel: gConv,
            pleLayerIds: t.ple_layer_ids, pleEmbedDim: t.ple_embed_dim, pleConvKernel: t.ple_conv_kernel_size,
            ngramSize: t.ngram_size, headsPerNgram: t.heads_per_ngram, ngramVocabSizeBase: t.ngram_vocab_size_base,
            makeNgramVocabDivisibleBy: t.make_ngram_vocab_size_divisible_by, splitNgramParts: t.split_ngram_parts,
            mtpLayers: resolvedMtpLayers, mtpUseDedicatedEmbeddings: t.mtp_use_dedicated_embeddings ?? false,
            eosTokenId: eos, mropeSection: mropeSec,
            visDepth: vc?.depth ?? 27, visHidden: vc?.hidden_size ?? 1152,
            visHeads: vc?.num_heads ?? 16, visIntermediate: vc?.intermediate_size ?? 4304,
            visPatch: vc?.patch_size ?? 16, visTemporal: vc?.temporal_patch_size ?? 2,
            visMerge: vc?.spatial_merge_size ?? 2, visPosEmbeds: vc?.num_position_embeddings ?? 2304,
            visOut: vc?.out_hidden_size ?? t.hidden_size,
            imageTokenId: env.image_token_id ?? 248056,
            videoTokenId: env.video_token_id ?? 248057,
            visionStartTokenId: env.vision_start_token_id ?? 248053,
            visionEndTokenId: env.vision_end_token_id ?? 248054,
            hasQKNorm: true,
            usesPlusOneRMSNorm: familySpec.usesPlusOneRMSNorm,
            quantization: t.quantization ?? t.quantization_config ?? env.quantization ?? env.quantization_config,
            hasMropeSection: hasMrope,
            normTopKProb: t.norm_topk_prob ?? (resolvedNumExperts != nil),
            eosTokenIds: eosList,
            envelopeModelType: env.model_type,
            loopSettings: parsedLoop,
            mhcConfig: parsedMHC,
            firstKDenseReplace: max(0, t.first_k_dense_replace ?? 0),
            moeLayerFreq: max(1, t.moe_layer_freq ?? 1),
            routedScalingFactor: t.routed_scaling_factor ?? 1.0,
            swigluLimit: (t.mlp_clamp == false) ? nil : (t.swiglu_limit ?? t.swiglu_limits),
            topkMethod: t.topk_method,
            sharedExpertHasGate: (t.n_shared_experts == nil),
            mlaUseNoPE: isNoPEMLA,
            kvLoraRank: t.kv_lora_rank,
            qLoraRank: t.q_lora_rank,
            qkNopeHeadDim: t.qk_nope_head_dim,
            qkRopeHeadDim: t.qk_rope_head_dim,
            vHeadDim: t.v_head_dim,
            indexKPoolCompress: t.index_kpool_compress ?? false,
            indexKPoolAlwaysSelectTail: t.index_kpool_always_select_tail ?? false,
            indexShareForMTP: t.index_share_for_mtp_iteration ?? false,
            indexerRopeInterleave: t.indexer_rope_interleave ?? false,
            gdnVectorDecay: isKDA,
            gdnLowRankDim: gdnLowRank,
            gdnGateLowerBound: linCfg?.gate_lower_bound,
            gdnSeparateConv: isKDA)
    }

    /// Decodes a flat HF/MLX config into a descriptor.
    internal static func decodeStandard(_ data: Data) throws -> QWModelDescriptor {
        let c = try QWStandardConfigDecoder.decode(data)
        let layerTypes = [String](repeating: "full_attention", count: c.numLayers)
        var caps: Set<Capability> = []
        if c.numExperts != nil { caps.insert(.moe) }
        if c.mtpLayers > 0 { caps.insert(.mtp) }
        let usesPlusOne = QWModelFamilyRegistry.shared.spec(forModelType: c.modelType)?.usesPlusOneRMSNorm ?? false
        return QWModelDescriptor(
            family: c.family, modelType: c.modelType, capabilities: caps,
            hidden: c.hidden, vocab: c.vocab, denseIntermediate: c.intermediate,
            numLayers: c.numLayers, layerTypes: layerTypes,
            fullAttentionInterval: 1, rmsEps: c.rmsEps,
            maxPositionEmbeddings: c.maxPositionEmbeddings,
            qHeads: c.qHeads, kvHeads: c.kvHeads, headDim: c.headDim,
            attnOutputGate: c.attnOutputGate, partialRotaryFactor: c.partialRotaryFactor, ropeTheta: c.ropeTheta,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: c.numExperts, expertsPerTok: c.expertsPerTok, moeIntermediate: c.moeIntermediate,
            sharedExpertIntermediate: c.sharedExpertIntermediate, expertGateType: nil, hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 0, gdnVHeads: 0, gdnHeadDim: 0, gdnQKVChannels: 0, gdnZChannels: 0, gdnConvKernel: 0,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: c.mtpLayers, mtpUseDedicatedEmbeddings: c.mtpUseDedicatedEmbeddings,
            eosTokenId: c.eosTokenId, mropeSection: (0, 0, 0), hasQKNorm: c.hasQKNorm,
            usesPlusOneRMSNorm: usesPlusOne,
            quantization: c.quantization, hasMropeSection: false,
            tieWordEmbeddings: c.tieWordEmbeddings, normTopKProb: c.normTopKProb, eosTokenIds: c.eosTokenIds)
    }
}

/// HF configs sometimes carry eos as a scalar, sometimes as a list.
internal struct JSONIntOrList: Decodable {
    enum E: Error { case shape }
    let value: Int
    let values: [Int]
    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let i = try? c.decode(Int.self) {
            value = i
            values = [i]
            return
        }
        if let a = try? c.decode([Int].self), let f = a.first {
            value = f
            values = a
            return
        }
        throw E.shape
    }
}
