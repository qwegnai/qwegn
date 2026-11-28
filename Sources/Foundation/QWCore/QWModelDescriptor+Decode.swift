import Foundation

extension QWModelDescriptor {

    // MARK: - Fail-closed decode from validated config JSON

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

    internal static func isDrafterJSON(_ obj: [String: Any]) -> Bool {
        if obj["dflash_config"] != nil || obj["selected_prefix_code_bridge"] != nil { return true }
        if let archs = obj["architectures"] as? [String],
           archs.contains(where: { $0.localizedCaseInsensitiveContains("DraftModel") || $0.localizedCaseInsensitiveContains("DFlash") }) {
            return true
        }
        if let mt = (obj["model_type"] as? String)?.lowercased(),
           mt.contains("dflash") || mt.contains("dspine") {
            return true
        }
        return false
    }

    /// Decode the pinned HF config.json into a descriptor, re-deriving every
    /// channel count and refusing on a violated identity or cadence.
    public static func decode(configJSON data: Data) throws -> QWModelDescriptor {
        if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           isDrafterJSON(obj) {
            throw DescriptorError.drafterModel("DFlash2 drafter model (cannot be loaded as a primary model)")
        }
        let rawRoot = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        let rawText = (rawRoot?["text_config"] as? [String: Any]) ?? rawRoot ?? [:]
        let env = try JSONDecoder().decode(ConfigEnvelope.self, from: data)
        let t: ConfigEnvelope.Text
        if let tc = env.text_config {
            t = tc
        } else {
            do {
                t = try JSONDecoder().decode(ConfigEnvelope.Text.self, from: data)
            } catch DecodingError.keyNotFound(let key, _) {
                throw DescriptorError.invariant("flat config missing '\(key.stringValue)'")
            } catch DecodingError.valueNotFound(_, let ctx) {
                let name = ctx.codingPath.last?.stringValue ?? "required field"
                throw DescriptorError.invariant("flat config missing '\(name)'")
            }
        }
        func inv(_ cond: Bool, _ msg: String) throws {
            if !cond { throw DescriptorError.invariant(msg) }
        }
        guard let resolvedModelType = env.text_config != nil ? (t.model_type ?? env.model_type) : t.model_type else {
            throw DescriptorError.invariant("flat config missing 'model_type'")
        }
        guard let textFamilySpec = QWModelFamilyRegistry.shared.spec(forModelType: resolvedModelType) else {
            throw QWModelDescriptorError.unsupportedModelType(resolvedModelType)
        }
        let envFamilySpec = env.model_type.flatMap { QWModelFamilyRegistry.shared.spec(forModelType: $0) }
        let familySpec = envFamilySpec ?? textFamilySpec
        let family = familySpec.family

        try validateUnsupportedFeatures(text: t, env: env, familySpec: familySpec,
                                        rawRoot: rawRoot, rawText: rawText, inv: inv)

        try inv(t.hidden_size > 0, "hidden_size \(t.hidden_size) <= 0")
        try inv(t.vocab_size > 0, "vocab_size \(t.vocab_size) <= 0")
        try inv(t.num_hidden_layers > 0, "num_hidden_layers \(t.num_hidden_layers) <= 0")
        try inv(t.num_attention_heads > 0, "num_attention_heads \(t.num_attention_heads) <= 0")
        let resolvedKVHeads = t.num_key_value_heads ?? t.num_attention_heads
        try inv(resolvedKVHeads > 0 && t.num_attention_heads % resolvedKVHeads == 0,
                "num_attention_heads \(t.num_attention_heads) not divisible by num_key_value_heads \(resolvedKVHeads)")
        let resolvedMaxPos = t.max_position_embeddings ?? t.model_max_length
            ?? env.max_position_embeddings ?? env.model_max_length ?? 0
        try inv(resolvedMaxPos > 0, "max_position_embeddings \(resolvedMaxPos) <= 0")
        let resolvedNormEps = try resolveNormEps(text: t, familySpec: familySpec)
        try inv(resolvedNormEps > 0 && resolvedNormEps.isFinite, "rms_norm_eps \(resolvedNormEps) invalid")
        let rawLayerTypes: [String]
        if let explicitTypes = t.layer_types {
            if familySpec.defaultAllFullAttention,
               let odd = explicitTypes.first(where: { $0 != "full_attention" && $0 != "sliding_attention" }) {
                throw DescriptorError.invariant("unsupported layer_types '\(odd)'")
            }
            rawLayerTypes = explicitTypes
        } else if let ratios = t.compress_ratios {
            try inv(ratios.count >= t.num_hidden_layers,
                    "compress_ratios count \(ratios.count) < num_hidden_layers \(t.num_hidden_layers)")
            rawLayerTypes = (0..<t.num_hidden_layers).map { l in
                ratios[l] > 0 ? "compressed_attention" : "full_attention"
            }
        } else if let gs = t.layer_group_size, gs > 0 {
            rawLayerTypes = (0..<t.num_hidden_layers).map { l in
                (l + 1) % gs == 0 ? "full_attention" : "linear_attention"
            }
        } else if let explicitFA = try resolveExplicitFullAttentionLayers(text: t, inv: inv) {
            rawLayerTypes = explicitFA
        } else if rawText["local_layer_ids"] != nil || rawText["loop_config"] != nil {
            let localSet = Set((rawText["local_layer_ids"] as? [Int]) ?? [])
            rawLayerTypes = (0..<t.num_hidden_layers).map { localSet.contains($0) ? "sliding_attention" : "full_attention" }
        } else if let lfmTypes = resolveLayerTypesForLFM(text: t, familySpec: familySpec) {
            rawLayerTypes = lfmTypes
        } else if t.attention_variant == "block_sparse_gqa" || familySpec.defaultBlockSparseGQA
                    || familySpec.defaultAllFullAttention {
            rawLayerTypes = Array(repeating: "full_attention", count: t.num_hidden_layers)
        } else {
            throw DescriptorError.invariant("flat config missing 'layer_types'")
        }
        let normalizedLayerTypes = rawLayerTypes.map { kind -> String in
            kind == "deepseek_sparse_attention" ? "full_attention" : kind
        }
        try inv(normalizedLayerTypes.count == t.num_hidden_layers,
                "layer_types count \(normalizedLayerTypes.count) != num_hidden_layers \(t.num_hidden_layers)")
        let isGQAKind: (String) -> Bool = { $0 == "full_attention" || $0 == "sliding_attention" }
        if let badKind = normalizedLayerTypes.first(where: {
            !isGQAKind($0) && $0 != "linear_attention" && $0 != "compressed_attention" && $0 != "conv"
        }) {
            throw DescriptorError.invariant("layer_types contain unknown kind '\(badKind)'")
        }

        let hasNonGQALayers = normalizedLayerTypes.contains("linear_attention")
            || normalizedLayerTypes.contains("compressed_attention")
        let resolvedFAInterval: Int
        if let explicitInterval = t.full_attention_interval ?? t.layer_group_size ?? dictInt(rawText, "n_layers_per_attn_res_block") {
            resolvedFAInterval = explicitInterval
        } else if let firstFA = normalizedLayerTypes.firstIndex(where: isGQAKind) {
            let candidate = firstFA + 1
            let fullIndices = normalizedLayerTypes.indices.filter { isGQAKind(normalizedLayerTypes[$0]) }
            let uniformCadence = fullIndices.allSatisfy { ($0 + 1) % candidate == 0 }
            resolvedFAInterval = uniformCadence ? candidate : 0
        } else {
            resolvedFAInterval = 1
        }

        let hasCompressed = normalizedLayerTypes.contains("compressed_attention")
        let hasHybridMLA = t.hybrid_mla_kv_lora_rank != nil || t.hybrid_mla_attention != nil
        let isNoPEMLA = (t.mla_use_nope ?? t.use_mla_nope)
            ?? (t.kv_lora_rank != nil && t.use_mla != false && !hasCompressed && !hasHybridMLA && t.qk_rope_head_dim == nil)
        let hasUnprefixedDecoupledMLA = !isNoPEMLA && !hasHybridMLA && !hasCompressed
            && t.kv_lora_rank != nil && t.qk_nope_head_dim != nil
        let rawExplicitHeadDim = t.head_dim ?? t.attention_head_size
        let resolvedHeadDim: Int
        let resolvedV2DigestHeadDim: Int
        if let hd = rawExplicitHeadDim, hd > 0 {
            resolvedV2DigestHeadDim = hd
            if hasUnprefixedDecoupledMLA,
               let nope = t.qk_nope_head_dim,
               let rope = t.qk_rope_head_dim,
               nope + rope > 0,
               hd == rope {
                resolvedHeadDim = nope + rope
            } else {
                resolvedHeadDim = hd
            }
        } else if let qkHd = t.qk_head_dim, qkHd > 0 {
            resolvedHeadDim = qkHd
            resolvedV2DigestHeadDim = qkHd
        } else if let nope = t.qk_nope_head_dim, let rope = t.qk_rope_head_dim, nope + rope > 0 {
            resolvedHeadDim = nope + rope
            resolvedV2DigestHeadDim = nope + rope
        } else if rawExplicitHeadDim == nil
                    && (familySpec.defaultAllFullAttention || familySpec.defaultHeadDimFromHidden) {
            try inv(t.hidden_size % t.num_attention_heads == 0,
                    "hidden_size \(t.hidden_size) is not divisible by num_attention_heads \(t.num_attention_heads)")
            let derived = t.hidden_size / t.num_attention_heads
            resolvedHeadDim = derived
            resolvedV2DigestHeadDim = derived
        } else if let badHd = rawExplicitHeadDim {
            throw DescriptorError.invariant("head_dim \(badHd) <= 0")
        } else {
            throw DescriptorError.invariant("flat config missing 'head_dim'")
        }
        if !isNoPEMLA && !hasHybridMLA && !hasUnprefixedDecoupledMLA && !hasCompressed {
            if familySpec.defaultRole == .decisionEncoder {
                try inv(resolvedHeadDim >= 2 && resolvedHeadDim % 2 == 0 && resolvedHeadDim <= 256,
                        "head_dim \(resolvedHeadDim) must be even in 2...256")
            } else {
                try inv(resolvedHeadDim % 32 == 0 && resolvedHeadDim <= 256,
                        "head_dim \(resolvedHeadDim) must be a multiple of 32 in 32...256")
            }
        }
        let resolvedPartialRotary: Double
        if let prf = t.partial_rotary_factor ?? t.rope_parameters?.partial_rotary_factor
            ?? t.rope_parameters?.full_attention?.partial_rotary_factor
            ?? t.rope_parameters?.sliding_attention?.partial_rotary_factor {
            resolvedPartialRotary = prf
        } else if let rp = t.rotary_percent {
            resolvedPartialRotary = rp
        } else if isNoPEMLA || hasHybridMLA || hasUnprefixedDecoupledMLA || t.compress_ratios != nil
                    || t.attention_variant == "block_sparse_gqa" || familySpec.defaultBlockSparseGQA
                    || familySpec.defaultAllFullAttention || familySpec.defaultPartialRotaryOne
                    || rawText["loop_config"] != nil {
            resolvedPartialRotary = 1.0
        } else {
            throw DescriptorError.invariant("flat config missing 'partial_rotary_factor'")
        }

        // Minimal synthetic hybrid unit fixtures omit `rope_theta`; standard full-attention configs require it.
        let ropeTheta: Double
        if let explicitTheta = t.rope_theta ?? t.rope_parameters?.rope_theta
            ?? t.rope_parameters?.full_attention?.rope_theta
            ?? t.global_rope_theta ?? t.local_rope_theta
            ?? dictDouble(rawRoot?["mtp_config"] as? [String: Any], "rope_theta") {
            ropeTheta = explicitTheta
        } else if !familySpec.defaultAllFullAttention {
            ropeTheta = 10000.0
        } else {
            throw DescriptorError.invariant("flat config missing 'rope_theta'")
        }
        try inv(resolvedPartialRotary > 0 && resolvedPartialRotary <= 1.0 && resolvedPartialRotary.isFinite,
                "partial_rotary_factor \(resolvedPartialRotary) invalid")
        try inv(ropeTheta > 0 && ropeTheta.isFinite, "rope_theta \(ropeTheta) invalid")
        let rawMropeSec = t.rope_parameters?.mrope_section ?? env.mrope_section
        if familySpec.defaultAllFullAttention && rawMropeSec != nil {
            throw DescriptorError.invariant("unsupported 'mrope_section'")
        }
        let mropeSec: (w: Int, h: Int, t: Int)
        let hasMrope: Bool
        if let sec = rawMropeSec {
            try inv(sec.count == 3, "mrope_section must have 3 elements")
            mropeSec = (sec[0], sec[1], sec[2])
            hasMrope = true
        } else if familySpec.defaultAllFullAttention || rawText["loop_config"] != nil
                    || family == .lfm2 || family == .lfm3 || family == .lfm2Encoder || family == .modernBert
                    || hasHybridMLA || hasUnprefixedDecoupledMLA || isNoPEMLA || hasCompressed {
            mropeSec = (0, 0, 0)
            hasMrope = false
        } else {
            mropeSec = (11, 11, 10)
            let rotHalf = Int(Double(resolvedHeadDim) * resolvedPartialRotary) / 2
            hasMrope = (rotHalf == 32)
        }

        let moeConfig = try validateAndResolveMoE(text: t, familySpec: familySpec, inv: inv)
        let resolvedNumExperts = moeConfig.numExperts
        if resolvedNumExperts == nil && resolvedFAInterval > 0 && hasNonGQALayers {
            for l in 0..<t.num_hidden_layers {
                let want = (l + 1) % resolvedFAInterval == 0
                try inv(isGQAKind(normalizedLayerTypes[l]) == want, "layer \(l) cadence")
            }
        }
        let hasLinearLayers = normalizedLayerTypes.contains("linear_attention")
        let linCfg = t.linear_attn_config
        let rawLinKV = t.num_kv_heads_for_linear_attn
        let defaultLinQK = ((rawLinKV ?? 0) > 0 ? rawLinKV! : t.num_attention_heads)
        let qkHeads = hasLinearLayers ? (t.linear_num_key_heads ?? linCfg?.num_heads ?? defaultLinQK) : 0
        let vHeads = hasLinearLayers ? (t.linear_num_value_heads ?? linCfg?.num_heads ?? t.num_attention_heads) : 0
        let gHeadDim = hasLinearLayers ? (t.linear_key_head_dim ?? linCfg?.head_dim ?? t.head_dim ?? 0) : 0
        let gConv = hasLinearLayers ? (t.linear_conv_kernel_dim ?? linCfg?.short_conv_kernel_size ?? t.short_conv_kernel_size ?? 0) : 0
        if hasLinearLayers {
            try inv(qkHeads > 0 && vHeads > 0 && gHeadDim > 0, "GDN heads/dim missing")
            try inv((t.linear_value_head_dim ?? gHeadDim) == gHeadDim, "GDN v-head dim != qk head dim")
        }
        let qkv = (2 * qkHeads + vHeads) * gHeadDim
        let z = vHeads * gHeadDim

        let linType = (linCfg?.Linear_type ?? linCfg?.linear_type ?? "").lowercased()
        let noKDALora = (t.no_kda_lora == true) || (t.use_kda_lora == false)
        let isKDA = hasLinearLayers && (linType == "kda" || linCfg?.gate_lower_bound != nil
            || t.kda_safe_gate != nil || t.kda_lower_bound != nil || t.no_kda_lora != nil || t.use_kda_lora != nil)
        let gdnLowRank = noKDALora ? nil : (linCfg?.low_rank_dim ?? (isKDA ? gHeadDim : nil))
        let kdaLower = linCfg?.gate_lower_bound ?? t.kda_lower_bound
        let parsedKDAConfig: KDAConfig? = isKDA ? KDAConfig(
            qkHeads: qkHeads, vHeads: vHeads, headDim: gHeadDim, convKernel: gConv,
            gateProjection: noKDALora ? .fullRank : .lowRank(dim: gdnLowRank ?? gHeadDim),
            safeGate: t.kda_safe_gate ?? false,
            gateLowerBound: kdaLower,
            outputNormAct: .sigmoid,
            separateConv: true
        ) : nil

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
            let heads = (t.ngram_size ?? 0) > 0 ? ((t.ngram_size ?? 0) - 1) * (t.heads_per_ngram ?? 0) : -1
            try inv(heads > 0 && (t.ple_embed_dim ?? 0) % heads == 0,
                    "ple_embed_dim \(t.ple_embed_dim ?? -1) % heads \(heads) != 0")
            for id in pleIds {
                try inv(id >= 1 && id <= t.num_hidden_layers, "ple_layer_ids one-indexed violation: \(id)")
                try inv(normalizedLayerTypes[id - 1] == "linear_attention",
                        "PLE id \(id) must land on linear_attention (got \(normalizedLayerTypes[id-1]))")
            }
        }
        let eos: Int
        let eosList: [Int]
        if let eosVal = t.eos_token_id ?? env.eos_token_id ?? env.generation_config?.eos_token_id {
            eos = eosVal.value
            eosList = eosVal.values
        } else if familySpec.allowsOptionalEosTokenId {
            let fallback = t.pad_token_id ?? env.pad_token_id ?? t.bos_token_id ?? env.bos_token_id ?? 0
            eos = fallback
            eosList = [fallback]
        } else {
            throw DescriptorError.invariant("flat config missing 'eos_token_id'")
        }

        let resolvedSharedIntermediate: Int?
        if let explicitShared = t.shared_expert_intermediate_size {
            resolvedSharedIntermediate = explicitShared
        } else if let moeShared = t.moe_shared_expert_intermediate_size {
            resolvedSharedIntermediate = moeShared * max(1, t.num_shared_experts ?? t.n_shared_experts ?? 1)
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
        let rawMHCVal = t.mhc ?? env.mhc ?? (t.enable_hyper_connections == true ? .flag(true) : nil)
        let parsedMHC: MHCConfig?
        switch rawMHCVal {
        case .config(let rm):
            if let s = rm.start, let e = rm.end, s >= 0, e > s, e <= t.num_hidden_layers {
                parsedMHC = MHCConfig(start: s, end: e, streams: max(2, rm.streams ?? 2),
                                      iterations: max(1, rm.iterations ?? 20),
                                      fixedIdentity: rm.fixed_identity ?? false, preScale: rm.pre_scale ?? 1.0,
                                      hcEps: t.hc_eps, hasExitHead: t.enable_hyper_connections == true)
            } else {
                parsedMHC = nil
            }
        case .flag(true):
            parsedMHC = MHCConfig(start: 0, end: t.num_hidden_layers,
                                  streams: max(2, t.num_residual_streams ?? t.hc_mult ?? 4),
                                  iterations: max(1, t.mhc_sinkhorn_iterations ?? t.hc_sinkhorn_iters ?? 20),
                                  fixedIdentity: false, preScale: 1.0,
                                  hcEps: t.hc_eps ?? (familySpec.defaultMHCEpsFromRMSNorm ? Float(resolvedNormEps) : nil),
                                  hasExitHead: t.enable_hyper_connections == true)
        case .flag(false), nil:
            parsedMHC = nil
        }
        if t.hc_count != nil || parsedMHC != nil { caps.insert(.hyperConnections) }
        let mtpCfgLayers = dictInt(rawRoot?["mtp_config"] as? [String: Any], "num_hidden_layers")
        let resolvedMtpLayers = max(0, t.mtp_num_hidden_layers ?? t.num_nextn_predict_layers ?? t.mtp_num_layers ?? mtpCfgLayers ?? 0)
        if resolvedMtpLayers > 0 { caps.insert(.mtp) }

        let (resolvedRopeScaling, parsedYarn) = try parseRoPEScaling(text: t, env: env)

        let parsedHCA: HCAConfig? = try {
            guard normalizedLayerTypes.contains("compressed_attention"),
                  let ratios = t.compress_ratios else { return nil }
            guard let defRatio = ratios.first(where: { $0 > 0 }) else {
                throw DescriptorError.invariant("compressed_attention missing positive compress_ratios entry")
            }
            guard let qLoraRank = t.q_lora_rank else {
                throw DescriptorError.invariant("HCA missing q_lora_rank")
            }
            guard let kvLoraRank = t.kv_lora_rank else {
                throw DescriptorError.invariant("HCA missing kv_lora_rank")
            }
            guard let qkNope = t.qk_nope_head_dim ?? (rawText["compress_qk_nope_head_dim"] as? Int) else {
                throw DescriptorError.invariant("HCA missing qk_nope_head_dim")
            }
            guard let qkRope = t.qk_rope_head_dim ?? (rawText["compress_qk_rope_head_dim"] as? Int) else {
                throw DescriptorError.invariant("HCA missing qk_rope_head_dim")
            }
            guard let slidingWindow = t.sliding_window else {
                throw DescriptorError.invariant("HCA missing sliding_window")
            }
            let hcaVDim = t.v_head_dim ?? (rawText["compress_v_head_dim"] as? Int) ?? resolvedHeadDim
            // Documented test fallback: minimal HCA unit fixtures omit `o_groups`/`o_lora_rank`.
            return HCAConfig(
                compressRatios: ratios,
                defaultCompressRatio: defRatio,
                qHeads: t.num_attention_heads,
                kvHeads: resolvedKVHeads,
                headDim: resolvedHeadDim,
                qLoraRank: qLoraRank,
                kvLoraRank: kvLoraRank,
                qkNopeHeadDim: qkNope,
                qkRopeHeadDim: qkRope,
                vHeadDim: hcaVDim,
                oGroups: t.o_groups ?? 8,
                oLoraRank: t.o_lora_rank ?? 1024,
                slidingWindow: slidingWindow,
                compressRopeTheta: t.compress_rope_theta ?? ropeTheta,
                yarn: (t.hca_rope_type == nil || t.hca_rope_type?.lowercased() == "yarn") ? parsedYarn : nil,
                useAttnMerger: t.use_attn_merger ?? true,
                useVHA: t.use_vha_attention ?? false,
                hasAttnSinkBias: t.add_full_attention_sink_bias ?? false,
                gatedAttnUseQLora: t.gated_attn_use_q_lora ?? false
            )
        }()

        let resolvedRopeInterleave = t.rope_interleave ?? t.indexer_rope_interleave ?? familySpec.defaultRopeInterleave
        let parsedHybridMLA: HybridMLAConfig? = try {
            guard hasHybridMLA else { return nil }
            guard let qLora = t.hybrid_mla_q_lora_rank ?? t.q_lora_rank else {
                throw DescriptorError.invariant("HybridMLA missing q_lora_rank")
            }
            guard let kvLora = t.hybrid_mla_kv_lora_rank ?? t.kv_lora_rank else {
                throw DescriptorError.invariant("HybridMLA missing kv_lora_rank")
            }
            guard let nope = t.hybrid_mla_qk_nope_head_dim ?? t.qk_nope_head_dim else {
                throw DescriptorError.invariant("HybridMLA missing qk_nope_head_dim")
            }
            guard let rope = t.hybrid_mla_qk_rope_head_dim ?? t.qk_rope_head_dim else {
                throw DescriptorError.invariant("HybridMLA missing qk_rope_head_dim")
            }
            guard let vDim = t.hybrid_mla_v_head_dim ?? t.v_head_dim else {
                throw DescriptorError.invariant("HybridMLA missing v_head_dim")
            }
            let mlaYarn: YarnRopeConfig? = (t.rope_type?.lowercased() == "yarn") ? parsedYarn : nil
            return HybridMLAConfig(
                qHeads: t.hybrid_mla_num_attention_heads ?? t.num_attention_heads,
                qLoraRank: qLora,
                kvLoraRank: kvLora,
                qkNopeHeadDim: nope,
                qkRopeHeadDim: rope,
                vHeadDim: vDim,
                decoupledRope: !isNoPEMLA && rope > 0,
                ropeInterleave: resolvedRopeInterleave,
                slidingWindow: t.sliding_window ?? 0,
                hasAttnSinkBias: t.add_full_attention_sink_bias ?? false,
                useVHA: t.use_vha_attention ?? false,
                indexerUseFastHadamard: t.use_fast_hadamard ?? false,
                splitKvBProj: t.mqa_split_kv_b_proj ?? false,
                yarn: mlaYarn
            )
        }()

        let mropeLayout: MRoPELayout = hasMrope
            ? (familySpec.defaultBailingVideoRope ? .bailingVideoRope : .qwenInterleaved) : .none1D
        let mlaSparsity: MLAConfig.Sparsity = (resolvedIdxHeads != nil)
            ? ((t.index_kpool_compress == true) ? .keyPool : .dsa(slidingWindow: t.sliding_window ?? 0))
            : .dense(slidingWindow: t.sliding_window ?? 0)
        let denseAttnLayerSet = try resolveDenseAttentionLayerSet(text: t, inv: inv)
        let granularity = (t.gated_attention_proj_granularity_type ?? "").lowercased()
        let gateDeclared = (t.output_gate_type != nil) || (granularity == "head_wise" || granularity == "element_wise")
        let attnGate: Bool = t.attn_output_gate ?? t.use_gated_attn ?? t.gated_attention ?? (gateDeclared || familySpec.defaultAttnOutputGate)
        let parsedUnifiedMLA: MLAConfig? = try {
            if let h = parsedHybridMLA {
                let qProj: MLAConfig.QProjection = h.qLoraRank > 0 ? .factored(rank: h.qLoraRank) : .direct
                let outMode: MLAConfig.OutputGateMode = attnGate
                    ? (qProj == .direct ? .headWiseFromHidden : .elementWiseFromQLora) : .none
                let mlaTraits = try resolveMLAScalingAndNormTraits(
                    text: t, familySpec: familySpec, qLoraRank: h.qLoraRank, kvLoraRank: h.kvLoraRank, inv: inv
                )
                return MLAConfig(
                    qHeads: h.qHeads, qProjection: qProj, kvLoraRank: h.kvLoraRank,
                    qkNopeHeadDim: h.qkNopeHeadDim, qkRopeHeadDim: h.qkRopeHeadDim, vHeadDim: h.vHeadDim,
                    rope: h.decoupledRope ? MLARopeConfig(dim: h.qkRopeHeadDim, theta: ropeTheta, interleave: h.ropeInterleave, mropeLayout: mropeLayout, yarn: h.yarn) : nil,
                    sparsity: mlaSparsity, outputGateMode: outMode,
                    useVHA: h.useVHA, hasAttnSinkBias: h.hasAttnSinkBias, splitKvBProj: h.splitKvBProj,
                    indexerUseFastHadamard: h.indexerUseFastHadamard,
                    alphaQ: mlaTraits.alphaQ, alphaKV: mlaTraits.alphaKV,
                    indexerKNormIsRMS: mlaTraits.indexerKNormIsRMS, denseAttentionLayers: denseAttnLayerSet
                )
            }
            if isNoPEMLA, let kvLora = t.kv_lora_rank, kvLora > 0 {
                let qRank = t.q_lora_rank ?? 0
                let qProj: MLAConfig.QProjection = qRank > 0 ? .factored(rank: qRank) : .direct
                let mlaTraits = try resolveMLAScalingAndNormTraits(
                    text: t, familySpec: familySpec, qLoraRank: qRank, kvLoraRank: kvLora, inv: inv
                )
                return MLAConfig(
                    qHeads: t.num_attention_heads, qProjection: qProj, kvLoraRank: kvLora,
                    qkNopeHeadDim: t.qk_nope_head_dim ?? resolvedHeadDim, qkRopeHeadDim: 0,
                    vHeadDim: t.v_head_dim ?? resolvedHeadDim, rope: nil, sparsity: mlaSparsity,
                    outputGateMode: attnGate ? .elementWiseFromHidden : .none,
                    alphaQ: mlaTraits.alphaQ, alphaKV: mlaTraits.alphaKV,
                    indexerKNormIsRMS: mlaTraits.indexerKNormIsRMS, denseAttentionLayers: denseAttnLayerSet
                )
            }
            guard hasUnprefixedDecoupledMLA, let kvLora = t.kv_lora_rank, let nope = t.qk_nope_head_dim else { return nil }
            let rope = t.qk_rope_head_dim ?? 0
            let vDim = t.v_head_dim ?? resolvedHeadDim
            try inv(kvLora > 0 && nope > 0 && vDim > 0, "MLA dimensions must be positive")
            let qRank = t.q_lora_rank ?? 0
            let qProj: MLAConfig.QProjection = qRank > 0 ? .factored(rank: qRank) : .direct
            let outMode: MLAConfig.OutputGateMode = granularity == "head_wise" ? .headWiseFromHidden
                : ((granularity == "element_wise" || t.gated_attention == true) ? .elementWiseFromHidden
                    : (attnGate ? (qProj == .direct ? .elementWiseFromHidden : .elementWiseFromQLora) : .none))
            let mlaTraits = try resolveMLAScalingAndNormTraits(
                text: t, familySpec: familySpec, qLoraRank: qRank, kvLoraRank: kvLora, inv: inv
            )
            return MLAConfig(
                qHeads: t.num_attention_heads, qProjection: qProj, kvLoraRank: kvLora,
                qkNopeHeadDim: nope, qkRopeHeadDim: rope, vHeadDim: vDim,
                rope: rope > 0 ? MLARopeConfig(dim: rope, theta: ropeTheta, interleave: resolvedRopeInterleave, mropeLayout: mropeLayout, yarn: parsedYarn) : nil,
                sparsity: mlaSparsity, outputGateMode: outMode,
                useVHA: t.use_vha_attention ?? false, hasAttnSinkBias: t.add_full_attention_sink_bias ?? false,
                splitKvBProj: t.mqa_split_kv_b_proj ?? false, indexerUseFastHadamard: t.use_fast_hadamard ?? false,
                alphaQ: mlaTraits.alphaQ, alphaKV: mlaTraits.alphaKV,
                indexerKNormIsRMS: mlaTraits.indexerKNormIsRMS, denseAttentionLayers: denseAttnLayerSet
            )
        }()

        let allowsGelu = familySpec.defaultRole == .decisionEncoder
            || rawText["decision_head"] != nil
            || rawRoot?["decision_head"] != nil
        let parsedAct: ActivationKind
        if let rawAct = t.hidden_act {
            let act = rawAct.lowercased()
            if act == "situ" {
                parsedAct = .situ(beta: t.activation_situ_beta ?? 4.0,
                                  linearBeta: t.activation_situ_linear_beta ?? 25.0)
            } else if act == "silu" {
                parsedAct = .silu
            } else if allowsGelu && (act == "gelu" || act == "gelu_erf") {
                parsedAct = .gelu
            } else if allowsGelu && (act == "gelu_new" || act == "gelu_pytorch_tanh" || act == "gelu_fast" || act == "quick_gelu") {
                parsedAct = .geluTanh
            } else {
                throw DescriptorError.invariant("unsupported hidden_act '\(rawAct)'")
            }
        } else {
            parsedAct = .silu
        }

        let rawLoop = t.loop_settings ?? env.loop_settings
        let hasLoopConfig = rawText["loop_config"] != nil
        let parsedLoop: LoopSettings? = rawLoop.flatMap { rl in
            guard let s = rl.start, let e = rl.end, s >= 0, e > s, e <= t.num_hidden_layers else { return nil }
            return LoopSettings(start: s, end: e, evalCount: max(1, rl.eval_count ?? rl.counts?.first ?? 1), counts: rl.counts)
        } ?? (hasLoopConfig ? inkling2LoopSettings(rawText: rawText, numLayers: t.num_hidden_layers) : nil)
        let parsedInk2 = (hasLoopConfig || rawText["d_rel"] != nil)
            ? try decodeInkling2Config(text: t, env: env, rawRoot: rawRoot, rawText: rawText, inv: inv)
            : nil

        let resolvedSlidingWindow: Int?
        let resolvedLayerSlidingWindows: [Int]?
        let effectiveSW = t.sliding_window ?? dictInt(rawText, "sliding_window_size") ?? t.local_attention
        if !normalizedLayerTypes.contains("compressed_attention"), !hasHybridMLA {
            if let sw = effectiveSW { try inv(sw >= 0, "sliding_window \(sw) < 0") }
            if let mwl = t.max_window_layers { try inv(mwl >= 0, "max_window_layers \(mwl) < 0") }
            if let pat = t.sliding_window_pattern { try inv(pat >= 1, "sliding_window_pattern \(pat) < 1") }
            if normalizedLayerTypes.contains("sliding_attention") {
                guard let sw = effectiveSW, sw > 0 else {
                    throw DescriptorError.invariant("layer_types 'sliding_attention' requires positive sliding_window")
                }
                let effective = sw < resolvedMaxPos ? sw : 0
                resolvedSlidingWindow = nil
                resolvedLayerSlidingWindows = normalizedLayerTypes.map { $0 == "sliding_attention" ? effective : 0 }
            } else if (t.use_sliding_window ?? true),
                      let sw = effectiveSW, sw > 0, sw < resolvedMaxPos {
                if let mwl = t.max_window_layers {
                    resolvedSlidingWindow = nil
                    resolvedLayerSlidingWindows = (0..<t.num_hidden_layers).map { $0 < mwl ? 0 : sw }
                } else if let pat = t.sliding_window_pattern {
                    resolvedSlidingWindow = nil
                    resolvedLayerSlidingWindows = pat == 1 ? nil
                        : (0..<t.num_hidden_layers).map { ($0 + 1) % pat == 0 ? 0 : sw }
                } else {
                    resolvedSlidingWindow = sw
                    resolvedLayerSlidingWindows = nil
                }
            } else {
                resolvedSlidingWindow = nil
                resolvedLayerSlidingWindows = nil
            }
        } else {
            resolvedSlidingWindow = nil
            resolvedLayerSlidingWindows = nil
        }

        let parsedBlockSparseGQA = try decodeBlockSparseGQAConfig(text: t, inv: inv)
        let parsedGatedNorm = try decodeGatedNormConfig(text: t, inv: inv)
        let parsedGDNGating = try decodeGDNGatingConfig(
            text: t, hasLinearLayers: hasLinearLayers, resolvedNormEps: resolvedNormEps, inv: inv
        )
        let parsedSoftcap = try decodeFinalLogitSoftcapping(text: t, inkling2: parsedInk2, inv: inv)

        let resolvedGateType = t.scoring_func ?? t.router_score_function ?? t.score_function ?? familySpec.defaultExpertGateType
        let vd = try resolveVisionDims(
            vc: env.vision_config,
            rawVision: rawRoot?["vision_config"] as? [String: Any],
            hidden: t.hidden_size,
            isInkling2: parsedInk2?.vision != nil
        )
        let hasVis = env.vision_config != nil
        let disableMergerProj = (env.vision_config?.disable_merger_proj == true)
        let resolvedVisOut = (hasVis && disableMergerProj) ? t.hidden_size : vd.out
        let resolvedSwiGLU = try Self.decodeSwiGLUClampConfig(
            text: t, routedLimits: t.expert_swiglu_limit_list,
            sharedLimits: t.share_expert_swiglu_limit_list, inv: inv
        )
        let resolvedSharedGate = t.moe_shared_expert_gate ?? t.use_shared_expert_gate ?? t.use_shared_expert_sigmoid
            ?? familySpec.defaultSharedExpertHasGate ?? (t.n_shared_experts == nil && t.moe_shared_expert_intermediate_size == nil)
        let resolvedRouteScale = t.routed_scaling_factor ?? dictFloat(rawText, "route_scale") ?? 1.0
        let resolvedNormTopK = t.norm_topk_prob ?? (resolvedNumExperts != nil ? familySpec.defaultNormTopKProb : false)
        let resolvedEnableExpertBias = t.moe_router_enable_expert_bias
            ?? (dictBool(rawText, "use_gate_bias") ?? familySpec.moeRouterEnableExpertBias)
        let resolvedNGroup = max(1, t.n_group ?? 1)
        let resolvedTopkGroup = min(resolvedNGroup, max(1, t.topk_group ?? resolvedNGroup))
        let resolvedGroupTopM = (resolvedGateType == "sigmoid" || t.topk_method == "noaux_tc" || resolvedNGroup > 1) ? 2 : 1
        let parsedMoERouting: MoERoutingConfig? = resolvedNumExperts != nil ? MoERoutingConfig(
            nGroup: resolvedNGroup, topkGroup: resolvedTopkGroup, groupTopM: resolvedGroupTopM,
            sharedExpertHasGate: resolvedSharedGate, scoreFunction: resolvedGateType ?? "softmax",
            enableExpertBias: t.topk_method == "noaux_tc" || resolvedEnableExpertBias,
            routedScalingFactor: resolvedRouteScale,
            routedScalingFactorLearnable: t.routed_scaling_factor_learnable ?? false,
            normTopKProb: resolvedNormTopK
        ) : nil
        let resolvedDenseInter = parsedInk2 != nil ? nil : resolveAdjustedIntermediate(text: t)
        let mtpDedicatedEmb = t.mtp_use_dedicated_embeddings ?? false
        let mtpIsMoE = (resolvedNumExperts != nil) && !familySpec.defaultMTPIsDense
        let mtpHasIndexer = parsedUnifiedMLA?.hasIndexer ?? (resolvedIdxHeads != nil)
        let parsedMTP: MTPConfig? = resolvedMtpLayers > 0 ? MTPConfig(
            layers: resolvedMtpLayers,
            fusionMode: (t.mtp_shared_last_layer == true) ? .sharedLastLayerEcho
                : ((parsedMHC != nil && familySpec.defaultMTPSplitEProjHProj) ? .splitEProjHProj : .concatEHProj),
            useDedicatedEmbeddings: mtpDedicatedEmb,
            hasFinalNorm: t.num_nextn_predict_layers != nil && !familySpec.defaultMTPSplitEProjHProj,
            isMoE: mtpIsMoE,
            intermediateSize: mtpIsMoE ? t.moe_intermediate_size : resolvedDenseInter,
            attentionIsSparse: t.nextn_is_sparse ?? familySpec.defaultMTPAttentionIsSparse ?? mtpHasIndexer,
            attentionIsLinear: t.nextn_is_linear ?? false,
            sharedEmbedAndHead: !mtpDedicatedEmb
        ) : nil
        let parsedDSpark = try decodeDSparkConfig(
            text: t, rawRoot: rawRoot, rawText: rawText,
            ropeTheta: ropeTheta, vocab: t.vocab_size, numLayers: t.num_hidden_layers, inv: inv
        )
        let visType = env.vision_config?.model_type ?? ""
        let patchEmbedKind: VisionTowerConfig.PatchEmbedKind = switch visType {
        case "lfm3_vl_vision", "lfm2_vl_vision": .lfmHybridConv
        case "siglip2_vision_model", "siglip_vision_model": .siglip2NaFlex
        default: parsedInk2?.vision != nil ? .unfoldedPatchProj : .conv3d
        }
        let visNormEps = Float(
            env.vision_config?.norm_eps
                ?? env.vision_config?.layer_norm_eps
                ?? env.vision_config?.rms_norm_eps
                ?? parsedInk2?.vision?.layerNormEps
                ?? 1e-6
        )
        let visRopeTheta = env.vision_config?.rope_theta
            ?? env.vision_config?.rope_parameters?.rope_theta
            ?? 10_000.0
        let projNormEps = Float(
            env.projector_layer_norm_eps
                ?? env.vision_config?.norm_eps
                ?? env.vision_config?.layer_norm_eps
                ?? parsedInk2?.vision?.layerNormEps
                ?? 1e-6
        )
        let parsedVisionTower: VisionTowerConfig? = hasVis ? VisionTowerConfig(
            depth: vd.depth, hidden: vd.hidden, heads: vd.heads, intermediate: vd.inter,
            patch: vd.patch, temporal: vd.temporal, merge: vd.merge, numPosEmbeds: vd.pos,
            outDim: resolvedVisOut,
            mergerIntermediateDim: disableMergerProj ? t.hidden_size : (env.projector_hidden_size ?? parsedInk2?.vision?.projectionHiddenDim),
            projectorBias: disableMergerProj ? true : (env.projector_bias ?? parsedInk2?.vision?.projectorBias ?? true),
            normEps: visNormEps,
            ropeTheta: visRopeTheta,
            mropeLayout: mropeLayout,
            patchEmbedKind: patchEmbedKind,
            imageMean: familySpec.defaultVisionMean,
            imageStd: familySpec.defaultVisionStd,
            disableInternalMergerProj: disableMergerProj,
            kvHeads: env.vision_config?.num_key_value_heads,
            layerTypes: env.vision_config?.layer_types,
            convKernelSize: env.vision_config?.conv_kernel_size,
            projectorNormEps: projNormEps
        ) : nil
        let tieEmb = t.tie_word_embeddings ?? env.tie_word_embeddings ?? false
        let lfmTraits = decodeLFMAndDecisionTraits(
            text: t, env: env, familySpec: familySpec,
            normalizedLayerTypes: normalizedLayerTypes,
            resolvedNormEps: resolvedNormEps,
            resolvedMaxPos: resolvedMaxPos,
            tieWordEmbeddings: tieEmb,
            attnGate: attnGate,
            granularity: granularity
        )
        var resolvedCaps = caps
        if lfmTraits.decisionConfig != nil { resolvedCaps.insert(.decision) }
        let resolvedRole: QWModelRole = familySpec.defaultRole == .decisionEncoder
            ? .decisionEncoder
            : (resolvedCaps.contains(.vision) ? .multimodalConditionalLM : .causalLM)
        let descriptor = QWModelDescriptor(
            family: family, modelType: resolvedModelType, capabilities: resolvedCaps,
            hidden: t.hidden_size, vocab: t.vocab_size,
            denseIntermediate: resolvedDenseInter,
            numLayers: t.num_hidden_layers, layerTypes: normalizedLayerTypes, fullAttentionInterval: resolvedFAInterval,
            rmsEps: Float(resolvedNormEps), maxPositionEmbeddings: resolvedMaxPos,
            qHeads: t.num_attention_heads, kvHeads: resolvedKVHeads, headDim: resolvedHeadDim,
            attnOutputGate: attnGate, partialRotaryFactor: resolvedPartialRotary, ropeTheta: ropeTheta,
            indexerHeads: resolvedIdxHeads, indexerKvHeads: resolvedIdxKvHeads,
            indexerHeadDim: resolvedIdxHeadDim, indexerBudget: resolvedIdxBudget,
            indexerCompressRatio: resolvedIdxCompress,
            numExperts: resolvedNumExperts, expertsPerTok: t.num_experts_per_tok,
            moeIntermediate: t.moe_intermediate_size ?? (parsedInk2 != nil && resolvedNumExperts != nil ? t.intermediate_size : nil),
            sharedExpertIntermediate: resolvedSharedIntermediate, expertGateType: resolvedGateType,
            hcCount: t.hc_count, hcLowrank: t.hc_lowrank,
            gdnQKHeads: qkHeads, gdnVHeads: vHeads, gdnHeadDim: gHeadDim,
            gdnQKVChannels: qkv, gdnZChannels: z, gdnConvKernel: gConv,
            pleLayerIds: t.ple_layer_ids, pleEmbedDim: t.ple_embed_dim, pleConvKernel: t.ple_conv_kernel_size,
            ngramSize: t.ngram_size, headsPerNgram: t.heads_per_ngram, ngramVocabSizeBase: t.ngram_vocab_size_base,
            makeNgramVocabDivisibleBy: t.make_ngram_vocab_size_divisible_by, splitNgramParts: t.split_ngram_parts,
            mtpLayers: resolvedMtpLayers, mtpUseDedicatedEmbeddings: t.mtp_use_dedicated_embeddings ?? false,
            eosTokenId: eos, mropeSection: mropeSec,
            visDepth: vd.depth, visHidden: vd.hidden, visHeads: vd.heads, visIntermediate: vd.inter,
            visPatch: vd.patch, visTemporal: vd.temporal, visMerge: vd.merge, visPosEmbeds: vd.pos, visOut: resolvedVisOut,
            imageTokenId: env.image_token_id ?? (hasVis ? (parsedInk2 != nil ? 200054 : 248056) : -1),
            videoTokenId: env.video_token_id ?? (hasVis ? (parsedInk2 != nil ? -1 : 248057) : -1),
            visionStartTokenId: env.vision_start_token_id ?? (hasVis ? 248053 : -1),
            visionEndTokenId: env.vision_end_token_id ?? (hasVis ? 248054 : -1),
            hasQKNorm: t.use_qk_norm ?? t.qk_layernorm ?? t.qk_norm ?? familySpec.defaultHasQKNorm,
            hasQKVBias: (t.qkv_bias == true) || (t.attention_bias == true),
            hasAttnOutputBias: (t.attention_bias == true),
            usesPlusOneRMSNorm: familySpec.usesPlusOneRMSNorm,
            quantization: t.quantization ?? t.quantization_config ?? env.quantization ?? env.quantization_config,
            hasMropeSection: hasMrope,
            tieWordEmbeddings: tieEmb,
            normTopKProb: resolvedNormTopK,
            eosTokenIds: eosList,
            envelopeModelType: env.model_type,
            loopSettings: parsedLoop,
            mhcConfig: parsedMHC,
            firstKDenseReplace: moeConfig.firstKDenseReplace,
            mlpOnlyLayers: moeConfig.mlpOnlyLayers,
            decoderSparseStep: moeConfig.decoderSparseStep,
            moeLayerFreq: moeConfig.moeLayerFreq,
            routedScalingFactor: resolvedRouteScale,
            swigluLimit: resolvedSwiGLU?.denseLimit,
            topkMethod: t.topk_method,
            sharedExpertHasGate: resolvedSharedGate,
            mlaUseNoPE: isNoPEMLA,
            kvLoraRank: parsedHybridMLA?.kvLoraRank ?? t.kv_lora_rank,
            qLoraRank: parsedHybridMLA?.qLoraRank ?? t.q_lora_rank,
            qkNopeHeadDim: parsedHybridMLA?.qkNopeHeadDim ?? t.qk_nope_head_dim,
            qkRopeHeadDim: parsedHybridMLA?.qkRopeHeadDim ?? t.qk_rope_head_dim,
            vHeadDim: parsedHybridMLA?.vHeadDim ?? t.v_head_dim,
            indexKPoolCompress: t.index_kpool_compress ?? false,
            indexKPoolAlwaysSelectTail: t.index_kpool_always_select_tail ?? false,
            indexShareForMTP: t.index_share_for_mtp_iteration ?? false,
            indexerRopeInterleave: resolvedRopeInterleave,
            gdnVectorDecay: isKDA,
            gdnLowRankDim: gdnLowRank,
            gdnGateLowerBound: kdaLower,
            gdnSeparateConv: isKDA,
            hcaConfig: parsedHCA,
            hybridMLAConfig: parsedHybridMLA,
            blockSparseGQAConfig: parsedBlockSparseGQA,
            yarnRopeConfig: (!hasHybridMLA && parsedHCA == nil) ? parsedYarn : nil,
            ropeScaling: resolvedRopeScaling,
            slidingWindow: resolvedSlidingWindow,
            layerSlidingWindows: resolvedLayerSlidingWindows,
            activationKind: parsedAct,
            moeLatentSize: t.moe_latent_size,
            latentMoeUseNorm: t.latent_moe_use_norm ?? false,
            moeSplitFeatureRouting: t.moe_split_feature_routing ?? false,
            moeRouterEnableExpertBias: resolvedEnableExpertBias,
            routedScalingFactorLearnable: t.routed_scaling_factor_learnable ?? false,
            multimaxHead: (t.multimax?.lowercased() == "lm_head"),
            mtpSharedLastLayer: t.mtp_shared_last_layer ?? false,
            inkling2Config: parsedInk2,
            dsparkConfig: parsedDSpark,
            mlaConfig: parsedUnifiedMLA,
            kdaConfig: parsedKDAConfig,
            moeRoutingConfig: parsedMoERouting,
            swigluClampConfig: resolvedSwiGLU,
            mtpConfig: parsedMTP,
            visionTowerConfig: parsedVisionTower,
            ropeInterleave: resolvedRopeInterleave,
            role: resolvedRole,
            shortConvConfig: lfmTraits.shortConv,
            normKind: lfmTraits.normKind,
            useSandwichNorm: lfmTraits.useSandwichNorm,
            perKindRoPEConfig: lfmTraits.perKindRoPE,
            attnGateMode: lfmTraits.attnGateMode,
            decisionConfig: lfmTraits.decisionConfig,
            gatedNormConfig: parsedGatedNorm,
            gdnGatingConfig: parsedGDNGating,
            finalLogitSoftcapping: parsedSoftcap,
            v2DigestHeadDim: resolvedV2DigestHeadDim)
        let issues = descriptor.verifyInternalConsistency()
        if !issues.isEmpty {
            throw DescriptorError.invariant(issues.joined(separator: "; "))
        }
        return descriptor
    }
}
