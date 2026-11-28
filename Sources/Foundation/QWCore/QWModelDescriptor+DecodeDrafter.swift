// QWModelDescriptor+DecodeDrafter.swift — Drafter and standard HF config decoders.
import Foundation

extension QWModelDescriptor {
    /// Decodes a DFlash2 or DSpine speculative drafter `config.json` with `.speculativeDrafter` role.
    public static func decodeDrafter(configJSON data: Data) throws -> QWModelDescriptor {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw DescriptorError.invariant("drafter config JSON is not an object")
        }
        guard isDrafterJSON(root) else {
            throw DescriptorError.invariant("config JSON is not a speculative drafter model")
        }
        let tc = (root["text_config"] as? [String: Any]) ?? root
        let modelType = (tc["model_type"] as? String) ?? (root["model_type"] as? String) ?? ""
        guard let spec = QWModelFamilyRegistry.shared.spec(forModelType: modelType) else {
            throw QWModelDescriptorError.unsupportedModelType(modelType)
        }
        guard let hidden = tc["hidden_size"] as? Int, hidden > 0,
              let vocab = tc["vocab_size"] as? Int, vocab > 0,
              let numLayers = tc["num_hidden_layers"] as? Int, numLayers > 0,
              let qHeads = tc["num_attention_heads"] as? Int, qHeads > 0 else {
            throw DescriptorError.invariant("drafter config missing core dimensions")
        }
        let kvHeads = max(1, (tc["num_key_value_heads"] as? Int) ?? qHeads)
        let decoupledDim: Int? = {
            guard let nope = tc["qk_nope_head_dim"] as? Int,
                  let rope = tc["qk_rope_head_dim"] as? Int,
                  nope + rope > 0 else { return nil }
            return nope + rope
        }()
        let rawHeadDim = tc["head_dim"] as? Int
        let headDim: Int
        if let decoupledDim, rawHeadDim == nil || rawHeadDim == (tc["qk_rope_head_dim"] as? Int) {
            headDim = decoupledDim
        } else {
            headDim = max(1, rawHeadDim ?? (tc["qk_head_dim"] as? Int) ?? decoupledDim ?? (hidden / qHeads))
        }
        let intermediate = (tc["intermediate_size"] as? Int) ?? (hidden * 4)
        let rmsEps = Float((tc["rms_norm_eps"] as? Double) ?? 1e-6)
        let maxPos = (tc["max_position_embeddings"] as? Int) ?? 32768
        let ropeDict = (tc["rope_parameters"] as? [String: Any]) ?? (tc["rope_scaling"] as? [String: Any])
        let ropeTheta = (tc["rope_theta"] as? Double) ?? (ropeDict?["rope_theta"] as? Double) ?? 10000.0
        let partialRotary = (tc["partial_rotary_factor"] as? Double) ?? (ropeDict?["partial_rotary_factor"] as? Double) ?? 1.0
        let eosScalar = (tc["eos_token_id"] as? Int) ?? (tc["eos_token_id"] as? [Int])?.first ?? 0
        let eosList = (tc["eos_token_id"] as? [Int]) ?? [eosScalar]
        let layerTypes = (tc["layer_types"] as? [String]).flatMap { $0.count == numLayers ? $0 : nil }
            ?? [String](repeating: "full_attention", count: numLayers)
        let dcfg = root["dflash_config"] as? [String: Any]
        let gatedNormConfig: GatedNormConfig? = {
            let w = ((tc["layernorm_gating_weight"] as? NSNumber) ?? (root["layernorm_gating_weight"] as? NSNumber))?.floatValue ?? 2.0
            switch (tc["norm_type"] as? String) ?? (root["norm_type"] as? String) {
            case "GatedNorm": return GatedNormConfig(rank: 16, gatingWeight: w, zeroCentered: false)
            case "ZeroCenteredGatedNorm": return GatedNormConfig(rank: 16, gatingWeight: w, zeroCentered: true)
            default: return nil
            }
        }()
        let layernormType = (tc["layernorm_type"] as? String) ?? (root["layernorm_type"] as? String)
        let useSandwichNorm = ((tc["use_sandwich_norm"] as? Bool) == true) || (layernormType == "pre_post")
        let softcapNum = (tc["final_logit_softcapping"] as? NSNumber)
            ?? (dcfg?["final_logit_softcapping"] as? NSNumber)
            ?? (root["final_logit_softcapping"] as? NSNumber)
        let finalLogitSoftcapping: Float? = softcapNum.flatMap { n in
            let v = n.floatValue
            return (v > 0 && v.isFinite) ? v : nil
        }
        let ropeInterleave = (tc["rope_interleave"] as? Bool) ?? spec.defaultRopeInterleave
        let yarnCfg: YarnRopeConfig? = {
            guard let rd = ropeDict,
                  let rt = ((rd["rope_type"] as? String) ?? (rd["type"] as? String))?.lowercased(),
                  rt == "yarn",
                  let factor = (rd["factor"] as? NSNumber)?.doubleValue,
                  let origMax = rd["original_max_position_embeddings"] as? Int else { return nil }
            let betaFast = (rd["beta_fast"] as? NSNumber)?.doubleValue ?? 32.0
            let betaSlow = (rd["beta_slow"] as? NSNumber)?.doubleValue ?? 1.0
            let mscale = (rd["mscale"] as? NSNumber)?.doubleValue ?? 1.0
            let mscaleAllDim = (rd["mscale_all_dim"] as? NSNumber)?.doubleValue ?? 0.0
            return YarnRopeConfig(
                factor: factor, originalMaxPositionEmbeddings: origMax,
                betaFast: betaFast, betaSlow: betaSlow,
                mscale: mscale, mscaleAllDim: mscaleAllDim
            )
        }()
        let kvLoraRank = tc["kv_lora_rank"] as? Int
        let qLoraRank = tc["q_lora_rank"] as? Int
        let nopeDim = tc["qk_nope_head_dim"] as? Int
        let ropeDim = tc["qk_rope_head_dim"] as? Int
        let vDim = (tc["v_head_dim"] as? Int) ?? nopeDim
        let mlaCfg: MLAConfig? = {
            guard let kvR = kvLoraRank, kvR > 0, let nD = nopeDim, let rD = ropeDim, let vD = vDim else { return nil }
            let gatedAttn = (tc["gated_attention"] as? Bool) ?? false
            let useMlaScale = (tc["use_mla_scaling_factor"] as? Bool) ?? false
            let qR = qLoraRank ?? 0
            let alphaQ: Float = (useMlaScale && qR > 0) ? Float((Double(hidden) / Double(qR)).squareRoot()) : 1.0
            let alphaKV: Float = Float(useMlaScale ? (Double(hidden) / Double(kvR)).squareRoot() : 1.0)
            let ropeCfg = MLARopeConfig(
                dim: rD, theta: ropeTheta, interleave: ropeInterleave, mropeLayout: .none1D, yarn: yarnCfg
            )
            return MLAConfig(
                qHeads: qHeads,
                qProjection: qR > 0 ? .factored(rank: qR) : .direct,
                kvLoraRank: kvR, qkNopeHeadDim: nD, qkRopeHeadDim: rD, vHeadDim: vD,
                rope: ropeCfg, sparsity: .dense(slidingWindow: 0),
                outputGateMode: gatedAttn ? .elementWiseFromHidden : .none,
                alphaQ: alphaQ, alphaKV: alphaKV,
                indexerKNormIsRMS: spec.defaultIndexerKNormIsRMS
            )
        }()
        return QWModelDescriptor(
            family: spec.family, modelType: modelType, capabilities: [],
            hidden: hidden, vocab: vocab, denseIntermediate: intermediate,
            numLayers: numLayers, layerTypes: layerTypes, fullAttentionInterval: 1,
            rmsEps: rmsEps, maxPositionEmbeddings: maxPos,
            qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
            attnOutputGate: false, partialRotaryFactor: partialRotary, ropeTheta: ropeTheta,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: nil, expertsPerTok: nil, moeIntermediate: nil,
            sharedExpertIntermediate: nil, expertGateType: nil, hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 0, gdnVHeads: 0, gdnHeadDim: 0, gdnQKVChannels: 0, gdnZChannels: 0, gdnConvKernel: 0,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 0, mtpUseDedicatedEmbeddings: false, eosTokenId: eosScalar,
            mropeSection: (0, 0, 0),
            visDepth: 0, visHidden: 0, visHeads: 0, visIntermediate: 0,
            visPatch: 0, visTemporal: 0, visMerge: 0, visPosEmbeds: 0, visOut: 0,
            imageTokenId: -1, videoTokenId: -1, visionStartTokenId: -1, visionEndTokenId: -1,
            hasQKNorm: (tc["qk_norm"] as? Bool) ?? true,
            usesPlusOneRMSNorm: spec.usesPlusOneRMSNorm,
            hasMropeSection: false, eosTokenIds: eosList,
            kvLoraRank: kvLoraRank, qLoraRank: qLoraRank,
            qkNopeHeadDim: nopeDim, qkRopeHeadDim: ropeDim, vHeadDim: vDim,
            yarnRopeConfig: yarnCfg, ropeScaling: yarnCfg.map { .yarn($0) },
            mlaConfig: mlaCfg, ropeInterleave: ropeInterleave, role: .speculativeDrafter,
            normKind: gatedNormConfig == nil ? .rmsNorm(eps: rmsEps, hasUnitOffset: spec.usesPlusOneRMSNorm) : nil,
            useSandwichNorm: useSandwichNorm, gatedNormConfig: gatedNormConfig,
            finalLogitSoftcapping: finalLogitSoftcapping)
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
