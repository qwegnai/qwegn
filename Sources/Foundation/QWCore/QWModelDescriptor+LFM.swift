// QWModelDescriptor+LFM.swift — LFM2/LFM3/LFM2-Encoder/ModernBERT & 5-axis decision trait decoding.
import Foundation

extension QWModelDescriptor {

    /// Return a copy of this descriptor with an updated `decisionConfig`.
    public func withDecisionConfig(_ updated: QWDecisionModelConfig?) -> QWModelDescriptor {
        var nextCaps = capabilities
        if updated != nil { nextCaps.insert(.decision) } else { nextCaps.remove(.decision) }
        return QWModelDescriptor(
            family: family, modelType: modelType, capabilities: nextCaps,
            hidden: hidden, vocab: vocab, denseIntermediate: denseIntermediate,
            numLayers: numLayers, layerTypes: layerTypes, fullAttentionInterval: fullAttentionInterval,
            rmsEps: rmsEps, maxPositionEmbeddings: maxPositionEmbeddings,
            qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
            attnOutputGate: attnOutputGate, partialRotaryFactor: partialRotaryFactor, ropeTheta: ropeTheta,
            indexerHeads: indexerHeads, indexerKvHeads: indexerKvHeads, indexerHeadDim: indexerHeadDim,
            indexerBudget: indexerBudget, indexerCompressRatio: indexerCompressRatio,
            numExperts: numExperts, expertsPerTok: expertsPerTok, moeIntermediate: moeIntermediate,
            sharedExpertIntermediate: sharedExpertIntermediate, expertGateType: expertGateType,
            hcCount: hcCount, hcLowrank: hcLowrank,
            gdnQKHeads: gdnQKHeads, gdnVHeads: gdnVHeads, gdnHeadDim: gdnHeadDim,
            gdnQKVChannels: gdnQKVChannels, gdnZChannels: gdnZChannels, gdnConvKernel: gdnConvKernel,
            pleLayerIds: pleLayerIds, pleEmbedDim: pleEmbedDim, pleConvKernel: pleConvKernel,
            ngramSize: ngramSize, headsPerNgram: headsPerNgram, ngramVocabSizeBase: ngramVocabSizeBase,
            makeNgramVocabDivisibleBy: makeNgramVocabDivisibleBy, splitNgramParts: splitNgramParts,
            mtpLayers: mtpLayers, mtpUseDedicatedEmbeddings: mtpUseDedicatedEmbeddings, eosTokenId: eosTokenId,
            mropeSection: mropeSection,
            visDepth: visDepth, visHidden: visHidden, visHeads: visHeads,
            visIntermediate: visIntermediate, visPatch: visPatch, visTemporal: visTemporal,
            visMerge: visMerge, visPosEmbeds: visPosEmbeds, visOut: visOut,
            imageTokenId: imageTokenId, videoTokenId: videoTokenId,
            visionStartTokenId: visionStartTokenId, visionEndTokenId: visionEndTokenId,
            hasQKNorm: hasQKNorm, hasQKVBias: hasQKVBias, hasAttnOutputBias: hasAttnOutputBias,
            usesPlusOneRMSNorm: usesPlusOneRMSNorm,
            quantization: quantization, hasMropeSection: hasMropeSection,
            tieWordEmbeddings: tieWordEmbeddings, normTopKProb: normTopKProb,
            eosTokenIds: eosTokenIds, envelopeModelType: envelopeModelType,
            loopSettings: loopSettings, mhcConfig: mhcConfig,
            firstKDenseReplace: firstKDenseReplace, mlpOnlyLayers: mlpOnlyLayers,
            decoderSparseStep: decoderSparseStep, moeLayerFreq: moeLayerFreq,
            routedScalingFactor: routedScalingFactor, swigluLimit: swigluLimit,
            topkMethod: topkMethod, residentExpertsPerLayer: residentExpertsPerLayer,
            sharedExpertHasGate: sharedExpertHasGate,
            mlaUseNoPE: mlaUseNoPE, kvLoraRank: kvLoraRank, qLoraRank: qLoraRank,
            qkNopeHeadDim: qkNopeHeadDim, qkRopeHeadDim: qkRopeHeadDim, vHeadDim: vHeadDim,
            indexKPoolCompress: indexKPoolCompress, indexKPoolAlwaysSelectTail: indexKPoolAlwaysSelectTail,
            indexShareForMTP: indexShareForMTP, indexerRopeInterleave: indexerRopeInterleave,
            gdnVectorDecay: gdnVectorDecay, gdnLowRankDim: gdnLowRankDim,
            gdnGateLowerBound: gdnGateLowerBound, gdnSeparateConv: gdnSeparateConv,
            hcaConfig: hcaConfig, hybridMLAConfig: hybridMLAConfig,
            blockSparseGQAConfig: blockSparseGQAConfig,
            yarnRopeConfig: yarnRopeConfig, ropeScaling: ropeScaling,
            slidingWindow: slidingWindow, layerSlidingWindows: layerSlidingWindows,
            activationKind: activationKind, moeLatentSize: moeLatentSize,
            latentMoeUseNorm: latentMoeUseNorm,
            moeSplitFeatureRouting: moeSplitFeatureRouting,
            moeRouterEnableExpertBias: moeRouterEnableExpertBias,
            routedScalingFactorLearnable: routedScalingFactorLearnable,
            multimaxHead: multimaxHead, mtpSharedLastLayer: mtpSharedLastLayer,
            inkling2Config: inkling2Config, dsparkConfig: dsparkConfig,
            mlaConfig: mlaConfig, kdaConfig: kdaConfig, attentionSinkRelConfig: attentionSinkRelConfig,
            moeRoutingConfig: moeRoutingConfig, swigluClampConfig: swigluClampConfig,
            mtpConfig: mtpConfig, visionTowerConfig: visionTowerConfig,
            ropeInterleave: ropeInterleave, role: role,
            shortConvConfig: shortConvConfig, normKind: normKind,
            useSandwichNorm: useSandwichNorm, perKindRoPEConfig: perKindRoPEConfig,
            attnGateMode: attnGateMode, decisionConfig: updated,
            gatedNormConfig: gatedNormConfig, gdnGatingConfig: gdnGatingConfig,
            finalLogitSoftcapping: finalLogitSoftcapping, v2DigestHeadDim: v2DigestHeadDim
        )
    }

    internal static func appendLFMAndDecisionDigestFields(_ d: QWModelDescriptor, into f: inout [String]) {
        if let sc = d.shortConvConfig {
            f.append("sconv=\(sc.kernelSize),\(sc.hasBias ? 1 : 0),\(sc.isCentered ? 1 : 0)")
        }
        if d.useSandwichNorm || d.normKind != nil {
            let kindStr: String
            switch d.normKind {
            case .layerNorm(let eps, let bias):
                kindStr = "ln(\(String(eps.bitPattern, radix: 16)),\(bias ? 1 : 0))"
            case .rmsNorm(let eps, let unit):
                kindStr = "rms(\(String(eps.bitPattern, radix: 16)),\(unit ? 1 : 0))"
            case nil:
                kindStr = "-"
            }
            f.append("swnorm=\(d.useSandwichNorm ? 1 : 0),\(kindStr)")
        }
        if let pk = d.perKindRoPEConfig, !pk.byLayerKind.isEmpty {
            let parts = pk.byLayerKind.keys.sorted().compactMap { key -> String? in
                guard let e = pk.byLayerKind[key] else { return nil }
                return "\(key):\(String(e.ropeTheta.bitPattern, radix: 16)):\(String(e.partialRotaryFactor.bitPattern, radix: 16))"
            }
            f.append("pkrope=\(parts.joined(separator: ","))")
        }
        if d.attnGateMode != .none && d.attnGateMode != .packedElementWiseQProj {
            f.append("agate=\(d.attnGateMode.rawValue)")
        }
        if let dc = d.decisionConfig {
            f.append("topo=\(dc.topologyDigestToken)")
            f.append("dhead=\(dc.headDigestToken)")
        }
    }

    internal static func resolveNormEps(text t: ConfigEnvelope.Text, familySpec: QWModelFamilySpec) throws -> Double {
        if let eps = t.rms_norm_eps ?? t.norm_eps ?? t.layer_norm_eps ?? t.block_norm_eps {
            return eps
        }
        if familySpec.usesLayerNormByDefault {
            return 1e-5
        }
        throw DescriptorError.invariant("flat config missing 'rms_norm_eps'")
    }

    internal static func resolveAdjustedIntermediate(text t: ConfigEnvelope.Text) -> Int? {
        guard let inter = t.intermediate_size else { return nil }
        guard t.block_auto_adjust_ff_dim == true else { return inter }
        let mult = t.block_ffn_dim_multiplier ?? 1.0
        var adjusted = Int(Double(2 * inter) / 3.0 * mult)
        let m = max(1, t.block_multiple_of ?? 256)
        adjusted = ((adjusted + m - 1) / m) * m
        return max(m, adjusted)
    }

    internal static func resolveLayerTypesForLFM(
        text t: ConfigEnvelope.Text,
        familySpec: QWModelFamilySpec
    ) -> [String]? {
        guard familySpec.family == .modernBert else { return nil }
        let every = max(1, t.global_attn_every_n_layers ?? 1)
        let hasLocal = (t.local_attention ?? 0) > 0 && every > 1
        return (0..<t.num_hidden_layers).map { l in
            (hasLocal && l % every != 0) ? "sliding_attention" : "full_attention"
        }
    }

    internal struct DecodedLFMTraits {
        let shortConv: ShortConvConfig?
        let normKind: NormKind?
        let useSandwichNorm: Bool
        let perKindRoPE: PerKindRoPEConfig?
        let attnGateMode: AttnGateMode
        let decisionConfig: QWDecisionModelConfig?
    }

    internal static func decodeLFMAndDecisionTraits(
        text t: ConfigEnvelope.Text,
        env: ConfigEnvelope,
        familySpec: QWModelFamilySpec,
        normalizedLayerTypes: [String],
        resolvedNormEps: Double,
        resolvedMaxPos: Int,
        tieWordEmbeddings: Bool,
        attnGate: Bool,
        granularity: String
    ) -> DecodedLFMTraits {
        let family = familySpec.family
        let hasConv = normalizedLayerTypes.contains("conv")
        let shortConv: ShortConvConfig? = hasConv ? ShortConvConfig(
            kernelSize: max(1, t.conv_L_cache ?? 3),
            hasBias: t.conv_bias ?? false,
            isCentered: (family == .lfm2Encoder),
            stateKey: hasConv ? "conv" : nil
        ) : nil

        let sandwich = (t.use_sandwich_norm == true) || (t.layernorm_type == "pre_post")
        let isGatedNorm = (t.norm_type == "GatedNorm" || t.norm_type == "ZeroCenteredGatedNorm")
        let normKind: NormKind?
        if familySpec.usesLayerNormByDefault {
            normKind = .layerNorm(eps: Float(resolvedNormEps), hasBias: t.norm_bias ?? false)
        } else if isGatedNorm {
            normKind = nil
        } else if sandwich || family == .lfm2 || family == .lfm3 || family == .lfm2Encoder {
            normKind = .rmsNorm(eps: Float(resolvedNormEps), hasUnitOffset: familySpec.usesPlusOneRMSNorm)
        } else {
            normKind = nil
        }

        var pkEntries: [String: PerKindRoPEEntry] = [:]
        if let fa = t.rope_parameters?.full_attention, let theta = fa.rope_theta {
            pkEntries["full_attention"] = PerKindRoPEEntry(
                ropeTheta: theta,
                partialRotaryFactor: fa.partial_rotary_factor ?? 1.0,
                ropeType: fa.rope_type ?? fa.type ?? "default"
            )
        } else if let gTheta = t.global_rope_theta {
            pkEntries["full_attention"] = PerKindRoPEEntry(ropeTheta: gTheta)
        }
        if let sw = t.rope_parameters?.sliding_attention, let theta = sw.rope_theta {
            pkEntries["sliding_attention"] = PerKindRoPEEntry(
                ropeTheta: theta,
                partialRotaryFactor: sw.partial_rotary_factor ?? 1.0,
                ropeType: sw.rope_type ?? sw.type ?? "default"
            )
        } else if let lTheta = t.local_rope_theta {
            pkEntries["sliding_attention"] = PerKindRoPEEntry(ropeTheta: lTheta)
        }
        let perKindRoPE: PerKindRoPEConfig? = pkEntries.isEmpty ? nil : PerKindRoPEConfig(
            byLayerKind: pkEntries, zeroTailPartialRotation: family == .lfm3
        )

        let gateMode: AttnGateMode
        if granularity == "head_wise" || (attnGate && family == .lfm3 && granularity.isEmpty) {
            gateMode = .separateHeadWiseSigmoid
        } else if attnGate {
            gateMode = .packedElementWiseQProj
        } else {
            gateMode = .none
        }

        let decisionConfig = decodeDecisionModelConfig(
            text: t, env: env, familySpec: familySpec,
            resolvedMaxPos: resolvedMaxPos, tieWordEmbeddings: tieWordEmbeddings
        )
        return DecodedLFMTraits(
            shortConv: shortConv,
            normKind: normKind,
            useSandwichNorm: sandwich,
            perKindRoPE: perKindRoPE,
            attnGateMode: gateMode,
            decisionConfig: decisionConfig
        )
    }

    private static func decodeDecisionModelConfig(
        text t: ConfigEnvelope.Text,
        env: ConfigEnvelope,
        familySpec: QWModelFamilySpec,
        resolvedMaxPos: Int,
        tieWordEmbeddings: Bool
    ) -> QWDecisionModelConfig? {
        let family = familySpec.family
        if family == .lfm2Encoder {
            return QWDecisionModelConfig(
                framing: .optionMarkers(
                    layout: .stateQuestionOptions,
                    markerToken: "<|mask|>",
                    bosToken: "<bos>",
                    maxLength: env.max_length ?? resolvedMaxPos,
                    imageTextLength: env.image_text_length,
                    audioTextLength: env.audio_text_length
                ),
                topology: .bidirectionalEncoder(isolateMediaPrefix: true),
                head: .markerScorer(hiddenSize: t.hidden_size, headLayers: env.head_layers ?? 2, chains: ["default"]),
                hasStatePointer: false,
                hasGridReadout: false,
                usesNumbersThousandthsGrid: false,
                inlineTemperatures: env.temperatures ?? [:],
                defaultScoreReadout: "mean"
            )
        }
        let isD1AutoMap = env.auto_map?.values.contains(where: { $0.contains("D1Model") }) == true
        let hasD1Keys = env.d1_codes != nil || env.d1_attention != nil
            || env.d1_numbers != nil || env.d1_set_size != nil || isD1AutoMap
        if hasD1Keys {
            let packSet = (env.d1_attention?.lowercased() == "set")
            let maxSetSize = max(1, env.d1_set_size ?? 16)
            let codes = max(1, env.d1_codes ?? 1)
            let numbers = env.d1_numbers ?? false
            let lead = (family == .lfm3) ? "<think></think>" : ""
            return QWDecisionModelConfig(
                framing: .turnVerbalizer(
                    codesPerOption: codes,
                    numbers: numbers,
                    packSet: packSet,
                    maxSetSize: maxSetSize,
                    colonSlots: true,
                    stateSeparator: "\n\n",
                    assistantLead: lead,
                    chatML: true
                ),
                topology: packSet ? .packedSet(maxSetSize: maxSetSize) : .causal(supportsPrefixCache: true),
                head: .vocabVerbalizer(vocabSize: t.vocab_size, tieEmbeddings: tieWordEmbeddings, sparseGatherThreshold: 256),
                hasStatePointer: true,
                hasGridReadout: true,
                usesNumbersThousandthsGrid: numbers,
                boxEdgesMode: "median",
                visionMaxPixels: env.d1_vision_max_pixels,
                inlineTemperatures: env.temperatures ?? [:],
                defaultScoreReadout: "mean"
            )
        }
        let rawId2Label = t.id2label ?? env.id2label
        let archs = t.architectures ?? env.architectures ?? []
        let isSeqCls = archs.contains(where: { $0.hasSuffix("ForSequenceClassification") })
        let isGenerativeArch = archs.contains(where: {
            $0.hasSuffix("ForCausalLM") || $0.hasSuffix("ForConditionalGeneration")
        })
        let allowsBoilerplateId2Label = rawId2Label != nil && (!isGenerativeArch || familySpec.defaultRole == .decisionEncoder)
        if family == .modernBert || isSeqCls || allowsBoilerplateId2Label {
            var parsedLabels: [Int: String] = [:]
            if let raw = rawId2Label {
                for (k, v) in raw { if let idx = Int(k) { parsedLabels[idx] = v } }
            }
            let numLabels = max(1, max(parsedLabels.count, t.num_labels ?? env.num_labels ?? 2))
            let poolRaw = (t.classifier_pooling ?? env.classifier_pooling ?? "cls").lowercased()
            let pooling: QWDecisionModelConfig.PoolingKind = poolRaw == "mean" ? .meanNonPad
                : (poolRaw == "last" ? .lastToken : .firstToken)
            let isEncoder = (family == .modernBert || family == .lfm2Encoder)
            return QWDecisionModelConfig(
                framing: .sequencePair(maxLength: env.max_length ?? resolvedMaxPos),
                topology: isEncoder ? .bidirectionalEncoder(isolateMediaPrefix: false) : .causal(supportsPrefixCache: true),
                head: .pooledClassifier(pooling: pooling, numLabels: numLabels, id2label: parsedLabels),
                inlineTemperatures: env.temperatures ?? [:],
                defaultScoreReadout: "mean"
            )
        }
        return nil
    }
}
