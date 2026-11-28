// QWModelDescriptor+CapabilitySynthesis.swift — Synthesized L1-L7 capability bundles, RoPE scaling, and V3 digest helpers.
import Foundation

extension QWModelDescriptor {

    internal static func synthesizeCapabilityBundle(
        family: Family, modelType: String, capabilities: Set<Capability>,
        qHeads: Int, headDim: Int, attnOutputGate: Bool, ropeTheta: Double, ropeInterleave: Bool,
        indexerHeads: Int?, indexKPoolCompress: Bool, mlaUseNoPE: Bool,
        kvLoraRank: Int?, qLoraRank: Int?, qkNopeHeadDim: Int?, qkRopeHeadDim: Int?, vHeadDim: Int?,
        hybridMLAConfig: HybridMLAConfig?, yarnRopeConfig: YarnRopeConfig?,
        numExperts: Int?, denseIntermediate: Int?, sharedExpertHasGate: Bool,
        expertGateType: String?, topkMethod: String?, moeRouterEnableExpertBias: Bool,
        routedScalingFactor: Float, routedScalingFactorLearnable: Bool, normTopKProb: Bool,
        swigluLimit: Float?, mtpLayers: Int, mtpUseDedicatedEmbeddings: Bool, mtpSharedLastLayer: Bool,
        mhcConfig: MHCConfig?, gdnQKHeads: Int, gdnVHeads: Int, gdnHeadDim: Int, gdnConvKernel: Int,
        gdnVectorDecay: Bool, gdnLowRankDim: Int?, gdnGateLowerBound: Float?, gdnSeparateConv: Bool,
        visDepth: Int, visHidden: Int, visHeads: Int, visIntermediate: Int, visPatch: Int,
        visTemporal: Int, visMerge: Int, visPosEmbeds: Int, visOut: Int,
        hasMropeSection: Bool, slidingWindow: Int?, inkling2Config: Inkling2Config?
    ) -> (
        kda: KDAConfig?, mla: MLAConfig?, sinkRel: AttentionSinkRelConfig?, moe: MoERoutingConfig?,
        swiglu: SwiGLUClampConfig?, mtp: MTPConfig?, vision: VisionTowerConfig?
    ) {
        let spec = QWModelFamilyRegistry.shared.spec(forModelType: modelType)
        let kda: KDAConfig? = (gdnVectorDecay && gdnQKHeads > 0) ? KDAConfig(
            qkHeads: gdnQKHeads, vHeads: gdnVHeads, headDim: gdnHeadDim, convKernel: gdnConvKernel,
            gateProjection: gdnLowRankDim.map { .lowRank(dim: $0) } ?? .fullRank,
            safeGate: false, gateLowerBound: gdnGateLowerBound, outputNormAct: .sigmoid,
            separateConv: gdnSeparateConv
        ) : nil
        let sparsity: MLAConfig.Sparsity = (indexerHeads ?? 0) > 0
            ? (indexKPoolCompress ? .keyPool : .dsa(slidingWindow: hybridMLAConfig?.slidingWindow ?? slidingWindow ?? 0))
            : .dense(slidingWindow: hybridMLAConfig?.slidingWindow ?? slidingWindow ?? 0)
        let mla: MLAConfig? = {
            if let h = hybridMLAConfig {
                let qProj: MLAConfig.QProjection = h.qLoraRank > 0 ? .factored(rank: h.qLoraRank) : .direct
                let outMode: MLAConfig.OutputGateMode = attnOutputGate
                    ? (qProj == .direct ? .headWiseFromHidden : .elementWiseFromQLora) : .none
                let ropeCfg: MLARopeConfig? = h.decoupledRope ? MLARopeConfig(
                    dim: h.qkRopeHeadDim, theta: ropeTheta, interleave: h.ropeInterleave,
                    mropeLayout: hasMropeSection ? .qwenInterleaved : .none1D, yarn: h.yarn
                ) : nil
                return MLAConfig(
                    qHeads: h.qHeads, qProjection: qProj, kvLoraRank: h.kvLoraRank,
                    qkNopeHeadDim: h.qkNopeHeadDim, qkRopeHeadDim: h.qkRopeHeadDim, vHeadDim: h.vHeadDim,
                    rope: ropeCfg, sparsity: sparsity, outputGateMode: outMode,
                    useVHA: h.useVHA, hasAttnSinkBias: h.hasAttnSinkBias,
                    splitKvBProj: h.splitKvBProj, indexerUseFastHadamard: h.indexerUseFastHadamard
                )
            }
            if mlaUseNoPE, let kvR = kvLoraRank, kvR > 0 {
                let qProj: MLAConfig.QProjection = (qLoraRank ?? 0) > 0 ? .factored(rank: qLoraRank!) : .direct
                return MLAConfig(
                    qHeads: qHeads, qProjection: qProj, kvLoraRank: kvR,
                    qkNopeHeadDim: qkNopeHeadDim ?? headDim, qkRopeHeadDim: 0, vHeadDim: vHeadDim ?? headDim,
                    rope: nil, sparsity: sparsity, outputGateMode: attnOutputGate ? .elementWiseFromHidden : .none
                )
            }
            if let kvR = kvLoraRank, kvR > 0, let nope = qkNopeHeadDim, let ropeDim = qkRopeHeadDim {
                let qProj: MLAConfig.QProjection = (qLoraRank ?? 0) > 0 ? .factored(rank: qLoraRank!) : .direct
                let layout: MRoPELayout = hasMropeSection
                    ? ((spec?.defaultBailingVideoRope == true) ? .bailingVideoRope : .qwenInterleaved)
                    : .none1D
                let outMode: MLAConfig.OutputGateMode = attnOutputGate
                    ? ((family == .gigachat4) ? .elementWiseFromHidden : .headWiseFromHidden)
                    : .none
                return MLAConfig(
                    qHeads: qHeads, qProjection: qProj, kvLoraRank: kvR,
                    qkNopeHeadDim: nope, qkRopeHeadDim: ropeDim, vHeadDim: vHeadDim ?? headDim,
                    rope: ropeDim > 0 ? MLARopeConfig(
                        dim: ropeDim, theta: ropeTheta, interleave: ropeInterleave,
                        mropeLayout: layout, yarn: yarnRopeConfig
                    ) : nil,
                    sparsity: sparsity, outputGateMode: outMode,
                    indexerKNormIsRMS: spec?.defaultIndexerKNormIsRMS ?? false
                )
            }
            return nil
        }()
        let sinkRel: AttentionSinkRelConfig? = inkling2Config.map { ink in
            AttentionSinkRelConfig(
                hasSink: true, dRel: ink.dRel, relExtent: ink.relExtent,
                slidingWindowSize: ink.slidingWindowSize, swaQHeads: ink.swaQHeads,
                swaKVHeads: ink.swaKVHeads, swaHeadDim: ink.swaHeadDim,
                logScalingFloor: ink.logScalingNFloor, logScalingAlpha: ink.logScalingAlpha,
                qkL2ClipThreshold: ink.qkL2ClipThreshold, relL2ClipThreshold: ink.relL2ClipThreshold,
                valueNormNoGamma: ink.useValueNormNoGamma, finalLogitSoftcapping: ink.finalLogitSoftcapping,
                localLayerIds: ink.localLayerIds, sconvKernelSize: ink.sconvKernelSize,
                layersPerAttnResBlock: ink.layersPerAttnResBlock,
                attnResNormalizeValues: ink.attnResNormalizeValues
            )
        }
        let moe: MoERoutingConfig? = (numExperts ?? 0) > 0 ? MoERoutingConfig(
            nGroup: 1, topkGroup: 1, groupTopM: 2, sharedExpertHasGate: sharedExpertHasGate,
            scoreFunction: expertGateType ?? "softmax",
            enableExpertBias: topkMethod == "noaux_tc" || moeRouterEnableExpertBias,
            routedScalingFactor: routedScalingFactor,
            routedScalingFactorLearnable: routedScalingFactorLearnable,
            normTopKProb: normTopKProb
        ) : nil
        let swiglu: SwiGLUClampConfig? = swigluLimit.map {
            SwiGLUClampConfig(mode: .preSilu, denseLimit: $0)
        }
        let mtpIsMoE = (numExperts ?? 0) > 0 && !(spec?.defaultMTPIsDense ?? false)
        let mtpHasIdx = mla?.hasIndexer ?? ((indexerHeads ?? 0) > 0)
        let mtp: MTPConfig? = mtpLayers > 0 ? MTPConfig(
            layers: mtpLayers,
            fusionMode: mtpSharedLastLayer ? .sharedLastLayerEcho
                : ((family == .maxwellMoE && mhcConfig != nil) ? .splitEProjHProj : .concatEHProj),
            useDedicatedEmbeddings: mtpUseDedicatedEmbeddings,
            hasFinalNorm: false,
            isMoE: mtpIsMoE,
            intermediateSize: mtpIsMoE ? nil : denseIntermediate,
            attentionIsSparse: spec?.defaultMTPAttentionIsSparse ?? mtpHasIdx,
            attentionIsLinear: false,
            sharedEmbedAndHead: !mtpUseDedicatedEmbeddings
        ) : nil
        let vision: VisionTowerConfig? = capabilities.contains(.vision) ? VisionTowerConfig(
            depth: visDepth, hidden: visHidden, heads: visHeads, intermediate: visIntermediate,
            patch: visPatch, temporal: visTemporal, merge: visMerge, numPosEmbeds: visPosEmbeds,
            outDim: visOut, mergerIntermediateDim: inkling2Config?.vision?.projectionHiddenDim,
            projectorBias: inkling2Config?.vision?.projectorBias ?? true,
            normEps: Float(inkling2Config?.vision?.layerNormEps ?? 1e-6),
            mropeLayout: (spec?.defaultBailingVideoRope == true) ? .bailingVideoRope
                : (hasMropeSection ? .qwenInterleaved : .none1D),
            patchEmbedKind: inkling2Config?.vision != nil ? .unfoldedPatchProj : .conv3d,
            imageMean: spec?.defaultVisionMean ?? [0.5, 0.5, 0.5],
            imageStd: spec?.defaultVisionStd ?? [0.5, 0.5, 0.5]
        ) : nil
        return (kda, mla, sinkRel, moe, swiglu, mtp, vision)
    }

    internal static func appendCapabilityV3DigestFields(_ d: QWModelDescriptor, into f: inout [String]) {
        if let kda = d.kdaConfig, kda.isFullRank || kda.safeGate {
            f.append("kda_cfg=\(kda.isFullRank ? "full" : "lora\(kda.lowRankDim ?? 0)"),\(kda.safeGate ? 1 : 0),\(kda.outputNormAct.rawValue)")
        }
        if let mla = d.mlaConfig, mla.isDirectQ || mla.outputGateMode == .headWiseFromHidden || (d.hybridMLAConfig == nil && !d.mlaUseNoPE) {
            f.append("mla_cfg=\(mla.isDirectQ ? 1 : 0),\(mla.outputGateMode.rawValue),\(mla.hasIndexer ? 1 : 0),\(mla.rope?.interleave == true ? 1 : 0)")
        }
        if let moe = d.moeRoutingConfig, moe.nGroup > 1 {
            f.append("moe_grp=\(moe.nGroup),\(moe.topkGroup),\(moe.groupTopM),\(moe.scoreFunction)")
        }
        if let sw = d.swigluClampConfig, sw.mode == .postSilu || !sw.layerRoutedLimits.isEmpty || !sw.layerSharedLimits.isEmpty {
            let rHex = sw.layerRoutedLimits.map { String($0.bitPattern, radix: 16) }.joined(separator: ",")
            let sHex = sw.layerSharedLimits.map { String($0.bitPattern, radix: 16) }.joined(separator: ",")
            f.append("swiglu_layers=\(sw.mode.rawValue)|r=\(rHex)|s=\(sHex)")
        }
        if let vis = d.visionTowerConfig, vis.disableInternalMergerProj {
            f.append("vis_merger=\(vis.mergerIntermediateDim ?? 0),\(vis.outDim),1")
        }
        if d.ropeInterleave {
            f.append("rope_ilv=1")
        }
    }

    internal static func decodeSwiGLUClampConfig(
        text t: ConfigEnvelope.Text,
        routedLimits: [Float]?,
        sharedLimits: [Float]?,
        inv: (Bool, String) throws -> Void
    ) throws -> SwiGLUClampConfig? {
        let singleLimit: Float? = (t.mlp_clamp == false) ? nil : (t.swiglu_limit ?? t.swiglu_limits)
        let rList = routedLimits ?? []
        let sList = sharedLimits ?? []
        if let lim = singleLimit {
            try inv(lim > 0 && lim.isFinite, "swiglu_limit \(lim) invalid")
        }
        for v in rList { try inv(v >= 0 && v.isFinite, "expert_swiglu_limit_list value \(v) invalid") }
        for v in sList { try inv(v >= 0 && v.isFinite, "share_expert_swiglu_limit_list value \(v) invalid") }
        let hasPerLayer = rList.contains(where: { $0 > 0 }) || sList.contains(where: { $0 > 0 })
        if hasPerLayer {
            return SwiGLUClampConfig(
                mode: .postSilu,
                denseLimit: singleLimit,
                layerRoutedLimits: rList,
                layerSharedLimits: sList
            )
        }
        if let lim = singleLimit {
            return SwiGLUClampConfig(mode: .preSilu, denseLimit: lim)
        }
        return nil
    }

    internal static func parseRoPEScaling(
        text t: ConfigEnvelope.Text,
        env: ConfigEnvelope
    ) throws -> (RoPEScaling?, YarnRopeConfig?) {
        let candidates: [(type: String?, ropeType: String?, factor: Double?, orig: Int?,
                          betaFast: Double?, betaSlow: Double?, mscale: Double?, mscaleAllDim: Double?,
                          defaultMscale: Double)] = [
            t.rope_scaling.map {
                ($0.type, $0.rope_type, $0.factor, $0.original_max_position_embeddings,
                 $0.beta_fast, $0.beta_slow, $0.mscale, $0.mscale_all_dim, 0.0)
            },
            t.rope_parameters.map {
                ($0.type, $0.rope_type, $0.factor, $0.original_max_position_embeddings,
                 $0.beta_fast, $0.beta_slow, $0.mscale, $0.mscale_all_dim, 1.0)
            },
        ].compactMap { $0 }

        for c in candidates {
            let rawKind = c.ropeType ?? c.type
            guard let kind = rawKind?.lowercased() else {
                if c.factor != nil {
                    throw DescriptorError.invariant("unsupported rope_scaling without a type")
                }
                continue
            }
            if kind == "default" { continue }
            guard let factor = c.factor, factor > 0, factor.isFinite else {
                throw DescriptorError.invariant("rope_scaling '\(kind)' missing positive factor")
            }
            let orig = c.orig ?? t.original_max_position_embeddings ?? env.original_max_position_embeddings
            switch kind {
            case "yarn":
                guard let orig, orig > 0 else {
                    throw DescriptorError.invariant("rope_scaling 'yarn' missing original_max_position_embeddings")
                }
                let yarn = YarnRopeConfig(
                    factor: factor,
                    originalMaxPositionEmbeddings: orig,
                    betaFast: c.betaFast ?? 32.0,
                    betaSlow: c.betaSlow ?? 1.0,
                    mscale: c.mscale ?? c.defaultMscale,
                    mscaleAllDim: c.mscaleAllDim ?? 0.0
                )
                return (.yarn(yarn), yarn)
            case "linear":
                return (.linear(factor: factor), nil)
            case "dynamic":
                return (.dynamic(factor: factor, originalMaxPositionEmbeddings: orig), nil)
            default:
                throw DescriptorError.invariant("unsupported rope_scaling '\(rawKind ?? kind)'")
            }
        }
        return (nil, nil)
    }
}
