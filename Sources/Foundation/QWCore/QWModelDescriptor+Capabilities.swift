// QWModelDescriptor+Capabilities.swift — Config-driven capability structs for L1-L7.
import Foundation

/// Per-layer SwiGLU clamp configuration specialized via HAL function constants.
public struct QWSwiGLUClampSpec: Sendable, Hashable, Codable, Equatable {
    public let limit: Float
    public let postSilu: Bool

    public init(limit: Float, postSilu: Bool = false) {
        self.limit = limit
        self.postSilu = postSilu
    }
}

extension QWModelDescriptor {

    /// Spatial/temporal MRoPE rotary axis layout (`rope_parameters` / model config).
    public enum MRoPELayout: String, Sendable, Hashable, Codable, Equatable {
        case none1D
        case qwenInterleaved
        case bailingVideoRope
    }

    /// Config-sourced GDN / KDA linear-attention parameters.
    public struct KDAConfig: Sendable, Hashable, Codable, Equatable {
        public enum GateProjection: Sendable, Hashable, Codable, Equatable {
            case lowRank(dim: Int)
            case fullRank
        }

        public enum NormActivation: String, Sendable, Hashable, Codable, Equatable {
            case silu
            case sigmoid
        }

        public let qkHeads: Int
        public let vHeads: Int
        public let headDim: Int
        public let convKernel: Int
        public let gateProjection: GateProjection
        public let safeGate: Bool
        public let gateLowerBound: Float?
        public let outputNormAct: NormActivation
        public let separateConv: Bool

        public init(
            qkHeads: Int,
            vHeads: Int,
            headDim: Int,
            convKernel: Int,
            gateProjection: GateProjection,
            safeGate: Bool = false,
            gateLowerBound: Float? = nil,
            outputNormAct: NormActivation = .sigmoid,
            separateConv: Bool = true
        ) {
            self.qkHeads = qkHeads
            self.vHeads = vHeads
            self.headDim = headDim
            self.convKernel = convKernel
            self.gateProjection = gateProjection
            self.safeGate = safeGate
            self.gateLowerBound = gateLowerBound
            self.outputNormAct = outputNormAct
            self.separateConv = separateConv
        }

        public var lowRankDim: Int? {
            if case .lowRank(let dim) = gateProjection { return dim }
            return nil
        }

        public var isFullRank: Bool {
            if case .fullRank = gateProjection { return true }
            return false
        }
    }

    /// Rotary parameters bound to an MLA attention block.
    public struct MLARopeConfig: Sendable, Hashable, Codable, Equatable {
        public let dim: Int
        public let theta: Double
        public let interleave: Bool
        public let mropeLayout: MRoPELayout
        public let yarn: YarnRopeConfig?

        public init(
            dim: Int,
            theta: Double,
            interleave: Bool = false,
            mropeLayout: MRoPELayout = .none1D,
            yarn: YarnRopeConfig? = nil
        ) {
            self.dim = dim
            self.theta = theta
            self.interleave = interleave
            self.mropeLayout = mropeLayout
            self.yarn = yarn
        }
    }

    /// Unified config-sourced Multi-head Latent Attention traits.
    public struct MLAConfig: Sendable, Hashable, Codable, Equatable {
        public enum QProjection: Sendable, Hashable, Codable, Equatable {
            case factored(rank: Int)
            case direct
        }

        public enum Sparsity: Sendable, Hashable, Codable, Equatable {
            case dense(slidingWindow: Int)
            case keyPool
            case dsa(slidingWindow: Int)
        }

        public enum OutputGateMode: String, Sendable, Hashable, Codable, Equatable {
            case none
            case elementWiseFromQLora
            case headWiseFromHidden
            case elementWiseFromHidden
        }

        public let qHeads: Int
        public let qProjection: QProjection
        public let kvLoraRank: Int
        public let qkNopeHeadDim: Int
        public let qkRopeHeadDim: Int
        public let vHeadDim: Int
        public let rope: MLARopeConfig?
        public let sparsity: Sparsity
        public let outputGateMode: OutputGateMode
        public let useVHA: Bool
        public let hasAttnSinkBias: Bool
        public let splitKvBProj: Bool
        public let indexerUseFastHadamard: Bool
        public let alphaQ: Float
        public let alphaKV: Float
        public let indexerKNormIsRMS: Bool
        public let softmaxScale: Float
        public let denseAttentionLayers: Set<Int>

        public static func defaultSoftmaxScale(
            qkNopeHeadDim: Int,
            qkRopeHeadDim: Int,
            yarn: YarnRopeConfig?
        ) -> Float {
            let qkDim = max(1, qkNopeHeadDim + qkRopeHeadDim)
            var scale = 1.0 / Double(qkDim).squareRoot()
            if let y = yarn, y.factor > 1.0, y.mscaleAllDim > 0.0 {
                let ms = 0.1 * y.mscaleAllDim * Foundation.log(y.factor) + 1.0
                scale *= ms * ms
            }
            return Float(scale)
        }

        public init(
            qHeads: Int,
            qProjection: QProjection,
            kvLoraRank: Int,
            qkNopeHeadDim: Int,
            qkRopeHeadDim: Int,
            vHeadDim: Int,
            rope: MLARopeConfig? = nil,
            sparsity: Sparsity = .dense(slidingWindow: 0),
            outputGateMode: OutputGateMode = .none,
            useVHA: Bool = false,
            hasAttnSinkBias: Bool = false,
            splitKvBProj: Bool = false,
            indexerUseFastHadamard: Bool = false,
            alphaQ: Float = 1.0,
            alphaKV: Float = 1.0,
            indexerKNormIsRMS: Bool = false,
            softmaxScale: Float? = nil,
            denseAttentionLayers: Set<Int> = []
        ) {
            self.qHeads = qHeads
            self.qProjection = qProjection
            self.kvLoraRank = kvLoraRank
            self.qkNopeHeadDim = qkNopeHeadDim
            self.qkRopeHeadDim = qkRopeHeadDim
            self.vHeadDim = vHeadDim
            self.rope = rope
            self.sparsity = sparsity
            self.outputGateMode = outputGateMode
            self.useVHA = useVHA
            self.hasAttnSinkBias = hasAttnSinkBias
            self.splitKvBProj = splitKvBProj
            self.indexerUseFastHadamard = indexerUseFastHadamard
            self.alphaQ = alphaQ
            self.alphaKV = alphaKV
            self.indexerKNormIsRMS = indexerKNormIsRMS
            self.softmaxScale = softmaxScale ?? Self.defaultSoftmaxScale(
                qkNopeHeadDim: qkNopeHeadDim,
                qkRopeHeadDim: qkRopeHeadDim,
                yarn: rope?.yarn
            )
            self.denseAttentionLayers = denseAttentionLayers
        }

        public var qLoraRank: Int {
            if case .factored(let rank) = qProjection { return rank }
            return 0
        }

        public var isDirectQ: Bool {
            if case .direct = qProjection { return true }
            return false
        }

        public var slidingWindow: Int {
            switch sparsity {
            case .dense(let sw), .dsa(let sw): return sw
            case .keyPool: return 0
            }
        }

        public var hasIndexer: Bool {
            switch sparsity {
            case .dense: return false
            case .keyPool, .dsa: return true
            }
        }
    }

    /// Attention sinks, relative biases, per-layer SWA geometry, and log-length scaling traits.
    public struct AttentionSinkRelConfig: Sendable, Hashable, Codable, Equatable {
        public let hasSink: Bool
        public let dRel: Int
        public let relExtent: Int
        public let slidingWindowSize: Int?
        public let swaQHeads: Int?
        public let swaKVHeads: Int?
        public let swaHeadDim: Int?
        public let logScalingFloor: Int?
        public let logScalingAlpha: Float?
        public let qkL2ClipThreshold: Float?
        public let relL2ClipThreshold: Float?
        public let valueNormNoGamma: Bool
        public let finalLogitSoftcapping: Float?
        public let localLayerIds: Set<Int>
        public let sconvKernelSize: Int
        public let layersPerAttnResBlock: Int?
        public let attnResNormalizeValues: Bool

        public init(
            hasSink: Bool = false,
            dRel: Int = 0,
            relExtent: Int = 0,
            slidingWindowSize: Int? = nil,
            swaQHeads: Int? = nil,
            swaKVHeads: Int? = nil,
            swaHeadDim: Int? = nil,
            logScalingFloor: Int? = nil,
            logScalingAlpha: Float? = nil,
            qkL2ClipThreshold: Float? = nil,
            relL2ClipThreshold: Float? = nil,
            valueNormNoGamma: Bool = false,
            finalLogitSoftcapping: Float? = nil,
            localLayerIds: Set<Int> = [],
            sconvKernelSize: Int = 4,
            layersPerAttnResBlock: Int? = nil,
            attnResNormalizeValues: Bool = false
        ) {
            self.hasSink = hasSink
            self.dRel = dRel
            self.relExtent = relExtent
            self.slidingWindowSize = slidingWindowSize
            self.swaQHeads = swaQHeads
            self.swaKVHeads = swaKVHeads
            self.swaHeadDim = swaHeadDim
            self.logScalingFloor = logScalingFloor
            self.logScalingAlpha = logScalingAlpha
            self.qkL2ClipThreshold = qkL2ClipThreshold
            self.relL2ClipThreshold = relL2ClipThreshold
            self.valueNormNoGamma = valueNormNoGamma
            self.finalLogitSoftcapping = finalLogitSoftcapping
            self.localLayerIds = localLayerIds
            self.sconvKernelSize = sconvKernelSize
            self.layersPerAttnResBlock = layersPerAttnResBlock
            self.attnResNormalizeValues = attnResNormalizeValues
        }

        public func isSlidingLayer(_ layer: Int) -> Bool {
            localLayerIds.contains(layer)
        }
    }

    /// Group-limited MoE routing and shared-expert gating configuration.
    public struct MoERoutingConfig: Sendable, Hashable, Codable, Equatable {
        public let nGroup: Int
        public let topkGroup: Int
        public let groupTopM: Int
        public let sharedExpertHasGate: Bool?
        public let scoreFunction: String
        public let enableExpertBias: Bool
        public let routedScalingFactor: Float
        public let routedScalingFactorLearnable: Bool
        public let normTopKProb: Bool

        public init(
            nGroup: Int = 1,
            topkGroup: Int = 1,
            groupTopM: Int = 2,
            sharedExpertHasGate: Bool? = nil,
            scoreFunction: String = "softmax",
            enableExpertBias: Bool = false,
            routedScalingFactor: Float = 1.0,
            routedScalingFactorLearnable: Bool = false,
            normTopKProb: Bool = false
        ) {
            self.nGroup = max(1, nGroup)
            self.topkGroup = max(1, min(self.nGroup, topkGroup))
            self.groupTopM = max(1, min(2, groupTopM))
            self.sharedExpertHasGate = sharedExpertHasGate
            self.scoreFunction = scoreFunction
            self.enableExpertBias = enableExpertBias
            self.routedScalingFactor = routedScalingFactor
            self.routedScalingFactorLearnable = routedScalingFactorLearnable
            self.normTopKProb = normTopKProb
        }
    }

    /// Per-layer and dense SwiGLU clamping configuration.
    public struct SwiGLUClampConfig: Sendable, Hashable, Codable, Equatable {
        public enum Mode: String, Sendable, Hashable, Codable, Equatable {
            case preSilu
            case postSilu
        }

        public let mode: Mode
        public let denseLimit: Float?
        public let layerRoutedLimits: [Float]
        public let layerSharedLimits: [Float]

        public init(
            mode: Mode = .preSilu,
            denseLimit: Float? = nil,
            layerRoutedLimits: [Float] = [],
            layerSharedLimits: [Float] = []
        ) {
            self.mode = mode
            self.denseLimit = denseLimit
            self.layerRoutedLimits = layerRoutedLimits
            self.layerSharedLimits = layerSharedLimits
        }

        public func routedSpec(layer: Int) -> QWSwiGLUClampSpec? {
            let lim = (0..<layerRoutedLimits.count).contains(layer) ? layerRoutedLimits[layer] : (denseLimit ?? 0)
            guard lim > 0 else { return nil }
            return QWSwiGLUClampSpec(limit: lim, postSilu: mode == .postSilu)
        }

        public func sharedSpec(layer: Int) -> QWSwiGLUClampSpec? {
            let lim = (0..<layerSharedLimits.count).contains(layer) ? layerSharedLimits[layer] : (denseLimit ?? 0)
            guard lim > 0 else { return nil }
            return QWSwiGLUClampSpec(limit: lim, postSilu: mode == .postSilu)
        }

        public func distinctSpecs() -> [QWSwiGLUClampSpec] {
            var set = Set<QWSwiGLUClampSpec>()
            let isPost = (mode == .postSilu)
            if let d = denseLimit, d > 0 {
                set.insert(QWSwiGLUClampSpec(limit: d, postSilu: isPost))
            }
            for v in layerRoutedLimits where v > 0 {
                set.insert(QWSwiGLUClampSpec(limit: v, postSilu: isPost))
            }
            for v in layerSharedLimits where v > 0 {
                set.insert(QWSwiGLUClampSpec(limit: v, postSilu: isPost))
            }
            return set.sorted { ($0.limit, $0.postSilu ? 1 : 0) < ($1.limit, $1.postSilu ? 1 : 0) }
        }
    }

    /// Native Multi-Token Prediction (MTP) fusion and layer layout configuration.
    public struct MTPConfig: Sendable, Hashable, Codable, Equatable {
        public enum FusionMode: String, Sendable, Hashable, Codable, Equatable {
            case concatEHProj
            case splitEProjHProj
            case sharedLastLayerEcho
        }

        public let layers: Int
        public let fusionMode: FusionMode
        public let useDedicatedEmbeddings: Bool
        public let hasFinalNorm: Bool
        public let isMoE: Bool
        public let intermediateSize: Int?
        public let attentionIsSparse: Bool
        public let attentionIsLinear: Bool
        public let sharedEmbedAndHead: Bool

        public init(
            layers: Int,
            fusionMode: FusionMode,
            useDedicatedEmbeddings: Bool = false,
            hasFinalNorm: Bool = false,
            isMoE: Bool = false,
            intermediateSize: Int? = nil,
            attentionIsSparse: Bool = false,
            attentionIsLinear: Bool = false,
            sharedEmbedAndHead: Bool = true
        ) {
            self.layers = layers
            self.fusionMode = fusionMode
            self.useDedicatedEmbeddings = useDedicatedEmbeddings
            self.hasFinalNorm = hasFinalNorm
            self.isMoE = isMoE
            self.intermediateSize = intermediateSize
            self.attentionIsSparse = attentionIsSparse
            self.attentionIsLinear = attentionIsLinear
            self.sharedEmbedAndHead = sharedEmbedAndHead
        }
    }

    /// Vision tower and multimodal projector configuration.
    public struct VisionTowerConfig: Sendable, Hashable, Codable, Equatable {
        public enum PatchEmbedKind: String, Sendable, Hashable, Codable, Equatable {
            case conv3d
            case unfoldedPatchProj
            case lfmHybridConv
            case siglip2NaFlex
        }

        public let depth: Int
        public let hidden: Int
        public let heads: Int
        public let intermediate: Int
        public let patch: Int
        public let temporal: Int
        public let merge: Int
        public let numPosEmbeds: Int
        public let outDim: Int
        public let mergerIntermediateDim: Int?
        public let projectorBias: Bool
        public let normEps: Float
        public let ropeTheta: Double
        public let mropeLayout: MRoPELayout
        public let patchEmbedKind: PatchEmbedKind
        public let imageMean: [Float]
        public let imageStd: [Float]
        public let disableInternalMergerProj: Bool
        public let kvHeads: Int?
        public let layerTypes: [String]?
        public let convKernelSize: Int?
        public let projectorNormEps: Float

        public init(
            depth: Int,
            hidden: Int,
            heads: Int,
            intermediate: Int,
            patch: Int,
            temporal: Int,
            merge: Int,
            numPosEmbeds: Int,
            outDim: Int,
            mergerIntermediateDim: Int? = nil,
            projectorBias: Bool = true,
            normEps: Float = 1e-6,
            ropeTheta: Double = 10_000.0,
            mropeLayout: MRoPELayout = .qwenInterleaved,
            patchEmbedKind: PatchEmbedKind = .conv3d,
            imageMean: [Float] = [0.5, 0.5, 0.5],
            imageStd: [Float] = [0.5, 0.5, 0.5],
            disableInternalMergerProj: Bool = false,
            kvHeads: Int? = nil,
            layerTypes: [String]? = nil,
            convKernelSize: Int? = nil,
            projectorNormEps: Float = 1e-6
        ) {
            self.depth = depth
            self.hidden = hidden
            self.heads = heads
            self.intermediate = intermediate
            self.patch = patch
            self.temporal = temporal
            self.merge = merge
            self.numPosEmbeds = numPosEmbeds
            self.outDim = outDim
            self.mergerIntermediateDim = mergerIntermediateDim
            self.projectorBias = projectorBias
            self.normEps = normEps
            self.ropeTheta = ropeTheta
            self.mropeLayout = mropeLayout
            self.patchEmbedKind = patchEmbedKind
            self.imageMean = imageMean
            self.imageStd = imageStd
            self.disableInternalMergerProj = disableInternalMergerProj
            self.kvHeads = kvHeads
            self.layerTypes = layerTypes
            self.convKernelSize = convKernelSize
            self.projectorNormEps = projectorNormEps
        }
    }

    /// Top-level runtime stream and loop topology derived from config.
    public enum ExecutionTopology: Sendable, Hashable, Equatable {
        case standardSingleStream
        case sinkhornMHC(MHCConfig)
        case hyperStreamPLE(hcCount: Int, hcLowrank: Int?)
        case recurrentLoop(LoopSettings, inkling2: Inkling2Config?)
    }

    /// Resolved execution topology inferred strictly from decoded traits.
    public var executionTopology: ExecutionTopology {
        if let loop = loopSettings {
            return .recurrentLoop(loop, inkling2: inkling2Config)
        }
        if let mhc = mhcConfig {
            return .sinkhornMHC(mhc)
        }
        if let hc = hcCount, hc > 0 {
            return .hyperStreamPLE(hcCount: hc, hcLowrank: hcLowrank)
        }
        return .standardSingleStream
    }

    /// Resolved special token IDs for chat template framing, reasoning tags, and stop sets.
    public var specialTokens: QWModelSpecialTokens {
        var stops = eosTokenIds.filter { $0 >= 0 && $0 < vocab }
        if let spec = QWModelFamilyRegistry.shared.spec(forModelType: modelType),
           let defaults = spec.defaultSpecialTokens {
            for s in defaults.extraStopTokenIds where s >= 0 && s < vocab && !stops.contains(s) {
                stops.append(s)
            }
            if defaults.usesChatTemplateIds, defaults.imEndTokenId >= 0, defaults.imEndTokenId < vocab,
               !stops.contains(defaults.imEndTokenId) {
                stops.append(defaults.imEndTokenId)
            }
            return QWModelSpecialTokens(
                eosTokenId: eosTokenId,
                imStartTokenId: defaults.imStartTokenId,
                imEndTokenId: defaults.imEndTokenId,
                thinkTokenId: defaults.thinkTokenId,
                endThinkTokenId: defaults.endThinkTokenId,
                extraStopTokenIds: stops,
                usesChatTemplateIds: defaults.usesChatTemplateIds
            )
        }
        return QWModelSpecialTokens(
            eosTokenId: eosTokenId,
            extraStopTokenIds: stops,
            usesChatTemplateIds: usesChatTemplateIds
        )
    }

    /// Returns a copy with vision mean/std overridden (e.g. from `preprocessor_config.json`).
    public func withVisionNormalization(mean: [Float]?, std: [Float]?) -> QWModelDescriptor {
        guard let vt = visionTowerConfig else { return self }
        let nextMean = (mean?.count == 3) ? mean! : vt.imageMean
        let nextStd = (std?.count == 3) ? std! : vt.imageStd
        if nextMean == vt.imageMean && nextStd == vt.imageStd { return self }
        let updatedVT = VisionTowerConfig(
            depth: vt.depth, hidden: vt.hidden, heads: vt.heads, intermediate: vt.intermediate,
            patch: vt.patch, temporal: vt.temporal, merge: vt.merge, numPosEmbeds: vt.numPosEmbeds,
            outDim: vt.outDim, mergerIntermediateDim: vt.mergerIntermediateDim,
            projectorBias: vt.projectorBias, normEps: vt.normEps, ropeTheta: vt.ropeTheta,
            mropeLayout: vt.mropeLayout, patchEmbedKind: vt.patchEmbedKind,
            imageMean: nextMean, imageStd: nextStd,
            disableInternalMergerProj: vt.disableInternalMergerProj,
            kvHeads: vt.kvHeads,
            layerTypes: vt.layerTypes,
            convKernelSize: vt.convKernelSize
        )
        return QWModelDescriptor(
            family: family, modelType: modelType, capabilities: capabilities,
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
            usesPlusOneRMSNorm: usesPlusOneRMSNorm, quantization: quantization,
            hasMropeSection: hasMropeSection, tieWordEmbeddings: tieWordEmbeddings,
            normTopKProb: normTopKProb, eosTokenIds: eosTokenIds, envelopeModelType: envelopeModelType,
            loopSettings: loopSettings, mhcConfig: mhcConfig, firstKDenseReplace: firstKDenseReplace,
            mlpOnlyLayers: mlpOnlyLayers, decoderSparseStep: decoderSparseStep, moeLayerFreq: moeLayerFreq,
            routedScalingFactor: routedScalingFactor, swigluLimit: swigluLimit, topkMethod: topkMethod,
            residentExpertsPerLayer: residentExpertsPerLayer, sharedExpertHasGate: sharedExpertHasGate,
            mlaUseNoPE: mlaUseNoPE, kvLoraRank: kvLoraRank, qLoraRank: qLoraRank,
            qkNopeHeadDim: qkNopeHeadDim, qkRopeHeadDim: qkRopeHeadDim, vHeadDim: vHeadDim,
            indexKPoolCompress: indexKPoolCompress, indexKPoolAlwaysSelectTail: indexKPoolAlwaysSelectTail,
            indexShareForMTP: indexShareForMTP, indexerRopeInterleave: indexerRopeInterleave,
            gdnVectorDecay: gdnVectorDecay, gdnLowRankDim: gdnLowRankDim,
            gdnGateLowerBound: gdnGateLowerBound, gdnSeparateConv: gdnSeparateConv,
            hcaConfig: hcaConfig, hybridMLAConfig: hybridMLAConfig, blockSparseGQAConfig: blockSparseGQAConfig,
            yarnRopeConfig: yarnRopeConfig, ropeScaling: ropeScaling, slidingWindow: slidingWindow,
            layerSlidingWindows: layerSlidingWindows, activationKind: activationKind,
            moeLatentSize: moeLatentSize, latentMoeUseNorm: latentMoeUseNorm,
            moeSplitFeatureRouting: moeSplitFeatureRouting,
            moeRouterEnableExpertBias: moeRouterEnableExpertBias,
            routedScalingFactorLearnable: routedScalingFactorLearnable,
            multimaxHead: multimaxHead, mtpSharedLastLayer: mtpSharedLastLayer,
            inkling2Config: inkling2Config, dsparkConfig: dsparkConfig,
            mlaConfig: mlaConfig, kdaConfig: kdaConfig,
            attentionSinkRelConfig: attentionSinkRelConfig, moeRoutingConfig: moeRoutingConfig,
            swigluClampConfig: swigluClampConfig, mtpConfig: mtpConfig,
            visionTowerConfig: updatedVT, ropeInterleave: ropeInterleave, role: role,
            shortConvConfig: shortConvConfig, normKind: normKind,
            useSandwichNorm: useSandwichNorm, perKindRoPEConfig: perKindRoPEConfig,
            attnGateMode: attnGateMode, decisionConfig: decisionConfig,
            gatedNormConfig: gatedNormConfig, gdnGatingConfig: gdnGatingConfig,
            finalLogitSoftcapping: finalLogitSoftcapping, v2DigestHeadDim: v2DigestHeadDim
        )
    }

    /// Whether the pack carries Qwen3.8 chat template special IDs inside vocabulary range.
    public var usesChatTemplateIds: Bool {
        if let spec = QWModelFamilyRegistry.shared.spec(forModelType: modelType) {
            return spec.usesQwenChatTemplateIds && vocab > 248046
        }
        return [.qwen38Dense, .qwen38MoE, .qwen38FlashNext, .senseNovaLoop].contains(family) && vocab > 248046
    }
}
