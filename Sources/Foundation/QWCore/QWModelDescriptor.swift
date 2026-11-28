import Foundation
import CryptoKit

/// Execution role of a decoded model pack.
public enum QWModelRole: String, Sendable, Hashable, CaseIterable, Codable {
    case causalLM
    case multimodalConditionalLM
    case speculativeDrafter
    case decisionEncoder
}

/// C01/§2 — Immutable model identity + geometry for ANY supported model.
public struct QWModelDescriptor: Sendable, Hashable {

    public enum Family: String, Sendable, Hashable, CaseIterable, Codable {
        case qwen38Dense = "qwen3.8-27b"          // control adapter
        case qwen38MoE = "qwen3.8-moe"
        case qwen38FlashNext = "qwen3.8-flash-next"
        case senseNovaLoop = "sensenova_loop"
        case glm5Next = "glm5_next"
        case ernie55MoE = "ernie5_5_moe"
        case maxwellMoE = "maxwell_moe"
        case qwen3Dense = "qwen3-dense"            // standard HF/MLX, dense MLP
        case qwen3MoE = "qwen3-moe"                // standard HF/MLX, sparse MoE
        case qwen2Dense = "qwen2-dense"
        case llamaDense = "llama-dense"
        case mistralDense = "mistral-dense"
        case inkling2Loop = "inkling2_loop"
        case bailingMoeV3 = "bailing_moe_v3"
        case lfm2 = "lfm2"
        case lfm3 = "lfm3"
        case lfm2Encoder = "lfm2_encoder"
        case modernBert = "modernbert"
        case gigachat4 = "gigachat4"
    }

    public enum Capability: String, Sendable, Hashable, CaseIterable, Codable {
        case vision, moe, ple, qsaIndexer, mtp, hyperConnections, decision
    }

    /// Looped block execution configuration (e.g., `SenseNova-Loop-9B-Preview`).
    public struct LoopSettings: Sendable, Hashable, Codable, Equatable {
        public let start: Int
        public let end: Int
        public let evalCount: Int
        public let counts: [Int]?

        public init(start: Int, end: Int, evalCount: Int, counts: [Int]? = nil) {
            self.start = start
            self.end = end
            self.evalCount = evalCount
            self.counts = counts
        }
    }

    /// Multi-stream Hyper-Connections (mHC) configuration with Sinkhorn-Knopp doubly-stochastic mixing.
    public struct MHCConfig: Sendable, Hashable, Codable, Equatable {
        public let start: Int
        public let end: Int
        public let streams: Int
        public let iterations: Int
        public let fixedIdentity: Bool
        public let preScale: Float
        public let hcEps: Float?
        public let hasExitHead: Bool

        public init(start: Int, end: Int, streams: Int = 2, iterations: Int = 20,
                    fixedIdentity: Bool = false, preScale: Float = 1.0,
                    hcEps: Float? = nil, hasExitHead: Bool = false) {
            self.start = start
            self.end = end
            self.streams = streams
            self.iterations = iterations
            self.fixedIdentity = fixedIdentity
            self.preScale = preScale
            self.hcEps = hcEps
            self.hasExitHead = hasExitHead
        }
    }

    /// RoPE context-scaling specification decoded from HF `rope_scaling` / `rope_parameters`.
    public enum RoPEScaling: Sendable, Hashable, Codable, Equatable {
        case yarn(YarnRopeConfig)
        case linear(factor: Double)
        case dynamic(factor: Double, originalMaxPositionEmbeddings: Int?)
    }

    public let family: Family
    public let role: QWModelRole
    public let modelType: String                  // HF `model_type`
    public let envelopeModelType: String?         // Top-level `model_type` when wrapped (e.g. "sensenova_loop")
    public let capabilities: Set<Capability>
    public let loopSettings: LoopSettings?
    public let mhcConfig: MHCConfig?
    public let hcaConfig: HCAConfig?
    public let hybridMLAConfig: HybridMLAConfig?
    public let blockSparseGQAConfig: BlockSparseGQAConfig?
    public let yarnRopeConfig: YarnRopeConfig?
    public let ropeScaling: RoPEScaling?
    public let slidingWindow: Int?
    public let layerSlidingWindows: [Int]?
    public let activationKind: ActivationKind
    public let moeLatentSize: Int?
    public let latentMoeUseNorm: Bool
    public let moeSplitFeatureRouting: Bool
    public let moeRouterEnableExpertBias: Bool
    public let routedScalingFactorLearnable: Bool
    public let multimaxHead: Bool
    public let mtpSharedLastLayer: Bool
    public let inkling2Config: Inkling2Config?
    public let dsparkConfig: DSparkConfig?

    public let mlaConfig: MLAConfig?
    public let kdaConfig: KDAConfig?
    public let attentionSinkRelConfig: AttentionSinkRelConfig?
    public let moeRoutingConfig: MoERoutingConfig?
    public let swigluClampConfig: SwiGLUClampConfig?
    public let mtpConfig: MTPConfig?
    public let visionTowerConfig: VisionTowerConfig?
    public let ropeInterleave: Bool
    public let shortConvConfig: ShortConvConfig?
    public let normKind: NormKind?
    public let useSandwichNorm: Bool
    public let perKindRoPEConfig: PerKindRoPEConfig?
    public let attnGateMode: AttnGateMode
    public let decisionConfig: QWDecisionModelConfig?
    public let gatedNormConfig: GatedNormConfig?
    public let gdnGatingConfig: GDNGatingConfig?
    public let finalLogitSoftcapping: Float?
    internal let v2DigestHeadDim: Int

    public let hidden: Int
    public let vocab: Int
    public let denseIntermediate: Int?           // dense MLP intermediate
    public let numLayers: Int
    public let layerTypes: [String]               // authoritative per-layer kind
    public let fullAttentionInterval: Int
    public let rmsEps: Float
    public let maxPositionEmbeddings: Int

    // Sparse attention (QSA on Flash-Next; dense GQA on 27B; NoPE MLA + Key-Pool on GLM-5.3)
    public let qHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let attnOutputGate: Bool               // Q‖gate packing in q_proj
    public let partialRotaryFactor: Double
    public let ropeTheta: Double
    public let indexerHeads: Int?                 // QSA / Key-Pool indexer heads (nil = no indexer)
    public let indexerKvHeads: Int?
    public let indexerHeadDim: Int?
    public let indexerBudget: Int?
    public let indexerCompressRatio: Int?

    // NoPE Multi-head Latent Attention (MLA)
    public let mlaUseNoPE: Bool
    public let kvLoraRank: Int?
    public let qLoraRank: Int?
    public let qkNopeHeadDim: Int?
    public let qkRopeHeadDim: Int?
    public let vHeadDim: Int?

    // Key-Pool(4) Indexer & LSA
    public let indexKPoolCompress: Bool
    public let indexKPoolAlwaysSelectTail: Bool
    public let indexShareForMTP: Bool
    public let indexerRopeInterleave: Bool

    // MoE (nil = dense MLP)
    public let numExperts: Int?
    public let expertsPerTok: Int?
    public let moeIntermediate: Int?
    public let sharedExpertIntermediate: Int?
    public let sharedExpertHasGate: Bool
    public let expertGateType: String?            // "sigmoid" on FN / GLM-5.3
    public let firstKDenseReplace: Int
    public let mlpOnlyLayers: Set<Int>
    public let decoderSparseStep: Int
    public let moeLayerFreq: Int
    public let routedScalingFactor: Float
    public let swigluLimit: Float?
    public let topkMethod: String?
    public var residentExpertsPerLayer: Int?

    // Hyper-connections (4-lane gated residual stream)
    public let hcCount: Int?
    public let hcLowrank: Int?

    // GDN / KDA Linear Attention
    public let gdnQKHeads: Int
    public let gdnVHeads: Int
    public let gdnHeadDim: Int
    public let gdnQKVChannels: Int
    public let gdnZChannels: Int
    public let gdnConvKernel: Int
    public let gdnVectorDecay: Bool
    public let gdnLowRankDim: Int?
    public let gdnGateLowerBound: Float?
    public let gdnSeparateConv: Bool

    // PLE (nil = no PLE layer)
    public let pleLayerIds: [Int]?                // ONE-indexed (HF validator law)
    public let pleEmbedDim: Int?
    public let pleConvKernel: Int?
    public let ngramSize: Int?
    public let headsPerNgram: Int?
    public let ngramVocabSizeBase: Int?
    public let makeNgramVocabDivisibleBy: Int?
    public let splitNgramParts: Int?

    // MTP head
    public let mtpLayers: Int
    public let mtpUseDedicatedEmbeddings: Bool

    // Special tokens (tokenizer/asset identity, part of geometry binding)
    public let eosTokenId: Int
    public let mropeW: Int
    public let mropeH: Int
    public let mropeT: Int
    public let hasQKNorm: Bool
    public let hasQKVBias: Bool
    public let hasAttnOutputBias: Bool
    public let usesPlusOneRMSNorm: Bool
    public let quantization: QWQuantizationSpec?
    public let hasMropeSection: Bool
    public let tieWordEmbeddings: Bool
    public let normTopKProb: Bool
    public let eosTokenIds: [Int]
    public var mropeSection: (w: Int, h: Int, t: Int) { (mropeW, mropeH, mropeT) }

    // Vision tower (optional in config; defaults to Qwen3-VL tower geometry)
    public let visDepth: Int
    public let visHidden: Int
    public let visHeads: Int
    public let visIntermediate: Int
    public let visPatch: Int
    public let visTemporal: Int
    public let visMerge: Int
    public let visPosEmbeds: Int
    public let visOut: Int
    public let imageTokenId: Int
    public let videoTokenId: Int
    public let visionStartTokenId: Int
    public let visionEndTokenId: Int

    public init(family: Family, modelType: String, capabilities: Set<Capability>,
                hidden: Int, vocab: Int, denseIntermediate: Int?, numLayers: Int,
                layerTypes: [String], fullAttentionInterval: Int, rmsEps: Float,
                maxPositionEmbeddings: Int, qHeads: Int, kvHeads: Int, headDim: Int,
                attnOutputGate: Bool, partialRotaryFactor: Double, ropeTheta: Double,
                indexerHeads: Int?, indexerKvHeads: Int?, indexerHeadDim: Int?,
                indexerBudget: Int?, indexerCompressRatio: Int?,
                numExperts: Int?, expertsPerTok: Int?, moeIntermediate: Int?,
                sharedExpertIntermediate: Int?, expertGateType: String?,
                hcCount: Int?, hcLowrank: Int?,
                gdnQKHeads: Int, gdnVHeads: Int, gdnHeadDim: Int,
                gdnQKVChannels: Int, gdnZChannels: Int, gdnConvKernel: Int,
                pleLayerIds: [Int]?, pleEmbedDim: Int?, pleConvKernel: Int?,
                ngramSize: Int?, headsPerNgram: Int?, ngramVocabSizeBase: Int?,
                makeNgramVocabDivisibleBy: Int?, splitNgramParts: Int?,
                mtpLayers: Int, mtpUseDedicatedEmbeddings: Bool, eosTokenId: Int,
                mropeSection: (w: Int, h: Int, t: Int) = (11, 11, 10),
                visDepth: Int = 27, visHidden: Int = 1152, visHeads: Int = 16,
                visIntermediate: Int = 4304, visPatch: Int = 16, visTemporal: Int = 2,
                visMerge: Int = 2, visPosEmbeds: Int = 2304, visOut: Int? = nil,
                imageTokenId: Int = 248056, videoTokenId: Int = 248057,
                visionStartTokenId: Int = 248053, visionEndTokenId: Int = 248054,
                hasQKNorm: Bool = false,
                hasQKVBias: Bool = false,
                hasAttnOutputBias: Bool = false,
                usesPlusOneRMSNorm: Bool? = nil,
                quantization: QWQuantizationSpec? = nil,
                hasMropeSection: Bool = true,
                tieWordEmbeddings: Bool = false,
                normTopKProb: Bool = false,
                eosTokenIds: [Int] = [],
                envelopeModelType: String? = nil,
                loopSettings: LoopSettings? = nil,
                mhcConfig: MHCConfig? = nil,
                firstKDenseReplace: Int = 0,
                mlpOnlyLayers: Set<Int> = [],
                decoderSparseStep: Int = 1,
                moeLayerFreq: Int = 1,
                routedScalingFactor: Float = 1.0,
                swigluLimit: Float? = nil,
                topkMethod: String? = nil,
                residentExpertsPerLayer: Int? = nil,
                sharedExpertHasGate: Bool = true,
                mlaUseNoPE: Bool = false,
                kvLoraRank: Int? = nil,
                qLoraRank: Int? = nil,
                qkNopeHeadDim: Int? = nil,
                qkRopeHeadDim: Int? = nil,
                vHeadDim: Int? = nil,
                indexKPoolCompress: Bool = false,
                indexKPoolAlwaysSelectTail: Bool = false,
                indexShareForMTP: Bool = false,
                indexerRopeInterleave: Bool = false,
                gdnVectorDecay: Bool = false,
                gdnLowRankDim: Int? = nil,
                gdnGateLowerBound: Float? = nil,
                gdnSeparateConv: Bool = false,
                hcaConfig: HCAConfig? = nil,
                hybridMLAConfig: HybridMLAConfig? = nil,
                blockSparseGQAConfig: BlockSparseGQAConfig? = nil,
                yarnRopeConfig: YarnRopeConfig? = nil,
                ropeScaling: RoPEScaling? = nil,
                slidingWindow: Int? = nil,
                layerSlidingWindows: [Int]? = nil,
                activationKind: ActivationKind = .silu,
                moeLatentSize: Int? = nil,
                latentMoeUseNorm: Bool = false,
                moeSplitFeatureRouting: Bool = false,
                moeRouterEnableExpertBias: Bool = false,
                routedScalingFactorLearnable: Bool = false,
                multimaxHead: Bool = false,
                mtpSharedLastLayer: Bool = false,
                inkling2Config: Inkling2Config? = nil,
                dsparkConfig: DSparkConfig? = nil,
                mlaConfig: MLAConfig? = nil,
                kdaConfig: KDAConfig? = nil,
                attentionSinkRelConfig: AttentionSinkRelConfig? = nil,
                moeRoutingConfig: MoERoutingConfig? = nil,
                swigluClampConfig: SwiGLUClampConfig? = nil,
                mtpConfig: MTPConfig? = nil,
                visionTowerConfig: VisionTowerConfig? = nil,
                ropeInterleave: Bool? = nil,
                role: QWModelRole? = nil,
                shortConvConfig: ShortConvConfig? = nil,
                normKind: NormKind? = nil,
                useSandwichNorm: Bool = false,
                perKindRoPEConfig: PerKindRoPEConfig? = nil,
                attnGateMode: AttnGateMode? = nil,
                decisionConfig: QWDecisionModelConfig? = nil,
                gatedNormConfig: GatedNormConfig? = nil,
                gdnGatingConfig: GDNGatingConfig? = nil,
                finalLogitSoftcapping: Float? = nil,
                v2DigestHeadDim: Int? = nil) {
        self.family = family; self.modelType = modelType
        self.role = role ?? (capabilities.contains(.vision) ? .multimodalConditionalLM : .causalLM)
        self.shortConvConfig = shortConvConfig
        self.normKind = normKind
        self.useSandwichNorm = useSandwichNorm
        self.perKindRoPEConfig = perKindRoPEConfig
        self.attnGateMode = attnGateMode ?? (attnOutputGate ? .packedElementWiseQProj : .none)
        self.decisionConfig = decisionConfig
        self.gatedNormConfig = gatedNormConfig
        self.gdnGatingConfig = gdnGatingConfig
        self.finalLogitSoftcapping = finalLogitSoftcapping
        self.v2DigestHeadDim = v2DigestHeadDim ?? headDim
        self.inkling2Config = inkling2Config
        self.dsparkConfig = dsparkConfig
        self.envelopeModelType = envelopeModelType
        self.capabilities = capabilities
        self.loopSettings = loopSettings
        self.mhcConfig = mhcConfig
        self.hcaConfig = hcaConfig
        self.hybridMLAConfig = hybridMLAConfig
        self.blockSparseGQAConfig = blockSparseGQAConfig
        if let yarnRopeConfig {
            self.yarnRopeConfig = yarnRopeConfig
        } else if hybridMLAConfig == nil, hcaConfig == nil, case .yarn(let y) = ropeScaling {
            self.yarnRopeConfig = y
        } else {
            self.yarnRopeConfig = nil
        }
        self.ropeScaling = ropeScaling ?? yarnRopeConfig.map { .yarn($0) }
        self.slidingWindow = slidingWindow.flatMap { $0 > 0 ? $0 : nil }
        self.layerSlidingWindows = layerSlidingWindows
        self.activationKind = activationKind
        self.moeLatentSize = moeLatentSize
        self.latentMoeUseNorm = latentMoeUseNorm
        self.moeSplitFeatureRouting = moeSplitFeatureRouting
        self.moeRouterEnableExpertBias = moeRouterEnableExpertBias
        self.routedScalingFactorLearnable = routedScalingFactorLearnable
        self.multimaxHead = multimaxHead
        self.mtpSharedLastLayer = mtpSharedLastLayer
        self.hidden = hidden; self.vocab = vocab; self.denseIntermediate = denseIntermediate
        self.numLayers = numLayers; self.layerTypes = layerTypes
        self.fullAttentionInterval = fullAttentionInterval; self.rmsEps = rmsEps
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.qHeads = qHeads; self.kvHeads = kvHeads; self.headDim = headDim
        self.attnOutputGate = attnOutputGate
        self.partialRotaryFactor = partialRotaryFactor; self.ropeTheta = ropeTheta
        self.indexerHeads = indexerHeads; self.indexerKvHeads = indexerKvHeads
        self.indexerHeadDim = indexerHeadDim; self.indexerBudget = indexerBudget; self.indexerCompressRatio = indexerCompressRatio
        self.mlaUseNoPE = mlaUseNoPE; self.kvLoraRank = kvLoraRank; self.qLoraRank = qLoraRank
        self.qkNopeHeadDim = qkNopeHeadDim; self.qkRopeHeadDim = qkRopeHeadDim; self.vHeadDim = vHeadDim
        self.indexKPoolCompress = indexKPoolCompress
        self.indexKPoolAlwaysSelectTail = indexKPoolAlwaysSelectTail
        self.indexShareForMTP = indexShareForMTP
        self.indexerRopeInterleave = indexerRopeInterleave
        self.numExperts = numExperts; self.expertsPerTok = expertsPerTok; self.moeIntermediate = moeIntermediate
        self.sharedExpertIntermediate = sharedExpertIntermediate; self.sharedExpertHasGate = sharedExpertHasGate
        self.expertGateType = expertGateType
        self.firstKDenseReplace = firstKDenseReplace
        self.mlpOnlyLayers = mlpOnlyLayers
        self.decoderSparseStep = max(1, decoderSparseStep)
        self.moeLayerFreq = moeLayerFreq
        self.routedScalingFactor = routedScalingFactor; self.swigluLimit = swigluLimit
        self.topkMethod = topkMethod; self.residentExpertsPerLayer = residentExpertsPerLayer
        self.hcCount = hcCount; self.hcLowrank = hcLowrank
        self.gdnQKHeads = gdnQKHeads; self.gdnVHeads = gdnVHeads; self.gdnHeadDim = gdnHeadDim
        self.gdnQKVChannels = gdnQKVChannels; self.gdnZChannels = gdnZChannels; self.gdnConvKernel = gdnConvKernel
        self.gdnVectorDecay = gdnVectorDecay; self.gdnLowRankDim = gdnLowRankDim
        self.gdnGateLowerBound = gdnGateLowerBound; self.gdnSeparateConv = gdnSeparateConv
        self.pleLayerIds = pleLayerIds; self.pleEmbedDim = pleEmbedDim; self.pleConvKernel = pleConvKernel
        self.ngramSize = ngramSize; self.headsPerNgram = headsPerNgram; self.ngramVocabSizeBase = ngramVocabSizeBase
        self.makeNgramVocabDivisibleBy = makeNgramVocabDivisibleBy; self.splitNgramParts = splitNgramParts
        self.mtpLayers = mtpLayers; self.mtpUseDedicatedEmbeddings = mtpUseDedicatedEmbeddings; self.eosTokenId = eosTokenId
        self.mropeW = mropeSection.w; self.mropeH = mropeSection.h; self.mropeT = mropeSection.t
        self.visDepth = visDepth; self.visHidden = visHidden; self.visHeads = visHeads
        self.visIntermediate = visIntermediate; self.visPatch = visPatch; self.visTemporal = visTemporal
        self.visMerge = visMerge; self.visPosEmbeds = visPosEmbeds; self.visOut = visOut ?? hidden
        self.imageTokenId = imageTokenId; self.videoTokenId = videoTokenId
        self.visionStartTokenId = visionStartTokenId; self.visionEndTokenId = visionEndTokenId
        self.hasQKNorm = hasQKNorm
        self.hasQKVBias = hasQKVBias
        self.hasAttnOutputBias = hasAttnOutputBias
        let defaultPlusOne = QWModelFamilyRegistry.shared.spec(forModelType: modelType)?.usesPlusOneRMSNorm
            ?? [.qwen38Dense, .qwen38MoE, .senseNovaLoop].contains(family)
        self.usesPlusOneRMSNorm = usesPlusOneRMSNorm ?? defaultPlusOne
        self.quantization = quantization; self.hasMropeSection = hasMropeSection
        self.tieWordEmbeddings = tieWordEmbeddings; self.normTopKProb = normTopKProb
        self.eosTokenIds = eosTokenIds.isEmpty ? [eosTokenId] : eosTokenIds
        let defInterleave = QWModelFamilyRegistry.shared.spec(forModelType: modelType)?.defaultRopeInterleave ?? false
        let resolvedInterleave = ropeInterleave ?? (hybridMLAConfig?.ropeInterleave ?? (indexerRopeInterleave || defInterleave))
        self.ropeInterleave = resolvedInterleave
        let effSwLimit = swigluLimit
        let synth = Self.synthesizeCapabilityBundle(
            family: family, modelType: modelType, capabilities: capabilities,
            qHeads: qHeads, headDim: headDim, attnOutputGate: attnOutputGate,
            ropeTheta: ropeTheta, ropeInterleave: resolvedInterleave,
            indexerHeads: indexerHeads, indexKPoolCompress: indexKPoolCompress, mlaUseNoPE: mlaUseNoPE,
            kvLoraRank: kvLoraRank, qLoraRank: qLoraRank, qkNopeHeadDim: qkNopeHeadDim,
            qkRopeHeadDim: qkRopeHeadDim, vHeadDim: vHeadDim,
            hybridMLAConfig: hybridMLAConfig, yarnRopeConfig: self.yarnRopeConfig,
            numExperts: numExperts, denseIntermediate: denseIntermediate, sharedExpertHasGate: sharedExpertHasGate,
            expertGateType: expertGateType, topkMethod: topkMethod,
            moeRouterEnableExpertBias: moeRouterEnableExpertBias, routedScalingFactor: routedScalingFactor,
            routedScalingFactorLearnable: routedScalingFactorLearnable, normTopKProb: normTopKProb,
            swigluLimit: effSwLimit, mtpLayers: mtpLayers, mtpUseDedicatedEmbeddings: mtpUseDedicatedEmbeddings,
            mtpSharedLastLayer: mtpSharedLastLayer, mhcConfig: mhcConfig,
            gdnQKHeads: gdnQKHeads, gdnVHeads: gdnVHeads, gdnHeadDim: gdnHeadDim, gdnConvKernel: gdnConvKernel,
            gdnVectorDecay: gdnVectorDecay, gdnLowRankDim: gdnLowRankDim,
            gdnGateLowerBound: gdnGateLowerBound, gdnSeparateConv: gdnSeparateConv,
            visDepth: visDepth, visHidden: visHidden, visHeads: visHeads, visIntermediate: visIntermediate,
            visPatch: visPatch, visTemporal: visTemporal, visMerge: visMerge, visPosEmbeds: visPosEmbeds,
            visOut: visOut ?? hidden, hasMropeSection: hasMropeSection, slidingWindow: slidingWindow,
            inkling2Config: inkling2Config
        )
        self.kdaConfig = kdaConfig ?? synth.kda
        self.mlaConfig = mlaConfig ?? synth.mla
        self.attentionSinkRelConfig = attentionSinkRelConfig ?? synth.sinkRel
        self.moeRoutingConfig = moeRoutingConfig ?? synth.moe
        self.swigluClampConfig = swigluClampConfig ?? synth.swiglu
        self.mtpConfig = mtpConfig ?? synth.mtp
        self.visionTowerConfig = visionTowerConfig ?? synth.vision
    }

    // MARK: - Derived views

    /// True when 4-lane gated hyper-connections are active (`hc_count > 0`).
    public var usesMultiStreamHC: Bool { (hcCount ?? 0) > 0 }

    /// True when per-layer embeddings (PLE / n-gram table) are configured.
    public var hasPLE: Bool { !(pleLayerIds?.isEmpty ?? true) || capabilities.contains(.ple) }

    /// True when the model carries native MTP prediction layers or capability.
    public var supportsNativeMTP: Bool {
        mtpLayers > 0 || capabilities.contains(.mtp)
    }

    /// True when the model supports native speculative decoding (MTP layers/capability).
    public var usesNativeSpeculative: Bool { supportsNativeMTP }

    /// True when routed experts can be streamed via DenseModule's cold-expert streamer.
    public var supportsExpertStreaming: Bool {
        guard (numExperts ?? 0) > 0, !usesMultiStreamHC, !hasPLE else { return false }
        if let spec = QWModelFamilyRegistry.shared.spec(forModelType: modelType) {
            return spec.supportsExpertStreaming
        }
        return true
    }

    /// Maximum supported speculative draft depth for this model architecture.
    public var maxSpeculativeDrafts: Int {
        if role == .speculativeDrafter { return 0 }
        if let ds = dsparkConfig { return ds.maxDrafts }
        if usesMultiStreamHC || hasPLE { return 3 }
        if let spec = QWModelFamilyRegistry.shared.spec(forModelType: modelType) {
            return spec.maxSpeculativeDrafts
        }
        return 15
    }

    /// True when physical layer `l` uses a sparse MoE block rather than a dense MLP.
    public func isMoELayer(_ l: Int) -> Bool {
        guard let n = numExperts, n > 0, (0..<numLayers).contains(l) else { return false }
        guard !mlpOnlyLayers.contains(l), l >= firstKDenseReplace else { return false }
        if decoderSparseStep > 1, (l + 1) % decoderSparseStep != 0 { return false }
        let freq = max(1, moeLayerFreq)
        return (l - firstKDenseReplace) % freq == 0
    }

    /// Effective number of routed experts resident in RAM per MoE layer.
    public var effectiveResidentExperts: Int {
        guard let n = numExperts, n > 0 else { return 0 }
        guard let r = residentExpertsPerLayer, r > 0 else { return n }
        return min(n, r)
    }

    /// True when the MoE router applies an additive bias (`e_score_correction_bias` or `expert_bias`) for selection only.
    public var hasBiasCorrectedRouter: Bool {
        topkMethod == "noaux_tc" || moeRouterEnableExpertBias
    }

    /// Return a copy of this descriptor overriding `residentExpertsPerLayer`.
    public func withResidentExperts(_ count: Int?) -> QWModelDescriptor {
        QWModelDescriptor(
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
            usesPlusOneRMSNorm: usesPlusOneRMSNorm,
            quantization: quantization, hasMropeSection: hasMropeSection,
            tieWordEmbeddings: tieWordEmbeddings, normTopKProb: normTopKProb,
            eosTokenIds: eosTokenIds, envelopeModelType: envelopeModelType,
            loopSettings: loopSettings, mhcConfig: mhcConfig,
            firstKDenseReplace: firstKDenseReplace,
            mlpOnlyLayers: mlpOnlyLayers,
            decoderSparseStep: decoderSparseStep,
            moeLayerFreq: moeLayerFreq,
            routedScalingFactor: routedScalingFactor, swigluLimit: swigluLimit,
            topkMethod: topkMethod, residentExpertsPerLayer: count,
            sharedExpertHasGate: sharedExpertHasGate,
            mlaUseNoPE: mlaUseNoPE, kvLoraRank: kvLoraRank, qLoraRank: qLoraRank,
            qkNopeHeadDim: qkNopeHeadDim, qkRopeHeadDim: qkRopeHeadDim, vHeadDim: vHeadDim,
            indexKPoolCompress: indexKPoolCompress, indexKPoolAlwaysSelectTail: indexKPoolAlwaysSelectTail,
            indexShareForMTP: indexShareForMTP, indexerRopeInterleave: indexerRopeInterleave,
            gdnVectorDecay: gdnVectorDecay, gdnLowRankDim: gdnLowRankDim,
            gdnGateLowerBound: gdnGateLowerBound, gdnSeparateConv: gdnSeparateConv,
            hcaConfig: hcaConfig, hybridMLAConfig: hybridMLAConfig,
            blockSparseGQAConfig: blockSparseGQAConfig,
            yarnRopeConfig: yarnRopeConfig,
            ropeScaling: ropeScaling,
            slidingWindow: slidingWindow,
            layerSlidingWindows: layerSlidingWindows,
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
            attnGateMode: attnGateMode, decisionConfig: decisionConfig,
            gatedNormConfig: gatedNormConfig, gdnGatingConfig: gdnGatingConfig,
            finalLogitSoftcapping: finalLogitSoftcapping, v2DigestHeadDim: v2DigestHeadDim
        )
    }

    /// 0-based module indices of paged-KV GQA attention layers (`full_attention` or `sliding_attention`).
    public var fullAttentionLayers: [Int] { layerTypes.indices.filter { Self.isGQALayerKind(layerTypes[$0]) } }
    /// 0-based module indices carrying a PLE injection (ids are one-indexed).
    public var pleModuleIndices: [Int] { (pleLayerIds ?? []).map { $0 - 1 }.sorted() }
    public func isFullAttention(_ l: Int) -> Bool { (0..<numLayers).contains(l) && Self.isGQALayerKind(layerTypes[l]) }
    public func layerSlidingWindow(_ l: Int) -> Int {
        guard isFullAttention(l) else { return 0 }
        if let perLayer = layerSlidingWindows, l < perLayer.count { return max(0, perLayer[l]) }
        return max(0, slidingWindow ?? 0)
    }
    public var hasAnySlidingWindow: Bool { layerTypes.indices.contains { layerSlidingWindow($0) > 0 } }
    private static func isGQALayerKind(_ kind: String) -> Bool { kind == "full_attention" || kind == "sliding_attention" }
    /// q_proj output channels: Q‖gate when the attention carries an output gate.
    public var qProjOut: Int { (attnOutputGate ? 2 : 1) * qHeads * headDim }
    public var rotaryDims: Int { Int(Double(headDim) * partialRotaryFactor) }

    /// Starting channel offset within each attention head for LSA index projection.
    public var lsaHeadSliceStart: Int {
        if hcaConfig != nil && hasMLAAttention { return 0 }
        let unrot = headDim - rotaryDims
        return unrot >= 64 ? rotaryDims : 0
    }

    /// Number of channels per KV head fed into the LSA index projection.
    public var lsaHeadSliceDim: Int {
        if let hca = hcaConfig, hasMLAAttention { return hca.qkNopeHeadDim }
        return headDim - lsaHeadSliceStart
    }

    /// Total input dimension `kvHeads * lsaHeadSliceDim` for per-layer LSA `wq_idx` / `wk_idx` projections.
    public var lsaInputDim: Int {
        if let hca = hcaConfig, hasMLAAttention { return hca.qkNopeHeadDim }
        return kvHeads * lsaHeadSliceDim
    }

    /// Human-readable architecture identifier derived from the model's `model_type`.
    public var architectureDisplayName: String {
        if let envType = envelopeModelType, !envType.isEmpty {
            return envType
        }
        if modelType.hasSuffix("_text") {
            return String(modelType.dropLast(5))
        }
        return modelType
    }

    /// Ordered physical layer indices executed during a forward pass.
    public var executionPlan: [Int] {
        guard let loop = loopSettings,
              loop.evalCount > 1,
              loop.start >= 0,
              loop.end > loop.start,
              loop.end <= numLayers else {
            return Array(0..<numLayers)
        }
        var plan: [Int] = []
        plan.reserveCapacity(loop.start + (loop.end - loop.start) * loop.evalCount + (numLayers - loop.end))
        if loop.start > 0 {
            plan.append(contentsOf: 0..<loop.start)
        }
        let middle = Array(loop.start..<loop.end)
        for _ in 0..<loop.evalCount {
            plan.append(contentsOf: middle)
        }
        if loop.end < numLayers {
            plan.append(contentsOf: loop.end..<numLayers)
        }
        return plan
    }

    // MARK: - Digest

    public var geometryDigestV2: String {
        "qwg-geo-v2:" + Self.digestFields(self).joined(separator: "|")
    }
    public var geometryDigestV3: String {
        "qwg-geo-v3:" + Self.digestV3Fields(self).joined(separator: "|")
    }
    public var legacyGeoV1String: String {
        "qwg-geo-v1:\(hidden)|\(vocab)|\((family == .qwen38Dense ? denseIntermediate : nil) ?? 0)|"
        + "\(numLayers)|\(qHeads)|\(kvHeads)|\(headDim)|\(gdnQKVChannels)|\(gdnVHeads)"
    }
    public var geometryDigestV2Hex: String {
        sha256Hex(geometryDigestV2)
    }
    public var geometryDigestV3Hex: String {
        sha256Hex(geometryDigestV3)
    }

    internal static func digestFields(_ d: QWModelDescriptor) -> [String] {
        let kinds = d.layerTypes.map { kind -> String in
            if kind == "full_attention" { return "F" }
            if kind == "compressed_attention" { return "C" }
            if kind == "sliding_attention" { return "S" }
            return "L"
        }.joined()
        var f: [String] = [
            "model_type=\(d.modelType)",
            "hidden=\(d.hidden)", "vocab=\(d.vocab)",
            "layers=\(d.numLayers)",
            "layer_kinds=\(kinds)",
            "fa_interval=\(d.fullAttentionInterval)",
            "q=\(d.qHeads)", "kv=\(d.kvHeads)", "hd=\(d.v2DigestHeadDim)",
            "qgate=\(d.attnOutputGate ? 1 : 0)",
            "rot_f=0x\(String(d.partialRotaryFactor.bitPattern, radix: 16, uppercase: false))",
            "theta=0x\(String(d.ropeTheta.bitPattern, radix: 16, uppercase: false))",
            "gdn=\(d.gdnQKHeads),\(d.gdnVHeads),\(d.gdnHeadDim),\(d.gdnConvKernel)",
            "mtp=\(d.mtpLayers),\(d.mtpUseDedicatedEmbeddings ? 1 : 0)",
            "eos=\(d.eosTokenId)",
            "rms=0x\(String(d.rmsEps.bitPattern, radix: 16, uppercase: false))",
            "maxpos=\(d.maxPositionEmbeddings)",
        ]
        f.append("idx=\(d.indexerHeads ?? 0),\(d.indexerKvHeads ?? 0),\(d.indexerHeadDim ?? 0),\(d.indexerBudget ?? 0),\(d.indexerCompressRatio ?? 0)")
        f.append("moe=\(d.numExperts ?? 0),\(d.expertsPerTok ?? 0),\(d.moeIntermediate ?? 0),\(d.sharedExpertIntermediate ?? 0),\(d.expertGateType ?? "-")")
        f.append("hc=\(d.hcCount ?? 0),\(d.hcLowrank ?? 0)")
        f.append("ple=\((d.pleLayerIds ?? []).map(String.init).joined(separator: ",")),\(d.pleEmbedDim ?? 0),\(d.pleConvKernel ?? 0),\(d.ngramSize ?? 0),\(d.headsPerNgram ?? 0),\(d.ngramVocabSizeBase ?? 0),\(d.makeNgramVocabDivisibleBy ?? 0),\(d.splitNgramParts ?? 0)")
        if let lat = d.moeLatentSize, lat > 0 {
            f.append("lat=\(lat),\(d.mtpSharedLastLayer ? 1 : 0),\(d.multimaxHead ? 1 : 0)")
        }
        if let hca = d.hcaConfig {
            f.append("hca=\(hca.defaultCompressRatio),\(hca.oGroups),\(hca.oLoraRank),\(hca.slidingWindow)")
        }
        return f
    }

    /// 27B control descriptor matching Qwen3.8-27B `config.json`.
    public static var qwen38_27b: QWModelDescriptor {
        let numLayers = 64
        let interval = 4
        var types: [String] = []
        types.reserveCapacity(numLayers)
        for l in 0..<numLayers {
            types.append((l + 1) % interval == 0 ? "full_attention" : "linear_attention")
        }
        return QWModelDescriptor(
            family: .qwen38Dense, modelType: "qwen3_5", capabilities: [.vision, .mtp],
            hidden: 5120, vocab: 248320, denseIntermediate: 17408,
            numLayers: numLayers, layerTypes: types, fullAttentionInterval: interval,
            rmsEps: 1e-6, maxPositionEmbeddings: 262144,
            qHeads: 24, kvHeads: 4, headDim: 256, attnOutputGate: true,
            partialRotaryFactor: 0.25, ropeTheta: 1e7,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: nil, expertsPerTok: nil, moeIntermediate: nil,
            sharedExpertIntermediate: nil, expertGateType: nil, hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 16, gdnVHeads: 48, gdnHeadDim: 128, gdnQKVChannels: 10240,
            gdnZChannels: 6144, gdnConvKernel: 4,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 1, mtpUseDedicatedEmbeddings: false, eosTokenId: 248044, hasQKNorm: true)
    }
}

public func sha256Hex(_ s: String) -> String {
    SHA256.hash(data: Data(s.utf8)).map { String(format: "%02x", $0) }.joined()
}

public enum QWModelDescriptorError: Error, Equatable {
    /// config.json `model_type` has no registered family. Such a model is not runnable by this engine.
    case unsupportedModelType(String)
}
