import Foundation
import CryptoKit

/// C01/§2 — Immutable model identity + geometry for ANY supported model.
public struct QWModelDescriptor: Sendable, Hashable {

    public enum Family: String, Sendable, Hashable, CaseIterable, Codable {
        case qwen38Dense = "qwen3.8-27b"          // control adapter
        case qwen38FlashNext = "qwen3.8-flash-next"
        case qwen3Dense = "qwen3-dense"            // standard HF/MLX, dense MLP
        case qwen3MoE = "qwen3-moe"                // standard HF/MLX, sparse MoE
        case qwen2Dense = "qwen2-dense"
        case llamaDense = "llama-dense"
        case mistralDense = "mistral-dense"
    }

    public enum Capability: String, Sendable, Hashable, CaseIterable, Codable {
        case vision, moe, ple, qsaIndexer, mtp, hyperConnections
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

        public init(start: Int, end: Int, streams: Int = 2, iterations: Int = 20,
                    fixedIdentity: Bool = false, preScale: Float = 1.0) {
            self.start = start
            self.end = end
            self.streams = streams
            self.iterations = iterations
            self.fixedIdentity = fixedIdentity
            self.preScale = preScale
        }
    }

    public let family: Family
    public let modelType: String                  // HF `model_type`
    public let envelopeModelType: String?         // Top-level `model_type` when wrapped (e.g. "sensenova_loop")
    public let capabilities: Set<Capability>
    public let loopSettings: LoopSettings?
    public let mhcConfig: MHCConfig?

    // Text stack
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

    /// Resolved special token IDs for chat template framing, reasoning tags, and stop sets.
    public var specialTokens: QWModelSpecialTokens {
        let stops = eosTokenIds.filter { $0 >= 0 && $0 < vocab }
        return QWModelSpecialTokens(eosTokenId: eosTokenId,
                                     extraStopTokenIds: stops,
                                     usesChatTemplateIds: usesChatTemplateIds)
    }

    /// The Qwen3.8 pack's chat template carries im_start/im_end/think ids.
    public var usesChatTemplateIds: Bool {
        (family == .qwen38Dense || family == .qwen38FlashNext) && !modelType.hasPrefix("glm5")
    }

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
                gdnSeparateConv: Bool = false) {
        self.family = family; self.modelType = modelType
        self.envelopeModelType = envelopeModelType
        self.capabilities = capabilities
        self.loopSettings = loopSettings
        self.mhcConfig = mhcConfig
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
        self.firstKDenseReplace = firstKDenseReplace; self.moeLayerFreq = moeLayerFreq
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
        self.usesPlusOneRMSNorm = usesPlusOneRMSNorm ?? (family == .qwen38Dense && !modelType.hasPrefix("glm5"))
        self.quantization = quantization; self.hasMropeSection = hasMropeSection
        self.tieWordEmbeddings = tieWordEmbeddings; self.normTopKProb = normTopKProb
        self.eosTokenIds = eosTokenIds.isEmpty ? [eosTokenId] : eosTokenIds
    }

    // MARK: - Derived views

    /// True when the model carries native MTP prediction layers or capability.
    public var supportsNativeMTP: Bool {
        mtpLayers > 0 || capabilities.contains(.mtp)
    }

    /// Maximum supported speculative draft depth for this model architecture.
    public var maxSpeculativeDrafts: Int {
        family == .qwen38FlashNext ? 3 : 5
    }

    /// True when physical layer `l` uses a sparse MoE block rather than a dense MLP.
    public func isMoELayer(_ l: Int) -> Bool {
        guard let n = numExperts, n > 0, (0..<numLayers).contains(l) else { return false }
        guard l >= firstKDenseReplace else { return false }
        let freq = max(1, moeLayerFreq)
        return (l - firstKDenseReplace) % freq == 0
    }

    /// Effective number of routed experts resident in RAM per MoE layer.
    public var effectiveResidentExperts: Int {
        guard let n = numExperts, n > 0 else { return 0 }
        guard let r = residentExpertsPerLayer, r > 0 else { return n }
        return min(n, r)
    }

    /// True when the MoE router applies an additive bias (`e_score_correction_bias`) for selection only.
    public var hasBiasCorrectedRouter: Bool {
        topkMethod == "noaux_tc"
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
            hasQKNorm: hasQKNorm, usesPlusOneRMSNorm: usesPlusOneRMSNorm,
            quantization: quantization, hasMropeSection: hasMropeSection,
            tieWordEmbeddings: tieWordEmbeddings, normTopKProb: normTopKProb,
            eosTokenIds: eosTokenIds, envelopeModelType: envelopeModelType,
            loopSettings: loopSettings, mhcConfig: mhcConfig,
            firstKDenseReplace: firstKDenseReplace, moeLayerFreq: moeLayerFreq,
            routedScalingFactor: routedScalingFactor, swigluLimit: swigluLimit,
            topkMethod: topkMethod, residentExpertsPerLayer: count,
            sharedExpertHasGate: sharedExpertHasGate,
            mlaUseNoPE: mlaUseNoPE, kvLoraRank: kvLoraRank, qLoraRank: qLoraRank,
            qkNopeHeadDim: qkNopeHeadDim, qkRopeHeadDim: qkRopeHeadDim, vHeadDim: vHeadDim,
            indexKPoolCompress: indexKPoolCompress, indexKPoolAlwaysSelectTail: indexKPoolAlwaysSelectTail,
            indexShareForMTP: indexShareForMTP, indexerRopeInterleave: indexerRopeInterleave,
            gdnVectorDecay: gdnVectorDecay, gdnLowRankDim: gdnLowRankDim,
            gdnGateLowerBound: gdnGateLowerBound, gdnSeparateConv: gdnSeparateConv
        )
    }

    /// 0-based module indices of full-attention layers, from the authoritative list.
    public var fullAttentionLayers: [Int] { layerTypes.indices.filter { layerTypes[$0] == "full_attention" } }
    /// 0-based module indices carrying a PLE injection (ids are one-indexed).
    public var pleModuleIndices: [Int] { (pleLayerIds ?? []).map { $0 - 1 }.sorted() }
    public func isFullAttention(_ l: Int) -> Bool { (0..<numLayers).contains(l) && layerTypes[l] == "full_attention" }
    /// q_proj output channels: Q‖gate when the attention carries an output gate.
    public var qProjOut: Int { (attnOutputGate ? 2 : 1) * qHeads * headDim }
    public var rotaryDims: Int { Int(Double(headDim) * partialRotaryFactor) }

    /// Starting channel offset within each attention head for LSA index projection.
    public var lsaHeadSliceStart: Int {
        let unrot = headDim - rotaryDims
        return unrot >= 64 ? rotaryDims : 0
    }

    /// Number of channels per KV head fed into the LSA index projection.
    public var lsaHeadSliceDim: Int {
        headDim - lsaHeadSliceStart
    }

    /// Total input dimension `kvHeads * lsaHeadSliceDim` for per-layer LSA `wq_idx` / `wk_idx` projections.
    public var lsaInputDim: Int {
        kvHeads * lsaHeadSliceDim
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
    public var legacyGeoV1String: String {
        "qwg-geo-v1:\(hidden)|\(vocab)|\((family == .qwen38Dense ? denseIntermediate : nil) ?? 0)|"
        + "\(numLayers)|\(qHeads)|\(kvHeads)|\(headDim)|\(gdnQKVChannels)|\(gdnVHeads)"
    }
    public var geometryDigestV2Hex: String {
        sha256Hex(geometryDigestV2)
    }

    internal static func digestFields(_ d: QWModelDescriptor) -> [String] {
        var f: [String] = [
            "model_type=\(d.modelType)",
            "hidden=\(d.hidden)", "vocab=\(d.vocab)",
            "layers=\(d.numLayers)",
            "layer_kinds=\(d.layerTypes.map { $0 == "full_attention" ? "F" : "L" }.joined())",
            "fa_interval=\(d.fullAttentionInterval)",
            "q=\(d.qHeads)", "kv=\(d.kvHeads)", "hd=\(d.headDim)",
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

    /// Re-derives dependent counts; returns violated identities (empty = sane).
    public func verifyInternalConsistency() -> [String] {
        var bad: [String] = []
        if qProjOut != (attnOutputGate ? 2 : 1) * qHeads * headDim { bad.append("qProjOut") }
        if rotaryDims != Int(Double(headDim) * partialRotaryFactor) { bad.append("rotaryDims") }
        if hasMropeSection, mropeW + mropeH + mropeT != rotaryDims / 2 { bad.append("mropeSection") }
        if gdnQKVChannels != (2 * gdnQKHeads + gdnVHeads) * gdnHeadDim { bad.append("gdnQKVChannels") }
        if gdnZChannels != gdnVHeads * gdnHeadDim { bad.append("gdnZChannels") }
        if fullAttentionLayers.count != layerTypes.filter({ $0 == "full_attention" }).count { bad.append("full-layer count") }
        if fullAttentionInterval > 0 {
            let every = fullAttentionLayers.allSatisfy { ($0 + 1) % fullAttentionInterval == 0 }
            if !every { bad.append("full-layer cadence vs interval") }
        }
        if let ids = pleLayerIds {
            for id in ids where id < 1 || id > numLayers || layerTypes[id - 1] != "linear_attention" {
                bad.append("ple id \(id)")
            }
        }
        return bad
    }
}

public func sha256Hex(_ s: String) -> String {
    SHA256.hash(data: Data(s.utf8)).map { String(format: "%02x", $0) }.joined()
}

public enum QWModelDescriptorError: Error, Equatable {
    /// config.json `model_type` has no registered family. Such a model is not runnable by this engine.
    case unsupportedModelType(String)
}
