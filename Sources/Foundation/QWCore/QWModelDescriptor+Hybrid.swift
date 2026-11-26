// QWModelDescriptor+Hybrid.swift — Hybrid attention, Latent-MoE, SiTU, and LSA projection specs.
// Pure config-sourced value types shared across hybrid MLA, HCA, and Latent-MoE families.

import Foundation

extension QWModelDescriptor {

    /// YaRN rotary frequency ramp and attention scaling parameters (`rope_scaling`).
    public struct YarnRopeConfig: Sendable, Hashable, Codable, Equatable {
        public let factor: Double
        public let originalMaxPositionEmbeddings: Int
        public let betaFast: Double
        public let betaSlow: Double
        public let mscale: Double
        public let mscaleAllDim: Double

        public init(factor: Double, originalMaxPositionEmbeddings: Int,
                    betaFast: Double = 32.0, betaSlow: Double = 1.0,
                    mscale: Double = 0.0, mscaleAllDim: Double = 0.0) {
            self.factor = factor
            self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
            self.betaFast = betaFast
            self.betaSlow = betaSlow
            self.mscale = mscale
            self.mscaleAllDim = mscaleAllDim
        }
    }

    /// Computes Float32 inverse rotary frequencies (`rotaryDim / 2`) and cosine/sine amplitude scale
    /// matching HuggingFace / vLLM YaRN (`truncate = true`) or standard RoPE when `yarn == nil`.
    public static func computeRopeInvFreq(
        rotaryDim: Int,
        base: Double,
        yarn: YarnRopeConfig? = nil
    ) -> (invFreq: [Float], mscale: Float) {
        let pairs = max(0, rotaryDim / 2)
        guard pairs > 0 else { return ([], 1.0) }
        let dimF = Float(rotaryDim)
        let baseF = Float(base)
        let posFreqs = (0..<pairs).map { pow(baseF, Float($0 * 2) / dimF) }
        guard let y = yarn, y.factor > 1.0 else {
            return (posFreqs.map { 1.0 / $0 }, 1.0)
        }
        let factorF = Float(y.factor)
        let invEx = posFreqs.map { 1.0 / $0 }
        let invIn = posFreqs.map { 1.0 / (factorF * $0) }
        func corrDim(_ nr: Double) -> Double {
            (Double(rotaryDim) * Foundation.log(Double(y.originalMaxPositionEmbeddings) / (nr * 2.0 * Double.pi)))
                / (2.0 * Foundation.log(base))
        }
        let low = max(Foundation.floor(corrDim(y.betaFast)), 0.0)
        var high = min(Foundation.ceil(corrDim(y.betaSlow)), Double(rotaryDim - 1))
        if low == high { high += 0.001 }
        var out = [Float](repeating: 0, count: pairs)
        for i in 0..<pairs {
            let ramp = min(max((Float(i) - Float(low)) / Float(high - low), 0.0), 1.0)
            let eef = 1.0 - ramp
            out[i] = invIn[i] * (1.0 - eef) + invEx[i] * eef
        }
        func getMscale(_ scale: Double, _ ms: Double) -> Double {
            (scale <= 1.0 || ms == 0.0) ? 1.0 : (0.1 * ms * Foundation.log(scale) + 1.0)
        }
        let amp = Float(getMscale(y.factor, y.mscale) / getMscale(y.factor, y.mscaleAllDim))
        return (out, amp)
    }

    /// Hybrid Compressed Attention (HCA) block geometry (`compress_ratios[l] > 0`).
    public struct HCAConfig: Sendable, Hashable, Codable, Equatable {
        public let compressRatios: [Int]
        public let defaultCompressRatio: Int
        public let qHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let qLoraRank: Int
        public let kvLoraRank: Int
        public let qkNopeHeadDim: Int
        public let qkRopeHeadDim: Int
        public let vHeadDim: Int
        public let oGroups: Int
        public let oLoraRank: Int
        public let slidingWindow: Int
        public let compressRopeTheta: Double
        public let yarn: YarnRopeConfig?
        public let useAttnMerger: Bool
        public let useVHA: Bool
        public let hasAttnSinkBias: Bool
        public let gatedAttnUseQLora: Bool

        public init(compressRatios: [Int], defaultCompressRatio: Int = 128,
                    qHeads: Int, kvHeads: Int = 1, headDim: Int = 512,
                    qLoraRank: Int = 1024, kvLoraRank: Int = 512,
                    qkNopeHeadDim: Int = 448, qkRopeHeadDim: Int = 64, vHeadDim: Int = 512,
                    oGroups: Int = 8, oLoraRank: Int = 1024, slidingWindow: Int = 128,
                    compressRopeTheta: Double = 160_000.0, yarn: YarnRopeConfig? = nil,
                    useAttnMerger: Bool = true, useVHA: Bool = false,
                    hasAttnSinkBias: Bool = false, gatedAttnUseQLora: Bool = false) {
            self.compressRatios = compressRatios
            self.defaultCompressRatio = defaultCompressRatio
            self.qHeads = qHeads
            self.kvHeads = kvHeads
            self.headDim = headDim
            self.qLoraRank = qLoraRank
            self.kvLoraRank = kvLoraRank
            self.qkNopeHeadDim = qkNopeHeadDim
            self.qkRopeHeadDim = qkRopeHeadDim
            self.vHeadDim = vHeadDim
            self.oGroups = oGroups
            self.oLoraRank = oLoraRank
            self.slidingWindow = slidingWindow
            self.compressRopeTheta = compressRopeTheta
            self.yarn = yarn
            self.useAttnMerger = useAttnMerger
            self.useVHA = useVHA
            self.hasAttnSinkBias = hasAttnSinkBias
            self.gatedAttnUseQLora = gatedAttnUseQLora
        }
    }

    /// Hybrid / decoupled-RoPE Multi-head Latent Attention (MLA-DSA) geometry (`hybrid_mla_*`).
    public struct HybridMLAConfig: Sendable, Hashable, Codable, Equatable {
        public let qHeads: Int
        public let qLoraRank: Int
        public let kvLoraRank: Int
        public let qkNopeHeadDim: Int
        public let qkRopeHeadDim: Int
        public let vHeadDim: Int
        public let decoupledRope: Bool
        public let ropeInterleave: Bool
        public let slidingWindow: Int
        public let hasAttnSinkBias: Bool
        public let useVHA: Bool
        public let indexerUseFastHadamard: Bool
        public let splitKvBProj: Bool
        public let yarn: YarnRopeConfig?

        public init(qHeads: Int, qLoraRank: Int, kvLoraRank: Int,
                    qkNopeHeadDim: Int, qkRopeHeadDim: Int, vHeadDim: Int,
                    decoupledRope: Bool = true, ropeInterleave: Bool = false,
                    slidingWindow: Int = 0, hasAttnSinkBias: Bool = false,
                    useVHA: Bool = false, indexerUseFastHadamard: Bool = false,
                    splitKvBProj: Bool = false, yarn: YarnRopeConfig? = nil) {
            self.qHeads = qHeads
            self.qLoraRank = qLoraRank
            self.kvLoraRank = kvLoraRank
            self.qkNopeHeadDim = qkNopeHeadDim
            self.qkRopeHeadDim = qkRopeHeadDim
            self.vHeadDim = vHeadDim
            self.decoupledRope = decoupledRope
            self.ropeInterleave = ropeInterleave
            self.slidingWindow = slidingWindow
            self.hasAttnSinkBias = hasAttnSinkBias
            self.useVHA = useVHA
            self.indexerUseFastHadamard = indexerUseFastHadamard
            self.splitKvBProj = splitKvBProj
            self.yarn = yarn
        }
    }

    /// Block-sparse GQA (MBSA) indexer and block routing configuration (`attention_variant = "block_sparse_gqa"`).
    public struct BlockSparseGQAConfig: Sendable, Hashable, Codable, Equatable {
        public let blockSize: Int
        public let prefixBlocks: Int
        public let localBlocks: Int
        public let remoteBlocks: Int
        public let indexerHeads: Int
        public let indexerHeadDim: Int

        /// Maximum causal blocks retained per query (`prefix + local + remote`).
        public var maxSelectedBlocks: Int { prefixBlocks + localBlocks + remoteBlocks }
        /// Token count below which all causal blocks fit in `maxSelectedBlocks`.
        public var denseEquivalenceTokens: Int { maxSelectedBlocks * blockSize }

        public init(blockSize: Int, prefixBlocks: Int, localBlocks: Int,
                    remoteBlocks: Int, indexerHeads: Int = 0, indexerHeadDim: Int) {
            self.blockSize = blockSize
            self.prefixBlocks = prefixBlocks
            self.localBlocks = localBlocks
            self.remoteBlocks = remoteBlocks
            self.indexerHeads = indexerHeads
            self.indexerHeadDim = indexerHeadDim
        }
    }

    /// Gated/standard FFN nonlinearity (`SwiGLU`, `GELU`, or bounded `SituAndMul`).
    public enum ActivationKind: Sendable, Hashable, Codable, Equatable {
        case silu
        case gelu
        case geluTanh
        case situ(beta: Float, linearBeta: Float)
    }

    /// Weight-SVD LSA key-projection descriptor for a physical decoder layer.
    public struct LSAKeyProjectionSpec: Sendable, Hashable, Codable, Equatable {
        public enum Kind: String, Sendable, Hashable, Codable {
            case gqaKeyProj
            case hcaCompressWKV
        }
        public let kind: Kind
        public let tensorName: String
        public let headCount: Int
        public let headDim: Int
        public let sliceStart: Int
        public let sliceDim: Int
        public let inputDim: Int

        public init(kind: Kind, tensorName: String, headCount: Int, headDim: Int,
                    sliceStart: Int, sliceDim: Int, inputDim: Int) {
            self.kind = kind
            self.tensorName = tensorName
            self.headCount = headCount
            self.headDim = headDim
            self.sliceStart = sliceStart
            self.sliceDim = sliceDim
            self.inputDim = inputDim
        }
    }

    /// True when physical layer `l` uses Hybrid Compressed Attention (`compressed_attention`).
    public func isCompressedAttention(_ l: Int) -> Bool {
        (0..<numLayers).contains(l) && layerTypes[l] == "compressed_attention"
    }

    /// True when physical layer `l` uses linear GDN / KDA recurrence (`linear_attention`).
    public func isLinearAttention(_ l: Int) -> Bool {
        (0..<numLayers).contains(l) && layerTypes[l] == "linear_attention"
    }

    /// 0-based module indices of compressed-attention (HCA) layers.
    public var compressedAttentionLayers: [Int] {
        layerTypes.indices.filter { layerTypes[$0] == "compressed_attention" }
    }

    /// 0-based module indices of linear-attention (GDN/KDA) layers.
    public var linearAttentionLayers: [Int] {
        layerTypes.indices.filter { layerTypes[$0] == "linear_attention" }
    }

    /// True when the model executes MLA on its full-attention layers.
    public var hasMLAAttention: Bool {
        mlaConfig != nil || (mlaUseNoPE && (kvLoraRank ?? 0) > 0) || hybridMLAConfig != nil
    }

    /// Resolves the RoPE-free key projection tensor slice for training-free Weight-SVD LSA synthesis.
    public func lsaKeyProjectionSpec(layer l: Int) -> LSAKeyProjectionSpec? {
        guard (0..<numLayers).contains(l) else { return nil }
        if let ink2 = inkling2Config, ink2.isSlidingLayer(l) { return nil }
        if isCompressedAttention(l), let hca = hcaConfig, hca.qkNopeHeadDim > 0 {
            return LSAKeyProjectionSpec(
                kind: .hcaCompressWKV,
                tensorName: "model.layers.\(l).self_attn.core_attention.compressor.linear_wkv.weight",
                headCount: 1,
                headDim: hca.headDim,
                sliceStart: 0,
                sliceDim: hca.qkNopeHeadDim,
                inputDim: hidden
            )
        }
        if isFullAttention(l) && !hasMLAAttention && blockSparseGQAConfig == nil
            && capabilities.contains(.qsaIndexer) == false && lsaInputDim >= 64 {
            return LSAKeyProjectionSpec(
                kind: .gqaKeyProj,
                tensorName: "model.layers.\(l).self_attn.k_proj.weight",
                headCount: kvHeads,
                headDim: headDim,
                sliceStart: lsaHeadSliceStart,
                sliceDim: lsaHeadSliceDim,
                inputDim: hidden
            )
        }
        return nil
    }

    internal static func digestV3Fields(_ d: QWModelDescriptor) -> [String] {
        var f = digestFields(d)
        f.append("family=\(d.family.rawValue)")
        f.append("role=\(d.role.rawValue)")
        f.append("env_type=\(d.envelopeModelType ?? "-")")
        let gateBoundHex = d.gdnGateLowerBound.map { "0x" + String($0.bitPattern, radix: 16, uppercase: false) } ?? "-"
        f.append("kda=\(d.gdnVectorDecay ? 1 : 0),\(d.gdnLowRankDim ?? 0),\(d.gdnSeparateConv ? 1 : 0),\(gateBoundHex)")
        f.append("mla=\(d.mlaUseNoPE ? 1 : 0),\(d.kvLoraRank ?? 0),\(d.qLoraRank ?? 0),\(d.hybridMLAConfig != nil ? 1 : 0)")
        f.append("loop=\(d.loopSettings.map { "\($0.start),\($0.end),\($0.evalCount)" } ?? "-")")
        f.append("mhc=\(d.mhcConfig.map { "\($0.start),\($0.end),\($0.streams),\($0.iterations)" } ?? "-")")
        switch d.activationKind {
        case .silu:
            f.append("act=silu")
        case .gelu:
            f.append("act=gelu")
        case .geluTanh:
            f.append("act=gelu_tanh")
        case .situ(let beta, let linearBeta):
            let bHex = String(beta.bitPattern, radix: 16, uppercase: false)
            let lbHex = String(linearBeta.bitPattern, radix: 16, uppercase: false)
            f.append("act=situ,0x\(bHex),0x\(lbHex)")
        }
        let swigluHex = d.swigluLimit.map { "0x" + String($0.bitPattern, radix: 16, uppercase: false) } ?? "-"
        f.append("swiglu_lim=\(swigluHex)")
        f.append("route_scale=0x\(String(d.routedScalingFactor.bitPattern, radix: 16, uppercase: false))")
        f.append("norm_topk=\(d.normTopKProb ? 1 : 0)")
        f.append("tie_emb=\(d.tieWordEmbeddings ? 1 : 0)")
        f.append("qk_norm=\(d.hasQKNorm ? 1 : 0)")
        if d.hasQKVBias { f.append("qkv_bias=1") }
        if d.hasAttnOutputBias { f.append("o_bias=1") }
        f.append("plus1_rms=\(d.usesPlusOneRMSNorm ? 1 : 0)")
        if let bsa = d.blockSparseGQAConfig {
            f.append("bsa=\(bsa.blockSize),\(bsa.prefixBlocks),\(bsa.localBlocks),\(bsa.remoteBlocks)")
        }
        func appendYarn(_ label: String, _ y: YarnRopeConfig?) {
            guard let y else { return }
            let vals = [y.factor, y.betaFast, y.betaSlow, y.mscale, y.mscaleAllDim]
                .map { "0x" + String($0.bitPattern, radix: 16, uppercase: false) }
                .joined(separator: ",")
            f.append("\(label)=\(vals),\(y.originalMaxPositionEmbeddings)")
        }
        appendYarn("yarn", d.yarnRopeConfig)
        appendYarn("hca_yarn", d.hcaConfig?.yarn)
        appendYarn("mla_yarn", d.hybridMLAConfig?.yarn)
        if case .linear(let factor) = d.ropeScaling {
            f.append("rope_scale=linear,0x\(String(factor.bitPattern, radix: 16, uppercase: false))")
        } else if case .dynamic(let factor, let orig) = d.ropeScaling {
            f.append("rope_scale=dynamic,0x\(String(factor.bitPattern, radix: 16, uppercase: false)),\(orig ?? 0)")
        }
        if let sw = d.slidingWindow, sw > 0 { f.append("sw=\(sw)") }
        if let swL = d.layerSlidingWindows { f.append("sw_layers=\(swL.map(String.init).joined(separator: ","))") }
        if !d.mlpOnlyLayers.isEmpty { f.append("mlp_only=\(d.mlpOnlyLayers.sorted().map(String.init).joined(separator: ","))") }
        if d.decoderSparseStep > 1 { f.append("sparse_step=\(d.decoderSparseStep)") }
        if d.capabilities.contains(.vision) {
            f.append("vis=\(d.visDepth),\(d.visHidden),\(d.visHeads),\(d.visIntermediate),\(d.visPatch),\(d.visMerge),\(d.visPosEmbeds),\(d.visOut),\(d.imageTokenId),\(d.videoTokenId)")
        } else {
            f.append("vis=-")
        }
        appendInkling2DigestFields(d, into: &f)
        appendCapabilityV3DigestFields(d, into: &f)
        appendLFMAndDecisionDigestFields(d, into: &f)
        return f
    }

    /// Validates core positive dimensions, divisibility, layer kinds, and dependent counts.
    public func verifyInternalConsistency() -> [String] {
        var bad: [String] = []
        if hidden <= 0 { bad.append("hidden") }
        if vocab <= 0 { bad.append("vocab") }
        if numLayers <= 0 { bad.append("numLayers") }
        if qHeads <= 0 { bad.append("qHeads") }
        if kvHeads <= 0 { bad.append("kvHeads") }
        if headDim <= 0 { bad.append("headDim") }
        if qHeads > 0, kvHeads > 0, qHeads % kvHeads != 0 { bad.append("qHeads%kvHeads") }
        if maxPositionEmbeddings <= 0 { bad.append("maxPositionEmbeddings") }
        if !(rmsEps > 0 && rmsEps.isFinite) { bad.append("rmsEps") }
        if !(ropeTheta > 0 && ropeTheta.isFinite) { bad.append("ropeTheta") }
        if !(partialRotaryFactor > 0 && partialRotaryFactor <= 1.0 && partialRotaryFactor.isFinite) {
            bad.append("partialRotaryFactor")
        }
        if let n = numExperts {
            if n <= 0 { bad.append("numExperts") }
            if let k = expertsPerTok, k <= 0 || k > n { bad.append("expertsPerTok") }
        }
        let validKinds: Set<String> = ["full_attention", "linear_attention", "compressed_attention", "sliding_attention", "conv"]
        if layerTypes.count != numLayers { bad.append("layerTypes.count") }
        if !layerTypes.allSatisfy({ validKinds.contains($0) }) { bad.append("layerTypes") }
        if let swL = layerSlidingWindows, (swL.count != numLayers || swL.contains(where: { $0 < 0 })) {
            bad.append("layerSlidingWindows")
        }
        let mropeSum = mropeW + mropeH + mropeT
        let mlaRopeHalf = (mlaConfig?.qkRopeHeadDim ?? hybridMLAConfig?.qkRopeHeadDim ?? 0) / 2
        if hasMropeSection, mropeSum != rotaryDims / 2 && (mlaRopeHalf <= 0 || mropeSum != mlaRopeHalf) {
            bad.append("mropeSection")
        }
        if !linearAttentionLayers.isEmpty, (gdnQKHeads <= 0 || gdnVHeads <= 0 || gdnHeadDim <= 0) {
            bad.append("gdnDims")
        }
        if gdnQKVChannels != (2 * gdnQKHeads + gdnVHeads) * gdnHeadDim { bad.append("gdnQKVChannels") }
        if gdnZChannels != gdnVHeads * gdnHeadDim { bad.append("gdnZChannels") }
        if fullAttentionInterval > 0, layerTypes.count == numLayers,
           layerTypes.contains("linear_attention") || layerTypes.contains("compressed_attention") {
            let every = fullAttentionLayers.allSatisfy { ($0 + 1) % fullAttentionInterval == 0 }
            if !every { bad.append("full-layer cadence vs interval") }
        }
        if let ids = pleLayerIds {
            for id in ids where id < 1 || id > numLayers || id - 1 >= layerTypes.count
                || layerTypes[id - 1] != "linear_attention" {
                bad.append("ple id \(id)")
            }
        }
        return bad
    }

    internal struct ResolvedMoESchedule {
        let numExperts: Int?
        let firstKDenseReplace: Int
        let mlpOnlyLayers: Set<Int>
        let decoderSparseStep: Int
        let moeLayerFreq: Int
    }

    private static func rawKeyPresent(_ dict: [String: Any]?, _ key: String) -> Bool {
        guard let val = dict?[key] else { return false }
        return !(val is NSNull)
    }

    internal static func validateUnsupportedFeatures(
        text t: ConfigEnvelope.Text,
        env: ConfigEnvelope,
        familySpec: QWModelFamilySpec,
        rawRoot: [String: Any]?,
        rawText: [String: Any],
        inv: (Bool, String) throws -> Void
    ) throws {
        let archs = t.architectures ?? env.architectures
        let allowedSuffixes = ["ForCausalLM", "ForConditionalGeneration", "ForSequenceClassification", "ForDecision"]
        if let archs, !archs.isEmpty,
           !archs.contains(where: { arch in
               allowedSuffixes.contains(where: { arch.hasSuffix($0) }) || arch.hasPrefix("D1")
           }) {
            throw DescriptorError.invariant("unsupported architecture '\(archs[0])'")
        }
        if familySpec.defaultAllFullAttention, !familySpec.allowsAutoMap,
           rawKeyPresent(rawText, "auto_map") || rawKeyPresent(rawRoot, "auto_map") {
            throw DescriptorError.invariant("unsupported 'auto_map'")
        }
        try inv(t.mlp_bias != true, "unsupported 'mlp_bias'")
        try inv(t.output_router_logits != true, "unsupported 'output_router_logits'")
    }

    internal static func validateAndResolveMoE(
        text t: ConfigEnvelope.Text,
        familySpec: QWModelFamilySpec,
        inv: (Bool, String) throws -> Void
    ) throws -> ResolvedMoESchedule {
        let family = familySpec.family
        let resolvedMLPOnly = Set(t.mlp_only_layers ?? [])
        if let only = t.mlp_only_layers {
            for idx in only {
                try inv(idx >= 0 && idx < t.num_hidden_layers, "mlp_only_layers index \(idx) out of range")
            }
        }
        let resolvedFirstKDenseReplace: Int
        if let fk = t.first_k_dense_replace {
            try inv(fk >= 0, "first_k_dense_replace \(fk) < 0")
            resolvedFirstKDenseReplace = fk
        } else if let denseLayers = t.mlp_only_layers, !denseLayers.isEmpty,
                  Set(denseLayers) == Set(0..<denseLayers.count) {
            resolvedFirstKDenseReplace = denseLayers.count
        } else {
            resolvedFirstKDenseReplace = 0
        }
        let resolvedDecoderSparseStep = t.decoder_sparse_step ?? 1
        try inv(resolvedDecoderSparseStep >= 1, "decoder_sparse_step \(resolvedDecoderSparseStep) < 1")
        let resolvedMoELayerFreq = t.moe_layer_freq ?? 1
        try inv(resolvedMoELayerFreq >= 1, "moe_layer_freq \(resolvedMoELayerFreq) < 1")

        let rawNumExperts = t.num_experts ?? t.n_routed_experts
        if let rawNumExperts {
            try inv(rawNumExperts >= 0, "num_experts \(rawNumExperts) < 0")
        }
        if familySpec.disallowsMoEConfig {
            try inv(rawNumExperts == nil,
                    "num_experts \(rawNumExperts ?? 0) on the dense family '\(family.rawValue)'")
            if let moeW = t.moe_intermediate_size, moeW > 0 {
                throw DescriptorError.invariant("moe_intermediate_size \(moeW) on the dense family '\(family.rawValue)'")
            }
            if let sharedW = t.shared_expert_intermediate_size, sharedW > 0 {
                throw DescriptorError.invariant("shared_expert_intermediate_size \(sharedW) on the dense family '\(family.rawValue)'")
            }
        }
        if familySpec.requiresSparseMoEConfig {
            guard let experts = rawNumExperts else {
                throw DescriptorError.invariant("flat config missing 'num_experts' on sparse family '\(family.rawValue)'")
            }
            try inv(experts >= 32 && experts % 32 == 0 && experts <= 1024,
                    "num_experts \(experts) must be a multiple of 32 in 32...1024")
            guard let topK = t.num_experts_per_tok else {
                throw DescriptorError.invariant("flat config missing 'num_experts_per_tok'")
            }
            try inv(topK > 0 && topK <= 16 && topK <= experts,
                    "num_experts_per_tok \(topK) must be in 1...min(16, \(experts))")
            guard let moeW = t.moe_intermediate_size else {
                throw DescriptorError.invariant("flat config missing 'moe_intermediate_size'")
            }
            try inv(moeW > 0, "moe_intermediate_size \(moeW) <= 0")
        }
        if let sw = t.shared_expert_intermediate_size ?? t.moe_shared_expert_intermediate_size {
            try inv(sw >= 0, "shared_expert_intermediate_size \(sw) < 0")
        }
        let resolvedNumExperts = rawNumExperts.flatMap { $0 > 0 ? $0 : nil }
        if let n = rawNumExperts, let k = t.num_experts_per_tok {
            try inv(n > 0 && k > 0 && k <= n, "MoE top-k \(k) > experts \(n)")
        }
        if let ng = t.n_group {
            try inv(ng >= 1, "n_group \(ng) < 1")
            if let n = resolvedNumExperts {
                try inv(n >= ng && n % ng == 0, "num_experts \(n) not divisible by n_group \(ng)")
            }
            if let tg = t.topk_group {
                try inv(tg >= 1 && tg <= ng, "topk_group \(tg) out of range 1...\(ng)")
            }
        }
        let hasDenseLayer: Bool = {
            guard resolvedNumExperts != nil else { return true }
            return (0..<t.num_hidden_layers).contains { layer in
                if resolvedMLPOnly.contains(layer) || layer < resolvedFirstKDenseReplace { return true }
                if resolvedDecoderSparseStep > 1 && (layer + 1) % resolvedDecoderSparseStep != 0 { return true }
                if resolvedMoELayerFreq > 1 && (layer - resolvedFirstKDenseReplace) % resolvedMoELayerFreq != 0 { return true }
                return false
            }
        }()
        if hasDenseLayer || familySpec.disallowsMoEConfig {
            guard let inter = t.intermediate_size else {
                throw DescriptorError.invariant("flat config missing 'intermediate_size'")
            }
            try inv(inter > 0, "intermediate_size \(inter) <= 0")
        }
        return ResolvedMoESchedule(
            numExperts: resolvedNumExperts,
            firstKDenseReplace: resolvedFirstKDenseReplace,
            mlpOnlyLayers: resolvedMLPOnly,
            decoderSparseStep: resolvedDecoderSparseStep,
            moeLayerFreq: resolvedMoELayerFreq
        )
    }

    /// Return a copy of this descriptor with updated attention bias flags.
    public func withAttentionBias(qkv: Bool? = nil, output: Bool? = nil) -> QWModelDescriptor {
        let nextQKV = qkv ?? hasQKVBias
        let nextOutput = output ?? hasAttnOutputBias
        if nextQKV == hasQKVBias && nextOutput == hasAttnOutputBias { return self }
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
            hasQKNorm: hasQKNorm, hasQKVBias: nextQKV, hasAttnOutputBias: nextOutput,
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
            attnGateMode: attnGateMode, decisionConfig: decisionConfig
        )
    }
}
