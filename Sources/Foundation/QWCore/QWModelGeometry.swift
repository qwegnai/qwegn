import Foundation

/// Per-model geometry read from the model's own `config.json` (via `QWModelDescriptor`).
/// Every value comes from the loaded model, so any supported checkpoint (27B, Flash-Next, ...)
/// gets its own geometry with no rebuild.
public struct QWModelGeometry: Sendable, Equatable {
    public static let qwen38_27b = QWModelGeometry(descriptor: .qwen38_27b)

    public let descriptor: QWModelDescriptor

    public init(descriptor: QWModelDescriptor) {
        self.descriptor = descriptor
    }

    /// Loads the geometry of the pack at `packPath` from its own `assets/config.json`.
    /// No fallback: a missing or invalid config throws.
    public static func load(packPath: String) throws -> QWModelGeometry {
        // The root is the single authority for where a model's files live: the
        // config and the assets resolve through the same resolver, so a flat
        // root and a legacy root both work and never disagree.
        let root = QWModelRoot(url: URL(fileURLWithPath: packPath))
        guard let configURL = root.configURL else {
            throw QWModelRootError.missingConfig(
                root.directory.appendingPathComponent("config.json").path)
        }
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: Data(contentsOf: configURL)))
    }

    public var hidden: Int { descriptor.hidden }
    public var vocab: Int { descriptor.vocab }
    /// Dense MLP intermediate width (0 when the family has no dense MLP).
    public var intermediate: Int { descriptor.denseIntermediate ?? 0 }
    /// A sparse MoE layer's expert intermediate width. Zero for a dense
    /// model — the MoE's width, not the dense MLP's.
    public var moeIntermediate: Int { descriptor.moeIntermediate ?? 0 }
    /// Effective routed-expert input/output dimension (`moe_latent_size` when Latent-MoE is active, else `hidden`).
    public var moeRoutedDim: Int {
        if let lat = descriptor.moeLatentSize, lat > 0 { return lat }
        return hidden
    }
    public var moeLatentSize: Int? { descriptor.moeLatentSize }
    public var latentMoeUseNorm: Bool { descriptor.latentMoeUseNorm }
    public var moeSplitFeatureRouting: Bool { descriptor.moeSplitFeatureRouting }
    public var routedScalingFactorLearnable: Bool { descriptor.routedScalingFactorLearnable }
    public var activationKind: QWModelDescriptor.ActivationKind { descriptor.activationKind }
    public var multimaxHead: Bool { descriptor.multimaxHead }
    public var mtpSharedLastLayer: Bool { descriptor.mtpSharedLastLayer }
    /// The number of experts (zero for a dense model).
    public var numExperts: Int { descriptor.numExperts ?? 0 }
    /// How many experts a token routes to.
    public var expertsPerTok: Int { descriptor.expertsPerTok ?? 0 }
    /// Whether the router's top-k weights are renormalised to sum to one.
    /// The model's own flag (`norm_topk_prob`) — never a constant.
    public var normTopKProb: Bool { descriptor.normTopKProb }
    /// A shared expert's intermediate width (zero when the family has none).
    /// The standard path has no shared expert, so a non-zero width is a
    /// model the engine refuses by name.
    public var sharedExpertIntermediate: Int { descriptor.sharedExpertIntermediate ?? 0 }
    public var sharedExpertHasGate: Bool { descriptor.sharedExpertHasGate }
    public var firstKDenseReplace: Int { descriptor.firstKDenseReplace }
    public var moeLayerFreq: Int { descriptor.moeLayerFreq }
    public var routedScalingFactor: Float { descriptor.routedScalingFactor }
    public var swigluLimit: Float? { descriptor.swigluLimit }
    public var gatedNormConfig: GatedNormConfig? { descriptor.gatedNormConfig }
    public var gdnGatingConfig: GDNGatingConfig? { descriptor.gdnGatingConfig }
    public var useSandwichNorm: Bool { descriptor.useSandwichNorm }
    public var finalLogitSoftcapping: Float? { descriptor.finalLogitSoftcapping }
    public var hasBiasCorrectedRouter: Bool { descriptor.hasBiasCorrectedRouter }
    public var effectiveResidentExperts: Int { descriptor.effectiveResidentExperts }
    public func withResidentExperts(_ count: Int?) -> QWModelGeometry {
        guard let count else { return self }
        var d = descriptor
        d.residentExpertsPerLayer = count
        return QWModelGeometry(descriptor: d)
    }
    public func isMoELayer(_ layer: Int) -> Bool { descriptor.isMoELayer(layer) }
    public var numLayers: Int { descriptor.numLayers }
    public var executionPlan: [Int] { descriptor.executionPlan }
    public var numVirtualLayers: Int { descriptor.executionPlan.count }
    public var loopSettings: QWModelDescriptor.LoopSettings? { descriptor.loopSettings }
    public var mhcConfig: QWModelDescriptor.MHCConfig? { descriptor.mhcConfig }
    public var fullAttentionInterval: Int { descriptor.fullAttentionInterval }
    public var rmsEps: Float { descriptor.rmsEps }
    public var maxPositionEmbeddings: Int { descriptor.maxPositionEmbeddings }
    public var yarnRopeConfig: QWModelDescriptor.YarnRopeConfig? { descriptor.yarnRopeConfig }
    public var originalMaxPositionEmbeddings: Int {
        descriptor.yarnRopeConfig?.originalMaxPositionEmbeddings ?? maxPositionEmbeddings
    }

    /// Authoritative per-layer kind from the config's `layer_types`.
    public func isFullAttention(_ layer: Int) -> Bool {
        descriptor.isFullAttention(layer)
    }

    public func isCompressedAttention(_ layer: Int) -> Bool {
        descriptor.isCompressedAttention(layer)
    }

    public func isLinearAttention(_ layer: Int) -> Bool {
        descriptor.isLinearAttention(layer)
    }

    public func layerSlidingWindow(_ layer: Int) -> Int {
        descriptor.layerSlidingWindow(layer)
    }

    public func virtualLayerSlidingWindow(_ virtualLayer: Int) -> Int {
        let plan = descriptor.executionPlan
        guard (0..<plan.count).contains(virtualLayer) else { return 0 }
        return descriptor.layerSlidingWindow(plan[virtualLayer])
    }

    public var hasAnySlidingWindow: Bool {
        descriptor.hasAnySlidingWindow
    }

    /// Authoritative per-virtual-layer kind along `executionPlan`.
    public func isVirtualLayerFullAttention(_ virtualLayer: Int) -> Bool {
        let plan = descriptor.executionPlan
        guard (0..<plan.count).contains(virtualLayer) else { return false }
        return descriptor.isFullAttention(plan[virtualLayer])
    }

    public func isVirtualLayerCompressedAttention(_ virtualLayer: Int) -> Bool {
        let plan = descriptor.executionPlan
        guard (0..<plan.count).contains(virtualLayer) else { return false }
        return descriptor.isCompressedAttention(plan[virtualLayer])
    }

    public func isVirtualLayerGDN(_ virtualLayer: Int) -> Bool {
        let plan = descriptor.executionPlan
        guard (0..<plan.count).contains(virtualLayer) else { return false }
        return descriptor.isLinearAttention(plan[virtualLayer])
    }

    public var qHeads: Int { descriptor.qHeads }
    public var kvHeads: Int { descriptor.kvHeads }
    public var headDim: Int { descriptor.headDim }
    public var attnOutputGate: Bool { descriptor.attnOutputGate }
    public var usesNoPEMLA: Bool { descriptor.mlaUseNoPE && (descriptor.kvLoraRank ?? 0) > 0 }
    public var hcaConfig: QWModelDescriptor.HCAConfig? { descriptor.hcaConfig }
    public var hybridMLAConfig: QWModelDescriptor.HybridMLAConfig? { descriptor.hybridMLAConfig }
    public var blockSparseGQAConfig: QWModelDescriptor.BlockSparseGQAConfig? { descriptor.blockSparseGQAConfig }
    public var hasDedicatedMHCMTP: Bool {
        descriptor.mtpConfig?.fusionMode == .splitEProjHProj && mhcConfig != nil
    }
    public var usesMultiStreamMTPHidden: Bool {
        (mtpSharedLastLayer || hasDedicatedMHCMTP) && mhcConfig != nil
    }
    public var mlaConfig: QWModelDescriptor.MLAConfig? { descriptor.mlaConfig }
    public var kdaConfig: QWModelDescriptor.KDAConfig? { descriptor.kdaConfig }
    public var attentionSinkRelConfig: QWModelDescriptor.AttentionSinkRelConfig? { descriptor.attentionSinkRelConfig }
    public var moeRoutingConfig: QWModelDescriptor.MoERoutingConfig? { descriptor.moeRoutingConfig }
    public var swigluClampConfig: QWModelDescriptor.SwiGLUClampConfig? { descriptor.swigluClampConfig }
    public var mtpConfig: QWModelDescriptor.MTPConfig? { descriptor.mtpConfig }
    public var visionTowerConfig: QWModelDescriptor.VisionTowerConfig? { descriptor.visionTowerConfig }
    public var ropeInterleave: Bool { descriptor.ropeInterleave }
    public var dsparkConfig: QWModelDescriptor.DSparkConfig? { descriptor.dsparkConfig }
    public var mlaHasDirectQ: Bool { mlaConfig?.isDirectQ ?? (qLoraRank <= 0) }
    public var mlaHasIndexer: Bool { mlaConfig?.hasIndexer ?? (indexerHeads > 0) }
    public var hasSubquadraticAttention: Bool {
        mlaHasIndexer || hcaConfig != nil || blockSparseGQAConfig != nil || indexerHeads > 0
    }
    public var kdaFgbChannels: Int {
        guard let kda = kdaConfig else {
            return gdnVectorDecay ? (2 * gdnLowRankDim + gdnVHeads) : gdnVHeads
        }
        return kda.isFullRank ? (2 * gdnZChannels) : (2 * (kda.lowRankDim ?? gdnHeadDim) + gdnVHeads)
    }
    public func routedSwiGLUClamp(layer: Int) -> QWSwiGLUClampSpec? {
        swigluClampConfig?.routedSpec(layer: layer)
    }
    public func sharedSwiGLUClamp(layer: Int) -> QWSwiGLUClampSpec? {
        swigluClampConfig?.sharedSpec(layer: layer)
    }
    public var usesDecoupledRopeMLA: Bool {
        (mlaConfig?.qkRopeHeadDim ?? 0) > 0 || (descriptor.hybridMLAConfig?.decoupledRope == true)
    }
    public var usesAnyMLA: Bool { usesNoPEMLA || descriptor.mlaConfig != nil || descriptor.hybridMLAConfig != nil }
    public var usesLatentOrCompressedAttention: Bool { usesAnyMLA || descriptor.hcaConfig != nil }
    public var kvLoraRank: Int { descriptor.mlaConfig?.kvLoraRank ?? descriptor.kvLoraRank ?? 0 }
    public var qLoraRank: Int { descriptor.mlaConfig?.qLoraRank ?? descriptor.qLoraRank ?? 0 }
    public var qkNopeHeadDim: Int { descriptor.qkNopeHeadDim ?? 0 }
    public var qkRopeHeadDim: Int { descriptor.qkRopeHeadDim ?? 0 }
    public var vHeadDim: Int { descriptor.vHeadDim ?? headDim }
    public var indexKPoolCompress: Bool { descriptor.indexKPoolCompress }
    public var indexKPoolAlwaysSelectTail: Bool { descriptor.indexKPoolAlwaysSelectTail }
    public var indexShareForMTP: Bool { descriptor.indexShareForMTP }
    public var indexerRopeInterleave: Bool { descriptor.indexerRopeInterleave }
    public var indexerHeads: Int { descriptor.indexerHeads ?? 0 }
    public var indexerHeadDim: Int { descriptor.indexerHeadDim ?? 0 }
    public var indexerBudget: Int { descriptor.indexerBudget ?? 0 }
    public var indexerCompressRatio: Int { max(1, descriptor.indexerCompressRatio ?? 1) }
    /// Block-shift (`log2(compressRatio)`) for `QWCoarsePoolIndex` on MLA/QSA indexers (`2` for Key-Pool-4, `0` for 1-token DSA).
    public var indexerPoolShift: Int {
        let r = indexerCompressRatio
        return r > 1 ? r.trailingZeroBitCount : 0
    }
    /// True when the model's norm weights are already stored as `γ − 1`:
    /// the engine's law is `1 + w` and the Qwen3.8 checkpoint stores
    /// `w = γ − 1` under it, while a standard HF checkpoint stores γ.
    public var normGammaIsPlusOne: Bool { descriptor.usesPlusOneRMSNorm }
    public var qProjOut: Int { descriptor.qProjOut }
    public var qChannels: Int { qHeads * headDim }
    public var kvChannels: Int { usesAnyMLA ? max(1, kvLoraRank / 2) : (kvHeads * headDim) }
    public var attnKRow0: Int { qProjOut }
    public var attnVRow0: Int { qProjOut + kvChannels }
    public var attnTotalN: Int { qProjOut + 2 * kvChannels }
    public var mbsaIndexerHeads: Int {
        (blockSparseGQAConfig?.indexerHeads ?? 0) > 0 ? blockSparseGQAConfig!.indexerHeads : kvHeads
    }
    public var mbsaIndexerHeadDim: Int {
        (blockSparseGQAConfig?.indexerHeadDim ?? 0) > 0 ? blockSparseGQAConfig!.indexerHeadDim : headDim
    }
    public var mbsaIdxCols: Int {
        blockSparseGQAConfig != nil ? (mbsaIndexerHeads + 1) * mbsaIndexerHeadDim : 0
    }
    public var attnProjTotalN: Int { attnTotalN + mbsaIdxCols }
    public var partialRotaryFactor: Double { descriptor.partialRotaryFactor }
    public var rotaryDims: Int { descriptor.rotaryDims }
    public var lsaHeadSliceStart: Int { descriptor.lsaHeadSliceStart }
    public var lsaHeadSliceDim: Int { descriptor.lsaHeadSliceDim }
    public var lsaInputDim: Int { descriptor.lsaInputDim }
    public var ropeTheta: Double { descriptor.ropeTheta }
    public var mropeSection: (w: Int, h: Int, t: Int) { descriptor.mropeSection }

    public var gdnQKHeads: Int { descriptor.gdnQKHeads }
    public var gdnVHeads: Int { descriptor.gdnVHeads }
    public var gdnHeadDim: Int { descriptor.gdnHeadDim }
    public var gdnQKVChannels: Int { descriptor.gdnQKVChannels }
    public var gdnZChannels: Int { descriptor.gdnZChannels }
    public var gdnConvKernel: Int { descriptor.gdnConvKernel }
    public var gdnVectorDecay: Bool { descriptor.gdnVectorDecay }
    public var gdnLowRankDim: Int { descriptor.gdnLowRankDim ?? 0 }
    public var gdnGateLowerBound: Float { descriptor.gdnGateLowerBound ?? -1e38 }
    public var gdnSeparateConv: Bool { descriptor.gdnSeparateConv }
    public var gdnGlogChannels: Int { gdnVectorDecay ? gdnZChannels : gdnVHeads }
    public var gdnARow0: Int { gdnQKVChannels + gdnZChannels }
    public var gdnBRow0: Int { gdnQKVChannels + gdnZChannels + gdnVHeads }
    public var gdnTotalN: Int { gdnQKVChannels + gdnZChannels + 2 * gdnVHeads }
    public var fullAttentionLayerCount: Int { executionPlan.filter { descriptor.isFullAttention($0) }.count }
    public var hcaLayerCount: Int { executionPlan.filter { descriptor.isCompressedAttention($0) }.count }
    public var gdnLayerCount: Int { executionPlan.filter { descriptor.isLinearAttention($0) }.count }
    public var mlaLayerCount: Int { usesAnyMLA ? fullAttentionLayerCount : 0 }

    public var mtpLayers: Int { descriptor.mtpLayers }
    public var mtpUseDedicatedEmbeddings: Bool { descriptor.mtpUseDedicatedEmbeddings }

    public var visDepth: Int { descriptor.visDepth }
    public var visHidden: Int { descriptor.visHidden }
    public var visHeads: Int { descriptor.visHeads }
    public var visIntermediate: Int { descriptor.visIntermediate }
    public var visPatch: Int { descriptor.visPatch }
    public var visTemporal: Int { descriptor.visTemporal }
    public var visMerge: Int { descriptor.visMerge }
    public var visPosEmbeds: Int { descriptor.visPosEmbeds }
    public var visOut: Int { descriptor.visOut }

    public var gqaRatio: Int { max(1, qHeads / max(1, kvHeads)) }

    /// Matches the specialized KV writer/pool stride of 4 KV heads × 256 head dims (1024 halves).
    public var matchesKVStride1024Kernel: Bool {
        kvHeads == 4 && headDim == 256
    }

    /// Matches the specialized GQA decode/prefill kernels targeting 4 KV heads × 256 dims at GQA ratio 6.
    public var matchesHead256GQA6Kernel: Bool {
        QWModelGeometry.matchesHead256GQA6Kernel(kvHeads: kvHeads, headDim: headDim, gqaRatio: gqaRatio)
    }

    /// Plan-level counterpart of `matchesHead256GQA6Kernel` so module pre-flight and driver agree.
    public static func matchesHead256GQA6Kernel(kvHeads: Int, headDim: Int, gqaRatio: Int) -> Bool {
        kvHeads == 4 && headDim == 256 && gqaRatio == 6
    }

    /// Whether this geometry can use the hardware NAX MMA paged FlashAttention prefill kernel.
    /// Requires 32-aligned headDim in 32...256, GQA ratio >= 1, and threadgroup width <= 1024.
    public var supportsNAXPagedAttnPrefill: Bool {
        headDim >= 32 && headDim <= 256 && (headDim % 32 == 0)
            && (gqaRatio >= 1) && (64 * gqaRatio <= 1024)
    }

    /// Whether this geometry can use the hardware Grouped NAX MoE prefill pipeline.
    /// Requires <= 512 experts and 64-element tile alignment on hidden and moeIntermediate.
    public var supportsGroupedNAXMoEPrefill: Bool {
        numExperts > 0 && numExperts <= 1024 && hidden > 0 && hidden % 64 == 0
            && moeRoutedDim > 0 && moeRoutedDim % 64 == 0
            && moeIntermediate > 0 && moeIntermediate % 64 == 0
    }

    public var imStartTokenId: Int { descriptor.specialTokens.imStartTokenId }
    public var imEndTokenId: Int { descriptor.specialTokens.imEndTokenId }
    public var imageTokenId: Int { descriptor.imageTokenId }
    public var videoTokenId: Int { descriptor.videoTokenId }
    public var visionStartTokenId: Int { descriptor.visionStartTokenId }
    public var visionEndTokenId: Int { descriptor.visionEndTokenId }
    public var mropeLayout: QWModelDescriptor.MRoPELayout {
        mlaConfig?.rope?.mropeLayout
            ?? descriptor.visionTowerConfig?.mropeLayout
            ?? (descriptor.hasMropeSection ? .qwenInterleaved : .none1D)
    }
    public var thinkTokenId: Int { descriptor.specialTokens.thinkTokenId }
    public var endThinkTokenId: Int { descriptor.specialTokens.endThinkTokenId }

    public func verifyInternalConsistency() -> [String] {
        descriptor.verifyInternalConsistency()
    }
}

/// Execution-plan-derived sharing schedule for recurrent global-attention visits.
public struct QWLSALoopReusePlan: Sendable, Equatable {
    public let physicalLayerForFullAttnLayer: [Int]
    public let sourceForFullAttnLayer: [Int]
    public let isFollower: [Bool]
    public let uniqueLeaderIndices: [Int]
    public let reuseCacheSlotForLeader: [Int: Int]
    public let reuseSlotCount: Int

    public var hasReuse: Bool { isFollower.contains(true) }

    public static func build(physicalLayers: [Int], enabled: Bool, stride: Int = 0) -> QWLSALoopReusePlan {
        let count = physicalLayers.count
        var source = Array(0..<count)
        if enabled && count > 1 {
            let effectiveStride = stride >= 2 ? stride : 0
            var visitsByPhys: [Int: [Int]] = [:]
            for (idx, phys) in physicalLayers.enumerated() {
                visitsByPhys[phys, default: []].append(idx)
            }
            for visits in visitsByPhys.values where visits.count > 1 {
                var currentLeader = visits[0]
                for (visitOffset, fullIdx) in visits.enumerated() {
                    if visitOffset == 0 || (effectiveStride > 0 && visitOffset % effectiveStride == 0) {
                        currentLeader = fullIdx
                    }
                    source[fullIdx] = currentLeader
                }
            }
        }
        let followerFlags = (0..<count).map { source[$0] != $0 }
        var followerCountByLeader: [Int: Int] = [:]
        for (idx, src) in source.enumerated() where src != idx {
            followerCountByLeader[src, default: 0] += 1
        }
        let leaders = (0..<count).filter { source[$0] == $0 }
        var cacheSlots: [Int: Int] = [:]
        var nextSlot = 0
        for leader in leaders where (followerCountByLeader[leader] ?? 0) > 0 {
            cacheSlots[leader] = nextSlot
            nextSlot += 1
        }
        return QWLSALoopReusePlan(
            physicalLayerForFullAttnLayer: physicalLayers,
            sourceForFullAttnLayer: source,
            isFollower: followerFlags,
            uniqueLeaderIndices: leaders,
            reuseCacheSlotForLeader: cacheSlots,
            reuseSlotCount: nextSlot
        )
    }
}
