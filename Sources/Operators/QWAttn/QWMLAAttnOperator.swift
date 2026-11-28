// QWMLAAttnOperator.swift — Matrix-Absorbed Multi-head Latent Attention GPU operator
// supporting both NoPE MLA + Key-Pool(4) (GLM-5.x) and Decoupled-RoPE MLA-DSA (ERNIE-5.5).
// Manages per-MLA-layer indexer rings/pools, optional decoupled k_pe cache, e-means CSR,
// and shared latent/query/attention scratch in a single preallocated Metal region.

import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWQuant
import QWGEMM
import QWKV

public final class QWMLAAttnOperator: @unchecked Sendable {
    public let selectBatchRows: Int

    public let geometry: QWModelGeometry
    public let isDecoupledDSA: Bool
    public let hasIndexer: Bool
    public let isDirectQ: Bool
    public let outputGateMode: QWModelDescriptor.MLAConfig.OutputGateMode
    public let ropeInterleave: Bool
    public let mropeLayout: QWModelDescriptor.MRoPELayout
    public let maxContext: Int
    public let maxPrefillChunk: Int
    public let qHeads: Int
    public let qLoraRank: Int
    public let kvLoraRank: Int
    public let nopeDim: Int
    public let ropeDim: Int
    public let vHeadDim: Int
    public let idxHeads: Int
    public let idxDim: Int
    public let poolShift: Int
    public let windowTokens: Int
    public let useFastHadamard: Bool
    public let ropeMscale: Float
    public let alphaQ: Float
    public let alphaKV: Float
    public let indexerKNormIsRMS: Bool
    public let softmaxScaleOverride: Float?
    public let denseOnly: Bool
    public let oGateInQKV: Bool
    public let mtpDenseOnly: Bool
    public let topKPools: Int
    public let listStride: Int
    public let ringCap: Int
    public let maxBlocks: Int
    public let latentStride: Int
    public let qChannels: Int
    public let qRopeOffset: Int
    public let oGateOffset: Int
    public let qFullStride: Int

    public let region: QWMRegion
    public let layers: [LayerState]
    public weak var stateOwner: AnyObject? {
        didSet {
            if stateOwner !== oldValue { resetCoarseWatermarks() }
        }
    }

    public let mlaLatentScratch: QWTensorValue
    public let qResidScratch: QWTensorValue
    public let mlaQFullScratch: QWTensorValue
    public let idxWeightsScratch: QWTensorValue
    public let flashQSelScratch: QWTensorValue
    public let flashWSelScratch: QWTensorValue
    public let qAbsScratch: QWTensorValue
    public let uLatScratch: QWTensorValue
    public let yVhaScratch: QWTensorValue
    public let oGateScratch: QWTensorValue
    public let invFreqBuf: QWTensorValue
    public let zeroSinkBuf: QWTensorValue
    public let mropePosScratch: QWTensorValue
    public let ropeAxisMapBuf: QWTensorValue
    public let scoreScratch: QWTensorValue
    public let blockListScratch: QWTensorValue
    public let blockCountScratch: QWTensorValue

    internal let hal: QWHal
    internal let psoLatentPrep: any MTLComputePipelineState
    internal let psoDSALatentPrep: any MTLComputePipelineState
    internal let psoPoolKeys: any MTLComputePipelineState
    internal let psoGateWeightsProj: any MTLComputePipelineState
    internal let psoQRopeIdxPrep: any MTLComputePipelineState
    internal let psoIndexerScoreSelect: any MTLComputePipelineState
    internal let psoFlashIndexerSelectHeads: any MTLComputePipelineState
    internal let psoAbsorbQ: any MTLComputePipelineState
    internal let psoAbsorbQSub2: any MTLComputePipelineState
    internal let psoAbsorbQMMA: any MTLComputePipelineState
    internal let psoAbsorbQMMASub2: any MTLComputePipelineState
    internal let psoAbsorbQNAX: (any MTLComputePipelineState)?
    internal let psoAbsorbQNAXSub2: (any MTLComputePipelineState)?
    internal let psoSparseAttn: any MTLComputePipelineState
    internal let psoSparseAttnMMA: any MTLComputePipelineState
    internal let psoDSASparseAttn: any MTLComputePipelineState
    internal let psoDSASparseAttnMMA: any MTLComputePipelineState
    internal let psoIndexerSplit: (score: any MTLComputePipelineState, select: any MTLComputePipelineState)?
    internal let psoSparseAttnNAX: (any MTLComputePipelineState)?
    internal let psoDSASparseAttnNAX: (any MTLComputePipelineState)?
    internal let psoDSASparseAttnNAXH32: (any MTLComputePipelineState)?
    internal let psoDSASparseAttnNAXH64: (any MTLComputePipelineState)?
    internal let psoSparseAttnNAXH32: (any MTLComputePipelineState)?
    internal let psoSparseAttnSplit: (attend: any MTLComputePipelineState, combine: any MTLComputePipelineState)?
    internal let psoDSASparseAttnSplit: (attend: any MTLComputePipelineState, combine: any MTLComputePipelineState)?
    public static let splitMaxRows = 15
    public let attnSplits: Int
    internal let partOScratch: QWTensorValue
    internal let partMLScratch: QWTensorValue
    public var sparseAttnMMAEnabled = ProcessInfo.processInfo.environment["QW_MLA_ATTN_MMA"] != "0"
    public var dsaSharedMode = ProcessInfo.processInfo.environment["QW_MLA_DSA_SHARED"]?.lowercased() ?? "h64"
    public var projNAXEnabled = ProcessInfo.processInfo.environment["QW_MLA_PROJ_NAX"] != "0"
    public var flashIndexerEnabled = ProcessInfo.processInfo.environment["QW_MLA_FLASHINDEXER"] != "0"
    public var emeansCoarseEnabled: Bool
    internal let psoProjUV: any MTLComputePipelineState
    internal let psoProjUVSub2: any MTLComputePipelineState
    internal let psoProjUVMMA: any MTLComputePipelineState
    internal let psoProjUVMMASub2: any MTLComputePipelineState
    internal let psoProjUVNAX: (any MTLComputePipelineState)?
    internal let psoProjUVNAXSub2: (any MTLComputePipelineState)?
    internal let psoVHAGateRot: any MTLComputePipelineState
    internal let psoHeadGateProj: any MTLComputePipelineState
    internal let coarsePipeline: QWCoarseSelectionPipeline?
    private var emptyClusteredMacros: [Int]
    internal var coarsePipelines: QWCoarsePoolIndex.Pipelines? { coarsePipeline?.pipelines }
    internal var psoEmeansCoarse: QWCoarseSelectionPipeline.EmeansPSOTuple? { coarsePipeline?.emeansTuple }
    internal var coarseScratch: QWCoarsePoolIndex.Scratch? { coarsePipeline?.scratch }
    internal var clusteredMacros: [Int] {
        get { coarsePipeline?.watermarks ?? emptyClusteredMacros }
        set {
            if let coarsePipeline { coarsePipeline.watermarks = newValue }
            else { emptyClusteredMacros = newValue }
        }
    }

    public func resetCoarseWatermarks() {
        coarsePipeline?.resetWatermarks()
        for i in emptyClusteredMacros.indices { emptyClusteredMacros[i] = 0 }
    }

    public func truncatePosition(_ validPosition: Int) {
        if let coarsePipeline {
            coarsePipeline.truncatePosition(validPosition, poolShift: poolShift)
            return
        }
        let maxMacro = max(0, validPosition) / QWCoarsePoolIndex.macroChunkTokens(poolShift: poolShift)
        for i in emptyClusteredMacros.indices where emptyClusteredMacros[i] > maxMacro {
            emptyClusteredMacros[i] = maxMacro
        }
    }

    internal struct LatentPrepParams {
        var T: UInt32
        var pos0: UInt32
        var hidden: UInt32
        var qLoraRank: UInt32
        var kvLoraRank: UInt32
        var idxDim: UInt32
        var idxHeads: UInt32
        var ringMask: UInt32
        var rmsEps: Float
        var kNormEps: Float
        var kvMode: UInt32
    }

    internal struct IndexerParams {
        var T: UInt32
        var pos0: UInt32
        var qChannels: UInt32
        var idxHeads: UInt32
        var idxDim: UInt32
        var topKPools: UInt32
        var listStride: UInt32
        var scoreStride: UInt32
    }

    internal struct FlashIndexerParams {
        var T: UInt32
        var hidden: UInt32
        var fullIdxHeads: UInt32
        var selHeads: UInt32
        var idxDim: UInt32
        var qChannels: UInt32
        var qFullStride: UInt32
    }

    internal struct AbsorbParams {
        var T: UInt32
        var qHeads: UInt32
        var nopeDim: UInt32
        var vHeadDim: UInt32
        var kvLoraRank: UInt32
        var qFullStride: UInt32
        var bits: UInt32
        var group: UInt32
        var isIncumbent: UInt32
    }

    internal struct SparseAttnParams {
        var T: UInt32
        var pos0: UInt32
        var qHeads: UInt32
        var headDim: UInt32
        var kvLoraRank: UInt32
        var topKPools: UInt32
        var listStride: UInt32
        var kvMode: UInt32
    }

    internal static func kvModeCode(_ mode: QWKVMode) -> UInt32 {
        switch mode {
        case .f16: return 0
        case .q8: return 1
        case .q4: return 2
        }
    }

    public static func makeIfEnabled(hal: QWHal,
                                     geometry: QWModelGeometry,
                                     attnLayers: [Int],
                                     maxContext: Int,
                                     maxPrefillChunk: Int = 512,
                                     lsaConfig: QWLSAConfig = QWLSAConfig(),
                                     extraAttnLayers: Int = 0,
                                     alphaQ: Float = 1.0,
                                     alphaKV: Float = 1.0,
                                     indexerKNormIsRMS: Bool = false,
                                     softmaxScaleOverride: Float? = nil,
                                     denseOnly: Bool = false,
                                     oGateInQKV: Bool = false,
                                     mtpDenseOnly: Bool = false) throws -> QWMLAAttnOperator? {
        let hasMLA = geometry.usesAnyMLA && !attnLayers.isEmpty
        guard hasMLA else { return nil }
        return try QWMLAAttnOperator(
            hal: hal,
            geometry: geometry,
            numAttnLayers: attnLayers.count + max(0, extraAttnLayers),
            maxContext: maxContext,
            maxPrefillChunk: maxPrefillChunk,
            lsaConfig: lsaConfig,
            alphaQ: alphaQ,
            alphaKV: alphaKV,
            indexerKNormIsRMS: indexerKNormIsRMS,
            softmaxScaleOverride: softmaxScaleOverride,
            denseOnly: denseOnly,
            oGateInQKV: oGateInQKV,
            mtpDenseOnly: mtpDenseOnly
        )
    }

    public init(hal: QWHal,
                geometry: QWModelGeometry,
                numAttnLayers: Int,
                maxContext: Int,
                maxPrefillChunk: Int = 512,
                lsaConfig: QWLSAConfig = QWLSAConfig(),
                alphaQ: Float = 1.0,
                alphaKV: Float = 1.0,
                indexerKNormIsRMS: Bool = false,
                softmaxScaleOverride: Float? = nil,
                denseOnly: Bool = false,
                oGateInQKV: Bool = false,
                mtpDenseOnly: Bool = false) throws {
        self.hal = hal
        self.geometry = geometry
        self.alphaQ = alphaQ
        self.alphaKV = alphaKV
        self.indexerKNormIsRMS = indexerKNormIsRMS
        self.softmaxScaleOverride = softmaxScaleOverride
        self.denseOnly = denseOnly
        self.oGateInQKV = oGateInQKV
        self.mtpDenseOnly = mtpDenseOnly
        let mla = geometry.mlaConfig
        let hmla = geometry.hybridMLAConfig
        let dsa = geometry.usesDecoupledRopeMLA
        self.isDecoupledDSA = dsa
        self.maxContext = max(512, maxContext)
        self.maxPrefillChunk = max(32, maxPrefillChunk)

        let rawHasIndexer: Bool
        if let mla {
            self.qHeads = mla.qHeads
            self.qLoraRank = mla.qLoraRank
            self.kvLoraRank = mla.kvLoraRank
            self.nopeDim = mla.qkNopeHeadDim
            self.ropeDim = mla.qkRopeHeadDim
            self.vHeadDim = mla.vHeadDim
            rawHasIndexer = mla.hasIndexer
            self.isDirectQ = mla.isDirectQ
            self.outputGateMode = mla.outputGateMode
            self.poolShift = geometry.indexerPoolShift
            self.windowTokens = max(0, mla.slidingWindow)
            self.useFastHadamard = mla.indexerUseFastHadamard
            self.ropeInterleave = mla.rope?.interleave ?? geometry.ropeInterleave
            self.mropeLayout = mla.rope?.mropeLayout ?? .none1D
        } else if let hmla, dsa {
            self.qHeads = hmla.qHeads
            self.qLoraRank = hmla.qLoraRank
            self.kvLoraRank = hmla.kvLoraRank
            self.nopeDim = hmla.qkNopeHeadDim
            self.ropeDim = hmla.qkRopeHeadDim
            self.vHeadDim = hmla.vHeadDim
            rawHasIndexer = geometry.indexerHeads > 0
            self.isDirectQ = hmla.qLoraRank <= 0
            self.outputGateMode = geometry.attnOutputGate ? .elementWiseFromQLora : .none
            self.poolShift = geometry.indexerPoolShift
            self.windowTokens = max(0, hmla.slidingWindow)
            self.useFastHadamard = hmla.indexerUseFastHadamard
            self.ropeInterleave = hmla.ropeInterleave
            self.mropeLayout = .none1D
        } else {
            self.qHeads = geometry.qHeads
            self.qLoraRank = geometry.qLoraRank
            self.kvLoraRank = geometry.kvLoraRank
            self.nopeDim = geometry.qkNopeHeadDim > 0 ? geometry.qkNopeHeadDim : geometry.headDim
            self.ropeDim = 0
            self.vHeadDim = geometry.vHeadDim > 0 ? geometry.vHeadDim : geometry.headDim
            rawHasIndexer = geometry.indexerHeads > 0 || geometry.usesNoPEMLA
            self.isDirectQ = geometry.qLoraRank <= 0
            self.outputGateMode = geometry.attnOutputGate ? .elementWiseFromQLora : .none
            self.poolShift = 2
            self.windowTokens = 0
            self.useFastHadamard = false
            self.ropeInterleave = geometry.ropeInterleave
            self.mropeLayout = .none1D
        }
        self.hasIndexer = denseOnly ? false : rawHasIndexer

        if self.hasIndexer {
            self.idxHeads = max(1, geometry.indexerHeads)
            self.idxDim = max(32, geometry.indexerHeadDim)
        } else {
            self.idxHeads = 0
            self.idxDim = 0
        }
        let modelBudget = geometry.indexerBudget > 0 ? geometry.indexerBudget : 2048
        let requestedBudget = (lsaConfig.budget != QWLSAConfig.defaultBudget)
            ? lsaConfig.alignedBudget
            : modelBudget
        let gateInQKVCols = oGateInQKV
            ? ((self.outputGateMode == .headWiseFromHidden) ? self.qHeads : (self.qHeads * self.vHeadDim))
            : 0
        if !self.hasIndexer {
            self.topKPools = 0
            self.listStride = 1
            self.latentStride = self.qLoraRank + self.kvLoraRank + self.ropeDim + gateInQKVCols
            self.qRopeOffset = self.qHeads * self.nopeDim
            self.oGateOffset = oGateInQKV
                ? (self.qLoraRank + self.kvLoraRank + self.ropeDim)
                : (self.qHeads * (self.nopeDim + self.ropeDim))
            let hasInlineGate = !oGateInQKV && ((self.outputGateMode == .elementWiseFromQLora)
                || (mla == nil && geometry.attnOutputGate))
            self.qChannels = self.qHeads * (self.nopeDim + self.ropeDim + (hasInlineGate ? self.vHeadDim : 0))
            self.qFullStride = self.qChannels
            self.maxBlocks = max(128, self.maxContext)
        } else if dsa {
            let budgetTok = min(2048, max(1, requestedBudget))
            guard budgetTok + self.windowTokens <= 2560 else {
                throw QWAttnError.badShape(
                    "DSA budget (\(budgetTok)) + windowTokens (\(self.windowTokens)) exceeds 2560"
                )
            }
            self.topKPools = budgetTok
            self.listStride = max(512, ((budgetTok + self.windowTokens + 127) & ~127))
            self.latentStride = self.qLoraRank + self.kvLoraRank + self.ropeDim + self.idxDim + gateInQKVCols
            self.qRopeOffset = self.qHeads * self.nopeDim
            self.oGateOffset = oGateInQKV
                ? (self.qLoraRank + self.kvLoraRank + self.ropeDim + self.idxDim)
                : (self.qHeads * (self.nopeDim + self.ropeDim))
            let hasInlineGate = !oGateInQKV && ((self.outputGateMode == .elementWiseFromQLora)
                || (mla == nil && geometry.attnOutputGate))
            self.qChannels = self.qHeads * (self.nopeDim + self.ropeDim + (hasInlineGate ? self.vHeadDim : 0))
            self.qFullStride = self.qChannels + self.idxHeads * self.idxDim
            self.maxBlocks = max(128, self.maxContext >> self.poolShift)
        } else {
            let budgetTok = min(4096, max(128, requestedBudget))
            self.topKPools = max(1, min(1024, budgetTok / 4))
            self.listStride = max(512, self.topKPools)
            self.latentStride = self.qLoraRank + self.kvLoraRank + self.idxDim + gateInQKVCols
            self.qRopeOffset = self.qHeads * self.nopeDim
            self.oGateOffset = oGateInQKV
                ? (self.qLoraRank + self.kvLoraRank + self.idxDim)
                : (self.qHeads * self.nopeDim)
            self.qChannels = self.qHeads * self.nopeDim
            self.qFullStride = self.qChannels + self.idxHeads * self.idxDim
            self.maxBlocks = max(128, (self.maxContext + 3) / 4)
        }

        let resolvedRopeTheta = mla?.rope?.theta ?? geometry.ropeTheta
        let resolvedRopeYarn = mla?.rope?.yarn ?? hmla?.yarn
        let (invFreqArr, mscale) = dsa && self.ropeDim > 0
            ? QWModelDescriptor.computeRopeInvFreq(
                rotaryDim: self.ropeDim,
                base: resolvedRopeTheta,
                yarn: resolvedRopeYarn
            )
            : ([Float](repeating: 0, count: 32), Float(1.0))
        self.ropeMscale = mscale
        let ropeAxisCount = max(1, self.ropeDim / 2)
        var ropeAxisMap = [UInt32](repeating: 0, count: ropeAxisCount)
        let mSec = geometry.mropeSection
        switch self.mropeLayout {
        case .bailingVideoRope:
            let hwLimit = mSec.h + mSec.t
            for i in 0..<ropeAxisCount where i < hwLimit {
                ropeAxisMap[i] = (i & 1) == 0 ? 1 : 2
            }
        case .qwenInterleaved:
            for i in 0..<ropeAxisCount {
                if i % 3 == 1 && i < mSec.h * 3 {
                    ropeAxisMap[i] = 1
                } else if i % 3 == 2 && i < mSec.t * 3 {
                    ropeAxisMap[i] = 2
                } else {
                    ropeAxisMap[i] = 0
                }
            }
        case .none1D:
            break
        }

        var rc = 4096
        while rc < self.maxPrefillChunk * 2 { rc <<= 1 }
        self.ringCap = rc

        self.psoLatentPrep = try hal.pipeline("qwg_mla_latent_prep")
        self.psoDSALatentPrep = try hal.pipeline("qwg_mla_dsa_latent_prep")
        self.psoPoolKeys = try hal.pipeline("qwg_mla_pool_keys")
        self.psoGateWeightsProj = try hal.pipeline("qwg_mla_gate_weights_proj")
        self.psoQRopeIdxPrep = try hal.pipeline("qwg_mla_q_rope_idx_prep")
        self.psoIndexerScoreSelect = try hal.pipeline("qwg_mla_indexer_score_select")
        self.psoFlashIndexerSelectHeads = try hal.pipeline("qwg_mla_flashindexer_select_heads")
        self.psoAbsorbQ = try hal.pipeline("qwg_mla_absorb_q")
        self.psoAbsorbQSub2 = try hal.pipeline("qwg_mla_absorb_q_sub2")
        self.psoAbsorbQMMA = try hal.pipeline("qwg_mla_absorb_q_mma")
        self.psoAbsorbQMMASub2 = try hal.pipeline("qwg_mla_absorb_q_mma_sub2")
        let naxOK = hal.caps.isMetal4TensorCapable
            && ProcessInfo.processInfo.environment["QW_MLA_ATTN_NAX"] != "0"
        self.psoAbsorbQNAX = hal.caps.isMetal4TensorCapable ? try hal.pipeline("qwg_mla_absorb_q_nax") : nil
        self.psoAbsorbQNAXSub2 = hal.caps.isMetal4TensorCapable ? try hal.pipeline("qwg_mla_absorb_q_nax_sub2") : nil
        self.psoSparseAttn = try hal.pipeline("qwg_mla_sparse_attn")
        self.psoSparseAttnMMA = try hal.pipeline("qwg_mla_sparse_attn_mma")
        self.psoDSASparseAttn = try hal.pipeline("qwg_mla_dsa_sparse_attn")
        self.psoDSASparseAttnMMA = try hal.pipeline("qwg_mla_dsa_sparse_attn_mma")
        self.psoIndexerSplit = self.hasIndexer && (self.idxHeads == 32 || self.idxHeads == 64) && self.idxDim == 128
            && ProcessInfo.processInfo.environment["QW_MLA_IDX_SPLIT"] != "0"
            ? (try hal.pipeline("qwg_mla_indexer_score_mma"), try hal.pipeline("qwg_mla_indexer_select")) : nil
        self.psoSparseAttnNAX = naxOK ? try hal.pipeline("qwg_mla_sparse_attn_nax") : nil
        self.psoDSASparseAttnNAX = naxOK ? try hal.pipeline("qwg_mla_dsa_sparse_attn_nax") : nil
        self.psoDSASparseAttnNAXH32 = naxOK ? try hal.pipeline("qwg_mla_dsa_sparse_attn_nax_h32") : nil
        self.psoDSASparseAttnNAXH64 = naxOK ? try hal.pipeline("qwg_mla_dsa_sparse_attn_nax_h64") : nil
        self.psoSparseAttnNAXH32 = naxOK ? try hal.pipeline("qwg_mla_sparse_attn_nax_h32") : nil
        let splits = Int(ProcessInfo.processInfo.environment["QW_MLA_ATTN_SPLIT"] ?? "") ?? 16
        let splitOK = hal.caps.supportsSimdgroupMMA
            && ProcessInfo.processInfo.environment["QW_MLA_ATTN_NAX"] != "0"
        self.attnSplits = (splitOK && self.kvLoraRank == 512) ? max(0, min(64, splits)) : 0
        let splitKind = naxOK ? "nax" : "mma"
        self.psoSparseAttnSplit = self.attnSplits > 1
            ? (try hal.pipeline("qwg_mla_sparse_attn_\(splitKind)_split"), try hal.pipeline("qwg_mla_sparse_attn_combine")) : nil
        self.psoDSASparseAttnSplit = self.attnSplits > 1
            ? (try hal.pipeline("qwg_mla_dsa_sparse_attn_\(splitKind)_split"), try hal.pipeline("qwg_mla_dsa_sparse_attn_combine")) : nil
        self.psoProjUV = try hal.pipeline("qwg_mla_proj_uv")
        self.psoProjUVSub2 = try hal.pipeline("qwg_mla_proj_uv_sub2")
        self.psoProjUVMMA = try hal.pipeline("qwg_mla_proj_uv_mma")
        self.psoProjUVMMASub2 = try hal.pipeline("qwg_mla_proj_uv_mma_sub2")
        self.psoProjUVNAX = hal.caps.isMetal4TensorCapable ? try hal.pipeline("qwg_mla_proj_uv_nax") : nil
        self.psoProjUVNAXSub2 = hal.caps.isMetal4TensorCapable ? try hal.pipeline("qwg_mla_proj_uv_nax_sub2") : nil
        self.psoVHAGateRot = try hal.pipeline("qwg_mla_vha_gate_rot")
        self.psoHeadGateProj = try hal.pipeline("qwg_mla_head_gate_proj")

        let coarseMinCtx = dsa ? 16_384 : 32_768
        let useCoarse = self.hasIndexer && (self.psoIndexerSplit != nil) && (self.maxContext >= coarseMinCtx)
        let loadedCoarse = useCoarse
            ? try QWCoarseSelectionPipeline.LoadedPipelines.load(hal: hal, variant: .mla)
            : nil
        if let rawEnv = ProcessInfo.processInfo.environment["QW_MLA_EMEANS"]?.lowercased() {
            self.emeansCoarseEnabled = useCoarse && rawEnv != "0" && rawEnv != "false" && rawEnv != "off"
        } else {
            self.emeansCoarseEnabled = useCoarse && lsaConfig.mode != .off
        }
        self.emptyClusteredMacros = Array(repeating: 0, count: numAttnLayers)

        var cursor = 0
        func align256(_ x: Int) -> Int { (x + 255) & ~255 }
        func reserve(_ bytes: Int) -> Int {
            let off = cursor
            cursor = align256(cursor + max(256, bytes))
            return off
        }

        let candCap = dsa ? Self.dsaEmeansCandCap : Self.emeansCandCap
        let maxMacros = useCoarse ? QWCoarsePoolIndex.maxMacros(forMaxContext: self.maxContext, poolShift: self.poolShift) : 0
        var layerOffs: [(kPend: Int, gatePend: Int, pooled: Int, kPe: Int, hasIdx: Bool, csr: QWCoarseSelectionPipeline.CSROffsets)] = []
        layerOffs.reserveCapacity(numAttnLayers)
        for i in 0..<numAttnLayers {
            let layerHasIdx = self.hasIndexer && !(mtpDenseOnly && numAttnLayers > 0 && i == numAttnLayers - 1)
            let kOff = (dsa || !layerHasIdx) ? 0 : reserve(self.ringCap * self.idxDim * 2)
            let gOff = (dsa || !layerHasIdx) ? 0 : reserve(self.ringCap * self.idxDim * 2)
            let pOff = layerHasIdx ? reserve(((self.maxBlocks + 7) & ~7) * self.idxDim * 2) : reserve(0)
            let peRows = dsa ? max(self.maxBlocks, self.maxContext) : 0
            let peOff = (dsa && self.ropeDim > 0) ? reserve(((peRows + 7) & ~7) * self.ropeDim * 2) : 0
            let csr = QWCoarseSelectionPipeline.CSROffsets.reserve(
                enabled: useCoarse && layerHasIdx, maxMacros: maxMacros, indexDim: self.idxDim, reserve: reserve)
            layerOffs.append((kOff, gOff, pOff, peOff, layerHasIdx, csr))
        }

        let maxT = self.maxPrefillChunk
        let selectBatch = self.hasIndexer
            ? min(maxT, max(Self.minSelectBatchRows, Self.selectScratchBudgetBytes / max(1, self.maxBlocks * 4)))
            : 1
        self.selectBatchRows = selectBatch
        let maxLatentCols = max(
            self.latentStride,
            self.qLoraRank + self.kvLoraRank + self.ropeDim + self.idxDim + self.qHeads * self.vHeadDim
        )
        let latentOff = reserve(maxT * maxLatentCols * 2)
        let qResidCols = max(1, self.qLoraRank)
        let qResidOff = reserve(self.qLoraRank > 0 ? maxT * self.qLoraRank * 2 : 0)
        let qFullOff = reserve(maxT * self.qFullStride * 2)
        let idxWOff = reserve(self.hasIndexer ? maxT * self.idxHeads * 4 : 0)
        let hasFlashScratch = self.hasIndexer && self.idxHeads > Self.flashIndexerSelHeads
        let flashQSelOff = reserve(hasFlashScratch ? maxT * Self.flashIndexerSelHeads * self.idxDim * 2 : 0)
        let flashWSelOff = reserve(hasFlashScratch ? maxT * Self.flashIndexerSelHeads * 4 : 0)
        let qAbsOff = reserve(maxT * self.qHeads * self.kvLoraRank * 2)
        let uLatOff = reserve(maxT * self.qHeads * self.kvLoraRank * 2)
        let yVhaNeeded = dsa || self.outputGateMode != .none || geometry.attnOutputGate || (mla?.useVHA == true) || oGateInQKV
        let yVhaOff = reserve(yVhaNeeded ? maxT * self.qHeads * self.vHeadDim * 2 : 0)
        let oGateCols = self.outputGateMode == .headWiseFromHidden
            ? self.qHeads
            : (dsa ? max(1, self.qHeads * self.vHeadDim)
                   : (self.outputGateMode == .elementWiseFromHidden ? self.qHeads * self.vHeadDim : 1))
        let hasExternalGate = dsa || self.outputGateMode == .headWiseFromHidden || self.outputGateMode == .elementWiseFromHidden
        let oGateOff = reserve(hasExternalGate ? maxT * oGateCols * 2 : 0)
        let invFreqCount = max(1, invFreqArr.count)
        let invFreqOff = reserve(invFreqCount * 4)
        let zeroSinkCount = max(16, self.qHeads)
        let zeroSinkOff = reserve(zeroSinkCount * 4)
        let mropePosOff = reserve(3 * maxT * 4)
        let ropeAxisOff = reserve(ropeAxisCount * 4)
        let scoreBlocks = self.hasIndexer ? self.maxBlocks : 1
        let scoreOff = reserve(self.hasIndexer ? selectBatch * self.maxBlocks * 4 : 0)
        let listOff = reserve(self.hasIndexer ? maxT * self.listStride * 4 : 0)
        let countOff = reserve(self.hasIndexer ? maxT * 4 : 0)
        let splitRows = min(maxT, Self.splitMaxRows) * max(1, self.attnSplits) * self.qHeads
        let partOOff = reserve(self.attnSplits > 1 ? splitRows * self.kvLoraRank * 2 : 0)
        let partMLOff = reserve(self.attnSplits > 1 ? splitRows * 8 : 0)
        let maxClusters = maxMacros * QWCoarsePoolIndex.clustersPerMacro
        let coarseScratchOffs = QWCoarseSelectionPipeline.ScratchOffsets.reserve(
            enabled: useCoarse, maxBatchRows: selectBatch,
            maxClusters: maxClusters, candCap: candCap, reserve: reserve)

        let totalBytes = max(4096, cursor)
        guard let buf = hal.device.makeBuffer(length: totalBytes, options: .storageModeShared) else {
            throw QWAllocError.outOfSpace(region: "mla-cache", asked: totalBytes, free: 0)
        }
        buf.label = "mla-cache"
        let bufPtr = buf.contents()
        memset(bufPtr, 0, totalBytes)
        invFreqArr.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let src = raw.baseAddress, !raw.isEmpty {
                bufPtr.advanced(by: invFreqOff).copyMemory(from: src, byteCount: raw.count)
            }
        }
        let negInf = Array(repeating: -Float.infinity, count: zeroSinkCount)
        negInf.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let src = raw.baseAddress {
                bufPtr.advanced(by: zeroSinkOff).copyMemory(from: src, byteCount: raw.count)
            }
        }
        ropeAxisMap.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let src = raw.baseAddress, !raw.isEmpty {
                bufPtr.advanced(by: ropeAxisOff).copyMemory(from: src, byteCount: raw.count)
            }
        }
        let reg = QWMRegion(buffer: buf, base: 0, length: totalBytes)
        self.region = reg

        var builtLayers: [LayerState] = []
        builtLayers.reserveCapacity(numAttnLayers)
        let storedIdxCols = max(1, self.idxDim)
        let peCapRows = max(self.maxBlocks, self.maxContext)
        for i in 0..<numAttnLayers {
            let o = layerOffs[i]
            let csr: QWCoarsePoolIndex.LayerCSR? = (useCoarse && o.hasIdx)
                ? o.csr.makeLayerCSR(region: reg, maxMacros: maxMacros, indexDim: self.idxDim)
                : nil
            let kPe: QWTensorValue? = (dsa && self.ropeDim > 0)
                ? QWTensorValue(region: reg, offset: o.kPe, shape: QWTensorShape([peCapRows, self.ropeDim], .f16))
                : nil
            let storedPooledRows = o.hasIdx ? self.maxBlocks : 1
            builtLayers.append(LayerState(
                fullAttentionIndex: i,
                kPending: QWTensorValue(region: reg, offset: o.kPend, shape: QWTensorShape([dsa || !o.hasIdx ? 1 : self.ringCap, storedIdxCols], .f16)),
                gatePending: QWTensorValue(region: reg, offset: o.gatePend, shape: QWTensorShape([dsa || !o.hasIdx ? 1 : self.ringCap, storedIdxCols], .f16)),
                pooledKeys: QWTensorValue(region: reg, offset: o.pooled, shape: QWTensorShape([storedPooledRows, storedIdxCols], .f16)),
                kPeCache: kPe,
                coarseCSR: csr,
                hasIndexer: o.hasIdx
            ))
        }
        self.layers = builtLayers
        self.coarsePipeline = loadedCoarse.map { loaded in
            let cScratch = coarseScratchOffs.makeScratch(
                region: reg, maxBatchRows: selectBatch, maxClusters: maxClusters, candCap: candCap)
            return QWCoarseSelectionPipeline(
                variant: .mla, loaded: loaded, scratch: cScratch, layerCount: numAttnLayers)
        }

        self.mlaLatentScratch = QWTensorValue(region: reg, offset: latentOff, shape: QWTensorShape([maxT, self.latentStride], .f16))
        self.qResidScratch = QWTensorValue(region: reg, offset: qResidOff, shape: QWTensorShape([maxT, qResidCols], .f16))
        self.mlaQFullScratch = QWTensorValue(region: reg, offset: qFullOff, shape: QWTensorShape([maxT, self.qFullStride], .f16))
        self.idxWeightsScratch = QWTensorValue(region: reg, offset: idxWOff, shape: QWTensorShape([maxT, max(1, self.idxHeads)], .f32))
        self.flashQSelScratch = QWTensorValue(region: reg, offset: flashQSelOff, shape: QWTensorShape([maxT, Self.flashIndexerSelHeads * max(1, self.idxDim)], .f16))
        self.flashWSelScratch = QWTensorValue(region: reg, offset: flashWSelOff, shape: QWTensorShape([maxT, Self.flashIndexerSelHeads], .f32))
        self.qAbsScratch = QWTensorValue(region: reg, offset: qAbsOff, shape: QWTensorShape([maxT, self.qHeads * self.kvLoraRank], .f16))
        self.uLatScratch = QWTensorValue(region: reg, offset: uLatOff, shape: QWTensorShape([maxT, self.qHeads * self.kvLoraRank], .f16))
        self.yVhaScratch = QWTensorValue(region: reg, offset: yVhaOff, shape: QWTensorShape([yVhaNeeded ? maxT : 1, self.qHeads * self.vHeadDim], .f16))
        self.oGateScratch = QWTensorValue(region: reg, offset: oGateOff, shape: QWTensorShape([maxT, oGateCols], .f16))
        self.invFreqBuf = QWTensorValue(region: reg, offset: invFreqOff, shape: QWTensorShape([invFreqCount], .f32))
        self.zeroSinkBuf = QWTensorValue(region: reg, offset: zeroSinkOff, shape: QWTensorShape([zeroSinkCount], .f32))
        self.mropePosScratch = QWTensorValue(region: reg, offset: mropePosOff, shape: QWTensorShape([3 * maxT], .i32))
        self.ropeAxisMapBuf = QWTensorValue(region: reg, offset: ropeAxisOff, shape: QWTensorShape([ropeAxisCount], .i32))
        self.scoreScratch = QWTensorValue(region: reg, offset: scoreOff, shape: QWTensorShape([selectBatch, scoreBlocks], .f32))
        self.blockListScratch = QWTensorValue(region: reg, offset: listOff, shape: QWTensorShape([maxT, self.listStride], .i32))
        self.blockCountScratch = QWTensorValue(region: reg, offset: countOff, shape: QWTensorShape([maxT], .i32))
        self.partOScratch = QWTensorValue(region: reg, offset: partOOff, shape: QWTensorShape([max(1, splitRows), self.kvLoraRank], .f16))
        self.partMLScratch = QWTensorValue(region: reg, offset: partMLOff, shape: QWTensorShape([max(1, splitRows), 2], .f32))
    }

}
