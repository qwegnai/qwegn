// QWMLAAttnOperator+Types.swift — Weight and KV state descriptors for MLA sparse attention.
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWKV

extension QWMLAAttnOperator {
    /// Optional low-rank bottleneck `GatedNorm` weights for `q_a_layernorm` (`1536`) or `kv_a_layernorm` (`512`).
    public struct GatedLatentNorm: Sendable {
        public let weight: QWTensorValue?
        public let gateUp: QWTensorValue
        public let gateDown: QWTensorValue
        public let rank: Int
        public let gatingWeight: Float
        public let zeroCentered: Bool

        public init(weight: QWTensorValue? = nil,
                    gateUp: QWTensorValue,
                    gateDown: QWTensorValue,
                    rank: Int = 16,
                    gatingWeight: Float = 2.0,
                    zeroCentered: Bool = false) {
            self.weight = weight
            self.gateUp = gateUp
            self.gateDown = gateDown
            self.rank = rank
            self.gatingWeight = gatingWeight
            self.zeroCentered = zeroCentered
        }
    }

    /// Per-session KV cache views, cursor position, and MLA/DSA state for `B = 1..8` batched decode waves.
    public struct DecodeBatchSlot: Sendable {
        public let kv: (k: QWKVLayerView, v: QWKVLayerView)
        public let position: Int
        public let state: LayerState
        public let mropePositions: [Int32]?

        public init(kv: (k: QWKVLayerView, v: QWKVLayerView),
                    position: Int,
                    state: LayerState,
                    mropePositions: [Int32]? = nil) {
            self.kv = kv
            self.position = position
            self.state = state
            self.mropePositions = mropePositions
        }
    }

    public struct LayerWeights: Sendable {
        public let qANormW: QWTensorValue
        public let kvANormW: QWTensorValue
        public let kNormW: QWTensorValue
        public let kNormB: QWTensorValue
        public let wGateWeights: QWTensorValue
        public let ape: QWTensorValue?
        public let qBProj: QWWeightView
        public let kvBProj: QWWeightView
        public let vhaMix: QWTensorValue?
        public let attnSink: QWTensorValue?
        public let hasOGate: Bool
        public let oGateF16: QWTensorValue?
        public let oGateProj: QWWeightView?
        public let flashGateW: QWTensorValue?
        public let flashBias: QWTensorValue?
        public let qAGatedNorm: GatedLatentNorm?
        public let kvAGatedNorm: GatedLatentNorm?
        public let alphaQ: Float?
        public let alphaKV: Float?
        public let indexerKNormIsRMS: Bool?
        public let softmaxScaleOverride: Float?
        public let denseOnly: Bool
        public let oGateInQKV: Bool

        public init(qANormW: QWTensorValue,
                    kvANormW: QWTensorValue,
                    kNormW: QWTensorValue,
                    kNormB: QWTensorValue,
                    wGateWeights: QWTensorValue,
                    ape: QWTensorValue?,
                    qBProj: QWWeightView,
                    kvBProj: QWWeightView,
                    vhaMix: QWTensorValue? = nil,
                    attnSink: QWTensorValue? = nil,
                    hasOGate: Bool = false,
                    oGateF16: QWTensorValue? = nil,
                    oGateProj: QWWeightView? = nil,
                    flashGateW: QWTensorValue? = nil,
                    flashBias: QWTensorValue? = nil,
                    qAGatedNorm: GatedLatentNorm? = nil,
                    kvAGatedNorm: GatedLatentNorm? = nil,
                    alphaQ: Float? = nil,
                    alphaKV: Float? = nil,
                    indexerKNormIsRMS: Bool? = nil,
                    softmaxScaleOverride: Float? = nil,
                    denseOnly: Bool = false,
                    oGateInQKV: Bool = false) {
            self.qANormW = qANormW
            self.kvANormW = kvANormW
            self.kNormW = kNormW
            self.kNormB = kNormB
            self.wGateWeights = wGateWeights
            self.ape = ape
            self.qBProj = qBProj
            self.kvBProj = kvBProj
            self.vhaMix = vhaMix
            self.attnSink = attnSink
            self.hasOGate = hasOGate
            self.oGateF16 = oGateF16
            self.oGateProj = oGateProj
            self.flashGateW = flashGateW
            self.flashBias = flashBias
            self.qAGatedNorm = qAGatedNorm
            self.kvAGatedNorm = kvAGatedNorm
            self.alphaQ = alphaQ
            self.alphaKV = alphaKV
            self.indexerKNormIsRMS = indexerKNormIsRMS
            self.softmaxScaleOverride = softmaxScaleOverride
            self.denseOnly = denseOnly
            self.oGateInQKV = oGateInQKV
        }
    }

    public struct LayerState: Sendable {
        public let fullAttentionIndex: Int
        public let kPending: QWTensorValue
        public let gatePending: QWTensorValue
        public let pooledKeys: QWTensorValue
        public let kPeCache: QWTensorValue?
        public let coarseCSR: QWCoarsePoolIndex.LayerCSR?
        public let hasIndexer: Bool

        public init(fullAttentionIndex: Int,
                    kPending: QWTensorValue,
                    gatePending: QWTensorValue,
                    pooledKeys: QWTensorValue,
                    kPeCache: QWTensorValue?,
                    coarseCSR: QWCoarsePoolIndex.LayerCSR?,
                    hasIndexer: Bool = true) {
            self.fullAttentionIndex = fullAttentionIndex
            self.kPending = kPending
            self.gatePending = gatePending
            self.pooledKeys = pooledKeys
            self.kPeCache = kPeCache
            self.coarseCSR = coarseCSR
            self.hasIndexer = hasIndexer
        }
    }

    public static let minSelectBatchRows = 32
    public static let emeansProbeClusters = 320
    public static let emeansCandCap = 10240
    public static let dsaEmeansProbeClusters = 480
    public static let dsaEmeansCandCap = 16384
    public static let selectScratchBudgetBytes = 16 << 20
    public static let flashIndexerSelHeads = 8

    /// Returns whether a call on `mlaW` runs dense-only MLA (`idxHeads = 0, idxDim = 0, topK = 0`).
    public func isLayerDenseOnly(_ mlaW: LayerWeights, state: LayerState? = nil) -> Bool {
        if denseOnly || mlaW.denseOnly || !hasIndexer { return true }
        if let state, !state.hasIndexer { return true }
        return false
    }

    /// Effective QKV/latent row stride (`[q_a; kv_a; k_pe; (wk); (o_gate)]`) for `mlaW`.
    public func latentStride(for mlaW: LayerWeights, state: LayerState? = nil) -> Int {
        let effDense = isLayerDenseOnly(mlaW, state: state)
        let effIdxDim = effDense ? 0 : idxDim
        let effOGateInQKV = oGateInQKV || mlaW.oGateInQKV
        let gateCols = effOGateInQKV
            ? ((outputGateMode == .headWiseFromHidden) ? qHeads : (qHeads * vHeadDim))
            : 0
        return qLoraRank + kvLoraRank + ropeDim + effIdxDim + gateCols
    }

    /// Effective main-attention query channels (excluding indexer `q_idx`) for `mlaW`.
    public func qChannels(for mlaW: LayerWeights) -> Int {
        let effOGateInQKV = oGateInQKV || mlaW.oGateInQKV
        let hasSeparateGate = effOGateInQKV || mlaW.oGateProj != nil || mlaW.oGateF16 != nil
            || outputGateMode == .headWiseFromHidden || outputGateMode == .elementWiseFromHidden
        let hasInlineGate = !hasSeparateGate && (
            outputGateMode == .elementWiseFromQLora
            || mlaW.hasOGate
            || (geometry.mlaConfig == nil && geometry.attnOutputGate)
        )
        return qHeads * (nopeDim + ropeDim + (hasInlineGate ? vHeadDim : 0))
    }

    /// Effective `mlaQFullScratch` row stride (`[q_nope; q_rope; (inline_gate); (q_idx)]`) for `mlaW`.
    public func qFullStride(for mlaW: LayerWeights, state: LayerState? = nil) -> Int {
        let effDense = isLayerDenseOnly(mlaW, state: state)
        let effIdxHeads = effDense ? 0 : idxHeads
        let effIdxDim = effDense ? 0 : idxDim
        return qChannels(for: mlaW) + effIdxHeads * effIdxDim
    }

    /// Effective softmax scale (`> 0` override or `0.0` so Metal uses `rsqrt(float(headDim))`).
    public func effectiveSoftmaxScale(for mlaW: LayerWeights) -> Float {
        if let s = mlaW.softmaxScaleOverride, s > 0 { return s }
        if let s = softmaxScaleOverride, s > 0 { return s }
        return 0.0
    }
}
