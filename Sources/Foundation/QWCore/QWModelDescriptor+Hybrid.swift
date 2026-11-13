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

    /// Gated FFN nonlinearity (`SwiGLU` vs bounded `SituAndMul`).
    public enum ActivationKind: Sendable, Hashable, Codable, Equatable {
        case silu
        case situ(beta: Float, linearBeta: Float)

        public var isSitu: Bool {
            if case .situ = self { return true }
            return false
        }
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

    /// True when the model executes MLA (either NoPE MLA or hybrid decoupled-RoPE MLA-DSA) on its full-attention layers.
    public var hasMLAAttention: Bool {
        (mlaUseNoPE && (kvLoraRank ?? 0) > 0) || hybridMLAConfig != nil
    }

    /// True when NextN / MTP layer `mtpOffset` (`0..<mtpLayers`) runs MLA-DSA rather than HCA.
    public func isNextNLayerDSA(_ mtpOffset: Int = 0) -> Bool {
        let idx = numLayers + max(0, mtpOffset)
        if let ratios = hcaConfig?.compressRatios, idx < ratios.count {
            return ratios[idx] <= 0
        }
        return hasMLAAttention
    }

    /// Resolves the RoPE-free key projection tensor slice for training-free Weight-SVD LSA synthesis.
    public func lsaKeyProjectionSpec(layer l: Int) -> LSAKeyProjectionSpec? {
        guard (0..<numLayers).contains(l) else { return nil }
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
        if isFullAttention(l) && !hasMLAAttention && capabilities.contains(.qsaIndexer) == false && lsaInputDim >= 64 {
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
}
