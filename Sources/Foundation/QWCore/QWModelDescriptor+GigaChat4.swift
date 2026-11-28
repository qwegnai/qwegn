// QWModelDescriptor+GigaChat4.swift — GatedNorm, GDNGating, decoupled MLA scaling, and NextN MTP config decoding.
import Foundation

/// Low-rank SwiGLU-sigmoid gated RMSNorm capability (`norm_type = "GatedNorm"` / `"ZeroCenteredGatedNorm"`).
public struct GatedNormConfig: Sendable, Hashable, Equatable, Codable {
    public let rank: Int
    public let gatingWeight: Float
    public let zeroCentered: Bool

    public init(rank: Int = 16, gatingWeight: Float = 2.0, zeroCentered: Bool = false) {
        self.rank = rank
        self.gatingWeight = gatingWeight
        self.zeroCentered = zeroCentered
    }
}

/// Output normalization and gating parameters for GatedDeltaNet linear-attention blocks.
public struct GDNGatingConfig: Sendable, Hashable, Equatable, Codable {
    public let outputNormAct: QWModelDescriptor.KDAConfig.NormActivation
    public let outputGateScale: Float
    public let outputNormEps: Float
    public let zeroCentered: Bool

    public init(
        outputNormAct: QWModelDescriptor.KDAConfig.NormActivation = .sigmoid,
        outputGateScale: Float = 1.0,
        outputNormEps: Float = 1e-6,
        zeroCentered: Bool = false
    ) {
        self.outputNormAct = outputNormAct
        self.outputGateScale = outputGateScale
        self.outputNormEps = outputNormEps
        self.zeroCentered = zeroCentered
    }
}

extension QWModelDescriptor {
    public typealias GatedNormConfig = QWCore.GatedNormConfig
    public typealias GDNGatingConfig = QWCore.GDNGatingConfig

    internal static func resolveExplicitFullAttentionLayers(
        text t: ConfigEnvelope.Text,
        inv: (Bool, String) throws -> Void
    ) throws -> [String]? {
        guard t.full_attention_layers != nil || t.linear_attention_type != nil else { return nil }
        if let linType = t.linear_attention_type {
            let norm = linType.lowercased()
            try inv(norm == "gigachat4gateddeltanet" || norm == "gated_deltanet" || norm == "gdn",
                    "unsupported linear_attention_type '\(linType)'")
        }
        let faList = t.full_attention_layers ?? []
        var faSet = Set<Int>()
        for idx in faList {
            try inv(idx >= 0 && idx < t.num_hidden_layers,
                    "full_attention_layers index \(idx) out of range 0..<\(t.num_hidden_layers)")
            faSet.insert(idx)
        }
        return (0..<t.num_hidden_layers).map { faSet.contains($0) ? "full_attention" : "linear_attention" }
    }

    internal static func resolveDenseAttentionLayerSet(
        text t: ConfigEnvelope.Text,
        inv: (Bool, String) throws -> Void
    ) throws -> Set<Int> {
        guard let list = t.dense_attention_layers, !list.isEmpty else { return [] }
        var out = Set<Int>()
        for idx in list {
            try inv(idx >= 0 && idx < t.num_hidden_layers,
                    "dense_attention_layers index \(idx) out of range 0..<\(t.num_hidden_layers)")
            out.insert(idx)
        }
        return out
    }

    internal static func decodeGatedNormConfig(
        text t: ConfigEnvelope.Text,
        inv: (Bool, String) throws -> Void
    ) throws -> GatedNormConfig? {
        if let lnType = t.layernorm_type {
            try inv(lnType == "pre_post" || lnType == "pre", "unsupported layernorm_type '\(lnType)'")
        }
        guard let rawNormType = t.norm_type else { return nil }
        let weight = t.layernorm_gating_weight ?? 2.0
        try inv(weight > 0 && weight.isFinite, "layernorm_gating_weight \(weight) invalid")
        switch rawNormType {
        case "GatedNorm":
            return GatedNormConfig(rank: 16, gatingWeight: weight, zeroCentered: false)
        case "ZeroCenteredGatedNorm":
            return GatedNormConfig(rank: 16, gatingWeight: weight, zeroCentered: true)
        case "RMSNorm", "LlamaRMSNorm", "ZeroCenteredRMSNorm", "LayerNorm":
            return nil
        default:
            throw DescriptorError.invariant("unsupported norm_type '\(rawNormType)'")
        }
    }

    internal static func decodeGDNGatingConfig(
        text t: ConfigEnvelope.Text,
        hasLinearLayers: Bool,
        resolvedNormEps: Double,
        inv: (Bool, String) throws -> Void
    ) throws -> GDNGatingConfig? {
        guard hasLinearLayers else { return nil }
        guard t.linear_gating_type != nil || t.linear_sigmoid_gate_scale != nil || t.linear_attn_o_norm_eps != nil else {
            return nil
        }
        let eps = Float(t.linear_attn_o_norm_eps ?? resolvedNormEps)
        try inv(eps > 0 && eps.isFinite, "linear_attn_o_norm_eps \(eps) invalid")
        let gType = t.linear_gating_type ?? "gated_rmsnorm"
        let scale = t.linear_sigmoid_gate_scale ?? 1.0
        try inv(scale > 0 && scale.isFinite, "linear_sigmoid_gate_scale \(scale) invalid")
        switch gType {
        case "gated_rmsnorm":
            return GDNGatingConfig(outputNormAct: .silu, outputGateScale: 1.0, outputNormEps: eps, zeroCentered: false)
        case "gated_rmsnorm_sigmoid":
            return GDNGatingConfig(outputNormAct: .sigmoid, outputGateScale: scale, outputNormEps: eps, zeroCentered: false)
        case "zero_centered_gated_rmsnorm":
            return GDNGatingConfig(outputNormAct: .silu, outputGateScale: 1.0, outputNormEps: eps, zeroCentered: true)
        case "zero_centered_gated_rmsnorm_sigmoid":
            return GDNGatingConfig(outputNormAct: .sigmoid, outputGateScale: scale, outputNormEps: eps, zeroCentered: true)
        default:
            throw DescriptorError.invariant("unsupported linear_gating_type '\(gType)'")
        }
    }

    internal static func resolveMLAScalingAndNormTraits(
        text t: ConfigEnvelope.Text,
        familySpec: QWModelFamilySpec,
        qLoraRank: Int,
        kvLoraRank: Int,
        inv: (Bool, String) throws -> Void
    ) throws -> (alphaQ: Float, alphaKV: Float, indexerKNormIsRMS: Bool) {
        let useScale = t.use_mla_scaling_factor ?? false
        let h = Double(t.hidden_size)
        let alphaQ: Float = (useScale && qLoraRank > 0) ? Float((h / Double(qLoraRank)).squareRoot()) : 1.0
        let alphaKV: Float = (useScale && kvLoraRank > 0) ? Float((h / Double(kvLoraRank)).squareRoot()) : 1.0
        let idxRMS: Bool
        if let idxNorm = t.indexer_norm_type {
            switch idxNorm {
            case "LlamaRMSNorm", "RMSNorm", "ZeroCenteredRMSNorm", "GatedNorm", "ZeroCenteredGatedNorm":
                idxRMS = true
            case "LayerNorm":
                idxRMS = false
            default:
                throw DescriptorError.invariant("unsupported indexer_norm_type '\(idxNorm)'")
            }
        } else {
            idxRMS = familySpec.defaultIndexerKNormIsRMS
        }
        return (alphaQ, alphaKV, idxRMS)
    }

    internal static func decodeFinalLogitSoftcapping(
        text t: ConfigEnvelope.Text,
        inkling2: Inkling2Config?,
        inv: (Bool, String) throws -> Void
    ) throws -> Float? {
        if let cap = t.final_logit_softcapping {
            try inv(cap > 0 && cap.isFinite, "final_logit_softcapping \(cap) invalid")
            return cap
        }
        return inkling2?.finalLogitSoftcapping
    }

    internal static func decodeBlockSparseGQAConfig(
        text t: ConfigEnvelope.Text,
        inv: (Bool, String) throws -> Void
    ) throws -> BlockSparseGQAConfig? {
        guard t.attention_variant == "block_sparse_gqa" else { return nil }
        guard let blk = t.block_sparse_gqa_block_size,
              let pfxWin = t.block_sparse_gqa_prefix_window_size,
              let locWin = t.block_sparse_gqa_local_window_size,
              let remote = t.block_sparse_gqa_num_remote_blocks,
              let idxDim = t.block_sparse_gqa_indexer_head_dim,
              let idxHeads = t.block_sparse_gqa_indexer_n_heads,
              blk > 0, pfxWin > 0, locWin > 0, remote > 0,
              pfxWin % blk == 0, locWin % blk == 0,
              idxHeads > 0, idxDim > 0, idxDim % 32 == 0 else {
            throw DescriptorError.invariant("block_sparse_gqa parameters missing or invalid")
        }
        return BlockSparseGQAConfig(
            blockSize: blk,
            prefixBlocks: pfxWin / blk,
            localBlocks: locWin / blk,
            remoteBlocks: remote,
            indexerHeads: idxHeads,
            indexerHeadDim: idxDim
        )
    }

    internal static func appendGigaChat4V3DigestFields(_ d: QWModelDescriptor, into f: inout [String]) {
        if let gn = d.gatedNormConfig {
            let wHex = String(gn.gatingWeight.bitPattern, radix: 16, uppercase: false)
            f.append("gnorm=\(gn.rank),0x\(wHex),\(gn.zeroCentered ? 1 : 0)")
        }
        if let gg = d.gdnGatingConfig {
            let sHex = String(gg.outputGateScale.bitPattern, radix: 16, uppercase: false)
            let eHex = String(gg.outputNormEps.bitPattern, radix: 16, uppercase: false)
            f.append("gdn_gate=\(gg.outputNormAct.rawValue),0x\(sHex),0x\(eHex),\(gg.zeroCentered ? 1 : 0)")
        }
        if let mla = d.mlaConfig, (mla.alphaQ != 1.0 || mla.alphaKV != 1.0 || mla.indexerKNormIsRMS || !mla.denseAttentionLayers.isEmpty) {
            let aqHex = String(mla.alphaQ.bitPattern, radix: 16, uppercase: false)
            let akvHex = String(mla.alphaKV.bitPattern, radix: 16, uppercase: false)
            let smHex = String(mla.softmaxScale.bitPattern, radix: 16, uppercase: false)
            let denseStr = mla.denseAttentionLayers.sorted().map(String.init).joined(separator: ",")
            f.append("mla_scale=0x\(aqHex),0x\(akvHex),0x\(smHex),\(mla.indexerKNormIsRMS ? 1 : 0),\(denseStr)")
        }
        if let cap = d.finalLogitSoftcapping, d.inkling2Config == nil {
            f.append("softcap=0x\(String(cap.bitPattern, radix: 16, uppercase: false))")
        }
        if d.headDim != d.v2DigestHeadDim {
            f.append("eff_hd=\(d.headDim)")
        }
    }
}
