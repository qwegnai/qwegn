import Foundation
import QWCore
import QWQuant

// Pack-backed expert source: each expert is dequantized on demand from the HQ4 expert stacks. The pack is
// mmap-backed and read-only, so concurrent reads are safe; a dequant failure here is a loader bug, not input.
public final class FNPackExperts: FNLatentExpertSource, @unchecked Sendable {
    let pack: QWQuant.Pack
    let gateUpName: String
    let downName: String
    let latentDName: String?
    let latentUName: String?

    public init(pack: QWQuant.Pack, gateUpName: String, downName: String,
                latentDName: String? = nil, latentUName: String? = nil) {
        self.pack = pack; self.gateUpName = gateUpName; self.downName = downName
        self.latentDName = latentDName; self.latentUName = latentUName
    }

    public func gateUp(_ e: Int) -> [Float] {
        // Geometry is validated when the pack is opened; a failure here is a loader bug.
        try! pack.dequantExpertQ4g(gateUpName, expert: e)
    }

    public func down(_ e: Int) -> [Float] {
        try! pack.dequantExpertQ4g(downName, expert: e)
    }

    public func latentD(_ g: Int) -> [Float] {
        try! pack.dequantExpertQ4g(latentDName!, expert: g)
    }

    public func latentU(_ g: Int) -> [Float] {
        try! pack.dequantExpertQ4g(latentUName!, expert: g)
    }
}

// Loads FlashNext decoder layers from a QWP1 HQ4 pack or a SafeTensors container.
// Names follow the pack: `model.language_model.layers.N.*`.
public struct FNPackLoader {
    public let pack: QWQuant.Pack?
    public let st: QWSafeTensorsContainer?
    public let descriptor: QWModelDescriptor
    public let isMLXCheckpoint: Bool

    static let prefix = "model.language_model."

    public init(pack: QWQuant.Pack, descriptor: QWModelDescriptor) {
        self.pack = pack
        self.st = nil
        self.descriptor = descriptor
        self.isMLXCheckpoint = false
    }

    public init(st: QWSafeTensorsContainer, descriptor: QWModelDescriptor) {
        self.pack = nil
        self.st = st
        self.descriptor = descriptor
        self.isMLXCheckpoint = st.isMLXCheckpoint
    }

    public func layerPrefix(_ layer: Int) -> String { "\(Self.prefix)layers.\(layer)." }

    /// Dense fp32 tensor: bf16 raw entries verbatim, HQ4 2-D entries or SafeTensors entries dequantized.
    public func tensor(_ name: String) throws -> [Float] {
        if let pack {
            if try pack.entry(name).isRawBf16 { return try pack.rawBf16(name) }
            return try pack.dequantQ4g(name)
        }
        guard let st else { throw QWSafeTensorsError.missingTensor(name) }
        var vals = try st.readFloats(name)
        if isMLXCheckpoint && name.contains("norm") && !name.hasSuffix("linear_attn.norm.weight") {
            for i in 0..<vals.count { vals[i] -= 1.0 }
        }
        return vals
    }

    public func hyperConnection(prefix: String, withInjection: Bool) throws -> FNHyperConnection {
        let hc = descriptor.hcCount!, lowrank = descriptor.hcLowrank!
        return FNHyperConnection(
            hc: hc, hidden: descriptor.hidden, lowrank: lowrank, eps: descriptor.rmsEps,
            normWeight: try tensor(prefix + "hc_norm.weight"),
            downWeight: try tensor(prefix + "input_mix_weight_down.weight"),
            upWeight: try tensor(prefix + "input_mix_weight_up.weight"),
            injectWeight: withInjection ? try tensor(prefix + "block_inject_weight.weight") : [])
    }

    public func gatedDeltaNet(layer: Int) throws -> FNGatedDeltaNet {
        let p = layerPrefix(layer) + "linear_attn."
        let d = descriptor
        return FNGatedDeltaNet(
            hidden: d.hidden, kHeads: d.gdnQKHeads, vHeads: d.gdnVHeads, kDim: d.gdnHeadDim, vDim: d.gdnHeadDim,
            convKernel: d.gdnConvKernel, eps: d.rmsEps,
            inProjQKV: try tensor(p + "in_proj_qkv.weight"), inProjZ: try tensor(p + "in_proj_z.weight"),
            inProjB: try tensor(p + "in_proj_b.weight"), inProjA: try tensor(p + "in_proj_a.weight"),
            convWeight: try tensor(p + "conv1d.weight"), dtBias: try tensor(p + "dt_bias"),
            aLog: try tensor(p + "A_log"), normWeight: try tensor(p + "norm.weight"),
            outProj: try tensor(p + "out_proj.weight"))
    }

    /// `indexerBudget` overrides the descriptor's budget (real-weight tests only; nil = descriptor value).
    public func fullAttention(layer: Int, indexerBudget: Int? = nil, prefix: String? = nil) throws -> FNAttention {
        let p = (prefix ?? layerPrefix(layer)) + "self_attn."
        let ix = p + "indexer."
        let d = descriptor
        let indexer = FNQSAIndexer(
            heads: d.indexerHeads!, kvHeads: d.indexerKvHeads!, headDim: d.indexerHeadDim!,
            budget: indexerBudget ?? d.indexerBudget!, compress: d.indexerCompressRatio!, eps: d.rmsEps,
            qkProj: try tensor(ix + "index_qk_proj.weight"),
            qNorm: try tensor(ix + "q_layernorm.weight"), kNorm: try tensor(ix + "k_layernorm.weight"))
        return FNAttention(
            hidden: d.hidden, heads: d.qHeads, kvHeads: d.kvHeads, headDim: d.headDim, eps: d.rmsEps,
            qProj: try tensor(p + "q_proj.weight"), kProj: try tensor(p + "k_proj.weight"),
            vProj: try tensor(p + "v_proj.weight"), oProj: try tensor(p + "o_proj.weight"),
            qNorm: try tensor(p + "q_norm.weight"), kNorm: try tensor(p + "k_norm.weight"),
            indexer: indexer, rope: FNRope(rotaryDim: d.rotaryDims, theta: d.ropeTheta))
    }

    public func moe(layer: Int, normTopK: Bool, prefix: String? = nil) throws -> FNSparseMoE {
        let m = (prefix ?? layerPrefix(layer)) + "mlp."
        let d = descriptor
        var latentGroups = 0, latentDim = 0, latentLabels: [Int] = []
        var latentDName: String? = nil, latentUName: String? = nil
        if let pack,
           let dEntry = try? pack.entry(m + "experts.latent_d"),
           dEntry.shape.count == 3 {
            latentGroups = dEntry.shape[0]
            latentDim = dEntry.shape[1]
            latentLabels = try tensor(m + "latent_labels").map { Int($0.rounded()) }
            latentDName = m + "experts.latent_d"
            latentUName = m + "experts.latent_u"
        }
        let src = FNPackExperts(pack: pack!, gateUpName: m + "experts.gate_up_proj",
                                downName: m + "experts.down_proj",
                                latentDName: latentDName, latentUName: latentUName)
        return FNSparseMoE(
            hidden: d.hidden, experts: d.numExperts!, topK: d.expertsPerTok!,
            intermediate: d.moeIntermediate!, sharedIntermediate: d.sharedExpertIntermediate!, normTopK: normTopK,
            gate: try tensor(m + "gate.weight"),
            source: src,
            sharedGate: try tensor(m + "shared_expert.gate_proj.weight"),
            sharedUp: try tensor(m + "shared_expert.up_proj.weight"),
            sharedDown: try tensor(m + "shared_expert.down_proj.weight"),
            sharedGateScalar: try tensor(m + "shared_expert_gate.weight"),
            latentGroups: latentGroups, latentDim: latentDim, latentLabels: latentLabels,
            latentSource: latentDName != nil ? src : nil)
    }

    /// PLE layer attached to decoder layer `layer` (dims from the descriptor; dilation = ngram size, as in HF).
    public func ple(layer: Int) throws -> FNPLE {
        let p = layerPrefix(layer) + "ple."
        let d = descriptor
        return FNPLE(
            hidden: d.hidden, hc: d.hcCount!, embedDim: d.pleEmbedDim!, convKernel: d.pleConvKernel!,
            dilation: d.ngramSize!, eps: d.rmsEps,
            keyProj: try tensor(p + "key_proj.weight"), valueProj: try tensor(p + "value_proj.weight"),
            normKey: try tensor(p + "norm_key.weight"), normQuery: try tensor(p + "norm_query.weight"),
            normConv: try tensor(p + "norm_conv.weight"), convWeight: try tensor(p + "conv1d.weight"))
    }

    public func pleEmbedding(_ layer: Int, pleRows: (any FNNgramRows)?) -> FNNgramEmbedding? {
        guard let rows = pleRows, let index = descriptor.pleModuleIndices.firstIndex(of: layer) else { return nil }
        return FNNgramEmbedding(
            vocab: descriptor.vocab, ngram: descriptor.ngramSize!, headsPerNgram: descriptor.headsPerNgram!,
            baseVocab: descriptor.ngramVocabSizeBase!, seed: Self.ngramSeed,
            eos: Int64(descriptor.eosTokenId), pleLayerIndex: index, rows: rows)
    }

    /// Text decoder layer `layer` (or MTP layer when `prefix` is supplied).
    public func decoderLayer(_ layer: Int, normTopK: Bool, pleRows: (any FNNgramRows)? = nil,
                             prefix: String? = nil) throws -> FNDecoderLayer {
        let isFull = (prefix != nil) || descriptor.isFullAttention(layer)
        let mixer: FNMixer = isFull
            ? .full(try fullAttention(layer: layer, prefix: prefix))
            : .linear(try gatedDeltaNet(layer: layer))
        let p = prefix ?? layerPrefix(layer)
        var pleLayer: FNPLE? = nil
        var pleEmb: FNNgramEmbedding? = nil
        if prefix == nil, let emb = pleEmbedding(layer, pleRows: pleRows) {
            pleLayer = try ple(layer: layer)
            pleEmb = emb
        }
        return FNDecoderLayer(
            mixer: mixer,
            attnHC: try hyperConnection(prefix: p + "attn_hyper_connection.", withInjection: true),
            mlpHC: try hyperConnection(prefix: p + "mlp_hyper_connection.", withInjection: true),
            moe: try moe(layer: layer, normTopK: normTopK, prefix: prefix),
            ple: pleLayer, pleEmbedding: pleEmb)
    }

    /// N-gram seed of the PLE hashing: absent from the assets config; 1234 is what the pack's router constants use.
    public static let ngramSeed = 1234

    /// Full text model: embedding, every decoder layer, the final mixer and lm_head.
    public func model(normTopK: Bool, pleRows: any FNNgramRows) throws -> FNModel {
        let d = descriptor
        var layers: [FNDecoderLayer] = []
        for i in 0..<d.numLayers {
            layers.append(try decoderLayer(i, normTopK: normTopK, pleRows: pleRows))
        }
        return FNModel(
            vocab: d.vocab, hidden: d.hidden, hc: d.hcCount!,
            embedding: try tensor(Self.prefix + "embed_tokens.weight"), layers: layers,
            finalMixer: try hyperConnection(prefix: Self.prefix + "hyper_connection_mixer.", withInjection: false),
            lmHead: try tensor("lm_head.weight"))
    }
}
