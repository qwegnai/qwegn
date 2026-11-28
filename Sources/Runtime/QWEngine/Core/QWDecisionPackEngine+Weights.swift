// QWDecisionPackEngine+Weights.swift — Zero-family QWP1/SafeTensors loader for generalized Metal decision packs.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWDecision
import QWGEMM
import QWHAL
import QWLoad
import QWQuant

struct QWDecisionNormTensor: @unchecked Sendable {
    let weight: QWTensorValue
    let bias: QWTensorValue?
}

struct QWDecisionConvWeights: @unchecked Sendable {
    let inProj: QWLinearWeight?
    let convW: QWTensorValue
    let convB: QWTensorValue?
    let outProj: QWLinearWeight
    let kernelSize: Int
    let channels: Int
    let isCentered: Bool
    let isGated: Bool
}

struct QWDecisionAttnLayerWeights: @unchecked Sendable {
    let qProj: QWLinearWeight
    let kProj: QWLinearWeight
    let vProj: QWLinearWeight
    let oProj: QWLinearWeight
    let qBias: QWTensorValue?
    let kBias: QWTensorValue?
    let vBias: QWTensorValue?
    let oBias: QWTensorValue?
    let qNorm: QWTensorValue?
    let kNorm: QWTensorValue?
    let gateProj: QWLinearWeight?
    let isSliding: Bool
    let isBidirectional: Bool
    let slidingWindow: Int
    let ropeKind: String
}

enum QWDecisionLayerOperatorWeights: @unchecked Sendable {
    case conv(QWDecisionConvWeights)
    case attn(QWDecisionAttnLayerWeights)
}

struct QWDecisionLoadedLayer: @unchecked Sendable {
    let preBlockNorm: QWDecisionNormTensor?
    let postBlockNorm: QWDecisionNormTensor?
    let op: QWDecisionLayerOperatorWeights
    let preFFNNorm: QWDecisionNormTensor?
    let postFFNNorm: QWDecisionNormTensor?
    let w13: QWLinearWeight?
    let w1: QWLinearWeight?
    let w3: QWLinearWeight?
    let ffnUpBias: QWTensorValue?
    let w2: QWLinearWeight
    let ffnDownBias: QWTensorValue?
}

struct QWDecisionPackWeights: @unchecked Sendable {
    let embedTokens: QWTensorValue
    let posEmbedBF16: QWTensorValue?
    let embedNorm: QWDecisionNormTensor?
    let finalNorm: QWDecisionNormTensor?
    let layers: [QWDecisionLoadedLayer]
    let head: QWDecisionPackHeadWeights
    let usesRotatedBasis: Bool
    let retainedBuffers: [any MTLBuffer]
}

enum QWDecisionWeightSource {
    case qwp(QWQuant.Pack)
    case safetensors(QWSafeTensorsContainer)

    func has(_ name: String) -> Bool {
        switch self {
        case .qwp(let p): return p.tensors[name] != nil
        case .safetensors(let s): return s.tensor(name) != nil
        }
    }

    func adviseDontNeed() {
        switch self {
        case .qwp(let p): p.adviseDontNeed()
        case .safetensors(let s): s.adviseDontNeed()
        }
    }
}

enum QWDecisionWeightLoader {
    static func probeRotatedBasis(root: QWModelRoot, descriptor: QWModelDescriptor) throws -> Bool {
        guard root.kind == .qwp1 else { return false }
        let pack = try QWQuant.Pack(url: root.weights, descriptor: descriptor)
        for (_, entry) in pack.tensors where PlaneLayout.resolve(entry) == .hq4 {
            if entry.hq?.vq != "aq" { return true }
        }
        return false
    }

    static func load(root: QWModelRoot, descriptor: QWModelDescriptor, hal: QWHal) throws -> QWDecisionPackWeights {
        let source: QWDecisionWeightSource
        switch root.kind {
        case .qwp1:
            source = .qwp(try QWQuant.Pack(url: root.weights, descriptor: descriptor))
        case .safetensorsSingle, .safetensorsSharded:
            source = .safetensors(try QWSafeTensorsContainer(url: root.weights, onlyTensors: nil))
        default:
            throw QWDecisionError.backendFailed("unsupported weight container for native decision engine: \(String(describing: root.kind))")
        }
        defer { source.adviseDontNeed() }

        var buffers: [any MTLBuffer] = []
        let hidden = descriptor.hidden
        let vocab = descriptor.vocab
        let ffn = descriptor.denseIntermediate ?? descriptor.hidden
        let qDim = descriptor.qHeads * descriptor.headDim
        let kvDim = descriptor.kvHeads * descriptor.headDim
        let prefix = resolvePrefix(source)

        let embName = resolveName(source, [
            prefix + "embed_tokens.weight",
            prefix + "tok_embeddings.weight",
            prefix + "embeddings.word_embeddings.weight",
            prefix + "embeddings.tok_embeddings.weight",
            "embed_tokens.weight",
            "embeddings.word_embeddings.weight",
        ]) ?? "\(prefix)embed_tokens.weight"
        let embedTokens = try loadBF16Tensor(name: embName, rows: vocab, cols: hidden, source: source, hal: hal, buffers: &buffers)

        let posEmbedName = resolveName(source, [
            prefix + "embeddings.position_embeddings.weight",
            "embeddings.position_embeddings.weight",
        ])
        let posEmbedBF16: QWTensorValue? = try posEmbedName.flatMap { pName in
            guard let dims = tensorShape(name: pName, source: source), dims.count == 2, dims[1] == hidden else { return nil }
            return try loadBF16Tensor(name: pName, rows: dims[0], cols: hidden, source: source, hal: hal, buffers: &buffers)
        }
        let embedNorm = try loadOptionalNorm(
            weightCandidates: [prefix + "embeddings.LayerNorm.weight", prefix + "embeddings.norm.weight", "embeddings.LayerNorm.weight"],
            biasCandidates: [prefix + "embeddings.LayerNorm.bias", prefix + "embeddings.norm.bias", "embeddings.LayerNorm.bias"],
            count: hidden, source: source, hal: hal, buffers: &buffers
        )

        let finalNorm = try loadOptionalNorm(
            weightCandidates: [
                prefix + "embedding_norm.weight", prefix + "norm.weight",
                prefix + "final_norm.weight", prefix + "encoder.norm.weight", "norm.weight",
            ],
            biasCandidates: [
                prefix + "embedding_norm.bias", prefix + "norm.bias",
                prefix + "final_norm.bias", prefix + "encoder.norm.bias", "norm.bias",
            ],
            count: hidden, source: source, hal: hal, buffers: &buffers
        )

        var anyRotated = false
        var loadedLayers: [QWDecisionLoadedLayer] = []
        loadedLayers.reserveCapacity(descriptor.numLayers)

        for l in 0..<descriptor.numLayers {
            let lp = resolveLayerPrefix(source: source, prefix: prefix, layer: l)
            let preBlock = try loadOptionalNorm(
                weightCandidates: [lp + "operator_norm.weight", lp + "input_layernorm.weight", lp + "attn_norm.weight", lp + "attention_norm.weight"],
                biasCandidates: [lp + "operator_norm.bias", lp + "input_layernorm.bias", lp + "attn_norm.bias", lp + "attention_norm.bias"],
                count: hidden, source: source, hal: hal, buffers: &buffers
            )
            let postBlock = try loadOptionalNorm(
                weightCandidates: [lp + "post_operator_norm.weight", lp + "attention.output.LayerNorm.weight", lp + "post_attention_layernorm.weight"],
                biasCandidates: [lp + "post_operator_norm.bias", lp + "attention.output.LayerNorm.bias", lp + "post_attention_layernorm.bias"],
                count: hidden, source: source, hal: hal, buffers: &buffers
            )
            let preFFN = try loadOptionalNorm(
                weightCandidates: [lp + "ffn_norm.weight", lp + "mlp_norm.weight", lp + "post_attention_layernorm.weight", lp + "ffn.norm.weight"],
                biasCandidates: [lp + "ffn_norm.bias", lp + "mlp_norm.bias", lp + "post_attention_layernorm.bias", lp + "ffn.norm.bias"],
                count: hidden, source: source, hal: hal, buffers: &buffers
            )
            let postFFN = try loadOptionalNorm(
                weightCandidates: [lp + "post_ffn_norm.weight", lp + "output.LayerNorm.weight"],
                biasCandidates: [lp + "post_ffn_norm.bias", lp + "output.LayerNorm.bias"],
                count: hidden, source: source, hal: hal, buffers: &buffers
            )

            let lKind = descriptor.layerTypes[l]
            let op: QWDecisionLayerOperatorWeights
            if lKind == "conv" || lKind == "shortconv" || lKind == "centered_conv" {
                let kernel = descriptor.shortConvConfig?.kernelSize ?? 7
                let isCentered = (lKind == "centered_conv") || (descriptor.shortConvConfig?.isCentered == true)
                let inName = resolveName(source, [lp + "conv.in_proj.weight", lp + "short_conv.in_proj.weight"])
                let inProj: QWLinearWeight? = try inName.map { name in
                    let cols = tensorShape(name: name, source: source)?[0] ?? (3 * hidden)
                    return try loadLinear(name: name, n: cols, k: hidden, source: source, hal: hal, buffers: &buffers)
                }
                let convWName = resolveName(source, [lp + "conv.conv.weight", lp + "short_conv.conv.weight", lp + "conv.weight"]) ?? (lp + "conv.conv.weight")
                let convW = try loadF16Vector(name: convWName, count: hidden * kernel, source: source, hal: hal, buffers: &buffers)
                let convB = try resolveName(source, [lp + "conv.conv.bias", lp + "conv.bias", lp + "short_conv.conv.bias"]).map {
                    try loadF16Vector(name: $0, count: hidden, source: source, hal: hal, buffers: &buffers)
                }
                let outName = resolveName(source, [lp + "conv.out_proj.weight", lp + "short_conv.out_proj.weight"]) ?? (lp + "conv.out_proj.weight")
                let outProj = try loadLinear(name: outName, n: hidden, k: hidden, source: source, hal: hal, buffers: &buffers)
                anyRotated = anyRotated || (inProj?.needsRotation ?? false) || outProj.needsRotation
                op = .conv(QWDecisionConvWeights(
                    inProj: inProj, convW: convW, convB: convB, outProj: outProj,
                    kernelSize: kernel, channels: hidden, isCentered: isCentered,
                    isGated: (inProj?.n ?? 0) == 3 * hidden
                ))
            } else {
                let qName = resolveName(source, [lp + "self_attn.q_proj.weight", lp + "attn.q_proj.weight", lp + "attention.self.query.weight"]) ?? (lp + "self_attn.q_proj.weight")
                let kName = resolveName(source, [lp + "self_attn.k_proj.weight", lp + "attn.k_proj.weight", lp + "attention.self.key.weight"]) ?? (lp + "self_attn.k_proj.weight")
                let vName = resolveName(source, [lp + "self_attn.v_proj.weight", lp + "attn.v_proj.weight", lp + "attention.self.value.weight"]) ?? (lp + "self_attn.v_proj.weight")
                let oName = resolveName(source, [lp + "self_attn.o_proj.weight", lp + "attn.Wo.weight", lp + "attn.o_proj.weight", lp + "attention.output.dense.weight"]) ?? (lp + "self_attn.o_proj.weight")

                let qW = try loadLinear(name: qName, n: qDim, k: hidden, source: source, hal: hal, buffers: &buffers)
                let kW = try loadLinear(name: kName, n: kvDim, k: hidden, source: source, hal: hal, buffers: &buffers)
                let vW = try loadLinear(name: vName, n: kvDim, k: hidden, source: source, hal: hal, buffers: &buffers)
                let oW = try loadLinear(name: oName, n: hidden, k: qDim, source: source, hal: hal, buffers: &buffers)

                let qBias = try loadOptionalVector(candidates: [lp + "self_attn.q_proj.bias", lp + "attn.q_proj.bias", lp + "attention.self.query.bias"], count: qDim, source: source, hal: hal, buffers: &buffers)
                let kBias = try loadOptionalVector(candidates: [lp + "self_attn.k_proj.bias", lp + "attn.k_proj.bias", lp + "attention.self.key.bias"], count: kvDim, source: source, hal: hal, buffers: &buffers)
                let vBias = try loadOptionalVector(candidates: [lp + "self_attn.v_proj.bias", lp + "attn.v_proj.bias", lp + "attention.self.value.bias"], count: kvDim, source: source, hal: hal, buffers: &buffers)
                let oBias = try loadOptionalVector(candidates: [lp + "self_attn.o_proj.bias", lp + "attn.Wo.bias", lp + "attn.o_proj.bias", lp + "attention.output.dense.bias"], count: hidden, source: source, hal: hal, buffers: &buffers)

                let qNorm = try loadOptionalVector(candidates: [lp + "self_attn.q_norm.weight", lp + "self_attn.q_layernorm.weight"], count: descriptor.headDim, source: source, hal: hal, buffers: &buffers)
                let kNorm = try loadOptionalVector(candidates: [lp + "self_attn.k_norm.weight", lp + "self_attn.k_layernorm.weight"], count: descriptor.headDim, source: source, hal: hal, buffers: &buffers)
                let gateW = try resolveName(source, [lp + "self_attn.gate_proj.weight"]).map {
                    try loadLinear(name: $0, n: descriptor.qHeads, k: hidden, source: source, hal: hal, buffers: &buffers)
                }
                anyRotated = anyRotated || qW.needsRotation || kW.needsRotation || vW.needsRotation
                    || oW.needsRotation || (gateW?.needsRotation ?? false)
                let sw = descriptor.layerSlidingWindow(l)
                op = .attn(QWDecisionAttnLayerWeights(
                    qProj: qW, kProj: kW, vProj: vW, oProj: oW,
                    qBias: qBias, kBias: kBias, vBias: vBias, oBias: oBias,
                    qNorm: qNorm, kNorm: kNorm, gateProj: gateW,
                    isSliding: (lKind == "sliding_attention") || (sw > 0),
                    isBidirectional: lKind == "encoder_attention" || lKind == "bidirectional_attention",
                    slidingWindow: sw, ropeKind: lKind
                ))
            }

            let w1Name = resolveName(source, [
                lp + "feed_forward.w1.weight", lp + "mlp.gate_proj.weight",
                lp + "mlp.fc1.weight", lp + "mlp.Wi.weight", lp + "intermediate.dense.weight",
            ]) ?? (lp + "feed_forward.w1.weight")
            let w3Name = resolveName(source, [lp + "feed_forward.w3.weight", lp + "mlp.up_proj.weight"])
            let w2Name = resolveName(source, [
                lp + "feed_forward.w2.weight", lp + "mlp.down_proj.weight",
                lp + "mlp.fc2.weight", lp + "mlp.Wo.weight", lp + "output.dense.weight",
            ]) ?? (lp + "feed_forward.w2.weight")

            let (w13, w1Sep, w3Sep) = try loadFFNUpOrGate(
                w1Name: w1Name, w3Name: w3Name, nEach: ffn, k: hidden,
                source: source, hal: hal, buffers: &buffers
            )
            let ffnUpBias = try loadOptionalVector(
                candidates: [lp + "mlp.fc1.bias", lp + "mlp.Wi.bias", lp + "intermediate.dense.bias", lp + "feed_forward.w1.bias"],
                count: ffn, source: source, hal: hal, buffers: &buffers
            )
            let w2 = try loadLinear(name: w2Name, n: hidden, k: ffn, source: source, hal: hal, buffers: &buffers)
            let ffnDownBias = try loadOptionalVector(
                candidates: [lp + "mlp.fc2.bias", lp + "mlp.Wo.bias", lp + "output.dense.bias", lp + "feed_forward.w2.bias"],
                count: hidden, source: source, hal: hal, buffers: &buffers
            )
            anyRotated = anyRotated || (w13?.needsRotation ?? false)
                || (w1Sep?.needsRotation ?? false) || (w3Sep?.needsRotation ?? false) || w2.needsRotation

            loadedLayers.append(QWDecisionLoadedLayer(
                preBlockNorm: preBlock, postBlockNorm: postBlock, op: op,
                preFFNNorm: preFFN, postFFNNorm: postFFN,
                w13: w13, w1: w1Sep, w3: w3Sep, ffnUpBias: ffnUpBias,
                w2: w2, ffnDownBias: ffnDownBias
            ))
        }

        let headKind = descriptor.decisionConfig?.head
            ?? .vocabVerbalizer(vocabSize: vocab, tieEmbeddings: descriptor.tieWordEmbeddings, sparseGatherThreshold: 256)
        let headWeights = try loadHeadWeights(
            head: headKind, activationKind: descriptor.activationKind,
            prefix: prefix, hidden: hidden, source: source, hal: hal,
            buffers: &buffers, embedTokens: embedTokens
        )

        return QWDecisionPackWeights(
            embedTokens: embedTokens, posEmbedBF16: posEmbedBF16, embedNorm: embedNorm,
            finalNorm: finalNorm, layers: loadedLayers, head: headWeights,
            usesRotatedBasis: anyRotated, retainedBuffers: buffers
        )
    }

    static func resolvePrefix(_ source: QWDecisionWeightSource) -> String {
        for cand in ["model.language_model.", "language_model.model.", "model.", "encoder.", ""] {
            if source.has(cand + "embed_tokens.weight") || source.has(cand + "tok_embeddings.weight")
                || source.has(cand + "embeddings.word_embeddings.weight") || source.has(cand + "embeddings.tok_embeddings.weight") {
                return cand
            }
        }
        return "model."
    }

    private static func resolveLayerPrefix(source: QWDecisionWeightSource, prefix: String, layer: Int) -> String {
        let candidates = [
            "\(prefix)layers.\(layer).",
            "\(prefix)encoder.layers.\(layer).",
            "\(prefix)encoder.layer.\(layer).",
            "layers.\(layer).",
            "encoder.layer.\(layer).",
        ]
        for lp in candidates {
            if source.has(lp + "operator_norm.weight") || source.has(lp + "input_layernorm.weight")
                || source.has(lp + "attn_norm.weight") || source.has(lp + "self_attn.q_proj.weight")
                || source.has(lp + "attention.self.query.weight") || source.has(lp + "conv.conv.weight") {
                return lp
            }
        }
        return "\(prefix)layers.\(layer)."
    }

    static func resolveName(_ source: QWDecisionWeightSource, _ candidates: [String]) -> String? {
        candidates.first(where: { source.has($0) })
    }

    private static func loadOptionalVector(
        candidates: [String], count: Int, source: QWDecisionWeightSource, hal: QWHal, buffers: inout [any MTLBuffer]
    ) throws -> QWTensorValue? {
        guard let name = resolveName(source, candidates) else { return nil }
        return try loadF16Vector(name: name, count: count, source: source, hal: hal, buffers: &buffers)
    }

    private static func loadOptionalNorm(
        weightCandidates: [String], biasCandidates: [String], count: Int,
        source: QWDecisionWeightSource, hal: QWHal, buffers: inout [any MTLBuffer]
    ) throws -> QWDecisionNormTensor? {
        guard let wName = resolveName(source, weightCandidates) else { return nil }
        let w = try loadF16Vector(name: wName, count: count, source: source, hal: hal, buffers: &buffers)
        let b = try loadOptionalVector(candidates: biasCandidates, count: count, source: source, hal: hal, buffers: &buffers)
        return QWDecisionNormTensor(weight: w, bias: b)
    }

    private static func loadFFNUpOrGate(
        w1Name: String,
        w3Name: String?,
        nEach: Int,
        k: Int,
        source: QWDecisionWeightSource,
        hal: QWHal,
        buffers: inout [any MTLBuffer]
    ) throws -> (w13: QWLinearWeight?, w1: QWLinearWeight?, w3: QWLinearWeight?) {
        guard let w3Name else {
            let w1 = try loadLinear(name: w1Name, n: nEach, k: k, source: source, hal: hal, buffers: &buffers)
            return (nil, w1, nil)
        }
        if let fused = try tryLoadFusedQWPOrST(w1Name: w1Name, w3Name: w3Name, nEach: nEach, k: k, source: source, hal: hal, buffers: &buffers) {
            return (fused, nil, nil)
        }
        // Fallback: concatenate w1 and w3 into a BF16 w13 bank so mixed-container SwiGLU runs on NAX.
        let b1 = try loadBF16Tensor(name: w1Name, rows: nEach, cols: k, source: source, hal: hal, buffers: &buffers)
        let b3 = try loadBF16Tensor(name: w3Name, rows: nEach, cols: k, source: source, hal: hal, buffers: &buffers)
        let halfBytes = nEach * k * 2
        let totalBytes = halfBytes * 2
        let buf = try allocSharedBuffer(bytes: totalBytes, hal: hal, buffers: &buffers)
        let dst = buf.contents()
        withExtendedLifetime(buf) {
            withExtendedLifetime(b1.region.buffer) {
                dst.copyMemory(from: b1.region.pointer.advanced(by: b1.offset), byteCount: halfBytes)
            }
            withExtendedLifetime(b3.region.buffer) {
                dst.advanced(by: halfBytes).copyMemory(from: b3.region.pointer.advanced(by: b3.offset), byteCount: halfBytes)
            }
        }
        let slice = QWFPBufferSlice(buffer: buf, offset: 0, bytes: totalBytes)
        return (.bf16(QWGemmFP.BF16Bank(weights: slice, n: 2 * nEach, k: k)), nil, nil)
    }

    private static func tryLoadFusedQWPOrST(
        w1Name: String, w3Name: String, nEach: Int, k: Int,
        source: QWDecisionWeightSource, hal: QWHal, buffers: inout [any MTLBuffer]
    ) throws -> QWLinearWeight? {
        switch source {
        case .safetensors(let st):
            guard let t1 = st.tensor(w1Name), let t3 = st.tensor(w3Name),
                  t1.dtype == .bf16, t3.dtype == .bf16,
                  t1.byteLength == nEach * k * 2, t3.byteLength == nEach * k * 2 else { return nil }
            let totalBytes = (2 * nEach) * k * 2
            let buf = try allocSharedBuffer(bytes: totalBytes, hal: hal, buffers: &buffers)
            let dst = buf.contents()
            try withExtendedLifetime(buf) {
                try st.withBytes(of: w1Name) { (r1: UnsafeRawBufferPointer) in
                    if let p1 = r1.baseAddress { dst.copyMemory(from: p1, byteCount: t1.byteLength) }
                }
                try st.withBytes(of: w3Name) { (r3: UnsafeRawBufferPointer) in
                    if let p3 = r3.baseAddress {
                        dst.advanced(by: t1.byteLength).copyMemory(from: p3, byteCount: t3.byteLength)
                    }
                }
            }
            let slice = QWFPBufferSlice(buffer: buf, offset: 0, bytes: totalBytes)
            return .bf16(QWGemmFP.BF16Bank(weights: slice, n: 2 * nEach, k: k))
        case .qwp(let pack):
            let e1 = try pack.entry(w1Name)
            let e3 = try pack.entry(w3Name)
            guard let l1 = PlaneLayout.resolve(e1), let o1 = l1.offset(e1), let b1 = l1.byteCount(e1),
                  let l3 = PlaneLayout.resolve(e3), let o3 = l3.offset(e3), let b3 = l3.byteCount(e3) else { return nil }
            if e1.rawOff != nil && e3.rawOff != nil {
                guard e1.kind != "f32", e3.kind != "f32", e1.kind != "f16", e3.kind != "f16",
                      b1 == nEach * k * 2, b3 == nEach * k * 2 else { return nil }
                let totalBytes = b1 + b3
                let buf = try allocSharedBuffer(bytes: totalBytes, hal: hal, buffers: &buffers)
                let dst = buf.contents()
                try withExtendedLifetime(buf) {
                    try pack.withRawPointer(at: o1, byteCount: b1) { (p1: UnsafeRawPointer) in dst.copyMemory(from: p1, byteCount: b1) }
                    try pack.withRawPointer(at: o3, byteCount: b3) { (p3: UnsafeRawPointer) in dst.advanced(by: b1).copyMemory(from: p3, byteCount: b3) }
                }
                return .bf16(QWGemmFP.BF16Bank(weights: QWFPBufferSlice(buffer: buf, offset: 0, bytes: totalBytes), n: 2 * nEach, k: k))
            }
            guard let k1 = containerKind(from: e1), let k3 = containerKind(from: e3), k1 == k3, b1 == b3 else { return nil }
            let totalBytes = b1 + b3
            let buf = try allocSharedBuffer(bytes: totalBytes, hal: hal, buffers: &buffers)
            let dst = buf.contents()
            try withExtendedLifetime(buf) {
                try pack.withRawPointer(at: o1, byteCount: b1) { (p1: UnsafeRawPointer) in dst.copyMemory(from: p1, byteCount: b1) }
                try pack.withRawPointer(at: o3, byteCount: b3) { (p3: UnsafeRawPointer) in dst.advanced(by: b1).copyMemory(from: p3, byteCount: b3) }
            }
            let tv = QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length), offset: 0, shape: QWTensorShape([totalBytes / 2], .f16))
            guard let view = QWWeightView(blocks: tv, n: 2 * nEach, k: k, containerKind: k1) else { return nil }
            return .quant(view)
        }
    }

    static func loadLinear(
        name: String, n: Int, k: Int, source: QWDecisionWeightSource, hal: QWHal, buffers: inout [any MTLBuffer]
    ) throws -> QWLinearWeight {
        if case .qwp(let pack) = source {
            let e = try pack.entry(name)
            if let layout = PlaneLayout.resolve(e), layout == .hq4,
               let kind = containerKind(from: e),
               let off = layout.offset(e), let bytes = layout.byteCount(e) {
                let buf = try allocSharedBuffer(bytes: bytes, hal: hal, buffers: &buffers)
                let dst = buf.contents()
                try withExtendedLifetime(buf) {
                    try pack.withRawPointer(at: off, byteCount: bytes) { (ptr: UnsafeRawPointer) in
                        if bytes > 0 { dst.copyMemory(from: ptr, byteCount: bytes) }
                    }
                }
                let tv = QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length), offset: 0, shape: QWTensorShape([bytes / 2], .f16))
                guard let view = QWWeightView(blocks: tv, n: n, k: k, containerKind: kind) else {
                    throw QWDecisionError.backendFailed("invalid QWWeightView [\(n), \(k)] for \(name)")
                }
                return .quant(view)
            }
        }
        let tv = try loadBF16Tensor(name: name, rows: n, cols: k, source: source, hal: hal, buffers: &buffers)
        return .bf16(QWGemmFP.BF16Bank(weights: tv, n: n, k: k))
    }

    static func loadBF16Tensor(
        name: String, rows: Int, cols: Int, source: QWDecisionWeightSource, hal: QWHal, buffers: inout [any MTLBuffer]
    ) throws -> QWTensorValue {
        let count = rows * cols
        let byteLen = count * 2
        let buf = try allocSharedBuffer(bytes: byteLen, hal: hal, buffers: &buffers)
        let dst = buf.contents()
        switch source {
        case .safetensors(let st):
            guard let t = st.tensor(name) else { throw QWDecisionError.backendFailed("missing SafeTensors tensor: \(name)") }
            try withExtendedLifetime(buf) {
                try st.withBytes(of: name) { (raw: UnsafeRawBufferPointer) in
                    guard let src = raw.baseAddress else { return }
                    convertToBF16(src: src, srcDtype: t.dtype.rawValue, count: count, expectedSrcBytes: t.byteLength, dst: dst)
                }
            }
        case .qwp(let pack):
            let e = try pack.entry(name)
            guard let layout = PlaneLayout.resolve(e), let off = layout.offset(e), let bytes = layout.byteCount(e) else {
                throw QWDecisionError.backendFailed("invalid QWP1 layout for \(name)")
            }
            if layout == .raw {
                try withExtendedLifetime(buf) {
                    try pack.withRawPointer(at: off, byteCount: bytes) { (src: UnsafeRawPointer) in
                        convertToBF16(src: src, srcDtype: e.kind ?? "bf16", count: count, expectedSrcBytes: bytes, dst: dst)
                    }
                }
            } else {
                let f32 = try pack.dequantQ4g(name)
                let bf16Dst = dst.assumingMemoryBound(to: UInt16.self)
                for i in 0..<min(count, f32.count) { bf16Dst[i] = f32ToBF16(f32[i]) }
            }
        }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length), offset: 0, shape: QWTensorShape([rows, cols], .bf16))
    }

    static func loadF16Vector(
        name: String, count: Int, source: QWDecisionWeightSource, hal: QWHal, buffers: inout [any MTLBuffer]
    ) throws -> QWTensorValue {
        let buf = try allocSharedBuffer(bytes: count * 2, hal: hal, buffers: &buffers)
        let dst = buf.contents().assumingMemoryBound(to: Float16.self)
        switch source {
        case .safetensors(let st):
            guard let t = st.tensor(name) else { throw QWDecisionError.backendFailed("missing SafeTensors tensor: \(name)") }
            try withExtendedLifetime(buf) {
                try st.withBytes(of: name) { (raw: UnsafeRawBufferPointer) in
                    guard let src = raw.baseAddress else { return }
                    convertToF16(src: src, srcDtype: t.dtype.rawValue, count: count, expectedSrcBytes: t.byteLength, dst: dst)
                }
            }
        case .qwp(let pack):
            let e = try pack.entry(name)
            guard let layout = PlaneLayout.resolve(e), layout == .raw,
                  let off = layout.offset(e) else {
                throw QWDecisionError.backendFailed("expected raw norm/conv tensor for \(name)")
            }
            let byteCount = count * e.rawElemBytes
            try withExtendedLifetime(buf) {
                try pack.withRawPointer(at: off, byteCount: byteCount) { (src: UnsafeRawPointer) in
                    convertToF16(src: src, srcDtype: e.kind ?? "bf16", count: count, expectedSrcBytes: byteCount, dst: dst)
                }
            }
        }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length), offset: 0, shape: QWTensorShape([count], .f16))
    }

    private static func containerKind(from e: QWQTensorEntry) -> QWWeightView.ContainerKind? {
        guard PlaneLayout.resolve(e) == .hq4 else { return nil }
        let rotated = e.hq?.vq != "aq"
        if e.isSHVQ { return rotated ? .shvq(bits: e.hqBits, group: e.hqGroup) : nil }
        if e.isIncumbentHQ4 { return .incumbentHQ4G32 }
        return rotated ? .rotatedHQ(bits: e.hqBits, group: e.hqGroup) : .nativeAffine(bits: e.hqBits, group: e.hqGroup)
    }

    static func allocSharedBuffer(bytes: Int, hal: QWHal, buffers: inout [any MTLBuffer]) throws -> any MTLBuffer {
        let len = max(64, qwAlignUp(bytes, 64))
        guard let buf = hal.device.makeBuffer(length: len, options: .storageModeShared) else {
            throw QWDecisionError.backendFailed("failed to allocate Metal buffer of \(len) bytes")
        }
        if len > bytes { memset(buf.contents().advanced(by: bytes), 0, len - bytes) }
        buffers.append(buf)
        return buf
    }

    static func convertToBF16(
        src: UnsafeRawPointer, srcDtype: String, count: Int, expectedSrcBytes: Int, dst: UnsafeMutableRawPointer
    ) {
        let low = srcDtype.lowercased()
        let out = dst.assumingMemoryBound(to: UInt16.self)
        if low == "f32" && expectedSrcBytes >= count * 4 {
            let fp = src.assumingMemoryBound(to: Float.self)
            for i in 0..<count { out[i] = f32ToBF16(fp[i]) }
        } else if low == "f16" && expectedSrcBytes >= count * 2 {
            let hp = src.assumingMemoryBound(to: Float16.self)
            for i in 0..<count { out[i] = f32ToBF16(Float(hp[i])) }
        } else if expectedSrcBytes >= count * 2 {
            dst.copyMemory(from: src, byteCount: count * 2)
        }
    }

    static func convertToF16(
        src: UnsafeRawPointer, srcDtype: String, count: Int, expectedSrcBytes: Int, dst: UnsafeMutablePointer<Float16>
    ) {
        let low = srcDtype.lowercased()
        if low == "f32" && expectedSrcBytes >= count * 4 {
            let fp = src.assumingMemoryBound(to: Float.self)
            for i in 0..<count { dst[i] = Float16(fp[i]) }
        } else if low == "f16" && expectedSrcBytes >= count * 2 {
            UnsafeMutableRawPointer(dst).copyMemory(from: src, byteCount: count * 2)
        } else if expectedSrcBytes >= count * 2 {
            let bp = src.assumingMemoryBound(to: UInt16.self)
            for i in 0..<count { dst[i] = Float16(Float(bitPattern: UInt32(bp[i]) << 16)) }
        }
    }

    @inline(__always)
    private static func f32ToBF16(_ v: Float) -> UInt16 {
        var bits = v.bitPattern
        if (bits & 0x7F80_0000) == 0x7F80_0000 { return UInt16(truncatingIfNeeded: bits >> 16) }
        bits = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: bits >> 16)
    }

    static func isVisionOrMediaTensorName(_ name: String) -> Bool {
        let low = name.lowercased()
        return low.contains("vision") || low.contains("visual")
            || low.contains("patch_embed") || low.contains("multi_modal_projector")
            || low.hasPrefix("vis_") || low.contains("projector.")
            || low.contains("mm_projector") || low.contains("audio")
    }

    static func estimateResidentBytes(root: QWModelRoot, descriptor: QWModelDescriptor?, loadVision: Bool) -> Int? {
        let hasVision = (descriptor?.visDepth ?? 0) > 0 || descriptor?.visionTowerConfig != nil
        let isTied = descriptor?.tieWordEmbeddings ?? false
        var total = 0
        var visBytes = 0
        switch root.kind {
        case .qwp1:
            guard let descriptor, let pack = try? QWQuant.Pack(url: root.weights, descriptor: descriptor) else { return nil }
            for (name, entry) in pack.tensors {
                if isTied && (name == "lm_head.weight" || name == "model.lm_head.weight") { continue }
                let isVis = isVisionOrMediaTensorName(name)
                if isVis && !loadVision { continue }
                guard let layout = PlaneLayout.resolve(entry), let rawBytes = layout.byteCount(entry) else { continue }
                let loadedBytes = (entry.rawOff != nil && entry.kind == "f32") ? (rawBytes / 2) : rawBytes
                let aligned = max(64, qwAlignUp(loadedBytes, 64))
                total += aligned
                if isVis { visBytes += aligned }
            }
        case .safetensorsSingle, .safetensorsSharded:
            guard let st = try? QWSafeTensorsContainer(url: root.weights, onlyTensors: nil) else { return nil }
            for (name, tensor) in st.tensors {
                if isTied && (name == "lm_head.weight" || name == "model.lm_head.weight") { continue }
                let isVis = isVisionOrMediaTensorName(name)
                if isVis && !loadVision { continue }
                let loadedBytes = (tensor.dtype == .f32) ? (tensor.byteLength / 2) : tensor.byteLength
                let aligned = max(64, qwAlignUp(loadedBytes, 64))
                total += aligned
                if isVis { visBytes += aligned }
            }
        default:
            return nil
        }
        guard total > 0 else { return nil }
        let runtimeOverhead = (96 << 20) + ((loadVision && hasVision) ? (visBytes + (48 << 20)) : 0)
        return total + runtimeOverhead
    }
}

extension QWDecisionPackEngine {
    public static func estimateResidentBytes(at packPath: String, loadVision: Bool) -> Int? {
        let root = QWModelRoot(url: URL(fileURLWithPath: packPath))
        let desc: QWModelDescriptor? = {
            guard let cfgURL = root.configURL, let data = try? Data(contentsOf: cfgURL) else { return nil }
            return try? QWModelDescriptor.decode(configJSON: data)
        }()
        return QWDecisionWeightLoader.estimateResidentBytes(root: root, descriptor: desc, loadVision: loadVision)
    }
}
