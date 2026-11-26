// QWDecisionPackEngine+HeadWeights.swift — Head-specific tensor loaders for all 4 QWDecisionModelConfig.HeadKind variants.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWDecision
import QWHAL
import QWHead
import QWLoad
import QWQuant

struct QWDecisionMarkerChainWeights: @unchecked Sendable {
    let w1: QWTensorValue
    let b1: QWTensorValue?
    let w2: QWTensorValue?
    let b2: QWTensorValue?
    let midDim: Int
    let outDim: Int
    let activation: QWDecisionHeadActivation
}

struct QWDecisionMarkerScorerHeadWeights: @unchecked Sendable {
    let typeEmb: QWTensorValue?
    let headNormW: QWTensorValue?
    let headNormB: QWTensorValue?
    let chains: [String: QWDecisionMarkerChainWeights]
}

struct QWDecisionPooledHeadWeights: @unchecked Sendable {
    let pooling: QWDecisionModelConfig.PoolingKind
    let numLabels: Int
    let poolDenseW: QWTensorValue?
    let poolDenseB: QWTensorValue?
    let poolDenseAct: QWDecisionHeadActivation
    let classifierW: QWTensorValue
    let classifierB: QWTensorValue?
}

struct QWDecisionJointHeadWeights: @unchecked Sendable {
    let projDim: Int
    let queryW: QWTensorValue
    let queryB: QWTensorValue?
    let optionW: QWTensorValue
    let optionB: QWTensorValue?
}

enum QWDecisionPackHeadWeights: @unchecked Sendable {
    case vocabVerbalizer(lmHeadBF16: QWTensorValue?)
    case markerScorer(QWDecisionMarkerScorerHeadWeights)
    case pooledClassifier(QWDecisionPooledHeadWeights)
    case jointOptionProjector(QWDecisionJointHeadWeights)
}

extension QWDecisionWeightLoader {
    static func loadHeadWeights(
        head: QWDecisionModelConfig.HeadKind,
        activationKind: QWModelDescriptor.ActivationKind,
        prefix: String,
        hidden: Int,
        source: QWDecisionWeightSource,
        hal: QWHal,
        buffers: inout [any MTLBuffer],
        embedTokens: QWTensorValue
    ) throws -> QWDecisionPackHeadWeights {
        let headAct: QWDecisionHeadActivation = switch activationKind {
        case .silu, .situ: .silu
        case .gelu, .geluTanh: .gelu
        }
        switch head {
        case .vocabVerbalizer(_, let tie, _):
            let lmHeadBF16: QWTensorValue?
            if tie {
                lmHeadBF16 = embedTokens
            } else if let lmName = resolveName(source, ["lm_head.weight", prefix + "lm_head.weight"]) {
                lmHeadBF16 = try loadBF16Tensor(
                    name: lmName, rows: embedTokens.shape.dims[0], cols: hidden,
                    source: source, hal: hal, buffers: &buffers
                )
            } else {
                lmHeadBF16 = embedTokens
            }
            return .vocabVerbalizer(lmHeadBF16: lmHeadBF16)

        case .markerScorer(let headH, _, let chainNames):
            let h = headH > 0 ? headH : hidden
            let typeEmb = try resolveName(source, [
                "type_emb.weight", "type_embedding.weight", prefix + "type_emb.weight", "head.type_emb",
            ]).map { try loadF16Vector(name: $0, count: h, source: source, hal: hal, buffers: &buffers) }
            let normW = try resolveName(source, [
                "head_norm.weight", "scorer_norm.weight", "head.norm.weight",
            ]).map { try loadF16Vector(name: $0, count: h, source: source, hal: hal, buffers: &buffers) }
            let normB = try resolveName(source, [
                "head_norm.bias", "scorer_norm.bias", "head.norm.bias",
            ]).map { try loadF16Vector(name: $0, count: h, source: source, hal: hal, buffers: &buffers) }

            var loadedChains: [String: QWDecisionMarkerChainWeights] = [:]
            for chain in Set(["default"] + chainNames) {
                if let cw = try tryLoadMarkerChain(
                    chain: chain, hidden: h, activation: headAct,
                    source: source, hal: hal, buffers: &buffers
                ) {
                    loadedChains[chain] = cw
                }
            }
            guard !loadedChains.isEmpty else {
                throw QWDecisionError.backendFailed("missing markerScorer head weights in decision pack")
            }
            return .markerScorer(QWDecisionMarkerScorerHeadWeights(
                typeEmb: typeEmb, headNormW: normW, headNormB: normB, chains: loadedChains
            ))

        case .pooledClassifier(let pooling, let numLabels, _):
            let densePair = try tryLoadLinearF16(
                weightCandidates: ["head.dense.weight", "pooler.dense.weight", "classifier.dense.weight"],
                biasCandidates: ["head.dense.bias", "pooler.dense.bias", "classifier.dense.bias"],
                expectedCols: hidden,
                source: source, hal: hal, buffers: &buffers
            )
            let clsPair = try tryLoadLinearF16(
                weightCandidates: [
                    "classifier.weight", "classifier.out_proj.weight", "head.out_proj.weight",
                    "score.weight", "score.out_proj.weight", "linear.weight",
                ],
                biasCandidates: [
                    "classifier.bias", "classifier.out_proj.bias", "head.out_proj.bias",
                    "score.bias", "score.out_proj.bias", "linear.bias",
                ],
                expectedCols: densePair?.rows ?? hidden,
                source: source, hal: hal, buffers: &buffers
            )
            guard let clsPair else {
                throw QWDecisionError.backendFailed("missing pooledClassifier head weights in decision pack")
            }
            return .pooledClassifier(QWDecisionPooledHeadWeights(
                pooling: pooling,
                numLabels: max(1, min(numLabels, clsPair.rows)),
                poolDenseW: densePair?.weight,
                poolDenseB: densePair?.bias,
                poolDenseAct: headAct,
                classifierW: clsPair.weight,
                classifierB: clsPair.bias
            ))

        case .jointOptionProjector(let projDim):
            let qPair = try tryLoadLinearF16(
                weightCandidates: ["query_proj.weight", "head.query_proj.weight", "joint_proj.weight"],
                biasCandidates: ["query_proj.bias", "head.query_proj.bias", "joint_proj.bias"],
                expectedCols: hidden,
                source: source, hal: hal, buffers: &buffers
            )
            let oPair = try tryLoadLinearF16(
                weightCandidates: ["option_proj.weight", "head.option_proj.weight", "joint_proj.weight"],
                biasCandidates: ["option_proj.bias", "head.option_proj.bias", "joint_proj.bias"],
                expectedCols: hidden,
                source: source, hal: hal, buffers: &buffers
            )
            guard let q = qPair ?? oPair, let o = oPair ?? qPair else {
                throw QWDecisionError.backendFailed("missing jointOptionProjector weights in decision pack")
            }
            let resolvedDim = projDim > 0 ? min(projDim, q.rows) : q.rows
            return .jointOptionProjector(QWDecisionJointHeadWeights(
                projDim: resolvedDim, queryW: q.weight, queryB: q.bias, optionW: o.weight, optionB: o.bias
            ))
        }
    }

    private static func tryLoadMarkerChain(
        chain: String,
        hidden: Int,
        activation: QWDecisionHeadActivation,
        source: QWDecisionWeightSource,
        hal: QWHal,
        buffers: inout [any MTLBuffer]
    ) throws -> QWDecisionMarkerChainWeights? {
        let prefixes = chain == "default"
            ? ["head.", "scorer.", "classifier.", "score_head.", ""]
            : ["scorers.\(chain).", "heads.\(chain).", "head.\(chain).", "\(chain)."]
        var w1Cands: [String] = []
        var b1Cands: [String] = []
        var w2Cands: [String] = []
        var b2Cands: [String] = []
        for p in prefixes {
            w1Cands.append(contentsOf: [p + "0.weight", p + "fc1.weight", p + "proj.weight", p + "weight"])
            b1Cands.append(contentsOf: [p + "0.bias", p + "fc1.bias", p + "proj.bias", p + "bias"])
            w2Cands.append(contentsOf: [p + "2.weight", p + "3.weight", p + "fc2.weight", p + "out_proj.weight"])
            b2Cands.append(contentsOf: [p + "2.bias", p + "3.bias", p + "fc2.bias", p + "out_proj.bias"])
        }
        guard let first = try tryLoadLinearF16(
            weightCandidates: w1Cands, biasCandidates: b1Cands,
            expectedCols: hidden, source: source, hal: hal, buffers: &buffers
        ) else {
            return nil
        }
        let second = try tryLoadLinearF16(
            weightCandidates: w2Cands, biasCandidates: b2Cands,
            expectedCols: first.rows, source: source, hal: hal, buffers: &buffers
        )
        return QWDecisionMarkerChainWeights(
            w1: first.weight, b1: first.bias,
            w2: second?.weight, b2: second?.bias,
            midDim: first.rows,
            outDim: second?.rows ?? first.rows,
            activation: activation
        )
    }

    private static func tryLoadLinearF16(
        weightCandidates: [String],
        biasCandidates: [String],
        expectedCols: Int,
        source: QWDecisionWeightSource,
        hal: QWHal,
        buffers: inout [any MTLBuffer]
    ) throws -> (weight: QWTensorValue, bias: QWTensorValue?, rows: Int)? {
        guard let wName = resolveName(source, weightCandidates),
              let dims = tensorShape(name: wName, source: source),
              dims.count == 2 else {
            return nil
        }
        let rows = dims[0]
        let cols = dims[1]
        guard rows > 0, cols == expectedCols else { return nil }
        let wTensor = try loadF16Matrix(name: wName, rows: rows, cols: cols, source: source, hal: hal, buffers: &buffers)
        let bTensor = try resolveName(source, biasCandidates).flatMap { bName -> QWTensorValue? in
            guard let bDims = tensorShape(name: bName, source: source),
                  bDims.reduce(1, *) == rows else { return nil }
            return try loadF16Vector(name: bName, count: rows, source: source, hal: hal, buffers: &buffers)
        }
        return (wTensor, bTensor, rows)
    }

    static func tensorShape(name: String, source: QWDecisionWeightSource) -> [Int]? {
        switch source {
        case .safetensors(let st):
            return st.tensor(name)?.shape
        case .qwp(let pack):
            return pack.tensors[name]?.shape
        }
    }

    static func loadF16Matrix(
        name: String,
        rows: Int,
        cols: Int,
        source: QWDecisionWeightSource,
        hal: QWHal,
        buffers: inout [any MTLBuffer]
    ) throws -> QWTensorValue {
        let count = rows * cols
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
            if let off = e.rawOff, let layout = PlaneLayout.resolve(e), let bytes = layout.byteCount(e) {
                try withExtendedLifetime(buf) {
                    try pack.withRawPointer(at: off, byteCount: bytes) { (src: UnsafeRawPointer) in
                        convertToF16(src: src, srcDtype: e.kind ?? "bf16", count: count, expectedSrcBytes: bytes, dst: dst)
                    }
                }
            } else {
                let f32 = try pack.dequantQ4g(name)
                for i in 0..<min(count, f32.count) { dst[i] = Float16(f32[i]) }
            }
        }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length), offset: 0, shape: QWTensorShape([rows, cols], .f16))
    }
}
