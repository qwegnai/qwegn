// QWDecisionPackEngine+Heads.swift — Readout head & dense embedding execution on generalized Metal 4 backbone.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWDecision
import QWGEMM
import QWHAL
import QWHead
import QWNorm

extension QWDecisionPackEngine {
    private struct MarkerItemEval {
        let itemIndex: Int
        let chain: QWDecisionMarkerChainWeights
        let count: Int
        let outTensor: QWTensorValue
    }

    private struct PooledItemEval {
        let itemIndex: Int
        let outTensor: QWTensorValue
        let outDim: Int
    }

    private struct JointItemEval {
        let itemIndex: Int
        let projDim: Int
        let queryOut: QWTensorValue
        let optOut: QWTensorValue
        let optCount: Int
    }

    func executeNonVocabHeads(
        enc: any MTLComputeCommandEncoder,
        cb: any MTLCommandBuffer,
        items: [PreparedDecisionSeq],
        seqOffsets: [Int],
        final16: QWTensorValue,
        totalSeqLen: Int
    ) throws -> [(slotIdx: Int, output: QWDecisionSlotOutput)] {
        let hidden = descriptor.hidden
        switch weights.head {
        case .vocabVerbalizer:
            return []

        case .markerScorer(let headW):
            var evals: [MarkerItemEval] = []
            for (idx, item) in items.enumerated() {
                let positions = item.markerPositions.isEmpty
                    ? [max(0, item.tokens.count - 1)]
                    : item.markerPositions
                let chain = headW.chains[item.chain] ?? headW.chains["default"]
                guard let chain else {
                    throw QWDecisionError.backendFailed("no marker chain weights for '\(item.chain)'")
                }
                let globalPos = positions.map { Int32(seqOffsets[idx] + max(0, min(item.tokens.count - 1, $0))) }
                let posT = try scratchArena.tensorI32(key: "mkPos_\(idx)", values: globalPos, shape: [globalPos.count])
                let gath16 = try scratchArena.tensor(key: "mkGath_\(idx)", shape: [globalPos.count, hidden], dtype: .f16)
                let norm16 = headW.headNormW != nil
                    ? (try scratchArena.tensor(key: "mkNorm_\(idx)", shape: [globalPos.count, hidden], dtype: .f16))
                    : gath16
                let mid32 = try scratchArena.tensor(key: "mkMid_\(idx)", shape: [globalPos.count, chain.midDim], dtype: .f32)
                let out32 = try scratchArena.tensor(key: "mkOut_\(idx)", shape: [globalPos.count, chain.outDim], dtype: .f32)

                try headOp.encodeMarkerGather(
                    enc, hiddenState: final16, positions: posT, typeEmb: headW.typeEmb, out: gath16,
                    seqLen: totalSeqLen, count: globalPos.count, hidden: hidden
                )
                enc.memoryBarrier(scope: .buffers)
                if let hw = headW.headNormW {
                    try encodeNormTensor(
                        enc, x16: gath16, norm: QWDecisionNormTensor(weight: hw, bias: headW.headNormB),
                        out16: norm16, rows: globalPos.count, hidden: hidden
                    )
                    enc.memoryBarrier(scope: .buffers)
                }
                if let w2 = chain.w2 {
                    try headOp.encodeDecisionLinear(
                        enc, x: norm16, weight: chain.w1, bias: chain.b1, out: mid32,
                        rows: globalPos.count, inFeatures: hidden, outFeatures: chain.midDim, activation: chain.activation
                    )
                    enc.memoryBarrier(scope: .buffers)
                    try headOp.encodeDecisionLinear(
                        enc, x: mid32, weight: w2, bias: chain.b2, out: out32,
                        rows: globalPos.count, inFeatures: chain.midDim, outFeatures: chain.outDim, activation: .none
                    )
                } else {
                    try headOp.encodeDecisionLinear(
                        enc, x: norm16, weight: chain.w1, bias: chain.b1, out: out32,
                        rows: globalPos.count, inFeatures: hidden, outFeatures: chain.outDim, activation: .none
                    )
                }
                evals.append(MarkerItemEval(itemIndex: idx, chain: chain, count: globalPos.count, outTensor: out32))
            }
            try finishAndCheck(cb: cb, enc: enc, label: "marker scorer")
            var results: [(slotIdx: Int, output: QWDecisionSlotOutput)] = []
            for ev in evals {
                let rawPtr = ev.outTensor.region.pointer.advanced(by: ev.outTensor.offset).assumingMemoryBound(to: Float.self)
                let scores = Array(UnsafeBufferPointer(start: rawPtr, count: ev.count * ev.chain.outDim))
                for t in items[ev.itemIndex].slotTargets {
                    results.append((t.slotIdx, QWDecisionSlotOutput(markerLogits: scores)))
                }
            }
            return results

        case .pooledClassifier(let headW):
            var evals: [PooledItemEval] = []
            for (idx, item) in items.enumerated() {
                let sLen = item.tokens.count
                let off = seqOffsets[idx]
                let itemH = final16.slice(byteOffset: off * hidden * 2, count: sLen * hidden, dtype: .f16)
                let pooled16 = try scratchArena.tensor(key: "pool16_\(idx)", shape: [1, hidden], dtype: .f16)
                let out32 = try scratchArena.tensor(key: "clsOut32_\(idx)", shape: [1, headW.numLabels], dtype: .f32)
                try headOp.encodePool(
                    enc, hiddenState: itemH, out: pooled16,
                    seqLen: sLen, validLen: sLen, hidden: hidden, pooling: headW.pooling
                )
                enc.memoryBarrier(scope: .buffers)
                if let denseW = headW.poolDenseW {
                    let midDim = denseW.shape.dims[0]
                    let mid32 = try scratchArena.tensor(key: "clsMid32_\(idx)", shape: [1, midDim], dtype: .f32)
                    try headOp.encodeDecisionLinear(
                        enc, x: pooled16, weight: denseW, bias: headW.poolDenseB, out: mid32,
                        rows: 1, inFeatures: hidden, outFeatures: midDim, activation: headW.poolDenseAct
                    )
                    enc.memoryBarrier(scope: .buffers)
                    try headOp.encodeDecisionLinear(
                        enc, x: mid32, weight: headW.classifierW, bias: headW.classifierB, out: out32,
                        rows: 1, inFeatures: midDim, outFeatures: headW.numLabels, activation: .none
                    )
                } else {
                    try headOp.encodeDecisionLinear(
                        enc, x: pooled16, weight: headW.classifierW, bias: headW.classifierB, out: out32,
                        rows: 1, inFeatures: hidden, outFeatures: headW.numLabels, activation: .none
                    )
                }
                evals.append(PooledItemEval(itemIndex: idx, outTensor: out32, outDim: headW.numLabels))
            }
            try finishAndCheck(cb: cb, enc: enc, label: "pooled classifier")
            var results: [(slotIdx: Int, output: QWDecisionSlotOutput)] = []
            for ev in evals {
                let rawPtr = ev.outTensor.region.pointer.advanced(by: ev.outTensor.offset).assumingMemoryBound(to: Float.self)
                let logits = Array(UnsafeBufferPointer(start: rawPtr, count: ev.outDim))
                for t in items[ev.itemIndex].slotTargets {
                    results.append((t.slotIdx, QWDecisionSlotOutput(pooledLogits: logits)))
                }
            }
            return results

        case .jointOptionProjector(let headW):
            var evals: [JointItemEval] = []
            for (idx, item) in items.enumerated() {
                let sLen = item.tokens.count
                let off = seqOffsets[idx]
                let qPos = Int32(off + max(0, min(sLen - 1, item.stateEndPos > 0 ? item.stateEndPos - 1 : 0)))
                let optPos = (item.markerPositions.isEmpty ? [max(0, sLen - 1)] : item.markerPositions)
                    .map { Int32(off + max(0, min(sLen - 1, $0))) }
                let qPosT = try scratchArena.tensorI32(key: "jQPos_\(idx)", values: [qPos], shape: [1])
                let oPosT = try scratchArena.tensorI32(key: "jOPos_\(idx)", values: optPos, shape: [optPos.count])
                let qGath = try scratchArena.tensor(key: "jQGath_\(idx)", shape: [1, hidden], dtype: .f16)
                let oGath = try scratchArena.tensor(key: "jOGath_\(idx)", shape: [optPos.count, hidden], dtype: .f16)
                let qOut = try scratchArena.tensor(key: "jQOut_\(idx)", shape: [1, headW.projDim], dtype: .f32)
                let oOut = try scratchArena.tensor(key: "jOOut_\(idx)", shape: [optPos.count, headW.projDim], dtype: .f32)

                try headOp.encodeMarkerGather(
                    enc, hiddenState: final16, positions: qPosT, typeEmb: nil, out: qGath,
                    seqLen: totalSeqLen, count: 1, hidden: hidden
                )
                try headOp.encodeMarkerGather(
                    enc, hiddenState: final16, positions: oPosT, typeEmb: nil, out: oGath,
                    seqLen: totalSeqLen, count: optPos.count, hidden: hidden
                )
                enc.memoryBarrier(scope: .buffers)
                try headOp.encodeDecisionLinear(
                    enc, x: qGath, weight: headW.queryW, bias: headW.queryB, out: qOut,
                    rows: 1, inFeatures: hidden, outFeatures: headW.projDim, activation: .none
                )
                try headOp.encodeDecisionLinear(
                    enc, x: oGath, weight: headW.optionW, bias: headW.optionB, out: oOut,
                    rows: optPos.count, inFeatures: hidden, outFeatures: headW.projDim, activation: .none
                )
                evals.append(JointItemEval(
                    itemIndex: idx, projDim: headW.projDim, queryOut: qOut, optOut: oOut, optCount: optPos.count
                ))
            }
            try finishAndCheck(cb: cb, enc: enc, label: "joint option projector")
            var results: [(slotIdx: Int, output: QWDecisionSlotOutput)] = []
            for ev in evals {
                let qPtr = ev.queryOut.region.pointer.advanced(by: ev.queryOut.offset).assumingMemoryBound(to: Float.self)
                let oPtr = ev.optOut.region.pointer.advanced(by: ev.optOut.offset).assumingMemoryBound(to: Float.self)
                let scale = 1.0 / sqrtf(Float(max(1, ev.projDim)))
                var scores = [Float](repeating: 0, count: ev.optCount)
                for m in 0..<ev.optCount {
                    var acc: Float = 0
                    let base = m * ev.projDim
                    for d in 0..<ev.projDim { acc += qPtr[d] * oPtr[base + d] }
                    scores[m] = acc * scale
                }
                for t in items[ev.itemIndex].slotTargets {
                    results.append((t.slotIdx, QWDecisionSlotOutput(markerLogits: scores)))
                }
            }
            return results
        }
    }

    func executeEmbedChunkOnGPU(
        _ items: [PreparedDecisionSeq],
        pooling: QWDecisionModelConfig.PoolingKind,
        normalize: Bool,
        truncateDimensions: Int?
    ) throws -> [[Float]] {
        guard !items.isEmpty else { return [] }
        let hidden = descriptor.hidden
        var seqOffsets: [Int] = []
        seqOffsets.reserveCapacity(items.count)
        var totalSeqLen = 0
        var maxItemSeqLen = 1
        for item in items {
            seqOffsets.append(totalSeqLen)
            totalSeqLen += item.tokens.count
            maxItemSeqLen = max(maxItemSeqLen, item.tokens.count)
        }
        guard totalSeqLen > 0 else { return items.map { _ in [] } }

        guard let (cb, enc) = hal.begin() else {
            throw QWDecisionError.backendFailed("failed to open Metal command buffer for embeddings")
        }
        let final16 = try encodeBackbone(
            enc: enc, items: items, seqOffsets: seqOffsets,
            totalSeqLen: totalSeqLen, maxItemSeqLen: maxItemSeqLen
        )
        var pooledTensors: [QWTensorValue] = []
        pooledTensors.reserveCapacity(items.count)
        for (idx, item) in items.enumerated() {
            let sLen = item.tokens.count
            let itemH = final16.slice(byteOffset: seqOffsets[idx] * hidden * 2, count: sLen * hidden, dtype: .f16)
            let pooled16 = try scratchArena.tensor(key: "embPool16_\(idx)", shape: [1, hidden], dtype: .f16)
            try headOp.encodePool(
                enc, hiddenState: itemH, out: pooled16,
                seqLen: sLen, validLen: sLen, hidden: hidden, pooling: pooling
            )
            pooledTensors.append(pooled16)
        }
        try finishAndCheck(cb: cb, enc: enc, label: "embedding pool")

        return pooledTensors.map { pt in
            let ptr = pt.region.pointer.advanced(by: pt.offset).assumingMemoryBound(to: Float16.self)
            var vec = [Float](repeating: 0, count: hidden)
            for d in 0..<hidden { vec[d] = Float(ptr[d]) }
            return QWEmbeddingMath.postProcess(vec, normalize: normalize, truncateDimensions: truncateDimensions)
        }
    }

    func finishAndCheck(cb: any MTLCommandBuffer, enc: any MTLComputeCommandEncoder, label: String) throws {
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        guard cb.status == .completed else {
            throw QWDecisionError.backendFailed("decision \(label) CB failed (status=\(cb.status.rawValue))")
        }
    }
}
