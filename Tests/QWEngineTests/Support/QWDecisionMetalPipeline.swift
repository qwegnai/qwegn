// QWDecisionMetalPipeline.swift — Generalized Metal 4 decision backbone & head pipeline.

import Foundation
import Metal
import QWAlloc
import QWAttn
import QWCore
import QWDecision
import QWGEMM
import QWGDN
import QWHAL
import QWHead
import QWNorm
import QWRope

/// Capability-driven Metal 4 execution pipeline for generalized decision backbones and heads.
public final class QWDecisionMetalPipeline: @unchecked Sendable {
    public let hal: QWHal
    private let shortConv: QWShortConv
    private let topoAttn: QWDecisionTopologyAttn
    private let headOp: QWDecisionHeadOperator
    private let builder: QWDecisionTensorBuilder

    public init(hal: QWHal) throws {
        self.hal = hal
        self.shortConv = try QWShortConv(hal: hal)
        self.topoAttn = try QWDecisionTopologyAttn(hal: hal)
        self.headOp = try QWDecisionHeadOperator(hal: hal)
        self.builder = QWDecisionTensorBuilder(hal: hal)
    }

    /// Reads an `.f16` Metal tensor into `[Float]` for readout inspection and parity assertions.
    public func readF16AsFloats(_ tensor: QWTensorValue, count: Int) -> [Float] {
        let ptr = tensor.region.pointer.advanced(by: tensor.offset).assumingMemoryBound(to: Float16.self)
        var out = [Float]()
        out.reserveCapacity(count)
        for i in 0..<count {
            out.append(Float(ptr[i]))
        }
        return out
    }

    /// Evaluates stacked generalized decision layers (`[seqLen, hidden]` FP16) on Metal 4.
    public func runBackbone(
        hiddenIn: QWTensorValue,
        seqLen: Int,
        hidden: Int,
        layers: [QWDecisionLayerWeights],
        finalNorm: (kind: QWDecisionNormKind, weights: QWDecisionNormWeights)? = nil,
        groupIds: [Int32]? = nil,
        stateEndPos: Int = 0,
        mediaPrefixLen: Int = 0
    ) throws -> QWTensorValue {
        guard seqLen > 0, hidden > 0, hidden % 8 == 0 else {
            throw QWDecisionError.invalidQuestion("decision backbone requires seqLen > 0 and hidden divisible by 8")
        }
        guard hiddenIn.shape.dtype == .f16, hiddenIn.shape.elementCount == seqLen * hidden else {
            throw QWDecisionError.invalidQuestion("hiddenIn shape/dtype mismatch")
        }

        let resolvedGroups = groupIds ?? [Int32](repeating: 0, count: seqLen)
        let groupTensor = try builder.makeI32(resolvedGroups, shape: [seqLen])

        let x16 = try builder.makeEmptyTensor(shape: [seqLen, hidden], dtype: .f16)
        let x32 = try builder.makeEmptyTensor(shape: [seqLen, hidden], dtype: .f32)
        let normed16 = try builder.makeEmptyTensor(shape: [seqLen, hidden], dtype: .f16)
        let branch32 = try builder.makeEmptyTensor(shape: [seqLen, hidden], dtype: .f32)
        let branch16 = try builder.makeEmptyTensor(shape: [seqLen, hidden], dtype: .f16)

        let srcPtr = hiddenIn.region.pointer.advanced(by: hiddenIn.offset)
        memcpy(x16.region.pointer.advanced(by: x16.offset), srcPtr, seqLen * hidden * 2)
        let xSrcH = srcPtr.assumingMemoryBound(to: Float16.self)
        let xDstF = x32.region.pointer.advanced(by: x32.offset).assumingMemoryBound(to: Float.self)
        for i in 0..<(seqLen * hidden) {
            xDstF[i] = Float(xSrcH[i])
        }

        var ropeCache: [String: (table: QWTensorValue, rotaryDims: Int)] = [:]
        var maxChannels = hidden
        var maxFFN = hidden
        var maxQElem = hidden
        var maxKVElem = hidden
        var maxGatePad = 8
        for layer in layers {
            maxFFN = max(maxFFN, layer.spec.ffnDim)
            switch layer.spec.block {
            case .shortConvCausal(let ch, _, _, _), .shortConvCentered(let ch, _, _, _):
                maxChannels = max(maxChannels, ch)
            case .attention(_, let qH, let kvH, let hd, let rope, _, _, _):
                maxQElem = max(maxQElem, qH * hd)
                maxKVElem = max(maxKVElem, kvH * hd)
                maxGatePad = max(maxGatePad, qwAlignUp(seqLen * qH, 8))
                if let rope {
                    let key = headDimKey(hd: hd, entry: rope)
                    if ropeCache[key] == nil {
                        let built = try QWRopeTables.makePerKindTable(rows: seqLen, headDim: hd, entry: rope)
                        let t = try builder.makeF32(built.cosSin, shape: [seqLen, built.rotaryDims])
                        ropeCache[key] = (t, built.rotaryDims)
                    }
                }
            }
        }

        let convIn32 = try builder.makeEmptyTensor(shape: [seqLen * maxChannels], dtype: .f32)
        let convIn16 = try builder.makeEmptyTensor(shape: [seqLen * maxChannels], dtype: .f16)
        let convOut16 = try builder.makeEmptyTensor(shape: [seqLen * maxChannels], dtype: .f16)
        let qBuf32 = try builder.makeEmptyTensor(shape: [seqLen * maxQElem], dtype: .f32)
        let kBuf32 = try builder.makeEmptyTensor(shape: [seqLen * maxKVElem], dtype: .f32)
        let vBuf32 = try builder.makeEmptyTensor(shape: [seqLen * maxKVElem], dtype: .f32)
        let qBuf16 = try builder.makeEmptyTensor(shape: [seqLen * maxQElem], dtype: .f16)
        let kBuf16 = try builder.makeEmptyTensor(shape: [seqLen * maxKVElem], dtype: .f16)
        let vBuf16 = try builder.makeEmptyTensor(shape: [seqLen * maxKVElem], dtype: .f16)
        let qRot16 = try builder.makeEmptyTensor(shape: [seqLen * maxQElem], dtype: .f16)
        let kRot16 = try builder.makeEmptyTensor(shape: [seqLen * maxKVElem], dtype: .f16)
        let attnOut16 = try builder.makeEmptyTensor(shape: [seqLen * maxQElem], dtype: .f16)
        let gateBuf32 = try builder.makeEmptyTensor(shape: [maxGatePad], dtype: .f32)
        let gateBuf16 = try builder.makeEmptyTensor(shape: [maxGatePad], dtype: .f16)
        let ffnMid32 = try builder.makeEmptyTensor(shape: [seqLen * maxFFN], dtype: .f32)

        guard let initialPair = hal.begin() else {
            throw QWDecisionError.backendFailed("failed to open Metal command buffer")
        }
        var cb = initialPair.0
        var enc = initialPair.1
        var x32Stale = false

        for layer in layers {
            let spec = layer.spec
            try encodeNorm(
                cb: cb, enc: &enc, x16: x16, out16: normed16,
                kind: spec.preBlockNorm, weights: layer.preBlockNorm, rows: seqLen, hidden: hidden
            )

            switch spec.block {
            case .shortConvCausal(let ch, let kSize, _, let hasInProj),
                 .shortConvCentered(let ch, let kSize, _, let hasInProj):
                guard let scW = layer.shortConv else {
                    throw QWDecisionError.invalidQuestion("missing ShortConv weights")
                }
                let isCentered: Bool = if case .shortConvCentered = spec.block { true } else { false }
                let cIn32 = convIn32.slice(byteOffset: 0, count: seqLen * ch, dtype: .f32)
                let cIn16 = convIn16.slice(byteOffset: 0, count: seqLen * ch, dtype: .f16)
                let cOut16 = convOut16.slice(byteOffset: 0, count: seqLen * ch, dtype: .f16)
                let convInput: QWTensorValue
                if hasInProj, let inW = scW.inProjW {
                    try headOp.encodeDecisionLinear(
                        enc, x: normed16, weight: inW, bias: scW.inProjB,
                        out: cIn32, rows: seqLen, inFeatures: hidden, outFeatures: ch, activation: .none
                    )
                    enc.memoryBarrier(scope: .buffers)
                    try QWGemm.encodeCast(enc, hal: hal, x32: cIn32, y16: cIn16, count: seqLen * ch)
                    enc.memoryBarrier(scope: .buffers)
                    convInput = cIn16
                } else {
                    convInput = normed16
                }
                if isCentered {
                    try shortConv.encodeCentered(
                        enc, x: convInput, weight: scW.convW, bias: scW.convB, out: cOut16,
                        seqLen: seqLen, channels: ch, kernelSize: kSize, mediaPrefixLen: mediaPrefixLen
                    )
                } else {
                    try shortConv.encodeCausal(
                        enc, x: convInput, weight: scW.convW, bias: scW.convB,
                        stateIn: nil, out: cOut16, stateOut: nil,
                        seqLen: seqLen, channels: ch, kernelSize: kSize
                    )
                }
                enc.memoryBarrier(scope: .buffers)
                try headOp.encodeDecisionLinear(
                    enc, x: cOut16, weight: scW.outProjW, bias: scW.outProjB,
                    out: branch32, rows: seqLen, inFeatures: ch, outFeatures: hidden, activation: .none
                )

            case .attention(let topo, let qH, let kvH, let hd, let rope, let zeroTail, let hasGate, let scale):
                guard let aw = layer.attention else {
                    throw QWDecisionError.invalidQuestion("missing Attention weights")
                }
                let qElems = seqLen * qH * hd
                let kvElems = seqLen * kvH * hd
                let q32 = qBuf32.slice(byteOffset: 0, count: qElems, dtype: .f32)
                let k32 = kBuf32.slice(byteOffset: 0, count: kvElems, dtype: .f32)
                let v32 = vBuf32.slice(byteOffset: 0, count: kvElems, dtype: .f32)
                let q16 = qBuf16.slice(byteOffset: 0, count: qElems, dtype: .f16)
                let k16 = kBuf16.slice(byteOffset: 0, count: kvElems, dtype: .f16)
                let v16 = vBuf16.slice(byteOffset: 0, count: kvElems, dtype: .f16)
                let qRot = qRot16.slice(byteOffset: 0, count: qElems, dtype: .f16)
                let kRot = kRot16.slice(byteOffset: 0, count: kvElems, dtype: .f16)
                let aOut = attnOut16.slice(byteOffset: 0, count: qElems, dtype: .f16)

                try headOp.encodeDecisionLinear(
                    enc, x: normed16, weight: aw.qW, bias: aw.qB, out: q32,
                    rows: seqLen, inFeatures: hidden, outFeatures: qH * hd, activation: .none
                )
                try headOp.encodeDecisionLinear(
                    enc, x: normed16, weight: aw.kW, bias: aw.kB, out: k32,
                    rows: seqLen, inFeatures: hidden, outFeatures: kvH * hd, activation: .none
                )
                try headOp.encodeDecisionLinear(
                    enc, x: normed16, weight: aw.vW, bias: aw.vB, out: v32,
                    rows: seqLen, inFeatures: hidden, outFeatures: kvH * hd, activation: .none
                )
                enc.memoryBarrier(scope: .buffers)
                try QWGemm.encodeCast(enc, hal: hal, x32: q32, y16: q16, count: qElems)
                try QWGemm.encodeCast(enc, hal: hal, x32: k32, y16: k16, count: kvElems)
                try QWGemm.encodeCast(enc, hal: hal, x32: v32, y16: v16, count: kvElems)
                enc.memoryBarrier(scope: .buffers)

                let activeQ: QWTensorValue
                let activeK: QWTensorValue
                if let rope, let entry = ropeCache[headDimKey(hd: hd, entry: rope)] {
                    try QWRopeApply.encodeZeroTail(
                        enc, hal: hal, x: q16, out: qRot, table: entry.table,
                        seqLen: seqLen, numHeads: qH, headDim: hd, rotaryDims: entry.rotaryDims,
                        posOffset: 0, zeroTail: zeroTail
                    )
                    try QWRopeApply.encodeZeroTail(
                        enc, hal: hal, x: k16, out: kRot, table: entry.table,
                        seqLen: seqLen, numHeads: kvH, headDim: hd, rotaryDims: entry.rotaryDims,
                        posOffset: 0, zeroTail: zeroTail
                    )
                    enc.memoryBarrier(scope: .buffers)
                    activeQ = qRot
                    activeK = kRot
                } else {
                    activeQ = q16
                    activeK = k16
                }

                try topoAttn.encodePrefill(
                    enc, q: activeQ, k: activeK, v: v16, groupIds: groupTensor, out: aOut,
                    seqLen: seqLen, qHeads: qH, kvHeads: kvH, headDim: hd,
                    topology: topo, stateEndPos: stateEndPos, mediaPrefixLen: mediaPrefixLen, scale: scale
                )
                if hasGate, let gw = aw.gateW {
                    let gateAligned = qwAlignUp(seqLen * qH, 8)
                    let g32 = gateBuf32.slice(byteOffset: 0, count: seqLen * qH, dtype: .f32)
                    let g32Pad = gateBuf32.slice(byteOffset: 0, count: gateAligned, dtype: .f32)
                    let g16Pad = gateBuf16.slice(byteOffset: 0, count: gateAligned, dtype: .f16)
                    let g16 = gateBuf16.slice(byteOffset: 0, count: seqLen * qH, dtype: .f16)
                    try headOp.encodeDecisionLinear(
                        enc, x: normed16, weight: gw, bias: aw.gateB, out: g32,
                        rows: seqLen, inFeatures: hidden, outFeatures: qH, activation: .none
                    )
                    enc.memoryBarrier(scope: .buffers)
                    try QWGemm.encodeCast(enc, hal: hal, x32: g32Pad, y16: g16Pad, count: gateAligned)
                    enc.memoryBarrier(scope: .buffers)
                    try topoAttn.encodeHeadwiseSigmoidGate(enc, out: aOut, gate: g16, seqLen: seqLen, qHeads: qH, headDim: hd)
                }
                enc.memoryBarrier(scope: .buffers)
                try headOp.encodeDecisionLinear(
                    enc, x: aOut, weight: aw.oW, bias: aw.oB, out: branch32,
                    rows: seqLen, inFeatures: qH * hd, outFeatures: hidden, activation: .none
                )
            }

            enc.memoryBarrier(scope: .buffers)
            try encodeResidualMerge(
                cb: &cb, enc: &enc, x32Stale: &x32Stale,
                branch32: branch32, branch16: branch16,
                x16: x16, x32: x32, postNormKind: spec.postBlockNorm,
                postNormWeights: layer.postBlockNorm, rows: seqLen, hidden: hidden
            )
            enc.memoryBarrier(scope: .buffers)

            try encodeNorm(
                cb: cb, enc: &enc, x16: x16, out16: normed16,
                kind: spec.preFFNNorm, weights: layer.preFFNNorm, rows: seqLen, hidden: hidden
            )
            let mid32 = ffnMid32.slice(byteOffset: 0, count: seqLen * spec.ffnDim, dtype: .f32)
            try headOp.encodeDecisionLinear(
                enc, x: normed16, weight: layer.ffnUpW, bias: layer.ffnUpB, out: mid32,
                rows: seqLen, inFeatures: hidden, outFeatures: spec.ffnDim, activation: spec.ffnActivation
            )
            enc.memoryBarrier(scope: .buffers)
            try headOp.encodeDecisionLinear(
                enc, x: mid32, weight: layer.ffnDownW, bias: layer.ffnDownB, out: branch32,
                rows: seqLen, inFeatures: spec.ffnDim, outFeatures: hidden, activation: .none
            )
            enc.memoryBarrier(scope: .buffers)
            try encodeResidualMerge(
                cb: &cb, enc: &enc, x32Stale: &x32Stale,
                branch32: branch32, branch16: branch16,
                x16: x16, x32: x32, postNormKind: spec.postFFNNorm,
                postNormWeights: layer.postFFNNorm, rows: seqLen, hidden: hidden
            )
            enc.memoryBarrier(scope: .buffers)
        }

        if let finalNorm {
            try encodeNorm(
                cb: cb, enc: &enc, x16: x16, out16: normed16,
                kind: finalNorm.kind, weights: finalNorm.weights, rows: seqLen, hidden: hidden
            )
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            guard cb.status == .completed else {
                throw QWDecisionError.backendFailed("decision backbone CB failed (status=\(cb.status.rawValue))")
            }
            return normed16
        }

        hal.end(cb, enc)
        cb.waitUntilCompleted()
        guard cb.status == .completed else {
            throw QWDecisionError.backendFailed("decision backbone CB failed (status=\(cb.status.rawValue))")
        }
        return x16
    }

    /// Evaluates BF16 sparse candidate vocabulary logits over `[rows, hidden]` `.f16` hidden states.
    public func evaluateSparseVerbalizerBF16(
        hiddenStates: QWTensorValue,
        lmHeadBF16: QWTensorValue,
        rows: Int,
        hidden: Int,
        vocabSize: Int,
        candidateIds: [Int32]
    ) throws -> [Float] {
        guard !candidateIds.isEmpty else { return [] }
        let candTensor = try builder.makeI32(candidateIds, shape: [candidateIds.count])
        let outTensor = try builder.makeEmptyTensor(shape: [rows, candidateIds.count], dtype: .f32)
        guard let (cb, enc) = hal.begin() else {
            throw QWDecisionError.backendFailed("failed to open Metal command buffer")
        }
        try headOp.encodeSparseVocabRowsBF16(
            enc, hiddenState: hiddenStates, lmHeadBF16: lmHeadBF16,
            candidateIds: candTensor, out: outTensor,
            rows: rows, candCount: candidateIds.count, hidden: hidden, vocabSize: vocabSize
        )
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        guard cb.status == .completed else {
            throw QWDecisionError.backendFailed("sparse verbalizer CB failed (status=\(cb.status.rawValue))")
        }
        return readFloatTensor(outTensor, count: rows * candidateIds.count)
    }

    /// Evaluates state-pointer start/end logits across `stateHidden [stateLen, hidden]`.
    public func evaluateStatePointer(
        stateHidden: QWTensorValue,
        qStart: QWTensorValue,
        qEnd: QWTensorValue,
        stateLen: Int,
        hidden: Int
    ) throws -> (start: [Float], end: [Float]) {
        let outStart = try builder.makeEmptyTensor(shape: [stateLen], dtype: .f32)
        let outEnd = try builder.makeEmptyTensor(shape: [stateLen], dtype: .f32)
        guard let (cb, enc) = hal.begin() else {
            throw QWDecisionError.backendFailed("failed to open Metal command buffer")
        }
        try headOp.encodePointerScores(
            enc, stateHidden: stateHidden, qStart: qStart, qEnd: qEnd,
            outStart: outStart, outEnd: outEnd, stateLen: stateLen, hidden: hidden
        )
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        guard cb.status == .completed else {
            throw QWDecisionError.backendFailed("state pointer CB failed (status=\(cb.status.rawValue))")
        }
        return (readFloatTensor(outStart, count: stateLen), readFloatTensor(outEnd, count: stateLen))
    }

    /// Evaluates option-marker gather + optional type embedding + 2-layer MLP head (`[count, outDim]`).
    public func evaluateMarkerScorer(
        hiddenStates: QWTensorValue,
        seqLen: Int,
        hidden: Int,
        positions: [Int32],
        typeEmb: QWTensorValue?,
        w1: QWTensorValue,
        b1: QWTensorValue?,
        w2: QWTensorValue,
        b2: QWTensorValue?,
        midDim: Int,
        outDim: Int,
        activation: QWDecisionHeadActivation = .gelu
    ) throws -> [Float] {
        guard !positions.isEmpty else { return [] }
        let posTensor = try builder.makeI32(positions, shape: [positions.count])
        let gathered = try builder.makeEmptyTensor(shape: [positions.count, hidden], dtype: .f16)
        let mid = try builder.makeEmptyTensor(shape: [positions.count, midDim], dtype: .f32)
        let out = try builder.makeEmptyTensor(shape: [positions.count, outDim], dtype: .f32)
        guard let (cb, enc) = hal.begin() else {
            throw QWDecisionError.backendFailed("failed to open Metal command buffer")
        }
        try headOp.encodeMarkerGather(
            enc, hiddenState: hiddenStates, positions: posTensor, typeEmb: typeEmb,
            out: gathered, seqLen: seqLen, count: positions.count, hidden: hidden
        )
        enc.memoryBarrier(scope: .buffers)
        try headOp.encodeDecisionLinear(
            enc, x: gathered, weight: w1, bias: b1, out: mid,
            rows: positions.count, inFeatures: hidden, outFeatures: midDim, activation: activation
        )
        enc.memoryBarrier(scope: .buffers)
        try headOp.encodeDecisionLinear(
            enc, x: mid, weight: w2, bias: b2, out: out,
            rows: positions.count, inFeatures: midDim, outFeatures: outDim, activation: .none
        )
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        guard cb.status == .completed else {
            throw QWDecisionError.backendFailed("marker scorer CB failed (status=\(cb.status.rawValue))")
        }
        return readFloatTensor(out, count: positions.count * outDim)
    }

    /// Evaluates pooling (`.firstToken`/`.lastToken`/`.meanNonPad`) + 2-layer classifier MLP.
    public func evaluatePooledClassifier(
        hiddenStates: QWTensorValue,
        seqLen: Int,
        validLen: Int,
        hidden: Int,
        pooling: QWDecisionModelConfig.PoolingKind,
        w1: QWTensorValue,
        b1: QWTensorValue?,
        w2: QWTensorValue,
        b2: QWTensorValue?,
        midDim: Int,
        outDim: Int,
        activation: QWDecisionHeadActivation = .gelu
    ) throws -> [Float] {
        let pooled = try builder.makeEmptyTensor(shape: [1, hidden], dtype: .f16)
        let mid = try builder.makeEmptyTensor(shape: [1, midDim], dtype: .f32)
        let out = try builder.makeEmptyTensor(shape: [1, outDim], dtype: .f32)
        guard let (cb, enc) = hal.begin() else {
            throw QWDecisionError.backendFailed("failed to open Metal command buffer")
        }
        try headOp.encodePool(
            enc, hiddenState: hiddenStates, out: pooled,
            seqLen: seqLen, validLen: validLen, hidden: hidden, pooling: pooling
        )
        enc.memoryBarrier(scope: .buffers)
        try headOp.encodeDecisionLinear(
            enc, x: pooled, weight: w1, bias: b1, out: mid,
            rows: 1, inFeatures: hidden, outFeatures: midDim, activation: activation
        )
        enc.memoryBarrier(scope: .buffers)
        try headOp.encodeDecisionLinear(
            enc, x: mid, weight: w2, bias: b2, out: out,
            rows: 1, inFeatures: midDim, outFeatures: outDim, activation: .none
        )
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        guard cb.status == .completed else {
            throw QWDecisionError.backendFailed("pooled classifier CB failed (status=\(cb.status.rawValue))")
        }
        return readFloatTensor(out, count: outDim)
    }

    private func encodeNorm(
        cb: any MTLCommandBuffer,
        enc: inout any MTLComputeCommandEncoder,
        x16: QWTensorValue,
        out16: QWTensorValue,
        kind: QWDecisionNormKind,
        weights: QWDecisionNormWeights?,
        rows: Int,
        hidden: Int
    ) throws {
        switch kind {
        case .none:
            enc.endEncoding()
            guard let blit = cb.makeBlitCommandEncoder() else {
                throw QWDecisionError.backendFailed("failed to open blit encoder")
            }
            blit.copy(
                from: x16.region.buffer, sourceOffset: x16.region.base + x16.offset,
                to: out16.region.buffer, destinationOffset: out16.region.base + out16.offset,
                size: rows * hidden * 2
            )
            blit.endEncoding()
            guard let next = cb.makeComputeCommandEncoder() else {
                throw QWDecisionError.backendFailed("failed to reopen compute encoder")
            }
            enc = next
        case .layerNorm(let eps):
            guard let weights else { throw QWDecisionError.invalidQuestion("missing LayerNorm weights") }
            try QWNorm.encodeLayerNorm(
                enc, hal: hal, x: x16, gamma: weights.weight, beta: weights.bias,
                out: out16, rows: rows, hidden: hidden, eps: eps
            )
            enc.memoryBarrier(scope: .buffers)
        case .rmsNorm(let eps, let unitOffset):
            guard let weights else { throw QWDecisionError.invalidQuestion("missing RMSNorm weights") }
            try QWNorm.encodeArbitraryRMSNorm(
                enc, hal: hal, x: x16, gamma: weights.weight, out: out16,
                rows: rows, hidden: hidden, eps: eps, hasUnitOffset: unitOffset
            )
            enc.memoryBarrier(scope: .buffers)
        }
    }

    private func encodeResidualMerge(
        cb: inout any MTLCommandBuffer,
        enc: inout any MTLComputeCommandEncoder,
        x32Stale: inout Bool,
        branch32: QWTensorValue,
        branch16: QWTensorValue,
        x16: QWTensorValue,
        x32: QWTensorValue,
        postNormKind: QWDecisionNormKind,
        postNormWeights: QWDecisionNormWeights?,
        rows: Int,
        hidden: Int
    ) throws {
        switch postNormKind {
        case .none:
            if x32Stale {
                try syncX32FromX16(cb: &cb, enc: &enc, x16: x16, x32: x32, count: rows * hidden)
                x32Stale = false
            }
            try QWGemm.encodeAddF32(enc, hal: hal, a: x32, b: branch32, out: x32, count: rows * hidden)
            enc.memoryBarrier(scope: .buffers)
            try QWGemm.encodeCast(enc, hal: hal, x32: x32, y16: x16, count: rows * hidden)
        case .rmsNorm(let eps, let unitOffset):
            guard let weights = postNormWeights else {
                throw QWDecisionError.invalidQuestion("missing post-RMSNorm weights")
            }
            try QWGemm.encodeCast(enc, hal: hal, x32: branch32, y16: branch16, count: rows * hidden)
            enc.memoryBarrier(scope: .buffers)
            try QWNorm.encodePostRMSNormResidualAdd(
                enc, hal: hal, residual: x16, branch: branch16, gamma: weights.weight,
                rows: rows, hidden: hidden, eps: eps, hasUnitOffset: unitOffset
            )
            x32Stale = true
        case .layerNorm(let eps):
            guard let weights = postNormWeights else {
                throw QWDecisionError.invalidQuestion("missing post-LayerNorm weights")
            }
            if x32Stale {
                try syncX32FromX16(cb: &cb, enc: &enc, x16: x16, x32: x32, count: rows * hidden)
            }
            try QWGemm.encodeAddF32(enc, hal: hal, a: x32, b: branch32, out: x32, count: rows * hidden)
            enc.memoryBarrier(scope: .buffers)
            try QWGemm.encodeCast(enc, hal: hal, x32: x32, y16: branch16, count: rows * hidden)
            enc.memoryBarrier(scope: .buffers)
            try QWNorm.encodeLayerNorm(
                enc, hal: hal, x: branch16, gamma: weights.weight, beta: weights.bias,
                out: x16, rows: rows, hidden: hidden, eps: eps
            )
            x32Stale = true
        }
    }

    private func syncX32FromX16(
        cb: inout any MTLCommandBuffer,
        enc: inout any MTLComputeCommandEncoder,
        x16: QWTensorValue,
        x32: QWTensorValue,
        count: Int
    ) throws {
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        guard cb.status == .completed else {
            throw QWDecisionError.backendFailed("decision backbone CB failed before x32 sync (status=\(cb.status.rawValue))")
        }
        let src = x16.region.pointer.advanced(by: x16.offset).assumingMemoryBound(to: Float16.self)
        let dst = x32.region.pointer.advanced(by: x32.offset).assumingMemoryBound(to: Float.self)
        for i in 0..<count {
            dst[i] = Float(src[i])
        }
        guard let next = hal.begin() else {
            throw QWDecisionError.backendFailed("failed to reopen Metal command buffer")
        }
        cb = next.0
        enc = next.1
    }

    private func readFloatTensor(_ tensor: QWTensorValue, count: Int) -> [Float] {
        let ptr = tensor.region.pointer.advanced(by: tensor.offset).assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    private func headDimKey(hd: Int, entry: PerKindRoPEEntry) -> String {
        "\(hd):\(entry.ropeTheta):\(entry.partialRotaryFactor)"
    }
}
