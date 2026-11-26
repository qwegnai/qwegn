// QWDecisionPackEngine+Forward.swift — Coalesced single-CB Metal 4 forward pass across norms, activations, and heads.

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
import QWMM
import QWNorm
import QWRope

extension QWDecisionPackEngine {
    struct PreparedDecisionSeq {
        let tokens: [Int]
        let visualSpans: [QWVisualSpan]
        let encodedImages: [(grid: QWVisionGrid, features: [Float])]
        let topology: QWDecisionModelConfig.AttentionTopology
        let stateEndPos: Int
        let questionSpans: [Range<Int>]
        let markerPositions: [Int]
        let chain: String
        let expandedStateLen: Int
        let origToExpandedPtr: [Int?]
        let origPtrLen: Int
        let slotTargets: [(slotIdx: Int, pos: Int, endPos: Int?, isPtr: Bool)]
    }

    private struct ChunkPointerTarget {
        let itemIndex: Int
        let seqOffset: Int
        let expandedStateLen: Int
        let origToExpandedPtr: [Int?]
        let slotIdx: Int
        let pos: Int
        let endPos: Int?
        let startOut: QWTensorValue
        let endOut: QWTensorValue
    }

    func executeBatchOnGPU(
        _ items: [PreparedDecisionSeq],
        candidateIds: [Int]
    ) throws -> [(slotIdx: Int, output: QWDecisionSlotOutput)] {
        guard !items.isEmpty else { return [] }
        let maxTokensPerCB = 16_384
        var allResults: [(slotIdx: Int, output: QWDecisionSlotOutput)] = []
        allResults.reserveCapacity(items.reduce(0) { $0 + $1.slotTargets.count })

        var start = 0
        while start < items.count {
            var end = start
            var tokenSum = 0
            while end < items.count {
                let len = items[end].tokens.count
                if end > start && tokenSum + len > maxTokensPerCB { break }
                tokenSum += len
                end += 1
            }
            let chunk = Array(items[start..<end])
            let chunkRes = try executeBatchChunkOnGPU(chunk, candidateIds: candidateIds)
            allResults.append(contentsOf: chunkRes)
            start = end
        }
        return allResults
    }

    private func executeBatchChunkOnGPU(
        _ items: [PreparedDecisionSeq],
        candidateIds: [Int]
    ) throws -> [(slotIdx: Int, output: QWDecisionSlotOutput)] {
        var seqOffsets: [Int] = []
        seqOffsets.reserveCapacity(items.count)
        var totalSeqLen = 0
        var maxItemSeqLen = 1
        for item in items {
            seqOffsets.append(totalSeqLen)
            let len = item.tokens.count
            totalSeqLen += len
            if len > maxItemSeqLen { maxItemSeqLen = len }
        }
        guard totalSeqLen > 0 else { return [] }

        guard let (cb, enc) = hal.begin() else {
            throw QWDecisionError.backendFailed("failed to open Metal command buffer")
        }
        let final16 = try encodeBackbone(
            enc: enc, items: items, seqOffsets: seqOffsets,
            totalSeqLen: totalSeqLen, maxItemSeqLen: maxItemSeqLen
        )

        guard case .vocabVerbalizer(let lmHeadBF16) = weights.head else {
            return try executeNonVocabHeads(
                enc: enc, cb: cb, items: items, seqOffsets: seqOffsets,
                final16: final16, totalSeqLen: totalSeqLen
            )
        }

        let hidden = descriptor.hidden
        var globalCodeTargets: [(slotIdx: Int, globalPos: Int)] = []
        var ptrTargets: [ChunkPointerTarget] = []
        for (idx, item) in items.enumerated() {
            let off = seqOffsets[idx]
            let sLen = item.tokens.count
            for t in item.slotTargets {
                if t.isPtr {
                    if item.expandedStateLen > 0 {
                        let ptrIdx = ptrTargets.count
                        let oS = try scratchArena.tensor(key: "ptrS_\(ptrIdx)", shape: [item.expandedStateLen], dtype: .f32)
                        let oE = try scratchArena.tensor(key: "ptrE_\(ptrIdx)", shape: [item.expandedStateLen], dtype: .f32)
                        ptrTargets.append(ChunkPointerTarget(
                            itemIndex: idx, seqOffset: off, expandedStateLen: item.expandedStateLen,
                            origToExpandedPtr: item.origToExpandedPtr, slotIdx: t.slotIdx,
                            pos: max(0, min(sLen - 1, t.pos)),
                            endPos: t.endPos.map { max(0, min(sLen - 1, $0)) },
                            startOut: oS, endOut: oE
                        ))
                    }
                } else {
                    globalCodeTargets.append((slotIdx: t.slotIdx, globalPos: off + t.pos))
                }
            }
        }

        let validCandIds = candidateIds.filter { $0 >= 0 && $0 < descriptor.vocab }
        let uniquePosList = Array(Set(globalCodeTargets.map(\.globalPos))).sorted()
        var codeOutTensor: QWTensorValue?
        if !uniquePosList.isEmpty && !validCandIds.isEmpty, let lmHeadBF16 {
            let codePosTensor = try scratchArena.tensorI32(key: "codePos", values: uniquePosList.map(Int32.init), shape: [uniquePosList.count])
            let codeGathered16 = try scratchArena.tensor(key: "codeGath16", shape: [uniquePosList.count, hidden], dtype: .f16)
            let codeCandTensor = try scratchArena.tensorI32(key: "codeCand", values: validCandIds.map(Int32.init), shape: [validCandIds.count])
            let outT = try scratchArena.tensor(key: "codeOut32", shape: [uniquePosList.count, validCandIds.count], dtype: .f32)
            codeOutTensor = outT
            try headOp.encodeMarkerGather(
                enc, hiddenState: final16, positions: codePosTensor, typeEmb: nil, out: codeGathered16,
                seqLen: totalSeqLen, count: uniquePosList.count, hidden: hidden
            )
            enc.memoryBarrier(scope: .buffers)
            try headOp.encodeSparseVocabRowsBF16(
                enc, hiddenState: codeGathered16, lmHeadBF16: lmHeadBF16, candidateIds: codeCandTensor,
                out: outT, rows: uniquePosList.count, candCount: validCandIds.count,
                hidden: hidden, vocabSize: descriptor.vocab
            )
        }

        for pt in ptrTargets {
            let stateSlice = final16.slice(
                byteOffset: pt.seqOffset * hidden * 2, count: pt.expandedStateLen * hidden, dtype: .f16
            )
            let sPos = pt.seqOffset + pt.pos
            let ePos = pt.seqOffset + (pt.endPos ?? pt.pos)
            let qStart = final16.slice(byteOffset: sPos * hidden * 2, count: hidden, dtype: .f16)
            let qEnd = final16.slice(byteOffset: ePos * hidden * 2, count: hidden, dtype: .f16)
            try headOp.encodePointerScores(
                enc, stateHidden: stateSlice, qStart: qStart, qEnd: qEnd,
                outStart: pt.startOut, outEnd: pt.endOut,
                stateLen: pt.expandedStateLen, hidden: hidden
            )
        }

        try finishAndCheck(cb: cb, enc: enc, label: "forward")
        var results: [(slotIdx: Int, output: QWDecisionSlotOutput)] = []
        if let codeOutTensor {
            let posToRow = Dictionary(uniqueKeysWithValues: uniquePosList.enumerated().map { ($1, $0) })
            let outPtr = codeOutTensor.region.pointer.advanced(by: codeOutTensor.offset).assumingMemoryBound(to: Float.self)
            let stride = validCandIds.count
            for ct in globalCodeTargets {
                guard let rowIdx = posToRow[ct.globalPos] else { continue }
                var map: [Int: Float] = [:]
                map.reserveCapacity(stride)
                let base = rowIdx * stride
                for (cIdx, tid) in validCandIds.enumerated() { map[tid] = outPtr[base + cIdx] }
                results.append((ct.slotIdx, QWDecisionSlotOutput(tokenLogits: map)))
            }
        } else {
            for ct in globalCodeTargets { results.append((ct.slotIdx, QWDecisionSlotOutput())) }
        }

        for pt in ptrTargets {
            let sRaw = pt.startOut.region.pointer.advanced(by: pt.startOut.offset).assumingMemoryBound(to: Float.self)
            let eRaw = pt.endOut.region.pointer.advanced(by: pt.endOut.offset).assumingMemoryBound(to: Float.self)
            let sArr: [Float]
            let eArr: [Float]
            if pt.origToExpandedPtr.isEmpty {
                sArr = Array(UnsafeBufferPointer(start: sRaw, count: pt.expandedStateLen))
                eArr = Array(UnsafeBufferPointer(start: eRaw, count: pt.expandedStateLen))
            } else {
                sArr = pt.origToExpandedPtr.map { expIdx in
                    guard let expIdx, expIdx < pt.expandedStateLen else { return -Float.infinity }
                    return sRaw[expIdx]
                }
                eArr = pt.origToExpandedPtr.map { expIdx in
                    guard let expIdx, expIdx < pt.expandedStateLen else { return -Float.infinity }
                    return eRaw[expIdx]
                }
            }
            results.append((pt.slotIdx, QWDecisionSlotOutput(pointerStartLogits: sArr, pointerEndLogits: eArr)))
        }
        return results
    }

    func encodeBackbone(
        enc: any MTLComputeCommandEncoder,
        items: [PreparedDecisionSeq],
        seqOffsets: [Int],
        totalSeqLen: Int,
        maxItemSeqLen: Int
    ) throws -> QWTensorValue {
        let hidden = descriptor.hidden
        let ffnDim = descriptor.denseIntermediate ?? descriptor.hidden
        let qHeads = descriptor.qHeads
        let kvHeads = descriptor.kvHeads
        let headDim = descriptor.headDim
        let convCh = hidden

        let x16 = try scratchArena.tensor(key: "x16", shape: [totalSeqLen, hidden], dtype: .f16)
        let normed16 = try scratchArena.tensor(key: "normed16", shape: [totalSeqLen, hidden], dtype: .f16)
        let branch16 = try scratchArena.tensor(key: "branch16", shape: [totalSeqLen, hidden], dtype: .f16)
        let ffnIn16 = try scratchArena.tensor(key: "ffnIn16", shape: [totalSeqLen, hidden], dtype: .f16)
        let ffnGu16 = try scratchArena.tensor(key: "ffnGu16", shape: [totalSeqLen, 2 * ffnDim], dtype: .f16)
        let ffnAct16 = try scratchArena.tensor(key: "ffnAct16", shape: [totalSeqLen, ffnDim], dtype: .f16)
        let convBcx16 = try scratchArena.tensor(key: "convBcx16", shape: [totalSeqLen, 3 * convCh], dtype: .f16)
        let convOut16 = try scratchArena.tensor(key: "convOut16", shape: [totalSeqLen, convCh], dtype: .f16)
        let qBuf16 = try scratchArena.tensor(key: "qBuf16", shape: [totalSeqLen, qHeads * headDim], dtype: .f16)
        let kBuf16 = try scratchArena.tensor(key: "kBuf16", shape: [totalSeqLen, kvHeads * headDim], dtype: .f16)
        let vBuf16 = try scratchArena.tensor(key: "vBuf16", shape: [totalSeqLen, kvHeads * headDim], dtype: .f16)
        let qRot16 = try scratchArena.tensor(key: "qRot16", shape: [totalSeqLen, qHeads * headDim], dtype: .f16)
        let kRot16 = try scratchArena.tensor(key: "kRot16", shape: [totalSeqLen, kvHeads * headDim], dtype: .f16)
        let attnOut16 = try scratchArena.tensor(key: "attnOut16", shape: [totalSeqLen, qHeads * headDim], dtype: .f16)
        let gatePadCount = max(8, qwAlignUp(totalSeqLen * qHeads, 8))
        let gateBuf16 = try scratchArena.tensor(key: "gateBuf16", shape: [gatePadCount], dtype: .f16)
        let scratch32Count = max(totalSeqLen * 2 * ffnDim, max(totalSeqLen * 3 * convCh, max(totalSeqLen * qHeads * headDim, totalSeqLen * hidden)))
        let scratch32 = try scratchArena.tensor(key: "scratch32", shape: [scratch32Count], dtype: .f32)

        let isLayerNorm = {
            if case .layerNorm = descriptor.normKind { return true }
            return false
        }()
        let allSandwichRMS = !isLayerNorm && weights.layers.allSatisfy { $0.postBlockNorm != nil && $0.postFFNNorm != nil }
        let residualF32 = !allSandwichRMS ? (try scratchArena.tensor(key: "residualF32", shape: [totalSeqLen, hidden], dtype: .f32)) : nil
        let final16 = try scratchArena.tensor(key: "final16", shape: [totalSeqLen, hidden], dtype: .f16)

        var flatGroupIds = [Int32]()
        flatGroupIds.reserveCapacity(totalSeqLen)
        for (idx, item) in items.enumerated() {
            let xSlice = x16.slice(byteOffset: seqOffsets[idx] * hidden * 2, count: item.tokens.count * hidden, dtype: .f16)
            fillInputEmbeddings(tokens: item.tokens, encodedImages: item.encodedImages, visualSpans: item.visualSpans, into: xSlice)
            flatGroupIds.append(contentsOf: buildGroupIds(
                seqLen: item.tokens.count, stateEndPos: item.stateEndPos, questionSpans: item.questionSpans
            ))
        }

        if let embNorm = weights.embedNorm {
            try encodeNormTensor(enc, x16: x16, norm: embNorm, out16: normed16, rows: totalSeqLen, hidden: hidden)
            enc.memoryBarrier(scope: .buffers)
            try QWGemm.encodeCastF16ToF32(enc, hal: hal, x16: normed16, y32: scratch32.slice(byteOffset: 0, count: totalSeqLen * hidden, dtype: .f32), count: totalSeqLen * hidden)
            enc.memoryBarrier(scope: .buffers)
            try QWGemm.encodeCast(enc, hal: hal, x32: scratch32.slice(byteOffset: 0, count: totalSeqLen * hidden, dtype: .f32), y16: x16, count: totalSeqLen * hidden)
            enc.memoryBarrier(scope: .buffers)
        }

        let useGPUCastForResidual = (residualF32 != nil) && ((totalSeqLen * hidden) % 8 == 0)
        if let residualF32 {
            if useGPUCastForResidual {
                try QWGemm.encodeCastF16ToF32(enc, hal: hal, x16: x16, y32: residualF32, count: totalSeqLen * hidden)
                enc.memoryBarrier(scope: .buffers)
            } else {
                let src = x16.region.pointer.advanced(by: x16.offset).assumingMemoryBound(to: Float16.self)
                let dst = residualF32.region.pointer.advanced(by: residualF32.offset).assumingMemoryBound(to: Float.self)
                for i in 0..<(totalSeqLen * hidden) { dst[i] = Float(src[i]) }
            }
        }

        let groupTensor = try scratchArena.tensorI32(key: "groupIds", values: flatGroupIds, shape: [totalSeqLen])
        let ropeTables = try scratchArena.ropeTables(for: descriptor, maxSeqLen: maxItemSeqLen)

        for layer in weights.layers {
            try encodeOptionalNorm(enc, x16: x16, norm: layer.preBlockNorm, out16: normed16, rows: totalSeqLen, hidden: hidden)

            switch layer.op {
            case .conv(let cw):
                let convSrc16: QWTensorValue
                if let inProj = cw.inProj {
                    if inProj.needsRotation && hal.weightBasis.rotatesActivations {
                        try QWGemm.encodeFwht32(enc, hal: hal, x: normed16, count: totalSeqLen * hidden)
                        enc.memoryBarrier(scope: .buffers)
                    }
                    let projCount = cw.isGated ? (totalSeqLen * 3 * cw.channels) : (totalSeqLen * cw.channels)
                    let projDst = convBcx16.slice(byteOffset: 0, count: projCount, dtype: .f16)
                    try encodeLinear(enc, w: inProj, x16: normed16, y16: projDst, scratch32: scratch32, rows: totalSeqLen)
                    enc.memoryBarrier(scope: .buffers)
                    convSrc16 = projDst
                } else {
                    convSrc16 = normed16
                }
                for (idx, item) in items.enumerated() {
                    let sLen = item.tokens.count
                    let off = seqOffsets[idx]
                    let outSlice = convOut16.slice(byteOffset: off * convCh * 2, count: sLen * convCh, dtype: .f16)
                    if cw.isGated {
                        let bcxSlice = convSrc16.slice(byteOffset: off * 3 * convCh * 2, count: sLen * 3 * convCh, dtype: .f16)
                        try shortConv.encodeGatedCausal(
                            enc, bcx: bcxSlice, weight: cw.convW, bias: cw.convB, out: outSlice,
                            seqLen: sLen, channels: cw.channels, kernelSize: cw.kernelSize
                        )
                    } else if cw.isCentered {
                        let inSlice = convSrc16.slice(byteOffset: off * convCh * 2, count: sLen * convCh, dtype: .f16)
                        let mediaLen = item.visualSpans.last.map { $0.tokenOffset + $0.tokenCount } ?? 0
                        try shortConv.encodeCentered(
                            enc, x: inSlice, weight: cw.convW, bias: cw.convB, out: outSlice,
                            seqLen: sLen, channels: cw.channels, kernelSize: cw.kernelSize, mediaPrefixLen: mediaLen
                        )
                    } else {
                        let inSlice = convSrc16.slice(byteOffset: off * convCh * 2, count: sLen * convCh, dtype: .f16)
                        try shortConv.encodeCausal(
                            enc, x: inSlice, weight: cw.convW, bias: cw.convB,
                            stateIn: nil, out: outSlice, stateOut: nil,
                            seqLen: sLen, channels: cw.channels, kernelSize: cw.kernelSize
                        )
                    }
                }
                enc.memoryBarrier(scope: .buffers)
                if cw.outProj.needsRotation && hal.weightBasis.rotatesActivations {
                    try QWGemm.encodeFwht32(enc, hal: hal, x: convOut16, count: totalSeqLen * cw.channels)
                    enc.memoryBarrier(scope: .buffers)
                }
                try encodeOutAndResidual(
                    enc: enc, w: cw.outProj, bias: nil, in16: convOut16, branch16: branch16,
                    postNorm: layer.postBlockNorm, x16: x16, residualF32: residualF32,
                    scratch32: scratch32, seqLen: totalSeqLen, hidden: hidden
                )

            case .attn(let aw):
                let g16 = gateBuf16.slice(byteOffset: 0, count: totalSeqLen * qHeads, dtype: .f16)
                if let gateProj = aw.gateProj {
                    try encodeLinear(enc, w: gateProj, x16: normed16, y16: g16, scratch32: scratch32, rows: totalSeqLen)
                }
                if (aw.qProj.needsRotation || aw.kProj.needsRotation || aw.vProj.needsRotation) && hal.weightBasis.rotatesActivations {
                    try QWGemm.encodeFwht32(enc, hal: hal, x: normed16, count: totalSeqLen * hidden)
                    enc.memoryBarrier(scope: .buffers)
                }
                try encodeLinear(enc, w: aw.qProj, x16: normed16, y16: qBuf16, scratch32: scratch32, rows: totalSeqLen)
                try encodeLinear(enc, w: aw.kProj, x16: normed16, y16: kBuf16, scratch32: scratch32, rows: totalSeqLen)
                try encodeLinear(enc, w: aw.vProj, x16: normed16, y16: vBuf16, scratch32: scratch32, rows: totalSeqLen)
                enc.memoryBarrier(scope: .buffers)
                if let qb = aw.qBias { try addBiasIfAligned(enc, x16: qBuf16, bias: qb, rows: totalSeqLen, cols: qHeads * headDim) }
                if let kb = aw.kBias { try addBiasIfAligned(enc, x16: kBuf16, bias: kb, rows: totalSeqLen, cols: kvHeads * headDim) }
                if let vb = aw.vBias { try addBiasIfAligned(enc, x16: vBuf16, bias: vb, rows: totalSeqLen, cols: kvHeads * headDim) }
                if aw.qBias != nil || aw.kBias != nil || aw.vBias != nil { enc.memoryBarrier(scope: .buffers) }

                let rmsEps = descriptor.rmsEps
                if let qN = aw.qNorm {
                    try QWNorm.encodeArbitraryRMSNorm(
                        enc, hal: hal, x: qBuf16, gamma: qN, out: qBuf16,
                        rows: totalSeqLen * qHeads, hidden: headDim, eps: rmsEps, hasUnitOffset: false
                    )
                }
                if let kN = aw.kNorm {
                    try QWNorm.encodeArbitraryRMSNorm(
                        enc, hal: hal, x: kBuf16, gamma: kN, out: kBuf16,
                        rows: totalSeqLen * kvHeads, hidden: headDim, eps: rmsEps, hasUnitOffset: false
                    )
                }
                if aw.qNorm != nil || aw.kNorm != nil { enc.memoryBarrier(scope: .buffers) }

                let rt = ropeTables[aw.ropeKind] ?? ropeTables["default"]
                if let rt {
                    for (idx, item) in items.enumerated() {
                        let sLen = item.tokens.count
                        let off = seqOffsets[idx]
                        let qInSlice = qBuf16.slice(byteOffset: off * qHeads * headDim * 2, count: sLen * qHeads * headDim, dtype: .f16)
                        let qOutSlice = qRot16.slice(byteOffset: off * qHeads * headDim * 2, count: sLen * qHeads * headDim, dtype: .f16)
                        let kInSlice = kBuf16.slice(byteOffset: off * kvHeads * headDim * 2, count: sLen * kvHeads * headDim, dtype: .f16)
                        let kOutSlice = kRot16.slice(byteOffset: off * kvHeads * headDim * 2, count: sLen * kvHeads * headDim, dtype: .f16)
                        let rtSlice = rt.table.slice(byteOffset: 0, count: sLen * rt.rotaryDims, dtype: .f32)
                        try QWRopeApply.encodeZeroTail(
                            enc, hal: hal, x: qInSlice, out: qOutSlice, table: rtSlice,
                            seqLen: sLen, numHeads: qHeads, headDim: headDim, rotaryDims: rt.rotaryDims,
                            posOffset: 0, zeroTail: rt.zeroTailArg
                        )
                        try QWRopeApply.encodeZeroTail(
                            enc, hal: hal, x: kInSlice, out: kOutSlice, table: rtSlice,
                            seqLen: sLen, numHeads: kvHeads, headDim: headDim, rotaryDims: rt.rotaryDims,
                            posOffset: 0, zeroTail: rt.zeroTailArg
                        )
                    }
                    enc.memoryBarrier(scope: .buffers)
                }

                let qActiveAll = (rt != nil) ? qRot16 : qBuf16
                let kActiveAll = (rt != nil) ? kRot16 : kBuf16
                for (idx, item) in items.enumerated() {
                    let sLen = item.tokens.count
                    let off = seqOffsets[idx]
                    let qSlice = qActiveAll.slice(byteOffset: off * qHeads * headDim * 2, count: sLen * qHeads * headDim, dtype: .f16)
                    let kSlice = kActiveAll.slice(byteOffset: off * kvHeads * headDim * 2, count: sLen * kvHeads * headDim, dtype: .f16)
                    let vSlice = vBuf16.slice(byteOffset: off * kvHeads * headDim * 2, count: sLen * kvHeads * headDim, dtype: .f16)
                    let grpSlice = groupTensor.slice(byteOffset: off * 4, count: sLen, dtype: .i32)
                    let outSlice = attnOut16.slice(byteOffset: off * qHeads * headDim * 2, count: sLen * qHeads * headDim, dtype: .f16)
                    let attnTopo: QWDecisionModelConfig.AttentionTopology = aw.isBidirectional
                        ? .bidirectionalEncoder(isolateMediaPrefix: false)
                        : (aw.isSliding ? .causal(supportsPrefixCache: false) : item.topology)
                    let mediaLen = {
                        if case .bidirectionalEncoder(true) = attnTopo {
                            return item.visualSpans.last.map { $0.tokenOffset + $0.tokenCount } ?? 0
                        }
                        return 0
                    }()
                    try topoAttn.encodePrefill(
                        enc, q: qSlice, k: kSlice, v: vSlice, groupIds: grpSlice, out: outSlice,
                        seqLen: sLen, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
                        topology: attnTopo, stateEndPos: aw.isSliding ? 0 : item.stateEndPos,
                        mediaPrefixLen: mediaLen, slidingWindow: aw.isSliding ? aw.slidingWindow : 0
                    )
                }
                if aw.gateProj != nil {
                    enc.memoryBarrier(scope: .buffers)
                    try topoAttn.encodeHeadwiseSigmoidGate(
                        enc, out: attnOut16, gate: g16, seqLen: totalSeqLen, qHeads: qHeads, headDim: headDim
                    )
                }
                enc.memoryBarrier(scope: .buffers)
                if aw.oProj.needsRotation && hal.weightBasis.rotatesActivations {
                    try QWGemm.encodeFwht32(enc, hal: hal, x: attnOut16, count: totalSeqLen * qHeads * headDim)
                    enc.memoryBarrier(scope: .buffers)
                }
                try encodeOutAndResidual(
                    enc: enc, w: aw.oProj, bias: aw.oBias, in16: attnOut16, branch16: branch16,
                    postNorm: layer.postBlockNorm, x16: x16, residualF32: residualF32,
                    scratch32: scratch32, seqLen: totalSeqLen, hidden: hidden
                )
            }
            enc.memoryBarrier(scope: .buffers)

            try encodeOptionalNorm(enc, x16: x16, norm: layer.preFFNNorm, out16: ffnIn16, rows: totalSeqLen, hidden: hidden)

            if let w13 = layer.w13 {
                if w13.needsRotation && hal.weightBasis.rotatesActivations {
                    try QWGemm.encodeFwht32(enc, hal: hal, x: ffnIn16, count: totalSeqLen * hidden)
                    enc.memoryBarrier(scope: .buffers)
                }
                try encodeLinear(enc, w: w13, x16: ffnIn16, y16: ffnGu16, scratch32: scratch32, rows: totalSeqLen)
                enc.memoryBarrier(scope: .buffers)
                let fuseActRot = layer.w2.needsRotation && hal.weightBasis.rotatesActivations
                try QWGemmFP.encodeSwiGLUSoftcap(
                    enc, hal: hal, gu: QWFPBufferSlice(ffnGu16), act: QWFPBufferSlice(ffnAct16),
                    rows: totalSeqLen, intermediate: ffnDim, gateCap: 0, upCap: 0,
                    inDtype: .f16, outDtype: .f16, rotateFwht32: fuseActRot
                )
            } else if let w1 = layer.w1 {
                if w1.needsRotation && hal.weightBasis.rotatesActivations {
                    try QWGemm.encodeFwht32(enc, hal: hal, x: ffnIn16, count: totalSeqLen * hidden)
                    enc.memoryBarrier(scope: .buffers)
                }
                try encodeLinear(enc, w: w1, x16: ffnIn16, y16: ffnAct16, scratch32: scratch32, rows: totalSeqLen)
                enc.memoryBarrier(scope: .buffers)
                if let upB = layer.ffnUpBias {
                    try addBiasIfAligned(enc, x16: ffnAct16, bias: upB, rows: totalSeqLen, cols: ffnDim)
                    enc.memoryBarrier(scope: .buffers)
                }
                let tanhApprox = (descriptor.activationKind == .geluTanh)
                try encodeGeluF16(enc, x: ffnAct16, count: totalSeqLen * ffnDim, tanhApprox: tanhApprox)
            }
            enc.memoryBarrier(scope: .buffers)
            try encodeOutAndResidual(
                enc: enc, w: layer.w2, bias: layer.ffnDownBias, in16: ffnAct16, branch16: branch16,
                postNorm: layer.postFFNNorm, x16: x16, residualF32: residualF32,
                scratch32: scratch32, seqLen: totalSeqLen, hidden: hidden
            )
            enc.memoryBarrier(scope: .buffers)
        }

        if let fn = weights.finalNorm {
            try encodeNormTensor(enc, x16: x16, norm: fn, out16: final16, rows: totalSeqLen, hidden: hidden)
            enc.memoryBarrier(scope: .buffers)
            return final16
        }
        return x16
    }

    func encodeNormTensor(
        _ enc: any MTLComputeCommandEncoder,
        x16: QWTensorValue,
        norm: QWDecisionNormTensor,
        out16: QWTensorValue,
        rows: Int,
        hidden: Int
    ) throws {
        switch descriptor.normKind {
        case .layerNorm(let eps, _):
            try QWNorm.encodeLayerNorm(
                enc, hal: hal, x: x16, gamma: norm.weight, beta: norm.bias,
                out: out16, rows: rows, hidden: hidden, eps: eps
            )
        case .rmsNorm(let eps, let unitOffset):
            try QWNorm.encodeArbitraryRMSNorm(
                enc, hal: hal, x: x16, gamma: norm.weight, out: out16,
                rows: rows, hidden: hidden, eps: eps, hasUnitOffset: unitOffset
            )
        case nil:
            try QWNorm.encodeArbitraryRMSNorm(
                enc, hal: hal, x: x16, gamma: norm.weight, out: out16,
                rows: rows, hidden: hidden, eps: descriptor.rmsEps, hasUnitOffset: descriptor.usesPlusOneRMSNorm
            )
        }
        enc.memoryBarrier(scope: .buffers)
    }

    private func encodeOptionalNorm(
        _ enc: any MTLComputeCommandEncoder,
        x16: QWTensorValue,
        norm: QWDecisionNormTensor?,
        out16: QWTensorValue,
        rows: Int,
        hidden: Int
    ) throws {
        if let norm {
            try encodeNormTensor(enc, x16: x16, norm: norm, out16: out16, rows: rows, hidden: hidden)
            return
        }
        let s32 = try scratchArena.tensor(key: "normCopy32", shape: [rows, hidden], dtype: .f32)
        try QWGemm.encodeCastF16ToF32(enc, hal: hal, x16: x16, y32: s32, count: rows * hidden)
        enc.memoryBarrier(scope: .buffers)
        try QWGemm.encodeCast(enc, hal: hal, x32: s32, y16: out16, count: rows * hidden)
        enc.memoryBarrier(scope: .buffers)
    }

    private func encodeGeluF16(
        _ enc: any MTLComputeCommandEncoder,
        x: QWTensorValue,
        count: Int,
        tanhApprox: Bool
    ) throws {
        guard count > 0 else { return }
        let pso = try hal.pipeline(tanhApprox ? "qwg_vis_gelu_tanh_f16" : "qwg_vis_gelu_erf_f16")
        var countU = UInt32(count)
        enc.setComputePipelineState(pso)
        x.bind(enc, index: 0)
        enc.setBytes(&countU, length: 4, index: 1)
        enc.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(count, 256), height: 1, depth: 1)
        )
    }

    private func addBiasIfAligned(
        _ enc: any MTLComputeCommandEncoder,
        x16: QWTensorValue,
        bias: QWTensorValue,
        rows: Int,
        cols: Int
    ) throws {
        guard cols % 8 == 0 else { return }
        try QWNorm.encodeAddBiasF16(enc, hal: hal, x: x16, bias: bias, rows: rows, cols: cols)
    }

    private func encodeLinearF32(
        _ enc: any MTLComputeCommandEncoder,
        w: QWDecisionLinearW,
        x16: QWTensorValue,
        y32: QWTensorValue,
        rows: Int
    ) throws {
        switch w {
        case .bf16(let bank):
            try QWGemmFP.encodeBF16(
                enc, hal: hal, weight: bank, x: QWFPBufferSlice(x16), y: QWFPBufferSlice(y32),
                m: rows, inDtype: .f16, outDtype: .f32
            )
        case .quant(let view):
            if hal.caps.isMetal4TensorCapable && (view.n % 64 == 0 || !view.isIncumbentContainer) && view.k % 64 == 0 {
                let yTmp16 = y32.slice(byteOffset: 0, count: rows * view.n, dtype: .f16)
                try QWGemm.encodeNaxF32(enc, hal: hal, weight: view, x16: x16, y16: yTmp16, y32: y32, rows: rows)
            } else {
                try QWGemmAQ.encode(
                    enc, hal: hal, blocks: view.blocks, x: x16, y: y32,
                    n: view.n, k: view.k, bits: view.containerKind.bits, group: view.containerKind.groupSize,
                    rows: rows, isSHVQ: view.containerKind.isSHVQ
                )
            }
        }
    }

    private func encodeLinear(
        _ enc: any MTLComputeCommandEncoder,
        w: QWDecisionLinearW,
        x16: QWTensorValue,
        y16: QWTensorValue,
        scratch32: QWTensorValue,
        rows: Int
    ) throws {
        switch w {
        case .bf16(let bank):
            try QWGemmFP.encodeBF16(
                enc, hal: hal, weight: bank, x: QWFPBufferSlice(x16), y: QWFPBufferSlice(y16),
                m: rows, inDtype: .f16, outDtype: .f16
            )
        case .quant(let view):
            if hal.caps.isMetal4TensorCapable && (view.n % 64 == 0 || !view.isIncumbentContainer) && view.k % 64 == 0 {
                try QWGemm.encodeNaxF32(enc, hal: hal, weight: view, x16: x16, y16: y16, y32: y16, rows: rows)
            } else {
                let y32 = scratch32.slice(byteOffset: 0, count: rows * view.n, dtype: .f32)
                try QWGemmAQ.encode(
                    enc, hal: hal, blocks: view.blocks, x: x16, y: y32,
                    n: view.n, k: view.k, bits: view.containerKind.bits, group: view.containerKind.groupSize,
                    rows: rows, isSHVQ: view.containerKind.isSHVQ
                )
                enc.memoryBarrier(scope: .buffers)
                try QWGemm.encodeCast(enc, hal: hal, x32: y32, y16: y16, count: rows * view.n)
            }
        }
    }

    private func encodeOutAndResidual(
        enc: any MTLComputeCommandEncoder,
        w: QWDecisionLinearW,
        bias: QWTensorValue?,
        in16: QWTensorValue,
        branch16: QWTensorValue,
        postNorm: QWDecisionNormTensor?,
        x16: QWTensorValue,
        residualF32: QWTensorValue?,
        scratch32: QWTensorValue,
        seqLen: Int,
        hidden: Int
    ) throws {
        let isLayerNorm = {
            if case .layerNorm = descriptor.normKind { return true }
            return false
        }()
        if let postNorm, !isLayerNorm {
            try encodeLinear(enc, w: w, x16: in16, y16: branch16, scratch32: scratch32, rows: seqLen)
            enc.memoryBarrier(scope: .buffers)
            if let bias {
                try addBiasIfAligned(enc, x16: branch16, bias: bias, rows: seqLen, cols: hidden)
                enc.memoryBarrier(scope: .buffers)
            }
            let eps = descriptor.normKind?.eps ?? descriptor.rmsEps
            let unitOffset: Bool = {
                if case .rmsNorm(_, let u) = descriptor.normKind { return u }
                return descriptor.usesPlusOneRMSNorm
            }()
            try QWNorm.encodePostRMSNormResidualAdd(
                enc, hal: hal, residual: x16, branch: branch16, gamma: postNorm.weight,
                rows: seqLen, hidden: hidden, eps: eps, hasUnitOffset: unitOffset
            )
            return
        }
        guard let residualF32 else { return }
        let b32 = scratch32.slice(byteOffset: 0, count: seqLen * hidden, dtype: .f32)
        if let bias {
            try encodeLinear(enc, w: w, x16: in16, y16: branch16, scratch32: scratch32, rows: seqLen)
            enc.memoryBarrier(scope: .buffers)
            try addBiasIfAligned(enc, x16: branch16, bias: bias, rows: seqLen, cols: hidden)
            enc.memoryBarrier(scope: .buffers)
            try QWGemm.encodeCastF16ToF32(enc, hal: hal, x16: branch16, y32: b32, count: seqLen * hidden)
        } else {
            try encodeLinearF32(enc, w: w, x16: in16, y32: b32, rows: seqLen)
        }
        enc.memoryBarrier(scope: .buffers)
        try QWGemm.encodeAddF32(enc, hal: hal, a: residualF32, b: b32, out: residualF32, count: seqLen * hidden)
        enc.memoryBarrier(scope: .buffers)
        if let postNorm, isLayerNorm {
            try QWGemm.encodeCast(enc, hal: hal, x32: residualF32, y16: branch16, count: seqLen * hidden)
            enc.memoryBarrier(scope: .buffers)
            try encodeNormTensor(enc, x16: branch16, norm: postNorm, out16: x16, rows: seqLen, hidden: hidden)
            try QWGemm.encodeCastF16ToF32(enc, hal: hal, x16: x16, y32: residualF32, count: seqLen * hidden)
            enc.memoryBarrier(scope: .buffers)
        } else {
            try QWGemm.encodeCast(enc, hal: hal, x32: residualF32, y16: x16, count: seqLen * hidden)
        }
    }

    private func fillInputEmbeddings(
        tokens: [Int],
        encodedImages: [(grid: QWVisionGrid, features: [Float])],
        visualSpans: [QWVisualSpan],
        into x16: QWTensorValue
    ) {
        _ = encodedImages
        let hidden = descriptor.hidden
        let embPtr = weights.embedTokens.region.pointer.advanced(by: weights.embedTokens.offset).assumingMemoryBound(to: UInt16.self)
        let posPtr = weights.posEmbedBF16.map { $0.region.pointer.advanced(by: $0.offset).assumingMemoryBound(to: UInt16.self) }
        let posRows = weights.posEmbedBF16?.shape.dims.first ?? 0
        let dstPtr = x16.region.pointer.advanced(by: x16.offset).assumingMemoryBound(to: Float16.self)
        for (rowIdx, tok) in tokens.enumerated() {
            let clampedTok = max(0, min(descriptor.vocab - 1, tok))
            let srcRow = embPtr.advanced(by: clampedTok * hidden)
            let dstRow = dstPtr.advanced(by: rowIdx * hidden)
            if let posPtr, posRows > 0 {
                let pRow = posPtr.advanced(by: (rowIdx % posRows) * hidden)
                for d in 0..<hidden {
                    let eVal = Float(bitPattern: UInt32(srcRow[d]) << 16)
                    let pVal = Float(bitPattern: UInt32(pRow[d]) << 16)
                    dstRow[d] = Float16(eVal + pVal)
                }
            } else {
                for d in 0..<hidden {
                    dstRow[d] = Float16(Float(bitPattern: UInt32(srcRow[d]) << 16))
                }
            }
        }
        for span in visualSpans {
            let feats = span.embeddings
            let count = min(span.tokenCount, feats.count / hidden)
            for localRow in 0..<count {
                let dstRow = dstPtr.advanced(by: (span.tokenOffset + localRow) * hidden)
                let srcOff = localRow * hidden
                for d in 0..<hidden {
                    dstRow[d] = Float16(feats[srcOff + d])
                }
            }
        }
    }

    private func buildGroupIds(seqLen: Int, stateEndPos: Int, questionSpans: [Range<Int>]) -> [Int32] {
        if questionSpans.isEmpty && stateEndPos <= 0 {
            return [Int32](repeating: 0, count: seqLen)
        }
        var groups = [Int32](repeating: 0, count: seqLen)
        for (qIdx, span) in questionSpans.enumerated() {
            let lo = max(0, span.lowerBound)
            let hi = min(seqLen, span.upperBound)
            for p in lo..<hi { groups[p] = Int32(qIdx + 1) }
        }
        return groups
    }
}
