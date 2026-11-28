// QWDFlashDrafter+Propose.swift — DFlash 2 / DSpine block-diffusion
// context update, draft proposal, and candidate selection.

import Darwin
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHead
import QWHAL
import QWLoad
import QWQuant

extension QWDFlashDrafter {
    public func encodeCaptureTargetHidden(
        _ enc: any MTLComputeCommandEncoder,
        source: QWTensorValue,
        srcRowOffset: Int,
        dstRowOffset: Int,
        rows: Int,
        slot: Int
    ) {
        guard rows > 0, (0..<config.targetLayerIds.count).contains(slot) else { return }
        enc.setComputePipelineState(psoCaptureTarget)
        source.bind(enc, index: 0)
        targetHidden.bind(enc, index: 1)
        var sOff = UInt32(srcRowOffset), dOff = UInt32(dstRowOffset), slotU = UInt32(slot)
        var hiddenU4 = UInt32(config.hiddenSize / 8)
        var totalSlots = UInt32(config.targetLayerIds.count)
        enc.setBytes(&sOff, length: 4, index: 2)
        enc.setBytes(&dOff, length: 4, index: 3)
        enc.setBytes(&slotU, length: 4, index: 4)
        enc.setBytes(&hiddenU4, length: 4, index: 5)
        enc.setBytes(&totalSlots, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    public func encodeCaptureHCTargetHidden(
        _ enc: any MTLComputeCommandEncoder,
        base: QWTensorValue,
        srcRowOffset: Int,
        dstRowOffset: Int,
        rows: Int,
        slot: Int,
        hc: Int,
        deferredInject: (src: QWTensorValue, inj: QWTensorValue, isMHC: Bool)? = nil
    ) {
        guard rows > 0, hc > 0, (0..<config.targetLayerIds.count).contains(slot) else { return }
        enc.setComputePipelineState(psoCaptureHC)
        base.bind(enc, index: 0)
        if let inj = deferredInject {
            inj.src.bind(enc, index: 1)
            inj.inj.bind(enc, index: 2)
        } else {
            base.bind(enc, index: 1)
            base.bind(enc, index: 2)
        }
        targetHidden.bind(enc, index: 3)
        var sOff = UInt32(srcRowOffset), dOff = UInt32(dstRowOffset), slotU = UInt32(slot)
        var hiddenU4 = UInt32(config.hiddenSize / 8)
        var totalSlots = UInt32(config.targetLayerIds.count)
        var hcU = UInt32(hc)
        var flagsU = UInt32(deferredInject.map { $0.isMHC ? 2 : 1 } ?? 0)
        enc.setBytes(&sOff, length: 4, index: 4)
        enc.setBytes(&dOff, length: 4, index: 5)
        enc.setBytes(&slotU, length: 4, index: 6)
        enc.setBytes(&hiddenU4, length: 4, index: 7)
        enc.setBytes(&totalSlots, length: 4, index: 8)
        enc.setBytes(&hcU, length: 4, index: 9)
        enc.setBytes(&flagsU, length: 4, index: 10)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    public func encodeStagePropBF16(
        _ enc: any MTLComputeCommandEncoder,
        table: QWTensorValue,
        propX: QWTensorValue,
        anchorToken: Int,
        maskToken: Int,
        rows: Int
    ) {
        encodeStagePropBF16(
            enc, tableBuffer: table.region.buffer, byteOffset: table.region.base + table.offset,
            propX: propX, anchorToken: anchorToken, maskToken: maskToken, rows: rows
        )
    }

    public func encodeStagePropBF16(
        _ enc: any MTLComputeCommandEncoder,
        tableBuffer: any MTLBuffer,
        byteOffset: Int,
        propX: QWTensorValue,
        anchorToken: Int,
        maskToken: Int,
        rows: Int
    ) {
        guard rows > 0 else { return }
        let safeAnchor = Self.clampToken(anchorToken, vocab: config.vocabSize)
        let safeMask = Self.clampToken(maskToken, vocab: config.vocabSize)
        enc.setComputePipelineState(psoStagePropBF16)
        enc.setBuffer(tableBuffer, offset: byteOffset, index: 0)
        propX.bind(enc, index: 1)
        var anchorI = Int32(safeAnchor), maskI = Int32(safeMask)
        var hiddenU = UInt32(config.hiddenSize), vocabU = UInt32(config.vocabSize)
        enc.setBytes(&anchorI, length: 4, index: 2)
        enc.setBytes(&maskI, length: 4, index: 3)
        enc.setBytes(&hiddenU, length: 4, index: 4)
        enc.setBytes(&vocabU, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: (config.hiddenSize + 255) / 256, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    public func encodeStageProp(
        _ enc: any MTLComputeCommandEncoder,
        embedTable: QWDeviceEmbedTable,
        propX: QWTensorValue,
        anchorToken: Int,
        maskToken: Int,
        rows: Int
    ) throws {
        guard rows > 0 else { return }
        let safeAnchor = Self.clampToken(anchorToken, vocab: config.vocabSize)
        let safeMask = Self.clampToken(maskToken, vocab: config.vocabSize)
        switch embedTable.storage {
        case .bf16(let table):
            encodeStagePropBF16(enc, table: table, propX: propX,
                                anchorToken: safeAnchor, maskToken: safeMask, rows: rows)
        case .nativeAffine(let blocks, let bits, let group):
            let tokPtr = propTokens.region.buffer.contents()
                .advanced(by: propTokens.region.base + propTokens.offset)
                .assumingMemoryBound(to: Int32.self)
            tokPtr[0] = Int32(safeAnchor)
            for r in 1..<rows { tokPtr[r] = Int32(safeMask) }
            try QWGemmAQ.encodeEmbedAQ(
                enc, hal: hal, blocks: blocks,
                token: propTokens.slice(byteOffset: 0, count: rows),
                out: propX, hidden: config.hiddenSize, hc: 1, bits: bits, group: group, rows: rows
            )
        }
    }

    private struct DFlashTapPackP {
        var srcPlaneStrideU4: UInt32
        var srcRowOffset: UInt32
        var dstRowOffset: UInt32
        var hiddenU4: UInt32
        var totalSlots: UInt32
        var useRowMap: UInt32
        var rowMap: (
            UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
            UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32
        )
    }

    public func encodePackBF16Taps(
        _ enc: any MTLComputeCommandEncoder,
        source: any MTLBuffer,
        srcOffset: Int = 0,
        srcPlaneStrideElems: Int,
        srcRowOffset: Int = 0,
        dstRowOffset: Int = 0,
        rows: Int,
        rowMap: [Int]? = nil
    ) {
        let totalSlots = config.targetLayerIds.count
        guard rows > 0, totalSlots > 0 else { return }
        if let rowMap {
            precondition(rows <= 16, "encodePackBF16Taps rowMap supports at most 16 rows, got \(rows)")
            precondition(rowMap.count == rows, "encodePackBF16Taps rowMap count \(rowMap.count) != rows \(rows)")
        }
        let useMap = (rowMap != nil)
        func rm(_ idx: Int) -> UInt32 {
            guard useMap, let rowMap, idx < rowMap.count else { return 0 }
            return UInt32(rowMap[idx])
        }
        var p = DFlashTapPackP(
            srcPlaneStrideU4: UInt32(srcPlaneStrideElems / 8),
            srcRowOffset: UInt32(srcRowOffset),
            dstRowOffset: UInt32(dstRowOffset),
            hiddenU4: UInt32(config.hiddenSize / 8),
            totalSlots: UInt32(totalSlots),
            useRowMap: useMap ? 1 : 0,
            rowMap: (rm(0), rm(1), rm(2), rm(3), rm(4), rm(5), rm(6), rm(7),
                     rm(8), rm(9), rm(10), rm(11), rm(12), rm(13), rm(14), rm(15))
        )
        enc.setComputePipelineState(psoPackBF16Taps)
        enc.setBuffer(source, offset: srcOffset, index: 0)
        targetHidden.bind(enc, index: 1)
        enc.setBytes(&p, length: MemoryLayout<DFlashTapPackP>.stride, index: 2)
        enc.dispatchThreadgroups(
            MTLSize(width: rows, height: totalSlots, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    func encodeLinearMatmul(
        _ enc: any MTLComputeCommandEncoder,
        weight: DFlashLinearWeight,
        input: QWTensorValue,
        output: QWTensorValue,
        rows: Int
    ) throws {
        switch weight {
        case .quant(let w):
            try encodeQ4Matmul(enc, weight: w, input: input, output: output, rows: rows)
        case .bf16(let bank):
            try QWGemmFP.encodeBF16(
                enc, hal: hal, weight: bank,
                x: QWFPBufferSlice(input), y: QWFPBufferSlice(output),
                m: rows, inDtype: .f16, outDtype: .f16
            )
        }
    }

    func encodeLinearGateUpMatmul(
        _ enc: any MTLComputeCommandEncoder,
        weight: DFlashLinearWeight,
        input: QWTensorValue,
        output: QWTensorValue,
        rows: Int
    ) throws {
        switch weight {
        case .quant(let w):
            try encodeGateUpMatmul(enc, weight: w, input: input, output: output, rows: rows)
        case .bf16(let bank):
            let guScratch = QWTensorValue(
                region: draftLogits.region,
                offset: draftLogits.offset,
                shape: QWTensorShape([rows * 2 * config.intermediateSize], .f16)
            )
            try QWGemmFP.encodeBF16(
                enc, hal: hal, weight: bank,
                x: QWFPBufferSlice(input), y: QWFPBufferSlice(guScratch),
                m: rows, inDtype: .f16, outDtype: .f16
            )
            encodeSiluMulF16(enc, gu: guScratch, out: output, rows: rows, i: config.intermediateSize)
        }
    }

    func encodeQ4Matmul(_ enc: any MTLComputeCommandEncoder,
                        weight: QWWeightView,
                        input: QWTensorValue,
                        output: QWTensorValue,
                        rows: Int) throws {
        if !Self.usesQ4GMKernel(weight) {
            try encodeAQMatmul(enc, weight: weight, input: input, output: output, rows: rows)
            return
        }
        let n = weight.n, k = weight.k
        let minSimdRows = isMetal4TensorCapable ? 4 : 9
        let useNaxSmall = supportsSimdgroupGemm && rows >= minSimdRows && n % 64 == 0 && k % 64 == 0
        if rows == 1 {
            enc.setComputePipelineState(psoGemvT4)
            weight.blocks.bind(enc, index: 0)
            input.bind(enc, index: 1)
            output.bind(enc, index: 3)
            var nU = UInt32(n), kU = UInt32(k)
            enc.setBytes(&nU, length: 4, index: 4)
            enc.setBytes(&kU, length: 4, index: 5)
            output.bind(enc, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: (n + 3) / 4, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else if !useNaxSmall, let pso = psoDownBatched[rows] {
            let isAQ4 = !weight.isIncumbentContainer
            let rowPaired = Self.usesRowPairs(weight, activeRows: rows)
            let resolved = isAQ4
                ? (rowPaired ? (psoDownRowPairedAQ4[rows] ?? pso) : (psoDownBatchedAQ4[rows] ?? pso))
                : ((rowPaired ? psoDownRowPaired[rows] : nil) ?? pso)
            enc.setComputePipelineState(resolved)
            weight.blocks.bind(enc, index: 0)
            input.bind(enc, index: 1)
            output.bind(enc, index: 3)
            var nU = UInt32(n) | (isAQ4 ? 0x8000_0000 : 0), kU = UInt32(k)
            enc.setBytes(&nU, length: 4, index: 4)
            enc.setBytes(&kU, length: 4, index: 5)
            let tptg = rowPaired && (isAQ4 ? psoDownRowPairedAQ4[rows] != nil : psoDownRowPaired[rows] != nil) ? 64 : 128
            enc.dispatchThreadgroups(MTLSize(width: (n + 3) / 4, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1))
        } else if !useNaxSmall {
            var r0 = 0
            while r0 < rows {
                let c = min(8, rows - r0)
                try encodeQ4Matmul(enc, weight: weight, input: input.slice(byteOffset: r0 * k * 2, count: c * k),
                                   output: output.slice(byteOffset: r0 * n * 2, count: c * n), rows: c)
                r0 += c
            }
        } else if !weight.isIncumbentContainer {
            try encodeAQMatmul(enc, weight: weight, input: input, output: output, rows: rows)
        } else if let pso = psoGemmBatched {
            encodeSmallTileNAX(enc, pso: pso, weight: weight, input: input, output: output, rows: rows)
        } else {
            throw QWDFlashError.state("QWDFlashDrafter: missing NAX GEMM pipeline")
        }
    }

    func encodeRMSNorm(_ enc: any MTLComputeCommandEncoder,
                       x: QWTensorValue, gamma: QWTensorValue,
                       y: QWTensorValue, yRot: QWTensorValue,
                       rows: Int, mode: UInt32, epsOverride: Float? = nil,
                       yRot32: QWTensorValue? = nil) {
        enc.setComputePipelineState(psoRMSNorm)
        x.bind(enc, index: 0)
        gamma.bind(enc, index: 1)
        y.bind(enc, index: 2)
        yRot.bind(enc, index: 3)
        var eps = epsOverride ?? config.rmsNormEps, modeU = mode, hiddenU = UInt32(config.hiddenSize)
        enc.setBytes(&eps, length: 4, index: 4)
        enc.setBytes(&modeU, length: 4, index: 5)
        enc.setBytes(&hiddenU, length: 4, index: 6)
        (yRot32 ?? propRot32).bind(enc, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    func encodeF16GemvBatched(_ enc: any MTLComputeCommandEncoder,
                              w: QWTensorValue, x: QWTensorValue, y: QWTensorValue,
                              n: Int, k: Int, m: Int) {
        enc.setComputePipelineState(psoF16GemvBatched)
        w.bind(enc, index: 0)
        x.bind(enc, index: 1)
        y.bind(enc, index: 2)
        var nU = UInt32(n), kU = UInt32(k), mU = UInt32(m)
        enc.setBytes(&nU, length: 4, index: 3)
        enc.setBytes(&kU, length: 4, index: 4)
        enc.setBytes(&mU, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: (n + 3) / 4, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    func encodeConvApply(_ enc: any MTLComputeCommandEncoder,
                         x: QWTensorValue, dyn: QWTensorValue,
                         baseKernel: QWTensorValue, outOrResidual: QWTensorValue,
                         rows: Int, mode: UInt32) {
        enc.setComputePipelineState(psoConvApply)
        x.bind(enc, index: 0)
        dyn.bind(enc, index: 1)
        baseKernel.bind(enc, index: 2)
        outOrResidual.bind(enc, index: 3)
        var modeU = mode
        let hiddenU4Int = config.hiddenSize / 8
        var hiddenU4 = UInt32(hiddenU4Int)
        var convGroupsU = UInt32(config.hiddenSize / max(1, config.convGroupSize))
        enc.setBytes(&modeU, length: 4, index: 4)
        enc.setBytes(&hiddenU4, length: 4, index: 5)
        enc.setBytes(&convGroupsU, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: max(1, (hiddenU4Int + 31) / 32), depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    public func encodeContextUpdate(_ cb: any MTLCommandBuffer,
                                    slot: QWDFlashSessionSlot,
                                    startPosition: Int,
                                    totalCount: Int) throws {
        let captureRows = min(totalCount, config.maxCacheSize)
        guard captureRows > 0 else { return }
        guard let enc = cb.makeComputeCommandEncoder() else {
            throw QWDFlashError.state("QWDFlashDrafter: failed to create context encoder")
        }
        defer { enc.endEncoding() }
        try encodeContextUpdate(enc, slot: slot, startPosition: startPosition, totalCount: totalCount)
    }

    public func encodeContextUpdate(_ enc: any MTLComputeCommandEncoder,
                                    slot: QWDFlashSessionSlot,
                                    startPosition: Int,
                                    totalCount: Int) throws {
        if let mla = mlaState {
            try encodeMLAContextUpdate(enc, slot: slot, startPosition: startPosition, totalCount: totalCount, mla: mla)
            return
        }
        let captureRows = min(totalCount, config.maxCacheSize)
        guard captureRows > 0 else { return }
        let posOffset = startPosition + (totalCount - captureRows)
        let rowDrop = totalCount - captureRows
        let rowStrideBytes = config.targetLayerIds.count * config.hiddenSize * 2
        let fcInput: QWTensorValue
        if rowDrop > 0 && targetHidden.bytes >= totalCount * rowStrideBytes {
            fcInput = targetHidden.slice(
                byteOffset: rowDrop * rowStrideBytes,
                count: captureRows * config.targetLayerIds.count * config.hiddenSize,
                dtype: .f16
            )
        } else {
            fcInput = targetHidden
        }

        try encodeLinearMatmul(enc, weight: fc, input: fcInput, output: hCtx, rows: captureRows)
        enc.memoryBarrier(scope: .buffers)
        encodeRMSNorm(enc, x: hCtx, gamma: hiddenNorm, y: hCtx, yRot: hCtx, rows: captureRows, mode: 1)
        enc.memoryBarrier(scope: .buffers)

        var cOff = UInt32(posOffset), eps = config.rmsNormEps
        var maxCacheU = UInt32(config.maxCacheSize), kvHeadsU = UInt32(config.numKeyValueHeads)
        for l in 0..<config.numLayers {
            let lw = layers[l]
            try encodeLinearMatmul(enc, weight: lw.kvProj, input: hCtx, output: kvCtx, rows: captureRows)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(psoRopeKVCtx)
            kvCtx.bind(enc, index: 0)
            lw.kNorm.bind(enc, index: 1)
            invFreq.bind(enc, index: 2)
            slot.kLayers[l].bind(enc, index: 3)
            slot.vLayers[l].bind(enc, index: 4)
            enc.setBytes(&cOff, length: 4, index: 5)
            enc.setBytes(&eps, length: 4, index: 6)
            enc.setBytes(&maxCacheU, length: 4, index: 7)
            enc.setBytes(&kvHeadsU, length: 4, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: captureRows, height: config.numKeyValueHeads, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
        }
    }

    public func enqueueGreedyPropose(
        hal: QWHal,
        slot: QWDFlashSessionSlot,
        cacheOffset: Int,
        pendingToken: Int,
        draftCount: Int,
        draftVocab: Int? = nil,
        headStorage: QWHeadStorage? = nil,
        encodeHeadLogits: (_ enc: any MTLComputeCommandEncoder, _ xRot16: QWTensorValue, _ xNorm16: QWTensorValue,
                           _ xRot32: QWTensorValue?, _ yLogits32: QWTensorValue, _ rows: Int, _ vocabRows: Int) throws -> Void,
        encodeAnchorMaskEmbed: (_ enc: any MTLComputeCommandEncoder, _ propX32: QWTensorValue,
                                _ anchorToken: Int, _ maskToken: Int, _ rows: Int) throws -> Void,
        journalBuffer: any MTLBuffer,
        journalByteOffset: Int
    ) throws -> any MTLCommandBuffer {
        guard journalByteOffset >= 0, journalByteOffset % 4 == 0,
              journalByteOffset + draftCount * 4 <= journalBuffer.length,
              journalBuffer !== draftTokensOut.region.buffer else {
            throw QWDFlashError.state("QWDFlashDrafter: invalid async journal destination")
        }
        return try withoutActuallyEscaping(encodeAnchorMaskEmbed) { anchorEmbed in
            try encodeAndCommitPropose(
                hal: hal, slot: slot, cacheOffset: cacheOffset, pendingToken: pendingToken,
                draftCount: draftCount, draftVocab: draftVocab, headStorage: headStorage,
                encodeHeadLogits: encodeHeadLogits, encodeAnchorMaskEmbed: anchorEmbed,
                embedding: nil, stochastic: nil, preEncode: nil, preContextUpdate: nil,
                journalCopy: (journalBuffer, journalByteOffset)
            )
        }
    }

    public func propose(
        hal: QWHal,
        slot: QWDFlashSessionSlot,
        cacheOffset: Int,
        pendingToken: Int,
        draftCount: Int,
        draftVocab: Int? = nil,
        headStorage: QWHeadStorage? = nil,
        encodeHeadLogits: (_ enc: any MTLComputeCommandEncoder, _ xRot16: QWTensorValue, _ xNorm16: QWTensorValue,
                           _ xRot32: QWTensorValue?, _ yLogits32: QWTensorValue, _ rows: Int, _ vocabRows: Int) throws -> Void,
        encodeAnchorMaskEmbed: ((_ enc: any MTLComputeCommandEncoder, _ propX32: QWTensorValue,
                                 _ anchorToken: Int, _ maskToken: Int, _ rows: Int) throws -> Void)? = nil,
        embedding: ((Int, QWTensorValue) throws -> Void)? = nil,
        stochastic: (temperature: Float, stream: QWMTPRandomStream)? = nil,
        preEncode: ((_ enc: any MTLComputeCommandEncoder) throws -> Void)? = nil,
        preContextUpdate: (slot: QWDFlashSessionSlot, startPosition: Int, totalCount: Int)? = nil,
        completed: ((any MTLCommandBuffer) throws -> Void)? = nil
    ) throws -> (drafts: [Int], proposals: [QWMTPGumbelProposal]) {
        let cb = try encodeAndCommitPropose(
            hal: hal, slot: slot, cacheOffset: cacheOffset, pendingToken: pendingToken,
            draftCount: draftCount, draftVocab: draftVocab, headStorage: headStorage,
            encodeHeadLogits: encodeHeadLogits, encodeAnchorMaskEmbed: encodeAnchorMaskEmbed,
            embedding: embedding, stochastic: stochastic, preEncode: preEncode,
            preContextUpdate: preContextUpdate,
            journalCopy: nil
        )
        cb.waitUntilCompleted()
        guard cb.status == .completed, cb.error == nil else {
            throw QWDFlashError.state("QWDFlashDrafter GPU failure: \(String(describing: cb.error))")
        }
        try completed?(cb)

        let D = draftCount
        let V = config.vocabSize
        let topK = config.selectorTopK
        let safeAnchor = Self.clampToken(pendingToken, vocab: V)
        let vocabRows = draftVocab.map { min(V, max(1, $0)) } ?? V
        if let stochastic {
            if let bs = dspineScratch {
                return try selectDSpineStochastic(
                    temperature: stochastic.temperature, stream: stochastic.stream, draftCount: D, topK: topK, bs: bs
                )
            }
            return try selectStochastic(
                temperature: stochastic.temperature, stream: stochastic.stream,
                anchorToken: safeAnchor, draftCount: D, topK: topK, validVocab: vocabRows
            )
        } else {
            let ptr = draftTokensOut.region.buffer.contents()
                .advanced(by: draftTokensOut.region.base + draftTokensOut.offset)
                .assumingMemoryBound(to: Int32.self)
            let drafts = (0..<D).map { Self.clampToken(Int(ptr[$0]), vocab: vocabRows) }
            return (drafts, [])
        }
    }

    private func encodeAndCommitPropose(
        hal: QWHal,
        slot: QWDFlashSessionSlot,
        cacheOffset: Int,
        pendingToken: Int,
        draftCount: Int,
        draftVocab: Int?,
        headStorage: QWHeadStorage?,
        encodeHeadLogits: (_ enc: any MTLComputeCommandEncoder, _ xRot16: QWTensorValue, _ xNorm16: QWTensorValue,
                           _ xRot32: QWTensorValue?, _ yLogits32: QWTensorValue, _ rows: Int, _ vocabRows: Int) throws -> Void,
        encodeAnchorMaskEmbed: ((_ enc: any MTLComputeCommandEncoder, _ propX32: QWTensorValue,
                                 _ anchorToken: Int, _ maskToken: Int, _ rows: Int) throws -> Void)?,
        embedding: ((Int, QWTensorValue) throws -> Void)?,
        stochastic: (temperature: Float, stream: QWMTPRandomStream)?,
        preEncode: ((_ enc: any MTLComputeCommandEncoder) throws -> Void)?,
        preContextUpdate: (slot: QWDFlashSessionSlot, startPosition: Int, totalCount: Int)?,
        journalCopy: (buffer: any MTLBuffer, byteOffset: Int)?
    ) throws -> any MTLCommandBuffer {
        let D = draftCount
        let M = D + 1
        let maxDrafts = max(7, config.blockSize - 1)
        guard (1...maxDrafts).contains(D) else {
            throw QWDFlashError.state("QWDFlashDrafter: invalid draftCount \(D) (expected 1...\(maxDrafts))")
        }
        let H = config.hiddenSize
        let V = config.vocabSize
        let vocabRows = draftVocab.map { min(V, max(1, $0)) } ?? V
        guard vocabRows <= 393_216 else {
            throw QWDFlashError.state("QWDFlashDrafter: vocabRows \(vocabRows) exceeds 393216 limit")
        }
        let R = config.selectorRank
        let topK = config.selectorTopK
        let hasConv = config.convKernelSize > 0
        let convProjDim = hasConv ? (2 * config.convKernelSize * (H / config.convGroupSize)) : 0
        let safeAnchor = Self.clampToken(pendingToken, vocab: V)
        let safeMask = Self.clampToken(config.maskTokenId, vocab: V)

        if encodeAnchorMaskEmbed == nil {
            guard let embedding else {
                throw QWDFlashError.state("QWDFlashDrafter: missing anchor/mask embedding provider")
            }
            let row0 = QWTensorValue(region: propNorm.region, offset: propNorm.offset, shape: QWTensorShape([H], .f16))
            let row1 = QWTensorValue(region: propNorm.region, offset: propNorm.offset + H * 2, shape: QWTensorShape([H], .f16))
            try embedding(safeAnchor, row0)
            try embedding(safeMask, row1)
            let src16 = propNorm.region.buffer.contents()
                .advanced(by: propNorm.region.base + propNorm.offset)
                .assumingMemoryBound(to: Float16.self)
            let dst32 = propX.region.buffer.contents()
                .advanced(by: propX.region.base + propX.offset)
                .assumingMemoryBound(to: Float.self)
            for i in 0..<H {
                dst32[i] = Float(src16[i])
                dst32[H + i] = Float(src16[H + i])
            }
            if M > 2 {
                let maskRow32 = UnsafeRawPointer(dst32.advanced(by: H))
                withExtendedLifetime(propX.region.buffer) {
                    for m in 2..<M {
                        UnsafeMutableRawPointer(dst32.advanced(by: m * H)).copyMemory(from: maskRow32, byteCount: H * 4)
                    }
                }
            }
        }

        guard let (cb, enc) = hal.beginUnretained(.serial) else {
            throw QWDFlashError.state("QWDFlashDrafter: command buffer creation failed")
        }
        var encoderEnded = false
        defer {
            if !encoderEnded {
                enc.endEncoding()
            }
        }

        if let preEncode {
            try preEncode(enc)
            enc.memoryBarrier(scope: .buffers)
        }
        if let pre = preContextUpdate, min(pre.totalCount, config.maxCacheSize) > 0 {
            try encodeContextUpdate(enc, slot: pre.slot, startPosition: pre.startPosition, totalCount: pre.totalCount)
            enc.memoryBarrier(scope: .buffers)
        }
        if let encodeAnchorMaskEmbed {
            let propXSlice = QWTensorValue(region: propX.region, offset: propX.offset, shape: QWTensorShape([M * H], .f32))
            try encodeAnchorMaskEmbed(enc, propXSlice, safeAnchor, safeMask, M)
            enc.memoryBarrier(scope: .buffers)
        }

        if let mla = mlaState {
            let draftRowByteOffset = config.shiftLabel ? 0 : (H * 4)
            let propXDraftSlice = QWTensorValue(
                region: propX.region, offset: propX.offset + draftRowByteOffset, shape: QWTensorShape([D * H], .f32)
            )
            let propNormSlice = QWTensorValue(region: propNorm.region, offset: propNorm.offset,
                                              shape: QWTensorShape([D * H], .f16))
            let propRotSlice = QWTensorValue(region: propRot.region, offset: propRot.offset,
                                             shape: QWTensorShape([D * H], .f16))
            let propRot32Slice = QWTensorValue(region: propRot32.region, offset: propRot32.offset,
                                               shape: QWTensorShape([D * H], .f32))
            try encodeMLAProposeLayersAndFinalNorm(
                enc, slot: slot, cacheOffset: cacheOffset, rows: M, draftCount: D,
                propXDraftSlice: propXDraftSlice, propNormSlice: propNormSlice,
                propRotSlice: propRotSlice, propRot32Slice: propRot32Slice, mla: mla
            )
        } else {
            if config.isDSpine {
                encodeRMSNorm(enc, x: propX, gamma: layers[0].ln1, y: propNorm, yRot: propRot, rows: M, mode: 2)
            }

            var maxCacheU = UInt32(config.maxCacheSize)
            var isCausalU = UInt32(config.draftBlockCausal ? 1 : 0)
            var qHeadsU = UInt32(config.numAttentionHeads)
            var kvHeadsU = UInt32(config.numKeyValueHeads)

            for l in 0..<config.numLayers {
                let lw = layers[l]
                if !config.isDSpine, let acProj = lw.attnConvProj, let acBase = lw.attnConvBase, let pDyn = propDyn {
                    encodeRMSNorm(enc, x: propX, gamma: lw.ln1, y: propNorm, yRot: propNorm, rows: M, mode: 0)
                    encodeF16GemvBatched(enc, w: acProj, x: propNorm, y: pDyn, n: convProjDim, k: H, m: M)
                    encodeConvApply(enc, x: propNorm, dyn: pDyn, baseKernel: acBase, outOrResidual: propRot, rows: M, mode: 0)
                }
                try encodeLinearMatmul(enc, weight: lw.qkvProj, input: propRot, output: qkvProp, rows: M)

                enc.setComputePipelineState(psoRopeQKProp)
                qkvProp.bind(enc, index: 0)
                lw.qNorm.bind(enc, index: 1)
                lw.kNorm.bind(enc, index: 2)
                invFreq.bind(enc, index: 3)
                var pOff = UInt32(cacheOffset), eps = config.rmsNormEps
                enc.setBytes(&pOff, length: 4, index: 4)
                enc.setBytes(&eps, length: 4, index: 5)
                enc.setBytes(&qHeadsU, length: 4, index: 6)
                enc.setBytes(&kvHeadsU, length: 4, index: 7)
                enc.dispatchThreadgroups(MTLSize(width: M, height: config.numAttentionHeads + config.numKeyValueHeads, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

                enc.setComputePipelineState(psoAttn)
                qkvProp.bind(enc, index: 0)
                slot.kLayers[l].bind(enc, index: 1)
                slot.vLayers[l].bind(enc, index: 2)
                attnOut.bind(enc, index: 3)
                var nOff = UInt32(cacheOffset), mU = UInt32(M)
                enc.setBytes(&nOff, length: 4, index: 4)
                enc.setBytes(&mU, length: 4, index: 5)
                enc.setBytes(&maxCacheU, length: 4, index: 6)
                enc.setBytes(&isCausalU, length: 4, index: 7)
                enc.setBytes(&qHeadsU, length: 4, index: 8)
                enc.setBytes(&kvHeadsU, length: 4, index: 9)
                enc.dispatchThreadgroups(MTLSize(width: config.numAttentionHeads, height: M, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

                if config.isDSpine {
                    try encodeLinearMatmul(enc, weight: lw.oProj, input: attnOut, output: propDown, rows: M)
                    encodeRMSNorm(enc, x: propX, gamma: lw.ln2, y: propDown, yRot: propRot, rows: M, mode: 3)
                } else if let acBase = lw.attnConvBase, let mcProj = lw.mlpConvProj,
                          let mcBase = lw.mlpConvBase, let pDyn = propDyn {
                    try encodeLinearMatmul(enc, weight: lw.oProj, input: attnOut, output: propNorm, rows: M)
                    encodeConvApply(enc, x: propNorm, dyn: pDyn, baseKernel: acBase, outOrResidual: propX, rows: M, mode: 1)
                    encodeRMSNorm(enc, x: propX, gamma: lw.ln2, y: propNorm, yRot: propNorm, rows: M, mode: 0)
                    encodeF16GemvBatched(enc, w: mcProj, x: propNorm, y: pDyn, n: convProjDim, k: H, m: M)
                    encodeConvApply(enc, x: propNorm, dyn: pDyn, baseKernel: mcBase, outOrResidual: propRot, rows: M, mode: 0)
                }

                try encodeLinearGateUpMatmul(enc, weight: lw.gateUpProj, input: propRot, output: mlpInter, rows: M)
                try encodeLinearMatmul(enc, weight: lw.downProj, input: mlpInter, output: propDown, rows: M)
                if let bw = dspineBridge, let bs = dspineScratch {
                    encodeDSpineBridgeLayer(enc, layer: l, anchorToken: safeAnchor, rows: M, bw: bw, bs: bs)
                } else if let mcBase = lw.mlpConvBase, let pDyn = propDyn {
                    encodeConvApply(enc, x: propDown, dyn: pDyn, baseKernel: mcBase, outOrResidual: propX, rows: M, mode: 1)
                }
            }

            let draftRowByteOffset = config.shiftLabel ? 0 : (H * 4)
            let propXDraftSlice = dspineScratch?.propXPost4
                ?? QWTensorValue(
                    region: propX.region, offset: propX.offset + draftRowByteOffset, shape: QWTensorShape([D * H], .f32)
                )
            let propNormSlice = QWTensorValue(region: propNorm.region, offset: propNorm.offset,
                                              shape: QWTensorShape([D * H], .f16))
            let propRotSlice = QWTensorValue(region: propRot.region, offset: propRot.offset,
                                             shape: QWTensorShape([D * H], .f16))
            let propRot32Slice = QWTensorValue(region: propRot32.region, offset: propRot32.offset,
                                               shape: QWTensorShape([D * H], .f32))
            encodeRMSNorm(enc, x: propXDraftSlice, gamma: finalNorm,
                          y: propNormSlice, yRot: propRotSlice, rows: D, mode: 5, yRot32: propRot32Slice)
        }

        let propNormSlice = QWTensorValue(region: propNorm.region, offset: propNorm.offset,
                                          shape: QWTensorShape([D * H], .f16))
        let propRotSlice = QWTensorValue(region: propRot.region, offset: propRot.offset,
                                         shape: QWTensorShape([D * H], .f16))
        let propRot32Slice = QWTensorValue(region: propRot32.region, offset: propRot32.offset,
                                           shape: QWTensorShape([D * H], .f32))
        let logitsSlice = QWTensorValue(region: draftLogits.region, offset: draftLogits.offset,
                                        shape: QWTensorShape([D * vocabRows], .f32))
        if !config.isDSpine, let selProj = selectorHiddenProj, let hProjOut = hiddenProjOut {
            encodeF16GemvBatched(enc, w: selProj, x: propNormSlice, y: hProjOut, n: R, k: H, m: D)
        }

        try encodeHeadLogits(enc, propRotSlice, propNormSlice, propRot32Slice, logitsSlice, D, vocabRows)

        var vocabU = UInt32(vocabRows), topKU = UInt32(topK)
        enc.setComputePipelineState(psoTop16Stage1)
        logitsSlice.bind(enc, index: 0)
        top16PartialsVal.bind(enc, index: 1)
        top16PartialsId.bind(enc, index: 2)
        enc.setBytes(&vocabU, length: 4, index: 3)
        enc.setBytes(&topKU, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 256, height: D, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        enc.setComputePipelineState(psoTop16Stage2)
        top16PartialsVal.bind(enc, index: 0)
        top16PartialsId.bind(enc, index: 1)
        unaryOut.bind(enc, index: 2)
        candidatesOut.bind(enc, index: 3)
        enc.setBytes(&topKU, length: 4, index: 4)
        enc.setBytes(&vocabU, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: D, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        if let bw = dspineBridge, let bs = dspineScratch {
            guard let headStorage else {
                throw QWDFlashError.state("QWDFlashDrafter: DSpine refinement requires headStorage")
            }
            try encodeDSpineRefinementTail(
                enc, draftCount: D, topK: topK, headStorage: headStorage,
                bw: bw, bs: bs, stochastic: stochastic != nil
            )
        } else if stochastic == nil,
                  let hProjOut = hiddenProjOut,
                  let pBook = predecessorCodebook,
                  let sBook = successorCodebook {
            var anchorI = Int32(safeAnchor), draftCountU = UInt32(D), selectorVocabU = UInt32(vocabRows)
            var softcapF = config.finalLogitSoftcapping
            var softcapModeU = config.selectorSoftcapMode.rawValue
            var rankU = UInt32(R)
            enc.setComputePipelineState(psoSelectorGreedy)
            hProjOut.bind(enc, index: 0)
            unaryOut.bind(enc, index: 1)
            candidatesOut.bind(enc, index: 2)
            pBook.bind(enc, index: 3)
            sBook.bind(enc, index: 4)
            draftTokensOut.bind(enc, index: 5)
            enc.setBytes(&anchorI, length: 4, index: 6)
            enc.setBytes(&draftCountU, length: 4, index: 7)
            enc.setBytes(&topKU, length: 4, index: 8)
            enc.setBytes(&selectorVocabU, length: 4, index: 9)
            enc.setBytes(&softcapF, length: 4, index: 10)
            enc.setBytes(&softcapModeU, length: 4, index: 11)
            enc.setBytes(&rankU, length: 4, index: 12)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }

        encoderEnded = true
        enc.endEncoding()
        if let journalCopy {
            guard let blit = cb.makeBlitCommandEncoder() else {
                throw QWDFlashError.state("QWDFlashDrafter: journal copy encoder failed")
            }
            blit.copy(
                from: draftTokensOut.region.buffer,
                sourceOffset: draftTokensOut.region.base + draftTokensOut.offset,
                to: journalCopy.buffer,
                destinationOffset: journalCopy.byteOffset,
                size: D * 4
            )
            blit.endEncoding()
        }
        cb.commit()
        return cb
    }

    static func clampToken(_ token: Int, vocab: Int) -> Int {
        (0..<vocab).contains(token) ? token : 0
    }
}
