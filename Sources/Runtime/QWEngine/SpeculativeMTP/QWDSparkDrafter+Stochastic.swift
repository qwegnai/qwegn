// QWDSparkDrafter+Stochastic.swift — Stochastic proposal and QWSpeculativeRoundResolver adapters for DSpark.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWHead
import QWHAL
import QWMTP

extension QWDSparkDrafter {
    func proposeStochastic(
        driver: QWEngineDriver,
        session: QWEngineSession,
        slot: SessionSlot,
        pos0: Int,
        anchorToken: Int,
        requestedDrafts: Int,
        owner: QWStochasticSession,
        copyEmbedRow: ((Int, QWTensorValue) throws -> Void)? = nil,
        completed: ((any MTLCommandBuffer) throws -> Void)? = nil
    ) throws -> (drafts: [Int], proposals: [QWMTPGumbelProposal]) {
        let draftLimit = min(requestedDrafts, config.maxDrafts)
        guard draftLimit >= 1 else { return ([], []) }

        let (dists, drafts, confProbs) = try encodeAndSampleStochasticSteps(
            driver: driver, session: session, slot: slot, pos0: pos0,
            anchorToken: anchorToken, draftLimit: draftLimit,
            owner: owner, copyEmbedRow: copyEmbedRow, completed: completed
        )
        let kStar = (schedulerEnabled && confProbs.count == drafts.count)
            ? confidenceScheduler.optimalDraftCount(confidences: confProbs, fallbackMax: drafts.count)
            : drafts.count
        lastTailConfidence = (schedulerEnabled && kStar > 0 && kStar <= confProbs.count)
            ? confProbs[kStar - 1]
            : nil
        try owner.stream.beginGumbelRound(drafts: kStar, vocabulary: weights.vocab, maxDrafts: 15)
        var proposals: [QWMTPGumbelProposal] = []
        proposals.reserveCapacity(kStar)
        for dist in dists.prefix(kStar) {
            proposals.append(try owner.stream.sampleGumbelProposal(dist))
        }
        return (Array(drafts.prefix(kStar)), proposals)
    }

    private func encodeAndSampleStochasticSteps(
        driver: QWEngineDriver,
        session: QWEngineSession,
        slot: SessionSlot,
        pos0: Int,
        anchorToken: Int,
        draftLimit: Int,
        owner: QWStochasticSession,
        copyEmbedRow: ((Int, QWTensorValue) throws -> Void)?,
        completed: ((any MTLCommandBuffer) throws -> Void)?
    ) throws -> (dists: [QWSamplingDistribution], drafts: [Int], confProbs: [Float]) {
        let B = config.blockSize
        let pPtr = promptToksBuf.contents().assumingMemoryBound(to: Int32.self)
        let seqPtr = tokSeqBuf.contents().assumingMemoryBound(to: Int32.self)
        withExtendedLifetime(promptToksBuf) {
            withExtendedLifetime(tokSeqBuf) {
                pPtr[0] = Int32(anchorToken)
                seqPtr[0] = Int32(anchorToken)
                for i in 1..<B { pPtr[i] = Int32(config.noiseTokenId) }
            }
        }
        if driver.mtpDeviceEmbedding == nil, let copyEmbedRow {
            for r in 0..<B {
                let rowVal = blkX.slice(byteOffset: r * weights.hidden * 2, count: weights.hidden, dtype: .f16)
                try copyEmbedRow(Int(pPtr[r]), rowVal)
            }
        }

        let headRows = (config.sampleOffset == 0) ? draftLimit : B
        guard let blockCB = hal.makeUnretainedCommandBuffer() else {
            throw QWEngineDriverError.state("DSpark stochastic block CB failed")
        }
        var enc = try makeProposeEncoder(blockCB)
        do {
            try encodeBlockAndBaseLogits(&enc, cb: blockCB, driver: driver, slot: slot, pos0: pos0,
                                         anchorToken: anchorToken, headRows: headRows)
        } catch {
            enc.endEncoding()
            throw error
        }
        enc.endEncoding()
        blockCB.commit()
        blockCB.waitUntilCompleted()
        guard blockCB.status == .completed else {
            throw QWEngineDriverError.state("DSpark block forward GPU failure: \(String(describing: blockCB.error))")
        }
        try completed?(blockCB)

        let previewStream = QWMTPRandomStream(cloning: try owner.stream.frozenState())
        try previewStream.beginGumbelRound(drafts: draftLimit, vocabulary: weights.vocab, maxDrafts: 15)
        var drafts: [Int] = []
        var dists: [QWSamplingDistribution] = []
        var confProbs: [Float] = []
        drafts.reserveCapacity(draftLimit)
        dists.reserveCapacity(draftLimit)
        confProbs.reserveCapacity(draftLimit)

        let sampleOff = config.sampleOffset
        let baseSlice = baseLogits.slice(
            byteOffset: sampleOff * weights.vocab * 4,
            count: draftLimit * weights.vocab,
            dtype: .f32
        )
        let confPtr = confOut.region.buffer.contents().advanced(by: confOut.region.base + confOut.offset)
            .assumingMemoryBound(to: Float.self)

        for j in 0..<draftLimit {
            guard let stepCB = hal.makeUnretainedCommandBuffer(),
                  let stepEnc = stepCB.makeComputeCommandEncoder() else {
                throw QWEngineDriverError.state("DSpark Markov stochastic step encoder failed")
            }
            hal.touchKeepAlive(stepEnc)
            do {
                useResources(on: stepEnc)
                encodeMarkovLogitsStep(stepEnc, baseLogitsSlice: baseSlice, stepIdx: j)
                stepEnc.memoryBarrier(scope: .buffers)
                if let mm = driver.multimaxParams {
                    try driver.head.encodeMultiMax(stepEnc, logits: stepLogits, params: mm, count: weights.vocab)
                }
                if schedulerEnabled, weights.confidenceHead != nil {
                    encodeConfidenceDot(stepEnc, stepIdx: j, hiddenRowIdx: sampleOff + j)
                }
            } catch {
                stepEnc.endEncoding()
                throw error
            }
            stepEnc.endEncoding()
            stepCB.commit()
            stepCB.waitUntilCompleted()
            guard stepCB.status == .completed else {
                throw QWEngineDriverError.state("DSpark Markov step GPU failure: \(String(describing: stepCB.error))")
            }
            try completed?(stepCB)

            let draftHistory = session.activeRepetitionHistory(extraTokens: drafts)
            let dist = try owner.distribution(logits: stepLogits, repetitionHistory: draftHistory)
            let sampled = try previewStream.sampleGumbelProposal(dist)
            drafts.append(sampled.token)
            dists.append(dist)
            seqPtr[j + 1] = Int32(sampled.token)
            if schedulerEnabled, weights.confidenceHead != nil {
                confProbs.append(confPtr[j])
            }
        }
        return (dists, drafts, confProbs)
    }
}
