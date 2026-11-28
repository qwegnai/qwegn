// MTPVerification+Pipeline.swift — Async Commit -> Draft -> Verify single-wait pipeline helpers.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWHead
import QWKV
import QWMTP

extension QWEngineDriver {
    enum AsyncNeuralDraftPlan {
        case pld([Int])
        case asyncNeural(draftCBs: [any MTLCommandBuffer], draftedPairs: Int)
    }

    func waitForPriorCommitCB(_ cb: any MTLCommandBuffer, session: QWEngineSession) throws {
        cb.waitUntilCompleted()
        if cb.status != .completed {
            admission.commandBufferErrors += 1
            session.stateFailed = true
            throw QWEngineDriverError.state("greedy commit GPU failure: \(String(describing: cb.error))")
        }
    }

    func prepareAsyncNeuralDraft(
        session: QWEngineSession,
        hidden: QWTensorValue,
        views: (k: QWKVLayerView, v: QWKVLayerView),
        pending: Int,
        count: Int,
        basePairs: Int,
        mtpCache: QWKVCache,
        mtpHandle: QWKVHandle,
        table: QWMTPEmbeddingTable,
        owner: QWStochasticSession? = nil
    ) throws -> AsyncNeuralDraftPlan? {
        let resolvedStrategy = (try? QWDraftStrategy.resolve(mtpDraftStrategy)) ?? .hybrid
        let (extraPending, previousToken) = pendingHistoryContext(session: session, pending: pending)
        let isStochastic = owner != nil
        let canUseCanonical = owner == nil || owner?.config.isGreedy == true
        if canUseCanonical,
           let chosen = proposeCanonicalOrFilteredDrafts(
               constraint: activeConstraint?.constraint,
               maxCount: count,
               fallbackCandidates: {
                   resolvedStrategy != .mtp ? tryZeroCostContextDrafts(
                       session: session, pending: pending, extraPending: extraPending,
                       previousToken: previousToken, count: count, strategy: resolvedStrategy
                   ) : nil
               }
           ), !chosen.isEmpty {
            return .pld(chosen)
        }
        let neuralCount: Int
        if echoDrafter != nil {
            neuralCount = (resolvedStrategy == .mtp || isStochastic) ? count : 1
        } else {
            let neuralCap = QWAdaptiveDraftPolicy.neuralDepthCap(
                isStochastic: isStochastic, stageProfiling: false,
                hasDeviceEmbedding: true, numExperts: geometry.numExperts)
            var n = (resolvedStrategy == .ngram && !isStochastic) ? 1 : count
            if resolvedStrategy == .hybrid, let neuralCap {
                n = min(n, max(1, neuralCap))
            }
            neuralCount = n
        }
        let hasPostContinuation = !isStochastic && resolvedStrategy != .mtp && count > neuralCount && (
            QWContextCopyDrafter.drafts(
                history: session.tokenHistory, extraToken: extraPending,
                maxCount: 1, minMatchLen: 5) != nil
            || geometry.numExperts == 0 || resolvedStrategy == .ngram
        )
        let maxChain = isStochastic ? 15 : 64
        guard !hasPostContinuation, neuralCount >= 1, neuralCount <= maxChain else {
            return nil
        }
        let stochParams: QWStochasticSelectionParams?
        if let owner, !owner.config.isGreedy {
            let drawBase = try owner.stream.reserveDraws(UInt64(neuralCount))
            stochParams = QWStochasticSelectionParams(
                temperature: owner.config.temperature, topK: owner.config.topK, topP: owner.config.topP,
                repetitionPenalty: 1.0, seed: owner.config.seed, philoxOffsetBase: drawBase)
        } else {
            stochParams = nil
        }
        let ids = table.tokens.contents().assumingMemoryBound(to: Int32.self)
        ids[0] = Int32(pending)
        let draftCBs: [any MTLCommandBuffer]
        if let echoDrafter {
            draftCBs = try echoDrafter.enqueueSingleCB(
                driver: self, table: table, initialStream: hidden, kv: views,
                pendingToken: pending, cacheRows: basePairs, drafts: neuralCount,
                writeLogitsOnFinalDraft: false, journalTokenOffset: 1, stochastic: stochParams)
        } else if let mtpDraftEncoder {
            draftCBs = try QWMTPDraftChain.enqueueDeviceChain(
                hal: hal, draft: mtpDraftEncoder, headStorage: headStorage,
                draftHead: mtpDraftHead, hidden: hidden, kv: views,
                pendingToken: pending, drafts: neuralCount, position: session.position,
                cacheRows: basePairs, mropeDelta: session.roundRecord.mropeDelta,
                table: table, journalTokenOffset: 1, stochastic: stochParams)
        } else {
            return nil
        }
        mtpCache.markWritten(mtpHandle, upTo: basePairs + neuralCount)
        return .asyncNeural(draftCBs: draftCBs, draftedPairs: neuralCount)
    }

    func tryZeroCostContextDrafts(
        session: QWEngineSession,
        pending: Int,
        extraPending: Int?,
        previousToken: Int?,
        count: Int,
        strategy: QWDraftStrategy
    ) -> [Int]? {
        if let pd = QWContextCopyDrafter.periodicDrafts(
            history: session.tokenHistory, extraToken: extraPending,
            maxCount: count, minRepeats: 3),
           !pd.isEmpty {
            return pd
        }
        let isMoE = geometry.numExperts > 0
        let minMatch = (strategy == .ngram) ? 3 : (isMoE ? 6 : 5)
        if let match = QWContextCopyDrafter.drafts(
            history: session.tokenHistory, extraToken: extraPending,
            maxCount: count, minMatchLen: minMatch),
           !match.tokens.isEmpty {
            let take = match.matchLen >= 6
                ? count
                : min(strategy == .ngram ? count : 2, match.tokens.count)
            var out = Array(match.tokens.prefix(max(1, take)))
            if strategy == .ngram, out.count < count {
                session.tokenRecycler.appendDrafts(
                    into: &out, maxCount: count,
                    previousToken: previousToken, pendingToken: pending,
                    maxUnigramFallbacks: isMoE ? 0 : 2)
            }
            return out
        }
        if strategy == .ngram {
            var recycled: [Int] = []
            recycled.reserveCapacity(count)
            session.tokenRecycler.appendDrafts(
                into: &recycled, maxCount: count,
                previousToken: previousToken, pendingToken: pending,
                maxUnigramFallbacks: isMoE ? 0 : 2)
            if !recycled.isEmpty { return recycled }
        }
        return nil
    }

    func executeAsyncNeuralRound(
        _ draftPlan: AsyncNeuralDraftPlan,
        session: QWEngineSession,
        verifyWindow: QWVerifyWindow,
        pending: Int,
        table: QWMTPEmbeddingTable,
        priorCommitCB: (any MTLCommandBuffer)?,
        isUnconstrained: Bool,
        owner: QWStochasticSession?
    ) throws -> (
        window: QWVerifyWindow,
        drafts: [Int],
        draftProposals: [QWSpeculativeRoundResolver.DraftProposal],
        gpuVerifyResolution: QWGPUVerifyResolution?,
        draftedPairs: Int
    ) {
        switch draftPlan {
        case .pld(let chosen):
            let window = try verifyTokensCore(
                session, explicitTokens: [pending] + chosen, rowCount: 1 + chosen.count,
                pendingToken: nil, useDeviceEmbed: table, priorCommitCB: priorCommitCB,
                inFlightDraftCBs: []).window
            lastRoundUsedPLD = true
            lastDraftedPairs = 0
            mtpDraftProposals += chosen.count
            return (window, chosen, owner != nil ? chosen.map { .greedyToken($0) } : [], nil, 0)
        case .asyncNeural(let draftCBs, let neuralCount):
            let draftSelector = echoDrafter?.selector ?? mtpDraftEncoder?.scratch.selector
            let gpuVerifyEligible = owner != nil && !(owner?.config.isGreedy ?? true)
                && isUnconstrained && owner?.config.repetitionPenalty == 1.0
                && effectiveRelaxedMargin == 0
                && !mtpStageProfiling && !mtpVerificationProfiling
                && !(owner!.config.topP >= 1.0 && owner!.config.topK >= geometry.vocab)
                && draftSelector != nil
            if gpuVerifyEligible, let owner, let draftSelector {
                let lcWin = try owner.stream.beginLCRound(drafts: neuralCount, maxDrafts: 15)
                verifyWindow.stochasticVerifyBox.pending = QWVerifyWindow.PendingStochasticVerify(
                    params: QWStochasticSelectionParams(
                        temperature: owner.config.temperature, topK: owner.config.topK, topP: owner.config.topP,
                        repetitionPenalty: 1.0, seed: owner.config.seed, philoxOffsetBase: lcWin.start),
                    draftProposals: draftSelector, draftProposalSlotBase: 0,
                    repetitionHistoryByRow: [], draftCount: neuralCount)
            }
            defer { verifyWindow.stochasticVerifyBox.pending = nil }
            let (verifiedWindow, resolvedTokens) = try verifyTokensCore(
                session, explicitTokens: nil, rowCount: 1 + neuralCount,
                pendingToken: pending, useDeviceEmbed: table,
                priorCommitCB: priorCommitCB, inFlightDraftCBs: draftCBs)
            let drafts = Array(resolvedTokens.dropFirst())
            lastRoundUsedPLD = false
            lastDraftedPairs = neuralCount
            mtpDraftProposals += drafts.count
            let gpuRes = gpuVerifyEligible
                ? try verifyWindow.targetSelector?.readGPUVerifyResolution(draftCount: neuralCount)
                : nil
            let proposals: [QWSpeculativeRoundResolver.DraftProposal]
            if let owner {
                if gpuRes?.resolved == true || owner.config.isGreedy {
                    proposals = drafts.map { .greedyToken($0) }
                } else {
                    proposals = try readAsyncDraftProposals(drafts: neuralCount)
                }
            } else {
                proposals = []
            }
            return (verifiedWindow, drafts, proposals, gpuRes, neuralCount)
        }
    }

    func readAsyncDraftProposals(drafts: Int) throws -> [QWSpeculativeRoundResolver.DraftProposal] {
        if let echoDrafter { return try echoDrafter.readSparseProposals(drafts: drafts) }
        if let mtpDraftEncoder { return try QWMTPDraftChain.readSparseProposals(draft: mtpDraftEncoder, drafts: drafts) }
        throw QWEngineDriverError.state("async stochastic draft proposals require ECHO or NextN encoder")
    }

    func resolveStochasticViaUnifiedResolver(
        window: QWVerifyWindow, session: QWEngineSession, owner: QWStochasticSession,
        pending: Int, drafts: [Int], proposals: [QWSpeculativeRoundResolver.DraftProposal],
        gpuResolution: QWGPUVerifyResolution?, maxNew: Int, stopIDs: Set<Int>,
        constraint: (any QWTokenConstraint)?, activeSampler: QWSampler?,
        margin: Float, minProb: Float
    ) throws -> (decision: QWGreedyRoundDecision, witness: QWStochasticRoundWitness) {
        let p = window.logits.region.pointer.advanced(by: window.logits.offset).assumingMemoryBound(to: Float.self)
        let ctx = pendingHistoryContext(session: session, pending: pending)
        var extraHistory: [Int] = []
        extraHistory.reserveCapacity(1 + drafts.count)
        let useRepPenalty = owner.config.repetitionPenalty != 1.0
        let resolution = try QWSpeculativeRoundResolver.resolve(
            draftProposals: proposals,
            gpuResolution: gpuResolution,
            stream: owner.stream,
            isGreedy: owner.config.isGreedy,
            maxNew: maxNew,
            stopIDs: stopIDs,
            constraint: constraint,
            margin: margin,
            minProbability: minProb,
            targetDistribution: { row, draftPrefix, probe in
                let repHistory: [Int]
                if useRepPenalty {
                    extraHistory.removeAll(keepingCapacity: true)
                    if let extra = ctx.extraPending { extraHistory.append(extra) }
                    extraHistory.append(contentsOf: draftPrefix)
                    repHistory = session.activeRepetitionHistory(extraTokens: extraHistory)
                } else {
                    repHistory = []
                }
                if let activeSampler, let probe, try probe.writeAllowedTokens(into: &activeSampler.allowedScratch) {
                    let rowBuf = UnsafeBufferPointer(start: p + row * geometry.vocab, count: geometry.vocab)
                    return try activeSampler.constrainedDistributionFromScratch(
                        logits: rowBuf, repetitionHistory: repHistory)
                }
                return try owner.distribution(logits: window.logitRow(row), repetitionHistory: repHistory)
            })
        let decision = QWGreedyRoundDecision(
            acceptedDrafts: resolution.acceptedDrafts,
            tokens: resolution.tokens,
            stopped: resolution.stopped,
            stoppedToken: resolution.stoppedToken)
        return (decision, QWStochasticRoundWitness(owner: owner, drawEnd: owner.stream.drawsConsumed))
    }
}
