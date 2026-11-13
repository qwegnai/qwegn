// MTPVerification+Hybrid.swift — Hybrid shared-last-layer ECHO + suffix/periodic PLD
// speculative proposal and commit-time exact stream pair resolution for QWEngineDriver.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL
import QWKV
import QWMTP

extension QWEngineDriver {
    static let moeSpecExpertBudgetEnv: String? = ProcessInfo.processInfo.environment["QW_MOE_SPEC_EXPERT_BUDGET"]

    /// Returns the active MoE-Spec (arXiv:2602.16052) unique-expert budget across `rows` verification tokens,
    /// or `0` when all routed experts should be evaluated losslessly.
    func activeMoESpecExpertBudget(rows: Int) -> Int {
        guard rows > 1, rows <= 4, geometry.numExperts > 0, geometry.expertsPerTok > 0 else {
            return 0
        }
        if let raw = Self.moeSpecExpertBudgetEnv {
            if raw.lowercased() == "auto" {
                let acceptRate = mtpDraftProposals >= 4
                    ? Double(mtpDraftAccepted) / Double(max(1, mtpDraftProposals))
                    : 0.75
                return QWMTPAdaptiveDepth.moeSpecDraftExpertBudget(
                    geometry: geometry, verifyRows: rows, acceptRate: acceptRate)
            }
            if let explicit = Int(raw), explicit > 0 {
                return QWMTPAdaptiveDepth.moeSpecDraftExpertBudget(
                    geometry: geometry, verifyRows: rows, explicitBudget: explicit)
            }
        }
        if tuning.moeSpecBudget > 0 {
            return QWMTPAdaptiveDepth.moeSpecDraftExpertBudget(
                geometry: geometry, verifyRows: rows, explicitBudget: tuning.moeSpecBudget)
        }
        return 0
    }

    struct EchoRoundProposal {
        let drafts: [Int]
        let proposals: [QWMTPGumbelProposal]
        let draftedPairs: Int
    }

    func proposeEchoOrPLDRound(
        session: QWEngineSession,
        echoDrafter: QWEchoDraftEncoder,
        hidden: QWTensorValue,
        views: (k: QWKVLayerView, v: QWKVLayerView),
        pending: Int,
        count: Int,
        basePairs: Int,
        owner: QWStochasticSession?,
        draftCompleted: ((any MTLCommandBuffer) throws -> Void)?
    ) throws -> EchoRoundProposal {
        let effectiveCount: Int
        if mtpAdaptiveDepth && geometry.numExperts > 0 && mtpDraftProposals >= 8 && count > 1 {
            let trunkBits = layersW.first?.gu.view.bits ?? 8
            let expertBits = layersW.compactMap({ $0.moe?.gateUp.bits }).first ?? trunkBits
            let acceptRate = Double(mtpDraftAccepted) / Double(max(1, mtpDraftProposals))
            effectiveCount = QWMTPAdaptiveDepth.optimalDraftCount(
                geometry: geometry, trunkBits: trunkBits, expertBits: expertBits,
                acceptRate: acceptRate, maxDrafts: count)
        } else {
            effectiveCount = count
        }

        let resolvedStrategy = (try? QWDraftStrategy.resolve(mtpDraftStrategy)) ?? .hybrid
        let fullHistory: [Int]
        if session.tokenHistory.count > session.position, session.tokenHistory.last == pending {
            fullHistory = session.tokenHistory
        } else {
            fullHistory = session.tokenHistory + [pending]
        }
        if owner == nil && !mtpStageProfiling && resolvedStrategy != .mtp {
            if let pd = QWContextCopyDrafter.periodicDrafts(
                history: fullHistory, maxCount: effectiveCount, minRepeats: 3),
               !pd.isEmpty {
                lastRoundUsedPLD = true
                lastEchoDraftedPairs = 0
                mtpDraftProposals += pd.count
                return EchoRoundProposal(drafts: pd, proposals: [], draftedPairs: 0)
            }
            let minMatch = (resolvedStrategy == .ngram) ? 3 : 5
            if let match = QWContextCopyDrafter.drafts(
                history: fullHistory, maxCount: effectiveCount, minMatchLen: minMatch),
               !match.tokens.isEmpty {
                let take = match.matchLen >= 6
                    ? effectiveCount
                    : min(resolvedStrategy == .ngram ? effectiveCount : 2, match.tokens.count)
                let chosen = Array(match.tokens.prefix(max(1, take)))
                lastRoundUsedPLD = true
                lastEchoDraftedPairs = 0
                mtpDraftProposals += chosen.count
                return EchoRoundProposal(drafts: chosen, proposals: [], draftedPairs: 0)
            }
        }

        let neuralCount = (resolvedStrategy == .mtp || mtpStageProfiling || owner != nil)
            ? effectiveCount : 1
        var proposals: [QWMTPGumbelProposal] = []
        if let owner {
            try owner.stream.beginGumbelRound(drafts: neuralCount, vocabulary: geometry.vocab)
        }
        var draftHistory = fullHistory
        let sampledToken: ((QWTensorValue) throws -> Int)? = owner.map { owner in
            { logits in
                let proposal = try owner.stream.sampleGumbelProposal(
                    owner.distribution(logits: logits, repetitionHistory: draftHistory))
                proposals.append(proposal)
                draftHistory.append(proposal.token)
                return proposal.token
            }
        }
        let needsAdaptiveStop = mtpAdaptiveDepth && (neuralCount > 1 || mtpStageProfiling)
        let stopAfterDraft: ((QWTensorValue, Int) throws -> Bool)? = needsAdaptiveStop ? { logits, completedDrafts in
            if completedDrafts >= neuralCount && !self.mtpStageProfiling {
                return false
            }
            let values = logits.region.pointer.advanced(by: logits.offset).assumingMemoryBound(to: Float.self)
            let entropy = try QWMTPAdaptiveDepth.normalizedEntropy(
                UnsafeBufferPointer(start: values, count: logits.shape.elementCount))
            self.pendingMTPAdaptiveEntropy?.append(entropy)
            return try QWMTPAdaptiveDepth.shouldStop(
                normalizedEntropy: entropy, threshold: self.mtpAdaptiveEntropyThreshold,
                completedDrafts: completedDrafts, maximumDrafts: neuralCount)
        } : nil

        let tDraft0 = Self.decodeTraceEnabled ? Date() : nil
        var drafts = try echoDrafter.propose(
            driver: self, initialStream: hidden, kv: views,
            pendingToken: pending, cacheRows: basePairs, drafts: neuralCount,
            sampledToken: sampledToken, shouldStopAfterDraft: stopAfterDraft,
            writeLogitsOnFinalDraft: mtpStageProfiling, completed: draftCompleted)
        let draftedPairs = drafts.count
        if owner == nil, !mtpStageProfiling, resolvedStrategy != .mtp,
           effectiveCount > drafts.count, let d0 = drafts.first,
           let cont = QWContextCopyDrafter.drafts(
               history: fullHistory, appendedToken: d0,
               maxCount: effectiveCount - drafts.count, minMatchLen: 4),
           !cont.tokens.isEmpty {
            let extra = cont.matchLen >= 6 ? (effectiveCount - drafts.count) : 1
            drafts.append(contentsOf: cont.tokens.prefix(max(1, extra)))
        }
        if let tDraft0 {
            FileHandle.standardError.write(Data(String(
                format: "[echo-draft] pos=%d count=%d wall=%.2fms\n",
                session.position, drafts.count, Date().timeIntervalSince(tDraft0) * 1000).utf8))
        }
        lastEchoDraftedPairs = draftedPairs
        mtpDraftProposals += drafts.count
        return EchoRoundProposal(drafts: drafts, proposals: proposals, draftedPairs: draftedPairs)
    }

    func encodeEchoCommitResolution(
        _ cb: any MTLCommandBuffer,
        session: QWEngineSession,
        echoDrafter: QWEchoDraftEncoder,
        window: QWVerifyWindow,
        trunkHidden: QWTensorValue,
        basePairs: Int,
        retained: Int,
        usedPLD: Bool
    ) throws {
        guard let views = session.mtpKVViews.first, let windowStream = window.mhcStream else {
            throw QWEngineDriverError.state("ECHO commit resolution requires MTP KV views and window mHC stream")
        }
        let streamElems = echoDrafter.streams * geometry.hidden
        let streamBytes = streamElems * 4

        if usedPLD {
            guard retained > 0 else { return }
            let embSlice = try echoDrafter.stageEmbeddings(driver: self, tokens: verifiedInputs[0..<retained])
            let prevSlice: QWTensorValue
            if retained == 1 {
                prevSlice = trunkHidden
            } else {
                guard let prepBlit = cb.makeBlitCommandEncoder() else {
                    throw QWEngineDriverError.state("ECHO PLD commit stage blit encoder")
                }
                prepBlit.copy(from: trunkHidden.region.buffer,
                              sourceOffset: trunkHidden.region.base + trunkHidden.offset,
                              to: echoDrafter.prevStreamBuf.region.buffer,
                              destinationOffset: echoDrafter.prevStreamBuf.region.base + echoDrafter.prevStreamBuf.offset,
                              size: streamBytes)
                prepBlit.copy(from: windowStream.region.buffer,
                              sourceOffset: windowStream.region.base + windowStream.offset,
                              to: echoDrafter.prevStreamBuf.region.buffer,
                              destinationOffset: echoDrafter.prevStreamBuf.region.base + echoDrafter.prevStreamBuf.offset + streamBytes,
                              size: (retained - 1) * streamBytes)
                prepBlit.endEncoding()
                prevSlice = echoDrafter.prevStreamBuf.slice(byteOffset: 0, count: retained * streamElems, dtype: .f32)
            }
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw QWEngineDriverError.state("ECHO PLD commit seed compute encoder")
            }
            do {
                bindResidentResources(enc)
                enc.useResource(echoDrafter.region.buffer, usage: [.read, .write])
                try echoDrafter.encodeSeedChunk(enc, driver: self, prevStream: prevSlice,
                                                embeddings: embSlice, kv: views,
                                                cacheRows: basePairs, rows: retained)
            } catch {
                enc.endEncoding()
                throw error
            }
            enc.endEncoding()
            return
        }

        guard retained >= 2 else { return }
        let extraRows = retained - 1
        let prevSlice = windowStream.slice(byteOffset: 0, count: extraRows * streamElems, dtype: .f32)
        let embSlice = try echoDrafter.stageEmbeddings(driver: self, tokens: verifiedInputs[1..<retained])
        guard let enc = cb.makeComputeCommandEncoder() else {
            throw QWEngineDriverError.state("ECHO verify refresh compute encoder")
        }
        do {
            bindResidentResources(enc)
            enc.useResource(echoDrafter.region.buffer, usage: [.read, .write])
            try echoDrafter.encodeSeedChunk(enc, driver: self, prevStream: prevSlice,
                                            embeddings: embSlice, kv: views,
                                            cacheRows: basePairs + 1, rows: extraRows)
        } catch {
            enc.endEncoding()
            throw error
        }
        enc.endEncoding()
    }

    func finalizeEchoPairCursor(
        session: QWEngineSession,
        mtpCache: QWKVCache,
        mtpHandle: QWKVHandle,
        basePairs: Int,
        draftedPairs: Int,
        retainedPairs: Int
    ) throws {
        let currentWritten = basePairs + draftedPairs
        let targetWritten = basePairs + retainedPairs
        if targetWritten > currentWritten {
            mtpCache.markWritten(mtpHandle, upTo: targetWritten)
        } else if targetWritten < currentWritten {
            try mtpCache.rollback(mtpHandle, by: currentWritten - targetWritten)
            session.mtpKVViews = [try mtpCache.layerViews(mtpHandle, layer: 0)]
        }
    }

    func greedyRoundBoundary(_ session: QWEngineSession) throws -> QWGreedyRoundBoundary {
        guard let mtpCache, let handle = session.mtpKV else {
            throw QWEngineDriverError.state("greedy round pair cache is absent")
        }
        return QWGreedyRoundBoundary(position: session.position, generated: session.generated,
                                     record: session.roundRecord, hiddenValid: session.trunkHiddenValid,
                                     trunkRewind: try kvCache.rollbackGeneration(session.kv),
                                     pairRewind: try mtpCache.rollbackGeneration(handle))
    }

    /// A guarded round publication door, not generation integration. The draft
    /// owner must have materialized N MTP pairs from the saved round boundary.
    func commitGreedyVerification(_ session: QWEngineSession, round: QWPreparedGreedyRound,
                                  drain: Bool = true,
                                  permit: QWGenerationPermit? = nil) throws {
        try commitPreparedVerification(session, round: round, stochastic: false, drain: drain, permit: permit)
    }

    func commitStochasticVerification(_ session: QWEngineSession, round: QWPreparedStochasticRound,
                                      drain: Bool = true,
                                      permit: QWGenerationPermit? = nil) throws {
        try commitPreparedVerification(session, round: round.authorization, stochastic: true, drain: drain, permit: permit)
    }
}
