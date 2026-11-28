// MTPVerification+Hybrid.swift — Hybrid shared-last-layer ECHO + suffix/periodic PLD
// speculative proposal and commit-time exact stream pair resolution for QWEngineDriver.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWHead
import QWKV
import QWMTP

extension QWEngineDriver {
    static let moeSpecExpertBudgetEnv: String? = ProcessInfo.processInfo.environment["QW_MOE_SPEC_EXPERT_BUDGET"]

    /// Returns the active MoE-Spec (arXiv:2602.16052) unique-expert budget across `rows` verification tokens,
    /// or `0` when all routed experts should be evaluated losslessly.
    func activeMoESpecExpertBudget(rows: Int) -> Int {
        guard rows > 1, rows <= 16, geometry.numExperts > 0, geometry.expertsPerTok > 0 else {
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

    struct MTPRoundProposal {
        let drafts: [Int]
        let proposals: [QWMTPGumbelProposal]
        let draftProposals: [QWSpeculativeRoundResolver.DraftProposal]
        let draftedPairs: Int

        init(
            drafts: [Int],
            proposals: [QWMTPGumbelProposal] = [],
            draftProposals: [QWSpeculativeRoundResolver.DraftProposal] = [],
            draftedPairs: Int
        ) {
            self.drafts = drafts
            self.proposals = proposals
            self.draftProposals = draftProposals
            self.draftedPairs = draftedPairs
        }
    }

    func pendingHistoryContext(session: QWEngineSession, pending: Int) -> (extraPending: Int?, previousToken: Int?) {
        let inHistory = session.tokenHistory.count > session.position && session.tokenHistory.last == pending
        if inHistory {
            let prev = session.tokenHistory.count >= 2 ? session.tokenHistory[session.tokenHistory.count - 2] : nil
            return (nil, prev)
        }
        return (pending, session.tokenHistory.last)
    }

    private func adaptiveDraftCount(count: Int, defaultTrunkBits: Int) -> Int {
        guard mtpAdaptiveDepth && geometry.numExperts > 0 && mtpDraftProposals >= 8 && count > 1 else {
            return count
        }
        let trunkBits = layersW.first?.gu.view.bits ?? defaultTrunkBits
        let expertBits = layersW.compactMap({ $0.moe?.gateUp.bits }).first ?? trunkBits
        let acceptRate = Double(mtpDraftAccepted) / Double(max(1, mtpDraftProposals))
        return QWAdaptiveDraftPolicy.optimalDraftCount(
            geometry: geometry, trunkBits: trunkBits, expertBits: expertBits,
            acceptRate: acceptRate, maxDrafts: count)
    }

    private func selectDFlashPLDCandidates(
        session: QWEngineSession,
        pending: Int,
        count: Int,
        resolvedStrategy: QWDraftStrategy
    ) -> [Int] {
        let ctx = pendingHistoryContext(session: session, pending: pending)
        var cand = QWContextCopyDrafter.selectHybridPLDDrafts(
            history: session.tokenHistory,
            extraToken: ctx.extraPending,
            previousToken: ctx.previousToken,
            pendingToken: pending,
            mode: resolvedStrategy,
            maxDrafts: count,
            recycler: session.tokenRecycler
        )
        if cand.count > 7 && cand.count < 12 {
            cand = Array(cand.prefix(7))
        }
        return cand
    }

    private func encodeDFlashHeadLogits(
        _ enc: any MTLComputeCommandEncoder,
        xRot16: QWTensorValue,
        xNorm16: QWTensorValue,
        yLogits32: QWTensorValue,
        rows: Int,
        vocabRows: Int
    ) throws {
        if let draftHead = mtpDraftHead {
            if rows == 1 {
                try draftHead.encode(enc, hal: hal, x: xRot16, y: yLogits32, vocabRows: vocabRows)
            } else {
                try draftHead.encodeBatched(enc, hal: hal, x: xRot16, y: yLogits32, m: rows, vocabRows: vocabRows)
            }
            return
        }
        try QWDFlashDrafter.encodeSlicedHeadLogits(
            enc, hal: hal, headStorage: headStorage, geometry: geometry,
            xRot16: xRot16, xNorm16: xNorm16, yLogits32: yLogits32, rows: rows, vocabRows: vocabRows
        )
    }

    private func dflashAnchorEmbed(
        _ dflashDrafter: QWDFlashDrafter, table: QWMTPEmbeddingTable
    ) -> (any MTLComputeCommandEncoder, QWTensorValue, Int, Int, Int) throws -> Void {
        { enc, propX32, anchorToken, maskToken, rows in
            dflashDrafter.encodeStagePropBF16(
                enc, tableBuffer: table.buffer, byteOffset: table.byteOffset,
                propX: propX32, anchorToken: anchorToken, maskToken: maskToken, rows: rows)
        }
    }

    func prepareAsyncDFlashOrPLDRound(
        session: QWEngineSession, dflashDrafter: QWDFlashDrafter, dSlot: QWDFlashSessionSlot,
        pending: Int, count: Int, table: QWMTPEmbeddingTable
    ) throws -> AsyncNeuralDraftPlan {
        let resolvedStrategy = (try? QWDraftStrategy.resolve(mtpDraftStrategy)) ?? .hybrid
        if let cand = proposeCanonicalOrFilteredDrafts(
            constraint: activeConstraint?.constraint, maxCount: count,
            fallbackCandidates: {
                guard resolvedStrategy != .mtp else { return nil }
                let raw = selectDFlashPLDCandidates(
                    session: session, pending: pending, count: count, resolvedStrategy: resolvedStrategy)
                return raw.isEmpty ? nil : raw
            }), !cand.isEmpty {
            session.dflashRoundStartNs = nil
            return .pld(cand)
        }
        let canAdapt = resolvedStrategy != .mtp
        let activeCount = canAdapt ? session.dflashPolicy.selectDraftCount(maxDrafts: count) : count
        session.dflashRoundStartNs = canAdapt ? DispatchTime.now().uptimeNanoseconds : nil
        table.tokens.contents().assumingMemoryBound(to: Int32.self)[0] = Int32(pending)
        let draftCB: any MTLCommandBuffer
        do {
            draftCB = try dflashDrafter.enqueueGreedyPropose(
                hal: hal, slot: dSlot, cacheOffset: session.dflashCacheOffset,
                pendingToken: pending, draftCount: activeCount, headStorage: headStorage,
                encodeHeadLogits: { enc, xRot16, xNorm16, _, yLogits32, rows, vocabRows in
                    try self.encodeDFlashHeadLogits(
                        enc, xRot16: xRot16, xNorm16: xNorm16, yLogits32: yLogits32, rows: rows, vocabRows: vocabRows)
                },
                encodeAnchorMaskEmbed: dflashAnchorEmbed(dflashDrafter, table: table),
                journalBuffer: table.tokens, journalByteOffset: 4)
        } catch {
            admission.commandBufferErrors += 1; throw error
        }
        return .asyncNeural(draftCBs: [draftCB], draftedPairs: activeCount)
    }

    func proposeDFlashOrPLDRound(
        session: QWEngineSession, dflashDrafter: QWDFlashDrafter, dSlot: QWDFlashSessionSlot,
        pending: Int, count: Int, owner: QWStochasticSession?,
        draftCompleted: ((any MTLCommandBuffer) throws -> Void)?
    ) throws -> MTPRoundProposal {
        let resolvedStrategy = (try? QWDraftStrategy.resolve(mtpDraftStrategy)) ?? .hybrid
        let canUseCanonical = (owner == nil || owner?.config.isGreedy == true) && !mtpStageProfiling
        let canAdapt = !mtpStageProfiling && resolvedStrategy != .mtp
        if canUseCanonical,
           let cand = proposeCanonicalOrFilteredDrafts(
               constraint: activeConstraint?.constraint, maxCount: count,
               fallbackCandidates: {
                   (resolvedStrategy != .mtp) ? selectDFlashPLDCandidates(
                       session: session, pending: pending, count: count, resolvedStrategy: resolvedStrategy) : nil
               }), !cand.isEmpty {
            lastRoundUsedPLD = true; mtpDraftProposals += cand.count; session.dflashRoundStartNs = nil
            return MTPRoundProposal(drafts: cand, proposals: [], draftedPairs: cand.count)
        }
        let activeCount = canAdapt ? session.dflashPolicy.selectDraftCount(maxDrafts: count) : count
        session.dflashRoundStartNs = canAdapt ? DispatchTime.now().uptimeNanoseconds : nil
        if let owner { try owner.stream.beginGumbelRound(drafts: activeCount, vocabulary: geometry.vocab, maxDrafts: 15) }
        let proposed: (drafts: [Int], proposals: [QWMTPGumbelProposal])
        do {
            proposed = try dflashDrafter.propose(
                hal: hal, slot: dSlot, cacheOffset: session.dflashCacheOffset,
                pendingToken: pending, draftCount: activeCount, headStorage: headStorage,
                encodeHeadLogits: { enc, xRot16, xNorm16, _, yLogits32, rows, vocabRows in
                    try self.encodeDFlashHeadLogits(
                        enc, xRot16: xRot16, xNorm16: xNorm16, yLogits32: yLogits32, rows: rows, vocabRows: vocabRows)
                },
                encodeAnchorMaskEmbed: mtpDeviceEmbedding.map { dflashAnchorEmbed(dflashDrafter, table: $0) },
                embedding: { try self.copyEmbedRow($0, to: $1) },
                stochastic: owner.map { ($0.config.isGreedy ? Float(1e-6) : Float($0.config.temperature), $0.stream) },
                completed: draftCompleted)
        } catch {
            admission.commandBufferErrors += 1; throw error
        }
        mtpDraftProposals += proposed.drafts.count
        return MTPRoundProposal(drafts: proposed.drafts, proposals: proposed.proposals, draftedPairs: proposed.drafts.count)
    }

    var isWeightFreeSpeculation: Bool {
        verifyWindow != nil && mtpDraftEncoder == nil && echoDrafter == nil && dflashDrafter == nil && dsparkDrafter == nil
    }

    func proposeWeightFreeDrafts(session: QWEngineSession, pending: Int, count: Int) -> [Int]? {
        guard count > 0 else { return nil }
        let (extraPending, previousToken) = pendingHistoryContext(session: session, pending: pending)
        if let pd = QWContextCopyDrafter.periodicDrafts(
            history: session.tokenHistory, extraToken: extraPending,
            maxCount: count, minRepeats: 3),
           !pd.isEmpty {
            return Array(pd.prefix(count))
        }
        if let match = QWContextCopyDrafter.drafts(
            history: session.tokenHistory, extraToken: extraPending,
            maxCount: count, minMatchLen: 3),
           !match.tokens.isEmpty {
            let take = match.matchLen >= 5 ? count : min(3, match.tokens.count)
            var out = Array(match.tokens.prefix(max(1, take)))
            if out.count < count {
                session.tokenRecycler.appendDrafts(
                    into: &out, maxCount: count,
                    previousToken: previousToken, pendingToken: pending,
                    maxUnigramFallbacks: 0)
            }
            return out
        }
        var recycled: [Int] = []
        recycled.reserveCapacity(count)
        session.tokenRecycler.appendDrafts(
            into: &recycled, maxCount: count,
            previousToken: previousToken, pendingToken: pending,
            maxUnigramFallbacks: 0)
        return recycled.isEmpty ? nil : Array(recycled.prefix(count))
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
    ) throws -> MTPRoundProposal {
        try proposeHybridRound(
            session: session, pending: pending, count: count,
            defaultTrunkBits: 8, owner: owner, traceTag: "[echo-draft]",
            useFullNeuralDepth: { $0 == .mtp || mtpStageProfiling || owner != nil },
            proposeNeural: { neuralCount, sampledToken, stopAfterDraft in
                if let owner, !owner.config.isGreedy, !mtpStageProfiling, !mtpAdaptiveDepth,
                   mtpDeviceEmbedding != nil {
                    let drawBase = try owner.stream.reserveDraws(UInt64(neuralCount))
                    let params = QWStochasticSelectionParams(
                        temperature: owner.config.temperature, topK: owner.config.topK, topP: owner.config.topP,
                        repetitionPenalty: 1.0, seed: owner.config.seed, philoxOffsetBase: drawBase)
                    return try echoDrafter.proposeStochastic(
                        driver: self, initialStream: hidden, kv: views,
                        pendingToken: pending, cacheRows: basePairs, drafts: neuralCount,
                        params: params, writeLogitsOnFinalDraft: false, completed: draftCompleted)
                }
                let toks = try echoDrafter.propose(
                    driver: self, initialStream: hidden, kv: views,
                    pendingToken: pending, cacheRows: basePairs, drafts: neuralCount,
                    sampledToken: sampledToken, shouldStopAfterDraft: stopAfterDraft,
                    writeLogitsOnFinalDraft: mtpStageProfiling, completed: draftCompleted)
                return (toks, [])
            })
    }

    func proposeNextNOrPLDRound(
        session: QWEngineSession,
        draftEncoder: QWMTPDraftEncoder,
        hidden: QWTensorValue,
        views: (k: QWKVLayerView, v: QWKVLayerView),
        pending: Int,
        count: Int,
        basePairs: Int,
        owner: QWStochasticSession?,
        draftCompleted: ((any MTLCommandBuffer) throws -> Void)?
    ) throws -> MTPRoundProposal {
        let neuralCap = QWAdaptiveDraftPolicy.neuralDepthCap(
            isStochastic: owner != nil, stageProfiling: mtpStageProfiling,
            hasDeviceEmbedding: mtpDeviceEmbedding != nil, numExperts: geometry.numExperts)
        return try proposeHybridRound(
            session: session, pending: pending, count: count,
            defaultTrunkBits: 4, owner: owner, traceTag: "[draft]",
            maxHybridNeuralDepth: neuralCap,
            useFullNeuralDepth: { !($0 == .ngram && owner == nil && !mtpStageProfiling) },
            proposeNeural: { neuralCount, sampledToken, stopAfterDraft in
                if let owner, !owner.config.isGreedy, !mtpStageProfiling, !mtpAdaptiveDepth,
                   let table = mtpDeviceEmbedding {
                    let drawBase = try owner.stream.reserveDraws(UInt64(neuralCount))
                    let params = QWStochasticSelectionParams(
                        temperature: owner.config.temperature, topK: owner.config.topK, topP: owner.config.topP,
                        repetitionPenalty: 1.0, seed: owner.config.seed, philoxOffsetBase: drawBase)
                    let res = try QWMTPDraftChain.runStochastic(
                        hal: hal, draft: draftEncoder, headStorage: headStorage,
                        draftHead: mtpDraftHead, hidden: hidden, kv: views,
                        pendingToken: pending, drafts: neuralCount, position: session.position,
                        cacheRows: basePairs, mropeDelta: session.roundRecord.mropeDelta,
                        deviceEmbedding: table, params: params,
                        completed: draftCompleted, gpuFailure: { self.commandBufferErrors += 1 })
                    return (res.tokens, res.proposals)
                }
                let devEmb = (owner != nil && !(owner?.config.isGreedy ?? false)) ? nil : mtpDeviceEmbedding
                let toks = try QWMTPDraftChain.run(
                    hal: hal, draft: draftEncoder, headStorage: headStorage,
                    draftHead: mtpDraftHead, hidden: hidden, kv: views,
                    pendingToken: pending, drafts: neuralCount, position: session.position, cacheRows: basePairs,
                    mropeDelta: session.roundRecord.mropeDelta,
                    selection: .gpuChecked, deviceEmbedding: devEmb,
                    embedding: { try self.copyEmbedRow($0, to: $1) },
                    sampledToken: sampledToken, shouldStopAfterDraft: stopAfterDraft,
                    writeLogitsOnFinalDraft: mtpStageProfiling,
                    completed: draftCompleted, gpuFailure: { self.commandBufferErrors += 1 })
                return (toks, [])
            })
    }

    private func proposeHybridRound(
        session: QWEngineSession,
        pending: Int,
        count: Int,
        defaultTrunkBits: Int,
        owner: QWStochasticSession?,
        traceTag: String,
        maxHybridNeuralDepth: Int? = nil,
        useFullNeuralDepth: (QWDraftStrategy) -> Bool,
        proposeNeural: (
            _ neuralCount: Int,
            _ sampledToken: ((QWTensorValue) throws -> Int)?,
            _ stopAfterDraft: ((QWTensorValue, Int) throws -> Bool)?
        ) throws -> (drafts: [Int], proposals: [QWSpeculativeRoundResolver.DraftProposal])
    ) throws -> MTPRoundProposal {
        let effectiveCount = adaptiveDraftCount(count: count, defaultTrunkBits: defaultTrunkBits)
        let resolvedStrategy = (try? QWDraftStrategy.resolve(mtpDraftStrategy)) ?? .hybrid
        let (extraPending, previousToken) = pendingHistoryContext(session: session, pending: pending)
        let canUseCanonical = (owner == nil || owner?.config.isGreedy == true) && !mtpStageProfiling
        let canUseContextPLD = canUseCanonical && resolvedStrategy != .mtp
        if canUseCanonical,
           let chosen = proposeCanonicalOrFilteredDrafts(
               constraint: activeConstraint?.constraint,
               maxCount: count,
               fallbackCandidates: {
                   canUseContextPLD ? tryZeroCostContextDrafts(
                       session: session, pending: pending, extraPending: extraPending,
                       previousToken: previousToken, count: count, strategy: resolvedStrategy
                   ) : nil
               }),
           !chosen.isEmpty {
            lastRoundUsedPLD = true
            lastDraftedPairs = 0
            mtpDraftProposals += chosen.count
            return MTPRoundProposal(
                drafts: chosen,
                draftProposals: owner != nil ? chosen.map { .greedyToken($0) } : [],
                draftedPairs: 0)
        }

        var neuralCount = useFullNeuralDepth(resolvedStrategy) ? effectiveCount : 1
        if resolvedStrategy == .hybrid, !mtpStageProfiling, let maxHybridNeuralDepth {
            neuralCount = min(neuralCount, max(1, maxHybridNeuralDepth))
        }
        var extraDrafts: [Int] = extraPending.map { [$0] } ?? []
        var fallbackProposals: [QWSpeculativeRoundResolver.DraftProposal] = []
        let sampledToken: ((QWTensorValue) throws -> Int)? = owner.flatMap { owner in
            guard !owner.config.isGreedy else { return nil }
            return { logits in
                let draftHistory = owner.config.repetitionPenalty == 1.0
                    ? [] : session.activeRepetitionHistory(extraTokens: extraDrafts)
                let dist = try owner.distribution(logits: logits, repetitionHistory: draftHistory)
                let tok = try dist.sampleToken(uniform: try owner.stream.nextUniformUpper24(), version: .upper24)
                fallbackProposals.append(.distribution(token: tok, distribution: dist))
                extraDrafts.append(tok)
                return tok
            }
        }
        let needsAdaptiveStop = mtpAdaptiveDepth && owner == nil && (neuralCount > 1 || mtpStageProfiling)
        let stopAfterDraft: ((QWTensorValue, Int) throws -> Bool)? = needsAdaptiveStop ? { logits, completedDrafts in
            if completedDrafts >= neuralCount && !self.mtpStageProfiling { return false }
            let values = logits.region.pointer.advanced(by: logits.offset).assumingMemoryBound(to: Float.self)
            let entropy = try QWMTPAdaptiveDepth.normalizedEntropy(
                UnsafeBufferPointer(start: values, count: logits.shape.elementCount))
            self.pendingMTPAdaptiveEntropy?.append(entropy)
            return try QWMTPAdaptiveDepth.shouldStop(
                normalizedEntropy: entropy, threshold: self.mtpAdaptiveEntropyThreshold,
                completedDrafts: completedDrafts, maximumDrafts: neuralCount)
        } : nil

        let tDraft0 = Self.decodeTraceEnabled ? Date() : nil
        let result = try proposeNeural(neuralCount, sampledToken, stopAfterDraft)
        var drafts = result.drafts
        var draftProposals = result.proposals.isEmpty ? fallbackProposals : result.proposals
        if owner?.config.isGreedy == true && draftProposals.isEmpty {
            draftProposals = drafts.map { .greedyToken($0) }
        }
        let draftedPairs = drafts.count
        let isMoE = geometry.numExperts > 0
        if owner == nil, !mtpStageProfiling, resolvedStrategy != .mtp, count > drafts.count {
            let contMinMatch = (isMoE && resolvedStrategy != .ngram) ? 6 : 4
            if let cont = QWContextCopyDrafter.drafts(
                history: session.tokenHistory, extraToken: extraPending, trailingTokens: drafts,
                maxCount: count - drafts.count, minMatchLen: contMinMatch),
               !cont.tokens.isEmpty {
                let extra = cont.matchLen >= 6 ? (count - drafts.count) : 1
                drafts.append(contentsOf: cont.tokens.prefix(max(1, extra)))
            }
            if !isMoE || resolvedStrategy == .ngram, drafts.count < count {
                session.tokenRecycler.appendDrafts(
                    into: &drafts, maxCount: count,
                    previousToken: previousToken, pendingToken: pending,
                    maxUnigramFallbacks: isMoE ? 0 : 2)
            }
        }
        if let tDraft0 {
            FileHandle.standardError.write(Data(String(
                format: "%@ pos=%d count=%d wall=%.2fms\n",
                traceTag, session.position, drafts.count, Date().timeIntervalSince(tDraft0) * 1000).utf8))
        }
        lastDraftedPairs = draftedPairs
        mtpDraftProposals += drafts.count
        return MTPRoundProposal(drafts: drafts, draftProposals: draftProposals, draftedPairs: draftedPairs)
    }

    private func stageCommitDeviceEmbeddings(
        _ enc: any MTLComputeCommandEncoder, tokens: ArraySlice<Int>, destination: QWTensorValue
    ) throws {
        guard let table = mtpDeviceEmbedding, tokens.count <= 32 else { return }
        let ids = table.tokens.contents().assumingMemoryBound(to: Int32.self)
        for (i, tok) in tokens.enumerated() { ids[32 + i] = Int32(tok) }
        enc.useResource(table.buffer, usage: .read)
        enc.useResource(table.tokens, usage: .read)
        try table.encodeRows(enc, tokensBuffer: table.tokens, tokensByteOffset: 128,
                             rows: tokens.count, destination: destination)
    }

    func encodeEchoCommitResolution(
        _ cb: any MTLCommandBuffer, session: QWEngineSession, echoDrafter: QWEchoDraftEncoder,
        window: QWVerifyWindow, trunkHidden: QWTensorValue, basePairs: Int,
        retained: Int, usedPLD: Bool
    ) throws {
        guard let views = session.mtpKVViews.first, let windowStream = window.mhcStream else {
            throw QWEngineDriverError.state("ECHO commit resolution requires MTP KV views and window mHC stream")
        }
        let streamElems = echoDrafter.streams * geometry.hidden, streamBytes = streamElems * 4
        let hasDeviceEmbed = mtpDeviceEmbedding != nil
        let targetRows = usedPLD ? retained : (retained - 1)
        guard targetRows >= 1 else { return }
        let tokens = usedPLD ? verifiedInputs[0..<retained] : verifiedInputs[1..<retained]
        let embSlice = hasDeviceEmbed
            ? echoDrafter.embedBuf.slice(byteOffset: 0, count: targetRows * geometry.hidden, dtype: .f16)
            : try echoDrafter.stageEmbeddings(driver: self, tokens: tokens)
        let prevSlice: QWTensorValue
        if !usedPLD {
            prevSlice = windowStream.slice(byteOffset: 0, count: targetRows * streamElems, dtype: .f32)
        } else if retained == 1 {
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
            throw QWEngineDriverError.state("ECHO verify refresh compute encoder")
        }
        do {
            hal.touchKeepAlive(enc)
            enc.useResource(echoDrafter.region.buffer, usage: [.read, .write])
            if hasDeviceEmbed { try stageCommitDeviceEmbeddings(enc, tokens: tokens, destination: embSlice) }
            try echoDrafter.encodeSeedChunk(
                enc, driver: self, prevStream: prevSlice, embeddings: embSlice, kv: views,
                cacheRows: usedPLD ? basePairs : (basePairs + 1), rows: targetRows)
        } catch {
            enc.endEncoding()
            throw error
        }
        enc.endEncoding()
    }

    func encodeNextNCommitResolution(
        _ cb: any MTLCommandBuffer, session: QWEngineSession, draft: QWMTPDraftEncoder,
        window: QWVerifyWindow, trunkHidden: QWTensorValue, basePairs: Int,
        retained: Int, usedPLD: Bool
    ) throws {
        guard let views = session.mtpKVViews.first else {
            throw QWEngineDriverError.state("NextN commit resolution requires MTP KV views")
        }
        let h = geometry.hidden
        let seedable = min(retained, max(0, draft.positionCapacity - session.position))
        let rows = usedPLD ? seedable : (seedable - 1)
        guard seedable >= (usedPLD ? 1 : 2) else { return }
        let tokens = usedPLD ? verifiedInputs[0..<seedable] : verifiedInputs[1..<seedable]
        let embSlice = QWTensorValue(region: window.embeddings.region,
                                     offset: usedPLD ? window.embeddings.offset : (window.embeddings.offset + h * 2),
                                     shape: QWTensorShape([rows * h], .f16))
        let trunkSlice = QWTensorValue(region: window.hidden.region,
                                       offset: usedPLD ? window.hidden.offset : (window.hidden.offset + h * 2),
                                       shape: QWTensorShape([rows * h], .f16))
        let prevHidden = usedPLD ? trunkHidden : (try window.hiddenRow(0))
        let startPos = usedPLD ? session.position : (session.position + 1)
        let startCacheRow = usedPLD ? basePairs : (basePairs + 1)
        if mtpDeviceEmbedding == nil {
            for r in (usedPLD ? 0 : 1)..<seedable {
                try copyEmbedRow(verifiedInputs[r], to: try window.embeddingRow(r))
            }
        }
        guard let enc = cb.makeComputeCommandEncoder() else {
            throw QWEngineDriverError.state("NextN commit seed compute encoder")
        }
        do {
            hal.touchKeepAlive(enc)
            if mtpDeviceEmbedding != nil { try stageCommitDeviceEmbeddings(enc, tokens: tokens, destination: embSlice) }
            if let mw = mtpWeights, rows >= 2, rows <= psc.maxChunk, h % 256 == 0 {
                let seedMRoPE = mlaOperator.flatMap {
                    session.mropeSlice(pos: startPos, count: rows, mropeLayout: $0.mropeLayout)
                }
                try encodeBatchedMTPPairSeed(
                    enc, mtpWeights: mw, embeddings: embSlice, hidden: trunkSlice,
                    previous: prevHidden, pos: startPos, tStart: 0, pairs: rows,
                    cacheRows: startCacheRow, views: views,
                    mropePositions: seedMRoPE, mropeDelta: session.roundRecord.mropeDelta)
            } else {
                try QWMTPPromptSeeder.encode(
                    enc, draft: draft, embeddings: embSlice, trunkHidden: trunkSlice,
                    previousHidden: prevHidden, position: startPos, count: rows,
                    cacheRows: startCacheRow, kv: views, mropeDelta: session.roundRecord.mropeDelta)
            }
        } catch {
            enc.endEncoding()
            throw error
        }
        enc.endEncoding()
    }

    func finalizeMTPPairCursor(
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
        let pairRewind: UInt64
        if let mtpCache, let handle = session.mtpKV {
            pairRewind = try mtpCache.rollbackGeneration(handle)
        } else if isWeightFreeSpeculation || dflashDrafter != nil || dsparkDrafter != nil {
            pairRewind = 0
        } else {
            throw QWEngineDriverError.state("greedy round pair cache is absent")
        }
        return QWGreedyRoundBoundary(position: session.position, generated: session.generated,
                                     record: session.roundRecord, hiddenValid: session.trunkHiddenValid,
                                     trunkRewind: try kvCache.rollbackGeneration(session.kv),
                                     pairRewind: pairRewind)
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
