// FNGPUDFlash.swift — DFlash 2 / DSpine speculative decoding, hybrid PLD bypass,
// sliding-window capture, prefix-cache persistence, and residency helpers for Flash-Next.
import Darwin
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHead
import QWHAL
import QWLoad
import QWMTP
import QWQuant
import QWAttn
import QWHyperPLE

public enum FNGPUDFlash {
    public static let maxDrafts = QWFourSlotRollbackRing.maxVerifyRows - 1
    public static let defaultMTPPackSavingsBytes = QWDFlashFootprint.defaultMTPPackSavingsBytes
    public static let defaultDFlashWeightBytes = QWDFlashFootprint.defaultDFlashWeightBytes
    public static let defaultDFlashScratchBytes = QWDFlashFootprint.defaultDFlashScratchBytes

    public static func isDFlashName(_ raw: String?) -> Bool {
        QWDFlashFootprint.isDFlashName(raw)
    }

    public static func adjustedPerTokenKVBytes(_ perTokenBytes: Int, usesDFlash: Bool) -> Int {
        QWDFlashFootprint.adjustedPerTokenKVBytes(perTokenBytes, usesDFlash: usesDFlash)
    }

    static func resolveDrafterPackPath(targetDescriptor: QWModelDescriptor?, explicitPath: String?) -> String {
        QWDFlashFootprint.resolveDrafterPackPath(targetDescriptor: targetDescriptor, explicitPath: explicitPath)
    }

    /// Draft vocab rows for DFlash proposals (`QWG_DRAFT_VOCAB`, default 98304, 0 = full vocab).
    static func resolveDraftVocab(fullVocab: Int) -> Int? {
        QWDFlashDrafter.resolveDraftVocab(fullVocab: fullVocab, defaultCap: 98_304)
    }

    public static func drafterWeightBytes(packPath: String?, isDSpine: Bool) -> Int {
        QWDFlashFootprint.drafterWeightBytes(packPath: packPath, isDSpine: isDSpine)
    }

    public static func estimateFootprint(
        targetDescriptor: QWModelDescriptor, targetPackPath: String?, dflashPackPath: String?, isDFlash: Bool
    ) -> (nativeMTPWeightBytes: Int, dflashWeightBytes: Int, dflashScratchBytes: Int) {
        QWDFlashFootprint.estimateFootprint(
            targetDescriptor: targetDescriptor, targetPackPath: targetPackPath,
            dflashPackPath: dflashPackPath, isDFlash: isDFlash
        )
    }

    static func prefillWindowEnd(historyCount: Int, targetEnd: Int?) -> Int {
        min(historyCount, targetEnd ?? historyCount)
    }

    static func prefillCaptureSlice(startPos: Int, rows: Int, historyCount: Int, maxCacheSize: Int) -> (localStart: Int, captureRows: Int, ctxStartPos: Int)? {
        QWDFlashConfig.prefillCaptureSlice(
            startPos: startPos, rows: rows, windowEnd: historyCount, maxCacheSize: maxCacheSize
        ).map { ($0.localStart, $0.captureRows, $0.ctxStartPos) }
    }

    static func encodeLayerCapture(
        _ enc: any MTLComputeCommandEncoder, dflash: QWDFlashDrafter, layerIndex: Int, deferOut: Bool,
        base: QWTensorValue, src: QWTensorValue, inj: QWTensorValue, cur: QWTensorValue,
        srcRowOffset: Int, dstRowOffset: Int, rows: Int, hc: Int
    ) {
        guard rows > 0, (0..<dflash.targetLayerSlotMap.count).contains(layerIndex),
              case let slot = dflash.targetLayerSlotMap[layerIndex], slot >= 0 else { return }
        enc.memoryBarrier(scope: .buffers)
        dflash.encodeCaptureHCTargetHidden(
            enc, base: deferOut ? base : cur, srcRowOffset: srcRowOffset, dstRowOffset: dstRowOffset,
            rows: rows, slot: slot, hc: hc, deferredInject: deferOut ? (src: src, inj: inj, isMHC: false) : nil)
        enc.memoryBarrier(scope: .buffers)
    }

    static func dflashGeometryMarker(config: QWDFlashConfig) -> String {
        QWDFlashDrafter.prefixGeometryMarker(config: config)
    }

    static func prefixPayloadBytes(config: QWDFlashConfig, tokenCount: Int) -> Int {
        QWDFlashDrafter.prefixPayloadBytes(config: config, tokenCount: tokenCount)
    }

    static func appendPrefixPayload(
        to dstBase: UnsafeMutableRawPointer, at writeOffset: inout Int,
        dflash: QWDFlashDrafter, marker: String, tokenCount: Int
    ) {
        appendPrefixKV(to: dstBase, at: &writeOffset, slots: dflash.sessionSlots.first, config: dflash.config,
                       marker: marker, tokenCount: tokenCount)
    }

    static func appendPrefixKV(
        to dstBase: UnsafeMutableRawPointer, at writeOffset: inout Int,
        slots: QWDFlashSessionSlot?, config cfg: QWDFlashConfig, marker: String, tokenCount: Int
    ) {
        QWDFlashDrafter.appendPrefixKV(
            to: dstBase, at: &writeOffset, slot: slots, config: cfg, marker: marker, tokenCount: tokenCount
        )
    }

    static func restorePrefixPayload(
        from base: UnsafeRawPointer, dataCount: Int, offset: inout Int,
        dflash: QWDFlashDrafter, expectedMarker: String, expectedTokens: Int
    ) -> Int? {
        restorePrefixKV(from: base, dataCount: dataCount, offset: &offset, slots: dflash.sessionSlots.first,
                        config: dflash.config, expectedMarker: expectedMarker, expectedTokens: expectedTokens)
    }

    static func restorePrefixKV(
        from base: UnsafeRawPointer, dataCount: Int, offset: inout Int,
        slots: QWDFlashSessionSlot?, config cfg: QWDFlashConfig, expectedMarker: String, expectedTokens: Int
    ) -> Int? {
        QWDFlashDrafter.restorePrefixKV(
            from: base, dataCount: dataCount, offset: &offset, slot: slots, config: cfg,
            expectedMarker: expectedMarker, expectedTokens: expectedTokens
        )
    }

    static func validatePrefixBlob(data: Data, expectedMarker: String, expectedTokens: Int, config: QWDFlashConfig) -> Bool {
        QWDFlashDrafter.validatePrefixBlob(
            data: data, expectedMarker: expectedMarker, expectedTokens: expectedTokens, config: config
        )
    }

    static func encodeHeadLogits(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal, headWeight: QWWeightView,
        xRot16: QWTensorValue, xNorm16: QWTensorValue, xRot32: QWTensorValue?,
        yLogits32: QWTensorValue, rows: Int, vocabRows: Int
    ) throws {
        guard let hw = QWHeadWeight(
            blocks: headWeight.blocks, n: headWeight.n, k: headWeight.k,
            bits: headWeight.bits, groupSize: headWeight.groupSize,
            isRotated: headWeight.isRotated,
            isIncumbentContainer: headWeight.isIncumbentContainer,
            isSHVQ: headWeight.isSHVQ
        ) else {
            throw QWDFlashError.state("FNGPUDFlash: invalid head weight for vocabRows=\(vocabRows)")
        }
        try QWDFlashDrafter.encodeSlicedHeadLogits(
            enc, hal: hal, headStorage: .q4g(hw), geometry: hal.geometry,
            xRot16: xRot16, xNorm16: xNorm16, xRot32: xRot32,
            yLogits32: yLogits32, rows: rows, vocabRows: vocabRows
        )
    }

    static func selectHybridBypassDrafts(
        history: [Int64], pendingToken: Int, mode: QWDraftStrategy, maxDrafts: Int, recycler: QWTokenRecycler
    ) -> [Int] {
        QWContextCopyDrafter.selectHybridPLDDrafts(
            history: history, extraToken: Int64(pendingToken),
            previousToken: history.last.map(Int.init),
            pendingToken: pendingToken, mode: mode,
            maxDrafts: maxDrafts, recycler: recycler
        )
    }

    static func maxFittingContext(activeCapacity: Int) -> Int {
        QWDFlashFootprint.maxFittingContext(activeCapacity: activeCapacity)
    }

    static func consumePendingContextUpdate(
        pos: Int, count: inout Int, slot: QWDFlashSessionSlot?
    ) -> (slot: QWDFlashSessionSlot, startPosition: Int, totalCount: Int)? {
        let pending = count
        count = 0
        guard pending > 0, let slot else { return nil }
        return (slot, pos, pending)
    }

    static func consumePendingContextUpdate(
        _ session: inout FNGPUSessionData, slot: QWDFlashSessionSlot?
    ) -> (slot: QWDFlashSessionSlot, startPosition: Int, totalCount: Int)? {
        consumePendingContextUpdate(pos: session.dflashPendingCtxPos,
                                    count: &session.dflashPendingCtxCount,
                                    slot: slot)
    }

    static func step(
        _ engine: QWHyperStreamRuntime, session: inout FNGPUSessionData, token: Int,
        vStack: FNGPUDecodeStack, dflash: QWDFlashDrafter
    ) throws -> Int {
        guard let dSlot = dflash.sessionSlots.first else {
            throw QWDFlashError.state("FNGPUDFlash: missing DFlash session slot")
        }
        let prevCount = session.history.count
        let dflashCap = min(maxDrafts, max(1, dflash.config.blockSize - 1))
        let maxCap = max(1, min(min(session.maxDrafts, dflashCap), engine.activeCapacity - prevCount - 2))
        let stepT0 = Date(), trackRep = session.sampler.config.repetitionPenalty != 1.0
        let isUnconstrained = session.constraint?.isUnconstrainedNow ?? true, isGreedy = session.sampler.config.isGreedy
        let canSpecConstrained = session.constraint.map {
            $0.isUnconstrainedNow || (QWConstrainedSpeculationGate.isEnabled && $0.allowsSpeculation)
        } ?? true
        let allowPLD = isUnconstrained && isGreedy && session.seenTokens.isEmpty && !session.sampler.config.hasExplicitSeed
        let preCtx = consumePendingContextUpdate(&session, slot: dSlot)
        let canonical = (!isUnconstrained && QWConstrainedSpeculationGate.isEnabled)
            ? (session.constraint?.proposeCanonicalDrafts(maxCount: maxCap) ?? [])
            : []
        var drafts = !canonical.isEmpty
            ? canonical
            : (allowPLD
                ? selectHybridBypassDrafts(history: session.history, pendingToken: token,
                                           mode: session.draftMode, maxDrafts: maxCap, recycler: session.tokenRecycler)
                : [])
        let usedPLD = !drafts.isEmpty
        var proposals: [QWMTPGumbelProposal] = [], draftGPUMs = 0.0
        let stream: QWMTPRandomStream? = (!usedPLD && !isGreedy) ? QWMTPRandomStream(cloning: try session.ensureDFlashStreamState()) : nil
        if !usedPLD {
            let draftCount = session.dflashPolicy.selectDraftCount(maxDrafts: maxCap)
            if let stream { try stream.beginGumbelRound(drafts: draftCount, vocabulary: engine.vocab, maxDrafts: dflashCap) }
            do {
                let proposed = try dflash.propose(
                    hal: engine.ctx.hal, slot: dSlot, cacheOffset: session.dflashCacheOffset,
                    pendingToken: token, draftCount: draftCount, draftVocab: resolveDraftVocab(fullVocab: engine.vocab),
                    headStorage: engine.dflashHeadStorage,
                    encodeHeadLogits: { enc, xRot16, xNorm16, xRot32, yLogits32, rows, vocabRows in
                        try encodeHeadLogits(enc, hal: engine.ctx.hal, headWeight: engine.head.weight, xRot16: xRot16,
                                             xNorm16: xNorm16, xRot32: xRot32, yLogits32: yLogits32, rows: rows, vocabRows: vocabRows)
                    },
                    encodeAnchorMaskEmbed: { enc, propX32, anchorToken, maskToken, rows in
                        try dflash.encodeStageProp(enc, embedTable: engine.embedTable, propX: propX32,
                                                   anchorToken: anchorToken, maskToken: maskToken, rows: rows)
                    },
                    stochastic: stream.map { (Float(max(session.sampler.config.temperature, 1e-6)), $0) },
                    preContextUpdate: preCtx,
                    completed: { cb in draftGPUMs = max(0, (cb.gpuEndTime - cb.gpuStartTime) * 1000.0) })
                drafts = proposed.drafts; proposals = proposed.proposals
            } catch { stream?.cancelActiveRound(); throw error }
        }
        let dCount = drafts.count, activeRows = 1 + dCount
        let prevBefore = prevCount > 0 ? Int(session.history[prevCount - 1]) : nil
        session.history.append(Int64(token)); drafts.forEach { session.history.append(Int64($0)) }
        if engine.residentKVHistory.count == prevCount {
            engine.residentKVHistory.append(Int64(token)); drafts.forEach { engine.residentKVHistory.append(Int64($0)) }
        }
        engine.highWaterPos = max(engine.highWaterPos, session.history.count)

        let margin: Float = (usedPLD || !isGreedy || !isUnconstrained) ? 0 : session.relaxedMargin
        _ = try vStack.forwardRows(
            history: session.history, embedTable: engine.embedTable, head: engine.head,
            readStream: false, verifyAllRows: true, mtp: nil, activeRows: activeRows,
            relaxedMargin: margin, relaxedRows: margin > 0 ? activeRows : 0,
            relaxedMinProb: session.relaxedMinProb, dflash: dflash, preContextUpdate: usedPLD ? preCtx : nil)
        session.specProposals += dCount
        let (tok0, accepted) = try resolveVerifiedTokens(
            engine, session: &session, drafts: drafts, proposals: proposals, stream: stream,
            prevCount: prevCount, prevBefore: prevBefore, trackRep: trackRep,
            verifyMargin: margin, usedPLD: usedPLD, canSpecConstrained: canSpecConstrained)
        if accepted < dCount {
            let rejected = dCount - accepted
            session.history.removeLast(rejected)
            if engine.residentKVHistory.count == prevCount + activeRows { engine.residentKVHistory.removeLast(rejected) }
            vStack.rollbackVerifyRows(rejectedCount: rejected)
        }
        let roundMs = Date().timeIntervalSince(stepT0) * 1000.0
        session.specAccepted += accepted
        if !usedPLD { session.dflashPolicy.observeRound(draftCount: dCount, accepted: accepted, roundMs: roundMs) }
        session.dflashPendingCtxPos = prevCount; session.dflashPendingCtxCount = 1 + accepted
        session.dflashCacheOffset = prevCount + 1 + accepted; session.headToken = -1
        let divisor = Double(1 + session.pendingTokens.count)
        session.stepMs = roundMs / divisor; session.lastStepGPUMillis = (draftGPUMs + vStack.lastGPUMillis) / divisor
        session.roundTokenCounts.append(1)
        return tok0
    }

    private static func resolveVerifiedTokens(
        _ engine: QWHyperStreamRuntime, session: inout FNGPUSessionData,
        drafts: [Int], proposals: [QWMTPGumbelProposal], stream: QWMTPRandomStream?,
        prevCount: Int, prevBefore: Int?, trackRep: Bool, verifyMargin: Float,
        usedPLD: Bool, canSpecConstrained: Bool
    ) throws -> (tok0: Int, accepted: Int) {
        let dCount = drafts.count
        let startUnconstrained = session.constraint?.isUnconstrainedNow ?? true
        let probe = (!startUnconstrained && canSpecConstrained) ? session.constraint?.makeSpeculativeProbe() : nil
        if !proposals.isEmpty, let stream, (startUnconstrained || probe != nil) {
            let logitsPtr = engine.arena.cpuPointer(of: engine.head.logits).assumingMemoryBound(to: Float.self)
            var targets: [QWSamplingDistribution] = []
            var probeAlive = true
            for row in 0..<(1 + dCount) {
                if !probeAlive, let fallback = targets.last {
                    targets.append(fallback)
                    continue
                }
                let rowPtr = UnsafeBufferPointer(start: logitsPtr.advanced(by: row * engine.vocab), count: engine.vocab)
                let rep = trackRep
                    ? FNGPUSessionData.loopRepetitionTokens(
                        in: Array((session.recentGeneratedTokens + drafts.prefix(row))
                            .suffix(FNGPUSessionData.repetitionPenaltyWindow)))
                    : []
                if let probe, try probe.writeAllowedTokens(into: &session.sampler.allowedScratch) {
                    targets.append(try session.sampler.constrainedDistributionFromScratch(
                        logits: rowPtr, repetitionHistory: rep))
                    if row < dCount, (try? probe.accept(drafts[row])) == nil {
                        probeAlive = false
                    }
                } else {
                    targets.append(try session.sampler.distribution(logits: rowPtr, repetitionHistory: rep))
                    if let probe, row < dCount, (try? probe.accept(drafts[row])) == nil {
                        probeAlive = false
                    }
                }
            }
            let res = session.relaxedMargin > 0
                ? try stream.resolveRelaxedGumbelRound(proposals: proposals, targetDistributions: targets,
                                                       margin: Double(session.relaxedMargin))
                : try stream.resolveGumbelRound(proposals: proposals, targetDistributions: targets)
            session.dflashStreamState = try stream.frozenState()
            let boundaryTok = session.constraint?.speculationBoundaryToken
            var committedAccepted = res.accepted
            var finalTok = res.finalToken
            let commitProbe = session.constraint?.makeSpeculativeProbe()
            let commitConstraint = commitProbe ?? session.constraint
            if let constraint = commitConstraint {
                for idx in 0..<res.accepted {
                    let tok = drafts[idx]
                    try constraint.accept(tok)
                    if !canSpecConstrained && (tok == boundaryTok || !constraint.isUnconstrainedNow) {
                        committedAccepted = idx
                        finalTok = tok
                        break
                    }
                }
                if committedAccepted == res.accepted {
                    try constraint.accept(finalTok)
                }
            }
            if committedAccepted == 0 {
                if commitProbe != nil { try session.constraint?.accept(finalTok) }
                if trackRep { session.recordSeenToken(finalTok, vocab: engine.vocab) }
                return (finalTok, 0)
            }
            let tok0 = drafts[0]
            if commitProbe != nil { try session.constraint?.accept(tok0) }
            if trackRep { session.recordSeenToken(tok0, vocab: engine.vocab) }
            for i in 1..<committedAccepted {
                session.pendingTokens.append(drafts[i]); if trackRep { session.recordSeenToken(drafts[i], vocab: engine.vocab) }
            }
            session.pendingTokens.append(finalTok)
            if commitProbe != nil { session.pendingConstraintAcceptCount += committedAccepted }
            if trackRep { session.recordSeenToken(finalTok, vocab: engine.vocab) }
            return (tok0, committedAccepted)
        }
        let v = engine.head.argmaxTokens(count: 1 + dCount)
        if verifyMargin == 0 && session.sampler.config.isGreedy && startUnconstrained {
            for row in (0..<(1 + dCount)).reversed() {
                let prev = (row == 0) ? prevBefore : Int(session.history[prevCount + row - 1])
                session.tokenRecycler.recordTransition(previous: prev, current: Int(session.history[prevCount + row]), next: v[row])
            }
        }
        var tok0 = 0, accepted = 0
        let greedyProbe = session.constraint?.makeSpeculativeProbe()
        let roundConstraint = greedyProbe ?? session.constraint
        let boundaryTok = roundConstraint?.speculationBoundaryToken
        let rowMargin: Float = (!usedPLD && !startUnconstrained && session.sampler.config.isGreedy)
            ? session.relaxedMargin : 0
        let prevPendingCount = session.pendingTokens.count
        for r in 0..<dCount {
            let draftTok = drafts[r], sampled: Int
            let rowUnconstrained = roundConstraint?.isUnconstrainedNow ?? true
            if v[r] == draftTok && session.sampler.config.isGreedy && !session.seenTokens.contains(draftTok) && rowUnconstrained {
                try roundConstraint?.accept(draftTok); sampled = draftTok
            } else {
                sampled = try engine.sampleFromHead(
                    argmaxToken: v[r], seenTokens: session.seenTokens,
                    seenTokenList: session.seenTokenList,
                    sampler: session.sampler, row: r, constraint: roundConstraint,
                    draftToken: draftTok, relaxedMargin: rowMargin,
                    relaxedMinProb: session.relaxedMinProb)
            }
            if r == 0 { tok0 = sampled } else { session.pendingTokens.append(sampled) }
            let hitBoundary = !canSpecConstrained
                && ((boundaryTok != nil && sampled == boundaryTok) || !(roundConstraint?.isUnconstrainedNow ?? true))
            let matchesTarget = canSpecConstrained ? (sampled == draftTok) : (sampled == draftTok && v[r] == draftTok)
            if matchesTarget && !hitBoundary {
                accepted += 1; if trackRep { session.recordSeenToken(draftTok, vocab: engine.vocab) }
            } else {
                if trackRep { session.recordSeenToken(sampled, vocab: engine.vocab) }; break
            }
        }
        if accepted == dCount {
            let bonus = try engine.sampleFromHead(argmaxToken: v[dCount], seenTokens: session.seenTokens,
                                                  seenTokenList: session.seenTokenList,
                                                  sampler: session.sampler, row: dCount, constraint: roundConstraint)
            if trackRep { session.recordSeenToken(bonus, vocab: engine.vocab) }
            session.pendingTokens.append(bonus)
        }
        if greedyProbe != nil {
            try session.constraint?.accept(tok0)
            session.pendingConstraintAcceptCount += (session.pendingTokens.count - prevPendingCount)
        }
        return (tok0, accepted)
    }
}
