// QWInkling2Runtime+StochasticSpec.swift — Variational, stochastic & weight-free PLD speculative decode.
import Foundation
import QWCore
import QWHead
import QWMTP

extension QWInkling2Runtime {
    static func canSpeculate(constraint: (any QWTokenConstraint)?) -> Bool {
        guard let constraint else { return true }
        if constraint.isUnconstrainedNow { return true }
        return QWConstrainedSpeculationGate.isEnabled && constraint.allowsSpeculation
    }

    static func loopPenaltyHistory(
        baseRecent: [Int],
        baseActive: [Int],
        draftPrefix: ArraySlice<Int>,
        vocab: Int
    ) -> [Int] {
        guard !draftPrefix.isEmpty else { return baseActive }
        var window = baseRecent
        var active = baseActive
        for token in draftPrefix {
            recordGeneratedTokenForPenalty(token, vocab: vocab, recentGenerated: &window, activeLoopTokens: &active)
        }
        return active
    }

    func consumeResolvedRound(
        _ resolution: QWSpeculativeRoundResolver.Resolution, vocab: Int, effectiveMaxNew: Int,
        stopIDs: Set<Int>, constraint: (any QWTokenConstraint)? = nil, usePenalty: Bool,
        recentGeneratedForPenalty: inout [Int], penaltyHistory: inout [Int],
        history: inout [Int], generated: inout [Int], stopTokenID: inout Int?,
        onStopToken: ((Int) -> Void)?, onToken: ((Int) -> Bool)?
    ) throws -> (accepted: Int, emitted: Int, stopped: Bool, cancelled: Bool) {
        var accepted = 0, emitted = 0, stopped = false, cancelled = false
        func emit(_ tok: Int) throws -> EmittedTokenOutcome {
            try consumeSpeculativeToken(
                tok, vocab: vocab, stopIDs: stopIDs, constraint: constraint, usePenalty: usePenalty,
                recentGeneratedForPenalty: &recentGeneratedForPenalty, penaltyHistory: &penaltyHistory,
                history: &history, generated: &generated, stopTokenID: &stopTokenID,
                onStopToken: onStopToken, onToken: onToken)
        }
        for idx in 0..<resolution.tokens.count {
            let outcome = try emit(resolution.tokens[idx])
            if outcome == .stopped { stopped = true; break }
            emitted += 1
            if idx < resolution.acceptedDrafts { accepted += 1 }
            if outcome == .cancelled { cancelled = true; break }
            if generated.count >= effectiveMaxNew { break }
        }
        if !stopped && !cancelled && resolution.stopped, let stopTok = resolution.stoppedToken {
            if try emit(stopTok) == .stopped { stopped = true }
        }
        return (accepted, emitted, stopped, cancelled)
    }

    func activeMoESpecBudget(verifyRows: Int) -> Int {
        guard verifyRows > 1, geometry.numExperts > 0, geometry.expertsPerTok > 0 else { return 0 }
        if let raw = ProcessInfo.processInfo.environment["QW_MOE_SPEC_EXPERT_BUDGET"] {
            if raw.lowercased() == "auto" {
                return QWMTPAdaptiveDepth.moeSpecDraftExpertBudget(geometry: geometry, verifyRows: verifyRows)
            }
            if let explicit = Int(raw), explicit > 0 {
                return QWMTPAdaptiveDepth.moeSpecDraftExpertBudget(
                    geometry: geometry, verifyRows: verifyRows, explicitBudget: explicit)
            }
        }
        if configuredMoESpecBudget > 0 {
            return QWMTPAdaptiveDepth.moeSpecDraftExpertBudget(
                geometry: geometry, verifyRows: verifyRows, explicitBudget: configuredMoESpecBudget)
        }
        return 0
    }

    func resolveSpeculativeMargin(request: QWGenerationRequest) -> (margin: Float, minLogProb: Float) {
        let reqBackend = request.speculative.mtpBackend?.lowercased()
        let margin: Float = {
            if request.sampling.hasExplicitSeed { return 0 }
            if reqBackend == "variational" || reqBackend == "relaxed" || reqBackend == "plm" {
                let m = request.speculative.mtpRelaxedMargin > 0
                    ? request.speculative.mtpRelaxedMargin : configuredRelaxedMargin
                return Float(max(0.0, m))
            }
            if reqBackend == "standard" || reqBackend == "dflash2" || reqBackend == "dspine" { return 0 }
            return Float(max(0.0, configuredRelaxedMargin))
        }()
        let minProbRaw: Double = (reqBackend != nil)
            ? request.speculative.mtpRelaxedMinProb
            : (configuredRelaxedMinProb > 0 ? configuredRelaxedMinProb : request.speculative.mtpRelaxedMinProb)
        let minLogProb: Float = (margin > 0 && minProbRaw > 0) ? Float(log(min(1.0, minProbRaw))) : -.infinity
        return (margin, minLogProb)
    }

    func runFlexibleSpeculativeDecode(
        dspark: QWInkling2DSpark?,
        requestedDrafts: Int,
        allowPLD: Bool = true,
        sampler: QWSampler,
        constraint: (any QWTokenConstraint)? = nil,
        allowedScratch: inout [Int],
        effectiveMargin: Float,
        minLogProb: Float,
        promptTokens: [Int],
        pos: inout Int,
        effectiveMaxNew: Int,
        stopIDs: Set<Int>,
        generated: inout [Int],
        recentGeneratedForPenalty: inout [Int],
        penaltyHistory: inout [Int],
        stepMs: inout [Double],
        stepCounts: inout [Int],
        stopTokenID: inout Int?,
        onToken: ((Int) -> Bool)?,
        onStopToken: ((Int) -> Void)?,
        profiler: QWInkling2Profiler? = nil
    ) throws -> (stats: SpeculativeStats, finished: Bool) {
        let clock = ContinuousClock()
        let V = ink2.unpaddedVocabSize
        let logitsPtr = session.logitsF32Buf.contents().assumingMemoryBound(to: Float.self)
        let isGreedy = sampler.config.isGreedy
        let usePenalty = sampler.config.repetitionPenalty != 1.0
        var stats = SpeculativeStats()
        let minProbFloor: Float = minLogProb == -.infinity ? 0 : Float(exp(Double(minLogProb)))
        let relaxedGreedySampler = (isGreedy && effectiveMargin > 0)
            ? QWSampler(config: QWSamplerConfig(
                temperature: 1.0, topK: V, topP: 1.0,
                uniformVersion: sampler.config.uniformVersion,
                repetitionPenalty: sampler.config.repetitionPenalty))
            : nil
        let streamSeed: UInt64 = sampler.config.hasExplicitSeed
            ? (sampler.config.seed ^ 0x5354_4F43_4853_5045)
            : UInt64.random(in: 1...UInt64.max)
        let stream: QWMTPRandomStream? = isGreedy
            ? nil
            : QWMTPRandomStream(seed: streamSeed, drawIndex: sampler.drawsConsumed)

        session.tokenRecycler.reset()
        session.tokenRecycler.ingestSequence(promptTokens)
        if let lastPrompt = promptTokens.last {
            let prev = promptTokens.count >= 2 ? promptTokens[promptTokens.count - 2] : nil
            session.tokenRecycler.recordTransition(previous: prev, current: lastPrompt, next: generated[0])
        }

        var history = [Int]()
        history.reserveCapacity(promptTokens.count + effectiveMaxNew)
        history.append(contentsOf: promptTokens)
        history.append(generated[0])

        let explicitTreeEnv = ProcessInfo.processInfo.environment["QWG_DSPARK_TREE"] != nil
        let pldCap = min(15, dspark != nil ? max(7, requestedDrafts) : requestedDrafts)
        let linCap = dspark.map { min(requestedDrafts, max(1, $0.config.blockSize - 1)) } ?? 0
        func maxVerifyDrafts(cap: Int) -> Int {
            let remB = effectiveMaxNew - generated.count
            let remC = (capacity - pos) - 2
            return max(0, min(cap, remB - 1, remC))
        }

        func proposeConstrainedOrPLD(maxCount: Int, allowRecycler: Bool) -> [Int] {
            guard maxCount > 0 else { return [] }
            if let constraint, !constraint.isUnconstrainedNow, QWConstrainedSpeculationGate.isEnabled {
                let canonical = constraint.proposeCanonicalDrafts(maxCount: maxCount)
                if !canonical.isEmpty { return canonical }
            }
            guard allowPLD else { return [] }
            let raw = makePLDDrafts(history: history, maxCount: maxCount, allowRecyclerFallback: allowRecycler)
            guard !raw.isEmpty else { return [] }
            guard let constraint, !constraint.isUnconstrainedNow else { return raw }
            guard let probe = constraint.makeSpeculativeProbe() else { return [] }
            var filtered: [Int] = []
            filtered.reserveCapacity(raw.count)
            for tok in raw {
                guard (try? probe.accept(tok)) != nil else { break }
                filtered.append(tok)
            }
            return filtered
        }

        func resolveRound(
            drafts: [Int],
            margin: Float,
            budget: Int,
            rowStopIDs: Set<Int>,
            allowedScratch: inout [Int]
        ) throws -> QWSpeculativeRoundResolver.Resolution {
            let proposals = drafts.map { QWSpeculativeRoundResolver.DraftProposal.greedyToken($0) }
            return try QWSpeculativeRoundResolver.resolve(
                draftProposals: proposals, stream: stream, isGreedy: isGreedy,
                maxNew: budget, stopIDs: rowStopIDs, constraint: constraint,
                margin: margin, minProbability: minProbFloor
            ) { row, draftPrefix, probe in
                let rowPtr = UnsafeBufferPointer(start: logitsPtr.advanced(by: row * V), count: V)
                let rep = usePenalty
                    ? Self.loopPenaltyHistory(
                        baseRecent: recentGeneratedForPenalty, baseActive: penaltyHistory,
                        draftPrefix: draftPrefix, vocab: V)
                    : []
                let hasMask = try probe?.writeAllowedTokens(into: &allowedScratch) ?? false
                if isGreedy {
                    if margin > 0, row < drafts.count, let relaxedGreedySampler {
                        return hasMask
                            ? try allowedScratch.withUnsafeBufferPointer {
                                try relaxedGreedySampler.constrainedDistribution(
                                    logits: rowPtr, allowed: $0, repetitionHistory: rep)
                            }
                            : try relaxedGreedySampler.distribution(logits: rowPtr, repetitionHistory: rep)
                    }
                    let tok = hasMask
                        ? try allowedScratch.withUnsafeBufferPointer {
                            try sampler.constrainedToken(logits: rowPtr, allowed: $0, repetitionHistory: rep)
                        }
                        : try sampler.penalisedGreedyToken(logits: rowPtr, repetitionHistory: rep)
                    return try QWSamplingDistribution.sparse(vocabulary: V, ids: [tok], weights: [1.0])
                }
                return hasMask
                    ? try allowedScratch.withUnsafeBufferPointer {
                        try sampler.constrainedDistribution(logits: rowPtr, allowed: $0, repetitionHistory: rep)
                    }
                    : try sampler.distribution(logits: rowPtr, repetitionHistory: rep)
            }
        }

        var pendingDrafts: [Int]?, pendingWasPLD = false
        var lowYieldStreak = 0, cooldownSteps = 0

        let initPld = proposeConstrainedOrPLD(maxCount: maxVerifyDrafts(cap: pldCap), allowRecycler: dspark == nil)
        if !initPld.isEmpty {
            pendingDrafts = initPld; pendingWasPLD = true
        } else if let ds = dspark {
            let cnt = maxVerifyDrafts(cap: linCap)
            if cnt > 0 {
                let d = try ds.draftBlock(anchor: generated[generated.count - 1], count: cnt, profiler: profiler)
                if !d.isEmpty { pendingDrafts = d }
            }
        }

        var verifyTokens = [Int]()
        verifyTokens.reserveCapacity(16)

        while generated.count < effectiveMaxNew {
            let roundStart = clock.now
            let anchor = generated[generated.count - 1]

            if pendingDrafts == nil {
                let pldNow = proposeConstrainedOrPLD(
                    maxCount: maxVerifyDrafts(cap: pldCap), allowRecycler: dspark == nil)
                if !pldNow.isEmpty {
                    pendingDrafts = pldNow; pendingWasPLD = true; cooldownSteps = 0
                }
            }

            guard let drafts = pendingDrafts, !drafts.isEmpty else {
                try forwardPass.forward(
                    tokens: [anchor], posOffset: pos, runHeadRows: 1, greedyHead: false,
                    updateState: true, captureTapBuf: dspark?.tapBuf,
                    tapVisits: dspark?.config.targetLayerIds ?? [], profiler: profiler
                )
                if let ds = dspark {
                    try ds.ingestContextTaps(count: 1, posOffset: pos, rowStride: 1, profiler: profiler)
                }
                pos += 1
                let headTokPtr = session.headTokensBuf.contents().assumingMemoryBound(to: Int32.self)
                let tok = try sampleRowToken(
                    row: 0, vocab: V, useGreedyHead: false, logitsPtr: logitsPtr,
                    headTokPtr: headTokPtr, sampler: sampler, penaltyHistory: penaltyHistory,
                    constraint: constraint, allowedScratch: &allowedScratch)
                stepMs.append(elapsedSeconds(roundStart.duration(to: clock.now)) * 1000.0)
                let outcome = try consumeSpeculativeToken(
                    tok, vocab: V, stopIDs: stopIDs, constraint: constraint, usePenalty: usePenalty,
                    recentGeneratedForPenalty: &recentGeneratedForPenalty, penaltyHistory: &penaltyHistory,
                    history: &history, generated: &generated, stopTokenID: &stopTokenID,
                    onStopToken: onStopToken, onToken: onToken
                )
                if outcome == .stopped { stepCounts.append(0); return (stats, true) }
                stepCounts.append(1)
                if outcome == .cancelled { return (stats, true) }
                if cooldownSteps > 0 { cooldownSteps -= 1 }
                let nextInit = proposeConstrainedOrPLD(
                    maxCount: maxVerifyDrafts(cap: pldCap), allowRecycler: dspark == nil)
                if !nextInit.isEmpty {
                    pendingDrafts = nextInit; pendingWasPLD = true
                } else if let ds = dspark, cooldownSteps == 0, generated.count < effectiveMaxNew {
                    let cnt = maxVerifyDrafts(cap: linCap)
                    if cnt > 0 {
                        let d = try ds.draftBlock(anchor: tok, count: cnt, profiler: profiler)
                        if !d.isEmpty { pendingDrafts = d; pendingWasPLD = false }
                    }
                }
                continue
            }
            let usedPLDThisRound = pendingWasPLD
            pendingDrafts = nil

            verifyTokens.removeAll(keepingCapacity: true)
            verifyTokens.append(anchor)
            verifyTokens.append(contentsOf: drafts)
            let B = verifyTokens.count

            try forwardPass.forward(
                tokens: verifyTokens, posOffset: pos, runHeadRows: B, greedyHead: false,
                updateState: true, captureTapBuf: dspark?.tapBuf,
                tapVisits: dspark?.config.targetLayerIds ?? [], recordVerifyConv: true,
                profiler: profiler, moeSpecBudget: activeMoESpecBudget(verifyRows: B)
            )

            let roundMargin: Float = usedPLDThisRound ? 0 : effectiveMargin
            let roundRes = try resolveRound(
                drafts: drafts, margin: roundMargin,
                budget: effectiveMaxNew - generated.count, rowStopIDs: stopIDs,
                allowedScratch: &allowedScratch
            )
            let (accepted, emitted, stopped, cancelled) = try consumeResolvedRound(
                roundRes, vocab: V, effectiveMaxNew: effectiveMaxNew, stopIDs: stopIDs,
                constraint: constraint, usePenalty: usePenalty,
                recentGeneratedForPenalty: &recentGeneratedForPenalty, penaltyHistory: &penaltyHistory,
                history: &history, generated: &generated, stopTokenID: &stopTokenID,
                onStopToken: onStopToken, onToken: onToken
            )

            let oldPos = pos, kept = 1 + accepted
            pos += kept
            if dspark != nil && !explicitTreeEnv && !usedPLDThisRound {
                if accepted == 0 { lowYieldStreak += 1 } else { lowYieldStreak = 0 }
                if lowYieldStreak >= 2 { cooldownSteps = 3; lowYieldStreak = 0 }
            }

            let continueSpec = !stopped && !cancelled && generated.count < effectiveMaxNew
            if let ds = dspark {
                let pldNext = continueSpec
                    ? proposeConstrainedOrPLD(maxCount: maxVerifyDrafts(cap: pldCap), allowRecycler: false) : []
                let inCooldown = !explicitTreeEnv && continueSpec && pldNext.isEmpty && cooldownSteps > 0
                let nextDCount = (continueSpec && pldNext.isEmpty && !inCooldown)
                    ? maxVerifyDrafts(cap: linCap) : 0
                let gpuNext = try ds.finishSpeculativeRound(
                    session: session, kept: kept, oldPos: oldPos, rowStride: B,
                    ingestTaps: continueSpec, nextAnchor: generated[generated.count - 1],
                    nextDraftCount: nextDCount, profiler: profiler
                )
                if !pldNext.isEmpty {
                    pendingDrafts = pldNext; pendingWasPLD = true; cooldownSteps = 0
                } else if nextDCount > 0 && !gpuNext.isEmpty {
                    pendingDrafts = gpuNext; pendingWasPLD = false
                }
            } else {
                try session.commitVerifyConv(hal: hal, kept: kept, profiler: profiler)
            }

            stepMs.append(elapsedSeconds(roundStart.duration(to: clock.now)) * 1000.0)
            stepCounts.append(emitted)
            stats.recordRound(emitted: emitted, proposals: drafts.count, accepted: accepted)
            if stopped || cancelled { return (stats, true) }
        }
        return (stats, generated.count >= effectiveMaxNew)
    }
}
