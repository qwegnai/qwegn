// MTPVerification+Rounds.swift — Speculative draft-verify round lifecycle on QWEngineDriver.

import Foundation
import Darwin
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWMLP
import QWHead
import QWMTP
import QWQuant

extension QWEngineDriver {
    /// True while a speculative verify window fits below both `ropeRows` and the draft encoder's RoPE ceiling.
    func canRunSpeculativeRound(
        _ session: QWEngineSession,
        constraintOverride: (any QWTokenConstraint)? = nil
    ) -> Bool {
        guard let verifyWindow, !session.batchedDecodeTouched,
              !session.sessionFinished, !session.stateFailed, !session.verificationPending,
              session.trunkHiddenValid,
              session.roundRecord.consumedCount == session.position,
              session.roundRecord.absolutePosition == session.position,
              session.roundRecord.outputCount == session.generated else {
            return false
        }
        if let mtpCache {
            guard let handle = session.mtpKV,
                  mtpCache.written(handle) == session.roundRecord.mtpPhysicalRows else {
                return false
            }
        }
        if dflashDrafter != nil {
            guard session.dflashSlot != nil, session.dflashCacheOffset == session.position else {
                return false
            }
        }
        if dsparkDrafter != nil {
            guard session.dsparkSlot != nil, session.dsparkCacheOffset == session.position else {
                return false
            }
        }
        let count = verifyWindow.rows - 1
        if lsaOperator != nil,
           session.position < lsaConfig.denseBypassThreshold,
           session.position + verifyWindow.rows > lsaConfig.denseBypassThreshold {
            return false
        }
        let effConstraint = constraintOverride ?? activeConstraint?.constraint
        if isWeightFreeSpeculation {
            guard canSpeculateConstraint(effConstraint),
                  let pending = session.roundRecord.pendingToken else {
                return false
            }
            let hasCanonical = !(effConstraint?.proposeCanonicalDrafts(maxCount: count).isEmpty ?? true)
            if !hasCanonical && proposeWeightFreeDrafts(session: session, pending: pending, count: count) == nil {
                return false
            }
        }
        let draftCapacity = (dflashDrafter != nil || dsparkDrafter != nil ? nil : mtpDraftEncoder?.positionCapacity) ?? ropeRows
        return session.position > 0
            && session.position <= ropeRows - verifyWindow.rows
            && session.position <= draftCapacity - count
    }

    /// Diagnostic two-phase round owner. Output is tentative until commit succeeds;
    /// scratch stays leased across prefix selection until commit/discard/finish.
    /// No user callback runs under flight, and ordinary generation never calls it.
    func prepareGreedyRound(_ session: QWEngineSession, maxNew: Int,
                            stopIDs: Set<Int> = [],
                            permit: QWGenerationPermit? = nil) throws -> QWPreparedGreedyRound {
        try prepareRound(session, maxNew: maxNew, stopIDs: stopIDs, stochastic: false, permit: permit)
    }

    func prepareStochasticRound(_ session: QWEngineSession, maxNew: Int,
                                stopIDs: Set<Int> = [],
                                permit: QWGenerationPermit? = nil) throws -> QWPreparedStochasticRound {
        QWPreparedStochasticRound(authorization:
            try prepareRound(session, maxNew: maxNew, stopIDs: stopIDs, stochastic: true, permit: permit))
    }

    func prepareRound(_ session: QWEngineSession, maxNew: Int, stopIDs: Set<Int>,
                              stochastic: Bool, permit: QWGenerationPermit?) throws -> QWPreparedGreedyRound {
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate(permit)
        let profileStart = mtpStageProfiling ? DispatchTime.now().uptimeNanoseconds : nil
        let dsparkCanOverlap = dsparkDrafter.map {
            QWDSparkDrafter.allowsPriorCommitOverlap(
                schedulerEnabled: $0.schedulerEnabled,
                verifyRows: verifyWindow?.rows ?? 0,
                maxDrafts: $0.config.maxDrafts
            )
        } ?? false
        let canOverlapPriorCommit = !mtpStageProfiling && !mtpVerificationProfiling
            && !QWDSparkDrafter.stageProfileEnabled && coldEvents == nil && mtpDeviceEmbedding != nil
            && (dflashDrafter != nil || dsparkCanOverlap || echoDrafter != nil || mtpDraftEncoder != nil)
        var pendingCommitCB: (any MTLCommandBuffer)?
        if let cb = session.lastCommitCB {
            session.lastCommitCB = nil
            if canOverlapPriorCommit {
                if cb.status == .error || syncCommitFailure(for: session) {
                    recordCommandBufferError(for: cb)
                    session.stateFailed = true
                }
                pendingCommitCB = cb
            } else if echoDrafter != nil || mtpDraftEncoder != nil || dflashDrafter != nil || dsparkDrafter != nil || isWeightFreeSpeculation {
                cb.waitUntilCompleted()
                _ = syncCommitFailure(for: session, cb: cb)
            } else {
                _ = syncCommitFailure(for: session, cb: cb.status == .error ? cb : nil)
            }
        } else {
            _ = syncCommitFailure(for: session)
        }
        func drainPendingCommitIfNeeded() throws {
            if let cb = pendingCommitCB {
                pendingCommitCB = nil
                try waitForPriorCommitCB(cb, session: session)
            }
        }
        let dflashReady = dflashDrafter == nil || (session.dflashSlot != nil && session.dflashCacheOffset == session.position)
        let dsparkReady = dsparkDrafter == nil || (session.dsparkSlot != nil && session.dsparkCacheOffset == session.position)
        let hasValidMTPCache = (mtpCache != nil && session.mtpKV != nil)
            || ((isWeightFreeSpeculation || dflashDrafter != nil || dsparkDrafter != nil) && mtpCache == nil && session.mtpKV == nil)
        let rr = session.roundRecord
        guard (session.stochastic != nil) == stochastic,
              session.stochastic?.stream.isRoundActive != true,
              session.stochastic?.stream.drawsConsumed == (stochastic ? rr.samplerDrawIndex : nil),
              verifiedSession == nil, let verifyWindow,
              (mtpDraftEncoder != nil || echoDrafter != nil || dflashDrafter != nil || dsparkDrafter != nil || isWeightFreeSpeculation),
              dflashReady, dsparkReady, hasValidMTPCache,
              let hidden = session.trunkHidden, let pending = rr.pendingToken,
              !session.sessionFinished, !session.stateFailed, !session.verificationPending,
              session.trunkHiddenValid, session.position > 0, session.position <= ropeRows - verifyWindow.rows,
              rr.consumedCount == session.position, rr.absolutePosition == session.position, rr.outputCount == session.generated,
              maxNew > 0, session.generated <= Int.max - verifyWindow.rows,
              (0..<geometry.vocab).contains(pending), stopIDs.allSatisfy({ (0..<geometry.vocab).contains($0) }) else {
            pendingCommitCB?.waitUntilCompleted()
            throw QWEngineDriverError.state("greedy round requires a seeded boundary and valid output budget")
        }
        let activeDraftEncoder = (dflashDrafter == nil && dsparkDrafter == nil) ? mtpDraftEncoder : nil
        let count = verifyWindow.rows - 1, basePairs = rr.mtpPhysicalRows
        let nextNResolving = activeDraftEncoder != nil && echoDrafter == nil && !mtpStageProfiling
        let nextNMaxPairs = nextNResolving ? min(verifyWindow.rows, max(count, (activeDraftEncoder?.positionCapacity ?? ropeRows) - session.position)) : count
        let pairSlotsNeeded = echoDrafter != nil ? verifyWindow.rows : nextNMaxPairs
        do {
            try kvCache.validate(session.kv)
            if let mtpCache, let handle = session.mtpKV {
                try mtpCache.validate(handle)
                guard mtpCache.written(handle) == basePairs, basePairs >= 0, basePairs <= QWKVCache.maxTokensPerSeq - pairSlotsNeeded else {
                    throw QWEngineDriverError.state("greedy round cache or RoPE boundary mismatch")
                }
            }
        } catch {
            pendingCommitCB?.waitUntilCompleted()
            throw error
        }
        guard kvCache.written(session.kv) == session.position, session.position <= (activeDraftEncoder?.positionCapacity ?? ropeRows) - count else {
            pendingCommitCB?.waitUntilCompleted()
            throw QWEngineDriverError.state("greedy round cache or RoPE boundary mismatch")
        }
        let needsSparseClaim = (hcaOperator.map { $0.stateOwner !== session } ?? false)
            || (mlaOperator.map { $0.stateOwner !== session } ?? false)
            || (lsaOperator.map { $0.stateOwner !== session } ?? false)
            || (mbsaOperator.map { $0.stateOwner !== session } ?? false)
        let needsMTPCacheWait = (mtpCache != nil && session.mtpKV != nil) && (
            basePairs + pairSlotsNeeded > session.mtpKV!.tokenCapacity
            || mtpCache!.hasSharedPages(session.mtpKV!, positions: basePairs..<(basePairs + pairSlotsNeeded))
        )
        let needsTrunkCacheWait = session.position + verifyWindow.rows > session.kv.tokenCapacity
            || kvCache.hasSharedPages(session.kv, positions: session.position..<(session.position + verifyWindow.rows))
            || kvCache.hasUnallocatedPages(session.kv, positions: session.position..<(session.position + verifyWindow.rows))
        if needsSparseClaim || needsTrunkCacheWait || needsMTPCacheWait {
            try drainPendingCommitIfNeeded()
        }
        try claimMLAIndexerState(session)
        // Reserve both windows before draft execution; never grow a cache while
        // one of its command buffers is in flight.
        if session.position + verifyWindow.rows > session.kv.tokenCapacity {
            try kvCache.grow(session.kv, to: session.position + verifyWindow.rows)
            session.attnKVViews = try kvCache.layerViewsAll(session.kv, layers: attnLayers)
        }
        try kvCache.prepareWrites(session.kv, positions: session.position..<(session.position + verifyWindow.rows))
        if session.attnKVViews.isEmpty {
            session.attnKVViews = try kvCache.layerViewsAll(session.kv, layers: attnLayers)
        }
        let views: (k: QWKVLayerView, v: QWKVLayerView)?
        if let mtpCache, let handle = session.mtpKV {
            if basePairs + pairSlotsNeeded > handle.tokenCapacity {
                try mtpCache.grow(handle, to: basePairs + pairSlotsNeeded)
                session.mtpKVViews = [try mtpCache.layerViews(handle, layer: 0)]
            }
            try mtpCache.prepareWrites(handle, positions: basePairs..<(basePairs + pairSlotsNeeded))
            if let current = session.mtpKVViews.first {
                views = current
            } else {
                let v = try mtpCache.layerViews(handle, layer: 0)
                session.mtpKVViews = [v]
                views = v
            }
        } else {
            views = nil
        }
        let boundary = try greedyRoundBoundary(session)
        speculativeCoordinator.beginRoundPreparation(collectAdaptiveEntropy: mtpAdaptiveDepth && !stochastic)
        do {
            let owner = session.stochastic
            var proposals: [QWMTPGumbelProposal] = []
            var draftProposals: [QWSpeculativeRoundResolver.DraftProposal] = []
            var gpuVerifyResolution: QWGPUVerifyResolution?
            var drafts: [Int] = []
            var draftedPairs = 0
            let draftAccumulator = profileStart.map { _ in QWMTPStageAccumulator() }
            let draftCompleted: ((any MTLCommandBuffer) throws -> Void)? = draftAccumulator.map { accumulator in
                { try accumulator.record($0) }
            }
            let verifyStart = mtpStageProfiling ? DispatchTime.now().uptimeNanoseconds : nil
            let verifyAccumulator = verifyStart.map { _ in QWMTPStageAccumulator() }
            let verifyCompleted: ((any MTLCommandBuffer) throws -> Void)? = verifyAccumulator.map { accumulator in
                { try accumulator.record($0) }
            }
            let window: QWVerifyWindow
            let isUnconstrained = activeConstraint?.constraint.isUnconstrainedNow ?? true
            let asyncSpecAllowed = isUnconstrained || canSpeculateActiveConstraint
            if owner == nil, !mtpStageProfiling, !mtpVerificationProfiling, coldEvents == nil,
               !mtpAdaptiveDepth, asyncSpecAllowed, let dsparkDrafter, !dsparkDrafter.schedulerEnabled,
               count <= dsparkDrafter.config.maxDrafts, !QWDSparkDrafter.stageProfileEnabled,
               let table = mtpDeviceEmbedding {
                guard let dsSlot = session.dsparkSlot else {
                    try? drainPendingCommitIfNeeded()
                    throw QWEngineDriverError.state("DSpark slot unavailable for speculative round")
                }
                let draftPlan = try prepareAsyncDSparkOrPLDRound(
                    session: session, dsparkDrafter: dsparkDrafter, dsSlot: dsSlot,
                    pending: pending, count: count, table: table)
                let commitCB = pendingCommitCB
                pendingCommitCB = nil
                let verified = try verifyAsyncSlotDraftPlan(
                    draftPlan, session: session, pending: pending, table: table, priorCommitCB: commitCB)
                window = verified.window; drafts = verified.drafts; draftedPairs = verified.draftedPairs
            } else if owner == nil, !mtpStageProfiling, !mtpVerificationProfiling, coldEvents == nil,
               asyncSpecAllowed, let dflashDrafter,
               let table = mtpDeviceEmbedding {
                guard let dSlot = session.dflashSlot else {
                    try? drainPendingCommitIfNeeded()
                    throw QWEngineDriverError.state("DFlash slot unavailable for speculative round")
                }
                let draftPlan = try prepareAsyncDFlashOrPLDRound(
                    session: session, dflashDrafter: dflashDrafter, dSlot: dSlot,
                    pending: pending, count: count, table: table)
                let commitCB = pendingCommitCB
                pendingCommitCB = nil
                let verified = try verifyAsyncSlotDraftPlan(
                    draftPlan, session: session, pending: pending, table: table, priorCommitCB: commitCB)
                window = verified.window; drafts = verified.drafts; draftedPairs = verified.draftedPairs
            } else if !mtpStageProfiling, !mtpVerificationProfiling, coldEvents == nil,
               !mtpAdaptiveDepth, asyncSpecAllowed, dflashDrafter == nil, dsparkDrafter == nil, !isWeightFreeSpeculation,
               let mtpCache, let handle = session.mtpKV, let views,
               let table = mtpDeviceEmbedding,
               let draftPlan = try prepareAsyncNeuralDraft(
                   session: session, hidden: hidden, views: views, pending: pending,
                   count: count, basePairs: basePairs, mtpCache: mtpCache, mtpHandle: handle,
                   table: table, owner: owner) {
                let commitCB = pendingCommitCB
                pendingCommitCB = nil
                let asyncResult = try executeAsyncNeuralRound(
                    draftPlan, session: session, verifyWindow: verifyWindow,
                    pending: pending, table: table, priorCommitCB: commitCB,
                    isUnconstrained: isUnconstrained, owner: owner)
                window = asyncResult.window
                drafts = asyncResult.drafts
                draftProposals = asyncResult.draftProposals
                gpuVerifyResolution = asyncResult.gpuVerifyResolution
                draftedPairs = asyncResult.draftedPairs
            } else {
                if Self.shouldDrainPendingCommitInSyncBranch(
                    hasOwner: owner != nil,
                    adaptiveDepth: mtpAdaptiveDepth,
                    hasDeviceEmbedding: mtpDeviceEmbedding != nil,
                    hasColdEvents: coldEvents != nil,
                    hasDSpark: dsparkDrafter != nil
                ) {
                    try drainPendingCommitIfNeeded()
                }
                if let dsparkDrafter {
                    guard let dsSlot = session.dsparkSlot else {
                        try? drainPendingCommitIfNeeded()
                        throw QWEngineDriverError.state("DSpark slot unavailable for speculative round")
                    }
                    let proposed = try proposeDSparkOrPLDRound(
                        session: session, dsparkDrafter: dsparkDrafter, dsSlot: dsSlot,
                        pending: pending, count: count, owner: owner, draftCompleted: draftCompleted)
                    drafts = proposed.drafts; proposals = proposed.proposals; draftedPairs = proposed.draftedPairs
                    if session.dsparkRoundStartNs != nil {
                        session.dsparkRoundStartNs = DispatchTime.now().uptimeNanoseconds
                    }
                } else if let dflashDrafter {
                    guard let dSlot = session.dflashSlot else {
                        try? drainPendingCommitIfNeeded()
                        throw QWEngineDriverError.state("DFlash slot unavailable for speculative round")
                    }
                    let proposed = try proposeDFlashOrPLDRound(
                        session: session, dflashDrafter: dflashDrafter, dSlot: dSlot,
                        pending: pending, count: count, owner: owner, draftCompleted: draftCompleted)
                    drafts = proposed.drafts; proposals = proposed.proposals; draftedPairs = proposed.draftedPairs
                } else if let echoDrafter {
                    guard let views else {
                        try? drainPendingCommitIfNeeded()
                        throw QWEngineDriverError.state("ECHO round requires MTP KV views")
                    }
                    let proposed = try proposeEchoOrPLDRound(
                        session: session, echoDrafter: echoDrafter, hidden: hidden,
                        views: views, pending: pending, count: count, basePairs: basePairs,
                        owner: owner, draftCompleted: draftCompleted)
                    drafts = proposed.drafts; draftProposals = proposed.draftProposals; draftedPairs = proposed.draftedPairs
                } else if isWeightFreeSpeculation {
                    guard let chosen = proposeCanonicalOrFilteredDrafts(
                        constraint: activeConstraint?.constraint,
                        maxCount: count,
                        fallbackCandidates: {
                            proposeWeightFreeDrafts(session: session, pending: pending, count: count)
                        }
                    ) else {
                        try? drainPendingCommitIfNeeded()
                        throw QWEngineDriverError.state("weight-free round had no context or recycler drafts")
                    }
                    drafts = chosen
                    if owner != nil { draftProposals = chosen.map { .greedyToken($0) } }
                    draftedPairs = 0; lastRoundUsedPLD = true; mtpDraftProposals += drafts.count
                } else {
                    guard let mtpDraftEncoder, let views else {
                        try? drainPendingCommitIfNeeded()
                        throw QWEngineDriverError.state("MTP draft encoder is absent")
                    }
                    let proposed = try proposeNextNOrPLDRound(
                        session: session, draftEncoder: mtpDraftEncoder, hidden: hidden,
                        views: views, pending: pending, count: count, basePairs: basePairs,
                        owner: owner, draftCompleted: draftCompleted)
                    drafts = proposed.drafts; draftProposals = proposed.draftProposals; draftedPairs = proposed.draftedPairs
                }
                if let mtpCache, let handle = session.mtpKV {
                    mtpCache.markWritten(handle, upTo: basePairs + draftedPairs)
                }
                if canOverlapPriorCommit {
                    let embedTable = mtpDeviceEmbedding
                    let commitCB = pendingCommitCB
                    pendingCommitCB = nil
                    window = try verifyTokensCore(
                        session, explicitTokens: [pending] + drafts, rowCount: 1 + drafts.count,
                        pendingToken: nil, useDeviceEmbed: embedTable,
                        priorCommitCB: commitCB, inFlightDraftCBs: [], completed: verifyCompleted).window
                } else {
                    try drainPendingCommitIfNeeded()
                    window = try verifyTokensLocked(session, tokens: [pending] + drafts, completed: verifyCompleted)
                }
            }
            var rawDecision: QWGreedyRoundDecision
            let evidence: QWStochasticRoundEvidence?
            let rawTargetsForLease: [Int]?
            let margin: Float = lastRoundUsedPLD ? 0 : effectiveRelaxedMargin
            let minProb: Float = margin > 0 ? effectiveRelaxedMinProb : 0
            let useConstrainedResolver = !isUnconstrained
                && (activeConstraint.map { canSpeculateConstraint($0.constraint) } ?? false)
            if let owner {
                let resolvedActive = useConstrainedResolver ? activeConstraint : nil
                if !proposals.isEmpty {
                    let resolved = try resolveGumbelRoundDecision(
                        window: window, session: session, owner: owner, pending: pending,
                        drafts: drafts, proposals: proposals, maxNew: maxNew, stopIDs: stopIDs,
                        active: resolvedActive, margin: margin, drawStart: boundary.record.samplerDrawIndex)
                    rawDecision = resolved.decision
                    preparedStochasticWitness = resolved.witness
                    evidence = resolved.evidence
                } else {
                    let effProposals = draftProposals.count == drafts.count
                        ? draftProposals : drafts.map { .greedyToken($0) }
                    let resolved = try resolveStochasticViaUnifiedResolver(
                        window: window, session: session, owner: owner, pending: pending,
                        drafts: drafts, proposals: effProposals, gpuResolution: gpuVerifyResolution,
                        maxNew: maxNew, stopIDs: stopIDs, constraint: activeConstraint?.constraint,
                        activeSampler: resolvedActive?.sampler, margin: margin, minProb: minProb)
                    rawDecision = resolved.decision
                    preparedStochasticWitness = resolved.witness
                    evidence = nil
                }
                rawTargetsForLease = nil
            } else if useConstrainedResolver, let active = activeConstraint {
                let resolved = try resolveConstrainedGreedyRound(
                    window: window, session: session, pending: pending,
                    drafts: drafts, maxNew: maxNew, stopIDs: stopIDs,
                    active: active, margin: margin, minProb: minProb)
                rawDecision = resolved.decision; rawTargetsForLease = resolved.targets; evidence = nil
            } else {
                let withTargets = try window.greedyDecisionWithTargets(
                    drafts: drafts, maxNew: maxNew, stopIDs: stopIDs,
                    activeRows: drafts.count + 1, relaxedMargin: margin, relaxedMinProb: minProb)
                rawDecision = withTargets.decision; rawTargetsForLease = withTargets.targets; evidence = nil
            }
            if owner == nil, !useConstrainedResolver,
               let boundaryTok = activeConstraint?.constraint.speculationBoundaryToken,
               let idx = rawDecision.tokens.firstIndex(of: boundaryTok),
               idx + 1 < rawDecision.tokens.count {
                rawDecision = QWGreedyRoundDecision(
                    acceptedDrafts: min(rawDecision.acceptedDrafts, idx),
                    tokens: Array(rawDecision.tokens.prefix(idx + 1)),
                    stopped: false, stoppedToken: nil)
            }
            let decision = rawDecision
            let lease = QWGreedyRoundLease(boundary: boundary, decision: decision,
                                           stochasticEvidence: evidence, rawTargets: rawTargetsForLease)
            try speculativeCoordinator.recordPendingProfile(
                profileStart: profileStart, verifyStart: verifyStart,
                draftAccumulator: draftAccumulator, verifyAccumulator: verifyAccumulator,
                isWeightFree: isWeightFreeSpeculation)
            preparedGreedyLease = lease
            return lease.prepared
        } catch {
            // A submitted prefix may have changed pair/trunk bytes. Without a
            // completed verified lease, do not expose an apparently retryable round.
            try? drainPendingCommitIfNeeded()
            session.stateFailed = true
            session.stochastic?.stream.cancelActiveRound()
            if let owner = session.stochastic { session.roundRecord.samplerDrawIndex = owner.stream.drawsConsumed }
            throw error
        }
    }

    func commitPreparedVerification(_ session: QWEngineSession, round: QWPreparedGreedyRound,
                                            stochastic: Bool, drain: Bool = true,
                                            permit: QWGenerationPermit?) throws {
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate(permit)
        let profileStart = mtpStageProfiling ? DispatchTime.now().uptimeNanoseconds : nil
        let hasValidMTPCache = (mtpCache != nil && session.mtpKV != nil)
            || ((isWeightFreeSpeculation || dflashDrafter != nil || dsparkDrafter != nil) && mtpCache == nil && session.mtpKV == nil)
        guard (preparedStochasticWitness != nil) == stochastic,
              (session.stochastic != nil) == stochastic,
              verifiedSession === session, session.verificationPending,
              !session.stateFailed, !session.sessionFinished,
              let verifyWindow, let gdnReplayCheckpoint, hasValidMTPCache,
              let lease = preparedGreedyLease,
              let trunkHidden = session.trunkHidden, (2...verifyWindow.rows).contains(verifiedInputs.count),
              verifiedInputs.first == session.roundRecord.pendingToken else {
            throw QWEngineDriverError.state("greedy commit requires a live verified draft round")
        }
        try lease.validate(round, current: greedyRoundBoundary(session))
        try preparedStochasticWitness?.validate(session)
        guard !mtpStageProfiling || pendingMTPProfile != nil else {
            throw QWEngineDriverError.state("MTP profile is missing its prepared stages")
        }
        defer {
            session.verificationPending = false
            speculativeCoordinator.clearActiveVerification(matching: session)
        }
        let decision = round.decision
        let drafts = Array(verifiedInputs.dropFirst())
        if !stochastic {
            let commitMargin: Float = lastRoundUsedPLD ? 0 : effectiveRelaxedMargin
            let commitMinProb: Float = commitMargin > 0 ? effectiveRelaxedMinProb : 0
            let rawTargets: [Int]
            if let cachedTargets = lease.rawTargets, cachedTargets.count == verifiedInputs.count {
                rawTargets = cachedTargets
            } else {
                let (full, computedTargets) = try verifyWindow.greedyDecisionWithTargets(
                    drafts: drafts, maxNew: verifiedInputs.count, activeRows: verifiedInputs.count,
                    relaxedMargin: commitMargin, relaxedMinProb: commitMinProb)
                guard decision.acceptedDrafts == full.acceptedDrafts,
                      decision.tokens == Array(full.tokens.prefix(decision.tokens.count)),
                      decision.tokens.count <= full.tokens.count else {
                    throw QWEngineDriverError.state("greedy commit decision does not match verified target rows")
                }
                rawTargets = computedTargets
            }
            if commitMargin == 0, (activeConstraint?.constraint.isUnconstrainedNow ?? true) {
                let pendingInHistory = session.tokenHistory.count > session.position
                    && session.tokenHistory.last == verifiedInputs[0]
                let prevBeforePending: Int? = pendingInHistory
                    ? (session.tokenHistory.count >= 2 ? session.tokenHistory[session.tokenHistory.count - 2] : nil)
                    : session.tokenHistory.last
                for row in (0..<min(verifiedInputs.count, rawTargets.count)).reversed() {
                    let prev = (row == 0) ? prevBeforePending : verifiedInputs[row - 1]
                    session.tokenRecycler.recordTransition(
                        previous: prev, current: verifiedInputs[row], next: rawTargets[row])
                }
            }
        }
        guard session.generated <= Int.max - decision.tokens.count else {
            throw QWEngineDriverError.state("round output count overflows")
        }
        try kvCache.validate(session.kv)
        if let mtpCache, let mtpHandle = session.mtpKV {
            try mtpCache.validate(mtpHandle)
        }
        let basePairs = session.roundRecord.mtpPhysicalRows
        let usesManagedPairs = echoDrafter != nil || (mtpDraftEncoder != nil && dflashDrafter == nil && dsparkDrafter == nil
            && (lastRoundUsedPLD || lastDraftedPairs < drafts.count || (mtpQuadraticSeeding && !mtpStageProfiling)))
        let draftedPairs = (echoDrafter != nil || (mtpDraftEncoder != nil && dflashDrafter == nil && dsparkDrafter == nil))
            ? lastDraftedPairs : drafts.count
        let maxAddedPairs = echoDrafter != nil ? verifiedInputs.count
            : (usesManagedPairs && mtpDraftEncoder != nil
                ? min(verifiedInputs.count, max(drafts.count, mtpDraftEncoder!.positionCapacity - session.position))
                : drafts.count)
        guard kvCache.written(session.kv) == session.position + verifiedInputs.count else {
            throw QWEngineDriverError.state("greedy commit requires complete trunk and draft pair windows")
        }
        if let mtpCache, let mtpHandle = session.mtpKV {
            guard basePairs >= 0, basePairs <= QWKVCache.maxTokensPerSeq - maxAddedPairs,
                  mtpCache.written(mtpHandle) == basePairs + draftedPairs else {
                throw QWEngineDriverError.state("greedy commit requires complete trunk and draft pair windows")
            }
        }
        let retained = decision.retainedRows
        let retainedPairs: Int
        if echoDrafter != nil {
            retainedPairs = retained
        } else if usesManagedPairs, let mtpDraftEncoder {
            retainedPairs = min(retained, max(0, mtpDraftEncoder.positionCapacity - session.position))
        } else if mtpCache != nil {
            retainedPairs = try decision.retainedPairRows(drafts: drafts.count)
        } else {
            retainedPairs = 0
        }
        guard let cb = hal.makeUnretainedCommandBuffer() else { throw QWEngineDriverError.state("greedy commit command buffer") }
        if let echoDrafter {
            try encodeEchoCommitResolution(cb, session: session, echoDrafter: echoDrafter,
                                           window: verifyWindow, trunkHidden: trunkHidden,
                                           basePairs: basePairs, retained: retained,
                                           usedPLD: lastRoundUsedPLD)
        } else if usesManagedPairs, let mtpDraftEncoder {
            try encodeNextNCommitResolution(cb, session: session, draft: mtpDraftEncoder,
                                            window: verifyWindow, trunkHidden: trunkHidden,
                                            basePairs: basePairs, retained: retained,
                                            usedPLD: lastRoundUsedPLD)
        }
        try verifyWindow.encodeRetainedPrefix(cb, hal: hal, checkpoint: gdnReplayCheckpoint, decoder: gdnDecode,
                                              rec: session.recSlab, ring: session.ringSlab,
                                              trunkHidden: trunkHidden, activeRows: verifiedInputs.count,
                                              retaining: retained)
        if mtpStageProfiling && retained == 0 && geometry.gdnLayerCount == 0 {
            let src4 = QWTensorValue(region: verifyWindow.hidden.region, offset: verifyWindow.hidden.offset,
                                     shape: QWTensorShape([2], .f16))
            let dst4 = QWTensorValue(region: verifyWindow.embeddings.region, offset: verifyWindow.embeddings.offset,
                                     shape: QWTensorShape([2], .f16))
            if let enc = cb.makeComputeCommandEncoder() {
                defer { enc.endEncoding() }
                try QWVerifyWindow.encodeCopy(enc, hal: hal, source: src4, destination: dst4)
            }
        }
        if let dflashDrafter, let dSlot = session.dflashSlot, retained > 0 {
            try dflashDrafter.encodeContextUpdate(cb, slot: dSlot, startPosition: session.position, totalCount: retained)
        }
        if let dsparkDrafter, let dsSlot = session.dsparkSlot, retained > 0 {
            try dsparkDrafter.encodeContextIngestCB(cb, driver: self, slot: dsSlot, pos0: session.position, rows: retained)
        }
        if !mtpStageProfiling && !drain {
            session.lastCommitCB = cb
            cb.addCompletedHandler { [weak self, weak session] buffer in
                if buffer.status != .completed {
                    self?.recordAsyncCommandBufferFailure(buffer, session: session)
                }
            }
        }
        cb.commit()
        if mtpStageProfiling || drain {
            cb.waitUntilCompleted()
            guard cb.status == .completed else {
                recordCommandBufferError(for: cb)
                session.stateFailed = true
                throw QWEngineDriverError.state("greedy commit GPU failure: \(String(describing: cb.error))")
            }
        }
        let usedPLDThisRound = lastRoundUsedPLD
        let resolutionGPU: Double?
        do {
            resolutionGPU = mtpStageProfiling ? try QWMTPStageTiming.gpuDurationMs(cb) : nil
            let trunkRollback = verifiedInputs.count - retained
            if trunkRollback > 0 {
                try kvCache.rollback(session.kv, by: trunkRollback)
                session.attnKVViews = try kvCache.layerViewsAll(session.kv, layers: attnLayers)
                mlaOperator?.truncatePosition(session.position + retained)
                hcaOperator?.truncatePosition(session.position + retained)
            }
            if geometry.hasAnySlidingWindow {
                try kvCache.recycleExpiredWindowPages(
                    session.kv,
                    currentPosition: session.position + retained,
                    safetyMargin: geometry.descriptor.maxSpeculativeDrafts + QWKVCache.pageTokens)
            }
            if let mtpCache, let mtpHandle = session.mtpKV {
                if echoDrafter != nil || usesManagedPairs {
                    try finalizeMTPPairCursor(session: session, mtpCache: mtpCache, mtpHandle: mtpHandle,
                                              basePairs: basePairs, draftedPairs: draftedPairs,
                                              retainedPairs: retainedPairs)
                } else {
                    let pairRollback = drafts.count - retainedPairs
                    if pairRollback > 0 {
                        try mtpCache.rollback(mtpHandle, by: pairRollback)
                        session.mtpKVViews = [try mtpCache.layerViews(mtpHandle, layer: 0)]
                    }
                }
            }
            lastRoundUsedPLD = false
            lastDraftedPairs = 0
        } catch {
            // GPU state has already resolved; partly changed host tables cannot
            // be retried as a fresh round.
            session.stateFailed = true
            throw error
        }
        let profile: QWMTPRoundProfile?
        do {
            profile = try speculativeCoordinator.buildCommittedProfile(
                profileStart: profileStart, resolutionGPU: resolutionGPU,
                retainedRows: retained, drafts: drafts,
                acceptedDrafts: decision.acceptedDrafts, backend: mtpBackend.rawValue)
        } catch {
            session.stateFailed = true
            throw error
        }
        if let profile, Self.decodeTraceEnabled {
            FileHandle.standardError.write(Data(String(
                format: "[mtp-round] pos=%d retained=%d/%d draft_gpu=%.2fms verify_gpu=%.2fms res_gpu=%.2fms\n",
                session.position, retained - 1, drafts.count,
                profile.draft.gpuMs, profile.verification.gpuMs, profile.resolution.gpuMs).utf8))
        }
        session.position += retained
        if dflashDrafter != nil, session.dflashSlot != nil { session.dflashCacheOffset = session.position }
        if dsparkDrafter != nil, session.dsparkSlot != nil { session.dsparkCacheOffset = session.position }
        recordDSparkRoundVerifyLatency(session, verifyRows: verifiedInputs.count)
        session.generated += decision.tokens.count
        session.roundRecord.consumedCount = session.position
        session.roundRecord.absolutePosition = session.position
        session.roundRecord.outputCount = session.generated
        session.roundRecord.mtpPhysicalRows = basePairs + retainedPairs
        mtpDraftAccepted += decision.acceptedDrafts
        if dflashDrafter != nil {
            if !mtpStageProfiling && !usedPLDThisRound {
                let roundMs = session.dflashRoundStartNs.map { Double(DispatchTime.now().uptimeNanoseconds - $0) / 1_000_000 }
                session.dflashPolicy.observeRound(
                    draftCount: drafts.count, accepted: decision.acceptedDrafts, roundMs: roundMs,
                    isGreedy: !stochastic, hasExplicitSeed: session.stochastic?.config.hasExplicitSeed ?? false)
            }
            session.dflashRoundStartNs = nil
        }
        if stochastic, (activeConstraint?.constraint.isUnconstrainedNow ?? true) {
            for tok in decision.tokens {
                if let prev1 = session.tokenHistory.last {
                    let prev2 = session.tokenHistory.count >= 2 ? session.tokenHistory[session.tokenHistory.count - 2] : nil
                    session.tokenRecycler.recordTransition(previous: prev2, current: prev1, next: tok)
                }
                session.tokenHistory.append(tok)
            }
        } else {
            session.tokenHistory.append(contentsOf: decision.tokens)
        }
        if let active = activeConstraint {
            for tok in decision.tokens { try active.constraint.accept(tok) }
        }
        if let owner = session.stochastic { session.roundRecord.samplerDrawIndex = owner.stream.drawsConsumed }
        if let pending = decision.pendingToken { session.roundRecord.pendingToken = pending }
        if retained > 0 { session.roundRecord.trunkHiddenSeed = nil } // GPU carry is authoritative.
        session.verificationPending = false
        speculativeCoordinator.finishCommit(profile: profile)
    }
}
