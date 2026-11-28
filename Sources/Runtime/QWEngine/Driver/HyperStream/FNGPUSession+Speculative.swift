// FNGPUSession+Speculative.swift — Native MTP speculative step, GPU LC verification, and chain drafting.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWHead
import QWMTP

enum FNMTPChain {
    /// Chained single-row MTP passes after each verify round (d2/d3 drafts); capped by maxDrafts - 1.
    static let passes = Int(ProcessInfo.processInfo.environment["QWG_MTP_CHAIN"] ?? "") ?? 2
    static let debug = ProcessInfo.processInfo.environment["QWG_CHAIN_DEBUG"] != nil
}

extension FNGPUSessionData {
    static func precomputeLoopRepetitionByRow(
        recentGenerated: [Int], seenTokenList: [Int], drafts: [Int], vocab: Int
    ) -> [[Int]] {
        var penWindow = recentGenerated, activePen = seenTokenList
        var out: [[Int]] = []
        out.reserveCapacity(1 + drafts.count)
        for row in 0...drafts.count {
            out.append(activePen)
            if row < drafts.count {
                QWTrunkRuntimeSupport.recordGeneratedTokenForPenalty(
                    drafts[row], vocab: vocab, window: repetitionPenaltyWindow,
                    recentGenerated: &penWindow, activeLoopTokens: &activePen
                )
            }
        }
        return out
    }
}

extension QWHyperStreamRuntime {
    static func makeMTPSelectionParams(
        config: QWSamplerConfig, vocab: Int, repetitionPenalty: Float, seed: UInt64, philoxOffsetBase: UInt64
    ) -> QWStochasticSelectionParams {
        QWStochasticSelectionParams(
            temperature: config.temperature, topK: config.topK > 0 ? min(config.topK, vocab) : vocab,
            topP: config.topP, repetitionPenalty: repetitionPenalty, seed: seed, philoxOffsetBase: philoxOffsetBase
        )
    }

    func speculativeStep(
        _ session: inout FNGPUSessionData, token: Int, vStack: FNGPUDecodeStack, mtp: FNGPUMTP
    ) throws -> Int {
        let prevCount = session.history.count, cfg = session.sampler.config
        let isGreedy = cfg.isGreedy, hasExplicitSeed = cfg.hasExplicitSeed
        let trackRep = cfg.repetitionPenalty != 1.0
        session.history.append(Int64(token))
        let (drafts, stepMTP) = draftsFor(&session, mtp: mtp)
        let dCount = drafts.count, activeRows = 1 + dCount
        for d in drafts { session.history.append(Int64(d)) }
        if residentKVHistory.count == prevCount {
            residentKVHistory.append(Int64(token))
            for d in drafts { residentKVHistory.append(Int64(d)) }
        }
        highWaterPos = max(highWaterPos, session.history.count)

        let isUnconstrained = session.constraint?.isUnconstrainedNow ?? true
        let repByRow = (!isGreedy && trackRep)
            ? FNGPUSessionData.precomputeLoopRepetitionByRow(
                recentGenerated: session.recentGeneratedTokens, seenTokenList: session.seenTokenList,
                drafts: drafts, vocab: vocab)
            : []
        let stream: QWMTPRandomStream? = isGreedy ? nil : QWMTPRandomStream(cloning: try session.ensureMTPStreamState())
        let hasSparseProps = !isGreedy && dCount > 0 && session.mtpSourcedDrafts == dCount
            && session.mtpProposals.count >= dCount
            && zip(session.mtpProposals.prefix(dCount), drafts).allSatisfy { $0.0.isSparse && $0.0.token == $0.1 }
        let useGPUVerify = isUnconstrained && hasSparseProps && (stepMTP != nil)

        var draftReq: FNGPUStochasticDraftRequest? = nil
        var verifyReq: FNGPUStochasticVerifyRequest? = nil
        if useGPUVerify, let stream, let stepMTP {
            let dVocab = stepMTP.mtpHead.vocab
            let base0 = try stream.reserveDraws(UInt64(dVocab))
            draftReq = FNGPUStochasticDraftRequest(
                params: Self.makeMTPSelectionParams(
                    config: cfg, vocab: dVocab, repetitionPenalty: 1.0, seed: stream.seed, philoxOffsetBase: base0))
            let lcWin = try stream.beginLCRound(drafts: dCount, maxDrafts: max(15, dCount))
            verifyReq = FNGPUStochasticVerifyRequest(
                draftProposals: stepMTP.mtpHead.selector, draftCount: dCount,
                params: Self.makeMTPSelectionParams(
                    config: cfg, vocab: vocab, repetitionPenalty: cfg.repetitionPenalty,
                    seed: stream.seed, philoxOffsetBase: lcWin.start),
                repetitionHistoryByRow: repByRow)
        }
        let stepT0 = Date()
        do {
            _ = try vStack.forwardRows(
                history: session.history, embedTable: embedTable, head: head,
                readStream: false, verifyAllRows: true, mtp: stepMTP, activeRows: activeRows,
                relaxedMargin: (isGreedy && isUnconstrained) ? session.relaxedMargin : 0,
                relaxedRows: session.mtpSourcedDrafts + 1, relaxedMinProb: session.relaxedMinProb,
                stochasticVerify: verifyReq, mtpStochasticDraft: draftReq)
        } catch { stream?.cancelActiveRound(); throw error }
        let verifyMs = Date().timeIntervalSince(stepT0) * 1000.0
        if !isGreedy && !hasExplicitSeed {
            session.chainPolicy.observeVerifyRound(
                rows: activeRows, ms: verifyMs, isGreedy: isGreedy, hasExplicitSeed: hasExplicitSeed)
        }
        session.specProposals += dCount

        let (tok0, accepted, committedTokens, needsResync) = try resolveSpeculativeRound(
            &session, drafts: drafts, stepMTP: stepMTP, stream: stream, verifyReq: verifyReq,
            activeRows: activeRows, isUnconstrained: isUnconstrained, isGreedy: isGreedy,
            trackRep: trackRep, repByRow: repByRow)

        let rejected = dCount - accepted
        if rejected > 0 {
            session.history.removeLast(rejected)
            if residentKVHistory.count == prevCount + activeRows { residentKVHistory.removeLast(rejected) }
            vStack.rollbackVerifyRows(rejectedCount: rejected)
            session.mtpActive = true
        }
        session.specAccepted += accepted
        session.chainPolicy.observeAcceptance(draftCount: session.mtpSourcedDrafts, accepted: accepted)

        let (chainGPUMs, localChainMs) = try updatePostVerifyMTPDrafts(
            &session, stepMTP: stepMTP, stream: stream, vStack: vStack,
            committedTokens: committedTokens, prevCount: prevCount, activeRows: activeRows,
            accepted: accepted, needsResync: needsResync, isGreedy: isGreedy, hasExplicitSeed: hasExplicitSeed)
        if let stream { session.mtpStreamState = try stream.frozenState() }

        session.headToken = -1
        let roundMs = Date().timeIntervalSince(stepT0) * 1000.0
        let divisor = Double(1 + session.pendingTokens.count)
        session.stepMs = roundMs / divisor
        session.lastStepGPUMillis = (vStack.lastGPUMillis + chainGPUMs) / divisor
        session.roundTokenCounts.append(1)
        if FNMTPChain.debug {
            let msg = "R rows=\(activeRows) acc=\(accepted) gpu=\(String(format: "%.2f", vStack.lastGPUMillis)) wall=\(String(format: "%.2f", roundMs)) chain=\(String(format: "%.2f", localChainMs))\n"
            FileHandle.standardError.write(msg.data(using: .utf8)!)
        }
        return tok0
    }

    private func resolveSpeculativeRound(
        _ session: inout FNGPUSessionData, drafts: [Int], stepMTP: FNGPUMTP?, stream: QWMTPRandomStream?,
        verifyReq: FNGPUStochasticVerifyRequest?, activeRows: Int, isUnconstrained: Bool,
        isGreedy: Bool, trackRep: Bool, repByRow: [[Int]]
    ) throws -> (tok0: Int, accepted: Int, tokens: [Int], needsResync: Bool) {
        let dCount = drafts.count
        if isGreedy {
            let v = head.argmaxTokens(count: activeRows)
            session.mtpDraft = stepMTP?.mtpHead.argmaxToken() ?? session.mtpDraft
            let rowMargin: Float = (!isUnconstrained && !session.periodicDrafts) ? session.relaxedMargin : 0
            let canSpecConstrained = session.constraint.map {
                $0.isUnconstrainedNow || (QWConstrainedSpeculationGate.isEnabled && $0.allowsSpeculation)
            } ?? true
            let probeClone = session.constraint?.makeSpeculativeProbe()
            let roundConstraint = probeClone ?? session.constraint
            let res = try QWSpeculativeTokenResolver.resolveVerifiedRound(
                drafts: drafts, targetArgmaxes: v, vocabulary: vocab, isGreedy: true,
                isUnconstrainedNow: { roundConstraint?.isUnconstrainedNow ?? true },
                boundaryToken: roundConstraint?.speculationBoundaryToken,
                canSpeculateConstrained: canSpecConstrained,
                isRepetitionBlocked: { session.seenTokens.contains($0) },
                acceptConstrained: { try roundConstraint?.accept($0) },
                sampleRow: { row, argmaxTok in
                    try sampleFromHead(argmaxToken: argmaxTok, seenTokens: session.seenTokens,
                                       seenTokenList: session.seenTokenList, sampler: session.sampler, row: row,
                                       constraint: roundConstraint, draftToken: row < dCount ? drafts[row] : nil,
                                       relaxedMargin: row < session.mtpSourcedDrafts ? rowMargin : 0,
                                       relaxedMinProb: session.relaxedMinProb)
                },
                onCommittedToken: { if trackRep { session.recordSeenToken($0, vocab: vocab) } }
            )
            let tok0 = res.firstToken
            if probeClone != nil { try session.constraint?.accept(tok0) }
            if !res.trailingTokens.isEmpty {
                session.pendingTokens.append(contentsOf: res.trailingTokens)
                if probeClone != nil { session.pendingConstraintAcceptCount += res.trailingTokens.count }
            }
            return (tok0, res.acceptedDrafts, res.tokens, res.needsResync(targetArgmaxes: v))
        }
        guard let stream else { throw FNGPUError.notHQ4("stochastic MTP random stream") }
        let gpuRes = verifyReq != nil ? try head.selector.readGPUVerifyResolution(draftCount: dCount) : nil
        let proposals = try buildStochasticDraftProposals(
            session: session, drafts: drafts, skipDetails: gpuRes?.resolved == true)
        let logitsPtr = arena.cpuPointer(of: head.logits).assumingMemoryBound(to: Float.self)
        let roundSampler = session.sampler
        let res = try QWSpeculativeRoundResolver.resolve(
            draftProposals: proposals, gpuResolution: gpuRes, stream: stream, isGreedy: false,
            maxNew: activeRows, stopIDs: [], constraint: session.constraint, margin: 0, minProbability: 0
        ) { row, _, probe in
            let rowPtr = UnsafeBufferPointer(start: logitsPtr.advanced(by: row * vocab), count: vocab)
            let rep = trackRep ? repByRow[row] : []
            if let probe, try probe.writeAllowedTokens(into: &roundSampler.allowedScratch) {
                return try roundSampler.constrainedDistributionFromScratch(logits: rowPtr, repetitionHistory: rep)
            }
            return try roundSampler.distribution(logits: rowPtr, repetitionHistory: rep)
        }
        let tok0 = res.firstToken ?? 0
        if session.constraint != nil {
            try session.constraint?.accept(tok0)
            session.pendingConstraintAcceptCount += res.trailingTokens.count
        }
        if !res.trailingTokens.isEmpty { session.pendingTokens.append(contentsOf: res.trailingTokens) }
        if trackRep { for t in res.tokens { session.recordSeenToken(t, vocab: vocab) } }
        return (tok0, res.acceptedDrafts, res.tokens, !res.usedGPUFastPath || res.constraintBoundaryReached)
    }

    private func buildStochasticDraftProposals(
        session: FNGPUSessionData, drafts: [Int], skipDetails: Bool
    ) throws -> [QWSpeculativeRoundResolver.DraftProposal] {
        if skipDetails { return Array(repeating: .greedyToken(0), count: drafts.count) }
        return try drafts.indices.map { i in
            guard i < session.mtpSourcedDrafts, i < session.mtpProposals.count else { return .greedyToken(drafts[i]) }
            let sp = session.mtpProposals[i]
            if sp.isSparse { return .sparse(token: sp.token, ids: sp.ids, probs: sp.probs) }
            if i == 0, let ovLogits = session.mtpOverflowLogits {
                let dist = try QWSampler(config: QWSamplerConfig(
                    temperature: sp.temperature, topK: ovLogits.count, topP: 1.0, uniformVersion: .upper24
                )).distribution(logits: ovLogits)
                return .distribution(token: sp.token, distribution: dist)
            }
            return .greedyToken(drafts[i])
        }
    }

    private func updatePostVerifyMTPDrafts(
        _ session: inout FNGPUSessionData, stepMTP: FNGPUMTP?, stream: QWMTPRandomStream?,
        vStack: FNGPUDecodeStack, committedTokens: [Int], prevCount: Int, activeRows: Int,
        accepted: Int, needsResync: Bool, isGreedy: Bool, hasExplicitSeed: Bool
    ) throws -> (chainGPUMs: Double, localChainMs: Double) {
        session.mtpChain = []; session.mtpProposals = []; session.mtpOverflowLogits = nil
        guard let stepMTP else { return (0, 0) }
        let dVocab = stepMTP.mtpHead.vocab, cfg = session.sampler.config
        var prop0: QWSparseProposalBuffer? = nil
        if !isGreedy && !needsResync {
            let p0 = try stepMTP.mtpHead.selector.readSparseProposal(row: 0)
            prop0 = p0
            if !p0.isSparse { session.mtpOverflowLogits = stepMTP.mtpHead.readLogits() }
        }
        let mtpMax = min(FNGPUSessionData.defaultMTPDrafts, session.maxDrafts)
        let allowChain = isGreedy || needsResync || (prop0?.isSparse ?? false)
        let passes = (FNMTPChain.passes > 0 && mtpMax > 1 && session.mtpActive && !session.periodicDrafts && allowChain)
            ? session.chainPolicy.selectChainPasses(maxPasses: min(FNMTPChain.passes, mtpMax - 1)) : 0
        guard passes > 0 || needsResync else {
            if let prop0 {
                session.mtpDraft = prop0.token; session.mtpChain = [prop0.token]
                session.mtpChainPos = prevCount + accepted + 2; session.mtpProposals = [prop0]
            }
            return (0, 0)
        }
        let chainT0 = Date()
        var syncReq: FNGPUStochasticDraftRequest? = nil
        var chainParams: [QWStochasticSelectionParams]? = nil
        if let stream {
            if needsResync {
                syncReq = FNGPUStochasticDraftRequest(
                    params: Self.makeMTPSelectionParams(
                        config: cfg, vocab: dVocab, repetitionPenalty: 1.0,
                        seed: stream.seed, philoxOffsetBase: try stream.reserveDraws(UInt64(dVocab))),
                    proposalSlotBase: 0
                )
            }
            if passes > 0 {
                chainParams = try (0..<passes).map { _ in
                    Self.makeMTPSelectionParams(
                        config: cfg, vocab: dVocab, repetitionPenalty: 1.0,
                        seed: stream.seed, philoxOffsetBase: try stream.reserveDraws(UInt64(dVocab)))
                }
            }
        }
        guard let pair = ctx.hal.begin(.concurrent) else { throw FNGPUError.notHQ4("chain command buffer") }
        let cb = pair.0; var enc = pair.1
        let firstStream: QWTensorValue
        if !needsResync {
            firstStream = stepMTP.lastProjectedStream(verifyRows: activeRows)
        } else {
            let syncRows = isGreedy ? 1 : max(1, min(committedTokens.count, 8))
            if isGreedy {
                stepMTP.writeToken(Int32(committedTokens.last ?? 0))
            } else {
                stepMTP.writeTokens(Array(committedTokens.prefix(syncRows)).map(Int32.init))
            }
            let byteOff = isGreedy ? accepted * vStack.width * 4 : 0
            let selStream = vStack.lastStreamTensor.slice(byteOffset: byteOff, count: syncRows * vStack.width, dtype: .f32)
            let pos0 = isGreedy ? (prevCount + accepted) : prevCount
            try stepMTP.encode(
                &enc, cb: cb, streamPrev: selStream,
                tokens: stepMTP.tokensBuf.slice(byteOffset: 0, count: syncRows, dtype: .i32),
                pos: pos0, rows: syncRows, projectRows: 1, stochasticDraft: syncReq)
            enc.memoryBarrier(scope: .buffers)
            firstStream = stepMTP.lastProjectedStream(verifyRows: syncRows, selectedRow: syncRows - 1)
        }
        if passes > 0 {
            try stepMTP.encodeChain(&enc, cb: cb, firstStream: firstStream,
                                    startPos: prevCount + accepted + 1, passes: passes,
                                    stochasticParams: chainParams)
        }
        ctx.hal.end(cb, enc); cb.waitUntilCompleted()
        if let err = cb.error { throw err }
        let chainGPUMs = max(0, (cb.gpuEndTime - cb.gpuStartTime) * 1000.0)
        if !isGreedy {
            var sparseList: [QWSparseProposalBuffer] = []
            for slot in 0..<(1 + passes) {
                let sp = (slot == 0 && !needsResync && prop0 != nil)
                    ? prop0! : try stepMTP.mtpHead.selector.readSparseProposal(row: slot)
                if sp.isSparse { sparseList.append(sp) }
                else if slot == 0 && passes == 0 { session.mtpOverflowLogits = stepMTP.mtpHead.readLogits(); sparseList.append(sp) }
                else { break }
            }
            if !sparseList.isEmpty {
                let toks = sparseList.map(\.token)
                session.mtpDraft = toks[0]; session.mtpChain = toks
                session.mtpChainPos = prevCount + accepted + 2; session.mtpProposals = sparseList
            } else {
                session.mtpDraft = -1; session.mtpActive = false
            }
        } else if passes > 0 {
            let chain = stepMTP.readChain(count: passes + 1)
            session.mtpDraft = chain[0]; session.mtpChain = chain; session.mtpChainPos = prevCount + accepted + 2
        } else {
            session.mtpDraft = stepMTP.mtpHead.argmaxToken()
        }
        let chainMs = Date().timeIntervalSince(chainT0) * 1000.0
        if passes > 0 && !isGreedy && !hasExplicitSeed {
            session.chainPolicy.observeChain(passes: passes, ms: chainMs, isGreedy: isGreedy, hasExplicitSeed: hasExplicitSeed)
        }
        return (chainGPUMs, chainMs)
    }

    /// Drafts 1..3 tokens using MTP or the unified context-copy drafter (`QWContextCopyDrafter`).
    func draftsFor(_ session: inout FNGPUSessionData, mtp: FNGPUMTP) -> ([Int], FNGPUMTP?) {
        let maxD = min(session.maxDrafts, FNGPUSessionData.draftCapacity)
        if maxD > 0, QWConstrainedSpeculationGate.isEnabled,
           let constraint = session.constraint, !constraint.isUnconstrainedNow {
            let canonical = constraint.proposeCanonicalDrafts(maxCount: maxD)
            if !canonical.isEmpty {
                session.mtpSourcedDrafts = 0; session.periodicDrafts = true
                return (canonical, mtp)
            }
        }
        let proposal = QWContextCopyDrafter.proposeDrafts(
            history: session.history, mode: session.draftMode,
            maxDrafts: session.maxDrafts, draftCapacity: FNGPUSessionData.draftCapacity,
            mtpDraft: session.mtpDraft, mtpChain: session.mtpChain,
            mtpChainPos: session.mtpChainPos, mtpActive: session.mtpActive,
            relaxedMargin: session.relaxedMargin, quadraticSeeding: session.quadraticSeeding
        )
        session.mtpActive = proposal.mtpActive
        session.mtpSourcedDrafts = proposal.mtpSourcedDrafts
        session.periodicDrafts = proposal.periodicDrafts
        return (proposal.drafts, proposal.runMTPOnVerify ? mtp : nil)
    }
}
