// QWInkling2Runtime+Speculative.swift — Pipelined PLD + DSpark linear/tree speculative decode loop.
import Foundation
import Metal
import QWCore
import QWHead

extension QWInkling2Runtime {
    struct SpeculativeStats {
        var label: String = "dspark"
        var stepTokenCounts: [Int] = []
        var proposals: Int = 0
        var accepted: Int = 0

        mutating func recordRound(emitted: Int, proposals: Int, accepted: Int) {
            stepTokenCounts.append(emitted)
            self.proposals += proposals
            self.accepted += accepted
        }

        var summary: String? {
            guard !stepTokenCounts.isEmpty else { return nil }
            return QWTrunkRuntimeSupport.formatSpeculativeSummary(
                label: label,
                accepted: accepted,
                proposals: proposals,
                stepTokenCounts: stepTokenCounts
            )
        }
    }

    private enum PendingVerify {
        case linear(drafts: [Int])
        case tree(tokens: [Int], plan: Ink2TreeVerifyPlan)
    }

    func makePLDDrafts(history: [Int], maxCount: Int, allowRecyclerFallback: Bool = false) -> [Int] {
        guard maxCount > 0, !history.isEmpty else { return [] }
        let minPld = allowRecyclerFallback ? 1 : min(4, maxCount)
        if let periodic = QWContextCopyDrafter.periodicDrafts(
            history: history, maxCount: maxCount, minRepeats: 3
        ), periodic.count >= minPld {
            return Array(periodic.prefix(maxCount))
        }
        let copyMinLen = allowRecyclerFallback ? 3 : 6
        var out: [Int] = []
        if let match = QWContextCopyDrafter.drafts(
            history: history, maxCount: maxCount, minMatchLen: copyMinLen
        ), !match.tokens.isEmpty {
            out = Array(match.tokens.prefix(maxCount))
        }
        let prev = history.count >= 2 ? history[history.count - 2] : nil
        if !out.isEmpty || allowRecyclerFallback {
            session.tokenRecycler.appendDrafts(
                into: &out, maxCount: maxCount,
                previousToken: prev, pendingToken: history[history.count - 1],
                maxUnigramFallbacks: 0
            )
            if out.count >= minPld { return out }
        }
        return []
    }

    func runDSparkSpeculativeDecode(
        dspark: QWInkling2DSpark,
        requestedDrafts: Int,
        allowPLD: Bool = true,
        promptTokens: [Int],
        pos: inout Int,
        effectiveMaxNew: Int,
        stopIDs: Set<Int>,
        generated: inout [Int],
        stepMs: inout [Double],
        stepCounts: inout [Int],
        stopTokenID: inout Int?,
        onToken: ((Int) -> Bool)?,
        onStopToken: ((Int) -> Void)?,
        profiler: QWInkling2Profiler? = nil
    ) throws -> (stats: SpeculativeStats, finished: Bool) {
        let clock = ContinuousClock()
        let headTokPtr = session.headTokensBuf.contents().assumingMemoryBound(to: Int32.self)
        var stats = SpeculativeStats()
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

        let pldCap = min(15, max(7, requestedDrafts))
        let treeCap = min(15, max(7, requestedDrafts))
        let linCap = min(requestedDrafts, max(1, dspark.config.blockSize - 1))

        func pldDrafts(maxCount: Int) -> [Int] {
            allowPLD ? makePLDDrafts(history: history, maxCount: maxCount, allowRecyclerFallback: false) : []
        }

        let explicitTreeEnv = ProcessInfo.processInfo.environment["QWG_DSPARK_TREE"] != nil

        func maxVerifyDrafts(cap: Int) -> Int {
            let remB = effectiveMaxNew - generated.count
            let remC = (capacity - pos) - 2
            return max(0, min(cap, remB - 1, remC))
        }

        func planNextGPU(pldCount: Int) -> QWInkling2DSpark.NextDraft {
            guard pldCount == 0 else { return .none }
            if dspark.treeMode != .linear {
                let remDrafts = maxVerifyDrafts(cap: treeCap)
                guard remDrafts > 0 else { return .none }
                if let tpl = DSparkTreeTemplate.make(
                    mode: dspark.treeMode, maxNodes: 1 + remDrafts, maxDepth: max(1, dspark.config.blockSize - 1)
                ) {
                    return .tree(tpl)
                }
            }
            let lin = maxVerifyDrafts(cap: linCap)
            return lin > 0 ? .linear(lin) : .none
        }

        var pending: PendingVerify?
        let initPldBudget = maxVerifyDrafts(cap: pldCap)
        let initPld = pldDrafts(maxCount: initPldBudget)
        var pendingWasPLD = !initPld.isEmpty
        var lowYieldStreak = 0
        var cooldownSteps = 0
        if !initPld.isEmpty {
            pending = .linear(drafts: initPld)
        } else {
            switch planNextGPU(pldCount: 0) {
            case .none:
                return (stats, false)
            case .linear(let cnt):
                let drafts = try dspark.draftBlock(
                    anchor: generated[generated.count - 1], count: cnt, profiler: profiler
                )
                if !drafts.isEmpty { pending = .linear(drafts: drafts) }
            case .tree(let tpl):
                let toks = try dspark.draftTree(
                    anchor: generated[generated.count - 1], template: tpl, profiler: profiler
                )
                pending = .tree(tokens: toks, plan: tpl.plan)
            }
        }

        var verifyTokens = [Int]()
        verifyTokens.reserveCapacity(16)
        var pendingRoundCB: (any MTLCommandBuffer)? = nil
        func drainPendingRoundCB() throws {
            guard let cb = pendingRoundCB else { return }
            pendingRoundCB = nil
            cb.waitUntilCompleted()
            if let err = cb.error {
                throw QWInkling2Error.badState("DSpark async round CB failed: \(err.localizedDescription)")
            }
        }
        defer {
            if let cb = pendingRoundCB {
                pendingRoundCB = nil
                cb.waitUntilCompleted()
            }
        }

        while let active = pending, generated.count < effectiveMaxNew {
            let roundStart = clock.now
            let anchor = generated[generated.count - 1]
            let activePlan: Ink2TreeVerifyPlan?
            let maxDepthProposals: Int

            verifyTokens.removeAll(keepingCapacity: true)
            switch active {
            case .linear(let drafts):
                verifyTokens.append(anchor)
                verifyTokens.append(contentsOf: drafts)
                activePlan = nil
                maxDepthProposals = drafts.count
            case .tree(let toks, let plan):
                verifyTokens.append(contentsOf: toks)
                activePlan = plan
                maxDepthProposals = Int(plan.depths.max() ?? 0)
            }
            let B = verifyTokens.count

            try forwardPass.forward(
                tokens: verifyTokens,
                posOffset: pos,
                runHeadRows: B,
                greedyHead: true,
                updateState: true,
                captureTapBuf: dspark.tapBuf,
                tapVisits: dspark.config.targetLayerIds,
                recordVerifyConv: true,
                treePlan: activePlan,
                profiler: profiler,
                moeSpecBudget: activeMoESpecBudget(verifyRows: B)
            )
            try drainPendingRoundCB()

            var accepted = 0
            var keptNodes: [Int] = []
            keptNodes.reserveCapacity(min(B, 16))
            var emitted = 0
            var stopped = false
            var cancelled = false

            var currNode = 0
            var unusedRecent: [Int] = []
            var unusedPenalty: [Int] = []
            while true {
                let tok = Int(headTokPtr[currNode])
                keptNodes.append(currNode)
                let outcome = try consumeSpeculativeToken(
                    tok, vocab: ink2.unpaddedVocabSize, stopIDs: stopIDs,
                    recentGeneratedForPenalty: &unusedRecent, penaltyHistory: &unusedPenalty,
                    history: &history, generated: &generated, stopTokenID: &stopTokenID,
                    onStopToken: onStopToken, onToken: onToken
                )
                if outcome == .stopped {
                    stopped = true
                    break
                }
                emitted += 1
                if outcome == .cancelled {
                    cancelled = true
                    break
                }
                if generated.count >= effectiveMaxNew {
                    break
                }
                switch active {
                case .linear(let drafts):
                    let nextR = currNode + 1
                    if currNode < drafts.count && tok == drafts[currNode] {
                        accepted += 1
                        currNode = nextR
                    } else {
                        currNode = -1
                    }
                case .tree(let toks, let plan):
                    if let child = plan.children[currNode].first(where: { toks[$0] == tok }) {
                        accepted += 1
                        currNode = child
                    } else {
                        currNode = -1
                    }
                }
                if currNode < 0 { break }
            }

            let oldPos = pos
            let safeKept = keptNodes.isEmpty ? [0] : keptNodes
            pos += safeKept.count
            let continueSpec = !stopped && !cancelled && generated.count < effectiveMaxNew
            let pldNext = continueSpec ? pldDrafts(maxCount: maxVerifyDrafts(cap: pldCap)) : []
            if !explicitTreeEnv && !pendingWasPLD {
                if accepted == 0 { lowYieldStreak += 1 } else { lowYieldStreak = 0 }
                if lowYieldStreak >= 2 { cooldownSteps = 3; lowYieldStreak = 0 }
            }
            let inCooldown = !explicitTreeEnv && continueSpec && pldNext.isEmpty && cooldownSteps > 0
            let gpuNextReq = (continueSpec && !inCooldown) ? planNextGPU(pldCount: pldNext.count) : .none
            let nextAnchor = generated[generated.count - 1]

            let (gpuResult, roundCB) = try dspark.finishUnifiedSpeculativeRound(
                session: session,
                oldPos: oldPos,
                rowStride: B,
                keptNodes: safeKept,
                ingestTaps: continueSpec,
                nextAnchor: nextAnchor,
                nextDraft: gpuNextReq,
                profiler: profiler,
                allowAsyncNoDraft: true
            )
            pendingRoundCB = roundCB

            let elapsedMs = elapsedSeconds(roundStart.duration(to: clock.now)) * 1000.0
            stepMs.append(elapsedMs)
            stepCounts.append(emitted)
            stats.recordRound(emitted: emitted, proposals: maxDepthProposals, accepted: accepted)

            if stopped || cancelled {
                try drainPendingRoundCB()
                return (stats, true)
            }

            if !continueSpec {
                pending = nil
            } else if !pldNext.isEmpty {
                pending = .linear(drafts: pldNext)
                pendingWasPLD = true
                cooldownSteps = 0
            } else if inCooldown {
                pending = nil
                while cooldownSteps > 0 && generated.count < effectiveMaxNew {
                    let pldTry = pldDrafts(maxCount: maxVerifyDrafts(cap: pldCap))
                    if !pldTry.isEmpty {
                        pending = .linear(drafts: pldTry)
                        pendingWasPLD = true
                        cooldownSteps = 0
                        break
                    }
                    let cStart = clock.now
                    let stepPos = pos
                    let anchorTok = generated[generated.count - 1]
                    try forwardPass.forward(
                        tokens: [anchorTok], posOffset: stepPos, runHeadRows: 1, greedyHead: true,
                        updateState: true, captureTapBuf: dspark.tapBuf,
                        tapVisits: dspark.config.targetLayerIds, profiler: profiler
                    )
                    try drainPendingRoundCB()
                    pos = stepPos + 1
                    let cTok = Int(headTokPtr[0])
                    let cOutcome = try consumeSpeculativeToken(
                        cTok, vocab: ink2.unpaddedVocabSize, stopIDs: stopIDs,
                        recentGeneratedForPenalty: &unusedRecent, penaltyHistory: &unusedPenalty,
                        history: &history, generated: &generated, stopTokenID: &stopTokenID,
                        onStopToken: onStopToken, onToken: nil
                    )
                    if cOutcome == .stopped {
                        stepMs.append(elapsedSeconds(cStart.duration(to: clock.now)) * 1000.0)
                        stepCounts.append(0)
                        return (stats, true)
                    }
                    let canContinue = generated.count < effectiveMaxNew
                    if cooldownSteps > 1 && canContinue {
                        pendingRoundCB = try dspark.ingestContextTaps(
                            count: 1, posOffset: stepPos, rowStride: 1, profiler: profiler, wait: false
                        )
                        stepMs.append(elapsedSeconds(cStart.duration(to: clock.now)) * 1000.0)
                        stepCounts.append(1)
                        if onToken?(cTok) == false {
                            try drainPendingRoundCB()
                            return (stats, true)
                        }
                        cooldownSteps -= 1
                    } else if canContinue {
                        let pldAfter = pldDrafts(maxCount: maxVerifyDrafts(cap: pldCap))
                        if !pldAfter.isEmpty {
                            pendingRoundCB = try dspark.ingestContextTaps(
                                count: 1, posOffset: stepPos, rowStride: 1, profiler: profiler, wait: false
                            )
                            stepMs.append(elapsedSeconds(cStart.duration(to: clock.now)) * 1000.0)
                            stepCounts.append(1)
                            if onToken?(cTok) == false {
                                try drainPendingRoundCB()
                                return (stats, true)
                            }
                            pending = .linear(drafts: pldAfter)
                            pendingWasPLD = true
                            cooldownSteps = 0
                        } else {
                            stepMs.append(elapsedSeconds(cStart.duration(to: clock.now)) * 1000.0)
                            stepCounts.append(1)
                            if onToken?(cTok) == false {
                                return (stats, true)
                            }
                            let nextReq = planNextGPU(pldCount: 0)
                            switch nextReq {
                            case .none:
                                pendingRoundCB = try dspark.ingestContextTaps(
                                    count: 1, posOffset: stepPos, rowStride: 1, profiler: profiler, wait: false
                                )
                                pending = nil
                            case .linear:
                                let drafts = try dspark.ingestAndDraft(
                                    count: 1, posOffset: stepPos, rowStride: 1,
                                    anchor: cTok, nextDraft: nextReq, profiler: profiler
                                )
                                pending = drafts.isEmpty ? nil : .linear(drafts: drafts)
                                pendingWasPLD = false
                            case .tree(let tpl):
                                let toks = try dspark.ingestAndDraft(
                                    count: 1, posOffset: stepPos, rowStride: 1,
                                    anchor: cTok, nextDraft: nextReq, profiler: profiler
                                )
                                pending = toks.isEmpty ? nil : .tree(tokens: toks, plan: tpl.plan)
                                pendingWasPLD = false
                            }
                            cooldownSteps = 0
                        }
                    } else {
                        stepMs.append(elapsedSeconds(cStart.duration(to: clock.now)) * 1000.0)
                        stepCounts.append(1)
                        if onToken?(cTok) == false {
                            return (stats, true)
                        }
                        cooldownSteps -= 1
                    }
                }
            } else {
                pendingWasPLD = false
                switch gpuNextReq {
                case .none:
                    pending = nil
                case .linear:
                    pending = gpuResult.isEmpty ? nil : .linear(drafts: gpuResult)
                case .tree(let tpl):
                    pending = gpuResult.isEmpty ? nil : .tree(tokens: gpuResult, plan: tpl.plan)
                }
            }
        }
        try drainPendingRoundCB()
        return (stats, generated.count >= effectiveMaxNew)
    }
}
