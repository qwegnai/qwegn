// QWInkling2Runtime+StochasticSpec.swift — Variational, stochastic & weight-free PLD speculative decode.
import Accelerate
import Foundation
import QWCore
import QWHead

extension QWInkling2Runtime {
    static func canSpeculate(constraint: (any QWTokenConstraint)?) -> Bool {
        guard let constraint else { return true }
        if constraint.isUnconstrainedNow { return true }
        return QWConstrainedSpeculationGate.isEnabled && constraint.allowsSpeculation
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
        let usePenalty = sampler.config.repetitionPenalty != 1.0
        var penScratch = (sampler.config.isGreedy && usePenalty) ? [Float](repeating: 0, count: V) : []
        var stats = SpeculativeStats()
        let minProbFloor: Float = minLogProb == -.infinity ? 0 : Float(exp(Double(minLogProb)))

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

        var drawCounter = sampler.drawsConsumed
        var sysRng = SystemRandomNumberGenerator()
        func nextUniform() -> Double {
            if sampler.config.hasExplicitSeed {
                let u = QWPhilox.u(drawCounter, seed: sampler.config.seed, version: sampler.config.uniformVersion)
                drawCounter &+= 1
                return min(0.9999999999999999, max(0.0, Double(u)))
            }
            return Double.random(in: 0.0..<1.0, using: &sysRng)
        }

        let explicitTreeEnv = ProcessInfo.processInfo.environment["QWG_DSPARK_TREE"] != nil
        let pldCap = min(15, dspark != nil ? max(7, requestedDrafts) : requestedDrafts)
        let linCap = dspark.map { min(requestedDrafts, max(1, $0.config.blockSize - 1)) } ?? 0
        func maxVerifyDrafts(cap: Int) -> Int {
            let remB = effectiveMaxNew - generated.count
            let remC = (capacity - pos) - 2
            return max(0, min(cap, remB - 1, remC))
        }

        func resolveGreedy(rowPtr: UnsafePointer<Float>, draft: Int?) throws -> (tok: Int, accepted: Bool) {
            let activePtr: UnsafePointer<Float>
            if usePenalty && !penaltyHistory.isEmpty {
                penScratch.withUnsafeMutableBufferPointer { dst in
                    _ = memcpy(dst.baseAddress!, rowPtr, V * MemoryLayout<Float>.stride)
                    let rep = sampler.config.repetitionPenalty
                    for token in penaltyHistory where token >= 0 && token < V {
                        let v = dst[token]
                        dst[token] = v < 0 ? v * rep : v / rep
                    }
                }
                activePtr = penScratch.withUnsafeBufferPointer { $0.baseAddress! }
            } else {
                activePtr = rowPtr
            }
            var maxVal: Float = 0
            var maxIdx: vDSP_Length = 0
            vDSP_maxvi(activePtr, 1, &maxVal, &maxIdx, vDSP_Length(V))
            guard maxVal.isFinite else {
                throw QWEngineDriverError.state("greedy verification received non-finite logits")
            }
            let argmaxTok = Int(maxIdx)
            guard let draft, (0..<V).contains(draft) else { return (argmaxTok, false) }
            if draft == argmaxTok { return (argmaxTok, true) }
            if effectiveMargin > 0 && (maxVal - activePtr[draft]) <= effectiveMargin {
                if minLogProb == -.infinity { return (draft, true) }
                let cutoff = maxVal - 25.0
                var sumExp = 0.0
                for i in 0..<V where activePtr[i] >= cutoff {
                    sumExp += exp(Double(activePtr[i] - maxVal))
                }
                let draftLogProb = Float(Double(activePtr[draft] - maxVal) - log(max(sumExp, 1e-30)))
                if draftLogProb >= minLogProb { return (draft, true) }
            }
            return (argmaxTok, false)
        }

        func resolveStochastic(rowPtr: UnsafePointer<Float>, draft: Int?) throws -> (tok: Int, accepted: Bool) {
            let rowBuf = UnsafeBufferPointer(start: rowPtr, count: V)
            let dist = try sampler.distribution(
                logits: rowBuf, repetitionHistory: (usePenalty && !penaltyHistory.isEmpty) ? penaltyHistory : []
            )
            guard let draft, (0..<V).contains(draft) else {
                return (try dist.sampleToken(uniform: nextUniform(), version: sampler.config.uniformVersion), false)
            }
            let pTarget = Self.lookupProb(dist: dist, token: draft)
            let acceptProb = effectiveMargin > 0 ? min(1.0, pTarget * exp(Double(effectiveMargin))) : pTarget
            if nextUniform() < acceptProb {
                return (draft, true)
            }
            let rep = Self.sampleResidual(dist: dist, exclude: draft, excludeProb: pTarget, uniform: nextUniform())
            return (rep, false)
        }

        var pendingWasPLD = false

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

        func sampleRow(_ r: Int, draft: Int?, rowMargin: Float) throws -> (tok: Int, accepted: Bool) {
            let rowPtr = logitsPtr.advanced(by: r * V)
            let rowBuf = UnsafeBufferPointer(start: rowPtr, count: V)
            let repHist = (usePenalty && !penaltyHistory.isEmpty) ? penaltyHistory : []
            if let constraint, try constraint.writeAllowedTokens(into: &allowedScratch) {
                if sampler.config.isGreedy {
                    let strictTok = try allowedScratch.withUnsafeBufferPointer { buf in
                        try sampler.constrainedToken(logits: rowBuf, allowed: buf, repetitionHistory: repHist)
                    }
                    guard let draft, (0..<V).contains(draft) else { return (strictTok, false) }
                    if draft == strictTok { return (strictTok, true) }
                    if rowMargin > 0 {
                        let passes = allowedScratch.withUnsafeBufferPointer { buf in
                            sampler.clearsConstrainedVariationalGate(
                                logits: rowBuf, allowed: buf, draftToken: draft, targetToken: strictTok,
                                repetitionHistory: repHist, margin: rowMargin, minProbability: minProbFloor
                            )
                        }
                        if passes { return (draft, true) }
                    }
                    return (strictTok, false)
                }
                let dist = try allowedScratch.withUnsafeBufferPointer { buf in
                    try sampler.constrainedDistribution(logits: rowBuf, allowed: buf, repetitionHistory: repHist)
                }
                guard let draft, (0..<V).contains(draft) else {
                    return (try dist.sampleToken(uniform: nextUniform(), version: sampler.config.uniformVersion), false)
                }
                let pTarget = Self.lookupProb(dist: dist, token: draft)
                let acceptProb = rowMargin > 0 ? min(1.0, pTarget * exp(Double(rowMargin))) : pTarget
                if nextUniform() < acceptProb {
                    return (draft, true)
                }
                let rep = Self.sampleResidual(dist: dist, exclude: draft, excludeProb: pTarget, uniform: nextUniform())
                return (rep, false)
            }
            if sampler.config.isGreedy {
                return try resolveGreedy(rowPtr: rowPtr, draft: draft)
            }
            return try resolveStochastic(rowPtr: rowPtr, draft: draft)
        }

        var pendingDrafts: [Int]?
        var lowYieldStreak = 0
        var cooldownSteps = 0

        let initMax = maxVerifyDrafts(cap: pldCap)
        let initPld = proposeConstrainedOrPLD(maxCount: initMax, allowRecycler: dspark == nil)
        if !initPld.isEmpty {
            pendingDrafts = initPld
            pendingWasPLD = true
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
                let singleMax = maxVerifyDrafts(cap: pldCap)
                let pldNow = proposeConstrainedOrPLD(maxCount: singleMax, allowRecycler: dspark == nil)
                if !pldNow.isEmpty {
                    pendingDrafts = pldNow
                    pendingWasPLD = true
                    cooldownSteps = 0
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
                let (tok, _) = try sampleRow(0, draft: nil, rowMargin: 0)
                stepMs.append(elapsedSeconds(roundStart.duration(to: clock.now)) * 1000.0)
                let outcome = try consumeSpeculativeToken(
                    tok, vocab: V, stopIDs: stopIDs, constraint: constraint, usePenalty: usePenalty,
                    recentGeneratedForPenalty: &recentGeneratedForPenalty, penaltyHistory: &penaltyHistory,
                    history: &history, generated: &generated, stopTokenID: &stopTokenID,
                    onStopToken: onStopToken, onToken: onToken
                )
                if outcome == .stopped {
                    stepCounts.append(0)
                    return (stats, true)
                }
                stepCounts.append(1)
                if outcome == .cancelled { return (stats, true) }
                if cooldownSteps > 0 { cooldownSteps -= 1 }
                let nextInit = proposeConstrainedOrPLD(
                    maxCount: maxVerifyDrafts(cap: pldCap), allowRecycler: dspark == nil)
                if !nextInit.isEmpty {
                    pendingDrafts = nextInit
                    pendingWasPLD = true
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
                tokens: verifyTokens,
                posOffset: pos,
                runHeadRows: B,
                greedyHead: false,
                updateState: true,
                captureTapBuf: dspark?.tapBuf,
                tapVisits: dspark?.config.targetLayerIds ?? [],
                recordVerifyConv: true,
                profiler: profiler,
                moeSpecBudget: activeMoESpecBudget(verifyRows: B)
            )

            var accepted = 0
            var emitted = 0
            var stopped = false
            var cancelled = false
            let roundMargin: Float = usedPLDThisRound ? 0 : effectiveMargin

            for r in 0..<B {
                let targetDraft = r < drafts.count ? drafts[r] : nil
                let (tok, didAccept) = try sampleRow(r, draft: targetDraft, rowMargin: roundMargin)
                let outcome = try consumeSpeculativeToken(
                    tok, vocab: V, stopIDs: stopIDs, constraint: constraint, usePenalty: usePenalty,
                    recentGeneratedForPenalty: &recentGeneratedForPenalty, penaltyHistory: &penaltyHistory,
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
                    if targetDraft != nil && didAccept { accepted += 1 }
                    break
                }
                if targetDraft != nil && didAccept {
                    accepted += 1
                } else {
                    break
                }
            }

            let oldPos = pos
            let kept = 1 + accepted
            pos += kept
            if dspark != nil && !explicitTreeEnv && !usedPLDThisRound {
                if accepted == 0 { lowYieldStreak += 1 } else { lowYieldStreak = 0 }
                if lowYieldStreak >= 2 { cooldownSteps = 3; lowYieldStreak = 0 }
            }

            let continueSpec = !stopped && !cancelled && generated.count < effectiveMaxNew
            if let ds = dspark {
                let pldNext = continueSpec
                    ? proposeConstrainedOrPLD(maxCount: maxVerifyDrafts(cap: pldCap), allowRecycler: false)
                    : []
                let inCooldown = !explicitTreeEnv && continueSpec && pldNext.isEmpty && cooldownSteps > 0
                let nextDCount = (continueSpec && pldNext.isEmpty && !inCooldown)
                    ? maxVerifyDrafts(cap: linCap) : 0
                let gpuNext = try ds.finishSpeculativeRound(
                    session: session, kept: kept, oldPos: oldPos, rowStride: B,
                    ingestTaps: continueSpec, nextAnchor: generated[generated.count - 1],
                    nextDraftCount: nextDCount, profiler: profiler
                )
                if !pldNext.isEmpty {
                    pendingDrafts = pldNext
                    pendingWasPLD = true
                    cooldownSteps = 0
                } else if nextDCount > 0 && !gpuNext.isEmpty {
                    pendingDrafts = gpuNext
                    pendingWasPLD = false
                }
            } else {
                try session.commitVerifyConv(hal: hal, kept: kept, profiler: profiler)
            }

            let elapsedMs = elapsedSeconds(roundStart.duration(to: clock.now)) * 1000.0
            stepMs.append(elapsedMs)
            stepCounts.append(emitted)
            stats.recordRound(emitted: emitted, proposals: drafts.count, accepted: accepted)

            if stopped || cancelled {
                return (stats, true)
            }
        }
        return (stats, generated.count >= effectiveMaxNew)
    }

    private static func lookupProb(dist: QWSamplingDistribution, token: Int) -> Double {
        let ids = dist.survivorIds
        var lo = 0
        var hi = ids.count - 1
        while lo <= hi {
            let mid = (lo + hi) >> 1
            let midId = ids[mid]
            if midId == token { return dist.survivorProbs[mid] }
            if midId < token { lo = mid + 1 } else { hi = mid - 1 }
        }
        return 0.0
    }

    private static func sampleResidual(
        dist: QWSamplingDistribution,
        exclude: Int,
        excludeProb: Double,
        uniform: Double
    ) -> Int {
        let residualMass = 1.0 - excludeProb
        guard residualMass > 1e-12 else {
            return (try? dist.sampleToken(uniform: uniform, version: .upper24)) ?? dist.survivorIds[0]
        }
        let target = uniform * residualMass
        var cum = 0.0
        var lastCandidate = dist.survivorIds[0]
        for i in dist.survivorIds.indices {
            let id = dist.survivorIds[i]
            if id == exclude { continue }
            lastCandidate = id
            cum += dist.survivorProbs[i]
            if cum > target { return id }
        }
        return lastCandidate
    }
}
