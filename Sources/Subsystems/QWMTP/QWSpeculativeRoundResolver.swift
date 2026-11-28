// QWSpeculativeRoundResolver.swift — Unified stochastic/constrained speculative
// round acceptance and bonus resolution for all runtimes and drafters.

import Foundation
import QWCore
import QWHead

public enum QWSpeculativeRoundResolver {
    public enum DraftProposal {
        case sparse(token: Int, ids: [Int], probs: [Double])
        case greedyToken(Int)
        case fullLogZ(
            token: Int, temperature: Float, topK: Int, topP: Float,
            logits: UnsafeBufferPointer<Float>, maxLogit: Float? = nil, logSumExp: Float? = nil,
            repetitionPenalty: Float = 1.0, repetitionHistory: [Int] = []
        )
        case distribution(token: Int, distribution: QWSamplingDistribution)

        public var token: Int {
            switch self {
            case .sparse(let tok, _, _): return tok
            case .greedyToken(let tok): return tok
            case .fullLogZ(let tok, _, _, _, _, _, _, _, _): return tok
            case .distribution(let tok, _): return tok
            }
        }

        /// Converts a GPU `QWSparseProposalBuffer` into a lossless `DraftProposal` without recomputing `gumbelArgmax`.
        public static func fromSparseBuffer(
            _ buffer: QWSparseProposalBuffer,
            logits: UnsafeBufferPointer<Float>? = nil,
            repetitionPenalty: Float = 1.0,
            repetitionHistory: [Int] = []
        ) throws -> DraftProposal {
            if buffer.isSparse {
                return .sparse(token: buffer.token, ids: buffer.ids, probs: buffer.probs)
            }
            guard let logits, !logits.isEmpty else {
                throw QWMTP.ProtocolError.invalid("non-sparse GPU proposal fallback requires row logits")
            }
            let tok = buffer.gumbelToken >= 0 ? buffer.gumbelToken : buffer.token
            return .fullLogZ(
                token: tok, temperature: buffer.temperature,
                topK: logits.count, topP: 1.0, logits: logits,
                maxLogit: buffer.maxLogit, logSumExp: buffer.logSumExp,
                repetitionPenalty: repetitionPenalty, repetitionHistory: repetitionHistory
            )
        }
    }

    public struct Resolution: Sendable, Equatable {
        public let acceptedDrafts: Int
        public let tokens: [Int]
        public let stopped: Bool
        public let stoppedToken: Int?
        public let usedGPUFastPath: Bool
        public let constraintBoundaryReached: Bool

        public var retainedRows: Int { tokens.count }
        public var firstToken: Int? { tokens.first }
        public var trailingTokens: ArraySlice<Int> { tokens.dropFirst() }
        public var pendingToken: Int? { tokens.last }
    }

    /// Resolves a speculative round via GPU fast-path when valid/unconstrained, or lazy row-by-row
    /// Leviathan–Chen `(p - q)^+` sampling stopping immediately on first reject, stop, cap, or boundary.
    public static func resolve(
        draftProposals: [DraftProposal],
        gpuResolution: QWGPUVerifyResolution? = nil,
        stream: QWMTPRandomStream? = nil,
        isGreedy: Bool = false,
        maxNew: Int,
        stopIDs: Set<Int> = [],
        constraint: (any QWTokenConstraint)? = nil,
        margin: Float = 0,
        minProbability: Float = 0,
        targetDistribution: (
            _ row: Int, _ draftPrefix: ArraySlice<Int>, _ constraintProbe: (any QWTokenConstraint)?
        ) throws -> QWSamplingDistribution
    ) throws -> Resolution {
        guard maxNew >= 0, margin.isFinite, margin >= 0, minProbability.isFinite, minProbability >= 0 else {
            throw QWMTP.ProtocolError.invalid("invalid maxNew, margin, or minProbability")
        }
        if let gpu = gpuResolution, gpu.resolved, margin == 0,
           constraint?.isUnconstrainedNow ?? true {
            return try resolveGPUFastPath(
                gpu: gpu, draftCount: draftProposals.count, stream: stream,
                maxNew: maxNew, stopIDs: stopIDs, constraint: constraint
            )
        }
        guard maxNew > 0 else {
            if stream?.lcRound != nil { try stream?.completeLCRound() }
            return Resolution(acceptedDrafts: 0, tokens: [], stopped: false, stoppedToken: nil,
                              usedGPUFastPath: false, constraintBoundaryReached: false)
        }

        let dCount = draftProposals.count
        let probe: (any QWTokenConstraint)?
        if let constraint {
            if let cloned = constraint.makeSpeculativeProbe() {
                probe = cloned
            } else if constraint.isUnconstrainedNow {
                probe = nil
            } else {
                throw QWMTP.ProtocolError.invalid("constrained speculative resolution requires makeSpeculativeProbe()")
            }
        } else {
            probe = nil
        }

        if dCount == 0 {
            let dist = try targetDistribution(0, [][...], probe)
            let tok = try sampleStandalone(dist: dist, stream: stream, isGreedy: isGreedy)
            let hitStop = stopIDs.contains(tok)
            return Resolution(
                acceptedDrafts: 0, tokens: hitStop ? [] : [tok], stopped: hitStop,
                stoppedToken: hitStop ? tok : nil, usedGPUFastPath: false, constraintBoundaryReached: false
            )
        }

        let lcWindow: QWMTPRandomStream.LCRoundWindow?
        if !isGreedy, let stream {
            if let active = stream.lcRound {
                guard active.drafts == dCount else {
                    throw QWMTP.ProtocolError.invalid("active LC round draft count mismatch")
                }
                lcWindow = active
            } else {
                lcWindow = try stream.beginLCRound(drafts: dCount, maxDrafts: max(15, dCount))
            }
        } else {
            lcWindow = nil
        }

        do {
            let resolution = try resolveLazyLCRows(
                draftProposals: draftProposals, lcWindow: lcWindow, isGreedy: isGreedy,
                maxNew: maxNew, stopIDs: stopIDs, constraint: constraint, probe: probe,
                margin: margin, minProbability: minProbability, targetDistribution: targetDistribution
            )
            if lcWindow != nil { try stream?.completeLCRound() }
            return resolution
        } catch {
            if lcWindow != nil { stream?.cancelActiveRound() }
            throw error
        }
    }

    private static func resolveGPUFastPath(
        gpu: QWGPUVerifyResolution, draftCount: Int, stream: QWMTPRandomStream?,
        maxNew: Int, stopIDs: Set<Int>, constraint: (any QWTokenConstraint)?
    ) throws -> Resolution {
        guard !gpu.tokens.isEmpty, (0...draftCount).contains(gpu.acceptedCount),
              gpu.tokens.count == gpu.acceptedCount + 1 else {
            throw QWMTP.ProtocolError.invalid("malformed QWGPUVerifyResolution")
        }
        if let stream, draftCount > 0 {
            if let active = stream.lcRound {
                guard active.drafts == draftCount else {
                    throw QWMTP.ProtocolError.invalid("active LC round draft count mismatch")
                }
                try stream.completeLCRound()
            } else {
                _ = try stream.beginLCRound(drafts: draftCount, maxDrafts: max(15, draftCount))
                try stream.completeLCRound()
            }
        }
        var rawTokens = gpu.tokens
        var accepted = gpu.acceptedCount
        var boundaryReached = false
        if let boundaryTok = constraint?.speculationBoundaryToken,
           let idx = rawTokens.firstIndex(of: boundaryTok) {
            boundaryReached = true
            if idx + 1 < rawTokens.count {
                rawTokens = Array(rawTokens.prefix(idx + 1))
                accepted = min(accepted, idx)
            }
        }
        var emitted: [Int] = []
        emitted.reserveCapacity(min(maxNew, rawTokens.count))
        var stoppedToken: Int?
        for tok in rawTokens.prefix(maxNew) {
            if stopIDs.contains(tok) { stoppedToken = tok; break }
            emitted.append(tok)
        }
        let finalAccepted = min(accepted, emitted.count + (stoppedToken != nil ? 1 : 0))
        return Resolution(
            acceptedDrafts: finalAccepted, tokens: emitted, stopped: stoppedToken != nil,
            stoppedToken: stoppedToken, usedGPUFastPath: true, constraintBoundaryReached: boundaryReached
        )
    }

    private static func resolveLazyLCRows(
        draftProposals: [DraftProposal], lcWindow: QWMTPRandomStream.LCRoundWindow?,
        isGreedy: Bool, maxNew: Int, stopIDs: Set<Int>, constraint: (any QWTokenConstraint)?,
        probe: (any QWTokenConstraint)?, margin: Float, minProbability: Float,
        targetDistribution: (
            _ row: Int, _ draftPrefix: ArraySlice<Int>, _ constraintProbe: (any QWTokenConstraint)?
        ) throws -> QWSamplingDistribution
    ) throws -> Resolution {
        let dCount = draftProposals.count
        let draftTokens = draftProposals.map(\.token)
        var emitted: [Int] = []
        emitted.reserveCapacity(min(maxNew, dCount + 1))
        var acceptedDrafts = 0, boundaryReached = false, allAccepted = true
        var stoppedToken: Int?

        for row in 0..<dCount {
            let targetDist = try targetDistribution(row, draftTokens[0..<row], probe)
            guard !targetDist.survivorIds.isEmpty,
                  targetDist.survivorIds.count == targetDist.survivorProbs.count else {
                throw QWMTP.ProtocolError.invalid("empty target distribution at row \(row)")
            }
            let (chosen, accepted) = try resolveSingleRow(
                proposal: draftProposals[row], targetDist: targetDist, lcWindow: lcWindow,
                row: row, isGreedy: isGreedy, margin: margin, minProbability: minProbability
            )
            if stopIDs.contains(chosen) {
                if accepted { acceptedDrafts += 1 }
                stoppedToken = chosen
                allAccepted = false
                break
            }
            emitted.append(chosen)
            if !accepted { allAccepted = false; break }
            acceptedDrafts += 1
            if emitted.count >= maxNew { allAccepted = false; break }
            if let probe {
                try probe.accept(chosen)
                let hitBoundary = !probe.allowsSpeculation &&
                    (!probe.isUnconstrainedNow || chosen == probe.speculationBoundaryToken)
                if hitBoundary { boundaryReached = true; allAccepted = false; break }
            } else if let boundaryTok = constraint?.speculationBoundaryToken, chosen == boundaryTok {
                boundaryReached = true
                allAccepted = false
                break
            }
        }

        if allAccepted && emitted.count < maxNew && stoppedToken == nil && !boundaryReached {
            let bonusDist = try targetDistribution(dCount, draftTokens[0..<dCount], probe)
            guard !bonusDist.survivorIds.isEmpty else {
                throw QWMTP.ProtocolError.invalid("empty bonus target distribution")
            }
            let bonusTok: Int
            if !isGreedy, let lcWindow {
                bonusTok = try bonusDist.sampleToken(uniform: lcWindow.bonusUniform, version: .upper24)
            } else {
                bonusTok = modeToken(of: bonusDist)
            }
            if stopIDs.contains(bonusTok) { stoppedToken = bonusTok } else { emitted.append(bonusTok) }
        }

        return Resolution(
            acceptedDrafts: acceptedDrafts, tokens: emitted, stopped: stoppedToken != nil,
            stoppedToken: stoppedToken, usedGPUFastPath: false, constraintBoundaryReached: boundaryReached
        )
    }

    private static func resolveSingleRow(
        proposal: DraftProposal, targetDist: QWSamplingDistribution,
        lcWindow: QWMTPRandomStream.LCRoundWindow?, row: Int, isGreedy: Bool,
        margin: Float, minProbability: Float
    ) throws -> (token: Int, accepted: Bool) {
        let draftToken = proposal.token
        let pDraft = lookupProbability(in: targetDist, token: draftToken)
        if isGreedy || lcWindow == nil {
            let tMode = modeToken(of: targetDist)
            if pDraft > 0 && (draftToken == tMode || clearsRelaxedMargin(
                pDraft: pDraft, targetDist: targetDist, margin: margin, minProbability: minProbability
            )) { return (draftToken, true) }
            return (tMode, false)
        }
        guard let lcWindow else { return (modeToken(of: targetDist), false) }
        if pDraft > 0 && margin > 0 && clearsRelaxedMargin(
            pDraft: pDraft, targetDist: targetDist, margin: margin, minProbability: minProbability
        ) { return (draftToken, true) }
        let evaluator = try ProposalEvaluator(proposal: proposal)
        let qDraft = try evaluator.probability(of: draftToken, targetVocabulary: targetDist.vocabulary)
        let acceptProb: Double = pDraft <= 0 ? 0.0 : (qDraft <= 0 ? 1.0 : min(1.0, pDraft / qDraft))
        if lcWindow.acceptUniform(row: row) < acceptProb {
            return (draftToken, true)
        }
        let residual = try sampleResidual(
            targetDist: targetDist, evaluator: evaluator, uniform: lcWindow.residualUniform(row: row)
        )
        return (residual, false)
    }

    private static func sampleStandalone(
        dist: QWSamplingDistribution, stream: QWMTPRandomStream?, isGreedy: Bool
    ) throws -> Int {
        guard !dist.survivorIds.isEmpty else {
            throw QWMTP.ProtocolError.invalid("empty target distribution")
        }
        if !isGreedy, let stream {
            return try dist.sampleToken(uniform: try stream.nextUniformUpper24(), version: .upper24)
        }
        return modeToken(of: dist)
    }

    private static func modeToken(of dist: QWSamplingDistribution) -> Int {
        var bestTok = dist.survivorIds[0], bestProb = -1.0
        for (tok, prob) in zip(dist.survivorIds, dist.survivorProbs) {
            if prob > bestProb || (prob == bestProb && tok < bestTok) {
                bestProb = prob
                bestTok = tok
            }
        }
        return bestTok
    }

    private static func lookupProbability(in dist: QWSamplingDistribution, token: Int) -> Double {
        let ids = dist.survivorIds
        var lo = 0, hi = ids.count
        while lo < hi {
            let mid = lo + ((hi - lo) >> 1)
            if ids[mid] < token { lo = mid + 1 } else { hi = mid }
        }
        return lo < ids.count && ids[lo] == token ? dist.survivorProbs[lo] : 0.0
    }

    private static func clearsRelaxedMargin(
        pDraft: Double, targetDist: QWSamplingDistribution, margin: Float, minProbability: Float
    ) -> Bool {
        guard margin > 0, pDraft > 0 else { return false }
        if minProbability > 0 && pDraft < Double(minProbability) { return false }
        var pMode = 0.0
        for p in targetDist.survivorProbs where p > pMode { pMode = p }
        guard pMode > 0 else { return false }
        return Foundation.log(pDraft) >= Foundation.log(pMode) - Double(margin)
    }

    private static func sampleResidual(
        targetDist: QWSamplingDistribution, evaluator: ProposalEvaluator, uniform: Double
    ) throws -> Int {
        let count = targetDist.survivorIds.count
        var weights = [Double](repeating: 0, count: count)
        var mass = 0.0, fallbackIdx = count - 1
        for i in 0..<count {
            let qProb = try evaluator.probability(of: targetDist.survivorIds[i], targetVocabulary: targetDist.vocabulary)
            let w = max(0.0, targetDist.survivorProbs[i] - qProb)
            weights[i] = w
            if w > 0 { mass += w; fallbackIdx = i }
        }
        guard mass.isFinite && mass > 1e-15 else {
            return try targetDist.sampleToken(uniform: uniform, version: .upper24)
        }
        var cumulative = 0.0
        for i in 0..<count where weights[i] > 0 {
            cumulative += weights[i] / mass
            if cumulative > uniform { return targetDist.survivorIds[i] }
        }
        return targetDist.survivorIds[fallbackIdx]
    }
}

private enum ProposalEvaluator {
    case greedy(Int)
    case sortedSparse(ids: [Int], probs: [Double])
    case logZDirect(
        temperature: Double, logits: UnsafeBufferPointer<Float>, maxLogit: Float,
        logSumExp: Double, repetitionPenalty: Float, penaltySet: Set<Int>
    )

    init(proposal: QWSpeculativeRoundResolver.DraftProposal) throws {
        switch proposal {
        case .greedyToken(let tok):
            self = .greedy(tok)
        case .distribution(_, let dist):
            self = .sortedSparse(ids: dist.survivorIds, probs: dist.survivorProbs)
        case .sparse(_, let ids, let probs):
            guard ids.count == probs.count, !ids.isEmpty else {
                throw QWMTP.ProtocolError.invalid("sparse draft proposal ids/probs invalid")
            }
            let normalized = Self.normalizedSorted(ids: ids, probs: probs)
            self = .sortedSparse(ids: normalized.ids, probs: normalized.probs)
        case .fullLogZ(_, let temp, let topK, let topP, let logits, let maxLogit, let logSumExp, let rep, let hist):
            guard temp.isFinite, temp > 0, !logits.isEmpty else {
                throw QWMTP.ProtocolError.invalid("invalid fullLogZ proposal parameters")
            }
            if let maxLogit, let logSumExp, maxLogit.isFinite, logSumExp.isFinite,
               topK >= logits.count, topP >= 1.0 {
                self = .logZDirect(
                    temperature: Double(temp), logits: logits, maxLogit: maxLogit,
                    logSumExp: Double(logSumExp), repetitionPenalty: rep,
                    penaltySet: (rep != 1 && !hist.isEmpty) ? Set(hist) : []
                )
            } else {
                let cfg = QWSamplerConfig(
                    temperature: temp, topK: max(1, min(topK, logits.count)),
                    topP: topP, uniformVersion: .upper24, repetitionPenalty: rep
                )
                let dist = try QWSampler(config: cfg).distribution(logits: logits, repetitionHistory: hist)
                self = .sortedSparse(ids: dist.survivorIds, probs: dist.survivorProbs)
            }
        }
    }

    private static func normalizedSorted(ids: [Int], probs: [Double]) -> (ids: [Int], probs: [Double]) {
        var ascending = true, sum = 0.0
        for i in ids.indices {
            if i > 0 && ids[i] <= ids[i - 1] { ascending = false }
            sum += max(0.0, probs[i])
        }
        if ascending && sum > 0 {
            return abs(sum - 1.0) <= 1e-12 ? (ids, probs) : (ids, probs.map { max(0.0, $0) / sum })
        }
        var merged: [Int: Double] = [:]
        for (id, p) in zip(ids, probs) where p.isFinite && p > 0 { merged[id, default: 0] += p }
        let sorted = merged.sorted { $0.key < $1.key }
        let total = max(1e-300, sorted.reduce(0.0) { $0 + $1.value })
        return (sorted.map(\.key), sorted.map { $0.value / total })
    }

    func probability(of token: Int, targetVocabulary: Int) throws -> Double {
        switch self {
        case .greedy(let tok):
            return token == tok ? 1.0 : 0.0
        case .sortedSparse(let ids, let probs):
            var lo = 0, hi = ids.count
            while lo < hi {
                let mid = lo + ((hi - lo) >> 1)
                if ids[mid] < token { lo = mid + 1 } else { hi = mid }
            }
            return lo < ids.count && ids[lo] == token ? probs[lo] : 0.0
        case .logZDirect(let temp, let logits, let maxLogit, let logSumExp, let rep, let penaltySet):
            guard (0..<logits.count).contains(token), (0..<targetVocabulary).contains(token) else { return 0.0 }
            var val = logits[token]
            if rep != 1 && penaltySet.contains(token) { val = val < 0 ? val * rep : val / rep }
            let logP = Double(val - maxLogit) / temp - logSumExp
            return logP < -700.0 ? 0.0 : Foundation.exp(logP)
        }
    }
}
