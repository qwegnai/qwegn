// QWDFlashAdaptivePolicy.swift — Online throughput-optimal window selector
// for block-diffusion drafters (DFlash 2 / DSpine) on M = 1 + D verification.

import Foundation

/// Picks the DFlash/DSpine draft window `D` in `{3, 5, 7, maxDrafts}` that
/// maximizes expected committed tokens per millisecond (`E[tokens(D)] / costMs(D)`).
public struct QWDFlashAdaptivePolicy: Sendable {
    public static let fixedWindow: Bool =
        ProcessInfo.processInfo.environment["QWG_DFLASH_FIXED_WINDOW"] == "1"

    static let maxSupportedDrafts = 15
    private static let exploreEvery = 16

    private let isSparseMoE: Bool
    private var acc: [Double]
    private var costMsBucket: (b3: Double, b5: Double, b7: Double, b15: Double)
    private var b2CostMs: Double = 25.5
    private var observedBuckets: UInt8 = 0
    private var rounds: Int = 0
    private var lastDraftCount: Int = 0
    private var lastAcceptedAll: Bool = false
    private var consecutiveFullAccepts: Int = 0

    public init(isSparseMoE: Bool = false) {
        self.isSparseMoE = isSparseMoE
        self.acc = [
            0.78, 0.70, 0.65, 0.62, 0.60,
            0.58, 0.56, 0.55, 0.54, 0.53,
            0.52, 0.51, 0.50, 0.50, 0.50,
        ]
        if isSparseMoE {
            self.costMsBucket = (30.5, 39.5, 46.5, 98.0)
        } else {
            self.costMsBucket = (47.0, 53.0, 58.4, 107.0)
        }
    }

    public mutating func selectDraftCount(maxDrafts: Int) -> Int {
        guard maxDrafts > 1 && !Self.fixedWindow else { return maxDrafts }
        let cap = min(maxDrafts, Self.maxSupportedDrafts)
        rounds += 1
        if isSparseMoE {
            return selectSparseMoEDraftCount(cap: cap)
        }
        if rounds == 1 {
            let initial = min(7, cap)
            lastDraftCount = initial
            return initial
        }
        if lastAcceptedAll {
            if lastDraftCount == cap {
                return cap
            }
            if lastDraftCount < 7 || consecutiveFullAccepts >= 2 || (rounds == 2 && lastDraftCount == 7) {
                let promoted = (lastDraftCount < 7) ? min(7, cap) : cap
                lastDraftCount = promoted
                return promoted
            }
        }
        if rounds % Self.exploreEvery == 0 {
            let mid = min(7, cap)
            if lastDraftCount < mid {
                lastDraftCount = mid
                return mid
            }
            if lastDraftCount < cap && reach(drafts: mid) >= 0.35 {
                lastDraftCount = cap
                return cap
            }
        }
        let best = bestCandidate(from: [min(3, cap), min(5, cap), min(7, cap), cap])
        lastDraftCount = best
        return best
    }

    private mutating func selectSparseMoEDraftCount(cap: Int) -> Int {
        if rounds == 1 {
            let initial = min(3, cap)
            lastDraftCount = initial
            return initial
        }
        let fastSparse = (observedBuckets & 2) != 0 && costMsBucket.b3 < 28.0
        let effCap = (fastSparse && reach(drafts: 5) < 0.65) ? min(5, cap) : cap
        if lastAcceptedAll {
            if lastDraftCount >= effCap { return effCap }
            if fastSparse && lastDraftCount < 3 {
                let promoted = min(3, effCap)
                lastDraftCount = promoted
                return promoted
            }
            let minConsec = fastSparse ? 1 : 2
            let minReach = fastSparse ? 0.22 : 0.52
            if consecutiveFullAccepts >= minConsec && (reach(drafts: lastDraftCount) >= minReach || (fastSparse && consecutiveFullAccepts >= 2)) {
                let promoted = nextSparseBucket(after: lastDraftCount, cap: effCap)
                lastDraftCount = promoted
                return promoted
            }
            return min(lastDraftCount, effCap)
        }
        if fastSparse, lastDraftCount <= 3, reach(drafts: 2) >= 0.48 {
            let promoted = min(5, effCap)
            lastDraftCount = promoted
            return promoted
        }
        if rounds % Self.exploreEvery == 0 {
            let b3 = min(3, effCap)
            if lastDraftCount < b3 {
                lastDraftCount = b3
                return b3
            }
            if lastDraftCount < effCap && reach(drafts: lastDraftCount) >= 0.48 {
                let explored = nextSparseBucket(after: lastDraftCount, cap: effCap)
                lastDraftCount = explored
                return explored
            }
        }
        let best = bestCandidate(from: [min(2, effCap), min(3, effCap), min(5, effCap), min(7, effCap), effCap])
        lastDraftCount = best
        return best
    }

    private func nextSparseBucket(after current: Int, cap: Int) -> Int {
        if current < 3 { return min(3, cap) }
        if current < 5 { return min(5, cap) }
        if current < 7 { return min(7, cap) }
        return cap
    }

    private func bestCandidate(from candidates: [Int]) -> Int {
        var best = candidates[0]
        var bestRate = expectedTokens(drafts: best) / bucketCost(drafts: best)
        for cand in candidates.dropFirst() where cand != best {
            let rate = expectedTokens(drafts: cand) / bucketCost(drafts: cand)
            if rate > bestRate {
                bestRate = rate
                best = cand
            }
        }
        return best
    }

    public mutating func observeRound(draftCount: Int, accepted: Int, roundMs: Double?) {
        guard draftCount > 0 else { return }
        lastAcceptedAll = (accepted >= draftCount)
        consecutiveFullAccepts = lastAcceptedAll ? (consecutiveFullAccepts + 1) : 0
        let alpha = (rounds <= 5 && accepted <= 1) ? 0.55 : 0.18
        let limit = min(draftCount, acc.count)
        for i in 0..<limit where accepted >= i {
            let target = accepted > i ? 1.0 : 0.0
            acc[i] += alpha * (target - acc[i])
        }
        guard let rawMs = roundMs, rawMs > 0 else { return }
        let msAlpha = 0.20
        let wasFirstObservation = (observedBuckets == 0)
        let ms = (wasFirstObservation && rawMs > 90.0) ? min(rawMs, isSparseMoE ? 25.0 : 47.0) : rawMs
        if isSparseMoE && draftCount <= 2 {
            observedBuckets |= 1
            b2CostMs = wasFirstObservation ? ms : (b2CostMs + msAlpha * (ms - b2CostMs))
        } else if draftCount <= 3 {
            observedBuckets |= 2
            costMsBucket.b3 = wasFirstObservation ? ms : (costMsBucket.b3 + msAlpha * (ms - costMsBucket.b3))
        } else if draftCount <= 5 {
            observedBuckets |= 4
            costMsBucket.b5 = wasFirstObservation ? ms : (costMsBucket.b5 + msAlpha * (ms - costMsBucket.b5))
        } else if draftCount <= 7 {
            observedBuckets |= 8
            costMsBucket.b7 = wasFirstObservation ? ms : (costMsBucket.b7 + msAlpha * (ms - costMsBucket.b7))
        } else {
            observedBuckets |= 16
            costMsBucket.b15 = wasFirstObservation ? ms : (costMsBucket.b15 + msAlpha * (ms - costMsBucket.b15))
        }
        if isSparseMoE, (observedBuckets & 2) != 0 {
            let b3 = costMsBucket.b3
            let fastSparse = b3 < 28.0
            if (observedBuckets & 1) == 0 { b2CostMs = b3 * (fastSparse ? 0.95 : (25.5 / 30.5)) }
            if (observedBuckets & 4) == 0 { costMsBucket.b5 = b3 * (fastSparse ? 1.12 : (39.5 / 30.5)) }
            if (observedBuckets & 8) == 0 { costMsBucket.b7 = b3 * (fastSparse ? 1.38 : (46.5 / 30.5)) }
            if (observedBuckets & 16) == 0 { costMsBucket.b15 = b3 * (fastSparse ? 2.35 : (98.0 / 30.5)) }
        }
    }

    public func expectedTokens(drafts d: Int) -> Double {
        let limit = min(max(0, d), acc.count)
        var total = 1.0
        var prob = 1.0
        for i in 0..<limit {
            prob *= acc[i]
            total += prob
        }
        return total
    }

    private func reach(drafts d: Int) -> Double {
        let limit = min(max(0, d), acc.count)
        var prob = 1.0
        for i in 0..<limit { prob *= acc[i] }
        return prob
    }

    private func bucketCost(drafts d: Int) -> Double {
        if isSparseMoE && d <= 2 { return b2CostMs }
        if d <= 3 { return costMsBucket.b3 }
        if d <= 5 { return costMsBucket.b5 }
        if d <= 7 { return costMsBucket.b7 }
        return costMsBucket.b15
    }
}
