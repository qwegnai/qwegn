// QWAdaptiveDraftPolicy.swift — Unified pre-round online E[tokens(D)]/costMs(D)
// depth controller for autoregressive chains and block-diffusion drafters.

import Foundation
import QWCore

/// Unified pre-round speculative draft depth policy for block-diffusion
/// (`[2, 3, 5, 7, 15]`) and autoregressive chain (`1...maxDrafts`) drafters.
public struct QWAdaptiveDraftPolicy: Sendable {
    public enum Mode: Sendable, Equatable {
        case blockDiffusion(isSparseMoE: Bool)
        case autoregressiveChain
    }

    public static let fixedWindow: Bool =
        ProcessInfo.processInfo.environment["QWG_DFLASH_FIXED_WINDOW"] == "1"

    static let maxSupportedDrafts = 15
    private static let exploreEvery = 16
    private static let chainAlpha = 0.15

    public let mode: Mode
    private var acc: [Double]
    private var costMsBucket: (b3: Double, b5: Double, b7: Double, b15: Double)
    private var b2CostMs: Double = 25.5
    private var observedBuckets: UInt8 = 0
    private var rowMs: [Double]
    private var passMs: Double
    private var rounds: Int = 0
    private var lastDraftCount: Int = 0
    private var lastAcceptedAll: Bool = false
    private var consecutiveFullAccepts: Int = 0

    public init(mode: Mode, initialPassMs: Double = 1.3) {
        self.mode = mode
        self.passMs = initialPassMs
        self.rowMs = [0, 15, 18, 22, 25, 29, 32, 35, 38]
        switch mode {
        case .blockDiffusion(let isSparseMoE):
            self.acc = [0.78, 0.70, 0.65, 0.62, 0.60, 0.58, 0.56, 0.55, 0.54, 0.53, 0.52, 0.51, 0.50, 0.50, 0.50]
            self.costMsBucket = isSparseMoE ? (30.5, 39.5, 46.5, 98.0) : (47.0, 53.0, 58.4, 107.0)
        case .autoregressiveChain:
            self.acc = [0.8] + Array(repeating: 0.7, count: Self.maxSupportedDrafts - 1)
            self.costMsBucket = (22.0, 29.0, 35.0, 60.0)
        }
    }

    public init(isSparseMoE: Bool = false) { self.init(mode: .blockDiffusion(isSparseMoE: isSparseMoE)) }

    public mutating func selectDraftCount(maxDrafts: Int) -> Int {
        guard maxDrafts > 1 else { return max(0, maxDrafts) }
        switch mode {
        case .blockDiffusion(let isSparseMoE):
            guard !Self.fixedWindow else { return maxDrafts }
            return selectBlockDraftCount(maxDrafts: maxDrafts, isSparseMoE: isSparseMoE)
        case .autoregressiveChain:
            return 1 + selectChainPasses(maxPasses: min(Self.maxSupportedDrafts, maxDrafts) - 1)
        }
    }

    public mutating func selectChainPasses(maxPasses: Int) -> Int {
        rounds += 1
        guard maxPasses > 0 else { return 0 }
        if rounds % Self.exploreEvery == 0 { return maxPasses }
        var best = 0, bestRate = 0.0, tokens = 1.0 + acc[0], reach = acc[0]
        for p in 0...maxPasses {
            if p > 0 { reach *= acc[min(p, acc.count - 1)]; tokens += reach }
            let rate = tokens / max(1e-6, rowMs[min(2 + p, rowMs.count - 1)] + Double(p) * passMs)
            if rate > bestRate { bestRate = rate; best = p }
        }
        return best
    }

    public mutating func observeAcceptance(draftCount: Int, accepted: Int) {
        guard draftCount > 0 else { return }
        lastAcceptedAll = (accepted >= draftCount)
        consecutiveFullAccepts = lastAcceptedAll ? (consecutiveFullAccepts + 1) : 0
        let alpha: Double = if case .blockDiffusion = mode {
            (rounds <= 5 && accepted <= 1) ? 0.55 : 0.18
        } else { Self.chainAlpha }
        for i in 0..<min(draftCount, acc.count) where accepted >= i {
            acc[i] += alpha * ((accepted > i ? 1.0 : 0.0) - acc[i])
        }
    }

    /// Updates acceptance unconditionally; wall-clock timing updates require `!isGreedy && !hasExplicitSeed` (Law 11).
    public mutating func observeRound(
        draftCount: Int, accepted: Int, roundMs: Double?,
        isGreedy: Bool = false, hasExplicitSeed: Bool = false
    ) {
        observeAcceptance(draftCount: draftCount, accepted: accepted)
        guard !isGreedy && !hasExplicitSeed, let rawMs = roundMs, rawMs > 0 else { return }
        switch mode {
        case .blockDiffusion(let isSparseMoE):
            observeBlockCost(draftCount: draftCount, rawMs: rawMs, isSparseMoE: isSparseMoE)
        case .autoregressiveChain:
            observeVerifyRound(rows: 1 + draftCount, ms: rawMs, isGreedy: isGreedy, hasExplicitSeed: hasExplicitSeed)
        }
    }

    /// Updates per-verify-row EMA wall time when `!isGreedy && !hasExplicitSeed` (Law 11).
    public mutating func observeVerifyRound(
        rows: Int, ms: Double, isGreedy: Bool = false, hasExplicitSeed: Bool = false
    ) {
        guard !isGreedy && !hasExplicitSeed, rows >= 1, rows < rowMs.count, ms > 0 else { return }
        rowMs[rows] += Self.chainAlpha * (ms - rowMs[rows])
    }

    /// Updates per-chain-pass EMA wall time when `!isGreedy && !hasExplicitSeed` (Law 11).
    public mutating func observeChain(
        passes: Int, ms: Double, isGreedy: Bool = false, hasExplicitSeed: Bool = false
    ) {
        guard !isGreedy && !hasExplicitSeed, passes > 0, ms > 0 else { return }
        passMs += Self.chainAlpha * (ms / Double(passes) - passMs)
    }

    public func expectedTokens(drafts d: Int) -> Double {
        let limit = min(max(0, d), acc.count)
        var total = 1.0, prob = 1.0
        for i in 0..<limit { prob *= acc[i]; total += prob }
        return total
    }

    /// Caps NextN neural chain depth to 1 when host-embedding gather is active or on sparse MoE,
    /// in both greedy and stochastic (`T > 0`) modes.
    public static func neuralDepthCap(
        isStochastic: Bool, stageProfiling: Bool, hasDeviceEmbedding: Bool, numExperts: Int
    ) -> Int? {
        let _ = isStochastic
        return (!stageProfiling && (!hasDeviceEmbedding || numExperts > 0)) ? 1 : nil
    }

    /// Chooses the optimal draft depth `k` in `1...maxDrafts` from model weight bytes and acceptance rate.
    public static func optimalDraftCount(
        geometry: QWModelGeometry, trunkBits: Int = 4, expertBits: Int = 4,
        acceptRate: Double, maxDrafts: Int, expertOverlap: Double = 0.25
    ) -> Int {
        let clampedMax = max(1, min(maxSupportedDrafts, maxDrafts))
        guard clampedMax > 1 else { return 1 }
        let alpha = min(1.0, max(0.0, acceptRate.isFinite ? acceptRate : 0.0))
        let h = Double(max(1, geometry.hidden))
        let tB = Double(max(1, trunkBits)) / 8.0, eB = Double(max(1, expertBits)) / 8.0
        var trunkParams = 0.0, moeLayers = 0
        for layer in geometry.executionPlan {
            if geometry.isFullAttention(layer) {
                if geometry.usesAnyMLA {
                    let rope = geometry.usesNoPEMLA ? 0 : geometry.qkRopeHeadDim
                    let idx = geometry.mlaHasIndexer ? max(32, geometry.indexerHeadDim) : 0
                    let qIn = geometry.mlaHasDirectQ ? geometry.hidden : geometry.qLoraRank
                    let qkv = Double((geometry.mlaHasDirectQ ? 0 : geometry.qLoraRank) + geometry.kvLoraRank + rope + idx)
                    let gate = geometry.attnOutputGate ? Double(geometry.qHeads * geometry.vHeadDim) : 0.0
                    let qB = Double(geometry.qHeads * (geometry.qkNopeHeadDim + rope)) + gate
                    let kvB = Double(geometry.qHeads * (geometry.qkNopeHeadDim + geometry.vHeadDim))
                    trunkParams += qkv * h + qB * Double(qIn) + kvB * Double(geometry.kvLoraRank) + h * Double(geometry.qHeads * geometry.vHeadDim)
                } else {
                    trunkParams += Double(geometry.attnTotalN) * h + h * Double(geometry.qChannels)
                }
            } else if geometry.isCompressedAttention(layer), let hca = geometry.hcaConfig {
                let z = Double(max(1, hca.oGroups) * hca.oLoraRank), inC = Double(hca.qLoraRank + 3 * hca.headDim)
                trunkParams += inC * h + (Double(hca.qHeads * hca.headDim) + z) * Double(hca.qLoraRank)
                    + Double(hca.qHeads * hca.headDim * hca.oLoraRank) + h * z
            } else {
                trunkParams += Double(geometry.gdnTotalN) * h + h * Double(geometry.gdnZChannels)
            }
            if geometry.isMoELayer(layer) {
                moeLayers += 1
                trunkParams += Double(geometry.moeSplitFeatureRouting ? 2 * geometry.numExperts : geometry.numExperts) * h * (2.0 / tB)
                if let lat = geometry.moeLatentSize, lat > 0 { trunkParams += 2.0 * h * Double(lat) }
                if geometry.sharedExpertIntermediate > 0 { trunkParams += 3.0 * h * Double(geometry.sharedExpertIntermediate) }
            } else if geometry.intermediate > 0 {
                trunkParams += 3.0 * h * Double(geometry.intermediate)
            }
        }
        let headBytes = Double(geometry.vocab) * h * tB
        let bTrunk = trunkParams * tB + headBytes
        let bRouted: Double
        if moeLayers > 0 && geometry.numExperts > 0 && geometry.expertsPerTok > 0 {
            let routed = Double(moeLayers * geometry.expertsPerTok) * 3.0 * Double(geometry.moeRoutedDim) * Double(geometry.moeIntermediate)
            let coldFrac = Double(max(0, geometry.numExperts - geometry.effectiveResidentExperts)) / Double(geometry.numExperts)
            bRouted = routed * eB * (1.0 + coldFrac * 34.0)
        } else { bRouted = 0.0 }
        let rho = min(0.95, max(0.0, max(expertOverlap, geometry.numExperts > 0 ? Double(geometry.expertsPerTok) / Double(geometry.numExperts) : 0.0)))
        let bDraft = (trunkParams * tB + bRouted) / Double(max(1, geometry.numVirtualLayers)) + headBytes
        var bestK = 1, bestEff = -Double.infinity, expTok = 1.0, aPow = 1.0
        for k in 1...clampedMax {
            aPow *= alpha; expTok += aPow
            let eff = expTok / max(1.0, bTrunk + bRouted * (1.0 + Double(k) * (1.0 - rho)) + Double(k) * bDraft)
            if eff > bestEff { bestEff = eff; bestK = k }
        }
        return bestK
    }

    private mutating func selectBlockDraftCount(maxDrafts: Int, isSparseMoE: Bool) -> Int {
        let cap = min(maxDrafts, Self.maxSupportedDrafts)
        rounds += 1
        if isSparseMoE { return selectSparseMoEBlockCount(cap: cap) }
        if rounds == 1 { lastDraftCount = min(7, cap); return lastDraftCount }
        if lastAcceptedAll {
            if lastDraftCount == cap { return cap }
            if lastDraftCount < 7 || consecutiveFullAccepts >= 2 || (rounds == 2 && lastDraftCount == 7) {
                lastDraftCount = (lastDraftCount < 7) ? min(7, cap) : cap; return lastDraftCount
            }
        }
        if rounds % Self.exploreEvery == 0 {
            let mid = min(7, cap)
            if lastDraftCount < mid { lastDraftCount = mid; return mid }
            if lastDraftCount < cap && reach(drafts: mid) >= 0.35 { lastDraftCount = cap; return cap }
        }
        lastDraftCount = bestBucket(from: [min(3, cap), min(5, cap), min(7, cap), cap], isSparseMoE: false)
        return lastDraftCount
    }

    private mutating func selectSparseMoEBlockCount(cap: Int) -> Int {
        if rounds == 1 { lastDraftCount = min(3, cap); return lastDraftCount }
        let fastSparse = (observedBuckets & 2) != 0 && costMsBucket.b3 < 28.0
        let effCap = (fastSparse && reach(drafts: 5) < 0.65) ? min(5, cap) : cap
        if lastAcceptedAll {
            if lastDraftCount >= effCap { return effCap }
            if fastSparse && lastDraftCount < 3 { lastDraftCount = min(3, effCap); return lastDraftCount }
            let minConsec = fastSparse ? 1 : 2, minReach = fastSparse ? 0.22 : 0.52
            if consecutiveFullAccepts >= minConsec && (reach(drafts: lastDraftCount) >= minReach || (fastSparse && consecutiveFullAccepts >= 2)) {
                lastDraftCount = nextSparseBucket(after: lastDraftCount, cap: effCap); return lastDraftCount
            }
            return min(lastDraftCount, effCap)
        }
        if fastSparse, lastDraftCount <= 3, reach(drafts: 2) >= 0.48 { lastDraftCount = min(5, effCap); return lastDraftCount }
        if rounds % Self.exploreEvery == 0 {
            let b3 = min(3, effCap)
            if lastDraftCount < b3 { lastDraftCount = b3; return b3 }
            if lastDraftCount < effCap && reach(drafts: lastDraftCount) >= 0.48 {
                lastDraftCount = nextSparseBucket(after: lastDraftCount, cap: effCap); return lastDraftCount
            }
        }
        lastDraftCount = bestBucket(from: [min(2, effCap), min(3, effCap), min(5, effCap), min(7, effCap), effCap], isSparseMoE: true)
        return lastDraftCount
    }

    private func nextSparseBucket(after current: Int, cap: Int) -> Int {
        if current < 3 { return min(3, cap) }
        if current < 5 { return min(5, cap) }
        if current < 7 { return min(7, cap) }
        return cap
    }

    private func bestBucket(from candidates: [Int], isSparseMoE: Bool) -> Int {
        var best = candidates[0], bestRate = expectedTokens(drafts: best) / bucketCost(drafts: best, isSparseMoE: isSparseMoE)
        for cand in candidates.dropFirst() where cand != best {
            let rate = expectedTokens(drafts: cand) / bucketCost(drafts: cand, isSparseMoE: isSparseMoE)
            if rate > bestRate { bestRate = rate; best = cand }
        }
        return best
    }

    private mutating func observeBlockCost(draftCount: Int, rawMs: Double, isSparseMoE: Bool) {
        let msAlpha = 0.20, first = (observedBuckets == 0)
        let ms = (first && rawMs > 90.0) ? min(rawMs, isSparseMoE ? 25.0 : 47.0) : rawMs
        if isSparseMoE && draftCount <= 2 {
            observedBuckets |= 1; b2CostMs = first ? ms : (b2CostMs + msAlpha * (ms - b2CostMs))
        } else if draftCount <= 3 {
            observedBuckets |= 2; costMsBucket.b3 = first ? ms : (costMsBucket.b3 + msAlpha * (ms - costMsBucket.b3))
        } else if draftCount <= 5 {
            observedBuckets |= 4; costMsBucket.b5 = first ? ms : (costMsBucket.b5 + msAlpha * (ms - costMsBucket.b5))
        } else if draftCount <= 7 {
            observedBuckets |= 8; costMsBucket.b7 = first ? ms : (costMsBucket.b7 + msAlpha * (ms - costMsBucket.b7))
        } else {
            observedBuckets |= 16; costMsBucket.b15 = first ? ms : (costMsBucket.b15 + msAlpha * (ms - costMsBucket.b15))
        }
        if isSparseMoE, (observedBuckets & 2) != 0 {
            let b3 = costMsBucket.b3, fast = b3 < 28.0
            if (observedBuckets & 1) == 0 { b2CostMs = b3 * (fast ? 0.95 : (25.5 / 30.5)) }
            if (observedBuckets & 4) == 0 { costMsBucket.b5 = b3 * (fast ? 1.12 : (39.5 / 30.5)) }
            if (observedBuckets & 8) == 0 { costMsBucket.b7 = b3 * (fast ? 1.38 : (46.5 / 30.5)) }
            if (observedBuckets & 16) == 0 { costMsBucket.b15 = b3 * (fast ? 2.35 : (98.0 / 30.5)) }
        }
    }

    private func reach(drafts d: Int) -> Double {
        let limit = min(max(0, d), acc.count)
        var prob = 1.0
        for i in 0..<limit { prob *= acc[i] }
        return prob
    }

    private func bucketCost(drafts d: Int, isSparseMoE: Bool) -> Double {
        if isSparseMoE && d <= 2 { return b2CostMs }
        if d <= 3 { return costMsBucket.b3 }
        if d <= 5 { return costMsBucket.b5 }
        if d <= 7 { return costMsBucket.b7 }
        return costMsBucket.b15
    }
}
