import Foundation
import QWCore

enum QWMTPAdaptiveDepth {
    /// Chooses the optimal draft depth `k` in `1...maxDrafts` from model weight bytes and acceptance rate
    /// ("Expert Bytes, Kernel Bandwidth and the Best Draft Length", arXiv:2610.34440).
    static func optimalDraftCount(
        geometry: QWModelGeometry,
        trunkBits: Int = 4,
        expertBits: Int = 4,
        acceptRate: Double,
        maxDrafts: Int,
        expertOverlap: Double = 0.25
    ) -> Int {
        let clampedMax = max(1, min(15, maxDrafts))
        guard clampedMax > 1 else { return 1 }
        let alpha = min(1.0, max(0.0, acceptRate.isFinite ? acceptRate : 0.0))
        let h = Double(max(1, geometry.hidden))
        let tBytesPerElem = Double(max(1, trunkBits)) / 8.0
        let eBytesPerElem = Double(max(1, expertBits)) / 8.0

        var trunkParams = 0.0
        var moeLayerCount = 0
        for layer in geometry.executionPlan {
            if geometry.isFullAttention(layer) {
                if geometry.usesAnyMLA {
                    let qkvCols = Double(geometry.qLoraRank + geometry.kvLoraRank + geometry.indexerHeadDim)
                    let gateCols = geometry.attnOutputGate ? Double(geometry.qHeads * geometry.vHeadDim) : 0.0
                    let qBCols = Double(geometry.qHeads * (geometry.qkNopeHeadDim + geometry.qkRopeHeadDim)) + gateCols
                    let kvBCols = Double(geometry.qHeads * (geometry.qkNopeHeadDim + geometry.vHeadDim))
                    let oCols = Double(geometry.qHeads * geometry.vHeadDim)
                    trunkParams += qkvCols * h
                    trunkParams += qBCols * Double(geometry.qLoraRank)
                    trunkParams += kvBCols * Double(geometry.kvLoraRank)
                    trunkParams += h * oCols
                } else {
                    trunkParams += Double(geometry.attnTotalN) * h
                    trunkParams += h * Double(geometry.qChannels)
                }
            } else if geometry.isCompressedAttention(layer), let hca = geometry.hcaConfig {
                let inCols = Double(hca.qLoraRank + 3 * hca.headDim)
                let zCols = Double(max(1, hca.oGroups) * hca.oLoraRank)
                let qUpCols = Double(hca.qHeads * hca.headDim) + zCols
                let woAParams = Double(hca.qHeads * hca.headDim * hca.oLoraRank)
                trunkParams += inCols * h + qUpCols * Double(hca.qLoraRank) + woAParams + h * zCols
            } else {
                trunkParams += Double(geometry.gdnTotalN) * h
                trunkParams += h * Double(geometry.gdnZChannels)
            }
            if geometry.isMoELayer(layer) {
                moeLayerCount += 1
                let routerRows = Double(geometry.moeSplitFeatureRouting ? 2 * geometry.numExperts : geometry.numExperts)
                trunkParams += routerRows * h * (2.0 / tBytesPerElem)
                if let latent = geometry.moeLatentSize, latent > 0 {
                    trunkParams += 2.0 * h * Double(latent)
                }
                if geometry.sharedExpertIntermediate > 0 {
                    trunkParams += 3.0 * h * Double(geometry.sharedExpertIntermediate)
                }
            } else if geometry.intermediate > 0 {
                trunkParams += 3.0 * h * Double(geometry.intermediate)
            }
        }
        let headBytes = Double(geometry.vocab) * h * tBytesPerElem
        let bTrunk = trunkParams * tBytesPerElem + headBytes

        let bRoutedSingle: Double
        if moeLayerCount > 0 && geometry.numExperts > 0 && geometry.expertsPerTok > 0 {
            let routedDim = Double(geometry.moeRoutedDim)
            let routedParamsPerTok = Double(moeLayerCount * geometry.expertsPerTok) * 3.0 * routedDim * Double(geometry.moeIntermediate)
            let coldCount = max(0, geometry.numExperts - geometry.effectiveResidentExperts)
            let coldFrac = Double(coldCount) / Double(geometry.numExperts)
            // Effective bandwidth cost multiplier when a fraction of experts stream from SSD (~13 GB/s vs ~450 GB/s RAM).
            let coldBandwidthPenalty = 1.0 + coldFrac * 34.0
            bRoutedSingle = routedParamsPerTok * eBytesPerElem * coldBandwidthPenalty
        } else {
            bRoutedSingle = 0.0
        }

        let randomOverlap = geometry.numExperts > 0
            ? Double(geometry.expertsPerTok) / Double(geometry.numExperts)
            : 0.0
        let rho = min(0.95, max(0.0, max(expertOverlap, randomOverlap)))

        let numVirt = Double(max(1, geometry.numVirtualLayers))
        let bDraftLayer = (trunkParams * tBytesPerElem + bRoutedSingle) / numVirt
        let bDraft = bDraftLayer + headBytes

        var bestK = 1
        var bestEfficiency = -Double.infinity
        var expectedTokens = 1.0 // sum_{j=0..k} alpha^j; starts at j=0
        var alphaPow = 1.0
        for k in 1...clampedMax {
            alphaPow *= alpha
            expectedTokens += alphaPow
            let bRoutedUnion = bRoutedSingle * (1.0 + Double(k) * (1.0 - rho))
            let totalBytes = bTrunk + bRoutedUnion + Double(k) * bDraft
            let efficiency = expectedTokens / max(1.0, totalBytes)
            if efficiency > bestEfficiency {
                bestEfficiency = efficiency
                bestK = k
            }
        }
        return bestK
    }

    /// Computes the MoE-Spec (arXiv:2602.16052) unique expert budget across `verifyRows` tokens,
    /// preserving all `expertsPerTok` root-token experts and scaling draft tokens by survival mass.
    static func moeSpecDraftExpertBudget(
        geometry: QWModelGeometry,
        verifyRows: Int,
        acceptRate: Double = 0.75,
        explicitBudget: Int? = nil
    ) -> Int {
        let kTop = max(1, geometry.expertsPerTok)
        let rows = max(1, verifyRows)
        let maxUnion = rows * kTop
        if let explicitBudget, explicitBudget > 0 {
            return max(kTop, min(maxUnion, explicitBudget))
        }
        guard rows > 1 else { return kTop }
        let alpha = min(1.0, max(0.0, acceptRate.isFinite ? acceptRate : 0.75))
        var survivalMass = 1.0
        var alphaPow = 1.0
        for _ in 1..<rows {
            alphaPow *= alpha
            survivalMass += alphaPow
        }
        let rawBudget = Int((Double(kTop) * survivalMass).rounded(.up))
        return max(kTop, min(maxUnion, rawBudget))
    }

    static func normalizedEntropy(_ logits: UnsafeBufferPointer<Float>) throws -> Double {
        guard let base = logits.baseAddress, logits.count > 1 else {
            throw QWEngineDriverError.state("adaptive draft entropy requires at least two logits")
        }
        let n = logits.count
        var maxFloat = -Float.infinity
        for i in 0..<n {
            let value = base[i]
            guard value.isFinite else {
                throw QWEngineDriverError.state("adaptive draft entropy received non-finite logits")
            }
            if value > maxFloat {
                maxFloat = value
            }
        }
        let maximum = Double(maxFloat)
        let cutoff = maxFloat - 25.0
        var partition = 0.0
        var weightedOffset = 0.0
        for i in 0..<n {
            let value = base[i]
            guard value > cutoff else { continue }
            let offset = Double(value) - maximum
            let mass = exp(offset)
            partition += mass
            weightedOffset += mass * offset
        }
        guard partition.isFinite, partition > 0, weightedOffset.isFinite else {
            throw QWEngineDriverError.state("adaptive draft entropy normalization failed")
        }
        let entropy = log(partition) - weightedOffset / partition
        let normalized = entropy / log(Double(n))
        guard normalized.isFinite, (0...1.000001).contains(normalized) else {
            throw QWEngineDriverError.state("adaptive draft entropy is outside its valid range")
        }
        return min(normalized, 1)
    }

    static func shouldStop(normalizedEntropy: Double, threshold: Double,
                           completedDrafts: Int, maximumDrafts: Int) throws -> Bool {
        guard normalizedEntropy.isFinite, (0...1).contains(normalizedEntropy),
              threshold.isFinite, (0...1).contains(threshold),
              completedDrafts >= 1, completedDrafts <= maximumDrafts,
              (1...5).contains(maximumDrafts) else {
            throw QWEngineDriverError.state("adaptive draft-depth policy received invalid input")
        }
        return completedDrafts < maximumDrafts && normalizedEntropy >= threshold
    }
}
