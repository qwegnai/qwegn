import Foundation
import QWCore
import QWMTP

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
        QWAdaptiveDraftPolicy.optimalDraftCount(
            geometry: geometry,
            trunkBits: trunkBits,
            expertBits: expertBits,
            acceptRate: acceptRate,
            maxDrafts: maxDrafts,
            expertOverlap: expertOverlap
        )
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
              (1...15).contains(maximumDrafts) else {
            throw QWEngineDriverError.state("adaptive draft-depth policy received invalid input")
        }
        return completedDrafts < maximumDrafts && normalizedEntropy >= threshold
    }

    /// Caps `.hybrid` NextN neural chain depth to 1 when host-embedding gather is active
    /// or on sparse MoE (`numExperts > 0`), in both greedy and stochastic (`T > 0`) modes.
    static func hybridNextNNeuralDepthCap(
        isStochastic: Bool,
        stageProfiling: Bool,
        hasDeviceEmbedding: Bool,
        numExperts: Int
    ) -> Int? {
        QWAdaptiveDraftPolicy.neuralDepthCap(
            isStochastic: isStochastic,
            stageProfiling: stageProfiling,
            hasDeviceEmbedding: hasDeviceEmbedding,
            numExperts: numExperts
        )
    }
}
