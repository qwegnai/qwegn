// QWTrunkRuntimeSupport.swift — Shared trunk penalty-window, sampling, decision-logit, GPU-turn, and spec-summary helpers.
import Foundation
import QWCore
import QWDecision
import QWHead

enum QWTrunkRuntimeSupport {
    static let repetitionPenaltyWindow = 256

    static func recordGeneratedTokenForPenalty(
        _ token: Int,
        vocab: Int,
        window: Int = repetitionPenaltyWindow,
        recentGenerated: inout [Int],
        activeLoopTokens: inout [Int]
    ) {
        guard token >= 0, token < vocab else { return }
        let effectiveWindow = max(0, window)
        guard effectiveWindow > 0 else {
            recentGenerated.removeAll(keepingCapacity: true)
            activeLoopTokens.removeAll(keepingCapacity: true)
            return
        }
        if recentGenerated.capacity < effectiveWindow + 1 {
            recentGenerated.reserveCapacity(effectiveWindow + 1)
        }
        recentGenerated.append(token)
        if recentGenerated.count > effectiveWindow {
            let drop = recentGenerated.count - effectiveWindow
            let keep = recentGenerated.count - drop
            recentGenerated.withUnsafeMutableBufferPointer { buf in
                if let base = buf.baseAddress, keep > 0 {
                    memmove(base, base.advanced(by: drop), keep * MemoryLayout<Int>.stride)
                }
            }
            recentGenerated.removeLast(drop)
        }
        QWSampler.loopRepetitionTokens(in: recentGenerated, into: &activeLoopTokens)
    }

    static func sampleToken(
        sampler: QWSampler,
        logitsRow: UnsafeBufferPointer<Float>,
        penaltyHistory: [Int],
        constraint: (any QWTokenConstraint)?,
        allowedScratch: inout [Int]
    ) throws -> Int {
        if let constraint, try constraint.writeAllowedTokens(into: &allowedScratch) {
            return try allowedScratch.withUnsafeBufferPointer { buf in
                try sampler.constrainedToken(logits: logitsRow, allowed: buf, repetitionHistory: penaltyHistory)
            }
        }
        if sampler.config.isGreedy {
            return try sampler.penalisedGreedyToken(logits: logitsRow, repetitionHistory: penaltyHistory)
        }
        return try sampler.sample(logits: logitsRow, repetitionHistory: penaltyHistory).token
    }

    static func extractDecisionSlotOutput(
        logitsRow: UnsafePointer<Float>,
        vocab: Int,
        candidateTokenIds: [Int]
    ) -> QWDecisionSlotOutput {
        if !candidateTokenIds.isEmpty {
            var sparse: [Int: Float] = [:]
            sparse.reserveCapacity(candidateTokenIds.count)
            for tid in candidateTokenIds where tid >= 0 && tid < vocab {
                sparse[tid] = logitsRow[tid]
            }
            return QWDecisionSlotOutput(tokenLogits: sparse)
        }
        let full = vocab > 0 ? Array(UnsafeBufferPointer(start: logitsRow, count: vocab)) : []
        return QWDecisionSlotOutput(fullLogits: full)
    }

    @inline(__always)
    static func withGPUTurn<T>(
        arbitrator: QWGPUTurnArbitrator?,
        _ body: () throws -> T
    ) rethrows -> T {
        guard let arbitrator else { return try body() }
        return try arbitrator.withPrimaryStepSync(body)
    }

    static func formatSpeculativeSummary(
        label: String,
        accepted: Int,
        proposals: Int,
        stepTokenCounts: [Int]
    ) -> String? {
        guard proposals > 0 else { return nil }
        let rounds = max(1, stepTokenCounts.count)
        let totalTok = stepTokenCounts.reduce(0, +)
        return String(
            format: "%@=%d/%d, rounds=%d, mean=%.2f tok/rnd",
            label, accepted, proposals, rounds, Double(totalTok) / Double(rounds)
        )
    }

    static func formatRoundsSummary(stepTokenCounts: [Int]) -> String? {
        guard !stepTokenCounts.isEmpty else { return nil }
        let rounds = stepTokenCounts.count
        let totalTok = stepTokenCounts.reduce(0, +)
        return String(format: "rounds=%d, mean=%.2f tok/rnd", rounds, Double(totalTok) / Double(rounds))
    }

    static func recordGPUBusy(gpuSeconds: Double, wallSeconds: Double) {
        if gpuSeconds.isFinite, wallSeconds.isFinite, gpuSeconds > 0, wallSeconds > 0 {
            QWTelemetryRegistry.shared.record("gpu.busy_ratio", gpuSeconds / wallSeconds, provenance: .measured)
        }
    }
}
