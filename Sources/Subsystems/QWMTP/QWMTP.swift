// QWMTP.swift — C16: MTP speculative-decode protocol (Leviathan accept /
// residual-resample), the orchestrator's accept core. Pure CPU, no Metal:
// this file is bound bit-exactly by the golden replay (mtpA_*.bin fixtures
// from Scripts/oracle/gen_mtp_fixtures.py) BEFORE any GPU leg exists —
// golden-first law. The draft/verify legs (MTP block GEMMs, trunk batched
// forward) route through the gated components (QWGemm fc GEMV, C09 decode
// attn, C12/C13 GDN, C15 head) and land with the engine loop; this file is
import Foundation
import Accelerate
import QWHead

public enum QWMTP {
    public enum ProtocolError: Error, CustomStringConvertible {
        case invalid(String)
        public var description: String {
            switch self { case .invalid(let message): return "QWMTP protocol: \(message)" }
        }
    }

    public struct RoundResult: Sendable, Equatable {
        public let accepted: Int          // prefix length 0...N
        public let finalToken: Int        // t (always commits)
        public let nDrawn: Int            // uniforms consumed (acc+2 | N+1)
    }

    public struct GumbelRoundResult: Sendable, Equatable {
        public let accepted: Int
        public let finalToken: Int
        public let draftTokens: [Int]
    }

    /// Couples already-transformed sampler distributions. Using survivor
    /// probabilities preserves the configured temperature, top-k and top-p
    /// semantics while sharing token-indexed Philox Gumbel noise.
    public static func checkedGumbelCouplingRound(draftDistributions: [QWSamplingDistribution],
                                                  targetDistributions: [QWSamplingDistribution],
                                                  seed: UInt64,
                                                  offset: UInt64 = 0,
                                                  maxDrafts: Int = 5) throws -> GumbelRoundResult {
        let n = draftDistributions.count
        guard maxDrafts >= 1, (1...maxDrafts).contains(n), targetDistributions.count == n + 1,
              let vocabulary = targetDistributions.first?.vocabulary,
              vocabulary > 0,
              (draftDistributions + targetDistributions).allSatisfy({
                  $0.vocabulary == vocabulary && !$0.survivorIds.isEmpty
                      && $0.survivorIds.count == $0.survivorProbs.count
                      && zip($0.survivorIds, $0.survivorProbs).allSatisfy {
                          (0..<vocabulary).contains($0.0) && $0.1.isFinite && $0.1 > 0
                      }
              }) else {
            throw ProtocolError.invalid("Gumbel sampler distributions have invalid shapes or probabilities")
        }
        let width = UInt64(vocabulary)
        let (counterCount, countOverflow) = UInt64(n + 1).multipliedReportingOverflow(by: width)
        let (_, endOverflow) = offset.addingReportingOverflow(counterCount)
        guard !countOverflow, !endOverflow else {
            throw ProtocolError.invalid("Gumbel Philox counter overflow")
        }
        var winnerPairs: [[Int]] = []
        winnerPairs.reserveCapacity(n)
        for row in 0..<n {
            let (rowOffset, rowOverflow) = UInt64(row).multipliedReportingOverflow(by: width)
            let (base, baseOverflow) = offset.addingReportingOverflow(rowOffset)
            guard !rowOverflow, !baseOverflow else {
                throw ProtocolError.invalid("Gumbel Philox counter overflow")
            }
            winnerPairs.append([
                try gumbelArgmax(draftDistributions[row], seed: seed, base: base),
                try gumbelArgmax(targetDistributions[row], seed: seed, base: base)
            ])
        }
        let bonusOffset = UInt64(n) * width
        return try checkedGumbelCouplingRound(winnerPairs: winnerPairs,
            bonusTarget: gumbelArgmax(targetDistributions[n], seed: seed, base: offset + bonusOffset),
            vocabulary: vocabulary, maxDrafts: maxDrafts)
    }

    public static func checkedGumbelWinner(distribution: QWSamplingDistribution,
                                           seed: UInt64, offset: UInt64) throws -> Int {
        guard distribution.vocabulary > 0, !distribution.survivorIds.isEmpty,
              distribution.survivorIds.count == distribution.survivorProbs.count,
              zip(distribution.survivorIds, distribution.survivorProbs).allSatisfy({
                  (0..<distribution.vocabulary).contains($0.0) && $0.1.isFinite && $0.1 > 0
              }),
              offset <= UInt64.max - UInt64(distribution.vocabulary - 1) else {
            throw ProtocolError.invalid("Gumbel sampler distribution or counter is invalid")
        }
        return try gumbelArgmax(distribution, seed: seed, base: offset)
    }

    /// Couples draft and target Gumbel-max draws with the same token-indexed
    /// Philox noise. A mismatch commits the target winner and stops the prefix.
    /// Counters advance by vocabulary element, not by acceptance decision.
    public static func checkedGumbelCouplingRound(draftLogits: [[Float]],
                                                  targetLogits: [[Float]],
                                                  seed: UInt64,
                                                  offset: UInt64 = 0,
                                                  temperature: Float = 1) throws -> GumbelRoundResult {
        let n = draftLogits.count
        guard (1...5).contains(n), targetLogits.count == n + 1,
              let vocabulary = targetLogits.first?.count, vocabulary > 0,
              temperature.isFinite, temperature > 0 else {
            throw ProtocolError.invalid("Gumbel row shape, vocabulary or temperature")
        }
        for row in draftLogits + targetLogits {
            guard row.count == vocabulary, row.allSatisfy(\.isFinite) else {
                throw ProtocolError.invalid("Gumbel logits must be finite and have equal vocabulary size")
            }
        }
        let width = UInt64(vocabulary)
        var winnerPairs: [[Int]] = []
        winnerPairs.reserveCapacity(n)
        for row in 0..<n {
            let (rowOffset, rowOverflow) = UInt64(row).multipliedReportingOverflow(by: width)
            let (base, baseOverflow) = offset.addingReportingOverflow(rowOffset)
            guard !rowOverflow, !baseOverflow else {
                throw ProtocolError.invalid("Gumbel Philox counter overflow")
            }
            let draft = try gumbelArgmax(draftLogits[row], seed: seed, base: base,
                                         temperature: Double(temperature))
            let target = try gumbelArgmax(targetLogits[row], seed: seed, base: base,
                                          temperature: Double(temperature))
            winnerPairs.append([draft, target])
            if draft != target {
                return GumbelRoundResult(accepted: row, finalToken: target,
                                         draftTokens: winnerPairs.map { $0[0] })
            }
        }
        let (bonusOffset, bonusOverflow) = UInt64(n).multipliedReportingOverflow(by: width)
        let (bonusBase, baseOverflow) = offset.addingReportingOverflow(bonusOffset)
        guard !bonusOverflow, !baseOverflow else {
            throw ProtocolError.invalid("Gumbel Philox counter overflow")
        }
        let bonus = try gumbelArgmax(targetLogits[n], seed: seed, base: bonusBase,
                                     temperature: Double(temperature))
        return try checkedGumbelCouplingRound(winnerPairs: winnerPairs,
                                              bonusTarget: bonus, vocabulary: vocabulary)
    }

    /// Resolves the integer winner pairs emitted by the GPU comparison kernel.
    public static func checkedGumbelCouplingRound(winnerPairs: [[Int]],
                                                  bonusTarget: Int,
                                                  vocabulary: Int,
                                                  maxDrafts: Int = 5) throws -> GumbelRoundResult {
        guard maxDrafts >= 1, (1...maxDrafts).contains(winnerPairs.count), vocabulary > 0,
              (0..<vocabulary).contains(bonusTarget),
              winnerPairs.allSatisfy({
                  $0.count == 2 && (0..<vocabulary).contains($0[0]) && (0..<vocabulary).contains($0[1])
              }) else {
            throw ProtocolError.invalid("Gumbel winner-pair shape or token ID")
        }
        var accepted = 0
        for (row, pair) in winnerPairs.enumerated() {
            if pair[0] != pair[1] {
                return GumbelRoundResult(accepted: row, finalToken: pair[1],
                                         draftTokens: winnerPairs.prefix(row + 1).map { $0[0] })
            }
            accepted += 1
        }
        return GumbelRoundResult(accepted: accepted, finalToken: bonusTarget,
                                 draftTokens: winnerPairs.map { $0[0] })
    }

    // MARK: - Variational MTP Relaxed Acceptance (arXiv:2610.26693)

    /// Relaxed greedy acceptance (arXiv:2610.26693): accepts draft `d_i` when
    /// `p_target(d_i) >= exp(-margin) * max_v p_target(v)` and, for a positive
    /// `minProbability`, `p_target(d_i) >= minProbability` (the fidelity floor).
    public static func checkedRelaxedGreedyRound(drafts: [Int],
                                                 targetTokens: [Int],
                                                 targetLogits: [[Float]],
                                                 vocabulary: Int,
                                                 margin: Float,
                                                 minProbability: Float = 0,
                                                 maxDrafts: Int = 15) throws -> GumbelRoundResult {
        let n = drafts.count
        guard maxDrafts >= 1, (1...maxDrafts).contains(n),
              targetTokens.count == n + 1, targetLogits.count == n + 1,
              vocabulary > 0, margin.isFinite, margin >= 0,
              minProbability.isFinite, (0...1).contains(minProbability),
              drafts.allSatisfy({ (0..<vocabulary).contains($0) }),
              targetTokens.allSatisfy({ (0..<vocabulary).contains($0) }),
              targetLogits.allSatisfy({ $0.count == vocabulary && $0.allSatisfy(\.isFinite) }) else {
            throw ProtocolError.invalid("relaxed greedy round shape, vocabulary or margin is invalid")
        }
        var accepted = 0
        for row in 0..<n {
            let draftToken = drafts[row]
            let targetToken = targetTokens[row]
            if draftToken != targetToken {
                let draftLogit = targetLogits[row][draftToken]
                let targetLogit = targetLogits[row][targetToken]
                if margin <= 0 || draftLogit < targetLogit - margin
                    || !clearsProbabilityFloor(targetLogits[row], token: draftToken, minProbability: minProbability) {
                    return GumbelRoundResult(accepted: row, finalToken: targetToken,
                                             draftTokens: Array(drafts.prefix(row + 1)))
                }
            }
            accepted += 1
        }
        return GumbelRoundResult(accepted: accepted, finalToken: targetTokens[n],
                                 draftTokens: drafts)
    }

    /// `softmax(logits)[token] >= minProbability`; always true for `minProbability <= 0`.
    public static func clearsProbabilityFloor(_ logits: [Float], token: Int, minProbability: Float) -> Bool {
        guard minProbability > 0 else { return true }
        guard minProbability < 1, (0..<logits.count).contains(token),
              let peak = logits.max() else { return false }
        var sum: Double = 0
        for logit in logits { sum += exp(Double(logit - peak)) }
        return Double(logits[token] - peak) - log(sum) >= log(Double(minProbability))
    }

    /// Relaxed Gumbel coupling round (arXiv:2610.26693): accepts draft winner `d_i` when
    /// `d_i == t_i` or `log p_target(d_i) >= log p_target(t_i) - margin`.
    public static func checkedRelaxedGumbelCouplingRound(draftDistributions: [QWSamplingDistribution],
                                                         targetDistributions: [QWSamplingDistribution],
                                                         seed: UInt64,
                                                         offset: UInt64 = 0,
                                                         margin: Double = 0,
                                                         maxDrafts: Int = 15) throws -> GumbelRoundResult {
        let n = draftDistributions.count
        guard maxDrafts >= 1, (1...maxDrafts).contains(n), targetDistributions.count == n + 1,
              margin.isFinite, margin >= 0,
              let vocabulary = targetDistributions.first?.vocabulary,
              vocabulary > 0,
              (draftDistributions + targetDistributions).allSatisfy({
                  $0.vocabulary == vocabulary && !$0.survivorIds.isEmpty
                      && $0.survivorIds.count == $0.survivorProbs.count
                      && zip($0.survivorIds, $0.survivorProbs).allSatisfy {
                          (0..<vocabulary).contains($0.0) && $0.1.isFinite && $0.1 > 0
                      }
              }) else {
            throw ProtocolError.invalid("relaxed Gumbel sampler distributions or margin invalid")
        }
        let width = UInt64(vocabulary)
        let (counterCount, countOverflow) = UInt64(n + 1).multipliedReportingOverflow(by: width)
        let (_, endOverflow) = offset.addingReportingOverflow(counterCount)
        guard !countOverflow, !endOverflow else {
            throw ProtocolError.invalid("Gumbel Philox counter overflow")
        }
        var winnerPairs: [[Int]] = []
        winnerPairs.reserveCapacity(n)
        for row in 0..<n {
            let (rowOffset, rowOverflow) = UInt64(row).multipliedReportingOverflow(by: width)
            let (base, baseOverflow) = offset.addingReportingOverflow(rowOffset)
            guard !rowOverflow, !baseOverflow else {
                throw ProtocolError.invalid("Gumbel Philox counter overflow")
            }
            let draftWinner = try gumbelArgmax(draftDistributions[row], seed: seed, base: base)
            var targetWinner = try gumbelArgmax(targetDistributions[row], seed: seed, base: base)
            if draftWinner != targetWinner, margin > 0 {
                let pDraft = probability(targetDistributions[row], token: draftWinner)
                let pTarget = probability(targetDistributions[row], token: targetWinner)
                if pDraft > 0, pTarget > 0, log(pDraft) >= log(pTarget) - margin {
                    targetWinner = draftWinner
                }
            }
            winnerPairs.append([draftWinner, targetWinner])
        }
        let bonusOffset = UInt64(n) * width
        return try checkedGumbelCouplingRound(
            winnerPairs: winnerPairs,
            bonusTarget: gumbelArgmax(targetDistributions[n], seed: seed, base: offset + bonusOffset),
            vocabulary: vocabulary,
            maxDrafts: maxDrafts
        )
    }

    /// Relaxed sparse Leviathan rejection sampling (arXiv:2610.26693):
    /// Scales the acceptance ratio by `exp(margin)`: `u < min(1, exp(margin) * p(d_i) / q(d_i))`.
    public static func checkedRelaxedAcceptRound(draftDistributions: [QWSamplingDistribution],
                                                 targetDistributions: [QWSamplingDistribution],
                                                 drafts: [Int],
                                                 uniforms: [Double],
                                                 offset: Int = 0,
                                                 margin: Double = 0,
                                                 maxDrafts: Int = 5) throws -> RoundResult {
        guard margin.isFinite, margin >= 0 else {
            throw ProtocolError.invalid("relaxed margin must be finite and non-negative")
        }
        if margin == 0 {
            return try checkedAcceptRound(draftDistributions: draftDistributions,
                                          targetDistributions: targetDistributions,
                                          drafts: drafts, uniforms: uniforms,
                                          offset: offset, maxDrafts: maxDrafts)
        }
        let n = drafts.count
        guard maxDrafts >= 1, (1...maxDrafts).contains(n), draftDistributions.count == n,
              targetDistributions.count == n + 1, offset >= 0,
              offset <= uniforms.count, uniforms.count - offset >= n + 1 else {
            throw ProtocolError.invalid("sparse row shape, draft count or decision uniform window")
        }
        let vocabulary = targetDistributions[0].vocabulary
        try validateSparse(draftDistributions, vocabulary: vocabulary)
        try validateSparse(targetDistributions, vocabulary: vocabulary)
        guard uniforms[offset..<(offset + n + 1)].allSatisfy({ $0.isFinite && $0 >= 0 && $0 < 1 }) else {
            throw ProtocolError.invalid("decision uniform is outside [0,1)")
        }
        for i in 0..<n {
            guard probability(draftDistributions[i], token: drafts[i]) > 0 else {
                throw ProtocolError.invalid("draft ID absent from its proposal support")
            }
        }
        let relaxScale = exp(margin)
        for i in 0..<n {
            let q = draftDistributions[i], p = targetDistributions[i]
            let pDraft = probability(p, token: drafts[i])
            let qDraft = probability(q, token: drafts[i])
            let ratio = (pDraft * relaxScale) / qDraft
            if uniforms[offset + i] < min(1.0, ratio) { continue }
            var mass = 0.0
            walkResidual(draft: q, target: p) { _, value in mass += value }
            guard mass.isFinite, mass > 0 else {
                throw ProtocolError.invalid("zero or invalid sparse residual mass")
            }
            var cumulative = 0.0, selected: Int?
            walkResidual(draft: q, target: p) { token, value in
                guard value > 0 else { return }
                cumulative += value / mass
                if selected == nil, uniforms[offset + i + 1] < cumulative { selected = token }
            }
            if selected == nil {
                walkResidual(draft: q, target: p) { token, value in
                    if value > 0 { selected = token }
                }
            }
            guard let token = selected else {
                throw ProtocolError.invalid("sparse residual has no positive support")
            }
            return RoundResult(accepted: i, finalToken: token, nDrawn: i + 2)
        }
        let bonus = targetDistributions[n]
        var cumulative = 0.0, lastPositive: Int?
        for index in bonus.survivorIds.indices {
            let value = bonus.survivorProbs[index]
            if value > 0 { lastPositive = bonus.survivorIds[index] }
            cumulative += value
            if uniforms[offset + n] < cumulative {
                return RoundResult(accepted: n, finalToken: bonus.survivorIds[index], nDrawn: n + 1)
            }
        }
        guard let token = lastPositive else {
            throw ProtocolError.invalid("sparse bonus has no positive support")
        }
        return RoundResult(accepted: n, finalToken: token, nDrawn: n + 1)
    }

    private static func gumbelArgmax(_ logits: [Float], seed: UInt64, base: UInt64,
                                     temperature: Double) throws -> Int {
        var bestToken: Int?
        var bestScore = -Double.infinity
        for token in logits.indices {
            let (counter, overflow) = base.addingReportingOverflow(UInt64(token))
            guard !overflow else { throw ProtocolError.invalid("Gumbel Philox counter overflow") }
            let uniform = max(Double(QWPhilox.u(counter, seed: seed, version: .upper24)),
                              1.0 / 16_777_216.0)
            let score = Double(logits[token]) / temperature - log(-log(uniform))
            if score > bestScore {
                bestScore = score
                bestToken = token
            }
        }
        guard let bestToken else { throw ProtocolError.invalid("Gumbel logits have no selectable token") }
        return bestToken
    }

    private static func gumbelArgmax(_ distribution: QWSamplingDistribution,
                                     seed: UInt64, base: UInt64) throws -> Int {
        var bestToken: Int?
        var bestScore = -Double.infinity
        for (token, probability) in zip(distribution.survivorIds, distribution.survivorProbs) {
            let (counter, overflow) = base.addingReportingOverflow(UInt64(token))
            guard !overflow else { throw ProtocolError.invalid("Gumbel Philox counter overflow") }
            let uniform = max(Double(QWPhilox.u(counter, seed: seed, version: .upper24)),
                              1.0 / 16_777_216.0)
            let score = log(probability) - log(-log(uniform))
            if score > bestScore || (score == bestScore && bestToken.map { token < $0 } != false) {
                bestScore = score
                bestToken = token
            }
        }
        guard let bestToken else {
            throw ProtocolError.invalid("Gumbel distribution has no selectable token")
        }
        return bestToken
    }

    /// Sparse transformed supports are immutable sampler snapshots. Residual
    /// scans visit target IDs in ascending order without a vocabulary-sized row.
    public static func checkedAcceptRound(draftDistributions: [QWSamplingDistribution],
                                          targetDistributions: [QWSamplingDistribution],
                                          drafts: [Int], uniforms: [Double],
                                          offset: Int = 0,
                                          maxDrafts: Int = 2) throws -> RoundResult {
        let n = drafts.count
        guard maxDrafts >= 1, (1...maxDrafts).contains(n), draftDistributions.count == n,
              targetDistributions.count == n + 1, offset >= 0,
              offset <= uniforms.count, uniforms.count - offset >= n + 1 else {
            throw ProtocolError.invalid("sparse row shape, draft count or decision uniform window")
        }
        let vocabulary = targetDistributions[0].vocabulary
        try validateSparse(draftDistributions, vocabulary: vocabulary)
        try validateSparse(targetDistributions, vocabulary: vocabulary)
        guard uniforms[offset..<(offset + n + 1)].allSatisfy({ $0.isFinite && $0 >= 0 && $0 < 1 }) else {
            throw ProtocolError.invalid("decision uniform is outside [0,1)")
        }
        for i in 0..<n {
            guard probability(draftDistributions[i], token: drafts[i]) > 0 else {
                throw ProtocolError.invalid("draft ID absent from its proposal support")
            }
        }
        for i in 0..<n {
            let q = draftDistributions[i], p = targetDistributions[i]
            let ratio = probability(p, token: drafts[i]) / probability(q, token: drafts[i])
            if uniforms[offset + i] < min(1, ratio) { continue }
            var mass = 0.0
            walkResidual(draft: q, target: p) { _, value in mass += value }
            guard mass.isFinite, mass > 0 else {
                throw ProtocolError.invalid("zero or invalid sparse residual mass")
            }
            var cumulative = 0.0, selected: Int?
            walkResidual(draft: q, target: p) { token, value in
                guard value > 0 else { return }
                cumulative += value / mass
                if selected == nil, uniforms[offset + i + 1] < cumulative { selected = token }
            }
            // Roundoff at the upper CDF endpoint must return positive support,
            // not a trailing zero-probability token.
            if selected == nil {
                walkResidual(draft: q, target: p) { token, value in
                    if value > 0 { selected = token }
                }
            }
            guard let token = selected else {
                throw ProtocolError.invalid("sparse residual has no positive support")
            }
            return RoundResult(accepted: i, finalToken: token, nDrawn: i + 2)
        }
        let bonus = targetDistributions[n]
        var cumulative = 0.0, lastPositive: Int?
        for index in bonus.survivorIds.indices {
            let value = bonus.survivorProbs[index]
            if value > 0 { lastPositive = bonus.survivorIds[index] }
            cumulative += value
            if uniforms[offset + n] < cumulative {
                return RoundResult(accepted: n, finalToken: bonus.survivorIds[index], nDrawn: n + 1)
            }
        }
        guard let token = lastPositive else {
            throw ProtocolError.invalid("sparse bonus has no positive support")
        }
        return RoundResult(accepted: n, finalToken: token, nDrawn: n + 1)
    }

    private static func validateSparse(_ rows: [QWSamplingDistribution], vocabulary: Int) throws {
        for row in rows {
            let mass = row.survivorProbs.reduce(0, +)
            guard row.vocabulary == vocabulary, !row.survivorIds.isEmpty,
                  mass.isFinite, abs(mass - 1) <= 1e-8 else {
                throw ProtocolError.invalid("sparse distribution vocabulary or normalization invalid")
            }
        }
    }

    private static func probability(_ distribution: QWSamplingDistribution, token: Int) -> Double {
        let ids = distribution.survivorIds
        var low = 0, high = ids.count
        while low < high {
            let mid = low + (high - low) / 2
            if ids[mid] < token { low = mid + 1 } else { high = mid }
        }
        return low < ids.count && ids[low] == token ? distribution.survivorProbs[low] : 0
    }

    private static func walkResidual(draft: QWSamplingDistribution, target: QWSamplingDistribution,
                                     visit: (Int, Double) -> Void) {
        var proposal = 0
        for index in target.survivorIds.indices {
            let token = target.survivorIds[index]
            while proposal < draft.survivorIds.count, draft.survivorIds[proposal] < token { proposal += 1 }
            let q = proposal < draft.survivorIds.count && draft.survivorIds[proposal] == token
                ? draft.survivorProbs[proposal] : 0
            visit(token, max(0, target.survivorProbs[index] - q))
        }
    }

    /// Validated reference boundary. Proposal rows must be the distributions
    /// which actually sampled the draft IDs; decision uniforms use [0,1).
    public static func checkedAcceptRound(draftProbs: [[Double]], targetRows: [[Double]],
                                          drafts: [Int], uniforms: [Double],
                                          offset: Int = 0) throws -> RoundResult {
        let n = drafts.count
        guard (1...2).contains(n), draftProbs.count == n, targetRows.count == n + 1,
              let vocabulary = targetRows.first?.count, vocabulary > 0,
              offset >= 0, offset <= uniforms.count, uniforms.count - offset >= n + 1 else {
            throw ProtocolError.invalid("row shape, draft count or decision uniform window")
        }
        for row in draftProbs + targetRows {
            guard row.count == vocabulary, row.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
                throw ProtocolError.invalid("nonfinite, negative or malformed distribution")
            }
            let mass = row.reduce(0, +)
            guard mass.isFinite, abs(mass - 1) <= 1e-8 else {
                throw ProtocolError.invalid("distribution is not normalized")
            }
        }
        for (i, token) in drafts.enumerated() {
            guard (0..<vocabulary).contains(token), draftProbs[i][token] > 0 else {
                throw ProtocolError.invalid("draft ID absent from its proposal support")
            }
        }
        guard uniforms[offset..<(offset + n + 1)].allSatisfy({ $0.isFinite && $0 >= 0 && $0 < 1 }) else {
            throw ProtocolError.invalid("decision uniform is outside [0,1)")
        }
        // Validate the reachable rejection before entering the legacy core's
        // residual precondition. Floating-point rows can differ in total mass.
        for i in 0..<n {
            let token = drafts[i]
            if uniforms[offset + i] < min(1, targetRows[i][token] / draftProbs[i][token]) { continue }
            let residual = zip(targetRows[i], draftProbs[i]).reduce(0.0) { $0 + max(0, $1.0 - $1.1) }
            guard residual.isFinite, residual > 0 else {
                throw ProtocolError.invalid("zero or invalid residual mass")
            }
            break
        }
        return acceptRound(draftProbs: draftProbs, targetRows: targetRows,
                           drafts: drafts, uniforms: uniforms, offset: offset)
    }

    /// Inverse-CDF draw: ascending strict-cumsum scan (f64). The golden's
    /// `_scan` law verbatim — a `<=` here changes which token commits on
    /// the boundary mass; fixtures bind the strict form.
    public static func scan(_ p: UnsafeBufferPointer<Double>, _ u: Double) -> Int {
        var c = 0.0
        for i in 0..<p.count {
            c += p[i]
            if u < c { return i }
        }
        return p.count &- 1               // u within fp tail of 1.0
    }

    /// One speculative round over N drafts.
    ///   draftProbs[i] — draft distribution that SAMPLED d_i (length V)
    ///   targetRows[i] — target distribution at position s+i (length V);
    public static func acceptRound(draftProbs: [UnsafeBufferPointer<Double>],
                                   targetRows: [UnsafeBufferPointer<Double>],
                                   drafts: [Int],
                                   uniforms: UnsafeBufferPointer<Double>) -> RoundResult {
        let n = drafts.count
        precondition(draftProbs.count == n && targetRows.count == n + 1,
                     "shape law: N drafts ⇒ N draft rows + N+1 target rows")
        var drawn = 0
        for i in 0..<n {
            let d = drafts[i]
            let q = draftProbs[i][d], p = targetRows[i][d]
            let ratio = q > 0 ? p / q : .infinity
            let u = uniforms[drawn]; drawn += 1
            if u < min(1.0, ratio) { continue }           // accept d_i
            // reject: residual resample t ~ norm(max(0, p_row − q_row))
            let t = rejectResample(draft: draftProbs[i], target: targetRows[i],
                                   u: uniforms[drawn])
            drawn += 1
            return RoundResult(accepted: i, finalToken: t, nDrawn: drawn)
        }
        // all accepted: bonus from the last target row
        let t = targetRows[n].withUnsafeBufferPointer { scan($0, uniforms[drawn]) }
        drawn += 1
        return RoundResult(accepted: n, finalToken: t, nDrawn: drawn)
    }

    /// numpy-order residual resample (see RESID law): elementwise
    /// max(0, p−q) → SUM → divide each element by the sum → scan.
    /// (Scaling u by the sum looks equal but is not: the last ulp of the
    static func rejectResample(draft: UnsafeBufferPointer<Double>,
                               target: UnsafeBufferPointer<Double>,
                               u: Double) -> Int {
        precondition(draft.count == target.count, "draft/target rows differ in V")
        var res = [Double](repeating: 0, count: target.count)
        var sum = 0.0
        res.withUnsafeMutableBufferPointer { rp in
            let dp = draft.baseAddress!, tp = target.baseAddress!
            var i = 0
            while i < rp.count { rp[i] = Swift.max(0.0, tp[i] - dp[i]); i += 1 }
            vDSP_sveD(rp.baseAddress!, 1, &sum, vDSP_Length(rp.count))
        }
        precondition(sum > 0, "zero residual mass — q ≥ p pointwise (impossible q≠p)")
        res.withUnsafeMutableBufferPointer { rp in
            var i = 0
            while i < rp.count { rp[i] /= sum; i += 1 }      // divide-first law
        }
        return res.withUnsafeBufferPointer { scan($0, u) }
    }

    /// Heap-array convenience for the engine loop and the golden replay:
    /// maps rows to pointers inside ONE scope (no pointer escapes its
    /// buffer) and starts consuming uniforms at `offset` — the caller
    public static func acceptRound(draftProbs: [[Double]], targetRows: [[Double]],
                                   drafts: [Int], uniforms: [Double],
                                   offset: Int = 0) -> RoundResult {
        func withBufferList(_ arrays: [[Double]], _ index: Int, _ acc: [UnsafeBufferPointer<Double>],
                            _ body: ([UnsafeBufferPointer<Double>]) -> RoundResult) -> RoundResult {
            if index == arrays.count { return body(acc) }
            return arrays[index].withUnsafeBufferPointer { ptr in
                var nextAcc = acc
                nextAcc.append(ptr)
                return withBufferList(arrays, index + 1, nextAcc, body)
            }
        }
        return uniforms.withUnsafeBufferPointer { u in
            let view = UnsafeBufferPointer(start: u.baseAddress!.advanced(by: offset),
                                           count: u.count - offset)
            return withBufferList(draftProbs, 0, []) { dptr in
                withBufferList(targetRows, 0, []) { tptr in
                    acceptRound(draftProbs: dptr, targetRows: tptr,
                                drafts: drafts, uniforms: view)
                }
            }
        }
    }
}
