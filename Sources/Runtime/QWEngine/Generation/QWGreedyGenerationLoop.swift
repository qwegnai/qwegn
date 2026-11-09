import Foundation
import QWCore

/// Deterministic speculation backoff state machine: tracks rolling draft acceptance,
/// adapts recommended draft depth, and switches to serial decode with exponential cooldown
/// when acceptance drops below `minAcceptance` at long contexts (`position >= 131_072`).
struct QWSpeculationBackoff: Equatable, Sendable {
    static let minPosition = 131_072
    var drafts: Int
    var minAcceptance: Double = 0.35
    var window: Int = 8
    var initialCooldown: Int = 64
    var maxCooldown: Int = 512

    private(set) var windowRounds: Int = 0
    private(set) var windowAccepted: Int = 0
    private(set) var serialLeft: Int = 0
    private(set) var currentCooldown: Int?
    private(set) var recommendedDrafts: Int

    init(
        drafts: Int,
        minAcceptance: Double = 0.35,
        window: Int = 8,
        initialCooldown: Int = 64,
        maxCooldown: Int = 512
    ) {
        self.drafts = drafts
        self.minAcceptance = minAcceptance
        self.window = window
        self.initialCooldown = initialCooldown
        self.maxCooldown = maxCooldown
        self.recommendedDrafts = max(0, drafts)
    }

    var shouldSpeculate: Bool {
        drafts > 0 && serialLeft == 0
    }

    mutating func recordSpeculativeRound(acceptedDrafts: Int) {
        guard drafts > 0 else { return }
        windowRounds += 1
        windowAccepted += acceptedDrafts
        guard windowRounds >= window else { return }
        let rate = Double(windowAccepted) / Double(windowRounds * drafts)
        let cooldown = currentCooldown ?? initialCooldown
        if rate < minAcceptance {
            serialLeft = cooldown
            currentCooldown = min(maxCooldown, cooldown * 2)
            recommendedDrafts = max(1, drafts / 2)
        } else {
            currentCooldown = initialCooldown
            recommendedDrafts = rate < 0.60 ? max(1, drafts - 1) : drafts
        }
        windowRounds = 0
        windowAccepted = 0
    }

    mutating func recordSerialStep() {
        serialLeft = max(0, serialLeft - 1)
    }

    static func armed(position: Int, drafts: Int) -> QWSpeculationBackoff? {
        position >= minPosition && drafts > 0 ? QWSpeculationBackoff(drafts: drafts) : nil
    }
}

/// Output ownership seam: speculative closure returns only after state commit.
enum QWGreedyGenerationLoop {
    struct Result {
        let tokens: [Int]
        let stepMs: [Double]
        let stepTokenCounts: [Int]
    }

    static func run(vocab: Int, first: Int, maxNew: Int, stopIDs: Set<Int>,
                    onToken: ((Int) -> Bool)?,
                    ordinary: (Int) throws -> Int,
                    committedRound: ((Int) throws -> QWGreedyRoundDecision)?,
                    streamsCommittedRounds: Bool = false,
                    onStopToken: ((Int) -> Void)? = nil,
                    backoff: QWSpeculationBackoff? = nil,
                    canSpeculate: (() -> Bool)? = nil) throws -> Result {
        guard maxNew >= 0, (0..<vocab).contains(first),
              stopIDs.allSatisfy({ (0..<vocab).contains($0) }) else {
            throw QWEngineDriverError.state("invalid greedy generation budget or tokens")
        }
        var tokens: [Int] = [], stepMs: [Double] = [], counts: [Int] = []
        func result() -> Result { Result(tokens: tokens, stepMs: stepMs, stepTokenCounts: counts) }
        guard maxNew > 0, !stopIDs.contains(first) else { return result() }
        tokens.append(first)
        if onToken?(first) == false { return result() }
        var pending = first
        var activeBackoff = backoff
        while tokens.count < maxNew {
            let start = Date()
            if let committedRound, (onToken == nil || streamsCommittedRounds),
               (activeBackoff?.shouldSpeculate ?? true), (canSpeculate?() ?? true) {
                let decision = try committedRound(maxNew - tokens.count)
                guard decision.tokens.count <= maxNew - tokens.count,
                      decision.tokens.allSatisfy({ (0..<vocab).contains($0) && !stopIDs.contains($0) }),
                      decision.stopped == (decision.stoppedToken != nil),
                      decision.stoppedToken.map({ (0..<vocab).contains($0) && stopIDs.contains($0) }) ?? true,
                      !decision.tokens.isEmpty || decision.stopped else {
                    throw QWEngineDriverError.state("committed greedy round returned invalid output")
                }
                stepMs.append(Date().timeIntervalSince(start) * 1000)
                counts.append(decision.tokens.count)
                tokens.append(contentsOf: decision.tokens)
                activeBackoff?.recordSpeculativeRound(acceptedDrafts: decision.acceptedDrafts)
                var cancelAfterRound = false
                for token in decision.tokens {
                    // A round is already committed. Finish delivering its
                    // output before honoring a callback cancellation request.
                    if onToken?(token) == false { cancelAfterRound = true }
                }
                if decision.stopped {
                    if let token = decision.stoppedToken { onStopToken?(token) }
                    break
                }
                if cancelAfterRound { break }
                guard let next = decision.pendingToken else {
                    throw QWEngineDriverError.state("committed round has no pending token")
                }
                pending = next
            } else {
                let next = try ordinary(pending)
                activeBackoff?.recordSerialStep()
                guard (0..<vocab).contains(next) else { throw QWEngineDriverError.badToken(next) }
                stepMs.append(Date().timeIntervalSince(start) * 1000)
                counts.append(stopIDs.contains(next) ? 0 : 1)
                // A stop token ends the request and is reported, never
                // appended: the caller decides what to do with it.
                if stopIDs.contains(next) { onStopToken?(next); break }
                tokens.append(next)
                pending = next
                if onToken?(next) == false { break }
            }
        }
        return result()
    }
}
