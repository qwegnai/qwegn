import Foundation

/// A tentative output prefix bound to one preparation, not publishable output.
struct QWPreparedGreedyRound {
    fileprivate let identity: UUID
    let decision: QWGreedyRoundDecision
    let stochasticEvidence: QWStochasticRoundEvidence?

    fileprivate init(identity: UUID, decision: QWGreedyRoundDecision,
                     stochasticEvidence: QWStochasticRoundEvidence? = nil) {
        self.identity = identity
        self.decision = decision
        self.stochasticEvidence = stochasticEvidence
    }

    func truncating(to count: Int) throws -> QWPreparedGreedyRound {
        QWPreparedGreedyRound(identity: identity, decision: try decision.truncating(to: count),
                             stochasticEvidence: stochasticEvidence)
    }
}

struct QWGreedyRoundBoundary: Equatable {
    let position: Int
    let generated: Int
    let record: QWRoundRecord
    let hiddenValid: Bool
    let trunkRewind: UInt64
    let pairRewind: UInt64
}

/// Immutable baseline and output authorization for the driver's scratch lease.
struct QWGreedyRoundLease {
    let boundary: QWGreedyRoundBoundary
    let prepared: QWPreparedGreedyRound

    init(boundary: QWGreedyRoundBoundary, decision: QWGreedyRoundDecision,
         stochasticEvidence: QWStochasticRoundEvidence? = nil) {
        self.boundary = boundary
        prepared = QWPreparedGreedyRound(identity: UUID(), decision: decision,
                                       stochasticEvidence: stochasticEvidence)
    }

    func validateBoundary(_ current: QWGreedyRoundBoundary) throws {
        guard current == boundary else {
            throw QWEngineDriverError.state("prepared greedy round boundary changed")
        }
    }

    func validate(_ round: QWPreparedGreedyRound, current: QWGreedyRoundBoundary) throws {
        try validateBoundary(current)
        guard round.identity == prepared.identity,
              round.decision == (try prepared.decision.truncating(to: round.decision.tokens.count)) else {
            throw QWEngineDriverError.state("greedy commit requires an authorized prepared output prefix")
        }
    }
}
