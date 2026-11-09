import Foundation
import QWHead

public struct QWMTPProposal: Sendable {
    public let token: Int
    public let distribution: QWSamplingDistribution
    fileprivate let owner: UUID
    fileprivate let drawIndex: UInt64
}

public struct QWMTPGumbelProposal: Sendable {
    public let token: Int
    public let distribution: QWSamplingDistribution
    fileprivate let owner: UUID
    fileprivate let counterBase: UInt64
}

/// One serialized owner, with no rewind/reset API. Engine callers must hold
/// their session flight lock. Legacy rejection draws and row-indexed Gumbel
/// counters share one persisted counter space.
public final class QWMTPRandomStream {
    public struct State: Sendable, Equatable {
        public let seed: UInt64
        public let drawsConsumed: UInt64
    }
    public let seed: UInt64
    public private(set) var drawsConsumed: UInt64
    public var isRoundActive: Bool { expectedDrafts != 0 }
    private let owner = UUID()
    private var expectedDrafts = 0
    private var sampled = 0
    private var gumbelRound: (start: UInt64, vocabulary: Int)?

    public init(seed: UInt64, drawIndex: UInt64 = 0) {
        self.seed = seed
        drawsConsumed = drawIndex
    }

    public convenience init(cloning state: State) {
        self.init(seed: state.seed, drawIndex: state.drawsConsumed)
    }

    public func frozenState() throws -> State {
        guard !isRoundActive else {
            throw QWMTP.ProtocolError.invalid("cannot capture an active proposal stream")
        }
        return State(seed: seed, drawsConsumed: drawsConsumed)
    }

    public func sampleTarget(_ distribution: QWSamplingDistribution) throws -> Int {
        guard !isRoundActive, drawsConsumed < UInt64.max else {
            throw QWMTP.ProtocolError.invalid("target draw requires an idle stream with counter capacity")
        }
        let token = try distribution.sampleToken(
            uniform: Double(QWPhilox.u(drawsConsumed, seed: seed, version: .upper24)), version: .upper24)
        drawsConsumed += 1
        return token
    }

    public func beginRound(drafts: Int, maxDrafts: Int = 2) throws {
        guard !isRoundActive, maxDrafts >= 1, (1...maxDrafts).contains(drafts) else {
            throw QWMTP.ProtocolError.invalid("stream round is active or draft count invalid")
        }
        let maximumDraws = UInt64(2 * drafts + 1)
        guard drawsConsumed <= UInt64.max - maximumDraws else {
            throw QWMTP.ProtocolError.invalid("stream cannot reserve a complete round without counter overflow")
        }
        expectedDrafts = drafts
        sampled = 0
        gumbelRound = nil
    }

    public func beginGumbelRound(drafts: Int, vocabulary: Int, maxDrafts: Int = 5) throws {
        guard !isRoundActive, maxDrafts >= 1, (1...maxDrafts).contains(drafts), vocabulary > 0 else {
            throw QWMTP.ProtocolError.invalid("Gumbel round draft count or vocabulary is invalid")
        }
        let (counterCount, overflow) = UInt64(drafts + 1).multipliedReportingOverflow(by: UInt64(vocabulary))
        guard !overflow, drawsConsumed <= UInt64.max - counterCount else {
            throw QWMTP.ProtocolError.invalid("stream cannot reserve a complete Gumbel round without counter overflow")
        }
        expectedDrafts = drafts
        sampled = 0
        gumbelRound = (drawsConsumed, vocabulary)
    }

    public func sampleProposal(_ distribution: QWSamplingDistribution) throws -> QWMTPProposal {
        guard isRoundActive, gumbelRound == nil, sampled < expectedDrafts else {
            throw QWMTP.ProtocolError.invalid("begin a round before sampling; resolve its full proposal window first")
        }
        let index = drawsConsumed
        let uniform = Double(QWPhilox.u(index, seed: seed, version: .upper24))
        let token = try distribution.sampleToken(uniform: uniform, version: .upper24)
        drawsConsumed += 1
        sampled += 1
        return QWMTPProposal(token: token, distribution: distribution, owner: owner, drawIndex: index)
    }

    public func sampleGumbelProposal(_ distribution: QWSamplingDistribution) throws -> QWMTPGumbelProposal {
        guard isRoundActive, let round = gumbelRound, sampled < expectedDrafts,
              distribution.vocabulary == round.vocabulary,
              drawsConsumed <= UInt64.max - UInt64(round.vocabulary) else {
            throw QWMTP.ProtocolError.invalid("Gumbel proposal does not match the reserved round")
        }
        let base = drawsConsumed
        let token = try QWMTP.checkedGumbelWinner(distribution: distribution, seed: seed, offset: base)
        drawsConsumed += UInt64(round.vocabulary)
        sampled += 1
        return QWMTPGumbelProposal(token: token, distribution: distribution,
                                   owner: owner, counterBase: base)
    }

    public func accept(proposals: [QWMTPProposal],
                       targetDistributions: [QWSamplingDistribution]) throws -> QWMTP.RoundResult {
        guard isRoundActive, gumbelRound == nil,
              sampled == expectedDrafts, proposals.count == expectedDrafts else {
            throw QWMTP.ProtocolError.invalid("stream decision needs the complete active proposal window")
        }
        let first = drawsConsumed - UInt64(sampled)
        for (index, proposal) in proposals.enumerated() {
            guard proposal.owner == owner, proposal.drawIndex == first + UInt64(index) else {
                throw QWMTP.ProtocolError.invalid("proposal is foreign, stale, reordered or not the latest window")
            }
        }
        let uniforms = (0...expectedDrafts).map {
            Double(QWPhilox.u(drawsConsumed + UInt64($0), seed: seed, version: .upper24))
        }
        let result = try QWMTP.checkedAcceptRound(
            draftDistributions: proposals.map(\.distribution), targetDistributions: targetDistributions,
            drafts: proposals.map(\.token), uniforms: uniforms, maxDrafts: max(2, expectedDrafts))
        drawsConsumed += UInt64(result.nDrawn)
        expectedDrafts = 0
        sampled = 0
        return result
    }

    public func resolveGumbelRound(proposals: [QWMTPGumbelProposal],
                                   targetDistributions: [QWSamplingDistribution]) throws -> QWMTP.GumbelRoundResult {
        guard isRoundActive, let round = gumbelRound,
              sampled == expectedDrafts, proposals.count == expectedDrafts,
              targetDistributions.count == expectedDrafts + 1,
              drawsConsumed == round.start + UInt64(expectedDrafts) * UInt64(round.vocabulary) else {
            throw QWMTP.ProtocolError.invalid("Gumbel decision needs a complete reserved proposal window")
        }
        for (index, proposal) in proposals.enumerated() {
            let expectedBase = round.start + UInt64(index) * UInt64(round.vocabulary)
            guard proposal.owner == owner, proposal.counterBase == expectedBase,
                  proposal.distribution.vocabulary == round.vocabulary else {
                throw QWMTP.ProtocolError.invalid("Gumbel proposal is foreign, stale, reordered or mismatched")
            }
        }
        let result = try QWMTP.checkedGumbelCouplingRound(
            draftDistributions: proposals.map(\.distribution), targetDistributions: targetDistributions,
            seed: seed, offset: round.start, maxDrafts: max(5, expectedDrafts))
        drawsConsumed = round.start + UInt64(expectedDrafts + 1) * UInt64(round.vocabulary)
        expectedDrafts = 0
        sampled = 0
        gumbelRound = nil
        return result
    }

    public func resolveRelaxedGumbelRound(proposals: [QWMTPGumbelProposal],
                                          targetDistributions: [QWSamplingDistribution],
                                          margin: Double) throws -> QWMTP.GumbelRoundResult {
        guard isRoundActive, let round = gumbelRound,
              sampled == expectedDrafts, proposals.count == expectedDrafts,
              targetDistributions.count == expectedDrafts + 1,
              drawsConsumed == round.start + UInt64(expectedDrafts) * UInt64(round.vocabulary) else {
            throw QWMTP.ProtocolError.invalid("Gumbel decision needs a complete reserved proposal window")
        }
        for (index, proposal) in proposals.enumerated() {
            let expectedBase = round.start + UInt64(index) * UInt64(round.vocabulary)
            guard proposal.owner == owner, proposal.counterBase == expectedBase,
                  proposal.distribution.vocabulary == round.vocabulary else {
                throw QWMTP.ProtocolError.invalid("Gumbel proposal is foreign, stale, reordered or mismatched")
            }
        }
        let result = try QWMTP.checkedRelaxedGumbelCouplingRound(
            draftDistributions: proposals.map(\.distribution),
            targetDistributions: targetDistributions,
            seed: seed,
            offset: round.start,
            margin: margin,
            maxDrafts: max(5, expectedDrafts)
        )
        drawsConsumed = round.start + UInt64(expectedDrafts + 1) * UInt64(round.vocabulary)
        expectedDrafts = 0
        sampled = 0
        gumbelRound = nil
        return result
    }

    /// Proposal draws are burned even if the GPU/request owner discards work.
    /// A decision-validation error does not consume any decision uniforms.
    public func discardRound() throws {
        guard isRoundActive else { throw QWMTP.ProtocolError.invalid("no active stream round to discard") }
        cancelActiveRound()
    }

    /// Idempotent request-failure/finish cleanup. Already consumed draws remain
    /// consumed, including partial proposal windows.
    public func cancelActiveRound() {
        expectedDrafts = 0
        sampled = 0
        gumbelRound = nil
    }
}
