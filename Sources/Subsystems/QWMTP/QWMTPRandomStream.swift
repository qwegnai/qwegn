import Foundation
import QWHead

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
    public struct LCRoundWindow: Sendable, Equatable {
        public let seed: UInt64
        public let start: UInt64
        public let drafts: Int
        public var stride: UInt64 { UInt64(2 * drafts + 1) }

        public func acceptUniform(row: Int) -> Double {
            Double(QWPhilox.u(start + UInt64(2 * row), seed: seed, version: .upper24))
        }

        public func residualUniform(row: Int) -> Double {
            Double(QWPhilox.u(start + UInt64(2 * row + 1), seed: seed, version: .upper24))
        }

        public var bonusUniform: Double {
            Double(QWPhilox.u(start + UInt64(2 * drafts), seed: seed, version: .upper24))
        }
    }

    public let seed: UInt64
    public private(set) var drawsConsumed: UInt64
    public var isRoundActive: Bool { expectedDrafts != 0 || gumbelRound != nil || lcRound != nil }
    public private(set) var lcRound: LCRoundWindow?
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

    public func nextUniformUpper24() throws -> Double {
        guard !isRoundActive, drawsConsumed < UInt64.max else {
            throw QWMTP.ProtocolError.invalid("uniform draw requires an idle stream with counter capacity")
        }
        let u = Double(QWPhilox.u(drawsConsumed, seed: seed, version: .upper24))
        drawsConsumed += 1
        return u
    }

    public func reserveDraws(_ count: UInt64) throws -> UInt64 {
        guard !isRoundActive, drawsConsumed <= UInt64.max - count else {
            throw QWMTP.ProtocolError.invalid("draw reservation requires an idle stream with counter capacity")
        }
        let base = drawsConsumed
        drawsConsumed += count
        return base
    }

    @discardableResult
    public func beginLCRound(drafts: Int, maxDrafts: Int = 15) throws -> LCRoundWindow {
        guard !isRoundActive, maxDrafts >= 1, (1...maxDrafts).contains(drafts) else {
            throw QWMTP.ProtocolError.invalid("LC round draft count is invalid or round already active")
        }
        let stride = UInt64(2 * drafts + 1)
        guard drawsConsumed <= UInt64.max - stride else {
            throw QWMTP.ProtocolError.invalid("stream cannot reserve a complete LC round without counter overflow")
        }
        let window = LCRoundWindow(seed: seed, start: drawsConsumed, drafts: drafts)
        drawsConsumed += stride
        expectedDrafts = drafts
        sampled = 0
        lcRound = window
        return window
    }

    @discardableResult
    public func completeLCRound() throws -> LCRoundWindow {
        guard let window = lcRound else {
            throw QWMTP.ProtocolError.invalid("no active LC round to complete")
        }
        lcRound = nil
        expectedDrafts = 0
        sampled = 0
        return window
    }

    public func beginGumbelRound(drafts: Int, vocabulary: Int, maxDrafts: Int = 15) throws {
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
            seed: seed, offset: round.start, maxDrafts: max(15, expectedDrafts))
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
            maxDrafts: max(15, expectedDrafts)
        )
        drawsConsumed = round.start + UInt64(expectedDrafts + 1) * UInt64(round.vocabulary)
        expectedDrafts = 0
        sampled = 0
        gumbelRound = nil
        return result
    }

    /// Idempotent request-failure/finish cleanup. Already consumed draws remain
    /// consumed, including partial proposal windows.
    public func cancelActiveRound() {
        expectedDrafts = 0
        sampled = 0
        gumbelRound = nil
        lcRound = nil
    }
}
