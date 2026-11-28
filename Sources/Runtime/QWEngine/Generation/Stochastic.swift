// Stochastic.swift — QWEngineDriver Stochastic family (split from QWEngineDriver.swift; same-module extension).
import Foundation
import Darwin
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWMLP
import QWHead
import QWMTP
import QWQuant

extension QWEngineDriver {
    func sampleStochasticTarget(_ session: QWEngineSession,
                                       permit: QWGenerationPermit) throws -> Int {
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate(permit)
        guard let owner = session.stochastic, !session.sessionFinished, !session.stateFailed,
              !session.verificationPending, verifiedSession == nil else {
            throw QWEngineDriverError.state("target sampling requires an idle live stochastic session")
        }
        do {
            let repHistory = session.activeRepetitionHistory()
            let token: Int
            if let active = activeConstraint, let allowed = try active.constraint.allowedTokens() {
                let rawLogits = sc.logits.region.pointer.advanced(by: sc.logits.offset)
                    .assumingMemoryBound(to: Float.self)
                let rowBuf = UnsafeBufferPointer(start: rawLogits, count: geometry.vocab)
                let repSet = owner.config.repetitionPenalty == 1.0 ? Set<Int>() : Set(repHistory)
                token = try active.sampler.constrainedToken(logits: rowBuf, allowed: allowed, repetitionHistory: repSet)
            } else {
                token = try owner.stream.sampleTarget(
                    owner.distribution(logits: sc.logits, repetitionHistory: repHistory))
            }
            try activeConstraint?.constraint.accept(token)
            if (activeConstraint?.constraint.isUnconstrainedNow ?? true), let prev1 = session.tokenHistory.last {
                let prev2 = session.tokenHistory.count >= 2 ? session.tokenHistory[session.tokenHistory.count - 2] : nil
                session.tokenRecycler.recordTransition(previous: prev2, current: prev1, next: token)
            }
            if session.tokenHistory.count == session.position {
                session.tokenHistory.append(token)
            }
            session.roundRecord.consumedCount = session.position
            session.roundRecord.absolutePosition = session.position
            session.roundRecord.outputCount = session.generated
            session.roundRecord.pendingToken = token
            session.roundRecord.samplerDrawIndex = owner.stream.drawsConsumed
            return token
        } catch {
            // A trunk step already ran. Do not publish a retryable boundary
            // whose prediction was not sampled by the configured stream.
            session.stateFailed = true
            throw error
        }
    }

    func decodeStochastic(_ session: QWEngineSession, token: Int) throws -> Int {
        try requestGate.withRequest(flight: flight) { permit in
            try decodeStochastic(session, token: token, permit: permit)
        }
    }

    func decodeStochastic(_ session: QWEngineSession, token: Int,
                                  permit: QWGenerationPermit) throws -> Int {
        guard let owner = session.stochastic, !owner.stream.isRoundActive,
              owner.stream.drawsConsumed < UInt64.max, session.roundRecord.pendingToken == token else {
            throw QWEngineDriverError.state("stochastic decode requires its pending token and available draw capacity")
        }
        _ = try decodeStep(session, token: token, permit: permit)
        return try sampleStochasticTarget(session, permit: permit)
    }

    /// Explicit v2 diagnostic path. Callbacks select target-only sampling, not
    /// tentative speculative output. The public historical API is unchanged.
    public func generateExperimentalStochastic(session: QWEngineSession, prompt: [Int], maxNew: Int,
                                        stopIds: Set<Int> = [], speculative: Bool = true,
                                        onToken: ((Int) -> Bool)? = nil) throws -> [Int] {
        try generateExperimentalStochastic(session: session, prompt: prompt, maxNew: maxNew,
                                           stopIds: stopIds, speculative: speculative,
                                           onToken: onToken,
                                           onStopToken: { self.lastStopToken = $0 })
    }

    /// The stochastic turn, with the stop token reported.
    public func generateExperimentalStochastic(session: QWEngineSession, prompt: [Int], maxNew: Int,
                                        stopIds: Set<Int> = [], speculative: Bool = true,
                                        onToken: ((Int) -> Bool)? = nil,
                                        onStopToken: ((Int) -> Void)? = nil) throws -> [Int] {
        try requestGate.withRequest(flight: flight) { permit in
            guard let owner = session.stochastic, !owner.stream.isRoundActive,
                  maxNew >= 0, stopIds.allSatisfy({ (0..<geometry.vocab).contains($0) }),
                  !speculative || mtpDrafts > 0 else {
                throw QWEngineDriverError.state("invalid stochastic generation configuration or budget")
            }
            lastExperimentalGeneration = nil
            guard !prompt.isEmpty else { return [] }
            guard session.position + prompt.count < ropeRows else {
                throw QWEngineDriverError.posBeyondTable(session.position + prompt.count)
            }
            let effectiveMaxNew = min(maxNew, max(0, ropeRows - (session.position + prompt.count)))
            guard effectiveMaxNew > 0 else { return [] }
            guard owner.stream.drawsConsumed < UInt64.max else {
                throw QWEngineDriverError.state("stochastic target draw counter exhausted")
            }
            QWGemm.configureDenseNax(
                temperature: Double(owner.config.temperature),
                topP: Double(owner.config.topP),
                topK: owner.config.topK
            )
            let started = Date()
            _ = try prefill(session, tokens: prompt, mode: gdnPrefillMode, permit: permit)
            let first = try sampleStochasticTarget(session, permit: permit)
            let prefillSec = Date().timeIntervalSince(started), decoding = Date()
            let committedRound: ((Int) throws -> QWGreedyRoundDecision)? = speculative ? { remaining in
                let round = try self.prepareStochasticRound(session, maxNew: remaining,
                                                            stopIDs: stopIds, permit: permit)
                try self.commitStochasticVerification(session, round: round, drain: false, permit: permit)
                if let cb = session.lastCommitCB {
                    let failed = cb.status == .error || self.syncCommitFailure(for: session, cb: nil)
                    if failed {
                        session.lastCommitCB = nil
                        self.recordCommandBufferError(for: cb)
                        session.stateFailed = true
                        throw QWEngineDriverError.state("commit GPU failure during streaming speculative generation")
                    }
                }
                return round.decision
            } : nil
            defer {
                if let cb = session.lastCommitCB {
                    session.lastCommitCB = nil
                    cb.waitUntilCompleted()
                    _ = self.syncCommitFailure(for: session, cb: cb)
                }
            }
            let output = try QWGreedyGenerationLoop.run(vocab: geometry.vocab, first: first, maxNew: effectiveMaxNew, stopIDs: stopIds,
                onToken: onToken, ordinary: { token in
                    try self.decodeStochastic(session, token: token, permit: permit)
                }, committedRound: committedRound, streamsCommittedRounds: true, onStopToken: onStopToken,
                backoff: .armed(position: session.position, drafts: mtpDrafts),
                canSpeculate: {
                    self.canRunSpeculativeRound(session)
                        && self.canSpeculateActiveConstraint
                })
            if let cb = session.lastCommitCB {
                session.lastCommitCB = nil
                cb.waitUntilCompleted()
                _ = syncCommitFailure(for: session, cb: cb)
            } else {
                _ = syncCommitFailure(for: session, cb: nil)
            }
            guard !session.stateFailed else {
                throw QWEngineDriverError.state("commit GPU failure during experimental stochastic generation")
            }
            lastExperimentalGeneration = QWExperimentalGenerationStats(
                promptTokens: prompt.count, generatedTokens: output.tokens.count,
                prefillSec: prefillSec, decodeSec: Date().timeIntervalSince(decoding),
                roundMs: output.stepMs, roundTokenCounts: output.stepTokenCounts)
            return output.tokens
        }
    }

    /// Opt-in greedy speculative generation. Token callbacks receive only the
    /// initial token or tokens from committed verification rounds. Returning
    /// false stops after the current committed round.
    public func generateExperimentalGreedy(session: QWEngineSession, prompt: [Int], maxNew: Int,
                                           stopIds: Set<Int> = [],
                                           onToken: ((Int) -> Bool)? = nil,
                                           onStopToken: ((Int) -> Void)? = nil) throws -> [Int] {
        try requestGate.withRequest(flight: flight) { permit in
            guard session.stochastic == nil else {
                throw QWEngineDriverError.state("greedy generation cannot use a sampled session")
            }
            return try generateExperimentalGreedy(session: session, prompt: prompt, maxNew: maxNew,
                                           stopIds: stopIds, onToken: onToken,
                                           onStopToken: onStopToken, permit: permit)
        }
    }

    func generateExperimentalGreedy(session: QWEngineSession, prompt: [Int], maxNew: Int,
                                            stopIds: Set<Int>, onToken: ((Int) -> Bool)?,
                                            onStopToken: ((Int) -> Void)?,
                                            permit: QWGenerationPermit) throws -> [Int] {
        guard maxNew >= 0, stopIds.allSatisfy({ (0..<geometry.vocab).contains($0) }) else {
            throw QWEngineDriverError.state("invalid experimental generation budget or stop IDs")
        }
        lastExperimentalGeneration = nil
        guard !prompt.isEmpty else { return [] }
        guard mtpDrafts > 0 else { throw QWEngineDriverError.state("experimental generation requires MTP drafts") }
        guard session.position + prompt.count < ropeRows else {
            throw QWEngineDriverError.posBeyondTable(session.position + prompt.count)
        }
        let effectiveMaxNew = min(maxNew, max(0, ropeRows - (session.position + prompt.count)))
        guard effectiveMaxNew > 0 else { return [] }
        QWGemm.configureDenseNax(temperature: 0, topP: 1, topK: 1)
        let started = Date()
        let first = try prefill(session, tokens: prompt, mode: gdnPrefillMode, permit: permit)
        let prefillSec = Date().timeIntervalSince(started)
        let decoding = Date()
        let output = try QWGreedyGenerationLoop.run(vocab: geometry.vocab, 
            first: first, maxNew: effectiveMaxNew, stopIDs: stopIds, onToken: onToken,
            ordinary: { try self.decodeStep(session, token: $0, permit: permit) },
            committedRound: { remaining in
                let round = try self.prepareGreedyRound(session, maxNew: remaining, stopIDs: stopIds, permit: permit)
                try self.commitGreedyVerification(session, round: round, drain: false, permit: permit)
                if let cb = session.lastCommitCB {
                    let failed = cb.status == .error || self.syncCommitFailure(for: session, cb: nil)
                    if failed {
                        session.lastCommitCB = nil
                        self.recordCommandBufferError(for: cb)
                        session.stateFailed = true
                        throw QWEngineDriverError.state("commit GPU failure during streaming speculative generation")
                    }
                }
                return round.decision
            }, streamsCommittedRounds: true, onStopToken: onStopToken,
            backoff: .armed(position: session.position, drafts: mtpDrafts),
            canSpeculate: {
                self.canRunSpeculativeRound(session)
                    && self.canSpeculateActiveConstraint
            })
        if let cb = session.lastCommitCB {
            session.lastCommitCB = nil
            cb.waitUntilCompleted()
            _ = syncCommitFailure(for: session, cb: cb)
        } else {
            _ = syncCommitFailure(for: session, cb: nil)
        }
        guard !session.stateFailed else {
            throw QWEngineDriverError.state("commit GPU failure during experimental generation")
        }
        lastExperimentalGeneration = QWExperimentalGenerationStats(
            promptTokens: prompt.count, generatedTokens: output.tokens.count,
            prefillSec: prefillSec, decodeSec: Date().timeIntervalSince(decoding),
            roundMs: output.stepMs, roundTokenCounts: output.stepTokenCounts)
        return output.tokens
    }
}
