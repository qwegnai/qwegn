import Foundation
import QWAlloc
import QWCore
import QWHead
import QWMTP

struct QWStochasticState: Sendable, Equatable {
    let config: QWSamplerConfig
    let stream: QWMTPRandomStream.State
}

/// The driver request/flight owner serializes all access.
final class QWStochasticSession {
    let geometry: QWModelGeometry
    let config: QWSamplerConfig
    let stream: QWMTPRandomStream
    private let sampler: QWSampler

    init(geometry: QWModelGeometry, config: QWSamplerConfig, cloning state: QWMTPRandomStream.State? = nil) throws {
        guard config.uniformVersion == .upper24, state == nil || state?.seed == config.seed else {
            throw QWEngineDriverError.state("stochastic engine requires explicit v2 and a matching seed")
        }
        let sampler = QWSampler(config: config)
        _ = try sampler.distribution(logits: [0])
        self.geometry = geometry
        self.config = config
        self.sampler = sampler
        stream = state.map { QWMTPRandomStream(cloning: $0) } ?? QWMTPRandomStream(seed: config.seed)
    }

    func frozenState() throws -> QWStochasticState {
        QWStochasticState(config: config, stream: try stream.frozenState())
    }

    func distribution(logits: QWTensorValue,
                      repetitionHistory: [Int] = []) throws -> QWSamplingDistribution {
        guard logits.shape.dims.count == 1, logits.shape.dtype == .f32,
              (1...geometry.vocab).contains(logits.shape.dims[0]),
              logits.region.base >= 0, logits.region.base <= logits.region.buffer.length,
              logits.region.length >= 0,
              logits.region.length <= logits.region.buffer.length - logits.region.base,
              logits.offset >= 0, logits.offset <= logits.region.length,
              logits.region.base % 4 == 0, logits.offset % 4 == 0,
              logits.bytes <= logits.region.length - logits.offset else {
            throw QWEngineDriverError.state("stochastic sampling requires a complete FP32 logit row")
        }
        let pointer = logits.region.pointer.advanced(by: logits.offset).assumingMemoryBound(to: Float.self)
        return try sampler.distribution(
            logits: UnsafeBufferPointer(start: pointer, count: logits.shape.dims[0]),
            repetitionHistory: repetitionHistory)
    }
}

struct QWPreparedStochasticRound {
    let authorization: QWPreparedGreedyRound
    var decision: QWGreedyRoundDecision { authorization.decision }
    var evidence: QWStochasticRoundEvidence? { authorization.stochasticEvidence }

    func truncating(to count: Int) throws -> QWPreparedStochasticRound {
        QWPreparedStochasticRound(authorization: try authorization.truncating(to: count))
    }
}

struct QWStochasticRoundEvidence {
    let proposals: [QWMTPGumbelProposal]
    let targets: [QWSamplingDistribution]
    let result: QWMTP.GumbelRoundResult
    let drawStart: UInt64
    let drawEnd: UInt64
}

struct QWStochasticRoundWitness {
    let owner: QWStochasticSession
    let drawEnd: UInt64

    func validate(_ session: QWEngineSession) throws {
        guard session.stochastic === owner, !owner.stream.isRoundActive,
              owner.stream.drawsConsumed == drawEnd else {
            throw QWEngineDriverError.state("stochastic round stream identity or draw boundary changed")
        }

    }
}

extension QWGreedyRoundDecision {
    static func sampled(geometry: QWModelGeometry, proposals: [QWMTPGumbelProposal], result: QWMTP.GumbelRoundResult,
                        maxNew: Int, stopIDs: Set<Int>, maxDrafts: Int = 5) throws -> QWGreedyRoundDecision {
        guard maxDrafts >= 1, (1...maxDrafts).contains(proposals.count), (0...proposals.count).contains(result.accepted),
              (0..<geometry.vocab).contains(result.finalToken), maxNew > 0,
              result.draftTokens == proposals.prefix(min(result.accepted + 1, proposals.count)).map(\.token),
              stopIDs.allSatisfy({ (0..<geometry.vocab).contains($0) }) else {
            throw QWEngineDriverError.state("invalid sampled round result or output policy")
        }
        let candidates = proposals.prefix(result.accepted).map(\.token) + [result.finalToken]
        var tokens: [Int] = [], stoppedToken: Int?
        for token in candidates.prefix(maxNew) {
            if stopIDs.contains(token) { stoppedToken = token; break }
            tokens.append(token)
        }
        return QWGreedyRoundDecision(acceptedDrafts: result.accepted, tokens: tokens,
                                     stopped: stoppedToken != nil, stoppedToken: stoppedToken)
    }
}
