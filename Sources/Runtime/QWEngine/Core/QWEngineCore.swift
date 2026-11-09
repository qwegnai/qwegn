// QWEngineCore.swift — The one generation loop, over the module seam.
// The core owns the loop, the sampler and the session pool. A module owns
// the numerics. Neither one branches on the model family.

import Foundation
import QWCore
import QWHead

/// Everything the core needs to serve one model.
public struct QWEngineCoreConfiguration: Sendable {
    /// The frozen numeric plan of the loaded model.
    public var plan: QWEnginePlan
    /// The module that owns the numerics.
    public var module: any QWInferenceModule
    /// The sampling contract used when a request does not name one.
    public var sampling: QWSamplerConfig
    /// The speculation contract used when a request does not name one.
    public var speculative: QWSpeculativeParameters
    /// How many sessions may be live at once.
    public var maxSessions: Int

    public init(plan: QWEnginePlan, module: any QWInferenceModule,
                sampling: QWSamplerConfig, maxSessions: Int,
                speculative: QWSpeculativeParameters = QWSpeculativeParameters()) {
        self.plan = plan
        self.module = module
        self.sampling = sampling
        self.maxSessions = maxSessions
        self.speculative = speculative
    }
}

/// What one generation produced.
public struct QWGenerationResult: Sendable {
    /// The generated tokens, in order.
    public let tokens: [Int]
    /// Per-step latency in milliseconds.
    public let stepMs: [Double]
    /// How many tokens each step committed.
    public let stepTokenCounts: [Int]
    /// Wall time the prefill took. Absent when nothing measured it.
    public let prefillSeconds: Double?
    /// Wall time from the request to the first generated token.
    public let firstTokenSeconds: Double?
    /// Wall time the decode took.
    public let decodeSeconds: Double?
    /// GPU time the prefill spent, as the module measured it.
    public let prefillGPUSeconds: Double?
    /// The stop token that ended the turn, when one did. A turn that
    /// ran to its limit has no stop token: absence is not a zero.
    public let stopTokenID: Int?
    /// What the prefix cache did for this turn. Absent when no cache
    /// answered the prompt.
    public let cache: QWCacheAnswer?
    /// Speculative decoding telemetry summary, when speculative rounds ran.
    public let speculativeSummary: String?
    /// Per-token log-probabilities when `collectLogprobs` was requested.
    public let tokenLogprobs: [QWTokenLogprob]

    public init(tokens: [Int], stepMs: [Double], stepTokenCounts: [Int],
                prefillSeconds: Double? = nil, firstTokenSeconds: Double? = nil,
                decodeSeconds: Double? = nil, prefillGPUSeconds: Double? = nil,
                stopTokenID: Int? = nil, cache: QWCacheAnswer? = nil,
                speculativeSummary: String? = nil,
                tokenLogprobs: [QWTokenLogprob] = []) {
        self.tokens = tokens
        self.stepMs = stepMs
        self.stepTokenCounts = stepTokenCounts
        self.prefillSeconds = prefillSeconds
        self.firstTokenSeconds = firstTokenSeconds
        self.decodeSeconds = decodeSeconds
        self.prefillGPUSeconds = prefillGPUSeconds
        self.stopTokenID = stopTokenID
        self.cache = cache
        self.speculativeSummary = speculativeSummary
        self.tokenLogprobs = tokenLogprobs
    }
}

/// A session the core allocated and will recycle.
public final class QWEngineSessionHandle: @unchecked Sendable {
    /// The pool slot this session came from.
    public let id: Int
    /// The module-owned state behind it.
    public let state: any QWModuleSessionState

    public init(id: Int, state: any QWModuleSessionState) {
        self.id = id
        self.state = state
    }
}

/// The shared engine core: one loop, one sampler, one session pool.
public final class QWEngineCore: @unchecked Sendable {
    private let configuration: QWEngineCoreConfiguration
    private let pool: QWEngineSessionPool

    public init(configuration: QWEngineCoreConfiguration) throws {
        // The module and the plan must be the same model: the plan freezes
        // the vocabulary the loop guards, the module owns the real one.
        guard configuration.module.plan.digest == configuration.plan.digest else {
            throw QWModuleError.planMismatch(model: configuration.plan.modelId)
        }
        // The plan freezes the KV mode; the module says which modes it has.
        // A mismatch is refused loudly — never quietly downgraded to f16.
        let mode = configuration.plan.kvStorageMode
        let supported = type(of: configuration.module).supportedKVStorageModes
        guard supported.contains(mode) else {
            throw QWModuleError.unsupportedKVStorageMode(model: configuration.plan.modelId,
                                                         mode: mode,
                                                         supported: supported.sorted())
        }
        self.configuration = configuration
        self.pool = try QWEngineSessionPool(module: configuration.module,
                                            maxSessions: max(1, configuration.maxSessions))
    }

    /// Takes a session from the pool, or fails loudly when it is empty.
    public func acquireSession(sampling: QWSamplerConfig? = nil,
                               speculative: QWSpeculativeParameters? = nil) throws -> QWEngineSessionHandle {
        let contract = sampling ?? configuration.sampling
        let drafts = speculative ?? configuration.speculative
        guard let handle = try pool.acquire(sampling: contract, speculative: drafts) else {
            throw QWModuleError.sessionExhausted
        }
        return handle
    }

    /// Returns a session to the pool.
    public func releaseSession(_ handle: QWEngineSessionHandle) {
        pool.release(handle)
    }

    /// Generates tokens for a prompt. The loop is the core's; the sampling is
    /// the module's — the session is handed the request's contract, and a
    /// module that cannot honour it says so instead of answering greedily.
    public func generate(prompt: [Int], maxNew: Int, stopIDs: Set<Int>,
                         sampling: QWSamplerConfig? = nil,
                         speculative: QWSpeculativeParameters? = nil,
                         onToken: ((Int) -> Bool)? = nil,
                         onStopToken: ((Int) -> Void)? = nil) throws -> QWGenerationResult {
        let sampling = sampling ?? configuration.sampling
        let drafts = speculative ?? configuration.speculative
        // C5 (prefill entry): validate the WHOLE token list before any
        // encoder opens — a bad id must fail loud, not mid-stream.
        for token in prompt where token < 0 || token >= configuration.plan.vocab {
            throw QWModuleError.invalidToken(token)
        }
        let handle = try acquireSession(sampling: sampling, speculative: drafts)
        defer { releaseSession(handle) }
        let state = handle.state
        // The request is over when this returns: the module gives back what
        // it held for it — the engine session, the admission permit, the KV
        // pages — before the state goes back to the pool.
        defer { configuration.module.finish(state) }

        // The prompt is fed exactly once: the head runs on the last chunk and its
        // token is the first generated one — the dense engine's contract. An
        // empty prompt generates token 0, which the dense engine does too.
        var first = 0
        var chunkStart = 0
        var prefillSeconds: Double = 0
        do {
            // The module gets the request before the clock starts: what a
            // request needs up front must not land inside the measured prefill.
            try configuration.module.prepare(state, prompt: prompt, maxNew: maxNew)
            let prefillStart = Date()
            while chunkStart < prompt.count {
                let end = min(chunkStart + configuration.plan.prefillChunkRows, prompt.count)
                first = try configuration.module.encodePrefill(state, tokens: prompt,
                                                               chunk: chunkStart..<end,
                                                               runsHead: end == prompt.count)
                chunkStart = end
            }
            prefillSeconds = Date().timeIntervalSince(prefillStart)
        } catch let error as QWEngineError {
            // The engine reports with its own error type; the core speaks the
            // module's language — the prefill included.
            throw QWModuleError.from(error)
        }
        let prefillGPUSeconds = configuration.module.prefillGPUSeconds(state)
        // The first decode step feeds the generated token at the position
        // right after the prompt.
        var position = prompt.count
        let decodeStart = Date()
        // The module attributes the time of the step that produced a token.
        // A module that commits several tokens in one step splits that time
        // across them, so the caller never reports a busy GPU as a free step.
        var attributed: [Double] = []
        let output: QWGreedyGenerationLoop.Result
        do {
            output = try QWGreedyGenerationLoop.run(
                vocab: configuration.plan.vocab, first: first, maxNew: maxNew, stopIDs: stopIDs,
                onToken: onToken,
                ordinary: { pending in
                    let started = Date()
                    let next = try configuration.module.encodeDecodeStep(state, token: pending, position: position)
                    position += 1
                    let measured = Date().timeIntervalSince(started) * 1000.0
                    let reported = configuration.module.lastStepMilliseconds(state)
                    attributed.append(reported > 0 ? reported : measured)
                    return next
                },
                committedRound: nil,
                onStopToken: onStopToken)
        } catch let error as QWEngineError {
            // The shared loop reports with the dense driver's error type;
            // the core speaks the module's language.
            throw QWModuleError.from(error)
        }
        // The module's attribution wins where it has one: it knows how many
        // tokens its step committed.
        let stepMs = attributed.count == output.stepMs.count ? attributed : output.stepMs
        let moduleRounds = configuration.module.roundTokenCounts(state)
        let stepTokenCounts = !moduleRounds.isEmpty ? moduleRounds : output.stepTokenCounts
        return QWGenerationResult(tokens: output.tokens, stepMs: stepMs,
                                  stepTokenCounts: stepTokenCounts,
                                  prefillSeconds: prefillSeconds,
                                  firstTokenSeconds: prefillSeconds,
                                  decodeSeconds: output.stepMs.isEmpty ? 0 : Date().timeIntervalSince(decodeStart),
                                  prefillGPUSeconds: prefillGPUSeconds)
    }
}

extension QWModuleError {
    /// Maps the shared loop's error type onto the module's own.
    static func from(_ error: QWEngineError) -> QWModuleError {
        switch error {
        case .badToken(let token):
            return .invalidToken(token)
        case .state(let message):
            return .invalidState(message)
        default:
            return .invalidState(String(describing: error))
        }
    }
}
