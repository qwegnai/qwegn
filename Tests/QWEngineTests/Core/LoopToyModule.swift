import Foundation
import QWCore
import QWHead
@testable import QWEngine

/// A module with no model behind it: it counts what it was asked to do and
/// emits a deterministic token sequence. It is the proof that a new model
/// shape is a data change, not an engine change.
final class LoopToyModule: QWInferenceModule, @unchecked Sendable {
    static let family: QWModelDescriptor.Family = .qwen38Dense
    static let capabilities: Set<QWKnobCapability> = []
    static let supportedKVStorageModes: Set<String> = ["f16"]
    static let requiredDoors: [String] = []

    /// The module's own session state.
    final class State: QWModuleSessionState, @unchecked Sendable {
        var activeTokens: Int = 0
        var decodeSteps: Int = 0
        var resets: Int = 0

        func reset(sampling: QWSamplerConfig, speculative: QWSpeculativeParameters) throws {
            activeTokens = 0
            decodeSteps = 0
            resets += 1
            self.sampling = sampling
            self.speculative = speculative
            self.onReset?(sampling, speculative)
        }

        /// The sampling contract the session was handed.
        var sampling: QWSamplerConfig = .greedy
        /// The speculation contract the session was handed.
        var speculative: QWSpeculativeParameters = QWSpeculativeParameters()
        /// Called with the contract every time the session is handed out.
        var onReset: ((QWSamplerConfig, QWSpeculativeParameters) -> Void)?
    }

    /// One decode step the core asked for.
    struct Decode: Equatable {
        /// The token the core fed in.
        let token: Int
        /// The position the core fed it at.
        let position: Int
    }

    let plan: QWEnginePlan
    /// The step time the toy attributes to the token it produced. The core
    /// must use the module's attribution when it has one.
    var attributedStepMs: Double = 0
    private let lock = NSLock()
    private var _prefillChunks: [Range<Int>] = []
    private var _decoded: [Decode] = []
    /// The sampling contract the session was handed, in order.
    private var _sampling: [QWSamplerConfig] = []
    private var _prepared: [[Int]] = []
    /// The requests the toy has finished, counted.
    private var _finished: Int = 0
    /// The speculation contract the session was handed, in order.
    private var _speculative: [QWSpeculativeParameters] = []

    /// The toy reports no cache events, but it accepts the sink.
    var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)?

    /// The prefill chunks the core asked for, in order.
    var prefillChunks: [Range<Int>] {
        lock.lock(); defer { lock.unlock() }
        return _prefillChunks
    }

    /// The decode steps the core asked for, in order.
    var decoded: [Decode] {
        lock.lock(); defer { lock.unlock() }
        return _decoded
    }

    /// The sampling contracts the sessions were handed, in order.
    var samplingContracts: [QWSamplerConfig] {
        lock.lock(); defer { lock.unlock() }
        return _sampling
    }

    /// The speculation contracts the sessions were handed, in order.
    var speculativeContracts: [QWSpeculativeParameters] {
        lock.lock(); defer { lock.unlock() }
        return _speculative
    }

    init(plan: QWEnginePlan) {
        self.plan = plan
    }

    static func make(plan: QWEnginePlan,
                     engine: (any QWLoadedEngine)?) throws -> any QWInferenceModule {
        LoopToyModule(plan: plan)
    }

    /// The toy module builds no engine: it proves the seam, not a model.
    static func makeEngine(options: QWEngineBuildOptions) throws -> any QWInferenceEngine {
        throw QWModuleError.notImplemented("LoopToyModule.makeEngine")
    }

    var kvStorageMode: String { plan.kvStorageMode }

    var hasPrefixCache: Bool { false }

    func configurePrefixCache(_ config: QWCacheConfig) throws {
        throw QWModuleError.notImplemented("LoopToyModule.configurePrefixCache")
    }

    func makeSessionState() throws -> any QWModuleSessionState {
        let state = State()
        state.onReset = { [weak self] sampling, speculative in
            guard let self else { return }
            self.lock.lock(); defer { self.lock.unlock() }
            self._sampling.append(sampling)
            self._speculative.append(speculative)
        }
        return state
    }

    func prepare(_ state: any QWModuleSessionState, prompt: [Int], maxNew: Int) throws {
        lock.lock(); defer { lock.unlock() }
        _prepared.append(prompt)
    }

    /// The toy holds nothing across a request: there is nothing to release.
    func finish(_ state: any QWModuleSessionState) {
        lock.lock(); defer { lock.unlock() }
        _finished += 1
    }

    /// How many requests the toy has finished.
    var finishedRequests: Int {
        lock.lock(); defer { lock.unlock() }
        return _finished
    }

    /// The prompts the module was prepared for, in order.
    var prepared: [[Int]] {
        lock.lock(); defer { lock.unlock() }
        return _prepared
    }

    func encodePrefill(_ state: any QWModuleSessionState, tokens: [Int],
                       chunk: Range<Int>, runsHead: Bool) throws -> Int {
        guard let toy = state as? State else {
            throw QWModuleError.notImplemented("LoopToyModule got a foreign session state")
        }
        toy.activeTokens += chunk.count
        lock.lock(); defer { lock.unlock() }
        _prefillChunks.append(chunk)
        // The head runs on the last chunk only, and its token is the first
        // generated one — the same contract the dense engine's prefill has.
        guard runsHead, let last = tokens.last else { return 0 }
        return (last + 1) % plan.vocab
    }

    func lastStepMilliseconds(_ state: any QWModuleSessionState) -> Double {
        attributedStepMs
    }

    func prefillGPUSeconds(_ state: any QWModuleSessionState) -> Double {
        0
    }

    func sampleLogits(_ state: any QWModuleSessionState) throws -> [Float] {
        // The toy has no logits. It refuses rather than answering with a row
        // of zeros — a numeric zero is not "unavailable".
        throw QWModuleError.notImplemented("LoopToyModule.sampleLogits")
    }

    func encodeDecodeStep(_ state: any QWModuleSessionState, token: Int,
                          position: Int) throws -> Int {
        guard let toy = state as? State else {
            throw QWModuleError.notImplemented("LoopToyModule got a foreign session state")
        }
        toy.decodeSteps += 1
        toy.activeTokens += 1
        lock.lock(); defer { lock.unlock() }
        _decoded.append(Decode(token: token, position: position))
        return (token + 1) % plan.vocab
    }
}
