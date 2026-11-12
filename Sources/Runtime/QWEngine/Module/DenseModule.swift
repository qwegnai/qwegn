// DenseModule.swift — The dense (qwen3_5) family module.
// The adapter: it owns no numerics, it drives the dense engine through the
// seam the core loops over.

import Foundation
import QWCore
import QWHead
import QWKV

/// One dense request behind the seam.
/// The engine owns the numerics; this owns the request: the session it
/// admitted, the permit it holds, and the timing it attributes.
final class DenseSessionState: QWModuleSessionState, @unchecked Sendable {
    /// The engine the request belongs to.
    let engine: QWEngineDriver
    /// The engine session this request owns.
    var engineSession: QWEngineSession?
    /// The permit this request holds on the engine.
    var permit: QWGenerationPermit?
    /// The sampling contract the request runs under.
    var sampling: QWSamplerConfig = .greedy
    /// The speculation contract the request runs under.
    var speculative: QWSpeculativeParameters = QWSpeculativeParameters()
    /// The wall time of the last decode step, in milliseconds.
    var stepMs: Double = 0
    /// The GPU seconds this request's prefill spent.
    var prefillGPUSec: Double = 0
    /// Where the prefill ledger stood before this request's prefill.
    var prefillMark: Int = 0
    /// Whether the prefill ledger was on before this request asked for it.
    var ledgerWasEnabled: Bool = false

    init(engine: QWEngineDriver) {
        self.engine = engine
    }

    var activeTokens: Int { engineSession?.position ?? 0 }

    func reset(sampling: QWSamplerConfig, speculative: QWSpeculativeParameters) throws {
        // A request that ended without `finish` still holds the engine. Give
        // the KV pages and the permit back before the next request starts.
        release()
        self.sampling = sampling
        self.speculative = speculative
        stepMs = 0
        prefillGPUSec = 0
        prefillMark = 0
    }

    /// Gives the engine back: the session's KV pages first, then the permit.
    func release() {
        if let engineSession { engine.finishSession(engineSession) }
        engineSession = nil
        if let permit { engine.releaseRequest(permit) }
        permit = nil
        // The prefill ledger goes back to how the engine had it.
        engine.prefillTimingLedger.setEnabled(ledgerWasEnabled)
    }
}

/// The dense 27B family, served by `QWEngineDriver`.
/// Not final: the standard families share this lifecycle (see
/// `QWStandardModule`), and only the contract differs.
open class DenseModule: QWInferenceModule, QWInferenceEngine, @unchecked Sendable {
    open class var family: QWModelDescriptor.Family { .qwen38Dense }

    /// The dense engine ships every capability the family can offer:
    /// DFlash, rope-YaRN, a real GDN prefill choice, adaptive MTP, vision.
    open class var capabilities: Set<QWKnobCapability> { [
        .prefixCache, .kvQuantQ8, .kvQuantQ4, .gdnPrefill, .dflash, .ropeYaRN,
        .speculative, .mtpAdaptive, .vision, .lsa,
    ] }

    /// `QWKVMode` has no f32 path in the dense engine.
    open class var supportedKVStorageModes: Set<String> { ["f16", "q8", "q4"] }

    /// Every kernel door the dense engine dispatches.
    open class var requiredDoors: [String] { QWKernelInventory.dense }

    /// The dense engine. Nil before the engine is loaded.
    public let engine: QWEngineDriver?
    /// The frozen numeric plan of the loaded model.
    public let plan: QWEnginePlan
    private var storedOnCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)?
    /// Where the module reports what its prefix cache did.
    public var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)? {
        get { engine?.onCacheEvent ?? storedOnCacheEvent }
        set {
            storedOnCacheEvent = newValue
            engine?.onCacheEvent = newValue
        }
    }
    /// The kernel each layer of the loaded model runs.
    private let kernels: [any QWLayerKernel]

    public required init(engine: QWEngineDriver) {
        self.engine = engine
        self.plan = engine.plan
        self.kernels = engine.layerKernels
    }

    /// A module with no engine yet. Every call that needs the engine fails
    /// loud — it never pretends to serve.
    public required init(plan: QWEnginePlan) {
        self.engine = nil
        self.plan = plan
        self.kernels = []
    }

    open class func make(plan: QWEnginePlan,
                         engine: (any QWLoadedEngine)?) throws -> any QWInferenceModule {
        guard let engine else { return Self(plan: plan) }
        guard let dense = engine as? QWEngineDriver else {
            throw QWModuleError.unsupportedFamily(family)
        }
        return Self(engine: dense)
    }

    public var capabilities: Set<QWKnobCapability> {
        engine?.capabilities ?? Self.capabilities
    }

    public var prefixCache: QWPrefixCacheState {
        engine?.prefixCache ?? QWPrefixCacheState(isConfigured: false)
    }

    public var report: QWEngineReport {
        engine?.report ?? QWEngineReport(
            capacityLog: "", arenaBytes: 0, weightUsedBytes: 0, weightReservedBytes: 0
        )
    }

    public var lastTurn: QWGenerationResult? { engine?.lastTurn }
    public var supportsTokenConstraints: Bool { engine?.supportsTokenConstraints ?? true }

    public func makeConversation() -> QWConversation? {
        engine?.makeConversation()
    }

    public func generate(request: QWGenerationRequest, onToken: ((Int) -> Bool)?) throws -> QWGenerationResult {
        guard let engine else {
            throw QWModuleError.notImplemented("DenseModule has no engine yet")
        }
        return try engine.generate(request: request, onToken: onToken)
    }

    public func buildStandalonePrefixCache(tokens: [Int], outputURL: URL, pinAsSharedBase: Bool = true) throws -> Int {
        guard let engine else {
            throw QWModuleError.notImplemented("DenseModule has no engine yet")
        }
        return try engine.buildStandalonePrefixCache(tokens: tokens, outputURL: outputURL, pinAsSharedBase: pinAsSharedBase)
    }

    public func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) {
        guard let engine else {
            throw QWModuleError.notImplemented("DenseModule has no engine yet")
        }
        return try engine.mountStandalonePrefixCache(from: fileURL)
    }

    public func unmountSharedBasePrefixes() {
        engine?.unmountSharedBasePrefixes()
    }

    /// The kernel each layer of the loaded model runs. The plan resolved the
    /// kinds when the model was loaded; the hot loop looks a kernel up.
    public var layerKernels: [any QWLayerKernel] { kernels }

    /// The KV storage mode the plan was frozen with.
    public var kvStorageMode: String { plan.kvStorageMode }

    /// Whether a prefix cache is configured on the engine right now.
    public var hasPrefixCache: Bool { engine?.hasPrefixCache ?? false }

    /// Configures the prefix cache on the underlying engine when present.
    public func configurePrefixCache(_ config: QWCacheConfig) throws {
        guard let engine else {
            throw QWModuleError.notImplemented(
                "DenseModule.configurePrefixCache — the seam prefill does not consult the cache yet")
        }
        try engine.configurePrefixCache(config: config)
    }

    public func makeSessionState() throws -> any QWModuleSessionState {
        guard let engine else {
            throw QWModuleError.notImplemented("DenseModule has no engine yet")
        }
        return DenseSessionState(engine: engine)
    }

    /// Prepares the engine for this request: the session is admitted for the
    /// prompt plus the tokens the request may generate, and the request
    /// holds the engine's admission permit until it finishes.
    public func prepare(_ state: any QWModuleSessionState, prompt: [Int], maxNew: Int) throws {
        let session = try unwrap(state)
        // The module gets the request's contract: a request it cannot serve
        // is refused here, not quietly downgraded later.
        try checkContract(sampling: session.sampling, speculative: session.speculative)
        guard let engine else {
            throw QWModuleError.notImplemented("DenseModule has no engine yet")
        }
        // The whole prompt is checked before any chunk opens an encoder: a
        // bad id must fail loud, not mid-stream.
        for token in prompt where token < 0 || token >= engine.geometry.vocab {
            throw QWModuleError.invalidToken(token)
        }
        // The seam's prefill is chunked. The engine's naive GDN mode decodes
        // the prompt one token at a time, which the seam cannot express — the
        // module refuses it rather than quietly running a different path.
        guard engine.gdnPrefillMode != .naive else {
            throw QWModuleError.notImplemented("DenseModule serves the chunked prefill only")
        }
        // The dense engine has one scratch and one KV cache: two requests
        // must never interleave, so the request holds the permit from here
        // until `finish` gives it back.
        let permit = try engine.acquireRequest()
        session.permit = permit
        do {
            session.engineSession = try engine.admitSession(promptTokens: prompt.count + maxNew)
        } catch {
            engine.releaseRequest(permit)
            session.permit = nil
            throw error
        }
        // The prefill's GPU time is measured by the engine's ledger. The
        // module turns it on for this request and puts it back afterwards.
        session.ledgerWasEnabled = engine.prefillTimingLedger.isEnabled
        engine.prefillTimingLedger.setEnabled(true)
        session.prefillMark = engine.prefillTimingLedger.values().count
    }

    /// Whether the module can serve a request's contract.
    /// The dense module serves the greedy contract only: a sampled, penalised
    /// or speculative request is refused, never quietly answered without it.
    func checkContract(sampling: QWSamplerConfig,
                       speculative: QWSpeculativeParameters) throws {
        guard sampling.isGreedy, sampling.repetitionPenalty == 1.0 else {
            throw QWModuleError.notImplemented(
                "DenseModule serves greedy sampling without a repetition penalty")
        }
        guard speculative.effectiveDrafts == 0 else {
            throw QWModuleError.notImplemented(
                "DenseModule serves no speculative round — the request asked for \(speculative.effectiveDrafts) drafts")
        }
    }

    /// The request is over: the KV pages and the permit go back to the engine.
    public func finish(_ state: any QWModuleSessionState) {
        guard let session = state as? DenseSessionState else { return }
        session.release()
    }

    /// Runs one prefill chunk and returns the head's token for its last row.
    public func encodePrefill(_ state: any QWModuleSessionState, tokens: [Int],
                               chunk: Range<Int>, runsHead: Bool) throws -> Int {
        guard let engine else {
            throw QWModuleError.notImplemented("DenseModule has no engine yet")
        }
        let session = try unwrap(state)
        guard let permit = session.permit, let engineSession = session.engineSession else {
            throw QWModuleError.invalidState("the dense module has no live request")
        }
        // The prompt is fed exactly once: each chunk the seam names is fed
        // once, and the head runs on the last one. The engine's own chunking
        // is the same size, so one seam chunk is one engine chunk — a longer
        // one is split, never dropped.
        engineSession.tokenHistory = tokens
        var offset = chunk.lowerBound
        var token = 0
        while offset < chunk.upperBound {
            let stepCap = QWEngineDriver.adaptivePrefillChunkCap(
                position: engineSession.position,
                maxChunk: engine.prefillChunkSize,
                isSubquadraticAttention: engine.usesSubquadraticLongPrefill)
            let count = min(stepCap, chunk.upperBound - offset)
            let last = offset + count == chunk.upperBound
            token = try engine.prefillChunk(engineSession, tokens: tokens,
                                            start: offset, count: count,
                                            runsHead: runsHead && last,
                                            mode: engine.gdnPrefillMode, permit: permit)
            offset += count
        }
        if runsHead {
            // The prefill's GPU time, as the engine's ledger measured it.
            session.prefillGPUSec = engine.prefillGPUSecondsSince(mark: session.prefillMark)
            if engine.prefillScratchEviction.shouldEvict(wiredEngineBytes: engine.arena.committedBytes) {
                engine.evictPrefillScratch()
            }
        }
        return token
    }

    /// Runs one decode step and returns the token the model produced.
    public func encodeDecodeStep(_ state: any QWModuleSessionState, token: Int,
                                 position: Int) throws -> Int {
        guard let engine else {
            throw QWModuleError.notImplemented("DenseModule has no engine yet")
        }
        let session = try unwrap(state)
        guard let permit = session.permit, let engineSession = session.engineSession else {
            throw QWModuleError.invalidState("the dense module has no live request")
        }
        let started = Date()
        let next = try engine.decodeStep(engineSession, token: token, permit: permit)
        // The module attributes the step's time to the token it produced.
        session.stepMs = Date().timeIntervalSince(started) * 1000.0
        return next
    }

    /// The step time the module attributes to the token it produced.
    public func lastStepMilliseconds(_ state: any QWModuleSessionState) -> Double {
        (state as? DenseSessionState)?.stepMs ?? 0
    }

    /// The GPU seconds the prefill of this session spent.
    public func prefillGPUSeconds(_ state: any QWModuleSessionState) -> Double {
        (state as? DenseSessionState)?.prefillGPUSec ?? 0
    }

    /// The logit row the shared sampler consumes. The greedy head already
    /// picked the token; this row is what a caller samples from.
    public func sampleLogits(_ state: any QWModuleSessionState) throws -> [Float] {
        guard let engine else {
            throw QWModuleError.notImplemented("DenseModule has no engine yet")
        }
        return engine.logitVector()
    }

    /// The concrete session behind the seam. A state this module did not make
    /// is an error, never a silently mis-typed session.
    private func unwrap(_ state: any QWModuleSessionState) throws -> DenseSessionState {
        guard let session = state as? DenseSessionState else {
            throw QWModuleError.unsupportedFamily(Self.family)
        }
        return session
    }
}

/// The dense engine is a loaded engine the registry can hand over.
extension QWEngineDriver: QWLoadedEngine {}

extension DenseModule {
    /// Builds the dense engine for a configuration. The module owns how
    /// its engine is made and configured, so the consumer names no
    /// concrete engine: it asks the module for an engine.
    public static func makeEngine(
        packPath: String,
        maxTokens: Int,
        kvMode: QWKVMode,
        mtpDrafts: Int,
        mtpBackend: QWMTPBackend?,
        prefillScratchEviction: QWPrefillScratchEvictionPolicy,
        gdnPrefill: String,
        cacheConfig: QWCacheConfig,
        mtpAdaptiveDepth: Bool,
        mtpAdaptiveEntropyThreshold: Double,
        mtpRelaxedMargin: Double = 1.5,
        mtpRelaxedMinProb: Double = QWSpeculativeParameters.defaultRelaxedMinProb,
        mtpDraftStrategy: String = "hybrid",
        mtpQuadraticSeeding: Bool = true,
        ropeYarnFactor: Double?,
        dflashDraftPackPath: String?,
        dflashSelectorTopK: Int,
        prefillChunkSize: Int,
        lsaConfig: QWLSAConfig = QWLSAConfig(),
        residentExpertsPerLayer: Int? = nil,
        osReserveGiB: Double? = nil,
        tuning: QWEngineTuning = .default
    ) throws -> QWEngineDriver {
        let engine = try QWEngineDriver(
            packPath: packPath,
            maxTokens: maxTokens,
            kvMode: kvMode,
            mtpDrafts: mtpDrafts,
            mtpBackend: mtpBackend,
            prefillScratchEviction: prefillScratchEviction,
            mtpAdaptiveDepth: mtpAdaptiveDepth,
            mtpAdaptiveEntropyThreshold: mtpAdaptiveEntropyThreshold,
            mtpRelaxedMargin: mtpRelaxedMargin,
            mtpDraftStrategy: mtpDraftStrategy,
            mtpQuadraticSeeding: mtpQuadraticSeeding,
            ropeYarnFactor: ropeYarnFactor,
            dflashDraftPackPath: dflashDraftPackPath,
            dflashSelectorTopK: dflashSelectorTopK,
            osReserveGiB: osReserveGiB,
            prefillChunkSize: prefillChunkSize,
            lsaConfig: lsaConfig,
            residentExpertsPerLayer: residentExpertsPerLayer,
            tuning: tuning
        )
        engine.mtpRelaxedMinProb = min(1.0, max(0.0, mtpRelaxedMinProb))
        switch gdnPrefill.lowercased() {
        case "naive": engine.gdnPrefillMode = GDNPrefillMode.naive
        case "chunked": engine.gdnPrefillMode = GDNPrefillMode.chunked
        case "regres": engine.gdnPrefillMode = GDNPrefillMode.regres
        default: break
        }
        if cacheConfig.warmByteLimit > 0 {
            try engine.configurePrefixCache(config: cacheConfig)
        }
        return engine
    }

    /// Builds the dense engine from what a consumer asked for. The
    /// consumer fills the options it knows; this module reads the ones
    /// the dense family understands.
    public static func makeEngine(options: QWEngineBuildOptions) throws -> any QWInferenceEngine {
        try makeEngine(
            packPath: options.packPath,
            maxTokens: options.maxTokens,
            kvMode: try Self.kvMode(from: options.kvMode),
            mtpDrafts: options.mtpDrafts,
            mtpBackend: options.mtpBackend,
            prefillScratchEviction: options.prefillScratchEviction,
            gdnPrefill: options.gdnPrefill,
            cacheConfig: options.cacheConfig
                ?? QWCacheConfig(warmByteLimit: 0, ssdByteLimit: 0,
                                 cacheDirectory: options.prefixCacheDirectory
                                    ?? QWCacheConfig.defaultCacheDirectory,
                                 wipeOnStart: false),
            mtpAdaptiveDepth: options.mtpAdaptiveDepth ?? false,
            mtpAdaptiveEntropyThreshold: options.mtpAdaptiveEntropyThreshold ?? 0,
            mtpRelaxedMargin: options.mtpRelaxedMargin,
            mtpRelaxedMinProb: options.mtpRelaxedMinProb,
            mtpDraftStrategy: options.mtpDraftStrategy,
            mtpQuadraticSeeding: options.mtpQuadraticSeeding,
            ropeYarnFactor: options.ropeYarnFactor,
            dflashDraftPackPath: options.dflashDraftPackPath,
            dflashSelectorTopK: options.dflashSelectorTopK ?? 16,
            prefillChunkSize: options.prefillChunkSize,
            lsaConfig: options.lsaConfig,
            residentExpertsPerLayer: options.residentExpertsPerLayer,
            osReserveGiB: options.osReserveGiB,
            tuning: options.tuning
        )
    }

    /// The KV layout the consumer asked for. A layout this family does
    /// not implement is refused, never silently replaced.
    static func kvMode(from raw: String) throws -> QWKVMode {
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "f16": return .f16
        case "q8": return .q8
        case "q4": return .q4
        default: throw QWModuleError.invalidState("unknown KV mode '\(raw)'")
        }
    }
}
