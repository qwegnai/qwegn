// QWEngineLifecycleActor.swift — Server model lifecycle and inference execution actor.
// Manages lazy/eager model loading, explicit unload/restart, prefix caching,
// multimodal vision encoding, and multi-family inference execution.

import Foundation
import Metal
import QWCli
import QWCore
import QWDecision
import QWEngine
import QWHAL
import QWHead
import QWKV
import QWMM
import QWQuant
import QWToken
import QWVision

public actor QWEngineLifecycleActor {
    public struct LoadedState: Sendable {
        public let modelBackend: any QWModelBackend
        public var descriptor: QWModelDescriptor { modelBackend.descriptor }
        public let tokenizer: QWTokenizer
        public let chatTemplate: QWChatTemplate
        public var visionBackend: (any QWVisionBackend)? { modelBackend.visionBackend }
        public var params: QWEngineParametersInfo { modelBackend.parametersInfo }

        public init(
            modelBackend: any QWModelBackend,
            tokenizer: QWTokenizer,
            chatTemplate: QWChatTemplate
        ) {
            self.modelBackend = modelBackend
            self.tokenizer = tokenizer
            self.chatTemplate = chatTemplate
        }
    }

    enum State: Sendable {
        case unloaded
        case loading(Task<LoadedResult, any Error>)
        case loaded(LoadedState)
        case unloading
        case failed(String)
    }

    /// A load's result, and the moment the weights arrived. The moment is
    /// taken by the load itself, so a busy actor cannot make the load
    /// look shorter than it was.
    struct LoadedResult: Sendable {
        let loaded: LoadedState
        let arrivedAt: Date
    }

    /// The load the actor runs. The seam lets the lifecycle be observed
    /// without a model behind it. It is synchronous on purpose: a seam that
    /// could suspend would hide a load that still held the actor.
    typealias Load = @Sendable (Config, QWServerLogger, QWModelBackendRegistry) throws -> LoadedState

    /// Thread-safe snapshot of read-only status and configuration fields so
    /// HTTP status/model/dashboard routes and SSE header setup never block
    /// waiting for the actor lock while prefill or decode is executing.
    final class StatusSnapshotBox: @unchecked Sendable {
        let lock = NSLock()
        var config: Config
        var phase: QWEngineLifecyclePhase
        var phaseSince: Date
        var failureDetail: String?
        var capabilities: [QWKnobCapability]
        var loadedParams: QWEngineParametersInfo?
        var supportsThinkingTags: Bool = true
        var isInkling2Template: Bool = false
        var toolCallEnvelope: QWToolCallEnvelope = .standard
        var lastLoggedSamplingMessage: String?
        var mountedMemoryPack: QWEngineInfoResponse.MountedMemoryPackInfo? = nil

        init(config: Config, phaseSince: Date) {
            self.config = config
            self.phase = .unloaded
            self.phaseSince = phaseSince
            self.failureDetail = nil
            self.capabilities = []
            self.loadedParams = nil
        }

        func update(state: State, since: Date, config: Config) {
            let nextPhase: QWEngineLifecyclePhase
            var nextDetail: String? = nil
            var nextCapabilities: [QWKnobCapability] = []
            var nextParams: QWEngineParametersInfo? = nil
            var nextSupportsThinkingTags = true
            var nextIsInkling2Template = false
            var nextToolCallEnvelope: QWToolCallEnvelope = .standard
            switch state {
            case .unloaded:
                nextPhase = .unloaded
            case .loading:
                nextPhase = .loading
            case .loaded(let loaded):
                nextPhase = .loaded
                nextCapabilities = Array(loaded.modelBackend.capabilities)
                nextParams = loaded.params
                nextSupportsThinkingTags = loaded.chatTemplate.supportsThinkingTags
                nextIsInkling2Template = loaded.chatTemplate.isInkling2Template
                nextToolCallEnvelope = loaded.chatTemplate.toolCallEnvelope
            case .unloading:
                nextPhase = .unloading
            case .failed(let message):
                nextPhase = .failed
                nextDetail = message
            }
            lock.lock()
            self.config = config
            self.phase = nextPhase
            self.phaseSince = since
            self.failureDetail = nextDetail
            self.capabilities = nextCapabilities
            self.loadedParams = nextParams
            self.supportsThinkingTags = nextSupportsThinkingTags
            self.isInkling2Template = nextIsInkling2Template
            self.toolCallEnvelope = nextToolCallEnvelope
            lock.unlock()
        }

        func updateConfig(_ newConfig: Config) {
            lock.lock()
            self.config = newConfig
            lock.unlock()
        }

        func isLoaded() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            return phase == .loaded
        }

        func lifecyclePhase() -> QWEngineLifecyclePhase {
            lock.lock()
            defer { lock.unlock() }
            return phase
        }

        func loadedModelName() -> String? {
            lock.lock()
            defer { lock.unlock() }
            return phase == .loaded ? config.packName : nil
        }

        func lifecycleSince() -> Date? {
            lock.lock()
            defer { lock.unlock() }
            return phaseSince
        }

        func lifecycleInfo() -> QWEngineLifecycleInfo {
            lock.lock()
            defer { lock.unlock() }
            let model: String? = (phase == .unloaded) ? nil : config.packName
            return QWEngineLifecycleInfo(phase: phase, since: phaseSince,
                                         model: model, detail: failureDetail)
        }

        func engineState() -> QWEngineState {
            lock.lock()
            defer { lock.unlock() }
            let model: String? = (phase == .unloaded) ? nil : config.packName
            let info = QWEngineLifecycleInfo(phase: phase, since: phaseSince,
                                             model: model, detail: failureDetail)
            let isRes = (phase == .loaded)
            return QWEngineState(lifecycle: info,
                                 isLoaded: isRes,
                                 loadedModelName: isRes ? config.packName : nil,
                                 capabilities: capabilities)
        }

        func getModelName() -> String {
            lock.lock()
            defer { lock.unlock() }
            return config.modelName
        }

        func getConfig() -> Config {
            lock.lock()
            defer { lock.unlock() }
            return config
        }

        func getLoadedParamsInfo() -> QWEngineParametersInfo? {
            lock.lock()
            defer { lock.unlock() }
            return loadedParams
        }

        func getSupportsThinkingTags() -> Bool {
            lock.withLock { supportsThinkingTags }
        }

        func getIsInkling2Template() -> Bool {
            lock.withLock { isInkling2Template }
        }

        func getToolCallEnvelope() -> QWToolCallEnvelope {
            lock.withLock { toolCallEnvelope }
        }

        func shouldLogSamplingMessage(_ message: String) -> Bool {
            lock.lock()
            defer { lock.unlock() }
            if lastLoggedSamplingMessage == message {
                return false
            }
            lastLoggedSamplingMessage = message
            return true
        }

        func resetLoggedSamplingMessage(_ initialMessage: String? = nil) {
            lock.lock()
            lastLoggedSamplingMessage = initialMessage
            lock.unlock()
        }

        func updateMountedMemoryPack(_ info: QWEngineInfoResponse.MountedMemoryPackInfo?) {
            lock.lock()
            mountedMemoryPack = info
            lock.unlock()
        }

        func getMountedMemoryPackInfo() -> QWEngineInfoResponse.MountedMemoryPackInfo? {
            lock.lock()
            defer { lock.unlock() }
            return mountedMemoryPack
        }
    }

    // Internal, not private: the chat handler lives in
    // QWEngineLifecycleActor+Chat.swift, and private members
    // are invisible from another file.
    var config: Config
    let configPath: String?
    let logger: QWServerLogger
    let backendRegistry: QWModelBackendRegistry
    let load: Load
    public let gpuTurnArbitrator = QWGPUTurnArbitrator()
    public let residencyPool = QWResidencyPool()
    let decisionLoader: DecisionRunnerLoader
    var decisionSidecar: QWLoadedDecisionSidecar? = nil
    let decisionStatusBox = QWDecisionSidecarStatusBox()
    var state: State = .unloaded
    var phaseSince: Date
    let statusBox: StatusSnapshotBox
    /// Counts the loads this actor has begun. A load may only publish its
    /// result while its own number is still the current one — so an unload
    /// that arrives while the weights are in flight keeps the last word.
    var loadGeneration: UInt64 = 0
    struct CachedTokenPrefixEntry: Sendable {
        let packPath: String
        let cacheSalt: String
        let prefixWire: String
        let prefixTokens: [Int]
        let systemTokenCount: Int?
    }

    final class TokenPrefixCacheBox: @unchecked Sendable {
        private let lock = NSLock()
        private var entries: [CachedTokenPrefixEntry] = []

        func lookupBestPrefix(
            newPrefixWire: String,
            packPath: String,
            saltKey: String,
            markerUTF8: [UInt8]
        ) -> CachedTokenPrefixEntry? {
            let newUTF8Count = newPrefixWire.utf8.count
            return lock.withLock {
                var bestIdx: Int? = nil
                var bestLen = 0
                for (idx, entry) in entries.enumerated()
                where entry.packPath == packPath && entry.cacheSalt == saltKey && !entry.prefixWire.isEmpty {
                    let count = entry.prefixWire.utf8.count
                    guard count > bestLen, count <= newUTF8Count,
                          newPrefixWire.utf8.starts(with: entry.prefixWire.utf8) else { continue }
                    if count == newUTF8Count {
                        bestIdx = idx; bestLen = count
                    } else {
                        let deltaStart = newPrefixWire.utf8.index(newPrefixWire.utf8.startIndex, offsetBy: count)
                        if newPrefixWire.utf8[deltaStart...].starts(with: markerUTF8) {
                            bestIdx = idx; bestLen = count
                        }
                    }
                }
                guard let idx = bestIdx else { return nil }
                let matched = entries.remove(at: idx)
                entries.append(matched)
                return matched
            }
        }

        func store(_ entry: CachedTokenPrefixEntry, maxEntries: Int) {
            lock.withLock {
                entries.removeAll {
                    $0.packPath == entry.packPath && $0.cacheSalt == entry.cacheSalt && $0.prefixWire == entry.prefixWire
                }
                entries.append(entry)
                if entries.count > maxEntries {
                    entries.removeFirst(entries.count - maxEntries)
                }
            }
        }

        func clear() {
            lock.withLock { entries.removeAll(keepingCapacity: true) }
        }
    }

    final class MemoryDeltaCacheBox: @unchecked Sendable {
        private let lock = NSLock()
        private var cachedRoot: String = ""
        private var cachedStamp: String = ""
        private var cachedCount: Int = 0
        private var cachedAt: Date = .distantPast
        private static let ttlSeconds: TimeInterval = 2.0

        func cachedOrCompute(for manifest: QWMemoryManifest, rootPath: String) -> Int {
            let stamp = "\(manifest.packName)|\(manifest.gitHeadSHA ?? "")|\(manifest.files.count)|\(manifest.caches.count)"
            let now = Date()
            if let hit = lock.withLock({ () -> Int? in
                guard cachedRoot == rootPath,
                      cachedStamp == stamp,
                      now.timeIntervalSince(cachedAt) < Self.ttlSeconds else {
                    return nil
                }
                return cachedCount
            }) {
                return hit
            }
            let delta = QWMemoryPack.computeDelta(for: manifest)
            let count = delta.modifiedFiles.count + delta.deletedFiles.count
            lock.withLock {
                cachedRoot = rootPath
                cachedStamp = stamp
                cachedCount = count
                cachedAt = Date()
            }
            return count
        }

        func store(rootPath: String, manifest: QWMemoryManifest, modifiedFileCount: Int) {
            let stamp = "\(manifest.packName)|\(manifest.gitHeadSHA ?? "")|\(manifest.files.count)|\(manifest.caches.count)"
            lock.withLock {
                cachedRoot = rootPath
                cachedStamp = stamp
                cachedCount = modifiedFileCount
                cachedAt = Date()
            }
        }

        func invalidate() {
            lock.withLock {
                cachedRoot = ""
                cachedStamp = ""
                cachedAt = .distantPast
            }
        }
    }

    /// Non-reentrant FIFO async mutex protecting serialized vision weights access.
    final class VisionAccessLock: @unchecked Sendable {
        private let lock = NSLock()
        private var busy = false
        private var waiters: [CheckedContinuation<Void, Never>] = []

        func withLock<T>(_ operation: () async throws -> T) async rethrows -> T {
            await acquire()
            defer { release() }
            return try await operation()
        }

        private func acquire() async {
            let acquiredImmediately = lock.withLock { () -> Bool in
                if !busy {
                    busy = true
                    return true
                }
                return false
            }
            if acquiredImmediately { return }
            await withCheckedContinuation { cont in
                let resumeNow = lock.withLock { () -> Bool in
                    if !busy {
                        busy = true
                        return true
                    }
                    waiters.append(cont)
                    return false
                }
                if resumeNow { cont.resume() }
            }
        }

        private func release() {
            let next: CheckedContinuation<Void, Never>? = lock.withLock {
                if waiters.isEmpty {
                    busy = false
                    return nil
                }
                return waiters.removeFirst()
            }
            next?.resume()
        }
    }

    struct AdmittedRequest: @unchecked Sendable {
        let loaded: LoadedState
        let config: Config
        let loadGeneration: UInt64
    }

    let tokenPrefixCache = TokenPrefixCacheBox()
    let memoryDeltaCache = MemoryDeltaCacheBox()
    let vaultSetBox = QWMountedVaultSetBox()
    let visionCoordinator = VisionAccessLock()
    var modelTransitionActive: Bool = false
    var modelTransitionWaiters: [CheckedContinuation<Void, Never>] = []
    var userUnloadSequence: UInt64 = 0
    var configRevision: UInt64 = 0
    struct ActiveMemoryPackState: Sendable {
        let rootURL: URL
        let manifest: QWMemoryManifest
        let artifact: QWMemoryManifest.CacheArtifact
        let cacheFileURL: URL
        let tokens: [Int]
        let byteCount: Int
    }
    var activeMemoryPack: ActiveMemoryPackState? = nil
    var preMountKVMode: QWKVMode? = nil
    var preMountMaxContext: Int? = nil
    var inFlightRequests: Int = 0
    var inFlightDrainWaiters: [CheckedContinuation<Void, Never>] = []

    func withExclusiveTransition<T>(_ operation: () async throws -> T) async rethrows -> T {
        while modelTransitionActive {
            await withCheckedContinuation { cont in
                modelTransitionWaiters.append(cont)
            }
        }
        modelTransitionActive = true
        defer {
            modelTransitionActive = false
            let waiters = modelTransitionWaiters
            modelTransitionWaiters.removeAll(keepingCapacity: true)
            for waiter in waiters { waiter.resume() }
        }
        return try await operation()
    }

    func waitForModelTransitionsIfNeeded() async {
        while modelTransitionActive {
            await withCheckedContinuation { cont in
                modelTransitionWaiters.append(cont)
            }
        }
    }

    func waitForModelTransitionsAndSelectIfNeeded(
        requestedModel: String?,
        allowSameFamilyQuantSwitch: Bool = false
    ) async throws {
        await waitForModelTransitionsIfNeeded()
        try await selectModelIfNeeded(requestedModel, allowSameFamilyQuantSwitch: allowSameFamilyQuantSwitch)
    }

    func beginInFlightRequest() { inFlightRequests += 1 }
    func endInFlightRequest() {
        inFlightRequests = max(0, inFlightRequests - 1)
        if inFlightRequests == 0 && !inFlightDrainWaiters.isEmpty {
            let waiters = inFlightDrainWaiters
            inFlightDrainWaiters.removeAll(keepingCapacity: true)
            for waiter in waiters { waiter.resume() }
        }
    }

    func waitForInFlightRequests() async {
        guard inFlightRequests > 0 else { return }
        await withCheckedContinuation { continuation in
            if inFlightRequests == 0 {
                continuation.resume()
            } else {
                inFlightDrainWaiters.append(continuation)
            }
        }
    }

    /// How long the unload stays in `.unloading` before the phase settles
    /// to `.unloaded`. The release itself is quick; the window is what
    /// makes the phase observable to a reader.
    static let unloadPhaseWindow: UInt64 = 50_000_000

    public init(
        config: Config,
        logger: QWServerLogger = .shared,
        configPath: String? = nil,
        backendRegistry: QWModelBackendRegistry = .shared,
        decisionLoader: @escaping DecisionRunnerLoader = QWEngineLifecycleActor.defaultDecisionLoader
    ) {
        self.init(config: config, logger: logger, configPath: configPath,
                  backendRegistry: backendRegistry,
                  decisionLoader: decisionLoader,
                  load: { config, logger, registry in
                      try Self.performLoad(config: config, logger: logger, backendRegistry: registry)
                  })
    }

    init(
        config: Config,
        logger: QWServerLogger = .shared,
        configPath: String? = nil,
        backendRegistry: QWModelBackendRegistry = .shared,
        decisionLoader: @escaping DecisionRunnerLoader = QWEngineLifecycleActor.defaultDecisionLoader,
        load: @escaping Load
    ) {
        let initialSince = Date()
        self.config = config
        self.configPath = configPath
        self.logger = logger
        self.backendRegistry = backendRegistry
        self.decisionLoader = decisionLoader
        self.load = load
        self.phaseSince = initialSince
        self.statusBox = StatusSnapshotBox(config: config, phaseSince: initialSince)
    }

    public static func denseNaxSelectionLogMessage(temperature: Double, topP: Double, topK: Int,
                                                   paramsSource: String? = nil) -> String {
        QWSamplingParameters(temperature: temperature, topP: topP, topK: topK)
            .denseNaxSelectionLogMessage(paramsSource: paramsSource)
    }

    /// Where the server's resolved sampling params came from, derived from the
    /// resolver's per-key provenance so a CLI/env/model-config value is never
    /// labeled as the settings default.
    static func samplingParamsSourceLabel(_ sourceSummary: [String: String]) -> String {
        let sources = Set(["temperature", "topP", "topK"].compactMap { sourceSummary[$0] })
        guard sources.count == 1, let s = sources.first else {
            let joined = sources.sorted().joined(separator: "+")
            return "server default (source: \(joined.isEmpty ? "unrecorded" : joined))"
        }
        let label: String
        switch s {
        case "config-file": label = "settings.json"
        case "cli": label = "cli flag"
        case "env": label = "env"
        case "model-config": label = "model default"
        default: label = s
        }
        return "server default (from \(label))"
    }

    nonisolated public func isLoaded() -> Bool {
        statusBox.isLoaded()
    }

    /// The phase the engine is in right now.
    nonisolated public func lifecyclePhase() -> QWEngineLifecyclePhase {
        statusBox.lifecyclePhase()
    }

    /// The loaded model's name, when one is loaded. This is the pack that
    /// is really resident — not the name the user gave it.
    nonisolated public func loadedModelName() -> String? {
        statusBox.loadedModelName()
    }

    /// When the engine entered the phase it is in right now.
    nonisolated public func lifecycleSince() -> Date? {
        statusBox.lifecycleSince()
    }

    /// The phase, when the engine entered it, the model it holds and why
    /// it failed — the whole truth, and nothing dressed up.
    nonisolated public func lifecycleInfo() -> QWEngineLifecycleInfo {
        statusBox.lifecycleInfo()
    }

    /// The engine's state in one look. The phase, the resident pack and
    /// whether a model is resident are read together, so one response
    /// cannot contradict itself.
    nonisolated public func engineState() -> QWEngineState {
        statusBox.engineState()
    }

    /// Records a new state and the moment the engine entered it.
    func transition(to newState: State, since: Date? = nil) {
        let timestamp = since ?? Date()
        state = newState
        phaseSince = timestamp
        statusBox.update(state: newState, since: timestamp, config: config)
    }

    nonisolated public func getModelName() -> String { statusBox.getModelName() }
    nonisolated public func getConfig() -> Config { statusBox.getConfig() }
    nonisolated public func supportsThinkingTags() -> Bool { statusBox.getSupportsThinkingTags() }
    nonisolated public func isInkling2Template() -> Bool { statusBox.getIsInkling2Template() }
    nonisolated public func toolCallEnvelope() -> QWToolCallEnvelope { statusBox.getToolCallEnvelope() }
    nonisolated public func activeToolEnvelopes() -> [QWToolCallEnvelope] {
        QWToolCallEnvelope.envelopes(prioritizing: statusBox.getToolCallEnvelope())
    }
    nonisolated public func getMountedMemoryPackInfo() -> QWEngineInfoResponse.MountedMemoryPackInfo? {
        statusBox.getMountedMemoryPackInfo()
    }

    nonisolated func logSamplingSelectionIfChanged(_ message: String) {
        if statusBox.shouldLogSamplingMessage(message) {
            logger.log(.info, message)
        }
    }

    public func ensureLoaded() async throws -> LoadedState {
        while case .unloading = state {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        switch state {
        case .loaded(let loaded):
            return loaded
        case .loading(let task):
            // The load is already on its way: wait for that one.
            return try await task.value.loaded
        case .unloading:
            throw QWEngineError.unloading
        case .unloaded, .failed:
            let config = self.config
            let logger = self.logger
            let registry = self.backendRegistry
            let load = self.load
            let generation = loadGeneration &+ 1
            loadGeneration = generation
            // The load runs off the actor: the weights take seconds to
            // arrive, and a reader must be able to ask what is happening
            // while they do.
            let loadTask = Task.detached { () -> LoadedResult in
                let loaded = try load(config, logger, registry)
                return LoadedResult(loaded: loaded, arrivedAt: Date())
            }
            transition(to: .loading(loadTask))
            do {
                let result = try await loadTask.value
                // An unload that arrived while the weights were in flight
                // keeps the last word: the model does not come back on
                // its own. A request that is already running keeps the
                // weights it was given — the phase describes what the
                // engine holds now, not what one request still holds.
                if loadGeneration == generation {
                    let primaryBytes = max(
                        Self.deviceAllocatedBytes(),
                        UInt64(max(0, (result.loaded.params.weightsSizeBytes ?? 0) + (result.loaded.params.kvCacheSizeBytes ?? 0)))
                    )
                    let primaryLease = QWResidencyLease(
                        slot: .primaryLLM,
                        modelName: config.packName,
                        packPath: config.packPath,
                        estimatedBytes: primaryBytes,
                        computeUnit: "metal",
                        bypassesGPUArbitrator: false
                    )
                    await residencyPool.commitLease(primaryLease)
                    guard loadGeneration == generation else {
                        _ = await residencyPool.releaseLease(for: .primaryLLM)
                        throw QWEngineError.unloading
                    }
                    let initialSamplingMsg = Self.denseNaxSelectionLogMessage(
                        temperature: config.temperature, topP: config.topP, topK: config.topK,
                        paramsSource: Self.samplingParamsSourceLabel(config.sourceSummary))
                    statusBox.resetLoggedSamplingMessage(initialSamplingMsg)
                    if let active = activeMemoryPack {
                        _ = try? result.loaded.modelBackend.mountStandalonePrefixCache(from: active.cacheFileURL)
                    }
                    if let unified = (result.loaded.modelBackend as? QWUnifiedModelBackend)?.engine as? QWEngine {
                        unified.gpuTurnArbitrator = self.gpuTurnArbitrator
                    }
                    transition(to: .loaded(result.loaded), since: result.arrivedAt)
                    return result.loaded
                }
                throw QWEngineError.unloading
            } catch {
                if loadGeneration == generation { transition(to: .failed(Self.detail(for: error))) }
                throw error
            }
        }
    }

    /// Atomically resolves model selection, ensures loaded state under current generation, and retains an in-flight lease.
    func beginAdmittedRequest(
        requestedModel: String?,
        allowSameFamilyQuantSwitch: Bool = false
    ) async throws -> AdmittedRequest {
        while true {
            await waitForModelTransitionsIfNeeded()
            let preUnloadSeq = userUnloadSequence
            try await selectModelIfNeeded(requestedModel, allowSameFamilyQuantSwitch: allowSameFamilyQuantSwitch)
            if userUnloadSequence != preUnloadSeq {
                throw QWEngineError.unloading
            }
            let preConfigRev = configRevision
            _ = try await ensureLoaded()
            await waitForModelTransitionsIfNeeded()
            if userUnloadSequence != preUnloadSeq {
                throw QWEngineError.unloading
            }
            if case .loaded(let currentLoaded) = state,
               !modelTransitionActive,
               configRevision == preConfigRev {
                beginInFlightRequest()
                return AdmittedRequest(loaded: currentLoaded, config: config, loadGeneration: loadGeneration)
            }
        }
    }

    public func loadExplicit() async throws -> LoadedState {
        try await ensureLoaded()
    }

    /// Lets go of the resident model and reports the memory that came
    /// back — measured, never assumed.
    public func unloadExplicit(waitForInFlight: Bool = true) async throws -> Double? {
        guard waitForInFlight else {
            return try await unloadModel(waitForInFlight: false, forReconfigure: false)
        }
        return try await withExclusiveTransition {
            try await self.unloadModel(waitForInFlight: true, forReconfigure: false)
        }
    }

    func unloadModel(waitForInFlight: Bool = true, forReconfigure: Bool = false) async throws -> Double? {
        if !forReconfigure {
            userUnloadSequence &+= 1
        }
        tokenPrefixCache.clear()
        memoryDeltaCache.invalidate()
        switch state {
        case .unloaded:
            // Nothing was resident, so there is nothing to reclaim and
            // no number to report.
            return nil
        case .loading(let task):
            // The weights are on their way. Wait for them so they can be
            // put away rather than arriving into a server that let go.
            _ = try? await task.value
            // The unload keeps the last word over a load that may still
            // be publishing its result.
            loadGeneration &+= 1
            switch state {
            case .loaded:
                // The loader published it before the unload arrived.
                return try await unloadModel(waitForInFlight: waitForInFlight, forReconfigure: forReconfigure)
            case .failed:
                // The load failed; the failure is still the last truth.
                return nil
            default:
                transition(to: .unloaded)
                return nil
            }
        case .unloading:
            throw QWEngineError.unloading
        case .failed:
            // Nothing was loaded, so there is nothing to reclaim. The
            // failure stays the last truth until a load is attempted.
            return nil
        case .loaded:
            if waitForInFlight {
                await waitForInFlightRequests()
            }
        }
        // The switch is over, so the model is not held by it any more:
        // the reading below sees what the release really gave back.
        return await releaseResidentModel()
    }

}
