// QWBatchScheduler.swift — Unified Continuous Batching Scheduler for QWEngine.
// Coordinates single-stream passthrough and multi-session continuous batching
// across QWEngineDriver modules, coalescing concurrent bursts cleanly.

import Foundation
import QWCore
import QWDecision
import QWHead

/// Tracks requests rendering chat templates or tokenizing so concurrent bursts coalesce cleanly.
public enum QWBatchAdmissionTracker {
    public static func beginPreparing() {
        QWBatchAdmissionCounter.beginPreparing()
    }

    public static func endPreparing() {
        QWBatchAdmissionCounter.endPreparing()
    }

    public static var preparingCount: Int {
        QWBatchAdmissionCounter.preparingCount
    }
}

/// Public entry point callers invoke around template rendering/tokenization to assist batch coalescing.
public enum QWConcurrentBatchTracker {
    public static func beginPreparing() { QWBatchAdmissionTracker.beginPreparing() }
    public static func endPreparing() { QWBatchAdmissionTracker.endPreparing() }
}

/// Unified request scheduler over any `QWInferenceEngine`.
public final class QWBatchScheduler: @unchecked Sendable {
    public static let maxSlots = 16

    private let underlying: any QWInferenceEngine
    let driver: QWEngineDriver?
    private let injectedTrunk: (any QWTrunkExecutor)?
    private let lock = NSLock()
    private let exclusiveCond = NSCondition()
    private let driverWorker = QWDriverWorkerThread()
    private var activeCount: Int = 0
    private var pendingQueue: [QWBatchScheduledJob] = []
    private var isDriving: Bool = false
    private var isExclusiveRunning: Bool = false
    private var recordedMaxWaveSize: Int = 0

    private var activeTrunk: (any QWTrunkExecutor)? {
        injectedTrunk ?? driver?.trunkExecutor
    }

    private var activeMaxBatchSlots: Int {
        activeTrunk?.maxBatchSlots ?? 1
    }

    public init(engine: any QWInferenceEngine) {
        self.underlying = engine
        self.driver = Self.resolveDriver(engine)
        self.injectedTrunk = nil
    }

    init(engine: any QWInferenceEngine, trunkExecutor: any QWTrunkExecutor) {
        self.underlying = engine
        self.driver = nil
        self.injectedTrunk = trunkExecutor
    }

    deinit {
        driverWorker.shutdown()
    }

    static func resolveDriver(_ engine: any QWInferenceEngine) -> QWEngineDriver? {
        var current: any QWInferenceEngine = engine
        while let unified = current as? QWEngine {
            current = unified.underlying
        }
        if let direct = current as? QWEngineDriver {
            return direct
        }
        if let dense = current as? DenseModule {
            return dense.engine
        }
        return nil
    }

    private func acquireGenerationGate() {
        exclusiveCond.lock()
        while isExclusiveRunning {
            exclusiveCond.wait()
        }
        lock.withLock { activeCount += 1 }
        exclusiveCond.unlock()
    }

    private func releaseGenerationGate() {
        exclusiveCond.lock()
        let shouldBroadcast = lock.withLock { () -> Bool in
            activeCount = max(0, activeCount - 1)
            return activeCount == 0 && !isDriving
        }
        if shouldBroadcast {
            exclusiveCond.broadcast()
        }
        exclusiveCond.unlock()
    }

    private func runExclusive<T>(_ body: () throws -> T) rethrows -> T {
        exclusiveCond.lock()
        while isExclusiveRunning || lock.withLock({ activeCount > 0 || isDriving }) {
            exclusiveCond.wait()
        }
        isExclusiveRunning = true
        exclusiveCond.unlock()
        defer {
            exclusiveCond.lock()
            isExclusiveRunning = false
            exclusiveCond.broadcast()
            exclusiveCond.unlock()
        }
        return try body()
    }

    public func buildStandalonePrefixCache(
        tokens: [Int],
        outputURL: URL,
        pinAsSharedBase: Bool,
        onPrefillProgress: (@Sendable (Int, Int) -> Void)?
    ) throws -> Int {
        try runExclusive {
            try underlying.buildStandalonePrefixCache(
                tokens: tokens,
                outputURL: outputURL,
                pinAsSharedBase: pinAsSharedBase,
                onPrefillProgress: onPrefillProgress
            )
        }
    }

    public func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) {
        try runExclusive {
            try underlying.mountStandalonePrefixCache(from: fileURL)
        }
    }

    public func unmountSharedBasePrefixes() {
        runExclusive {
            underlying.unmountSharedBasePrefixes()
        }
    }

    public func computePromptLogprobs(
        promptTokens: [Int],
        sampling: QWSamplerConfig,
        topLogprobsCount: Int,
        shouldStop: (@Sendable () -> Bool)? = nil
    ) throws -> [QWTokenLogprob]? {
        guard let trunk = activeTrunk else { return nil }
        return try runExclusive {
            try trunk.computeEchoPromptLogprobs(
                promptTokens: promptTokens,
                sampling: sampling,
                topLogprobsCount: topLogprobsCount,
                shouldStop: shouldStop
            )
        }
    }

    /// Number of currently executing generation requests on this scheduler.
    public var inFlightRequests: Int {
        lock.withLock { activeCount }
    }

    /// Step-boundary GPU arbitrator forwarded to the active trunk (`ADR-002`).
    public var gpuTurnArbitrator: QWGPUTurnArbitrator? {
        get { activeTrunk?.gpuTurnArbitrator }
        set {
            driver?.gpuTurnArbitrator = newValue
            injectedTrunk?.gpuTurnArbitrator = newValue
        }
    }

    /// Evaluates single-pass decision logits at `positions` inside `tokens`.
    public func evaluateDecisionLogits(
        tokens: [Int],
        positions: [Int],
        candidateTokenIds: [Int],
        visualSpans: [QWVisualSpan] = []
    ) throws -> [QWDecisionSlotOutput] {
        guard let trunk = activeTrunk else {
            throw QWDecisionError.backendFailed("decision logit evaluation requires a loaded QWEngineDriver")
        }
        return try runExclusive {
            try trunk.evaluateDecisionLogits(
                tokens: tokens,
                positions: positions,
                candidateTokenIds: candidateTokenIds,
                visualSpans: visualSpans
            )
        }
    }

    var peakCoalescedWaveSize: Int {
        lock.withLock { recordedMaxWaveSize }
    }

    /// Runs `request` through single-stream fast path or batched continuous scheduling.
    public func generate(
        request: QWGenerationRequest,
        onToken: ((Int) -> Bool)? = nil
    ) throws -> QWGenerationResult {
        acquireGenerationGate()
        defer { releaseGenerationGate() }

        guard driver != nil || activeTrunk != nil else {
            return try underlying.generate(request: request, onToken: onToken)
        }
        if let trunk = activeTrunk {
            if request.usePrefixCache && !trunk.hasPrefixCacheTier {
                throw QWEngineDriverError.state("the request wants a prefix cache this engine does not have")
            }
            if trunk.maxBatchSlots <= 1 {
                return try trunk.runGeneration(request: request, onToken: onToken, onStopToken: nil)
            }
        }

        let job = QWBatchScheduledJob(request: request, onToken: onToken)
        let becameLeader = lock.withLock { () -> Bool in
            pendingQueue.append(job)
            guard !isDriving else { return false }
            isDriving = true
            return true
        }
        guard becameLeader else {
            return try job.wait()
        }

        coalescePreparingBurst(maxSlots: min(Self.maxSlots, activeMaxBatchSlots))
        enum FastDecision {
            case solo
            case dispatchBatch
        }
        let canInterleaveSolo = QWInterleavedPrefillGate.isEnabled
            && (activeTrunk?.supportsInterleavedSolo == true)
            && request.visualSpans.isEmpty
        let decision = lock.withLock { () -> FastDecision in
            if !canInterleaveSolo && pendingQueue.count == 1 && pendingQueue[0] === job {
                pendingQueue.removeFirst()
                recordedMaxWaveSize = max(recordedMaxWaveSize, 1)
                return .solo
            }
            return .dispatchBatch
        }
        if decision == .solo {
            defer {
                exclusiveCond.lock()
                let needsDrive = lock.withLock { () -> Bool in
                    if pendingQueue.isEmpty {
                        isDriving = false
                        return false
                    }
                    return true
                }
                if needsDrive {
                    exclusiveCond.unlock()
                    driverWorker.async { [weak self] in self?.driveLoop() }
                } else {
                    if lock.withLock({ activeCount == 0 }) {
                        exclusiveCond.broadcast()
                    }
                    exclusiveCond.unlock()
                }
            }
            return try runSolo(request, onToken: onToken)
        }

        driverWorker.async { [weak self] in self?.driveLoop() }
        return try job.wait()
    }

    private func coalescePreparingBurst(maxSlots: Int) {
        guard QWBatchAdmissionTracker.preparingCount > 0 else { return }
        var waitedUs = 0
        while waitedUs < 10_000 {
            let queued = lock.withLock { pendingQueue.count }
            if queued >= maxSlots { break }
            if QWBatchAdmissionTracker.preparingCount == 0 && waitedUs >= 1_000 { break }
            usleep(250)
            waitedUs += 250
        }
    }

    private func runSolo(_ request: QWGenerationRequest, onToken: ((Int) -> Bool)?) throws -> QWGenerationResult {
        if let trunk = activeTrunk {
            return try trunk.runGeneration(request: request, onToken: onToken, onStopToken: nil)
        }
        return try underlying.generate(request: request, onToken: onToken)
    }

    private func pollPendingJobs(freeSlots: Int, maxWave: Int) -> [QWBatchScheduledJob] {
        guard freeSlots > 0 else { return [] }
        return lock.withLock {
            let cap = min(freeSlots, maxWave)
            let n = min(cap, pendingQueue.count)
            guard n > 0 else { return [] }
            let polled = Array(pendingQueue.prefix(n))
            pendingQueue.removeFirst(n)
            recordedMaxWaveSize = max(recordedMaxWaveSize, 1 + n)
            return polled
        }
    }

    private func requeueDelayedJobs(_ delayed: [QWBatchScheduledJob]) {
        guard !delayed.isEmpty else { return }
        lock.withLock {
            pendingQueue.insert(contentsOf: delayed, at: 0)
        }
    }

    private func driveLoop() {
        while true {
            let maxWave = min(Self.maxSlots, activeMaxBatchSlots)
            if QWBatchAdmissionTracker.preparingCount > 0 {
                coalescePreparingBurst(maxSlots: maxWave)
            }
            enum Wave {
                case idle
                case solo(QWBatchScheduledJob)
                case batch([QWBatchScheduledJob])
            }
            exclusiveCond.lock()
            let wave = lock.withLock { () -> Wave in
                if pendingQueue.isEmpty {
                    isDriving = false
                    return .idle
                }
                if pendingQueue.count == 1 {
                    let only = pendingQueue[0]
                    let routeSoloViaBatch = QWInterleavedPrefillGate.isEnabled
                        && (activeTrunk?.supportsInterleavedSolo == true)
                        && only.request.visualSpans.isEmpty
                    if !routeSoloViaBatch {
                        recordedMaxWaveSize = max(recordedMaxWaveSize, 1)
                        return .solo(pendingQueue.removeFirst())
                    }
                }
                let take = min(maxWave, pendingQueue.count)
                let slice = Array(pendingQueue.prefix(take))
                pendingQueue.removeFirst(take)
                recordedMaxWaveSize = max(recordedMaxWaveSize, slice.count)
                return .batch(slice)
            }
            if case .idle = wave {
                if lock.withLock({ activeCount == 0 }) {
                    exclusiveCond.broadcast()
                }
                exclusiveCond.unlock()
            } else {
                exclusiveCond.unlock()
            }

            switch wave {
            case .idle:
                return
            case .solo(let solo):
                do {
                    solo.complete(.success(try runSolo(solo.request, onToken: solo.onToken)))
                } catch {
                    solo.complete(.failure(error))
                }
            case .batch(let jobs):
                if let trunk = activeTrunk {
                    trunk.runScheduledBatch(
                        initialJobs: jobs,
                        pollPending: { self.pollPendingJobs(freeSlots: $0, maxWave: maxWave) },
                        requeueDelayed: { self.requeueDelayedJobs($0) }
                    )
                } else {
                    for job in jobs {
                        job.complete(.failure(QWEngineDriverError.state("batch scheduler driver missing")))
                    }
                }
            }
        }
    }
}

extension QWEngineDriver {
    func computeEchoPromptLogprobs(
        prompt: [Int],
        sampling: QWSamplerConfig,
        topLogprobsCount: Int,
        shouldStop: (@Sendable () -> Bool)?
    ) throws -> [QWTokenLogprob] {
        try trunkExecutor.computeEchoPromptLogprobs(
            promptTokens: prompt,
            sampling: sampling,
            topLogprobsCount: topLogprobsCount,
            shouldStop: shouldStop
        )
    }

    func computeStandardEchoPromptLogprobs(
        prompt: [Int],
        sampling: QWSamplerConfig,
        topLogprobsCount: Int,
        shouldStop: (@Sendable () -> Bool)?
    ) throws -> [QWTokenLogprob] {
        guard prompt.count >= 2 else { return [] }
        for t in prompt where t < 0 || t >= geometry.vocab {
            throw QWEngineDriverError.badToken(t)
        }
        let session = try makeSession(promptTokens: prompt.count)
        defer { finishSession(session) }
        return try requestGate.withRequest(flight: flight) { permit in
            guard prompt.count < ropeRows else {
                throw QWEngineDriverError.posBeyondTable(prompt.count)
            }
            let topK = max(0, min(20, topLogprobsCount))
            let lpSampler = QWSampler(config: sampling)
            var out: [QWTokenLogprob] = []
            out.reserveCapacity(prompt.count - 1)
            func recordLogprob(chosenToken: Int) {
                let rawLogits = sc.logits.region.pointer.advanced(by: sc.logits.offset)
                    .assumingMemoryBound(to: Float.self)
                let rowBuf = UnsafeBufferPointer(start: rawLogits, count: geometry.vocab)
                out.append(lpSampler.computeTokenLogprob(logits: rowBuf, chosenToken: chosenToken, topK: topK))
            }
            _ = try prefillChunk(
                session, tokens: prompt, start: 0, count: 1,
                runsHead: true, mode: gdnPrefillMode, permit: permit
            )
            recordLogprob(chosenToken: prompt[1])
            for i in 1..<(prompt.count - 1) {
                if shouldStop?() == true { break }
                _ = try decodeStep(session, token: prompt[i], permit: permit)
                recordLogprob(chosenToken: prompt[i + 1])
            }
            return out
        }
    }
}

private final class QWDriverWorkerThread: @unchecked Sendable {
    private let condition = NSCondition()
    private var queue: [@Sendable () -> Void] = []
    private var started = false
    private var stopped = false

    func async(_ block: @escaping @Sendable () -> Void) {
        condition.lock()
        if stopped {
            condition.unlock()
            return
        }
        queue.append(block)
        let needStart = !started
        if needStart { started = true }
        condition.signal()
        condition.unlock()
        if needStart {
            let thread = Thread { self.runLoop() }
            thread.name = "qwegn.qwengine.batch.driver"
            thread.qualityOfService = .userInitiated
            thread.stackSize = 8 * 1024 * 1024
            thread.start()
        }
    }

    func shutdown() {
        condition.lock()
        stopped = true
        condition.broadcast()
        condition.unlock()
    }

    private func runLoop() {
        while true {
            condition.lock()
            while queue.isEmpty && !stopped {
                condition.wait()
            }
            if queue.isEmpty && stopped {
                condition.unlock()
                return
            }
            let work = queue.removeFirst()
            condition.unlock()
            autoreleasepool {
                work()
            }
        }
    }
}
