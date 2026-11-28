// QWStandardTrunkExecutor.swift — QWTrunkExecutor adapter unifying the standard QWEngineDriver control plane.
import Foundation
import QWCore
import QWDecision
import QWHead
import QWMTP

/// Adapts standard dense/hybrid `QWEngineDriver` state onto `QWTrunkExecutor` so callers dispatch uniformly.
final class QWStandardTrunkExecutor: QWTrunkExecutor, @unchecked Sendable {
    unowned let driver: QWEngineDriver
    private let stateLock = NSLock()
    private var _onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)?
    private var _gpuTurnArbitrator: QWGPUTurnArbitrator?
    private var cachedHeadScratch: QWBatchHeadScratch?

    init(driver: QWEngineDriver) {
        self.driver = driver
    }

    var hasPrefixCacheTier: Bool { driver.prefixStore != nil }
    var arenaBytes: Int { driver.arena.committedBytes }
    var weightReservedBytes: Int { driver.arena.regionSizes.weights }
    var weightUsedBytes: Int { driver.arena.weightsSub.usedBytes }
    var activeCapacity: Int { driver.kvCache.freePageCount }
    var maxBatchSlots: Int {
        driver.admission.totalSlots > 0 ? driver.admission.totalSlots : QWEngineDriver.gdnSlotCount()
    }
    var supportsInterleavedSolo: Bool {
        driver.gdnPrefillMode != .naive && maxBatchSlots >= 2
    }
    var steadyStateAllocations: Int { driver.standardSteadyStateAllocations }
    var dflashDrafter: QWDFlashDrafter? { driver.standardState?.dflashDrafter }
    var prefixCacheState: QWPrefixCacheState { driver.standardPrefixCacheState }
    var lastResult: QWGenerationResult? { driver.standardLastTurn }

    var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)? {
        get { stateLock.withLock { _onCacheEvent } }
        set { stateLock.withLock { _onCacheEvent = newValue } }
    }

    var gpuTurnArbitrator: QWGPUTurnArbitrator? {
        get { stateLock.withLock { _gpuTurnArbitrator } }
        set { stateLock.withLock { _gpuTurnArbitrator = newValue } }
    }

    func hasCachedPrefix(tokens: [Int], cacheSalt: String?) -> Bool {
        driver.hasStandardCachedPrefix(tokens: tokens, cacheSalt: cacheSalt)
    }

    func buildCapabilities() -> Set<QWKnobCapability> {
        driver.buildStandardCapabilities()
    }

    func buildReport() -> QWEngineReport {
        driver.buildStandardReport()
    }

    func runGeneration(
        request: QWGenerationRequest,
        onToken: ((Int) -> Bool)?,
        onStopToken _: ((Int) -> Void)?
    ) throws -> QWGenerationResult {
        try driver.runStandardGeneration(request: request, onToken: onToken)
    }

    func configurePrefixCache(_ config: QWCacheConfig) throws {
        try driver.configureStandardPrefixCache(config: config)
    }

    func buildStandalonePrefixCache(
        tokens: [Int],
        outputURL: URL,
        pinAsSharedBase: Bool,
        onPrefillProgress: (@Sendable (Int, Int) -> Void)?
    ) throws -> Int {
        try driver.buildStandardStandalonePrefixCache(
            tokens: tokens,
            outputURL: outputURL,
            pinAsSharedBase: pinAsSharedBase,
            onPrefillProgress: onPrefillProgress
        )
    }

    func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) {
        try driver.mountStandardStandalonePrefixCache(from: fileURL)
    }

    func unmountSharedBasePrefixes() {
        driver.unmountStandardSharedBasePrefixes()
    }

    func computeEchoPromptLogprobs(
        promptTokens: [Int],
        sampling: QWSamplerConfig,
        topLogprobsCount: Int,
        shouldStop: (@Sendable () -> Bool)?
    ) throws -> [QWTokenLogprob] {
        try driver.computeStandardEchoPromptLogprobs(
            prompt: promptTokens,
            sampling: sampling,
            topLogprobsCount: topLogprobsCount,
            shouldStop: shouldStop
        )
    }

    func evaluateDecisionLogits(
        tokens: [Int],
        positions: [Int],
        candidateTokenIds: [Int],
        visualSpans: [QWVisualSpan]
    ) throws -> [QWDecisionSlotOutput] {
        try driver.evaluateStandardDecisionLogits(
            tokens: tokens,
            positions: positions,
            candidateTokenIds: candidateTokenIds,
            visualSpans: visualSpans
        )
    }

    func runScheduledBatch(
        initialJobs: [QWBatchScheduledJob],
        pollPending: (_ freeSlots: Int) -> [QWBatchScheduledJob],
        requeueDelayed: ([QWBatchScheduledJob]) -> Void
    ) {
        let scratch = ensureHeadScratch()
        driver.runStandardScheduledBatch(
            initialJobs: initialJobs,
            headScratch: scratch,
            pollPending: pollPending,
            requeueDelayed: requeueDelayed
        )
    }

    private func ensureHeadScratch() -> QWBatchHeadScratch? {
        stateLock.withLock {
            if let existing = cachedHeadScratch { return existing }
            guard let built = try? QWBatchHeadScratch(hal: driver.hal, vocab: driver.geometry.vocab) else {
                return nil
            }
            cachedHeadScratch = built
            return built
        }
    }
}
