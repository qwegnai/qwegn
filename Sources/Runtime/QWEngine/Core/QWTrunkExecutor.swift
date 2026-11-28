// QWTrunkExecutor.swift — Unified internal protocol for specialized trunk executors under QWEngineDriver.

import Foundation
import QWCore
import QWDecision
import QWHead
import QWMTP

/// Internal execution contract for specialized trunk runtimes hosted behind `QWEngineDriver`.
protocol QWTrunkExecutor: AnyObject, Sendable {
    var hasPrefixCacheTier: Bool { get }
    var arenaBytes: Int { get }
    var weightReservedBytes: Int { get }
    var weightUsedBytes: Int { get }
    var activeCapacity: Int { get }
    var maxBatchSlots: Int { get }
    var supportsInterleavedSolo: Bool { get }
    var steadyStateAllocations: Int { get }
    var dflashDrafter: QWDFlashDrafter? { get }
    var prefixCacheState: QWPrefixCacheState { get }
    var lastResult: QWGenerationResult? { get }
    var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)? { get set }
    var gpuTurnArbitrator: QWGPUTurnArbitrator? { get set }

    func hasCachedPrefix(tokens: [Int], cacheSalt: String?) -> Bool
    func buildCapabilities() -> Set<QWKnobCapability>
    func buildReport() -> QWEngineReport
    func runGeneration(
        request: QWGenerationRequest,
        onToken: ((Int) -> Bool)?,
        onStopToken: ((Int) -> Void)?
    ) throws -> QWGenerationResult
    func configurePrefixCache(_ config: QWCacheConfig) throws
    func buildStandalonePrefixCache(
        tokens: [Int],
        outputURL: URL,
        pinAsSharedBase: Bool,
        onPrefillProgress: (@Sendable (Int, Int) -> Void)?
    ) throws -> Int
    func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int)
    func unmountSharedBasePrefixes()
    func computeEchoPromptLogprobs(
        promptTokens: [Int],
        sampling: QWSamplerConfig,
        topLogprobsCount: Int,
        shouldStop: (@Sendable () -> Bool)?
    ) throws -> [QWTokenLogprob]
    func evaluateDecisionLogits(
        tokens: [Int],
        positions: [Int],
        candidateTokenIds: [Int],
        visualSpans: [QWVisualSpan]
    ) throws -> [QWDecisionSlotOutput]
    func runScheduledBatch(
        initialJobs: [QWBatchScheduledJob],
        pollPending: (_ freeSlots: Int) -> [QWBatchScheduledJob],
        requeueDelayed: ([QWBatchScheduledJob]) -> Void
    )
    func setRelaxedMinProb(_ value: Double)
}

extension QWTrunkExecutor {
    var maxBatchSlots: Int { 1 }
    var supportsInterleavedSolo: Bool { false }
    func setRelaxedMinProb(_ value: Double) {}

    func evaluateDecisionLogits(
        tokens _: [Int],
        positions _: [Int],
        candidateTokenIds _: [Int],
        visualSpans _: [QWVisualSpan]
    ) throws -> [QWDecisionSlotOutput] {
        throw QWDecisionError.backendFailed(
            "decision logit extraction is not supported by this trunk executor"
        )
    }

    func runScheduledBatch(
        initialJobs: [QWBatchScheduledJob],
        pollPending _: (_ freeSlots: Int) -> [QWBatchScheduledJob],
        requeueDelayed _: ([QWBatchScheduledJob]) -> Void
    ) {
        for job in initialJobs {
            do {
                let res = try runGeneration(request: job.request, onToken: job.onToken, onStopToken: nil)
                job.complete(.success(res))
            } catch {
                job.complete(.failure(error))
            }
        }
    }
}

extension QWHyperStreamRuntime: QWTrunkExecutor {
    var dflashDrafter: QWDFlashDrafter? { dflash }
    var maxBatchSlots: Int { batchMaxSlots }
}

extension QWInkling2Runtime: QWTrunkExecutor {
    var dflashDrafter: QWDFlashDrafter? { dflash }
    func setRelaxedMinProb(_ value: Double) {
        configuredRelaxedMinProb = value
    }
}
