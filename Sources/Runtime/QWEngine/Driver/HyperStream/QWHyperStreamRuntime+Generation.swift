// QWHyperStreamRuntime+Generation.swift — Generation, speculation policy, and telemetry on QWHyperStreamRuntime.

import Foundation
import QWCore
import QWGEMM
import QWHAL
import QWHead
import QWMTP

/// HyperStream speculative-decode policy helpers shared across resolution and generation.
enum FNGPUSessionState {
    /// Whether the session runs the speculative path (`effectiveDrafts > 0` and supported).
    static func useSpec(sampling: QWSamplerConfig, speculative: QWSpeculativeParameters,
                        runtime: QWHyperStreamRuntime) -> Bool {
        guard !(sampling.hasExplicitSeed && !sampling.isGreedy) else { return false }
        return speculative.effectiveDrafts > 0 && runtime.supportsSpeculativeDecode
    }

    /// Which drafter the session runs: hybrid, ngram, or neural MTP.
    static func draftMode(
        for speculative: QWSpeculativeParameters,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> QWDraftStrategy {
        if let backend = speculative.mtpBackend, !backend.isEmpty {
            _ = try QWMTPBackend.resolve(name: backend.lowercased())
        }
        if speculative.draftStrategyIsResolved != true,
           let envDraft = environment["FN_DRAFT"], !envDraft.isEmpty {
            return try QWDraftStrategy.resolve(envDraft)
        }
        return try QWDraftStrategy.resolve(speculative.mtpDraftStrategy)
    }
}

extension QWHyperStreamRuntime {
    public var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)? {
        get { stateLock.withLock { _onCacheEvent } }
        set { stateLock.withLock { _onCacheEvent = newValue } }
    }

    public var gpuTurnArbitrator: QWGPUTurnArbitrator? {
        get { stateLock.withLock { _gpuTurnArbitrator } }
        set { stateLock.withLock { _gpuTurnArbitrator = newValue } }
    }

    public var prefixCacheHit: Bool { stateLock.withLock { _prefixCacheHit } }
    public var prefixCacheMatchedTokens: Int { stateLock.withLock { _prefixCacheMatchedTokens } }
    public var prefixCacheTier: FNPrefixCacheTier? { stateLock.withLock { _prefixCacheTier } }

    public internal(set) var naxLogMessage: String {
        get { stateLock.withLock { _naxLogMessage } }
        set { stateLock.withLock { _naxLogMessage = newValue } }
    }

    public internal(set) var lastResult: QWGenerationResult? {
        get { stateLock.withLock { _lastResult } }
        set { stateLock.withLock { _lastResult = newValue } }
    }

    func wireRuntimeCallbacks() {
        batchScheduler.onCacheEvaluated = { [weak self] sessionData in
            self?.recordCacheEvaluation(sessionData)
        }
        batchScheduler.onSlot0CacheEvaluated = nil
        batchScheduler.onStepTiming = { gpuMs, wallMs in
            QWTrunkRuntimeSupport.recordGPUBusy(gpuSeconds: gpuMs, wallSeconds: wallMs)
        }
        onCacheRetained = { [weak self] tokens, chargedBytes, storeTotalBytes, warmLimitBytes in
            self?.onCacheEvent?(.retained(tokens: tokens, chargedBytes: chargedBytes,
                                          storeTotalBytes: storeTotalBytes, warmLimitBytes: warmLimitBytes))
        }
        onCacheSkippedOversize = { [weak self] tokens, charge, warmLimitBytes in
            self?.onCacheEvent?(.skippedOversize(tokens: tokens, charge: charge, warmLimitBytes: warmLimitBytes))
        }
        onCacheEvicted = { [weak self] tokens, freedBytes in
            self?.onCacheEvent?(.evicted(tokens: tokens, freedBytes: freedBytes))
        }
        wireWarmStoreCallbacks()
    }

    func wireWarmStoreCallbacks() {
        prefixCacheWarmStore?.onEvict = { [weak self] tokCount, freedBytes in
            self?.onCacheEvicted?(tokCount, freedBytes)
        }
    }

    func recordCacheEvaluation(_ sessionData: FNGPUSessionData) {
        let (hit, matched, tier, sink) = stateLock.withLock {
            _prefixCacheHit = sessionData.prefixCacheHit
            _prefixCacheMatchedTokens = sessionData.prefixCacheMatchedTokens
            _prefixCacheTier = sessionData.prefixCacheTier
            return (_prefixCacheHit, _prefixCacheMatchedTokens, _prefixCacheTier, _onCacheEvent)
        }
        guard let sink else { return }
        if hit {
            sink(.hit(tier: tier?.rawValue ?? "",
                      matchedTokens: matched,
                      totalTokens: sessionData.history.count))
        } else if hasPrefixCacheTier {
            sink(.miss(totalTokens: sessionData.history.count))
        }
    }

    func recordGenerationResult(_ result: QWGenerationResult, specProposals: Int, specAccepted: Int) {
        stateLock.withLock {
            _lastResult = result
            _lastSpecProposals = specProposals
            _lastSpecAccepted = specAccepted
        }
    }

    func buildCapabilities() -> Set<QWKnobCapability> {
        var caps = QWModelCapabilities.resolve(descriptor: descriptor, hasVisionWeights: hasVisionWeights)
        if !supportsSpeculativeDecode { caps.remove(.speculative) }
        return caps
    }

    var prefixCacheState: QWPrefixCacheState {
        guard hasPrefixCacheTier else { return QWPrefixCacheState(isConfigured: false) }
        let disk = prefixCacheDiskState
        let warm = prefixCacheWarmState
        return QWPrefixCacheState(
            isConfigured: true, warmBytes: warm?.bytes, ssdBytes: disk?.bytes,
            warmEntries: warm?.entries, ssdEntries: disk?.entries, warmLimitBytes: prefixCacheWarmLimitBytes)
    }

    public func hasCachedPrefix(tokens: [Int], cacheSalt: String?) -> Bool {
        guard hasPrefixCacheTier, !tokens.isEmpty,
              tokens.allSatisfy({ (0..<vocab).contains($0) }) else {
            return false
        }
        let rawHist = tokens.map(Int64.init)
        let cacheHistory = QWVisualSpan.saltedCacheHistory(history: rawHist, visualSpans: [], cacheSalt: cacheSalt)
        let mtpFlags: [Bool] = (mtp != nil) ? [true, false] : [false]
        let hasColdSSD = prefixCacheSSDByteLimit > 0 && prefixCacheDirectory != nil
        if hasColdSSD {
            Self.ssdWriteQueue.sync {}
        }
        for hasMTP in mtpFlags {
            let hash = Self.fnPrefixHash(
                tokens: cacheHistory[...], kvMode: kvMode, hasMTP: hasMTP, dflashMarker: dflashPrefixMarker)
            let name = String(format: "fn_entry_%016llx_%d.qwcache", hash, cacheHistory.count)
            if prefixCacheWarmStore?.contains(name: name) == true {
                return true
            }
            if cacheHistory != rawHist, let warm = prefixCacheWarmStore {
                let rawHash = Self.fnPrefixHash(
                    tokens: rawHist[...], kvMode: kvMode, hasMTP: hasMTP, dflashMarker: dflashPrefixMarker)
                let rawName = String(format: "fn_entry_%016llx_%d.qwcache", rawHash, rawHist.count)
                if warm.isSharedBaseEntry(rawName) && warm.contains(name: rawName) {
                    return true
                }
            }
            if hasColdSSD, let cacheDir = prefixCacheDirectory,
               FileManager.default.fileExists(atPath: cacheDir.appendingPathComponent(name).path) {
                return true
            }
        }
        return false
    }

    func buildReport() -> QWEngineReport {
        QWEngineReport(
            capacityLog: capacityLogMessage,
            arenaBytes: arenaBytes,
            weightUsedBytes: weightUsedBytes,
            weightReservedBytes: weightReservedBytes,
            steadyAllocations: steadyStateAllocations,
            pipelineSeconds: QWHalPSOStats.seconds,
            detailLines: naxLogMessage + prefillStackNote + dflashReportLine,
            contextCapacity: capacity,
            weightFormat: headWeightFormatLabel
        )
    }

    var batchMaxSlots: Int { FNGPUBatchScheduler.maxSlots }

    func configurePrefixCache(_ config: QWCacheConfig) throws {
        try batchScheduler.runExclusive {
            let warmBytes = max(0, config.warmByteLimit)
            let ssdBytes = max(0, config.ssdByteLimit)
            let entryLimit = max(1, config.entryLimit)
            let warmAdmission = QWResidencyAdmission.clampWarmCache(
                requestedWarmBytes: warmBytes,
                requestedSessions: entryLimit,
                wiredEngineBytes: arenaBytes,
                measuredAfterWire: true,
                snapshot: QWLiveDeviceMemorySnapshotSource(maxWorkingSetBytes: hal.caps.maxWorkingSetBytes).capture()
            )
            self.prefixCacheSSDByteLimit = ssdBytes
            if warmAdmission.admittedWarmBytes > 0 {
                self.prefixCacheWarmStore = FNPrefixWarmStore(
                    byteLimit: warmAdmission.admittedWarmBytes,
                    maxSlots: warmAdmission.maxConcurrentSessions
                )
            } else {
                self.prefixCacheWarmStore = nil
            }
            if ssdBytes > 0 {
                let dir = config.cacheDirectory
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                if config.wipeOnStart, let items = try? FileManager.default.contentsOfDirectory(
                    at: dir, includingPropertiesForKeys: nil) {
                    for url in items where url.lastPathComponent.hasPrefix("fn_entry_") && url.pathExtension == "qwcache" {
                        try? FileManager.default.removeItem(at: url)
                    }
                }
                self.prefixCacheDirectory = dir
            } else {
                self.prefixCacheDirectory = nil
            }
            wireWarmStoreCallbacks()
        }
    }

    func makeBatchJob(
        request: QWGenerationRequest,
        onToken: ((Int) -> Bool)?,
        onStopToken: ((Int) -> Void)?,
        scheduledJob: QWBatchScheduledJob? = nil
    ) throws -> FNBatchJob {
        if request.usePrefixCache && !hasPrefixCacheTier {
            throw QWEngineDriverError.state("the request wants a prefix cache this engine does not have")
        }
        for token in request.prompt where token < 0 || token >= plan.vocab {
            throw QWModuleError.invalidToken(token)
        }
        let maxCap = capacity
        guard request.prompt.count < maxCap else {
            throw FNGPUError.contextDoesNotFit(
                requested: request.prompt.count + 1,
                maxFitting: max(0, maxCap - 1)
            )
        }
        let effectiveMaxNew = min(max(0, request.maxNew), max(0, maxCap - request.prompt.count))
        naxLogMessage = QWGemm.configureDenseNax(
            temperature: Double(request.sampling.temperature),
            topP: Double(request.sampling.topP),
            topK: request.sampling.topK
        ).logMessage
        let useSpec = FNGPUSessionState.useSpec(
            sampling: request.sampling, speculative: request.speculative, runtime: self)
        let draftMode = try FNGPUSessionState.draftMode(for: request.speculative)
        let specCap = dflash != nil
            ? min(FNGPUDFlash.maxDrafts, dflashMaxDrafts)
            : FNGPUSessionData.defaultMTPDrafts
        return FNBatchJob(
            prompt: request.prompt,
            maxNew: effectiveMaxNew,
            stopIDs: request.stopIDs,
            sampling: request.sampling,
            useSpec: useSpec && !request.collectLogprobs,
            draftMode: draftMode,
            maxDrafts: max(1, min(specCap, request.speculative.effectiveDrafts)),
            relaxedMargin: request.sampling.hasExplicitSeed ? 0 : request.speculative.effectiveRelaxedMargin,
            relaxedMinProb: request.speculative.effectiveRelaxedMinProb,
            quadraticSeeding: request.speculative.mtpQuadraticSeeding,
            collectLogprobs: request.collectLogprobs,
            topLogprobsCount: request.topLogprobsCount,
            onPrefillProgress: request.onPrefillProgress,
            onToken: onToken,
            onStopToken: onStopToken,
            onTokenLogprob: request.onTokenLogprob,
            constraint: request.constraint,
            retainPrefixLength: request.retainPrefixLength,
            visualSpans: request.visualSpans,
            usePrefixCache: request.usePrefixCache,
            cacheSalt: request.cacheSalt,
            scheduledJob: scheduledJob
        )
    }

    func mapBatchError(_ error: any Error) -> any Error {
        if case FNGPUError.invalidToken(let tok) = error {
            return QWModuleError.invalidToken(tok)
        }
        if let engineErr = error as? QWEngineError {
            return QWModuleError.from(engineErr)
        }
        return error
    }

    func finalizeGenerationResult(from out: FNBatchJobOutput, promptCount: Int) -> QWGenerationResult {
        let cache = hasPrefixCacheTier
            ? QWCacheAnswer(hit: out.prefixCacheHit, matchedTokens: out.prefixCacheMatchedTokens,
                            totalTokens: promptCount, tier: out.prefixCacheTier?.rawValue)
            : nil
        let specLabel = dflash.map { $0.config.isDSpine ? "dspine" : "dflash2" } ?? "mtp"
        let specSummary = QWTrunkRuntimeSupport.formatSpeculativeSummary(
            label: specLabel,
            accepted: out.specAccepted,
            proposals: out.specProposals,
            stepTokenCounts: out.stepTokenCounts
        )
        let finalResult = QWGenerationResult(
            tokens: out.tokens, stepMs: out.stepMs, stepTokenCounts: out.stepTokenCounts,
            prefillSeconds: out.prefillSeconds, firstTokenSeconds: out.firstTokenSeconds,
            decodeSeconds: out.decodeSeconds, prefillGPUSeconds: out.prefillGPUSeconds,
            stopTokenID: out.stopTokenID, cache: cache, speculativeSummary: specSummary,
            tokenLogprobs: out.tokenLogprobs)
        recordGenerationResult(finalResult, specProposals: out.specProposals, specAccepted: out.specAccepted)
        return finalResult
    }

    func runScheduledBatch(
        initialJobs: [QWBatchScheduledJob],
        pollPending: (_ freeSlots: Int) -> [QWBatchScheduledJob],
        requeueDelayed: ([QWBatchScheduledJob]) -> Void
    ) {
        batchScheduler.runScheduledBatch(
            initialJobs: initialJobs,
            pollPending: pollPending,
            requeueDelayed: requeueDelayed
        )
    }

    /// Generates tokens for a request through `QWHyperStreamRuntime`'s continuous batch scheduler.
    func runGeneration(
        request: QWGenerationRequest,
        onToken: ((Int) -> Bool)?,
        onStopToken: ((Int) -> Void)?
    ) throws -> QWGenerationResult {
        guard !request.prompt.isEmpty else {
            return QWGenerationResult(tokens: [], stepMs: [], stepTokenCounts: [])
        }
        let job = try makeBatchJob(request: request, onToken: onToken, onStopToken: onStopToken)
        let out: FNBatchJobOutput
        do {
            out = try batchScheduler.submit(job)
        } catch {
            throw mapBatchError(error)
        }
        return finalizeGenerationResult(from: out, promptCount: request.prompt.count)
    }
}
