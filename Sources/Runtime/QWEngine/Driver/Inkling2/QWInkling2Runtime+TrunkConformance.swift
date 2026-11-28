// QWInkling2Runtime+TrunkConformance.swift — QWTrunkExecutor conformance, prefix setup, and token emission helpers.
import Foundation
import Metal
import QWAlloc
import QWAttn
import QWCore
import QWHAL
import QWHead
import QWKV
import QWMTP
import QWQuant

extension QWInkling2Runtime {
    enum EmittedTokenOutcome {
        case continued
        case stopped
        case cancelled
    }

    struct PrefixSetupResult {
        let canUseCache: Bool
        let cacheHit: Bool
        let matchedTokens: Int
        let pos: Int
        let dsparkActive: Bool
        let dflashActive: Bool
        let restoredExactFinalNorm: Data?
        let retainTarget: Int?
        let totalPromptTokens: Int

        var cacheAnswer: QWCacheAnswer? {
            guard canUseCache else { return nil }
            return QWCacheAnswer(
                hit: cacheHit,
                matchedTokens: matchedTokens,
                totalTokens: totalPromptTokens,
                tier: cacheHit ? "warm" : nil
            )
        }
    }

    var hasVisionWeights: Bool { weights.hasVisionWeights }
    var hasPrefixCacheTier: Bool { prefixCache.isEnabled }
    var activeCapacity: Int { capacity }
    var maxBatchSlots: Int { waveSlots.count }
    var companionResidentBytes: Int {
        waveSlots.dropFirst().reduce(0) { $0 + $1.residentStateBytes }
    }
    var arenaBytes: Int {
        weights.totalBytes + (dspark?.totalBytes ?? 0) + (dflash?.totalBytes ?? 0)
            + (dflashTapBuf?.length ?? 0) + (lsaOperator?.region.buffer.length ?? 0)
            + session.residentStateBytes + companionResidentBytes + session.scratchBytes
    }
    var weightUsedBytes: Int { weights.totalBytes + (dspark?.totalBytes ?? 0) + (dflash?.weightBytes ?? 0) }
    var weightReservedBytes: Int { weights.totalBytes + (dspark?.totalBytes ?? 0) + (dflash?.weightBytes ?? 0) }
    var steadyStateAllocations: Int { 0 }

    var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)? {
        get { stateLock.withLock { _onCacheEvent } }
        set { stateLock.withLock { _onCacheEvent = newValue } }
    }

    var gpuTurnArbitrator: QWGPUTurnArbitrator? {
        get { stateLock.withLock { _gpuTurnArbitrator } }
        set { stateLock.withLock { _gpuTurnArbitrator = newValue } }
    }

    @inline(__always)
    func withGPUTurn<T>(_ body: () throws -> T) rethrows -> T {
        try QWTrunkRuntimeSupport.withGPUTurn(arbitrator: gpuTurnArbitrator, body)
    }

    var lastResult: QWGenerationResult? {
        get { stateLock.withLock { _lastResult } }
        set { stateLock.withLock { _lastResult = newValue } }
    }

    func wireCacheCallbacks() {
        prefixCache.onRetain = { [weak self] tokens, charged, total, limit in
            self?.onCacheEvent?(.retained(tokens: tokens, chargedBytes: charged,
                                          storeTotalBytes: total, warmLimitBytes: limit))
        }
        prefixCache.onSkipOversize = { [weak self] tokens, charge, limit in
            self?.onCacheEvent?(.skippedOversize(tokens: tokens, charge: charge, warmLimitBytes: limit))
        }
        prefixCache.onEvict = { [weak self] tokens, freed in
            self?.onCacheEvent?(.evicted(tokens: tokens, freedBytes: freed))
        }
    }

    func buildCapabilities() -> Set<QWKnobCapability> {
        QWModelCapabilities.resolve(descriptor: descriptor, weightFormat: weightFormat, hasVisionWeights: hasVisionWeights)
    }

    var prefixCacheState: QWPrefixCacheState {
        guard prefixCache.isEnabled, let st = prefixCache.state else {
            return QWPrefixCacheState(isConfigured: false)
        }
        return QWPrefixCacheState(
            isConfigured: true, warmBytes: st.bytes, ssdBytes: nil,
            warmEntries: st.entries, ssdEntries: nil, warmLimitBytes: st.limit
        )
    }

    func hasCachedPrefix(tokens: [Int], cacheSalt: String?) -> Bool {
        guard !tokens.isEmpty, prefixCache.isEnabled else { return false }
        guard let entry = prefixCache.matchLongestPrefix(
            tokens: tokens, salt: cacheSalt, limit: tokens.count, allowExact: true
        ) else { return false }
        return entry.key.tokens.count == tokens.count
    }

    func buildReport() -> QWEngineReport {
        let draftLabel = dflash != nil ? ", dflash=\(activeMtpDrafts)"
            : (dspark != nil ? ", dspark=\(activeMtpDrafts)" : "")
        let lsaLabel: String = {
            guard let lsa = lsaOperator else { return "" }
            let reuseTag = lsa.loopReusePlan.hasReuse ? ",loop_share" : ""
            return ", lsa=\(lsa.config.mode.rawValue)(k=\(lsa.config.alignedBudget),blk=\(lsa.config.alignedClusterBlock)\(reuseTag))"
        }()
        let lines = String(
            format: "runtime:                 inkling2-loop (visits=%d, kv=%@, capacity=%d%@%@)\n",
            session.executionPlan.count, rawKVMode, capacity, draftLabel, lsaLabel
        )
        return QWEngineReport(
            capacityLog: capacityLogMessage,
            arenaBytes: arenaBytes,
            weightUsedBytes: weightUsedBytes,
            weightReservedBytes: weightReservedBytes,
            steadyAllocations: steadyStateAllocations,
            pipelineSeconds: QWHalPSOStats.seconds,
            detailLines: lines,
            contextCapacity: capacity,
            weightFormat: weightFormat.name
        )
    }

    func configurePrefixCache(_ config: QWCacheConfig) throws {
        try execLock.withLock {
            guard config.warmByteLimit >= 0, config.entryLimit > 0 else {
                throw QWEngineDriverError.state("invalid prefix cache configuration")
            }
            let warmAdm = QWResidencyAdmission.clampWarmCache(
                requestedWarmBytes: config.warmByteLimit,
                requestedSessions: config.maxConcurrentSessions,
                wiredEngineBytes: arenaBytes,
                measuredAfterWire: true,
                snapshot: QWLiveDeviceMemorySnapshotSource(maxWorkingSetBytes: hal.caps.maxWorkingSetBytes).capture()
            )
            prefixCache.reconfigure(
                warmByteLimit: warmAdm.admittedWarmBytes,
                maxEntries: max(1, config.entryLimit)
            )
        }
    }

    func buildStandalonePrefixCache(
        tokens _: [Int], outputURL _: URL, pinAsSharedBase _: Bool,
        onPrefillProgress _: (@Sendable (Int, Int) -> Void)?
    ) throws -> Int {
        throw QWEngineError.state("standalone .qwcache export is not supported for Inkling-2 checkpoints")
    }

    func mountStandalonePrefixCache(from _: URL) throws -> (tokens: [Int], byteCount: Int) {
        throw QWEngineError.state("standalone .qwcache mount is not supported for Inkling-2 checkpoints")
    }

    func unmountSharedBasePrefixes() {
        execLock.withLock { prefixCache.clear() }
    }

    func elapsedSeconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) * 1e-18
    }

    static let repetitionPenaltyWindow = QWTrunkRuntimeSupport.repetitionPenaltyWindow

    static func recordGeneratedTokenForPenalty(
        _ token: Int,
        vocab: Int,
        recentGenerated: inout [Int],
        activeLoopTokens: inout [Int]
    ) {
        QWTrunkRuntimeSupport.recordGeneratedTokenForPenalty(
            token, vocab: vocab, recentGenerated: &recentGenerated, activeLoopTokens: &activeLoopTokens
        )
    }

    func sampleToken(
        sampler: QWSampler,
        logitsRow: UnsafeBufferPointer<Float>,
        penaltyHistory: [Int],
        constraint: (any QWTokenConstraint)?,
        allowedScratch: inout [Int]
    ) throws -> Int {
        try QWTrunkRuntimeSupport.sampleToken(
            sampler: sampler, logitsRow: logitsRow, penaltyHistory: penaltyHistory,
            constraint: constraint, allowedScratch: &allowedScratch
        )
    }

    func resetSharedAuxOperators() {
        dspark?.reset()
        if let lsa = lsaOperator {
            lsa.setPinnedBaseState(nil, position: 0)
            _ = lsa.restoreIndexerState(Data(), position: 0)
            lsa.stateOwner = nil
        }
    }

    func preparePrefixState(
        for request: QWGenerationRequest,
        slot: Ink2SlotState,
        allowAuxOperators: Bool
    ) -> PrefixSetupResult {
        let canUseCache = request.usePrefixCache && prefixCache.isEnabled
            && (!allowAuxOperators || request.visualSpans.isEmpty || request.cacheSalt != nil)
        var cacheHit = false
        var matchedTokens = 0
        var pos = 0
        var dsparkActive = allowAuxOperators && dspark != nil
        var dflashActive = allowAuxOperators && dflash != nil
        var restoredFromCache = false
        var restoredExactFinalNorm: Data? = nil

        if canUseCache, request.prompt.count >= 1,
           let entry = prefixCache.matchLongestPrefix(
               tokens: request.prompt, salt: request.cacheSalt,
               limit: request.prompt.count, allowExact: true
           ) {
            let outcome = QWInkling2PrefixCache.restore(
                entry: entry, into: slot, kvModeRaw: session.kvModeRaw,
                dspark: allowAuxOperators ? dspark : nil,
                dflash: allowAuxOperators ? dflash : nil,
                lsaOperator: allowAuxOperators ? lsaOperator : nil,
                stateOwner: allowAuxOperators ? session : nil
            )
            if outcome.baseOk {
                if allowAuxOperators {
                    if dspark != nil && !outcome.dsparkOk { dsparkActive = false }
                    if dflash != nil && !outcome.dflashOk { dflashActive = false }
                }
                pos = entry.key.tokens.count
                if pos == request.prompt.count { restoredExactFinalNorm = entry.finalNormRow }
                cacheHit = true
                matchedTokens = pos
                restoredFromCache = true
                onCacheEvent?(.hit(tier: "warm", matchedTokens: pos, totalTokens: request.prompt.count))
            }
        }
        if !restoredFromCache {
            slot.resetConvStates()
            if allowAuxOperators { resetSharedAuxOperators() }
            if canUseCache {
                onCacheEvent?(.miss(totalTokens: request.prompt.count))
            }
        }

        let retainTarget: Int? = if canUseCache {
            if let explicitLen = request.retainPrefixLength, explicitLen > 0, explicitLen <= request.prompt.count {
                explicitLen
            } else {
                request.prompt.count
            }
        } else {
            nil
        }
        return PrefixSetupResult(
            canUseCache: canUseCache, cacheHit: cacheHit, matchedTokens: matchedTokens,
            pos: pos, dsparkActive: dsparkActive, dflashActive: dflashActive,
            restoredExactFinalNorm: restoredExactFinalNorm, retainTarget: retainTarget,
            totalPromptTokens: request.prompt.count
        )
    }

    func storePrefixSnapshot(
        slot: Ink2SlotState,
        tokens: [Int],
        salt: String?,
        dspark: QWInkling2DSpark? = nil,
        dflash: QWDFlashDrafter? = nil,
        lsaOperator: QWLSAOperator? = nil,
        lastChunkRowsForFinalNorm: Int? = nil
    ) {
        let normRow = lastChunkRowsForFinalNorm.map {
            QWInkling2PrefixCache.captureFinalNormRow(
                session: session, lastChunkRows: $0, hidden: descriptor.hidden
            )
        }
        let snap = QWInkling2PrefixCache.capture(
            slot: slot, kvModeRaw: session.kvModeRaw,
            dspark: dspark, dflash: dflash, lsaOperator: lsaOperator,
            tokens: tokens, salt: salt,
            slidingWindow: ink2.slidingWindowSize, finalNormRow: normRow
        )
        prefixCache.store(entry: snap)
    }

    @inline(__always)
    func sampleRowToken(
        row: Int,
        vocab: Int,
        useGreedyHead: Bool,
        logitsPtr: UnsafePointer<Float>,
        headTokPtr: UnsafePointer<Int32>,
        sampler: QWSampler,
        penaltyHistory: [Int],
        constraint: (any QWTokenConstraint)?,
        allowedScratch: inout [Int]
    ) throws -> Int {
        if useGreedyHead {
            return Int(headTokPtr[row])
        }
        let rowBuf = UnsafeBufferPointer(start: logitsPtr.advanced(by: row * vocab), count: vocab)
        return try sampleToken(
            sampler: sampler, logitsRow: rowBuf, penaltyHistory: penaltyHistory,
            constraint: constraint, allowedScratch: &allowedScratch
        )
    }

    @inline(__always)
    func consumeEmittedToken(
        _ token: Int,
        row: Int = 0,
        vocab: Int,
        logitsPtr: UnsafePointer<Float>? = nil,
        stopIDs: Set<Int>,
        constraint: (any QWTokenConstraint)? = nil,
        lpSampler: QWSampler? = nil,
        topKCount: Int = 0,
        usePenalty: Bool = false,
        recentGeneratedForPenalty: inout [Int],
        penaltyHistory: inout [Int],
        generated: inout [Int],
        collectedLogprobs: inout [QWTokenLogprob],
        stopTokenID: inout Int?,
        onTokenLogprob: ((QWTokenLogprob) -> Void)? = nil,
        onStopToken: ((Int) -> Void)?,
        onToken: ((Int) -> Bool)?
    ) throws -> EmittedTokenOutcome {
        if stopIDs.contains(token) {
            stopTokenID = token
            onStopToken?(token)
            return .stopped
        }
        try constraint?.accept(token)
        if let lpSampler, let logitsPtr {
            let rowBuf = UnsafeBufferPointer(start: logitsPtr.advanced(by: row * vocab), count: vocab)
            let lp = lpSampler.computeTokenLogprob(logits: rowBuf, chosenToken: token, topK: topKCount)
            collectedLogprobs.append(lp)
            onTokenLogprob?(lp)
        }
        generated.append(token)
        if usePenalty {
            Self.recordGeneratedTokenForPenalty(
                token, vocab: vocab,
                recentGenerated: &recentGeneratedForPenalty,
                activeLoopTokens: &penaltyHistory
            )
        }
        return (onToken?(token) == false) ? .cancelled : .continued
    }

    @inline(__always)
    func consumeSpeculativeToken(
        _ token: Int,
        vocab: Int,
        stopIDs: Set<Int>,
        constraint: (any QWTokenConstraint)? = nil,
        usePenalty: Bool = false,
        recentGeneratedForPenalty: inout [Int],
        penaltyHistory: inout [Int],
        history: inout [Int],
        generated: inout [Int],
        stopTokenID: inout Int?,
        onStopToken: ((Int) -> Void)?,
        onToken: ((Int) -> Bool)?
    ) throws -> EmittedTokenOutcome {
        var unusedLogprobs: [QWTokenLogprob] = []
        let outcome = try consumeEmittedToken(
            token, row: 0, vocab: vocab, logitsPtr: nil,
            stopIDs: stopIDs, constraint: constraint,
            lpSampler: nil, topKCount: 0, usePenalty: usePenalty,
            recentGeneratedForPenalty: &recentGeneratedForPenalty,
            penaltyHistory: &penaltyHistory,
            generated: &generated, collectedLogprobs: &unusedLogprobs,
            stopTokenID: &stopTokenID, onTokenLogprob: nil,
            onStopToken: onStopToken, onToken: onToken
        )
        if outcome != .stopped {
            let prev2 = history.count >= 2 ? history[history.count - 2] : nil
            session.tokenRecycler.recordTransition(previous: prev2, current: history[history.count - 1], next: token)
            history.append(token)
        }
        return outcome
    }
}
