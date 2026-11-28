// PrefixCache.swift — QWEngineDriver PrefixCache family (split from QWEngineDriver.swift; same-module extension).
import Foundation
import Darwin
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWMLP
import QWHead
import QWMTP
import QWQuant

typealias QWAsyncColdWriteController = QWCore.QWAsyncColdWriteController

extension QWEngineDriver {
    public func configurePrefixCache(config: QWCacheConfig) throws {
        try trunkExecutor.configurePrefixCache(config)
    }

    func configureStandardPrefixCache(config: QWCacheConfig) throws {
        try withFlightPermit { try configurePrefixCacheLocked(config: config) }
    }

    private func configurePrefixCacheLocked(config: QWCacheConfig) throws {
        guard config.warmByteLimit >= 0, config.entryLimit > 0, verifiedSession == nil else {
            throw QWEngineDriverError.state("invalid prefix cache limits or unresolved verification")
        }
        let wiredEngineBytes = max(arena.committedBytes, capacityPlan.requiredBytes)
        let warmAdmission = QWResidencyAdmission.clampWarmCache(
            requestedWarmBytes: config.warmByteLimit,
            requestedSessions: config.maxConcurrentSessions,
            wiredEngineBytes: wiredEngineBytes,
            measuredAfterWire: true,
            snapshot: QWLiveDeviceMemorySnapshotSource(maxWorkingSetBytes: hal.caps.maxWorkingSetBytes).capture()
        )
        coldWriteController.reopenAndDrain()
        var effectiveConfig = config
        effectiveConfig.warmByteLimit = warmAdmission.admittedWarmBytes
        effectiveConfig.maxConcurrentSessions = warmAdmission.maxConcurrentSessions
        if effectiveConfig.warmByteLimit == 0 {
            prefixStore = nil
            cacheDiskManager = nil
            kvCache.onPagePressure = nil
            mtpCache?.onPagePressure = nil
        } else {
            let coordinator = try QWPrefixCacheCoordinator(config: effectiveConfig, vocab: geometry.vocab)
            let store = coordinator.warmStore!
            store.onEvict = { [weak self, weak store] key, value, allowsSuffix in
                guard let self else { return }
                QWPrefixCacheCoordinator.scheduleColdSpill(
                    key: key, value: value, allowsSuffix: allowsSuffix,
                    warmHasSharedBase: store?.hasSharedBase == true,
                    diskManager: self.cacheDiskManager,
                    coldWriteController: self.coldWriteController,
                    kvCache: self.kvCache, mtpCache: self.mtpCache,
                    onEvent: self.onCacheEvent
                )
            }
            self.prefixStore = store
            self.cacheDiskManager = coordinator.diskManager
            let pressureHandler: @Sendable () -> Bool = { [weak self] in
                self?.prefixStore?.evictOldest() ?? false
            }
            kvCache.onPagePressure = pressureHandler
            mtpCache?.onPagePressure = pressureHandler
        }
        lastPrefixCacheStats = nil
    }

    public func configurePrefixCache(byteLimit: Int, entryLimit: Int = 16) throws {
        try configurePrefixCache(config: QWCacheConfig(warmByteLimit: byteLimit, ssdByteLimit: 0, entryLimit: entryLimit))
    }

    func restorePrefixFromDisk(_ hit: QWCacheDiskManager.DiskHit,
                                       store: QWEnginePrefixStore<QWEngineCachedPrefix>) throws -> QWEngineCachedPrefix? {
        guard let diskManager = cacheDiskManager else { return nil }
        let restored = try diskManager.readDirectIntoCache(hit: hit, kvCache: kvCache, mtpCache: mtpCache)
        try validateMLAIndexerState(restored.state.mlaIndexerData, position: restored.state.position)
        let sharedPrefix = QWEngineSharedPrefix(state: restored.state, trunk: restored.trunk, mtp: restored.mtp)
        let cached = QWEngineCachedPrefix(prefix: sharedPrefix, nextToken: restored.nextToken, logits: restored.logits)
        _ = try? store.insert(tokens: hit.metadata.tokens, mode: hit.metadata.mode, value: cached,
                          payloadBytes: cached.memoryBytes, allowsSuffix: hit.metadata.allowsSuffix)
        return cached
    }

    func prefixPagePressure<T>(_ store: QWEnginePrefixStore<QWEngineCachedPrefix>,
                                       session: QWEngineSession? = nil,
                                       _ body: () throws -> T) throws -> T {
        while true {
            do { return try body() }
            catch let error as QWKVError {
                guard case .outOfPages = error, session?.stateFailed != true,
                      session?.verificationPending != true, store.evictOldest() else { throw error }
            }
        }
    }

    func withFlightPermit<T>(_ permit: QWGenerationPermit? = nil, _ body: () throws -> T) throws -> T {
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate(permit)
        return try body()
    }

    func maybePrefixPagePressure<T>(session: QWEngineSession? = nil, _ body: () throws -> T) throws -> T {
        if let store = prefixStore {
            return try prefixPagePressure(store, session: session, body)
        }
        return try body()
    }

    func resetIntermediatePrediction(_ session: QWEngineSession) {
        session.generated = 0
        session.roundRecord.outputCount = 0
        session.roundRecord.pendingToken = nil
    }

    func syncTokenHistoryAndRecycler(_ session: QWEngineSession, tokens: [Int], promptCount: Int) {
        session.tokenHistory = tokens
        session.promptTokenCount = promptCount
        session.tokenRecycler.reset()
        session.tokenRecycler.ingestSequence(tokens)
    }

    func prepareSuffixProgressAndWrites(
        _ session: QWEngineSession,
        matched: Int,
        promptCount: Int,
        onProgress: ((Int, Int) -> Void)?,
        permit: QWGenerationPermit
    ) throws {
        activePrefillBasePosition = matched
        let computedTotal = max(0, promptCount - matched)
        activePrefillTotalTokens = computedTotal
        onProgress?(0, computedTotal)
        if matched > 0, let store = prefixStore {
            try prepareSuffixKVCacheWrites(session, store: store, from: matched, to: promptCount, permit: permit)
        }
    }

    func awaitCommitAndCheck(_ session: QWEngineSession, context: String) throws {
        if let cb = session.lastCommitCB {
            cb.waitUntilCompleted()
            session.lastCommitCB = nil
            if syncCommitFailure(for: session, cb: cb) {
                throw QWEngineDriverError.state("commit GPU failure \(context)")
            }
        }
    }

    func executeSpeculativeRoundStep(
        _ session: QWEngineSession, remaining: Int, stopIDs: Set<Int>,
        drain: Bool, waitForStreamCommit: Bool = false,
        store: QWEnginePrefixStore<QWEngineCachedPrefix>?, permit: QWGenerationPermit
    ) throws -> QWGreedyRoundDecision {
        let decision: QWGreedyRoundDecision
        if session.stochastic != nil {
            let round = try maybePrefixPagePressure(session: session) {
                try prepareStochasticRound(session, maxNew: remaining, stopIDs: stopIDs, permit: permit)
            }
            try commitStochasticVerification(session, round: round, drain: drain, permit: permit)
            decision = round.decision
        } else {
            let round = try maybePrefixPagePressure(session: session) {
                try prepareGreedyRound(session, maxNew: remaining, stopIDs: stopIDs, permit: permit)
            }
            try commitGreedyVerification(session, round: round, drain: drain, permit: permit)
            decision = round.decision
        }
        if !drain, session.stochastic != nil, waitForStreamCommit {
            try awaitCommitAndCheck(session, context: "during streaming speculative generation")
        } else if !drain, session.stochastic == nil {
            let cb = session.lastCommitCB
            if syncCommitFailure(for: session, cb: cb?.status == .error ? cb : nil) {
                session.lastCommitCB = nil
                throw QWEngineDriverError.state("commit GPU failure during streaming speculative generation")
            }
        }
        return decision
    }

    func cachedAdmission(prompt: [Int], mode: GDNPrefillMode,
                                 store: QWEnginePrefixStore<QWEngineCachedPrefix>,
                                 permit: QWGenerationPermit) throws
        -> (session: QWEngineSession, first: Int?, matched: Int, tier: String?) {
        try withFlightPermit(permit) {
            let policyKey = prefixPolicyKey(mode: mode)
            let lookupTokens = QWVisualSpan.saltedCacheTokens(
                prompt: prompt, visualSpans: activeVisualSpans, vocab: geometry.vocab
            )
            var located = try QWPrefixCacheCoordinator.lookupTwoTier(
                lookupTokens: lookupTokens, policyKey: policyKey, store: store,
                diskManager: cacheDiskManager, onEvent: onCacheEvent,
                restoreFromDisk: { try restorePrefixFromDisk($0, store: store) }
            )
            if QWRadixKVGate.isEnabled, canSlicePureAttentionPages, mode != .naive,
               let donorHit = try store.lookupDeepestSharedPageDonor(
                   tokens: lookupTokens,
                   mode: policyKey,
                   minExclusiveTokens: located?.hit.tokenCount ?? 0
               ),
               let sliced = try slicePureAttentionCachedPrefix(
                   donorHit.donor, toPageAlignedTokens: donorHit.pageAlignedTokens
               ) {
                located = (
                    hit: QWEnginePrefixStore.Hit(value: sliced, tokenCount: donorHit.pageAlignedTokens),
                    tier: QWPrefixCacheTier.warmRAM.rawValue
                )
            }
            guard let located else {
                onCacheEvent?(.miss(totalTokens: prompt.count))
                return (try admitSession(promptTokens: prompt.count), nil, 0, nil)
            }
            let resolvedHit = located.hit, hitTier = located.tier
            onCacheEvent?(.hit(tier: hitTier, matchedTokens: resolvedHit.tokenCount, totalTokens: prompt.count))
            let value = resolvedHit.value
            guard QWPrefixCacheCoordinator.validateCachedValue(
                value, matchedTokens: resolvedHit.tokenCount, logitsBytes: sc.logits.bytes, vocab: geometry.vocab
            ) else {
                throw QWEngineDriverError.state("cached prefix state, prediction or logits invalid")
            }
            let session = try admitSession(promptTokens: prompt.count, prefix: value.prefix)
            if resolvedHit.tokenCount == prompt.count {
                if !value.logits.isEmpty {
                    let logitsDst = sc.logits.region.pointer.advanced(by: sc.logits.offset)
                    value.logits.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                        if let base = raw.baseAddress {
                            logitsDst.copyMemory(from: base, byteCount: raw.count)
                        }
                    }
                }
                sc.token.region.pointer.advanced(by: sc.token.offset).assumingMemoryBound(to: Int32.self).pointee = Int32(value.nextToken)
                return (session, value.nextToken, resolvedHit.tokenCount, hitTier)
            }
            // Complete cached chunk ran a head an intermediate chunk omits; reset prediction counters.
            guard session.generated == 1 else {
                finishAdmittedSessionAfterFailure(session)
                throw QWEngineDriverError.state("cached intermediate chunk has invalid prediction count")
            }
            resetIntermediatePrediction(session)
            return (session, nil, resolvedHit.tokenCount, hitTier)
        }
    }

    /// Admission failure cleanup already holds flight; public finish would lock
    /// it again. This session has never submitted GPU work.
    func finishAdmittedSessionAfterFailure(_ session: QWEngineSession) {
        session.sessionFinished = true
        if let handle = session.mtpKV { mtpCache?.finish(handle) }
        kvCache.finish(session.kv)
        admission.releaseSlot(GdnSlot(recSlab: session.recSlab, ringSlab: session.ringSlab,
                                    rec: session.rec, ring: session.ring,
                                    trunkHidden: session.trunkHidden,
                                    dflashSlot: session.dflashSlot,
                                    dsparkSlot: session.dsparkSlot))
    }

    func retainCachedPrompt(_ session: QWEngineSession, prompt: [Int], mode: GDNPrefillMode,
                                    first: Int, store: QWEnginePrefixStore<QWEngineCachedPrefix>,
                                    permit: QWGenerationPermit, isSharedBase: Bool = false,
                                    pinToSlot: Bool = true) throws -> Bool {
        try withFlightPermit(permit) {
            guard session.stochastic == nil else {
                throw QWEngineDriverError.state("lookup cache accepts only unconditioned prompt state")
            }
            try awaitCommitAndCheck(session, context: "before retaining prefix cache")
            let policyKey = prefixPolicyKey(mode: mode)
            let cacheTokens = QWVisualSpan.saltedCacheTokens(
                prompt: prompt, visualSpans: activeVisualSpans, vocab: geometry.vocab
            )
            let keyBytes = try store.keyBytes(tokens: cacheTokens, mode: policyKey)
            let stateBytes = session.recSlab.bytes + session.ringSlab.bytes + (session.trunkHidden?.bytes ?? 0) +
                activeSparseIndexerStateBytes(position: session.position) +
                (session.dflashSlot.map { $0.kSlab.bytes + $0.vSlab.bytes } ?? 0) +
                (session.dsparkSlot?.ringBytes ?? 0)
            let sharedBaseHit = isSharedBase ? nil : store.sharedBasePrefix(matching: cacheTokens, mode: policyKey)
            let trunkWritten = kvCache.written(session.kv)
            let mtpWritten = session.mtpKV.flatMap { mtpCache?.written($0) } ?? 0
            let trunkKVBytes = kvCache.sharedPrefixBytes(tokenCount: trunkWritten)
            let mtpKVBytes = mtpCache?.sharedPrefixBytes(tokenCount: mtpWritten) ?? 0
            let pinnedBaseKVBytes = (sharedBaseHit?.value.prefix.trunk.memoryBytes ?? 0)
                + (sharedBaseHit?.value.prefix.mtp?.memoryBytes ?? 0)
            let sharedKVBytes = QWRadixKVGate.isEnabled ? (trunkKVBytes + mtpKVBytes) : pinnedBaseKVBytes
            let incrementalKVBytes = QWTwoTierPrefixCache.incrementalCharge(
                fullBytes: trunkKVBytes + mtpKVBytes, sharedBaseBytes: sharedKVBytes
            )
            let charge = stateBytes + incrementalKVBytes + sc.logits.bytes
            let minRequiredBytes = stateBytes
                + QWTwoTierPrefixCache.incrementalCharge(fullBytes: trunkKVBytes + mtpKVBytes, sharedBaseBytes: pinnedBaseKVBytes)
                + sc.logits.bytes + keyBytes
            guard keyBytes <= store.byteLimit,
                  charge <= (store.byteLimit - keyBytes),
                  (!QWRadixKVGate.isEnabled || minRequiredBytes <= store.byteLimit) else {
                onCacheEvent?(.skippedOversize(tokens: cacheTokens.count, charge: max(charge + keyBytes, minRequiredBytes), warmLimitBytes: store.byteLimit))
                return false
            }
            let prefix = try QWEngineSessionState.captureSharedPrefix(
                session, cache: kvCache, mtpCache: mtpCache,
                maxBytes: (store.byteLimit - keyBytes - sc.logits.bytes) + sharedKVBytes,
                mlaIndexerData: captureMLAIndexerState(session)
            )
            let value = QWEngineCachedPrefix(prefix: prefix, nextToken: first,
                logits: Data(bytes: sc.logits.region.pointer.advanced(by: sc.logits.offset), count: sc.logits.bytes))
            let actualPayloadCharge = QWRadixKVGate.isEnabled
                ? value.memoryBytes
                : QWTwoTierPrefixCache.incrementalCharge(fullBytes: value.memoryBytes, sharedBaseBytes: sharedKVBytes)
            let beforeStoreBytes = store.chargedBytes
            let inserted = try store.insert(tokens: cacheTokens, mode: policyKey, value: value,
                                            payloadBytes: actualPayloadCharge, allowsSuffix: mode != .naive,
                                            isSharedBase: isSharedBase, pinToSlot: pinToSlot)
            if inserted {
                let reportedCharge = QWRadixKVGate.isEnabled
                    ? max(keyBytes + stateBytes + sc.logits.bytes, store.chargedBytes - beforeStoreBytes)
                    : (actualPayloadCharge + keyBytes)
                QWPrefixCacheCoordinator.spillEntryIfEligible(
                    diskManager: cacheDiskManager, isSharedBase: isSharedBase,
                    warmHasSharedBase: store.hasSharedBase, tokens: cacheTokens, mode: policyKey,
                    value: value, kvCache: kvCache, mtpCache: mtpCache, allowsSuffix: mode != .naive
                )
                onCacheEvent?(.retained(tokens: cacheTokens.count, chargedBytes: reportedCharge, storeTotalBytes: store.chargedBytes, warmLimitBytes: store.byteLimit))
            }
            return inserted
        }
    }

    func prepareSuffixKVCacheWrites(
        _ session: QWEngineSession,
        store: QWEnginePrefixStore<QWEngineCachedPrefix>,
        from startPos: Int,
        to endPos: Int,
        permit: QWGenerationPermit
    ) throws {
        let count = endPos - startPos
        guard count > 0 else { return }
        try prefixPagePressure(store, session: session) {
            flight.lock()
            defer { flight.unlock() }
            try requestGate.validate(permit)
            try kvCache.prepareWrites(session.kv, positions: session.position..<endPos)
            if let mtpCache, let handle = session.mtpKV {
                let rows = mtpCache.written(handle)
                try mtpCache.prepareWrites(handle, positions: rows..<(rows + count))
            }
        }
    }

    private func pinSharedBaseValue(
        store: QWEnginePrefixStore<QWEngineCachedPrefix>,
        tokens: [Int],
        mode: String,
        value: QWEngineCachedPrefix
    ) throws -> Bool {
        let keyBytes = try store.keyBytes(tokens: tokens, mode: mode)
        store.ensureByteLimit(atLeast: value.memoryBytes + keyBytes + (1024 << 20))
        let inserted = try store.insert(
            tokens: tokens, mode: mode, value: value,
            payloadBytes: value.memoryBytes, allowsSuffix: true, isSharedBase: true
        )
        if inserted {
            lsaOperator?.setPinnedBaseState(value.prefix.state.mlaIndexerData, position: value.prefix.state.position)
        }
        return inserted
    }
    /// Suffix reuse is limited to complete shipping prefill chunks.
    public func generateCachedGreedy(prompt: [Int], maxNew: Int, stopIds: Set<Int> = [],
                              speculative: Bool = false, retainPrefixLength: Int? = nil,
                              onToken: ((Int) -> Bool)? = nil,
                              inspectCompleted: ((QWEngineSession) throws -> Void)? = nil) throws -> [Int] {
        try generateCached(prompt: prompt, maxNew: maxNew, stopIds: stopIds, speculative: speculative,
                           retainPrefixLength: retainPrefixLength, onToken: onToken,
                           inspectCompleted: inspectCompleted, stochastic: nil)
    }

    public func generateCachedStochastic(prompt: [Int], maxNew: Int, config: QWSamplerConfig,
                                  stopIds: Set<Int> = [], speculative: Bool = false,
                                  retainPrefixLength: Int? = nil,
                                  onToken: ((Int) -> Bool)? = nil,
                                  inspectCompleted: ((QWEngineSession) throws -> Void)? = nil) throws -> [Int] {
        var normConfig = config
        normConfig.uniformVersion = .upper24
        if normConfig.temperature <= 0 {
            normConfig.temperature = 1
            normConfig.topK = 1
            normConfig.topP = 1
        }
        return try generateCached(prompt: prompt, maxNew: maxNew, stopIds: stopIds, speculative: speculative,
                                  retainPrefixLength: retainPrefixLength, onToken: onToken,
                                  inspectCompleted: inspectCompleted,
                                  stochastic: QWStochasticSession(geometry: geometry, config: normConfig))
    }

    func generateCached(prompt: [Int], maxNew: Int, stopIds: Set<Int>, speculative: Bool,
                                retainPrefixLength: Int? = nil,
                                onToken: ((Int) -> Bool)?,
                                inspectCompleted: ((QWEngineSession) throws -> Void)?,
                                stochastic owner: QWStochasticSession?) throws -> [Int] {
        try requestGate.withRequest(flight: flight) { permit in
            guard let store = prefixStore,
                  QWBatchPartitionSelection.validateRequestBounds(
                      prompt: prompt, maxNew: maxNew, stopIDs: stopIds, vocab: geometry.vocab
                  ),
                  !speculative || mtpDrafts > 0 else {
                throw QWEngineDriverError.state("invalid cached generation request or unconfigured cache")
            }
            guard prompt.count < ropeRows else {
                throw QWEngineDriverError.posBeyondTable(prompt.count)
            }
            lastPrefixCacheStats = nil
            let effectiveMaxNew = QWBatchPartitionSelection.clampedMaxNew(
                requested: maxNew, promptCount: prompt.count, activeCapacity: ropeRows
            )
            if effectiveMaxNew == 0 { return [] }
            let mode = gdnPrefillMode, started = Date()
            let loaded = try prefixPagePressure(store) {
                try cachedAdmission(prompt: prompt, mode: mode, store: store, permit: permit)
            }
            let session = loaded.session
            defer { finishSession(session) }
            session.tokenHistory = Array(prompt.prefix(loaded.matched))
            session.promptTokenCount = prompt.count
            if let owner { session.stochastic = owner }
            var first: Int
            let unconditionedFirst: Int
            var skippedOversize = false
            var computePrefillSec: Double? = nil
            if let cached = loaded.first {
                unconditionedFirst = cached
                first = try selectConstrainedToken(session: session, rawArgmax: cached)
                session.roundRecord.pendingToken = first
                activePrefillProgress?(0, 0)
            } else {
                // Reserve COW for the entire suffix before its first GPU chunk; retrying partial prefill is forbidden.
                try prepareSuffixProgressAndWrites(
                    session, matched: loaded.matched, promptCount: prompt.count,
                    onProgress: activePrefillProgress, permit: permit
                )
                let firstVis = activeVisualSpans.map(\.tokenOffset).min() ?? prompt.count
                let policyKeyForRetain = prefixPolicyKey(mode: mode)
                let periodicInterval = (QWRadixKVGate.isEnabled && !canSlicePureAttentionPages && mode != .naive)
                    ? 4096 : 0
                let retainPoints = QWTwoTierPrefixCache.computeRetainPoints(
                    explicitRetainLength: retainPrefixLength,
                    matchedTokens: loaded.matched,
                    firstVisualTokenOffset: firstVis,
                    promptCount: prompt.count,
                    sharedBranchPrefixLength: store.sharedBranchPrefixLength(
                        for: Array(prompt.prefix(firstVis)), mode: policyKeyForRetain
                    ),
                    periodicCheckpointInterval: periodicInterval
                )
                let prefillStart = Date()
                if !retainPoints.isEmpty {
                    var totalComputeSec = 0.0
                    for retainLen in retainPoints where retainLen > session.position && retainLen < prompt.count {
                        let segStart = Date()
                        let savedConstraint = activeConstraint
                        activeConstraint = nil
                        let midFirst: Int
                        do {
                            midFirst = try prefill(session, tokens: Array(prompt[session.position..<retainLen]),
                                                   mode: mode, permit: permit, evictScratchOnCompletion: false)
                            activeConstraint = savedConstraint
                        } catch {
                            activeConstraint = savedConstraint
                            throw error
                        }
                        totalComputeSec += Date().timeIntervalSince(segStart)
                        do {
                            let savedStochastic = session.stochastic
                            session.stochastic = nil
                            defer { session.stochastic = savedStochastic }
                            skippedOversize = try !retainCachedPrompt(
                                session, prompt: Array(prompt.prefix(retainLen)),
                                mode: mode, first: midFirst, store: store, permit: permit,
                                isSharedBase: false
                            )
                        }
                        // Reset intermediate head prediction counters and reserve COW for tail tokens.
                        resetIntermediatePrediction(session)
                        session.tokenHistory = Array(prompt.prefix(retainLen))
                        try prepareSuffixKVCacheWrites(session, store: store, from: retainLen, to: prompt.count, permit: permit)
                    }
                    let tailStart = Date()
                    first = try prefill(session, tokens: Array(prompt[session.position..<prompt.count]),
                                        mode: mode, permit: permit)
                    computePrefillSec = totalComputeSec + Date().timeIntervalSince(tailStart)
                } else {
                    first = try prefill(session, tokens: Array(prompt.dropFirst(loaded.matched)),
                                        mode: mode, permit: permit)
                    computePrefillSec = Date().timeIntervalSince(prefillStart)
                }
                unconditionedFirst = Int(sc.token.region.buffer.contents()
                    .advanced(by: sc.token.region.base + sc.token.offset)
                    .assumingMemoryBound(to: Int32.self).pointee)
            }
            let needsFullPromptRetain = loaded.first == nil &&
                !(retainPrefixLength != nil && retainPrefixLength! > loaded.matched && retainPrefixLength! < prompt.count)
            if owner != nil {
                session.promptTokenCount = prompt.count
                first = try sampleStochasticTarget(session, permit: permit)
            }
            syncTokenHistoryAndRecycler(
                session,
                tokens: session.tokenHistory.count == prompt.count + 1 ? session.tokenHistory : (prompt + [first]),
                promptCount: prompt.count
            )
            let firstTokenSec = Date().timeIntervalSince(started)
            let prefillSec = computePrefillSec ?? firstTokenSec
            var emittedFirst = false
            var cancelledOnFirst = false
            if effectiveMaxNew > 0, !stopIds.contains(first), let onToken {
                emittedFirst = true
                if onToken(first) == false {
                    cancelledOnFirst = true
                }
            }
            if needsFullPromptRetain {
                let savedStochastic = session.stochastic
                let savedConstraint = activeConstraint
                session.stochastic = nil
                activeConstraint = nil
                defer {
                    session.stochastic = savedStochastic
                    activeConstraint = savedConstraint
                }
                skippedOversize = try !retainCachedPrompt(session, prompt: prompt, mode: mode,
                                                          first: unconditionedFirst,
                                                          store: store, permit: permit)
            }
            let decoding = Date()
            let loopOnToken: ((Int) -> Bool)? = onToken.map { cb in
                { token in
                    if emittedFirst {
                        emittedFirst = false
                        return !cancelledOnFirst
                    }
                    return cb(token)
                }
            }
            let committedRound: ((Int) throws -> QWGreedyRoundDecision)? = speculative ? { remaining in
                try self.executeSpeculativeRoundStep(
                    session, remaining: remaining, stopIDs: stopIds, drain: false,
                    waitForStreamCommit: onToken != nil, store: store, permit: permit
                )
            } : nil
            let output = try QWGreedyGenerationLoop.run(vocab: geometry.vocab, first: first, maxNew: effectiveMaxNew, stopIDs: stopIds,
                onToken: loopOnToken, ordinary: { token in
                    try self.prefixPagePressure(store, session: session) {
                        if owner != nil {
                            return try self.decodeStochastic(session, token: token, permit: permit)
                        }
                        return try self.decodeStep(session, token: token, permit: permit)
                    }
                }, committedRound: committedRound, streamsCommittedRounds: true,
                onStopToken: { self.lastStopToken = $0 },
                backoff: .armed(position: session.position, drafts: mtpDrafts),
                canSpeculate: {
                    self.canRunSpeculativeRound(session)
                        && self.canSpeculateActiveConstraint
                })
            try awaitCommitAndCheck(session, context: "during generation")
            let snap = QWPrefixCacheCoordinator.buildMetricsSnapshot(
                promptCount: prompt.count, output: output, prefillSec: prefillSec,
                decodeSec: Date().timeIntervalSince(decoding), firstTokenSec: firstTokenSec,
                speculative: speculative, matchedTokens: loaded.matched, tier: loaded.tier,
                skippedOversize: skippedOversize, store: store
            )
            (lastGeneration, lastExperimentalGeneration, lastPrefixCacheStats) = (snap.generation, snap.experimental, snap.cache)
            try inspectCompleted?(session)
            return output.tokens
        }
    }

    /// Prefills `tokens`, writes a standalone `.qwcache` file at `outputURL`, and optionally pins it in warm RAM.
    public func buildStandalonePrefixCache(
        tokens: [Int],
        outputURL: URL,
        pinAsSharedBase: Bool = true,
        onPrefillProgress: (@Sendable (Int, Int) -> Void)?
    ) throws -> Int {
        try trunkExecutor.buildStandalonePrefixCache(
            tokens: tokens,
            outputURL: outputURL,
            pinAsSharedBase: pinAsSharedBase,
            onPrefillProgress: onPrefillProgress
        )
    }

    func buildStandardStandalonePrefixCache(
        tokens: [Int],
        outputURL: URL,
        pinAsSharedBase: Bool = true,
        onPrefillProgress: (@Sendable (Int, Int) -> Void)?
    ) throws -> Int {
        return try requestGate.withRequest(flight: flight) { permit in
            guard !tokens.isEmpty, tokens.count <= ropeRows,
                  tokens.allSatisfy({ (0..<geometry.vocab).contains($0) }) else {
                throw QWEngineDriverError.state("invalid standalone prefix tokens")
            }
            activePrefillProgress = onPrefillProgress
            activePrefillTotalTokens = nil
            activePrefillBasePosition = 0
            defer {
                activePrefillProgress = nil
                activePrefillTotalTokens = nil
                activePrefillBasePosition = 0
            }
            let mode = gdnPrefillMode
            let session = try admitSession(promptTokens: tokens.count)
            defer { finishSession(session) }
            let first = try prefill(session, tokens: tokens, mode: mode, permit: permit)
            try awaitCommitAndCheck(session, context: "while building standalone prefix cache")
            lsaOperator?.setPinnedBaseState(nil, position: 0)
            let prefix = try captureSharedPrefix(session, maxBytes: Int.max, permit: permit)
            let logits = Data(bytes: sc.logits.region.pointer.advanced(by: sc.logits.offset), count: sc.logits.bytes)
            let policyKey = prefixPolicyKey(mode: mode)
            guard let written = try QWCacheDiskManager.writeStandaloneFile(
                to: outputURL,
                tokens: tokens,
                mode: policyKey,
                nextToken: first,
                state: prefix.state,
                logits: logits,
                trunk: prefix.trunk,
                kvCache: kvCache,
                mtp: prefix.mtp,
                mtpCache: mtpCache,
                allowsSuffix: mode != .naive
            ) else {
                throw QWEngineDriverError.state("failed to write standalone prefix cache")
            }
            if pinAsSharedBase, let store = prefixStore {
                let cached = QWEngineCachedPrefix(prefix: prefix, nextToken: first, logits: logits)
                _ = try? pinSharedBaseValue(store: store, tokens: tokens, mode: policyKey, value: cached)
            }
            return written.totalBytes
        }
    }

    /// Mounts a standalone `.qwcache` file from `fileURL` into warm RAM as a pinned shared base prefix.
    public func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) {
        try trunkExecutor.mountStandalonePrefixCache(from: fileURL)
    }

    func mountStandardStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) {
        return try withFlightPermit {
            if prefixStore == nil {
                try configurePrefixCacheLocked(config: QWCacheConfig(warmByteLimit: 512 << 20, ssdByteLimit: 0, entryLimit: 32))
            }
            guard let store = prefixStore else {
                throw QWEngineDriverError.state("prefix store unavailable for mounting memory pack")
            }
            let read = try QWCacheDiskManager.readStandaloneFile(at: fileURL, kvCache: kvCache, mtpCache: mtpCache)
            guard QWPrefixCacheCoordinator.validateStandaloneRead(
                tokens: read.metadata.tokens, mode: read.metadata.mode, statePosition: read.state.position,
                logitsCount: read.logits.count, logitsBytes: sc.logits.bytes, nextToken: read.nextToken,
                vocab: geometry.vocab, hasReadMTP: read.mtp != nil, hasEngineMTP: mtpCache != nil
            ) else {
                throw QWEngineDriverError.state("standalone prefix cache payload, logits, mode, or MTP mismatch")
            }
            lsaOperator?.setPinnedBaseState(nil, position: 0)
            try validateMLAIndexerState(read.state.mlaIndexerData, position: read.state.position)
            let sharedPrefix = QWEngineSharedPrefix(state: read.state, trunk: read.trunk, mtp: read.mtp)
            let cached = QWEngineCachedPrefix(prefix: sharedPrefix, nextToken: read.nextToken, logits: read.logits)
            guard try pinSharedBaseValue(store: store, tokens: read.metadata.tokens, mode: read.metadata.mode, value: cached) else {
                throw QWEngineDriverError.state("failed to pin mounted prefix cache into warm store")
            }
            return (read.metadata.tokens, cached.memoryBytes)
        }
    }

    /// Unmounts all pinned shared-base prefixes from the warm prefix store.
    public func unmountSharedBasePrefixes() {
        trunkExecutor.unmountSharedBasePrefixes()
    }

    func unmountStandardSharedBasePrefixes() {
        guard flight.try() else { return }
        defer { flight.unlock() }
        lsaOperator?.setPinnedBaseState(nil, position: 0)
        prefixStore?.removeSharedBaseEntries()
    }
}
