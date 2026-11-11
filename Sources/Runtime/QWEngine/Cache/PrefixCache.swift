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
import QWState

enum QWCacheIO {
    static func attempt<T>(
        _ operation: QWPrefixCacheIOOperation,
        onEvent: (@Sendable (QWPrefixCacheEvent) -> Void)?,
        _ body: () throws -> T
    ) -> T? {
        do {
            return try body()
        } catch {
            let message = String(describing: error)
            if let onEvent {
                onEvent(.ioFailure(operation: operation, message: message))
            } else {
                NSLog("Prefix cache SSD %@ failed: %@", operation.rawValue, message)
            }
            return nil
        }
    }
}

extension QWEngineDriver {
    private static let maxOpportunisticColdWriteBytes = 256 << 20
    public func configurePrefixCache(config: QWCacheConfig) throws {
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate()
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
        var effectiveConfig = config
        effectiveConfig.warmByteLimit = warmAdmission.admittedWarmBytes
        effectiveConfig.maxConcurrentSessions = warmAdmission.maxConcurrentSessions
        if effectiveConfig.warmByteLimit == 0 {
            prefixStore = nil
            cacheDiskManager = nil
        } else {
            let coordinator = try QWPrefixCacheCoordinator(config: effectiveConfig, vocab: geometry.vocab)
            let store = coordinator.warmStore!
            let diskManager = coordinator.diskManager
            store.onEvict = { [weak self, weak store] key, value, allowsSuffix in
                guard let self else { return }
                self.onCacheEvent?(.evicted(tokens: key.tokens.count, freedBytes: value.memoryBytes))
                guard store?.hasSharedBase != true,
                      let dm = self.cacheDiskManager,
                      value.memoryBytes <= Self.maxOpportunisticColdWriteBytes,
                      value.memoryBytes <= dm.config.ssdByteLimit else { return }
                _ = QWCacheIO.attempt(.write, onEvent: self.onCacheEvent) {
                    try dm.writeDirect(
                        tokens: key.tokens, mode: key.mode, nextToken: value.nextToken,
                        state: value.prefix.state, logits: value.logits, trunk: value.prefix.trunk,
                        kvCache: self.kvCache, mtp: value.prefix.mtp, mtpCache: self.mtpCache,
                        allowsSuffix: allowsSuffix)
                }
            }
            self.prefixStore = store
            self.cacheDiskManager = diskManager
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

    func cachedAdmission(prompt: [Int], mode: GDNPrefillMode,
                                 store: QWEnginePrefixStore<QWEngineCachedPrefix>,
                                 permit: QWGenerationPermit) throws
        -> (session: QWEngineSession, first: Int?, matched: Int, tier: String?) {
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate(permit)
        let policyKey = prefixPolicyKey(mode: mode)
        let lookupTokens = QWVisualSpan.saltedCacheTokens(
            prompt: prompt, visualSpans: activeVisualSpans, vocab: geometry.vocab
        )
        var hit = lookupTokens.isEmpty ? nil : try store.lookup(tokens: lookupTokens, mode: policyKey)
        var hitTier: String? = nil
        if hit != nil {
            hitTier = QWPrefixCacheTier.warmRAM.rawValue
        } else if !lookupTokens.isEmpty, let diskManager = cacheDiskManager,
           let diskHit = diskManager.lookup(tokens: lookupTokens, mode: policyKey),
           diskHit.matchedTokens >= 256 || !diskManager.config.wipeOnStart && lookupTokens.count <= 16 {
            if let result = QWCacheIO.attempt(.restore, onEvent: onCacheEvent, {
                try restorePrefixFromDisk(diskHit, store: store)
            }), let restored = result {
                hit = QWEnginePrefixStore.Hit(value: restored, tokenCount: diskHit.matchedTokens)
                hitTier = QWPrefixCacheTier.coldSSD.rawValue
            }
        }
        guard let resolvedHit = hit else {
            onCacheEvent?(.miss(totalTokens: prompt.count))
            return (try admitSession(promptTokens: prompt.count), nil, 0, nil)
        }
        onCacheEvent?(.hit(tier: hitTier ?? QWPrefixCacheTier.warmRAM.rawValue, matchedTokens: resolvedHit.tokenCount, totalTokens: prompt.count))
        let value = resolvedHit.value
        guard value.prefix.state.stochastic == nil,
              value.prefix.position == resolvedHit.tokenCount, value.logits.count == sc.logits.bytes,
              (0..<geometry.vocab).contains(value.nextToken) else {
            throw QWEngineDriverError.state("cached prefix state, prediction or logits invalid")
        }
        let session = try admitSession(promptTokens: prompt.count, prefix: value.prefix)
        if resolvedHit.tokenCount == prompt.count {
            value.logits.withUnsafeBytes {
                sc.logits.region.pointer.advanced(by: sc.logits.offset)
                    .copyMemory(from: $0.baseAddress!, byteCount: $0.count)
            }
            sc.token.region.pointer.advanced(by: sc.token.offset)
                .assumingMemoryBound(to: Int32.self).pointee = Int32(value.nextToken)
            return (session, value.nextToken, resolvedHit.tokenCount, hitTier)
        }
        // A complete cached chunk ran a head that an intermediate fresh chunk
        // would omit. Its prediction is not output and must not affect counters.
        guard session.generated == 1 else {
            finishAdmittedSessionAfterFailure(session)
            throw QWEngineDriverError.state("cached intermediate chunk has invalid prediction count")
        }
        session.generated = 0
        if mtpCache != nil {
            session.roundRecord.outputCount = 0
            session.roundRecord.pendingToken = nil
        }
        return (session, nil, resolvedHit.tokenCount, hitTier)
    }

    /// Admission failure cleanup already holds flight; public finish would lock
    /// it again. This session has never submitted GPU work.
    func finishAdmittedSessionAfterFailure(_ session: QWEngineSession) {
        session.sessionFinished = true
        if let handle = session.mtpKV { mtpCache?.finish(handle) }
        kvCache.finish(session.kv)
        slotLock.withLock {
            freeSlots.append(GdnSlot(recSlab: session.recSlab, ringSlab: session.ringSlab,
                                    rec: session.rec, ring: session.ring,
                                    trunkHidden: session.trunkHidden,
                                    dflashSlot: session.dflashSlot))
        }
    }

    func retainCachedPrompt(_ session: QWEngineSession, prompt: [Int], mode: GDNPrefillMode,
                                    first: Int, store: QWEnginePrefixStore<QWEngineCachedPrefix>,
                                    permit: QWGenerationPermit, isSharedBase: Bool = false) throws -> Bool {
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate(permit)
        guard session.stochastic == nil else {
            throw QWEngineDriverError.state("lookup cache accepts only unconditioned prompt state")
        }
        let policyKey = prefixPolicyKey(mode: mode)
        let cacheTokens = QWVisualSpan.saltedCacheTokens(
            prompt: prompt, visualSpans: activeVisualSpans, vocab: geometry.vocab
        )
        let keyBytes = try store.keyBytes(tokens: cacheTokens, mode: policyKey)
        let stateBytes = session.recSlab.bytes + session.ringSlab.bytes + (session.trunkHidden?.bytes ?? 0) +
            activeSparseIndexerStateBytes(position: session.position)
        let trunkPages = (kvCache.written(session.kv) + QWKVCache.pageTokens - 1) / QWKVCache.pageTokens
        let trunkBytes = trunkPages * attnLayers.count * 2 * kvCache.bytesPerPage
        let mtpBytes: Int = {
            guard let mtpCache, let handle = session.mtpKV else { return 0 }
            let pages = (mtpCache.written(handle) + QWKVCache.pageTokens - 1) / QWKVCache.pageTokens
            return pages * 2 * mtpCache.bytesPerPage
        }()
        let sharedBaseHit = isSharedBase ? nil : store.sharedBasePrefix(matching: cacheTokens, mode: policyKey)
        let sharedKVBytes = (sharedBaseHit?.value.prefix.trunk.memoryBytes ?? 0)
            + (sharedBaseHit?.value.prefix.mtp?.memoryBytes ?? 0)
        let incrementalKVBytes = max(0, trunkBytes + mtpBytes - sharedKVBytes)
        let charge = stateBytes + incrementalKVBytes + sc.logits.bytes
        let maxSingleEntryBytes = store.byteLimit - keyBytes
        guard keyBytes <= store.byteLimit, charge <= maxSingleEntryBytes else {
            onCacheEvent?(.skippedOversize(tokens: cacheTokens.count, charge: charge + keyBytes, warmLimitBytes: store.byteLimit))
            return false
        }
        let captureMaxBytes = (store.byteLimit - keyBytes - sc.logits.bytes) + sharedKVBytes
        let prefix = try QWEngineSessionState.captureSharedPrefix(session, cache: kvCache, mtpCache: mtpCache,
                                                                 maxBytes: captureMaxBytes,
                                                                 mlaIndexerData: captureMLAIndexerState(session))
        let value = QWEngineCachedPrefix(prefix: prefix, nextToken: first,
            logits: Data(bytes: sc.logits.region.pointer.advanced(by: sc.logits.offset), count: sc.logits.bytes))
        let actualPayloadCharge = max(0, value.memoryBytes - sharedKVBytes)
        let inserted = try store.insert(tokens: cacheTokens, mode: policyKey, value: value, payloadBytes: actualPayloadCharge,
            allowsSuffix: mode != .naive, isSharedBase: isSharedBase)
        if inserted {
            if !isSharedBase, !store.hasSharedBase,
               let dm = self.cacheDiskManager, !dm.config.wipeOnStart,
               value.memoryBytes <= Self.maxOpportunisticColdWriteBytes,
               value.memoryBytes <= dm.config.ssdByteLimit {
                _ = try? dm.writeDirect(
                    tokens: cacheTokens,
                    mode: policyKey,
                    nextToken: value.nextToken,
                    state: value.prefix.state,
                    logits: value.logits,
                    trunk: value.prefix.trunk,
                    kvCache: self.kvCache,
                    mtp: value.prefix.mtp,
                    mtpCache: self.mtpCache,
                    allowsSuffix: mode != .naive
                )
            }
            onCacheEvent?(.retained(tokens: cacheTokens.count, chargedBytes: actualPayloadCharge + keyBytes, storeTotalBytes: store.chargedBytes, warmLimitBytes: store.byteLimit))
        }
        return inserted
    }

    /// Diagnostic owned-session interface. Production lookup remains gated.
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
        try generateCached(prompt: prompt, maxNew: maxNew, stopIds: stopIds, speculative: speculative,
                           retainPrefixLength: retainPrefixLength, onToken: onToken,
                           inspectCompleted: inspectCompleted,
                           stochastic: QWStochasticSession(geometry: geometry, config: config))
    }

    func generateCached(prompt: [Int], maxNew: Int, stopIds: Set<Int>, speculative: Bool,
                                retainPrefixLength: Int? = nil,
                                onToken: ((Int) -> Bool)?,
                                inspectCompleted: ((QWEngineSession) throws -> Void)?,
                                stochastic owner: QWStochasticSession?) throws -> [Int] {
        try requestGate.withRequest(flight: flight) { permit in
            guard let store = prefixStore, !prompt.isEmpty, maxNew >= 0, prompt.count <= ropeRows,
                  prompt.allSatisfy({ (0..<geometry.vocab).contains($0) }),
                  stopIds.allSatisfy({ (0..<geometry.vocab).contains($0) }),
                  !speculative || mtpDrafts > 0 else {
                throw QWEngineDriverError.state("invalid cached generation request or unconfigured cache")
            }
            lastPrefixCacheStats = nil
            if maxNew == 0 { return [] }
            let mode = gdnPrefillMode, started = Date()
            let loaded = try prefixPagePressure(store) {
                try cachedAdmission(prompt: prompt, mode: mode, store: store, permit: permit)
            }
            let session = loaded.session
            defer { finishSession(session) }
            var first: Int
            let unconditionedFirst: Int
            var skippedOversize = false
            var computePrefillSec: Double? = nil
            if let cached = loaded.first {
                unconditionedFirst = cached
                first = try selectConstrainedToken(session: session, rawArgmax: cached)
                if mtpCache != nil {
                    session.roundRecord.pendingToken = first
                }
            } else {
                if loaded.matched > 0 {
                    // Reserve COW for the entire suffix before its first GPU
                    // chunk; retrying a partially completed prefill is forbidden.
                    try prefixPagePressure(store, session: session) {
                        flight.lock()
                        defer { flight.unlock() }
                        try requestGate.validate(permit)
                        try kvCache.prepareWrites(session.kv, positions: session.position..<prompt.count)
                        if let mtpCache, let handle = session.mtpKV {
                            let rows = mtpCache.written(handle)
                            let count = prompt.count - loaded.matched
                            try mtpCache.prepareWrites(handle, positions: rows..<(rows + count))
                        }
                    }
                }
                let firstVis = activeVisualSpans.map(\.tokenOffset).min() ?? prompt.count
                let effectiveRetainLen: Int? = {
                    if let explicit = retainPrefixLength { return explicit <= firstVis ? explicit : nil }
                    if firstVis >= 64 {
                        let autoRetain = firstVis - 4
                        if autoRetain >= 60, autoRetain > loaded.matched + 4, autoRetain < firstVis {
                            return autoRetain
                        }
                    }
                    if let branchLen = store.sharedBranchPrefixLength(for: Array(prompt.prefix(firstVis))),
                       branchLen > loaded.matched, branchLen < firstVis {
                        return branchLen
                    }
                    return nil
                }()
                let prefillStart = Date()
                if let retainLen = effectiveRetainLen, retainLen > loaded.matched, retainLen < prompt.count {
                    let savedConstraint = activeConstraint
                    activeConstraint = nil
                    let midFirst: Int
                    do {
                        midFirst = try prefill(session, tokens: Array(prompt[loaded.matched..<retainLen]),
                                               mode: mode, permit: permit)
                        activeConstraint = savedConstraint
                    } catch {
                        activeConstraint = savedConstraint
                        throw error
                    }
                    let midSec = Date().timeIntervalSince(prefillStart)
                    skippedOversize = try !retainCachedPrompt(
                        session, prompt: Array(prompt.prefix(retainLen)),
                        mode: mode, first: midFirst, store: store, permit: permit,
                        isSharedBase: retainPrefixLength != nil && !store.hasSharedBase
                    )
                    // Reset intermediate head prediction counters and reserve COW for tail tokens
                    session.generated = 0
                    if mtpCache != nil {
                        session.roundRecord.outputCount = 0
                        session.roundRecord.pendingToken = nil
                    }
                    try prefixPagePressure(store, session: session) {
                        flight.lock()
                        defer { flight.unlock() }
                        try requestGate.validate(permit)
                        try kvCache.prepareWrites(session.kv, positions: session.position..<prompt.count)
                        if let mtpCache, let handle = session.mtpKV {
                            let rows = mtpCache.written(handle)
                            let count = prompt.count - retainLen
                            try mtpCache.prepareWrites(handle, positions: rows..<(rows + count))
                        }
                    }
                    let tailStart = Date()
                    first = try prefill(session, tokens: Array(prompt[retainLen..<prompt.count]),
                                        mode: mode, permit: permit)
                    computePrefillSec = midSec + Date().timeIntervalSince(tailStart)
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
            if let owner, activeConstraint == nil {
                session.stochastic = owner
                first = try sampleStochasticTarget(session, permit: permit)
            }
            session.tokenHistory = prompt + [first]
            let firstTokenSec = Date().timeIntervalSince(started)
            let prefillSec = computePrefillSec ?? firstTokenSec
            var emittedFirst = false
            var cancelledOnFirst = false
            if maxNew > 0, !stopIds.contains(first), let onToken {
                emittedFirst = true
                if onToken(first) == false {
                    cancelledOnFirst = true
                }
            }
            if needsFullPromptRetain {
                let savedStochastic = session.stochastic
                session.stochastic = nil
                defer { session.stochastic = savedStochastic }
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
                if owner != nil {
                    let round = try self.prefixPagePressure(store, session: session) {
                        try self.prepareStochasticRound(session, maxNew: remaining, stopIDs: stopIds, permit: permit)
                    }
                    try self.commitStochasticVerification(session, round: round, drain: false, permit: permit)
                    if onToken != nil {
                        session.lastCommitCB?.waitUntilCompleted()
                        session.lastCommitCB = nil
                        guard !session.stateFailed else {
                            throw QWEngineDriverError.state("commit GPU failure during streaming speculative generation")
                        }
                    }
                    return round.decision
                } else {
                    let round = try self.prefixPagePressure(store, session: session) {
                        try self.prepareGreedyRound(session, maxNew: remaining, stopIDs: stopIds, permit: permit)
                    }
                    try self.commitGreedyVerification(session, round: round, drain: false, permit: permit)
                    if onToken != nil {
                        session.lastCommitCB?.waitUntilCompleted()
                        session.lastCommitCB = nil
                        guard !session.stateFailed else {
                            throw QWEngineDriverError.state("commit GPU failure during streaming speculative generation")
                        }
                    }
                    return round.decision
                }
            } : nil
            let output = try QWGreedyGenerationLoop.run(vocab: geometry.vocab, first: first, maxNew: maxNew, stopIDs: stopIds,
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
                canSpeculate: { self.activeConstraint?.constraint.isUnconstrainedNow ?? true })
            session.lastCommitCB?.waitUntilCompleted()
            session.lastCommitCB = nil
            guard !session.stateFailed else {
                throw QWEngineDriverError.state("commit GPU failure during generation")
            }
            lastGeneration = QWGenerationStats(promptTokens: prompt.count, generatedTokens: output.tokens.count,
                prefillSec: prefillSec, decodeSec: Date().timeIntervalSince(decoding),
                firstTokenSec: firstTokenSec, stepMs: output.stepMs)
            if speculative {
                lastExperimentalGeneration = QWExperimentalGenerationStats(
                    promptTokens: prompt.count, generatedTokens: output.tokens.count,
                    prefillSec: prefillSec, decodeSec: Date().timeIntervalSince(decoding),
                    roundMs: output.stepMs, roundTokenCounts: output.stepTokenCounts)
            } else {
                lastExperimentalGeneration = nil
            }
            lastPrefixCacheStats = QWPrefixCacheStats(
                hit: loaded.matched > 0,
                matchedTokens: loaded.matched,
                chargedBytes: store.chargedBytes,
                evictions: store.evictions,
                skippedOversize: skippedOversize,
                tier: loaded.matched > 0 ? loaded.tier : nil
            )
            try inspectCompleted?(session)
            return output.tokens
        }
    }

    /// Prefills `tokens`, writes a standalone `.qwcache` file at `outputURL`, and optionally pins it in warm RAM.
    public func buildStandalonePrefixCache(
        tokens: [Int],
        outputURL: URL,
        pinAsSharedBase: Bool = true
    ) throws -> Int {
        try requestGate.withRequest(flight: flight) { permit in
            guard !tokens.isEmpty, tokens.count <= ropeRows,
                  tokens.allSatisfy({ (0..<geometry.vocab).contains($0) }) else {
                throw QWEngineDriverError.state("invalid standalone prefix tokens")
            }
            let mode = gdnPrefillMode
            let session = try admitSession(promptTokens: tokens.count)
            defer { finishSession(session) }
            let first = try prefill(session, tokens: tokens, mode: mode, permit: permit)
            session.lastCommitCB?.waitUntilCompleted()
            session.lastCommitCB = nil
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
                let keyBytes = try store.keyBytes(tokens: tokens, mode: policyKey)
                store.ensureByteLimit(atLeast: cached.memoryBytes + keyBytes + (1024 << 20))
                _ = try? store.insert(
                    tokens: tokens,
                    mode: policyKey,
                    value: cached,
                    payloadBytes: cached.memoryBytes,
                    allowsSuffix: mode != .naive,
                    isSharedBase: true
                )
                lsaOperator?.setPinnedBaseState(prefix.state.mlaIndexerData, position: prefix.state.position)
            }
            return written.totalBytes
        }
    }

    /// Mounts a standalone `.qwcache` file from `fileURL` into warm RAM as a pinned shared base prefix.
    public func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) {
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate()
        if prefixStore == nil {
            try configurePrefixCache(config: QWCacheConfig(warmByteLimit: 512 << 20, ssdByteLimit: 0, entryLimit: 32))
        }
        guard let store = prefixStore else {
            throw QWEngineDriverError.state("prefix store unavailable for mounting memory pack")
        }
        let read = try QWCacheDiskManager.readStandaloneFile(
            at: fileURL,
            kvCache: kvCache,
            mtpCache: mtpCache
        )
        lsaOperator?.setPinnedBaseState(nil, position: 0)
        try validateMLAIndexerState(read.state.mlaIndexerData, position: read.state.position)
        let sharedPrefix = QWEngineSharedPrefix(state: read.state, trunk: read.trunk, mtp: read.mtp)
        let cached = QWEngineCachedPrefix(prefix: sharedPrefix, nextToken: read.nextToken, logits: read.logits)
        let policyKey = prefixPolicyKey(mode: gdnPrefillMode)
        let keyBytes = try store.keyBytes(tokens: read.metadata.tokens, mode: policyKey)
        let requiredBytes = cached.memoryBytes + keyBytes + (1024 << 20)
        store.ensureByteLimit(atLeast: requiredBytes)
        let inserted = try store.insert(
            tokens: read.metadata.tokens,
            mode: policyKey,
            value: cached,
            payloadBytes: cached.memoryBytes,
            allowsSuffix: true,
            isSharedBase: true
        )
        guard inserted else {
            throw QWEngineDriverError.state("failed to pin mounted prefix cache into warm store")
        }
        lsaOperator?.setPinnedBaseState(read.state.mlaIndexerData, position: read.state.position)
        return (read.metadata.tokens, cached.memoryBytes)
    }

    /// Unmounts all pinned shared-base prefixes from the warm prefix store.
    public func unmountSharedBasePrefixes() {
        guard flight.try() else { return }
        defer { flight.unlock() }
        lsaOperator?.setPinnedBaseState(nil, position: 0)
        prefixStore?.removeSharedBaseEntries()
    }
}
