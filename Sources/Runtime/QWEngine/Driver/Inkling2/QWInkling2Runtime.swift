// QWInkling2Runtime.swift — Inference runtime, admission, cache & sampling for Inkling-2.
import Foundation
import Metal
import QWAttn
import QWCore
import QWHAL
import QWHead
import QWKV
import QWQuant

final class QWInkling2Runtime: @unchecked Sendable {
    let descriptor: QWModelDescriptor
    let ink2: QWModelDescriptor.Inkling2Config
    let geometry: QWModelGeometry
    let plan: QWEnginePlan
    let layerKernels: [any QWLayerKernel]
    let weightFormat: QWWeightFormat
    let weightFormatDeclineReason: String?
    let hal: QWHal
    let weights: QWInkling2Weights
    let session: QWInkling2Session
    let waveSlots: [Ink2SlotState]
    let forwardPass: QWInkling2Forward
    let dspark: QWInkling2DSpark?
    let lsaOperator: QWLSAOperator?
    let requestedMtpDrafts: Int
    let activeMtpDrafts: Int
    let configuredPrefillChunkSize: Int
    let configuredRelaxedMargin: Double
    var configuredRelaxedMinProb: Double
    let configuredMoESpecBudget: Int
    let prefixCache: QWInkling2PrefixCache
    let capacity: Int
    let capacityPlan: QWCapacityPlan
    let capacityLogMessage: String
    let kvMode: QWKVMode
    let rawKVMode: String
    let prefillProfiler: QWInkling2Profiler?
    let decodeProfiler: QWInkling2Profiler?

    let execLock = NSLock()
    let stateLock = NSLock()
    private var _onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)?
    var _lastResult: QWGenerationResult?

    var hasVisionWeights: Bool { weights.hasVisionWeights }
    var hasPrefixCacheTier: Bool { prefixCache.isEnabled }
    var activeCapacity: Int { capacity }
    var maxBatchSlots: Int { waveSlots.count }
    var companionResidentBytes: Int {
        waveSlots.dropFirst().reduce(0) { $0 + $1.residentStateBytes }
    }
    var arenaBytes: Int {
        weights.totalBytes + (dspark?.totalBytes ?? 0) + (lsaOperator?.region.buffer.length ?? 0)
            + session.residentStateBytes + companionResidentBytes + session.scratchBytes
    }
    var weightUsedBytes: Int { weights.totalBytes + (dspark?.totalBytes ?? 0) }
    var weightReservedBytes: Int { weights.totalBytes + (dspark?.totalBytes ?? 0) }
    var steadyStateAllocations: Int { 0 }

    var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)? {
        get { stateLock.withLock { _onCacheEvent } }
        set { stateLock.withLock { _onCacheEvent = newValue } }
    }

    var lastResult: QWGenerationResult? {
        get { stateLock.withLock { _lastResult } }
        set { stateLock.withLock { _lastResult = newValue } }
    }

    init(
        packPath: String,
        maxTokens: Int,
        kvMode: QWKVMode,
        rawKVMode: String,
        osReserveGiB: Double?,
        prewarmDoors: [String],
        prefixCacheWarmByteLimit: Int,
        maxConcurrentSessions: Int,
        mtpDrafts: Int = 0,
        loadVision: Bool = false,
        prefillChunkSize: Int = QWInkling2Session.maxPrefillChunk,
        lsaConfig: QWLSAConfig = QWLSAConfig(),
        mtpRelaxedMargin: Double = 0.0,
        mtpRelaxedMinProb: Double = 0.0,
        moeSpecBudget: Int = 0
    ) throws {
        self.configuredRelaxedMargin = mtpRelaxedMargin
        self.configuredRelaxedMinProb = mtpRelaxedMinProb
        self.configuredMoESpecBudget = moeSpecBudget
        let expanded = (packPath as NSString).expandingTildeInPath
        let location = try QWModelLocation(url: URL(fileURLWithPath: expanded))
        let configURL = location.assets.appendingPathComponent("config.json")
        let desc = try QWModelDescriptor.decode(configJSON: Data(contentsOf: configURL))
        guard let ink2 = desc.inkling2Config else {
            throw QWInkling2Error.missingConfig("descriptor has no inkling2Config")
        }
        let kvModeRaw: UInt32
        switch rawKVMode {
        case "f16": kvModeRaw = 0
        case "q8": kvModeRaw = 1
        case "q4": kvModeRaw = 2
        default: throw QWEngineDriverError.unsupportedKVMode(rawKVMode)
        }

        self.descriptor = desc
        self.ink2 = ink2
        self.kvMode = kvMode
        self.rawKVMode = rawKVMode
        self.geometry = QWModelGeometry(descriptor: desc)

        let wDir = location.weights.hasDirectoryPath
            ? location.weights
            : location.weights.deletingLastPathComponent()
        let dsparkCandidates = [
            wDir.appendingPathComponent("dspark.qwp"),
            location.assets.appendingPathComponent("dspark.qwp"),
            wDir.appendingPathComponent("mtp.safetensors"),
            location.assets.appendingPathComponent("mtp.safetensors")
        ]
        let resolvedMtpURL = dsparkCandidates.first { FileManager.default.fileExists(atPath: $0.path) }
        let clampedRequestedDrafts = max(0, min(15, mtpDrafts))
        self.requestedMtpDrafts = clampedRequestedDrafts
        let dsparkCfg = (clampedRequestedDrafts > 0 && resolvedMtpURL != nil) ? ink2.dspark : nil
        let resolvedActiveDrafts = dsparkCfg.map { min(clampedRequestedDrafts, max(0, $0.blockSize - 1)) } ?? 0
        self.activeMtpDrafts = resolvedActiveDrafts

        let isQWP = location.kind == .qwp1
        let qwpPack: QWQuant.Pack? = isQWP ? try QWQuant.Pack(url: location.weights, descriptor: desc) : nil
        let root = QWModelRoot(url: URL(fileURLWithPath: expanded))
        let formatChoice = try QWEngineDriver.weightFormatChoice(
            for: root,
            quantization: desc.quantization,
            geometry: geometry,
            mtpDrafts: resolvedActiveDrafts,
            pack: qwpPack
        )
        self.weightFormat = formatChoice.format
        self.weightFormatDeclineReason = formatChoice.declineReason

        let dsparkWeightEst: Int = (resolvedActiveDrafts > 0 && dsparkCfg != nil && resolvedMtpURL != nil)
            ? QWInkling2DSpark.estimateWeightBytes(
                mtpURL: resolvedMtpURL!, config: dsparkCfg!, unpaddedVocab: ink2.unpaddedVocabSize
            )
            : 0

        let baseModelBytes: Int
        if isQWP,
           let attrs = try? FileManager.default.attributesOfItem(atPath: location.weights.path),
           let fileSize = attrs[.size] as? Int {
            baseModelBytes = fileSize
        } else {
            baseModelBytes = try QWAdmissionPlan.modelResidentBytes(descriptor: desc)
        }
        let availBytes = QWCapacityBudget.availableBytes(osReserveGiB: osReserveGiB)
        let lsaHint = Self.resolveLSAAdmissionHint(packPath: expanded, descriptor: desc, lsaConfig: lsaConfig)
        let sizing = try Self.computeAdmissionSizing(
            descriptor: desc,
            ink2: ink2,
            baseModelBytes: baseModelBytes,
            dsparkWeightBytes: dsparkWeightEst,
            dsparkCfg: dsparkCfg,
            activeMtpDrafts: resolvedActiveDrafts,
            rawKVMode: rawKVMode,
            requestedTokens: maxTokens,
            prefillChunkSize: prefillChunkSize,
            lsaConfig: lsaConfig,
            lsaIndexDim: lsaHint.indexDim,
            lsaMultiHeadDirect: lsaHint.isMultiHeadDirect,
            availableBytes: availBytes
        )
        let capPlan = sizing.capacityPlan
        guard capPlan.fits else {
            throw QWEngineError.state(
                "Inkling-2 context \(maxTokens) exceeds memory budget (maxFitting=\(capPlan.maxFittingContext))")
        }
        let maxChunkPlan = max(32, min(8192, prefillChunkSize))
        let alignedCap = sizing.alignedCap
        let finalCap = sizing.finalCap
        self.capacity = finalCap
        self.capacityPlan = capPlan
        self.capacityLogMessage = try QWEngineBootstrap.admit(
            weightsBytes: sizing.weightsEst,
            fixedScratchBytes: sizing.fixedScratchEst,
            kvMode: rawKVMode,
            perTokenBytes: sizing.perTokBytes,
            maxTokens: alignedCap,
            totalPlannedBytes: sizing.totalPlannedBytes,
            availableBytes: sizing.availableBytes,
            osReserveGiB: osReserveGiB
        )

        self.plan = try QWEngineBootstrap.makePlan(
            geometry: self.geometry, packPath: expanded,
            prefillChunkRows: maxChunkPlan, kvStorageMode: rawKVMode
        )
        self.layerKernels = try QWLayerKernelRegistry.resolve(plan: self.plan)
        let hal = try QWHal(geometry: self.geometry, weightBasis: formatChoice.format.basis)
        self.hal = hal
        self.prefillProfiler = QWInkling2Profiler.makeIfEnabled(mode: .prefill, device: hal.device)
        self.decodeProfiler = QWInkling2Profiler.makeIfEnabled(mode: .decode, device: hal.device)

        let st: QWSafeTensorsContainer?
        let weights: QWInkling2Weights
        if let pack = qwpPack {
            st = nil
            weights = try QWInkling2Weights(
                qwp: pack, device: hal.device, descriptor: desc, ink2: ink2, loadVision: loadVision
            )
        } else {
            let openedSt = try QWSafeTensorsContainer(url: location.weights)
            st = openedSt
            weights = try QWInkling2Weights(
                device: hal.device, st: openedSt, descriptor: desc, ink2: ink2, loadVision: loadVision
            )
        }
        let session = try QWInkling2Session(
            device: hal.device, descriptor: desc, ink2: ink2, capacity: finalCap,
            kvModeRaw: kvModeRaw, enableSpeculative: true,
            enableQWPScratch: weights.isQuantizedQWP, prefillChunkSize: prefillChunkSize
        )
        self.configuredPrefillChunkSize = session.maxChunk
        let globalVisitIndices = desc.executionPlan.indices.filter { !ink2.isSlidingLayer(desc.executionPlan[$0]) }
        let effectiveLSAConfig = lsaConfig.resolvingAdaptiveBudget(forMaxContext: finalCap)
        let lsaOperator = try QWLSAOperator.makeIfEnabled(
            hal: hal, geometry: self.geometry, lsaConfig: effectiveLSAConfig, packPath: expanded,
            pack: qwpPack, safetensors: st, attnLayers: globalVisitIndices,
            maxContext: finalCap, maxPrefillChunk: session.maxChunk
        )
        let forwardPass = try QWInkling2Forward(
            hal: hal, weights: weights, session: session, lsaOperator: lsaOperator
        )
        let builtDSpark: QWInkling2DSpark?
        if resolvedActiveDrafts > 0, let dsCfg = dsparkCfg, let mtpURL = resolvedMtpURL {
            builtDSpark = try QWInkling2DSpark(
                hal: hal, mtpURL: mtpURL, descriptor: desc, ink2: ink2,
                config: dsCfg, weights: weights, capacity: finalCap, maxPrefillChunk: session.maxChunk
            )
        } else {
            builtDSpark = nil
        }
        let wavePlan = Self.planCompanionWaveSlots(
            sizing: sizing,
            maxConcurrentSessions: maxConcurrentSessions
        )
        var builtWaveSlots: [Ink2SlotState] = [session.primarySlotState]
        if wavePlan.targetSlots > 1 {
            for slotIdx in 1..<wavePlan.targetSlots {
                let comp = try QWInkling2Session.makeCompanionSlotState(
                    device: hal.device, descriptor: desc, ink2: ink2,
                    slotIndex: slotIdx, capacity: wavePlan.waveSlotCap,
                    maxChunk: session.maxChunk, kvModeRaw: kvModeRaw, kvMode: kvMode
                )
                builtWaveSlots.append(comp)
            }
        }
        self.weights = weights
        self.session = session
        self.waveSlots = builtWaveSlots
        self.forwardPass = forwardPass
        self.dspark = builtDSpark
        self.lsaOperator = lsaOperator

        hal.makeResident(weights.residentBuffers)
        hal.makeResident(session.allStateBuffers)
        for comp in builtWaveSlots.dropFirst() {
            hal.makeResident(comp.stateBuffers)
        }
        if let lsa = lsaOperator { hal.makeResident([lsa.region.buffer]) }
        if let ds = builtDSpark { hal.makeResident(ds.residentBuffers) }
        hal.adoptPendingResidency()
        hal.primeResidency()
        try QWEngineDriver.prewarmPipelines(hal: hal, doors: prewarmDoors)

        let companionWired = builtWaveSlots.dropFirst().reduce(0) { $0 + $1.residentStateBytes }
        let actualWired = weights.totalBytes + (builtDSpark?.totalBytes ?? 0)
            + (lsaOperator?.region.buffer.length ?? 0) + session.residentStateBytes
            + companionWired + session.scratchBytes
        let warmAdm = QWResidencyAdmission.clampWarmCache(
            requestedWarmBytes: prefixCacheWarmByteLimit,
            requestedSessions: maxConcurrentSessions,
            wiredEngineBytes: actualWired,
            osReserveGiB: osReserveGiB,
            measuredAfterWire: true,
            snapshot: QWLiveDeviceMemorySnapshotSource(maxWorkingSetBytes: hal.caps.maxWorkingSetBytes).capture()
        )
        self.prefixCache = QWInkling2PrefixCache(
            warmByteLimit: warmAdm.admittedWarmBytes,
            maxEntries: max(32, warmAdm.maxConcurrentSessions)
        )
        wireCacheCallbacks()
    }

    private func wireCacheCallbacks() {
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

    func buildReport() -> QWEngineReport {
        let dsparkLabel = dspark != nil ? ", dspark=\(activeMtpDrafts)" : ""
        let lsaLabel: String = {
            guard let lsa = lsaOperator else { return "" }
            let reuseTag = lsa.loopReusePlan.hasReuse ? ",loop_share" : ""
            return ", lsa=\(lsa.config.mode.rawValue)(k=\(lsa.config.alignedBudget),blk=\(lsa.config.alignedClusterBlock)\(reuseTag))"
        }()
        let lines = String(
            format: "runtime:                 inkling2-loop (visits=%d, kv=%@, capacity=%d%@%@)\n",
            session.executionPlan.count, rawKVMode, capacity, dsparkLabel, lsaLabel
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

    static let repetitionPenaltyWindow = 256

    static func recordGeneratedTokenForPenalty(
        _ token: Int,
        vocab: Int,
        recentGenerated: inout [Int],
        activeLoopTokens: inout [Int]
    ) {
        guard token >= 0, token < vocab else { return }
        if recentGenerated.capacity < repetitionPenaltyWindow + 1 {
            recentGenerated.reserveCapacity(repetitionPenaltyWindow + 1)
        }
        recentGenerated.append(token)
        if recentGenerated.count > repetitionPenaltyWindow {
            let drop = recentGenerated.count - repetitionPenaltyWindow
            let keep = recentGenerated.count - drop
            recentGenerated.withUnsafeMutableBufferPointer { buf in
                if let base = buf.baseAddress, keep > 0 {
                    memmove(base, base.advanced(by: drop), keep * MemoryLayout<Int>.stride)
                }
            }
            recentGenerated.removeLast(drop)
        }
        QWSampler.loopRepetitionTokens(in: recentGenerated, into: &activeLoopTokens)
    }

    func sampleToken(
        sampler: QWSampler,
        logitsRow: UnsafeBufferPointer<Float>,
        penaltyHistory: [Int],
        constraint: (any QWTokenConstraint)?,
        allowedScratch: inout [Int]
    ) throws -> Int {
        if let constraint, try constraint.writeAllowedTokens(into: &allowedScratch) {
            return try allowedScratch.withUnsafeBufferPointer { buf in
                try sampler.constrainedToken(logits: logitsRow, allowed: buf, repetitionHistory: penaltyHistory)
            }
        }
        if sampler.config.isGreedy {
            return try sampler.penalisedGreedyToken(logits: logitsRow, repetitionHistory: penaltyHistory)
        }
        return try sampler.sample(logits: logitsRow, repetitionHistory: penaltyHistory).token
    }

    func runGeneration(
        request: QWGenerationRequest,
        onToken: ((Int) -> Bool)?,
        onStopToken: ((Int) -> Void)?
    ) throws -> QWGenerationResult {
        try execLock.withLock {
            try runGenerationLocked(request: request, onToken: onToken, onStopToken: onStopToken)
        }
    }

    func runGenerationLocked(
        request: QWGenerationRequest,
        onToken: ((Int) -> Bool)?,
        onStopToken: ((Int) -> Void)?
    ) throws -> QWGenerationResult {
        guard !request.prompt.isEmpty else {
            return QWGenerationResult(tokens: [], stepMs: [], stepTokenCounts: [])
        }
        for tok in request.prompt where tok < 0 || tok >= descriptor.vocab {
            throw QWModuleError.invalidToken(tok)
        }
        guard request.prompt.count < capacity else {
            throw QWEngineError.state(
                "prompt length \(request.prompt.count) meets/exceeds capacity \(capacity)")
        }

        let clock = ContinuousClock()
            let prefillStart = clock.now
            let canUseCache = request.usePrefixCache && prefixCache.isEnabled
                && (request.visualSpans.isEmpty || request.cacheSalt != nil)
            var cacheHit = false
            var matchedTokens = 0
            var pos = 0
            var dsparkActive = dspark != nil

            var restoredFromCache = false
            var restoredExactFinalNorm: Data? = nil
            if canUseCache, request.prompt.count >= 1,
               let entry = prefixCache.matchLongestPrefix(
                   tokens: request.prompt, salt: request.cacheSalt,
                   limit: request.prompt.count, allowExact: true
               ) {
                let outcome = QWInkling2PrefixCache.restore(
                    entry: entry, into: session, dspark: dspark, lsaOperator: lsaOperator
                )
                if outcome.baseOk {
                    if dspark != nil && !outcome.dsparkOk { dsparkActive = false }
                    pos = entry.key.tokens.count
                    if pos == request.prompt.count { restoredExactFinalNorm = entry.finalNormRow }
                    cacheHit = true
                    matchedTokens = pos
                    restoredFromCache = true
                    onCacheEvent?(.hit(tier: "warm", matchedTokens: pos, totalTokens: request.prompt.count))
                }
            }
            if !restoredFromCache {
                session.resetConvStates()
                dspark?.reset()
                lsaOperator?.setPinnedBaseState(nil, position: 0)
                if let lsa = lsaOperator {
                    _ = lsa.restoreIndexerState(Data(), position: 0)
                    lsa.stateOwner = nil
                }
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

            let V = ink2.unpaddedVocabSize
            let logitsPtr = session.logitsF32Buf.contents().assumingMemoryBound(to: Float.self)
            let headTokPtr = session.headTokensBuf.contents().assumingMemoryBound(to: Int32.self)
            let sampler = QWSampler(config: request.sampling)
            let lpSampler = request.collectLogprobs ? QWSampler(config: request.sampling) : nil
            let topKCount = max(0, min(20, request.topLogprobsCount))
            let usePenalty = sampler.config.repetitionPenalty != 1.0
            let canUseGreedyHeadBase = sampler.config.isGreedy
                && request.constraint == nil && lpSampler == nil

            prefillProfiler?.reset()
            decodeProfiler?.reset()
            if let normRow = restoredExactFinalNorm {
                try forwardPass.runHeadFromFinalNorm(
                    finalNormRow: normRow, greedyHead: canUseGreedyHeadBase, profiler: prefillProfiler
                )
                request.onPrefillProgress?(pos, request.prompt.count)
            }
            while pos < request.prompt.count {
                let targetLimit: Int = if let r = retainTarget, r > pos, r < request.prompt.count {
                    r
                } else {
                    request.prompt.count
                }
                let rem = targetLimit - pos
                let denseTailTokens = 64
                let chunkSize: Int
                if rem > session.maxChunk {
                    let nChunks = (rem + session.maxChunk - 1) / session.maxChunk
                    chunkSize = min(session.maxChunk, max(64, ((rem + nChunks - 1) / nChunks + 63) & ~63))
                } else if let lsa = lsaOperator,
                          targetLimit == request.prompt.count,
                          request.prompt.count > lsa.config.denseBypassThreshold,
                          rem > denseTailTokens {
                    chunkSize = rem - denseTailTokens
                } else {
                    chunkSize = rem
                }
                let nextEnd = pos + chunkSize
                let chunk = Array(request.prompt[pos..<nextEnd])
                let isLast = (nextEnd == request.prompt.count)
                let forceDenseTail = isLast && lsaOperator != nil && chunk.count <= denseTailTokens
                let tapDS = dsparkActive ? dspark : nil
                try forwardPass.forward(
                    tokens: chunk,
                    posOffset: pos,
                    runHeadRows: isLast ? 1 : 0,
                    greedyHead: isLast && canUseGreedyHeadBase,
                    updateState: true,
                    captureTapBuf: tapDS?.tapBuf,
                    tapVisits: tapDS?.config.targetLayerIds ?? [],
                    visualSpans: request.visualSpans,
                    forceDenseAttention: forceDenseTail,
                    profiler: prefillProfiler
                )
                if let ds = tapDS {
                    try ds.ingestContextTaps(
                        count: chunk.count, posOffset: pos, rowStride: chunk.count, profiler: prefillProfiler
                    )
                }
                pos = nextEnd
                request.onPrefillProgress?(pos, request.prompt.count)
                if let r = retainTarget, pos == r {
                    let normRow = isLast
                        ? QWInkling2PrefixCache.captureFinalNormRow(session: session, lastChunkRows: chunk.count, hidden: descriptor.hidden)
                        : nil
                    let snap = QWInkling2PrefixCache.capture(
                        session: session,
                        dspark: tapDS,
                        lsaOperator: lsaOperator,
                        tokens: Array(request.prompt.prefix(r)),
                        salt: request.cacheSalt,
                        slidingWindow: ink2.slidingWindowSize,
                        finalNormRow: normRow
                    )
                    prefixCache.store(entry: snap)
                }
            }
            prefillProfiler?.printSummary()

            let prefillSec = elapsedSeconds(prefillStart.duration(to: clock.now))
            let effectiveMaxNew = min(max(0, request.maxNew), max(0, capacity - pos))
            var recentGeneratedForPenalty: [Int] = []
            var penaltyHistory: [Int] = []
            var allowedScratch: [Int] = []
            var generated: [Int] = []
            var stepMs: [Double] = []
            var stepCounts: [Int] = []
            var collectedLogprobs: [QWTokenLogprob] = []
            var stopTokenID: Int? = nil
            var specStats = SpeculativeStats()

            let decodeStart = clock.now
            var firstTokenSec = prefillSec
            let specMarg = resolveSpeculativeMargin(request: request)
            let strategy = request.speculative.mtpDraftStrategy
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            let isNgramOnly = (strategy == "ngram")
            let allowPLD = (strategy != "mtp")
            let targetDrafts = min(15, max(requestedMtpDrafts, request.speculative.mtpDrafts))
            let wantsSpec = request.speculative.enabled && targetDrafts > 0
                && !request.collectLogprobs && Self.canSpeculate(constraint: request.constraint)
            let useDSparkSpec = wantsSpec && request.constraint == nil && !isNgramOnly
                && dsparkActive && dspark != nil && activeMtpDrafts > 0
                && sampler.config.isGreedy && !usePenalty && specMarg.margin == 0

            if effectiveMaxNew > 0 {
                let tok0: Int
                if canUseGreedyHeadBase {
                    tok0 = Int(headTokPtr[0])
                } else {
                    let firstRow = UnsafeBufferPointer(start: logitsPtr, count: V)
                    tok0 = try sampleToken(
                        sampler: sampler, logitsRow: firstRow, penaltyHistory: penaltyHistory,
                        constraint: request.constraint, allowedScratch: &allowedScratch
                    )
                }
                firstTokenSec = elapsedSeconds(prefillStart.duration(to: clock.now))
                if request.stopIDs.contains(tok0) {
                    stopTokenID = tok0
                    onStopToken?(tok0)
                } else {
                    try request.constraint?.accept(tok0)
                    if let lpSampler {
                        let firstRow = UnsafeBufferPointer(start: logitsPtr, count: V)
                        let lp = lpSampler.computeTokenLogprob(logits: firstRow, chosenToken: tok0, topK: topKCount)
                        collectedLogprobs.append(lp)
                        request.onTokenLogprob?(lp)
                    }
                    generated.append(tok0)
                    if usePenalty {
                        Self.recordGeneratedTokenForPenalty(
                            tok0, vocab: V, recentGenerated: &recentGeneratedForPenalty, activeLoopTokens: &penaltyHistory
                        )
                    }
                    if onToken?(tok0) != false {
                        var decodeDone = generated.count >= effectiveMaxNew
                        if !decodeDone, useDSparkSpec, let ds = dspark {
                            let specRes = try runDSparkSpeculativeDecode(
                                dspark: ds,
                                requestedDrafts: targetDrafts,
                                allowPLD: allowPLD,
                                promptTokens: request.prompt,
                                pos: &pos,
                                effectiveMaxNew: effectiveMaxNew,
                                stopIDs: request.stopIDs,
                                generated: &generated,
                                stepMs: &stepMs,
                                stepCounts: &stepCounts,
                                stopTokenID: &stopTokenID,
                                onToken: onToken,
                                onStopToken: onStopToken,
                                profiler: decodeProfiler
                            )
                            specStats = specRes.stats
                            decodeDone = specRes.finished
                        } else if !decodeDone, wantsSpec {
                            let activeDS = (!isNgramOnly && dsparkActive) ? dspark : nil
                            let reqDrafts = targetDrafts
                            if reqDrafts > 0 {
                                let specRes = try runFlexibleSpeculativeDecode(
                                    dspark: activeDS, requestedDrafts: reqDrafts, allowPLD: allowPLD,
                                    sampler: sampler, constraint: request.constraint, allowedScratch: &allowedScratch,
                                    effectiveMargin: specMarg.margin, minLogProb: specMarg.minLogProb,
                                    promptTokens: request.prompt, pos: &pos, effectiveMaxNew: effectiveMaxNew,
                                    stopIDs: request.stopIDs, generated: &generated,
                                    recentGeneratedForPenalty: &recentGeneratedForPenalty,
                                    penaltyHistory: &penaltyHistory,
                                    stepMs: &stepMs, stepCounts: &stepCounts, stopTokenID: &stopTokenID,
                                    onToken: onToken, onStopToken: onStopToken, profiler: decodeProfiler
                                )
                                specStats = specRes.stats
                                decodeDone = specRes.finished
                            }
                        }
                        while !decodeDone && generated.count < effectiveMaxNew {
                            let stepStart = clock.now
                            let lastTok = generated[generated.count - 1]
                            let stepGreedyHead = canUseGreedyHeadBase && (!usePenalty || penaltyHistory.isEmpty)
                            try forwardPass.forward(
                                tokens: [lastTok], posOffset: pos, runHeadRows: 1,
                                greedyHead: stepGreedyHead,
                                updateState: true, profiler: decodeProfiler
                            )
                            pos += 1
                            let tok: Int
                            if stepGreedyHead {
                                tok = Int(headTokPtr[0])
                            } else {
                                let row = UnsafeBufferPointer(start: logitsPtr, count: V)
                                tok = try sampleToken(
                                    sampler: sampler, logitsRow: row, penaltyHistory: penaltyHistory,
                                    constraint: request.constraint, allowedScratch: &allowedScratch
                                )
                            }
                            let elapsedMs = elapsedSeconds(stepStart.duration(to: clock.now)) * 1000.0
                            stepMs.append(elapsedMs)
                            if request.stopIDs.contains(tok) {
                                stepCounts.append(0)
                                stopTokenID = tok
                                onStopToken?(tok)
                                break
                            }
                            try request.constraint?.accept(tok)
                            if let lpSampler {
                                let row = UnsafeBufferPointer(start: logitsPtr, count: V)
                                let lp = lpSampler.computeTokenLogprob(logits: row, chosenToken: tok, topK: topKCount)
                                collectedLogprobs.append(lp)
                                request.onTokenLogprob?(lp)
                            }
                            stepCounts.append(1)
                            generated.append(tok)
                            if usePenalty {
                                Self.recordGeneratedTokenForPenalty(
                                    tok, vocab: V, recentGenerated: &recentGeneratedForPenalty,
                                    activeLoopTokens: &penaltyHistory
                                )
                            }
                            if onToken?(tok) == false { break }
                        }
                    }
                }
            }

            let decodeSec = elapsedSeconds(decodeStart.duration(to: clock.now))
            decodeProfiler?.printSummary()
            let cacheAnswer: QWCacheAnswer? = canUseCache
                ? QWCacheAnswer(hit: cacheHit, matchedTokens: matchedTokens,
                                totalTokens: request.prompt.count, tier: cacheHit ? "warm" : nil)
                : nil
            let result = QWGenerationResult(
                tokens: generated,
                stepMs: stepMs,
                stepTokenCounts: stepCounts,
                prefillSeconds: prefillSec,
                firstTokenSeconds: firstTokenSec,
                decodeSeconds: decodeSec,
                stopTokenID: stopTokenID,
                cache: cacheAnswer,
                speculativeSummary: specStats.summary,
                tokenLogprobs: collectedLogprobs
            )
        stateLock.withLock { _lastResult = result }
        return result
    }

    func computeEchoPromptLogprobs(
        promptTokens: [Int],
        sampling: QWSamplerConfig,
        topLogprobsCount: Int,
        shouldStop: (@Sendable () -> Bool)?
    ) throws -> [QWTokenLogprob] {
        guard promptTokens.count >= 2 else { return [] }
        for tok in promptTokens where tok < 0 || tok >= descriptor.vocab {
            throw QWEngineDriverError.badToken(tok)
        }
        guard promptTokens.count <= capacity else {
            throw QWEngineError.state("echo prompt length \(promptTokens.count) exceeds capacity \(capacity)")
        }
        return try execLock.withLock {
            session.resetConvStates()
            let sampler = QWSampler(config: sampling)
            let topK = max(0, min(20, topLogprobsCount))
            let V = ink2.unpaddedVocabSize
            let nEval = promptTokens.count - 1
            var out: [QWTokenLogprob] = []
            out.reserveCapacity(nEval)
            let logitsPtr = session.logitsF32Buf.contents().assumingMemoryBound(to: Float.self)
            var pos = 0
            while pos < nEval {
                if shouldStop?() == true { break }
                let end = min(nEval, pos + 16)
                let chunk = Array(promptTokens[pos..<end])
                try forwardPass.forward(tokens: chunk, posOffset: pos, runHeadRows: chunk.count, updateState: true)
                for row in 0..<chunk.count {
                    let rowPtr = logitsPtr.advanced(by: row * V)
                    let buf = UnsafeBufferPointer(start: rowPtr, count: V)
                    out.append(sampler.computeTokenLogprob(
                        logits: buf, chosenToken: promptTokens[pos + row + 1], topK: topK
                    ))
                }
                pos = end
            }
            return out
        }
    }
}
