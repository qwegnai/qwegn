// QWInkling2Runtime.swift — Inference runtime, admission, cache & sampling for Inkling-2.
import Foundation
import Metal
import QWAttn
import QWCore
import QWHAL
import QWHead
import QWKV
import QWMTP
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
    let dflash: QWDFlashDrafter?
    let dflashTapBuf: (any MTLBuffer)?
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
    var _onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)?
    var _gpuTurnArbitrator: QWGPUTurnArbitrator?
    var _lastResult: QWGenerationResult?

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
        mtpBackend: QWMTPBackend = .standard,
        dflashDraftPackPath: String? = nil,
        dflashSelectorTopK: Int? = nil,
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
        let dflashSel = try Self.resolveDFlashStartup(
            descriptor: desc, packPath: expanded, mtpBackend: mtpBackend,
            requestedDrafts: clampedRequestedDrafts, dflashDraftPackPath: dflashDraftPackPath,
            dflashSelectorTopK: dflashSelectorTopK, prefillChunkSize: prefillChunkSize
        )
        let dsparkCfg = (dflashSel == nil && clampedRequestedDrafts > 0 && resolvedMtpURL != nil) ? ink2.dspark : nil
        let resolvedActiveDrafts = dflashSel.map { min(clampedRequestedDrafts, max(0, $0.config.blockSize - 1)) }
            ?? dsparkCfg.map { min(clampedRequestedDrafts, max(0, $0.blockSize - 1)) }
            ?? 0
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

        let dsparkWeightEst: Int = dflashSel?.weightBytes
            ?? ((resolvedActiveDrafts > 0 && dsparkCfg != nil && resolvedMtpURL != nil)
                ? QWInkling2DSpark.estimateWeightBytes(
                    mtpURL: resolvedMtpURL!, config: dsparkCfg!, unpaddedVocab: ink2.unpaddedVocabSize)
                : 0)

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
            extraFixedScratchBytes: dflashSel?.fixedBytes ?? 0,
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
                qwp: pack, device: hal.device, descriptor: desc, ink2: ink2
            )
        } else {
            let openedSt = try QWSafeTensorsContainer(url: location.weights)
            st = openedSt
            weights = try QWInkling2Weights(
                device: hal.device, st: openedSt, descriptor: desc, ink2: ink2
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
        let builtDFlash: QWDFlashDrafter?
        let builtDFlashTapBuf: (any MTLBuffer)?
        if let dfSel = dflashSel, resolvedActiveDrafts > 0 {
            builtDSpark = nil
            let headRotated: Bool = {
                if case .q4g(let hw) = weights.lmHeadStorage { return hw.view.isRotated }
                return false
            }()
            builtDFlash = try QWDFlashDrafter(
                geometry: self.geometry, hal: hal, packPath: dfSel.path,
                drafts: resolvedActiveDrafts, selectorTopK: dflashSelectorTopK,
                sessionSlotCount: 1, headRotated: headRotated
            )
            let tapBytes = max(256, dfSel.config.targetLayerIds.count * session.maxChunk * desc.hidden * 2)
            guard let tBuf = hal.device.makeBuffer(length: tapBytes, options: .storageModeShared) else {
                throw QWInkling2Error.allocationFailed("dflash tapBuf \(tapBytes) bytes")
            }
            builtDFlashTapBuf = tBuf
        } else if resolvedActiveDrafts > 0, let dsCfg = dsparkCfg, let mtpURL = resolvedMtpURL {
            builtDSpark = try QWInkling2DSpark(
                hal: hal, mtpURL: mtpURL, descriptor: desc, ink2: ink2,
                config: dsCfg, weights: weights, capacity: finalCap, maxPrefillChunk: session.maxChunk
            )
            builtDFlash = nil
            builtDFlashTapBuf = nil
        } else {
            builtDSpark = nil
            builtDFlash = nil
            builtDFlashTapBuf = nil
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
        self.dflash = builtDFlash
        self.dflashTapBuf = builtDFlashTapBuf
        self.lsaOperator = lsaOperator

        hal.makeResident(weights.residentBuffers)
        hal.makeResident(session.allStateBuffers)
        for comp in builtWaveSlots.dropFirst() {
            hal.makeResident(comp.stateBuffers)
        }
        if let lsa = lsaOperator { hal.makeResident([lsa.region.buffer]) }
        if let ds = builtDSpark { hal.makeResident(ds.residentBuffers) }
        if let df = builtDFlash { hal.makeResident(df.residentBuffers) }
        if let dt = builtDFlashTapBuf { hal.makeResident([dt]) }
        hal.adoptPendingResidency()
        hal.primeResidency()
        try QWEngineDriver.prewarmPipelines(hal: hal, doors: prewarmDoors)

        let companionWired = builtWaveSlots.dropFirst().reduce(0) { $0 + $1.residentStateBytes }
        let actualWired = weights.totalBytes + (builtDSpark?.totalBytes ?? 0)
            + (builtDFlash?.totalBytes ?? 0) + (builtDFlashTapBuf?.length ?? 0)
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
        let prefix = preparePrefixState(for: request, slot: session.primarySlotState, allowAuxOperators: true)
        var pos = prefix.pos
        let dsparkActive = prefix.dsparkActive
        let dflashActive = prefix.dflashActive
        let retainTarget = prefix.retainTarget

        let V = ink2.unpaddedVocabSize
        let logitsPtr = session.logitsF32Buf.contents().assumingMemoryBound(to: Float.self)
        let headTokPtr = session.headTokensBuf.contents().assumingMemoryBound(to: Int32.self)
        let sampler = QWSampler(config: request.sampling)
        let lpSampler = request.collectLogprobs ? QWSampler(config: request.sampling) : nil
        let topKCount = max(0, min(20, request.topLogprobsCount))
        let usePenalty = sampler.config.repetitionPenalty != 1.0
        let isUnconstrained = (request.constraint == nil || request.constraint?.isUnconstrainedNow == true)
        let canUseGreedyHeadBase = sampler.config.isGreedy
            && isUnconstrained && lpSampler == nil

        prefillProfiler?.reset()
        decodeProfiler?.reset()
        if let normRow = prefix.restoredExactFinalNorm {
            try withGPUTurn {
                try forwardPass.runHeadFromFinalNorm(
                    finalNormRow: normRow, greedyHead: canUseGreedyHeadBase, profiler: prefillProfiler
                )
            }
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
            let tapDF = dflashActive ? dflash : nil
            let tapBuf = tapDS?.tapBuf ?? (tapDF != nil ? dflashTapBuf : nil)
            let tapVisits = tapDS?.config.targetLayerIds ?? tapDF?.config.targetLayerIds ?? []
            try withGPUTurn {
                try forwardPass.forward(
                    tokens: chunk, posOffset: pos, runHeadRows: isLast ? 1 : 0,
                    greedyHead: isLast && canUseGreedyHeadBase, updateState: true,
                    captureTapBuf: tapBuf, tapVisits: tapVisits, visualSpans: request.visualSpans,
                    forceDenseAttention: forceDenseTail, profiler: prefillProfiler
                )
                if let ds = tapDS {
                    try ds.ingestContextTaps(
                        count: chunk.count, posOffset: pos, rowStride: chunk.count, profiler: prefillProfiler
                    )
                } else if let df = tapDF, let dtBuf = dflashTapBuf,
                          let dfWindowEnd = Optional(retainTarget.map { pos < $0 ? $0 : request.prompt.count } ?? request.prompt.count),
                          let capSlice = QWDFlashConfig.prefillCaptureSlice(
                              startPos: pos, rows: chunk.count, windowEnd: dfWindowEnd,
                              maxCacheSize: df.config.maxCacheSize
                          ) {
                    try ingestDFlashPrefillSlice(df: df, tapBuf: dtBuf, chunkRows: chunk.count, slice: capSlice)
                }
            }
            pos = nextEnd
            request.onPrefillProgress?(pos, request.prompt.count)
            if let r = retainTarget, pos == r {
                storePrefixSnapshot(
                    slot: session.primarySlotState,
                    tokens: Array(request.prompt.prefix(r)),
                    salt: request.cacheSalt,
                    dspark: tapDS,
                    dflash: tapDF,
                    lsaOperator: lsaOperator,
                    lastChunkRowsForFinalNorm: isLast ? chunk.count : nil
                )
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
        let useDFlashSpec = wantsSpec && isUnconstrained && !isNgramOnly
            && dflashActive && dflash != nil && dflashTapBuf != nil && activeMtpDrafts > 0
            && specMarg.margin == 0
        let useDSparkSpec = !useDFlashSpec && wantsSpec && isUnconstrained && !isNgramOnly
            && dsparkActive && dspark != nil && activeMtpDrafts > 0
            && sampler.config.isGreedy && !usePenalty && specMarg.margin == 0

        if effectiveMaxNew > 0 {
            let tok0 = try sampleRowToken(
                row: 0, vocab: V, useGreedyHead: canUseGreedyHeadBase,
                logitsPtr: logitsPtr, headTokPtr: headTokPtr,
                sampler: sampler, penaltyHistory: penaltyHistory,
                constraint: request.constraint, allowedScratch: &allowedScratch
            )
            firstTokenSec = elapsedSeconds(prefillStart.duration(to: clock.now))
            let emit0 = try consumeEmittedToken(
                tok0, row: 0, vocab: V, logitsPtr: logitsPtr,
                stopIDs: request.stopIDs, constraint: request.constraint,
                lpSampler: lpSampler, topKCount: topKCount, usePenalty: usePenalty,
                recentGeneratedForPenalty: &recentGeneratedForPenalty,
                penaltyHistory: &penaltyHistory, generated: &generated,
                collectedLogprobs: &collectedLogprobs, stopTokenID: &stopTokenID,
                onTokenLogprob: request.onTokenLogprob, onStopToken: onStopToken,
                onToken: onToken
            )
            if emit0 == .continued {
                var decodeDone = generated.count >= effectiveMaxNew
                if !decodeDone, useDFlashSpec, let df = dflash, let tapBuf = dflashTapBuf {
                    let specRes = try runDFlashSpeculativeDecode(
                        dflash: df, tapBuf: tapBuf, requestedDrafts: targetDrafts,
                        allowPLD: allowPLD, sampler: sampler,
                        promptTokens: request.prompt, pos: &pos, effectiveMaxNew: effectiveMaxNew,
                        stopIDs: request.stopIDs, generated: &generated,
                        recentGeneratedForPenalty: &recentGeneratedForPenalty,
                        penaltyHistory: &penaltyHistory, stepMs: &stepMs, stepCounts: &stepCounts,
                        stopTokenID: &stopTokenID, onToken: onToken, onStopToken: onStopToken,
                        profiler: decodeProfiler
                    )
                    specStats = specRes.stats
                    decodeDone = specRes.finished
                } else if !decodeDone, useDSparkSpec, let ds = dspark {
                    let specRes = try runDSparkSpeculativeDecode(
                        dspark: ds, requestedDrafts: targetDrafts, allowPLD: allowPLD,
                        promptTokens: request.prompt, pos: &pos, effectiveMaxNew: effectiveMaxNew,
                        stopIDs: request.stopIDs, generated: &generated,
                        stepMs: &stepMs, stepCounts: &stepCounts, stopTokenID: &stopTokenID,
                        onToken: onToken, onStopToken: onStopToken, profiler: decodeProfiler
                    )
                    specStats = specRes.stats
                    decodeDone = specRes.finished
                } else if !decodeDone, wantsSpec, targetDrafts > 0 {
                    let activeDS = (!isNgramOnly && dsparkActive) ? dspark : nil
                    let specRes = try runFlexibleSpeculativeDecode(
                        dspark: activeDS, requestedDrafts: targetDrafts, allowPLD: allowPLD,
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
                while !decodeDone && generated.count < effectiveMaxNew {
                    let stepStart = clock.now
                    let lastTok = generated[generated.count - 1]
                    let stepGreedyHead = canUseGreedyHeadBase && (!usePenalty || penaltyHistory.isEmpty)
                    try withGPUTurn {
                        try forwardPass.forward(
                            tokens: [lastTok], posOffset: pos, runHeadRows: 1,
                            greedyHead: stepGreedyHead,
                            updateState: true, profiler: decodeProfiler
                        )
                    }
                    pos += 1
                    let tok = try sampleRowToken(
                        row: 0, vocab: V, useGreedyHead: stepGreedyHead,
                        logitsPtr: logitsPtr, headTokPtr: headTokPtr,
                        sampler: sampler, penaltyHistory: penaltyHistory,
                        constraint: request.constraint, allowedScratch: &allowedScratch
                    )
                    stepMs.append(elapsedSeconds(stepStart.duration(to: clock.now)) * 1000.0)
                    let outcome = try consumeEmittedToken(
                        tok, row: 0, vocab: V, logitsPtr: logitsPtr,
                        stopIDs: request.stopIDs, constraint: request.constraint,
                        lpSampler: lpSampler, topKCount: topKCount, usePenalty: usePenalty,
                        recentGeneratedForPenalty: &recentGeneratedForPenalty,
                        penaltyHistory: &penaltyHistory, generated: &generated,
                        collectedLogprobs: &collectedLogprobs, stopTokenID: &stopTokenID,
                        onTokenLogprob: request.onTokenLogprob, onStopToken: onStopToken,
                        onToken: onToken
                    )
                    if outcome == .stopped {
                        stepCounts.append(0)
                        break
                    }
                    stepCounts.append(1)
                    if outcome == .cancelled { break }
                }
            }
        }

        let decodeSec = elapsedSeconds(decodeStart.duration(to: clock.now))
        decodeProfiler?.printSummary()
        let result = QWGenerationResult(
            tokens: generated,
            stepMs: stepMs,
            stepTokenCounts: stepCounts,
            prefillSeconds: prefillSec,
            firstTokenSeconds: firstTokenSec,
            decodeSeconds: decodeSec,
            stopTokenID: stopTokenID,
            cache: prefix.cacheAnswer,
            speculativeSummary: specStats.summary,
            tokenLogprobs: collectedLogprobs
        )
        stateLock.withLock { _lastResult = result }
        return result
    }
}
