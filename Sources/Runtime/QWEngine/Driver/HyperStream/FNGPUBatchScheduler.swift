// FNGPUBatchScheduler.swift — Multi-session continuous batching scheduler for QWHyperStreamRuntime.
// Runs single-stream requests on Slot 0 with full capacity, prefix cache, and speculative decoding,
// and dynamically coalesces 2..4 concurrent requests into lockstep batched decode across 4 slots.

import Foundation
import QWAlloc
import QWAttn
import QWCore
import QWHAL
import QWHead
import QWHyperPLE

/// Per-slot state for multi-session continuous batching on `QWHyperStreamRuntime`.
public struct FNBatchSlotState {
    public var gdnStates: [[QWTensorValue]]
    public var gdnStateParity: [Int]
    public var gdnConvState: [[QWTensorValue]]
    public var gdnConvParity: [Int]
    public var pleHistory: [[QWTensorValue]]
    public var pleParity: [Int]
    public var highWaterPos: Int
}

/// Output metrics of one completed generation job from `FNGPUBatchScheduler`.
public struct FNBatchJobOutput: Sendable {
    public let tokens: [Int]
    public let stepMs: [Double]
    public let stepTokenCounts: [Int]
    public let prefillSeconds: Double
    public let firstTokenSeconds: Double
    public let decodeSeconds: Double
    public let prefillGPUSeconds: Double
    public let stopTokenID: Int?
    public let prefixCacheHit: Bool
    public let prefixCacheMatchedTokens: Int
    public let prefixCacheTier: FNPrefixCacheTier?
    public let specProposals: Int
    public let specAccepted: Int
    public let tokenLogprobs: [QWTokenLogprob]
}

/// One generation request submitted to `FNGPUBatchScheduler`.
public final class FNBatchJob: @unchecked Sendable {
    public let prompt: [Int]
    public let maxNew: Int
    public let stopIDs: Set<Int>
    public let sampling: QWSamplerConfig
    public let useSpec: Bool
    public let draftMode: QWDraftStrategy
    public let collectLogprobs: Bool
    public let topLogprobsCount: Int
    public let onToken: ((Int) -> Bool)?
    public let onStopToken: ((Int) -> Void)?
    public let onTokenLogprob: ((QWTokenLogprob) -> Void)?

    var session: FNGPUSessionData
    var output = QWBatchSlotOutputState()
    var emittedLogprobsCount: Int = 0
    var generatedTokens: [Int] { get { output.generated } set { output.generated = newValue } }
    var stepMs: [Double] { get { output.stepMs } set { output.stepMs = newValue } }
    var stepTokenCounts: [Int] { get { output.stepTokenCounts } set { output.stepTokenCounts = newValue } }
    var prefillSeconds: Double { get { output.prefillSeconds } set { output.prefillSeconds = newValue } }
    var decodeStart: Date? { get { output.decodeStartedAt } set { output.decodeStartedAt = newValue } }
    var stopTokenID: Int? { get { output.stoppedToken } set { output.stoppedToken = newValue } }
    var lastToken: Int = 0
    var position: Int = 0
    var inBatchMode: Bool = false
    var isPrefilling: Bool = false
    var prefillCursor: Int = 0
    var effectiveMaxNew: Int = 0
    var prefillWallAcc: Double = 0
    let scheduledJob: QWBatchScheduledJob?
    let promise = QWBatchJobPromise<FNBatchJobOutput>()

    public convenience init(prompt: [Int], maxNew: Int, stopIDs: Set<Int>,
                            sampling: QWSamplerConfig, useSpec: Bool, draftMode: QWDraftStrategy,
                            maxDrafts: Int = FNGPUSessionData.draftCapacity,
                            relaxedMargin: Float = 0,
                            relaxedMinProb: Float = 0,
                            quadraticSeeding: Bool = true,
                            collectLogprobs: Bool = false, topLogprobsCount: Int = 0,
                            onPrefillProgress: (@Sendable (Int, Int) -> Void)? = nil,
                            onToken: ((Int) -> Bool)?, onStopToken: ((Int) -> Void)?,
                            onTokenLogprob: ((QWTokenLogprob) -> Void)? = nil,
                            constraint: (any QWTokenConstraint)? = nil,
                            retainPrefixLength: Int? = nil,
                            visualSpans: [QWVisualSpan] = [],
                            usePrefixCache: Bool = true,
                            cacheSalt: String? = nil) {
        self.init(
            prompt: prompt, maxNew: maxNew, stopIDs: stopIDs,
            sampling: sampling, useSpec: useSpec, draftMode: draftMode,
            maxDrafts: maxDrafts, relaxedMargin: relaxedMargin, relaxedMinProb: relaxedMinProb,
            quadraticSeeding: quadraticSeeding, collectLogprobs: collectLogprobs,
            topLogprobsCount: topLogprobsCount, onPrefillProgress: onPrefillProgress,
            onToken: onToken, onStopToken: onStopToken, onTokenLogprob: onTokenLogprob,
            constraint: constraint, retainPrefixLength: retainPrefixLength,
            visualSpans: visualSpans, usePrefixCache: usePrefixCache, cacheSalt: cacheSalt,
            scheduledJob: nil
        )
    }

    init(prompt: [Int], maxNew: Int, stopIDs: Set<Int>,
         sampling: QWSamplerConfig, useSpec: Bool, draftMode: QWDraftStrategy,
         maxDrafts: Int = FNGPUSessionData.draftCapacity,
         relaxedMargin: Float = 0,
         relaxedMinProb: Float = 0,
         quadraticSeeding: Bool = true,
         collectLogprobs: Bool = false, topLogprobsCount: Int = 0,
         onPrefillProgress: (@Sendable (Int, Int) -> Void)? = nil,
         onToken: ((Int) -> Bool)?, onStopToken: ((Int) -> Void)?,
         onTokenLogprob: ((QWTokenLogprob) -> Void)? = nil,
         constraint: (any QWTokenConstraint)? = nil,
         retainPrefixLength: Int? = nil,
         visualSpans: [QWVisualSpan] = [],
         usePrefixCache: Bool = true,
         cacheSalt: String? = nil,
         scheduledJob: QWBatchScheduledJob?) {
        self.scheduledJob = scheduledJob
        self.prompt = prompt
        self.maxNew = maxNew
        self.stopIDs = stopIDs
        self.sampling = sampling
        self.useSpec = useSpec && !collectLogprobs && (constraint?.allowsSpeculation ?? true)
        self.draftMode = draftMode
        self.collectLogprobs = collectLogprobs
        self.topLogprobsCount = topLogprobsCount
        self.onToken = onToken
        self.onStopToken = onStopToken
        self.onTokenLogprob = onTokenLogprob
        var s = FNGPUSessionData(history: prompt.map { Int64($0) },
                                 sampler: QWSampler(config: sampling),
                                 useSpec: useSpec && !collectLogprobs,
                                 collectLogprobs: collectLogprobs,
                                 topLogprobsCount: topLogprobsCount,
                                 constraint: constraint,
                                 retainPrefixLength: retainPrefixLength,
                                 visualSpans: visualSpans,
                                 usePrefixCache: usePrefixCache,
                                 cacheSalt: cacheSalt,
                                 onPrefillProgress: onPrefillProgress)
        s.draftMode = draftMode
        s.maxDrafts = max(1, min(FNGPUSessionData.draftCapacity, maxDrafts))
        s.relaxedMargin = max(0, relaxedMargin)
        s.relaxedMinProb = min(1, max(0, relaxedMinProb))
        s.quadraticSeeding = quadraticSeeding
        self.session = s
    }
}

/// Coordinates 1..4 concurrent inference sessions on `QWHyperStreamRuntime`.
public final class FNGPUBatchScheduler: @unchecked Sendable {
    public static let maxSlots = 4

    unowned let engine: QWHyperStreamRuntime
    let coordinator = QWBatchSlotCoordinator<FNBatchJob>(maxSlots: FNGPUBatchScheduler.maxSlots)

    var slotStates: [FNBatchSlotState]
    var batchArena: QWArena?
    var batchStack: FNGPUDecodeStack?
    var currentActiveSlot: Int = 0
    var currentTotalSlots: Int = 1
    var externalPollBlocked: Bool = false

    /// Optional callback invoked on any slot's prefix cache hit/miss evaluation.
    public var onCacheEvaluated: ((FNGPUSessionData) -> Void)?
    /// Optional callback invoked on Slot 0 prefix cache hit/miss.
    public var onSlot0CacheEvaluated: ((FNGPUSessionData) -> Void)?
    /// Optional callback invoked after each decode step with `(gpuMs, wallMs)`.
    public var onStepTiming: ((Double, Double) -> Void)?

    public init(engine: QWHyperStreamRuntime) {
        self.engine = engine
        let numLayers = engine.layerRecords.count
        var s0GdnStates = [[QWTensorValue]](repeating: [], count: numLayers)
        var s0GdnStateParity = [Int](repeating: 0, count: numLayers)
        var s0GdnConvState = [[QWTensorValue]](repeating: [], count: numLayers)
        var s0GdnConvParity = [Int](repeating: 0, count: numLayers)
        var s0PleHistory = [[QWTensorValue]](repeating: [], count: numLayers)
        var s0PleParity = [Int](repeating: 0, count: numLayers)
        for (i, rec) in engine.layerRecords.enumerated() {
            if let g = rec.gdn {
                s0GdnStates[i] = g.states
                s0GdnStateParity[i] = g.stateParity
                s0GdnConvState[i] = g.convState
                s0GdnConvParity[i] = g.convParity
            }
            if let p = rec.weights.ple {
                s0PleHistory[i] = p.history
                s0PleParity[i] = p.parity
            }
        }
        let slot0 = FNBatchSlotState(
            gdnStates: s0GdnStates, gdnStateParity: s0GdnStateParity,
            gdnConvState: s0GdnConvState, gdnConvParity: s0GdnConvParity,
            pleHistory: s0PleHistory, pleParity: s0PleParity,
            highWaterPos: 0
        )
        self.slotStates = [slot0]
    }

    /// Submits `job` to the scheduler and returns its `FNBatchJobOutput` when completed.
    public func submit(_ job: FNBatchJob) throws -> FNBatchJobOutput {
        if coordinator.submit(job) {
            driveLoop()
        }
        return try job.promise.wait()
    }

    func runScheduledBatch(
        initialJobs: [QWBatchScheduledJob],
        pollPending: (_ freeSlots: Int) -> [QWBatchScheduledJob],
        requeueDelayed: ([QWBatchScheduledJob]) -> Void
    ) {
        guard !initialJobs.isEmpty else { return }
        let initialBatchJobs = convertScheduledJobs(initialJobs)
        guard !initialBatchJobs.isEmpty else { return }
        coordinator.runExclusive {
            externalPollBlocked = false
            coordinator.withPendingAndSlots { pending, _ in
                pending.append(contentsOf: initialBatchJobs)
            }
            withoutActuallyEscaping(pollPending) { pollEsc in
                withoutActuallyEscaping(requeueDelayed) { requeueEsc in
                    driveLoop(
                        pollExternalPending: pollEsc,
                        requeueExternalDelayed: requeueEsc
                    )
                }
            }
            externalPollBlocked = false
        }
    }

    func convertScheduledJobs(_ scheduledJobs: [QWBatchScheduledJob]) -> [FNBatchJob] {
        var batchJobs: [FNBatchJob] = []
        batchJobs.reserveCapacity(scheduledJobs.count)
        for sJob in scheduledJobs {
            if sJob.request.prompt.isEmpty {
                sJob.complete(.success(QWGenerationResult(tokens: [], stepMs: [], stepTokenCounts: [])))
                continue
            }
            do {
                let job = try engine.makeBatchJob(
                    request: sJob.request,
                    onToken: sJob.onToken,
                    onStopToken: nil,
                    scheduledJob: sJob
                )
                batchJobs.append(job)
            } catch {
                sJob.complete(.failure(engine.mapBatchError(error)))
            }
        }
        return batchJobs
    }

    public func runExclusive<T>(_ body: () throws -> T) rethrows -> T {
        try coordinator.runExclusive(body)
    }

    func slotCapacity(totalSlots: Int) -> Int {
        let compress = engine.layerRecords.compactMap(\.attn).first?.compress ?? 4
        return QWBatchPartitionSelection.slotCapacity(
            fullCapacity: engine.capacity, totalSlots: totalSlots, alignment: compress
        )
    }

    func ensureBatchResources() throws {
        if batchStack != nil && slotStates.count == Self.maxSlots { return }
        let bArena = try QWArena(hal: engine.hal, weights: 0, activations: 512 << 20, kv: 0, gdnStates: 0, keepAlive: false)
        let act = bArena.activationsRegion()
        let numLayers = engine.layerRecords.count

        for _ in slotStates.count..<Self.maxSlots {
            var gdnStates = [[QWTensorValue]](repeating: [], count: numLayers)
            let gdnStateParity = [Int](repeating: 0, count: numLayers)
            var gdnConvState = [[QWTensorValue]](repeating: [], count: numLayers)
            let gdnConvParity = [Int](repeating: 0, count: numLayers)
            var pleHistory = [[QWTensorValue]](repeating: [], count: numLayers)
            let pleParity = [Int](repeating: 0, count: numLayers)
            for (i, rec) in engine.layerRecords.enumerated() {
                if let g = rec.gdn {
                    let sBuf = try act.allocate(g.states[0].shape)
                    memset(bArena.cpuPointer(of: sBuf), 0, sBuf.bytes)
                    gdnStates[i] = Array(repeating: sBuf, count: 8)
                    let c0 = try act.allocate(g.convState[0].shape)
                    let c1 = try act.allocate(g.convState[1].shape)
                    let c2 = try act.allocate(g.convState[2].shape)
                    let c3 = try act.allocate(g.convState[3].shape)
                    memset(bArena.cpuPointer(of: c0), 0, c0.bytes)
                    memset(bArena.cpuPointer(of: c1), 0, c1.bytes)
                    memset(bArena.cpuPointer(of: c2), 0, c2.bytes)
                    memset(bArena.cpuPointer(of: c3), 0, c3.bytes)
                    gdnConvState[i] = [c0, c1, c2, c3, c0, c1, c2, c3]
                }
                if let p = rec.weights.ple {
                    let h0 = try act.allocate(p.history[0].shape)
                    let h1 = try act.allocate(p.history[1].shape)
                    memset(bArena.cpuPointer(of: h0), 0, h0.bytes)
                    memset(bArena.cpuPointer(of: h1), 0, h1.bytes)
                    pleHistory[i] = [h0, h1, h0, h1, h0, h1, h0, h1]
                }
            }
            slotStates.append(FNBatchSlotState(
                gdnStates: gdnStates, gdnStateParity: gdnStateParity,
                gdnConvState: gdnConvState, gdnConvParity: gdnConvParity,
                pleHistory: pleHistory, pleParity: pleParity,
                highWaterPos: 0
            ))
        }

        var bSlots: [FNGPUDecodeStack.Slot] = []
        var sharedQSAScores4: QWTensorValue?
        let sharedGDN = engine.layerRecords.compactMap(\.gdn).first
        let sharedAttn = engine.layerRecords.compactMap(\.attn).first
        let sharedBatchScr: FNGPUScratch? = try engine.layerRecords.first.map { firstRec in
            try FNGPUScratch(arena: bArena, T: Self.maxSlots, hidden: engine.hidden, hc: engine.hc,
                             lowrank: firstRec.lowrank, moe: firstRec.weights.moe, ple: nil,
                             gdn: sharedGDN, attn: sharedAttn, extraEmbedDim: engine.maxPLEDim,
                             maxCapacity: engine.capacity)
        }
        for rec in engine.layerRecords {
            let st4 = try rec.attnState.flatMap { st in
                try rec.attn.map { a -> FNGPUAttnDecodeState in
                    let s4 = try FNGPUAttnDecodeState(sharingCachesOf: st, arena: bArena, mixer: a,
                                                      rows: Self.maxSlots, sharedQSAScores: sharedQSAScores4)
                    if sharedQSAScores4 == nil { sharedQSAScores4 = s4.qsaScores }
                    return s4
                }
            }
            let scr4: FNGPUScratch
            if rec.weights.ple != nil || sharedBatchScr == nil {
                scr4 = try FNGPUScratch(arena: bArena, T: Self.maxSlots, hidden: engine.hidden, hc: engine.hc,
                                        lowrank: rec.lowrank, moe: rec.weights.moe, ple: rec.weights.ple,
                                        gdn: rec.gdn ?? sharedGDN, attn: rec.attn ?? sharedAttn,
                                        maxCapacity: engine.capacity)
            } else {
                scr4 = sharedBatchScr!
            }
            bSlots.append(.init(weights: rec.weights, scratch: scr4, attn: st4, ple: rec.ple))
        }
        self.batchArena = bArena
        self.batchStack = try FNGPUDecodeStack(arena: bArena, ctx: engine.ctx, slots: bSlots,
                                               hidden: engine.hidden, hc: engine.hc, rows: Self.maxSlots,
                                               gpuPLETable: engine.gpuPLETable)
    }

    func syncSlotFromEngine(_ slotIdx: Int) {
        guard slotIdx >= 0 && slotIdx < slotStates.count else { return }
        for (i, rec) in engine.layerRecords.enumerated() {
            if let g = rec.gdn {
                slotStates[slotIdx].gdnStateParity[i] = g.stateParity
                slotStates[slotIdx].gdnConvParity[i] = g.convParity
            }
            if let p = rec.weights.ple {
                slotStates[slotIdx].pleParity[i] = p.parity
            }
        }
        slotStates[slotIdx].highWaterPos = max(slotStates[slotIdx].highWaterPos, engine.highWaterPos)
    }

    func activateSlot(_ slotIdx: Int, totalSlots: Int) {
        guard slotIdx >= 0 && slotIdx < slotStates.count else { return }
        for (i, rec) in engine.layerRecords.enumerated() {
            if let g = rec.gdn {
                g.states = slotStates[slotIdx].gdnStates[i]
                g.stateParity = slotStates[slotIdx].gdnStateParity[i]
                g.convState = slotStates[slotIdx].gdnConvState[i]
                g.convParity = slotStates[slotIdx].gdnConvParity[i]
            }
            if let p = rec.weights.ple {
                p.history = slotStates[slotIdx].pleHistory[i]
                p.parity = slotStates[slotIdx].pleParity[i]
            }
            if let a = rec.attnState, let m = rec.attn {
                a.selectPartition(slot: slotIdx, totalSlots: totalSlots, mixer: m)
            }
        }
        if let mtp = engine.mtp, case .attention(let m) = mtp.layer.mixer {
            mtp.attnState1.selectPartition(slot: slotIdx, totalSlots: totalSlots, mixer: m)
        }
        engine.currentPartitionSlot = slotIdx
        engine.currentPartitionTotalSlots = totalSlots
        if engine.residentPartitionTotalSlots != totalSlots {
            for s in 1..<Self.maxSlots {
                engine.residentKVHistoryBySlot[s]?.removeAll(keepingCapacity: true)
            }
            engine.residentPartitionTotalSlots = totalSlots
        }
        let sCap = slotCapacity(totalSlots: totalSlots)
        if totalSlots > 1, slotIdx == 0,
           let r0 = engine.residentKVHistoryBySlot[0], r0.count > sCap {
            engine.residentKVHistoryBySlot[0]?.removeAll(keepingCapacity: true)
        }
        let slotStart = slotIdx * sCap
        let priorGlobalInSlot = totalSlots > 1 ? max(0, min(sCap, slotStates[0].highWaterPos - slotStart)) : 0
        let effectiveWater = min(sCap, max(slotStates[slotIdx].highWaterPos, priorGlobalInSlot))
        slotStates[slotIdx].highWaterPos = effectiveWater
        engine.highWaterPos = effectiveWater
        currentActiveSlot = slotIdx
        currentTotalSlots = totalSlots
    }

    static func resetExtraSlotRecurrentStates(
        slotStates: inout [FNBatchSlotState],
        gdnFlags: [Bool],
        pleFlags: [Bool],
        arena: QWArena
    ) {
        guard slotStates.count > 1 else { return }
        for s in 1..<slotStates.count {
            for i in gdnFlags.indices where gdnFlags[i] {
                QWHyperStreamRuntime.resetRecurrentSlice(slotStates[s].gdnStates[i], parity: &slotStates[s].gdnStateParity[i], arena: arena)
                QWHyperStreamRuntime.resetRecurrentSlice(slotStates[s].gdnConvState[i], parity: &slotStates[s].gdnConvParity[i], arena: arena)
            }
            for i in pleFlags.indices where pleFlags[i] {
                QWHyperStreamRuntime.resetRecurrentSlice(slotStates[s].pleHistory[i], parity: &slotStates[s].pleParity[i], arena: arena)
            }
        }
    }

    func cleanupExtraSlotsBeforeSingleMode() {
        guard slotStates.count > 1 else { return }
        let gdnFlags = engine.layerRecords.map { $0.gdn != nil }
        let pleFlags = engine.layerRecords.map { $0.weights.ple != nil }
        Self.resetExtraSlotRecurrentStates(
            slotStates: &slotStates, gdnFlags: gdnFlags, pleFlags: pleFlags, arena: batchArena ?? engine.arena
        )
        let sCap = slotCapacity(totalSlots: Self.maxSlots)
        for s in 1..<slotStates.count {
            if slotStates[s].highWaterPos > 0 {
                let used = min(sCap, slotStates[s].highWaterPos)
                for rec in engine.layerRecords {
                    if let a = rec.attnState, let m = rec.attn {
                        a.selectPartition(slot: s, totalSlots: Self.maxSlots, mixer: m)
                        let cap = max(1, a.capacity)
                        let u = min(cap, used)
                        let idxRowBytes = (a.indexK.bytes / cap) * u
                        let poolBlocks = (u + 3) / 4
                        let poolBytes = min(a.pooled.bytes, (a.pooled.bytes / max(1, cap / 4)) * poolBlocks)
                        memset(engine.arena.cpuPointer(of: a.indexK), 0, idxRowBytes)
                        memset(engine.arena.cpuPointer(of: a.pooled), 0, poolBytes)
                    }
                }
                if let mtp = engine.mtp, case .attention(let m) = mtp.layer.mixer {
                    mtp.attnState1.selectPartition(slot: s, totalSlots: Self.maxSlots, mixer: m)
                    mtp.resetState(usedPositions: used)
                }
            }
            engine.residentKVHistoryBySlot[s]?.removeAll(keepingCapacity: true)
            slotStates[s].highWaterPos = 0
        }
    }

    func finishJob(slot: Int, job: FNBatchJob, error: (any Error)? = nil) {
        coordinator.setJob(nil, at: slot)
        externalPollBlocked = false
        if let error {
            job.promise.complete(.failure(error))
            job.scheduledJob?.complete(.failure(engine.mapBatchError(error)))
            return
        }
        let decSec = job.stepMs.isEmpty ? 0 : (job.decodeStart.map { Date().timeIntervalSince($0) } ?? 0)
        let out = FNBatchJobOutput(
            tokens: job.generatedTokens,
            stepMs: job.stepMs,
            stepTokenCounts: job.stepTokenCounts,
            prefillSeconds: job.prefillSeconds,
            firstTokenSeconds: job.prefillSeconds,
            decodeSeconds: decSec,
            prefillGPUSeconds: job.session.prefillGPUSec,
            stopTokenID: job.stopTokenID,
            prefixCacheHit: job.session.prefixCacheHit,
            prefixCacheMatchedTokens: job.session.prefixCacheMatchedTokens,
            prefixCacheTier: job.session.prefixCacheTier,
            specProposals: job.session.specProposals,
            specAccepted: job.session.specAccepted,
            tokenLogprobs: Array(job.session.tokenLogprobs.prefix(job.generatedTokens.count))
        )
        job.promise.complete(.success(out))
        if let scheduledJob = job.scheduledJob {
            let result = engine.finalizeGenerationResult(from: out, promptCount: job.prompt.count)
            scheduledJob.complete(.success(result))
        }
    }

    /// Emits `token` to `job`, returning `true` if `job` finished (hit stop token, `onToken` returned false, or reached `maxNew`).
    func emitAndCheckDone(token: Int, slot: Int, job: FNBatchJob) -> Bool {
        if token < 0 || token >= engine.vocab {
            finishJob(slot: slot, job: job, error: FNGPUError.invalidToken(token))
            return true
        }
        let accepted = job.output.recordPredictedToken(
            token,
            stopIDs: job.stopIDs,
            maxNew: job.maxNew,
            onStopToken: job.onStopToken,
            onAccepted: { tok in
                job.lastToken = tok
                if job.collectLogprobs, job.emittedLogprobsCount < job.session.tokenLogprobs.count {
                    let lp = job.session.tokenLogprobs[job.emittedLogprobsCount]
                    job.emittedLogprobsCount += 1
                    job.onTokenLogprob?(lp)
                }
                return job.onToken?(tok) ?? true
            }
        )
        if !accepted || job.output.finished {
            finishJob(slot: slot, job: job)
            return true
        }
        return false
    }

}
