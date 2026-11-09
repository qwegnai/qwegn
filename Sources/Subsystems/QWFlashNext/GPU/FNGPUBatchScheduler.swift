// FNGPUBatchScheduler.swift — Multi-session continuous batching scheduler for FNGPUEngine.
// Runs single-stream requests on Slot 0 with full capacity, prefix cache, and speculative decoding,
// and dynamically coalesces 2..4 concurrent requests into lockstep batched decode across 4 slots.

import Foundation
import QWAlloc
import QWCore
import QWHAL
import QWHead

/// Per-slot state for multi-session continuous batching on `FNGPUEngine`.
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
    var generatedTokens: [Int] = []
    var emittedLogprobsCount: Int = 0
    var stepMs: [Double] = []
    var stepTokenCounts: [Int] = []
    var prefillSeconds: Double = 0
    var decodeStart: Date?
    var stopTokenID: Int?
    var lastToken: Int = 0
    var position: Int = 0
    var inBatchMode: Bool = false
    let done = DispatchSemaphore(value: 0)
    var outcome: Result<FNBatchJobOutput, any Error>?

    public init(prompt: [Int], maxNew: Int, stopIDs: Set<Int>,
                sampling: QWSamplerConfig, useSpec: Bool, draftMode: QWDraftStrategy,
                maxDrafts: Int = FNGPUSessionData.draftCapacity,
                relaxedMargin: Float = 0,
                relaxedMinProb: Float = 0,
                quadraticSeeding: Bool = true,
                collectLogprobs: Bool = false, topLogprobsCount: Int = 0,
                onToken: ((Int) -> Bool)?, onStopToken: ((Int) -> Void)?,
                onTokenLogprob: ((QWTokenLogprob) -> Void)? = nil,
                constraint: (any QWTokenConstraint)? = nil,
                retainPrefixLength: Int? = nil,
                visualSpans: [QWVisualSpan] = []) {
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
                                 visualSpans: visualSpans)
        s.draftMode = draftMode
        s.maxDrafts = max(1, min(FNGPUSessionData.draftCapacity, maxDrafts))
        s.relaxedMargin = max(0, relaxedMargin)
        s.relaxedMinProb = min(1, max(0, relaxedMinProb))
        s.quadraticSeeding = quadraticSeeding
        self.session = s
    }
}

/// Tracks requests currently rendering chat templates / tokenizing on the server so concurrent bursts coalesce cleanly.
public enum FNBatchAdmissionTracker {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var count: Int = 0

    public static func beginPreparing() {
        lock.lock()
        count += 1
        lock.unlock()
    }

    public static func endPreparing() {
        lock.lock()
        count = max(0, count - 1)
        lock.unlock()
    }

    public static var preparingCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }
}

/// Coordinates 1..4 concurrent inference sessions on `FNGPUEngine`.
public final class FNGPUBatchScheduler: @unchecked Sendable {
    public static let maxSlots = 4

    private unowned let engine: FNGPUEngine
    private let lock = NSLock()
    private var pendingQueue: [FNBatchJob] = []
    private var activeJobs: [FNBatchJob?] = [nil, nil, nil, nil]
    private var isDriving = false

    private var slotStates: [FNBatchSlotState]
    private var batchArena: QWArena?
    private var batchStack: FNGPUDecodeStack?
    private var currentActiveSlot: Int = 0
    private var currentTotalSlots: Int = 1

    /// Optional callback invoked on any slot's prefix cache hit/miss evaluation.
    public var onCacheEvaluated: ((FNGPUSessionData) -> Void)?
    /// Optional callback invoked on Slot 0 prefix cache hit/miss.
    public var onSlot0CacheEvaluated: ((FNGPUSessionData) -> Void)?
    /// Optional callback invoked after each decode step with `(gpuMs, wallMs)`.
    public var onStepTiming: ((Double, Double) -> Void)?

    public init(engine: FNGPUEngine) {
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
        lock.lock()
        pendingQueue.append(job)
        let becomeDriver = !isDriving
        if becomeDriver { isDriving = true }
        lock.unlock()

        if becomeDriver {
            driveLoop()
        }
        job.done.wait()
        return try job.outcome!.get()
    }

    private func slotCapacity(totalSlots: Int) -> Int {
        guard totalSlots > 1 else { return engine.capacity }
        let compress = engine.layerRecords.compactMap(\.attn).first?.compress ?? 4
        return max(compress, ((engine.capacity / totalSlots) / compress) * compress)
    }

    private func ensureBatchResources() throws {
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
                                               hidden: engine.hidden, hc: engine.hc, rows: Self.maxSlots)
    }

    private func syncSlotFromEngine(_ slotIdx: Int) {
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

    private func activateSlot(_ slotIdx: Int, totalSlots: Int) {
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

    private func wipeSlotState(_ slotIdx: Int, totalSlots: Int) {
        activateSlot(slotIdx, totalSlots: totalSlots)
        let sCap = slotCapacity(totalSlots: totalSlots)
        let slotStart = slotIdx * sCap
        let priorGlobalInSlot = max(0, min(sCap, slotStates[0].highWaterPos - slotStart))
        let used = min(sCap, max(slotStates[slotIdx].highWaterPos, priorGlobalInSlot))
        let ptrArena = slotIdx == 0 ? engine.arena : (batchArena ?? engine.arena)
        for (i, rec) in engine.layerRecords.enumerated() {
            if let g = rec.gdn {
                memset(ptrArena.cpuPointer(of: g.states[0]), 0, g.states[0].bytes)
                g.stateParity = 0
                slotStates[slotIdx].gdnStateParity[i] = 0
                memset(ptrArena.cpuPointer(of: g.convState[0]), 0, g.convState[0].bytes)
                g.convParity = 0
                slotStates[slotIdx].gdnConvParity[i] = 0
            }
            if let p = rec.weights.ple {
                memset(ptrArena.cpuPointer(of: p.history[0]), 0, p.history[0].bytes)
                p.parity = 0
                slotStates[slotIdx].pleParity[i] = 0
            }
            if used > 0, let a = rec.attnState {
                let cap = max(1, a.capacity)
                let u = min(cap, used)
                let idxRowBytes = (a.indexK.bytes / cap) * u
                let poolBlocks = (u + 3) / 4
                let poolBytes = min(a.pooled.bytes, (a.pooled.bytes / max(1, cap / 4)) * poolBlocks)
                memset(engine.arena.cpuPointer(of: a.indexK), 0, idxRowBytes)
                memset(engine.arena.cpuPointer(of: a.pooled), 0, poolBytes)
            }
        }
        slotStates[slotIdx].highWaterPos = 0
        engine.highWaterPos = 0
    }

    private func cleanupExtraSlotsBeforeSingleMode() {
        guard slotStates.count > 1 else { return }
        let sCap = slotCapacity(totalSlots: Self.maxSlots)
        for s in 1..<slotStates.count where slotStates[s].highWaterPos > 0 {
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
            engine.residentKVHistoryBySlot[s]?.removeAll(keepingCapacity: true)
            slotStates[s].highWaterPos = 0
        }
    }

    private func finishJob(slot: Int, job: FNBatchJob, error: (any Error)? = nil) {
        activeJobs[slot] = nil
        if let error {
            job.outcome = .failure(error)
        } else {
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
            job.outcome = .success(out)
        }
        job.done.signal()
    }

    /// Emits `token` to `job`, returning `true` if `job` finished (hit stop token, `onToken` returned false, or reached `maxNew`).
    private func emitAndCheckDone(token: Int, slot: Int, job: FNBatchJob) -> Bool {
        if token < 0 || token >= engine.vocab {
            finishJob(slot: slot, job: job, error: FNGPUError.invalidToken(token))
            return true
        }
        if job.stopIDs.contains(token) {
            job.stopTokenID = token
            job.onStopToken?(token)
            finishJob(slot: slot, job: job)
            return true
        }
        job.generatedTokens.append(token)
        job.lastToken = token
        if job.collectLogprobs, job.emittedLogprobsCount < job.session.tokenLogprobs.count {
            let lp = job.session.tokenLogprobs[job.emittedLogprobsCount]
            job.emittedLogprobsCount += 1
            job.onTokenLogprob?(lp)
        }
        let keepGoing = job.onToken?(token) ?? true
        if !keepGoing || job.generatedTokens.count >= job.maxNew {
            finishJob(slot: slot, job: job)
            return true
        }
        return false
    }

    private func driveLoop() {
        while true {
            // Micro-coalesce concurrent requests currently tokenizing when starting a new wave.
            let activeCount0 = activeJobs.compactMap { $0 }.count
            if activeCount0 == 0 {
                var waitedUs = 0
                while waitedUs < 15_000 {
                    lock.lock()
                    let qCount = pendingQueue.count
                    lock.unlock()
                    let prep = FNBatchAdmissionTracker.preparingCount
                    if qCount >= Self.maxSlots { break }
                    if prep == 0 && waitedUs >= 1_500 { break }
                    usleep(250)
                    waitedUs += 250
                }
            }

            // Admit waiting jobs into free slots.
            let sCap4 = slotCapacity(totalSlots: Self.maxSlots)
            var newlyAdmitted: [Int] = []
            lock.lock()
            if !pendingQueue.isEmpty {
                let currentActive = activeJobs.compactMap { $0 }.count
                let totalCandidate = currentActive + pendingQueue.count
                let firstFitsIn4 = (pendingQueue[0].prompt.count + pendingQueue[0].maxNew + 4 <= sCap4)
                if currentActive == 0 && (totalCandidate == 1 || !firstFitsIn4) {
                    let job = pendingQueue.removeFirst()
                    job.inBatchMode = false
                    activeJobs[0] = job
                    newlyAdmitted.append(0)
                } else {
                    // Never transition an already-running single-stream Slot 0 job into partitioned batch mode
                    // mid-flight: let it finish its current single-stream turn cleanly first.
                    let slot0CanBatch: Bool
                    if let j0 = activeJobs[0] {
                        slot0CanBatch = j0.inBatchMode
                            && (j0.session.history.count + (j0.maxNew - j0.generatedTokens.count) + 4 <= sCap4)
                    } else {
                        slot0CanBatch = true
                    }
                    if slot0CanBatch {
                        for s in 0..<Self.maxSlots where activeJobs[s] == nil && !pendingQueue.isEmpty {
                            let cand = pendingQueue[0]
                            let fitsIn4 = (cand.prompt.count + cand.maxNew + 4 <= sCap4)
                            if fitsIn4 || (s == 0 && activeJobs.compactMap({ $0 }).isEmpty) {
                                let job = pendingQueue.removeFirst()
                                job.inBatchMode = fitsIn4 && (totalCandidate > 1 || s > 0)
                                activeJobs[s] = job
                                newlyAdmitted.append(s)
                                if !job.inBatchMode { break }
                            } else {
                                break
                            }
                        }
                    }
                }
            }
            lock.unlock()

            if !newlyAdmitted.isEmpty {
                let allActiveSlots = (0..<Self.maxSlots).filter { activeJobs[$0] != nil }
                let useMultiSlot = allActiveSlots.count > 1 || allActiveSlots.contains(where: { $0 > 0 }) || activeJobs[allActiveSlots[0]]!.inBatchMode
                if useMultiSlot {
                    do {
                        try ensureBatchResources()
                    } catch {
                        for s in newlyAdmitted {
                            if let j = activeJobs[s] { finishJob(slot: s, job: j, error: error) }
                        }
                        continue
                    }
                    for s in allActiveSlots { activeJobs[s]?.inBatchMode = true }
                }

                // Prefill each newly admitted slot.
                for s in newlyAdmitted {
                    guard let job = activeJobs[s] else { continue }
                    do {
                        if job.inBatchMode {
                            activateSlot(s, totalSlots: Self.maxSlots)
                            job.session.useSpec = false
                            if !engine.hasPrefixCacheTier {
                                try engine.preparePrefill(forTokenCount: job.prompt.count)
                            }
                            engine.beginSession(&job.session, maxNew: job.maxNew)
                            let prefillStart = Date()
                            var chunkStart = 0
                            var firstTok = 0
                            let chunkRows = engine.dynamicPrefillChunkRows(forTotalTokens: job.prompt.count)
                            while chunkStart < job.prompt.count {
                                let end = min(chunkStart + chunkRows, job.prompt.count)
                                let isFirst = chunkStart == 0
                                firstTok = try engine.encodePrefillChunk(&job.session, chunk: chunkStart..<end,
                                                                         runsHead: end == job.prompt.count)
                                if isFirst {
                                    onCacheEvaluated?(job.session)
                                    onSlot0CacheEvaluated?(job.session)
                                }
                                chunkStart = end
                            }
                            job.prefillSeconds = Date().timeIntervalSince(prefillStart)
                            syncSlotFromEngine(s)
                            job.position = job.prompt.count
                            if job.maxNew <= 0 {
                                engine.flushPendingPrefixCache(&job.session)
                                job.decodeStart = Date()
                                finishJob(slot: s, job: job)
                            } else {
                                _ = emitAndCheckDone(token: firstTok, slot: s, job: job)
                                engine.flushPendingPrefixCache(&job.session)
                                job.decodeStart = Date()
                            }
                        } else {
                            cleanupExtraSlotsBeforeSingleMode()
                            activateSlot(0, totalSlots: 1)
                            if !engine.hasPrefixCacheTier {
                                try engine.preparePrefill(forTokenCount: job.prompt.count)
                            }
                            engine.beginSession(&job.session, maxNew: job.maxNew)
                            let prefillStart = Date()
                            var chunkStart = 0
                            var firstTok = 0
                            let chunkRows = engine.dynamicPrefillChunkRows(forTotalTokens: job.prompt.count)
                            while chunkStart < job.prompt.count {
                                let end = min(chunkStart + chunkRows, job.prompt.count)
                                let isFirst = chunkStart == 0
                                firstTok = try engine.encodePrefillChunk(&job.session, chunk: chunkStart..<end,
                                                                         runsHead: end == job.prompt.count)
                                if isFirst {
                                    onCacheEvaluated?(job.session)
                                    onSlot0CacheEvaluated?(job.session)
                                }
                                chunkStart = end
                            }
                            job.prefillSeconds = Date().timeIntervalSince(prefillStart)
                            syncSlotFromEngine(0)
                            job.position = job.prompt.count
                            if job.maxNew <= 0 {
                                engine.flushPendingPrefixCache(&job.session)
                                job.decodeStart = Date()
                                finishJob(slot: 0, job: job)
                            } else {
                                _ = emitAndCheckDone(token: firstTok, slot: 0, job: job)
                                engine.flushPendingPrefixCache(&job.session)
                                job.decodeStart = Date()
                            }
                        }
                    } catch {
                        finishJob(slot: s, job: job, error: error)
                    }
                }
                // Re-check pendingQueue in case new requests arrived while prefill was executing on the GPU.
                lock.lock()
                let moreWaiting = !pendingQueue.isEmpty
                lock.unlock()
                if moreWaiting && activeJobs.contains(where: { $0 == nil }) {
                    continue
                }
            }

            let activeSlots = (0..<Self.maxSlots).filter { activeJobs[$0] != nil }
            if activeSlots.isEmpty {
                lock.lock()
                if pendingQueue.isEmpty {
                    cleanupExtraSlotsBeforeSingleMode()
                    activateSlot(0, totalSlots: 1)
                    isDriving = false
                    lock.unlock()
                    return
                }
                lock.unlock()
                continue
            }

            // Single-stream fast path on Slot 0 (full capacity + speculative decoding).
            if activeSlots.count == 1 && activeSlots[0] == 0 && !activeJobs[0]!.inBatchMode {
                let job = activeJobs[0]!
                activateSlot(0, totalSlots: 1)
                do {
                    let prevRounds = job.session.roundTokenCounts.count
                    let started = Date()
                    let nextTok = try engine.encodeDecodeToken(&job.session, token: job.lastToken, position: job.position)
                    job.position += 1
                    let measured = Date().timeIntervalSince(started) * 1000.0
                    let reported = job.session.stepMs > 0 ? job.session.stepMs : measured
                    let gpuMs = job.session.lastStepGPUMillis
                    if gpuMs > 0 && reported > 0 {
                        onStepTiming?(gpuMs, reported)
                    }
                    job.stepMs.append(reported)
                    if job.session.roundTokenCounts.count > prevRounds {
                        job.stepTokenCounts = job.session.roundTokenCounts
                    } else if job.session.roundTokenCounts.isEmpty {
                        job.stepTokenCounts.append(1)
                    } else {
                        job.stepTokenCounts = job.session.roundTokenCounts
                    }
                    syncSlotFromEngine(0)
                    _ = emitAndCheckDone(token: nextTok, slot: 0, job: job)
                } catch {
                    finishJob(slot: 0, job: job, error: error)
                }
                continue
            }

            // Multi-session batched decode path across all `B` active slots (`1 <= B <= 4`).
            var validSlots: [Int] = []
            var histories: [[Int64]] = []
            for s in activeSlots {
                guard let job = activeJobs[s] else { continue }
                if job.session.history.count + 1 >= sCap4 {
                    finishJob(slot: s, job: job,
                              error: FNGPUError.contextDoesNotFit(requested: job.session.history.count + 1,
                                                                  maxFitting: sCap4 - 1))
                    continue
                }
                let prevCount = job.session.history.count
                let tok = job.lastToken
                job.session.history.append(Int64(tok))
                if engine.residentKVHistoryBySlot[s]?.count == prevCount {
                    engine.residentKVHistoryBySlot[s]?.append(Int64(tok))
                }
                if job.session.sampler.config.repetitionPenalty != 1.0 && tok >= 0 && tok < engine.vocab {
                    job.session.seenTokens.insert(tok)
                }
                slotStates[s].highWaterPos = max(slotStates[s].highWaterPos, job.session.history.count)
                validSlots.append(s)
                histories.append(job.session.history)
            }
            guard !validSlots.isEmpty, let bStack = batchStack else { continue }

            do {
                let started = Date()
                let rawArgmaxes = try bStack.forwardBatch(
                    histories: histories,
                    activeSlots: validSlots,
                    totalSlots: Self.maxSlots,
                    slotStates: &slotStates,
                    embedTable: engine.embedTable,
                    head: engine.head
                )
                let wallMs = Date().timeIntervalSince(started) * 1000.0
                let gpuMs = bStack.lastGPUMillis
                if gpuMs > 0 && wallMs > 0 {
                    onStepTiming?(gpuMs, wallMs)
                }
                for (b, s) in validSlots.enumerated() {
                    guard let job = activeJobs[s] else { continue }
                    let nextTok = try engine.sampleFromHead(
                        argmaxToken: rawArgmaxes[b],
                        seenTokens: job.session.seenTokens,
                        sampler: job.session.sampler,
                        row: b,
                        constraint: job.session.constraint
                    )
                    if job.session.collectLogprobs {
                        engine.recordTokenLogprob(&job.session, token: nextTok, row: b)
                    }
                    job.position += 1
                    job.stepMs.append(wallMs)
                    job.stepTokenCounts.append(1)
                    _ = emitAndCheckDone(token: nextTok, slot: s, job: job)
                }
            } catch {
                for s in validSlots {
                    if let job = activeJobs[s] { finishJob(slot: s, job: job, error: error) }
                }
            }
        }
    }
}
