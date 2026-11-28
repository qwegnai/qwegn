// QWInkling2Runtime+Wavefront.swift — Continuous B=2..4 wavefront scheduler loop for Inkling-2.
import Foundation
import QWCore
import QWHead

extension QWInkling2Runtime {
    private struct WaveSlotState {
        let slot: Ink2SlotState
        let job: QWBatchScheduledJob
        let sampler: QWSampler
        let lpSampler: QWSampler?
        let topKCount: Int
        let usePenalty: Bool
        let canUseGreedyHead: Bool
        let effectiveMaxNew: Int
        let decodeStart: ContinuousClock.Instant
        let prefillSec: Double
        var firstTokenSec: Double
        let cacheAnswer: QWCacheAnswer?
        var pos: Int
        var pendingToken: Int
        var recentGeneratedForPenalty: [Int] = []
        var penaltyHistory: [Int] = []
        var allowedScratch: [Int] = []
        var generated: [Int] = []
        var stepMs: [Double] = []
        var stepCounts: [Int] = []
        var collectedLogprobs: [QWTokenLogprob] = []
        var stopTokenID: Int? = nil
        var finished: Bool = false
        var failedError: (any Error)? = nil
    }

    private func validateWaveRequest(_ req: QWGenerationRequest) -> (any Error)? {
        if req.usePrefixCache && !hasPrefixCacheTier {
            return QWEngineDriverError.state("the request wants a prefix cache this engine does not have")
        }
        for tok in req.prompt where tok < 0 || tok >= descriptor.vocab {
            return QWModuleError.invalidToken(tok)
        }
        if req.prompt.count >= capacity {
            return QWEngineError.state(
                "prompt length \(req.prompt.count) meets/exceeds capacity \(capacity)"
            )
        }
        return nil
    }

    private func isWavefrontEligible(_ req: QWGenerationRequest, slotCap: Int) -> Bool {
        guard req.visualSpans.isEmpty, !req.prompt.isEmpty, req.maxNew > 0 else { return false }
        let effNew = min(req.maxNew, max(0, capacity - req.prompt.count))
        guard effNew > 0 else { return false }
        let maxPos = req.prompt.count + effNew
        if maxPos > slotCap { return false }
        if let lsa = lsaOperator, maxPos > lsa.config.denseBypassThreshold { return false }
        return true
    }

    private func runSequentialJob(_ job: QWBatchScheduledJob) {
        do {
            if job.request.usePrefixCache && !hasPrefixCacheTier {
                throw QWEngineDriverError.state("the request wants a prefix cache this engine does not have")
            }
            let res = try runGenerationLocked(request: job.request, onToken: job.onToken, onStopToken: nil)
            job.complete(.success(res))
        } catch {
            job.complete(.failure(error))
        }
    }

    private func completeWaveSlot(_ state: WaveSlotState, clock: ContinuousClock) {
        if let err = state.failedError {
            state.job.complete(.failure(err))
            return
        }
        let decodeSec = elapsedSeconds(state.decodeStart.duration(to: clock.now))
        let result = QWGenerationResult(
            tokens: state.generated,
            stepMs: state.stepMs,
            stepTokenCounts: state.stepCounts,
            prefillSeconds: state.prefillSec,
            firstTokenSeconds: state.firstTokenSec,
            decodeSeconds: decodeSec,
            stopTokenID: state.stopTokenID,
            cache: state.cacheAnswer,
            speculativeSummary: nil,
            tokenLogprobs: state.collectedLogprobs
        )
        stateLock.withLock { _lastResult = result }
        state.job.complete(.success(result))
    }

    private func admitWaveJob(
        _ job: QWBatchScheduledJob,
        slot: Ink2SlotState,
        clock: ContinuousClock
    ) throws -> WaveSlotState? {
        let req = job.request
        let prefillStart = clock.now
        let prefix = preparePrefixState(for: req, slot: slot, allowAuxOperators: false)
        var pos = prefix.pos
        let retainTarget = prefix.retainTarget

        let V = ink2.unpaddedVocabSize
        let logitsPtr = session.logitsF32Buf.contents().assumingMemoryBound(to: Float.self)
        let headTokPtr = session.headTokensBuf.contents().assumingMemoryBound(to: Int32.self)
        let sampler = QWSampler(config: req.sampling)
        let lpSampler = req.collectLogprobs ? QWSampler(config: req.sampling) : nil
        let topKCount = max(0, min(20, req.topLogprobsCount))
        let usePenalty = sampler.config.repetitionPenalty != 1.0
        let canUseGreedyHead = sampler.config.isGreedy && !usePenalty
            && req.constraint == nil && lpSampler == nil

        if let retainLen = retainTarget, retainLen > pos, retainLen < req.prompt.count {
            let startPos = pos
            let seg = Array(req.prompt[pos..<retainLen])
            try forwardPass.forwardOnSlot(
                slot: slot, tokens: seg, posOffset: startPos, runHeadRows: 0,
                greedyHead: false, useLSA: false, profiler: prefillProfiler,
                onPrefillProgress: { done, _ in req.onPrefillProgress?(startPos + done, req.prompt.count) }
            )
            pos = retainLen
            storePrefixSnapshot(slot: slot, tokens: Array(req.prompt.prefix(retainLen)), salt: req.cacheSalt)
        }

        if let normRow = prefix.restoredExactFinalNorm {
            try forwardPass.runHeadFromFinalNorm(
                finalNormRow: normRow, greedyHead: canUseGreedyHead, profiler: prefillProfiler
            )
            req.onPrefillProgress?(pos, req.prompt.count)
        } else if pos < req.prompt.count {
            let startPos = pos
            let seg = Array(req.prompt[pos..<req.prompt.count])
            let lastChunkRows = try forwardPass.forwardOnSlot(
                slot: slot, tokens: seg, posOffset: startPos, runHeadRows: 1,
                greedyHead: canUseGreedyHead, useLSA: false, profiler: prefillProfiler,
                onPrefillProgress: { done, _ in req.onPrefillProgress?(startPos + done, req.prompt.count) }
            )
            pos = req.prompt.count
            if let retainLen = retainTarget, pos == retainLen {
                storePrefixSnapshot(
                    slot: slot, tokens: Array(req.prompt.prefix(retainLen)),
                    salt: req.cacheSalt, lastChunkRowsForFinalNorm: lastChunkRows
                )
            }
        }

        let prefillSec = elapsedSeconds(prefillStart.duration(to: clock.now))
        let effectiveMaxNew = min(max(0, req.maxNew), max(0, slot.capacity - pos))
        let decodeStart = clock.now
        var state = WaveSlotState(
            slot: slot, job: job, sampler: sampler, lpSampler: lpSampler, topKCount: topKCount,
            usePenalty: usePenalty, canUseGreedyHead: canUseGreedyHead, effectiveMaxNew: effectiveMaxNew,
            decodeStart: decodeStart, prefillSec: prefillSec, firstTokenSec: prefillSec,
            cacheAnswer: prefix.cacheAnswer, pos: pos, pendingToken: 0
        )

        guard effectiveMaxNew > 0 else {
            state.finished = true
            completeWaveSlot(state, clock: clock)
            return nil
        }

        let tok0 = try sampleRowToken(
            row: 0, vocab: V, useGreedyHead: canUseGreedyHead,
            logitsPtr: logitsPtr, headTokPtr: headTokPtr,
            sampler: sampler, penaltyHistory: [],
            constraint: req.constraint, allowedScratch: &state.allowedScratch
        )
        state.firstTokenSec = elapsedSeconds(prefillStart.duration(to: clock.now))
        let emit0 = try consumeEmittedToken(
            tok0, row: 0, vocab: V, logitsPtr: logitsPtr,
            stopIDs: req.stopIDs, constraint: req.constraint,
            lpSampler: lpSampler, topKCount: topKCount, usePenalty: usePenalty,
            recentGeneratedForPenalty: &state.recentGeneratedForPenalty,
            penaltyHistory: &state.penaltyHistory, generated: &state.generated,
            collectedLogprobs: &state.collectedLogprobs, stopTokenID: &state.stopTokenID,
            onTokenLogprob: req.onTokenLogprob, onStopToken: nil, onToken: job.onToken
        )
        if emit0 == .stopped {
            state.finished = true
            completeWaveSlot(state, clock: clock)
            return nil
        }
        state.pendingToken = tok0
        if emit0 == .cancelled || state.generated.count >= effectiveMaxNew {
            state.finished = true
            completeWaveSlot(state, clock: clock)
            return nil
        }
        return state
    }

    private func consumeDecodedRow(
        state: inout WaveSlotState,
        row: Int,
        elapsedMs: Double,
        usedGreedyHead: Bool,
        V: Int,
        logitsPtr: UnsafeMutablePointer<Float>,
        headTokPtr: UnsafeMutablePointer<Int32>
    ) {
        do {
            let tok = try sampleRowToken(
                row: row, vocab: V, useGreedyHead: usedGreedyHead,
                logitsPtr: logitsPtr, headTokPtr: headTokPtr,
                sampler: state.sampler, penaltyHistory: state.penaltyHistory,
                constraint: state.job.request.constraint, allowedScratch: &state.allowedScratch
            )
            state.stepMs.append(elapsedMs)
            let outcome = try consumeEmittedToken(
                tok, row: row, vocab: V, logitsPtr: logitsPtr,
                stopIDs: state.job.request.stopIDs, constraint: state.job.request.constraint,
                lpSampler: state.lpSampler, topKCount: state.topKCount, usePenalty: state.usePenalty,
                recentGeneratedForPenalty: &state.recentGeneratedForPenalty,
                penaltyHistory: &state.penaltyHistory, generated: &state.generated,
                collectedLogprobs: &state.collectedLogprobs, stopTokenID: &state.stopTokenID,
                onTokenLogprob: state.job.request.onTokenLogprob, onStopToken: nil, onToken: state.job.onToken
            )
            if outcome == .stopped {
                state.stepCounts.append(0)
                state.finished = true
                return
            }
            state.stepCounts.append(1)
            state.pendingToken = tok
            if outcome == .cancelled || state.generated.count >= state.effectiveMaxNew {
                state.finished = true
            }
        } catch {
            state.failedError = error
            state.finished = true
        }
    }

    /// Runs continuous B=2..4 Inkling-2 batch wavefront under `execLock`, preserving solo fallback.
    func runScheduledBatch(
        initialJobs: [QWBatchScheduledJob],
        pollPending: (_ freeSlots: Int) -> [QWBatchScheduledJob],
        requeueDelayed: ([QWBatchScheduledJob]) -> Void
    ) {
        execLock.withLock {
            resetSharedAuxOperators()
            defer { resetSharedAuxOperators() }
            guard waveSlots.count > 1 else {
                for job in initialJobs { runSequentialJob(job) }
                return
            }

            let waveSlotCap = waveSlots.dropFirst().first?.capacity ?? waveSlots[0].capacity
            var eligible: [QWBatchScheduledJob] = []
            var sequential: [QWBatchScheduledJob] = []
            for job in initialJobs {
                if let err = validateWaveRequest(job.request) {
                    job.complete(.failure(err))
                } else if isWavefrontEligible(job.request, slotCap: waveSlotCap) {
                    eligible.append(job)
                } else {
                    sequential.append(job)
                }
            }
            guard eligible.count >= 2 else {
                for job in eligible { runSequentialJob(job) }
                for job in sequential { runSequentialJob(job) }
                return
            }

            let clock = ContinuousClock()
            let V = ink2.unpaddedVocabSize
            let logitsPtr = session.logitsF32Buf.contents().assumingMemoryBound(to: Float.self)
            let headTokPtr = session.headTokensBuf.contents().assumingMemoryBound(to: Int32.self)
            prefillProfiler?.reset()
            decodeProfiler?.reset()

            var freeSlotIndices = Array(0..<waveSlots.count)
            var active: [WaveSlotState] = []
            var overflow = eligible

            func releaseSlot(_ idx: Int) {
                freeSlotIndices.append(idx)
                freeSlotIndices.sort()
            }

            func tryAdmit(_ job: QWBatchScheduledJob) {
                guard !freeSlotIndices.isEmpty else { return }
                let slotIdx = freeSlotIndices.removeFirst()
                do {
                    if let state = try admitWaveJob(job, slot: waveSlots[slotIdx], clock: clock) {
                        active.append(state)
                    } else {
                        releaseSlot(slotIdx)
                    }
                } catch {
                    job.complete(.failure(error))
                    releaseSlot(slotIdx)
                }
            }

            while !freeSlotIndices.isEmpty && !overflow.isEmpty {
                tryAdmit(overflow.removeFirst())
            }

            while !active.isEmpty || !overflow.isEmpty {
                while !freeSlotIndices.isEmpty && !overflow.isEmpty {
                    tryAdmit(overflow.removeFirst())
                }
                if !freeSlotIndices.isEmpty && !active.isEmpty {
                    let polled = pollPending(freeSlotIndices.count)
                    var delayed: [QWBatchScheduledJob] = []
                    for job in polled {
                        if let err = validateWaveRequest(job.request) {
                            job.complete(.failure(err))
                        } else if !freeSlotIndices.isEmpty,
                                  isWavefrontEligible(job.request, slotCap: waveSlotCap) {
                            tryAdmit(job)
                        } else {
                            delayed.append(job)
                        }
                    }
                    if !delayed.isEmpty { requeueDelayed(delayed) }
                }

                for i in stride(from: active.count - 1, through: 0, by: -1) {
                    if active[i].finished || active[i].pos >= active[i].slot.capacity {
                        let done = active.remove(at: i)
                        completeWaveSlot(done, clock: clock)
                        releaseSlot(done.slot.slotIndex)
                    }
                }
                if active.isEmpty { continue }

                if active.count == 1 {
                    let stepStart = clock.now
                    let useGreedy = active[0].canUseGreedyHead
                    do {
                        try forwardPass.forwardOnSlot(
                            slot: active[0].slot, tokens: [active[0].pendingToken],
                            posOffset: active[0].pos, runHeadRows: 1,
                            greedyHead: useGreedy, useLSA: false, profiler: decodeProfiler
                        )
                        let elapsedMs = elapsedSeconds(stepStart.duration(to: clock.now)) * 1000.0
                        active[0].pos += 1
                        consumeDecodedRow(
                            state: &active[0], row: 0, elapsedMs: elapsedMs, usedGreedyHead: useGreedy,
                            V: V, logitsPtr: logitsPtr, headTokPtr: headTokPtr
                        )
                    } catch {
                        active[0].failedError = error
                        active[0].finished = true
                    }
                } else {
                    let allGreedy = active.allSatisfy(\.canUseGreedyHead)
                    let stepStart = clock.now
                    do {
                        try forwardPass.forwardWavefrontStep(
                            slots: active.map(\.slot),
                            positions: active.map(\.pos),
                            tokens: active.map(\.pendingToken),
                            greedyHead: allGreedy,
                            profiler: decodeProfiler
                        )
                        let elapsedMs = elapsedSeconds(stepStart.duration(to: clock.now)) * 1000.0
                        for b in 0..<active.count {
                            active[b].pos += 1
                            consumeDecodedRow(
                                state: &active[b], row: b, elapsedMs: elapsedMs, usedGreedyHead: allGreedy,
                                V: V, logitsPtr: logitsPtr, headTokPtr: headTokPtr
                            )
                        }
                    } catch {
                        for b in 0..<active.count {
                            active[b].failedError = error
                            active[b].finished = true
                        }
                    }
                }

                for i in stride(from: active.count - 1, through: 0, by: -1) where active[i].finished {
                    let done = active.remove(at: i)
                    completeWaveSlot(done, clock: clock)
                    releaseSlot(done.slot.slotIndex)
                }
            }

            if !sequential.isEmpty {
                resetSharedAuxOperators()
                for job in sequential { runSequentialJob(job) }
            }
        }
    }
}
