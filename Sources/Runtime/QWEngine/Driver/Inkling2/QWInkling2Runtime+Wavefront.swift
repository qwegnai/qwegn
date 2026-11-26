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

    private func resetSharedAuxOperators() {
        dspark?.reset()
        if let lsa = lsaOperator {
            lsa.setPinnedBaseState(nil, position: 0)
            _ = lsa.restoreIndexerState(Data(), position: 0)
            lsa.stateOwner = nil
        }
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
        let canUseCache = req.usePrefixCache && prefixCache.isEnabled
        var cacheHit = false
        var matchedTokens = 0
        var pos = 0
        var restored = false
        var restoredExactFinalNorm: Data? = nil

        if canUseCache, req.prompt.count >= 1,
           let entry = prefixCache.matchLongestPrefix(
               tokens: req.prompt, salt: req.cacheSalt, limit: req.prompt.count, allowExact: true
           ) {
            let outcome = QWInkling2PrefixCache.restore(
                entry: entry, into: slot, kvModeRaw: session.kvModeRaw,
                dspark: nil, lsaOperator: nil, stateOwner: nil
            )
            if outcome.baseOk {
                pos = entry.key.tokens.count
                if pos == req.prompt.count { restoredExactFinalNorm = entry.finalNormRow }
                cacheHit = true
                matchedTokens = pos
                restored = true
                onCacheEvent?(.hit(tier: "warm", matchedTokens: pos, totalTokens: req.prompt.count))
            }
        }
        if !restored {
            slot.resetConvStates()
            if canUseCache {
                onCacheEvent?(.miss(totalTokens: req.prompt.count))
            }
        }

        let retainTarget: Int? = if canUseCache {
            if let explicitLen = req.retainPrefixLength, explicitLen > 0, explicitLen <= req.prompt.count {
                explicitLen
            } else {
                req.prompt.count
            }
        } else {
            nil
        }

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
            let snap = QWInkling2PrefixCache.capture(
                slot: slot, kvModeRaw: session.kvModeRaw, dspark: nil, lsaOperator: nil,
                tokens: Array(req.prompt.prefix(retainLen)), salt: req.cacheSalt,
                slidingWindow: ink2.slidingWindowSize
            )
            prefixCache.store(entry: snap)
        }

        if let normRow = restoredExactFinalNorm {
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
                let normRow = QWInkling2PrefixCache.captureFinalNormRow(
                    session: session, lastChunkRows: lastChunkRows, hidden: descriptor.hidden
                )
                let snap = QWInkling2PrefixCache.capture(
                    slot: slot, kvModeRaw: session.kvModeRaw, dspark: nil, lsaOperator: nil,
                    tokens: Array(req.prompt.prefix(retainLen)), salt: req.cacheSalt,
                    slidingWindow: ink2.slidingWindowSize, finalNormRow: normRow
                )
                prefixCache.store(entry: snap)
            }
        }

        let prefillSec = elapsedSeconds(prefillStart.duration(to: clock.now))
        let effectiveMaxNew = min(max(0, req.maxNew), max(0, slot.capacity - pos))
        let decodeStart = clock.now
        let cacheAnswer: QWCacheAnswer? = canUseCache
            ? QWCacheAnswer(hit: cacheHit, matchedTokens: matchedTokens,
                            totalTokens: req.prompt.count, tier: cacheHit ? "warm" : nil)
            : nil
        var state = WaveSlotState(
            slot: slot, job: job, sampler: sampler, lpSampler: lpSampler, topKCount: topKCount,
            usePenalty: usePenalty, canUseGreedyHead: canUseGreedyHead, effectiveMaxNew: effectiveMaxNew,
            decodeStart: decodeStart, prefillSec: prefillSec, firstTokenSec: prefillSec,
            cacheAnswer: cacheAnswer, pos: pos, pendingToken: 0
        )

        guard effectiveMaxNew > 0 else {
            state.finished = true
            completeWaveSlot(state, clock: clock)
            return nil
        }

        var allowedScratch: [Int] = []
        let tok0: Int
        if canUseGreedyHead {
            tok0 = Int(headTokPtr[0])
        } else {
            let row0 = UnsafeBufferPointer(start: logitsPtr, count: V)
            tok0 = try sampleToken(
                sampler: sampler, logitsRow: row0, penaltyHistory: [],
                constraint: req.constraint, allowedScratch: &allowedScratch
            )
        }
        state.firstTokenSec = elapsedSeconds(prefillStart.duration(to: clock.now))
        if req.stopIDs.contains(tok0) {
            state.stopTokenID = tok0
            state.finished = true
            completeWaveSlot(state, clock: clock)
            return nil
        }
        try req.constraint?.accept(tok0)
        if let lpSampler {
            let row0 = UnsafeBufferPointer(start: logitsPtr, count: V)
            let lp = lpSampler.computeTokenLogprob(logits: row0, chosenToken: tok0, topK: topKCount)
            state.collectedLogprobs.append(lp)
            req.onTokenLogprob?(lp)
        }
        state.generated.append(tok0)
        state.pendingToken = tok0
        if usePenalty {
            Self.recordGeneratedTokenForPenalty(
                tok0, vocab: V,
                recentGenerated: &state.recentGeneratedForPenalty,
                activeLoopTokens: &state.penaltyHistory
            )
        }
        if job.onToken?(tok0) == false || state.generated.count >= effectiveMaxNew {
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
            let tok: Int
            if usedGreedyHead {
                tok = Int(headTokPtr[row])
            } else {
                let rowBuf = UnsafeBufferPointer(start: logitsPtr.advanced(by: row * V), count: V)
                tok = try sampleToken(
                    sampler: state.sampler, logitsRow: rowBuf, penaltyHistory: state.penaltyHistory,
                    constraint: state.job.request.constraint, allowedScratch: &state.allowedScratch
                )
            }
            state.stepMs.append(elapsedMs)
            if state.job.request.stopIDs.contains(tok) {
                state.stepCounts.append(0)
                state.stopTokenID = tok
                state.finished = true
                return
            }
            try state.job.request.constraint?.accept(tok)
            if let lpSampler = state.lpSampler {
                let rowBuf = UnsafeBufferPointer(start: logitsPtr.advanced(by: row * V), count: V)
                let lp = lpSampler.computeTokenLogprob(logits: rowBuf, chosenToken: tok, topK: state.topKCount)
                state.collectedLogprobs.append(lp)
                state.job.request.onTokenLogprob?(lp)
            }
            state.stepCounts.append(1)
            state.generated.append(tok)
            state.pendingToken = tok
            if state.usePenalty {
                Self.recordGeneratedTokenForPenalty(
                    tok, vocab: V,
                    recentGenerated: &state.recentGeneratedForPenalty,
                    activeLoopTokens: &state.penaltyHistory
                )
            }
            if state.job.onToken?(tok) == false || state.generated.count >= state.effectiveMaxNew {
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
