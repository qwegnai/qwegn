// FNGPUBatchScheduler+Drive.swift — Multi-slot decode scheduler loop for QWHyperStreamRuntime.

import Foundation
import QWAlloc
import QWCore
import QWHAL
import QWHead

extension FNGPUBatchScheduler {
    func driveLoop(
        pollExternalPending: ((_ freeSlots: Int) -> [QWBatchScheduledJob])? = nil,
        requeueExternalDelayed: (([QWBatchScheduledJob]) -> Void)? = nil
    ) {
        while true {
            if pollExternalPending == nil && coordinator.activeSlotIndices().isEmpty {
                QWBatchAdmissionCounter.coalescePreparingBurst(maxSlots: Self.maxSlots) {
                    coordinator.queuedCount
                }
            }

            if let pollExternalPending, !externalPollBlocked {
                let activeNow = coordinator.activeSlotIndices()
                let slot0Unbatched = activeNow.count == 1
                    && activeNow[0] == 0
                    && coordinator.job(at: 0)?.inBatchMode == false
                let freeSlots = Self.maxSlots - activeNow.count
                if !activeNow.isEmpty && !slot0Unbatched && freeSlots > 0 && !coordinator.hasPending {
                    let polled = pollExternalPending(freeSlots)
                    if !polled.isEmpty {
                        let converted = convertScheduledJobs(polled)
                        if !converted.isEmpty {
                            coordinator.withPendingAndSlots { pending, _ in
                                pending.append(contentsOf: converted)
                            }
                        }
                    }
                }
            }

            let sCap4 = slotCapacity(totalSlots: Self.maxSlots)
            let newlyAdmitted = admitWaitingJobs(sCap4: sCap4)

            if let requeueExternalDelayed, coordinator.hasPending {
                let delayed = coordinator.withPendingAndSlots { pending, _ -> [QWBatchScheduledJob] in
                    var out: [QWBatchScheduledJob] = []
                    pending.removeAll { job in
                        guard let scheduled = job.scheduledJob else { return false }
                        out.append(scheduled)
                        return true
                    }
                    return out
                }
                if !delayed.isEmpty {
                    externalPollBlocked = true
                    requeueExternalDelayed(delayed)
                }
            }

            if !newlyAdmitted.isEmpty {
                let allActiveSlots = coordinator.activeSlotIndices()
                let useMultiSlot = allActiveSlots.count > 1
                    || allActiveSlots.contains(where: { $0 > 0 })
                    || (coordinator.job(at: allActiveSlots[0])?.inBatchMode == true)
                if useMultiSlot {
                    do {
                        try ensureBatchResources()
                    } catch {
                        for s in newlyAdmitted {
                            if let j = coordinator.job(at: s) { finishJob(slot: s, job: j, error: error) }
                        }
                        continue
                    }
                    for s in allActiveSlots { coordinator.job(at: s)?.inBatchMode = true }
                }

                let hadDecodingPeer = allActiveSlots.contains { slotIdx in
                    guard !newlyAdmitted.contains(slotIdx), let existing = coordinator.job(at: slotIdx) else {
                        return false
                    }
                    return !existing.isPrefilling
                }
                for s in newlyAdmitted {
                    guard let job = coordinator.job(at: s) else { continue }
                    let hasDecodeNow = hadDecodingPeer || coordinator.activeSlotIndices().contains { idx in
                        idx != s && coordinator.job(at: idx)?.isPrefilling == false && !newlyAdmitted.contains(idx)
                    }
                    do {
                        if hasDecodeNow && job.inBatchMode && QWInterleavedPrefillGate.isEnabled {
                            try beginInterleavedPrefillJob(slot: s, job: job, sCap4: sCap4)
                        } else {
                            try prefillAdmittedJob(slot: s, job: job, sCap4: sCap4)
                        }
                    } catch {
                        finishJob(slot: s, job: job, error: error)
                    }
                }
                if coordinator.hasPending && coordinator.hasFreeSlot {
                    continue
                }
            }

            let activeSlots = coordinator.activeSlotIndices()
            if activeSlots.isEmpty {
                let parked = coordinator.tryParkDriverIfIdle {
                    cleanupExtraSlotsBeforeSingleMode()
                    activateSlot(0, totalSlots: 1)
                }
                if parked { return }
                continue
            }

            // Recover Slot 0 to full-capacity single-stream mode once other batch slots drain.
            if activeSlots.count == 1 && activeSlots[0] == 0,
               let job0 = coordinator.job(at: 0), job0.inBatchMode,
               !coordinator.hasPending && QWBatchAdmissionCounter.preparingCount == 0 {
                cleanupExtraSlotsBeforeSingleMode()
                job0.inBatchMode = false
                if job0.generatedTokens.count <= 1 {
                    job0.session.useSpec = job0.useSpec
                }
            }

            // Single-stream fast path on Slot 0 (full capacity + speculative decoding).
            if activeSlots.count == 1 && activeSlots[0] == 0,
               let job = coordinator.job(at: 0), !job.inBatchMode, !job.isPrefilling {
                stepSingleStreamJob(job)
                continue
            }

            let decodingSlots = activeSlots.filter { coordinator.job(at: $0)?.isPrefilling == false }
            if !decodingSlots.isEmpty {
                stepMultiSlotBatch(activeSlots: decodingSlots, sCap4: sCap4)
            }
            if let pfSlot = activeSlots.first(where: { coordinator.job(at: $0)?.isPrefilling == true }),
               let pfJob = coordinator.job(at: pfSlot) {
                let hasDecodingNow = coordinator.activeSlotIndices().contains {
                    coordinator.job(at: $0)?.isPrefilling == false
                }
                do {
                    try stepInterleavedPrefillSlice(slot: pfSlot, job: pfJob, hasDecodingPeers: hasDecodingNow)
                } catch {
                    finishJob(slot: pfSlot, job: pfJob, error: error)
                }
            }
        }
    }

    private func admitWaitingJobs(sCap4: Int) -> [Int] {
        coordinator.withPendingAndSlots { pendingQueue, activeJobs in
            guard !pendingQueue.isEmpty else { return [] }
            let slot0Fits: Bool
            if let j0 = activeJobs[0] {
                slot0Fits = QWBatchPartitionSelection.fitsSlotCapacity(
                    usedTokens: j0.session.history.count,
                    remainingTokens: j0.maxNew - j0.generatedTokens.count,
                    slotCapacity: sCap4
                )
            } else {
                slot0Fits = true
            }
            let demands = pendingQueue.map {
                QWBatchPartitionSelection.SlotAdmissionDemand(promptCount: $0.prompt.count, maxNew: $0.maxNew)
            }
            let plan = QWBatchPartitionSelection.planAdmissions(
                maxSlots: Self.maxSlots,
                slotCapacity: sCap4,
                slot0ActiveInBatchMode: activeJobs[0]?.inBatchMode == true,
                slot0FitsInBatch: slot0Fits,
                occupiedSlots: activeJobs.map { $0 != nil },
                pendingDemands: demands
            )
            var newlyAdmitted: [Int] = []
            for item in plan {
                let job = pendingQueue.removeFirst()
                job.inBatchMode = item.inBatchMode
                activeJobs[item.slot] = job
                newlyAdmitted.append(item.slot)
            }
            return newlyAdmitted
        }
    }

    private func stepSingleStreamJob(_ job: FNBatchJob) {
        activateSlot(0, totalSlots: 1)
        let activeCap = engine.activeCapacity
        let capLimit = max(0, activeCap - job.prompt.count)
        if (job.session.pendingTokens.isEmpty && job.session.history.count + 1 >= activeCap)
            || job.generatedTokens.count >= capLimit {
            finishJob(slot: 0, job: job)
            return
        }
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
            if job.session.roundTokenCounts.count > prevRounds || !job.session.roundTokenCounts.isEmpty {
                job.stepTokenCounts = job.session.roundTokenCounts
            } else {
                job.stepTokenCounts.append(1)
            }
            syncSlotFromEngine(0)
            let done = emitAndCheckDone(token: nextTok, slot: 0, job: job)
            if !done && job.generatedTokens.count >= capLimit {
                finishJob(slot: 0, job: job)
            }
        } catch {
            finishJob(slot: 0, job: job, error: error)
        }
    }

    private func stepMultiSlotBatch(activeSlots: [Int], sCap4: Int) {
        var validSlots: [Int] = []
        var histories: [[Int64]] = []
        let slotCap = min(sCap4, engine.activeCapacity)
        for s in activeSlots {
            guard let job = coordinator.job(at: s) else { continue }
            if job.session.history.count + 1 >= slotCap
                || job.generatedTokens.count >= max(0, slotCap - job.prompt.count) {
                finishJob(slot: s, job: job)
                continue
            }
            let prevCount = job.session.history.count
            let tok = job.lastToken
            job.session.history.append(Int64(tok))
            if engine.residentKVHistoryBySlot[s]?.count == prevCount {
                engine.residentKVHistoryBySlot[s]?.append(Int64(tok))
            }
            slotStates[s].highWaterPos = max(slotStates[s].highWaterPos, job.session.history.count)
            validSlots.append(s)
            histories.append(job.session.history)
        }
        guard !validSlots.isEmpty, let bStack = batchStack else { return }

        let rawArgmaxes: [Int]
        let wallMs: Double
        do {
            let started = Date()
            rawArgmaxes = try bStack.forwardBatch(
                histories: histories,
                activeSlots: validSlots,
                totalSlots: Self.maxSlots,
                slotStates: &slotStates,
                embedTable: engine.embedTable,
                head: engine.head
            )
            wallMs = Date().timeIntervalSince(started) * 1000.0
            let gpuMs = bStack.lastGPUMillis
            if gpuMs > 0 && wallMs > 0 {
                onStepTiming?(gpuMs, wallMs)
            }
        } catch {
            for s in validSlots {
                if let job = coordinator.job(at: s) { finishJob(slot: s, job: job, error: error) }
            }
            return
        }

        for (b, s) in validSlots.enumerated() {
            guard let job = coordinator.job(at: s) else { continue }
            do {
                let nextTok = try engine.sampleFromHead(
                    argmaxToken: rawArgmaxes[b],
                    seenTokens: job.session.seenTokens,
                    seenTokenList: job.session.seenTokenList,
                    sampler: job.session.sampler,
                    row: b,
                    constraint: job.session.constraint
                )
                if job.session.sampler.config.repetitionPenalty != 1.0 {
                    job.session.recordSeenToken(nextTok, vocab: engine.vocab)
                }
                if job.session.collectLogprobs {
                    engine.recordTokenLogprob(&job.session, token: nextTok, row: b)
                }
                job.position += 1
                job.stepMs.append(wallMs)
                job.stepTokenCounts.append(1)
                let done = emitAndCheckDone(token: nextTok, slot: s, job: job)
                if !done && (job.session.history.count + 1 >= slotCap
                             || job.generatedTokens.count >= max(0, slotCap - job.prompt.count)) {
                    finishJob(slot: s, job: job)
                }
            } catch {
                finishJob(slot: s, job: job, error: error)
            }
        }
    }

    /// Validates prompt fit, prefills an admitted job on its slot, and clamps output to remaining capacity.
    private func prefillAdmittedJob(slot s: Int, job: FNBatchJob, sCap4: Int) throws {
        if job.inBatchMode {
            activateSlot(s, totalSlots: Self.maxSlots)
            job.session.useSpec = (s == 0) ? job.useSpec : false
        } else {
            cleanupExtraSlotsBeforeSingleMode()
            activateSlot(0, totalSlots: 1)
        }
        let activeCap = job.inBatchMode ? min(sCap4, engine.activeCapacity) : engine.activeCapacity
        guard job.prompt.count < activeCap else {
            throw FNGPUError.contextDoesNotFit(requested: job.prompt.count + 1,
                                                maxFitting: max(0, activeCap - 1))
        }
        let effectiveMaxNew = QWBatchPartitionSelection.clampedMaxNew(
            requested: job.maxNew, promptCount: job.prompt.count, activeCapacity: activeCap
        )
        if !engine.hasPrefixCacheTier {
            try engine.preparePrefill(forTokenCount: job.prompt.count)
        }
        engine.beginSession(&job.session, maxNew: effectiveMaxNew)
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
                if s == 0 {
                    onSlot0CacheEvaluated?(job.session)
                }
            }
            chunkStart = end
        }
        job.prefillSeconds = Date().timeIntervalSince(prefillStart)
        if job.inBatchMode { job.session.useSpec = false }
        let syncSlot = job.inBatchMode ? s : 0
        syncSlotFromEngine(syncSlot)
        slotStates[syncSlot].highWaterPos = engine.highWaterPos
        job.position = job.prompt.count
        if effectiveMaxNew <= 0 {
            engine.flushPendingPrefixCache(&job.session)
            job.decodeStart = Date()
            finishJob(slot: syncSlot, job: job)
        } else {
            let done = emitAndCheckDone(token: firstTok, slot: syncSlot, job: job)
            engine.flushPendingPrefixCache(&job.session)
            job.decodeStart = Date()
            if !done && job.generatedTokens.count >= effectiveMaxNew {
                finishJob(slot: syncSlot, job: job)
            }
        }
    }

    private func beginInterleavedPrefillJob(slot s: Int, job: FNBatchJob, sCap4: Int) throws {
        activateSlot(s, totalSlots: Self.maxSlots)
        job.session.useSpec = false
        let activeCap = min(sCap4, engine.activeCapacity)
        guard job.prompt.count < activeCap else {
            throw FNGPUError.contextDoesNotFit(requested: job.prompt.count + 1,
                                                maxFitting: max(0, activeCap - 1))
        }
        let effectiveMaxNew = QWBatchPartitionSelection.clampedMaxNew(
            requested: job.maxNew, promptCount: job.prompt.count, activeCapacity: activeCap
        )
        if !engine.hasPrefixCacheTier {
            try engine.preparePrefill(forTokenCount: job.prompt.count)
        }
        engine.beginSession(&job.session, maxNew: effectiveMaxNew)
        syncSlotFromEngine(s)
        job.effectiveMaxNew = effectiveMaxNew
        job.prefillCursor = 0
        job.prefillWallAcc = 0
        job.isPrefilling = true
    }

    private func stepInterleavedPrefillSlice(slot s: Int, job: FNBatchJob, hasDecodingPeers: Bool) throws {
        activateSlot(s, totalSlots: Self.maxSlots)
        job.session.plePrecomputed = false
        let fullChunk = engine.dynamicPrefillChunkRows(forTotalTokens: job.prompt.count)
        let configuredSlice = QWInterleavedPrefillGate.sliceTokens
        let stepRows = hasDecodingPeers && configuredSlice > 0 ? min(configuredSlice, fullChunk) : fullChunk
        let start = job.prefillCursor
        let end = min(start + max(1, stepRows), job.prompt.count)
        let isFirst = (start == 0)
        let isLast = (end == job.prompt.count)
        let t0 = Date()
        let firstTok = try engine.encodePrefillChunk(&job.session, chunk: start..<end, runsHead: isLast)
        job.prefillWallAcc += Date().timeIntervalSince(t0)
        if isFirst {
            onCacheEvaluated?(job.session)
            if s == 0 { onSlot0CacheEvaluated?(job.session) }
        }
        syncSlotFromEngine(s)
        slotStates[s].highWaterPos = engine.highWaterPos
        job.prefillCursor = isLast ? end : max(end, min(max(0, job.prompt.count - 1), job.session.prefillPos))
        guard isLast else { return }

        job.isPrefilling = false
        job.prefillSeconds = job.prefillWallAcc
        job.position = job.prompt.count
        if job.effectiveMaxNew <= 0 {
            engine.flushPendingPrefixCache(&job.session)
            job.decodeStart = Date()
            finishJob(slot: s, job: job)
        } else {
            let done = emitAndCheckDone(token: firstTok, slot: s, job: job)
            engine.flushPendingPrefixCache(&job.session)
            job.decodeStart = Date()
            if !done && job.generatedTokens.count >= job.effectiveMaxNew {
                finishJob(slot: s, job: job)
            }
        }
    }
}
