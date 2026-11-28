// DriverBatchDecode.swift — Multi-session continuous batched decode for QWEngineDriver.

import Foundation
import Metal
import ObjectiveC
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWKV
import QWHead
import QWMTP

/// Preallocated 8-row head output and argmax scratch shared across batched steps.
final class QWBatchHeadScratch {
    static let maxBatchRows = 8
    let buffer: any MTLBuffer
    let vocab: Int
    let argmaxSlots: Int
    private let region: QWMRegion
    private let logitsOffset: Int
    private let keysHiOffset: Int
    private let keysLoOffset: Int
    private let tokensOffset: Int

    init(hal: QWHal, vocab: Int) throws {
        func align256(_ bytes: Int) -> Int { (bytes + 255) / 256 * 256 }
        let slots = max(1, max(QWHead.argmaxSlots(vocab), QWHead.batchedArgmaxSlots(vocab)))
        let rows = Self.maxBatchRows
        let logitsBytes = align256(rows * max(1, vocab) * 4)
        let keysBytes = align256(rows * slots * 4)
        let tokensBytes = align256(rows * 4)
        let total = logitsBytes + 2 * keysBytes + tokensBytes
        guard let buf = hal.device.makeBuffer(length: total, options: .storageModeShared) else {
            throw QWEngineDriverError.state("failed to allocate batched head scratch")
        }
        self.buffer = buf; self.vocab = vocab; self.argmaxSlots = slots
        self.region = QWMRegion(buffer: buf, base: 0, length: total)
        self.logitsOffset = 0; self.keysHiOffset = logitsBytes
        self.keysLoOffset = logitsBytes + keysBytes; self.tokensOffset = logitsBytes + 2 * keysBytes
    }

    private func rowTensor(_ base: Int, stride: Int, _ row: Int, _ count: Int, _ dt: QWDType) -> QWTensorValue {
        QWTensorValue(region: region, offset: base + row * stride, shape: QWTensorShape([count], dt))
    }

    func logitsView(rows: Int) -> QWTensorValue { rowTensor(logitsOffset, stride: 0, 0, rows * vocab, .f32) }
    func logitRow(_ row: Int) -> QWTensorValue { rowTensor(logitsOffset, stride: vocab * 4, row, vocab, .f32) }
    func keysHiView(rows: Int) -> QWTensorValue { rowTensor(keysHiOffset, stride: 0, 0, rows * argmaxSlots, .i32) }
    func keysHiRow(_ row: Int) -> QWTensorValue { rowTensor(keysHiOffset, stride: argmaxSlots * 4, row, argmaxSlots, .i32) }
    func keysLoView(rows: Int) -> QWTensorValue { rowTensor(keysLoOffset, stride: 0, 0, rows * argmaxSlots, .i32) }
    func keysLoRow(_ row: Int) -> QWTensorValue { rowTensor(keysLoOffset, stride: argmaxSlots * 4, row, argmaxSlots, .i32) }
    func tokensView(rows: Int) -> QWTensorValue { rowTensor(tokensOffset, stride: 0, 0, rows, .i32) }
    func tokenRow(_ row: Int) -> QWTensorValue { rowTensor(tokensOffset, stride: 4, row, 1, .i32) }

    func readToken(_ row: Int) -> Int {
        Int(buffer.contents().advanced(by: tokensOffset + row * 4).assumingMemoryBound(to: Int32.self).pointee)
    }

    func logitBuffer(_ row: Int) -> UnsafeBufferPointer<Float> {
        let ptr = buffer.contents().advanced(by: logitsOffset + row * vocab * 4).assumingMemoryBound(to: Float.self)
        return UnsafeBufferPointer(start: ptr, count: vocab)
    }
}

/// Thread-safe handle for one submitted request tracked across a batch wave.
final class QWBatchScheduledJob: @unchecked Sendable {
    let request: QWGenerationRequest
    let onToken: ((Int) -> Bool)?
    private let promise = QWBatchJobPromise<QWGenerationResult>()

    init(request: QWGenerationRequest, onToken: ((Int) -> Bool)?) {
        self.request = request
        self.onToken = onToken
    }

    func complete(_ result: Result<QWGenerationResult, Error>) {
        promise.complete(result)
    }

    func wait() throws -> QWGenerationResult {
        do {
            return try promise.wait()
        } catch QWBatchSchedulerError.missingJobOutcome {
            throw QWEngineDriverError.state("batch job finished without an outcome")
        }
    }
}

extension QWEngineDriver {
    nonisolated(unsafe) private static var batchHeadScratchAssocKey: UInt8 = 0
    nonisolated(unsafe) private static var batchMLPAssocKey: UInt8 = 0

    var canRunBatchedDecode: Bool {
        hcaOperator == nil && echoDrafter == nil && mbsaOperator == nil
            && psc.maxChunk >= QWBatchHeadScratch.maxBatchRows
            && geometry.vocab > 0
            && (geometry.mhcConfig == nil || (psc.mhcStream != nil && psc.mhcControls != nil))
            && (geometry.gdnLayerCount == 0 || psc.dn32.shape.elementCount >= QWBatchHeadScratch.maxBatchRows * 2 * geometry.gdnVHeads)
    }

    func sharedBatchHeadScratch() throws -> QWBatchHeadScratch {
        try slotLock.withLock {
            if let existing = objc_getAssociatedObject(self, &Self.batchHeadScratchAssocKey) as? QWBatchHeadScratch {
                return existing
            }
            let created = try QWBatchHeadScratch(hal: hal, vocab: geometry.vocab)
            objc_setAssociatedObject(self, &Self.batchHeadScratchAssocKey, created, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
            return created
        }
    }

    func sharedBatchMLP() throws -> QWMTPBatchedMLP {
        if let existing = mtpBatchedMLP, existing.rows >= QWBatchHeadScratch.maxBatchRows {
            return existing
        }
        return try slotLock.withLock {
            if let existing = objc_getAssociatedObject(self, &Self.batchMLPAssocKey) as? QWMTPBatchedMLP {
                return existing
            }
            let bytes = QWMTPBatchedMLP.allocationBytes(geometry: geometry, rows: QWBatchHeadScratch.maxBatchRows)
            guard let buf = hal.device.makeBuffer(length: bytes + 1024, options: .storageModeShared) else {
                throw QWEngineDriverError.state("failed to allocate batched MLP scratch")
            }
            let alloc = QWSubAllocator(region: QWMRegion(buffer: buf, base: 0, length: bytes + 1024))
            let created = try QWMTPBatchedMLP(geometry: geometry, hal: hal, allocator: alloc, rows: QWBatchHeadScratch.maxBatchRows)
            objc_setAssociatedObject(self, &Self.batchMLPAssocKey, created, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
            return created
        }
    }

    var batchScLogitsBuffer: UnsafeBufferPointer<Float> {
        let ptr = sc.logits.region.pointer.advanced(by: sc.logits.offset).assumingMemoryBound(to: Float.self)
        return UnsafeBufferPointer(start: ptr, count: geometry.vocab)
    }

    /// Executes one batched trunk + LM head step across B=2...8 active sessions in one command buffer.
    func forwardBatchStep(
        _ sessions: [QWEngineSession], tokens: [Int],
        permit: QWGenerationPermit? = nil, headScratch: QWBatchHeadScratch? = nil,
        greedyOnly: Bool = false
    ) throws -> (rawIds: [Int], stepMs: Double, scratch: QWBatchHeadScratch) {
        let B = sessions.count
        guard (2...QWBatchHeadScratch.maxBatchRows).contains(B), tokens.count == B else {
            throw QWEngineDriverError.state("batched decode requires B=2...\(QWBatchHeadScratch.maxBatchRows) matching tokens")
        }
        guard canRunBatchedDecode else {
            throw QWEngineDriverError.state("model geometry/operators require quantum-interleaved batching")
        }
        return try withFlightPermit(permit) {
            try forwardBatchStepLocked(sessions, tokens: tokens, headScratch: headScratch, greedyOnly: greedyOnly)
        }
    }

    private func forwardBatchStepLocked(
        _ sessions: [QWEngineSession], tokens: [Int], headScratch: QWBatchHeadScratch?, greedyOnly: Bool
    ) throws -> (rawIds: [Int], stepMs: Double, scratch: QWBatchHeadScratch) {
        let arb = gpuTurnArbitrator
        arb?.acquirePrimaryStepSync()
        defer { arb?.releasePrimaryStep() }
        let B = sessions.count
        guard verifiedSession == nil else {
            throw QWEngineDriverError.state("resolve pending verification before batched decode")
        }
        let tStep0 = Date()
        for b in 0..<B {
            let sess = sessions[b]
            guard !sess.sessionFinished, !sess.stateFailed, !sess.verificationPending else {
                throw QWEngineDriverError.state("cannot run batched step on finished or failed session")
            }
            let pos = sess.position
            guard pos < ropeRows else { throw QWEngineDriverError.posBeyondTable(pos) }
            guard tokens[b] >= 0, tokens[b] < geometry.vocab else { throw QWEngineDriverError.badToken(tokens[b]) }
            try maybePrefixPagePressure(session: sess) {
                if sess.kv.tokenCapacity <= pos {
                    try kvCache.grow(sess.kv, to: pos + QWKVCache.pageTokens)
                    sess.attnKVViews = try kvCache.layerViewsAll(sess.kv, layers: attnLayers)
                }
                try kvCache.prepareWrites(sess.kv, positions: pos..<(pos + 1))
            }
        }
        let scratch = try headScratch ?? sharedBatchHeadScratch()
        try prepareLSASessionsForBatchStep(sessions)
        try prepareMLASessionsForBatchStep(sessions)
        try copyEmbedRows(tokens, start: 0, count: B, to: prefixRows(psc.x, rows: B, cols: geometry.hidden, dt: .f16))
        if geometry.sharedExpertIntermediate > 0 && !geometry.sharedExpertHasGate, let s = psc.moe {
            let gPtr = s.gate.region.pointer.advanced(by: s.gate.offset).assumingMemoryBound(to: Float.self)
            for i in 0..<B { gPtr[i] = 1.0 }
        }
        guard let pair = hal.begin() else { throw QWEngineDriverError.state("cmdbuffer") }
        var cb: any MTLCommandBuffer = pair.0, enc: any MTLComputeCommandEncoder = pair.1
        activeCB = cb; activeEnc = enc; stepCBs = []
        var encoderEnded = false, committedFinalCB = false, stepCBsDrained = false
        defer {
            if !encoderEnded { (activeEnc ?? enc).endEncoding() }
            if !stepCBsDrained {
                if committedFinalCB { cb.waitUntilCompleted() }
                for sliceCb in stepCBs ?? [] { sliceCb.waitUntilCompleted() }
                if let pipe = coldEvents {
                    for bcb in pipe.bridgeCBs { bcb.waitUntilCompleted() }
                    pipe.shutdown(); coldEvents = nil
                }
                flushPendingColdUnloads()
            }
            activeCB = nil; activeEnc = nil; stepCBs = nil
        }
        for (v, p) in geometry.executionPlan.enumerated() {
            guard let denseKernel = layerKernels[p] as? DenseLayerKernel else {
                throw QWEngineDriverError.state("non-dense layer kernel encountered in batched decode")
            }
            try denseKernel.encodeBatchDecode(activeEnc ?? enc, driver: self, sessions: sessions, layer: v, weights: layersW[p])
        }
        cb = activeCB ?? cb; enc = activeEnc ?? enc
        try encodeBatchedHead(enc, scratch: scratch, count: B, greedyOnly: greedyOnly)
        encoderEnded = true
        hal.end(cb, enc)
        committedFinalCB = true
        let allCbs = (stepCBs?.isEmpty == false) ? (stepCBs! + [cb]) : [cb]
        defer { stepCBsDrained = true }
        try drainBatchStepCommandBuffers(allCbs)
        var rawIds: [Int] = []
        rawIds.reserveCapacity(B)
        for b in 0..<B {
            let pos = sessions[b].position
            sessions[b].position += 1; sessions[b].generated += 1
            sessions[b].batchedDecodeTouched = true; sessions[b].mlaIndexerStash = nil
            kvCache.markWritten(sessions[b].kv, upTo: pos + 1)
            if geometry.hasAnySlidingWindow {
                try kvCache.recycleExpiredWindowPages(
                    sessions[b].kv,
                    currentPosition: pos + 1,
                    safetyMargin: geometry.descriptor.maxSpeculativeDrafts + QWKVCache.pageTokens)
            }
            rawIds.append(scratch.readToken(b))
        }
        return (rawIds, Date().timeIntervalSince(tStep0) * 1000, scratch)
    }

    private func drainBatchStepCommandBuffers(_ cbs: [any MTLCommandBuffer]) throws {
        for sliceCb in cbs { sliceCb.waitUntilCompleted() }
        let coldFailure: (any Error)? = coldEvents.flatMap { pipe in
            let r = pipe.takeStepResult()
            pipe.tuneBridge()
            return r.failure
        }
        flushPendingColdUnloads()
        if let coldFailure { throw QWEngineDriverError.state("cold event staging: \(coldFailure)") }
        for sliceCb in cbs {
            if let err = sliceCb.error {
                commandBufferErrors += 1
                throw QWEngineDriverError.state("batched decode cb: \(err.localizedDescription)")
            }
            if sliceCb.status != .completed {
                throw QWEngineDriverError.state("batched decode cb did not complete")
            }
        }
    }

    private func encodeBatchedHead(
        _ enc: any MTLComputeCommandEncoder,
        scratch: QWBatchHeadScratch,
        count B: Int,
        greedyOnly: Bool = false
    ) throws {
        enc.useResource(scratch.buffer, usage: [.read, .write])
        let xBatch = prefixRows(psc.ln1, rows: B, cols: geometry.hidden, dt: .f16)
        let needsHeadFwht: Bool = switch headStorage {
        case .f16: true
        case .bf16: hal.weightBasis.rotatesActivations
        case .q4g: false
        }
        if needsHeadFwht {
            encodeFwht32Fast(enc, x: xBatch)
            enc.memoryBarrier(scope: .buffers)
        }
        let softcap = geometry.finalLogitSoftcapping ?? 0.0
        if greedyOnly && multimaxParams == nil {
            try head.encodeBatchedGreedyArgmax(
                enc, storage: headStorage, x: xBatch,
                keysHi: scratch.keysHiView(rows: B),
                keysLo: scratch.keysLoView(rows: B),
                tokens: scratch.tokensView(rows: B),
                M: B, N: geometry.vocab, softcap: softcap
            )
            return
        }
        let logitsBatch = scratch.logitsView(rows: B)
        try head.encode(enc, storage: headStorage, x: xBatch, gs: nil, y: logitsBatch, M: B, softcap: softcap)
        if let mm = multimaxParams {
            enc.memoryBarrier(scope: .buffers)
            try head.encodeMultiMax(enc, logits: logitsBatch, params: mm, count: B * geometry.vocab)
        }
        enc.memoryBarrier(scope: .buffers)
        for b in 0..<B {
            try head.encodeArgmax(enc, y: scratch.logitRow(b), keysHi: scratch.keysHiRow(b),
                                  keysLo: scratch.keysLoRow(b), token: scratch.tokenRow(b), N: geometry.vocab)
            enc.memoryBarrier(scope: .buffers)
        }
    }

    // MARK: - Continuous multi-session scheduler loop (B=1..8 + slice-interleaved prefill)

    private func runScheduledBatchLocked(
        initialJobs: [QWBatchScheduledJob],
        headScratch: QWBatchHeadScratch?,
        permit: QWGenerationPermit,
        pollPending: (_ freeSlots: Int) -> [QWBatchScheduledJob],
        requeueDelayed: ([QWBatchScheduledJob]) -> Void
    ) {
        let scratch = headScratch ?? (try? sharedBatchHeadScratch())
        var active: [QWBatchSlotState] = []
        defer { for slot in active where !slot.session.sessionFinished { finishSession(slot.session) } }
        var wave = QWBatchWaveState()

        func drainActiveCommits() throws {
            for idx in active.indices {
                try awaitCommitAndCheck(active[idx].session, context: "before multi-slot batch transition")
            }
        }
        func failSlot(at idx: Int, error: any Error) {
            let failed = active.remove(at: idx)
            finishSession(failed.session)
            failed.job.complete(.failure(error))
            wave.clearBlock()
        }
        func completeSlot(at idx: Int) {
            let slot = active.remove(at: idx)
            do {
                try awaitCommitAndCheck(slot.session, context: "before completing batch slot")
                finishAndCompleteSlot(slot)
            } catch {
                finishSession(slot.session)
                slot.job.complete(.failure(error))
            }
            wave.clearBlock()
        }
        func pruneSlots(where predicate: (QWBatchSlotState) -> Bool) {
            for b in stride(from: active.count - 1, through: 0, by: -1) where predicate(active[b]) {
                completeSlot(at: b)
            }
        }
        func admitQueue(_ jobs: [QWBatchScheduledJob]) {
            guard !jobs.isEmpty else { return }
            do {
                try drainActiveCommits()
            } catch {
                for b in stride(from: active.count - 1, through: 0, by: -1) { failSlot(at: b, error: error) }
                for job in jobs { job.complete(.failure(error)) }
                return
            }
            let waveCommonPrefix = Self.commonWavePrefixLength(for: jobs)
            var waveLeaderUsed = false
            var deferred: [QWBatchScheduledJob] = []
            for job in jobs {
                if !deferred.isEmpty { deferred.append(job); continue }
                let hasDecodingPeers = active.contains(where: \.isDecoding)
                let isLeader = !waveLeaderUsed && job.request.usePrefixCache && job.request.visualSpans.isEmpty
                if isLeader { waveLeaderUsed = true }
                do {
                    if let slot = try admitBatchJobForScheduling(
                        job, commonWavePrefix: waveCommonPrefix, isWaveLeader: isLeader,
                        hasDecodingPeers: hasDecodingPeers, permit: permit
                    ) {
                        if slot.output.finished { finishAndCompleteSlot(slot) } else { active.append(slot) }
                    }
                } catch let kvErr as QWKVError where !active.isEmpty {
                    if case .outOfPages = kvErr { deferred.append(job) } else { job.complete(.failure(kvErr)) }
                } catch {
                    job.complete(.failure(error))
                }
            }
            if deferred.isEmpty { wave.clearBlock() }
            else { wave.markBlocked(freeWatermark: kvCache.freePageCount); requeueDelayed(deferred) }
        }

        admitQueue(initialJobs)
        while !active.isEmpty {
            wave.refreshBlock(currentFree: kvCache.freePageCount)
            let openSlots = slotLock.withLock { freeSlots.count }
            if wave.canPollPending(openSlots: openSlots) {
                let newlyPolled = pollPending(openSlots)
                if !newlyPolled.isEmpty { admitQueue(newlyPolled) }
            }
            pruneSlots { $0.isDecoding && $0.session.position >= ropeRows }
            if active.isEmpty { break }

            let decodingIndices = active.indices.filter { active[$0].isDecoding }
            if !decodingIndices.isEmpty {
                let localEligible = eligibleBatchSlotIndices(
                    slots: decodingIndices.map { idx in
                        (active[idx].session.position, active[idx].maxNew - active[idx].output.generated.count)
                    },
                    maxRows: QWBatchHeadScratch.maxBatchRows
                )
                let stepIndices = localEligible.map { decodingIndices[$0] }
                let decRoundRobin = decodingIndices[wave.roundRobinIndex % decodingIndices.count]
                let useBatched = QWBatchPartitionSelection.shouldRunBatchedStep(
                    stepIndices: stepIndices, activeCount: decodingIndices.count, roundRobinIndex: decRoundRobin,
                    canRunBatchedDecode: canRunBatchedDecode,
                    hasSpeculative: stepIndices.contains { canSpeculateSlot(active[$0]) },
                    allowsSpeculativeBatch: geometry.numExperts == 0
                )
                if useBatched {
                    do { try drainActiveCommits() } catch {
                        for slotIdx in stepIndices.sorted(by: >) { failSlot(at: slotIdx, error: error) }
                        continue
                    }
                    let greedyOnly = multimaxParams == nil && stepIndices.allSatisfy { idx in
                        let slot = active[idx]
                        return slot.lpSampler == nil && slot.constraint == nil && slot.session.stochastic == nil
                    }
                    let stepResult: (rawIds: [Int], stepMs: Double, scratch: QWBatchHeadScratch)
                    do {
                        stepResult = try forwardBatchStep(
                            stepIndices.map { active[$0].session }, tokens: stepIndices.map { active[$0].pendingToken },
                            permit: permit, headScratch: scratch, greedyOnly: greedyOnly)
                    } catch {
                        for slotIdx in stepIndices.sorted(by: >) { failSlot(at: slotIdx, error: error) }
                        continue
                    }
                    var failed: [(Int, any Error)] = []
                    for (batchIdx, slotIdx) in stepIndices.enumerated() {
                        active[slotIdx].output.markDecodeStartedIfNeeded()
                        do {
                            try consumeBatchPredictedToken(
                                slot: &active[slotIdx], rawArgmax: stepResult.rawIds[batchIdx],
                                logitsTensor: stepResult.scratch.logitRow(batchIdx),
                                logitBuf: stepResult.scratch.logitBuffer(batchIdx), stepMsValue: stepResult.stepMs)
                        } catch { failed.append((slotIdx, error)) }
                    }
                    for (slotIdx, slotError) in failed.sorted(by: { $0.0 > $1.0 }) { failSlot(at: slotIdx, error: slotError) }
                    pruneSlots { $0.output.finished }
                    wave.advanceRoundRobin(
                        stepIndices: localEligible, activeCount: decodingIndices.count,
                        mixedShortAndLong: localEligible.count < decodingIndices.count
                    )
                } else {
                    let decPos = wave.roundRobinIndex % decodingIndices.count
                    let idx = decodingIndices[decPos]
                    let allowAsyncSolo = (active.count == 1) && (QWBatchAdmissionCounter.preparingCount == 0)
                    let quantum = QWBatchPartitionSelection.serialQuantum(
                        canRunBatchedDecode: canRunBatchedDecode, activeCount: decodingIndices.count,
                        canSpeculateCurrent: canSpeculateSlot(active[idx])
                    )
                    if !allowAsyncSolo {
                        do { try drainActiveCommits() } catch {
                            failSlot(at: idx, error: error)
                            continue
                        }
                    }
                    for _ in 0..<quantum where idx < active.count && !active[idx].output.finished {
                        if active[idx].session.position >= ropeRows { active[idx].output.finished = true; break }
                        do {
                            try stepSingleSlot(&active[idx], allowAsyncCommit: allowAsyncSolo, permit: permit)
                        } catch {
                            failSlot(at: idx, error: error)
                            break
                        }
                    }
                    if idx < active.count, active[idx].output.finished {
                        completeSlot(at: idx)
                        wave.finishSerialSlotRemoved(activeCount: active.filter(\.isDecoding).count)
                    } else {
                        wave.advanceSerialSlot(activeCount: active.filter(\.isDecoding).count)
                    }
                }
            }

            if let pfIdx = active.firstIndex(where: { !$0.isDecoding }) {
                let hasDecodingNow = active.contains(where: \.isDecoding)
                    || QWBatchAdmissionCounter.preparingCount > 0
                do {
                    try drainActiveCommits()
                    try stepPrefillingSlot(&active[pfIdx], hasDecodingPeers: hasDecodingNow, permit: permit)
                    if active[pfIdx].output.finished {
                        completeSlot(at: pfIdx)
                    }
                } catch let kvErr as QWKVError where active.count > 1 {
                    if case .outOfPages = kvErr {
                        let delayed = active.remove(at: pfIdx)
                        finishSession(delayed.session)
                        wave.markBlocked(freeWatermark: kvCache.freePageCount)
                        requeueDelayed([delayed.job])
                    } else {
                        failSlot(at: pfIdx, error: kvErr)
                    }
                } catch {
                    failSlot(at: pfIdx, error: error)
                }
            }
        }
    }

    func runScheduledBatch(
        initialJobs: [QWBatchScheduledJob],
        headScratch _: QWBatchHeadScratch? = nil,
        pollPending: (_ freeSlots: Int) -> [QWBatchScheduledJob],
        requeueDelayed: ([QWBatchScheduledJob]) -> Void
    ) {
        trunkExecutor.runScheduledBatch(
            initialJobs: initialJobs,
            pollPending: pollPending,
            requeueDelayed: requeueDelayed
        )
    }

    func runStandardScheduledBatch(
        initialJobs: [QWBatchScheduledJob],
        headScratch: QWBatchHeadScratch? = nil,
        pollPending: (_ freeSlots: Int) -> [QWBatchScheduledJob],
        requeueDelayed: ([QWBatchScheduledJob]) -> Void
    ) {
        withDriverGenerateLock {
            let permit: QWGenerationPermit
            do { permit = try acquireRequest() } catch {
                for job in initialJobs { job.complete(.failure(error)) }
                return
            }
            defer { releaseRequest(permit) }
            runScheduledBatchLocked(
                initialJobs: initialJobs, headScratch: headScratch, permit: permit,
                pollPending: pollPending, requeueDelayed: requeueDelayed)
        }
    }

    private func canSpeculateSlot(_ slot: QWBatchSlotState) -> Bool {
        slot.isDecoding && QWBatchPartitionSelection.isSlotSpeculatable(
            speculativeEnabled: slot.speculativeEnabled,
            remaining: slot.maxNew - slot.output.generated.count,
            hasLogprobsSampler: slot.lpSampler != nil,
            isUnconstrainedNow: slot.constraint?.isUnconstrainedNow ?? true,
            hasConstraint: slot.constraint != nil,
            hasConstraintSampler: slot.constraintSampler != nil,
            canSpeculateConstrained: canSpeculateConstraint(slot.constraint)
        ) && canRunSpeculativeRound(slot.session, constraintOverride: slot.constraint)
    }

    private func stepSingleSlot(
        _ slot: inout QWBatchSlotState,
        allowAsyncCommit: Bool,
        permit: QWGenerationPermit
    ) throws {
        slot.output.markDecodeStartedIfNeeded()
        let saved = (activeConstraint, activeCollectLogprobs, activeRelaxedMarginOverride, mtpRelaxedMinProb)
        defer {
            (activeConstraint, activeCollectLogprobs, activeRelaxedMarginOverride, mtpRelaxedMinProb) = saved
        }
        activeRelaxedMarginOverride = slot.relaxedMarginOverride
        if let minP = slot.relaxedMinProb { mtpRelaxedMinProb = minP }
        let remaining = slot.maxNew - slot.output.generated.count
        if canSpeculateSlot(slot) {
            activeConstraint = slot.constraint.flatMap { c in slot.constraintSampler.map { (c, $0) } }
            activeCollectLogprobs = false
            let t0 = Date()
            let decision = try executeSpeculativeRoundStep(
                slot.session, remaining: remaining, stopIDs: slot.stopIDs,
                drain: !allowAsyncCommit,
                waitForStreamCommit: allowAsyncCommit && slot.job.onToken != nil,
                store: prefixStore, permit: permit
            )
            if let next = decision.pendingToken { slot.pendingToken = next }
            slot.output.recordSpeculativeRound(
                tokens: decision.tokens, stepMsValue: Date().timeIntervalSince(t0) * 1000,
                maxNew: slot.maxNew, stopped: decision.stopped, stoppedToken: decision.stoppedToken,
                onToken: slot.job.onToken
            )
            return
        }
        activeConstraint = nil
        activeCollectLogprobs = (slot.lpSampler != nil || slot.constraint != nil || slot.session.stochastic != nil)
        let t0 = Date()
        let nextArgmax: Int = try withFlightPermit(permit) {
            guard let id = try forwardStep(slot.session, token: slot.pendingToken) else {
                throw QWEngineDriverError.state("serial forward step returned nil")
            }
            return id
        }
        try consumeBatchPredictedToken(
            slot: &slot, rawArgmax: nextArgmax, logitsTensor: sc.logits,
            logitBuf: batchScLogitsBuffer, stepMsValue: Date().timeIntervalSince(t0) * 1000
        )
    }

    private func selectBatchToken(
        slot: QWBatchSlotState, rawArgmax: Int, logitsTensor: QWTensorValue,
        logitBuf: UnsafeBufferPointer<Float>, repArr: [Int]
    ) throws -> Int {
        func sampleStochastic(_ owner: QWStochasticSession) throws -> Int {
            try owner.stream.sampleTarget(owner.distribution(logits: logitsTensor, repetitionHistory: repArr))
        }
        if let constraint = slot.constraint, let sampler = slot.constraintSampler {
            let chosen: Int
            if let allowed = try constraint.allowedTokens() {
                let rep = sampler.config.repetitionPenalty == 1.0 ? Set<Int>() : Set(repArr)
                chosen = try sampler.constrainedToken(logits: logitBuf, allowed: allowed, repetitionHistory: rep)
            } else if let owner = slot.session.stochastic {
                chosen = try sampleStochastic(owner)
            } else if sampler.config.isGreedy {
                chosen = sampler.config.repetitionPenalty == 1.0 ? rawArgmax
                    : try sampler.penalisedGreedyToken(logits: logitBuf, repetitionHistory: Set(repArr))
            } else {
                chosen = try sampler.sampleToken(
                    logits: logitBuf, repetitionHistory: sampler.config.repetitionPenalty == 1.0 ? [] : repArr
                )
            }
            try constraint.accept(chosen)
            return chosen
        }
        return try slot.session.stochastic.map(sampleStochastic) ?? rawArgmax
    }

    func consumeBatchPredictedToken(
        slot: inout QWBatchSlotState, rawArgmax: Int, logitsTensor: QWTensorValue,
        logitBuf: UnsafeBufferPointer<Float>, stepMsValue: Double?
    ) throws {
        let repArr = slot.session.activeRepetitionHistory()
        let chosen = try selectBatchToken(
            slot: slot, rawArgmax: rawArgmax, logitsTensor: logitsTensor, logitBuf: logitBuf, repArr: repArr
        )
        if let owner = slot.session.stochastic {
            slot.session.roundRecord.samplerDrawIndex = owner.stream.drawsConsumed
        }
        slot.pendingToken = chosen
        slot.session.roundRecord.pendingToken = chosen
        slot.session.roundRecord.outputCount = slot.session.generated
        guard slot.output.beginPredictedToken(chosen, stopIDs: slot.stopIDs, stepMsValue: stepMsValue) else { return }
        if let lpSampler = slot.lpSampler {
            let lp = lpSampler.computeTokenLogprob(logits: logitBuf, chosenToken: chosen, topK: slot.topKCount)
            slot.logprobs.append(lp)
            slot.job.request.onTokenLogprob?(lp)
        }
        if (slot.constraint?.isUnconstrainedNow ?? true), let prev1 = slot.session.tokenHistory.last {
            let prev2 = slot.session.tokenHistory.count >= 2
                ? slot.session.tokenHistory[slot.session.tokenHistory.count - 2] : nil
            slot.session.tokenRecycler.recordTransition(previous: prev2, current: prev1, next: chosen)
        }
        slot.session.tokenHistory.append(chosen)
        slot.output.finishPredictedToken(maxNew: slot.maxNew, accepted: slot.job.onToken?(chosen) ?? true)
    }

    func finishAndCompleteSlot(_ slot: QWBatchSlotState) {
        finishSession(slot.session)
        let decodeSec = slot.output.decodeDurationSeconds() ?? 0
        let summary = slot.output.speculativeSummary(
            enabled: slot.speculativeEnabled, proposals: mtpDraftProposals, accepted: mtpDraftAccepted
        )
        lastPromptTokens = slot.job.request.prompt.count
        lastStopToken = slot.output.stoppedToken
        lastRequestedBackend = slot.job.request.speculative.mtpBackend
        if slot.output.speculativeRounds > 0 {
            lastGeneration = nil
            lastExperimentalGeneration = QWExperimentalGenerationStats(
                promptTokens: slot.job.request.prompt.count,
                generatedTokens: slot.output.generated.count,
                prefillSec: slot.output.prefillSeconds,
                decodeSec: decodeSec,
                roundMs: slot.output.stepMs,
                roundTokenCounts: slot.output.stepTokenCounts
            )
        } else {
            lastExperimentalGeneration = nil
            lastGeneration = QWGenerationStats(
                promptTokens: slot.job.request.prompt.count,
                generatedTokens: slot.output.generated.count,
                prefillSec: slot.output.prefillSeconds,
                decodeSec: decodeSec,
                firstTokenSec: slot.output.firstTokenSeconds,
                stepMs: slot.output.stepMs
            )
        }
        if let cache = slot.cacheAnswer, let store = prefixStore {
            lastPrefixCacheStats = QWPrefixCacheStats(
                hit: cache.hit, matchedTokens: cache.matchedTokens,
                chargedBytes: store.chargedBytes, evictions: store.evictions,
                skippedOversize: false, tier: cache.tier
            )
        }
        slot.job.complete(.success(QWGenerationResult(
            tokens: slot.output.generated, stepMs: slot.output.stepMs,
            stepTokenCounts: slot.output.stepTokenCounts,
            prefillSeconds: slot.output.prefillSeconds,
            firstTokenSeconds: slot.output.firstTokenSeconds,
            decodeSeconds: slot.output.decodeDurationSeconds(),
            prefillGPUSeconds: slot.output.prefillGPUSeconds > 0 ? slot.output.prefillGPUSeconds : nil,
            stopTokenID: slot.output.stoppedToken, cache: slot.cacheAnswer,
            speculativeSummary: summary, tokenLogprobs: slot.logprobs
        )))
    }
}
