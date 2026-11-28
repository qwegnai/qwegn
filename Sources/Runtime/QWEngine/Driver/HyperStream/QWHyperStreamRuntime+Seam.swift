// QWHyperStreamRuntime+Seam.swift — Prefill stack residency and module seam on QWHyperStreamRuntime.

import Foundation
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWHead
import QWQuant
import QWAttn
import QWHyperPLE

extension QWHyperStreamRuntime {
    /// Prefill arena budget: ~1.45 MB per prefill row plus tiled QSA score scratch.
    static func prefillActivationBytes(maxRows: Int, capacity: Int = 4096) -> Int {
        let qsaScoreBytes = max(min(maxRows, FNGPUScratch.defaultQSAScoreTileRows) * max(1, capacity / 4), min(maxRows, 4096) * 2048) * 4
        return maxRows * 1_450_000 + qsaScoreBytes + (16 << 20)
    }

    /// Computes actual resident GPU weight bytes from the checkpoint rather than a static placeholder.
    static func estimateWeightBytes(loader: FNPackLoader, useStandardBasis: Bool) -> Int {
        let hasHiBitDense = loader.pack?.tensors.values.contains(where: { $0.shape.count == 2 && $0.hqBits > 4 }) ?? false
        let minEstimate = (hasHiBitDense ? 3_408 : 2_384) << 20
        if let st = loader.st {
            var total = 0
            for t in st.tensors.values where !t.name.contains("ple_embed_tokens") {
                total += (!useStandardBasis && t.dtype == .bf16 && t.shape.count >= 2) ? (t.byteLength * 20) / 64 : t.byteLength
            }
            return max(minEstimate, total)
        }
        return minEstimate
    }

    private func buildPrefillStack(arena: QWArena, rows: Int) throws -> FNGPUPrefillStack {
        let firstRec = layerRecords[0]
        let sharedAttn = layerRecords.compactMap(\.attn).first
        let sharedGDN = layerRecords.compactMap(\.gdn).first
        let pscr = try FNGPUScratch(arena: arena, T: rows, hidden: hidden, hc: hc, lowrank: firstRec.lowrank,
                                    moe: firstRec.weights.moe, ple: firstRec.weights.ple,
                                    gdn: sharedGDN, attn: sharedAttn, extraEmbedDim: maxPLEDim, maxCapacity: capacity)
        var prefillSlots: [FNGPUPrefillStack.Slot] = []
        for rec in layerRecords {
            let pleBuf = try rec.weights.ple.map { try arena.activationsRegion().allocate(QWTensorShape([rows * $0.embedDim], .f32)) }
            let pleBufAlt = try (gpuPLETable == nil ? rec.weights.ple : nil).map {
                try arena.activationsRegion().allocate(QWTensorShape([rows * $0.embedDim], .f32))
            }
            prefillSlots.append(.init(weights: rec.weights, attn: rec.attnState, ple: rec.ple,
                                      pleEmbed: pleBuf, pleEmbedAlt: pleBufAlt, scratch: pscr))
        }
        return try FNGPUPrefillStack(arena: arena, ctx: ctx, slots: prefillSlots, hidden: hidden, hc: hc,
                                     gpuPLETable: gpuPLETable)
    }

    /// Dynamic prefill ceiling: respects configured `maxPrefillRows`.
    var dynamicMaxPrefillRows: Int { max(32, maxPrefillRows) }

    private func ensurePrefillStack(rows: Int, allowShrink: Bool = true) throws -> FNGPUPrefillStack {
        let cap = dynamicMaxPrefillRows
        let allocRows = max(32, min(cap, ((rows + 31) / 32) * 32))
        if let cached = cachedPrefill, rows <= cached.stack.maxRows,
           !(evictPrefillScratch && allowShrink && cached.rows > 1024 && allocRows <= 512) {
            cached.stack.setActiveRows(rows)
            return cached.stack
        }
        let neededBytes = Self.prefillActivationBytes(maxRows: allocRows, capacity: capacity)
        let pArena: QWArena
        if let existing = prefillArena, existing.regionSizes.act >= neededBytes,
           !(evictPrefillScratch && allowShrink
             && existing.regionSizes.act > Self.prefillActivationBytes(maxRows: 1024, capacity: capacity)
             && allocRows <= 512) {
            existing.activationsSub.reset()
            pArena = existing
        } else {
            cachedPrefill = nil; prefillArena = nil
            let arenaRows = (!evictPrefillScratch && allocRows > 1024) ? cap : allocRows
            let arenaBytes = Self.prefillActivationBytes(maxRows: arenaRows, capacity: capacity)
            let created = try QWArena(hal: hal, weights: 0, activations: arenaBytes, kv: 0, gdnStates: 0, keepAlive: false)
            created.prefault()
            prefillArena = created; pArena = created
        }
        let stk = try buildPrefillStack(arena: pArena, rows: allocRows)
        stk.setActiveRows(rows)
        cachedPrefill = (rows: allocRows, stack: stk)
        lastPrefillStackRows = allocRows
        return stk
    }

    /// Releases the temporary prefill scratch arena once prefill has completed when scratch eviction is enabled.
    public func evictPrefillScratchIfNeeded() {
        guard evictPrefillScratch else { return }
        cachedPrefill = nil; prefillArena = nil
    }

    /// Runs ONE prefill chunk and advances the session's cursor. The head
    /// runs on the last chunk, and its token is the first generated one.
    private func prefillChunk(_ session: inout FNGPUSessionData, history: [Int64], cacheHistory: [Int64],
                              chunkSize: Int, targetEnd: Int? = nil, gpuSec: inout Double) throws {
        let arb = gpuTurnArbitrator
        arb?.acquirePrimaryStepSync()
        defer { arb?.releasePrimaryStep() }
        let pos = session.prefillPos
        let limitPos = FNGPUDFlash.prefillWindowEnd(historyCount: history.count, targetEnd: targetEnd)
        let subchunk = FNGPUPrefillStack.resolveSubchunk(pos: pos, limitPos: limitPos, chunkSize: chunkSize,
                                                         hidden: hidden, hc: hc, visualSpans: session.visualSpans)
        let rows = subchunk.rows
        let isLast = (pos + rows == history.count)
        let isCheckpointEnd = (targetEnd != nil && pos + rows == limitPos && limitPos < history.count)
        let runHeadNow = isLast || isCheckpointEnd
        let runMTP = (session.useSpec || (hasPrefixCacheTier && session.usePrefixCache)) && mtp != nil
        let runDFlash = (session.useSpec || (hasPrefixCacheTier && session.usePrefixCache)) && dflash != nil
        session.lastRows = rows
        if rows == 1 {
            // Same single-row staging: the id goes into head.token, the gather runs on-device.
            arena.write(head.token, from: [Int32(history[pos])])
            session.headToken = Int(history[pos])
            let prefix = Array(history.prefix(pos + 1))
            let df = runDFlash ? dflash : nil
            let dfSlot = df?.sessionSlots.first
            if let streamIn = subchunk.visualStreamIn {
                session.lastArgmax = try decodeStack.forwardToken(
                    history: prefix, streamIn: streamIn, head: head, readStream: false,
                    dflash: df, dflashSlot: dfSlot
                ).token
            } else {
                session.lastArgmax = try decodeStack.forwardToken(
                    history: prefix, embedTable: embedTable, head: head, readStream: false,
                    dflash: df, dflashSlot: dfSlot
                ).token
            }
            if df != nil {
                session.dflashCacheOffset = pos + 1
                session.dflashPendingCtxCount = 0
            }
            session.headToken = session.lastArgmax
            gpuSec += decodeStack.lastGPUMillis / 1000.0
            if runMTP, let mtp {
                let nextTok = isLast ? Int32(session.lastArgmax) : Int32(history[pos + 1])
                let preInj = decodeStack.slots.last.map { ($0.scratch.h2, $0.scratch.moeOut, $0.scratch.injB) }
                session.mtpDraft = try mtp.forwardPrefill(streamBuf: decodeStack.lastStreamTensor, preInject: preInj,
                                                          nextTokens: [nextTok], startPos: pos,
                                                          prefillScratch: mtp.scratch1, projectLastRow: runHeadNow)
                gpuSec += mtp.lastGPUMillis / 1000.0
            }
            session.prefillPos += 1
            session.plePrecomputed = false
            if residentKVHistory.count == pos {
                residentKVHistory.append(cacheHistory[pos])
            }
            highWaterPos = max(highWaterPos, session.prefillPos)
            session.reportPrefillProgress()
            return
        }
        let rebuiltStack = (cachedPrefill == nil || rows > cachedPrefill!.stack.maxRows)
        let prefillStack = try ensurePrefillStack(rows: rows, allowShrink: pos == 0 && session.visualSpans.isEmpty)
        if rebuiltStack || prefillStack.maxRows != lastPrefillStackRows {
            session.plePrecomputed = false
            session.pleParity = 0
        }
        let nextRows = FNGPUPrefillStack.resolveSubchunk(pos: pos + rows, limitPos: limitPos, chunkSize: chunkSize,
                                                         hidden: hidden, hc: hc, visualSpans: session.visualSpans).rows
        let nextChunkPos: Int? = (!isCheckpointEnd && nextRows > 1 && nextRows <= prefillStack.maxRows) ? (pos + rows) : nil
        let fusedMTP = runMTP ? mtp : nil
        let fusedDFlash = runDFlash ? dflash : nil
        let fusedDFlashSlot = fusedDFlash?.sessionSlots.first
        if let streamIn = subchunk.visualStreamIn {
            session.lastArgmax = try prefillStack.forward(history: history, streamIn: streamIn, head: head,
                                                          startPos: pos, readStream: false, runHead: runHeadNow,
                                                          mtp: fusedMTP, projectMTP: runHeadNow,
                                                          dflash: fusedDFlash, dflashSlot: fusedDFlashSlot,
                                                          dflashWindowEnd: limitPos,
                                                          pleParity: session.pleParity,
                                                          plePrecomputed: session.plePrecomputed,
                                                          nextChunkStartPos: nextChunkPos,
                                                          nextChunkCount: nextRows).token
        } else {
            session.lastArgmax = try prefillStack.forward(history: history, embedTable: embedTable, head: head,
                                                          startPos: pos, readStream: false, runHead: runHeadNow,
                                                          mtp: fusedMTP, projectMTP: runHeadNow,
                                                          dflash: fusedDFlash, dflashSlot: fusedDFlashSlot,
                                                          dflashWindowEnd: limitPos,
                                                          pleParity: session.pleParity,
                                                          plePrecomputed: session.plePrecomputed,
                                                          nextChunkStartPos: nextChunkPos,
                                                          nextChunkCount: nextRows).token
        }
        if fusedDFlash != nil {
            session.dflashCacheOffset = pos + rows
            session.dflashPendingCtxCount = 0
        }
        // The head wrote its argmax into `head.token`; the next step gathers
        // from it, so the session tracks what is on the device.
        if runHeadNow { session.headToken = session.lastArgmax }
        gpuSec += prefillStack.lastGPUMillis / 1000.0
        if fusedMTP != nil && runHeadNow {
            session.mtpDraft = prefillStack.lastMTPDraft
        }
        if nextChunkPos != nil {
            session.pleParity = 1 - session.pleParity
            session.plePrecomputed = true
        } else {
            session.plePrecomputed = false
        }
        session.prefillPos += rows
        if residentKVHistory.count == pos {
            residentKVHistory.append(contentsOf: cacheHistory[pos..<(pos + rows)])
        }
        highWaterPos = max(highWaterPos, session.prefillPos)
        session.reportPrefillProgress()
    }

    // MARK: - The module seam

    /// Computes the context-adaptive, 64-row-aligned prefill chunk size for a prompt of `totalTokens` tokens.
    /// Scales up to 8K (`QWPrefillLimits.maxRows = 8192`) on large contexts and balances chunk sizes evenly.
    public func dynamicPrefillChunkRows(forTotalTokens totalTokens: Int) -> Int {
        QWPrefillLimits.balancedChunkRows(forTotalTokens: totalTokens, maxChunkRows: dynamicMaxPrefillRows)
    }

    /// Prepares the prefill stack for a prompt of `count` tokens, so the
    /// allocation lands before the measured prefill and not inside it.
    public func preparePrefill(forTokenCount count: Int) throws {
        guard count > 1 else { return }
        let chunkRows = dynamicPrefillChunkRows(forTotalTokens: count)
        _ = try ensurePrefillStack(rows: min(chunkRows, count))
    }

    /// Starts a session: sizes the arena and prepares repetition-penalty tracking.
    public func beginSession(_ session: inout FNGPUSessionData, maxNew: Int) {
        _ = maxNew
        session.prefillStepChunk = 0
        session.recentGeneratedTokens.removeAll(keepingCapacity: true)
        session.seenTokens.removeAll(keepingCapacity: true)
        session.seenTokenList.removeAll(keepingCapacity: true)
    }

    /// Flushes a deferred prefix cache save after the first token has been emitted to the caller.
    public func flushPendingPrefixCache(_ session: inout FNGPUSessionData) {
        guard session.pendingPrefixSave else { evictPrefillScratchIfNeeded(); return }
        session.pendingPrefixSave = false
        if hasPrefixCacheTier && session.usePrefixCache {
            let cacheHistory = QWVisualSpan.saltedCacheHistory(
                history: session.history, visualSpans: session.visualSpans, cacheSalt: session.cacheSalt
            )
            savePrefixCache(history: cacheHistory, lastChunkRows: session.lastRows, useSpec: session.useSpec,
                            lastArgmax: session.lastArgmax, mtpDraft: session.mtpDraft, cacheDir: prefixCacheDirectory)
        }
        evictPrefillScratchIfNeeded()
    }

    /// Runs the prefill up to the core's chunk and returns the first generated token.
    public func encodePrefillChunk(_ session: inout FNGPUSessionData,
                                    chunk: Range<Int>, runsHead: Bool) throws -> Int {
        let history = session.history
        let cacheHistory = QWVisualSpan.saltedCacheHistory(
            history: history, visualSpans: session.visualSpans, cacheSalt: session.cacheSalt
        )
        let activeCap = activeCapacity
        guard history.count < activeCap else {
            throw FNGPUError.contextDoesNotFit(requested: history.count + 1,
                                                maxFitting: FNGPUDFlash.maxFittingContext(activeCapacity: activeCap))
        }
        var gpuSec = session.prefillGPUSec
        let canUsePrefixCache = hasPrefixCacheTier && session.usePrefixCache
        if !session.restoreDone {
            session.restoreDone = true
            let prevWater = highWaterPos
            if canUsePrefixCache, !cacheHistory.isEmpty,
               let restored = try lookupAndRestorePrefixCache(history: cacheHistory, rawHistory: history,
                                                              useSpec: session.useSpec,
                                                              cacheDir: prefixCacheDirectory, gpuSec: &gpuSec,
                                                              allowExact: true) {
                session.prefixCacheHit = true; session.prefixCacheMatchedTokens = restored.matched
                session.prefixCacheTier = restored.tier; session.prefillPos = restored.matched
                session.lastArgmax = restored.lastArgmax; session.headToken = restored.lastArgmax
                session.mtpDraft = restored.mtpDraft
                session.dflashCacheOffset = restored.dflashCacheOffset
                session.dflashPendingCtxPos = 0
                session.dflashPendingCtxCount = 0
                if !cacheHistory.isEmpty, residentKVHistory.count == restored.matched,
                   residentKVHistory[0] != cacheHistory[0] {
                    residentKVHistory[0] = cacheHistory[0]
                }
            } else {
                highWaterPos = max(highWaterPos, prevWater)
                wipeSessionState(overwriteTokens: history.count)
                session.dflashCacheOffset = 0
                session.dflashPendingCtxPos = 0
                session.dflashPendingCtxCount = 0
                residentKVHasMTP = (session.useSpec || canUsePrefixCache) && (mtp != nil || dflash != nil)
            }
            session.prefillProgressBasePos = session.prefillPos
            highWaterPos = max(highWaterPos, min(activeCap, history.count + 8))
            session.reportPrefillProgress()
        }
        let limit = runsHead ? history.count : min(chunk.upperBound, history.count)
        let autoSharedRetain = (prefixCacheWarmStore?.hasSharedBase == true && runsHead && history.count > session.prefillPos + 4)
            ? (history.count - 4) : nil
        var retainPoints: [Int] = []
        if let baseRetain = session.retainPrefixLength ?? prefixCacheWarmStore?.sharedBranchPrefixLength(for: cacheHistory),
           baseRetain >= 128, baseRetain < history.count, baseRetain > session.prefillPos {
            retainPoints.append(baseRetain)
        }
        if let autoSharedRetain, autoSharedRetain >= 128, autoSharedRetain < history.count,
           autoSharedRetain > max(session.prefillPos, retainPoints.last ?? 0) + 4 {
            retainPoints.append(autoSharedRetain)
        }
        if canUsePrefixCache && !retainPoints.isEmpty {
            let hasMTP = (session.useSpec || canUsePrefixCache) && mtp != nil
            for retainLen in retainPoints where session.prefillPos < retainLen {
                let baseSlice = cacheHistory.prefix(retainLen)
                let baseHash = Self.fnPrefixHash(tokens: baseSlice, kvMode: kvMode, hasMTP: hasMTP,
                                                  dflashMarker: dflashPrefixMarker)
                let baseName = String(format: "fn_entry_%016llx_%d.qwcache", baseHash, retainLen)
                if prefixCacheWarmStore?.contains(name: baseName) != true {
                    let baseLimit = min(limit, retainLen)
                    let baseStepChunk = dynamicPrefillChunkRows(forTotalTokens: max(1, retainLen - session.prefillPos))
                    while session.prefillPos < baseLimit {
                        let nextRows = min(baseStepChunk, retainLen - session.prefillPos)
                        if !runsHead && session.prefillPos + nextRows > baseLimit { break }
                        try prefillChunk(&session, history: history, cacheHistory: cacheHistory,
                                         chunkSize: baseStepChunk, targetEnd: retainLen, gpuSec: &gpuSec)
                    }
                    if session.prefillPos == retainLen {
                        savePrefixCache(history: Array(baseSlice), lastChunkRows: session.lastRows, useSpec: session.useSpec,
                                        lastArgmax: session.lastArgmax, mtpDraft: session.mtpDraft,
                                        cacheDir: prefixCacheDirectory,
                                        isSharedBase: false)
                    }
                }
            }
        }
        if session.prefillStepChunk <= 0 {
            let matched = session.prefixCacheHit ? session.prefixCacheMatchedTokens : 0
            let uncached = max(1, history.count - max(session.prefillPos, matched))
            let baseChunk = dynamicPrefillChunkRows(forTotalTokens: uncached)
            session.prefillStepChunk = session.prefixCacheHit ? min(cachedPrefill?.stack.maxRows ?? dynamicMaxPrefillRows, baseChunk) : baseChunk
        }
        let stepChunk = session.prefillStepChunk
        while session.prefillPos < limit {
            guard FNGPUPrefillStack.deferredSubchunkRows(
                pos: session.prefillPos, limit: limit, runsHead: runsHead, totalCount: history.count,
                stepChunk: stepChunk, hidden: hidden, hc: hc, visualSpans: session.visualSpans
            ) != nil else { break }
            try prefillChunk(&session, history: history, cacheHistory: cacheHistory, chunkSize: stepChunk, gpuSec: &gpuSec)
        }
        if runsHead && session.prefillPos >= history.count && !session.hasFirstToken {
            session.prefillGPUSec = gpuSec
            let firstTok = try drawFirstToken(&session)
            if canUsePrefixCache { session.pendingPrefixSave = true } else { evictPrefillScratchIfNeeded() }
            return firstTok
        }
        session.prefillGPUSec = gpuSec
        return session.hasFirstToken ? session.firstToken : session.lastArgmax
    }

    /// Runs one decode step and returns the token the model produced (using `sampler: QWSampler`).
    public func encodeDecodeToken(_ session: inout FNGPUSessionData, token: Int,
                                   position: Int) throws -> Int {
        if session.pendingPrefixSave { flushPendingPrefixCache(&session) }
        // A speculative round that committed two tokens already fed both to
        // the model. The second one is handed back now, without a GPU round.
        if !session.pendingTokens.isEmpty {
            if !session.roundTokenCounts.isEmpty {
                session.roundTokenCounts[session.roundTokenCounts.count - 1] += 1
            }
            let queued = session.pendingTokens.removeFirst()
            if session.pendingConstraintAcceptCount > 0 {
                session.pendingConstraintAcceptCount -= 1
                try session.constraint?.accept(queued)
            }
            return queued
        }
        let activeCap = activeCapacity
        guard session.history.count + 1 < activeCap else {
            // The context is full: the request cannot be served in full, and
            // a truncated answer is not an answer the caller asked for.
            throw FNGPUError.contextDoesNotFit(requested: session.history.count + 1,
                                                maxFitting: FNGPUDFlash.maxFittingContext(activeCapacity: activeCap))
        }
        let arb = gpuTurnArbitrator
        arb?.acquirePrimaryStepSync()
        defer { arb?.releasePrimaryStep() }
        let canSpecConstraint = session.constraint.map {
            $0.isUnconstrainedNow || (QWConstrainedSpeculationGate.isEnabled && $0.allowsSpeculation)
        } ?? true
        if session.useSpec, let vStack = verifyStack,
           canSpecConstraint,
           session.history.count + 8 < activeCap {
            if let mtp {
                return try speculativeStep(&session, token: token, vStack: vStack, mtp: mtp)
            }
            if let dflash {
                return try FNGPUDFlash.step(self, session: &session, token: token, vStack: vStack, dflash: dflash)
            }
        }
        let dflashSlot = dflash?.sessionSlots.first
        let preContextUpdate = FNGPUDFlash.consumePendingContextUpdate(&session, slot: dflashSlot)
        // The gather reads the token from the device, so the token to feed is
        // staged before the step runs.
        if session.headToken != token {
            arena.write(head.token, from: [Int32(token)])
            session.headToken = token
        }
        let prevCount = session.history.count
        session.history.append(Int64(token))
        if residentKVHistory.count == prevCount { residentKVHistory.append(Int64(token)) }
        highWaterPos = max(highWaterPos, session.history.count)
        let stepT0 = Date()
        let r = try decodeStack.forwardToken(history: session.history, embedTable: embedTable,
                                              head: head, readStream: false,
                                              dflash: dflash, dflashSlot: dflashSlot,
                                              preContextUpdate: preContextUpdate)
        if dflash != nil {
            session.dflashCacheOffset = session.history.count
        }
        session.stepMs = Date().timeIntervalSince(stepT0) * 1000.0
        session.lastStepGPUMillis = decodeStack.lastGPUMillis
        session.headToken = r.token
        let next = try sampleFromHead(argmaxToken: r.token, seenTokens: session.seenTokens,
                                      seenTokenList: session.seenTokenList,
                                      sampler: session.sampler, constraint: session.constraint)
        if session.sampler.config.repetitionPenalty != 1.0 { session.recordSeenToken(next, vocab: vocab) }
        if session.collectLogprobs { recordTokenLogprob(&session, token: next, row: 0) }
        if next != r.token {
            arena.write(head.token, from: [Int32(next)])
            session.headToken = next
        }
        return next
    }
}
