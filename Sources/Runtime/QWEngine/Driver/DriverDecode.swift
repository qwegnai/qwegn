// DriverDecode.swift — QWEngineDriver DriverDecode family (split from QWEngineDriver.swift; same-module extension).
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

extension QWEngineDriver {
    public func decodeStep(_ session: QWEngineSession, token: Int) throws -> Int {
        try decodeStep(session, token: token, permit: nil)
    }

    func decodeStep(_ session: QWEngineSession, token: Int,
                            permit: QWGenerationPermit?) throws -> Int {
        try admission.withFlight {
            try admission.requestGate.validate(permit)
            guard permit != nil || session.stochastic == nil else {
                throw QWEngineDriverError.state("use the stochastic request door for a sampled session")
            }
            guard speculativeCoordinator.verifiedSession == nil else {
                throw QWEngineDriverError.state("resolve verification before using driver scratch")
            }
            guard let next = try forwardStep(session, token: token) else {
                throw QWEngineDriverError.state("ordinary forward did not produce a token")
            }
            if session.stochastic == nil {
                if (activeConstraint?.constraint.isUnconstrainedNow ?? true), let prev1 = session.tokenHistory.last {
                    let prev2 = session.tokenHistory.count >= 2 ? session.tokenHistory[session.tokenHistory.count - 2] : nil
                    session.tokenRecycler.recordTransition(previous: prev2, current: prev1, next: next)
                }
                session.tokenHistory.append(next)
            }
            return next
        }
    }

    /// Ordinary single-token forward pass across the 64-layer trunk.
    func forwardStep(_ session: QWEngineSession, token: Int) throws -> Int? {
        let arb = gpuTurnArbitrator
        arb?.acquirePrimaryStepSync()
        defer { arb?.releasePrimaryStep() }
        let tStep0 = Date()
        if let prevCB = session.lastCommitCB {
            session.lastCommitCB = nil
            prevCB.waitUntilCompleted()
            if prevCB.status != .completed {
                admission.commandBufferErrors += 1
                session.stateFailed = true
                throw QWEngineDriverError.state("prior speculative commit GPU failure before serial step")
            }
        }
        let pos = session.position
        guard pos < ropeRows else { throw QWEngineDriverError.posBeyondTable(pos) }
        guard token >= 0, token < geometry.vocab else { throw QWEngineDriverError.badToken(token) }
        try claimMLAIndexerState(session)
        let seedMTP = (mtpScratch != nil || dflashDrafter != nil || dsparkDrafter != nil) && !session.batchedDecodeTouched
        let mtpRows = seedMTP ? try prepareMTPSeeding(session, count: 1) : 0
        if session.kv.tokenCapacity <= pos {
            try kvCache.grow(session.kv, to: pos + QWKVCache.pageTokens)
            session.attnKVViews = try kvCache.layerViewsAll(session.kv, layers: attnLayers)
        }
        try kvCache.prepareWrites(session.kv, positions: pos..<(pos + 1))

        let tEncode0 = Date()
        try copyEmbedRow(token, to: sc.x)
        if session.generated == 0 && !activeVisualSpans.isEmpty {
            applyVisualSpans(start: pos, count: 1, to: sc.x)
        }
        if let mtpScratch {
            let src = sc.x.region.buffer.contents().advanced(by: sc.x.region.base + sc.x.offset)
            let dst = mtpScratch.embedding.region.buffer.contents().advanced(by: mtpScratch.embedding.region.base + mtpScratch.embedding.offset)
            memcpy(dst, src, sc.x.bytes)
        }
        var (cb, enc) = try commandFrame.begin(hal: hal)
        commandFrame.resetColdTelemetry()
        commandFrame.beginDecodeProfileStep(device: hal.device, enabled: Self.decodeProfileEnabled)
        profileStage("embed")
        var allCBsDrained = false
        defer {
            if allCBsDrained {
                commandFrame.closeActiveEncoder()
                commandFrame.clearActiveBindings()
            } else {
                commandFrame.abortIncomplete(
                    admission: admission, session: nil,
                    resetDecodeProfiler: Self.decodeProfileEnabled)
            }
        }
        // The residual stream is sc.x for the WHOLE step: the shipping norm
        // door (qwg_rmsnorm_residual) advances it IN PLACE (residual :=
        // half(x + r), register-resident sv — parity-proven leg-2c).
        // (The CPU embed write above lands before the encoder's first
        // dispatch is scheduled — the same happens-before Metal gives for
        // host writes vs command-buffer work; no barrier needed.)

        for (v, p) in geometry.executionPlan.enumerated() {
            let lw = layersW[p]
            // The layer's kind was resolved when the plan was read: the loop
            // looks the kernel up instead of switching on the kind.
            try layerKernels[p].encodeDecode(QWLayerDecodeContext(
                driver: self, enc: activeEnc ?? enc, session: session, position: pos, layer: v, weights: lw))
            if seedMTP && dsparkDrafter != nil {
                let nextP = (v + 1 < geometry.executionPlan.count) ? geometry.executionPlan[v + 1] : nil
                encodeDSparkTargetTapIfNeeded(
                    activeEnc ?? enc, physicalLayer: p, nextPhysicalLayer: nextP,
                    mhcStream: sc.mhcStream, mhcControls: sc.mhcControls,
                    branchDn16: sc.dn16, residualX16: sc.x,
                    srcRowOffset: 0, dstRowOffset: 0, rows: 1)
            }
        }
        profileStage("head")
        cb = activeCB ?? cb
        enc = activeEnc ?? enc

        try encodeTargetHeadGreedy(enc, x: sc.ln1,
                                   writeLogits: session.stochastic != nil || activeCollectLogprobs || activeConstraint != nil)
        if seedMTP, (mtpScratch != nil || dflashDrafter != nil || dsparkDrafter != nil) {
            activeEnc = nil
            enc.endEncoding()
            let seedEmbeddings = mtpScratch?.embedding ?? sc.x
            let seedHidden: QWTensorValue = echoDrafter != nil
                ? sc.mhcStream!
                : (hasSandwichGatedNorm ? sc.x : sc.ln1)
            try encodeMTPSeeding(cb, session: session, embeddings: seedEmbeddings,
                                 hidden: seedHidden, count: 1, cacheRows: mtpRows)
            activeCB = nil
            cb.commit()
        } else {
            activeEnc = nil
            activeCB = nil
            hal.end(cb, enc)
        }
        stepCBs?.append(cb)
        let encodeUs = Date().timeIntervalSince(tEncode0) * 1e6
        cpuEncodeLedger.record(micros: encodeUs)
        let allCbs = stepCBs ?? [cb]
        for stepCb in allCbs {
            stepCb.waitUntilCompleted()
        }
        let coldFailure = commandFrame.drainColdEvents(waitUntilIdle: true)
        commandFrame.flushPendingColdUnloads()
        commandFrame.endDecodeProfileStep(enabled: Self.decodeProfileEnabled)
        allCBsDrained = true
        for stepCb in allCbs where stepCb.status != .completed || stepCb.error != nil {
            commandBufferErrors += 1   // M1 triad
        }
        let hasSpecState = mtpCache != nil || verifyWindow != nil
        if let coldFailure {
            if hasSpecState { session.stateFailed = true }
            throw QWEngineDriverError.state("cold event staging: \(coldFailure)")
        }
        for stepCb in allCbs {
            if let e = stepCb.error {
                if hasSpecState { session.stateFailed = true }
                throw QWEngineDriverError.state("step cb: \(e.localizedDescription)")
            }
            if stepCb.status != .completed {
                if hasSpecState { session.stateFailed = true }
                throw QWEngineDriverError.state("step cb completion failed (status=\(stepCb.status.rawValue))")
            }
        }
        if ColdExpertStreamer.verifyPromotions {
            var bad = 0, checked = 0
            for lw in layersW {
                guard let moe = lw.moe, let st = moe.coldStreamer, st.isDynamic else { continue }
                checked += st.promotedForVerify.count
                let g = moe.gateUp.blocks, d = moe.down.blocks
                bad += try st.verifyPromoted(
                    guBank: g.region.buffer.contents().advanced(by: g.region.base + g.offset),
                    downBank: d.region.buffer.contents().advanced(by: d.region.base + d.offset))
            }
            FileHandle.standardError.write(Data("[expert-cache] verified=\(checked) mismatched=\(bad)\n".utf8))
        }
        // GPU time = union of every CB's GPU interval in this step (cold
        // staging splits the step into one CB per MoE layer); host-wait gaps
        // between CBs are idle, not busy.
        let gpuUs = Self.gpuBusyMicros(allCbs)
        if gpuUs.isFinite, gpuUs > 0 { decodeGpuLedger.record(micros: gpuUs) }
        // The engine's own GPU busy ratio: the step's GPU busy time over the
        // wall time the step took. A ratio, not a percentage the host had to
        // guess at.
        let wallUs = Date().timeIntervalSince(tStep0) * 1e6
        QWTrunkRuntimeSupport.recordGPUBusy(gpuSeconds: gpuUs, wallSeconds: wallUs)
        if let n = stepCBs?.count, n > 1 {
            QWTelemetryRegistry.shared.record("decode.cb_per_step", Double(n), provenance: .measured)
            QWTelemetryRegistry.shared.record("decode.cold_stage_host_us", decodeTel.coldStageUs, provenance: .measured)
        }
        if Self.decodeTraceEnabled {
            FileHandle.standardError.write(Data(String(
                format: "[decode] pos=%d wall=%.2fms gpu_busy=%.2fms encode=%.2fms cbs=%d cold_stage_host=%.2fms\n",
                pos, wallUs / 1000, gpuUs / 1000, encodeUs / 1000, stepCBs?.count ?? 1,
                decodeTel.coldStageUs / 1000).utf8))
            FileHandle.standardError.write(Data(String(
                format: "[decode]   cold: wait=%.2fms io=%.2fms staged=%d bridge=%u\n",
                decodeTel.coldWaitUs / 1000, decodeTel.coldIOUs / 1000, decodeTel.coldStaged,
                coldEvents?.bridgeIters ?? 0).utf8))
        }

        session.position += 1
        session.generated += 1
        // H4 (full): the C09 decode door materialises row `pos` IN KERNEL
        // (split-0 epilogue) — the cache cursor must cover it so
        // rollback(by:)/snapshots price the true retained window. Monotonic,
        // zero GPU cost.
        kvCache.markWritten(session.kv, upTo: pos + 1)
        if geometry.hasAnySlidingWindow {
            try kvCache.recycleExpiredWindowPages(
                session.kv,
                currentPosition: pos + 1,
                safetyMargin: geometry.descriptor.maxSpeculativeDrafts + QWKVCache.pageTokens)
        }
        if seedMTP || (isWeightFreeSpeculation && !session.batchedDecodeTouched) {
            commitMTPSeeding(session, position: pos, count: 1, cacheRows: mtpRows)
        }
        let rawId = Int(sc.token.region.buffer.contents()
            .advanced(by: sc.token.region.base + sc.token.offset)
            .assumingMemoryBound(to: Int32.self).pointee)
        let next = try selectConstrainedToken(session: session, rawArgmax: rawId)
        session.roundRecord.pendingToken = next
        session.roundRecord.outputCount = session.generated
        return next
    }

    /// Selects a token under `activeConstraint` when set, or returns `rawArgmax`.
    func selectConstrainedToken(session: QWEngineSession, rawArgmax: Int) throws -> Int {
        guard let active = activeConstraint, session.stochastic == nil else { return rawArgmax }
        let rawLogits = sc.logits.region.pointer.advanced(by: sc.logits.offset)
            .assumingMemoryBound(to: Float.self)
        let rowBuf = UnsafeBufferPointer(start: rawLogits, count: geometry.vocab)
        let repArr = active.sampler.config.repetitionPenalty == 1.0 ? [] : session.activeRepetitionHistory()
        let token: Int
        if let allowed = try active.constraint.allowedTokens() {
            token = try active.sampler.constrainedToken(logits: rowBuf, allowed: allowed, repetitionHistory: Set(repArr))
        } else if active.sampler.config.isGreedy && active.sampler.config.repetitionPenalty == 1.0 {
            token = rawArgmax
        } else if active.sampler.config.isGreedy {
            token = try active.sampler.penalisedGreedyToken(logits: rowBuf, repetitionHistory: Set(repArr))
        } else {
            token = try active.sampler.sampleToken(logits: rowBuf, repetitionHistory: repArr)
        }
        try active.constraint.accept(token)
        return token
    }

    /// QW_DECODE_TRACE=1 prints one wall/GPU/encode/cold-staging line per step.
    static let decodeTraceEnabled = ProcessInfo.processInfo.environment["QW_DECODE_TRACE"] == "1"

    /// QW_DECODE_PROFILE=1 splits the decode encoder at sub-stage boundaries
    /// (stage-boundary timestamp counters) and prints per-stage GPU ms/token.
    /// Diagnostic only: the extra encoders add a small per-boundary cost.
    static let decodeProfileEnabled = ProcessInfo.processInfo.environment["QW_DECODE_PROFILE"] == "1"
    /// QW_PREFILL_PROFILE=1 does the same per prefill chunk (GPU ms per chunk and per token).
    static let prefillProfileEnabled = ProcessInfo.processInfo.environment["QW_PREFILL_PROFILE"] == "1"

    /// Start a new timed encoder for `stage` on the active CB (no-op unless profiling).
    func profileStage(_ stage: String) {
        commandFrame.profileStage(stage, hal: hal,
                                  enabled: Self.decodeProfileEnabled || Self.prefillProfileEnabled)
    }

    /// Rotates `enc` to a fresh compute encoder on `activeCB` (preserving stage profiling when enabled).
    func rotateActiveEncoder(_ enc: inout any MTLComputeCommandEncoder, stage: String = "moe.experts_nax") throws {
        try commandFrame.rotateActiveEncoder(
            &enc, stage: stage, hal: hal,
            profilingEnabled: Self.decodeProfileEnabled || Self.prefillProfileEnabled)
    }

    func beginPrefillProfileChunk() {
        commandFrame.beginPrefillProfileChunk(device: hal.device, enabled: Self.prefillProfileEnabled)
    }

    func endPrefillProfileChunk(tokens: Int) {
        commandFrame.endPrefillProfileChunk(tokens: tokens, enabled: Self.prefillProfileEnabled)
    }

    /// QW_ROUTE_DUMP=<path> appends every cold-staged MoE layer's routed ids (diagnostic).
    static let routeDumpHandle: FileHandle? = {
        guard let p = ProcessInfo.processInfo.environment["QW_ROUTE_DUMP"] else { return nil }
        FileManager.default.createFile(atPath: p, contents: nil)
        return FileHandle(forWritingAtPath: p)
    }()

    /// Union of the GPU execution intervals of `cbs`, in microseconds.
    /// Overlapping CBs (e.g. a resident-expert CB running beside cold IO) are
    /// counted once; idle gaps between CBs are excluded.
    static func gpuBusyMicros(_ cbs: [any MTLCommandBuffer]) -> Double {
        var spans: [(Double, Double)] = []
        spans.reserveCapacity(cbs.count)
        for c in cbs where c.gpuEndTime > c.gpuStartTime && c.gpuStartTime > 0 {
            spans.append((c.gpuStartTime, c.gpuEndTime))
        }
        guard !spans.isEmpty else { return 0 }
        spans.sort { $0.0 < $1.0 }
        var total = 0.0
        var (lo, hi) = spans[0]
        for (s, e) in spans.dropFirst() {
            if s > hi {
                total += hi - lo
                (lo, hi) = (s, e)
            } else if e > hi {
                hi = e
            }
        }
        total += hi - lo
        return total * 1e6
    }

    private final class CommittedCBBox: @unchecked Sendable {
        let cb: any MTLCommandBuffer
        private let lock = NSLock()
        private var isCommitted = false

        init(_ cb: any MTLCommandBuffer) { self.cb = cb }

        var committed: Bool { lock.withLock { isCommitted } }

        func commit() {
            lock.withLock {
                guard !isCommitted else { return }
                isCommitted = true
                cb.commit()
            }
        }
    }

    func stageColdExpertsSpan(
        moe: MoEW, s: MoEScratch, startRow: Int, totalRows: Int,
        isVerifyWindow: Bool = false,
        onBeforeIO: ((inout any MTLComputeCommandEncoder, any MTLCommandBuffer, Int) throws -> Void)? = nil,
        onAfterGateUpIO: ((any MTLComputeCommandEncoder, Int) throws -> Void)? = nil
    ) throws -> (nextRow: Int, stagedCount: Int) {
        guard let streamer = moe.coldStreamer,
              let curCB = activeCB, let curEnc = activeEnc else { return (totalRows, 0) }
        let tStage0 = Date()
        defer { decodeTel.coldStageUs += Date().timeIntervalSince(tStage0) * 1e6 }
        activeEnc = nil
        activeCB = nil
        hal.end(curCB, curEnc)
        stepCBs?.append(curCB)
        let effectiveColdSlots = streamer.effectiveColdBudget(isVerifyWindow: isVerifyWindow).slots
        let canPrecommitResident = !streamer.isDynamic
            && (totalRows - startRow) * moe.expertsPerTok <= effectiveColdSlots
        if canPrecommitResident, let onBeforeIO {
            guard let resPair = hal.begin() else {
                throw QWEngineDriverError.state("cold expert resident overlap cmdbuffer")
            }
            var resEnc = resPair.1
            var resOpen = true
            defer { if resOpen { resEnc.endEncoding() } }
            try onBeforeIO(&resEnc, resPair.0, totalRows)
            resOpen = false
            hal.end(resPair.0, resEnc)
            stepCBs?.append(resPair.0)
        }
        let tWait0 = Date()
        curCB.waitUntilCompleted()
        decodeTel.coldWaitUs += Date().timeIntervalSince(tWait0) * 1e6
        if let err = curCB.error {
            commandBufferErrors += 1
            throw QWEngineDriverError.state("cold expert route cb: \(err.localizedDescription)")
        }
        if curCB.status != .completed {
            commandBufferErrors += 1
            throw QWEngineDriverError.state("cold expert route cb completion failed (status=\(curCB.status.rawValue))")
        }
        flushPendingColdUnloads()
        let idxPtr = s.idx.region.buffer.contents()
            .advanced(by: s.idx.region.base + s.idx.offset)
            .assumingMemoryBound(to: UInt32.self)
        dumpRoute(streamer: streamer, ids: idxPtr + startRow * moe.expertsPerTok,
                  rows: totalRows - startRow, routes: moe.expertsPerTok)
        let prep = try streamer.prepareSpan(
            idxPtr: idxPtr, startRow: startRow, totalRows: totalRows,
            routes: moe.expertsPerTok, isVerifyWindow: isVerifyWindow)
        if !canPrecommitResident, prep.stagedCount > 0, let onBeforeIO {
            guard let resPair = hal.begin() else {
                throw QWEngineDriverError.state("cold expert resident overlap cmdbuffer")
            }
            var resEnc = resPair.1
            var resOpen = true
            defer { if resOpen { resEnc.endEncoding() } }
            try onBeforeIO(&resEnc, resPair.0, prep.nextRow)
            resOpen = false
            hal.end(resPair.0, resEnc)
            stepCBs?.append(resPair.0)
        }
        decodeTel.coldStaged += prep.stagedCount
        let tIO0 = Date()
        defer { decodeTel.coldIOUs += Date().timeIntervalSince(tIO0) * 1e6 }
        if prep.stagedCount > 0 {
            if let onAfterGateUpIO {
                guard let guPair = hal.begin() else {
                    throw QWEngineDriverError.state("cold expert gateup overlap cmdbuffer")
                }
                var guOpen = true
                defer { if guOpen { guPair.1.endEncoding() } }
                try onAfterGateUpIO(guPair.1, prep.nextRow)
                guOpen = false
                guPair.1.endEncoding()
                let guBox = CommittedCBBox(guPair.0)
                defer { if guBox.committed { stepCBs?.append(guBox.cb) } }
                try streamer.executeTwoPhaseBatch(
                    guBatch: prep.guBatch,
                    downBatch: prep.downBatch,
                    onGateUpLoaded: {
                        guBox.commit()
                    }
                )
            } else {
                try streamer.executeBatch(guBatch: prep.guBatch, downBatch: prep.downBatch)
            }
        }
        guard let nextPair = hal.begin() else {
            throw QWEngineDriverError.state("cold expert cmdbuffer")
        }
        streamer.commitPromotion(prep.transaction)
        activeCB = nextPair.0
        activeEnc = nextPair.1
        return (prep.nextRow, prep.stagedCount)
    }

    private func dumpRoute(streamer: ColdExpertStreamer, ids: UnsafePointer<UInt32>, rows: Int, routes: Int) {
        guard let fh = Self.routeDumpHandle else { return }
        // Pre-rewrite physical expert ids: [layerTag u32, rows u32, routes u32, ids...].
        var hdr: [UInt32] = [UInt32(streamer.layerPrefix.split(separator: ".").compactMap { Int($0) }.last ?? 0),
                             UInt32(rows), UInt32(routes)]
        fh.write(Data(bytes: &hdr, count: 12))
        fh.write(Data(bytes: ids, count: rows * routes * 4))
    }

    /// Whether `stageColdExpertsMapped` can stage this MoE's whole span in one pass.
    func canStageColdExpertsMapped(moe: MoEW, isVerifyWindow: Bool = false) -> Bool {
        // GPU span map + resident/cold overlap (tuning.prefillSpanMap); off restores the host-mapped `stageColdExpertsSpan`.
        guard tuning.prefillSpanMap, let st = moe.coldStreamer,
              activeCB != nil, activeEnc != nil else { return false }
        return st.canMapWholeSpan(isVerifyWindow: isVerifyWindow)
    }

    /// Whole-span dynamic-residency staging with no host round trip before the resident pass: the route CB maps
    /// ids to bank rows (`qwg_moe_span_ids`), `resident` is committed before the route readback, the host stages
    /// misses into the GPU-chosen slots, `coldGateUp` commits once gate/up lands. Returns the staged count.
    func stageColdExpertsMapped(
        moe: MoEW, s: MoEScratch, rows: Int,
        resident: (inout any MTLComputeCommandEncoder, any MTLCommandBuffer, QWTensorValue) throws -> Void,
        coldGateUp: (any MTLComputeCommandEncoder) throws -> Void
    ) throws -> Int {
        guard let streamer = moe.coldStreamer, let curCB = activeCB, let curEnc = activeEnc,
              let slotMap = streamer.attachGPUSlotMap(device: hal.device) else {
            throw QWEngineDriverError.state("span map staging without a dynamic streamer or active CB")
        }
        let tStage0 = Date()
        defer { decodeTel.coldStageUs += Date().timeIntervalSince(tStage0) * 1e6 }
        let routes = moe.expertsPerTok
        let count = rows * routes
        if (spanMapIds?.length ?? 0) < 4 * count {
            guard let buf = hal.device.makeBuffer(length: 4 * count, options: .storageModeShared) else {
                throw QWEngineDriverError.state("span map id buffer")
            }
            spanMapIds = buf
        }
        let mapBuf = spanMapIds!
        let mapped = QWTensorValue(region: QWMRegion(buffer: mapBuf, base: 0, length: mapBuf.length),
                                   offset: 0, shape: QWTensorShape([count], .i32))
        try QWGemm.encodeMoESpanIds(curEnc, hal: hal, ids: s.idx, slotMap: slotMap, out: mapped, count: count,
                                    experts: streamer.totalExperts, residentExperts: streamer.residentExperts)
        activeEnc = nil
        activeCB = nil
        hal.end(curCB, curEnc)
        stepCBs?.append(curCB)
        guard let resPair = hal.begin() else {
            throw QWEngineDriverError.state("span map resident cmdbuffer")
        }
        let resCB = resPair.0
        var resEnc = resPair.1
        var resOpen = true
        defer { if resOpen { resEnc.endEncoding() } }
        try resident(&resEnc, resCB, mapped)
        resOpen = false
        hal.end(resCB, resEnc)
        stepCBs?.append(resCB)

        let tWait0 = Date()
        curCB.waitUntilCompleted()
        decodeTel.coldWaitUs += Date().timeIntervalSince(tWait0) * 1e6
        if let err = curCB.error {
            commandBufferErrors += 1
            throw QWEngineDriverError.state("cold expert route cb: \(err.localizedDescription)")
        }
        if curCB.status != .completed {
            commandBufferErrors += 1
            throw QWEngineDriverError.state("cold expert route cb completion failed (status=\(curCB.status.rawValue))")
        }
        let idxPtr = s.idx.region.buffer.contents()
            .advanced(by: s.idx.region.base + s.idx.offset)
            .assumingMemoryBound(to: UInt32.self)
        dumpRoute(streamer: streamer, ids: idxPtr, rows: rows, routes: routes)
        let prep = try streamer.prepareSpanMapped(
            ids: idxPtr, mapped: mapBuf.contents().assumingMemoryBound(to: Int32.self), count: count)
        decodeTel.coldStaged += prep.stagedCount

        let tIO0 = Date()
        defer { decodeTel.coldIOUs += Date().timeIntervalSince(tIO0) * 1e6 }
        if prep.stagedCount > 0 {
            guard let (guCB, guEnc) = hal.begin() else {
                throw QWEngineDriverError.state("span map cold gateup cmdbuffer")
            }
            var guOpen = true
            defer { if guOpen { guEnc.endEncoding() } }
            try coldGateUp(guEnc)
            guOpen = false
            guEnc.endEncoding()
            let guBox = CommittedCBBox(guCB)
            defer { if guBox.committed { stepCBs?.append(guBox.cb) } }
            try streamer.executeTwoPhaseBatch(guBatch: prep.guBatch, downBatch: prep.downBatch,
                                              onGateUpLoaded: { guBox.commit() })
        }
        guard let nextPair = hal.begin() else {
            throw QWEngineDriverError.state("cold expert cmdbuffer")
        }
        streamer.commitPromotion(prep.transaction)
        activeCB = nextPair.0
        activeEnc = nextPair.1
        return prep.stagedCount
    }

    func unloadColdExpertsIfNeeded(moe: MoEW, stagedCount: Int) throws {
        guard stagedCount > 0, let streamer = moe.coldStreamer else { return }
        if streamer.isDynamic {
            try encodeColdPromotions(moe: moe, streamer: streamer)
            return
        }
        pendingColdUnloads.append((streamer: streamer, stagedCount: stagedCount))
    }

    /// Copies this span's staged experts into their promoted bank slots, in
    /// GPU order after the layer's cold compute and before the next layer's
    /// staging (which waits on this CB) can overwrite the staging pool.
    func encodeColdPromotions(moe: MoEW, streamer: ColdExpertStreamer) throws {
        guard !streamer.pendingPromotions.isEmpty else { return }
        defer { streamer.pendingPromotions.removeAll(keepingCapacity: true) }
        guard let enc = activeEnc, let coldGU = moe.coldGateUp, let coldDown = moe.coldDown else {
            throw QWEngineDriverError.state("dynamic expert promotion has no active encoder or cold bank")
        }
        func view(_ base: QWTensorValue, byteOffset: Int, bytes: Int) -> QWTensorValue {
            QWTensorValue(region: base.region, offset: base.offset + byteOffset,
                          shape: QWTensorShape([bytes / 2], .f16))
        }
        let guBytes = streamer.guExpertBytes
        let dnBytes = streamer.downExpertBytes
        for (staging, bank) in streamer.pendingPromotions {
            try QWVerifyWindow.encodeCopy(enc, hal: hal,
                source: view(coldGU.blocks, byteOffset: staging * guBytes, bytes: guBytes),
                destination: view(moe.gateUp.blocks, byteOffset: bank * guBytes, bytes: guBytes))
            try QWVerifyWindow.encodeCopy(enc, hal: hal,
                source: view(coldDown.blocks, byteOffset: staging * dnBytes, bytes: dnBytes),
                destination: view(moe.down.blocks, byteOffset: bank * dnBytes, bytes: dnBytes))
        }
        enc.memoryBarrier(scope: .buffers)
    }

    // MARK: - prefill ────────────────────────────────────────────────────────
}
