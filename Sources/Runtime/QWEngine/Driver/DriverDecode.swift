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
import QWState

extension QWEngineDriver {
    public func decodeStep(_ session: QWEngineSession, token: Int) throws -> Int {
        try decodeStep(session, token: token, permit: nil)
    }

    func decodeStep(_ session: QWEngineSession, token: Int,
                            permit: QWGenerationPermit?) throws -> Int {
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate(permit)
        guard permit != nil || session.stochastic == nil else {
            throw QWEngineDriverError.state("use the stochastic request door for a sampled session")
        }
        guard verifiedSession == nil else { throw QWEngineDriverError.state("resolve verification before using driver scratch") }
        guard let next = try forwardStep(session, token: token) else {
            throw QWEngineDriverError.state("ordinary forward did not produce a token")
        }
        session.tokenHistory.append(next)
        return next
    }

    /// Ordinary single-token forward pass across the 64-layer trunk.
    func forwardStep(_ session: QWEngineSession, token: Int) throws -> Int? {
        // The step's wall time, from the first work to the last wait. The
        // GPU's share of it is the engine's own busy ratio: measured, never
        // estimated from the token rate.
        let tStep0 = Date()
        let pos = session.position
        guard pos < ropeRows else { throw QWEngineDriverError.posBeyondTable(pos) }
        // C5 guard UP FRONT (C16 law): a throw after hal.begin() would
        // SIGABRT via encoder dealloc — validate before the encoder opens.
        guard token >= 0, token < geometry.vocab else { throw QWEngineDriverError.badToken(token) }
        try claimMLAIndexerState(session)
        let mtpRows = try prepareMTPSeeding(session, count: 1)
        // KV capacity: grow a page ahead so the C09 fused page write never
        // overruns (grow() is page-granular; tokenCapacity = pages·256).
        if session.kv.tokenCapacity <= pos {
            try kvCache.grow(session.kv, to: pos + QWKVCache.pageTokens)
            session.attnKVViews = try kvCache.layerViewsAll(session.kv, layers: attnLayers)
        }
        try kvCache.prepareWrites(session.kv, positions: pos..<(pos + 1))

        let tEncode0 = Date()   // CPU-tail ledger: begin→end only (the wait
        // below is GPU time and never belongs to the ≤1%-of-step encode gate)
        // x = embed[token] (CPU memcpy of one f16 row — 10 KiB, sub-ms).
        try copyEmbedRow(token, to: sc.x)
        if session.generated == 0 && !activeVisualSpans.isEmpty {
            applyVisualSpans(start: pos, count: 1, to: sc.x)
        }
        if let mtpScratch {
            let src = sc.x.region.buffer.contents().advanced(by: sc.x.region.base + sc.x.offset)
            let dst = mtpScratch.embedding.region.buffer.contents().advanced(by: mtpScratch.embedding.region.base + mtpScratch.embedding.offset)
            memcpy(dst, src, sc.x.bytes)
        }
        guard let pair = hal.begin() else { throw QWEngineDriverError.state("cmdbuffer") }
        var cb: any MTLCommandBuffer = pair.0
        var enc: any MTLComputeCommandEncoder = pair.1
        activeCB = cb
        activeEnc = enc
        stepCBs = []
        decodeTel.coldStageUs = 0
        decodeTel.coldWaitUs = 0
        decodeTel.coldIOUs = 0
        decodeTel.coldStaged = 0
        beginDecodeProfileStep()
        profileStage("embed")
        var encoderEnded = false
        defer {
            if !encoderEnded { (activeEnc ?? enc).endEncoding() }
            activeCB = nil
            activeEnc = nil
            stepCBs = nil
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
        }
        profileStage("head")
        cb = activeCB ?? cb
        enc = activeEnc ?? enc

        try encodeTargetHeadGreedy(enc, x: sc.ln1,
                                   writeLogits: session.stochastic != nil || activeCollectLogprobs || activeConstraint != nil)
        if let mtpScratch {
            enc.endEncoding()
            encoderEnded = true
            let seedHidden: QWTensorValue = echoDrafter != nil ? sc.mhcStream! : sc.ln1
            try encodeMTPSeeding(cb, session: session, embeddings: mtpScratch.embedding,
                                 hidden: seedHidden, count: 1, cacheRows: mtpRows)
            cb.commit()
        } else {
            encoderEnded = true
            hal.end(cb, enc)
        }
        let encodeUs = Date().timeIntervalSince(tEncode0) * 1e6
        cpuEncodeLedger.record(micros: encodeUs)
        cb.waitUntilCompleted()          // end BEFORE wait (20-min-hang law)
        if let pipe = coldEvents {
            let r = pipe.takeStepResult()
            pipe.tuneBridge()
            decodeTel.coldWaitUs += r.waitUs
            decodeTel.coldIOUs += r.ioUs
            decodeTel.coldStaged += r.staged
            if let f = r.failure { throw QWEngineDriverError.state("cold event staging: \(f)") }
        }
        flushPendingColdUnloads()
        endDecodeProfileStep()
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
        if cb.error != nil { commandBufferErrors += 1 }   // M1 triad (the
        stepCBs?.append(cb)
        // GPU time = union of every CB's GPU interval in this step (cold
        // staging splits the step into one CB per MoE layer); host-wait gaps
        // between CBs are idle, not busy.
        let gpuUs = Self.gpuBusyMicros(stepCBs ?? [cb])
        if gpuUs.isFinite, gpuUs > 0 { decodeGpuLedger.record(micros: gpuUs) }
        // The engine's own GPU busy ratio: the step's GPU busy time over the
        // wall time the step took. A ratio, not a percentage the host had to
        // guess at.
        let wallUs = Date().timeIntervalSince(tStep0) * 1e6
        if gpuUs.isFinite, gpuUs > 0, wallUs > 0 {
            QWTelemetryRegistry.shared.record("gpu.busy_ratio", gpuUs / wallUs, provenance: .measured)
        }
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
        // H3 FIX: a failed command buffer leaves sc.token STALE (last
        // step's id) and position untouched — returning it silently
        // corrupts the stream and every parity ledger after it. The count
        // stays (the triad gate); the caller now gets a throw instead of
        // garbage. Session state stays consistent: nothing was committed.
        if mtpCache != nil, cb.status != .completed {
            session.stateFailed = true
            throw QWEngineDriverError.state("MTP step GPU completion failed: \(String(describing: cb.error))")
        }
        if let e = cb.error {
            if mtpCache != nil { session.stateFailed = true }
            throw QWEngineDriverError.state("step cb: \(e.localizedDescription)")
        }

        session.position += 1
        session.generated += 1
        // H4 (full): the C09 decode door materialises row `pos` IN KERNEL
        // (split-0 epilogue) — the cache cursor must cover it so
        // rollback(by:)/snapshots price the true retained window. Monotonic,
        // zero GPU cost.
        kvCache.markWritten(session.kv, upTo: pos + 1)
        commitMTPSeeding(session, position: pos, count: 1, cacheRows: mtpRows)
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
        guard let active = activeConstraint else { return rawArgmax }
        let rawLogits = sc.logits.region.pointer.advanced(by: sc.logits.offset)
            .assumingMemoryBound(to: Float.self)
        let rowBuf = UnsafeBufferPointer(start: rawLogits, count: geometry.vocab)
        let token: Int
        if let allowed = try active.constraint.allowedTokens() {
            let repSet = active.sampler.config.repetitionPenalty == 1.0 ? Set<Int>() : Set(session.tokenHistory)
            token = try active.sampler.constrainedToken(logits: rowBuf, allowed: allowed, repetitionHistory: repSet)
        } else if active.sampler.config.isGreedy && active.sampler.config.repetitionPenalty == 1.0 {
            token = rawArgmax
        } else if active.sampler.config.isGreedy {
            token = try active.sampler.penalisedGreedyToken(logits: rowBuf, repetitionHistory: Set(session.tokenHistory))
        } else {
            let repArr = active.sampler.config.repetitionPenalty == 1.0 ? [] : session.tokenHistory
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
        guard Self.decodeProfileEnabled || Self.prefillProfileEnabled,
              let p = decodeTel.profiler, p.hasSegmentCapacity,
              let cb = activeCB, let cur = activeEnc else { return }
        cur.endEncoding()
        guard let enc = try? p.makeEncoder(commandBuffer: cb, stage: stage) else {
            fatalError("QW_DECODE_PROFILE: encoder creation failed after ending the active encoder")
        }
        hal.touchKeepAlive(enc)
        activeEnc = enc
    }

    /// Rotates `enc` to a fresh compute encoder on `activeCB` (preserving stage profiling when enabled).
    func rotateActiveEncoder(_ enc: inout any MTLComputeCommandEncoder, stage: String = "moe.experts_nax") throws {
        if (Self.decodeProfileEnabled || Self.prefillProfileEnabled),
           let p = decodeTel.profiler, p.hasSegmentCapacity,
           let cb = activeCB {
            enc.endEncoding()
            guard let next = try? p.makeEncoder(commandBuffer: cb, stage: stage) else {
                throw QWEngineDriverError.state("profile encoder rotation failed")
            }
            hal.touchKeepAlive(next)
            activeEnc = next
            enc = next
            return
        }
        guard let cb = activeCB else {
            throw QWEngineDriverError.state("rotateActiveEncoder requires activeCB")
        }
        try hal.rotateEncoder(cb, &enc)
        activeEnc = enc
    }

    private func beginDecodeProfileStep() {
        guard Self.decodeProfileEnabled, decodeTel.profiler == nil else { return }
        guard QWGPUStageProfiler<String>.isSupported(on: hal.device) else {
            FileHandle.standardError.write(Data("[decode-profile] stage sampling unsupported\n".utf8))
            return
        }
        decodeTel.profiler = try? QWGPUStageProfiler<String>(device: hal.device, segmentCapacity: 4096)
    }

    private func endDecodeProfileStep() {
        guard Self.decodeProfileEnabled, let p = decodeTel.profiler else { return }
        defer { p.reset() }
        guard let pairs = try? p.resolveByStage() else { return }
        var stepTotal: [String: Double] = [:]
        for (stage, us) in pairs { stepTotal[stage, default: 0] += us }
        for (stage, us) in stepTotal {
            let e = decodeTel.stageTotals[stage] ?? (0, 0)
            decodeTel.stageTotals[stage] = (e.us + us, e.n + 1)
        }
        decodeTel.profiledSteps += 1
        guard decodeTel.profiledSteps % 32 == 0 else { return }
        let n = Double(decodeTel.profiledSteps)
        let total = decodeTel.stageTotals.values.reduce(0) { $0 + $1.us } / n
        var s = String(format: "[decode-profile] steps=%d sampled_gpu=%.2fms/token\n", decodeTel.profiledSteps, total / 1000)
        for (k, v) in decodeTel.stageTotals.sorted(by: { $0.value.us > $1.value.us }) {
            s += String(format: "[decode-profile]   %-28@ %7.3f ms/token  %5.1f%%\n",
                        k as NSString, v.us / n / 1000, 100 * v.us / n / total)
        }
        FileHandle.standardError.write(Data(s.utf8))
    }

    func beginPrefillProfileChunk() {
        guard Self.prefillProfileEnabled, decodeTel.profiler == nil,
              QWGPUStageProfiler<String>.isSupported(on: hal.device) else { return }
        decodeTel.profiler = try? QWGPUStageProfiler<String>(device: hal.device, segmentCapacity: 4096)
        decodeTel.prefillChunkStart = (decodeTel.coldStageUs, decodeTel.coldWaitUs, decodeTel.coldIOUs,
                                       decodeTel.coldStaged, Date())
    }

    func endPrefillProfileChunk(tokens: Int) {
        guard Self.prefillProfileEnabled, let p = decodeTel.profiler else { return }
        defer { decodeTel.profiler = nil }
        guard let pairs = try? p.resolveByStage() else { return }
        var totals: [String: Double] = [:]
        for (stage, us) in pairs { totals[stage, default: 0] += us }
        let total = totals.values.reduce(0, +)
        var s = String(format: "[prefill-profile] tokens=%d sampled_gpu=%.2fms (%.1f tok/s)\n",
                       tokens, total / 1000, Double(tokens) / max(total, 1) * 1e6)
        if let c = decodeTel.prefillChunkStart {
            // Cold expert CBs run outside the stage sampler; wall vs sampled shows their share.
            s += String(format: "[prefill-profile]   wall=%.2fms cold: staged=%d stage=%.2fms (gpu-wait %.2fms, io %.2fms)\n",
                        Date().timeIntervalSince(c.t0) * 1000, decodeTel.coldStaged - c.staged,
                        (decodeTel.coldStageUs - c.stageUs) / 1000, (decodeTel.coldWaitUs - c.waitUs) / 1000,
                        (decodeTel.coldIOUs - c.ioUs) / 1000)
        }
        for (k, us) in totals.sorted(by: { $0.value > $1.value }) {
            s += String(format: "[prefill-profile]   %-28@ %8.2f ms  %5.1f%%\n",
                        k as NSString, us / 1000, 100 * us / max(total, 1))
        }
        FileHandle.standardError.write(Data(s.utf8))
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

    func stageColdExpertsSpan(
        moe: MoEW, s: MoEScratch, startRow: Int, totalRows: Int,
        onBeforeIO: ((inout any MTLComputeCommandEncoder, any MTLCommandBuffer, Int) throws -> Void)? = nil,
        onAfterGateUpIO: ((any MTLComputeCommandEncoder, Int) throws -> Void)? = nil
    ) throws -> (nextRow: Int, stagedCount: Int) {
        guard let streamer = moe.coldStreamer,
              let curCB = activeCB, let curEnc = activeEnc else { return (totalRows, 0) }
        let tStage0 = Date()
        defer { decodeTel.coldStageUs += Date().timeIntervalSince(tStage0) * 1e6 }
        hal.end(curCB, curEnc)
        stepCBs?.append(curCB)
        var resCB: (any MTLCommandBuffer)?
        // A dynamic-residency streamer maps router ids to bank slots on the CPU,
        // so resident compute cannot be encoded before the route readback.
        let canPrecommitResident = !streamer.isDynamic
            && (totalRows - startRow) * moe.expertsPerTok <= streamer.stagingSlots
        if canPrecommitResident, let onBeforeIO {
            guard let resPair = hal.begin() else {
                throw QWEngineDriverError.state("cold expert resident overlap cmdbuffer")
            }
            var resEnc = resPair.1
            try onBeforeIO(&resEnc, resPair.0, totalRows)
            hal.end(resPair.0, resEnc)
            resCB = resPair.0
        }
        let tWait0 = Date()
        curCB.waitUntilCompleted()
        decodeTel.coldWaitUs += Date().timeIntervalSince(tWait0) * 1e6
        if let err = curCB.error {
            commandBufferErrors += 1
            throw QWEngineDriverError.state("cold expert route cb: \(err.localizedDescription)")
        }
        let idxPtr = s.idx.region.buffer.contents()
            .advanced(by: s.idx.region.base + s.idx.offset)
            .assumingMemoryBound(to: UInt32.self)
        dumpRoute(streamer: streamer, ids: idxPtr + startRow * moe.expertsPerTok,
                  rows: totalRows - startRow, routes: moe.expertsPerTok)
        let prep = try streamer.prepareSpan(
            idxPtr: idxPtr, startRow: startRow, totalRows: totalRows, routes: moe.expertsPerTok)
        if !canPrecommitResident, prep.stagedCount > 0, let onBeforeIO {
            guard let resPair = hal.begin() else {
                throw QWEngineDriverError.state("cold expert resident overlap cmdbuffer")
            }
            var resEnc = resPair.1
            try onBeforeIO(&resEnc, resPair.0, prep.nextRow)
            hal.end(resPair.0, resEnc)
            resCB = resPair.0
        }
        var coldGuCB: (any MTLCommandBuffer)?
        decodeTel.coldStaged += prep.stagedCount
        let tIO0 = Date()
        defer { decodeTel.coldIOUs += Date().timeIntervalSince(tIO0) * 1e6 }
        if prep.stagedCount > 0 {
            if let onAfterGateUpIO {
                guard let guPair = hal.begin() else {
                    throw QWEngineDriverError.state("cold expert gateup overlap cmdbuffer")
                }
                try onAfterGateUpIO(guPair.1, prep.nextRow)
                guPair.1.endEncoding()
                struct BoxedCB: @unchecked Sendable { let cb: any MTLCommandBuffer }
                let guCB = BoxedCB(cb: guPair.0)
                coldGuCB = guPair.0
                try streamer.executeTwoPhaseBatch(
                    guBatch: prep.guBatch,
                    downBatch: prep.downBatch,
                    onGateUpLoaded: {
                        guCB.cb.commit()
                    }
                )
            } else {
                try streamer.executeBatch(prep.guBatch + prep.downBatch)
            }
        }
        if let resCB { stepCBs?.append(resCB) }
        if let coldGuCB { stepCBs?.append(coldGuCB) }
        if let resCB, let err = resCB.error {
            commandBufferErrors += 1
            throw QWEngineDriverError.state("cold expert resident overlap cb: \(err.localizedDescription)")
        }
        if let coldGuCB, let err = coldGuCB.error {
            commandBufferErrors += 1
            throw QWEngineDriverError.state("cold expert gateup overlap cb: \(err.localizedDescription)")
        }
        guard let nextPair = hal.begin() else {
            throw QWEngineDriverError.state("cold expert cmdbuffer")
        }
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
    func canStageColdExpertsMapped(moe: MoEW) -> Bool {
        // GPU span map + resident/cold overlap (tuning.prefillSpanMap); off restores the host-mapped `stageColdExpertsSpan`.
        guard tuning.prefillSpanMap, let st = moe.coldStreamer,
              activeCB != nil, activeEnc != nil else { return false }
        return st.canMapWholeSpan()
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
        hal.end(curCB, curEnc)
        stepCBs?.append(curCB)
        guard let resPair = hal.begin() else {
            throw QWEngineDriverError.state("span map resident cmdbuffer")
        }
        let resCB = resPair.0
        var resEnc = resPair.1
        try resident(&resEnc, resCB, mapped)
        hal.end(resCB, resEnc)

        let tWait0 = Date()
        curCB.waitUntilCompleted()
        decodeTel.coldWaitUs += Date().timeIntervalSince(tWait0) * 1e6
        if let err = curCB.error {
            commandBufferErrors += 1
            throw QWEngineDriverError.state("cold expert route cb: \(err.localizedDescription)")
        }
        let idxPtr = s.idx.region.buffer.contents()
            .advanced(by: s.idx.region.base + s.idx.offset)
            .assumingMemoryBound(to: UInt32.self)
        dumpRoute(streamer: streamer, ids: idxPtr, rows: rows, routes: routes)
        let prep = try streamer.prepareSpanMapped(
            ids: idxPtr, mapped: mapBuf.contents().assumingMemoryBound(to: Int32.self), count: count)
        decodeTel.coldStaged += prep.stagedCount

        var coldGuCB: (any MTLCommandBuffer)?
        let tIO0 = Date()
        if prep.stagedCount > 0 {
            guard let (guCB, guEnc) = hal.begin() else {
                throw QWEngineDriverError.state("span map cold gateup cmdbuffer")
            }
            try coldGateUp(guEnc)
            guEnc.endEncoding()
            struct BoxedCB: @unchecked Sendable { let cb: any MTLCommandBuffer }
            let boxed = BoxedCB(cb: guCB)
            coldGuCB = guCB
            try streamer.executeTwoPhaseBatch(guBatch: prep.guBatch, downBatch: prep.downBatch,
                                              onGateUpLoaded: { boxed.cb.commit() })
        }
        decodeTel.coldIOUs += Date().timeIntervalSince(tIO0) * 1e6
        stepCBs?.append(resCB)
        if let coldGuCB { stepCBs?.append(coldGuCB) }
        if let err = resCB.error {
            commandBufferErrors += 1
            throw QWEngineDriverError.state("span map resident cb: \(err.localizedDescription)")
        }
        if let coldGuCB, let err = coldGuCB.error {
            commandBufferErrors += 1
            throw QWEngineDriverError.state("span map cold gateup cb: \(err.localizedDescription)")
        }
        guard let nextPair = hal.begin() else {
            throw QWEngineDriverError.state("cold expert cmdbuffer")
        }
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
