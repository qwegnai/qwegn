// MTPVerification.swift — QWEngineDriver MTPVerification family (split from QWEngineDriver.swift; same-module extension).
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
    static let moeSpecExpertBudgetEnv: String? = ProcessInfo.processInfo.environment["QW_MOE_SPEC_EXPERT_BUDGET"]

    /// Returns the active MoE-Spec (arXiv:2602.16052) unique-expert budget across `rows` verification tokens,
    /// or `0` when all routed experts should be evaluated losslessly.
    func activeMoESpecExpertBudget(rows: Int) -> Int {
        guard rows > 1, rows <= 4, geometry.numExperts > 0, geometry.expertsPerTok > 0 else {
            return 0
        }
        if let raw = Self.moeSpecExpertBudgetEnv {
            if raw.lowercased() == "auto" {
                let acceptRate = mtpDraftProposals >= 4
                    ? Double(mtpDraftAccepted) / Double(max(1, mtpDraftProposals))
                    : 0.75
                return QWMTPAdaptiveDepth.moeSpecDraftExpertBudget(
                    geometry: geometry, verifyRows: rows, acceptRate: acceptRate)
            }
            if let explicit = Int(raw), explicit > 0 {
                return QWMTPAdaptiveDepth.moeSpecDraftExpertBudget(
                    geometry: geometry, verifyRows: rows, explicitBudget: explicit)
            }
        }
        return 0
    }

    func verifyTokensLocked(_ session: QWEngineSession, tokens: [Int],
                                   completed: ((any MTLCommandBuffer) throws -> Void)? = nil) throws -> QWVerifyWindow {
        guard verifiedSession == nil, let verifyWindow, let gdnReplayCheckpoint,
              (2...verifyWindow.rows).contains(tokens.count), !session.sessionFinished, !session.stateFailed,
              !session.verificationPending, session.trunkHiddenValid,
              session.position > 0, session.position <= ropeRows - tokens.count else {
            throw QWEngineDriverError.state("verify requires a seeded live session, B=2/3, and available positions")
        }
        try kvCache.validate(session.kv)
        guard kvCache.written(session.kv) == session.position else {
            throw QWEngineDriverError.state("verify trunk cursor mismatch")
        }
        for token in tokens where token < 0 || token >= geometry.vocab { throw QWEngineDriverError.badToken(token) }
        try claimMLAIndexerState(session)
        let end = session.position + tokens.count
        if end > session.kv.tokenCapacity {
            try kvCache.grow(session.kv, to: end)
            session.attnKVViews = try kvCache.layerViewsAll(session.kv, layers: attnLayers)
        }
        try kvCache.prepareWrites(session.kv, positions: session.position..<end)
        for (row, token) in tokens.enumerated() {
            try copyEmbedRow(token, to: verifyWindow.hiddenRow(row))
        }
        guard let cb = hal.queue.makeCommandBuffer() else { throw QWEngineDriverError.state("verify command buffer") }
        let stageProfiler = mtpVerificationProfiling
            ? try QWGPUStageProfiler<QWMTPVerificationStage>(device: hal.device,
                                                            segmentCapacity: geometry.numVirtualLayers * 7 + 3)
            : nil
        let tVerify0 = Self.decodeTraceEnabled ? Date() : nil
        self.activeCB = cb
        self.stepCBs = stageProfiler == nil ? [] : nil
        decodeTel.coldStageUs = 0
        decodeTel.coldWaitUs = 0
        decodeTel.coldIOUs = 0
        decodeTel.coldStaged = 0
        defer {
            self.activeCB = nil
            self.activeEnc = nil
            self.stepCBs = nil
        }
        verificationGpuMsByStage = nil
        try verifyWindow.encode(cb, hal: hal, head: head,
                                checkpoint: gdnReplayCheckpoint,
                                rec: session.recSlab, ring: session.ringSlab,
                                headStorage: headStorage,
                                q4Head: mtpVerifyHeadQ4 ? mtpDraftHead : nil,
                                activeRows: tokens.count,
                                unrotateForF16Head: true,
                                windowEncoder: {
                                    try self.encodeLayerMajorVerification(cb, session: session, window: verifyWindow,
                                                                          checkpoint: gdnReplayCheckpoint,
                                                                          activeRows: tokens.count,
                                                                          stageProfiler: stageProfiler)
                                },
                                currentCommandBuffer: { self.activeCB ?? cb },
                                stageProfiler: stageProfiler) { _ in }
        let finalCB = self.activeCB ?? cb
        finalCB.commit()
        finalCB.waitUntilCompleted()
        if let pipe = coldEvents {
            let r = pipe.takeStepResult()
            pipe.tuneBridge()
            decodeTel.coldWaitUs += r.waitUs
            decodeTel.coldIOUs += r.ioUs
            decodeTel.coldStaged += r.staged
            if let f = r.failure { throw QWEngineDriverError.state("cold event staging: \(f)") }
        }
        flushPendingColdUnloads()
        if let tVerify0 {
            let allCBs = (self.stepCBs ?? []) + [finalCB]
            let gpuUs = Self.gpuBusyMicros(allCBs)
            let wallUs = Date().timeIntervalSince(tVerify0) * 1e6
            FileHandle.standardError.write(Data(String(
                format: "[verify] pos=%d rows=%d wall=%.2fms gpu_busy=%.2fms cbs=%d cold_stage=%.2fms wait=%.2fms io=%.2fms staged=%d\n",
                session.position, tokens.count, wallUs / 1000, gpuUs / 1000, allCBs.count,
                decodeTel.coldStageUs / 1000, decodeTel.coldWaitUs / 1000, decodeTel.coldIOUs / 1000, decodeTel.coldStaged).utf8))
        }
        guard finalCB.status == .completed else {
            commandBufferErrors += 1
            session.stateFailed = true
            throw QWEngineDriverError.state("verify GPU failure: \(String(describing: finalCB.error))")
        }
        do {
            if let splitCBs = self.stepCBs, !splitCBs.isEmpty {
                for splitCB in splitCBs {
                    try completed?(splitCB)
                }
            }
            try completed?(finalCB)
            if let stageProfiler {
                let result = try stageProfiler.resolve(
                    totalGpuMicros: QWMTPStageTiming.gpuDurationMs(finalCB) * 1000)
                verificationGpuMsByStage = Dictionary(uniqueKeysWithValues:
                    result.gpuMicrosByStage.map { ($0.key.rawValue, $0.value / 1000) })
            }
        }
        catch {
            session.stateFailed = true
            throw error
        }
        do { _ = try verifyWindow.completedTargets(activeRows: tokens.count) }
        catch {
            session.stateFailed = true
            throw error
        }
        kvCache.markWritten(session.kv, upTo: end)
        session.verificationPending = true
        verifiedSession = session
        verifiedInputs = tokens
        return verifyWindow
    }

    /// Reorders independent projection rows, not each row's arithmetic.
    /// GDN executes rows in causal order. Shared attention keeps per-query
    /// causal limits after materializing the complete verification window.
    func encodeLayerMajorVerification(_ cb: any MTLCommandBuffer, session: QWEngineSession,
                                              window: QWVerifyWindow,
                                              checkpoint: QWGDNReplayCheckpoint,
                                              activeRows: Int,
                                              stageProfiler: QWGPUStageProfiler<QWMTPVerificationStage>? = nil) throws {
        var currentCB = cb
        func makeEncoder(_ stage: QWMTPVerificationStage) throws -> any MTLComputeCommandEncoder {
            let activeBuf = self.activeCB ?? currentCB
            currentCB = activeBuf
            if let stageProfiler {
                return try stageProfiler.makeEncoder(commandBuffer: activeBuf, stage: stage)
            }
            guard let encoder = activeBuf.makeComputeCommandEncoder() else {
                throw QWEngineDriverError.state("layer-major verification encoder")
            }
            return encoder
        }
        var enc = try makeEncoder(.norm)
        var encoderOpen = true
        self.activeCB = currentCB
        self.activeEnc = enc
        defer {
            if encoderOpen { enc.endEncoding() }
            self.activeEnc = nil
        }
        func beginStage(_ stage: QWMTPVerificationStage) throws {
            guard stageProfiler != nil else { return }
            enc.endEncoding()
            encoderOpen = false
            enc = try makeEncoder(stage)
            encoderOpen = true
            self.activeEnc = enc
        }
        guard (2...window.rows).contains(activeRows) else {
            throw QWEngineDriverError.state("layer-major active row count is invalid")
        }
        let h = geometry.hidden
        guard let normalized = window.normalized,
              let mlpRows = window.mlpRows,
              let mtpBatchedTrunk,
              let mtpBatchedMLP,
              let partials = mtpSharedAttentionPartials else {
            throw QWEngineDriverError.state("batched trunk verification buffers are absent")
        }
        let plan = geometry.executionPlan
        try encodeRMSNormFast(enc, x: window.hidden, gamma: layersW[0].ln1, out: normalized, rows: activeRows)
        for (layer, p) in plan.enumerated() {
            let weights = layersW[p]
            let nextP = (layer + 1 < plan.count) ? plan[layer + 1] : nil
            let gamma = nextP.map { layersW[$0].ln1 } ?? finalNorm
            if let mhc = weights.mhc, let mhcCfg = geometry.mhcConfig,
               let mhcStream = window.mhcStream, let mhcControls = window.mhcControls {
                let prevP = layer > 0 ? plan[layer - 1] : -1
                let hasDeferredPrevFFN = (p != mhcCfg.start) && (prevP >= mhcCfg.start && prevP < mhcCfg.end - 1)
                if p == mhcCfg.start {
                    try encodeMHCEnter(enc, x: window.hidden, stream: mhcStream, rows: activeRows)
                    enc.memoryBarrier(scope: .buffers)
                }
                let prevBranch = hasDeferredPrevFFN
                    ? prefixRows(mtpBatchedMLP.downRows, rows: activeRows, cols: h, dt: .f16)
                    : nil
                try encodeMHCRouterNorm(enc, stream: mhcStream, w: mhc.attnW, b: mhc.attnB,
                                        gamma: weights.ln1, controls: mhcControls, out: normalized,
                                        rows: activeRows, mergeBranchOut: prevBranch)
                enc.memoryBarrier(scope: .buffers)
            }
            try beginStage(.inputProjection)
            let activeNormalized = prefixRows(normalized, rows: activeRows, cols: h, dt: .f16)
            let isGDN = (weights.kind == .gdn || weights.kind == .gdnMoe)
            try encodeVerificationLayerInputProjection(
                enc, weights: weights, activeNormalized: activeNormalized,
                mtpBatchedTrunk: mtpBatchedTrunk, mtpBatchedMLP: mtpBatchedMLP,
                activeRows: activeRows)
            try beginStage(isGDN ? .gdn : .attention)
            try encodeVerificationLayerCore(
                enc, session: session, layer: layer, weights: weights,
                activeNormalized: activeNormalized,
                checkpoint: checkpoint, mtpBatchedTrunk: mtpBatchedTrunk,
                mtpBatchedMLP: mtpBatchedMLP, partials: partials,
                activeRows: activeRows)
            try beginStage(.outputProjection)
            let weight = isGDN ? weights.gdn!.out : weights.attn!.o
            let outCols = isGDN ? geometry.gdnZChannels : geometry.qHeads * geometry.headDim
            try mtpBatchedMLP.encodeProjection(enc,
                input: prefixRows(mtpBatchedTrunk.outputInputs, rows: activeRows,
                                  cols: outCols, dt: .f16),
                weight: weight, output: mtpBatchedMLP.downRows,
                activeRows: activeRows, preparedSums: isGDN)
            if let mhc = weights.mhc, geometry.mhcConfig != nil,
               let mhcStream = window.mhcStream, let mhcControls = window.mhcControls {
                enc.memoryBarrier(scope: .buffers)
                try encodeMHCRouterNorm(enc, stream: mhcStream, w: mhc.ffnW, b: mhc.ffnB,
                                        gamma: weights.ln2, controls: mhcControls, out: normalized,
                                        rows: activeRows,
                                        mergeBranchOut: prefixRows(mtpBatchedMLP.downRows, rows: activeRows, cols: h, dt: .f16))
                enc.memoryBarrier(scope: .buffers)
            } else {
                try encodeRMSNormResidualFast(enc, x: mtpBatchedMLP.downRows, gamma: weights.ln2,
                    out: normalized, residual: window.hidden, rows: activeRows)
                enc.memoryBarrier(scope: .buffers)
            }
            if let moe = weights.moe {
                guard let moeScratch = window.moeScratch, let moeDn32 = window.moeDn32 else {
                    throw QWEngineDriverError.state("MTP verification MoE scratch is absent")
                }
                try beginStage(.mlpGateUp)
                try beginStage(.mlpDown)
                self.activeCB = currentCB
                self.activeEnc = enc
                try QWMoEChain.encode(
                    enc: enc, hal: hal, moe: moe, s: moeScratch,
                    x16: prefixRows(normalized, rows: activeRows, cols: h, dt: .f16),
                    dn32: prefixRows(moeDn32, rows: activeRows, cols: h, dt: .f32),
                    dn16: prefixRows(mtpBatchedMLP.downRows, rows: activeRows, cols: h, dt: .f16),
                    rows: activeRows, geometry: geometry, driver: self)
                if let active = self.activeEnc { enc = active }
                if let activeBuf = self.activeCB { currentCB = activeBuf }
                enc.memoryBarrier(scope: .buffers)
            } else {
                try beginStage(.mlpGateUp)
                let activeMlpRows = prefixRows(mlpRows, rows: activeRows,
                                               cols: geometry.intermediate, dt: .f16)
                let fuseFwht = !weights.gu.view.isIncumbentContainer && geometry.intermediate > 0 && (geometry.intermediate % 32 == 0)
                try mtpBatchedMLP.encodeGateUp(enc,
                    input: prefixRows(normalized, rows: activeRows, cols: h, dt: .f16),
                    weight: weights.gu.view,
                    output: activeMlpRows,
                    activeRows: activeRows,
                    fuseFwht32: fuseFwht)
                if !fuseFwht {
                    encodeFwht32Fast(enc, x: activeMlpRows)
                    enc.memoryBarrier(scope: .buffers)
                }
                try beginStage(.mlpDown)
                try mtpBatchedMLP.encodeDown(enc,
                    input: activeMlpRows,
                    weight: weights.down, activeRows: activeRows)
            }
            try beginStage(.norm)
            if let _ = weights.mhc, let mhcCfg = geometry.mhcConfig,
               let mhcStream = window.mhcStream, let mhcControls = window.mhcControls {
                let isLeave = (p == mhcCfg.end - 1)
                let nextIsMHC = nextP.map { $0 >= mhcCfg.start && $0 < mhcCfg.end } ?? false
                let deferPostFFN = !isLeave && nextIsMHC && (nextP != mhcCfg.start)
                if !deferPostFFN {
                    enc.memoryBarrier(scope: .buffers)
                    try encodeMHCMerge(enc, stream: mhcStream, branchOut: mtpBatchedMLP.downRows,
                                       controls: mhcControls, xOut: window.hidden,
                                       nextGamma: gamma, nextNormOut: normalized,
                                       rows: activeRows, isLeave: isLeave, sinkhornIters: mhcCfg.iterations,
                                       hasNextNorm: isLeave)
                    enc.memoryBarrier(scope: .buffers)
                }
            } else {
                try encodeRMSNormResidualFast(enc, x: mtpBatchedMLP.downRows, gamma: gamma,
                    out: normalized, residual: window.hidden, rows: activeRows)
                enc.memoryBarrier(scope: .buffers)
            }
            if let dflashDrafter, p < dflashDrafter.targetLayerSlotMap.count {
                let slot = dflashDrafter.targetLayerSlotMap[p]
                if slot >= 0 {
                    dflashDrafter.encodeCaptureTargetHidden(
                        enc, source: window.hidden, srcRowOffset: 0, dstRowOffset: 0, rows: activeRows, slot: slot)
                    enc.memoryBarrier(scope: .buffers)
                }
            }
        }
        try QWVerifyWindow.encodeCopy(enc, hal: hal,
            source: prefixRows(normalized, rows: activeRows, cols: h, dt: .f16),
            destination: prefixRows(window.hidden, rows: activeRows, cols: h, dt: .f16))
        enc.memoryBarrier(scope: .buffers)
    }

    /// Diagnostic two-phase round owner. Output is tentative until commit succeeds;
    /// scratch stays leased across prefix selection until commit/discard/finish.
    /// No user callback runs under flight, and ordinary generation never calls it.
    func prepareGreedyRound(_ session: QWEngineSession, maxNew: Int,
                            stopIDs: Set<Int> = [],
                            permit: QWGenerationPermit? = nil) throws -> QWPreparedGreedyRound {
        try prepareRound(session, maxNew: maxNew, stopIDs: stopIDs, stochastic: false, permit: permit)
    }

    func prepareStochasticRound(_ session: QWEngineSession, maxNew: Int,
                                stopIDs: Set<Int> = [],
                                permit: QWGenerationPermit? = nil) throws -> QWPreparedStochasticRound {
        QWPreparedStochasticRound(authorization:
            try prepareRound(session, maxNew: maxNew, stopIDs: stopIDs, stochastic: true, permit: permit))
    }

    func prepareRound(_ session: QWEngineSession, maxNew: Int, stopIDs: Set<Int>,
                              stochastic: Bool, permit: QWGenerationPermit?) throws -> QWPreparedGreedyRound {
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate(permit)
        guard !mtpAdaptiveDepth || !stochastic else {
            throw QWEngineDriverError.state("QW_MTP_ADAPTIVE_DEPTH currently supports greedy rounds only")
        }
        let profileStart = mtpStageProfiling ? DispatchTime.now().uptimeNanoseconds : nil
        if let cb = session.lastCommitCB, cb.status == .error {
            session.stateFailed = true
        }
        guard (session.stochastic != nil) == stochastic,
              session.stochastic?.stream.isRoundActive != true,
              session.stochastic?.stream.drawsConsumed == (stochastic ? session.roundRecord.samplerDrawIndex : nil),
              verifiedSession == nil, let verifyWindow, (mtpDraftEncoder != nil || dflashDrafter != nil),
              let mtpCache, let handle = session.mtpKV, let hidden = session.trunkHidden,
              let pending = session.roundRecord.pendingToken,
              !session.sessionFinished, !session.stateFailed, !session.verificationPending,
              session.trunkHiddenValid, session.position > 0,
              session.position <= ropeRows - verifyWindow.rows,
              session.roundRecord.consumedCount == session.position,
              session.roundRecord.absolutePosition == session.position,
              session.roundRecord.outputCount == session.generated,
              maxNew > 0, session.generated <= Int.max - verifyWindow.rows,
              (0..<geometry.vocab).contains(pending),
              stopIDs.allSatisfy({ (0..<geometry.vocab).contains($0) }) else {
            throw QWEngineDriverError.state("greedy round requires a seeded boundary and valid output budget")
        }
        let count = verifyWindow.rows - 1, basePairs = session.roundRecord.mtpPhysicalRows
        try kvCache.validate(session.kv)
        try mtpCache.validate(handle)
        guard kvCache.written(session.kv) == session.position,
              mtpCache.written(handle) == basePairs,
              basePairs >= 0, basePairs <= QWKVCache.maxTokensPerSeq - count,
              session.position <= (mtpDraftEncoder?.positionCapacity ?? ropeRows) - count else {
            throw QWEngineDriverError.state("greedy round cache or RoPE boundary mismatch")
        }
        try claimMLAIndexerState(session)
        // Reserve both windows before draft execution; never grow a cache while
        // one of its command buffers is in flight.
        if session.position + verifyWindow.rows > session.kv.tokenCapacity {
            try kvCache.grow(session.kv, to: session.position + verifyWindow.rows)
            session.attnKVViews = try kvCache.layerViewsAll(session.kv, layers: attnLayers)
        }
        if basePairs + count > handle.tokenCapacity {
            try mtpCache.grow(handle, to: basePairs + count)
            session.mtpKVViews = [try mtpCache.layerViews(handle, layer: 0)]
        }
        try kvCache.prepareWrites(session.kv, positions: session.position..<(session.position + verifyWindow.rows))
        try mtpCache.prepareWrites(handle, positions: basePairs..<(basePairs + count))
        if session.attnKVViews.isEmpty {
            session.attnKVViews = try kvCache.layerViewsAll(session.kv, layers: attnLayers)
        }
        let views: (k: QWKVLayerView, v: QWKVLayerView)
        if let current = session.mtpKVViews.first {
            views = current
        } else {
            let v = try mtpCache.layerViews(handle, layer: 0)
            session.mtpKVViews = [v]
            views = v
        }
        let boundary = try greedyRoundBoundary(session)
        lastMTPRoundProfile = nil
        pendingMTPProfile = nil
        pendingMTPAdaptiveEntropy = mtpAdaptiveDepth ? [] : nil
        preparedStochasticWitness = nil
        do {
            let owner = session.stochastic
            var proposals: [QWMTPGumbelProposal] = []
            var drafts: [Int] = []
            let draftAccumulator = profileStart.map { _ in QWMTPStageAccumulator() }
            let draftCompleted: ((any MTLCommandBuffer) throws -> Void)? = draftAccumulator.map { accumulator in
                { try accumulator.record($0) }
            }
            if let dflashDrafter, let dSlot = session.dflashSlot {
                if let owner {
                    try owner.stream.beginGumbelRound(drafts: count, vocabulary: geometry.vocab, maxDrafts: 15)
                }
                let proposed: (drafts: [Int], proposals: [QWMTPGumbelProposal])
                do {
                    proposed = try dflashDrafter.propose(
                        hal: hal, slot: dSlot, cacheOffset: session.dflashCacheOffset,
                        pendingToken: pending, draftCount: count, head: head,
                        headStorage: headStorage, draftHead: mtpDraftHead,
                        embedding: { try self.copyEmbedRow($0, to: $1) },
                        stochasticOwner: owner, completed: draftCompleted)
                } catch {
                    commandBufferErrors += 1
                    throw error
                }
                drafts = proposed.drafts
                proposals = proposed.proposals
                mtpDraftProposals += drafts.count
            } else {
                guard let mtpDraftEncoder else {
                    throw QWEngineDriverError.state("MTP draft encoder is absent")
                }
                let effectiveCount: Int
                if mtpAdaptiveDepth && geometry.numExperts > 0 && mtpDraftProposals >= 8 && count > 1 {
                    let trunkBits = layersW.first?.gu.view.bits ?? 4
                    let expertBits = layersW.compactMap({ $0.moe?.gateUp.bits }).first ?? trunkBits
                    let acceptRate = Double(mtpDraftAccepted) / Double(max(1, mtpDraftProposals))
                    effectiveCount = QWMTPAdaptiveDepth.optimalDraftCount(
                        geometry: geometry,
                        trunkBits: trunkBits,
                        expertBits: expertBits,
                        acceptRate: acceptRate,
                        maxDrafts: count)
                } else {
                    effectiveCount = count
                }
                if let owner { try owner.stream.beginGumbelRound(drafts: effectiveCount, vocabulary: geometry.vocab) }
                var draftHistory = session.tokenHistory + [pending]
                let sampledToken: ((QWTensorValue) throws -> Int)? = owner.map { owner in
                    { logits in
                        let proposal = try owner.stream.sampleGumbelProposal(
                            owner.distribution(logits: logits, repetitionHistory: draftHistory))
                        proposals.append(proposal)
                        draftHistory.append(proposal.token)
                        return proposal.token
                    }
                }
                let needsAdaptiveStop = mtpAdaptiveDepth && (effectiveCount > 1 || mtpStageProfiling)
                let stopAfterDraft: ((QWTensorValue, Int) throws -> Bool)? = needsAdaptiveStop ? { logits, completedDrafts in
                    if completedDrafts >= effectiveCount && !self.mtpStageProfiling {
                        return false
                    }
                    let values = logits.region.pointer.advanced(by: logits.offset).assumingMemoryBound(to: Float.self)
                    let entropy = try QWMTPAdaptiveDepth.normalizedEntropy(
                        UnsafeBufferPointer(start: values, count: logits.shape.elementCount))
                    self.pendingMTPAdaptiveEntropy?.append(entropy)
                    return try QWMTPAdaptiveDepth.shouldStop(normalizedEntropy: entropy,
                        threshold: self.mtpAdaptiveEntropyThreshold,
                        completedDrafts: completedDrafts, maximumDrafts: effectiveCount)
                } : nil
                let tDraft0 = Self.decodeTraceEnabled ? Date() : nil
                drafts = try QWMTPDraftChain.run(
                    hal: hal, draft: mtpDraftEncoder, headStorage: headStorage,
                    draftHead: mtpDraftHead, hidden: hidden, kv: views,
                    pendingToken: pending, drafts: effectiveCount, position: session.position, cacheRows: basePairs,
                    selection: .gpuChecked,
                    deviceEmbedding: stochastic ? nil : mtpDeviceEmbedding,
                    embedding: { try self.copyEmbedRow($0, to: $1) },
                    sampledToken: sampledToken,
                    shouldStopAfterDraft: stopAfterDraft,
                    writeLogitsOnFinalDraft: mtpStageProfiling,
                    completed: draftCompleted,
                    gpuFailure: { self.commandBufferErrors += 1 })
                if let tDraft0 {
                    FileHandle.standardError.write(Data(String(
                        format: "[draft] pos=%d count=%d wall=%.2fms\n",
                        session.position, drafts.count, Date().timeIntervalSince(tDraft0) * 1000).utf8))
                }
                mtpDraftProposals += drafts.count
            }
        

            mtpCache.markWritten(handle, upTo: basePairs + drafts.count)
            let verifyStart = mtpStageProfiling ? DispatchTime.now().uptimeNanoseconds : nil
            let verifyAccumulator = verifyStart.map { _ in QWMTPStageAccumulator() }
            let verifyCompleted: ((any MTLCommandBuffer) throws -> Void)? = verifyAccumulator.map { accumulator in
                { try accumulator.record($0) }
            }
            let window = try verifyTokensLocked(session, tokens: [pending] + drafts, completed: verifyCompleted)
            let decision: QWGreedyRoundDecision
            let evidence: QWStochasticRoundEvidence?
            let margin = effectiveRelaxedMargin
            if let owner {
                var targets: [QWSamplingDistribution] = []
                for row in 0..<(drafts.count + 1) {
                    let history = session.tokenHistory + [pending] + Array(drafts.prefix(row))
                    targets.append(try owner.distribution(logits: window.logitRow(row),
                                                          repetitionHistory: Array(history)))
                }
                let result = margin > 0
                    ? try owner.stream.resolveRelaxedGumbelRound(proposals: proposals,
                                                                 targetDistributions: targets,
                                                                 margin: Double(margin))
                    : try owner.stream.resolveGumbelRound(proposals: proposals,
                                                          targetDistributions: targets)
                decision = try QWGreedyRoundDecision.sampled(geometry: geometry, 
                    proposals: proposals, result: result, maxNew: maxNew, stopIDs: stopIDs,
                    maxDrafts: dflashDrafter != nil ? 15 : 5)
                preparedStochasticWitness = QWStochasticRoundWitness(owner: owner,
                    drawEnd: owner.stream.drawsConsumed)
                evidence = QWStochasticRoundEvidence(proposals: proposals, targets: targets, result: result,
                    drawStart: boundary.record.samplerDrawIndex, drawEnd: owner.stream.drawsConsumed)
            } else {
                var rawDecision = try window.greedyDecision(drafts: drafts, maxNew: maxNew, stopIDs: stopIDs,
                                                            activeRows: drafts.count + 1,
                                                            relaxedMargin: margin,
                                                            relaxedMinProb: effectiveRelaxedMinProb)
                if let boundary = activeConstraint?.constraint.speculationBoundaryToken,
                   let idx = rawDecision.tokens.firstIndex(of: boundary),
                   idx + 1 < rawDecision.tokens.count {
                    rawDecision = QWGreedyRoundDecision(
                        acceptedDrafts: rawDecision.acceptedDrafts,
                        tokens: Array(rawDecision.tokens.prefix(idx + 1)),
                        stopped: false,
                        stoppedToken: nil
                    )
                }
                decision = rawDecision
                evidence = nil
            }
            let lease = QWGreedyRoundLease(boundary: boundary, decision: decision, stochasticEvidence: evidence)
            if let profileStart, let verifyStart, let draftAccumulator, let verifyAccumulator {
                pendingMTPProfile = (
                    try QWMTPStageTiming(hostMs: Double(verifyStart - profileStart) / 1_000_000,
                                         gpuMs: draftAccumulator.gpuMs, commandBuffers: draftAccumulator.commandBuffers),
                    try QWMTPStageTiming(hostMs: Double(DispatchTime.now().uptimeNanoseconds - verifyStart) / 1_000_000,
                                         gpuMs: verifyAccumulator.gpuMs, commandBuffers: verifyAccumulator.commandBuffers))
            }
            preparedGreedyLease = lease
            return lease.prepared
        } catch {
            // A submitted prefix may have changed pair/trunk bytes. Without a
            // completed verified lease, do not expose an apparently retryable round.
            session.stateFailed = true
            session.stochastic?.stream.cancelActiveRound()
            if let owner = session.stochastic { session.roundRecord.samplerDrawIndex = owner.stream.drawsConsumed }
            throw error
        }
    }

    func greedyRoundBoundary(_ session: QWEngineSession) throws -> QWGreedyRoundBoundary {
        guard let mtpCache, let handle = session.mtpKV else {
            throw QWEngineDriverError.state("greedy round pair cache is absent")
        }
        return QWGreedyRoundBoundary(position: session.position, generated: session.generated,
                                    record: session.roundRecord, hiddenValid: session.trunkHiddenValid,
                                    trunkRewind: try kvCache.rollbackGeneration(session.kv),
                                    pairRewind: try mtpCache.rollbackGeneration(handle))
    }

    /// A guarded round publication door, not generation integration. The draft
    /// owner must have materialized N MTP pairs from the saved round boundary.
    func commitGreedyVerification(_ session: QWEngineSession, round: QWPreparedGreedyRound,
                                  drain: Bool = true,
                                  permit: QWGenerationPermit? = nil) throws {
        try commitPreparedVerification(session, round: round, stochastic: false, drain: drain, permit: permit)
    }

    func commitStochasticVerification(_ session: QWEngineSession, round: QWPreparedStochasticRound,
                                      drain: Bool = true,
                                      permit: QWGenerationPermit? = nil) throws {
        try commitPreparedVerification(session, round: round.authorization, stochastic: true, drain: drain, permit: permit)
    }

    func commitPreparedVerification(_ session: QWEngineSession, round: QWPreparedGreedyRound,
                                            stochastic: Bool, drain: Bool = true,
                                            permit: QWGenerationPermit?) throws {
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate(permit)
        let profileStart = mtpStageProfiling ? DispatchTime.now().uptimeNanoseconds : nil
        guard (preparedStochasticWitness != nil) == stochastic,
              (session.stochastic != nil) == stochastic,
              verifiedSession === session, session.verificationPending,
              !session.stateFailed, !session.sessionFinished,
              let verifyWindow, let gdnReplayCheckpoint, let mtpCache, let mtpHandle = session.mtpKV,
              let preparedGreedyLease,
              let trunkHidden = session.trunkHidden, (2...verifyWindow.rows).contains(verifiedInputs.count),
              verifiedInputs.first == session.roundRecord.pendingToken else {
            throw QWEngineDriverError.state("greedy commit requires a live verified draft round")
        }
        try preparedGreedyLease.validate(round, current: greedyRoundBoundary(session))
        try preparedStochasticWitness?.validate(session)
        guard !mtpStageProfiling || pendingMTPProfile != nil else {
            throw QWEngineDriverError.state("MTP profile is missing its prepared stages")
        }
        let decision = round.decision
        let drafts = Array(verifiedInputs.dropFirst())
        if !stochastic {
            let full = try verifyWindow.greedyDecision(drafts: drafts, maxNew: verifiedInputs.count,
                                                       activeRows: verifiedInputs.count,
                                                       relaxedMargin: effectiveRelaxedMargin,
                                                       relaxedMinProb: effectiveRelaxedMinProb)
            guard decision.acceptedDrafts == full.acceptedDrafts,
                  decision.tokens == Array(full.tokens.prefix(decision.tokens.count)),
                  decision.tokens.count <= full.tokens.count else {
                throw QWEngineDriverError.state("greedy commit decision does not match verified target rows")
            }
        }
        guard session.generated <= Int.max - decision.tokens.count else {
            throw QWEngineDriverError.state("round output count overflows")
        }
        try kvCache.validate(session.kv)
        try mtpCache.validate(mtpHandle)
        let basePairs = session.roundRecord.mtpPhysicalRows
        guard basePairs >= 0, basePairs <= QWKVCache.maxTokensPerSeq - drafts.count,
              kvCache.written(session.kv) == session.position + verifiedInputs.count,
              mtpCache.written(mtpHandle) == basePairs + drafts.count else {
            throw QWEngineDriverError.state("greedy commit requires complete trunk and draft pair windows")
        }
        let retained = decision.retainedRows
        let retainedPairs = try decision.retainedPairRows(drafts: drafts.count)
        guard let cb = hal.queue.makeCommandBuffer() else { throw QWEngineDriverError.state("greedy commit command buffer") }
        try verifyWindow.encodeRetainedPrefix(cb, hal: hal, checkpoint: gdnReplayCheckpoint, decoder: gdnDecode,
                                              rec: session.recSlab, ring: session.ringSlab,
                                              trunkHidden: trunkHidden, activeRows: verifiedInputs.count,
                                              retaining: retained)
        if let dflashDrafter, let dSlot = session.dflashSlot, retained > 0 {
            try dflashDrafter.encodeContextUpdate(cb, slot: dSlot, startPosition: session.position, totalCount: retained)
        }
        if !mtpStageProfiling && !drain {
            session.lastCommitCB = cb
            cb.addCompletedHandler { [weak self, weak session] buffer in
                if buffer.status != .completed {
                    self?.commandBufferErrors += 1
                    session?.stateFailed = true
                }
            }
        }
        cb.commit()
        if mtpStageProfiling || drain {
            cb.waitUntilCompleted()
            guard cb.status == .completed else {
                commandBufferErrors += 1
                session.stateFailed = true
                throw QWEngineDriverError.state("greedy commit GPU failure: \(String(describing: cb.error))")
            }
        }
        let resolutionGPU: Double?
        do {
            resolutionGPU = mtpStageProfiling ? try QWMTPStageTiming.gpuDurationMs(cb) : nil
            let trunkRollback = verifiedInputs.count - retained
            if trunkRollback > 0 {
                try kvCache.rollback(session.kv, by: trunkRollback)
                session.attnKVViews = try kvCache.layerViewsAll(session.kv, layers: attnLayers)
            }
            let pairRollback = drafts.count - retainedPairs
            if pairRollback > 0 {
                try mtpCache.rollback(mtpHandle, by: pairRollback)
                session.mtpKVViews = [try mtpCache.layerViews(mtpHandle, layer: 0)]
            }
        

        } catch {
            // GPU state has already resolved; partly changed host tables cannot
            // be retried as a fresh round.
            session.stateFailed = true
            throw error
        }
        let profile: QWMTPRoundProfile?
        do {
            if let profileStart, let resolutionGPU {
                guard let pendingMTPProfile else {
                    throw QWEngineDriverError.state("MTP profile is missing its prepared stages")
                }
                profile = QWMTPRoundProfile(drafts: drafts.count, retainedRows: retained,
                    draft: pendingMTPProfile.draft, verification: pendingMTPProfile.verification,
                    resolution: try QWMTPStageTiming(
                        hostMs: Double(DispatchTime.now().uptimeNanoseconds - profileStart) / 1_000_000,
                        gpuMs: resolutionGPU, commandBuffers: 1),
                    draftTokens: drafts, acceptedDrafts: decision.acceptedDrafts,
                    backend: mtpBackend.rawValue,
                    verificationGpuMsByStage: verificationGpuMsByStage,
                    adaptiveEntropyByDraft: pendingMTPAdaptiveEntropy)
            } else {
                profile = nil
            }
        } catch {
            session.stateFailed = true
            throw error
        }
        session.position += retained
        if dflashDrafter != nil {
            session.dflashCacheOffset = session.position
        }
        session.generated += decision.tokens.count
        session.roundRecord.consumedCount = session.position
        session.roundRecord.absolutePosition = session.position
        session.roundRecord.outputCount = session.generated
        session.roundRecord.mtpPhysicalRows = basePairs + retainedPairs
        mtpDraftAccepted += decision.acceptedDrafts
    

        session.tokenHistory.append(contentsOf: decision.tokens)
        if let active = activeConstraint {
            for tok in decision.tokens {
                try active.constraint.accept(tok)
            }
        }
        if let owner = session.stochastic { session.roundRecord.samplerDrawIndex = owner.stream.drawsConsumed }
        if let pending = decision.pendingToken { session.roundRecord.pendingToken = pending }
        if retained > 0 {
            session.roundRecord.trunkHiddenSeed = nil // GPU carry is authoritative.
        }
        session.verificationPending = false
        verifiedSession = nil
        verifiedInputs.removeAll(keepingCapacity: true)
        self.preparedGreedyLease = nil
        preparedStochasticWitness = nil
        pendingMTPProfile = nil
        pendingMTPAdaptiveEntropy = nil
        lastMTPRoundProfile = profile
    }

}
