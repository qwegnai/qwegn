import Metal
import QWAlloc
import QWCore
import QWHAL
import QWHead
import QWKV
import QWMTP

/// Ordered diagnostic backend. Mapped embedding feedback needs one completion
/// wait per chain; the CPU-reference gather needs one wait per draft.
enum QWMTPDraftChain {
    enum Selection {
        case cpuReference
        case gpuChecked
    }

    static func run(hal: QWHal, draft: QWMTPDraftEncoder,
                    headWeight: QWTensorValue? = nil, headStorage: QWHeadStorage? = nil,
                    draftHead: QWMTPTDraftHead? = nil,
                    hidden: QWTensorValue, kv: (k: QWKVLayerView, v: QWKVLayerView),
                    pendingToken: Int, drafts: Int, position: Int, cacheRows: Int,
                    mropeDelta: Int = 0,
                    selection: Selection = .cpuReference,
                    deviceEmbedding: QWMTPEmbeddingTable? = nil,
                    embedding: (Int, QWTensorValue) throws -> Void,
                    sampledToken: ((QWTensorValue) throws -> Int)? = nil,
                    shouldStopAfterDraft: ((QWTensorValue, Int) throws -> Bool)? = nil,
                    writeLogitsOnFinalDraft: Bool = true,
                    completed: ((any MTLCommandBuffer) throws -> Void)? = nil,
                    gpuFailure: () -> Void) throws -> [Int] {
        guard let storage = headStorage ?? headWeight.map(QWHeadStorage.f16) else {
            throw QWEngineDriverError.state("draft chain requires a target head")
        }
        let vocabulary = draft.scratch.draftLogits.shape.elementCount
        guard (1...15).contains(drafts), (0..<vocabulary).contains(pendingToken),
              position > 0, position <= draft.positionCapacity - drafts,
              cacheRows >= 0, cacheRows <= QWKVCache.maxTokensPerSeq - drafts else {
            throw QWEngineDriverError.state("draft chain token, count, or boundary is invalid")
        }
        var tokens: [Int] = []
        if let table = deviceEmbedding, shouldStopAfterDraft == nil {
            guard sampledToken == nil, selection == .gpuChecked else {
                throw QWEngineDriverError.state("device draft embedding geometry/selection mismatch")
            }
            return try runDeviceAndDrain(
                hal: hal, draft: draft, storage: storage, draftHead: draftHead,
                hidden: hidden, kv: kv, pendingToken: pendingToken, drafts: drafts,
                position: position, cacheRows: cacheRows, mropeDelta: mropeDelta,
                table: table, stochastic: nil, completed: completed, gpuFailure: gpuFailure)
        }
        let useDeviceEmbedding = deviceEmbedding != nil
        if let table = deviceEmbedding {
            guard sampledToken == nil, selection == .gpuChecked,
                  table.vocabulary == vocabulary,
                  table.hidden == draft.scratch.embedding.shape.elementCount else {
                throw QWEngineDriverError.state("device draft embedding geometry/selection mismatch")
            }
            let selector = draft.scratch.selector
            selector.token.region.pointer.advanced(by: selector.token.offset)
                .assumingMemoryBound(to: Int32.self).pointee = Int32(pendingToken)
        }
        let activeDriver = draft.driver
        if draft.hasColdMoE && activeDriver == nil {
            throw QWEngineDriverError.state("MTP draft MoE cold experts require an attached driver")
        }
        var token = pendingToken
        for row in 0..<drafts {
            if !useDeviceEmbedding {
                try embedding(token, draft.scratch.embedding)
            }
            guard let cb = hal.makeUnretainedCommandBuffer(),
                  var enc = cb.makeComputeCommandEncoder() else {
                throw QWEngineDriverError.state("draft chain command encoder")
            }
            hal.touchKeepAlive(enc)
            if let activeDriver {
                activeDriver.stepCBs = []
                activeDriver.activeCB = cb
                activeDriver.activeEnc = enc
            }
            var stepDrained = false
            defer {
                if let activeDriver, let openEnc = activeDriver.activeEnc {
                    activeDriver.activeEnc = nil
                    openEnc.endEncoding()
                }
                if !stepDrained, let activeDriver {
                    for pendingCb in (activeDriver.stepCBs ?? []) {
                        pendingCb.waitUntilCompleted()
                        if pendingCb.status != .completed || pendingCb.error != nil {
                            activeDriver.commandBufferErrors += 1
                        }
                    }
                    _ = drainColdState(driver: activeDriver)
                }
                activeDriver?.stepCBs = nil
                activeDriver?.activeCB = nil
                activeDriver?.activeEnc = nil
            }
            let stepCbs: [any MTLCommandBuffer]
            do {
                if let table = deviceEmbedding {
                    try table.encode(enc, token: draft.scratch.selector.token,
                                     destination: draft.scratch.embedding)
                }
                let prevID = ObjectIdentifier(enc)
                try draft.encode(&enc, embedding: draft.scratch.embedding,
                                 hidden: row == 0 ? hidden : draft.nextStepHidden,
                                 kv: kv, cacheRows: cacheRows + row, absolutePosition: position + row,
                                 mropeDelta: mropeDelta,
                                 driver: activeDriver, cb: activeDriver?.activeCB ?? cb)
                if let active = activeDriver?.activeEnc { enc = active }
                if ObjectIdentifier(enc) != prevID {
                    hal.touchKeepAlive(enc)
                }
                if sampledToken == nil, selection == .gpuChecked {
                    let writeLogits = shouldStopAfterDraft != nil && (row + 1 < drafts || writeLogitsOnFinalDraft)
                    try draft.encodeHeadAndSelect(enc, storage: storage, draftHead: draftHead,
                                                  writeLogits: writeLogits)
                } else {
                    try draft.encodeHead(enc, storage: storage, draftHead: draftHead)
                }
                let finalCB = activeDriver?.activeCB ?? cb
                let finalEnc = activeDriver?.activeEnc ?? enc
                let extraCBs = activeDriver?.stepCBs ?? []
                activeDriver?.activeEnc = nil
                activeDriver?.activeCB = nil
                finalEnc.endEncoding()
                finalCB.commit()
                stepCbs = extraCBs + [finalCB]
            } catch {
                if activeDriver == nil {
                    enc.endEncoding()
                }
                throw error
            }
            for sub in stepCbs { sub.waitUntilCompleted() }
            let coldFailure = drainColdState(driver: activeDriver)
            stepDrained = true
            if let coldFailure {
                gpuFailure()
                throw QWEngineDriverError.state("draft chain cold staging failed: \(coldFailure)")
            }
            for sub in stepCbs {
                guard sub.status == .completed, sub.error == nil else {
                    if let activeDriver { activeDriver.commandBufferErrors += 1 }
                    gpuFailure()
                    throw QWEngineDriverError.state("draft chain GPU failure: \(String(describing: sub.error))")
                }
                try completed?(sub)
            }
            if let sampledToken {
                token = try sampledToken(draft.scratch.draftLogits)
                guard (0..<vocabulary).contains(token) else { throw QWEngineDriverError.badToken(token) }
            } else {
                switch selection {
                case .cpuReference:
                    let scores = draft.scratch.draftLogits.region.pointer
                        .advanced(by: draft.scratch.draftLogits.offset).assumingMemoryBound(to: Float.self)
                    token = try QWVerifyWindow.greedyToken(UnsafeBufferPointer(start: scores, count: vocabulary))
                case .gpuChecked:
                    token = try draft.scratch.selector.completedToken()
                }
            }
            tokens.append(token)
            if let shouldStopAfterDraft, try shouldStopAfterDraft(draft.scratch.draftLogits, tokens.count) {
                break
            }
        }
        return tokens
    }

    static func runStochastic(
        hal: QWHal, draft: QWMTPDraftEncoder, headStorage: QWHeadStorage,
        draftHead: QWMTPTDraftHead? = nil, hidden: QWTensorValue,
        kv: (k: QWKVLayerView, v: QWKVLayerView), pendingToken: Int,
        drafts: Int, position: Int, cacheRows: Int, mropeDelta: Int = 0,
        deviceEmbedding: QWMTPEmbeddingTable, params: QWStochasticSelectionParams,
        completed: ((any MTLCommandBuffer) throws -> Void)? = nil,
        gpuFailure: () -> Void
    ) throws -> (tokens: [Int], proposals: [QWSpeculativeRoundResolver.DraftProposal]) {
        let tokens = try runDeviceAndDrain(
            hal: hal, draft: draft, storage: headStorage, draftHead: draftHead,
            hidden: hidden, kv: kv, pendingToken: pendingToken, drafts: drafts,
            position: position, cacheRows: cacheRows, mropeDelta: mropeDelta,
            table: deviceEmbedding, stochastic: params, completed: completed, gpuFailure: gpuFailure)
        return (tokens, try readSparseProposals(draft: draft, drafts: drafts))
    }

    static func readSparseProposals(
        draft: QWMTPDraftEncoder, drafts: Int
    ) throws -> [QWSpeculativeRoundResolver.DraftProposal] {
        let vocabulary = draft.scratch.draftLogits.shape.elementCount
        var proposals: [QWSpeculativeRoundResolver.DraftProposal] = []
        proposals.reserveCapacity(drafts)
        for row in 0..<drafts {
            let buf = try draft.scratch.selector.readSparseProposal(row: row)
            if buf.isSparse {
                proposals.append(try QWSpeculativeRoundResolver.DraftProposal.fromSparseBuffer(buf))
            } else {
                let rowLogits = try draft.scratch.draftLogitRow(row)
                let ptr = rowLogits.region.pointer.advanced(by: rowLogits.offset).assumingMemoryBound(to: Float.self)
                proposals.append(try QWSpeculativeRoundResolver.DraftProposal.fromSparseBuffer(
                    buf, logits: UnsafeBufferPointer(start: ptr, count: vocabulary)))
            }
        }
        return proposals
    }

    private static func runDeviceAndDrain(
        hal: QWHal, draft: QWMTPDraftEncoder, storage: QWHeadStorage,
        draftHead: QWMTPTDraftHead?, hidden: QWTensorValue,
        kv: (k: QWKVLayerView, v: QWKVLayerView), pendingToken: Int,
        drafts: Int, position: Int, cacheRows: Int, mropeDelta: Int,
        table: QWMTPEmbeddingTable, stochastic: QWStochasticSelectionParams?,
        completed: ((any MTLCommandBuffer) throws -> Void)?,
        gpuFailure: () -> Void
    ) throws -> [Int] {
        let vocabulary = draft.scratch.draftLogits.shape.elementCount
        let activeDriver = draft.driver
        var allDrained = false
        let cbs = try runDeviceChain(
            hal: hal, draft: draft, headStorage: storage, draftHead: draftHead,
            hidden: hidden, kv: kv, pendingToken: pendingToken, drafts: drafts,
            position: position, cacheRows: cacheRows, mropeDelta: mropeDelta,
            table: table, stochastic: stochastic)
        defer {
            if !allDrained {
                for sub in cbs { sub.waitUntilCompleted() }
                drainColdState(driver: activeDriver)
            }
        }
        for sub in cbs { sub.waitUntilCompleted() }
        let coldFailure = drainColdState(driver: activeDriver)
        allDrained = true
        if let coldFailure {
            gpuFailure()
            throw QWEngineDriverError.state("device draft chain cold staging failed: \(coldFailure)")
        }
        for sub in cbs {
            guard sub.status == .completed, sub.error == nil else {
                if let activeDriver { activeDriver.commandBufferErrors += 1 }
                gpuFailure()
                throw QWEngineDriverError.state("device draft chain GPU failure: \(String(describing: sub.error))")
            }
            try completed?(sub)
        }
        let ids = table.tokens.contents().assumingMemoryBound(to: Int32.self)
        var tokens: [Int] = []
        tokens.reserveCapacity(drafts)
        for row in 0..<drafts {
            let id = Int(ids[row])
            guard (0..<vocabulary).contains(id) else {
                throw QWEngineDriverError.state("device draft chain invalid token \(id) at row \(row)")
            }
            tokens.append(id)
        }
        return tokens
    }

    @discardableResult
    private static func drainColdState(driver: QWEngineDriver?) -> (any Error)? {
        guard let driver else { return nil }
        var coldFailure: (any Error)?
        if let pipe = driver.coldEvents {
            pipe.waitUntilIdle()
            let r = pipe.takeStepResult()
            pipe.tuneBridge()
            driver.decodeTel.coldWaitUs += r.waitUs
            driver.decodeTel.coldIOUs += r.ioUs
            driver.decodeTel.coldStaged += r.staged
            coldFailure = r.failure
        }
        driver.flushPendingColdUnloads()
        return coldFailure
    }

    private static func runDeviceChain(
        hal: QWHal, draft: QWMTPDraftEncoder, headStorage: QWHeadStorage,
        draftHead: QWMTPTDraftHead? = nil, hidden: QWTensorValue,
        kv: (k: QWKVLayerView, v: QWKVLayerView), pendingToken: Int,
        drafts: Int, position: Int, cacheRows: Int, mropeDelta: Int = 0,
        table: QWMTPEmbeddingTable, journalTokenOffset: Int = 0,
        stochastic: QWStochasticSelectionParams? = nil
    ) throws -> [any MTLCommandBuffer] {
        let vocabulary = draft.scratch.draftLogits.shape.elementCount
        guard (1...15).contains(drafts), (0..<vocabulary).contains(pendingToken),
              position > 0, position <= draft.positionCapacity - drafts,
              cacheRows >= 0, cacheRows <= QWKVCache.maxTokensPerSeq - drafts,
              journalTokenOffset >= 0, (journalTokenOffset + drafts) * 4 <= table.tokens.length,
              table.vocabulary == vocabulary,
              table.hidden == draft.scratch.embedding.shape.elementCount else {
            throw QWEngineDriverError.state("device draft embedding geometry/selection mismatch")
        }
        let activeDriver = draft.driver
        if draft.hasColdMoE && activeDriver == nil {
            throw QWEngineDriverError.state("MTP draft MoE cold experts require an attached driver")
        }
        let selector = draft.scratch.selector
        selector.token.region.pointer.advanced(by: selector.token.offset)
            .assumingMemoryBound(to: Int32.self).pointee = Int32(pendingToken)
        guard let cb = hal.makeUnretainedCommandBuffer(), var enc = cb.makeComputeCommandEncoder() else {
            throw QWEngineDriverError.state("device draft chain encoder")
        }
        hal.touchKeepAlive(enc)
        var boundID = ObjectIdentifier(enc)
        if let activeDriver {
            activeDriver.stepCBs = []
            activeDriver.activeCB = cb
            activeDriver.activeEnc = enc
        }
        defer {
            activeDriver?.stepCBs = nil
            activeDriver?.activeCB = nil
            activeDriver?.activeEnc = nil
        }
        let journalTensor = QWTensorValue(
            region: QWMRegion(buffer: table.tokens, base: 0, length: table.tokens.length),
            offset: 0, shape: QWTensorShape([table.tokens.length / 4], .i32))
        do {
            for row in 0..<drafts {
                try table.encode(enc, token: selector.token, destination: draft.scratch.embedding)
                try draft.encode(&enc, embedding: draft.scratch.embedding,
                                 hidden: row == 0 ? hidden : draft.nextStepHidden,
                                 kv: kv, cacheRows: cacheRows + row, absolutePosition: position + row,
                                 mropeDelta: mropeDelta,
                                 driver: activeDriver, cb: activeDriver?.activeCB ?? cb)
                if let active = activeDriver?.activeEnc { enc = active }
                if ObjectIdentifier(enc) != boundID {
                    hal.touchKeepAlive(enc)
                    boundID = ObjectIdentifier(enc)
                }
                if var stepParams = stochastic {
                    let rowLogits = try draft.scratch.draftLogitRow(row)
                    try draft.encodeHead(enc, storage: headStorage, draftHead: draftHead, destination: rowLogits)
                    enc.memoryBarrier(scope: .buffers)
                    stepParams.philoxOffsetBase &+= UInt64(row)
                    try selector.encodeStochasticDraft(
                        enc, hal: hal, logits: rowLogits, params: stepParams, rows: 1,
                        proposalSlotBase: row, destination: selector.token,
                        journal: journalTensor, journalOffset: journalTokenOffset + row)
                } else {
                    try draft.encodeHeadAndSelect(enc, storage: headStorage, draftHead: draftHead, writeLogits: false)
                    enc.memoryBarrier(scope: .buffers)
                    let saved = QWTensorValue(
                        region: QWMRegion(buffer: table.tokens, base: 0, length: table.tokens.length),
                        offset: (journalTokenOffset + row) * 4, shape: QWTensorShape([1], .i32))
                    try QWVerifyWindow.encodeCopy(enc, hal: hal, source: selector.token, destination: saved)
                }
                if row + 1 < drafts { enc.memoryBarrier(scope: .buffers) }
            }
            let finalCB = activeDriver?.activeCB ?? cb
            let finalEnc = activeDriver?.activeEnc ?? enc
            let extraCBs = activeDriver?.stepCBs ?? []
            activeDriver?.activeEnc = nil
            activeDriver?.activeCB = nil
            finalEnc.endEncoding()
            finalCB.commit()
            return extraCBs + [finalCB]
        } catch {
            if let activeDriver {
                if let openEnc = activeDriver.activeEnc {
                    activeDriver.activeEnc = nil
                    openEnc.endEncoding()
                }
                for pendingCb in (activeDriver.stepCBs ?? []) {
                    pendingCb.waitUntilCompleted()
                    if pendingCb.status != .completed || pendingCb.error != nil {
                        activeDriver.commandBufferErrors += 1
                    }
                }
                _ = drainColdState(driver: activeDriver)
            } else {
                enc.endEncoding()
            }
            throw error
        }
    }

    static func enqueueDeviceChain(
        hal: QWHal, draft: QWMTPDraftEncoder, headStorage: QWHeadStorage,
        draftHead: QWMTPTDraftHead? = nil, hidden: QWTensorValue,
        kv: (k: QWKVLayerView, v: QWKVLayerView), pendingToken: Int,
        drafts: Int, position: Int, cacheRows: Int, mropeDelta: Int = 0,
        table: QWMTPEmbeddingTable, journalTokenOffset: Int = 0,
        stochastic: QWStochasticSelectionParams? = nil
    ) throws -> [any MTLCommandBuffer] {
        let cbs = try runDeviceChain(
            hal: hal, draft: draft, headStorage: headStorage, draftHead: draftHead,
            hidden: hidden, kv: kv, pendingToken: pendingToken, drafts: drafts,
            position: position, cacheRows: cacheRows, mropeDelta: mropeDelta, table: table,
            journalTokenOffset: journalTokenOffset, stochastic: stochastic)
        guard !cbs.isEmpty else {
            throw QWEngineDriverError.state("device draft chain produced no command buffer")
        }
        return cbs
    }
}
