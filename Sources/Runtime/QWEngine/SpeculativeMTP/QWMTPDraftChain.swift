import Metal
import QWAlloc
import QWCore
import QWHAL
import QWHead
import QWKV

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
            let activeDriver = draft.driver
            var allDrained = false
            let cbs = try runDeviceChain(
                hal: hal, draft: draft, headStorage: storage, draftHead: draftHead,
                hidden: hidden, kv: kv, pendingToken: pendingToken, drafts: drafts,
                position: position, cacheRows: cacheRows, mropeDelta: mropeDelta, table: table)
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
            for row in 0..<drafts {
                let id = Int(ids[row])
                guard (0..<vocabulary).contains(id) else {
                    throw QWEngineDriverError.state("device draft chain invalid token \(id) at row \(row)")
                }
                tokens.append(id)
            }
            return tokens
        }
        let useDeviceEmbedding = deviceEmbedding != nil
        if let table = deviceEmbedding {
            guard sampledToken == nil, selection == .gpuChecked,
                  table.vocabulary == vocabulary,
                  table.hidden == draft.scratch.embedding.shape.elementCount else {
                throw QWEngineDriverError.state("device draft embedding geometry/selection mismatch")
            }
            let selector = draft.scratch.greedySelector
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
                    try table.encode(enc, token: draft.scratch.greedySelector.token,
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
                    token = try draft.scratch.greedySelector.completedToken()
                }
            }
            tokens.append(token)
            if let shouldStopAfterDraft, try shouldStopAfterDraft(draft.scratch.draftLogits, tokens.count) {
                break
            }
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
        table: QWMTPEmbeddingTable, journalTokenOffset: Int = 0
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
        let selector = draft.scratch.greedySelector
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
                try draft.encodeHeadAndSelect(enc, storage: headStorage, draftHead: draftHead, writeLogits: false)
                enc.memoryBarrier(scope: .buffers)
                let saved = QWTensorValue(region: QWMRegion(buffer: table.tokens, base: 0, length: table.tokens.length),
                                          offset: (journalTokenOffset + row) * 4, shape: QWTensorShape([1], .i32))
                try QWVerifyWindow.encodeCopy(enc, hal: hal, source: selector.token, destination: saved)
                enc.memoryBarrier(scope: .buffers)
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
        table: QWMTPEmbeddingTable, journalTokenOffset: Int = 0
    ) throws -> [any MTLCommandBuffer] {
        let cbs = try runDeviceChain(
            hal: hal, draft: draft, headStorage: headStorage, draftHead: draftHead,
            hidden: hidden, kv: kv, pendingToken: pendingToken, drafts: drafts,
            position: position, cacheRows: cacheRows, mropeDelta: mropeDelta, table: table,
            journalTokenOffset: journalTokenOffset)
        guard !cbs.isEmpty else {
            throw QWEngineDriverError.state("device draft chain produced no command buffer")
        }
        return cbs
    }
}
