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
        guard (1...5).contains(drafts), (0..<vocabulary).contains(pendingToken),
              position > 0, position <= draft.positionCapacity - drafts,
              cacheRows >= 0, cacheRows <= QWKVCache.maxTokensPerSeq - drafts else {
            throw QWEngineDriverError.state("draft chain token, count, or boundary is invalid")
        }
        var tokens: [Int] = []
        if let table = deviceEmbedding, shouldStopAfterDraft == nil {
            guard sampledToken == nil,
                  selection == .gpuChecked, table.vocabulary == vocabulary,
                  table.hidden == draft.scratch.embedding.shape.elementCount else {
                throw QWEngineDriverError.state("device draft embedding geometry/selection mismatch")
            }
            let selector = draft.scratch.greedySelector
            selector.token.region.pointer.advanced(by: selector.token.offset)
                .assumingMemoryBound(to: Int32.self).pointee = Int32(pendingToken)
            guard let cb = hal.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
                throw QWEngineDriverError.state("device draft chain encoder")
            }
            do {
                for row in 0..<drafts {
                    try table.encode(enc, token: selector.token, destination: draft.scratch.embedding)
                    try draft.encode(enc, embedding: draft.scratch.embedding,
                        hidden: row == 0 ? hidden : draft.scratch.postNorm,
                        kv: kv, cacheRows: cacheRows + row, absolutePosition: position + row)
                    try draft.encodeHeadAndSelect(enc, storage: storage, draftHead: draftHead, writeLogits: false)
                    enc.memoryBarrier(scope: .buffers)
                    let saved = QWTensorValue(region: QWMRegion(buffer: table.tokens, base: 0, length: 256),
                        offset: row * 4, shape: QWTensorShape([1], .i32))
                    try QWVerifyWindow.encodeCopy(enc, hal: hal, source: selector.token, destination: saved)
                    enc.memoryBarrier(scope: .buffers)
                }
            } catch {
                enc.endEncoding()
                throw error
            }
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            guard cb.status == .completed else {
                gpuFailure()
                throw QWEngineDriverError.state("device draft chain GPU failure: \(String(describing: cb.error))")
            }
            try completed?(cb)
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
        var token = pendingToken
        for row in 0..<drafts {
            if !useDeviceEmbedding {
                try embedding(token, draft.scratch.embedding)
            }
            guard let cb = hal.queue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else {
                throw QWEngineDriverError.state("draft chain command encoder")
            }
            do {
                if let table = deviceEmbedding {
                    try table.encode(enc, token: draft.scratch.greedySelector.token,
                                     destination: draft.scratch.embedding)
                }
                try draft.encode(enc, embedding: draft.scratch.embedding,
                                 hidden: row == 0 ? hidden : draft.scratch.postNorm,
                                 kv: kv, cacheRows: cacheRows + row, absolutePosition: position + row)
                if sampledToken == nil, selection == .gpuChecked {
                    let writeLogits = shouldStopAfterDraft != nil && (row + 1 < drafts || writeLogitsOnFinalDraft)
                    try draft.encodeHeadAndSelect(enc, storage: storage, draftHead: draftHead,
                                                  writeLogits: writeLogits)
                } else {
                    try draft.encodeHead(enc, storage: storage, draftHead: draftHead)
                }
            } catch {
                enc.endEncoding()
                throw error
            }
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            guard cb.status == .completed else {
                gpuFailure()
                throw QWEngineDriverError.state("draft chain GPU failure: \(String(describing: cb.error))")
            }
            try completed?(cb)
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
}
