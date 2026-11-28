import Metal
import QWAlloc
import QWCore
import QWHAL
import QWHead
import QWGEMM
import QWGDN

/// Driver-wide scratch, owned by a single unresolved verification window.
struct QWVerifyWindow {
    struct PendingStochasticVerify {
        let params: QWStochasticSelectionParams
        let draftProposals: QWTokenSelector
        let draftProposalSlotBase: Int
        let repetitionHistoryByRow: [[Int]]
        let draftCount: Int
    }

    final class StochasticVerifyBox: @unchecked Sendable {
        var pending: PendingStochasticVerify?
    }

    let geometry: QWModelGeometry
    let rows: Int
    let vocabulary: Int
    let embeddings: QWTensorValue
    let hidden: QWTensorValue
    let logits: QWTensorValue
    /// Group sums for a batched q4 verification head; allocated only when enabled.
    let gsHead: QWTensorValue?
    let headGroupSums: QWTensorValue
    let normalized: QWTensorValue?
    let mlpRows: QWTensorValue?
    let moeScratch: QWEngineDriver.MoEScratch?
    let moeDn32: QWTensorValue?
    let mhcStream: QWTensorValue?
    let mhcControls: QWTensorValue?
    let targetSelector: QWTokenSelector?
    let targetTokens: QWTensorValue?
    let stochasticVerifyBox = StochasticVerifyBox()

    static func allocationBytes(geometry: QWModelGeometry, rows: Int, vocabulary: Int? = nil,
                                layerMajor: Bool = false, gpuTargets: Bool = false,
                                q4Head: Bool = false, useGroupedNAX: Bool = true) -> Int {
        let vocabulary = vocabulary ?? geometry.vocab
        precondition((2...16).contains(rows) && (1...geometry.vocab).contains(vocabulary))
        func aligned(_ bytes: Int) -> Int { (bytes + 255) / 256 * 256 }
        let base = 2 * aligned(rows * geometry.hidden * 2) + aligned(rows * vocabulary * 4)
        let mhcStreams = geometry.mhcConfig?.streams ?? 2
        let mhcCtrlCount = mhcStreams + mhcStreams * mhcStreams
        let mhcExtra = (layerMajor && geometry.mhcConfig != nil)
            ? aligned(rows * mhcStreams * geometry.hidden * 4) + aligned(rows * mhcCtrlCount * 4)
            : 0
        let moeExtra = (layerMajor && geometry.numExperts > 0)
            ? QWEngineDriver.moeScratchAllocationBytes(rows: rows, geometry: geometry, useGroupedNAX: useGroupedNAX)
                + aligned(rows * geometry.hidden * 4)
            : 0
        let effInter = max(geometry.intermediate, geometry.moeIntermediate)
        return base + (layerMajor ? aligned(rows * geometry.hidden * 2) +
            aligned(rows * effInter * 2) : 0) +
            moeExtra +
            mhcExtra +
            aligned(rows * geometry.hidden / 128 * 4) +
            (gpuTargets ? QWTokenSelector.allocationBytes(vocabLimit: geometry.vocab, vocabulary: vocabulary, maxRows: rows) + 256 : 0) +
            (q4Head ? aligned(rows * geometry.hidden / 128 * 4) : 0)
    }

    init(geometry: QWModelGeometry, allocator: QWSubAllocator, rows: Int, vocabulary: Int? = nil,
         layerMajor: Bool = false, gpuTargets: Bool = false, q4Head: Bool = false,
         useGroupedNAX: Bool = true) throws {
        let vocabulary = vocabulary ?? geometry.vocab
        self.geometry = geometry
        guard (2...16).contains(rows), (1...geometry.vocab).contains(vocabulary) else {
            throw QWEngineDriverError.state("verify window requires B=2...16 and a valid vocabulary")
        }
        self.rows = rows
        self.vocabulary = vocabulary
        let effInter = max(1, max(geometry.intermediate, geometry.moeIntermediate))
        embeddings = try allocator.allocate(QWTensorShape([rows * geometry.hidden], .f16))
        hidden = try allocator.allocate(QWTensorShape([rows * geometry.hidden], .f16))
        logits = try allocator.allocate(QWTensorShape([rows * vocabulary], .f32))
        gsHead = q4Head ? try allocator.allocate(QWTensorShape([rows * geometry.hidden / 128], .f32)) : nil
        headGroupSums = try allocator.allocate(QWTensorShape([rows * geometry.hidden / 128], .f32))
        normalized = layerMajor ? try allocator.allocate(QWTensorShape([rows * geometry.hidden], .f16)) : nil
        mlpRows = layerMajor ? try allocator.allocate(QWTensorShape([rows * effInter], .f16)) : nil
        if layerMajor && geometry.numExperts > 0 {
            moeScratch = try QWEngineDriver.makeMoEScratch(
                { n, dt in try allocator.allocate(QWTensorShape([n], dt)) },
                rows: rows, geometry: geometry, useGroupedNAX: useGroupedNAX)
            moeDn32 = try allocator.allocate(QWTensorShape([rows * geometry.hidden], .f32))
        } else {
            moeScratch = nil
            moeDn32 = nil
        }
        let mhcStreams = geometry.mhcConfig?.streams ?? 2
        let mhcCtrlCount = mhcStreams + mhcStreams * mhcStreams
        mhcStream = (layerMajor && geometry.mhcConfig != nil)
            ? try allocator.allocate(QWTensorShape([rows * mhcStreams * geometry.hidden], .f32))
            : nil
        mhcControls = (layerMajor && geometry.mhcConfig != nil)
            ? try allocator.allocate(QWTensorShape([rows * mhcCtrlCount], .f32))
            : nil
        targetSelector = gpuTargets
            ? try QWTokenSelector(vocabLimit: geometry.vocab, allocator: allocator, vocabulary: vocabulary, maxRows: rows)
            : nil
        targetTokens = gpuTargets ? try allocator.allocate(QWTensorShape([rows], .i32)) : nil
    }

    func hiddenRow(_ row: Int) throws -> QWTensorValue {
        guard (0..<rows).contains(row) else { throw QWEngineDriverError.state("verify hidden row is invalid") }
        return QWTensorValue(region: hidden.region, offset: hidden.offset + row * geometry.hidden * 2,
                             shape: QWTensorShape([geometry.hidden], .f16))
    }

    func logitRow(_ row: Int) throws -> QWTensorValue {
        guard (0..<rows).contains(row) else { throw QWEngineDriverError.state("verify logit row is invalid") }
        return QWTensorValue(region: logits.region, offset: logits.offset + row * vocabulary * 4,
                             shape: QWTensorShape([vocabulary], .f32))
    }

    func embeddingRow(_ row: Int) throws -> QWTensorValue {
        guard (0..<rows).contains(row) else { throw QWEngineDriverError.state("verify embedding row is invalid") }
        return QWTensorValue(region: embeddings.region, offset: embeddings.offset + row * geometry.hidden * 2,
                             shape: QWTensorShape([geometry.hidden], .f16))
    }

    func mhcStreamRow(_ row: Int) throws -> QWTensorValue {
        let streams = geometry.mhcConfig?.streams ?? 2
        let elems = streams * geometry.hidden
        guard let mhcStream, (0..<rows).contains(row) else {
            throw QWEngineDriverError.state("layer-major mHC stream row unavailable")
        }
        return QWTensorValue(region: mhcStream.region, offset: mhcStream.offset + row * elems * 4,
                             shape: QWTensorShape([elems], .f32))
    }

    func encodeHead(_ enc: any MTLComputeCommandEncoder, hal: QWHal, head: QWHead,
                    storage: QWHeadStorage, q4Head: QWMTPTDraftHead? = nil,
                    activeRows: Int? = nil, unrotateForF16Head: Bool = false,
                    multimaxParams: QWTensorValue? = nil,
                    headInput: QWTensorValue? = nil,
                    softcap: Float = 0.0) throws {
        let count = activeRows ?? rows
        guard (2...rows).contains(count) else {
            throw QWEngineDriverError.state("verify head active row count is invalid")
        }
        let srcHidden = headInput ?? hidden
        if let q4Head {
            if count <= 16 {
                guard let gsHead else {
                    throw QWEngineDriverError.state("q4 verify head requires window gsHead scratch")
                }
                let input = QWTensorValue(region: srcHidden.region, offset: srcHidden.offset,
                                          shape: QWTensorShape([count * geometry.hidden], .f16))
                let sums = QWTensorValue(region: gsHead.region, offset: gsHead.offset,
                                         shape: QWTensorShape([count * geometry.hidden / 128], .f32))
                let output = QWTensorValue(region: logits.region, offset: logits.offset,
                                           shape: QWTensorShape([count * vocabulary], .f32))
                try q4Head.encodeBatched(enc, hal: hal, x: input, gs: sums, y: output, m: count)
            } else {
                for row in 0..<count {
                    enc.memoryBarrier(scope: .buffers)
                    let xRow = QWTensorValue(region: srcHidden.region,
                                             offset: srcHidden.offset + row * geometry.hidden * 2,
                                             shape: QWTensorShape([geometry.hidden], .f16))
                    try q4Head.encode(enc, hal: hal, x: xRow,
                                      gs: headGroupSums, y: logitRow(row))
                    enc.memoryBarrier(scope: .buffers)
                }
            }
            if softcap > 0 {
                enc.memoryBarrier(scope: .buffers)
                let allLogits = QWTensorValue(region: logits.region, offset: logits.offset,
                                              shape: QWTensorShape([count * vocabulary], .f32))
                try head.encodeSoftcapF32(enc, logits: allLogits, count: count * vocabulary, softcap: softcap)
            }
        } else {
            switch storage {
            case .f16(let weight):
                let input = QWTensorValue(region: srcHidden.region, offset: srcHidden.offset,
                                          shape: QWTensorShape([count * geometry.hidden], .f16))
                let output = QWTensorValue(region: logits.region, offset: logits.offset,
                                           shape: QWTensorShape([count * vocabulary], .f32))
                if unrotateForF16Head {
                    try QWGemm.encodeFwht32(enc, hal: hal, x: input)
                    enc.memoryBarrier(scope: .buffers)
                }
                try QWHead.encodeF16(enc, hal: hal, w: weight, x: input, y: output,
                                     geometry: geometry, M: count, N: vocabulary, softcap: softcap)
                if unrotateForF16Head {
                    enc.memoryBarrier(scope: .buffers)
                    try QWGemm.encodeFwht32(enc, hal: hal, x: input)
                }
            case .bf16(let weight):
                let input = QWTensorValue(region: srcHidden.region, offset: srcHidden.offset,
                                          shape: QWTensorShape([count * geometry.hidden], .f16))
                let output = QWTensorValue(region: logits.region, offset: logits.offset,
                                           shape: QWTensorShape([count * vocabulary], .f32))
                if unrotateForF16Head {
                    try QWGemm.encodeFwht32(enc, hal: hal, x: input)
                    enc.memoryBarrier(scope: .buffers)
                }
                try QWHead.encodeBF16(enc, hal: hal, w: weight, x: input, y: output,
                                      geometry: geometry, M: count, N: vocabulary, softcap: softcap)
                if unrotateForF16Head {
                    enc.memoryBarrier(scope: .buffers)
                    try QWGemm.encodeFwht32(enc, hal: hal, x: input)
                }
            case .q4g(let weight):
                guard weight.view.n == vocabulary else {
                    throw QWEngineDriverError.state("verify head vocab mismatch")
                }
                let input = QWTensorValue(region: srcHidden.region, offset: srcHidden.offset,
                                          shape: QWTensorShape([count * geometry.hidden], .f16))
                let sums = QWTensorValue(region: headGroupSums.region, offset: headGroupSums.offset,
                                         shape: QWTensorShape([count * geometry.hidden / 128], .f32))
                let output = QWTensorValue(region: logits.region, offset: logits.offset,
                                           shape: QWTensorShape([count * vocabulary], .f32))
                try QWHead.encodeBatched(enc, hal: hal, view: weight.view,
                                         x: input, gs: sums, y: output, m: count, softcap: softcap)
            }
        }
        if let mm = multimaxParams {
            enc.memoryBarrier(scope: .buffers)
            let allLogits = QWTensorValue(region: logits.region, offset: logits.offset,
                                          shape: QWTensorShape([count * vocabulary], .f32))
            try head.encodeMultiMax(enc, logits: allLogits, params: mm, count: count * vocabulary)
            enc.memoryBarrier(scope: .buffers)
        }
        if let selector = targetSelector, let journal = targetTokens {
            let activeLogits = QWTensorValue(region: logits.region, offset: logits.offset,
                shape: QWTensorShape([count * vocabulary], .f32))
            let saved = QWTensorValue(region: journal.region, offset: journal.offset,
                shape: QWTensorShape([count], .i32))
            try selector.encodeGreedyBatched(enc, hal: hal, logits: activeLogits, rows: count,
                                             ordering: .shippingTarget, destination: saved)
            if let stoch = stochasticVerifyBox.pending, count == stoch.draftCount + 1 {
                enc.memoryBarrier(scope: .buffers)
                try selector.encodeStochasticVerifyLC(
                    enc, hal: hal, logits: activeLogits,
                    draftProposals: stoch.draftProposals,
                    draftCount: stoch.draftCount,
                    params: stoch.params,
                    draftProposalSlotBase: stoch.draftProposalSlotBase,
                    repetitionHistoryByRow: stoch.repetitionHistoryByRow
                )
            }
        }
    }

    /// Completed immutable row decisions. The unresolved window lease prevents
    /// scratch reuse until commit or discard, including the target-token journal.
    func completedTargets(activeRows: Int? = nil) throws -> [Int] {
        guard let targetTokens else { throw QWEngineDriverError.state("GPU target journal is disabled") }
        let ids = targetTokens.region.pointer.advanced(by: targetTokens.offset).assumingMemoryBound(to: Int32.self)
        var result: [Int] = []
        let count = activeRows ?? rows
        guard (2...rows).contains(count) else {
            throw QWEngineDriverError.state("GPU target active row count is invalid")
        }
        for row in 0..<count {
            let id = Int(ids[row])
            guard (0..<vocabulary).contains(id) else {
                throw QWEngineDriverError.state("GPU target row \(row) rejected non-finite logits or invalid token \(id)")
            }
            result.append(id)
        }
        return result
    }

    /// Computes both the accepted prefix decision and the unrelaxed target argmax
    /// vector across every active verification row in a single logit pass.
    func greedyDecisionWithTargets(
        drafts: [Int], maxNew: Int, stopIDs: Set<Int> = [],
        activeRows: Int? = nil, relaxedMargin: Float = 0,
        relaxedMinProb: Float = 0
    ) throws -> (decision: QWGreedyRoundDecision, targets: [Int]) {
        let count = activeRows ?? rows
        guard drafts.count + 1 == count else { throw QWEngineDriverError.state("greedy verify draft count mismatch") }
        let p = logits.region.pointer.advanced(by: logits.offset).assumingMemoryBound(to: Float.self)
        var rawTargets: [Int]
        if targetTokens != nil {
            rawTargets = try completedTargets(activeRows: count)
        } else {
            rawTargets = []
            rawTargets.reserveCapacity(count)
            for row in 0..<count {
                rawTargets.append(try Self.greedyToken(
                    UnsafeBufferPointer(start: p + row * vocabulary, count: vocabulary),
                    ordering: .shippingTarget))
            }
        }
        var acceptedTargets = rawTargets
        if relaxedMargin > 0 {
            for row in 0..<drafts.count {
                let draftId = drafts[row]
                let targetId = acceptedTargets[row]
                if draftId != targetId {
                    guard (0..<vocabulary).contains(draftId), (0..<vocabulary).contains(targetId) else { break }
                    let rowLogits = p + row * vocabulary
                    let draftScore = rowLogits[draftId]
                    let targetScore = rowLogits[targetId]
                    if draftScore.isFinite && targetScore.isFinite && draftScore >= targetScore - relaxedMargin,
                       QWSpeculativeTokenResolver.clearsProbabilityFloor(
                           UnsafeBufferPointer(start: rowLogits, count: vocabulary),
                           draftScore: draftScore, maxScore: targetScore,
                           minProbability: relaxedMinProb) {
                        acceptedTargets[row] = draftId
                    } else {
                        break
                    }
                }
            }
        }
        let decision = try QWGreedyRoundDecision.make(
            geometry: geometry, drafts: drafts, targetTokens: acceptedTargets,
            maxNew: maxNew, stopIDs: stopIDs)
        return (decision: decision, targets: rawTargets)
    }

    enum GreedyOrdering {
        case numericalDraft
        case shippingTarget
    }

    static func greedyToken(_ scores: UnsafeBufferPointer<Float>,
                            ordering: GreedyOrdering = .numericalDraft) throws -> Int {
        guard !scores.isEmpty else { throw QWEngineDriverError.state("greedy logits are empty") }
        var bestKey: UInt32 = 0, token = 0
        for (id, score) in scores.enumerated() {
            guard score.isFinite else { throw QWEngineDriverError.state("non-finite greedy logits") }
            let bits = ordering == .numericalDraft && score == 0 ? UInt32(0) : score.bitPattern
            let key = bits & 0x80000000 != 0 ? ~bits : bits ^ 0x80000000
            if key > bestKey { bestKey = key; token = id }
        }
        return token
    }

    /// Resolve GPU trunk state only. KV/MTP/metadata publication belongs to the
    /// round owner after successful completion of this command buffer.
    func encodeRetainedPrefix(_ cb: any MTLCommandBuffer, hal: QWHal,
                              checkpoint: QWGDNReplayCheckpoint, decoder: QWGDNDecode,
                              rec: QWTensorValue, ring: QWTensorValue,
                              trunkHidden: QWTensorValue, activeRows: Int? = nil,
                              retaining count: Int) throws {
        let verifiedRows = activeRows ?? rows
        guard (2...rows).contains(verifiedRows), (0...verifiedRows).contains(count),
              checkpoint.rows == rows else {
            throw QWEngineDriverError.state("verify retained prefix is invalid")
        }
        try checkpoint.validateResolution(rec: rec, ring: ring, activeRows: verifiedRows)
        let carryFromStream = trunkHidden.shape.dtype == .f32
        let carrySource: QWTensorValue? = count > 0
            ? (carryFromStream ? try mhcStreamRow(count - 1) : try hiddenRow(count - 1))
            : nil
        if let carrySource {
            let carryStorage = carryFromStream ? mhcStream! : hidden
            try Self.validateCopy(source: carrySource, destination: trunkHidden)
            if carryStorage.region.buffer === trunkHidden.region.buffer {
                let a = carryStorage.region.base + carryStorage.offset
                let b = trunkHidden.region.base + trunkHidden.offset
                guard a + carryStorage.bytes <= b || b + trunkHidden.bytes <= a else {
                    throw QWEngineDriverError.state("retained hidden carry overlaps verification rows")
                }
            }
            _ = try hal.pipeline("qwg_verify_copy")
        }
        if checkpoint.layers == 0 {
            if count < verifiedRows {
                try checkpoint.validateRetained(verifiedRows)
            }
            if let carrySource {
                guard let enc = cb.makeComputeCommandEncoder() else {
                    throw QWEngineDriverError.state("retained hidden encoder")
                }
                hal.touchKeepAlive(enc)
                defer { enc.endEncoding() }
                try Self.encodeCopy(enc, hal: hal, source: carrySource, destination: trunkHidden)
            }
            return
        }
        // Even all-accept can be shortened by stops, callbacks, or maxNew.
        if count < verifiedRows {
            if try checkpoint.rollbackTapeIsStable(retaining: count, activeRows: verifiedRows) {
                let skipEmptyRollback = checkpoint.outOfPlaceVerifyActive && count == 0 && carrySource == nil
                if !skipEmptyRollback {
                    guard let enc = cb.makeComputeCommandEncoder() else {
                        throw QWEngineDriverError.state("retained rollback encoder")
                    }
                    hal.touchKeepAlive(enc)
                    do {
                        try checkpoint.encodeRollbackTape(enc, decoder: decoder, rec: rec,
                                                          activeRows: verifiedRows, retaining: count,
                                                          verifyStability: false)
                        if let carrySource {
                            try Self.encodeCopy(enc, hal: hal, source: carrySource, destination: trunkHidden)
                        }
                    } catch {
                        enc.endEncoding()
                        throw error
                    }
                    enc.endEncoding()
                }
                guard let rings = cb.makeBlitCommandEncoder() else {
                    throw QWEngineDriverError.state("retained ring encoder")
                }
                do {
                    try checkpoint.encodeReplayRings(rings, ring: ring, retaining: count)
                } catch {
                    rings.endEncoding()
                    throw error
                }
                rings.endEncoding()
            } else {
                guard let blit = cb.makeBlitCommandEncoder() else {
                    throw QWEngineDriverError.state("retained restore encoder")
                }
                do {
                    try checkpoint.encodeRestore(blit, rec: rec, ring: ring, restoreRing: false)
                    try checkpoint.encodeReplayRings(blit, ring: ring, retaining: count)
                } catch {
                    blit.endEncoding()
                    throw error
                }
                blit.endEncoding()
                if count > 0 || carrySource != nil {
                    guard let enc = cb.makeComputeCommandEncoder() else {
                        throw QWEngineDriverError.state("retained replay encoder")
                    }
                    hal.touchKeepAlive(enc)
                    do {
                        if count > 0 {
                            try checkpoint.encodeReplay(enc, decoder: decoder, rec: rec, retaining: count)
                        }
                        if let carrySource {
                            try Self.encodeCopy(enc, hal: hal, source: carrySource, destination: trunkHidden)
                        }
                    } catch {
                        enc.endEncoding()
                        throw error
                    }
                    enc.endEncoding()
                }
            }
        } else {
            if checkpoint.outOfPlaceVerifyActive {
                guard let commitBlit = cb.makeBlitCommandEncoder() else {
                    throw QWEngineDriverError.state("retained full rec commit encoder")
                }
                do {
                    try checkpoint.encodeCommitFullRec(commitBlit, rec: rec)
                } catch {
                    commitBlit.endEncoding()
                    throw error
                }
                commitBlit.endEncoding()
            }
            if let carrySource {
                guard let enc = cb.makeComputeCommandEncoder() else { throw QWEngineDriverError.state("retained hidden encoder") }
                hal.touchKeepAlive(enc)
                defer { enc.endEncoding() }
                try Self.encodeCopy(enc, hal: hal, source: carrySource, destination: trunkHidden)
            }
        }
    }

    /// Ordered reference backend: one command buffer, causal B=1 trunk rows,
    /// then a shared-weight B-row head. It deliberately does not use F5.
    func encode(_ cb: any MTLCommandBuffer, hal: QWHal, head: QWHead,
                checkpoint: QWGDNReplayCheckpoint,
                rec: QWTensorValue, ring: QWTensorValue, headStorage: QWHeadStorage,
                q4Head: QWMTPTDraftHead? = nil,
                activeRows: Int? = nil,
                unrotateForF16Head: Bool = false,
                multimaxParams: QWTensorValue? = nil,
                headInput: QWTensorValue? = nil,
                softcap: Float = 0.0,
                outOfPlaceGDN: Bool = false,
                windowEncoder: (() throws -> Void)? = nil,
                currentCommandBuffer: (() -> (any MTLCommandBuffer))? = nil,
                stageProfiler: QWGPUStageProfiler<QWMTPVerificationStage>? = nil,
                rowEncoder: (Int) throws -> Void) throws {
        let count = activeRows ?? rows
        guard checkpoint.rows == rows, (2...rows).contains(count) else {
            throw QWEngineDriverError.state("verify/checkpoint row mismatch")
        }
        if checkpoint.layers == 0 {
            try checkpoint.markZeroLayerBackup(rec: rec, ring: ring)
        } else {
            guard let blit = cb.makeBlitCommandEncoder() else { throw QWEngineDriverError.state("verify backup encoder") }
            do {
                try checkpoint.encodeBackup(blit, rec: rec, ring: ring,
                                            skipRecCopy: outOfPlaceGDN && windowEncoder != nil)
            } catch {
                blit.endEncoding()
                throw error
            }
            blit.endEncoding()
        }
        if let windowEncoder { try windowEncoder() }
        else { for row in 0..<count { try rowEncoder(row) } }
        try checkpoint.validateCompleteCapture(activeRows: count)
        let headCB = currentCommandBuffer?() ?? cb
        let enc: any MTLComputeCommandEncoder
        if let stageProfiler {
            enc = try stageProfiler.makeEncoder(commandBuffer: headCB, stage: .head)
        } else {
            guard let encoder = headCB.makeComputeCommandEncoder() else {
                throw QWEngineDriverError.state("verify head encoder")
            }
            enc = encoder
        }
        hal.touchKeepAlive(enc)
        defer { enc.endEncoding() }
        try encodeHead(enc, hal: hal, head: head, storage: headStorage,
                       q4Head: q4Head, activeRows: count,
                       unrotateForF16Head: unrotateForF16Head,
                       multimaxParams: multimaxParams,
                       headInput: headInput, softcap: softcap)
    }

    static func encodeCopy(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                           source: QWTensorValue, destination: QWTensorValue) throws {
        try validateCopy(source: source, destination: destination)
        enc.setComputePipelineState(try hal.pipeline("qwg_verify_copy"))
        source.bind(enc, index: 0)
        destination.bind(enc, index: 1)
        var count = UInt32(source.bytes / 4)
        enc.setBytes(&count, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private static func validateCopy(source: QWTensorValue, destination: QWTensorValue) throws {
        func valid(_ t: QWTensorValue) -> Bool {
            t.offset >= 0 && t.region.base >= 0 && t.bytes > 0 && t.bytes % 4 == 0
                && t.offset <= t.region.length && t.bytes <= t.region.length - t.offset
                && t.region.base <= t.region.buffer.length
                && t.region.length <= t.region.buffer.length - t.region.base
                && (t.region.base + t.offset) % 4 == 0
        }
        guard valid(source), valid(destination), source.bytes / 4 <= Int(UInt32.max),
              source.bytes == destination.bytes,
              source.shape.dtype == destination.shape.dtype else {
            throw QWEngineDriverError.state("verify copy geometry, alignment, or extent is invalid")
        }
        if source.region.buffer === destination.region.buffer {
            let a = source.region.base + source.offset, b = destination.region.base + destination.offset
            guard a + source.bytes <= b || b + destination.bytes <= a else {
                throw QWEngineDriverError.state("verify copy source/destination overlap")
            }
        }
    }
}
