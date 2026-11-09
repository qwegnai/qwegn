import Metal
import QWAlloc
import QWCore
import QWHAL
import QWHead
import QWGEMM
import QWGDN
import QWMTP

/// Driver-wide scratch, owned by a single unresolved verification window.
struct QWVerifyWindow {
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
    let targetSelector: QWMTPGreedySelector?
    let targetTokens: QWTensorValue?

    static func allocationBytes(geometry: QWModelGeometry, rows: Int, vocabulary: Int? = nil,
                                layerMajor: Bool = false, gpuTargets: Bool = false,
                                q4Head: Bool = false) -> Int {
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
            ? QWEngineDriver.moeScratchAllocationBytes(rows: rows, geometry: geometry) + aligned(rows * geometry.hidden * 4)
            : 0
        let effInter = max(geometry.intermediate, geometry.moeIntermediate)
        return base + (layerMajor ? aligned(rows * geometry.hidden * 2) +
            aligned(rows * effInter * 2) : 0) +
            moeExtra +
            mhcExtra +
            aligned(geometry.hidden / 128 * 4) +
            (gpuTargets ? QWMTPGreedySelector.allocationBytes(geometry: geometry, vocabulary: vocabulary) + 256 : 0) +
            (q4Head ? aligned(rows * geometry.hidden / 128 * 4) : 0)
    }

    init(geometry: QWModelGeometry, allocator: QWSubAllocator, rows: Int, vocabulary: Int? = nil,
         layerMajor: Bool = false, gpuTargets: Bool = false, q4Head: Bool = false) throws {
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
        headGroupSums = try allocator.allocate(QWTensorShape([geometry.hidden / 128], .f32))
        normalized = layerMajor ? try allocator.allocate(QWTensorShape([rows * geometry.hidden], .f16)) : nil
        mlpRows = layerMajor ? try allocator.allocate(QWTensorShape([rows * effInter], .f16)) : nil
        if layerMajor && geometry.numExperts > 0 {
            moeScratch = try QWEngineDriver.makeMoEScratch(
                { n, dt in try allocator.allocate(QWTensorShape([n], dt)) },
                rows: rows, geometry: geometry)
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
        targetSelector = gpuTargets ? try QWMTPGreedySelector(geometry: geometry, allocator: allocator, vocabulary: vocabulary) : nil
        targetTokens = gpuTargets ? try allocator.allocate(QWTensorShape([rows], .i32)) : nil
    }

    func normalizedRow(_ row: Int) throws -> QWTensorValue {
        guard let normalized, (0..<rows).contains(row) else {
            throw QWEngineDriverError.state("layer-major normalized row unavailable")
        }
        return QWTensorValue(region: normalized.region, offset: normalized.offset + row * geometry.hidden * 2,
                             shape: QWTensorShape([geometry.hidden], .f16))
    }

    func mlpRow(_ row: Int) throws -> QWTensorValue {
        guard let mlpRows, (0..<rows).contains(row) else {
            throw QWEngineDriverError.state("layer-major MLP row unavailable")
        }
        let effInter = max(1, max(geometry.intermediate, geometry.moeIntermediate))
        return QWTensorValue(region: mlpRows.region, offset: mlpRows.offset + row * effInter * 2,
                             shape: QWTensorShape([effInter], .f16))
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

    func encodeHead(_ enc: any MTLComputeCommandEncoder, hal: QWHal, head: QWHead,
                    storage: QWHeadStorage, q4Head: QWMTPTDraftHead? = nil,
                    activeRows: Int? = nil, unrotateForF16Head: Bool = false) throws {
        let count = activeRows ?? rows
        guard (2...rows).contains(count) else {
            throw QWEngineDriverError.state("verify head active row count is invalid")
        }
        if let q4Head {
            // EXP-12 opt-in (QW_MTP_VERIFY_HEAD_Q4=1): the batched q4 head —
            // one 606 MiB read feeds all rows (probe: 2.44× f16 at M=2).
            // LOSSY: the operator accepted verify-verdict drift; golden
            // parity is voided on this path by design. hidden is [rows·K]
            // contiguous f16 — the FLAT 128-groups are exactly the per-row
            // groups (batched-MLP group-sum law), so ONE launch covers all
            // rows.
            if count <= 16 {
                guard let gsHead else {
                    throw QWEngineDriverError.state("q4 verify head requires window gsHead scratch")
                }
                let input = QWTensorValue(region: hidden.region, offset: hidden.offset,
                                          shape: QWTensorShape([count * geometry.hidden], .f16))
                let sums = QWTensorValue(region: gsHead.region, offset: gsHead.offset,
                                         shape: QWTensorShape([count * geometry.hidden / 128], .f32))
                let output = QWTensorValue(region: logits.region, offset: logits.offset,
                                           shape: QWTensorShape([count * vocabulary], .f32))
                try q4Head.encodeBatched(enc, hal: hal, x: input, gs: sums, y: output, m: count)
            } else {
                for row in 0..<count {
                    enc.memoryBarrier(scope: .buffers)
                    try q4Head.encode(enc, hal: hal, x: hiddenRow(row),
                                      gs: headGroupSums, y: logitRow(row))
                    enc.memoryBarrier(scope: .buffers)
                }
            }
        } else {
            switch storage {
            case .f16(let weight):
                let input = QWTensorValue(region: hidden.region, offset: hidden.offset,
                                          shape: QWTensorShape([count * geometry.hidden], .f16))
                let output = QWTensorValue(region: logits.region, offset: logits.offset,
                                           shape: QWTensorShape([count * vocabulary], .f32))
                if unrotateForF16Head {
                    try QWGemm.encodeFwht32(enc, hal: hal, x: input)
                    enc.memoryBarrier(scope: .buffers)
                }
                try QWHead.encodeF16(enc, hal: hal, w: weight, x: input, y: output,
                                     geometry: geometry, M: count, N: vocabulary)
                if unrotateForF16Head {
                    enc.memoryBarrier(scope: .buffers)
                    try QWGemm.encodeFwht32(enc, hal: hal, x: input)
                }
            case .q4g(let weight):
                // Phase 3 Step 3.0: ONE batched plane pass feeds all rows.
                // The per-row GEMV loop this replaces re-read the full 606
                // MiB per row (~1.8 GB at rows=3, ~3.6 GB at rows=6); the
                // batched kernel is BIT-EXACT row-for-row vs that loop
                // (same pinned fold tree + lane law + shuffle), so this is
                // a pure speed change — zero token drift on the q4g pack
                // path, no new lossiness beyond what the repack chose.
                guard weight.view.n == vocabulary else {
                    throw QWEngineDriverError.state("verify head vocab mismatch")
                }
                let input = QWTensorValue(region: hidden.region, offset: hidden.offset,
                                          shape: QWTensorShape([count * geometry.hidden], .f16))
                let sums = QWTensorValue(region: headGroupSums.region, offset: headGroupSums.offset,
                                         shape: QWTensorShape([count * geometry.hidden / 128], .f32))
                let output = QWTensorValue(region: logits.region, offset: logits.offset,
                                           shape: QWTensorShape([count * vocabulary], .f32))
                try QWHead.encodeBatched(enc, hal: hal, view: weight.view,
                                         x: input, gs: sums, y: output, m: count)
            }
        }
        if let selector = targetSelector, let journal = targetTokens {
            for row in 0..<count {
                let scores = QWTensorValue(region: logits.region, offset: logits.offset + row * vocabulary * 4,
                    shape: QWTensorShape([vocabulary], .f32))
                let saved = QWTensorValue(region: journal.region, offset: journal.offset + row * 4,
                    shape: QWTensorShape([1], .i32))
                try selector.encode(enc, hal: hal, logits: scores, ordering: .shippingTarget, destination: saved)
            }
        }
    }

    /// Encodes shared-noise draft/target argmax pairs for each draft row and
    /// the target-only bonus row into `[draft, target]` i32 pairs.
    func encodeGumbelComparisons(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                 draftLogits: [QWTensorValue], seed: UInt64,
                                 startingPosition: UInt64, temperature: Float,
                                 winners: QWTensorValue) throws {
        guard draftLogits.count + 1 == rows, temperature.isFinite, temperature > 0,
              winners.shape.dtype == .i32, winners.shape.dims == [2 * rows],
              winners.bytes == 2 * rows * MemoryLayout<Int32>.size else {
            throw QWEngineDriverError.state("Gumbel verification shape or temperature is invalid")
        }
        func valid(_ value: QWTensorValue, dtype: QWDType, elements: Int) -> Bool {
            value.shape.dtype == dtype && value.shape.dims == [elements]
                && value.bytes == elements * (dtype == .f32 || dtype == .i32 ? 4 : 2)
                && value.offset >= 0 && value.region.base >= 0
                && value.offset <= value.region.length
                && value.bytes <= value.region.length - value.offset
                && value.region.base <= value.region.buffer.length
                && value.region.length <= value.region.buffer.length - value.region.base
                && (value.region.base + value.offset) % 4 == 0
        }
        guard valid(winners, dtype: .i32, elements: 2 * rows),
              draftLogits.allSatisfy({ valid($0, dtype: .f32, elements: vocabulary) }) else {
            throw QWEngineDriverError.state("Gumbel verification tensor extent or alignment is invalid")
        }
        var inputs = draftLogits
        inputs.append(contentsOf: try (0..<rows).map { try logitRow($0) })
        for input in inputs {
            if input.region.buffer === winners.region.buffer {
                let inputStart = input.region.base + input.offset
                let winnerStart = winners.region.base + winners.offset
                guard inputStart + input.bytes <= winnerStart ||
                      winnerStart + winners.bytes <= inputStart else {
                    throw QWEngineDriverError.state("Gumbel winner buffer aliases verification logits")
                }
            }
        }
        let (lastPosition, rowOverflow) = startingPosition.addingReportingOverflow(UInt64(rows - 1))
        let maxCounter = (UInt64.max - UInt64(vocabulary - 1)) / UInt64(vocabulary)
        guard !rowOverflow, lastPosition <= maxCounter else {
            throw QWEngineDriverError.state("Gumbel verification Philox counter overflows")
        }
        let pso = try hal.pipeline("qwg_gumbel_sample_argmax")
        guard pso.maxTotalThreadsPerThreadgroup >= 256 else {
            throw QWEngineDriverError.state("Gumbel pipeline requires 256 threads per threadgroup")
        }
        for row in 0..<rows {
            let target = try logitRow(row)
            let draft = row < draftLogits.count ? draftLogits[row] : target
            let result = QWTensorValue(region: winners.region,
                offset: winners.offset + row * 2 * MemoryLayout<Int32>.size,
                shape: QWTensorShape([2], .i32))
            guard valid(target, dtype: .f32, elements: vocabulary),
                  valid(result, dtype: .i32, elements: 2) else {
                throw QWEngineDriverError.state("Gumbel target row or output tensor extent is invalid")
            }
            enc.setComputePipelineState(pso)
            draft.bind(enc, index: 0)
            target.bind(enc, index: 1)
            result.bind(enc, index: 2)
            var vocab = UInt32(vocabulary), rowPosition = startingPosition + UInt64(row)
            var seedValue = seed, temp = temperature
            enc.setBytes(&vocab, length: MemoryLayout<UInt32>.size, index: 3)
            enc.setBytes(&seedValue, length: MemoryLayout<UInt64>.size, index: 4)
            enc.setBytes(&rowPosition, length: MemoryLayout<UInt64>.size, index: 5)
            enc.setBytes(&temp, length: MemoryLayout<Float>.size, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
        }
    }

    /// Publishes the completed GPU winner pairs through QWMTP's checked
    /// coupling resolver. Call only after the producing command buffer ends.
    func completedGumbelDecision(winners: QWTensorValue) throws -> QWMTP.GumbelRoundResult {
        guard winners.shape.dtype == .i32, winners.shape.dims == [2 * rows],
              winners.offset >= 0, winners.region.base >= 0,
              winners.bytes == 2 * rows * MemoryLayout<Int32>.size,
              winners.offset <= winners.region.length,
              winners.bytes <= winners.region.length - winners.offset,
              winners.region.base <= winners.region.buffer.length,
              winners.region.length <= winners.region.buffer.length - winners.region.base,
              (winners.region.base + winners.offset) % MemoryLayout<Int32>.alignment == 0 else {
            throw QWEngineDriverError.state("completed Gumbel winner buffer is invalid")
        }
        let ids = winners.region.pointer.advanced(by: winners.offset).assumingMemoryBound(to: Int32.self)
        var pairs: [[Int]] = []
        pairs.reserveCapacity(rows - 1)
        for row in 0..<(rows - 1) {
            pairs.append([Int(ids[row * 2]), Int(ids[row * 2 + 1])])
        }
        return try QWMTP.checkedGumbelCouplingRound(winnerPairs: pairs,
            bonusTarget: Int(ids[(rows - 1) * 2 + 1]), vocabulary: vocabulary)
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

    /// Completed target logits use shipping bit-key order (+0 above -0),
    /// with lowest-ID ties. Never call before command-buffer completion.
    func greedyDecision(drafts: [Int], maxNew: Int, stopIDs: Set<Int> = [],
                        activeRows: Int? = nil, relaxedMargin: Float = 0,
                        relaxedMinProb: Float = 0) throws -> QWGreedyRoundDecision {
        let count = activeRows ?? rows
        guard drafts.count + 1 == count else { throw QWEngineDriverError.state("greedy verify draft count mismatch") }
        let p = logits.region.pointer.advanced(by: logits.offset).assumingMemoryBound(to: Float.self)
        var targets: [Int]
        if targetTokens != nil {
            targets = try completedTargets(activeRows: count)
        } else {
            targets = []
            targets.reserveCapacity(count)
            for row in 0..<count {
                targets.append(try Self.greedyToken(UnsafeBufferPointer(start: p + row * vocabulary, count: vocabulary),
                                                   ordering: .shippingTarget))
            }
        }
        if relaxedMargin > 0 {
            for row in 0..<drafts.count {
                let draftId = drafts[row]
                let targetId = targets[row]
                if draftId != targetId {
                    guard (0..<vocabulary).contains(draftId), (0..<vocabulary).contains(targetId) else { break }
                    let rowLogits = p + row * vocabulary
                    let draftScore = rowLogits[draftId]
                    let targetScore = rowLogits[targetId]
                    if draftScore.isFinite && targetScore.isFinite && draftScore >= targetScore - relaxedMargin,
                       Self.clearsProbabilityFloor(UnsafeBufferPointer(start: rowLogits, count: vocabulary),
                                                   draftScore: draftScore, maxScore: targetScore,
                                                   minProbability: relaxedMinProb) {
                        targets[row] = draftId
                    } else {
                        break
                    }
                }
            }
        }
        return try QWGreedyRoundDecision.make(geometry: geometry, drafts: drafts, targetTokens: targets, maxNew: maxNew, stopIDs: stopIDs)
    }

    /// Variational fidelity floor: a relaxed draft also needs
    /// `softmax(logits)[draft] >= minProbability`, so a flat row cannot
    /// wave through a low-probability token that merely sits within the margin.
    static func clearsProbabilityFloor(_ logits: UnsafeBufferPointer<Float>, draftScore: Float,
                                       maxScore: Float, minProbability: Float) -> Bool {
        guard minProbability > 0 else { return true }
        guard minProbability < 1 else { return false }
        var sum: Double = 0
        for score in logits { sum += exp(Double(score - maxScore)) }
        let logProb = Double(draftScore - maxScore) - log(sum)
        return logProb >= log(Double(minProbability))
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
        if count > 0 {
            try Self.validateCopy(source: hiddenRow(count - 1), destination: trunkHidden)
            if hidden.region.buffer === trunkHidden.region.buffer {
                let a = hidden.region.base + hidden.offset, b = trunkHidden.region.base + trunkHidden.offset
                guard a + hidden.bytes <= b || b + trunkHidden.bytes <= a else {
                    throw QWEngineDriverError.state("retained hidden carry overlaps verification rows")
                }
            }
            _ = try hal.pipeline("qwg_verify_copy")
        }
        // Even all-accept can be shortened by stops, callbacks, or maxNew.
        if count < verifiedRows {
            if try checkpoint.rollbackTapeIsStable(retaining: count, activeRows: verifiedRows) {
                guard let enc = cb.makeComputeCommandEncoder() else {
                    throw QWEngineDriverError.state("retained rollback encoder")
                }
                do {
                    try checkpoint.encodeRollbackTape(enc, decoder: decoder, rec: rec,
                                                      activeRows: verifiedRows, retaining: count)
                } catch {
                    enc.endEncoding()
                    throw error
                }
                enc.endEncoding()
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
                    try checkpoint.encodeRestore(blit, rec: rec, ring: ring)
                } catch {
                    blit.endEncoding()
                    throw error
                }
                blit.endEncoding()
                guard let enc = cb.makeComputeCommandEncoder() else {
                    throw QWEngineDriverError.state("retained replay encoder")
                }
                do {
                    try checkpoint.encodeReplay(enc, decoder: decoder, rec: rec, retaining: count)
                } catch {
                    enc.endEncoding()
                    throw error
                }
                enc.endEncoding()
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
            }
        }
        if count > 0 {
            guard let enc = cb.makeComputeCommandEncoder() else { throw QWEngineDriverError.state("retained hidden encoder") }
            defer { enc.endEncoding() }
            try Self.encodeCopy(enc, hal: hal, source: hiddenRow(count - 1), destination: trunkHidden)
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
                windowEncoder: (() throws -> Void)? = nil,
                currentCommandBuffer: (() -> (any MTLCommandBuffer))? = nil,
                stageProfiler: QWGPUStageProfiler<QWMTPVerificationStage>? = nil,
                rowEncoder: (Int) throws -> Void) throws {
        let count = activeRows ?? rows
        guard checkpoint.rows == rows, (2...rows).contains(count) else {
            throw QWEngineDriverError.state("verify/checkpoint row mismatch")
        }
        guard let blit = cb.makeBlitCommandEncoder() else { throw QWEngineDriverError.state("verify backup encoder") }
        do {
            try checkpoint.encodeBackup(blit, rec: rec, ring: ring)
        } catch {
            blit.endEncoding()
            throw error
        }
        blit.endEncoding()
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
        defer { enc.endEncoding() }
        try encodeHead(enc, hal: hal, head: head, storage: headStorage,
                       q4Head: q4Head, activeRows: count,
                       unrotateForF16Head: unrotateForF16Head)
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
