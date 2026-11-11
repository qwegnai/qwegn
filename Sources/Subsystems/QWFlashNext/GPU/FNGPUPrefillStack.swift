import Foundation
import Metal
import QWAlloc
import QWGEMM
import QWHAL
import QWCore

/// Batched prefill: the whole prompt runs through the stack in one command buffer per layer, with T rows per layer.
/// The weights are read once per layer for all T tokens, so prefill costs a single pass over the model instead of
/// one pass per token. The GDN state and conv state carry over into the decode stack that follows.
public final class FNGPUPrefillStack {
    public struct Slot {
        public let weights: FNGPULayerWeights
        public let attn: FNGPUAttnDecodeState?    // the cache the prefill fills
        public let ple: FNNgramEmbedding?          // the n-gram embedding of this layer's PLE
        public let pleEmbed: QWTensorValue?        // its uploaded rows (one per prompt token, parity 0)
        public let pleEmbedAlt: QWTensorValue?     // alternate parity buffer for CPU/GPU pipelined staging
        public let scratch: FNGPUScratch            // per-layer scratch, T rows
        public init(weights: FNGPULayerWeights, attn: FNGPUAttnDecodeState?, ple: FNNgramEmbedding?,
                    pleEmbed: QWTensorValue?, pleEmbedAlt: QWTensorValue? = nil, scratch: FNGPUScratch) {
            self.weights = weights
            self.attn = attn
            self.ple = ple
            self.pleEmbed = pleEmbed
            self.pleEmbedAlt = pleEmbedAlt
            self.scratch = scratch
        }
    }

    enum StreamStage {
        case host([Float])
        case gathered(FNGPUEmbedTable)
    }

    public let slots: [Slot]
    public let width: Int, maxRows: Int
    public private(set) var rows: Int
    let arena: QWArena, ctx: FNGPUContext
    let streamA: QWTensorValue, streamB: QWTensorValue, tokensBuf: QWTensorValue
    let gpuPLE: (table: FNGPUPLETable, embeddings: [FNNgramEmbedding], headCount: Int,
                 rowIds: QWTensorValue, dstOffsets0: QWTensorValue, dstOffsets1: QWTensorValue)?
    public private(set) var lastStreamTensor: QWTensorValue
    /// Signalled by the host once a chunk's PLE rows are staged; the first PLE layer's command buffer waits on it.
    private let pleReady: (any MTLSharedEvent)?
    private var pleReadyValue: UInt64 = 0

    public init(arena: QWArena, ctx: FNGPUContext, slots: [Slot], hidden: Int, hc: Int,
                gpuPLETable: FNGPUPLETable? = nil) throws {
        self.arena = arena; self.ctx = ctx; self.slots = slots
        self.width = hc * hidden
        let r = slots.first?.scratch.T ?? 1
        self.maxRows = r
        self.rows = r
        let act = arena.activationsRegion()
        streamA = try act.allocate(QWTensorShape([r * width], .f32))
        streamB = try act.allocate(QWTensorShape([r * width], .f32))
        tokensBuf = try act.allocate(QWTensorShape([r], .i32))
        lastStreamTensor = streamA
        let pleSlots = slots.compactMap { slot -> (FNNgramEmbedding, QWTensorValue, QWTensorValue)? in
            guard let e = slot.ple, let b0 = slot.pleEmbed else { return nil }
            return (e, b0, slot.pleEmbedAlt ?? b0)
        }
        if let table = gpuPLETable, let first = pleSlots.first {
            let hcNgram = first.0.heads.sizes.count
            let uniform = hcNgram > 0 && pleSlots.allSatisfy {
                $0.0.heads.sizes.count == hcNgram && $0.0.rows.headDim == table.headDim
                    && $0.1.region.buffer === act.region.buffer && $0.2.region.buffer === act.region.buffer
            }
            if uniform {
                let rowIds = try act.allocate(QWTensorShape([pleSlots.count * r * hcNgram], .i32))
                let dst0 = try act.allocate(QWTensorShape([pleSlots.count], .i32))
                let dst1 = try act.allocate(QWTensorShape([pleSlots.count], .i32))
                arena.write(dst0, from: pleSlots.map { UInt32(($0.1.region.base - act.region.base + $0.1.offset) / 4) })
                arena.write(dst1, from: pleSlots.map { UInt32(($0.2.region.base - act.region.base + $0.2.offset) / 4) })
                self.gpuPLE = (table, pleSlots.map(\.0), hcNgram, rowIds, dst0, dst1)
            } else {
                self.gpuPLE = nil
            }
        } else {
            self.gpuPLE = nil
        }
        pleReady = (self.gpuPLE == nil) ? ctx.hal.device.makeSharedEvent() : nil
    }

    /// Adjusts the active row count for the next prefill chunk without reallocating scratch buffers.
    public func setActiveRows(_ newRows: Int) {
        precondition(newRows >= 1 && newRows <= maxRows, "active rows (\(newRows)) must be in 1...\(maxRows)")
        guard newRows != rows || slots.first?.scratch.T != newRows else { return }
        self.rows = newRows
        for slot in slots {
            slot.scratch.T = newRows
        }
    }

    /// Runs `mtp` over the prefill chunk that just completed, warming up the MTP KV cache at `startPos..<(startPos + rows)`
    /// and returning the MTP draft token for row `rows - 1` when `projectLastRow` is true.
    public func forwardMTP(mtp: FNGPUMTP, nextTokens: [Int32], startPos: Int = 0, projectLastRow: Bool = true) throws -> Int {
        guard let last = slots.last else { return 0 }
        let preInj = (base: last.scratch.h2, src: last.scratch.moeOut, scale: last.scratch.injB)
        return try mtp.forwardPrefill(streamBuf: lastStreamTensor, preInject: preInj, nextTokens: nextTokens,
                                      startPos: startPos, prefillScratch: last.scratch, projectLastRow: projectLastRow)
    }

    /// Runs the whole prompt (or chunk starting at `startPos`). `streamIn` holds the T width-wide rows (the embedding, expanded over the hc groups).
    /// The last row's logits come back from the head; the stream itself never leaves the device.
    public func forward(history: [Int64], streamIn: [Float], head: FNGPUHead, startPos: Int = 0,
                        readStream: Bool = true, runHead: Bool = true,
                        mtp: FNGPUMTP? = nil, projectMTP: Bool = false,
                        pleParity: Int = 0, plePrecomputed: Bool = false,
                        nextChunkStartPos: Int? = nil,
                        nextChunkCount: Int? = nil) throws -> (token: Int, stream: [Float]) {
        precondition(streamIn.count == rows * width && history.count >= startPos + rows)
        return try runPrefill(history: history, stage: .host(streamIn), head: head, startPos: startPos,
                              readStream: readStream, runHead: runHead,
                              mtp: mtp, projectMTP: projectMTP,
                              pleParity: pleParity, plePrecomputed: plePrecomputed,
                              nextChunkStartPos: nextChunkStartPos,
                              nextChunkCount: nextChunkCount)
    }

    /// Runs the prompt chunk starting at `startPos`, gathering all `rows` embeddings on-device from `embedTable`.
    public func forward(history: [Int64], embedTable: FNGPUEmbedTable, head: FNGPUHead, startPos: Int = 0,
                        readStream: Bool = false, runHead: Bool = true,
                        mtp: FNGPUMTP? = nil, projectMTP: Bool = false,
                        pleParity: Int = 0, plePrecomputed: Bool = false,
                        nextChunkStartPos: Int? = nil,
                        nextChunkCount: Int? = nil) throws -> (token: Int, stream: [Float]) {
        precondition(embedTable.hc * embedTable.hidden == width && history.count >= startPos + rows)
        return try runPrefill(history: history, stage: .gathered(embedTable), head: head, startPos: startPos,
                              readStream: readStream, runHead: runHead,
                              mtp: mtp, projectMTP: projectMTP,
                              pleParity: pleParity, plePrecomputed: plePrecomputed,
                              nextChunkStartPos: nextChunkStartPos,
                              nextChunkCount: nextChunkCount)
    }

    /// Stages all 12 PLE n-gram embeddings for `startPos..<(startPos + count)` in a single flattened parallel dispatch.
    private func pleStageTargets(parity: Int) -> [(ple: FNNgramEmbedding, addr: Int)] {
        slots.compactMap { slot in
            guard let e = slot.ple,
                  let buf = (parity == 0 ? slot.pleEmbed : (slot.pleEmbedAlt ?? slot.pleEmbed)) else {
                return nil
            }
            let ptr = arena.cpuPointer(of: buf).assumingMemoryBound(to: Float.self)
            return (e, Int(bitPattern: ptr))
        }
    }

    public func stagePLE(history: [Int64], startPos: Int, count: Int? = nil, parity: Int = 0) {
        if gpuPLE != nil { return }
        Self.stagePLETargets(pleStageTargets(parity: parity), history: history,
                             startPos: startPos, count: count ?? rows)
    }

    private static func stagePLETargets(_ targets: [(ple: FNNgramEmbedding, addr: Int)],
                                        history: [Int64], startPos: Int, count: Int) {
        QWPLEOperator.stageTargets(targets, history: history, startPos: startPos, count: count)
    }

    static func chunkEnd(start: Int, count: Int, layersPerCB: Int, firstPLE: Int?) -> Int {
        QWLayerStackSchedule.prefillChunkEnd(start: start, count: count, layersPerCB: layersPerCB, firstPLE: firstPLE)
    }

    private func runPrefill(history: [Int64], stage: StreamStage, head: FNGPUHead, startPos: Int,
                            readStream: Bool, runHead: Bool,
                            mtp: FNGPUMTP?, projectMTP: Bool,
                            pleParity: Int, plePrecomputed: Bool,
                            nextChunkStartPos: Int?,
                            nextChunkCount: Int? = nil) throws -> (token: Int, stream: [Float]) {
        let trace = FNGPUExecutionTrace.enabled ? FNGPUExecutionTrace() : nil
        if let gple = gpuPLE {
            let idPtr = arena.cpuPointer(of: gple.rowIds).assumingMemoryBound(to: UInt32.self)
            QWPLEOperator.writeRowIds(gple.embeddings, into: idPtr,
                                      history: history, startPos: startPos, count: rows)
        }
        let targets = (gpuPLE != nil || plePrecomputed) ? [] : pleStageTargets(parity: pleParity)
        let firstPLE = (!plePrecomputed && !targets.isEmpty) ? slots.firstIndex { $0.weights.ple != nil } : nil
        var pleStaged = gpuPLE != nil || plePrecomputed || targets.isEmpty
        if !pleStaged, (firstPLE == nil || firstPLE == 0 || pleReady == nil) {
            Self.stagePLETargets(targets, history: history, startPos: startPos, count: rows)
            pleStaged = true
        }
        // The CPU stages the PLE rows while the layers before the first PLE layer are encoded and run; that
        // layer's command buffer waits for the host signal on the GPU, so no submission blocks on staging.
        let pleGroup = DispatchGroup()
        var pleWait: (event: any MTLSharedEvent, value: UInt64, layer: Int)?
        if !pleStaged, let consumer = firstPLE, let event = pleReady {
            pleReadyValue += 1
            let value = pleReadyValue
            let rowsNow = rows
            pleWait = (event, value, consumer)
            DispatchQueue.global(qos: .userInteractive).async(group: pleGroup) {
                Self.stagePLETargets(targets, history: history, startPos: startPos, count: rowsNow)
                event.signaledValue = value
            }
            pleStaged = true
        }
        defer { pleGroup.wait() }
        trace?.preparedPLE()
        var cur = streamA, nxt = streamB
        let needHead = runHead || readStream || (mtp != nil && startPos + rows >= history.count)
        let layersPerCB = 12
        let totalSlots = max(1, slots.count)
        var cbs: [any MTLCommandBuffer] = []
        var chunkStart = 0
        while chunkStart < totalSlots {
            let encodeStart = trace?.clock
            let chunkEnd = slots.isEmpty ? 0 : Self.chunkEnd(start: chunkStart, count: slots.count,
                                                              layersPerCB: layersPerCB,
                                                              firstPLE: pleWait?.layer)
            let isFirstChunk = (chunkStart == 0)
            let isLastChunk = (chunkEnd >= slots.count)
            let waitsForPLE = pleWait.map { chunkStart <= $0.layer && $0.layer < max(chunkEnd, chunkStart + 1) } ?? false
            let begun = waitsForPLE
                ? ctx.hal.begin(.concurrent, waitingFor: pleWait!.event, value: pleWait!.value)
                : ctx.hal.begin(.concurrent)
            guard let pair = begun else {
                throw NSError(domain: "FNGPUPrefillStack", code: 1, userInfo: [NSLocalizedDescriptionKey: "no command buffer"])
            }
            let cb = pair.0
            var enc = pair.1
            if isFirstChunk {
                switch stage {
                case .host(let streamIn):
                    arena.write(streamA, from: streamIn)
                case .gathered(let embedTable):
                    let tokIds = (startPos..<(startPos + rows)).map { Int32(history[$0]) }
                    arena.write(tokensBuf, from: tokIds)
                    try embedTable.gather(enc: enc, hal: ctx.hal, token: tokensBuf,
                                          out: streamA, rows: rows, hc: embedTable.hc)
                }
                if let gple = gpuPLE {
                    let dstOffs = (pleParity == 0) ? gple.dstOffsets0 : gple.dstOffsets1
                    try QWGemm.encodeFnPLEVQ2Gather(enc, hal: ctx.hal, table: gple.table,
                                                    rowIds: gple.rowIds, targetFloatOffsets: dstOffs,
                                                    outRegion: arena.activationsRegion().region,
                                                    rowsPerTarget: rows * gple.headCount,
                                                    numTargets: gple.embeddings.count)
                }
                enc.memoryBarrier(scope: .buffers)
            }
            for i in chunkStart..<chunkEnd {
                let slot = slots[i]
                let inInj: (base: QWTensorValue, src: QWTensorValue, scale: QWTensorValue)? =
                    (i > 0 && slot.weights.ple == nil)
                    ? (slots[i - 1].scratch.h2, slots[i - 1].scratch.moeOut, slots[i - 1].scratch.injB)
                    : nil
                let deferOut = (i + 1 < slots.count) ? (slots[i + 1].weights.ple == nil) : true
                let pleBuf = (pleParity == 0) ? slot.pleEmbed : (slot.pleEmbedAlt ?? slot.pleEmbed)
                try FNGPULayerPasses.layerPrefill(&enc, cb: cb, ctx: ctx, layer: slot.weights, s: slot.scratch, stream: cur, out: nxt,
                                                  attnState: slot.attn, pos: startPos, pleEmbed: pleBuf,
                                                  inInject: inInj, deferOutInject: deferOut)
                swap(&cur, &nxt)
            }
            if isLastChunk {
                lastStreamTensor = cur
                let last = cur.slice(byteOffset: (rows - 1) * width * 4, count: width, dtype: .f32)
                if needHead {
                    let tailInj = slots.last.map { s -> (base: QWTensorValue, src: QWTensorValue, scale: QWTensorValue) in
                        let r = rows - 1, w = s.scratch.width, h = s.scratch.hidden, c = s.scratch.hc
                        return (s.scratch.h2.slice(byteOffset: r * w * 4, count: w, dtype: .f32),
                                s.scratch.moeOut.slice(byteOffset: r * h * 4, count: h, dtype: .f32),
                                s.scratch.injB.slice(byteOffset: r * c * 4, count: c, dtype: .f32))
                    }
                    try head.encode(enc, hal: ctx.hal, stream: last, inInject: tailInj)
                }
                if let mtp, let lastSlot = slots.last {
                    let nextToks = (0..<rows).map { i -> Int32 in
                        (startPos + i + 1 < history.count) ? Int32(history[startPos + i + 1]) : 0
                    }
                    mtp.writeTokens(nextToks)
                    if startPos + rows >= history.count {
                        let lastTokSlice = mtp.tokensBuf.slice(byteOffset: (rows - 1) * 4, count: 1, dtype: .i32)
                        try QWGemm.encodeArgmaxFin(enc, hal: ctx.hal, keysHi: head.keysHi, keysLo: head.keysLo,
                                                   token: lastTokSlice, slots: QWGemm.blockArgmaxSlots(head.vocab), rows: 1)
                        enc.memoryBarrier(scope: .buffers)
                    }
                    let preInj = (base: lastSlot.scratch.h2, src: lastSlot.scratch.moeOut, scale: lastSlot.scratch.injB)
                    try mtp.encode(&enc, cb: cb, streamPrev: cur, tokens: mtp.tokensBuf, pos: startPos, rows: rows,
                                   scratch: lastSlot.scratch, preInject: preInj,
                                   projectRows: projectMTP ? 1 : 0, usePrefillPath: rows > 1)
                }
            }
            trace?.submitting(cb, lower: chunkStart, upper: chunkEnd, encodeStart: encodeStart)
            ctx.hal.end(cb, enc)
            cbs.append(cb)
            if slots.isEmpty { break }
            chunkStart = chunkEnd
        }
        let last = cur.slice(byteOffset: (rows - 1) * width * 4, count: width, dtype: .f32)
        if let nextPos = nextChunkStartPos {
            let prepareStart = trace?.clock
            stagePLE(history: history, startPos: nextPos, count: nextChunkCount, parity: 1 - pleParity)
            trace?.preparedNextPLE(since: prepareStart)
        }
        let waitStart = trace?.clock
        cbs.last?.waitUntilCompleted()
        trace?.waitedForCompletion(since: waitStart)
        var gpuTotal = 0.0
        for cb in cbs {
            if let err = cb.error { throw err }
            gpuTotal += (cb.gpuEndTime - cb.gpuStartTime) * 1000
        }
        lastGPUMillis = gpuTotal
        try trace?.emit(phase: "prefill", position: startPos, rows: rows,
                        pleRoute: gpuPLE != nil ? "gpu" : (slots.contains { $0.ple != nil } ? "host" : "none"),
                        mappedPLEBytes: gpuPLE?.table.buffer.length ?? 0, includesMTP: mtp != nil)
        let tok = needHead ? head.argmaxToken() : 0
        lastMTPDraft = (mtp != nil && projectMTP) ? mtp!.mtpHead.argmaxToken() : 0
        guard readStream else { return (tok, []) }
        return (tok, arena.read(last, as: Float.self, count: width))
    }

    /// One n-gram embedding row per prompt token: the context ends at that token.
    static func pleRows(_ e: FNNgramEmbedding, _ history: [Int64]) -> [Float] {
        e.embed(history)
    }

    /// GPU time of the last prefill command buffer, ms.
    public private(set) var lastGPUMillis: Double = 0
    /// Draft token produced by fused MTP during the last prefill command buffer (when `projectMTP` was true).
    public private(set) var lastMTPDraft: Int = 0

    /// Splits a prefill range at visual span boundaries and expands visual embeddings across `hc` streams.
    public static func resolveSubchunk(pos: Int, limitPos: Int, chunkSize: Int,
                                       hidden: Int, hc: Int,
                                       visualSpans: [QWVisualSpan]) -> (rows: Int, visualStreamIn: [Float]?) {
        let baseRows = min(chunkSize, max(0, limitPos - pos))
        guard baseRows > 0, !visualSpans.isEmpty else { return (baseRows, nil) }
        let width = hc * hidden
        for span in visualSpans {
            let spanStart = span.tokenOffset
            let spanEnd = spanStart + span.tokenCount
            if pos >= spanStart && pos < spanEnd {
                let rows = min(baseRows, spanEnd - pos)
                let localStart = (pos - spanStart) * hidden
                var streamIn = [Float](repeating: 0, count: rows * width)
                span.embeddings.withUnsafeBufferPointer { src in
                    streamIn.withUnsafeMutableBufferPointer { dst in
                        for r in 0..<rows {
                            let srcRow = localStart + r * hidden
                            let dstRow = r * width
                            for c in 0..<hc {
                                let dstBase = dstRow + c * hidden
                                for d in 0..<hidden {
                                    dst[dstBase + d] = src[srcRow + d]
                                }
                            }
                        }
                    }
                }
                return (rows, streamIn)
            } else if spanStart > pos {
                return (min(baseRows, spanStart - pos), nil)
            }
        }
        return (baseRows, nil)
    }
}
