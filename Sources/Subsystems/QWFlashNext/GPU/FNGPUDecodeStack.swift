import Foundation
import Metal
import QWAlloc
import QWGEMM
import QWGDN
import QWHAL

/// Chains decoder layers token by token: GDN and PLE state, and per-attention-layer KV/index caches, persist across
/// calls. One command buffer per step. `history` is the whole token sequence so far (the PLE n-gram context).
public final class FNGPUDecodeStack {
    public struct Slot {
        public let weights: FNGPULayerWeights
        public let scratch: FNGPUScratch          // T = 1
        public let attn: FNGPUAttnDecodeState?    // non-nil for full-attention layers
        public let ple: FNNgramEmbedding?         // host n-gram rows
        public init(weights: FNGPULayerWeights, scratch: FNGPUScratch, attn: FNGPUAttnDecodeState?, ple: FNNgramEmbedding?) {
            self.weights = weights; self.scratch = scratch; self.attn = attn; self.ple = ple
        }
    }

    /// How a step's input stream is staged: host slice or on-device bf16 gather driven by `head.token` or `tokensBuf`.
    enum StreamStage {
        case host([Float])
        case gathered(FNGPUEmbedTable)
        case gatheredRows(FNGPUEmbedTable, [Int32])
    }

    public let slots: [Slot]
    public let width: Int
    let rows: Int
    let arena: QWArena, ctx: FNGPUContext
    let streamA: QWTensorValue, streamB: QWTensorValue, tokensBuf: QWTensorValue
    let pleTargets: [(ple: FNNgramEmbedding, addr: Int)]
    let cuts: [Int]
    let steps: [[Step]]
    public private(set) var lastStreamTensor: QWTensorValue

    public init(arena: QWArena, ctx: FNGPUContext, slots: [Slot], hidden: Int, hc: Int, rows: Int = 1) throws {
        self.arena = arena; self.ctx = ctx; self.slots = slots
        self.width = hc * hidden
        self.rows = rows
        let act = arena.activationsRegion()
        streamA = try act.allocate(QWTensorShape([rows * width], .f32))
        streamB = try act.allocate(QWTensorShape([rows * width], .f32))
        tokensBuf = try act.allocate(QWTensorShape([rows], .i32))
        lastStreamTensor = streamA
        pleTargets = slots.compactMap { slot in
            guard let e = slot.ple, slot.weights.ple != nil else { return nil }
            let ptr = arena.cpuPointer(of: slot.scratch.pleEmbed).assumingMemoryBound(to: Float.self)
            return (e, Int(bitPattern: ptr))
        }
        let c = Self.decodeCuts(slotCount: slots.count)
        self.cuts = c
        var chunkHasPLE: [Bool] = []
        for idx in 0..<(c.count - 1) {
            var hasPLE = false
            for i in c[idx]..<c[idx + 1] where slots[i].ple != nil { hasPLE = true; break }
            chunkHasPLE.append(hasPLE)
        }
        self.steps = Self.stepOrder(chunkHasPLE: chunkHasPLE)
    }

    /// Runs all slots over `activeRows ?? rows` draft-verify rows, gathering embeddings on-device from `embedTable`.
    public func forwardRows(history: [Int64], embedTable: FNGPUEmbedTable, head: FNGPUHead,
                            readStream: Bool = false, verifyAllRows: Bool = false,
                            mtp: FNGPUMTP? = nil, activeRows: Int? = nil,
                            relaxedMargin: Float = 0, relaxedRows: Int? = nil,
                            relaxedMinProb: Float = 0) throws -> (stream: [Float], token: Int) {
        let k = activeRows ?? rows
        precondition(k >= 1 && k <= rows && !history.isEmpty && history.count >= k && embedTable.hc * embedTable.hidden == width)
        precondition(!verifyAllRows || Self.supportsRollbackWindow(k),
                     "verify rows exceed the four-slot recurrent and PLE rollback ring")
        let toks = history.suffix(k).map { Int32($0) }
        return try runRows(history: history, stage: .gatheredRows(embedTable, toks), head: head,
                           readStream: readStream, verifyAllRows: verifyAllRows, mtp: mtp,
                           activeRows: k, relaxedMargin: relaxedMargin, relaxedRows: relaxedRows,
                           relaxedMinProb: relaxedMinProb)
    }

    /// Rolls back `rejectedCount` verify rows (`1 <= rejectedCount < rows`) after draft rejection,
    /// restoring the exact post-accepted state across all GDN recurrent/conv states and PLE history in O(1)
    /// via 4-slot ring buffer parity subtraction.
    public func rollbackVerifyRows(rejectedCount: Int) {
        precondition(rejectedCount >= 0 && rejectedCount < QWFourSlotRollbackRing.maxVerifyRows,
                     "rejected rows exceed the rollback ring")
        guard rejectedCount > 0 else { return }
        for slot in slots {
            if case .gdn(let g) = slot.weights.mixer {
                g.convParity = QWFourSlotRollbackRing.rollback(g.convParity, by: rejectedCount)
                g.stateParity = QWFourSlotRollbackRing.rollback(g.stateParity, by: rejectedCount)
            }
            if let p = slot.weights.ple {
                p.parity = QWFourSlotRollbackRing.rollback(p.parity, by: rejectedCount)
            }
        }
    }

    static func supportsRollbackWindow(_ rows: Int) -> Bool {
        QWFourSlotRollbackRing.supportsWindow(rows)
    }

    public func rollbackLastVerifyRow() throws {
        rollbackVerifyRows(rejectedCount: 1)
    }

    /// The chunk boundaries. Exposed so the ordering test can pin them.
    public static func decodeCuts(slotCount: Int) -> [Int] {
        QWLayerStackSchedule.decodeCuts(slotCount: slotCount, hasPLE: true)
    }

    /// What the decode stack does between chunks.
    public typealias Step = QWLayerStackSchedule.Step

    /// The steps for each chunk, in chunk order. The chunk that does NOT
    /// consume the embedding is committed FIRST, so the GPU is busy while
    /// the SSD reads are in flight; the wait falls in the consuming chunk.
    public static func stepOrder(chunkHasPLE: [Bool]) -> [[Step]] {
        QWLayerStackSchedule.stepOrder(chunkHasPLE: chunkHasPLE)
    }

    private func runRows(history: [Int64], stage: StreamStage, head: FNGPUHead,
                         readStream: Bool, verifyAllRows: Bool = false,
                         mtp: FNGPUMTP? = nil, activeRows: Int? = nil,
                         relaxedMargin: Float = 0, relaxedRows: Int? = nil,
                         relaxedMinProb: Float = 0) throws -> (stream: [Float], token: Int) {
        let k = activeRows ?? rows
        // Rows at or beyond `relaxedRows` verify by exact argmax.
        let relaxedLimit = min(relaxedRows ?? k, k)
        if k != rows {
            for slot in slots { slot.scratch.T = k }
        }
        defer {
            if k != rows {
                for slot in slots { slot.scratch.T = rows }
            }
        }
        let pos = history.count - k   // row 0 sits at absolute position `pos`
        let numPLE = pleTargets.count
        var pleGroup: DispatchGroup? = nil
        if numPLE > 0 {
            let grp = DispatchGroup()
            let targets = pleTargets
            let hc = targets[0].ple.heads.sizes.count
            let headsPerPLE = k * hc
            let totalHeads = numPLE * headsPerPLE
            grp.enter()
            DispatchQueue.global(qos: .userInteractive).async {
                if totalHeads > 4 {
                    DispatchQueue.concurrentPerform(iterations: totalHeads) { idx in
                        let pIdx = idx / headsPerPLE
                        let rem = idx - pIdx * headsPerPLE
                        let r = rem / hc
                        let head = rem - r * hc
                        let target = targets[pIdx]
                        let dst = UnsafeMutablePointer<Float>(bitPattern: target.addr)!
                        target.ple.embedTokenHead(into: dst, tokens: history, startPos: pos, t: r, head: head)
                    }
                } else {
                    for idx in 0..<(numPLE * k) {
                        let pIdx = idx / k
                        let r = idx - pIdx * k
                        let target = targets[pIdx]
                        let dst = UnsafeMutablePointer<Float>(bitPattern: target.addr)!
                        target.ple.embedToken(into: dst, tokens: history, startPos: pos, t: r)
                    }
                }
                grp.leave()
            }
            pleGroup = grp
        }
        var cbs: [any MTLCommandBuffer] = []
        var cur = streamA, nxt = streamB
        for c in 0..<(cuts.count - 1) {
            let lo = cuts[c], hi = cuts[c + 1]
            guard let (cb, enc) = ctx.hal.begin(.concurrent) else {
                throw NSError(domain: "FNGPUDecodeStack", code: 1, userInfo: [NSLocalizedDescriptionKey: "no command buffer"])
            }
            if c == 0 {
                switch stage {
                case .host(let streamIn):
                    arena.write(streamA, from: streamIn)
                case .gathered(let embedTable):
                    try embedTable.gather(enc: enc, hal: ctx.hal, token: head.token, out: streamA,
                                          rows: 1, hc: embedTable.hc)
                    enc.memoryBarrier(scope: .buffers)
                case .gatheredRows(let embedTable, let toks):
                    arena.write(tokensBuf, from: toks)
                    try embedTable.gather(enc: enc, hal: ctx.hal, token: tokensBuf, out: streamA,
                                          rows: k, hc: embedTable.hc)
                    enc.memoryBarrier(scope: .buffers)
                }
            }
            for i in lo..<hi {
                let slot = slots[i]
                let inInj: (base: QWTensorValue, src: QWTensorValue, scale: QWTensorValue)? =
                    (i > 0 && slot.weights.ple == nil)
                    ? (slots[i - 1].scratch.h2, slots[i - 1].scratch.moeOut, slots[i - 1].scratch.injB)
                    : nil
                let deferLast = verifyAllRows || !readStream || k == 1
                let deferOut = (i + 1 < slots.count) ? (slots[i + 1].weights.ple == nil) : deferLast
                try FNGPULayerPasses.layer(enc, ctx: ctx, layer: slot.weights, s: slot.scratch, stream: cur, out: nxt,
                                           attnState: slot.attn, pos: pos, inInject: inInj, deferOutInject: deferOut)
                swap(&cur, &nxt)
            }
            if hi == slots.count {
                if verifyAllRows {
                    let tailInj = slots.last.map { ($0.scratch.h2, $0.scratch.moeOut, $0.scratch.injB) }
                    let variational: QWVariationalAcceptance? =
                        (relaxedMargin > 0 && relaxedLimit > 1)
                        ? QWVariationalAcceptance(logits: head.logits, draftTokens: tokensBuf, vocab: head.vocab,
                                                  totalRows: relaxedLimit, margin: relaxedMargin,
                                                  minLogProb: QWVariationalAcceptance.minLogProb(forMinProbability: relaxedMinProb))
                        : nil
                    try head.encode(enc, hal: ctx.hal, stream: cur, inInject: tailInj, rows: k, variational: variational)
                    if let mtp {
                        enc.memoryBarrier(scope: .buffers)
                        try mtp.encode(enc, streamPrev: cur, tokens: head.token, pos: pos, rows: k,
                                       projectRows: 1, draftTokens: tokensBuf)
                    }
                } else {
                    let deferLast = !readStream || k == 1
                    let tailInj: (base: QWTensorValue, src: QWTensorValue, scale: QWTensorValue)? = !deferLast ? nil : slots.last.map { last in
                        let r = k - 1, w = last.scratch.width, h = last.scratch.hidden, c = last.scratch.hc
                        return (last.scratch.h2.slice(byteOffset: r * w * 4, count: w, dtype: .f32),
                                last.scratch.moeOut.slice(byteOffset: r * h * 4, count: h, dtype: .f32),
                                last.scratch.injB.slice(byteOffset: r * c * 4, count: c, dtype: .f32))
                    }
                    try head.encode(enc, hal: ctx.hal, stream: view(cur, row: k - 1), inInject: tailInj, rows: 1)
                    if let mtp, k == 1 {
                        enc.memoryBarrier(scope: .buffers)
                        try mtp.encode(enc, streamPrev: cur, tokens: head.token, pos: pos, rows: 1, projectRows: 1)
                    }
                }
            }
            // The wait comes before the commit of the chunk that CONSUMES
            // the embedding; the chunks before it are committed first, so the
            // GPU is already busy while the SSD reads are in flight. Each
            // chunk's own step list drives this, so the order cannot drift.
            for step in steps[c] {
                switch step {
                case .waitForPLE:
                    if let g = pleGroup { g.wait(); pleGroup = nil }
                case .commit:
                    ctx.hal.end(cb, enc)
                    cbs.append(cb)
                }
            }
        }
        lastStreamTensor = cur
        cbs.last?.waitUntilCompleted()
        var gpuTotal = 0.0
        for cb in cbs {
            if let err = cb.error { throw err }
            gpuTotal += (cb.gpuEndTime - cb.gpuStartTime) * 1000
        }
        lastGPUMillis = gpuTotal
        let tok = head.argmaxToken()
        guard readStream else { return ([], tok) }
        return (arena.read(cur, as: Float.self, count: k * width), tok)
    }

    private func view(_ t: QWTensorValue, row: Int) -> QWTensorValue {
        t.slice(byteOffset: row * width * 4, count: width, dtype: .f32)
    }

    static func stageBatchPLE(targets: [(ple: FNNgramEmbedding, addr: Int)], histories: [[Int64]],
                              positions: [Int], parallelHeads: Bool) {
        QWPLEOperator.stageBatch(targets: targets, histories: histories,
                                 positions: positions, parallelHeads: parallelHeads)
    }

    /// GPU time of the last `forward` command buffer, ms (command-buffer start to end).
    public private(set) var lastGPUMillis: Double = 0

    /// Runs all slots for the token at position history.count - 1, then the final mixer and the LM head.
    /// Residual stream is gathered on-device from the raw-bf16 table using `head.token` — no host staging.
    public func forwardToken(history: [Int64], embedTable: FNGPUEmbedTable, head: FNGPUHead,
                             readStream: Bool = true, mtp: FNGPUMTP? = nil) throws -> (token: Int, stream: [Float]) {
        precondition(!history.isEmpty && embedTable.hc * embedTable.hidden == width)
        let r = try runRows(history: history, stage: .gathered(embedTable), head: head, readStream: readStream, mtp: mtp)
        return (r.token, r.stream)
    }

    /// Runs all slots for a single pre-expanded host stream row (`width = hc * hidden`).
    public func forwardToken(history: [Int64], streamIn: [Float], head: FNGPUHead,
                             readStream: Bool = true, mtp: FNGPUMTP? = nil) throws -> (token: Int, stream: [Float]) {
        precondition(!history.isEmpty && streamIn.count == width)
        let r = try runRows(history: history, stage: .host(streamIn), head: head, readStream: readStream, mtp: mtp)
        return (r.token, r.stream)
    }

    /// Runs one batched decode step across `B` active sessions (`1 <= B <= rows`), returning the `B` argmax tokens.
    public func forwardBatch(histories: [[Int64]], activeSlots: [Int], totalSlots: Int,
                             slotStates: inout [FNBatchSlotState],
                             embedTable: FNGPUEmbedTable, head: FNGPUHead) throws -> [Int] {
        let B = activeSlots.count
        precondition(B >= 1 && B <= rows && histories.count == B && embedTable.hc * embedTable.hidden == width)
        let toks = histories.map { Int32($0.last!) }
        let positions = histories.map { $0.count - 1 }
        let numPLE = pleTargets.count
        var pleGroup: DispatchGroup? = nil
        if numPLE > 0 {
            let grp = DispatchGroup()
            let targets = pleTargets
            grp.enter()
            DispatchQueue.global(qos: .userInteractive).async {
                Self.stageBatchPLE(targets: targets, histories: histories, positions: positions,
                                   parallelHeads: true)
                grp.leave()
            }
            pleGroup = grp
        }
        var cbs: [any MTLCommandBuffer] = []
        var cur = streamA, nxt = streamB
        for c in 0..<(cuts.count - 1) {
            let lo = cuts[c], hi = cuts[c + 1]
            guard let (cb, enc) = ctx.hal.begin(.concurrent) else {
                throw NSError(domain: "FNGPUDecodeStack", code: 1, userInfo: [NSLocalizedDescriptionKey: "no command buffer"])
            }
            if c == 0 {
                arena.write(tokensBuf, from: toks)
                try embedTable.gather(enc: enc, hal: ctx.hal, token: tokensBuf, out: streamA,
                                      rows: B, hc: embedTable.hc)
                enc.memoryBarrier(scope: .buffers)
            }
            for i in lo..<hi {
                let slot = slots[i]
                let inInj: (base: QWTensorValue, src: QWTensorValue, scale: QWTensorValue)? =
                    (i > 0 && slot.weights.ple == nil)
                    ? (slots[i - 1].scratch.h2, slots[i - 1].scratch.moeOut, slots[i - 1].scratch.injB)
                    : nil
                let deferOut = (i + 1 < slots.count) ? (slots[i + 1].weights.ple == nil) : true
                try FNGPULayerPasses.layerBatch(enc, ctx: ctx, layer: slot.weights, s: slot.scratch,
                                                stream: cur, out: nxt, attnState: slot.attn,
                                                activeSlots: activeSlots, positions: positions,
                                                totalSlots: totalSlots, slotStates: &slotStates,
                                                inInject: inInj, deferOutInject: deferOut)
                swap(&cur, &nxt)
            }
            if hi == slots.count {
                let tailInj = slots.last.map { ($0.scratch.h2, $0.scratch.moeOut, $0.scratch.injB) }
                try head.encode(enc, hal: ctx.hal, stream: cur, inInject: tailInj, rows: B)
            }
            for step in steps[c] {
                switch step {
                case .waitForPLE:
                    if let g = pleGroup { g.wait(); pleGroup = nil }
                case .commit:
                    ctx.hal.end(cb, enc)
                    cbs.append(cb)
                }
            }
        }
        lastStreamTensor = cur
        cbs.last?.waitUntilCompleted()
        var gpuTotal = 0.0
        for cb in cbs {
            if let err = cb.error { throw err }
            gpuTotal += (cb.gpuEndTime - cb.gpuStartTime) * 1000
        }
        lastGPUMillis = gpuTotal
        return head.argmaxTokens(count: B)
    }
}
