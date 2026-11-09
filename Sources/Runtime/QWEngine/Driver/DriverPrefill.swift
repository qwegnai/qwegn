// DriverPrefill.swift — QWEngineDriver DriverPrefill family (split from QWEngineDriver.swift; same-module extension).
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
    func stridedRow(_ tv: QWTensorValue, row: Int, cols: Int, dt: QWDType, rowStrideBytes: Int? = nil) -> QWTensorValue {
        let stride = rowStrideBytes ?? Int((Double(cols) * dt.bytesPerParam).rounded())
        return QWTensorValue(
            region: tv.region,
            offset: tv.offset + row * stride,
            shape: QWTensorShape([cols], dt)
        )
    }

    func prefixRows(_ tv: QWTensorValue, rows: Int, cols: Int, dt: QWDType) -> QWTensorValue {
        return QWTensorValue(
            region: tv.region,
            offset: tv.offset,
            shape: QWTensorShape([rows * cols], dt)
        )
    }

    func prefillChunk(_ session: QWEngineSession, tokens: [Int],
                              start: Int, count: Int,
                              runsHead: Bool = true, mode: GDNPrefillMode,
                              permit: QWGenerationPermit?) throws -> Int {
        let T = count
        guard T > 0, T <= psc.maxChunk, start >= 0, start + T <= tokens.count else {
            throw QWEngineDriverError.state("prefillChunk invalid T=\(T)")
        }
        flight.lock()
        defer { flight.unlock() }
        try requestGate.validate(permit)
        guard verifiedSession == nil else { throw QWEngineDriverError.state("resolve verification before prefill") }
        try claimMLAIndexerState(session)

        let pos = session.position
        guard pos + T <= ropeRows else { throw QWEngineDriverError.posBeyondTable(pos + T) }
        let mtpRows = try prepareMTPSeeding(session, count: T)
        if session.kv.tokenCapacity < pos + T {
            try kvCache.grow(session.kv, to: pos + T + QWKVCache.pageTokens)
            session.attnKVViews = try kvCache.layerViewsAll(session.kv, layers: attnLayers)
        }
        try kvCache.prepareWrites(session.kv, positions: pos..<(pos + T))

        let H = geometry.hidden

        // NOTE (P2): prefillChunk deliberately stays on the SERIAL encoder.
        // Its per-token loops place barriers only at REGION seams, not at
        // every per-token producer→consumer edge — the serial encoder's
        let encodeStart = prefillTimingLedger.isEnabled ? Date() : nil
        guard let firstPair = hal.begin() else { throw QWEngineDriverError.state("cmdbuffer") }
        var cb: any MTLCommandBuffer = firstPair.0
        var enc: any MTLComputeCommandEncoder = firstPair.1
        activeCB = cb
        activeEnc = enc
        stepCBs = []
        beginPrefillProfileChunk()
        defer {
            activeCB = nil
            activeEnc = nil
            stepCBs = nil
        }
        var committedCbs: [any MTLCommandBuffer] = []

        if ProcessInfo.processInfo.environment["QW_DEBUG_LAYERS"] == "1" {
            fputs("GPU prefill tokens[\(start)..<\(start+T)]: \(Array(tokens[start..<(start+T)]))\n", stderr)
        }
        try copyEmbedRows(tokens, start: start, count: T, to: psc.x)
        if !activeVisualSpans.isEmpty {
            applyVisualSpans(start: pos, count: T, to: psc.x)
        }
        if let mtpPromptEmbeddings {
            let src = psc.x.region.buffer.contents().advanced(by: psc.x.region.base + psc.x.offset)
            let dst = mtpPromptEmbeddings.region.buffer.contents().advanced(by: mtpPromptEmbeddings.region.base + mtpPromptEmbeddings.offset)
            memcpy(dst, src, T * H * 2)
        }

        for (v, p) in geometry.executionPlan.enumerated() {
            let lw = layersW[p]
            cb = activeCB ?? cb
            enc = activeEnc ?? enc
            let shouldSlice = (v == 16 || (pos >= 131_072 && v > 16 && v % 8 == 0)) && T >= 16
            if shouldSlice {
                hal.end(cb, enc)
                if !pendingColdUnloads.isEmpty {
                    cb.waitUntilCompleted()
                    flushPendingColdUnloads()
                }
                committedCbs.append(cb)
                stepCBs?.append(cb)
                guard let nextPair = hal.begin() else { throw QWEngineDriverError.state("cmdbuffer slice") }
                cb = nextPair.0
                enc = nextPair.1
                activeCB = cb
                activeEnc = enc
            }
            // The layer's kind was resolved when the plan was read: the loop
            // looks the kernel up instead of switching on the kind.
            try layerKernels[p].encodePrefillChunk(QWLayerPrefillContext(
                driver: self, enc: activeEnc ?? enc, session: session, position: pos, layer: v,
                weights: lw, rows: T, mode: mode))
        }
        cb = activeCB ?? cb
        enc = activeEnc ?? enc

        var nextId = 0
        if runsHead {
            profileStage("head")
            enc = activeEnc ?? enc
            let lastLn1 = stridedRow(psc.ln1, row: T - 1, cols: H, dt: .f16)
            try encodeTargetHeadGreedy(enc, x: lastLn1,
                                       writeLogits: session.stochastic != nil || activeCollectLogprobs || activeConstraint != nil)
            // M2 FIX: `generated` is a COMMITTED counter — it advances after
            // the cb lands (below), never before. An errored chunk threw and
            // left the count honest.
        }

        if let mtpPromptEmbeddings {
            enc.endEncoding()
            try encodeMTPSeeding(cb, session: session,
                                 embeddings: prefixRows(mtpPromptEmbeddings, rows: T, cols: H, dt: .f16),
                                 hidden: prefixRows(psc.ln1, rows: T, cols: H, dt: .f16),
                                 count: T, cacheRows: mtpRows)
            cb.commit()
        } else {
            hal.end(cb, enc)
        }
        committedCbs.append(cb)
        stepCBs?.append(cb)
        let encodeMicros = encodeStart.map { Date().timeIntervalSince($0) * 1e6 }
        let allCbs = stepCBs ?? committedCbs
        for sliceCb in allCbs {
            sliceCb.waitUntilCompleted()
        }
        flushPendingColdUnloads()
        endPrefillProfileChunk(tokens: T)
        if let encodeMicros {
            let gpuMicros = Self.gpuBusyMicros(allCbs)
            prefillTimingLedger.record(
                encodeMicros: encodeMicros,
                gpuMicros: gpuMicros > 0 ? gpuMicros : nil)
        }
        for sliceCb in allCbs where sliceCb.error != nil {
            commandBufferErrors += 1   // M1 triad (prefill path)
        }
        // H3 FIX (prefill): same law as decodeStep — a failed chunk cb
        // would return a stale token and leave position/state half-true.
        for sliceCb in allCbs {
            if mtpCache != nil, sliceCb.status != .completed {
                session.stateFailed = true
                throw QWEngineDriverError.state("MTP prefill GPU completion failed: \(String(describing: sliceCb.error))")
            }
            if let e = sliceCb.error {
                if mtpCache != nil { session.stateFailed = true }
                throw QWEngineDriverError.state("prefill cb: \(e.localizedDescription)")
            }
        }
        if runsHead {
            let rawId = Int(sc.token.region.buffer.contents()
                .advanced(by: sc.token.region.base + sc.token.offset)
                .assumingMemoryBound(to: Int32.self).pointee)
            nextId = try selectConstrainedToken(session: session, rawArgmax: rawId)
            session.generated += 1    // M2: committed only after the cb landed (the head ran once)
            session.tokenHistory.append(nextId)
        }

        session.position += T
        // H4 (full): every attention layer in this chunk materialised rows
        // pos…pos+T−1 in-kernel (the attnDecode split-0 writes). Raise the
        // cache cursor once per committed chunk (monotonic max; the rows of
        // ALL 16 attn layers live under this one token-count cursor — the
        // same single-cursor law as append()/markWritten from decodeStep).
        kvCache.markWritten(session.kv, upTo: pos + T)
        commitMTPSeeding(session, position: pos, count: T, cacheRows: mtpRows)
        if mtpCache != nil, runsHead {
            session.roundRecord.pendingToken = nextId
            session.roundRecord.outputCount = session.generated
        }
        return nextId
    }

    /// GDN prefill: chunked K1/K2/K3 scan (default) or naive C12 streamed decode fallback.
    @discardableResult
    public func prefill(_ session: QWEngineSession, tokens: [Int],
                        mode: GDNPrefillMode? = nil) throws -> Int {
        try requestGate.withRequest(flight: flight) { permit in
            guard session.stochastic == nil else {
                throw QWEngineDriverError.state("use stochastic prefill for a sampled session")
            }
            return try prefill(session, tokens: tokens, mode: mode, permit: permit)
        }
    }

    func prefill(_ session: QWEngineSession, tokens: [Int],
                         mode: GDNPrefillMode?, permit: QWGenerationPermit) throws -> Int {
        guard !tokens.isEmpty else { return 0 }
        session.tokenHistory = tokens
        let effectiveMode = mode ?? gdnPrefillMode
        switch effectiveMode {
        case .naive:
            var lastToken = 0
            for t in tokens { lastToken = try decodeStep(session, token: t, permit: permit) }
            return lastToken
        case .chunked, .regres:
            // .regres only swaps the GDN-layer recurrence kernel (inner
            // switch in prefillChunk, keyed off this request's mode); the
            // surrounding batched-chunk architecture is identical to
            // .chunked (EXP-9 sub-item 1, opt-in).
            // C5 (prefill entry): validate the WHOLE token list before any
            // chunk opens an encoder — a bad id must fail loud, not mid-stream.
            for t in tokens where t < 0 || t >= geometry.vocab { throw QWEngineDriverError.badToken(t) }
            var offset = 0
            var lastToken = 0
            let maxChunk = psc.maxChunk
            while offset < tokens.count {
                let stepCap = Self.adaptivePrefillChunkCap(position: session.position, maxChunk: maxChunk)
                let count = min(stepCap, tokens.count - offset)
                let isLast = (offset + count == tokens.count)
                lastToken = try prefillChunk(session, tokens: tokens,
                                             start: offset, count: count,
                                             runsHead: isLast, mode: effectiveMode, permit: permit)
                offset += count
            }
            if prefillScratchEviction.shouldEvict() {
                evictPrefillScratch()
            }
            return lastToken
        }
    }

    /// Greedy generate: prefill the prompt (chunked or naive fallback;
    /// only the last step's output is kept as the first generated token), then decode.
    /// Returns the generated ids (prompt excluded).
    public func generate(session: QWEngineSession, prompt: [Int], maxNew: Int,
                         stopIds: Set<Int> = [],
                         onToken: ((Int) -> Bool)? = nil) throws -> [Int] {
        try requestGate.withRequest(flight: flight) { permit in
            guard session.stochastic == nil else {
                throw QWEngineDriverError.state("use stochastic generation for a sampled session")
            }
            return try generate(session: session, prompt: prompt, maxNew: maxNew,
                         stopIds: stopIds, onToken: onToken, permit: permit)
        }
    }

    func generate(session: QWEngineSession, prompt: [Int], maxNew: Int,
                          stopIds: Set<Int>, onToken: ((Int) -> Bool)?,
                          permit: QWGenerationPermit) throws -> [Int] {
        guard !prompt.isEmpty else { return [] }
        QWGemm.configureDenseNax(temperature: 0, topP: 1, topK: 1)
        let t0 = Date()
        var next = try prefill(session, tokens: prompt, mode: gdnPrefillMode, permit: permit)
        let prefillSec = Date().timeIntervalSince(t0)
        var out: [Int] = []
        var stepMs: [Double] = []
        func sample(_ body: () throws -> Void) rethrows {
            let s0 = Date(); try body(); stepMs.append(Date().timeIntervalSince(s0) * 1000)
        }
        if stopIds.contains(next) {
            lastStopToken = next
            lastGeneration = QWGenerationStats(promptTokens: prompt.count,
                generatedTokens: 0, prefillSec: prefillSec, decodeSec: 0,
                firstTokenSec: prefillSec, stepMs: [])
            return out
        }
        out.append(next)
        if onToken?(next) == false {
            lastGeneration = QWGenerationStats(promptTokens: prompt.count,
                generatedTokens: 1, prefillSec: prefillSec, decodeSec: 0,
                firstTokenSec: prefillSec, stepMs: [])
            return out
        }
        let d0 = Date()
        while out.count < maxNew {
            try sample { next = try decodeStep(session, token: next, permit: permit) }
            if stopIds.contains(next) { lastStopToken = next; break }
            out.append(next)
            if onToken?(next) == false { break }
        }
        lastGeneration = QWGenerationStats(promptTokens: prompt.count,
            generatedTokens: out.count, prefillSec: prefillSec,
            decodeSec: Date().timeIntervalSince(d0),
            firstTokenSec: prefillSec, stepMs: stepMs)
        return out
    }
}
