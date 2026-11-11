// FNGPUSession.swift — One Flash-Next session's mutable state.
// The engine owns the numerics; this owns the request. The module seam hands
// one of these to the engine per session, so the engine never has to guess
// whose prompt it is running.

import Foundation
import QWAlloc
import QWCore
import QWHead

extension FNGPUEngine {
    public struct GenerationResult: Sendable {
        public let tokens: [Int]
        public let prefillSec: Double?
        public let prefillGPUSec: Double?
        public let decodeSec: Double?
        public let firstTokenSec: Double?
        public let stepMs: [Double]
        public let denseNaxLogMessage: String
        public let prefixCacheHit: Bool
        public let prefixCacheMatchedTokens: Int

        public init(tokens: [Int], prefillSec: Double?, prefillGPUSec: Double?, decodeSec: Double?,
                    firstTokenSec: Double?, stepMs: [Double], denseNaxLogMessage: String,
                    prefixCacheHit: Bool = false, prefixCacheMatchedTokens: Int = 0) {
            self.tokens = tokens
            self.prefillSec = prefillSec
            self.prefillGPUSec = prefillGPUSec
            self.decodeSec = decodeSec
            self.firstTokenSec = firstTokenSec
            self.stepMs = stepMs
            self.denseNaxLogMessage = denseNaxLogMessage
            self.prefixCacheHit = prefixCacheHit
            self.prefixCacheMatchedTokens = prefixCacheMatchedTokens
        }
    }
}

/// A Flash-Next session: the prompt so far, the shared sampler, and the
/// cursors the prefill walks with.
public struct FNGPUSessionData {
    /// The token history the model has seen.
    public var history: [Int64]
    /// The shared sampler. Its draw counter lives here, so the sequence is
    /// the session's own.
    public var sampler: QWSampler
    /// How far the prefill has consumed the prompt.
    public var prefillPos: Int
    /// The head's token for the last prefill row.
    public var lastArgmax: Int
    /// The rows the last prefill chunk ran.
    public var lastRows: Int
    /// The PLE parity the next chunk needs.
    public var pleParity: Int
    /// Whether the PLE for the next chunk is already computed.
    public var plePrecomputed: Bool
    /// The MTP draft the last prefill chunk produced.
    public var mtpDraft: Int
    /// Chained MTP drafts `[d1, d2, ...]` and the history length they are valid for.
    public var mtpChain: [Int]
    public var mtpChainPos: Int
    /// How many leading drafts of the current round came from the MTP head or its chain.
    /// Only these rows use variational acceptance; context-copy drafts verify exactly.
    var mtpSourcedDrafts: Int
    /// The current drafts continue a detected period; they verify by exact argmax.
    var periodicDrafts: Bool
    /// Online cost/acceptance model that picks the MTP chain depth each round.
    var chainPolicy: FNChainPolicy
    /// Whether the session runs the speculative path.
    public var useSpec: Bool
    /// The first generated token, sampled once and kept. The prefill is
    /// chunked, and the token must not be drawn twice.
    public var firstToken: Int
    /// Whether the first token has been drawn.
    public var hasFirstToken: Bool
    /// The GPU seconds the prefill spent.
    public var prefillGPUSec: Double
    /// Tokens a speculative round already committed but has not handed back
    /// yet. The round fed them to the model; the caller still has to see
    /// them, so they wait here instead of being drawn twice.
    public var pendingTokens: [Int]
    /// What `head.token` holds on the device. The gather reads it, so the
    /// session tracks it instead of re-reading the device every step.
    public var headToken: Int
    /// Whether the context-copy drafter has taken the wheel from the MTP.
    public var mtpActive: Bool
    /// Which drafter the session runs: "hybrid", "ngram" or the MTP itself.
    public var draftMode: QWDraftStrategy
    /// Maximum draft tokens per speculative verification round (`1...draftCapacity`).
    public var maxDrafts: Int
    /// Verify-window ceiling: the 8-slot GDN state ring holds the verified row plus
    /// up to seven drafts for dynamic wide context-copy (PLD) rounds.
    public static let draftCapacity = 7
    /// Default neural MTP draft ceiling on novel text (`1...defaultMTPDrafts`).
    public static let defaultMTPDrafts = 3
    /// Variational MTP relaxed acceptance log-logit margin in nats (`0` disables relaxation).
    public var relaxedMargin: Float
    /// Variational MTP relaxed acceptance floor: a non-argmax draft also needs `p_target(draft) >= relaxedMinProb` (`0` disables the floor).
    public var relaxedMinProb: Float
    /// Whether verified hidden states seed the MTP on all-accept rounds.
    public var quadraticSeeding: Bool
    /// Whether the prefix cache has been consulted for this prompt.
    public var restoreDone: Bool
    /// Whether the prefix cache served this prompt.
    public var prefixCacheHit: Bool
    /// How many tokens the restore reused.
    public var prefixCacheMatchedTokens: Int
    /// The tier that served the prefix: the RAM or the disk. Absent
    /// when no prefix was served.
    public var prefixCacheTier: FNPrefixCacheTier?
    /// The wall time, in milliseconds, the engine attributes to the token
    /// the last step produced. A speculative round splits its time across
    /// the tokens it committed, so no token is reported as free.
    public var stepMs: Double
    /// The GPU time, in milliseconds, of this session's last step. It
    /// belongs to the session: another session's step has its own.
    public var lastStepGPUMillis: Double
    /// Tokens committed per speculative verification round in this session.
    public var roundTokenCounts: [Int]
    /// Unique valid token IDs seen in `history` so far, maintained incrementally
    /// so repetition-penalty sampling is O(unique) instead of O(history.count) per step.
    public var seenTokens: Set<Int>
    /// Total speculative draft proposals in this session.
    public var specProposals: Int
    /// Total accepted speculative draft tokens in this session.
    public var specAccepted: Int
    /// Whether the prefix cache snapshot from prefill is pending persistence after first-token emission.
    public var pendingPrefixSave: Bool
    /// Whether per-token log-probabilities are collected for this session.
    public var collectLogprobs: Bool
    /// Number of top candidate logprobs (`0..20`) to collect per token when `collectLogprobs` is true.
    public var topLogprobsCount: Int
    /// Collected per-token log-probabilities in generation order.
    public var tokenLogprobs: [QWTokenLogprob]
    /// The request's token constraint (structured output). A constrained
    /// session never speculates: drafts are not grammar-verified.
    public var constraint: (any QWTokenConstraint)?
    /// Optional shared base prefix length (e.g. system prompt + tool definitions)
    /// to checkpoint into the prefix cache before branching into the session turn.
    public var retainPrefixLength: Int?
    /// Merged visual feature spans sorted by `tokenOffset` in the expanded prompt.
    public var visualSpans: [QWVisualSpan]

    public init(history: [Int64], sampler: QWSampler, useSpec: Bool = false,
                collectLogprobs: Bool = false, topLogprobsCount: Int = 0,
                constraint: (any QWTokenConstraint)? = nil,
                retainPrefixLength: Int? = nil,
                visualSpans: [QWVisualSpan] = []) {
        self.history = history
        self.sampler = sampler
        self.constraint = constraint
        self.retainPrefixLength = retainPrefixLength
        self.visualSpans = visualSpans.sorted { $0.tokenOffset < $1.tokenOffset }
        self.useSpec = useSpec && !collectLogprobs && (constraint?.allowsSpeculation ?? true)
        prefillPos = 0
        lastArgmax = 0
        lastRows = 1
        pleParity = 0
        plePrecomputed = false
        mtpDraft = 0
        mtpChain = []
        mtpChainPos = -1
        mtpSourcedDrafts = 0
        periodicDrafts = false
        chainPolicy = FNChainPolicy()
        firstToken = 0
        hasFirstToken = false
        prefillGPUSec = 0
        pendingTokens = []
        headToken = 0
        mtpActive = true
        draftMode = .hybrid
        maxDrafts = Self.defaultMTPDrafts
        relaxedMargin = 0
        relaxedMinProb = 0
        quadraticSeeding = true
        restoreDone = false
        prefixCacheHit = false
        prefixCacheMatchedTokens = 0
        prefixCacheTier = nil
        stepMs = 0
        lastStepGPUMillis = 0
        roundTokenCounts = []
        seenTokens = []
        specProposals = 0
        specAccepted = 0
        pendingPrefixSave = false
        self.collectLogprobs = collectLogprobs
        self.topLogprobsCount = topLogprobsCount
        self.tokenLogprobs = []
    }

    /// The number of tokens the session has consumed.
    public var activeTokens: Int { history.count }
}

// MARK: - The session's state wipe

/// Picks the MTP chain depth maximising expected committed tokens/ms: E(p) = 1 + a0 + a0·a1 (p ≥ 1) + a0·a1·a2 (p ≥ 2)
/// over cost(p) = verify(2 + p rows) + p chain passes. `acc[i]`: EMA acceptance of draft i given drafts 0..<i accepted;
/// `rowMs`: EMA round wall by verify rows. A periodic full-depth round keeps the deeper acceptances fresh.
struct FNChainPolicy {
    var acc: [Double] = [0.8, 0.7, 0.7]
    var rowMs: [Double] = [0, 15, 18, 22, 25, 29, 32, 35, 38]
    var passMs: Double = 1.3
    var rounds = 0
    static let alpha = 0.15
    static let exploreEvery = 16

    mutating func observeAcceptance(mtpDrafts: Int, accepted: Int) {
        for i in 0..<min(mtpDrafts, acc.count) where accepted >= i {
            acc[i] += Self.alpha * ((accepted > i ? 1 : 0) - acc[i])
        }
    }

    mutating func observeRound(rows: Int, ms: Double) {
        guard rows >= 1 && rows < rowMs.count && ms > 0 else { return }
        rowMs[rows] += Self.alpha * (ms - rowMs[rows])
    }

    mutating func observeChain(passes: Int, ms: Double) {
        guard passes > 0 && ms > 0 else { return }
        passMs += Self.alpha * (ms / Double(passes) - passMs)
    }

    mutating func passes(maxPasses: Int) -> Int {
        rounds += 1
        guard maxPasses > 0 else { return 0 }
        if rounds % Self.exploreEvery == 0 { return maxPasses }
        var best = 0, bestRate = 0.0, tokens = 1 + acc[0], reach = acc[0]
        for p in 0...maxPasses {
            if p > 0 { reach *= acc[min(p, acc.count - 1)]; tokens += reach }
            let rows = min(2 + p, rowMs.count - 1)
            let rate = tokens / (rowMs[rows] + Double(p) * passMs)
            if rate > bestRate { bestRate = rate; best = p }
        }
        return best
    }
}

enum FNMTPChain {
    /// Chained single-row MTP passes after each verify round (d2/d3 drafts); capped by maxDrafts - 1.
    static let passes = Int(ProcessInfo.processInfo.environment["QWG_MTP_CHAIN"] ?? "") ?? 2
    nonisolated(unsafe) static var ms = 0.0
    static let debug = ProcessInfo.processInfo.environment["QWG_CHAIN_DEBUG"] != nil
}

extension FNGPUEngine {
    /// Wipes the recurrent state this session used: the GDN states, the PLE
    /// history, the attention KV rows above the water mark and the MTP's
    /// positions. The wipe belongs to the session, not to the engine.
    func wipeSessionState(overwriteTokens: Int = 0) {
        let activeCap = activeCapacity
        let used = min(activeCap, highWaterPos)
        guard used > 0 else {
            residentKVHistory.removeAll(keepingCapacity: true)
            return
        }
        let skipIdx = min(used, max(0, overwriteTokens))
        let skipBlocks = skipIdx / 4
        DispatchQueue.concurrentPerform(iterations: layerRecords.count) { idx in
            let rec = layerRecords[idx]
            if let g = rec.gdn {
                memset(arena.cpuPointer(of: g.states[0]), 0, g.states[0].bytes)
                g.stateParity = 0
                memset(arena.cpuPointer(of: g.convState[0]), 0, g.convState[0].bytes)
                g.convParity = 0
            }
            if let p = rec.weights.ple {
                memset(arena.cpuPointer(of: p.history[0]), 0, p.history[0].bytes)
                p.parity = 0
            }
            if let a = rec.attnState {
                let cap = max(1, a.capacity)
                let u = min(cap, used), s = min(cap, skipIdx)
                if u > s {
                    let rowStride = a.indexK.bytes / cap
                    memset(arena.cpuPointer(of: a.indexK).advanced(by: s * rowStride), 0, (u - s) * rowStride)
                }
                let bStart = min(cap / 4, skipBlocks)
                let bEnd = (u + 3) / 4
                if bEnd > bStart {
                    let blockStride = a.pooled.bytes / max(1, cap / 4)
                    let startByte = min(a.pooled.bytes, bStart * blockStride)
                    let endByte = min(a.pooled.bytes, bEnd * blockStride)
                    if endByte > startByte {
                        memset(arena.cpuPointer(of: a.pooled).advanced(by: startByte), 0, endByte - startByte)
                    }
                }
            }
        }
        if skipIdx > 0 {
            mtp?.resetStateTail(from: max(0, skipIdx - 1), usedPositions: used)
        } else {
            mtp?.resetState(usedPositions: used)
        }
        highWaterPos = 0
        residentKVHistory.removeAll(keepingCapacity: true)
    }

    /// Zeroes only the attention indexK and pooled block tail from `startPos` up to `highWaterPos`,
    /// preserving the already-resident KV and index prefix in `0..<startPos`.
    func wipeSessionTail(from startPos: Int) {
        let activeCap = activeCapacity
        let used = min(activeCap, highWaterPos)
        let s = min(activeCap, max(0, startPos))
        let e = max(s, used)
        guard e > s || (e + 3) / 4 > s / 4 else { return }
        DispatchQueue.concurrentPerform(iterations: layerRecords.count) { idx in
            let rec = layerRecords[idx]
            if let a = rec.attnState {
                let cap = max(1, a.capacity)
                let se = min(cap, e), ss = min(cap, s)
                if se > ss {
                    let rowStride = a.indexK.bytes / cap
                    memset(arena.cpuPointer(of: a.indexK).advanced(by: ss * rowStride), 0, (se - ss) * rowStride)
                }
                let bStart = ss / 4
                let bEnd = (se + 3) / 4
                if bEnd > bStart {
                    let blockStride = a.pooled.bytes / max(1, cap / 4)
                    let startByte = min(a.pooled.bytes, bStart * blockStride)
                    let endByte = min(a.pooled.bytes, bEnd * blockStride)
                    if endByte > startByte {
                        memset(arena.cpuPointer(of: a.pooled).advanced(by: startByte), 0, endByte - startByte)
                    }
                }
            }
        }
        mtp?.resetStateTail(from: s, usedPositions: e)
    }

    /// Draws the first generated token once and keeps it: the prefill is
    /// chunked, and the token must not be drawn twice.
    func drawFirstToken(_ session: inout FNGPUSessionData) throws -> Int {
        session.hasFirstToken = true
        let tok = try sampleFromHead(argmaxToken: session.lastArgmax,
                                     seenTokens: session.seenTokens, sampler: session.sampler,
                                     constraint: session.constraint)
        session.firstToken = tok
        if session.collectLogprobs {
            recordTokenLogprob(&session, token: tok, row: 0)
        }
        reportFirstTokenLogits()
        return tok
    }

    /// Records the full-vocabulary log-probability of `token` from row `row` of `head.logits`.
    func recordTokenLogprob(_ session: inout FNGPUSessionData, token: Int, row: Int = 0) {
        let n = vocab
        let rawLogits = arena.cpuPointer(of: head.logits)
            .assumingMemoryBound(to: Float.self)
            .advanced(by: row * n)
        let rowBuf = UnsafeBufferPointer(start: rawLogits, count: n)
        let lp = session.sampler.computeTokenLogprob(
            logits: rowBuf,
            chosenToken: token,
            topK: session.topLogprobsCount
        )
        session.tokenLogprobs.append(lp)
    }

    /// Prints the top-10 logits of the first generated token when the
    /// diagnostic is on. It is a debugging aid for greedy divergence.
    func reportFirstTokenLogits() {
        guard ProcessInfo.processInfo.environment["QWG_DEBUG_TOPK"] != nil else { return }
        let logits = head.readLogits()
        let top = logits.indices.sorted { logits[$0] > logits[$1] }.prefix(10)
        let line = top.map { "\($0):\(String(format: "%.4f", logits[$0]))" }.joined(separator: " ")
        FileHandle.standardError.write(("TOPK " + line + "\n").data(using: .utf8)!)
    }

    /// Samples one token from the head's logit row with the SHARED sampler.
    /// Both families go through `QWSampler`, so the same seed and the same
    /// logits pick the same token on either engine.
    func sampleFromHead(argmaxToken: Int, seenTokens: Set<Int>,
                        sampler: QWSampler, row: Int = 0,
                        constraint: (any QWTokenConstraint)? = nil) throws -> Int {
        guard let constraint else {
            return try unconstrainedSample(argmaxToken: argmaxToken, seenTokens: seenTokens,
                                           sampler: sampler, row: row)
        }
        let token: Int
        if let allowed = try constraint.allowedTokens() {
            if sampler.config.isGreedy && sampler.config.repetitionPenalty == 1.0 && allowed.contains(argmaxToken) {
                token = argmaxToken
            } else {
                let rawLogits = arena.cpuPointer(of: head.logits)
                    .assumingMemoryBound(to: Float.self)
                    .advanced(by: row * vocab)
                token = try sampler.constrainedToken(
                    logits: UnsafeBufferPointer(start: rawLogits, count: vocab),
                    allowed: allowed,
                    repetitionHistory: sampler.config.repetitionPenalty == 1.0 ? [] : seenTokens)
            }
        } else {
            token = try unconstrainedSample(argmaxToken: argmaxToken, seenTokens: seenTokens,
                                            sampler: sampler, row: row)
        }
        try constraint.accept(token)
        return token
    }

    private func unconstrainedSample(argmaxToken: Int, seenTokens: Set<Int>,
                                     sampler: QWSampler, row: Int) throws -> Int {
        let config = sampler.config
        if config.isGreedy && config.repetitionPenalty == 1.0 {
            return argmaxToken
        }
        let n = vocab
        let rawLogits = arena.cpuPointer(of: head.logits)
            .assumingMemoryBound(to: Float.self)
            .advanced(by: row * n)
        let rowBuf = UnsafeBufferPointer(start: rawLogits, count: n)
        if config.isGreedy {
            return try sampler.penalisedGreedyToken(logits: rowBuf, repetitionHistory: seenTokens)
        }
        return try sampler.sampleToken(
            logits: rowBuf,
            repetitionHistory: config.repetitionPenalty == 1.0 ? [] : Array(seenTokens)
        )
    }

    /// One speculative round: the token is fed, 1..3 drafts are proposed, and the
    /// model verifies all rows in one pass, accepting the longest matching prefix
    /// and rolling back any rejected tail in O(1) via 4-slot ring buffer parity subtraction.
    func speculativeStep(_ session: inout FNGPUSessionData, token: Int,
                         vStack: FNGPUDecodeStack, mtp: FNGPUMTP) throws -> Int {
        let prevCount = session.history.count
        let trackRep = session.sampler.config.repetitionPenalty != 1.0
        session.history.append(Int64(token))
        if trackRep && token >= 0 && token < vocab { session.seenTokens.insert(token) }
        let (drafts, stepMTP) = draftsFor(&session, mtp: mtp)
        let dCount = drafts.count
        let activeRows = 1 + dCount
        for d in drafts { session.history.append(Int64(d)) }
        if residentKVHistory.count == prevCount {
            residentKVHistory.append(Int64(token))
            for d in drafts { residentKVHistory.append(Int64(d)) }
        }
        highWaterPos = max(highWaterPos, session.history.count)
        let stepT0 = Date()
        _ = try vStack.forwardRows(history: session.history, embedTable: embedTable, head: head,
                                    readStream: false, verifyAllRows: true, mtp: stepMTP,
                                    activeRows: activeRows, relaxedMargin: session.relaxedMargin,
                                    relaxedRows: session.mtpSourcedDrafts + 1,
                                    relaxedMinProb: session.relaxedMinProb)
        let verifyMs = Date().timeIntervalSince(stepT0) * 1000.0
        session.chainPolicy.observeRound(rows: activeRows, ms: verifyMs)
        let v = head.argmaxTokens(count: activeRows)
        let mTok = stepMTP?.mtpHead.argmaxToken() ?? session.mtpDraft
        session.specProposals += dCount
        var tok0 = 0
        var accepted = 0
        for r in 0..<dCount {
            let draftTok = drafts[r]
            let sampled: Int
            if v[r] == draftTok && (session.constraint?.isUnconstrainedNow ?? true) {
                try session.constraint?.accept(draftTok)
                sampled = draftTok
            } else {
                sampled = try sampleFromHead(argmaxToken: v[r], seenTokens: session.seenTokens,
                                             sampler: session.sampler, row: r,
                                             constraint: session.constraint)
            }
            if r == 0 {
                tok0 = sampled
            } else {
                session.pendingTokens.append(sampled)
            }
            if sampled == draftTok && v[r] == draftTok {
                accepted += 1
                if trackRep && draftTok >= 0 && draftTok < vocab { session.seenTokens.insert(draftTok) }
            } else {
                if trackRep && sampled >= 0 && sampled < vocab { session.seenTokens.insert(sampled) }
                break
            }
        }
        if accepted == dCount {
            let bonusTok = try sampleFromHead(argmaxToken: v[dCount], seenTokens: session.seenTokens,
                                              sampler: session.sampler, row: dCount,
                                              constraint: session.constraint)
            if trackRep && bonusTok >= 0 && bonusTok < vocab { session.seenTokens.insert(bonusTok) }
            session.pendingTokens.append(bonusTok)
        } else {
            let rejected = dCount - accepted
            session.history.removeLast(rejected)
            if residentKVHistory.count == prevCount + activeRows {
                residentKVHistory.removeLast(rejected)
            }
            vStack.rollbackVerifyRows(rejectedCount: rejected)
            session.mtpActive = true
        }
        session.specAccepted += accepted
        session.chainPolicy.observeAcceptance(mtpDrafts: session.mtpSourcedDrafts, accepted: accepted)
        session.mtpDraft = mTok
        session.mtpChain = []
        var chainGPUMs = 0.0
        if let stepMTP {
            var sel = 0
            while sel < dCount && v[sel] == drafts[sel] { sel += 1 }
            let lastTok = session.pendingTokens.last ?? tok0
            let mtpMax = min(FNGPUSessionData.defaultMTPDrafts, session.maxDrafts)
            let passes = (FNMTPChain.passes > 0 && mtpMax > 1 && session.mtpActive && !session.periodicDrafts)
                ? session.chainPolicy.passes(maxPasses: min(FNMTPChain.passes, mtpMax - 1))
                : 0
            let needsResync = sel != accepted || lastTok != v[accepted]
            if passes > 0 || needsResync {
                let chainT0 = Date()
                guard let pair = ctx.hal.begin(.concurrent) else {
                    throw FNGPUError.notHQ4("chain command buffer")
                }
                let cb = pair.0
                var enc = pair.1
                let firstStream: QWTensorValue
                if !needsResync {
                    firstStream = stepMTP.lastProjectedStream(verifyRows: activeRows)
                } else {
                    stepMTP.writeTokens([Int32(lastTok)])
                    let selStream = vStack.lastStreamTensor.slice(
                        byteOffset: accepted * vStack.width * 4,
                        count: vStack.width,
                        dtype: .f32
                    )
                    try stepMTP.encode(
                        &enc,
                        cb: cb,
                        streamPrev: selStream,
                        tokens: stepMTP.tokensBuf.slice(byteOffset: 0, count: 1, dtype: .i32),
                        pos: prevCount + accepted,
                        rows: 1,
                        projectRows: 1
                    )
                    enc.memoryBarrier(scope: .buffers)
                    firstStream = stepMTP.lastProjectedStream(verifyRows: 1)
                }
                if passes > 0 {
                    try stepMTP.encodeChain(&enc, cb: cb, firstStream: firstStream,
                                            startPos: prevCount + accepted + 1, passes: passes)
                }
                ctx.hal.end(cb, enc)
                cb.waitUntilCompleted()
                if let err = cb.error { throw err }
                chainGPUMs = max(0, (cb.gpuEndTime - cb.gpuStartTime) * 1000.0)
                if passes > 0 {
                    let chain = stepMTP.readChain(count: passes + 1)
                    session.mtpDraft = chain[0]
                    session.mtpChain = chain
                    session.mtpChainPos = prevCount + accepted + 2
                } else {
                    session.mtpDraft = stepMTP.mtpHead.argmaxToken()
                }
                let chainMs = Date().timeIntervalSince(chainT0) * 1000
                if passes > 0 {
                    session.chainPolicy.observeChain(passes: passes, ms: chainMs)
                }
                FNMTPChain.ms += chainMs
            }
        }
        session.headToken = -1
        let roundMs = Date().timeIntervalSince(stepT0) * 1000.0
        let divisor = Double(1 + session.pendingTokens.count)
        session.stepMs = roundMs / divisor
        session.lastStepGPUMillis = (vStack.lastGPUMillis + chainGPUMs) / divisor
        session.roundTokenCounts.append(1)
        if FNMTPChain.debug {
            FileHandle.standardError.write("R rows=\(activeRows) acc=\(accepted) gpu=\(String(format: "%.2f", vStack.lastGPUMillis)) wall=\(String(format: "%.2f", roundMs)) chain=\(String(format: "%.2f", FNMTPChain.ms))\n".data(using: .utf8)!)
            FNMTPChain.ms = 0
        }
        return tok0
    }

    /// Drafts 1..3 tokens using MTP or the unified context-copy drafter (`QWContextCopyDrafter`).
    func draftsFor(_ session: inout FNGPUSessionData, mtp: FNGPUMTP) -> ([Int], FNGPUMTP?) {
        let proposal = QWContextCopyDrafter.proposeDrafts(
            history: session.history,
            mode: session.draftMode,
            maxDrafts: session.maxDrafts,
            draftCapacity: FNGPUSessionData.draftCapacity,
            mtpDraft: session.mtpDraft,
            mtpChain: session.mtpChain,
            mtpChainPos: session.mtpChainPos,
            mtpActive: session.mtpActive,
            relaxedMargin: session.relaxedMargin,
            quadraticSeeding: session.quadraticSeeding
        )
        session.mtpActive = proposal.mtpActive
        session.mtpSourcedDrafts = proposal.mtpSourcedDrafts
        session.periodicDrafts = proposal.periodicDrafts
        return (proposal.drafts, proposal.runMTPOnVerify ? mtp : nil)
    }
}
