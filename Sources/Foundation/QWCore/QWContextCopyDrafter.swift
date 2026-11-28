// QWContextCopyDrafter.swift — Zero-copy suffix-matching & periodic drafter (PLD)
// shared across all model families for .ngram and .hybrid speculative decoding.

import Foundation

private struct VirtualIntSequence<T: FixedWidthInteger> {
    let base: UnsafeBufferPointer<T>
    let extraFirst: T?
    let trailing: UnsafeBufferPointer<Int>

    var count: Int {
        base.count + (extraFirst != nil ? 1 : 0) + trailing.count
    }

    @inline(__always)
    subscript(index: Int) -> Int {
        let baseCount = base.count
        if index < baseCount { return Int(base[index]) }
        let rem = index - baseCount
        if let extraFirst {
            return rem == 0 ? Int(extraFirst) : trailing[rem - 1]
        }
        return trailing[rem]
    }
}

public enum QWContextCopyDrafter {
    /// Draft proposal result from the unified hybrid/ngram/mtp draft selector.
    public struct Proposal: Sendable, Equatable {
        public let drafts: [Int]
        public let mtpActive: Bool
        public let mtpSourcedDrafts: Int
        public let periodicDrafts: Bool
        public let runMTPOnVerify: Bool

        public init(drafts: [Int], mtpActive: Bool, mtpSourcedDrafts: Int,
                    periodicDrafts: Bool, runMTPOnVerify: Bool) {
            self.drafts = drafts
            self.mtpActive = mtpActive
            self.mtpSourcedDrafts = mtpSourcedDrafts
            self.periodicDrafts = periodicDrafts
            self.runMTPOnVerify = runMTPOnVerify
        }
    }

    /// Returns up to `maxCount` next tokens when the tail of `history` is a repeating periodic cycle of
    /// period `1...maxPeriod` repeated at least `minRepeats` times.
    public static func periodicDrafts(history: [Int64], maxCount: Int = 3,
                                      maxPeriod: Int = 8, minRepeats: Int = 2) -> [Int]? {
        history.withUnsafeBufferPointer { base in
            [].withUnsafeBufferPointer { empty in
                periodicCore(VirtualIntSequence(base: base, extraFirst: nil, trailing: empty),
                             maxCount: maxCount, maxPeriod: maxPeriod, minRepeats: minRepeats)
            }
        }
    }

    public static func periodicDrafts(history: [Int], maxCount: Int = 3,
                                      maxPeriod: Int = 8, minRepeats: Int = 2) -> [Int]? {
        periodicDrafts(history: history, extraToken: nil,
                       maxCount: maxCount, maxPeriod: maxPeriod, minRepeats: minRepeats)
    }

    public static func periodicDrafts(history: [Int], extraToken: Int?, maxCount: Int = 3,
                                      maxPeriod: Int = 8, minRepeats: Int = 2) -> [Int]? {
        history.withUnsafeBufferPointer { base in
            [].withUnsafeBufferPointer { empty in
                periodicCore(VirtualIntSequence(base: base, extraFirst: extraToken, trailing: empty),
                             maxCount: maxCount, maxPeriod: maxPeriod, minRepeats: minRepeats)
            }
        }
    }

    /// Single-pass suffix matcher returning up to `maxCount` continuation tokens and the matched suffix length.
    /// When `appendedToken` is supplied, matches against `history + [appendedToken]` without copying `history`.
    public static func drafts(history: [Int64], appendedToken: Int64? = nil, maxCount: Int = 3,
                              minMatchLen: Int = 3, maxMatchLen: Int = 20) -> (tokens: [Int], matchLen: Int)? {
        history.withUnsafeBufferPointer { base in
            [].withUnsafeBufferPointer { empty in
                draftsCore(VirtualIntSequence(base: base, extraFirst: appendedToken, trailing: empty),
                           maxCount: maxCount, minMatchLen: minMatchLen, maxMatchLen: maxMatchLen)
            }
        }
    }

    public static func drafts(history: [Int], appendedToken: Int? = nil, maxCount: Int = 3,
                              minMatchLen: Int = 3, maxMatchLen: Int = 20) -> (tokens: [Int], matchLen: Int)? {
        drafts(history: history, extraToken: appendedToken, trailingTokens: [],
               maxCount: maxCount, minMatchLen: minMatchLen, maxMatchLen: maxMatchLen)
    }

    public static func drafts(history: [Int], extraToken: Int? = nil, trailingTokens: [Int] = [],
                              maxCount: Int = 3, minMatchLen: Int = 3,
                              maxMatchLen: Int = 20) -> (tokens: [Int], matchLen: Int)? {
        history.withUnsafeBufferPointer { base in
            trailingTokens.withUnsafeBufferPointer { trail in
                draftsCore(VirtualIntSequence(base: base, extraFirst: extraToken, trailing: trail),
                           maxCount: maxCount, minMatchLen: minMatchLen, maxMatchLen: maxMatchLen)
            }
        }
    }

    public static func draft(history: [Int64], minMatchLen: Int = 6, maxMatchLen: Int = 20) -> Int? {
        drafts(history: history, maxCount: 1, minMatchLen: minMatchLen, maxMatchLen: maxMatchLen)?.tokens.first
    }

    private static func periodicCore<T: FixedWidthInteger>(
        _ seq: VirtualIntSequence<T>, maxCount: Int, maxPeriod: Int, minRepeats: Int
    ) -> [Int]? {
        let count = seq.count
        guard maxCount > 0, maxPeriod >= 1, minRepeats >= 1 else { return nil }
        let upperPeriod = min(maxPeriod, count / minRepeats)
        guard upperPeriod >= 1 else { return nil }
        for p in 1...upperPeriod {
            let needed = p * minRepeats
            guard count >= needed else { break }
            var match = true
            let base = count - p
            for r in 1..<minRepeats {
                let off = count - (r + 1) * p
                for i in 0..<p where seq[off + i] != seq[base + i] {
                    match = false
                    break
                }
                if !match { break }
            }
            if match {
                var out: [Int] = []
                out.reserveCapacity(maxCount)
                for idx in 0..<maxCount {
                    out.append(seq[base + (idx % p)])
                }
                return out
            }
        }
        return nil
    }

    private static func draftsCore<T: FixedWidthInteger>(
        _ seq: VirtualIntSequence<T>, maxCount: Int, minMatchLen: Int, maxMatchLen: Int
    ) -> (tokens: [Int], matchLen: Int)? {
        let count = seq.count
        guard maxCount > 0, minMatchLen >= 1, maxMatchLen >= minMatchLen, count > minMatchLen else { return nil }
        let maxN = min(maxMatchLen, count - 1)
        guard maxN >= minMatchLen else { return nil }
        let lastTok = seq[count - 1]
        let minEnd = max(minMatchLen - 1, count - 1 - 32768)
        var bestLen = 0
        var bestEnd = -1
        for endPos in stride(from: count - 2, through: minEnd, by: -1) {
            guard seq[endPos] == lastTok else { continue }
            let limit = min(maxN, endPos + 1)
            if limit <= bestLen { break }
            var m = 1
            while m < limit && seq[endPos - m] == seq[count - 1 - m] {
                m += 1
            }
            if m >= minMatchLen && m > bestLen {
                bestLen = m
                bestEnd = endPos
                if bestLen == maxN { break }
            }
        }
        guard bestLen >= minMatchLen, bestEnd >= 0 else { return nil }
        let avail = min(maxCount, count - 1 - bestEnd)
        guard avail > 0 else { return nil }
        if count > 2048 && bestLen <= 3 {
            let nextTok = seq[bestEnd + 1]
            let checkMin = max(bestLen - 1, bestEnd - 512)
            var confirmed = false
            if bestEnd > checkMin {
                for prevEnd in stride(from: bestEnd - 1, through: checkMin, by: -1) {
                    guard seq[prevEnd] == lastTok else { continue }
                    var same = true
                    for m in 1..<bestLen where seq[prevEnd - m] != seq[count - 1 - m] {
                        same = false
                        break
                    }
                    if same {
                        if seq[prevEnd + 1] != nextTok { return nil }
                        confirmed = true
                        break
                    }
                }
            }
            if bestLen == 2 && !confirmed { return nil }
        }
        var out: [Int] = []
        out.reserveCapacity(avail)
        for k in 1...avail {
            out.append(seq[bestEnd + k])
        }
        return (tokens: out, matchLen: bestLen)
    }

    /// Selects 1..draftCapacity speculative tokens combining periodic cycles, suffix PLD, and neural MTP drafts.
    public static func proposeDrafts(
        history: [Int64],
        mode: QWDraftStrategy,
        maxDrafts: Int,
        draftCapacity: Int = 7,
        mtpDraft: Int,
        mtpChain: [Int],
        mtpChainPos: Int,
        mtpActive: Bool,
        relaxedMargin: Float,
        quadraticSeeding: Bool
    ) -> Proposal {
        let maxD = max(1, min(draftCapacity, maxDrafts))
        let pldMax = maxDrafts >= 3 ? max(maxD, draftCapacity) : maxD
        if mode == .ngram {
            if let match = drafts(history: history, maxCount: pldMax, minMatchLen: 3) {
                let take = match.matchLen >= 6 ? pldMax : (match.matchLen >= 5 ? maxD : 1)
                return Proposal(drafts: Array(match.tokens.prefix(take)), mtpActive: mtpActive,
                                mtpSourcedDrafts: 0, periodicDrafts: false, runMTPOnVerify: false)
            }
            return Proposal(drafts: [mtpDraft], mtpActive: mtpActive,
                            mtpSourcedDrafts: 0, periodicDrafts: false, runMTPOnVerify: false)
        }
        if mode == .mtp {
            return Proposal(drafts: [mtpDraft], mtpActive: true,
                            mtpSourcedDrafts: 1, periodicDrafts: false, runMTPOnVerify: true)
        }
        if let pd = periodicDrafts(history: history, maxCount: pldMax) {
            return Proposal(drafts: pd, mtpActive: false,
                            mtpSourcedDrafts: 0, periodicDrafts: true, runMTPOnVerify: true)
        }
        let isVariational = relaxedMargin > 0
        let isLongContext = history.count > 2048
        let isUltraLongContext = history.count > 98_304
        let minLen = isVariational ? (isUltraLongContext ? 4 : 3) : (mtpActive ? 4 : 3)
        if let match = drafts(history: history, maxCount: pldMax, minMatchLen: minLen),
           (!isLongContext || match.tokens.first == mtpDraft || match.matchLen >= 5 || !mtpActive) {
            let take: Int
            if (!mtpActive && match.matchLen >= 4) || match.matchLen >= 6 {
                take = pldMax
            } else if isVariational && match.matchLen >= minLen {
                take = maxD
            } else {
                take = 1
            }
            return Proposal(drafts: Array(match.tokens.prefix(take)), mtpActive: false,
                            mtpSourcedDrafts: 0, periodicDrafts: false, runMTPOnVerify: true)
        }
        if maxD > 1, mtpDraft >= 0 {
            let chainMinLen = isUltraLongContext
                ? 4
                : ((isVariational && quadraticSeeding) ? 2 : (isVariational ? 3 : 4))
            if let cont = drafts(history: history, appendedToken: Int64(mtpDraft),
                                 maxCount: max(1, pldMax - 1), minMatchLen: chainMinLen) {
                let extra: Int
                if cont.matchLen >= 6 {
                    extra = max(1, pldMax - 1)
                } else if cont.matchLen >= (isUltraLongContext ? 5 : 3) {
                    extra = maxD - 1
                } else {
                    extra = 1
                }
                var out = [mtpDraft]
                out.append(contentsOf: cont.tokens.prefix(extra))
                return Proposal(drafts: out, mtpActive: mtpActive,
                                mtpSourcedDrafts: 1, periodicDrafts: false, runMTPOnVerify: true)
            }
        }
        if maxD > 1, mtpChain.count > 1, mtpChainPos == history.count, mtpChain[0] == mtpDraft {
            let chain = Array(mtpChain.prefix(maxD))
            return Proposal(drafts: chain, mtpActive: mtpActive,
                            mtpSourcedDrafts: chain.count, periodicDrafts: false, runMTPOnVerify: true)
        }
        return Proposal(drafts: [mtpDraft], mtpActive: mtpActive,
                        mtpSourcedDrafts: 1, periodicDrafts: false, runMTPOnVerify: true)
    }

    /// Selects hybrid periodic + suffix PLD + recycler drafts for DFlash bypass rounds.
    public static func selectHybridPLDDrafts(
        history: [Int64],
        extraToken: Int64? = nil,
        previousToken: Int? = nil,
        pendingToken: Int,
        mode: QWDraftStrategy,
        maxDrafts: Int,
        recycler: QWTokenRecycler
    ) -> [Int] {
        selectHybridPLDDraftsCore(
            history: history, extraToken: extraToken, previousToken: previousToken,
            pendingToken: pendingToken, mode: mode, maxDrafts: maxDrafts, recycler: recycler
        )
    }

    public static func selectHybridPLDDrafts(
        history: [Int],
        extraToken: Int? = nil,
        previousToken: Int? = nil,
        pendingToken: Int,
        mode: QWDraftStrategy,
        maxDrafts: Int,
        recycler: QWTokenRecycler
    ) -> [Int] {
        selectHybridPLDDraftsCore(
            history: history, extraToken: extraToken, previousToken: previousToken,
            pendingToken: pendingToken, mode: mode, maxDrafts: maxDrafts, recycler: recycler
        )
    }

    private static func selectHybridPLDDraftsCore<T: FixedWidthInteger>(
        history: [T],
        extraToken: T?,
        previousToken: Int?,
        pendingToken: Int,
        mode: QWDraftStrategy,
        maxDrafts: Int,
        recycler: QWTokenRecycler
    ) -> [Int] {
        guard maxDrafts > 0, mode != .mtp else { return [] }
        let minPld = (mode == .ngram) ? min(4, maxDrafts) : min(7, maxDrafts)
        let periodic = history.withUnsafeBufferPointer { base in
            [].withUnsafeBufferPointer { empty in
                periodicCore(VirtualIntSequence(base: base, extraFirst: extraToken, trailing: empty),
                             maxCount: maxDrafts, maxPeriod: 8, minRepeats: 3)
            }
        }
        if let pld = periodic, pld.count >= minPld {
            return Array(pld.prefix(maxDrafts))
        }
        var drafts: [Int] = []
        let match = history.withUnsafeBufferPointer { base in
            [].withUnsafeBufferPointer { empty in
                draftsCore(VirtualIntSequence(base: base, extraFirst: extraToken, trailing: empty),
                           maxCount: maxDrafts, minMatchLen: 6, maxMatchLen: 20)
            }
        }
        if let match, !match.tokens.isEmpty {
            drafts = Array(match.tokens.prefix(maxDrafts))
        }
        if !drafts.isEmpty || mode == .ngram {
            recycler.appendDrafts(
                into: &drafts, maxCount: maxDrafts,
                previousToken: previousToken, pendingToken: pendingToken,
                maxUnigramFallbacks: 0
            )
            if drafts.count >= minPld { return drafts }
        }
        return []
    }
}
