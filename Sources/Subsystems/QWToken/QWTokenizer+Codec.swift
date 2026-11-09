// QWTokenizer codec — encode (NFC → specials-split → regex words →
// ByteLevel → BPE merges), decode (piece join + inverse byte map), and the
// incremental (streaming) decoder with UTF-8 partial buffering.
// Every rule is bound by the 10k goldens; the String-== Unicode-canonical
// trap is handled by doing ALL merge/special matching on UTF-8 BYTES and
// keeping vocab symbol strings inside the byte-mapped range (no scalar in
// 32…126 ∪ 160…255 has a canonical composition with another member — ccc 0
import Foundation
import QWCore

/// Module marker (was Placeholder.swift).

extension QWTokenizer {
    func piece(_ id: Int) -> String? { idToPiece[id] }

    public func tokenId(for token: String) -> Int? {
        vocab[token] ?? added.first(where: { $0.s == token })?.id
    }

    public func resolveSpecialTokens(fallback: QWModelSpecialTokens) -> QWModelSpecialTokens {
        let imStart = tokenId(for: "<|im_start|>") ?? fallback.imStartTokenId
        let imEnd = tokenId(for: "<|im_end|>") ?? fallback.imEndTokenId
        let think = tokenId(for: "<think>") ?? fallback.thinkTokenId
        let endThink = tokenId(for: "</think>") ?? fallback.endThinkTokenId
        var extra = fallback.extraStopTokenIds
        for s in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"] {
            if let id = tokenId(for: s), !extra.contains(id) {
                extra.append(id)
            }
        }
        let hasChatIds = fallback.usesChatTemplateIds || tokenId(for: "<|im_end|>") != nil
        return QWModelSpecialTokens(
            eosTokenId: fallback.eosTokenId,
            imStartTokenId: imStart,
            imEndTokenId: imEnd,
            thinkTokenId: think,
            endThinkTokenId: endThink,
            extraStopTokenIds: extra,
            usesChatTemplateIds: hasChatIds
        )
    }

    // MARK: BPE one word (already ByteLevel-mapped) → ids
    /// Merge law (engine parity): min-rank adjacent pair wins, leftmost on
    /// ties (strict `<` scan), ALL occurrences of that pair merge in one
    /// left-to-right pass; repeat until no ranked pair remains.
    func bpeWord(_ bytesSlice: ArraySlice<UInt8>) -> [Int] {
        guard !bytesSlice.isEmpty else { return [] }
        var word: [UInt32] = []
        word.reserveCapacity(bytesSlice.count)
        for b in bytesSlice {
            word.append(byteToSymId[Int(b)])
        }
        return bpeMerge(word)
    }

    func bpeMerge(_ initialWord: [UInt32]) -> [Int] {
        var word = initialWord
        guard !word.isEmpty else { return [] }
        if word.count == 1 {
            return [symTok[word[0]] ?? -1]
        }
        while word.count > 1 {
            var bestKey = UInt64.max
            var bestRank = Int.max
            let count = word.count
            for i in 0..<(count - 1) {
                let k = UInt64(word[i]) << 20 | UInt64(word[i + 1])
                if let r = ranks[k], r < bestRank {
                    bestRank = r
                    bestKey = k
                }
            }
            guard bestKey != UInt64.max, let merged = pairMerged[bestKey] else { break }
            let a = UInt32(bestKey >> 20), b = UInt32(bestKey & 0xFFFFF)
            var w = 0
            var r = 0
            while r < word.count {
                if r + 1 < word.count && word[r] == a && word[r + 1] == b {
                    word[w] = merged
                    r += 2
                } else {
                    word[w] = word[r]
                    r += 1
                }
                w += 1
            }
            word.removeSubrange(w..<word.count)
        }
        return word.map { symTok[$0] ?? -1 }
    }

    // MARK: encode
    /// tokenizers parity: addSpecialTokens is accepted but a no-op — this
    /// checkpoint's post-processor adds nothing; pinned by goldens
    public func encode(_ text: String, addSpecialTokens: Bool = false) throws -> [Int] {
        _ = addSpecialTokens   // no-op law (see above)
        let nfc = usesNFC ? text.precomposedStringWithCanonicalMapping : text
        var ids: [Int] = []
        // specials split on UTF-8 BYTES (String searching is canonical-
        // comparing — the trap): locate byte ranges, greedy leftmost-longest.
        let bytes = Array(nfc.utf8)
        let nBytes = bytes.count
        var matches: [(loc: Int, len: Int, ord: Int, id: Int)] = []

        var bi = 0
        while bi < nBytes {
            while bi < nBytes && !addedFirstByte[Int(bytes[bi])] { bi += 1 }
            guard bi < nBytes else { break }
            var matched = false
            for p in addedPats {
                if bi + p.pat.count <= nBytes && bytes[bi..<(bi + p.pat.count)].elementsEqual(p.pat) {
                    matches.append((bi, p.pat.count, p.ord, p.id))
                    bi += p.pat.count
                    matched = true
                    break
                }
            }
            if !matched {
                bi += 1
            }
        }

        func encodeChunk(_ cLo: Int, _ cHi: Int) throws {
            guard cHi > cLo else { return }
            if case .metaspaceByteFallback(let rep, let scheme, _) = mode {
                var seg = String(decoding: bytes[cLo..<cHi], as: UTF8.self)
                let startsWithSpace = seg.hasPrefix(" ")
                seg = seg.replacingOccurrences(of: " ", with: rep)
                if (scheme == "always" || (scheme == "first" && cLo == 0)) && !startsWithSpace {
                    seg = rep + seg
                }
                var syms: [UInt32] = []
                syms.reserveCapacity(seg.unicodeScalars.count)
                for sc in seg.unicodeScalars {
                    let s = String(sc)
                    if let sid = symId[s] {
                        syms.append(sid)
                    } else {
                        for b in s.utf8 { syms.append(byteToSymId[Int(b)]) }
                    }
                }
                for g in bpeMerge(syms) {
                    if g < 0 { throw Error.unresolvedByte(String(seg.prefix(16))) }
                    ids.append(g)
                }
                return
            }
            let seg = String(decoding: bytes[cLo..<cHi], as: UTF8.self)
            let ns = seg as NSString
            let u2b = QWTokenizer.utf16ToByteTable(seg)
            var ranges: [NSRange] = []
            split.enumerateMatches(in: seg,
                                   range: NSRange(location: 0, length: ns.length)) { mr, _, _ in
                guard let r = mr?.range, r.length > 0 else { return }
                ranges.append(r)
            }
            var cursor = 0
            for r in ranges {
                let b0 = u2b[r.location], b1 = u2b[r.location + r.length]
                if b0 > cursor || cursor > b1 {
                    throw Error.unresolvedByte(String(String(
                        decoding: bytes[cLo + cursor ..< max(cLo + cursor, cLo + b0)],
                        as: UTF8.self).prefix(16)))
                }
                let got = bpeWord(bytes[(cLo + b0)..<(cLo + b1)])
                if got.isEmpty {
                    throw Error.unresolvedByte(String(String(
                        decoding: bytes[(cLo + b0)..<(cLo + b1)], as: UTF8.self).prefix(16)))
                }
                for g in got {
                    if g < 0 {
                        throw Error.unresolvedByte(String(String(
                            decoding: bytes[(cLo + b0)..<(cLo + b1)], as: UTF8.self).prefix(16)))
                    }
                    ids.append(g)
                }
                cursor = max(cursor, b1)
            }
            if cursor < cHi - cLo {
                throw Error.unresolvedByte(String(String(
                    decoding: bytes[cLo + cursor ..< cHi], as: UTF8.self).prefix(16)))
            }
        }

        func encodeSegment(_ lo: Int, _ hi: Int) throws {
            guard hi > lo else { return }
            let chunkSize = 16384
            if hi - lo <= chunkSize {
                try encodeChunk(lo, hi)
                return
            }
            var cur = lo
            while cur < hi {
                var nextEnd = min(cur + chunkSize, hi)
                if nextEnd < hi {
                    var scan = nextEnd
                    while scan > cur && bytes[scan - 1] != 0x0A {
                        scan -= 1
                    }
                    if scan > cur {
                        nextEnd = scan
                    }
                }
                try encodeChunk(cur, nextEnd)
                cur = nextEnd
            }
        }
        var pos = 0
        for m in matches {
            if m.loc < pos { continue }              // overlapped by longer earlier match
            if m.loc > pos { try encodeSegment(pos, m.loc) }
            ids.append(m.id); pos = m.loc + m.len
        }
        if pos < bytes.count { try encodeSegment(pos, bytes.count) }
        return ids
    }

    /// utf16-unit → byte-offset table (one entry per unit, plus a sentinel
    /// for the end position). Match ends always land on scalar boundaries,
    /// so per-scalar replication is exact.
    static func utf16ToByteTable(_ s: String) -> [Int] {
        var table = [Int]()
        table.reserveCapacity(s.utf16.count + 1)
        var b = 0
        for sc in s.unicodeScalars {
            let w = qwUTF8Width(of: sc)
            for _ in 0..<sc.utf16.count { table.append(b) }
            b += w
        }
        table.append(b)
        return table
    }

    /// The exact output bytes of every token id when decoded mid-sequence, indexed by id.
    /// Special ids and ids without a piece are nil; metaspace first-piece stripping is not modelled.
    public func tokenByteTable() -> [[UInt8]?] {
        let count = (idToPiece.keys.max() ?? -1) + 1
        var table = [[UInt8]?](repeating: nil, count: max(0, count))
        for (id, piece) in idToPiece where !specialIds.contains(id) {
            var bytes: [UInt8] = []
            switch mode {
            case .byteLevel:
                for sc in piece.unicodeScalars {
                    if let b = byteUnmap[sc] {
                        bytes.append(b)
                    } else {
                        bytes.append(contentsOf: String(sc).utf8)
                    }
                }
            case .metaspaceByteFallback(let rep, _, _):
                if let fb = byteFallbackUnmap[id] {
                    bytes.append(fb)
                } else {
                    bytes.append(contentsOf: piece.replacingOccurrences(of: rep, with: " ").utf8)
                }
            }
            table[id] = bytes
        }
        return table
    }

    // MARK: decode
    /// Rust-equivalent (BPE::decode + ByteLevel post-processor): every id's
    /// VOCAB PIECE joins to one string; the joined string is mapped
    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) throws -> String {
        var out: [UInt8] = []
        var sawFirstPiece = false
        for id in ids {
            guard let piece = idToPiece[id] else {
                throw Error.unresolvedByte("id \(id) (no vocab piece at all)")
            }
            if specialIds.contains(id) {
                if skipSpecialTokens { continue }
                out.append(contentsOf: piece.utf8)     // content verbatim
                continue
            }
            switch mode {
            case .byteLevel:
                for sc in piece.unicodeScalars {
                    if let b = byteUnmap[sc] {
                        out.append(b)
                    } else {
                        out.append(contentsOf: String(sc).utf8)   // lossy path (see above)
                    }
                }
            case .metaspaceByteFallback(let rep, _, _):
                if let fb = byteFallbackUnmap[id] {
                    out.append(fb)
                } else {
                    var replaced = piece.replacingOccurrences(of: rep, with: " ")
                    if !sawFirstPiece && stripLeadingSpace && replaced.hasPrefix(" ") {
                        replaced.removeFirst()
                    }
                    out.append(contentsOf: replaced.utf8)
                }
                sawFirstPiece = true
            }
        }
        return String(decoding: out, as: UTF8.self)
    }
}

/// UTF-8 byte width of a scalar (exact encoder widths, allocation-free).
func qwUTF8Width(of sc: Unicode.Scalar) -> Int {
    switch sc.value {
    case 0x00..<0x80: return 1
    case 0x80..<0x800: return 2
    case 0x800..<0x10000: return 3
    default: return 4
    }
}

// MARK: - streaming (incremental) decoder
/// C22 needs decode-to-tokenize-while-generating: feed ids one at a time;
/// every emitted chunk is a UTF-8 scalar-boundary prefix, and the full
public struct QWStreamDecoder {
    let tok: QWTokenizer
    var pending: [UInt8] = []
    var sawFirstPiece: Bool = false

    public init(_ tok: QWTokenizer) { self.tok = tok }

    public mutating func decode(_ id: Int, skipSpecialTokens: Bool = false) throws -> String {
        if tok.specialIds.contains(id) {
            if skipSpecialTokens { return "" }
            // Special content is ASCII marker text: drain pending first
            // (order law of the one-shot join), then emit the content.
            var s = drainPending()
            guard let piece = tok.idToPiece[id] else {
                throw QWTokenizer.Error.unresolvedByte("id \(id) (no vocab piece at all)")
            }
            s += piece
            return s
        }
        guard let piece = tok.idToPiece[id] else {
            throw QWTokenizer.Error.unresolvedByte("id \(id) (no vocab piece at all)")
        }
        switch tok.mode {
        case .byteLevel:
            for sc in piece.unicodeScalars {
                if let b = tok.byteUnmap[sc] {
                    pending.append(b)
                } else {
                    pending.append(contentsOf: String(sc).utf8)   // lossy path (see decode)
                }
            }
        case .metaspaceByteFallback(let rep, _, _):
            if let fb = tok.byteFallbackUnmap[id] {
                pending.append(fb)
            } else {
                var replaced = piece.replacingOccurrences(of: rep, with: " ")
                if !sawFirstPiece && tok.stripLeadingSpace && replaced.hasPrefix(" ") {
                    replaced.removeFirst()
                }
                pending.append(contentsOf: replaced.utf8)
            }
            sawFirstPiece = true
        }
        return drainPending()
    }

    /// End of stream: emit the lossy remainder (== one-shot lossy tail).
    public mutating func flush() -> String {
        guard !pending.isEmpty else { return "" }
        let s = String(decoding: pending, as: UTF8.self)
        pending.removeAll(keepingCapacity: false)
        return s
    }

    /// Emit the longest prefix ending on a scalar boundary; hold a
    /// structurally incomplete final sequence.
    private mutating func drainPending() -> String {
        let n = pending.count
        guard n > 0 else { return "" }
        // Walk back over trailing continuation bytes (max 3).
        var k = 0
        while k < 4, n - 1 - k >= 0, (pending[n - 1 - k] & 0xC0) == 0x80 { k += 1 }
        var hold = 0
        if n - k == 0 {
            hold = k                      // lone continuation run (orphan)
        } else {
            let lead = pending[n - k - 1]
            let w = QWStreamDecoder.utf8LeadWidth(lead)
            if k + 1 < w { hold = k + 1 }  // incomplete: lead + k continuations < w
        }
        let end = n - hold
        guard end > 0 else { return "" }
        let s = String(decoding: pending[0..<end], as: UTF8.self)
        pending.removeFirst(end)
        return s
    }

    /// Strict structural lead-byte width (0x80…0xC1 / 0xF5… are invalid
    /// leads — width 1 so they never hold continuation bytes).
    private static func utf8LeadWidth(_ lead: UInt8) -> Int {
        switch lead {
        case 0x00...0x7F: return 1
        case 0xC2...0xDF: return 2
        case 0xE0...0xEF: return 3
        case 0xF0...0xF4: return 4
        default: return 1
        }
    }
}
