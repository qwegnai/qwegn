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
    private func directVocabTokenId(bytes: [UInt8], lo: Int, hi: Int) -> Int? {
        var scalars = String.UnicodeScalarView()
        scalars.reserveCapacity(hi - lo)
        for bi in lo..<hi {
            scalars.append(byteMap[Int(bytes[bi])])
        }
        return vocab[String(scalars)]
    }

    // MARK: BPE one word (already ByteLevel-mapped) → ids
    /// Merge law (engine parity): min-rank adjacent pair wins, leftmost on
    /// ties (strict `<` scan), ALL occurrences of that pair merge in one
    /// left-to-right pass; repeat until no ranked pair remains.
    private func bpeMergeInPlace(_ word: inout [UInt32]) {
        var count = word.count
        while count > 1 {
            var bestRank = Int.max
            var bestA: UInt32 = 0
            var bestB: UInt32 = 0
            var bestMerged: UInt32 = 0
            word.withUnsafeBufferPointer { buf in
                guard let p = buf.baseAddress else { return }
                var i = 0
                while i < count - 1 {
                    let a = p[i]
                    let b = p[i + 1]
                    let k = (UInt64(a) << 32) | UInt64(b)
                    if let rule = mergeRules[k], rule.rank < bestRank {
                        bestRank = rule.rank
                        bestA = a
                        bestB = b
                        bestMerged = rule.merged
                    }
                    i += 1
                }
            }
            guard bestRank != Int.max else { break }
            var w = 0
            word.withUnsafeMutableBufferPointer { buf in
                guard let p = buf.baseAddress else { return }
                var r = 0
                while r < count {
                    if r + 1 < count && p[r] == bestA && p[r + 1] == bestB {
                        p[w] = bestMerged
                        r += 2
                    } else {
                        p[w] = p[r]
                        r += 1
                    }
                    w += 1
                }
            }
            word.removeSubrange(w..<count)
            count = w
        }
    }

    @inline(__always)
    static func isPureASCII(_ bytes: [UInt8], _ lo: Int, _ hi: Int) -> Bool {
        guard hi > lo else { return true }
        return bytes.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return true }
            var i = lo
            let end8 = hi - 7
            while i < end8 {
                let w = UnsafeRawPointer(base.advanced(by: i)).loadUnaligned(as: UInt64.self)
                if (w & 0x8080_8080_8080_8080) != 0 { return false }
                i += 8
            }
            while i < hi {
                if base[i] >= 0x80 { return false }
                i += 1
            }
            return true
        }
    }

    // MARK: encode
    /// Encodes `text` to token IDs; applies `TemplateProcessing` special tokens only when `addSpecialTokens == true`.
    public func encode(_ text: String, addSpecialTokens: Bool = false) throws -> [Int] {
        let rawBytes = Array(text.utf8)
        if rawBytes.isEmpty {
            return addSpecialTokens ? (postProcessorPrefixIds + postProcessorSuffixIds) : []
        }
        var ids: [Int] = []
        ids.reserveCapacity(max(8, rawBytes.count / 3 + 8 + (addSpecialTokens ? postProcessorPrefixIds.count + postProcessorSuffixIds.count : 0)))
        if addSpecialTokens && !postProcessorPrefixIds.isEmpty {
            ids.append(contentsOf: postProcessorPrefixIds)
        }
        let cacheEnabled = rawBytes.count >= 128
        var wordCache = QWWordMergeCache()
        var wordScratch: [UInt32] = []
        var tokScratch: [Int] = []
        var pieceBytes: [UInt8] = []
        var spans: [(Int, Int)] = []
        var nextSpans: [(Int, Int)] = []
        wordScratch.reserveCapacity(64)
        tokScratch.reserveCapacity(32)

        func emitBPEFromScratch(
            keyBytes: [UInt8],
            keyLo: Int,
            keyHi: Int,
            cacheKeyHash: UInt64?,
            unresolvedText: @autoclosure () -> String
        ) throws {
            bpeMergeInPlace(&wordScratch)
            tokScratch.removeAll(keepingCapacity: true)
            let symCount = symTok.count
            for sid in wordScratch {
                let idx = Int(sid)
                let tid = idx < symCount ? symTok[idx] : -1
                if tid < 0 {
                    throw Error.unresolvedByte(unresolvedText())
                }
                tokScratch.append(tid)
            }
            if let hash = cacheKeyHash {
                let len = keyHi - keyLo
                keyBytes.withUnsafeBufferPointer { buf in
                    guard let base = buf.baseAddress else { return }
                    wordCache.insert(base.advanced(by: keyLo), count: len, hash: hash, tokens: tokScratch)
                }
            }
            ids.append(contentsOf: tokScratch)
        }

        func encodeChunk(
            _ bytes: [UInt8],
            _ cLo: Int,
            _ cHi: Int,
            isAtTextStart: Bool,
            isFirstChunk: Bool
        ) throws {
            guard cHi > cLo else { return }
            if case .metaspaceByteFallback(let rep, let scheme, let splitWords) = mode {
                var seg = String(decoding: bytes[cLo..<cHi], as: UTF8.self)
                seg = seg.replacingOccurrences(of: " ", with: rep)
                let shouldPrepend = !rep.isEmpty && !seg.hasPrefix(rep)
                    && ((scheme == "always" && isFirstChunk) || (scheme == "first" && isAtTextStart && isFirstChunk))
                if shouldPrepend {
                    seg = rep + seg
                }
                let wordPieces = splitWords && !rep.isEmpty
                    ? Self.splitMetaspaceMergedWithNext(seg, delimiter: rep)
                    : [seg[...]]
                for piece in wordPieces where !piece.isEmpty {
                    var missHash: UInt64? = nil
                    if cacheEnabled {
                        pieceBytes.removeAll(keepingCapacity: true)
                        pieceBytes.append(contentsOf: piece.utf8)
                        let len = pieceBytes.count
                        if len <= QWWordMergeCache.maxKeyBytes {
                            let (hit, h) = pieceBytes.withUnsafeBufferPointer { buf -> (Bool, UInt64) in
                                guard let base = buf.baseAddress else { return (false, 0) }
                                let hash = QWWordMergeCache.hashBytes(base, count: len)
                                return (wordCache.lookup(base, count: len, hash: hash, into: &ids), hash)
                            }
                            if hit { continue }
                            missHash = h
                        }
                    }
                    wordScratch.removeAll(keepingCapacity: true)
                    for sc in piece.unicodeScalars {
                        let s = String(sc)
                        if let sid = symId[s] {
                            wordScratch.append(sid)
                        } else {
                            for b in s.utf8 { wordScratch.append(byteToSymId[Int(b)]) }
                        }
                    }
                    try emitBPEFromScratch(
                        keyBytes: pieceBytes, keyLo: 0, keyHi: pieceBytes.count,
                        cacheKeyHash: missHash, unresolvedText: String(piece.prefix(16))
                    )
                }
                return
            }
            spans.removeAll(keepingCapacity: true)
            spans.append((cLo, cHi))
            let chunkIsASCII = Self.isPureASCII(bytes, cLo, cHi)
            for step in preSteps {
                let re: NSRegularExpression
                let matchesOnly: Bool
                switch step {
                case .digits(_, let r):
                    re = r
                    matchesOnly = false
                case .splitIsolated(let r):
                    re = r
                    matchesOnly = false
                case .splitRemovedInverted(let r):
                    re = r
                    matchesOnly = true
                }
                nextSpans.removeAll(keepingCapacity: true)
                for (sLo, sHi) in spans where sHi > sLo {
                    let seg = String(decoding: bytes[sLo..<sHi], as: UTF8.self)
                    let segBytes = sHi - sLo
                    let segIsASCII = chunkIsASCII || Self.isPureASCII(bytes, sLo, sHi)
                    let utf16Len = segIsASCII ? segBytes : (seg as NSString).length
                    var u2b: [Int]? = nil
                    var cursor = 0
                    re.enumerateMatches(in: seg, range: NSRange(location: 0, length: utf16Len)) { mr, _, _ in
                        guard let r = mr?.range, r.length > 0 else { return }
                        let b0: Int
                        let b1: Int
                        if segIsASCII {
                            b0 = r.location
                            b1 = r.location + r.length
                        } else {
                            let table = u2b ?? {
                                let built = QWTokenizer.utf16ToByteTable(seg)
                                u2b = built
                                return built
                            }()
                            b0 = table[r.location]
                            b1 = table[r.location + r.length]
                        }
                        if !matchesOnly && b0 > cursor {
                            nextSpans.append((sLo + cursor, sLo + b0))
                        }
                        if b1 > b0 {
                            nextSpans.append((sLo + b0, sLo + b1))
                        }
                        cursor = max(cursor, b1)
                    }
                    if !matchesOnly && cursor < segBytes {
                        nextSpans.append((sLo + cursor, sHi))
                    }
                }
                swap(&spans, &nextSpans)
            }
            for (pLo, pHi) in spans where pHi > pLo {
                let byteLen = pHi - pLo
                if byteLen == 1 {
                    let tid = byteToTokId[Int(bytes[pLo])]
                    if tid < 0 {
                        throw Error.unresolvedByte(String(String(
                            decoding: bytes[pLo..<pHi], as: UTF8.self).prefix(16)))
                    }
                    ids.append(tid)
                    continue
                }
                var missHash: UInt64? = nil
                if cacheEnabled && byteLen <= QWWordMergeCache.maxKeyBytes {
                    let (hit, h) = bytes.withUnsafeBufferPointer { buf -> (Bool, UInt64) in
                        guard let base = buf.baseAddress else { return (false, 0) }
                        let ptr = base.advanced(by: pLo)
                        let hash = QWWordMergeCache.hashBytes(ptr, count: byteLen)
                        return (wordCache.lookup(ptr, count: byteLen, hash: hash, into: &ids), hash)
                    }
                    if hit { continue }
                    missHash = h
                }
                if ignoreMerges, let directTokId = directVocabTokenId(bytes: bytes, lo: pLo, hi: pHi) {
                    if let hash = missHash {
                        tokScratch.removeAll(keepingCapacity: true)
                        tokScratch.append(directTokId)
                        bytes.withUnsafeBufferPointer { buf in
                            guard let base = buf.baseAddress else { return }
                            wordCache.insert(base.advanced(by: pLo), count: byteLen, hash: hash, tokens: tokScratch)
                        }
                    }
                    ids.append(directTokId)
                    continue
                }
                wordScratch.removeAll(keepingCapacity: true)
                for bi in pLo..<pHi {
                    wordScratch.append(byteToSymId[Int(bytes[bi])])
                }
                try emitBPEFromScratch(
                    keyBytes: bytes, keyLo: pLo, keyHi: pHi, cacheKeyHash: missHash,
                    unresolvedText: String(String(decoding: bytes[pLo..<pHi], as: UTF8.self).prefix(16))
                )
            }
        }

        func encodeSegment(_ bytes: [UInt8], _ lo: Int, _ hi: Int, isAtTextStart: Bool) throws {
            guard hi > lo else { return }
            let chunkSize = 16384
            if hi - lo <= chunkSize {
                try encodeChunk(bytes, lo, hi, isAtTextStart: isAtTextStart, isFirstChunk: true)
                return
            }
            var cur = lo
            while cur < hi {
                var nextEnd = min(cur + chunkSize, hi)
                if nextEnd < hi {
                    nextEnd = Self.chooseChunkSplitBoundary(bytes: bytes, cur: cur, limit: nextEnd, hi: hi, mode: mode)
                }
                try encodeChunk(bytes, cur, nextEnd, isAtTextStart: isAtTextStart, isFirstChunk: cur == lo)
                cur = nextEnd
            }
        }

        func encodeNormalizedSlice(_ sliceBytes: [UInt8], _ lo: Int, _ hi: Int, isAtTextStart: Bool) throws {
            guard hi > lo else { return }
            if normAddedPats.isEmpty {
                try encodeSegment(sliceBytes, lo, hi, isAtTextStart: isAtTextStart)
                return
            }
            let normMatches = Self.matchAddedTokens(
                in: sliceBytes, lo: lo, hi: hi, patterns: normAddedPats, firstByte: normAddedFirstByte
            )
            var pos = lo
            for m in normMatches {
                if m.loc > pos {
                    try encodeSegment(sliceBytes, pos, m.loc, isAtTextStart: isAtTextStart && pos == lo)
                }
                ids.append(m.id)
                pos = m.end
            }
            if pos < hi {
                try encodeSegment(sliceBytes, pos, hi, isAtTextStart: isAtTextStart && pos == lo)
            }
        }

        let rawMatches = rawAddedPats.isEmpty
            ? []
            : Self.matchAddedTokens(in: rawBytes, lo: 0, hi: rawBytes.count, patterns: rawAddedPats, firstByte: rawAddedFirstByte)
        if rawMatches.isEmpty {
            if !usesNFC || Self.isPureASCII(rawBytes, 0, rawBytes.count) {
                try encodeNormalizedSlice(rawBytes, 0, rawBytes.count, isAtTextStart: true)
            } else {
                let nfc = Array(text.precomposedStringWithCanonicalMapping.utf8)
                try encodeNormalizedSlice(nfc, 0, nfc.count, isAtTextStart: true)
            }
            if addSpecialTokens && !postProcessorSuffixIds.isEmpty {
                ids.append(contentsOf: postProcessorSuffixIds)
            }
            return ids
        }

        var rawPos = 0
        for m in rawMatches {
            if m.loc > rawPos {
                let atStart = rawPos == 0
                if !usesNFC || Self.isPureASCII(rawBytes, rawPos, m.loc) {
                    try encodeNormalizedSlice(rawBytes, rawPos, m.loc, isAtTextStart: atStart)
                } else {
                    let rawSub = String(decoding: rawBytes[rawPos..<m.loc], as: UTF8.self)
                    let normBytes = Array(rawSub.precomposedStringWithCanonicalMapping.utf8)
                    try encodeNormalizedSlice(normBytes, 0, normBytes.count, isAtTextStart: atStart)
                }
            }
            ids.append(m.id)
            rawPos = m.end
        }
        if rawPos < rawBytes.count {
            if !usesNFC || Self.isPureASCII(rawBytes, rawPos, rawBytes.count) {
                try encodeNormalizedSlice(rawBytes, rawPos, rawBytes.count, isAtTextStart: false)
            } else {
                let rawSub = String(decoding: rawBytes[rawPos..<rawBytes.count], as: UTF8.self)
                let normBytes = Array(rawSub.precomposedStringWithCanonicalMapping.utf8)
                try encodeNormalizedSlice(normBytes, 0, normBytes.count, isAtTextStart: false)
            }
        }
        if addSpecialTokens && !postProcessorSuffixIds.isEmpty {
            ids.append(contentsOf: postProcessorSuffixIds)
        }
        return ids
    }

    private static func chooseChunkSplitBoundary(
        bytes: [UInt8],
        cur: Int,
        limit: Int,
        hi: Int,
        mode: Mode
    ) -> Int {
        func findParagraphBoundary() -> Int? {
            var scan = limit
            var trailingRunStart: Int? = nil
            if bytes[scan] == 0x0A || bytes[scan] == 0x0D {
                while scan > cur && (bytes[scan - 1] == 0x0A || bytes[scan - 1] == 0x0D) {
                    scan -= 1
                }
                if scan > cur {
                    trailingRunStart = scan
                }
            }
            while scan > cur {
                if bytes[scan - 1] == 0x0A && bytes[scan] != 0x0A && bytes[scan] != 0x0D {
                    return scan
                }
                scan -= 1
            }
            return trailingRunStart
        }
        func findLeadingWhitespaceBoundary(includeTab: Bool) -> Int? {
            var ws = limit
            while ws > cur && bytes[ws - 1] != 0x20 && (!includeTab || bytes[ws - 1] != 0x09) {
                ws -= 1
            }
            guard ws > cur else { return nil }
            while ws > cur && (bytes[ws - 1] == 0x20 || (includeTab && bytes[ws - 1] == 0x09)) {
                ws -= 1
            }
            return ws > cur ? ws : nil
        }

        if case .metaspaceByteFallback = mode {
            if let ws = findLeadingWhitespaceBoundary(includeTab: false) { return ws }
            if let nl = findParagraphBoundary() { return nl }
            if let ts = findLeadingWhitespaceBoundary(includeTab: true) { return ts }
        } else {
            if let nl = findParagraphBoundary() { return nl }
            if let ws = findLeadingWhitespaceBoundary(includeTab: true) { return ws }
        }
        var end = limit
        while end < hi && (bytes[end] & 0xC0) == 0x80 {
            end += 1
        }
        return end
    }

    private static func splitMetaspaceMergedWithNext(_ s: String, delimiter: String) -> [Substring] {
        var out: [Substring] = []
        var pieceStart = s.startIndex
        var searchStart = s.startIndex
        while searchStart < s.endIndex,
              let range = s.range(of: delimiter, range: searchStart..<s.endIndex) {
            if range.lowerBound > pieceStart {
                out.append(s[pieceStart..<range.lowerBound])
                pieceStart = range.lowerBound
            }
            searchStart = range.upperBound
        }
        if pieceStart < s.endIndex {
            out.append(s[pieceStart..<s.endIndex])
        }
        return out
    }

    private static func matchAddedTokens(
        in bytes: [UInt8],
        lo: Int,
        hi: Int,
        patterns: [AddedPattern],
        firstByte: [Bool]
    ) -> [(loc: Int, end: Int, id: Int)] {
        var matches: [(loc: Int, end: Int, id: Int)] = []
        var bi = lo
        var lastMatchEnd = lo
        while bi < hi {
            while bi < hi && !firstByte[Int(bytes[bi])] { bi += 1 }
            guard bi < hi else { break }
            var matched = false
            for p in patterns {
                let patEnd = bi + p.pat.count
                guard patEnd <= hi, bytes[bi..<patEnd].elementsEqual(p.pat) else { continue }
                if p.singleWord {
                    if bi > lastMatchEnd, let prev = utf8ScalarBefore(bytes, index: bi, minIndex: lastMatchEnd), isWordScalar(prev.scalar) {
                        continue
                    }
                    if patEnd < hi, let next = utf8ScalarAt(bytes, index: patEnd, maxIndex: hi), isWordScalar(next.scalar) {
                        continue
                    }
                }
                var matchStart = bi
                var matchEnd = patEnd
                if p.lstrip {
                    while matchStart > lastMatchEnd,
                          let prev = utf8ScalarBefore(bytes, index: matchStart, minIndex: lastMatchEnd),
                          prev.scalar.properties.isWhitespace {
                        matchStart = prev.start
                    }
                }
                if p.rstrip {
                    while matchEnd < hi,
                          let next = utf8ScalarAt(bytes, index: matchEnd, maxIndex: hi),
                          next.scalar.properties.isWhitespace {
                        matchEnd = next.end
                    }
                }
                matches.append((matchStart, matchEnd, p.id))
                lastMatchEnd = matchEnd
                bi = matchEnd
                matched = true
                break
            }
            if !matched {
                bi += 1
            }
        }
        return matches
    }

    private static func isWordScalar(_ sc: Unicode.Scalar) -> Bool {
        sc == "_" || sc.properties.isAlphabetic || sc.properties.numericType != nil
    }

    private static func utf8ScalarBefore(_ bytes: [UInt8], index: Int, minIndex: Int) -> (scalar: Unicode.Scalar, start: Int)? {
        guard index > minIndex else { return nil }
        var start = index - 1
        var steps = 0
        while start > minIndex && steps < 3 && (bytes[start] & 0xC0) == 0x80 {
            start -= 1
            steps += 1
        }
        var iter = bytes[start..<index].makeIterator()
        var utf8 = UTF8()
        guard case .scalarValue(let sc) = utf8.decode(&iter), iter.next() == nil else { return nil }
        return (sc, start)
    }

    private static func utf8ScalarAt(_ bytes: [UInt8], index: Int, maxIndex: Int) -> (scalar: Unicode.Scalar, end: Int)? {
        guard index < maxIndex else { return nil }
        let limit = min(maxIndex, index + 4)
        var iter = bytes[index..<limit].makeIterator()
        var utf8 = UTF8()
        guard case .scalarValue(let sc) = utf8.decode(&iter) else { return nil }
        return (sc, index + qwUTF8Width(of: sc))
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
        for (id, piece) in idToPiece where id >= 0 && !specialIds.contains(id) {
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
            guard let piece = tok.idToPiece[id] else {
                throw QWTokenizer.Error.unresolvedByte("id \(id) (no vocab piece at all)")
            }
            // Flush incomplete UTF-8 sequences before emitting special text so
            // replacement characters do not reorder across special token boundaries.
            var s = flush()
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
