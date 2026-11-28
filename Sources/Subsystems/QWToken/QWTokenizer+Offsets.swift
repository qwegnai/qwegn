// QWTokenizer+Offsets.swift — Bijective byte/UTF-16/scalar offset encoding and single-token helpers.
import Foundation

/// Byte, UTF-16, and Unicode-scalar offsets inside source text for one emitted token ID.
public struct QWTokenSpan: Sendable, Hashable, Equatable {
    public let tokenId: Int
    public let utf8Range: Range<Int>
    public let utf16Range: Range<Int>
    public let scalarRange: Range<Int>

    public init(tokenId: Int, utf8Range: Range<Int>, utf16Range: Range<Int>, scalarRange: Range<Int>) {
        self.tokenId = tokenId
        self.utf8Range = utf8Range
        self.utf16Range = utf16Range
        self.scalarRange = scalarRange
    }
}

/// Token IDs paired with bijective source-text spans from `QWTokenizer.encodeWithOffsets`.
public struct QWTokenizedSpanList: Sendable, Hashable, Equatable {
    public let text: String
    public let ids: [Int]
    public let spans: [QWTokenSpan]

    public init(text: String, ids: [Int], spans: [QWTokenSpan]) {
        self.text = text
        self.ids = ids
        self.spans = spans
    }

    /// Slices the exact source substring covered by a token index range.
    public func substring(forTokenRange range: Range<Int>) -> String {
        let clampedLo = max(0, range.lowerBound)
        let clampedHi = min(spans.count, range.upperBound)
        guard clampedLo < clampedHi else { return "" }
        let bLo = spans[clampedLo].utf8Range.lowerBound
        let bHi = spans[clampedHi - 1].utf8Range.upperBound
        let raw = Array(text.utf8)
        let sLo = max(0, min(bLo, raw.count))
        let sHi = max(sLo, min(bHi, raw.count))
        return String(decoding: raw[sLo..<sHi], as: UTF8.self)
    }
}

extension QWTokenizer {
    /// Returns all registered special / added token strings sorted longest-first for escaping.
    public func allSpecialTokenStrings() -> [String] {
        var set = Set<String>()
        for entry in added where !entry.s.isEmpty {
            if specialIds.contains(entry.id) || entry.s.hasPrefix("<") || entry.s.hasPrefix("[") {
                set.insert(entry.s)
            }
        }
        for sid in specialIds {
            if let p = piece(sid), !p.isEmpty { set.insert(p) }
        }
        return set.sorted { lhs, rhs in
            if lhs.utf8.count != rhs.utf8.count { return lhs.utf8.count > rhs.utf8.count }
            return lhs < rhs
        }
    }

    /// Looks up a single token ID for an exact special string or a standalone 1-token string.
    public func singleTokenId(forText text: String) -> Int? {
        if let exactAdded = added.first(where: { $0.s == text }) {
            return exactAdded.id
        }
        guard let encoded = try? encode(text, addSpecialTokens: false), encoded.count == 1 else {
            return nil
        }
        return encoded[0]
    }

    /// Exact vocab or added-token ID lookup for a piece string.
    public func pieceTokenId(_ pieceString: String) -> Int? {
        tokenId(for: pieceString)
    }

    /// Encodes `text` with `encode` identity preserved and aligns each token to source offsets.
    public func encodeWithOffsets(_ text: String, addSpecialTokens: Bool = false) throws -> QWTokenizedSpanList {
        let ids = try encode(text, addSpecialTokens: addSpecialTokens)
        guard !ids.isEmpty else {
            return QWTokenizedSpanList(text: text, ids: [], spans: [])
        }
        let nfcText = usesNFC ? text.precomposedStringWithCanonicalMapping : text
        let alignTarget = nfcText
        let targetBytes = Array(alignTarget.utf8)
        let nfcToOrigU8 = Self.buildNFCByteMap(original: text, normalized: alignTarget)
        let origBytes = Array(text.utf8)
        let (u8To16, u8ToSc) = Self.buildUTF8IndexMaps(for: text, utf8Count: origBytes.count)

        let addedById: [Int: [UInt8]] = Dictionary(
            added.map { ($0.id, Array($0.s.utf8)) },
            uniquingKeysWith: { first, _ in first }
        )
        var cursor = 0
        var spans: [QWTokenSpan] = []
        spans.reserveCapacity(ids.count)

        let pfxCount = (addSpecialTokens && ids.starts(with: postProcessorPrefixIds)) ? postProcessorPrefixIds.count : 0
        let sfxCount = (addSpecialTokens && (ids.count - pfxCount) >= postProcessorSuffixIds.count
            && ids.suffix(postProcessorSuffixIds.count).elementsEqual(postProcessorSuffixIds))
            ? postProcessorSuffixIds.count : 0
        for tid in ids.prefix(pfxCount) {
            spans.append(QWTokenSpan(tokenId: tid, utf8Range: 0..<0, utf16Range: 0..<0, scalarRange: 0..<0))
        }
        let bodyIds = ids[pfxCount..<(ids.count - sfxCount)]

        for tid in bodyIds {
            let emitted = rawTokenBytes(for: tid, addedById: addedById)
            var tokBytes = emitted
            if case .metaspaceByteFallback = mode,
               !tokBytes.isEmpty, tokBytes[0] == 0x20,
               cursor >= targetBytes.count || targetBytes[cursor] != 0x20 {
                tokBytes.removeFirst()
            }

            let alignedLo: Int
            let alignedHi: Int
            if tokBytes.isEmpty {
                alignedLo = cursor
                alignedHi = cursor
            } else if cursor + tokBytes.count <= targetBytes.count,
                      targetBytes[cursor..<(cursor + tokBytes.count)].elementsEqual(tokBytes) {
                alignedLo = cursor
                alignedHi = cursor + tokBytes.count
            } else if addedById[tid] != nil,
                      let found = Self.findSubslice(tokBytes, in: targetBytes, from: cursor) {
                alignedLo = found
                alignedHi = found + tokBytes.count
            } else {
                alignedLo = cursor
                alignedHi = min(targetBytes.count, cursor + tokBytes.count)
            }
            cursor = alignedHi

            let origLo = nfcToOrigU8[min(alignedLo, nfcToOrigU8.count - 1)]
            let origHi = max(origLo, nfcToOrigU8[min(alignedHi, nfcToOrigU8.count - 1)])
            let u16Lo = u8To16[origLo]
            let u16Hi = max(u16Lo, u8To16[origHi])
            let scLo = u8ToSc[origLo]
            let scHi = max(scLo, u8ToSc[origHi])
            spans.append(QWTokenSpan(
                tokenId: tid,
                utf8Range: origLo..<origHi,
                utf16Range: u16Lo..<u16Hi,
                scalarRange: scLo..<scHi
            ))
        }
        let endU8 = origBytes.count
        let endU16 = text.utf16.count
        let endSc = text.unicodeScalars.count
        for tid in ids.suffix(sfxCount) {
            spans.append(QWTokenSpan(
                tokenId: tid,
                utf8Range: endU8..<endU8,
                utf16Range: endU16..<endU16,
                scalarRange: endSc..<endSc
            ))
        }
        return QWTokenizedSpanList(text: text, ids: ids, spans: spans)
    }

    private func rawTokenBytes(for tid: Int, addedById: [Int: [UInt8]]) -> [UInt8] {
        if let special = addedById[tid] { return special }
        if specialIds.contains(tid), let p = idToPiece[tid] { return Array(p.utf8) }
        guard let piece = idToPiece[tid] else { return [] }
        switch mode {
        case .byteLevel:
            var out: [UInt8] = []
            out.reserveCapacity(piece.utf8.count)
            for sc in piece.unicodeScalars {
                if let b = byteUnmap[sc] {
                    out.append(b)
                } else {
                    out.append(contentsOf: String(sc).utf8)
                }
            }
            return out
        case .metaspaceByteFallback(let rep, _, _):
            if let fb = byteFallbackUnmap[tid] { return [fb] }
            return Array(piece.replacingOccurrences(of: rep, with: " ").utf8)
        }
    }

    private static func findSubslice(_ needle: [UInt8], in haystack: [UInt8], from start: Int) -> Int? {
        guard !needle.isEmpty, start + needle.count <= haystack.count else { return nil }
        let limit = haystack.count - needle.count
        for i in start...limit where haystack[i..<(i + needle.count)].elementsEqual(needle) {
            return i
        }
        return nil
    }

    private static func buildUTF8IndexMaps(for text: String, utf8Count: Int) -> ([Int], [Int]) {
        var u16Map = [Int](repeating: text.utf16.count, count: utf8Count + 1)
        var scMap = [Int](repeating: text.unicodeScalars.count, count: utf8Count + 1)
        var b = 0
        var u16 = 0
        var scIdx = 0
        for sc in text.unicodeScalars {
            let w = qwUTF8Width(of: sc)
            for k in 0..<w where (b + k) <= utf8Count {
                u16Map[b + k] = u16
                scMap[b + k] = scIdx
            }
            b += w
            u16 += sc.utf16.count
            scIdx += 1
        }
        u16Map[utf8Count] = u16
        scMap[utf8Count] = scIdx
        return (u16Map, scMap)
    }

    private static func buildNFCByteMap(original: String, normalized: String) -> [Int] {
        let origLen = original.utf8.count
        let normLen = normalized.utf8.count
        if origLen == normLen && original == normalized {
            return Array(0...origLen)
        }
        var map = [Int](repeating: origLen, count: normLen + 1)
        var nb = 0
        var ob = 0
        var nScalars = normalized.unicodeScalars.makeIterator()
        var origClusters = original.makeIterator()
        while let ch = origClusters.next() {
            let chOrigW = ch.utf8.count
            let chNorm = String(ch).precomposedStringWithCanonicalMapping
            let chNormW = chNorm.utf8.count
            for k in 0..<chNormW where (nb + k) <= normLen {
                map[nb + k] = ob + min(k, chOrigW)
            }
            for _ in chNorm.unicodeScalars { _ = nScalars.next() }
            nb += chNormW
            ob += chOrigW
        }
        map[normLen] = origLen
        return map
    }
}
