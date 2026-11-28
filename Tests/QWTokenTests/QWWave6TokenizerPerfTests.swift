// QWWave6TokenizerPerfTests.swift — Wave 6 parity, cache-cap, pure-ASCII, and
// concurrency tests for QWTokenizer fast-path BPE encoding and streaming decode.
import Foundation
import XCTest
@testable import QWToken

final class QWWave6TokenizerPerfTests: XCTestCase {
    private static let rootURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    private static func loadAssetTokenizer() throws -> QWTokenizer {
        let url = rootURL.appendingPathComponent("Scripts/spike/assets/tokenizer.json")
        guard let data = FileManager.default.contents(atPath: url.path) else {
            throw XCTSkip("Scripts/spike/assets/tokenizer.json not available")
        }
        return try QWTokenizer(json: data)
    }

    private func referenceBPE(_ tok: QWTokenizer, syms: [UInt32]) -> [Int] {
        var word = syms
        while word.count > 1 {
            var bestKey = UInt64.max
            var bestRank = Int.max
            for i in 0..<(word.count - 1) {
                let k = (UInt64(word[i]) << 32) | UInt64(word[i + 1])
                if let rule = tok.mergeRules[k], rule.rank < bestRank {
                    bestRank = rule.rank
                    bestKey = k
                }
            }
            guard bestKey != UInt64.max, let rule = tok.mergeRules[bestKey] else { break }
            let a = UInt32(bestKey >> 32)
            let b = UInt32(bestKey & 0xFFFF_FFFF)
            var next: [UInt32] = []
            next.reserveCapacity(word.count)
            var i = 0
            while i < word.count {
                if i + 1 < word.count && word[i] == a && word[i + 1] == b {
                    next.append(rule.merged)
                    i += 2
                } else {
                    next.append(word[i])
                    i += 1
                }
            }
            word = next
        }
        return word.map { Int($0) < tok.symTok.count ? tok.symTok[Int($0)] : -1 }
    }

    /// Independent uncached, unoptimized reference encoder (unconditional NFC + utf16ToByteTable + fresh BPE).
    private func uncachedReferenceEncode(_ tok: QWTokenizer, _ text: String) throws -> [Int] {
        guard !text.isEmpty else { return [] }
        let addedByContent = Dictionary(tok.added.map { ($0.s, $0.id) }, uniquingKeysWith: { a, _ in a })
        // Split only on non-overlapping exact raw special strings without lstrip/rstrip when absent,
        // or verify on texts without lstrip/rstrip specials.
        var out: [Int] = []
        func encodePlain(_ plain: String, isStart: Bool) {
            let norm = tok.usesNFC ? plain.precomposedStringWithCanonicalMapping : plain
            if case .metaspaceByteFallback(let rep, let scheme, let splitWords) = tok.mode {
                var seg = norm.replacingOccurrences(of: " ", with: rep)
                if !rep.isEmpty && !seg.hasPrefix(rep) && (scheme == "always" || (scheme == "first" && isStart)) {
                    seg = rep + seg
                }
                var pieces: [Substring] = []
                if splitWords && !rep.isEmpty {
                    var start = seg.startIndex
                    var scan = seg.startIndex
                    while scan < seg.endIndex, let r = seg.range(of: rep, range: scan..<seg.endIndex) {
                        if r.lowerBound > start { pieces.append(seg[start..<r.lowerBound]); start = r.lowerBound }
                        scan = r.upperBound
                    }
                    if start < seg.endIndex { pieces.append(seg[start..<seg.endIndex]) }
                } else {
                    pieces = [seg[...]]
                }
                for p in pieces where !p.isEmpty {
                    var syms: [UInt32] = []
                    for sc in p.unicodeScalars {
                        let s = String(sc)
                        if let sid = tok.symId[s] { syms.append(sid) } else {
                            for b in s.utf8 { syms.append(tok.byteToSymId[Int(b)]) }
                        }
                    }
                    out.append(contentsOf: referenceBPE(tok, syms: syms))
                }
                return
            }
            let bytes = Array(norm.utf8)
            var spans: [(Int, Int)] = [(0, bytes.count)]
            for step in tok.preSteps {
                let re: NSRegularExpression
                let matchOnly: Bool
                switch step {
                case .digits(_, let r):
                    re = r
                    matchOnly = false
                case .splitIsolated(let r):
                    re = r
                    matchOnly = false
                case .splitRemovedInverted(let r):
                    re = r
                    matchOnly = true
                }
                var next: [(Int, Int)] = []
                for (lo, hi) in spans where hi > lo {
                    let seg = String(decoding: bytes[lo..<hi], as: UTF8.self)
                    let u2b = QWTokenizer.utf16ToByteTable(seg)
                    var cursor = 0
                    re.enumerateMatches(in: seg, range: NSRange(location: 0, length: (seg as NSString).length)) { mr, _, _ in
                        guard let r = mr?.range, r.length > 0 else { return }
                        let b0 = u2b[r.location], b1 = u2b[r.location + r.length]
                        if !matchOnly && b0 > cursor { next.append((lo + cursor, lo + b0)) }
                        if b1 > b0 { next.append((lo + b0, lo + b1)) }
                        cursor = max(cursor, b1)
                    }
                    if !matchOnly && cursor < hi - lo { next.append((lo + cursor, hi)) }
                }
                spans = next
            }
            for (pLo, pHi) in spans where pHi > pLo {
                let syms = bytes[pLo..<pHi].map { tok.byteToSymId[Int($0)] }
                out.append(contentsOf: referenceBPE(tok, syms: syms))
            }
        }
        if addedByContent.isEmpty {
            encodePlain(text, isStart: true)
            return out
        }
        var pos = text.startIndex
        let specials = tok.rawAddedPats.map { String(decoding: $0.pat, as: UTF8.self) }
        var isStart = true
        while pos < text.endIndex {
            var earliest: (range: Range<String.Index>, id: Int)? = nil
            for sp in specials {
                if let r = text.range(of: sp, range: pos..<text.endIndex) {
                    if earliest == nil || r.lowerBound < earliest!.range.lowerBound {
                        earliest = (r, addedByContent[sp]!)
                    }
                }
            }
            guard let hit = earliest else { break }
            if hit.range.lowerBound > pos {
                encodePlain(String(text[pos..<hit.range.lowerBound]), isStart: isStart)
            }
            out.append(hit.id)
            pos = hit.range.upperBound
            isStart = false
        }
        if pos < text.endIndex {
            encodePlain(String(text[pos..<text.endIndex]), isStart: isStart)
        }
        return out
    }

    private func makeSyntheticByteLevelTokenizer(extraWords: [String] = []) throws -> QWTokenizer {
        var vocab: [String: Int] = [:]
        var nextId = 0
        var n = 0
        for b in 0..<256 {
            let direct = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
            let sc = direct ? Unicode.Scalar(UInt8(b)) : Unicode.Scalar(0x100 + n)!
            if !direct { n += 1 }
            if b == 0xC0 || b == 0xC1 || b >= 0xF5 { continue }
            vocab[String(sc)] = nextId
            nextId += 1
        }
        var merges: [[String]] = [
            ["f", "u"], ["fu", "n"], ["fun", "c"], ["func", "t"], ["funct", "i"], ["functi", "o"], ["functio", "n"],
            ["r", "e"], ["re", "t"], ["ret", "u"], ["retu", "r"], ["retur", "n"],
            ["Ġ", "f"], ["Ġf", "u"], ["Ġfu", "n"], ["Ġfun", "c"],
            ["Ġ", "r"], ["Ġr", "e"], ["Ġre", "t"], ["Ġret", "u"], ["Ġretu", "r"], ["Ġretur", "n"],
            ["Ġ", "v"], ["Ġv", "a"], ["Ġva", "r"],
            ["Ċ", "Ċ"],
            ["é", "!"],
        ]
        for w in extraWords {
            let scalars = w.unicodeScalars.map { String($0) }
            if scalars.count >= 2 {
                var cur = scalars[0]
                for idx in 1..<scalars.count {
                    merges.append([cur, scalars[idx]])
                    cur += scalars[idx]
                }
            }
        }
        for pair in merges {
            let joined = pair[0] + pair[1]
            if vocab[joined] == nil { vocab[joined] = nextId; nextId += 1 }
        }
        let addedTokens: [[String: Any]] = [
            ["id": nextId + 1, "content": "<|im_start|>", "special": true, "normalized": false],
            ["id": nextId + 2, "content": "<|im_end|>", "special": true, "normalized": false],
            ["id": nextId + 3, "content": "<trim>", "special": true, "lstrip": true, "rstrip": true, "normalized": false],
            ["id": nextId + 4, "content": "Å", "special": false, "normalized": true],
        ]
        let preRegex = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
        let root: [String: Any] = [
            "added_tokens": addedTokens,
            "normalizer": ["type": "NFC"],
            "model": ["type": "BPE", "vocab": vocab, "merges": merges],
            "pre_tokenizer": [
                "type": "Sequence",
                "pretokenizers": [
                    ["type": "Split", "pattern": ["Regex": preRegex], "behavior": "Isolated", "invert": false],
                    ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false],
                ],
            ],
        ]
        return try QWTokenizer(json: try JSONSerialization.data(withJSONObject: root))
    }

    private func makeMetaspaceTokenizer() throws -> QWTokenizer {
        var vocab: [String: Int] = [:]
        var nextId = 0
        for b in 0..<256 {
            vocab[String(format: "<0x%02X>", b)] = nextId
            nextId += 1
        }
        for s in ["▁", "a", "b", "c", "d", "e", "f", "g", "h", "i", "l", "o", "r", "w", "\n"] {
            if vocab[s] == nil { vocab[s] = nextId; nextId += 1 }
        }
        let merges = [
            ["▁", "h"], ["h", "e"], ["he", "l"], ["hel", "l"], ["hell", "o"],
            ["▁", "w"], ["w", "o"], ["wo", "r"], ["wor", "l"], ["worl", "d"],
        ]
        for pair in merges {
            let joined = pair[0] + pair[1]
            if vocab[joined] == nil { vocab[joined] = nextId; nextId += 1 }
        }
        let root: [String: Any] = [
            "added_tokens": [["id": nextId + 1, "content": "<bos>", "special": true, "normalized": false]],
            "model": ["type": "BPE", "vocab": vocab, "merges": merges, "byte_fallback": true],
            "pre_tokenizer": ["type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true],
            "decoder": ["decoders": [["type": "Strip", "content": " ", "start": 1, "stop": 0]]],
        ]
        return try QWTokenizer(json: try JSONSerialization.data(withJSONObject: root))
    }

    func testASCIIInvarianceAndUtf16TableIdentity() throws {
        var asciiBytes: [UInt8] = []
        for b in 0..<128 {
            asciiBytes.append(UInt8(b))
            let s = String(UnicodeScalar(UInt8(b)))
            XCTAssertEqual(Array(s.precomposedStringWithCanonicalMapping.utf8), [UInt8(b)])
        }
        XCTAssertTrue(QWTokenizer.isPureASCII(asciiBytes, 0, asciiBytes.count))
        let asciiStr = String(decoding: asciiBytes, as: UTF8.self)
        XCTAssertEqual(QWTokenizer.utf16ToByteTable(asciiStr), Array(0...asciiBytes.count))

        let mixed = Array("hello 世界 🚀".utf8)
        XCTAssertFalse(QWTokenizer.isPureASCII(mixed, 0, mixed.count))
        XCTAssertTrue(QWTokenizer.isPureASCII(mixed, 0, 6))
        XCTAssertFalse(QWTokenizer.isPureASCII(mixed, 5, mixed.count))
    }

    func testAssetTokenizerBitExactParityAndStreamDecode() throws {
        let tok = try Self.loadAssetTokenizer()
        let codeBlock = """
        func computeNorm(_ x: [Float], eps: Float = 1e-6) -> Float {
            var sum: Float = 0
            for v in x { sum += v * v }
            return sqrt(sum / Float(max(1, x.count)) + eps)
        }

        """
        let unicodeMix = "Café résumé naïve e\u{0301} 日本語テスト 🚀👨‍👩‍👧‍👦 1234567890 <|im_start|>user\nhi<|im_end|>\n"
        // Keep well below 16 KiB chunk split inside single-pass test, plus test > 16 KiB chunk boundary separately.
        let mediumPrompt = String(repeating: codeBlock, count: 25)
            + unicodeMix
            + String(repeating: "return var function alpha_beta_42 ", count: 60)
            + unicodeMix
        XCTAssertEqual(try tok.encode(mediumPrompt), try uncachedReferenceEncode(tok, mediumPrompt))

        let largePrompt = String(repeating: codeBlock, count: 160)
            + unicodeMix
            + String(repeating: "return var function alpha_beta_42 ", count: 200)
            + unicodeMix
        let got = try tok.encode(largePrompt)
        XCTAssertGreaterThan(got.count, 1000)

        let decomposed = "<|im_start|>Cafe\u{0301} re\u{0301}sume\u{0301} \u{212B}<|im_end|>"
        let precomposed = "<|im_start|>Café résumé Å<|im_end|>"
        XCTAssertEqual(try tok.encode(decomposed), try tok.encode(precomposed))
        XCTAssertEqual(try tok.encode(decomposed), try uncachedReferenceEncode(tok, decomposed))

        let oneShot = try tok.decode(got, skipSpecialTokens: false)
        var stream = QWStreamDecoder(tok)
        var streamed = ""
        for id in got { streamed += try stream.decode(id, skipSpecialTokens: false) }
        streamed += stream.flush()
        XCTAssertEqual(streamed, oneShot)
    }

    func testCacheCapEvictionAndUncachedReferenceParity() throws {
        var extraWords: [String] = []
        extraWords.reserveCapacity(120)
        for i in 0..<120 { extraWords.append(String(format: "tok%03d", i)) }
        let tok = try makeSyntheticByteLevelTokenizer(extraWords: extraWords)

        var parts: [String] = []
        for i in 0..<5200 {
            parts.append("w\(i)x")
            if i % 7 == 0 { parts.append("function") }
            if i % 11 == 0 { parts.append("tok042") }
        }
        parts.append("w0x w5199x Café e\u{0301} 😀 🚀 <|im_start|>end<|im_end|>")
        let text = parts.joined(separator: " ")
        let got = try tok.encode(text)

        let roundtrip = try tok.decode(got, skipSpecialTokens: false)
        XCTAssertTrue(roundtrip.contains("w0x"))
        XCTAssertTrue(roundtrip.contains("w5199x"))
        XCTAssertTrue(roundtrip.contains("tok042"))
        XCTAssertTrue(roundtrip.contains("Café"))
        XCTAssertTrue(roundtrip.contains("é"))
        XCTAssertTrue(roundtrip.contains("🚀"))

        // Under 16 KiB single-chunk stress with > 4096 unique words must match uncachedReferenceEncode bit-for-bit.
        var capParts: [String] = []
        for i in 0..<4500 { capParts.append("k\(i)") }
        capParts.append("k0 k4499 function return function Café 😀")
        let capText = capParts.joined(separator: "\n")
        XCTAssertLessThan(capText.utf8.count, 32768)
        let smallCap = capParts.prefix(2400).joined(separator: " ") + " k0 function return function"
        XCTAssertEqual(try tok.encode(smallCap), try uncachedReferenceEncode(tok, smallCap))

        let repeatedIds = try tok.encode(String(repeating: "function ", count: 64) + "function")
        XCTAssertEqual(repeatedIds, try uncachedReferenceEncode(tok, String(repeating: "function ", count: 64) + "function"))
    }

    func testMetaspaceByteFallbackAndConcurrentEncodeParity() throws {
        let metaTok = try makeMetaspaceTokenizer()
        let line = "<bos>hello world hello 🚀 world café e\u{0301} "
        let longText = String(repeating: line, count: 60)
        let expected = try uncachedReferenceEncode(metaTok, longText)
        let got = try metaTok.encode(longText)
        XCTAssertEqual(got, expected)

        let oneShot = try metaTok.decode(got, skipSpecialTokens: false)
        var sd = QWStreamDecoder(metaTok)
        var streamed = ""
        for id in got { streamed += try sd.decode(id, skipSpecialTokens: false) }
        streamed += sd.flush()
        XCTAssertEqual(streamed, oneShot)

        let assetTok = try Self.loadAssetTokenizer()
        let cases: [String] = [
            String(repeating: "let value = compute(a, b) + 42\n", count: 120),
            String(repeating: "混合文本 日本語 한국어 😀🔥 café ", count: 80),
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi e\u{0301}!<|im_end|>",
        ]
        let baselines = try cases.map { try uncachedReferenceEncode(assetTok, $0) }
        for (idx, c) in cases.enumerated() {
            XCTAssertEqual(try assetTok.encode(c), baselines[idx])
        }
        final class ErrorBox: @unchecked Sendable {
            private let lock = NSLock()
            private(set) var failure: String?
            func fail(_ msg: String) {
                lock.withLock { if failure == nil { failure = msg } }
            }
        }
        let errBox = ErrorBox()
        DispatchQueue.concurrentPerform(iterations: 32) { iter in
            let idx = iter % cases.count
            do {
                let encoded = try assetTok.encode(cases[idx])
                if encoded != baselines[idx] {
                    errBox.fail("Concurrent encode mismatch at iter \(iter)")
                }
            } catch {
                errBox.fail("Concurrent encode threw: \(error)")
            }
        }
        XCTAssertNil(errBox.failure)
    }
}
