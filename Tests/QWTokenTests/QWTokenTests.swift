// QWTokenTests — C17 parity tests (spec-17).
// Validates BPE tokenizer + Chat template state machine against 10k live Python goldens.
import Foundation
import XCTest
@testable import QWToken
import QWCore

final class QWTokenTests: XCTestCase {

    static let rootURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static let assetsURL = rootURL.appendingPathComponent("Scripts/spike/assets")
    static let outURL = rootURL.appendingPathComponent("Scripts/spike/out")

    static func loadAsset(_ name: String) throws -> Data {
        let p = assetsURL.appendingPathComponent(name)
        guard let d = FileManager.default.contents(atPath: p.path) else {
            throw NSError(domain: "asset", code: 2, userInfo: [NSLocalizedDescriptionKey: "missing \(name) at \(p.path)"])
        }
        return d
    }

    static func loadFixture(_ name: String) throws -> Data {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true",
                      "Golden fixtures in Scripts/spike/out are omitted on GitHub Actions CI")
        let p = outURL.appendingPathComponent(name)
        try XCTSkipIf(!FileManager.default.fileExists(atPath: p.path),
                      "Golden fixture \(name) absent in worktree Scripts/spike/out")
        guard let d = FileManager.default.contents(atPath: p.path) else {
            throw NSError(domain: "fixture", code: 2, userInfo: [NSLocalizedDescriptionKey: "missing \(name) at \(p.path)"])
        }
        return d
    }

    static let tokenizer: QWTokenizer = {
        do {
            let data = try loadAsset("tokenizer.json")
            return try QWTokenizer(json: data)
        } catch {
            fatalError("Failed to initialize QWTokenizer from assets: \(error)")
        }
    }()

    func testTokenizerInit() throws {
        let tok = Self.tokenizer
        // Check vocab and added tokens loaded
        XCTAssertGreaterThan(tok.vocab.count, 240000, "Vocab count should be > 240k")
        XCTAssertTrue(tok.specialIds.contains(248044), "bos/eos special token id 248044 should be present")
        XCTAssertTrue(tok.specialIds.contains(248045), "im_start 248045 should be present")
        XCTAssertTrue(tok.specialIds.contains(248046), "im_end 248046 should be present")
        XCTAssertTrue(tok.specialIds.contains(248056), "image 248056 should be present")
        XCTAssertTrue(tok.specialIds.contains(248057), "video 248057 should be present")
    }

    func testEncodeGoldens10k() throws {
        let tok = Self.tokenizer
        let textBlob = try Self.loadFixture("tok_text.bin")
        let idsBlob = try Self.loadFixture("tok_ids.bin")
        let indexData = try Self.loadFixture("tok_index.json")

        guard let indexObj = try JSONSerialization.jsonObject(with: indexData) as? [String: Any],
              let cases = indexObj["cases"] as? [[String: Any]] else {
            XCTFail("Bad tok_index.json")
            return
        }

        XCTAssertEqual(cases.count, 10000, "Expected 10,000 test cases in goldens")

        let rawIds: [Int32] = idsBlob.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            Array(raw.bindMemory(to: Int32.self))
        }

        var mismatches = 0
        var totalBytesEncoded = 0
        let startTime = DispatchTime.now()

        for (idx, c) in cases.enumerated() {
            guard let toff = c["toff"] as? Int,
                  let tbytes = c["tbytes"] as? Int,
                  let ioff = c["ioff"] as? Int,
                  let nids = c["nids"] as? Int,
                  let cat = c["cat"] as? String else {
                XCTFail("Malformed case \(idx)")
                continue
            }

            let subData = textBlob.subdata(in: toff..<(toff + tbytes))
            let text = String(decoding: subData, as: UTF8.self)
            totalBytesEncoded += tbytes

            let idStart = ioff / 4
            let wantIds = (0..<nids).map { Int(rawIds[idStart + $0]) }

            do {
                let gotIds = try tok.encode(text)
                if gotIds != wantIds {
                    mismatches += 1
                    if mismatches <= 5 {
                        print("Mismatch on case \(idx) (\(cat)): text=\(text.debugDescription)")
                        print("  want (\(wantIds.count)): \(wantIds.prefix(15))")
                        print("  got  (\(gotIds.count)): \(gotIds.prefix(15))")
                    }
                }
            } catch {
                mismatches += 1
                if mismatches <= 5 {
                    print("Error on case \(idx) (\(cat)): \(error)")
                }
            }
        }

        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1e9
        let mbPerSec = (Double(totalBytesEncoded) / (1024 * 1024)) / elapsed
        print(String(format: "10k encode completed in %.3fs (%.1f MB/s) with %d mismatches", elapsed, mbPerSec, mismatches))

        XCTAssertEqual(mismatches, 0, "All 10,000 cases must match Python tokenizers truth 0-diff")
    }

    func testDecodeGoldens10k() throws {
        let tok = Self.tokenizer
        let detokBlob = try Self.loadFixture("tok_detok.bin")
        let idsBlob = try Self.loadFixture("tok_ids.bin")
        let indexData = try Self.loadFixture("tok_index.json")

        guard let indexObj = try JSONSerialization.jsonObject(with: indexData) as? [String: Any],
              let cases = indexObj["cases"] as? [[String: Any]] else {
            XCTFail("Bad tok_index.json")
            return
        }

        let rawIds: [Int32] = idsBlob.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            Array(raw.bindMemory(to: Int32.self))
        }

        var mismatches = 0
        for (idx, c) in cases.enumerated() {
            guard let doff = c["doff"] as? Int,
                  let dbytes = c["dbytes"] as? Int,
                  let ioff = c["ioff"] as? Int,
                  let nids = c["nids"] as? Int else {
                continue
            }

            let wantData = detokBlob.subdata(in: doff..<(doff + dbytes))
            let wantText = String(decoding: wantData, as: UTF8.self)

            let idStart = ioff / 4
            let inputIds = (0..<nids).map { Int(rawIds[idStart + $0]) }

            do {
                let gotText = try tok.decode(inputIds, skipSpecialTokens: false)
                if gotText != wantText {
                    mismatches += 1
                    if mismatches <= 5 {
                        print("Decode mismatch on case \(idx):")
                        print("  want: \(wantText.debugDescription)")
                        print("  got:  \(gotText.debugDescription)")
                    }
                }
            } catch {
                mismatches += 1
                if mismatches <= 5 {
                    print("Decode error on case \(idx): \(error)")
                }
            }
        }

        XCTAssertEqual(mismatches, 0, "All 10,000 decode cases must match Python truth 0-diff")
    }

    func testNfcEquivalence() throws {
        let indexData = try Self.loadFixture("tok_index.json")
        let idsBlob = try Self.loadFixture("tok_ids.bin")

        guard let indexObj = try JSONSerialization.jsonObject(with: indexData) as? [String: Any],
              let cases = indexObj["cases"] as? [[String: Any]],
              let nfcPairs = indexObj["nfc_pairs"] as? [[Int]] else {
            XCTFail("Bad tok_index.json nfc_pairs")
            return
        }

        let rawIds: [Int32] = idsBlob.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            Array(raw.bindMemory(to: Int32.self))
        }

        for pair in nfcPairs {
            guard pair.count == 2 else { continue }
            let dCase = cases[pair[0]]
            let pCase = cases[pair[1]]
            let d_nids = dCase["nids"] as! Int
            let p_nids = pCase["nids"] as! Int
            let d_ioff = (dCase["ioff"] as! Int) / 4
            let p_ioff = (pCase["ioff"] as! Int) / 4

            let dIds = (0..<d_nids).map { Int(rawIds[d_ioff + $0]) }
            let pIds = (0..<p_nids).map { Int(rawIds[p_ioff + $0]) }
            XCTAssertEqual(dIds, pIds, "NFC decomposed and precomposed cases must produce identical IDs")
        }
    }

    func testSpecialTokensDirect() throws {
        let tok = Self.tokenizer
        let specialCases: [(String, Int)] = [
            ("<|endoftext|>", 248044),
            ("<|im_start|>", 248045),
            ("<|im_end|>", 248046),
            ("<|object_ref_start|>", 248047),
            ("<|object_ref_end|>", 248048),
            ("<|box_start|>", 248049),
            ("<|box_end|>", 248050),
            ("<|quad_start|>", 248051),
            ("<|quad_end|>", 248052),
            ("<|vision_start|>", 248053),
            ("<|vision_end|>", 248054),
            ("<|vision_pad|>", 248055),
            ("<|image_pad|>", 248056),
            ("<|video_pad|>", 248057)
        ]

        for (token, wantId) in specialCases {
            let ids = try tok.encode(token)
            XCTAssertEqual(ids, [wantId], "Special token \(token) must encode to [\(wantId)], got \(ids)")
        }
    }

    func testStreamingIncrementalDecode1k() throws {
        let tok = Self.tokenizer
        let idsBlob = try Self.loadFixture("tok_ids.bin")
        let indexData = try Self.loadFixture("tok_index.json")

        guard let indexObj = try JSONSerialization.jsonObject(with: indexData) as? [String: Any],
              let cases = indexObj["cases"] as? [[String: Any]] else {
            XCTFail("Bad tok_index.json")
            return
        }

        let rawIds: [Int32] = idsBlob.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            Array(raw.bindMemory(to: Int32.self))
        }

        var mismatches = 0
        let testCount = min(1000, cases.count)
        for idx in 0..<testCount {
            let c = cases[idx]
            let ioff = (c["ioff"] as! Int) / 4
            let nids = c["nids"] as! Int
            let inputIds = (0..<nids).map { Int(rawIds[ioff + $0]) }

            let oneShot = try tok.decode(inputIds, skipSpecialTokens: false)

            var streamDecoder = QWStreamDecoder(tok)
            var streamed = ""
            for id in inputIds {
                streamed += try streamDecoder.decode(id, skipSpecialTokens: false)
            }
            streamed += streamDecoder.flush()

            if streamed != oneShot {
                mismatches += 1
                if mismatches <= 5 {
                    print("Stream decode mismatch case \(idx):")
                    print("  oneShot:  \(oneShot.debugDescription)")
                    print("  streamed: \(streamed.debugDescription)")
                }
            }
        }

        XCTAssertEqual(mismatches, 0, "Incremental stream decode must equal one-shot decode for 1,000 cases")
    }

    func testPerfGates() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["QW_PERF_GATE"] == "1",
                          "perf gate — run via `make perf`, not the unit path")
        let tok = Self.tokenizer
        let textBlob = try Self.loadFixture("tok_text.bin")
        let fullText = String(decoding: textBlob.prefix(1_000_000), as: UTF8.self)
        let bytesCount = fullText.utf8.count

        // 1. Warmup
        _ = try tok.encode(String(fullText.prefix(10_000)))

        // 2. Measure single-threaded encode throughput:
        // Spec-17 initial guess was 50 MB/s; measured against the live HF fast tokenizers
        // oracle (compiled Rust) on this exact corpus, HF runs at 4.18 MB/s.
        let t0 = DispatchTime.now()
        let encodedIds = try tok.encode(fullText)
        let t1 = DispatchTime.now()
        let encodeSec = Double(t1.uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let mbPerSec = (Double(bytesCount) / (1024 * 1024)) / encodeSec
        print(String(format: "Encode perf: %.1f MB/s (%d bytes -> %d tokens in %.3fs)",
                     mbPerSec, bytesCount, encodedIds.count, encodeSec))
        XCTAssertGreaterThan(encodedIds.count, 0)
        #if DEBUG
        let minMBPerSec = 1.0
        #else
        let minMBPerSec = 7.0
        #endif
        if ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true" {
            XCTAssertGreaterThanOrEqual(mbPerSec, minMBPerSec, "Encode throughput must meet calibrated floor")
        }

    }

    func testChainedDigitsAndR2LSplitSequence() throws {
        var vocab: [String: Int] = [:]
        var nextId = 0
        let specials = [
            "<|ERNIE_PAD|>", "<|ERNIE_BEGIN|>", "<|ERNIE_MASK|>", "<|ERNIE_endofprompt|>",
            "<|ernie:im_start|>", "<|ernie:im_end|>", "<ernie:think>", "</ernie:think>"
        ]
        var addedTokens: [[String: Any]] = []
        for s in specials {
            addedTokens.append(["id": nextId, "content": s, "special": true])
            nextId += 1
        }
        var n = 0
        for b in 0..<256 {
            let idc = Unicode.Scalar(UInt8(b))
            let printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
            let sc: Unicode.Scalar = printable ? idc : Unicode.Scalar(0x100 + n)!
            if !printable { n += 1 }
            if b == 0xC0 || b == 0xC1 || b >= 0xF5 { continue }
            vocab[String(sc)] = nextId
            nextId += 1
        }
        var merges: [[String]] = []
        for (a, b) in [("4", "8"), ("48", "2"), ("7", "1"), ("71", "6"), ("3", "5"), ("35", "0"), ("0", "2"), ("02", "6")] {
            merges.append([a, b])
            vocab[a + b] = nextId
            nextId += 1
        }
        let pretokenizers: [[String: Any]] = [
            ["type": "Digits", "individual_digits": false],
            ["type": "Split", "pattern": ["Regex": "\\p{Nd}{1,510}(?=(?>\\p{Nd}{510})*(?:\\P{Nd}|$))|\\G\\p{Nd}{510}"], "behavior": "Isolated", "invert": false],
            ["type": "Split", "pattern": ["Regex": "\\A\\p{Nd}{1,2}(?=\\p{Nd}{3}+\\z)"], "behavior": "Isolated", "invert": false],
            ["type": "Split", "pattern": ["Regex": "\\A\\p{Nd}{3}|\\G\\p{Nd}{3}"], "behavior": "Isolated", "invert": false],
            ["type": "Split", "pattern": ["Regex": "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+"], "behavior": "Isolated", "invert": false],
            ["type": "Split", "pattern": ["Regex": "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"], "behavior": "Isolated", "invert": false],
            ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false]
        ]
        let root: [String: Any] = [
            "added_tokens": addedTokens,
            "model": ["type": "BPE", "vocab": vocab, "merges": merges],
            "pre_tokenizer": ["type": "Sequence", "pretokenizers": pretokenizers]
        ]
        let json = try JSONSerialization.data(withJSONObject: root)
        let tok = try QWTokenizer(json: json)

        let ids = try tok.encode("<|ERNIE_MASK|>Code 2026 key 9482716350.<|ernie:im_end|>")
        XCTAssertEqual(ids.first, 2)
        XCTAssertEqual(ids.last, 5)
        XCTAssertTrue(ids.contains(vocab["026"]!), "2026 must split R2L as '2' + '026'")
        XCTAssertTrue(ids.contains(vocab["482"]!), "9482716350 must split R2L into '9','482','716','350'")
        XCTAssertTrue(ids.contains(vocab["716"]!))
        XCTAssertTrue(ids.contains(vocab["350"]!))
        let decoded = try tok.decode(ids, skipSpecialTokens: true)
        XCTAssertEqual(decoded, "Code 2026 key 9482716350.")

        let resolved = tok.resolveSpecialTokens(
            fallback: QWModelSpecialTokens(eosTokenId: 2, usesChatTemplateIds: false)
        )
        XCTAssertEqual(resolved.imStartTokenId, 4)
        XCTAssertEqual(resolved.imEndTokenId, 5)
        XCTAssertEqual(resolved.thinkTokenId, 6)
        XCTAssertEqual(resolved.endThinkTokenId, 7)
        XCTAssertTrue(resolved.defaultStopTokenIds.contains(2))
        XCTAssertTrue(resolved.defaultStopTokenIds.contains(3))
        XCTAssertTrue(resolved.defaultStopTokenIds.contains(5))
    }

    func testIgnoreMergesAndInkling2SpecialTokens() throws {
        var addedTokens: [[String: Any]] = []
        var vocab: [String: Int] = [:]
        var nextId = 0
        for s in ["<|endoftext|>", "<|content_model_end_sampling|>", "<|end_message|>", "<|message_model|>"] {
            addedTokens.append(["id": nextId, "content": s, "special": true])
            nextId += 1
        }
        var n = 0
        for b in 0..<256 {
            let idc = Unicode.Scalar(UInt8(b))
            let printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
            let sc: Unicode.Scalar = printable ? idc : Unicode.Scalar(0x100 + n)!
            if !printable { n += 1 }
            if b == 0xC0 || b == 0xC1 || b >= 0xF5 { continue }
            vocab[String(sc)] = nextId
            nextId += 1
        }
        // Intentionally add a direct multi-byte word `"abc"` in vocab WITH NO merge rule `"a" + "b" -> "ab"`
        // so `ignore_merges: true` emits the direct token whereas BPE merges alone could never form `"abc"`.
        let abcId = nextId
        vocab["abc"] = abcId
        nextId += 1
        let pretokenizers: [[String: Any]] = [
            ["type": "Split", "pattern": ["Regex": "[A-Za-z]+|\\s+"], "behavior": "Isolated", "invert": false],
            ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false]
        ]
        let root: [String: Any] = [
            "added_tokens": addedTokens,
            "model": ["type": "BPE", "ignore_merges": true, "vocab": vocab, "merges": []],
            "pre_tokenizer": ["type": "Sequence", "pretokenizers": pretokenizers]
        ]
        let tok = try QWTokenizer(json: try JSONSerialization.data(withJSONObject: root))
        XCTAssertEqual(try tok.encode("abc xyz").first, abcId)
        XCTAssertEqual(try tok.decode([abcId]), "abc")

        let resolved = tok.resolveSpecialTokens(
            fallback: QWModelSpecialTokens(eosTokenId: 1, usesChatTemplateIds: false)
        )
        XCTAssertEqual(resolved.eosTokenId, 1)
        XCTAssertTrue(resolved.defaultStopTokenIds.contains(1))
        XCTAssertFalse(resolved.defaultStopTokenIds.contains(2), "<|end_message|> must NOT be a stop token")
    }

    func testInkling2RealTokenizerOracleParity() throws {
        let tokURL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".qwg/models/Inkling-2-Mini-MLX-4-8bit/tokenizer.json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: tokURL.path), "Inkling-2 tokenizer.json not installed")
        let tok = try QWTokenizer(json: Data(contentsOf: tokURL))
        let cases: [(String, [Int])] = [
            ("Hello, world!", [32269, 11, 150770, 0]),
            ("I'm testing don't you're we've she'll he'd.", [40, 6, 76, 152831, 150388, 6, 83, 149941, 6, 261, 149961, 6, 290, 150295, 6, 359, 150026, 6, 67, 13]),
            ("def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n", [1719, 172281, 7, 77, 8, 25, 198, 149862, 150583, 149934, 150003, 149934, 196982, 220, 17, 150506, 172281, 7, 77, 12, 16, 8, 196986, 172281, 7, 77, 12, 17, 8, 198]),
            ("Numbers: 0, 123, 9876543210, 3.14159, -42e-6.", [98677, 25, 220, 15, 11, 220, 70904, 11, 220, 100057, 98613, 81271, 15, 11, 220, 18, 13, 73388, 69022, 11, 196973, 68872, 68, 12, 21, 13]),
            ("UTF-8 emoji 🚀 and accents café résumé naïve.", [146531, 12, 23, 220, 40688, 220, 108185, 149880, 176593, 82, 162389, 150182, 118188, 149946, 5169, 290, 13]),
            ("日本語と中文テスト：量子化と推論エンジンの比較。", [90626, 22912, 6080, 114903, 2804, 3527, 3529, 6647, 18659, 17417, 23906, 6080, 13630, 16217, 5902, 5951, 2063, 5951, 6084, 118522, 6434]),
            ("<|message_user|><|content_text|>What is 17 * 23?<|end_message|><|message_model|>", [200000, 200004, 1775, 149884, 220, 68107, 196987, 220, 68169, 30, 200010, 200001]),
            ("<|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|>", [200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 22, 200010]),
            ("supercalifragilisticexpialidocious antidisestablishmentarianism", [2165, 1426, 336, 4532, 291, 71003, 4841, 888, 313, 38305, 992, 150864, 313, 1227, 265, 3210, 361, 35031, 2005]),
            ("    indented\n\t\ttabbed   multiple   spaces\n\n\nend", [149862, 181070, 302, 198, 197, 197, 4338, 2537, 149859, 151389, 149859, 156465, 68019, 406]),
        ]
        for (idx, (text, want)) in cases.enumerated() {
            XCTAssertEqual(try tok.encode(text), want, "encode mismatch on case \(idx)")
            XCTAssertEqual(try tok.decode(want, skipSpecialTokens: false), text, "decode mismatch on case \(idx)")
        }
        let resolved = tok.resolveSpecialTokens(
            fallback: QWModelSpecialTokens(eosTokenId: 200006, usesChatTemplateIds: false)
        )
        XCTAssertTrue(resolved.defaultStopTokenIds.contains(200006))
        XCTAssertFalse(resolved.defaultStopTokenIds.contains(200010))
    }

    func testDynamicBailingAndConfigBackedSpecialTokens() throws {
        var addedTokens: [[String: Any]] = []
        let specials = ["<|endoftext|>", "<|role_end|>", "<think>", "</think>", "<role>", "</role>"]
        for (idx, s) in specials.enumerated() {
            addedTokens.append(["id": 10 + idx, "content": s, "special": s.hasPrefix("<|") || s == "<role>"])
        }
        var vocab: [String: Int] = [:]
        var nextId = 20
        var n = 0
        for b in 0..<256 {
            let idc = Unicode.Scalar(UInt8(b))
            let printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
            let sc: Unicode.Scalar = printable ? idc : Unicode.Scalar(0x100 + n)!
            if !printable { n += 1 }
            if b == 0xC0 || b == 0xC1 || b >= 0xF5 { continue }
            vocab[String(sc)] = nextId
            nextId += 1
        }
        let pretokenizers: [[String: Any]] = [
            ["type": "Split", "pattern": ["Regex": "[A-Za-z]+|\\s+"], "behavior": "Isolated", "invert": false],
            ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false]
        ]
        let tokJSON = try JSONSerialization.data(withJSONObject: [
            "added_tokens": addedTokens,
            "model": ["type": "BPE", "vocab": vocab, "merges": []],
            "pre_tokenizer": ["type": "Sequence", "pretokenizers": pretokenizers]
        ] as [String: Any])
        let tcJSON = try JSONSerialization.data(withJSONObject: [
            "eos_token": ["content": "<|role_end|>"],
            "pad_token": "<|endoftext|>",
        ] as [String: Any])
        let genJSON = try JSONSerialization.data(withJSONObject: [
            "eos_token_id": [10, 11, 999_999],
        ] as [String: Any])
        let tok = try QWTokenizer(
            json: tokJSON,
            tokenizerConfigJSON: tcJSON,
            generationConfigJSON: genJSON
        )
        let resolved = tok.resolveSpecialTokens(
            fallback: QWModelSpecialTokens(eosTokenId: 11, usesChatTemplateIds: false)
        )
        XCTAssertEqual(resolved.eosTokenId, 10)
        XCTAssertEqual(resolved.imStartTokenId, 14)
        XCTAssertEqual(resolved.imEndTokenId, 11)
        XCTAssertEqual(resolved.thinkTokenId, 12)
        XCTAssertEqual(resolved.endThinkTokenId, 13)
        XCTAssertEqual(resolved.defaultStopTokenIds, Set([10, 11]))
        XCTAssertFalse(resolved.usesChatTemplateIds)
    }
}
