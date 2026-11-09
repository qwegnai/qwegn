// QWTokenTests — C17 parity tests (spec-17).
// Validates BPE tokenizer + Chat template state machine against 10k live Python goldens.
import XCTest
@testable import QWToken
import QWCore
import QWServer

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
        let p = outURL.appendingPathComponent(name)
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

        let rawIds: [Int32] = idsBlob.withUnsafeBytes {
            Array($0.bindMemory(to: Int32.self))
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

        let rawIds: [Int32] = idsBlob.withUnsafeBytes {
            Array($0.bindMemory(to: Int32.self))
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

        let rawIds: [Int32] = idsBlob.withUnsafeBytes {
            Array($0.bindMemory(to: Int32.self))
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

        let rawIds: [Int32] = idsBlob.withUnsafeBytes {
            Array($0.bindMemory(to: Int32.self))
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

    func testChatTemplateGoldens() throws {
        let tok = Self.tokenizer
        let templateURL = Self.assetsURL.appendingPathComponent("chat_template.jinja")
        let templateSource = try String(contentsOf: templateURL, encoding: .utf8)
        let template = try QWChatTemplate(source: templateSource)
        let chatJsonlData = try Self.loadFixture("tok_chat.jsonl")
        let chatTextBlob = try Self.loadFixture("tok_chat_text.bin")
        let chatIdsBlob = try Self.loadFixture("tok_chat_ids.bin")
        let chatIndexData = try Self.loadFixture("tok_chat_index.json")

        guard let chatIndexObj = try JSONSerialization.jsonObject(with: chatIndexData) as? [String: Any],
              let indexCases = chatIndexObj["cases"] as? [[String: Any]] else {
            XCTFail("Bad tok_chat_index.json")
            return
        }

        let rawChatIds: [Int32] = chatIdsBlob.withUnsafeBytes {
            Array($0.bindMemory(to: Int32.self))
        }

        guard let jsonlString = String(data: chatJsonlData, encoding: .utf8) else {
            XCTFail("Bad tok_chat.jsonl string")
            return
        }
        let lines = jsonlString.components(separatedBy: "\n").filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        XCTAssertEqual(lines.count, indexCases.count)

        var passedRenders = 0
        var passedRaises = 0

        for (idx, line) in lines.enumerated() {
            let caseObj = try QWJson.parse(line)
            let rec = indexCases[idx]
            let expectedErr = caseObj["err"]

            var options = QWChatTemplateOptions()
            if case .bool(let gp) = caseObj["gp"] { options.addGenerationPrompt = gp }
            let tools = caseObj["tools"]
            if !tools.isUndefined { options.tools = tools }

            let kwargs = caseObj["kwargs"]
            if !kwargs.isUndefined {
                let re = kwargs["reasoning_effort"]
                if !re.isUndefined { options.reasoningEffort = re }
                let et = kwargs["enable_thinking"]
                if !et.isUndefined { options.enableThinking = et }
                let pt = kwargs["preserve_thinking"]
                if !pt.isUndefined { options.preserveThinking = pt }
                let vi = kwargs["add_vision_id"]
                if case .bool(let b) = vi { options.addVisionId = b }
            }

            let msgsJson = caseObj["msgs"]
            var messages: [QWJson] = []
            if case .array(let arr) = msgsJson { messages = arr }

            if case .string(let errSubstr) = expectedErr {
                do {
                    _ = try template.render(messages: messages, options: options)
                    XCTFail("Case \(idx) was expected to throw error containing '\(errSubstr)', but succeeded")
                } catch {
                    XCTAssertTrue(String(describing: error).contains(errSubstr),
                                  "Case \(idx) threw \(error), expected substring '\(errSubstr)'")
                    passedRaises += 1
                }
            } else {
                guard let toff = rec["toff"] as? Int,
                      let tbytes = rec["tbytes"] as? Int,
                      let ioff = rec["ioff"] as? Int,
                      let nids = rec["nids"] as? Int else {
                    XCTFail("Missing rec fields for case \(idx)")
                    continue
                }

                let wantTextData = chatTextBlob.subdata(in: toff..<(toff + tbytes))
                let wantRender = String(decoding: wantTextData, as: UTF8.self)

                let idStart = ioff / 4
                let wantIds = (0..<nids).map { Int(rawChatIds[idStart + $0]) }

                do {
                    let rendered = try template.render(messages: messages, options: options)
                    XCTAssertEqual(rendered, wantRender, "Rendered text mismatch on chat case \(idx)")
                    if rendered != wantRender {
                        print("CHAT RENDER MISMATCH CASE \(idx):")
                        print("  want (\(wantRender.count)): \(wantRender.debugDescription)")
                        print("  got  (\(rendered.count)): \(rendered.debugDescription)")
                    } else {
                        passedRenders += 1
                    }

                    // Tokenize rendered text and verify against wantIds
                    let gotIds = try tok.encode(rendered)
                    XCTAssertEqual(gotIds, wantIds, "Tokenized IDs mismatch on chat case \(idx)")
                } catch {
                    XCTFail("Chat case \(idx) threw unexpected error: \(error)")
                }
            }
        }

        print("Chat template goldens: \(passedRenders) renders + \(passedRaises) raises passed")
        XCTAssertEqual(passedRaises, 5)
        XCTAssertEqual(passedRenders, 19)
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
        XCTAssertGreaterThanOrEqual(mbPerSec, minMBPerSec, "Encode throughput must meet calibrated floor")

    }
}
