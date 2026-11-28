// QWChatTemplateGoldenTests.swift — C17 chat template goldens against Python Jinja2 fixtures.
import XCTest
import QWCore
@testable import QWServer
import QWToken

final class QWChatTemplateGoldenTests: XCTestCase {
    private static let rootURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    private static let assetsURL = rootURL.appendingPathComponent("Scripts/spike/assets")
    private static let outURL = rootURL.appendingPathComponent("Scripts/spike/out")

    private static func loadAsset(_ name: String) throws -> Data {
        let p = assetsURL.appendingPathComponent(name)
        guard let d = FileManager.default.contents(atPath: p.path) else {
            throw NSError(domain: "asset", code: 2, userInfo: [NSLocalizedDescriptionKey: "missing \(name) at \(p.path)"])
        }
        return d
    }

    private static func loadFixture(_ name: String) throws -> Data {
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

    func testChatTemplateGoldens() throws {
        let tok = try QWTokenizer(json: try Self.loadAsset("tokenizer.json"))
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

        let rawChatIds: [Int32] = chatIdsBlob.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            Array(raw.bindMemory(to: Int32.self))
        }

        guard let jsonlString = String(data: chatJsonlData, encoding: .utf8) else {
            XCTFail("Bad tok_chat.jsonl string")
            return
        }
        let lines = jsonlString.components(separatedBy: "\n").filter {
            !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
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
                    if rendered == wantRender { passedRenders += 1 }
                    let gotIds = try tok.encode(rendered)
                    XCTAssertEqual(gotIds, wantIds, "Tokenized IDs mismatch on chat case \(idx)")
                } catch {
                    XCTFail("Chat case \(idx) threw unexpected error: \(error)")
                }
            }
        }

        XCTAssertEqual(passedRaises, 5)
        XCTAssertEqual(passedRenders, 19)
    }
}
