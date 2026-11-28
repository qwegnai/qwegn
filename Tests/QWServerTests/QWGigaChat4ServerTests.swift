// QWGigaChat4ServerTests.swift — Tests reasoning kwargs, GCML XML tool calls, multi-token tool constraints, and role_sep prefix caching.
import Foundation
import XCTest
import QWCore
import QWHead
@testable import QWServer
import QWToken

final class QWGigaChat4ServerTests: XCTestCase {
    private static func makeSyntheticReasoningTemplate() -> String {
        let bo = "{" + "%"
        let bc = "%" + "}"
        let vo = "{" + "{"
        let vc = "}" + "}"
        return [
            "\(bo) if reasoning is not defined \(bc)\(bo) set reasoning = true \(bc)\(bo) endif \(bc)",
            "# <｜GCML｜tool_calls>",
            "\(vo) messages[0].content \(vc)",
            "\(bo) if add_generation_prompt \(bc)",
            "\(bo) if reasoning \(bc)<think>\(bo) else \(bc)<think></think>\n\n\(bo) endif \(bc)",
            "\(bo) endif \(bc)",
        ].joined()
    }

    func testChatTemplateKwargsReasoningThreadsThroughPolicyAndTemplate() throws {
        let reqOffJSON = Data("""
        {"model":"gigachat4","messages":[{"role":"user","content":"Hi"}],"chat_template_kwargs":{"reasoning":false}}
        """.utf8)
        let reqOff = try QWServerRequestDecoder.decode(ChatCompletionRequest.self, from: reqOffJSON)
        XCTAssertEqual(reqOff.chatTemplateKwargs?.reasoning, false)
        XCTAssertTrue(reqOff.hasExplicitThinkingSetting)
        XCTAssertFalse(QWChatPolicyResolver.effectiveEnableThinking(for: reqOff, defaultValue: true))

        let reqOnJSON = Data("""
        {"model":"gigachat4","messages":[{"role":"user","content":"Hi"}],"chat_template_kwargs":{"reasoning":true}}
        """.utf8)
        let reqOn = try QWServerRequestDecoder.decode(ChatCompletionRequest.self, from: reqOnJSON)
        XCTAssertEqual(reqOn.chatTemplateKwargs?.reasoning, true)
        XCTAssertTrue(QWChatPolicyResolver.effectiveEnableThinking(for: reqOn, defaultValue: false))

        let tpl = try QWChatTemplate(source: Self.makeSyntheticReasoningTemplate())
        XCTAssertEqual(tpl.toolCallEnvelope, .gcml)
        XCTAssertFalse(tpl.usesStandardToolCallHint)
        XCTAssertTrue(tpl.supportsThinkingTags)

        let msgs: [QWJson] = [.object([("role", .string("user")), ("content", .string("Ping"))])]
        var optsOff = QWChatTemplateOptions()
        optsOff.enableThinking = .bool(false)
        optsOff.reasoning = .bool(false)
        let renderedOff = try tpl.render(messages: msgs, options: optsOff)
        XCTAssertTrue(renderedOff.hasSuffix("<think></think>\n\n"))

        var optsOn = QWChatTemplateOptions()
        optsOn.enableThinking = .bool(true)
        optsOn.reasoning = .bool(true)
        let renderedOn = try tpl.render(messages: msgs, options: optsOn)
        XCTAssertTrue(renderedOn.hasSuffix("<think>"))
    }

    func testRealGigaChat4ChatTemplateRendersInstantReasoningAndToolsIfPresent() throws {
        let tplPath = NSHomeDirectory() + "/.cache/huggingface/hub/models--ai-sage--GigaChat4-37B-A4B/"
            + "snapshots/1bf9a4b2b8d7d0f970e48d6458be77d49e07bcfc/chat_template.jinja"
        try XCTSkipUnless(FileManager.default.fileExists(atPath: tplPath), "local GigaChat4 chat_template.jinja absent")

        let source = try String(contentsOfFile: tplPath, encoding: .utf8)
        let tpl = try QWChatTemplate(
            source: source,
            tokenizerContext: ["bos_token": .string("<s>"), "eos_token": .string("</s>")]
        )
        XCTAssertEqual(tpl.toolCallEnvelope, .gcml)
        XCTAssertTrue(tpl.supportsThinkingTags)

        let tool = try QWJson.parse("""
        {"type":"function","function":{"name":"get_weather","description":"Fetch forecast","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}}}
        """)
        let msgs: [QWJson] = [
            .object([("role", .string("system")), ("content", .string("Be concise."))]),
            .object([("role", .string("user")), ("content", .string("Weather in Moscow?"))]),
        ]

        var instantOpts = QWChatTemplateOptions()
        instantOpts.tools = .array([tool])
        instantOpts.enableThinking = .bool(false)
        instantOpts.reasoning = .bool(false)
        let instantWire = try tpl.render(messages: msgs, options: instantOpts)
        XCTAssertTrue(instantWire.hasPrefix("<s>system<|role_sep|>\nBe concise.<|message_sep|>\n\nfunction descriptions<|role_sep|>\n"))
        XCTAssertTrue(instantWire.contains("namespace functions"))
        XCTAssertTrue(instantWire.contains("get_weather"))
        XCTAssertTrue(instantWire.contains("city: string"))
        XCTAssertTrue(instantWire.hasSuffix("assistant<|role_sep|>\n<think></think>\n\n"))

        var thinkOpts = QWChatTemplateOptions()
        thinkOpts.tools = .array([tool])
        thinkOpts.enableThinking = .bool(true)
        thinkOpts.reasoning = .bool(true)
        let thinkWire = try tpl.render(messages: msgs, options: thinkOpts)
        XCTAssertTrue(thinkWire.hasSuffix("assistant<|role_sep|>\n<think>"))

        let multiTurnReq = ChatCompletionRequest(
            model: "gigachat4",
            messages: [
                ChatMessage(role: "system", text: "Be concise."),
                ChatMessage(role: "user", text: "Weather in Moscow?"),
                ChatMessage(
                    role: "assistant",
                    content: nil,
                    toolCalls: [
                        ToolCall(
                            id: "call_1",
                            type: "function",
                            function: ToolCallFunction(name: "get_weather", arguments: #"{"city":"Moscow","days":2}"#)
                        ),
                    ],
                    reasoningContent: "Need Moscow forecast."
                ),
                ChatMessage(role: "tool", content: ChatMessageContent(string: #"{"temp_c":18}"#), toolCallId: "call_1"),
            ],
            tools: [tool]
        )
        let multiTurnWire = try tpl.render(messages: multiTurnReq.toQWJsonMessages(), options: thinkOpts)
        XCTAssertTrue(multiTurnWire.contains("<｜GCML｜tool_calls>"))
        XCTAssertTrue(multiTurnWire.contains("<｜GCML｜invoke name=\"get_weather\">"))
        XCTAssertTrue(multiTurnWire.contains("<｜GCML｜parameter name=\"city\" string=\"true\">Moscow</｜GCML｜parameter>"))
        XCTAssertTrue(multiTurnWire.contains("<｜GCML｜parameter name=\"days\" string=\"false\">2</｜GCML｜parameter>"))
        XCTAssertTrue(multiTurnWire.hasSuffix("assistant<|role_sep|>\n<think>"))
    }

    func testGCMLToolCallParserAndStreamParserHonorStringAttribute() throws {
        let tool = try QWJson.parse("""
        {"type":"function","function":{"name":"run_query","parameters":{"type":"object","properties":{"limit":{"type":"integer"},"raw_id":{"type":"string"},"flag":{"type":"string"},"filter":{"type":"object"}}}}}
        """)
        let sample = "<think>Check query args.</think>Running query."
            + "<｜GCML｜tool_calls>"
            + "<｜GCML｜invoke name=\"run_query\">"
            + "<｜GCML｜parameter name=\"limit\" string=\"false\">25</｜GCML｜parameter>"
            + "<｜GCML｜parameter name=\"raw_id\" string=\"true\">00123</｜GCML｜parameter>"
            + "<｜GCML｜parameter name=\"flag\" string=\"true\">true</｜GCML｜parameter>"
            + "<｜GCML｜parameter name=\"filter\" string=\"false\">{\"active\":true,\"ids\":[1,2]}</｜GCML｜parameter>"
            + "</｜GCML｜invoke>"
            + "</｜GCML｜tool_calls>"

        let direct = QWToolCallParser.parse(
            text: sample,
            tools: [tool],
            startsInThinking: false,
            toolEnvelopes: QWToolCallEnvelope.envelopes(prioritizing: .gcml)
        )
        XCTAssertEqual(direct.reasoningContent, "Check query args.")
        XCTAssertEqual(direct.content, "Running query.")
        XCTAssertEqual(direct.finishReason, "tool_calls")
        XCTAssertEqual(direct.toolCalls?.count, 1)
        XCTAssertEqual(direct.toolCalls?[0].function.name, "run_query")
        XCTAssertEqual(
            direct.toolCalls?[0].function.arguments,
            #"{"limit":25,"raw_id":"00123","flag":"true","filter":{"active":true,"ids":[1,2]}}"#
        )

        let stream = QWChatStreamParser(
            startsInReasoning: false,
            allowToolCalls: true,
            allowReasoningTags: true,
            toolEnvelopes: QWToolCallEnvelope.envelopes(prioritizing: .gcml)
        )
        var cursor = sample.startIndex
        while cursor < sample.endIndex {
            let next = sample.index(cursor, offsetBy: 3, limitedBy: sample.endIndex) ?? sample.endIndex
            _ = stream.feed(String(sample[cursor..<next]))
            cursor = next
        }
        _ = stream.finish()
        let streamed = stream.result(tools: [tool], truncated: false)
        XCTAssertEqual(streamed.reasoningContent, direct.reasoningContent)
        XCTAssertEqual(streamed.content, direct.content)
        XCTAssertEqual(streamed.toolCalls?.first?.function, direct.toolCalls?.first?.function)
        XCTAssertEqual(
            QWToolCallParser.firstCompleteToolCallEnd(
                in: sample,
                envelopes: QWToolCallEnvelope.envelopes(prioritizing: .gcml)
            ),
            sample.endIndex
        )
    }

    func testMultiTokenGCMLToolCallConstraintIgnoresLiteralLessThanInProse() throws {
        let pieces: [String?] = [
            "<", "｜GCML｜", "tool", "_calls", ">",
            "if a ", " b ", "\n",
            "<｜GCML｜invoke", " name=\"get_weather\">", " name=\"bad_fn\">",
            "<｜GCML｜parameter", " name=\"city\" string=\"true\">", " name=\"bogus\" string=\"true\">",
            "Moscow", "</｜GCML｜parameter>", "</｜GCML｜invoke>", "</｜GCML｜tool_calls>",
        ]
        let table = QWJSONTokenTable(tokenBytes: pieces.map { $0.map { Array($0.utf8) } }, vocabulary: pieces.count)
        let tool = try QWJson.parse("""
        {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}
        """)

        let autoConstraint = try XCTUnwrap(QWToolCallConstraint(
            tools: [tool],
            forcedToolName: nil,
            table: table,
            toolCallOpenTokens: [0, 1, 2, 3, 4],
            toolCallEndToken: 17,
            requireToolCall: false
        ))
        try autoConstraint.accept(5) // "if a "
        try autoConstraint.accept(0) // "<" — prefix match 1/5, must NOT enter tool mode!
        XCTAssertNil(try autoConstraint.allowedTokens(), "literal '<' in prose must remain unconstrained")
        try autoConstraint.accept(6) // " b " — resets prefix match to 0
        try autoConstraint.accept(8) //prose mention outside envelope stays unconstrained
        XCTAssertNil(try autoConstraint.allowedTokens())

        // Now emit the full 5-token <｜GCML｜tool_calls> sequence.
        for tokId in [0, 1, 2, 3, 4] {
            try autoConstraint.accept(tokId)
        }
        try autoConstraint.accept(7) // "\n"
        try autoConstraint.accept(8) // "<｜GCML｜invoke"
        XCTAssertEqual(try autoConstraint.allowedTokens(), [9], "must restrict invoke name to get_weather")
        try autoConstraint.accept(9)
        try autoConstraint.accept(11) // "<｜GCML｜parameter"
        XCTAssertEqual(try autoConstraint.allowedTokens(), [12], "must restrict parameter name to city")
        try autoConstraint.accept(12)
        try autoConstraint.accept(14)
        try autoConstraint.accept(15)
        try autoConstraint.accept(16)
        try autoConstraint.accept(17)
        XCTAssertTrue(autoConstraint.isUnconstrainedNow)

        let forcedConstraint = try XCTUnwrap(QWToolCallConstraint(
            tools: [tool],
            forcedToolName: "get_weather",
            table: table,
            toolCallOpenTokens: [0, 1, 2, 3, 4],
            toolCallEndToken: 17,
            requireToolCall: true
        ))
        for expected in [0, 1, 2, 3, 4] {
            XCTAssertEqual(try forcedConstraint.allowedTokens(), [expected])
            try forcedConstraint.accept(expected)
        }
    }

    func testStructuredOutputConstraintMultiTokenToolPrefixAndSpeculativeMask() throws {
        let pieces: [String?] = [
            "{", "}", "\"", "k", "\":", "1", " ",
            "<", "｜GCML｜", "tool", "_calls", ">", nil,
        ]
        let table = QWJSONTokenTable(tokenBytes: pieces.map { $0.map { Array($0.utf8) } }, vocabulary: pieces.count)
        let grammar = QWJSONGrammar.anyObject()

        let toolBranch = QWStructuredOutputConstraint(
            grammar: grammar,
            table: table,
            stopTokens: [12],
            toolCallOpenTokens: [7, 8, 9, 10, 11]
        )
        XCTAssertEqual(try toolBranch.allowedTokens(), [0, 6, 7])
        try toolBranch.accept(6) // leading space keeps choice open
        XCTAssertEqual(try toolBranch.allowedTokens(), [0, 6, 7])
        try toolBranch.accept(7) // "<" starts multi-token tool prefix
        XCTAssertEqual(try toolBranch.allowedTokens(), [8])
        XCTAssertEqual(toolBranch.speculationBoundaryToken, 8)
        if QWConstrainedSpeculationGate.isEnabled {
            XCTAssertEqual(toolBranch.proposeCanonicalDrafts(maxCount: 8), [8, 9, 10, 11])
        }

        for tokId in [8, 9, 10, 11] {
            try toolBranch.accept(tokId)
        }
        XCTAssertNil(try toolBranch.allowedTokens(), "full GCML open sequence frees tool generation")

        let jsonBranch = QWStructuredOutputConstraint(
            grammar: grammar,
            table: table,
            stopTokens: [12],
            toolCallOpenTokens: [7, 8, 9, 10, 11]
        )
        try jsonBranch.accept(0) // "{"
        try jsonBranch.accept(2) // "\""
        try jsonBranch.accept(3) // "k"
        try jsonBranch.accept(4) // "\":"
        try jsonBranch.accept(2) // "\""
        try jsonBranch.accept(7) // "<" inside JSON string must NOT switch to free tool mode
        XCTAssertNotNil(try jsonBranch.allowedTokens())
    }

    func testIncrementalPrefixCacheHandlesRoleBeforeRoleSepMarker() throws {
        var vocab: [String: Int] = ["<s>": 1, "</s>": 2]
        var nextId = 10
        var n = 0
        for b in 0..<256 {
            let printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
            let sc = printable ? Unicode.Scalar(UInt8(b)) : Unicode.Scalar(0x100 + n)!
            if !printable { n += 1 }
            if b == 0xC0 || b == 0xC1 || b >= 0xF5 { continue }
            vocab[String(sc)] = nextId
            nextId += 1
        }
        let tokObj: [String: Any] = [
            "version": "1.0",
            "model": ["type": "BPE", "vocab": vocab, "merges": [[String]]()],
            "pre_tokenizer": [
                "type": "Sequence",
                "pretokenizers": [
                    ["type": "Split", "pattern": ["Regex": #"[A-Za-z]+|\s+|."#], "behavior": "Isolated", "invert": false] as [String: Any],
                    ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false] as [String: Any],
                ],
            ] as [String: Any],
            "added_tokens": [
                ["id": 1, "content": "<s>", "special": true],
                ["id": 2, "content": "</s>", "special": true],
                ["id": 128000, "content": "<|role_sep|>\n", "special": true],
                ["id": 128001, "content": "<|message_sep|>\n\n", "special": true],
                ["id": 128014, "content": "<think>", "special": true],
                ["id": 128015, "content": "</think>", "special": true],
            ],
        ]
        let tok = try QWTokenizer(json: try JSONSerialization.data(withJSONObject: tokObj))
        let actor = QWEngineLifecycleActor(
            config: .init(packPath: "/tmp/gc4-pack", modelName: "gigachat4", host: "127.0.0.1", port: 0),
            logger: QWServerLogger(level: .none)
        )

        let wire1 = "<s>system<|role_sep|>\nsys<|message_sep|>\n\nuser<|role_sep|>\nhi<|message_sep|>\n\nassistant<|role_sep|>\n<think>"
        let res1 = try actor.encodeWireIncrementallyWithSystemPrefix(
            wire: wire1, tokenizer: tok, packPath: "/tmp/gc4-pack", cachePrompt: true
        )
        XCTAssertEqual(res1.tokens, try tok.encode(wire1))
        XCTAssertNotNil(res1.systemPrefixTokenCount)

        let wire2 = "<s>system<|role_sep|>\nsys<|message_sep|>\n\nuser<|role_sep|>\nhi<|message_sep|>\n\n"
            + "assistant<|role_sep|>\n<think></think>ok<|message_sep|>\n\nuser<|role_sep|>\nho<|message_sep|>\n\nassistant<|role_sep|>\n<think>"
        let res2 = try actor.encodeWireIncrementallyWithSystemPrefix(
            wire: wire2, tokenizer: tok, packPath: "/tmp/gc4-pack", cachePrompt: true
        )
        XCTAssertEqual(res2.tokens, try tok.encode(wire2))
        XCTAssertEqual(res2.systemPrefixTokenCount, res1.systemPrefixTokenCount)
    }
}
