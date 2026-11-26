// QWInkling2ChatTests.swift — Tests Inkling-2 chat template, thinking effort, and tool-call streaming/parsing.
import XCTest
import QWCore
import QWMM
@testable import QWServer
import QWToken
import QWVision

final class QWInkling2ChatTests: XCTestCase {
    private static let minimalInk2Jinja = """
    {# <|message_model|><|content_thinking|><|content_text|><|content_invoke_tool_json|><|end_message|><|content_model_end_sampling|> #}
    {{ messages[0].content }}
    """

    func testEffortMappingAndThinkingOffRender() throws {
        let tpl = try QWChatTemplate(source: Self.minimalInk2Jinja)
        XCTAssertTrue(tpl.isInkling2Template)
        XCTAssertTrue(tpl.supportsThinkingTags)

        let msgs: [QWJson] = [
            .object([("role", .string("system")), ("content", .string("Be concise."))]),
            .object([("role", .string("user")), ("content", .string("Hi"))]),
        ]

        var opts = QWChatTemplateOptions()
        opts.enableThinking = .bool(false)
        opts.reasoningEffort = .string("xhigh")
        let off = try tpl.render(messages: msgs, options: opts)
        XCTAssertEqual(
            off,
            "<|message_system|><|content_text|>Be concise.<|end_message|>"
                + "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
                + "<|message_user|><|content_text|>Hi<|end_message|>"
        )

        for (effort, expected) in [("low", "0.2"), ("medium", "0.7"), ("high", "0.9"), ("xhigh", "0.99")] {
            var onOpts = QWChatTemplateOptions()
            onOpts.enableThinking = .bool(true)
            onOpts.reasoningEffort = .string(effort)
            let wire = try tpl.render(messages: msgs, options: onOpts)
            XCTAssertTrue(wire.contains("<|message_system|><|content_text|>Thinking effort level: \(expected)<|end_message|>"))
        }
    }

    func testToolsHistoryAndMultimodalRender() throws {
        let tpl = try QWChatTemplate(source: Self.minimalInk2Jinja)
        let weatherTool: QWJson = .object([
            ("type", .string("function")),
            ("function", .object([
                ("name", .string("get_weather")),
                ("description", .string("Fetch weather")),
                ("parameters", .object([
                    ("type", .string("object")),
                    ("properties", .object([
                        ("city", .object([("type", .string("string"))])),
                    ])),
                ])),
            ])),
        ])
        let msgs: [QWJson] = [
            .object([("role", .string("user")), ("content", .array([
                .object([("type", .string("image_url")), ("image_url", .object([("url", .string("a.png"))]))]),
                .object([("type", .string("text")), ("text", .string("Forecast?"))]),
            ]))]),
            .object([
                ("role", .string("assistant")),
                ("reasoning_content", .string("Call weather tool.")),
                ("content", .string("")),
                ("tool_calls", .array([
                    .object([
                        ("id", .string("call_1")),
                        ("type", .string("function")),
                        ("function", .object([
                            ("name", .string("get_weather")),
                            ("arguments", .string("{\"units\":\"metric\",\"city\":\"Paris\"}")),
                        ])),
                    ]),
                ])),
            ]),
            .object([("role", .string("tool")), ("tool_call_id", .string("call_1")), ("content", .string("18C"))]),
        ]

        var opts = QWChatTemplateOptions()
        opts.tools = .array([weatherTool])
        opts.enableThinking = .bool(true)
        opts.reasoningEffort = .string("medium")
        let wire = try tpl.render(messages: msgs, options: opts)

        XCTAssertTrue(wire.hasPrefix("<|message_system|>tool_declare<|content_xml|>"))
        XCTAssertTrue(wire.contains("<|message_user|><|content_image|><|unused_200054|><|end_message|>"))
        XCTAssertTrue(wire.contains("<|message_model|><|content_thinking|>Call weather tool.<|end_message|>"))
        XCTAssertTrue(wire.contains("<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"units\":\"metric\"}}<|end_message|><|content_model_end_sampling|>"))
        XCTAssertTrue(wire.contains("<|message_tool|>get_weather<|content_text|>18C<|end_message|>"))
    }

    func testStreamParserInkling2ChannelsAndToolsAcrossSplits() throws {
        let sample = "<|message_model|><|content_thinking|>Step 1. Step 2.<|end_message|>"
            + "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"days\":2}}<|end_message|>"
            + "<|message_model|>calc<|content_invoke_tool_json|>{\"name\":\"calc\",\"args\":{\"expr\":\"2+2\"}}<|end_message|>"
            + "<|message_model|><|content_text|>Done.<|end_message|><|content_model_end_sampling|>"

        let parser = QWChatStreamParser(startsInReasoning: true, allowToolCalls: true, allowReasoningTags: true, allowInk2Markers: true)
        var deltas: [QWChatStreamParser.Delta] = []
        var idx = sample.startIndex
        while idx < sample.endIndex {
            let next = sample.index(idx, offsetBy: 3, limitedBy: sample.endIndex) ?? sample.endIndex
            deltas.append(contentsOf: parser.feed(String(sample[idx..<next])))
            idx = next
        }
        deltas.append(contentsOf: parser.finish())

        let res = parser.result(tools: nil, truncated: false)
        XCTAssertEqual(res.reasoningContent, "Step 1. Step 2.")
        XCTAssertEqual(res.content, "Done.")
        XCTAssertEqual(res.finishReason, "tool_calls")
        XCTAssertEqual(res.toolCalls?.count, 2)
        XCTAssertEqual(res.toolCalls?[0].function.name, "get_weather")
        XCTAssertEqual(res.toolCalls?[0].function.arguments, "{\"city\":\"Paris\",\"days\":2}")
        XCTAssertEqual(res.toolCalls?[1].function.name, "calc")

        let streamedReasoning = deltas.compactMap { if case .reasoning(let s) = $0 { return s }; return nil }.joined()
        let streamedContent = deltas.compactMap { if case .content(let s) = $0 { return s }; return nil }.joined()
        XCTAssertEqual(streamedReasoning, "Step 1. Step 2.")
        XCTAssertEqual(streamedContent, "Done.")
    }

    func testStreamParserDropsIncompleteInk2ToolOnTruncation() throws {
        let partial = "<|message_model|><|content_text|>Wait<|end_message|><|message_model|>fn<|content_invoke_tool_json|>{\"name\":\"fn\",\"args\":"
        let res = QWToolCallParser.parse(text: partial, tools: nil, startsInThinking: false, truncated: true, allowInk2Markers: true)
        XCTAssertEqual(res.content, "Wait")
        XCTAssertNil(res.toolCalls)
        XCTAssertEqual(res.finishReason, "length")
    }

    func testRealInkling2AssetsWhenPresent() throws {
        let modelDir = (NSString("~/.qwg/models/Inkling-2-Mini-MLX-4-8bit").expandingTildeInPath)
        try XCTSkipIf(!FileManager.default.fileExists(atPath: modelDir + "/chat_template.jinja"),
                      "Inkling-2-Mini model assets not present locally")
        let assets = try QWModelRuntimeAssets.load(packPath: modelDir)
        XCTAssertTrue(assets.chatTemplate.isInkling2Template)
        XCTAssertTrue(assets.chatTemplate.supportsThinkingTags)
        let tok = try QWTokenizer(json: assets.tokenizerJSON)
        XCTAssertNil(QWChatPolicyResolver.makeThinkingBudgetConstraint(
            maxThinkingTokens: 64, endThinkTokenId: 200010, vocabularySize: assets.descriptor.vocab,
            tokenizer: tok, supportsTokenConstraints: true, inner: nil
        ))
    }

    func testInkling2MarkerRenderAnd40x40PadPreprocessing() throws {
        let tpl = try QWChatTemplate(source: Self.minimalInk2Jinja)
        let msgs: [QWJson] = [
            .object([("role", .string("user")), ("content", .string("<|vision_start|><|image_pad|><|vision_end|>Describe this"))]),
        ]
        var opts = QWChatTemplateOptions()
        opts.enableThinking = .bool(false)
        let wire = try tpl.render(messages: msgs, options: opts)
        XCTAssertTrue(wire.contains("<|message_user|><|content_image|><|unused_200054|><|end_message|><|message_user|><|content_text|>Describe this<|end_message|>"))

        let vCfg = QWVisionConfig(
            dim: 1024, heads: 16, headDim: 64, depth: 23, mlpInner: 4096,
            patch: 20, temporal: 1, merge: 2, numPosEmbeds: 1024, outDim: 2304,
            imageTokenId: 200054, videoTokenId: -1, architecture: .inkling2
        )
        let spec = QWMMSpec(visionConfig: vCfg)
        // 50x45 RGB image pads to 80x80 => rows=2, cols=2, grid.tokens=4
        let rgb = [UInt8](repeating: 255, count: 50 * 45 * 3)
        let res = try QWImageDecoder.normalizePadAndPatchifyInkling2(rgb: rgb, height: 50, width: 45, spec: spec)
        XCTAssertEqual(res.grid.tokens, 4)
        XCTAssertEqual(res.grid.patches, 16)
        XCTAssertEqual(res.patches.count, 16 * 1200)
        // First patch (0,0) top-left pixel is 255 -> +1.0
        XCTAssertEqual(res.patches[0], 1.0, accuracy: 1e-5)
        // Bottom-right patch bottom-right pixel (79,79) is padded -> (-1/255 - 0.5)/0.5
        let padExpected = Float((-1.0 / 255.0 - 0.5) / 0.5)
        XCTAssertEqual(res.patches.last!, padExpected, accuracy: 1e-5)
    }

    private func makeInk2TestTokenizer() throws -> QWTokenizer {
        var vocab: [String: Int] = [:]
        for b in 0..<256 { vocab[String(format: "<0x%02X>", b)] = b }
        let obj: [String: Any] = [
            "model": ["type": "BPE", "byte_fallback": true, "vocab": vocab, "merges": [String]()],
            "pre_tokenizer": ["type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": "always", "split": false],
            "added_tokens": [
                ["id": 256, "content": "<|im_end|>", "special": true],
                ["id": 257, "content": "<|end_message|>", "special": true],
                ["id": 258, "content": "<|message_model|>", "special": true],
                ["id": 259, "content": "<|content_thinking|>", "special": true],
                ["id": 260, "content": "<|content_text|>", "special": true],
                ["id": 261, "content": "<|content_invoke_tool_json|>", "special": true],
                ["id": 262, "content": "<|content_model_end_sampling|>", "special": true],
            ],
        ]
        return try QWTokenizer(json: JSONSerialization.data(withJSONObject: obj))
    }

    func testPreparePromptThinkingOffForcesContentTextChannelAndEndMessageStop() throws {
        let tok = try makeInk2TestTokenizer()
        let tpl = try QWChatTemplate(source: Self.minimalInk2Jinja)
        let cfg = QWEngineLifecycleActor.Config(packPath: "/tmp/ink2-synthetic")

        for body in [
            "{\"model\":\"ink2\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"enable_thinking\":false}",
            "{\"model\":\"ink2\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"reasoning_effort\":\"none\"}",
        ] {
            let req = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(body.utf8))
            let prepared = try QWChatTurnPreparer.preparePrompt(
                request: req, config: cfg, chatTemplate: tpl, tokenizer: tok,
                vocabularySize: 263, supportsTokenConstraints: true
            )
            XCTAssertFalse(prepared.shouldThink)
            XCTAssertTrue(prepared.wire.hasSuffix("<|message_model|><|content_text|>"))
            XCTAssertTrue(prepared.stops.contains(257))
        }

        let toolsBody = """
        {
          "model": "ink2",
          "messages": [{"role": "user", "content": "Weather?"}],
          "enable_thinking": false,
          "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
        }
        """
        let toolsReq = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(toolsBody.utf8))
        let toolsPrepared = try QWChatTurnPreparer.preparePrompt(
            request: toolsReq, config: cfg, chatTemplate: tpl, tokenizer: tok,
            vocabularySize: 263, supportsTokenConstraints: true
        )
        XCTAssertFalse(toolsPrepared.wire.hasSuffix("<|message_model|><|content_text|>"))
        XCTAssertFalse(toolsPrepared.stops.contains(257))

        let onBody = "{\"model\":\"ink2\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"enable_thinking\":true}"
        let onReq = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(onBody.utf8))
        let onPrepared = try QWChatTurnPreparer.preparePrompt(
            request: onReq, config: cfg, chatTemplate: tpl, tokenizer: tok,
            vocabularySize: 263, supportsTokenConstraints: true
        )
        XCTAssertTrue(onPrepared.shouldThink)
        XCTAssertFalse(onPrepared.wire.hasSuffix("<|message_model|><|content_text|>"))
        XCTAssertFalse(onPrepared.stops.contains(257))
    }

    func testChatCompletionResponseEmptyContentSerializationVsToolCallsNull() throws {
        let reasoningOnly = ChatCompletionResponse(
            id: "chatcmpl-1", model: "ink2", content: "",
            toolCalls: nil, reasoningContent: "Still thinking...",
            promptTokens: 10, completionTokens: 20, finishReason: "length"
        )
        let rObj = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(reasoningOnly)) as? [String: Any])
        let rChoice = try XCTUnwrap((rObj["choices"] as? [[String: Any]])?.first)
        let rMsg = try XCTUnwrap(rChoice["message"] as? [String: Any])
        XCTAssertEqual(rMsg["content"] as? String, "")
        XCTAssertEqual(rMsg["reasoning_content"] as? String, "Still thinking...")

        let withToolCall = ChatCompletionResponse(
            id: "chatcmpl-2", model: "ink2", content: nil,
            toolCalls: [ToolCall(id: "call_1", function: ToolCallFunction(name: "fn", arguments: "{}"))],
            reasoningContent: nil, promptTokens: 10, completionTokens: 5, finishReason: "tool_calls"
        )
        let tObj = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(withToolCall)) as? [String: Any])
        let tChoice = try XCTUnwrap((tObj["choices"] as? [[String: Any]])?.first)
        let tMsg = try XCTUnwrap(tChoice["message"] as? [String: Any])
        XCTAssertTrue(tMsg["content"] is NSNull)
    }
}
