import XCTest
@testable import QWServer
import QWToken

final class QWServerSchemaTests: XCTestCase {
    func testSimpleTextChatCompletionRequest() throws {
        let jsonStr = """
        {
            "model": "qwen3.8-27b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "stream": false,
            "max_tokens": 128,
            "temperature": 0.7,
            "stop": ["<|im_end|>"]
        }
        """
        let data = jsonStr.data(using: .utf8)!
        let req = try JSONDecoder().decode(ChatCompletionRequest.self, from: data)

        XCTAssertEqual(req.model, "qwen3.8-27b")
        XCTAssertEqual(req.messages.count, 2)
        XCTAssertEqual(req.messages[0].role, "system")
        XCTAssertEqual(req.messages[0].content?.textValue, "You are a helpful assistant.")
        XCTAssertEqual(req.messages[1].role, "user")
        XCTAssertEqual(req.messages[1].content?.textValue, "Hello, how are you?")
        XCTAssertEqual(req.isStreaming, false)
        XCTAssertEqual(req.effectiveMaxTokens, 128)
        XCTAssertEqual(req.stop, ["<|im_end|>"])
        XCTAssertEqual(req.extractImageUrls().count, 0)

        let qwMsgs = req.toQWJsonMessages()
        XCTAssertEqual(qwMsgs.count, 2)
    }

    func testMultimodalChatCompletionRequestWithImageUrls() throws {
        let jsonStr = """
        {
            "model": "qwen3.8-27b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
                            }
                        }
                    ]
                }
            ],
            "stream": true,
            "max_tokens": 256
        }
        """
        let data = jsonStr.data(using: .utf8)!
        let req = try JSONDecoder().decode(ChatCompletionRequest.self, from: data)

        XCTAssertEqual(req.messages.count, 1)
        XCTAssertEqual(req.isStreaming, true)
        XCTAssertEqual(req.effectiveMaxTokens, 256)

        let msg = req.messages[0]
        XCTAssertEqual(msg.role, "user")
        XCTAssertEqual(msg.content?.textValue, "What is in this image?")

        let images = req.extractImageUrls()
        XCTAssertEqual(images.count, 1)
        XCTAssertEqual(images[0], "data:image/jpeg;base64,/9j/4AAQSkZJRg==")
    }

    func testChatCompletionResponseSerialization() throws {
        let resp = ChatCompletionResponse(
            id: "chatcmpl-test1234",
            model: "qwen3.8-27b",
            content: "I am doing well, thank you!",
            promptTokens: 25,
            completionTokens: 8
        )
        let data = try JSONEncoder().encode(resp)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        XCTAssertEqual(json?["id"] as? String, "chatcmpl-test1234")
        XCTAssertEqual(json?["object"] as? String, "chat.completion")
        XCTAssertEqual(json?["model"] as? String, "qwen3.8-27b")

        let choices = json?["choices"] as? [[String: Any]]
        XCTAssertEqual(choices?.count, 1)
        let message = choices?[0]["message"] as? [String: Any]
        XCTAssertEqual(message?["content"] as? String, "I am doing well, thank you!")
        XCTAssertEqual(message?["role"] as? String, "assistant")
        XCTAssertEqual(choices?[0]["finish_reason"] as? String, "stop")

        let usage = json?["usage"] as? [String: Any]
        XCTAssertEqual(usage?["prompt_tokens"] as? Int, 25)
        XCTAssertEqual(usage?["completion_tokens"] as? Int, 8)
        XCTAssertEqual(usage?["total_tokens"] as? Int, 33)
    }

    func testToolCallParsingAndSerialization() throws {
        let modelOutput = """
        Let me make the call.

        <tool_call>
        <function=terminal>
        <parameter=command>
        curl -s "https://api.github.com"
        </parameter>
        </function>
        </tool_call>
        """

        let parsed = QWToolCallParser.parse(text: modelOutput)
        XCTAssertEqual(parsed.content, "Let me make the call.")
        XCTAssertEqual(parsed.finishReason, "tool_calls")
        XCTAssertNotNil(parsed.toolCalls)
        XCTAssertEqual(parsed.toolCalls?.count, 1)

        let tool = try XCTUnwrap(parsed.toolCalls?.first)
        XCTAssertEqual(tool.function.name, "terminal")
        XCTAssertTrue(tool.function.arguments.contains("curl -s \\\"https://api.github.com\\\""))

        let resp = ChatCompletionResponse(
            id: "chatcmpl-tool-test",
            model: "qwen3.8-27b",
            content: parsed.content,
            toolCalls: parsed.toolCalls,
            promptTokens: 40,
            completionTokens: 30,
            finishReason: parsed.finishReason
        )

        let data = try JSONEncoder().encode(resp)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let choices = json?["choices"] as? [[String: Any]]
        XCTAssertEqual(choices?[0]["finish_reason"] as? String, "tool_calls")

        let message = choices?[0]["message"] as? [String: Any]
        XCTAssertEqual(message?["content"] as? String, "Let me make the call.")
        let toolCalls = message?["tool_calls"] as? [[String: Any]]
        XCTAssertEqual(toolCalls?.count, 1)
        let firstCall = toolCalls?[0]
        XCTAssertEqual(firstCall?["type"] as? String, "function")
        let funcObj = firstCall?["function"] as? [String: Any]
        XCTAssertEqual(funcObj?["name"] as? String, "terminal")
    }

    func testJsonToolCallParsing() throws {
        let modelOutput = """
        <tool_call>
        {"name": "fetch_weather", "arguments": {"city": "Paris", "days": 3}}
        </tool_call>
        """

        let parsed = QWToolCallParser.parse(text: modelOutput)
        XCTAssertNil(parsed.content)
        XCTAssertEqual(parsed.finishReason, "tool_calls")
        XCTAssertEqual(parsed.toolCalls?.count, 1)

        let tool = try XCTUnwrap(parsed.toolCalls?.first)
        XCTAssertEqual(tool.function.name, "fetch_weather")
        XCTAssertTrue(tool.function.arguments.contains("\"city\":\"Paris\"") || tool.function.arguments.contains("\"city\": \"Paris\""))
    }

    func testToolRequestWithToolsAndHistory() throws {
        let requestJson = """
        {
            "model": "qwen3.8-27b",
            "messages": [
                {"role": "user", "content": "Run curl on example.com"},
                {
                    "role": "assistant",
                    "content": "Running the command now.",
                    "tool_calls": [
                        {
                            "id": "call_12345",
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "arguments": "{\\"command\\": \\"curl -s https://example.com\\"}"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_12345",
                    "content": "Example Domain"
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "description": "Run shell command",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {"type": "string"}
                            },
                            "required": ["command"]
                        }
                    }
                }
            ]
        }
        """

        let data = requestJson.data(using: .utf8)!
        let req = try JSONDecoder().decode(ChatCompletionRequest.self, from: data)

        XCTAssertNotNil(req.tools)
        XCTAssertEqual(req.tools?.count, 1)
        XCTAssertEqual(req.messages.count, 3)

        let qwMsgs = req.toQWJsonMessages()
        XCTAssertEqual(qwMsgs.count, 3)

        let templateURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/spike/assets/chat_template.jinja")
        let templateSource = try String(contentsOf: templateURL, encoding: .utf8)
        let template = try QWChatTemplate(source: templateSource)
        var opts = QWChatTemplateOptions()
        opts.tools = .array(req.tools!)
        opts.addGenerationPrompt = true

        let rendered = try template.render(messages: qwMsgs, options: opts)
        XCTAssertTrue(rendered.contains("<tools>"))
        XCTAssertTrue(rendered.contains("terminal"))
        XCTAssertTrue(rendered.contains("<tool_call>"))
        XCTAssertTrue(rendered.contains("<tool_response>"))
        XCTAssertTrue(rendered.contains("Example Domain"))
    }

    func testChatCompletionChunkSerialization() throws {
        let chunk = ChatCompletionChunk(
            id: "chatcmpl-stream1",
            model: "qwen3.8-27b",
            content: "Hello"
        )
        let data = try JSONEncoder().encode(chunk)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        XCTAssertEqual(json?["id"] as? String, "chatcmpl-stream1")
        XCTAssertEqual(json?["object"] as? String, "chat.completion.chunk")
        let choices = json?["choices"] as? [[String: Any]]
        let delta = choices?[0]["delta"] as? [String: Any]
        XCTAssertEqual(delta?["content"] as? String, "Hello")
    }

    func testModelLifecycleSerialization() throws {
        let loadResp = ModelLifecycleResponse(status: "ok", model: "qwen3.8-27b", loaded: true)
        let loadData = try JSONEncoder().encode(loadResp)
        let loadJson = try JSONSerialization.jsonObject(with: loadData) as? [String: Any]
        XCTAssertEqual(loadJson?["loaded"] as? Bool, true)

        let unloadResp = ModelLifecycleResponse(status: "ok", model: "qwen3.8-27b", loaded: false, memoryReclaimedMB: 16896.0)
        let unloadData = try JSONEncoder().encode(unloadResp)
        let unloadJson = try JSONSerialization.jsonObject(with: unloadData) as? [String: Any]
        XCTAssertEqual(unloadJson?["loaded"] as? Bool, false)
        XCTAssertEqual(unloadJson?["memoryReclaimedMB"] as? Double, 16896.0)
    }

    func testDeveloperRoleAndMidReasoningEffortNormalizationWithChatTemplate() throws {
        let requestJson = """
        {
            "model": "qwen3.8-flash-next",
            "reasoning_effort": "mid",
            "messages": [
                {"role": "developer", "content": "You are Mavis inside MiniMax Code."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "system", "content": "<system-reminder>Stay concise</system-reminder>"},
                {"role": "user", "content": "Run probe"}
            ]
        }
        """
        let req = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(requestJson.utf8))
        XCTAssertEqual(req.effectiveReasoningEffort(defaultValue: "xhigh"), "medium")
        XCTAssertEqual(ChatCompletionRequest.normalizeReasoningEffort("high"), "xhigh")
        XCTAssertEqual(ChatCompletionRequest.normalizeReasoningEffort("max"), "xhigh")
        XCTAssertEqual(ChatCompletionRequest.normalizeReasoningEffort("minimal"), "low")
        XCTAssertEqual(ChatCompletionRequest.normalizeReasoningEffort("none"), "low")

        let qwMsgs = req.toQWJsonMessages()
        XCTAssertEqual(qwMsgs.count, 5)
        XCTAssertEqual(qwMsgs[0]["role"], .string("system"))
        XCTAssertEqual(qwMsgs[3]["role"], .string("user"))

        let templateURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/spike/assets/chat_template.jinja")
        let templateSource = try String(contentsOf: templateURL, encoding: .utf8)
        let template = try QWChatTemplate(source: templateSource)

        var opts = QWChatTemplateOptions()
        opts.addGenerationPrompt = true
        opts.enableThinking = .bool(req.effectiveEnableThinking(defaultValue: true))
        opts.reasoningEffort = .string(req.effectiveReasoningEffort(defaultValue: "xhigh"))

        let rendered = try template.render(messages: qwMsgs, options: opts)
        XCTAssertTrue(rendered.contains("You are Mavis inside MiniMax Code."))
        XCTAssertTrue(rendered.contains("Run probe"))
    }

    func testGLM5ToolCallParsingAndChatTemplateRendering() throws {
        let modelOutput = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>"
        let toolJson = try QWJson.parse("""
        [{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}]
        """)
        let tools: [QWJson] = {
            if case .array(let arr) = toolJson { return arr }
            return []
        }()
        let parsed = QWToolCallParser.parse(text: modelOutput, tools: tools)
        XCTAssertEqual(parsed.finishReason, "tool_calls")
        XCTAssertEqual(parsed.toolCalls?.count, 1)
        XCTAssertEqual(parsed.toolCalls?.first?.function.name, "get_weather")
        XCTAssertEqual(parsed.toolCalls?.first?.function.arguments, #"{"city":"Paris","days":3}"#)

        let parallelOutput = "<tool_call>list_files</tool_call><tool_call>run_command<arg_key>cmd</arg_key><arg_value>npm test</arg_value></tool_call>"
        let parallelParsed = QWToolCallParser.parse(text: parallelOutput)
        XCTAssertEqual(parallelParsed.toolCalls?.count, 2)
        XCTAssertEqual(parallelParsed.toolCalls?.first?.function.name, "list_files")
        XCTAssertEqual(parallelParsed.toolCalls?.first?.function.arguments, "{}")
        XCTAssertEqual(parallelParsed.toolCalls?.last?.function.name, "run_command")
        XCTAssertEqual(parallelParsed.toolCalls?.last?.function.arguments, #"{"cmd":"npm test"}"#)

        let literalAndParallel = "<tool_call>write_file<arg_key>body</arg_key><arg_value>a <tool_call>x</tool_call> b</arg_value></tool_call>\n<tool_call>list_files</tool_call>"
        let literalParsed = QWToolCallParser.parse(text: literalAndParallel)
        XCTAssertEqual(literalParsed.toolCalls?.count, 2)
        XCTAssertEqual(literalParsed.toolCalls?.first?.function.arguments, #"{"body":"a <tool_call>x</tool_call> b"}"#)
        XCTAssertEqual(literalParsed.toolCalls?.last?.function.name, "list_files")

        let glm5TemplatePath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".qwg/models/GLM-5.3-Flash/assets/chat_template.jinja").path
        guard FileManager.default.fileExists(atPath: glm5TemplatePath) else {
            throw XCTSkip("requires a local GLM-5.3-Flash model pack; skipped")
        }
        let src = try String(contentsOfFile: glm5TemplatePath, encoding: .utf8)
        let tpl = try QWChatTemplate(source: src)
        let reqJson = """
        {
          "model": "GLM-5.3-Flash",
          "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\\"city\\":\\"Paris\\",\\"days\\":3}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "{\\"temp\\": 22}"}
          ],
          "tools": [{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}]
        }
        """
        let req = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(reqJson.utf8))
        var opts = QWChatTemplateOptions()
        opts.addGenerationPrompt = true
        opts.enableThinking = .bool(false)
        opts.tools = .array(req.tools ?? [])
        let rendered = try tpl.render(messages: req.toQWJsonMessages(), options: opts)
        XCTAssertTrue(rendered.contains("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"))
        XCTAssertTrue(rendered.contains("<|observation|><tool_response>{\"temp\": 22}</tool_response>"))
        XCTAssertTrue(rendered.hasSuffix("<|assistant|>\u{3c}think\u{3e}\u{3c}/think\u{3e}"))
    }

    func testResolveReplyModelNameAndKeepaliveGate() {
        var settings = QWServerSettings()
        settings.modelName = "Qwen3.8-Flash-Next"
        settings.modelAlias = "flash-next"
        settings.modelAliases = ["Qwen3.8-Flash-Next": "flash-next"]
        let discovered = [
            DiscoveredModel(name: "Qwen3.8-Flash-Next", packPath: "/tmp/Qwen3.8-Flash-Next/model.qwp")
        ]

        XCTAssertEqual(
            QWHttpServer.resolveReplyModelName(
                requested: nil,
                serverModel: "Qwen3.8-Flash-Next",
                settings: settings,
                strictNotFound: true,
                discovered: discovered
            ),
            "flash-next"
        )
        XCTAssertEqual(
            QWHttpServer.resolveReplyModelName(
                requested: "Qwen3.8-Flash-Next",
                serverModel: "Qwen3.8-Flash-Next",
                settings: settings,
                strictNotFound: true,
                discovered: discovered
            ),
            "flash-next"
        )
        XCTAssertNil(
            QWHttpServer.resolveReplyModelName(
                requested: "claude-3.5-sonnet",
                serverModel: "Qwen3.8-Flash-Next",
                settings: settings,
                strictNotFound: true,
                discovered: discovered
            )
        )
        XCTAssertEqual(
            QWHttpServer.resolveReplyModelName(
                requested: "claude-3.5-sonnet",
                serverModel: "Qwen3.8-Flash-Next",
                settings: settings,
                strictNotFound: false,
                discovered: discovered
            ),
            "flash-next"
        )

        let gate = QWStreamKeepaliveGate()
        XCTAssertFalse(gate.isStopped())
        gate.stop()
        XCTAssertTrue(gate.isStopped())
    }

    func testChatTurnPreparerSynthesizesSchemaAndForcedToolInstructions() throws {
        var vocab: [String: Int] = [:]
        for b in 0..<256 {
            vocab[String(format: "<0x%02X>", b)] = b
        }
        let tokObj: [String: Any] = [
            "model": [
                "type": "BPE",
                "byte_fallback": true,
                "vocab": vocab,
                "merges": [String]()
            ],
            "pre_tokenizer": [
                "type": "Metaspace",
                "replacement": "\u{2581}",
                "prepend_scheme": "always",
                "split": false
            ],
            "added_tokens": [
                ["id": 256, "content": "<|im_end|>", "special": true],
                ["id": 257, "content": "</think>", "special": true],
                ["id": 258, "content": "<tool_call>", "special": true],
                ["id": 259, "content": "</tool_call>", "special": true]
            ]
        ]
        let tokData = try JSONSerialization.data(withJSONObject: tokObj)
        let tok = try QWTokenizer(json: tokData)
        let templateURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/spike/assets/chat_template.jinja")
        let chatTemplate = try QWChatTemplate(source: String(contentsOf: templateURL, encoding: .utf8))
        let reqJson = """
        {
          "model": "test",
          "messages": [{"role": "user", "content": "Give JSON"}],
          "response_format": {"type": "json_object"},
          "enable_thinking": false
        }
        """
        let req = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(reqJson.utf8))
        let cfg = QWEngineLifecycleActor.Config(
            packPath: "/tmp/synthetic-pack",
            systemPrompt: "Base system."
        )
        let prepared = try QWChatTurnPreparer.preparePrompt(
            request: req,
            config: cfg,
            chatTemplate: chatTemplate,
            tokenizer: tok,
            vocabularySize: 260,
            supportsTokenConstraints: true
        )
        XCTAssertFalse(prepared.shouldThink)
        XCTAssertTrue(prepared.enforceJson)
        XCTAssertNotNil(prepared.activeConstraint)
        if case .string(let sysContent) = prepared.messages.first?.get("content") {
            XCTAssertTrue(sysContent.contains("Base system."))
            XCTAssertTrue(sysContent.contains("MUST output ONLY a single raw, valid JSON object"))
        } else {
            XCTFail("expected synthesized system message at index 0")
        }
    }
}
