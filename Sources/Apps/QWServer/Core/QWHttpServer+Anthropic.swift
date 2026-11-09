import Foundation
import Network
import QWEngine
import QWVision

/// POST /v1/messages — the Anthropic Messages API. The request is
/// validated against the Anthropic schema, and the answer is shaped
/// like the Messages API answer: blocks, stop_reason, usage.
extension QWHttpServer {

    func handleMessages(connection: NWConnection, body: Data) async throws {
        guard let parsed = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any] else {
            sendAnthropicError(connection, status: 400, type: "invalid_request_error",
                               message: "request body must be a JSON object")
            return
        }

        guard let rawMessages = parsed["messages"] as? [[String: Any]], !rawMessages.isEmpty else {
            sendAnthropicError(connection, status: 400, type: "invalid_request_error",
                               message: "messages: expected required field present")
            return
        }

        guard let maxTokens = (parsed["max_tokens"] as? Int) ?? (parsed["max_tokens"] as? NSNumber)?.intValue, maxTokens > 0 else {
            sendAnthropicError(connection, status: 400, type: "invalid_request_error",
                               message: "max_tokens: expect number greater than 0")
            return
        }

        let requested = (parsed["model"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard let replyModel = resolveReplyModelName(requested: requested, strictNotFound: true) else {
            sendAnthropicError(connection, status: 404, type: "not_found_error",
                               message: "model: model \"\(requested)\" not found")
            return
        }

        var translated: [[String: Any]] = []
        if let system = parsed["system"] as? String, !system.isEmpty {
            translated.append(["role": "system", "content": system])
        } else if let blocks = parsed["system"] as? [[String: Any]] {
            let text = blocks.compactMap { block -> String? in
                (block["type"] as? String) == "text" ? (block["text"] as? String) : nil
            }.joined(separator: "\n")
            if !text.isEmpty { translated.append(["role": "system", "content": text]) }
        }
        for msg in rawMessages {
            translated.append(contentsOf: Self.translateAnthropicMessage(msg))
        }

        let thinkingDict = parsed["thinking"] as? [String: Any]
        let thinkingType = (thinkingDict?["type"] as? String)?.lowercased()
        let thinkingEnabled = (thinkingType == "enabled")

        var payload: [String: Any] = [
            "messages": translated,
            "max_tokens": maxTokens,
            "enable_thinking": thinkingEnabled,
        ]
        if !requested.isEmpty { payload["model"] = replyModel }
        if (parsed["stream"] as? Bool) == true { payload["stream"] = true }
        if let t = (parsed["temperature"] as? NSNumber)?.doubleValue { payload["temperature"] = t }
        if let tp = (parsed["top_p"] as? NSNumber)?.doubleValue { payload["top_p"] = tp }
        if let k = (parsed["top_k"] as? NSNumber)?.intValue { payload["top_k"] = k }
        if let stop = parsed["stop_sequences"] as? [String], !stop.isEmpty { payload["stop"] = stop }

        if let rawTools = parsed["tools"] as? [[String: Any]], !rawTools.isEmpty {
            let oaiTools: [[String: Any]] = rawTools.compactMap { t in
                guard let name = t["name"] as? String, !name.isEmpty else { return nil }
                var fn: [String: Any] = ["name": name]
                if let desc = t["description"] as? String { fn["description"] = desc }
                if let schema = t["input_schema"] { fn["parameters"] = schema }
                return ["type": "function", "function": fn]
            }
            if !oaiTools.isEmpty { payload["tools"] = oaiTools }
        }

        if let tc = parsed["tool_choice"] as? [String: Any] {
            let tcType = (tc["type"] as? String ?? "auto").lowercased()
            switch tcType {
            case "none":
                payload["tool_choice"] = "none"
            case "any", "required":
                payload["tool_choice"] = "required"
            case "tool":
                if let name = tc["name"] as? String, !name.isEmpty {
                    payload["tool_choice"] = ["type": "function", "function": ["name": name]]
                }
            default:
                payload["tool_choice"] = "auto"
            }
            if (tc["disable_parallel_tool_use"] as? Bool) == true {
                payload["parallel_tool_calls"] = false
            }
        }

        let req: ChatCompletionRequest
        do {
            req = try JSONDecoder().decode(ChatCompletionRequest.self,
                                           from: JSONSerialization.data(withJSONObject: payload))
        } catch {
            sendAnthropicError(connection, status: 400, type: "invalid_request_error",
                               message: "body: \(error.localizedDescription)")
            return
        }

        let msgId = "msg_" + UUID().uuidString.replacingOccurrences(of: "-", with: "")
        let hasActiveTools = !(req.tools?.isEmpty ?? true) && !req.isToolChoiceNone
        let shouldThink = req.effectiveEnableThinking(defaultValue: thinkingEnabled)

        if (payload["stream"] as? Bool) == true {
            sendSSEHeaders(connection)

            func sendEvent(_ event: String, _ json: [String: Any]) {
                guard let data = try? JSONSerialization.data(withJSONObject: json) else { return }
                let head = Data("event: \(event)\ndata: ".utf8)
                var out = Data(capacity: head.count + data.count + 2)
                out.append(head)
                out.append(data)
                out.append(Data("\n\n".utf8))
                connection.send(content: out, isComplete: false, completion: .idempotent)
            }

            sendEvent("message_start", [
                "type": "message_start",
                "message": [
                    "id": msgId, "type": "message", "role": "assistant", "model": replyModel,
                    "content": [Any](),
                    "stop_reason": NSNull(), "stop_sequence": NSNull(),
                    "usage": ["input_tokens": 0, "output_tokens": 0],
                ] as [String: Any],
            ])
            sendEvent("ping", ["type": "ping"])

            let streamParser = QWChatStreamParser(startsInReasoning: shouldThink, allowToolCalls: hasActiveTools)

            final class StreamBlockState: @unchecked Sendable {
                var openBlockType: String? = nil
                var openBlockIndex: Int = -1
                var nextBlockIndex: Int = 0
            }
            let blockState = StreamBlockState()

            func closeCurrentBlockIfOpen() {
                guard blockState.openBlockType != nil else { return }
                sendEvent("content_block_stop", ["type": "content_block_stop", "index": blockState.openBlockIndex])
                blockState.openBlockType = nil
            }

            func emitDelta(_ delta: QWChatStreamParser.Delta) {
                switch delta {
                case .reasoning(let text):
                    guard thinkingEnabled, !text.isEmpty else { return }
                    if blockState.openBlockType != "thinking" {
                        closeCurrentBlockIfOpen()
                        let idx = blockState.nextBlockIndex
                        blockState.nextBlockIndex += 1
                        blockState.openBlockIndex = idx
                        blockState.openBlockType = "thinking"
                        sendEvent("content_block_start", [
                            "type": "content_block_start", "index": idx,
                            "content_block": ["type": "thinking", "thinking": "", "signature": ""] as [String: Any],
                        ])
                    }
                    sendEvent("content_block_delta", [
                        "type": "content_block_delta", "index": blockState.openBlockIndex,
                        "delta": ["type": "thinking_delta", "thinking": text] as [String: Any],
                    ])
                case .content(let text):
                    guard !text.isEmpty else { return }
                    if blockState.openBlockType != "text" {
                        closeCurrentBlockIfOpen()
                        let idx = blockState.nextBlockIndex
                        blockState.nextBlockIndex += 1
                        blockState.openBlockIndex = idx
                        blockState.openBlockType = "text"
                        sendEvent("content_block_start", [
                            "type": "content_block_start", "index": idx,
                            "content_block": ["type": "text", "text": ""] as [String: Any],
                        ])
                    }
                    sendEvent("content_block_delta", [
                        "type": "content_block_delta", "index": blockState.openBlockIndex,
                        "delta": ["type": "text_delta", "text": text] as [String: Any],
                    ])
                }
            }

            let result: ChatCompletionExecutionResult
            do {
                result = try await actor.handleChatCompletion(request: req, onToken: { piece in
                    for d in streamParser.feed(piece) { emitDelta(d) }
                })
            } catch {
                sendEvent("error", [
                    "type": "error",
                    "error": ["type": "api_error", "message": "\(error)"],
                ])
                connection.send(content: Data(), isComplete: true, completion: .contentProcessed({ _ in
                    connection.cancel()
                }))
                return
            }

            for d in streamParser.finish() { emitDelta(d) }
            closeCurrentBlockIfOpen()

            let parsedResult = streamParser.result(tools: hasActiveTools ? req.tools : nil, truncated: result.truncated)
            let effectiveCalls: [ToolCall]?
            if let calls = parsedResult.toolCalls, !calls.isEmpty {
                effectiveCalls = (req.parallelToolCalls == false) ? Array(calls.prefix(1)) : calls
            } else {
                effectiveCalls = nil
            }

            if let calls = effectiveCalls, !calls.isEmpty {
                for call in calls {
                    let idx = blockState.nextBlockIndex
                    blockState.nextBlockIndex += 1
                    sendEvent("content_block_start", [
                        "type": "content_block_start", "index": idx,
                        "content_block": [
                            "type": "tool_use", "id": call.id, "name": call.function.name,
                            "input": [String: Any](),
                        ] as [String: Any],
                    ])
                    sendEvent("content_block_delta", [
                        "type": "content_block_delta", "index": idx,
                        "delta": ["type": "input_json_delta", "partial_json": call.function.arguments] as [String: Any],
                    ])
                    sendEvent("content_block_stop", ["type": "content_block_stop", "index": idx])
                }
            } else if blockState.nextBlockIndex == 0 {
                sendEvent("content_block_start", [
                    "type": "content_block_start", "index": 0,
                    "content_block": ["type": "text", "text": ""] as [String: Any],
                ])
                sendEvent("content_block_stop", ["type": "content_block_stop", "index": 0])
            }

            let (stopReason, stopSeq) = Self.resolveStopReason(
                matchedStop: result.matchedStopSequence,
                hasToolCalls: !(effectiveCalls?.isEmpty ?? true),
                truncated: result.truncated
            )
            let usageDict = Self.buildUsageDict(result: result, thinkingEnabled: thinkingEnabled)

            sendEvent("message_delta", [
                "type": "message_delta",
                "delta": ["stop_reason": stopReason, "stop_sequence": stopSeq],
                "usage": usageDict,
            ])
            sendEvent("message_stop", ["type": "message_stop"])
            connection.send(content: Data(), isComplete: true, completion: .contentProcessed({ _ in
                connection.cancel()
            }))
        } else {
            let result: ChatCompletionExecutionResult
            do {
                result = try await actor.handleChatCompletion(request: req, onToken: nil)
            } catch let mmErr as QWMultimodalError {
                sendAnthropicError(connection, status: 400, type: "invalid_request_error", message: "\(mmErr)")
                return
            }
            let parsedResult = QWToolCallParser.parse(
                text: result.content,
                tools: req.tools,
                startsInThinking: result.startsInThinking,
                truncated: result.truncated,
                allowToolCalls: hasActiveTools
            )
            let effectiveCalls: [ToolCall]?
            if let calls = parsedResult.toolCalls, !calls.isEmpty {
                effectiveCalls = (req.parallelToolCalls == false) ? Array(calls.prefix(1)) : calls
            } else {
                effectiveCalls = nil
            }

            var contentBlocks: [[String: Any]] = []
            if thinkingEnabled, let reasoning = parsedResult.reasoningContent, !reasoning.isEmpty {
                contentBlocks.append(["type": "thinking", "thinking": reasoning, "signature": ""])
            }
            if let text = parsedResult.content, !text.isEmpty {
                contentBlocks.append(["type": "text", "text": text])
            }
            if let calls = effectiveCalls, !calls.isEmpty {
                for call in calls {
                    let inputDict: [String: Any]
                    if let d = call.function.arguments.data(using: .utf8),
                       let obj = (try? JSONSerialization.jsonObject(with: d)) as? [String: Any] {
                        inputDict = obj
                    } else {
                        inputDict = [:]
                    }
                    contentBlocks.append([
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.function.name,
                        "input": inputDict,
                    ])
                }
            }
            if contentBlocks.isEmpty {
                contentBlocks.append(["type": "text", "text": ""])
            }

            let (stopReason, stopSeq) = Self.resolveStopReason(
                matchedStop: result.matchedStopSequence,
                hasToolCalls: !(effectiveCalls?.isEmpty ?? true),
                truncated: result.truncated
            )
            let usageDict = Self.buildUsageDict(result: result, thinkingEnabled: thinkingEnabled)

            let response: [String: Any] = [
                "id": msgId,
                "type": "message",
                "role": "assistant",
                "model": replyModel,
                "content": contentBlocks,
                "stop_reason": stopReason,
                "stop_sequence": stopSeq,
                "usage": usageDict,
            ]
            if let data = try? JSONSerialization.data(withJSONObject: response) {
                sendRawData(connection, status: 200, contentType: "application/json", data: data)
            } else {
                sendAnthropicError(connection, status: 500, type: "api_error",
                                   message: "the answer could not be encoded")
            }
        }
    }

    private static func resolveStopReason(
        matchedStop: String?,
        hasToolCalls: Bool,
        truncated: Bool
    ) -> (String, Any) {
        if let matchedStop {
            return ("stop_sequence", matchedStop)
        }
        if hasToolCalls && !truncated {
            return ("tool_use", NSNull())
        }
        if truncated {
            return ("max_tokens", NSNull())
        }
        return ("end_turn", NSNull())
    }

    private static func buildUsageDict(
        result: ChatCompletionExecutionResult,
        thinkingEnabled: Bool
    ) -> [String: Any] {
        var usage: [String: Any] = [
            "input_tokens": result.promptTokens,
            "output_tokens": result.completionTokens,
            "cache_read_input_tokens": result.cachedTokens,
            "cache_creation_input_tokens": (result.cachedTokens == 0 && result.promptTokens > 0) ? result.promptTokens : 0,
        ]
        if let rToks = result.reasoningTokens, (thinkingEnabled || rToks > 0) {
            usage["output_tokens_details"] = ["thinking_tokens": rToks]
        }
        return usage
    }

    /// Translates one Anthropic message into one or more OpenAI-style messages
    /// (supporting `text`, `thinking`, `image`, `tool_use`, and `tool_result` blocks).
    private static func translateAnthropicMessage(_ msg: [String: Any]) -> [[String: Any]] {
        let role = msg["role"] as? String ?? "user"
        if let s = msg["content"] as? String {
            return [["role": role, "content": s]]
        }
        guard let blocks = msg["content"] as? [[String: Any]] else {
            return [["role": role, "content": ""]]
        }

        var textParts: [String] = []
        var imageUrls: [String] = []
        var reasoningParts: [String] = []
        var toolCalls: [[String: Any]] = []
        var toolResults: [[String: Any]] = []

        for block in blocks {
            let bType = block["type"] as? String ?? ""
            switch bType {
            case "text":
                if let t = block["text"] as? String { textParts.append(t) }
            case "thinking":
                if let th = block["thinking"] as? String { reasoningParts.append(th) }
            case "image":
                if let src = block["source"] as? [String: Any],
                   let data = src["data"] as? String, !data.isEmpty {
                    let mediaType = (src["media_type"] as? String) ?? "image/png"
                    imageUrls.append("data:\(mediaType);base64,\(data)")
                }
            case "tool_use":
                let id = (block["id"] as? String) ?? QWToolCallParser.generateCallId()
                let name = (block["name"] as? String) ?? ""
                let inputObj = block["input"] ?? [String: Any]()
                let argsJson: String
                if let s = inputObj as? String {
                    argsJson = s
                } else if let d = try? JSONSerialization.data(withJSONObject: inputObj, options: [.sortedKeys, .withoutEscapingSlashes]),
                          let str = String(data: d, encoding: .utf8) {
                    argsJson = str
                } else {
                    argsJson = "{}"
                }
                toolCalls.append([
                    "id": id,
                    "type": "function",
                    "function": ["name": name, "arguments": argsJson],
                ])
            case "tool_result":
                let callId = (block["tool_use_id"] as? String) ?? ""
                let resultText = messageText(block["content"])
                toolResults.append([
                    "role": "tool",
                    "tool_call_id": callId,
                    "content": resultText,
                ])
            default:
                break
            }
        }

        var out: [[String: Any]] = []
        if !toolResults.isEmpty {
            out.append(contentsOf: toolResults)
        }
        let joinedText = textParts.joined(separator: "")
        if !imageUrls.isEmpty {
            var contentArr: [[String: Any]] = []
            if !joinedText.isEmpty {
                contentArr.append(["type": "text", "text": joinedText])
            }
            for url in imageUrls {
                contentArr.append(["type": "image_url", "image_url": ["url": url]])
            }
            out.append(["role": role, "content": contentArr])
        } else if !toolCalls.isEmpty || !joinedText.isEmpty || !reasoningParts.isEmpty || out.isEmpty {
            var entry: [String: Any] = ["role": role]
            if !joinedText.isEmpty || toolCalls.isEmpty {
                entry["content"] = joinedText
            }
            if !toolCalls.isEmpty {
                entry["tool_calls"] = toolCalls
            }
            if !reasoningParts.isEmpty {
                entry["reasoning_content"] = reasoningParts.joined(separator: "\n")
            }
            out.append(entry)
        }
        return out
    }

    /// The text a Messages content carries: a string, or the joined
    /// text of its blocks.
    private static func messageText(_ content: Any?) -> String {
        if let s = content as? String { return s }
        if let blocks = content as? [[String: Any]] {
            return blocks.compactMap { block -> String? in
                (block["type"] as? String) == "text" ? (block["text"] as? String) : nil
            }.joined(separator: "")
        }
        return ""
    }

    /// The error shape the Messages API defines: the body says
    /// `type: error` and the reason is inside it.
    private func sendAnthropicError(_ connection: NWConnection, status: Int,
                                    type: String, message: String) {
        let payload: [String: Any] = [
            "type": "error",
            "error": ["type": type, "message": message],
        ]
        if let data = try? JSONSerialization.data(withJSONObject: payload) {
            sendRawData(connection, status: status, contentType: "application/json", data: data)
        } else {
            sendSimpleResponse(connection, status: status, body: message)
        }
    }
}
