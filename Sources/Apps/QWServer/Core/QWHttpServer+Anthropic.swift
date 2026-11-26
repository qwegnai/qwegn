// QWHttpServer+Anthropic.swift — Anthropic Messages API (/v1/messages and /v1/messages/count_tokens)
// using QWSSESession, QWChatPolicyResolver, and direct Swift request mapping.

import Foundation
import Network
import QWEngine
import QWVision

extension QWHttpServer {

    func handleMessagesCountTokens(
        connection: NWConnection,
        headers: [String: String] = [:],
        body: Data,
        trace: QWRequestTraceContext? = nil
    ) async throws {
        let effectiveTrace = trace ?? QWClientIdentity.makeTraceContext(
            method: "POST", route: "/v1/messages/count_tokens", headers: headers, peerEndpoint: connection.endpoint
        )
        guard let parsedReq = parseAnthropicRequest(
            connection: connection, headers: headers, body: body, requireMaxTokens: false, trace: effectiveTrace
        ) else {
            return
        }
        do {
            let count = try await actor.countChatPromptTokens(request: parsedReq.req)
            let payload: [String: Any] = ["input_tokens": count]
            let data = try JSONSerialization.data(withJSONObject: payload)
            sendRawData(connection, status: 200, contentType: "application/json", data: data)
        } catch {
            sendAnthropicCaughtError(connection, error: error, trace: effectiveTrace)
        }
    }

    func handleMessages(
        connection: NWConnection,
        headers: [String: String] = [:],
        body: Data,
        trace: QWRequestTraceContext? = nil
    ) async throws {
        let baseTrace = trace ?? QWClientIdentity.makeTraceContext(
            method: "POST", route: "/v1/messages", headers: headers, peerEndpoint: connection.endpoint
        )
        guard let parsedReq = parseAnthropicRequest(
            connection: connection, headers: headers, body: body, requireMaxTokens: true, trace: baseTrace
        ) else {
            return
        }
        let req = parsedReq.req
        let replyModel = parsedReq.replyModel
        let thinkingEnabled = parsedReq.thinkingEnabled

        let msgId = req.effectiveRequestId ?? ("msg_" + UUID().uuidString.replacingOccurrences(of: "-", with: ""))
        let effectiveTrace = baseTrace.withRequestId(msgId)
        var hdrs = ["x-request-id: \(msgId)"]
        if let sid = req.effectiveSessionId {
            hdrs.append("x-session-id: \(sid)")
        }
        let extraHeaders = hdrs
        let serverConfig = actor.getConfig()
        let streamPolicy = QWChatPolicyResolver.resolveStreamPolicy(
            for: req,
            defaultEnableThinking: thinkingEnabled,
            defaultMaxTokens: serverConfig.maxTokens,
            modelMaxLength: serverConfig.modelMaxLength,
            supportsThinkingTags: actor.supportsThinkingTags()
        )
        let shouldStop = generationShouldStop(for: connection)

        if req.isStreaming {
            final class PromptTokenBox: @unchecked Sendable {
                private let lock = NSLock()
                private var inputTokens = 0
                var value: Int { lock.withLock { inputTokens } }
                func set(_ n: Int) { lock.withLock { inputTokens = max(0, n) } }
            }
            let promptTokenBox = PromptTokenBox()
            let session = makeSSESession(for: connection, format: .sse, extraHeaders: extraHeaders)
            defer { session.stopAll() }

            @Sendable func sendEvent(_ event: String, _ json: [String: Any]) {
                session.sendSSEJsonObject(event: event, fields: json)
            }

            let streamParser = QWChatStreamParser(
                startsInReasoning: streamPolicy.shouldThink,
                allowToolCalls: streamPolicy.allowToolCalls,
                allowInk2Markers: actor.isInkling2Template() && (!streamPolicy.isJsonResponseFormat || streamPolicy.shouldThink)
            )

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
                result = try await actor.handleChatCompletion(
                    request: req,
                    trace: effectiveTrace,
                    onStreamReady: { [weak self] in
                        guard let self else { return }
                        let readyPolicy = QWChatPolicyResolver.resolveStreamPolicy(
                            for: req,
                            defaultEnableThinking: thinkingEnabled,
                            defaultMaxTokens: serverConfig.maxTokens,
                            modelMaxLength: serverConfig.modelMaxLength,
                            supportsThinkingTags: self.actor.supportsThinkingTags()
                        )
                        streamParser.setStartsInReasoning(
                            readyPolicy.shouldThink,
                            allowInk2Markers: self.actor.isInkling2Template() && (!readyPolicy.isJsonResponseFormat || readyPolicy.shouldThink)
                        )
                        session.begin()
                        sendEvent("message_start", [
                            "type": "message_start",
                            "message": [
                                "id": msgId, "type": "message", "role": "assistant", "model": replyModel,
                                "content": [Any](),
                                "stop_reason": NSNull(), "stop_sequence": NSNull(),
                                "usage": ["input_tokens": promptTokenBox.value, "output_tokens": 0],
                            ] as [String: Any],
                        ])
                        sendEvent("ping", ["type": "ping"])
                    },
                    onPromptTokensComputed: { count in
                        promptTokenBox.set(count)
                    },
                    onToken: { piece in
                        session.stopKeepaliveIfRunning()
                        for d in streamParser.feed(piece) { emitDelta(d) }
                    },
                    shouldStop: shouldStop
                )
            } catch {
                if shouldStop() { return }
                if !session.isStarted {
                    sendAnthropicCaughtError(connection, error: error, trace: effectiveTrace)
                    return
                }
                logger.log(.error, "HTTP POST \(effectiveTrace.route) (stream) failed | \(effectiveTrace.logTag): \(error)")
                session.finishWithNamedSSEError(
                    event: "error",
                    fields: ["type": "error", "error": ["type": "api_error", "message": "\(error)"]],
                    doneMarker: false
                )
                return
            }

            guard !shouldStop() else { return }
            for d in streamParser.finish() { emitDelta(d) }
            closeCurrentBlockIfOpen()

            let parsedResult = streamParser.result(
                tools: streamPolicy.allowToolCalls ? req.tools : nil,
                truncated: result.truncated
            )
            let effectiveCalls = QWChatPolicyResolver.capToolCalls(
                parsedResult.toolCalls,
                parallelToolCalls: req.parallelToolCalls
            )

            if let calls = effectiveCalls {
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
            let usageDict = Self.buildUsageDict(
                result: result, thinkingEnabled: thinkingEnabled, cachingRequested: req.effectiveCachePrompt
            )

            sendEvent("message_delta", [
                "type": "message_delta",
                "delta": ["stop_reason": stopReason, "stop_sequence": stopSeq],
                "usage": usageDict,
            ])
            sendEvent("message_stop", ["type": "message_stop"])
            session.finish(doneMarker: false)
        } else {
            let result: ChatCompletionExecutionResult
            do {
                result = try await actor.handleChatCompletion(
                    request: req, trace: effectiveTrace, onToken: nil, shouldStop: shouldStop
                )
            } catch {
                if shouldStop() { return }
                sendAnthropicCaughtError(connection, error: error, trace: effectiveTrace)
                return
            }
            guard !shouldStop() else { return }
            let parsedResult = QWToolCallParser.parse(
                text: result.content,
                tools: req.tools,
                startsInThinking: result.startsInThinking,
                truncated: result.truncated,
                allowToolCalls: streamPolicy.allowToolCalls,
                allowInk2Markers: actor.isInkling2Template() && (!streamPolicy.isJsonResponseFormat || result.startsInThinking)
            )
            let effectiveCalls = QWChatPolicyResolver.capToolCalls(
                parsedResult.toolCalls,
                parallelToolCalls: req.parallelToolCalls
            )

            var contentBlocks: [[String: Any]] = []
            if thinkingEnabled, let reasoning = parsedResult.reasoningContent, !reasoning.isEmpty {
                contentBlocks.append(["type": "thinking", "thinking": reasoning, "signature": ""])
            }
            if let text = parsedResult.content, !text.isEmpty {
                contentBlocks.append(["type": "text", "text": text])
            }
            if let calls = effectiveCalls {
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
            let usageDict = Self.buildUsageDict(
                result: result, thinkingEnabled: thinkingEnabled, cachingRequested: req.effectiveCachePrompt
            )

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
                sendRawData(connection, status: 200, contentType: "application/json", data: data, extraHeaders: extraHeaders)
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
        thinkingEnabled: Bool,
        cachingRequested: Bool = true
    ) -> [String: Any] {
        let createdTokens = (cachingRequested && result.cacheActive)
            ? max(0, result.promptTokens - result.cachedTokens)
            : 0
        var usage: [String: Any] = [
            "input_tokens": result.promptTokens,
            "output_tokens": result.completionTokens,
            "cache_read_input_tokens": result.cachedTokens,
            "cache_creation_input_tokens": createdTokens,
        ]
        if let rToks = result.reasoningTokens, (thinkingEnabled || rToks > 0) {
            usage["output_tokens_details"] = ["thinking_tokens": rToks]
        }
        return usage
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

    private func sendAnthropicCaughtError(
        _ connection: NWConnection,
        error: any Error,
        trace: QWRequestTraceContext? = nil
    ) {
        let route = trace?.route ?? "/v1/messages"
        let tag = trace.map { " | \($0.logTag)" } ?? ""
        if let valErr = error as? AnthropicValidationError {
            logger.log(.warn, "HTTP POST \(route) invalid request\(tag): \(valErr.message)")
            sendAnthropicError(connection, status: valErr.status, type: valErr.type, message: valErr.message)
        } else if let ctxErr = error as? QWServerContextLengthError {
            logger.log(.warn, "HTTP POST \(route) context length exceeded\(tag): \(ctxErr.description)")
            sendAnthropicError(connection, status: 400, type: "invalid_request_error", message: ctxErr.description)
        } else if error is QWMultimodalError || error is QWJSONSchemaError
                    || error is QWMetadataValidationError || error is QWChatTemplateError
                    || error is DecodingError {
            logger.log(.warn, "HTTP POST \(route) invalid request\(tag): \(error)")
            sendAnthropicError(connection, status: 400, type: "invalid_request_error", message: "\(error)")
        } else {
            logger.log(.error, "HTTP POST \(route) failed\(tag): \(error)")
            sendAnthropicError(connection, status: 500, type: "api_error", message: "\(error)")
        }
    }

    private func parseAnthropicRequest(
        connection: NWConnection,
        headers: [String: String],
        body: Data,
        requireMaxTokens: Bool,
        trace: QWRequestTraceContext? = nil
    ) -> (req: ChatCompletionRequest, replyModel: String, thinkingEnabled: Bool)? {
        let parsed: AnthropicMessagesRequest
        do {
            parsed = try QWServerRequestDecoder.decode(AnthropicMessagesRequest.self, from: body)
        } catch let decErr as DecodingError {
            let msg: String
            switch decErr {
            case .typeMismatch(_, let ctx) where ctx.codingPath.isEmpty,
                 .dataCorrupted(let ctx) where ctx.codingPath.isEmpty:
                msg = "request body must be a JSON object"
            case .typeMismatch(_, let ctx) where ctx.codingPath.first?.stringValue == "messages":
                msg = "messages: expected required field present"
            default:
                msg = "\(decErr)"
            }
            let tag = trace.map { " | \($0.logTag)" } ?? ""
            logger.log(.warn, "HTTP POST \(trace?.route ?? "/v1/messages") invalid JSON\(tag): \(msg)")
            sendAnthropicError(connection, status: 400, type: "invalid_request_error", message: msg)
            return nil
        } catch {
            let tag = trace.map { " | \($0.logTag)" } ?? ""
            logger.log(.warn, "HTTP POST \(trace?.route ?? "/v1/messages") invalid JSON\(tag)")
            sendAnthropicError(connection, status: 400, type: "invalid_request_error",
                               message: "request body must be a JSON object")
            return nil
        }

        let requested = (parsed.model ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let mapped: (request: ChatCompletionRequest, thinkingEnabled: Bool)
        do {
            // Check messages & max_tokens before model lookup to match Anthropic validation ordering.
            guard let msgs = parsed.messages, !msgs.isEmpty else {
                throw AnthropicValidationError(message: "messages: expected required field present")
            }
            if requireMaxTokens {
                guard let mt = QWSchemaMetadataCoercion.strictInt(parsed.maxTokens), mt > 0 else {
                    throw AnthropicValidationError(message: "max_tokens: expect integer greater than 0")
                }
            } else if let rawMT = parsed.maxTokens, rawMT != .null, rawMT != .undefined,
                      QWSchemaMetadataCoercion.strictInt(rawMT) == nil {
                throw AnthropicValidationError(message: "max_tokens: expect integer")
            }
            guard let replyModel = resolveReplyModelName(requested: requested, strictNotFound: true) else {
                throw AnthropicValidationError(
                    status: 404,
                    type: "not_found_error",
                    message: "model: model \"\(requested)\" not found"
                )
            }
            mapped = try parsed.toChatCompletionRequest(
                replyModel: replyModel,
                requireMaxTokens: requireMaxTokens,
                headers: headers
            )
            try mapped.request.validateMetadata()
            try actor.validateRequestCapabilities(mapped.request)
            return (mapped.request, replyModel, mapped.thinkingEnabled)
        } catch {
            sendAnthropicCaughtError(connection, error: error, trace: trace)
            return nil
        }
    }
}
