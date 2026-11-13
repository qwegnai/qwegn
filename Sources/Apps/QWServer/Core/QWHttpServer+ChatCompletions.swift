import Foundation
import Network
import QWCore
import QWEngine
import QWVision

extension QWHttpServer {

    func handleChatCompletions(connection: NWConnection, body: Data) async throws {
        let decoder = JSONDecoder()
        let req: ChatCompletionRequest
        do {
            req = try decoder.decode(ChatCompletionRequest.self, from: body)
        } catch {
            let err = ServerErrorPayload(
                message: "Malformed JSON payload: \(error.localizedDescription)",
                type: "invalid_request_error",
                code: 400
            )
            self.sendJsonResponse(connection, status: 400, object: err)
            return
        }

        if let n = req.n, n > 1 {
            let err = ServerErrorPayload(
                message: "n > 1 is not supported (got n=\(n))",
                type: "invalid_request_error",
                code: 400
            )
            self.sendJsonResponse(connection, status: 400, object: err)
            return
        }

        if req.continueFinalMessage == true && req.addGenerationPrompt == true {
            let err = ServerErrorPayload(
                message: "continue_final_message and add_generation_prompt cannot both be true",
                type: "invalid_request_error",
                code: 400
            )
            self.sendJsonResponse(connection, status: 400, object: err)
            return
        }

        do {
            _ = try QWStructuredOutput.grammar(for: req)
            try self.actor.validateRequestCapabilities(req)
        } catch {
            let err = ServerErrorPayload(message: "\(error)", type: "invalid_request_error", code: 400)
            self.sendJsonResponse(connection, status: 400, object: err)
            return
        }

        let returnModelName = resolveReplyModelName(requested: req.model, strictNotFound: false) ?? self.actor.getModelName()
        let reqId = "chatcmpl-\(UUID().uuidString.prefix(12))"

        let hasActiveTools = !(req.tools?.isEmpty ?? true) && !req.isToolChoiceNone
        let isConstrainedJson = (req.responseFormatType == "json_schema" || req.responseFormatType == "json_object")
            && !hasActiveTools
        let allowToolCalls = hasActiveTools
        let allowReasoningTags = !isConstrainedJson

        if req.isStreaming {
            sendSSEHeaders(connection)

            let sseEncoder = JSONEncoder()
            let ssePrefix = Data("data: ".utf8)
            let sseSuffix = Data("\n\n".utf8)
            func sendChunk(_ chunk: ChatCompletionChunk) {
                if let chunkData = try? sseEncoder.encode(chunk) {
                    var sseData = Data(capacity: ssePrefix.count + chunkData.count + sseSuffix.count)
                    sseData.append(ssePrefix)
                    sseData.append(chunkData)
                    sseData.append(sseSuffix)
                    connection.send(content: sseData, isComplete: false, completion: .idempotent)
                }
            }

            sendChunk(ChatCompletionChunk(id: reqId, model: returnModelName, role: "assistant"))

            let serverConfig = self.actor.getConfig()
            let shouldThink = req.effectiveEnableThinking(defaultValue: serverConfig.enableThinking)
            let streamParser = QWChatStreamParser(
                startsInReasoning: shouldThink,
                allowToolCalls: allowToolCalls,
                allowReasoningTags: allowReasoningTags
            )

            final class StreamLogprobBox: @unchecked Sendable {
                var pending: [ChatLogprobTokenEntry] = []
                func drain() -> ChoiceLogprobs? {
                    guard !pending.isEmpty else { return nil }
                    let out = ChoiceLogprobs(content: pending)
                    pending.removeAll(keepingCapacity: true)
                    return out
                }
            }
            let lpBox = StreamLogprobBox()

            let (keepaliveGate, keepaliveTask) = startSSEKeepalive(connection: connection)
            defer {
                keepaliveGate.stop()
                keepaliveTask.cancel()
            }

            let result: ChatCompletionExecutionResult
            do {
                result = try await self.actor.handleChatCompletion(
                    request: req,
                    onToken: { tokenPiece in
                        keepaliveGate.stop()
                        keepaliveTask.cancel()
                        let deltas = streamParser.feed(tokenPiece)
                        for delta in deltas {
                            let chunkLP = lpBox.drain()
                            switch delta {
                            case .reasoning(let text):
                                sendChunk(ChatCompletionChunk(
                                    id: reqId, model: returnModelName,
                                    reasoningContent: text, logprobs: chunkLP
                                ))
                            case .content(let text):
                                sendChunk(ChatCompletionChunk(
                                    id: reqId, model: returnModelName,
                                    content: text, logprobs: chunkLP
                                ))
                            }
                        }
                    },
                    onTokenLogprob: (req.logprobs == true) ? { lp in
                        lpBox.pending.append(lp)
                    } : nil
                )
            } catch {
                self.logger.log(.error, "HTTP POST /v1/chat/completions (stream) failed: \(error)")
                let errPayload = ServerErrorPayload(message: "\(error)", type: "server_error", code: 500)
                if let errData = try? sseEncoder.encode(errPayload) {
                    var sseErr = Data(capacity: ssePrefix.count + errData.count + sseSuffix.count)
                    sseErr.append(ssePrefix)
                    sseErr.append(errData)
                    sseErr.append(sseSuffix)
                    connection.send(content: sseErr, isComplete: false, completion: .idempotent)
                }
                let doneEvent = "data: [DONE]\n\n"
                connection.send(content: doneEvent.data(using: .utf8), isComplete: true, completion: .contentProcessed({ _ in
                    connection.cancel()
                }))
                return
            }

            let tailDeltas = streamParser.finish()
            for delta in tailDeltas {
                let chunkLP = lpBox.drain()
                switch delta {
                case .reasoning(let text):
                    sendChunk(ChatCompletionChunk(
                        id: reqId, model: returnModelName,
                        reasoningContent: text, logprobs: chunkLP
                    ))
                case .content(let text):
                    sendChunk(ChatCompletionChunk(
                        id: reqId, model: returnModelName,
                        content: text, logprobs: chunkLP
                    ))
                }
            }
            if let remainingLP = lpBox.drain() {
                sendChunk(ChatCompletionChunk(id: reqId, model: returnModelName, logprobs: remainingLP))
            }

            let parsed = streamParser.result(tools: allowToolCalls ? req.tools : nil, truncated: result.truncated)
            let effectiveCalls: [ToolCall]?
            if let calls = parsed.toolCalls, !calls.isEmpty {
                effectiveCalls = (req.parallelToolCalls == false) ? Array(calls.prefix(1)) : calls
            } else {
                effectiveCalls = nil
            }

            if let toolCalls = effectiveCalls, !toolCalls.isEmpty {
                for (idx, call) in toolCalls.enumerated() {
                    let fn = ToolCallChunk.FunctionChunk(name: call.function.name, arguments: call.function.arguments)
                    let callChunk = ToolCallChunk(index: idx, id: call.id, type: "function", function: fn)
                    sendChunk(ChatCompletionChunk(id: reqId, model: returnModelName, toolCall: callChunk))
                }
            }

            let usage = ChatCompletionResponse.Usage(
                promptTokens: result.promptTokens,
                completionTokens: result.completionTokens,
                totalTokens: result.promptTokens + result.completionTokens,
                cachedTokens: result.cachedTokens,
                reasoningTokens: result.reasoningTokens
            )
            sendChunk(ChatCompletionChunk(id: reqId, model: returnModelName, finishReason: parsed.finishReason, usage: usage))

            let doneEvent = "data: [DONE]\n\n"
            connection.send(content: doneEvent.data(using: .utf8), isComplete: true, completion: .contentProcessed({ _ in
                connection.cancel()
            }))
        } else {
            let result = try await self.actor.handleChatCompletion(request: req, onToken: nil)
            let parsed = QWToolCallParser.parse(
                text: result.content,
                tools: req.tools,
                startsInThinking: result.startsInThinking,
                truncated: result.truncated,
                allowToolCalls: allowToolCalls,
                allowReasoningTags: allowReasoningTags
            )
            let effectiveCalls: [ToolCall]?
            if let calls = parsed.toolCalls, !calls.isEmpty {
                effectiveCalls = (req.parallelToolCalls == false) ? Array(calls.prefix(1)) : calls
            } else {
                effectiveCalls = nil
            }
            let resp = ChatCompletionResponse(
                id: reqId,
                model: returnModelName,
                content: parsed.content,
                toolCalls: effectiveCalls,
                reasoningContent: parsed.reasoningContent,
                logprobs: result.logprobs.map { ChoiceLogprobs(content: $0) },
                promptTokens: result.promptTokens,
                completionTokens: result.completionTokens,
                cachedTokens: result.cachedTokens,
                reasoningTokens: result.reasoningTokens,
                finishReason: parsed.finishReason
            )
            self.sendJsonResponse(connection, status: 200, object: resp)
        }
    }
}
