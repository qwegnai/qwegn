// QWHttpServer+Completions.swift — Legacy OpenAI POST /v1/completions handler.

import Foundation
import Network
import QWCore
import QWEngine

extension QWHttpServer {

    func handleCompletions(
        connection: NWConnection,
        headers: [String: String] = [:],
        body: Data,
        trace: QWRequestTraceContext? = nil
    ) async throws {
        let baseTrace = trace ?? QWClientIdentity.makeTraceContext(
            method: "POST", route: "/v1/completions", headers: headers, peerEndpoint: connection.endpoint
        )
        let startedAt = Date()
        let req: CompletionRequest
        do {
            var parsedReq = try QWServerRequestDecoder.decode(CompletionRequest.self, from: body)
            parsedReq.httpHeaders = headers
            req = parsedReq
        } catch {
            logger.log(.warn, "HTTP POST \(baseTrace.route) malformed JSON | \(baseTrace.logTag): \(error.localizedDescription)")
            sendOpenAIError(connection, status: 400, message: "Malformed JSON payload: \(error.localizedDescription)")
            return
        }

        if let n = req.n, n > 1 {
            sendOpenAIError(connection, status: 400, message: "n > 1 is not supported (got n=\(n))")
            return
        }
        if let maxT = req.maxTokens, maxT < 0 {
            sendOpenAIError(connection, status: 400, message: "max_tokens must be >= 0")
            return
        }
        if let lp = req.logprobs, lp < 0 {
            sendOpenAIError(connection, status: 400, message: "logprobs must be >= 0")
            return
        }

        let replyModel = resolveReplyModelName(requested: req.model, strictNotFound: false) ?? actor.getModelName()
        let reqId = baseTrace.requestId ?? "cmpl-\(UUID().uuidString.prefix(12))"
        let effectiveTrace = baseTrace.withRequestId(reqId)
        let shouldStop = generationShouldStop(for: connection)
        let created = Int(Date().timeIntervalSince1970)
        var extraHeaders = ["x-request-id: \(reqId)"]
        if let sid = effectiveTrace.sessionId {
            extraHeaders.append("x-session-id: \(sid)")
        }

        @Sendable func logCompletionCancelled() {
            logger.logSimpleRequest(
                method: "POST", path: effectiveTrace.route, statusCode: 499,
                client: effectiveTrace.client, requestId: effectiveTrace.requestId,
                sessionId: effectiveTrace.sessionId, peerHost: effectiveTrace.peerHost,
                detail: "cancelled", totalSec: Date().timeIntervalSince(startedAt)
            )
        }

        if req.stream ?? false {
            let session = makeSSESession(for: connection, format: .sse, extraHeaders: extraHeaders)
            defer { session.stopAll() }
            @Sendable func sendChunk(_ chunk: CompletionResponse) {
                session.sendSSEDataFrame(chunk)
            }

            let result: CompletionExecutionResult
            do {
                result = try await actor.handleTextCompletion(
                    request: req,
                    onStreamReady: {
                        session.begin()
                    },
                    onEchoPrompt: { idx, text, logprobs in
                        session.stopKeepaliveIfRunning()
                        let choice = CompletionChoice(text: text, index: idx, logprobs: logprobs, finishReason: nil)
                        sendChunk(CompletionResponse(id: reqId, model: replyModel, choices: [choice], created: created))
                    },
                    onChoiceToken: { idx, piece, logprobs in
                        session.stopKeepaliveIfRunning()
                        let choice = CompletionChoice(text: piece, index: idx, logprobs: logprobs, finishReason: nil)
                        sendChunk(CompletionResponse(id: reqId, model: replyModel, choices: [choice], created: created))
                    },
                    shouldStop: shouldStop
                )
            } catch {
                if shouldStop() {
                    logCompletionCancelled()
                    return
                }
                if !session.isStarted {
                    sendCompletionError(connection, error: error, trace: effectiveTrace)
                    return
                }
                logger.log(.error, "HTTP POST \(effectiveTrace.route) (stream) failed | \(effectiveTrace.logTag): \(error)")
                session.finishWithOpenAIError(error)
                return
            }

            guard !shouldStop() else {
                logCompletionCancelled()
                return
            }
            let finalChoices = result.choices.map {
                CompletionChoice(text: "", index: $0.index, logprobs: nil, finishReason: $0.finishReason)
            }
            let includeUsage = req.streamOptions?.includeUsage ?? true
            let usage = includeUsage
                ? CompletionUsage(promptTokens: result.totalPromptTokens, completionTokens: result.totalCompletionTokens)
                : nil
            sendChunk(CompletionResponse(
                id: reqId, model: replyModel,
                choices: finalChoices.isEmpty ? [CompletionChoice(text: "", index: 0, finishReason: "stop")] : finalChoices,
                usage: usage, created: created
            ))
            session.finish(doneMarker: true)
            logger.logRequest(
                method: "POST", path: effectiveTrace.route, statusCode: 200,
                promptTokens: result.totalPromptTokens, completionTokens: result.totalCompletionTokens,
                ttftMs: nil, decodeTokPerSec: nil, totalSec: Date().timeIntervalSince(startedAt),
                client: effectiveTrace.client, requestId: effectiveTrace.requestId,
                sessionId: effectiveTrace.sessionId, peerHost: effectiveTrace.peerHost
            )
        } else {
            let result: CompletionExecutionResult
            do {
                result = try await actor.handleTextCompletion(request: req, shouldStop: shouldStop)
            } catch {
                if shouldStop() {
                    logCompletionCancelled()
                    return
                }
                sendCompletionError(connection, error: error, trace: effectiveTrace)
                return
            }
            guard !shouldStop() else {
                logCompletionCancelled()
                return
            }
            let choices = result.choices.map {
                CompletionChoice(text: $0.text, index: $0.index, logprobs: $0.logprobs, finishReason: $0.finishReason)
            }
            let resp = CompletionResponse(
                id: reqId,
                model: replyModel,
                choices: choices,
                usage: CompletionUsage(promptTokens: result.totalPromptTokens, completionTokens: result.totalCompletionTokens),
                created: created
            )
            sendJsonResponse(connection, status: 200, object: resp, extraHeaders: extraHeaders)
            logger.logRequest(
                method: "POST", path: effectiveTrace.route, statusCode: 200,
                promptTokens: result.totalPromptTokens, completionTokens: result.totalCompletionTokens,
                ttftMs: nil, decodeTokPerSec: nil, totalSec: Date().timeIntervalSince(startedAt),
                client: effectiveTrace.client, requestId: effectiveTrace.requestId,
                sessionId: effectiveTrace.sessionId, peerHost: effectiveTrace.peerHost
            )
        }
    }

    func handleTokenizerInfo(connection: NWConnection, model: String?) async throws {
        do {
            let info = try await actor.handleTokenizerInfo(model: model)
            sendJsonResponse(connection, status: 200, object: info)
        } catch {
            sendCompletionError(connection, error: error)
        }
    }

    func handleTokenize(connection: NWConnection, body: Data) async throws {
        let req: TokenizeRequest
        do {
            req = try QWServerRequestDecoder.decode(TokenizeRequest.self, from: body)
        } catch {
            sendOpenAIError(connection, status: 400, message: "Malformed JSON payload: \(error.localizedDescription)")
            return
        }
        do {
            let resp = try await actor.handleTokenize(request: req)
            sendJsonResponse(connection, status: 200, object: resp)
        } catch {
            sendCompletionError(connection, error: error)
        }
    }

    func handleDetokenize(connection: NWConnection, body: Data) async throws {
        let req: DetokenizeRequest
        do {
            req = try QWServerRequestDecoder.decode(DetokenizeRequest.self, from: body)
        } catch {
            sendOpenAIError(connection, status: 400, message: "Malformed JSON payload: \(error.localizedDescription)")
            return
        }
        do {
            let resp = try await actor.handleDetokenize(request: req)
            sendJsonResponse(connection, status: 200, object: resp)
        } catch {
            sendCompletionError(connection, error: error)
        }
    }

    private func sendCompletionError(
        _ connection: NWConnection,
        error: any Error,
        trace: QWRequestTraceContext? = nil
    ) {
        let route = trace?.route ?? "/v1/completions"
        let tag = trace.map { " | \($0.logTag)" } ?? ""
        if let ctxErr = error as? QWServerContextLengthError {
            logger.log(.warn, "HTTP POST \(route) context length exceeded\(tag): \(ctxErr.description)")
            sendOpenAIError(connection, status: 400, message: ctxErr.description,
                            type: "invalid_request_error", code: "context_length_exceeded")
        } else if let inputErr = error as? CompletionInvalidInputError {
            logger.log(.warn, "HTTP POST \(route) invalid request\(tag): \(inputErr.message)")
            sendOpenAIError(connection, status: 400, message: inputErr.message, type: "invalid_request_error")
        } else {
            logger.log(.error, "HTTP POST \(route) failed\(tag): \(error)")
            sendOpenAIError(connection, status: 500, message: "\(error)", type: "server_error")
        }
    }
}
