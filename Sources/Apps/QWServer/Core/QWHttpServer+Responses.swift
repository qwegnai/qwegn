// QWHttpServer+Responses.swift — OpenAI POST /v1/responses, GET/DELETE /v1/responses/:id,
// previous_response_id chaining, background mode, and SSE events via QWSSESession.

import Foundation
import Network
import QWCore
import QWEngine
import QWToken
import QWVision

extension QWHttpServer {

    func handleCreateResponse(
        connection: NWConnection,
        headers: [String: String] = [:],
        body: Data,
        trace: QWRequestTraceContext? = nil
    ) async throws {
        let baseTrace = trace ?? QWClientIdentity.makeTraceContext(
            method: "POST", route: "/v1/responses", headers: headers, peerEndpoint: connection.endpoint
        )
        let req: OpenAIResponseCreateRequest
        do {
            req = try QWServerRequestDecoder.decode(OpenAIResponseCreateRequest.self, from: body)
        } catch {
            logger.log(.warn, "HTTP POST \(baseTrace.route) malformed JSON | \(baseTrace.logTag): \(error.localizedDescription)")
            sendOpenAIError(connection, status: 400, message: "Malformed JSON payload: \(error.localizedDescription)")
            return
        }

        if let maxOut = req.maxOutputTokens, maxOut <= 0 {
            sendOpenAIError(connection, status: 400, message: "max_output_tokens must be greater than 0")
            return
        }
        if (req.background ?? false) && (req.store == false) {
            sendOpenAIError(connection, status: 400, message: "background mode requires store to be true")
            return
        }

        let prevEntry: QWResponsesStore.StoredResponse?
        if let prevId = req.previousResponseId?.trimmingCharacters(in: .whitespacesAndNewlines), !prevId.isEmpty {
            guard let found = responsesStore.get(prevId) else {
                sendOpenAIError(connection, status: 400,
                                message: "Previous response '\(prevId)' was not found.",
                                code: "previous_response_not_found")
                return
            }
            guard found.resource.status == "completed" || found.resource.status == "incomplete" else {
                sendOpenAIError(connection, status: 400,
                                message: "Previous response '\(prevId)' is in status '\(found.resource.status)'.")
                return
            }
            prevEntry = found
        } else {
            prevEntry = nil
        }

        let inputMessages: [ChatMessage]
        do {
            inputMessages = try OpenAIResponsesMapper.parseInput(req.input, hasPrevious: prevEntry != nil)
        } catch {
            sendOpenAIError(connection, status: 400, message: "\(error)")
            return
        }

        var conversationMessages = prevEntry?.historyMessages ?? []
        conversationMessages.append(contentsOf: inputMessages)
        guard !conversationMessages.isEmpty else {
            sendOpenAIError(connection, status: 400, message: "input must not be empty")
            return
        }

        var fullMessages = conversationMessages
        if let instructions = req.instructions, !instructions.isEmpty {
            fullMessages.insert(ChatMessage(role: "system", text: instructions), at: 0)
        }

        let (normalizedTools, resourceTools, normalizedToolChoice, resourceToolChoice) =
            OpenAIResponsesMapper.normalizeTools(tools: req.tools, toolChoice: req.toolChoice, maxToolCalls: req.maxToolCalls)

        let chatReq: ChatCompletionRequest
        let thinkingExplicitlyDisabled: Bool
        do {
            (chatReq, thinkingExplicitlyDisabled) = try OpenAIResponsesMapper.buildChatRequest(
                from: req, messages: fullMessages, tools: normalizedTools,
                toolChoice: normalizedToolChoice, headers: headers
            )
            try chatReq.validateMetadata()
            _ = try QWStructuredOutput.grammar(for: chatReq)
            try actor.validateRequestCapabilities(chatReq)
        } catch {
            sendOpenAIError(connection, status: 400, message: "\(error)")
            return
        }

        let respId = chatReq.effectiveRequestId ?? ("resp_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))
        let effectiveTrace = baseTrace.withRequestId(respId)
        let hasClientRequestId = (chatReq.effectiveRequestId != nil)
        let replyModel = resolveReplyModelName(requested: req.model, strictNotFound: false) ?? actor.getModelName()
        var extraHeaders = ["x-request-id: \(respId)"]
        if let sid = chatReq.effectiveSessionId { extraHeaders.append("x-session-id: \(sid)") }

        let serverCfg = actor.getConfig()
        let template = OpenAIResponseTemplate(
            id: respId, createdAt: Int(Date().timeIntervalSince1970), model: replyModel, req: req,
            tools: resourceTools, toolChoice: resourceToolChoice,
            resolvedTemperature: req.temperature ?? serverCfg.temperature,
            resolvedTopP: req.topP ?? serverCfg.topP,
            textFormat: OpenAIResponsesMapper.resourceTextConfig(from: req.text)
        )
        let shouldStore = req.store ?? true
        let streamPolicy = QWChatPolicyResolver.resolveStreamPolicy(
            for: chatReq,
            defaultEnableThinking: serverCfg.enableThinking,
            defaultMaxTokens: serverCfg.maxTokens,
            modelMaxLength: serverCfg.modelMaxLength,
            supportsThinkingTags: actor.supportsThinkingTags(),
            thinkingExplicitlyDisabled: thinkingExplicitlyDisabled
        )

        if (req.background ?? false) && !(req.stream ?? false) {
            let queued = template.makeResource(status: "queued", completedAt: nil, output: [], usage: nil)
            let queuedEntry = QWResponsesStore.StoredResponse(resource: queued, historyMessages: conversationMessages)
            if hasClientRequestId {
                guard responsesStore.putIfAbsent(queuedEntry) else {
                    sendOpenAIError(connection, status: 409, message: "Duplicate response id '\(respId)'.", code: "duplicate_request_id")
                    return
                }
            } else {
                responsesStore.put(queuedEntry)
            }
            let bgAllowInk2 = actor.isInkling2Template() && (!streamPolicy.isJsonResponseFormat || streamPolicy.shouldThink)
            let parentNodeId = prevEntry?.sessionNodeId
            let bgTask = Task { [weak self] in
                guard let self else { return }
                await self.runBackgroundResponse(
                    template: template, chatReq: chatReq, baseHistory: conversationMessages,
                    parentSessionNodeId: parentNodeId,
                    hasActiveTools: streamPolicy.hasActiveTools, allowReasoningTags: streamPolicy.allowReasoningTags,
                    thinkingExplicitlyDisabled: thinkingExplicitlyDisabled, allowInk2Markers: bgAllowInk2,
                    trace: effectiveTrace
                )
            }
            responsesStore.attachTask(id: respId, task: bgTask)
            sendJsonResponse(connection, status: 200, object: queued, extraHeaders: extraHeaders)
            return
        }

        let reservedClientEntry = hasClientRequestId && shouldStore
        if reservedClientEntry {
            let reserved = template.makeResource(status: "in_progress", completedAt: nil, output: [], usage: nil)
            guard responsesStore.putIfAbsent(QWResponsesStore.StoredResponse(resource: reserved, historyMessages: conversationMessages)) else {
                sendOpenAIError(connection, status: 409, message: "Duplicate response id '\(respId)'.", code: "duplicate_request_id")
                return
            }
        }

        if req.stream ?? false {
            try await runStreamingResponse(
                connection: connection, extraHeaders: extraHeaders, template: template,
                chatReq: chatReq, baseHistory: conversationMessages, shouldStore: shouldStore,
                parentSessionNodeId: prevEntry?.sessionNodeId,
                reservedClientEntry: reservedClientEntry,
                streamPolicy: streamPolicy,
                thinkingExplicitlyDisabled: thinkingExplicitlyDisabled,
                trace: effectiveTrace
            )
            return
        }

        let shouldStop = generationShouldStop(for: connection)
        let result: ChatCompletionExecutionResult
        do {
            result = try await actor.handleChatCompletion(
                request: chatReq, trace: effectiveTrace, onToken: nil, shouldStop: shouldStop
            )
        } catch {
            if reservedClientEntry { _ = responsesStore.delete(respId) }
            if shouldStop() { return }
            sendChatError(connection, error: error, path: effectiveTrace.route, trace: effectiveTrace)
            return
        }
        if shouldStop() {
            if reservedClientEntry { _ = responsesStore.delete(respId) }
            return
        }

        let (finalResource, nextHistory) = OpenAIResponsesMapper.finalizeResponse(
            template: template, req: req, chatReq: chatReq, result: result, baseHistory: conversationMessages,
            hasActiveTools: streamPolicy.hasActiveTools, allowReasoningTags: streamPolicy.allowReasoningTags,
            thinkingExplicitlyDisabled: thinkingExplicitlyDisabled,
            allowInk2Markers: actor.isInkling2Template()
        )
        if shouldStore {
            let node = try? sessionTree.checkpoint(
                parentId: prevEntry?.sessionNodeId,
                label: respId,
                modelName: replyModel,
                kvMode: String(describing: serverCfg.kvMode),
                cacheSalt: chatReq.cacheSalt,
                tokenIds: [],
                metadata: ["response_id": respId]
            )
            responsesStore.put(QWResponsesStore.StoredResponse(
                resource: finalResource,
                historyMessages: nextHistory,
                sessionNodeId: node?.id
            ))
        }
        sendJsonResponse(connection, status: 200, object: finalResource, extraHeaders: extraHeaders)
    }

    func handleGetResponse(connection: NWConnection, responseId: String) {
        guard let stored = responsesStore.get(responseId) else {
            sendOpenAIError(connection, status: 404, message: "Response with id '\(responseId)' not found.", code: "not_found")
            return
        }
        sendJsonResponse(connection, status: 200, object: stored.resource)
    }

    func handleDeleteResponse(connection: NWConnection, responseId: String) {
        guard responsesStore.delete(responseId) else {
            sendOpenAIError(connection, status: 404, message: "Response with id '\(responseId)' not found.", code: "not_found")
            return
        }
        let payload: [String: Any] = ["id": responseId, "object": "response", "deleted": true]
        if let data = try? JSONSerialization.data(withJSONObject: payload) {
            sendRawData(connection, status: 200, contentType: "application/json", data: data)
        } else {
            sendSimpleResponse(connection, status: 200, body: "{\"id\":\"\(responseId)\",\"object\":\"response\",\"deleted\":true}")
        }
    }

    private func runBackgroundResponse(
        template: OpenAIResponseTemplate,
        chatReq: ChatCompletionRequest,
        baseHistory: [ChatMessage],
        parentSessionNodeId: String?,
        hasActiveTools: Bool,
        allowReasoningTags: Bool,
        thinkingExplicitlyDisabled: Bool,
        allowInk2Markers: Bool,
        trace: QWRequestTraceContext? = nil
    ) async {
        do {
            let result = try await actor.handleChatCompletion(
                request: chatReq, trace: trace, onToken: nil, shouldStop: { Task.isCancelled }
            )
            if Task.isCancelled { return }
            let (finalResource, nextHistory) = OpenAIResponsesMapper.finalizeResponse(
                template: template, req: template.req, chatReq: chatReq, result: result, baseHistory: baseHistory,
                hasActiveTools: hasActiveTools, allowReasoningTags: allowReasoningTags,
                thinkingExplicitlyDisabled: thinkingExplicitlyDisabled,
                allowInk2Markers: allowInk2Markers
            )
            let node = try? sessionTree.checkpoint(
                parentId: parentSessionNodeId,
                label: template.id,
                modelName: template.model,
                kvMode: String(describing: actor.getConfig().kvMode),
                cacheSalt: chatReq.cacheSalt,
                tokenIds: [],
                metadata: ["response_id": template.id]
            )
            responsesStore.update(id: template.id, resource: finalResource, historyMessages: nextHistory, sessionNodeId: node?.id)
        } catch {
            if Task.isCancelled { return }
            let code = (error is QWServerContextLengthError) ? "context_length_exceeded" : "server_error"
            let failed = template.makeResource(
                status: "failed", completedAt: nil, output: [], usage: nil,
                error: OpenAIResponseResource.ErrorInfo(code: code, message: "\(error)")
            )
            responsesStore.update(id: template.id, resource: failed)
        }
    }

    private func runStreamingResponse(
        connection: NWConnection,
        extraHeaders: [String],
        template: OpenAIResponseTemplate,
        chatReq: ChatCompletionRequest,
        baseHistory: [ChatMessage],
        shouldStore: Bool,
        parentSessionNodeId: String?,
        reservedClientEntry: Bool,
        streamPolicy: QWChatStreamPolicy,
        thinkingExplicitlyDisabled: Bool,
        trace: QWRequestTraceContext? = nil
    ) async throws {
        let shouldStop = generationShouldStop(for: connection)
        let serverConfig = actor.getConfig()
        let streamParser = QWChatStreamParser(
            startsInReasoning: streamPolicy.shouldThink,
            allowToolCalls: streamPolicy.allowToolCalls,
            allowReasoningTags: streamPolicy.allowReasoningTags,
            allowInk2Markers: actor.isInkling2Template() && (!streamPolicy.isJsonResponseFormat || streamPolicy.shouldThink)
        )
        let session = makeSSESession(for: connection, format: .sse, extraHeaders: extraHeaders)
        let emitter = ResponsesSSECoordinator(session: session, template: template)
        defer { session.stopAll() }

        let result: ChatCompletionExecutionResult
        do {
            result = try await actor.handleChatCompletion(
                request: chatReq,
                trace: trace,
                onStreamReady: { [weak self] in
                    guard let self else { return }
                    let readyPolicy = QWChatPolicyResolver.resolveStreamPolicy(
                        for: chatReq,
                        defaultEnableThinking: serverConfig.enableThinking,
                        defaultMaxTokens: serverConfig.maxTokens,
                        modelMaxLength: serverConfig.modelMaxLength,
                        supportsThinkingTags: self.actor.supportsThinkingTags(),
                        thinkingExplicitlyDisabled: thinkingExplicitlyDisabled
                    )
                    streamParser.setStartsInReasoning(
                        readyPolicy.shouldThink,
                        allowReasoningTags: readyPolicy.allowReasoningTags,
                        allowInk2Markers: self.actor.isInkling2Template() && (!readyPolicy.isJsonResponseFormat || readyPolicy.shouldThink)
                    )
                    session.begin()
                    emitter.emitLifecycleEvent(type: "response.created", status: "in_progress")
                    emitter.emitLifecycleEvent(type: "response.in_progress", status: "in_progress")
                },
                onToken: { piece in
                    session.stopKeepaliveIfRunning()
                    for delta in streamParser.feed(piece) {
                        emitter.handleDelta(delta, allowReasoning: !thinkingExplicitlyDisabled)
                    }
                },
                shouldStop: shouldStop
            )
        } catch {
            if reservedClientEntry { _ = responsesStore.delete(template.id) }
            if shouldStop() { return }
            if !session.isStarted {
                sendChatError(connection, error: error, path: trace?.route ?? "/v1/responses", trace: trace)
                return
            }
            let tag = trace.map { " | \($0.logTag)" } ?? ""
            logger.log(.error, "HTTP POST \(trace?.route ?? "/v1/responses") (stream) failed\(tag): \(error)")
            let code = (error is QWServerContextLengthError) ? "context_length_exceeded" : "server_error"
            let failed = template.makeResource(
                status: "failed", completedAt: nil, output: [], usage: nil,
                error: OpenAIResponseResource.ErrorInfo(code: code, message: "\(error)")
            )
            emitter.finishWithFailed(failed)
            return
        }

        if shouldStop() {
            if reservedClientEntry { _ = responsesStore.delete(template.id) }
            return
        }
        for delta in streamParser.finish() {
            emitter.handleDelta(delta, allowReasoning: !thinkingExplicitlyDisabled)
        }
        let parsed = streamParser.result(
            tools: streamPolicy.allowToolCalls ? chatReq.tools : nil,
            truncated: result.truncated
        )
        let cappedCalls = OpenAIResponsesMapper.capToolCalls(
            parsed.toolCalls, parallelToolCalls: template.req.parallelToolCalls, maxToolCalls: template.req.maxToolCalls
        )
        let itemStatus = result.truncated ? "incomplete" : "completed"
        emitter.finishContentItems(status: itemStatus)
        if let calls = cappedCalls {
            emitter.emitFunctionCalls(calls, status: itemStatus)
        }
        emitter.ensureAtLeastOneItem(status: itemStatus)

        let (finalResource, nextHistory) = OpenAIResponsesMapper.buildFinalizedResource(
            template: template, parsed: parsed, effectiveCalls: cappedCalls, result: result,
            baseHistory: baseHistory, thinkingExplicitlyDisabled: thinkingExplicitlyDisabled,
            streamedItemIDs: emitter.snapshotItemIDs
        )
        if shouldStore {
            let node = try? sessionTree.checkpoint(
                parentId: parentSessionNodeId,
                label: template.id,
                modelName: template.model,
                kvMode: String(describing: serverConfig.kvMode),
                cacheSalt: chatReq.cacheSalt,
                tokenIds: [],
                metadata: ["response_id": template.id]
            )
            responsesStore.put(QWResponsesStore.StoredResponse(
                resource: finalResource,
                historyMessages: nextHistory,
                sessionNodeId: node?.id
            ))
        }
        let terminalType = result.truncated ? "response.incomplete" : "response.completed"
        emitter.finishWithCompleted(type: terminalType, resource: finalResource)
    }
}

private final class ResponsesSSECoordinator: @unchecked Sendable {
    private struct LifecyclePayload: Encodable {
        let type: String
        let sequenceNumber: Int
        let response: OpenAIResponseResource
        enum CodingKeys: String, CodingKey {
            case type, response
            case sequenceNumber = "sequence_number"
        }
    }

    private let lock = NSLock()
    private let session: QWSSESession
    private let template: OpenAIResponseTemplate
    private var seq = 0
    private var nextOutputIndex = 0
    private var reasoningId: String? = nil
    private var reasoningIndex: Int? = nil
    private var reasoningText = ""
    private var reasoningClosed = false
    private var messageId: String? = nil
    private var messageIndex: Int? = nil
    private var messageText = ""
    private var messageClosed = false
    private var functionCallIds: [String] = []

    init(session: QWSSESession, template: OpenAIResponseTemplate) {
        self.session = session
        self.template = template
    }

    var snapshotItemIDs: OpenAIResponsesMapper.StreamedItemIDs {
        lock.withLock {
            OpenAIResponsesMapper.StreamedItemIDs(
                reasoningId: reasoningId,
                messageId: messageId,
                functionCallIds: functionCallIds
            )
        }
    }

    func emitLifecycleEvent(type: String, status: String) {
        let resource = template.makeResource(status: status, completedAt: nil, output: [], usage: nil)
        let s = lock.withLock { () -> Int in let v = seq; seq += 1; return v }
        session.sendSSENamedEvent(event: type, object: LifecyclePayload(type: type, sequenceNumber: s, response: resource))
    }

    func finishWithFailed(_ resource: OpenAIResponseResource) {
        let s = lock.withLock { () -> Int in let v = seq; seq += 1; return v }
        session.finishWithNamedSSEErrorObject(
            event: "response.failed",
            object: LifecyclePayload(type: "response.failed", sequenceNumber: s, response: resource),
            doneMarker: true
        )
    }

    func finishWithCompleted(type: String, resource: OpenAIResponseResource) {
        let s = lock.withLock { () -> Int in let v = seq; seq += 1; return v }
        session.sendSSENamedEvent(event: type, object: LifecyclePayload(type: type, sequenceNumber: s, response: resource))
        session.finish(doneMarker: true)
    }

    func handleDelta(_ delta: QWChatStreamParser.Delta, allowReasoning: Bool) {
        switch delta {
        case .reasoning(let text):
            guard allowReasoning, !text.isEmpty else { return }
            let openInfo = lock.withLock { () -> (id: String, idx: Int, newlyOpened: Bool) in
                if let rid = reasoningId, let ridx = reasoningIndex {
                    reasoningText += text
                    return (rid, ridx, false)
                }
                let rid = "rs_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(16)
                let ridx = nextOutputIndex
                nextOutputIndex += 1
                reasoningId = rid
                reasoningIndex = ridx
                reasoningText = text
                return (rid, ridx, true)
            }
            if openInfo.newlyOpened {
                emitJsonEvent("response.output_item.added", [
                    "output_index": openInfo.idx,
                    "item": ["type": "reasoning", "id": openInfo.id, "summary": [Any]()] as [String: Any]
                ])
                emitJsonEvent("response.reasoning_summary_part.added", [
                    "item_id": openInfo.id, "output_index": openInfo.idx, "summary_index": 0,
                    "part": ["type": "summary_text", "text": ""]
                ])
            }
            emitJsonEvent("response.reasoning_summary_text.delta", [
                "item_id": openInfo.id, "output_index": openInfo.idx, "summary_index": 0, "delta": text
            ])
        case .content(let text):
            guard !text.isEmpty else { return }
            closeReasoningIfNeeded()
            let openInfo = lock.withLock { () -> (id: String, idx: Int, newlyOpened: Bool) in
                if let mid = messageId, let midx = messageIndex {
                    messageText += text
                    return (mid, midx, false)
                }
                let mid = "msg_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(16)
                let midx = nextOutputIndex
                nextOutputIndex += 1
                messageId = mid
                messageIndex = midx
                messageText = text
                return (mid, midx, true)
            }
            if openInfo.newlyOpened {
                emitJsonEvent("response.output_item.added", [
                    "output_index": openInfo.idx,
                    "item": [
                        "type": "message", "id": openInfo.id, "status": "in_progress",
                        "role": "assistant", "content": [Any]()
                    ] as [String: Any]
                ])
                emitJsonEvent("response.content_part.added", [
                    "item_id": openInfo.id, "output_index": openInfo.idx, "content_index": 0,
                    "part": ["type": "output_text", "text": "", "annotations": [Any]()] as [String: Any]
                ])
            }
            emitJsonEvent("response.output_text.delta", [
                "item_id": openInfo.id, "output_index": openInfo.idx, "content_index": 0, "delta": text
            ])
        }
    }

    func finishContentItems(status: String) {
        closeReasoningIfNeeded()
        closeMessageIfNeeded(status: status)
    }

    func ensureAtLeastOneItem(status: String) {
        let count = lock.withLock { nextOutputIndex }
        guard count == 0 else { return }
        let (mid, midx) = lock.withLock { () -> (String, Int) in
            let id = "msg_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(16)
            let idx = nextOutputIndex
            nextOutputIndex += 1
            messageId = id
            return (id, idx)
        }
        let part: [String: Any] = ["type": "output_text", "text": "", "annotations": [Any]()]
        emitJsonEvent("response.output_item.added", [
            "output_index": midx,
            "item": ["type": "message", "id": mid, "status": "in_progress", "role": "assistant", "content": [Any]()] as [String: Any]
        ])
        emitJsonEvent("response.content_part.added", [
            "item_id": mid, "output_index": midx, "content_index": 0, "part": part
        ])
        emitJsonEvent("response.output_text.done", [
            "item_id": mid, "output_index": midx, "content_index": 0, "text": ""
        ])
        emitJsonEvent("response.content_part.done", [
            "item_id": mid, "output_index": midx, "content_index": 0, "part": part
        ])
        emitJsonEvent("response.output_item.done", [
            "output_index": midx,
            "item": ["type": "message", "id": mid, "status": status, "role": "assistant", "content": [part]] as [String: Any]
        ])
    }

    func emitFunctionCalls(_ calls: [ToolCall], status: String) {
        for call in calls {
            let fcid = "fc_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(16)
            let idx = lock.withLock { () -> Int in
                let v = nextOutputIndex
                nextOutputIndex += 1
                functionCallIds.append(fcid)
                return v
            }
            emitJsonEvent("response.output_item.added", [
                "output_index": idx,
                "item": [
                    "type": "function_call", "id": fcid, "call_id": call.id,
                    "name": call.function.name, "arguments": "", "status": "in_progress"
                ] as [String: Any]
            ])
            emitJsonEvent("response.function_call_arguments.delta", [
                "item_id": fcid, "output_index": idx, "delta": call.function.arguments
            ])
            emitJsonEvent("response.function_call_arguments.done", [
                "item_id": fcid, "output_index": idx, "arguments": call.function.arguments
            ])
            emitJsonEvent("response.output_item.done", [
                "output_index": idx,
                "item": [
                    "type": "function_call", "id": fcid, "call_id": call.id,
                    "name": call.function.name, "arguments": call.function.arguments, "status": status
                ] as [String: Any]
            ])
        }
    }

    private func closeReasoningIfNeeded() {
        let info = lock.withLock { () -> (id: String, idx: Int, text: String)? in
            guard let id = reasoningId, let idx = reasoningIndex, !reasoningClosed else { return nil }
            reasoningClosed = true
            return (id, idx, reasoningText)
        }
        guard let info else { return }
        emitJsonEvent("response.reasoning_summary_text.done", [
            "item_id": info.id, "output_index": info.idx, "summary_index": 0, "text": info.text
        ])
        emitJsonEvent("response.reasoning_summary_part.done", [
            "item_id": info.id, "output_index": info.idx, "summary_index": 0,
            "part": ["type": "summary_text", "text": info.text]
        ])
        emitJsonEvent("response.output_item.done", [
            "output_index": info.idx,
            "item": [
                "type": "reasoning", "id": info.id,
                "summary": [["type": "summary_text", "text": info.text]]
            ] as [String: Any]
        ])
    }

    private func closeMessageIfNeeded(status: String) {
        let info = lock.withLock { () -> (id: String, idx: Int, text: String)? in
            guard let id = messageId, let idx = messageIndex, !messageClosed else { return nil }
            messageClosed = true
            return (id, idx, messageText)
        }
        guard let info else { return }
        let part: [String: Any] = ["type": "output_text", "text": info.text, "annotations": [Any]()]
        emitJsonEvent("response.output_text.done", [
            "item_id": info.id, "output_index": info.idx, "content_index": 0, "text": info.text
        ])
        emitJsonEvent("response.content_part.done", [
            "item_id": info.id, "output_index": info.idx, "content_index": 0, "part": part
        ])
        emitJsonEvent("response.output_item.done", [
            "output_index": info.idx,
            "item": [
                "type": "message", "id": info.id, "status": status,
                "role": "assistant", "content": [part]
            ] as [String: Any]
        ])
    }

    private func emitJsonEvent(_ type: String, _ fields: [String: Any]) {
        let s = lock.withLock { () -> Int in let v = seq; seq += 1; return v }
        var body = fields
        body["type"] = type
        body["sequence_number"] = s
        session.sendSSEJsonObject(event: type, fields: body)
    }
}
