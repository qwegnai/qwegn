// QWServerSchema+OpenAIResponsesMapper.swift — Direct Swift mapping from
// OpenAI /v1/responses input items, tools, and text format into ChatCompletionRequest.

import Foundation
import QWCore
import QWToken

public enum OpenAIResponsesMapper {
    public static func parseInput(_ input: QWJson?, hasPrevious: Bool) throws -> [ChatMessage] {
        guard let input, !input.isUndefined, input != .null else {
            if hasPrevious { return [] }
            throw OpenAIResponseValidationError("input is required")
        }
        switch input {
        case .string(let text):
            return [ChatMessage(role: "user", text: text)]
        case .array(let items):
            return try parseItemArray(items)
        default:
            throw OpenAIResponseValidationError("input must be a string or an array of items")
        }
    }

    private static func parseItemArray(_ items: [QWJson]) throws -> [ChatMessage] {
        var out: [ChatMessage] = []
        var pendingReasoning: String? = nil

        for item in items {
            guard case .object = item else {
                throw OpenAIResponseValidationError("each item in input must be a JSON object")
            }
            let itemType: String
            if case .string(let t) = item.get("type") { itemType = t.lowercased() } else { itemType = "" }
            switch itemType {
            case "function_call":
                let callId: String
                if case .string(let cid) = item.get("call_id"), !cid.isEmpty { callId = cid }
                else if case .string(let id) = item.get("id"), !id.isEmpty { callId = id }
                else { callId = QWToolCallParser.generateCallId() }
                guard case .string(let name) = item.get("name"), !name.isEmpty else {
                    throw OpenAIResponseValidationError("function_call item is missing name")
                }
                let argsStr = QWSchemaMetadataCoercion.canonicalJSONString(item.get("arguments")) ?? "{}"
                let tc = ToolCall(id: callId, type: "function", function: ToolCallFunction(name: name, arguments: argsStr))
                if let last = out.last, last.role == "assistant" {
                    var merged = last.toolCalls ?? []
                    merged.append(tc)
                    out[out.count - 1] = ChatMessage(
                        role: "assistant", content: last.content, toolCalls: merged,
                        reasoningContent: last.reasoningContent ?? pendingReasoning
                    )
                } else {
                    out.append(ChatMessage(
                        role: "assistant", content: nil, toolCalls: [tc], reasoningContent: pendingReasoning
                    ))
                }
                pendingReasoning = nil
            case "function_call_output":
                guard case .string(let callId) = item.get("call_id"), !callId.isEmpty else {
                    throw OpenAIResponseValidationError("function_call_output item is missing call_id")
                }
                out.append(ChatMessage(
                    role: "tool",
                    content: ChatMessageContent(string: extractToolOutputText(item.get("output"))),
                    toolCallId: callId
                ))
            case "reasoning":
                let text = extractReasoningText(from: item)
                if !text.isEmpty {
                    pendingReasoning = pendingReasoning.map { $0 + "\n" + text } ?? text
                }
            default:
                guard case .string(let rawRole) = item.get("role") else {
                    throw OpenAIResponseValidationError("input item must specify role or a supported item type")
                }
                let role = rawRole.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                let content = try parseMessageContent(item.get("content"))
                var msgReasoning: String? = nil
                if case .string(let r) = item.get("reasoning_content") { msgReasoning = r }
                else if case .string(let r) = item.get("reasoning") { msgReasoning = r }
                if role == "assistant", let pending = pendingReasoning {
                    msgReasoning = msgReasoning.map { pending + "\n" + $0 } ?? pending
                    pendingReasoning = nil
                }
                out.append(ChatMessage(role: role, content: content, reasoningContent: msgReasoning))
            }
        }
        return out
    }

    private static func parseMessageContent(_ raw: QWJson) throws -> ChatMessageContent {
        switch raw {
        case .string(let s):
            return ChatMessageContent(string: s)
        case .array(let arr):
            var parts: [ChatMessageContent.Part] = []
            for p in arr {
                guard case .object = p else {
                    throw OpenAIResponseValidationError("message content part must be an object")
                }
                let pType: String
                if case .string(let s) = p.get("type") { pType = s.lowercased() } else { pType = "" }
                switch pType {
                case "input_text", "output_text", "text", "summary_text", "reasoning_text":
                    if case .string(let t) = p.get("text") { parts.append(.text(t)) }
                case "refusal":
                    if case .string(let t) = p.get("refusal") { parts.append(.text(t)) }
                    else if case .string(let t) = p.get("text") { parts.append(.text(t)) }
                case "input_image", "image_url":
                    if case .string(let u) = p.get("image_url"), !u.isEmpty {
                        parts.append(.imageUrl(u))
                    } else if case .string(let u) = p.get("image_url").get("url"), !u.isEmpty {
                        parts.append(.imageUrl(u))
                    }
                case "input_video", "video_url", "video":
                    let nested = p.get("video_url").isUndefined ? p.get("video") : p.get("video_url")
                    let fps = QWSchemaMetadataCoercion.strictDouble(nested.get("fps"))
                        ?? QWSchemaMetadataCoercion.strictDouble(p.get("fps"))
                    let maxFrames = QWSchemaMetadataCoercion.strictInt(nested.get("max_frames"))
                        ?? QWSchemaMetadataCoercion.strictInt(p.get("max_frames"))
                    let maxPixels = QWSchemaMetadataCoercion.strictInt(nested.get("max_pixels"))
                        ?? QWSchemaMetadataCoercion.strictInt(p.get("max_pixels"))
                    if case .string(let u) = nested, !u.isEmpty {
                        parts.append(.videoUrl(QWVideoInputSpec(url: u, fps: fps, maxFrames: maxFrames, maxPixels: maxPixels)))
                    } else if case .string(let u) = nested.get("url"), !u.isEmpty {
                        parts.append(.videoUrl(QWVideoInputSpec(url: u, fps: fps, maxFrames: maxFrames, maxPixels: maxPixels)))
                    } else if case .string(let u) = p.get("url"), !u.isEmpty {
                        parts.append(.videoUrl(QWVideoInputSpec(url: u, fps: fps, maxFrames: maxFrames, maxPixels: maxPixels)))
                    }
                default:
                    if case .string(let t) = p.get("text") { parts.append(.text(t)) }
                }
            }
            return ChatMessageContent(parts: parts)
        case .null, .undefined:
            return ChatMessageContent(string: "")
        default:
            throw OpenAIResponseValidationError("message content must be a string or array")
        }
    }

    private static func extractToolOutputText(_ raw: QWJson) -> String {
        switch raw {
        case .string(let s): return s
        case .array(let arr):
            return arr.compactMap { part -> String? in
                if case .string(let t) = part.get("text") { return t }
                return nil
            }.joined(separator: "\n")
        case .undefined, .null: return ""
        default: return QWSchemaMetadataCoercion.canonicalJSONString(raw) ?? ""
        }
    }

    private static func extractReasoningText(from item: QWJson) -> String {
        var chunks: [String] = []
        for key in ["summary", "content"] {
            if case .array(let arr) = item.get(key) {
                for part in arr {
                    if case .string(let t) = part.get("text"), !t.isEmpty { chunks.append(t) }
                }
            }
        }
        return chunks.joined(separator: "")
    }

    public static func normalizeTools(
        tools: [QWJson]?,
        toolChoice: QWJson?,
        maxToolCalls: Int?
    ) -> ([QWJson]?, [OpenAIResponseResource.FunctionToolInfo], QWJson?, QWJson) {
        var allowedNames: Set<String>? = nil
        var mappedChoice: QWJson? = toolChoice
        if let tc = toolChoice, case .string(let t) = tc.get("type"), t.lowercased() == "allowed_tools" {
            if case .array(let list) = tc.get("tools") {
                var set = Set<String>()
                for entry in list {
                    if case .string(let name) = entry.get("name"), !name.isEmpty { set.insert(name) }
                }
                allowedNames = set
            }
            if case .string(let mode) = tc.get("mode") { mappedChoice = .string(mode) }
        }
        if maxToolCalls == 0 { mappedChoice = .string("none") }

        var chatTools: [QWJson] = []
        var resTools: [OpenAIResponseResource.FunctionToolInfo] = []
        for rawTool in tools ?? [] {
            let fnObj = rawTool.get("function").isUndefined ? rawTool : rawTool.get("function")
            guard case .string(let name) = fnObj.get("name"), !name.isEmpty else { continue }
            if let allowedNames, !allowedNames.contains(name) { continue }
            let desc: String? = { if case .string(let d) = fnObj.get("description") { return d }; return nil }()
            let params: QWJson? = {
                let p = fnObj.get("parameters")
                return p.isUndefined || p == .null ? nil : p
            }()
            let strict: Bool? = {
                if case .bool(let b) = fnObj.get("strict") { return b }
                if case .bool(let b) = rawTool.get("strict") { return b }
                return nil
            }()
            resTools.append(OpenAIResponseResource.FunctionToolInfo(
                name: name, description: desc, parameters: params, strict: strict
            ))
            var fnPairs: [(String, QWJson)] = [("name", .string(name))]
            if let desc { fnPairs.append(("description", .string(desc))) }
            if let params { fnPairs.append(("parameters", params)) }
            chatTools.append(.object([("type", .string("function")), ("function", .object(fnPairs))]))
        }
        return (chatTools.isEmpty ? nil : chatTools, resTools, mappedChoice, toolChoice ?? .string("auto"))
    }

    public static func resourceTextConfig(from text: OpenAIResponseCreateRequest.TextOption?) -> OpenAIResponseResource.TextConfig {
        guard let fmt = text?.format, case .string(let rawType) = fmt.get("type") else {
            return OpenAIResponseResource.TextConfig(verbosity: text?.verbosity)
        }
        switch rawType.lowercased() {
        case "json_object":
            return OpenAIResponseResource.TextConfig(
                format: .object([("type", .string("json_object"))]), verbosity: text?.verbosity
            )
        case "json_schema":
            let name: String = { if case .string(let n) = fmt.get("name") { return n }; return "output" }()
            let desc: QWJson = { if case .string(let d) = fmt.get("description") { return .string(d) }; return .null }()
            let strict: Bool = { if case .bool(let s) = fmt.get("strict") { return s }; return true }()
            return OpenAIResponseResource.TextConfig(
                format: .object([
                    ("type", .string("json_schema")), ("name", .string(name)),
                    ("description", desc), ("schema", .null), ("strict", .bool(strict)),
                ]),
                verbosity: text?.verbosity
            )
        default:
            return OpenAIResponseResource.TextConfig(verbosity: text?.verbosity)
        }
    }

    /// Builds a `ChatCompletionRequest` directly in Swift without intermediate JSON encoding/decoding.
    public static func buildChatRequest(
        from req: OpenAIResponseCreateRequest,
        messages: [ChatMessage],
        tools: [QWJson]?,
        toolChoice: QWJson?,
        headers: [String: String]
    ) throws -> (request: ChatCompletionRequest, thinkingExplicitlyDisabled: Bool) {
        var responseFormat: QWJson? = nil
        if let fmt = req.text?.format, case .string(let fmtType) = fmt.get("type") {
            let lower = fmtType.lowercased()
            if lower == "json_object" {
                responseFormat = .object([("type", .string("json_object"))])
            } else if lower == "json_schema" {
                var jsPairs: [(String, QWJson)] = []
                if case .string(let name) = fmt.get("name") { jsPairs.append(("name", .string(name))) }
                if !fmt.get("schema").isUndefined { jsPairs.append(("schema", fmt.get("schema"))) }
                if case .bool(let strict) = fmt.get("strict") { jsPairs.append(("strict", .bool(strict))) }
                else { jsPairs.append(("strict", .bool(true))) }
                responseFormat = .object([("type", .string("json_schema")), ("json_schema", .object(jsPairs))])
            }
        }

        let effort = req.reasoning?.effort?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let isExplicitOff = effort.map { ["none", "off", "false", "disabled", "disable"].contains($0) } ?? false
        let rawEffort = req.reasoning?.effort.flatMap { $0.isEmpty ? nil : $0 }
        let enableThinking: Bool? = isExplicitOff ? false : req.enableThinking

        let chatReq = ChatCompletionRequest(
            model: req.model,
            messages: messages,
            tools: tools,
            toolChoice: toolChoice,
            parallelToolCalls: req.parallelToolCalls,
            responseFormat: responseFormat,
            maxTokens: req.maxOutputTokens,
            temperature: req.temperature,
            topP: req.topP,
            enableThinking: enableThinking,
            reasoningEffort: rawEffort,
            reasoningBudget: req.reasoningBudget,
            metadata: req.metadata,
            user: req.user ?? req.safetyIdentifier,
            promptCacheKey: req.promptCacheKey,
            cacheSalt: req.cacheSalt,
            cachePrompt: req.cachePrompt,
            store: req.store,
            httpHeaders: headers
        )
        return (chatReq, isExplicitOff || (enableThinking == false))
    }

    public static func capToolCalls(_ calls: [ToolCall]?, parallelToolCalls: Bool?, maxToolCalls: Int?) -> [ToolCall]? {
        QWChatPolicyResolver.capToolCalls(calls, parallelToolCalls: parallelToolCalls, maxToolCalls: maxToolCalls)
    }

    public struct StreamedItemIDs: Sendable {
        public var reasoningId: String? = nil
        public var messageId: String? = nil
        public var functionCallIds: [String] = []

        public init(reasoningId: String? = nil, messageId: String? = nil, functionCallIds: [String] = []) {
            self.reasoningId = reasoningId
            self.messageId = messageId
            self.functionCallIds = functionCallIds
        }
    }

    public static func finalizeResponse(
        template: OpenAIResponseTemplate,
        req: OpenAIResponseCreateRequest,
        chatReq: ChatCompletionRequest,
        result: ChatCompletionExecutionResult,
        baseHistory: [ChatMessage],
        hasActiveTools: Bool,
        allowReasoningTags: Bool,
        thinkingExplicitlyDisabled: Bool,
        allowInk2Markers: Bool = false,
        toolEnvelopes: [QWToolCallEnvelope] = QWToolCallEnvelope.knownEnvelopes
    ) -> (OpenAIResponseResource, [ChatMessage]) {
        let isConstrainedJson = (chatReq.responseFormatType == "json_schema" || chatReq.responseFormatType == "json_object") && !hasActiveTools
        let effectiveStartsInThinking = result.startsInThinking && !thinkingExplicitlyDisabled
        let effectiveAllowTags = allowReasoningTags || ((!isConstrainedJson || effectiveStartsInThinking) && !thinkingExplicitlyDisabled)
        let effectiveAllowInk2 = allowInk2Markers && (!isConstrainedJson || effectiveStartsInThinking)
        let parsed = QWToolCallParser.parse(
            text: result.content, tools: chatReq.tools,
            startsInThinking: effectiveStartsInThinking,
            truncated: result.truncated, allowToolCalls: hasActiveTools,
            allowReasoningTags: effectiveAllowTags,
            allowInk2Markers: effectiveAllowInk2,
            toolEnvelopes: toolEnvelopes
        )
        let cappedCalls = capToolCalls(
            parsed.toolCalls, parallelToolCalls: req.parallelToolCalls, maxToolCalls: req.maxToolCalls
        )
        return buildFinalizedResource(
            template: template, parsed: parsed, effectiveCalls: cappedCalls, result: result,
            baseHistory: baseHistory, thinkingExplicitlyDisabled: thinkingExplicitlyDisabled
        )
    }

    public static func buildFinalizedResource(
        template: OpenAIResponseTemplate,
        parsed: QWToolCallParser.ParseResult,
        effectiveCalls: [ToolCall]?,
        result: ChatCompletionExecutionResult,
        baseHistory: [ChatMessage],
        thinkingExplicitlyDisabled: Bool,
        streamedItemIDs: StreamedItemIDs? = nil
    ) -> (OpenAIResponseResource, [ChatMessage]) {
        let itemStatus = result.truncated ? "incomplete" : "completed"
        var outputItems: [QWJson] = []
        let reasoningText = thinkingExplicitlyDisabled ? nil : parsed.reasoningContent
        if let rText = reasoningText, !rText.isEmpty {
            let rid = streamedItemIDs?.reasoningId ?? ("rs_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(16))
            outputItems.append(.object([
                ("type", .string("reasoning")),
                ("id", .string(rid)),
                ("summary", .array([.object([("type", .string("summary_text")), ("text", .string(rText))])]))
            ]))
        }
        let msgId = streamedItemIDs?.messageId ?? ("msg_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(16))
        if let text = parsed.content, !text.isEmpty {
            outputItems.append(makeMessageOutputItem(id: msgId, text: text, status: itemStatus))
        } else if effectiveCalls?.isEmpty ?? true {
            outputItems.append(makeMessageOutputItem(id: msgId, text: parsed.content ?? "", status: itemStatus))
        }
        for (idx, call) in (effectiveCalls ?? []).enumerated() {
            let fcid = (streamedItemIDs?.functionCallIds.indices.contains(idx) == true
                ? streamedItemIDs?.functionCallIds[idx] : nil)
                ?? ("fc_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(16))
            outputItems.append(.object([
                ("type", .string("function_call")),
                ("id", .string(fcid)),
                ("call_id", .string(call.id)),
                ("name", .string(call.function.name)),
                ("arguments", .string(call.function.arguments)),
                ("status", .string(itemStatus))
            ]))
        }

        let rToks = thinkingExplicitlyDisabled ? 0 : (result.reasoningTokens ?? 0)
        let usage = OpenAIResponseResource.UsageInfo(
            inputTokens: result.promptTokens, outputTokens: result.completionTokens,
            cachedTokens: result.cachedTokens, reasoningTokens: rToks
        )
        let incomplete = result.truncated
            ? OpenAIResponseResource.IncompleteDetails(reason: "max_output_tokens")
            : nil
        let resource = template.makeResource(
            status: result.truncated ? "incomplete" : "completed",
            completedAt: Int(Date().timeIntervalSince1970),
            output: outputItems,
            usage: usage,
            incompleteDetails: incomplete
        )

        var nextHistory = baseHistory
        nextHistory.append(ChatMessage(
            role: "assistant",
            content: parsed.content.map { ChatMessageContent(string: $0) },
            toolCalls: effectiveCalls,
            reasoningContent: reasoningText
        ))
        return (resource, nextHistory)
    }

    private static func makeMessageOutputItem(id: String, text: String, status: String) -> QWJson {
        .object([
            ("type", .string("message")),
            ("id", .string(id)),
            ("status", .string(status)),
            ("role", .string("assistant")),
            ("content", .array([.object([
                ("type", .string("output_text")), ("text", .string(text)), ("annotations", .array([]))
            ])]))
        ])
    }
}
