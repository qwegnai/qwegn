import Foundation
import QWCore
import QWToken

public struct ChatCompletionRequest: Codable, Sendable {
    public struct ChatTemplateKwargs: Codable, Sendable {
        public let enableThinking: Bool?
        public let reasoning: Bool?
        public let reasoningEffort: String?

        enum CodingKeys: String, CodingKey {
            case enableThinking = "enable_thinking"
            case reasoning
            case reasoningEffort = "reasoning_effort"
        }

        public init(enableThinking: Bool? = nil, reasoning: Bool? = nil, reasoningEffort: String? = nil) {
            self.enableThinking = enableThinking
            self.reasoning = reasoning
            self.reasoningEffort = reasoningEffort
        }
    }

    public struct ReasoningConfig: Codable, Sendable {
        public let enabled: Bool?
        public let effort: String?

        public init(enabled: Bool? = nil, effort: String? = nil) {
            self.enabled = enabled
            self.effort = effort
        }
    }

    public let model: String?
    public let messages: [ChatMessage]
    public let tools: [QWJson]?
    public let toolChoice: QWJson?
    public let parallelToolCalls: Bool?
    public let responseFormat: QWJson?
    public let logprobs: Bool?
    public let topLogprobs: Int?
    public let n: Int?
    public let continueFinalMessage: Bool?
    public let addGenerationPrompt: Bool?
    public let stream: Bool?
    public let maxTokens: Int?
    public let maxCompletionTokens: Int?
    public let minTokens: Int?
    public let ignoreEos: Bool?
    public let temperature: Double?
    public let topP: Double?
    public let topK: Int?
    public let repetitionPenalty: Double?
    public let seed: Int?
    public let stop: [String]?
    public let enableThinking: Bool?
    public let reasoningEffort: String?
    public let reasoning: ReasoningConfig?
    public let reasoningBudget: Int?
    public let chatTemplateKwargs: ChatTemplateKwargs?
    public let metadata: [String: String]?
    public let user: String?
    public let sessionId: String?
    public let promptCacheKey: String?
    public let cacheSalt: String?
    public let cachePrompt: Bool?
    public let store: Bool?
    public var httpHeaders: [String: String]? = nil

    enum CodingKeys: String, CodingKey {
        case model, messages, tools, logprobs, n, stream, temperature, seed, stop, reasoning, metadata, user, store
        case toolChoice = "tool_choice"
        case parallelToolCalls = "parallel_tool_calls"
        case responseFormat = "response_format"
        case topLogprobs = "top_logprobs"
        case continueFinalMessage = "continue_final_message"
        case addGenerationPrompt = "add_generation_prompt"
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case minTokens = "min_tokens"
        case ignoreEos = "ignore_eos"
        case topP = "top_p"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case enableThinking = "enable_thinking"
        case reasoningEffort = "reasoning_effort"
        case reasoningBudget = "reasoning_budget"
        case chatTemplateKwargs = "chat_template_kwargs"
        case sessionId = "session_id"
        case promptCacheKey = "prompt_cache_key"
        case cacheSalt = "cache_salt"
        case cachePrompt = "cache_prompt"
    }

    public init(
        model: String? = nil,
        messages: [ChatMessage],
        tools: [QWJson]? = nil,
        toolChoice: QWJson? = nil,
        parallelToolCalls: Bool? = nil,
        responseFormat: QWJson? = nil,
        logprobs: Bool? = nil,
        topLogprobs: Int? = nil,
        n: Int? = nil,
        continueFinalMessage: Bool? = nil,
        addGenerationPrompt: Bool? = nil,
        stream: Bool? = nil,
        maxTokens: Int? = nil,
        maxCompletionTokens: Int? = nil,
        minTokens: Int? = nil,
        ignoreEos: Bool? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        repetitionPenalty: Double? = nil,
        seed: Int? = nil,
        stop: [String]? = nil,
        enableThinking: Bool? = nil,
        reasoningEffort: String? = nil,
        reasoning: ReasoningConfig? = nil,
        reasoningBudget: Int? = nil,
        chatTemplateKwargs: ChatTemplateKwargs? = nil,
        metadata: [String: String]? = nil,
        user: String? = nil,
        sessionId: String? = nil,
        promptCacheKey: String? = nil,
        cacheSalt: String? = nil,
        cachePrompt: Bool? = nil,
        store: Bool? = nil,
        httpHeaders: [String: String]? = nil
    ) {
        self.model = model
        self.messages = messages
        self.tools = tools
        self.toolChoice = toolChoice
        self.parallelToolCalls = parallelToolCalls
        self.responseFormat = responseFormat
        self.logprobs = logprobs
        self.topLogprobs = topLogprobs
        self.n = n
        self.continueFinalMessage = continueFinalMessage
        self.addGenerationPrompt = addGenerationPrompt
        self.stream = stream
        self.maxTokens = maxTokens
        self.maxCompletionTokens = maxCompletionTokens
        self.minTokens = minTokens
        self.ignoreEos = ignoreEos
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.stop = stop
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
        self.reasoning = reasoning
        self.reasoningBudget = reasoningBudget
        self.chatTemplateKwargs = chatTemplateKwargs
        self.metadata = metadata
        self.user = user
        self.sessionId = sessionId
        self.promptCacheKey = promptCacheKey
        self.cacheSalt = cacheSalt
        self.cachePrompt = cachePrompt
        self.store = store
        self.httpHeaders = httpHeaders
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.model = try container.decodeIfPresent(String.self, forKey: .model)
        self.messages = try container.decode([ChatMessage].self, forKey: .messages)
        self.tools = try container.decodeIfPresent([QWJson].self, forKey: .tools)
        self.toolChoice = try container.decodeIfPresent(QWJson.self, forKey: .toolChoice)
        self.parallelToolCalls = try container.decodeIfPresent(Bool.self, forKey: .parallelToolCalls)
        self.responseFormat = try container.decodeIfPresent(QWJson.self, forKey: .responseFormat)
        self.logprobs = try container.decodeIfPresent(Bool.self, forKey: .logprobs)
        self.topLogprobs = try container.decodeIfPresent(Int.self, forKey: .topLogprobs)
        self.n = try container.decodeIfPresent(Int.self, forKey: .n)
        self.continueFinalMessage = try container.decodeIfPresent(Bool.self, forKey: .continueFinalMessage)
        self.addGenerationPrompt = try container.decodeIfPresent(Bool.self, forKey: .addGenerationPrompt)
        self.stream = try container.decodeIfPresent(Bool.self, forKey: .stream)
        self.maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
        self.maxCompletionTokens = try container.decodeIfPresent(Int.self, forKey: .maxCompletionTokens)
        self.minTokens = try container.decodeIfPresent(Int.self, forKey: .minTokens)
        self.ignoreEos = try container.decodeIfPresent(Bool.self, forKey: .ignoreEos)
        self.temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        self.topP = try container.decodeIfPresent(Double.self, forKey: .topP)
        self.topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        self.repetitionPenalty = try container.decodeIfPresent(Double.self, forKey: .repetitionPenalty)
        self.seed = try container.decodeIfPresent(Int.self, forKey: .seed)
        self.enableThinking = try container.decodeIfPresent(Bool.self, forKey: .enableThinking)
        self.reasoningEffort = try container.decodeIfPresent(String.self, forKey: .reasoningEffort)
        self.reasoning = try container.decodeIfPresent(ReasoningConfig.self, forKey: .reasoning)
        self.reasoningBudget = try container.decodeIfPresent(Int.self, forKey: .reasoningBudget)
        self.chatTemplateKwargs = try container.decodeIfPresent(ChatTemplateKwargs.self, forKey: .chatTemplateKwargs)
        self.user = try container.decodeIfPresent(String.self, forKey: .user)
        self.sessionId = try container.decodeIfPresent(String.self, forKey: .sessionId)
        self.promptCacheKey = try container.decodeIfPresent(String.self, forKey: .promptCacheKey)
        self.cacheSalt = try container.decodeIfPresent(String.self, forKey: .cacheSalt)
        self.cachePrompt = try container.decodeIfPresent(Bool.self, forKey: .cachePrompt)
        self.store = try container.decodeIfPresent(Bool.self, forKey: .store)
        self.metadata = try QWSchemaMetadataCoercion.decodeOptionalMetadata(from: container, forKey: .metadata)
        self.stop = try QWSchemaMetadataCoercion.decodeOptionalStop(from: container, forKey: .stop)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(model, forKey: .model)
        try container.encode(messages, forKey: .messages)
        try container.encodeIfPresent(tools, forKey: .tools)
        try container.encodeIfPresent(toolChoice, forKey: .toolChoice)
        try container.encodeIfPresent(parallelToolCalls, forKey: .parallelToolCalls)
        try container.encodeIfPresent(responseFormat, forKey: .responseFormat)
        try container.encodeIfPresent(logprobs, forKey: .logprobs)
        try container.encodeIfPresent(topLogprobs, forKey: .topLogprobs)
        try container.encodeIfPresent(n, forKey: .n)
        try container.encodeIfPresent(continueFinalMessage, forKey: .continueFinalMessage)
        try container.encodeIfPresent(addGenerationPrompt, forKey: .addGenerationPrompt)
        try container.encodeIfPresent(stream, forKey: .stream)
        try container.encodeIfPresent(maxTokens, forKey: .maxTokens)
        try container.encodeIfPresent(maxCompletionTokens, forKey: .maxCompletionTokens)
        try container.encodeIfPresent(minTokens, forKey: .minTokens)
        try container.encodeIfPresent(ignoreEos, forKey: .ignoreEos)
        try container.encodeIfPresent(temperature, forKey: .temperature)
        try container.encodeIfPresent(topP, forKey: .topP)
        try container.encodeIfPresent(topK, forKey: .topK)
        try container.encodeIfPresent(repetitionPenalty, forKey: .repetitionPenalty)
        try container.encodeIfPresent(seed, forKey: .seed)
        try container.encodeIfPresent(stop, forKey: .stop)
        try container.encodeIfPresent(enableThinking, forKey: .enableThinking)
        try container.encodeIfPresent(reasoningEffort, forKey: .reasoningEffort)
        try container.encodeIfPresent(reasoning, forKey: .reasoning)
        try container.encodeIfPresent(reasoningBudget, forKey: .reasoningBudget)
        try container.encodeIfPresent(chatTemplateKwargs, forKey: .chatTemplateKwargs)
        try container.encodeIfPresent(metadata, forKey: .metadata)
        try container.encodeIfPresent(user, forKey: .user)
        try container.encodeIfPresent(sessionId, forKey: .sessionId)
        try container.encodeIfPresent(promptCacheKey, forKey: .promptCacheKey)
        try container.encodeIfPresent(cacheSalt, forKey: .cacheSalt)
        try container.encodeIfPresent(cachePrompt, forKey: .cachePrompt)
        try container.encodeIfPresent(store, forKey: .store)
    }

    public func validateMetadata() throws {
        guard let metadata else { return }
        if metadata.count > 16 {
            throw QWMetadataValidationError("metadata may contain at most 16 entries (got \(metadata.count))")
        }
        for (key, value) in metadata {
            if key.isEmpty || key.count > 64 {
                throw QWMetadataValidationError("metadata key '\(key)' must be 1...64 characters")
            }
            if value.count > 512 {
                throw QWMetadataValidationError("metadata['\(key)'] exceeds 512 characters")
            }
        }
    }

    public func metadataValue(for key: String) -> String? {
        guard let metadata else { return nil }
        if let exact = metadata[key] {
            let trimmed = exact.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
        for (k, v) in metadata where k.caseInsensitiveCompare(key) == .orderedSame {
            let trimmed = v.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
        return nil
    }

    private func headerValue(for name: String) -> String? {
        guard let httpHeaders else { return nil }
        if let exact = httpHeaders[name] {
            let trimmed = exact.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
        for (k, v) in httpHeaders where k.caseInsensitiveCompare(name) == .orderedSame {
            let trimmed = v.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
        return nil
    }

    private static func parseMetadataBool(_ raw: String?) -> Bool? {
        guard let lowered = raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              !lowered.isEmpty else { return nil }
        switch lowered {
        case "true", "1", "yes", "on": return true
        case "false", "0", "no", "off": return false
        default: return nil
        }
    }

    private static func sanitizeIdentifier(_ raw: String?, maxLength: Int = 64) -> String? {
        guard let trimmed = raw?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty else { return nil }
        let filtered = String(trimmed.unicodeScalars.filter { s in
            (s.value >= 0x30 && s.value <= 0x39)
                || (s.value >= 0x41 && s.value <= 0x5A)
                || (s.value >= 0x61 && s.value <= 0x7A)
                || s == "." || s == "_" || s == ":" || s == "-"
        }.prefix(maxLength))
        return filtered.isEmpty ? nil : filtered
    }

    public var effectiveRequestId: String? {
        Self.sanitizeIdentifier(headerValue(for: "x-request-id") ?? metadataValue(for: "request_id") ?? metadataValue(for: "x_request_id"))
    }

    public var effectiveSessionId: String? {
        Self.sanitizeIdentifier(sessionId ?? headerValue(for: "x-session-id") ?? metadataValue(for: "session_id") ?? metadataValue(for: "conversation_id"))
    }

    public var effectiveUser: String? {
        Self.sanitizeIdentifier(user ?? metadataValue(for: "user_id") ?? metadataValue(for: "user"))
    }

    public var effectivePresetId: String? {
        Self.sanitizeIdentifier(metadataValue(for: "preset_id") ?? metadataValue(for: "preset"))
    }

    public var effectiveCachePrompt: Bool {
        cachePrompt ?? Self.parseMetadataBool(metadataValue(for: "cache_prompt") ?? metadataValue(for: "use_prefix_cache")) ?? true
    }

    public var effectiveCacheSalt: String? {
        Self.sanitizeIdentifier(cacheSalt ?? promptCacheKey ?? metadataValue(for: "cache_salt") ?? metadataValue(for: "prompt_cache_key"))
    }

    public var effectiveUseMemoryPack: Bool {
        Self.parseMetadataBool(metadataValue(for: "use_memory_pack")) ?? true
    }

    public var effectiveWorkspacePath: String? {
        metadataValue(for: "workspace_path") ?? metadataValue(for: "repo_path") ?? metadataValue(for: "cwd")
    }

    public var effectiveStore: Bool {
        store ?? Self.parseMetadataBool(metadataValue(for: "store")) ?? true
    }

    public var effectiveExactSpeculation: Bool {
        Self.parseMetadataBool(metadataValue(for: "exact_speculation") ?? metadataValue(for: "lossless_speculation")) ?? false
    }

    public var rawReasoningEffort: String? {
        reasoningEffort ?? chatTemplateKwargs?.reasoningEffort ?? reasoning?.effort
    }

    public static func normalizeReasoningEffort(_ raw: String?, fallback: String = "xhigh") -> String {
        QWChatPolicyResolver.normalizeReasoningEffort(raw, fallback: fallback)
    }

    public func effectiveReasoningEffort(defaultValue: String) -> String {
        QWChatPolicyResolver.normalizeReasoningEffort(rawReasoningEffort, fallback: defaultValue)
    }

    public var hasExplicitThinkingSetting: Bool {
        enableThinking != nil || reasoningEffort != nil || reasoningBudget != nil
            || chatTemplateKwargs?.enableThinking != nil || chatTemplateKwargs?.reasoning != nil
            || chatTemplateKwargs?.reasoningEffort != nil
            || reasoning?.enabled != nil || reasoning?.effort != nil
    }

    public func effectiveEnableThinking(defaultValue: Bool) -> Bool {
        QWChatPolicyResolver.effectiveEnableThinking(for: self, defaultValue: defaultValue)
    }

    /// True when the final message in `messages` is an assistant text message to be continued in-place.
    public var isAssistantPrefill: Bool {
        guard addGenerationPrompt != true, continueFinalMessage != false,
              let last = messages.last else { return false }
        let r = last.role.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return r == "assistant" && (last.toolCalls?.isEmpty ?? true) && !last.extractTextContent().isEmpty
    }

    /// Returns `"json_object"` or `"json_schema"` when `responseFormat` specifies one.
    public var responseFormatType: String? {
        guard let rf = responseFormat else { return nil }
        if case .string(let s) = rf.get("type") {
            return s.lowercased()
        }
        return nil
    }

    /// Returns whether `tool_choice` explicitly suppresses tool calls (`"none"` or `{"type": "none"}`).
    public var isToolChoiceNone: Bool {
        guard let tc = toolChoice else { return false }
        if case .string(let s) = tc {
            return s.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "none"
        }
        if case .string(let t) = tc.get("type") {
            return t.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "none"
        }
        return false
    }

    /// Returns the forced tool name when `tool_choice` names a specific tool, or `""` when `"required"`/`"any"`.
    public var forcedToolChoiceName: String? {
        guard let tc = toolChoice else { return nil }
        if case .string(let s) = tc {
            let low = s.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if low == "required" || low == "any" { return "" }
            return nil
        }
        if case .string(let t) = tc.get("type") {
            let low = t.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if low == "any" || low == "required" { return "" }
            if low == "function", case .string(let name) = tc.get("function").get("name"), !name.isEmpty {
                return name
            }
            if low == "tool", case .string(let name) = tc.get("name"), !name.isEmpty {
                return name
            }
        }
        if case .string(let name) = tc.get("name"), !name.isEmpty {
            return name
        }
        return nil
    }

    public var effectiveMaxTokens: Int {
        maxTokens ?? maxCompletionTokens ?? 512
    }

    public var isStreaming: Bool {
        stream ?? false
    }

    /// Normalizes OpenAI/agent message roles (`developer` -> `system`, `function` -> `tool`)
    /// and ensures `system` only appears at index 0 so strict Jinja chat templates never throw.
    public func toQWJsonMessages() -> [QWJson] {
        var normalizedMessages: [(role: String, msg: ChatMessage)] = []
        for msg in messages {
            let lowered = msg.role.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let canonicalRole: String
            switch lowered {
            case "system", "developer": canonicalRole = "system"
            case "user": canonicalRole = "user"
            case "assistant", "model": canonicalRole = "assistant"
            case "tool", "function": canonicalRole = "tool"
            default: canonicalRole = "user"
            }
            normalizedMessages.append((canonicalRole, msg))
        }

        // Merge consecutive leading system/developer messages into a single index-0 system message.
        var mergedLeadingSystemText: [String] = []
        var firstNonSystemIndex = 0
        while firstNonSystemIndex < normalizedMessages.count && normalizedMessages[firstNonSystemIndex].role == "system" {
            if let text = normalizedMessages[firstNonSystemIndex].msg.content?.textValue, !text.isEmpty {
                mergedLeadingSystemText.append(text)
            }
            firstNonSystemIndex += 1
        }

        var result: [QWJson] = []
        let hasNonSystemMessage = firstNonSystemIndex < normalizedMessages.count
        if firstNonSystemIndex > 0 {
            let leadingRole = hasNonSystemMessage ? "system" : "user"
            result.append(.object([
                ("role", .string(leadingRole)),
                ("content", .string(mergedLeadingSystemText.joined(separator: "\n\n")))
            ]))
        }

        for idx in firstNonSystemIndex..<normalizedMessages.count {
            let (rawRole, msg) = normalizedMessages[idx]
            let roleStr = (rawRole == "system") ? "user" : rawRole
            var pairs: [(String, QWJson)] = [("role", .string(roleStr))]
            if let text = msg.content?.templateTextValue {
                pairs.append(("content", .string(text)))
            } else {
                pairs.append(("content", .null))
            }
            if let reasoning = msg.reasoningContent {
                pairs.append(("reasoning_content", .string(reasoning)))
            }
            if let toolCalls = msg.toolCalls, !toolCalls.isEmpty {
                var callArray: [QWJson] = []
                for call in toolCalls {
                    var parsedArgs: QWJson = .object([])
                    if let json = try? QWJson.parse(call.function.arguments), case .object = json {
                        parsedArgs = json
                    } else {
                        parsedArgs = .string(call.function.arguments)
                    }
                    let funcObj: QWJson = .object([
                        ("name", .string(call.function.name)),
                        ("arguments", parsedArgs)
                    ])
                    callArray.append(.object([
                        ("id", .string(call.id)),
                        ("type", .string(call.type)),
                        ("function", funcObj)
                    ]))
                }
                pairs.append(("tool_calls", .array(callArray)))
            }
            if let toolCallId = msg.toolCallId {
                pairs.append(("tool_call_id", .string(toolCallId)))
            }
            if let name = msg.name {
                pairs.append(("name", .string(name)))
            }
            result.append(.object(pairs))
        }
        return result
    }

    /// Extracts all image references present in user messages.
    public func extractImageUrls() -> [String] {
        var list: [String] = []
        for msg in messages {
            if let urls = msg.content?.imageUrls {
                list.append(contentsOf: urls)
            }
        }
        return list
    }

    /// Extracts all video input specifications present in user messages.
    public func extractVideoInputs() -> [QWVideoInputSpec] {
        var list: [QWVideoInputSpec] = []
        for msg in messages {
            if let vids = msg.content?.videoInputs {
                list.append(contentsOf: vids)
            }
        }
        return list
    }

    /// Returns a copy of this request with an explicit resolved reasoning effort tier (`off|low|xhigh`).
    public func withResolvedReasoningEffort(_ resolved: String) -> ChatCompletionRequest {
        ChatCompletionRequest(
            model: model,
            messages: messages,
            tools: tools,
            toolChoice: toolChoice,
            parallelToolCalls: parallelToolCalls,
            responseFormat: responseFormat,
            logprobs: logprobs,
            topLogprobs: topLogprobs,
            n: n,
            continueFinalMessage: continueFinalMessage,
            addGenerationPrompt: addGenerationPrompt,
            stream: stream,
            maxTokens: maxTokens,
            maxCompletionTokens: maxCompletionTokens,
            minTokens: minTokens,
            ignoreEos: ignoreEos,
            temperature: temperature,
            topP: topP,
            topK: topK,
            repetitionPenalty: repetitionPenalty,
            seed: seed,
            stop: stop,
            enableThinking: resolved == "off" ? false : true,
            reasoningEffort: resolved,
            reasoning: nil,
            reasoningBudget: reasoningBudget,
            chatTemplateKwargs: nil,
            metadata: metadata,
            user: user,
            sessionId: sessionId,
            promptCacheKey: promptCacheKey,
            cacheSalt: cacheSalt,
            cachePrompt: cachePrompt,
            store: store,
            httpHeaders: httpHeaders
        )
    }
}
