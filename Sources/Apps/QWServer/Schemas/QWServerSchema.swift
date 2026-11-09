import Foundation
import QWCore
import QWToken

public struct ChatMessageContent: Codable, Sendable {
    public enum Part: Codable, Sendable {
        case text(String)
        case imageUrl(String)

        enum CodingKeys: String, CodingKey {
            case type
            case text
            case imageUrl = "image_url"
        }

        enum ImageUrlKeys: String, CodingKey {
            case url
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let type = try container.decode(String.self, forKey: .type)
            if type == "text" {
                let text = try container.decode(String.self, forKey: .text)
                self = .text(text)
            } else if type == "image_url" {
                let imgContainer = try container.nestedContainer(keyedBy: ImageUrlKeys.self, forKey: .imageUrl)
                let url = try imgContainer.decode(String.self, forKey: .url)
                self = .imageUrl(url)
            } else {
                self = .text("")
            }
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .text(let t):
                try container.encode("text", forKey: .type)
                try container.encode(t, forKey: .text)
            case .imageUrl(let u):
                try container.encode("image_url", forKey: .type)
                var imgContainer = container.nestedContainer(keyedBy: ImageUrlKeys.self, forKey: .imageUrl)
                try imgContainer.encode(u, forKey: .url)
            }
        }
    }

    public let rawText: String?
    public let parts: [Part]?

    public init(string: String) {
        self.rawText = string
        self.parts = nil
    }

    public init(parts: [Part]) {
        self.rawText = nil
        self.parts = parts
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            self.rawText = str
            self.parts = nil
        } else if let p = try? container.decode([Part].self) {
            self.rawText = nil
            self.parts = p
        } else {
            self.rawText = ""
            self.parts = nil
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if let rawText {
            try container.encode(rawText)
        } else if let parts {
            try container.encode(parts)
        } else {
            try container.encode("")
        }
    }

    public var textValue: String {
        if let rawText { return rawText }
        if let parts {
            var res = ""
            for p in parts {
                if case .text(let t) = p {
                    if !res.isEmpty { res += "\n" }
                    res += t
                }
            }
            return res
        }
        return ""
    }

    /// Content string formatted for chat-template rendering, inserting `<|vision_start|><|image_pad|><|vision_end|>`
    /// for each `image_url` part when the text does not already include explicit `<|image_pad|>` markers.
    public var templateTextValue: String {
        if let rawText { return rawText }
        guard let parts else { return "" }
        let plainText = textValue
        if plainText.contains("<|image_pad|>") { return plainText }
        var res = ""
        for p in parts {
            switch p {
            case .text(let t):
                if !res.isEmpty && !res.hasSuffix(">") { res += "\n" }
                res += t
            case .imageUrl:
                res += "<|vision_start|><|image_pad|><|vision_end|>"
            }
        }
        return res
    }

    public var imageUrls: [String] {
        guard let parts else { return [] }
        var urls: [String] = []
        for p in parts {
            if case .imageUrl(let u) = p {
                urls.append(u)
            }
        }
        return urls
    }
}

public struct ChatMessage: Codable, Sendable {
    public let role: String
    public let content: ChatMessageContent?
    public let toolCalls: [ToolCall]?
    public let toolCallId: String?
    public let name: String?
    public let reasoningContent: String?

    enum CodingKeys: String, CodingKey {
        case role
        case content
        case toolCalls = "tool_calls"
        case toolCallId = "tool_call_id"
        case name
        case reasoningContent = "reasoning_content"
    }

    public init(
        role: String,
        content: ChatMessageContent? = nil,
        toolCalls: [ToolCall]? = nil,
        toolCallId: String? = nil,
        name: String? = nil,
        reasoningContent: String? = nil
    ) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolCallId = toolCallId
        self.name = name
        self.reasoningContent = reasoningContent
    }

    public init(role: String, text: String) {
        self.role = role
        self.content = ChatMessageContent(string: text)
        self.toolCalls = nil
        self.toolCallId = nil
        self.name = nil
        self.reasoningContent = nil
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.role = try container.decode(String.self, forKey: .role)
        self.content = try container.decodeIfPresent(ChatMessageContent.self, forKey: .content)
        self.toolCalls = try container.decodeIfPresent([ToolCall].self, forKey: .toolCalls)
        self.toolCallId = try container.decodeIfPresent(String.self, forKey: .toolCallId)
        self.name = try container.decodeIfPresent(String.self, forKey: .name)
        self.reasoningContent = try container.decodeIfPresent(String.self, forKey: .reasoningContent)
    }

    public func extractTextContent() -> String {
        content?.textValue ?? ""
    }
}

public struct ChatLogprobTokenEntry: Codable, Sendable {
    public struct TopLogprob: Codable, Sendable {
        public let token: String
        public let logprob: Float
        public let bytes: [Int]?

        public init(token: String, logprob: Float, bytes: [Int]? = nil) {
            self.token = token
            self.logprob = logprob
            self.bytes = bytes
        }
    }

    public let token: String
    public let logprob: Float
    public let bytes: [Int]?
    public let topLogprobs: [TopLogprob]

    enum CodingKeys: String, CodingKey {
        case token
        case logprob
        case bytes
        case topLogprobs = "top_logprobs"
    }

    public init(token: String, logprob: Float, bytes: [Int]? = nil, topLogprobs: [TopLogprob] = []) {
        self.token = token
        self.logprob = logprob
        self.bytes = bytes
        self.topLogprobs = topLogprobs
    }
}

public struct ChoiceLogprobs: Codable, Sendable {
    public let content: [ChatLogprobTokenEntry]

    public init(content: [ChatLogprobTokenEntry]) {
        self.content = content
    }
}

public struct ChatCompletionRequest: Codable, Sendable {
    public struct ChatTemplateKwargs: Codable, Sendable {
        public let enableThinking: Bool?
        public let reasoningEffort: String?

        enum CodingKeys: String, CodingKey {
            case enableThinking = "enable_thinking"
            case reasoningEffort = "reasoning_effort"
        }
    }

    public struct ReasoningConfig: Codable, Sendable {
        public let enabled: Bool?
        public let effort: String?
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
    public let chatTemplateKwargs: ChatTemplateKwargs?

    enum CodingKeys: String, CodingKey {
        case model
        case messages
        case tools
        case toolChoice = "tool_choice"
        case parallelToolCalls = "parallel_tool_calls"
        case responseFormat = "response_format"
        case logprobs
        case topLogprobs = "top_logprobs"
        case n
        case continueFinalMessage = "continue_final_message"
        case addGenerationPrompt = "add_generation_prompt"
        case stream
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case minTokens = "min_tokens"
        case ignoreEos = "ignore_eos"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case seed
        case stop
        case enableThinking = "enable_thinking"
        case reasoningEffort = "reasoning_effort"
        case reasoning
        case chatTemplateKwargs = "chat_template_kwargs"
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
        self.chatTemplateKwargs = try container.decodeIfPresent(ChatTemplateKwargs.self, forKey: .chatTemplateKwargs)

        if let singleStop = try? container.decode(String.self, forKey: .stop) {
            self.stop = [singleStop]
        } else {
            self.stop = try container.decodeIfPresent([String].self, forKey: .stop)
        }
    }

    public var rawReasoningEffort: String? {
        reasoningEffort ?? chatTemplateKwargs?.reasoningEffort ?? reasoning?.effort
    }

    public static func normalizeReasoningEffort(_ raw: String?, fallback: String = "xhigh") -> String {
        guard let trimmed = raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              !trimmed.isEmpty else {
            if fallback.lowercased() == raw?.lowercased() {
                return "xhigh"
            }
            return normalizeReasoningEffort(fallback, fallback: "xhigh")
        }
        switch trimmed {
        case "xhigh", "high", "max", "maximum", "deep":
            return "xhigh"
        case "medium", "mid", "med", "normal", "auto", "default":
            return "medium"
        case "low", "min", "minimal", "light", "fast", "quick", "none", "off", "false", "disabled", "disable":
            return "low"
        default:
            if fallback.lowercased() == trimmed {
                return "xhigh"
            }
            return normalizeReasoningEffort(fallback, fallback: "xhigh")
        }
    }

    public func effectiveReasoningEffort(defaultValue: String) -> String {
        Self.normalizeReasoningEffort(rawReasoningEffort, fallback: defaultValue)
    }

    public func effectiveEnableThinking(defaultValue: Bool) -> Bool {
        if isAssistantPrefill {
            return false
        }
        // Structured output (`json_object` / `json_schema`) disables thinking unless explicitly enabled
        // so the response starts directly with valid JSON without consuming max_tokens on `<think>`.
        if let fmtType = responseFormatType, (fmtType == "json_object" || fmtType == "json_schema") {
            return enableThinking ?? chatTemplateKwargs?.enableThinking ?? reasoning?.enabled ?? false
        }
        if let explicit = enableThinking ?? chatTemplateKwargs?.enableThinking ?? reasoning?.enabled {
            return explicit
        }
        if let effort = rawReasoningEffort?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(), !effort.isEmpty {
            if effort == "none" || effort == "off" || effort == "false" || effort == "disabled" || effort == "disable" {
                return false
            }
            return true
        }
        return defaultValue
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
            case "system", "developer":
                canonicalRole = "system"
            case "user":
                canonicalRole = "user"
            case "assistant", "model":
                canonicalRole = "assistant"
            case "tool", "function":
                canonicalRole = "tool"
            default:
                canonicalRole = "user"
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
            var pairs: [(String, QWJson)] = [
                ("role", .string(roleStr))
            ]
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
                    let callObj: QWJson = .object([
                        ("id", .string(call.id)),
                        ("type", .string(call.type)),
                        ("function", funcObj)
                    ])
                    callArray.append(callObj)
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
}

public struct ChatCompletionResponse: Codable, Sendable {
    public struct Choice: Codable, Sendable {
        public struct Message: Codable, Sendable {
            public let role: String
            public let content: String?
            public let toolCalls: [ToolCall]?
            public let reasoningContent: String?

            enum CodingKeys: String, CodingKey {
                case role
                case content
                case toolCalls = "tool_calls"
                case reasoningContent = "reasoning_content"
            }

            public init(role: String, content: String?, toolCalls: [ToolCall]? = nil, reasoningContent: String? = nil) {
                self.role = role
                self.content = content
                self.toolCalls = toolCalls
                self.reasoningContent = reasoningContent
            }

            public func encode(to encoder: Encoder) throws {
                var container = encoder.container(keyedBy: CodingKeys.self)
                try container.encode(role, forKey: .role)
                if let content {
                    try container.encode(content, forKey: .content)
                } else {
                    try container.encodeNil(forKey: .content)
                }
                try container.encodeIfPresent(toolCalls, forKey: .toolCalls)
                try container.encodeIfPresent(reasoningContent, forKey: .reasoningContent)
            }
        }
        public let index: Int
        public let message: Message
        public let logprobs: ChoiceLogprobs?
        public let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index
            case message
            case logprobs
            case finishReason = "finish_reason"
        }
    }

    public struct Usage: Codable, Sendable {
        public struct PromptTokensDetails: Codable, Sendable {
            public let cachedTokens: Int

            enum CodingKeys: String, CodingKey {
                case cachedTokens = "cached_tokens"
            }

            public init(cachedTokens: Int) {
                self.cachedTokens = cachedTokens
            }
        }

        public struct CompletionTokensDetails: Codable, Sendable {
            public let reasoningTokens: Int

            enum CodingKeys: String, CodingKey {
                case reasoningTokens = "reasoning_tokens"
            }

            public init(reasoningTokens: Int) {
                self.reasoningTokens = reasoningTokens
            }
        }

        public let promptTokens: Int
        public let completionTokens: Int
        public let totalTokens: Int
        public let promptTokensDetails: PromptTokensDetails?
        public let completionTokensDetails: CompletionTokensDetails?

        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
            case promptTokensDetails = "prompt_tokens_details"
            case completionTokensDetails = "completion_tokens_details"
        }

        public init(promptTokens: Int, completionTokens: Int, totalTokens: Int,
                    cachedTokens: Int? = nil, reasoningTokens: Int? = nil) {
            self.promptTokens = promptTokens
            self.completionTokens = completionTokens
            self.totalTokens = totalTokens
            self.promptTokensDetails = cachedTokens.map { PromptTokensDetails(cachedTokens: $0) }
            self.completionTokensDetails = reasoningTokens.map { CompletionTokensDetails(reasoningTokens: $0) }
        }
    }

    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [Choice]
    public let usage: Usage

    public init(
        id: String,
        model: String,
        content: String?,
        toolCalls: [ToolCall]? = nil,
        reasoningContent: String? = nil,
        logprobs: ChoiceLogprobs? = nil,
        promptTokens: Int,
        completionTokens: Int,
        cachedTokens: Int? = nil,
        reasoningTokens: Int? = nil,
        finishReason: String = "stop"
    ) {
        self.id = id
        self.object = "chat.completion"
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        self.choices = [
            Choice(
                index: 0,
                message: Choice.Message(
                    role: "assistant",
                    content: content,
                    toolCalls: toolCalls,
                    reasoningContent: reasoningContent
                ),
                logprobs: logprobs,
                finishReason: finishReason
            )
        ]
        self.usage = Usage(
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            totalTokens: promptTokens + completionTokens,
            cachedTokens: cachedTokens,
            reasoningTokens: reasoningTokens
        )
    }
}

public struct ChatCompletionChunk: Codable, Sendable {
    public struct Choice: Codable, Sendable {
        public struct Delta: Codable, Sendable {
            public let role: String?
            public let content: String?
            public let toolCalls: [ToolCallChunk]?
            public let reasoningContent: String?

            enum CodingKeys: String, CodingKey {
                case role
                case content
                case toolCalls = "tool_calls"
                case reasoningContent = "reasoning_content"
            }

            public init(
                role: String? = nil,
                content: String? = nil,
                toolCalls: [ToolCallChunk]? = nil,
                reasoningContent: String? = nil
            ) {
                self.role = role
                self.content = content
                self.toolCalls = toolCalls
                self.reasoningContent = reasoningContent
            }
        }
        public let index: Int
        public let delta: Delta
        public let logprobs: ChoiceLogprobs?
        public let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index
            case delta
            case logprobs
            case finishReason = "finish_reason"
        }
    }

    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [Choice]
    public let usage: ChatCompletionResponse.Usage?

    public init(
        id: String,
        model: String,
        role: String? = nil,
        content: String? = nil,
        toolCall: ToolCallChunk? = nil,
        reasoningContent: String? = nil,
        logprobs: ChoiceLogprobs? = nil,
        finishReason: String? = nil,
        usage: ChatCompletionResponse.Usage? = nil
    ) {
        self.id = id
        self.object = "chat.completion.chunk"
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        let toolCalls = toolCall != nil ? [toolCall!] : nil
        self.choices = [
            Choice(
                index: 0,
                delta: Choice.Delta(
                    role: role,
                    content: content,
                    toolCalls: toolCalls,
                    reasoningContent: reasoningContent
                ),
                logprobs: logprobs,
                finishReason: finishReason
            )
        ]
        self.usage = usage
    }
}
