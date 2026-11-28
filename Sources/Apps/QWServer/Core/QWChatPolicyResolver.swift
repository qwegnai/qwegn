// QWChatPolicyResolver.swift — Consolidated thinking, stream parser, sampling,
// token-constraint budget, and prefix-cache/speculation policy resolution.

import Foundation
import QWCore
import QWEngine
import QWHead
import QWToken

public struct QWServerContextLengthError: Error, CustomStringConvertible, Sendable {
    public let promptTokens: Int
    public let maxContext: Int

    public init(promptTokens: Int, maxContext: Int) {
        self.promptTokens = promptTokens
        self.maxContext = maxContext
    }

    public var description: String {
        "This model's maximum context length is \(maxContext) tokens. However, your messages resulted in \(promptTokens) tokens. Please reduce the length of the messages."
    }
}

/// Resolved thinking enablement and token-budget policy for a chat turn.
public struct QWChatThinkingPolicy: Sendable, Equatable {
    public let shouldThink: Bool
    public let thinkingBudget: Int?

    public init(shouldThink: Bool, thinkingBudget: Int?) {
        self.shouldThink = shouldThink
        self.thinkingBudget = thinkingBudget
    }
}

/// Consolidated endpoint parser and template gating policy for a chat turn.
public struct QWChatStreamPolicy: Sendable, Equatable {
    public let hasActiveTools: Bool
    public let isJsonResponseFormat: Bool
    public let allowToolCalls: Bool
    public let shouldThink: Bool
    public let thinkingBudget: Int?
    public let allowReasoningTags: Bool

    public init(
        hasActiveTools: Bool,
        isJsonResponseFormat: Bool,
        allowToolCalls: Bool,
        shouldThink: Bool,
        thinkingBudget: Int?,
        allowReasoningTags: Bool
    ) {
        self.hasActiveTools = hasActiveTools
        self.isJsonResponseFormat = isJsonResponseFormat
        self.allowToolCalls = allowToolCalls
        self.shouldThink = shouldThink
        self.thinkingBudget = thinkingBudget
        self.allowReasoningTags = allowReasoningTags
    }
}

/// Resolved context and max-new-token generation budget for a request.
public struct QWChatBudgetResolution: Sendable, Equatable {
    public let effectiveMaxContext: Int
    public let requestedMaxNew: Int
    public let clampedMaxNew: Int

    public init(effectiveMaxContext: Int, requestedMaxNew: Int, clampedMaxNew: Int) {
        self.effectiveMaxContext = effectiveMaxContext
        self.requestedMaxNew = requestedMaxNew
        self.clampedMaxNew = clampedMaxNew
    }
}

/// Merged sampling parameters and provenance label for logging.
public struct QWChatSamplingResolution: Sendable {
    public let parameters: QWSamplingParameters
    public let hasRequestOverride: Bool
    public let sourceLabel: String

    public init(parameters: QWSamplingParameters, hasRequestOverride: Bool, sourceLabel: String) {
        self.parameters = parameters
        self.hasRequestOverride = hasRequestOverride
        self.sourceLabel = sourceLabel
    }
}

/// Resolved KV prefix-cache retention and exact-speculation flags.
public struct QWChatCachePolicy: Sendable, Equatable {
    public let usePrefixCache: Bool
    public let cacheSalt: String?
    public let retainPrefixLength: Int?
    public let exactSpeculation: Bool

    public init(
        usePrefixCache: Bool,
        cacheSalt: String?,
        retainPrefixLength: Int?,
        exactSpeculation: Bool
    ) {
        self.usePrefixCache = usePrefixCache
        self.cacheSalt = cacheSalt
        self.retainPrefixLength = retainPrefixLength
        self.exactSpeculation = exactSpeculation
    }
}

/// Single authority for chat/completion policy resolution across handlers, preparers, and engine execution.
public enum QWChatPolicyResolver {

    private static let effortThinkingCaps: [String: Int] = [
        "low": 256, "min": 256, "minimal": 256, "light": 256, "fast": 256, "quick": 256,
        "medium": 1024, "mid": 1024, "med": 1024, "normal": 1024, "auto": 1024, "default": 1024,
        "high": 4096, "xhigh": 8192, "max": 8192, "maximum": 8192, "deep": 8192,
    ]

    public static func isAutoReasoningEffort(_ raw: String?) -> Bool {
        raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "auto"
    }

    public static func resolveAutoReasoningEffort(
        tier: String,
        confidence: Double,
        confidenceThreshold: Double = 0.7
    ) -> String? {
        guard confidence >= confidenceThreshold else { return nil }
        switch tier.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "trivial":
            return "off"
        case "moderate":
            return "low"
        case "deep":
            return "xhigh"
        default:
            return nil
        }
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

    public static func effectiveEnableThinking(for request: ChatCompletionRequest, defaultValue: Bool) -> Bool {
        if request.isAssistantPrefill { return false }
        let kwargThinking = request.chatTemplateKwargs?.enableThinking ?? request.chatTemplateKwargs?.reasoning
        if let fmtType = request.responseFormatType, (fmtType == "json_object" || fmtType == "json_schema") {
            return request.enableThinking ?? kwargThinking ?? request.reasoning?.enabled ?? false
        }
        if let explicit = request.enableThinking ?? kwargThinking ?? request.reasoning?.enabled {
            return explicit
        }
        if let effort = request.rawReasoningEffort?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
           !effort.isEmpty {
            return !(effort == "none" || effort == "off" || effort == "false" || effort == "disabled" || effort == "disable")
        }
        if let budget = request.reasoningBudget { return budget > 0 }
        return defaultValue
    }

    public static func reservedThinkingBudget(maxNewTokens: Int, clampedFallback: Bool = false) -> Int? {
        let maxNew = max(0, maxNewTokens)
        if maxNew <= 32 {
            return clampedFallback ? max(1, maxNew / 2) : nil
        }
        return max(16, maxNew - min(256, max(16, maxNew / 4)))
    }

    public static func resolveThinkingPolicy(
        for request: ChatCompletionRequest,
        defaultEnableThinking: Bool,
        defaultMaxTokens: Int,
        modelMaxLength: Int? = nil,
        overrideMaxNewTokens: Int? = nil
    ) -> QWChatThinkingPolicy {
        guard effectiveEnableThinking(for: request, defaultValue: defaultEnableThinking) else {
            return QWChatThinkingPolicy(shouldThink: false, thinkingBudget: nil)
        }
        let fmtType = request.responseFormatType
        let isJsonFormat = fmtType == "json_object" || fmtType == "json_schema"
        let maxNew = max(
            0,
            overrideMaxNewTokens ?? request.maxTokens ?? request.maxCompletionTokens ?? modelMaxLength ?? defaultMaxTokens
        )
        let reserved = reservedThinkingBudget(maxNewTokens: maxNew)
        if request.reasoningBudget == nil,
           let raw = request.rawReasoningEffort?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
           let cap = effortThinkingCaps[raw] {
            let budget = reserved.map { min(cap, $0) } ?? max(1, min(cap, maxNew / 2))
            return QWChatThinkingPolicy(shouldThink: true, thinkingBudget: budget)
        }
        if request.reasoningBudget == nil {
            if !request.hasExplicitThinkingSetting || isJsonFormat {
                if let reserved {
                    return QWChatThinkingPolicy(shouldThink: true, thinkingBudget: reserved)
                }
                return isJsonFormat
                    ? QWChatThinkingPolicy(shouldThink: true, thinkingBudget: max(1, maxNew / 2))
                    : QWChatThinkingPolicy(shouldThink: false, thinkingBudget: nil)
            }
            return QWChatThinkingPolicy(shouldThink: true, thinkingBudget: reserved)
        }
        return QWChatThinkingPolicy(shouldThink: true, thinkingBudget: request.reasoningBudget.map { max(0, $0) })
    }

    /// Resolves unified tool, JSON-format, thinking, and parser-tag policies across endpoints and preparers.
    public static func resolveStreamPolicy(
        for request: ChatCompletionRequest,
        defaultEnableThinking: Bool,
        defaultMaxTokens: Int,
        modelMaxLength: Int? = nil,
        supportsThinkingTags: Bool,
        thinkingExplicitlyDisabled: Bool = false,
        overrideMaxNewTokens: Int? = nil
    ) -> QWChatStreamPolicy {
        let hasActiveTools = !(request.tools?.isEmpty ?? true) && !request.isToolChoiceNone
        let fmtType = request.responseFormatType
        let isJsonResponseFormat = (fmtType == "json_schema" || fmtType == "json_object") && !hasActiveTools
        let thinking = resolveThinkingPolicy(
            for: request,
            defaultEnableThinking: defaultEnableThinking,
            defaultMaxTokens: defaultMaxTokens,
            modelMaxLength: modelMaxLength,
            overrideMaxNewTokens: overrideMaxNewTokens
        )
        let shouldThink = thinking.shouldThink && supportsThinkingTags && !thinkingExplicitlyDisabled
        let thinkingBudget = (supportsThinkingTags && !thinkingExplicitlyDisabled) ? thinking.thinkingBudget : nil
        let allowReasoningTags = (!isJsonResponseFormat || shouldThink) && !thinkingExplicitlyDisabled
        return QWChatStreamPolicy(
            hasActiveTools: hasActiveTools,
            isJsonResponseFormat: isJsonResponseFormat,
            allowToolCalls: hasActiveTools,
            shouldThink: shouldThink,
            thinkingBudget: thinkingBudget,
            allowReasoningTags: allowReasoningTags
        )
    }

    public static func capToolCalls(
        _ calls: [ToolCall]?,
        parallelToolCalls: Bool?,
        maxToolCalls: Int? = nil
    ) -> [ToolCall]? {
        guard let calls, !calls.isEmpty else { return nil }
        var limit = calls.count
        if parallelToolCalls == false { limit = min(limit, 1) }
        if let maxToolCalls { limit = min(limit, max(0, maxToolCalls)) }
        let sliced = Array(calls.prefix(limit))
        return sliced.isEmpty ? nil : sliced
    }

    public static func effectiveContextBudget(configMaxContext: Int, backendContextBudget: Int) -> Int {
        min(configMaxContext, backendContextBudget > 0 ? backendContextBudget : configMaxContext)
    }

    public static func resolveChatBudget(
        for request: ChatCompletionRequest,
        promptTokenCount: Int,
        config: QWEngineLifecycleActor.Config,
        backendContextBudget: Int
    ) throws -> QWChatBudgetResolution {
        let effectiveMaxContext = effectiveContextBudget(
            configMaxContext: config.maxContext,
            backendContextBudget: backendContextBudget
        )
        guard promptTokenCount < effectiveMaxContext else {
            throw QWServerContextLengthError(promptTokens: promptTokenCount, maxContext: effectiveMaxContext)
        }
        let requestedMaxNew = request.maxTokens ?? request.maxCompletionTokens
            ?? config.modelMaxLength.map { max(1, $0 - promptTokenCount) }
            ?? config.maxTokens
        let clampedMaxNew = min(requestedMaxNew, max(1, effectiveMaxContext - promptTokenCount))
        return QWChatBudgetResolution(
            effectiveMaxContext: effectiveMaxContext,
            requestedMaxNew: requestedMaxNew,
            clampedMaxNew: clampedMaxNew
        )
    }

    public static func makeThinkingBudgetConstraint(
        maxThinkingTokens: Int,
        endThinkTokenId: Int,
        endThinkCloseTag: String? = nil,
        vocabularySize: Int,
        tokenizer: QWTokenizer,
        supportsTokenConstraints: Bool,
        inner: (any QWTokenConstraint)?
    ) -> QWThinkingBudgetConstraint? {
        if tokenizer.tokenId(for: "<|content_thinking|>") != nil {
            return nil
        }
        let resolvedPiece = tokenizer.piece(endThinkTokenId)
        let hasKnownEndThink = (resolvedPiece.map { $0.hasPrefix("</") && $0.hasSuffix(">") } ?? false)
            || (endThinkCloseTag.flatMap { tokenizer.tokenId(for: $0) } != nil)
        guard supportsTokenConstraints,
              endThinkTokenId >= 0,
              endThinkTokenId < vocabularySize,
              (hasKnownEndThink || endThinkTokenId != QWModelSpecialTokens().endThinkTokenId) else {
            return nil
        }
        let tokenBytes = QWStructuredOutput.tokenBytes(for: tokenizer, vocabulary: vocabularySize)
        return QWThinkingBudgetConstraint(
            maxThinkingTokens: maxThinkingTokens,
            endThinkTokenId: endThinkTokenId,
            endThinkCloseTag: endThinkCloseTag,
            vocabularySize: vocabularySize,
            tokenBytes: tokenBytes,
            inner: inner
        )
    }

    public static func resolveActiveConstraint(
        prepared: QWPreparedChatPrompt,
        request: ChatCompletionRequest,
        didHealPeel: Bool,
        healConstraint: (any QWTokenConstraint)?,
        requestedMaxNew: Int,
        clampedMaxNew: Int,
        tokenizer: QWTokenizer,
        vocabularySize: Int,
        supportsTokenConstraints: Bool
    ) -> (any QWTokenConstraint)? {
        let fmtType = request.responseFormatType
        let isJsonFormat = fmtType == "json_schema" || fmtType == "json_object"
        if !didHealPeel, prepared.shouldThink, request.reasoningBudget == nil,
           (!request.hasExplicitThinkingSetting || isJsonFormat),
           clampedMaxNew < requestedMaxNew,
           let clampedBudget = reservedThinkingBudget(maxNewTokens: clampedMaxNew, clampedFallback: true),
           let constraint = makeThinkingBudgetConstraint(
               maxThinkingTokens: clampedBudget,
               endThinkTokenId: prepared.endThinkTokenId,
               vocabularySize: vocabularySize,
               tokenizer: tokenizer,
               supportsTokenConstraints: supportsTokenConstraints,
               inner: prepared.innerConstraint
           ) {
            return constraint
        }
        return healConstraint
    }

    public static func resolveSampling(
        temperature: Double?,
        topP: Double?,
        topK: Int?,
        repetitionPenalty: Double?,
        seed: Int?,
        config: QWEngineLifecycleActor.Config
    ) -> QWChatSamplingResolution {
        let parameters = config.samplingParameters.merging(
            temperature: temperature,
            topP: topP,
            topK: topK,
            repetitionPenalty: repetitionPenalty,
            seed: seed
        )
        let hasOverride = temperature != nil || topP != nil || topK != nil
            || repetitionPenalty != nil || seed != nil
        let sourceLabel = hasOverride
            ? "request override"
            : QWEngineLifecycleActor.samplingParamsSourceLabel(config.sourceSummary)
        return QWChatSamplingResolution(
            parameters: parameters,
            hasRequestOverride: hasOverride,
            sourceLabel: sourceLabel
        )
    }

    public static func resolveSampling(
        for request: ChatCompletionRequest,
        config: QWEngineLifecycleActor.Config
    ) -> QWChatSamplingResolution {
        resolveSampling(
            temperature: request.temperature,
            topP: request.topP,
            topK: request.topK,
            repetitionPenalty: request.repetitionPenalty,
            seed: request.seed,
            config: config
        )
    }

    public static func resolveSampling(
        for request: CompletionRequest,
        config: QWEngineLifecycleActor.Config
    ) -> QWChatSamplingResolution {
        resolveSampling(
            temperature: request.temperature,
            topP: request.topP,
            topK: request.topK,
            repetitionPenalty: request.repetitionPenalty,
            seed: request.seed,
            config: config
        )
    }

    public static func resolveCacheAndSpeculation(
        for request: ChatCompletionRequest,
        systemPrefixTokenCount: Int?,
        firstVisualTokenOffset: Int,
        isConstrainedJson: Bool,
        hasExplicitSeed: Bool
    ) -> QWChatCachePolicy {
        let usePrefixCache = request.effectiveCachePrompt
        let cacheSalt = request.effectiveCacheSalt
        let retainPrefixLength: Int? = (usePrefixCache
            && (systemPrefixTokenCount ?? 0) >= 128
            && (systemPrefixTokenCount ?? 0) < firstVisualTokenOffset)
            ? systemPrefixTokenCount
            : nil
        let exactSpeculation = isConstrainedJson || hasExplicitSeed || request.effectiveExactSpeculation
        return QWChatCachePolicy(
            usePrefixCache: usePrefixCache,
            cacheSalt: cacheSalt,
            retainPrefixLength: retainPrefixLength,
            exactSpeculation: exactSpeculation
        )
    }

    public static func buildLedgerConfig(
        request: ChatCompletionRequest,
        config: QWEngineLifecycleActor.Config,
        hasMountedManifest: Bool,
        trace: QWRequestTraceContext? = nil
    ) -> [String: String] {
        var out: [String: String] = [
            "kvMode": String(describing: config.kvMode),
            "mtpDrafts": String(config.mtpDrafts),
            "speculative": config.speculativeDecodingEnabled ? "on" : "off",
            "cachePrompt": request.effectiveCachePrompt ? "on" : "off",
            "memPack": hasMountedManifest ? "mounted" : (request.effectiveUseMemoryPack ? "auto" : "off"),
            "route": trace?.route ?? "/v1/chat/completions",
            "client": trace?.client ?? QWClientIdentity.resolve(headers: request.httpHeaders ?? [:]).client
        ]
        if let peer = trace?.peerHost { out["peer"] = peer }
        if let sid = request.effectiveSessionId { out["sessionId"] = sid }
        if let preset = request.effectivePresetId { out["preset"] = preset }
        if let salt = request.effectiveCacheSalt { out["cacheKey"] = salt }
        if request.effectiveExactSpeculation { out["exactSpec"] = "on" }
        if let user = request.effectiveUser { out["user"] = user }
        return out
    }
}
