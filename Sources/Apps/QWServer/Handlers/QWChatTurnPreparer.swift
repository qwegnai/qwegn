// QWChatTurnPreparer.swift — Deep module preparing chat prompts, system-message
// synthesis (JSON schema & forced tool choice), Jinja wire rendering, stop
// tokens, and token-level grammar/tool constraints for QWServer chat execution.

import Foundation
import QWCore
import QWEngine
import QWHead
import QWToken

/// Immutable prepared chat turn produced by `QWChatTurnPreparer`.
public struct QWPreparedChatPrompt: @unchecked Sendable {
    public let messages: [QWJson]
    public let templateOptions: QWChatTemplateOptions
    public let wire: String
    public let hasActiveTools: Bool
    public let shouldThink: Bool
    public let isAssistantPrefill: Bool
    public let isConstrainedJson: Bool
    public let enforceJson: Bool
    public let jsonConstraint: QWStructuredOutputConstraint?
    public let activeConstraint: (any QWTokenConstraint)?
    public let stops: Set<Int>
    public let imEndTokenId: Int
    public let endThinkTokenId: Int
    public let userTextContainsImEnd: Bool
    public let customStopStrings: [String]
}

/// Pure, in-process chat turn preparer testable directly at its interface.
public enum QWChatTurnPreparer {

    /// Synthesizes system instructions, renders the chat template wire string,
    /// and builds token-level grammar or tool-call constraints for `request`.
    public static func preparePrompt(
        request: ChatCompletionRequest,
        config: QWEngineLifecycleActor.Config,
        chatTemplate: QWChatTemplate,
        tokenizer tok: QWTokenizer,
        descriptor: QWModelDescriptor,
        supportsTokenConstraints: Bool,
        memoryManifest: QWMemoryManifest? = nil
    ) throws -> QWPreparedChatPrompt {
        try preparePrompt(
            request: request,
            config: config,
            chatTemplate: chatTemplate,
            tokenizer: tok,
            specialTokensFallback: descriptor.specialTokens,
            vocabularySize: descriptor.vocab,
            supportsTokenConstraints: supportsTokenConstraints,
            memoryManifest: memoryManifest
        )
    }

    /// Synthesizes system instructions, renders the chat template wire string,
    /// and builds token-level grammar or tool-call constraints for `request`.
    public static func preparePrompt(
        request: ChatCompletionRequest,
        config: QWEngineLifecycleActor.Config,
        chatTemplate: QWChatTemplate,
        tokenizer tok: QWTokenizer,
        specialTokensFallback: QWModelSpecialTokens = QWModelSpecialTokens(),
        vocabularySize: Int,
        supportsTokenConstraints: Bool,
        memoryManifest: QWMemoryManifest? = nil
    ) throws -> QWPreparedChatPrompt {
        let hasActiveTools = !(request.tools?.isEmpty ?? true) && !request.isToolChoiceNone
        let fmtType = request.responseFormatType
        let isAssistantPrefill = request.isAssistantPrefill
        let shouldThink = request.effectiveEnableThinking(defaultValue: config.enableThinking)

        var extraSystemLines: [String] = []
        if let fmtType, (fmtType == "json_schema" || fmtType == "json_object") {
            let nested = request.responseFormat?.get("json_schema").get("schema") ?? .undefined
            let schemaVal = nested.isUndefined ? (request.responseFormat?.get("schema") ?? .undefined) : nested
            let schemaEncoder = JSONEncoder()
            schemaEncoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            if fmtType == "json_schema", !schemaVal.isUndefined,
               let data = try? schemaEncoder.encode(schemaVal),
               let rawSchemaStr = String(data: data, encoding: .utf8) {
                let schemaStr = rawSchemaStr
                    .replacingOccurrences(of: "<", with: "\\u003c")
                    .replacingOccurrences(of: ">", with: "\\u003e")
                extraSystemLines.append(
                    "When providing your final answer (not calling a tool), you MUST output ONLY a single raw, valid JSON object conforming strictly to this JSON Schema (include all required properties with their exact types, no additional properties beyond the schema, no markdown code fences, and no prose):\n\(schemaStr)"
                )
            } else {
                extraSystemLines.append(
                    "When providing your final answer (not calling a tool), you MUST output ONLY a single raw, valid JSON object (no markdown code fences and no prose)."
                )
            }
        }
        if hasActiveTools, let forcedTool = request.forcedToolChoiceName {
            if forcedTool.isEmpty {
                extraSystemLines.append("You MUST call at least one of the provided tools in this response using <tool_call>.")
            } else {
                extraSystemLines.append("You MUST call the tool '\(forcedTool)' in this response using <tool_call>.")
            }
            if request.parallelToolCalls == false {
                extraSystemLines.append("Call at most ONE tool in this response; do not emit multiple tool calls.")
            }
        }

        var msgs = request.toQWJsonMessages()
        if let manifest = memoryManifest {
            Self.applyMemoryVaultAndTailDelta(
                to: &msgs,
                manifest: manifest,
                isAssistantPrefill: isAssistantPrefill
            )
        }
        if !config.systemPrompt.isEmpty {
            let hasSystem = request.messages.contains(where: {
                let r = $0.role.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                return r == "system" || r == "developer"
            })
            if !hasSystem {
                msgs.insert(.object([("role", .string("system")), ("content", .string(config.systemPrompt))]), at: 0)
            }
        }
        if !extraSystemLines.isEmpty {
            let extraBlock = extraSystemLines.joined(separator: "\n\n")
            if let first = msgs.first, case .string(let r) = first.get("role"), r == "system",
               case .string(let existing) = first.get("content") {
                let combined = existing.isEmpty ? extraBlock : (existing + "\n\n" + extraBlock)
                msgs[0] = .object([("role", .string("system")), ("content", .string(combined))])
            } else {
                msgs.insert(.object([("role", .string("system")), ("content", .string(extraBlock))]), at: 0)
            }
        }

        var tplOpts = QWChatTemplateOptions()
        tplOpts.addGenerationPrompt = true
        tplOpts.enableThinking = .bool(shouldThink)
        let effort = request.effectiveReasoningEffort(defaultValue: config.reasoningEffort)
        tplOpts.reasoningEffort = .string(effort)
        if hasActiveTools, let tools = request.tools {
            tplOpts.tools = .array(tools)
        }

        let grammar = try QWStructuredOutput.grammar(for: request)
        let toolCallTokenId = hasActiveTools ? tok.tokenId(for: "<tool_call>") : nil
        let enforceJson = grammar != nil && supportsTokenConstraints
            && (!hasActiveTools || toolCallTokenId != nil)
        if grammar != nil && !enforceJson && QWStructuredOutput.isStrict(request) {
            throw QWJSONSchemaError("strict json_schema needs an engine with constrained decoding")
        }
        let isConstrainedJson = (fmtType == "json_schema" || fmtType == "json_object") && !enforceJson
            && !hasActiveTools && !shouldThink && !isAssistantPrefill && !(request.logprobs ?? false)

        var wire: String
        if isAssistantPrefill, let lastMsg = request.messages.last {
            let prefixMsgs = Array(msgs.dropLast())
            wire = try chatTemplate.render(messages: prefixMsgs, options: tplOpts)
            wire += lastMsg.extractTextContent()
        } else {
            wire = try chatTemplate.render(messages: msgs, options: tplOpts)
            if isConstrainedJson {
                wire += "{"
            }
        }

        let specialTokens = tok.resolveSpecialTokens(fallback: specialTokensFallback)
        var stops: Set<Int> = specialTokens.defaultStopTokenIds
        let imEndTokenId = specialTokens.imEndTokenId
        let endThinkTokenId = specialTokens.endThinkTokenId
        let userTextContainsImEnd = request.messages.contains { $0.extractTextContent().contains("<|im_end|>") }
        if userTextContainsImEnd {
            stops.remove(imEndTokenId)
        }
        var builtConstraint: QWStructuredOutputConstraint?
        if enforceJson, let grammar {
            let table = QWStructuredOutput.tokenTable(for: tok, vocabulary: vocabularySize)
            let constraint = QWStructuredOutputConstraint(
                grammar: grammar, table: table, stopTokens: specialTokens.defaultStopTokenIds,
                startsInThinking: shouldThink, endThinkToken: endThinkTokenId, toolCallToken: toolCallTokenId)
            if isAssistantPrefill, let lastMsg = request.messages.last {
                try constraint.consumePrefix(lastMsg.extractTextContent())
            }
            builtConstraint = constraint
        }
        let jsonConstraint = builtConstraint
        var builtToolConstraint: QWToolCallConstraint?
        if jsonConstraint == nil, hasActiveTools, supportsTokenConstraints,
           let tools = request.tools, toolCallTokenId != nil {
            let table = QWStructuredOutput.tokenTable(for: tok, vocabulary: vocabularySize)
            builtToolConstraint = QWToolCallConstraint(
                tools: tools, forcedToolName: request.forcedToolChoiceName,
                table: table, toolCallToken: toolCallTokenId,
                toolCallEndToken: tok.tokenId(for: "</tool_call>"))
        }
        let activeConstraint: (any QWTokenConstraint)? = jsonConstraint ?? builtToolConstraint
        let customStopStrings: [String] = (request.ignoreEos == true)
            ? []
            : (request.stop ?? []).filter { !$0.isEmpty }

        return QWPreparedChatPrompt(
            messages: msgs,
            templateOptions: tplOpts,
            wire: wire,
            hasActiveTools: hasActiveTools,
            shouldThink: shouldThink,
            isAssistantPrefill: isAssistantPrefill,
            isConstrainedJson: isConstrainedJson,
            enforceJson: enforceJson,
            jsonConstraint: jsonConstraint,
            activeConstraint: activeConstraint,
            stops: stops,
            imEndTokenId: imEndTokenId,
            endThinkTokenId: endThinkTokenId,
            userTextContainsImEnd: userTextContainsImEnd,
            customStopStrings: customStopStrings
        )
    }

    /// Injects causal Vault recalls on user and active tool turns and appends any external
    /// working-tree delta to the final message so historical prefix tokens never shift.
    private static func applyMemoryVaultAndTailDelta(
        to msgs: inout [QWJson],
        manifest: QWMemoryManifest,
        isAssistantPrefill: Bool
    ) {
        let hasVaultFiles = manifest.files.contains(where: { !$0.inCortex })
        var seenVaultPaths: Set<String> = []
        var agentModifiedPaths: Set<String> = []
        var lastAssistantToolQuery = ""

        for idx in msgs.indices {
            let msgRole = role(of: msgs[idx])
            let msgText = textContent(of: msgs[idx])
            if msgRole == "assistant" {
                lastAssistantToolQuery = extractAssistantToolStepContext(
                    from: msgs[idx],
                    manifest: manifest,
                    seenVaultPaths: &seenVaultPaths,
                    agentModifiedPaths: &agentModifiedPaths
                )
                continue
            }
            guard hasVaultFiles, !msgText.contains("<|memory_vault_recall|>") else { continue }
            if msgRole == "user" {
                let recall = QWMemoryVault.recallVaultContext(
                    query: msgText,
                    manifest: manifest,
                    excludedPaths: seenVaultPaths,
                    preserveHistoricalSnapshot: idx < msgs.count - 1
                )
                if !recall.isEmpty {
                    seenVaultPaths.formUnion(recall.recalledFiles)
                    msgs[idx] = appendingContent(recall.recallBlock, to: msgs[idx])
                }
            } else if msgRole == "tool" {
                let isLastToolOfStep = (idx == msgs.count - 1) || (role(of: msgs[idx + 1]) != "tool")
                guard isLastToolOfStep else { continue }
                let combinedQuery = lastAssistantToolQuery + "\n" + String(msgText.prefix(1500))
                lastAssistantToolQuery = ""
                let recall = QWMemoryVault.recallVaultContext(
                    query: combinedQuery,
                    manifest: manifest,
                    excludedPaths: seenVaultPaths,
                    preserveHistoricalSnapshot: idx < msgs.count - 1
                )
                if !recall.isEmpty {
                    seenVaultPaths.formUnion(recall.recalledFiles)
                    msgs[idx] = appendingContent(recall.recallBlock, to: msgs[idx])
                }
            }
        }
    }

    private static func extractAssistantToolStepContext(
        from msg: QWJson,
        manifest: QWMemoryManifest,
        seenVaultPaths: inout Set<String>,
        agentModifiedPaths: inout Set<String>
    ) -> String {
        var queryParts: [String] = []
        let content = textContent(of: msg)
        if !content.isEmpty { queryParts.append(String(content.prefix(600))) }
        if case .string(let reasoning) = msg.get("reasoning_content"), !reasoning.isEmpty {
            queryParts.append(String(reasoning.suffix(600)))
        }
        guard case .array(let calls) = msg.get("tool_calls") else {
            return queryParts.joined(separator: "\n")
        }
        for call in calls {
            let fnObj = call.get("function")
            let fnName: String
            if case .string(let n) = fnObj.get("name") { fnName = n.lowercased() } else { fnName = "" }
            let argsObj = fnObj.get("arguments")
            var rawPath: String?
            for key in ["path", "file_path", "TargetFile", "file"] {
                if case .string(let p) = argsObj.get(key), !p.isEmpty {
                    rawPath = p
                    break
                }
            }
            if let rawPath {
                queryParts.append(rawPath)
                if let rel = resolveManifestRelativePath(rawPath, in: manifest) {
                    if ["read_file", "write_file", "patch", "edit_file", "replace_in_file"].contains(fnName) {
                        seenVaultPaths.insert(rel)
                    }
                    if ["write_file", "patch", "edit_file", "replace_in_file"].contains(fnName) {
                        agentModifiedPaths.insert(rel)
                    }
                }
            }
            for key in ["command", "pattern", "query", "symbol"] {
                if case .string(let val) = argsObj.get(key), !val.isEmpty {
                    queryParts.append(String(val.prefix(400)))
                }
            }
        }
        return queryParts.joined(separator: "\n")
    }

    private static func resolveManifestRelativePath(_ rawPath: String, in manifest: QWMemoryManifest) -> String? {
        let trimmed = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let rootPrefix = manifest.rootPath.hasSuffix("/") ? manifest.rootPath : (manifest.rootPath + "/")
        let candidate: String
        if trimmed.hasPrefix(rootPrefix) {
            candidate = String(trimmed.dropFirst(rootPrefix.count))
        } else if trimmed.hasPrefix("./") {
            candidate = String(trimmed.dropFirst(2))
        } else {
            candidate = trimmed
        }
        if manifest.files.contains(where: { $0.relativePath == candidate }) {
            return candidate
        }
        return manifest.files.first(where: { candidate.hasSuffix("/" + $0.relativePath) })?.relativePath
    }

    private static func role(of msg: QWJson) -> String {
        if case .string(let r) = msg.get("role") { return r }
        return ""
    }

    private static func textContent(of msg: QWJson) -> String {
        if case .string(let s) = msg.get("content") { return s }
        return ""
    }

    private static func appendingContent(_ suffix: String, to message: QWJson) -> QWJson {
        guard !suffix.isEmpty, case .object(var pairs) = message else { return message }
        for idx in pairs.indices where pairs[idx].0 == "content" {
            if case .string(let existing) = pairs[idx].1 {
                pairs[idx].1 = .string(existing + suffix)
                return .object(pairs)
            } else if case .null = pairs[idx].1 {
                pairs[idx].1 = .string(suffix)
                return .object(pairs)
            }
        }
        pairs.append(("content", .string(suffix)))
        return .object(pairs)
    }
}
