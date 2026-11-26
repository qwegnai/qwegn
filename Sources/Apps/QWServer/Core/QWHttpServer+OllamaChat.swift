// QWHttpServer+OllamaChat.swift — Disk model stats and Ollama /api/show + /api/chat handlers
// using QWSSESession, QWChatPolicyResolver, and direct OllamaChatRequest mapping.

import Foundation
import Network
import QWCore
import QWEngine
import QWToken

/// The facts a model carries on disk: the bytes it occupies and when
/// those bytes changed. Everything the compatible endpoints publish
/// about a model is read from the file system, never invented.
public enum QWModelDiskStats {
    public static func sizeBytes(of path: String?) -> Int64? {
        guard let path, !path.isEmpty else { return nil }
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: path, isDirectory: &isDir) else { return nil }
        if isDir.boolValue {
            var total: Int64 = 0
            if let en = fm.enumerator(atPath: path) {
                for case let rel as String in en {
                    let attrs = try? fm.attributesOfItem(atPath: (path as NSString).appendingPathComponent(rel))
                    if let n = attrs?[.size] as? NSNumber { total += n.int64Value }
                }
            }
            return total
        }
        let attrs = try? fm.attributesOfItem(atPath: path)
        return (attrs?[.size] as? NSNumber)?.int64Value
    }

    public static func modifiedRFC3339(of path: String?) -> String? {
        guard let path, !path.isEmpty else { return nil }
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        guard let date = attrs?[.modificationDate] as? Date else { return nil }
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyy-MM-dd'T'HH:mm:ssZ"
        return f.string(from: date)
    }

    public static func modifiedUnix(of path: String?) -> Int? {
        guard let path, !path.isEmpty else { return nil }
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        guard let date = attrs?[.modificationDate] as? Date else { return nil }
        return Int(date.timeIntervalSince1970)
    }
}

struct OllamaChatResponse: Codable {
    let model: String
    let createdAt: String
    let done: Bool
    let doneReason: String?
    let promptEvalCount: Int
    let evalCount: Int
    let message: OllamaChatMessage

    enum CodingKeys: String, CodingKey {
        case model, done, message
        case createdAt = "created_at"
        case doneReason = "done_reason"
        case promptEvalCount = "prompt_eval_count"
        case evalCount = "eval_count"
    }

    static func now() -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f.string(from: Date())
    }
}

struct OllamaToolCallPayload: Codable {
    struct FunctionPayload: Codable {
        let name: String
        let arguments: QWJson
    }
    let function: FunctionPayload
}

struct OllamaChatMessage: Codable {
    let role: String
    let content: String
    let thinking: String?
    let toolCalls: [OllamaToolCallPayload]?

    enum CodingKeys: String, CodingKey {
        case role, content, thinking
        case toolCalls = "tool_calls"
    }

    init(
        role: String,
        content: String,
        thinking: String? = nil,
        toolCalls: [OllamaToolCallPayload]? = nil
    ) {
        self.role = role
        self.content = content
        self.thinking = thinking
        self.toolCalls = toolCalls
    }
}

extension QWHttpServer {

    static func ollamaErrorStatus(for error: any Error) -> Int {
        if error is QWMemoryWorkspaceAccessError { return 403 }
        if error is QWMetadataValidationError
            || error is QWMultimodalError
            || error is QWServerContextLengthError
            || error is QWJSONSchemaError {
            return 400
        }
        return 500
    }

    static func ollamaParameterSize(modelName: String, descriptor: QWModelDescriptor?) -> String {
        if let desc = descriptor {
            let h = desc.hidden
            let v = desc.vocab
            let layers = desc.numLayers
            let embed = v * h
            let attnPerLayer = 2 * h * (desc.qHeads * desc.headDim) + 2 * h * (desc.kvHeads * desc.headDim)
            let mlpPerLayer: Int
            if let experts = desc.numExperts, let moeInter = desc.moeIntermediate, experts > 0 {
                let shared = (desc.sharedExpertIntermediate ?? 0) * 3 * h
                mlpPerLayer = experts * 3 * h * moeInter + shared
            } else {
                mlpPerLayer = 3 * h * (desc.denseIntermediate ?? h * 4)
            }
            let totalParams = Double(embed * 2 + layers * (attnPerLayer + mlpPerLayer))
            if totalParams >= 1e9 {
                let billions = totalParams / 1e9
                return billions >= 10
                    ? String(format: "%.0fB", billions.rounded())
                    : String(format: "%.1fB", billions)
            }
            if totalParams >= 1e6 {
                return String(format: "%.0fM", (totalParams / 1e6).rounded())
            }
        }
        if let match = modelName.range(of: #"(?i)(?:^|[-_ /])(\d+(?:\.\d+)?[BM])(?=$|[-_ ])"#, options: .regularExpression) {
            return modelName[match].trimmingCharacters(in: CharacterSet(charactersIn: "-_ /")).uppercased()
        }
        return "unknown"
    }

    func handleOllamaShow(connection: NWConnection, body: Data) {
        let parsedBody = (!body.isEmpty ? (try? JSONSerialization.jsonObject(with: body)) as? [String: Any] : nil) ?? [:]
        let requested = ((parsedBody["model"] as? String) ?? (parsedBody["name"] as? String) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let cfg = actor.getConfig()
        let activeName = actor.getModelName()
        let params = actor.getLoadedParamsInfo()
        let settings = QWSettingsManager.shared.loadSettings(from: settingsFileURL)

        let resolvedModelName: String
        let resolvedPackPath: String
        let descriptor: QWModelDescriptor?
        if requested.isEmpty
            || requested.caseInsensitiveCompare(activeName) == .orderedSame
            || requested.caseInsensitiveCompare(settings.alias(for: activeName)) == .orderedSame {
            resolvedModelName = activeName
            resolvedPackPath = cfg.packPath
            descriptor = params?.modelDescriptor ?? (!cfg.packPath.isEmpty ? QWModelCatalog.loadPrimaryDescriptor(at: cfg.packPath) : nil)
        } else {
            let discovered = modelCatalogScanner()
            let ref = settings.resolveModelReference(requested, discovered: discovered)
            let exists = discovered.contains(where: {
                $0.name.caseInsensitiveCompare(ref.modelName) == .orderedSame || $0.packPath == ref.packPath
            }) || (!ref.packPath.isEmpty && FileManager.default.fileExists(atPath: ref.packPath))
            guard exists else {
                sendJsonResponse(connection, status: 404, object: ["error": "model \"\(requested)\" not found"])
                return
            }
            resolvedModelName = ref.modelName
            resolvedPackPath = ref.packPath
            descriptor = !resolvedPackPath.isEmpty ? QWModelCatalog.loadPrimaryDescriptor(at: resolvedPackPath) : nil
        }

        let isActiveModel = resolvedModelName.caseInsensitiveCompare(activeName) == .orderedSame
            || (!cfg.packPath.isEmpty && resolvedPackPath == cfg.packPath)
        let family = descriptor?.architectureDisplayName ?? descriptor?.modelType ?? "unknown"
        let paramSize = Self.ollamaParameterSize(modelName: resolvedModelName, descriptor: descriptor)
        let quantLevel: String
        if resolvedPackPath == cfg.packPath, let loadedHead = params?.headStorage {
            quantLevel = loadedHead
        } else if !resolvedPackPath.isEmpty {
            quantLevel = QWBackendFacts.headStorageLabel(packPath: resolvedPackPath)
        } else {
            quantLevel = String(describing: cfg.kvMode)
        }
        let contextLen = descriptor?.maxPositionEmbeddings ?? params?.contextBudget ?? cfg.maxContext
        let packHasVision = (descriptor?.visDepth ?? 0) > 0 || descriptor?.visionTowerConfig != nil
        let hasVision = isActiveModel && actor.isLoaded() ? (packHasVision && cfg.loadVision) : packHasVision
        let supportsThinking: Bool
        if isActiveModel && actor.isLoaded() {
            supportsThinking = actor.supportsThinkingTags()
        } else if !resolvedPackPath.isEmpty {
            supportsThinking = QWChatTemplate.detectSupportsThinkingTags(atPackPath: resolvedPackPath) ?? false
        } else {
            supportsThinking = false
        }
        var capabilities = ["completion"]
        if hasVision { capabilities.append("vision") }
        if supportsThinking { capabilities.append("thinking") }
        let details: [String: Any] = [
            "format": "qwg",
            "family": family,
            "parameter_size": paramSize,
            "quantization_level": quantLevel
        ]
        let resp: [String: Any] = [
            "modelfile": "# Modelfile for Qwegine\nFROM \(resolvedModelName)",
            "parameters": "num_ctx \(contextLen)",
            "details": details,
            "capabilities": capabilities,
            "model_info": ["context_length": contextLen]
        ]
        if let data = try? JSONSerialization.data(withJSONObject: resp) {
            sendRawData(connection, status: 200, contentType: "application/json", data: data)
        } else {
            sendJsonResponse(connection, status: 200, object: ["status": "ok"])
        }
    }

    static func mapOllamaToolCalls(_ calls: [ToolCall]?) -> [OllamaToolCallPayload]? {
        guard let calls, !calls.isEmpty else { return nil }
        return calls.map { call in
            let parsedArgs = (try? QWJson.parse(call.function.arguments)) ?? QWJson.object([])
            return OllamaToolCallPayload(function: .init(name: call.function.name, arguments: parsedArgs))
        }
    }

    /// POST /api/chat — the Ollama chat endpoint.
    func handleOllamaChat(
        connection: NWConnection,
        headers: [String: String] = [:],
        body: Data,
        trace: QWRequestTraceContext? = nil
    ) async throws {
        let effectiveTrace = trace ?? QWClientIdentity.makeTraceContext(
            method: "POST", route: "/api/chat", headers: headers, peerEndpoint: connection.endpoint
        )
        let req: ChatCompletionRequest
        do {
            let ollamaReq = try QWServerRequestDecoder.decode(OllamaChatRequest.self, from: body)
            var mapped = ollamaReq.toChatCompletionRequest()
            mapped.httpHeaders = headers
            req = mapped
            try req.validateMetadata()
            _ = try QWStructuredOutput.grammar(for: req)
            try actor.validateRequestCapabilities(req)
        } catch let valErr {
            let code = Self.ollamaErrorStatus(for: valErr)
            let status = code == 500 ? 400 : code
            logger.log(.warn, "HTTP POST \(effectiveTrace.route) invalid request | \(effectiveTrace.logTag): \(valErr)")
            sendJsonResponse(connection, status: status, object: ["error": "invalid request: \(valErr)"])
            return
        }

        let requested = (req.model ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard let replyModel = resolveReplyModelName(requested: requested, strictNotFound: true) else {
            sendJsonResponse(connection, status: 404, object: ["error": "model \"\(requested)\" not found, try pulling it first"])
            return
        }

        let created = OllamaChatResponse.now()
        let shouldStop = generationShouldStop(for: connection)
        let cfg = actor.getConfig()
        let policy = QWChatPolicyResolver.resolveStreamPolicy(
            for: req,
            defaultEnableThinking: cfg.enableThinking,
            defaultMaxTokens: cfg.maxTokens,
            modelMaxLength: cfg.modelMaxLength,
            supportsThinkingTags: actor.supportsThinkingTags()
        )

        if req.isStreaming {
            let streamParser = QWChatStreamParser(
                startsInReasoning: policy.shouldThink,
                allowToolCalls: policy.allowToolCalls,
                allowReasoningTags: policy.allowReasoningTags,
                allowInk2Markers: actor.isInkling2Template() && (!policy.isJsonResponseFormat || policy.shouldThink)
            )
            let session = makeSSESession(for: connection, format: .ndjson, enableKeepalive: false)
            defer { session.stopAll() }
            let emitDelta: @Sendable (QWChatStreamParser.Delta) -> Void = { delta in
                switch delta {
                case .reasoning(let t):
                    session.sendNDJSONLine(OllamaChatResponse(
                        model: replyModel, createdAt: created, done: false, doneReason: nil,
                        promptEvalCount: 0, evalCount: 0,
                        message: OllamaChatMessage(role: "assistant", content: "", thinking: t)
                    ))
                case .content(let t):
                    session.sendNDJSONLine(OllamaChatResponse(
                        model: replyModel, createdAt: created, done: false, doneReason: nil,
                        promptEvalCount: 0, evalCount: 0,
                        message: OllamaChatMessage(role: "assistant", content: t)
                    ))
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
                            defaultEnableThinking: cfg.enableThinking,
                            defaultMaxTokens: cfg.maxTokens,
                            modelMaxLength: cfg.modelMaxLength,
                            supportsThinkingTags: self.actor.supportsThinkingTags()
                        )
                        streamParser.setStartsInReasoning(
                            readyPolicy.shouldThink,
                            allowReasoningTags: readyPolicy.allowReasoningTags,
                            allowInk2Markers: self.actor.isInkling2Template() && (!readyPolicy.isJsonResponseFormat || readyPolicy.shouldThink)
                        )
                        session.begin(startKeepalive: false)
                    },
                    onToken: { piece in
                        for delta in streamParser.feed(piece) {
                            emitDelta(delta)
                        }
                    },
                    shouldStop: shouldStop
                )
            } catch {
                if shouldStop() { return }
                if !session.isStarted {
                    logger.log(.error, "HTTP POST \(effectiveTrace.route) failed | \(effectiveTrace.logTag): \(error)")
                    sendJsonResponse(connection, status: Self.ollamaErrorStatus(for: error), object: ["error": "\(error)"])
                    return
                }
                logger.log(.error, "HTTP POST \(effectiveTrace.route) (stream) failed | \(effectiveTrace.logTag): \(error)")
                session.finishWithNDJSONError(error)
                return
            }

            guard !shouldStop() else { return }
            for delta in streamParser.finish() {
                emitDelta(delta)
            }
            let parsed = streamParser.result(
                tools: policy.allowToolCalls ? req.tools : nil,
                truncated: result.truncated
            )
            let effectiveCalls = QWChatPolicyResolver.capToolCalls(
                parsed.toolCalls,
                parallelToolCalls: req.parallelToolCalls
            )
            let done = OllamaChatResponse(
                model: replyModel, createdAt: created, done: true,
                doneReason: result.truncated ? "length" : "stop",
                promptEvalCount: result.promptTokens, evalCount: result.completionTokens,
                message: OllamaChatMessage(
                    role: "assistant",
                    content: "",
                    toolCalls: Self.mapOllamaToolCalls(effectiveCalls)
                )
            )
            session.sendNDJSONLine(done)
            session.finish(doneMarker: false)
        } else {
            let result: ChatCompletionExecutionResult
            do {
                result = try await actor.handleChatCompletion(
                    request: req, trace: effectiveTrace, onToken: nil, shouldStop: shouldStop
                )
            } catch {
                if shouldStop() { return }
                logger.log(.error, "HTTP POST \(effectiveTrace.route) failed | \(effectiveTrace.logTag): \(error)")
                sendJsonResponse(connection, status: Self.ollamaErrorStatus(for: error), object: ["error": "\(error)"])
                return
            }
            guard !shouldStop() else { return }
            let parsed = QWToolCallParser.parse(
                text: result.content,
                tools: policy.allowToolCalls ? req.tools : nil,
                startsInThinking: result.startsInThinking,
                truncated: result.truncated,
                allowToolCalls: policy.allowToolCalls,
                allowReasoningTags: policy.allowReasoningTags,
                allowInk2Markers: actor.isInkling2Template() && (!policy.isJsonResponseFormat || result.startsInThinking)
            )
            let effectiveCalls = QWChatPolicyResolver.capToolCalls(
                parsed.toolCalls,
                parallelToolCalls: req.parallelToolCalls
            )
            let resp = OllamaChatResponse(
                model: replyModel, createdAt: created, done: true,
                doneReason: result.truncated ? "length" : "stop",
                promptEvalCount: result.promptTokens, evalCount: result.completionTokens,
                message: OllamaChatMessage(
                    role: "assistant",
                    content: parsed.content ?? "",
                    thinking: parsed.reasoningContent,
                    toolCalls: Self.mapOllamaToolCalls(effectiveCalls)
                )
            )
            sendJsonResponse(connection, status: 200, object: resp)
        }
    }
}
