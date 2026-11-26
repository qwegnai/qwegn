// QWHttpServer+MCP.swift — Stateless JSON-RPC 2.0 MCP server shared by HTTP POST /mcp and stdio CLI.

import Foundation
import Network
import QWCore
import QWDecision
import QWToken

/// Outcome of dispatching a single JSON-RPC 2.0 MCP message.
public enum QWMCPDispatchOutcome: Sendable, Equatable {
    case notificationAccepted
    case jsonRPCResponse(Data)
}

/// Shared MCP JSON-RPC 2.0 dispatcher serving `qwg_decide`, `laya_ask`, `laya_guard`, and `laya_status`.
public struct QWMCPDispatcher: Sendable {
    public let actor: QWEngineLifecycleActor
    public let episodicStore: QWEpisodicTranscriptStore

    public init(
        actor: QWEngineLifecycleActor,
        episodicStore: QWEpisodicTranscriptStore = QWEpisodicTranscriptStore()
    ) {
        self.actor = actor
        self.episodicStore = episodicStore
    }

    public func dispatch(body: Data) async -> QWMCPDispatchOutcome {
        guard let root = try? QWServerRequestDecoder.decode(QWJson.self, from: body),
              case .object = root else {
            return .jsonRPCResponse(Self.encodeError(id: .null, code: -32700, message: "Parse error: invalid JSON"))
        }
        let rpcId = root.get("id")
        let isNotification = rpcId.isUndefined
        guard case .string(let method) = root.get("method") else {
            return .jsonRPCResponse(Self.encodeError(
                id: isNotification ? .null : rpcId,
                code: -32600,
                message: "Invalid Request: missing method"
            ))
        }
        let params = root.get("params")

        if method == "notifications/initialized" || isNotification && method.hasPrefix("notifications/") {
            return .notificationAccepted
        }

        switch method {
        case "initialize":
            let result: QWJson = .object([
                ("protocolVersion", .string("2025-03-26")),
                ("capabilities", .object([("tools", .object([]))])),
                ("serverInfo", .object([
                    ("name", .string("qwegine")),
                    ("version", .string(QWVersion.current)),
                ])),
            ])
            return .jsonRPCResponse(Self.encodeResult(id: rpcId, result: result))

        case "ping":
            return .jsonRPCResponse(Self.encodeResult(id: rpcId, result: .object([])))

        case "tools/list":
            let result: QWJson = .object([("tools", .array(Self.toolDefinitions()))])
            return .jsonRPCResponse(Self.encodeResult(id: rpcId, result: result))

        case "tools/call":
            guard case .string(let toolName) = params.get("name"), !toolName.isEmpty else {
                return .jsonRPCResponse(Self.encodeError(id: rpcId, code: -32602, message: "Invalid params: missing tool name"))
            }
            let args = params.get("arguments").isUndefined ? QWJson.object([]) : params.get("arguments")
            let callResult = await executeTool(name: toolName, arguments: args)
            return .jsonRPCResponse(Self.encodeResult(id: rpcId, result: callResult))

        default:
            return .jsonRPCResponse(Self.encodeError(id: rpcId, code: -32601, message: "Method not found: \(method)"))
        }
    }

    private func executeTool(name: String, arguments: QWJson) async -> QWJson {
        do {
            let textPayload: String
            switch name {
            case "qwg_decide":
                let req = try QWServerRequestDecoder.decode(QWDecisionHTTPRequest.self, from: Data(arguments.tojson.utf8))
                let envelope = try await actor.decide(try req.toDecisionRequest(), preferredModel: req.model)
                textPayload = try Self.encodeToString(QWDecisionHTTPResponse(envelope: envelope))

            case "laya_ask":
                let req = try QWServerRequestDecoder.decode(QWLayaHTTPRequest.self, from: Data(arguments.tojson.utf8))
                let envelope = try await actor.decide(try req.toDecisionRequest(), preferredModel: req.model)
                textPayload = try Self.encodeToString(QWLayaHTTPResponse(envelope: envelope))

            case "laya_guard":
                let cmd: String = {
                    if case .string(let s) = arguments.get("command") { return s }
                    if case .string(let s) = arguments.get("state") { return s }
                    return arguments.get("context").stringified
                }()
                let trimmedCmd = cmd.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmedCmd.isEmpty else {
                    return Self.toolResponse(text: "{\"error\":\"missing command argument\"}", isError: true)
                }
                let railResult = await makeCognitiveRails().guardCommand(trimmedCmd)
                textPayload = try Self.encodeToString(railResult)

            case "laya_status":
                let sidecar = actor.decisionSidecarStatus()
                let primaryState = actor.engineState()
                let doc: [String: Any] = [
                    "sidecar_loaded": sidecar.loaded,
                    "sidecar": (try? JSONSerialization.jsonObject(with: JSONEncoder().encode(sidecar))) ?? [:],
                    "primary_loaded": primaryState.isLoaded,
                    "primary_model": primaryState.loadedModelName as Any,
                    "primary_phase": primaryState.lifecycle.phase.rawValue,
                ]
                let data = try JSONSerialization.data(withJSONObject: doc, options: [.sortedKeys])
                textPayload = String(decoding: data, as: UTF8.self)

            default:
                if let cogOut = try await executeCognitiveOrVaultTool(name: name, arguments: arguments) {
                    textPayload = cogOut
                } else {
                    return Self.toolResponse(text: "{\"error\":\"unknown tool '\(name)'\"}", isError: true)
                }
            }
            return Self.toolResponse(text: textPayload, isError: false)
        } catch let decErr as QWDecisionError {
            var errObj: [String: Any] = ["error": decErr.description]
            if case .unsupportedLabelSubset(_, let allowed) = decErr {
                errObj["allowed_labels"] = allowed
            }
            let data = (try? JSONSerialization.data(withJSONObject: errObj, options: [.sortedKeys])) ?? Data("{\"error\":\"decision failed\"}".utf8)
            return Self.toolResponse(text: String(decoding: data, as: UTF8.self), isError: true)
        } catch {
            let msg = error.localizedDescription
            let data = (try? JSONSerialization.data(withJSONObject: ["error": msg], options: [.sortedKeys]))
                ?? Data("{\"error\":\"internal error\"}".utf8)
            return Self.toolResponse(text: String(decoding: data, as: UTF8.self), isError: true)
        }
    }

    private static func toolResponse(text: String, isError: Bool) -> QWJson {
        .object([
            ("content", .array([
                .object([("type", .string("text")), ("text", .string(text))]),
            ])),
            ("isError", .bool(isError)),
        ])
    }

    private static func encodeToString<T: Encodable>(_ value: T) throws -> String {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        return String(decoding: try enc.encode(value), as: UTF8.self)
    }

    private static func encodeResult(id: QWJson, result: QWJson) -> Data {
        let resolvedId: QWJson = id.isUndefined ? .null : id
        let envelope: QWJson = .object([
            ("jsonrpc", .string("2.0")),
            ("id", resolvedId),
            ("result", result),
        ])
        return Data((envelope.tojson + "\n").utf8)
    }

    private static func encodeError(id: QWJson, code: Int, message: String) -> Data {
        let envelope: QWJson = .object([
            ("jsonrpc", .string("2.0")),
            ("id", id),
            ("error", .object([
                ("code", .int(code)),
                ("message", .string(message)),
            ])),
        ])
        return Data((envelope.tojson + "\n").utf8)
    }

    private static func toolSchema(_ name: String, _ desc: String, properties: [(String, QWJson)], required: [String]) -> QWJson {
        .object([
            ("name", .string(name)),
            ("description", .string(desc)),
            ("inputSchema", .object([
                ("type", .string("object")),
                ("properties", .object(properties)),
                ("required", .array(required.map { .string($0) })),
            ])),
        ])
    }

    private static func toolDefinitions() -> [QWJson] {
        [
            toolSchema(
                "qwg_decide",
                "Evaluate structured single-pass System-1 decision questions (binary/noul, choice, multiple, score, read, estimate, box) with calibrated probabilities.",
                properties: [
                    ("model", .object([("type", .string("string"))])),
                    ("state", .object([("type", .array([.string("string"), .string("object")]))])),
                    ("questions", .object([("type", .array([.string("array"), .string("object")]))])),
                    ("truncation", .object([("type", .string("string"))])),
                    ("score_readout", .object([("type", .string("string"))])),
                    ("chain", .object([("type", .string("string"))])),
                ],
                required: ["questions"]
            ),
            toolSchema(
                "laya_ask",
                "Evaluate a concise System-1 decision question over context up to 60,000 characters.",
                properties: [
                    ("context", .object([("type", .string("string"))])),
                    ("question", .object([("type", .string("string"))])),
                    ("kind", .object([("type", .string("string"))])),
                    ("options", .object([("type", .string("array")), ("items", .object([("type", .string("string"))]))])),
                    ("chain", .object([("type", .string("string"))])),
                ],
                required: ["question"]
            ),
            toolSchema(
                "laya_guard",
                "Classify shell/tool command safety into allow, confirm, or deny with calibrated confidence.",
                properties: [
                    ("command", .object([("type", .string("string"))])),
                    ("model", .object([("type", .string("string"))])),
                ],
                required: ["command"]
            ),
            toolSchema(
                "laya_status",
                "Report decision sidecar and primary LLM residency, compute unit, and latency metrics.",
                properties: [],
                required: []
            ),
        ] + cognitiveAndVaultToolDefinitions()
    }
}

extension QWHttpServer {
    func handleMCPRoute(
        connection: NWConnection,
        headers: [String: String] = [:],
        body: Data,
        trace: QWRequestTraceContext? = nil
    ) async {
        let startedAt = Date()
        let root = try? QWServerRequestDecoder.decode(QWJson.self, from: body)
        let rpcMethod: String? = if case .string(let m) = root?.get("method") { m } else { nil }
        var toolName: String? = nil
        var mcpClientName: String? = nil
        if rpcMethod == "tools/call",
           case .string(let name) = root?.get("params").get("name"), !name.isEmpty {
            toolName = name
        } else if rpcMethod == "initialize",
                  case .string(let name) = root?.get("params").get("clientInfo").get("name"), !name.isEmpty {
            mcpClientName = name
        }

        let baseTrace = trace ?? QWClientIdentity.makeTraceContext(
            method: "POST", route: "/mcp", headers: headers, peerEndpoint: connection.endpoint
        )
        let effectiveTrace: QWRequestTraceContext = if let mcpClientName,
                                                      QWClientIdentity.isGenericClient(baseTrace.client) {
            baseTrace.withClient(mcpClientName)
        } else {
            baseTrace
        }
        let detail = toolName.map { "tools/call(\($0))" } ?? rpcMethod

        let dispatcher = QWMCPDispatcher(actor: actor, episodicStore: episodicStore)
        let status: Int
        switch await dispatcher.dispatch(body: body) {
        case .notificationAccepted:
            status = 202
            sendRawData(connection, status: 202, contentType: "application/json", data: Data())
        case .jsonRPCResponse(let data):
            status = 200
            sendRawData(connection, status: 200, contentType: "application/json", data: data)
        }
        logger.logSimpleRequest(
            method: "POST", path: effectiveTrace.route, statusCode: status,
            client: effectiveTrace.client, requestId: effectiveTrace.requestId,
            sessionId: effectiveTrace.sessionId, peerHost: effectiveTrace.peerHost,
            detail: detail, totalSec: Date().timeIntervalSince(startedAt)
        )
    }
}
