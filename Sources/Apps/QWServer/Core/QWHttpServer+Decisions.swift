// QWHttpServer+Decisions.swift — HTTP handlers for /v1/decisions, /v1/systemone, and /v1/qwegn.

import Foundation
import Network
import QWCore
import QWDecision
import QWEngine

extension QWHttpServer {
    private func resolveRequestDefaults(forceAlwaysEmitBox: Bool) -> QWDecisionRequestDefaults {
        let cfg = actor.getConfig()
        let readout = QWScoreReadout(rawValue: cfg.decisionDefaultScoreReadout.lowercased()) ?? .mean
        let truncation = QWTruncationPolicy(rawValue: cfg.decisionDefaultTruncation.lowercased())
        return QWDecisionRequestDefaults(
            scoreReadout: readout,
            truncation: truncation,
            alwaysEmitBox: forceAlwaysEmitBox ? true : cfg.decisionAlwaysEmitBox
        )
    }

    private func decodeHTTPRequestPayload(
        headers: [String: String],
        body: Data,
        forceAlwaysEmitBox: Bool
    ) throws -> QWDecisionPreparedHTTPInput {
        let defaults = resolveRequestDefaults(forceAlwaysEmitBox: forceAlwaysEmitBox)
        if QWHttpSecurity.isMultipartFormDataContentType(headers) {
            let multi = try QWDecisionMultipartParser.parse(headers: headers, body: body)
            let httpReq = try QWServerRequestDecoder.decode(QWDecisionHTTPRequest.self, from: multi.jsonBody)
            return try httpReq.prepareExecution(defaults: defaults, multipartParts: multi.namedParts)
        }
        let httpReq = try QWServerRequestDecoder.decode(QWDecisionHTTPRequest.self, from: body)
        return try httpReq.prepareExecution(defaults: defaults, multipartParts: [:])
    }

    private func decodeHTTPBatchPayloads(
        headers: [String: String],
        body: Data,
        forceAlwaysEmitBox: Bool
    ) throws -> [QWDecisionPreparedHTTPInput] {
        let defaults = resolveRequestDefaults(forceAlwaysEmitBox: forceAlwaysEmitBox)
        let batchReq: QWDecisionBatchHTTPRequest
        let parts: [String: Data]
        if QWHttpSecurity.isMultipartFormDataContentType(headers) {
            let multi = try QWDecisionMultipartParser.parse(headers: headers, body: body)
            batchReq = try QWServerRequestDecoder.decode(QWDecisionBatchHTTPRequest.self, from: multi.jsonBody)
            parts = multi.namedParts
        } else {
            batchReq = try QWServerRequestDecoder.decode(QWDecisionBatchHTTPRequest.self, from: body)
            parts = [:]
        }
        let items = batchReq.resolvedItems
        guard !items.isEmpty else {
            throw QWDecisionError.invalidQuestion("batch request requires non-empty 'items'")
        }
        return try items.map { try $0.prepareExecution(defaults: defaults, multipartParts: parts) }
    }

    func handleDecisionRoute(
        connection: NWConnection,
        headers: [String: String],
        body: Data,
        trace: QWRequestTraceContext? = nil
    ) async {
        let effectiveTrace = trace ?? QWClientIdentity.makeTraceContext(
            method: "POST", route: "/v1/decisions", headers: headers, peerEndpoint: connection.endpoint
        )
        let startedAt = Date()
        do {
            let prepared = try decodeHTTPRequestPayload(headers: headers, body: body, forceAlwaysEmitBox: false)
            let envelope = try await actor.decide(prepared.request, preferredModel: prepared.preferredModel)
            if prepared.isOpenAIDialect {
                sendJsonResponse(
                    connection, status: 200,
                    object: QWOpenAIDecisionHTTPResponse(envelope: envelope, meta: prepared.questionMeta)
                )
            } else {
                sendJsonResponse(connection, status: 200, object: QWDecisionHTTPResponse(envelope: envelope))
            }
            logger.logSimpleRequest(
                method: "POST", path: effectiveTrace.route, statusCode: 200,
                client: effectiveTrace.client, requestId: effectiveTrace.requestId,
                sessionId: effectiveTrace.sessionId, peerHost: effectiveTrace.peerHost,
                detail: "\(envelope.modelName) (\(envelope.backendKind))",
                totalSec: Date().timeIntervalSince(startedAt)
            )
        } catch {
            sendDecisionError(connection, error: error, trace: effectiveTrace)
        }
    }

    func handleDecisionBatchRoute(
        connection: NWConnection,
        headers: [String: String],
        body: Data,
        trace: QWRequestTraceContext? = nil
    ) async {
        let effectiveTrace = trace ?? QWClientIdentity.makeTraceContext(
            method: "POST", route: "/v1/decisions/batch", headers: headers, peerEndpoint: connection.endpoint
        )
        let startedAt = Date()
        do {
            let preparedItems = try decodeHTTPBatchPayloads(headers: headers, body: body, forceAlwaysEmitBox: false)
            let envelopes = try await executeCoalescedBatch(preparedItems)
            let results = envelopes.enumerated().map { idx, env -> QWDecisionHTTPResponse in
                _ = idx
                return QWDecisionHTTPResponse(envelope: env)
            }
            sendJsonResponse(connection, status: 200, object: QWDecisionBatchHTTPResponse(results: results))
            logger.logSimpleRequest(
                method: "POST", path: effectiveTrace.route, statusCode: 200,
                client: effectiveTrace.client, requestId: effectiveTrace.requestId,
                sessionId: effectiveTrace.sessionId, peerHost: effectiveTrace.peerHost,
                detail: "batch=\(envelopes.count)",
                totalSec: Date().timeIntervalSince(startedAt)
            )
        } catch {
            sendDecisionError(connection, error: error, trace: effectiveTrace)
        }
    }

    func handleSystemOneRoute(
        connection: NWConnection,
        headers: [String: String],
        body: Data,
        trace: QWRequestTraceContext? = nil
    ) async {
        let effectiveTrace = trace ?? QWClientIdentity.makeTraceContext(
            method: "POST", route: "/v1/systemone", headers: headers, peerEndpoint: connection.endpoint
        )
        let startedAt = Date()
        do {
            let prepared = try decodeHTTPRequestPayload(headers: headers, body: body, forceAlwaysEmitBox: true)
            let envelope = try await actor.decide(prepared.request, preferredModel: prepared.preferredModel)
            sendJsonResponse(
                connection, status: 200,
                object: QWSystemOneHTTPResponse(envelope: envelope, meta: prepared.questionMeta)
            )
            logger.logSimpleRequest(
                method: "POST", path: effectiveTrace.route, statusCode: 200,
                client: effectiveTrace.client, requestId: effectiveTrace.requestId,
                sessionId: effectiveTrace.sessionId, peerHost: effectiveTrace.peerHost,
                detail: "\(envelope.modelName) (\(envelope.backendKind))",
                totalSec: Date().timeIntervalSince(startedAt)
            )
        } catch {
            sendDecisionError(connection, error: error, trace: effectiveTrace)
        }
    }

    func handleSystemOneBatchRoute(
        connection: NWConnection,
        headers: [String: String],
        body: Data,
        trace: QWRequestTraceContext? = nil
    ) async {
        let effectiveTrace = trace ?? QWClientIdentity.makeTraceContext(
            method: "POST", route: "/v1/systemone/batch", headers: headers, peerEndpoint: connection.endpoint
        )
        let startedAt = Date()
        do {
            let preparedItems = try decodeHTTPBatchPayloads(headers: headers, body: body, forceAlwaysEmitBox: true)
            let envelopes = try await executeCoalescedBatch(preparedItems)
            let results = zip(envelopes, preparedItems).map { env, prep in
                QWSystemOneHTTPResponse(envelope: env, meta: prep.questionMeta)
            }
            sendJsonResponse(connection, status: 200, object: QWSystemOneBatchHTTPResponse(results: results))
            logger.logSimpleRequest(
                method: "POST", path: effectiveTrace.route, statusCode: 200,
                client: effectiveTrace.client, requestId: effectiveTrace.requestId,
                sessionId: effectiveTrace.sessionId, peerHost: effectiveTrace.peerHost,
                detail: "batch=\(envelopes.count)",
                totalSec: Date().timeIntervalSince(startedAt)
            )
        } catch {
            sendDecisionError(connection, error: error, trace: effectiveTrace)
        }
    }

    private func executeCoalescedBatch(
        _ preparedItems: [QWDecisionPreparedHTTPInput]
    ) async throws -> [QWDecisionExecutionEnvelope] {
        guard !preparedItems.isEmpty else { return [] }
        let distinctModels = Set(preparedItems.map { ($0.preferredModel ?? "").trimmingCharacters(in: .whitespacesAndNewlines) })
        if distinctModels.count <= 1 {
            return try await actor.decideBatch(
                preparedItems.map(\.request),
                preferredModel: preparedItems[0].preferredModel
            )
        }
        var out: [QWDecisionExecutionEnvelope] = []
        out.reserveCapacity(preparedItems.count)
        var start = 0
        while start < preparedItems.count {
            let modelKey = preparedItems[start].preferredModel
            var end = start + 1
            while end < preparedItems.count && preparedItems[end].preferredModel == modelKey {
                end += 1
            }
            let sliceReqs = preparedItems[start..<end].map(\.request)
            out.append(contentsOf: try await actor.decideBatch(sliceReqs, preferredModel: modelKey))
            start = end
        }
        return out
    }

    func handleQwegnAskRoute(
        connection: NWConnection,
        headers: [String: String] = [:],
        body: Data,
        trace: QWRequestTraceContext? = nil
    ) async {
        let effectiveTrace = trace ?? QWClientIdentity.makeTraceContext(
            method: "POST", route: "/v1/qwegn", headers: headers, peerEndpoint: connection.endpoint
        )
        let startedAt = Date()
        do {
            let askReq = try QWServerRequestDecoder.decode(QWQwegnAskHTTPRequest.self, from: body)
            let domainReq = try askReq.toDecisionRequest()
            let envelope = try await actor.decide(domainReq, preferredModel: askReq.model)
            sendJsonResponse(connection, status: 200, object: QWQwegnAskHTTPResponse(envelope: envelope))
            logger.logSimpleRequest(
                method: "POST", path: effectiveTrace.route, statusCode: 200,
                client: effectiveTrace.client, requestId: effectiveTrace.requestId,
                sessionId: effectiveTrace.sessionId, peerHost: effectiveTrace.peerHost,
                detail: "\(envelope.modelName) (\(envelope.backendKind))",
                totalSec: Date().timeIntervalSince(startedAt)
            )
        } catch {
            sendDecisionError(connection, error: error, trace: effectiveTrace)
        }
    }

    func handleDecisionLoadRoute(connection: NWConnection, body: Data) async {
        do {
            let req = try QWServerRequestDecoder.decode(QWDecisionLoadHTTPRequest.self, from: body)
            let rawRef = req.packPath ?? req.model ?? ""
            guard !rawRef.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw QWDecisionError.invalidQuestion("decision model load requires 'pack_path' or 'model'")
            }
            let allowUnregistered = ProcessInfo.processInfo.environment["QWG_ALLOW_UNREGISTERED_MODEL_PATHS"] == "1"
            let settings = QWSettingsManager.shared.loadSettings()
            let ref = settings.resolveModelReference(rawRef, allowArbitraryFilesystemPaths: true)
            let resolvedPath = ref.packPath.isEmpty ? rawRef.expandingTildePath : ref.packPath
            if !allowUnregistered {
                let candURL = URL(fileURLWithPath: resolvedPath)
                let inRegistered = QWHttpSecurity.canonicalDirectoryContainment(candURL, inside: QWPaths.modelsDirectoryURL)
                    || QWHttpSecurity.canonicalDirectoryContainment(candURL, inside: QWPaths.hfHubCacheDirectoryURL)
                guard inRegistered else {
                    throw QWMetadataValidationError("Decision model pack must reside under ~/.qwg/models or HuggingFace cache")
                }
            }
            let unit = QWDecisionComputeUnit(rawValue: (req.computeUnit ?? settings.decisionComputeUnit).lowercased()) ?? .auto
            let loadVision = req.loadVision ?? settings.decisionLoadVision
            let status = try await actor.loadDecisionSidecar(
                packPath: resolvedPath,
                computeUnit: unit,
                loadVision: loadVision
            )
            sendJsonResponse(connection, status: 200, object: status)
        } catch {
            sendDecisionError(connection, error: error)
        }
    }

    func handleDecisionUnloadRoute(connection: NWConnection) async {
        let status = await actor.unloadDecisionSidecar()
        sendJsonResponse(connection, status: 200, object: status)
    }

    func handleDecisionStatusRoute(connection: NWConnection) {
        let status = actor.decisionSidecarStatus()
        sendJsonResponse(connection, status: 200, object: status)
    }

    func sendDecisionError(
        _ connection: NWConnection,
        error: any Error,
        trace: QWRequestTraceContext? = nil
    ) {
        let route = trace?.route ?? "/v1/decisions"
        let tag = trace.map { " | \($0.logTag)" } ?? ""
        if let decErr = error as? QWDecisionError {
            switch decErr {
            case .unsupportedLabelSubset(_, let allowed):
                logger.log(.warn, "HTTP POST \(route) invalid labels\(tag): \(decErr.description)")
                let payload = QWDecisionErrorPayload(
                    message: decErr.description,
                    type: "invalid_request_error",
                    code: 400,
                    allowedLabels: allowed
                )
                sendJsonResponse(connection, status: 400, object: payload)
            case .invalidQuestion, .contextOverflow, .missingSingleTokenAlias, .calibrationFailed:
                logger.log(.warn, "HTTP POST \(route) invalid request\(tag): \(decErr.description)")
                let payload = QWDecisionErrorPayload(
                    message: decErr.description,
                    type: "invalid_request_error",
                    code: 400
                )
                sendJsonResponse(connection, status: 400, object: payload)
            case .backendFailed(let msg):
                let isUnavailable = msg.localizedCaseInsensitiveContains("no decision sidecar")
                    || msg.localizedCaseInsensitiveContains("no embedding-capable")
                let status = isUnavailable ? 503 : 500
                logger.log(isUnavailable ? .warn : .error, "HTTP POST \(route) backend error\(tag): \(decErr.description)")
                let payload = QWDecisionErrorPayload(
                    message: decErr.description,
                    type: isUnavailable ? "service_unavailable" : "server_error",
                    code: status
                )
                sendJsonResponse(connection, status: status, object: payload)
            }
            return
        }
        if let metaErr = error as? QWMetadataValidationError {
            logger.log(.warn, "HTTP POST \(route) forbidden\(tag): \(metaErr.description)")
            let payload = QWDecisionErrorPayload(message: metaErr.description, type: "invalid_request_error", code: 403)
            sendJsonResponse(connection, status: 403, object: payload)
            return
        }
        logger.log(.warn, "HTTP POST \(route) malformed payload\(tag): \(error.localizedDescription)")
        let payload = QWDecisionErrorPayload(
            message: "Malformed decision payload: \(error.localizedDescription)",
            type: "invalid_request_error",
            code: 400
        )
        sendJsonResponse(connection, status: 400, object: payload)
    }
}
