// QWHttpServer+Cognitive.swift — HTTP routes for /v1/cognitive/* cognitive rails & episodic store (G3).

import Darwin
import Foundation
import Network
import QWCore
import QWDecision

private struct QWCognitiveGuardHTTPRequest: Codable {
    let command: String
    let actingRail: Double?
    let extraRedLines: [String]?

    enum CodingKeys: String, CodingKey {
        case command
        case actingRail = "acting_rail"
        case extraRedLines = "extra_red_lines"
    }
}

private struct QWCognitiveRerankScreenHTTPRequest: Codable {
    let query: String
    let passages: [QWRerankPassageInput]
    let relevanceThreshold: Double?
    let injectionThreshold: Double?
    let topK: Int?

    enum CodingKeys: String, CodingKey {
        case query, passages
        case relevanceThreshold = "relevance_threshold"
        case injectionThreshold = "injection_threshold"
        case topK = "top_k"
    }
}

private struct QWCognitiveCompactHTTPRequest: Codable {
    let turns: [QWCompactionTurnInput]
    let keepTailCount: Int?
    let dropConfidenceRail: Double?
    let archiveSessionId: String?

    enum CodingKeys: String, CodingKey {
        case turns
        case keepTailCount = "keep_tail_count"
        case dropConfidenceRail = "drop_confidence_rail"
        case archiveSessionId = "archive_session_id"
    }
}

private struct QWCognitiveCompactHTTPResponse: Codable {
    let decisions: [QWCompactionDecision]
    let fallbackUsed: Bool
    let archivedCount: Int
    let latencyMs: Double

    enum CodingKeys: String, CodingKey {
        case decisions
        case fallbackUsed = "fallback_used"
        case archivedCount = "archived_count"
        case latencyMs = "latency_ms"
    }
}

private struct QWEpisodicRecoverHTTPRequest: Codable {
    let sessionId: String
    let query: String
    let limit: Int?

    enum CodingKeys: String, CodingKey {
        case query, limit
        case sessionId = "session_id"
    }
}

private struct QWEpisodicRecoverHTTPResponse: Codable {
    let sessionId: String
    let turns: [QWEpisodicRecoveredTurn]

    enum CodingKeys: String, CodingKey {
        case turns
        case sessionId = "session_id"
    }
}

private struct QWCognitiveSkillsHTTPRequest: Codable {
    let userTurn: String
    let catalog: [QWSkillDescriptor]
    let shortlistThreshold: Double?
    let verifyThreshold: Double?
    let maxSelected: Int?

    enum CodingKeys: String, CodingKey {
        case catalog
        case userTurn = "user_turn"
        case shortlistThreshold = "shortlist_threshold"
        case verifyThreshold = "verify_threshold"
        case maxSelected = "max_selected"
    }
}

private struct QWCognitiveSearchHTTPRequest: Codable {
    let query: String
    let passages: [String]
    let candidateQueries: [String]?
    let priorQueries: [String]?
    let confidenceRail: Double?

    enum CodingKeys: String, CodingKey {
        case query, passages
        case candidateQueries = "candidate_queries"
        case priorQueries = "prior_queries"
        case confidenceRail = "confidence_rail"
    }
}

extension QWHttpServer {
    func makeCognitiveRails() -> QWCognitiveRails {
        guard actor.isDecisionSidecarLoaded() else {
            return QWCognitiveRails(runner: nil)
        }
        let act = self.actor
        return QWCognitiveRails(decideFn: { req in
            try await act.decide(req).response
        })
    }

    func handleCognitiveRoute(connection: NWConnection, path: String, body: Data) async {
        do {
            switch path {
            case "/v1/cognitive/guard":
                let req = try QWServerRequestDecoder.decode(QWCognitiveGuardHTTPRequest.self, from: body)
                let res = await makeCognitiveRails().guardCommand(
                    req.command,
                    actingRail: req.actingRail ?? 0.70,
                    extraRedLines: req.extraRedLines ?? []
                )
                sendJsonResponse(connection, status: 200, object: res)

            case "/v1/cognitive/rerank-screen":
                let req = try QWServerRequestDecoder.decode(QWCognitiveRerankScreenHTTPRequest.self, from: body)
                let res = await makeCognitiveRails().screenAndRerankPassages(
                    query: req.query,
                    passages: req.passages,
                    relevanceThreshold: req.relevanceThreshold ?? 0.50,
                    injectionThreshold: req.injectionThreshold ?? 0.50,
                    topK: req.topK ?? 8
                )
                sendJsonResponse(connection, status: 200, object: res)

            case "/v1/cognitive/compact":
                let req = try QWServerRequestDecoder.decode(QWCognitiveCompactHTTPRequest.self, from: body)
                let res = await makeCognitiveRails().selectCompaction(
                    turns: req.turns,
                    keepTailCount: req.keepTailCount ?? 6,
                    dropConfidenceRail: req.dropConfidenceRail ?? 0.70
                )
                var archivedCount = 0
                if let sid = req.archiveSessionId, !sid.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    let dropOrSummarizeIndices = Set(
                        res.decisions.filter { $0.action == .drop || $0.action == .summarize }.map(\.index)
                    )
                    let toArchive = req.turns
                        .filter { dropOrSummarizeIndices.contains($0.index) }
                        .map { QWEpisodicTurn(index: $0.index, role: $0.role, text: $0.text) }
                    archivedCount = try episodicStore.archive(sessionId: sid, turns: toArchive)
                }
                sendJsonResponse(connection, status: 200, object: QWCognitiveCompactHTTPResponse(
                    decisions: res.decisions,
                    fallbackUsed: res.fallbackUsed,
                    archivedCount: archivedCount,
                    latencyMs: res.latencyMs
                ))

            case "/v1/cognitive/episodic/recover":
                let req = try QWServerRequestDecoder.decode(QWEpisodicRecoverHTTPRequest.self, from: body)
                let hits = try episodicStore.recover(sessionId: req.sessionId, query: req.query, limit: req.limit ?? 5)
                sendJsonResponse(connection, status: 200, object: QWEpisodicRecoverHTTPResponse(
                    sessionId: req.sessionId,
                    turns: hits
                ))

            case "/v1/cognitive/skills":
                let req = try QWServerRequestDecoder.decode(QWCognitiveSkillsHTTPRequest.self, from: body)
                let res = await makeCognitiveRails().selectSkills(
                    userTurn: req.userTurn,
                    catalog: req.catalog,
                    shortlistThreshold: req.shortlistThreshold ?? 0.02,
                    verifyThreshold: req.verifyThreshold ?? 0.50,
                    maxSelected: req.maxSelected ?? 3
                )
                sendJsonResponse(connection, status: 200, object: res)

            case "/v1/cognitive/search":
                let req = try QWServerRequestDecoder.decode(QWCognitiveSearchHTTPRequest.self, from: body)
                let res = await makeCognitiveRails().evaluateSearchSufficiency(
                    query: req.query,
                    passages: req.passages,
                    candidateQueries: req.candidateQueries ?? [],
                    priorQueries: req.priorQueries ?? [],
                    confidenceRail: req.confidenceRail ?? 0.70
                )
                sendJsonResponse(connection, status: 200, object: res)

            case "/v1/cognitive/supervise":
                let req = try QWServerRequestDecoder.decode(QWSuperviseInput.self, from: body)
                let res = await makeCognitiveRails().superviseDelegatedRun(req)
                sendJsonResponse(connection, status: 200, object: res)

            case "/v1/cognitive/route":
                let req = try QWServerRequestDecoder.decode(QWRouteTurnInput.self, from: body)
                let res = await makeCognitiveRails().routeTurn(req)
                if res.mode == .shadow {
                    Self.appendShadowRouteLog(input: req, result: res)
                }
                sendJsonResponse(connection, status: 200, object: res)

            default:
                let err = ServerErrorPayload(message: "Not found: POST \(path)", type: "invalid_request_error", code: 404)
                sendJsonResponse(connection, status: 404, object: err)
            }
        } catch let epiErr as QWEpisodicStoreError {
            let err = ServerErrorPayload(message: epiErr.description, type: "invalid_request_error", code: 400)
            sendJsonResponse(connection, status: 400, object: err)
        } catch {
            let err = ServerErrorPayload(message: "Malformed cognitive payload: \(error.localizedDescription)", type: "invalid_request_error", code: 400)
            sendJsonResponse(connection, status: 400, object: err)
        }
    }

    private static func appendShadowRouteLog(input: QWRouteTurnInput, result: QWRouteTurnResult) {
        let logDir = QWPaths.rootDirectoryURL.appendingPathComponent("logs", isDirectory: true)
        let fm = FileManager.default
        try? fm.createDirectory(at: logDir, withIntermediateDirectories: true, attributes: [.posixPermissions: NSNumber(value: Int16(0o700))])
        let file = logDir.appendingPathComponent("decision-routes.jsonl", isDirectory: false)
        let record: [String: Any] = [
            "timestamp_epoch_ms": Int64(Date().timeIntervalSince1970 * 1000),
            "context_tokens": input.contextTokens,
            "current_tier": input.currentTier,
            "recommended_tier": result.recommendedTier,
            "would_switch": result.wouldSwitch,
            "difficulty": result.difficulty,
            "domain": result.domain,
            "costly_risk": result.costlyRisk,
            "confidence": result.confidence,
            "sticky_guard_applied": result.stickyGuardApplied,
        ]
        guard let lineData = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]) else { return }
        var out = lineData
        out.append(0x0A)
        if fm.fileExists(atPath: file.path), let h = try? FileHandle(forWritingTo: file) {
            defer { try? h.close() }
            _ = try? h.seekToEnd()
            try? h.write(contentsOf: out)
        } else {
            try? out.write(to: file, options: .atomic)
        }
        chmod(file.path, mode_t(0o600))
    }
}
