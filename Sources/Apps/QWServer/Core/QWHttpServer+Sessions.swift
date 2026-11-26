// QWHttpServer+Sessions.swift — HTTP routes for /v1/sessions/* QWSessionTree endpoints (G4).

import Foundation
import Network
import QWCore
import QWEngine

private struct QWSessionCheckpointHTTPRequest: Codable {
    let id: String?
    let parentId: String?
    let label: String?
    let modelName: String?
    let kvMode: String?
    let cacheSalt: String?
    let tokenIds: [Int]
    let recurrentCheckpointId: String?
    let metadata: [String: String]?

    enum CodingKeys: String, CodingKey {
        case id, label, metadata
        case parentId = "parent_id"
        case modelName = "model_name"
        case kvMode = "kv_mode"
        case cacheSalt = "cache_salt"
        case tokenIds = "token_ids"
        case recurrentCheckpointId = "recurrent_checkpoint_id"
    }
}

private struct QWSessionForkHTTPRequest: Codable {
    let parentId: String
    let count: Int
    let labels: [String]?
    let metadata: [String: String]?

    enum CodingKeys: String, CodingKey {
        case count, labels, metadata
        case parentId = "parent_id"
    }
}

private struct QWSessionRestoreHTTPRequest: Codable {
    let nodeId: String

    enum CodingKeys: String, CodingKey {
        case nodeId = "node_id"
    }
}

private struct QWSessionPruneHTTPRequest: Codable {
    let nodeId: String
    let cascade: Bool?

    enum CodingKeys: String, CodingKey {
        case cascade
        case nodeId = "node_id"
    }
}

private struct QWSessionListHTTPResponse: Codable {
    let object: String
    let data: [QWSessionNode]
    let stats: QWSessionTreeStats
}

private struct QWSessionDetailHTTPResponse: Codable {
    let node: QWSessionNode
    let lineage: [QWSessionNode]
}

private struct QWSessionForkHTTPResponse: Codable {
    let parentId: String
    let children: [QWSessionNode]

    enum CodingKeys: String, CodingKey {
        case children
        case parentId = "parent_id"
    }
}

private struct QWSessionPruneHTTPResponse: Codable {
    let removedIds: [String]
    let stats: QWSessionTreeStats

    enum CodingKeys: String, CodingKey {
        case stats
        case removedIds = "removed_ids"
    }
}

extension QWHttpServer {
    func handleSessionsGetRoute(connection: NWConnection, path: String) {
        if path == "/v1/sessions" {
            let list = sessionTree.listNodes()
            sendJsonResponse(
                connection,
                status: 200,
                object: QWSessionListHTTPResponse(object: "list", data: list, stats: sessionTree.stats())
            )
            return
        }
        let prefix = "/v1/sessions/"
        guard path.hasPrefix(prefix) else {
            let err = ServerErrorPayload(message: "Not found: GET \(path)", type: "invalid_request_error", code: 404)
            sendJsonResponse(connection, status: 404, object: err)
            return
        }
        let id = String(path.dropFirst(prefix.count))
        do {
            guard let node = sessionTree.node(id: id) else {
                let err = ServerErrorPayload(message: "Session node '\(id)' not found", type: "invalid_request_error", code: 404)
                sendJsonResponse(connection, status: 404, object: err)
                return
            }
            let lineage = try sessionTree.lineage(of: id)
            sendJsonResponse(connection, status: 200, object: QWSessionDetailHTTPResponse(node: node, lineage: lineage))
        } catch {
            let err = ServerErrorPayload(message: "\(error)", type: "invalid_request_error", code: 404)
            sendJsonResponse(connection, status: 404, object: err)
        }
    }

    func handleSessionsPostRoute(connection: NWConnection, path: String, body: Data) async {
        do {
            switch path {
            case "/v1/sessions/checkpoint":
                let req = try QWServerRequestDecoder.decode(QWSessionCheckpointHTTPRequest.self, from: body)
                let cfg = actor.getConfig()
                let node = try sessionTree.checkpoint(
                    parentId: req.parentId,
                    label: req.label,
                    modelName: req.modelName ?? cfg.modelName,
                    kvMode: req.kvMode ?? String(describing: cfg.kvMode),
                    cacheSalt: req.cacheSalt,
                    tokenIds: req.tokenIds,
                    recurrentCheckpointId: req.recurrentCheckpointId,
                    metadata: req.metadata ?? [:],
                    id: req.id
                )
                sendJsonResponse(connection, status: 200, object: node)

            case "/v1/sessions/fork":
                let req = try QWServerRequestDecoder.decode(QWSessionForkHTTPRequest.self, from: body)
                let children = try sessionTree.fork(
                    from: req.parentId,
                    count: req.count,
                    labels: req.labels,
                    metadata: req.metadata ?? [:]
                )
                sendJsonResponse(connection, status: 200, object: QWSessionForkHTTPResponse(parentId: req.parentId, children: children))

            case "/v1/sessions/restore":
                let req = try QWServerRequestDecoder.decode(QWSessionRestoreHTTPRequest.self, from: body)
                let targetNode = sessionTree.node(id: req.nodeId)
                let liveHit = if let targetNode, !targetNode.tokenIds.isEmpty {
                    await actor.hasCachedPrefix(tokens: targetNode.tokenIds, cacheSalt: targetNode.cacheSalt)
                } else {
                    false
                }
                let result = try sessionTree.restore(nodeId: req.nodeId) { _, _ in liveHit }
                sendJsonResponse(connection, status: 200, object: result)

            case "/v1/sessions/prune":
                let req = try QWServerRequestDecoder.decode(QWSessionPruneHTTPRequest.self, from: body)
                let removed = try sessionTree.prune(nodeId: req.nodeId, cascade: req.cascade ?? false)
                sendJsonResponse(connection, status: 200, object: QWSessionPruneHTTPResponse(
                    removedIds: removed, stats: sessionTree.stats()
                ))

            default:
                let err = ServerErrorPayload(message: "Not found: POST \(path)", type: "invalid_request_error", code: 404)
                sendJsonResponse(connection, status: 404, object: err)
            }
        } catch let treeErr as QWSessionTreeError {
            let status: Int = if case .nodeNotFound = treeErr { 404 } else { 400 }
            let err = ServerErrorPayload(message: treeErr.description, type: "invalid_request_error", code: status)
            sendJsonResponse(connection, status: status, object: err)
        } catch {
            let err = ServerErrorPayload(message: "Malformed session request: \(error.localizedDescription)", type: "invalid_request_error", code: 400)
            sendJsonResponse(connection, status: 400, object: err)
        }
    }
}
