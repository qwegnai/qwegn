// QWHttpServer+VaultV2.swift — Multi-pack Cognitive Vault v2 HTTP routes
// (`GET /v1/memory/packs`, `POST /v1/memory/mount`, `POST /v1/memory/unmount`, `POST /v1/memory/query`).

import Foundation
import Network
import QWCore
import QWDecision
import QWEngine

private struct QWMemoryQueryHTTPRequest: Decodable {
    let query: String
    let workspacePath: String?
    let packNames: [String]?
    let maxFiles: Int?
    let maxTokens: Int?
    let expandASTHops: Int?
    let screen: Bool?
    let includeCortexFiles: Bool?

    enum CodingKeys: String, CodingKey {
        case query, screen
        case workspacePath = "workspace_path"
        case packNames = "pack_names"
        case maxFiles = "max_files"
        case maxTokens = "max_tokens"
        case expandASTHops = "expand_ast_hops"
        case includeCortexFiles = "include_cortex_files"
    }
}

extension QWHttpServer {
    func handleMemoryPacksListRoute(connection: NWConnection) {
        struct PacksResponse: Encodable {
            let primary: QWEngineInfoResponse.MountedMemoryPackInfo?
            let packs: [QWMountedVaultPackSummary]
        }
        let payload = PacksResponse(
            primary: actor.getMountedMemoryPackInfo(),
            packs: actor.listMountedVaultPacks()
        )
        sendJsonResponse(connection, status: 200, object: payload)
    }

    func handleMemoryMountV2Route(connection: NWConnection, body: Data) async throws {
        guard !body.isEmpty,
              let payload = try? JSONSerialization.jsonObject(with: body) as? [String: Any] else {
            let err = ServerErrorPayload(message: "Missing JSON body in memory mount request.", code: 400)
            sendJsonResponse(connection, status: 400, object: err)
            return
        }
        let auxiliary = (payload["auxiliary"] as? Bool) ?? false
        let paths = (payload["paths"] as? [String]) ?? []
        let docsPaths = (payload["docs"] as? [String]) ?? []
        if !paths.isEmpty || !docsPaths.isEmpty || auxiliary {
            var mountedSummaries: [QWMountedVaultPackSummary] = []
            if let singlePath = payload["path"] as? String, !singlePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                let s = try await actor.mountAuxiliaryVaultPack(
                    at: singlePath,
                    packName: payload["pack_name"] as? String,
                    autoIndexIfMissing: true,
                    overrideRoots: allowedWorkspaceRoots
                )
                mountedSummaries.append(s)
            }
            for p in paths where !p.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                let s = try await actor.mountAuxiliaryVaultPack(
                    at: p, autoIndexIfMissing: true, overrideRoots: allowedWorkspaceRoots
                )
                mountedSummaries.append(s)
            }
            for d in docsPaths where !d.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                let s = try await actor.mountAuxiliaryVaultPack(
                    at: d, autoIndexIfMissing: true, overrideRoots: allowedWorkspaceRoots
                )
                mountedSummaries.append(s)
            }
            struct MultiMountOut: Encodable {
                let status: String
                let packs: [QWMountedVaultPackSummary]
            }
            sendJsonResponse(connection, status: 200, object: MultiMountOut(status: "ok", packs: mountedSummaries))
            return
        }
        try await handleMemoryMountRoute(connection: connection, body: body)
    }

    func handleMemoryUnmountRoute(connection: NWConnection, body: Data) async {
        var packId: String? = nil
        if !body.isEmpty,
           let payload = try? JSONSerialization.jsonObject(with: body) as? [String: Any] {
            packId = (payload["pack_id"] as? String)
                ?? (payload["pack_name"] as? String)
                ?? (payload["path"] as? String)
        }
        await actor.unmountMemoryPack(packIdentifier: packId)
        struct UnmountOut: Encodable {
            let status: String
            let packs: [QWMountedVaultPackSummary]
        }
        sendJsonResponse(connection, status: 200, object: UnmountOut(status: "ok", packs: actor.listMountedVaultPacks()))
    }

    func handleMemoryQueryRoute(connection: NWConnection, body: Data) async throws {
        guard !body.isEmpty,
              let req = try? QWServerRequestDecoder.decode(QWMemoryQueryHTTPRequest.self, from: body) else {
            let err = ServerErrorPayload(message: "Invalid JSON payload ('query' required) for /v1/memory/query.", code: 400)
            sendJsonResponse(connection, status: 400, object: err)
            return
        }
        let trimmed = req.query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            let err = ServerErrorPayload(message: "Non-empty 'query' is required.", code: 400)
            sendJsonResponse(connection, status: 400, object: err)
            return
        }
        let outcome = try await actor.queryVaultV2(
            query: trimmed,
            workspacePath: req.workspacePath,
            packNames: req.packNames,
            maxFiles: req.maxFiles ?? 4,
            maxTokens: req.maxTokens ?? QWMemoryVault.defaultVaultRecallTokenBudget,
            expandASTHops: req.expandASTHops ?? 1,
            screen: req.screen ?? true,
            includeCortexFiles: req.includeCortexFiles ?? true,
            overrideRoots: allowedWorkspaceRoots
        )
        sendJsonResponse(connection, status: 200, object: outcome)
    }
}
