// QWMCPTools+CognitiveAndVault.swift — MCP tool definitions and handlers for Cognitive Rails & Vault Recall (G3).

import Foundation
import QWCore
import QWDecision
import QWEngine
import QWToken

extension QWMCPDispatcher {
    private static func objProp(_ type: String) -> QWJson {
        .object([("type", .string(type))])
    }

    private static func schema(_ name: String, _ desc: String, properties: [(String, QWJson)], required: [String]) -> QWJson {
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

    static func cognitiveAndVaultToolDefinitions() -> [QWJson] {
        [
            schema(
                "qwg_rerank_screen",
                "Rerank candidate passages against a query while filtering prompt injections via System-1 + local screen.",
                properties: [
                    ("query", objProp("string")),
                    ("passages", objProp("array")),
                    ("relevance_threshold", objProp("number")),
                    ("injection_threshold", objProp("number")),
                    ("top_k", objProp("integer")),
                ],
                required: ["query", "passages"]
            ),
            schema(
                "qwg_compact_select",
                "Select keep/summarize/drop actions for conversation turns while pinning system, pinned, and tail turns.",
                properties: [
                    ("turns", objProp("array")),
                    ("keep_tail_count", objProp("integer")),
                    ("drop_confidence_rail", objProp("number")),
                    ("archive_session_id", objProp("string")),
                ],
                required: ["turns"]
            ),
            schema(
                "qwg_episodic_recover",
                "Recover compacted historical turns from the encrypted-perm episodic store using BM25 + substring match.",
                properties: [
                    ("session_id", objProp("string")),
                    ("query", objProp("string")),
                    ("limit", objProp("integer")),
                ],
                required: ["session_id", "query"]
            ),
            schema(
                "qwg_skill_select",
                "Run two-stage skill catalog shortlisting and verification, filtering missing-on-disk skills.",
                properties: [
                    ("user_turn", objProp("string")),
                    ("catalog", objProp("array")),
                    ("max_selected", objProp("integer")),
                ],
                required: ["user_turn", "catalog"]
            ),
            schema(
                "qwg_search_enough",
                "Evaluate whether retrieved evidence suffices to answer a query, detecting conflicts and deduping follow-up queries.",
                properties: [
                    ("query", objProp("string")),
                    ("passages", objProp("array")),
                    ("candidate_queries", objProp("array")),
                    ("prior_queries", objProp("array")),
                ],
                required: ["query", "passages"]
            ),
            schema(
                "qwg_supervise",
                "Supervise a delegated background task with exit-code and silence fact overrides plus alert streak filtering.",
                properties: [
                    ("task_id", objProp("string")),
                    ("exit_code", objProp("integer")),
                    ("seconds_since_output", objProp("number")),
                    ("tail_output", objProp("string")),
                    ("consecutive_alerts", objProp("integer")),
                ],
                required: ["task_id"]
            ),
            schema(
                "qwg_route_turn",
                "Classify per-turn difficulty, domain, and costly risk with sticky 32K context guard and shadow mode support.",
                properties: [
                    ("prompt", objProp("string")),
                    ("context_tokens", objProp("integer")),
                    ("current_tier", objProp("string")),
                    ("has_tools", objProp("boolean")),
                    ("policy", objProp("object")),
                ],
                required: ["prompt"]
            ),
            schema(
                "qwg_vault_query",
                "Query the workspace-contained .qwg/manifest.json Memory Vault and optionally screen recalled passages.",
                properties: [
                    ("workspace_path", objProp("string")),
                    ("query", objProp("string")),
                    ("max_files", objProp("integer")),
                    ("screen", objProp("boolean")),
                ],
                required: ["workspace_path", "query"]
            ),
            schema(
                "qwg_vault_status",
                "Report mounted Memory Pack status and optional workspace manifest health inside allowed workspace roots.",
                properties: [
                    ("workspace_path", objProp("string")),
                ],
                required: []
            ),
        ]
    }

    func makeCognitiveRails() -> QWCognitiveRails {
        guard actor.isDecisionSidecarLoaded() else {
            return QWCognitiveRails(runner: nil)
        }
        let act = self.actor
        return QWCognitiveRails(decideFn: { req in
            try await act.decide(req).response
        })
    }

    func executeCognitiveOrVaultTool(name: String, arguments: QWJson) async throws -> String? {
        let rawData = Data(arguments.tojson.utf8)
        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]

        func encode(_ val: some Encodable) throws -> String {
            String(decoding: try encoder.encode(val), as: UTF8.self)
        }

        switch name {
        case "qwg_rerank_screen":
            struct Args: Decodable {
                let query: String
                let passages: [QWRerankPassageInput]
                let relevance_threshold: Double?
                let injection_threshold: Double?
                let top_k: Int?
            }
            let a = try decoder.decode(Args.self, from: rawData)
            let res = await makeCognitiveRails().screenAndRerankPassages(
                query: a.query, passages: a.passages,
                relevanceThreshold: a.relevance_threshold ?? 0.50,
                injectionThreshold: a.injection_threshold ?? 0.50,
                topK: a.top_k ?? 8
            )
            return try encode(res)

        case "qwg_compact_select":
            struct Args: Decodable {
                let turns: [QWCompactionTurnInput]
                let keep_tail_count: Int?
                let drop_confidence_rail: Double?
                let archive_session_id: String?
            }
            let a = try decoder.decode(Args.self, from: rawData)
            let res = await makeCognitiveRails().selectCompaction(
                turns: a.turns,
                keepTailCount: a.keep_tail_count ?? 6,
                dropConfidenceRail: a.drop_confidence_rail ?? 0.70
            )
            if let sid = a.archive_session_id, !sid.isEmpty {
                let idxs = Set(res.decisions.filter { $0.action == .drop || $0.action == .summarize }.map(\.index))
                let turns = a.turns.filter { idxs.contains($0.index) }.map { QWEpisodicTurn(index: $0.index, role: $0.role, text: $0.text) }
                _ = try episodicStore.archive(sessionId: sid, turns: turns)
            }
            return try encode(res)

        case "qwg_episodic_recover":
            struct Args: Decodable { let session_id: String; let query: String; let limit: Int? }
            let a = try decoder.decode(Args.self, from: rawData)
            let hits = try episodicStore.recover(sessionId: a.session_id, query: a.query, limit: a.limit ?? 5)
            return try encode(hits)

        case "qwg_skill_select":
            struct Args: Decodable { let user_turn: String; let catalog: [QWSkillDescriptor]; let max_selected: Int? }
            let a = try decoder.decode(Args.self, from: rawData)
            let res = await makeCognitiveRails().selectSkills(userTurn: a.user_turn, catalog: a.catalog, maxSelected: a.max_selected ?? 3)
            return try encode(res)

        case "qwg_search_enough":
            struct Args: Decodable {
                let query: String
                let passages: [String]
                let candidate_queries: [String]?
                let prior_queries: [String]?
            }
            let a = try decoder.decode(Args.self, from: rawData)
            let res = await makeCognitiveRails().evaluateSearchSufficiency(
                query: a.query, passages: a.passages,
                candidateQueries: a.candidate_queries ?? [], priorQueries: a.prior_queries ?? []
            )
            return try encode(res)

        case "qwg_supervise":
            let a = try decoder.decode(QWSuperviseInput.self, from: rawData)
            return try encode(await makeCognitiveRails().superviseDelegatedRun(a))

        case "qwg_route_turn":
            let a = try decoder.decode(QWRouteTurnInput.self, from: rawData)
            return try encode(await makeCognitiveRails().routeTurn(a))

        case "qwg_vault_query":
            struct Args: Decodable { let workspace_path: String; let query: String; let max_files: Int?; let screen: Bool? }
            let a = try decoder.decode(Args.self, from: rawData)
            let rootURL = try actor.resolveValidatedMemoryRootURL(at: a.workspace_path)
            let manifest = try QWMemoryManifest.load(from: rootURL)
            let recall = QWMemoryVault.recallVaultContext(
                query: a.query, manifest: manifest, includeCortexFiles: true, maxFiles: a.max_files ?? 4
            )
            var screenedItems: [QWRerankScreenItem]? = nil
            if a.screen == true && !recall.recalledFiles.isEmpty {
                let passages = recall.recalledFiles.enumerated().map { idx, rel in
                    let fileURL = rootURL.appendingPathComponent(rel)
                    let content = (try? String(contentsOf: fileURL, encoding: .utf8)) ?? ""
                    return QWRerankPassageInput(id: rel, text: String(content.prefix(2048)), metadata: ["rank": "\(idx)"])
                }
                screenedItems = await makeCognitiveRails().screenAndRerankPassages(query: a.query, passages: passages).items
            }
            struct VaultQueryOut: Encodable {
                let workspace: String
                let recalled_files: [String]
                let estimated_tokens: Int
                let recall_block: String
                let screened_items: [QWRerankScreenItem]?
            }
            return try encode(VaultQueryOut(
                workspace: rootURL.path,
                recalled_files: recall.recalledFiles,
                estimated_tokens: recall.estimatedTokens,
                recall_block: recall.recallBlock,
                screened_items: screenedItems
            ))

        case "qwg_vault_status":
            struct Args: Decodable { let workspace_path: String? }
            let a = try decoder.decode(Args.self, from: rawData)
            let mounted = actor.statusBox.getMountedMemoryPackInfo()
            var manifestFiles: Int? = nil
            var resolvedWorkspace: String? = nil
            if let ws = a.workspace_path, !ws.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                let rootURL = try actor.resolveValidatedMemoryRootURL(at: ws)
                resolvedWorkspace = rootURL.path
                if let manifest = try? QWMemoryManifest.load(from: rootURL) {
                    manifestFiles = manifest.files.count
                }
            }
            struct VaultStatusOut: Encodable {
                let mounted: Bool
                let pack_name: String?
                let token_count: Int?
                let workspace: String?
                let manifest_file_count: Int?
            }
            return try encode(VaultStatusOut(
                mounted: mounted != nil,
                pack_name: mounted?.packName,
                token_count: mounted?.tokenCount,
                workspace: resolvedWorkspace,
                manifest_file_count: manifestFiles
            ))

        default:
            return nil
        }
    }
}
