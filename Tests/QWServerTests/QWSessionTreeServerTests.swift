// QWSessionTreeServerTests.swift — HTTP /v1/sessions/* & QWResponsesStore sessionNodeId tests (G4).

import Foundation
import QWCore
import QWEngine
import QWToken
import XCTest
@testable import QWServer

final class QWSessionTreeServerTests: XCTestCase {
    func testSessionsEndpointsAndResponsesStoreChaining() async throws {
        let actor = QWEngineLifecycleActor(
            config: QWEngineLifecycleActor.Config(packPath: "/tmp/unused", modelName: "qwen3.5-mock"),
            logger: QWServerLogger(level: .none)
        )
        let tree = QWSessionTree()
        let server = QWHttpServer(
            actor: actor,
            host: "127.0.0.1",
            port: 0,
            logger: QWServerLogger(level: .none),
            finderRevealRunner: nil,
            sessionTree: tree
        )
        try server.start()
        defer { server.stop() }
        let port = Int(server.listener?.port?.rawValue ?? 0)

        func send(_ method: String, _ path: String, body: [String: Any]? = nil) async throws -> (Int, [String: Any]) {
            var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
            req.httpMethod = method
            if let body {
                req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                req.httpBody = try JSONSerialization.data(withJSONObject: body)
            }
            let (data, resp) = try await URLSession.shared.data(for: req)
            let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
            let json = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
            return (code, json)
        }

        // 1. Checkpoint
        let (cCode, cJson) = try await send("POST", "/v1/sessions/checkpoint", body: [
            "label": "turn-1",
            "token_ids": [10, 20, 30],
            "kv_mode": "q4",
            "metadata": ["branch": "main"],
        ])
        XCTAssertEqual(cCode, 200)
        let rootId = try XCTUnwrap(cJson["id"] as? String)

        // 2. Fork into 2 parallel branches
        let (fCode, fJson) = try await send("POST", "/v1/sessions/fork", body: [
            "parent_id": rootId,
            "count": 2,
            "labels": ["branch-a", "branch-b"],
        ])
        XCTAssertEqual(fCode, 200)
        let children = try XCTUnwrap(fJson["children"] as? [[String: Any]])
        XCTAssertEqual(children.count, 2)
        let childA = try XCTUnwrap(children[0]["id"] as? String)

        // 3. GET /v1/sessions and GET /v1/sessions/:id + 404 check
        let (lCode, lJson) = try await send("GET", "/v1/sessions")
        XCTAssertEqual(lCode, 200)
        XCTAssertEqual((lJson["data"] as? [[String: Any]])?.count, 3)

        let (dCode, dJson) = try await send("GET", "/v1/sessions/\(childA)")
        XCTAssertEqual(dCode, 200)
        let lineage = try XCTUnwrap(dJson["lineage"] as? [[String: Any]])
        XCTAssertEqual(lineage.count, 2)

        let (mCode, _) = try await send("GET", "/v1/sessions/does-not-exist")
        XCTAssertEqual(mCode, 404)

        // 4. POST /v1/sessions/restore (no loaded primary -> needs_replay=true)
        let (rCode, rJson) = try await send("POST", "/v1/sessions/restore", body: ["node_id": childA])
        XCTAssertEqual(rCode, 200)
        XCTAssertEqual(rJson["kv_cache_hit"] as? Bool, false)
        XCTAssertEqual(rJson["needs_replay"] as? Bool, true)
        XCTAssertEqual(rJson["pinned_prefix_tokens"] as? Int, 3)

        // 5. Prune subtree via cascade=true
        let (pCode, pJson) = try await send("POST", "/v1/sessions/prune", body: ["node_id": rootId, "cascade": true])
        XCTAssertEqual(pCode, 200)
        XCTAssertEqual((pJson["removed_ids"] as? [String])?.count, 3)

        // 6. QWResponsesStore sessionNodeId storage field
        let res = OpenAIResponseResource(
            id: "resp_1", object: "response", createdAt: 100, completedAt: 101, status: "completed",
            incompleteDetails: nil, model: "qwen3.5-mock", previousResponseId: nil, instructions: nil,
            output: [], error: nil, tools: [], toolChoice: .string("auto"), truncation: "disabled",
            parallelToolCalls: true, text: OpenAIResponseResource.TextConfig(), topP: 1.0,
            presencePenalty: 0.0, frequencyPenalty: 0.0, topLogprobs: 0, temperature: 1.0,
            reasoning: nil, usage: nil, maxOutputTokens: nil, maxToolCalls: nil,
            store: true, background: false, serviceTier: "default", metadata: [:],
            safetyIdentifier: nil, promptCacheKey: nil
        )
        let store = QWResponsesStore()
        store.put(QWResponsesStore.StoredResponse(resource: res, historyMessages: [], sessionNodeId: "sess_abc"))
        XCTAssertEqual(store.get("resp_1")?.sessionNodeId, "sess_abc")
    }
}
