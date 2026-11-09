import XCTest
@testable import QWServer
import Foundation

final class QWServerIntegrationTests: XCTestCase {
    func testHttpServerEndpointRoutingAndCors() async throws {
        let port = 19095
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/model.qwp",
            modelName: "test-model",
            host: "127.0.0.1",
            port: port
        )
        let logger = QWServerLogger(level: .none)
        let actor = QWEngineLifecycleActor(config: config, logger: logger)
        let server = QWHttpServer(actor: actor, host: "127.0.0.1", port: port, logger: logger)

        try server.start()
        defer { server.stop() }

        // Allow listener to bind
        try await Task.sleep(nanoseconds: 100_000_000)

        let session = URLSession.shared
        let baseURL = URL(string: "http://127.0.0.1:\(port)")!

        // 1. Test GET /health
        let healthURL = baseURL.appendingPathComponent("health")
        let (healthData, healthResp) = try await session.data(from: healthURL)
        let httpHealth = healthResp as! HTTPURLResponse
        XCTAssertEqual(httpHealth.statusCode, 200)
        let healthJson = try JSONSerialization.jsonObject(with: healthData) as? [String: Any]
        // The status is the engine's phase, not a fixed "ok": a server that
        // has not loaded its model says so.
        XCTAssertEqual(healthJson?["status"] as? String, "unloaded")
        XCTAssertEqual(healthJson?["model"] as? String, "test-model")
        XCTAssertEqual(healthJson?["loaded"] as? Bool, false) // lazy loaded, so initially false
        let healthLifecycle = healthJson?["lifecycle"] as? [String: Any]
        XCTAssertEqual(healthLifecycle?["phase"] as? String, "unloaded")
        XCTAssertNil(healthLifecycle?["model"] as? String)

        // 1b. Test GET /v1/dashboard — the card list both renderers draw
        let dashboardURL = baseURL.appendingPathComponent("v1/dashboard")
        let (dashboardData, dashboardResp) = try await session.data(from: dashboardURL)
        let httpDashboard = dashboardResp as! HTTPURLResponse
        XCTAssertEqual(httpDashboard.statusCode, 200)
        let dashboardJson = try JSONSerialization.jsonObject(with: dashboardData) as? [String: Any]
        XCTAssertEqual(dashboardJson?["schemaVersion"] as? Int, 1)
        // No model is resident, so the dashboard names none.
        XCTAssertNil(dashboardJson?["modelId"] as? String)
        let dashboardLifecycle = dashboardJson?["lifecycle"] as? [String: Any]
        XCTAssertEqual(dashboardLifecycle?["phase"] as? String, "unloaded")
        let cards = dashboardJson?["cards"] as? [[String: Any]]
        XCTAssertNotNil(cards)
        XCTAssertFalse(cards?.isEmpty ?? true)
        // A card carries its provenance, so a renderer can say where the
        // number came from. What it says depends on the host.
        let gpuCard = cards?.first { $0["id"] as? String == "gpu.device" }
        XCTAssertNotNil(gpuCard)
        let provenance = gpuCard?["provenance"] as? String
        XCTAssertTrue(provenance == "measured" || provenance == "unavailable")

        // 2. Test GET /v1/models
        let modelsURL = baseURL.appendingPathComponent("v1/models")
        let (modelsData, modelsResp) = try await session.data(from: modelsURL)
        let httpModels = modelsResp as! HTTPURLResponse
        XCTAssertEqual(httpModels.statusCode, 200)
        let modelsJson = try JSONSerialization.jsonObject(with: modelsData) as? [String: Any]
        XCTAssertEqual(modelsJson?["object"] as? String, "list")
        let modelList = modelsJson?["data"] as? [[String: Any]]
        XCTAssertNotNil(modelList)
        XCTAssertTrue((modelList?.count ?? 0) >= 1)
        XCTAssertEqual(modelList?[0]["id"] as? String, "test-model")
        XCTAssertEqual(modelList?[0]["loaded"] as? Bool, false)

        // 3. Test POST /v1/models/unload
        var unloadReq = URLRequest(url: baseURL.appendingPathComponent("v1/models/unload"))
        unloadReq.httpMethod = "POST"
        let (unloadData, unloadResp) = try await session.data(for: unloadReq)
        let httpUnload = unloadResp as! HTTPURLResponse
        XCTAssertEqual(httpUnload.statusCode, 200)
        let unloadJson = try JSONSerialization.jsonObject(with: unloadData) as? [String: Any]
        XCTAssertEqual(unloadJson?["status"] as? String, "ok")
        XCTAssertEqual(unloadJson?["loaded"] as? Bool, false)

        // 4. Test GET /api/tags
        let tagsURL = baseURL.appendingPathComponent("api/tags")
        let (tagsData, tagsResp) = try await session.data(from: tagsURL)
        let httpTags = tagsResp as! HTTPURLResponse
        XCTAssertEqual(httpTags.statusCode, 200)
        let tagsJson = try JSONSerialization.jsonObject(with: tagsData) as? [String: Any]
        let tagsList = tagsJson?["models"] as? [[String: Any]]
        XCTAssertNotNil(tagsList)
        XCTAssertTrue((tagsList?.count ?? 0) >= 1)
        XCTAssertEqual(tagsList?[0]["name"] as? String, "test-model")

        // 5. Test OPTIONS CORS preflight
        var optReq = URLRequest(url: baseURL.appendingPathComponent("v1/chat/completions"))
        optReq.httpMethod = "OPTIONS"
        let (_, optResp) = try await session.data(for: optReq)
        let httpOpt = optResp as! HTTPURLResponse
        XCTAssertEqual(httpOpt.statusCode, 204)
        XCTAssertEqual(httpOpt.value(forHTTPHeaderField: "Access-Control-Allow-Origin"), "*")
        XCTAssertNotNil(httpOpt.value(forHTTPHeaderField: "Access-Control-Allow-Methods"))

        // 6. Test 404 for unknown route
        let unknownURL = baseURL.appendingPathComponent("unknown/path")
        let (unknownData, unknownResp) = try await session.data(from: unknownURL)
        let httpUnknown = unknownResp as! HTTPURLResponse
        XCTAssertEqual(httpUnknown.statusCode, 404)
        let unknownJson = try JSONSerialization.jsonObject(with: unknownData) as? [String: Any]
        XCTAssertNotNil(unknownJson?["error"])
    }
}
