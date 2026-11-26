// QWClientIdentityTests.swift — Unit and live-server tests for client fingerprinting and trace enrichment.

import Foundation
import Network
import QWCore
@testable import QWServer
import XCTest

private extension QWHttpServer {
    var testBoundPort: UInt16? { listener?.port?.rawValue }
}

final class QWClientIdentityTests: XCTestCase {
    func testHeaderResolutionPriorityOrder() {
        let headersAll: [String: String] = [
            "x-qwegn-client": "my-app",
            "x-client-name": "other-client",
            "x-title": "OpenCode",
            "originator": "codex_cli_rs",
            "user-agent": "claude-cli/1.0",
            "referer": "http://127.0.0.1:8080/ui/",
        ]
        XCTAssertEqual(QWClientIdentity.resolve(headers: headersAll, mcpClientName: "mcp-ide").client, "my-app")

        XCTAssertEqual(
            QWClientIdentity.resolve(headers: ["x-client-name": "custom-cli", "x-title": "Cline"]).client,
            "custom-cli"
        )
        XCTAssertEqual(
            QWClientIdentity.resolve(headers: ["x-title": "Roo Code", "originator": "codex_cli_rs"]).client,
            "Roo-Code"
        )
        XCTAssertEqual(
            QWClientIdentity.resolve(headers: ["originator": "codex_cli_rs", "user-agent": "curl/8.7.1"]).client,
            "codex"
        )
        XCTAssertEqual(
            QWClientIdentity.resolve(headers: ["user-agent": "undici"], mcpClientName: "CursorMCP").client,
            "CursorMCP"
        )
        XCTAssertEqual(
            QWClientIdentity.resolve(headers: ["referer": "http://127.0.0.1:8080/ui/chat"]).client,
            "webui"
        )
        XCTAssertEqual(
            QWClientIdentity.resolve(headers: ["x-stainless-lang": "python"]).client,
            "stainless-py"
        )
        XCTAssertEqual(
            QWClientIdentity.resolve(headers: ["anthropic-version": "2023-06-01"]).client,
            "anthropic-sdk"
        )
        XCTAssertEqual(QWClientIdentity.resolve(headers: [:]).client, "unknown")
    }

    func testUserAgentFingerprintingAndBrowserNormalization() {
        XCTAssertEqual(QWClientIdentity.resolve(headers: ["user-agent": "claude-cli/2.1.0"]).client, "claude-code")
        XCTAssertEqual(QWClientIdentity.resolve(headers: ["user-agent": "codex_cli_rs/0.1.0"]).client, "codex")
        XCTAssertEqual(QWClientIdentity.resolve(headers: ["user-agent": "opencode/0.4.2"]).client, "opencode")
        XCTAssertEqual(QWClientIdentity.resolve(headers: ["user-agent": "hermes-agent/1.2"]).client, "hermes")
        XCTAssertEqual(QWClientIdentity.resolve(headers: ["user-agent": "llmprobe/1.0"]).client, "llmprobe")
        XCTAssertEqual(QWClientIdentity.resolve(headers: ["user-agent": "Cursor/0.48.1"]).client, "cursor")
        XCTAssertEqual(QWClientIdentity.resolve(headers: ["user-agent": "Zed/0.180.0"]).client, "zed")
        XCTAssertEqual(QWClientIdentity.resolve(headers: ["user-agent": "curl/8.7.1"]).client, "curl/8.7.1")
        XCTAssertEqual(
            QWClientIdentity.resolve(headers: ["user-agent": "OpenAI/Python 1.66.0", "x-stainless-lang": "python"]).client,
            "openai-py"
        )
        XCTAssertEqual(
            QWClientIdentity.resolve(headers: [
                "user-agent": "Anthropic/JS 0.39.0",
                "x-stainless-lang": "js",
                "anthropic-version": "2023-06-01",
            ]).client,
            "anthropic-js"
        )
        XCTAssertEqual(
            QWClientIdentity.resolve(headers: [
                "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
                "referer": "http://127.0.0.1:8080/ui",
            ]).client,
            "webui"
        )
        XCTAssertEqual(
            QWClientIdentity.resolve(headers: [
                "user-agent": "Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Chrome/134.0.0.0 Safari/537.36",
            ]).client,
            "browser/chrome"
        )
        XCTAssertEqual(
            QWClientIdentity.resolve(headers: [
                "user-agent": "Mozilla/5.0 (Macintosh) AppleWebKit/605.1.15 Version/18.0 Safari/605.1.15",
            ]).client,
            "browser/safari"
        )
    }

    func testSanitizationAndDebugHeaderRedaction() {
        let dirty = "bad\r\nclient\u{001B}[31m name " + String(repeating: "x", count: 40)
        let cleaned = QWClientIdentity.sanitizeLabel(dirty)
        XCTAssertNotNil(cleaned)
        XCTAssertLessThanOrEqual(cleaned?.count ?? 99, 32)
        XCTAssertFalse(cleaned?.contains("\r") ?? true)
        XCTAssertFalse(cleaned?.contains("\n") ?? true)
        XCTAssertFalse(cleaned?.contains("\u{001B}") ?? true)

        let summary = QWClientIdentity.debugHeaderSummary(headers: [
            "authorization": "Bearer super-secret-token",
            "x-api-key": "sk-secret",
            "cookie": "session=secretcookie",
            "x-custom-secret": "should-not-log",
            "user-agent": "curl/8.7.1",
            "x-title": "MyTool",
            "x-request-id": "req-42",
        ])
        XCTAssertTrue(summary.contains("user-agent=curl/8.7.1"))
        XCTAssertTrue(summary.contains("x-title=MyTool"))
        XCTAssertTrue(summary.contains("x-request-id=req-42"))
        XCTAssertFalse(summary.contains("super-secret-token"))
        XCTAssertFalse(summary.contains("sk-secret"))
        XCTAssertFalse(summary.contains("secretcookie"))
        XCTAssertFalse(summary.contains("should-not-log"))
        XCTAssertFalse(summary.lowercased().contains("authorization"))
        XCTAssertFalse(summary.lowercased().contains("x-api-key"))
        XCTAssertFalse(summary.lowercased().contains("cookie"))
    }

    func testNonLoopbackPeerHostFiltering() {
        XCTAssertNil(QWClientIdentity.nonLoopbackPeerHost(from: .hostPort(host: .ipv4(.loopback), port: 12345)))
        XCTAssertNil(QWClientIdentity.nonLoopbackPeerHost(from: .hostPort(host: .ipv6(.loopback), port: 12345)))
        XCTAssertNil(QWClientIdentity.nonLoopbackPeerHost(from: .hostPort(host: NWEndpoint.Host("::ffff:127.0.0.1"), port: 12345)))
        XCTAssertEqual(
            QWClientIdentity.nonLoopbackPeerHost(from: .hostPort(host: NWEndpoint.Host("192.168.1.42"), port: 54321)),
            "192.168.1.42"
        )
    }

    func testBuildLedgerConfigIncludesClientAndRoute() throws {
        let json = """
        {"model":"test","messages":[{"role":"user","content":"hi"}]}
        """
        var req = try QWServerRequestDecoder.decode(ChatCompletionRequest.self, from: Data(json.utf8))
        req.httpHeaders = ["user-agent": "hermes/1.0", "x-session-id": "sess-99"]
        let cfg = QWEngineLifecycleActor.Config(packPath: "/tmp/m", modelName: "test")
        let trace = QWClientIdentity.makeTraceContext(
            method: "POST",
            route: "/v1/messages",
            headers: req.httpHeaders ?? [:],
            peerEndpoint: .hostPort(host: NWEndpoint.Host("10.0.0.5"), port: 50000)
        )
        let ledger = QWChatPolicyResolver.buildLedgerConfig(
            request: req,
            config: cfg,
            hasMountedManifest: false,
            trace: trace
        )
        XCTAssertEqual(ledger["client"], "hermes")
        XCTAssertEqual(ledger["route"], "/v1/messages")
        XCTAssertEqual(ledger["peer"], "10.0.0.5")
        XCTAssertEqual(ledger["sessionId"], "sess-99")
    }

    func testLiveServerCorsHeadersAndMcpInitializeClientLogging() async throws {
        let cfg = QWEngineLifecycleActor.Config(
            packPath: "/dummy/model.qwp",
            modelName: "trace-test-model",
            host: "127.0.0.1",
            port: 0
        )
        let logger = QWServerLogger(level: .info)
        let actor = QWEngineLifecycleActor(config: cfg, logger: logger)
        let server = QWHttpServer(actor: actor, host: "127.0.0.1", port: 0, logger: logger)
        try server.start()
        defer { server.stop() }
        try await Task.sleep(nanoseconds: 50_000_000)

        let port = try XCTUnwrap(server.testBoundPort)
        let baseURL = URL(string: "http://127.0.0.1:\(port)")!

        var optReq = URLRequest(url: baseURL.appendingPathComponent("v1/chat/completions"))
        optReq.httpMethod = "OPTIONS"
        optReq.setValue("http://127.0.0.1:\(port)", forHTTPHeaderField: "Origin")
        let (_, optResp) = try await URLSession.shared.data(for: optReq)
        let httpOpt = try XCTUnwrap(optResp as? HTTPURLResponse)
        XCTAssertEqual(httpOpt.statusCode, 204)
        let allowHeaders = (httpOpt.value(forHTTPHeaderField: "Access-Control-Allow-Headers") ?? "").lowercased()
        XCTAssertTrue(allowHeaders.contains("x-qwegn-client"))
        XCTAssertTrue(allowHeaders.contains("x-title"))

        var mcpReq = URLRequest(url: baseURL.appendingPathComponent("mcp"))
        mcpReq.httpMethod = "POST"
        mcpReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        mcpReq.setValue("req-mcp-1", forHTTPHeaderField: "X-Request-ID")
        mcpReq.setValue("curl/8.7.1", forHTTPHeaderField: "User-Agent")
        mcpReq.httpBody = Data("""
        {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientInfo":{"name":"claude-desktop"}}}
        """.utf8)
        let (_, mcpResp) = try await URLSession.shared.data(for: mcpReq)
        XCTAssertEqual((mcpResp as? HTTPURLResponse)?.statusCode, 200)

        let logs = logger.getRecentLogs()
        XCTAssertTrue(logs.contains { line in
            line.contains("POST /mcp -> 200 OK")
                && line.contains("client=claude-desktop")
                && line.contains("req=req-mcp-1")
                && line.contains("initialize")
        })
    }
}
