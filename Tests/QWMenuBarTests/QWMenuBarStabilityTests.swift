// QWMenuBarStabilityTests.swift — Regression tests for IPv6 URLs, Bearer auth, pipe buffer cap, and lifecycle guards.

import Foundation
import QWCore
@testable import QWMenuBarUI
@testable import QWServer
import XCTest

private extension QWHttpServer {
    var testBoundPort: UInt16? {
        listener?.port?.rawValue
    }
}

final class QWMenuBarStabilityTests: XCTestCase {
    @MainActor
    func testIPv6AndWildcardURLConstructionAndBearerAuthHeaders() {
        XCTAssertEqual(
            TelemetryObservable.serverURL(host: "::1", port: 8080, path: "/v1/dashboard")?.absoluteString,
            "http://[::1]:8080/v1/dashboard"
        )
        XCTAssertEqual(
            TelemetryObservable.serverURL(host: "[::1]", port: 8080, path: "health")?.absoluteString,
            "http://[::1]:8080/health"
        )
        XCTAssertEqual(
            TelemetryObservable.serverURL(host: "fe80::1%en0", port: 8080, path: "/ui")?.absoluteString,
            "http://[fe80::1%25en0]:8080/ui"
        )
        XCTAssertEqual(
            TelemetryObservable.serverURL(host: "::", port: 8080, path: "/health")?.absoluteString,
            "http://127.0.0.1:8080/health"
        )
        XCTAssertEqual(
            TelemetryObservable.serverURL(host: "0.0.0.0", port: 8080, path: "/health")?.absoluteString,
            "http://127.0.0.1:8080/health"
        )
        XCTAssertNil(TelemetryObservable.serverURL(host: "127.0.0.1", port: 0, path: "/health"))

        var settings = QWServerSettings()
        settings.host = "::1"
        settings.port = 9090
        settings.apiKey = "secret-bearer-token"
        let telemetry = TelemetryObservable(startServices: false, enableAutoLifecycle: false, initialSettings: settings)
        let req = telemetry.authorizedRequest(path: "/v1/models/unload", method: "POST")
        XCTAssertEqual(req?.url?.absoluteString, "http://[::1]:9090/v1/models/unload")
        XCTAssertEqual(req?.httpMethod, "POST")
        XCTAssertEqual(req?.value(forHTTPHeaderField: "Authorization"), "Bearer secret-bearer-token")
        XCTAssertEqual(req?.value(forHTTPHeaderField: "X-Qwegn-Client"), "menubar")

        telemetry.settings.apiKey = "   "
        let unauthReq = telemetry.authorizedRequest(path: "/health")
        XCTAssertNil(unauthReq?.value(forHTTPHeaderField: "Authorization"))
        XCTAssertEqual(unauthReq?.value(forHTTPHeaderField: "X-Qwegn-Client"), "menubar")
    }

    @MainActor
    func testTelemetryObservableAuthenticatesOverIPv6WithBearerKey() async throws {
        let cfg = QWEngineLifecycleActor.Config(
            packPath: "/dummy/model.qwp",
            modelName: "ipv6-auth-model",
            host: "::1",
            port: 0
        )
        let logger = QWServerLogger(level: .none)
        let actor = QWEngineLifecycleActor(config: cfg, logger: logger)
        let server = QWHttpServer(
            actor: actor,
            host: "::1",
            port: 0,
            logger: logger,
            apiKey: "menubar-ipv6-secret"
        )
        try server.start()
        defer { server.stop() }
        try await Task.sleep(nanoseconds: 50_000_000)

        let boundPort = Int(try XCTUnwrap(server.testBoundPort))
        var badSettings = QWServerSettings()
        badSettings.host = "::1"
        badSettings.port = boundPort
        badSettings.apiKey = "wrong-key"
        let unauthTelemetry = TelemetryObservable(
            startServices: false,
            enableAutoLifecycle: false,
            initialSettings: badSettings
        )
        let badHealthy = await unauthTelemetry.checkServerHealth()
        XCTAssertFalse(badHealthy)

        var okSettings = QWServerSettings()
        okSettings.host = "::1"
        okSettings.port = boundPort
        okSettings.apiKey = "menubar-ipv6-secret"
        let telemetry = TelemetryObservable(
            startServices: false,
            enableAutoLifecycle: false,
            initialSettings: okSettings
        )
        let okHealthy = await telemetry.checkServerHealth()
        XCTAssertTrue(okHealthy)

        telemetry.fetchDashboard()
        let deadline = Date().addingTimeInterval(2.0)
        while telemetry.connection != .connected && Date() < deadline {
            try await Task.sleep(nanoseconds: 25_000_000)
        }
        XCTAssertEqual(telemetry.connection, .connected)
        XCTAssertNotNil(telemetry.dashboard)
    }

    func testChildPipeLineBufferCapsSizeAndPreservesUTF8Boundaries() {
        let buffer = ChildPipeLineBuffer(maxBufferBytes: 16)
        var emitted: [String] = []

        // Multibyte UTF-8 ("é" is 2 bytes) straddling byte 16 must split cleanly.
        let payload = Data((String(repeating: "a", count: 15) + "é" + String(repeating: "b", count: 20)).utf8)
        buffer.append(payload) { emitted.append($0) }
        XCTAssertLessThanOrEqual(buffer.bufferedByteCount, 16)
        buffer.flush { emitted.append($0) }
        XCTAssertEqual(buffer.bufferedByteCount, 0)
        XCTAssertEqual(emitted.joined(), String(repeating: "a", count: 15) + "é" + String(repeating: "b", count: 20))

        var lines: [String] = []
        buffer.append(Data("first\r\nsecond\npart".utf8)) { lines.append($0) }
        XCTAssertEqual(lines, ["first", "second"])
        XCTAssertEqual(buffer.bufferedByteCount, 4)
        buffer.flush { lines.append($0) }
        XCTAssertEqual(lines, ["first", "second", "part"])
    }

    @MainActor
    func testRestartReentrancyUnhealthyAttachAndPIDIdentityGuards() async throws {
        var settings = QWServerSettings()
        settings.modelPackPath = ""
        settings.modelName = ""
        let telemetry = TelemetryObservable(startServices: false, enableAutoLifecycle: false, initialSettings: settings)

        telemetry.engineState = .starting
        telemetry.isStartingProcess = true
        telemetry.lastKnownServerPid = 999_999
        telemetry.resolveUnhealthyAttach(pid: 999_999, processAlive: false)
        XCTAssertEqual(telemetry.engineState, .stopped)
        XCTAssertFalse(telemetry.isStartingProcess)
        XCTAssertNil(telemetry.lastKnownServerPid)

        telemetry.engineState = .starting
        telemetry.isStartingProcess = true
        telemetry.resolveUnhealthyAttach(pid: 999_999, processAlive: true)
        if case .error = telemetry.engineState {
            XCTAssertFalse(telemetry.isStartingProcess)
        } else {
            XCTFail("Expected .error state after unhealthy live attach, got \(telemetry.engineState)")
        }

        // Calling restartServerProcess twice synchronously must trigger only one task.
        telemetry.engineState = .stopped
        telemetry.restartServerProcess()
        let firstTask = telemetry.restartTask
        XCTAssertTrue(telemetry.isRestartingProcess)
        XCTAssertTrue(telemetry.isDaemonTransitioning)
        XCTAssertNotNil(firstTask)
        telemetry.restartServerProcess()
        XCTAssertEqual(telemetry.restartTask, firstTask)
        telemetry.restartTask?.cancel()
        await firstTask?.value
        XCTAssertFalse(telemetry.isRestartingProcess)
        XCTAssertFalse(telemetry.isDaemonTransitioning)

        // Current test runner process is not named `qwg`.
        XCTAssertFalse(TelemetryObservable.isVerifiedQWGProcess(pid: ProcessInfo.processInfo.processIdentifier))
        XCTAssertFalse(TelemetryObservable.isVerifiedQWGProcess(pid: 0))
        XCTAssertFalse(TelemetryObservable.isVerifiedQWGProcess(pid: -1))
    }
}
