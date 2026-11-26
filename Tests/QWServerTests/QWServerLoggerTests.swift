import XCTest
@testable import QWServer

final class QWServerLoggerTests: XCTestCase {
    func testLogLevelOrderingAndFromString() {
        XCTAssertTrue(QWLogLevel.debug < QWLogLevel.info)
        XCTAssertTrue(QWLogLevel.info < QWLogLevel.warn)
        XCTAssertTrue(QWLogLevel.warn < QWLogLevel.error)
        XCTAssertTrue(QWLogLevel.error < QWLogLevel.none)

        XCTAssertEqual(QWLogLevel.fromString("debug"), .debug)
        XCTAssertEqual(QWLogLevel.fromString("DEBUG"), .debug)
        XCTAssertEqual(QWLogLevel.fromString("info"), .info)
        XCTAssertEqual(QWLogLevel.fromString("warn"), .warn)
        XCTAssertEqual(QWLogLevel.fromString("warning"), .warn)
        XCTAssertEqual(QWLogLevel.fromString("error"), .error)
        XCTAssertEqual(QWLogLevel.fromString("none"), .none)
        XCTAssertEqual(QWLogLevel.fromString("off"), .none)
        XCTAssertEqual(QWLogLevel.fromString("unknown"), .info)
    }

    func testLogRequestExecutionDoesNotCrash() {
        let logger = QWServerLogger(level: .info)
        // Verify multiple non-allocating request log calls run without errors or race conditions
        for i in 0..<10 {
            logger.logRequest(
                method: "POST",
                path: "/v1/chat/completions",
                statusCode: 200,
                promptTokens: 40 + i,
                completionTokens: 100 + i * 2,
                imagesCount: i % 2,
                ttftMs: 14.5 + Double(i),
                decodeTokPerSec: 48.2 + Double(i) * 0.1,
                totalSec: 2.34 + Double(i) * 0.05
            )
        }

        // Test with level none (should early return without formatting)
        logger.setLevel(.none)
        logger.logRequest(
            method: "POST",
            path: "/v1/chat/completions",
            statusCode: 200,
            promptTokens: 10,
            completionTokens: 10,
            imagesCount: 0,
            ttftMs: 5.0,
            decodeTokPerSec: 50.0,
            totalSec: 0.2
        )
    }

    func testSimpleLogMessages() {
        let logger = QWServerLogger(level: .debug)
        logger.log(.debug, "Test debug message")
        logger.log(.info, "Test info message")
        logger.log(.warn, "Test warning message")
        logger.log(.error, "Test error message")
    }

    func testEnrichedLogRequestAndSimpleRequestFormatting() {
        final class CapturedBox: @unchecked Sendable {
            private let lock = NSLock()
            private var items: [String] = []
            func append(_ line: String) { lock.withLock { items.append(line) } }
            var count: Int { lock.withLock { items.count } }
        }
        let logger = QWServerLogger(level: .info)
        let captured = CapturedBox()
        let subId = logger.subscribe { line in captured.append(line) }
        defer { logger.unsubscribe(subId) }

        logger.logRequest(
            method: "POST",
            path: "/v1/messages",
            statusCode: 200,
            promptTokens: 512,
            completionTokens: 64,
            cacheHitTokens: 480,
            cacheTier: "warm",
            cacheActive: true,
            specSummary: String(repeating: "s", count: 320),
            ttftMs: 24.3,
            decodeTokPerSec: 81.2,
            totalSec: 0.81,
            client: "claude-code",
            requestId: "msg_8f2a9c",
            sessionId: "sess_42",
            peerHost: "192.168.1.50"
        )
        logger.logRequest(
            method: "POST",
            path: "/v1/chat/completions",
            statusCode: 200,
            promptTokens: 10,
            completionTokens: 5,
            ttftMs: 5.0,
            decodeTokPerSec: 50.0,
            totalSec: 0.2
        )
        logger.logSimpleRequest(
            method: "POST",
            path: "/mcp",
            statusCode: 202,
            client: "cursor",
            requestId: "req_1",
            detail: "notifications/initialized",
            totalSec: 0.01
        )
        logger.logSimpleRequest(
            method: "POST",
            path: "/v1/responses",
            statusCode: 499,
            client: "webui",
            requestId: "resp_9",
            detail: "cancelled",
            totalSec: 1.25
        )

        let recent = logger.getRecentLogs()
        XCTAssertEqual(captured.count, 4)
        XCTAssertEqual(recent.count, 4)

        let msgLine = recent[0]
        XCTAssertGreaterThan(msgLine.utf8.count, 512)
        XCTAssertTrue(msgLine.contains("POST /v1/messages -> 200 OK | client=claude-code req=msg_8f2a9c sess=sess_42 peer=192.168.1.50 | "))
        XCTAssertTrue(msgLine.contains("512 prompt (hit 480 [warm]) + 64 gen tok"))

        let legacyLine = recent[1]
        XCTAssertTrue(legacyLine.contains("POST /v1/chat/completions -> 200 OK | 10 prompt + 5 gen tok"))
        XCTAssertFalse(legacyLine.contains("client="))

        let mcpLine = recent[2]
        XCTAssertTrue(mcpLine.contains("POST /mcp -> 202 OK | client=cursor req=req_1 | notifications/initialized | 0.01s"))

        let cancelLine = recent[3]
        XCTAssertTrue(cancelLine.contains("POST /v1/responses -> 499 ERR | client=webui req=resp_9 | cancelled | 1.25s"))
    }
}
