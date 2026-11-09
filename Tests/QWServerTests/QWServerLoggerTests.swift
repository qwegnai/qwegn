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
}
