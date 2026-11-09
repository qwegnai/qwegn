// QWRequestRecordTests.swift — The record says what a turn cost, and
// nothing about what was asked.

import XCTest
@testable import QWEngine

final class QWRequestRecordTests: XCTestCase {
    /// The ring is bounded: a server that remembered every request would
    /// be a transcript, not a measurement.
    func testTheRingIsBoundedAtSixtyFour() {
        let store = QWRequestRecordStore()
        for index in 0..<200 {
            store.append(QWRequestRecord(
                id: "req-\(index)", time: Date(), modelId: "m",
                promptTokens: index, generatedTokens: 1))
        }
        XCTAssertEqual(store.recent(1_000).count, QWRequestRecordStore.capacity)
        XCTAssertEqual(store.recent(1_000).count, 64)
        // The newest survive; the oldest are forgotten.
        XCTAssertEqual(store.recent(1).last?.promptTokens, 199)
    }

    /// The record's own fields: a list of sizes, timings and names. A
    /// field that could hold the request's text would have to be added
    /// here on purpose.
    func testTheRecordHasNoFieldThatCouldHoldThePrompt() throws {
        let record = QWRequestRecord(
            id: "req-1", time: Date(), modelId: "m",
            promptTokens: 10, generatedTokens: 5, cachedTokens: 2,
            cacheHitTier: "warm_ram", ttftMs: 12, prefillMs: 30,
            decodeMs: 90, decodeTps: 55, stageGPU: ["prefill": 4],
            config: ["kvMode": "f16"], outcome: "ok")
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let json = try XCTUnwrap(String(data: try encoder.encode(record), encoding: .utf8))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        XCTAssertEqual(Set(object.keys), ["id", "time", "modelId", "promptTokens", "generatedTokens",
                                          "cachedTokens", "cacheHitTier", "ttftMs", "prefillMs",
                                          "decodeMs", "decodeTps", "stageGPU", "config", "outcome"])
    }

    /// A timing nobody measured is absent, not zero.
    func testAnUnmeasuredTimingStaysAbsent() {
        let record = QWRequestRecord(
            id: "req-1", time: Date(), modelId: "m",
            promptTokens: 10, generatedTokens: 5)
        XCTAssertNil(record.ttftMs)
        XCTAssertNil(record.prefillMs)
        XCTAssertNil(record.decodeMs)
        XCTAssertNil(record.decodeTps)
        XCTAssertNil(record.cacheHitTier)
        XCTAssertTrue(record.stageGPU.isEmpty)
    }

    /// The ids are a sequence, not a secret, and they do not repeat.
    func testTheRequestIdsAreASequence() {
        let store = QWRequestRecordStore()
        XCTAssertEqual(store.nextId(), "req-1")
        XCTAssertEqual(store.nextId(), "req-2")
        XCTAssertNotEqual(store.nextId(), store.nextId())
    }
}
