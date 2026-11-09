// QWKnobValueRoundTripTests.swift — The knob value's wire shape.
// A profile file is written by the app and read by the app, and sometimes
// by a person. The value must survive the round trip and say what it is.

import XCTest
import QWCore

final class QWKnobValueRoundTripTests: XCTestCase {
    func testEveryKindSurvivesTheRoundTrip() throws {
        let values: [QWKnobValue] = [
            .int(512), .int(-1), .int(0),
            .double(0.7), .double(0.0), .double(-2.5),
            .bool(true), .bool(false),
            .string(""), .string("chunked"), .string("low"),
        ]
        for value in values {
            let data = try JSONEncoder().encode(["v": value])
            let decoded = try JSONDecoder().decode([String: QWKnobValue].self, from: data)
            XCTAssertEqual(decoded["v"], value, "\(value) did not survive the round trip")
        }
    }

    /// The payload says which kind it is: a number alone would not say
    /// whether it was an integer or a double.
    func testThePayloadNamesTheKind() throws {
        let data = try JSONEncoder().encode(["v": QWKnobValue.double(0.7)])
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let wrapped = try XCTUnwrap(json["v"] as? [String: Any])
        XCTAssertEqual(wrapped["double"] as? Double, 0.7)
        XCTAssertNil(wrapped["int"])
    }

    /// A payload that names no kind is refused, not guessed at.
    func testAPayloadWithoutAKindIsRefused() {
        let json = #"{"v":{"number":1}}"#.data(using: .utf8)!
        XCTAssertThrowsError(try JSONDecoder().decode([String: QWKnobValue].self, from: json))
    }
}
