import Foundation
import XCTest
@testable import QWEngine

final class QWGenerationRequestGateTests: XCTestCase {
    func testOwnerRawNestedForeignAndStalePermits() throws {
        let gate = QWGenerationRequestGate(), other = QWGenerationRequestGate()
        let flight = NSLock(), otherFlight = NSLock()
        var old: QWGenerationPermit?
        try gate.validate()
        try gate.withRequest(flight: flight) { permit in
            old = permit
            XCTAssertTrue(flight.try(), "callbacks must not hold flight")
            flight.unlock()
            try gate.validate(permit)
            XCTAssertThrowsError(try gate.validate())
            XCTAssertThrowsError(try gate.withRequest(flight: flight) { _ in })
            try other.withRequest(flight: otherFlight) { foreign in
                XCTAssertThrowsError(try gate.validate(foreign))
                XCTAssertThrowsError(try other.validate(permit))
            }
        }
        try gate.validate()
        XCTAssertThrowsError(try gate.validate(old))
        try gate.withRequest(flight: flight) { permit in
            XCTAssertThrowsError(try gate.validate(old))
            try gate.validate(permit)
        }
    }

    func testThrowReleasesOwnerAndBusyFlightDoesNotReserve() throws {
        enum Failure: Error { case injected }
        let gate = QWGenerationRequestGate(), flight = NSLock()
        flight.lock()
        XCTAssertThrowsError(try gate.withRequest(flight: flight) { _ in XCTFail("must not enter") })
        flight.unlock()
        try gate.validate()
        XCTAssertThrowsError(try gate.withRequest(flight: flight) { _ in throw Failure.injected })
        try gate.validate()
        try gate.withRequest(flight: flight) { try gate.validate($0) }
    }
}
