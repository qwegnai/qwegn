import Foundation
import XCTest
@testable import QWFlashNext

final class FNGPURouteTraceTests: XCTestCase {
    func testRowsRetainOrderAndPrunedRoutes() throws {
        let report = try FNGPURouteTrace.Report(phase: "verify", position: 63, rows: 2,
                                               layer: 1, experts: 4, routes: 2, ids: [3, 0, 3, -1])
        let data = try JSONEncoder().encode(report)
        let decoded = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(decoded["ids"] as? [Int], [3, 0, 3, -1])
        XCTAssertEqual(decoded["schema"] as? Int, 1)
        XCTAssertEqual(decoded["rows"] as? Int, 2)
        XCTAssertEqual(decoded["layer"] as? Int, 1)
    }

    func testInvalidRouteShapesAndIDsFailExplicitly() {
        for (rows, experts, routes, ids) in [
            (0, 4, 2, [Int32]()), (1, 0, 2, [0, 1]), (1, 4, 0, []),
            (2, 4, 2, [0, 1]), (1, 4, 2, [0, 4]), (1, 4, 2, [-2, 1]),
        ] {
            XCTAssertThrowsError(try FNGPURouteTrace.Report(
                phase: "decode", position: 0, rows: rows, layer: 1,
                experts: experts, routes: routes, ids: ids))
        }
    }
}
