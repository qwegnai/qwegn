// QWBF16Tests.swift — Unit tests for IEEE-754 round-to-nearest-even float-to-bfloat16.

import XCTest
@testable import QWQuant

final class QWBF16Tests: XCTestCase {
    func testExactValues() {
        XCTAssertEqual(QWBF16.fromFloatRNE(0.0), 0x0000)
        XCTAssertEqual(QWBF16.fromFloatRNE(-0.0), 0x8000)
        XCTAssertEqual(QWBF16.fromFloatRNE(1.0), 0x3F80)
        XCTAssertEqual(QWBF16.fromFloatRNE(-1.0), 0xBF80)
        XCTAssertEqual(QWBF16.fromFloatRNE(Float.infinity), 0x7F80)
        XCTAssertEqual(QWBF16.fromFloatRNE(-Float.infinity), 0xFF80)
    }

    func testRoundToNearestEvenHalfway() {
        // Halfway between 0x3F80 and 0x3F81: 0x3F80_8000
        // LSB is 0 (even), so it rounds DOWN to 0x3F80.
        let halfwayEven = Float(bitPattern: 0x3F80_8000)
        XCTAssertEqual(QWBF16.fromFloatRNE(halfwayEven), 0x3F80)

        // Halfway between 0x3F81 and 0x3F82: 0x3F81_8000
        // LSB is 1 (odd), so it rounds UP to 0x3F82 (even).
        let halfwayOdd = Float(bitPattern: 0x3F81_8000)
        XCTAssertEqual(QWBF16.fromFloatRNE(halfwayOdd), 0x3F82)
    }

    func testRoundOffHalfway() {
        // Slightly above halfway: 0x3F80_8001 -> rounds UP to 0x3F81
        let aboveHalfway = Float(bitPattern: 0x3F80_8001)
        XCTAssertEqual(QWBF16.fromFloatRNE(aboveHalfway), 0x3F81)

        // Slightly below halfway: 0x3F80_7FFF -> rounds DOWN to 0x3F80
        let belowHalfway = Float(bitPattern: 0x3F80_7FFF)
        XCTAssertEqual(QWBF16.fromFloatRNE(belowHalfway), 0x3F80)
    }

    func testNaNHandling() {
        let nan = Float.nan
        let bfNan = QWBF16.fromFloatRNE(nan)
        // Exponent must be all 1s and mantissa non-zero
        XCTAssertEqual(bfNan & 0x7F80, 0x7F80)
        XCTAssertNotEqual(bfNan & 0x007F, 0)
    }

    func testToFloatRoundTrip() {
        let original = Float(1.5)
        let bf = QWBF16.fromFloatRNE(original)
        let roundTripped = QWBF16.toFloat(bf)
        XCTAssertEqual(roundTripped, original)
    }
}
