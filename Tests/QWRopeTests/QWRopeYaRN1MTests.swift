import XCTest
import Foundation
@testable import QWRope
import QWCore

final class QWRopeYaRN1MTests: XCTestCase {

    func testYaRN1MConfigurationAndScaleFactor() {
        // Scaling from 256K (262,144) to 1M (1,048,576) requires factor = 4.0
        let yarn = QWYaRNConfig.config1M()
        XCTAssertEqual(yarn.factor, 4.0, accuracy: 1e-5)
        XCTAssertEqual(yarn.originalMaxPosition, 262_144)

        // get_mscale = 0.1 * ln(factor) + 1.0
        // for factor = 4.0: 0.1 * ln(4) + 1.0 = 0.1 * 1.38629 + 1.0 = 1.13863
        let (invFreqs, scaling) = QWRopeTables.invFreqYarn(yarn, geometry: .qwen38_27b)
        XCTAssertEqual(scaling, Float(0.1 * log(4.0) + 1.0), accuracy: 1e-5)
        XCTAssertEqual(invFreqs.count, 32)

        // Verify frequency bands:
        // Lower dimensions (higher frequency) should remain close to default invFreq
        // Higher dimensions (lower frequency) should be scaled down by approx 1/factor
        let defFreqs = QWRopeTables.invFreqDefault()
        XCTAssertEqual(invFreqs[0], defFreqs[0], accuracy: 1e-6, "High frequency pairs should be unmodified")
        XCTAssertLessThan(invFreqs[31], defFreqs[31], "Low frequency pairs must be scaled down")
        XCTAssertEqual(invFreqs[31], defFreqs[31] / yarn.factor, accuracy: 1e-4, "Highest pair must be fully interpolated")
    }

    func testYaRN1MTableGenerationAndBoundaryPositions() {
        let yarn = QWYaRNConfig.config1M()
        // Generate a compact test table of 1024 rows with 1M YaRN config
        let tables = QWRopeTables(rows: 1024, yarn: yarn, geometry: .qwen38_27b)
        XCTAssertEqual(tables.rows, 1024)
        XCTAssertEqual(tables.cosSin.count, 1024 * 64)

        // Position 0: cos = scaling, sin = 0
        let row0 = tables.cosSinRow(0)
        let scaling = tables.attentionScaling
        for i in 0..<32 {
            XCTAssertEqual(row0[i], scaling, accuracy: 1e-5)
            XCTAssertEqual(row0[32 + i], 0.0, accuracy: 1e-5)
        }

        // Test mRoPE position row for vision at 1M scale
        let mropeRow = tables.cosSinRowMrope((t: 500, h: 250, w: 750))
        XCTAssertEqual(mropeRow.count, 64)
        for i in 0..<64 {
            XCTAssertTrue(mropeRow[i].isFinite)
        }
    }
}
