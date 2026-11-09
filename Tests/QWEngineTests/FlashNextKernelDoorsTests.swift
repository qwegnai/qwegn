import Foundation
import XCTest
@testable import QWEngine

final class FlashNextKernelDoorsTests: XCTestCase {
    func testRequiredDoorsIncludeStridedBatchKernels() {
        XCTAssertTrue(FlashNextModule.requiredDoors.contains("qwg_hc_norm_strided"))
        XCTAssertTrue(FlashNextModule.requiredDoors.contains("qwg_ple_gate_strided"))
        XCTAssertTrue(FlashNextModule.requiredDoors.contains("qwg_hc_norm"))
        XCTAssertTrue(FlashNextModule.requiredDoors.contains("qwg_ple_gate"))
    }
}
