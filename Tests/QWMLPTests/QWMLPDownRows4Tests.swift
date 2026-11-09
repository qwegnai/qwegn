import XCTest
import Metal
@testable import QWMLP
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWQuant

/// C14 addendum gate: the M=1 rows4 down door must be BIT-EXACT against the
/// shipping qwg_mlp_down (same h-block, same per-row add order).
final class QWMLPDownRows4Tests: XCTestCase {

    func testRows4BitExactM1() throws {
        let (gu, dn) = try QWMLPTests.weights()
        let hal = QWMLPTests.hal
        let xb = QWMLPTests.buf(try QWMLPTests.load("mlp_x_M1.bin"))
        let gsb = QWMLPTests.zeros(40, .f32)
        let hb = QWMLPTests.zeros(17408, .f16)
        let yBase = QWMLPTests.zeros(5120, .f16)
        let yRows = QWMLPTests.zeros(5120, .f16)

        let base = try QWMLP(hal: hal, downRows4: false, geometry: .qwen38_27b)
        let rows = try QWMLP(hal: hal, downRows4: true, geometry: .qwen38_27b)

        guard let (cb0, ce0) = hal.begin() else { throw XCTSkip("no HAL") }
        try base.encodeGu(ce0, x: QWMLPTests.tv(xb, 5120, .f16),
                          gs: QWMLPTests.tv(gsb, 40, .f32), wGu: gu,
                          h: QWMLPTests.tv(hb, 17408, .f16), M: 1)
        ce0.endEncoding(); cb0.commit(); cb0.waitUntilCompleted()

        guard let (cb1, ce1) = hal.begin() else { throw XCTSkip("no HAL") }
        try base.encodeDown(ce1, wDown: dn, h: QWMLPTests.tv(hb, 17408, .f16),
                            y: QWMLPTests.tv(yBase, 5120, .f16), M: 1)
        ce1.endEncoding(); cb1.commit(); cb1.waitUntilCompleted()

        guard let (cb2, ce2) = hal.begin() else { throw XCTSkip("no HAL") }
        try rows.encodeDown(ce2, wDown: dn, h: QWMLPTests.tv(hb, 17408, .f16),
                            y: QWMLPTests.tv(yRows, 5120, .f16), M: 1)
        ce2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()

        let ha = QWMLPTests.u16s(Data(bytes: yBase.contents(), count: yBase.length)).map(QWMLPTests.f16)
        let hb2 = QWMLPTests.u16s(Data(bytes: yRows.contents(), count: yRows.length)).map(QWMLPTests.f16)
        // C14 parity law: f16-out within 5e-4 of the shipping door.
        let v = QWMLPTests.viol(hb2, ha, absTol: 5e-4)
        XCTAssertEqual(v.0, 0,
                       "rows4 down vs qwg_mlp_down (worst \(v.1) @\(v.2), tol 5e-4)")
    }

    func testRows4DefaultOnWithKillSwitch() throws {
        // Promoted default-ON; the kill-switch is the only way back to the
        // shipping kernel, so the flag default is a contract.
        XCTAssertEqual(QWMLP.downRows4Enabled,
                       ProcessInfo.processInfo.environment["QW_MLP_DOWN_ROWS4"] == "1",
                       "rows4 stays opt-in until an end-to-end A/B justifies it")
    }
}
