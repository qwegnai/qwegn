// QWHQ8PackTests.swift — HQ8_G32_M5 planar entry fields, plane extents,
// and CPU rot-domain dequant vs the packer golden. CPU-only: safe under a
// resident GPU tenant. Full note: docs/STYLE_NOTES.md#QWHQ8PackTests.swift.
import XCTest
@testable import QWQuant

final class QWHQ8PackTests: XCTestCase {

    private func fixture(_ name: String) -> URL {
        var d = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<8 {
            let c = d.appendingPathComponent("Tests/QWQuantTests/fixtures/\(name)")
            if FileManager.default.fileExists(atPath: c.path) { return c }
            d = d.deletingLastPathComponent()
        }
        fatalError("\(name) fixture missing — rebuild via Scripts/quantize/pack.py --selfpack-hq8")
    }

    func testEntryFieldsAndPlaneExtents() throws {
        let pack = try QWQuant.Pack(url: fixture("selfpack_hq8.qwp"))
        let e = try pack.entry("w_test8")
        XCTAssertEqual(e.shape, [64, 256])
        XCTAssertNotNil(e.hq8Off)
        XCTAssertNotNil(e.hq8SOff)
        XCTAssertNil(e.hq4Off)
        XCTAssertEqual(e.hq8PayloadBytes, 64 * 256)
        XCTAssertEqual(e.hq8ScaleBytes, 64 * 256 / 32 * 2)
    }

    func testDequantRotMatchesPackerGolden() throws {
        let pack = try QWQuant.Pack(url: fixture("selfpack_hq8.qwp"))
        let got = try pack.dequantHQ8Rot("w_test8")
        XCTAssertEqual(got.count, 64 * 256)
        let golden = try Data(contentsOf: fixture("selfpack_hq8.qwp.dq8rot.f32.bin"))
        XCTAssertEqual(golden.count, 64 * 256 * 4)
        var worst: Float = 0
        golden.withUnsafeBytes { g in
            let ref = g.bindMemory(to: Float32.self)
            for i in 0..<got.count {
                worst = max(worst, abs(got[i] - ref[i]))
            }
        }
        XCTAssertLessThanOrEqual(worst, 1e-6, "rot-domain dequant must match packer golden")
    }

    func testLegacySelfpackHasNoHQ8Entries() throws {
        let pack = try QWQuant.Pack(url: fixture("selfpack.qwp"))
        let e = try pack.entry("w_test")
        XCTAssertNil(e.hq8Off)
        XCTAssertNotNil(e.hq4Off)
    }
}
