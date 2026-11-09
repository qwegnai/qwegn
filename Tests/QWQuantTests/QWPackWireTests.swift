// QWPackWireTests.swift — residency-phase wire law (mlock of the pack
// mmap; the exact inverse of adviseDontNeed). CPU/OS-only: no Metal, no
// GPU window, safe to run while a serving model is resident.
// Full note: docs/STYLE_NOTES.md#QWPackWireTests.swift.
import XCTest
@testable import QWQuant

final class QWPackWireTests: XCTestCase {

    private func fixture(_ name: String) -> URL {
        var d = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<8 {
            let c = d.appendingPathComponent("Tests/QWQuantTests/fixtures/\(name)")
            if FileManager.default.fileExists(atPath: c.path) { return c }
            d = d.deletingLastPathComponent()
        }
        fatalError("\(name) fixture missing — rebuild via Scripts/quantize/pack.py --selfpack")
    }

    func testWireFullAndIdempotent() throws {
        let pack = try QWQuant.Pack(url: fixture("selfpack.qwp"))
        let total = pack.packedByteCount
        XCTAssertGreaterThan(total, 0)
        XCTAssertEqual(total, (try Data(contentsOf: fixture("selfpack.qwp"))).count,
                       "packedByteCount must be the mmap span readers bound")
        let w1 = pack.wire()
        let w2 = pack.wire()                       // idempotent
        XCTAssertEqual(w2, w1)
        XCTAssertEqual(pack.wiredBytes, w1)        // telemetry parity
        XCTAssertTrue(w1 == 0 || w1 == total,
                      "small fixture must wire fully or refuse cleanly, got \(w1)/\(total)")
    }

    func testWireCapIsHard() throws {
        let pack = try QWQuant.Pack(url: fixture("selfpack.qwp"))
        let cap = pack.packedByteCount / 2
        let w = pack.wire(limitBytes: cap)
        XCTAssertLessThanOrEqual(w, cap, "wire must never exceed limitBytes")
        XCTAssertEqual(pack.wiredBytes, w)
        XCTAssertTrue(w == 0 || w == cap,
                      "single-chunk fixture must pin exactly the cap or refuse, got \(w)")
    }

    func testWireZeroCapTouchesNothing() throws {
        let pack = try QWQuant.Pack(url: fixture("selfpack.qwp"))
        XCTAssertEqual(pack.wire(limitBytes: 0), 0)
        XCTAssertEqual(pack.wiredBytes, 0)
    }

    func testWireNeverErrorsOnLiveFile() throws {
        // The wire must survive reading a pack whose pages the OS already
        // owns (second open of the same file — the engine+test co-residency
        // class): always 0 ≤ wired ≤ span, never a throw, never a signal.
        let url = fixture("selfpack.qwp")
        let a = try QWQuant.Pack(url: url)
        let b = try QWQuant.Pack(url: url)
        let wa = a.wire()
        let wb = b.wire()
        XCTAssertTrue(wa >= 0 && wa <= a.packedByteCount)
        XCTAssertTrue(wb >= 0 && wb <= b.packedByteCount)
    }
}
