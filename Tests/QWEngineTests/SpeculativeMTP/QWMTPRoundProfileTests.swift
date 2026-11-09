import Foundation
import XCTest
@testable import QWEngine

final class QWMTPRoundProfileTests: XCTestCase {
    func testTimestampAndStageValidation() throws {
        XCTAssertEqual(try QWMTPStageTiming.gpuDurationMs(start: 2, end: 2.025), 25, accuracy: 1e-9)
        for (start, end) in [(0.0, 1.0), (2, 1), (1, 1), (.nan, 2), (1, .infinity)] {
            XCTAssertThrowsError(try QWMTPStageTiming.gpuDurationMs(start: start, end: end))
        }
        for (host, gpu, buffers) in [(0.0, 1.0, 1), (1, 0, 1), (1, 2, 1),
                                     (2, 1, 0), (.nan, 1, 1), (2, .infinity, 1)] {
            XCTAssertThrowsError(try QWMTPStageTiming(hostMs: host, gpuMs: gpu, commandBuffers: buffers))
        }
    }

    func testRoundAccountingAndSerialization() throws {
        let draft = try QWMTPStageTiming(hostMs: 8, gpuMs: 6, commandBuffers: 2)
        let verify = try QWMTPStageTiming(hostMs: 32, gpuMs: 30, commandBuffers: 1)
        let resolve = try QWMTPStageTiming(hostMs: 2, gpuMs: 1, commandBuffers: 1)
        let profile = QWMTPRoundProfile(drafts: 2, retainedRows: 1,
                                       draft: draft, verification: verify, resolution: resolve,
                                       verificationGpuMsByStage: ["head": 6, "mlpGateUp": 12],
                                       adaptiveEntropyByDraft: [0.25, 0.75])
        XCTAssertEqual(profile.hostMs, 42)
        XCTAssertEqual(profile.gpuMs, 37)
        XCTAssertEqual(profile.verificationGpuMsByStage?["head"], 6)
        XCTAssertEqual(profile.adaptiveEntropyByDraft, [0.25, 0.75])
        XCTAssertEqual(try JSONDecoder().decode(QWMTPRoundProfile.self,
            from: JSONEncoder().encode(profile)), profile)
    }

    func testBackendIdentityRoundTrips() throws {
        // L0: the profile carries the backend that produced the round;
        // pre-backend reports (nil) still decode for evidence compatibility.
        let draft = try QWMTPStageTiming(hostMs: 8, gpuMs: 6, commandBuffers: 2)
        let verify = try QWMTPStageTiming(hostMs: 32, gpuMs: 30, commandBuffers: 1)
        let resolve = try QWMTPStageTiming(hostMs: 2, gpuMs: 1, commandBuffers: 1)
        let profile = QWMTPRoundProfile(drafts: 2, retainedRows: 1,
                                       draft: draft, verification: verify, resolution: resolve,
                                       backend: QWMTPBackend.dflash2.rawValue)
        XCTAssertEqual(profile.backend, "dflash2")
        let decoded = try JSONDecoder().decode(QWMTPRoundProfile.self, from: JSONEncoder().encode(profile))
        XCTAssertEqual(decoded, profile)
        // A legacy record without the field decodes as nil. JSONEncoder's
        // key order is not guaranteed, so strip the field with or without
        // a leading comma.
        var json = String(decoding: try JSONEncoder().encode(profile), as: UTF8.self)
        if let stripped = json.range(of: ",\"backend\":\"dflash2\"") {
            json.removeSubrange(stripped)
        } else if let stripped = json.range(of: "\"backend\":\"dflash2\",") {
            json.removeSubrange(stripped)
        }
        XCTAssertFalse(json.contains("\"backend\""), "legacy strip failed: \(json)")
        let legacyDecoded = try JSONDecoder().decode(QWMTPRoundProfile.self, from: Data(json.utf8))
        XCTAssertNil(legacyDecoded.backend)
        XCTAssertNotEqual(legacyDecoded, profile)
    }
}
