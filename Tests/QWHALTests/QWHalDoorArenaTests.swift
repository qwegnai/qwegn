import XCTest
import Metal
@testable import QWHAL

/// C02 addendum gates: batched doors must not drain, the door stack must not
/// hash/allocate, and post-freeze pipeline hits must take no lock.
final class QWHalDoorArenaTests: XCTestCase {

    private func makeHAL() throws -> QWHal {
        try QWHal(metallibPath: QWHALTests.anyMetallib(), geometry: .qwen38_27b)
    }

    func testBatchedDoorsDrainOncePerFlush() throws {
        let hal = try makeHAL()
        guard QWHal.doorBatchEnabled else { throw XCTSkip("door batching disabled") }
        hal.resetDoorDiagnostics()

        hal.beginDoors()
        for _ in 0..<64 {
            guard let enc = hal.doorBegin() else { return XCTFail("doorBegin nil") }
            hal.doorEnd(enc)
        }
        XCTAssertEqual(hal.doorDrainCount, 0, "a batched door must never commit+wait")
        XCTAssertEqual(hal.doorSubmissionCount, 64)

        hal.flushDoors(site: "test")
        XCTAssertEqual(hal.doorDrainCount, 1, "one commit+wait per flush")
        XCTAssertEqual(hal.doorSubmissionCount, 64)
    }

    func testStandaloneDoorsDrainIndividually() throws {
        let hal = try makeHAL()
        hal.resetDoorDiagnostics()
        for _ in 0..<3 {
            guard let enc = hal.doorBegin() else { return XCTFail("doorBegin nil") }
            hal.doorEnd(enc)
        }
        XCTAssertEqual(hal.doorDrainCount, 3, "unbatched door = standalone drain")
    }

    func testFrozenPipelineHitsTakeNoLock() throws {
        let hal = try makeHAL()
        _ = try hal.pipeline("qw_probe")
        hal.freezePipelines()
        let before = hal.pipelineLockAcquisitions
        for _ in 0..<100 { _ = try hal.pipeline("qw_probe") }
        XCTAssertEqual(hal.pipelineLockAcquisitions, before,
                       "post-freeze hits must short-circuit before the lock")
    }

    func testDoorStackBalancesAcrossNesting() throws {
        let hal = try makeHAL()
        guard QWHal.doorBatchEnabled else { throw XCTSkip("door batching disabled") }
        hal.resetDoorDiagnostics()
        hal.beginDoors()
        let outer = hal.doorBegin()
        XCTAssertNotNil(outer)
        hal.beginDoors()
        let inner = hal.doorBegin()
        XCTAssertNotNil(inner)
        if let inner { hal.doorEnd(inner) }
        if let outer { hal.doorEnd(outer) }
        hal.closeDoors()
        hal.closeDoors()
        XCTAssertEqual(hal.doorDrainCount, 1)
    }
}
