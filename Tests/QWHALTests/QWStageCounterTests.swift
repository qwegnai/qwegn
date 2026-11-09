import XCTest
import Metal
@testable import QWHAL

/// A1 (docs/PERF_REWORK_REVIEW_2026-10-18.md): QW_GPU_STAGES door sampling must
/// attribute GPU microseconds per region without host drains, and the default
/// (sampling off) door behavior must be untouched.
final class QWStageCounterTests: XCTestCase {

    private func dispatchProbe(_ hal: QWHal, _ enc: any MTLComputeCommandEncoder) throws {
        let p = try hal.pipeline("qw_probe")
        let buf = hal.device.makeBuffer(length: 4, options: .storageModeShared)!
        enc.setComputePipelineState(p)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    }

    func testStageSamplingAttributesPerRegion() throws {
        let hal = QWHALTests.hal
        guard QWGPUStageProfiler<String>.isSupported(on: hal.device) else { return }
        _ = hal.takeStageMicros()
        hal.stageSamplingEnabled = true
        defer { hal.stageSamplingEnabled = false }
        hal.beginDoors()
        for i in 0..<4 {
            guard let e = hal.doorBegin(region: "probe-\(i)") else {
                return XCTFail("doorBegin nil with sampling on")
            }
            try dispatchProbe(hal, e)
            hal.doorEnd(e)
        }
        hal.closeDoors()
        let micros = hal.takeStageMicros()
        XCTAssertEqual(Set(micros.keys), Set(["probe-0", "probe-1", "probe-2", "probe-3"]),
                       "every door region must appear in the GPU attribution")
    }

    func testSamplingDefaultOffKeepsSharedEncoder() throws {
        let hal = QWHALTests.hal
        XCTAssertFalse(QWHal.stageSamplingDefault, "QW_GPU_STAGES must default OFF")
        hal.beginDoors()
        defer { hal.closeDoors() }
        var seen: [(any MTLComputeCommandEncoder)] = []
        for _ in 0..<2 {
            guard let e = hal.doorBegin(region: "a") else { return XCTFail("doorBegin nil") }
            try dispatchProbe(hal, e)
            hal.doorEnd(e)   // balanced: every door ends before the next begins
            seen.append(e)
        }
        XCTAssertTrue(seen[0] === seen[1], "unsampled doors must reuse the batch encoder (C29 law)")
    }

    func testResolveOnEmptyWindowIsNoOp() throws {
        let hal = QWHALTests.hal
        _ = hal.takeStageMicros()
        hal.beginDoors()
        hal.closeDoors()
        XCTAssertTrue(hal.takeStageMicros().isEmpty)
    }
}
