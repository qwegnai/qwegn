import XCTest
import Metal
@testable import QWHAL

/// C02 bench/gate (specs/02-qwhal.md): device facts must match
/// docs/HARDWARE_FACTS.md; pipeline cache must be O(1) after first use;
/// measured-BW constants must equal the frozen S3 numbers.
final class QWHALTests: XCTestCase {

    static let hal: QWHal = {
        // Tests run without a metallib; HAL init requires one, so point at
        // any metallib the test bundle can reach, else skip gracefully.
        try! QWHal(metallibPath: QWHALTests.anyMetallib(), geometry: .qwen38_27b)
    }()

    static func anyMetallib() -> String {
        // Build a trivial metallib on the fly into the temp dir.
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void qw_probe(device float* out [[buffer(0)]]) { out[0] = 1; }
        """
        let tmp = FileManager.default.temporaryDirectory
        let msl = tmp.appendingPathComponent("qw_probe.metal")
        let lib = tmp.appendingPathComponent("qw_probe.metallib")
        try! src.write(to: msl, atomically: true, encoding: .utf8)
        try? FileManager.default.removeItem(at: lib)   // rebuild always (cheap)
        if true {
            // Two-step: metal -c → .air, metallib → .metallib (a single
            // `metal -o lib` also works; two steps give clear errors).
            let air = tmp.appendingPathComponent("qw_probe.air")
            let c = Process()
            c.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
            c.arguments = ["-sdk", "macosx", "metal", "-c", msl.path, "-o", air.path]
            try! c.run(); c.waitUntilExit()
            precondition(c.terminationStatus == 0, "metal compile failed")
            let l = Process()
            l.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
            l.arguments = ["-sdk", "macosx", "metallib", air.path, "-o", lib.path]
            try! l.run(); l.waitUntilExit()
            precondition(l.terminationStatus == 0, "metallib link failed")
        }
        return lib.path
    }

    func testDeviceFactsMatchHardwareDoc() throws {
        let caps = Self.hal.caps
        XCTAssertEqual(caps.name, "Apple M5 Max",
                       "HAL must run on the budgeted chip (HARDWARE_FACTS.md)")
        XCTAssertTrue(caps.unifiedMemory)
        XCTAssertTrue(caps.familyApple9)
        XCTAssertTrue(caps.familyApple10)
        XCTAssertFalse(caps.familyApple11, "apple11 must be absent per probe")
        XCTAssertGreaterThan(caps.maxWorkingSetBytes, 60_000_000_000,
                             "55.56 GB checkpoint must be GPU-resident per HARDWARE_FACTS")
        XCTAssertEqual(caps.measuredReadBWBps, 585e9, accuracy: 1e9)
        XCTAssertEqual(caps.measuredL2BWBps, 950e9, accuracy: 50e9)
    }

    func testPipelineCacheIsCached() throws {
        let hal = Self.hal
        let p1 = try hal.pipeline("qw_probe")
        let p2 = try hal.pipeline("qw_probe")
        XCTAssertTrue(p1 === p2, "pipeline(name) must return the cached PSO")
        XCTAssertThrowsError(try hal.pipeline("no_such_kernel_xyz")) { err in
            guard case QWHalError.kernelMissing(let n) = err else {
                return XCTFail("wrong error: \(err)")
            }
            XCTAssertEqual(n, "no_such_kernel_xyz")
        }
    }

    func testBeginEndRoundtrip() throws {
        guard let (cb, enc) = Self.hal.begin() else { return XCTFail("begin nil") }
        let p = try Self.hal.pipeline("qw_probe")
        var out: Float = 0
        enc.setComputePipelineState(p)
        let buf = Self.hal.device.makeBuffer(length: 4, options: .storageModeShared)!
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        out = buf.contents().load(as: Float.self)
        XCTAssertEqual(out, 1.0)
    }

    func testDefaultMetallibPathResolvesInsideAppBundle() throws {
        if ProcessInfo.processInfo.environment["QW_METALLIB"] != nil { return }
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory.appendingPathComponent("qwhal_bundle_\(UUID().uuidString)", isDirectory: true)
        defer { try? fm.removeItem(at: tmp) }
        let emptyBin = tmp.appendingPathComponent("usr/local/bin", isDirectory: true)
        let appBundle = tmp.appendingPathComponent("Applications/Qwegn.app", isDirectory: true)
        let appRes = appBundle.appendingPathComponent("Contents/Resources", isDirectory: true)
        try fm.createDirectory(at: emptyBin, withIntermediateDirectories: true)
        try fm.createDirectory(at: appRes, withIntermediateDirectories: true)
        let rotLib = appRes.appendingPathComponent("kernel.metallib")
        let stdLib = appRes.appendingPathComponent("kernel_std.metallib")
        try Data([0x4D, 0x54, 0x4C, 0x42]).write(to: rotLib)
        try Data([0x4D, 0x54, 0x4C, 0x42]).write(to: stdLib)

        let resolvedRot = QWHal.defaultMetallibPath(basis: .rotated, appBundleURL: appBundle, searchDirs: [emptyBin])
        let resolvedStd = QWHal.defaultMetallibPath(basis: .standard, appBundleURL: appBundle, searchDirs: [emptyBin])
        XCTAssertEqual(resolvedRot, rotLib.path)
        XCTAssertEqual(resolvedStd, stdLib.path)
    }
}
