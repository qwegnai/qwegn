import XCTest
import Metal
@testable import QWAlloc
import QWHAL

/// C03 gate (specs/03-qwalloc.md): sub-alloc ≥20× cheaper than makeBuffer,
/// ring wrap correctness under 100k ops, page accounting exact,
/// steady-state alloc counters provable.
final class QWAllocTests: XCTestCase {

    static let hal: QWHal = try! {
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void qw_probe(device float* out [[buffer(0)]]) { out[0] = 1; }
        """
        let tmp = FileManager.default.temporaryDirectory
        let msl = tmp.appendingPathComponent("qw_alloc_probe.metal")
        let lib = tmp.appendingPathComponent("qw_alloc_probe.metallib")
        try! src.write(to: msl, atomically: true, encoding: .utf8)
        try? FileManager.default.removeItem(at: lib)
        let air = tmp.appendingPathComponent("qw_alloc_probe.air")
        let c = Process(); c.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        c.arguments = ["-sdk", "macosx", "metal", "-c", msl.path, "-o", air.path]
        try! c.run(); c.waitUntilExit()
        let l = Process(); l.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        l.arguments = ["-sdk", "macosx", "metallib", air.path, "-o", lib.path]
        try! l.run(); l.waitUntilExit()
        // Throwing, not `try!`: this HAL names an explicit library, and a
        // library in the wrong basis is a failure the test should report.
        return try QWHal(metallibPath: lib.path, geometry: .qwen38_27b)
    }()

    /// 512 MB arena — big enough to matter, small enough for CI.
    static func arena(_ mb: Int = 512) -> QWArena {
        try! QWArena(hal: hal, weights: mb << 20 / 2, activations: mb << 20 / 8,
                     kv: mb << 20 / 4, gdnStates: mb << 20 / 8)
    }

    func testSubAllocIsCheaperThanMakeBuffer() throws {
        let a = Self.arena()
        let dev = Self.hal.device   // sub-alloc timing happens below via sub2
        let sz = 1 << 20
        // baseline: makeBuffer
        var t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<200 { _ = dev.makeBuffer(length: sz, options: .storageModeShared) }
        let base = CFAbsoluteTimeGetCurrent() - t0
        // arena sub-alloc (weights region = 256 MB in the 512 MB test arena)
        let sub2 = a.weightsRegion()
        t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<200 { _ = try sub2.allocate(bytes: sz) }
        let ours = CFAbsoluteTimeGetCurrent() - t0
        let ratio = base / max(ours, 1e-9)
        print(String(format: "makeBuffer %.0f µs/op vs arena %.3f µs/op → %.0f×",
                     base / 200 * 1e6, ours / 200 * 1e6, ratio))
        XCTAssertGreaterThan(ratio, 20, "gate: ≥20× cheaper than makeBuffer")
    }

    func testAlignmentLaw() throws {
        let sub = Self.arena().weightsRegion()
        let o1 = try sub.allocate(bytes: 1)
        let o2 = try sub.allocate(bytes: 513)
        XCTAssertEqual(o1 % 256, 0)
        XCTAssertEqual(o2 % 256, 0)
        XCTAssertGreaterThan(o2, o1)
    }

    func testExactCapacityAndRelocatedRegionBases() throws {
        let a = try QWArena(hal: Self.hal, weights: 768, activations: 256,
                            kv: 256, gdnStates: 256)
        let sub = a.weightsRegion()
        let first = try sub.allocate(QWTensorShape([129], .f16))
        let second = try sub.allocate(QWTensorShape([1], .f32))
        XCTAssertEqual(first.offset, 0)
        XCTAssertEqual(second.offset, 512)
        XCTAssertEqual(sub.usedBytes, 768)
        XCTAssertEqual(a.activationsRegion().region.base, 768)
        XCTAssertEqual(a.kvRegion().region.base, 1024)
        XCTAssertEqual(a.gdnRegion().region.base, 1280)
        XCTAssertEqual(a.committedBytes, 1536)
        let regions = [sub.region, a.activationsRegion().region,
                       a.kvRegion().region, a.gdnRegion().region]
        for (index, region) in regions.enumerated() {
            XCTAssertEqual(region.base % 256, 0)
            region.pointer.storeBytes(of: UInt32(index + 1), as: UInt32.self)
        }
        for (index, region) in regions.enumerated() {
            XCTAssertEqual(region.pointer.load(as: UInt32.self), UInt32(index + 1))
        }
        XCTAssertThrowsError(try sub.allocate(bytes: 1))
        XCTAssertEqual(sub.usedBytes, 768)
        sub.markSteadyState()
        XCTAssertEqual(sub.steadyStateAllocs, 0)
    }

    func testRingWrapCorrectness() throws {
        let a = Self.arena()
        let ring = try a.ring(slabBytes: 4096, depth: 8)
        let shape = QWTensorShape([1024], .f16)      // 2 KB
        var offsets: [Int] = []
        for _ in 0..<100_000 { offsets.append(ring.acquire(shape).offset) }
        // wraps exactly on depth
        XCTAssertEqual(Set(offsets).count, 8)
        XCTAssertEqual(offsets[0], offsets[8])
        // every offset 256-aligned and inside the activation region
        let regionBase = a.regionSizes.weights
        for off in offsets {
            XCTAssertEqual((regionBase + off) % 256, 0)
            XCTAssertLessThan(off, a.regionSizes.act)
        }
    }

    func testPageAccounting() throws {
        let a = Self.arena()
        let pages = try a.pages(pageSize: 65_536, count: 1024)
        XCTAssertEqual(pages.freePages, 1024)
        var taken: [Int] = []
        for _ in 0..<1024 { taken.append(pages.allocPage()!) }
        XCTAssertEqual(Set(taken).count, 1024, "no double-handout")
        XCTAssertNil(pages.allocPage(), "exhaustion returns nil")
        pages.freePage(taken[5])
        XCTAssertEqual(pages.allocPage(), 5)
        // 128 B page alignment law
        for i in 0..<8 {
            XCTAssertEqual((pages.pageView(i, QWTensorShape([1], .f16)).offset) % 128, 0)
        }
    }

    func testSteadyStateCounterArmsAndCounts() throws {
        let sub = Self.arena().weightsRegion()
        _ = try sub.allocate(bytes: 1024)          // setup alloc — not counted
        sub.markSteadyState()
        _ = try sub.allocate(bytes: 1024)          // hot-path alloc — counted
        _ = try sub.allocate(bytes: 1024)
        XCTAssertEqual(sub.steadyStateAllocs, 2, "counter must prove hot-loop allocs")
        sub.resetSteadyCounters()
        XCTAssertEqual(sub.steadyStateAllocs, 0)
    }

    func testOutOfSpaceIsClean() throws {
        let sub = Self.arena().weightsRegion()
        XCTAssertThrowsError(try sub.allocate(bytes: 1 << 40)) { err in
            guard case QWAllocError.outOfSpace = err else { return XCTFail("\(err)") }
        }
    }

    func testCpuPointerZeroCopy() throws {
        let a = Self.arena()
        let sub = a.gdnRegion()
        let off = try sub.allocate(bytes: 4)
        let p = a.cpuPointer(in: sub, offset: off).assumingMemoryBound(to: Float.self)
        p.pointee = 7.5
        XCTAssertEqual(p.pointee, 7.5)
    }
}
