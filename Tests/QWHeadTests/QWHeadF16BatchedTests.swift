import XCTest
import Metal
import QWAlloc
import QWCore
import QWHAL
import QWHead

final class QWHeadF16BatchedTests: XCTestCase {
    func testAllRowsBitExactAgainstShippingGemv() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let h = QWModelGeometry.qwen38_27b.hidden
        for n in [1, 257, 1024] {
            let arena = try QWArena(hal: hal, weights: n * h * 2 + 256,
                                    activations: 8 * h * 2 + 16 * n * 4 + 2048,
                                    kv: 256, gdnStates: 256)
            let w = try arena.weightsRegion().allocate(QWTensorShape([n * h], .f16))
            let a = arena.activationsRegion()
            let x = try a.allocate(QWTensorShape([8 * h], .f16))
            let y = try a.allocate(QWTensorShape([8 * n], .f32))
            let control = try a.allocate(QWTensorShape([8 * n], .f32))
            let wp = w.region.pointer.advanced(by: w.offset).assumingMemoryBound(to: Float16.self)
            let xp = x.region.pointer.advanced(by: x.offset).assumingMemoryBound(to: Float16.self)
            for i in 0..<(n * h) { wp[i] = Float16(Float((i * 17 + i / h) % 127 - 63) / 137) }
            for i in 0..<(8 * h) { xp[i] = Float16(Float((i * 31) % 113 - 56) / 53) }
            for m in 1...8 {
                func prefix(_ tensor: QWTensorValue, count: Int) -> QWTensorValue {
                    QWTensorValue(region: tensor.region, offset: tensor.offset,
                                  shape: QWTensorShape([count], tensor.shape.dtype))
                }
                let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
                let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
                defer { if cb.status == .notEnqueued { enc.endEncoding() } }
                try QWHead.encodeF16(enc, hal: hal, w: w, x: prefix(x, count: m * h),
                                     y: prefix(y, count: m * n), geometry: .qwen38_27b, M: m, N: n)
                for row in 0..<m {
                    let xr = QWTensorValue(region: x.region, offset: x.offset + row * h * 2,
                                           shape: QWTensorShape([h], .f16))
                    let yr = QWTensorValue(region: control.region, offset: control.offset + row * n * 4,
                                           shape: QWTensorShape([n], .f32))
                    try QWHead.encodeF16(enc, hal: hal, w: w, x: xr, y: yr, geometry: .qwen38_27b, N: n)
                }
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                XCTAssertEqual(cb.status, .completed, "\(String(describing: cb.error))")
                let got = y.region.pointer.advanced(by: y.offset).assumingMemoryBound(to: Float.self)
                let ref = control.region.pointer.advanced(by: control.offset).assumingMemoryBound(to: Float.self)
                for i in 0..<(m * n) {
                    XCTAssertTrue(got[i].isFinite)
                    XCTAssertEqual(got[i].bitPattern, ref[i].bitPattern, "M=\(m), N=\(n), output=\(i)")
                }
            }
        }
    }

    func testInvalidWindowShapesAreRejectedBeforeDispatch() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let arena = try QWArena(hal: hal, weights: 2 * QWModelGeometry.qwen38_27b.hidden + 256,
                                activations: 6 * QWModelGeometry.qwen38_27b.hidden + 2048, kv: 256, gdnStates: 256)
        let w = try arena.weightsRegion().allocate(QWTensorShape([QWModelGeometry.qwen38_27b.hidden], .f16))
        let a = arena.activationsRegion()
        let x = try a.allocate(QWTensorShape([QWModelGeometry.qwen38_27b.hidden], .f16))
        let y = try a.allocate(QWTensorShape([1], .f32))
        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        defer { enc.endEncoding() }
        for m in [0, 9] {
            XCTAssertThrowsError(try QWHead.encodeF16(enc, hal: hal, w: w, x: x, y: y, geometry: .qwen38_27b, M: m, N: 1))
        }
        XCTAssertThrowsError(try QWHead.encodeF16(enc, hal: hal, w: w, x: x, y: y, geometry: .qwen38_27b, N: 0))
        let misaligned = QWTensorValue(region: w.region, offset: w.offset + 2, shape: w.shape)
        XCTAssertThrowsError(try QWHead.encodeF16(enc, hal: hal, w: misaligned, x: x, y: y, geometry: .qwen38_27b, N: 1))
        let shortRegion = QWMRegion(buffer: w.region.buffer, base: w.region.base, length: 16)
        let truncated = QWTensorValue(region: shortRegion, offset: 0, shape: w.shape)
        XCTAssertThrowsError(try QWHead.encodeF16(enc, hal: hal, w: truncated, x: x, y: y, geometry: .qwen38_27b, N: 1))
        let misalignedOutput = QWTensorValue(region: y.region, offset: y.offset + 2, shape: y.shape)
        XCTAssertThrowsError(try QWHead.encodeF16(enc, hal: hal, w: w, x: x, y: misalignedOutput, geometry: .qwen38_27b, N: 1))
    }
}
