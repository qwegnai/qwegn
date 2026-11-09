import XCTest
import Metal
import QWAlloc
import QWCore
import QWHAL
@testable import QWEngine

final class QWMTPGreedySelectorTests: XCTestCase {
    func testCheckedArgmaxMatchesReferenceAtTileAndVocabularyBoundaries() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        XCTAssertEqual(QWMTPGreedySelector.allocationBytes(geometry: QWTestGeometry.g27, vocabulary: QWModelGeometry.qwen38_27b.vocab), 8_448)
        for vocabulary in [1, 33, 256, 257, 1025, QWModelGeometry.qwen38_27b.vocab] {
            let arena = try QWArena(hal: hal, weights: 256,
                                    activations: vocabulary * 4 + 16_384, kv: 256, gdnStates: 256)
            let a = arena.activationsRegion()
            let selector = try QWMTPGreedySelector(geometry: QWTestGeometry.g27, allocator: a, vocabulary: vocabulary)
            XCTAssertEqual(a.usedBytes, QWMTPGreedySelector.allocationBytes(geometry: QWTestGeometry.g27, vocabulary: vocabulary))
            let logits = try a.allocate(QWTensorShape([vocabulary], .f32))
            let p = logits.region.pointer.advanced(by: logits.offset).assumingMemoryBound(to: Float.self)
            _ = try hal.pipeline("qwg_mtp_argmax_checked")
            _ = try hal.pipeline("qwg_head_argmax_fin")
            a.markSteadyState()
            func submit(ordering: QWVerifyWindow.GreedyOrdering = .numericalDraft) throws {
                let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
                let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
                do {
                    try selector.encode(enc, hal: hal, logits: logits, ordering: ordering)
                } catch {
                    enc.endEncoding()
                    throw error
                }
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                XCTAssertEqual(cb.status, .completed, "\(String(describing: cb.error))")
            }
            for scenario in 0..<5 {
                for i in 0..<vocabulary { p[i] = -Float(i % 103 + 1) }
                switch scenario {
                case 0: p[vocabulary - 1] = 7
                case 1:
                    p[0] = 7; p[vocabulary - 1] = 7
                case 2:
                    p[0] = -0.0; p[vocabulary - 1] = 0.0
                case 3:
                    p[0] = -Float.leastNonzeroMagnitude
                    p[vocabulary - 1] = Float.leastNonzeroMagnitude
                default:
                    p[min(255, vocabulary - 1)] = Float.greatestFiniteMagnitude
                    p[vocabulary - 1] = Float.greatestFiniteMagnitude
                }
                for ordering in [QWVerifyWindow.GreedyOrdering.numericalDraft, .shippingTarget] {
                    let expected = try QWVerifyWindow.greedyToken(UnsafeBufferPointer(start: p, count: vocabulary),
                                                                ordering: ordering)
                    try submit(ordering: ordering)
                    XCTAssertEqual(try selector.completedToken(), expected, "V=\(vocabulary), case=\(scenario)")
                }
            }
            for id in Set([0, min(255, vocabulary - 1), vocabulary - 1]) {
                for invalid in [Float.nan, Float.infinity, -Float.infinity,
                                Float(bitPattern: 0x7f800001), Float(bitPattern: 0xffc00001)] {
                    for i in 0..<vocabulary { p[i] = Float(i % 17) }
                    p[id] = invalid
                    try submit()
                    XCTAssertThrowsError(try selector.completedToken(), "bad row \(id) must poison the whole reduction")
                    p[id] = 99
                    try submit()
                    XCTAssertEqual(try selector.completedToken(), id, "every dispatch must overwrite the invalid sentinel")
                }
            }
            XCTAssertEqual(a.steadyStateAllocs, 0)
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            defer { enc.endEncoding() }
            let bad = QWTensorValue(region: logits.region, offset: logits.region.length,
                                    shape: logits.shape)
            XCTAssertThrowsError(try selector.encode(enc, hal: hal, logits: bad))
            let alias = QWTensorValue(region: selector.keysHi.region, offset: selector.keysHi.offset,
                                      shape: logits.shape)
            XCTAssertThrowsError(try selector.encode(enc, hal: hal, logits: alias))
        }
    }
}
