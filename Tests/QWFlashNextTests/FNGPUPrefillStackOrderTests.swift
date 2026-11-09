import XCTest
@testable import QWFlashNext

final class FNGPUPrefillStackOrderTests: XCTestCase {
    func testSplitOccursOnlyBeforeTheFirstConsumer() {
        XCTAssertEqual(FNGPUPrefillStack.chunkEnd(start: 0, count: 48, layersPerCB: 12, firstPLE: 1), 1)
        XCTAssertEqual(FNGPUPrefillStack.chunkEnd(start: 1, count: 48, layersPerCB: 12, firstPLE: nil), 2)
        XCTAssertEqual(FNGPUPrefillStack.chunkEnd(start: 0, count: 48, layersPerCB: 12, firstPLE: 12), 1)
        XCTAssertEqual(FNGPUPrefillStack.chunkEnd(start: 0, count: 48, layersPerCB: 12, firstPLE: 0), 1)
        XCTAssertEqual(FNGPUPrefillStack.chunkEnd(start: 12, count: 48, layersPerCB: 12, firstPLE: 17), 17)
        XCTAssertEqual(FNGPUPrefillStack.chunkEnd(start: 40, count: 48, layersPerCB: 12, firstPLE: nil), 48)
    }

    func testCommandBufferSizesRampUpToLayersPerCB() {
        var start = 0
        var spans: [Int] = []
        while start < 48 {
            let end = FNGPUPrefillStack.chunkEnd(start: start, count: 48, layersPerCB: 12, firstPLE: nil)
            spans.append(end - start)
            start = end
        }
        XCTAssertEqual(spans, [1, 1, 2, 4, 8, 12, 12, 8])
    }

    func testEveryLayerIsCoveredAndFirstConsumerStartsAChunk() {
        for count in 1...48 {
            for consumer in 0..<count {
                var start = 0
                var covered: [Int] = []
                while start < count {
                    let end = FNGPUPrefillStack.chunkEnd(start: start, count: count,
                                                          layersPerCB: 12, firstPLE: consumer)
                    XCTAssertGreaterThan(end, start)
                    XCTAssertLessThanOrEqual(end, count)
                    if start < consumer { XCTAssertLessThanOrEqual(end, consumer) }
                    covered.append(contentsOf: start..<end)
                    start = end
                }
                XCTAssertEqual(covered, Array(0..<count))
            }
        }
    }
}
