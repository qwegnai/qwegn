// QWMTPGreedySelectorTests.swift — Two-stage GPU token selector parity & allocation tests (via QWHead.QWTokenSelector).

import Foundation
import Metal
import XCTest
import QWAlloc
import QWCore
import QWHAL
import QWHead
@testable import QWEngine

final class QWMTPGreedySelectorTests: XCTestCase {
    private static func allocateLogits(
        _ allocator: QWSubAllocator, rows: Int, vocabulary: Int, values: [Float]
    ) throws -> QWTensorValue {
        let tensor = try allocator.allocate(QWTensorShape([rows * vocabulary], .f32))
        let dst = tensor.region.pointer.advanced(by: tensor.offset).assumingMemoryBound(to: Float.self)
        values.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            dst.update(from: base, count: values.count)
        }
        return tensor
    }

    private static func selectorOrdering(for ordering: QWGreedyOrdering) -> QWVerifyWindow.GreedyOrdering {
        ordering == .numericalDraft ? .numericalDraft : .shippingTarget
    }

    private static func submit(_ hal: QWHal, _ encode: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = hal.begin() else {
            throw QWEngineDriverError.state("failed to create compute encoder")
        }
        do {
            try encode(enc)
        } catch {
            enc.endEncoding()
            throw error
        }
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        if cb.status != .completed {
            throw cb.error ?? QWEngineDriverError.state("command buffer failed")
        }
    }

    func testSingleAndBatchedSelectionMatchCPUOrderingsAndJournal() throws {
        let hal = try QWHal(geometry: .qwen38_27b), vocabulary = 1_300
        let arena = try QWArena(hal: hal, weights: 0, activations: 1 << 20, kv: 0, gdnStates: 0)
        let a = arena.activationsRegion()
        let before = a.usedBytes
        let selector = try QWTokenSelector(
            vocabLimit: QWTestGeometry.g27.vocab, allocator: a, vocabulary: vocabulary, maxRows: 4
        )
        XCTAssertEqual(
            a.usedBytes - before,
            QWTokenSelector.allocationBytes(vocabLimit: QWTestGeometry.g27.vocab, vocabulary: vocabulary, maxRows: 4)
        )
        let dst = try a.allocate(QWTensorShape([4], .i32))

        var values = (0..<(3 * vocabulary)).map { idx in
            let row = idx / vocabulary, col = idx % vocabulary
            return Float(((col * 7 + row * 13) % 97)) * 0.01 - 0.5
        }
        let infBit = Float.infinity.bitPattern, nanBit = Float.nan.bitPattern
        values[vocabulary - 1] = 3.0; values[vocabulary - 2] = 3.0
        values[vocabulary + 0] = 4.0; values[vocabulary + 512] = 4.0
        values[2 * vocabulary + 700] = 5.0
        let logits = try Self.allocateLogits(a, rows: 3, vocabulary: vocabulary, values: values)
        let row0Slice = logits.slice(byteOffset: 0, count: vocabulary, dtype: .f32)

        for ordering in [QWGreedyOrdering.numericalDraft, .shippingTarget] {
            try Self.submit(hal) { enc in
                try selector.encodeGreedy(enc, hal: hal, logits: row0Slice, ordering: ordering)
            }
            try Self.submit(hal) { enc in
                try selector.encodeGreedyBatched(enc, hal: hal, logits: logits, rows: 3, ordering: ordering, destination: dst)
            }
            let row0 = try selector.completedToken()
            let batched = try selector.completedTokens(from: dst, count: 3)
            let expected = try (0..<3).map { row in
                try Array(values[(row * vocabulary)..<((row + 1) * vocabulary)]).withUnsafeBufferPointer { buf in
                    try QWVerifyWindow.greedyToken(buf, ordering: Self.selectorOrdering(for: ordering))
                }
            }
            XCTAssertEqual(row0, expected[0])
            XCTAssertEqual(batched, expected)
        }

        for bad in [Float(bitPattern: infBit), Float(bitPattern: nanBit)] {
            var badValues = values
            badValues[2 * vocabulary + 123] = bad
            let badLogits = try Self.allocateLogits(a, rows: 3, vocabulary: vocabulary, values: badValues)
            try Self.submit(hal) { enc in
                try selector.encodeGreedyBatched(
                    enc, hal: hal, logits: badLogits, rows: 3, ordering: .shippingTarget, destination: dst
                )
            }
            XCTAssertThrowsError(try selector.completedTokens(from: dst, count: 3))
        }
    }

    func testAllocationBytesHandlesSmallAndLargeVocabLimits() throws {
        let small = QWTokenSelector.allocationBytes(vocabLimit: 256, vocabulary: 256, maxRows: 3)
        let large = QWTokenSelector.allocationBytes(
            vocabLimit: QWTestGeometry.g27.vocab, vocabulary: QWTestGeometry.g27.vocab, maxRows: 4
        )
        XCTAssertGreaterThan(small, 0)
        XCTAssertGreaterThan(large, small)
    }

    func testOutOfRangeVocabularyAndRowGuards() throws {
        let hal = try QWHal(geometry: .qwen38_27b), vocabulary = 128
        let arena = try QWArena(hal: hal, weights: 0, activations: 256 << 10, kv: 0, gdnStates: 0)
        let a = arena.activationsRegion()
        XCTAssertThrowsError(try QWTokenSelector(vocabLimit: QWTestGeometry.g27.vocab, allocator: a, vocabulary: 0))
        XCTAssertThrowsError(
            try QWTokenSelector(vocabLimit: QWTestGeometry.g27.vocab, allocator: a, vocabulary: QWTestGeometry.g27.vocab + 1)
        )
        let selector = try QWTokenSelector(vocabLimit: QWTestGeometry.g27.vocab, allocator: a, vocabulary: vocabulary, maxRows: 2)
        let logits = try a.allocate(QWTensorShape([2 * vocabulary], .f32))
        let smallDst = try a.allocate(QWTensorShape([1], .i32))
        let (cb, enc) = try XCTUnwrap(hal.begin())
        defer { enc.endEncoding(); cb.commit(); cb.waitUntilCompleted() }
        XCTAssertThrowsError(
            try selector.encodeGreedyBatched(enc, hal: hal, logits: logits, rows: 3, ordering: .shippingTarget, destination: smallDst)
        )
        XCTAssertThrowsError(
            try selector.encodeGreedyBatched(enc, hal: hal, logits: logits, rows: 2, ordering: .shippingTarget, destination: smallDst)
        )
    }
}
