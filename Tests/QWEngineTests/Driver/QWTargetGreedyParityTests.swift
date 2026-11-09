import XCTest
import Metal
import QWAlloc
import QWCore
import QWHAL
import QWHead
@testable import QWEngine

final class QWTargetGreedyParityTests: XCTestCase {
    func testTargetReferenceMatchesShippingGPUArgmax() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let head = try QWHead(hal: hal, geometry: .qwen38_27b)
        for vocabulary in [1, 33, 256, 257, 1025, QWModelGeometry.qwen38_27b.vocab] {
            let arena = try QWArena(hal: hal, weights: 256,
                                    activations: vocabulary * 4 + 16_384, kv: 256, gdnStates: 256)
            let a = arena.activationsRegion()
            let logits = try a.allocate(QWTensorShape([vocabulary], .f32))
            let hi = try a.allocate(QWTensorShape([QWHead.argmaxSlots(vocabulary)], .i32))
            let lo = try a.allocate(QWTensorShape([QWHead.argmaxSlots(vocabulary)], .i32))
            let token = try a.allocate(QWTensorShape([1], .i32))
            let p = logits.region.pointer.advanced(by: logits.offset).assumingMemoryBound(to: Float.self)
            a.markSteadyState()
            for scenario in 0..<8 {
                for i in 0..<vocabulary { p[i] = -Float(i % 103 + 1) }
                switch scenario {
                case 0: p[vocabulary - 1] = 7
                case 1: p[0] = 7; p[vocabulary - 1] = 7
                case 2: p[0] = -0.0; p[vocabulary - 1] = 0.0
                case 3: p[0] = 0.0; p[vocabulary - 1] = -0.0
                case 4:
                    for i in 0..<vocabulary { p[i] = -0.0 }
                case 5:
                    p[0] = -Float.leastNonzeroMagnitude
                    p[vocabulary - 1] = Float.leastNonzeroMagnitude
                case 6:
                    p[min(255, vocabulary - 1)] = Float.greatestFiniteMagnitude
                    p[vocabulary - 1] = Float.greatestFiniteMagnitude
                default:
                    var bits: UInt32 = 0x12345678
                    for i in 0..<vocabulary {
                        bits = bits &* 1664525 &+ 1013904223
                        let finite = bits & 0x7f800000 == 0x7f800000 ? bits ^ 0x00800000 : bits
                        p[i] = Float(bitPattern: finite)
                    }
                }
                let expected = try QWVerifyWindow.greedyToken(
                    UnsafeBufferPointer(start: p, count: vocabulary), ordering: .shippingTarget)
                let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
                let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
                do {
                    try head.encodeArgmax(enc, y: logits, keysHi: hi, keysLo: lo, token: token, N: vocabulary)
                } catch {
                    enc.endEncoding()
                    throw error
                }
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                XCTAssertEqual(cb.status, .completed, "\(String(describing: cb.error))")
                let got = Int(token.region.pointer.advanced(by: token.offset).assumingMemoryBound(to: Int32.self).pointee)
                XCTAssertEqual(expected, got, "V=\(vocabulary), case=\(scenario)")
                if scenario == 2 && vocabulary > 1 {
                    XCTAssertEqual(expected, vocabulary - 1, "+0 outranks -0 in the shipping target")
                    XCTAssertEqual(try QWVerifyWindow.greedyToken(UnsafeBufferPointer(start: p, count: vocabulary)), 0,
                                   "draft numerical ties remain unchanged")
                }
            }
            XCTAssertEqual(a.steadyStateAllocs, 0)
        }
    }

    func testVerificationUsesTargetOrderForAcceptanceAndBonus() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        for rows in [2, 3] {
            let arena = try QWArena(hal: hal, weights: 256, activations: 128 << 10,
                                    kv: 256, gdnStates: 256)
            let window = try QWVerifyWindow(geometry: QWTestGeometry.g27, allocator: arena.activationsRegion(), rows: rows, vocabulary: 33)
            let p = window.logits.region.pointer.advanced(by: window.logits.offset).assumingMemoryBound(to: Float.self)
            for i in 0..<rows * 33 { p[i] = -1 }
            for row in 0..<rows { p[row * 33] = -0.0; p[row * 33 + 32] = 0.0 }
            let accepted = try window.greedyDecision(drafts: Array(repeating: 32, count: rows - 1), maxNew: rows)
            XCTAssertEqual(accepted.acceptedDrafts, rows - 1)
            XCTAssertEqual(accepted.tokens, Array(repeating: 32, count: rows))
            let rejected = try window.greedyDecision(drafts: Array(repeating: 0, count: rows - 1), maxNew: rows)
            XCTAssertEqual(rejected.acceptedDrafts, 0)
            XCTAssertEqual(rejected.tokens, [32])
            let stopped = try window.greedyDecision(drafts: Array(repeating: 32, count: rows - 1),
                                                    maxNew: rows, stopIDs: [32])
            XCTAssertTrue(stopped.stopped)
            XCTAssertEqual(stopped.retainedRows, 0)
            for invalid in [Float.nan, Float.infinity, -Float.infinity] {
                p[(rows - 1) * 33 + 1] = invalid
                XCTAssertThrowsError(try window.greedyDecision(drafts: Array(repeating: 0, count: rows - 1), maxNew: rows),
                                     "all target rows must be finite even after early rejection")
            }
        }
    }
}
