import XCTest
import Metal
import QWAlloc
import QWCore
import QWHAL
import QWGEMM
import QWQuant
import QWHead

/// The head serves the family's containers through the family's twin
/// (`QWGemmAQ`), never by reading them with the incumbent's kernel: the
/// logits must match the host reference of the family's arithmetic.
final class QWHeadContainerTests: XCTestCase {
    private let rows = 32
    private let k = 256

    private func arena(weights: Int) throws -> QWArena {
        try QWArena(hal: Self.hal, weights: weights + (8 << 20),
                    activations: (8 << 20), kv: 256, gdnStates: 256)
    }

    private static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    /// A bank of the given container, as raw bytes.
    private func bank(bits: Int, group: Int) -> Data {
        var out = Data()
        let maxValue = UInt32((1 << bits) - 1)
        for r in 0..<rows {
            for g in 0..<(k / group) {
                let values = (0..<group).map { i in
                    UInt32((i + r * 7 + g * 13) % Int(maxValue + 1))
                }
                out.append(try! QWAQBlock.pack(bits: bits, group: group, values: values,
                                               scale: 0.00390625, bias: -0.0078125))
            }
        }
        return out
    }

    /// Activation row `r`, already rounded to f16 so host and device see the same values.
    private func activation(row r: Int) -> [Float16] {
        (0..<k).map { i in Float16(Float((i * 5 + r * 11) % 17 - 8) / 16) }
    }

    /// The host reference: one logit per bank row, from the family's own arithmetic.
    private func reference(_ bytes: Data, bits: Int, group: Int, x: [Float16]) -> [Float] {
        let blockBytes = QWAQBlock.bytes(bits: bits, group: group)
        let xs = x.map(Float.init)
        return bytes.withUnsafeBytes { raw in
            (0..<rows).map { r in
                (0..<(k / group)).reduce(Float(0)) { acc, g in
                    let at = (r * (k / group) + g) * blockBytes
                    let block = UnsafeRawBufferPointer(rebasing: raw[at..<(at + blockBytes)])
                    return acc + QWAQBlock.dot(group: block, bits: bits, group: group,
                                               x: Array(xs[(g * group)..<((g + 1) * group)]))
                }
            }
        }
    }

    private func assertLogits(_ got: UnsafePointer<Float>, _ want: [Float], _ label: String) {
        for (i, w) in want.enumerated() {
            XCTAssertEqual(got[i], w, accuracy: 1e-3 + abs(w) * 1e-3, "\(label) row \(i)")
        }
    }

    /// The batched head path serves the family container one row at a time through its twin.
    func testTheBatchedHeadServesTheFamilyContainerThroughItsTwin() throws {
        let bytes = bank(bits: 3, group: 64)
        let arena = try arena(weights: bytes.count)
        let w = try arena.weightsRegion().allocate(QWTensorShape([bytes.count / 2], .f16))
        arena.copyIn(w, from: bytes)
        let view = try XCTUnwrap(QWWeightView(blocks: w, n: rows, k: k,
                                              bits: 3, groupSize: 64, isRotated: false))
        let act = arena.activationsRegion()
        let x = try act.allocate(QWTensorShape([2 * k], .f16))
        let y = try act.allocate(QWTensorShape([2 * rows], .f32))
        let xs = [activation(row: 0), activation(row: 1)]
        let xp = arena.cpuPointer(of: x).assumingMemoryBound(to: Float16.self)
        for (r, row) in xs.enumerated() { for (i, v) in row.enumerated() { xp[r * k + i] = v } }
        let cb = try XCTUnwrap(Self.hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        try QWHead.encodeBatched(enc, hal: Self.hal, view: view, x: x, y: y, m: 2)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let yp = arena.cpuPointer(of: y).assumingMemoryBound(to: Float.self)
        for r in 0..<2 {
            assertLogits(yp + r * rows, reference(bytes, bits: 3, group: 64, x: xs[r]), "batch \(r)")
        }
    }

    /// The full-logit head path serves the family container through its twin.
    func testTheFullLogitHeadServesTheFamilyContainerThroughItsTwin() throws {
        let bytes = bank(bits: 3, group: 64)
        let arena = try arena(weights: bytes.count)
        let w = try arena.weightsRegion().allocate(QWTensorShape([bytes.count / 2], .f16))
        arena.copyIn(w, from: bytes)
        let weight = try XCTUnwrap(QWHeadWeight(blocks: w, n: rows, k: k,
                                               bits: 3, groupSize: 64, isRotated: false))
        let act = arena.activationsRegion()
        let x = try act.allocate(QWTensorShape([k], .f16))
        let y = try act.allocate(QWTensorShape([rows], .f32))
        let xs = activation(row: 0)
        let xp = arena.cpuPointer(of: x).assumingMemoryBound(to: Float16.self)
        for (i, v) in xs.enumerated() { xp[i] = v }
        let cb = try XCTUnwrap(Self.hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        try QWHead.encodeQ4g(enc, hal: Self.hal, weight: weight, x: x, y: y)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        assertLogits(arena.cpuPointer(of: y).assumingMemoryBound(to: Float.self),
                     reference(bytes, bits: 3, group: 64, x: xs), "full")
    }

    /// The control: the INCUMBENT's container is served by its own kernel path.
    func testTheIncumbentContainerIsServed() throws {
        let bytes = bank(bits: 4, group: 32)
        let arena = try arena(weights: bytes.count)
        let w = try arena.weightsRegion().allocate(QWTensorShape([bytes.count / 2], .f16))
        arena.copyIn(w, from: bytes)
        let weight = try XCTUnwrap(QWHeadWeight(blocks: w, n: rows, k: k,
                                               bits: 4, groupSize: 32, isRotated: true))
        let act = arena.activationsRegion()
        let x = try act.allocate(QWTensorShape([k], .f16))
        let y = try act.allocate(QWTensorShape([rows], .f32))
        let cb = try XCTUnwrap(Self.hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        defer { if cb.status == .notEnqueued { enc.endEncoding() } }
        XCTAssertNoThrow(try QWHead.encodeQ4g(enc, hal: Self.hal, weight: weight,
                                               x: x, y: y))
    }
}
