// The loader binds the family's destination at a ROW OFFSET inside a
// concatenated bank — the gate and the up projection are two tensors in one
// bank, and the second starts at the first one's row count. Metal must
// accept that offset for every pair the family holds, and the bytes must
// land where the loader says they will. The (4, 128) pair's rows are 2176
// bytes at k=4096, which is not a multiple of 256, so it is the pair that
// would catch a rule the others never meet.
import XCTest
import Foundation
import Metal
@testable import QWGEMM
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// The destination offset the loader binds must be accepted, and the bytes
/// must land where it says.
final class QWAQRepackOffsetTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    /// The dtype the checkpoint wrote its scales in: 0 is bf16, 1 is f16.
    private enum ScaleDType: UInt32 { case bf16 = 0, f16 = 1 }

    /// The activation width the model's own weights have.
    private static let k = 4096
    /// The rows the checkpoint carries: two is enough to move the offset.
    private static let rows = 2

    private func step(_ f: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
        try f(enc)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "step failed: \(String(describing: cb.error))")
    }

    /// One checkpoint: the payload, the scales, the biases, in the dtype the
    /// checkpoint wrote them in.
    private func makeCheckpoint(bits: Int, group: Int,
                                dtype: ScaleDType) -> (payload: Data, scales: [UInt16], biases: [UInt16]) {
        let groups = Self.k / group
        var seed: UInt32 = 0x1357_9BDF
        var payload = Data()
        var scales: [UInt16] = []
        var biases: [UInt16] = []
        for _ in 0..<(Self.rows * groups) {
            let s = Float(Int32(truncatingIfNeeded: seed >> 8) % 2000) / 20000.0 + 0.001
            let b = -Float(Int32(truncatingIfNeeded: seed >> 20) % 2000) / 2000.0
            seed = seed &* 1664525 &+ 1013904223
            scales.append(dtype == .bf16 ? QWBF16.fromFloatRNE(s) : Float16(s).bitPattern)
            biases.append(dtype == .bf16 ? QWBF16.fromFloatRNE(b) : Float16(b).bitPattern)
        }
        for _ in 0..<Self.rows {
            let bytes = Self.k * bits / 8
            var row = [UInt8](repeating: 0, count: bytes)
            for i in 0..<Self.k {
                seed = seed &* 1664525 &+ 1013904223
                let code = (seed >> 8) % UInt32(1 << bits)
                let bit = i * bits
                let byte = bit >> 3
                let off = UInt32(bit & 7)
                let take = min(8 - Int(off), bits)
                let mask = UInt32((1 << take) - 1)
                row[byte] = UInt8(UInt32(row[byte]) | ((code & mask) << off))
                if take < bits {
                    row[byte + 1] = UInt8(UInt32(row[byte + 1]) | ((code >> UInt32(take)) << 0))
                }
            }
            payload.append(contentsOf: row)
        }
        return (payload, scales, biases)
    }

    /// Repacks the checkpoint into a destination bound at `dstOffset`, and
    /// returns the bytes the device wrote there.
    private func repackAt(bits: Int, group: Int, dstOffset: Int, dtype: ScaleDType,
                          arena: QWArena) throws -> [UInt8] {
        let groups = Self.k / group
        let groupBytes = QWAQBlock.bytes(bits: bits, group: group)
        let bankBytes = Self.rows * groups * groupBytes
        let checkpoint = makeCheckpoint(bits: bits, group: group, dtype: dtype)
        let dst = try arena.weightsRegion()
            .allocate(QWTensorShape([(bankBytes + dstOffset) / 2], .f16))
        let payload = try arena.weightsRegion()
            .allocate(QWTensorShape([checkpoint.payload.count / 2], .f16))
        arena.copyIn(payload, from: checkpoint.payload)
        let scales = try arena.weightsRegion()
            .allocate(QWTensorShape([checkpoint.scales.count], .f16))
        var scaleData = Data()
        for s in checkpoint.scales { scaleData.append(contentsOf: [UInt8(s & 0xFF), UInt8(s >> 8)]) }
        arena.copyIn(scales, from: scaleData)
        let biases = try arena.weightsRegion()
            .allocate(QWTensorShape([checkpoint.biases.count], .f16))
        var biasData = Data()
        for b in checkpoint.biases { biasData.append(contentsOf: [UInt8(b & 0xFF), UInt8(b >> 8)]) }
        arena.copyIn(biases, from: biasData)
        let flags = try arena.weightsRegion().allocate(QWTensorShape([1], .i32))
        arena.write(flags, from: [Int32(0)])

        try step { enc in
            let pso = try Self.hal.pipeline("qwg_repack_mlx_affine_to_aq_m5")
            enc.setComputePipelineState(pso)
            payload.bind(enc, index: 0)
            scales.bind(enc, index: 1)
            biases.bind(enc, index: 2)
            // The loader's own binding: the destination at a row offset.
            enc.setBuffer(dst.region.buffer, offset: dst.region.base + dst.offset + dstOffset, index: 3)
            var rows = UInt32(Self.rows), cols = UInt32(Self.k), b = UInt32(bits), g = UInt32(group)
            var dt = dtype.rawValue
            enc.setBytes(&rows, length: 4, index: 4)
            enc.setBytes(&cols, length: 4, index: 5)
            enc.setBytes(&b, length: 4, index: 6)
            enc.setBytes(&g, length: 4, index: 7)
            enc.setBytes(&dt, length: 4, index: 8)
            flags.bind(enc, index: 9)
            let total = Self.rows * groups
            let tpg = 64
            enc.dispatchThreadgroups(MTLSize(width: (total + tpg - 1) / tpg, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
        }
        let refusals = arena.read(flags, as: Int32.self, count: 1)[0]
        XCTAssertEqual(refusals, 0, "the scales and biases must survive the f16 header exactly")
        let base = dst.region.buffer.contents()
            .advanced(by: dst.region.base + dst.offset + dstOffset)
            .assumingMemoryBound(to: UInt8.self)
        return Array(UnsafeBufferPointer(start: base, count: bankBytes))
    }

    /// The destination offset the loader binds must be accepted, for every
    /// pair the family holds, and the bytes must land at the offset.
    func testTheDestinationOffsetIsAcceptedForEveryPair() throws {
        for (bits, group) in QWAQBlock.supported {
            let groupBytes = QWAQBlock.bytes(bits: bits, group: group)
            let rowBytes = (Self.k / group) * groupBytes
            let checkpoint = makeCheckpoint(bits: bits, group: group, dtype: .bf16)
            for row in 0..<3 {
                let offset = row * rowBytes
                let arena = try QWArena(hal: Self.hal, weights: (8 << 20),
                                        activations: (4 << 20), kv: 0, gdnStates: 0)
                let got = try repackAt(bits: bits, group: group, dstOffset: offset,
                                       dtype: .bf16, arena: arena)
                // The header: the checkpoint's own scale and bias, in f16.
                let scale = Float16(bitPattern: UInt16(got[0]) | (UInt16(got[1]) << 8))
                let bias = Float16(bitPattern: UInt16(got[2]) | (UInt16(got[3]) << 8))
                let wantScale = QWBF16.toFloat(checkpoint.scales[0])
                let wantBias = QWBF16.toFloat(checkpoint.biases[0])
                XCTAssertEqual(Float(scale), wantScale,
                               "\(bits)-bit G\(group) at row offset \(offset): the scale must land "
                               + "at the offset the loader bound")
                XCTAssertEqual(Float(bias), wantBias,
                               "\(bits)-bit G\(group) at row offset \(offset): the bias must land "
                               + "at the offset the loader bound")
                // The payload: the checkpoint's own bytes, verbatim. The bank
                // is laid out group by group, so the first group's payload is
                // the checkpoint's first `group` codes.
                let payloadBytes = group * bits / 8
                let gotPayload = Array(got[4..<(4 + payloadBytes)])
                let wantPayload = Array(checkpoint.payload.prefix(payloadBytes))
                XCTAssertEqual(gotPayload, wantPayload,
                               "\(bits)-bit G\(group) at row offset \(offset): the payload must be "
                               + "the checkpoint's own bytes (first \(payloadBytes) bytes)")
            }
        }
    }
}
