// The family's embedding gather over its own container: the token's row,
// decoded on the device. ONE kernel serves every pair the family holds, so
// the test runs every pair — and the reference is the family's own law,
// `value(code) = code * scale + bias`, computed exactly in f64.
import XCTest
import Foundation
import Metal
@testable import QWGEMM
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// The family's embedding gather, over the family's blocks, for every pair.
final class QWAQEmbedGatherTests: XCTestCase {

    /// The bank's header: f16-exact (the format refuses a header it cannot
    /// hold) but NOT a power of two, so the values are not f16-exact either.
    private static let scale: Float = 1025.0 / 4096.0
    private static let bias: Float = -683.0 / 4096.0

    /// A small geometry: the gather needs a HAL and nothing else from it.
    private static func geometry() throws -> QWModelGeometry {
        let text: [String: Any] = [
            "model_type": "qwen3_5_text",
            "hidden_size": 256,
            "vocab_size": 64,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "full_attention_interval": 2,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "intermediate_size": 512,
            "partial_rotary_factor": 1.0,
            "rope_theta": 1e7,
            "eos_token_id": 2,
            "linear_key_head_dim": 32,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
        ]
        let data = try JSONSerialization.data(withJSONObject: ["text_config": text])
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    /// The table in the family's container: one group block after another,
    /// row by row, with the codes a real checkpoint would carry.
    private func makeTable(bits: Int, group: Int, vocab: Int, hidden: Int) -> Data {
        var table = Data()
        let maxValue = UInt32((1 << bits) - 1)
        for r in 0..<vocab {
            for g in 0..<(hidden / group) {
                let values = (0..<group).map { i in
                    UInt32((i * 3 + r * 7 + g * 13) % Int(maxValue + 1))
                }
                table.append(try! QWAQBlock.pack(bits: bits, group: group, values: values,
                                                 scale: Self.scale, bias: Self.bias))
            }
        }
        return table
    }

    /// One column of the expected row in f64 and the f32 ulp tolerance
    /// (product ulp + result ulp, since the bias can cancel the product).
    private func hostColumn(bits: Int, group: Int, row: Int, hidden: Int,
                            table: Data, column: Int) -> (value: Double, tolerance: Double) {
        let groups = hidden / group
        let bytesPerGroup = QWAQBlock.bytes(bits: bits, group: group)
        let g = column / group, j = column % group
        let offset = (row * groups + g) * bytesPerGroup
        let slice = table.subdata(in: offset..<(offset + bytesPerGroup))
        return slice.withUnsafeBytes { raw -> (Double, Double) in
            let (scale, bias) = QWAQBlock.header(raw)
            let payload = UnsafeRawBufferPointer(rebasing: raw.dropFirst(4))
            let code = Double(QWAQBlock.readBits(payload, bits: bits, index: j))
            let value = code * Double(scale) + Double(bias)
            let ulp = 0.000000059604644775390625   // 2^-24, f32's half-ulp factor
            let tol = abs(code * Double(scale)) * ulp + max(abs(value), abs(Double(bias))) * ulp + 1e-30
            return (value, tol)
        }
    }

    private func step(_ hal: QWHal,
                      _ f: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
        try f(enc)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "step failed: \(String(describing: cb.error))")
    }

    /// The gather reads the token's row and writes it out `hc` times, for
    /// every pair the family holds. The reference is the family's own law.
    func testTheGatherReadsTheTokenRowForEveryPair() throws {
        let hal = try QWHal(geometry: try Self.geometry())
        let hidden = 256, vocab = 8, hc = 3
        let tokens = [0, 3, 7]
        for (bits, group) in QWAQBlock.supported {
            let table = makeTable(bits: bits, group: group, vocab: vocab, hidden: hidden)
            let rowBytes = QWAQBlock.rowBytes(bits: bits, group: group, k: hidden)
            let arena = try QWArena(hal: hal, weights: table.count + (4 << 20),
                                    activations: (4 << 20), kv: 0, gdnStates: 0)
            let blocks = try arena.weightsRegion().allocate(QWTensorShape([table.count / 2], .f16))
            arena.copyIn(blocks, from: table)
            let token = try arena.activationsRegion().allocate(QWTensorShape([tokens.count], .i32))
            arena.write(token, from: tokens.map { Int32($0) })
            let out = try arena.activationsRegion()
                .allocate(QWTensorShape([tokens.count * hidden * hc], .f32))

            try step(hal) { enc in
                try QWGemmAQ.encodeEmbedAQ(enc, hal: hal, blocks: blocks, token: token, out: out,
                                           hidden: hidden, hc: hc, bits: bits, group: group,
                                           rows: tokens.count)
            }
            let got = arena.read(out, as: Float.self, count: tokens.count * hidden * hc)

            for (t, tok) in tokens.enumerated() {
                for d in 0..<hidden {
                    let ref = hostColumn(bits: bits, group: group, row: tok, hidden: hidden,
                                         table: table, column: d)
                    for c in 0..<hc {
                        let v = got[t * hidden * hc + c * hidden + d]
                        XCTAssertLessThanOrEqual(abs(Double(v) - ref.value), ref.tolerance,
                                                 "\(bits)-bit group \(group), token \(tok), copy \(c), "
                                                 + "column \(d): got \(v), want \(ref.value)")
                    }
                }
            }
            XCTAssertEqual(rowBytes * vocab, table.count,
                           "\(bits)-bit group \(group): the container must hold exactly the table")
        }
    }

    /// The gather must also work for ONE copy of the row — the MTP path
    /// reads the embedding once, not once per hyper-connection stream.
    func testTheGatherReadsOneCopyOfTheRow() throws {
        let hal = try QWHal(geometry: try Self.geometry())
        let hidden = 256, vocab = 4
        for (bits, group) in QWAQBlock.supported {
            let table = makeTable(bits: bits, group: group, vocab: vocab, hidden: hidden)
            let arena = try QWArena(hal: hal, weights: table.count + (4 << 20),
                                    activations: (4 << 20), kv: 0, gdnStates: 0)
            let blocks = try arena.weightsRegion().allocate(QWTensorShape([table.count / 2], .f16))
            arena.copyIn(blocks, from: table)
            let token = try arena.activationsRegion().allocate(QWTensorShape([1], .i32))
            arena.write(token, from: [Int32(2)])
            let out = try arena.activationsRegion().allocate(QWTensorShape([hidden], .f32))
            try step(hal) { enc in
                try QWGemmAQ.encodeEmbedAQ(enc, hal: hal, blocks: blocks, token: token, out: out,
                                           hidden: hidden, hc: 1, bits: bits, group: group, rows: 1)
            }
            let got = arena.read(out, as: Float.self, count: hidden)
            for d in 0..<hidden {
                let ref = hostColumn(bits: bits, group: group, row: 2, hidden: hidden,
                                     table: table, column: d)
                XCTAssertLessThanOrEqual(abs(Double(got[d]) - ref.value), ref.tolerance,
                                         "\(bits)-bit group \(group), column \(d): got \(got[d])")
            }
        }
    }
}
