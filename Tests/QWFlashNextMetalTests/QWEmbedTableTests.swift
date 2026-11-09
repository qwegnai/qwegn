// The Flash-Next embedding table over a QUANTIZED checkpoint: the family's
// own blocks, at the checkpoint's NATIVE size, read by a device-side gather.
// The acceptance criterion is the memory one — the table must not be the
// inflated bf16 plane — and the table must still read the checkpoint's values.
import XCTest
import Foundation
import Metal
@testable import QWFlashNext
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWQuant

/// The embedding table's container, for a quantized checkpoint.
final class QWEmbedTableTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qwg-embed-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    /// Writes a minimal single-file safetensors: 8-byte header length + JSON
    /// header + payload.
    private func writeSafetensors(_ name: String, tensors: [(String, [Int], String, [UInt8])]) throws {
        var header: [String: Any] = [:]
        var payload = Data()
        for (tname, shape, dtype, bytes) in tensors {
            header[tname] = ["dtype": dtype, "shape": shape,
                             "data_offsets": [payload.count, payload.count + bytes.count]]
            payload.append(contentsOf: bytes)
        }
        let headerJSON = try JSONSerialization.data(withJSONObject: header)
        var out = Data()
        out.append(contentsOf: withUnsafeBytes(of: UInt64(headerJSON.count).littleEndian) { Array($0) })
        out.append(headerJSON)
        out.append(payload)
        try out.write(to: dir.appendingPathComponent(name))
    }

    /// The MLX affine row, as the checkpoint writes it: 2/4/8 bits in words,
    /// every other width in one continuous little-endian bit stream.
    private func mlxRow(values: [Int], bits: Int) -> [UInt8] {
        if bits == 2 || bits == 4 || bits == 8 {
            let valsPerWord = 32 / bits
            var words = [UInt32](repeating: 0, count: values.count / valsPerWord)
            for (i, v) in values.enumerated() {
                words[i / valsPerWord] |= UInt32(v) << UInt32((i % valsPerWord) * bits)
            }
            var bytes = [UInt8]()
            bytes.reserveCapacity(words.count * 4)
            for w in words {
                for b in 0..<4 { bytes.append(UInt8((w >> (8 * UInt32(b))) & 0xFF)) }
            }
            return bytes
        }
        var bytes = [UInt8](repeating: 0, count: (values.count * bits + 7) / 8)
        for (i, v) in values.enumerated() {
            for b in 0..<bits where (v >> b) & 1 == 1 {
                let bitPos = i * bits + b
                bytes[bitPos >> 3] |= UInt8(1 << (bitPos & 7))
            }
        }
        return bytes
    }

    /// A checkpoint's affine embedding: the codes, and the group's own scale
    /// and bias — f16-exact but NOT powers of two, so the values are not
    /// f16-exact either.
    private func makeCheckpoint(vocab: Int, hidden: Int, bits: Int, group: Int,
                                scales: [Float], biases: [Float]) throws -> (codes: [[Int]], bytes: [UInt8]) {
        var codes = [[Int]]()
        var payload = [UInt8]()
        for r in 0..<vocab {
            let row = (0..<hidden).map { c in (c * 5 + r * 3 + (c / group) * 7) % (1 << bits) }
            codes.append(row)
            payload.append(contentsOf: mlxRow(values: row, bits: bits))
        }
        var headerBytes = [UInt8]()
        for s in scales { headerBytes.append(contentsOf: bf16(s)) }
        for b in biases { headerBytes.append(contentsOf: bf16(b)) }
        return (codes, payload + headerBytes)
    }

    /// A bf16 bit pattern, little-endian.
    private func bf16(_ v: Float) -> [UInt8] {
        let bits = UInt32(v.bitPattern) >> 16
        return [UInt8(bits & 0xFF), UInt8((bits >> 8) & 0xFF)]
    }

    private static func geometry() throws -> QWModelGeometry {
        let text: [String: Any] = [
            "model_type": "qwen3_5_text", "hidden_size": 256, "vocab_size": 64,
            "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
            "full_attention_interval": 2, "rms_norm_eps": 1e-6, "max_position_embeddings": 4096,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
            "intermediate_size": 512, "partial_rotary_factor": 1.0, "rope_theta": 1e7,
            "eos_token_id": 2, "linear_key_head_dim": 32, "linear_num_key_heads": 2,
            "linear_num_value_heads": 4, "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4,
        ]
        let data = try JSONSerialization.data(withJSONObject: ["text_config": text])
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    /// A quantized checkpoint's table must be the FAMILY's blocks, at the
    /// checkpoint's native size — and never the inflated bf16 plane.
    func testAQuantizedCheckpointKeepsItsOwnBytes() throws {
        let vocab = 16, hidden = 256, bits = 3, group = 64
        let groups = hidden / group
        let scale = Float(129) / Float(512), bias = Float(-171) / Float(1024)
        let made = try makeCheckpoint(vocab: vocab, hidden: hidden, bits: bits, group: group,
                                      scales: [Float](repeating: scale, count: vocab * groups),
                                      biases: [Float](repeating: bias, count: vocab * groups))
        let payloadBytes = made.bytes.count - vocab * groups * 2 * 2
        let headerBytes = Array(made.bytes[payloadBytes...])
        try writeSafetensors("model.safetensors", tensors: [
            ("model.embed_tokens.weight", [vocab, hidden * bits / 32], "U32", Array(made.bytes.prefix(payloadBytes))),
            ("model.embed_tokens.scales", [vocab, groups], "BF16", Array(headerBytes.prefix(vocab * groups * 2))),
            ("model.embed_tokens.biases", [vocab, groups], "BF16", Array(headerBytes.dropFirst(vocab * groups * 2))),
        ])
        let container = try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors"))
        let hal = try QWHal(geometry: try Self.geometry())
        let arena = try QWArena(hal: hal, weights: (4 << 20), activations: (4 << 20), kv: 0, gdnStates: 0)

        let table = try FNGPUEmbedTable.load(arena: arena, container: container,
                                             name: "model.embed_tokens.weight",
                                             vocab: vocab, hidden: hidden, hc: 2)
        guard case .nativeAffine(let blocks, let gotBits, let gotGroup) = table.storage else {
            return XCTFail("a quantized checkpoint must keep its own bytes, got \(table.containerNote)")
        }
        XCTAssertEqual(gotBits, bits)
        XCTAssertEqual(gotGroup, group)
        let rowBytes = QWAQBlock.rowBytes(bits: bits, group: group, k: hidden)
        XCTAssertEqual(blocks.bytes, vocab * rowBytes,
                       "the table must be the family's container, byte for byte")
        XCTAssertLessThan(blocks.bytes, vocab * hidden * 2,
                          "the table must not be the inflated bf16 plane")
        XCTAssertTrue(table.containerNote.contains("family"), "the container must be named: \(table.containerNote)")
    }

    /// The 1.27 GiB plane is what the table must NOT be. On the 27B shape
    /// the family's container is 198 MB against the plane's 1.27 GB — the
    /// ratio is arithmetic, and it is checked without allocating either.
    func testTheFamilyContainerIsTheSmallOneOnTheRealShape() throws {
        let vocab = 248320, hidden = 2560
        let family = vocab * QWAQBlock.rowBytes(bits: 2, group: 64, k: hidden)
        let plane = vocab * hidden * 2
        XCTAssertEqual(family, 198_656_000, "the family's container on the 27B shape")
        XCTAssertEqual(plane, 1_271_398_400, "the bf16 plane on the 27B shape")
        XCTAssertLessThan(family, plane / 6, "the family's container must be a fraction of the plane")
    }

    /// The table must still READ the checkpoint's values: the family's law,
    /// `value(code) = code * scale + bias`, gathered on the device.
    func testTheGatherReadsTheCheckpointsOwnValues() throws {
        let vocab = 16, hidden = 256, bits = 3, group = 64
        let groups = hidden / group
        let scale = Float(129) / Float(512), bias = Float(-171) / Float(1024)
        let made = try makeCheckpoint(vocab: vocab, hidden: hidden, bits: bits, group: group,
                                      scales: [Float](repeating: scale, count: vocab * groups),
                                      biases: [Float](repeating: bias, count: vocab * groups))
        let payloadBytes = made.bytes.count - vocab * groups * 2 * 2
        let headerBytes = Array(made.bytes[payloadBytes...])
        try writeSafetensors("model.safetensors", tensors: [
            ("model.embed_tokens.weight", [vocab, hidden * bits / 32], "U32", Array(made.bytes.prefix(payloadBytes))),
            ("model.embed_tokens.scales", [vocab, groups], "BF16", Array(headerBytes.prefix(vocab * groups * 2))),
            ("model.embed_tokens.biases", [vocab, groups], "BF16", Array(headerBytes.dropFirst(vocab * groups * 2))),
        ])
        let container = try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors"))
        let hal = try QWHal(geometry: try Self.geometry())
        let arena = try QWArena(hal: hal, weights: (4 << 20), activations: (4 << 20), kv: 0, gdnStates: 0)
        let table = try FNGPUEmbedTable.load(arena: arena, container: container,
                                             name: "model.embed_tokens.weight",
                                             vocab: vocab, hidden: hidden, hc: 2)

        let token = try arena.activationsRegion().allocate(QWTensorShape([1], .i32))
        arena.write(token, from: [Int32(5)])
        let out = try arena.activationsRegion().allocate(QWTensorShape([hidden * 2], .f32))
        guard let (cb, enc) = hal.begin() else { return XCTFail("no command buffer") }
        try table.gather(enc: enc, hal: hal, token: token, out: out, rows: 1, hc: 2)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        let got = arena.read(out, as: Float.self, count: hidden * 2)

        for d in 0..<hidden {
            let code = Double(made.codes[5][d])
            let want = code * Double(scale) + Double(bias)
            for c in 0..<2 {
                XCTAssertLessThanOrEqual(abs(Double(got[c * hidden + d]) - want),
                                         abs(want) * 1e-5 + 1e-6,
                                         "column \(d), copy \(c): got \(got[c * hidden + d]), want \(want)")
            }
        }
    }

    /// A pair the family does not hold is not a refusal: the incumbent's
    /// container holds any width. The reason is NAMED.
    func testAPairTheFamilyDoesNotHoldKeepsTheIncumbentsContainerAndNamesIt() throws {
        let vocab = 8, hidden = 256, bits = 5, group = 64
        let groups = hidden / group
        let scale = Float(1025) / Float(4096), bias = Float(-683) / Float(4096)
        let made = try makeCheckpoint(vocab: vocab, hidden: hidden, bits: bits, group: group,
                                      scales: [Float](repeating: scale, count: vocab * groups),
                                      biases: [Float](repeating: bias, count: vocab * groups))
        let payloadBytes = made.bytes.count - vocab * groups * 2 * 2
        let headerBytes = Array(made.bytes[payloadBytes...])
        try writeSafetensors("model.safetensors", tensors: [
            ("model.embed_tokens.weight", [vocab, hidden * bits / 32], "U32", Array(made.bytes.prefix(payloadBytes))),
            ("model.embed_tokens.scales", [vocab, groups], "BF16", Array(headerBytes.prefix(vocab * groups * 2))),
            ("model.embed_tokens.biases", [vocab, groups], "BF16", Array(headerBytes.dropFirst(vocab * groups * 2))),
        ])
        let container = try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors"))
        let hal = try QWHal(geometry: try Self.geometry())
        let arena = try QWArena(hal: hal, weights: (4 << 20), activations: (4 << 20), kv: 0, gdnStates: 0)

        let table = try FNGPUEmbedTable.load(arena: arena, container: container,
                                             name: "model.embed_tokens.weight",
                                             vocab: vocab, hidden: hidden, hc: 1)
        guard case .bf16 = table.storage else {
            return XCTFail("a pair outside the family must keep the incumbent's container")
        }
        XCTAssertTrue(table.containerNote.contains("5-bit group 64"),
                      "the reason must be named: \(table.containerNote)")
    }
}
