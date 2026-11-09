import XCTest
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWQuant

/// The load-time writers on the GPU, against the host contract.
/// The quantiser writes the incumbent packer's bytes; the repacker keeps
/// MLX's codes where they are.
final class FNMetalOnlineQuantTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    /// The committed goldens, written by Scripts/oracle/gen_aq_fixtures.py.
    static let fixtureRoot: URL = {
        var d = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<8 {
            let c = d.appendingPathComponent("Tests/QWQuantTests/fixtures")
            if FileManager.default.fileExists(atPath: c.path) { return c }
            d = d.deletingLastPathComponent()
        }
        fatalError("Tests/QWQuantTests/fixtures not found")
    }()

    static func golden(_ name: String) -> [String: Any] {
        let url = fixtureRoot.appendingPathComponent(name)
        guard let data = try? Data(contentsOf: url),
              let doc = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            fatalError("cannot read \(url)")
        }
        return doc
    }

    static func bytes(_ hex: String) -> [UInt8] {
        var out: [UInt8] = []
        out.reserveCapacity(hex.count / 2)
        var i = hex.startIndex
        while i < hex.endIndex {
            let next = hex.index(i, offsetBy: 2)
            out.append(UInt8(hex[i..<next], radix: 16) ?? 0)
            i = next
        }
        return out
    }

    private func step(_ f: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
        try f(enc)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "step failed: \(String(describing: cb.error))")
    }

    /// A deterministic weight matrix in the range the packer sees.
    private func makeWeights(rows: Int, cols: Int) -> [Float] {
        var seed: UInt32 = 0xB16B00B5
        return (0..<(rows * cols)).map { _ in
            seed = seed &* 1664525 &+ 1013904223
            return Float(Int32(truncatingIfNeeded: seed >> 8) % 2048) / 1024.0 - 1.0
        }
    }

    /// The f32 value as the bf16 bits the kernel reads.
    private func bf16(_ value: Float) -> UInt16 {
        var bits = value.bitPattern
        let rounding: UInt32 = 0x7FFF + ((bits >> 16) & 1)
        bits = (bits &+ rounding) & 0xFFFF_0000
        return UInt16(truncatingIfNeeded: bits >> 16)
    }

    /// The value as the kernel will read it: a checkpoint's scale is bf16,
    /// and bf16's 8 significand bits fit f16's 11 exactly.
    private func bf16Value(_ value: Float) -> Float {
        Float(bitPattern: UInt32(bf16(value)) << 16)
    }

    func testTheGPUQuantiserWritesTheHostBytes() throws {
        let rows = 4, cols = 64
        let weights = makeWeights(rows: rows, cols: cols)
        let src = weights.map { bf16($0) }
        // The device reads bf16, so the host reference must read the same
        // values the bf16 holds, not the f32 they came from.
        let rounded = src.map { Float(bitPattern: UInt32($0) << 16) }
        let bytes = rows * (cols / 32) * 20
        let arena = try QWArena(hal: Self.hal, weights: bytes + (1 << 20),
                                activations: (1 << 20), kv: 0, gdnStates: 0)
        let inBuf = try arena.weightsRegion().allocate(QWTensorShape([src.count], .f16))
        arena.write(inBuf, from: src)
        let out = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
        try step { enc in
            let pso = try Self.hal.pipeline("qwg_quantize_bf16_to_hq4_g32_m5")
            enc.setComputePipelineState(pso)
            inBuf.bind(enc, index: 0)
            out.bind(enc, index: 1)
            var r32 = UInt32(rows), c32 = UInt32(cols)
            enc.setBytes(&r32, length: 4, index: 4)
            enc.setBytes(&c32, length: 4, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: 64, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
        let got = arena.read(out, as: UInt8.self, count: bytes)
        var expected = [UInt8]()
        for r in 0..<rows {
            for g in 0..<(cols / 32) {
                let slice = Array(rounded[(r * cols + g * 32)..<(r * cols + g * 32 + 32)])
                expected.append(contentsOf: QWAQProducer.hq4G32Block(slice))
            }
        }
        XCTAssertEqual(got, expected, "the GPU quantiser must write the host's bytes")
    }

    func testTheGPURepackerWritesTheHostBytes() throws {
        // MLX's planes: the packed codes, one scale per group, one bias per
        // group. The 4-bit codes are packed densely, as MLX packs them.
        let rows = 3, cols = 128, group = 64, bits = 4
        let groups = cols / group
        var packed = [UInt8]()
        var scales = [Float]()
        var biases = [Float]()
        for r in 0..<rows {
            for g in 0..<groups {
                var groupBytes = [UInt8](repeating: 0, count: group * bits / 8)
                for i in 0..<group {
                    let code = UInt8((i + r * 3 + g * 5) % 16)
                    let b = i / 2
                    groupBytes[b] = i % 2 == 0 ? UInt8(groupBytes[b] | code)
                                              : UInt8(groupBytes[b] | (code << 4))
                }
                packed.append(contentsOf: groupBytes)
                scales.append(bf16Value(0.01 + Float(r) * 0.002 + Float(g) * 0.001))
                biases.append(bf16Value(-0.5 - Float(r) * 0.01))
            }
        }
        let groupBytes = group * bits / 8 + 4
        let bytes = rows * groups * groupBytes
        let arena = try QWArena(hal: Self.hal, weights: bytes + (1 << 20),
                                activations: (1 << 20), kv: 0, gdnStates: 0)
        let inBuf = try arena.weightsRegion().allocate(QWTensorShape([packed.count], .f16))
        arena.copyIn(inBuf, from: Data(packed))
        let scaleBuf = try arena.activationsRegion().allocate(QWTensorShape([scales.count], .f16))
        arena.write(scaleBuf, from: scales.map { bf16($0) })
        let biasBuf = try arena.activationsRegion().allocate(QWTensorShape([biases.count], .f16))
        arena.write(biasBuf, from: biases.map { bf16($0) })
        let flags = try arena.activationsRegion().allocate(QWTensorShape([1], .i32))
        arena.write(flags, from: [Int32(0)])
        let out = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
        try step { enc in
            let pso = try Self.hal.pipeline("qwg_repack_mlx_affine_to_aq_m5")
            enc.setComputePipelineState(pso)
            inBuf.bind(enc, index: 0)
            scaleBuf.bind(enc, index: 1)
            biasBuf.bind(enc, index: 2)
            out.bind(enc, index: 3)
            var r32 = UInt32(rows), c32 = UInt32(cols), b32 = UInt32(bits), g32 = UInt32(group)
            var dt: UInt32 = 0
            enc.setBytes(&r32, length: 4, index: 4)
            enc.setBytes(&c32, length: 4, index: 5)
            enc.setBytes(&b32, length: 4, index: 6)
            enc.setBytes(&g32, length: 4, index: 7)
            enc.setBytes(&dt, length: 4, index: 8)
            flags.bind(enc, index: 9)
            enc.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        XCTAssertEqual(arena.read(flags, as: Int32.self, count: 1)[0], 0,
                       "a scale or a bias f16 cannot hold must be flagged, not rounded")
        let got = arena.read(out, as: UInt8.self, count: bytes)
        var expected = [UInt8]()
        let payload = group * bits / 8
        for r in 0..<rows {
            for g in 0..<groups {
                let at = (r * groups + g) * payload
                expected.append(contentsOf: try QWAQProducer.repackMLXGroup(
                    bits: bits, group: group, packed: Array(packed[at..<(at + payload)]),
                    scale: scales[r * groups + g], bias: biases[r * groups + g]))
            }
        }
        XCTAssertEqual(got, expected, "the GPU repacker must write the host's bytes")
    }

    func testTheGPURepackerCopiesTheOtherWidthsVerbatim() throws {
        // The 8-bit width has no magic and no de-interleave: the payload is
        // copied and the bias is verbatim.
        let rows = 2, cols = 64, group = 64, bits = 8
        let groups = cols / group
        let payload = group * bits / 8
        var packed = [UInt8]()
        var scales = [Float]()
        var biases = [Float]()
        for r in 0..<rows {
            for g in 0..<groups {
                packed.append(contentsOf: (0..<payload).map { UInt8(($0 * 5 + r * 3 + g) % 256) })
                scales.append(bf16Value(0.02 + Float(r) * 0.003 + Float(g) * 0.001))
                biases.append(bf16Value(-0.25 + Float(g) * 0.01))
            }
        }
        let groupBytes = payload + 4
        let bytes = rows * groups * groupBytes
        let arena = try QWArena(hal: Self.hal, weights: bytes + (1 << 20),
                                activations: (1 << 20), kv: 0, gdnStates: 0)
        let inBuf = try arena.weightsRegion().allocate(QWTensorShape([packed.count], .f16))
        arena.copyIn(inBuf, from: Data(packed))
        let scaleBuf = try arena.activationsRegion().allocate(QWTensorShape([scales.count], .f16))
        arena.write(scaleBuf, from: scales.map { bf16($0) })
        let biasBuf = try arena.activationsRegion().allocate(QWTensorShape([biases.count], .f16))
        arena.write(biasBuf, from: biases.map { bf16($0) })
        let flags = try arena.activationsRegion().allocate(QWTensorShape([1], .i32))
        arena.write(flags, from: [Int32(0)])
        let out = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
        try step { enc in
            let pso = try Self.hal.pipeline("qwg_repack_mlx_affine_to_aq_m5")
            enc.setComputePipelineState(pso)
            inBuf.bind(enc, index: 0)
            scaleBuf.bind(enc, index: 1)
            biasBuf.bind(enc, index: 2)
            out.bind(enc, index: 3)
            var r32 = UInt32(rows), c32 = UInt32(cols), b32 = UInt32(bits), g32 = UInt32(group)
            var dt: UInt32 = 0
            enc.setBytes(&r32, length: 4, index: 4)
            enc.setBytes(&c32, length: 4, index: 5)
            enc.setBytes(&b32, length: 4, index: 6)
            enc.setBytes(&g32, length: 4, index: 7)
            enc.setBytes(&dt, length: 4, index: 8)
            flags.bind(enc, index: 9)
            enc.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        XCTAssertEqual(arena.read(flags, as: Int32.self, count: 1)[0], 0,
                       "a scale or a bias f16 cannot hold must be flagged, not rounded")
        let got = arena.read(out, as: UInt8.self, count: bytes)
        for b in 0..<(rows * groups) {
            let at = b * groupBytes
            let want = try QWAQProducer.repackMLXGroup(
                bits: bits, group: group,
                packed: Array(packed[(b * payload)..<(b * payload + payload)]),
                scale: scales[b], bias: biases[b])
            XCTAssertEqual(Array(got[at..<(at + groupBytes)]), [UInt8](want), "group \(b)")
        }
    }

    func testTheGPUQuantiserWritesThePythonPackersBytes() throws {
        // The golden is the incumbent Python packer's own output
        // (Scripts/quantize/hq4.py) for a fixed weight matrix.
        let doc = Self.golden("aq_quantizer_golden.json")
        let rows = doc["rows"] as! Int
        let cols = doc["cols"] as! Int
        let src = Self.bytes(doc["bf16"] as! String)
        let want = Self.bytes(doc["blocks"] as! String)
        XCTAssertEqual(src.count, rows * cols * 2, "the golden holds bf16 values")
        let arena = try QWArena(hal: Self.hal, weights: want.count + (1 << 20),
                                activations: (1 << 20), kv: 0, gdnStates: 0)
        let inBuf = try arena.weightsRegion().allocate(QWTensorShape([src.count / 2], .f16))
        arena.write(inBuf, from: src)
        let out = try arena.weightsRegion().allocate(QWTensorShape([want.count / 2], .f16))
        try step { enc in
            let pso = try Self.hal.pipeline("qwg_quantize_bf16_to_hq4_g32_m5")
            enc.setComputePipelineState(pso)
            inBuf.bind(enc, index: 0)
            out.bind(enc, index: 1)
            var r32 = UInt32(rows), c32 = UInt32(cols)
            enc.setBytes(&r32, length: 4, index: 4)
            enc.setBytes(&c32, length: 4, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: 64, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
        let got = arena.read(out, as: UInt8.self, count: want.count)
        XCTAssertEqual(got, want, "the GPU quantiser must match the Python packer byte for byte")
    }

    func testTheGPURepackerReproducesMLXsCodes() throws {
        // The golden is MLX's own: its packed plane, its codes and its
        // dequantised values. The codes are what the repack must preserve.
        let doc = Self.golden("aq_repack_golden.json")
        let rows = doc["rows"] as! Int
        let cols = doc["cols"] as! Int
        let group = doc["group"] as! Int
        let bits = doc["bits"] as! Int
        let packed = Self.bytes(doc["packed"] as! String)
        let scales = Self.bytes(doc["scales"] as! String)
        let biases = Self.bytes(doc["biases"] as! String)
        let wantCodes = Self.bytes(doc["codes"] as! String)
        let wantValues = Self.bytes(doc["values"] as! String)
        let groups = cols / group
        let payload = group * bits / 8
        let groupBytes = payload + 4
        let bytes = rows * groups * groupBytes
        let arena = try QWArena(hal: Self.hal, weights: bytes + (1 << 20),
                                activations: (1 << 20), kv: 0, gdnStates: 0)
        let inBuf = try arena.weightsRegion().allocate(QWTensorShape([packed.count], .f16))
        arena.copyIn(inBuf, from: Data(packed))
        // The golden's planes are f32; a checkpoint writes bf16, and the
        // kernel reads the dtype the checkpoint used.
        let scaleBuf = try arena.activationsRegion().allocate(QWTensorShape([scales.count / 4], .f16))
        arena.write(scaleBuf, from: (0..<(scales.count / 4)).map { i in
            let at = i * 4
            let bits = UInt32(scales[at]) | (UInt32(scales[at + 1]) << 8)
                | (UInt32(scales[at + 2]) << 16) | (UInt32(scales[at + 3]) << 24)
            return bf16(Float(bitPattern: bits))
        })
        let biasBuf = try arena.activationsRegion().allocate(QWTensorShape([biases.count / 4], .f16))
        arena.write(biasBuf, from: (0..<(biases.count / 4)).map { i in
            let at = i * 4
            let bits = UInt32(biases[at]) | (UInt32(biases[at + 1]) << 8)
                | (UInt32(biases[at + 2]) << 16) | (UInt32(biases[at + 3]) << 24)
            return bf16(Float(bitPattern: bits))
        })
        let flags = try arena.activationsRegion().allocate(QWTensorShape([1], .i32))
        arena.write(flags, from: [Int32(0)])
        let out = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
        try step { enc in
            let pso = try Self.hal.pipeline("qwg_repack_mlx_affine_to_aq_m5")
            enc.setComputePipelineState(pso)
            inBuf.bind(enc, index: 0)
            scaleBuf.bind(enc, index: 1)
            biasBuf.bind(enc, index: 2)
            out.bind(enc, index: 3)
            var r32 = UInt32(rows), c32 = UInt32(cols), b32 = UInt32(bits), g32 = UInt32(group)
            var dt: UInt32 = 0
            enc.setBytes(&r32, length: 4, index: 4)
            enc.setBytes(&c32, length: 4, index: 5)
            enc.setBytes(&b32, length: 4, index: 6)
            enc.setBytes(&g32, length: 4, index: 7)
            enc.setBytes(&dt, length: 4, index: 8)
            flags.bind(enc, index: 9)
            enc.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        let got = arena.read(out, as: UInt8.self, count: bytes)

        // The codes, exactly: the family's payload is one continuous
        // little-endian bit stream, so element j sits at bits
        // [bits*j, bits*(j+1)) — the bytes MLX itself wrote.
        var codes: [UInt8] = []
        for b in 0..<(rows * groups) {
            let payloadStart = b * groupBytes + 4
            for j in 0..<group {
                let bit = j * bits
                let byte = payloadStart + (bit >> 3)
                let off = UInt32(bit & 7)
                let take = min(8 - Int(off), bits)
                var code = UInt32(got[byte]) >> off
                if take < bits { code |= UInt32(got[byte + 1]) << UInt32(take) }
                codes.append(UInt8(code & ((1 << bits) - 1)))
            }
        }
        XCTAssertEqual(codes, wantCodes, "the repack must preserve MLX's codes")

        // The values, within what the f16 bias and scale cost.
        var decoded: [Float] = []
        for b in 0..<(rows * groups) {
            let at = b * groupBytes
            let block = Data(got[at..<(at + groupBytes)])
            var values: [Float] = []
            block.withUnsafeBytes { QWAQBlock.unpack(group: $0, bits: bits, group: group, into: &values) }
            decoded.append(contentsOf: values)
        }
        let mlx = wantValues.withUnsafeBytes { [Float]($0.bindMemory(to: Float.self)) }
        for i in 0..<min(decoded.count, mlx.count) {
            let at = (i / group) * 4
            let bits32 = UInt32(scales[at]) | (UInt32(scales[at + 1]) << 8)
                | (UInt32(scales[at + 2]) << 16) | (UInt32(scales[at + 3]) << 24)
            // MLX's affine scales are SIGNED — half the real checkpoint's
            // are negative — so the bound is on the magnitude.
            let scale = abs(Float(bitPattern: bits32))
            XCTAssertLessThan(abs(decoded[i] - mlx[i]), 0.6 * scale + 1e-6, "element \(i)")
        }
    }
}
