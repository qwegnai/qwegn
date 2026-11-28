// QWWave2KernelParityTests.swift — Parity tests for inverted-loop AQ rows4_m and SHVQ GDN rows4_m_body (M=2,4,5,8 vs M=1).
import Foundation
import Metal
import XCTest
@testable import QWGEMM
import QWAlloc
import QWCore
import QWHAL
import QWQuant

final class QWWave2KernelParityTests: XCTestCase {
    private static func makeHal() throws -> QWHal {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && MTLCreateSystemDefaultDevice() != nil,
            "Wave 2 Metal kernel parity requires bare-metal Apple Silicon GPU"
        )
        return try QWHal(geometry: try makeHybridGeometry())
    }

    private static func makeHybridGeometry() throws -> QWModelGeometry {
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

    private func makeAQBank(bits: Int, group: Int, n: Int, k: Int, seedInit: UInt32 = 0x1234_5678) throws -> Data {
        var seed = seedInit ^ UInt32(bits * 97 + group)
        var bank = Data()
        let mask = UInt32((1 << bits) - 1)
        for r in 0..<n {
            for g in 0..<(k / group) {
                let vals = (0..<group).map { idx -> UInt32 in
                    seed = seed &* 1664525 &+ 1013904223
                    return ((seed >> 12) ^ UInt32(r * 7 + g * 11 + idx)) & mask
                }
                let sc = Float(Float16(0.025 * (1.0 + 0.1 * Float((r + g) % 5))))
                let bi = Float(Float16(-0.35 + 0.05 * Float((r + g) % 4)))
                bank.append(try QWAQBlock.pack(bits: bits, group: group, values: vals, scale: sc, bias: bi))
            }
        }
        return bank
    }

    private func makeSHVQBank(bits: Int, group: Int, n: Int, k: Int, seedInit: UInt32 = 0x9E37_79B9) -> Data {
        let groups = k / group
        let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
        var seed = seedInit ^ UInt32(bits * 131 + group)
        var bank = Data(count: n * groups * gBytes)
        bank.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let base = (r * groups + g) * gBytes
                    let sc = Float16(0.05 * (1.0 + 0.15 * Float((r + g) % 7))).bitPattern
                    ptr[base] = UInt8(sc & 0xFF)
                    ptr[base + 1] = UInt8(sc >> 8)
                    for b in 2..<gBytes {
                        seed = seed &* 1664525 &+ 1013904223
                        ptr[base + b] = UInt8((seed >> 16) & 0xFF)
                    }
                }
            }
        }
        return bank
    }

    private func makeActivations(count: Int, seedInit: UInt32) -> [Float16] {
        var seed = seedInit
        return (0..<count).map { _ in
            seed = seed &* 1664525 &+ 1013904223
            return Float16(Float(Int32(truncatingIfNeeded: seed >> 8) % 512) / 512.0 - 0.5)
        }
    }

    private func step(_ hal: QWHal, _ f: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
        try f(enc)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "GPU step failed: \(String(describing: cb.error))")
    }

    func testAQRows4MParityVsM1AcrossBitWidths() throws {
        let hal = try Self.makeHal()
        let n = 19, k = 256, maxM = 8
        let specs: [(bits: Int, group: Int)] = [(2, 32), (3, 64), (4, 32), (6, 64), (8, 32)]

        for (bits, group) in specs {
            let bank = try makeAQBank(bits: bits, group: group, n: n, k: k)
            let arena = try QWArena(hal: hal, weights: bank.count + (1 << 20),
                                    activations: (4 << 20), kv: 0, gdnStates: 0)
            let wBuf = try arena.weightsRegion().allocate(QWTensorShape([(bank.count + 1) / 2], .f16))
            arena.copyIn(wBuf, from: bank)
            let act = arena.activationsRegion()
            let xVals = makeActivations(count: maxM * k, seedInit: UInt32(0x3100 + bits * 19 + group))
            let xBuf = try act.allocate(QWTensorShape([maxM * k], .f16))
            arena.write(xBuf, from: xVals)

            let yRef16 = try act.allocate(QWTensorShape([maxM * n], .f16))
            let yRef32 = try act.allocate(QWTensorShape([maxM * n], .f32))
            try step(hal) { enc in
                for r in 0..<maxM {
                    let xr = xBuf.slice(byteOffset: r * k * 2, count: k, dtype: .f16)
                    let yr16 = yRef16.slice(byteOffset: r * n * 2, count: n, dtype: .f16)
                    let yr32 = yRef32.slice(byteOffset: r * n * 4, count: n, dtype: .f32)
                    try QWGemmAQ.encodeDecodeT4(enc, hal: hal, blocks: wBuf, x: xr, y16: yr16,
                                                n: n, k: k, bits: bits, group: group, rows: 1, isSHVQ: false)
                    try QWGemmAQ.encode(enc, hal: hal, blocks: wBuf, x: xr, y: yr32,
                                        n: n, k: k, bits: bits, group: group, rows: 1, isSHVQ: false)
                }
            }
            let ref16 = arena.read(yRef16, as: UInt16.self, count: maxM * n).map { Float(Float16(bitPattern: $0)) }
            let ref32 = arena.read(yRef32, as: Float.self, count: maxM * n)

            for m in [2, 4, 5, 8] {
                let xm = xBuf.slice(byteOffset: 0, count: m * k, dtype: .f16)
                let ym16 = try act.allocate(QWTensorShape([m * n], .f16))
                let ym32 = try act.allocate(QWTensorShape([m * n], .f32))
                try step(hal) { enc in
                    try QWGemmAQ.encodeDecodeT4(enc, hal: hal, blocks: wBuf, x: xm, y16: ym16,
                                                n: n, k: k, bits: bits, group: group, rows: m, isSHVQ: false)
                    try QWGemmAQ.encode(enc, hal: hal, blocks: wBuf, x: xm, y: ym32,
                                        n: n, k: k, bits: bits, group: group, rows: m, isSHVQ: false)
                }
                let got16 = arena.read(ym16, as: UInt16.self, count: m * n).map { Float(Float16(bitPattern: $0)) }
                let got32 = arena.read(ym32, as: Float.self, count: m * n)
                let tol16: Float = 0.0
                let tol32: Float = 0.0
                for i in 0..<(m * n) {
                    XCTAssertEqual(got16[i], ref16[i], accuracy: tol16,
                                   "AQ\(bits)_G\(group) dx_h16_t4 m=\(m) idx=\(i)")
                    XCTAssertEqual(got32[i], ref32[i], accuracy: tol32,
                                   "AQ\(bits)_G\(group) v4_f32 m=\(m) idx=\(i)")
                }
            }
        }
    }

    func testSHVQGDNRows4MParityVsM1AcrossFormats() throws {
        let hal = try Self.makeHal()
        let geom = hal.geometry
        let k = geom.hidden
        let vHeads = geom.gdnVHeads
        let totalN = geom.gdnQKVChannels + geom.gdnZChannels + 2 * vHeads
        let ctrlStride = 2 * vHeads
        let maxM = 8
        let specs: [(bits: Int, group: Int)] = [(1, 64), (15, 32), (15, 64), (2, 32), (3, 64), (4, 32)]

        for (bits, group) in specs {
            let bank = makeSHVQBank(bits: bits, group: group, n: totalN, k: k)
            let kernelName = try XCTUnwrap(QWGemmAQ.gdnInProjDecodeKernelName(bits: bits, group: group, isSHVQ: true))
            let pso = try hal.pipeline(kernelName)

            let arena = try QWArena(hal: hal, weights: bank.count + (1 << 20),
                                    activations: (4 << 20), kv: 0, gdnStates: 0)
            let wBuf = try arena.weightsRegion().allocate(QWTensorShape([(bank.count + 1) / 2], .f16))
            arena.copyIn(wBuf, from: bank)
            let act = arena.activationsRegion()
            let xVals = makeActivations(count: maxM * k, seedInit: UInt32(0x6200 + bits * 23 + group))
            let xBuf = try act.allocate(QWTensorShape([maxM * k], .f16))
            arena.write(xBuf, from: xVals)
            let aLogBuf = try act.allocate(QWTensorShape([vHeads], .f32))
            arena.write(aLogBuf, from: (0..<vHeads).map { -1.5 + 0.25 * Float($0) })
            let dtBiasBuf = try act.allocate(QWTensorShape([vHeads], .f32))
            arena.write(dtBiasBuf, from: (0..<vHeads).map { 0.2 - 0.1 * Float($0) })

            let yRef16 = try act.allocate(QWTensorShape([maxM * totalN], .f16))
            let yRef32 = try act.allocate(QWTensorShape([totalN], .f32))
            let ctrlRef = try act.allocate(QWTensorShape([maxM * ctrlStride], .f32))
            try step(hal) { enc in
                enc.setComputePipelineState(pso)
                wBuf.bind(enc, index: 0)
                yRef32.bind(enc, index: 3)
                aLogBuf.bind(enc, index: 9)
                dtBiasBuf.bind(enc, index: 10)
                var kU = UInt32(k)
                var nFlag = UInt32(totalN) | 0x4000_0000
                enc.setBytes(&nFlag, length: 4, index: 4)
                enc.setBytes(&kU, length: 4, index: 5)
                for r in 0..<maxM {
                    let xr = xBuf.slice(byteOffset: r * k * 2, count: k, dtype: .f16)
                    let yr16 = yRef16.slice(byteOffset: r * totalN * 2, count: totalN, dtype: .f16)
                    let br = ctrlRef.slice(byteOffset: r * ctrlStride * 4, count: vHeads, dtype: .f32)
                    let gr = ctrlRef.slice(byteOffset: (r * ctrlStride + vHeads) * 4, count: vHeads, dtype: .f32)
                    xr.bind(enc, index: 1)
                    yr16.bind(enc, index: 8)
                    br.bind(enc, index: 11)
                    gr.bind(enc, index: 12)
                    enc.dispatchThreadgroups(MTLSize(width: (totalN + 15) / 16, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                }
            }
            let ref16 = arena.read(yRef16, as: UInt16.self, count: maxM * totalN).map { Float(Float16(bitPattern: $0)) }
            let refCtrl = arena.read(ctrlRef, as: Float.self, count: maxM * ctrlStride)

            for m in [2, 4, 5, 8] {
                let xm = xBuf.slice(byteOffset: 0, count: m * k, dtype: .f16)
                let ym16 = try act.allocate(QWTensorShape([m * totalN], .f16))
                let ctrlM = try act.allocate(QWTensorShape([m * ctrlStride], .f32))
                try step(hal) { enc in
                    enc.setComputePipelineState(pso)
                    wBuf.bind(enc, index: 0)
                    xm.bind(enc, index: 1)
                    yRef32.bind(enc, index: 3)
                    var nFlag = UInt32(totalN) | 0x4000_0000 | (UInt32(m) << 24)
                    var kU = UInt32(k)
                    enc.setBytes(&nFlag, length: 4, index: 4)
                    enc.setBytes(&kU, length: 4, index: 5)
                    ym16.bind(enc, index: 8)
                    aLogBuf.bind(enc, index: 9)
                    dtBiasBuf.bind(enc, index: 10)
                    ctrlM.bind(enc, index: 11)
                    ctrlM.bind(enc, index: 12)
                    enc.dispatchThreadgroups(MTLSize(width: (totalN + 15) / 16, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                }
                let got16 = arena.read(ym16, as: UInt16.self, count: m * totalN).map { Float(Float16(bitPattern: $0)) }
                let gotCtrl = arena.read(ctrlM, as: Float.self, count: m * ctrlStride)
                for i in 0..<(m * totalN) {
                    XCTAssertEqual(got16[i], ref16[i], accuracy: 1e-3,
                                   "SHVQ\(bits)_G\(group) GDN y16 m=\(m) idx=\(i)")
                }
                for i in 0..<(m * ctrlStride) {
                    XCTAssertEqual(gotCtrl[i], refCtrl[i], accuracy: 1e-4,
                                   "SHVQ\(bits)_G\(group) GDN ctrl m=\(m) idx=\(i)")
                }
            }
        }
    }
}
