import Foundation
import Metal
import XCTest
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWQuant

final class QWSmallPrefillTests: XCTestCase {
    func testHQ8SmallTileMatchesOriginalFloatBits() throws {
        for basis in [QWWeightBasis.rotated, .standard] {
            let hal = try QWHal(geometry: .qwen38_27b, weightBasis: basis)
            for (n, k) in [(19, 64), (64, 2560), (320, 320), (32, 6144), (64, 128), (64, 96), (320, 64)] {
                var data = Data()
                for row in 0..<n {
                    for group in 0..<(k / 32) {
                        let codes = (0..<32).map { UInt32(($0 * 13 + row * 7 + group * 3) % 256) }
                        data.append(try QWAQBlock.pack(bits: 8, group: 32, values: codes,
                                                      scale: 0.00390625, bias: -0.5))
                    }
                }
                let arena = try QWArena(hal: hal, weights: data.count + (1 << 20),
                                        activations: 16 << 20, kv: 0, gdnStates: 0)
                let blocks = try arena.weightsRegion().allocate(QWTensorShape([data.count / 2], .f16))
                arena.copyIn(blocks, from: data)
                let weight = try XCTUnwrap(QWWeightView(blocks: blocks, n: n, k: k, bits: 8,
                                                        groupSize: 32, isRotated: basis == .rotated))
                let act = arena.activationsRegion()
                for rows in [31, 32, 33, 48, 63, 64, 65] {
                    let x = try act.allocate(QWTensorShape([rows * k], .f32))
                    let outputs = try [act.allocate(QWTensorShape([rows * n], .f32)),
                                       act.allocate(QWTensorShape([rows * n], .f32))]
                    var seed: UInt32 = 718
                    arena.write(x, from: (0..<(rows * k)).map { _ -> Float in
                        seed = seed &* 1664525 &+ 1013904223
                        return Float(Int(seed >> 16) - 32768) / 32768
                    })
                    for epi: UInt32 in [0, 1, 2, 3] {
                        if epi == 3 && n % 32 != 0 { continue }
                        let scale: Float = epi == 3 ? 4 : 2
                        guard let (cb, enc) = hal.begin() else {
                            XCTFail("could not create Metal command buffer")
                            return
                        }
                        enc.setComputePipelineState(try hal.pipeline("qwg_gemm_aq8_g32_f32_bm"))
                        blocks.bind(enc, index: 0)
                        x.bind(enc, index: 1)
                        outputs[0].bind(enc, index: 2)
                        enc.setUInt32s(rows, n, k, Int(epi), startIndex: 3)
                        enc.setPod(scale, index: 7)
                        enc.dispatchThreadgroups(MTLSize(width: (n + 31) / 32, height: (rows + 63) / 64, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)
                        try QWGemm.encodeHQ4F32BM(enc, hal: hal, weight: weight, x: x,
                                                 y: outputs[1], rows: rows, epi: epi, scale: scale)
                        hal.end(cb, enc)
                        cb.waitUntilCompleted()
                        XCTAssertNil(cb.error)
                        let a = arena.read(outputs[0], as: Float.self, count: rows * n)
                        let b = arena.read(outputs[1], as: Float.self, count: rows * n)
                        XCTAssertTrue(a.allSatisfy(\.isFinite))
                        XCTAssertEqual(a.map(\.bitPattern), b.map(\.bitPattern),
                                       "\(basis) \(n)x\(k), rows \(rows), epi \(epi)")
                    }
                }
            }
        }
    }

    func testSHVQ4SmallTileMatchesOriginalFloatBits() throws {
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)
        for group in [32, 64] {
            let gBytes = QWSHVQBlock.bytes(bits: 4, group: group)
            for (n, k) in [(19, 64), (64, 2560), (320, 320), (32, 6144), (64, 128), (320, 64)] {
                var data = Data(count: n * (k / group) * gBytes)
                data.withUnsafeMutableBytes { raw in
                    guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                    var s: UInt32 = 0x9E37_79B9 ^ UInt32(group)
                    for r in 0..<n {
                        for g in 0..<(k / group) {
                            let base = (r * (k / group) + g) * gBytes
                            let sc = Float16(0.03125 * Float(1 + (r + g) % 4)).bitPattern
                            ptr[base] = UInt8(sc & 0xFF)
                            ptr[base + 1] = UInt8(sc >> 8)
                            for b in 2..<gBytes {
                                s = s &* 1664525 &+ 1013904223
                                ptr[base + b] = UInt8((s >> 16) & 0xFF)
                            }
                        }
                    }
                }
                let arena = try QWArena(hal: hal, weights: data.count + (1 << 20),
                                        activations: 16 << 20, kv: 0, gdnStates: 0)
                let blocks = try arena.weightsRegion().allocate(QWTensorShape([(data.count + 1) / 2], .f16))
                arena.copyIn(blocks, from: data)
                let weight = try XCTUnwrap(QWWeightView(blocks: blocks, n: n, k: k, bits: 4,
                                                        groupSize: group, isRotated: true,
                                                        isSHVQ: true))
                let act = arena.activationsRegion()
                let refName = "qwg_gemm_shvq4_g\(group)_f32_bm"
                for rows in [31, 32, 33, 48, 63, 64, 65] {
                    let x = try act.allocate(QWTensorShape([rows * k], .f32))
                    let outputs = try [act.allocate(QWTensorShape([rows * n], .f32)),
                                       act.allocate(QWTensorShape([rows * n], .f32))]
                    var seed: UInt32 = 919 ^ UInt32(rows)
                    arena.write(x, from: (0..<(rows * k)).map { _ -> Float in
                        seed = seed &* 1664525 &+ 1013904223
                        return Float(Int(seed >> 16) - 32768) / 32768
                    })
                    for epi: UInt32 in [0, 1, 2, 3] {
                        if epi == 3 && n % 32 != 0 { continue }
                        let scale: Float = epi == 3 ? 4 : 2
                        guard let (cb, enc) = hal.begin() else {
                            XCTFail("could not create Metal command buffer")
                            return
                        }
                        enc.setComputePipelineState(try hal.pipeline(refName))
                        blocks.bind(enc, index: 0)
                        x.bind(enc, index: 1)
                        outputs[0].bind(enc, index: 2)
                        enc.setUInt32s(rows, n, k, Int(epi), startIndex: 3)
                        enc.setPod(scale, index: 7)
                        enc.dispatchThreadgroups(
                            MTLSize(width: (n + 31) / 32, height: (rows + 63) / 64, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)
                        try QWGemm.encodeHQ4F32BM(enc, hal: hal, weight: weight, x: x,
                                                 y: outputs[1], rows: rows, epi: epi, scale: scale)
                        hal.end(cb, enc)
                        cb.waitUntilCompleted()
                        XCTAssertNil(cb.error)
                        let a = arena.read(outputs[0], as: Float.self, count: rows * n)
                        let b = arena.read(outputs[1], as: Float.self, count: rows * n)
                        XCTAssertTrue(a.allSatisfy(\.isFinite))
                        XCTAssertEqual(a.map(\.bitPattern), b.map(\.bitPattern),
                                       "SHVQ4 g\(group) \(n)x\(k), rows \(rows), epi \(epi)")
                    }
                }
            }
        }
    }

    func testHQ4AndAQ4SmallTileMatchesOriginalFloatBits() throws {
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)
        for isIncumbent in [true, false] {
            let groups = isIncumbent ? [32] : [32, 64]
            for group in groups {
                for (n, k) in [(19, 64), (64, 2560), (320, 320), (32, 6144), (64, 128), (320, 64)] {
                    var data = Data()
                    for row in 0..<n {
                        for g in 0..<(k / group) {
                            let codes = (0..<group).map { UInt32(($0 * 7 + row * 5 + g * 3) % 16) }
                            if isIncumbent {
                                data.append(QWAQProducer.hq4G32Block(codes: codes, scale: 1.0 / 16))
                            } else {
                                data.append(try QWAQBlock.pack(bits: 4, group: group, values: codes, scale: 1.0 / 16, bias: -0.5))
                            }
                        }
                    }
                    let arena = try QWArena(hal: hal, weights: data.count + (1 << 20), activations: 16 << 20, kv: 0, gdnStates: 0)
                    let blocks = try arena.weightsRegion().allocate(QWTensorShape([data.count / 2], .f16))
                    arena.copyIn(blocks, from: data)
                    let weight = try XCTUnwrap(QWWeightView(
                        blocks: blocks, n: n, k: k, bits: 4, groupSize: group,
                        isRotated: true, isIncumbentContainer: isIncumbent))
                    let act = arena.activationsRegion()
                    let refName = isIncumbent ? "qwg_gemm_q4g_f32_bm" : "qwg_gemm_aq4_g\(group)_f32_bm"
                    for rows in [31, 32, 33, 48, 63, 64, 65] {
                        let x = try act.allocate(QWTensorShape([rows * k], .f32))
                        let outputs = try [act.allocate(QWTensorShape([rows * n], .f32)),
                                           act.allocate(QWTensorShape([rows * n], .f32))]
                        var seed: UInt32 = 412 ^ UInt32(rows + group)
                        arena.write(x, from: (0..<(rows * k)).map { _ -> Float in
                            seed = seed &* 1664525 &+ 1013904223
                            return Float(Int(seed >> 16) - 32768) / 32768
                        })
                        for epi: UInt32 in [0, 1, 2, 3] {
                            if epi == 3 && n % 32 != 0 { continue }
                            let scale: Float = epi == 3 ? 4 : 2
                            guard let (cb, enc) = hal.begin() else {
                                XCTFail("could not create Metal command buffer"); return
                            }
                            enc.setComputePipelineState(try hal.pipeline(refName))
                            blocks.bind(enc, index: 0); x.bind(enc, index: 1); outputs[0].bind(enc, index: 2)
                            enc.setUInt32s(rows, n, k, Int(epi), startIndex: 3); enc.setPod(scale, index: 7)
                            enc.dispatchThreadgroups(MTLSize(width: (n + 31) / 32, height: (rows + 63) / 64, depth: 1),
                                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                            enc.memoryBarrier(scope: .buffers)
                            try QWGemm.encodeHQ4F32BM(enc, hal: hal, weight: weight, x: x,
                                                      y: outputs[1], rows: rows, epi: epi, scale: scale)
                            hal.end(cb, enc); cb.waitUntilCompleted(); XCTAssertNil(cb.error)
                            let a = arena.read(outputs[0], as: Float.self, count: rows * n)
                            let b = arena.read(outputs[1], as: Float.self, count: rows * n)
                            XCTAssertTrue(a.allSatisfy(\.isFinite))
                            XCTAssertEqual(a.map(\.bitPattern), b.map(\.bitPattern),
                                           "\(refName) \(n)x\(k), rows \(rows), epi \(epi)")
                        }
                    }
                }
            }
        }
    }
}
