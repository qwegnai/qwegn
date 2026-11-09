import XCTest
import Foundation
import Metal
@testable import QWGEMM
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// The incumbent's decode GEMV, checked against a host reference over the
/// bench's own data: the block layout, the magic's fold, and the fact that
/// each block reads its own slice of the activations.
final class QWHQ4GEMVTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private func splitmix64(_ x: UInt64) -> UInt64 {
        var z = x &+ 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    private func seed(row: Int, group: Int) -> UInt64 {
        splitmix64(UInt64(bitPattern: Int64(row)) &* 0x0000_0001_0000_03B3
                   &+ UInt64(group) &* 0x9E37_79B9_7F4A_7C15)
    }

    private func codes(row: Int, group: Int, bits: Int, count: Int) -> [UInt32] {
        let s = seed(row: row, group: group)
        let mask = UInt64((1 << bits) - 1)
        var out = [UInt32](repeating: 0, count: count)
        for i in 0..<count { out[i] = UInt32(splitmix64(s &+ UInt64(i)) & mask) }
        return out
    }

    private func scale(row: Int, group: Int) -> Float {
        let s = splitmix64(seed(row: row, group: group) &+ 0x5DEE_CE66_D)
        let unit = Double(s % 1_000_000) / 1_000_000.0
        return Float(Float16(unit * 0.02 + 0.001))
    }

    private func activations(cols: Int) -> [Float] {
        var out = [Float](repeating: 0, count: cols)
        for i in 0..<cols {
            let unit = Double(splitmix64(UInt64(i) &* 0x9E37_79B9_7F4A_7C15) % 2_000_001)
                / 1_000_000.0 - 1.0
            out[i] = Float(Float16(unit))
        }
        return out
    }

    func testTheDecodeGEMVMatchesTheHost() throws {
        let rows = 64, cols = 256
        let groups = cols / 32
        var bank = [UInt8](repeating: 0, count: rows * groups * 20)
        for r in 0..<rows {
            for g in 0..<groups {
                let sc = scale(row: r, group: g)
                let cs = codes(row: r, group: g, bits: 4, count: 32)
                let bias = Float(Float16(-1032.0 * sc))
                let off = (r * groups + g) * 20
                bank[off] = UInt8(Float16(sc).bitPattern & 0xFF)
                bank[off + 1] = UInt8(Float16(sc).bitPattern >> 8)
                bank[off + 2] = UInt8(Float16(bias).bitPattern & 0xFF)
                bank[off + 3] = UInt8(Float16(bias).bitPattern >> 8)
                for i in 0..<32 {
                    let word = i / 8
                    let shift = UInt32(QWAQProducer.hq4NibbleBitOffset(i % 8))
                    var current = UInt32(bank[off + 4 + word * 4])
                        | (UInt32(bank[off + 4 + word * 4 + 1]) << 8)
                        | (UInt32(bank[off + 4 + word * 4 + 2]) << 16)
                        | (UInt32(bank[off + 4 + word * 4 + 3]) << 24)
                    current |= (cs[i] & 0xF) << shift
                    bank[off + 4 + word * 4] = UInt8(current & 0xFF)
                    bank[off + 4 + word * 4 + 1] = UInt8((current >> 8) & 0xFF)
                    bank[off + 4 + word * 4 + 2] = UInt8((current >> 16) & 0xFF)
                    bank[off + 4 + word * 4 + 3] = UInt8((current >> 24) & 0xFF)
                }
            }
        }
        let xs = activations(cols: cols)
        let device = Self.hal.device
        guard let weights = device.makeBuffer(length: bank.count, options: .storageModeShared),
              let x = device.makeBuffer(length: cols * 2, options: .storageModeShared),
              let y = device.makeBuffer(length: rows * 4, options: .storageModeShared) else {
            XCTFail("no buffer"); return
        }
        bank.withUnsafeBytes { weights.contents().copyMemory(from: $0.baseAddress!, byteCount: bank.count) }
        let xb = x.contents().bindMemory(to: UInt16.self, capacity: cols)
        for i in 0..<cols { xb[i] = Float16(xs[i]).bitPattern }
        guard let cb = Self.hal.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { XCTFail("no encoder"); return }
        let pso = try Self.hal.pipeline("qwg_gemv_q4g_v4")
        enc.setComputePipelineState(pso)
        enc.setBuffer(weights, offset: 0, index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 3)
        var n32 = UInt32(rows), k32 = UInt32(cols)
        enc.setBytes(&n32, length: 4, index: 4)
        enc.setBytes(&k32, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "the dispatch failed: \(String(describing: cb.error))")
        let got = y.contents().bindMemory(to: Float.self, capacity: rows)
        for r in 0..<4 {
            var value: Float = 0
            var magnitude: Float = 0
            for g in 0..<groups {
                let off = (r * groups + g) * 20
                let sc = Float(Float16(bitPattern: UInt16(bank[off]) | (UInt16(bank[off + 1]) << 8)))
                let bi = Float(Float16(bitPattern: UInt16(bank[off + 2]) | (UInt16(bank[off + 3]) << 8)))
                for w in 0..<4 {
                    let word = UInt32(bank[off + 4 + w * 4])
                        | (UInt32(bank[off + 4 + w * 4 + 1]) << 8)
                        | (UInt32(bank[off + 4 + w * 4 + 2]) << 16)
                        | (UInt32(bank[off + 4 + w * 4 + 3]) << 24)
                    for j in 0..<8 {
                        let nib = (word >> UInt32(QWAQProducer.hq4NibbleBitOffset(j))) & 0xF
                        let v = Float(Float16((1024.0 + Float(nib)) * sc + bi))
                        let product = v * xs[w * 8 + j + g * 32]
                        value += product
                        magnitude += abs(product)
                    }
                }
            }
            let scale = max(abs(Double(value)), Double(magnitude), 1e-6)
            let error = abs(Double(got[r]) - Double(value)) / scale
            XCTAssertLessThanOrEqual(error, 1e-3,
                                     "row \(r): got \(got[r]) reference \(value) magnitude \(magnitude)")
        }
    }
}
