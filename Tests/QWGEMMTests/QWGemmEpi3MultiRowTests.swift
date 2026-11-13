// QWGemmEpi3MultiRowTests.swift — Multi-row (M=2..5) epi=3 parity against row-by-row M=1.
import Foundation
import Metal
import XCTest
@testable import QWGEMM
import QWAlloc
import QWCore
import QWHAL
import QWQuant

final class QWGemmEpi3MultiRowTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private func makeHQ4Bank(n: Int, k: Int) -> Data {
        let gBytes = 20
        var d = Data(capacity: n * (k / 32) * gBytes)
        var seed: UInt32 = 0x1357_2468
        for r in 0..<n {
            for g in 0..<(k / 32) {
                let sc = Float16(0.02 * (1.0 + 0.1 * Float((r + g) % 7))).bitPattern
                d.append(contentsOf: [UInt8(sc & 0xFF), UInt8(sc >> 8), 0, 0])
                for _ in 0..<4 {
                    seed = seed &* 1664525 &+ 1013904223
                    let w = (seed >> 16) | ((seed & 0xFFFF) << 16)
                    d.append(contentsOf: [
                        UInt8(w & 0xFF), UInt8((w >> 8) & 0xFF),
                        UInt8((w >> 16) & 0xFF), UInt8((w >> 24) & 0xFF),
                    ])
                }
            }
        }
        return d
    }

    private func makeAQBank(bits: Int, group: Int, n: Int, k: Int) -> Data {
        var bank = Data()
        for r in 0..<n {
            for g in 0..<(k / group) {
                let values = (0..<group).map { i in UInt32((i * 3 + r * 7 + g * 13) % (1 << bits)) }
                let block = try! QWAQBlock.pack(bits: bits, group: group, values: values, scale: 0.03125, bias: -0.5)
                bank.append(block)
            }
        }
        return bank
    }

    private func makeSHVQBank(bits: Int, group: Int, n: Int, k: Int) -> Data {
        let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
        var seed: UInt32 = 0x9E37_79B9 ^ UInt32(bits * 131 + group)
        var bank = Data(count: n * (k / group) * gBytes)
        bank.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<(k / group) {
                    let base = (r * (k / group) + g) * gBytes
                    let sc = Float16(0.0625 * (1.0 + 0.125 * Float((r + g) % 5))).bitPattern
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

    private func verifyEpi3Parity(label: String, bank: Data, n: Int, k: Int, bits: Int,
                                  group: Int, isRotated: Bool, isSHVQ: Bool) throws {
        let arena = try QWArena(hal: Self.hal, weights: bank.count + (2 << 20),
                                activations: (8 << 20), kv: 0, gdnStates: 0)
        let wBuf = try arena.weightsRegion().allocate(QWTensorShape([(bank.count + 1) / 2], .f16))
        arena.copyIn(wBuf, from: bank)
        let view = try XCTUnwrap(QWWeightView(blocks: wBuf, n: n, k: k, bits: bits,
                                              groupSize: group, isRotated: isRotated, isSHVQ: isSHVQ))
        let act = arena.activationsRegion()
        let maxRows = 5
        let xT = try act.allocate(QWTensorShape([maxRows * k], .f32))
        let yMulti = try act.allocate(QWTensorShape([maxRows * n], .f32))
        let yRef = try act.allocate(QWTensorShape([maxRows * n], .f32))
        let xs = (0..<(maxRows * k)).map { Float(($0 % 23) - 11) * 0.08 }
        arena.copyIn(xT, from: xs.withUnsafeBufferPointer { (buf: UnsafeBufferPointer<Float>) in Data(buffer: buf) })

        for rows in [2, 3, 4, 5] {
            guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
            try QWGemm.encodeHQ4F32V4(enc, hal: Self.hal, weight: view, x: xT, y: yMulti,
                                      rows: rows, epi: 3, scale: 2.0)
            for r in 0..<rows {
                try QWGemm.encodeHQ4F32V4(
                    enc, hal: Self.hal, weight: view,
                    x: xT.slice(byteOffset: r * k * 4, count: k, dtype: .f32),
                    y: yRef.slice(byteOffset: r * n * 4, count: n, dtype: .f32),
                    rows: 1, epi: 3, scale: 2.0
                )
            }
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error, "\(label) rows=\(rows): \(String(describing: cb.error))")

            let got = arena.read(yMulti, as: Float.self, count: rows * n)
            let ref = arena.read(yRef, as: Float.self, count: rows * n)
            var worst: Float = 0, peak: Float = 1e-6
            for i in 0..<(rows * n) {
                worst = max(worst, abs(got[i] - ref[i]))
                peak = max(peak, abs(ref[i]))
            }
            XCTAssertLessThan(worst / peak, 1e-4, "\(label) rows=\(rows): worst=\(worst) peak=\(peak)")
        }
    }

    func testEpi3MultiRowParityAcrossQuantFamilies() throws {
        let n = 128
        try verifyEpi3Parity(label: "HQ4_G32_K128", bank: makeHQ4Bank(n: n, k: 128),
                             n: n, k: 128, bits: 4, group: 32, isRotated: true, isSHVQ: false)
        try verifyEpi3Parity(label: "HQ4_G32_K512", bank: makeHQ4Bank(n: n, k: 512),
                             n: n, k: 512, bits: 4, group: 32, isRotated: true, isSHVQ: false)
        try verifyEpi3Parity(label: "AQ8_G32_K512", bank: makeAQBank(bits: 8, group: 32, n: n, k: 512),
                             n: n, k: 512, bits: 8, group: 32, isRotated: false, isSHVQ: false)
        try verifyEpi3Parity(label: "SHVQ4_G64_K512", bank: makeSHVQBank(bits: 4, group: 64, n: n, k: 512),
                             n: n, k: 512, bits: 4, group: 64, isRotated: true, isSHVQ: true)
    }
}
