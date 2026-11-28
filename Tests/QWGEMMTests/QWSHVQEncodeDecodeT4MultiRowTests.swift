// QWSHVQEncodeDecodeT4MultiRowTests.swift — Parity tests for SHVQ15 FP32 scale
// (`M=2..9`), `QWGemmAQ.encodeDecodeT4(rows:)`, and `encodeDecodeBF16` `M=9` (`5+4`).
import Foundation
import Metal
import XCTest
@testable import QWGEMM
import QWAlloc
import QWCore
import QWHAL
import QWQuant

final class QWSHVQEncodeDecodeT4MultiRowTests: XCTestCase {
    private func makeSHVQBank(bits: Int, group: Int, n: Int, k: Int, scale: Float = 0.0625) -> Data {
        let groups = k / group
        let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
        var seed: UInt32 = 0x7A31_9B45 ^ UInt32(bits * 131 + group)
        var bank = Data(count: n * groups * gBytes)
        bank.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let base = (r * groups + g) * gBytes
                    let sc = Float16(scale * (1.0 + 0.17 * Float((r + g) % 7))).bitPattern
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

    private func unpackRows(bits: Int, group: Int, n: Int, k: Int, bank: Data) -> [[Float]] {
        let groups = k / group
        let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
        var rows = [[Float]]()
        rows.reserveCapacity(n)
        for r in 0..<n {
            var row = [Float]()
            row.reserveCapacity(k)
            var tmp = [Float]()
            for g in 0..<groups {
                let off = (r * groups + g) * gBytes
                let sub = bank.subdata(in: off..<(off + gBytes))
                sub.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                    QWSHVQBlock.unpack(group: raw, bits: bits, group: group, into: &tmp)
                }
                row.append(contentsOf: tmp)
            }
            rows.append(row)
        }
        return rows
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

    func testSHVQ15AndEncodeDecodeT4MultiRowParity() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.supportsSimdgroupMMA == true,
            "SHVQ Metal decode parity requires bare-metal Apple Silicon GPU"
        )
        let hal = try QWHal(geometry: .qwen38_27b)
        let n = 19, k = 256, maxM = 9
        let specs: [(bits: Int, group: Int)] = [(15, 32), (15, 64), (1, 64), (4, 32)]

        for (bits, group) in specs {
            let bank = makeSHVQBank(bits: bits, group: group, n: n, k: k)
            let wRows = unpackRows(bits: bits, group: group, n: n, k: k, bank: bank)
            let arena = try QWArena(hal: hal, weights: bank.count + (1 << 20),
                                    activations: (4 << 20), kv: 0, gdnStates: 0)
            let wBuf = try arena.weightsRegion().allocate(QWTensorShape([(bank.count + 1) / 2], .f16))
            arena.copyIn(wBuf, from: bank)
            let act = arena.activationsRegion()

            let xVals = makeActivations(count: maxM * k, seedInit: UInt32(0x4100 + bits * 17 + group))
            let xBuf = try act.allocate(QWTensorShape([maxM * k], .f16))
            arena.write(xBuf, from: xVals)
            let ySeq = try act.allocate(QWTensorShape([maxM * n], .f16))

            try step(hal) { enc in
                for r in 0..<maxM {
                    let xr = xBuf.slice(byteOffset: r * k * 2, count: k, dtype: .f16)
                    let yr = ySeq.slice(byteOffset: r * n * 2, count: n, dtype: .f16)
                    try QWGemmAQ.encodeDecodeT4(enc, hal: hal, blocks: wBuf, x: xr, y16: yr,
                                                n: n, k: k, bits: bits, group: group, rows: 1, isSHVQ: true)
                }
            }
            let seqBits = arena.read(ySeq, as: UInt16.self, count: maxM * n)

            for m in [2, 4, 5, 8, 9] {
                let xm = xBuf.slice(byteOffset: 0, count: m * k, dtype: .f16)
                let ym = try act.allocate(QWTensorShape([m * n], .f16))
                try step(hal) { enc in
                    try QWGemmAQ.encodeDecodeT4(enc, hal: hal, blocks: wBuf, x: xm, y16: ym,
                                                n: n, k: k, bits: bits, group: group, rows: m, isSHVQ: true)
                }
                let gotBits = arena.read(ym, as: UInt16.self, count: m * n)
                for row in 0..<m {
                    for col in 0..<n {
                        let idx = row * n + col
                        let got = Float(Float16(bitPattern: gotBits[idx]))
                        let ref1 = Float(Float16(bitPattern: seqBits[idx]))
                        var cpuAcc: Float = 0, mag: Float = 0
                        for j in 0..<k {
                            let p = wRows[col][j] * Float(xVals[row * k + j])
                            cpuAcc += p
                            mag += abs(p)
                        }
                        let tol = max(2e-3, mag * 2e-3)
                        XCTAssertEqual(got, ref1, accuracy: tol,
                                       "SHVQ\(bits)_G\(group) m=\(m) (\(row),\(col)) vs M=1: got=\(got) ref1=\(ref1)")
                        XCTAssertEqual(got, Float(Float16(cpuAcc)), accuracy: tol,
                                       "SHVQ\(bits)_G\(group) m=\(m) (\(row),\(col)) vs CPU: got=\(got) cpu=\(cpuAcc)")
                    }
                }
            }
        }
    }

    func testEncodeDecodeBF16M9FivePlusFourParity() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.supportsSimdgroupMMA == true,
            "SHVQ BF16 M=9 parity requires bare-metal Apple Silicon GPU"
        )
        let hal = try QWHal(geometry: .qwen38_27b)
        let n = 32, k = 256, rows = 9

        for (bits, group) in [(15, 64), (4, 32)] {
            let bank = makeSHVQBank(bits: bits, group: group, n: n, k: k)
            let arena = try QWArena(hal: hal, weights: bank.count + (1 << 20),
                                    activations: (4 << 20), kv: 0, gdnStates: 0)
            let wBuf = try arena.weightsRegion().allocate(QWTensorShape([(bank.count + 1) / 2], .f16))
            arena.copyIn(wBuf, from: bank)
            let act = arena.activationsRegion()
            let xVals = makeActivations(count: rows * k, seedInit: UInt32(0x5200 + bits))
            let xBuf = try act.allocate(QWTensorShape([rows * k], .f16))
            arena.write(xBuf, from: xVals)
            let ySeq = try act.allocate(QWTensorShape([rows * n], .bf16))
            let yM9 = try act.allocate(QWTensorShape([rows * n], .bf16))

            try step(hal) { enc in
                for r in 0..<rows {
                    let xr = xBuf.slice(byteOffset: r * k * 2, count: k, dtype: .f16)
                    let yr = ySeq.slice(byteOffset: r * n * 2, count: n, dtype: .bf16)
                    try QWGemmAQ.encodeDecodeBF16(enc, hal: hal, blocks: wBuf, x: xr, yBF16: yr,
                                                  n: n, k: k, bits: bits, group: group, rows: 1, isSHVQ: true)
                }
                try QWGemmAQ.encodeDecodeBF16(enc, hal: hal, blocks: wBuf, x: xBuf, yBF16: yM9,
                                              n: n, k: k, bits: bits, group: group, rows: rows, isSHVQ: true)
            }
            let seqBits = arena.read(ySeq, as: UInt16.self, count: rows * n)
            let m9Bits = arena.read(yM9, as: UInt16.self, count: rows * n)
            func bf16Ordered(_ u: UInt16) -> Int32 {
                (u & 0x7FFF) == 0 ? 0 : ((u & 0x8000) != 0 ? -Int32(u & 0x7FFF) : Int32(u))
            }
            for idx in 0..<(rows * n) {
                let ulp = abs(bf16Ordered(m9Bits[idx]) - bf16Ordered(seqBits[idx]))
                XCTAssertLessThanOrEqual(ulp, 1, "SHVQ\(bits)_G\(group) BF16 M=9 ULP=\(ulp) at \(idx)")
            }
        }
    }
}
