import XCTest
import Foundation
import Metal
@testable import QWGEMM
import QWCore
import QWHAL
import QWAlloc
import QWQuant

/// The decode family's benchmark, measured the way the format's spec says to
/// measure it. The protocol and what the benchmark may claim are in
/// specs/04-qwquant.md, "Benchmark protocol".
final class QWGemmAQBenchTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    /// The spec's flush size: 256 MB, written by the GPU, so the bank's lines
    /// are gone from the cache before the timed dispatch reads them.
    private static let flushBytes = 256 << 20
    private var flushBuffer: (any MTLBuffer)?

    private func flushCache() throws {
        let buffer: any MTLBuffer
        if let existing = flushBuffer {
            buffer = existing
        } else {
            guard let made = Self.hal.device.makeBuffer(length: Self.flushBytes,
                                                        options: .storageModeShared) else {
                throw XCTSkip("no flush buffer")
            }
            buffer = made
            flushBuffer = buffer
        }
        guard let cb = Self.hal.queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { throw XCTSkip("no command buffer") }
        blit.__fill(buffer, range: NSRange(location: 0, length: Self.flushBytes), value: 0xA5)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
    }

    /// One dispatch on a cold cache, timed on the GPU.
    private func timedOnce(_ encode: (any MTLComputeCommandEncoder) throws -> Void) throws -> Double {
        try flushCache()
        guard let cb = Self.hal.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("no command buffer") }
        try encode(enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        // The GPU timestamps are in SECONDS on this platform — the repo's own
        // bench relies on that, scaling them by 1e6 to reach microseconds.
        return Double(cb.gpuEndTime - cb.gpuStartTime) * 1000.0   // ms
    }

    private func makeActivations(k: Int) -> [Float] {
        var seed: UInt32 = 0x5EED_C0DE
        return (0..<k).map { _ in
            seed = seed &* 1664525 &+ 1013904223
            return Float(Int32(truncatingIfNeeded: seed >> 8) % 512) / 512.0 - 0.5
        }
    }

    /// One group of a real quantisation: the packer's own scale — 0.3 over 7.5,
    /// one fp16 cannot hold exactly — with pseudo-random codes. The header is
    /// rounded on purpose: this bank probes the format as it is.
    private func groupBytes(bits: Int, group: Int) -> [UInt8] {
        let scale = Float(0.3 / 7.5)
        let bias = -0.5 * scale
        var seed: UInt32 = 0xA5A5_1234
        let values: [UInt32] = (0..<group).map { _ in
            seed = seed &* 1664525 &+ 1013904223
            return UInt32((seed >> 8) % UInt32(1 << bits))
        }
        return Array(QWAQBlock.packRoundedHeader(bits: bits, group: group, values: values,
                                                 scale: scale, bias: bias))
    }

    /// A bank of real blocks: one group's bytes, replicated across the rows.
    /// The bytes are real quantised weights rather than a pattern — so the
    /// arithmetic is the arithmetic the engine will run.
    private func bankBytes(bits: Int, group: Int, rows: Int, k: Int) -> Data {
        let one = groupBytes(bits: bits, group: group)
        let groups = rows * (k / group)
        var out = Data(count: one.count * groups)
        out.withUnsafeMutableBytes { dst in
            guard let dstBase = dst.baseAddress else { return }
            one.withUnsafeBytes { src in
                guard let srcBase = src.baseAddress else { return }
                for g in 0..<groups {
                    dstBase.advanced(by: g * one.count)
                        .copyMemory(from: srcBase, byteCount: one.count)
                }
            }
        }
        return out
    }

    /// The incumbent's own bank: `Block_HQ4_G32` rows, written by the
    /// incumbent's packer. It is the control — the same problem, in the
    /// incumbent's container, read by the incumbent's kernel.
    private func incumbentBankBytes(rows: Int, k: Int) -> Data {
        var seed: UInt32 = 0x51C0_9E77
        let weights: [Float] = (0..<32).map { _ in
            seed = seed &* 1664525 &+ 1013904223
            return Float(Int32(truncatingIfNeeded: seed >> 8) % 512) / 512.0 - 0.5
        }
        let one = QWAQProducer.hq4G32Block(weights)
        let blocks = rows * (k / 32)
        var out = Data(count: one.count * blocks)
        out.withUnsafeMutableBytes { dst in
            guard let dstBase = dst.baseAddress else { return }
            one.withUnsafeBytes { src in
                guard let srcBase = src.baseAddress else { return }
                for b in 0..<blocks {
                    dstBase.advanced(by: b * one.count)
                        .copyMemory(from: srcBase, byteCount: one.count)
                }
            }
        }
        return out
    }

    func testEveryWidthAgainstTheIncumbent() throws {
        let n = 4096, k = 4096
        let rounds = 5

        // The incumbent's bank: 5.00 bpw, the control every pair is measured
        // against. The family's own 4-bit G32 pair reads the SAME number of
        // bytes, so that pair is a like-for-like kernel comparison.
        let hq4Bytes = n * (k / 32) * QWQuant.blockBytes
        var banks: [String: QWTensorValue] = [:]
        var bankBytesRead: [String: Int] = [:]
        var sources: [String: Data] = [:]

        let pairs = QWAQBlock.supported
        var totalBytes = hq4Bytes
        for (bits, group) in pairs {
            let name = QWAQBlock.formatName(bits: bits, group: group)
            let rowBytes = QWGemmAQ.rowBytes(bits: bits, group: group, k: k)!
            let bytes = n * rowBytes
            bankBytesRead[name] = bytes
            sources[name] = bankBytes(bits: bits, group: group, rows: n, k: k)
            totalBytes += bytes
        }

        let arena = try QWArena(hal: Self.hal, weights: (totalBytes + (16 << 20)),
                                activations: (16 << 20), kv: 0, gdnStates: 0)
        let incumbentTensor = try arena.weightsRegion().allocate(QWTensorShape([hq4Bytes / 2], .f16))
        arena.copyIn(incumbentTensor, from: incumbentBankBytes(rows: n, k: k))
        for (bits, group) in pairs {
            let name = QWAQBlock.formatName(bits: bits, group: group)
            let bytes = bankBytesRead[name]!
            let tensor = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
            arena.copyIn(tensor, from: sources[name]!)
            banks[name] = tensor
        }

        let act = arena.activationsRegion()
        let x16 = try act.allocate(QWTensorShape([k], .f16))
        arena.write(x16, from: makeActivations(k: k).map { Float16($0) })
        let y16 = try act.allocate(QWTensorShape([n], .f16))

        func incumbentDispatch(_ enc: any MTLComputeCommandEncoder) throws {
            let pso = try Self.hal.pipeline("qwg_gemv_q4g_dx_h16_t4")
            enc.setComputePipelineState(pso)
            incumbentTensor.bind(enc, index: 0)
            x16.bind(enc, index: 1)
            y16.bind(enc, index: 3)
            var n32 = UInt32(n) | 0x4000_0000
            var k32 = UInt32(k)
            enc.setBytes(&n32, length: 4, index: 4)
            enc.setBytes(&k32, length: 4, index: 5)
            y16.bind(enc, index: 8)
            let groups = (n + 15) / 16
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }

        // Warm every pipeline before the first sample: the first dispatch of
        // a kernel pays for its pipeline's setup, and a cold pipeline is not
        // what the engine measures.
        _ = try timedOnce(incumbentDispatch)
        for (bits, group) in pairs {
            let name = QWAQBlock.formatName(bits: bits, group: group)
            _ = try timedOnce { enc in
                try QWGemmAQ.encodeDecodeT4(enc, hal: Self.hal, blocks: banks[name]!, x: x16,
                                            y16: y16, n: n, k: k, bits: bits, group: group)
            }
        }

        // Alternate the candidates: a clock ramp during the run must land on
        // all of them, not on whichever ran last.
        var samples: [String: [Double]] = [:]
        for _ in 0..<rounds {
            samples["incumbent", default: []].append(try timedOnce(incumbentDispatch))
            for (bits, group) in pairs {
                let name = QWAQBlock.formatName(bits: bits, group: group)
                samples[name, default: []].append(try timedOnce { enc in
                    try QWGemmAQ.encodeDecodeT4(enc, hal: Self.hal, blocks: banks[name]!,
                                                x: x16, y16: y16, n: n, k: k,
                                                bits: bits, group: group)
                })
            }
        }

        func median(_ values: [Double]) -> Double {
            let sorted = values.sorted()
            return sorted[sorted.count / 2]
        }
        let incumbent = median(samples["incumbent"]!)
        XCTAssertGreaterThan(incumbent, 0)

        var report: [String] = []
        for (bits, group) in pairs {
            let name = QWAQBlock.formatName(bits: bits, group: group)
            let bytes = bankBytesRead[name]!
            let t = median(samples[name]!)
            let bpw = QWAQBlock.bitsPerWeight(bits: bits, group: group)
            report.append("""
            \(name) (\(String(format: "%.2f", bpw)) bpw, \(bytes / 1024) KiB): \
            \(String(format: "%.4f", t)) ms \
            (\(String(format: "%.0f", Double(bytes) / (t / 1000.0) / 1e9)) GB/s) \
            vs incumbent \(String(format: "%.4f", incumbent)) ms \
            — \(String(format: "%.2f", incumbent / t))x
            """)
        }
        print("""
        decode GEMV \(n)x\(k), cold cache (256 MB flush), median of \(rounds): \
        incumbent \(String(format: "%.4f", incumbent)) ms \
        (\(String(format: "%.0f", Double(hq4Bytes) / (incumbent / 1000.0) / 1e9)) GB/s over \
        \(hq4Bytes / 1024) KiB, 5.00 bpw)

        \(report.joined(separator: "\n"))

        """)

        // The deterministic claim: the family's sub-4-bit pairs read strictly
        // fewer bytes than the incumbent's inflated bank. That is the whole
        // point of the family, and it does not depend on the clock.
        for (bits, group) in pairs where bits < 4 {
            let name = QWAQBlock.formatName(bits: bits, group: group)
            XCTAssertLessThan(bankBytesRead[name]!, hq4Bytes,
                              "\(name) must read fewer bytes than the incumbent")
        }

        // The family's 4-bit G32 pair reads the SAME number of bytes as the
        // incumbent's bank, so its kernel must not be structurally slower.
        let g32Median = median(samples[QWAQBlock.formatName(bits: 4, group: 32)]!)
        XCTAssertLessThanOrEqual(g32Median, incumbent * 1.5,
                                 "the family's 4-bit G32 kernel must not be structurally slower "
                                 + "than the incumbent's")
    }

    func testEveryHQWidthF32V4AgainstTheIncumbent() throws {
        let n = 4096, k = 4096
        let rounds = 5
        let hq4Bytes = n * (k / 32) * QWQuant.blockBytes
        let pairs = QWAQBlock.hqSupported

        var banks: [String: QWWeightView] = [:]
        var bankBytesRead: [String: Int] = [:]
        var sources: [String: Data] = [:]
        var totalBytes = hq4Bytes
        for (bits, group) in pairs {
            let name = "HQ\(bits)_G\(group)"
            let rowBytes = QWGemmAQ.rowBytes(bits: bits, group: group, k: k)!
            let bytes = n * rowBytes
            bankBytesRead[name] = bytes
            sources[name] = bankBytes(bits: bits, group: group, rows: n, k: k)
            totalBytes += bytes
        }

        let arena = try QWArena(hal: Self.hal, weights: (totalBytes + (16 << 20)),
                                activations: (16 << 20), kv: 0, gdnStates: 0)
        let incumbentTensor = try arena.weightsRegion().allocate(QWTensorShape([hq4Bytes / 2], .f16))
        arena.copyIn(incumbentTensor, from: incumbentBankBytes(rows: n, k: k))
        let incumbentView = QWWeightView(blocks: incumbentTensor, n: n, k: k,
                                         bits: 4, groupSize: 32, isRotated: true, isIncumbentContainer: true)!

        for (bits, group) in pairs {
            let name = "HQ\(bits)_G\(group)"
            let bytes = bankBytesRead[name]!
            let tensor = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
            arena.copyIn(tensor, from: sources[name]!)
            banks[name] = QWWeightView(blocks: tensor, n: n, k: k,
                                       bits: bits, groupSize: group, isRotated: true, isIncumbentContainer: false)!
        }

        let act = arena.activationsRegion()
        let x32 = try act.allocate(QWTensorShape([4 * k], .f32))
        arena.write(x32, from: makeActivations(k: 4 * k))
        let y32 = try act.allocate(QWTensorShape([4 * n], .f32))

        func median(_ values: [Double]) -> Double {
            let sorted = values.sorted()
            return sorted[sorted.count / 2]
        }

        for rows in [1, 2, 4] {
            _ = try timedOnce { enc in
                try QWGemm.encodeHQ4F32V4(enc, hal: Self.hal, weight: incumbentView, x: x32, y: y32, rows: rows)
            }
            for (bits, group) in pairs {
                let name = "HQ\(bits)_G\(group)"
                _ = try timedOnce { enc in
                    try QWGemm.encodeHQ4F32V4(enc, hal: Self.hal, weight: banks[name]!, x: x32, y: y32, rows: rows)
                }
            }

            var samples: [String: [Double]] = [:]
            for _ in 0..<rounds {
                samples["incumbent", default: []].append(try timedOnce { enc in
                    try QWGemm.encodeHQ4F32V4(enc, hal: Self.hal, weight: incumbentView, x: x32, y: y32, rows: rows)
                })
                for (bits, group) in pairs {
                    let name = "HQ\(bits)_G\(group)"
                    samples[name, default: []].append(try timedOnce { enc in
                        try QWGemm.encodeHQ4F32V4(enc, hal: Self.hal, weight: banks[name]!, x: x32, y: y32, rows: rows)
                    })
                }
            }

            let incumbent = median(samples["incumbent"]!)
            var report: [String] = []
            for (bits, group) in pairs {
                let name = "HQ\(bits)_G\(group)"
                let bytes = bankBytesRead[name]!
                let t = median(samples[name]!)
                let bpw = QWAQBlock.bitsPerWeight(bits: bits, group: group)
                report.append("""
                \(name) (\(String(format: "%.2f", bpw)) bpw, \(bytes / 1024) KiB): \
                \(String(format: "%.4f", t)) ms \
                (\(String(format: "%.0f", Double(bytes) / (t / 1000.0) / 1e9)) GB/s) \
                vs incumbent \(String(format: "%.4f", incumbent)) ms \
                — \(String(format: "%.2f", incumbent / t))x
                """)
            }
            print("""
            f32_v4 GEMV rows=\(rows) \(n)x\(k), cold cache (256 MB flush), median of \(rounds): \
            incumbent \(String(format: "%.4f", incumbent)) ms \
            (\(String(format: "%.0f", Double(hq4Bytes) / (incumbent / 1000.0) / 1e9)) GB/s over \
            \(hq4Bytes / 1024) KiB, 5.00 bpw)

            \(report.joined(separator: "\n"))

            """)
        }
    }
}
