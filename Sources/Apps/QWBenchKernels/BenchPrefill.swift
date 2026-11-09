// BenchPrefill.swift — the prefill GEMM, raced over the two containers.
//
// The decode suites race the family's GEMV against the incumbent's. The
// prefill is another kernel on another shape: M tokens, N rows, K columns.
// This suite races the family's prefill twin against the incumbent's over the
// SAME logical weights, for every pair the family supports. It is bench-only:
// the engine picks the prefill kernel by container, not from the winner map.
import Foundation
import Metal
import QWCore
import QWHAL
import QWQuant

/// The containers the prefill race reads, and the kernel each one dispatches.
enum PrefillCandidate {
    /// The family's own container, at one of its pairs.
    case family(bits: Int, group: Int)
    /// The incumbent's container.
    case incumbent

    /// The kernel, as the metallib spells it. The family's names mirror
    /// `QWGemmAQ.gemmKernelName`, which is what the engine resolves them by.
    var name: String {
        switch self {
        case .family(let bits, let group): return "qwg_gemm_aq\(bits)_g\(group)_h16_bm128"
        case .incumbent: return "qwg_gemm_q4g_h16_bm128"
        }
    }

    /// The tokens one tile holds: both kernels are raced at BM=128.
    static let bm = 128
}

/// The buffers one prefill measurement needs.
struct PrefillWork {
    let weights: any MTLBuffer
    let x: any MTLBuffer
    let y: any MTLBuffer
    /// The bank the weights came from: the host reference reads it back.
    let bank: Data
    /// The activations: `m` rows of `k` values, token after token.
    let activations: [Float]
    let m: Int
    let n: Int
    let k: Int
}

/// The prefill race: the incumbent's tiled GEMM against the family's, pair by pair.
enum PrefillSuites {
    /// The tokens one measurement prefills.
    static let tokens = 4096
    /// The rows the projection produces: the model's own width.
    static let rows = 4096
    /// The pairs the family supports.
    static var pairs: [(bits: Int, group: Int)] {
        QWAQBlock.supported.map { (bits: $0.bits, group: $0.group) }
    }

    /// One candidate's measurement.
    struct Point {
        let name: String
        let bits: Int
        let group: Int
        let milliseconds: Double
        /// The worst it disagreed with the host by.
        let error: Double
        /// The container it read: the incumbent's, or the family's.
        let isFamily: Bool
    }

    /// One pair's race: the two candidates and what they cost.
    struct Race {
        let bits: Int
        let group: Int
        let family: Point
        let incumbent: Point
    }

    /// Races the incumbent against every pair of the family.
    static func run(hal: QWHal, cols: Int, samples: Int, repeats: Int) throws -> [Race] {
        var races: [Race] = []
        // The incumbent's bank is the same for every pair: it is built once.
        let incumbent = try makeWork(hal: hal, candidate: .incumbent, cols: cols)
        let incumbentError = try verify(hal: hal, work: incumbent, candidate: .incumbent)
        for (bits, group) in pairs {
            let familyCandidate = PrefillCandidate.family(bits: bits, group: group)
            let family = try makeWork(hal: hal, candidate: familyCandidate, cols: cols)
            let familyError = try verify(hal: hal, work: family, candidate: familyCandidate)
            var familyTimes: [Double] = []
            var incumbentTimes: [Double] = []
            for _ in 0..<samples {
                // The two are raced ALTERNATELY, so a clock ramp between them
                // lands on both rather than on one.
                familyTimes.append(try time(hal: hal, work: family,
                                            candidate: familyCandidate, repeats: repeats))
                incumbentTimes.append(try time(hal: hal, work: incumbent,
                                               candidate: .incumbent, repeats: repeats))
            }
            races.append(Race(bits: bits, group: group,
                              family: Point(name: familyCandidate.name, bits: bits, group: group,
                                            milliseconds: BenchRunner.median(familyTimes),
                                            error: familyError, isFamily: true),
                              incumbent: Point(name: PrefillCandidate.incumbent.name, bits: 4,
                                               group: 32, milliseconds: BenchRunner.median(incumbentTimes),
                                               error: incumbentError, isFamily: false)))
        }
        return races
    }

    /// The buffers and the bank for one candidate.
    static func makeWork(hal: QWHal, candidate: PrefillCandidate, cols: Int) throws -> PrefillWork {
        let device = hal.device
        let bank: Data
        switch candidate {
        case .family(let bits, let group):
            bank = AQBanks.make(bits: bits, group: group, rows: rows, cols: cols)
        case .incumbent:
            bank = AQBanks.makeIncumbent4G32(rows: rows, cols: cols)
        }
        let weights = try BenchRunner.buffer(device, bytes: bank.count)
        bank.withUnsafeBytes { src in
            weights.contents().copyMemory(from: src.baseAddress!, byteCount: bank.count)
        }
        let activations = makeActivations(m: tokens, k: cols)
        let x = try BenchRunner.buffer(device, bytes: tokens * cols * 2)
        let xb = x.contents().bindMemory(to: UInt16.self, capacity: tokens * cols)
        for i in 0..<(tokens * cols) { xb[i] = Float16(activations[i]).bitPattern }
        let y = try BenchRunner.buffer(device, bytes: tokens * rows * 2)
        return PrefillWork(weights: weights, x: x, y: y, bank: bank,
                           activations: activations, m: tokens, n: rows, k: cols)
    }

    /// One dispatch of the candidate's kernel.
    static func encode(_ enc: any MTLComputeCommandEncoder,
                       pso: any MTLComputePipelineState, work: PrefillWork) {
        enc.setComputePipelineState(pso)
        enc.setBuffer(work.weights, offset: 0, index: 0)
        enc.setBuffer(work.x, offset: 0, index: 1)
        enc.setBuffer(work.y, offset: 0, index: 2)
        enc.setBuffer(work.y, offset: 0, index: 3)
        var m32 = UInt32(work.m), n32 = UInt32(work.n), k32 = UInt32(work.k)
        enc.setBytes(&m32, length: 4, index: 4)
        enc.setBytes(&n32, length: 4, index: 5)
        enc.setBytes(&k32, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: (work.n + 63) / 64,
                                         height: (work.m + PrefillCandidate.bm - 1)
                                            / PrefillCandidate.bm, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Checks the candidate's output against the host, at the timed shape.
    static func verify(hal: QWHal, work: PrefillWork,
                       candidate: PrefillCandidate) throws -> Double {
        guard let (cb, enc) = hal.begin() else { throw BenchError.noTiming(candidate.name) }
        let pso = try hal.pipeline(candidate.name)
        encode(enc, pso: pso, work: work)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        if let error = cb.error {
            throw BenchError.kernelFailed(candidate.name, String(describing: error))
        }
        let got = work.y.contents().bindMemory(to: UInt16.self, capacity: work.m * work.n)
        var worst = 0.0
        for token in [0, 1, work.m / 2, work.m - 1] {
            for row in [0, 1, work.n / 2, work.n - 1] {
                let x = Array(work.activations[(token * work.k)..<((token + 1) * work.k)])
                let reference = referenceRow(candidate: candidate, work: work, row: row, x: x)
                let value = Float(Float16(bitPattern: got[token * work.n + row]))
                worst = max(worst, BenchRunner.relativeError(got: value, reference: reference))
            }
        }
        guard worst <= 1e-3 else {
            throw BenchError.verificationFailed(candidate.name, worst)
        }
        return worst
    }

    /// Times one candidate's dispatches, on the bench's own protocol.
    static func time(hal: QWHal, work: PrefillWork, candidate: PrefillCandidate,
                     repeats: Int) throws -> Double {
        let pso = try hal.pipeline(candidate.name)
        return try BenchRunner.time(hal: hal, name: candidate.name,
                                     warm: { enc in encode(enc, pso: pso, work: work) },
                                     timed: { enc in encode(enc, pso: pso, work: work) },
                                     repeats: repeats)
    }

    /// The host's answer for one row of one token: the weights in the
    /// container's own arithmetic, rounded to f16 as the device rounds them.
    static func referenceRow(candidate: PrefillCandidate, work: PrefillWork,
                             row: Int, x: [Float]) -> BenchReference {
        switch candidate {
        case .family(let bits, let group):
            return familyReferenceRow(bits: bits, group: group, work: work, row: row, x: x)
        case .incumbent:
            return incumbentReferenceRow(work: work, row: row, x: x)
        }
    }

    /// The family's row: the group's own scale and bias, and the code's bits.
    private static func familyReferenceRow(bits: Int, group: Int, work: PrefillWork,
                                           row: Int, x: [Float]) -> BenchReference {
        let groups = work.k / group
        let groupBytes = QWAQBlock.bytes(bits: bits, group: group)
        var value: Float = 0
        var magnitude: Float = 0
        work.bank.withUnsafeBytes { raw in
            for g in 0..<groups {
                let off = (row * groups + g) * groupBytes
                let slice = UnsafeRawBufferPointer(start: raw.baseAddress! + off, count: groupBytes)
                let (scale, bias) = QWAQBlock.header(slice)
                let payload = UnsafeRawBufferPointer(rebasing: slice.dropFirst(4))
                for i in 0..<group {
                    let code = Float(QWAQBlock.readBits(payload, bits: bits, index: i))
                    let weight = Float(Float16(code * scale + bias))
                    let product = weight * x[g * group + i]
                    value += product
                    magnitude += abs(product)
                }
            }
        }
        return BenchReference(value: value, magnitude: magnitude)
    }

    /// The incumbent's row: the magic's form, `(1024 + code) * scale + bias`,
    /// with the folded bias the container's header holds.
    private static func incumbentReferenceRow(work: PrefillWork, row: Int,
                                              x: [Float]) -> BenchReference {
        let groups = work.k / 32
        var value: Float = 0
        var magnitude: Float = 0
        work.bank.withUnsafeBytes { raw in
            for g in 0..<groups {
                let off = row * groups * 20 + g * 20
                let slice = UnsafeRawBufferPointer(start: raw.baseAddress! + off, count: 20)
                let scale = Float(Float16(bitPattern: UInt16(slice[0]) | (UInt16(slice[1]) << 8)))
                let bias = Float(Float16(bitPattern: UInt16(slice[2]) | (UInt16(slice[3]) << 8)))
                for w in 0..<4 {
                    let word = slice.loadUnaligned(fromByteOffset: 4 + w * 4, as: UInt32.self)
                    for j in 0..<8 {
                        let nib = word >> UInt32(QWAQProducer.hq4NibbleBitOffset(j)) & 0xF
                        let weight = Float(Float16((1024.0 + Float(nib)) * scale + bias))
                        // The block's own slice: block g covers columns
                        // g * 32 ..< g * 32 + 32 of the activation vector.
                        let product = weight * x[g * 32 + w * 8 + j]
                        value += product
                        magnitude += abs(product)
                    }
                }
            }
        }
        return BenchReference(value: value, magnitude: magnitude)
    }

    /// The activations: `m` rows of `k` values, every row its own stream.
    static func makeActivations(m: Int, k: Int) -> [Float] {
        var out = [Float](repeating: 0, count: m * k)
        var seed: UInt64 = 0x5EED_C0DE_5EED_C0DE
        for i in 0..<(m * k) {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            let unit = Double((seed >> 33) % 2_000_001) / 1_000_000.0 - 1.0
            out[i] = Float(Float16(unit))
        }
        return out
    }
}
